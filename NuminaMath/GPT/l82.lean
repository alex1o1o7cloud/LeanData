import Mathlib

namespace sum_of_smallest_and_largest_prime_between_1_and_50_l82_82393

theorem sum_of_smallest_and_largest_prime_between_1_and_50 :
  let primes := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47] in
  primes.minimum = some 2 ∧ primes.maximum = some 47 →
  2 + 47 = 49 := by
  intros h
  sorry

end sum_of_smallest_and_largest_prime_between_1_and_50_l82_82393


namespace no_polynomial_satisfies_condition_l82_82129

theorem no_polynomial_satisfies_condition :
  ¬ (∃ (f : Polynomial ℝ), 1 ≤ f.degree ∧ ∀ x : ℝ, f(x^2) = (f(x))^2 + f(f(x))) :=
by sorry

end no_polynomial_satisfies_condition_l82_82129


namespace find_u_plus_v_l82_82643

theorem find_u_plus_v (u v : ℤ) (huv : 0 < v ∧ v < u) (h_area : u * u + 3 * u * v = 451) : u + v = 21 := 
sorry

end find_u_plus_v_l82_82643


namespace range_of_g_l82_82858

open Real

noncomputable def g (x : ℝ) : ℝ := (arccos x) ^ 2 + (arcsin x) ^ 2

theorem range_of_g : 
  set.range (λ x, g x) = set.Icc ((π ^ 2) / 4) ((π ^ 2) / 2) :=
sorry

end range_of_g_l82_82858


namespace minimum_segments_for_cube_l82_82461

theorem minimum_segments_for_cube (side_length : ℝ) (segment_length : ℝ) :
  side_length = 2 →
  segment_length = 3 →
  ∃ (n : ℕ), n = 6 ∧
  (∀ (p1 p2 : ℝ^3), p1 ≠ p2 ∧ 
  p1 ∈ cube_vertices side_length ∧ 
  p2 ∈ cube_vertices side_length ∧ 
  opposite_vertices p1 p2 → 
  minimum_segments_to_connect p1 p2 segment_length = n) :=
by
  sorry

end minimum_segments_for_cube_l82_82461


namespace divide_into_groups_l82_82302

theorem divide_into_groups (m n k : ℕ) (h₁ : 0 < m) (h₂ : 0 < n) (h₃ : 0 < k) (h₄ : m ≥ n) (h₅ : (n * (n + 1) / 2) = m * k) :
  ∃ (groups : fin k → fin m → ℕ), (∀ i : fin k, list.sum (list.map (λ j, groups i j) (list.range m)) = m) ∧ 
                                  (list.perm (list.join (list.map (λ i, list.filter_map some (list.of_fn (groups i))) (list.range k))) (list.range (n + 1))) := 
sorry

end divide_into_groups_l82_82302


namespace all_hexahedrons_have_circumscribed_sphere_l82_82749

open EuclideanGeometry

/-- A statement defining that a set of three planes intersect a parallelepiped, cutting it into 8 hexahedrons. -/
def three_planes_cut_parallelepiped (planes : ℝ → ℝ → ℝ → Prop) (parallelepiped : Set (ℝ × ℝ × ℝ)) (hexahedrons : List (Set (ℝ × ℝ × ℝ))) : Prop :=
  ∃ (p1 p2 p3 : ℝ → ℝ → ℝ → Prop),
    (∀ p, p ∈ planes ↔ (p = p1 ∨ p = p2 ∨ p = p3)) ∧
    (∀ h, h ∈ hexahedrons →
      (∃ quadrilaterals : List (Set (ℝ × ℝ × ℝ)),
        (∀ q, q ∈ quadrilaterals → is_quadrilateral q) ∧
        (∀ p t1 t2 t3 t4, p ∈ quadrilaterals → 
         (t1 ∈ p → t2 ∈ p → t3 ∈ p → t4 ∈ p → 
          ∃ c : ℝ × ℝ × ℝ, 
          c ∈ Spheres.center [t1, t2, t3, t4]))))

/-- Definition of an inscribable quadrilateral using basic Euclidean Geometry properties -/
def is_inscribable_quadrilateral (q : Set (ℝ × ℝ × ℝ)) : Prop :=
  ∃ a b c d : ℝ × ℝ × ℝ, a ∈ q ∧ b ∈ q ∧ c ∈ q ∧ d ∈ q ∧
  ∃ O : ℝ × ℝ × ℝ, dist O a = dist O b ∧ dist O b = dist O c ∧ dist O c = dist O d

/-- The main theorem stating if one hexahedron of the given intersected parallelepiped can have a circumscribed sphere,
  then all hexahedrons can have a circumscribed sphere. -/
theorem all_hexahedrons_have_circumscribed_sphere
  (planes : ℝ → ℝ → ℝ → Prop)
  (parallelepiped : Set (ℝ × ℝ × ℝ))
  (hexahedrons : List (Set (ℝ × ℝ × ℝ)))
  (hx : three_planes_cut_parallelepiped planes parallelepiped hexahedrons)
  (∃ h : Set (ℝ × ℝ × ℝ), h ∈ hexahedrons ∧ is_circumscribable_sphere h) :
  ∀ h ∈ hexahedrons, is_circumscribable_sphere h := 
sorry

end all_hexahedrons_have_circumscribed_sphere_l82_82749


namespace hyperbola_asymptotes_l82_82163

theorem hyperbola_asymptotes
  (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (C : ∀ x y : ℝ, (x^2) / (a^2) - (y^2) / (b^2) = 1)
  (D : (0, ± b) is_valid_vertex)
  (A B : (2a, ± sqrt(3) * b) are_valid_intersections)
  (centroid_ABD_on_circle : ∀ (G : ∀ x y : ℝ, (x - 0)^2 + (y - 0)^2 = (a^2 + b^2)))
  : asymptotes C = λ (y : ℝ) (x : ℝ), y = ± (sqrt(14) / 4) * x :=
by sorry

end hyperbola_asymptotes_l82_82163


namespace problem_prove_a5_b5_c5_l82_82216

theorem problem_prove_a5_b5_c5 :
  ∀ (a b c : ℝ), 
    a + b + c = 1 ∧ 
    a^2 + b^2 + c^2 = 3 ∧ 
    a^3 + b^3 + c^3 = 4 → 
    a^5 + b^5 + c^5 = 11 / 3 :=
by
  intros a b c h
  cases h with h1 h2
  cases h2 with h2 h3
  sorry

end problem_prove_a5_b5_c5_l82_82216


namespace can_end_up_with_17_l82_82319

theorem can_end_up_with_17 : 
  ∃ (sequence : list ℕ → list ℕ),
    (sequence [1, 2, 4, 8, 16, 32] = [17]) ∧
    (∀ l, sequence l = [l.head] → true) := 
sorry

end can_end_up_with_17_l82_82319


namespace triangle_area_l82_82620

theorem triangle_area {a b : ℝ} (h : a ≠ 0) :
  (∃ x y : ℝ, 3 * x + a * y = 12) → b = 24 / a ↔ (∃ x y : ℝ, x = 4 ∧ y = 12 / a ∧ b = (1/2) * 4 * (12 / a)) :=
by
  sorry

end triangle_area_l82_82620


namespace find_angle_C_l82_82552

-- Definitions and conditions
variables (A B C D E F : Type) [angleABC: Angle A B C]
variables [angleDEF: Angle D E F]
variables (angleA : angleABC = 55) (angleE : angleDEF = 45)
variables (congruence : TriangleCongruent A B C D E F)

-- The statement to prove
theorem find_angle_C (h1 : congruence) (h2 : angleA) (h3 : angleE) : angleABC.angleC = 80 := 
by 
  sorry

end find_angle_C_l82_82552


namespace sum_of_possible_values_d_l82_82801

theorem sum_of_possible_values_d :
  (∃ (d : ℕ), d ∈ {2, 3}) → (2 + 3 = 5) :=
by
  sorry

end sum_of_possible_values_d_l82_82801


namespace equation_has_solution_iff_l82_82371

open Real

theorem equation_has_solution_iff (a : ℝ) : 
  (∃ x : ℝ, (1/3)^|x| + a - 1 = 0) ↔ (0 ≤ a ∧ a < 1) :=
by
  sorry

end equation_has_solution_iff_l82_82371


namespace islanders_liars_count_l82_82388

theorem islanders_liars_count (n : ℕ) (two_liars_truth : bool) (circular_arrangement : bool) 
(h1 : n = 2017)
(h2 : two_liars_truth = tt)
(h3 : circular_arrangement = tt) :
  ∃ liars : ℕ, liars = 1344 :=
by {
  sorry
}

end islanders_liars_count_l82_82388


namespace scheduling_valid_orders_l82_82048

theorem scheduling_valid_orders : 
  let n := 8 in
  let factorial := Nat.factorial n in
  let valid_orders := factorial / 2 / 2 in
  valid_orders = 10080 :=
by
  let n := 8
  let factorial := Nat.factorial n
  let valid_orders := factorial / 2 / 2
  show valid_orders = 10080 from sorry

end scheduling_valid_orders_l82_82048


namespace no_solution_xn_yn_zn_l82_82953

theorem no_solution_xn_yn_zn (x y z n : ℕ) (h : n ≥ z) : ¬ (x^n + y^n = z^n) :=
sorry

end no_solution_xn_yn_zn_l82_82953


namespace inscribed_circle_radius_l82_82089

theorem inscribed_circle_radius {DE DF EF : ℝ} (hDE: DE = 26) (hDF: DF = 16) (hEF: EF = 18):
  let s := (DE + DF + EF) / 2 in
  let K := Real.sqrt (s * (s - DE) * (s - DF) * (s - EF)) in
  ∃ r : ℝ, K = r * s ∧ r = 2 * Real.sqrt 14 :=
by
  sorry

end inscribed_circle_radius_l82_82089


namespace ellipse_equation_l82_82523

theorem ellipse_equation (a : ℝ) (x y : ℝ) (h : (x, y) = (-3, 2)) :
  (∃ a : ℝ, ∀ x y : ℝ, x^2 / 15 + y^2 / 10 = 1) ↔ (x, y) ∈ { p : ℝ × ℝ | p.1^2 / 15 + p.2^2 / 10 = 1 } :=
by
  have h1 : 15 = a^2 := by
    sorry
  have h2 : 10 = a^2 - 5 := by
    sorry
  sorry

end ellipse_equation_l82_82523


namespace bob_earns_more_than_alice_l82_82475

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r) ^ n

theorem bob_earns_more_than_alice :
  let P : ℝ := 40000
  let r_A : ℝ := 0.05
  let r_B : ℝ := 0.025
  let n_A : ℕ := 3
  let n_B : ℕ := 6
  let A_A := compound_interest P r_A n_A
  let A_B := compound_interest P r_B n_B
  in A_B - A_A = 66 :=
by
  sorry

end bob_earns_more_than_alice_l82_82475


namespace sine_of_angle_through_point_l82_82565

theorem sine_of_angle_through_point {a : ℝ} (h : a < 0) : 
  let P := (-4 * a, 3 * a) in
  sin (angle_of_point P) = -3 / 5 :=
by
  { sorry }

end sine_of_angle_through_point_l82_82565


namespace expected_value_of_game_l82_82816

theorem expected_value_of_game :
  let heads_prob := 1 / 4
  let tails_prob := 1 / 2
  let edge_prob := 1 / 4
  let gain_heads := 4
  let loss_tails := -3
  let gain_edge := 0
  let expected_value := heads_prob * gain_heads + tails_prob * loss_tails + edge_prob * gain_edge
  expected_value = -0.5 :=
by
  sorry

end expected_value_of_game_l82_82816


namespace collinear_E_F_G_l82_82909

open EuclideanGeometry

noncomputable def trapezoid (A B C D : Point) : Prop :=
  Collinear D A  ∧ Collinear B C 

noncomputable def circles (A B C D : Point) : Prop :=
  (circle_circumference A B C) ∧ (circle_circumference A C D) ∧ (circle_circumference A B D) ∧ (circle_circumference C B D)

theorem collinear_E_F_G
  {A B C D G E F : Point} 
  (h_trapezoid: AD ⟨A, D⟩ = BC ⟨B, C⟩)
  (h_intersect: IntersectSegment ⟨A, B⟩ ⟨D, C⟩ = G)
  (h_common_external_tangents_1: ExternalTangents (circle_circumference A B C) (circle_circumference A C D) = E)
  (h_common_external_tangents_2: ExternalTangents (circle_circumference A B D) (circle_circumference C B D) = F) :
  Collinear E F G :=
  sorry 

end collinear_E_F_G_l82_82909


namespace exactly_one_negative_x_or_y_l82_82527

theorem exactly_one_negative_x_or_y
  (x1 x2 x3 y1 y2 y3 : ℝ)
  (x1_ne_zero : x1 ≠ 0) (x2_ne_zero : x2 ≠ 0) (x3_ne_zero : x3 ≠ 0)
  (y1_ne_zero : y1 ≠ 0) (y2_ne_zero : y2 ≠ 0) (y3_ne_zero : y3 ≠ 0)
  (h1 : x1 * x2 * x3 = - y1 * y2 * y3)
  (h2 : x1^2 + x2^2 + x3^2 = y1^2 + y2^2 + y3^2)
  (h3 : x1 + y1 + x2 + y2 ≥ x3 + y3 ∧ x2 + y2 + x3 + y3 ≥ x1 + y1 ∧ x3 + y3 + x1 + y1 ≥ x2 + y2)
  (h4 : (x1 + y1)^2 + (x2 + y2)^2 ≥ (x3 + y3)^2 ∧ (x2 + y2)^2 + (x3 + y3)^2 ≥ (x1 + y1)^2 ∧ (x3 + y3)^2 + (x1 + y1)^2 ≥ (x2 + y2)^2) :
  ∃! (a : ℝ), (a = x1 ∨ a = x2 ∨ a = x3 ∨ a = y1 ∨ a = y2 ∨ a = y3) ∧ a < 0 :=
sorry

end exactly_one_negative_x_or_y_l82_82527


namespace balloons_left_after_distribution_l82_82772

theorem balloons_left_after_distribution :
  (22 + 40 + 70 + 90) % 10 = 2 := by
  sorry

end balloons_left_after_distribution_l82_82772


namespace m_plus_n_l82_82518

noncomputable def domain_of_function :
  {x : ℝ // 3 < x ∧ x < 27} :=
sorry

def length_of_interval {x : ℝ // 3 < x ∧ x < 27} : ℝ :=
by
  exact 24

def length_as_fraction {x : ℝ // 3 < x ∧ x < 27} : ℚ :=
by
  exact 24

def are_coprime : ℚ → ℚ → Prop
| a, b => a.numerator.gcd b.denominator = 1

def coprime_24_1 : are_coprime 24 1 := by exact rfl

theorem m_plus_n (h₁ : length_as_fraction = 24) (h₂ : coprime_24_1) : 24 + 1 = 25 :=
by
  exact congr_arg2 (.+.) rfl rfl

end m_plus_n_l82_82518


namespace values_of_x_l82_82645

theorem values_of_x (M N P : ℝ) (h1 : ∀ x : ℝ, (x + M) * (2 * x + 40) / ((x + P) * (x + 10)) = 3)
  (h2 : ∀ x : ℝ, (x = -10 ∨ x = -20) → ¬((x + M) * (2 * x + 40) / ((x + P) * (x + 10)) = 3)) :
  is_sum [-10, -20] (-30) :=
by
  sorry

def is_sum (l : List ℝ) (s : ℝ) : Prop := s = l.sum

end values_of_x_l82_82645


namespace monotonic_intervals_min_value_of_a_log_inequality_l82_82928

-- Define the function f(x) = x / log(x) - a * x
def f (x : ℝ) (a : ℝ) : ℝ := x / (Real.log x) - a * x

-- Problem 1: Monotonic intervals of f(x) when a = 0
theorem monotonic_intervals :
  (∀ x, 0 < x ∧ x < 1 → deriv (f x 0) x < 0) ∧
  (∀ x, 1 < x ∧ x < Real.exp 1 → deriv (f x 0) x < 0) ∧
  (∀ x, x > Real.exp 1 → deriv (f x 0) x > 0) :=
by sorry

-- Problem 2: Minimum value of a for decreasing f(x) on (2, +∞)
theorem min_value_of_a :
  ∀ a, (∀ x, x > 2 → deriv (f x a) x ≤ 0) → a ≥ 1/4 :=
by sorry

-- Problem 3: Logarithmic inequality for x > 0
theorem log_inequality (x : ℝ) (hx : x > 0) :
  Real.log x > 1 / Real.exp x - 3 / (4 * x^2) :=
by sorry

end monotonic_intervals_min_value_of_a_log_inequality_l82_82928


namespace coffee_machine_price_l82_82632

noncomputable def original_machine_price : ℝ :=
  let coffees_prior_cost_per_day := 2 * 4
  let new_coffees_cost_per_day := 3
  let daily_savings := coffees_prior_cost_per_day - new_coffees_cost_per_day
  let total_savings := 36 * daily_savings
  let discounted_price := total_savings
  let discount := 20
  discounted_price + discount

theorem coffee_machine_price
  (coffees_prior_cost_per_day : ℝ := 2 * 4)
  (new_coffees_cost_per_day : ℝ := 3)
  (daily_savings : ℝ := coffees_prior_cost_per_day - new_coffees_cost_per_day)
  (total_savings : ℝ := 36 * daily_savings)
  (discounted_price : ℝ := total_savings)
  (discount : ℝ := 20) :
  original_machine_price = 200 :=
by
  sorry

end coffee_machine_price_l82_82632


namespace union_sets_l82_82911

open Set

variable {α : Type*}

def A : Set α := {x | x > 1}

def B : Set α := {x | -1 < x ∧ x < 2}

theorem union_sets (x : α) : x ∈ A ∪ B ↔ x > -1 := by
  sorry

end union_sets_l82_82911


namespace volume_pyramid_TABC_l82_82335

-- Define T as the apex of the pyramid, and A, B, C as the base points in a 3D space.
variables (A B C T : ℝ^3)
-- Define the lengths of TA, TB, TC
variables (TA TB TC : ℝ)
-- Define the distances TA, TB, and TC are 10 units each
axiom dist_TA : TA = 10
axiom dist_TB : TB = 10
axiom dist_TC : TC = 10

-- Define the orthogonality conditions
axiom ortho_TA_TB : TA.perp TB
axiom ortho_TB_TC : TB.perp TC
axiom ortho_TC_TA : TC.perp TA

-- Prove the volume of the pyramid TABC is 500/3 cubic units
theorem volume_pyramid_TABC :
  volume_pyramid T A B C = 500 / 3 :=
sorry

end volume_pyramid_TABC_l82_82335


namespace find_omega_and_symmetry_axis_find_b_and_A_l82_82930

noncomputable def f (ω x : ℝ) : ℝ :=
  cos (ω * x) * sin (ω * x - π / 3) + sqrt 3 * cos (ω * x) ^ 2 - sqrt 3 / 4

axiom ω_pos : ω > 0
axiom distance_symmetry_center : ∃ x, f ω x - f ω (x + π / 2 / ω) = 0

theorem find_omega_and_symmetry_axis :
  ω = 1 ∧ 
  (∃ k : ℤ, ∀ x, f 1 x = f 1 (x + k * π / 2 + π / 12)) :=
sorry

variables {A B C : ℝ} {a b c : ℝ}
axiom angle_A_condition : f 1 A = sqrt 3 / 4
axiom angle_C_condition : sin C = 1 / 3
axiom side_a : a = sqrt 3

theorem find_b_and_A :
  A = π / 6 ∧ b = (3 + 2 * sqrt 6) / 3 :=
sorry

end find_omega_and_symmetry_axis_find_b_and_A_l82_82930


namespace find_y_l82_82130

theorem find_y (y : ℝ) : 
  2 ≤ y / (3 * y - 4) ∧ y / (3 * y - 4) < 5 ↔ y ∈ Set.Ioc (10 / 7) (8 / 5) := 
sorry

end find_y_l82_82130


namespace gcd_98_196_is_98_l82_82718

def gcd_98_196 : ℕ := Nat.gcd 98 196

theorem gcd_98_196_is_98 : gcd_98_196 = 98 := by
  have h : 196 = 98 * 2 := rfl
  have h_div : Nat.gcd 98 196 = 98 :=
    Nat.gcd_dvd _ _ 196 (by rw [h]; apply dvd_mul_left)
  exact h_div

end gcd_98_196_is_98_l82_82718


namespace range_of_x_l82_82550

theorem range_of_x (x : ℝ) (h1 : 0 ≤ x) (h2 : x < 2 * Real.pi) (h3 : Real.sqrt (1 - Real.sin (2 * x)) = Real.sin x - Real.cos x) :
    Real.pi / 4 ≤ x ∧ x ≤ 5 * Real.pi / 4 :=
by
  sorry

end range_of_x_l82_82550


namespace distance_from_circle_center_to_line_l82_82355

theorem distance_from_circle_center_to_line :
  let center := (1, 0)
  let line := (2, 1, -1)  -- representing 2x + y - 1 = 0
  let distance := (λ (point line : ℕ × ℤ × ℤ), (|line.1 * point.1 + line.2 * point.2 + line.3|) / Real.sqrt (line.1 ^ 2 + line.2 ^ 2))
  distance center line = Real.sqrt 5 / 5 := 
sorry

end distance_from_circle_center_to_line_l82_82355


namespace average_cost_is_thirteen_l82_82480

noncomputable def averageCostPerPen (pensCost shippingCost : ℝ) (totalPens : ℕ) : ℕ :=
  Nat.ceil ((pensCost + shippingCost) * 100 / totalPens)

theorem average_cost_is_thirteen :
  averageCostPerPen 29.85 8.10 300 = 13 :=
by
  sorry

end average_cost_is_thirteen_l82_82480


namespace number_of_full_boxes_l82_82747

theorem number_of_full_boxes (peaches_in_basket baskets_eaten_peaches box_capacity : ℕ) (h1 : peaches_in_basket = 23) (h2 : baskets = 7) (h3 : eaten_peaches = 7) (h4 : box_capacity = 13) :
  (peaches_in_basket * baskets - eaten_peaches) / box_capacity = 11 :=
by
  sorry

end number_of_full_boxes_l82_82747


namespace number_of_solutions_l82_82729

noncomputable def f (x : ℝ) : ℝ := 3 * x^4 - 4 * x^3 - 12 * x^2 + 12

theorem number_of_solutions : ∃ x1 x2, f x1 = 0 ∧ f x2 = 0 ∧ x1 ≠ x2 ∧
  ∀ x, f x = 0 → (x = x1 ∨ x = x2) :=
by
  sorry

end number_of_solutions_l82_82729


namespace area_of_trapezoid_MBCN_l82_82982

variables {AB BC MN : ℝ}
variables {Area_ABCD Area_MBCN : ℝ}
variables {Height : ℝ}

-- Given conditions
def cond1 : Area_ABCD = 40 := sorry
def cond2 : AB = 8 := sorry
def cond3 : BC = 5 := sorry
def cond4 : MN = 2 := sorry
def cond5 : Height = 5 := sorry

-- Define the theorem to be proven
theorem area_of_trapezoid_MBCN : 
  Area_ABCD = AB * BC → MN + BC = 6 → Height = 5 →
  Area_MBCN = (1/2) * (MN + BC) * Height → 
  Area_MBCN = 15 :=
by
  intros h1 h2 h3 h4
  sorry

end area_of_trapezoid_MBCN_l82_82982


namespace find_m_same_foci_l82_82918

theorem find_m_same_foci (m : ℝ) 
(hyperbola_eq : ∃ x y : ℝ, x^2 - y^2 = m) 
(ellipse_eq : ∃ x y : ℝ, 2 * x^2 + 3 * y^2 = m + 1) 
(same_foci : ∀ a b : ℝ, (x^2 - y^2 = m) ∧ (2 * x^2 + 3 * y^2 = m + 1) → 
               let c_ellipse := (m + 1) / 6
               let c_hyperbola := 2 * m
               c_ellipse = c_hyperbola ) : 
m = 1 / 11 := 
sorry

end find_m_same_foci_l82_82918


namespace union_of_A_and_B_l82_82308

open Set

theorem union_of_A_and_B : 
  let A := {x : ℝ | x + 2 > 0}
  let B := {y : ℝ | ∃ (x : ℝ), y = Real.cos x}
  A ∪ B = {z : ℝ | z > -2} := 
by
  intros
  sorry

end union_of_A_and_B_l82_82308


namespace min_distance_curve_line_l82_82788

noncomputable theory

def curve (x : ℝ) := x^2 - real.log x
def line (x y : ℝ) := x - y - 4

theorem min_distance_curve_line : 
  ∀ P : ℝ × ℝ, P.2 = curve P.1 → 
    ∃ D : ℝ, D = 2 * real.sqrt 2 := sorry

end min_distance_curve_line_l82_82788


namespace property_P_k_holds_for_all_k_l82_82177

noncomputable def majority {n : ℕ} (seqs : List (Fin n → Bool)) : Fin n → Bool := fun i =>
  if List.count (fun s => s i = true) seqs ≤ List.length seqs / 2 then false else true

def has_property_P_k (S : Set (Fin n → Bool)) (k : ℕ) : Prop :=
  ∀ seqs : List (Fin n → Bool), seqs.length = 2 * k + 1 → List.forall (λ s, s ∈ S) seqs → majority seqs ∈ S

theorem property_P_k_holds_for_all_k (n : ℕ) (S : Set (Fin n → Bool)) :
  (∀ k > 0, has_property_P_k S k) :=
sorry

end property_P_k_holds_for_all_k_l82_82177


namespace correct_equations_l82_82431

theorem correct_equations (x y : ℝ) :
  (9 * x - y = 4) → (y - 8 * x = 3) → (9 * x - y = 4 ∧ y - 8 * x = 3) :=
by
  intros h1 h2
  exact ⟨h1, h2⟩

end correct_equations_l82_82431


namespace farthest_points_hyperbola_l82_82005

noncomputable def farthest_points_locus (A B : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {P | (P.1 ^ 2 - P.2 ^ 2 = (dist A B / 2)^2)}

theorem farthest_points_hyperbola (A B : ℝ × ℝ) :
  farthest_points_locus A B = {P | (P.1 ^ 2 - P.2 ^ 2) = (dist A B / 2)^2} :=
  sorry

end farthest_points_hyperbola_l82_82005


namespace lines_planes_proposition_l82_82175

variables (m n : Line) (α β : Plane)

-- Definitions of the conditions
def parallel_lines := m ∥ n
def perpendicular_line_plane := n ⟂ β
def line_in_plane := m ⊂ α

-- Definition of the proof problem
theorem lines_planes_proposition (h1 : parallel_lines m n) (h2 : perpendicular_line_plane n β) (h3 : line_in_plane m α) : α ⟂ β :=
sorry

end lines_planes_proposition_l82_82175


namespace football_cost_l82_82773

theorem football_cost (cost_shorts cost_shoes money_have money_need : ℝ)
  (h_shorts : cost_shorts = 2.40)
  (h_shoes : cost_shoes = 11.85)
  (h_have : money_have = 10)
  (h_need : money_need = 8) :
  (money_have + money_need - (cost_shorts + cost_shoes) = 3.75) :=
by
  -- Proof goes here
  sorry

end football_cost_l82_82773


namespace number_of_colors_l82_82380

-- Definition of digit sum function
def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Theorem stating the number of different colors (digit sums) between 1 and 2021 is 28
theorem number_of_colors : (Finset.image digit_sum (Finset.range 2021).succ).card = 28 :=
by
  -- Proof goes here
  sorry

end number_of_colors_l82_82380


namespace difference_in_base_7_l82_82849

theorem difference_in_base_7 : 
  ∀ (a b : ℕ), a = 4512 ∧ b = 2345 → (a - b) % 7 = 2144 :=
by
  intros a b h
  cases h with ha hb
  sorry

end difference_in_base_7_l82_82849


namespace eval_f_comp_f_l82_82542

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then (-x) ^ (1 / 2) else Real.log x / Real.log 2

theorem eval_f_comp_f :
  f (f (1 / 4)) = Real.sqrt 2 :=
by
  -- Proof will be provided here
  sorry

end eval_f_comp_f_l82_82542


namespace tangent_line_l82_82567

noncomputable def f (x : ℝ) : ℝ := Real.sqrt x
noncomputable def g (x : ℝ) : ℝ := 1 / x

theorem tangent_line (x y : ℝ) (h_inter : y = f x ∧ y = g x) :
  (x - 2 * y + 1 = 0) :=
by
  sorry

end tangent_line_l82_82567


namespace adam_change_l82_82064

-- Defining the given amount Adam has and the cost of the airplane.
def amountAdamHas : ℝ := 5.00
def costOfAirplane : ℝ := 4.28

-- Statement of the theorem to be proven.
theorem adam_change : amountAdamHas - costOfAirplane = 0.72 := by
  sorry

end adam_change_l82_82064


namespace area_swept_by_AP_l82_82162

noncomputable def pointA := (2 : ℝ, 0 : ℝ)

def pointP (t : ℝ) : ℝ × ℝ :=
  (Real.sin (2 * t - Real.pi / 3), Real.cos (2 * t - Real.pi / 3))

theorem area_swept_by_AP (t1 t2 : ℝ) (h1 : t1 = Real.pi / 12) (h2 : t2 = Real.pi / 4) :
  let θ := 2 * t2 - 2 * t1 in
  let r := 1 in
  let area := (1 / 2) * r^2 * θ in
  area = Real.pi / 6 :=
by 
  sorry

end area_swept_by_AP_l82_82162


namespace log2_x_value_l82_82558

noncomputable def log_base (a b : ℝ) : ℝ := Real.log b / Real.log a

theorem log2_x_value
  (x : ℝ)
  (h : log_base (5 * x) (2 * x) = log_base (625 * x) (8 * x)) :
  log_base 2 x = Real.log 5 / (2 * Real.log 2 - 3 * Real.log 5) :=
by
  sorry

end log2_x_value_l82_82558


namespace directrix_of_parabola_l82_82132

noncomputable def parabola_directrix (x : ℝ) : ℝ := 4 * x^2 + 4 * x + 1

theorem directrix_of_parabola :
  ∃ (y : ℝ) (x : ℝ), parabola_directrix x = y ∧ y = 4 * (x + 1/2)^2 + 3/4 ∧ y - 1/16 = 11/16 :=
by
  sorry

end directrix_of_parabola_l82_82132


namespace decrypt_text_l82_82502

variable (encrypted_text : String)
variable (pairs : List (Char × Char))

theorem decrypt_text :
  ∃ decrypted_text : String, decrypted_text = "Когда я употребляю какое-нибудь слово, - сказал Шалтай-Болтай достаточно презрительно, - оно значит только то, что я хочу, чтобы оно означало, ни больше, ни меньше." ∧
  (∀ (c : Char), c ∈ encrypted_text → (c ∈ pairs.map Prod.fst ∨ c ∈ pairs.map Prod.snd)) :=
by
  sorry

end decrypt_text_l82_82502


namespace det_matrix_4x4_is_neg_40_l82_82491

def matrix_4x4 : Matrix (Fin 4) (Fin 4) ℤ := ![
  ![3, 1, -1, 2],
  ![-3, 1, 4, -5],
  ![2, 0, 1, -1],
  ![3, -5, 4, -4] 
]

theorem det_matrix_4x4_is_neg_40 : matrix.det matrix_4x4 = -40 := 
by
  sorry

end det_matrix_4x4_is_neg_40_l82_82491


namespace math_problem_l82_82789

theorem math_problem :
  (625.3729 * (4500 + 2300 ^ 2) - Real.sqrt 84630) / (1500 ^ 3 * 48 ^ 2) = 0.0004257 :=
by
  sorry

end math_problem_l82_82789


namespace shara_shells_final_count_l82_82690

def initial_shells : ℕ := 20
def first_vacation_found : ℕ := 5 * 3 + 6
def first_vacation_lost : ℕ := 4
def second_vacation_found : ℕ := 4 * 2 + 7
def second_vacation_gifted : ℕ := 3
def third_vacation_found : ℕ := 8 + 4 + 3 * 2
def third_vacation_misplaced : ℕ := 5

def total_shells_after_first_vacation : ℕ :=
  initial_shells + first_vacation_found - first_vacation_lost

def total_shells_after_second_vacation : ℕ :=
  total_shells_after_first_vacation + second_vacation_found - second_vacation_gifted

def total_shells_after_third_vacation : ℕ :=
  total_shells_after_second_vacation + third_vacation_found - third_vacation_misplaced

theorem shara_shells_final_count : total_shells_after_third_vacation = 62 := by
  sorry

end shara_shells_final_count_l82_82690


namespace even_three_digit_numbers_sum_tens_units_14_l82_82212

theorem even_three_digit_numbers_sum_tens_units_14 : 
  ∃ n : ℕ, 
  (100 ≤ n ∧ n < 1000) ∧ 
  (n % 2 = 0) ∧
  (let t := (n / 10) % 10 in let u := n % 10 in t + u = 14) ∧
  n = 18 := sorry

end even_three_digit_numbers_sum_tens_units_14_l82_82212


namespace schedule_lecturers_l82_82815

theorem schedule_lecturers :
  let num_lecturers := 7
  ∧ let permutations := Nat.factorial num_lecturers
  ∧ let restricted_permutations := permutations / 2
  in restricted_permutations = 2520 :=
by
    let num_lecturers := 7
    let permutations := Nat.factorial num_lecturers
    let restricted_permutations := permutations / 2
    show restricted_permutations = 2520
    sorry

end schedule_lecturers_l82_82815


namespace possible_angles_of_quadrilateral_l82_82625

theorem possible_angles_of_quadrilateral (A B C D : Type) [quadrilateral A B C D]
    (h1 : side_eq A B B C)
    (h2 : side_eq B C C D)
    (h3 : ∠A = 70)
    (h4 : ∠B = 100) :
    (∠C = 60 ∧ ∠D = 130) ∨ (∠C = 140 ∧ ∠D = 50) := sorry

end possible_angles_of_quadrilateral_l82_82625


namespace asymptotes_of_hyperbola_eq_m_l82_82526

theorem asymptotes_of_hyperbola_eq_m :
  ∀ (m : ℝ), (∀ (x y : ℝ), (x^2 / 16 - y^2 / 25 = 1) → (y = m * x ∨ y = -m * x)) → m = 5 / 4 :=
by 
  sorry

end asymptotes_of_hyperbola_eq_m_l82_82526


namespace exponent_base_16_l82_82593

theorem exponent_base_16 (x : ℝ) : 16 ^ x = 4 ^ 14 → x = 7 :=
by
  sorry  

end exponent_base_16_l82_82593


namespace fourth_student_number_systematic_sampling_l82_82243

theorem fourth_student_number_systematic_sampling :
  ∀ (students : Finset ℕ), students = Finset.range 55 →
  ∀ (sample_size : ℕ), sample_size = 4 →
  ∀ (numbers_in_sample : Finset ℕ),
  numbers_in_sample = {3, 29, 42} →
  ∃ (fourth_student : ℕ), fourth_student = 44 :=
  by sorry

end fourth_student_number_systematic_sampling_l82_82243


namespace find_distance_OP_l82_82201

noncomputable def distance_OP (P : ℝ × ℝ) := real.sqrt 33

theorem find_distance_OP :
  (∃ P : ℝ × ℝ,
     let F1 := (-real.sqrt 7, 0),
         F2 := (real.sqrt 7, 0),
         hyperbola := λ P : ℝ × ℝ, P.1^2 / 4 - P.2^2 / 3 = 1,
         distance_ratio := λ P : ℝ × ℝ, dist P F1 / dist P F2 = 2 in
     hyperbola P ∧ distance_ratio P) → ∃ P : ℝ × ℝ, dist (0, 0) P = distance_OP P :=
by
  sorry

end find_distance_OP_l82_82201


namespace N_composite_l82_82871

theorem N_composite :
  let N := 7 * 9 * 13 + 2020 * 2018 * 2014 in
  ¬Nat.prime N := 
by
  sorry

end N_composite_l82_82871


namespace tetrahedron_faces_congruent_l82_82271

theorem tetrahedron_faces_congruent
  (tetrahedron : Type)
  (vertices : tetrahedron → set ℝ³)
  (angles_sum_180 : ∀ (v : tetrahedron), 
    let angles := {α ∈ ℝ | ∃ (faces ∈ vertices v), α ∈ faces} in 
    sum angles = 180) :
  ∀ (f g : tetrahedron), congruent_face f g :=
sorry

end tetrahedron_faces_congruent_l82_82271


namespace sum_of_digits_n_l82_82426

-- Helper function to compute the sum of digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |> List.sum

theorem sum_of_digits_n :
  ∃ n, n = Int.gcd (4665 - 1305) (Int.gcd (6905 - 4665) (6905 - 1305)) ∧ sum_of_digits n = 4 :=
by
  have h : 4665 - 1305 = 3360 := by norm_num
  have h1 : 6905 - 4665 = 2240 := by norm_num
  have h2 : 6905 - 1305 = 5600 := by norm_num
  let n := Int.gcd 3360 (Int.gcd 2240 5600)
  use n
  split
  · suffices : n = 1120, from this
    rw [h, h1, h2]
    exact Eq.symm (Int.gcd_gcd_1120 3360 2240 5600)
  · unfold sum_of_digits
    norm_num
    exact rfl

end sum_of_digits_n_l82_82426


namespace average_height_of_females_at_school_l82_82700

-- Define the known quantities and conditions
variable (total_avg_height male_avg_height female_avg_height : ℝ)
variable (male_count female_count : ℕ)

-- Given conditions
def conditions :=
  total_avg_height = 180 ∧ 
  male_avg_height = 185 ∧ 
  male_count = 2 * female_count ∧
  (male_count + female_count) * total_avg_height = male_count * male_avg_height + female_count * female_avg_height

-- The theorem we want to prove
theorem average_height_of_females_at_school (total_avg_height male_avg_height female_avg_height : ℝ)
    (male_count female_count : ℕ) (h : conditions total_avg_height male_avg_height female_avg_height male_count female_count) :
    female_avg_height = 170 :=
  sorry

end average_height_of_females_at_school_l82_82700


namespace hexagon_monochromatic_triangles_l82_82044

theorem hexagon_monochromatic_triangles :
  let hexagon_edges := 15 -- $\binom{6}{2}$
  let monochromatic_tri_prob := (1 / 3) -- Prob of one triangle being monochromatic
  let combinations := 20 -- $\binom{6}{3}$, total number of triangles in K_6
  let exactly_two_monochromatic := (combinations.choose 2) * (monochromatic_tri_prob ^ 2) * ((2 / 3) ^ 18)
  (exactly_two_monochromatic = 49807360 / 3486784401) := sorry

end hexagon_monochromatic_triangles_l82_82044


namespace product_of_real_parts_of_solutions_l82_82696

theorem product_of_real_parts_of_solutions (x : ℂ) :
  (x * x - 4 * x = -4 - 4 * complex.I) →
  let a : ℂ := 2 - real.sqrt 2
  let b : ℂ := 2 + real.sqrt 2
  (a.re = 2 - real.sqrt 2) ∧ (b.re = 2 + real.sqrt 2) →
  a.re * b.re = 2 := by
  sorry

end product_of_real_parts_of_solutions_l82_82696


namespace simplify_expr1_simplify_expr2_l82_82347

-- Proof problem for the first expression
theorem simplify_expr1 (x y : ℤ) : (2 - x + 3 * y + 8 * x - 5 * y - 6) = (7 * x - 2 * y -4) := 
by 
   -- Proving steps would go here
   sorry

-- Proof problem for the second expression
theorem simplify_expr2 (a b : ℤ) : (15 * a^2 * b - 12 * a * b^2 + 12 - 4 * a^2 * b - 18 + 8 * a * b^2) = (11 * a^2 * b - 4 * a * b^2 - 6) := 
by 
   -- Proving steps would go here
   sorry

end simplify_expr1_simplify_expr2_l82_82347


namespace twins_future_age_l82_82080

variable {x y : ℝ}

-- Conditions
def twins_current_age (x : ℝ) : Prop := x^2 = 8

-- Proof statement
theorem twins_future_age (h : twins_current_age x) : y = 3 :=
by
  have h1 : x = Real.sqrt 8 := by sorry
  have h2 : (x + y)^2 = x^2 + 17 := by sorry
  have h3 : 25 = 8 + 17 := by
    norm_num
  sorry

#eval twins_future_age

end twins_future_age_l82_82080


namespace jake_more_balloons_than_allan_emily_l82_82477

def balloons_allan : ℕ := 6
def balloons_jake_initial : ℕ := 3
def balloons_jake_additional : ℕ := 4
def balloons_emily : ℕ := 5

def combined_balloons_allan_emily : ℕ := balloons_allan + balloons_emily
def total_balloons_jake : ℕ := balloons_jake_initial + balloons_jake_additional

theorem jake_more_balloons_than_allan_emily :
  total_balloons_jake - combined_balloons_allan_emily = -4 := by
    sorry

end jake_more_balloons_than_allan_emily_l82_82477


namespace cost_of_plane_ticket_l82_82991

theorem cost_of_plane_ticket 
  (total_cost : ℤ) (hotel_cost_per_day_per_person : ℤ) (num_people : ℤ) (num_days : ℤ) (plane_ticket_cost_per_person : ℤ) :
  total_cost = 120 →
  hotel_cost_per_day_per_person = 12 →
  num_people = 2 →
  num_days = 3 →
  (total_cost - num_people * hotel_cost_per_day_per_person * num_days) = num_people * plane_ticket_cost_per_person →
  plane_ticket_cost_per_person = 24 :=
by
  intros h1 h2 h3 h4 h5
  -- Proof steps would go here
  sorry

end cost_of_plane_ticket_l82_82991


namespace chromatic_number_three_element_subsets_l82_82429

open finset

noncomputable theory

def chromatic_number {V : Type*} (G : simple_graph V) : ℕ :=
  find_min (λ k, ∃ f : V → fin k, ∀ ⦃v w⦄, G.adj v w → f v ≠ f w)

def three_element_subsets (n : ℕ) : finset (finset ℕ) :=
  (finset.range n).powerset_len 3

def G (n : ℕ) : simple_graph (finset ℕ) :=
{ adj := λ A B, (A ∩ B).card = 1,
  symm := λ A B h, by rwa [inter_comm],
  loopless := λ A h, by
    { rw [←inter_self A] at h,
      exact nat.not_lt_zero _ (nat.card_self_ne_zero h) } }

theorem chromatic_number_three_element_subsets (k : ℕ) :
  chromatic_number (G (2^k)) = (2^k - 1) * (2^k - 2) / 6 :=
begin
  sorry
end

end chromatic_number_three_element_subsets_l82_82429


namespace time_to_cover_length_l82_82844

def escalator_speed : ℝ := 8  -- The speed of the escalator in feet per second
def person_speed : ℝ := 2     -- The speed of the person in feet per second
def escalator_length : ℝ := 160 -- The length of the escalator in feet

theorem time_to_cover_length : 
  (escalator_length / (escalator_speed + person_speed) = 16) :=
by 
  sorry

end time_to_cover_length_l82_82844


namespace find_angle_A_find_range_sinA_sinB_sinC_l82_82554

variables (a b c : ℝ)
variables (A B C : ℝ)
variables (AB AC : ℝ)

-- Given conditions:
-- Sides opposite to angles in a triangle
-- Dot product condition
-- Trigonometric condition

axiom sides_opposite_to_angles_triangle : ∀ (a b c A B C : ℝ), a / sin A = b / sin B = c / sin C ∧ a = b ∧ a = c

axiom given_dot_product_condition : AB * AC = 4
axiom given_trigonometric_condition : a * c * sin B = 8 * sin A

-- Proof to find value of angle A
theorem find_angle_A : A = π/3 := sorry

-- Proof to find range of sin A * sin B * sin C
theorem find_range_sinA_sinB_sinC : 0 < sin A * sin B * sin C ∧ sin A * sin B * sin C ≤ (3 * sqrt 3) / 8 := sorry

end find_angle_A_find_range_sinA_sinB_sinC_l82_82554


namespace rearrange_1_to_200_l82_82494

theorem rearrange_1_to_200 : ∃ (f : ℕ → ℕ), bijective f ∧ 
  (∀ n, n < 199 → (f (n + 1) = f n + 3 ∨ f (n + 1) = f n + 5)) ∧
  (∀ n, f n ≥ 1 ∧ f n ≤ 200) :=
sorry

end rearrange_1_to_200_l82_82494


namespace polynomial_coefficients_sum_2017_l82_82540

theorem polynomial_coefficients_sum_2017 :
  ∃ (a : fin 2018 → ℝ),
  (∀ x : ℝ, (1 - 2 * x)^2017 = ∑ i in finset.range 2018, a i * (x - 1) ^ i)
  → (a ⟨1, by norm_num⟩ - 2 * a ⟨2, by norm_num⟩ + 3 * a ⟨3, by norm_num⟩
     - 4 * a ⟨4, by norm_num⟩ + ... - 2016 * a ⟨2016, by norm_num⟩ 
     + 2017 * a ⟨2017, by norm_num⟩ = -4034) :=
begin
  have : ∀ f : ℕ → ℕ, ∀ (j : ℕ), 
    ∑ i : ℕ in finset.range (2018), f i * (-2)^j
    → sorry,
end

end polynomial_coefficients_sum_2017_l82_82540


namespace fraction_draw_l82_82970

theorem fraction_draw (john_wins : ℚ) (mike_wins : ℚ) (h_john : john_wins = 4 / 9) (h_mike : mike_wins = 5 / 18) :
    1 - (john_wins + mike_wins) = 5 / 18 :=
by
    rw [h_john, h_mike]
    sorry

end fraction_draw_l82_82970


namespace num_students_third_class_num_students_second_class_l82_82802

-- Definition of conditions for both problems
def class_student_bounds (n : ℕ) : Prop := 40 < n ∧ n ≤ 50
def option_one_cost (n : ℕ) : ℕ := 40 * n * 7 / 10
def option_two_cost (n : ℕ) : ℕ := 40 * (n - 6) * 8 / 10

-- Problem Part 1
theorem num_students_third_class (x : ℕ) (h1 : class_student_bounds x) (h2 : option_one_cost x = option_two_cost x) : x = 48 := 
sorry

-- Problem Part 2
theorem num_students_second_class (y : ℕ) (h1 : class_student_bounds y) (h2 : option_one_cost y < option_two_cost y) : y = 49 ∨ y = 50 := 
sorry

end num_students_third_class_num_students_second_class_l82_82802


namespace Laran_sells_small_posters_for_6_l82_82996

/-
Laran sells her small posters for $6 each.
-/
theorem Laran_sells_small_posters_for_6 :
  ∀ (poster_total : ℕ) (large_posters : ℕ) (small_posters : ℕ) 
    (large_poster_cost : ℕ) (small_poster_cost : ℕ) 
    (large_poster_price : ℕ) (small_poster_price : ℕ) 
    (weekly_profit : ℕ),

    (poster_total = 5) →
    (large_posters = 2) →
    (small_posters = poster_total - large_posters) →
    (large_poster_cost = 5) →
    (small_poster_cost = 3) →
    (large_poster_price = 10) →
    (weekly_profit = 95) →
    (daily_profit: ℕ := weekly_profit / 5) →
    (daily_profit_large := large_posters * (large_poster_price - large_poster_cost)) →
    (daily_profit_small := daily_profit - daily_profit_large) →
    (profit_per_small_poster := daily_profit_small / small_posters) →
    (small_poster_price = small_poster_cost + profit_per_small_poster) →
    small_poster_price = 6 := 
by
  intros,
  sorry  -- The actual proof goes here

end Laran_sells_small_posters_for_6_l82_82996


namespace inequality_sum_reciprocal_l82_82691

theorem inequality_sum_reciprocal (n : ℕ) (h : 1 ≤ n): 
  (2 * n) / (3 * n + 1) ≤ (∑ k in Finset.Icc (n + 1) (2 * n), (1 / k : ℝ)) ∧ 
  (∑ k in Finset.Icc (n + 1) (2 * n), (1 / k : ℝ)) ≤ (3 * n + 1) / (4 * n + 4) :=
sorry

end inequality_sum_reciprocal_l82_82691


namespace mean_difference_l82_82708

variable (S : ℝ)  -- Sum of the correct incomes excluding the highest income
variable (n : ℕ) (h_n : n = 1200)
variable (i_correct i_incorrect : ℝ) 
  (h_correct : i_correct = 105000) 
  (h_incorrect : i_incorrect = 1050000)

theorem mean_difference :
  let mean_actual := (S + i_correct) / n in  
  let mean_incorrect := (S + i_incorrect) / n in
  mean_incorrect - mean_actual = 787.5 :=
by
  sorry

end mean_difference_l82_82708


namespace tangent_parallel_x_axis_implies_a_eq_1_h_a_lt_zero_for_a_in_range_l82_82937

-- Definitions for given functions
def f (x a : ℝ) := (x - a - 1) * Real.exp x
def g (x a : ℝ) := (1 / 2) * x * x - a * x
def F (x a : ℝ) := f x a - (a + 1) * g x a
def h (a : ℝ) := (1 / 2) * (a^3 + a^2) - Real.exp a

-- Proof that the tangent to f(x) at (1, f(1)) is parallel to the x-axis implies a = 1
theorem tangent_parallel_x_axis_implies_a_eq_1 (f : ℝ → ℝ) (a : ℝ) 
  (h₁ : ∀ x, f x = (x - a - 1) * Real.exp x) 
  (tangent_parallel : Derivative f 1 = 0) : 
  a = 1 := 
sorry

-- Proof that if -1 < a < -3/4, then h(a) < 0
theorem h_a_lt_zero_for_a_in_range (a : ℝ)
  (h₁ : ∀ x, F x a = f x a - (a + 1) * g x a)
  (h₂ : -1 < a ∧ a < -3/4) :
  h a < 0 := 
sorry

end tangent_parallel_x_axis_implies_a_eq_1_h_a_lt_zero_for_a_in_range_l82_82937


namespace solution_is_consecutive_even_integers_l82_82497

def consecutive_even_integers_solution_exists : Prop :=
  ∃ (x y z w : ℕ), (x + y + z + w = 68) ∧ 
                   (y = x + 2) ∧ (z = x + 4) ∧ (w = x + 6) ∧
                   (x % 2 = 0) ∧ (y % 2 = 0) ∧ (z % 2 = 0) ∧ (w % 2 = 0)

theorem solution_is_consecutive_even_integers : consecutive_even_integers_solution_exists :=
sorry

end solution_is_consecutive_even_integers_l82_82497


namespace line_intersects_circle_and_passes_through_point_l82_82938

noncomputable def is_line (m : ℝ) : ℝ → ℝ → ℝ := λ x y, (m + 1) * x + 2 * y + 2 * m - 2

noncomputable def is_circle : ℝ → ℝ → ℝ := λ x y, x^2 + (y - 1)^2 - 9

theorem line_intersects_circle_and_passes_through_point (m : ℝ) :
    ∃ (x y : ℝ), is_line m x y = 0 ∧ is_circle x y = 0 ∧ (is_line m (-2) 2 = 0) :=
begin
  sorry
end

end line_intersects_circle_and_passes_through_point_l82_82938


namespace find_number_l82_82776

-- Define the average of the first 7 positive multiples of a number x
def average_of_multiples (x : ℕ) : ℕ :=
  (x + 2*x + 3*x + 4*x + 5*x + 6*x + 7*x) / 7

-- Define the median of the first 3 positive multiples of a number n
def median_of_multiples (n : ℕ) : ℕ :=
  2 * n

-- Define the number n
def n : ℕ := 16

-- Assuming the condition a^2 - b^2 = 0, prove that x = 8
theorem find_number (x : ℕ) : 
  let a := average_of_multiples x in
  let b := median_of_multiples n in
  a^2 - b^2 = 0 → x = 8 :=
by
  intros a b h
  sorry

end find_number_l82_82776


namespace proof_problem_l82_82111

noncomputable def f (x : ℝ) : ℝ :=
  Real.log ((1 + Real.sqrt x) / (1 - Real.sqrt x))

theorem proof_problem (x : ℝ) (h1 : 0 ≤ x) (h2 : x < 1) :
  f ( (5 * x + 2 * x^2) / (1 + 5 * x + 3 * x^2) ) = Real.sqrt 5 * f x :=
by
  sorry

end proof_problem_l82_82111


namespace compare_subtract_one_l82_82220

theorem compare_subtract_one (a b : ℝ) (h : a < b) : a - 1 < b - 1 :=
sorry

end compare_subtract_one_l82_82220


namespace point_D_is_on_y_axis_l82_82014

def is_on_y_axis (p : ℝ × ℝ) : Prop := p.fst = 0

def point_A : ℝ × ℝ := (3, 0)
def point_B : ℝ × ℝ := (1, 2)
def point_C : ℝ × ℝ := (2, 1)
def point_D : ℝ × ℝ := (0, -3)

theorem point_D_is_on_y_axis : is_on_y_axis point_D :=
by
  sorry

end point_D_is_on_y_axis_l82_82014


namespace poly_has_int_solution_iff_l82_82522

theorem poly_has_int_solution_iff (a : ℤ) : 
  (a > 0 ∧ (∃ x : ℤ, a * x^2 + 2 * (2 * a - 1) * x + 4 * a - 7 = 0)) ↔ (a = 1 ∨ a = 5) :=
by {
  sorry
}

end poly_has_int_solution_iff_l82_82522


namespace find_dividend_l82_82888

-- Define the given constants
def quotient : ℕ := 909899
def divisor : ℕ := 12

-- Define the dividend as the product of divisor and quotient
def dividend : ℕ := divisor * quotient

-- The theorem stating the equality we need to prove
theorem find_dividend : dividend = 10918788 := by
  sorry

end find_dividend_l82_82888


namespace proposition_1_proposition_4_l82_82105

-- Definitions
variable {a b c : Type} (Line : Type) (Plane : Type)
variable (a b c : Line) (γ : Plane)

-- Given conditions
variable (distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)

-- Parallel and perpendicular relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)

-- Propositions to prove
theorem proposition_1 (H1 : parallel a b) (H2 : parallel b c) : parallel a c := sorry

theorem proposition_4 (H3 : perpendicular a γ) (H4 : perpendicular b γ) : parallel a b := sorry

end proposition_1_proposition_4_l82_82105


namespace tangent_line_at_2_l82_82192

-- Define the function f(x)
def f (x : ℝ) : ℝ := 2 * x^2 - x * (4 - f 2)

-- Define f'
def f' (x : ℝ) : ℝ := 4 * x - f' 2

-- Statement of the theorem
theorem tangent_line_at_2 (x : ℝ) (y : ℝ) : 
  (4 * x - y - 8 = 0) ↔ (x = 2 ∧ y = f 2) := 
by 
  sorry

end tangent_line_at_2_l82_82192


namespace percent_decrease_correct_l82_82434

noncomputable def lawn_chair_orig := 74.95
noncomputable def lawn_chair_sale := 59.95
noncomputable def grill_orig := 120
noncomputable def grill_sale := 100
noncomputable def patio_table_orig := 250
noncomputable def patio_table_sale := 225

noncomputable def total_original_price := lawn_chair_orig + grill_orig + patio_table_orig
noncomputable def total_sale_price := lawn_chair_sale + grill_sale + patio_table_sale
noncomputable def total_decrease := total_original_price - total_sale_price
noncomputable def percent_decrease := (total_decrease / total_original_price) * 100

theorem percent_decrease_correct :
  percent_decrease ≈ 13.48 :=
sorry

end percent_decrease_correct_l82_82434


namespace greening_cost_of_triangle_l82_82746

noncomputable def greening_cost (a b c cost_per_sqm : ℝ) : ℝ :=
  if (a^2 + b^2 = c^2) then
    let area := 1 / 2 * a * b in
    area * cost_per_sqm
  else
    0

theorem greening_cost_of_triangle :
  greening_cost 8 15 17 50 = 3000 :=
by
  sorry

end greening_cost_of_triangle_l82_82746


namespace Phoenix_roots_prod_l82_82504

def Phoenix_eqn (a b c : ℝ) : Prop :=
  a ≠ 0 ∧ a + b + c = 0

theorem Phoenix_roots_prod {m n : ℝ} (hPhoenix : Phoenix_eqn 1 m n)
  (hEqualRoots : (m^2 - 4 * n) = 0) : m * n = -2 :=
by sorry

end Phoenix_roots_prod_l82_82504


namespace average_speed_l82_82392

-- Conditions as Lean definitions:
def speed_1 := 10 -- mph
def speed_2 := 50 -- mph
def time_fraction_at_speed_2 := 0.75 -- fraction of total time

-- Question translated into a theorem:
theorem average_speed
  (T : ℝ) -- total time of the journey
  (hT : T > 0) -- ensuring total time is positive
  (distance_50 : ℝ := speed_2 * (time_fraction_at_speed_2 * T))
  (time_fraction_at_speed_1 := 1 - time_fraction_at_speed_2) -- fraction of time at the first speed
  (distance_10 : ℝ := speed_1 * (time_fraction_at_speed_1 * T)) :
  (distance_50 + distance_10) / T = 40 :=
by
  sorry

end average_speed_l82_82392


namespace overall_average_mark_l82_82250

theorem overall_average_mark (n : ℕ) (avg_first_10 avg_last_11 mark_11 : ℕ) (h_n : n = 22)
  (h_avg_first_10 : avg_first_10 = 55) (h_avg_last_11 : avg_last_11 = 40)
  (h_mark_11 : mark_11 = 66) : 
  let total_marks := 10 * avg_first_10 + (n - 11) * avg_last_11 + mark_11 in
  let overall_avg := total_marks / n in
  overall_avg = 45 := by
    sorry

end overall_average_mark_l82_82250


namespace betty_needs_five_boxes_l82_82488

def betty_oranges (total_oranges first_box second_box max_per_box : ℕ) : ℕ :=
  let remaining_oranges := total_oranges - (first_box + second_box)
  let full_boxes := remaining_oranges / max_per_box
  let extra_box := if remaining_oranges % max_per_box == 0 then 0 else 1
  full_boxes + 2 + extra_box

theorem betty_needs_five_boxes :
  betty_oranges 120 30 25 30 = 5 := 
by
  sorry

end betty_needs_five_boxes_l82_82488


namespace triangle_area_given_medians_l82_82672

theorem triangle_area_given_medians :
  ∀ (A B C D E G : Type) [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C] 
  [InnerProductSpace ℝ D] [InnerProductSpace ℝ E] [metricSpace G],
  (medians_intersect_at_centroid A B C G) ∧
  (medians_form_angle A D B E 45) ∧ 
  (medians_lengths A D 18 B E 24) → 
  (area_of_triangle A B C) = (288 * Real.sqrt 2) :=
by
  sorry

end triangle_area_given_medians_l82_82672


namespace min_value_trig_expr_l82_82136

theorem min_value_trig_expr (x : ℝ) (h1 : 0 < x) (h2 : x < π / 2) : 
  (sin x + sec x)^2 + (cos x + csc x)^2 ≥ 3 := 
sorry

end min_value_trig_expr_l82_82136


namespace square_area_l82_82977

theorem square_area
  (X Y Z W M N O : Type)
  [add_comm_monoid O]
  (square_XYZW : is_square X Y Z W)
  (on_XZ_M : M ∈ segment X Z)
  (on_XY_N : N ∈ segment X Y)
  (perpendicular_YM_WN : is_perpendicular (line_through Y M) (line_through W N))
  (distance_YO : distance Y O = 8)
  (distance_MO : distance M O = 9) :
  area square_XYZW = 256 :=
sorry

end square_area_l82_82977


namespace evaluate_N_l82_82124

theorem evaluate_N (N : ℕ) :
    988 + 990 + 992 + 994 + 996 = 5000 - N → N = 40 :=
by
  sorry

end evaluate_N_l82_82124


namespace sum_of_roots_quadratic_l82_82221

theorem sum_of_roots_quadratic :
  ∀ (a b : ℝ), (a^2 - a - 2 = 0) → (b^2 - b - 2 = 0) → (a + b = 1) :=
by
  intro a b
  intros
  sorry

end sum_of_roots_quadratic_l82_82221


namespace two_disjoint_odd_cycles_in_K10_l82_82755

theorem two_disjoint_odd_cycles_in_K10
  (K10 : simple_graph (fin 10))
  (color : ∀ {u v : fin 10}, u ≠ v → bool) : 
  (∃ (C1 C2 : finset (fin 10)), 
      K10.is_cycle C1 ∧ C1.card % 2 = 1 ∧ 
      K10.is_cycle C2 ∧ C2.card % 2 = 1 ∧ 
      disjoint C1 C2 ∧ 
      (∀ (u v : fin 10) (h : u ≠ v), 
          u ∈ C1 ∨ u ∈ C2 → v ∈ C1 ∨ v ∈ C2 → color h = color h)) :=
sorry

end two_disjoint_odd_cycles_in_K10_l82_82755


namespace sum_of_extreme_values_of_g_l82_82499

def g (x : ℝ) : ℝ := abs (x - 1) + abs (x - 5) - 2 * abs (x - 3)

theorem sum_of_extreme_values_of_g :
  ∃ (min_val max_val : ℝ), 
    (∀ x : ℝ, 1 ≤ x ∧ x ≤ 6 → g x ≥ min_val) ∧ 
    (∀ x : ℝ, 1 ≤ x ∧ x ≤ 6 → g x ≤ max_val) ∧ 
    (min_val = -8) ∧ 
    (max_val = 0) ∧ 
    (min_val + max_val = -8) := 
by
  sorry

end sum_of_extreme_values_of_g_l82_82499


namespace value_of_a_l82_82961

def f (a : ℝ) (x : ℝ) : ℝ :=
if x ≥ 0 then a + a^x else 3 + (a - 1) * x

theorem value_of_a (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) (h₂ : ∀ x₁ x₂, x₁ ≠ x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) > 0) : a ≥ 2 :=
sorry

end value_of_a_l82_82961


namespace geometric_locus_of_broken_line_endpoints_l82_82797

open Real

theorem geometric_locus_of_broken_line_endpoints (a : ℝ) (x y z : ℝ) :
  (∃ (l : ℕ → ℝ) (i : ℕ → ℝ × ℝ × ℝ),
    (∀ n, (i n).1 ≠ 0 ∧ (i n).2 ≠ 0 ∧ (i n).3 ≠ 0) ∧
    (∀ n, (i n).1 = sign x * abs (i n).1 ∧ (i n).2 = sign y * abs (i n).2 ∧ (i n).3 = sign z * abs (i n).3) ∧
    (∑ n, (l n)) = a ∧
    (∑ n, (abs ((i n).1)) + abs ((i n).2) + abs ((i n).3)) = abs x + abs y + abs z) →
  abs x + abs y + abs z ≤ a :=
sorry

end geometric_locus_of_broken_line_endpoints_l82_82797


namespace cheolsu_weight_l82_82092

variable (C M : ℝ)

theorem cheolsu_weight:
  (C = (2/3) * M) →
  (C + 72 = 2 * M) →
  C = 36 :=
by
  intros h1 h2
  sorry

end cheolsu_weight_l82_82092


namespace seven_masters_possible_eight_masters_impossible_l82_82241

-- Define the structure of the tournament and the requirements
def chess_tournament := 
  ∀ (participants : ℕ) (games_per_participant : ℕ),
    ∃ wins draws total_points max_points points_required : ℕ,
    participants = 12 ∧
    games_per_participant = 11 ∧
    total_points = 66 ∧
    max_points = if 11 * 1 then 11 else undefined ∧ -- Maximum points if a player wins all games
    points_required = max_points * 7 / 10 + 1 ∧ -- More than 70% of max points
    ∀ (master_participants : ℕ),
      (master_participants = 7 → total_points / 12 > points_required) ∧ -- Check if 7 participants can achieve points_required
      (master_participants = 8 → total_points / 12 <= points_required) -- Check if 8 participants cannot achieve points_required

-- Translate to specific statements
theorem seven_masters_possible :
  chess_tournament → ∃ (master_participants : ℕ), master_participants = 7 ∧ (total_points := 66 / 12 > (11 * 7 / 10 + 1)) :=
sorry

theorem eight_masters_impossible :
  chess_tournament → ∃ (master_participants : ℕ), master_participants ≠ 8 ∧ (total_points := 66 / 12 <= (11 * 7 / 10 + 1)) :=
sorry

end seven_masters_possible_eight_masters_impossible_l82_82241


namespace product_of_even_areas_eq_odd_areas_l82_82460

theorem product_of_even_areas_eq_odd_areas (n : ℕ) (areas : Fin (2 * n) → ℝ)
  (h1 : ∀ i : Fin n, areas ⟨2 * i.1, _⟩ = areas ⟨2 * i.1 + 1, _⟩) :
  (∏ i in Finset.range n, areas ⟨2 * i, _⟩) = (∏ i in Finset.range n, areas ⟨2 * i + 1, _⟩) :=
    sorry

end product_of_even_areas_eq_odd_areas_l82_82460


namespace f_periodic_even_l82_82867

def f (x : ℝ) : ℝ := Real.sin (Real.pi / 4 + x) * Real.sin (Real.pi / 4 - x)

theorem f_periodic_even : (∀ x : ℝ, f x = f (-x)) ∧ (∀ x : ℝ, f (x + Real.pi) = f x) :=
by
  sorry

end f_periodic_even_l82_82867


namespace find_functions_l82_82127

noncomputable def prob (f g : ℝ → ℝ) : Prop :=
∀ x y : ℝ, f(x - f(y)) = x * f(y) - y * f(x) + g(x)

theorem find_functions (f g : ℝ → ℝ) (h : prob f g) :
    (∀ x, f(x) = 0 ∧ g(x) = 0) ∨ (∀ x, f(x) = x ∧ g(x) = 0) :=
sorry

end find_functions_l82_82127


namespace solutions_divisible_by_2_alpha_plus_1_l82_82642

theorem solutions_divisible_by_2_alpha_plus_1 {α : ℕ} {q : ℕ} (h1 : q % 2 = 1) (h2 : 0 < q) :
  ∀ (m : ℕ) (n : ℕ), n = 2^α * q →
  (∃ f_n_m, f_n_m = {x : ℕ → ℤ // ∑ i in range n, (x i)^2 = m}.card) →
  2^(α + 1) ∣ f_n_m :=
by sorry

end solutions_divisible_by_2_alpha_plus_1_l82_82642


namespace triangle_area_correct_l82_82647

noncomputable def triangle_area : ℝ :=
  let a := ![2, -3, 1]
  let b := ![4, -1, 5]
  (∥ a.cross_product b ∥) / 2

theorem triangle_area_correct : 
  triangle_area = real.sqrt 332 / 2 :=
sorry

end triangle_area_correct_l82_82647


namespace man_speed_is_approximately_54_009_l82_82451

noncomputable def speed_in_kmh (d : ℝ) (t : ℝ) : ℝ := 
  -- Convert distance to kilometers and time to hours
  let distance_km := d / 1000
  let time_hours := t / 3600
  distance_km / time_hours

theorem man_speed_is_approximately_54_009 :
  abs (speed_in_kmh 375.03 25 - 54.009) < 0.001 := 
by
  sorry

end man_speed_is_approximately_54_009_l82_82451


namespace function_extreme_and_slope_l82_82936

noncomputable def function_with_conditions (a b c : ℝ) : ℝ → ℝ := 
  λ x, x^3 + 3 * a * x^2 + 3 * b * x + c

noncomputable def function_derivative (a b c : ℝ) : ℝ → ℝ := 
  λ x, 3 * x^2 + 6 * a * x + 3 * b

theorem function_extreme_and_slope (a b c : ℝ) :
  (function_derivative a b c 2 = 0) ∧
  (function_derivative a b c 1 = -3) →
  (∀ x : ℝ, (x < 0 ∨ x > 2 → function_derivative (-1) 0 c x > 0) ∧
            (0 < x ∧ x < 2 → function_derivative (-1) 0 c x < 0)) ∧
  (function_with_conditions (-1) 0 c 0 - function_with_conditions (-1) 0 c 2 = 4) :=
by
  sorry

end function_extreme_and_slope_l82_82936


namespace siamese_cats_initial_l82_82459

theorem siamese_cats_initial (S : ℕ) (h1 : 20 + S - 20 = 12) : S = 12 :=
by
  sorry

end siamese_cats_initial_l82_82459


namespace average_gas_mileage_round_trip_l82_82039

def distance_home := 150
def mpg_home := 30
def distance_return := 120
def mpg_return := 25
def total_distance := distance_home + distance_return

def gasoline_home : ℝ := distance_home / mpg_home
def gasoline_return : ℝ := distance_return / mpg_return
def total_gasoline : ℝ := gasoline_home + gasoline_return

def average_mileage : ℝ := total_distance / total_gasoline

theorem average_gas_mileage_round_trip : average_mileage = 28 := by
  sorry

end average_gas_mileage_round_trip_l82_82039


namespace abs_expr_eq_sqrt_10_l82_82188

-- Given complex number z and its conjugate
def z : ℂ := -1 - I
def z_conjugate : ℂ := -1 + I

-- Define the expression (1 - z) * z_conjugate
noncomputable def expr : ℂ := (1 - z) * z_conjugate

-- Proof statement: absolute value of expr is equal to sqrt(10)
theorem abs_expr_eq_sqrt_10 : abs expr = Real.sqrt 10 :=
by sorry

end abs_expr_eq_sqrt_10_l82_82188


namespace tangent_line_eq_l82_82889

theorem tangent_line_eq (x y : ℝ) (h_curve : y = x^3 + x + 1) (h_point : x = 1 ∧ y = 3) : 
  y = 4 * x - 1 := 
sorry

end tangent_line_eq_l82_82889


namespace max_value_of_y_l82_82891

noncomputable def y (x : ℝ) := 
  Real.tan (x + Real.pi / 3) -
  Real.tan (x + Real.pi / 4) +
  Real.cos (x + Real.pi / 4)

theorem max_value_of_y : 
  ∃ (max_y : ℝ), max_y = 
    - (Real.cot (Real.pi / 12 + Real.pi / 4)) - 
    Real.tan (Real.pi / 4) + Real.cos (Real.pi / 4) ∧ 
  ∀ (x : ℝ), 
    - 2 * Real.pi / 3 ≤ x ∧ x ≤ - Real.pi / 2 → 
    y x ≤ max_y :=
begin
  let max_y := - 1 / Real.sqrt 3 - 1 + Real.sqrt 2 / 2,
  use max_y,
  sorry
end

end max_value_of_y_l82_82891


namespace factorization_problem_l82_82357

theorem factorization_problem 
  (C D : ℤ)
  (h1 : 15 * y ^ 2 - 76 * y + 48 = (C * y - 16) * (D * y - 3))
  (h2 : C * D = 15)
  (h3 : C * (-3) + D * (-16) = -76)
  (h4 : (-16) * (-3) = 48) : 
  C * D + C = 20 :=
by { sorry }

end factorization_problem_l82_82357


namespace gcd_problem_l82_82794

variable (A B : ℕ)
variable (hA : A = 2 * 3 * 5)
variable (hB : B = 2 * 2 * 5 * 7)

theorem gcd_problem : Nat.gcd A B = 10 :=
by
  -- Proof is omitted.
  sorry

end gcd_problem_l82_82794


namespace price_per_foot_l82_82769

theorem price_per_foot (area : ℝ) (total_cost : ℝ) (price_per_foot : ℝ) :
  area = 289 → total_cost = 3876 → price_per_foot = total_cost / (4 * real.sqrt area) → price_per_foot = 57 :=
by
  intros h_area h_total_cost h_price_per_foot
  rw [h_area, h_total_cost] at h_price_per_foot
  norm_num at h_price_per_foot
  exact h_price_per_foot

end price_per_foot_l82_82769


namespace calculate_S_2_2000_minus_1_l82_82261

def largest_odd_divisor (n : ℕ) : ℕ :=
if n % 2 = 1 then n else largest_odd_divisor (n / 2)

def S (n : ℕ) : ℕ :=
(nat.sum $ λ i, largest_odd_divisor i) 1 (λ i, i <= n)

theorem calculate_S_2_2000_minus_1 : 
  S (2^2000 - 1) = (4^2000 - 1) / 3 :=
sorry

end calculate_S_2_2000_minus_1_l82_82261


namespace number_is_580_l82_82033

noncomputable def find_number (x : ℝ) : Prop :=
  0.20 * x = 116

theorem number_is_580 (x : ℝ) (h : find_number x) : x = 580 :=
  by sorry

end number_is_580_l82_82033


namespace part1_part2_l82_82579

noncomputable def f (x : ℝ) (θ : ℝ) : ℝ := Real.sin (2 * x + θ)

theorem part1 (h₀ : (θ : ℝ) ∈ (0, Real.pi / 2))
  (h₁ : f (Real.pi / 6) θ = 1) : θ = Real.pi / 6 :=
by
  sorry

theorem part2 (h₂ : ∀ x ∈ Set.Icc (0 : ℝ) (Real.pi / 4), 
    ∃ y ∈ Set.Icc (Real.sin(Real.pi / 6)) (1 : ℝ), f x (Real.pi / 6) = y) :
  Set.image (fun x => f x (Real.pi / 6)) (Set.Icc (0 : ℝ) (Real.pi / 4)) = Set.Icc (1 / 2) 1 :=
by
  sorry

end part1_part2_l82_82579


namespace bug_visits_all_vertices_exactly_once_l82_82798

def tetrahedron_vertices : Finset ℕ := {0, 1, 2, 3}

def valid_path (path : List ℕ) : Prop :=
  path.length = 3 ∧
  (∀ i < 3, path.nth i ∈ tetrahedron_vertices) ∧
  (List.nodup path)

def three_moves_paths (start : ℕ) :=
  { path : List ℕ // valid_path (start :: path) ∧ path.head! ≠ start}

def probability_of_visiting_all_vertices (start : ℕ) : ℤ :=
  if start ∈ tetrahedron_vertices then 1 else 0

theorem bug_visits_all_vertices_exactly_once :
  ∀ start ∈ tetrahedron_vertices, 
  probability_of_visiting_all_vertices start = 1 :=
by
  intros start h
  rw [probability_of_visiting_all_vertices]
  split_ifs
  · reflexivity
  · contradiction

end bug_visits_all_vertices_exactly_once_l82_82798


namespace number_of_negative_x_values_l82_82897

theorem number_of_negative_x_values : 
  let N := {n : ℕ // 1 ≤ n ∧ n^2 < 180} 
  in cardinal.mk N = 13 :=
begin
  sorry
end

end number_of_negative_x_values_l82_82897


namespace max_lines_no_triangle_l82_82906

theorem max_lines_no_triangle 
  (n : ℕ) (h : n ≥ 3) (h_no_collinear : ∀ (A B C : ℝ × ℝ), (A ≠ B ∧ A ≠ C ∧ B ≠ C) → ¬ collinear A B C) :
  ∃ K : ℕ, 
    (∀ (lines : finset (ℝ × ℝ)), lines.card = K → ¬ (∃ (A B C : ℝ × ℝ), lines.contains (A, B) ∧ lines.contains (B, C) ∧ lines.contains (A, C))) ∧ 
    (if even n then K = (n^2)/4 else K = (n^2 - 1)/4) :=
sorry

end max_lines_no_triangle_l82_82906


namespace sum_distinct_prime_divisors_of_2520_l82_82766

theorem sum_distinct_prime_divisors_of_2520 : 
  let distinct_prime_divisors (n : ℕ) := {p : ℕ | p.prime ∧ p ∣ n}
  let sum_distinct_prime_divisors (n : ℕ) := (distinct_prime_divisors n).sum id
  sum_distinct_prime_divisors 2520 = 17 :=
by
  sorry

end sum_distinct_prime_divisors_of_2520_l82_82766


namespace sin_C_eq_sqrt_15_div_8_l82_82265

variable {a b c : ℝ}
variable {A B C : ℝ}

-- Given conditions
-- cos A = 1/4
def cos_A (cos_A_val : ℝ) := (cos_A_val = 1/4)
-- b = 2c
def b_eq_2c (b_val c_val : ℝ) := (b_val = 2 * c_val)

-- To prove: sin C = sqrt(15) / 8
theorem sin_C_eq_sqrt_15_div_8 
  (A_val B_val C_val a_val b_val c_val sin_C_val : ℝ)
  (cos_A_cond : cos_A A_val)
  (b_cond : b_eq_2c b_val c_val)
  (h1 : ∀ {x}, sin x = sqrt (1 - cos x ^ 2))
  : sin C_val = sqrt(15) / 8 := 
sorry

end sin_C_eq_sqrt_15_div_8_l82_82265


namespace number_499_in_S_number_499_is_9475_the_499th_number_in_S_is_9475_l82_82294

def S : Set ℕ := { n | ∃ k : ℕ, n = 19 * k + 13 }

theorem number_499_in_S : ∃ n ∈ S, n = 19 * 498 + 13 :=
by
  exists 19 * 498 + 13
  split
  · use 498
    rfl
  rfl

theorem number_499_is_9475 : (19 * 498 + 13) = 9475 := 
by
  calc
    19 * 498 + 13 = 9475 := by norm_num

theorem the_499th_number_in_S_is_9475 : ∃ n ∈ S, n = 9475 :=
by
  exists 9475
  split
  · apply number_499_in_S
  apply number_499_is_9475
  rfl

end number_499_in_S_number_499_is_9475_the_499th_number_in_S_is_9475_l82_82294


namespace max_consecutive_terms_divisible_by_m_l82_82301

theorem max_consecutive_terms_divisible_by_m (m : ℕ) (h : m > 1) (x : ℕ → ℕ)
  (h₁ : ∀ i, 0 ≤ i ∧ i < m → x i = 2 ^ i)
  (h₂ : ∀ i, i ≥ m → x i = ∑ j in finset.range m, x (i - j - 1)) :
  ∃ k, (∀ n, x n % m = 0 → (n + k) < m ∧ (x (n + k) % m = 0)) ∧ k = m - 1 :=
sorry

end max_consecutive_terms_divisible_by_m_l82_82301


namespace f_ordering_l82_82596

noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 1 then x ^ (1 / 1998) else
  if x % 2 = 0 then f (x - 2) else -f (-x)

theorem f_ordering :
  f (101 / 17) < f (98 / 19) ∧ f (98 / 19) < f (104 / 15) := by
  sorry

end f_ordering_l82_82596


namespace jack_grassy_time_is_6_l82_82631

def jack_sandy_time := 19
def jill_total_time := 32
def jill_time_delay := 7
def jack_total_time : ℕ := jill_total_time - jill_time_delay
def jack_grassy_time : ℕ := jack_total_time - jack_sandy_time

theorem jack_grassy_time_is_6 : jack_grassy_time = 6 := by 
  have h1: jack_total_time = 25 := by sorry
  have h2: jack_grassy_time = 6 := by sorry
  exact h2

end jack_grassy_time_is_6_l82_82631


namespace inequality_satisfied_l82_82176

theorem inequality_satisfied (f : ℝ → ℝ) (h : ∀ x, 0 < x ∧ x < (π / 2) → f(x) * tan(x) > deriv f x) :
  sqrt 3 * f(π / 6) > f(π / 3) :=
sorry

end inequality_satisfied_l82_82176


namespace sculpture_cost_NAD_to_CNY_l82_82323

def NAD_to_USD (nad : ℕ) : ℕ := nad / 8
def USD_to_CNY (usd : ℕ) : ℕ := usd * 5

theorem sculpture_cost_NAD_to_CNY (nad : ℕ) : (nad = 160) → (USD_to_CNY (NAD_to_USD nad) = 100) :=
by
  intro h1
  rw [h1]
  -- NAD_to_USD 160 = 160 / 8
  have h2 : NAD_to_USD 160 = 20 := rfl
  -- USD_to_CNY 20 = 20 * 5
  have h3 : USD_to_CNY 20 = 100 := rfl
  -- Concluding the theorem
  rw [h2, h3]
  reflexivity

end sculpture_cost_NAD_to_CNY_l82_82323


namespace polygons_after_cuts_l82_82084

theorem polygons_after_cuts (initial_polygons : ℕ) (cuts : ℕ) 
  (initial_vertices : ℕ) (max_vertices_added_per_cut : ℕ) :
  (initial_polygons = 10) →
  (cuts = 51) →
  (initial_vertices = 100) →
  (max_vertices_added_per_cut = 4) →
  ∃ p, (p < 5 ∧ p ≥ 3) :=
by
  intros h_initial_polygons h_cuts h_initial_vertices h_max_vertices_added_per_cut
  -- proof steps would go here
  sorry

end polygons_after_cuts_l82_82084


namespace maximal_value_sum_areas_l82_82664

variable {A B C P A1 B1 C1 : Point}
variable (p R r : ℝ)
variable [triangle : Triangle ABC]

-- Assuming definitions and conditions related to the problem
axiom P_in_interior_of_ABC : InteriorABC P
axiom AP_intersect_circumcircle_at_A1 : IntersectCircumcircle ABC P A1
axiom BP_intersect_circumcircle_at_B1 : IntersectCircumcircle ABC P B1
axiom CP_intersect_circumcircle_at_C1 : IntersectCircumcircle ABC P C1
axiom semiperimeter_def : p = Semiperimeter ABC
axiom circumradius_def : R = Circumradius ABC
axiom inradius_def : r = Inradius ABC

theorem maximal_value_sum_areas :
  Area (A1BC) + Area (B1AC) + Area (C1AB) ≤ p * (R - r) :=
sorry

end maximal_value_sum_areas_l82_82664


namespace product_of_first_five_terms_l82_82373

def a : ℕ → ℝ
| 0       := 1 / 2
| (n + 1) := 2 * a n

def b (n: ℕ) := 1 / a n

def T (n : ℕ) := ∏ i in Finset.range n, b (i + 1)

theorem product_of_first_five_terms :
  T 5 = 1 / 32 := by
  sorry

end product_of_first_five_terms_l82_82373


namespace smallest_shift_l82_82703

-- Define the function f and its periodic property
def periodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f(x - p) = f(x)

-- Main theorem statement
theorem smallest_shift (f : ℝ → ℝ) (p : ℝ) (hp : periodic f p) : 
  (∃ (a : ℝ), a > 0 ∧ (∀ x, f((x - a) / 4) = f(x / 4)) ∧ ∀ b, (b > 0 → (∀ x, f((x - b) / 4) = f(x / 4)) → a ≤ b)) :=
begin
  use 80,
  split,
  { linarith },
  split,
  { sorry }, -- Proof of the equality
  { sorry }, -- Proof that 80 is the smallest positive such number
end

end smallest_shift_l82_82703


namespace Angie_age_ratio_l82_82079

-- Define Angie's age as a variable
variables (A : ℕ)

-- Give the condition
def Angie_age_condition := A + 4 = 20

-- State the theorem to be proved
theorem Angie_age_ratio (h : Angie_age_condition A) : (A : ℚ) / (A + 4) = 4 / 5 := 
sorry

end Angie_age_ratio_l82_82079


namespace f_of_2_l82_82934

theorem f_of_2 (a : ℝ) (h_pos : a > 0) (h_neq : a ≠ 1) (h_f1 : a + a⁻¹ = 3) :
  (a^2 + a⁻²) = 7 :=
by
  unfold_pow a
  sorry

end f_of_2_l82_82934


namespace shape_is_quadrilateral_l82_82597

theorem shape_is_quadrilateral (cost_per_side total_cost : ℕ) (h_cost : cost_per_side = 69) (h_total : total_cost = 276) :
  total_cost / cost_per_side = 4 :=
by
  rw [h_cost, h_total]
  sorry

end shape_is_quadrilateral_l82_82597


namespace Tammy_runs_10_laps_per_day_l82_82706

theorem Tammy_runs_10_laps_per_day
  (total_distance_per_week : ℕ)
  (track_length : ℕ)
  (days_per_week : ℕ)
  (h1 : total_distance_per_week = 3500)
  (h2 : track_length = 50)
  (h3 : days_per_week = 7) :
  (total_distance_per_week / track_length) / days_per_week = 10 := by
  sorry

end Tammy_runs_10_laps_per_day_l82_82706


namespace intersection_N_complement_M_l82_82186

def U : Set ℝ := Set.univ
def M : Set ℝ := {x : ℝ | x < -2 ∨ x > 2}
def CU_M : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 2}
def N : Set ℝ := {x : ℝ | (1 - x) / (x - 3) > 0}

theorem intersection_N_complement_M :
  N ∩ CU_M = {x : ℝ | 1 < x ∧ x ≤ 2} :=
sorry

end intersection_N_complement_M_l82_82186


namespace cuboid_layers_l82_82822

theorem cuboid_layers (V : ℕ) (n_blocks : ℕ) (volume_per_block : ℕ) (blocks_per_layer : ℕ)
  (hV : V = 252) (hvol : volume_per_block = 1) (hblocks : n_blocks = V / volume_per_block) (hlayer : blocks_per_layer = 36) :
  (n_blocks / blocks_per_layer) = 7 :=
by
  sorry

end cuboid_layers_l82_82822


namespace trajectory_and_line_eq_l82_82560

theorem trajectory_and_line_eq (M F : ℝ × ℝ) (d : ℝ) (x1 y1 x2 y2 : ℝ) :
  let dM_F := real.sqrt ((M.1 - F.1) ^ 2 + (M.2 - F.2) ^ 2)
  in (dM_F < 4 * abs (M.1 + 6)) →
  (M.1 = (y1 ^ 2) / 8 ∧ M.2 = y1 ∧ M.1 = (y2 ^ 2) / 8 ∧ M.2 = y2) →
  (y1 + y2) / 2 = -1 →
  ((4 * x1 + y1 = 15) ∧ (4 * x2 + y2 = 15)) :=
by
  intro h1 h2 h3
  have h_traj : M.1 = y1 ^ 2 / 8 ∧ M.2 = y1 := sorry
  have h_eq : M.1 = y2 ^ 2 / 8 ∧ M.2 = y2 := sorry
  have midpoint_eq : (y1 + y2) / 2 = -1 := sorry
  have line_eq : (4 * x1 + y1 = 15) ∧ (4 * x2 + y2 = 15) := sorry
  exact ⟨h_traj, h_eq, midpoint_eq, line_eq⟩

end trajectory_and_line_eq_l82_82560


namespace delete_one_column_unique_rows_l82_82248

theorem delete_one_column_unique_rows (n : ℕ) (table : Fin n → Fin n → ℕ) 
  (h_unique_rows : ∀ i j : Fin n, i ≠ j → ∃ k : Fin n, table i k ≠ table j k) :
  ∃ k : Fin n, ∀ i j : Fin n, i ≠ j → ∃ l : Fin (n-1), table i (if l.val < k.val then l else l.succ) ≠ table j (if l.val < k.val then l else l.succ) :=
begin 
  -- We will provide the proof here.
  sorry
end

end delete_one_column_unique_rows_l82_82248


namespace difference_of_squares_example_l82_82409

theorem difference_of_squares_example (a b : ℕ) (h1 : a = 123) (h2 : b = 23) : a^2 - b^2 = 14600 :=
by
  rw [h1, h2]
  sorry

end difference_of_squares_example_l82_82409


namespace sphere_volume_of_cubed_vertices_l82_82377

-- Define the conditions and the required properties
def cube_edge_length : ℝ := 2
def sphere_radius : ℝ := (cube_edge_length * Real.sqrt 3) / 2
def sphere_volume (r : ℝ) : ℝ := (4/3) * Real.pi * r^3

theorem sphere_volume_of_cubed_vertices :
  (∃ V, (∀ v ∈ V, ∃ r, r * r = v) ∧
       (cube_edge_length = 2) ∧ 
       sphere_volume (Real.sqrt 3) = 4 * Real.sqrt 3 * Real.pi) :=
by
    -- Let V be the set of vertices of a cube with edge length 2
    use { 
      -- we will define the coordinates of vertices here, but for simplicity, let's keep it abstract 
    }
    split
    -- Condition stating vertices should lie on the sphere's surface, omitted detailed set values
    · intro v hv
      use sphere_radius
      sorry -- proof showing vertices are on the sphere's surface
    split
    -- Given edge length
    · exact cube_edge_length
    -- Proving the volume calculation
    · exact (sphere_volume (Real.sqrt 3))
    sorry -- Verify detailed proof about volume computation

end sphere_volume_of_cubed_vertices_l82_82377


namespace transformed_plane_contains_point_l82_82659

theorem transformed_plane_contains_point 
  (k : ℚ) (A : ℝ × ℝ × ℝ) (a : ℝ × ℝ × ℝ → ℝ → Prop)
  (h_k : k = 5/6)
  (h_A : A = (1/3, 1, 1))
  (h_a : ∀ x y z, a (x, y, z) 6 = (3 * x - y + 5 * z - 6 = 0)) :
  a (1/3, 1, 1) 5 :=
by
  have h_t : a' (x, y, z) D = a (x, y, z) (k * D),
  {
    intros x y z D,
    rw [h_k, h_a, mul_comm],
  }
  rw [h_A, h_a],
  norm_num,
  sorry

end transformed_plane_contains_point_l82_82659


namespace A_2021_is_45_21_l82_82081

def pos_odd_nums : ℕ → ℕ
| n => 2 * n + 1

def sum_of_first_n (n : ℕ) : ℕ :=
(n * (n + 1)) / 2

def find_group (k : ℕ) : ℕ :=
if h : ∃ i, sum_of_first_n i ≥ k then
  Classical.some h
else
  0

def group_position (k : ℕ) : ℕ × ℕ :=
let i := find_group k in
(i, k - sum_of_first_n (i - 1))

theorem A_2021_is_45_21 :
  group_position 1011 = (45, 21) :=
sorry

end A_2021_is_45_21_l82_82081


namespace pages_in_novel_l82_82994

def stories_per_week := 3
def pages_per_story := 50
def total_weeks := 12
def reams := 3
def sheets_per_ream := 500
def pages_per_sheet := 2

theorem pages_in_novel :
  let total_pages_stories := stories_per_week * pages_per_story * total_weeks,
      total_pages_available := reams * sheets_per_ream * pages_per_sheet
  in total_pages_available - total_pages_stories = 1200 :=
by
  sorry

end pages_in_novel_l82_82994


namespace binomial_expectation_variance_l82_82166

open ProbabilityTheory

-- Conditions
def binomial_pmf (n : ℕ) (p : ℝ) : ProbabilityMassFunction ℕ :=
  ProbabilityMassFunction.ofMeasure (binomial n p)

-- Binomial random variable X
def X : ProbabilityMassFunction ℕ := binomial_pmf 10 0.6

-- Statement
theorem binomial_expectation_variance :
  let E_X := X.μf.sum (fun k => k * X p k) in -- Expectation
  let D_X := X.μf.sum (fun k => (k - E_X)^2 * X p k) in -- Variance
  E_X = 6 ∧ D_X = 2.4 :=
sorry

end binomial_expectation_variance_l82_82166


namespace find_x_value_l82_82869

open Real

theorem find_x_value (x : ℝ) (h1 : 0 < x) (h2 : x < 180) 
(h3 : tan (150 * π / 180 - x * π / 180) = (sin (150 * π / 180) - sin (x * π / 180)) / (cos (150 * π / 180) - cos (x * π / 180))) :
x = 120 :=
sorry

end find_x_value_l82_82869


namespace log_problem_l82_82902

theorem log_problem (x: ℝ) (h: log 3 (log 4 (log 2 x)) = 0) : x = 16 :=
by
sorry

end log_problem_l82_82902


namespace parabola_equation_fixed_point_AB_l82_82164

-- Definitions and conditions
def point_on_parabola (P : ℝ × ℝ) : Prop :=
  let F : ℝ × ℝ := (0, 1)
  let l := -2
  let distance_to_F (P : ℝ × ℝ) := (P.1 - F.1)^2 + (P.2 - F.2)^2
  let distance_to_l (P : ℝ × ℝ) := (P.2 - l)^2 
  distance_to_F P = distance_to_l P - 1

theorem parabola_equation (x y : ℝ) : point_on_parabola (x, y) → x^2 = 4 * y :=
  sorry

theorem fixed_point_AB (a : ℝ) :
  let E : ℝ × ℝ := (a, -2),
  let A := (a + sqrt (a^2 + 8), ((a + sqrt (a^2 + 8))^2) / 4),
  let B := (a - sqrt (a^2 + 8), ((a - sqrt (a^2 + 8))^2) / 4),
  let line_AB := λ (x y : ℝ), y - 2 = (a / 2) * x,
  ∀ x y, line_AB x y → (0, 2) =
  sorry

end parabola_equation_fixed_point_AB_l82_82164


namespace larger_triangle_perimeter_l82_82482

def is_similar (a b c : ℕ) (x y z : ℕ) : Prop :=
  x * c = z * a ∧
  x * c = z * b ∧
  y * c = z * a ∧
  y * c = z * c ∧
  a ≠ b ∧ c ≠ b

def is_isosceles (a b c : ℕ) : Prop :=
  a = b ∧ a ≠ c

theorem larger_triangle_perimeter (a b c x y z : ℕ) 
  (h1 : is_isosceles a b c) 
  (h2 : is_similar a b c x y z) 
  (h3 : c = 12) 
  (h4 : z = 36)
  (h5 : a = 7) 
  (h6 : b = 7) : 
  x + y + z = 78 :=
sorry

end larger_triangle_perimeter_l82_82482


namespace candy_bar_cost_l82_82635

open Nat Real

def cost_per_driveway : ℝ := 1.5
def driveways_shoveled : ℕ := 10
def fraction_spent : ℝ := 1/6
def number_lollipops : ℕ := 4
def lollipop_cost : ℝ := 0.25
def candy_bars_bought : ℕ := 2

theorem candy_bar_cost 
  (h1 : cost_per_driveway = 1.5)
  (h2 : driveways_shoveled = 10)
  (h3 : fraction_spent = 1/6)
  (h4 : number_lollipops = 4)
  (h5 : lollipop_cost = 0.25)
  (h6 : candy_bars_bought = 2) : 
  (cost_per_driveway * driveways_shoveled * fraction_spent - number_lollipops * lollipop_cost) / candy_bars_bought = 0.75 :=
sorry

end candy_bar_cost_l82_82635


namespace range_of_a_l82_82790

noncomputable def f : ℝ → ℝ := sorry

-- assumptions/conditions
axiom f_defined : ∀ x, -1 < x ∧ x < 1 → f x ≠ 0
axiom f_odd : ∀ x, f (-x) = -f x
axiom f_decreasing : ∀ x y, -1 < x ∧ x < y ∧ y < 1 → f y < f x
axiom f_positive : ∀ a, -1 < a ∧ a < 1 → f (1 - a) + f (1 - a^2) > 0

theorem range_of_a (a : ℝ) : 1 < a ∧ a < Real.sqrt 2 :=
by
  have h1 : f(1 - a) > f(a^2 - 1), from sorry,
  have h2 : 1 - a < a^2 - 1, from sorry,
  have h3 : -1 < a^2 - 1 < 1, from sorry,
  have h4 : -1 < 1 - a < 1, from sorry,
  sorry

end range_of_a_l82_82790


namespace sum_of_roots_eq_65_l82_82658

-- Definition of the function g
def g (x : ℝ) : ℝ := 12 * x + 5

-- The problem: the sum of all x that satisfy g⁻¹(x) = g((3 * x)⁻¹)
theorem sum_of_roots_eq_65 :
  (∑ x in {x : ℝ | g⁻¹(x) = g((3 * x)⁻¹)}, x) = 65 :=
sorry

end sum_of_roots_eq_65_l82_82658


namespace monica_tiles_l82_82674

-- Define the dimensions of the living room
def living_room_length : ℕ := 20
def living_room_width : ℕ := 15

-- Define the size of the border tiles and inner tiles
def border_tile_size : ℕ := 2
def inner_tile_size : ℕ := 3

-- Prove the number of tiles used is 44
theorem monica_tiles (border_tile_count inner_tile_count total_tiles : ℕ)
  (h_border : border_tile_count = ((2 * ((living_room_length - 4) / border_tile_size) + 2 * ((living_room_width - 4) / border_tile_size) - 4)))
  (h_inner : inner_tile_count = (176 / (inner_tile_size * inner_tile_size)))
  (h_total : total_tiles = border_tile_count + inner_tile_count) :
  total_tiles = 44 :=
by
  sorry

end monica_tiles_l82_82674


namespace paint_per_statue_calculation_l82_82510

theorem paint_per_statue_calculation (total_paint : ℚ) (num_statues : ℕ) (expected_paint_per_statue : ℚ) :
  total_paint = 7 / 8 → num_statues = 14 → expected_paint_per_statue = 7 / 112 → 
  total_paint / num_statues = expected_paint_per_statue :=
by
  intros htotal hnum_expected hequals
  rw [htotal, hnum_expected, hequals]
  -- Using the fact that:
  -- total_paint / num_statues = (7 / 8) / 14
  -- This can be rewritten as (7 / 8) * (1 / 14) = 7 / (8 * 14) = 7 / 112
  sorry

end paint_per_statue_calculation_l82_82510


namespace arithmetic_mean_calculation_l82_82853

theorem arithmetic_mean_calculation (x : ℝ) 
  (h : (x + 10 + 20 + 3 * x + 15 + 3 * x + 6) / 5 = 30) : 
  x = 14.142857 :=
by
  sorry

end arithmetic_mean_calculation_l82_82853


namespace distance_to_circle_center_is_sqrt3_l82_82260

-- Definitions of polar to Cartesian conversion, circle representation etc.
noncomputable def polar_to_cartesian (r θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

-- Definition for circle in Cartesian coordinates
def circle_center : ℝ × ℝ := (1, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Conditions as definitions
def point_polar : ℝ × ℝ := (2, Real.pi / 3)
def point_cartesian : ℝ × ℝ := polar_to_cartesian point_polar.1 point_polar.2

-- Prove that the distance between the point (2, π/3) and the center of circle ρ = 2 cos θ is √3
theorem distance_to_circle_center_is_sqrt3 : distance point_cartesian circle_center = Real.sqrt 3 :=
by sorry

end distance_to_circle_center_is_sqrt3_l82_82260


namespace frustum_surface_area_is_11pi_l82_82907

noncomputable def frustum_surface_area
  (r_top r_bottom h : ℝ) : ℝ :=
  let slant_height := height := sqrt (r_bottom - r_top) ^ 2 + h ^ 2
  π * r_top ^ 2 + π * r_bottom ^ 2 + π * (r_top + r_bottom) * slant_height

theorem frustum_surface_area_is_11pi :
  frustum_surface_area 1 2 (sqrt 3) = 11 * π :=
sorry

end frustum_surface_area_is_11pi_l82_82907


namespace election_including_past_officers_l82_82351

def election_problem (total_candidates past_officers: ℕ) (num_positions : ℕ) (num_includes: ℕ) : ℕ :=
  ∑ k in finset.range (num_includes + 1), (nat.choose past_officers k) * (nat.choose (total_candidates - past_officers) (num_positions - k))

theorem election_including_past_officers 
  (total_candidates : ℕ) (past_officers: ℕ) (num_positions : ℕ) (min_past_officers num_includes: ℕ) 
  (total_candidates = 20) 
  (past_officers = 5) 
  (num_positions = 6) 
  (min_past_officers = 1)
  (num_includes = 3) : 
  election_problem total_candidates past_officers num_positions num_includes = 33215 := 
sorry

end election_including_past_officers_l82_82351


namespace problem_conditions_l82_82578

def f (x : ℝ) : ℝ := 3^(-|x|) - 3^(|x|)

theorem problem_conditions :
  ¬(∀ x : ℝ, f(-x) = f(x)) ∧
  (∀ x : ℝ, f(x) ≤ f(0)) ∧
  (∀ x y : ℝ, (0 < x ∧ x < y) → f(y) < f(x)) ∧
  (f (-3) < f 2) :=
by
  -- Proof omitted
  sorry

end problem_conditions_l82_82578


namespace functional_equation_solution_l82_82128

noncomputable def f : ℝ → ℝ := sorry

theorem functional_equation_solution :
  (∀ x y : ℝ, f (x + y) * f (x - y) = (f x + f y) ^ 2 - 4 * x ^ 2 * f y + 2 * x * y) ↔ 
  (f = λ x, 0) ∨ (f = λ x, (7 / 8) * x ^ 2) :=
sorry

end functional_equation_solution_l82_82128


namespace tangent_lengths_identity_l82_82289

theorem tangent_lengths_identity
  (a b c BC AC AB : ℝ)
  (sqrt_a sqrt_b sqrt_c : ℝ)
  (h1 : sqrt_a^2 = a)
  (h2 : sqrt_b^2 = b)
  (h3 : sqrt_c^2 = c) :
  a * BC + c * AB - b * AC = BC * AC * AB :=
sorry

end tangent_lengths_identity_l82_82289


namespace cosine_angle_between_vectors_l82_82131

/-- The cosine of the angle between the vectors 𝐀𝐁 and 𝐀𝐂 is (1/√2) given the points A(-1, -2, 1), B(-4, -2, 5), and C(-8, -2, 2) --/
theorem cosine_angle_between_vectors :
  let A := (-1, -2, 1 : ℝ × ℝ × ℝ)
  let B := (-4, -2, 5 : ℝ × ℝ × ℝ)
  let C := (-8, -2, 2 : ℝ × ℝ × ℝ)
  let AB := (B.1 - A.1, B.2 - A.2, B.3 - A.3)
  let AC := (C.1 - A.1, C.2 - A.2, C.3 - A.3)
  let dot_product := AB.1 * AC.1 + AB.2 * AC.2 + AB.3 * AC.3
  let magnitude_AB := Real.sqrt (AB.1^2 + AB.2^2 + AB.3^2)
  let magnitude_AC := Real.sqrt (AC.1^2 + AC.2^2 + AC.3^2)
  cos_angle (AB : ℝ × ℝ × ℝ) (AC : ℝ × ℝ × ℝ) := 
    dot_product / (magnitude_AB * magnitude_AC) 
  in cos_angle = 1 / Real.sqrt 2 :=
by
  sorry

end cosine_angle_between_vectors_l82_82131


namespace area_of_U_l82_82823

noncomputable def equilateral_triangle_area (side_length : ℝ) : ℝ :=
  (side_length^2 * sqrt 3) / 4

noncomputable def transformed_area (original_area : ℝ) (expansion_factor : ℝ) : ℝ :=
  original_area * expansion_factor

theorem area_of_U :
  let side_length := sqrt 3
  let equilateral_triangle_area := equilateral_triangle_area side_length
  let expansion_factor := 4 in
  transformed_area equilateral_triangle_area expansion_factor = 3 * sqrt 3 :=
by
  sorry

end area_of_U_l82_82823


namespace factor_expression_l82_82125

theorem factor_expression (x : ℝ) : (45 * x^3 - 135 * x^7) = 45 * x^3 * (1 - 3 * x^4) :=
by
  sorry

end factor_expression_l82_82125


namespace find_bottom_left_number_l82_82275

-- Definition of a hypothesis representing the 3x3 grid with positions
variables {x a b c y : ℤ}
def grid := (λ (z : ℤ), 2 + x + a + b = 4 + x + b + c 
                           ∧ a + b + z + y = b + c + y + 3 
                           ∧ a = c + 2 
                           ∧ z = 1)

-- The main theorem stating that the required number in the bottom-left cell is 1
theorem find_bottom_left_number (x a b c y : ℤ) (h : grid 1): 
  ∃ z : ℤ, z = 1 :=
sorry -- proof not required

end find_bottom_left_number_l82_82275


namespace minimum_cuts_l82_82759

theorem minimum_cuts (n : Nat) : n >= 50 :=
by
  sorry

end minimum_cuts_l82_82759


namespace can_form_square_from_rectangles_l82_82988

-- Define the side length of the square and the total perimeter of rectangles
def side_length_square : ℕ := 16
def total_perimeter_rectangles : ℕ := 100

-- Define the property we want to prove
theorem can_form_square_from_rectangles
  (s : ℕ) (t : ℕ) (h_side : s = side_length_square) (h_perim : t = total_perimeter_rectangles) :
  ∃ rects : list (ℕ × ℕ),
    (rects.sum (λ r, 2 * (r.fst + r.snd)) = t) ∧ 
    can_form_square_of_side_length s rects := sorry


end can_form_square_from_rectangles_l82_82988


namespace union_sets_l82_82939

open Set

theorem union_sets (x y : ℝ):
  let A := {x, y}
  let B := {1, Real.log (x + 2) / Real.log 3}
  (1 + x ∈ A) ∧ (1 + x ∈ B)  → 
  (A ∪ B = {0, 1, Real.log 2 / Real.log 3} ∨ A ∪ B = {-1, 0, 1}) :=
sorry

end union_sets_l82_82939


namespace matrix_eq_l82_82890

open Matrix

variable {α : Type*} [Field α]
variables {N : Matrix (Fin 2) (Fin 2) α} (w : Fin 2 → α)

theorem matrix_eq : (∀ w, N.vecMul w = (-3 : α) • w) →
  N = of_fun ![![(-3 : α), 0], ![0, (-3 : α)]] :=
by
  intro h
  sorry

end matrix_eq_l82_82890


namespace expected_points_for_9_days_l82_82969

theorem expected_points_for_9_days (p : ℝ) (h : p = 2 / 3) :
  (E : ℝ) 
  (Z : ℝ) 
  (HZ1 : E(Z) = 8 * (11 / 27) + 10 * (16 / 27)) 
  (HZ2 : E = 9 * (E(Z))):
  E = 248 / 3 := 
sorry

end expected_points_for_9_days_l82_82969


namespace min_value_a_l82_82924

noncomputable def equation_has_real_solutions (a : ℝ) : Prop :=
  ∃ x1 x2 : ℝ, 9 * x1 - (4 + a) * 3 * x1 + 4 = 0 ∧ 9 * x2 - (4 + a) * 3 * x2 + 4 = 0

theorem min_value_a : ∀ a : ℝ, 
  equation_has_real_solutions a → 
  a ≥ 2 :=
sorry

end min_value_a_l82_82924


namespace nine_otimes_three_l82_82863

def otimes (a b : ℤ) : ℤ := a + (4 * a) / (3 * b)

theorem nine_otimes_three : otimes 9 3 = 13 := by
  sorry

end nine_otimes_three_l82_82863


namespace range_magnitude_l82_82588

def a (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ)
noncomputable def b : ℝ × ℝ := (Real.sqrt 3, -1)

theorem range_magnitude (θ : ℝ) : 
  0 ≤ |(2 : ℝ) • a θ - b| ∧ |(2 : ℝ) • a θ - b| ≤ 4 :=
by
  sorry

end range_magnitude_l82_82588


namespace area_of_given_triangle_l82_82865

-- Definition of a triangle
structure Triangle where
  a b c : ℝ
  -- Side lengths must be positive
  h_a : a > 0
  h_b : b > 0
  h_c : c > 0

-- Given triangle with specific side lengths
def given_triangle : Triangle :=
  { a := 15, b := 36, c := 39,
    h_a := by norm_num,
    h_b := by norm_num,
    h_c := by norm_num }

-- Calculation of the area of the given triangle
noncomputable def area (t : Triangle) : ℝ :=
  if h : t.a^2 + t.b^2 = t.c^2 then
    (1 / 2) * t.a * t.b
  else if h : t.b^2 + t.c^2 = t.a^2 then
    (1 / 2) * t.b * t.c
  else if h : t.c^2 + t.a^2 = t.b^2 then
    (1 / 2) * t.c * t.a
  else
    -- General formula can be used here if needed, e.g., Heron's formula
    sorry

-- Main statement to be proven
theorem area_of_given_triangle : area given_triangle = 270 := 
sorry

end area_of_given_triangle_l82_82865


namespace sum_c_n_l82_82184

noncomputable def a_n (n : ℕ) : ℕ := 2 * n - 1

noncomputable def b_n (n : ℕ) : ℕ := 3 ^ (n - 1)

noncomputable def c_n (n : ℕ) : ℕ := a_n n * b_n n

theorem sum_c_n (n : ℕ) : (finset.range n).sum c_n = (n - 1) * 3^n + 1 :=
sorry

end sum_c_n_l82_82184


namespace nearest_integer_to_x_plus_2y_l82_82958

theorem nearest_integer_to_x_plus_2y
  (x y : ℝ)
  (hx : x ≠ 0)
  (hy : y ≠ 0)
  (h1 : |x| + 2 * y = 6)
  (h2 : |x| * y + x^3 = 2) :
  Int.floor (x + 2 * y + 0.5) = 6 :=
by sorry

end nearest_integer_to_x_plus_2y_l82_82958


namespace find_a_value_l82_82191

theorem find_a_value (a x : ℝ) (h1 : 6 * (x + 8) = 18 * x) (h2 : 6 * x - 2 * (a - x) = 2 * a + x) : a = 7 :=
by
  sorry

end find_a_value_l82_82191


namespace person_B_arrives_first_l82_82330

theorem person_B_arrives_first
  (a b : ℝ) (s : ℝ) (ha : a > 0) (hb : b > 0) (hs : s > 0) :
  let T := (s / 2) * (1 / a + 1 / b),
      t := s / (2 * (a + b)) in
  T > 2 * t :=
by
  sorry

end person_B_arrives_first_l82_82330


namespace exists_D_on_hypotenuse_l82_82627

open Function

variables {A B C D : Type*} [MetricSpace A] [MetricSpace B] [MetricSpace C] 
          [MetricSpace D]

def is_right_triangle (A B C : A) : Prop :=
  -- definition of right triangle with hypotenuse AB
  sorry

def is_hypotenuse (AB : A) : Prop :=
  -- definition of hypotenuse AB 
  sorry

def eq_incircles_radii (D C A B : A) : Prop :=
  -- definition of equal radii for inscribed circles of triangles DCA and DCB
  sorry

theorem exists_D_on_hypotenuse (A B C D : A)
  (h1 : is_right_triangle A B C)
  (h2 : is_hypotenuse AB)
  (h3 : eq_incircles_radii D C A B)
  : ∃ D ∈ AB, ∀ D, eq_incircles_radii D C A B :=
begin
  sorry,
end

end exists_D_on_hypotenuse_l82_82627


namespace least_boxes_l82_82382
-- Definitions and conditions
def isPerfectCube (n : ℕ) : Prop := ∃ (k : ℕ), k^3 = n

def isFactor (a b : ℕ) : Prop := ∃ k, a * k = b

def numBoxes (N boxSize : ℕ) : ℕ := N / boxSize

-- Specific conditions for our problem
theorem least_boxes (N : ℕ) (boxSize : ℕ) 
  (h1 : N ≠ 0) 
  (h2 : isPerfectCube N)
  (h3 : isFactor boxSize N)
  (h4 : boxSize = 45): 
  numBoxes N boxSize = 75 :=
by
  sorry

end least_boxes_l82_82382


namespace ruler_perpendicular_line_exists_l82_82713

theorem ruler_perpendicular_line_exists (l : Line) (ground : Plane) :
  ∀ (place_ruler: Line), ∃ (line_ground: Line), line_ground ⊥ place_ruler :=
sorry

end ruler_perpendicular_line_exists_l82_82713


namespace probability_h_neg_and_pq_even_l82_82781

-- Definitions for the conditions
def p_is_positive_integer : Prop := ∃ p : ℕ, 1 ≤ p ∧ p ≤ 10
def quadratic_h (p : ℕ) : ℤ := p^2 - 13 * p + 40
def quadratic_q (h : ℤ) : ℤ := h^2 + 7 * h + 12

-- Probability calculation for the given conditions
theorem probability_h_neg_and_pq_even : (∑ p in finset.filter (λ p, let h := quadratic_h p in
    (h < 0) ∧ ((p + quadratic_q h) % 2 = 0)) (finset.range 11), 1) * (10 : ℚ)⁻¹ = 1 / 10 := sorry

end probability_h_neg_and_pq_even_l82_82781


namespace sum_of_roots_of_equation_l82_82102

theorem sum_of_roots_of_equation :
  let f := (3 * (x: ℝ) + 4) * (x - 5) + (3 * x + 4) * (x - 7)
  ∀ x : ℝ, f = 0 → ∑ (roots : ℝ) in {x | f x = 0}, x = -2 :=
by {
  let f := (3 * (x: ℝ) + 4) * (x - 5) + (3 * x + 4) * (x - 7),
  sorry
}

end sum_of_roots_of_equation_l82_82102


namespace area_triangle_apb_l82_82468

open EuclideanGeometry

-- Definitions of given conditions
def square_side_length := 8
def point_P_in_square (S : square) (P : point) := S.inside P

/-- Definition of equal segments -/
def equal_segments (A B C P : point) : Prop :=
  dist P A = dist P B ∧ dist P A = dist P C

-- Definition of perpendicularity
def perpendicular (P C : point) (FD : line) : Prop :=
  is_perpendicular (line_through P C) FD

-- Main theorem statement
theorem area_triangle_apb (S : square)
  (A B C D P : point)
  (H_square : S.length = square_side_length)
  (H_in_square : point_P_in_square S P)
  (H_eq_segs : equal_segments A B C P)
  (H_perpendicular : perpendicular P C (line_through F D)) :
  area (triangle A P B) = 12 := sorry

end area_triangle_apb_l82_82468


namespace positive_difference_between_solutions_l82_82137

theorem positive_difference_between_solutions :
  (∃ (x₁ x₂ : ℝ), (sqrt3 (9 - x₁^2 / 4) = -3) ∧ (sqrt3 (9 - x₂^2 / 4) = -3)
  ∧ (x₁ - x₂ = 24) ∨ (x₂ - x₁ = 24)) :=
by
  sorry

end positive_difference_between_solutions_l82_82137


namespace ratio_wrong_to_correct_l82_82470

theorem ratio_wrong_to_correct (total_sums correct_sums : ℕ) 
  (h1 : total_sums = 36) (h2 : correct_sums = 12) : 
  (total_sums - correct_sums) / correct_sums = 2 :=
by {
  -- Proof will go here
  sorry
}

end ratio_wrong_to_correct_l82_82470


namespace parenthesis_removal_correctness_l82_82841

theorem parenthesis_removal_correctness (x y z : ℝ) : 
  (x^2 - (x - y + 2 * z) ≠ x^2 - x + y - 2 * z) ∧
  (x - (-2 * x + 3 * y - 1) ≠ x + 2 * x - 3 * y + 1) ∧
  (3 * x + 2 * (x - 2 * y + 1) ≠ 3 * x + 2 * x - 4 * y + 2) ∧
  (-(x - 2) - 2 * (x^2 + 2) = -x + 2 - 2 * x^2 - 4) :=
by
  sorry

end parenthesis_removal_correctness_l82_82841


namespace dvd_player_movie_ratio_l82_82354

theorem dvd_player_movie_ratio (M D : ℝ) (h1 : D = M + 63) (h2 : D = 81) : D / M = 4.5 :=
by
  sorry

end dvd_player_movie_ratio_l82_82354


namespace trig_identity_l82_82913

variable (α : ℝ)

theorem trig_identity (h : Real.sin (α - 70 * Real.pi / 180) = α) : 
  Real.cos (α + 20 * Real.pi / 180) = -α := by
  sorry

end trig_identity_l82_82913


namespace find_p_l82_82416

noncomputable def vector_p (t : ℚ) : ℚ × ℚ :=
  let x := -7 * t + 5
  let y := 2 * t + 2
  (x, y)

theorem find_p :
  ∃ t : ℚ, 
    vector_p t = (48 / 53, 168 / 53) ∧
    (let (x, y) := vector_p t in x * -7 + y * 2 = 0) :=
by
  sorry

end find_p_l82_82416


namespace symmetric_difference_probability_tends_to_zero_l82_82651

noncomputable def varlimsup {α : Type*} (f : ℕ → Set α) : Set α := 
  {x | ∀ n, ∃ m > n, x ∈ f m}

noncomputable def underlim {α : Type*} (f : ℕ → Set α) : Set α := 
  {x | ∃ n, ∀ m > n, x ∈ f m}

noncomputable def symmetric_difference {α : Type*} (A B : Set α) : Set α := 
  (A ∪ B) \ (A ∩ B)

theorem symmetric_difference_probability_tends_to_zero {α : Type*} [MeasureTheory.ProbabilityMeasure α]
  (A : Set α) (A_n : ℕ → Set α)
  (h_converge: A = varlimsup A_n ∧ A = underlim A_n) :
  tendsto (λ n, MeasureTheory.Measure.ofSet (symmetric_difference A A_n)) atTop (𝓝 0) := 
sorry

end symmetric_difference_probability_tends_to_zero_l82_82651


namespace max_m_for_ineq_l82_82940

open Real

theorem max_m_for_ineq (m : ℝ) (hm : 0 < m) :
  (∀ (x1 x2 : ℝ), 0 < x1 ∧ x1 < m ∧ 0 < x2 ∧ x2 < m ∧ x1 < x2 → x1^x2 < x2^x1) →
  m ≤ exp(1) :=
by
  sorry

end max_m_for_ineq_l82_82940


namespace Carol_optimal_choice_l82_82840

-- Definitions
def Alice_choice := UniformDist (set.Icc 0 1)
def Bob_choice := UniformDist (set.Icc (1 / 3) (2 / 3))
def Carol_winning_condition (a b c : ℝ) := (a < c ∧ c < b) ∨ (a > c ∧ c > b)

-- Main statement
theorem Carol_optimal_choice : ∀ a b : ℝ, 
    (a ∈ set.Icc 0 1) →
    (b ∈ set.Icc (1 / 3) (2 / 3)) →
    (c ∈ set.Icc 0 1) →
    (∀ (c : ℝ), c = 1 / 2 → Carol_winning_condition a b c) :=
sorry

end Carol_optimal_choice_l82_82840


namespace equal_segments_l82_82295

-- Given a triangle ABC and D as the foot of the bisector from B
variables (A B C D E F : Point) (ABC : Triangle A B C) (Dfoot : BisectorFoot B A C D) 

-- Given that the circumcircles of triangles ABD and BCD intersect sides AB and BC at E and F respectively
variables (circABD : Circumcircle A B D) (circBCD : Circumcircle B C D)
variables (intersectAB : Intersect circABD A B E) (intersectBC : Intersect circBCD B C F)

-- The theorem to prove that AE = CF
theorem equal_segments : AE = CF :=
by
  sorry

end equal_segments_l82_82295


namespace train_crosses_in_expected_time_l82_82774

-- Problem Definition: Time taken for the train to cross the platform
noncomputable def time_to_cross_platform (train_length : ℕ) (platform_length : ℕ) (speed_kmph : ℕ) : ℝ :=
  (train_length + platform_length : ℝ) / (speed_kmph * (1000 / 3600))

-- Condition Definitions
def train_length := 250 -- meters
def platform_length := 520 -- meters
def speed_kmph := 55 -- kilometers per hour

-- Expected time to cross the platform according to the given conditions
def expected_time := 50.39 -- seconds

-- Lean statement to prove the expected time approximately equals to the calculated time
theorem train_crosses_in_expected_time :
  abs (time_to_cross_platform train_length platform_length speed_kmph - expected_time) < 0.01 :=
by
  sorry

end train_crosses_in_expected_time_l82_82774


namespace insert_digits_identical_l82_82387

theorem insert_digits_identical (A B : List Nat) (hA : A.length = 2007) (hB : B.length = 2007)
  (hErase : ∃ (C : List Nat) (erase7A : List Nat → List Nat) (erase7B : List Nat → List Nat),
    (erase7A A = C) ∧ (erase7B B = C) ∧ (C.length = 2000)) :
  ∃ (D : List Nat) (insert7A : List Nat → List Nat) (insert7B : List Nat → List Nat),
    (insert7A A = D) ∧ (insert7B B = D) ∧ (D.length = 2014) := sorry

end insert_digits_identical_l82_82387


namespace zeros_in_expansion_of_large_number_l82_82952

noncomputable def count_zeros (n : ℕ) : ℕ :=
  (n.to_string.filter (λ c, c = '0')).length

theorem zeros_in_expansion_of_large_number :
  count_zeros ((999999999999997 : ℕ)^2) = 29 :=
by {
  -- We'll provide the actual proof here
  sorry
}

end zeros_in_expansion_of_large_number_l82_82952


namespace smallest_positive_period_area_of_triangle_l82_82571

noncomputable def f (x : ℝ) : ℝ := 2 * sin x * cos x - 2 * (sin x)^2

theorem smallest_positive_period (T : ℝ) :
  (∀ x : ℝ, f (x + T) = f x) ∧ T > 0 → T = Real.pi := sorry

theorem area_of_triangle (A : ℝ) (c b : ℝ) :
  f A = 0 → A ∈ (0, Real.pi) → c = 1 → b = Real.sqrt 2 → 
  let area := (1 / 2) * b * c * sin A in 
  area = 1 / 2 := sorry

end smallest_positive_period_area_of_triangle_l82_82571


namespace find_point_M_l82_82965

-- Define the curve
def curve (x : ℝ) : ℝ := 2 * x^2 + 1

-- Define the slope of the tangent line at a point on the curve
def tangent_slope (x : ℝ) : ℝ := (curve' x)

-- Problem statement
theorem find_point_M :
  ∃ (M : ℝ × ℝ), M = (-1, 3) ∧
  (∃ x : ℝ, tangent_slope x = -4 ∧ curve x = M.2 ∧ x = M.1) :=
by
  sorry

end find_point_M_l82_82965


namespace sum_of_roots_eq_l82_82100

noncomputable def polynomial_sum_of_roots : ℚ :=
  let p := (λ x : ℚ, (3 * x + 4) * (x - 5) + (3 * x + 4) * (x - 7))
  let roots := [(-4) / 3, 6]
  roots.sum

theorem sum_of_roots_eq :
  let p := (λ x : ℚ, (3 * x + 4) * (x - 5) + (3 * x + 4) * (x - 7))
  ∑ root in [(-4) / 3, 6], root = 14 / 3 :=
by
  sorry

end sum_of_roots_eq_l82_82100


namespace time_after_122h_39m_44s_is_2h39m44s_l82_82989

def hours_to_12_hour_format (hours : ℕ) : ℕ :=
  hours % 12

def add_time_to_midnight (hours minutes seconds : ℕ) : (ℕ × ℕ × ℕ) :=
  let added_seconds := seconds
  let added_minutes := minutes
  let added_hours := hours_to_12_hour_format (0 + hours)
  (added_hours, added_minutes, added_seconds)

def sum_of_time_digits (time : (ℕ × ℕ × ℕ)) : ℕ :=
  time.fst + time.snd + time.snd.snd

theorem time_after_122h_39m_44s_is_2h39m44s :
  sum_of_time_digits (add_time_to_midnight 122 39 44) = 85 := by
  sorry

end time_after_122h_39m_44s_is_2h39m44s_l82_82989


namespace complex_square_expression_l82_82088

theorem complex_square_expression : 
  let z := (1 - complex.i) / real.sqrt 2
  let a := 0
  let b := -1
  (z ^ 2 = complex.of_real a + complex.i * b) → 
  (a^2 - b^2 = -1) :=
by
  intros z a b h
  have h1 : z = (1 - complex.i) / real.sqrt 2 := rfl
  have h2 : a = 0 := rfl
  have h3 : b = -1 := rfl
  sorry

end complex_square_expression_l82_82088


namespace max_abs_x_2y_l82_82539

theorem max_abs_x_2y (x y : ℝ) (h : 2 * x^2 + 3 * y^2 ≤ 12) : 
  |x + 2 * y| ≤ sqrt 22 :=
sorry

end max_abs_x_2y_l82_82539


namespace vectors_parallel_eq_l82_82208

theorem vectors_parallel_eq (x : ℝ) (a b : ℝ) 
  (ha : a = (-3, 2)) (hb : b = (x, -4)) 
  (h_parallel : ∃ λ : ℝ, λ ≠ 0 ∧ b = λ • a) : x = 6 :=
sorry

end vectors_parallel_eq_l82_82208


namespace Faye_can_still_make_8_bouquets_l82_82516

theorem Faye_can_still_make_8_bouquets (total_flowers : ℕ) (wilted_flowers : ℕ) (flowers_per_bouquet : ℕ) 
(h1 : total_flowers = 88) 
(h2 : wilted_flowers = 48) 
(h3 : flowers_per_bouquet = 5) : 
(total_flowers - wilted_flowers) / flowers_per_bouquet = 8 := 
by
  sorry

end Faye_can_still_make_8_bouquets_l82_82516


namespace find_m_l82_82900

theorem find_m (m : ℝ) : 
  let A := {-1, 1, 3}
  let B := {1, m}
  let A_cap_B := A ∩ B
  A_cap_B = {1, 3} → 
  m = 3 := 
by 
  sorry

end find_m_l82_82900


namespace smallest_value_of_a_l82_82733

noncomputable def polynomial : Polynomial ℝ := Polynomial.C 1806 - Polynomial.C b * Polynomial.X + Polynomial.C a * (Polynomial.X ^ 2) - Polynomial.X ^ 3

theorem smallest_value_of_a (a b : ℝ) (r1 r2 r3 : ℝ) 
  (h_roots : ∀ x, Polynomial.eval x polynomial = 0 ↔ x = r1 ∨ x = r2 ∨ x = r3)
  (h_factors : 1806 = r1 * r2 * r3)
  (h_pos : r1 > 0 ∧ r2 > 0 ∧ r3 > 0)
  (h_int : r1 ∈ ℤ ∧ r2 ∈ ℤ ∧ r3 ∈ ℤ) :
  a = r1 + r2 + r3 → a = 56 :=
by 
  sorry

end smallest_value_of_a_l82_82733


namespace solve_a_b_intervals_of_monotonicity_l82_82359

-- Define the function f(x)
def f (x : ℝ) (a b c : ℝ) : ℝ := x^3 + a * x^2 + b * x + c

-- Define the derivative of f(x)
def f' (x : ℝ) (a b : ℝ) : ℝ := 3 * x^2 + 2 * a * x + b

-- Problem statements in Lean 4
theorem solve_a_b
  (h1 : f' (-2/3 : ℝ) a b = 0)
  (h2 : f' (1 : ℝ) a b = 0) :
  a = -1/2 ∧ b = -2 :=
sorry

theorem intervals_of_monotonicity
  (a b c : ℝ)
  (h_a : a = -1/2)
  (h_b : b = -2)
  (h_c : c ∈ set.univ) :
  (∀ x, -2/3 < x ∧ x < 1 → f' x a b < 0) ∧
  (∀ x, x < -2/3 ∨ x > 1 → f' x a b > 0) :=
sorry

end solve_a_b_intervals_of_monotonicity_l82_82359


namespace find_angle_A_find_range_sinA_sinB_sinC_l82_82555

variables (a b c : ℝ)
variables (A B C : ℝ)
variables (AB AC : ℝ)

-- Given conditions:
-- Sides opposite to angles in a triangle
-- Dot product condition
-- Trigonometric condition

axiom sides_opposite_to_angles_triangle : ∀ (a b c A B C : ℝ), a / sin A = b / sin B = c / sin C ∧ a = b ∧ a = c

axiom given_dot_product_condition : AB * AC = 4
axiom given_trigonometric_condition : a * c * sin B = 8 * sin A

-- Proof to find value of angle A
theorem find_angle_A : A = π/3 := sorry

-- Proof to find range of sin A * sin B * sin C
theorem find_range_sinA_sinB_sinC : 0 < sin A * sin B * sin C ∧ sin A * sin B * sin C ≤ (3 * sqrt 3) / 8 := sorry

end find_angle_A_find_range_sinA_sinB_sinC_l82_82555


namespace new_average_mark_of_remaining_students_l82_82711

theorem new_average_mark_of_remaining_students:
  ∀ (n A excluded_students average_excluded average_remaining : ℕ) (H : n = 9) (H_A : A = 60) (H_excluded : excluded_students = 5) 
    (H_average_excluded : average_excluded = 44) (H_average_remaining : average_remaining = ((n * A - excluded_students * average_excluded) / (n - excluded_students))),
  average_remaining = 80 := by
  intros n A excluded_students average_excluded average_remaining H H_A H_excluded H_average_excluded H_average_remaining
  rw [H, H_A, H_excluded, H_average_excluded]
  norm_num at H_average_remaining
  exact H_average_remaining

end new_average_mark_of_remaining_students_l82_82711


namespace can_form_palindrome_l82_82419

def is_palindrome (s : String) : Prop :=
  s = s.reverse

def shuffled_string := "колнёиошакапеллашёланп"
def palindrome_string := "лёша на полке клопа нашёл"

theorem can_form_palindrome : ∃ s, s = shuffled_string ∧ is_palindrome palindrome_string :=
by 
  sorry

end can_form_palindrome_l82_82419


namespace train_speed_l82_82832

theorem train_speed (distance_km : ℝ) (time_min : ℝ) (time_hrs : ℝ):
  distance_km = 8 → time_min = 6 → time_hrs = time_min / 60 → 
  (distance_km / time_hrs = 80) :=
by
  intros h1 h2 h3
  rw [←h1, ←h2, ←h3]
  sorry

end train_speed_l82_82832


namespace standard_equation_of_ellipse_equation_of_line_PF1_length_of_PN_constant_l82_82189

noncomputable def ellipse := (x y : ℝ) (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) :=
  x^2 / a^2 + y^2 / b^2 = 1

variable (a b c : ℝ)
variables (h1 : 0 < a) (h2 : 0 < b) (hab : a > b)

theorem standard_equation_of_ellipse 
  (h_a : 2 * a = 4)
  (h_e : (1 / 2) = c / a) 
  (h_bc : b^2 = a^2 - c^2):
  ellipse 2 (sqrt 3) a_pos b_pos =
  (x / 2)^2 + (y / sqrt 3)^2 = 1 := sorry

variable (x1 y1 y2 : ℝ)
variable (Q : ellipse x1 y2 a b h1 hab)
variable (A : ellipse 2 0 a b h1 hab)
variables (AP AQ : ℝ)
variable (HAPQ : AP * AQ = 3)

theorem equation_of_line_PF1 
  (h_line_eqk : y1 = (sqrt 15 / 15) * (x1 + 1)) :
  (y = sqrt 15 / 15 * (x + 1) ∨ y = -sqrt 15 / 15 * (x + 1)) := sorry

variable (M : ℝ) (h_M : M = x1 / 4)
variable (PN : ℝ) (h_Plength : PN = 3 / 2)

theorem length_of_PN_constant : 
  (∀ (x1 : ℝ), PN = 3 / 2) := sorry

end standard_equation_of_ellipse_equation_of_line_PF1_length_of_PN_constant_l82_82189


namespace find_angle_A_range_sine_product_l82_82557

variable (a b c : ℝ)
variable (A B C : ℝ)
variable (AB AC : ℝ)

-- Conditions given
axiom sides : a > 0 ∧ b > 0 ∧ c > 0
axiom angles : 0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π
axiom dot_product : AB = 4
axiom product_sines : a * c * sin B = 8 * sin A

-- Proof statements
theorem find_angle_A : A = π / 3 := sorry
theorem range_sine_product : ∀ (x : ℝ), x = sin A * sin B * sin C -> 0 < x ∧ x ≤ 3 * sqrt 3 / 8 := sorry

end find_angle_A_range_sine_product_l82_82557


namespace frame_width_proof_l82_82358

noncomputable section

-- Define the given conditions
def perimeter_square_opening := 60 -- cm
def perimeter_entire_frame := 180 -- cm

-- Define what we need to prove: the width of the frame
def width_of_frame : ℕ := 5 -- cm

-- Define a function to calculate the side length of a square
def side_length_of_square (perimeter : ℕ) : ℕ :=
  perimeter / 4

-- Define the side length of the square opening
def side_length_opening := side_length_of_square perimeter_square_opening

-- Use the given conditions to calculate the frame's width
-- Given formulas in the solution steps:
--  2 * (3 * side_length + 4 * d) + 2 * (side_length + 2 * d) = perimeter_entire_frame
theorem frame_width_proof (d : ℕ) (perim_square perim_frame : ℕ) :
  perim_square = perimeter_square_opening →
  perim_frame = perimeter_entire_frame →
  2 * (3 * side_length_of_square perim_square + 4 * d) 
  + 2 * (side_length_of_square perim_square + 2 * d) 
  = perim_frame →
  d = width_of_frame := 
by 
  intros h1 h2 h3
  -- The proof will go here
  sorry

end frame_width_proof_l82_82358


namespace number_in_center_is_five_l82_82476

theorem number_in_center_is_five
  (grid : Fin 3 → Fin 3 → ℕ)
  (distinct : ∀ i j, grid i j ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9})
  (consecutive_adjacent : ∀ i1 j1 i2 j2, 
    abs (grid i1 j1 - grid i2 j2) = 1 → 
    (abs (i1 - i2) + abs (j1 - j2)) = 1)
  (corners_sum_20 : grid 0 0 + grid 0 2 + grid 2 0 + grid 2 2 = 20) :
  grid 1 1 = 5 :=
by
  sorry

end number_in_center_is_five_l82_82476


namespace max_k_constant_for_right_triangle_l82_82304

theorem max_k_constant_for_right_triangle (a b c : ℝ) (h : a^2 + b^2 = c^2) (h1 : a ≤ b) (h2 : b < c) :
  a^2 * (b + c) + b^2 * (c + a) + c^2 * (a + b) ≥ (2 + 3*Real.sqrt 2) * a * b * c :=
by 
  sorry

end max_k_constant_for_right_triangle_l82_82304


namespace f_monotone_seq_inequalities_seq_ratio_inequality_l82_82935

open Real 

-- Definitions
def f (x : ℝ) : ℝ := x^3 - x^2 + x / 2 + 1 / 4

axiom x0_in_interval : ∃ x0 : ℝ, 0 < x0 ∧ x0 < 1 / 2 ∧ f x0 = x0

-- Question 1: Prove f is monotonically increasing
theorem f_monotone : ∀ x : ℝ, f'(x) > 0 :=
sorry

-- Sequences definitions
def x_seq : ℕ → ℝ 
| 0 := 0
| (n + 1) := f (x_seq n)

def y_seq : ℕ → ℝ 
| 0 := 1 / 2
| (n + 1) := f (y_seq n)

-- Question 2: Prove the inequality chain for sequences
theorem seq_inequalities : ∀ n : ℕ, x_seq n < x_seq (n + 1) ∧ x_seq (n + 1) < x0 ∧ y_seq (n + 1) < y_seq n ∧ x0 < y_seq (n + 1) :=
sorry

-- Question 3: Prove the ratio inequality
theorem seq_ratio_inequality : ∀ n : ℕ, (y_seq (n + 1) - x_seq (n + 1)) / (y_seq n - x_seq n) < 1 / 2 :=
sorry

end f_monotone_seq_inequalities_seq_ratio_inequality_l82_82935


namespace cartons_per_stack_l82_82799

-- Declare the variables and conditions
def total_cartons := 799
def stacks := 133

-- State the theorem
theorem cartons_per_stack : (total_cartons / stacks) = 6 := by
  sorry

end cartons_per_stack_l82_82799


namespace Kavi_sold_on_Tuesday_l82_82995

theorem Kavi_sold_on_Tuesday :
  ∀ (initial_stock : ℕ) (sales_mon : ℕ) (sales_wed : ℕ) (sales_thu : ℕ) (sales_fri : ℕ) (unsold_percent : ℕ),
  initial_stock = 600 →
  sales_mon = 25 →
  sales_wed = 100 →
  sales_thu = 110 →
  sales_fri = 145 →
  unsold_percent = 25 →
  let unsold := initial_stock * unsold_percent / 100 in
  let total_sold := initial_stock - unsold in
  let sales_excluding_tue := sales_mon + sales_wed + sales_thu + sales_fri in
  let sales_tue := total_sold - sales_excluding_tue in
  sales_tue = 70 :=
by
  intros _ _ _ _ _ _ h₁ h₂ h₃ h₄ h₅ h₆
  let unsold := initial_stock * unsold_percent / 100
  let total_sold := initial_stock - unsold
  let sales_excluding_tue := sales_mon + sales_wed + sales_thu + sales_fri
  let sales_tue := total_sold - sales_excluding_tue
  have h : sales_tue = 70 := sorry
  exact h

end Kavi_sold_on_Tuesday_l82_82995


namespace sum_of_distinct_integers_l82_82531

theorem sum_of_distinct_integers (a b c d : ℤ) (h : (a - 1) * (b - 1) * (c - 1) * (d - 1) = 25) (distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) : a + b + c + d = 4 :=
by
    sorry

end sum_of_distinct_integers_l82_82531


namespace candle_position_10_walls_candle_position_6_walls_l82_82090

-- Condition for a room with 10 walls
def room_with_10_walls : Prop := 
  -- This definition should encapsulate the properties of a room with 10 walls.
  -- We don't need the exact geometric configuration, it is just a placeholder
  -- to assert that such a room exists.
  ∃ (room : Type) (candle : room), room = 10_walls

-- Proof statement for room with 10 walls
theorem candle_position_10_walls : room_with_10_walls → ∃ (candle_position : ℝ³), 
  ¬fully_illuminates_any_wall candle_position 10_walls :=
sorry

-- Condition for a room with 6 walls
def room_with_6_walls : Prop := 
  -- Similar placeholder definition for a 6-walled room.
  ∃ (room : Type) (candle : room), room = 6_walls

-- Proof statement for room with 6 walls
theorem candle_position_6_walls : room_with_6_walls → ∃ (candle_position : ℝ³), 
  ¬fully_illuminates_any_wall candle_position 6_walls :=
sorry

end candle_position_10_walls_candle_position_6_walls_l82_82090


namespace sum_of_roots_eq_l82_82099

noncomputable def polynomial_sum_of_roots : ℚ :=
  let p := (λ x : ℚ, (3 * x + 4) * (x - 5) + (3 * x + 4) * (x - 7))
  let roots := [(-4) / 3, 6]
  roots.sum

theorem sum_of_roots_eq :
  let p := (λ x : ℚ, (3 * x + 4) * (x - 5) + (3 * x + 4) * (x - 7))
  ∑ root in [(-4) / 3, 6], root = 14 / 3 :=
by
  sorry

end sum_of_roots_eq_l82_82099


namespace cindy_gave_lisa_marbles_l82_82495

-- Definitions for the given conditions
def cindy_initial_marbles : ℕ := 20
def lisa_initial_marbles := cindy_initial_marbles - 5
def lisa_final_marbles := lisa_initial_marbles + 19

-- Theorem we need to prove
theorem cindy_gave_lisa_marbles :
  ∃ n : ℕ, lisa_final_marbles = lisa_initial_marbles + n ∧ n = 19 :=
by
  sorry

end cindy_gave_lisa_marbles_l82_82495


namespace bridge_length_is_250_l82_82062

/-- Define the constants for the problem -/
def train_length : ℝ := 125   -- length of the train in meters
def train_speed_kmh : ℝ := 45 -- speed of the train in km/hr
def crossing_time : ℝ := 30   -- time to cross the bridge in seconds

/-- Conversion factor from km/hr to m/s -/
def kmh_to_ms (speed : ℝ) : ℝ := speed * (1000 / 3600)

/-- Speed of the train in m/s -/
def train_speed_ms : ℝ := kmh_to_ms train_speed_kmh

/-- Total distance covered by the train while crossing the bridge -/
def total_distance_covered : ℝ := train_speed_ms * crossing_time

/-- Length of the bridge -/
def bridge_length : ℝ := total_distance_covered - train_length

/-- Theorem statement asserting the length of the bridge is 250 meters -/
theorem bridge_length_is_250 : bridge_length = 250 := by
  sorry

end bridge_length_is_250_l82_82062


namespace a10_correct_l82_82628

noncomputable def sequence (n : Nat) : ℚ
| 0     => 2
| (n+1) => sequence n / (1 + 3 * (sequence n))

theorem a10_correct : sequence 9 = 2 / 55 := 
by sorry

end a10_correct_l82_82628


namespace problem_solution_l82_82397

noncomputable def n : ℝ := 120 - 60 * Real.sqrt 2

theorem problem_solution
  (arrives_between : ∀ M : {M1 M2 : ℕ // 0 ≤ M1 ∧ M1 ≤ 120 ∧ 0 ≤ M2 ∧ M2 ≤ 120}, 
                      |M.val.1 - M.val.2| ≤ n → true) 
  (probability_condition : ∀ M1 M2 : ℕ, |M1 - M2| ≤ n → (60/120)^2 = 0.5)
  (p q r : ℕ)
  (h1 : p = 120)
  (h2 : q = 60)
  (h3 : r = 2)
  (r_not_divisible_square_of_prime : ∀ k : ℕ, ∀ prime_k : Nat.Prime k, r ≠ k^2) :
  p + q + r = 182 :=
begin
  sorry
end

end problem_solution_l82_82397


namespace value_of_c_l82_82228

theorem value_of_c
    (x y c : ℝ)
    (h1 : 3 * x - 5 * y = 5)
    (h2 : x / (x + y) = c)
    (h3 : x - y = 2.999999999999999) :
    c = 0.7142857142857142 :=
by
    sorry

end value_of_c_l82_82228


namespace dot_product_constant_l82_82548

/-- Define the points P and Q -/ 
def P : ℝ × ℝ := (-2 * Real.sqrt 2, 0)
def Q : ℝ × ℝ := (2 * Real.sqrt 2, 0)
def N : ℝ × ℝ := (11 / 4, 0)

/-- Define curve C as the equation of trajectory for the moving point M -/
def curve_C (M : ℝ × ℝ) : Prop := 
  let x := M.1 in
  let y := M.2 in
  (x^2) / 8 + (y^2) / 4 = 1

/-- Define the line y = k(x - 1) -/
def line (k : ℝ) (M : ℝ × ℝ) : Prop :=
  let x := M.1 in
  let y := M.2 in
  y = k * (x - 1)

/-- Prove that the dot product of vectors NA and NB is a constant value -/
theorem dot_product_constant (k : ℝ) 
  (A B : ℝ × ℝ) 
  (hA : A ∈ curve_C ∧ line k A) 
  (hB : B ∈ curve_C ∧ line k B) :
  let x1 := A.1 in
  let y1 := A.2 in
  let x2 := B.1 in
  let y2 := B.2 in
  let NA := (x1 - N.1, y1 - N.2) in
  let NB := (x2 - N.1, y2 - N.2) in
  (NA.1 * NB.1 + NA.2 * NB.2) = -7 / 16 :=
by 
  admit

end dot_product_constant_l82_82548


namespace sin_285_value_l82_82095

noncomputable def sin_285 (θ_360_sub_75 θ_sin_45 θ_cos_45 θ_sin_30 θ_cos_30 : ℝ) : Prop :=
θ_360_sub_75 = 285 ∧ 
θ_sin_45 = 45 ∧ 
θ_cos_45 = 45 ∧ 
θ_sin_30 = 30 ∧ 
θ_cos_30 = 30 → 
  sin 285 = - (sin (θ_sin_45 + θ_sin_30) * θ_cos_30 + cos θ_cos_45 * sin θ_sin_30) =
            - ((θ_360_sub_75/2) * (θ_cos_30/2) + (θ_sin_45/2) * (θ_sin_30/2)) = -(\frac{\sqrt{6} + \sqrt{2}}{4})

theorem sin_285_value :
  ∀ (θ_360_sub_75 θ_sin_45 θ_cos_45 θ_sin_30 θ_cos_30 : ℝ),
  sin_285 θ_360_sub_75 θ_sin_45 θ_cos_45 θ_sin_30 θ_cos_30 :=
begin
  intros,
  sorry
end

end sin_285_value_l82_82095


namespace girl_own_doll_girl_another_doll_l82_82880

theorem girl_own_doll (n : Nat) (hn : n > 1) : (n % 4 = 0 ∨ n % 4 = 1) ↔ (∃ swaps : List (Nat × Nat), is_permutation swaps (List.range n) (List.range n)) :=
sorry

theorem girl_another_doll (n : Nat) (hn : n > 1) : (n ≠ 3) ↔ (∃ swaps : List (Nat × Nat), ∀ i, i < n → List.index_of_nth (apply_swaps swaps (List.range n)) i ≠ i) :=
sorry

end girl_own_doll_girl_another_doll_l82_82880


namespace find_f_2022_l82_82362

def satisfies_condition (f : ℝ → ℝ) : Prop :=
  ∀ a b : ℝ, f ((a + 2 * b) / 3) = (f a + 2 * f b) / 3

variables (f : ℝ → ℝ)
  (h_condition : satisfies_condition f)
  (h_f1 : f 1 = 1)
  (h_f4 : f 4 = 7)

theorem find_f_2022 : f 2022 = 4043 :=
  sorry

end find_f_2022_l82_82362


namespace surface_area_of_ball_l82_82435

/-- Define the given conditions:
1. A ball is placed inside a regular tetrahedron with each edge measuring 4 units.
2. The tetrahedron is lying on a horizontal surface.
3. Water is poured into the tetrahedron from the top, causing the ball to float upwards.
4. When the volume of the water reaches 7/8 of the volume of the tetrahedron, the ball just touches
   each of the tetrahedron’s side faces and the water surface.
-/
def ball_surface_area (edge_length : ℝ) (water_volume_fraction : ℝ) (ball_touches_faces : Prop) : ℝ :=
  if (edge_length = 4 ∧ water_volume_fraction = 7 / 8 ∧ ball_touches_faces) then 
    2 * π / 3 
  else 
    0

/-- The problem statement:
Prove that the surface area of the ball, given the above conditions, is 2π/3 square units.
-/
theorem surface_area_of_ball :
  ball_surface_area 4 (7 / 8) true = 2 * π / 3 :=
by
  sorry

end surface_area_of_ball_l82_82435


namespace find_original_number_l82_82411

noncomputable def original_number (x : ℝ) : Prop :=
  1000 * x = 3 / x

theorem find_original_number (x : ℝ) (h : original_number x) : x = (Real.sqrt 30) / 100 :=
sorry

end find_original_number_l82_82411


namespace locus_of_moving_point_l82_82296

open Real

theorem locus_of_moving_point
  (M N P Q T E : ℝ × ℝ)
  (a b : ℝ)
  (h_ellipse_M : M.1^2 / 48 + M.2^2 / 16 = 1)
  (h_P : P = (-M.1, M.2))
  (h_Q : Q = (-M.1, -M.2))
  (h_T : T = (M.1, -M.2))
  (h_ellipse_N : N.1^2 / 48 + N.2^2 / 16 = 1)
  (h_perp : (M.1 - N.1) * (M.1 + N.1) + (M.2 - N.2) * (M.2 + N.2) = 0)
  (h_intersection : ∃ x y : ℝ, (y - Q.2) = (N.2 - Q.2)/(N.1 - Q.1) * (x - Q.1) ∧ (y - P.2) = (T.2 - P.2)/(T.1 - P.1) * (x - P.1) ∧ E = (x, y)) : 
  (E.1^2 / 12 + E.2^2 / 4 = 1) :=
  sorry

end locus_of_moving_point_l82_82296


namespace man_speed_correct_l82_82793

-- Define the conditions
def train_length : ℝ := 500 -- in meters
def train_speed_km_hr : ℝ := 63 -- in km/hr
def crossing_time : ℝ := 29.997600191984642 -- in seconds

-- Define conversion factor
def km_per_hr_to_m_per_s (v : ℝ) : ℝ :=
  v * (1000 / 3600)

-- Define the corresponding speed in m/s
def train_speed_m_s : ℝ :=
  km_per_hr_to_m_per_s train_speed_km_hr

-- Define the man's speed in m/s
def man_speed_m_s : ℝ :=
  train_speed_m_s - (train_length / crossing_time)

-- Define conversion factor from m/s to km/hr
def m_per_s_to_km_per_hr (v : ℝ) : ℝ :=
  v * (3600 / 1000)

-- Define the man's speed in km/hr
def man_speed_km_hr : ℝ :=
  m_per_s_to_km_per_hr man_speed_m_s

theorem man_speed_correct :
  man_speed_km_hr ≈ 2.9988 := by
  sorry

end man_speed_correct_l82_82793


namespace sum_of_roots_of_equation_l82_82103

theorem sum_of_roots_of_equation :
  let f := (3 * (x: ℝ) + 4) * (x - 5) + (3 * x + 4) * (x - 7)
  ∀ x : ℝ, f = 0 → ∑ (roots : ℝ) in {x | f x = 0}, x = -2 :=
by {
  let f := (3 * (x: ℝ) + 4) * (x - 5) + (3 * x + 4) * (x - 7),
  sorry
}

end sum_of_roots_of_equation_l82_82103


namespace isosceles_right_triangle_measure_l82_82786

theorem isosceles_right_triangle_measure (a XY YZ : ℝ) 
    (h1 : XY > YZ) 
    (h2 : a^2 = 25 / (1/2)) : XY = 10 :=
by
  sorry

end isosceles_right_triangle_measure_l82_82786


namespace number_of_students_speaking_two_languages_l82_82244

variables (G H M GH GM HM GHM N : ℕ)

def students_speaking_two_languages (G H M GH GM HM GHM N : ℕ) : ℕ :=
  G + H + M - (GH + GM + HM) + GHM

theorem number_of_students_speaking_two_languages 
  (h_total : N = 22)
  (h_G : G = 6)
  (h_H : H = 15)
  (h_M : M = 6)
  (h_GHM : GHM = 1)
  (h_students : N = students_speaking_two_languages G H M GH GM HM GHM N): 
  GH + GM + HM = 6 := 
by 
  unfold students_speaking_two_languages at h_students 
  sorry

end number_of_students_speaking_two_languages_l82_82244


namespace percentage_charge_l82_82065

def car_cost : ℝ := 14600
def initial_savings : ℝ := 14500
def trip_charge : ℝ := 1.5
def number_of_trips : ℕ := 40
def grocery_value : ℝ := 800
def final_savings_needed : ℝ := car_cost - initial_savings

-- The amount earned from trips
def amount_from_trips : ℝ := number_of_trips * trip_charge

-- The amount needed from percentage charge on groceries
def amount_from_percentage (P: ℝ) : ℝ := grocery_value * P

-- The required amount from percentage charge on groceries
def required_amount_from_percentage : ℝ := final_savings_needed - amount_from_trips

theorem percentage_charge (P: ℝ) (h: amount_from_percentage P = required_amount_from_percentage) : P = 0.05 :=
by 
  -- Proof follows from the given condition that amount_from_percentage P = required_amount_from_percentage
  sorry

end percentage_charge_l82_82065


namespace jill_total_trip_duration_is_101_l82_82276

def first_bus_wait_time : Nat := 12
def first_bus_ride_time : Nat := 30
def first_bus_delay_time : Nat := 5

def walk_time_to_train : Nat := 10
def train_wait_time : Nat := 8
def train_ride_time : Nat := 20
def train_delay_time : Nat := 3

def second_bus_wait_time : Nat := 20
def second_bus_ride_time : Nat := 6

def route_b_combined_time := (second_bus_wait_time + second_bus_ride_time) / 2

def total_trip_duration : Nat := 
  first_bus_wait_time + first_bus_ride_time + first_bus_delay_time +
  walk_time_to_train + train_wait_time + train_ride_time + train_delay_time +
  route_b_combined_time

theorem jill_total_trip_duration_is_101 : total_trip_duration = 101 := by
  sorry

end jill_total_trip_duration_is_101_l82_82276


namespace evelyn_total_marbles_l82_82513

def initial_marbles := 95
def marbles_from_henry := 9
def marbles_from_grace := 12
def number_of_cards := 6
def marbles_per_card := 4

theorem evelyn_total_marbles :
  initial_marbles + marbles_from_henry + marbles_from_grace + number_of_cards * marbles_per_card = 140 := 
by 
  sorry

end evelyn_total_marbles_l82_82513


namespace quadrilateral_BF_length_l82_82430

theorem quadrilateral_BF_length
  (A B C D E F : Point)
  (right_angle_A : ∡ A = 90°)
  (right_angle_C : ∡ C = 90°)
  (on_first_segment : E ∈ line_segment A C)
  (on_second_segment : F ∈ line_segment A C)
  (DE_perpendicular : perpendicular D E A C)
  (BF_perpendicular : perpendicular B F A C)
  (AE : dist A E = 3)
  (DE : dist D E = 5)
  (CE : dist C E = 7) :
  dist B F = 5 := 
  sorry

end quadrilateral_BF_length_l82_82430


namespace A_finishes_job_in_12_days_l82_82420

variable (A B : ℝ)

noncomputable def work_rate_A_and_B := (1 / 40)
noncomputable def work_rate_A := (1 / A)
noncomputable def work_rate_B := (1 / B)

theorem A_finishes_job_in_12_days
  (h1 : work_rate_A + work_rate_B = work_rate_A_and_B)
  (h2 : 10 * work_rate_A_and_B = 1 / 4)
  (h3 : 9 * work_rate_A = 3 / 4) :
  A = 12 :=
  sorry

end A_finishes_job_in_12_days_l82_82420


namespace rotten_pineapples_l82_82748

theorem rotten_pineapples (initial sold fresh remaining rotten: ℕ) 
  (h1: initial = 86) 
  (h2: sold = 48) 
  (h3: fresh = 29) 
  (h4: remaining = initial - sold) 
  (h5: rotten = remaining - fresh) : 
  rotten = 9 := by 
  sorry

end rotten_pineapples_l82_82748


namespace min_value_ineq_l82_82290

theorem min_value_ineq (a b c k : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hk : 0 < k) : 
  (ka / b + kb / c + kc / a) ≥ 3k :=
sorry

end min_value_ineq_l82_82290


namespace max_value_of_f_on_interval_l82_82770

noncomputable def f (x : ℝ) : ℝ := x^2 * (3 - x)

theorem max_value_of_f_on_interval : 
  ∃ x ∈ set.Icc (0 : ℝ) 3, f x = 4 :=
by
  sorry

end max_value_of_f_on_interval_l82_82770


namespace alyssa_final_money_l82_82068

-- Definitions based on conditions
def weekly_allowance : Int := 8
def spent_on_movies : Int := weekly_allowance / 2
def earnings_from_washing_car : Int := 8

-- The statement to prove
def final_amount : Int := (weekly_allowance - spent_on_movies) + earnings_from_washing_car

-- The theorem expressing the problem
theorem alyssa_final_money : final_amount = 12 := by
  sorry

end alyssa_final_money_l82_82068


namespace equal_share_of_tea_l82_82490

def totalCups : ℕ := 10
def totalPeople : ℕ := 5
def cupsPerPerson : ℕ := totalCups / totalPeople

theorem equal_share_of_tea : cupsPerPerson = 2 := by
  sorry

end equal_share_of_tea_l82_82490


namespace driver_net_pay_is_25_dollars_per_hour_l82_82806

-- Define the conditions of the problem
def hours := 3
def speed := 50  -- miles per hour
def fuel_efficiency := 25  -- miles per gallon
def pay_per_mile := 0.60  -- dollars per mile
def fuel_cost_per_gallon := 2.50  -- dollars per gallon

-- Statement: the net rate of pay per hour should be 25 dollars per hour
theorem driver_net_pay_is_25_dollars_per_hour :
  ( (  pay_per_mile * (speed * hours) - ( (speed * hours) / fuel_efficiency ) * fuel_cost_per_gallon ) / hours ) = 25 := 
sorry

end driver_net_pay_is_25_dollars_per_hour_l82_82806


namespace negation_statement_l82_82367

theorem negation_statement (h : ∀ x : ℝ, |x - 2| + |x - 4| > 3) : 
  ∃ x0 : ℝ, |x0 - 2| + |x0 - 4| ≤ 3 :=
sorry

end negation_statement_l82_82367


namespace real_roots_exist_l82_82508

theorem real_roots_exist (K : ℝ) : 
  ∃ x : ℝ, x = K^2 * (x - 1) * (x - 3) :=
by
  sorry  -- Proof goes here

end real_roots_exist_l82_82508


namespace annual_rate_equivalence_l82_82878

noncomputable def quarterly_rate (annual_rate : ℝ) : ℝ :=
  annual_rate / 4

noncomputable def compounded_rate (quarterly_rate : ℝ) : ℝ :=
  (1 + quarterly_rate) ^ 4

noncomputable def equivalent_annual_rate (annual_compounded : ℝ) : ℝ :=
  (annual_compounded - 1) * 100

theorem annual_rate_equivalence (annual_rate : ℝ) (r : ℝ) :
  annual_rate = 5 -> r ≈ 5.09 / 100 := by
  let rq := quarterly_rate annual_rate
  let compounded := compounded_rate rq
  let ra := equivalent_annual_rate compounded
  sorry

end annual_rate_equivalence_l82_82878


namespace part1_part2_l82_82195

noncomputable def f (a x : ℝ) : ℝ := log10 (20 / (x + 10) + a)

theorem part1 : ∃ a : ℝ, (∀ x : ℝ, f a x = -f a (-x)) ∧ a = -1 :=
by
  sorry

theorem part2 (h : ∀ x : ℝ, f (-1) x = -f (-1) (-x)) : 
  { x : ℝ | f (-1) x > 0 } = { x : ℝ | -10 < x ∧ x < 0 } :=
by
  sorry

end part1_part2_l82_82195


namespace work_completion_l82_82422

theorem work_completion (W : ℝ) (b_days : ℝ) (a_days : ℝ) : (b_days = 7) → (a_days = 10) → (∀ (time : ℝ), time = 70 / 17) :=
by
  intros _ _
  exact rfl
  sorry

end work_completion_l82_82422


namespace geometry_intersection_ellipse_proof_l82_82338

noncomputable def intersection_is_ellipse (R : ℝ) (φ : ℝ) : Prop :=
  ∀ (cylinder : Cylinder ℝ) (plane : Plane ℝ),
  (plane.intersects_cylinder_lateral_surface cylinder ∧ 
   ¬ plane.is_perpendicular_to_cylinder_axis cylinder ∧ 
   ¬ plane.intersects_cylinder_bases cylinder)
   → 
  (plane.intersection_with_cylinder cylinder).is_ellipse ∧
  (plane.intersection_with_cylinder cylinder).major_diameter = 2 * R ∧
  (plane.intersection_with_cylinder cylinder).minor_diameter = 2 * R * Real.cos φ

theorem geometry_intersection_ellipse_proof {R φ : ℝ} (cylinder : Cylinder ℝ) (plane : Plane ℝ)
  (h_plane_intersects : plane.intersects_cylinder_lateral_surface cylinder)
  (h_plane_not_perpendicular : ¬ plane.is_perpendicular_to_cylinder_axis cylinder)
  (h_plane_not_intersects_bases : ¬ plane.intersects_cylinder_bases cylinder) :
  (plane.intersection_with_cylinder cylinder).is_ellipse ∧
  (plane.intersection_with_cylinder cylinder).major_diameter = 2 * R ∧
  (plane.intersection_with_cylinder cylinder).minor_diameter = 2 * R * Real.cos φ :=
by
  sorry

end geometry_intersection_ellipse_proof_l82_82338


namespace fold_point_area_l82_82268

theorem fold_point_area (AB AC : ℝ) (B : ℝ) (q r s : ℕ) :
  AB = 45 → AC = 81 → B = 90 → 
  s = 3 → s ≠ 0 → -- s non divisible by the square of any prime (as a positive integer and state not 0)
  ∃ q r, (168.75 * Real.pi - 253.125 * Real.sqrt s) = (q * real.pi - r * real.sqrt s) ∧ q + r + s = 424 :=
by
  intros h1 h2 h3 h4 h5
  use 168
  use 253
  sorry

end fold_point_area_l82_82268


namespace sum_of_coeffs_l82_82515

noncomputable def a (x : ℤ) := 2 * x^2
noncomputable def b (y : ℤ) := 3 * y^2

def factor_8x8_minus_243y8 [Ring R] (x y : R) : (R × R × R) :=
  (2 * x^2 - 3 * y^2, 2 * x^2 + 3 * y^2, 4 * x^4 + 9 * y^4)

theorem sum_of_coeffs (x y : ℤ) :
  let factors := factor_8x8_minus_243y8 x y in
  let c1 := factors.1 in
  let c2 := factors.2 in
  let c3 := factors.3 in
  (c1.coeff 1 + c1.coeff 0 + c2.coeff 1 + c2.coeff 0 + c3.coeff 1 + c3.coeff 0) = 17 :=
by
  sorry

end sum_of_coeffs_l82_82515


namespace N_composite_l82_82872

theorem N_composite :
  let N := 7 * 9 * 13 + 2020 * 2018 * 2014 in
  ¬Nat.prime N := 
by
  sorry

end N_composite_l82_82872


namespace minimum_value_log_sum_l82_82160

noncomputable def minimum_log_sum (x : ℝ) (h : x > 1) : ℝ :=
  log 9 / log x + log x / (log 27)

theorem minimum_value_log_sum (x : ℝ) (h : x > 1) : 
  minimum_log_sum x h ≥ 2 * real.sqrt 6 / 3 := 
  sorry

end minimum_value_log_sum_l82_82160


namespace D_is_incenter_of_ABC_l82_82756

-- Definitions and conditions
variables {Point Circle Line : Type}
variables (A B C D O : Point)
variables (circle1 circle2 : Circle)
variables (secant : Line)

-- Assumptions based on the problem conditions
axiom circle1_intersects_circle2_at_A_and_B : circle1 ∩ circle2 = {A, B}
axiom O_is_center_of_circle1 : center circle1 = O
axiom O_lies_on_circle2 : O ∈ circle2
axiom secant_intersects_circle2_at_D_and_C : secant ∩ circle2 = {D, C}
axiom secant_passes_through_O : O ∈ secant

-- Theorem to prove that D is the incenter of triangle ABC
theorem D_is_incenter_of_ABC 
(h1 : circle1_intersects_circle2_at_A_and_B)
(h2 : O_is_center_of_circle1)
(h3 : O_lies_on_circle2)
(h4 : secant_intersects_circle2_at_D_and_C)
(h5 : secant_passes_through_O) : 
is_incenter D A B C :=
sorry

end D_is_incenter_of_ABC_l82_82756


namespace head_start_distance_l82_82021

theorem head_start_distance (v_A v_B L H : ℝ) (h1 : v_A = 15 / 13 * v_B)
    (h2 : t_A = L / v_A) (h3 : t_B = (L - H) / v_B) (h4 : t_B = t_A - 0.25 * L / v_B) :
    H = 23 / 60 * L :=
sorry

end head_start_distance_l82_82021


namespace largest_absolute_value_of_p_minus_1_m_plus_n_l82_82735

theorem largest_absolute_value_of_p_minus_1 (a r : ℝ) (h_geom_seq : a > 0 ∧ r > 0 ∧ a * (1 + r + r^2) = 10) :
  let p_x := (λ x, (x - a) * (x - a * r) * (x - a * r^2)) in
  |p_x (-1)| <= 2197 / 27 := sorry

theorem m_plus_n : 
  let m := 2197 in
  let n := 27 in
  m + n = 2224 := sorry

end largest_absolute_value_of_p_minus_1_m_plus_n_l82_82735


namespace tangent_parallel_to_line_l82_82375

def f (x : ℝ) : ℝ := x ^ 3 + x - 2

theorem tangent_parallel_to_line (x y : ℝ) : 
  (y = 4 * x - 1) ↔ (x = 1 ∧ y = 0) ∨ (x = -1 ∧ y = -4) :=
by
  sorry

end tangent_parallel_to_line_l82_82375


namespace triangle_properties_l82_82267

variables (a b c A B : ℝ)

-- Condition that m is parallel to n
def parallel_condition : Prop := 
  a * sin B = sqrt 3 * b * cos A

-- Given values
def given_values : Prop :=
  a = sqrt 7 ∧ b = 2

-- Prove A = π/3 and area is 3√3/2
theorem triangle_properties :
  (parallel_condition a b A B) →
  (given_values a b) →
  (A = π/3 ∧ (1 / 2 * b * c * sin A = 3 * sqrt 3 / 2)) :=
by
  sorry

end triangle_properties_l82_82267


namespace no_integer_of_form_xyxy_in_base10_is_perfect_cube_smallest_base_b_for_perfect_cube_of_form_xyxy_l82_82345

-- Problem 1: Form xyxy in base 10 cannot be a perfect cube
theorem no_integer_of_form_xyxy_in_base10_is_perfect_cube :
  ∀ x y : ℕ, ¬ ∃ z : ℕ, (101 * (10 * x + y) = z^3) :=
by
  sorry

-- Problem 2: Find the smallest base b > 1 such that xyxy in base b is a perfect cube
theorem smallest_base_b_for_perfect_cube_of_form_xyxy (b x y : ℕ) :
  (b > 1) → (xyxy_in_base_b_is_perfect_cube b x y → b = 7) :=
by
  sorry

-- Auxiliary definition for Problem 2 to clarify the form xyxy in any base
def xyxy_in_base_b (b x y : ℕ) : ℕ :=
  (b^3 * x) + (b^2 * y) + (b * x) + y

-- Define the condition of being a perfect cube for the form xyxy in base b
def xyxy_in_base_b_is_perfect_cube (b x y : ℕ) : Prop :=
  ∃ z : ℕ, xyxy_in_base_b b x y = z^3

end no_integer_of_form_xyxy_in_base10_is_perfect_cube_smallest_base_b_for_perfect_cube_of_form_xyxy_l82_82345


namespace find_M_l82_82146

theorem find_M : ∃ M : ℕ, M > 0 ∧ 18 ^ 2 * 45 ^ 2 = 15 ^ 2 * M ^ 2 ∧ M = 54 := by
  use 54
  sorry

end find_M_l82_82146


namespace sum_of_squares_of_coefficients_is_375_l82_82767

def p (x : ℚ) : ℚ := 5 * (x^4 + 2*x^3 + 3*x^2 + 1)

theorem sum_of_squares_of_coefficients_is_375 :
  let c := [5, 10, 15, 5] in
  (c.map (λ a, a^2)).sum = 375 :=
by
  sorry

end sum_of_squares_of_coefficients_is_375_l82_82767


namespace problem_statement_l82_82109

noncomputable def math_problem (X Y P A B S T : Point) : Prop :=
  (∃ (X Y P : Point) 
     (P_not_on_XY : ¬ collinear X Y P) 
     (A_on_bisector_XYP : is_bisector A X Y P)
     (B_on_bisector_YXP : is_bisector B Y X P)
     (S_intersects_XP: intersects S A B X P)
     (T_intersects_YP: intersects T A B Y P),
     |XS| * |YT| = |XY|^2)

-- Helper definitions to assist in the problem statement
axiom Point : Type
axiom collinear : Point → Point → Point → Prop
axiom is_bisector : Point → Point → Point → Point → Prop 
axiom intersects : Point → Point → Point → Point → Point → Prop 

theorem problem_statement 
  (X Y P A B S T : Point) 
  (P_not_on_XY : ¬ collinear X Y P) 
  (A_on_bisector_XYP : is_bisector A X Y P)
  (B_on_bisector_YXP : is_bisector B Y X P)
  (S_intersects_XP: intersects S A B X P)
  (T_intersects_YP: intersects T A B Y P) :
  math_problem X Y P A B S T := 
sorry

end problem_statement_l82_82109


namespace triangle_rectangle_ratio_l82_82481

theorem triangle_rectangle_ratio (s b w l : ℕ) 
(h1 : 2 * s + b = 60) 
(h2 : 2 * (w + l) = 60) 
(h3 : 2 * w = l) 
(h4 : b = w) 
: s / w = 5 / 2 := 
by 
  sorry

end triangle_rectangle_ratio_l82_82481


namespace smallest_base_for_100_base_3_digits_l82_82765

theorem smallest_base_for_100_base_3_digits :
  ∃ b : ℕ, b * b ≤ 100 ∧ 100 < b * b * b ∧ 
           ∀ c : ℕ, c * c ≤ 100 ∧ 100 < c * c * c → c ≥ b :=
begin
  let b := 5,
  use b,
  split,
  { linarith, }, -- this will solve b^2 ≤ 100
  split,
  { linarith, }, -- this will solve 100 < b^3
  { intros c hc1 hc2,
    linarith, -- this should handle the minimality check for c
  },
end

end smallest_base_for_100_base_3_digits_l82_82765


namespace smallest_difference_l82_82753

noncomputable def triangle_lengths (DE EF FD : ℕ) : Prop :=
  DE < EF ∧ EF ≤ FD ∧ DE + EF + FD = 3010 ∧ DE + EF > FD ∧ EF + FD > DE ∧ FD + DE > EF

theorem smallest_difference :
  ∃ (DE EF FD : ℕ), triangle_lengths DE EF FD ∧ EF - DE = 1 :=
by
  sorry

end smallest_difference_l82_82753


namespace diagonal_of_unit_square_is_sqrt2_l82_82639

-- Define a square with side length 1
def sideLength (s : ℝ) := s = 1

-- Define the diagonal of the square
def diagonalLength (s : ℝ) : ℝ := s * Real.sqrt 2

-- Prove that the diagonal length of the square with side length 1 is √2
theorem diagonal_of_unit_square_is_sqrt2 (s : ℝ) (h : sideLength s) :
    diagonalLength s = Real.sqrt 2 :=
by
  -- Since s = 1, substituting gives the length of the diagonal
  rw [sideLength] at h
  simp [diagonalLength, h]
  sorry

end diagonal_of_unit_square_is_sqrt2_l82_82639


namespace angle_BAC_eq_60_l82_82860

theorem angle_BAC_eq_60
  (ABC : Type)
  [inner_product_space ℝ ABC]
  (A B C O : ABC)
  (h_incircle : ∀ (P : ABC), P ≠ A ∧ P ≠ B ∧ P ≠ C → dist P O = dist P ([A, B, C] : set ABC))
  (h_angle_ABC : angle A B C = real.pi / 180 * 75)
  (h_angle_BCA : angle B C A = real.pi / 180 * 45) :
  angle B A C = real.pi / 180 * 60 :=
sorry

end angle_BAC_eq_60_l82_82860


namespace binomial_negative_two_pow_l82_82492

theorem binomial_negative_two_pow :
  ∑ k in Finset.range (101 + 1), (if k % 2 = 0 then (Nat.choose 101 k) * (2^k) else -(Nat.choose 101 k) * (2^k)) = -(2^101) :=
by
  sorry

end binomial_negative_two_pow_l82_82492


namespace length_of_other_diagonal_l82_82496

theorem length_of_other_diagonal
  (Area : ℝ)
  (d1 : ℝ)
  (hArea : Area = 21.46)
  (hd1 : d1 = 7.4) :
  ∃ d2 : ℝ, d2 ≈ 5.8 ∧ Area = (d1 * d2) / 2 :=
by
  use 5.8
  sorry

end length_of_other_diagonal_l82_82496


namespace planar_graph_three_colorable_l82_82683

theorem planar_graph_three_colorable (G : Type) [graph G] [planar G] : 
  ∃ (colors : V(G) → ℕ), (∀ v w : V(G), edge G v w → colors v ≠ colors w) ∧
  (∀ cycle : list (V(G)), ∀ v w : V(G), v ∈ cycle → w ∈ cycle → v ≠ w → colors v ≠ colors w) :=
sorry

end planar_graph_three_colorable_l82_82683


namespace base_7_digits_of_4300_l82_82210

-- Define the conditions about powers of 7 and base-7 representation
def power_of_seven (n : ℕ) : ℕ := 7^n

-- Define the condition for the problem
def largest_power_of_seven_le_4300: {n : ℕ // power_of_seven n ≤ 4300} :=
  ⟨4, by norm_num [power_of_seven, nat.pow]; exact dec_trivial⟩

-- State the theorem about the number of digits in the base-7 representation of 4300
theorem base_7_digits_of_4300 : ∃ d : ℕ, d = 5 ∧ 
  (∀ n : ℕ, n ≤ 4300 → n < power_of_seven (largest_power_of_seven_le_4300.val + 1)) :=
begin
  use 5,
  split,
  { refl },
  { intros n hn, 
    have : largest_power_of_seven_le_4300.val = 4 := rfl,
    rw this,
    have h_powers : power_of_seven 5 = 16807 := by norm_num [power_of_seven, nat.pow],
    have h_upper : 4300 < 16807 := by norm_num,
    exact h_upper,
  }
end

end base_7_digits_of_4300_l82_82210


namespace students_appeared_l82_82614

def passed (T : ℝ) : ℝ := 0.35 * T
def B_grade_range (T : ℝ) : ℝ := 0.25 * T
def failed (T : ℝ) : ℝ := T - passed T

theorem students_appeared (T : ℝ) (hp : passed T = 0.35 * T)
    (hb : B_grade_range T = 0.25 * T) (hf : failed T = 481) :
    T = 740 :=
by
  -- proof goes here
  sorry

end students_appeared_l82_82614


namespace fraction_oj_is_5_over_13_l82_82757

def capacity_first_pitcher : ℕ := 800
def capacity_second_pitcher : ℕ := 500
def fraction_oj_first_pitcher : ℚ := 1 / 4
def fraction_oj_second_pitcher : ℚ := 3 / 5

def amount_oj_first_pitcher : ℚ := capacity_first_pitcher * fraction_oj_first_pitcher
def amount_oj_second_pitcher : ℚ := capacity_second_pitcher * fraction_oj_second_pitcher

def total_amount_oj : ℚ := amount_oj_first_pitcher + amount_oj_second_pitcher
def total_capacity : ℚ := capacity_first_pitcher + capacity_second_pitcher

def fraction_oj_large_container : ℚ := total_amount_oj / total_capacity

theorem fraction_oj_is_5_over_13 : fraction_oj_large_container = (5 / 13) := by
  -- Proof would go here
  sorry

end fraction_oj_is_5_over_13_l82_82757


namespace tan_identity_l82_82230

theorem tan_identity (A B : ℝ) (tan : ℝ → ℝ)
  (hA : A = real.pi / 18)
  (hB : B = real.pi / 9)
  (h_tan_pi_6 : tan (real.pi / 6) = 1 / real.sqrt 3) :
  (1 + tan A) * (1 + tan B) = 1 + real.sqrt 3 * (tan A + tan B) :=
sorry

end tan_identity_l82_82230


namespace ellipse_standard_equation_and_line_l82_82170

theorem ellipse_standard_equation_and_line (a b c : ℝ) (P : ℝ × ℝ)
    (h1: a = 2 * c) (h2: 2 * b = 2 * sqrt 3) (h3 : a^2 = b^2 + c^2) (h4: P = (0, 2))
    : ( ∃ eq : Prop, eq = (∃ x y : ℝ, x^2 / 4 + y^2 / 3 = 1) ) ∧ 
      ( ∃ k : ℝ, (1 + (k^2)) * x^2 + 4 * k * 2 * x + 4 = 0 ∧ 
          (16 - 12 * k^2) / (3 + 4 * k^2) = 2 ∧ 
          (y = k * x + 2) ) :=
by
  sorry

end ellipse_standard_equation_and_line_l82_82170


namespace find_value_l82_82916

noncomputable def f : ℝ → ℝ := sorry -- f is defined as per our conditions

-- Definitions of conditions
axiom odd_function : ∀ x : ℝ, f (-x) = -f (x)
axiom symmetry_about_line_1 : ∀ x : ℝ, f (x) = f (2 - x)
axiom f_on_interval : ∀ x : ℝ, x ∈ set.Icc (-1) 0 → f (x) = -x

-- Theorem to prove
theorem find_value : f 2015 + f 2016 = 1 :=
sorry

end find_value_l82_82916


namespace range_of_a_for_monotonic_f_l82_82925

theorem range_of_a_for_monotonic_f {a : ℝ} : 
  (∀ x y : ℝ, x ≤ y → f a x ≤ f a y) → 4 ≤ a ∧ a < 8 := 
by
  -- Assume \( f(x) \) is monotonically increasing on \( (-\infty, +\infty) \)
  -- the code simply states the conditions and expected result, omitting proof details
  sorry

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x > 1 then x^2 else (4 - a/2) * x - 1

end range_of_a_for_monotonic_f_l82_82925


namespace boy_speed_in_kmph_l82_82036

-- Define the conditions
def side_length : ℕ := 35
def time_seconds : ℕ := 56

-- Perimeter of the square field
def perimeter : ℕ := 4 * side_length

-- Speed in meters per second
def speed_mps : ℚ := perimeter / time_seconds

-- Speed in kilometers per hour
def speed_kmph : ℚ := speed_mps * (3600 / 1000)

-- Theorem stating the boy's speed is 9 km/hr
theorem boy_speed_in_kmph : speed_kmph = 9 :=
by
  sorry

end boy_speed_in_kmph_l82_82036


namespace sum_at_simple_interest_l82_82061

noncomputable def find_principal_sum (R : ℝ) : ℝ :=
  let P := (15000 / 50) in P

theorem sum_at_simple_interest (R : ℝ) : 
  let P := find_principal_sum R in
  (P * (R + 5) * 10 / 100) - (P * R * 10 / 100) = 150 → P = 300 :=
by
  let P := find_principal_sum R
  sorry

end sum_at_simple_interest_l82_82061


namespace max_height_l82_82795

def height (t : ℝ) : ℝ := -20 * t^2 + 40 * t + 20

theorem max_height : ∃ t₀ : ℝ, Height.height t₀ = 40 ∧ (∀ t : ℝ, Height.height t ≤ 40) := by
  sorry

end max_height_l82_82795


namespace johns_total_cost_l82_82469

def original_costs : ℝ := 250 + (3 * 325) + (4 * 450)
def discount_amount : ℝ := 
  (if original_costs > 2000 then (0.15 * (original_costs - 2000) + 0.10 * 1000) 
   else if original_costs > 1000 then 0.10 * (original_costs - 1000) 
   else 0)
def discounted_cost : ℝ := original_costs - discount_amount
def sales_tax : ℝ := 0.08 * discounted_cost
def total_cost : ℝ := discounted_cost + sales_tax

theorem johns_total_cost : total_cost = 2992.95 :=
by
  unfold original_costs discount_amount discounted_cost sales_tax total_cost
  sorry

end johns_total_cost_l82_82469


namespace cubic_equation_real_root_l82_82856

theorem cubic_equation_real_root (b : ℝ) : ∃ x : ℝ, x^3 + b * x + 25 = 0 := 
sorry

end cubic_equation_real_root_l82_82856


namespace total_attendance_l82_82837

theorem total_attendance (A C : ℕ) (ticket_sales : ℕ) (adult_ticket_cost child_ticket_cost : ℕ) (total_collected : ℕ)
    (h1 : C = 18) (h2 : ticket_sales = 50) (h3 : adult_ticket_cost = 8) (h4 : child_ticket_cost = 1)
    (h5 : ticket_sales = adult_ticket_cost * A + child_ticket_cost * C) :
    A + C = 22 :=
by {
  sorry
}

end total_attendance_l82_82837


namespace fields_fertilized_in_25_days_l82_82992

-- Definitions from conditions
def fertilizer_per_horse_per_day : ℕ := 5
def number_of_horses : ℕ := 80
def fertilizer_needed_per_acre : ℕ := 400
def number_of_acres : ℕ := 20
def acres_fertilized_per_day : ℕ := 4

-- Total fertilizer produced per day
def total_fertilizer_per_day : ℕ := fertilizer_per_horse_per_day * number_of_horses

-- Total fertilizer needed
def total_fertilizer_needed : ℕ := fertilizer_needed_per_acre * number_of_acres

-- Days to collect enough fertilizer
def days_to_collect_fertilizer : ℕ := total_fertilizer_needed / total_fertilizer_per_day

-- Days to spread fertilizer
def days_to_spread_fertilizer : ℕ := number_of_acres / acres_fertilized_per_day

-- Calculate the total time until all fields are fertilized
def total_days : ℕ := days_to_collect_fertilizer + days_to_spread_fertilizer

-- Theorem statement
theorem fields_fertilized_in_25_days : total_days = 25 :=
by
  sorry

end fields_fertilized_in_25_days_l82_82992


namespace sum_of_coordinates_reflection_l82_82334

def point (x y : ℝ) : Type := (x, y)

variable (y : ℝ)

theorem sum_of_coordinates_reflection :
  let A := point 3 y in
  let B := point 3 (-y) in
  A.1 + A.2 + B.1 + B.2 = 6 :=
by
  sorry

end sum_of_coordinates_reflection_l82_82334


namespace tshirts_equation_l82_82803

theorem tshirts_equation (x : ℝ) 
    (hx : x > 0)
    (march_cost : ℝ := 120000)
    (april_cost : ℝ := 187500)
    (april_increase : ℝ := 1.4)
    (cost_increase : ℝ := 5) :
    120000 / x + 5 = 187500 / (1.4 * x) :=
by 
  sorry

end tshirts_equation_l82_82803


namespace min_value_of_n_l82_82720

-- Define the condition
def condition (n : ℤ) : Prop :=
  let S := (n + (n + 1) + (n + 2) + ... + (n + 20))
  S > 2019

-- The theorem statement
theorem min_value_of_n (n : ℤ) (h : condition n) : n ≥ 87 :=
sorry

end min_value_of_n_l82_82720


namespace marbles_missing_l82_82317

theorem marbles_missing : 
  let total_marbles := 3 * 7
  let required_marbles := 6 * 6
  let missing_marbles := required_marbles - total_marbles
  in missing_marbles = 15 :=
by
  -- define the total, required, and missing marbles
  let total_marbles := 3 * 7
  let required_marbles := 6 * 6
  let missing_marbles :=  required_marbles - total_marbles
  
  -- specify the equality to prove
  show missing_marbles = 15
  sorry

end marbles_missing_l82_82317


namespace appropriate_sampling_method_l82_82967

open Real  -- Ensure usage of Real numbers for fractions and such

theorem appropriate_sampling_method
  (N : ℕ) (n : ℕ) (N1 : ℕ) (N2 : ℕ) (N3 : ℕ)
  (hN : N = 600)
  (hn : n = 60)
  (hN1 : N1 = 230)
  (hN2 : N2 = 290)
  (hN3 : N3 = 80) :
  ∃ (n1 n2 n3 : ℕ),
  n1 = (N1 * n / N).toInt ∧
  n2 = (N2 * n / N).toInt ∧
  n3 = (N3 * n / N).toInt ∧
  n1 = 23 ∧
  n2 = 29 ∧
  n3 = 8 ∧
  stratified_sampling_method =
    ∀ x ∈ {N1, N2, N3}, x * n / N ∈ {n1, n2, n3} := by
  sorry

end appropriate_sampling_method_l82_82967


namespace rabbit_travel_time_l82_82051

/-- A rabbit moves at a constant speed of 3 miles per hour. How long does it take for the rabbit to travel 2 miles? Express your answer in minutes. -/
theorem rabbit_travel_time :
  ∀ (r d : ℕ), (r = 3) → (d = 2) → (t : ℕ), (t = (d * 60) / r) → t = 40 :=
by
  intros r d hr hd t ht
  rw [hr, hd] at ht
  exact ht

end rabbit_travel_time_l82_82051


namespace area_of_lune_l82_82827

theorem area_of_lune :
  let d1 := 2
  let d2 := 4
  let r1 := d1 / 2
  let r2 := d2 / 2
  let height := r2 - r1
  let area_triangle := (1 / 2) * d1 * height
  let area_semicircle_small := (1 / 2) * π * r1^2
  let area_combined := area_triangle + area_semicircle_small
  let area_sector_large := (1 / 4) * π * r2^2
  let area_lune := area_combined - area_sector_large
  area_lune = 1 - (1 / 2) * π := 
by
  sorry

end area_of_lune_l82_82827


namespace angle_between_vectors_l82_82219

open Real
open Vector

variables {a b : ℝ^2}

theorem angle_between_vectors (ha : a ≠ 0) (hb : b ≠ 0) (h : ∥a + b∥ = ∥a - b∥) : angle a b = π / 2 :=
sorry

end angle_between_vectors_l82_82219


namespace find_side_a_find_area_l82_82604

-- Definitions from the conditions
variables {A B C : ℝ} 
variables {a b c : ℝ}
variable (angle_B: B = 120 * Real.pi / 180)
variable (side_b: b = Real.sqrt 7)
variable (side_c: c = 1)

-- The first proof problem: Prove that a = 2 given the above conditions
theorem find_side_a (h_angle_B: B = 120 * Real.pi / 180)
  (h_side_b: b = Real.sqrt 7) (h_side_c: c = 1)
  (h_cos_formula: b^2 = a^2 + c^2 - 2 * a * c * Real.cos B) : a = 2 :=
  by
  sorry

-- The second proof problem: Prove that the area is sqrt(3)/2 given the above conditions
theorem find_area (h_angle_B: B = 120 * Real.pi / 180)
  (h_side_b: b = Real.sqrt 7) (h_side_c: c = 1)
  (h_side_a: a = 2) : (1 / 2) * a * c * Real.sin B = Real.sqrt 3 / 2 :=
  by
  sorry

end find_side_a_find_area_l82_82604


namespace longest_side_length_l82_82374

-- Define the conditions
def condition1 (x y : ℝ) : Prop := x + y ≤ 4
def condition2 (x y : ℝ) : Prop := 3 * x + y ≥ 3
def condition3 (x y : ℝ) : Prop := x ≥ 0
def condition4 (x y : ℝ) : Prop := y ≥ 0

-- Define the theorem to prove
theorem longest_side_length : 
  ∃ x1 y1 x2 y2 : ℝ, 
    (condition1 x1 y1 ∧ condition2 x1 y1 ∧ condition3 x1 y1 ∧ condition4 x1 y1) ∧
    (condition1 x2 y2 ∧ condition2 x2 y2 ∧ condition3 x2 y2 ∧ condition4 x2 y2) ∧
    (√10 = real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)) :=
sorry

end longest_side_length_l82_82374


namespace isosceles_obtuse_triangle_l82_82483

theorem isosceles_obtuse_triangle (A B C : ℝ) (h_isosceles: A = B)
  (h_obtuse: A + B + C = 180) 
  (h_max_angle: C = 157.5): A = 11.25 :=
by
  sorry

end isosceles_obtuse_triangle_l82_82483


namespace area_of_triangle_ABC_l82_82603

variable {α : Type} [LinearOrder α] [Field α]

-- Given: 
variables (A B C D E F : α) (area_ABC area_BDA area_DCA : α)

-- Conditions:
variable (midpoint_D : 2 * D = B + C)
variable (ratio_AE_EC : 3 * E = A + C)
variable (ratio_AF_FD : 2 * F = A + D)
variable (area_DEF : area_ABC / 6 = 12)

-- To Show:
theorem area_of_triangle_ABC :
  area_ABC = 96 :=
by
  sorry

end area_of_triangle_ABC_l82_82603


namespace positive_difference_solutions_l82_82140

theorem positive_difference_solutions:
  ∀ x : ℝ, (∃ a b : ℝ, (sqrt (sqrt ((9 - (a^2 / 4))) = -3) ∧ sqrt (sqrt ((9 - (b^2 / 4))) = -3) ∧ (abs (a - b) = 24))) :=
begin
  sorry
end

end positive_difference_solutions_l82_82140


namespace multiplicative_inverse_of_AB_l82_82305

noncomputable def A : ℕ := 123456
noncomputable def B : ℕ := 162037
def N : ℕ := 466390

theorem multiplicative_inverse_of_AB :
  ∃ N : ℕ, N > 0 ∧ N < 1000000 ∧ (N * (A * B) ≡ 1 [MOD 1000000]) :=
begin
  use N,
  split,
  -- N > 0
  sorry,
  split,
  -- N < 1000000
  sorry,
  -- N * (A * B) ≡ 1 [MOD 1000000]
  sorry
end

end multiplicative_inverse_of_AB_l82_82305


namespace cost_in_chinese_yuan_l82_82329

theorem cost_in_chinese_yuan
  (usd_to_nad : ℝ := 8)
  (usd_to_cny : ℝ := 5)
  (sculpture_cost_nad : ℝ := 160) :
  sculpture_cost_nad / usd_to_nad * usd_to_cny = 100 := 
by
  sorry

end cost_in_chinese_yuan_l82_82329


namespace infinite_triples_l82_82684

theorem infinite_triples:
  ∃ (f : ℕ → ℕ × ℕ × ℕ), ∀ n : ℕ, 
    let (x, y, z) := f n in x^2 + y^2 = z^2022 :=
by
  sorry

end infinite_triples_l82_82684


namespace exists_five_digit_number_with_digit_sum_and_divisible_by_31_l82_82877

def digit_sum (n : ℕ) : ℕ := (n.digits 10).sum

theorem exists_five_digit_number_with_digit_sum_and_divisible_by_31 :
  ∃ n : ℕ, 10000 ≤ n ∧ n ≤ 99999 ∧ digit_sum n = 31 ∧ n % 31 = 0 := 
begin
  use 93775,
  split, { exact le_of_eq rfl },
  split, { exact le_of_eq rfl },
  split,
  { -- prove digit sum
    sorry },
  { -- prove divisible by 31
    sorry },
end

end exists_five_digit_number_with_digit_sum_and_divisible_by_31_l82_82877


namespace div_by_10_3pow_l82_82962

theorem div_by_10_3pow
    (m : ℤ)
    (n : ℕ)
    (h : (3^n + m) % 10 = 0) :
    (3^(n + 4) + m) % 10 = 0 := by
  sorry

end div_by_10_3pow_l82_82962


namespace incorrect_statements_are_1_2_4_l82_82418

theorem incorrect_statements_are_1_2_4:
    let statements := ["Inductive reasoning and analogical reasoning both involve reasoning from specific to general.",
                       "When making an analogy, it is more appropriate to use triangles in a plane and parallelepipeds in space as the objects of analogy.",
                       "'All multiples of 9 are multiples of 3, if a number m is a multiple of 9, then m must be a multiple of 3' is an example of syllogistic reasoning.",
                       "In deductive reasoning, as long as it follows the form of deductive reasoning, the conclusion is always correct."]
    let incorrect_statements := {1, 2, 4}
    incorrect_statements = {i | i ∈ [1, 2, 3, 4] ∧
                             ((i = 1 → ¬(∃ s, s ∈ statements ∧ s = statements[0])) ∧ 
                              (i = 2 → ¬(∃ s, s ∈ statements ∧ s = statements[1])) ∧ 
                              (i = 3 → ∃ s, s ∈ statements ∧ s = statements[2]) ∧ 
                              (i = 4 → ¬(∃ s, s ∈ statements ∧ s = statements[3])))} :=
by
  sorry

end incorrect_statements_are_1_2_4_l82_82418


namespace rectangular_prism_volume_l82_82055

theorem rectangular_prism_volume
  (l w h : ℝ)
  (h1 : l * w = 15)
  (h2 : w * h = 10)
  (h3 : l * h = 6) :
  l * w * h = 30 := by
  sorry

end rectangular_prism_volume_l82_82055


namespace area_under_parabola_eq_one_third_l82_82709

theorem area_under_parabola_eq_one_third :
  (∫ x in 0..1, x^2) = 1 / 3 :=
by
  sorry

end area_under_parabola_eq_one_third_l82_82709


namespace positive_difference_solutions_l82_82142

theorem positive_difference_solutions:
  ∀ x : ℝ, (∃ a b : ℝ, (sqrt (sqrt ((9 - (a^2 / 4))) = -3) ∧ sqrt (sqrt ((9 - (b^2 / 4))) = -3) ∧ (abs (a - b) = 24))) :=
begin
  sorry
end

end positive_difference_solutions_l82_82142


namespace expand_polynomials_correct_l82_82514

-- Define the polynomial p1
def p1 : ℚ[X] := 3 * X^2 + 4 * X - 5

-- Define the polynomial p2
def p2 : ℚ[X] := 4 * X^3 - 3 * X + 2

-- Define the expected result of the multiplication
def expected : ℚ[X] := 12 * X^5 + 16 * X^4 - 29 * X^3 - 6 * X^2 + 23 * X - 10

-- Statement to prove
theorem expand_polynomials_correct : (p1 * p2) = expected :=
by sorry

end expand_polynomials_correct_l82_82514


namespace hannah_jerry_difference_l82_82209

-- Define the calculations of Hannah (H) and Jerry (J)
def H : Int := 10 - (3 * 4)
def J : Int := 10 - 3 + 4

-- Prove that H - J = -13
theorem hannah_jerry_difference : H - J = -13 := by
  sorry

end hannah_jerry_difference_l82_82209


namespace unique_poly_deg_3_conditioned_l82_82214

noncomputable def is_poly_deg_3 (f : ℚ[X]) : Prop :=
  leadingCoeff f ≠ 0 ∧ natDegree f = 3

theorem unique_poly_deg_3_conditioned :
  ∃! (f : ℚ[X]), is_poly_deg_3 f ∧ (∀ x, eval x^2 f = (eval x f)^2) ∧ eval 1 f = eval (-1) f :=
by sorry

end unique_poly_deg_3_conditioned_l82_82214


namespace min_sum_abc_l82_82222

theorem min_sum_abc (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^2 + 2 * a * b + 2 * a * c + 4 * b * c = 16) : 
  a + b + c = "boxed_​answer" := by
  sorry

end min_sum_abc_l82_82222


namespace even_perfect_square_factors_l82_82211

theorem even_perfect_square_factors :
  let a_choices := {2, 4, 6}
  let b_choices := {0, 2}
  let c_choices := {0, 2, 4, 6, 8}
  ∀ (a ∈ a_choices) (b ∈ b_choices) (c ∈ c_choices) 
  if (a ≥ 2 ∧ a % 2 = 0 ∧ b % 2 = 0 ∧ c % 2 = 0) then
    3 * 2 * 5 = 30 := by
  sorry

end even_perfect_square_factors_l82_82211


namespace campers_rowing_in_the_morning_l82_82699

theorem campers_rowing_in_the_morning (total_campers : ℕ) (afternoon_campers : ℕ) 
  (total_condition : total_campers = 60) (afternoon_condition : afternoon_campers = 7) : 
  total_campers - afternoon_campers = 53 :=
by
  rw [total_condition, afternoon_condition]
  rfl

end campers_rowing_in_the_morning_l82_82699


namespace cats_left_correct_l82_82457

-- Define initial conditions
def siamese_cats : ℕ := 13
def house_cats : ℕ := 5
def sold_cats : ℕ := 10

-- Define the total number of cats initially
def total_cats_initial : ℕ := siamese_cats + house_cats

-- Define the number of cats left after the sale
def cats_left : ℕ := total_cats_initial - sold_cats

-- Prove the number of cats left is 8
theorem cats_left_correct : cats_left = 8 :=
by 
  sorry

end cats_left_correct_l82_82457


namespace range_of_m_l82_82224

theorem range_of_m 
  (m : Real) 
  (h : (m + 4)^(-1/2) < (3 - 2*m)^(-1/2)) : 
  -1/3 < m ∧ m < 3/2 := 
  sorry

end range_of_m_l82_82224


namespace smallest_number_divisible_l82_82428

theorem smallest_number_divisible (n : ℕ) 
    (h1 : (n - 20) % 15 = 0) 
    (h2 : (n - 20) % 30 = 0)
    (h3 : (n - 20) % 45 = 0)
    (h4 : (n - 20) % 60 = 0) : 
    n = 200 :=
sorry

end smallest_number_divisible_l82_82428


namespace sum_of_numbers_le_1_1_l82_82008

theorem sum_of_numbers_le_1_1 :
  let nums := [1.4, 0.9, 1.2, 0.5, 1.3]
  let filtered := nums.filter (fun x => x <= 1.1)
  filtered.sum = 1.4 :=
by
  let nums := [1.4, 0.9, 1.2, 0.5, 1.3]
  let filtered := nums.filter (fun x => x <= 1.1)
  have : filtered = [0.9, 0.5] := sorry
  have : filtered.sum = 1.4 := sorry
  exact this

end sum_of_numbers_le_1_1_l82_82008


namespace tan_two_alpha_l82_82157

theorem tan_two_alpha (α β : ℝ) (h₁ : Real.tan (α - β) = -3/2) (h₂ : Real.tan (α + β) = 3) :
  Real.tan (2 * α) = 3/11 := 
sorry

end tan_two_alpha_l82_82157


namespace islanders_liars_count_l82_82389

theorem islanders_liars_count (n : ℕ) (two_liars_truth : bool) (circular_arrangement : bool) 
(h1 : n = 2017)
(h2 : two_liars_truth = tt)
(h3 : circular_arrangement = tt) :
  ∃ liars : ℕ, liars = 1344 :=
by {
  sorry
}

end islanders_liars_count_l82_82389


namespace ratio_of_radii_of_truncated_cone_l82_82830

theorem ratio_of_radii_of_truncated_cone 
  (R r s : ℝ) 
  (h1 : s = Real.sqrt (R * r)) 
  (h2 : (π * (R^2 + r^2 + R * r) * (2 * s) / 3) = 3 * (4 * π * s^3 / 3)) :
  R / r = 7 := 
sorry

end ratio_of_radii_of_truncated_cone_l82_82830


namespace medals_award_count_l82_82973

theorem medals_award_count :
  let total_ways (n k : ℕ) := n.factorial / (n - k).factorial
  ∃ (award_ways : ℕ), 
    let no_americans := total_ways 6 3
    let one_american := 4 * 3 * total_ways 6 2
    award_ways = no_americans + one_american ∧
    award_ways = 480 :=
by
  sorry

end medals_award_count_l82_82973


namespace average_speed_correct_l82_82436

-- Conditions
def speed_uphill := 30 -- km/hr
def speed_downhill := 70 -- km/hr
def distance_uphill := 100 -- km
def distance_downhill := 50 -- km

-- Total distance
def total_distance := distance_uphill + distance_downhill -- 150 km

-- Time taken for each part
def time_uphill := distance_uphill / speed_uphill -- 10/3 hours
def time_downhill := distance_downhill / speed_downhill -- 5/7 hours

-- Total time
def total_time := time_uphill + time_downhill -- 85/21 hours

-- Average speed
def average_speed := total_distance / total_time

-- Theorem to prove the average speed
theorem average_speed_correct : average_speed ≈ 37.06 := by
  have h1: total_distance = 150 := rfl
  have h2: time_uphill = 10 / 3 := rfl
  have h3: time_downhill = 5 / 7 := rfl
  have h4: total_time = 85 / 21 := by 
    rw [← h2, ← h3]
    linarith
  have h5: average_speed = 3150 / 85 := by
    rw [← h1, ← h4]
    norm_num
  show average_speed ≈ 37.06
  rw h5
  norm_num
  linarith

end average_speed_correct_l82_82436


namespace ham_cost_l82_82993

theorem ham_cost (bread_cost cheese_cost sandwich_cost : ℝ)
  (h_bread_cost : bread_cost = 0.15)
  (h_cheese_cost : cheese_cost = 0.35)
  (h_sandwich_cost : sandwich_cost = 0.90) :
  let ham_cost := sandwich_cost - (2 * bread_cost + cheese_cost)
  in ham_cost = 0.25 :=
by
  rw [h_bread_cost, h_cheese_cost, h_sandwich_cost]
  simp only [ham_cost]
  sorry

end ham_cost_l82_82993


namespace distance_between_points_l82_82486

noncomputable def car_speed := 90
noncomputable def train_speed := 60
noncomputable def time_to_min_distance := 2
noncomputable def equilateral := true
noncomputable def cos_120 := - (1 / 2 : ℝ)

theorem distance_between_points :
  (∀ (S : ℝ), 
  equilateral →
  (S - car_speed * time_to_min_distance) ^ 2 + (train_speed * time_to_min_distance) ^ 2 + 
  (S - car_speed * time_to_min_distance) * (train_speed * time_to_min_distance) = 
  (S - 90 * 2) ^ 2 + (60 * 2) ^ 2 + (S - 90 * 2) * (60 * 2) →
  S = 210) :=
by
  sorry

end distance_between_points_l82_82486


namespace circle_polar_equation_and_slope_l82_82626

theorem circle_polar_equation_and_slope (
  hC: ∀ x y : ℝ, (x + 6)^2 + y^2 = 25,
  hl: ∃ θ α : ℝ, theta = alpha,
  hAB: ∃ A B : ℝ × ℝ, (A.1 + 6)^2 + A.2^2 = 25 ∧ (B.1 + 6)^2 + B.2^2 = 25 ∧ dist A B = sqrt 10
): (∀ ρ θ : ℝ, ρ^2 + 12 * ρ * cos θ + 11 = 0) ∧ (∃ (α : ℝ), tan α = sqrt 15 / 3 ∨ tan α = - sqrt 15 / 3) :=
sorry

end circle_polar_equation_and_slope_l82_82626


namespace function_domain_l82_82403

theorem function_domain :
  ∀ x : ℝ, x ≠ -5 ↔ ∃ y : ℝ, y = (4 * x - 2) / (2 * x + 10) :=
by
  intros x
  split
  sorry

end function_domain_l82_82403


namespace find_symmetry_center_l82_82931

noncomputable def f (x : ℝ) : ℝ := (1 / 2) * Real.sin (2 * x + π / 6)

theorem find_symmetry_center :
  ∃ x₀ ∈ Icc 0 (π / 2), (∃ y₀, ∀ x : ℝ, f (2 * x₀ - x) = f x) ∧ x₀ = 5 * π / 12 := 
by
  sorry

end find_symmetry_center_l82_82931


namespace tangent_line_at_origin_range_of_m_number_of_solutions_l82_82932

noncomputable def f (x : ℝ) : ℝ := Real.sin x
noncomputable def g (x : ℝ) : ℝ := Real.exp x * Real.cos x

-- (I)
theorem tangent_line_at_origin :
  tangent_line_at g (0, g 0) = { l | ∃ k, l = k * (1, 0) ∧ (k = 1) } :=
sorry

-- (II)
theorem range_of_m (m : ℝ) : 
  (∀ x ∈ Set.Icc (-(Real.pi / 2)) 0, g x ≥ x * f x + m) →
  m ≤ -Real.pi / 2 :=
sorry

-- (III)
theorem number_of_solutions :
  Set.card {x | x ∈ Set.Icc (-(Real.pi / 2)) (Real.pi / 2) ∧ g x = x * f x} = 2 :=
sorry

end tangent_line_at_origin_range_of_m_number_of_solutions_l82_82932


namespace angle_is_60_degrees_l82_82601

namespace AngleSupplement

theorem angle_is_60_degrees (α : ℝ) (h_sup : 180 - α = 2 * α) : α = 60 :=
by
  -- This is where the proof would go
  sorry

end AngleSupplement

end angle_is_60_degrees_l82_82601


namespace max_chord_line_of_circle_l82_82716

theorem max_chord_line_of_circle (x y : ℝ) (line_eq : Prop) (point_eq : Prop) (circle_eq : Prop) :
  (point_eq = ((2, 1) : ℝ × ℝ)) ∧ (circle_eq = (λ (x y : ℝ), (x - 1)^2 + (y + 2)^2 = 5)) →
  (∃ (a b c : ℝ), line_eq = (λ (x y : ℝ), a * x + b * y + c = 0) ∧ (a = 3) ∧ (b = -1) ∧ (c = -5)) :=
by sorry

end max_chord_line_of_circle_l82_82716


namespace problem1_problem2_problem3_l82_82573

theorem problem1 (f : ℝ → ℝ) : 
  ∀ x : ℝ, f x = real.exp (2 * x) - 1 - 2 * x - 0 * x ^ 2 →
    (∀ x < 0, deriv f x < 0) ∧ (∀ x > 0, deriv f x > 0) :=
by
  intro f hf
  sorry

theorem problem2 (f : ℝ → ℝ) (k : ℝ) : 
  (∀ x ≥ 0, f x ≥ 0) →
  ∀ k, f = (fun x => real.exp (2 * x) - 1 - 2 * x - k * x ^ 2) → 
    k ≤ 2 :=
by
  intro f k hf hk
  sorry

theorem problem3 (n : ℕ) :
  ∀ n > 0, (real.exp (2 * n) - 1) / (real.exp 2 - 1) ≥ (2 * n^3 + n) / 3 :=
by
  intro n hn
  sorry

end problem1_problem2_problem3_l82_82573


namespace number_of_positive_factors_l82_82589

theorem number_of_positive_factors (n : ℕ) (h : n = 34992) : nat.totient (34992) = 40 :=
sorry

end number_of_positive_factors_l82_82589


namespace nina_widgets_purchase_l82_82676

theorem nina_widgets_purchase (P : ℝ) (h1 : 8 * (P - 1) = 24) (h2 : 24 / P = 6) : true :=
by
  sorry

end nina_widgets_purchase_l82_82676


namespace max_sticks_form_obtuse_triangle_l82_82538

theorem max_sticks_form_obtuse_triangle (n : ℕ) (a : Fin n → ℝ) (h : ∀ (i j k : Fin n), i.val < j.val → j.val < k.val → a i * a i + a j * a j < a k * a k) : n ≤ 4 :=
begin
  sorry
end

end max_sticks_form_obtuse_triangle_l82_82538


namespace final_sum_is_correct_l82_82971

-- Define initial readings on the calculators
def initial_readings : List Int := [2, 0, -1]

-- Define the function describing the operation performed on each calculator per turn
def next_value (value : Int) : Int :=
  if value = 2 then value ^ 2
  else if value = 0 then 0
  else if value = -1 then -value
  else value

-- Define the behavior after 44 participants
noncomputable def final_values : List Int :=
  let powers_of_two := 2 ^ (2 ^ 44)
  let zero_squared := 0
  let negated_neg_one := 1
  [powers_of_two, zero_squared, negated_neg_one]

-- Define the final sum
noncomputable def final_sum := (final_values.foldl (Int.add) 0)

-- The statement to prove
theorem final_sum_is_correct : final_sum = 2 ^ (2 ^ 44) + 1 := by sorry

end final_sum_is_correct_l82_82971


namespace solve_quadratic_equation_l82_82697

theorem solve_quadratic_equation (x : ℝ) : x^2 + 4 * x = 5 ↔ x = 1 ∨ x = -5 := sorry

end solve_quadratic_equation_l82_82697


namespace slower_speed_is_l82_82456

def slower_speed_problem
  (faster_speed : ℝ)
  (additional_distance : ℝ)
  (actual_distance : ℝ)
  (v : ℝ) :
  Prop :=
  actual_distance / v = (actual_distance + additional_distance) / faster_speed

theorem slower_speed_is
  (h1 : faster_speed = 25)
  (h2 : additional_distance = 20)
  (h3 : actual_distance = 13.333333333333332)
  : ∃ v : ℝ,  slower_speed_problem faster_speed additional_distance actual_distance v ∧ v = 10 :=
by {
  sorry
}

end slower_speed_is_l82_82456


namespace base5_addition_correct_l82_82342

-- Definitions to interpret base-5 numbers
def base5_to_base10 (n : List ℕ) : ℕ :=
  n.reverse.foldl (λ acc d => acc * 5 + d) 0

-- Conditions given in the problem
def num1 : ℕ := base5_to_base10 [2, 0, 1, 4]  -- (2014)_5 in base-10
def num2 : ℕ := base5_to_base10 [2, 2, 3]    -- (223)_5 in base-10

-- Statement to prove
theorem base5_addition_correct :
  base5_to_base10 ([2, 0, 1, 4]) + base5_to_base10 ([2, 2, 3]) = base5_to_base10 ([2, 2, 4, 2]) :=
by
  -- Proof goes here
  sorry

#print axioms base5_addition_correct

end base5_addition_correct_l82_82342


namespace sum_of_squares_fraction_l82_82178

variable {x1 x2 x3 y1 y2 y3 : ℝ}

theorem sum_of_squares_fraction :
  x1 + x2 + x3 = 0 → y1 + y2 + y3 = 0 → x1 * y1 + x2 * y2 + x3 * y3 = 0 →
  (x1^2 / (x1^2 + x2^2 + x3^2)) + (y1^2 / (y1^2 + y2^2 + y3^2)) = 2 / 3 :=
by
  intros h1 h2 h3
  sorry

end sum_of_squares_fraction_l82_82178


namespace smallest_positive_θ_l82_82933

noncomputable def f (x : ℝ) : ℝ := sqrt 3 * cos x - sin x

def is_symmetrical_about (g : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x : ℝ, g (2 * a - x) = g x

theorem smallest_positive_θ :
  ∃ θ > 0, is_symmetrical_about (λ x, f (x - θ)) (π / 6) ∧ ∀ ε > 0, ε < θ → ¬is_symmetrical_about (λ x, f (x - ε)) (π / 6) :=
sorry

end smallest_positive_θ_l82_82933


namespace daily_productivity_increase_l82_82439

variable (p : ℝ) (r : ℝ)

def weekly_increase (p : ℝ) : ℝ := p * (1 + 0.02)
def daily_increase (p : ℝ) (r : ℝ) : ℝ := p * (1 + r)^5

theorem daily_productivity_increase :
  (∃ r, (1 + r)^5 = 1.02) ∧ (r ≈ 0.00396) :=
by
  sorry

end daily_productivity_increase_l82_82439


namespace edric_hourly_rate_l82_82121

-- Define conditions
def edric_monthly_salary : ℝ := 576
def edric_weekly_hours : ℝ := 8 * 6 -- 48 hours
def average_weeks_per_month : ℝ := 4.33
def edric_monthly_hours : ℝ := edric_weekly_hours * average_weeks_per_month -- Approx 207.84 hours

-- Define the expected result
def edric_expected_hourly_rate : ℝ := 2.77

-- Proof statement
theorem edric_hourly_rate :
  edric_monthly_salary / edric_monthly_hours = edric_expected_hourly_rate :=
by
  sorry

end edric_hourly_rate_l82_82121


namespace not_all_teams_have_same_victories_probability_quixajuba_first_probability_three_teams_tied_first_l82_82152

-- Part (a)
theorem not_all_teams_have_same_victories :
  ¬∃ (v : ℕ), ∀ t : ℕ, t ≥ 0 → t < 4 → v = 1.5 :=
sorry

-- Part (b)
theorem probability_quixajuba_first : 
  ∈state-space → ∈state-space →
  (P({q_win_all_games})) = 1/8 :=
sorry

-- Part (c)
theorem probability_three_teams_tied_first :
  ∈state-space → ∈state-space →
  (P({three_teams_tied_for_first})) = 1/8 :=
sorry

end not_all_teams_have_same_victories_probability_quixajuba_first_probability_three_teams_tied_first_l82_82152


namespace smallest_number_l82_82478

-- Define the four numbers
def A : ℝ := -1.5
def B : ℝ := 0
def C : ℝ := 2
def D : ℝ := -| -2.5 |

-- State the theorem
theorem smallest_number : D < A ∧ D < B ∧ D < C := by
  sorry

end smallest_number_l82_82478


namespace driver_speed_ratio_l82_82396

theorem driver_speed_ratio (V1 V2 x : ℝ) (h : V1 > 0 ∧ V2 > 0 ∧ x > 0)
  (meet_halfway : ∀ t1 t2, t1 = x / (2 * V1) ∧ t2 = x / (2 * V2))
  (earlier_start : ∀ t1 t2, t1 = t2 + x / (2 * (V1 + V2))) :
  V2 / V1 = (1 + Real.sqrt 5) / 2 := by
  sorry

end driver_speed_ratio_l82_82396


namespace new_water_depth_l82_82807

def fish_tank_base_length : ℝ := 20 -- base length in cm
def fish_tank_base_width : ℝ := 40 -- base width in cm
def fish_tank_height : ℝ := 30 -- tank height in cm
def initial_fill_ratio : ℝ := 0.5 -- tank is half full initially
def additional_volume : ℝ := 4000 -- additional water volume in cm³

def base_area : ℝ := fish_tank_base_length * fish_tank_base_width -- area of the base in cm²
def initial_depth : ℝ := fish_tank_height * initial_fill_ratio -- initially half full depth
def additional_depth : ℝ := additional_volume / base_area -- depth from additional water volume

def new_depth : ℝ := initial_depth + additional_depth -- new depth of the water

theorem new_water_depth : new_depth = 20 := by
  sorry

end new_water_depth_l82_82807


namespace number_of_primes_satisfying_congruence_l82_82893

theorem number_of_primes_satisfying_congruence :
  let primes_in_range := [101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199] in
  let condition (p : ℕ) := p ∈ primes_in_range ∧ ∃ x y : ℤ, x^11 + y^16 ≡ 2013 [MOD p] in
  (finset.filter condition (finset.fromList primes_in_range)).card = 21 :=
by
  let primes_in_range := [101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199]
  have eq_card : (finset.filter (λ p, (p ∈ primes_in_range) ∧ (∃ x y : ℤ, x^11 + y^16 ≡ 2013 [MOD p])) (finset.fromList primes_in_range)).card = 21
  sorry

end number_of_primes_satisfying_congruence_l82_82893


namespace number_of_leopards_l82_82123

variables 
  (lions_saturday elephants_saturday rhinos_monday warthogs_monday buffaloes_sunday total_animals : ℕ)
  (leopards_sunday : ℕ)

-- Given conditions
def saturday_animals := lions_saturday + elephants_saturday
def monday_animals := rhinos_monday + warthogs_monday
def total_weekend_animals := saturday_animals + monday_animals
def sunday_animals := total_animals - total_weekend_animals

-- Hypotheses
axiom h1 : lions_saturday = 3
axiom h2 : elephants_saturday = 2
axiom h3 : rhinos_monday = 5
axiom h4 : warthogs_monday = 3
axiom h5 : buffaloes_sunday = 2
axiom h6 : total_animals = 20

-- Prove the number of leopards on Sunday
theorem number_of_leopards : leopards_sunday = sunday_animals - buffaloes_sunday := by
  simp [saturday_animals, monday_animals, total_weekend_animals, sunday_animals]
  rw [h1, h2, h3, h4, h5, h6]
  norm_num
  sorry

end number_of_leopards_l82_82123


namespace new_average_of_remaining_numbers_l82_82107

theorem new_average_of_remaining_numbers (sum_12 avg_12 n1 n2 : ℝ) 
  (h1 : avg_12 = 90)
  (h2 : sum_12 = 1080)
  (h3 : n1 = 80)
  (h4 : n2 = 85)
  : (sum_12 - n1 - n2) / 10 = 91.5 := 
by
  sorry

end new_average_of_remaining_numbers_l82_82107


namespace linear_function_quadrants_l82_82917

theorem linear_function_quadrants (k b : ℝ) 
  (h1 : k < 0)
  (h2 : b < 0) 
  : k * b > 0 := 
sorry

end linear_function_quadrants_l82_82917


namespace number_of_oxygen_atoms_l82_82441

/-- Given a compound has 1 H, 1 Cl, and a certain number of O atoms and the molecular weight of the compound is 68 g/mol,
    prove that the number of O atoms is 2. -/
theorem number_of_oxygen_atoms (atomic_weight_H: ℝ) (atomic_weight_Cl: ℝ) (atomic_weight_O: ℝ) (molecular_weight: ℝ) (n : ℕ):
    atomic_weight_H = 1.0 →
    atomic_weight_Cl = 35.5 →
    atomic_weight_O = 16.0 →
    molecular_weight = 68.0 →
    molecular_weight = atomic_weight_H + atomic_weight_Cl + n * atomic_weight_O →
    n = 2 :=
by
  sorry

end number_of_oxygen_atoms_l82_82441


namespace find_CK_find_angle_BCA_l82_82269

variable (CK1 AK2 AC : ℝ)
variable (R r : ℝ)
constants O1 O2 O K1 K2 K K3 : Type

def CK (R r : ℝ) : ℝ := 6 * (R / r)

theorem find_CK (h1 : CK1 = 6) 
                (h2 : AK2 = 8) 
                (h3 : AC = 21) 
                (h4 : R / r = 3 / 2) :
                CK R r = 9 := by
  sorry

def ∠BCA (O : Type) : ℝ := 60

theorem find_angle_BCA (O1 O : Type) 
                       (K1 K2 K K3 : Type) 
                       (h1 : true) -- CK1 = 6 
                       (h2 : true) -- AK2 = 8 
                       (h3 : true) -- AC = 21 
                       (h4 : true) -- radius conditions 
                       (h5 : O1 = "Center of circumcision") :
                       ∠BCA O = 60 := by 
  sorry

end find_CK_find_angle_BCA_l82_82269


namespace sum_of_valid_k_under_25_l82_82407

-- Define the tetromino condition
def can_be_covered_by_tetrominos (k : ℕ) : Prop :=
  ∃ n : ℕ, k = 4 * n

-- The provable statement
theorem sum_of_valid_k_under_25 : 
  (∑ k in (Finset.filter can_be_covered_by_tetrominos (Finset.range 26)), k) = 84 := by
  sorry

end sum_of_valid_k_under_25_l82_82407


namespace find_m_l82_82898

theorem find_m : ∃ m : ℤ, 2^5 - 7 = 3^3 + m ∧ m = -2 :=
by
  use -2
  sorry

end find_m_l82_82898


namespace norma_total_cards_l82_82318

theorem norma_total_cards (initial_cards : ℝ) (additional_cards : ℝ) (total_cards : ℝ) 
  (h1 : initial_cards = 88) (h2 : additional_cards = 70) : total_cards = 158 :=
by
  sorry

end norma_total_cards_l82_82318


namespace proof_equation_of_ellipse_proof_equation_of_line_l82_82077

noncomputable def ellipse_eq (a b : ℝ) : Prop := 
  ∀ x y : ℝ, (x = 1 ∧ y = 3/2) → 
  (x^2 / a^2 + y^2 / b^2 = 1)

theorem proof_equation_of_ellipse (a b : ℝ) (ha : a > b) (hb : b > 0) 
  (h1A : (1:ℝ)^2 / a^2 + (3/2:ℝ)^2 / b^2 = 1) 
  (ecc : (sqrt (a^2 - b^2) / a = 1/2)) : 
  a^2 = 4 ∧ b^2 = 3 := 
sorry

theorem proof_equation_of_line (k : ℝ) (hline_cond : 17 * k^4 + k^2 - 18 = 0) : 
  k = 1 ∨ k = -1 := 
sorry

end proof_equation_of_ellipse_proof_equation_of_line_l82_82077


namespace number_of_solutions_is_odd_l82_82284

theorem number_of_solutions_is_odd :
  let m := { s : (ℕ × ℕ × ℕ × ℕ × ℕ) | 
    let a := s.1,
    let b := s.2.1,
    let c := s.2.2.1,
    let d := s.2.2.2.1,
    let e := s.2.2.2.2,
    0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧ 
    (1 / a + 1 / b + 1 / c + 1 / d + 1 / e = 1)
  }.to_finset.card
  in m % 2 = 1 := 
by
  sorry

end number_of_solutions_is_odd_l82_82284


namespace total_attendance_l82_82835

-- Defining the given conditions
def adult_ticket_cost : ℕ := 8
def child_ticket_cost : ℕ := 1
def total_amount_collected : ℕ := 50
def number_of_child_tickets : ℕ := 18

-- Formulating the proof problem
theorem total_attendance (A : ℕ) (C : ℕ) (H1 : C = number_of_child_tickets)
  (H2 : adult_ticket_cost * A + child_ticket_cost * C = total_amount_collected) :
  A + C = 22 := by
  sorry

end total_attendance_l82_82835


namespace icing_cubes_count_l82_82040

theorem icing_cubes_count :
  let n := 5
  let total_cubes := n * n * n
  let side_faces := 4
  let cubes_per_edge_per_face := (n - 2) * (n - 1)
  let shared_edges := 4
  let icing_cubes := (side_faces * cubes_per_edge_per_face) / 2
  icing_cubes = 32 := sorry

end icing_cubes_count_l82_82040


namespace original_team_members_l82_82834

theorem original_team_members (m p total_points : ℕ) (h_m : m = 3) (h_p : p = 2) (h_total : total_points = 12) :
  (total_points / p) + m = 9 := by
  sorry

end original_team_members_l82_82834


namespace units_digit_of_sum_is_4_l82_82896

-- Definitions and conditions based on problem
def base_8_add (a b : List Nat) : List Nat :=
    sorry -- Function to perform addition in base 8, returning result as a list of digits

def units_digit (a : List Nat) : Nat :=
    a.headD 0  -- Function to get the units digit of the result

-- The list representation for the digits of 65 base 8 and 37 base 8
def sixty_five_base8 := [6, 5]
def thirty_seven_base8 := [3, 7]

-- The theorem that asserts the final result
theorem units_digit_of_sum_is_4 : units_digit (base_8_add sixty_five_base8 thirty_seven_base8) = 4 :=
    sorry

end units_digit_of_sum_is_4_l82_82896


namespace original_number_l82_82413

theorem original_number 
  (x : ℝ)
  (h₁ : 0 < x)
  (h₂ : 1000 * x = 3 * (1 / x)) : 
  x = (Real.sqrt 30) / 100 :=
sorry

end original_number_l82_82413


namespace proof1_proof2_l82_82364

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ :=
  |a * x - 2| - |x + 2|

-- Statement for proof 1
theorem proof1 (x : ℝ)
  (a : ℝ) (h : a = 2) (hx : f 2 x ≤ 1) : -1/3 ≤ x ∧ x ≤ 5 :=
sorry

-- Statement for proof 2
theorem proof2 (a : ℝ)
  (h : ∀ x : ℝ, -4 ≤ f a x ∧ f a x ≤ 4) : a = 1 ∨ a = -1 :=
sorry

end proof1_proof2_l82_82364


namespace fraction_pow_zero_l82_82402

theorem fraction_pow_zero :
  let a := 7632148
  let b := -172836429
  (a / b ≠ 0) → (a / b)^0 = 1 := by
  sorry

end fraction_pow_zero_l82_82402


namespace neg_p_l82_82650

open Classical

variables {x : ℤ} (A : Set ℤ)

def is_even (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k
def A := {n : ℤ | is_even n}

def p : Prop := ∀ x : ℤ, 2 * x ∈ A

theorem neg_p : ¬ p ↔ ∃ x : ℤ, 2 * x ∉ A :=
by sorry

end neg_p_l82_82650


namespace value_A_correct_l82_82723

noncomputable def value_of_A (H MATH TEAM MEET : ℕ) : ℕ :=
let MAT := MATH - H in
let E := TEAM - MAT in
let MT := MEET - 2 * E in
TEAM - E - MT

theorem value_A_correct (H MATH TEAM MEET : ℕ) (hH : H = 8) (hMATH : MATH = 40) (hTEAM : TEAM = 50) (hMEET : MEET = 44) :
  value_of_A H MATH TEAM MEET = 24 :=
by
  rw [value_of_A, hH, hMATH, hTEAM, hMEET]
  sorry

end value_A_correct_l82_82723


namespace num_games_played_l82_82379

theorem num_games_played (n : ℕ) (h : n = 14) : (n.choose 2) = 91 :=
by
  sorry

end num_games_played_l82_82379


namespace fg_of_3_is_94_l82_82957

def g (x : ℕ) : ℕ := 4 * x + 5
def f (x : ℕ) : ℕ := 6 * x - 8

theorem fg_of_3_is_94 : f (g 3) = 94 := by
  sorry

end fg_of_3_is_94_l82_82957


namespace hyperbola_eccentricity_sqrt5_l82_82543

theorem hyperbola_eccentricity_sqrt5 (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h_asymptote_parabola_intersect : ∃ x y, y = (b / a) * x ∧ y = x^2 + 1) :
  let e := sqrt ((a ^ 2 + b ^ 2) / a ^ 2) in e = sqrt 5 :=
by
  sorry 

end hyperbola_eccentricity_sqrt5_l82_82543


namespace polynomial_has_k_plus_one_nonzero_coeffs_l82_82640

def P (k : ℕ) (Q : ℕ → ℕ) (x : ℕ) : ℕ := ((x - 1) ^ k) * (Q x)

theorem polynomial_has_k_plus_one_nonzero_coeffs (Q : ℕ → ℕ) (hQ : ∃ x, Q x ≠ 0) (k : ℕ) : ∃ l : ℕ, l ≥ k + 1 ∧ ∀ i, i < l → P k Q i ≠ 0 :=
begin
  -- Lean proof goes here
  sorry
end

end polynomial_has_k_plus_one_nonzero_coeffs_l82_82640


namespace value_of_a_l82_82562

theorem value_of_a (a : ℝ) :
  (∃ (l1 l2 : (ℝ × ℝ × ℝ)),
   l1 = (1, -a, a) ∧ l2 = (3, 1, 2) ∧
   (∃ (m1 m2 : ℝ), 
    (m1 = (1 : ℝ) / a ∧ m2 = -3) ∧ 
    (m1 * m2 = -1))) → a = 3 :=
by sorry

end value_of_a_l82_82562


namespace sum_of_solutions_eq_65_l82_82656

theorem sum_of_solutions_eq_65 :
  let g (x : ℝ) := 12 * x + 5 in
  let g_inv (y : ℝ) := (y - 5) / 12 in
  (∀ x : ℝ, x = g_inv (g ((3 * x)⁻¹)) ∧ g_inv x = g ((3 * x)⁻¹)) →
  (∃ s : ℝ, s = 65) :=
by
  let g (x : ℝ) := 12 * x + 5
  let g_inv (y : ℝ) := (y - 5) / 12
  sorry

end sum_of_solutions_eq_65_l82_82656


namespace find_m_l82_82202

def hyperbola (x y m : ℝ) : Prop := x^2 - (y^2 / (m^2)) = 1

def distance_from_focus_to_asymptote_is_4 (m : ℝ) : Prop :=
  ∃ c : ℝ, (m > 0) ∧ (c^2 = 1 + m^2) ∧ ((m * c) / (Real.sqrt (1 + m^2)) = 4)

theorem find_m (m : ℝ) (h : m > 0) :
  (distance_from_focus_to_asymptote_is_4 m) → (m = 4) :=
begin
  sorry -- proof is omitted
end

end find_m_l82_82202


namespace equation_OF_l82_82617

open_locale real

variables (a b c p : ℝ)

-- Assumptions
variables (ha : a ≠ 0)
variables (hb : b ≠ 0)
variables (hc : c ≠ 0)
variables (hp : p ≠ 0)

theorem equation_OF :
  (1 / c - 1 / b) * x + (1 / p - 1 / a) * y = 0 →
  (1 / c - 1 / b) * x + (1 / p - 1 / a) * y = 0 :=
by
  sorry

end equation_OF_l82_82617


namespace base_of_first_logarithm_l82_82587

theorem base_of_first_logarithm :
  let seq_prod := ∏ i in finset.range (80 - 3 + 1) + 3, real.log (real.of_nat (i + 1)) / real.log (real.of_nat i)
  seq_prod = 4 → 
  real.log (4 : ℝ) / real.log (3 : ℝ) = 1 / 4 := by
  intro h
  sorry

end base_of_first_logarithm_l82_82587


namespace mike_total_payment_l82_82673

theorem mike_total_payment 
  (old_camera_cost : ℕ)
  (lens_cost : ℕ)
  (discount : ℕ)
  (percentage_increase : ℕ)
  (new_camera_extra_cost := old_camera_cost * percentage_increase / 100)
  (new_camera_cost := old_camera_cost + new_camera_extra_cost)
  (lens_cost_after_discount := lens_cost - discount)
  (total_payment := new_camera_cost + lens_cost_after_discount) :
  old_camera_cost = 4000 →
  lens_cost = 400 → 
  discount = 200 →
  percentage_increase = 30 →
  total_payment = 5400 :=
by
  intros h_old h_lens h_discount h_percentage
  rw [h_old, h_lens, h_discount, h_percentage]
  unfold new_camera_extra_cost new_camera_cost lens_cost_after_discount total_payment
  norm_num
  done

end mike_total_payment_l82_82673


namespace sealed_envelope_digit_l82_82825

theorem sealed_envelope_digit :
  ∃ (d : ℕ), d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
             (∃ (s : Finset (ℕ → Prop)), s.card = 3 ∧ 
               (∃ x ∈ s, x d) ∧ 
               (∀ y ∈ ({I, II, III, IV} : Finset (ℕ → Prop)) \ s, 
                  ¬ (y d))) :=
begin
  let I := λ d, d % 2 = 0,
  let II := λ d, d ≠ 5,
  let III := λ d, d > 4,
  let IV := λ d, d ≠ 6,
  sorry
end

end sealed_envelope_digit_l82_82825


namespace max_value_of_m_l82_82285

theorem max_value_of_m (n : ℕ) (h1 : n > 1) :
  ∃ (m : ℕ), (∀ T : Finset (Finset (Fin (n * m))),
    (T.card = 2 * n) ∧ 
    (∀ t ∈ T, t.card = m) ∧ 
    (∀ {t1 t2 : Finset (Fin (n * m))}, t1 ≠ t2 → (t1 ∩ t2).card ≤ 1) ∧ 
    (∀ i : Fin (n * m), (filter (∈ {i}) T).card = 2)) → 
  m ≤ 2 * n - 1 :=
begin
  sorry
end

end max_value_of_m_l82_82285


namespace second_frog_hops_l82_82532

-- Definitions
def frog_hops (first second third fourth : ℕ) : Prop :=
  first = 4 * second ∧
  second = 2 * third ∧
  fourth = 3 * second ∧
  first + second + third + fourth = 156

-- Statement to be proved
theorem second_frog_hops : ∃ (x : ℕ), frog_hops (4 * x) x (x / 2) (3 * x) ∧ x = 18 :=
begin
  sorry
end

end second_frog_hops_l82_82532


namespace percentage_cut_in_magazine_budget_l82_82474

noncomputable def magazine_budget_cut (original_budget : ℕ) (cut_amount : ℕ) : ℕ :=
  (cut_amount * 100) / original_budget

theorem percentage_cut_in_magazine_budget : 
  magazine_budget_cut 940 282 = 30 :=
by
  sorry

end percentage_cut_in_magazine_budget_l82_82474


namespace seating_arrangements_count_l82_82612

theorem seating_arrangements_count :
  let lakers := 4
  let bulls := 3
  let heat := 2
  let celtics := 3
  lakers + bulls + heat + celtics = 12 →
  (4! * 4! * 3! * 2! * 3! = 20736) :=
by
  intros lakers bulls heat celtics h
  sorry

end seating_arrangements_count_l82_82612


namespace position_after_5_steps_l82_82471

theorem position_after_5_steps 
  (step_length : ℕ) (total_steps : ℕ) (total_distance : ℕ) (initial_position : ℕ) 
  (h1 : step_length = 48 / 8) (h2 : total_steps = 8) : 
  let z := initial_position + 5 * step_length in
  z = 30 := 
by
  sorry

end position_after_5_steps_l82_82471


namespace radius_for_visibility_l82_82804

def is_concentric (hex_center : ℝ × ℝ) (circle_center : ℝ × ℝ) : Prop :=
  hex_center = circle_center

def regular_hexagon (side_length : ℝ) : Prop :=
  side_length = 3

theorem radius_for_visibility
  (r : ℝ)
  (hex_center : ℝ × ℝ)
  (circle_center : ℝ × ℝ)
  (P_visible: ℝ)
  (prob_Four_sides_visible: ℝ ) :
  is_concentric hex_center circle_center →
  regular_hexagon 3 →
  prob_Four_sides_visible = 1 / 3 →
  P_visible = 4 →
  r = 2.6 :=
by sorry

end radius_for_visibility_l82_82804


namespace solve_system_l82_82698

theorem solve_system :
  {p : ℝ × ℝ // 
    (p.1 + |p.2| = 3 ∧ 2 * |p.1| - p.2 = 3) ∧
    (p = (2, 1) ∨ p = (0, -3) ∨ p = (-6, 9))} :=
by { sorry }

end solve_system_l82_82698


namespace wall_volume_is_128512_l82_82366

noncomputable def wall_volume (width : ℝ) (height : ℝ) (length : ℝ) : ℝ :=
  width * height * length

theorem wall_volume_is_128512 : 
  ∀ (w : ℝ) (h : ℝ) (l : ℝ), 
  h = 6 * w ∧ l = 7 * h ∧ w = 8 → 
  wall_volume w h l = 128512 := 
by
  sorry

end wall_volume_is_128512_l82_82366


namespace hyperbola_asymptotes_equation_l82_82583

noncomputable def hyperbola_asymptotes (a b : ℝ) (ha : a > 0) (hb : b > 0) (e : ℝ)
  (h_eq : e = 5 / 3)
  (h_hyperbola : ∀ x y : ℝ, (x^2)/(a^2) - (y^2)/(b^2) = 1) :
  String :=
by
  sorry

theorem hyperbola_asymptotes_equation : 
  ∀ a b : ℝ, ∀ ha : a > 0, ∀ hb : b > 0, ∀ e : ℝ,
  e = 5 / 3 →
  (∀ x y : ℝ, (x^2)/(a^2) - (y^2)/(b^2) = 1) →
  ( ∀ (x : ℝ), x ≠ 0 → y = (4/3)*x ∨ y = -(4/3)*x
  )
  :=
by
  intros _
  sorry

end hyperbola_asymptotes_equation_l82_82583


namespace cos_A_eq_sqrt3_div3_of_conditions_l82_82606

noncomputable def given_conditions
  (a b c : ℝ)
  (A B C : ℝ)
  (h1 : (Real.sqrt 3 * b - c) * Real.cos A = a * Real.cos C)
  (h2 : a ≠ 0) 
  (h3 : b ≠ 0) 
  (h4 : c ≠ 0) 
  (h5 : A ≠ 0) 
  (h6 : B ≠ 0) 
  (h7 : C ≠ 0) : Prop :=
  (Real.cos A = Real.sqrt 3 / 3)

theorem cos_A_eq_sqrt3_div3_of_conditions
  (a b c : ℝ)
  (A B C : ℝ)
  (h1 : (Real.sqrt 3 * b - c) * Real.cos A = a * Real.cos C)
  (h2 : a ≠ 0) 
  (h3 : b ≠ 0) 
  (h4 : c ≠ 0) 
  (h5 : A ≠ 0) 
  (h6 : B ≠ 0) 
  (h7 : C ≠ 0) :
  Real.cos A = Real.sqrt 3 / 3 :=
sorry

end cos_A_eq_sqrt3_div3_of_conditions_l82_82606


namespace abs_f_at_1_eq_20_l82_82649

noncomputable def fourth_degree_polynomial (f : ℝ → ℝ) : Prop :=
  ∃ p : Polynomial ℝ, p.degree = 4 ∧ ∀ x, f x = p.eval x

theorem abs_f_at_1_eq_20 
  (f : ℝ → ℝ)
  (h_f_poly : fourth_degree_polynomial f)
  (h_f_neg2 : |f (-2)| = 10)
  (h_f_0 : |f 0| = 10)
  (h_f_3 : |f 3| = 10)
  (h_f_7 : |f 7| = 10) :
  |f 1| = 20 := 
sorry

end abs_f_at_1_eq_20_l82_82649


namespace isosceles_triangle_height_l82_82855

theorem isosceles_triangle_height (a : ℝ) (φ : ℝ) (h : ℝ) : 
  h = a * Real.cos(φ) → ∀ (φ1 φ2 : ℝ), φ1 < φ2 → 
  h1 = a * Real.cos(φ1) → h2 = a * Real.cos(φ2) → h1 > h2 :=
by
  intros h_eq φ1 φ2 φ1_lt_φ2 h1_eq h2_eq
  rw [h1_eq, h2_eq]
  apply mul_lt_mul_of_pos_left
  exact Real.cos_lt_cos_of_lt φ1_lt_φ2
  exact lt_of_le_of_ne (Real.cos_nonneg_of_nonneg_of_le_pi_div_two (le_of_lt φ1_lt_φ2))
    (ne_of_apply_ne (Real.cos) (ne_of_lt φ1_lt_φ2))

end isosceles_triangle_height_l82_82855


namespace find_cost_price_l82_82423

theorem find_cost_price (C : ℝ) (h1 : 1.12 * C + 18 = 1.18 * C) : C = 300 :=
by
  sorry

end find_cost_price_l82_82423


namespace infinitesolutions_k_l82_82501

-- Define the system of equations as given in the problem
def system_of_equations (x y k : ℝ) : Prop :=
  (3 * x - 4 * y = 5) ∧ (9 * x - 12 * y = k)

-- State the theorem that describes the condition for infinitely many solutions
theorem infinitesolutions_k (k : ℝ) :
  (∀ (x y : ℝ), system_of_equations x y k) ↔ k = 15 :=
by
  sorry

end infinitesolutions_k_l82_82501


namespace most_cost_effective_plan_l82_82277

noncomputable def cost_plan_A (minutes : ℕ) : ℝ :=
  3 + 0.3 * minutes

noncomputable def cost_plan_B (minutes : ℕ) : ℝ :=
  let total_cost := 6 + 0.2 * minutes in
  if total_cost > 10 then total_cost * 0.9 else total_cost

noncomputable def cost_plan_C (minutes : ℕ) : ℝ :=
  5 + if minutes > 100 then 0.35 * (minutes - 100) else 0

theorem most_cost_effective_plan (minutes : ℕ) (h : minutes = 45) :
  cost_plan_C minutes < min (cost_plan_A minutes) (cost_plan_B minutes) :=
by
  rw h
  -- Here would be the proof, which is not required as per the instructions
  sorry

end most_cost_effective_plan_l82_82277


namespace lori_beanie_babies_times_l82_82313

theorem lori_beanie_babies_times (l s : ℕ) (h1 : l = 300) (h2 : l + s = 320) : l = 15 * s :=
by
  sorry

end lori_beanie_babies_times_l82_82313


namespace cost_price_of_book_l82_82035

theorem cost_price_of_book
(marked_price : ℝ)
(list_price : ℝ)
(cost_price : ℝ)
(h1 : marked_price = 69.85)
(h2 : list_price = marked_price * 0.85)
(h3 : list_price = cost_price * 1.25) :
cost_price = 65.75 :=
by
  sorry

end cost_price_of_book_l82_82035


namespace part1_part2_l82_82310

-- Definition of the function
def f (a x : ℝ) := |x - a|

-- Proof statement for question 1
theorem part1 (a : ℝ)
  (h : ∀ x : ℝ, f a x ≤ 2 ↔ 1 ≤ x ∧ x ≤ 5) :
  a = 3 := by
  sorry

-- Auxiliary function for question 2
def g (a x : ℝ) := f a (2 * x) + f a (x + 2)

-- Proof statement for question 2
theorem part2 (m : ℝ)
  (h : ∀ x : ℝ, g 3 x ≥ m) :
  m ≤ 1/2 := by
  sorry

end part1_part2_l82_82310


namespace focus_coordinates_l82_82086

-- Define the points marking the endpoints of the minor axis
def point1 := (1.5, -1) : ℝ × ℝ
def point2 := (4.5, -1) : ℝ × ℝ

-- Define the points marking the endpoints of the major axis
def point3 := (3, 2) : ℝ × ℝ
def point4 := (3, -4) : ℝ × ℝ

-- Define the midpoint (center of the ellipse) calculation
def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

-- Calculate the center of the ellipse
def center := midpoint point1 point2  -- Assuming center is calculated via minor axis

-- Define the lengths of the axes
def length (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def minor_axis_length := length point1 point2
def major_axis_length := length point3 point4

-- Define the semi-axes
def a := major_axis_length / 2
def b := minor_axis_length / 2

-- The distance between the center and the foci
def c := real.sqrt (a^2 - b^2)

-- The focus with the greater y-coordinate
def focus_greater_y_coordinate := (3, center.2 + c)

theorem focus_coordinates :
  focus_greater_y_coordinate = (3, 1.598) :=
sorry

end focus_coordinates_l82_82086


namespace mean_proportional_64_81_l82_82780

-- Define the mean proportional between two numbers
def mean_proportional (a b : ℕ) : ℝ :=
  Real.sqrt (a * b)

-- Definition of the numbers 64 and 81
def a : ℕ := 64
def b : ℕ := 81

-- Theorem stating that the mean proportional between 64 and 81 is 72
theorem mean_proportional_64_81 : mean_proportional a b = 72 := by
  sorry

end mean_proportional_64_81_l82_82780


namespace sum_of_roots_eq_l82_82101

noncomputable def polynomial_sum_of_roots : ℚ :=
  let p := (λ x : ℚ, (3 * x + 4) * (x - 5) + (3 * x + 4) * (x - 7))
  let roots := [(-4) / 3, 6]
  roots.sum

theorem sum_of_roots_eq :
  let p := (λ x : ℚ, (3 * x + 4) * (x - 5) + (3 * x + 4) * (x - 7))
  ∑ root in [(-4) / 3, 6], root = 14 / 3 :=
by
  sorry

end sum_of_roots_eq_l82_82101


namespace find_varphi_l82_82592

noncomputable def f (θ : ℝ) : ℝ := sin θ - sqrt 3 * cos θ

theorem find_varphi (θ varphi : ℝ) (h1 : f θ = 2 * sin (θ + varphi)) (h2 : -π < varphi ∧ varphi < π) :
  varphi = -π / 3 :=
sorry

end find_varphi_l82_82592


namespace remainder_of_sum_l82_82238

theorem remainder_of_sum (x y z : ℕ) (h1 : x % 15 = 6) (h2 : y % 15 = 9) (h3 : z % 15 = 3) : 
  (x + y + z) % 15 = 3 := 
  sorry

end remainder_of_sum_l82_82238


namespace smallest_base_value_l82_82072

theorem smallest_base_value :
  let A := Nat.ofDigits [8,5] 9
  let B := Nat.ofDigits [2,1,0] 6
  let C := Nat.ofDigits [1,0,0,0] 4
  let D := Nat.ofDigits [1,1,1,1,1,1] 2
  D < A ∧ D < B ∧ D < C :=
by
  let A := Nat.ofDigits [8,5] 9
  let B := Nat.ofDigits [2,1,0] 6
  let C := Nat.ofDigits [1,0,0,0] 4
  let D := Nat.ofDigits [1,1,1,1,1,1] 2
  sorry

end smallest_base_value_l82_82072


namespace custom_operation_correct_l82_82959

noncomputable def custom_operation (a b c : ℕ) : ℝ :=
  (Real.sqrt ((a + 2) * (b + 3))) / (c + 1)

theorem custom_operation_correct : custom_operation 6 15 5 = 2 := by
  sorry

end custom_operation_correct_l82_82959


namespace latest_start_time_proof_l82_82638

noncomputable def latest_start_time_meeting_constraints (task_times: List ℕ) (dinner_time: ℕ) (break_times: ℕ) (end_time: ℕ) : ℕ :=
  end_time - (task_times.sum + dinner_time + break_times)

theorem latest_start_time_proof : 
    ∀ (homework clean_room trash dishwasher dog sister lawn dinner breaks end_time: ℕ),
    let task_times := [homework, clean_room, trash, dishwasher, dog, sister, lawn] in
    task_times.sum + dinner + breaks = 230 ∧ 
    end_time = 20 ∧ -- Considering 8:00 pm as 20:00 in 24-hour format
    (∀ t ∈ [task_times.headI + task_times.get 1, task_times.get 4 + task_times.get 5 + 5, lawn], t <= 120 )  → -- breaks considered
    latest_start_time_meeting_constraints task_times dinner breaks end_time = 17 :=
by
  intros
  simp [latest_start_time_meeting_constraints]
  sorry

end latest_start_time_proof_l82_82638


namespace tan_theta_l82_82126

noncomputable def side_ba (BC AC : ℝ) : ℝ := 
  real.sqrt ((BC ^ 2) - (AC ^ 2))

theorem tan_theta (BC AC : ℝ) (hBC : BC = 25) (hAC : AC = 20) : 
  real.tan (real.arcsin (AC / BC)) = 3 / 4 :=
by
  -- Definitions from the problem statement:
  let BA := side_ba BC AC
  have hBA : BA = 15 := by sorry  -- Proof that BA = 15 is omitted

  -- Using the formula for tan(theta) = opposite / adjacent:
  let tan_theta := BA / AC
  have htan : tan_theta = 3 / 4 := by sorry  -- Proof that tan_theta = 3/4 is omitted 

  -- Therefore, we have:
  exact htan ▸ rfl

end tan_theta_l82_82126


namespace total_highlighters_l82_82425

-- Define the number of highlighters of each color
def pink_highlighters : ℕ := 10
def yellow_highlighters : ℕ := 15
def blue_highlighters : ℕ := 8

-- Prove the total number of highlighters
theorem total_highlighters : pink_highlighters + yellow_highlighters + blue_highlighters = 33 :=
by
  sorry

end total_highlighters_l82_82425


namespace maximum_value_of_function_l82_82235

theorem maximum_value_of_function (a : ℕ) (ha : 0 < a) : 
  ∃ x : ℝ, x + Real.sqrt (13 - 2 * a * x) = 7 :=
by
  sorry

end maximum_value_of_function_l82_82235


namespace area_triangle_CEF_l82_82975

noncomputable def sqrt (x : ℝ) : ℝ := Real.sqrt x

def quadrilateral_ABCD (AB AD BC CD angle_DAB : ℝ) : Prop :=
  AB = 2 ∧ AD = 3 ∧ BC = sqrt(7) ∧ CD = sqrt(7) ∧ angle_DAB = 60

def semigircle_gamma1 (AB E : ℝ) : Prop :=
  E ≠ 2 ∧ sqrt(4 - (E - 1)^2) ∈ Set.Icc 0 2

def semigircle_gamma2 (AD F : ℝ) : Prop :=
  F ≠ 3 ∧ sqrt(9 - (F - 1.5)^2) ∈ Set.Icc 0 3

def equilateral_triangle (C E F : Prop) : Prop :=
  ∀ x y z, dist x y = dist y z ∧ dist y z = dist z x

def area_of_triangle (side : ℝ) : ℝ :=
  (sqrt(3) / 4) * side^2

theorem area_triangle_CEF :
  quadrilateral_ABCD 2 3 (sqrt 7) (sqrt 7) 60 →
  semigircle_gamma1 2 E →
  semigircle_gamma2 3 F →
  equilateral_triangle C E F →
  area_of_triangle (dist E F) = (277 * sqrt(3)) / 76 :=
by
  intros h1 h2 h3 h4
  sorry

end area_triangle_CEF_l82_82975


namespace simplify_f_l82_82536

noncomputable def f (α : ℝ) : ℝ :=
  (Real.sin (α - 3 * Real.pi) * Real.cos (2 * Real.pi - α) * Real.sin (-α + 3 / 2 * Real.pi)) /
  (Real.cos (-Real.pi - α) * Real.sin (-Real.pi - α))

theorem simplify_f (α : ℝ) (h : Real.sin (α - 3 / 2 * Real.pi) = 1 / 5) : f α = -1 / 5 := by
  sorry

end simplify_f_l82_82536


namespace midpoint_polar_coordinates_correct_l82_82246

noncomputable def midpoint_polar_coordinates (r₁ θ₁ r₂ θ₂ : ℝ) (h₁ : r₁ > 0) (h₂ : r₂ > 0) (hθ₁ : 0 ≤ θ₁) (hθ₂ : θ₂ < 2 * π) : (ℝ × ℝ) :=
  let x₁ := r₁ * Float.cos θ₁
  let y₁ := r₁ * Float.sin θ₁
  let x₂ := r₂ * Float.cos θ₂
  let y₂ := r₂ * Float.sin θ₂
  let mx := (x₁ + x₂) / 2
  let my := (y₁ + y₂) / 2
  let mr := Float.sqrt (mx * mx + my * my)
  let mθ := Float.atan2 my mx
  (mr, mθ)

theorem midpoint_polar_coordinates_correct :
  midpoint_polar_coordinates 10 (Float.pi / 6) 10 (11 * Float.pi / 6) 10 10 (by norm_num) (by norm_num) (by norm_num) (by norm_num) = (10, Float.pi / 3) :=
sorry

end midpoint_polar_coordinates_correct_l82_82246


namespace sin_alpha_beta_lt_l82_82161

theorem sin_alpha_beta_lt (α β : ℝ) (hα : 0 < α) (hα' : α < π / 2) (hβ : 0 < β) (hβ' : β < π / 2) :
  sin (α + β) < sin α + sin β :=
sorry

end sin_alpha_beta_lt_l82_82161


namespace jane_additional_miles_l82_82273

noncomputable def additional_miles 
  (initial_distance : ℕ) 
  (initial_speed : ℕ) 
  (additional_speed : ℕ) 
  (total_average_speed : ℕ) : ℕ :=
  let initial_time := (initial_distance / initial_speed : ℚ)
  let h : ℚ := ((initial_distance / initial_speed) + h)
  let total_distance := (initial_distance + additional_speed * h)
  let total_time := initial_time + h
  let avg_speed := (total_distance / total_time) 
  let needed_h := 1/2
  additional_speed * needed_h

theorem jane_additional_miles : 
  additional_miles 20 40 80 60 = 40 := 
by 
  -- Proof omitted 
  sorry

end jane_additional_miles_l82_82273


namespace battery_life_in_standby_l82_82314

noncomputable def remaining_battery_life (b_s : ℝ) (b_a : ℝ) (t_total : ℝ) (t_active : ℝ) : ℝ :=
  let standby_rate := 1 / b_s
  let active_rate := 1 / b_a
  let standby_time := t_total - t_active
  let consumption_active := t_active * active_rate
  let consumption_standby := standby_time * standby_rate
  let total_consumption := consumption_active + consumption_standby
  let remaining_battery := 1 - total_consumption
  remaining_battery * b_s

theorem battery_life_in_standby :
  remaining_battery_life 30 4 10 1.5 = 10.25 := sorry

end battery_life_in_standby_l82_82314


namespace arithmetic_sequence_common_difference_divisible_by_p_l82_82286

theorem arithmetic_sequence_common_difference_divisible_by_p 
  (n : ℕ) (a : ℕ → ℕ) (h1 : n ≥ 2021) (h2 : ∀ i j, 1 ≤ i → i < j → j ≤ n → a i < a j) 
  (h3 : a 1 > 2021) (h4 : ∀ i, 1 ≤ i → i ≤ n → Nat.Prime (a i)) : 
  ∀ p, Nat.Prime p → p < 2021 → ∃ d, (∀ m, 2 ≤ m → a m = a 1 + (m - 1) * d) ∧ p ∣ d := 
sorry

end arithmetic_sequence_common_difference_divisible_by_p_l82_82286


namespace total_reading_materials_l82_82997

theorem total_reading_materials 
  (magazines : ℕ) 
  (newspapers : ℕ) 
  (h_magazines : magazines = 425) 
  (h_newspapers : newspapers = 275) : 
  magazines + newspapers = 700 := 
by 
  sorry

end total_reading_materials_l82_82997


namespace sculpture_cost_in_chinese_yuan_l82_82324

theorem sculpture_cost_in_chinese_yuan
  (usd_to_nad : ℝ)
  (usd_to_cny : ℝ)
  (cost_nad : ℝ)
  (h1 : usd_to_nad = 8)
  (h2 : usd_to_cny = 5)
  (h3 : cost_nad = 160) :
  (cost_nad / usd_to_nad) * usd_to_cny = 100 :=
by
  sorry

end sculpture_cost_in_chinese_yuan_l82_82324


namespace standard_eq_and_min_MN_l82_82569

-- Definitions of given conditions
def ellipse (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) :=
  ∀ (x y : ℝ), (x^2 / a^2 + y^2 / b^2 = 1)
  
def eccentricity (e : ℝ) :=
  e * e = (5*e - 2) / 2

def max_distance_to_focus (a b c : ℝ) :=
  a + c = 3 ∧ c / a = 1 / 2 ∧ a^2 - b^2 = c^2

-- Theorem to be proven
theorem standard_eq_and_min_MN :
  ∃ (a b : ℝ) (h1: 0 < a) (h2: 0 < b),
    ellipse a b h1 h2 ∧ max_distance_to_focus a b (a / 2) ∧ a = 2 ∧ b = sqrt 3 ∧
    (∀ P, ∃ M N, segment_MN_minimal P M N) := sorry

-- Note: Here, proofs that involve further definitions like segment_MN_minimal would be formalized,
-- but it's indicated through the keyword "sorry".

end standard_eq_and_min_MN_l82_82569


namespace camryn_practice_days_l82_82854

theorem camryn_practice_days :
  ∃ (d : ℕ), (d = Nat.lcmList [11, 3, 7, 13, 5]) ∧ d = 15015 :=
by
  use Nat.lcmList [11, 3, 7, 13, 5]
  split
  . rfl
  . sorry

end camryn_practice_days_l82_82854


namespace even_three_digit_numbers_sum_tens_units_14_l82_82213

theorem even_three_digit_numbers_sum_tens_units_14 : 
  ∃ n : ℕ, 
  (100 ≤ n ∧ n < 1000) ∧ 
  (n % 2 = 0) ∧
  (let t := (n / 10) % 10 in let u := n % 10 in t + u = 14) ∧
  n = 18 := sorry

end even_three_digit_numbers_sum_tens_units_14_l82_82213


namespace find_cos_alpha_find_beta_l82_82551

-- Definitions of the given conditions
variables (α β : ℝ)
-- Conditions
def alpha_acute : Prop := 0 < α ∧ α < π / 2
def beta_acute : Prop := 0 < β ∧ β < π / 2
def sin_alpha_min_pi_third : Prop := sin (α - π / 3) = 3 * sqrt 3 / 14
def cos_alpha_plus_beta : Prop := cos (α + β) = -11 / 14

-- Problem Statement
theorem find_cos_alpha (hα : alpha_acute α) (hβ : beta_acute β)
  (h1 : sin_alpha_min_pi_third α) (h2 : cos_alpha_plus_beta α β) :
  cos α = 1 / 7 := 
sorry

theorem find_beta (hα : alpha_acute α) (hβ : beta_acute β)
  (h1 : sin_alpha_min_pi_third α) (h2 : cos_alpha_plus_beta α β) (h_cos_alpha : cos α = 1 / 7) :
  β = π / 3 :=
sorry

end find_cos_alpha_find_beta_l82_82551


namespace hyperbola_eq_with_given_eccentricity_and_shared_focus_l82_82715

theorem hyperbola_eq_with_given_eccentricity_and_shared_focus 
  (eccentricity_hyperbola : ℝ)
  (ellipse_a : ℝ)
  (ellipse_b : ℝ)
  (focus : ℝ)
  (hyperbola_a : ℝ)
  (hyperbola_b_sq : ℝ) :
  let c := (sqrt (ellipse_a^2 - ellipse_b^2)),
      a := c / eccentricity_hyperbola,
      b_sq := c^2 - a^2 in
  (c = focus) ∧
  (b_sq = hyperbola_b_sq) ∧
  (a = hyperbola_a) → 
  a = 2 ∧ b_sq = 12 → (∃ (x y : ℝ), 
  (x^2)/(hyperbola_a^2) - (y^2)/(hyperbola_b_sq) = 1) :=
by
  sorry

end hyperbola_eq_with_given_eccentricity_and_shared_focus_l82_82715


namespace competition_arrangements_l82_82511

noncomputable def count_arrangements (students : Fin 4) (events : Fin 3) : Nat :=
  -- The actual counting function is not implemented
  sorry

theorem competition_arrangements (students : Fin 4) (events : Fin 3) :
  let count := count_arrangements students events
  (∃ (A B C D : Fin 4), 
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ 
    B ≠ C ∧ B ≠ D ∧ 
    C ≠ D ∧ 
    (A ≠ 0) ∧ 
    count = 24) := sorry

end competition_arrangements_l82_82511


namespace snooker_tournament_total_cost_l82_82829

def VIP_cost : ℝ := 45
def GA_cost : ℝ := 20
def total_tickets_sold : ℝ := 320
def vip_and_general_admission_relationship := 276

def total_cost_of_tickets : ℝ := 6950

theorem snooker_tournament_total_cost 
  (V G : ℝ)
  (h1 : VIP_cost * V + GA_cost * G = total_cost_of_tickets)
  (h2 : V + G = total_tickets_sold)
  (h3 : V = G - vip_and_general_admission_relationship) : 
  VIP_cost * V + GA_cost * G = total_cost_of_tickets := 
by {
  sorry
}

end snooker_tournament_total_cost_l82_82829


namespace concyclic_endpoints_l82_82154

-- Given a triangle ABC
variables {A B C A' A'' B' B'' C' C'' O : Type}

-- Conditions
variables (triangle_ABC : Triangle A B C)
variables (segment_A : Segment A' A'')
variables (segment_B : Segment B' B'')
variables (segment_C : Segment C' C'')

-- Incenter of the triangle
variables (incenter_O : Incenter triangle_ABC)

-- Concyclic endpoints
theorem concyclic_endpoints (h : ∀ x ∈ {A', A'', B', B'', C', C''}, IsEndpointOfSegment x triangle_ABC) : Concyclic {A', A'', B', B'', C', C''} :=
sorry

end concyclic_endpoints_l82_82154


namespace angle_bisector_of_ABC_l82_82093

-- Defining points and circles
variables (A B C C_1 : Type)
noncomputable def S_A : Type := { p : Type // p = A ∨ p = C }
noncomputable def S_B : Type := { p : Type // p = B ∨ p = C }
noncomputable def S_Center : Type := { O : Type // ∀ P Q : S_A, ∀ R S : S_B, line_through O A ∧ line_through O B }

def S : Type := { t : Type // tangential_to t S_A ∧ tangential_to t S_B ∧ tangent_to_line t (segment A B) C_1 }

-- Main theorem stating the condition
theorem angle_bisector_of_ABC (h1 : passes_through S_A A) (h2 : passes_through S_A C)
  (h3 : passes_through S_B B) (h4 : passes_through S_B C)
  (h5 : centers_lie_on_line S_Center A B)
  (h6 : tangential_to S S_A) (h7 : tangential_to S S_B)
  (h8 : tangent_to_line S (segment A B) C_1) :
  is_angle_bisector (segment C C_1) (triangle A B C) :=
sorry

end angle_bisector_of_ABC_l82_82093


namespace minimum_omega_l82_82577

noncomputable def f (ω x varphi : ℝ) := real.cos (ω * x + varphi)

theorem minimum_omega
  (T ω varphi : ℝ)
  (hω : ω > 0)
  (hvarphi : 0 < varphi ∧ varphi < π)
  (hT : T = 2 * π / ω)
  (hT_f : f ω T varphi = 1/2)
  (x : ℝ)
  (hx : x = 7 * π / 3)
  (hx_crit : ∀ x, is_critical_point (f ω x varphi) x)
  : ω = 2 / 7 :=
sorry

end minimum_omega_l82_82577


namespace sin_inequality_l82_82339

theorem sin_inequality (n : ℕ) (hn : n > 0) (θ : ℝ) : |Real.sin(n * θ)| ≤ n * |Real.sin(θ)| :=
sorry

end sin_inequality_l82_82339


namespace opposite_face_A_is_E_l82_82818

-- Axiomatically defining the basic conditions from the problem statement.

-- We have six labels for the faces of a net
inductive Face : Type
| A | B | C | D | E | F

open Face

-- Define the adjacency relation
def adjacent (x y : Face) : Prop :=
  (x = A ∧ y = B) ∨ (x = A ∧ y = D) ∨ (x = B ∧ y = A) ∨ (x = D ∧ y = A)

-- Define the "not directly attached" relationship
def not_adjacent (x y : Face) : Prop :=
  ¬adjacent x y

-- Given the conditions in the problem statement
axiom condition1 : adjacent A B
axiom condition2 : adjacent A D
axiom condition3 : not_adjacent A E

-- The proof objective is to show that E is the face opposite to A
theorem opposite_face_A_is_E : ∃ (F : Face), 
  (∀ x : Face, adjacent A x ∨ not_adjacent A x) → (∀ y : Face, adjacent A y ↔ y ≠ E) → E = F :=
sorry

end opposite_face_A_is_E_l82_82818


namespace imaginary_part_of_z_l82_82719

-- Let 'z' be the complex number \(\frac {2i}{1-i}\)
noncomputable def z : ℂ := (2 * Complex.I) / (1 - Complex.I)

theorem imaginary_part_of_z :
  z.im = 1 :=
sorry

end imaginary_part_of_z_l82_82719


namespace average_speed_of_trip_l82_82443

theorem average_speed_of_trip (d1 d2 s1 s2 : ℕ)
  (h1 : d1 = 30) (h2 : d2 = 30)
  (h3 : s1 = 60) (h4 : s2 = 30) :
  (d1 + d2) / (d1 / s1 + d2 / s2) = 40 :=
by sorry

end average_speed_of_trip_l82_82443


namespace N_is_composite_l82_82874

def N : ℕ := 7 * 9 * 13 + 2020 * 2018 * 2014

theorem N_is_composite : ¬ Prime N :=
by {
  sorry
}

end N_is_composite_l82_82874


namespace barry_more_votes_than_joey_l82_82607

theorem barry_more_votes_than_joey {M B J X : ℕ} 
  (h1 : M = 66)
  (h2 : J = 8)
  (h3 : M = 3 * B)
  (h4 : B = 2 * (J + X)) :
  B - J = 14 := by
  sorry

end barry_more_votes_than_joey_l82_82607


namespace two_digit_number_l82_82740

theorem two_digit_number (x y : ℕ) (h1 : x + y = 7) (h2 : (x + 2) + 10 * (y + 2) = 2 * (x + 10 * y) - 3) : (10 * y + x) = 25 :=
by
  sorry

end two_digit_number_l82_82740


namespace hexagon_area_ratio_l82_82644

theorem hexagon_area_ratio (ABCDEF : Type) [regular_hexagon ABCDEF]
  (P Q R S T U : ABCDEF → Type)
  (hP : ∀ A B : ABCDEF, ratio A P B = 1 / 3)
  (hQ : ∀ B C : ABCDEF, ratio B Q C = 1 / 3)
  (hR : ∀ C D : ABCDEF, ratio C R D = 1 / 3)
  (hS : ∀ D E : ABCDEF, ratio D S E = 1 / 3)
  (hT : ∀ E F : ABCDEF, ratio E T F = 1 / 3)
  (hU : ∀ F A : ABCDEF, ratio F U A = 1 / 3)
  (hex_bound : smaller_regular_hexagon_bound ABCDEF P Q R S T U):
  let m := (4 : ℤ)
  let n := (9 : ℤ)
  m + n = 13 := 
sorry

end hexagon_area_ratio_l82_82644


namespace num_roots_Q7_right_half_plane_l82_82894

noncomputable def Q7 (z : ℂ) : ℂ := z^7 - 2 * z - 5

theorem num_roots_Q7_right_half_plane :
  ∃! (n : ℕ), (∃ (R : ℝ), ∀ (z : ℂ), (Re(z) > 0) → Q7(z) = 0 → n = 3) :=
sorry

end num_roots_Q7_right_half_plane_l82_82894


namespace evaluate_expression_l82_82882

theorem evaluate_expression : sqrt (7 + 4 * sqrt 3) - sqrt (7 - 4 * sqrt 3) = 2 * sqrt 3 := 
  sorry

end evaluate_expression_l82_82882


namespace find_third_side_length_l82_82247

noncomputable def cos θ : ℝ :=
  if θ = 150 then - (Real.sqrt 3) / 2 else 0 -- This is an ad-hoc definition for this specific problem.

theorem find_third_side_length :
  ∀ (a b : ℝ), a = 6 → b = 9 → cos 150 = -(Real.sqrt 3) / 2 →
  let c := Real.sqrt (a^2 + b^2 - 2 * a * b * cos 150) in
  c = Real.sqrt (117 + 54 * Real.sqrt 3) :=
begin
  intros a b ha hb hcos,
  rw [ha, hb],
  suffices : Real.sqrt (6^2 + 9^2 - 2 * 6 * 9 * cos 150) = Real.sqrt (117 + 54 * Real.sqrt 3),
  exact this,
  simp [hcos], -- Use the cosine value and simplify.
  sorry,
end

end find_third_side_length_l82_82247


namespace net_change_in_price_l82_82964

/-- If the price of a book is decreased by 30%, then increased by 20%, and finally taxed at a rate
    of 15%, the net change in the price is a decrease of 3.4% from the original price. -/
theorem net_change_in_price (P : ℝ) :
  let decreased_price := P - 0.30 * P,
      increased_price := decreased_price + 0.20 * decreased_price,
      final_price := increased_price + 0.15 * increased_price
  in final_price - P = -0.034 * P :=
by sorry

end net_change_in_price_l82_82964


namespace siding_cost_l82_82343

noncomputable def front_wall_width : ℝ := 10
noncomputable def front_wall_height : ℝ := 8
noncomputable def triangle_base : ℝ := 10
noncomputable def triangle_height : ℝ := 4
noncomputable def panel_area : ℝ := 100
noncomputable def panel_cost : ℝ := 30

theorem siding_cost :
  let front_wall_area := front_wall_width * front_wall_height
  let triangle_area := (1 / 2) * triangle_base * triangle_height
  let total_area := front_wall_area + triangle_area
  let panels_needed := total_area / panel_area
  let total_cost := panels_needed * panel_cost
  total_cost = 30 := sorry

end siding_cost_l82_82343


namespace find_a_l82_82618

-- Define point
structure Point where
  x : ℝ
  y : ℝ

-- Define curves
def C1 (a x : ℝ) : ℝ := a * x^3 + 1
def C2 (P : Point) : Prop := P.x^2 + P.y^2 = 5 / 2

-- Define the tangent slope function for curve C1
def tangent_slope_C1 (a x : ℝ) : ℝ := 3 * a * x^2

-- State the problem that we need to prove
theorem find_a (a x₀ y₀ : ℝ) (h1 : y₀ = C1 a x₀) (h2 : C2 ⟨x₀, y₀⟩) (h3 : y₀ = 3 * a * x₀^3) 
  (ha_pos : 0 < a) : a = 4 := 
  by
    sorry

end find_a_l82_82618


namespace intersection_in_second_quadrant_l82_82010

theorem intersection_in_second_quadrant (k : ℝ) (x y : ℝ) 
  (hk : 0 < k) (hk2 : k < 1/2) 
  (h1 : k * x - y = k - 1) 
  (h2 : k * y - x = 2 * k) : 
  x < 0 ∧ y > 0 := 
sorry

end intersection_in_second_quadrant_l82_82010


namespace greatest_integer_b_l82_82507

theorem greatest_integer_b (b : ℤ) : (b : ℝ) < real.sqrt 40 → b ≤ 6 := by
  sorry

end greatest_integer_b_l82_82507


namespace extremum_at_2_implies_a_neg2_monotonicity_of_f_l82_82193

noncomputable def f (x a : ℝ) := -a * Real.log x + (1/2) * x^2 + (a - 1) * x + 1 

theorem extremum_at_2_implies_a_neg2 (a : ℝ) : 
  (∃ x : ℝ, x = 2 ∧ deriv (λ x : ℝ, f x a) x = 0) → a = -2 := sorry

theorem monotonicity_of_f (a : ℝ) (ha : a < 0) : 
  (∀ x : ℝ, 0 < x) → 
  (if -1 < a then 
    (∀ x : ℝ, (0 < x ∧ x ≤ -a → deriv (λ x : ℝ, f x a) x > 0) ∧ 
       (1 ≤ x → deriv (λ x : ℝ, f x a) x > 0) ∧ 
       (-a ≤ x ∧ x ≤ 1 → deriv (λ x : ℝ, f x a) x < 0)) 
  else if a = -1 then 
    (∀ x : ℝ, 0 < x → deriv (λ x : ℝ, f x a) x > 0) 
  else 
    (∀ x : ℝ, (0 < x ∧ x ≤ 1 → deriv (λ x : ℝ, f x a) x > 0) ∧ 
       (1 ≤ x ∧ x ≤ -a → deriv (λ x : ℝ, f x a) x < 0) ∧ 
       (-a ≤ x → deriv (λ x : ℝ, f x a) x > 0))) := sorry

end extremum_at_2_implies_a_neg2_monotonicity_of_f_l82_82193


namespace probability_even_blue_and_prime_yellow_l82_82000

def is_even (n : ℕ) : Prop := n % 2 = 0
def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem probability_even_blue_and_prime_yellow :
  (∑ b in Finset.filter is_even (Finset.range 1 9), ∑ y in Finset.filter is_prime (Finset.range 1 11), (1 : ℚ) / 80) = 1 / 5 :=
by
  sorry

end probability_even_blue_and_prime_yellow_l82_82000


namespace last_two_nonzero_digits_of_70_fact_l82_82721

theorem last_two_nonzero_digits_of_70_fact : 
  let M := 70! / 10^16
  in M % 100 = 44 := 
sorry

end last_two_nonzero_digits_of_70_fact_l82_82721


namespace intersection_P_Q_eq_Q_l82_82665

def P : Set ℝ := { x | x < 2 }
def Q : Set ℝ := { x | x^2 ≤ 1 }

theorem intersection_P_Q_eq_Q : P ∩ Q = Q := 
sorry

end intersection_P_Q_eq_Q_l82_82665


namespace tangent_circle_property_l82_82998

noncomputable def midpoint (A B : Point) : Point := 
  mk_point ((A.x + B.x) / 2) ((A.y + B.y) / 2)

theorem tangent_circle_property
  (O A B M U T P : Point)
  (C1 : Circle)
  (C2 : Circle)
  (R r PA PB PT : ℝ)
  (hO : O = center C1)
  (hOM : OM = diameter C2)
  (hAB : ¬(AB = diameter C1))
  (hM : M = midpoint A B)
  (hC2 : U = center C2)
  (hT : T ∈ C2)
  (hTang : ∀ Q, Q ∈ tangent_line T C2 → Q ∈ C1 → Q = P ∨ Q ≠ P)
  : PA^2 + PB^2 = 4 * PT^2 :=
sorry

end tangent_circle_property_l82_82998


namespace find_f_2022_l82_82360

-- Define a function f that satisfies the given conditions
def f (x : ℝ) : ℝ := sorry

axiom f_condition : ∀ a b : ℝ, f ((a + 2 * b) / 3) = (f(a) + 2 * f(b)) / 3
axiom f_1 : f 1 = 1
axiom f_4 : f 4 = 7

-- The main theorem to prove
theorem find_f_2022 : f 2022 = 4043 :=
by
  sorry

end find_f_2022_l82_82360


namespace alice_wins_probability_l82_82613

theorem alice_wins_probability : 
  ∀ (voters : ℕ) (initial_votes_alice : ℕ) (initial_votes_celia : ℕ) 
    (bob_votes_alice : ℕ) (remaining_voters : ℕ),
  voters = 2019 →
  initial_votes_alice = 1 →
  initial_votes_celia = 1 →
  bob_votes_alice = 1 →
  remaining_voters = 2016 →
  let total_initial_votes_alice := initial_votes_alice + bob_votes_alice in
  let total_initial_votes_celia := initial_votes_celia in
  let final_votes_alice := total_initial_votes_alice + remaining_voters * (total_initial_votes_alice/(total_initial_votes_alice + total_initial_votes_celia)) in
  let final_votes_celia := total_initial_votes_celia + remaining_voters * (total_initial_votes_celia/(total_initial_votes_alice + total_initial_votes_celia)) in
  (final_votes_alice > final_votes_celia) ↔ (1513/2017) := 
  sorry

end alice_wins_probability_l82_82613


namespace total_attendance_l82_82836

-- Defining the given conditions
def adult_ticket_cost : ℕ := 8
def child_ticket_cost : ℕ := 1
def total_amount_collected : ℕ := 50
def number_of_child_tickets : ℕ := 18

-- Formulating the proof problem
theorem total_attendance (A : ℕ) (C : ℕ) (H1 : C = number_of_child_tickets)
  (H2 : adult_ticket_cost * A + child_ticket_cost * C = total_amount_collected) :
  A + C = 22 := by
  sorry

end total_attendance_l82_82836


namespace distance_AB_l82_82307

noncomputable def polar_distance (r1 r2 : ℝ) (phi1 phi2 : ℝ) : ℝ :=
  sqrt (r1^2 + r2^2 - 2 * r1 * r2 * Real.cos (phi1 - phi2))

theorem distance_AB : 
  polar_distance 4 10 φ1 φ2 = sqrt (116 - 40 * Real.sqrt 2) :=
by 
  have h : φ1 - φ2 = Real.pi / 4 := sorry
  rw [polar_distance, h, Real.cos_pi_div_four]
  sorry

end distance_AB_l82_82307


namespace intersection_of_A_and_B_l82_82172

def A : Set ℤ := { x | -4 ≤ x ∧ x ≤ 3 }
def B : Set ℕ := { x | x + 1 < 3 }

theorem intersection_of_A_and_B : A ∩ B = {0, 1} := by
  sorry

end intersection_of_A_and_B_l82_82172


namespace friendship_arrangements_count_l82_82122

noncomputable def number_of_friendship_arrangements : ℕ :=
  1125

-- Conditions
def eight_individuals {α : Type*} (Alex Ben Charlie Derek Evelyn Fiona George Hannah : α) := 
  true

def equal_number_of_friends {α : Type*} (f : α → α → Prop) (lst : list α) :=
  ∀ (x y : α), (x ∈ lst) ∧ (y ∈ lst) → (list.countp (f x) lst = list.countp (f y) lst)

def has_some_but_not_all_others {α : Type*} (f : α → α → Prop) (lst : list α) :=
  ∀ x ∈ lst, 1 ≤ list.countp (f x) lst ∧ list.countp (f x) lst < (lst.length - 1)

def no_friends_outside_group {α : Type*} (f : α → α → Prop) (lst : list α) := 
  true

theorem friendship_arrangements_count {α : Type*} (Alex Ben Charlie Derek Evelyn Fiona George Hannah : α)
    (f : α → α → Prop)
    (lst := [Alex, Ben, Charlie, Derek, Evelyn, Fiona, George, Hannah]) :
  eight_individuals Alex Ben Charlie Derek Evelyn Fiona George Hannah →
  equal_number_of_friends f lst →
  has_some_but_not_all_others f lst →
  no_friends_outside_group f lst →
  number_of_friendship_arrangements = 1125 :=
by
  intros _ _ _ _
  sorry

end friendship_arrangements_count_l82_82122


namespace profit_due_to_faulty_meter_l82_82020

-- Define the conditions
def cost_price : ℝ := 1 -- assuming cost price per kg as 1 unit of currency
def faulty_weight : ℝ := 900 -- 900 grams instead of 1000 grams
def expected_weight : ℝ := 1000 -- 1000 grams per kilogram

-- Calculating the shortfall in weight
def shortfall_weight : ℝ := expected_weight - faulty_weight 
def shortfall_ratio : ℝ := shortfall_weight / expected_weight -- 10%

-- Define the profit percent
def profit_percent : ℝ := (shortfall_weight / expected_weight) * 100

-- Theorem stating the profit percentage
theorem profit_due_to_faulty_meter : profit_percent = 10 :=
by
  -- Here will be the proof which is skipped using sorry
  sorry

end profit_due_to_faulty_meter_l82_82020


namespace number_of_zeros_expansion_l82_82950

noncomputable def big_number := 10 ^ 15 - 3

theorem number_of_zeros_expansion : 
  let x := big_number in 
  (number_of_zeros (x ^ 2) = 15) :=
by
  sorry

end number_of_zeros_expansion_l82_82950


namespace interest_rate_A_to_B_l82_82450

theorem interest_rate_A_to_B :
  ∀ (principal : ℝ) (rate_C : ℝ) (time : ℝ) (gain_B : ℝ) (interest_C : ℝ) (interest_A : ℝ),
    principal = 3500 →
    rate_C = 0.13 →
    time = 3 →
    gain_B = 315 →
    interest_C = principal * rate_C * time →
    gain_B = interest_C - interest_A →
    interest_A = principal * (R / 100) * time →
    R = 10 := by
  sorry

end interest_rate_A_to_B_l82_82450


namespace money_left_after_spending_l82_82315

noncomputable def Mildred_spent_on_fruits := 25
noncomputable def Candice_spent_on_vegetables := 35
noncomputable def Joseph_original_clothes_price := 45
noncomputable def Discount_rate := 0.80
noncomputable def Sales_tax_rate := 0.10
noncomputable def Gift_cost := 30
noncomputable def Total_money_given := 180

theorem money_left_after_spending :
  let total_spent_on_fruits_and_vegetables := Mildred_spent_on_fruits + Candice_spent_on_vegetables
  let discounted_price := Discount_rate * Joseph_original_clothes_price
  let sales_tax := Sales_tax_rate * discounted_price
  let total_spent_on_clothes := discounted_price + sales_tax
  let total_spent := total_spent_on_fruits_and_vegetables + total_spent_on_clothes + Gift_cost
  Total_money_given - total_spent = 50.40 :=
by
  sorry

end money_left_after_spending_l82_82315


namespace monotonic_intervals_of_f_l82_82182

theorem monotonic_intervals_of_f (f : ℝ → ℝ) 
    (hf : ∀ x, deriv f x = 2 * Real.cos (2 * x + Real.pi / 6)) :
    (∀ x ∈ set.Icc 0 (Real.pi / 6), 0 ≤ deriv f x) ∧ 
    (∀ x ∈ set.Icc (2 * Real.pi / 3) Real.pi, 0 ≤ deriv f x) :=
by
  sorry

end monotonic_intervals_of_f_l82_82182


namespace max_n_for_factorable_polynomial_l82_82519

theorem max_n_for_factorable_polynomial : 
  ∃ n : ℤ, (∀ A B : ℤ, AB = 108 → n = 6 * B + A) ∧ n = 649 :=
by
  sorry

end max_n_for_factorable_polynomial_l82_82519


namespace janet_roses_l82_82274

def total_flowers (used_flowers extra_flowers : Nat) : Nat :=
  used_flowers + extra_flowers

def number_of_roses (total tulips : Nat) : Nat :=
  total - tulips

theorem janet_roses :
  ∀ (used_flowers extra_flowers tulips : Nat),
  used_flowers = 11 → extra_flowers = 4 → tulips = 4 →
  number_of_roses (total_flowers used_flowers extra_flowers) tulips = 11 :=
by
  intros used_flowers extra_flowers tulips h_used h_extra h_tulips
  rw [h_used, h_extra, h_tulips]
  -- proof steps skipped
  sorry

end janet_roses_l82_82274


namespace greatest_of_5_consecutive_integers_l82_82006

theorem greatest_of_5_consecutive_integers (m n : ℤ) (h : 5 * n + 10 = m^3) : (n + 4) = 202 := by
sorry

end greatest_of_5_consecutive_integers_l82_82006


namespace range_of_a_l82_82739

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, ¬ (x^2 - 2 * x + 3 ≤ a^2 - 2 * a - 1)) ↔ (-1 < a ∧ a < 3) :=
sorry

end range_of_a_l82_82739


namespace hank_donated_percentage_l82_82943

variable (A_c D_c A_b D_b A_l D_t D_l p : ℝ) (h1 : A_c = 100) (h2 : D_c = 0.90 * A_c)
variable (h3 : A_b = 80) (h4 : D_b = 0.75 * A_b) (h5 : A_l = 50) (h6 : D_t = 200)

theorem hank_donated_percentage :
  D_l = D_t - (D_c + D_b) → 
  p = (D_l / A_l) * 100 → 
  p = 100 :=
by
  sorry

end hank_donated_percentage_l82_82943


namespace hit_rate_of_person_B_l82_82399

theorem hit_rate_of_person_B : 
  ∀ (P_A P_union : ℝ), 
  P_A = 0.4 → P_union = 0.7 →
  (∀ P_A P_B, P_union = P_A + P_B - P_A * P_B) →
  ∃ (P_B : ℝ), P_B = 0.5 :=
by
  intros P_A P_union hPA hPUnion hIndep
  use 0.5
  sorry

end hit_rate_of_person_B_l82_82399


namespace zero_of_log_function_l82_82378

noncomputable def log_base_2 (x : ℝ) : ℝ := Real.log x / Real.log 2

theorem zero_of_log_function : ∃ x : ℝ, log_base_2 (3 - 2 * x) = 0 ↔ x = 1 :=
by
  -- We define log_base_2(3 - 2 * x) and then find x for which the equation equals zero
  sorry

end zero_of_log_function_l82_82378


namespace number_of_zeros_expansion_l82_82949

noncomputable def big_number := 10 ^ 15 - 3

theorem number_of_zeros_expansion : 
  let x := big_number in 
  (number_of_zeros (x ^ 2) = 15) :=
by
  sorry

end number_of_zeros_expansion_l82_82949


namespace smallest_value_of_a_l82_82734

noncomputable def polynomial : Polynomial ℝ := Polynomial.C 1806 - Polynomial.C b * Polynomial.X + Polynomial.C a * (Polynomial.X ^ 2) - Polynomial.X ^ 3

theorem smallest_value_of_a (a b : ℝ) (r1 r2 r3 : ℝ) 
  (h_roots : ∀ x, Polynomial.eval x polynomial = 0 ↔ x = r1 ∨ x = r2 ∨ x = r3)
  (h_factors : 1806 = r1 * r2 * r3)
  (h_pos : r1 > 0 ∧ r2 > 0 ∧ r3 > 0)
  (h_int : r1 ∈ ℤ ∧ r2 ∈ ℤ ∧ r3 ∈ ℤ) :
  a = r1 + r2 + r3 → a = 56 :=
by 
  sorry

end smallest_value_of_a_l82_82734


namespace factor_expression_l82_82884

theorem factor_expression :
  (8 * x^6 + 36 * x^4 - 5) - (2 * x^6 - 6 * x^4 + 5) = 2 * (3 * x^6 + 21 * x^4 - 5) :=
by
  sorry

end factor_expression_l82_82884


namespace alyssa_money_after_movies_and_carwash_l82_82070

theorem alyssa_money_after_movies_and_carwash : 
  ∀ (allowance spent earned : ℕ), 
  allowance = 8 → 
  spent = allowance / 2 → 
  earned = 8 → 
  (allowance - spent + earned = 12) := 
by 
  intros allowance spent earned h_allowance h_spent h_earned 
  rw [h_allowance, h_spent, h_earned] 
  simp 
  sorry

end alyssa_money_after_movies_and_carwash_l82_82070


namespace percentage_calculation_l82_82227

theorem percentage_calculation
  (x : ℝ)
  (hx : x = 16)
  (h : 0.15 * 40 - (P * x) = 2) :
  P = 0.25 := by
  sorry

end percentage_calculation_l82_82227


namespace cosine_angle_between_ST_and_QR_l82_82394

variables {P Q R S T : Type*}
variables [AddCommGroup P] [VectorSpace ℝ P] [InnerProductSpace ℝ P]
variables (PQ PS PR PT PR : P)
variables (angle : ℝ)

-- Assume conditions
def conditions (h1 : (2 : ℝ) = 2)
               (h2 : (2 : ℝ) = 2)
               (h3 : (3 : ℝ) = 3)
               (h4 : (sqrt 7 : ℝ) = real.sqrt 7)
               (h5 : inner_product_space ℝ P PQ * inner_product_space ℝ P PS +
                    inner_product_space ℝ P PR * inner_product_space ℝ P PT = 3) : Prop :=
  true

-- Proof statement
theorem cosine_angle_between_ST_and_QR
  (PQ PS PR PT : P) (angle : ℝ)
  (h : conditions 
         (2 : ℝ) 
         (2 : ℝ) 
         (3 : ℝ) 
         (sqrt 7 : ℝ) 
         (inner_product_space ℝ P PQ * inner_product_space ℝ P PS + 
          inner_product_space ℝ P PR * inner_product_space ℝ P PT = 3)) : 
  cosine_angle ≠ 0 :=
sorry

end cosine_angle_between_ST_and_QR_l82_82394


namespace find_four_digit_numbers_l82_82521

theorem find_four_digit_numbers (x : ℕ) (hx : x ≥ 1000 ∧ x < 10000) (h7 : x % 7 = 0) :
  ∃ y : ℕ, 10 ≤ y ∧ y < 22 ∧ x = y^3 + y^2 :=
begin
  sorry
end

example := find_four_digit_numbers 1386 ⟨nat.ge_of_eq_decide_true rfl, nat.lt_of_succ_le_decide_true rfl⟩ rfl
example := find_four_digit_numbers 1200 ⟨nat.ge_of_eq_decide_true rfl, nat.lt_of_succ_le_decide_true rfl⟩ rfl

end find_four_digit_numbers_l82_82521


namespace even_function_increasing_condition_l82_82960

noncomputable def is_even_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f x = f (-x)

noncomputable def is_increasing_on_interval (f : ℝ → ℝ) (S : set ℝ) : Prop :=
∀ x y : ℝ, x ∈ S → y ∈ S → x < y → f x < f y

theorem even_function_increasing_condition (f : ℝ → ℝ) 
  (h1 : is_even_function f)
  (h2 : is_increasing_on_interval f (set.Iic (-1))) :
  f 2 < f (-1.5) ∧ f (-1.5) < f (-1) :=
by
  sorry

end even_function_increasing_condition_l82_82960


namespace intersect_AB_complement_UA_range_of_m_l82_82433

open Set

def U := {x : ℝ | 1 < x ∧ x < 7}
def A1 := {x : ℝ | 2 <= x ∧ x < 5}
def B1 := {x : ℝ | 3 * x - 7 >= 8 - 2 * x}
def A2 := {x : ℝ | -2 <= x ∧ x <= 7}
def B2 (m : ℝ) := {x : ℝ | m + 1 < x ∧ x < 2 * m - 1}

theorem intersect_AB :
  A1 ∩ B1 = {x : ℝ | 3 <= x ∧ x < 5} :=
sorry

theorem complement_UA :
  U \ A1 = {x : ℝ | (1 < x ∧ x < 2) ∨ (5 <= x ∧ x < 7)} :=
sorry

theorem range_of_m {m : ℝ} :
  (Union (B2 m) A2 = A2) → (m <= 4) :=
sorry

end intersect_AB_complement_UA_range_of_m_l82_82433


namespace problem_I_problem_II_l82_82198

noncomputable theory

-- Definitions and conditions
def f (a : ℝ) (x : ℝ) : ℝ := x - a^x
def g (a : ℝ) : ℝ := f a (Real.log a⁻¹ / Real.log a)  -- Based on t = log_a(1 / log a)

-- Problem (Ⅰ): When a = e, prove that for all x ≥ 0, if f(x) ≤ b - 1/2 x^2, then b ≥ 1
theorem problem_I (b : ℝ) : (∀ x : ℝ, 0 ≤ x → f Real.exp x ≤ b - (1/2) * x^2) → 1 ≤ b := sorry

-- Problem (Ⅱ): Prove that the minimum value of g(a) for a > 0, a ≠ 1 is -1
theorem problem_II : ∃ a : ℝ, 1 < a ∧ g a = -1 := sorry

end problem_I_problem_II_l82_82198


namespace tangent_line_at_1_range_of_m_l82_82929

noncomputable def f (x : ℝ) (a : ℝ) := (x^2 - 2 * x) * Real.log x + a * x^2 + 2
noncomputable def g (x : ℝ) (a : ℝ) := f x a - x - 2

theorem tangent_line_at_1 (x y : ℝ) (a : ℝ) (h_a : a = -1) 
    (tangent_eq : 3 * x + y = 4) : 
    tangent_eq = true := 
  sorry

theorem range_of_m (m a : ℝ) (h_a : a > 0) 
    (h_zero : ∀ x, g x a = 0) 
    (h_bound : ∀ x, e^(-2) < x ∧ x < e → g x a ≤ m) : 
    2 * e^2 - 3 * e ≤ m := 
  sorry

end tangent_line_at_1_range_of_m_l82_82929


namespace intersection_M_N_l82_82666

open Set Real

noncomputable def f (x : ℝ) : ℝ := Real.log (1 - abs x)

def M : Set ℝ := {x | x^2 - x ≤ 0}
def N : Set ℝ := {x | -1 < x ∧ x < 1}

theorem intersection_M_N : M ∩ N = {x | 0 ≤ x ∧ x < 1} :=
by
  sorry

end intersection_M_N_l82_82666


namespace total_infections_14_days_l82_82879

def f (x : ℕ) : ℕ := 2 * x^2 + 3 * x + 300

def total_infections (n : ℕ) : ℕ := (Finset.range n).sum (λ x, f (x + 1))

theorem total_infections_14_days : total_infections 14 = 5846 := 
by
  sorry

end total_infections_14_days_l82_82879


namespace car_and_truck_arrival_time_simultaneous_l82_82024

theorem car_and_truck_arrival_time_simultaneous {t_car t_truck : ℕ} 
    (h1 : t_car = 8 * 60 + 16) -- Car leaves at 08:16
    (h2 : t_truck = 9 * 60) -- Truck leaves at 09:00
    (h3 : t_car_arrive = 10 * 60 + 56) -- Car arrives at 10:56
    (h4 : t_truck_arrive = 12 * 60 + 20) -- Truck arrives at 12:20
    (h5 : t_truck_exit = t_car_exit + 2) -- Truck leaves tunnel 2 minutes after car
    : (t_car_exit + t_car_tunnel_time = 10 * 60) ∧ (t_truck_exit + t_truck_tunnel_time = 10 * 60) :=
  sorry

end car_and_truck_arrival_time_simultaneous_l82_82024


namespace N_is_composite_l82_82873

def N : ℕ := 7 * 9 * 13 + 2020 * 2018 * 2014

theorem N_is_composite : ¬ Prime N :=
by {
  sorry
}

end N_is_composite_l82_82873


namespace conference_partition_l82_82847

noncomputable theory

def partition_exists (n : ℕ) (G : simple_graph (fin n)) : Prop :=
  ∃ (partition : fin n → bool),
    ∀ (i : fin n), let other_group := G.neighbor_set i \ {j | partition j = partition i} in
    let same_group := G.neighbor_set i \ other_group in
    other_group.card ≥ same_group.card

theorem conference_partition {n : ℕ} (G : simple_graph (fin n)) : partition_exists n G :=
sorry

end conference_partition_l82_82847


namespace T_five_three_l82_82112

def T (a b : ℤ) : ℤ := 4 * a + 6 * b + 2

theorem T_five_three : T 5 3 = 40 := by
  sorry

end T_five_three_l82_82112


namespace min_distance_to_line_l82_82585

-- Define the conditions
def polar_eq (ρ θ : ℝ) : Prop := ρ * Math.sin(θ + π / 4) = sqrt 2 / 2

def parametric_circle (θ : ℝ) : ℝ × ℝ := (2 * Math.cos θ, -2 + 2 * Math.sin θ)

-- Define the Cartesian forms
def line_eq (x y : ℝ) : Prop := x + y - 1 = 0

def circle_eq (x y : ℝ) : Prop := x^2 + (y + 2)^2 = 4

-- Define the distance function
def distance_from_center (x y : ℝ) : ℝ :=
  abs (x - y - 1) / sqrt 2

-- Prove the minimum distance
theorem min_distance_to_line : ∀ x y : ℝ,
  circle_eq x y →
  line_eq x y →
  distance_from_center 0 (-2) = (3 * sqrt 2 / 2) - 2 :=
by
  sorry

end min_distance_to_line_l82_82585


namespace marcus_leah_together_l82_82761

def num_games_with_combination (n k : ℕ) : ℕ :=
  Nat.choose n k

def num_games_together (total_players players_per_game : ℕ) (games_with_each_combination: ℕ) : ℕ :=
  total_players / players_per_game * games_with_each_combination

/-- Prove that Marcus and Leah play 210 games together. -/
theorem marcus_leah_together :
  let total_players := 12
  let players_per_game := 6
  let total_games := num_games_with_combination total_players players_per_game
  let marc_per_game := total_games / 2
  let together_pcnt := 5 / 11
  together_pcnt * marc_per_game = 210 :=
by
  sorry

end marcus_leah_together_l82_82761


namespace iron_ball_radius_correct_l82_82465

noncomputable def iron_ball_radius (d h : ℝ) : ℝ :=
let V_cylinder := Real.pi * (d / 2) ^ 2 * h in
Real.cbrt ((3 * V_cylinder) / (4 * Real.pi))

theorem iron_ball_radius_correct :
  iron_ball_radius 32 9 = 12 :=
by
  sorry

end iron_ball_radius_correct_l82_82465


namespace possible_values_of_N_l82_82281

noncomputable def sum_of_remainders (N : ℕ) : ℕ :=
  (∑ q in Finset.range N, N % (q + 1))

theorem possible_values_of_N (N : ℕ) (hN : N > 1) (h_sum : sum_of_remainders N < N) : N = 2 :=
begin
  sorry
end

end possible_values_of_N_l82_82281


namespace dhoni_spent_on_rent_l82_82509

-- Define the conditions
variable (last_month_earnings : ℝ)
variable (spent_on_rent_percent : ℝ)
variable (spent_on_dishwasher_percent : ℝ)
variable (leftover_earnings_percent : ℝ)

-- Given conditions
def conditions : Prop := 
  last_month_earnings = 100 ∧ 
  spent_on_dishwasher_percent = spent_on_rent_percent - 10 ∧
  leftover_earnings_percent = 52.5 ∧
  (spent_on_rent_percent + spent_on_dishwasher_percent) = 47.5

-- The question to solve: what percent did Dhoni spend on rent?
def spent_on_rent : ℝ := spent_on_rent_percent

-- The correct answer based on the given conditions
def correct_answer : ℝ := 28.75

-- The proof problem to show the percent Dhoni spent on rent
theorem dhoni_spent_on_rent (h : conditions last_month_earnings spent_on_rent_percent spent_on_dishwasher_percent leftover_earnings_percent) :
  spent_on_rent last_month_earnings spent_on_rent_percent spent_on_dishwasher_percent leftover_earnings_percent = correct_answer :=
sorry

end dhoni_spent_on_rent_l82_82509


namespace y_sequence_integer_7_l82_82106

noncomputable def y_sequence (n : ℕ) : ℝ :=
  match n with
  | 0     => 0    -- Not used, 0 index case
  | 1     => (2:ℝ)^(1/3)
  | k + 1 => (y_sequence k)^(2^(1/3))

theorem y_sequence_integer_7 : ∃ n : ℕ, (∀ m < n, ¬ ∃ k : ℤ, y_sequence m = k) ∧ (∃ k : ℤ, y_sequence n = k) ∧ n = 7 :=
by {
  sorry
}

end y_sequence_integer_7_l82_82106


namespace time_for_A_l82_82037

theorem time_for_A (A B C : ℝ) 
  (h1 : 1/B + 1/C = 1/3) 
  (h2 : 1/A + 1/C = 1/2) 
  (h3 : 1/B = 1/30) : 
  A = 5/2 := 
by
  sorry

end time_for_A_l82_82037


namespace gcd_m_n_is_one_l82_82404

/-- Definition of m -/
def m : ℕ := 130^2 + 241^2 + 352^2

/-- Definition of n -/
def n : ℕ := 129^2 + 240^2 + 353^2 + 2^3

/-- Proof statement: The greatest common divisor of m and n is 1 -/
theorem gcd_m_n_is_one : Nat.gcd m n = 1 := by
  sorry

end gcd_m_n_is_one_l82_82404


namespace tire_circumference_is_4_l82_82232

noncomputable def circumference_of_tire
  (speed_kmh : ℕ) 
  (revolutions_per_minute : ℕ) : ℕ :=
  let speed_mpm := (speed_kmh * 1000) / 60 in
  speed_mpm / revolutions_per_minute

theorem tire_circumference_is_4
  (speed_kmh : ℕ)
  (revolutions_per_minute : ℕ)
  (h_speed : speed_kmh = 96)
  (h_revolutions : revolutions_per_minute = 400) :
  circumference_of_tire speed_kmh revolutions_per_minute = 4 :=
by
  sorry

end tire_circumference_is_4_l82_82232


namespace liquid_X_percentage_l82_82845

-- Define the initial conditions and the problem
def initial_conditions (Y: Type) (weight: ℝ) (X_percentage: ℝ) (water_percentage: ℝ) (Z_percentage: ℝ)
                      (evaporated_water: ℝ) (added_weight: ℝ) :=
  X_percentage = 0.40 ∧ water_percentage = 0.45 ∧ Z_percentage = 0.15 ∧
  evaporated_water = 6 ∧ weight = 18 ∧ added_weight = 5

def final_X_percentage (initial_weight X_weight water_weight Z_weight evaporation_weight added_weight: ℝ): ℝ :=
  let remaining_weight := initial_weight - evaporation_weight in
  let new_X_weight := X_weight + 0.40 * added_weight in
  let new_water_weight := water_weight - evaporation_weight + 0.45 * added_weight in
  let new_Z_weight := Z_weight + 0.15 * added_weight in
  let total_new_weight := new_X_weight + new_water_weight + new_Z_weight in
  (new_X_weight / total_new_weight) * 100

theorem liquid_X_percentage: ∀ (Y: Type) (weight X_percentage water_percentage Z_percentage evaporated_water added_weight: ℝ),
  initial_conditions Y weight X_percentage water_percentage Z_percentage evaporated_water added_weight →
  final_X_percentage weight (0.40 * weight) (0.45 * weight) (0.15 * weight) evaporated_water added_weight ≈ 54.12 :=
by
  intros Y weight X_percentage water_percentage Z_percentage evaporated_water added_weight h,
  cases h with hp hrest,
  have hint_weight : initial_weight = 18, from sorry,
  have heval_weight : evaporated_weight = 6, from sorry,
  have hadd_weight : added_weight = 5, from sorry,
  have hX_weight : X_weight = 0.40 * initial_weight, from sorry,
  have hwater_weight : water_weight = 0.45 * initial_weight, from sorry,
  have hZ_weight : Z_weight = 0.15 * initial_weight, from sorry,
  sorry

end liquid_X_percentage_l82_82845


namespace binary_number_div_8_remainder_l82_82410

def binary_remainder : ℕ := 3

theorem binary_number_div_8_remainder :
  let n := 0b110111001011 in
  let d := 8 in
  n % d = binary_remainder :=
by
  sorry

end binary_number_div_8_remainder_l82_82410


namespace match_probability_l82_82796

def probability_of_matching_shoes : ℚ :=
  let total_shoes := 10
  let pairs := 5
  let total_ways_to_select_two_shoes := Nat.choose total_shoes 2
  let ways_to_select_matching_pair := pairs
  (ways_to_select_matching_pair : ℚ) / total_ways_to_select_two_shoes

theorem match_probability:
  ∀ (total_shoes pairs : ℕ) (h1 : total_shoes = 10) (h2 : pairs = 5),
  (let total_ways_to_select_two_shoes := Nat.choose total_shoes 2
   let ways_to_select_matching_pair := pairs
   (ways_to_select_matching_pair : ℚ) / total_ways_to_select_two_shoes) = (1 / 9) :=
begin
  intros,
  rw [h1, h2],
  dsimp,
  have h_comb : Nat.choose 10 2 = 45,
  { norm_num, },
  have h_fraction : (5 : ℚ) / 45 = 1 / 9,
  { norm_num, },
  rw h_comb,
  exact h_fraction,
  sorry,
end

end match_probability_l82_82796


namespace speed_in_still_water_l82_82421

variables {V_m V_s : ℝ}

-- Conditions
def downstream_condition : Prop := (45 : ℝ) = (V_m + V_s) * 3
def upstream_condition : Prop := (18 : ℝ) = (V_m - V_s) * 3

-- Theorem to prove
theorem speed_in_still_water {V_m V_s : ℝ} 
  (h1 : downstream_condition)
  (h2 : upstream_condition) : V_m = 10.5 :=
begin
  sorry
end

end speed_in_still_water_l82_82421


namespace sum_of_possible_values_l82_82737

theorem sum_of_possible_values {x : ℝ} :
  (3 * (x - 3)^2 = (x - 2) * (x + 5)) →
  (∃ (x1 x2 : ℝ), x1 + x2 = 10.5) :=
by sorry

end sum_of_possible_values_l82_82737


namespace transitivity_parallelism_l82_82682

noncomputable def parallel_lines {α β : Type} [affine_space α β] [affine_space β α] 
  (a b c : α) : Prop :=
∀ (p : β), a ∥ b → (∀ q : β, q ∈ p ∧ a ← q → b → q) 

theorem transitivity_parallelism {α β : Type} [affine_space α β] [affine_space β α] 
  (a b c : α) : 
  (parallel_lines a b ∧ parallel_lines b c) → parallel_lines a c :=
begin
  sorry -- Proof to be filled in
end

end transitivity_parallelism_l82_82682


namespace akiko_scores_more_than_michiko_l82_82621

-- Define variables
variables Chandra Akiko Michiko Bailey : ℕ
variable TeamPoints : ℕ

-- Given conditions
def condition1 : Prop := Chandra = 2 * Akiko
def condition2 : Prop := Michiko = Bailey / 2
def condition3 : Prop := Bailey = 14
def condition4 : Prop := TeamPoints = 54
def condition5 : Prop := 14 + 7 + Akiko + Chandra = 54

-- Main proof statement
theorem akiko_scores_more_than_michiko : ∀ Akiko Michiko Chandra Bailey TeamPoints,
  (Chandra = 2 * Akiko) →
  (Michiko = Bailey / 2) →
  (Bailey = 14) →
  (TeamPoints = 54) →
  (14 + 7 + Akiko + Chandra = 54) →
  (Akiko - Michiko = 4) :=
by
  intros Akiko Michiko Chandra Bailey TeamPoints h1 h2 h3 h4 h5
  sorry

end akiko_scores_more_than_michiko_l82_82621


namespace solve_for_x_l82_82118

theorem solve_for_x : ∃ x : ℕ, 2^4 + x = 3^3 - 7 ∧ x = 4 :=
by
  have h1 : 2^4 = 16 := by norm_num
  have h2 : 3^3 = 27 := by norm_num
  use 4
  split
  case left => 
    calc
      2^4 + 4 = 16 + 4 := by rw h1
      ...     = 20     := by norm_num
      3^3 - 7 = 27 - 7 := by rw h2
      ...     = 20     := by norm_num
  case right =>
    rfl

end solve_for_x_l82_82118


namespace flower_pots_cost_difference_l82_82670

theorem flower_pots_cost_difference :
  ∃ d x : ℝ,
    (x + 5 * d = 1.925) ∧ 
    (6 * x + 15 * d = 7.80) ∧ 
    (d = 0.25) :=
by {
  use [0, 0],
  sorry
}

end flower_pots_cost_difference_l82_82670


namespace inequality_true_if_and_only_if_n_eq_3_or_5_l82_82303

theorem inequality_true_if_and_only_if_n_eq_3_or_5
  (n : ℕ)
  (hn : n > 2)
  (a : fin n → ℝ) :
  (∀ a, let A_n := finset.univ.sum (λ i : fin n, finset.univ.prod (λ j : fin n, if i ≠ j then (a i - a j) else 1)) 
    in 0 ≤ A_n) ↔ (n = 3 ∨ n = 5) :=
sorry

end inequality_true_if_and_only_if_n_eq_3_or_5_l82_82303


namespace parrots_per_cage_l82_82819

theorem parrots_per_cage (P : ℕ) (total_birds total_cages parakeets_per_cage : ℕ)
  (h1 : total_cages = 4)
  (h2 : parakeets_per_cage = 2)
  (h3 : total_birds = 40)
  (h4 : total_birds = total_cages * (P + parakeets_per_cage)) :
  P = 8 :=
by
  sorry

end parrots_per_cage_l82_82819


namespace square_side_length_l82_82728

theorem square_side_length (x : ℝ) (h : 4 * x = 2 * x^2) : x = 0 ∨ x = 2 := 
by
suffices h' : x^2 - 2 * x = 0, from sorry,
calc
  4 * x = 2 * x^2  : h
  ... = x^2 - 2 * x : sorry

end square_side_length_l82_82728


namespace dice_probability_l82_82754

def is_odd (n : ℕ) : Prop :=
  n = 1 ∨ n = 3 ∨ n = 5

def is_prime (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5

theorem dice_probability :
  (∑ i in {1, 2, 3, 4, 5, 6}.to_finset, if is_odd i then 1 else 0) * 
  (∑ j in {1, 2, 3, 4, 5, 6}.to_finset, if is_prime j then 1 else 0) / 36 = 1 / 4 :=
sorry

end dice_probability_l82_82754


namespace find_value_l82_82594

theorem find_value (N : ℝ) (h : 1.20 * N = 6000) : 0.20 * N = 1000 :=
sorry

end find_value_l82_82594


namespace splitting_contains_2015_l82_82181

theorem splitting_contains_2015 (m : ℕ) (h1 : m > 1) 
    (h2 : 2015 ∈ (finset.range (m^3)).filter (λ x, x % 2 = 1)) :
    m = 45 :=
sorry

end splitting_contains_2015_l82_82181


namespace distinct_possible_values_for_Z_l82_82252

theorem distinct_possible_values_for_Z :
  ∃ (Z : ℕ), 1 ≤ Z ∧ Z ≤ 9 ∧ (∀ X Y W : ℕ, W ≠ X ∧ X ≠ Y ∧ Y ≠ Z ∧ X ≠ 0 ∧ (W + X + X * 100 + Y * 10 + W + X * 1000 + Z * 100 + Y * 10 + W + Z)
  = (Z + X * 1000 + Z * 100 + Y * 10 + Z) → (Z = X)) ∧ Z ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} :=
sorry

end distinct_possible_values_for_Z_l82_82252


namespace exists_disjoint_translations_l82_82229

open Set

theorem exists_disjoint_translations (n k m : ℕ) (A : Finset ℕ) (S : Finset ℕ) 
  (hA : A.card = k) (hS : S = Finset.range (n + 1)) 
  (h : n > (m - 1) * (Nat.choose k 2 + 1)) : 
  ∃ t : Fin m → ℕ, ∀ i j : Fin m, i ≠ j → (A.map (Function.functor.comp (λ x, x + t i) Nat.int)) ∩ (A.map (Function.functor.comp (λ x, x + t j) Nat.int)) = ∅ :=
by
  sorry

end exists_disjoint_translations_l82_82229


namespace scarlett_oil_problem_l82_82744

theorem scarlett_oil_problem : 
  ∀ (initial_oil added_oil total_oil : ℝ), 
    initial_oil = 0.17 → added_oil = 0.67 → total_oil = initial_oil + added_oil → total_oil = 0.84 :=
by
  intros initial_oil added_oil total_oil h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end scarlett_oil_problem_l82_82744


namespace problem_solution_eq_l82_82866

theorem problem_solution_eq : 
  { x : ℝ | (x ^ 2 - 9) / (x ^ 2 - 1) > 0 } = { x : ℝ | x > 3 ∨ x < -3 } :=
by
  sorry

end problem_solution_eq_l82_82866


namespace axis_of_symmetry_interval_of_increase_minimum_value_in_interval_l82_82926

noncomputable def f (x : ℝ) : ℝ :=
  (Real.cos x)^2 + Real.sin x * Real.cos x - 1/2

theorem axis_of_symmetry :
  ∃ k : ℤ, ∀ x : ℝ, f (x) = f (π/8 + k * π/2) := sorry

theorem interval_of_increase (k : ℤ) :
  ∀ x : ℝ, k * π - 3 * π / 8 ≤ x ∧ x ≤ k * π + π / 8 → f x ≤ f (x + ε) ∀ ε > 0 := sorry

theorem minimum_value_in_interval :
  ∃ x : ℝ, 0 ≤ x ∧ x ≤ π / 2 ∧ f x = -1/2 := sorry

end axis_of_symmetry_interval_of_increase_minimum_value_in_interval_l82_82926


namespace equal_numbers_l82_82629

namespace MathProblem

theorem equal_numbers 
  (x y z : ℝ) 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (hz : 0 < z) 
  (h : x^2 / y + y^2 / z + z^2 / x = x^2 / z + z^2 / y + y^2 / x) : 
  x = y ∨ x = z ∨ y = z :=
by
  sorry

end MathProblem

end equal_numbers_l82_82629


namespace square_area_in_right_triangle_l82_82350

theorem square_area_in_right_triangle 
  (AB CD : ℝ) (hAB : AB = 34) (hCD : CD = 66) : 
  ∃ (x : ℝ), x^2 = 2244 :=
by 
  use √2244
  field_simp
  norm_num
  sorry

end square_area_in_right_triangle_l82_82350


namespace max_dot_product_on_circle_l82_82663

theorem max_dot_product_on_circle :
  (∃(x y : ℝ),
    x^2 + (y - 3)^2 = 1 ∧
    2 ≤ y ∧ y ≤ 4 ∧
    (∀(y : ℝ), (2 ≤ y ∧ y ≤ 4 →
      (x^2 + y^2 - 4) ≤ 12))) := by
  sorry

end max_dot_product_on_circle_l82_82663


namespace modulo_remainder_product_l82_82764

theorem modulo_remainder_product :
  let a := 2022
  let b := 2023
  let c := 2024
  let d := 2025
  let n := 17
  (a * b * c * d) % n = 0 :=
by
  sorry

end modulo_remainder_product_l82_82764


namespace square_side_length_l82_82726

theorem square_side_length (x : ℝ) (h : 4 * x = 2 * x^2) : x = 2 :=
by 
  sorry

end square_side_length_l82_82726


namespace parabola_standard_eq_line_m_tangent_l82_82165

open Real

variables (p k : ℝ) (x y : ℝ)

-- Definitions based on conditions
def parabola_equation (p : ℝ) : Prop := ∀ x y : ℝ, x^2 = 2 * p * y
def line_m (k : ℝ) : Prop := ∀ x y : ℝ, y = k * x + 6

-- Problem statement
theorem parabola_standard_eq (p : ℝ) (hp : p = 2) :
  parabola_equation p ↔ (∀ x y : ℝ, x^2 = 4 * y) :=
sorry

theorem line_m_tangent (k : ℝ) (x1 x2 : ℝ)
  (hpq : x1 + x2 = 4 * k ∧ x1 * x2 = -24)
  (hk : k = 1/2 ∨ k = -1/2) :
  line_m k ↔ ((k = 1/2 ∧ ∀ x y : ℝ, y = 1/2 * x + 6) ∨ (k = -1/2 ∧ ∀ x y : ℝ, y = -1/2 * x + 6)) :=
sorry

end parabola_standard_eq_line_m_tangent_l82_82165


namespace problem1_l82_82028

theorem problem1 : 20 + (-14) - (-18) + 13 = 37 :=
by
  sorry

end problem1_l82_82028


namespace select_two_squares_not_sharing_edge_l82_82984

theorem select_two_squares_not_sharing_edge (grid : fin 3 × fin 3) :
  ∃ ways, ways = 24 :=
sorry

end select_two_squares_not_sharing_edge_l82_82984


namespace find_AE_l82_82805

variables 
  (a : ℝ) 
  (α β : ℝ) 
  (AK KB : ℝ) 
  (angle_BCK angle_CBE : ℝ)

-- Conditions given in the problem
def conditions : Prop :=
  AK = a ∧ KB = a ∧ angle_BCK = α ∧ angle_CBE = β

-- The theorem to prove that AE is equal to the given expression
theorem find_AE (h : conditions) : 
  let AE := (a / (2 * sin α)) * (sqrt (sin β ^ 2 + 8 * sin α ^ 2) - sin β) in 
  True := 
sorry

end find_AE_l82_82805


namespace infinite_series_sum_l82_82500

noncomputable def series_sum : ℝ :=
  ∞.sum (λ n: ℕ, if (n % 3 = 0) then (1 / 27^(n/3)) * if (n/3 % 2 = 0) then 1 else -1 else 0)

theorem infinite_series_sum :
  series_sum = 15 / 26 :=
by sorry

end infinite_series_sum_l82_82500


namespace positive_difference_l82_82143

theorem positive_difference (x : ℝ) (h : real.cbrt (9 - x^2 / 4) = -3) : 
  |12 - (-12)| = 24 :=
by
  sorry

end positive_difference_l82_82143


namespace sum_numerator_denominator_of_repeating_decimal_l82_82415

theorem sum_numerator_denominator_of_repeating_decimal (x : ℚ) (h : x = 36 / 99) : 
  (x.num + x.denom) = 15 :=
by {
  have simplified_fraction : x = 4 / 11,
  { rw h,
    norm_num,
    apply_fun (λ y, y / 9),
    norm_num,
  },
  rw simplified_fraction,
  norm_num,
}

end sum_numerator_denominator_of_repeating_decimal_l82_82415


namespace ratio_of_circumference_to_area_l82_82438

theorem ratio_of_circumference_to_area (r : ℝ) (h : r = 8) :
  let C := 2 * Real.pi * r,
      A := Real.pi * r^2 in
  C / A = 1 / 4 :=
by {
  rw h,
  let C := 2 * Real.pi * 8,
  let A := Real.pi * 8^2,
  have hC : C = 16 * Real.pi := by rw [C]; ring,
  have hA : A = 64 * Real.pi := by rw [A]; ring,
  rw [hC, hA],
  field_simp,
  norm_num,
  exact rfl,
}

end ratio_of_circumference_to_area_l82_82438


namespace unique_not_in_range_l82_82498

open Real

noncomputable def f (a b c d x : ℝ) : ℝ := (a * x + b) / (c * x + d)

theorem unique_not_in_range (a b c d : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : c ≠ 0) (h₄ : d ≠ 0)
  (h₅ : f a b c d 10 = 10) (h₆ : f a b c d 50 = 50) 
  (h₇ : ∀ x, x ≠ -d / c → f a b c d (f a b c d x) = x) :
  ∃! x, ¬ ∃ y, f a b c d y = x :=
  sorry

end unique_not_in_range_l82_82498


namespace sphere_radius_l82_82058

-- Definitions and conditions from the problem
def angle_dihedral : ℝ := 60 * (π / 180)  -- dihedral angle in radians
def spherical_distance : ℝ := 2 * π      -- spherical distance in cm
def angle_AOB : ℝ := 120 * (π / 180)     -- angle AOB in radians

theorem sphere_radius (r : ℝ) 
  (h1 : angle_dihedral = 60 * (π / 180))
  (h2 : spherical_distance = 2 * π)
  (h3 : angle_AOB = 120 * (π / 180)) :
  r = 3 :=
sorry

end sphere_radius_l82_82058


namespace sequence_term_a1000_l82_82610

theorem sequence_term_a1000 :
  ∃ (a : ℕ → ℕ), a 1 = 1007 ∧ a 2 = 1008 ∧
  (∀ n, n ≥ 1 → a n + a (n + 1) + a (n + 2) = 2 * n) ∧
  a 1000 = 1673 :=
by
  sorry

end sequence_term_a1000_l82_82610


namespace find_m_l82_82905

theorem find_m (m : ℝ) (h : (m ^ 2 - 3 * m + m ^ 2 * complex.I) - (4 + (5 * m + 6) * complex.I) = 0) :
  m = -1 :=
sorry

end find_m_l82_82905


namespace min_possible_value_a_plus_b_l82_82306

noncomputable def min_value_a_plus_b : ℕ :=
  let a := 6
  let b := 19
  if 3125 ∣ (a^5 + b^5) ∧ (a + b) % 5 = 0 ∧ 5 ∣ a = false ∧ 5 ∣ b = false then 
      a + b 
  else 
      sorry

theorem min_possible_value_a_plus_b (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : ¬ (5 ∣ a)) (h4 : ¬ (5 ∣ b))
  (h5 : 3125 ∣ (a^5 + b^5)) : a + b = 25 :=
begin
  exact min_value_a_plus_b,
end

end min_possible_value_a_plus_b_l82_82306


namespace xy_sum_and_product_l82_82904

theorem xy_sum_and_product (x y : ℝ) (h1 : x + y = 2 * sqrt 3) (h2 : x * y = sqrt 6) : x^2 * y + x * y^2 = 6 * sqrt 2 :=
by
  sorry

end xy_sum_and_product_l82_82904


namespace percentage_proof_l82_82234

theorem percentage_proof (a : ℝ) (paise : ℝ) (x : ℝ) (h1: paise = 85) (h2: a = 170) : 
  (x/100) * a = paise ↔ x = 50 := 
by
  -- The setup includes:
  -- paise = 85
  -- a = 170
  -- We prove that x% of 170 equals 85 if and only if x = 50.
  sorry

end percentage_proof_l82_82234


namespace part1_part2_l82_82535

section ProofProblem

variable {x y : ℝ}
def A := 2 * x^2 + x * y + 3 * y
def B := x^2 - x * y

-- Proof Problem 1
theorem part1 (h : (x + 2)^2 + abs (y - 3) = 0) : A - 2 * B = -9 := by
  sorry

-- Proof Problem 2
theorem part2 (h : ∀ y, A - 2 * B = 3 * y * (x + 1)) : x = -1 := by
  sorry

end ProofProblem

end part1_part2_l82_82535


namespace wall_length_l82_82467

theorem wall_length (mirror_side length width : ℝ) (h_mirror : mirror_side = 21) (h_width : width = 28) 
  (h_area_relation : (mirror_side * mirror_side) * 2 = width * length) : length = 31.5 :=
by
  -- here you start the proof, but it's not required for the statement
  sorry

end wall_length_l82_82467


namespace standard_equation_of_circle_l82_82600

-- Define the conditions
def radius (C : Circle) : Real := 1
def center_in_first_quadrant (C : Circle) : Prop := 0 < C.center_x ∧ 0 < C.center_y
def tangent_line (C : Circle) : Prop := (∃ x y, (x, y) = C.center ∧ abs((4 * x - 3 * y) / sqrt(4^2 + (-3)^2)) = 1)
def tangent_x_axis (C : Circle) : Prop := C.center_y = 1

-- Define the theorem
theorem standard_equation_of_circle 
    (C : Circle)
    (hr : radius C = 1)
    (hq : center_in_first_quadrant C)
    (tl : tangent_line C)
    (tx : tangent_x_axis C) : 
    C.equation = "((x - 2)^2 + (y - 1)^2 = 1)" :=
begin
  sorry
end

end standard_equation_of_circle_l82_82600


namespace sum_of_solutions_eq_65_l82_82655

theorem sum_of_solutions_eq_65 :
  let g (x : ℝ) := 12 * x + 5 in
  let g_inv (y : ℝ) := (y - 5) / 12 in
  (∀ x : ℝ, x = g_inv (g ((3 * x)⁻¹)) ∧ g_inv x = g ((3 * x)⁻¹)) →
  (∃ s : ℝ, s = 65) :=
by
  let g (x : ℝ) := 12 * x + 5
  let g_inv (y : ℝ) := (y - 5) / 12
  sorry

end sum_of_solutions_eq_65_l82_82655


namespace saree_sale_price_correct_l82_82007

variable (original_price : ℝ) (discounts : List ℝ)

def apply_discount (price : ℝ) (discount : ℝ) : ℝ :=
  price - (price * discount / 100)

def successive_discounts (price : ℝ) (discounts : List ℝ) : ℝ :=
  discounts.foldl apply_discount price

theorem saree_sale_price_correct :
  let original_price := 1200
  let discounts := [5, 2, 3, 4, 3.5]
  |Float.round 2 (successive_discounts original_price discounts) - 1003.92| < 0.01 :=
by
  sorry

end saree_sale_price_correct_l82_82007


namespace Danny_finishes_first_l82_82881

-- Definitions based on the conditions
variables (E D F : ℝ)    -- Garden areas for Emily, Danny, Fiona
variables (e d f : ℝ)    -- Mowing rates for Emily, Danny, Fiona
variables (start_time : ℝ)

-- Condition definitions
def emily_garden_size := E = 3 * D
def emily_garden_size_fiona := E = 5 * F
def fiona_mower_speed_danny := f = (1/4) * d
def fiona_mower_speed_emily := f = (1/5) * e

-- Prove Danny finishes first
theorem Danny_finishes_first 
  (h1 : emily_garden_size E D)
  (h2 : emily_garden_size_fiona E F)
  (h3 : fiona_mower_speed_danny f d)
  (h4 : fiona_mower_speed_emily f e) : 
  (start_time ≤ (5/12) * (start_time + E/d) ∧ start_time ≤ (E/f)) -> (start_time + E/d < start_time + E/e) -> 
  true := 
sorry -- proof is omitted

end Danny_finishes_first_l82_82881


namespace slope_intercept_of_line_l82_82738

-- Define the given line equation
def line_eq (x y : ℝ) := 2 * x + y + 1 = 0

-- State the proof problem in Lean 4
theorem slope_intercept_of_line :
  (∀ (x y : ℝ), line_eq x y) → ∃ (k b : ℝ), k = -2 ∧ b = -1 :=
begin
  intros hxy,
  use [-2, -1],
  split;
  sorry
end

end slope_intercept_of_line_l82_82738


namespace distance_between_parallel_planes_l82_82517

-- Definitions for the two planes
def plane1 (x y z : ℝ) : Prop := 2 * x - 4 * y + 4 * z = 10
def plane2 (x y z : ℝ) : Prop := 2 * x - 4 * y + 4 * z = 4

-- Function to compute distance between a point and a plane
def point_plane_distance (x₀ y₀ z₀ A B C D : ℝ) : ℝ :=
  |A * x₀ + B * y₀ + C * z₀ - D| / real.sqrt (A ^ 2 + B ^ 2 + C ^ 2)

-- Example point on the second plane
def example_point := (2 : ℝ, 0 : ℝ, 0 : ℝ)

-- Proving the distance between the two parallel planes
theorem distance_between_parallel_planes : point_plane_distance 2 0 0 2 (-4) 4 10 = 1 := by
  -- The proof would go here.
  sorry

end distance_between_parallel_planes_l82_82517


namespace geometric_sequence_fourth_term_l82_82133

theorem geometric_sequence_fourth_term (a₁ a₂ : ℝ) (h₁ : a₁ = 3) (h₂ : a₂ = 1/3) :
    ∃ a₄ : ℝ, a₄ = 1/243 :=
sorry

end geometric_sequence_fourth_term_l82_82133


namespace cos_alpha_value_l82_82159

open Real

theorem cos_alpha_value (α : ℝ) (h1 : sin (α + π / 3) = 3 / 5) (h2 : π / 6 < α ∧ α < 5 * π / 6) :
  cos α = (3 * sqrt 3 - 4) / 10 :=
by
  sorry

end cos_alpha_value_l82_82159


namespace line_intersects_circle_midpoint_locus_l82_82541

-- Condition definitions
def circle_C (x y : ℝ) : Prop := (x + 2) ^ 2 + y ^ 2 = 5
def line_l (x y m : ℝ) : Prop := m * x - y + 1 + 2 * m = 0

-- Prove that the line intersects the circle
theorem line_intersects_circle (m : ℝ) :
  ∀ x y : ℝ, circle_C x y → line_l x y m → ∃ x y : ℝ, circle_C x y ∧ line_l x y m :=
by 
sory

-- Prove the locus of the midpoint of chord AB
theorem midpoint_locus (x y : ℝ) :
  (∀ m : ℝ, ∃ a b : ℝ, line_l a b m ∧ circle_C a b ∧ line_l a b m ∧ circle_C a b) →
  (x + 2) ^ 2 + (y - 1 / 2) ^ 2 = 1 / 4 :=
by
  sorry

end line_intersects_circle_midpoint_locus_l82_82541


namespace analytic_expression_l82_82580

-- Define the function and constraints
def f (x : ℝ) := sqrt 3 * sin (ω * x + φ)
variable (ω : ℝ) (φ : ℝ)

-- Given conditions
axiom ω_pos : ω > 0
axiom φ_range : -π / 2 < φ ∧ φ < π / 2
axiom BC_distance : real.dist (x_highest, f x_highest) (x_lowest, f x_lowest) = 4
axiom symmetry_center : (x_sym, 0) = (1/3, 0)

-- Proof target:
theorem analytic_expression :
  f(x) = sqrt 3 * sin (π / 2 * x - π / 6) :=
sorry

end analytic_expression_l82_82580


namespace Anna_score_unique_l82_82846

theorem Anna_score_unique (s c w : ℕ) (hs : s > 100) (h_score_formula : s = 50 + 5 * c - 2 * w) 
    (h_unique : ∀ (s' : ℕ), s' < s → s' > 100 → (∃ (c' w' : ℕ), s' = 50 + 5 * c' - 2 * w' ∧ c' + w' ≤ 50) →
    ∃ (c unique loc w'), s = 50 + 5 * c - 2 * w ∧ c + w ≤ 50 := 
begin
  sorry
end

end Anna_score_unique_l82_82846


namespace molecular_weight_of_4_moles_of_AlPO4_l82_82406

-- Definitions based on conditions
def atomic_weight_Al := 26.98  -- g/mol
def atomic_weight_P := 30.97   -- g/mol
def atomic_weight_O := 16.00   -- g/mol
def moles_AlPO4 := 4

noncomputable def molecular_weight_AlPO4 :=
  atomic_weight_Al + atomic_weight_P + 4 * atomic_weight_O

theorem molecular_weight_of_4_moles_of_AlPO4 :
  moles_AlPO4 * molecular_weight_AlPO4 = 487.80 :=
by
  sorry

end molecular_weight_of_4_moles_of_AlPO4_l82_82406


namespace rectangular_prism_diagonal_eq_l82_82053

-- Define the dimensions of the rectangular prism
def width : ℝ := 12
def height : ℝ := 15
def length : ℝ := 30

-- Define the function to compute the diagonal of a rectangular prism
def diagonal (w h l : ℝ) : ℝ :=
  Real.sqrt (w^2 + h^2 + l^2)

-- State the theorem
theorem rectangular_prism_diagonal_eq :
  diagonal width height length = 3 * Real.sqrt 141 :=
by
  sorry

end rectangular_prism_diagonal_eq_l82_82053


namespace arithmetic_sequence_common_difference_l82_82255

theorem arithmetic_sequence_common_difference (a_n : ℕ → ℤ) (h1 : a_n 5 = 3) (h2 : a_n 6 = -2) : a_n 6 - a_n 5 = -5 :=
by
  sorry

end arithmetic_sequence_common_difference_l82_82255


namespace pond_to_field_area_ratio_l82_82722

theorem pond_to_field_area_ratio :
  (∀ (L W : ℝ), L = 20 ∧ L = 2 * W →
  let A_field := L * W
  let A_pond := 5 * 5 in
  A_pond / A_field = 1 / 8) := sorry

end pond_to_field_area_ratio_l82_82722


namespace segmentation_interval_l82_82752

theorem segmentation_interval (population_size sample_size interval : ℕ) 
  (h_population : population_size = 800) 
  (h_sample : sample_size = 40) 
  (h_calculation : interval = population_size / sample_size) :
  interval = 20 :=
by {
  rw [h_population, h_sample] at h_calculation,
  exact h_calculation,
  sorry
}

end segmentation_interval_l82_82752


namespace total_money_made_l82_82059

def dvd_price : ℕ := 240
def dvd_quantity : ℕ := 8
def washing_machine_price : ℕ := 898

theorem total_money_made : dvd_price * dvd_quantity + washing_machine_price = 240 * 8 + 898 :=
by
  sorry

end total_money_made_l82_82059


namespace find_ratio_l82_82920

theorem find_ratio (a b : ℝ) (h1 : ∀ x, ax^2 + bx + 2 < 0 ↔ (x < -1/2 ∨ x > 1/3)) :
  (a - b) / a = 5 / 6 := 
sorry

end find_ratio_l82_82920


namespace prob_first_quadrant_l82_82980

structure Point where
  x : ℝ
  y : ℝ

def inFirstQuadrant (p : Point) : Prop := p.x > 0 ∧ p.y > 0

def points : List Point := [
  { x := 1, y := 2 },  -- A
  { x := -3, y := 4 }, -- B
  { x := -2, y := -3 },-- C
  { x := 4, y := 3 },  -- D
  { x := 2, y := -3 }  -- E
]

def firstQuadrantPoints (ps : List Point) : List Point :=
  ps.filter inFirstQuadrant

def probabilityFirstQuadrant (ps : List Point) : ℚ :=
  (firstQuadrantPoints ps).length / ps.length

theorem prob_first_quadrant :
  probabilityFirstQuadrant points = 2/5 :=
by
  sorry

end prob_first_quadrant_l82_82980


namespace difference_of_k_max_and_min_l82_82875

theorem difference_of_k_max_and_min
    (T : Type)
    (faces : Finset (Finset T))
    (p : ℕ → Set T)
    (P : Set T := ⋃ j, p j)
    (S : Set T := ⋃ face in faces, face)
    (k : ℕ)
    (h_distinct_planes : ∀ i j, i ≠ j → disjoint (p i) (p j))
    (h_midpoints : ∀ face ∈ faces, ∃ midpoints, ∀ (x y ∈ face) (hxy : x ≠ y), midpoints (x, y)) :
  let max_k := 4 in
  let min_k := 4 in
  max_k - min_k = 0 :=
by
  sorry

end difference_of_k_max_and_min_l82_82875


namespace even_number_position_l82_82082

theorem even_number_position (n : ℕ) (h_n : n = 2000) :
  ∃ row col, row = 250 ∧ col = 1 :=
by
  use (250, 1)
  sorry

end even_number_position_l82_82082


namespace comprehensive_survey_l82_82017

def suitable_for_census (s: String) : Prop := 
  s = "Surveying the heights of all classmates in the class"

theorem comprehensive_survey : suitable_for_census "Surveying the heights of all classmates in the class" :=
by
  sorry

end comprehensive_survey_l82_82017


namespace largest_divisor_n4_n2_l82_82115

theorem largest_divisor_n4_n2 (n : ℤ) : (6 : ℤ) ∣ (n^4 - n^2) :=
sorry

end largest_divisor_n4_n2_l82_82115


namespace sculpture_cost_NAD_to_CNY_l82_82322

def NAD_to_USD (nad : ℕ) : ℕ := nad / 8
def USD_to_CNY (usd : ℕ) : ℕ := usd * 5

theorem sculpture_cost_NAD_to_CNY (nad : ℕ) : (nad = 160) → (USD_to_CNY (NAD_to_USD nad) = 100) :=
by
  intro h1
  rw [h1]
  -- NAD_to_USD 160 = 160 / 8
  have h2 : NAD_to_USD 160 = 20 := rfl
  -- USD_to_CNY 20 = 20 * 5
  have h3 : USD_to_CNY 20 = 100 := rfl
  -- Concluding the theorem
  rw [h2, h3]
  reflexivity

end sculpture_cost_NAD_to_CNY_l82_82322


namespace linear_inequality_always_satisfied_l82_82026

-- Define the problem

theorem linear_inequality_always_satisfied
  (n : ℕ) (a : Fin (n+1) → Fin n → ℝ) (b : Fin (n+1) → ℝ) :
  ∃ (ineq : Fin (n+1) → ℝ → Prop),
    (∀ x : Fin n → ℝ, ∃ i : Fin (n+1), ineq i (∑ j, a i j * x j + b i)) :=
sorry

end linear_inequality_always_satisfied_l82_82026


namespace geometric_sequence_general_term_l82_82173

-- Definitions and conditions
variable (S : ℕ → ℝ) (a : ℕ → ℝ)
variable (n : ℕ)
axiom h1 : S n = a 1 * (1 - 2 ^ n) / (1 - 2)

-- Specific conditions based on the problem
axiom h2 : S 3 = 14
axiom h3 : ∀ n ∈ ℕ, a n = a 1 * 2 ^ (n - 1)

-- The goal to prove
theorem geometric_sequence_general_term
  (n : ℕ) (hn :  n ∈ ℕ) : a n = 2 ^ n := 
sorry

end geometric_sequence_general_term_l82_82173


namespace friends_picnic_l82_82695

theorem friends_picnic : ∃ (n k : ℕ), n = 6 ∧ k = 3 ∧ nat.choose n k = 20 :=
by
  use [6, 3]
  split
  . exact rfl
  split
  . exact rfl
  . exact Nat.choose_eq_factorial_div_factorial

end friends_picnic_l82_82695


namespace gcd_X4_Y4_X5_Y5_l82_82031

theorem gcd_X4_Y4_X5_Y5 (A : ℤ) (hA : A % 2 = 1)
  (X Y : ℂ) 
  (hXY1 : X + Y = -A) 
  (hXY2 : X * Y = -1) :
  Int.gcd ((X^4 + Y^4).natAbs) ((X^5 + Y^5).natAbs) = 1 :=
sorry

end gcd_X4_Y4_X5_Y5_l82_82031


namespace M_inter_N_is_5_l82_82205

/-- Define the sets M and N. -/
def M : Set ℕ := {1, 3, 5, 7}
def N : Set ℕ := {2, 5, 8}

/-- Prove the intersection of M and N is {5}. -/
theorem M_inter_N_is_5 : M ∩ N = {5} :=
by
  sorry

end M_inter_N_is_5_l82_82205


namespace actual_distance_traveled_l82_82231

theorem actual_distance_traveled 
  (D : ℝ) (t : ℝ)
  (h1 : 8 * t = D)
  (h2 : 12 * t = D + 20) : 
  D = 40 :=
by
  sorry

end actual_distance_traveled_l82_82231


namespace max_sum_positive_n_l82_82590

variable {a : ℕ → ℝ}

-- Conditions of the arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d
def s_n (a : ℕ → ℝ) (n : ℕ) : ℝ := (n + 1) * a 1 + (0 to n).sum (λ x, x * (a (x + 1) - a x))
def first_term_positive (a : ℕ → ℝ) : Prop := a 1 > 0
def sum_a4_a5_positive (a : ℕ → ℝ) : Prop := a 4 + a 5 > 0
def prod_a4_a5_negative (a : ℕ → ℝ) : Prop := a 4 * a 5 < 0

theorem max_sum_positive_n 
  (arith_seq : is_arithmetic_sequence a)
  (a1_pos : first_term_positive a)
  (a4_a5_pos : sum_a4_a5_positive a)
  (a4_a5_neg : prod_a4_a5_negative a) : 
  ∃ n : ℕ, n = 8 ∧ ∀ k : ℕ, (k < n → s_n a k > 0) ∧ (n ≤ k → s_n a k ≤ 0) := 
sorry

end max_sum_positive_n_l82_82590


namespace solve_for_d_plus_f_l82_82381

theorem solve_for_d_plus_f (a b c d e f : ℂ) (h1 : b = 2) (h2 : e = -2 * a - c) (h3 : a + b * complex.I + c + d * complex.I + e + f * complex.I = 3 - 2 * complex.I) : d + f = -4 :=
by
  sorry

end solve_for_d_plus_f_l82_82381


namespace new_profit_percentage_l82_82489

theorem new_profit_percentage :
  ∃ (C : ℝ), 
  C > 0 ∧ 
  let S := 659.9999999999994 in
  let SP' := 702 in
  let CP' := 0.90 * C in
  let P' := SP' - CP' in
  let PP' := (P' / CP') * 100 in
  (S = 1.10 * C ∧ PP' = 30) :=
begin
  use 600,
  split,
  { -- proof that C > 0
    norm_num,
  },
  { -- proof that the conditions lead to the correct new profit percentage
    intros,
    calc 
      659.9999999999994 = 1.10 * 600 : by norm_num
      ...,
    sorry
  }
end

end new_profit_percentage_l82_82489


namespace interval_of_x_l82_82134

theorem interval_of_x (x : ℝ) : 
  (3 * x ∈ set.Ioo 2 4 ∧ 4 * x ∈ set.Ioo 3 5) ↔ x ∈ set.Ioo (3 / 4) (5 / 4) :=
by
  sorry

end interval_of_x_l82_82134


namespace number_of_children_l82_82743

theorem number_of_children (total_passengers men women : ℕ) (h1 : total_passengers = 54) (h2 : men = 18) (h3 : women = 26) : 
  total_passengers - men - women = 10 :=
by sorry

end number_of_children_l82_82743


namespace premium_percentage_is_20_l82_82813

-- Definitions based on the conditions
def investment_amount : ℝ := 14400
def face_value : ℝ := 100
def dividend_percentage : ℝ := 0.05
def total_dividend : ℝ := 600

-- Main theorem to prove
theorem premium_percentage_is_20 :
  let dividend_per_share := (face_value * dividend_percentage) in
  let number_of_shares := (total_dividend / dividend_per_share) in
  let cost_per_share := (investment_amount / number_of_shares) in
  let premium_per_share := (cost_per_share - face_value) in
  let premium_percentage := (premium_per_share / face_value) * 100 in
  premium_percentage = 20 :=
by
  sorry

end premium_percentage_is_20_l82_82813


namespace centers_form_square_l82_82320

-- Define what it means for points to be centers of squares on the sides of a parallelogram
variables {A B C D M1 M2 M3 M4 : Type}

-- Assume points A, B, C, D form a parallelogram
def is_parallelogram (A B C D : Type) : Prop :=
  -- This is a simplified definition assuming Type is endowed with
  -- necessary structures like coordinates and distance 
  sorry

-- Define squares constructed outside the sides of the parallelogram
def squares_constructed_outside (A B C D M1 M2 M3 M4 : Type) : Prop :=
  -- This is a simplified definition assuming M1 is the center of the square on AB, 
  -- M2 is the center of the square on BC, M3 is the center of the square on CD,
  -- and M4 is the center of the square on DA constructed outside of the parallelogram
  sorry

-- Formalize the proof problem as a Lean theorem
theorem centers_form_square (A B C D M1 M2 M3 M4 : Type) :
  is_parallelogram A B C D → squares_constructed_outside A B C D M1 M2 M3 M4 →
  -- The centers M1, M2, M3, M4 form a square
  is_square M1 M2 M3 M4 :=
sorry

end centers_form_square_l82_82320


namespace triangle_perimeter_l82_82731

open Real

-- Define the line equation
def line_equation (x y : ℝ) : Prop := x / 3 + y / 4 = 1

-- Define the points A and B
def A : ℝ × ℝ := (3, 0)
def B : ℝ × ℝ := (0, 4)

-- Define the distance function using the Pythagorean theorem
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the perimeter of triangle
def perimeter (A B : ℝ × ℝ) : ℝ :=
  3 + 4 + distance A B

-- The main statement
theorem triangle_perimeter :
  perimeter A B = 12 :=
by 
  -- Here would go the proof steps
  sorry

end triangle_perimeter_l82_82731


namespace hyperbola_half_focal_length_center_origin_foci_axes_distance_asymptote_l82_82353

theorem hyperbola_half_focal_length_center_origin_foci_axes_distance_asymptote
  (h : ∀ x y : ℝ, x^2 - y^2 / 2 = 1) 
  (center_origin : ∃ c : ℝ × ℝ, c = (0, 0)) 
  (foci_axes : ∃ f1 f2 : ℝ × ℝ, (f1.1 = 0 ∨ f1.2 = 0) ∧ (f2.1 = 0 ∨ f2.2 = 0)) 
  (distance_asymptote_P : ∀ k : ℝ, abs((k * -2) / sqrt (k^2 + 1)) = 2 * sqrt 6 / 3)
  (line_through_P_slope : ∃ A B : ℝ × ℝ, ∃ line : ℝ → ℝ × ℝ, line = (λ t, (-2 + (sqrt 2 / 2) * t, (sqrt 2 / 2) * t)))
  (intersects_y_axis_at_M : ∃ M : ℝ × ℝ, M = (0, -2 * (sqrt 2 / 2)))
  (PM_geometric_mean_PA_PB : ∃ PA PB PM : ℝ, PM^2 = PA * PB ∧ PA * PB = 6) :
  ∃ c : ℝ, c = sqrt 3 :=
by
  sorry

end hyperbola_half_focal_length_center_origin_foci_axes_distance_asymptote_l82_82353


namespace triangle_neg3_4_l82_82113

def triangle (a b : ℚ) : ℚ := -a + b

theorem triangle_neg3_4 : triangle (-3) 4 = 7 := 
by 
  sorry

end triangle_neg3_4_l82_82113


namespace magnitude_of_z_eq_sqrt_five_l82_82180

noncomputable def z (a b : ℝ) : ℂ := a + b * Complex.I

theorem magnitude_of_z_eq_sqrt_five (a b : ℝ) (h₁ : 2 * z a b - Complex.conj (z a b) = 1 + 6 * Complex.I) : Complex.abs (z a b) = Real.sqrt 5 :=
by 
  have h₂ : z a b = 1 + 2 * Complex.I := by sorry
  calc
  Complex.abs (z a b) = Complex.abs (1 + 2 * Complex.I) : by rwa h₂
  ... = Real.sqrt 5 : by sorry

end magnitude_of_z_eq_sqrt_five_l82_82180


namespace base_5_rep_nonzero_digits_l82_82999

theorem base_5_rep_nonzero_digits (a : ℕ) (h : (5^1994 - 1) ∣ a) : 
  ∃ (d : ℕ), d ≥ 1994 ∧ (count_nonzero_digits_in_base_5 a) ≥ d :=
by sorry

/-- Counts the nonzero digits in the base 5 representation -/
def count_nonzero_digits_in_base_5 (n : ℕ) : ℕ :=
  let digits := (nat.digits 5 n);
  digits.count (≠ 0)

end base_5_rep_nonzero_digits_l82_82999


namespace black_more_than_blue_l82_82966

noncomputable def number_of_pencils := 8
noncomputable def number_of_blue_pens := 2 * number_of_pencils
noncomputable def number_of_red_pens := number_of_pencils - 2
noncomputable def total_pens := 48

-- Given the conditions
def satisfies_conditions (K B P : ℕ) : Prop :=
  P = number_of_pencils ∧
  B = number_of_blue_pens ∧
  K + B + number_of_red_pens = total_pens

-- Prove the number of more black pens than blue pens
theorem black_more_than_blue (K B P : ℕ) : satisfies_conditions K B P → (K - B) = 10 := by
  sorry

end black_more_than_blue_l82_82966


namespace simplify_and_evaluate_expr_l82_82694

-- Define the variables x and y
variables (x y : ℝ)
-- Define the expression to be simplified and evaluated
def expr := 6 * x^2 * y * (-2 * x * y + y^3) / (x * y^2)

-- The goal is to prove that the expression equals -36 when x = 2 and y = -1
theorem simplify_and_evaluate_expr (hx : x = 2) (hy : y = -1) : expr x y = -36 := by
  -- By substituting the values of x and y, we should get the expected result.
  sorry

end simplify_and_evaluate_expr_l82_82694


namespace sqrt_sum_equality_l82_82408

theorem sqrt_sum_equality : 
  sqrt ((3 - 2 * real.sqrt 3) ^ 2) + sqrt ((3 + 2 * real.sqrt 3) ^ 2) = 6 := 
by
  sorry

end sqrt_sum_equality_l82_82408


namespace min_points_game_eleven_l82_82262

theorem min_points_game_eleven (s7 s8 s9 s10 : ℕ) (avg10 : ℕ) (avg11 : ℕ)
  (h7 : s7 = 21) (h8 : s8 = 15) (h9 : s9 = 12) (h10 : s10 = 19)
  (h_avg10 : (s7 + s8 + s9 + s10) / 4 > (sum of the first six games / 6) )
  (h_avg11 : avg11 > 20) :
  let total10 := s7 + s8 + s9 + s10
  let total11 := avg11 * 11
  total11 - total10 ≥ 58 :=
sorry

end min_points_game_eleven_l82_82262


namespace bowling_prize_arrangements_l82_82085

theorem bowling_prize_arrangements : ∃ (n : ℕ), n = 64 ∧ 
(let game_outcomes := 2 in
let matches := 6 in
n = (game_outcomes ^ matches)) :=
by
  let game_outcomes := 2
  let matches := 6
  existsi (game_outcomes ^ matches)
  exact ⟨rfl, sorry⟩

end bowling_prize_arrangements_l82_82085


namespace number_of_solutions_is_four_l82_82368

def satisfies_equation (x : ℤ) : Prop :=
  (x^2 - x - 1)^(x + 2) = 1

def num_integer_solutions : ℕ :=
  { x : ℤ | satisfies_equation x }.to_finset.card

theorem number_of_solutions_is_four : num_integer_solutions = 4 :=
by
  sorry

end number_of_solutions_is_four_l82_82368


namespace farmer_rice_division_l82_82042

noncomputable def ounces_per_container (total_pounds: ℚ) (num_containers: ℕ) : ℚ :=
  total_pounds * 16 / num_containers

theorem farmer_rice_division :
  ounces_per_container (49 + 3 / 4) 7 ≈ 114 :=
by
  sorry

end farmer_rice_division_l82_82042


namespace man_son_ratio_in_two_years_l82_82814

noncomputable def man_and_son_age_ratio (M S : ℕ) (h1 : M = S + 25) (h2 : S = 23) : ℕ × ℕ :=
  let S_in_2_years := S + 2
  let M_in_2_years := M + 2
  (M_in_2_years / S_in_2_years, S_in_2_years / S_in_2_years)

theorem man_son_ratio_in_two_years : man_and_son_age_ratio 48 23 (by norm_num) (by norm_num) = (2, 1) :=
  sorry

end man_son_ratio_in_two_years_l82_82814


namespace students_circle_no_regular_exists_zero_regular_school_students_l82_82485

noncomputable def students_circle_no_regular (n : ℕ) 
    (student : ℕ → String)
    (neighbor_right : ℕ → ℕ)
    (lies_to : ℕ → ℕ → Bool) : Prop :=
  ∀ i, student i = "Gymnasium student" →
    (if lies_to i (neighbor_right i)
     then (student (neighbor_right i) ≠ "Gymnasium student")
     else student (neighbor_right i) = "Gymnasium student") →
    (if lies_to (neighbor_right i) i
     then (student i ≠ "Gymnasium student")
     else student i = "Gymnasium student")

theorem students_circle_no_regular_exists_zero_regular_school_students
  (n : ℕ) 
  (student : ℕ → String)
  (neighbor_right : ℕ → ℕ)
  (lies_to : ℕ → ℕ → Bool)
  (h : students_circle_no_regular n student neighbor_right lies_to)
  : (∀ i, student i ≠ "Regular school student") :=
sorry

end students_circle_no_regular_exists_zero_regular_school_students_l82_82485


namespace sin_three_pi_minus_alpha_l82_82912

noncomputable def alpha := sorry
axiom cos_condition : cos (π + alpha) = - 1 / 2
axiom alpha_interval : 3 * π / 2 < alpha ∧ alpha < 2 * π

theorem sin_three_pi_minus_alpha :
  sin (3 * π - alpha) = - (sqrt 3) / 2 :=
sorry

end sin_three_pi_minus_alpha_l82_82912


namespace shopper_saved_percentage_l82_82464

-- Definition of the problem conditions
def amount_saved : ℝ := 4
def amount_spent : ℝ := 36

-- Lean 4 statement to prove the percentage saved
theorem shopper_saved_percentage : (amount_saved / (amount_spent + amount_saved)) * 100 = 10 := by
  sorry

end shopper_saved_percentage_l82_82464


namespace no_natural_number_exists_l82_82876

open Nat

theorem no_natural_number_exists (p : ℕ) (m : ℕ) :
  (10 * 94 ∣ p) → (repunit m ∣ p) → (digit_sum p < m) → False := by
  sorry

-- Utility function to compute repunit
def repunit (m : ℕ) : ℕ :=
  if m = 0 then 0 else (10 ^ m - 1) / 9

-- Function to compute digit sum
def digit_sum (n : ℕ) : ℕ :=
  if n = 0 then 0 else (n % 10) + digit_sum (n / 10)

end no_natural_number_exists_l82_82876


namespace probability_of_full_house_is_6_div_4165_l82_82004

open Nat

-- Definitions for combinations
def choose (n k : ℕ) : ℕ :=
  n.choose k

-- Initial conditions
def total_number_of_outcomes : ℕ := choose 52 5

def successful_outcomes : ℕ :=
  let choose_3_cards_of_rank : ℕ := 13 * choose 4 3
  let choose_2_cards_of_other_rank : ℕ := 12 * choose 4 2
  choose_3_cards_of_rank * choose_2_cards_of_other_rank

-- Given this setup, we need to prove the probability is 6/4165
def probability_full_house : ℚ :=
  successful_outcomes / total_number_of_outcomes

theorem probability_of_full_house_is_6_div_4165 :
  probability_full_house = 6 / 4165 := by
  sorry

end probability_of_full_house_is_6_div_4165_l82_82004


namespace algorithm_output_is_127_l82_82253
-- Import the entire Mathlib library

-- Define the possible values the algorithm can output
def possible_values : List ℕ := [15, 31, 63, 127]

-- Define the property where the value is of the form 2^n - 1
def is_exp2_minus_1 (x : ℕ) := ∃ n : ℕ, x = 2^n - 1

-- Define the main theorem to prove the algorithm's output is 127
theorem algorithm_output_is_127 : (∀ x ∈ possible_values, is_exp2_minus_1 x) →
                                      ∃ n : ℕ, 127 = 2^n - 1 :=
by
  -- Define the conditions and the proof steps are left out
  sorry

end algorithm_output_is_127_l82_82253


namespace sequence_periodic_l82_82586

noncomputable def sequence (a : ℕ → ℚ) : Prop :=
  a 1 = 2 ∧ (∀ n : ℕ, a (n + 1) = 1 - 1 / a n)

theorem sequence_periodic (a : ℕ → ℚ) (h : sequence a) : a 2009 = 1 / 2 :=
  sorry

end sequence_periodic_l82_82586


namespace coordinates_P_correct_l82_82256

noncomputable def coordinates_of_P : ℝ × ℝ :=
  let x_distance_to_y_axis : ℝ := 5
  let y_distance_to_x_axis : ℝ := 4
  -- x-coordinate must be negative, y-coordinate must be positive
  let x_coord : ℝ := -x_distance_to_y_axis
  let y_coord : ℝ := y_distance_to_x_axis
  (x_coord, y_coord)

theorem coordinates_P_correct:
  coordinates_of_P = (-5, 4) :=
by
  sorry

end coordinates_P_correct_l82_82256


namespace number_of_subsets_of_set_with_6_and_7_l82_82946

theorem number_of_subsets_of_set_with_6_and_7 :
  let S := {1, 2, 3, 4, 5, 6, 7} in
  finset.card {A ∈ (finset.powerset S) | {6, 7} ⊆ A} = 32 :=
by
  sorry

end number_of_subsets_of_set_with_6_and_7_l82_82946


namespace intersection_of_A_and_B_l82_82901

theorem intersection_of_A_and_B :
  let A := {1, 3}
  let B := {3, 4, 5}
  A ∩ B = {3} :=
by
  let A := {1, 3}
  let B := {3, 4, 5}
  sorry

end intersection_of_A_and_B_l82_82901


namespace geometric_sum_s9_l82_82236

variable (S : ℕ → ℝ)

theorem geometric_sum_s9
  (h1 : S 3 = 7)
  (h2 : S 6 = 63) :
  S 9 = 511 :=
by
  sorry

end geometric_sum_s9_l82_82236


namespace unique_zero_in_interval_l82_82503

-- Define the properties of the function f
axiom f : ℝ → ℝ
axiom h_mono : StrictMono f
axiom h_equiv : ∀ x : ℝ, 0 < x → f (f x - Real.log2 x) = 3

-- Define the function g
def g (x : ℝ) : ℝ := f x + x - 4

-- Theorem statement
theorem unique_zero_in_interval : ∃! x, (0 < x) ∧ (g x = 0) :=
sorry

end unique_zero_in_interval_l82_82503


namespace solve_sequence_problem_l82_82545

variable (a : ℕ → ℝ)
variable (S : ℕ → ℝ)
variable (λ : ℝ)
variable (n : ℕ)

-- Given conditions
def condition1 : Prop := ∀ n, S (n + 1) = 2 * λ * S n + 1
def condition2 : Prop := λ > 0
def condition3 : Prop := a 1 = 1
def condition4 : Prop := a 3 = 4

-- Conclusion to prove
def conclusion : Prop := λ = 1 ∧ (∀ n, a n = 2^(n - 1))

-- Proof statement
theorem solve_sequence_problem (h1 : condition1) (h2 : condition2) (h3 : condition3) (h4 : condition4) : conclusion :=
by 
  sorry

end solve_sequence_problem_l82_82545


namespace surface_area_of_cuboid_from_cubes_l82_82240

theorem surface_area_of_cuboid_from_cubes (s : ℕ) (n : ℕ) (h_s : s = 8) (h_n : n = 3) :
  let l := n * s,
      w := s,
      h := s,
      SA := 2 * l * w + 2 * l * h + 2 * w * h
  in SA = 896 := by
  sorry

end surface_area_of_cuboid_from_cubes_l82_82240


namespace initial_deposit_l82_82078

theorem initial_deposit (A : ℝ) (P : ℝ) (r : ℝ) (n : ℕ) (t : ℝ)
  (hA : A = 5304.5)
  (hr : r = 0.12)
  (hn : n = 4)
  (ht : t = 0.5) :
  P = 5000 :=
by
  have h1 : A = P * (1 + r / ↑n)^(n * t),
  { sorry },
  have h2 : A = 5304.5,
  { exact hA },
  have h3 : P * (1 + 0.12 / 4)^(4 * 0.5) = 5304.5,
  { rw [hr, hn, ht, h1, h2],
    sorry },
  have h4 : P = 5304.5 / (1 + 0.12 / 4)^2,
  { rw [hn, ht] at h3,
    sorry },
  have h5 : P = 5304.5 / 1.0609,
  { sorry },
  have h6 : P ≈ 5000,
  { sorry },
  sorry

end initial_deposit_l82_82078


namespace range_of_f_gt_zero_l82_82559

noncomputable theory

def is_even_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f x = f (-x)

def problem_conditions {f : ℝ → ℝ} :
  (∀ x ≠ 0, deriv f x = f' x) ∧
  (f 1 = 0) ∧
  (is_even_function f) ∧
  (∀ x > 0, x * deriv f x < 2 * f x) :=
sorry

theorem range_of_f_gt_zero {f : ℝ → ℝ} (h : problem_conditions f) :
  {x : ℝ | f x > 0} = (-set.Ioo (-1) 0) ∪ (set.Ioo 0 1) :=
sorry

end range_of_f_gt_zero_l82_82559


namespace complex_real_imag_equal_l82_82563

theorem complex_real_imag_equal (a : ℝ) :
  let z := (1 + 3 * Complex.i) * (2 * a + Complex.i) in
  z.re = z.im → a = -1 := by
  sorry

end complex_real_imag_equal_l82_82563


namespace remainder_of_sum_of_remainders_l82_82646

theorem remainder_of_sum_of_remainders :
  let R := {3^0 % 512, 3^1 % 512, 3^2 % 512, 3^3 % 512, 3^4 % 512, 3^5 % 512,
            3^6 % 512, 3^7 % 512, 3^8 % 512, 3^9 % 512, 3^{10} % 512, 3^{11} % 512}
  let S := (Finset.sum (Finset.finRange 12) (λ n, (3^n % 512)))
  S % 512 = 72 := by
  sorry

end remainder_of_sum_of_remainders_l82_82646


namespace average_of_first_40_results_l82_82352

theorem average_of_first_40_results 
  (A : ℝ)
  (avg_other_30 : ℝ := 40)
  (avg_all_70 : ℝ := 34.285714285714285) : A = 30 :=
by 
  let sum1 := A * 40
  let sum2 := avg_other_30 * 30
  let combined_sum := sum1 + sum2
  let combined_avg := combined_sum / 70
  have h1 : combined_avg = avg_all_70 := by sorry
  have h2 : combined_avg = 34.285714285714285 := by sorry
  have h3 : combined_sum = (A * 40) + (40 * 30) := by sorry
  have h4 : (A * 40) + 1200 = 2400 := by sorry
  have h5 : A * 40 = 1200 := by sorry
  have h6 : A = 1200 / 40 := by sorry
  have h7 : A = 30 := by sorry
  exact h7

end average_of_first_40_results_l82_82352


namespace square_side_length_l82_82725

theorem square_side_length (x : ℝ) (h : 4 * x = 2 * x^2) : x = 2 :=
by 
  sorry

end square_side_length_l82_82725


namespace statistics_students_ratio_l82_82316

noncomputable def numStudents : ℕ := 120
noncomputable def percentSeniors : ℝ := 0.90
noncomputable def numSeniorsInStatistics : ℕ := 54

theorem statistics_students_ratio :
  let S := numSeniorsInStatistics / percentSeniors in
  let T := numStudents in
  S / T = 1 / 2 :=
by
  sorry

end statistics_students_ratio_l82_82316


namespace find_m_is_360_l82_82312

-- Define the angles alpha and beta
def alpha : ℝ := Real.pi / 45
def beta : ℝ := Real.pi / 36

-- Define the tangent value for the initial line l
def tan_theta : ℝ := 2 / 11

-- Define the angle difference
def angle_diff : ℝ := 2 * (beta - alpha)

noncomputable def find_smallest_m (theta : ℝ) : ℕ :=
  if H : 2 * (beta - alpha) > 0 then
    let m := (2 * Real.pi) / angle_diff
    if m > 0 then m.to_nat else 1
  else
    1

theorem find_m_is_360 : find_smallest_m (Real.arctan tan_theta) = 360 :=
by
  unfold find_smallest_m
  -- The following steps would involve proving the equivalence, which we skip with sorry.
  sorry

end find_m_is_360_l82_82312


namespace sum_of_digits_triangular_array_l82_82833

theorem sum_of_digits_triangular_array (N : ℕ) (h : N * (N + 1) / 2 = 5050) : 
  Nat.digits 10 N = [1, 0, 0] := by
  sorry

end sum_of_digits_triangular_array_l82_82833


namespace domain_transformation_l82_82915

theorem domain_transformation (f : ℝ → ℝ) :
  (∀ x, 0 ≤ x ∧ x ≤ 1 → ∃ y, f y = x) →
  (∀ x, -0.5 ≤ x ∧ x ≤ 0 → ∃ y, f (2*x + 1) = y) :=
by
  intros h x
  use (2*x + 1)
  sorry

end domain_transformation_l82_82915


namespace f_monotone_solve_inequality_l82_82908

-- Definitions based on given conditions
def f : ℝ → ℝ := sorry -- Function f: ℝ → ℝ
axiom functional_eq {a b : ℝ} : f(a + b) = f(a) + f(b) - 1
axiom f_pos {x : ℝ} (hx : x > 0) : f(x) > 1
axiom f_4 : f(4) = 3

-- Monotonicity statement
theorem f_monotone : ∀ x1 x2 : ℝ, x1 < x2 → f(x1) < f(x2) := sorry

-- Inequality solution statement
theorem solve_inequality (m : ℝ) (h : -1 < m ∧ m < 4 / 3) : f(3 * m^2 - m - 2) < 2 := sorry

end f_monotone_solve_inequality_l82_82908


namespace cost_price_A_l82_82826

theorem cost_price_A (CP_A : ℝ) (CP_B : ℝ) (SP_C : ℝ) 
(h1 : CP_B = 1.20 * CP_A)
(h2 : SP_C = 1.25 * CP_B)
(h3 : SP_C = 225) : 
CP_A = 150 := 
by 
  sorry

end cost_price_A_l82_82826


namespace Faye_money_left_l82_82886

def Faye_initial_amount : ℝ := 20
def given_by_mother (initial_amount : ℝ) : ℝ := 2 * initial_amount
def total_amount (initial_amount : ℝ) (amount_given : ℝ) : ℝ := initial_amount + amount_given
def cupcake_cost : ℝ := 1.5
def total_cupcake_cost (cost_per_cupcake : ℝ) (quantity : ℕ) : ℝ := cost_per_cupcake * quantity
def cookie_box_cost : ℝ := 3
def total_cookie_cost (cost_per_box : ℝ) (quantity : ℕ) : ℝ := cost_per_box * quantity
def total_spent (cupcake_total : ℝ) (cookie_total : ℝ) : ℝ := cupcake_total + cookie_total
def amount_left (total_amount : ℝ) (total_spent : ℝ) : ℝ := total_amount - total_spent

theorem Faye_money_left :
  let initial_amount := Faye_initial_amount in
  let amount_given := given_by_mother initial_amount in
  let total := total_amount initial_amount amount_given in
  let cupcake_total := total_cupcake_cost cupcake_cost 10 in
  let cookie_total := total_cookie_cost cookie_box_cost 5 in
  let spent := total_spent cupcake_total cookie_total in
  amount_left total spent = 30 := by
  sorry

end Faye_money_left_l82_82886


namespace sum_of_roots_l82_82096
-- Import Mathlib to cover all necessary functionality.

-- Define the function representing the given equation.
def equation (x : ℝ) : ℝ :=
  (3 * x + 4) * (x - 5) + (3 * x + 4) * (x - 7)

-- State the theorem to be proved.
theorem sum_of_roots : (6 : ℝ) + (-4 / 3) = 14 / 3 :=
by
  sorry

end sum_of_roots_l82_82096


namespace monkey_climb_pole_l82_82049

theorem monkey_climb_pole (H : ℕ := 25) (ascent : ℕ := 3) (slip : ℕ := 2) : ∃ t : ℕ, t = 45 ∧
  let net_gain := ascent - slip in
  let cycles_to_within_3m := (H - ascent) / net_gain in
  let time_to_within_3m := cycles_to_within_3m * 2 in
  let final_climb := 1 in
  time_to_within_3m + final_climb = t :=
begin
  sorry
end

end monkey_climb_pole_l82_82049


namespace total_attendance_l82_82838

theorem total_attendance (A C : ℕ) (ticket_sales : ℕ) (adult_ticket_cost child_ticket_cost : ℕ) (total_collected : ℕ)
    (h1 : C = 18) (h2 : ticket_sales = 50) (h3 : adult_ticket_cost = 8) (h4 : child_ticket_cost = 1)
    (h5 : ticket_sales = adult_ticket_cost * A + child_ticket_cost * C) :
    A + C = 22 :=
by {
  sorry
}

end total_attendance_l82_82838


namespace log_a_x_inequality_l82_82927

noncomputable def f (x : ℝ) : ℝ :=
x^2 + x - 2

theorem log_a_x_inequality (a : ℝ) (h₁ : 0 < a ∧ a < 1) :
  (∀ x : ℝ, 0 < x ∧ x < 1 / 2 → f x < Real.log x / Real.log a - 2) →
  a ∈ Set.Icc (Real.cbrt 4 / 4) 1 :=
begin
  sorry
end

end log_a_x_inequality_l82_82927


namespace alyssa_final_money_l82_82069

-- Definitions based on conditions
def weekly_allowance : Int := 8
def spent_on_movies : Int := weekly_allowance / 2
def earnings_from_washing_car : Int := 8

-- The statement to prove
def final_amount : Int := (weekly_allowance - spent_on_movies) + earnings_from_washing_car

-- The theorem expressing the problem
theorem alyssa_final_money : final_amount = 12 := by
  sorry

end alyssa_final_money_l82_82069


namespace no_root_is_nth_root_l82_82692

open polynomial

noncomputable theory

def irreducible_polynomial : polynomial ℚ :=
  polynomial.C 1 * polynomial.X^5 - polynomial.C 1 * polynomial.X^4 -
  polynomial.C 4 * polynomial.X^3 + polynomial.C 4 * polynomial.X^2 +
  polynomial.C 2

theorem no_root_is_nth_root (n : ℕ) (h_pos : n > 0)
  (h_irred : irreducible_polynomial.irreducible)
  (h_galois_S5 : irreducible_polynomial.galois_group ≃ equiv.perm (fin 5))
  (h_two_non_real_roots : ∃! z : ℂ, irreducible_polynomial.eval z = 0 ∧ z.im ≠ 0) :

  ∀ α : ℂ, is_rat_nth_root α n → ¬(is_root α) :=
sorry

-- We assume the definition of is_rat_nth_root and is_root as follows for clarity.
def is_rat_nth_root (α : ℂ) (n : ℕ) : Prop := ∃ q : ℚ, α = complex.cpow (complex.of_real q) (complex.of_real (1 / n))
def is_root (α : ℂ) : Prop := irreducible_polynomial.eval α = 0

end no_root_is_nth_root_l82_82692


namespace consecutiveProbability_sumProdProbability_l82_82385

-- Definition of the boxes and balls
def box := {1, 2, 3, 4, 5}

def drawnPairs := (box × box).toFinset

-- Question 1: consecutive integers
def favorableConsecutive : Finset (ℕ × ℕ) := drawnPairs.filter (λ (x, y), abs (x - y) = 1)
def probabilityConsecutive := (favorableConsecutive.card : ℚ) / drawnPairs.card

theorem consecutiveProbability :
  probabilityConsecutive = 8 / 25 :=
by
  sorry

-- Question 2: sum and product not less than 5
def favorableSumProd : Finset (ℕ × ℕ) := drawnPairs.filter (λ (x, y), x + y ≥ 5 ∧ x * y ≥ 5)
def probabilitySumProd := (favorableSumProd.card : ℚ) / drawnPairs.card

theorem sumProdProbability :
  probabilitySumProd = 17 / 25 :=
by
  sorry

end consecutiveProbability_sumProdProbability_l82_82385


namespace predict_sales_l82_82437

section
variables (x_vals y_vals : List ℕ) 

def avg (lst : List ℕ) : ℚ :=
(lst.foldl (.+) 0) / lst.length

def regression_eqn (b : ℚ) (x : ℚ) : ℚ :=
b * x - 41

-- Average calculations based on given data
def avg_x := avg [4, 6, 8, 10, 12]
def avg_y := avg [5, 25, 35, 70, 90]

-- Calculation of b based on the data
def b : ℚ := (avg_y + 41) / avg_x

-- Prove the predicted y value when x = 16
theorem predict_sales : regression_eqn b 16 = 131 := by
  sorry
end

end predict_sales_l82_82437


namespace net_price_change_l82_82599

noncomputable def net_change_in_price (P : ℝ) : ℝ :=
  (0.6 * P * 1.35 - P) / P * 100

theorem net_price_change (P : ℝ) : net_change_in_price P = -19 :=
by
  unfold net_change_in_price
  calc
    (0.6 * P * 1.35 - P) / P * 100 = (0.81 * P - P) / P * 100 : by ring
    ... = (-0.19 * P) / P * 100 : by ring
    ... = -0.19 * 100 : by field_simp
    ... = -19 : by norm_num

end net_price_change_l82_82599


namespace min_students_scoring_at_least_60_l82_82689

theorem min_students_scoring_at_least_60
  (scores : List ℕ)
  (total_score : ∑ scores = 8250)
  (top_scores : ∀ s, s ∈ [88, 85, 80] → s ∈ scores)
  (min_score : ∃ s ∈ scores, s = 30)
  (max_students_with_same_score : ∀ s ∈ (List.range 60).map (λ n, n + 30), (List.count scores s) ≤ 3) :
  (List.count scores (λ s, s ≥ 60) ≥ 21) := 
sorry

end min_students_scoring_at_least_60_l82_82689


namespace lines_intersect_l82_82046

variables {s v : ℝ}

def line1 (s : ℝ) : ℝ × ℝ :=
  (3 - 2 * s, 4 + 3 * s)

def line2 (v : ℝ) : ℝ × ℝ :=
  (1 - 3 * v, 5 + 2 * v)

theorem lines_intersect :
  ∃ s v : ℝ, line1 s = line2 v ∧ line1 s = (25 / 13, 73 / 13) :=
by
  sorry

end lines_intersect_l82_82046


namespace right_triangle_properties_l82_82251

-- Definitions based on given conditions:
def right_triangle (DE DF : ℝ) (angle_DEF : Prop) : Prop :=
DE = 6 ∧ DF = 8 ∧ angle_DEF

def midpoint (EF N : ℝ) : Prop :=
N = EF / 2

-- Hypotenuse calculation:
def hypotenuse (DE DF EF : ℝ) : Prop :=
EF = Real.sqrt (DE^2 + DF^2)

-- Median length calculation:
def median_length (EF DN : ℝ) : Prop :=
DN = EF / 2

-- Area calculation:
def area_triangle (DE DF A : ℝ) : Prop :=
A = 1 / 2 * DE * DF

-- Centroid distance calculation:
def centroid_distance (DN G : ℝ) : Prop :=
G = 2 / 3 * DN

-- Statement of the problem to prove:
theorem right_triangle_properties 
(DE DF EF DN A G : ℝ) (angle_DEF : Prop) 
(h1 : right_triangle DE DF angle_DEF)
(h2 : hypotenuse DE DF EF) 
(h3 : midpoint EF DN) 
(h4 : median_length EF DN)
(h5 : area_triangle DE DF A)
(h6 : centroid_distance DN G)
: DN = 5.0 ∧ A = 24.0 ∧ G = 3.3 :=
by sorry

end right_triangle_properties_l82_82251


namespace units_digit_of_8_pow_2022_l82_82009

theorem units_digit_of_8_pow_2022 : (8 ^ 2022) % 10 = 4 := 
by
  -- We here would provide the proof of this theorem
  sorry

end units_digit_of_8_pow_2022_l82_82009


namespace circle_radius_eq_two_l82_82506

theorem circle_radius_eq_two (x y : ℝ) : (x^2 + y^2 + 1 = 2 * x + 4 * y) → (∃ c : ℝ × ℝ, ∃ r : ℝ, ((x - c.1)^2 + (y - c.2)^2 = r^2) ∧ r = 2) := by
  sorry

end circle_radius_eq_two_l82_82506


namespace trapezoid_JKLM_perimeter_l82_82785

-- Definitions of the points
def J : ℝ × ℝ := (-3, -4)
def K : ℝ × ℝ := (-3, 1)
def L : ℝ × ℝ := (5, 7)
def M : ℝ × ℝ := (5, -4)

-- Definition to calculate distance between two points in ℝ²
def distance (A B : ℝ × ℝ) : ℝ :=
  real.sqrt ((B.1 - A.1) ^ 2 + (B.2 - A.2) ^ 2)

-- Definition of the perimeter of the trapezoid JKLM
def perimeter_JKLM : ℝ :=
  distance J K + distance K L + distance L M + distance M J

-- The statement to be proved
theorem trapezoid_JKLM_perimeter : perimeter_JKLM = 34 := sorry

end trapezoid_JKLM_perimeter_l82_82785


namespace last_three_digits_of_Alice_list_l82_82066

theorem last_three_digits_of_Alice_list : ∀ N : ℕ, (Alice_list N).nth_digit 998 = 2 ∧ (Alice_list N).nth_digit 999 = 1 ∧ (Alice_list N).nth_digit 1000 = 6 :=
by
  sorry

end last_three_digits_of_Alice_list_l82_82066


namespace total_selling_price_correct_l82_82454

-- Define the cost prices of the three articles
def cost_A : ℕ := 400
def cost_B : ℕ := 600
def cost_C : ℕ := 800

-- Define the desired profit percentages for the three articles
def profit_percent_A : ℚ := 40 / 100
def profit_percent_B : ℚ := 35 / 100
def profit_percent_C : ℚ := 25 / 100

-- Define the selling prices of the three articles
def selling_price_A : ℚ := cost_A * (1 + profit_percent_A)
def selling_price_B : ℚ := cost_B * (1 + profit_percent_B)
def selling_price_C : ℚ := cost_C * (1 + profit_percent_C)

-- Define the total selling price
def total_selling_price : ℚ := selling_price_A + selling_price_B + selling_price_C

-- The proof statement
theorem total_selling_price_correct : total_selling_price = 2370 :=
sorry

end total_selling_price_correct_l82_82454


namespace find_k_l82_82954

theorem find_k (x y k : ℝ) (hx : x = 2) (hy : y = 1) (h : k * x - y = 3) : k = 2 := by
  sorry

end find_k_l82_82954


namespace geometric_sum_inequality_l82_82758

theorem geometric_sum_inequality (n : ℕ) (hn : n ≥ 8) :
  (1 + ∑ i in finset.range (n-1), (1 / 2^i)) > (127 / 64) := 
by {
  sorry
}

end geometric_sum_inequality_l82_82758


namespace conic_section_hyperbola_l82_82116

theorem conic_section_hyperbola : 
  let eq := |y - 3| = sqrt((x + 4)^2 + 4 * y^2) in
  conic_section_type eq = "H" := sorry

end conic_section_hyperbola_l82_82116


namespace some_students_not_club_members_l82_82083

universe u

variable (Student ClubMember FraternityMember : Type u)
variable (isPunctual : Student → Prop)
variable (memberOfClub : Student → Prop)
variable (memberOfFraternity : Student → Prop)

-- Given conditions
axiom students_not_punctual : ∃ s : Student, ¬ isPunctual s
axiom all_club_members_punctual : ∀ c : Student, memberOfClub c → isPunctual c
axiom fraternity_not_club : ∀ f : FraternityMember, ¬ memberOfClub (f : Student)

-- The conclusion to prove
theorem some_students_not_club_members : ∃ s : Student, ¬ memberOfClub s :=
sorry

end some_students_not_club_members_l82_82083


namespace proof_problem_l82_82287

open Real Trigonometry

variables {A B C : ℝ}
-- Conditions of the problem
def conditions : Prop :=
  0 < A ∧ A < π ∧
  0 < B ∧ B < π ∧
  0 < C ∧ C < π ∧
  A + B + C = π ∧
  π/2 < B ∧
  cos(A)^2 + cos(B)^2 - 2 * sin(A) * sin(B) * cos(C) = 11 / 8 ∧
  cos(B)^2 + cos(C)^2 - 2 * sin(B) * sin(C) * cos(A) = 17 / 9

-- Question rewritten as a theorem to prove
theorem proof_problem (h : conditions) :
  cos(C)^2 + cos(A)^2 - 2 * sin(C) * sin(A) * cos(B) = (323 - 14 * sqrt 14) / 100 :=
sorry

end proof_problem_l82_82287


namespace necessary_but_not_sufficient_condition_l82_82817

theorem necessary_but_not_sufficient_condition (a b c d : ℝ) : 
  (a + b < c + d) → (a < c ∨ b < d) :=
sorry

end necessary_but_not_sufficient_condition_l82_82817


namespace find_angle_B_l82_82266

variables {A B C a b c : ℝ} (h1 : 3 * a * Real.cos C = 2 * c * Real.cos A) (h2 : Real.tan A = 1 / 3)

theorem find_angle_B (h1 : 3 * a * Real.cos C = 2 * c * Real.cos A) (h2 : Real.tan A = 1 / 3) : B = 3 * Real.pi / 4 :=
by
  sorry

end find_angle_B_l82_82266


namespace exists_palindrome_product_representation_l82_82120

-- Define a function that checks if a natural number is a palindrome
def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString in s = s.reverse

-- Define a function that counts the number of ways a natural number can be represented as the product of two palindromes
def palindrome_product_representations (n : ℕ) : ℕ :=
  Finset.card { p : ℕ × ℕ | p.1 * p.2 = n ∧ is_palindrome p.1 ∧ is_palindrome p.2 }

-- State the main theorem
theorem exists_palindrome_product_representation :
  ∃ n : ℕ, palindrome_product_representations n > 100 :=
sorry

end exists_palindrome_product_representation_l82_82120


namespace false_proposition_l82_82842

theorem false_proposition :
  (∀ (A B : Prop), (A → B) → (B → A)) →
  (∀ (a b : ℝ), (a = b) → (90° - a = 90° - b)) →
  (∀ (P Q : Prop), (¬ P → ¬ Q) → (P ↔ Q)) →
  ∀ (a b : ℝ), (a + b = 180° → (a = 180° - b)) → False :=
begin
  intros h1 h2 h3 h4,
  -- The proof needs to demonstrate that one of these hypotheses leads to a contradiction, but as per the instructions, we include 'sorry'
  sorry,
end

end false_proposition_l82_82842


namespace min_partition_of_square_into_unequal_squares_l82_82862

theorem min_partition_of_square_into_unequal_squares :
  ∃ (n : ℕ), n = 3 ∧ ∀ (sq : set (set point)), (is_partition sq ∧ is_square sq ∧ (∀ (s₁ s₂ ∈ sq), s₁ ≠ s₂ → size s₁ ≠ size s₂)) := sorry

end min_partition_of_square_into_unequal_squares_l82_82862


namespace ordered_pairs_count_l82_82944

open Nat

theorem ordered_pairs_count :
  (card { mn : ℕ × ℕ | let m := mn.1, n := mn.2 in m ≥ n ∧ (m - n) % 2 = 0 ∧ m^2 - n^2 = 72 }) = 3 :=
by
  sorry

end ordered_pairs_count_l82_82944


namespace volume_of_prism_l82_82057

theorem volume_of_prism (l w h : ℝ) (h1 : l * w = 15) (h2 : w * h = 10) (h3 : l * h = 6) :
  l * w * h = 30 :=
by
  sorry

end volume_of_prism_l82_82057


namespace find_n_l82_82921

variable {a : ℕ → ℝ}  -- Defining the sequence

-- Defining the conditions:
def a1 : Prop := a 1 = 1 / 3
def a2_plus_a5 : Prop := a 2 + a 5 = 4
def a_n_eq_33 (n : ℕ) : Prop := a n = 33

theorem find_n (n : ℕ) : a 1 = 1 / 3 → (a 2 + a 5 = 4) → (a n = 33) → n = 50 := 
by 
  intros h1 h2 h3 
  -- the complete proof can be done here
  sorry

end find_n_l82_82921


namespace find_a_l82_82365

theorem find_a (a : ℝ) : 
  (∃ m₁ m₂ : ℝ, m₁ = -1/2 ∧ m₂ = -a/2 ∧ m₁ * m₂ = -1) → a = -4 :=
by
  intros ⟨m₁, m₂, h₁, h₂, h_prod⟩
  sorry

end find_a_l82_82365


namespace base4_last_digit_390_l82_82861

theorem base4_last_digit_390 : 
  (Nat.digits 4 390).head! = 2 := sorry

end base4_last_digit_390_l82_82861


namespace muffins_sold_in_afternoon_l82_82280

variable (total_muffins : ℕ)
variable (morning_muffins : ℕ)
variable (remaining_muffins : ℕ)

theorem muffins_sold_in_afternoon 
  (h1 : total_muffins = 20) 
  (h2 : morning_muffins = 12) 
  (h3 : remaining_muffins = 4) : 
  (total_muffins - remaining_muffins - morning_muffins) = 4 := 
by
  sorry

end muffins_sold_in_afternoon_l82_82280


namespace ammonium_chloride_formation_l82_82892

theorem ammonium_chloride_formation (moles_NH3 moles_HCl moles_NH4Cl: ℝ) 
    (h1 : moles_NH3 = 3)
    (h2 : moles_NH4Cl = 3)
    (h3 : NH3 + HCl → NH4Cl) :
    moles_NH4Cl = moles_NH3 :=
by sorry

end ammonium_chloride_formation_l82_82892


namespace parabola_kite_area_104_l82_82398

noncomputable def parabola_vert_intersect_kite_area (a c : ℝ) : Prop :=
  let x_int1 := sqrt (3 / (2 * a))
  let x_int2 := sqrt (5 / c)
  let y_int1 := -3
  let y_int2 := 5
  let diag_x_intersection := 2 * x_int1
  let diag_y_intersection := y_int2 - y_int1
  let area_of_kite := 1 / 2 * diag_x_intersection * diag_y_intersection
  area_of_kite = 20

theorem parabola_kite_area_104 : 
  ∃ (a c : ℝ), parabola_vert_intersect_kite_area a c ∧ a + c = 1.04 :=
sorry

end parabola_kite_area_104_l82_82398


namespace liars_count_correct_l82_82391

-- Declare the definition representing the count of liars based on given conditions
def count_of_liars (n : ℕ) : ℕ := if n = 2017 then 1344 else 0

-- Theorem stating that under specified conditions, the number of liars is 1344
theorem liars_count_correct {n : ℕ} (h₁ : n = 2017)
  (h₂ : ∀ i : ℕ, i < n → (islander i).says "my neighbors are from the same tribe")
  (h₃ : ∀ i : ℕ, (islander i).tribe = knight ↔ (isNeighbor i).tribe = liar)
  (h₄ : ∃ i₁ i₂ : ℕ, i₁ < n ∧ i₂ < n ∧ (islander i₁).tribe = liar ∧ (islander i₂).tribe = liar
    ∧ (islander i₁).says "my neighbors are from the same tribe" ∧ (islander i₂).says "my neighbors are from the same tribe") :
  count_of_liars n = 1344 :=
by 
  -- Proof would go here
  sorry

-- Definitions for tribes, islanders, and what they say
inductive Tribe | knight | liar

structure Islander :=
  (tribe : Tribe)
  (says : String → Prop)

-- Function indicating neighbor
def isNeighbor (i : ℕ) : ℕ := (i + 1) % 2017 -- Circular table

-- Example instance for conceptual representation, not part of theorem but helps visualization
axiom islander : ℕ → Islander

end liars_count_correct_l82_82391


namespace minimum_value_of_y_l82_82200

noncomputable def y (θ : ℝ) : ℝ := 
  Real.tan θ + (Real.cos (2 * θ) + 1) / Real.sin (2 * θ)

theorem minimum_value_of_y : 
  ∃ θ ∈ Ioo 0 (π / 2), y θ = 2 ∧ ∀ θ' ∈ Ioo 0 (π / 2), y θ' ≥ y θ := sorry

end minimum_value_of_y_l82_82200


namespace greatest_m_is_999_l82_82150

def h (x : ℕ) : ℕ := 
  if x % 3 ≠ 0 then 1 
  else (3 ^ ((nat.log 3 x).toInt)).toNat

def T (m : ℕ) : ℕ := 
  ∑ k in finset.range (3 ^ (m - 1)) \ {0}, h (3 * k)

theorem greatest_m_is_999
  (h_def : ∀ x, x % 3 ≠ 0 → h x = 1 ∧ ∀ n, 3 ^ n ≤ x → 3 ^ (n + 1) > x → h x = 3 ^ n)
  (T_def : ∀ m, T m = 3 ^ m + (m - 1) * 3 ^ (m - 1))
  (hm : ∃ m < 1000, ∃ k, T m = 3 ^ (k + m - 1)) :
  ∃ m, m = 999 := 
sorry

end greatest_m_is_999_l82_82150


namespace sculpture_cost_NAD_to_CNY_l82_82321

def NAD_to_USD (nad : ℕ) : ℕ := nad / 8
def USD_to_CNY (usd : ℕ) : ℕ := usd * 5

theorem sculpture_cost_NAD_to_CNY (nad : ℕ) : (nad = 160) → (USD_to_CNY (NAD_to_USD nad) = 100) :=
by
  intro h1
  rw [h1]
  -- NAD_to_USD 160 = 160 / 8
  have h2 : NAD_to_USD 160 = 20 := rfl
  -- USD_to_CNY 20 = 20 * 5
  have h3 : USD_to_CNY 20 = 100 := rfl
  -- Concluding the theorem
  rw [h2, h3]
  reflexivity

end sculpture_cost_NAD_to_CNY_l82_82321


namespace sum_of_roots_l82_82098
-- Import Mathlib to cover all necessary functionality.

-- Define the function representing the given equation.
def equation (x : ℝ) : ℝ :=
  (3 * x + 4) * (x - 5) + (3 * x + 4) * (x - 7)

-- State the theorem to be proved.
theorem sum_of_roots : (6 : ℝ) + (-4 / 3) = 14 / 3 :=
by
  sorry

end sum_of_roots_l82_82098


namespace problem_l82_82919

open Real

theorem problem (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h : (1 / a) + (4 / b) + (9 / c) ≤ 36 / (a + b + c)) 
  : (2 * b + 3 * c) / (a + b + c) = 13 / 6 :=
sorry

end problem_l82_82919


namespace sets_equal_l82_82015

theorem sets_equal :
  let M := {x | x^2 - 2 * x + 1 = 0}
  let N := {1}
  M = N :=
by
  sorry

end sets_equal_l82_82015


namespace largest_natural_number_satisfying_conditions_l82_82135

theorem largest_natural_number_satisfying_conditions :
  ∃ (n : ℕ), 
  (let digits := [9, 6, 4, 3, 4, 3, 4, 6, 9] in
   ∀ i, 2 ≤ i → i < (Array.size digits - 1) → 
   digits[i] < (digits[i - 1] + digits[i + 1]) / 2) ∧ 
   n = 96433469 :=
begin
  sorry
end

end largest_natural_number_satisfying_conditions_l82_82135


namespace area_of_triangle_l82_82782

-- Define the side lengths of the triangle
def a : ℝ := 26
def b : ℝ := 25
def c : ℝ := 10

-- Calculate the semi-perimeter
def s : ℝ := (a + b + c) / 2

-- Using Heron's formula for area
noncomputable def area : ℝ := Real.sqrt (s * (s - a) * (s - b) * (s - c))

-- The theorem we need to prove is that the area is approximately 95 cm²
theorem area_of_triangle (h : a = 26 ∧ b = 25 ∧ c = 10) : area ≈ 95 := by
  sorry

end area_of_triangle_l82_82782


namespace find_angle_A_range_sine_product_l82_82556

variable (a b c : ℝ)
variable (A B C : ℝ)
variable (AB AC : ℝ)

-- Conditions given
axiom sides : a > 0 ∧ b > 0 ∧ c > 0
axiom angles : 0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π
axiom dot_product : AB = 4
axiom product_sines : a * c * sin B = 8 * sin A

-- Proof statements
theorem find_angle_A : A = π / 3 := sorry
theorem range_sine_product : ∀ (x : ℝ), x = sin A * sin B * sin C -> 0 < x ∧ x ≤ 3 * sqrt 3 / 8 := sorry

end find_angle_A_range_sine_product_l82_82556


namespace collinear_vectors_l82_82942

theorem collinear_vectors (x : ℝ) 
    (a : ℝ × ℝ := (2, 1)) 
    (b : ℝ × ℝ := (x, -1)) 
    (h : ∃ λ : ℝ, a.1 - b.1 = λ * b.1 ∧ a.2 - b.2 = λ * b.2) : 
    x = -2 :=
by
  sorry

end collinear_vectors_l82_82942


namespace pencils_purchased_l82_82792

theorem pencils_purchased (total_cost : ℝ) (num_pens : ℕ) (pen_price : ℝ) (pencil_price : ℝ) (num_pencils : ℕ) : 
  total_cost = (num_pens * pen_price) + (num_pencils * pencil_price) → 
  num_pens = 30 → 
  pen_price = 20 → 
  pencil_price = 2 → 
  total_cost = 750 →
  num_pencils = 75 :=
by
  sorry

end pencils_purchased_l82_82792


namespace positive_difference_solutions_l82_82141

theorem positive_difference_solutions:
  ∀ x : ℝ, (∃ a b : ℝ, (sqrt (sqrt ((9 - (a^2 / 4))) = -3) ∧ sqrt (sqrt ((9 - (b^2 / 4))) = -3) ∧ (abs (a - b) = 24))) :=
begin
  sorry
end

end positive_difference_solutions_l82_82141


namespace intersection_of_PE_and_QF_l82_82249

-- Definitions of the points and geometric objects
variables {A B C E F P Q : Type*}
variables (triangle_ABC : A)
variables (altitude_BE : B) (altitude_CF : C)
variables (circle_A_F_P : A) (circle_A_F_Q : A)
variables (point_P : P) (point_Q : Q)

-- Conditions
variables (acute_triangle_ABC : IsAcute (triangle_ABC))
variables (altitude_BE_condition : IsAltitude B E)
variables (altitude_CF_condition : IsAltitude C F)
variables (circles_passing_through_A_and_F_at_P : PassesThrough circle_A_F_P A F)
variables (circles_passing_through_A_and_F_at_Q : PassesThrough circle_A_F_Q A F)
variables (circles_tangent_to_BC_at_P : TangentToBC circle_A_F_P)
variables (circles_tangent_to_BC_at_Q : TangentToBC circle_A_F_Q)
variables (B_between_C_and_Q : Between B C Q)

open_locale classical

theorem intersection_of_PE_and_QF
  (h : acute_triangle_ABC)
  (h1 : altitude_BE_condition)
  (h2 : altitude_CF_condition)
  (h3 : circles_passing_through_A_and_F_at_P)
  (h4 : circles_passing_through_A_and_F_at_Q)
  (h5 : circles_tangent_to_BC_at_P)
  (h6 : circles_tangent_to_BC_at_Q)
  (h7 : B_between_C_and_Q) :
  ∃ W : Type*, (on_circumcircle_of_AEF W) ∧ (intersection_point PE QF W) :=
sorry

end intersection_of_PE_and_QF_l82_82249


namespace bus_stops_time_per_hour_l82_82778

theorem bus_stops_time_per_hour 
  (avg_speed_without_stoppages : ℝ) 
  (avg_speed_with_stoppages : ℝ) 
  (h1 : avg_speed_without_stoppages = 75) 
  (h2 : avg_speed_with_stoppages = 40) : 
  ∃ (stoppage_time : ℝ), stoppage_time = 28 :=
by
  sorry

end bus_stops_time_per_hour_l82_82778


namespace problem_solution_l82_82030

theorem problem_solution (a b c : ℝ)
  (h₁ : 10 = (6 / 100) * a)
  (h₂ : 6 = (10 / 100) * b)
  (h₃ : c = b / a) : c = 0.36 :=
by sorry

end problem_solution_l82_82030


namespace shortest_chord_eqn_through_P_l82_82045

theorem shortest_chord_eqn_through_P {P : ℝ × ℝ} (hP : P = (2, 3)) :
  ∃ (l : ℝ → ℝ), (∀ (x y : ℝ), (x^2 + y^2 = 25) → 
  (line_eqn l x y)) ∧
  chord_is_shortest_through P (0, 0) l ↔ 
  (l = λ x, -2/3 * x + 13/3) :=
by
  sorry

def line_eqn (l : ℝ → ℝ) (x y : ℝ) : Prop :=
  y = l x

def chord_is_shortest_through (p q r s : ℝ × ℝ) (l : ℝ → ℝ) : Prop :=
  let slope := (s.2 - q.2) / (s.1 - q.1) in
  ∀ (a b : ℝ × ℝ), (line_eqn l a.1 a.2) → (line_eqn l b.1 b.2) → 
  (dist a b < dist p q)

end shortest_chord_eqn_through_P_l82_82045


namespace positive_difference_between_solutions_l82_82138

theorem positive_difference_between_solutions :
  (∃ (x₁ x₂ : ℝ), (sqrt3 (9 - x₁^2 / 4) = -3) ∧ (sqrt3 (9 - x₂^2 / 4) = -3)
  ∧ (x₁ - x₂ = 24) ∨ (x₂ - x₁ = 24)) :=
by
  sorry

end positive_difference_between_solutions_l82_82138


namespace kim_monthly_expenses_l82_82279

-- Define the conditions

def initial_cost : ℝ := 25000
def monthly_revenue : ℝ := 4000
def payback_period : ℕ := 10

-- Define the proof statement
theorem kim_monthly_expenses :
  ∃ (E : ℝ), 
    (payback_period * (monthly_revenue - E) = initial_cost) → (E = 1500) :=
by
  sorry

end kim_monthly_expenses_l82_82279


namespace number_of_girls_with_straight_short_hair_l82_82742

theorem number_of_girls_with_straight_short_hair 
  (total_people : ℕ) 
  (percent_boys : ℕ) 
  (girls_long_hair : ℕ) 
  (girls_medium_hair : ℕ) 
  (girls_curly_short_hair : ℕ) : 
  (total_people = 250) → 
  (percent_boys = 60) → 
  (girls_long_hair = 40) → 
  (girls_medium_hair = 30) → 
  (girls_curly_short_hair = 25) →
  let total_girls := total_people - (total_people * percent_boys / 100) in
  let percent_girls_straight_short_hair := 
    100 - (girls_long_hair + girls_medium_hair + girls_curly_short_hair) in
  (total_girls * percent_girls_straight_short_hair / 100) = 5 :=
by
  intros total_people_eq percent_boys_eq girls_long_hair_eq girls_medium_hair_eq girls_curly_short_hair_eq
  let total_girls := total_people - (total_people * percent_boys / 100)
  let percent_girls_straight_short_hair :=
    100 - (girls_long_hair + girls_medium_hair + girls_curly_short_hair)
  have total_people_h : total_people = 250 := total_people_eq
  have percent_boys_h : percent_boys = 60 := percent_boys_eq
  have girls_long_hair_h : girls_long_hair = 40 := girls_long_hair_eq
  have girls_medium_hair_h : girls_medium_hair = 30 := girls_medium_hair_eq
  have girls_curly_short_hair_h : girls_curly_short_hair = 25 := girls_curly_short_hair_eq
  rw [total_people_h, percent_boys_h, girls_long_hair_h, girls_medium_hair_h, girls_curly_short_hair_h]
  exact sorry -- Proof will go here

end number_of_girls_with_straight_short_hair_l82_82742


namespace at_least_one_gt_one_l82_82002

theorem at_least_one_gt_one (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y > 2) : (x > 1) ∨ (y > 1) :=
sorry

end at_least_one_gt_one_l82_82002


namespace find_f_expression_solve_inequality_l82_82572

def f (x : ℝ) (a b : ℝ) : ℝ := (a * x + b) / (x - 2)

theorem find_f_expression (a b : ℝ) (ha : a = -1) (hb : b = 2) : 
  ∀ x, f x a b = (2 - x) / (x - 2) :=
by sorry

theorem solve_inequality (k : ℝ) (hk : k > 1) : 
  ∀ x, (2 - x) / (x - 2) < k ↔ 
    if 1 < k ∧ k < 2 then (1 < x ∧ x < k) ∨ (2 < x) 
    else if k = 2 then (1 < x ∧ x < 2) ∨ (2 < x) 
    else (1 < x ∧ x < 2) ∨ (k < x) :=
by sorry

end find_f_expression_solve_inequality_l82_82572


namespace triangle_area_l82_82257

/-- Given a triangle ABC with BC = 12 cm and AD perpendicular to BC with AD = 15 cm,
    prove that the area of triangle ABC is 90 square centimeters. -/
theorem triangle_area {BC AD : ℝ} (hBC : BC = 12) (hAD : AD = 15) :
  (1 / 2) * BC * AD = 90 := by
  sorry

end triangle_area_l82_82257


namespace find_number_whose_multiples_are_taken_for_average_l82_82019

theorem find_number_whose_multiples_are_taken_for_average :
  ∃ x : ℕ, 
    let a := (x + 2 * x + 3 * x + 4 * x + 5 * x + 6 * x + 7 * x) / 7 in
    let b := 2 * 12 in
    a^2 - b^2 = 0 ∧ x = 6 :=
begin
  sorry
end

end find_number_whose_multiples_are_taken_for_average_l82_82019


namespace percentage_gain_is_16_67_l82_82724

noncomputable def manufacturing_cost : ℝ := 180
noncomputable def transportation_cost_per_100 : ℝ := 500
noncomputable def selling_price : ℝ := 222

def total_cost_per_shoe : ℝ := manufacturing_cost + transportation_cost_per_100 / 100

def gain_per_shoe : ℝ := selling_price - total_cost_per_shoe

def percentage_gain : ℝ := (gain_per_shoe / selling_price) * 100

theorem percentage_gain_is_16_67 : percentage_gain = 16.67 := 
by {
  -- Simplifying the expression for proof
  sorry
}

end percentage_gain_is_16_67_l82_82724


namespace cos_x_025_count_cos_x_025_l82_82947

theorem cos_x_025 (x: ℝ) (h1: 0 ≤ x) (h2: x < 360) (h3: Real.cos x = 0.25) : 
  x = Real.arccos 0.25 ∨ x = 360 - Real.arccos 0.25 :=
begin
  sorry
end

theorem count_cos_x_025 : (∃ x1 x2, (0 ≤ x1 ∧ x1 < 360 ∧ Real.cos x1 = 0.25) ∧ (0 ≤ x2 ∧ x2 < 360 ∧ Real.cos x2 = 0.25) ∧ x1 ≠ x2) :=
begin
  use [Real.arccos 0.25, 360 - Real.arccos 0.25],
  split,
  {
    split, 
    exact Real.arccos_nonneg 0.25,
    split,
    {
      have h : 0 ≤ Real.arccos 0.25,
      exact Real.arccos_nonneg 0.25,
      exact h,
    },
    {
      rw Real.cos_arccos,
      norm_num,
      rw [← sub_eq_iff_eq_add, sub_self 360] at *,
      norm_num1,
    }
  },
  split,
  {
    split,
    norm_num1,
    split,
    {
      have h : 0 ≤ Real.arccos 0.25,
      exact Real.arccos_nonneg 0.25,
      exact mod_lt _ _,
    },
    {
      rw Real.sub_eq_iff_eq_add,
      norm_num,
    }
  },
  {
    intro h,
    linarith,
  }
end

end cos_x_025_count_cos_x_025_l82_82947


namespace abc_value_l82_82648

theorem abc_value (a b c : ℂ) (h1 : 2 * a * b + 3 * b = -21)
                   (h2 : 2 * b * c + 3 * c = -21)
                   (h3 : 2 * c * a + 3 * a = -21) :
                   a * b * c = 105.75 := 
sorry

end abc_value_l82_82648


namespace simplify_solve_sin2cos2_l82_82693

theorem simplify_solve_sin2cos2 (α : ℝ) (h : (sin (2 * π - α) * cos (π + α) * cos (π/2 + α) * cos (π/2 - α)) /
  (cos (π - α) * sin (3 * π - α) * sin (-α) * sin (π/2 + α)) = -2) : 
  sin α * sin α - sin α * cos α - 2 * cos α * cos α = 0 :=
sorry

end simplify_solve_sin2cos2_l82_82693


namespace positive_difference_l82_82145

theorem positive_difference (x : ℝ) (h : real.cbrt (9 - x^2 / 4) = -3) : 
  |12 - (-12)| = 24 :=
by
  sorry

end positive_difference_l82_82145


namespace frac_product_eq_l82_82851

noncomputable def product_fractions : ℚ :=
  ∏ i in (finset.range 75).map (nat.succ), (i : ℚ) / (i + 3)

theorem frac_product_eq :
  product_fractions = 1 / 76076 :=
by
  sorry

end frac_product_eq_l82_82851


namespace slope_of_intersection_line_is_one_l82_82395

open Real

-- Definitions of the circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6 * x + 4 * y - 8 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 8 * x + 2 * y + 4 = 0

-- The statement to prove that the slope of the line through the intersection points is 1
theorem slope_of_intersection_line_is_one :
  ∃ m : ℝ, (∀ x y : ℝ, circle1 x y → circle2 x y → (y = m * x + b)) ∧ m = 1 :=
by
  sorry

end slope_of_intersection_line_is_one_l82_82395


namespace karsyn_total_payment_l82_82278

-- Define the initial price of the phone
def initial_price : ℝ := 600

-- Define the discounted rate for the phone
def discount_rate_phone : ℝ := 0.20

-- Define the prices for additional items
def phone_case_price : ℝ := 25
def screen_protector_price : ℝ := 15

-- Define the discount rates
def discount_rate_125 : ℝ := 0.05
def discount_rate_150 : ℝ := 0.10
def final_discount_rate : ℝ := 0.03

-- Define the tax rate and fee
def exchange_rate_fee : ℝ := 0.02

noncomputable def total_payment (initial_price : ℝ) (discount_rate_phone : ℝ) 
  (phone_case_price : ℝ) (screen_protector_price : ℝ) (discount_rate_125 : ℝ) 
  (discount_rate_150 : ℝ) (final_discount_rate : ℝ) (exchange_rate_fee : ℝ) : ℝ :=
  let discounted_phone_price := initial_price * discount_rate_phone
  let additional_items_price := phone_case_price + screen_protector_price
  let total_before_discounts := discounted_phone_price + additional_items_price
  let total_after_first_discount := total_before_discounts * (1 - discount_rate_125)
  let total_after_second_discount := total_after_first_discount * (1 - discount_rate_150)
  let total_after_all_discounts := total_after_second_discount * (1 - final_discount_rate)
  let total_with_exchange_fee := total_after_all_discounts * (1 + exchange_rate_fee)
  total_with_exchange_fee

theorem karsyn_total_payment :
  total_payment initial_price discount_rate_phone phone_case_price screen_protector_price 
    discount_rate_125 discount_rate_150 final_discount_rate exchange_rate_fee = 135.35 := 
  by 
  -- Specify proof steps here
  sorry

end karsyn_total_payment_l82_82278


namespace minimum_value_cos2_sin_l82_82225

theorem minimum_value_cos2_sin (x : ℝ) (h : |x| ≤ Real.pi / 4) :
  ∃ (m : ℝ), (∀ y : ℝ, |y| ≤ Real.pi / 4 → cos y ^ 2 + sin y ≥ m) ∧ m = (1 - Real.sqrt 2) / 2 :=
by
  let f := λ x : ℝ, cos x ^ 2 + sin x
  use (1 - Real.sqrt 2) / 2
  sorry

end minimum_value_cos2_sin_l82_82225


namespace percent_increase_correct_l82_82440

variable (M N : ℝ)

def percent_increase (M N : ℝ) := 100 * (M - N) / (M + N)

theorem percent_increase_correct (M N : ℝ) :
  percent_increase M N = 100 * (M - N) / (M + N) :=
by
  rfl

end percent_increase_correct_l82_82440


namespace price_of_each_apple_l82_82687

theorem price_of_each_apple
  (bike_cost: ℝ) (repair_cost_percent: ℝ) (remaining_percentage: ℝ)
  (total_apples_sold: ℕ) (repair_cost: ℝ) (total_money_earned: ℝ)
  (price_per_apple: ℝ) :
  bike_cost = 80 →
  repair_cost_percent = 0.25 →
  remaining_percentage = 0.2 →
  total_apples_sold = 20 →
  repair_cost = repair_cost_percent * bike_cost →
  total_money_earned = repair_cost / (1 - remaining_percentage) →
  price_per_apple = total_money_earned / total_apples_sold →
  price_per_apple = 1.25 := 
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end price_of_each_apple_l82_82687


namespace lottery_ticket_correct_l82_82669

variable (W : ℝ)
variable (lottery_ticket : Real)
variable (left_for_fun : Real)

noncomputable def correct_amount := 12006

-- Conditions
variable (tax_paid : W = 2*lottery_ticket)
variable (loan_paid : lottery_ticket = 3*left_for_fun)
variable (savings_amount : 1000)
variable (stock_market_investment : 200)
variable (remaining_for_fun : left_for_fun = 2802)

-- Statement
theorem lottery_ticket_correct :
  (W = 2 * (3 * (2802 + 1000 + 200))) :=
sorry

end lottery_ticket_correct_l82_82669


namespace min_value_n_minus_m_l82_82581

def f (x : ℝ) : ℝ := log (x / 2) + 1 / 2
def g (x : ℝ) : ℝ := exp (x - 2)

theorem min_value_n_minus_m : ∀ (m : ℝ), ∃ (n : ℝ) (hn : 0 < n), g m = f n ∧ (n - m) = log 2 :=
by
  sorry

end min_value_n_minus_m_l82_82581


namespace triangle_sin_proj_l82_82605

noncomputable def TriangleSides := Π (A B C : Type) [A B C],
  (a b c : ℝ) -- sides opposite angles A, B and C
  (m n : Type) [m n], -- vectors 
  (dot_product : ℝ) : Prop :=
  ∃ (cos_A_B sin_A_B cos_B sin_B : ℝ), --> definitions and the conditions
  m = (cos_A_B, -sin_A_B) ∧ n = (cos_B, sin_B) ∧
  dot_product = cos_A_B * cos_B - sin_A_B * sin_B

theorem triangle_sin_proj:
  ∀ (A B C : Type) [A B C] (a b c : ℝ) (m n : Type) [m n],
  TriangleSides A B C a b c m n -3/5 → 
  ∃ sin_A : ℝ, sin_A > 0  -- the value of sin A found is greater than 0 and we can derive from the dot product
  ∃ proj_BA_BC : ℝ, proj_BA_BC = (|4*4| * √2) / (2 * √2)
  :=
sorry

end triangle_sin_proj_l82_82605


namespace unit_vector_parallel_to_a_l82_82895

variable (a : ℝ × ℝ)
variable (u : ℝ × ℝ)
variable (k : ℝ)

def magnitude (v : ℝ × ℝ) : ℝ := real.sqrt (v.1 ^ 2 + v.2 ^ 2)

def is_parallel (a u : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, u = (k * a.1, k * a.2)

def is_unit_vector (v : ℝ × ℝ) : Prop :=
  magnitude v = 1

theorem unit_vector_parallel_to_a :
  a = (12, 5) →
  is_unit_vector u →
  is_parallel a u →
  u = (12/13, 5/13) ∨ u = (-12/13, -5/13) :=
by
  intro ha hu hp
  sorry

end unit_vector_parallel_to_a_l82_82895


namespace overlapping_triangles_congruent_l82_82016

theorem overlapping_triangles_congruent :
  (∀ (Δ₁ Δ₂ : Triangle), (overlap Δ₁ Δ₂ → congruent Δ₁ Δ₂)) :=
by sorry

end overlapping_triangles_congruent_l82_82016


namespace parabola_focus_l82_82568

-- Definitions used in the conditions
def parabola_eq (p : ℝ) (x : ℝ) : ℝ := 2 * p * x^2
def passes_through (p : ℝ) : Prop := parabola_eq p 1 = 4

-- The proof that the coordinates of the focus are (0, 1/16) given the conditions
theorem parabola_focus (p : ℝ) (h : passes_through p) : p = 2 → (0, 1 / 16) = (0, 1 / (4 * p)) :=
by
  sorry

end parabola_focus_l82_82568


namespace complex_number_problem_l82_82922

theorem complex_number_problem (z : ℂ) (hz : z * complex.I = ( (complex.I + 1) / (complex.I - 1) ) ^ 2016) : 
  z = -complex.I :=
sorry

end complex_number_problem_l82_82922


namespace minimize_AB_plus_BF_l82_82156

noncomputable def A : ℝ × ℝ := (-2, 2)

def ellipse (x y : ℝ) := (x^2 / 25) + (y^2 / 16) = 1

noncomputable def F : ℝ × ℝ := (-3, 0)

noncomputable def B_min_coordinates : ℝ × ℝ := (-5/2 * Real.sqrt 3, 2)

theorem minimize_AB_plus_BF :
  ∀ B : ℝ × ℝ, ellipse B.1 B.2 →
  ∀ (d1 d2 : ℝ), 
  (d1 = Real.dist A B) →
  (d2 = Real.dist B F) →
  (d1 + 5/3 * d2) ≥ (Real.dist A B_min_coordinates) + 5/3 * (Real.dist B_min_coordinates F) := 
by
  sorry

end minimize_AB_plus_BF_l82_82156


namespace least_number_of_cars_per_work_day_l82_82679

-- Define the conditions as constants in Lean
def paul_work_hours_per_day := 8
def jack_work_hours_per_day := 8
def paul_cars_per_hour := 2
def jack_cars_per_hour := 3

-- Define the total number of cars Paul and Jack can change in a workday
def total_cars_per_day := (paul_cars_per_hour + jack_cars_per_hour) * paul_work_hours_per_day

-- State the theorem to be proved
theorem least_number_of_cars_per_work_day : total_cars_per_day = 40 := by
  -- Proof goes here
  sorry

end least_number_of_cars_per_work_day_l82_82679


namespace count_eight_digit_lucky_numbers_l82_82050

def is_lucky_number (n : ℕ) : Prop :=
  (0 < n) ∧ (∀ d ∈ (Nat.digits 10 n), d = 6 ∨ d = 8) ∧
  (∃ i, Nat.digits 10 n !! (i + 1) = some 8 ∧ Nat.digits 10 n !! i = some 8)

theorem count_eight_digit_lucky_numbers : 
  201 = (Nat.filter (λ n : ℕ, is_lucky_number n)
            (List.range (10^8))).length := 
sorry

end count_eight_digit_lucky_numbers_l82_82050


namespace num_outliers_is_1_l82_82857

def data_set := [3, 22, 30, 30, 35, 42, 42, 50, 55, 66]
def Q1 := 30
def Q3 := 50
def IQR := Q3 - Q1
def outlier_threshold := 1.5 * IQR
def lower_threshold := Q1 - outlier_threshold
def upper_threshold := Q3 + outlier_threshold

theorem num_outliers_is_1 : 
  ∀ x ∈ data_set, (x < lower_threshold ∨ x > upper_threshold) → x = 3 :=
begin
  sorry
end

end num_outliers_is_1_l82_82857


namespace reflect_across_x_axis_l82_82979

theorem reflect_across_x_axis (x y : ℝ) (A : ℝ × ℝ) (h : A = (2, 3)) :
  let A' := (A.1, -A.2) in A' = (2, -3) :=
sorry

end reflect_across_x_axis_l82_82979


namespace sum_of_roots_eq_65_l82_82657

-- Definition of the function g
def g (x : ℝ) : ℝ := 12 * x + 5

-- The problem: the sum of all x that satisfy g⁻¹(x) = g((3 * x)⁻¹)
theorem sum_of_roots_eq_65 :
  (∑ x in {x : ℝ | g⁻¹(x) = g((3 * x)⁻¹)}, x) = 65 :=
sorry

end sum_of_roots_eq_65_l82_82657


namespace probability_black_or_white_l82_82383

variable (P_red P_white P_black : ℝ)

axiom prob_red : P_red = 0.45
axiom prob_white : P_white = 0.25
axiom sum_probabilities : P_red + P_white + P_black = 1

theorem probability_black_or_white : P_black + P_white = 0.55 :=
by
  rw [prob_red, prob_white, sum_probabilities]
  sorry

end probability_black_or_white_l82_82383


namespace cosine_A_l82_82615

variable (A B C D : Point)
variable (angle : Point → Point → Point → ℝ)
variable (length : Point → Point → ℝ)

-- Conditions
variables (alpha : ℝ) 
variables (length_AB length_CD length_AD length_BC : ℝ) 
variables (Perimeter_ABCD : ℝ)

-- Given Conditions
hypothesis h1 : angle A B C = alpha
hypothesis h2 : angle C D A = alpha
hypothesis h3 : length A B = 150
hypothesis h4 : length C D = 150
hypothesis h5 : length A D ≠ length B C
hypothesis h6 : length A B + length B C + length C D + length A D = 520

-- Target
theorem cosine_A : cos (angle A B C) = 11 / 15 := 
by sorry

end cosine_A_l82_82615


namespace river_depth_is_correct_l82_82824

noncomputable def depth_of_river (width : ℝ) (flow_rate_kmph : ℝ) (volume_per_min : ℝ) : ℝ :=
  let flow_rate_mpm := (flow_rate_kmph * 1000) / 60
  let cross_sectional_area := volume_per_min / flow_rate_mpm
  cross_sectional_area / width

theorem river_depth_is_correct :
  depth_of_river 65 6 26000 = 4 :=
by
  -- Steps to compute depth (converted from solution)
  sorry

end river_depth_is_correct_l82_82824


namespace largest_6_digit_div_by_88_l82_82779

theorem largest_6_digit_div_by_88 : ∃ n : ℕ, 100000 ≤ n ∧ n ≤ 999999 ∧ 88 ∣ n ∧ (∀ m : ℕ, 100000 ≤ m ∧ m ≤ 999999 ∧ 88 ∣ m → m ≤ n) ∧ n = 999944 :=
by
  sorry

end largest_6_digit_div_by_88_l82_82779


namespace joe_has_more_shirts_l82_82839

theorem joe_has_more_shirts (alex_shirts : ℕ) (ben_shirts : ℕ) (ben_joe_diff : ℕ)
  (h_a : alex_shirts = 4)
  (h_b : ben_shirts = 15)
  (h_bj : ben_shirts = joe_shirts + ben_joe_diff)
  (h_bj_diff : ben_joe_diff = 8) :
  joe_shirts - alex_shirts = 3 :=
by {
  sorry
}

end joe_has_more_shirts_l82_82839


namespace sculpture_cost_in_chinese_yuan_l82_82326

theorem sculpture_cost_in_chinese_yuan
  (usd_to_nad : ℝ)
  (usd_to_cny : ℝ)
  (cost_nad : ℝ)
  (h1 : usd_to_nad = 8)
  (h2 : usd_to_cny = 5)
  (h3 : cost_nad = 160) :
  (cost_nad / usd_to_nad) * usd_to_cny = 100 :=
by
  sorry

end sculpture_cost_in_chinese_yuan_l82_82326


namespace remainder_T_mod_2021_l82_82149

-- Definition for sum of binary digits
def S2 (n : ℕ) : ℕ := (n.to_digits 2).sum

-- Definition for T as given in the problem
def T : ℕ :=
  (List.range (2021 + 1)).sum (λ k => (-1) ^ (S2 k) * k ^ 3)

-- The Lean statement to prove the problem
theorem remainder_T_mod_2021 : T % 2021 = 1980 := by
  sorry

end remainder_T_mod_2021_l82_82149


namespace largest_number_of_words_largest_number_of_words_n2_l82_82584

section problem

variables {α : Type*}

/-- Definition of the distance between two words. -/
def distance (A B : α) [DecidableEq α] : ℕ :=
(A.to_list.zip B.to_list).countp (λ x, x.fst ≠ x.snd)

/-- Definition of a word lying between two words. -/
def lies_between (A B C : α) [DecidableEq α] : Prop :=
distance A B = distance A C + distance C B

theorem largest_number_of_words {n : ℕ} (h : n ≠ 2) :
  ∃ S : set (fin n → bool), 
  (∀ A B C ∈ S, lies_between A B C ∨ lies_between B A C ∨ lies_between C A B) → 
  S.card ≤ n + 1 :=
sorry

theorem largest_number_of_words_n2 {n : ℕ} (h : n = 2) :
  ∃ S : set (fin n → bool), 
  (∀ A B C ∈ S, lies_between A B C ∨ lies_between B A C ∨ lies_between C A B) → 
  S.card ≤ n + 2 :=
sorry

end problem

end largest_number_of_words_largest_number_of_words_n2_l82_82584


namespace a_2011_eq_1_minus_1_div_m_l82_82903

-- Define the sequence a_n
def a : ℕ → ℝ → ℝ
| 1, m => 1 - 1 / m
| (n + 1), m => 1 - 1 / (a n m)

-- Statement to prove
theorem a_2011_eq_1_minus_1_div_m (m : ℝ) (h : m ≠ 0) : 
  a 2011 m = 1 - 1 / m :=
sorry

end a_2011_eq_1_minus_1_div_m_l82_82903


namespace rectangular_prism_volume_l82_82054

theorem rectangular_prism_volume
  (l w h : ℝ)
  (h1 : l * w = 15)
  (h2 : w * h = 10)
  (h3 : l * h = 6) :
  l * w * h = 30 := by
  sorry

end rectangular_prism_volume_l82_82054


namespace original_number_l82_82414

theorem original_number 
  (x : ℝ)
  (h₁ : 0 < x)
  (h₂ : 1000 * x = 3 * (1 / x)) : 
  x = (Real.sqrt 30) / 100 :=
sorry

end original_number_l82_82414


namespace perpendicular_line_through_point_l82_82060

theorem perpendicular_line_through_point (a b c : ℝ) (h : 2 * a - 6 * b - 14 = 0) : 
    (b + 3 * a - 7 = 0) :=
begin
  sorry
end

end perpendicular_line_through_point_l82_82060


namespace volume_of_prism_l82_82056

theorem volume_of_prism (l w h : ℝ) (h1 : l * w = 15) (h2 : w * h = 10) (h3 : l * h = 6) :
  l * w * h = 30 :=
by
  sorry

end volume_of_prism_l82_82056


namespace monotonic_increasing_iff_a_le_zero_exists_a_decreasing_increasing_l82_82537

noncomputable def f (x a : ℝ) : ℝ := exp x - a * x - 1

theorem monotonic_increasing_iff_a_le_zero (a : ℝ) :
  (∀ x : ℝ, (exp x - a) ≥ 0) → a ≤ 0 := 
sorry

theorem exists_a_decreasing_increasing (a : ℝ) :
  ((∀ x : ℝ, x ∈ Iic 0 → (exp x - a) ≤ 0) ∧ 
  (∀ x : ℝ, x ∈ Ici 0 → (exp x - a) ≥ 0)) → a = 1 :=
sorry

end monotonic_increasing_iff_a_le_zero_exists_a_decreasing_increasing_l82_82537


namespace find_f_2022_l82_82361

-- Define a function f that satisfies the given conditions
def f (x : ℝ) : ℝ := sorry

axiom f_condition : ∀ a b : ℝ, f ((a + 2 * b) / 3) = (f(a) + 2 * f(b)) / 3
axiom f_1 : f 1 = 1
axiom f_4 : f 4 = 7

-- The main theorem to prove
theorem find_f_2022 : f 2022 = 4043 :=
by
  sorry

end find_f_2022_l82_82361


namespace distance_planes_l82_82114

noncomputable def distance_between_planes 
  (plane1 : ℝ × ℝ × ℝ → ℝ) 
  (plane2 : ℝ × ℝ × ℝ → ℝ)
  (normal1 : ℝ × ℝ × ℝ)
  (normal2 : ℝ × ℝ × ℝ)
  (point : ℝ × ℝ × ℝ)
  : ℝ := sorry

theorem distance_planes
  (plane1_eq : ∀ (x y z : ℝ), 3 * x - y + 2 * z - 4 = 0)
  (plane2_eq : ∀ (x y z : ℝ), 6 * x - 2 * y + 4 * z + 3 = 0)
  (normal1_eq : normal1 = (3, -1, 2))
  (normal2_eq : normal2 = (6, -2, 4))
  (point_eq : point = (1, -1, 0)):
  distance_between_planes plane1_eq plane2_eq normal1_eq normal2_eq point_eq = 11 * sqrt 14 / 28 := sorry

end distance_planes_l82_82114


namespace most_entries_with_80_yuan_is_c_pass_pass_a_is_cost_effective_after_30_entries_l82_82800

noncomputable def most_entries_with_80_yuan : Nat :=
let cost_a := 120
let cost_b := 60
let cost_c := 40
let entry_b := 2
let entry_c := 3
let budget := 80
let entries_b := (budget - cost_b) / entry_b
let entries_c := (budget - cost_c) / entry_c
let entries_no_pass := budget / 10
if cost_a <= budget then 
  0
else
  max entries_b (max entries_c entries_no_pass)

theorem most_entries_with_80_yuan_is_c_pass : most_entries_with_80_yuan = 13 :=
by
  sorry

noncomputable def is_pass_a_cost_effective (x : Nat) : Prop :=
let cost_a := 120
let cost_b_entries := 60 + 2 * x
let cost_c_entries := 40 + 3 * x
let cost_no_pass := 10 * x
x > 30 → cost_a < cost_b_entries ∧ cost_a < cost_c_entries ∧ cost_a < cost_no_pass

theorem pass_a_is_cost_effective_after_30_entries : ∀ x : Nat, is_pass_a_cost_effective x :=
by
  sorry

end most_entries_with_80_yuan_is_c_pass_pass_a_is_cost_effective_after_30_entries_l82_82800


namespace find_infection_rate_l82_82349

noncomputable def influenza_transmission (x : ℕ) : Prop :=
  let total_infected := 4 + 4 * x + x * (4 + 4 * x)
  total_infected = 256

theorem find_infection_rate : ∃ x : ℕ, influenza_transmission x ∧ x = 7 :=
by
  use 7
  unfold influenza_transmission
  simp
  sorry

end find_infection_rate_l82_82349


namespace solutions_count_l82_82945

noncomputable def number_of_solutions (a : ℝ) : ℕ :=
if a < 0 then 1
else if 0 ≤ a ∧ a < Real.exp 1 then 0
else if a = Real.exp 1 then 1
else if a > Real.exp 1 then 2
else 0

theorem solutions_count (a : ℝ) :
  (a < 0 ∧ number_of_solutions a = 1) ∨
  (0 ≤ a ∧ a < Real.exp 1 ∧ number_of_solutions a = 0) ∨
  (a = Real.exp 1 ∧ number_of_solutions a = 1) ∨
  (a > Real.exp 1 ∧ number_of_solutions a = 2) :=
by {
  sorry
}

end solutions_count_l82_82945


namespace last_number_in_sequence_l82_82745

def sequence : ℕ → ℕ
| 0 := 3
| 1 := 15
| 2 := 17
| 3 := 51
| 4 := 53
| (n + 5) := if n % 2 = 0 then sequence n + (12 + 11 * (n / 2)) else sequence n + 2

theorem last_number_in_sequence : sequence 5 = 98 := 
by 
  sorry

end last_number_in_sequence_l82_82745


namespace sum_of_roots_l82_82097
-- Import Mathlib to cover all necessary functionality.

-- Define the function representing the given equation.
def equation (x : ℝ) : ℝ :=
  (3 * x + 4) * (x - 5) + (3 * x + 4) * (x - 7)

-- State the theorem to be proved.
theorem sum_of_roots : (6 : ℝ) + (-4 / 3) = 14 / 3 :=
by
  sorry

end sum_of_roots_l82_82097


namespace first_competitor_hotdogs_l82_82245

theorem first_competitor_hotdogs (x y z : ℕ) (h1 : y = 3 * x) (h2 : z = 2 * y) (h3 : z * 5 = 300) : x = 10 :=
sorry

end first_competitor_hotdogs_l82_82245


namespace quoted_value_of_stock_l82_82075

theorem quoted_value_of_stock (D Y Q : ℝ) (h1 : D = 8) (h2 : Y = 10) (h3 : Y = (D / Q) * 100) : Q = 80 :=
by 
  -- Insert proof here
  sorry

end quoted_value_of_stock_l82_82075


namespace system1_solution_system2_solution_l82_82348

-- System (1)
theorem system1_solution (x y : ℝ) (h1 : x + y = 1) (h2 : 3 * x + y = 5) : x = 2 ∧ y = -1 := sorry

-- System (2)
theorem system2_solution (x y : ℝ) (h1 : 3 * (x - 1) + 4 * y = 1) (h2 : 2 * x + 3 * (y + 1) = 2) : x = 16 ∧ y = -11 := sorry

end system1_solution_system2_solution_l82_82348


namespace at_least_two_sums_divisible_by_p_l82_82662

open Int

noncomputable def fractional_part (x : ℚ) : ℚ := x - floor(x)

theorem at_least_two_sums_divisible_by_p
  (p : ℕ) (hp : prime p) (hp_gt2 : 2 < p)
  (a b c d : ℤ)
  (ha : a % p ≠ 0) (hb : b % p ≠ 0) 
  (hc : c % p ≠ 0) (hd : d % p ≠ 0)
  (fractional_cond : ∀ (r : ℤ), (r % p ≠ 0) →
    (fractional_part r * a / p +
     fractional_part r * b / p +
     fractional_part r * c / p +
     fractional_part r * d / p) = 2) :
  ∃ u v, u ≠ v ∧ (u + v) % p = 0 :=
sorry

end at_least_two_sums_divisible_by_p_l82_82662


namespace equation_of_Gamma_max_area_AMN_l82_82183

-- Definition of fixed points
def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (3, 0)

-- Distance function
noncomputable def dist (P Q : ℝ × ℝ) : ℝ :=
  real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- The point P governed by the given ratio condition
def P (x y : ℝ) : ℝ × ℝ := (x, y)

-- Function to define the main condition |PA| / |PO| = 2
def condition (P : ℝ × ℝ) : Prop := dist P A = 2 * dist P O

-- Main theorem for proving equation of curve Γ
theorem equation_of_Gamma (x y : ℝ) (h : condition (P x y)) :
  (x + 1)^2 + y^2 = 4 :=
  sorry

-- Additional conditions for points B and C, and midpoint N
def B : ℝ × ℝ := (0, 0) -- This would be a solution point on Γ
def C : ℝ × ℝ := (0, 0) -- This would be another solution point on Γ
def N : ℝ × ℝ := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)

-- Properties of points B and C on Γ and |BC| = 2√3
def BC_condition (B C : ℝ × ℝ) : Prop :=
  dist B C = 2 * real.sqrt 3

-- Maximum area calculation for triangle AMN
theorem max_area_AMN (hBC : BC_condition B C):
  ∃ M : ℝ × ℝ, dist A M = 2*real.sqrt 3 ∧
  ∃ N : ℝ × ℝ, N = ((B.1 + C.1)/2, (B.2 + C.2)/2) ∧ 
  ∃ area : ℝ, area = 3*real.sqrt 3 :=
  sorry

end equation_of_Gamma_max_area_AMN_l82_82183


namespace rectangle_is_parallelogram_but_parallelogram_not_necessarily_rectangle_l82_82052

-- Define the concept of a Parallelogram
structure Parallelogram (α : Type*) :=
  (sides_parallel : ∀ (a b c d : α), a = b → c = d → a = d → c = b)

-- Define the concept of a Rectangle, which is a special type of Parallelogram
structure Rectangle (α : Type*) extends Parallelogram α :=
  (right_angle : ∀ (a b c : α), a = b → b = c → a ≠ c)

-- Theorem statement
theorem rectangle_is_parallelogram_but_parallelogram_not_necessarily_rectangle :
  ∀ (α : Type*), ∃ (r : Rectangle α), ∀ (p : Parallelogram α), (Rectangle α → Parallelogram α) ∧ ¬(Parallelogram α → Rectangle α) :=
by
  sorry

end rectangle_is_parallelogram_but_parallelogram_not_necessarily_rectangle_l82_82052


namespace z_axis_point_equidistant_l82_82263

theorem z_axis_point_equidistant :
  ∃ z : ℝ, (∀ p : ℝ × ℝ × ℝ, p = (0, 0, z) →
    dist (p.1, p.2, p.3) (1, 0, 2) = dist (p.1, p.2, p.3) (1, -3, 1)) ∧
    (0, 0, z) = (0, 0, -1) :=
by
  sorry

end z_axis_point_equidistant_l82_82263


namespace allowance_is_14_l82_82153

def initial := 11
def spent := 3
def final := 22

def allowance := final - (initial - spent)

theorem allowance_is_14 : allowance = 14 := by
  -- proof goes here
  sorry

end allowance_is_14_l82_82153


namespace S_n_formula_l82_82204

noncomputable def S_n (x y : ℝ) (n : ℕ) : ℝ :=
  ∑ i in Finset.range (n + 1), Real.log (x ^ (n - i) * y ^ i)

theorem S_n_formula (x y : ℝ) (n : ℕ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : Real.log x + Real.log y = 8) :
  S_n x y n = 4 * n^2 + 4 * n :=
sorry

end S_n_formula_l82_82204


namespace positive_difference_between_solutions_l82_82139

theorem positive_difference_between_solutions :
  (∃ (x₁ x₂ : ℝ), (sqrt3 (9 - x₁^2 / 4) = -3) ∧ (sqrt3 (9 - x₂^2 / 4) = -3)
  ∧ (x₁ - x₂ = 24) ∨ (x₂ - x₁ = 24)) :=
by
  sorry

end positive_difference_between_solutions_l82_82139


namespace matrixA_power_four_l82_82094

def matrixA : Matrix (Fin 2) (Fin 2) ℝ :=
  ![
    [1 + Real.sqrt 2, -1],
    [1, 1 + Real.sqrt 2]
  ]

theorem matrixA_power_four :
  matrixA ^ 4 = ![
    [0, -(7 + 5 * Real.sqrt 2)],
    [7 + 5 * Real.sqrt 2, 0]
  ] :=
by
  sorry

end matrixA_power_four_l82_82094


namespace sum_f_from_1_to_2016_l82_82576

noncomputable def f (x : ℝ) : ℝ := Real.cos (π / 3 * x)

theorem sum_f_from_1_to_2016 : (∑ i in Finset.range 2016, f (i + 1)) = 0 :=
by sorry

end sum_f_from_1_to_2016_l82_82576


namespace remainder_of_sum_l82_82668

theorem remainder_of_sum (c d : ℤ) (p q : ℤ) (h1 : c = 60 * p + 53) (h2 : d = 45 * q + 28) : 
  (c + d) % 15 = 6 := 
by
  sorry

end remainder_of_sum_l82_82668


namespace range_of_x_if_sin_theta_eq_log_l82_82955

noncomputable def range_of_x (θ : ℝ) (x : ℝ) : Prop :=
  -1 ≤ sin θ ∧ sin θ ≤ 1 ∧ 1 ≤ x ∧ x ≤ 4

theorem range_of_x_if_sin_theta_eq_log (θ x : ℝ) (h : sin θ = 1 - real.logb 2 x) : 
  -1 ≤ sin θ ∧ sin θ ≤ 1 → 1 ≤ x ∧ x ≤ 4 := by
  sorry

end range_of_x_if_sin_theta_eq_log_l82_82955


namespace cookies_none_of_ingredients_l82_82067

theorem cookies_none_of_ingredients (c : ℕ) (o : ℕ) (r : ℕ) (a : ℕ) (total_cookies : ℕ) :
  total_cookies = 48 ∧ c = total_cookies / 3 ∧ o = (3 * total_cookies + 4) / 5 ∧ r = total_cookies / 2 ∧ a = total_cookies / 8 → 
  ∃ n, n = 19 ∧ (∀ k, k = total_cookies - max c (max o (max r a)) → k ≤ n) :=
by sorry

end cookies_none_of_ingredients_l82_82067


namespace find_p_l82_82923

theorem find_p 
  (A B : ℝ × ℝ)
  (h1 : (A ≠ B) ∧ dist A B = 2)
  (h2 : ∀ t, t ∈ {A, B} → (∃ x y, t = (x, y) ∧ (x > 0 ∧ y > 0)))
  (h3 : ∀ t, t ∈ {A, B} → (t.1^2 / 8 + t.2^2 / 2 = 1))
  (h4 : ∀ t, t ∈ {A, B} → (t.2^2 = 2 * p * t.1 ∧ p > 0)) :
  p = 1 / 4 := 
sorry

end find_p_l82_82923


namespace find_x_for_collinear_vectors_l82_82239

noncomputable def collinear_vectors (a : ℝ × ℝ) (b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b = (k * a.1, k * a.2)

theorem find_x_for_collinear_vectors : ∀ (x : ℝ), collinear_vectors (2, -3) (x, 6) → x = -4 := by
  intros x h
  sorry

end find_x_for_collinear_vectors_l82_82239


namespace candidates_appeared_states_l82_82424

noncomputable def candidates_appeared (X : ℝ) : ℝ :=
  if 0.07 * X = 0.06 * X + 80 then X else 0

theorem candidates_appeared_states (X : ℝ) (h : 0.07 * X = 0.06 * X + 80) : 
  candidates_appeared X = 8000 :=
by
  unfold candidates_appeared
  rw if_pos h
  have h1 : 0.01 * X = 80 := by linarith
  have h2 : X = 80 / 0.01 := by linarith
  norm_num at h2
  exact h2

end candidates_appeared_states_l82_82424


namespace alyssa_money_after_movies_and_carwash_l82_82071

theorem alyssa_money_after_movies_and_carwash : 
  ∀ (allowance spent earned : ℕ), 
  allowance = 8 → 
  spent = allowance / 2 → 
  earned = 8 → 
  (allowance - spent + earned = 12) := 
by 
  intros allowance spent earned h_allowance h_spent h_earned 
  rw [h_allowance, h_spent, h_earned] 
  simp 
  sorry

end alyssa_money_after_movies_and_carwash_l82_82071


namespace triangle_covered_by_circles_l82_82169

noncomputable def altitudes (A B C H : Type) := sorry

theorem triangle_covered_by_circles
  (A B C : Type) [Inhabited A] [Inhabited B] [Inhabited C]
  (triangle_ABC : ∀ x, x ∈ A ∨ x ∈ B ∨ x ∈ C)
  (acute : ∀ x, x ∈ A ∨ x ∈ B ∨ x ∈ C → x = x)
  (circumcircle_a : ∀ (x : Type), x ∈ A → ∃ (r : ℝ), r = altitudes A B C A)
  (circumcircle_b : ∀ (x : Type), x ∈ B → ∃ (r : ℝ), r = altitudes A B C B)
  (circumcircle_c : ∀ (x : Type), x ∈ C → ∃ (r : ℝ), r = altitudes A B C C) :
  ∀ x, x ∈ A ∨ x ∈ B ∨ x ∈ C → x ∈ A ∨ x ∈ B ∨ x ∈ C :=
begin
  sorry
end

end triangle_covered_by_circles_l82_82169


namespace find_f_2022_l82_82363

def satisfies_condition (f : ℝ → ℝ) : Prop :=
  ∀ a b : ℝ, f ((a + 2 * b) / 3) = (f a + 2 * f b) / 3

variables (f : ℝ → ℝ)
  (h_condition : satisfies_condition f)
  (h_f1 : f 1 = 1)
  (h_f4 : f 4 = 7)

theorem find_f_2022 : f 2022 = 4043 :=
  sorry

end find_f_2022_l82_82363


namespace find_original_number_l82_82412

noncomputable def original_number (x : ℝ) : Prop :=
  1000 * x = 3 / x

theorem find_original_number (x : ℝ) (h : original_number x) : x = (Real.sqrt 30) / 100 :=
sorry

end find_original_number_l82_82412


namespace range_of_x_defined_l82_82370

theorem range_of_x_defined (x : ℝ) :
  (∃ y, y = sqrt (x - 3) / (x - 5)) ↔ (x ≥ 3 ∧ x ≠ 5) :=
by
  sorry

end range_of_x_defined_l82_82370


namespace parrots_per_cage_l82_82820

theorem parrots_per_cage (total_birds : ℕ) (num_cages : ℕ) (parakeets_per_cage : ℕ) (total_parrots : ℕ) :
  total_birds = 48 → num_cages = 6 → parakeets_per_cage = 2 → total_parrots = 36 →
  ∀ P : ℕ, (total_parrots = P * num_cages) → P = 6 :=
by
  intros h1 h2 h3 h4 P h5
  subst h1 h2 h3 h4
  sorry

end parrots_per_cage_l82_82820


namespace clock_angle_half_past_eight_l82_82270

/-- Prove that the angle between the hour and minute hands at half-past eight is 75 degrees -/
theorem clock_angle_half_past_eight : 
  ∀ (h m : ℕ), h = 8 ∧ m = 30 → (angle_between_hour_and_minute_hand h m) = 75 :=
by
  sorry

end clock_angle_half_past_eight_l82_82270


namespace fred_went_to_games_l82_82899

variable (m l t : ℕ)
variable (h_m : m = 35) (h_l : l = 11) (h_t : t = 47)

theorem fred_went_to_games (g : ℕ) (h_g : g = t - l) : g = 36 := by
  rw [h_m, h_l, h_t, h_g]
  sorry

end fred_went_to_games_l82_82899


namespace b_share_l82_82784

theorem b_share (a b c : ℕ) (h1 : a + b + c = 120) (h2 : a = b + 20) (h3 : a = c - 20) : b = 20 :=
by
  sorry

end b_share_l82_82784


namespace correct_num_of_expressions_l82_82293

open Real Logarithm

noncomputable def evaluate_expressions (x y : ℝ) (a : ℝ) (cond1 : x ≠ 0) (cond2 : y ≠ 0) (cond3 : 0 < a) (cond4 : a ≠ 1) : ℕ :=
  let exp1 := ¬ (x < 0 → log a (x^2) = 3 * log a x)
  let exp2 := log a (abs (x * y)) ≠ log a (abs x) + log a (abs y)
  let exp3 := ∀ (e : ℝ), e = ln x → x ≠ exp e
  let exp4 := ∀ (y : ℝ) (h : log 10 (ln y) = 0), ln y = 1 ∧ y = exp 1
  let exp5 := ∃ (x : ℝ), 2 ^ (1 + log 4 x) = 16 ∧ x = 4 ^ 3
  [exp1, exp2, exp3, exp4, exp5].count id

theorem correct_num_of_expressions (x y : ℝ) (a : ℝ) (cond1 : x ≠ 0) (cond2 : y ≠ 0) (cond3 : 0 < a) (cond4 : a ≠ 1) :
  evaluate_expressions x y a cond1 cond2 cond3 cond4 = 2 :=
sorry

end correct_num_of_expressions_l82_82293


namespace inequality_proof_l82_82914

def a := Real.cos (Real.pi / 8)
def b := Real.sin (Real.pi / 8)
def c := 0.3 ^ (-2)

theorem inequality_proof : c > a ∧ a > b := by
  sorry

end inequality_proof_l82_82914


namespace labourer_present_salary_l82_82811

theorem labourer_present_salary 
  (P : ℝ) 
  (h1 : ∀ n : ℕ, n > 0 → P * (1.4 : ℝ) ^ n) 
  (h2 : P * (1.4 : ℝ) ^ 3 = 8232) 
  : P = 3000 := by 
  sorry

end labourer_present_salary_l82_82811


namespace sequence_first_last_four_equal_l82_82297

theorem sequence_first_last_four_equal (S : List ℕ) (n : ℕ)
  (hS : S.length = n)
  (h_max : ∀ T : List ℕ, (∀ i j : ℕ, i < j → i ≤ n-5 → j ≤ n-5 → 
                        (S.drop i).take 5 ≠ (S.drop j).take 5) → T.length ≤ n)
  (h_distinct : ∀ i j : ℕ, i < j → i ≤ n-5 → j ≤ n-5 → 
                (S.drop i).take 5 ≠ (S.drop j).take 5) :
  (S.take 4 = S.drop (n-4)) :=
by
  sorry

end sequence_first_last_four_equal_l82_82297


namespace find_a_when_b_is_2_l82_82702

-- Basic definitions
variables (a b k : ℝ)
axiom varies_inversely (ha : a * b^3 = k) : Prop

-- Initial conditions
variables (h1 : btw a = 5)
variables (h2 : b = 1)
variables h3 : k = 5

-- Proof goal
theorem find_a_when_b_is_2 (h4 : b = 2) : a = 5/8 :=
by sorry

end find_a_when_b_is_2_l82_82702


namespace oranges_for_juice_l82_82741

theorem oranges_for_juice 
  (bags : ℕ) (oranges_per_bag : ℕ) (rotten_oranges : ℕ) (oranges_sold : ℕ)
  (h_bags : bags = 10)
  (h_oranges_per_bag : oranges_per_bag = 30)
  (h_rotten_oranges : rotten_oranges = 50)
  (h_oranges_sold : oranges_sold = 220):
  (bags * oranges_per_bag - rotten_oranges - oranges_sold = 30) :=
by 
  sorry

end oranges_for_juice_l82_82741


namespace log_sum_eq_l82_82768

theorem log_sum_eq :
  log 5 16 + 3 * log 5 2 + 5 * log 5 4 + 2 * log 5 64 = log 5 536870912 :=
by
  sorry

end log_sum_eq_l82_82768


namespace target_runs_is_282_l82_82622

-- Define the conditions
def run_rate_first_10_overs : ℝ := 3.2
def overs_first_segment : ℝ := 10
def run_rate_remaining_20_overs : ℝ := 12.5
def overs_second_segment : ℝ := 20

-- Define the calculation of runs in the first 10 overs
def runs_first_segment : ℝ := run_rate_first_10_overs * overs_first_segment

-- Define the calculation of runs in the remaining 20 overs
def runs_second_segment : ℝ := run_rate_remaining_20_overs * overs_second_segment

-- Define the target runs
def target_runs : ℝ := runs_first_segment + runs_second_segment

-- State the theorem
theorem target_runs_is_282 : target_runs = 282 :=
by
  -- This is where the proof would go, but it is omitted.
  sorry

end target_runs_is_282_l82_82622


namespace johns_change_percentage_is_18_percent_l82_82636

def prices : List ℝ := [12.50, 3.25, 7.75, 9.40, 5.60, 2.15]
def total_price (ps : List ℝ) := ps.foldl (· + ·) 0
def paid_amount : ℝ := 50.00
def change (total : ℝ) := paid_amount - total
def percentage (chg : ℝ) := (chg / paid_amount) * 100

theorem johns_change_percentage_is_18_percent :
  percentage (change (total_price prices)) = 18.7 :=
by 
  -- this theorem states that the calculated percentage equals to 18.7
  -- based on the given prices of items and the amount paid.
  sorry

end johns_change_percentage_is_18_percent_l82_82636


namespace sum_of_angles_eq_100_l82_82852

def A (x : ℝ) := x.sin ^ 2 / x.cos
def B (x : ℝ) := x.cos

noncomputable def sum_equal (x : ℝ) : Prop :=
  ∑ i in finset.range (46 - 3 + 1), 2 * (x + i).sin * 3.sin * (1 + (x + i - 3).sec * (x + i + 3).sec) =
  ∑ n in finset.range 4, (-1) ^ (n + 1) * A (3 * n + 3)

theorem sum_of_angles_eq_100 :
  sum_equal 3 → (3 + 4 + 46 + 47 = 100) :=
sorry

end sum_of_angles_eq_100_l82_82852


namespace decimal_2_09_is_209_percent_l82_82032

-- Definition of the conversion from decimal to percentage
def decimal_to_percentage (x : ℝ) := x * 100

-- Theorem statement
theorem decimal_2_09_is_209_percent : decimal_to_percentage 2.09 = 209 :=
by sorry

end decimal_2_09_is_209_percent_l82_82032


namespace greatest_value_of_n_l82_82023

theorem greatest_value_of_n : ∀ (n : ℤ), 102 * n^2 ≤ 8100 → n ≤ 8 :=
by 
  sorry

end greatest_value_of_n_l82_82023


namespace num_mappings_from_A_to_B_l82_82561

def f (x : ℝ) : ℤ := Int.floor x

def A : Set ℕ := {x | x ≠ 0 ∧ (2 : ℕ) ^ x ≤ x ^ 2}

def B : Set ℤ := {y | ∃ x, -1 ≤ x ∧ x < 1 ∧ y = f x}

theorem num_mappings_from_A_to_B : 
  (∃ (A_elements : Finset ℕ) (B_elements : Finset ℤ),
  (A_elements = {0, 1, 2, 3, 4}) ∧ (B_elements = {-1, 0}) ∧ 
  (∃ f : A_elements → B_elements, Function.Injective f ∧ Function.Surjective f ∧
   (Finset.card A_elements = 5) ∧ (Finset.card B_elements = 2))) :=
sorry

end num_mappings_from_A_to_B_l82_82561


namespace sum_of_A_and_B_l82_82386

theorem sum_of_A_and_B (A B : ℕ) (h1 : A ≠ B) (h2 : A < 10) (h3 : B < 10) :
  (10 * A + B) * 6 = 111 * B → A + B = 11 :=
by
  intros h
  sorry

end sum_of_A_and_B_l82_82386


namespace angle_between_vectors_l82_82547

variables {V : Type} [inner_product_space ℝ V]

/-- Given two non-zero vectors a and b such that |a| = 2|b| and (a - b) ⬝ b = 0, the angle 
between a and b is π/3. -/
theorem angle_between_vectors (a b : V) (ha : a ≠ 0) (hb : b ≠ 0)
  (h1 : ∥ a ∥ = 2 * ∥ b ∥)
  (h2 : ⟪ a - b, b ⟫ = 0) : 
  real.angle a b = real.angle (1 : ℝ) (1/2 : ℝ) :=
begin
  sorry
end

end angle_between_vectors_l82_82547


namespace a_mul_b_is_integer_l82_82528

-- Definitions of the sequences a_n and b_n
def a (n : ℕ) : ℝ := ∑' k : ℕ, (k ^ n) / (k.factorial : ℝ)
def b (n : ℕ) : ℝ := ∑' k : ℕ, ((-1 : ℝ) ^ k) * (k ^ n) / (k.factorial : ℝ)

-- The theorem we need to prove
theorem a_mul_b_is_integer (n : ℕ) (hn : n ≥ 1) : ∃ (z : ℤ), (a n) * (b n) = z := 
sorry

end a_mul_b_is_integer_l82_82528


namespace sum_of_first_2016_terms_l82_82564

open Function

noncomputable def arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

variables (a : ℕ → ℝ) (S : ℕ → ℝ)
variables (O A B C : Type) [AddCommGroup O] [AddCommGroup A] [AddCommGroup B] [AddCommGroup C]
variables [Module ℝ O] [Module ℝ A] [Module ℝ B] [Module ℝ C]
variables (a3 a2014 : ℝ)

/- Collinear condition and given equation -/
axiom collinear : collinear A (λ i, (if i = 0 then 0 else if i = 1 then a 3 else a 2014) • (λ x : ℝ, x • A))
axiom vector_equation : B = a 3 • A + a 2014 • C

/- Define sum of sequence -/
axiom seq_sum : ∀ n : ℕ, S n = (n * (a 1 + a n)) / 2

/- Prove that the sum of the first 2016 terms is 1008 -/
theorem sum_of_first_2016_terms (h : arithmetic_sequence a) (h2 : a 3 + a 2014 = 1) : S 2016 = 1008 :=
sorry

end sum_of_first_2016_terms_l82_82564


namespace households_with_dvd_player_l82_82972

noncomputable def numHouseholds : ℕ := 100
noncomputable def numWithCellPhone : ℕ := 90
noncomputable def numWithMP3Player : ℕ := 55
noncomputable def greatestWithAllThree : ℕ := 55 -- maximum x
noncomputable def differenceX_Y : ℕ := 25 -- x - y = 25

def numberOfDVDHouseholds : ℕ := 15

theorem households_with_dvd_player : ∀ (D : ℕ),
  D + 25 - D = 55 - 20 →
  D = numberOfDVDHouseholds :=
by
  intro D h
  sorry

end households_with_dvd_player_l82_82972


namespace imo_23rd_Problem_l82_82298

theorem imo_23rd_Problem 
    (S : Type) [metric_space S] [normed_group S] [bounded_space S]
    (L : ℕ → S)
    (h_square : ∀ i j, dist (L i) (L j) ≤ 100)
    (h_non_intersect : ∀ i j, i ≠ j → L i ≠ L j)
    (h_boundary : ∀ P ∈ boundary S, ∃ i, dist P (L i) ≤ 1/2)
    : ∃ X Y : ℕ, dist (L X) (L Y) ≤ 1 ∧ (198 ≤ path_dist L X Y) :=
  sorry

end imo_23rd_Problem_l82_82298


namespace sum_of_coordinates_reflection_l82_82333

def point (x y : ℝ) : Type := (x, y)

variable (y : ℝ)

theorem sum_of_coordinates_reflection :
  let A := point 3 y in
  let B := point 3 (-y) in
  A.1 + A.2 + B.1 + B.2 = 6 :=
by
  sorry

end sum_of_coordinates_reflection_l82_82333


namespace smallest_x_value_l82_82217

theorem smallest_x_value (x y : ℕ) (h1 : 0.75 = y / (210 + x)) : x = 2 :=
by
  sorry

end smallest_x_value_l82_82217


namespace cost_in_chinese_yuan_l82_82327

theorem cost_in_chinese_yuan
  (usd_to_nad : ℝ := 8)
  (usd_to_cny : ℝ := 5)
  (sculpture_cost_nad : ℝ := 160) :
  sculpture_cost_nad / usd_to_nad * usd_to_cny = 100 := 
by
  sorry

end cost_in_chinese_yuan_l82_82327


namespace find_m_parallel_l82_82533

theorem find_m_parallel (m : ℝ) (h : (m, 1) ∥ (2 - m, -2)) : m = -2 :=
sorry

end find_m_parallel_l82_82533


namespace wasting_water_notation_l82_82233

theorem wasting_water_notation (saving_wasting : ℕ → ℤ)
  (h_pos : saving_wasting 30 = 30) :
  saving_wasting 10 = -10 :=
by
  sorry

end wasting_water_notation_l82_82233


namespace polynomial_zero_l82_82282

noncomputable def poly_P (n : ℕ) : EuclideanSpace ℝ (Fin n) → ℝ := sorry

theorem polynomial_zero (n : ℕ) (P : EuclideanSpace ℝ (Fin n) → ℝ)
  (h1 : ∀ x : EuclideanSpace ℝ (Fin n), (∑ i : Fin n, (∂² (P x) (x i) (x i))) = 0)
  (h2 : ∃ k : ℕ, ∃ Q : EuclideanSpace ℝ (Fin n) → ℝ, ∀ x, P x = (∑ i, x i^2)^k * Q x) :
  P = λ x, 0 :=
by {
  sorry
}

end polynomial_zero_l82_82282


namespace alloy_chromium_amount_l82_82616

theorem alloy_chromium_amount
  (x : ℝ) -- The amount of the first alloy used (in kg)
  (h1 : 0.10 * x + 0.08 * 35 = 0.086 * (x + 35)) -- Condition based on percentages of chromium
  : x = 15 := 
by
  sorry

end alloy_chromium_amount_l82_82616


namespace equilateral_triangles_in_decagon_l82_82168

-- Definition of the polygon and vertices
def is_regular_polygon (vertices : List Point) (n : ℕ) : Prop :=
  vertices.length = n ∧ --- the polygon has n sides
  -- all sides have equal length
  (∀ i j k, (vertices i distance vertices j = vertices j distance vertices k)) ∧
  -- all angles are equal
  (∀ i, (∠(vertices i vertices (i+1)%n vertices (i+2)%n) = 2*π/n))

-- The specific problem conditions
def decagon_vertices : List Point := [B_1, B_2, B_3, B_4, B_5, B_6, B_7, B_8, B_9, B_10]

axiom decagon_is_regular : is_regular_polygon decagon_vertices 10

-- Proof statement
theorem equilateral_triangles_in_decagon : 
  exists (T : Finset (Triangle Point)), T.card = 70 ∧ 
  ∀ t ∈ T, 
    (∀ v ∈ t.vertices, v ∈ decagon_vertices)  ∧ 
    is_equilateral t :=
sorry

end equilateral_triangles_in_decagon_l82_82168


namespace range_of_lambda_l82_82185

variable (λ : ℝ) (a s : ℕ → ℝ)

-- Assume the sum of the first n terms
def s_n (n : ℕ) : ℝ := 3^n * (λ - n) - 6

-- Define the sequence a_n
def a_n (n : ℕ) : ℝ := s n - s (n-1)

-- Monotonically decreasing sequence condition
def monotonically_decreasing (a : ℕ → ℝ) : Prop := 
  ∀ n, a n > a (n + 1)

-- Given conditions
axiom sequence_conditions : monotonically_decreasing a ∧ 
  (∀ n, s n = 3^n * (λ - n) - 6)

theorem range_of_lambda : λ < 2 := by
  sorry

end range_of_lambda_l82_82185


namespace find_x1_l82_82462

noncomputable def cdf (x : ℝ) : ℝ :=
  1 / 2 + 1 / Real.pi * Real.arctan (x / 2)

def satisfies_prob_condition (P : ℝ → ℝ) (x : ℝ) : Prop :=
  (1 - P x) = 1 / 4

theorem find_x1 :
  ∃ x1 : ℝ, satisfies_prob_condition cdf x1 ∧ x1 = 2 :=
by {
  use 2,
  split,
  {
    rw [satisfies_prob_condition, cdf],
    norm_num,
    conv {to_rhs, norm_num1},
    norm_num1,
    exact True.intro,
  },
  {
    exact rfl,
  },
}

end find_x1_l82_82462


namespace domain_of_f_l82_82356

noncomputable def f (x : ℝ) : ℝ := 1 / x + Real.sqrt (-x^2 + x + 2)

theorem domain_of_f :
  {x : ℝ | -1 ≤ x ∧ x ≤ 2 ∧ x ≠ 0} = {x : ℝ | -1 ≤ x ∧ x ≤ 2 ∧ x ≠ 0} :=
by
  sorry

end domain_of_f_l82_82356


namespace cost_in_chinese_yuan_l82_82328

theorem cost_in_chinese_yuan
  (usd_to_nad : ℝ := 8)
  (usd_to_cny : ℝ := 5)
  (sculpture_cost_nad : ℝ := 160) :
  sculpture_cost_nad / usd_to_nad * usd_to_cny = 100 := 
by
  sorry

end cost_in_chinese_yuan_l82_82328


namespace option_A_correct_option_B_correct_option_C_incorrect_option_D_correct_combined_correct_options_l82_82771

theorem option_A_correct : ∀ x : ℝ, x^2 > -1 ↔ (¬ ∃ x : ℝ, x^2 ≤ -1) :=
begin
  sorry
end

theorem option_B_correct : (∃ x : ℝ, -3 < x ∧ x^2 ≤ 9) ↔ (¬ ∀ x : ℝ, -3 < x → x^2 > 9) :=
begin
  sorry
end

theorem option_C_incorrect : ¬ (∀ x y : ℝ, |x| > |y| → x > y) :=
begin
  sorry
end

theorem option_D_correct : ∀ m : ℝ, (∃ x : ℝ, x^2 - 2*x + m = 0 ∧ x > 0 ∧ ∃ y : ℝ, y^2 - 2*y + m = 0 ∧ y < 0) ↔ (m < 0) :=
begin
  sorry
end

theorem combined_correct_options : {A, B, D} = {A, B, C, D}.filter (λ o, o ∈ {'A', 'B', 'D'}) :=
begin
  sorry
end

end option_A_correct_option_B_correct_option_C_incorrect_option_D_correct_combined_correct_options_l82_82771


namespace sum_of_roots_of_equation_l82_82104

theorem sum_of_roots_of_equation :
  let f := (3 * (x: ℝ) + 4) * (x - 5) + (3 * x + 4) * (x - 7)
  ∀ x : ℝ, f = 0 → ∑ (roots : ℝ) in {x | f x = 0}, x = -2 :=
by {
  let f := (3 * (x: ℝ) + 4) * (x - 5) + (3 * x + 4) * (x - 7),
  sorry
}

end sum_of_roots_of_equation_l82_82104


namespace overall_profit_no_discount_l82_82828

theorem overall_profit_no_discount:
  let C_b := 100
  let C_p := 100
  let C_n := 100
  let profit_b := 42.5 / 100
  let profit_p := 35 / 100
  let profit_n := 20 / 100
  let S_b := C_b + (C_b * profit_b)
  let S_p := C_p + (C_p * profit_p)
  let S_n := C_n + (C_n * profit_n)
  let TCP := C_b + C_p + C_n
  let TSP := S_b + S_p + S_n
  let OverallProfit := TSP - TCP
  let OverallProfitPercentage := (OverallProfit / TCP) * 100
  OverallProfitPercentage = 32.5 :=
by sorry

end overall_profit_no_discount_l82_82828


namespace triangle_range_condition_l82_82602

def triangle_side_range (x : ℝ) : Prop :=
  (1 < x) ∧ (x < 17)

theorem triangle_range_condition (x : ℝ) (a : ℝ) (b : ℝ) :
  (a = 8) → (b = 9) → triangle_side_range x :=
by
  intros h1 h2
  dsimp [triangle_side_range]
  sorry

end triangle_range_condition_l82_82602


namespace b_squared_eq_99_75_l82_82446

noncomputable section

variables {z : ℂ} {a b : ℝ}

def f (z : ℂ) : ℂ := (a + b * Complex.I) * z^2

theorem b_squared_eq_99_75 
  (h1 : ∀ z : ℂ, |f z - z^2| = |f z|)
  (h2 : Complex.abs (a + b * Complex.I) = 10) : 
  b^2 = 99.75 :=
sorry

end b_squared_eq_99_75_l82_82446


namespace exists_number_with_digits_1_and_2_divisible_by_2_pow_l82_82336

theorem exists_number_with_digits_1_and_2_divisible_by_2_pow (n : ℕ) : 
  ∃ A, (∀ d ∈ digits 10 A, d = 1 ∨ d = 2) ∧ 2^n ∣ A :=
by
  sorry

end exists_number_with_digits_1_and_2_divisible_by_2_pow_l82_82336


namespace brayan_hourly_coffee_l82_82990

theorem brayan_hourly_coffee (I B : ℕ) (h1 : B = 2 * I) (h2 : I + B = 30) : B / 5 = 4 :=
by
  sorry

end brayan_hourly_coffee_l82_82990


namespace river_flow_volume_l82_82777

theorem river_flow_volume
  (depth : ℝ) (width : ℝ) (flow_rate_kmph : ℝ)
  (h_depth : depth = 3)
  (h_width : width = 36)
  (h_flow_rate_kmph : flow_rate_kmph = 2) :
  let flow_rate_mpm := (flow_rate_kmph * 1000) / 60 in
  let area := depth * width in
  let volume_per_minute := area * flow_rate_mpm in
  volume_per_minute = 3599.64 :=
by
  sorry

end river_flow_volume_l82_82777


namespace variance_new_sample_l82_82187

-- Definitions and conditions:
def sample_variance (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (1 / n) * (∑ i in finset.range n, (a i - (1 / n) * ∑ j in finset.range n, a j) ^ 2)

variables (a1 a2 a3 : ℝ)

-- Given condition: Variance of the sample {a1, a2, a3} is 2
variable (h_var_a : sample_variance ![a1, a2, a3] 3 = 2)

-- Define the new sample with a linear transformation
def new_sample : ℕ → ℝ
| 0 => 2 * a1 + 3
| 1 => 2 * a2 + 3
| 2 => 2 * a3 + 3
| _ => 0  -- Assuming we work within the index range {0, 1, 2}

-- The proof statement
theorem variance_new_sample : sample_variance new_sample 3 = 8 :=
sorry

end variance_new_sample_l82_82187


namespace ellipse_properties_ellipse_C1_standard_eq_l82_82529

theorem ellipse_properties : 
  ∀ (x y : ℝ), (x^2 / 4 + y^2 / 3 = 1) →
    (∀ (C1 : ℝ → ℝ → bool),
      (∀ x y, C1 x y → (x^2 / m^2 + y^2 / n^2 = 1) → 
      (∃ m n : ℝ, (m > n) ∧ (m^2 - n^2 = 2) ∧ (m^2 = 8) ∧ (n^2 = 6))))) →
    ∃ (m n : ℝ), (m > n ∧ m^2 - n^2 = 2 ∧ m = 2 ∧ n = sqrt 3) := 
begin
  sorry
end

theorem ellipse_C1_standard_eq : 
  (∀ (x y : ℝ), (x^2 / 4 + y^2 / 3 = 1) →
    (∃ m n : ℝ, m > n ∧ m^2 - n^2 = 2 ∧ (∀ (a b : ℝ), a = 2 ∧ b = sqrt 3 ∧ ((x^2 / (m^2 + y^2 / n^2) = 1) = (x^2 / (25/4) + y^2 / (21/4) = 1))))) := 
begin
  sorry
end

end ellipse_properties_ellipse_C1_standard_eq_l82_82529


namespace dataset_min_range_l82_82041

noncomputable def smallest_possible_range (x : Fin 7 → ℕ) : ℕ :=
  let x_sorted := List.sort (Finset.univ.image x).toList
  x_sorted.getLast x_sorted.length (by simp) - x_sorted.head (by simp)

theorem dataset_min_range :
  ∃ (x : Fin 7 → ℕ), (x 3 = 12) ∧ (x 4 = 18) ∧ ((sum univ x).val.makeSum = 105) ∧
    (smallest_possible_range x = 5) :=
by {
  let x := !([12, 12, 12, 18, 17, 17, 17] : Fin 7 → ℕ),
  use x,
  split,
  {
    exact rfl
  },
  split,
  {
    exact rfl
  },
  split,
  {
    simp [sum, Finset.sum_univ_succ],
    exact rfl
  },
  {
    simp [smallest_possible_range],
    exact rfl
  },
  sorry
}

end dataset_min_range_l82_82041


namespace train_passes_tree_in_18_seconds_l82_82472

noncomputable def train_length : ℝ := 250 -- in meters
noncomputable def train_speed_kmh : ℝ := 50 -- in km/hr
noncomputable def conversion_factor : ℝ := 5 / 18 -- from km/hr to m/s

noncomputable def train_speed_ms : ℝ := train_speed_kmh * conversion_factor
noncomputable def time_to_pass_tree : ℝ := train_length / train_speed_ms

theorem train_passes_tree_in_18_seconds:
  abs (time_to_pass_tree - 18) < 1 :=
by
  -- Definitions of the problem to be used
  have h_conversion : train_speed_ms = 50 * (5 / 18) := rfl
  have h_distance : train_length = 250 := rfl
  
  -- Calculation bounds to ensure correctness can be checked with actual values
  have h_time : time_to_pass_tree ≈ 18 := by {
    -- this is where the solution steps would be formalized
    sorry
  }
  
  show abs (time_to_pass_tree - 18) < 1, from sorry

end train_passes_tree_in_18_seconds_l82_82472


namespace sum_of_numbers_l82_82400

theorem sum_of_numbers (a b : ℝ)
  (h₁ : a > 0) (h₂ : b > 0)
  (h₃ : a = 4 * (b/4)^(1/2))
  (h₄ : b = a + 4)
  (h₅ : 16 = b + (b - a)) :
  a + b = 8 + 4 * real.sqrt 5 :=
by sorry

end sum_of_numbers_l82_82400


namespace pure_imaginary_solutions_l82_82750

theorem pure_imaginary_solutions :
  ∃ x : ℂ, 
    (x^6 - 6*x^5 + 15*x^4 - 20*x^3 + 27*x^2 - 18*x - 8 = 0) ∧ 
    (∃ y : ℝ, x = y * complex.I) ∧
    (y = 0 ∨ y = complex.I * sqrt ((-5 + sqrt 52) / 3) ∨ y = -complex.I * sqrt ((-5 + sqrt 52) / 3)) :=
sorry

end pure_imaginary_solutions_l82_82750


namespace final_amount_loss_l82_82455

def initial_amount : ℝ := 64
def wins : ℕ := 3
def losses : ℕ := 3
def win_multiplier : ℝ := 3 / 2
def loss_multiplier : ℝ := 1 / 2

theorem final_amount (init : ℝ) (w : ℕ) (l : ℕ) (wm : ℝ) (lm : ℝ) :
  init = initial_amount →
  w = wins →
  l = losses →
  wm = win_multiplier →
  lm = loss_multiplier →
  init * (wm ^ w) * (lm ^ l) = 27 :=
by
sorrry

theorem loss (final : ℝ) :
  final = 27 →
  (initial_amount - final) = 37 :=
by
sorrry

end final_amount_loss_l82_82455


namespace jane_last_day_vases_l82_82272

theorem jane_last_day_vases (vases_per_day : ℕ) (total_vases : ℕ) (days : ℕ) (day_arrange_total: days = 17) (vases_per_day_is_25 : vases_per_day = 25) (total_vases_is_378 : total_vases = 378) :
  (vases_per_day * (days - 1) >= total_vases) → (total_vases - vases_per_day * (days - 1)) = 0 :=
by
  intros h
  -- adding this line below to match condition ": (total_vases - vases_per_day * (days - 1)) = 0"
  sorry

end jane_last_day_vases_l82_82272


namespace trigonometric_identity_l82_82956

theorem trigonometric_identity (θ : ℝ) (h : Real.sin θ + Real.cos θ = Real.sqrt 2) :
  Real.tan θ + 1 / Real.tan θ = 2 :=
by
  sorry

end trigonometric_identity_l82_82956


namespace find_sum_of_solutions_sum_of_all_solutions_l82_82524

theorem find_sum_of_solutions (y : ℝ) (h : y + 49 / y = 14) : y = 7 :=
  sorry

theorem sum_of_all_solutions : ∑ y in {y | y + 49 / y = 14}, y = 14 :=
  sorry

end find_sum_of_solutions_sum_of_all_solutions_l82_82524


namespace minimal_k_condition_l82_82653

-- Definitions for the problem conditions
def S (n : ℕ) : Type :=
  fin n

variables (n k : ℕ) (A : fin k → set (S n))

-- The main theorem statement
theorem minimal_k_condition (h : k > log n / log 2) :
  ∃ (B : fin k → set (S n)),
    (∀ i, B i = A i ∨ B i = (S n \ A i)) ∧ (⋃ i, B i) = set.univ :=
sorry

end minimal_k_condition_l82_82653


namespace discount_percentage_l82_82018

theorem discount_percentage (discount amount_paid : ℝ) (h_discount : discount = 40) (h_paid : amount_paid = 120) : 
  (discount / (discount + amount_paid)) * 100 = 25 := by
  sorry

end discount_percentage_l82_82018


namespace dress_hat_color_correct_l82_82760

-- Define the types and variables
inductive Color
| pink
| purple
| turquoise

-- Define variables for the dresses and hats
def vera_dress : Color := Color.pink
def vera_hat : Color := Color.pink
def nadia_dress : Color := Color.purple
def nadia_hat : Color := Color.turquoise
def lyuba_dress : Color := Color.turquoise
def lyuba_hat : Color := Color.purple

-- Define the conditions
def condition1 := vera_dress = vera_hat
def condition2 := nadia_dress ≠ Color.pink ∧ nadia_hat ≠ Color.pink
def condition3 := lyuba_hat = Color.purple

-- Proof placeholder
theorem dress_hat_color_correct :
  vera_dress = Color.pink ∧ vera_hat = Color.pink ∧
  nadia_dress = Color.purple ∧ nadia_hat = Color.turquoise ∧
  lyuba_dress = Color.turquoise ∧ lyuba_hat = Color.purple
:= by
  unfold condition1 condition2 condition3
  split1
  sorry

end dress_hat_color_correct_l82_82760


namespace pos_int_fraction_iff_l82_82223

theorem pos_int_fraction_iff (p : ℕ) (hp : p > 0) : (∃ k : ℕ, 4 * p + 11 = k * (2 * p - 7)) ↔ (p = 4 ∨ p = 5) := 
sorry

end pos_int_fraction_iff_l82_82223


namespace fraction_is_one_fifth_l82_82887

theorem fraction_is_one_fifth
  (x a b : ℤ)
  (hx : x^2 = 25)
  (h2x : 2 * x = a * x / b + 9) :
  a = 1 ∧ b = 5 :=
by
  sorry

end fraction_is_one_fifth_l82_82887


namespace zeros_in_expansion_of_large_number_l82_82951

noncomputable def count_zeros (n : ℕ) : ℕ :=
  (n.to_string.filter (λ c, c = '0')).length

theorem zeros_in_expansion_of_large_number :
  count_zeros ((999999999999997 : ℕ)^2) = 29 :=
by {
  -- We'll provide the actual proof here
  sorry
}

end zeros_in_expansion_of_large_number_l82_82951


namespace at_least_one_alarm_rings_on_time_l82_82667

-- Definitions for the problem
def prob_A : ℝ := 0.5
def prob_B : ℝ := 0.6

def prob_not_A : ℝ := 1 - prob_A
def prob_not_B : ℝ := 1 - prob_B
def prob_neither_A_nor_B : ℝ := prob_not_A * prob_not_B
def prob_at_least_one : ℝ := 1 - prob_neither_A_nor_B

-- Final statement
theorem at_least_one_alarm_rings_on_time : prob_at_least_one = 0.8 :=
by sorry

end at_least_one_alarm_rings_on_time_l82_82667


namespace nested_sqrt_limit_l82_82512

theorem nested_sqrt_limit : ∃ x : Real, x = sqrt (3 - sqrt (3 - sqrt (3 - sqrt (3 - ⋯)))) ∧ x = (sqrt 13 - 1) / 2 := by
  sorry

end nested_sqrt_limit_l82_82512


namespace probability_divisible_by_2_l82_82831

-- Definition of the problem statement
def digits : List ℕ := [1, 2, 3, 4]

-- Definition of total permutations of three-digit numbers without repetition
def total_three_digit_numbers (d : List ℕ) : ℕ :=
  List.length (List.permutations d).filter (λ x, List.length x = 3)

-- Definition of the numbers that are divisible by 2
def divisible_by_two (d : List ℕ) : ℕ :=
  List.length (List.filter (λ x, x % 2 = 0) (List.permutations d).filter (λ x, List.length x = 3))

theorem probability_divisible_by_2 : 
  let total := total_three_digit_numbers digits
  let div_by_2 := divisible_by_two digits
  (total = 24) ∧ (div_by_2 = 12) → ((div_by_2 : ℚ) / (total : ℚ)) = (1 / 2) :=
by
  sorry

end probability_divisible_by_2_l82_82831


namespace tammy_laps_per_day_l82_82704

theorem tammy_laps_per_day :
  ∀ (total_distance_per_week distance_per_lap days_in_week : ℕ), 
  total_distance_per_week = 3500 → 
  distance_per_lap = 50 → 
  days_in_week = 7 → 
  (total_distance_per_week / distance_per_lap) / days_in_week = 10 :=
by
  intros total_distance_per_week distance_per_lap days_in_week h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end tammy_laps_per_day_l82_82704


namespace ab_value_l82_82341

theorem ab_value (a b : ℝ) (h1 : a + b = 5) (h2 : a^3 * b^2 + a^2 * b^3 = 20) : ab = 2 ∨ ab = -2 :=
by
  sorry

end ab_value_l82_82341


namespace first_1500_even_integers_digits_l82_82493

def numOfDigits (n: ℕ) : ℕ :=
  if n < 10 then 1
  else if n < 100 then 2
  else if n < 1000 then 3
  else if n < 10000 then 4
  else 0

def totalDigitsUsedUpTo (n: ℕ) : ℕ :=
  (List.range n).map (λ x => 2 * x + 2).map numOfDigits |>.sum

theorem first_1500_even_integers_digits :
  totalDigitsUsedUpTo 1500 = 5448 :=
by
  sorry

end first_1500_even_integers_digits_l82_82493


namespace Sherlock_Holmes_correct_l82_82675

def sum_of_divisors (n : ℕ) : ℕ := ∑ d in (list.range (n + 1)), if n % d = 0 then d else 0

def sum_of_reciprocals (n : ℕ) : ℚ := ∑ d in (list.range (n + 1)), if n % d = 0 then (1:ℚ)/d else 0

theorem Sherlock_Holmes_correct (n m : ℕ) 
  (h1 : sum_of_divisors n = sum_of_divisors m) 
  (h2 : sum_of_reciprocals n = sum_of_reciprocals m) : 
  n = m := 
sorry

end Sherlock_Holmes_correct_l82_82675


namespace cyclists_meeting_l82_82155

-- Define the velocities of the cyclists and the time variable
variables (v₁ v₂ t : ℝ)

-- Define the conditions for the problem
def condition1 : Prop := v₁ * t = v₂ * (2/3)
def condition2 : Prop := v₂ * t = v₁ * 1.5

-- Define the main theorem to be proven
theorem cyclists_meeting (h1 : condition1 v₁ v₂ t) (h2 : condition2 v₁ v₂ t) :
  t = 1 ∧ (v₁ / v₂ = 3 / 2) :=
by sorry

end cyclists_meeting_l82_82155


namespace chess_tournament_l82_82242

theorem chess_tournament (n : ℕ) (h : (n * (n - 1)) / 2 - ((n - 3) * (n - 4)) / 2 = 130) : n = 19 :=
sorry

end chess_tournament_l82_82242


namespace root_of_quadratic_l82_82549

theorem root_of_quadratic (b : ℝ) : 
  (-9)^2 + b * (-9) - 45 = 0 -> b = 4 :=
by
  sorry

end root_of_quadratic_l82_82549


namespace number_of_women_bathing_suits_correct_l82_82034

def men_bathing_suits : ℕ := 14797
def total_bathing_suits : ℕ := 19766

def women_bathing_suits : ℕ :=
  total_bathing_suits - men_bathing_suits

theorem number_of_women_bathing_suits_correct :
  women_bathing_suits = 19669 := by
  -- proof goes here
  sorry

end number_of_women_bathing_suits_correct_l82_82034


namespace marble_problem_l82_82449

def marble_prob := (red green blue white : ℕ) (draws : ℕ) : Prop := 
  red = 3 ∧ green = 4 ∧ blue = 8 ∧ white = 5 ∧ draws = 2 → 
  let total_marbles := red + green + blue + white in
  let prob_first_blue := (blue : ℚ) / (total_marbles : ℚ) in
  let prob_second_blue := (blue - 1 : ℚ) / (total_marbles - 1 : ℚ) in
  prob_first_blue * prob_second_blue = 14 / 95

theorem marble_problem : marble_prob 3 4 8 5 2 :=
by
  sorry

end marble_problem_l82_82449


namespace scientific_notation_l82_82029

theorem scientific_notation (n : ℕ) (h : n = 13976000) : 
  ∃ a b : ℝ, 1 ≤ a ∧ a < 10 ∧ b ∈ ℤ ∧ n = a * 10 ^ b ∧ a = 1.3976 ∧ b = 7 := 
by sorry

end scientific_notation_l82_82029


namespace circumscribed_sphere_radius_l82_82986

theorem circumscribed_sphere_radius (a : ℝ) 
  (A B C D : EuclideanSpace ℝ (Fin 3))
  (h_eq : dist A B = a)
  (h_A_CB : angle A C B = π / 2)
  (h_A_DB : angle A D B = π / 2) :
  ∃ M : EuclideanSpace ℝ (Fin 3), 
    dist M A = dist M B ∧ 
    dist M C = dist M D ∧
    dist M A = a / 2 :=
begin
  sorry
end

end circumscribed_sphere_radius_l82_82986


namespace drum_size_is_54_99_l82_82447

noncomputable def size_of_drum : ℝ :=
  let x := D - 6.11
  in if (6.11 + 0.10 * x = 0.20 * D) ∧ (6.11 + x = D)
     then D
     else sorry

theorem drum_size_is_54_99 :
  ∃ D : ℝ, 
    (6.11 + 0.10 * (D - 6.11) = 0.20 * D) ∧ 
    (6.11 + (D - 6.11) = D) ∧ 
    (D = 54.99) :=
begin
  use 54.99,
  split,
  {
    norm_num,
    rw [← add_sub_assoc],
    ring,
  },
  split,
  {
    ring_nf,
  },
  {
    refl,
  }
end

end drum_size_is_54_99_l82_82447


namespace perimeter_sequence_area_sequence_l82_82968

-- Define the geometric sequence for perimeters
theorem perimeter_sequence (P_0 : ℝ) : ∀ n : ℕ, 
  let P_n := P_0 * (1 / 2) ^ n in P_n = P_0 * (1 / 2) ^ n :=
by sorry

-- Define the geometric sequence for areas
theorem area_sequence (A_0 : ℝ) : ∀ n : ℕ, 
  let A_n := A_0 * (1 / 4) ^ n in A_n = A_0 * (1 / 4) ^ n :=
by sorry

end perimeter_sequence_area_sequence_l82_82968


namespace petya_addition_mistake_l82_82680

theorem petya_addition_mistake:
  ∃ (x y c : ℕ), x + y = 12345 ∧ (10 * x + c) + y = 44444 ∧ x = 3566 ∧ y = 8779 ∧ c = 5 := by
  sorry

end petya_addition_mistake_l82_82680


namespace points_on_line_l82_82151

theorem points_on_line (t : ℝ) : ∀ t : ℝ, let x := Real.sin t ^ 2 in let y := Real.cos t ^ 2 in x + y = 1 :=
by
  intro t
  let x := Real.sin t ^ 2
  let y := Real.cos t ^ 2
  have h : Real.sin t ^ 2 + Real.cos t ^ 2 = 1 := sorry
  exact h

end points_on_line_l82_82151


namespace double_probability_of_three_l82_82809

theorem double_probability_of_three : 
  log10 (7 / 5 : ℝ) + log10 (13 / 11 : ℝ) + log10 (15 / 13 : ℝ) = log10 (25 / 9 : ℝ) :=
by sorry

end double_probability_of_three_l82_82809


namespace total_commercial_time_proof_l82_82611

def program_durations : List ℕ := [30, 30, 30, 30, 30, 30, 30, 30]

def commercial_percentages : List ℝ := [0.20, 0.25, 0.00, 0.35, 0.40, 0.45, 0.15, 0.00]

def commercial_times (durations : List ℕ) (percentages : List ℝ) : List ℝ :=
  List.map₂ (λ d p => d * p) durations percentages

def total_commercial_time (times : List ℝ) : ℝ :=
  List.sum times

theorem total_commercial_time_proof :
  total_commercial_time (commercial_times program_durations commercial_percentages) = 54 :=
by sorry

end total_commercial_time_proof_l82_82611


namespace work_duration_l82_82783

noncomputable def P_work_days := 40
noncomputable def Q_work_days := 24
noncomputable def P_alone_days := 8

theorem work_duration (W : ℕ) :
  let P_rate := W / P_work_days,
      Q_rate := W / Q_work_days,
      work_done_by_P := P_alone_days * P_rate,
      remaining_work := W - work_done_by_P,
      combined_rate := P_rate + Q_rate,
      D := remaining_work / combined_rate
  in P_alone_days + D = 20 := sorry

end work_duration_l82_82783


namespace power_log_base_self_l82_82117

theorem power_log_base_self (a N : ℝ) (h : a > 0 ∧ a ≠ 1) (hN : N > 0) : a^log a N = N :=
by
  sorry

example : 3 ^ log 3 4 = 4 :=
by
  apply power_log_base_self
  exact ⟨by norm_num, by norm_num⟩
  norm_num

end power_log_base_self_l82_82117


namespace decimal_properties_l82_82791

theorem decimal_properties :
  (3.00 : ℝ) = (3 : ℝ) :=
by sorry

end decimal_properties_l82_82791


namespace set_addition_example_1_inequality_Snk_exists_self_generating_and_basis_set_l82_82309

-- Definitions
def add_sets (A B : set ℝ) : set ℝ := {c | ∃ a b, a ∈ A ∧ b ∈ B ∧ c = a + b}

-- Question (1)
theorem set_addition_example_1 : add_sets {0, 1, 2} {-1, 3} = {-1, 0, 1, 3, 4, 5} :=
sorry

-- Question (2)
noncomputable def a_n (n : ℕ) : ℝ := (2 / 3) * n
noncomputable def S_n (n : ℕ) : ℝ := n^2

theorem inequality_Snk (m n k : ℕ) (h1 : m + n = 3 * k) (h2 : m ≠ n) : 
  S_n m + S_n n - (9 / 2) * S_n k > 0 :=
sorry

-- Question (3)
-- Definitions for self-generating set and N* basis set
def is_self_generating_set (A : set ℤ) : Prop :=
∀ a ∈ A, ∃ b c ∈ A, a = b + c

def is_N_star_basis_set (A : set ℤ) : Prop :=
∀ n : ℕ, n > 0 → ∃ S : finset ℤ, (∀ x ∈ S, x ∈ A ∧ x ≠ 0) ∧ S.sum id = n

theorem exists_self_generating_and_basis_set : 
  ∃ A : set ℤ, is_self_generating_set A ∧ is_N_star_basis_set A :=
sorry

end set_addition_example_1_inequality_Snk_exists_self_generating_and_basis_set_l82_82309


namespace M_is_correct_ab_property_l82_82158

noncomputable def f (x : ℝ) : ℝ := |x + 1| + |x - 1|
def M : Set ℝ := {x | f x < 4}

theorem M_is_correct : M = {x | -2 < x ∧ x < 2} :=
sorry

theorem ab_property (a b : ℝ) (ha : a ∈ M) (hb : b ∈ M) : 2 * |a + b| < |4 + a * b| :=
sorry

end M_is_correct_ab_property_l82_82158


namespace glenda_speed_is_8_l82_82484

noncomputable def GlendaSpeed : ℝ :=
  let AnnSpeed := 6
  let Hours := 3
  let Distance := 42
  let AnnDistance := AnnSpeed * Hours
  let GlendaDistance := Distance - AnnDistance
  GlendaDistance / Hours

theorem glenda_speed_is_8 : GlendaSpeed = 8 := by
  sorry

end glenda_speed_is_8_l82_82484


namespace problem_part1_problem_part2_l82_82199

def f (x : ℝ) : ℝ := x^2 / (1 + x^2)

theorem problem_part1 (x : ℝ) (hx : x ∈ ({2, 3, 4} : Set ℝ)) : 
  f(x) + f(1 / x) = 1 :=
by
  sorry

theorem problem_part2 : 
  (∑ i in Finset.range 2016, 2 * f (i + 2)) + 
  (∑ i in Finset.range 2016, f (1 / (i + 2))) + 
  (∑ i in Finset.range 2016, (1 / (i + 2)^2) * f (i + 2)) = 4032 :=
by
  sorry

end problem_part1_problem_part2_l82_82199


namespace round_table_pairs_l82_82344

theorem round_table_pairs :
  let f := λ (table : list (bool × bool)), table.countp (λ p, p.1)
  let m := λ (table : list (bool × bool)), table.countp (λ p, p.2)
  ∀ table : list (bool × bool), table.length = 7 →
    let pairs := (f table, m table)
    list.length (list.dedup [ pairs | w ← fin_range 8, let table := generate_table w ]) = 6 :=
by
  sorry

end round_table_pairs_l82_82344


namespace KT_is_tangent_to_Γ₁_l82_82652

open Classical
noncomputable theory

variables {Γ Γ₁ l : Type} [circle Γ] [circle Γ₁] [line l]
          {R S T J A K : Γ}

-- Given conditions
axiom distinct_points_RS : R ≠ S
axiom not_diameter_RS : ¬ is_diameter R S
axiom tangent_l_at_R : is_tangent l Γ R
axiom S_midpoint_RT : is_midpoint S R T
axiom J_minor_arc_RS : J ∈ minor_arc R S Γ
axiom circumcircle_JST: is_circumcircle Γ₁ J S T
axiom intersection_A_l : A ∈ l ∧ A ∈ Γ₁
axiom intersection_A_closer_to_R : A ∈ l ∧ (distance A R < distance A other_point) where other_point ∈ Γ₁ -- additional condition to ensure A is closer

-- Problem statement to prove
theorem KT_is_tangent_to_Γ₁ : is_tangent (line_through K T) Γ₁ :=
sorry

end KT_is_tangent_to_Γ₁_l82_82652


namespace integer_count_and_bounds_l82_82108

theorem integer_count_and_bounds :
  let M := {n : ℤ | (-50 ≤ n ∧ n ≤ 250) ∧ (n^3 - 2*n^2 - 13*n - 10) % 13 = 0} in
  ∃ smallest largest : ℤ, 
    (smallest = -47) ∧ 
    (largest = 245) ∧ 
    (∃ count : ℕ, count = 46 ∧ count = (M.toFinset.card)) := 
sorry

end integer_count_and_bounds_l82_82108


namespace value_of_w_div_x_l82_82566

theorem value_of_w_div_x (w x y : ℝ) 
  (h1 : w / x = a) 
  (h2 : w / y = 1 / 5) 
  (h3 : (x + y) / y = 2.2) : 
  w / x = 6 / 25 := by
  sorry

end value_of_w_div_x_l82_82566


namespace inequality_proof_l82_82591

theorem inequality_proof (a b : ℝ) (h1 : a < b) (h2 : b < 0) : a < a - b :=
by
  sorry

end inequality_proof_l82_82591


namespace tammy_laps_per_day_l82_82705

theorem tammy_laps_per_day :
  ∀ (total_distance_per_week distance_per_lap days_in_week : ℕ), 
  total_distance_per_week = 3500 → 
  distance_per_lap = 50 → 
  days_in_week = 7 → 
  (total_distance_per_week / distance_per_lap) / days_in_week = 10 :=
by
  intros total_distance_per_week distance_per_lap days_in_week h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end tammy_laps_per_day_l82_82705


namespace derivative_is_correct_l82_82714

def y (x : ℝ) : ℝ := sin (2 * x) - cos (2 * x)

theorem derivative_is_correct (x : ℝ) : 
  deriv y x = 2 * sqrt 2 * cos (2 * x - π / 4) :=
sorry

end derivative_is_correct_l82_82714


namespace contrapositive_of_real_roots_l82_82012

noncomputable def contrapositive_of_real_roots_problem : Prop :=
  ∀ m : ℕ, m > 0 → 
           (¬ ∃ x : ℝ, x^2 + x - (m : ℤ) = 0) → 
           (m ≤ 0)

theorem contrapositive_of_real_roots : contrapositive_of_real_roots_problem :=
begin
  sorry
end

end contrapositive_of_real_roots_l82_82012


namespace tan_neg_1125_eq_one_l82_82376

noncomputable def problem_statement : Prop :=
  tan (Real.pi * (-1125 / 180)) = 1

theorem tan_neg_1125_eq_one : problem_statement :=
by
  sorry

end tan_neg_1125_eq_one_l82_82376


namespace shifted_roots_polynomial_l82_82654

-- Define the original polynomial
def original_polynomial (x : ℝ) : ℝ :=
  x^3 - 5 * x + 7

-- Define the shifted polynomial
def shifted_polynomial (x : ℝ) : ℝ :=
  x^3 + 9 * x^2 + 22 * x + 19

-- Define the roots condition
def is_root (p : ℝ → ℝ) (r : ℝ) : Prop :=
  p r = 0

-- State the theorem
theorem shifted_roots_polynomial :
  ∀ a b c : ℝ,
    is_root original_polynomial a →
    is_root original_polynomial b →
    is_root original_polynomial c →
    is_root shifted_polynomial (a - 3) ∧
    is_root shifted_polynomial (b - 3) ∧
    is_root shifted_polynomial (c - 3) :=
by
  intros a b c ha hb hc
  sorry

end shifted_roots_polynomial_l82_82654


namespace c_n_minus_b_n_b_n_plus_c_n_const_range_p_l82_82311

-- Define the sequences based on the given conditions
def a : ℕ → ℝ
| 0     := 4
| (n+1) := a n

def b : ℕ → ℝ
| 0     := 3
| (n+1) := (a n + c n) / 2

def c : ℕ → ℝ
| 0     := 5
| (n+1) := (a n + b n) / 2

-- (1) Prove the formula for c_n - b_n
theorem c_n_minus_b_n (n : ℕ) : c n - b n = 2 * (-1/2)^(n-1) :=
sorry

-- (2) Prove b_n + c_n is constant
theorem b_n_plus_c_n_const (n : ℕ) : b n + c n = 8 :=
sorry

-- Helper definition for the sum of the first n terms of sequence c
def S : ℕ → ℝ
| 0     := c 0
| (n+1) := S n + c (n+1)

-- (3) Prove the range of p
theorem range_p (p : ℝ) (n : ℕ) (hp : 1 ≤ p * (S n - 4 * n) ∧ p * (S n - 4 * n) ≤ 3) :
  2 ≤ p ∧ p ≤ 3 :=
sorry

end c_n_minus_b_n_b_n_plus_c_n_const_range_p_l82_82311


namespace geometric_sequence_a₅_l82_82258

variable {α : Type*} [DivisionRing α] [Inhabited α]
variable {a₁ a₃ a₅ : α}
variable (r : α)  -- common ratio of the geometric sequence

-- Define the geometric sequence conditions
def a₁_def : a₁ = 3 := sorry
def a₃_def : a₃ = 12 := sorry

-- The theorem stating that a₅ = 48 if the sequence is geometric and the given conditions hold
theorem geometric_sequence_a₅ :
  a₁ = 3 → a₃ = 12 → a₅ = a₃ * a₃ / a₁ → a₅ = 48 :=
by
  intros h₁ h₃ h₅
  rw [a₁_def, a₃_def] at h₅
  exact h₅

end geometric_sequence_a₅_l82_82258


namespace nickel_chocolates_l82_82688

theorem nickel_chocolates (N : ℕ) (h : 7 = N + 2) : N = 5 :=
by
  sorry

end nickel_chocolates_l82_82688


namespace distance_between_skew_lines_l82_82300

theorem distance_between_skew_lines
  (l m : ℝ^3 → Prop) -- l and m are skew lines in ℝ^3
  (A B C D E F : ℝ^3) -- Points on the lines and perpendiculars
  (h1 : l A) (h2 : l B) (h3 : l C) -- A, B, C are on line l
  (h4 : m D) (h5 : m E) (h6 : m F) -- D, E, F are on line m
  (h7 : dist A B = dist B C) -- AB = BC
  (h8 : A.D = D) (h9 : B.E = E) (h10 : C.F = F) -- AD, BE, CF are perpendiculars from A, B, C to m
  (h11 : dist A D = sqrt 15)
  (h12 : dist B E = 7/2)
  (h13 : dist C F = sqrt 10) :
  dist_lines l m = sqrt 6 := 
sorry

end distance_between_skew_lines_l82_82300


namespace inscribed_circumscribed_surface_area_ratio_l82_82372

theorem inscribed_circumscribed_surface_area_ratio (a : ℝ) (h : a > 0) :
  let r := a / 2,
      R := (a * Real.sqrt 3) / 2,
      S_inscribed := 4 * Real.pi * (r^2),
      S_circumscribed := 4 * Real.pi * (R^2)
  in S_inscribed / S_circumscribed = 1 / 3 := 
by
  -- Proof to be filled in
  sorry


end inscribed_circumscribed_surface_area_ratio_l82_82372


namespace triangle_lines_concurrent_l82_82910

noncomputable def triangle_intersection (A B C D E F G H : Point) (h1 : Triangle A B C) 
(h2 : AcuteTriangle A B C) (h3 : FootOfAltitude A D) 
(h4 : Square D C G F) (h5 : Square B D E H) 
(h6 : OnLine E F (Line_through A D)) (h7 : OppositeSides H A (Line_through B C)) 
(h8 : OppositeSides G A (Line_through B C)) : Prop :=
  concurrent (Line_through A D) (Line_through B G) (Line_through H C)

theorem triangle_lines_concurrent (A B C D E F G H : Point) 
    (h1 : Triangle A B C) (h2 : AcuteTriangle A B C) 
    (h3 : FootOfAltitude A D) (h4 : Square D C G F) 
    (h5 : Square B D E H) (h6 : OnLine E F (Line_through A D)) 
    (h7 : OppositeSides H A (Line_through B C)) 
    (h8 : OppositeSides G A (Line_through B C)) : 
    triangle_intersection A B C D E F G H h1 h2 h3 h4 h5 h6 h7 h8 := 
sorry

end triangle_lines_concurrent_l82_82910


namespace pet_store_cages_l82_82458

theorem pet_store_cages 
  (initial_puppies : ℕ) 
  (sold_puppies : ℕ) 
  (puppies_per_cage : ℕ) 
  (h_initial_puppies : initial_puppies = 45) 
  (h_sold_puppies : sold_puppies = 11) 
  (h_puppies_per_cage : puppies_per_cage = 7) 
  : (initial_puppies - sold_puppies + puppies_per_cage - 1) / puppies_per_cage = 5 :=
by sorry

end pet_store_cages_l82_82458


namespace problem_conditions_l82_82575

-- Define f
noncomputable def f (x a : ℝ) : ℝ :=
  (1/2 * x^2 - 2 * x) * Real.log x + 3/4 * x^2 - (a + 1) * x + 1

-- Define g(a)
noncomputable def g (a : ℝ) : ℝ :=
  let m := (Exists.some (exists_unique (λ m, (m-2) * Real.log m + 2 * m - a - 3 = 0))).val
  (1/2 * m^2 - 2 * m) * Real.log m + 3/4 * m^2 - (a + 1) * m + 1

-- Prove the conditions
theorem problem_conditions (x a : ℝ) :
  (∀ (x : ℝ), 1 < x → ((x-2) * Real.log x + 2 * x - a - 3 ≥ 0) → a ≤ -1) ∧
  ( -1 < a ∧  a < 1 → ∃ m ∈ Ioo 1 2, g(a) ∈ Ioo (-2 * Real.log 2) (7 / 4)) :=
  by sorry

end problem_conditions_l82_82575


namespace contrapositive_of_real_roots_l82_82011

noncomputable def contrapositive_of_real_roots_problem : Prop :=
  ∀ m : ℕ, m > 0 → 
           (¬ ∃ x : ℝ, x^2 + x - (m : ℤ) = 0) → 
           (m ≤ 0)

theorem contrapositive_of_real_roots : contrapositive_of_real_roots_problem :=
begin
  sorry
end

end contrapositive_of_real_roots_l82_82011


namespace second_discarded_number_l82_82712

theorem second_discarded_number (S : ℝ) (X : ℝ) (h1 : S / 50 = 62) (h2 : (S - 45 - X) / 48 = 62.5) : X = 55 := 
by
  sorry

end second_discarded_number_l82_82712


namespace connect_four_cities_l82_82487

open Real

-- Define the problem conditions
def city (x y : ℝ) : Prop :=
  x ∈ {0, 4} ∧ y ∈ {0, 4}

-- Ensure side length condition
def is_square (A B C D : (ℝ × ℝ)) : Prop :=
  A.1 = 0 ∧ A.2 = 0 ∧ B.1 = 4 ∧ B.2 = 0 ∧ C.1 = 4 ∧ C.2 = 4 ∧ D.1 = 0 ∧ D.2 = 4

-- The hypothesis that the total road length is less than 11
def road_length (cities : Fin 4 → ℝ × ℝ) (paths : Fin 6 → Fin 2 → ℝ × ℝ) : ℝ :=
  ∑ i in Finset.univ, dist ((paths i) 0) ((paths i) 1)

-- The goal is to find a configuration with a total length less than 11
theorem connect_four_cities :
  ∃ (cities : Fin 4 → ℝ × ℝ) (paths : Fin 6 → Fin 2 → ℝ × ℝ),
    is_square (cities 0) (cities 1) (cities 2) (cities 3) ∧ road_length cities paths < 11 :=
begin
  sorry
end

end connect_four_cities_l82_82487


namespace smallest_number_l82_82479

theorem smallest_number (s: Set ℤ) (h: s = {2, 1, -1, -2}) : ∃ m ∈ s, ∀ x ∈ s, m ≤ x :=
by
  use -2
  split
  { -- Proof that -2 is in the set
    rw h,
    simp,
  }
  { -- Proof that -2 is the smallest element in the set
    intros x h2,
    rw h at h2,
    simp at h2,
    fin_cases h2,
    { rfl }, -- case x = -2
    { linarith }, -- case x = -1
    { linarith }, -- case x = 1
    { linarith }, -- case x = 2
  }

end smallest_number_l82_82479


namespace find_m_n_l82_82981

variables (OA OB OC : ℝ)
variables (m n : ℝ)

theorem find_m_n (h1 : ∥OA∥ = 2) 
                 (h2 : ∥OB∥ = 2) 
                 (h3 : ∥OC∥ = sqrt 2) 
                 (h4 : tan (angle A OC) = 3) 
                 (h5 : angle B OC = 45) :
  OC = m * OA + n * OB ∧ (m, n) = (1/3, 1/3) :=
by sorry

end find_m_n_l82_82981


namespace smallest_positive_period_of_f_interval_of_monotonicity_of_f_max_value_of_f_min_value_of_f_range_of_m_with_exactly_three_zeros_l82_82194

-- Definition of the function f(x) = 2sin(2x + π / 4)
def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 4)

-- Definitions related to part 1: smallest positive period and interval of monotonicity
theorem smallest_positive_period_of_f :
  ∀ x : ℝ, f (x + Real.pi) = f x := sorry

theorem interval_of_monotonicity_of_f (k : ℤ) :
  ∀ x : ℝ, (k * Real.pi - 3 * Real.pi / 8 ≤ x ∧ x ≤ k * Real.pi + Real.pi / 8) → 
    (∀ y : ℝ, (k * Real.pi - 3 * Real.pi / 8 < y ∧ y < x) → f y < f x) := sorry

-- Definitions related to part 2: maximum and minimum values within [-π/8, π/2]
theorem max_value_of_f :
  ∃ x : ℝ, x = Real.pi / 8 ∧ ∀ y : ℝ, (-Real.pi / 8 ≤ y ∧ y ≤ Real.pi / 2) → f y ≤ f x := sorry

theorem min_value_of_f :
  ∃ x : ℝ, x = Real.pi / 2 ∧ ∀ y : ℝ, (-Real.pi / 8 ≤ y ∧ y ≤ Real.pi / 2) → f y ≥ f x := sorry

-- Definitions related to part 3: range of m such that g(x) = f(x) - sqrt(2) has exactly 3 zeros in [-π/8, m]
def g (x : ℝ) : ℝ := f x - Real.sqrt 2

theorem range_of_m_with_exactly_three_zeros (m : ℝ) :
  (3 = (∃ a b c : ℝ, -Real.pi / 8 ≤ a ∧ a ≤ m ∧ -Real.pi / 8 ≤ b ∧ b ≤ m ∧ -Real.pi / 8 ≤ c ∧ c ≤ m ∧ g a = 0 ∧ g b = 0 ∧ g c = 0 ∧ a ≠ b ∧ a ≠ c ∧ b ≠ c)) ↔ (5 * Real.pi / 4 ≤ m ∧ m < Real.pi) := sorry

end smallest_positive_period_of_f_interval_of_monotonicity_of_f_max_value_of_f_min_value_of_f_range_of_m_with_exactly_three_zeros_l82_82194


namespace math_problem_l82_82641

variable {n : ℕ} (a : Finₓ (n + 1) → ℝ) (h : ∀ j, a j ≠ 1)

theorem math_problem (h : ∀ j : Finₓ (n + 1), a j ≠ 1) :
  a 0 + ∑ i in Finₓ.range n, a (Finₓ.succ i) * ∏ j in Finₓ.range i, (1 - a j) = 1 - ∏ j in Finₓ.range (n + 1), (1 - a j) :=
by
  sorry

end math_problem_l82_82641


namespace problem1_problem2_l82_82630

noncomputable def f (x a c : ℝ) : ℝ := -3 * x^2 + a * (6 - a) * x + c

-- Problem 1: Prove that for c = 19, the inequality f(1, a, 19) > 0 holds for -2 < a < 8
theorem problem1 (a : ℝ) : f 1 a 19 > 0 ↔ -2 < a ∧ a < 8 :=
by sorry

-- Problem 2: Given that f(x) > 0 has solution set (-1, 3), find a and c
theorem problem2 (a c : ℝ) (hx : ∀ x, -1 < x ∧ x < 3 → f x a c > 0) : 
  (a = 3 + Real.sqrt 3 ∨ a = 3 - Real.sqrt 3) ∧ c = 9 :=
by sorry

end problem1_problem2_l82_82630


namespace hyperbola_standard_equation1_hyperbola_standard_equation2_l82_82432

-- Problem 1
theorem hyperbola_standard_equation1 :
  (∃ a b : ℝ, (∀ x y : ℝ, (x, y) = (-5, 2) →
    (c^2 = a^2 + b^2) ∧ (c = real.sqrt 6) ∧ (x^2 / a^2 - y^2 / b^2 = 1)) → 
    (x^2 / 5 - y^2 / 1 = 1)) :=
sorry

-- Problem 2
theorem hyperbola_standard_equation2 :
  (∃ m n : ℝ, (∀ x y : ℝ, ((x, y) = (3, -4 * real.sqrt 2) ∨ (x, y) = (9 / 4, 5)) →
    (m * x^2 - n * y^2 = 1)) → 
    (y^2 / 16 - x^2 / 9 = 1)) :=
sorry

end hyperbola_standard_equation1_hyperbola_standard_equation2_l82_82432


namespace jan_score_l82_82623

variable (B J An Je : ℕ)
# check: noncomputable def points_beth : B := 12 { B }
# check: noncomputable def points_judy : J := 8 { J }
# check: noncomputable def points_angel : An := 11 { An }
# check: noncomputable def first_team_more_points : Je := 3 { Je} := by
noncomputable def points_jan :   ℕ := by
theorem jan_score
  (HB : B = 12)
  (HJ : J = 8)
  (HAn : An := 11)
  (HT : (B + J) = ((Je) + An + HJ) + HTe )
  logic J := 
  sorry
  10

  #6 5 ℕ := variables

#output Jan scored $\boxed{10}$ points successfully.

end jan_score_l82_82623


namespace simplify_and_evaluate_l82_82346

theorem simplify_and_evaluate (a : ℤ) (h : a = -2) :
  ( ((a + 7) / (a - 1) - 2 / (a + 1)) / ((a^2 + 3 * a) / (a^2 - 1)) = -1/2 ) :=
by
  sorry

end simplify_and_evaluate_l82_82346


namespace no_definite_conclusion_from_hypotheses_l82_82206

universe u

variables {β ζ η : Type u}
variables (Beta : β → Prop) (Zeta : ζ → β) (Yota : ζ → η)

noncomputable def hypotheses (Beta_not_Zeta : ∃ x, Beta x ∧ ¬ (∃ y, x = Zeta y))
  (All_Zeta_Yotas: ∀ y, ∃ z, Yota y = z) : Prop :=
  ¬(∃ x, Beta x ∧ ¬ (∃ y, Yota (Zeta y) = x)) ∧
  ¬(∀ x, Beta x → ∃ y, Yota (Zeta y) = x) ∧
  ¬(∀ x, Beta x → ¬ (∃ y, Yota (Zeta y) = x)) ∧
  ¬(∃ x, Beta x ∧ (∃ y, Yota (Zeta y) = x))

theorem no_definite_conclusion_from_hypotheses :
  hypotheses Beta (∃ x, Beta x ∧ ¬ (∃ y, x = Zeta y)) (∀ y, ∃ z, Yota y = z) :=
  sorry

end no_definite_conclusion_from_hypotheses_l82_82206


namespace option_C_is_quadratic_l82_82013

theorem option_C_is_quadratic : ∀ (x : ℝ), (x = x^2) ↔ (∃ (a b c : ℝ), a ≠ 0 ∧ a*x^2 + b*x + c = 0) := 
by
  sorry

end option_C_is_quadratic_l82_82013


namespace general_term_formula_l82_82174

variable (a : ℕ → ℤ) -- A sequence of integers 
variable (d : ℤ) -- The common difference 

-- Conditions provided
axiom h1 : a 1 = 6
axiom h2 : a 3 + a 5 = 0
axiom h_arithmetic : ∀ n, a (n + 1) = a n + d -- Arithmetic progression condition

-- The general term formula we need to prove
theorem general_term_formula : ∀ n, a n = 8 - 2 * n := 
by 
  sorry -- Proof goes here


end general_term_formula_l82_82174


namespace factory_profit_relationship_maximize_profit_maximum_profit_value_l82_82444

noncomputable def daily_profit (k : ℝ) (x : ℝ) (t : ℝ) : ℝ := 
  (100 * (Real.exp 30) * (x - 20 - t)) / (Real.exp x)

theorem factory_profit_relationship (x : ℝ) (t : ℝ)  
  (h1 : 25 ≤ x)
  (h2 : x ≤ 40) 
  (h3 : 2 ≤ t) 
  (h4 : t ≤ 5) : 
  daily_profit (100 * (Real.exp 30)) x t = (100 * (Real.exp 30) * (x - 20 - t)) / (Real.exp x) := 
by
  sorry

theorem maximize_profit (x : ℝ) (h1 : 25 ≤ x) (h2 : x ≤ 40) (h5 : x = 26): 
  maximize (daily_profit (100 * (Real.exp 30)) x 5) := 
by
  sorry

theorem maximum_profit_value :
  maximize (daily_profit (100 * (Real.exp 30)) 26 5) = 100 * (Real.exp 4) :=
by
  sorry

end factory_profit_relationship_maximize_profit_maximum_profit_value_l82_82444


namespace perimeter_of_circle_l82_82427

theorem perimeter_of_circle (p : ℝ) (π : ℝ) (hπ : π ≈ 3.14159) (h : p = 28) : 2 * π * (p / 4) ≈ 43.98 := 
by
  sorry

end perimeter_of_circle_l82_82427


namespace students_still_enrolled_l82_82047

  theorem students_still_enrolled 
    (initial : ℕ)
    (interested : ℕ)
    (dropped_within_day : ℕ)
    (frustrated_left : ℕ)
    (increase_factor : ℕ)
    (scheduling_conflict : ℕ)
    (last_rally_enrollment : ℕ)
    (half_dropped : ℕ)
    (half_graduated : ℕ) :
    let after_initial_interest := initial + interested in
    let after_initial_drop := after_initial_interest - dropped_within_day in
    let after_frustration_drop := after_initial_drop - frustrated_left in
    let increased_enrollment := after_frustration_drop * increase_factor in
    let post_rally := after_frustration_drop + increased_enrollment in
    let post_scheduling_conflict := post_rally - scheduling_conflict in
    let post_last_rally := post_scheduling_conflict + last_rally_enrollment in
    let after_half_dropped := post_last_rally / 2 in
    let after_half_graduated := after_half_dropped / 2 in
    after_half_graduated = 19 := 
  sorry

  -- Assign values according to the problem
  lemma students_enrollment : students_still_enrolled 8 8 (8 / 4) 2 5 2 6 1 1 := 
  by sorry
  
end students_still_enrolled_l82_82047


namespace regular_pentagon_construction_l82_82337

-- Definitions and conditions as in Lean 4
def Line (α : Type*) := set (set.point α × set.point α)
def Circle (α : Type*) := set.point α × ℝ
def Point (α : Type*) := set.point α

-- Given points, lines, and circles
variables {α : Type*} [metric_space α] [norm α]
variables (e : Line α) (P Q R S D F A B C E : Point α)
variables (r : ℝ)
variables (k1 k2 k3 k4 k5 k6 : Circle α)

-- Conditions
def cond1 : Circle α := (P, r)
def cond2 : Circle α := (Q, r)
def cond3 : Circle α := (Q, dist Q R)
def cond4 : Circle α := (P, dist Q R)
def cond5 : Circle α := (S, dist S D)
def cond6 : Circle α := (D, dist A B)

-- Constraints for the points being intersections
def cond_intersection1 := (k1 ∩ e = {Q, R})
def cond_intersection2 := (k1 ∩ k2 = {S})
def cond_intersection3 := (k3 ∩ k4 = {D, F})
def cond_intersection4 := (k5 ∩ e = {A, B})
def cond_intersection5 := (k6 ∩ k5 = {C, E})

-- Proof statement in Lean
theorem regular_pentagon_construction :
  (cond_intersection1) → 
  (cond_intersection2) → 
  (cond_intersection3) → 
  (cond_intersection4) → 
  (cond_intersection5) → 
  ∃ (FB FA : Line α), 
    (FB ∩ k5 = {C}) ∧ (FA ∩ k5 = {E}) :=
by {
  sorry
}

end regular_pentagon_construction_l82_82337


namespace similar_triangles_area_ratio_l82_82553

open Real

noncomputable def area_ratio (r : ℝ) : ℝ := r^2

theorem similar_triangles_area_ratio : 
  ∀ (ABC A'B'C' : Type) 
  (similar : ∀ (a b c : ABC) (a' b' c' : A'B'C'), similar_triangles a b c a' b' c'),
  let r := (1 : ℝ) / 3 
  in area_ratio r = 1 / 9 := 
by 
  intros ABC A'B'C' similar r 
  simp [r, area_ratio]
  sorry

end similar_triangles_area_ratio_l82_82553


namespace correct_number_of_statements_l82_82570

-- Define the conditions as invalidity of the given statements
def statement_1_invalid : Prop := ¬ (true) -- INPUT a,b,c should use commas
def statement_2_invalid : Prop := ¬ (true) -- INPUT x=, 3 correct format
def statement_3_invalid : Prop := ¬ (true) -- 3=B , left side should be a variable name
def statement_4_invalid : Prop := ¬ (true) -- A=B=2, continuous assignment not allowed

-- Combine conditions
def all_statements_invalid : Prop := statement_1_invalid ∧ statement_2_invalid ∧ statement_3_invalid ∧ statement_4_invalid

-- State the theorem to prove
theorem correct_number_of_statements : all_statements_invalid → 0 = 0 := 
by sorry

end correct_number_of_statements_l82_82570


namespace complement_of_M_l82_82207

def U : set ℤ := {-2, -1, 0, 1, 2, 3, 4, 5, 6}
def M : set ℤ := {x | x > -1 ∧ x < 4}

theorem complement_of_M :
  U \ M = {-2, -1, 4, 5, 6} :=
by
  simp [U, M]
  sorry

end complement_of_M_l82_82207


namespace length_of_intersection_segment_l82_82259

-- Define the polar coordinates conditions
def curve_1 (ρ θ : ℝ) : Prop := ρ = 4 * Real.sin θ
def curve_2 (ρ θ : ℝ) : Prop := ρ * Real.cos θ = 1

-- Convert polar equations to Cartesian coordinates
def curve_1_cartesian (x y : ℝ) : Prop := x^2 + y^2 = 4 * y
def curve_2_cartesian (x y : ℝ) : Prop := x = 1

-- Define the intersection points and the segment length function
def segment_length (y1 y2 : ℝ) : ℝ := abs (y1 - y2)

-- The statement to prove
theorem length_of_intersection_segment :
  (curve_1_cartesian 1 (2 + Real.sqrt 3)) ∧ (curve_1_cartesian 1 (2 - Real.sqrt 3)) →
  (curve_2_cartesian 1 (2 + Real.sqrt 3)) ∧ (curve_2_cartesian 1 (2 - Real.sqrt 3)) →
  segment_length (2 + Real.sqrt 3) (2 - Real.sqrt 3) = 2 * Real.sqrt 3 := by
  sorry

end length_of_intersection_segment_l82_82259


namespace common_ratio_range_l82_82171

theorem common_ratio_range (q : ℝ) 
  (h1 : ∀ n, a_n = 8 * q^(n - 1))
  (h2 : ∀ n, b_n = Real.log2 (a_n))
  (h3 : ∀ n, S_n = ∑ i in Finset.range n, b_n i)
  (h4 : S_3 > S_4) : 
  2^(-3/2) < q ∧ q < 2^(-1) :=
by 
  sorry

end common_ratio_range_l82_82171


namespace gcd_a_b_l82_82763

def a := 130^2 + 250^2 + 360^2
def b := 129^2 + 249^2 + 361^2

theorem gcd_a_b : Int.gcd a b = 1 := 
by
  sorry

end gcd_a_b_l82_82763


namespace find_sides_of_triangle_l82_82264

theorem find_sides_of_triangle (
  {a b c : ℝ}
  (angle_C : ℝ) (area : ℝ)
  (h1 : a = 3)
  (h2 : angle_C = 2 * π / 3)
  (h3 : area = 3 * sqrt 3 / 4)
) : b = 1 ∧ c = sqrt(13) :=
by
  sorry

end find_sides_of_triangle_l82_82264


namespace euler_totient_divisibility_l82_82283

def euler_totient (n : ℕ) : ℕ := sorry

theorem euler_totient_divisibility (n : ℕ) (hn : 0 < n) : 2^(n * (n + 1)) ∣ 32 * euler_totient (2^(2^n) - 1) := 
sorry

end euler_totient_divisibility_l82_82283


namespace arithmetic_seq_difference_median_l82_82843

theorem arithmetic_seq_difference_median :
  ∀ (a : ℕ → ℝ) (n : ℕ), 
    (150 = n) →
    (∀ i, 20 ≤ a i ∧ a i ≤ 120) →
    (∑ i in finset.range n, a i = 12000) →
    (let M := (80 : ℝ) + 75 * (40 / 149 : ℝ)
     let m := (80 : ℝ) - 75 * (40 / 149 : ℝ)
     in M - m = 6000 / 149) :=
begin
  intros,
  have hA : (∑ i in finset.range n, a i) / n = 80,
  {
    -- Because the average of the sequence terms is 80
    rw [finset.sum_div, mul_comm, ←div_eq_iff (show (n : ℝ) ≠ 0, by norm_num : 150 ≠ 0)],
    exacts [A_2],
  },
  let d := (40 / 149 : ℝ),
  have hM : M = 80 + 75 * d,
  have hm : m = 80 - 75 * d,
  use sorry,
end

end arithmetic_seq_difference_median_l82_82843


namespace liars_count_correct_l82_82390

-- Declare the definition representing the count of liars based on given conditions
def count_of_liars (n : ℕ) : ℕ := if n = 2017 then 1344 else 0

-- Theorem stating that under specified conditions, the number of liars is 1344
theorem liars_count_correct {n : ℕ} (h₁ : n = 2017)
  (h₂ : ∀ i : ℕ, i < n → (islander i).says "my neighbors are from the same tribe")
  (h₃ : ∀ i : ℕ, (islander i).tribe = knight ↔ (isNeighbor i).tribe = liar)
  (h₄ : ∃ i₁ i₂ : ℕ, i₁ < n ∧ i₂ < n ∧ (islander i₁).tribe = liar ∧ (islander i₂).tribe = liar
    ∧ (islander i₁).says "my neighbors are from the same tribe" ∧ (islander i₂).says "my neighbors are from the same tribe") :
  count_of_liars n = 1344 :=
by 
  -- Proof would go here
  sorry

-- Definitions for tribes, islanders, and what they say
inductive Tribe | knight | liar

structure Islander :=
  (tribe : Tribe)
  (says : String → Prop)

-- Function indicating neighbor
def isNeighbor (i : ℕ) : ℕ := (i + 1) % 2017 -- Circular table

-- Example instance for conceptual representation, not part of theorem but helps visualization
axiom islander : ℕ → Islander

end liars_count_correct_l82_82390


namespace find_floor_abs_sum_eq_51_l82_82859

noncomputable def seq (x : ℕ → ℝ) (S: ℝ) (a : ℕ) : Prop := x a + a = S + 101

theorem find_floor_abs_sum_eq_51 (x : ℕ → ℝ) (S: ℝ):
  (∀ a, 1 ≤ a → a ≤ 100 → seq x S a) →
  (S = -5050 / 99) →
  (| S |.floor = 51) := by
  intro h1 h2
  rw [Real.abs_eq, Int.floor_eq, h2]
  norm_num
  sorry

end find_floor_abs_sum_eq_51_l82_82859


namespace divisible_by_square_of_k_l82_82299

theorem divisible_by_square_of_k (a b l : ℕ) (k : ℕ) (h1 : a > 1) (h2 : b > 1) (h3 : a % 2 = 1) (h4 : b % 2 = 1) (h5 : a + b = 2 ^ l) : k = 1 ↔ k^2 ∣ a^k + b^k := 
sorry

end divisible_by_square_of_k_l82_82299


namespace noah_energy_usage_correct_l82_82677

def bedroom_power : ℕ := 6
def office_power : ℕ := 3 * bedroom_power
def living_room_power : ℕ := 4 * bedroom_power
def kitchen_power : ℕ := 2 * bedroom_power
def bathroom_power : ℕ := 5 * bedroom_power

def bedroom_hours : ℕ := 2
def office_hours : ℕ := 3
def living_room_hours : ℕ := 4
def kitchen_hours : ℕ := 1
def bathroom_hours : ℝ := 1.5

def total_energy_usage : ℕ :=
  bedroom_power * bedroom_hours +
  office_power * office_hours +
  living_room_power * living_room_hours +
  kitchen_power * kitchen_hours +
  nat_ceil(bathroom_power * bathroom_hours)

theorem noah_energy_usage_correct :
  total_energy_usage = 219 := by
  sorry

end noah_energy_usage_correct_l82_82677


namespace perimeter_of_polygon_ABCDE_l82_82983

structure Point :=
(x : ℝ)
(y : ℝ)

def dist (p1 p2 : Point) : ℝ :=
  real.sqrt ((p2.x - p1.x) ^ 2 + (p2.y - p1.y) ^ 2)

noncomputable def perimeter_of_ABCDE : ℝ :=
  let A := Point.mk 0 8
  let B := Point.mk 4 8
  let C := Point.mk 4 4
  let D := Point.mk 9 0
  let E := Point.mk 0 0
  dist A B + dist B C + dist C D + dist D E + dist E A

theorem perimeter_of_polygon_ABCDE : perimeter_of_ABCDE = 25 + real.sqrt 41 :=
sorry

end perimeter_of_polygon_ABCDE_l82_82983


namespace sum_first_60_terms_l82_82736

def a (n : ℕ) : ℤ := sorry

def sequence_condition (n : ℕ) : Prop :=
  a (n + 1) + (-1) ^ n * a n = 3 * n - 1

theorem sum_first_60_terms :
  (∑ i in Finset.range 60, a i) = 780 :=
by
  sorry

end sum_first_60_terms_l82_82736


namespace prism_volume_l82_82119

def volume_of_oblique_triangular_prism (S d : ℝ) : ℝ :=
1 / 2 * d * S

theorem prism_volume (S d : ℝ) :
  volume_of_oblique_triangular_prism S d = (1 / 2) * d * S := by
sorry

end prism_volume_l82_82119


namespace solve_for_a_l82_82868

variable (a : ℝ)

-- Define the ellipse and hyperbola equations
def ellipse (x y : ℝ) := x^2 / 4 + y^2 / a^2 = 1
def hyperbola (x y : ℝ) := y^2 / 2 - x^2 / a = 1

-- Define the focal distances of the ellipse and hyperbola
def c_ellipse := Real.sqrt (a^2 - 2)
def c_hyperbola := Real.sqrt (1 + a)

-- Define the condition that the ellipse and hyperbola have the same foci
def same_foci := c_ellipse = c_hyperbola

-- The main theorem statement
theorem solve_for_a (h : same_foci a) : a = (1 + Real.sqrt 13) / 2 :=
sorry

end solve_for_a_l82_82868


namespace max_value_trig_expression_l82_82520

noncomputable def max_trig_expression : ℝ :=
  \left( \cos^2 \theta_1 \sin^2 \theta_2 + \cos^2 \theta_2 \sin^2 \theta_3 +
        \cos^2 \theta_3 \sin^2 \theta_4 + \cos^2 \theta_4 \sin^2 \theta_5 +
        \cos^2 \theta_5 \sin^2 \theta_1 \right)

theorem max_value_trig_expression :
  ∃ (θ1 θ2 θ3 θ4 θ5 : ℝ), max_trig_expression θ1 θ2 θ3 θ4 θ5 = 25 / 32 := sorry

end max_value_trig_expression_l82_82520


namespace range_of_theta_l82_82941

theorem range_of_theta (a b c : Line) (θ : ℝ) :
  skew a b ∧ angle a b = 40 ∧ skew c a ∧ skew c b ∧ angle c a = θ ∧ angle c b = θ 
  → (∃ (c1 c2 c3 c4 : Line), c1 ≠ c2 ∧ c2 ≠ c3 ∧ c3 ≠ c4 ∧ c1 ≠ c3 ∧ c1 ≠ c4 ∧ c2 ≠ c4 
     ∧ skew c1 a ∧ skew c1 b ∧ angle c1 a = θ ∧ angle c1 b = θ
     ∧ skew c2 a ∧ skew c2 b ∧ angle c2 a = θ ∧ angle c2 b = θ
     ∧ skew c3 a ∧ skew c3 b ∧ angle c3 a = θ ∧ angle c3 b = θ
     ∧ skew c4 a ∧ skew c4 b ∧ angle c4 a = θ ∧ angle c4 b = θ)
  → (70 < θ ∧ θ < 90) := by
  sorry

end range_of_theta_l82_82941


namespace cos_alpha_minus_beta_cos_alpha_cos_beta_l82_82534

variables (α β : ℝ)
variables (h_acute_α : 0 < α ∧ α < π / 2) (h_acute_beta : 0 < β ∧ β < π / 2)
variables (h_dist : (cos α - cos β) ^ 2 + (sin α - sin β) ^ 2 = (sqrt 10 / 5) ^ 2)
variables (h_tan_half_alpha : tan (α / 2) = 1 / 2)

theorem cos_alpha_minus_beta :
  cos (α - β) = 4 / 5 :=
sorry

theorem cos_alpha_cos_beta :
  (cos α = 3 / 5) ∧ (cos β = 24 / 25) :=
sorry

end cos_alpha_minus_beta_cos_alpha_cos_beta_l82_82534


namespace no_valid_n_l82_82732

theorem no_valid_n (n : ℤ) :
  let h : ℝ → ℝ := λ x, x^3 - 2 * x^2 - (n^2 + 2 * n) * x + 3 * n^2 + 6 * n + 3
  ¬ (∀ (z : ℤ), h z = 0 ∧ h 2 = 0) :=
by
  sorry

end no_valid_n_l82_82732


namespace count_integers_congruent_mod_l82_82215

theorem count_integers_congruent_mod (n : ℕ) (h₁ : n < 1200) (h₂ : n ≡ 3 [MOD 7]) : 
  ∃ (m : ℕ), (m = 171) :=
by
  sorry

end count_integers_congruent_mod_l82_82215


namespace sculpture_cost_in_chinese_yuan_l82_82325

theorem sculpture_cost_in_chinese_yuan
  (usd_to_nad : ℝ)
  (usd_to_cny : ℝ)
  (cost_nad : ℝ)
  (h1 : usd_to_nad = 8)
  (h2 : usd_to_cny = 5)
  (h3 : cost_nad = 160) :
  (cost_nad / usd_to_nad) * usd_to_cny = 100 :=
by
  sorry

end sculpture_cost_in_chinese_yuan_l82_82325


namespace a_eq_b_l82_82203

variables {a b c : ℝ} {α : ℝ}

-- The line equation is considered given by its coefficients a, b, and c
def line_eq (a b c : ℝ) (x y : ℝ) : Prop := a * x + b * y + c = 0

-- Given that the inclination angle α of the line satisfies sin α + cos α = 0
axiom sin_cos_eq_zero (α : ℝ) : sin α + cos α = 0

-- Prove that a and b satisfy a - b = 0
theorem a_eq_b (a b c α : ℝ) (h1 : sin_cos_eq_zero α) (h2 : ∀ x y, line_eq a b c x y) :
  a - b = 0 :=
sorry

end a_eq_b_l82_82203


namespace triangle_side_length_inequality_l82_82473

noncomputable def s (a b c : ℝ) : ℝ := real.sqrt a + real.sqrt b + real.sqrt c

noncomputable def t (a b c : ℝ) : ℝ := (1 / a) + (1 / b) + (1 / c)

theorem triangle_side_length_inequality (a b c : ℝ) (h_area : (1 / 2) * a * b * real.sin c = 1 / 4) 
(h_circumradius : 1 = 1)
(h_prod : a * b * c = 1) : 
s a b c < t a b c :=
sorry

end triangle_side_length_inequality_l82_82473


namespace find_bankers_gain_l82_82369

def bankers_gain (P TD : ℝ) : ℝ := (TD^2) / P

theorem find_bankers_gain (P TD : ℝ) (hP : P = 576) (hTD : TD = 96) :
  bankers_gain P TD = 16 :=
by {
  rw [bankers_gain, hP, hTD],
  norm_num,
}

end find_bankers_gain_l82_82369


namespace sin_alpha_l82_82218

variable (α : Real)
variable (hcos : Real.cos α = 3 / 5)
variable (htan : Real.tan α < 0)

theorem sin_alpha (α : Real) (hcos : Real.cos α = 3 / 5) (htan : Real.tan α < 0) :
  Real.sin α = -4 / 5 :=
sorry

end sin_alpha_l82_82218


namespace field_breadth_l82_82043

theorem field_breadth : 
  ∀ (b : ℝ), 
    let field_length := 90 in
    let tank_length := 25 in
    let tank_breadth := 20 in
    let tank_depth := 4 in
    let level_rise := 0.5 in
    (tank_length * tank_breadth * tank_depth) = 2000 →
    (field_length * b - tank_length * tank_breadth) = (2000 / level_rise) → 
    b = 50 := 
by
  intros b field_length tank_length tank_breadth tank_depth level_rise h1 h2
  have h1: tank_length * tank_breadth * tank_depth = 2000 := by sorry
  have h2: field_length * b - tank_length * tank_breadth = (2000 / level_rise) := by sorry
  have h3: 2000 = (90 * b - 25 * 20) * 0.5 := by sorry
  have h4: 4000 = 90 * b - 500 := by sorry
  have h5: 90 * b = 4500 := by sorry
  show b = 50 from eq_of_mul_eq_mul_left (ne_of_gt (by norm_num : (0 : ℝ) < 90)) h5 (by norm_num)
  sorry

end field_breadth_l82_82043


namespace common_area_of_rectangle_and_circle_eqn_l82_82074

theorem common_area_of_rectangle_and_circle_eqn :
  let rect_length := 8
  let rect_width := 4
  let circle_radius := 3
  let common_area := (3^2 * 2 * Real.pi / 4) - 2 * Real.sqrt 5  
  common_area = (9 * Real.pi / 2) - 2 * Real.sqrt 5 := 
sorry

end common_area_of_rectangle_and_circle_eqn_l82_82074


namespace smallest_abundant_composite_less_than_20_l82_82864

def is_abundant (n : ℕ) : Prop :=
  ∑ d in (finset.filter (λ x, x ∣ n ∧ x < n) (finset.range n)), d > n

theorem smallest_abundant_composite_less_than_20 :
  (∀ n, n < 20 → nat.prime n → ¬is_abundant n) ∧
  (is_abundant 12) ∧ (∃ n, n ≠ 12 ∧ n < 20 ∧ is_abundant n ∧ n > 12) :=
by {
  -- Proof to be done
  sorry
}

end smallest_abundant_composite_less_than_20_l82_82864


namespace travel_time_l82_82091

theorem travel_time (d1 d2 : ℝ) (s1 s2 : ℝ) (h1 : d1 = 180) (h2 : d2 = 280) (h3 : s1 = 60) (h4 : s2 = 80) : 
  d1 / s1 + d2 / s2 = 6.5 := 
by 
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end travel_time_l82_82091


namespace find_a1_of_geometric_sequence_l82_82179

noncomputable def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

theorem find_a1_of_geometric_sequence (a : ℕ → ℝ) (q : ℝ) 
  (hq : 0 < q) (h_geom : geometric_sequence a q)
  (h2 : a 2 = 2)
  (h_condition : a 3 * a 9 = 2 * (a 5) ^ 2) : 
  a 1 = sqrt 2 :=
by
  -- Proof is skipped for brevity
  sorry

end find_a1_of_geometric_sequence_l82_82179


namespace relation_between_m_n_k_ell_l82_82609

theorem relation_between_m_n_k_ell
  (m n k ℓ : ℕ)
  (student_teacher : ℕ → ℕ → Prop)  -- student_teacher i j means student i is taught by teacher j
  (H_teacher_exactly_k_students : ∀ t : ℕ, ∃ s : finset ℕ, s.card = k ∧ (∀ i ∈ s, student_teacher i t))
  (H_pair_students_ell_teachers : ∀ i j : ℕ, i ≠ j → finset.card (finset.filter (λ t, student_teacher i t ∧ student_teacher j t) (finset.range m)) = ℓ) :
  n * (n - 1) * ℓ = m * k * (k - 1) :=
by
  sorry

end relation_between_m_n_k_ell_l82_82609


namespace cartesian_eq_of_polar_eq_range_of_m_for_intersection_l82_82619

-- Define the curve C with parametric equations
def C (t : ℝ) : ℝ × ℝ := (√3 * Real.cos (2 * t), 2 * Real.sin t)

-- Define the line l in polar coordinates
def polar_line (rho theta : ℝ) (m : ℝ) : Prop :=
  rho * Real.sin (theta + π / 3) + m = 0

-- Define the line l in Cartesian coordinates
def cartesian_line (x y m : ℝ) : Prop :=
  √3 * x + y + 2 * m = 0

-- Proof statement 1: The Cartesian equation of the line l
theorem cartesian_eq_of_polar_eq (rho theta m : ℝ) :
  (polar_line rho theta m) -> ∀ x y, (x = rho * Real.cos theta) ∧ (y = rho * Real.sin theta) -> cartesian_line x y m :=
by
  -- Exact proof steps are left as an exercise
  sorry

-- Proof statement 2: Range of values for m for intersections
theorem range_of_m_for_intersection (m : ℝ) :
  (∃ t : ℝ, C t ∈ {p : ℝ × ℝ | cartesian_line p.1 p.2 m}) → (-19/12 ≤ m ∧ m ≤ 5/2) :=
by
  -- Exact proof steps are left as an exercise
  sorry

end cartesian_eq_of_polar_eq_range_of_m_for_intersection_l82_82619


namespace valid_a_values_l82_82530

theorem valid_a_values (a : ℝ) :
  (∀ x : ℝ, log (3 : ℝ) (2 * x^2 - x + 2 * a - 4 * a^2) + log (1 / 3 : ℝ) (x^2 + a * x - 2 * a^2) = 0 → x ≠ a + 1 ∧ x ≠ -a + 1) ∧
  (∀ x1 x2 : ℝ, x1 ≠ x2 → x1^2 + x2^2 < 1) ↔ 
  a ∈ Ioo 0 (1 / 3) ∪ Ioo (1 / 3) (2 / 5) :=
sorry

end valid_a_values_l82_82530


namespace factorial_trailing_zeros_l82_82948

theorem factorial_trailing_zeros (n : ℕ) : 
  (n = 50) → 
  (∑ k in (Finset.range (n // 5 + 1)), n // (5 ^ k)) = 12 := 
by
  intros h1
  sorry

end factorial_trailing_zeros_l82_82948


namespace inequality1_inequality2_inequality3_inequality4_inequality5_inequality6_inequality7_inequality8_inequality9_inequality10_l82_82661

variable {x y z : ℝ}

-- Condition: x, y, z ∈ ℝ_+
axiom x_pos : x > 0
axiom y_pos : y > 0
axiom z_pos : z > 0

noncomputable def p : ℝ := x + y + z
noncomputable def q : ℝ := x * y + y * z + z * x
noncomputable def r : ℝ := x * y * z

-- Question: Prove p² ≥ 3q
theorem inequality1 : p^2 ≥ 3 * q := sorry

-- Question: Prove p³ ≥ 27r
theorem inequality2 : p^3 ≥ 27 * r := sorry

-- Question: Prove pq ≥ 9r
theorem inequality3 : p * q ≥ 9 * r := sorry

-- Question: Prove q² ≥ 3pr
theorem inequality4 : q ^ 2 ≥ 3 * p * r := sorry

-- Question: Prove p²q + 3pr ≥ 4q²
theorem inequality5 : p^2 * q + 3 * p * r ≥ 4 * q^2 := sorry

-- Question: Prove p³ + 9r ≥ 4pq
theorem inequality6 : p^3 + 9 * r ≥ 4 * p * q := sorry

-- Question: Prove pq² ≥ 2p²r + 3qr
theorem inequality7 : p * q ^ 2 ≥ 2 * p^2 * r + 3 * q * r := sorry

-- Question: Prove pq² + 3qr ≥ 4p²r
theorem inequality8 : p * q ^ 2 + 3 * q * r ≥ 4 * p^2 * r := sorry

-- Question: Prove 2q³ + 9r² ≥ 7pqr
theorem inequality9 : 2 * q ^ 3 + 9 * r ^ 2 ≥ 7 * p * q * r := sorry

-- Question: Prove p⁴ + 4q² + 6pr ≥ 5p²q
theorem inequality10 : p^4 + 4 * q^2 + 6 * p * r ≥ 5 * p^2 * q := sorry

end inequality1_inequality2_inequality3_inequality4_inequality5_inequality6_inequality7_inequality8_inequality9_inequality10_l82_82661


namespace JEI_angle_value_l82_82985

variables (G B F H I J E : Type)

-- Conditions
def angle_GBF : ℝ := 20
def angle_GHI : ℝ := 130

-- Proof goal
theorem JEI_angle_value :
  angle_GBF + angle_GHI = 150 →
  ∀ B E H: Type, ∃ (φ: ℝ), 
  φ + angle_GBF + angle_GHI = 180 → φ = 30 :=
begin
  intros h_sum_angle_sum B E H,
  use 30,
  intros h_phi,
  linarith
end

end JEI_angle_value_l82_82985


namespace find_original_solution_percentage_l82_82821

noncomputable def original_solution_percentage (P : ℝ) : Prop :=
  let replaced_fraction := 0.5
  let replacement_percentage := 20.0
  let resultant_percentage := 50.0 in
  replaced_fraction * P + replaced_fraction * replacement_percentage = 1 * resultant_percentage

theorem find_original_solution_percentage : original_solution_percentage 80 :=
by
  intros
  sorry

end find_original_solution_percentage_l82_82821


namespace find_initial_students_l82_82974

-- Let S be the number of students at the start of the year.
variable (S : ℕ)

-- Condition 1: During the year, 3 students left.
axiom students_left : ℕ := 3

-- Condition 2: During the year, 42 new students came.
axiom students_came : ℕ := 42

-- Condition 3: There were 43 students in fourth grade at the end of the year.
axiom students_end : ℕ := 43

-- Net increase in number of students is given by:
def net_increase : ℕ := students_came - students_left

-- Final equation representing the total number of students at the end:
def final_equation : Prop := S + net_increase = students_end

-- We need to prove that S = 4 given the conditions.
theorem find_initial_students (h : final_equation) : S = 4 :=
sorry

end find_initial_students_l82_82974


namespace point_in_convex_ngon_l82_82025

noncomputable def lies_inside_convex_ngon (n : ℕ) (vertices : Fin n → ℂ) (z : ℂ) :=
(convex_hull (set.range vertices)).indicator 1 z = 1

theorem point_in_convex_ngon (n : ℕ) (vertices : Fin n → ℂ) (z : ℂ) 
  (h_convex: convex_hull (set.range vertices).indicator 1 z = 1)
  (h_eq: ∑ i : Fin n, (1 / (z - vertices i)) = 0) :
  lies_inside_convex_ngon n vertices z :=
by 
  sorry

end point_in_convex_ngon_l82_82025


namespace trapezoid_area_l82_82762

theorem trapezoid_area (base1 base2 height : ℕ) (h_base1 : base1 = 9) (h_base2 : base2 = 11) (h_height : height = 3) :
  (1 / 2 : ℚ) * (base1 + base2 : ℕ) * height = 30 :=
by
  sorry

end trapezoid_area_l82_82762


namespace time_left_for_nap_l82_82848

noncomputable def total_time : ℝ := 20
noncomputable def first_train_time : ℝ := 2 + 1
noncomputable def second_train_time : ℝ := 3 + 1
noncomputable def transfer_one_time : ℝ := 0.75 + 0.5
noncomputable def third_train_time : ℝ := 2 + 1
noncomputable def transfer_two_time : ℝ := 1
noncomputable def fourth_train_time : ℝ := 1
noncomputable def transfer_three_time : ℝ := 0.5
noncomputable def fifth_train_time_before_nap : ℝ := 1.5

noncomputable def total_activities_time : ℝ :=
  first_train_time +
  second_train_time +
  transfer_one_time +
  third_train_time +
  transfer_two_time +
  fourth_train_time +
  transfer_three_time +
  fifth_train_time_before_nap

theorem time_left_for_nap : total_time - total_activities_time = 4.75 := by
  sorry

end time_left_for_nap_l82_82848


namespace find_vector_v_l82_82288

def vector3 := ℝ × ℝ × ℝ

def cross_product (u v : vector3) : vector3 :=
  (u.2.1 * v.2.2 - u.2.2 * v.2.1,
   u.2.2 * v.1  - u.1   * v.2.2,
   u.1   * v.2.1 - u.2.1 * v.1)

def a : vector3 := (1, 2, 1)
def b : vector3 := (2, 0, -1)
def v : vector3 := (3, 2, 0)
def b_cross_a : vector3 := (2, 3, 4)
def a_cross_b : vector3 := (-2, 3, -4)

theorem find_vector_v :
  cross_product v a = b_cross_a ∧ cross_product v b = a_cross_b :=
sorry

end find_vector_v_l82_82288


namespace find_x_l82_82870

theorem find_x (x : ℝ) (h : (x * (x ^ 4) ^ (1/2)) ^ (1/4) = 2) : 
  x = 16 ^ (1/3) :=
sorry

end find_x_l82_82870


namespace problem_1_problem_2_l82_82196

noncomputable def f (a x : ℝ) := log (a * x + 1) + (x^3)/3 - x^2 - a * x
noncomputable def g (a x : ℝ) := log (x^2 * (a * x + 1)) + (x^3)/3 - 3 * a * x - f a x
noncomputable def φ (c b x : ℝ) := log x - c * x^2 - b * x
noncomputable def φ' (c b x : ℝ) := 1/x - 2 * c * x - b

theorem problem_1 (a : ℝ) (h1 : 0 < a) (h2 : a ≤ 4 + 3 * sqrt 2) :
  ∀ x ∈ set.Ici (4 : ℝ), 0 < f' a x :=
sorry

theorem problem_2 (a c b x1 x2 : ℝ)
  (h3 : a ≥ 3 * sqrt 2 / 2)
  (h4 : φ c b x1 = 0)
  (h5 : φ c b x2 = 0)
  (h6 : x1 < x2) :
  let y := (x1 - x2) * φ' c b ((x1 + x2) / 2) in y ≥ ln 2 - 2/3 :=
sorry

end problem_1_problem_2_l82_82196


namespace sum_sequence_l82_82546

theorem sum_sequence :
  ∀ {a : ℕ → ℕ}, (a 5 = 5 ∧ (∑ i in range 5, a i) / 5 = 3) →
  (∑ i in range 2016, (1 / (a i * a (i + 1))) : ℚ) = 2016 / 2017 :=
begin
  intros a h,
  sorry -- Proof omitted
end

end sum_sequence_l82_82546


namespace maximum_value_f_range_of_a_unique_solution_m_l82_82197

-- Define the function f(x)
def f (x : ℝ) (a : ℝ) : ℝ := Real.log x - a * x^2 - (1/2) * x

-- I. Prove that the maximum value of f(x) when a = 1/4 is -3/4
theorem maximum_value_f :
  ∃ x : ℝ, ∀ a : ℝ, a = 1/4 → f x 1/4 = -3/4 := sorry

-- II. Define the function g(x) and prove the range of values for a
def g (x : ℝ) (a : ℝ) : ℝ := f x a + a * x^2 + (1/2) * x + a / x

theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Ioc 0 3, (x - a) / x^2 ≤ 1/2) → a ≥ 1/2 := sorry

-- III. Prove that when a = 0, the equation 2m f(x) = x(x - 3m) has a unique solution for m = 1/2
theorem unique_solution_m :
  ∃ (m : ℝ), ∀ (a : ℝ), a = 0 → (∃! x, 2 * m * f x a = x * (x - 3 * m)) → m = 1/2 := sorry

end maximum_value_f_range_of_a_unique_solution_m_l82_82197


namespace books_in_series_l82_82384

theorem books_in_series (books_watched : ℕ) (movies_watched : ℕ) (read_more_movies_than_books : books_watched + 3 = movies_watched) (watched_movies : movies_watched = 19) : books_watched = 16 :=
by sorry

end books_in_series_l82_82384


namespace payment_on_thursday_l82_82671

noncomputable def total_credit : ℕ := 100
noncomputable def total_spent : ℕ := 100
noncomputable def payment_tuesday : ℕ := 15
noncomputable def remaining_payment : ℕ := 62

theorem payment_on_thursday :
  let payment_thursday := total_spent - payment_tuesday - remaining_payment in
  payment_thursday = 23 :=
by
  let payment_thursday := total_spent - payment_tuesday - remaining_payment
  show payment_thursday = 23 
  sorry

end payment_on_thursday_l82_82671


namespace incorrect_simplification_l82_82417

theorem incorrect_simplification : 
  (∀ x : Int, +(-x) = -x) ∧ (∀ x : Int, -(-x) = x) ∧ (|(-3 : Int)| ≠ -3) ∧ (∀ x : Int, -|x| = -x) :=
by
  sorry

end incorrect_simplification_l82_82417


namespace area_of_R_l82_82686

theorem area_of_R (AB BC : ℝ) (angle_B : ℝ) (h_AB : AB = 2) (h_BC : BC = 4) (h_angle_B : angle_B = 120) :
  let R_area := sqrt 3 in
  R_area = sqrt 3 :=
by {
  have hR : R_area = sqrt 3,
  { sorry },
  exact hR,
}

end area_of_R_l82_82686


namespace marbles_distribution_l82_82595

theorem marbles_distribution (marbles children : ℕ) (h1 : marbles = 60) (h2 : children = 7) :
  ∃ k, k = 3 → (∀ i < children, marbles / children + (if i < marbles % children then 1 else 0) < 9) → k = 3 :=
by
  sorry

end marbles_distribution_l82_82595


namespace problem1_problem2_l82_82787

theorem problem1 : 3 / Real.sqrt 3 + (Real.pi + Real.sqrt 3)^0 + abs (Real.sqrt 3 - 2) = 3 := 
by
  sorry

theorem problem2 : (3 * Real.sqrt 12 - 2 * Real.sqrt (1 / 3) + Real.sqrt 48) / Real.sqrt 3 = 28 / 3 :=
by
  sorry

end problem1_problem2_l82_82787


namespace binkie_gemstones_l82_82110

variables (F B S : ℕ)

theorem binkie_gemstones :
  (B = 4 * F) →
  (S = (1 / 2 : ℝ) * F - 2) →
  (S = 1) →
  B = 24 :=
by
  sorry

end binkie_gemstones_l82_82110


namespace range_of_t_l82_82237

variable (A B C : Type) [IsTriangle A B C]

noncomputable def t := dist B C

theorem range_of_t (angle_ABC : ℝ) (AC_length : ℝ) (t : ℝ) (unique_triangle : ∃! (△ : Triangle A B C), angle_ABC = π / 4 ∧ AC_length = 1 ∧ dist B C = t) : 
  t ∈ Icc 0 1 ∪ {Real.sqrt 2} := 
by
  have h_angle : angle B A C = π / 4 := unique_triangle.2.1
  have h_AC : dist A C = 1 := unique_triangle.2.2.1
  have h_t : dist B C = t := unique_triangle.2.2.2
  sorry

end range_of_t_l82_82237


namespace area_PQRS_l82_82976

noncomputable def area_quadrilateral (a b c d θ : ℝ) : ℝ :=
  let θ_rad := θ * Real.pi / 180
  (1/2) * a * b * Real.sin θ_rad + (1/2) * c * d * Real.sin θ_rad

theorem area_PQRS :
  let θ := 100
  let PQ := 5
  let QR := 6
  let RS := 4
  area_quadrilateral PQ QR QR RS θ = 26.5896 :=
by
  -- Definition of theta in radians
  let θ_rad := θ * Real.pi / 180 
  -- Evaluate the area expression
  let area := (1/2) * PQ * QR * Real.sin θ_rad + (1/2) * QR * RS * Real.sin θ_rad 
  -- Ensure the computed area is correct
  exact sorry

end area_PQRS_l82_82976


namespace amount_paid_with_l82_82678

-- Define the conditions as constants
def ticket_cost : ℝ := 5.92
def num_tickets : ℕ := 2
def borrowed_movie_cost : ℝ := 6.79
def change_received : ℝ := 1.37

-- Define the question as a theorem
theorem amount_paid_with :
  let total_cost := (num_tickets * ticket_cost) + borrowed_movie_cost in
  let amount_paid := total_cost + change_received in
  amount_paid = 20.00 :=
by
  sorry

end amount_paid_with_l82_82678


namespace arithmetic_sequence_a1_value_l82_82254

theorem arithmetic_sequence_a1_value (a : ℕ → ℤ) (d : ℤ) 
  (h1 : a 3 = -6) 
  (h2 : a 7 = a 5 + 4) 
  (h_seq : ∀ n, a (n+1) = a n + d) : 
  a 1 = -10 := 
by
  sorry

end arithmetic_sequence_a1_value_l82_82254


namespace smallest_solution_eq_l82_82147

noncomputable def fractional_part (x : ℝ) : ℝ := x - floor x

theorem smallest_solution_eq (x : ℝ) (h : floor x = 2 + 50 * fractional_part x) (hx : 0 ≤ fractional_part x ∧ fractional_part x < 1) : x = 2 :=
by
  sorry

end smallest_solution_eq_l82_82147


namespace height_of_building_l82_82405

theorem height_of_building 
  (shadow_building : ℝ) (shadow_lamp : ℝ) (height_lamp : ℝ) (height_building : ℝ) 
  (ratio_shadows : shadow_building / shadow_lamp = 3 / 2) 
  (ratio_height : height_building / height_lamp = shadow_building / shadow_lamp) :
  Int.round height_building = 38 :=
by
  have h_eq : height_building = (25 * 3) / 2 := by
    rw [← ratio_height, ratio_shadows]
    ring
  rw [h_eq]
  norm_num
  sorry

end height_of_building_l82_82405


namespace factor_expression_l82_82883

variable (y : ℝ)

theorem factor_expression :
  (16 * y^6 + 36 * y^4 - 9) - (4 * y^6 - 6 * y^4 + 9) = 6 * (2 * y^6 + 7 * y^4 - 3) := 
by
  sorry


end factor_expression_l82_82883


namespace smallest_b_for_q_ge_half_l82_82442

open Nat

def binomial (n k : ℕ) : ℕ := if h : k ≤ n then n.choose k else 0

def q (b : ℕ) : ℚ := (binomial (32 - b) 2 + binomial (b - 1) 2) / (binomial 38 2 : ℕ)

theorem smallest_b_for_q_ge_half : ∃ (b : ℕ), b = 18 ∧ q b ≥ 1 / 2 :=
by
  -- Prove and find the smallest b such that q(b) ≥ 1/2
  sorry

end smallest_b_for_q_ge_half_l82_82442


namespace number_of_valid_sequences_l82_82466

def W := (2, 2)
def X := (-2, 2)
def Y := (-2, -2)
def Z := (2, -2)

inductive Transform
| L  -- rotation of 90° counterclockwise
| R  -- rotation of 90° clockwise
| F  -- reflection across the line y = x
| G  -- reflection across the line y = -x

def apply_transform : Transform → (ℤ × ℤ) → (ℤ × ℤ)
| Transform.L (x, y) := (-y, x)
| Transform.R (x, y) := (y, -x)
| Transform.F (x, y) := (y, x)
| Transform.G (x, y) := (-y, -x)

def apply_sequence (seq : List Transform) : (ℤ × ℤ) → (ℤ × ℤ) :=
  seq.foldr (λ t p, apply_transform t p) 

def all_vertices_return_to_original_positions (seq : List Transform) : Prop :=
  apply_sequence seq W = W ∧
  apply_sequence seq X = X ∧
  apply_sequence seq Y = Y ∧
  apply_sequence seq Z = Z

theorem number_of_valid_sequences : 
  Nat.card ({ seq : List Transform // List.length seq = 24 ∧ all_vertices_return_to_original_positions seq }) = 4^23 := 
sorry

end number_of_valid_sequences_l82_82466


namespace real_root_of_quadratic_l82_82190

theorem real_root_of_quadratic (b : ℝ) : 
  (∃ x : ℝ, (x^2 - 2 * complex.I * x + b = 1)) → (b = 1) :=
by
  sorry

end real_root_of_quadratic_l82_82190


namespace sum_of_coordinates_reflection_l82_82331

theorem sum_of_coordinates_reflection (y : ℝ) :
  let A := (3, y)
  let B := (3, -y)
  A.1 + A.2 + B.1 + B.2 = 6 :=
by
  let A := (3, y)
  let B := (3, -y)
  sorry

end sum_of_coordinates_reflection_l82_82331


namespace no_more_than_two_points_intersection_l82_82681

open Set

variable {Point Line ConvexFigure : Type}
variable [AffineSpace Point Line] [ConvexSpace ConvexFigure Point]

-- Definitions for geometry/convex concepts
def is_internal_point (Φ : ConvexFigure) (O : Point) : Prop := 
  ∃ U : Set Point, IsOpen U ∧ O ∈ U ∧ U ⊆ Φ

def intersects_boundary_exactly_two_points (l : Line) (Φ : ConvexFigure) : Prop :=
  ∃ A B : Point, A ≠ B ∧ A ∈ boundary Φ ∧ B ∈ boundary Φ ∧ ∀ x ∈ l, x ∈ Φ ↔ x = A ∨ x = B

noncomputable def prove_line_intersects_boundary_at_most_two_points (Φ : ConvexFigure) (O : Point) (l : Line) :
    is_internal_point Φ O → 
    (∃ A B : Point, A ≠ B ∧ A ∈ boundary Φ ∧ B ∈ boundary Φ ∧ (O ∈ line_through A B inner/seg))
    :=
begin
  sorry
end

theorem no_more_than_two_points_intersection (Φ : ConvexFigure) (O : Point) (l : Line) : 
    is_internal_point Φ O → 
    (∀ x, x ∈ l → x ∈ Φ → x ∈ boundary Φ) → 
    (bounded Φ → intersects_boundary_exactly_two_points l Φ) → 
    (∃ x y, x ∈ Φ ∧ y ∈ Φ ∧ x ∈ l ∧ y ∈ l ∧ (x ≠ y → x ∈ boundary Φ ∧ y ∈ boundary Φ))
    :=
begin
  intros,
  sorry
end

end no_more_than_two_points_intersection_l82_82681


namespace find_a2013_l82_82598

-- Define the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a(n+1) = a(n) + d

-- Given the conditions: a_2 + a_4024 = 4 and a_n is an arithmetic sequence
variables {a : ℕ → ℝ}
axiom a2_a4024 : a 2 + a 4024 = 4
axiom is_arithmetic_seq : arithmetic_sequence a

-- The theorem to be proven: a_2013 = 2
theorem find_a2013 : a 2013 = 2 :=
sorry

end find_a2013_l82_82598


namespace lotto_probability_lotto_profit_l82_82448

theorem lotto_probability (total_numbers drawn_numbers special_numbers chosen_numbers favorable_outcomes total_outcomes : ℕ) :
  total_numbers = 90 ∧ drawn_numbers = 5 ∧ special_numbers = 7 ∧ chosen_numbers = 3 ∧ 
  favorable_outcomes = @Finset.card ℕ (Finset.powersetLen chosen_numbers (Finset.range special_numbers)) * 
  @Finset.card ℕ (Finset.powersetLen 2 (Finset.Ico special_numbers total_numbers)) ∧ 
  total_outcomes = @Finset.card ℕ (Finset.powersetLen drawn_numbers (Finset.range total_numbers)) →
  favorable_outcomes / total_outcomes * 100 ≈ 0.271 :=
by sorry

theorem lotto_profit (total_people tickets_cost three_match_pay two_match_pay total_tickets three_match_tickets two_match_tickets total_winnings total_cost total_profit per_person_profit : ℕ) :
  total_people = 10 ∧ tickets_cost = 60 ∧ three_match_pay = 7000 ∧ two_match_pay = 300 ∧
  total_tickets = @Finset.card ℕ (Finset.powersetLen 5 (Finset.range 7)) ∧ 
  three_match_tickets = 6 ∧ two_match_tickets = 12 ∧ 
  total_winnings = three_match_tickets * three_match_pay + two_match_tickets * two_match_pay ∧ 
  total_cost = total_tickets * tickets_cost ∧
  total_profit = total_winnings - total_cost ∧
  per_person_profit = total_profit / total_people →
  per_person_profit = 4434 :=
by sorry

end lotto_probability_lotto_profit_l82_82448


namespace max_min_terms_sequence_l82_82582

def sequence (n : ℕ) : ℝ := (3/4)^(n-1) * ((3/4)^(n-1) - 1)

theorem max_min_terms_sequence :
  sequence 1 = 0 ∧ sequence 3 = -(63 / 256) ∧ 
  (∀ n ≥ 1, (sequence n ≤ sequence 1 ∧ (sequence n ≥ sequence 3))) := sorry

end max_min_terms_sequence_l82_82582


namespace sum_of_coordinates_reflection_l82_82332

theorem sum_of_coordinates_reflection (y : ℝ) :
  let A := (3, y)
  let B := (3, -y)
  A.1 + A.2 + B.1 + B.2 = 6 :=
by
  let A := (3, y)
  let B := (3, -y)
  sorry

end sum_of_coordinates_reflection_l82_82332


namespace triangle_DEF_area_is_correct_l82_82063

noncomputable def area_of_triangle_DEF : ℝ :=
  let area_ABC := 944
  let D := midpoint A B
  let E := midpoint B C
  let F := midpoint A E
  (1 / 8) * area_ABC

theorem triangle_DEF_area_is_correct : 
  (area_of_triangle_DEF = 118) :=
by
  sorry

end triangle_DEF_area_is_correct_l82_82063


namespace number_of_children_tickets_l82_82751

theorem number_of_children_tickets 
    (x y : ℤ) 
    (h1 : x + y = 225) 
    (h2 : 6 * x + 9 * y = 1875) : 
    x = 50 := 
  sorry

end number_of_children_tickets_l82_82751


namespace cos_PQS_eq_uv_l82_82978

variable (P Q R S : Type) [metric_space P] [metric_space Q] [metric_space R] [metric_space S]

variable [has_inner P] [has_inner Q] [has_inner R] [has_inner S]

variable (u v : ℝ)

-- Conditions in the problem
axiom angle_PQS_right : ∡ P Q S = π / 2
axiom angle_PRS_right : ∡ P R S = π / 2
axiom angle_QRS_right : ∡ Q R S = π / 2
axiom define_u : u = real.sin (∡ P Q R)
axiom define_v : v = real.sin (∡ P S R)

-- The proof problem statement
theorem cos_PQS_eq_uv : real.cos (∡ P Q S) = u * v := by
  sorry

end cos_PQS_eq_uv_l82_82978


namespace integral_sin_div_t_approximation_l82_82850

theorem integral_sin_div_t_approximation :
  abs (∫ t in 0..1, (sin t) / t - 0.9444) < 10^(-4) :=
sorry

end integral_sin_div_t_approximation_l82_82850


namespace n_must_be_power_of_3_l82_82701

theorem n_must_be_power_of_3 (n : ℕ) (h1 : 0 < n) (h2 : Prime (4 ^ n + 2 ^ n + 1)) : ∃ k : ℕ, n = 3 ^ k :=
by
  sorry

end n_must_be_power_of_3_l82_82701


namespace investment_options_count_l82_82445

theorem investment_options_count : 
  ∀ (n k : ℕ), 
  let P := λ (n k : ℕ), n.choose k * k.factorial in
  let C := λ (n k : ℕ), n.choose k in
  (n = 5 ∧ k = 3) →
  P 5 3 + C 5 1 * C 3 2 * C 4 1 = 120 :=
by intros n k P C h
   cases h
   simp only [P, C, Nat.choose, Nat.factorial]
   sorry

end investment_options_count_l82_82445


namespace Tammy_runs_10_laps_per_day_l82_82707

theorem Tammy_runs_10_laps_per_day
  (total_distance_per_week : ℕ)
  (track_length : ℕ)
  (days_per_week : ℕ)
  (h1 : total_distance_per_week = 3500)
  (h2 : track_length = 50)
  (h3 : days_per_week = 7) :
  (total_distance_per_week / track_length) / days_per_week = 10 := by
  sorry

end Tammy_runs_10_laps_per_day_l82_82707


namespace largest_profit_share_l82_82148

theorem largest_profit_share 
  (r₁ r₂ r₃ r₄ r₅ : ℕ) 
  (total_profit : ℚ) 
  (h_ratio : r₁ = 2 ∧ r₂ = 3 ∧ r₃ = 4 ∧ r₄ = 1 ∧ r₅ = 5) 
  (h_profit : total_profit = 34000) 
  : let total_parts := r₁ + r₂ + r₃ + r₄ + r₅
      part_value := total_profit / total_parts
  in max (r₁ * part_value) (max (r₂ * part_value) (max (r₃ * part_value) (max (r₄ * part_value) (r₅ * part_value)))) = 11333.35 := 
by {
  sorry
}

end largest_profit_share_l82_82148


namespace arithmetic_sequence_terms_l82_82505

theorem arithmetic_sequence_terms :
  let a := -5
  let d := 4
  let l := 39
  ∃ n : ℕ, l = a + (n - 1) * d ∧ n = 12 :=
begin
  sorry
end

end arithmetic_sequence_terms_l82_82505


namespace mutually_exclusive_not_complementary_l82_82340

open Probability -- Using probability namespace

-- Definitions for the events
def red_card (c : char) : Prop :=
  c = 'r' -- red card event

def yellow_card (c : char) : Prop :=
  c = 'y' -- yellow card event

def blue_card (c : char) : Prop :=
  c = 'b' -- blue card event

def person_A_gets_red (A : char) : Prop :=
  red_card A -- Person A gets red card 

def person_B_gets_red (B : char) : Prop :=
  red_card B -- Person B gets red card 

-- Lean statement for the equivalent proof problem
theorem mutually_exclusive_not_complementary :
  ∀ (A B C : char), 
  (red_card A ∨ yellow_card A ∨ blue_card A) ∧
  (red_card B ∨ yellow_card B ∨ blue_card B) ∧
  (red_card C ∨ yellow_card C ∨ blue_card C) ∧
  (A ≠ B) ∧ (B ≠ C) ∧ (A ≠ C) →
  (person_A_gets_red A → ¬ person_B_gets_red B) ∧ 
  (¬ person_A_gets_red A → (person_B_gets_red B ∨ ¬ person_B_gets_red B)) := 
by 
  sorry

end mutually_exclusive_not_complementary_l82_82340


namespace volume_of_smaller_tetrahedron_l82_82608

noncomputable def volume_ratio_tetrahedron : ℚ :=
  let regular_tetrahedron := [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)]
  let midpoints := [
    (1/2 : ℚ, 1/2, 0, 0),
    (1/2, 0, 1/2, 0),
    (1/2, 0, 0, 1/2),
    (0, 1/2, 1/2, 0),
    (0, 1/2, 0, 1/2),
    (0, 0, 1/2, 1/2)]
  in
  let side_length_larger := Math.sqrt 2
  let side_length_smaller := Math.sqrt (1/2)
  let volume_ratio := (side_length_smaller / side_length_larger)^3
  volume_ratio

theorem volume_of_smaller_tetrahedron (m n : ℕ) (hm : m = 1) (hn : n = 8) :
  volume_ratio_tetrahedron = m / n :=
sorry

end volume_of_smaller_tetrahedron_l82_82608


namespace stratified_sampling_l82_82810

-- Conditions
def total_students : ℕ := 1200
def freshmen : ℕ := 300
def sophomores : ℕ := 400
def juniors : ℕ := 500
def sample_size : ℕ := 60
def probability : ℚ := sample_size / total_students

-- Number of students to be sampled from each grade
def freshmen_sampled : ℚ := freshmen * probability
def sophomores_sampled : ℚ := sophomores * probability
def juniors_sampled : ℚ := juniors * probability

-- Theorem to prove
theorem stratified_sampling :
  freshmen_sampled = 15 ∧ sophomores_sampled = 20 ∧ juniors_sampled = 25 :=
by
  -- The actual proof would go here
  sorry

end stratified_sampling_l82_82810


namespace smallest_degree_polynomial_l82_82452

-- Define the conditions
def has_rational_coefficients (p : Polynomial ℚ) : Prop := p.coeffs.∀(fun c => is_rational c)

def roots (n : ℕ) : Set ℂ :=
  { n + root_of_unity ^ k * cbrt(n+1) | k ∈ {0, 1, 2} }

-- Define the problem statement
theorem smallest_degree_polynomial :
  ∃ p : Polynomial ℚ, has_rational_coefficients p ∧
  (∀ n : ℕ, 1 ≤ n ∧ n ≤ 100 → ∀ z ∈ roots n, Polynomial.eval z p = 0) ∧
  Polynomial.degree p = 300 :=
by
  sorry

end smallest_degree_polynomial_l82_82452


namespace triangle_angle_bisector_length_l82_82987

theorem triangle_angle_bisector_length (PQ QR PR : ℝ) (hPQ : PQ = 8) (hQR : QR = 15) (hPR : PR = 17) :
  let PS := Math.sqrt (PQ^2 * QR / (PQ + PR)) in
  PS = Real.sqrt 87.04 :=
by
  -- sorry means the proof is omitted
  sorry

end triangle_angle_bisector_length_l82_82987


namespace find_n_l82_82812

-- Defining the parameters and conditions
def large_block_positions (n : ℕ) : ℕ := 199 * n + 110 * (n - 1)

-- Theorem statement
theorem find_n (h : large_block_positions n = 2362) : n = 8 :=
sorry

end find_n_l82_82812


namespace new_ratio_of_stamps_l82_82730

variable {x : ℕ} -- assuming non-negative counts of stamps

def k_initial := 5 * x
def a_initial := 3 * x

def k_after_gift := k_initial - 12
def a_after_gift := a_initial + 12

theorem new_ratio_of_stamps :
  k_after_gift = a_after_gift + 32 → (k_after_gift : ℚ) / a_after_gift = 4 / 3 := by
  sorry

end new_ratio_of_stamps_l82_82730


namespace smallest_degree_polynomial_l82_82453

-- Define the conditions
def has_rational_coefficients (p : Polynomial ℚ) : Prop := p.coeffs.∀(fun c => is_rational c)

def roots (n : ℕ) : Set ℂ :=
  { n + root_of_unity ^ k * cbrt(n+1) | k ∈ {0, 1, 2} }

-- Define the problem statement
theorem smallest_degree_polynomial :
  ∃ p : Polynomial ℚ, has_rational_coefficients p ∧
  (∀ n : ℕ, 1 ≤ n ∧ n ≤ 100 → ∀ z ∈ roots n, Polynomial.eval z p = 0) ∧
  Polynomial.degree p = 300 :=
by
  sorry

end smallest_degree_polynomial_l82_82453


namespace fill_box_with_cubes_l82_82775

-- Define the dimensions of the box
def boxLength : ℕ := 35
def boxWidth : ℕ := 20
def boxDepth : ℕ := 10

-- Define the greatest common divisor of the box dimensions
def gcdBoxDims : ℕ := Nat.gcd (Nat.gcd boxLength boxWidth) boxDepth

-- Define the smallest number of identical cubes that can fill the box
def smallestNumberOfCubes : ℕ := (boxLength / gcdBoxDims) * (boxWidth / gcdBoxDims) * (boxDepth / gcdBoxDims)

theorem fill_box_with_cubes :
  smallestNumberOfCubes = 56 :=
by
  -- Proof goes here
  sorry

end fill_box_with_cubes_l82_82775


namespace Faye_money_left_l82_82885

def Faye_initial_amount : ℝ := 20
def given_by_mother (initial_amount : ℝ) : ℝ := 2 * initial_amount
def total_amount (initial_amount : ℝ) (amount_given : ℝ) : ℝ := initial_amount + amount_given
def cupcake_cost : ℝ := 1.5
def total_cupcake_cost (cost_per_cupcake : ℝ) (quantity : ℕ) : ℝ := cost_per_cupcake * quantity
def cookie_box_cost : ℝ := 3
def total_cookie_cost (cost_per_box : ℝ) (quantity : ℕ) : ℝ := cost_per_box * quantity
def total_spent (cupcake_total : ℝ) (cookie_total : ℝ) : ℝ := cupcake_total + cookie_total
def amount_left (total_amount : ℝ) (total_spent : ℝ) : ℝ := total_amount - total_spent

theorem Faye_money_left :
  let initial_amount := Faye_initial_amount in
  let amount_given := given_by_mother initial_amount in
  let total := total_amount initial_amount amount_given in
  let cupcake_total := total_cupcake_cost cupcake_cost 10 in
  let cookie_total := total_cookie_cost cookie_box_cost 5 in
  let spent := total_spent cupcake_total cookie_total in
  amount_left total spent = 30 := by
  sorry

end Faye_money_left_l82_82885


namespace square_side_length_l82_82727

theorem square_side_length (x : ℝ) (h : 4 * x = 2 * x^2) : x = 0 ∨ x = 2 := 
by
suffices h' : x^2 - 2 * x = 0, from sorry,
calc
  4 * x = 2 * x^2  : h
  ... = x^2 - 2 * x : sorry

end square_side_length_l82_82727


namespace number_of_ways_to_sum_is_2_l82_82087

def digits : List ℕ := [9,8,7,6,5,4,3,2,1]

noncomputable def group_sum_to_n (xs : List ℕ) (n : ℕ) : ℕ :=
  if xs.length = 1 then xs.head!
  else if xs.length ≥ 2 then
    (xs.head! * 10 + xs.tail!.head!).sum
  else 0

theorem number_of_ways_to_sum_is_2 : ∃ ways, ways = 2 ∧ List.length (filter (λ g, (group_sum_to_n (digits.g) 99)) (groupings digits)) = ways := 
by {
  sorry
}

end number_of_ways_to_sum_is_2_l82_82087


namespace mean_of_sequence_starting_at_3_l82_82710

def arithmetic_sequence (start : ℕ) (n : ℕ) : List ℕ :=
List.range n |>.map (λ i => start + i)

def arithmetic_mean (seq : List ℕ) : ℚ := (seq.sum : ℚ) / seq.length

theorem mean_of_sequence_starting_at_3 : 
  ∀ (seq : List ℕ),
  seq = arithmetic_sequence 3 60 → 
  arithmetic_mean seq = 32.5 := 
by
  intros seq h
  rw [h]
  sorry

end mean_of_sequence_starting_at_3_l82_82710


namespace sequence_periodicity_l82_82544

theorem sequence_periodicity 
  (a b : ℤ)
  (a_seq : ℕ → ℤ) 
  (h1 : a_seq 1 = a) 
  (h2 : a_seq 2 = b) 
  (h_recur : ∀ n, 2 ≤ n → a_seq (n + 1) = a_seq n - a_seq (n - 1)) : 
  a_seq 100 = -a ∧ (∑ n in (Finset.range 100).map Finset.univ.inject, a_seq (n + 1)) = 2 * b - a :=
  sorry

end sequence_periodicity_l82_82544


namespace shaded_area_is_ten_square_inches_l82_82624

noncomputable def area_shaded_region : ℝ :=
  let area_square_4x4 := 16
  let area_triangle_DGF := 6
  area_square_4x4 - area_triangle_DGF

theorem shaded_area_is_ten_square_inches
  (area_square_4x4 : ℝ := 4 * 4)
  (area_triangle_DGF : ℝ := 1 / 2 * 3 * 4) :
  area_shaded_region = 10 := by
  simp [area_shaded_region, area_square_4x4, area_triangle_DGF]
  exact rfl

end shaded_area_is_ten_square_inches_l82_82624


namespace rachelle_GPA_probability_l82_82685

def probability_of_grade (subject : String) (grade : Char) : ℚ :=
  match subject, grade with
  | "Math", 'A'     => 1
  | "Science", 'A'  => 1
  | "English", 'A'  => 1/5
  | "English", 'B'  => 1/5
  | "English", 'C'  => 3/5
  | "History", 'A'  => 1/4
  | "History", 'B'  => 1/2
  | "History", 'C'  => 1/4
  | "Art", 'A'      => 1/3
  | "Art", 'B'      => 1/3
  | "Art", 'C'      => 1/3
  | _, _            => 0

def grade_points (grade : Char) : ℕ :=
  match grade with
  | 'A' => 4
  | 'B' => 3
  | 'C' => 2
  | _   => 0

/--
We need to prove that the probability Rachelle will achieve a GPA of at least 3.6
given the conditions about her grades and probability distributions is 17/40.
-/
theorem rachelle_GPA_probability :
  let points_needed := 18
  let points_provided := (grade_points 'A') + (grade_points 'A')
  let required_points := points_needed - points_provided
  let combinations_producing_required_points := sorry -- define combinations and their probabilities
  let total_probability := sorry -- final probability calculation
  total_probability = 17 / 40
:= by
  -- calculations and probability aggregations go here
  sorry

end rachelle_GPA_probability_l82_82685


namespace butterflies_original_number_l82_82633

theorem butterflies_original_number (x : ℕ) :
  (1.25 * x - 11 = 82) → x = 74 := by
  sorry

end butterflies_original_number_l82_82633


namespace find_value_of_y_l82_82226

theorem find_value_of_y (x y : ℕ) 
    (h1 : 2^x - 2^y = 3 * 2^12) 
    (h2 : x = 14) : 
    y = 13 := 
by
  sorry

end find_value_of_y_l82_82226


namespace minimum_employees_needed_l82_82463

theorem minimum_employees_needed
  (n_W : ℕ) (n_A : ℕ) (n_S : ℕ)
  (n_WA : ℕ) (n_AS : ℕ) (n_SW : ℕ)
  (n_WAS : ℕ)
  (h_W : n_W = 115)
  (h_A : n_A = 92)
  (h_S : n_S = 60)
  (h_WA : n_WA = 32)
  (h_AS : n_AS = 20)
  (h_SW : n_SW = 10)
  (h_WAS : n_WAS = 5) :
  n_W + n_A + n_S - (n_WA - n_WAS) - (n_AS - n_WAS) - (n_SW - n_WAS) + 2 * n_WAS = 225 :=
by
  sorry

end minimum_employees_needed_l82_82463


namespace find_garden_dimensions_l82_82963

noncomputable def length_diagonal_of_garden (P B : ℝ) (L D : ℝ) : Prop :=
  P = 600 ∧ B = 200 ∧ L = 100 ∧ D = real.sqrt ((100:ℝ)^2 + (200:ℝ)^2)

-- The exact statement to prove, using the conditions and targets identified
theorem find_garden_dimensions :
  length_diagonal_of_garden 600 200 100 223.61 :=
by
  sorry

end find_garden_dimensions_l82_82963


namespace João_more_4kg_than_2kg_l82_82637

-- Definitions based on the conditions
def João_bars (x y z : ℕ) : Prop :=
  x + y + z = 30 ∧ 2 * x + 3 * y + 4 * z = 100

-- Theorem that João has more 4 kg bars than 2 kg bars
theorem João_more_4kg_than_2kg (x y z : ℕ) (h : João_bars x y z) : z > x :=
  sorry

end João_more_4kg_than_2kg_l82_82637


namespace chain_length_of_cells_l82_82401

theorem chain_length_of_cells (d : ℝ) (n : ℝ) 
  (h₀ : d = 5 * 10^(-5)) 
  (h₁ : n = 2 * 10^3) : 
  (n * d) = 10^(-1) :=
by
  sorry

end chain_length_of_cells_l82_82401


namespace find_k_l82_82525

variables (a b k : ℝ) (x : ℂ)

theorem find_k (h1 : a = 5) (h2 : b = 7)
  (h_roots : ∀ x, x^2 + (↑b / ↑a) * x + (↑k / ↑a) = 0 ↔ x = (↑-b + complex.I * complex.sqrt (171)) / (2 * ↑a)
    ∨ x = (↑-b - complex.I * complex.sqrt (171)) / (2 * ↑a)) :
  k = 11 :=
by
  have h : (7 : ℝ) = ↑7 := by norm_cast
  have h_171 : (171 : ℝ) = ↑171 := by norm_cast
  rw [h171, hsqrt (171 : ℤ)] at h_roots
  sorry

end find_k_l82_82525


namespace Vlad_height_feet_l82_82003

theorem Vlad_height_feet 
  (sister_height_feet : ℕ)
  (sister_height_inches : ℕ)
  (vlad_height_diff : ℕ)
  (vlad_height_inches : ℕ)
  (vlad_height_feet : ℕ)
  (vlad_height_rem : ℕ)
  (sister_height := (sister_height_feet * 12) + sister_height_inches)
  (vlad_height := sister_height + vlad_height_diff)
  (vlad_height_feet_rem := (vlad_height / 12, vlad_height % 12)) 
  (h_sister_height : sister_height_feet = 2)
  (h_sister_height_inches : sister_height_inches = 10)
  (h_vlad_height_diff : vlad_height_diff = 41)
  (h_vlad_height : vlad_height = 75)
  (h_vlad_height_feet : vlad_height_feet = 6)
  (h_vlad_height_rem : vlad_height_rem = 3) :
  vlad_height_feet = 6 := by
  sorry

end Vlad_height_feet_l82_82003


namespace problem_I_problem_II_l82_82574

/-- Proof problem I: Given f(x) = |x - 1|, prove that the inequality f(x) ≥ 4 - |x - 1| implies x ≥ 3 or x ≤ -1 -/
theorem problem_I (x : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = |x - 1|) (h2 : f x ≥ 4 - |x - 1|) : x ≥ 3 ∨ x ≤ -1 :=
  sorry

/-- Proof problem II: Given f(x) = |x - 1| and 1/m + 1/(2*n) = 1 (m > 0, n > 0), prove that the minimum value of mn is 2 -/
theorem problem_II (f : ℝ → ℝ) (h1 : ∀ x, f x = |x - 1|) (m n : ℝ) (hm : m > 0) (hn : n > 0) (h2 : 1/m + 1/(2*n) = 1) : m*n ≥ 2 :=
  sorry

end problem_I_problem_II_l82_82574


namespace positive_difference_l82_82144

theorem positive_difference (x : ℝ) (h : real.cbrt (9 - x^2 / 4) = -3) : 
  |12 - (-12)| = 24 :=
by
  sorry

end positive_difference_l82_82144


namespace scientists_birth_in_may_percentage_l82_82717

def total_scientists : ℕ := 120
def may_scientists : ℕ := 7

def percentage_birth_in_may (total : ℕ) (born_in_may : ℕ) : ℚ := 
  (born_in_may.to_rat / total.to_rat) * 100

theorem scientists_birth_in_may_percentage :
  percentage_birth_in_may total_scientists may_scientists = 5.83 := 
sorry

end scientists_birth_in_may_percentage_l82_82717


namespace multiply_correctly_l82_82634

theorem multiply_correctly :
  ∃ x : ℤ, (20 - x = 60) ∧ (34 * x = -1360) :=
begin
  use -40,
  split,
  { linarith, },
  { linarith, },
end

end multiply_correctly_l82_82634


namespace percentage_error_in_square_area_percentage_error_l82_82022

theorem percentage_error_in_square_area (s : ℝ) (h : s > 0) : 
  (1.04 * s) ^ 2 = 1.0816 * s ^ 2 :=
by 
  sorry

theorem percentage_error (s : ℝ) (h : s > 0) : 
  ((1.0816 * s ^ 2 - s ^ 2) / s ^ 2) * 100 = 8.16 :=
by 
  calc
    ((1.0816 * s ^ 2 - s ^ 2) / s ^ 2) * 100 
        = (0.0816 * s ^ 2 / s ^ 2) * 100 : by sorry
    ... = 0.0816 * 100 : by sorry
    ... = 8.16 : by sorry

end percentage_error_in_square_area_percentage_error_l82_82022


namespace reflections_concur_and_are_on_circumcircle_l82_82292

variable {ABC : Type} [linear_ordered_field ABC]

open_locale classical

noncomputable def orthocenter (A B C : ABC) : ABC := sorry
noncomputable def circumcircle (A B C : ABC) : ABC := sorry
noncomputable def intersection_points (l : ABC) (A B C: ABC): ABC × ABC × ABC := sorry
noncomputable def reflection_line (l : ABC) (A B C : ABC) : ABC := sorry

theorem reflections_concur_and_are_on_circumcircle
  (A B C H : ABC) (l : ABC)
  (orthocenter_is_correct : H = orthocenter A B C)
  (P Q R : ABC := (intersection_points l A B C).fst)
  (orthocenter_intersections_line : H ∈ l):
  let l_A := reflection_line l B C
  let l_B := reflection_line l C A
  let l_C := reflection_line l A B
  let M := sorry, -- The common point of reflections
  M ∈ circumcircle A B C := sorry

end reflections_concur_and_are_on_circumcircle_l82_82292


namespace not_a_right_triangle_l82_82073

def is_right_triangle (angles : List ℕ) : Prop :=
  angles.contains 90

theorem not_a_right_triangle :
  ∀ angles, (angles = [45, 60, 75] → ¬is_right_triangle angles) :=
by
  intros angles h
  rw h
  unfold is_right_triangle
  intro contra
  cases contra
  sorry

end not_a_right_triangle_l82_82073


namespace min_a_n_over_n_l82_82167

noncomputable def a : ℕ → ℕ
| 1       => 33
| (n + 1) => a n + 2 * n

def a_n_over_n (n : ℕ) : ℚ :=
  (a n : ℚ) / n

theorem min_a_n_over_n {n : ℕ} (h1 : n > 0) (h2 : n ≤ 6) : 
  ∃ n, a_n_over_n n = 21 / 2 :=
by
  sorry

end min_a_n_over_n_l82_82167


namespace twice_largest_two_digit_from_3_5_9_l82_82001

def is_tens_and_units_digit (a b : ℕ) : Prop := (a ≠ b) ∧ (a ∈ {3, 5, 9}) ∧ (b ∈ {3, 5, 9})

def largest_two_digit_number (a b : ℕ) : ℕ := if a > b then 10 * a + b else 10 * b + a

theorem twice_largest_two_digit_from_3_5_9 :
  ∀ a b, is_tens_and_units_digit a b → 2 * (largest_two_digit_number a b) = 190 :=
by
  intros a b h
  sorry

end twice_largest_two_digit_from_3_5_9_l82_82001


namespace painted_pictures_in_june_l82_82076

theorem painted_pictures_in_june (J : ℕ) (h1 : J + (J + 2) + 9 = 13) : J = 1 :=
by
  -- Given condition translates to J + J + 2 + 9 = 13
  -- Simplification yields 2J + 11 = 13
  -- Solving 2J + 11 = 13 gives J = 1
  sorry

end painted_pictures_in_june_l82_82076


namespace least_constant_inequality_l82_82660

theorem least_constant_inequality (n : ℕ) (hn : n ≥ 2) (x : Fin n → ℝ) (hx : ∀ i, 0 ≤ x i) :
  (∑ i j in Finset.Icc (0 : Fin n) (n - 1: ℕ).to_finset, if i < j then x i * x j * (x i ^ 2 + x j ^ 2) else 0) ≤ 
  (1 / 8) * (∑ i in Finset.univ, x i) ^ 4 :=
sorry

end least_constant_inequality_l82_82660


namespace fewest_keystrokes_to_reach_500_l82_82038

-- Define operations on the display
def op_add1 (n : ℕ) : ℕ := n + 1
def op_sub1 (n : ℕ) : ℕ := n - 1
def op_mul2 (n : ℕ) : ℕ := n * 2

-- Prove the least number of keystrokes needed to reach 500 from 1
theorem fewest_keystrokes_to_reach_500 : 
  ∃ (k : ℕ), 
    (∀ n, 
      (n = 1 → (∃ ops : list (ℕ → ℕ), 
                   (list.length ops = k) ∧ 
                   (list.foldl (λ a f, f a) n ops = 500) ∧ 
                   (list.any ops (λ f, f = op_add1)) ∧ 
                   (list.any ops (λ f, f = op_sub1)) ∧ 
                   (list.any ops (λ f, f = op_mul2)))) ∧
    k = 13
  :=
sorry

end fewest_keystrokes_to_reach_500_l82_82038


namespace evaluate_expression_l82_82027

-- Introduce the expression as a Lean definition
def expression := (- (1 / 2))⁻¹ + (Real.pi - 3)^0 + abs (1 - Real.sqrt 2) + Real.sin (Real.pi / 4) * Real.sin (Real.pi / 6)

-- State the theorem to be proven
theorem evaluate_expression : expression = (5 * Real.sqrt 2) / 4 - 2 := 
by 
  sorry

end evaluate_expression_l82_82027


namespace discounted_price_of_milk_is_2_l82_82808

theorem discounted_price_of_milk_is_2:
  ∃ (P : ℝ), (P = 2) →
  (∀ (C : ℝ) (price_milk : ℝ), price_milk = 3 → (C - 1) * 5 + (3 * (price_milk - P)) = 8) :=
begin
  sorry
end

end discounted_price_of_milk_is_2_l82_82808


namespace number_of_solutions_l82_82291

def f0 (x : ℝ) : ℝ := x + |x-200| - |x+200|

def f : (ℕ → ℝ → ℝ)
| 0     := f0
| (n+1) := λ x, |f n x| - 2

theorem number_of_solutions : ∃! (x1 x2 : ℝ), (f 100 x1 = 0 ∧ f 100 x2 = 0 ∧ x1 ≠ x2) :=
by {
  -- Define the necessary conditions and final proof goal
  sorry
}

end number_of_solutions_l82_82291
