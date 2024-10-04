import Mathlib

namespace range_of_a_l281_281550

def f (x : ℝ) : ℝ := (2 ^ x - 2 ^ (-x)) / 2
def g (x : ℝ) : ℝ := (2 ^ x + 2 ^ (-x)) / 2

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → a * f x + g (2 * x) ≥ 0) → a ≥ -17 / 6 :=
by
  sorry

end range_of_a_l281_281550


namespace derivative_value_l281_281843

variable {f : ℝ → ℝ}
variable {x₀ : ℝ}

-- Given condition
axiom limit_given : (tendsto (λ (Δx : ℝ), (f (x₀ + 2 * Δx) - f x₀) / (3 * Δx)) (𝓝 0) (𝓝 1))

-- Theorem to prove
theorem derivative_value : deriv f x₀ = 3 / 2 := sorry

end derivative_value_l281_281843


namespace monotonicity_of_f_range_of_a_l281_281555

def f (x : ℝ) : ℝ := 2 * Real.log x + 1 / x

theorem monotonicity_of_f :
  (∀ x ∈ Set.Ioo 0 (1 / 2 : ℝ), f' x < 0) ∧ (∀ x ∈ Set.Ioi (1 / 2 : ℝ), f' x > 0) :=
sorry

theorem range_of_a (a : ℝ) :
  (∀ x ≥ 1, f x ≤ a * x) ↔ 1 ≤ a :=
sorry

end monotonicity_of_f_range_of_a_l281_281555


namespace solve_for_value_of_2x2y_l281_281483

open Real

-- Definitions as per the conditions
variables (x y : ℝ)

-- Condition definitions
def cond_1 : 2^x = 5 := sorry
def cond_2 : 4^y = 3 := sorry

-- Theorem stating the mathematical equivalence we aim to prove
theorem solve_for_value_of_2x2y (h1 : 2^x = 5) (h2 : 4^y = 3) : 2^(x + 2 * y) = 15 :=
by
  sorry

end solve_for_value_of_2x2y_l281_281483


namespace winning_prob_correct_l281_281730

noncomputable def probability_winning_contest : ℚ :=
  let probability_one_correct := 1 / 4
  let probability_one_incorrect := 3 / 4
  let probability_all_four_correct := probability_one_correct ^ 4
  let combinations_three_correct := 4
  let probability_three_correct := combinations_three_correct * (probability_one_correct ^ 3 * probability_one_incorrect)
  probability_all_four_correct + probability_three_correct

theorem winning_prob_correct :
  probability_winning_contest = 13 / 256 :=
by
  sorry

end winning_prob_correct_l281_281730


namespace diameter_length_of_circle_l281_281150

-- Definitions based on conditions
noncomputable def length_of_PD : ℝ := x
noncomputable def length_of_PC : ℝ := y
noncomputable def length_of_AP_extends_to_Q : Prop := PQ = PA - AQ

-- Main theorem statement
theorem diameter_length_of_circle (h1: length_of_PD = length_of_PC)
(h2: x ≠ y):
length_of_diameter = 2 * x :=
begin
  sorry
end

end diameter_length_of_circle_l281_281150


namespace sum_palindromic_primes_eq_1383_l281_281954

def is_prime (n : ℕ) : Prop := sorry -- Definition of prime numbers can be imported from Mathlib

def is_palindromic (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

def is_palindromic_prime (n : ℕ) : Prop :=
  is_prime n ∧ is_palindromic n ∧ n < 200 ∧ 100 ≤ n

def sum_palindromic_primes : ℕ :=
  (List.range 1000).filter is_palindromic_prime |> List.sum

theorem sum_palindromic_primes_eq_1383 : sum_palindromic_primes = 1383 :=
  sorry

end sum_palindromic_primes_eq_1383_l281_281954


namespace angle_QMP_deg_l281_281865

theorem angle_QMP_deg (J K L M P Q : Type*)
  [line_segment J K][line_segment L M][line_segment J L]
  (parallel_JK_LM : parallel J K L M)
  (angles : ∃ y : ℝ, angle Q M L = 2 * y ∧ angle L J P = 3 * y ∧ angle J P L = y) :
  angle Q M P = 72 :=
by
  sorry

end angle_QMP_deg_l281_281865


namespace optionA_optionC_l281_281907

variables (a b : ℝ) (hp : a > 0) (hq : b > 0)

def A (a b : ℝ) : ℝ := (a + b) / 2
def G (a b : ℝ) : ℝ := Real.sqrt (a * b)
def L (p a b : ℝ) : ℝ := (a^p + b^p) / (a^(p-1) + b^(p-1))

-- Prove L_{0.5}(a, b) ≤ A(a, b)
theorem optionA : L 0.5 a b ≤ A a b :=
sorry

-- Prove L_2(a, b) ≥ L_1(a, b)
theorem optionC : L 2 a b ≥ L 1 a b :=
sorry

end optionA_optionC_l281_281907


namespace least_possible_value_expression_l281_281758

theorem least_possible_value_expression :
  ∃ (x : ℝ), (x + 2) * (x + 3) * (x + 4) * (x + 5) + 2040 = 2039 :=
sorry

end least_possible_value_expression_l281_281758


namespace michael_truck_meetings_l281_281175

-- Define the initial conditions
def michael_speed := 7 -- in feet per second
def truck_speed := 14 -- in feet per second
def pail_distance := 150 -- in feet
def truck_stop_time := 20 -- in seconds
def initial_gap := 150 -- initial separation in feet when Michael starts

-- Define functions to compute their respective positions at time t
def michael_position (t : ℝ) : ℝ := michael_speed * t
def truck_position (t : ℝ) : ℝ := 
  let cycle_time := pail_distance / truck_speed + truck_stop_time
  let cycles := (t / cycle_time).floor
  let cycle_position := truck_speed * (t % cycle_time)
  if t % cycle_time < pail_distance / truck_speed then
    cycle_position
  else
    pail_distance -- truck stopped at the pail

-- State the theorem: they will meet 3 times
theorem michael_truck_meetings {t : ℝ} :
  ∃ t1 t2 t3 : ℝ, 
    michael_position t1 = truck_position t1 ∧
    michael_position t2 = truck_position t2 ∧
    michael_position t3 = truck_position t3 ∧
    t1 < t2 ∧ t2 < t3 :=
sorry

end michael_truck_meetings_l281_281175


namespace agent_007_encryption_l281_281348

theorem agent_007_encryption : ∃ (m n : ℕ), (0.07 : ℝ) = (1 / m : ℝ) + (1 / n : ℝ) := 
sorry

end agent_007_encryption_l281_281348


namespace cos_sin_range_l281_281512

-- Given point P(x, 1) with x >= 1, and theta such that its terminal side passes through P
def P (x : ℝ) : ℝ × ℝ := (x, 1)
def r (x : ℝ) : ℝ := sqrt (x^2 + 1)
def cosθ (x : ℝ) : ℝ := x / r x
def sinθ (x : ℝ) : ℝ := 1 / r x

theorem cos_sin_range (x : ℝ) (h : x ≥ 1) :
  1 < cosθ x + sinθ x ∧ cosθ x + sinθ x ≤ sqrt 2 :=
by
  sorry

end cos_sin_range_l281_281512


namespace log_sum_eq_seven_l281_281114

variable {b : ℕ → ℝ}

-- Geometric sequence property: b{i} * b{n+1-i} is constant for fixed n
axiom geom_seq {b : ℕ → ℝ} (n : ℕ) (h : ∀ i, 1 ≤ i ∧ i ≤ n → b i * b (n + 1 - i) = c) : c

-- All terms of the sequence are positive
axiom pos_terms {b : ℕ → ℝ} (h : ∀ n, 0 < b n) : True

-- Specific condition given in the problem
axiom given_condition {b : ℕ → ℝ} : b 7 * b 8 = 3

-- The theorem we need to prove
theorem log_sum_eq_seven {b : ℕ → ℝ} 
  (h_geom : ∀ i, 1 ≤ i ∧ i ≤ 14 → b i * b (14 + 1 - i) = 3) 
  (h_pos : ∀ n, 0 < b n) 
  (h_given : b 7 * b 8 = 3) : 
  ∑ i in (Finset.range 14).map (Function.embedding.subtype _), log 3 (b (i + 1)) = 7 := 
by 
  sorry

end log_sum_eq_seven_l281_281114


namespace verification_l281_281905

variables {a b : ℝ} (h_pos_a : 0 < a) (h_pos_b : 0 < b)

def arithmetic_mean (a b : ℝ) := (a + b) / 2
def geometric_mean (a b : ℝ) := real.sqrt (a * b)
def lehmer_mean (p a b : ℝ) := (a^p + b^p) / (a^(p-1) + b^(p-1))

theorem verification (h_pos_a : 0 < a) (h_pos_b : 0 < b) :
  lehmer_mean 0.5 a b ≤ arithmetic_mean a b ∧
  lehmer_mean 2 a b ≥ lehmer_mean 1 a b :=
by
  sorry

end verification_l281_281905


namespace lowest_degree_is_4_l281_281286

noncomputable def lowest_degree_polynomial (P : ℝ → ℝ) : ℕ :=
  if ∃ b : ℤ, (∀ coeff ∈ (P.coefficients), coeff < (b : ℝ) ∨ coeff > (b : ℝ)) ∧ (¬ ∃ coeff ∈ (P.coefficients), coeff = (b : ℝ))
  then Polynomial.natDegree P
  else 0

theorem lowest_degree_is_4 : ∀ (P : Polynomial ℝ), 
  (∃ b : ℤ, (∀ coeff ∈ P.coefficients, coeff < (b : ℝ) ∨ coeff > (b : ℝ)) ∧ (¬ ∃ coeff ∈ P.coefficients, coeff = (b : ℝ)))
  → lowest_degree_polynomial P = 4 :=
by
  sorry

end lowest_degree_is_4_l281_281286


namespace isosceles_triangles_in_ABC_l281_281521

theorem isosceles_triangles_in_ABC :
  ∀ (A B C D F P Q S : Type) [decidable_eq A] [decidable_eq B] [decidable_eq C],
  ∠ABC = 60 → ∠ACB = 45 →
  (AD : line A D) ∧ (CF : line C F) ∧ intersection AD CF P →
  (BE : angle_bisector B) ∧ intersection BE AD Q ∧ intersection BE CF S →
  number_of_isosceles_triangles(ABC, AD, CF, P, BE, Q, S) = 5 :=
by
  sorry

end isosceles_triangles_in_ABC_l281_281521


namespace find_cost_price_maximize_total_profit_l281_281698

/-
Problem part 1: Finding the cost price of types A and B
-/
theorem find_cost_price : 
  ∃ (x y : ℕ), 
    (7 * x + 8 * y = 380) ∧ 
    (10 * x + 6 * y = 380) ∧ 
    (x = 20) ∧ 
    (y = 30) := sorry

/-
Problem part 2: Maximizing the total profit
-/
theorem maximize_total_profit : 
  ∃ (a b : ℕ), 
    (a + b = 40) ∧ 
    (20 * a + 30 * b ≤ 900) ∧ 
    (5 * a + 7 * b ≥ 216) ∧ 
    (a = 30) ∧ 
    (b = 10) ∧ 
    (-2 * a + 280 = 220) := sorry

end find_cost_price_maximize_total_profit_l281_281698


namespace ellipse_eq_and_range_l281_281052

-- Definitions based on given conditions
def ellipse (a b : ℝ) := λ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1

def eccentricity (a b c : ℝ) := c / a

-- Conditions
def conditions (a b : ℝ) := a > b ∧ b > 0 ∧ eccentricity a b (Real.sqrt (a^2 - b^2)) = Real.sqrt 2 / 2 
def passes_through (x y : ℝ) := x = 1 ∧ y = Real.sqrt 2 / 2

-- Theorem statement translating the proof problem to Lean
theorem ellipse_eq_and_range (a b : ℝ) (P : ℝ × ℝ) :
  conditions a b →
  passes_through 1 (Real.sqrt 2 / 2) →
  P = (0,2) →
  (ellipse a b 1 (Real.sqrt 2 / 2) = 1) ∧
  ((P.1 = 0 ∧ P.2 = 2) →
  (∀ (k : ℝ), 
    (k > Real.sqrt 6 / 2 ∨ k < -Real.sqrt 6 / 2) →
    -Real.sqrt 6 / 4 < -2 * k / (1 + 2 * k^2) ∧ -2 * k / (1 + 2 * k^2) < Real.sqrt 6 / 4 ∨ 
    0 < -2 * k / (1 + 2 * k^2)  ∧ -2 * k / (1 + 2 * k^2) < Real.sqrt 6 / 4)) := by
  sorry

end ellipse_eq_and_range_l281_281052


namespace quadrant_of_w_l281_281457

def z : ℂ := 1 - 2 * complex.I

def z_inv : ℂ := 1 / z

def w : ℂ := z + z_inv

theorem quadrant_of_w :
  complex.re w > 0 ∧ complex.im w < 0 := by
  sorry

end quadrant_of_w_l281_281457


namespace constant_ratio_l281_281317

-- Define the geometric setup and conditions
-- A, B, C, D are points such that AB and CD are segments moving along a given angle and remain parallel
-- M is the intersection of AB and CD
variables {α : Type*} [linear_ordered_field α] {A B C D M : α}

-- Formalize the conditions
-- Assuming the geometric constraints and parallel movement
def segments_move_along_angle (A B C D M : α) : Prop :=
  ∀ (A' B' C' D' : α), (A ≠ A' ∧ B ≠ B' ∧ C ≠ C' ∧ D ≠ D') →
  is_parallel A B C D ∧ (M ∈ seg A B) ∧ (M ∈ seg C D)

-- Define the required ratio
def ratio (A B C D M : α) : α :=
  (M.1 - A.1) * (M.1 - B.1) / ((M.1 - C.1) * (M.1 - D.1)) 

-- Statement of the theorem
theorem constant_ratio (A B C D M : α) (h : segments_move_along_angle A B C D M) : 
  ∃ k : α, ∀ (A' B' C' D' M' : α), (segments_move_along_angle A' B' C' D' M') → ratio A B C D M = k :=
sorry

end constant_ratio_l281_281317


namespace min_sides_of_polygon_that_overlaps_after_rotation_l281_281494

theorem min_sides_of_polygon_that_overlaps_after_rotation (θ : ℝ) (n : ℕ) 
  (hθ: θ = 36) (hdiv: 360 % θ = 0) :
    n = 10 :=
by
  sorry

end min_sides_of_polygon_that_overlaps_after_rotation_l281_281494


namespace find_a10_l281_281064

-- Define the sequence a_n based on the given pattern
def sequence_an (n : ℕ) : ℕ :=
  let first_term := 1 + sum (range (n-1)) (λ k, 2 * (k + 1)) 
  first_term + sum (range n) (λ k, first_term + 2 * k)

theorem find_a10 (a : ℕ → ℕ) (h1 : a 1 = 1)
  (h2 : a 2 = 3 + 5)
  (h3 : a 3 = 7 + 9 + 11)
  (h4 : a 4 = 13 + 15 + 17 + 19)
  (hs : ∀ n, a n = (1 + sum (range (n-1)) (λ k, 2 * (k + 1))) + sum (range n) (λ k, 1 + 2 * (n-1 + k))) :
  a 10 = 1000 :=
by
  sorry


end find_a10_l281_281064


namespace eval_integral_ln_x_ln_one_minus_x_l281_281769

noncomputable def integral_ln_x_ln_one_minus_x : ℝ :=
  ∫ x in 0..1, (Real.log x) * (Real.log (1 - x))

theorem eval_integral_ln_x_ln_one_minus_x :
  integral_ln_x_ln_one_minus_x = 2 - Real.pi^2 / 6 :=
sorry

end eval_integral_ln_x_ln_one_minus_x_l281_281769


namespace product_log_condition_l281_281959

theorem product_log_condition (a b : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) 
  (h_eq : sqrt (2 * log 10 a) + sqrt (2 * log 10 b) + log 10 (sqrt a) + log 10 (sqrt b) = 104) 
  (h_int : ∃ m n : ℤ, sqrt (2 * log 10 a) = m ∧ sqrt (2 * log 10 b) = n ∧ m > 0 ∧ n > 0) :
  ab = 10 ^ 260 :=
by
  sorry

end product_log_condition_l281_281959


namespace edge_length_of_tetrahedron_l281_281134

theorem edge_length_of_tetrahedron
  (R : ℝ)
  (A B C D : point3d)
  (tetrahedron : tetrahedron ABCD)
  (sphere1 sphere2 : sphere)
  (h1 : sphere1.center = A)
  (h2 : sphere2.center = B)
  (h3 : sphere1.radius = 2 * R)
  (h4 : sphere2.radius = 3 * R)
  (h5 : spheres_tangent_externally sphere1 sphere2) :
  tetrahedron.edge_length ABCD = (5 * sqrt 6 + sqrt 32) * R :=
by
  sorry

end edge_length_of_tetrahedron_l281_281134


namespace maddie_total_cost_l281_281932

theorem maddie_total_cost :
  let price_palette := 15
  let price_lipstick := 2.5
  let price_hair_color := 4
  let num_palettes := 3
  let num_lipsticks := 4
  let num_hair_colors := 3
  let total_cost := (num_palettes * price_palette) + (num_lipsticks * price_lipstick) + (num_hair_colors * price_hair_color)
  total_cost = 67 := by
  sorry

end maddie_total_cost_l281_281932


namespace max_sum_non_zero_nats_l281_281842

theorem max_sum_non_zero_nats (O square : ℕ) (hO : O ≠ 0) (hsquare : square ≠ 0) :
  (O / 11 < 7 / square) ∧ (7 / square < 4 / 5) → O + square = 77 :=
by 
  sorry -- Proof omitted as requested

end max_sum_non_zero_nats_l281_281842


namespace boatworks_total_canoes_l281_281742

theorem boatworks_total_canoes : 
  let jan := 5
  let feb := 3 * jan
  let mar := 3 * feb
  let apr := 3 * mar
  jan + feb + mar + apr = 200 := 
by 
  sorry

end boatworks_total_canoes_l281_281742


namespace parabola_line_no_intersection_l281_281902

noncomputable def a_b_sum : ℝ := 
let discriminant_lt_zero := λ (n : ℝ), n^2 - 40 * n - 24 < 0 in
let roots := {-24, 40} in
roots.1 + roots.2

-- the theorem statement
theorem parabola_line_no_intersection :
  (∃ a b : ℝ, ∀ n : ℝ, (a < n ∧ n < b) ↔ (n^2 - 40 * n - 24 < 0)) → 
  (a_b_sum = 40) :=
sorry

end parabola_line_no_intersection_l281_281902


namespace find_x_of_isosceles_right_triangle_l281_281130

def dist_sq (p q : ℝ × ℝ × ℝ) : ℝ :=
  (q.1 - p.1)^2 + (q.2 - p.2)^2 + (q.3 - p.3)^2

theorem find_x_of_isosceles_right_triangle (x : ℝ) :
    let A := (4, 1, 9)
    let B := (10, -1, 6)
    let C := (x, 4, 3)
    isosceles_right_triangle A B C → x = 2 :=
sorry

-- Define isosceles right triangle property for completeness
def isosceles_right_triangle (A B C : ℝ × ℝ × ℝ) : Prop :=
    dist_sq A B = dist_sq A C 
    ∧ dist_sq B C = dist_sq A B + dist_sq A C

end find_x_of_isosceles_right_triangle_l281_281130


namespace polynomial_expansion_sum_l281_281852

theorem polynomial_expansion_sum :
  ∀ P Q R S : ℕ, ∀ x : ℕ, 
  (P = 4 ∧ Q = 10 ∧ R = 1 ∧ S = 21) → 
  ((x + 3) * (4 * x ^ 2 - 2 * x + 7) = P * x ^ 3 + Q * x ^ 2 + R * x + S) → 
  P + Q + R + S = 36 :=
by
  intros P Q R S x h1 h2
  sorry

end polynomial_expansion_sum_l281_281852


namespace quadratic_roots_identity_l281_281989

theorem quadratic_roots_identity :
  ∀ (x1 x2 : ℝ), (x1^2 - 3 * x1 - 4 = 0) ∧ (x2^2 - 3 * x2 - 4 = 0) →
  (x1^2 - 2 * x1 * x2 + x2^2) = 25 :=
by
  intros x1 x2 h
  sorry

end quadratic_roots_identity_l281_281989


namespace repeated_three_digit_not_divisible_by_101_l281_281724

theorem repeated_three_digit_not_divisible_by_101 (abc : ℕ) (h1: 100 ≤ abc ∧ abc < 1000) :
  let six_digit := abc * 1001 in
  ¬ (six_digit % 101 = 0) :=
by
  sorry

end repeated_three_digit_not_divisible_by_101_l281_281724


namespace cyclic_quadrilateral_l281_281575

variables (A B C D I1 I2 E F P : Type) [incircle : ∀ {α β γ} (h : triangle α β γ), ∃ I, incenter_of α β γ I]

-- Definitions for the conditions given in the problem
noncomputable def convexQuadrilateral (A B C D : Type) : Prop := sorry
noncomputable def incenter (α β γ I : Type) [triangle α β γ] : Prop := sorry
noncomputable def isIncenter (α β γ : Type) [triangle α β γ] (I : Type) : Prop := sorry
noncomputable def intersects (l1 l2 : Type) (P : Type) : Prop := sorry
noncomputable def lineThrough (P Q : Type) (l : Type) : Prop := sorry
noncomputable def pointsConcyclic (A B C D : Type) : Prop := sorry
noncomputable def lengthEqual (P Q R S : Type) : Prop := sorry

-- Conditions
variables (convex : convexQuadrilateral A B C D)
variables (inc1 : isIncenter A B C I1)
variables (inc2 : isIncenter D B C I2)
variables (lineE : lineThrough I1 E)
variables (lineF : lineThrough I2 F)
variables (intersectionP : intersects (lineThrough A B) (lineThrough D C) P)
variables (lengthEquality : lengthEqual P E P F)

-- The theorem statement
theorem cyclic_quadrilateral (h1 : convexQuadrilateral A B C D)
  (h2 : isIncenter A B C I1) 
  (h3 : isIncenter D B C I2) 
  (h4 : lineThrough I1 E)
  (h5 : lineThrough I2 F)
  (h6 : intersects (lineThrough A B) (lineThrough D C) P)
  (h7 : lengthEqual P E P F) :
  pointsConcyclic A B C D :=
sorry

end cyclic_quadrilateral_l281_281575


namespace radius_of_spheres_l281_281973

theorem radius_of_spheres (r : ℝ) : 
  (exists (radius : ℝ),
    radius > 0 ∧ 
    ∀ (spheres : fin 16 → sphere),
      (sphere.center (spheres 0) = (0.5, 0.5, 0.5)) ∧
      ∀ i ≠ 0, 
        sphere.radius (spheres i) = radius ∧
        sphere.tangent_to_center_sphere (spheres i) (spheres 0) ∧
        (sphere.tangent_to_faces (spheres i) = 3))
→ r = 1 / 3 :=
sorry

end radius_of_spheres_l281_281973


namespace problem_statement_l281_281829
open Real

noncomputable def l1 (α : ℝ) : ℝ → ℝ → Prop := fun x y => (sin α) * x - (cos α) * y + 1 = 0
noncomputable def l2 (α : ℝ) : ℝ → ℝ → Prop := fun x y => (sin α) * x + (cos α) * y + 1 = 0
noncomputable def l3 (α : ℝ) : ℝ → ℝ → Prop := fun x y => (cos α) * x - (sin α) * y + 1 = 0
noncomputable def l4 (α : ℝ) : ℝ → ℝ → Prop := fun x y => (cos α) * x + (sin α) * y + 1 = 0

theorem problem_statement:
  (∃ (α : ℝ), ∀ (x y : ℝ), l1 α x y → l2 α x y) ∧
  (∀ (α : ℝ), ∀ (x y : ℝ), l1 α x y → (sin α) * (cos α) + (-cos α) * (sin α) = 0) ∧
  (∃ (p : ℝ × ℝ), ∀ (α : ℝ), abs ((sin α) * p.1 - (cos α) * p.2 + 1) / sqrt ((sin α)^2 + (cos α)^2) = 1 ∧
                        abs ((sin α) * p.1 + (cos α) * p.2 + 1) / sqrt ((sin α)^2 + (cos α)^2) = 1 ∧
                        abs ((cos α) * p.1 - (sin α) * p.2 + 1) / sqrt ((cos α)^2 + (sin α)^2) = 1 ∧
                        abs ((cos α) * p.1 + (sin α) * p.2 + 1) / sqrt ((cos α)^2 + (sin α)^2) = 1) :=
sorry

end problem_statement_l281_281829


namespace correct_operation_l281_281657

theorem correct_operation (a : ℝ) : 
  (a^3 * a^2 ≠ a^6) ∧ 
  ((-4 * a^3)^2 = 16 * a^6) ∧ 
  (a^6 / a^6 ≠ 0) ∧ 
  ((a - 1)^2 ≠ a^2 - 1) := by
  sorry

end correct_operation_l281_281657


namespace quadratic_trinomial_unique_root_l281_281115

theorem quadratic_trinomial_unique_root (a b c : ℝ) :
  let new_trinomial := 2 * a * x^2 + (b + c) * x + (b + c) in
  discriminant new_trinomial = 0 →
  (b + c = 0 ∨ b + c = 8 * a) →
  ((b + c = 0 → ∀ x, new_trinomial = 0) ∧ 
   (b + c = 8 * a → ∀ x, new_trinomial = 2 * a * (x + 2)^2 → x = -2)) :=
by
  sorry

end quadratic_trinomial_unique_root_l281_281115


namespace mutually_exclusive_events_l281_281708

noncomputable def students : ℕ := 5
noncomputable def male_students : ℕ := 3
noncomputable def female_students : ℕ := 2
noncomputable def selected_students : ℕ := 2

def at_least_one_male (selected: list ℕ) : Prop :=
  selected.filter (λ s, s ≤ male_students > 0)

def all_female (selected: list ℕ) : Prop :=
  selected.all (λ s, s > male_students)

theorem mutually_exclusive_events :
  ∀ (selected: list ℕ), 
  (at_least_one_male selected ∧ all_female selected) → false := sorry

end mutually_exclusive_events_l281_281708


namespace sin_cos_identity_l281_281035

theorem sin_cos_identity (x : ℝ) (h : Real.cos x - 5 * Real.sin x = 2) : Real.sin x + 5 * Real.cos x = -28 / 13 := 
  sorry

end sin_cos_identity_l281_281035


namespace sin_cos_identity_l281_281036

theorem sin_cos_identity (x : ℝ) (h : Real.cos x - 5 * Real.sin x = 2) : Real.sin x + 5 * Real.cos x = -28 / 13 := 
  sorry

end sin_cos_identity_l281_281036


namespace arithmetic_to_geometric_seq_l281_281151

-- Condition definitions
def is_arithmetic_seq (a : ℕ → ℤ) (a₁ : ℤ) (d : ℤ) : Prop :=
  ∀ n, a n = a₁ + n * d

def forms_geometric_seq (a : ℕ → ℤ) : Prop :=
  a 1 * a 6 = (a 2)^2

-- Main problem statement
theorem arithmetic_to_geometric_seq (a : ℕ → ℤ)
  (h_arith : is_arithmetic_seq a 1 3)
  (h_geo : forms_geometric_seq a) :
  (∀ n, a n = 3 * n - 2) ∧
  (∀ n, let b := (λ n, (-1)^n * a n) in 
    let S := (λ n, ∑ i in finset.range n, b i) in
    S n = if n % 2 = 0 then (3 * n) / 2 else (1 - 3 * n) / 2) := 
sorry

end arithmetic_to_geometric_seq_l281_281151


namespace woman_work_rate_l281_281682

theorem woman_work_rate (M W : ℝ) (h1 : 10 * M + 15 * W = 1 / 8) (h2 : M = 1 / 100) : W = 1 / 600 :=
by 
  sorry

end woman_work_rate_l281_281682


namespace area_of_triangle_is_8_5_l281_281149

noncomputable def matrix_T : Matrix (Fin 2) (Fin 2) ℝ :=
  ![
    ![1, 1],
    ![0, 1]
  ]

noncomputable def vector_a : Fin 2 → ℝ :=
  ![3, 2]

noncomputable def vector_b : Fin 2 → ℝ :=
  ![-1, 5]

theorem area_of_triangle_is_8_5 :
  let T_a := matrix_T.mulVec vector_a,
      T_b := matrix_T.mulVec vector_b in
  let area_parallel := abs (T_a 0 * T_b 1 - T_b 0 * T_a 1) in
  let area_triangle := abs area_parallel / 2 in
  area_triangle = 8.5 :=
by
  sorry

end area_of_triangle_is_8_5_l281_281149


namespace part_1_part_2_part_3_l281_281056

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x * Real.log x - a * x
noncomputable def g (x : ℝ) : ℝ := -x^2 - 2

theorem part_1 {a : ℝ} : (∀ x > 0, f x a ≥ g x) → a ≤ -2 :=
  sorry

theorem part_2 {m : ℝ} (hm : m > 0) : 
  (∀ x ∈ set.Icc m (m + 3), Continuous (f x (-1))) → 
  ∀ x ∈ set.Icc m (m + 3), x ∈ {m, m+3} :=
  sorry

theorem part_3 : ∀ x > 0, Real.log x + 1 > 1 / Real.exp x - 2 / (Real.exp 1 * x) :=
  sorry

end part_1_part_2_part_3_l281_281056


namespace combined_weight_of_daughter_and_child_l281_281604

variables (M D C : ℝ)
axiom mother_daughter_grandchild_weight : M + D + C = 120
axiom daughter_weight : D = 48
axiom child_weight_fraction_of_grandmother : C = (1 / 5) * M

theorem combined_weight_of_daughter_and_child : D + C = 60 :=
  sorry

end combined_weight_of_daughter_and_child_l281_281604


namespace function_is_even_l281_281597

/-- Definition of the function y -/
def y (x : ℝ) : ℝ := sqrt (1 - x^2) + 9 / (1 + abs x)

/-- Define the domain condition for the given function -/
def domain_condition (x : ℝ) : Prop := (x ∈ Set.Ico (-1 : ℝ) 0) ∨ (x ∈ Set.Ioo 0 1)

/-- Proof that the function y(x) is even under the given domain conditions -/
theorem function_is_even (x : ℝ) (h : domain_condition x) : y (-x) = y x := by
  sorry

end function_is_even_l281_281597


namespace lowest_degree_poly_meets_conditions_l281_281300

-- Define a predicate that checks if a polynomial P meets the conditions
def poly_meets_conditions (P : ℚ[X]) (b : ℚ) : Prop :=
  (∀ x, coeff P x ≠ b) ∧ 
  (∃ x y, coeff P x < b ∧ coeff P y > b)

-- Statement of the theorem we want to prove
theorem lowest_degree_poly_meets_conditions : ∀ (b : ℚ), 
  ∃ (P : ℚ[X]), poly_meets_conditions P b ∧ degree P = 4 :=
begin
  sorry
end

end lowest_degree_poly_meets_conditions_l281_281300


namespace sqrt_nonneg_in_domain_l281_281828

-- Define the function f
def f (x : ℝ) : ℝ := x ^ (1 / 2 : ℝ)

-- State the theorem
theorem sqrt_nonneg_in_domain : ∀ x : ℝ, 0 ≤ x → 0 ≤ f x :=
by
  intro x hx
  show 0 ≤ x^(1/2 : ℝ)
  sorry

end sqrt_nonneg_in_domain_l281_281828


namespace no_points_within_circle_l281_281161

theorem no_points_within_circle (P A B : EuclideanSpace ℝ (Fin 2)) (circle_radius : ℝ := 2) :
  dist P A ^ 2 + dist P B ^ 2 = 20 → P ∉ Metric.Ball (0 : EuclideanSpace ℝ (Fin 2)) circle_radius :=
by
  sorry

end no_points_within_circle_l281_281161


namespace colors_per_person_l281_281502

theorem colors_per_person
    (total_colors : ℕ)
    (num_people : ℕ)
    (total_colors_eq : total_colors = 24)
    (num_people_eq : num_people = 3) :
    total_colors / num_people = 8 :=
by
  rw [total_colors_eq, num_people_eq]
  exact nat.div_eq_of_eq_mul_left (by norm_num : 0 < 3) (by norm_num : 24 = 3 * 8)
  -- sorry needed to skip the detailed proof steps

end colors_per_person_l281_281502


namespace election_winner_won_by_votes_l281_281868

theorem election_winner_won_by_votes (V : ℝ) (winner_votes : ℝ) (loser_votes : ℝ)
    (h1 : winner_votes = 0.62 * V)
    (h2 : winner_votes = 930)
    (h3 : loser_votes = 0.38 * V)
    : winner_votes - loser_votes = 360 := 
  sorry

end election_winner_won_by_votes_l281_281868


namespace option_B_correct_option_A_incorrect_option_C_incorrect_option_D_incorrect_l281_281654

theorem option_B_correct : (-4: ℤ)^2 = -16 :=
by
  calc
    (-4: ℤ)^2 = -(4 * 4) := by sorry -- Correct option

theorem option_A_incorrect : (2: ℤ)^4 ≠ 8 :=
by
  calc
    (2: ℤ)^4 = 16 := by sorry -- Incorrect option

theorem option_C_incorrect : (-8: ℤ) - 8 ≠ 0 :=
by
  calc
    (-8: ℤ) - 8 = -16 := by sorry -- Incorrect option

theorem option_D_incorrect : (-3: ℤ)^2 ≠ 6 :=
by
  calc
    (-3: ℤ)^2 = 9 := by sorry -- Incorrect option

end option_B_correct_option_A_incorrect_option_C_incorrect_option_D_incorrect_l281_281654


namespace equal_focal_distances_l281_281590

def ellipse1 (x y : ℝ) : Prop := x^2 / 25 + y^2 / 9 = 1
def ellipse2 (k x y : ℝ) (hk : k < 9) : Prop := x^2 / (25 - k) + y^2 / (9 - k) = 1

theorem equal_focal_distances (k : ℝ) (hk : k < 9) : 
  let f1 := 8
  let f2 := 8 
  f1 = f2 :=
by 
  sorry

end equal_focal_distances_l281_281590


namespace variance_exponential_distribution_std_dev_exponential_distribution_l281_281785

-- Definition of the exponential PDF
def exponential_pdf (λ : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then 0 else λ * Real.exp (-λ * x)

-- Statement for the variance
theorem variance_exponential_distribution (λ : ℝ) (hλ : 0 < λ) :
  ∫ (x : ℝ) (h : 0 ≤ x), x^2 * exponential_pdf λ x = (2 / λ) ∧ 
  (1 / λ - (1 / λ)) ^ 2 = 1 / (λ^2) := 
sorry

-- Statement for the standard deviation
theorem std_dev_exponential_distribution (λ : ℝ) (hλ : 0 < λ) (hdrv : (1 / λ^2) = 1 / λ^2) : 
  Real.sqrt (1 / λ^2) = 1 / λ :=
sorry

end variance_exponential_distribution_std_dev_exponential_distribution_l281_281785


namespace max_table_height_exists_l281_281132

theorem max_table_height_exists 
(triangle : Triangle)
(a b c : ℝ)
(h_par1 : segment a parallel_to segment b)
(h_par2 : segment a parallel_to segment c)
(h_par3 : segment b parallel_to segment c)
(h_area : ∃ area : ℝ, area_of_triangle triangle = area)
(h_abc : (triangle.side_a = 25) ∧ (triangle.side_b = 28) ∧ (triangle.side_c = 31)) :
∃ (k m n : ℕ), relatively_prime k n ∧ m_not_divisible_by_square_of_primes m ∧ maximum_table_height h = (42 * sqrt(2582)) / 28 ∧ 
compute_sum_of_k_m_n k m n = 2652 :=
sorry

end max_table_height_exists_l281_281132


namespace elvis_recording_time_l281_281764

theorem elvis_recording_time :
  ∀ (total_studio_time writing_time_per_song editing_time number_of_songs : ℕ),
  total_studio_time = 300 →
  writing_time_per_song = 15 →
  editing_time = 30 →
  number_of_songs = 10 →
  (total_studio_time - (number_of_songs * writing_time_per_song + editing_time)) / number_of_songs = 12 :=
by
  intros total_studio_time writing_time_per_song editing_time number_of_songs
  intros h1 h2 h3 h4
  sorry

end elvis_recording_time_l281_281764


namespace package_contains_12_rolls_l281_281694

-- Define the problem conditions
def package_cost : ℝ := 9
def individual_roll_cost : ℝ := 1
def savings_percent : ℝ := 0.25

-- Define the price per roll in the package
def price_per_roll_in_package := individual_roll_cost * (1 - savings_percent)

-- Define the number of rolls in the package
def number_of_rolls_in_package := package_cost / price_per_roll_in_package

-- Question: How many rolls are in the package?
theorem package_contains_12_rolls : number_of_rolls_in_package = 12 := by
  sorry

end package_contains_12_rolls_l281_281694


namespace nth_term_arithmetic_sequence_l281_281220

theorem nth_term_arithmetic_sequence (p q : ℤ) (h1 : 9 = p + 2 * q) (h2 : 3 * p - q = 9 + 2 * q) : 
   let d := 2 * q in
   let a_2010 := p + (2010 - 1) * d in
   a_2010 = 8041 :=
by
  sorry

end nth_term_arithmetic_sequence_l281_281220


namespace sum_of_palindromic_primes_lt_200_l281_281948

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_palindromic (n : ℕ) : Prop :=
  let str_n := n.toString
  str_n = str_n.reverse

def is_palindromic_prime (n : ℕ) : Prop :=
  is_prime n ∧ is_palindromic n ∧ is_prime (n.toString.reverse.toNat)

def palindromic_primes_less_than_200 : List ℕ :=
  [101, 113, 131, 151, 181, 191]

theorem sum_of_palindromic_primes_lt_200 : (palindromic_primes_less_than_200.foldl (· + ·) 0) = 868 := by
  sorry

end sum_of_palindromic_primes_lt_200_l281_281948


namespace subset_singleton_zero_l281_281104

def A := {x : ℝ | x > -3}

theorem subset_singleton_zero (A : set ℝ) (h : A = {x | x > -3}) : {0} ⊆ A :=
by
  rw h
  sorry

end subset_singleton_zero_l281_281104


namespace sequence_property_and_sum_l281_281802

-- Define the sequence a_n as an increasing geometric sequence
def geom_sequence (a : ℕ → ℕ) (q : ℕ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

-- Problem conditions
def conditions (a : ℕ → ℕ) (q a2 a4 a3 : ℕ) : Prop :=
  q > 1 ∧ a 2 + a 4 = 20 ∧ a 3 = 8

-- General formula for the sequence a_n
def general_formula (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a n = 2 ^ n

-- Define b_m based on terms of the sequence a_n in the interval (0, m]
def b_m (a : ℕ → ℕ) (m : ℕ) : ℕ :=
  {n : ℕ // 2 ^ n ≤ m}.card

-- Sum of the first 100 terms of the sequence b_m, denoted as S_100
def S_100 (b : ℕ → ℕ) : ℕ :=
  (Finset.range 100).sum b

theorem sequence_property_and_sum :
  ∃ a : ℕ → ℕ, ∃ q a2 a4 a3 : ℕ, conditions a q a2 a4 a3 →
  general_formula a ∧ S_100 (b_m a) = 480 := sorry

end sequence_property_and_sum_l281_281802


namespace find_m_plus_n_l281_281899

theorem find_m_plus_n :
  let r : Fin 20 → ℂ := fun i => complex_root_of_polynomial x (x^(20) - 7 * x^3 + 1)
  let sum := ∑ i in Finset.range 20, 1 / (r i)^2 + 1
  ∃ (m n : ℕ), Nat.coprime m n ∧ sum = m / n ∧ m + n = 240 := 
begin
  sorry
end

end find_m_plus_n_l281_281899


namespace G_is_3F_l281_281900

noncomputable def F (x : ℝ) : ℝ :=
  log ((1 + x) / (1 - x))

noncomputable def G (x : ℝ) : ℝ :=
  log ((1 + (3 * x - x ^ 3) / (1 + 3 * x ^ 2)) / (1 - (3 * x - x ^ 3) / (1 + 3 * x ^ 2)))

theorem G_is_3F (x : ℝ) : G x = 3 * F x := by
  sorry

end G_is_3F_l281_281900


namespace number_of_students_to_bring_donuts_l281_281008

theorem number_of_students_to_bring_donuts (students_brownies students_cookies students_donuts : ℕ) :
  (students_brownies * 12 * 2) + (students_cookies * 24 * 2) + (students_donuts * 12 * 2) = 2040 →
  students_brownies = 30 →
  students_cookies = 20 →
  students_donuts = 15 :=
by
  -- Proof skipped
  sorry

end number_of_students_to_bring_donuts_l281_281008


namespace part1_part2_part3_l281_281465

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (1 / x + a) * Real.log (1 + x)

-- Part (1) proof problem statement
theorem part1 (a : ℝ) (h : a = -1) : 
  tangent_eq :=
let f_x := f x a in
let df_x := diff f_x in
tangent_eq (df_x) = -Real.log(2) * (x - 1) in
sorry

-- Part (2) proof problem statement
theorem part2 : ∃ (a b : ℝ), a = 1/2 ∧ b = -1/2 ∧ symmetric_about (λ x, f (1 / x) a) b :=
sorry

-- Part (3) proof problem statement
theorem part3 (a : ℝ) : 
  extreme_val_range (exists_has_extreme_values f (0, ∞)) = Ioo 0 (1/2) :=
sorry

end part1_part2_part3_l281_281465


namespace eggs_left_for_sunny_side_up_l281_281631

-- Given conditions:
def ordered_dozen_eggs : ℕ := 3 * 12
def eggs_used_for_crepes (total_eggs : ℕ) : ℕ := total_eggs * 1 / 4
def eggs_after_crepes (total_eggs : ℕ) (used_for_crepes : ℕ) : ℕ := total_eggs - used_for_crepes
def eggs_used_for_cupcakes (remaining_eggs : ℕ) : ℕ := remaining_eggs * 2 / 3
def eggs_left (remaining_eggs : ℕ) (used_for_cupcakes : ℕ) : ℕ := remaining_eggs - used_for_cupcakes

-- Proposition:
theorem eggs_left_for_sunny_side_up : 
  eggs_left (eggs_after_crepes ordered_dozen_eggs (eggs_used_for_crepes ordered_dozen_eggs)) 
            (eggs_used_for_cupcakes (eggs_after_crepes ordered_dozen_eggs (eggs_used_for_crepes ordered_dozen_eggs))) = 9 :=
sorry

end eggs_left_for_sunny_side_up_l281_281631


namespace TimSpentTotal_l281_281851

variable (LunchCost : ℝ) (TipPercentage : ℝ)

def TotalAmountSpent (LunchCost : ℝ) (TipPercentage : ℝ) : ℝ := 
  LunchCost + (LunchCost * TipPercentage)

theorem TimSpentTotal (h1 : LunchCost = 50.50) (h2 : TipPercentage = 0.20) :
  TotalAmountSpent LunchCost TipPercentage = 60.60 := by
  sorry

end TimSpentTotal_l281_281851


namespace height_of_highest_wave_l281_281366

theorem height_of_highest_wave 
  (h_austin : ℝ) -- Austin's height
  (h_high : ℝ) -- Highest wave's height
  (h_short : ℝ) -- Shortest wave's height 
  (height_relation1 : h_high = 4 * h_austin + 2)
  (height_relation2 : h_short = h_austin + 4)
  (surfboard : ℝ) (surfboard_len : surfboard = 7)
  (short_wave_len : h_short = surfboard + 3) :
  h_high = 26 :=
by
  -- Define local variables with the values from given conditions
  let austin_height := 6        -- as per calculation: 10 - 4 = 6
  let highest_wave_height := 26 -- as per calculation: (6 * 4) + 2 = 26
  sorry

end height_of_highest_wave_l281_281366


namespace count_distinct_convex_pentagons_l281_281478

def angle_of_pentagon : ℝ := 108
def side_lengths := {1, 2, 3}

theorem count_distinct_convex_pentagons :
  (∀ (P : Type) [decidable_eq P] [fintype P],
     ∀ A B C D E : P, 
       angle_of_pentagon = 108 ∧ {1, 2, 3} ⊆ {side_length AB, side_length BC, side_length CD, side_length DE, side_length EA} → 
       distinct_convex_pentagons ⟹ 5) :=
sorry

end count_distinct_convex_pentagons_l281_281478


namespace hens_egg_laying_l281_281893

theorem hens_egg_laying :
  ∀ (hens: ℕ) (price_per_dozen: ℝ) (total_revenue: ℝ) (weeks: ℕ) (total_hens: ℕ),
  hens = 10 →
  price_per_dozen = 3 →
  total_revenue = 120 →
  weeks = 4 →
  total_hens = hens →
  (total_revenue / price_per_dozen / 12) * 12 = 480 →
  (480 / weeks) = 120 →
  (120 / hens) = 12 :=
by sorry

end hens_egg_laying_l281_281893


namespace value_of_X_l281_281094

theorem value_of_X : 
  let M := 2098 / 2 in
  let N := M * 2 in
  let X := M + N in
  X = 3147 :=
by
  sorry

end value_of_X_l281_281094


namespace vertex_coordinates_l281_281216

theorem vertex_coordinates (a h k : ℝ) :
  (∀ x : ℝ, 2 * (x - 3)^2 + 1 = a * (x - h)^2 + k) → (h, k) = (3, 1) :=
by
  intro h k
  sorry

end vertex_coordinates_l281_281216


namespace ratio_of_distances_l281_281633

theorem ratio_of_distances 
  (x : ℝ) -- distance walked by the first lady
  (h1 : 4 + x = 12) -- combined total distance walked is 12 miles 
  (h2 : ¬(x < 0)) -- distance cannot be negative
  (h3 : 4 ≠ 0) : -- the second lady walked 4 miles which is not zero
  x / 4 = 2 := -- the ratio of the distances is 2
by
  sorry

end ratio_of_distances_l281_281633


namespace length_of_EF_l281_281123

theorem length_of_EF {A B C D E F : Type} [Points A B C D E F] 
  (h_rect : is_rectangle A B C D)
  (h_AB : dist A B = 3)
  (h_BC : dist B C = 9)
  (h_folded : coincide A C)
  (h_pentagon : is_pentagon A B E F D) :
  dist E F = Real.sqrt 10 :=
sorry

end length_of_EF_l281_281123


namespace rainy_days_last_week_l281_281670

theorem rainy_days_last_week (n : ℤ) (R NR : ℕ) (h1 : n * R + 3 * NR = 20)
  (h2 : 3 * NR = n * R + 10) (h3 : R + NR = 7) : R = 2 :=
sorry

end rainy_days_last_week_l281_281670


namespace sunflower_taller_than_rose_bush_l281_281733

def mixed_number_to_improper_fraction (a b : Int) (c : Nat) : Rational :=
  (a * c + b).nat_div c

def rose_bush_height : Rational := mixed_number_to_improper_fraction 5 4 5
def sunflower_height : Rational := mixed_number_to_improper_fraction 9 3 5

def height_difference (a b : Rational) : Rational :=
  a - b

def improper_fraction_to_mixed_number (a b : Nat) : String :=
  let q := a / b
  let r := a % b
  if r = 0 then q.repr else q.repr ++ " " ++ (r.repr ++ "/" ++ b.repr)

theorem sunflower_taller_than_rose_bush : improper_fraction_to_mixed_number (height_difference sunflower_height rose_bush_height).num (height_difference sunflower_height rose_bush_height).den = "3 4/5" :=
  sorry

end sunflower_taller_than_rose_bush_l281_281733


namespace quadrilateral_is_right_trapezoid_l281_281196

structure Point (ℝ : Type*) := (x y : ℝ)

def A : Point ℝ := ⟨1, 2⟩
def B : Point ℝ := ⟨3, 6⟩
def C : Point ℝ := ⟨0, 5⟩
def D : Point ℝ := ⟨-1, 3⟩

def vector (p1 p2 : Point ℝ) : Point ℝ :=
  ⟨p2.x - p1.x, p2.y - p1.y⟩

def dot_product (v1 v2 : Point ℝ) : ℝ :=
  v1.x * v2.x + v1.y * v2.y

def parallel (v1 v2 : Point ℝ) : Prop :=
  v1.x * v2.y = v1.y * v2.x

def is_right_angle (v1 v2 : Point ℝ) : Prop :=
  dot_product v1 v2 = 0

def is_right_trapezoid (A B C D : Point ℝ) : Prop :=
  let AB := vector A B
  let AD := vector A D
  let DC := vector D C
  parallel AB DC ∧ is_right_angle AB AD

theorem quadrilateral_is_right_trapezoid : is_right_trapezoid A B C D := by
  sorry

end quadrilateral_is_right_trapezoid_l281_281196


namespace jenga_blocks_before_jess_turn_l281_281137

theorem jenga_blocks_before_jess_turn
  (total_blocks : ℕ := 54)
  (players : ℕ := 5)
  (rounds : ℕ := 5)
  (father_turn_blocks : ℕ := 1)
  (original_blocks := total_blocks - (players * rounds + father_turn_blocks))
  (jess_turn_blocks : ℕ := total_blocks - original_blocks):
  jess_turn_blocks = 28 := by
begin
  sorry
end

end jenga_blocks_before_jess_turn_l281_281137


namespace sum_of_triangle_areas_in_cube_l281_281613

theorem sum_of_triangle_areas_in_cube (m n p : ℕ) 
  (h_cube : ∀ V T,
    (V ⊆ {p : ℝ × ℝ × ℝ | p.1 = 0 ∨ p.1 = 2 ∨ p.2 = 0 ∨ p.2 = 2 ∨ p.3 = 0 ∨ p.3 = 2})
    ∧ (∀ t ∈ T, ∃ x₁ y₁ z₁ x₂ y₂ z₂ x₃ y₃ z₃, t = triangle (x₁, y₁, z₁) (x₂, y₂, z₂) (x₃, y₃, z₃))
    ∧ (∀ t ∈ T, area t = 48 + sqrt(2304) + sqrt(3072))) :
  m + n + p = 5424 :=
begin
  sorry
end

end sum_of_triangle_areas_in_cube_l281_281613


namespace square_diff_l281_281075

theorem square_diff (x y : ℝ) (h1 : (x + y)^2 = 81) (h2 : x * y = 18) : (x - y)^2 = 9 :=
by 
  sorry

end square_diff_l281_281075


namespace marble_ratio_l281_281754

-- Definitions based on conditions
def dan_marbles : ℕ := 5
def mary_marbles : ℕ := 10

-- Statement of the theorem to prove the ratio is 2:1
theorem marble_ratio : mary_marbles / dan_marbles = 2 := by
  sorry

end marble_ratio_l281_281754


namespace circumcircle_incenter_midpoints_problem_l281_281738

noncomputable theory

open EuclideanGeometry

/-- Let \( M \) and \( N \) be the midpoints of the minor arcs \( \overparen{AC} \) and \( \overparen{AB} \) of the circumcircle of the acute triangle \( \Delta ABC \). A line parallel to \( MN \) is drawn through point \( A \) and intersects the circumcircle at point \( A' \). Let \( I \) be the incenter of \( \Delta ABC \), and the extension of \( A'I \) intersects the circumcircle at point \( P \). Let the incenters of \( \Delta GAB \) and \( \Delta GAC \) be \( I_1 \) and \( I_2 \) respectively. We need to prove that:

1. \( MA' \cdot MP = NA' \cdot NP \)
2. Points \( P, G, I_1, and I_2 \) are concyclic.
-/
theorem circumcircle_incenter_midpoints_problem 
    {A B C M N A' P I G I₁ I₂: Point} (h1 : IsMidpoint M A C)
    (h2 : IsMidpoint N A B)
    (h3 : ParallelogramParallel A' MN A)
    (h4 : IsIncenter I A B C) 
    (h5 : LineExtendIntersect A' I Circumcircle P)
    (h6 : IsIncenter I₁ G A B)
    (h7 : IsIncenter I₂ G A C) :
    (MA' * MP = NA' * NP) ∧ Concyclic {P, G, I₁, I₂} := sorry


end circumcircle_incenter_midpoints_problem_l281_281738


namespace unique_hyperbolas_count_l281_281786

theorem unique_hyperbolas_count : 
  (Finset.card ((Finset.filter (fun b : ℕ => b > 1)
  (Finset.image (fun ⟨m, n⟩ => Nat.choose m n)
  ((Finset.Icc 1 5).product (Finset.Icc 1 5)))).toFinset)) = 6 := 
by
  sorry  

end unique_hyperbolas_count_l281_281786


namespace probability_bc_seated_next_l281_281360

theorem probability_bc_seated_next {P : ℝ} : 
  P = 2 / 3 :=
sorry

end probability_bc_seated_next_l281_281360


namespace circle_passing_through_and_tangent_l281_281700

noncomputable def circle_equation (x y : ℝ) : Prop :=
  (x - (-1)) ^ 2 + (y - 1) ^ 2 = 2

theorem circle_passing_through_and_tangent :
  ∃ Cx Cy r, (Cx + 1/2) ^ 2 + (Cy - 1/2) ^ 2 = 1/2 ∧
  (0 - 0) ^ 2 + (1 - Cy) ^ 2 = r ^ 2 ∧
  (Cx ^ 2 + Cy ^ 2 + 2 * Cx - 2 * Cy = 0 ∧ (Cx, Cy) = (-1/2, 1/2)) :=
begin
  use [-1/2, 1/2, √(1/2)],
  transitivity,
    rw [circle_equation],
    ring,
    linarith,
end

end circle_passing_through_and_tangent_l281_281700


namespace hcf_two_numbers_l281_281600

theorem hcf_two_numbers (H a b : ℕ) (coprime_ab : Nat.gcd a b = 1) 
    (lcm_factors : a * b = 150) (larger_num : H * a = 450 ∨ H * b = 450) : H = 30 := 
by
  sorry

end hcf_two_numbers_l281_281600


namespace eight_numbers_difference_divisible_by_seven_l281_281961

theorem eight_numbers_difference_divisible_by_seven (a b c d e f g h : ℕ) :
  ∃ x y, (x ∈ {a, b, c, d, e, f, g, h}) ∧ (y ∈ {a, b, c, d, e, f, g, h}) ∧ (x ≠ y) ∧ (∃ k, (x - y) = 7 * k) :=
by
  sorry

end eight_numbers_difference_divisible_by_seven_l281_281961


namespace students_with_uncool_parents_but_cool_siblings_l281_281112

-- The total number of students in the classroom
def total_students : ℕ := 40

-- The number of students with cool dads
def students_with_cool_dads : ℕ := 18

-- The number of students with cool moms
def students_with_cool_moms : ℕ := 22

-- The number of students with both cool dads and cool moms
def students_with_both_cool_parents : ℕ := 10

-- The number of students with cool siblings
def students_with_cool_siblings : ℕ := 8

-- The theorem we want to prove
theorem students_with_uncool_parents_but_cool_siblings
  (h1 : total_students = 40)
  (h2 : students_with_cool_dads = 18)
  (h3 : students_with_cool_moms = 22)
  (h4 : students_with_both_cool_parents = 10)
  (h5 : students_with_cool_siblings = 8) :
  8 = (students_with_cool_siblings) :=
sorry

end students_with_uncool_parents_but_cool_siblings_l281_281112


namespace incorrect_statement_C_l281_281798

-- Define the objects in the problem
variables (a b : Type) (α : Type)

-- Preconditions and properties
-- These assumptions encode the relationships between lines and planes
def perp_to_plane (l : Type) (p : Type) : Prop := sorry -- define perpendicularity
def subset_of_plane (l : Type) (p : Type) : Prop := sorry -- define subset
def parallel_to_line (l1 l2 : Type) : Prop := sorry -- define parallel lines
def parallel_to_plane (l : Type) (p : Type) : Prop := sorry -- define parallel to plane

-- Statements
def statement_A : Prop := perp_to_plane a α ∧ subset_of_plane b α → perp_to_plane a b
def statement_B : Prop := parallel_to_line a b ∧ perp_to_plane a α → perp_to_plane b α
def statement_C : Prop := parallel_to_plane a α ∧ subset_of_plane b α → parallel_to_line a b
def statement_D : Prop := perp_to_plane a b ∧ perp_to_plane b α → parallel_to_plane a α ∨ subset_of_plane a α

-- The theorem to prove
theorem incorrect_statement_C : ¬ statement_C :=
sorry

end incorrect_statement_C_l281_281798


namespace find_x_l281_281044

theorem find_x
    (x : ℝ)
    (l : ℝ := 4 * x)
    (w : ℝ := x + 8)
    (area_eq_twice_perimeter : l * w = 2 * (2 * l + 2 * w)) :
    x = 2 :=
by
  sorry

end find_x_l281_281044


namespace temperature_below_75_l281_281740

theorem temperature_below_75
  (T : ℝ)
  (H1 : ∀ T, T ≥ 75 → swimming_area_open)
  (H2 : ¬swimming_area_open) : 
  T < 75 :=
sorry

end temperature_below_75_l281_281740


namespace james_ali_difference_l281_281884

theorem james_ali_difference (J A T : ℝ) (h1 : J = 145) (h2 : T = 250) (h3 : J + A = T) :
  J - A = 40 :=
by
  sorry

end james_ali_difference_l281_281884


namespace sin_theta_plus_2cos_theta_eq_zero_l281_281040

theorem sin_theta_plus_2cos_theta_eq_zero (θ : ℝ) (h : Real.sin θ + 2 * Real.cos θ = 0) :
  (1 + Real.sin (2 * θ)) / (Real.cos θ)^2 = 1 :=
  sorry

end sin_theta_plus_2cos_theta_eq_zero_l281_281040


namespace lowest_degree_is_4_l281_281275

noncomputable def lowest_degree_polynomial (P : Polynomial ℤ) (b : ℤ) : Prop :=
  ∃ (b : ℤ), 
    let A_P := P.support in
    (∀ (a ∈ A_P), a < b ∨ a > b) ∧ 
    (¬(b ∈ A_P)) ∧
    (∃ (a1 a2 : ℤ), a1 ∈ A_P ∧ a2 ∈ A_P ∧ a1 < b ∧ a2 > b)

theorem lowest_degree_is_4 :
  ∀ P : Polynomial ℤ, 
    let b := lowest_degree_polynomial P in
    b P 4 :=
sorry

end lowest_degree_is_4_l281_281275


namespace parallel_vectors_t_eq_neg1_l281_281049

theorem parallel_vectors_t_eq_neg1 (t : ℝ) :
  let a := (1, -1)
  let b := (t, 1)
  (a.1 + b.1, a.2 + b.2) = (k * (a.1 - b.1), k * (a.2 - b.2)) -> t = -1 :=
by
  sorry

end parallel_vectors_t_eq_neg1_l281_281049


namespace fraction_paint_used_second_week_l281_281141

noncomputable def total_paint : ℕ := 360
noncomputable def paint_used_first_week : ℕ := total_paint / 4
noncomputable def remaining_paint_after_first_week : ℕ := total_paint - paint_used_first_week
noncomputable def total_paint_used : ℕ := 135
noncomputable def paint_used_second_week : ℕ := total_paint_used - paint_used_first_week
noncomputable def remaining_paint_after_first_week_fraction : ℚ := paint_used_second_week / remaining_paint_after_first_week

theorem fraction_paint_used_second_week : remaining_paint_after_first_week_fraction = 1 / 6 := by
  sorry

end fraction_paint_used_second_week_l281_281141


namespace real_solutions_of_equation_l281_281759

def f (x : ℝ) : ℝ :=
  (∑ n in (Finset.range 100).map (λ n, n + 1),
    (n : ℝ) * (x + 1) / (x ^ 2 - n ^ 2))

theorem real_solutions_of_equation : 
  ∀ (x : ℝ), f x = x + 1 → (Finset.Icc (-100) 100).card - 1 := sorry

end real_solutions_of_equation_l281_281759


namespace no_base_for_172_in_four_digits_with_odd_last_digit_l281_281423

theorem no_base_for_172_in_four_digits_with_odd_last_digit :
  ¬ (∃ b : ℕ, 4 ≤ digits_count 172 b ∧ digits_count 172 b ≤ 4 ∧ odd (172 % b)) :=
by
  -- assumes that digits_count 172 b computes the number of digits of 172 in base b
  -- and 172 % b gives the last digit of 172 in base b
  sorry

end no_base_for_172_in_four_digits_with_odd_last_digit_l281_281423


namespace sufficient_but_not_necessary_condition_l281_281224

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (x^2 - 2*x < 0) → (|x - 2| < 2) ∧ ¬(|x - 2| < 2) → (x^2 - 2*x < 0 ↔ |x-2| < 2) :=
sorry

end sufficient_but_not_necessary_condition_l281_281224


namespace r_divides_k_l281_281898

theorem r_divides_k (p : ℕ) (k r : ℕ) (hp : p.prime) (hk_pos : 0 < k) (hr_pos : 0 < r) (hp_gr : p > r) (div_cond : (p * k + r) ∣ (p^p + 1)) : r ∣ k :=
sorry

end r_divides_k_l281_281898


namespace lowest_degree_poly_meets_conditions_l281_281299

-- Define a predicate that checks if a polynomial P meets the conditions
def poly_meets_conditions (P : ℚ[X]) (b : ℚ) : Prop :=
  (∀ x, coeff P x ≠ b) ∧ 
  (∃ x y, coeff P x < b ∧ coeff P y > b)

-- Statement of the theorem we want to prove
theorem lowest_degree_poly_meets_conditions : ∀ (b : ℚ), 
  ∃ (P : ℚ[X]), poly_meets_conditions P b ∧ degree P = 4 :=
begin
  sorry
end

end lowest_degree_poly_meets_conditions_l281_281299


namespace sqrt_5_gt_2_l281_281748

theorem sqrt_5_gt_2 : Real.sqrt 5 > 2 :=
by
  have h1 : 2 = Real.sqrt 4 := by norm_num
  have h2 : Real.sqrt 5 > Real.sqrt 4 := by
    exact Real.sqrt_lt_sqrt (show 4 < 5 by norm_num) -- leveraging the fact that sqrt is strictly increasing for positive reals
  rw [h1] at h2
  exact h2

end sqrt_5_gt_2_l281_281748


namespace courier_total_distance_412_l281_281327

variable (D : ℝ)
variable (x : ℝ)
variable (h1 : 2 * x + 3 * x = D)
variable (h2 : (2 * x + 60) / (3 * x - 60) = 6 / 5)

theorem courier_total_distance_412.5 (h1 : 2 * x + 3 * x = D) (h2 : (2 * x + 60) / (3 * x - 60) = 6 / 5) :
  D = 412.5 := by
  sorry

end courier_total_distance_412_l281_281327


namespace vector_addition_l281_281068

def vector_a : ℝ × ℝ × ℝ := (3, -2, 1)
def vector_b : ℝ × ℝ × ℝ := (-1, 1, 4)

theorem vector_addition : 
  let a := vector_a
  let b := vector_b
  (a.1 + b.1, a.2 + b.2, a.3 + b.3) = (2, -1, 5) :=
by
  sorry

end vector_addition_l281_281068


namespace find_a_l281_281801

-- Let's define x_set as a set of real numbers
variables (x1 x2 x3 x4 x5 : ℝ)
-- Define variance of x_set
def variance (x1 x2 x3 x4 x5 : ℝ) : ℝ := sorry
-- Given condition, the variance is 2
axiom variance_condition : variance x1 x2 x3 x4 x5 = 2

-- Define the standard deviation
def stddev (xs : list ℝ) : ℝ := sorry
-- Given condition, the standard deviation of a * x_set is 2√2
axiom stddev_condition (a : ℝ) (ha : 0 < a) : 
  stddev [a * x1, a * x2, a * x3, a * x4, a * x5] = 2 * real.sqrt 2

-- The goal is to prove that a = 2
theorem find_a (a : ℝ) (ha : 0 < a) :
  stddev [a * x1, a * x2, a * x3, a * x4, a * x5] = 2 * real.sqrt 2 
  → variance x1 x2 x3 x4 x5 = 2 
  → a = 2 :=
by
  intros h_stddev h_variance
  sorry

end find_a_l281_281801


namespace problem_statement_l281_281010

-- Definition of the sum of digits
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Let S(n) = sum of digits of n
def S (n : ℕ) := sum_of_digits n

-- Problem statement: Prove there are exactly 4 values of n such that n + S(n) + S(S(n)) = 2017
theorem problem_statement :
  (finset.range 2018).filter (λ n, n + S n + S (S n) = 2017).card = 4 :=
sorry

end problem_statement_l281_281010


namespace range_of_a_l281_281033

variables (m a x y : ℝ)

def p (m a : ℝ) : Prop := m^2 + 12 * a^2 < 7 * a * m ∧ a > 0

def ellipse (m x y : ℝ) : Prop := (x^2)/(m-1) + (y^2)/(2-m) = 1

def q (m : ℝ) (x y : ℝ) : Prop := ellipse m x y ∧ 1 < m ∧ m < 3/2

theorem range_of_a :
  (∃ m, p m a → (∀ x y, q m x y)) → (1/3 ≤ a ∧ a ≤ 3/8) :=
sorry

end range_of_a_l281_281033


namespace average_of_six_consecutive_integers_l281_281583

theorem average_of_six_consecutive_integers (c : ℤ) (h : 0 < c) :
  let d := (c + (c + 1) + (c + 2) + (c + 3) + (c + 4) + (c + 5)) / 6 in
  (d - 1 + d + (d + 1) + (d + 2) + (d + 3) + (d + 4)) / 6 = c + 4 :=
by
  sorry

end average_of_six_consecutive_integers_l281_281583


namespace min_magnitude_value_l281_281015

noncomputable def a (t : ℝ) : ℝ × ℝ × ℝ := (1 - t, 1 - t, t)
noncomputable def b (t : ℝ) : ℝ × ℝ × ℝ := (2, t, t)

def magnitude (v : ℝ × ℝ × ℝ) : ℝ := 
  real.sqrt (v.1^2 + v.2^2 + v.3^2)

def vector_sub (v1 v2 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (v1.1 - v2.1, v1.2 - v2.2, v1.3 - v2.3)

theorem min_magnitude_value : 
  ∃ t : ℝ, magnitude (vector_sub (b t) (a t)) = (3 * real.sqrt 5) / 5 :=
by
  sorry

end min_magnitude_value_l281_281015


namespace lowest_degree_is_4_l281_281283

noncomputable def lowest_degree_polynomial (P : ℝ → ℝ) : ℕ :=
  if ∃ b : ℤ, (∀ coeff ∈ (P.coefficients), coeff < (b : ℝ) ∨ coeff > (b : ℝ)) ∧ (¬ ∃ coeff ∈ (P.coefficients), coeff = (b : ℝ))
  then Polynomial.natDegree P
  else 0

theorem lowest_degree_is_4 : ∀ (P : Polynomial ℝ), 
  (∃ b : ℤ, (∀ coeff ∈ P.coefficients, coeff < (b : ℝ) ∨ coeff > (b : ℝ)) ∧ (¬ ∃ coeff ∈ P.coefficients, coeff = (b : ℝ)))
  → lowest_degree_polynomial P = 4 :=
by
  sorry

end lowest_degree_is_4_l281_281283


namespace hyperbola_foci_asymptotes_distance_l281_281217

theorem hyperbola_foci_asymptotes_distance :
  let h := λ x y : ℝ, x^2 / 4 - y^2 / 3 = 1
      foci := (λ x : ℝ, (x, 0)) '' (set.Ioo (real.sqrt 7) (- real.sqrt 7))
      asymptotes := { p : ℝ × ℝ | (real.sqrt 3) * p.1 ± 2 * p.2 = 0 }
  in ∀ p ∈ foci, ∀ a ∈ asymptotes, (real.abs ((real.sqrt 3) * p.1)) / (real.sqrt 3 + 4) = real.sqrt 3 :=
begin
  sorry
end

end hyperbola_foci_asymptotes_distance_l281_281217


namespace lowest_degree_required_l281_281272

noncomputable def smallest_degree_poly (b : ℤ) : ℕ :=
  if h : ∃ P : Polynomial ℝ, (∀ x, (P.eval x ≠ b)) ∧
    (∃ y, (P.eval y > b)) ∧ (∃ z, (P.eval z < b)) 
  then Nat.find h 
  else 0

theorem lowest_degree_required :
  ∃ b : ℤ, smallest_degree_poly b = 4 :=
by
  -- b is some integer that fits the described conditions
  use 0
  sorry

end lowest_degree_required_l281_281272


namespace infinite_sum_equals_two_l281_281378

theorem infinite_sum_equals_two :
  ∑' k : ℕ, (8^k / ((4^k - 3^k) * (4^(k + 1) - 3^(k + 1)))) = 2 :=
sorry

end infinite_sum_equals_two_l281_281378


namespace find_a_l281_281445

variable (A B : Set ℤ) (a : ℤ)
variable (elem1 : 0 ∈ A) (elem2 : 1 ∈ A)
variable (elem3 : -1 ∈ B) (elem4 : 0 ∈ B) (elem5 : a + 3 ∈ B)

theorem find_a (h : A ⊆ B) : a = -2 := sorry

end find_a_l281_281445


namespace first_year_with_sum_15_l281_281642

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 1000) + (n % 1000 / 100) + (n % 100 / 10) + (n % 10)

theorem first_year_with_sum_15 : ∃ y > 2100, sum_of_digits y = 15 :=
  sorry

end first_year_with_sum_15_l281_281642


namespace M_subset_N_l281_281473

noncomputable def M_set : Set ℝ := { x | ∃ (k : ℤ), x = k / 4 + 1 / 4 }
noncomputable def N_set : Set ℝ := { x | ∃ (k : ℤ), x = k / 8 - 1 / 4 }

theorem M_subset_N : M_set ⊆ N_set :=
sorry

end M_subset_N_l281_281473


namespace marble_ratio_is_two_to_one_l281_281752

-- Conditions
def dan_blue_marbles : ℕ := 5
def mary_blue_marbles : ℕ := 10

-- Ratio definition
def marble_ratio : ℚ := mary_blue_marbles / dan_blue_marbles

-- Theorem statement
theorem marble_ratio_is_two_to_one : marble_ratio = 2 :=
by 
  -- Prove the statement here
  sorry

end marble_ratio_is_two_to_one_l281_281752


namespace log_fraction_identity_l281_281095

theorem log_fraction_identity (a b : ℝ) (h2 : Real.log 2 = a) (h3 : Real.log 3 = b) :
  (Real.log 12 / Real.log 15) = (2 * a + b) / (1 - a + b) := 
  sorry

end log_fraction_identity_l281_281095


namespace num_points_on_circle_with_distance_one_l281_281455

-- We define the circle
def is_on_circle (P : ℝ × ℝ) : Prop := P.1^2 + P.2^2 = 4

-- We define the line
def is_on_line (P : ℝ × ℝ) : Prop := 3 * P.1 - 4 * P.2 + 5 = 0

-- We define the distance function between a point and the line
def distance_to_line (P : ℝ × ℝ) : ℝ :=
  (abs (3 * P.1 - 4 * P.2 + 5)) / (sqrt (3^2 + (-4)^2))

-- We define the proof statement
theorem num_points_on_circle_with_distance_one :
  ({P : ℝ × ℝ | is_on_circle P ∧ distance_to_line P = 1}).card = 3 :=
sorry

end num_points_on_circle_with_distance_one_l281_281455


namespace rearrange_six_digit_sum_l281_281195

theorem rearrange_six_digit_sum (a1 a2 a3 a4 a5 a6 : ℕ) (h1 : a1 >= a4) (h2 : a4 >= a2) (h3 : a2 >= a5)
                               (h4 : a5 >= a3) (h5 : a3 >= a6) (h6 : a1 <= 9) (h7 : a2 <= 9) 
                               (h8 : a3 <= 9) (h9 : a4 <= 9) (h10 : a5 <= 9) (h11 : a6 <= 9) :
  (|a1 + a2 + a3 - (a4 + a5 + a6)| < 10) :=
sorry

end rearrange_six_digit_sum_l281_281195


namespace max_xy_of_conditions_l281_281163

noncomputable def max_xy : ℝ := 37.5

theorem max_xy_of_conditions (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 10 * x + 15 * y = 150) (h4 : x^2 + y^2 ≤ 100) :
  xy ≤ max_xy :=
by sorry

end max_xy_of_conditions_l281_281163


namespace toothpicks_150th_stage_l281_281595

theorem toothpicks_150th_stage :
  let a := 4 in  -- first term
  let d := 4 in  -- common difference
  let n := 150 in
  a + (n - 1) * d = 600 :=
by
  sorry

end toothpicks_150th_stage_l281_281595


namespace inequality_solution_l281_281000

def satisfies_inequality (x : ℝ) : Prop :=
  (2 / (x + 2) + 4 / (x + 8)) ≥ 3 / 4

def solution_set : Set ℝ :=
  { x | (-∞, -8) ∪ (-8, -4] ∪ (-2, 4] ∪ (4, ∞) = set.univ }

theorem inequality_solution :
  ∀ x : ℝ, satisfies_inequality x ↔ (x ∈ (-∞, -8) ∪ (-8, -4]) ∨ (x ∈ (-2, 4]) ∨ (x ∈ (4, ∞)) :=
by
  sorry

end inequality_solution_l281_281000


namespace new_car_fuel_consumption_l281_281714

-- Definitions of the conditions
variable (x : ℝ)
variable (h1 : ∀ x, 100 / x = 100 / (x + 2) + 4.2)
variable (h2 : x + 2 > 0)

-- The expected outcome
theorem new_car_fuel_consumption : x ≈ 5.97 :=
by
  -- We use sorry as a placeholder to skip the proof
  sorry

end new_car_fuel_consumption_l281_281714


namespace area_of_garden_l281_281933

theorem area_of_garden (L P : ℝ) (H1 : 1500 = 30 * L) (H2 : 1500 = 12 * P) (H3 : P = 2 * L + 2 * (P / 2 - L)) : 
  (L * (P/2 - L)) = 625 :=
by
  sorry

end area_of_garden_l281_281933


namespace combinatorics_sum_l281_281098

theorem combinatorics_sum :
  (Nat.choose 20 6 + Nat.choose 20 5 = 62016) :=
by
  sorry

end combinatorics_sum_l281_281098


namespace geometric_seq_general_term_k_value_range_l281_281031

variable (a_n : ℕ → ℝ) (k : ℝ)

-- Define the increasing geometric sequence with the given conditions
constant geometric_seq : ∃ a_1 q : ℝ, a_1 > 0 ∧ q > 1 ∧ (∀ n, a_n = a_1 * q ^ (n - 1)) ∧ (
  let a_1 := a_n 1 in let a_2 := a_n 2 in let a_3 := a_n 3 in
  a_1 * a_2 * a_3 = 8 ∧ 2 * (a_2 + 2) = (a_1 + 1) + (a_3 + 2))

-- State the proof problem for the first part
theorem geometric_seq_general_term :
  ∀ a_n, (∃ a_1 q, a_1 > 0 ∧ q > 1 ∧ (∀ n, a_n = a_1 * q ^ (n - 1)) ∧
           let a_1 := a_n 1 in let a_2 := a_n 2 in let a_3 := a_n 3 in
           a_1 * a_2 * a_3 = 8 ∧ 2 * (a_2 + 2) = (a_1 + 1) + (a_3 + 2)) →
  (∀ n, a_n n = 2 ^ (n - 1)) :=
by sorry

-- Define the inequality condition for the real number k
constant inequality_cond : ∀ n, (a_n n) ^ 2 + 2 ^ n * (a_n n) - k ≥ 0

-- State the proof problem for the second part
theorem k_value_range :
  (∀ a_n, (∃ a_1 q, a_1 > 0 ∧ q > 1 ∧ (∀ n, a_n = a_1 * q ^ (n - 1)) ∧
            let a_1 := a_n 1 in let a_2 := a_n 2 in let a_3 := a_n 3 in
            a_1 * a_2 * a_3 = 8 ∧ 2 * (a_2 + 2) = (a_1 + 1) + (a_3 + 2)) →
    (∀ n, a_n n = 2 ^ (n - 1)) →
    (∀ n, (a_n n) ^ 2 + 2 ^ n * (a_n n) - k ≥ 0) →
     k ≤ 3) :=
by sorry

end geometric_seq_general_term_k_value_range_l281_281031


namespace polynomial_factor_determined_l281_281397

theorem polynomial_factor_determined (d : ℝ) :
  (∀ x, (Q x = x^3 - 3 * x^2 + d * x - 27) → Q (-3) = 0) → d = -27 :=
by
  intro h
  sorry

end polynomial_factor_determined_l281_281397


namespace problem1_solution_problem2_solution_l281_281977

-- Problem 1: Prove the solution set for the given inequality
theorem problem1_solution (x : ℝ) : (2 < x ∧ x ≤ (7 / 2)) ↔ ((x + 1) / (x - 2) ≥ 3) := 
sorry

-- Problem 2: Prove the solution set for the given inequality
theorem problem2_solution (x a : ℝ) : 
  (a = 0 ∧ x = 0) ∨ 
  (a > 0 ∧ -a ≤ x ∧ x ≤ 2 * a) ∨ 
  (a < 0 ∧ 2 * a ≤ x ∧ x ≤ -a) ↔ 
  x^2 - a * x - 2 * a^2 ≤ 0 := 
sorry

end problem1_solution_problem2_solution_l281_281977


namespace court_cost_proof_l281_281173

-- Define all the given conditions
def base_fine : ℕ := 50
def penalty_rate : ℕ := 2
def mark_speed : ℕ := 75
def speed_limit : ℕ := 30
def school_zone_multiplier : ℕ := 2
def lawyer_fee_rate : ℕ := 80
def lawyer_hours : ℕ := 3
def total_owed : ℕ := 820

-- Define the calculation for the additional penalty
def additional_penalty : ℕ := (mark_speed - speed_limit) * penalty_rate

-- Define the calculation for the total fine
def total_fine : ℕ := (base_fine + additional_penalty) * school_zone_multiplier

-- Define the calculation for the lawyer's fee
def lawyer_fee : ℕ := lawyer_fee_rate * lawyer_hours

-- Define the calculation for the total of fine and lawyer's fee
def fine_and_lawyer_fee := total_fine + lawyer_fee

-- Prove the court costs
theorem court_cost_proof : total_owed - fine_and_lawyer_fee = 300 := by
  sorry

end court_cost_proof_l281_281173


namespace harmonic_inequality_for_positive_n_exists_large_harmonic_sum_l281_281666

-- Problem (a)
theorem harmonic_inequality_for_positive_n (n : ℕ) (h : 0 < n) :
  ∑ i in finset.range (2 * n + 1), if n < i then (1 : ℝ) / i else 0 ≥ 1 / 2 :=
by sorry

-- Problem (b)
theorem exists_large_harmonic_sum (N : ℝ) (h : N = 10 ^ 1995) :
  ∃ n : ℕ, 1 ≤ n ∧ ∑ i in finset.range (n + 1), (1 : ℝ) / i > N :=
by sorry

end harmonic_inequality_for_positive_n_exists_large_harmonic_sum_l281_281666


namespace exists_divisible_by_l_l281_281193

theorem exists_divisible_by_l (l : ℕ) (h1 : ¬ (2 ∣ l)) (h2 : ¬ (5 ∣ l)) : 
  ∃ n : ℕ, (n ≠ 0) ∧ (∀ d ∈ (toDigits 10 n), d = 1) ∧ (l ∣ n) :=
sorry

end exists_divisible_by_l_l281_281193


namespace find_DB_length_l281_281126

noncomputable def calculate_DB (AC AD : ℕ) (DC : ℕ) : ℝ :=
real.sqrt ((AC - AD) * 8)

theorem find_DB_length :
  ∀ (AC AD : ℕ), AC = 20 → AD = 8 → calculate_DB AC AD (AC - AD) = 4 * real.sqrt 6 :=
by
  intros AC AD hAC hAD
  rw [hAC, hAD]
  simp
  sorry

end find_DB_length_l281_281126


namespace range_of_m_l281_281223

theorem range_of_m (m : ℝ) : 
  (m - 1 < 0 ∧ 4 * m - 3 > 0) → (3 / 4 < m ∧ m < 1) := 
by
  sorry

end range_of_m_l281_281223


namespace maximum_distance_l281_281314

def highway_mileage : ℝ := 12.2
def city_mileage : ℝ := 7.6
def gallons : ℝ := 25

theorem maximum_distance (h : ℝ) (g : ℝ) :  h = highway_mileage → g = gallons → h * g = 305 :=
by
  intros h_eq g_eq
  rw [h_eq, g_eq]
  exact rfl  -- Placeholder, actual computation will be here

end maximum_distance_l281_281314


namespace find_x_sets_l281_281407

open Real

noncomputable def log : ℝ → ℝ := sorry -- Assuming log base 2 is defined

theorem find_x_sets (n : ℕ)
  (x : Fin (n+2) → ℝ)
  (h1 : x 0 = x (n+1))
  (h2 : ∀ k : Fin n, 2 * log (x k) * log (x (k + 1)) - (log (x k))^2 = 9) :
  (∀ k : Fin (n+2), x k = 8) ∨ (∀ k : Fin (n+2), x k = 1/8) := 
sorry

end find_x_sets_l281_281407


namespace polynomial_divisibility_l281_281773

theorem polynomial_divisibility (P : ℤ[X]) (hp : ∀ a b : ℤ, a + 2 * b ∣ (P.eval a) + 2 * (P.eval b)) :
  ∃ k : ℤ, P = Polynomial.C k * Polynomial.X :=
by
  sorry

end polynomial_divisibility_l281_281773


namespace min_band_members_l281_281174

def flutes_tryouts : ℕ := 20
def clarinets_tryouts : ℕ := 30
def trumpets_tryouts : ℕ := 60
def pianists_tryouts : ℕ := 20
def drummers_tryouts : ℕ := 16

def flutes_selected : ℕ := (4 * flutes_tryouts) / 5  -- 80% of 20
def clarinets_selected : ℕ := clarinets_tryouts / 2  -- 50% of 30
def trumpets_selected : ℕ := trumpets_tryouts / 3   -- 1/3 of 60
def pianists_selected : ℕ := pianists_tryouts / 10 -- 1/10 of 20
def drummers_selected : ℕ := (drummers_tryouts * 3) / 4 -- 3/4 of 16

def total_selected : ℕ := 
  flutes_selected + clarinets_selected + trumpets_selected + pianists_selected + drummers_selected

def target_ratio_total : ℕ := 20 -- Sum of the ratio parts 5+3+6+2+4

def least_band_members (n : ℕ) : Prop :=
  n >= total_selected ∧ n % target_ratio_total = 0

theorem min_band_members : ∃ n, least_band_members n ∧ n = 80 :=
begin
  sorry
end

end min_band_members_l281_281174


namespace sum_possible_values_A_B_l281_281484

theorem sum_possible_values_A_B : 
  ∀ (A B : ℕ), 
  (0 ≤ A ∧ A ≤ 9) ∧ 
  (0 ≤ B ∧ B ≤ 9) ∧ 
  ∃ k : ℕ, 28 + A + B = 9 * k 
  → (A + B = 8 ∨ A + B = 17) 
  → A + B = 25 :=
by
  sorry

end sum_possible_values_A_B_l281_281484


namespace coordinates_of_point_M_l281_281514

theorem coordinates_of_point_M 
  (M : ℝ × ℝ) 
  (dist_x_axis : abs M.2 = 5) 
  (dist_y_axis : abs M.1 = 4) 
  (second_quadrant : M.1 < 0 ∧ M.2 > 0) : 
  M = (-4, 5) := 
sorry

end coordinates_of_point_M_l281_281514


namespace sum_perimeters_bound_l281_281253

section
variables (ABCD : set (ℝ × ℝ)) (N : ℕ) (AC : set (ℝ × ℝ))

-- Define the condition that ABCD is a unit square
def is_unit_square (ABCD : set (ℝ × ℝ)) : Prop :=
  ∃ a b c d : ℝ × ℝ, [a, b, c, d] = [(0,0), (1,0), (1,1), (0,1)] ∧ ABCD = set.insert a (set.insert b (set.insert c {d}))

-- Define the decomposition of the unit square into N smaller squares
def divide_unit_square (ABCD : set (ℝ × ℝ)) (N : ℕ) : Prop :=
  N = 10^12 ∧ ∃ squares : set (set (ℝ × ℝ)), (∀ s ∈ squares, ∃ a b c d : ℝ × ℝ, [a, b, c, d] = [s1,s2,s3,s4] ∧ s = set.insert a (set.insert b (set.insert c {d}))) ∧
  (⋃ s ∈ squares, s) = ABCD

-- Define the diagonal of the unit square
def is_diagonal (AC : set (ℝ × ℝ)) (ABCD : set (ℝ × ℝ)) : Prop :=
  AC = {(t,t) | t ∈ Icc (0:ℝ) 1}

-- Define the sum of the perimeters of all squares intersecting the diagonal
def sum_perimeters_of_intersecting_squares (ABCD AC : set (ℝ × ℝ)) (N : ℕ) : ℝ := 
  ∑ s in {s ∈ divide_unit_square ABCD N | s ∩ AC ≠ ∅}, perimeter s

-- Statement of the theorem
theorem sum_perimeters_bound (ABCD AC : set (ℝ × ℝ)) (N : ℕ) :
  is_unit_square ABCD → 
  divide_unit_square ABCD N →
  is_diagonal AC ABCD → 
  sum_perimeters_of_intersecting_squares ABCD AC N ≤ 1500 :=
sorry

end

end sum_perimeters_bound_l281_281253


namespace gcd_6051_10085_l281_281599

theorem gcd_6051_10085 : Nat.gcd 6051 10085 = 2017 := by
  sorry

end gcd_6051_10085_l281_281599


namespace school_supplies_costs_correct_l281_281765

-- Define the quantities and prices
def totalGlueSticks := 27
def totalPencils := 40
def totalErasers := 15
def priceGlueStick := 1.00
def pricePencil := 0.50
def priceEraser := 0.75

def EmilyGlueSticks := 9
def EmilyPencils := 18
def EmilyErasers := 5

def SophieGlueSticks := 12
def SophiePencils := 14
def SophieErasers := 4

-- Calculate total costs for Emily, Sophie, and Sam
def totalCostForEmily := EmilyGlueSticks * priceGlueStick + EmilyPencils * pricePencil + EmilyErasers * priceEraser
def totalCostForSophie := SophieGlueSticks * priceGlueStick + SophiePencils * pricePencil + SophieErasers * priceEraser
def totalCostForSam := 
  (totalGlueSticks - EmilyGlueSticks - SophieGlueSticks) * priceGlueStick + 
  (totalPencils - EmilyPencils - SophiePencils) * pricePencil + 
  (totalErasers - EmilyErasers - SophieErasers) * priceEraser

-- Prove that the total costs are correct
theorem school_supplies_costs_correct :
  totalCostForEmily = 21.75 ∧
  totalCostForSophie = 22.00 ∧
  totalCostForSam = 14.50 :=
by
  sorry

end school_supplies_costs_correct_l281_281765


namespace NoneOfPropositionsCorrect_l281_281659

-- Definitions for propositions
def PropositionA (l : Line) (α : Plane) : Prop :=
  ∀ m : Line, (l ∥ α) → (∃ n ∈ α, m ∥ n) → (l ∥ m)

def PropositionB (A B C : Point) (α β : Plane) : Prop :=
  (A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ A ∈ α ∧ B ∈ α ∧ C ∈ α ∧ dist A β = dist B β ∧ dist B β = dist C β) ∧ (α ∥ β)

def PropositionC (b : Line) (α β : Plane) : Prop :=
  (b ∥ α ∧ α ∥ β) → (b ∥ β)

def PropositionD (a b : Line) (α : Plane) : Prop :=
  (a ∥ α ∧ a ∥ b ∧ ¬ (b ∥ α)) → (b ∥ α)

-- The theorem: None of the propositions A, B, C, or D is correct
theorem NoneOfPropositionsCorrect : ¬ (∃ l : Line, ∃ α : Plane, PropositionA l α) ∧
                                    ¬ (∃ (A B C : Point) (α β : Plane), PropositionB A B C α β) ∧
                                    ¬ (∃ b : Line, ∃ α β : Plane, PropositionC b α β) ∧
                                    ¬ (∃ a b : Line, ∃ α : Plane, PropositionD a b α) :=
by
  sorry

end NoneOfPropositionsCorrect_l281_281659


namespace boys_cannot_score_twice_as_girls_l281_281867

theorem boys_cannot_score_twice_as_girls :
  ∀ (participants : Finset ℕ) (boys girls : ℕ) (points : ℕ → ℝ),
    participants.card = 6 →
    boys = 2 →
    girls = 4 →
    (∀ p, p ∈ participants → points p = 1 ∨ points p = 0.5 ∨ points p = 0) →
    (∀ (p q : ℕ), p ∈ participants → q ∈ participants → p ≠ q → points p + points q = 1) →
    ¬ (∃ (boys_points girls_points : ℝ), 
      (∀ b ∈ (Finset.range 2), boys_points = points b) ∧
      (∀ g ∈ (Finset.range 4), girls_points = points g) ∧
      boys_points = 2 * girls_points) :=
by
  sorry

end boys_cannot_score_twice_as_girls_l281_281867


namespace solutions_to_system_of_equations_l281_281978

theorem solutions_to_system_of_equations :
  (1, 1, 1) = ( 1, 2 * 1^2 - 1, 2 * 1^2 - 1) ∧
  ( real.cos (2*real.pi/7), real.cos (6 * real.pi/7), real.cos (4 * real.pi/7)) =
    ( real.cos (2*real.pi/7), 2 * real.cos (4 * real.pi/7)^2 - 1, 2 * real.cos (6 * real.pi/7)^2 - 1) ∧
  ( real.cos (4 * real.pi/7), real.cos (2 * real.pi/7), real.cos (6 * real.pi/7)) =
    ( real.cos (4 * real.pi/7), 2 * real.cos (2 * real.pi/7)^2 - 1, 2 * real.cos (6 * real.pi/7)^2 - 1) ∧
  ( real.cos (6 * real.pi/7), real.cos (4 * real.pi/7), real.cos (2 * real.pi/7)) =
    ( real.cos (6 * real.pi/7), 2 * real.cos (4 * real.pi/7)^2 - 1, 2 * real.cos (2 * real.pi/7)^2 - 1) ∧
  ( real.cos (2 * real.pi/9), real.cos (8 * real.pi/9), real.cos (4 * real.pi/9)) =
    ( real.cos (2 * real.pi/9), 2 * real.cos (4 * real.pi/9)^2 - 1, 2 * real.cos (8 * real.pi/9)^2 - 1) ∧
  ( real.cos (4 * real.pi/9), real.cos (2 * real.pi/9), real.cos (8 * real.pi/9)) =
    ( real.cos (4 * real.pi/9), 2 * real.cos (2 * real.pi/9)^2 - 1, 2 * real.cos (8 * real.pi/9)^2 - 1) ∧
  ( -1 / 2, -1 / 2, -1 / 2) =
    ( -1 / 2, 2 * (-1 / 2)^2 - 1, 2 * (-1 / 2)^2 - 1) ∧
  ( real.cos (8 * real.pi/9), real.cos (4 * real.pi/9), real.cos (2 * real.pi/9)) =
    ( real.cos (8 * real.pi/9), 2 * real.cos (4 * real.pi/9)^2 - 1, 2 * real.cos (2 * real.pi/9)^2 - 1) ∧
  sorry

end solutions_to_system_of_equations_l281_281978


namespace rolls_in_package_l281_281696

theorem rolls_in_package (n : ℕ) :
  (9 : ℝ) = (n : ℝ) * (1 - 0.25) → n = 12 :=
by
  sorry

end rolls_in_package_l281_281696


namespace percentage_non_red_cars_l281_281863

theorem percentage_non_red_cars :
  let total_cars := 30000
  let honda_cars := 12000
  let honda_red_percentage := 80 * 0.01
  let toyota_cars := 10000
  let toyota_red_percentage := 75 * 0.01
  let nissan_cars := 8000
  let nissan_red_percentage := 60 * 0.01
  let total_red_cars := honda_cars * honda_red_percentage + toyota_cars * toyota_red_percentage + nissan_cars * nissan_red_percentage
  let total_non_red_cars := total_cars - total_red_cars
  let percentage_non_red_cars := (total_non_red_cars / total_cars) * 100
  percentage_non_red_cars = 27 := 
begin
  -- proof goes here
  sorry
end

end percentage_non_red_cars_l281_281863


namespace remaining_food_can_cater_children_l281_281709

theorem remaining_food_can_cater_children (A C : ℝ) 
  (h_food_adults : 70 * A = 90 * C) 
  (h_35_adults_ate : ∀ n: ℝ, (n = 35) → 35 * A = 35 * (9/7) * C) : 
  70 * A - 35 * A = 45 * C :=
by
  sorry

end remaining_food_can_cater_children_l281_281709


namespace minimum_value_l281_281412

def f (x y : ℝ) : ℝ := x * y / (x^2 + y^2)

def x_in_domain (x : ℝ) : Prop := (1/4 : ℝ) ≤ x ∧ x ≤ (2/3 : ℝ)
def y_in_domain (y : ℝ) : Prop := (1/5 : ℝ) ≤ y ∧ y ≤ (1/2 : ℝ)

theorem minimum_value (x y : ℝ) (hx : x_in_domain x) (hy : y_in_domain y) :
  ∃ x y, f x y = (2/5 : ℝ) := 
sorry

end minimum_value_l281_281412


namespace infinite_sum_l281_281375

theorem infinite_sum:
  ∑ k in (filter (λ n, n ≥ 1) (range (n + 1))) (8^k / ((4^k - 3^k) * (4^(k + 1) - 3^(k + 1)))) = 2 :=
sorry

end infinite_sum_l281_281375


namespace pills_per_week_l281_281529

-- Define the conditions
def pill_schedule (hours_between : ℕ) (daily_hours : ℕ := 24) (weekly_days : ℕ := 7) : ℕ :=
  (daily_hours / hours_between) * weekly_days

-- The declarations for scheduled pill intake
def pills_A_in_a_week := pill_schedule 6
def pills_B_in_a_week := pill_schedule 8
def pills_C_in_a_week := pill_schedule 12

-- The theorem statement
theorem pills_per_week :
  pills_A_in_a_week = 28 ∧ pills_B_in_a_week = 21 ∧ pills_C_in_a_week = 14 :=
by
  split
  { sorry }
  split
  { sorry }
  { sorry }

end pills_per_week_l281_281529


namespace complex_mn_mul_l281_281051

noncomputable def Z : ℂ := (1 + complex.i) * (2 + complex.i^607)

theorem complex_mn_mul (m n : ℝ) (hm : m = Z.re) (hn : n = Z.im) : 
  m * n = 3 := by 
sorry

end complex_mn_mul_l281_281051


namespace local_extrema_l281_281639

noncomputable def f (x : ℝ) := 3 * x^3 - 9 * x^2 + 3

theorem local_extrema :
  (∃ x, x = 0 ∧ ∀ δ > 0, ∃ ε > 0, ∀ y, abs (y - x) < ε → f y ≤ f x) ∧
  (∃ x, x = 2 ∧ ∀ δ > 0, ∃ ε > 0, ∀ y, abs (y - x) < ε → f y ≥ f x) :=
sorry

end local_extrema_l281_281639


namespace problem_statement_l281_281258

def star (a b : ℕ) : ℕ := a * a - b * b

theorem problem_statement :
  (3.star 2) + (4.star 3) + (5.star 4) + ... + (20.star 19) = 396 :=
by
  sorry

end problem_statement_l281_281258


namespace foci_distance_l281_281777

variable (x y : ℝ)

def ellipse_eq : Prop := (x^2 / 45) + (y^2 / 5) = 9

theorem foci_distance : ellipse_eq x y → (distance_between_foci : ℝ) = 12 * Real.sqrt 10 :=
by
  sorry

end foci_distance_l281_281777


namespace find_length_MN_l281_281674

-- Define the problem based on given conditions
def trapezoid (A B C D M N : Type) :=
  ∃ (AM BM AD BC MN : ℕ),
    AM : BM = 2 : 1 ∧
    AD = 18 ∧
    BC = 6 ∧
    (parallel : ∀ (l1 l2 : Type), l1 ≠ l2 → parallel l1 l2) ∧ -- Assuming existence of parallel at M
    (N_exists : ∃ N, line_through M parallel AD BC intersects CD at N) -- N is defined at the intersection

-- Statement that the length of MN given these conditions is 10
theorem find_length_MN : ∀ (A B C D M N : Type) (AM BM AD BC MN : ℕ), 
    trapezoid A B C D M N → MN = 10 :=
by
  -- Proof will be filled in here
  sorry

end find_length_MN_l281_281674


namespace quadrilateral_cosine_sum_zero_l281_281194

/-- If the sum of the cosines of the angles of a quadrilateral is zero, 
then the quadrilateral is either a parallelogram, a trapezium, or a cyclic quadrilateral. --/
theorem quadrilateral_cosine_sum_zero
  {α β γ δ : ℝ} -- Angles of the quadrilateral
  (h : Real.cos α + Real.cos β + Real.cos γ + Real.cos δ = 0) :
  (is_parallelogram α β γ δ) ∨ (is_trapezium α β γ δ) ∨ (is_cyclic_quadrilateral α β γ δ) := 
sorry -- Proof placeholder

end quadrilateral_cosine_sum_zero_l281_281194


namespace solve_fraction_equation_l281_281001

theorem solve_fraction_equation (x : ℝ) :
  (1 / (x^2 + 9 * x - 12) + 1 / (x^2 + 5 * x - 14) - 1 / (x^2 - 15 * x - 18) = 0) →
  x = 2 ∨ x = -9 ∨ x = 6 ∨ x = -3 :=
sorry

end solve_fraction_equation_l281_281001


namespace simplify_and_evaluate_expression_l281_281971

noncomputable def x : ℝ := (1 / 2)⁻¹ + (Real.pi - 1) ^ 0

theorem simplify_and_evaluate_expression :
  (let expr := (x - 3) / (x^2 - 1) - 2 / (x + 1) in
   let expr := expr / (x / (x^2 - 2 * x + 1)) in
   expr) = -(2 / 3) :=
by
  sorry

end simplify_and_evaluate_expression_l281_281971


namespace triangle_properties_l281_281871

noncomputable def problem_conditions (AB : ℝ) (angle_ADB : ℝ) (sin_A : ℝ) (sin_C : ℝ) :=
  AB = 25 ∧ angle_ADB = 90 ∧ sin_A = 4/5 ∧ sin_C = 1/5

theorem triangle_properties 
  (AB : ℝ) (angle_ADB : ℝ) (sin_A : ℝ) (sin_C : ℝ)
  (h_conditions : problem_conditions AB angle_ADB sin_A sin_C) :
  ∃ (DC : ℝ) (Area_ABD : ℝ), DC = 40 * real.sqrt 6 ∧ Area_ABD = 150 :=
by
  have h1 : AB = 25 := h_conditions.1
  have h2 : angle_ADB = 90 := h_conditions.2.1
  have h3 : sin_A = 4/5 := h_conditions.2.2.1
  have h4 : sin_C = 1/5 := h_conditions.2.2.2
  -- Additional proof steps go here
  sorry

end triangle_properties_l281_281871


namespace flower_perimeter_32_l281_281955

def regular_hexagon_inscribed (P : Type) [TopologicalSpace P] :=
  ∃ (C : TopologicalSpace P) (hC : Circumference C = 16), 
    hexagon_inscribed_in_circle P C

def flower_formed_by_arcs (P : Type) [TopologicalSpace P] (H : regular_hexagon_inscribed P) :=
  ∃ (F : TopologicalSpace P) (hF : formed_by_arcs_of_hexagon H F), True

theorem flower_perimeter_32 (P : Type) [TopologicalSpace P] (hH : regular_hexagon_inscribed P) (hF : flower_formed_by_arcs P hH) : 
  perimeter F = 32 :=
sorry

end flower_perimeter_32_l281_281955


namespace exists_quad_with_equal_tangents_l281_281398

theorem exists_quad_with_equal_tangents :
  ∃ (α β γ δ : ℝ), α + β + γ + δ = 360 ∧ tan α = 1 ∧ tan β = 1 ∧ tan γ = 1 ∧ tan δ = 1 :=
by
  sorry

end exists_quad_with_equal_tangents_l281_281398


namespace max_term_in_sequence_l281_281518

theorem max_term_in_sequence (a : ℕ → ℝ)
  (h : ∀ n, a n = (n+1) * (7/8)^n) :
  (∀ n, a n ≤ a 6 ∨ a n ≤ a 7) ∧ (a 6 = max (a 6) (a 7)) ∧ (a 7 = max (a 6) (a 7)) :=
sorry

end max_term_in_sequence_l281_281518


namespace lowest_degree_poly_meets_conditions_l281_281301

-- Define a predicate that checks if a polynomial P meets the conditions
def poly_meets_conditions (P : ℚ[X]) (b : ℚ) : Prop :=
  (∀ x, coeff P x ≠ b) ∧ 
  (∃ x y, coeff P x < b ∧ coeff P y > b)

-- Statement of the theorem we want to prove
theorem lowest_degree_poly_meets_conditions : ∀ (b : ℚ), 
  ∃ (P : ℚ[X]), poly_meets_conditions P b ∧ degree P = 4 :=
begin
  sorry
end

end lowest_degree_poly_meets_conditions_l281_281301


namespace eccentricity_range_l281_281924

variable (a b c e: ℝ)

def hyperbola := ∃ (a b : ℝ), (b ≠ 0) ∧ (a ≠ 0)

def points_A_B := 
  A : ℝ × ℝ := (a^2 / c, a*b / c)
  B : ℝ × ℝ := (a^2 / c, -a*b / c)

def right_focus (e : ℝ) := (sqrt (a^2 + b^2), 0)

def angle_AFB (F A B : ℝ × ℝ) := 60 < (A.1 * F.2 + A.2 * F.1) / (sqrt (A.1^2 + A.2^2) * sqrt (F.1^2 + F.2^2)) < 90

theorem eccentricity_range (h: hyperbola a b)
  (h1: 60 < angle_AFB (right_focus e a b) (a^2 / c, a*b / c) (a^2 / c, -a*b / c))
  : sqrt(2) < e ∧ e < 2 := sorry

end eccentricity_range_l281_281924


namespace a4_in_arithmetic_sequence_l281_281854

variable {a_n : ℕ → ℝ}

def is_arithmetic_sequence {a_n : ℕ → ℝ} : Prop :=
  ∀ n : ℕ, a_n = a_0 + d * n

theorem a4_in_arithmetic_sequence (h_seq : is_arithmetic_sequence a_n)
  (h_cond : a_n 3 + a_n 5 = 12) : a_n 4 = 6 :=
by
  sorry

end a4_in_arithmetic_sequence_l281_281854


namespace fifteenth_term_is_44_l281_281667

-- Define the conditions
def first_term : ℕ := 2
def common_difference : ℕ := 3
def term_number : ℕ := 15

-- Define the formula for the nth term of an arithmetic progression
def nth_term (a d n : ℕ) : ℕ := a + (n - 1) * d

-- Prove that the 15th term is 44
theorem fifteenth_term_is_44 : nth_term first_term common_difference term_number = 44 :=
by
  unfold nth_term first_term common_difference term_number
  sorry

end fifteenth_term_is_44_l281_281667


namespace isosceles_trapezoid_axial_not_central_l281_281352

-- Define axially symmetric figure and centrally symmetric figure
def axially_symmetric (fig : Type) : Prop := sorry
def centrally_symmetric (fig : Type) : Prop := sorry

-- Define specific geometric figures
structure LineSegment : Type := (exist : axially_symmetric LineSegment ∧ centrally_symmetric LineSegment)

structure Square : Type := (exist : axially_symmetric Square ∧ centrally_symmetric Square)

structure Circle : Type := (exist : axially_symmetric Circle ∧ centrally_symmetric Circle)

structure IsoscelesTrapezoid : Type := (exist : axially_symmetric IsoscelesTrapezoid ∧ ¬ centrally_symmetric IsoscelesTrapezoid)

structure Parallelogram : Type := (exist : centrally_symmetric Parallelogram ∧ ¬ axially_symmetric Parallelogram)

-- Proof that IsoscelesTrapezoid satisfies the given conditions
theorem isosceles_trapezoid_axial_not_central : axially_symmetric IsoscelesTrapezoid ∧ ¬ centrally_symmetric IsoscelesTrapezoid := by
  sorry

end isosceles_trapezoid_axial_not_central_l281_281352


namespace odd_integers_condition_l281_281403

theorem odd_integers_condition {n : ℤ} (hn : n > 1) :
  ∃ m : ℤ, ∃ p q : ℤ, 1 < p ∧ p < q ∧ prime p ∧ prime q ∧ (q - p) ∣ m ∧ p ∣ n^m + 1 ∧ q ∣ n^m + 1 ↔ odd n :=
by sorry

end odd_integers_condition_l281_281403


namespace graph_avoid_third_quadrant_l281_281859

theorem graph_avoid_third_quadrant (k : ℝ) : 
  (∀ x y : ℝ, y = (2 * k - 1) * x + k → ¬ (x < 0 ∧ y < 0)) ↔ 0 ≤ k ∧ k < (1 / 2) :=
by sorry

end graph_avoid_third_quadrant_l281_281859


namespace expected_value_is_90_l281_281331

noncomputable def expected_value_coins_heads : ℕ :=
  let nickel := 5
  let quarter := 25
  let half_dollar := 50
  let dollar := 100
  1/2 * (nickel + quarter + half_dollar + dollar)

theorem expected_value_is_90 : expected_value_coins_heads = 90 := by
  sorry

end expected_value_is_90_l281_281331


namespace lowest_degree_is_4_l281_281276

noncomputable def lowest_degree_polynomial (P : Polynomial ℤ) (b : ℤ) : Prop :=
  ∃ (b : ℤ), 
    let A_P := P.support in
    (∀ (a ∈ A_P), a < b ∨ a > b) ∧ 
    (¬(b ∈ A_P)) ∧
    (∃ (a1 a2 : ℤ), a1 ∈ A_P ∧ a2 ∈ A_P ∧ a1 < b ∧ a2 > b)

theorem lowest_degree_is_4 :
  ∀ P : Polynomial ℤ, 
    let b := lowest_degree_polynomial P in
    b P 4 :=
sorry

end lowest_degree_is_4_l281_281276


namespace inequality_problem_l281_281487

theorem inequality_problem (a : ℝ) (h : a = Real.cos (2 * Real.pi / 7)) : 
  2^(a - 1/2) < 2 * a :=
by
  sorry

end inequality_problem_l281_281487


namespace two_mathematicians_contemporaries_l281_281249

def contemporaries_probability :=
  let total_area := 600 * 600
  let triangle_area := 1/2 * 480 * 480
  let non_contemporaneous_area := 2 * triangle_area
  let contemporaneous_area := total_area - non_contemporaneous_area
  let probability := contemporaneous_area / total_area
  probability

theorem two_mathematicians_contemporaries :
  contemporaries_probability = 9 / 25 :=
by
  -- Skipping the intermediate proof steps
  sorry

end two_mathematicians_contemporaries_l281_281249


namespace range_of_m_l281_281806

def p (m : ℝ) : Prop := m^2 - 4 > 0 ∧ m > 0
def q (m : ℝ) : Prop := ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 > 0
def condition1 (m : ℝ) : Prop := p m ∨ q m
def condition2 (m : ℝ) : Prop := ¬ (p m ∧ q m)

theorem range_of_m (m : ℝ) : condition1 m ∧ condition2 m → (m ≥ 3 ∨ (1 < m ∧ m ≤ 2)) :=
by
  sorry

end range_of_m_l281_281806


namespace ellipse_area_proof_l281_281820

noncomputable def right_focus_of_ellipse (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (c : ℝ) : Prop :=
  c = (2 : ℝ) * Real.sqrt 2

noncomputable def ellipse_eq (a b : ℝ) (pt : ℝ × ℝ) :=
  pt.1 ^ 2 / a ^ 2 + pt.2 ^ 2 / b ^ 2 = 1

theorem ellipse_area_proof 
  (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) 
  (h_focus : right_focus_of_ellipse a b (2 * Real.sqrt 2))
  (pt : ℝ × ℝ)
  (h_pass : ellipse_eq a b (3, 1))
  (slope : ℝ)
  (P : ℝ × ℝ)
  (vertex_P : P = (-3, 2)) :
  (∃ (a b : ℝ), ellipse_eq 12 4 (3, 1)) ∧ (area_triangle P A B = 9 / 2) :=
sorry

end ellipse_area_proof_l281_281820


namespace john_annual_profit_l281_281144

namespace JohnProfit

def number_of_people_subletting := 3
def rent_per_person_per_month := 400
def john_rent_per_month := 900
def months_in_year := 12

theorem john_annual_profit 
  (h1 : number_of_people_subletting = 3)
  (h2 : rent_per_person_per_month = 400)
  (h3 : john_rent_per_month = 900)
  (h4 : months_in_year = 12) : 
  (number_of_people_subletting * rent_per_person_per_month - john_rent_per_month) * months_in_year = 3600 :=
by
  sorry

end JohnProfit

end john_annual_profit_l281_281144


namespace combined_angle_basic_astrophysics_nanotech_l281_281702

theorem combined_angle_basic_astrophysics_nanotech :
  let percentage_microphotonics : ℝ := 10
  let percentage_home_electronics : ℝ := 24
  let percentage_food_additives : ℝ := 15
  let percentage_gmo : ℝ := 29
  let percentage_industrial_lubricants : ℝ := 8
  let percentage_nanotechnology : ℝ := 7
  let total_percentage : ℝ := 100
  let percentage_basic_astrophysics := total_percentage - 
                                       (percentage_microphotonics + 
                                        percentage_home_electronics + 
                                        percentage_food_additives + 
                                        percentage_gmo + 
                                        percentage_industrial_lubricants + 
                                        percentage_nanotechnology)
  let combined_percentage := percentage_basic_astrophysics + 
                             percentage_nanotechnology
  let degrees_per_percentage : ℝ := 360 / total_percentage
  let combined_degrees := combined_percentage * degrees_per_percentage
  combined_degrees = 50.4 := by
begin
  sorry
end

end combined_angle_basic_astrophysics_nanotech_l281_281702


namespace symmetric_y_axis_l281_281855

theorem symmetric_y_axis (m n : ℤ) (h1 : n = 3) (h2 : m = -4) : 
  (m + n) ^ 2023 = -1 := 
by 
  rw [h2, h1] 
  norm_num }

/-
The statement in text form:
Given that point A(m, 3) is symmetric to point B(4, n) with respect to the y-axis, and thereby m = -4 and n = 3, prove that (m + n)^2023 = -1.
-/

end symmetric_y_axis_l281_281855


namespace lowest_degree_poly_meets_conditions_l281_281298

-- Define a predicate that checks if a polynomial P meets the conditions
def poly_meets_conditions (P : ℚ[X]) (b : ℚ) : Prop :=
  (∀ x, coeff P x ≠ b) ∧ 
  (∃ x y, coeff P x < b ∧ coeff P y > b)

-- Statement of the theorem we want to prove
theorem lowest_degree_poly_meets_conditions : ∀ (b : ℚ), 
  ∃ (P : ℚ[X]), poly_meets_conditions P b ∧ degree P = 4 :=
begin
  sorry
end

end lowest_degree_poly_meets_conditions_l281_281298


namespace investment_difference_l281_281680

noncomputable def Jose_investment (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r) ^ n

noncomputable def Alex_investment (P : ℝ) (r : ℝ) (m : ℕ) (n : ℕ) : ℝ :=
  P * (1 + r / m) ^ (m * n)

theorem investment_difference (P : ℝ) (r : ℝ) (m : ℕ) (n : ℕ) :
  Jose_investment P r n < Alex_investment P r m n :=
by
  have h_Jose : Jose_investment 30000 0.05 3 = 34728.75 := sorry
  have h_Alex : Alex_investment 30000 0.05 2 3 = 34802.2101 := sorry
  have h_diff : Alex_investment 30000 0.05 2 3 - Jose_investment 30000 0.05 3 = 73.46 := sorry
  linarith

end investment_difference_l281_281680


namespace find_range_a_l281_281057

def f (x : ℝ) : ℝ :=
  if x ∈ (set.Ico 0 (1/2)) then 1 / 4 ^ x
  else if x ∈ (set.Ioc (1/2) 1) then -x + 1
  else 0 -- assuming 0 for out-of-bound values for formalization

noncomputable def g (a : ℝ) (x : ℝ) : ℝ :=
  a * real.sin (π * x / 6) - a + 2

theorem find_range_a (a : ℝ) (x1 x2 : ℝ) 
  (hx1 : x1 ∈ set.Icc 0 1) 
  (hx2 : x2 ∈ set.Icc 0 1) 
  (h : f x1 = g a x2) : 1 ≤ a ∧ a ≤ 4 :=
sorry

end find_range_a_l281_281057


namespace arithmetic_sequence_general_term_l281_281124

theorem arithmetic_sequence_general_term :
  (∀ a₁ : ℤ, a₁ = 10 → ∀ d : ℤ, ∃ a : ℕ → ℤ, a 1 = 10 ∧ d = -3 ∧ ∀ n : ℕ, a n = 13 - 3 * n) →

  (∀ b₁ : ℤ, b₁ = 2 → ∀ b : ℕ → ℤ,
    (∃ f : ((ℕ → ℤ) → (ℕ → ℤ)) → Prop, f (λ b, b (n + 1) = b n + 2^n) ∧ ∀ n, b n = 2^n)) →

  (∀ c : ℕ → ℤ, ∀ a b : ℕ → ℤ, 
    ((∃ a, ∀ n, a (1) = 10 ∧ d = -3 ∧ a (n) = 13 - 3 * n) ∧ 
     (∃ b, ∀ b₁ : ℤ, b₁ = 2 ∧ ∀ n, b n = 2^n)) →
    ∀ (S : ℕ → ℤ) n, c n = a n * b n → 
       S n = (16 - 3 * n) *  2^(n + 1) - 32) :=
by sorry

end arithmetic_sequence_general_term_l281_281124


namespace divisor_of_p_l281_281169

theorem divisor_of_p (p q r s : ℕ) (h1 : Nat.gcd p q = 30) (h2 : Nat.gcd q r = 45) 
(h3 : Nat.gcd r s = 60) (h4 : 80 < Nat.gcd s p ∧ Nat.gcd s p < 120) : 
5 ∣ p :=
begin
  sorry
end

end divisor_of_p_l281_281169


namespace probability_diff_colors_l281_281240

def num_blue : ℕ := 5
def num_red : ℕ := 4
def num_yellow : ℕ := 3
def total_chips : ℕ := num_blue + num_red + num_yellow
def prob_diff_color : ℚ := (num_blue * (num_red + num_yellow) + num_red * (num_blue + num_yellow) + num_yellow * (num_blue + num_red)) / (total_chips * total_chips)

theorem probability_diff_colors : prob_diff_color = 47 / 72 := 
by 
  sorry

end probability_diff_colors_l281_281240


namespace pythagorean_quadrilateral_l281_281197

theorem pythagorean_quadrilateral 
  (a1 b1 c1 a2 b2 c2 : ℕ)
  (h1 : a1^2 + b1^2 = c1^2)
  (h2 : a2^2 + b2^2 = c2^2) :
  (∃ (OA OB OC OD : ℕ),
     OA = a1 * a2 ∧
     OB = a1 * b2 ∧
     OC = b1 * b2 ∧
     OD = a2 * b1 ∧
     OA * OC = OB * OD ∧
     ∃ R : ℕ, R = c1 * c2 / 2 ) :=
begin
  sorry
end

end pythagorean_quadrilateral_l281_281197


namespace count_difference_of_squares_l281_281351

theorem count_difference_of_squares :
  {n | ∃ m k : ℕ, 1 ≤ m ∧ m < n ∧ n ≤ 98 ∧ n = k^2 - m^2}.finite.card = 73 :=
by
  sorry

end count_difference_of_squares_l281_281351


namespace find_interest_rate_l281_281325

theorem find_interest_rate 
    (P : ℝ) (T : ℝ) (known_rate : ℝ) (diff : ℝ) (R : ℝ) :
    P = 7000 → T = 2 → known_rate = 0.18 → diff = 840 → (P * known_rate * T - (P * (R/100) * T) = diff) → R = 12 :=
by
  intros P_eq T_eq kr_eq diff_eq interest_eq
  simp only [P_eq, T_eq, kr_eq, diff_eq] at interest_eq
-- Solving equation is not required
  sorry

end find_interest_rate_l281_281325


namespace least_constant_c_for_sum_of_squares_l281_281004

theorem least_constant_c_for_sum_of_squares (n : ℕ) (x : ℕ → ℝ)
    (h1 : ∀ i, 1 ≤ i ∧ i ≤ n → x i ≥ -1)
    (h2 : ∑ i in Finset.range n, x i ^ 3 = 0) :
    (∑ i in Finset.range n, x i ^ 2) ≤ (n : ℝ) := sorry

end least_constant_c_for_sum_of_squares_l281_281004


namespace probability_top_card_10_or_face_l281_281726

theorem probability_top_card_10_or_face :
  let ranks := ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
  let suits := ["spades", "hearts", "diamonds", "clubs"]
  let deck := list.product suits ranks
  let face_cards := ["J", "Q", "K"]
  let num_10s := 4
  let num_face_cards := faces_cards.length * suits.length
  let favorable_outcomes := num_10s + num_face_cards
  let total_cards := deck.length
in (favorable_outcomes.to_rat / total_cards.to_rat) = (4 / 13 : ℚ) := by
  sorry

end probability_top_card_10_or_face_l281_281726


namespace sequence_negation_l281_281793

theorem sequence_negation (x : ℕ → ℝ) (x1_pos : x 1 > 0) (x1_neq1 : x 1 ≠ 1)
  (rec_seq : ∀ n : ℕ, x (n + 1) = (x n * (x n ^ 2 + 3)) / (3 * x n ^ 2 + 1)) :
  ∃ n : ℕ, x n ≤ x (n + 1) :=
sorry

end sequence_negation_l281_281793


namespace lowest_degree_polynomial_l281_281291

-- Define the conditions
def polynomial_conditions (P : ℕ → ℤ) (b : ℤ): Prop :=
  (∃ c, c > b ∧ c ∈ set.range P) ∧ (∃ d, d < b ∧ d ∈ set.range P) ∧ (b ∉ set.range P)

-- The main statement
theorem lowest_degree_polynomial : ∃ P : ℕ → ℤ, polynomial_conditions P 4 ∧ (∀ Q : ℕ → ℤ, polynomial_conditions Q 4 → degree Q >= 4) :=
sorry

end lowest_degree_polynomial_l281_281291


namespace min_positive_period_cos_2x_l281_281603

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x)

theorem min_positive_period_cos_2x :
  ∃ T > 0, ∀ x : ℝ, f (x + T) = f x ∧ ∀ T' > 0, (∀ x : ℝ, f (x + T') = f x) → T' ≥ T := 
sorry

end min_positive_period_cos_2x_l281_281603


namespace code_length_is_4_l281_281731

-- Definitions based on conditions provided
def code_length : ℕ := 4 -- Each code consists of 4 digits
def total_codes_with_leading_zeros : ℕ := 10^code_length -- Total possible codes allowing leading zeros
def total_codes_without_leading_zeros : ℕ := 9 * 10^(code_length - 1) -- Total possible codes disallowing leading zeros
def codes_lost_if_no_leading_zeros : ℕ := total_codes_with_leading_zeros - total_codes_without_leading_zeros -- Codes lost if leading zeros are disallowed
def manager_measured_codes_lost : ℕ := 10000 -- Manager's incorrect measurement

-- Theorem to be proved based on the problem
theorem code_length_is_4 : code_length = 4 :=
by
  sorry

end code_length_is_4_l281_281731


namespace prime_looking_count_l281_281372

def is_composite (n : ℕ) : Prop :=
  ∃ m k : ℕ, 1 < m ∧ 1 < k ∧ m * k = n

def is_prime_looking (n : ℕ) : Prop :=
  is_composite n ∧ ¬(n % 2 = 0) ∧ ¬(n % 3 = 0) ∧ ¬(n % 7 = 0)

theorem prime_looking_count :
  (Finset.filter is_prime_looking (Finset.range 2000)).card = 272 :=
sorry

end prime_looking_count_l281_281372


namespace distance_upstream_l281_281711

/-- Proof that the distance a man swims upstream is 18 km given certain conditions. -/
theorem distance_upstream (c : ℝ) (h1 : 54 / (12 + c) = 3) (h2 : 12 - c = 6) : (12 - c) * 3 = 18 :=
by
  sorry

end distance_upstream_l281_281711


namespace round_trip_time_correct_l281_281937

-- Definitions
def distance_to_mall : ℝ := 60
def speed_city : ℝ := 30
def speed_highway : ℝ := 90
def shopping_time : ℝ := 2
def speed_congestion : ℝ := 60

-- Assume the given city distance
def city_distance : ℝ := 18

-- Time calculations
def time_to_highway : ℝ := city_distance / speed_city
def highway_distance : ℝ := distance_to_mall - city_distance
def time_on_highway : ℝ := highway_distance / speed_highway
def time_on_congested_highway : ℝ := highway_distance / speed_congestion
def time_through_city_return : ℝ := city_distance / speed_city

-- Total travel time calculation
def total_travel_time : ℝ :=
  time_to_highway + time_on_highway + shopping_time + time_on_congested_highway + time_through_city_return

-- Proof statement
theorem round_trip_time_correct : total_travel_time = 4.367 := by
  sorry

end round_trip_time_correct_l281_281937


namespace arithmetic_base_conversion_l281_281749

-- We start with proving base conversions

def convert_base3_to_base10 (n : ℕ) : ℕ := 1 * (3^0) + 2 * (3^1) + 1 * (3^2)

def convert_base7_to_base10 (n : ℕ) : ℕ := 6 * (7^0) + 5 * (7^1) + 4 * (7^2) + 3 * (7^3)

def convert_base9_to_base10 (n : ℕ) : ℕ := 6 * (9^0) + 7 * (9^1) + 8 * (9^2) + 9 * (9^3)

-- Prove the main equality

theorem arithmetic_base_conversion:
  (2468 : ℝ) / convert_base3_to_base10 121 + convert_base7_to_base10 3456 - convert_base9_to_base10 9876 = -5857.75 :=
by
  have h₁ : convert_base3_to_base10 121 = 16 := by native_decide
  have h₂ : convert_base7_to_base10 3456 = 1266 := by native_decide
  have h₃ : convert_base9_to_base10 9876 = 7278 := by native_decide
  rw [h₁, h₂, h₃]
  sorry

end arithmetic_base_conversion_l281_281749


namespace infinite_sum_equals_two_l281_281379

theorem infinite_sum_equals_two :
  ∑' k : ℕ, (8^k / ((4^k - 3^k) * (4^(k + 1) - 3^(k + 1)))) = 2 :=
sorry

end infinite_sum_equals_two_l281_281379


namespace stratified_sampling_sample_size_l281_281697

theorem stratified_sampling_sample_size :
  let teachers := 200
  let male_students := 800
  let female_students := 600
  let male_sample := 100
  let prob_selection := male_sample / male_students := 1 / 8
  let teachers_sample := 200 * prob_selection
  let female_sample := 600 * prob_selection
  teachers_sample + female_sample + male_sample = 200 := by 
  sorry

end stratified_sampling_sample_size_l281_281697


namespace simplify_root_subtraction_l281_281203

axiom eight_cubed_root : 8^(1/3) = 2
axiom three_hundred_forty_three_cubed_root : 343^(1/3) = 7

theorem simplify_root_subtraction : 8^(1/3) - 343^(1/3) = -5 :=
by {
  rw [eight_cubed_root, three_hundred_forty_three_cubed_root],
  norm_num,
}

end simplify_root_subtraction_l281_281203


namespace area_of_larger_square_l281_281362

theorem area_of_larger_square (side_length : ℕ) (num_squares : ℕ)
  (h₁ : side_length = 2)
  (h₂ : num_squares = 8) : 
  (num_squares * side_length^2) = 32 :=
by
  sorry

end area_of_larger_square_l281_281362


namespace jenga_blocks_before_jess_turn_l281_281138

theorem jenga_blocks_before_jess_turn
  (total_blocks : ℕ := 54)
  (players : ℕ := 5)
  (rounds : ℕ := 5)
  (father_turn_blocks : ℕ := 1)
  (original_blocks := total_blocks - (players * rounds + father_turn_blocks))
  (jess_turn_blocks : ℕ := total_blocks - original_blocks):
  jess_turn_blocks = 28 := by
begin
  sorry
end

end jenga_blocks_before_jess_turn_l281_281138


namespace set_equality_l281_281609

open Set

variable {a b : ℝ}

theorem set_equality (h : {a, b / a, 1} = {a^2, a + b, 0}) : a^2013 + b^2014 = -1 :=
sorry

end set_equality_l281_281609


namespace angle_A_min_side_BC_l281_281878

variables {A B C : ℝ} {a b c : ℝ}

-- Define the vectors m and n
def m := (c - 2 * b, a)
def n := (Real.cos A, Real.cos C)

-- Define the conditions given in the problem
def ortho_condition := (c - 2 * b) * Real.cos A + a * Real.cos C = 0
def ab_ac_dot := (Real.cos A) * (2 * b) + (Real.sin A) * a = 4

-- Prove that the angle A equals π / 3
theorem angle_A (h : ortho_condition) : A = Real.pi / 3 := sorry

-- Prove that the minimum value of side BC (i.e., a) is 2sqrt(2)
theorem min_side_BC (h : ab_ac_dot) : a >= 2 * Real.sqrt 2 := sorry

end angle_A_min_side_BC_l281_281878


namespace ordered_triples_count_l281_281481

/-- Given the equations ab^2 = c, bc^2 = a, and ca^2 = b, prove that the number of ordered triples
(a, b, c) of non-zero real numbers satisfying these equations is 2 -/
theorem ordered_triples_count :
  {p : ℝ × ℝ × ℝ // 
    p.1 ≠ 0 ∧ p.2 ≠ 0 ∧ p.2 ≠ 0 ∧
    p.1 * p.2^2 = p.2 ∧
    p.2 * p.3^2 = p.1 ∧
    p.3 * p.0^2 = p.2}.finite.card = 2 :=
sorry

end ordered_triples_count_l281_281481


namespace simplify_expression_l281_281202

theorem simplify_expression :
  (8 : ℝ)^(1/3) - (343 : ℝ)^(1/3) = -5 :=
by
  sorry

end simplify_expression_l281_281202


namespace sculptures_not_on_display_approx_400_l281_281358

theorem sculptures_not_on_display_approx_400 (A : ℕ) (hA : A = 900) :
  (2 / 3 * A - 2 / 9 * A) = 400 := by
  sorry

end sculptures_not_on_display_approx_400_l281_281358


namespace equilateral_triangle_sections_l281_281120

theorem equilateral_triangle_sections (A B C P : Point) 
  (h_equilateral : equilateral △ABC)
  (h_P_inside : inside P △ABC)
  (h_AP_BP_CP: dist P A = dist P B ∧ dist P B = dist P C)
  (h_parallel_lines : ∀ p q r, parallel (line_through_points p P) (line_through_points q r) ∧ inside P △pqr) :
  ¬ (∃ six_sections : list (section P △ABC), 
      (∀ s ∈ six_sections, same_area s) ∧ 
      (∀ s ∈ six_sections, same_shape s) ∧ 
      (∀ s ∈ six_sections, section_is_triangle s) ∧
      ∃ quad1 quad2 : section P △ABC, is_parallelogram quad1 ∧ is_parallelogram quad2) :=
sorry

end equilateral_triangle_sections_l281_281120


namespace kite_intersection_l281_281441

theorem kite_intersection
  (A B C P Q E F : Type)
  (triangleABC : ∀ (a b c : A), acute_triangle a b c)
  (ab_lt_ac : AB < AC)
  (angle_bisector_PQ : lies_on_angle_bisector (A, B, C) P ∧ lies_on_angle_bisector (A, B, C) Q)
  (bp_perp_bisector : BP ⊥ angle_bisector (A, B, C))
  (cq_perp_bisector : CQ ⊥ angle_bisector (A, B, C))
  (point_on_sides : on_side E AB ∧ on_side F AC)
  (kite_AEPF : kite A E P F):
  intersects (line BC) (line PF) (line QE) :=
sorry

end kite_intersection_l281_281441


namespace triangle_ratio_l281_281520

theorem triangle_ratio (X Y Z E G Q : Type) (hx : X ≠ Y)
  (hy : Y ≠ Z) (hz : Z ≠ X)
  (E_on_XZ : E ∈ line[X,Z])
  (G_on_XY : G ∈ line[X,Y])
  (Q_on_intersection : Q ∈ (line[X,E] ∩ line[Y,G]))
  (ratio_XQ_QE : XQ / QE = 3 / 2)
  (ratio_GQ_QY : GQ / QY = 1 / 3) :
  XG / GY = 7 / 8 :=
sorry

end triangle_ratio_l281_281520


namespace geometric_Sn_over_n_sum_first_n_terms_l281_281923

-- The first problem statement translation to Lean 4
theorem geometric_Sn_over_n (a S : ℕ → ℕ) (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, 0 < n → n * a (n+1) = (n + 2) * S n) :
  ∃ r : ℕ, (r = 2 ∧ ∃ b : ℕ, b = 1 ∧ 
    ∀ n : ℕ, 0 < n → (S (n + 1)) / (n + 1) = r * (S n) / n) := 
sorry

-- The second problem statement translation to Lean 4
theorem sum_first_n_terms (a S : ℕ → ℕ) (T : ℕ → ℕ)
  (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, 0 < n → n * a (n + 1) = (n + 2) * S n)
  (h3 : ∀ n : ℕ, S n = n * 2^(n - 1)) :
  ∀ n : ℕ, T n = (n - 1) * 2^n + 1 :=
sorry

end geometric_Sn_over_n_sum_first_n_terms_l281_281923


namespace mathematicians_contemporaries_probability_l281_281252

noncomputable def probability_contemporaries : ℚ :=
  let overlap_area : ℚ := 129600
  let total_area : ℚ := 360000
  overlap_area / total_area

theorem mathematicians_contemporaries_probability :
  probability_contemporaries = 18 / 25 :=
by
  sorry

end mathematicians_contemporaries_probability_l281_281252


namespace solutions_are_correct_l281_281979

-- Define the system of equations
def eq1 (x y : ℚ) : Prop := 3 * x - y - 3 * x * y = -1
def eq2 (x y : ℚ) : Prop := 9 * x ^ 2 * y ^ 2 + 9 * x ^ 2 + y ^ 2 - 6 * x * y = 13

-- Define each of the solutions
def solution1 : ℚ × ℚ := (-2/3, 1)
def solution2 : ℚ × ℚ := (1, 1)
def solution3 : ℚ × ℚ := (-1/3, -3)
def solution4 : ℚ × ℚ := (-1/3, 2)

noncomputable def solve_system : list (ℚ × ℚ) :=
  [solution1, solution2, solution3, solution4]

-- The theorem statement to check all solutions
theorem solutions_are_correct : 
  ∀ (x y : ℚ), (x, y) ∈ solve_system -> eq1 x y ∧ eq2 x y :=
by
  intros x y h
  cases h
  case or.inl h_eq1 => sorry
  case or.inr h1 =>
    cases h1
    case or.inl h_eq2 => sorry
    case or.inr h2 => 
      cases h2
      case or.inl h_eq3 => sorry
      case or.inr h3 => sorry

end solutions_are_correct_l281_281979


namespace not_cheap_necessary_for_good_quality_l281_281582

/-- Sister Qian's statement: "you get what you pay for".
    This implies that if something is cheap, then it is of "not good quality." -/
def cheap_implies_not_good_quality := ∀ x : Prop, cheap x → ¬ good_quality x

/-- To prove: "not cheap" is a necessary condition for "good quality". -/
theorem not_cheap_necessary_for_good_quality : ∀ x : Prop, good_quality x → ¬ cheap x :=
by
  sorry

end not_cheap_necessary_for_good_quality_l281_281582


namespace solution_set_quadratic_inequality_l281_281416

theorem solution_set_quadratic_inequality :
  {x : ℝ | (x^2 - 3*x + 2) < 0} = {x : ℝ | 1 < x ∧ x < 2} :=
sorry

end solution_set_quadratic_inequality_l281_281416


namespace power_sum_mod_inverse_l281_281743

theorem power_sum_mod_inverse :
  (2^3 + (2^3)⁻¹) % 17 = 6 := by
begin
  -- Add the proof here.
  sorry
end

end power_sum_mod_inverse_l281_281743


namespace evaluate_expression_l281_281768

theorem evaluate_expression : (6^6) * (12^6) * (6^12) * (12^12) = 72^18 := 
by sorry

end evaluate_expression_l281_281768


namespace nested_package_exists_l281_281499

theorem nested_package_exists 
(packets : Fin 100 → Fin 101) 
(candies : Fin 100 → Fin 2019 → Prop)
(total_candies : ∀ i : Fin 100, ∃ k : ℕ, candies i k ∧ k < 2019)
(no_empty_packages : ∀ i : Fin 100, packets i > 0)
(unique_candy_counts : ∀ i j : Fin 100, i ≠ j → packets i ≠ packets j) : 
∃ i j : Fin 100, i ≠ j ∧ (∀ k : ℕ, candies i k → candies j k) :=
begin
  sorry
end

end nested_package_exists_l281_281499


namespace determine_a_l281_281022

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x < 1 then 2 * x + a else -x - 2 * a

theorem determine_a (a : ℝ) (h : a ≠ 0) (h_eq : f a (1 - a) = f a (1 + a)) : a = -3/4 :=
by
  sorry

end determine_a_l281_281022


namespace num_ordered_triples_eq_ten_l281_281336

theorem num_ordered_triples_eq_ten :
  ∃ (a b c : ℕ), 
  (1 ≤ a ∧ a ≤ b ∧ b ≤ c) ∧ 
  2 * (a * b + b * c + c * a) = a * b * c ∧ 
  { (a, b, c) | 1 ≤ a ∧ a ≤ b ∧ b ≤ c ∧ 2 * (a * b + b * c + c * a) = a * b * c }.to_finset.card = 10 :=
sorry

end num_ordered_triples_eq_ten_l281_281336


namespace periodic_sequence_l281_281420

theorem periodic_sequence {p : ℚ[X]} (degree_p : ∀ n, p.degree ≥ 2) (q : ℕ → ℚ) (h : ∀ n, q n = p.eval (q (n + 1))) :
  ∃ k > 0, ∀ n, q (n + k) = q n :=
sorry

end periodic_sequence_l281_281420


namespace find_a3_l281_281472

noncomputable def a : ℕ → ℤ
| 0       := sorry -- Not defined for n = 0 as per positive naturals
| 1       := 1
| (n+1) := a n + 2

theorem find_a3 : a 3 = 5 := by
  sorry

end find_a3_l281_281472


namespace tan_condition_sum_of_sides_l281_281862

noncomputable theory
open Real

variables {a b c : ℝ}
variables {A B C : ℝ}

-- Conditions for the first question
def geom_seq (a b c : ℝ) : Prop := (b / a) = (c / b)
def cos_B_condition (a b c : ℝ) : Prop := Real.cos B = 3 / 4

-- Proof that given a, b, and c form a geometric sequence and cos B = 3/4, certain values hold
theorem tan_condition (h1 : geom_seq a b c) (h2 : cos_B_condition a b c) :
  (1 / Real.tan A + 1 / Real.tan B = 2 * Real.sqrt 7 / 7 ∨ 
   1 / Real.tan A + 1 / Real.tan B = 8 * Real.sqrt 7 / 7) :=
sorry

-- Conditions for the second question
def dot_product_condition (a b c : ℝ) : Prop := (a * c * Real.cos B = 3 / 2)

-- Proof that given the conditions, a + c = 3
theorem sum_of_sides (h1 : geom_seq a b c) (h3 : dot_product_condition a b c) : 
  a + c = 3 :=
sorry

end tan_condition_sum_of_sides_l281_281862


namespace prob_multiples_of_3_or_4_l281_281997

open Set

def multiples_of_3_and_4 : Set ℕ := {n | n % 3 = 0 ∨ n % 4 = 0}

theorem prob_multiples_of_3_or_4 : 
  (1 / 2 : ℚ) = (Finite.toFinset (multiples_of_3_and_4 ∩ Icc 1 30)).card / 30 :=
begin
  sorry
end

end prob_multiples_of_3_or_4_l281_281997


namespace sum_palindromic_primes_l281_281951

/-- Definition of a palindromic prime -/
def is_palindromic_prime (p : ℕ) : Prop :=
  p < 200 ∧ p ≥ 100 ∧ p.isPrime ∧ (Nat.reverseDigits p).isPrime

/-- Sum of all palindromic primes less than 200 -/
theorem sum_palindromic_primes : 
  ∑ p in (Finset.filter is_palindromic_prime (Finset.range 200)).val, p = 1299 :=
sorry

end sum_palindromic_primes_l281_281951


namespace range_of_f_l281_281444

noncomputable def f (x y : ℝ) := (x^3 + y^3) / (x + y)^3

theorem range_of_f :
  (∀ x y : ℝ, 0 < x ∧ 0 < y ∧ x^2 + y^2 = 1 → (1 / 4) ≤ f x y ∧ f x y < 1) :=
by
  sorry

end range_of_f_l281_281444


namespace total_area_of_circles_is_pi_over_12_l281_281970

noncomputable def total_area_of_circles_in_sierpinski_triangle
  (side_length : ℝ)
  (inscribed_circle_area : ℝ → ℝ) : ℝ :=
if side_length = 1 then inscribed_circle_area (side_length^2 * (sqrt 3 / 4)) else 0

theorem total_area_of_circles_is_pi_over_12 :
  total_area_of_circles_in_sierpinski_triangle 1 (λ K, (π / 3 / sqrt 3) * K) = π / 12 :=
by
  sorry

end total_area_of_circles_is_pi_over_12_l281_281970


namespace correct_numbers_l281_281625

noncomputable def numbers_on_board (x : List ℕ) : Prop :=
  ∃ (x_1 x_2 : ℕ) (xs : List ℕ), list.sort nat.lt (x_1 :: xs ++ [x_2]) = x ∧
  35 * x_1 + list.sum (xs ++ [x_2]) = 592 ∧
  x_1 + list.sum (xs ++ [16 * x_2]) = 592

theorem correct_numbers :
  ∀ (x : List ℕ), numbers_on_board x → x = [15, 16, 17, 34] ∨ x = [15, 33, 34] :=
by
  intros
  sorry

end correct_numbers_l281_281625


namespace midpoint_of_segment_l281_281744

theorem midpoint_of_segment : 
  let A := (5 : ℤ, -8 : ℤ)
  let B := (-7 : ℤ, 6 : ℤ)
  let M := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  M = (-1, -1) := 
by
  let A := (5 : ℤ, -8 : ℤ)
  let B := (-7 : ℤ, 6 : ℤ)
  let M := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  have h1 : A.1 + B.1 = -2 := by norm_num
  have h2 : A.2 + B.2 = -2 := by norm_num
  have h3 : M = (-1, -1) := by 
    simp [M, h1, h2]
    norm_num
  exact h3

end midpoint_of_segment_l281_281744


namespace required_workers_l281_281363

variable (x : ℕ) (y : ℕ)

-- Each worker can produce x units of a craft per day.
-- A craft factory needs to produce 60 units of this craft per day.

theorem required_workers (h : x > 0) : y = 60 / x ↔ x * y = 60 :=
by sorry

end required_workers_l281_281363


namespace ellipse_eccentricity_l281_281822

noncomputable def eccentricity (a b : ℝ) (h : a > b > 0) : ℝ :=
  Real.sqrt (1 - b^2 / a^2)

theorem ellipse_eccentricity
  (a b : ℝ)
  (h1 : a > b)
  (h2 : b > 0)
  (h3 : b * 0 - a * 0 + 3 * a * b = a * Real.sqrt (a^2 + b^2)) :
  eccentricity a b ⟨h1, h2⟩ = Real.sqrt 14 / 4 :=
by
  sorry

end ellipse_eccentricity_l281_281822


namespace min_packs_126_l281_281974

-- Define the sizes of soda packs
def pack_sizes : List ℕ := [6, 12, 24, 48]

-- Define the total number of cans required
def total_cans : ℕ := 126

-- Define a function to calculate the minimum number of packs required
noncomputable def min_packs_to_reach_target (target : ℕ) (sizes : List ℕ) : ℕ :=
sorry -- Implementation will be complex dynamic programming or greedy algorithm

-- The main theorem statement to prove
theorem min_packs_126 (P : ℕ) (h1 : (min_packs_to_reach_target total_cans pack_sizes) = P) : P = 4 :=
sorry -- Proof not required

end min_packs_126_l281_281974


namespace S_n_proof_l281_281921

noncomputable def sequence_S (a : ℕ → ℤ) : ℕ → ℤ
| 0 := 0
| (n + 1) := sequence_S a n + a (n + 1)

theorem S_n_proof (a : ℕ → ℤ) (S : ℕ → ℤ)
  (h1 : a 1 = -1)
  (h2 : ∀ n, a (n + 1) = S n * S (n + 1)) 
  (h3 : ∀ n, S n = sequence_S a n) :
  ∀ n ≥ 1, S n = -1 / n :=
sorry

end S_n_proof_l281_281921


namespace janes_shadow_length_l281_281886

theorem janes_shadow_length (tree_height tree_shadow_length jane_height : ℝ) (h_tree_shadow : tree_shadow_length = 10) (h_jane_height : jane_height = 1.5) (h_tree_height : tree_height = 30) : ∃ S : ℝ, S = 0.5 :=
by
  let tree_height := 30
  let tree_shadow_length := 10
  let jane_height := 1.5
  use 0.5
  sorry

end janes_shadow_length_l281_281886


namespace number_of_true_propositions_is_one_l281_281814

variables {Line Plane : Type}
variables {m n : Line} {α β : Plane}

-- Definitions for parallel and perpendicular relations between lines and planes
def ParallelLinePlane (l : Line) (p : Plane) : Prop := sorry
def PerpendicularLinePlane (l : Line) (p : Plane) : Prop := sorry
def ParallelLines (l1 l2 : Line) : Prop := sorry
def PerpendicularLines (l1 l2 : Line) : Prop := sorry
def LineInPlane (l : Line) (p : Plane) : Prop := sorry
def ParallelPlanes (p1 p2 : Plane) : Prop := sorry

-- Propositions
def prop1 := ∀ m n α, ParallelLinePlane m α ∧ ParallelLinePlane n α → ParallelLines m n
def prop2 := ∀ m n α, ParallelLinePlane m α ∧ PerpendicularLinePlane n α → PerpendicularLines n m
def prop3 := ∀ m n α, PerpendicularLines m n ∧ PerpendicularLinePlane m α → ParallelLinePlane n α
def prop4 := ∀ m n α β, LineInPlane m α ∧ LineInPlane n β ∧ ParallelLines m n → ParallelPlanes α β

-- Proof problem for exactly one proposition being true
theorem number_of_true_propositions_is_one :
  (prop1 = false) ∧
  (prop2 = true) ∧
  (prop3 = false) ∧
  (prop4 = false)
:=
sorry

end number_of_true_propositions_is_one_l281_281814


namespace treasures_on_island_l281_281316

-- Define the propositions P and K
def P : Prop := ∃ p : Prop, p
def K : Prop := ∃ k : Prop, k

-- Define the claim by A
def A_claim : Prop := K ↔ P

-- Theorem statement as specified part (b)
theorem treasures_on_island (A_is_knight_or_liar : (A_claim ↔ true) ∨ (A_claim ↔ false)) : ∃ P, P :=
by
  sorry

end treasures_on_island_l281_281316


namespace composite_with_divisors_ratio_l281_281756

theorem composite_with_divisors_ratio (n : ℕ) (h : n = 4) :
  ∃ (d : ℕ) (k : ℕ), (1 = d_1 < d_2 < ⋯ < d_k = n) ∧
  ∀ i, 1 ≤ i < k → (d_{i+1} - d_i) / (d_{i} - d_{i-1}) = i :=
begin
  sorry
end

end composite_with_divisors_ratio_l281_281756


namespace trapezoid_base_solutions_l281_281214

theorem trapezoid_base_solutions :
  ∃ (b1 b2 : ℕ), (b1 + b2 = 90) ∧ (b1 % 5 = 0) ∧ (b2 % 5 = 0) ∧ (b1 ≤ b2) ∧ 
  (finset.card (finset.univ.filter (λ (b1 b2 : ℕ), (b1 + b2 = 90 ∧ b1 % 5 = 0 ∧ b2 % 5 = 0 ∧ b1 ≤ b2))) = 10) :=
sorry

end trapezoid_base_solutions_l281_281214


namespace remainder_of_sum_mod_13_l281_281419

theorem remainder_of_sum_mod_13 (a b c d e : ℕ) 
  (h1: a % 13 = 3) (h2: b % 13 = 5) (h3: c % 13 = 7) (h4: d % 13 = 9) (h5: e % 13 = 11) :
  (a + b + c + d + e) % 13 = 9 := 
by 
  sorry

end remainder_of_sum_mod_13_l281_281419


namespace meet_time_bike_ride_l281_281746

theorem meet_time_bike_ride :
    ∀ (Cassie_rate Brian_rate distance : ℝ) (Cassie_start Brian_start x : ℝ),
    Cassie_rate = 15 → 
    Brian_rate = 18 →
    Cassie_start = 8 →
    Brian_start = 8.75 →
    distance = 75 →
    x = 2.68 →
    Cassie_start + x = 10.683333 (approximately 10:41 AM) :=
begin
  -- Lean code will contain the actual proof here
  sorry
end

end meet_time_bike_ride_l281_281746


namespace minimum_value_fraction_l281_281452

theorem minimum_value_fraction (a b : ℝ) (h_min_inc : ∀ x y : ℝ, x < y → a^x + b < a^y + b) (h_b_pos : b > 0) (h_point : a + b = 3) :
  (∃ (a b : ℝ), b > 0 ∧ a + b = 3 ∧ monotone (λ x, a^x + b)) → (4 / (a - 1) + 1 / b) ≥ 9 / 2 :=
by
  sorry

end minimum_value_fraction_l281_281452


namespace max_plus_min_l281_281853

noncomputable def f : ℝ → ℝ := sorry

axiom condition1 (x₁ x₂ : ℝ) : f (x₁ + x₂) = f x₁ + f x₂ - 2016
axiom condition2 (x : ℝ) : x > 0 → f x > 2016

theorem max_plus_min (M N : ℝ) (hM : M = f 2016) (hN : N = f (-2016)) : M + N = 4032 :=
by
  sorry

end max_plus_min_l281_281853


namespace find_special_numbers_l281_281945

theorem find_special_numbers : 
  ∃ (n1 n2 : ℕ), 
    (100 ≤ n1 ∧ n1 < 1000 ∧ 
    (∀ (d1 d2 d3 : ℕ), n1 = 100 * d1 + 10 * d2 + d3 → d1 ≠ d2 ∧ d2 ≠ d3 ∧ d1 ≠ d3 ∧ d1 < d2 ∧ d2 < d3 ∧ 
    (String.head (String.words (to_string n1)).getD "" = String.head (String.words (to_string n1)).getD ""))) ∧
    (100 ≤ n2 ∧ n2 < 1000 ∧
    (∀ (d : ℕ), n2 = d * 111 → 
    let name_words := String.words (to_string n2) in
    (String.head name_words.head = "O") ∧ 
    (String.head name_words.tail.headD "" = "H") ∧ 
    (String.head (name_words.drop 2).headD "" = "E"))) ∧
    n1 = 147 ∧ n2 = 111 := by {
  sorry
}

end find_special_numbers_l281_281945


namespace num_ordered_quadruples_l281_281165

theorem num_ordered_quadruples (n : ℕ) :
  ∃ (count : ℕ), count = (1 / 3 : ℚ) * (n + 1) * (2 * n^2 + 4 * n + 3) ∧
  (∀ (k1 k2 k3 k4 : ℕ), k1 ≤ n ∧ k2 ≤ n ∧ k3 ≤ n ∧ k4 ≤ n → 
    ((k1 + k3) / 2 = (k2 + k4) / 2) → 
    count = (1 / 3 : ℚ) * (n + 1) * (2 * n^2 + 4 * n + 3)) :=
by sorry

end num_ordered_quadruples_l281_281165


namespace count_valid_arrangements_l281_281110

def grid := Array (Array Char)

def isValidGrid (g : grid) : Prop :=
  ∀ i j, ∃ k, g[i][k] = 'X' ∧ k ≠ j ∧ g[k][j] = 'X' ∧ k ≠ i

def validArrangements (g : grid) : Prop :=
  ∀ i, (∀ j, (g[i][j] = 'X' ∨ g[i][j] = 'Y' ∨ g[i][j] = 'Z')) ∧
  (∃! k, g[i][k] = 'X') ∧ (∃! k, g[i][k] = 'Y') ∧ (∃! k, g[i][k] = 'Z')

def conditionGrid (g : grid) : Prop :=
  g[0][0] = 'Y' ∧ g[1][1] = 'X'

theorem count_valid_arrangements :
  ∃ g : grid, conditionGrid g ∧ validArrangements g ∧ isValidGrid g → 2 :=
sorry

end count_valid_arrangements_l281_281110


namespace problem_l281_281093

variable (x y : ℝ)

theorem problem (h1 : (x + y)^2 = 81) (h2 : x * y = 18) : (x - y)^2 = 9 :=
by
  sorry

end problem_l281_281093


namespace concert_students_l281_281676

theorem concert_students (buses : ℕ) (students_per_bus : ℕ) (h₁ : buses = 8) (h₂ : students_per_bus = 45) : 
  buses * students_per_bus = 360 := 
by
  rw [h₁, h₂]
  calc
    8 * 45 = 360 := by norm_num

end concert_students_l281_281676


namespace find_multiplied_number_l281_281588

theorem find_multiplied_number (A B C D E : ℝ) 
  (h1 : (A + B + C + D + E) / 5 = 6.8)
  (h2 : ∃ X, X ∈ {A, B, C, D, E} ∧ ((A + B + C + D + E - X + 2 * X) / 5 = 9.2)) :
  ∃ X, X ∈ {A, B, C, D, E} ∧ X = 12 := 
sorry

end find_multiplied_number_l281_281588


namespace intersection_point_of_lines_l281_281475

theorem intersection_point_of_lines (x y : ℝ) :
  (2 * x - 3 * y = 3) ∧ (4 * x + 2 * y = 2) ↔ (x = 3/4) ∧ (y = -1/2) :=
by
  sorry

end intersection_point_of_lines_l281_281475


namespace inequality_solution_l281_281405

theorem inequality_solution (x : ℝ) (h1 : x ≠ -2) (h2 : x ≠ -6) :
  (1 / (x + 2) + 8 / (x + 6) ≥ 1) ↔ (x ∈ set.Ici 5 ∨ x ∈ set.Iic (-6) ∨ x ∈ set.Icc (-2) 5) :=
by
  sorry

end inequality_solution_l281_281405


namespace equal_segments_l281_281927

variables (K I A O Y Z X : Type)
variables [IsPoint K] [IsPoint I] [IsPoint A] [IsPoint O] [IsPoint Y] [IsPoint Z] [IsPoint X]
variables [FootOfMedian KIA O] [FootOfPerpendicular I (AngleBisector I O K) Y] [FootOfPerpendicular A (AngleBisector A O K) Z]
variables [IntersectionPoint K O Y Z X]

theorem equal_segments (hKO : is_median_segment K O)
                       (hIY : is_foot_perpendicular I (angle_bisector I O K) Y)
                       (hAZ : is_foot_perpendicular A (angle_bisector A O K) Z)
                       (hX : intersect_segments K O Y Z X) :
  distance Y X = distance Z X :=
sorry

end equal_segments_l281_281927


namespace sum_palindromic_primes_l281_281949

/-- Definition of a palindromic prime -/
def is_palindromic_prime (p : ℕ) : Prop :=
  p < 200 ∧ p ≥ 100 ∧ p.isPrime ∧ (Nat.reverseDigits p).isPrime

/-- Sum of all palindromic primes less than 200 -/
theorem sum_palindromic_primes : 
  ∑ p in (Finset.filter is_palindromic_prime (Finset.range 200)).val, p = 1299 :=
sorry

end sum_palindromic_primes_l281_281949


namespace plane_divides_median_ratio_l281_281170

noncomputable theory
open_locale classical

variables {A B C D : Type} [affine_space ℝ (euclidean_space ℝ (fin 3))]

-- Define points A, B, C, D
variables (a b c d : euclidean_space ℝ (fin 3))

-- Axioms ensuring the points are not coplanar
axiom not_coplanar : ¬ affine_independent ℝ ![a, b, c, d]

-- Define medians and centroids
def centroid (p1 p2 p3 : euclidean_space ℝ (fin 3)) : euclidean_space ℝ (fin 3) :=
  1 / 3 • (p1 + p2 + p3)

-- Define the required median for triangle ACD
def median (p1 p2 p3 : euclidean_space ℝ (fin 3)) : set (euclidean_space ℝ (fin 3)) :=
  affine_segment ℝ ![(p1 + p2) / 2, p3]

-- Plane passing through the centroid of triangle ABC which is parallel to AB and CD
def plane_through_centroid_parallel_lines (cen : euclidean_space ℝ (fin 3))
  (l1 l2 : euclidean_space ℝ (fin 3) → euclidean_space ℝ (fin 3)) : Prop :=
  ∀ x : euclidean_space ℝ (fin 3), (vector_span ℝ {cen + x | x ∈ vector_span ℝ {l1 a, l2 d}})
  
-- Define the problem
theorem plane_divides_median_ratio
  (cen : euclidean_space ℝ (fin 3)) 
  (parallel_planar : plane_through_centroid_parallel_lines cen (λ x, b - a) (λ x, d - c)) :
  divides_median (cen + (a - cen) / 2) (a + d) (1:2) :=
sorry

end plane_divides_median_ratio_l281_281170


namespace smallest_positive_period_max_and_min_values_l281_281462

noncomputable def f (x : ℝ) : ℝ := sin (2 * x + π / 6) + 2 * (sin x)^2

theorem smallest_positive_period :
  ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = π :=
sorry

theorem max_and_min_values :
  (∀ x, f x ≤ 2) ∧ (∀ x, f x ≥ 0) ∧
  (∃ x, f x = 2) ∧ (∃ x, f x = 0) :=
sorry

end smallest_positive_period_max_and_min_values_l281_281462


namespace count_valid_c_values_l281_281788

theorem count_valid_c_values : 
  ∃ count : ℕ, count = 334 ∧ ∀ c, 0 ≤ c ∧ c ≤ 2000 → 
    (∃ x, c = 9 * ⌊x⌋ + 3 * ⌈x⌉) ↔ (c % 12 = 0 ∨ c % 12 = 3) :=
sorry

end count_valid_c_values_l281_281788


namespace P_n_max_real_roots_zero_l281_281751

open Real

noncomputable def P_n (n : ℕ) (x : ℝ) : ℝ :=
  x^(2*n) - x^n + ∑ i in Finset.range n, (-1)^i * x^(n - 1 - i)

theorem P_n_max_real_roots_zero (n : ℕ) (hn : n > 0) : 
  card {x : ℝ | P_n n x = 0} = 0 :=
sorry

end P_n_max_real_roots_zero_l281_281751


namespace michael_birth_year_l281_281211

theorem michael_birth_year (first_imo_year : ℕ) (annual_event : ∀ n : ℕ, n > 0 → (first_imo_year + n) ≥ first_imo_year) 
  (michael_age_at_10th_imo : ℕ) (imo_count : ℕ) 
  (H1 : first_imo_year = 1959) (H2 : imo_count = 10) (H3 : michael_age_at_10th_imo = 15) : 
  (first_imo_year + imo_count - 1 - michael_age_at_10th_imo = 1953) := 
by 
  sorry

end michael_birth_year_l281_281211


namespace tour_group_visits_l281_281728

def Zhangjiajie : Prop := sorry
def Fenghuang : Prop := sorry
def Yan_Emperor_Mausoleum : Prop := sorry
def Shaoshan : Prop := sorry
def Langshan : Prop := sorry

theorem tour_group_visits :
  (Zhangjiajie → Fenghuang) ∧
  (Shaoshan ∨ Langshan) ∧
  (¬(Fenghuang ∧ Yan_Emperor_Mausoleum)) ∧
  (Yan_Emperor_Mausoleum ↔ Shaoshan) ∧
  (Langshan → (Zhangjiajie ∧ Shaoshan))
  → (¬Zhangjiajie ∧ ¬Fenghuang ∧ ¬Langshan ∧ Yan_Emperor_Mausoleum ∧ Shaoshan) :=
begin
  sorry
end

end tour_group_visits_l281_281728


namespace problem_inequality_problem_equality_condition_l281_281919

theorem problem_inequality (a b c : ℕ) (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) :
  (a^3 + b^3 + c^3) / 3 ≥ a * b * c + a + b + c :=
sorry

theorem problem_equality_condition (a b c : ℕ) :
  (a^3 + b^3 + c^3) / 3 = a * b * c + a + b + c ↔ a + 1 = b ∧ b + 1 = c :=
sorry

end problem_inequality_problem_equality_condition_l281_281919


namespace find_lambda_find_cosine_l281_281790

def vec (a b : ℝ) : ℝ × ℝ := (a, b)

def e1 := vec 1 0
def e2 := vec 0 1
def a (λ : ℝ) := (2 : ℝ) • e1 + λ • e2
def b := e1 - e2
def c := e1 + (2 : ℝ) • e2

-- Define a function to check parallelism of vectors
def parallel (x y : ℝ × ℝ) : Prop :=
  (x.1 * y.2 = x.2 * y.1)

-- Define dot product
def dot_product (x y : ℝ × ℝ) : ℝ :=
  x.1 * y.1 + x.2 * y.2

-- Define magnitude
def magnitude (x : ℝ × ℝ) : ℝ :=
  Real.sqrt (x.1 * x.1 + x.2 * x.2)

theorem find_lambda (λ : ℝ) (h : parallel (a λ) b) : λ = -2 := 
by  sorry

theorem find_cosine (λ : ℝ) (h : parallel (a λ) b) (hλ : λ = -2) : 
  (dot_product (a λ) c) / ((magnitude (a λ)) * (magnitude c)) = - Real.sqrt 10 / 10 := 
by  sorry

end find_lambda_find_cosine_l281_281790


namespace find_x_sets_l281_281409

theorem find_x_sets (n : ℕ) (x : Fin (n + 1) → ℝ) 
  (h1 : x 0 = x n)
  (h2 : ∀ k : Fin n, 2 * log 2 (x k) * log 2 (x (k + 1)) - (log 2 (x k)) ^ 2 = 9) :
  (∀ k : Fin (n + 1), x k = 8) ∨ (∀ k : Fin (n + 1), x k = 1 / 8) :=
begin
  sorry
end

end find_x_sets_l281_281409


namespace correct_statements_count_l281_281355

-- Definitions for the given conditions
def cond1 : Prop := ∀ (parallelogram : Type) [is_parallelogram parallelogram], 
  one_pair_opposite_sides_parallel parallelogram ∧ one_pair_opposite_sides_equal parallelogram

def cond2 : Prop := ∀ (quadrilateral : Type) (h1 : one_pair_opposite_sides_parallel quadrilateral) 
  (h2 : one_pair_opposite_sides_equal quadrilateral), is_parallelogram quadrilateral

def cond3 : Prop := ∀ (rhombus : Type) [is_rhombus rhombus], 
  diagonals_are_perpendicular rhombus

def cond4 : Prop := ∀ (quadrilateral : Type) (h : diagonals_are_perpendicular quadrilateral), 
  is_rhombus quadrilateral

-- Assertion of the number of correct statements
def num_correct_statements : Nat := 2

theorem correct_statements_count : (cond1 ∧ cond3) ∧ ¬(cond2 ∧ cond4) → num_correct_statements = 2 := sorry

end correct_statements_count_l281_281355


namespace norma_initial_cards_l281_281176

noncomputable def initial_cards (lost_cards : ℕ) (remaining_cards : ℕ) : ℕ :=
  lost_cards + remaining_cards

theorem norma_initial_cards (lost_cards remaining_cards : ℕ) (h_lost : lost_cards = 70) (h_remaining : remaining_cards = 18) :
  initial_cards lost_cards remaining_cards = 88 :=
by
  have h1 : initial_cards 70 18 = 70 + 18 := rfl
  rw [h_lost, h_remaining]
  rw [h1]
  norm_num
  sorry

end norma_initial_cards_l281_281176


namespace scoops_of_natural_seedless_raisins_l281_281841

theorem scoops_of_natural_seedless_raisins 
  (cost_natural : ℝ := 3.45) 
  (cost_golden : ℝ := 2.55) 
  (num_golden : ℝ := 20) 
  (cost_mixture : ℝ := 3) : 
  ∃ x : ℝ, (3.45 * x + 20 * 2.55 = 3 * (x + 20)) ∧ x = 20 :=
sorry

end scoops_of_natural_seedless_raisins_l281_281841


namespace intersection_complement_l281_281172

-- Definitions of the sets
def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 3}

-- The complement of B in U
def complement_U (U B : Set ℕ) : Set ℕ := U \ B

-- Statement to prove
theorem intersection_complement : A ∩ (complement_U U B) = {1} := 
by 
  sorry

end intersection_complement_l281_281172


namespace quadratic_polynomial_roots_l281_281209

theorem quadratic_polynomial_roots (x y z : ℝ) 
  (h1 : x + y = 15)
  (h2 : x * y = 36) :
  Polynomial := sorry

end quadratic_polynomial_roots_l281_281209


namespace meeting_time_l281_281634

theorem meeting_time
  (v1 v2 t2 : ℝ) (h1 : v1 = 6) (h2 : v2 = 4) (h3 : t2 = 600) :
  ∃ t1 : ℝ, t1 = 1200 ∧ (10 * t1 = 8 * t1 + v2 * t2) :=
by
  use 1200
  split
  - sorry
  - sorry

end meeting_time_l281_281634


namespace find_length_BC_l281_281877

namespace TriangleProblem

noncomputable def length_BC (AB AC AM : ℝ) (M_is_midpoint : Prop) : ℝ :=
  if AB = 5 ∧ AC = 7 ∧ AM = 4 ∧ M_is_midpoint then 2 * Real.sqrt(21) else 0

theorem find_length_BC :
  ∀ (AB AC AM : ℝ) (M_is_midpoint : Prop), AB = 5 → AC = 7 → AM = 4 → M_is_midpoint →
  length_BC AB AC AM M_is_midpoint = 2 * Real.sqrt(21) :=
by intros; simp [length_BC]; sorry

end TriangleProblem

end find_length_BC_l281_281877


namespace train_speed_with_16_coaches_is_correct_l281_281735

-- Define the conditions
def speed_without_coaches : ℝ := 90
def speed_with_16_coaches : ℝ := 78
def number_of_coaches : ℕ := 16

-- Define the function for the speed of the train
def speed (S₀ : ℝ) (k : ℝ) (n : ℕ) : ℝ :=
  S₀ - k * real.sqrt n

-- The proof problem
theorem train_speed_with_16_coaches_is_correct :
  ∃ (k : ℝ), speed speed_without_coaches k number_of_coaches = speed_with_16_coaches :=
sorry

end train_speed_with_16_coaches_is_correct_l281_281735


namespace zero_not_in_range_of_g_l281_281160

def g (x : ℝ) : ℤ :=
  if x > -3 then (Real.ceil (2 / (x + 3)))
  else if x < -3 then (Real.floor (2 / (x + 3)))
  else 0 -- g(x) is not defined at x = -3, hence this is a placeholder

noncomputable def range_g : Set ℤ := {n | ∃ x : ℝ, g x = n}

theorem zero_not_in_range_of_g : 0 ∉ range_g :=
by
  intros h,
  exact sorry

end zero_not_in_range_of_g_l281_281160


namespace estate_value_l281_281939

noncomputable theory

def total_estate_received_by_children (E : ℝ) (x : ℝ) : Prop :=
  5 * x = (1 / 2) * E

def total_estate_received_by_wife (x : ℝ) : ℝ :=
  6 * x

def total_estate_received_by_cooks : ℝ :=
  800 + 1200

def total_estate_definition (E x : ℝ) : Prop :=
  E = 10 * x ∧ E = 11 * x + total_estate_received_by_cooks

theorem estate_value (E x : ℝ) (hx : total_estate_received_by_children E x)
  (hc : total_estate_definition E x) : E = 20000 :=
  by 
    -- We introduce the assumptions
    have h1 : 5 * x = (1 / 2) * E := hx
    have h2 : E = 10 * x ∧ E = 11 * x + total_estate_received_by_cooks := hc

    -- From h2 we have two equalities for E
    cases h2 with h2a h2b

    -- We rewrite h2a to express x in terms of E and simplify
    rw [h2a] at h1
    have : x = 2000,
    { linarith, },
    rw [this] at h2a h2b,

    -- Substituting x to get the value of the estate
    have h3 : E = 10 * 2000 := h2a,
    exact h3

end estate_value_l281_281939


namespace probability_odd_divisor_of_factorial_l281_281369

theorem probability_odd_divisor_of_factorial (n : ℕ) (fact_n := nat.factorial n) :
  n = 15 → 
  let p := 2^11 * 3^6 * 5^3 * 7 * 11 * 13,
      total_divisors := ∏ (e : ℕ) in [11+1, 6+1, 3+1, 1+1, 1+1, 1+1], e,
      odd_divisors := ∏ (e : ℕ) in [6+1, 3+1, 1+1, 1+1, 1+1], e in
  p = fact_n →
  total_divisors = 1344 →
  odd_divisors = 224 →
  (224 / 1344 : ℚ) = 1 / 6 :=
by {
  sorry
}

end probability_odd_divisor_of_factorial_l281_281369


namespace cooling_time_condition_l281_281651

theorem cooling_time_condition :
  ∀ (θ0 θ1 θ1' θ0' : ℝ) (t : ℝ), 
    θ0 = 20 → θ1 = 100 → θ1' = 60 → θ0' = 20 →
    let θ := θ0 + (θ1 - θ0) * Real.exp (-t / 4)
    let θ' := θ0' + (θ1' - θ0') * Real.exp (-t / 4)
    (θ - θ' ≤ 10) → (t ≥ 5.52) :=
sorry

end cooling_time_condition_l281_281651


namespace distance_between_foci_of_ellipse_l281_281779

-- Definitions based on conditions in the problem
def ellipse_eq (x y : ℝ) : Prop := (x^2) / 45 + (y^2) / 5 = 9

-- Proof statement
theorem distance_between_foci_of_ellipse : 
  ∀ (x y : ℝ), ellipse_eq x y → 2 * (√ (405 - 45)) = 12 * (√ 10) :=
by 
  sorry

end distance_between_foci_of_ellipse_l281_281779


namespace fewest_candies_l281_281673

-- Define the number of candies for each person
variables {Candies : Type} [LinearOrder Candies] [DecidableRel ((≤) : Candies → Candies → Prop)]
variables (A B C : Candies)

-- Conditions specified in the problem
noncomputable def condition1 := A = 13
noncomputable def condition2 := A = B - 3
noncomputable def condition3 := A = C + 1
noncomputable def condition4 := ¬ (B = min A (min B C))
noncomputable def condition5 := abs (B - C) = 4
noncomputable def condition6 := A = 11
noncomputable def condition7 := C < A
noncomputable def condition8 := A = 10
noncomputable def condition9 := B = A + 2

-- Let F_OR_G be the statement that identifies a false statement out of three
noncomputable def false_statement (s1 s2 s3 : Prop) : Prop := (s1 ∧ ¬s2 ∧ ¬s3) ∨ (¬s1 ∧ s2 ∧ ¬s3) ∨ (¬s1 ∧ ¬s2 ∧ s3)

-- Prove the number of candies of the smallest person
theorem fewest_candies :
  ∃ (m : Candies), m = min A (min B C) ∧ m = 9 :=
by {
  -- Conditions from the problem
  have h1 : condition1 := rfl,
  have h2 : condition2 := sorry,
  have h3 : condition3 := sorry,
  have h4 : condition4 := sorry,
  have h5 : condition5 := sorry,
  have h6 : condition6 := sorry,
  have h7 : condition7 := sorry,
  have h8 : condition8 := sorry,
  have h9 : condition9 := sorry,
  
  -- Define false statements for each person
  have fA := false_statement condition1 condition2 condition3,
  have fB := false_statement condition4 condition5 condition6,
  have fC := false_statement condition7 condition8 condition9,

  -- Proof
  sorry
}

end fewest_candies_l281_281673


namespace tan_beta_eq_l281_281440

variable (α β : Real)
variable h1 : Real.sin α = 4/5
variable h2 : Real.tan (α - β) = 2/3

theorem tan_beta_eq :
  Real.tan β = 6/17 :=
by
  sorry

end tan_beta_eq_l281_281440


namespace integral_sqrt_circ_l281_281766

theorem integral_sqrt_circ {x : ℝ} 
    (h_diff : ∀ x ∈ set.Icc (-2 : ℝ) (2: ℝ), differentiable_at ℝ (λ x, sqrt(4 - x ^ 2)) x) :
    ∫ x in -2..2, sqrt(4 - x^2) = 2 * real.pi :=
sorry

end integral_sqrt_circ_l281_281766


namespace distinct_values_sum_of_squares_l281_281439

-- Define the sequence and conditions
def sequence (n : ℕ) : Prop :=
  ∀ i : ℕ, i < n → (a i = -1 ∨ a i = 0 ∨ a i = 1)

def sum_sequence_eq (n : ℕ) (s : ℕ) : Prop :=
  (∑ i in finset.range n, a i) = s

def sum_squares_with_offset_bound (n : ℕ) (lower_bound upper_bound : ℕ) : Prop :=
  lower_bound ≤ (∑ i in finset.range n, (a i + 1) * (a i + 1)) ∧
  (∑ i in finset.range n, (a i + 1) * (a i + 1)) ≤ upper_bound

def distinct_square_sums_count (n : ℕ) (count : ℕ) : Prop :=
  ∃ D : finset ℕ,
  (∀ s : ℕ, s ∈ D ↔
    ∃ a : ℕ → ℤ, sequence n a ∧
      sum_sequence_eq n s a 9 ∧
      sum_squares_with_offset_bound n a 101 111 ∧
      (∑ i in finset.range n, a i * a i) = s)
  ∧ D.card = count

-- Main statement
theorem distinct_values_sum_of_squares :
  distinct_square_sums_count 50 6 :=
sorry

end distinct_values_sum_of_squares_l281_281439


namespace lowest_degree_is_4_l281_281288

noncomputable def lowest_degree_polynomial (P : ℝ → ℝ) : ℕ :=
  if ∃ b : ℤ, (∀ coeff ∈ (P.coefficients), coeff < (b : ℝ) ∨ coeff > (b : ℝ)) ∧ (¬ ∃ coeff ∈ (P.coefficients), coeff = (b : ℝ))
  then Polynomial.natDegree P
  else 0

theorem lowest_degree_is_4 : ∀ (P : Polynomial ℝ), 
  (∃ b : ℤ, (∀ coeff ∈ P.coefficients, coeff < (b : ℝ) ∨ coeff > (b : ℝ)) ∧ (¬ ∃ coeff ∈ P.coefficients, coeff = (b : ℝ)))
  → lowest_degree_polynomial P = 4 :=
by
  sorry

end lowest_degree_is_4_l281_281288


namespace samantha_score_l281_281866

variables (correct_answers geometry_correct_answers incorrect_answers unanswered_questions : ℕ)
          (points_per_correct : ℝ := 1) (additional_geometry_points : ℝ := 0.5)

def total_score (correct_answers geometry_correct_answers : ℕ) : ℝ :=
  correct_answers * points_per_correct + geometry_correct_answers * additional_geometry_points

theorem samantha_score 
  (Samantha_correct : correct_answers = 15)
  (Samantha_geometry : geometry_correct_answers = 4)
  (Samantha_incorrect : incorrect_answers = 5)
  (Samantha_unanswered : unanswered_questions = 5) :
  total_score correct_answers geometry_correct_answers = 17 := 
by
  sorry

end samantha_score_l281_281866


namespace catering_service_comparison_l281_281942

theorem catering_service_comparison :
  ∃ (x : ℕ), 150 + 18 * x > 250 + 15 * x ∧ (∀ y : ℕ, y < x -> (150 + 18 * y ≤ 250 + 15 * y)) ∧ x = 34 :=
sorry

end catering_service_comparison_l281_281942


namespace complex_sum_equality_l281_281171

theorem complex_sum_equality (x : ℂ) (h : x ^ 1005 = 1) (hx : x ≠ 1) :
  (∑ k in Finset.range 1005, x ^ (2 * (k + 1)) / (x ^ (k + 1) - 1)) = 502.5 :=
begin
  sorry
end

end complex_sum_equality_l281_281171


namespace part1_part2_l281_281048

def S (n : ℕ) : ℚ := 1 / 2 * n^2 + 1 / 2 * n
def a (n : ℕ) : ℕ := n
def b (n : ℕ) : ℕ := n * 2^(n-1)

theorem part1 (n : ℕ) : a n = n :=
by {
  sorry
}

theorem part2 (n : ℕ) (T : ℕ → ℤ) : 
  (∀ m, T m = ∑ i in finset.range m, b (i+1)) →
  T n = (n-1) * 2^n + 1 :=
by {
  sorry
}

end part1_part2_l281_281048


namespace gasoline_price_decrease_l281_281882

theorem gasoline_price_decrease (a : ℝ) (h : 0 ≤ a) :
  8.1 * (1 - a / 100) ^ 2 = 7.8 :=
sorry

end gasoline_price_decrease_l281_281882


namespace expression_evaluation_l281_281668

theorem expression_evaluation (p q : ℝ) (h : p / q = 4 / 5) : (25 / 7 + (2 * q - p) / (2 * q + p)) = 4 :=
by {
  sorry
}

end expression_evaluation_l281_281668


namespace rolls_in_package_l281_281695

theorem rolls_in_package (n : ℕ) :
  (9 : ℝ) = (n : ℝ) * (1 - 0.25) → n = 12 :=
by
  sorry

end rolls_in_package_l281_281695


namespace infinite_series_sum_l281_281382

theorem infinite_series_sum : 
  ∑' k : ℕ, (8 ^ k) / ((4 ^ k - 3 ^ k) * (4 ^ (k + 1) - 3 ^ (k + 1))) = 2 :=
by 
  sorry

end infinite_series_sum_l281_281382


namespace square_diff_l281_281076

theorem square_diff (x y : ℝ) (h1 : (x + y)^2 = 81) (h2 : x * y = 18) : (x - y)^2 = 9 :=
by 
  sorry

end square_diff_l281_281076


namespace distinct_real_numbers_proof_l281_281541

variables {a b c : ℝ}

theorem distinct_real_numbers_proof (h₁ : a ≠ b) (h₂ : b ≠ c) (h₃ : c ≠ a)
  (h₄ : (a / (b - c) + b / (c - a) + c / (a - b)) = -1) :
  (a^3 / (b - c)^2) + (b^3 / (c - a)^2) + (c^3 / (a - b)^2) = 0 :=
sorry

end distinct_real_numbers_proof_l281_281541


namespace fraction_inequality_l281_281435

theorem fraction_inequality (k : ℕ) (hk : 2 ≤ k) : ( (k + 1) / 2 ) ^ k > nat.factorial k :=
sorry

end fraction_inequality_l281_281435


namespace exhibition_admission_fees_ratio_l281_281213

theorem exhibition_admission_fees_ratio
  (a c : ℕ)
  (h1 : 30 * a + 15 * c = 2925)
  (h2 : a % 5 = 0)
  (h3 : c % 5 = 0) :
  (a / 5 = c / 5) :=
by
  sorry

end exhibition_admission_fees_ratio_l281_281213


namespace arithmetic_fraction_subtraction_l281_281641

theorem arithmetic_fraction_subtraction :
  (2 + 4 + 6 + 8) / (1 + 3 + 5 + 7) - (1 + 3 + 5 + 7) / (2 + 4 + 6 + 8) = 9 / 20 :=
by
  sorry

end arithmetic_fraction_subtraction_l281_281641


namespace colombian_coffee_amount_l281_281142

theorem colombian_coffee_amount 
  (C B : ℝ) 
  (h1 : C + B = 100)
  (h2 : 8.75 * C + 3.75 * B = 635) :
  C = 52 := 
sorry

end colombian_coffee_amount_l281_281142


namespace lock_code_count_l281_281607

theorem lock_code_count : 
  let digits := {1, 2, 3, 4, 5, 6, 7} in
  let even_digits := {2, 4, 6} in
  ∃ choices : List (Fin 5 -> Fin 7), 
    (∀ c ∈ choices, c 0 ∈ even_digits ∧ c 4 ∈ even_digits ∧ (∀ i j, i ≠ j → c i ≠ c j)) → 
    choice.length = 360 := 
by
  sorry

end lock_code_count_l281_281607


namespace distance_between_homes_l281_281935

def speed (name : String) : ℝ :=
  if name = "Maxwell" then 4
  else if name = "Brad" then 6
  else 0

def meeting_time : ℝ := 4

def delay : ℝ := 1

def distance_covered (name : String) : ℝ :=
  if name = "Maxwell" then speed name * meeting_time
  else if name = "Brad" then speed name * (meeting_time - delay)
  else 0

def total_distance : ℝ :=
  distance_covered "Maxwell" + distance_covered "Brad"

theorem distance_between_homes : total_distance = 34 :=
by
  -- proof goes here
  sorry

end distance_between_homes_l281_281935


namespace first_figure_shaded_area_second_figure_shaded_area_l281_281247

-- Define the problem conditions for the first and second figure
def leg1 : ℝ := 4
def leg2 : ℝ := 7

-- The first figure's shaded area problem statement
theorem first_figure_shaded_area :
  let AB := leg2 in
  let AD := leg1 in
  let area_rectangle := AB * AD in
  let area_each_triangle := area_rectangle / 4 in
  (3 * area_each_triangle) = 21 := 
by
  have AB := leg2
  have AD := leg1
  have area_rectangle := AB * AD
  have area_each_triangle := area_rectangle / 4
  calc (3 * area_each_triangle) = 3 * (area_rectangle / 4) : by rfl
                         ... = (3 * (7 * 4)) / 4      : by simp [area_rectangle, AB, AD]
                         ... = 84 / 4                : by ring
                         ... = 21                    : by norm_num

-- The second figure's shaded area problem statement
theorem second_figure_shaded_area :
  let AD := leg1 in
  let AC := leg2 in
  let DE (x : ℝ) := x in
  let AE (y : ℝ) := y in
  let shaded_area (x : ℝ) := (2 * x) + 14 in
  2 * (33 / 14) + 14 = 18 + 5 / 7 := 
by
  let x := 33 / 14
  have shaded_area := (2 * x) + 14
  calc shaded_area = 2 * (33 / 14) + 14 : by rfl
                 ... = (66 / 14) + 14    : by ring
                 ... = (33 / 7) + 14     : by field_simp
                 ... = 4 + 5 / 7 + 14     : by ring
                 ... = 18 + 5 / 7          : by norm_num

end first_figure_shaded_area_second_figure_shaded_area_l281_281247


namespace ferris_wheel_time_to_height_l281_281686

-- Definitions based on the conditions of the problem
def radius : ℝ := 30
def revolution_time : ℝ := 90
def height_from_bottom : ℝ := 15

-- Theorem statement
theorem ferris_wheel_time_to_height :
  ∃ t : ℝ, (0 ≤ t ∧ t ≤ revolution_time / 2) ∧ (radius * (1 - real.cos (2 * real.pi * t / revolution_time)) = height_from_bottom) := 
sorry

end ferris_wheel_time_to_height_l281_281686


namespace y_minus_x_l281_281096

theorem y_minus_x (x y : ℝ)
  (h1 : sqrt(x+1) - sqrt(-1-x) = (x + y) ^ 2)
  (h2 : x = -1) : y - x = 2 :=
by sorry

end y_minus_x_l281_281096


namespace squared_difference_l281_281079

variable {x y : ℝ}

theorem squared_difference (h1 : (x + y)^2 = 81) (h2 : x * y = 18) :
  (x - y)^2 = 9 :=
by
  sorry

end squared_difference_l281_281079


namespace number_of_children_is_30_l281_281566

-- Informal statements
def total_guests := 80
def men := 40
def women := men / 2
def adults := men + women
def children := total_guests - adults
def children_after_adding_10 := children + 10

-- Formal proof statement
theorem number_of_children_is_30 :
  children_after_adding_10 = 30 := by
  sorry

end number_of_children_is_30_l281_281566


namespace equilateral_triangles_in_T_l281_281167

-- Define the set T as the set of all points (x, y) where x and y are in {0, 1, 2, 3}.
def T : set (ℕ × ℕ) := {p | p.1 ∈ {0, 1, 2, 3} ∧ p.2 ∈ {0, 1, 2, 3}}

-- Prove the number of equilateral triangles with vertices in T is 18.
theorem equilateral_triangles_in_T : 
  (count_equilateral_triangles T) = 18 := sorry

end equilateral_triangles_in_T_l281_281167


namespace mathematicians_contemporaries_probability_l281_281251

noncomputable def probability_contemporaries : ℚ :=
  let overlap_area : ℚ := 129600
  let total_area : ℚ := 360000
  overlap_area / total_area

theorem mathematicians_contemporaries_probability :
  probability_contemporaries = 18 / 25 :=
by
  sorry

end mathematicians_contemporaries_probability_l281_281251


namespace laran_weekly_profit_l281_281894

-- Definitions based on the problem conditions
def daily_posters_sold : ℕ := 5
def large_posters_sold_daily : ℕ := 2
def small_posters_sold_daily : ℕ := daily_posters_sold - large_posters_sold_daily

def price_large_poster : ℕ := 10
def cost_large_poster : ℕ := 5
def profit_large_poster : ℕ := price_large_poster - cost_large_poster

def price_small_poster : ℕ := 6
def cost_small_poster : ℕ := 3
def profit_small_poster : ℕ := price_small_poster - cost_small_poster

def daily_profit_large_posters : ℕ := large_posters_sold_daily * profit_large_poster
def daily_profit_small_posters : ℕ := small_posters_sold_daily * profit_small_poster
def total_daily_profit : ℕ := daily_profit_large_posters + daily_profit_small_posters

def school_days_week : ℕ := 5
def weekly_profit : ℕ := total_daily_profit * school_days_week

-- Statement to prove
theorem laran_weekly_profit : weekly_profit = 95 := sorry

end laran_weekly_profit_l281_281894


namespace production_statistics_relation_l281_281982

noncomputable def a : ℚ := (10 + 12 + 14 + 14 + 15 + 15 + 16 + 17 + 17 + 17) / 10
noncomputable def b : ℚ := (15 + 15) / 2
noncomputable def c : ℤ := 17

theorem production_statistics_relation : c > a ∧ a > b :=
by
  sorry

end production_statistics_relation_l281_281982


namespace perfect_square_iff_divisibility_l281_281960

theorem perfect_square_iff_divisibility (A : ℕ) :
  (∃ d : ℕ, A = d^2) ↔ ∀ n : ℕ, n > 0 → ∃ j : ℕ, 1 ≤ j ∧ j ≤ n ∧ n ∣ (A + j)^2 - A :=
sorry

end perfect_square_iff_divisibility_l281_281960


namespace sum_neg_one_exponents_even_odd_split_l281_281770

theorem sum_neg_one_exponents_even_odd_split :
  ∑ k in Finset.range 2006, (-1)^k = 0 :=
by
  sorry

end sum_neg_one_exponents_even_odd_split_l281_281770


namespace find_wall_width_l281_281628

def brick_length_cm : ℝ := 25
def brick_width_cm : ℝ := 11.25
def brick_height_cm : ℝ := 6
def num_bricks : ℕ := 6400
def wall_length_m : ℝ := 8
def wall_height_m : ℝ := 22.5
def total_volume_of_all_bricks_m³ : ℝ := 10.8

def brick_volume_cm³ : ℝ := brick_length_cm * brick_width_cm * brick_height_cm
def total_volume_of_bricks_cm³ : ℝ := brick_volume_cm³ * num_bricks
def total_volume_of_bricks_m³ : ℝ := total_volume_of_bricks_cm³ / 1000000

theorem find_wall_width : 
  let W := total_volume_of_all_bricks_m³ / (wall_length_m * wall_height_m) in
  W = 0.06 :=
by
  sorry

end find_wall_width_l281_281628


namespace median_of_combined_list_is_4041_l281_281385

-- Definition of the list elements
def doubled_numbers (n : ℕ) : List ℕ :=
  List.range (n + 1) |>.map (λ i => (i + 1) * 2)

def log_numbers (n : ℕ) : List ℕ :=
  List.range (n + 1) |>.map (λ i => Int.floor (Real.log (i + 1) / Real.log 2))

def squared_numbers (n : ℕ) : List ℕ :=
  List.range (n + 1) |>.map (λ i => (i + 1) * (i + 1))

-- Definition of the combined list and its median
def combined_list : List ℕ :=
  doubled_numbers 2020 ++ log_numbers 2020 ++ squared_numbers 2020

def median (l : List ℕ) : ℕ :=
  let sorted := l.qsort (≤)
  let n := l.length
  (sorted.get! (n / 2 - 1) + sorted.get! (n / 2)) / 2

-- The proof statement
theorem median_of_combined_list_is_4041 : median combined_list = 4041 := by
  sorry

end median_of_combined_list_is_4041_l281_281385


namespace distinct_solutions_sub_l281_281552

open Nat Real

theorem distinct_solutions_sub (p q : Real) (hpq_distinct : p ≠ q) (h_eqn_p : (p - 4) * (p + 4) = 17 * p - 68) (h_eqn_q : (q - 4) * (q + 4) = 17 * q - 68) (h_p_gt_q : p > q) : p - q = 9 := 
sorry

end distinct_solutions_sub_l281_281552


namespace sum_of_naturals_between_28_and_31_l281_281648

theorem sum_of_naturals_between_28_and_31 : ∑ n in finset.filter (λ x, 28 < x ∧ x ≤ 31) (finset.range 32) = 90 :=
by {
  sorry
}

end sum_of_naturals_between_28_and_31_l281_281648


namespace sum_of_numbers_in_range_l281_281646

def is_in_range (n : ℕ) : Prop := 28 < n ∧ n ≤ 31

theorem sum_of_numbers_in_range :
  (∑ n in Finset.filter is_in_range (Finset.range 32), n) = 90 :=
by
  sorry

end sum_of_numbers_in_range_l281_281646


namespace Xiaoliang_catches_up_in_h_l281_281309

-- Define the speeds and head start
def speed_Xiaobin : ℝ := 4  -- Xiaobin's speed in km/h
def speed_Xiaoliang : ℝ := 12  -- Xiaoliang's speed in km/h
def head_start : ℝ := 6  -- Xiaobin's head start in hours

-- Define the additional distance Xiaoliang needs to cover
def additional_distance : ℝ := speed_Xiaobin * head_start

-- Define the hourly distance difference between them
def speed_difference : ℝ := speed_Xiaoliang - speed_Xiaobin

-- Prove that Xiaoliang will catch up with Xiaobin in exactly 3 hours
theorem Xiaoliang_catches_up_in_h : (additional_distance / speed_difference) = 3 :=
by
  sorry

end Xiaoliang_catches_up_in_h_l281_281309


namespace part1_part2_part3_l281_281456

noncomputable def z := (1 : ℂ) + (1 : ℂ) * complex.I

theorem part1 (m : ℝ) (h : z * (m + 2 * complex.I) = (0 + (m + 2) * complex.I)) : m = 2 :=
sorry

theorem part2 (z1 : ℂ) (h1 : z1 = -1 + complex.I) : z1.re = -1 :=
sorry

theorem part3 (a b : ℝ) (z2 : ℂ) (h2 : z^2 + a * z + b = 1 - complex.I) (h3 : z2 = a + b * complex.I) : complex.abs z2 = 5 :=
sorry

end part1_part2_part3_l281_281456


namespace eval_g_six_times_at_2_l281_281542

def g (x : ℝ) : ℝ := x^2 - 4 * x + 4

theorem eval_g_six_times_at_2 : g (g (g (g (g (g 2))))) = 4 := sorry

end eval_g_six_times_at_2_l281_281542


namespace lowest_degree_required_l281_281274

noncomputable def smallest_degree_poly (b : ℤ) : ℕ :=
  if h : ∃ P : Polynomial ℝ, (∀ x, (P.eval x ≠ b)) ∧
    (∃ y, (P.eval y > b)) ∧ (∃ z, (P.eval z < b)) 
  then Nat.find h 
  else 0

theorem lowest_degree_required :
  ∃ b : ℤ, smallest_degree_poly b = 4 :=
by
  -- b is some integer that fits the described conditions
  use 0
  sorry

end lowest_degree_required_l281_281274


namespace general_solution_particular_solution1_valid_particular_solution2_valid_initial_condition1_initial_condition2_l281_281523

noncomputable def particular_solution1 (x : ℝ) : ℝ :=
  (Real.log (abs x) + 3) / x

noncomputable def particular_solution2 (x : ℝ) : ℝ :=
  (Real.log (abs x) + (5 * Real.exp 1 - 1)) / x

theorem general_solution (C x : ℝ) (hx : x ≠ 0) :
  let y := (Real.log (abs x) + C) / x in
  x^2 * deriv y x + x * y = 1 := by
sorry

theorem particular_solution1_valid (x : ℝ) (hx : x ≠ 0) :
  x ≠ 0 →
  let y := particular_solution1 x in
  x^2 * deriv y x + x * y = 1 := by
sorry

theorem particular_solution2_valid (x : ℝ) (hx : x ≠ 0) :
  x ≠ 0 →
  let y := particular_solution2 x in
  x^2 * deriv y x + x * y = 1 := by
sorry

theorem initial_condition1 : particular_solution1 1 = 3 := by
sorry

theorem initial_condition2 : particular_solution2 (-Real.exp 1) = -5 := by
sorry

end general_solution_particular_solution1_valid_particular_solution2_valid_initial_condition1_initial_condition2_l281_281523


namespace two_mathematicians_contemporaries_l281_281250

def contemporaries_probability :=
  let total_area := 600 * 600
  let triangle_area := 1/2 * 480 * 480
  let non_contemporaneous_area := 2 * triangle_area
  let contemporaneous_area := total_area - non_contemporaneous_area
  let probability := contemporaneous_area / total_area
  probability

theorem two_mathematicians_contemporaries :
  contemporaries_probability = 9 / 25 :=
by
  -- Skipping the intermediate proof steps
  sorry

end two_mathematicians_contemporaries_l281_281250


namespace best_actions_to_take_l281_281516

variable (entertainment_and_vulgarity : Prop)
def option_1 := "Choose cultural products with personality and new trends"
def option_2 := "Improve our ability to distinguish cultures of different natures"
def option_3 := "Enhance our own cultural aesthetic taste"
def option_4 := "Pursue high-end cultural life"

theorem best_actions_to_take (h : entertainment_and_vulgarity) :
  (option_2 ∧ option_3) = "The best actions for citizens to take are to improve our ability to distinguish cultures of different natures and enhance our own cultural aesthetic taste." :=
sorry

end best_actions_to_take_l281_281516


namespace infinite_solutions_abs_eq_l281_281996

theorem infinite_solutions_abs_eq (x : ℝ) : 
  (abs (x - 5) + x - 5 = 0) → (∃ (n : ℕ), ∃ (m > n), ∃ (y : ℝ), abs (y - 5) + y - 5 = 0) :=
by 
sorrry

end infinite_solutions_abs_eq_l281_281996


namespace minimum_k_exists_l281_281168

theorem minimum_k_exists (k : ℕ) (h : k > 0) :
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 →
    k * (a * b + b * c + c * a) > 5 * (a^2 + b^2 + c^2) →
    a + b > c ∧ a + c > b ∧ b + c > a) ↔ k = 6 :=
sorry

end minimum_k_exists_l281_281168


namespace order_of_a_b_c_l281_281041

noncomputable def a : ℝ := 2
noncomputable def b : ℝ := real.log 3 / real.log 2
noncomputable def c : ℝ := real.log 2 / real.log 3

theorem order_of_a_b_c : a > b ∧ b > c :=
by
  have ha : a = 2 := rfl
  have hb : b = real.log 3 / real.log 2 := rfl
  have hc : c = real.log 2 / real.log 3 := rfl
  sorry

end order_of_a_b_c_l281_281041


namespace sin_product_of_cos_roots_l281_281042

open Real 

theorem sin_product_of_cos_roots (α β : ℝ) (hα : 0 < α ∧ α < π) (hβ : 0 < β ∧ β < π)
  (h_cos_roots : ∀ c, Polynomial.eval (Polynomial.mk [5, -3, -1]) c = 0 ↔ c = cos α ∨ c = cos β) :
  sin α * sin β = (√7) / 5 := 
sorry

end sin_product_of_cos_roots_l281_281042


namespace max_cards_with_conditions_l281_281636

theorem max_cards_with_conditions :
  ∃ (cards : Finset (Finset ℕ)), 
    (∀ card ∈ cards, card.card = 4) ∧ 
    (∀ c1 c2 ∈ cards, c1 ≠ c2 → (Finset.card (c1 ∩ c2)) = 1) ∧ 
    Finset.card cards = 9 :=
sorry

end max_cards_with_conditions_l281_281636


namespace greatest_common_length_cords_l281_281560

theorem greatest_common_length_cords (l1 l2 l3 l4 : ℝ) (h1 : l1 = Real.sqrt 20) (h2 : l2 = Real.pi) (h3 : l3 = Real.exp 1) (h4 : l4 = Real.sqrt 98) : 
  ∃ d : ℝ, d = 1 ∧ (∀ k1 k2 k3 k4 : ℝ, k1 * d = l1 → k2 * d = l2 → k3 * d = l3 → k4 * d = l4 → ∀i : ℝ, i = d) :=
by
  sorry

end greatest_common_length_cords_l281_281560


namespace sub_square_divisible_by_1391_l281_281192

-- Given conditions
variable (n : ℕ)
variable (A : matrix (fin n) (fin n) ℕ)

-- Lean statement of the proof problem
theorem sub_square_divisible_by_1391 (H_large_n : n > 10000) : 
  ∃ i j k, (i + k < n) ∧ (j + k < n) ∧
            (∑ x in (finset.range k).product (finset.range k), A (i + x.1) (j + x.2)) % 1391 = 0 :=
sorry

end sub_square_divisible_by_1391_l281_281192


namespace perimeter_of_figure_l281_281246

noncomputable def triangle_side_length (A B C : Point) (h_equilateral : is_equilateral_triangle A B C) : ℝ := sorry

noncomputable def midpoint (A B : Point) : Point := sorry

theorem perimeter_of_figure
  (A B C D E F G H : Point)
  (hABC_equilateral : is_equilateral_triangle A B C)
  (hADE_equilateral : is_equilateral_triangle A D E)
  (hEFG_equilateral : is_equilateral_triangle E F G)
  (hD_midpoint : midpoint A C = D)
  (hH_midpoint : midpoint A E = H)
  (hAB_length : dist A B = 6) :
  perimeter [A, B, C, D, E, H, G] = 22.5 :=
by
  sorry

end perimeter_of_figure_l281_281246


namespace factor_expression_l281_281427

theorem factor_expression (x : ℝ) : 3 * x^2 - 75 = 3 * (x + 5) * (x - 5) := 
by 
  sorry

end factor_expression_l281_281427


namespace tan_theta_sub_9pi_l281_281039

theorem tan_theta_sub_9pi (θ : ℝ) (h : Real.cos (Real.pi + θ) = -1 / 2) : 
  Real.tan (θ - 9 * Real.pi) = Real.sqrt 3 :=
by
  sorry

end tan_theta_sub_9pi_l281_281039


namespace percentage_increase_from_March_to_January_l281_281747

variable {F J M : ℝ}

def JanuaryCondition (F J : ℝ) : Prop :=
  J = 0.90 * F

def MarchCondition (F M : ℝ) : Prop :=
  M = 0.75 * F

theorem percentage_increase_from_March_to_January (F J M : ℝ) (h1 : JanuaryCondition F J) (h2 : MarchCondition F M) :
  (J / M) = 1.20 := by 
  sorry

end percentage_increase_from_March_to_January_l281_281747


namespace correct_survey_method_l281_281307

def service_life_of_light_tubes (survey_method : String) : Prop :=
  survey_method = "comprehensive"

def viewership_rate_of_spring_festival_gala (survey_method : String) : Prop :=
  survey_method = "comprehensive"

def crash_resistance_of_cars (survey_method : String) : Prop :=
  survey_method = "sample"

def fastest_student_for_sports_meeting (survey_method : String) : Prop :=
  survey_method = "sample"

theorem correct_survey_method :
  ¬(service_life_of_light_tubes "comprehensive") ∧
  ¬(viewership_rate_of_spring_festival_gala "comprehensive") ∧
  ¬(crash_resistance_of_cars "sample") ∧
  (fastest_student_for_sports_meeting "sample") :=
sorry

end correct_survey_method_l281_281307


namespace varphi_is_inner_product_psi_is_inner_product_theta_is_not_inner_product_l281_281968

variable (n : ℕ)
variable (a b : Fin n → ℝ)
variable (c : ℝ)
variable (a1 a2 b1 b2 : ℝ)

noncomputable def varphi (a b : Fin n → ℝ) : ℝ := ∑ i in Finset.range n, (i + 1 : ℝ) ^ 2 * a i * b i
noncomputable def psi (a b : Fin n → ℝ) : ℝ := varphi n a b + (a 0 + a 1) * (b 0 + b 1)
noncomputable def theta (a b : Fin n → ℝ) : ℝ := ∑ i in Finset.range n, ∑ j in Finset.range n, if i ≠ j then a i * b j else 0

theorem varphi_is_inner_product : 
  (∀ a b, varphi n a b = varphi n b a) ∧ 
  (∀ a b c, varphi n (fun i ↦ a i + c * b i) b = varphi n a b + c * varphi n b b) ∧ 
  (∀ a, varphi n a a ≥ 0 ∧ (varphi n a a = 0 ↔ ∀ i, a i = 0)) :=
by sorry

theorem psi_is_inner_product : 
  (∀ a b, psi n a b = psi n b a) ∧ 
  (∀ a b c, psi n (fun i ↦ a i + c * b i) b = psi n a b + c * psi n b b) ∧ 
  (∀ a, psi n a a ≥ 0 ∧ (psi n a a = 0 ↔ ∀ i, a i = 0)) :=
by sorry

theorem theta_is_not_inner_product : 
  ¬(∀ a, theta n a a ≥ 0 ∧ (theta n a a = 0 ↔ ∀ i, a i = 0)) :=
by sorry

end varphi_is_inner_product_psi_is_inner_product_theta_is_not_inner_product_l281_281968


namespace quadratic_expression_value_l281_281598

theorem quadratic_expression_value (a b : ℝ) (h₁ : a ≠ 0) (h₂ : a + b - 1 = 1) : (1 - a - b) = -1 :=
sorry

end quadratic_expression_value_l281_281598


namespace valera_coins_l281_281873

def coins (x : ℕ) (y : ℕ) : bool := 
  y > x ∧ 
  (15 * x + 20 * y) % 5 = 0 ∧ 
  (15 * 0 + 20 * 2 = 40 ∨ 15 * 1 + 20 * 1 = 35 ∨ 15 * 2 + 20 * 0 = 30) ∧
  (let total := 15 * x + 20 * y in 
  let remaining := (4 * total / 5 - (if 15 * 0 + 20 * 2 = 40 then 40 else if 15 * 1 + 20 * 1 = 35 then 35 else 30)) / 2 in 
  remaining ≤ 60)

theorem valera_coins : coins 2 6 = true := 
 by sorry

end valera_coins_l281_281873


namespace octahedron_plane_pairs_l281_281072

structure Octahedron where
  edges : Finset (Fin 12)

def edge_pairs_forming_plane (o : Octahedron) : Nat :=
  Finset.card ((o.edges.product o.edges).filter (λ p, p.1 ≠ p.2))

theorem octahedron_plane_pairs :
  ∀ (o : Octahedron), edge_pairs_forming_plane o = 66 := by
  sorry

end octahedron_plane_pairs_l281_281072


namespace hyperbola_eccentricity_l281_281810

/-- Given a hyperbola with the equation x^2/a^2 - y^2/b^2 = 1, point B(0, b),
the line F1B intersects with the two asymptotes at points P and Q. 
We are given that vector QP = 4 * vector PF1. Prove that the eccentricity 
of the hyperbola is 3/2. -/
theorem hyperbola_eccentricity (a b : ℝ) (h_a : a > 0) (h_b : b > 0) 
  (F1 : ℝ × ℝ) (B : ℝ × ℝ) (P Q : ℝ × ℝ) 
  (h_F1 : F1 = (-c, 0)) (h_B : B = (0, b)) 
  (h_int_P : P = (-a * c / (c + a), b * c / (c + a)))
  (h_int_Q : Q = (a * c / (c - a), b * c / (c - a)))
  (h_vec : (Q.1 - P.1, Q.2 - P.2) = (4 * (P.1 - F1.1), 4 * (P.2 - F1.2))) :
  (eccentricity : ℝ) = 3 / 2 :=
sorry

end hyperbola_eccentricity_l281_281810


namespace cos_squared_sum_eq_3_div_2_l281_281474

variable (A B C : ℝ)

theorem cos_squared_sum_eq_3_div_2
  (h1 : cos A + cos B + cos C = 0)
  (h2 : sin A + sin B + sin C = 0) :
  cos A ^ 2 + cos B ^ 2 + cos C ^ 2 = 3 / 2 :=
by sorry

end cos_squared_sum_eq_3_div_2_l281_281474


namespace b_n1_bound_l281_281061

noncomputable def f (a : Finₓ (n + 1) → ℝ) (x : ℝ) : ℝ :=
  ∑ i in Finₓ.range (n + 1), a i * x^i

noncomputable def f_squared (a : Finₓ (n + 1) → ℝ) (x : ℝ) : ℝ :=
  (f a x)^2

theorem b_n1_bound (a : Finₓ (n + 1) → ℝ) (h : ∀ i, 0 ≤ a i ∧ a i ≤ a 0) :
  let b := λ i, (f_squared a x)^[i]
  b (n + 1) ≤ (1 / 2) * (f a 1)^2 :=
by
  sorry

end b_n1_bound_l281_281061


namespace price_correct_l281_281180

noncomputable def price_per_glass_on_second_day 
  (O : ℝ) 
  (price_first_day : ℝ) 
  (revenue_equal : 2 * O * price_first_day = 3 * O * P) 
  : ℝ := 0.40

theorem price_correct 
  (O : ℝ) 
  (price_first_day : ℝ) 
  (revenue_equal : 2 * O * price_first_day = 3 * O * 0.40) 
  : price_per_glass_on_second_day O price_first_day revenue_equal = 0.40 := 
by 
  sorry

end price_correct_l281_281180


namespace money_spent_l281_281626

theorem money_spent (initial_amount spent left : ℕ) (h_initial : initial_amount = 78) (h_left : left = 63) :
    spent = initial_amount - left :=
by
  sorry

example : ∃ spent, spent = 15 :=
begin
  use 15,
  have h_initial : 78 = 78 := rfl,
  have h_left : 63 = 63 := rfl,
  exact money_spent 78 15 63 h_initial h_left,
end

end money_spent_l281_281626


namespace quadratic_inequality_solution_sets_l281_281678

noncomputable def quadratic_function : ℝ → ℝ
| x => x^2 - 3*x - 3

theorem quadratic_inequality_solution_sets (a : ℝ) :
  a < 0 → {x : ℝ | quadratic_function x - (a + 3) * x + 3 < 0} = {x | x < 3 / a ∨ x > 1} ∧
  0 < a ∧ a < 3 → {x : ℝ | quadratic_function x - (a + 3) * x + 3 < 0} = {x | 1 < x ∧ x < 3 / a} ∧
  a = 3 → {x : ℝ | quadratic_function x - (a + 3) * x + 3 < 0} = ∅ ∧
  a > 3 → {x : ℝ | quadratic_function x - (a + 3) * x + 3 < 0} = {x | 3 / a < x ∧ x < 1} :=
begin
  sorry,
end

end quadratic_inequality_solution_sets_l281_281678


namespace min_value_3x_plus_y_l281_281831

theorem min_value_3x_plus_y 
  (x y : ℝ)
  (h1 : x > max (-3) y) 
  (h2 : (x + 3) * (x^2 - y^2) = 8) :
  ∃ x y : ℝ, 3 * x + y = 4 * real.sqrt 6 - 6 :=
sorry

end min_value_3x_plus_y_l281_281831


namespace triangle_medians_slope_l281_281135

theorem triangle_medians_slope (x y m : ℝ) 
  (h1 : ∃ a b c d : ℝ, right_triangle a b c d ∧ legs_parallel_x_y_axes a b c d)
  (h2 : ∀ a b c d : ℝ, (right_triangle a b c d ∧ legs_parallel_x_y_axes a b c d) →
  ((median_slope a b c d = 5 ∨ median_slope a b c d = 5) ∧ (median_slope a b c d = 5 ∨ median_slope a b c d = m))): 
  ∃ m₁ m₂ : ℝ, m₁ ≠ m₂ ∧ ((m₁ = 20 ∧ m₂ = 5/4) ∨ (m₁ = 5/4 ∧ m₂ = 20)) :=
by
  sorry

end triangle_medians_slope_l281_281135


namespace greatest_large_chips_l281_281622

def is_composite (n : ℕ) : Prop := ∃ a b : ℕ, 2 ≤ a ∧ 2 ≤ b ∧ n = a * b

theorem greatest_large_chips (s l : ℕ) (c : ℕ) (hc : is_composite c) (h : s + l = 60) (hs : s = l + c) :
  l ≤ 28 :=
sorry

end greatest_large_chips_l281_281622


namespace factor_expression_l281_281428

theorem factor_expression (x : ℝ) : 3 * x^2 - 75 = 3 * (x + 5) * (x - 5) := 
by 
  sorry

end factor_expression_l281_281428


namespace range_of_k_l281_281836

theorem range_of_k {x k : ℝ} :
  (∀ x, ((x - 2) * (x + 1) > 0) → ((2 * x + 7) * (x + k) < 0)) →
  (x = -3 ∨ x = -2) → 
  -3 ≤ k ∧ k < 2 :=
sorry

end range_of_k_l281_281836


namespace range_of_a_l281_281059

theorem range_of_a (a : ℝ) : 
  (∀ x : ℤ, x + 2 > 0 ∧ x - a ≤ 0 ↔ x ∈ {-1, 0, 1, 2}) ↔ (2 ≤ a ∧ a < 3) :=
by
  sorry

end range_of_a_l281_281059


namespace crayons_selection_l281_281235

theorem crayons_selection (n k : ℕ) (h_n : n = 15) (h_k : k = 5) :
  Nat.choose n k = 3003 := 
by
  rw [h_n, h_k]
  rfl

end crayons_selection_l281_281235


namespace hyperbola_asymptote_l281_281418

theorem hyperbola_asymptote (y x : ℝ) :
  (y^2 / 9 - x^2 / 16 = 1) → (y = x * 3 / 4 ∨ y = -x * 3 / 4) :=
sorry

end hyperbola_asymptote_l281_281418


namespace lean_proof_l281_281489

theorem lean_proof (a : ℝ) (h : a = real.cos (2 * real.pi / 7)) : 2^(a - 1/2) < 2 * a :=
sorry

end lean_proof_l281_281489


namespace squared_difference_l281_281081

variable {x y : ℝ}

theorem squared_difference (h1 : (x + y)^2 = 81) (h2 : x * y = 18) :
  (x - y)^2 = 9 :=
by
  sorry

end squared_difference_l281_281081


namespace problem_a_problem_b_l281_281675

theorem problem_a (n : ℕ) : 
  (1 + ∑ k in finset.range n, 1 / (k + 1)^2) < 2 :=
sorry

theorem problem_b (n : ℕ) : 
  (∑ k in finset.range n, 1 / (k + 2)^2) < 3 / 4 :=
sorry

end problem_a_problem_b_l281_281675


namespace trig_identity_l281_281394

theorem trig_identity : 
  cos (real.pi / 9) * cos (2 * real.pi / 9) - sin (real.pi / 9) * sin (2 * real.pi / 9) = 1 / 2 := 
begin
  sorry
end

end trig_identity_l281_281394


namespace negate_proposition_l281_281805

theorem negate_proposition :
  (¬ ∃ (x₀ : ℝ), x₀^2 + 2 * x₀ + 3 ≤ 0) ↔ (∀ (x : ℝ), x^2 + 2 * x + 3 > 0) :=
by
  sorry

end negate_proposition_l281_281805


namespace slope_angle_range_l281_281103

theorem slope_angle_range (k : ℝ) (x y : ℝ) :
  (∃ x y, y = k * x - real.sqrt 3 ∧ x + y - 3 = 0 ∧ x > 0 ∧ y > 0) →
  ∃ θ : ℝ, θ = real.arctan k ∧ (π / 6) < θ ∧ θ < (π / 2) :=
by
  sorry

end slope_angle_range_l281_281103


namespace solve_exponential_l281_281975

theorem solve_exponential (x : ℝ) : 5^(x + 5) = 125^(x + 1) ↔ x = 1 := by
  sorry

end solve_exponential_l281_281975


namespace sum_of_digits_1_to_9999_l281_281371

theorem sum_of_digits_1_to_9999 : (∑ n in Finset.range 10000, (n.digits 10).sum) = 194445 := by
  sorry

end sum_of_digits_1_to_9999_l281_281371


namespace find_difference_of_squares_l281_281085

variable (x y : ℝ)
variable (h1 : (x + y) ^ 2 = 81)
variable (h2 : x * y = 18)

theorem find_difference_of_squares : (x - y) ^ 2 = 9 := by
  sorry

end find_difference_of_squares_l281_281085


namespace problem_statement_l281_281679

noncomputable def y : ℝ := (2 + √3 / (2 + √3 / (2 + ...))) -- For the continuation symbol

theorem problem_statement:
  let y := (2 + √3 / (2 + √3 / (2 + (by sorry : ℝ)))) in
  ∃ A B C : ℤ, 
    (B % (2*2) ≠ 0) ∧
    (y^2 - 2 * y = √3) ∧
    (1 / ((y + 2) * (y - 3)) = (A + √B) / C) ∧
    (abs A + abs B + abs C = 30) :=
by sorry

end problem_statement_l281_281679


namespace find_divisor_l281_281650

theorem find_divisor :
  ∃ d : ℕ, (4499 + 1) % d = 0 ∧ d = 2 :=
by
  sorry

end find_divisor_l281_281650


namespace exists_k_such_that_n_eq_k_2010_l281_281446

theorem exists_k_such_that_n_eq_k_2010 (m n : ℕ) (h_pos_m : 0 < m) (h_pos_n : 0 < n)
  (h : m * n ∣ m ^ 2010 + n ^ 2010 + n) : ∃ k : ℕ, 0 < k ∧ n = k ^ 2010 := by
  sorry

end exists_k_such_that_n_eq_k_2010_l281_281446


namespace xiao_si_test_paper_l281_281992

def opposite_number (x y : ℤ) : Prop :=
  x + y = 0

def absolute_value (x : ℤ) : ℤ :=
  if x < 0 then -x else x

def difference (a b c d : ℤ) : ℤ :=
  max (max a b) (max c d) - min (min a b) (min c d)

def compare_power (x y : ℤ) : Prop :=
  x > y

def reciprocal (x : ℚ) : ℚ :=
  1 / x

def opposite (x : ℤ) : ℤ :=
  -x

def xiao_si_score (score : ℕ) : Prop :=
  let q1 := opposite_number (-9) (-(-9))
  let q2 := absolute_value 2019 = 2019
  let q3 := difference (-2.5 : ℤ) (-3 : ℤ) 0 1 = 4 
  let q4 := compare_power (2^3) (3^2)
  let q5 := let rec := reciprocal (-1/2) in
              rec = opposite (-6 + 4)
  (q1 = true) ∧ (q2 = true) ∧ (q3 = false) ∧ (q4 = false) ∧ (q5 = true) ∧ (score = 20)

theorem xiao_si_test_paper : xiao_si_score 20 :=
  by
    sorry

end xiao_si_test_paper_l281_281992


namespace lowest_degree_required_l281_281270

noncomputable def smallest_degree_poly (b : ℤ) : ℕ :=
  if h : ∃ P : Polynomial ℝ, (∀ x, (P.eval x ≠ b)) ∧
    (∃ y, (P.eval y > b)) ∧ (∃ z, (P.eval z < b)) 
  then Nat.find h 
  else 0

theorem lowest_degree_required :
  ∃ b : ℤ, smallest_degree_poly b = 4 :=
by
  -- b is some integer that fits the described conditions
  use 0
  sorry

end lowest_degree_required_l281_281270


namespace lowest_degree_polynomial_l281_281293

-- Define the conditions
def polynomial_conditions (P : ℕ → ℤ) (b : ℤ): Prop :=
  (∃ c, c > b ∧ c ∈ set.range P) ∧ (∃ d, d < b ∧ d ∈ set.range P) ∧ (b ∉ set.range P)

-- The main statement
theorem lowest_degree_polynomial : ∃ P : ℕ → ℤ, polynomial_conditions P 4 ∧ (∀ Q : ℕ → ℤ, polynomial_conditions Q 4 → degree Q >= 4) :=
sorry

end lowest_degree_polynomial_l281_281293


namespace cannot_be_sum_of_five_consecutive_odd_integers_l281_281655

theorem cannot_be_sum_of_five_consecutive_odd_integers (S : ℤ) (h : S = 150) :
  ¬ ∃ n : ℤ, (n % 2 = 1) ∧ (S = 5 * n + 20) :=
by
  intros n h1 h2
  rw [h] at h2
  have h3 := (h2 - 20) / 5
  have h4 : h3 = n
  contradiction

end cannot_be_sum_of_five_consecutive_odd_integers_l281_281655


namespace ellipse_line_intersection_l281_281459

-- Definitions of the conditions in the Lean 4 language
def ellipse_eq (x y : ℝ) : Prop := (x^2 / 4) + (y^2 / 2) = 1

def midpoint_eq (x1 y1 x2 y2 : ℝ) : Prop := (x1 + x2 = 1) ∧ (y1 + y2 = -2)

-- The problem statement
theorem ellipse_line_intersection :
  (∃ (l : ℝ → ℝ → Prop),
  (∀ x1 y1 x2 y2 : ℝ, ellipse_eq x1 y1 → ellipse_eq x2 y2 → midpoint_eq x1 y1 x2 y2 →
     l x1 y1 ∧ l x2 y2) ∧
  (∀ x y : ℝ, l x y → (x - 4 * y - 9 / 2 = 0))) :=
sorry

end ellipse_line_intersection_l281_281459


namespace angle_of_inclination_y_eq_sqrt3x_plus_3_l281_281259

theorem angle_of_inclination_y_eq_sqrt3x_plus_3 :
  ∃ α : ℝ, 0 < α ∧ α < π ∧ tan α = √3 ∧ α = π / 3 := by
  sorry

end angle_of_inclination_y_eq_sqrt3x_plus_3_l281_281259


namespace vanya_age_l281_281304

variables (V F S : ℕ)

def condition_1 : Prop := V = (1 / 3 : ℝ) * F
def condition_2 : Prop := V = 3 * S
def condition_3 : Prop := F = S + 40

theorem vanya_age : condition_1 ∧ condition_2 ∧ condition_3 → V = 15 := 
by 
  sorry

end vanya_age_l281_281304


namespace pqr_value_l281_281914

noncomputable def complex_numbers (p q r : ℂ) := p * q + 5 * q = -20 ∧ q * r + 5 * r = -20 ∧ r * p + 5 * p = -20

theorem pqr_value (p q r : ℂ) (h : complex_numbers p q r) : p * q * r = 80 := by
  sorry

end pqr_value_l281_281914


namespace y_range_l281_281760

-- The definition of y
def y (b : Fin 15 → ℝ) : ℝ := ∑ i, b i / 5^(i+1)

-- The strategy here is to define the possible ranges for the function y under the given constraints
theorem y_range (b : Fin 15 → ℝ) :
  (∀ i, b i = 0 ∨ b i = 3) →
  (0 ≤ y b ∧ y b < 3/20) ∨ (3/5 ≤ y b ∧ y b < 9/10) :=
sorry

end y_range_l281_281760


namespace train_length_l281_281729

theorem train_length
  (V L : ℝ)
  (h1 : L = V * 18)
  (h2 : L + 350 = V * 39) :
  L = 300 := 
by
  sorry

end train_length_l281_281729


namespace total_pens_is_50_l281_281864

theorem total_pens_is_50
  (red : ℕ) (black : ℕ) (blue : ℕ) (green : ℕ) (purple : ℕ) (total : ℕ)
  (h1 : red = 8)
  (h2 : black = 3 / 2 * red)
  (h3 : blue = black + 5 ∧ blue = 1 / 5 * total)
  (h4 : green = blue / 2)
  (h5 : purple = 5)
  : total = red + black + blue + green + purple := sorry

end total_pens_is_50_l281_281864


namespace perpendicular_midlines_l281_281958

noncomputable def circle {α : Type*} [metric_space α] (O : α) (r : ℝ) : set α :=
{ P | dist P O = r }

def midpoint {α : Type*} [metric_space α] (A B : α) : α :=
classical.some (real.exists_point_on_segment_of_segment_closed_ball (dist A B / 2) (metric.closed_ball_mem_closed_segment (dist_nonneg : dist A B ≥ 0)))

theorem perpendicular_midlines {α : Type*} [metric_space α] [inner_product_space ℝ α]
  (O A B M : α) (r : ℝ) (hA : dist A O = r) (hB : dist B O = r) (hM : M ∈ arc A B ∩ circle O r)
  (C := midpoint A O) (D := midpoint B O) (E := midpoint B M) (F := midpoint M A) :
  inner (D - F) (C - E) = 0 :=
sorry

end perpendicular_midlines_l281_281958


namespace total_time_spent_l281_281527

def one_round_time : ℕ := 30
def saturday_initial_rounds : ℕ := 1
def saturday_additional_rounds : ℕ := 10
def sunday_rounds : ℕ := 15

theorem total_time_spent :
  one_round_time * (saturday_initial_rounds + saturday_additional_rounds + sunday_rounds) = 780 := by
  sorry

end total_time_spent_l281_281527


namespace painting_time_correct_l281_281492

-- Define Sally's and John's painting rates
def sally_rate := 1 / 4
def john_rate := 1 / 6

-- Define the combined rate
def combined_rate := sally_rate + john_rate

-- Define the total work to be done (painting one house)
def total_work := 1

-- The time it takes for both Sally and John to paint the house together
def painting_time := total_work / combined_rate

theorem painting_time_correct : painting_time = 2.4 :=
by
  sorry

end painting_time_correct_l281_281492


namespace knicks_equivalent_to_fifty_knocks_kracks_equivalent_to_knicks_equivalent_to_fifty_knocks_l281_281850

-- Definitions based on problem conditions
def conversion_knicks_knacks (k : ℝ) : ℝ := k * 3 / 5
def conversion_knacks_knocks (k : ℝ) : ℝ := k * 5 / 2
def conversion_knocks_kracks (k : ℝ) : ℝ := k / 4

-- Questions reformulated as Lean theorems
theorem knicks_equivalent_to_fifty_knocks :
  conversion_knicks_knacks (conversion_knacks_knocks 50) = 100 / 3 :=
by sorry

theorem kracks_equivalent_to_knicks_equivalent_to_fifty_knocks : 
  conversion_knocks_kracks (conversion_knacks_knocks ((100 / 3) * 5 / 3)) = 25 / 3 :=
by sorry

end knicks_equivalent_to_fifty_knocks_kracks_equivalent_to_knicks_equivalent_to_fifty_knocks_l281_281850


namespace broker_wealth_increase_after_two_years_l281_281322

theorem broker_wealth_increase_after_two_years :
  let initial_investment : ℝ := 100
  let first_year_increase : ℝ := 0.75
  let second_year_decrease : ℝ := 0.30
  let end_first_year := initial_investment * (1 + first_year_increase)
  let end_second_year := end_first_year * (1 - second_year_decrease)
  end_second_year - initial_investment = 22.50 :=
by
  sorry

end broker_wealth_increase_after_two_years_l281_281322


namespace find_m_l281_281592

noncomputable def conic_eccentricity (m : ℝ) (h : m < 0) : Prop :=
  let e := (Real.sqrt (1 - m)) in
  e = Real.sqrt 7

theorem find_m : ∃ m < 0, conic_eccentricity m ∧ m = -6 := by
  sorry

end find_m_l281_281592


namespace log_inequality_l281_281429

theorem log_inequality {x y : ℝ} (h1 : log x < log y) (h2 : log y < 0) : 0 < x ∧ x < y ∧ y < 1 :=
by
  -- proof (N/A)
  sorry

end log_inequality_l281_281429


namespace lowest_degree_is_4_l281_281282

noncomputable def lowest_degree_polynomial (P : ℝ → ℝ) : ℕ :=
  if ∃ b : ℤ, (∀ coeff ∈ (P.coefficients), coeff < (b : ℝ) ∨ coeff > (b : ℝ)) ∧ (¬ ∃ coeff ∈ (P.coefficients), coeff = (b : ℝ))
  then Polynomial.natDegree P
  else 0

theorem lowest_degree_is_4 : ∀ (P : Polynomial ℝ), 
  (∃ b : ℤ, (∀ coeff ∈ P.coefficients, coeff < (b : ℝ) ∨ coeff > (b : ℝ)) ∧ (¬ ∃ coeff ∈ P.coefficients, coeff = (b : ℝ)))
  → lowest_degree_polynomial P = 4 :=
by
  sorry

end lowest_degree_is_4_l281_281282


namespace intersection_of_complements_l281_281925

-- Define the universal set U as a natural set with numbers <= 8
def U : Set ℕ := { x | x ≤ 8 }

-- Define the set A
def A : Set ℕ := { 1, 3, 7 }

-- Define the set B
def B : Set ℕ := { 2, 3, 8 }

-- Prove the statement for the intersection of the complements of A and B with respect to U
theorem intersection_of_complements : 
  ((U \ A) ∩ (U \ B)) = ({ 0, 4, 5, 6 } : Set ℕ) :=
by
  sorry

end intersection_of_complements_l281_281925


namespace candle_height_l281_281781

variable (h d a b x : ℝ)

theorem candle_height (h d a b : ℝ) : x = h * (1 + d / (a + b)) :=
by
  sorry

end candle_height_l281_281781


namespace log_prob_correct_l281_281248

def log_is_integer_probability : ℚ :=
  let S := {3^n | n in finset.range (20 + 1)}
  let pairs := finset.filter (λ (p : ℕ × ℕ), p.1 ≠ p.2) (finset.product S S)
  let valid_pairs := finset.filter (λ (p : ℕ × ℕ), ∃ z : ℤ, p.2 = p.1 ^ z) pairs
  let total_pairs : ℚ := finset.card pairs
  let valid_pairs_count : ℚ := finset.card valid_pairs
  valid_pairs_count / total_pairs

theorem log_prob_correct : log_is_integer_probability = 21 / 95 :=
by
  unfold log_is_integer_probability
  sorry

end log_prob_correct_l281_281248


namespace cos_theta_max_sin_value_l281_281019

noncomputable def cos_theta_max_sin (alpha beta : ℝ) (h1 : alpha + beta = (2 * π) / 3) (h2 : alpha > 0) (h3 : beta > 0) : ℝ :=
  if h4 : ∃ θ, θ = alpha ∧ (sin alpha + 2 * sin beta) = max (sin alpha + 2 * sin beta) then 
    cos (classical.some h4)
  else
    0

theorem cos_theta_max_sin_value (alpha beta : ℝ) (h1 : alpha + beta = (2 * π) / 3) (h2 : alpha > 0) (h3 : beta > 0) :
  cos_theta_max_sin alpha beta h1 h2 h3 = (√21) / 7 :=
sorry

end cos_theta_max_sin_value_l281_281019


namespace Linda_original_savings_l281_281558

def original_savings : ℝ :=
  S : ℝ

theorem Linda_original_savings :
  let S := (original_savings)
  (3/8) * S + (1/4) * S + (450 : ℝ) = S →
  (90/100) * (3/8) * S + (85/100) * (1/4) * S + (450 : ℝ) = S →
  S = 1000 :=
sorry

end Linda_original_savings_l281_281558


namespace perimeter_ratio_l281_281969

variables (K T k R r : ℝ)

-- Conditions given in the problem
def condition1 : Prop := (r = 2 * T / K)
def condition2 : Prop := (2 * T = R * k)

-- The statement we want to prove
theorem perimeter_ratio :
  condition1 K T r →
  condition2 T R k →
  K / k = R / r :=
by
  intros h1 h2
  sorry

end perimeter_ratio_l281_281969


namespace hockey_team_helmets_l281_281710

theorem hockey_team_helmets (r b : ℕ) 
  (h1 : b = r - 6) 
  (h2 : r * 3 = b * 5) : 
  r + b = 24 :=
by
  sorry

end hockey_team_helmets_l281_281710


namespace part1_part2_l281_281827

-- Definitions based on the given conditions
def f (a : ℝ) (x : ℝ) : ℝ := a * (x^2 - 1) - log x
def f_prime (a : ℝ) (x : ℝ) : ℝ := 2 * a * x - 1 / x

-- Assuming the necessary domain constraints for x and a:
-- x > 0 for part 1, and 0 < a < 1/2 for part 2.

-- Part (1): Prove the extreme value of f at a = 1/2 is 0
theorem part1 (x : ℝ) (h : x > 0) :
  f (1/2) x = 1/2 * (x^2 - 1) - log x :=
sorry

-- Part (2): Prove f'(x0) < 1 - 2a for 0 < a < 1/2 and given x0 > 0 is the larger root of f(x0) = 0
theorem part2 (a : ℝ) (h1 : 0 < a) (h2 : a < 1/2) (x0 : ℝ) (hx0 : x0 > 1)
  (hx0_root : f a x0 = 0) :
  f_prime a x0 < 1 - 2 * a :=
sorry

end part1_part2_l281_281827


namespace tenly_more_stuffed_animals_than_kenley_l281_281936

def mckenna_stuffed_animals := 34
def kenley_stuffed_animals := 2 * mckenna_stuffed_animals
def total_stuffed_animals_all := 175
def total_stuffed_animals_mckenna_kenley := mckenna_stuffed_animals + kenley_stuffed_animals
def tenly_stuffed_animals := total_stuffed_animals_all - total_stuffed_animals_mckenna_kenley
def stuffed_animals_difference := tenly_stuffed_animals - kenley_stuffed_animals

theorem tenly_more_stuffed_animals_than_kenley :
  stuffed_animals_difference = 5 := by
  sorry

end tenly_more_stuffed_animals_than_kenley_l281_281936


namespace simplest_square_root_l281_281660

theorem simplest_square_root :
  let a := Real.sqrt (1 / 2)
  let b := Real.sqrt 11
  let c := Real.sqrt 27
  let d := Real.sqrt 0.3
  (b < a ∧ b < c ∧ b < d) :=
sorry

end simplest_square_root_l281_281660


namespace euler_line_parallel_l281_281981

theorem euler_line_parallel (a : ℝ) : 
  let A := (-3, 0) : ℝ × ℝ,
      B := (3, 0) : ℝ × ℝ,
      C := (3, 3) : ℝ × ℝ,
      G := ((-3 + 3 + 3) / 3, (0 + 0 + 3) / 3) : ℝ × ℝ,
      O := ((3 + 3) / 2, (0 + 3) / 2) : ℝ × ℝ,
      m_euler := (O.2 - G.2) / (O.1 - G.1),
      line_euler := λ x y, y - G.2 = m_euler * (x - G.1),
      line_l := λ x y, a * x + (a^2 - 3) * y - 9 = 0 in
  G = (1, 1) ∧
  O = (3, 3 / 2) ∧
  m_euler = 1 / 4 ∧
  (a / 1 = (a^2 - 3) / 2) →
  a = -1 := by
  intros
  sorry

end euler_line_parallel_l281_281981


namespace simplify_root_subtraction_l281_281204

axiom eight_cubed_root : 8^(1/3) = 2
axiom three_hundred_forty_three_cubed_root : 343^(1/3) = 7

theorem simplify_root_subtraction : 8^(1/3) - 343^(1/3) = -5 :=
by {
  rw [eight_cubed_root, three_hundred_forty_three_cubed_root],
  norm_num,
}

end simplify_root_subtraction_l281_281204


namespace square_diff_l281_281074

theorem square_diff (x y : ℝ) (h1 : (x + y)^2 = 81) (h2 : x * y = 18) : (x - y)^2 = 9 :=
by 
  sorry

end square_diff_l281_281074


namespace original_number_l281_281493

theorem original_number (x : ℕ) : x * 16 = 3408 → x = 213 := by
  intro h
  sorry

end original_number_l281_281493


namespace gcd_fibonacci_succ_fibonacci_gcd_l281_281547

-- Definitions for Fibonacci sequence
def fibonacci : ℕ → ℕ 
| 0 := 1
| 1 := 1
| (n+2) := fibonacci n + fibonacci (n + 1)

-- Part (a): Prove that GCD(F(n), F(n+1)) = 1 for all n
theorem gcd_fibonacci_succ (n : ℕ) : Nat.gcd (fibonacci n) (fibonacci (n + 1)) = 1 := 
  sorry

-- Part (b): Show that F(gcd(m,n)) = gcd(F(m), F(n)) given the assumption
theorem fibonacci_gcd (m n : ℕ) (h : ∀ n m : ℕ, fibonacci (n + m) = fibonacci m * fibonacci (n + 1) + fibonacci (m - 1) * fibonacci n) : 
  fibonacci (Nat.gcd m n) = Nat.gcd (fibonacci m) (fibonacci n) := 
  sorry

end gcd_fibonacci_succ_fibonacci_gcd_l281_281547


namespace lowest_degree_is_4_l281_281284

noncomputable def lowest_degree_polynomial (P : ℝ → ℝ) : ℕ :=
  if ∃ b : ℤ, (∀ coeff ∈ (P.coefficients), coeff < (b : ℝ) ∨ coeff > (b : ℝ)) ∧ (¬ ∃ coeff ∈ (P.coefficients), coeff = (b : ℝ))
  then Polynomial.natDegree P
  else 0

theorem lowest_degree_is_4 : ∀ (P : Polynomial ℝ), 
  (∃ b : ℤ, (∀ coeff ∈ P.coefficients, coeff < (b : ℝ) ∨ coeff > (b : ℝ)) ∧ (¬ ∃ coeff ∈ P.coefficients, coeff = (b : ℝ)))
  → lowest_degree_polynomial P = 4 :=
by
  sorry

end lowest_degree_is_4_l281_281284


namespace total_revenue_increased_by_60_percent_l281_281706

theorem total_revenue_increased_by_60_percent (P Q : ℝ) (h1 : P > 0) (h2 : Q > 0) :
  (P - 0.20 * P) * (2 * Q) = 1.60 * (P * Q) :=
by
  have P_new := P - 0.20 * P
  have Q_new := 2 * Q
  have revenue_new := P_new * Q_new
  have original_revenue := P * Q
  calc 
    revenue_new = 0.80 * P * (2 * Q) : by rw [P_new, Q_new]
            ... = 1.60 * (P * Q) : by ring

end total_revenue_increased_by_60_percent_l281_281706


namespace tangent_slope_eq_l281_281054

noncomputable def f (b : ℝ) (x : ℝ) : ℝ := b * x + Real.log x

theorem tangent_slope_eq (b k : ℝ) (h_tangent : ∃ m : ℝ, f b m = k * m ∧ (b + 1 / m) = k) : 
  k - b = 1 / Real.exp 1 :=
begin
  sorry
end

end tangent_slope_eq_l281_281054


namespace tanA_over_tanB_l281_281879

noncomputable def tan_ratios (A B C a b c : ℝ) : Prop :=
  a * Real.cos B - b * Real.cos A + 2 * c = 0

theorem tanA_over_tanB {A B C a b c : ℝ} (h : tan_ratios A B C a b c) : 
  Real.tan A / Real.tan B = -1 / 3 :=
by
  sorry

end tanA_over_tanB_l281_281879


namespace blocks_before_jess_turn_l281_281140

def blocks_at_start : Nat := 54
def players : Nat := 5
def rounds : Nat := 5
def father_removes_block_in_6th_round : Nat := 1

theorem blocks_before_jess_turn :
    (blocks_at_start - (players * rounds + father_removes_block_in_6th_round)) = 28 :=
by 
    sorry

end blocks_before_jess_turn_l281_281140


namespace rachel_more_than_adam_l281_281576

variable (R J A : ℕ)

def condition1 := R = 75
def condition2 := R = J - 6
def condition3 := R > A
def condition4 := (R + J + A) / 3 = 72

theorem rachel_more_than_adam
  (h1 : condition1 R)
  (h2 : condition2 R J)
  (h3 : condition3 R A)
  (h4 : condition4 R J A) : 
  R - A = 15 := 
by
  sorry

end rachel_more_than_adam_l281_281576


namespace segment_CD_length_l281_281638

theorem segment_CD_length
  (A B C D E F : Type)
  [metric_space A]
  [metric_space B]
  [metric_space C]
  [metric_space D]
  [metric_space E]
  [metric_space F]
  (AB : ℝ) (BC : ℝ) (AC : ℝ)
  (DE : ℝ) (EF : ℝ)
  (angle_CAB : ℝ) (angle_EDF : ℝ)
  (h_lengths : AB = 8 ∧ BC = 18 ∧ AC = 14)
  (h_lengths_prime : DE = 4 ∧ EF = 9)
  (h_angle : angle_CAB = 100 ∧ angle_EDF = 100) :
  (CD = 8) :=
by sorry

end segment_CD_length_l281_281638


namespace tim_total_score_l281_281723

-- Definitions from conditions
def single_line_points : ℕ := 1000
def tetris_points : ℕ := 8 * single_line_points
def doubled_tetris_points : ℕ := 2 * tetris_points
def num_singles : ℕ := 6
def num_tetrises : ℕ := 4
def consecutive_tetrises : ℕ := 2
def regular_tetrises : ℕ := num_tetrises - consecutive_tetrises

-- Total score calculation
def total_score : ℕ :=
  num_singles * single_line_points +
  regular_tetrises * tetris_points +
  consecutive_tetrises * doubled_tetris_points

-- Prove that Tim's total score is 54000
theorem tim_total_score : total_score = 54000 :=
by 
  sorry

end tim_total_score_l281_281723


namespace problem_l281_281909

theorem problem (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a^3 + b^3 + c^3 = a^7 + b^7 + c^7) :
  a^2 + b^2 + c^2 = 6 / 7 := 
sorry

end problem_l281_281909


namespace speed_is_90_l281_281386

namespace DrivingSpeedProof

/-- Given the observation times and marker numbers, prove the speed of the car is 90 km/hr. -/
theorem speed_is_90 
  (X Y : ℕ)
  (h0 : X ≥ 0) (h1 : X ≤ 9)
  (h2 : Y = 8 * X)
  (h3 : Y ≥ 0) (h4 : Y ≤ 9)
  (noon_marker : 10 * X + Y = 18)
  (second_marker : 10 * Y + X = 81)
  (third_marker : 100 * X + Y = 108)
  : 90 = 90 :=
by {
  sorry
}

end DrivingSpeedProof

end speed_is_90_l281_281386


namespace total_time_over_weekend_l281_281526

def time_per_round : ℕ := 30
def rounds_saturday : ℕ := 11
def rounds_sunday : ℕ := 15

theorem total_time_over_weekend :
  (rounds_saturday * time_per_round) + (rounds_sunday * time_per_round) = 780 :=
by
  -- This is where the proof would go, but it is omitted as per instructions.
  sorry

end total_time_over_weekend_l281_281526


namespace probability_equivalence_l281_281501

noncomputable def probability_of_black_ball (balls : ℕ) (total_balls : ℕ) : ℚ :=
  if total_balls = 0 then 0 else balls / total_balls

variables (p1 p10 : ℚ)
variables (black white : ℕ)

-- Conditions:
-- There is 1 black ball and 9 white balls in the box
def condition_black_white : Prop :=
  black = 1 ∧ white = 9

-- 10 people draw one ball each in turn and then put it back
def condition_draw_replace : Prop := true

-- Define P1 as the probability of the first person drawing a black ball
def P1 : ℚ := probability_of_black_ball black (black + white)

-- Define P10 as the probability of the tenth person drawing a black ball
def P10 : ℚ := probability_of_black_ball black (black + white)

-- Proof target: P10 = P1
theorem probability_equivalence (h1 : condition_black_white) (h2 : condition_draw_replace) : 
  P10 = P1 :=
by
  -- Since condition_draw_replace is always true, it does not affect the outcome here.
  rw [h1.left, h1.right]
  unfold P10 P1 probability_of_black_ball
  simp only [Nat.add_comm, Nat.cast_bit0, Nat.cast_add, Nat.cast_one]
  sorry

end probability_equivalence_l281_281501


namespace domain_of_inverse_of_exponential_l281_281591

theorem domain_of_inverse_of_exponential :
  ∀ (f : ℝ → ℝ), (∀ x, f x = 3^x) → (∀ x, 0 < x ∧ x ≤ 2) →
  set.Ioo 1 9 = { y | ∃ x, 0 < x ∧ x ≤ 2 ∧ y = 3^x } :=
begin
  intros f hf hd,
  sorry
end

end domain_of_inverse_of_exponential_l281_281591


namespace point_not_on_graph_l281_281663

def on_graph (x y : ℚ) : Prop := y = x / (x + 2)

/-- Let's state the main theorem -/
theorem point_not_on_graph : ¬ on_graph 2 (2 / 3) := by
  sorry

end point_not_on_graph_l281_281663


namespace Karl_selects_five_crayons_l281_281237

theorem Karl_selects_five_crayons : ∃ (k : ℕ), k = 3003 ∧ (finset.card (finset.powerset_len 5 (finset.range 15))).nat_abs = k :=
by
  -- existence proof of k = 3003 and showing that k equals the combination count
  sorry

end Karl_selects_five_crayons_l281_281237


namespace mixtilinear_incircle_radius_l281_281245
open Real

variable (AB BC AC : ℝ)
variable (r_A : ℝ)

def triangle_conditions : Prop :=
  AB = 65 ∧ BC = 33 ∧ AC = 56

theorem mixtilinear_incircle_radius 
  (h : triangle_conditions AB BC AC)
  : r_A = 12.89 := 
sorry

end mixtilinear_incircle_radius_l281_281245


namespace proposition_d_correct_l281_281658

theorem proposition_d_correct (a b c : ℝ) (h : a > b) : a - c > b - c := 
by
  sorry

end proposition_d_correct_l281_281658


namespace price_correct_l281_281181

noncomputable def price_per_glass_on_second_day 
  (O : ℝ) 
  (price_first_day : ℝ) 
  (revenue_equal : 2 * O * price_first_day = 3 * O * P) 
  : ℝ := 0.40

theorem price_correct 
  (O : ℝ) 
  (price_first_day : ℝ) 
  (revenue_equal : 2 * O * price_first_day = 3 * O * 0.40) 
  : price_per_glass_on_second_day O price_first_day revenue_equal = 0.40 := 
by 
  sorry

end price_correct_l281_281181


namespace odd_binomials_power_of_two_l281_281962

theorem odd_binomials_power_of_two (n : ℕ) (h_pos : 0 < n) : ∃ k : ℕ, 2^k = ((finset.range (n + 1)).filter (λ h, (nat.choose n h) % 2 = 1)).card :=
by sorry

end odd_binomials_power_of_two_l281_281962


namespace vanya_number_l281_281637

def S (n : ℕ) : ℕ := n.digits 10 |>.sum

theorem vanya_number:
  (2014 + S 2014 = 2021) ∧ (1996 + S 1996 = 2021) := by
  sorry

end vanya_number_l281_281637


namespace total_payment_correct_l281_281685

-- Define the initial conditions
def initial_bill : Real := 314.16
def number_of_people : Nat := 9
def smallest_unit : Real := 0.01 -- 1 cent

-- Define the function to round up to the nearest cent
def round_up_to_nearest_cent (amount : Real) : Real :=
  let rounded := (amount * 100).ceil / 100
  rounded

-- Define the per person payment
def per_person_payment (total_bill : Real) (people : Nat) : Real :=
  total_bill / people

-- Calculate the total payment after rounding
def total_payment (total_bill : Real) (people : Nat) : Real :=
  (round_up_to_nearest_cent (per_person_payment total_bill people)) * people

-- Proposition to prove
theorem total_payment_correct :
  total_payment initial_bill number_of_people = 314.19 := by
  sorry

end total_payment_correct_l281_281685


namespace sin_cos_add_identity_l281_281370

theorem sin_cos_add_identity : 
  sin (5 * (Real.pi / 180)) * cos (55 * (Real.pi / 180)) + 
  cos (5 * (Real.pi / 180)) * sin (55 * (Real.pi / 180)) = 
  sqrt 3 / 2 :=
sorry

end sin_cos_add_identity_l281_281370


namespace find_fx_l281_281449

variable {e : ℝ} {a : ℝ} (f : ℝ → ℝ)

def odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f x

variable (hodd : odd_function f)
variable (hdef : ∀ x, -e ≤ x → x < 0 → f x = a * x + Real.log (-x))

theorem find_fx (x : ℝ) (hx : 0 < x ∧ x ≤ e) : f x = a * x - Real.log x :=
by
  sorry

end find_fx_l281_281449


namespace part1_part2_l281_281926

def A (x : ℝ) : Prop := x < -2 ∨ x > 3
def B (a : ℝ) (x : ℝ) : Prop := 1 - a < x ∧ x < a + 3

theorem part1 (x : ℝ) : (¬A x ∨ B 1 x) ↔ -2 ≤ x ∧ x < 4 :=
by
  sorry

theorem part2 (a : ℝ) : (∀ x, ¬(A x ∧ B a x)) ↔ a ≤ 0 :=
by
  sorry

end part1_part2_l281_281926


namespace find_f_2_solve_inequality_f_m_minus_2_l281_281221

noncomputable def f : ℝ → ℝ := sorry

axiom f_decreasing : ∀ {x y : ℝ}, (0 < x) → (0 < y) → x < y → f x > f y
axiom f_additivity : ∀ {x y : ℝ}, (0 < x) → (0 < y) → f (x + y) = f x + f y - 1
axiom f_at_4 : f 4 = 5

-- (1) Find the value of f(2)
theorem find_f_2 : f 2 = 3 := sorry 

-- (2) Solve the inequality f(m-2) ≥ 3
theorem solve_inequality_f_m_minus_2 (m : ℝ) : f (m - 2) ≥ 3 ↔ 2 < m ∧ m ≤ 4 :=
begin
  sorry
end

end find_f_2_solve_inequality_f_m_minus_2_l281_281221


namespace lowest_degree_is_4_l281_281287

noncomputable def lowest_degree_polynomial (P : ℝ → ℝ) : ℕ :=
  if ∃ b : ℤ, (∀ coeff ∈ (P.coefficients), coeff < (b : ℝ) ∨ coeff > (b : ℝ)) ∧ (¬ ∃ coeff ∈ (P.coefficients), coeff = (b : ℝ))
  then Polynomial.natDegree P
  else 0

theorem lowest_degree_is_4 : ∀ (P : Polynomial ℝ), 
  (∃ b : ℤ, (∀ coeff ∈ P.coefficients, coeff < (b : ℝ) ∨ coeff > (b : ℝ)) ∧ (¬ ∃ coeff ∈ P.coefficients, coeff = (b : ℝ)))
  → lowest_degree_polynomial P = 4 :=
by
  sorry

end lowest_degree_is_4_l281_281287


namespace exponent_property_l281_281991

theorem exponent_property (a : ℝ) : a^7 = a^3 * a^4 :=
by
  -- The proof statement follows from the properties of exponents:
  -- a^m * a^n = a^(m + n)
  -- Therefore, a^3 * a^4 = a^(3 + 4) = a^7.
  sorry

end exponent_property_l281_281991


namespace magnitude_a_plus_2b_l281_281450

-- Definitions of vectors a and b
def a : EuclideanSpace ℝ (Fin 2) := ![2,0]
def b : EuclideanSpace ℝ (Fin 2) := ![(cos (π / 3)), (sin (π / 3))]  -- since |b|=1

-- Theorem to prove the magnitude of a + 2b
theorem magnitude_a_plus_2b : ∥a + (2 : ℝ) • b∥ = 2 * Real.sqrt 3 := by
  sorry

end magnitude_a_plus_2b_l281_281450


namespace same_color_probability_l281_281073

variables (red blue green total : ℕ)
variables (select_ways same_color_ways : ℕ)

-- Given conditions
def conditions (red blue green total select_ways same_color_ways : ℕ) : Prop :=
  red = 7 ∧ blue = 5 ∧ green = 3 ∧ total = red + blue + green ∧
  select_ways = total.choose 3 ∧
  same_color_ways = red.choose 3 + blue.choose 3 + green.choose 3

-- The probability that all three plates are the same color equals 46/455
theorem same_color_probability :
  conditions red blue green total select_ways same_color_ways → 
  (same_color_ways : ℚ) / select_ways = 46 / 455 :=
by
  intros h
  cases h with h_red h_rest
  cases h_rest with h_blue h_rest
  cases h_rest with h_green h_rest
  cases h_rest with h_total h_rest
  cases h_rest with h_select h_fav
  rw [h_red, h_blue, h_green, h_total, h_select, h_fav]
  sorry

end same_color_probability_l281_281073


namespace max_prime_factors_of_c_l281_281208

-- Definitions of conditions
variables (c d : ℕ)
variable (prime_factor_count : ℕ → ℕ)
variable (gcd : ℕ → ℕ → ℕ)
variable (lcm : ℕ → ℕ → ℕ)

-- Conditions
axiom gcd_condition : prime_factor_count (gcd c d) = 11
axiom lcm_condition : prime_factor_count (lcm c d) = 44
axiom fewer_prime_factors : prime_factor_count c < prime_factor_count d

-- Proof statement
theorem max_prime_factors_of_c : prime_factor_count c ≤ 27 := 
sorry

end max_prime_factors_of_c_l281_281208


namespace find_positive_x_l281_281848

theorem find_positive_x (x : ℝ) (hx_pos : 0 < x) (hx_sqrt_eq : sqrt (12 * x) * sqrt (6 * x) * sqrt (5 * x) * sqrt (20 * x) = 20) : 
  x = real.sqrt (real.sqrt (1 / 18)) :=
sorry

end find_positive_x_l281_281848


namespace city_A_highest_average_annual_percentage_increase_l281_281232

noncomputable def percentage_increase (initial final : ℕ) : ℚ :=
  ((final - initial) : ℚ) / initial * 100

noncomputable def average_percentage_increase (data : List (ℕ × ℕ)) : ℚ :=
  data.map (λ ⟨initial, final⟩ => percentage_increase initial final) |>.sum / data.length

theorem city_A_highest_average_annual_percentage_increase :
  let city_A_data := [(30000, 45000), (45000, 67500), (67500, 101250)]
  let city_B_data := [(55000, 66000), (66000, 79200), (79200, 95040)]
  let city_C_data := [(80000, 96000), (96000, 115200), (115200, 138240)]
  let city_D_data := [(50000, 70000), (70000, 98000), (98000, 137200)]
  let city_E_data := [(120000, 144000), (144000, 172800), (172800, 207360)]
  average_percentage_increase city_A_data > average_percentage_increase city_B_data ∧
  average_percentage_increase city_A_data > average_percentage_increase city_C_data ∧
  average_percentage_increase city_A_data > average_percentage_increase city_D_data ∧
  average_percentage_increase city_A_data > average_percentage_increase city_E_data :=
by
  sorry

end city_A_highest_average_annual_percentage_increase_l281_281232


namespace lowest_degree_is_4_l281_281280

noncomputable def lowest_degree_polynomial (P : Polynomial ℤ) (b : ℤ) : Prop :=
  ∃ (b : ℤ), 
    let A_P := P.support in
    (∀ (a ∈ A_P), a < b ∨ a > b) ∧ 
    (¬(b ∈ A_P)) ∧
    (∃ (a1 a2 : ℤ), a1 ∈ A_P ∧ a2 ∈ A_P ∧ a1 < b ∧ a2 > b)

theorem lowest_degree_is_4 :
  ∀ P : Polynomial ℤ, 
    let b := lowest_degree_polynomial P in
    b P 4 :=
sorry

end lowest_degree_is_4_l281_281280


namespace pump_time_l281_281687

-- Definitions used in the conditions
def basement_length : Real := 20
def basement_width : Real := 40
def water_depth_inch : Real := 24
def pump_rate : Real := 10
def pumps : Nat := 4
def cubic_foot_to_gallons : Real := 7.5

-- Convert inches to feet
def water_depth_feet := water_depth_inch / 12

-- Calculate the volume of water in cubic feet
def water_volume_cubic_feet := basement_length * basement_width * water_depth_feet

-- Convert the volume of water to gallons
def water_volume_gallons := water_volume_cubic_feet * cubic_foot_to_gallons

-- Calculate the total pumping rate
def total_pumping_rate := pumps * pump_rate

-- Prove the time required to pump out all the water
theorem pump_time : (water_volume_gallons / total_pumping_rate) = 300 :=
by
  sorry

end pump_time_l281_281687


namespace children_count_after_addition_l281_281565

theorem children_count_after_addition :
  ∀ (total_guests men guests children_added : ℕ),
    total_guests = 80 →
    men = 40 →
    guests = (men + men / 2) →
    children_added = 10 →
    total_guests - guests + children_added = 30 :=
by
  intros total_guests men guests children_added h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end children_count_after_addition_l281_281565


namespace triangle_is_right_l281_281030

variable (m n : ℝ) (m_gt_1 : m > 1) (n_gt_0 : n > 0)
variable (P : ℝ × ℝ)
variable (F1 F2 : ℝ × ℝ)
variable (intersection : P = (sqrt(m * cosh(2 * t)), sinh(2 * t)) for some t)

def ellipse (P : ℝ × ℝ) (m : ℝ) := P.1 ^ 2 / m + P.2 ^ 2 = 1

def hyperbola (P : ℝ × ℝ) (n : ℝ) := P.1 ^ 2 / n - P.2 ^ 2 = 1

def same_foci (F1 F2 : ℝ × ℝ)(P E : ℝ × ℝ) := 
  sqrt((P.1 - F1.1)^2 + (P.2 - F1.2)^2) + sqrt((P.1 - F2.1)^2 + (P.2 - F2.2)^2) = 2 * sqrt(m) ∧
  sqrt((E.1 - F1.1)^2 + (E.2 - F1.2)^2) + sqrt((E.1 - F2.1)^2 + (E.2 - F2.2)^2) = 2 * sqrt(n+2)

theorem triangle_is_right:
  ellipse P m →
  hyperbola P n →
  same_foci F1 F2 P E →
  |F1.1 - F2.1| ^ 2 + |F1.2 - F2.2| ^ 2 = |P.1 - F1.1| ^ 2 + |P.2 - F1.2| ^ 2 + |P.1 - F2.1| ^ 2 + |P.2 - F2.2| ^ 2 →
  ∃ (C : Type) [triangle C], is_right_angle_triangle F1 P F2 :=
sorry

end triangle_is_right_l281_281030


namespace number_of_distinct_products_of_two_elements_of_T_l281_281539

def prime_factors_72000 : list (nat × nat) := [(2, 6), (3, 2), (5, 3)]

def T := {n : ℕ | ∃ a b c, 0 ≤ a ∧ a ≤ 6 ∧ 0 ≤ b ∧ b ≤ 2 ∧ 0 ≤ c ∧ c ≤ 3 ∧ n = 2^a * 3^b * 5^c}

def product_of_two_distinct_elements_of_T (n : ℕ) : Prop :=
  ∃ x y, x ∈ T ∧ y ∈ T ∧ x ≠ y ∧ n = x * y

theorem number_of_distinct_products_of_two_elements_of_T :
  {n : ℕ | product_of_two_distinct_elements_of_T n}.card = 451 :=
sorry

end number_of_distinct_products_of_two_elements_of_T_l281_281539


namespace inequality_problem_l281_281486

theorem inequality_problem (a : ℝ) (h : a = Real.cos (2 * Real.pi / 7)) : 
  2^(a - 1/2) < 2 * a :=
by
  sorry

end inequality_problem_l281_281486


namespace acute_triangle_angle_l281_281506

theorem acute_triangle_angle (a b c : ℝ) (A B C : ℝ) 
  (h₁ : a = 2 * b) 
  (h₂ : sin B = sqrt 3 / 4) 
  (h₃ : A + B + C = π) 
  (h₄ : 0 < A) (h₅ : A < π / 2) 
  (h₆ : 0 < B) (h₇ : B < π / 2) 
  (h₈ : 0 < C) (h₉ : C < π / 2) : 
  A = π / 3 :=
by
  sorry

end acute_triangle_angle_l281_281506


namespace worker_A_time_to_complete_job_l281_281664

theorem worker_A_time_to_complete_job (time_B : ℝ) (time_together : ℝ) (h1 : time_B = 10) (h2 : time_together = 4.444444444444445) :
  ∃ A : ℝ, A = 8 :=
by
  use 8
  sorry

end worker_A_time_to_complete_job_l281_281664


namespace custom_op_example_l281_281387

def custom_op (a b : ℝ) : ℝ := (a + 1)^2 - b^2

theorem custom_op_example : custom_op (sqrt 3 - 1) (-sqrt 7) = -4 := by
  sorry

end custom_op_example_l281_281387


namespace domain_of_function_l281_281219

theorem domain_of_function :
  (∀ x : ℝ, (x - 2 > 0) → (5 - x > 0) → (x > 2 ∧ x < 5)) :=
by
  -- Assume x is a real number and the conditions x - 2 > 0 and 5 - x > 0 hold
  intro x
  intro h1
  intro h2
  -- Simplify the conditions
  split
  -- Deduce x > 2 from x - 2 > 0
  exact h1
  -- Deduce x < 5 from 5 - x > 0
  exact h2

end domain_of_function_l281_281219


namespace lowest_degree_is_4_l281_281277

noncomputable def lowest_degree_polynomial (P : Polynomial ℤ) (b : ℤ) : Prop :=
  ∃ (b : ℤ), 
    let A_P := P.support in
    (∀ (a ∈ A_P), a < b ∨ a > b) ∧ 
    (¬(b ∈ A_P)) ∧
    (∃ (a1 a2 : ℤ), a1 ∈ A_P ∧ a2 ∈ A_P ∧ a1 < b ∧ a2 > b)

theorem lowest_degree_is_4 :
  ∀ P : Polynomial ℤ, 
    let b := lowest_degree_polynomial P in
    b P 4 :=
sorry

end lowest_degree_is_4_l281_281277


namespace range_of_a_l281_281461

def f (x : ℝ) : ℝ :=
if x ≥ 0 then x * Real.log (1 + x) + x^2
else -x * Real.log (1 - x) + x^2

theorem range_of_a (a : ℝ) : f (-a) + f (a) ≤ 2 * f (1) ↔ -1 ≤ a ∧ a ≤ 1 := by
  sorry

end range_of_a_l281_281461


namespace selection_with_at_least_one_female_l281_281504

theorem selection_with_at_least_one_female :
  ∃ (total_students : ℕ) (males females : ℕ) (choose : ℕ → ℕ → ℕ),
    total_students = 8 ∧ males = 4 ∧ females = 4 ∧ choose 8 3 - choose 4 3 = 52 :=
by
  let total_students := 8
  let males := 4
  let females := 4
  let choose := λ n k : ℕ, Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))
  use total_students, males, females, choose
  split
  repeat { sorry }

end selection_with_at_least_one_female_l281_281504


namespace cos_alpha_neg_3_5_l281_281511

open Real

variables {α : ℝ} (h_alpha : sin α = 4 / 5) (h_quadrant : π / 2 < α ∧ α < π)

theorem cos_alpha_neg_3_5 : cos α = -3 / 5 :=
by
  -- Proof omitted
  sorry

end cos_alpha_neg_3_5_l281_281511


namespace relationship_among_abc_l281_281816

noncomputable def a : ℝ := log 5 (4 : ℝ) - log 5 (2 : ℝ)
noncomputable def b : ℝ := log (exp 1) (2 / 3 : ℝ) + log (exp 1) (3 : ℝ)
noncomputable def c : ℝ := (10 : ℝ) ^ ((1 / 2 : ℝ) * log 10 (5 : ℝ))

theorem relationship_among_abc : a < b ∧ b < c := by
  sorry

end relationship_among_abc_l281_281816


namespace ratio_of_probabilities_l281_281177

-- Define the total number of balls and bins
def balls : ℕ := 20
def bins : ℕ := 6

-- Define the sets A and B based on the given conditions
def A : ℕ := Nat.choose bins 1 * Nat.choose (bins - 1) 1 * (Nat.factorial balls / (Nat.factorial 2 * Nat.factorial 5 * Nat.factorial 3 * Nat.factorial 3 * Nat.factorial 3 * Nat.factorial 3))
def B : ℕ := Nat.choose bins 2 * (Nat.factorial balls / (Nat.factorial 5 * Nat.factorial 5 * Nat.factorial 2 * Nat.factorial 2 * Nat.factorial 2 * Nat.factorial 2))

-- Define the probabilities p and q
def p : ℚ := A / (Nat.factorial balls * Nat.factorial bins)
def q : ℚ := B / (Nat.factorial balls * Nat.factorial bins)

-- Prove the ratio of probabilities p and q equals 2
theorem ratio_of_probabilities : p / q = 2 := by sorry

end ratio_of_probabilities_l281_281177


namespace Joan_balloons_l281_281888

variable (J : ℕ) -- Joan's blue balloons

theorem Joan_balloons (h : J + 41 = 81) : J = 40 :=
by
  sorry

end Joan_balloons_l281_281888


namespace right_triangle_acute_angles_complementary_l281_281966

theorem right_triangle_acute_angles_complementary 
  (T : Type) [triangle T] (right : is_right_triangle T) :
  are_complementary (acute_angles T) :=
sorry

end right_triangle_acute_angles_complementary_l281_281966


namespace number_of_possible_ticket_prices_l281_281233

/-- The number of possible ticket prices (whole number values of y) that are common divisors of 42 and 70 is 4. -/
theorem number_of_possible_ticket_prices :
  (finset.filter (λ d, 42 % d = 0 ∧ 70 % d = 0) (finset.range 71)).card = 4 :=
sorry

end number_of_possible_ticket_prices_l281_281233


namespace stratified_sampling_number_l281_281721

noncomputable def students_in_grade_10 : ℕ := 150
noncomputable def students_in_grade_11 : ℕ := 180
noncomputable def students_in_grade_12 : ℕ := 210
noncomputable def total_students : ℕ := students_in_grade_10 + students_in_grade_11 + students_in_grade_12
noncomputable def sample_size : ℕ := 72
noncomputable def selection_probability : ℚ := sample_size / total_students
noncomputable def combined_students_grade_10_11 : ℕ := students_in_grade_10 + students_in_grade_11

theorem stratified_sampling_number :
  combined_students_grade_10_11 * selection_probability = 44 := 
by
  sorry

end stratified_sampling_number_l281_281721


namespace max_reflections_l281_281332

theorem max_reflections (P Q R M : Type) (angle : ℝ) :
  0 < angle ∧ angle ≤ 30 ∧ (∃ n : ℕ, 10 * n = angle) →
  ∃ n : ℕ, n ≤ 3 :=
by
  sorry

end max_reflections_l281_281332


namespace point_P_is_orthocenter_l281_281736

open_locale big_operators
open real

variables {A B C P : Type} [real.ordered_ring A] [real.ordered_ring B] [real.ordered_ring C]

def is_orthocenter (A B C P : Type) := 
  ∃ (P : B), orthocenter A B C P

theorem point_P_is_orthocenter
  (A B C P : Type)
  [real.ordered_ring A]
  [real.ordered_ring B]
  [real.ordered_ring C]
  (h : ∀ (AB BC CA AP BP CP : ℝ), 
    AB^2 + CP^2 = BC^2 + AP^2 ∧ 
    BC^2 + AP^2 = CA^2 + BP^2 ∧ 
    CA^2 + BP^2 = AB^2 + CP^2) 
  : is_orthocenter A B C P :=
sorry

end point_P_is_orthocenter_l281_281736


namespace exponent_log_identity_l281_281846

theorem exponent_log_identity (x : ℝ) (h : x * Real.log 2 / Real.log 3 = 1) : 2^x = 3 :=
sorry

end exponent_log_identity_l281_281846


namespace infinite_sum_equals_two_l281_281380

theorem infinite_sum_equals_two :
  ∑' k : ℕ, (8^k / ((4^k - 3^k) * (4^(k + 1) - 3^(k + 1)))) = 2 :=
sorry

end infinite_sum_equals_two_l281_281380


namespace is_inside_circle_diameter_not_2_is_on_circle_line_is_disjoint_l281_281803

open Real

-- Define the circle C as { (x, y) | x^2 + (y + 3)^2 = 4 }
def circle (x y : ℝ) : Prop := x^2 + (y + 3)^2 = 4

-- Define point (1, -2) and (2, -3)
def pt1 : ℝ × ℝ := (1, -2)
def pt2 : ℝ × ℝ := (2, -3)

-- Define the line y = x
def line (x y : ℝ) : Prop := y = x

-- The radius of the circle is 2
def radius : ℝ := 2

-- The distance between two points (x1, y1) and (x2, y2)
def distance (x1 y1 x2 y2 : ℝ) : ℝ := sqrt ((x1 - x2)^2 + (y1 - y2)^2)

theorem is_inside_circle :
  distance 1 (-2) 0 (-3) < radius :=
  sorry

theorem diameter_not_2 :
  2 * radius ≠ 2 :=
  sorry

theorem is_on_circle :
  circle 2 (-3) :=
  sorry

theorem line_is_disjoint :
  sqrt2 * radius < abs 3 / sqrt2 :=
  sorry

end is_inside_circle_diameter_not_2_is_on_circle_line_is_disjoint_l281_281803


namespace car_return_speed_l281_281323

theorem car_return_speed (d : ℝ) (t1 : ℝ) (t2 : ℝ) (r : ℝ) (avg_speed : ℝ) :
  d = 180 ∧ t1 = d / 50 ∧ t2 = 1 ∧ avg_speed = 40 ∧
  (2 * d) / (t1 + d / r + t2) = avg_speed →
  r ≈ 40.91 :=
begin
  sorry
end

end car_return_speed_l281_281323


namespace fanfan_home_distance_l281_281941

theorem fanfan_home_distance (x y z : ℝ) 
  (h1 : x / 3 = 10) 
  (h2 : x / 3 + y / 2 = 25) 
  (h3 : x / 3 + y / 2 + z = 85) :
  x + y + z = 120 :=
sorry

end fanfan_home_distance_l281_281941


namespace possibles_divisors_of_fourth_power_l281_281849

theorem possibles_divisors_of_fourth_power (x d : ℕ) (h1 : ∃ n : ℕ, x = n ^ 4) (h2 : d = Nat.divisors x) :
  d = 213 :=
sorry

end possibles_divisors_of_fourth_power_l281_281849


namespace arc_length_of_sector_l281_281819

theorem arc_length_of_sector (r A l : ℝ) (h_r : r = 2) (h_A : A = π / 3) (h_area : A = 1 / 2 * r * l) : l = π / 3 :=
by
  rw [h_r, h_A] at h_area
  sorry

end arc_length_of_sector_l281_281819


namespace acute_angle_RIS_l281_281901

theorem acute_angle_RIS 
  (A B C I K L M R S : Point)
  (hI : is_incenter I A B C)
  (hK : incircle_touches_side K (BC A B C) (incircle A B C))
  (hL : incircle_touches_side L (CA A B C) (incircle A B C))
  (hM : incircle_touches_side M (AB A B C) (incircle A B C))
  (hR : intersects_line_through R B (parallel_to MK (line LM)) (line A R B))
  (hS : intersects_line_through S B (parallel_to MK (line LK)) (line S B C)) :
  acute_angle (∠ R I S) :=
sorry

end acute_angle_RIS_l281_281901


namespace sum_first_five_terms_l281_281230

noncomputable def a : ℕ → ℕ
| 1     := 1
| 2     := 2
| (n+2) := (a n + a (n + 1)) / 2

theorem sum_first_five_terms :
  a 1 + a 2 + a 3 + a 4 + a 5 = 15 :=
by
  -- This is where the proof would go, which is skipped as per the instructions
  sorry

end sum_first_five_terms_l281_281230


namespace coefficient_of_x5_l281_281390

noncomputable def binomial_expansion (a b : ℕ) (n : ℕ) : ℕ → ℚ :=
  λ r, (nat.choose n r) * ((1 / 2^r) * a * b) 
  sorry -- There's more to it, this is just a conceptual placeholder.

theorem coefficient_of_x5 :
  let x := 5, 
      a := (x^3 : ℚ), 
      b := (1 / (2 * sqrt x) : ℚ), 
      n := 5 in
  binomial_expansion a b n r = 5 / 2 := 
  sorry

end coefficient_of_x5_l281_281390


namespace find_larger_number_l281_281988

-- Define the conditions
variables (L S : ℕ)

theorem find_larger_number (h1 : L - S = 1365) (h2 : L = 6 * S + 15) : L = 1635 :=
sorry

end find_larger_number_l281_281988


namespace x_y_differ_by_one_l281_281920

theorem x_y_differ_by_one (x y : ℚ) (h : (1 + y) / (x - y) = x) : y = x - 1 :=
by
sorry

end x_y_differ_by_one_l281_281920


namespace range_m_l281_281796

variables (x m : ℝ)

def p : Prop := x^2 - 8 * x - 20 ≤ 0
def q : Prop := x^2 - 2 * x + 1 - m^2 ≤ 0 ∧ m > 0
def neg_p : Prop := x > 10 ∨ x < -2
def neg_q : Prop := x < 1 - m ∨ x > 1 + m ∧ m > 0

theorem range_m (h : ¬p → ¬q) : 0 < m ∧ m ≤ 3 := sorry

end range_m_l281_281796


namespace unitary_iff_decomposition_l281_281533

variable {n : ℕ} (A : Matrix (Fin n) (Fin n) ℂ) (S : Matrix (Fin n) (Fin n) ℂ)

-- Defining the conjugate transpose A^H
def conj_transpose (A : Matrix (Fin n) (Fin n) ℂ) : Matrix (Fin n) (Fin n) ℂ := A.transpose.conj

-- Given conditions
axiom h1 : n > 1
axiom h2 : conj_transpose A = A.transpose.conj -- Conjugate Transpose property

-- Proof problem statement
theorem unitary_iff_decomposition :
  (A ⬝ (conj_transpose A) = 1) ↔ ∃ (S : Matrix (Fin n) (Fin n) ℂ), A = S ⬝ (conj_transpose S)⁻¹ :=
sorry

end unitary_iff_decomposition_l281_281533


namespace base_8_to_base_10_and_calculate_cd_div_12_l281_281585

theorem base_8_to_base_10_and_calculate_cd_div_12 :
  \(\forall (c d : ℕ), 563_8 = 3 * c * 10 + d \rightarrow \frac{c \cdot d}{12} = \frac{7}{4} \) := by
  sorry

end base_8_to_base_10_and_calculate_cd_div_12_l281_281585


namespace sum_of_triangle_areas_in_cube_l281_281614

theorem sum_of_triangle_areas_in_cube (m n p : ℤ) (h : ∑ (areas : set ℝ) of triangles with vertices of 2 × 2 × 2 cube, areas = m + Real.sqrt n + Real.sqrt p) : m + n + p = 972 :=
sorry

end sum_of_triangle_areas_in_cube_l281_281614


namespace area_PQRS_l281_281507

-- Definition of the convex quadrilateral
variables (P Q R S : ℝ^2)

-- Given conditions
def is_convex_quad := convex_hull (set.insert P (set.insert Q (set.insert R {S})))

def PQ := dist P Q = 10
def QR := dist Q R = 5
def RS := dist R S = 12
def SP := dist S P = 12
def angle_RSP := ∠ R S P = real.pi / 3 -- 60 degrees in radians

-- Target to prove
theorem area_PQRS (h_convex : is_convex_quad) (h_PQ : PQ) (h_QR : QR) (h_RS : RS) (h_SP : SP) 
(h_angle_RSP : angle_RSP) : 
  ∃ a b c : ℕ, (a, c).coprime ∧ sqrt (a:ℝ) + b * sqrt (c:ℝ)  = 36 * sqrt 3 + 3 * sqrt 3601 / 4 ∧ a + b + c = 3607 := 
sorry

end area_PQRS_l281_281507


namespace cross_section_area_eq_40_3_l281_281589

variables (A B C A1 B1 C1 D E : Type)
variables [HasDistance A] [HasDistance B] [HasDistance C]
variables [HasDistance A1] [HasDistance C1] [HasDistance D] [HasDistance E]

-- Condition 1: ABC is an isosceles triangle with AB = BC = 5.
axiom AB_eq_5 : distance A B = 5
axiom BC_eq_5 : distance B C = 5

-- Condition 2: ∠ABC = 2 arcsin(3/5)
axiom angle_ABC : ∀ (A B C : Type), angle A B C = 2 * arcsin (3 / 5)

-- Condition 3: AD = 1/3 AC
axiom AD_eq_1_3_AC : distance A D = (1 / 3) * distance A C

-- Condition 4: EC1 = 1/3 A1C1
axiom EC1_eq_1_3_A1C1 : distance E C1 = (1 / 3) * distance A1 C1

-- Statement to prove: The area of the cross-section of the prism by this plane is 40/3
theorem cross_section_area_eq_40_3 : 
  cross_section_area_of_prism A B C A1 B1 C1 D E = 40 / 3 := sorry

end cross_section_area_eq_40_3_l281_281589


namespace problem_l281_281092

variable (x y : ℝ)

theorem problem (h1 : (x + y)^2 = 81) (h2 : x * y = 18) : (x - y)^2 = 9 :=
by
  sorry

end problem_l281_281092


namespace arithmetic_sequence_sum_l281_281447

noncomputable def a (n : ℕ) : ℤ := sorry
noncomputable def S (n : ℕ) : ℤ := sorry

theorem arithmetic_sequence_sum :
  (∀ n : ℕ, ∃ d : ℤ, a (n + 1) = a n + d) ∧
  d = -2 ∧
  a 7 = Int.sqrt (a 3 * a 9) ∧
  ∀ {m : ℕ}, S m = ∑ i in Finset.range m, a (i + 1) ∧
  S 10 = 110 :=
sorry

end arithmetic_sequence_sum_l281_281447


namespace measure_angle_ACB_l281_281630

theorem measure_angle_ACB 
  (A B C D E F : Type) 
  [AB AC AE CD D F E : Segment]      -- Defining segments
  (h1 : Segment_len AB = 3 * Segment_len AC)   -- Condition 1: AB = 3 * AC
  (h2 : Angle_at_point BAE = Angle_at_point ACD)   -- Condition 2: angles are equal
  (h3 : Is_equilateral_triangle CFE)   -- Condition 3: CFE is equilateral
  (h4 : Intersection_point AE CD F):   -- Condition 4: F is intersection of AE and CD
  Angle_at_point ACB = 60 :=
sorry

end measure_angle_ACB_l281_281630


namespace trig_expression_value_l281_281811

theorem trig_expression_value (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : cos α = 5 / 13) :
  (sin (α + π / 4) / cos (2 * α + 4 * π)) = - (13 * real.sqrt 2 / 14) :=
sorry

end trig_expression_value_l281_281811


namespace order_of_choosing_l281_281347

structure Heir :=
  (name : String)
  (saw_green_cloak : Bool)
  (gave_snuffbox : Bool)
  (started_carrying_sword : Bool)

def Alvaro : Heir := { name := "Alvaro", saw_green_cloak := false, gave_snuffbox := false, started_carrying_sword := true }
def Benito : Heir := { name := "Benito", saw_green_cloak := false, gave_snuffbox := true, started_carrying_sword := false }
def Vicente : Heir := { name := "Vicente", saw_green_cloak := true, gave_snuffbox := false, started_carrying_sword := false }

def saw_green_cloak (h : Heir) : Prop := h.saw_green_cloak
def gave_snuffbox (h : Heir) : Prop := h.gave_snuffbox
def started_carrying_sword (h : Heir) : Prop := h.started_carrying_sword

axiom condition1 (h : Heir) : saw_green_cloak h → h ≠ Alvaro
axiom condition2 (h : Heir) : gave_snuffbox h → h ≠ Alvaro
axiom condition3 : Vicente ≠ second → (Benito = first ∨ Benito = second)

theorem order_of_choosing :
  (Alvaro = first ∧ Vicente = second ∧ Benito = third) :=
by
  sorry

end order_of_choosing_l281_281347


namespace lowest_degree_polynomial_is_4_l281_281261

noncomputable def lowest_degree_polynomial : ℕ :=
  let exists_polynomial_of_degree_four_with_conditions := ∃ (P : Polynomial ℤ), P.degree = 4 ∧
    ∃ b : ℤ, ∀ coeff ∈ P.coeffs, (coeff < b ∨ coeff > b) ∧ coeff ≠ b
  let no_polynomial_of_degree_less_than_four_with_conditions := ∀ (d < 4), ¬∃ (P : Polynomial ℤ), P.degree = d ∧
    ∃ b : ℤ, ∀ coeff ∈ P.coeffs, (coeff < b ∨ coeff > b) ∧ coeff ≠ b
  if h₁ : exists_polynomial_of_degree_four_with_conditions ∧ no_polynomial_of_degree_less_than_four_with_conditions then 4 else 0

theorem lowest_degree_polynomial_is_4 :
  lowest_degree_polynomial = 4 :=
by
  sorry

end lowest_degree_polynomial_is_4_l281_281261


namespace number_of_zeros_l281_281813

-- Let's define the odd function with period and the given interval conditions.
noncomputable def f : ℝ → ℝ :=
  λ x, if x ∈ Ioo 0 (3 / 2) then log (x^2 - x + 1) else 0

-- Define properties of the function.
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

-- Proving the number of zeros in the interval [0, 6]
theorem number_of_zeros (f : ℝ → ℝ) 
  (h_odd : is_odd f) 
  (h_periodic : is_periodic f 3) 
  (h_def : ∀ x ∈ Ioo 0 (3/2), f x = log (x^2 - x + 1)) : 
  ∃ n : ℕ, n = 9 ∧ (∀ c ∈ Icc 0 6, f c = 0) := 
sorry

end number_of_zeros_l281_281813


namespace reflection_of_orthocenter_lies_on_circumcircle_l281_281573

noncomputable def reflection (A B : Point) : Point := sorry

structure Triangle (α : Type) :=
(A B C : α)

structure Orthocenter (T : Triangle Point) :=
(H : Point)

structure Midpoint (A B : Point) :=
(L : Point)

structure Circumcenter (T : Triangle Point) :=
(O : Point)

def lies_on_circumcircle (P : Point) (T : Triangle Point) : Prop := sorry

theorem reflection_of_orthocenter_lies_on_circumcircle 
    (T : Triangle Point)
    (H : Orthocenter T)
    (L : Midpoint T.A T.C)
    (O : Circumcenter T) :
    lies_on_circumcircle (reflection H.H L.L) T :=
sorry

end reflection_of_orthocenter_lies_on_circumcircle_l281_281573


namespace percentage_temporary_workers_l281_281113

-- Definitions based on the given conditions
def total_workers : ℕ := 100
def percentage_technicians : ℝ := 0.9
def percentage_non_technicians : ℝ := 0.1
def percentage_permanent_technicians : ℝ := 0.9
def percentage_permanent_non_technicians : ℝ := 0.1

-- Statement to prove that the percentage of temporary workers is 18%
theorem percentage_temporary_workers :
  100 * (1 - (percentage_permanent_technicians * percentage_technicians +
              percentage_permanent_non_technicians * percentage_non_technicians)) = 18 :=
by sorry

end percentage_temporary_workers_l281_281113


namespace find_difference_of_squares_l281_281084

variable (x y : ℝ)
variable (h1 : (x + y) ^ 2 = 81)
variable (h2 : x * y = 18)

theorem find_difference_of_squares : (x - y) ^ 2 = 9 := by
  sorry

end find_difference_of_squares_l281_281084


namespace sequence_general_term_l281_281517

theorem sequence_general_term (a : ℕ → ℝ) (h1 : a 1 = 1) (h2 : ∀ n, a (n + 1) = (1 / 2) * a n + 1) :
  ∀ n, a n = 2 - (1 / 2) ^ (n - 1) :=
by
  sorry

end sequence_general_term_l281_281517


namespace mario_paid_on_sunday_l281_281741

-- Noncomputable because we're using real number operations and division
noncomputable def condition_1 (original_price : ℝ) : ℝ := original_price * 1.50
def condition_2 (price : ℝ) : ℝ := price * 0.90
def condition_3 (price : ℝ) : ℝ := 0.85 * price
def shave_cost : ℝ := 10
def styling_cost : ℝ := 15
def paid_monday : ℝ := 18

theorem mario_paid_on_sunday (original_price haircut_price : ℝ)
  (h1 : condition_3 original_price = haircut_price)
  (h2 : paid_monday - shave_cost = haircut_price) :
  let sunday_price := condition_2 (condition_1 original_price + styling_cost) in
  sunday_price = 26.2035 := by
  sorry

end mario_paid_on_sunday_l281_281741


namespace ratio_of_area_of_geometric_figure_to_circle_l281_281326

-- Define the radius of the circle
def radius : ℝ := 3

-- Define the area of the original circle
def area_circle : ℝ := π * radius^2

-- Define the side length of the hexagon formed by the six arcs
def hexagon_side_length : ℝ := radius

-- Define the area of the hexagon
def area_hexagon : ℝ := (3 * real.sqrt 3 / 2) * hexagon_side_length^2

-- Define the ratio of the area of the hexagon to the area of the circle
def ratio : ℝ := area_hexagon / area_circle

-- Theorem stating the ratio equals the expected value
theorem ratio_of_area_of_geometric_figure_to_circle :
  ratio = (3 * real.sqrt 3) / (2 * π) := 
by 
  -- This proof is omitted and replaced with sorry
  sorry 

end ratio_of_area_of_geometric_figure_to_circle_l281_281326


namespace sum_a1_to_a5_eq_neg57_l281_281034

theorem sum_a1_to_a5_eq_neg57 (a : ℕ → ℕ) (x : ℕ → ℕ) :
  (1 + x) + (1 + x)^2 + (1 + x)^3 + (1 + x)^4 + (1 + x)^5 =
  a 0 + a 1 * (1 - x) + a 2 * (1 - x)^2 + a 3 * (1 - x)^3 + a 4 * (1 - x)^4 + a 5 * (1 - x)^5 →
  a 1 + a 2 + a 3 + a 4 + a 5 = -57 :=
by { sorry }

end sum_a1_to_a5_eq_neg57_l281_281034


namespace greatest_product_tens_units_l281_281129

theorem greatest_product_tens_units 
  (a b : ℕ)
  (h₁ : b = 5)
  (h₂ : 11_000 + 100 * a + b * 1 = x)
  (h₃ : x % 55 = 0) :
  a * b = 25 :=
by sorry

end greatest_product_tens_units_l281_281129


namespace height_on_hypotenuse_l281_281117

theorem height_on_hypotenuse (a b : ℝ) (h_right_triangle : a = 6 ∧ b = 8) : 
  let c := Real.sqrt (a * a + b * b) in
  let area := (1 / 2) * a * b in
  let h := (2 * area) / c in
  h = 4.8 :=
by
  sorry

end height_on_hypotenuse_l281_281117


namespace percentage_books_not_sold_is_60_percent_l281_281143

def initial_stock : ℕ := 700
def sold_monday : ℕ := 50
def sold_tuesday : ℕ := 82
def sold_wednesday : ℕ := 60
def sold_thursday : ℕ := 48
def sold_friday : ℕ := 40

def total_sold : ℕ := sold_monday + sold_tuesday + sold_wednesday + sold_thursday + sold_friday
def books_not_sold : ℕ := initial_stock - total_sold
def percentage_not_sold : ℚ := (books_not_sold * 100) / initial_stock

theorem percentage_books_not_sold_is_60_percent : percentage_not_sold = 60 := by
  sorry

end percentage_books_not_sold_is_60_percent_l281_281143


namespace inequality_solution_l281_281404

theorem inequality_solution (x : ℝ) :
  (2 / (x + 2) + 5 / (x + 4) ≥ 1) ↔ (x ∈ set.Icc (-4) (-2) ∪ set.Ioc (-2) 5) := 
by
  sorry

end inequality_solution_l281_281404


namespace ben_paints_area_l281_281350

theorem ben_paints_area (total_area : ℝ) (ratio_1 : ℝ) (ratio_2 : ℝ) (ben_parts : ℝ) :
  total_area = 270 →
  ratio_1 = 2 →
  ratio_2 = 7 →
  ben_parts = 7 →
  ben_parts / (ratio_1 + ratio_2) * total_area = 210 :=
by {
  intros h1 h2 h3 h4,
  rw [h1, h2, h3, h4],
  sorry,
}

end ben_paints_area_l281_281350


namespace equality_of_areas_l281_281987

variables {A B C D M P : Type*} 
variables [linear_ordered_field A] [linear_ordered_field B] 
variables (area : A → B) -- function associating an area measure

-- Conditions
variable h1 : AM = MC
variable h2 : intersects P AC BD -- intersect function assuming two lines intersect at a point P

-- Statements within the theorem
theorem equality_of_areas
  (h1 : AM = MC)
  (h2 : intersects P AC BD) :
  area (triangle A B M) + area (triangle A M D) = area (triangle B M C) + area (triangle C M D) :=
by
  sorry

end equality_of_areas_l281_281987


namespace smallest_value_exclusion_l281_281437

theorem smallest_value_exclusion (a b c : ℝ) (h_a : a ≠ 0) (h_symm : ∀ t : ℝ, (a * ((2 + t)^2) + b * (2 + t) + c) = (a * ((2 - t)^2) + b * (2 - t) + c)) :
  ¬ (∀ x ∈ {-1, 1, 2, 5}, (λ y, a * y^2 + b * y + c) x ≥ (λ y, a * y^2 + b * y + c) 1) :=
sorry

end smallest_value_exclusion_l281_281437


namespace min_value_of_expression_l281_281915

theorem min_value_of_expression (x y : ℤ) (h : 4 * x + 5 * y = 7) : ∃ k : ℤ, 
  5 * Int.natAbs (3 + 5 * k) - 3 * Int.natAbs (-1 - 4 * k) = 1 :=
sorry

end min_value_of_expression_l281_281915


namespace problem_statement_l281_281434

variable {α : Type*} [LinearOrderedField α]

-- Define what it means to form a geometric sequence
def is_geometric_sequence (a b c d : α) : Prop :=
  (b ≠ 0 ∧ c ≠ 0 ∧ (a / b) = (c / d))

-- Conditions in the problem
variable (a b c d : α)

-- Statement to prove that p => q and not (q => p)
theorem problem_statement (p : is_geometric_sequence a b c d) (q : a * d = b * c) :
  (∀ a b c d, is_geometric_sequence a b c d → a * d = b * c) ∧ ¬ (∀ a b c d, a * d = b * c → is_geometric_sequence a b c d) :=
by sorry

end problem_statement_l281_281434


namespace total_seeds_correct_l281_281185

def seeds_per_bed : ℕ := 6
def flower_beds : ℕ := 9
def total_seeds : ℕ := seeds_per_bed * flower_beds

theorem total_seeds_correct : total_seeds = 54 := by
  sorry

end total_seeds_correct_l281_281185


namespace volume_inequality_l281_281548

variable (A B C D P Q R E F G H : Point)
variable (V V1 : ℝ)

-- Required conditions
variable (h1 : P ∈ line AC) (h2 : Q ∈ line AD) (h3 : R ∈ line AB)
variable (h4 : E ∈ line BC) (h5 : F ∈ line BC)
variable (h6 : BE = EF) (h7 : EF = FC)
variable (h8 : G ∈ line AE) (h9 : H ∈ line AF)
variable (h10 : G ∈ line RP) (h11 : H ∈ line RP)
variable (V_eq : volume (tetrahedron A P R Q) = V)
variable (V1_eq : volume (tetrahedron A G H Q) = V1)

theorem volume_inequality (h_parallel_or_coincident : parallel RP BC ∨ RP = BC) :
  V ≥ 3 * V1 ∧ (V = 3 * V1 ↔ parallel RP BC ∨ RP = BC) := 
sorry

end volume_inequality_l281_281548


namespace average_cost_is_correct_l281_281734

variable (marker_cost shipping_cost : ℝ) (num_markers : ℕ)

-- Definitions based on conditions
def total_cost : ℝ := marker_cost + shipping_cost
def total_cost_in_cents : ℝ := total_cost * 100
def average_cost_per_marker : ℝ := total_cost_in_cents / num_markers

-- Statement of the proof problem
theorem average_cost_is_correct (h_marker_cost : marker_cost = 29.85) 
                                (h_shipping_cost : shipping_cost = 8.10) 
                                (h_num_markers : num_markers = 300) : 
  round average_cost_per_marker = 13 :=
by
  sorry

end average_cost_is_correct_l281_281734


namespace sufficient_but_not_necessary_condition_l281_281431

theorem sufficient_but_not_necessary_condition (a : ℝ) : (a > 1 → (1 / a < 1)) ∧ ¬((1 / a < 1) → a > 1) :=
by
  sorry

end sufficient_but_not_necessary_condition_l281_281431


namespace repeating_decimal_to_fraction_l281_281402

theorem repeating_decimal_to_fraction : (1.464646... = 145 / 99) :=
by
  sorry

end repeating_decimal_to_fraction_l281_281402


namespace ralph_fewer_pictures_l281_281198

-- Define the number of wild animal pictures Ralph and Derrick have.
def ralph_pictures : ℕ := 26
def derrick_pictures : ℕ := 34

-- The main theorem stating that Ralph has 8 fewer pictures than Derrick.
theorem ralph_fewer_pictures : derrick_pictures - ralph_pictures = 8 := by
  -- The proof is omitted, denoted by 'sorry'.
  sorry

end ralph_fewer_pictures_l281_281198


namespace same_solutions_a_value_l281_281108

theorem same_solutions_a_value (a x : ℝ) (h1 : 2 * x + 1 = 3) (h2 : 3 - (a - x) / 3 = 1) : a = 7 := by
  sorry

end same_solutions_a_value_l281_281108


namespace polynomial_remainder_695_l281_281644

theorem polynomial_remainder_695 (x : ℝ) :
  let p := 8*x^4 + 4*x^3 - 9*x^2 + 16*x - 28
  let d := 4*x - 12
  (d = 4 * (x - 3)) → p.eval 3 = 695 :=
by
  sorry

end polynomial_remainder_695_l281_281644


namespace problem_l281_281089

variable (x y : ℝ)

theorem problem (h1 : (x + y)^2 = 81) (h2 : x * y = 18) : (x - y)^2 = 9 :=
by
  sorry

end problem_l281_281089


namespace reflection_point_Q_l281_281570

theorem reflection_point_Q' :
  ∃ (Q' : ℝ × ℝ),
    (∃ (x y : ℝ) (a b : ℝ),
      3 * x + 7 = 32 - 2 * x ∧
      2 * a - b = 4 ∧ -a + 2 * b = -8 ∧
      y = a + b ∧
      Q = (x, y) ∧
      y = -4 ∧ x = 5) ∧
    Q' = (-5, -4) :=
begin
  sorry
end

end reflection_point_Q_l281_281570


namespace tangent_line_eq_at_1_exist_perpendicular_tangents_l281_281463

noncomputable def f (x : ℝ) : ℝ := x^2 - Real.log x

theorem tangent_line_eq_at_1 :
  let f1 := f 1 in
  let df := deriv f 1 in
  df = 1 ∧ f1 = 1 ∧ (∀ x, y - 1 = 1 * (x - 1) → y = x) :=
by
  -- Definitions and proofs will go here
  sorry

theorem exist_perpendicular_tangents :
  ∃ (x1 x2 : ℝ), x1 ∈ Icc (1 / 2) 1 ∧ x2 ∈ Icc (1 / 2) 1 ∧
      x1 < x2 ∧
      (2 * x1 - 1 / x1) = -1 ∧ (2 * x2 - 1 / x2) = 1 ∧
      let p1 := (x1, f x1) in
      let p2 := (x2, f x2) in
      p1 = (1/2, Real.log 2 + 1/4) ∧ p2 = (1, 1) :=
by
  -- Definitions and proofs will go here
  sorry

end tangent_line_eq_at_1_exist_perpendicular_tangents_l281_281463


namespace matrices_inverse_sum_l281_281994

open Matrix

noncomputable def M1 : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![x, 2, y], ![3, 3, 4], ![z, 6, w]]

noncomputable def M2 : Matrix (Fin 3) (Fin 3) ℤ :=
  ![[-6, m, -12], ![n, -14, p], ![3, q, 5]]

theorem matrices_inverse_sum 
    (h_inv : M1 * M2 = (1 : Matrix (Fin 3) (Fin 3) ℤ)) : x + y + z + w + m + n + p + q = 49 := by
  sorry

end matrices_inverse_sum_l281_281994


namespace part1_part2_l281_281464

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x + 1| + |x - 2|

-- Problem (1): Prove that m ≤ 3 given the condition f(x) - m ≥ 0
theorem part1 (m : ℝ) (h : ∀ x : ℝ, f x - m ≥ 0) : m ≤ 3 :=
sorry

-- Problem (2): Solve the inequality |x - 3| - 2x ≤ n + 1 and prove the solution set
theorem part2 (n : ℝ) (h : n = 3) : {x : ℝ | |x - 3| - 2x ≤ n + 1} = {x | x ≥ -1/3} :=
sorry

end part1_part2_l281_281464


namespace lock_and_key_requirements_l281_281684

/-- There are 7 scientists each with a key to an electronic lock which requires at least 4 scientists to open.
    - Prove that the minimum number of unique features (locks) the electronic lock must have is 35.
    - Prove that each scientist's key should have at least 20 features.
--/
theorem lock_and_key_requirements :
  ∃ (locks : ℕ) (features_per_key : ℕ), 
    locks = 35 ∧ features_per_key = 20 ∧
    (∀ (n_present : ℕ), n_present ≥ 4 → 7 - n_present ≤ 3) ∧
    (∀ (n_absent : ℕ), n_absent ≤ 3 → 7 - n_absent ≥ 4)
:= sorry

end lock_and_key_requirements_l281_281684


namespace g_zero_not_in_range_l281_281153

def g (x : ℝ) : ℤ :=
  if x > -3 then ⌈2 / (x + 3)⌉
  else ⌊2 / (x + 3)⌋

theorem g_zero_not_in_range :
  ¬ ∃ x : ℝ, x ≠ -3 ∧ g x = 0 := 
sorry

end g_zero_not_in_range_l281_281153


namespace value_of_y_l281_281305

theorem value_of_y (y : ℝ) (h : (sqrt y)^4 = 256) : y = 16 :=
by
  sorry

end value_of_y_l281_281305


namespace infinite_series_sum_l281_281381

theorem infinite_series_sum : 
  ∑' k : ℕ, (8 ^ k) / ((4 ^ k - 3 ^ k) * (4 ^ (k + 1) - 3 ^ (k + 1))) = 2 :=
by 
  sorry

end infinite_series_sum_l281_281381


namespace max_books_borrowed_l281_281669

variable (Students : Type) [Finite Students]
variable [h_class_size : Fintype.card Students = 100]

variable (Books : Students → ℕ)
variable (h_students : Finset Students)
variable (hst1 : (∑ s in h_students.filter (λ s, Books s = 0), 1) = 5)
variable (hst2 : (∑ s in h_students.filter (λ s, Books s = 1), 1) = 25)
variable (hst3 : (∑ s in h_students.filter (λ s, Books s = 2), 1) = 30)
variable (hst4 : (∑ s in h_students.filter (λ s, Books s = 3), 1) = 20)
variable (hst_rest : (∑ s in h_students.filter (λ s, Books s ≥ 4), 1) = 20)
variable (h_avg_books : (∑ s in h_students, Books s) = 3 * 100)

theorem max_books_borrowed : ∃ (student : Students), Books student = 79 :=
by {
  sorry
}

end max_books_borrowed_l281_281669


namespace find_xyz_l281_281757

noncomputable def problem_solution : set ℝ := {-1, 1, 2}

theorem find_xyz (x y z : ℝ) 
  (h1 : x + y + z = 2)
  (h2 : x^2 + y^2 + z^2 = 6)
  (h3 : x^3 + y^3 + z^3 = 8)
  : {x, y, z} = problem_solution :=
sorry

end find_xyz_l281_281757


namespace part1_part2_part3_l281_281466

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (1 / x + a) * Real.log (1 + x)

-- Part (1) proof problem statement
theorem part1 (a : ℝ) (h : a = -1) : 
  tangent_eq :=
let f_x := f x a in
let df_x := diff f_x in
tangent_eq (df_x) = -Real.log(2) * (x - 1) in
sorry

-- Part (2) proof problem statement
theorem part2 : ∃ (a b : ℝ), a = 1/2 ∧ b = -1/2 ∧ symmetric_about (λ x, f (1 / x) a) b :=
sorry

-- Part (3) proof problem statement
theorem part3 (a : ℝ) : 
  extreme_val_range (exists_has_extreme_values f (0, ∞)) = Ioo 0 (1/2) :=
sorry

end part1_part2_part3_l281_281466


namespace hyperbola_eccentricity_condition_l281_281995

theorem hyperbola_eccentricity_condition (m : ℝ) (hm : m > 0) :
  (sqrt (1 + m) > sqrt 2) ↔ (m > 1) :=
by
  sorry

end hyperbola_eccentricity_condition_l281_281995


namespace fraction_of_girls_in_primary_grades_l281_281118

theorem fraction_of_girls_in_primary_grades 
  (total_students : ℕ) (girls_fraction : ℚ) (boys_fraction_in_primary : ℚ)
  (middle_school_students : ℕ)
  (h_students : total_students = 800)
  (h_girls_fraction : girls_fraction = 5 / 8)
  (h_boys_fraction_in_primary : boys_fraction_in_primary = 2 / 5)
  (h_middle_school_students : middle_school_students = 330) :
  let number_of_girls := girls_fraction * total_students in
  let number_of_boys := total_students - number_of_girls in
  let number_of_boys_in_primary := boys_fraction_in_primary * number_of_boys in
  let primary_students := total_students - middle_school_students in
  let x := (primary_students - number_of_boys_in_primary) / number_of_girls in
  x = 7 / 10 :=
by
  sorry

end fraction_of_girls_in_primary_grades_l281_281118


namespace order_of_f_vales_l281_281432

def f (x : ℝ) : ℝ := Real.log x - Real.exp (-x)

def a : ℝ := 2 ^ Real.exp 1
def b : ℝ := Real.log 2
def c : ℝ := Real.log (Real.exp 1) / Real.log 2

theorem order_of_f_vales : f b < f c ∧ f c < f a := by
  sorry

end order_of_f_vales_l281_281432


namespace no_intersection_implies_parallel_or_skew_l281_281509

-- Definitions based on the conditions:
def three_dimensional_space := ℝ^3
def lines_positional_relationship (l1 l2: three_dimensional_space) : Prop :=
  (∃ p : three_dimensional_space, p ∈ l1 ∧ p ∈ l2) ∨ parallel l1 l2 ∨ skew l1 l2

-- The theorem to prove:
theorem no_intersection_implies_parallel_or_skew 
  (l1 l2 : three_dimensional_space) 
  (h : ¬ (∃ p : three_dimensional_space, p ∈ l1 ∧ p ∈ l2)) :
  parallel l1 l2 ∨ skew l1 l2 :=
sorry

end no_intersection_implies_parallel_or_skew_l281_281509


namespace zero_not_in_range_of_g_l281_281159

def g (x : ℝ) : ℤ :=
  if x > -3 then (Real.ceil (2 / (x + 3)))
  else if x < -3 then (Real.floor (2 / (x + 3)))
  else 0 -- g(x) is not defined at x = -3, hence this is a placeholder

noncomputable def range_g : Set ℤ := {n | ∃ x : ℝ, g x = n}

theorem zero_not_in_range_of_g : 0 ∉ range_g :=
by
  intros h,
  exact sorry

end zero_not_in_range_of_g_l281_281159


namespace petrol_expense_l281_281346

variable (Rent Milk Groceries Education Misc Petrol Savings Salary : ℕ)

-- Define the constants given in the problem
def Rent : ℕ := 5000
def Milk : ℕ := 1500
def Groceries : ℕ := 4500
def Education : ℕ := 2500
def Misc : ℕ := 5650
def Savings : ℕ := 2350
def Salary : ℕ := 23500

-- Define the savings as 10% of Salary
def Savings_eq : Savings = Salary / 10 := by
  unfold Savings Salary
  exact rfl

-- Define the total expenses plus the unknown petrol
def TotalExpenses : ℕ := Rent + Milk + Groceries + Education + Misc + Petrol - Savings

-- Define the total expenses to be equal to Salary
theorem petrol_expense : TotalExpenses = Salary → Petrol = 4350 := by
  unfold TotalExpenses
  assume h,
  calc
    Rent + Milk + Groceries + Education + Misc - Savings + Petrol
      = 5000 + 1500 + 4500 + 2500 + 5650 - 2350 + Petrol : by simp [Rent, Milk, Groceries, Education, Misc, Savings]
    ... = 19150 + Petrol : by ring
    ... = Salary : by rw [h, Salary]
    ... = 23500 : rfl
  show Petrol = 4350, by norm_num

#print petrol_expense

end petrol_expense_l281_281346


namespace if_and_only_if_condition_l281_281904

variable {V : Type*} [inner_product_space ℝ V]

noncomputable def vectors (a b : V) :=
   a ≠ 0 ∧ b ≠ 0 ∧ ¬ (∃ k : ℝ, a = k • b)

theorem if_and_only_if_condition (a b : V) (h : vectors a b) :
  (∥a + 2 • b∥ = ∥2 • a + b∥) ↔ (∥a∥ = ∥b∥) :=
  sorry

end if_and_only_if_condition_l281_281904


namespace foci_distance_l281_281776

variable (x y : ℝ)

def ellipse_eq : Prop := (x^2 / 45) + (y^2 / 5) = 9

theorem foci_distance : ellipse_eq x y → (distance_between_foci : ℝ) = 12 * Real.sqrt 10 :=
by
  sorry

end foci_distance_l281_281776


namespace lg_sum_geometric_seq_l281_281127

noncomputable def geometric_sequence (a : ℕ → ℝ) := ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem lg_sum_geometric_seq (a : ℕ → ℝ) (h1 : geometric_sequence a) (h2 : a 2 * a 5 * a 8 = 1) :
  Real.log (a 4) + Real.log (a 6) = 0 := 
sorry

end lg_sum_geometric_seq_l281_281127


namespace line_through_point_with_equal_intercepts_l281_281594

theorem line_through_point_with_equal_intercepts 
  (x y k : ℝ) 
  (h1 : (3 : ℝ) + (-6 : ℝ) + k = 0 ∨ 2 * (3 : ℝ) + (-6 : ℝ) = 0) 
  (h2 : k = 0 ∨ x + y + k = 0) : 
  (x = 1 ∨ x = 2) ∧ (k = -3 ∨ k = 0) :=
sorry

end line_through_point_with_equal_intercepts_l281_281594


namespace sin_plus_5cos_l281_281037

theorem sin_plus_5cos (x : Real) (h : cos x - 5 * sin x = 2) :
  sin x + 5 * cos x = -676 / 211 :=
sorry

end sin_plus_5cos_l281_281037


namespace sum_of_odd_coefficients_l281_281763

theorem sum_of_odd_coefficients (n : ℕ) (h : n = 11) : 
  let f := (1 - X)^n in 
  ∑ i in finset.filter (λ i, odd i) (finset.range (n + 1)), f.coeff i = -2^10 :=
by
  sorry

end sum_of_odd_coefficients_l281_281763


namespace find_y_l281_281422

variables (y : ℝ)
def v := ![2, y]
def w := ![12, 4]
def proj_w_v := (24 + 4 * y) / 160 * w

theorem find_y :
  proj_w_v = ![-9, -3] → y = -36 :=
by
  sorry

end find_y_l281_281422


namespace system_has_solution_l281_281070

theorem system_has_solution (a b : ℝ) :
  (∃ x y : ℝ, x^2 + y^2 = a^2 ∧ x + y = b) ↔ |b| ≤ sqrt 2 * |a| :=
by
  sorry

end system_has_solution_l281_281070


namespace lateral_surface_area_of_cylinder_l281_281720

-- Define the conditions
def height : ℝ := 8
def radius : ℝ := 3

-- Define the formula for the lateral surface area
def lateralSurfaceArea (r h : ℝ) : ℝ := 2 * Real.pi * r * h

-- Define the proof statement
theorem lateral_surface_area_of_cylinder :
  lateralSurfaceArea radius height = 48 * Real.pi := 
by
  -- sorry is used to skip the proof
  sorry

end lateral_surface_area_of_cylinder_l281_281720


namespace number_of_proper_subsets_l281_281227

theorem number_of_proper_subsets (M : Set ℕ) (hM : M = {0, 1, 2}) : M.toFinset.powerset.card - 1 = 7 := by
  -- empty "by" block as placeholder for the proof
  sorry

end number_of_proper_subsets_l281_281227


namespace problem_1_solution_problem_2_solution_l281_281111

variables (total_balls : ℕ) (red_balls : ℕ) (black_balls : ℕ) (white_balls : ℕ) (green_ball : ℕ)

def probability_of_red_or_black_ball (total_balls red_balls black_balls white_balls green_ball : ℕ) : ℚ :=
  (red_balls + black_balls : ℚ) / total_balls

def probability_of_at_least_one_red_ball (total_balls red_balls black_balls white_balls green_ball : ℕ) : ℚ :=
  (((red_balls * (total_balls - red_balls)) + ((red_balls * (red_balls - 1)) / 2)) : ℚ)
  / ((total_balls * (total_balls - 1) / 2) : ℚ)

theorem problem_1_solution :
  probability_of_red_or_black_ball 12 5 4 2 1 = 3 / 4 :=
by
  sorry

theorem problem_2_solution :
  probability_of_at_least_one_red_ball 12 5 4 2 1 = 15 / 22 :=
by
  sorry

end problem_1_solution_problem_2_solution_l281_281111


namespace division_remainder_l281_281643

noncomputable def P : Polynomial ℤ := 3 * X^5 - 8 * X^4 + 2 * X^3 + 17 * X^2 - 23 * X + 14
noncomputable def D : Polynomial ℤ := X^3 + 6 * X^2 - 4 * X + 7
noncomputable def R : Polynomial ℤ := -1080 * X^2 + 807 * X - 1120

theorem division_remainder :
  (P % D) = R :=
sorry

end division_remainder_l281_281643


namespace school_supplies_ratio_l281_281212

theorem school_supplies_ratio (S X : ℝ) : 
    let saved := 400
    let remaining := saved - S
    let spent_on_food := 0.5 * remaining
    let final_left := remaining - spent_on_food
    (final_left = 150) → (S / saved = 1 / 4) :=
by
  -- The conditions
  assume h : final_left = 150
  -- The proof would follow, but for now we put "sorry"
  sorry

end school_supplies_ratio_l281_281212


namespace sum_of_naturals_between_28_and_31_l281_281647

theorem sum_of_naturals_between_28_and_31 : ∑ n in finset.filter (λ x, 28 < x ∧ x ≤ 31) (finset.range 32) = 90 :=
by {
  sorry
}

end sum_of_naturals_between_28_and_31_l281_281647


namespace vasya_problem_l281_281183

theorem vasya_problem (n : ℕ) (a : Fin n → ℝ) (hpos : ∀ i, 0 < a i)
  (hb : ∃ b : Fin n → ℝ, (∀ i, b i ≥ a i) ∧ (∀ i j, ∃ k : ℤ, b i = b j * 2 ^ k) ∧ π i < π j)) 
  : ∃ (b : Fin n → ℝ), (∀ i, b i ≥ a i) ∧ (∀ i j, ∃ k : ℤ, b i = b j * 2^k) ∧ (∏ i, (b i) ≤ 2^((n-1)/2) * ∏ i, a i) :=
by
  sorry

end vasya_problem_l281_281183


namespace spadesuit_sum_l281_281009

def spadesuit (x : ℝ) : ℝ := x + (x^2 + x^3) / 2

theorem spadesuit_sum : spadesuit 1 + spadesuit 2 + spadesuit 3 + spadesuit 4 = 75 := 
by
  sorry

end spadesuit_sum_l281_281009


namespace problem_solution_l281_281045

noncomputable def coordinates_of_vertex_C (A : ℝ × ℝ) (k : ℝ) : ℝ × ℝ :=
  let t := -4 in
  (t, -t)

noncomputable def area_of_triangle (A : ℝ × ℝ) (B : ℝ × ℝ) (C : ℝ × ℝ) : ℝ :=
  let d := (|4 + 1 + 4|) / (Real.sqrt (4 + 1)) in
  let bc := (Real.sqrt ((-2 + 4)^2 + (0 - 4)^2)) in
  1 / 2 * d * 2 * bc

theorem problem_solution (A B C : ℝ × ℝ) :
  A = (2, 1) ∧ C = coordinates_of_vertex_C A (-1) ∧ area_of_triangle A B C = 9 := by
  sorry

end problem_solution_l281_281045


namespace solve_system_eq_l281_281584

theorem solve_system_eq (x y : ℝ) (h1 : 2 * x - y = 3) (h2 : 3 * x + 2 * y = 8) :
  x = 2 ∧ y = 1 :=
by
  sorry

end solve_system_eq_l281_281584


namespace Jungkook_fewest_erasers_l281_281530

-- Define the number of erasers each person has.
def Jungkook_erasers : ℕ := 6
def Jimin_erasers : ℕ := Jungkook_erasers + 4
def Seokjin_erasers : ℕ := Jimin_erasers - 3

-- Prove that Jungkook has the fewest erasers.
theorem Jungkook_fewest_erasers : Jungkook_erasers < Jimin_erasers ∧ Jungkook_erasers < Seokjin_erasers :=
by
  -- Proof goes here
  sorry

end Jungkook_fewest_erasers_l281_281530


namespace count_irrationals_l281_281356

def is_irrational (x : ℝ) : Prop := ¬ ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

theorem count_irrationals : 
  let S := {- (sqrt 2), 0.5, 1 / 2, sqrt 4} in
  S.count (λ x => is_irrational x) = 1 := 
by 
  sorry

end count_irrationals_l281_281356


namespace even_odd_sequences_equal_l281_281722

theorem even_odd_sequences_equal (a : ℕ → ℕ) (n : ℕ) (h : ∑ i in Finset.range (n + 1), (i + 1) * a (i + 1) = 1979) :
  ∃ (evens odds : ℕ), evens = odds :=
sorry

end even_odd_sequences_equal_l281_281722


namespace arithmetic_sequence_properties_l281_281442

/--
Given an arithmetic sequence \(\{a_n\}\) such that 
* \(a_3 = 7\) 
* \(a_5 + a_7 = 26\)
and the sum of the first \(n\) terms of \(\{a_n\}\) is \(S_n\), and the sequence \( \{b_n\} \) where \(b_n = \frac {1}{a_n^{2}-1}\) (where \(n ∈ \mathbb{N^*}\))

Prove that:
1. The general term of the sequence \(a_n\) is \( a_n = 2n + 1 \)
2. The sum of the first \(n\) terms, \(S_n\), is \( S_n = n^2 + 2n \)
3. The sum of the first \(n\) terms of the sequence \(\{b_n\}\), \(T_n\), is \( T_n = \frac{n}{4(n+1)} \)
-/
theorem arithmetic_sequence_properties (n : ℕ) (h₁ : a 3 = 7) (h₂ : a 5 + a 7 = 26)
  (a : ℕ → ℕ) (S : ℕ → ℕ) (b : ℕ → ℕ) (T : ℕ → ℕ) :
  (∀ n, a n = 2 * n + 1) ∧ (∀ n, S n = n ^ 2 + 2 * n) ∧ (∀ n, T n = n / (4 * (n + 1))) :=
by
  sorry

end arithmetic_sequence_properties_l281_281442


namespace lowest_degree_polynomial_l281_281289

-- Define the conditions
def polynomial_conditions (P : ℕ → ℤ) (b : ℤ): Prop :=
  (∃ c, c > b ∧ c ∈ set.range P) ∧ (∃ d, d < b ∧ d ∈ set.range P) ∧ (b ∉ set.range P)

-- The main statement
theorem lowest_degree_polynomial : ∃ P : ℕ → ℤ, polynomial_conditions P 4 ∧ (∀ Q : ℕ → ℤ, polynomial_conditions Q 4 → degree Q >= 4) :=
sorry

end lowest_degree_polynomial_l281_281289


namespace total_time_spent_l281_281528

def one_round_time : ℕ := 30
def saturday_initial_rounds : ℕ := 1
def saturday_additional_rounds : ℕ := 10
def sunday_rounds : ℕ := 15

theorem total_time_spent :
  one_round_time * (saturday_initial_rounds + saturday_additional_rounds + sunday_rounds) = 780 := by
  sorry

end total_time_spent_l281_281528


namespace price_per_glass_on_second_day_correct_l281_281178

noncomputable def price_per_glass_on_second_day (O : ℝ) (volume_per_glass : ℝ) : ℝ :=
  let P := (0.60 * 2) / 3 in
  P

theorem price_per_glass_on_second_day_correct 
  (O : ℝ) (volume_per_glass : ℝ)
  (h1 : 0 < O)
  (h2 : 0 < volume_per_glass)
  (h3 : (0.60 * (2 * O / volume_per_glass)) = ((price_per_glass_on_second_day O volume_per_glass) * (3 * O / volume_per_glass))) :
  price_per_glass_on_second_day O volume_per_glass = 0.40 := by
  sorry

end price_per_glass_on_second_day_correct_l281_281178


namespace parallel_perpendicular_l281_281043

theorem parallel_perpendicular (m n : Line) (α β : Plane)
  (h1 : m ≠ n) (h2 : α ≠ β)
  (h3 : m ∥ n) (h4 : m ⊥ α) :
  n ⊥ α := 
sorry

end parallel_perpendicular_l281_281043


namespace tens_of_meters_in_verst_l281_281388

theorem tens_of_meters_in_verst :
  let verst_in_arshins := 500 * 3 in
  let arshins_in_cm := verst_in_arshins * 71 in
  let cm_in_m := arshins_in_cm / 100 in
  let tens_of_m := cm_in_m / 10 in
  tens_of_m = 106 := by
  let verst_in_arshins := 500 * 3
  let arshins_in_cm := verst_in_arshins * 71
  let cm_in_m := arshins_in_cm / 100
  let tens_of_m := cm_in_m / 10
  sorry

end tens_of_meters_in_verst_l281_281388


namespace sin_plus_5cos_l281_281038

theorem sin_plus_5cos (x : Real) (h : cos x - 5 * sin x = 2) :
  sin x + 5 * cos x = -676 / 211 :=
sorry

end sin_plus_5cos_l281_281038


namespace parabola_focus_coincides_ellipse_focus_l281_281497

theorem parabola_focus_coincides_ellipse_focus (p : ℝ) :
  (∃ F : ℝ × ℝ, F = (2, 0) ∧ ∀ x y : ℝ, y^2 = 2 * p * x <-> x = p / 2)
  → p = 4 := 
by
  sorry 

end parabola_focus_coincides_ellipse_focus_l281_281497


namespace simplify_expression_l281_281577

theorem simplify_expression (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : 
  (a^3 - b^3) / (a * b^2) - (ab^2 - b^3) / (ab^2 - a^3) = (a^3 - ab^2 + b^4) / (a * b^2) :=
sorry

end simplify_expression_l281_281577


namespace fifth_digit_is_one_l281_281320

def self_descriptive_seven_digit_number (A B C D E F G : ℕ) : Prop :=
  A = 3 ∧ B = 2 ∧ C = 2 ∧ D = 1 ∧ E = 1 ∧ [A, B, C, D, E, F, G].count 0 = A ∧
  [A, B, C, D, E, F, G].count 1 = B ∧ [A, B, C, D, E, F, G].count 2 = C ∧
  [A, B, C, D, E, F, G].count 3 = D ∧ [A, B, C, D, E, F, G].count 4 = E

theorem fifth_digit_is_one
  (A B C D E F G : ℕ) (h : self_descriptive_seven_digit_number A B C D E F G) : E = 1 := by
  sorry

end fifth_digit_is_one_l281_281320


namespace chemical_reaction_proof_l281_281479

-- Define the given number of moles for each reactant
def moles_NaOH : ℕ := 4
def moles_NH4Cl : ℕ := 3

-- Define the balanced chemical equation stoichiometry
def stoichiometry_ratio_NaOH_NH4Cl : ℕ := 1

-- Define the product formation based on the limiting reactant
theorem chemical_reaction_proof
  (moles_NaOH : ℕ)
  (moles_NH4Cl : ℕ)
  (stoichiometry_ratio_NaOH_NH4Cl : ℕ)
  (h1 : moles_NaOH = 4)
  (h2 : moles_NH4Cl = 3)
  (h3 : stoichiometry_ratio_NaOH_NH4Cl = 1):
  (3 = 3 * 1) ∧
  (3 = 3 * 1) ∧
  (3 = 3 * 1) ∧
  (3 = moles_NH4Cl) ∧
  (1 = moles_NaOH - moles_NH4Cl) :=
by {
  -- Provide assumptions based on the problem
  sorry
}

end chemical_reaction_proof_l281_281479


namespace partition_triangle_l281_281897

theorem partition_triangle {n : ℕ}
  (A B C : Point)
  (P : Fin n → Point)
  (h_no_collinear : ∀ (i j k : Fin n), i ≠ j → j ≠ k → i ≠ k → ¬Collinear {P i, P j, P k}) :
  ∃ T : Fin (2 * n + 1) → Triangle,
  (∀ i, (T i).vertices ⊆ {A, B, C} ∪ set.range P) ∧
  (∃ m ≥ n + Int.ofNat (Nat.sqrt n) + 1, ∀ i < m, ∃ x ∈ (T i).vertices, x ∈ {A, B, C}) :=
sorry

end partition_triangle_l281_281897


namespace min_multiplications_12_numbers_l281_281257

theorem min_multiplications_12_numbers : 
  (∀ n : ℕ, n > 1 → min_multiplications n = n - 1) → min_multiplications 12 = 11 :=
sorry

def min_multiplications (n : ℕ) : ℕ :=
  if n > 1 then n - 1 else 0

end min_multiplications_12_numbers_l281_281257


namespace max_numbers_chosen_no_gcd_difference_l281_281310

theorem max_numbers_chosen_no_gcd_difference (n : ℕ) (hn : 0 < n) :
  ∃ S : set ℕ, S ⊆ {i | i ≤ 2 * n} ∧ S.card = n ∧
    ∀ x ∈ S, ∀ y ∈ S, x > y → x - y ≠ gcd x y :=
by
  -- placeholder for proof
  sorry

end max_numbers_chosen_no_gcd_difference_l281_281310


namespace find_a7_l281_281515

variable (a : ℕ → ℝ)

def arithmetic_sequence (d : ℝ) (a1 : ℝ) :=
  ∀ n, a n = a1 + (n - 1) * d

theorem find_a7
  (a : ℕ → ℝ)
  (d : ℝ)
  (a1 : ℝ)
  (h_arith : arithmetic_sequence a d a1)
  (h_a3 : a 3 = 7)
  (h_a5 : a 5 = 13):
  a 7 = 19 :=
by
  sorry

end find_a7_l281_281515


namespace integer_not_in_range_l281_281156

theorem integer_not_in_range (g : ℝ → ℤ) :
  (∀ x, x > -3 → g x = Int.ceil (2 / (x + 3))) ∧
  (∀ x, x < -3 → g x = Int.floor (2 / (x + 3))) →
  ∀ z : ℤ, (∃ x, g x = z) ↔ z ≠ 0 :=
by
  intros h z
  sorry

end integer_not_in_range_l281_281156


namespace area_of_triangle_abe_l281_281131

-- Declare the geometry setting and conditions
variables (A B C D E : Type) [plane_geometry A B C D E] -- assume plane_geometry defines the geometric context
variables (AD DC AB BE AE DE : ℝ)

-- Define the given lengths and relationships
axiom h1 : ∠AD DC = 90 -- AD is perpendicular to DC
axiom h2 : AD = 4
axiom h3 : AB = 4
axiom h4 : DC = 8
axiom h5 : BE ∥ AD -- BE is parallel to AD

-- Define the proof goal
theorem area_of_triangle_abe : 
  let a := AB,
      b := BE,
      h := a ∥ BE,
      area := (1/2) * a * b in
  area = 8 :=
begin
  sorry
end

end area_of_triangle_abe_l281_281131


namespace range_of_m_l281_281860

variable {x m : ℝ}

theorem range_of_m (h1 : x + 2 < 2 * m) (h2 : x - m < 0) (h3 : x < 2 * m - 2) : m ≤ 2 :=
sorry

end range_of_m_l281_281860


namespace sculpture_cost_in_inr_l281_281184

def convert_currency (n_cost : ℕ) (n_to_b_rate : ℕ) (b_to_i_rate : ℕ) : ℕ := 
  (n_cost / n_to_b_rate) * b_to_i_rate

theorem sculpture_cost_in_inr (n_cost : ℕ) (n_to_b_rate : ℕ) (b_to_i_rate : ℕ) :
  n_cost = 360 → 
  n_to_b_rate = 18 → 
  b_to_i_rate = 20 →
  convert_currency n_cost n_to_b_rate b_to_i_rate = 400 := by
  intros h1 h2 h3
  rw [h1, h2, h3]
  -- turns 360 / 18 * 20 = 400
  sorry

end sculpture_cost_in_inr_l281_281184


namespace sequence_conjecture_l281_281023

theorem sequence_conjecture (a : ℕ → ℚ) (h₁ : a 1 = 1) (h₂ : ∀ n, a (n + 1) = a n / (a n + 1)) :
  ∀ n : ℕ, 0 < n → a n = 1 / n := by
  sorry

end sequence_conjecture_l281_281023


namespace jane_wins_probability_l281_281340

noncomputable def probability_jane_wins : ℚ :=
  let losing_pairs := setOf (fun (i j : ℕ) => (abs (i - j) >= 4)) ∩ { (i, j) | 1 ≤ i ∧ i ≤ 7 ∧ 1 ≤ j ∧ j ≤ 7 }
  let total_outcomes := 7 * 7
  let losing_combinations := 2 * card losing_pairs
  let winning_combinations := total_outcomes - losing_combinations
  winning_combinations / total_outcomes

theorem jane_wins_probability :
  probability_jane_wins = 37 / 49 :=
by
  sorry

end jane_wins_probability_l281_281340


namespace Karl_selects_five_crayons_l281_281238

theorem Karl_selects_five_crayons : ∃ (k : ℕ), k = 3003 ∧ (finset.card (finset.powerset_len 5 (finset.range 15))).nat_abs = k :=
by
  -- existence proof of k = 3003 and showing that k equals the combination count
  sorry

end Karl_selects_five_crayons_l281_281238


namespace probability_divisible_by_3_l281_281618

theorem probability_divisible_by_3 :
  let M := (x : ℕ) * 100 + (y : ℕ) * 10 + 5 in
  (∀ M, 100 ≤ M ∧ M < 1000 ∧ M % 10 = 5 →
    let x := M / 100 in 
    let y := (M / 10) % 10 in
    (x + y + 5) % 3 = 0 ↔ (x + y) % 3 = 1) →
    1 / 3 := sorry

end probability_divisible_by_3_l281_281618


namespace maxValue_proof_l281_281063

noncomputable def maxValue (a b c : ℝ) : ℝ :=
  if a > 0 ∧ ∀ x : ℝ, a * x^2 + b * x + c ≥ 2 * a * x + b then
    2 * Real.sqrt 2 - 2
  else
    0

theorem maxValue_proof (a b c : ℝ) (h : a > 0) :
  (∀ x : ℝ, a * x^2 + b * x + c ≥ 2 * a * x + b) →
  maxValue a b c = 2 * Real.sqrt 2 - 2 :=
by
  sorry -- Proof is not required.

end maxValue_proof_l281_281063


namespace find_difference_of_squares_l281_281088

variable (x y : ℝ)
variable (h1 : (x + y) ^ 2 = 81)
variable (h2 : x * y = 18)

theorem find_difference_of_squares : (x - y) ^ 2 = 9 := by
  sorry

end find_difference_of_squares_l281_281088


namespace coin_change_count_ways_l281_281162

theorem coin_change_count_ways :
  ∃ n : ℕ, (∀ q h : ℕ, (25 * q + 50 * h = 1500) ∧ q > 0 ∧ h > 0 → (1 ≤ h ∧ h < 30)) ∧ n = 29 :=
  sorry

end coin_change_count_ways_l281_281162


namespace eunji_initial_money_l281_281400

-- Define the conditions
def snack_cost : ℕ := 350
def allowance : ℕ := 800
def money_left_after_pencil : ℕ := 550

-- Define what needs to be proven
theorem eunji_initial_money (initial_money : ℕ) :
  initial_money - snack_cost + allowance = money_left_after_pencil * 2 →
  initial_money = 650 :=
by
  sorry

end eunji_initial_money_l281_281400


namespace minimum_value_l281_281413

def f (x y : ℝ) : ℝ := x * y / (x^2 + y^2)

def x_in_domain (x : ℝ) : Prop := (1/4 : ℝ) ≤ x ∧ x ≤ (2/3 : ℝ)
def y_in_domain (y : ℝ) : Prop := (1/5 : ℝ) ≤ y ∧ y ≤ (1/2 : ℝ)

theorem minimum_value (x y : ℝ) (hx : x_in_domain x) (hy : y_in_domain y) :
  ∃ x y, f x y = (2/5 : ℝ) := 
sorry

end minimum_value_l281_281413


namespace problem_equivalent_l281_281787

-- Define gcd and lcm
def gcd (a b : ℕ) : ℕ := Nat.gcd a b
def lcm (a b : ℕ) : ℕ := Nat.lcm a b

-- Prove that the given equation holds if and only if n is prime
theorem problem_equivalent (n : ℕ) : 
  (∀ a b : ℕ, gcd a b * lcm a b = a * b) →
  (4 * ∑ k in Finset.range (n + 1), lcm n k = 
   1 + ∑ k in Finset.range (n + 1), gcd n k + 
   2 * n^2 * ∑ k in Finset.range (n + 1), 1 / gcd n k) ↔ 
  Nat.Prime n :=
sorry

end problem_equivalent_l281_281787


namespace max_product_of_roots_of_quadratic_l281_281830

theorem max_product_of_roots_of_quadratic :
  ∃ k : ℚ, 6 * k^2 - 8 * k + (4 / 3) = 0 ∧ (64 - 48 * k) ≥ 0 ∧ (∀ k' : ℚ, (64 - 48 * k') ≥ 0 → (k'/3) ≤ (4/9)) :=
by
  sorry

end max_product_of_roots_of_quadratic_l281_281830


namespace arithmetic_seq_proof_l281_281922

variables {a : ℕ → ℤ} {d : ℤ}

/-- The sequence a_n is arithmetic with common difference d, where d > 0 -/
def is_arithmetic_seq (a : ℕ → ℤ) (d : ℤ) : Prop :=
∀ n, a (n + 1) = a n + d

-- Given conditions
def conditions (a : ℕ → ℤ) (d : ℤ) : Prop :=
d > 0 ∧ a 0 + a 1 + a 2 = -9 ∧ a 0 * a 1 * a 2 = -15

-- The general formula for the sequence a_n
def general_formula (a : ℕ → ℤ) : Prop :=
∀ n, a n = 2 * n - 7

-- The sum of the first n terms of the sequence
def sum_first_n_terms (a : ℕ → ℤ) : Prop :=
∀ n, (∑ i in finset.range n, a i) = n^2 - 6 * n

-- Find all positive integers m such that a_m * a_{m+1} / a_{m+2} is a term in the sequence
def valid_m (a : ℕ → ℤ) (m : ℕ) : Prop :=
a (m - 1) * a m / a (m + 1) = ∃ n, a n

theorem arithmetic_seq_proof (a : ℕ → ℤ) (d : ℤ) :
  is_arithmetic_seq a d →
  conditions a d →
  general_formula a →
  sum_first_n_terms a →
  ∀ m : ℕ, valid_m a m → m = 2 :=
sorry

end arithmetic_seq_proof_l281_281922


namespace inequality_proof_l281_281097

theorem inequality_proof (a b : ℝ) (ha : a < 0) (hb : b > 0) : (1 / a < 1 / b) :=
sorry

end inequality_proof_l281_281097


namespace distance_between_A_and_B_l281_281691

def time : ℝ := 4.5
def speed : ℝ := 80

theorem distance_between_A_and_B :
  let distance : ℝ := speed * time in
  distance = 360 :=
by
  sorry

end distance_between_A_and_B_l281_281691


namespace find_f_neg5_l281_281433

-- Define the function f and the constants a, b, and c
def f (x : ℝ) (a b c : ℝ) : ℝ := a * x^5 + b * x^3 + c * x + 5

-- State the main theorem we want to prove
theorem find_f_neg5 (a b c : ℝ) (h : f 5 a b c = 9) : f (-5) a b c = 1 :=
by
  sorry

end find_f_neg5_l281_281433


namespace square_diff_l281_281077

theorem square_diff (x y : ℝ) (h1 : (x + y)^2 = 81) (h2 : x * y = 18) : (x - y)^2 = 9 :=
by 
  sorry

end square_diff_l281_281077


namespace sum_of_roots_l281_281837

open Polynomial

noncomputable def f (a b : ℝ) : Polynomial ℝ := Polynomial.C b + Polynomial.C a * X + X^2
noncomputable def g (c d : ℝ) : Polynomial ℝ := Polynomial.C d + Polynomial.C c * X + X^2

theorem sum_of_roots (a b c d : ℝ)
  (h1 : eval 1 (f a b) = eval 2 (g c d))
  (h2 : eval 1 (g c d) = eval 2 (f a b))
  (hf_roots : ∃ r1 r2 : ℝ, (f a b).roots = {r1, r2})
  (hg_roots : ∃ s1 s2 : ℝ, (g c d).roots = {s1, s2}) :
  (-(a + c) = 6) :=
sorry

end sum_of_roots_l281_281837


namespace remainder_of_x_pow_50_div_x_minus_1_cubed_l281_281414

noncomputable def remainder_when_x_pow_50_divided_by_x_minus_1_cubed : Polynomial ℤ :=
  let P := Polynomial.X ^ 50
  let Q := (Polynomial.X - 1) ^ 3
  let remainder := 1225 * Polynomial.X ^ 2 - 2500 * Polynomial.X + 1276
  sorry

theorem remainder_of_x_pow_50_div_x_minus_1_cubed :
  let P := Polynomial.X ^ 50
  let Q := (Polynomial.X - 1) ^ 3
  let remainder := 1225 * Polynomial.X ^ 2 - 2500 * Polynomial.X + 1276
  (P % Q) = remainder :=
by
  let P := Polynomial.X ^ 50
  let Q := (Polynomial.X - 1) ^ 3
  let remainder := 1225 * Polynomial.X ^ 2 - 2500 * Polynomial.X + 1276
  exact sorry

end remainder_of_x_pow_50_div_x_minus_1_cubed_l281_281414


namespace solve_arctan_equation_l281_281976

theorem solve_arctan_equation (x : ℝ) :
  arctan (1 / x^2) + arctan (1 / x^4) = π / 4 ↔ x = real.sqrt ((1 + real.sqrt 5) / 2) ∨ x = -real.sqrt ((1 + real.sqrt 5) / 2) :=
by
  -- The statement is correct, the proof is left as an exercise.
  sorry

end solve_arctan_equation_l281_281976


namespace g_zero_not_in_range_l281_281152

def g (x : ℝ) : ℤ :=
  if x > -3 then ⌈2 / (x + 3)⌉
  else ⌊2 / (x + 3)⌋

theorem g_zero_not_in_range :
  ¬ ∃ x : ℝ, x ≠ -3 ∧ g x = 0 := 
sorry

end g_zero_not_in_range_l281_281152


namespace height_of_highest_wave_l281_281367

theorem height_of_highest_wave 
  (h_austin : ℝ) -- Austin's height
  (h_high : ℝ) -- Highest wave's height
  (h_short : ℝ) -- Shortest wave's height 
  (height_relation1 : h_high = 4 * h_austin + 2)
  (height_relation2 : h_short = h_austin + 4)
  (surfboard : ℝ) (surfboard_len : surfboard = 7)
  (short_wave_len : h_short = surfboard + 3) :
  h_high = 26 :=
by
  -- Define local variables with the values from given conditions
  let austin_height := 6        -- as per calculation: 10 - 4 = 6
  let highest_wave_height := 26 -- as per calculation: (6 * 4) + 2 = 26
  sorry

end height_of_highest_wave_l281_281367


namespace lowest_degree_polynomial_is_4_l281_281263

noncomputable def lowest_degree_polynomial : ℕ :=
  let exists_polynomial_of_degree_four_with_conditions := ∃ (P : Polynomial ℤ), P.degree = 4 ∧
    ∃ b : ℤ, ∀ coeff ∈ P.coeffs, (coeff < b ∨ coeff > b) ∧ coeff ≠ b
  let no_polynomial_of_degree_less_than_four_with_conditions := ∀ (d < 4), ¬∃ (P : Polynomial ℤ), P.degree = d ∧
    ∃ b : ℤ, ∀ coeff ∈ P.coeffs, (coeff < b ∨ coeff > b) ∧ coeff ≠ b
  if h₁ : exists_polynomial_of_degree_four_with_conditions ∧ no_polynomial_of_degree_less_than_four_with_conditions then 4 else 0

theorem lowest_degree_polynomial_is_4 :
  lowest_degree_polynomial = 4 :=
by
  sorry

end lowest_degree_polynomial_is_4_l281_281263


namespace dice_product_probability_eq_24_l281_281653

theorem dice_product_probability_eq_24 : 
  let range := {n : ℕ | 1 ≤ n ∧ n ≤ 6} in
  let outcomes := (range × range × range × range) in
  let valid_outcomes := {tup : outcomes | tup.fst * tup.snd.fst * tup.snd.snd.fst * tup.snd.snd.snd = 24} in
  (valid_outcomes.card : ℚ) / (outcomes.card : ℚ) = 1 / 36 :=
sorry

end dice_product_probability_eq_24_l281_281653


namespace geometric_sequence_solution_l281_281011

theorem geometric_sequence_solution :
  ∃ (a b : ℤ),
    ((a + 6): ℤ, a, (b + 5): ℤ, b).tuple = (18, 12, 15, 10) ∨
    ((a + 6): ℤ, a, (b + 5): ℤ, b).tuple = (-12, -18, -10, -15) ∧
    (a + 6 - a = 6) ∧
    (b + 5 - b = 5) ∧
    (a^2 + (a + 6)^2 + b^2 + (b + 5)^2 = 793) ∧
    ((a + 6) / a = (b + 5) / b) :=
begin
  sorry
end

end geometric_sequence_solution_l281_281011


namespace y_coordinate_third_vertex_first_quadrant_l281_281359

noncomputable def y_coordinate_of_third_vertex (A B : (ℝ × ℝ)) (hA : A = (2, 3)) (hB : B = (10, 3)) : ℝ :=
  3 + 4 * Real.sqrt 3

theorem y_coordinate_third_vertex_first_quadrant (A B : (ℝ × ℝ)) (hA : A = (2, 3)) (hB : B = (10, 3)) (C : ℝ × ℝ) (hC : C.2 > 0) :
(C = (6, 3 + 4 * Real.sqrt 3)) → C.2 = 3 + 4 * Real.sqrt 3 :=
by
  intros h
  cases h
  refl

end y_coordinate_third_vertex_first_quadrant_l281_281359


namespace regions_in_quadrants_I_II_l281_281389

noncomputable def in_quadrants_I_II (x y: ℝ): Prop :=
  (y > x^2) ∧ (y > 4 - x)

theorem regions_in_quadrants_I_II :
  ∀ (x y: ℝ), in_quadrants_I_II x y → (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y > 0) :=
by
  intro x y
  intro h
  cases h with h1 h2
  sorry

end regions_in_quadrants_I_II_l281_281389


namespace bert_initial_amount_l281_281315

theorem bert_initial_amount (n : ℝ) (h : (1 / 2) * (3 / 4 * n - 9) = 12) : n = 44 :=
sorry

end bert_initial_amount_l281_281315


namespace sum_reciprocals_B_is_35_over_8_p_plus_q_is_43_l281_281538

def B : Set ℕ := {n | ∀ p, Prime p → p ∣ n → p = 2 ∨ p = 3 ∨ p = 5 ∨ p = 7}

noncomputable def sum_reciprocals_B : ℚ :=
  ∑' (n : ℕ) in B, 1 / n

theorem sum_reciprocals_B_is_35_over_8 : sum_reciprocals_B = 35 / 8 := by
  sorry

theorem p_plus_q_is_43 :
  let p := 35
  let q := 8
  p + q = 43 :=
by
  calc 35 + 8 = 43 : by norm_num

end sum_reciprocals_B_is_35_over_8_p_plus_q_is_43_l281_281538


namespace sum_of_odd_coefficients_l281_281616

theorem sum_of_odd_coefficients (a : ℝ) (h : (a + 1) * 16 = 32) : a = 3 :=
by
  sorry

end sum_of_odd_coefficients_l281_281616


namespace geometric_sequence_common_ratio_l281_281797

theorem geometric_sequence_common_ratio (a₁ : ℕ) (S₃ : ℕ) (q : ℤ) 
  (h₁ : a₁ = 2) (h₂ : S₃ = 6) : 
  (q = 1 ∨ q = -2) :=
by
  sorry

end geometric_sequence_common_ratio_l281_281797


namespace hyperbola_eq_l281_281451

/-- Given a hyperbola with center at the origin, 
    one focus at (-√5, 0), and a point P on the hyperbola such that 
    the midpoint of segment PF₁ has coordinates (0, 2), 
    then the equation of the hyperbola is x² - y²/4 = 1. --/
theorem hyperbola_eq (x y : ℝ) (P F1 : ℝ × ℝ) 
  (hF1 : F1 = (-Real.sqrt 5, 0)) 
  (hMidPoint : (P.1 + -Real.sqrt 5) / 2 = 0 ∧ (P.2 + 0) / 2 = 2) 
  : x^2 - y^2 / 4 = 1 := 
sorry

end hyperbola_eq_l281_281451


namespace price_per_glass_on_second_day_correct_l281_281179

noncomputable def price_per_glass_on_second_day (O : ℝ) (volume_per_glass : ℝ) : ℝ :=
  let P := (0.60 * 2) / 3 in
  P

theorem price_per_glass_on_second_day_correct 
  (O : ℝ) (volume_per_glass : ℝ)
  (h1 : 0 < O)
  (h2 : 0 < volume_per_glass)
  (h3 : (0.60 * (2 * O / volume_per_glass)) = ((price_per_glass_on_second_day O volume_per_glass) * (3 * O / volume_per_glass))) :
  price_per_glass_on_second_day O volume_per_glass = 0.40 := by
  sorry

end price_per_glass_on_second_day_correct_l281_281179


namespace prove_condition_BS_eq_CS_l281_281536

open_locale complex_conjugate

noncomputable def given_problem (A B C K S : ℂ) (ω : set ℂ) : Prop :=
  let circumcircle : set ℂ := ω in
  let is_tangent (P : ℂ) : Prop := ∀ Q : ℂ, Q ∈ circumcircle → abs (P - Q) = |Q - K| in
  let is_parallel (L1 L2 : set ℂ) : Prop := ∃ (k : ℂ), ∀ (z : ℂ), z ∈ L1 → z ∈ L2 ∨ z = z + k in
  circumcircle A ∧ circumcircle B ∧ circumcircle C ∧
  is_tangent A ∧ is_tangent B ∧
  is_parallel ({z | ∃ t : ℝ, z = K + t * (B - C)}) ({z | ∃ t : ℝ, z = A + t * (C - A)}) ∧
  S ∈ {z | ∃ t : ℝ, z = A + t * (C - A)} ∧
  B ≠ C ∧
  abs (B - S) = abs (C - S)

theorem prove_condition_BS_eq_CS (A B C K S : ℂ) (ω : set ℂ)
  (h : given_problem A B C K S ω) : abs (B - S) = abs (C - S) :=
sorry

end prove_condition_BS_eq_CS_l281_281536


namespace smallest_g_bottles_l281_281839

theorem smallest_g_bottles (P : ℕ) (hP : P ≥ 1) :
  let D := 2 * P + 3 in
  let G := 3 * D - 5 in
  G = 6 * P + 4 
:=
begin
  sorry
end

#eval smallest_g_bottles 1 (by decide)  -- This should evaluate to 10

end smallest_g_bottles_l281_281839


namespace pure_imaginary_when_sum_l281_281911

theorem pure_imaginary_when_sum (a : ℝ) : 
  (let z1 := a + 2 * Complex.I in let z2 := 3 - 4 * Complex.I in 
  (z1 + z2).re = 0) → a = -3 :=
by
  intro h
  -- We assume (z1 + z2).re = 0
  have h1 : (a + 3) = 0 := h
  -- Solve for a
  exact eq_of_add_eq_zero_right h1

end pure_imaginary_when_sum_l281_281911


namespace squared_difference_l281_281080

variable {x y : ℝ}

theorem squared_difference (h1 : (x + y)^2 = 81) (h2 : x * y = 18) :
  (x - y)^2 = 9 :=
by
  sorry

end squared_difference_l281_281080


namespace smallest_sum_of_digits_l281_281145

noncomputable def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

noncomputable def distinct_digits (n : ℕ) : Prop :=
  let digits := n.digits 10
  (digits.all (λ d, d < 10)) ∧ (digits.nodup)

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem smallest_sum_of_digits (a b S : ℕ) :
  is_three_digit a → is_three_digit b → distinct_digits a → distinct_digits b →
  b > a → (b.digits 10).head ∈ [8, 9] →
  (a.digits 10).head ∉ (b.digits 10) →
  S = a + b → is_three_digit S →
  sum_of_digits S = 1 :=
begin
  sorry
end

end smallest_sum_of_digits_l281_281145


namespace length_AB_is_10_l281_281817

-- Define the points A and B on the given lines such that their midpoint is P(0, 10/a)
def pointA (a : ℝ) : ℝ × ℝ := sorry
def pointB (a : ℝ) : ℝ × ℝ := sorry

-- Define the midpoint condition
def isMidpoint (P A B : ℝ × ℝ) : Prop :=
  P = (0, 10 / a) ∧
  (fst A + fst B) / 2 = 0 ∧
  (snd A + snd B) / 2 = 10 / a

-- Define the length of a segment
def length_segment (A B : ℝ × ℝ) : ℝ :=
  real.sqrt ((fst B - fst A)^2 + (snd B - snd A)^2)

theorem length_AB_is_10 (a : ℝ) :
  isMidpoint (0, 10 / a) (pointA a) (pointB a) → length_segment (pointA a) (pointB a) = 10 := sorry

end length_AB_is_10_l281_281817


namespace sequence_general_term_l281_281608

noncomputable def general_term (n : ℕ) : ℝ :=
  (-1)^(n+1) * (2 * n + 1) / 2^n

theorem sequence_general_term (n : ℕ) : 
  ∃ f : ℕ → ℝ, ∀ n, f n = (-1)^(n+1) * (2 * n + 1) / 2^n :=
begin
  use general_term,
  intros,
  refl,
end

end sequence_general_term_l281_281608


namespace curve_intersection_three_points_l281_281784

theorem curve_intersection_three_points (a : ℝ) :
  (∀ x y : ℝ, ((x^2 - y^2 = a^2) ∧ ((x-1)^2 + y^2 = 1)) → (a = 0)) :=
by
  sorry

end curve_intersection_three_points_l281_281784


namespace range_of_a_l281_281046

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def is_decreasing (f : ℝ → ℝ) (I : set ℝ) : Prop := ∀ x y, x ∈ I → y ∈ I → x < y → f y ≤ f x

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) 
  (h_even : is_even f) 
  (h_decreasing : is_decreasing f {x | x ≤ 0})
  (h_ineq : f a > f 2) : a < -2 ∨ a > 2 :=
sorry

end range_of_a_l281_281046


namespace find_x_values_for_y_eq_5_l281_281055

def piecewise_function (x : ℝ) : ℝ :=
  if x ≤ 0 then x^2 + 1 else -2 * x

theorem find_x_values_for_y_eq_5 : { x : ℝ | piecewise_function x = 5 } = {-2} :=
by
  sorry

end find_x_values_for_y_eq_5_l281_281055


namespace log_xy_eq_seven_fifths_l281_281485

theorem log_xy_eq_seven_fifths (x y : ℝ) (h1 : log (x * y ^ 2) = 2) (h2 : log (x ^ 3 * y) = 3) : log (x * y) = 7 / 5 :=
sorry

end log_xy_eq_seven_fifths_l281_281485


namespace circle_equation_condition_l281_281857

theorem circle_equation_condition (m : ℝ) : 
  (∃ h k r : ℝ, (r > 0) ∧ ∀ x y : ℝ, (x - h)^2 + (y - k)^2 = r^2 → x^2 + y^2 - 2*x - 4*y + m = 0) ↔ m < 5 :=
sorry

end circle_equation_condition_l281_281857


namespace cos_600_eq_neg_half_l281_281395

theorem cos_600_eq_neg_half : Real.cos (600 * Real.pi / 180) = -1 / 2 :=
by
  sorry

end cos_600_eq_neg_half_l281_281395


namespace solve_f_zero_k_eq_2_find_k_range_has_two_zeros_sum_of_reciprocals_l281_281018

-- Define the function f(x) based on the given conditions
def f (x k : ℝ) : ℝ := abs (x ^ 2 - 1) + x ^ 2 + k * x

-- Statement 1
theorem solve_f_zero_k_eq_2 :
  (∀ x : ℝ, f x 2 = 0 ↔ x = - (1 + Real.sqrt 3) / 2 ∨ x = -1 / 2) :=
sorry

-- Statement 2
theorem find_k_range_has_two_zeros (α β : ℝ) (hαβ : 0 < α ∧ α < β ∧ β < 2) :
  (∃ k : ℝ, f α k = 0 ∧ f β k = 0) ↔ - 7 / 2 < k ∧ k < -1 :=
sorry

-- Statement 3
theorem sum_of_reciprocals (α β : ℝ) (hαβ : 0 < α ∧ α < 1 ∧ 1 < β ∧ β < 2)
    (hα : f α (-1/α) = 0) (hβ : ∃ k : ℝ, f β k = 0) :
  (1 / α + 1 / β < 4) :=
sorry

end solve_f_zero_k_eq_2_find_k_range_has_two_zeros_sum_of_reciprocals_l281_281018


namespace find_x_sets_l281_281406

open Real

noncomputable def log : ℝ → ℝ := sorry -- Assuming log base 2 is defined

theorem find_x_sets (n : ℕ)
  (x : Fin (n+2) → ℝ)
  (h1 : x 0 = x (n+1))
  (h2 : ∀ k : Fin n, 2 * log (x k) * log (x (k + 1)) - (log (x k))^2 = 9) :
  (∀ k : Fin (n+2), x k = 8) ∨ (∀ k : Fin (n+2), x k = 1/8) := 
sorry

end find_x_sets_l281_281406


namespace range_r_prime_l281_281421

def is_composite (n : ℕ) : Prop := ∃ p q : ℕ, 1 < p ∧ p < n ∧ 1 < q ∧ q < n ∧ n = p * q
def distinct_prime_factors (n : ℕ) : Finset ℕ := (Nat.factors n).toFinset
def r' (n : ℕ) : ℕ := (distinct_prime_factors n).sum

theorem range_r_prime : 
  ∀ (n : ℕ), is_composite n → (∃ k : ℕ, k > 1 ∧ k ∉ prime_set ∧ ∃ xs : Finset ℕ, 
  (∀ x ∈ xs, Prime x) ∧ xs.sum = k) :=
begin
  intro n,
  assume hn : is_composite n,
  let k := r' n,
  have h2 : k > 1 := sorry,
  have h3 : k ∉ prime_set := sorry,
  have h4 : ∃ xs : Finset ℕ, (∀ x ∈ xs, Prime x) ∧ xs.sum = k := sorry,
  use k,
  exact ⟨h2, h3, h4⟩,
end

end range_r_prime_l281_281421


namespace lowest_degree_polynomial_is_4_l281_281267

noncomputable def lowest_degree_polynomial : ℕ :=
  let exists_polynomial_of_degree_four_with_conditions := ∃ (P : Polynomial ℤ), P.degree = 4 ∧
    ∃ b : ℤ, ∀ coeff ∈ P.coeffs, (coeff < b ∨ coeff > b) ∧ coeff ≠ b
  let no_polynomial_of_degree_less_than_four_with_conditions := ∀ (d < 4), ¬∃ (P : Polynomial ℤ), P.degree = d ∧
    ∃ b : ℤ, ∀ coeff ∈ P.coeffs, (coeff < b ∨ coeff > b) ∧ coeff ≠ b
  if h₁ : exists_polynomial_of_degree_four_with_conditions ∧ no_polynomial_of_degree_less_than_four_with_conditions then 4 else 0

theorem lowest_degree_polynomial_is_4 :
  lowest_degree_polynomial = 4 :=
by
  sorry

end lowest_degree_polynomial_is_4_l281_281267


namespace non_congruent_rectangles_count_l281_281480

theorem non_congruent_rectangles_count :
  let grid_width := 6
  let grid_height := 4
  let axis_aligned_rectangles := (grid_width.choose 2) * (grid_height.choose 2)
  let squares_1x1 := (grid_width - 1) * (grid_height - 1)
  let squares_2x2 := (grid_width - 2) * (grid_height - 2)
  let non_congruent_rectangles := axis_aligned_rectangles - (squares_1x1 + squares_2x2)
  non_congruent_rectangles = 67 := 
by {
  sorry
}

end non_congruent_rectangles_count_l281_281480


namespace integer_not_in_range_l281_281155

theorem integer_not_in_range (g : ℝ → ℤ) :
  (∀ x, x > -3 → g x = Int.ceil (2 / (x + 3))) ∧
  (∀ x, x < -3 → g x = Int.floor (2 / (x + 3))) →
  ∀ z : ℤ, (∃ x, g x = z) ↔ z ≠ 0 :=
by
  intros h z
  sorry

end integer_not_in_range_l281_281155


namespace problem_l281_281090

variable (x y : ℝ)

theorem problem (h1 : (x + y)^2 = 81) (h2 : x * y = 18) : (x - y)^2 = 9 :=
by
  sorry

end problem_l281_281090


namespace bert_money_left_l281_281368

theorem bert_money_left
  (initial_amount : ℚ)
  (hardware_store_fraction : ℚ)
  (dry_cleaners_amount : ℚ)
  (expected_leftover : ℚ) :
  initial_amount = 41.99 →
  hardware_store_fraction = 1/3 →
  dry_cleaners_amount = 7 →
  expected_leftover = 10.49 →
  let after_hardware := initial_amount * hardware_store_fraction in
  let remaining_after_hardware := initial_amount - after_hardware in
  let remaining_after_dry_cleaners := remaining_after_hardware - dry_cleaners_amount in
  let grocery_store_spent := remaining_after_dry_cleaners / 2 in
  let remaining_after_grocery_store := remaining_after_dry_cleaners - grocery_store_spent in
  abs (remaining_after_grocery_store - expected_leftover) < 0.01 :=
by sorry  -- Since rounding is involved, use an approximation within a cent

end bert_money_left_l281_281368


namespace repeating_cycle_is_142857_l281_281345

theorem repeating_cycle_is_142857 
  (h: (∀ n, (n % 6 = 4) → (nth_digit_of_repeating_decimal "142857" n = 8))) : 
  (repeating_sequence_of (0.142857 : ℚ)) = "142857" :=
sorry

end repeating_cycle_is_142857_l281_281345


namespace weight_vest_cost_l281_281885

noncomputable def weight_plate_cost (pound_price : ℝ) (pounds : ℝ) : ℝ :=
  pounds * pound_price

noncomputable def discounted_vest_cost (original_price : ℝ) (discount : ℝ) : ℝ :=
  original_price - discount

noncomputable def total_cost (weight_vest_cost : ℝ) (weight_plate_cost : ℝ) : ℝ :=
  weight_vest_cost + weight_plate_cost

theorem weight_vest_cost {pound_price : ℝ} {pounds : ℝ} {original_vest_price : ℝ} {discount : ℝ} {savings : ℝ} 
  (h1 : pound_price = 1.2) (h2 : pounds = 200) (h3 : original_vest_price = 700) (h4 : discount = 100) (h5 : savings = 110) :
  let weight_plate_cost := weight_plate_cost 1.2 200 in
  let discounted_cost := discounted_vest_cost 700 100 in
  let total_spending := discounted_cost - savings in
    ∃ X, total_cost X weight_plate_cost = total_spending ∧ X = 250 :=
by
  let weight_plate_cost := weight_plate_cost 1.2 200
  let discounted_cost := discounted_vest_cost 700 100
  let total_spending := discounted_cost - 110
  use 250
  sorry

end weight_vest_cost_l281_281885


namespace minimum_varphi_is_2_l281_281808

def varphi (a b : ℝ) := Real.sqrt (a^2 + b^2) + 1

noncomputable def minimum_varphi (a b : ℝ) (h : a^2 + b^2 - 4 * a + 3 = 0) : ℝ :=
  2

theorem minimum_varphi_is_2 (a b : ℝ) (h : a^2 + b^2 - 4 * a + 3 = 0) : 
  varphi a b >= 2 :=
sorry

end minimum_varphi_is_2_l281_281808


namespace list_of_all_possible_outcomes_probability_event_A_probability_event_B_l281_281321

def possible_outcomes := [(1,1), (1,2), (1,3), (1,4), (2,1), (2,2), (2,3), (2,4), (3,1), (3,2), (3,3), (3,4), (4,1), (4,2), (4,3), (4,4)]

def event_A (outcome : ℕ × ℕ) : Prop := outcome.1 + outcome.2 < 4
def event_B (outcome : ℕ × ℕ) : Prop := outcome.1 < outcome.2

theorem list_of_all_possible_outcomes :
  possible_outcomes = [(1,1), (1,2), (1,3), (1,4), (2,1), (2,2), (2,3), (2,4), (3,1), (3,2), (3,3), (3,4), (4,1), (4,2), (4,3), (4,4)] :=
by
  sorry

theorem probability_event_A :
  (List.countp event_A possible_outcomes) / (List.length possible_outcomes) = 3 / 16 :=
by
  sorry

theorem probability_event_B :
  (List.countp event_B possible_outcomes) / (List.length possible_outcomes) = 3 / 8 :=
by
  sorry

end list_of_all_possible_outcomes_probability_event_A_probability_event_B_l281_281321


namespace minimum_shift_value_l281_281821

noncomputable def determinant (x1 y1 x2 y2 : ℝ) : ℝ :=
  x1 * y2 - x2 * y1

def f (x : ℝ) : ℝ :=
  determinant (Real.sqrt 3) (Real.cos x) 1 (Real.sin x)

axiom shift_right_odd (ϕ : ℝ) (hϕ : ϕ > 0) : ∀ x, f (x + ϕ) = -f x

theorem minimum_shift_value : ∃ ϕ, ϕ = (5 * Real.pi) / 6 ∧ ∀ x, f (x + ϕ) = -f x :=
by
  use (5 * Real.pi) / 6
  split
  . rfl
  . exact shift_right_odd _ sorry

end minimum_shift_value_l281_281821


namespace good_games_count_l281_281940

-- Define the conditions
def games_from_friend : Nat := 50
def games_from_garage_sale : Nat := 27
def games_that_didnt_work : Nat := 74

-- Define the total games bought
def total_games_bought : Nat := games_from_friend + games_from_garage_sale

-- State the theorem to prove the number of good games
theorem good_games_count : total_games_bought - games_that_didnt_work = 3 :=
by
  sorry

end good_games_count_l281_281940


namespace product_x_z_l281_281122

variable (EF GH FG HE : ℝ)
variable (x z : ℝ)
variable (h_parallelogram : EF = GH ∧ FG = HE)
variable (h_EF : EF = 50)
variable (h_FG : FG = 4 * z ^ 2 + 1)
variable (h_GH : GH = 3 * x + 6)
variable (h_HE : HE = 81)

theorem product_x_z : x * z = (88 * real.sqrt 5) / 3 :=
by
  sorry

end product_x_z_l281_281122


namespace consecutive_integers_sum_l281_281605

theorem consecutive_integers_sum (x : ℕ) (h : x * (x + 1) = 380) : x + (x + 1) = 39 := by
  sorry

end consecutive_integers_sum_l281_281605


namespace problem_statement_l281_281551

noncomputable def f : ℝ → ℝ := sorry

theorem problem_statement (x : ℝ) :
  f(0) = 2008 ∧
  (∀ x : ℝ, f(x + 2) - f(x) ≤ 3 * 2^x) ∧
  (∀ x : ℝ, f(x + 6) - f(x) ≥ 63 * 2^x) →
  f(2008) = 2^2008 + 2007 :=
by
  sorry

end problem_statement_l281_281551


namespace polynomial_remainder_l281_281549

theorem polynomial_remainder :
  ∀ (Q : ℤ[X]),
  (∀ x, x = 15 → eval x Q = 10) ∧ 
  (∀ x, x = 19 → eval x Q = 8) 
  → 
  ∃ (c d : ℝ), remainder_coeff (Q) = c * (X : ℝ[X]) + d 
  ∧ c = -1 / 2 ∧ d = 17.5 := 
by
  intro Q hQ,
  have h1 : eval 15 Q = 10 := (hQ 15 rfl),
  have h2 : eval 19 Q = 8 := (hQ 19 rfl),
  sorry

end polynomial_remainder_l281_281549


namespace lowest_degree_poly_meets_conditions_l281_281297

-- Define a predicate that checks if a polynomial P meets the conditions
def poly_meets_conditions (P : ℚ[X]) (b : ℚ) : Prop :=
  (∀ x, coeff P x ≠ b) ∧ 
  (∃ x y, coeff P x < b ∧ coeff P y > b)

-- Statement of the theorem we want to prove
theorem lowest_degree_poly_meets_conditions : ∀ (b : ℚ), 
  ∃ (P : ℚ[X]), poly_meets_conditions P b ∧ degree P = 4 :=
begin
  sorry
end

end lowest_degree_poly_meets_conditions_l281_281297


namespace coefficient_of_monomial_degree_of_monomial_l281_281986

-- Definitions for the conditions
def monomial := -3^2 * x * y * z^2
def coefficient (m : ℤ) := -9
def degree (m : ℕ) := 4

-- Proving the statements
theorem coefficient_of_monomial : coefficient (monomial) = -9 := 
by sorry

theorem degree_of_monomial : degree (monomial) = 4 := 
by sorry

end coefficient_of_monomial_degree_of_monomial_l281_281986


namespace sum_of_coordinates_l281_281242

open Real

noncomputable def point_distance (p1 p2 : ℝ × ℝ) : ℝ :=
  sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

theorem sum_of_coordinates :
  let p1 := (10 + 8 * sqrt 3, 28)
  let p2 := (10 - 8 * sqrt 3, 28)
  let p3 := (10 + 8 * sqrt 3, 12)
  let p4 := (10 - 8 * sqrt 3, 12)
  let points := [p1, p2, p3, p4] in
  let sum_coordinates := list.foldr (λ p acc, acc + p.1 + p.2) 0 points in
  (∀ p ∈ points, |p.2 - 20| = 8) →
  (∀ p ∈ points, point_distance p (10, 20) = 16) →
  sum_coordinates = 120 :=
by
let p1 := (10 + 8 * sqrt 3, 28)
let p2 := (10 - 8 * sqrt 3, 28)
let p3 := (10 + 8 * sqrt 3, 12)
let p4 := (10 - 8 * sqrt 3, 12)
let points := [p1, p2, p3, p4]
let sum_coordinates := list.foldr (λ p acc, acc + p.1 + p.2) 0 points
assume h1 : (∀ p ∈ points, |p.2 - 20| = 8)
assume h2 : (∀ p ∈ points, point_distance p (10, 20) = 16)
show sum_coordinates = 120
{
  sorry
}

end sum_of_coordinates_l281_281242


namespace eighteen_to_mn_eq_l281_281586

theorem eighteen_to_mn_eq (m n : ℤ) (P Q : ℤ) (h1 : P = 3^m) (h2 : Q = 2^n) : 
  18^(m * n) = P^(2 * n) * Q^m := 
by 
  sorry

end eighteen_to_mn_eq_l281_281586


namespace andrew_paid_correct_amount_l281_281737

-- Definitions of the conditions
def cost_of_grapes : ℝ := 7 * 68
def cost_of_mangoes : ℝ := 9 * 48
def cost_of_apples : ℝ := 5 * 55
def cost_of_oranges : ℝ := 4 * 38

def total_cost_grapes_and_mangoes_before_discount : ℝ := cost_of_grapes + cost_of_mangoes
def discount_on_grapes_and_mangoes : ℝ := 0.10 * total_cost_grapes_and_mangoes_before_discount
def total_cost_grapes_and_mangoes_after_discount : ℝ := total_cost_grapes_and_mangoes_before_discount - discount_on_grapes_and_mangoes

def total_cost_all_fruits_before_tax : ℝ := total_cost_grapes_and_mangoes_after_discount + cost_of_apples + cost_of_oranges
def sales_tax : ℝ := 0.05 * total_cost_all_fruits_before_tax
def total_amount_to_pay : ℝ := total_cost_all_fruits_before_tax + sales_tax

-- Statement to be proved
theorem andrew_paid_correct_amount :
  total_amount_to_pay = 1306.41 :=
by
  sorry

end andrew_paid_correct_amount_l281_281737


namespace inequality_proof_l281_281164

variable {a b c : ℝ}

def H (a b c : ℝ) := 3 / (1/a + 1/b + 1/c)
def G (a b c : ℝ) := (a * b * c)^(1/3 : ℝ)
def A (a b c : ℝ) := (a + b + c) / 3
def Q (a b c : ℝ) := sqrt((a^2 + b^2 + c^2) / 3)

theorem inequality_proof (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (A a b c * G a b c) / (Q a b c * H a b c) ≥ (27/32)^(1/6 : ℝ) :=
by {
  sorry
}

end inequality_proof_l281_281164


namespace steps_probability_to_point_3_3_l281_281207

theorem steps_probability_to_point_3_3 : 
  let a := 35
  let b := 4096
  a + b = 4131 :=
by {
  sorry
}

end steps_probability_to_point_3_3_l281_281207


namespace lowest_degree_is_4_l281_281281

noncomputable def lowest_degree_polynomial (P : Polynomial ℤ) (b : ℤ) : Prop :=
  ∃ (b : ℤ), 
    let A_P := P.support in
    (∀ (a ∈ A_P), a < b ∨ a > b) ∧ 
    (¬(b ∈ A_P)) ∧
    (∃ (a1 a2 : ℤ), a1 ∈ A_P ∧ a2 ∈ A_P ∧ a1 < b ∧ a2 > b)

theorem lowest_degree_is_4 :
  ∀ P : Polynomial ℤ, 
    let b := lowest_degree_polynomial P in
    b P 4 :=
sorry

end lowest_degree_is_4_l281_281281


namespace lowest_degree_polynomial_l281_281294

-- Define the conditions
def polynomial_conditions (P : ℕ → ℤ) (b : ℤ): Prop :=
  (∃ c, c > b ∧ c ∈ set.range P) ∧ (∃ d, d < b ∧ d ∈ set.range P) ∧ (b ∉ set.range P)

-- The main statement
theorem lowest_degree_polynomial : ∃ P : ℕ → ℤ, polynomial_conditions P 4 ∧ (∀ Q : ℕ → ℤ, polynomial_conditions Q 4 → degree Q >= 4) :=
sorry

end lowest_degree_polynomial_l281_281294


namespace stitches_per_unicorn_l281_281373

theorem stitches_per_unicorn :
  (let number_stitches_per_minute := 4
   let flower_stitches := 60
   let godzilla_stitches := 800
   let flowers := 50
   let unicorns := 3
   let minutes := 1085
   let total_stitches_flower_godzilla := flower_stitches * flowers + godzilla_stitches
   let total_stitches := minutes * number_stitches_per_minute
   let remaining_stitches_unicorns := total_stitches - total_stitches_flower_godzilla
  in remaining_stitches_unicorns / unicorns = 180) :=
by
  sorry

end stitches_per_unicorn_l281_281373


namespace two_liters_to_pints_l281_281815

theorem two_liters_to_pints :
  (∀ (liter_to_pint_conversion_l : ℝ), liter_to_pint_conversion_l = 2.625 / 1.25 →
  (∀ (liters : ℝ), liters = 2 →
  ∃ (pints : ℝ), pints = liters * liter_to_pint_conversion_l ∧ pints = 4.2)) :=
by
  intros h_conversion h_two_liters
  rw h_two_liters
  use 2 * (2.625 / 1.25)
  split
  · -- prove conversion calculation
    sorry
  · -- prove pints value calculation
    have h_pints : 2 * (2.625 / 1.25) = 4.2,
    { exact calc
      2 * (2.625 / 1.25) = 2 * 2.1 : by sorry
                          ... = 4.2 : by sorry },
    exact h_pints

end two_liters_to_pints_l281_281815


namespace min_fraction_in_domain_l281_281410

theorem min_fraction_in_domain :
  ∃ x y : ℝ, (1/4 ≤ x ∧ x ≤ 2/3) ∧ (1/5 ≤ y ∧ y ≤ 1/2) ∧ 
    (∀ x' y' : ℝ, (1/4 ≤ x' ∧ x' ≤ 2/3) ∧ (1/5 ≤ y' ∧ y' ≤ 1/2) → 
      (xy / (x^2 + y^2) ≤ x'y' / (x'^2 + y'^2))) ∧ 
      xy / (x^2 + y^2) = 2/5 :=
sorry

end min_fraction_in_domain_l281_281410


namespace lowest_degree_required_l281_281269

noncomputable def smallest_degree_poly (b : ℤ) : ℕ :=
  if h : ∃ P : Polynomial ℝ, (∀ x, (P.eval x ≠ b)) ∧
    (∃ y, (P.eval y > b)) ∧ (∃ z, (P.eval z < b)) 
  then Nat.find h 
  else 0

theorem lowest_degree_required :
  ∃ b : ℤ, smallest_degree_poly b = 4 :=
by
  -- b is some integer that fits the described conditions
  use 0
  sorry

end lowest_degree_required_l281_281269


namespace p_squared_div_q_eq_zero_l281_281545

noncomputable def complex_roots_conditions (z1 z2 p q : ℂ) : Prop :=
  (z2 = complex.exp (2 * real.pi * complex.I / 3) * z1) ∧
  ((z1 + z2 = -p) ∧ (z1 * z2 = q)) ∧
  (abs (z1 - 0) = abs (z2 - 0))

theorem p_squared_div_q_eq_zero (z1 z2 p q : ℂ)
  (hz : complex_roots_conditions z1 z2 p q) :
  (p^2 / q) = 0 :=
sorry

end p_squared_div_q_eq_zero_l281_281545


namespace greatest_digit_sum_base9_l281_281260

theorem greatest_digit_sum_base9 (n : ℕ) (h1 : 0 < n) (h2 : n < 3000) : 
  ∃ m, (0 < m) ∧ (m < 3000) ∧ 
  let digit_sum := (m / 729) + ((m % 729) / 81) + ((m % 81) / 9) + (m % 9)
  in digit_sum ≤ 24 := 
sorry

end greatest_digit_sum_base9_l281_281260


namespace log_expression_simplify_l281_281460

theorem log_expression_simplify (p q r s x y : ℝ) 
  (h0 : p ≠ 0) 
  (h1 : q ≠ 0) 
  (h2 : r ≠ 0) 
  (h3 : s ≠ 0 )
  (h4 : x ≠ 0) 
  (h5 : y ≠ 0) :
  log ((p^2 / q) * (q^3 / r) * (r^2 / s)) - log (p^2 * y / (s^3 * x)) = log (q^2 * r * x * s^2 / y) :=
by sorry

end log_expression_simplify_l281_281460


namespace complement_event_A_l281_281505

def is_at_least_two_defective (n : ℕ) : Prop :=
  n ≥ 2

def is_at_most_one_defective (n : ℕ) : Prop :=
  n ≤ 1

theorem complement_event_A (n : ℕ) :
  (¬ is_at_least_two_defective n) ↔ is_at_most_one_defective n :=
by
  sorry

end complement_event_A_l281_281505


namespace correct_statement_is_D_l281_281662

-- Conditions
def statement_A := ∀ x ∈ ℝ, y = 1 / x → ¬ (x > 0 ∧ y decreases as x increases)
def statement_B := ∀ x ∈ ℝ, y = -2 * x → ¬ (translated 1 unit down from y = -2 * x - 1)
def statement_C := ∀ triangles, similar triangles → ¬ (similar triangles are congruent triangles)
def statement_D (red_ball: Type) (white_ball: Type) :=
  let bag := {red_ball, white_ball, white_ball} in
  let outcomes := {draw_two_balls_with_replacement} in
  #|outcomes| = 3

-- Theorem statement
theorem correct_statement_is_D : 
  statement_D (Type) (Type) → ¬ statement_A ∧ ¬ statement_B ∧ ¬ statement_C :=
by
  assume hD: statement_D (Type) (Type)
  have hA: ¬ statement_A := sorry
  have hB: ¬ statement_B := sorry
  have hC: ¬ statement_C := sorry
  exact ⟨hA, hB, hC⟩

end correct_statement_is_D_l281_281662


namespace min_fraction_in_domain_l281_281411

theorem min_fraction_in_domain :
  ∃ x y : ℝ, (1/4 ≤ x ∧ x ≤ 2/3) ∧ (1/5 ≤ y ∧ y ≤ 1/2) ∧ 
    (∀ x' y' : ℝ, (1/4 ≤ x' ∧ x' ≤ 2/3) ∧ (1/5 ≤ y' ∧ y' ≤ 1/2) → 
      (xy / (x^2 + y^2) ≤ x'y' / (x'^2 + y'^2))) ∧ 
      xy / (x^2 + y^2) = 2/5 :=
sorry

end min_fraction_in_domain_l281_281411


namespace SequenceOddIncreasingEvenDecreasing_l281_281917

theorem SequenceOddIncreasingEvenDecreasing (a : ℝ) (x : ℕ → ℝ) 
  (h₀ : 0 < a ∧ a < 1) 
  (h₁ : ∀ n, x 1 = a) 
  (h₂ : ∀ n, x (n + 1) = a ^ (x n)): 
  (∀ n, x (2*n + 1) < x (2*n + 3)) ∧ (∀ n, x (2*n + 2) > x (2*n + 4)) := 
sorry

end SequenceOddIncreasingEvenDecreasing_l281_281917


namespace incorrect_expression_l281_281491

theorem incorrect_expression (x y : ℝ) (h : x > y) : ¬ (3 - x > 3 - y) :=
by
  sorry

end incorrect_expression_l281_281491


namespace correct_statements_l281_281661

/--
Question: Which of the following statements is correct?
Statements under consideration:
1. A quadrilateral with a pair of parallel and equal opposite sides is a parallelogram.
2. A parallelogram with a pair of equal adjacent sides is a square.
3. A quadrilateral with equal diagonals is a rectangle.
4. A quadrilateral with supplementary angles is inscribed in a circle.

Correct Option: Statements 1 and 4 are true.
-/

-- Definitions derived from conditions
def quadrilateral (A B C D : Type*) := Quadrilateral A B C D
def parallelogram (A B C D : Type*) := Parallelogram A B C D
def square (A B C D : Type*) := Square A B C D
def rectangle (A B C D : Type*) := Rectangle A B C D
def inscribable (A B C D : Type*) := InscribableInCircle A B C D

-- Conditions in Lean
axiom parallel_and_equal_opposite_sides_imp_parallelogram 
  (A B C D : Type*) (q : quadrilateral A B C D) 
  (h1 : parallel A B C D) (h2 : equal opposite sides A B C D) : parallelogram A B C D

axiom equal_adjacent_sides_in_parallelogram 
  (A B C D : Type*) (p : parallelogram A B C D) (h : equal adjacent sides A B C D) : rhombus A B C D

axiom equal_diagonals_imp_rectangle 
  (A B C D : Type*) (q : quadrilateral A B C D) (h : equal diagonals A B C D) : rectangle A B C D

axiom supplementary_angles_inscribed_in_circle 
  (A B C D : Type*) (q : quadrilateral A B C D) (h : supplementary angles A B C D) : inscribable A B C D

-- The theorem confirming statements 1 and 4 are true.
theorem correct_statements 
  (A B C D : Type*) (q : quadrilateral A B C D)
  (h1 : parallel A B C D) (h2 : equal opposite sides A B C D)
  (h3 : supplementary angles A B C D) :
  parallelogram A B C D ∧ inscribable A B C D :=
sorry

end correct_statements_l281_281661


namespace multiples_of_seven_with_units_digit_three_l281_281071

theorem multiples_of_seven_with_units_digit_three :
  ∃ n : ℕ, n = 2 ∧ ∀ k : ℕ, (k < 150 ∧ k % 7 = 0 ∧ k % 10 = 3) ↔ (k = 63 ∨ k = 133) := by
  sorry

end multiples_of_seven_with_units_digit_three_l281_281071


namespace trainer_area_coverage_l281_281313

theorem trainer_area_coverage (side : ℝ) (radius : ℝ) : 
  side = 25 → radius = 140 → 
  ((π * radius^2) / 4) ≈ 15393.8 :=
by 
  intros h1 h2
  rw [h1, h2]
  sorry

end trainer_area_coverage_l281_281313


namespace numberOfIntersectionsIsFour_l281_281226

noncomputable def pointOfIntersection (f g : ℝ → ℝ) : ℝ × ℝ := sorry

-- Definitions of linear equations derived from given problem
def line1 (x : ℝ) : ℝ := x - 2
def line2 (x : ℝ) : ℝ := -3 * x + 4
def line3 (x : ℝ) : ℝ := -x + 2
def line4 (x : ℝ) : ℝ := (2 / 5) * x + (7 / 5)

-- Function to count distinct points of intersection
def countDistinctIntersections : ℕ :=
    let p1 := pointOfIntersection line1 line3
    let p2 := pointOfIntersection line1 line4
    let p3 := pointOfIntersection line2 line3
    let p4 := pointOfIntersection line2 line4
    sorry

-- Proof statement
theorem numberOfIntersectionsIsFour : countDistinctIntersections = 4 :=
by
  sorry

end numberOfIntersectionsIsFour_l281_281226


namespace fully_factor_expression_l281_281425

theorem fully_factor_expression (x : ℝ) : 3 * x^2 - 75 = 3 * (x + 5) * (x - 5) := 
by
  -- pending proof, represented by sorry
  sorry

end fully_factor_expression_l281_281425


namespace lowest_degree_polynomial_is_4_l281_281264

noncomputable def lowest_degree_polynomial : ℕ :=
  let exists_polynomial_of_degree_four_with_conditions := ∃ (P : Polynomial ℤ), P.degree = 4 ∧
    ∃ b : ℤ, ∀ coeff ∈ P.coeffs, (coeff < b ∨ coeff > b) ∧ coeff ≠ b
  let no_polynomial_of_degree_less_than_four_with_conditions := ∀ (d < 4), ¬∃ (P : Polynomial ℤ), P.degree = d ∧
    ∃ b : ℤ, ∀ coeff ∈ P.coeffs, (coeff < b ∨ coeff > b) ∧ coeff ≠ b
  if h₁ : exists_polynomial_of_degree_four_with_conditions ∧ no_polynomial_of_degree_less_than_four_with_conditions then 4 else 0

theorem lowest_degree_polynomial_is_4 :
  lowest_degree_polynomial = 4 :=
by
  sorry

end lowest_degree_polynomial_is_4_l281_281264


namespace sum_of_palindromic_primes_lt_200_l281_281946

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_palindromic (n : ℕ) : Prop :=
  let str_n := n.toString
  str_n = str_n.reverse

def is_palindromic_prime (n : ℕ) : Prop :=
  is_prime n ∧ is_palindromic n ∧ is_prime (n.toString.reverse.toNat)

def palindromic_primes_less_than_200 : List ℕ :=
  [101, 113, 131, 151, 181, 191]

theorem sum_of_palindromic_primes_lt_200 : (palindromic_primes_less_than_200.foldl (· + ·) 0) = 868 := by
  sorry

end sum_of_palindromic_primes_lt_200_l281_281946


namespace jackson_vacuuming_time_l281_281136

-- Definitions based on the conditions
def hourly_wage : ℕ := 5
def washing_dishes_time : ℝ := 0.5
def cleaning_bathroom_time : ℝ := 3 * washing_dishes_time
def total_earnings : ℝ := 30

-- The total time spent on chores
def total_chore_time (V : ℝ) : ℝ :=
  2 * V + washing_dishes_time + cleaning_bathroom_time

-- The main theorem that needs to be proven
theorem jackson_vacuuming_time :
  ∃ V : ℝ, hourly_wage * total_chore_time V = total_earnings ∧ V = 2 :=
by
  sorry

end jackson_vacuuming_time_l281_281136


namespace acute_angle_of_parallel_vectors_l281_281540

-- Given definitions
def a (α : ℝ) : ℝ × ℝ := (3/2, Real.sin α)
def b (α : ℝ) : ℝ × ℝ := (Real.cos α, 1/3)

-- Parallel vectors definition
def are_parallel (u v : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, u = (k * v.1, k * v.2)

-- The theorem to prove
theorem acute_angle_of_parallel_vectors (α : ℝ) (h : are_parallel (a α) (b α)) : α = π / 4 :=
  sorry

end acute_angle_of_parallel_vectors_l281_281540


namespace fencing_cost_correct_l281_281602

noncomputable def length : ℝ := 80
noncomputable def diff : ℝ := 60
noncomputable def cost_per_meter : ℝ := 26.50

-- Let's calculate the breadth first
noncomputable def breadth : ℝ := length - diff

-- Calculate the perimeter
noncomputable def perimeter : ℝ := 2 * (length + breadth)

-- Calculate the total cost
noncomputable def total_cost : ℝ := perimeter * cost_per_meter

theorem fencing_cost_correct : total_cost = 5300 := 
by 
  sorry

end fencing_cost_correct_l281_281602


namespace intersection_of_M_and_N_l281_281834

def M : Set ℤ := { x | -3 < x ∧ x < 3 }
def N : Set ℤ := { x | x < 1 }

theorem intersection_of_M_and_N : M ∩ N = {-2, -1, 0} := by
  sorry

end intersection_of_M_and_N_l281_281834


namespace find_ellipse_equation_l281_281596

theorem find_ellipse_equation (F1 F2 : ℝ × ℝ) (d : ℝ) (a b : ℝ) :
  F1 = (-5, 0) → F2 = (5, 0) → d = 26 → 2 * a = d → a = 13 →
  let c := 5 in
  let b_sq := a^2 - c^2 in
  b^2 = b_sq →
  b = sqrt 144 →
  ∃ (x y : ℝ), (x, y).fst^2 / a^2 + (x, y).snd^2 / b^2 = 1 :=
by
  sorry

end find_ellipse_equation_l281_281596


namespace a_n_general_term_T_n_sum_l281_281832

-- Define the sequence a_n
def a (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n+1 => 2 * (Finset.range n).sum a + 1

-- Define the sequence b_n
def b (n : ℕ) : ℕ :=
  3^(n-1) + 2*n - 1

-- Define the sum T_n of first n terms of b_n
def T (n : ℕ) : ℤ :=
  (Finset.range n).sum b

-- Proof statements
theorem a_n_general_term (n : ℕ) : a n = 3^(n-1) := sorry

theorem T_n_sum (n : ℕ) : T n = (3^n / 2 : ℤ) + n^2 - 1/2 := sorry

end a_n_general_term_T_n_sum_l281_281832


namespace diagonal_length_of_pentagon_l281_281005

theorem diagonal_length_of_pentagon (side : ℝ) (h_side : side = 12) : 
  ∃ (DB : ℝ), DB = 6 * real.sqrt (6 + 2 * real.sqrt 5) :=
by
  -- See that the length of each side of the regular pentagon is 12 units.
  have h1: side = 12 := h_side

  -- Let DB be the length of the diagonal.
  let DB := 6 * real.sqrt (6 + 2 * real.sqrt 5)

  -- The problem setup ensures that the value of DB is indeed based on the analysis provided.
  use DB
  -- Proof goes here...

  sorry

end diagonal_length_of_pentagon_l281_281005


namespace marble_ratio_is_two_to_one_l281_281753

-- Conditions
def dan_blue_marbles : ℕ := 5
def mary_blue_marbles : ℕ := 10

-- Ratio definition
def marble_ratio : ℚ := mary_blue_marbles / dan_blue_marbles

-- Theorem statement
theorem marble_ratio_is_two_to_one : marble_ratio = 2 :=
by 
  -- Prove the statement here
  sorry

end marble_ratio_is_two_to_one_l281_281753


namespace problem_statement_l281_281189

open Nat

theorem problem_statement (n : ℕ) (h : 0 < n) :
  ∑ k in finset.range (2 * n + 2), (1 : ℝ) / (n + 1 + k) > 1 := sorry

end problem_statement_l281_281189


namespace find_special_numbers_l281_281943

theorem find_special_numbers : 
  ∃ (n1 n2 : ℕ), 
    (100 ≤ n1 ∧ n1 < 1000 ∧ 
    (∀ (d1 d2 d3 : ℕ), n1 = 100 * d1 + 10 * d2 + d3 → d1 ≠ d2 ∧ d2 ≠ d3 ∧ d1 ≠ d3 ∧ d1 < d2 ∧ d2 < d3 ∧ 
    (String.head (String.words (to_string n1)).getD "" = String.head (String.words (to_string n1)).getD ""))) ∧
    (100 ≤ n2 ∧ n2 < 1000 ∧
    (∀ (d : ℕ), n2 = d * 111 → 
    let name_words := String.words (to_string n2) in
    (String.head name_words.head = "O") ∧ 
    (String.head name_words.tail.headD "" = "H") ∧ 
    (String.head (name_words.drop 2).headD "" = "E"))) ∧
    n1 = 147 ∧ n2 = 111 := by {
  sorry
}

end find_special_numbers_l281_281943


namespace jean_pants_tax_percentage_l281_281887

theorem jean_pants_tax_percentage :
  ∀ (pairs: ℕ) (original_price: ℝ) (discount_percentage: ℝ) (total_price_after_tax: ℝ),
    pairs = 10 →
    original_price = 45 →
    discount_percentage = 0.2 →
    total_price_after_tax = 396 →
    let discounted_price := original_price * (1 - discount_percentage) in
    let total_discounted_price := discounted_price * pairs in
    let tax_amount := total_price_after_tax - total_discounted_price in
    (tax_amount / total_discounted_price) * 100 = 10 :=
begin
  intros,
  simp [discounted_price, total_discounted_price, tax_amount],
  sorry
end

end jean_pants_tax_percentage_l281_281887


namespace inequality_relationship_l281_281792

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

def a := log_base 2 3.6
def b := log_base 4 3.2
def c := log_base 4 3.6

theorem inequality_relationship : a > c ∧ c > b := 
by 
  -- To be defined as per the logic and proof required 
  sorry

end inequality_relationship_l281_281792


namespace problem_1_problem_2_l281_281794

namespace VectorProblem

variable (a b : Vector ℝ) (m : ℝ) (c d : Vector ℝ)

def question_1 (h1 : ∥a∥ = 3) (h2 : ∥b∥ = 2) (h3 : angle a b = π / 3)
    (h4 : c = 3•a + 5•b) (h5 : d = m•a - 3•b) : Prop :=
  c ⬝ d = 0

def answer_1 : ℝ := 29 / 14

theorem problem_1 (h1 : ∥a∥ = 3) (h2 : ∥b∥ = 2) (h3 : angle a b = π / 3)
    (h4 : c = 3•a + 5•b) (h5 : d = answer_1•a - 3•b) : question_1 a b answer_1 c d h1 h2 h3 h4 h5 := 
sorry

def question_2 (h1 : ∥a∥ = 3) (h2 : ∥b∥ = 2) (h3 : angle a b = π / 3)
    (h4 : c = 3•a + 5•b) (h5 : d = m•a - 3•b) : Prop :=
  ∃ λ : ℝ,  c = λ•d

def answer_2 : ℝ := -9 / 5

theorem problem_2 (h1 : ∥a∥ = 3) (h2 : ∥b∥ = 2) (h3 : angle a b = π / 3)
    (h4 : c = 3•a + 5•b) (h5 : d = answer_2•a - 3•b) : question_2 a b answer_2 c d h1 h2 h3 h4 h5 := 
sorry

end VectorProblem

end problem_1_problem_2_l281_281794


namespace radius_of_circle_l281_281705

theorem radius_of_circle (P Q : ℝ) (h : P / Q = 25) : ∃ r : ℝ, 2 * π * r = Q ∧ π * r^2 = P ∧ r = 50 := 
by
  -- Proof starts here
  sorry

end radius_of_circle_l281_281705


namespace math_problem_statement_l281_281470

-- Define the existence of the polar coordinate system and its conditions
def PolarCoordinateSystem (O pole : ℝ × ℝ) (x_axis : ℝ) : Prop := 
  O = (0, 0) ∧ pole = (1, 0) ∧ x_axis = π / 2

-- Define the polar equation and the point M's coordinates
def PointMAndCurve (M : ℝ × ℝ) (ρ θ : ℝ) : Prop := 
  M = (2, β) ∧ ρ = 2 * Real.sin θ ∧ θ ∈ Set.Ico 0 (2 * π)

-- 1. Find the rectangular coordinate equation of curve C.
def RectangularCoordinateEquation (x y : ℝ) : Prop := 
  (x-0)^2 + (y-1)^2 = 1^2

-- 2. Given the chord length, find β and the ratio of arc lengths.
def ChordLengthAndArcRatio (chordLength β : ℝ) (arc_ratio : ℚ) : Prop :=
  chordLength = Real.sqrt 3 ∧ (β = Real.arcsin (Real.sqrt 3 / 2) ∨ β = 2*Real.arcsin (Real.sqrt 3 / 2)) ∧ arc_ratio = 3

-- The main theorem stating the equivalence
theorem math_problem_statement (O pole : ℝ × ℝ) (x_axis β chordLength : ℝ) (arc_ratio : ℚ) (M : ℝ × ℝ) (ρ θ x y : ℝ) 
  (h1 : PolarCoordinateSystem O pole x_axis)
  (h2 : PointMAndCurve M ρ θ)
  (h3 : ChordLengthAndArcRatio chordLength β arc_ratio) :
  RectangularCoordinateEquation x y ∧ chordLength = Real.sqrt 3 ∧ arc_ratio = 3 :=
by
  sorry

end math_problem_statement_l281_281470


namespace distance_to_right_directrix_l281_281800

-- Distance from a point on the ellipse to its right directrix
theorem distance_to_right_directrix (x y : ℝ)
  (h1 : (x^2) / 25 + (y^2) / 9 = 1) (h2 : abs (x + √16) = 4) :
  let e : ℝ := 4 / 5,
      p : ℝ := 6,
      d : ℝ := 15 / 2 in 
  d = 15 / 2 :=
by 
  -- The proof is skipped.
  sorry

end distance_to_right_directrix_l281_281800


namespace sum_palindromic_primes_eq_1383_l281_281953

def is_prime (n : ℕ) : Prop := sorry -- Definition of prime numbers can be imported from Mathlib

def is_palindromic (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

def is_palindromic_prime (n : ℕ) : Prop :=
  is_prime n ∧ is_palindromic n ∧ n < 200 ∧ 100 ≤ n

def sum_palindromic_primes : ℕ :=
  (List.range 1000).filter is_palindromic_prime |> List.sum

theorem sum_palindromic_primes_eq_1383 : sum_palindromic_primes = 1383 :=
  sorry

end sum_palindromic_primes_eq_1383_l281_281953


namespace sum_of_numbers_in_range_l281_281645

def is_in_range (n : ℕ) : Prop := 28 < n ∧ n ≤ 31

theorem sum_of_numbers_in_range :
  (∑ n in Finset.filter is_in_range (Finset.range 32), n) = 90 :=
by
  sorry

end sum_of_numbers_in_range_l281_281645


namespace optionA_optionC_l281_281908

variables (a b : ℝ) (hp : a > 0) (hq : b > 0)

def A (a b : ℝ) : ℝ := (a + b) / 2
def G (a b : ℝ) : ℝ := Real.sqrt (a * b)
def L (p a b : ℝ) : ℝ := (a^p + b^p) / (a^(p-1) + b^(p-1))

-- Prove L_{0.5}(a, b) ≤ A(a, b)
theorem optionA : L 0.5 a b ≤ A a b :=
sorry

-- Prove L_2(a, b) ≥ L_1(a, b)
theorem optionC : L 2 a b ≥ L 1 a b :=
sorry

end optionA_optionC_l281_281908


namespace zero_not_in_range_of_g_l281_281158

def g (x : ℝ) : ℤ :=
  if x > -3 then (Real.ceil (2 / (x + 3)))
  else if x < -3 then (Real.floor (2 / (x + 3)))
  else 0 -- g(x) is not defined at x = -3, hence this is a placeholder

noncomputable def range_g : Set ℤ := {n | ∃ x : ℝ, g x = n}

theorem zero_not_in_range_of_g : 0 ∉ range_g :=
by
  intros h,
  exact sorry

end zero_not_in_range_of_g_l281_281158


namespace highest_wave_height_l281_281364

-- Definitions of surfboard length and shortest wave conditions
def surfboard_length : ℕ := 7
def shortest_wave_height (H : ℕ) : ℕ := H + 4

-- Lean statement to be proved
theorem highest_wave_height (H : ℕ) (condition1 : H + 4 = surfboard_length + 3) : 
  4 * H + 2 = 26 :=
sorry

end highest_wave_height_l281_281364


namespace johns_age_l281_281007

variable (J : ℕ)

theorem johns_age :
  J - 5 = (1 / 2) * (J + 8) → J = 18 := by
    sorry

end johns_age_l281_281007


namespace ellipse_eq_slope_value_l281_281029

open Real

-- Given initial conditions
def ellipse (x y a b : ℝ) := (x^2 / a^2 + y^2 / b^2 = 1)
def vertex (x y : ℝ) := (x = 0 ∧ y = 1)
def semi_minor_axis := (b = 1)
def focal_length := (c = sqrt 3)
def line_eq (y x k : ℝ) := (y = k * x - 2)
def intersection_length (BC_len : ℝ) := (BC_len = 4 * sqrt 2 / 5)

-- Prove the equation of the ellipse given the conditions
theorem ellipse_eq (a b c x y k : ℝ) (h1 : b = 1)
    (h2 : c = sqrt 3) (h3 : vertex 0 1)
    (h4 : a^2 = 4) : ellipse x y a b :=
begin
  have b_pos : b > 0 := by linarith[h1],
  have a_pos : a > b := by linarith[h4, h1],
  calc x^2 / 4 + y^2 = 1 : sorry
end

-- Prove the value of k is ±1 given the conditions and intersection length|BC| = 4√2/5
theorem slope_value (a b c x y k : ℝ) (BC_len : ℝ)
    (h1 : b = 1) (h2 : c = sqrt 3)
    (h3 : semi_minor_axis)
    (h4 : focal_length)
    (h5 : a^2 = 4)
    (h6 : intersection_length BC_len) : k = 1 ∨ k = -1 :=
begin
  have b_pos : b > 0 := by linarith[h1],
  have a_pos : a > b := by linarith[h5, h1],
  calc y = k * x - 2 : sorry,
  apply sorry
end

end ellipse_eq_slope_value_l281_281029


namespace ratio_apples_peaches_l281_281621

theorem ratio_apples_peaches (total_fruits oranges peaches apples : ℕ)
  (h_total : total_fruits = 56)
  (h_oranges : oranges = total_fruits / 4)
  (h_peaches : peaches = oranges / 2)
  (h_apples : apples = 35) : apples / peaches = 5 := 
by
  sorry

end ratio_apples_peaches_l281_281621


namespace arithmetic_b_seq_sum_a_n_l281_281874

noncomputable def a : ℕ → ℕ
| 0       := 1
| (n + 1) := 2 * a n + 2^n

def b (n : ℕ) : ℕ := a n / 2^(n-1)

theorem arithmetic_b_seq : ∀ n : ℕ, b (n + 1) - b n = 1 := sorry

theorem sum_a_n (n : ℕ) : (∑ i in range n.succ, a i) = (n - 1) * 2^n + 1 := sorry

end arithmetic_b_seq_sum_a_n_l281_281874


namespace total_time_over_weekend_l281_281525

def time_per_round : ℕ := 30
def rounds_saturday : ℕ := 11
def rounds_sunday : ℕ := 15

theorem total_time_over_weekend :
  (rounds_saturday * time_per_round) + (rounds_sunday * time_per_round) = 780 :=
by
  -- This is where the proof would go, but it is omitted as per instructions.
  sorry

end total_time_over_weekend_l281_281525


namespace triangles_containing_C_interior_l281_281535

-- Define the set of vertices of a regular 25-sided polygon
def V : set (Point) := sorry

-- Define the center point C of the polygon
def C : Point := sorry

-- State the theorem to be proved
theorem triangles_containing_C_interior : 
  (number_of_triangles_containing_point V C) = 925 := 
  sorry

end triangles_containing_C_interior_l281_281535


namespace minimum_value_abs_diff_l281_281844

variable {x y : ℝ}

def is_valid (x y : ℝ) : Prop :=
  log 4 (x + 2 * y) + log 1 (x - 2 * y) = 1 ∧ x + 2 * y > 0 ∧ x - 2 * y > 0

theorem minimum_value_abs_diff (x y : ℝ) (h : is_valid x y) :
  |x| - |y| = √3 :=
sorry

end minimum_value_abs_diff_l281_281844


namespace maddie_total_payment_l281_281930

def makeup_cost := 3 * 15
def lipstick_cost := 4 * 2.50
def hair_color_cost := 3 * 4

def total_cost := makeup_cost + lipstick_cost + hair_color_cost

theorem maddie_total_payment : total_cost = 67 := by
  sorry

end maddie_total_payment_l281_281930


namespace maddie_total_cost_l281_281931

theorem maddie_total_cost :
  let price_palette := 15
  let price_lipstick := 2.5
  let price_hair_color := 4
  let num_palettes := 3
  let num_lipsticks := 4
  let num_hair_colors := 3
  let total_cost := (num_palettes * price_palette) + (num_lipsticks * price_lipstick) + (num_hair_colors * price_hair_color)
  total_cost = 67 := by
  sorry

end maddie_total_cost_l281_281931


namespace inequality_inequality_only_if_k_is_one_half_l281_281774

theorem inequality_inequality_only_if_k_is_one_half :
  (∀ t : ℝ, -1 < t ∧ t < 1 → (1 + t) ^ k * (1 - t) ^ (1 - k) ≤ 1) ↔ k = 1 / 2 :=
by
  sorry

end inequality_inequality_only_if_k_is_one_half_l281_281774


namespace lowest_degree_polynomial_l281_281290

-- Define the conditions
def polynomial_conditions (P : ℕ → ℤ) (b : ℤ): Prop :=
  (∃ c, c > b ∧ c ∈ set.range P) ∧ (∃ d, d < b ∧ d ∈ set.range P) ∧ (b ∉ set.range P)

-- The main statement
theorem lowest_degree_polynomial : ∃ P : ℕ → ℤ, polynomial_conditions P 4 ∧ (∀ Q : ℕ → ℤ, polynomial_conditions Q 4 → degree Q >= 4) :=
sorry

end lowest_degree_polynomial_l281_281290


namespace area_of_triangle_ABC_l281_281880

-- All necessary conditions for our proof problem
variables {A B C D E : Type}
variables [metric_space E]
variable (A : E) (B : E) (C : E) (D : E)
variable [linear_ordered_semifield E]
variable (BE : ℝ)

-- Define conditions
def condition1: Prop := dist A B = dist B C
def condition2: Prop := ∠A B D = 90 -- altitude implies right angle
def condition3: Prop := dist B E = 12
def condition4: Prop := ∠D B E = 45

-- Define the statement we want to prove
theorem area_of_triangle_ABC (A B C D E : E) :
  condition1 A B C → condition2 A B D → condition3 B E → condition4 D B E →
  area_triangle A B C = 144 :=
sorry

end area_of_triangle_ABC_l281_281880


namespace number_of_children_is_30_l281_281567

-- Informal statements
def total_guests := 80
def men := 40
def women := men / 2
def adults := men + women
def children := total_guests - adults
def children_after_adding_10 := children + 10

-- Formal proof statement
theorem number_of_children_is_30 :
  children_after_adding_10 = 30 := by
  sorry

end number_of_children_is_30_l281_281567


namespace distance_ratio_l281_281218

theorem distance_ratio (x : ℝ) (hx : abs x = 8) : abs (-4) / abs x = 1 / 2 :=
by {
  sorry
}

end distance_ratio_l281_281218


namespace sum_b_and_c_l281_281053

theorem sum_b_and_c (b c : ℝ) :
  (f : ℝ → ℝ) = (λ x, real.logb 3 ((2 * x ^ 2 + b * x + c) / (x ^ 2 + 1)))
  ∧ (∀ x, 0 ≤ f x ∧ f x ≤ 1) →
  b + c = 0 ∨ b + c = 4 :=
sorry

end sum_b_and_c_l281_281053


namespace area_of_triangle_with_rational_vertices_on_unit_circle_is_rational_l281_281392

def rational_coords_on_unit_circle (x₁ y₁ x₂ y₂ x₃ y₃ : ℚ) : Prop :=
  x₁^2 + y₁^2 = 1 ∧ x₂^2 + y₂^2 = 1 ∧ x₃^2 + y₃^2 = 1

theorem area_of_triangle_with_rational_vertices_on_unit_circle_is_rational
  (x₁ y₁ x₂ y₂ x₃ y₃ : ℚ)
  (h : rational_coords_on_unit_circle x₁ y₁ x₂ y₂ x₃ y₃) :
  ∃ (A : ℚ), A = 1 / 2 * abs (x₁ * (y₂ - y₃) + x₂ * (y₃ - y₁) + x₃ * (y₁ - y₂)) :=
sorry

end area_of_triangle_with_rational_vertices_on_unit_circle_is_rational_l281_281392


namespace lowest_degree_is_4_l281_281278

noncomputable def lowest_degree_polynomial (P : Polynomial ℤ) (b : ℤ) : Prop :=
  ∃ (b : ℤ), 
    let A_P := P.support in
    (∀ (a ∈ A_P), a < b ∨ a > b) ∧ 
    (¬(b ∈ A_P)) ∧
    (∃ (a1 a2 : ℤ), a1 ∈ A_P ∧ a2 ∈ A_P ∧ a1 < b ∧ a2 > b)

theorem lowest_degree_is_4 :
  ∀ P : Polynomial ℤ, 
    let b := lowest_degree_polynomial P in
    b P 4 :=
sorry

end lowest_degree_is_4_l281_281278


namespace solution_set_inequality_l281_281231

theorem solution_set_inequality (a x : ℝ) (h : a < -1) :
  (ax - 1) / (x + 1) < 0 ↔ x ∈ Iio (-1) ∪ Ioi (1 / a) :=
by
  sorry

end solution_set_inequality_l281_281231


namespace square_area_l281_281182

theorem square_area (x : ℝ) (h1 : BG = GH) (h2 : GH = HD) (h3 : BG = 20 * Real.sqrt 2) : x = 40 * Real.sqrt 2 → x^2 = 3200 :=
by
  sorry

end square_area_l281_281182


namespace certain_number_l281_281099

theorem certain_number (w certain_num : ℕ) 
  (h_w_positive : 0 < w) 
  (h_w_smallest : w = 468) 
  (h_factors : ∀ k, [16, 27, 2197].∀ (λ x, x ∣ (certain_num * w))) 
  : certain_num = 2028 := 
begin
  sorry
end

end certain_number_l281_281099


namespace fedya_deposit_l281_281244

theorem fedya_deposit (n : ℕ) (h1 : n < 30) (h2 : 847 * 100 % (100 - n) = 0) : 
  (847 * 100 / (100 - n) = 1100) :=
by
  sorry

end fedya_deposit_l281_281244


namespace sum_geq_n_l281_281903

noncomputable def sequence (a : ℕ → ℝ) : Prop :=
  ∀ k : ℕ, k > 0 → a k > 0 ∧ a (k + 1) ≥ k * a k / (a k ^ 2 + (k - 1))

theorem sum_geq_n (a : ℕ → ℝ) (h : sequence a) :
  ∀ n : ℕ, n ≥ 2 → ∑ i in Finset.range n, a (i + 1) ≥ n := by
  sorry

end sum_geq_n_l281_281903


namespace unique_decomposition_base4_l281_281964

def in_base4_A (n : ℕ) : Prop :=
  ∀ m < n, (m % 4 = 0 ∨ m % 4 = 2)

def in_base4_B (n : ℕ) : Prop :=
  ∀ m < n, (m % 4 = 0 ∨ m % 4 = 1)

theorem unique_decomposition_base4 (n : ℕ) :
  ∃ A B : Set ℕ, (∀ a, a ∈ A ↔ in_base4_A a) ∧ (∀ b, b ∈ B ↔ in_base4_B b) ∧
  infinite A ∧ infinite B ∧
  ∀ n, ∃! (a b : ℕ), a ∈ A ∧ b ∈ B ∧ n = a + b := 
by sorry

-- Usage of Set.infinite to represent that A and B are infinite sets

end unique_decomposition_base4_l281_281964


namespace sqrt_log_sum_l281_281767

theorem sqrt_log_sum :
  sqrt (log 2 8 + log 4 16) = sqrt 5 :=
by
  sorry

end sqrt_log_sum_l281_281767


namespace sin_cos_identity_l281_281016

variable (α : Real)

theorem sin_cos_identity (h : Real.sin α - Real.cos α = -5/4) : Real.sin α * Real.cos α = -9/32 :=
by
  sorry

end sin_cos_identity_l281_281016


namespace larry_wins_probability_l281_281531

noncomputable def probability (n : ℕ) : ℝ :=
  if n % 2 = 1 then (1/2)^(n) else 0

noncomputable def inf_geometric_sum (a : ℝ) (r : ℝ) : ℝ :=
  a / (1 - r)

theorem larry_wins_probability :
  inf_geometric_sum (1/2) (1/4) = 2/3 :=
by
  sorry

end larry_wins_probability_l281_281531


namespace find_smallest_solution_l281_281783

noncomputable def equation (x : ℝ) : Prop :=
  1 / (x - 3) + 1 / (x - 5) + 1 / (x - 7) = 4 / (x - 4)

noncomputable def smallest_solution : ℝ :=
  Inf (set_of (λ x : ℝ, equation x) \ {3, 5, 7, 4})

theorem find_smallest_solution :
  ∃ x : ℝ, equation x ∧ x = smallest_solution :=
sorry

end find_smallest_solution_l281_281783


namespace count_six_digit_numbers_with_seven_l281_281020

theorem count_six_digit_numbers_with_seven : 
  let digits := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
  let is_six_digit (n : ℕ) := 100000 ≤ n ∧ n < 1000000
  let contains_seven (n : ℕ) := n.digits 10 contains 7
  let count_numbers := {n : ℕ | is_six_digit n ∧ contains_seven n}.card
  count_numbers = 2002 :=
begin
  sorry
end

end count_six_digit_numbers_with_seven_l281_281020


namespace exponent_problem_l281_281640

theorem exponent_problem : (3^{-3})^0 + (3^0)^{-2} = 2 :=
by
  sorry

end exponent_problem_l281_281640


namespace Glenburgh_parade_squad_l281_281210

theorem Glenburgh_parade_squad : 
  ∃ m : ℤ, 20 * m < 1200 ∧ 20 * m % 28 = 6 ∧ 20 * m = 1160 :=
begin
  sorry
end

end Glenburgh_parade_squad_l281_281210


namespace ball_returns_to_bella_after_13_throws_l281_281627

theorem ball_returns_to_bella_after_13_throws :
  ∀ (n : ℕ) (girls : Fin 13) (throws : ℕ → Fin 13 → Fin 13),
    (throws 0 0 = 0) →
    (∀ k g, throws (k + 1) g = (throws k g + 6) % 13) →
    (throws 12 0 = 0) :=
by
  assume n girls throws
  intro h_start h_step
  sorry

end ball_returns_to_bella_after_13_throws_l281_281627


namespace brick_height_l281_281324

theorem brick_height (h : ℝ) : 
    let wall_length := 900
    let wall_width := 600
    let wall_height := 22.5
    let num_bricks := 7200
    let brick_length := 25
    let brick_width := 11.25
    wall_length * wall_width * wall_height = num_bricks * (brick_length * brick_width * h) -> 
    h = 67.5 := 
by
  intros
  sorry

end brick_height_l281_281324


namespace AF_BC_intersect_at_N_MN_passes_through_fixed_point_S_locus_of_midpoints_l281_281357

variables {M N A B C D E F P Q S: Point}
variables {segment_AB: LineSegment}
variables {AMCD MBEF : Square}
variables {circum_AMCD circum_MBEF : Circle}

/-
  Question 1 (a):
  Given M in the interior of AB, squares AMCD and MBEF constructed on the same side of AB,
  and circles circumscribed about these squares intersect at M and N.
  Prove that AF and BC intersect at N.
-/
theorem AF_BC_intersect_at_N (hM_in_AB: M ∈ segment_AB) 
  (h_squares_constructed : AMCD.is_square AMCD A M C D ∧ MBEF.is_square MBEF M B E F)
  (h_circles_intersect: circum_AMCD.intersects circum_MBEF at M and N) :
  intersects (line.through A F) (line.through B C) at N :=
by { sorry }

/-
  Question 1 (b):
  Given M in the interior of AB, squares AMCD and MBEF on the same side of AB,
  and circles circumscribed about these squares intersect at M and N.
  Prove that the lines MN pass through a fixed point S (independent of M).
-/
theorem MN_passes_through_fixed_point_S (hM_in_AB: M ∈ segment_AB) 
  (h_squares_constructed : AMCD.is_square AMCD A M C D ∧ MBEF.is_square MBEF M B E F)
  (h_circles_intersect: circum_AMCD.intersects circum_MBEF at M and N) :
  ∃ S, passes_through (line.through M N) S :=
by { sorry }

/-
  Question 1 (c):
  Given M in the interior of AB, squares AMCD and MBEF on the same side of AB,
  and circles circumscribed about these squares intersect at M and N.
  Prove that the locus of the midpoints of PQ as M varies is a line segment of length AB/2 centered over AB.
-/
theorem locus_of_midpoints (hM_in_AB: M ∈ segment_AB) 
  (h_squares_constructed : AMCD.is_square AMCD A M C D ∧ MBEF.is_square MBEF M B E F)
  (h_circles_intersect: circum_AMCD.intersects circum_MBEF at M and N) :
  locus (midpoint P Q) = central_segment (center AB) (length AB / 2) :=
by { sorry }

end AF_BC_intersect_at_N_MN_passes_through_fixed_point_S_locus_of_midpoints_l281_281357


namespace solve_for_m_l281_281824

noncomputable def f (x a : ℝ) : ℝ :=
2 * real.log x + x^2 - a * x + 2

def condition (a : ℝ) (m x0 : ℝ) : Prop :=
a ∈ Icc (-2 : ℝ) (0 : ℝ) ∧ x0 ∈ Icc (0 : ℝ) (1 : ℝ) ∧
f x0 a > a^2 + 3 * a + 2 - 2 * m * real.exp a * (a + 1)

theorem solve_for_m:
  ∀ a ∈ Icc (-2 : ℝ) (0 : ℝ),
  (∃ x0 ∈ Icc (0 : ℝ) (1 : ℝ), condition a m x0) ↔ m ∈ Icc (-1 / 2 : ℝ) (5 * real.exp (2 : ℝ) / 2) :=
by sorry

end solve_for_m_l281_281824


namespace oatmeal_cookies_l281_281559

theorem oatmeal_cookies (b c x : ℕ) (h_b : b = 6) (h_c : c = 3) (h_x : x = 2) :
  let total_cookies := b * c in
  let oatmeal_cookies := total_cookies - x in
  oatmeal_cookies = 16 :=
by 
  -- Definitions and equations
  let total_cookies := b * c
  let oatmeal_cookies := total_cookies - x
  -- Proof
  have h_tc : total_cookies = 18 := by linarith [h_b, h_c]
  have h_oatmeal : oatmeal_cookies = 16 := by linarith [h_tc, h_x]
  exact h_oatmeal

end oatmeal_cookies_l281_281559


namespace lowest_degree_is_4_l281_281279

noncomputable def lowest_degree_polynomial (P : Polynomial ℤ) (b : ℤ) : Prop :=
  ∃ (b : ℤ), 
    let A_P := P.support in
    (∀ (a ∈ A_P), a < b ∨ a > b) ∧ 
    (¬(b ∈ A_P)) ∧
    (∃ (a1 a2 : ℤ), a1 ∈ A_P ∧ a2 ∈ A_P ∧ a1 < b ∧ a2 > b)

theorem lowest_degree_is_4 :
  ∀ P : Polynomial ℤ, 
    let b := lowest_degree_polynomial P in
    b P 4 :=
sorry

end lowest_degree_is_4_l281_281279


namespace problem_statement_l281_281513

noncomputable def point := (ℝ × ℝ)

def A : point := (1, 3)
def B : point := (3, 1)
def C : point := (-1, 0)

def eq_line_AB : Prop := ∀ (x y : ℝ), x + y - 4 = 0

def eq_circle_C_tangent_to_AB : Prop := ∀ (x y : ℝ), (x + 1)^2 + y^2 = 25 / 2

theorem problem_statement :
  (eq_line_AB) ∧ (eq_circle_C_tangent_to_AB) :=
begin
  split,
  { -- Proof for eq_line_AB
    sorry },
  { -- Proof for eq_circle_C_tangent_to_AB
    sorry }
end

end problem_statement_l281_281513


namespace squared_difference_l281_281083

variable {x y : ℝ}

theorem squared_difference (h1 : (x + y)^2 = 81) (h2 : x * y = 18) :
  (x - y)^2 = 9 :=
by
  sorry

end squared_difference_l281_281083


namespace largest_angle_is_41_41_degrees_l281_281983

-- Define the triangle with sides such that the altitudes are 10, 12, and 15
noncomputable def triangle_sides (a b c : ℝ) : Prop :=
  let h1 := 10
  let h2 := 12
  let h3 := 15
  2 * (1 / (1/h1 + 1/h2 + 1/h3)) = a * b * c

-- Prove that the largest angle is approximately cos⁻¹(3/4)
theorem largest_angle_is_41_41_degrees :
  ∃ (a b c : ℝ), triangle_sides a b c ∧
  let largest_angle := Real.arccos (3 / 4) in specific_angle_approx_eq largest_angle (41.41 * (π / 180)) :=
sorry

end largest_angle_is_41_41_degrees_l281_281983


namespace ratio_current_to_boat_l281_281688

def boat_sailing_condition (distance : ℕ) (time_upstream : ℕ) (time_downstream : ℕ) (b c : ℚ) : Prop :=
  (b - c = distance / time_upstream) ∧ (b + c = distance / time_downstream)

theorem ratio_current_to_boat (b c : ℚ) (h : boat_sailing_condition 15 5 3 b c) :
  c / b = 1 / 4 :=
  by sorry

end ratio_current_to_boat_l281_281688


namespace sum_even_102_to_200_l281_281672

theorem sum_even_102_to_200 : 
  (∑ i in finset.filter (λ x, x % 2 = 0) (finset.range (200 + 1)), i) = 7550 := 
sorry

end sum_even_102_to_200_l281_281672


namespace find_imaginary_part_l281_281058

-- Define the given complex number 'z'
noncomputable def z : ℂ := sorry

-- Define the condition
def condition (z : ℂ) : Prop := z * ((1 : ℂ) + complex.I)^2

-- State the proof problem
theorem find_imaginary_part (a b : ℝ) (hz : z = a + b * complex.I)
  (hcond : condition z) :
  complex.im (z * ((1 + complex.I) ^ 2)) = 2 * a :=
sorry

end find_imaginary_part_l281_281058


namespace smallest_positive_period_of_f_minimum_value_of_f_l281_281761

noncomputable def f : ℝ → ℝ := λ x, (Real.sin x) ^ 2 + (Real.sin x) * (Real.cos x) + 1

theorem smallest_positive_period_of_f : ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = Real.pi :=
begin
  sorry
end

theorem minimum_value_of_f : ∃ m, m = (3 - Real.sqrt 2) / 2 ∧ ∀ x, f x ≥ m :=
begin
  sorry
end

end smallest_positive_period_of_f_minimum_value_of_f_l281_281761


namespace sum_k_l281_281762

noncomputable def special_sum : ℤ :=
  (∑ k in {k | ∃ (a0 a1 a2 : ℤ), 
            a2 ≠ 0 ∧ 
            0 ≤ a0 ∧ a0 ≤ 4 ∧ 
            0 ≤ a1 ∧ a1 ≤ 4 ∧ 
            0 ≤ a2 ∧ a2 ≤ 4 ∧ 
            (-4 * a2 + a1 = 0) ∧ 
            k = (3 * a2 - 2 * a1 + a0)}, k)

theorem sum_k : special_sum = -15 :=
by sorry

end sum_k_l281_281762


namespace quadratic_root_c_l281_281199

theorem quadratic_root_c (α : ℝ) (b c : ℝ) (h : 4 * (tan α) * (3 * c * cot α) = c) : c = 12 :=
sorry

end quadratic_root_c_l281_281199


namespace sum_of_triangle_areas_in_cube_l281_281615

theorem sum_of_triangle_areas_in_cube (m n p : ℤ) (h : ∑ (areas : set ℝ) of triangles with vertices of 2 × 2 × 2 cube, areas = m + Real.sqrt n + Real.sqrt p) : m + n + p = 972 :=
sorry

end sum_of_triangle_areas_in_cube_l281_281615


namespace profit_in_terms_of_S_l281_281102

variables {C S P : ℝ} (p n : ℝ)

-- Conditions as definitions
def profit_formula1 : ℝ := p * (C + S) / 2
def profit_formula2 : ℝ := S / n - C

-- Theorem statement using the conditions
theorem profit_in_terms_of_S (h1 : P = profit_formula1 p C S)
                              (h2 : P = profit_formula2 S n C) :
  P = S * (2 * n * p + 2 * p - n) / (n * (2 * p + n)) :=
by
  sorry

end profit_in_terms_of_S_l281_281102


namespace lowest_degree_polynomial_is_4_l281_281262

noncomputable def lowest_degree_polynomial : ℕ :=
  let exists_polynomial_of_degree_four_with_conditions := ∃ (P : Polynomial ℤ), P.degree = 4 ∧
    ∃ b : ℤ, ∀ coeff ∈ P.coeffs, (coeff < b ∨ coeff > b) ∧ coeff ≠ b
  let no_polynomial_of_degree_less_than_four_with_conditions := ∀ (d < 4), ¬∃ (P : Polynomial ℤ), P.degree = d ∧
    ∃ b : ℤ, ∀ coeff ∈ P.coeffs, (coeff < b ∨ coeff > b) ∧ coeff ≠ b
  if h₁ : exists_polynomial_of_degree_four_with_conditions ∧ no_polynomial_of_degree_less_than_four_with_conditions then 4 else 0

theorem lowest_degree_polynomial_is_4 :
  lowest_degree_polynomial = 4 :=
by
  sorry

end lowest_degree_polynomial_is_4_l281_281262


namespace probability_AB_hired_l281_281101

theorem probability_AB_hired :
  let candidates := {A, B, C, D, E}
  let total_ways := Nat.choose 5 3
  let favorable_ways := Nat.choose 3 1
  let prob := (favorable_ways : ℚ) / total_ways
  prob = 3 / 10 :=
by
  sorry

end probability_AB_hired_l281_281101


namespace total_apples_picked_l281_281561

-- Definitions based on conditions from part a)
def mike_apples : ℝ := 7.5
def nancy_apples : ℝ := 3.2
def keith_apples : ℝ := 6.1
def olivia_apples : ℝ := 12.4
def thomas_apples : ℝ := 8.6

-- The theorem we need to prove
theorem total_apples_picked : mike_apples + nancy_apples + keith_apples + olivia_apples + thomas_apples = 37.8 := by
    sorry

end total_apples_picked_l281_281561


namespace joan_balloons_l281_281891

-- Defining the condition
def melanie_balloons : ℕ := 41
def total_balloons : ℕ := 81

-- Stating the theorem
theorem joan_balloons :
  ∃ (joan_balloons : ℕ), joan_balloons = total_balloons - melanie_balloons ∧ joan_balloons = 40 :=
by
  -- Placeholder for the proof
  sorry

end joan_balloons_l281_281891


namespace lowest_degree_polynomial_is_4_l281_281266

noncomputable def lowest_degree_polynomial : ℕ :=
  let exists_polynomial_of_degree_four_with_conditions := ∃ (P : Polynomial ℤ), P.degree = 4 ∧
    ∃ b : ℤ, ∀ coeff ∈ P.coeffs, (coeff < b ∨ coeff > b) ∧ coeff ≠ b
  let no_polynomial_of_degree_less_than_four_with_conditions := ∀ (d < 4), ¬∃ (P : Polynomial ℤ), P.degree = d ∧
    ∃ b : ℤ, ∀ coeff ∈ P.coeffs, (coeff < b ∨ coeff > b) ∧ coeff ≠ b
  if h₁ : exists_polynomial_of_degree_four_with_conditions ∧ no_polynomial_of_degree_less_than_four_with_conditions then 4 else 0

theorem lowest_degree_polynomial_is_4 :
  lowest_degree_polynomial = 4 :=
by
  sorry

end lowest_degree_polynomial_is_4_l281_281266


namespace symmetric_points_hyperbola_parabola_l281_281454

theorem symmetric_points_hyperbola_parabola (m : ℝ)
  (h1 : ∃ M N : ℝ×ℝ, (M.1^2 - M.2^2 = 4) ∧ (N.1^2 - N.2^2 = 4)
    ∧ (M.2 - 2 * M.1 + m = -(N.2 - 2 * N.1 + m)))
  (h2 : ∃ P : ℝ×ℝ, P.1 = (M.1 + N.1) / 2 ∧ P.2 = (M.2 + N.2) / 2
    ∧ P.2^2 = 16 * P.1) :
  m = 0 ∨ m = 16 :=
sorry

end symmetric_points_hyperbola_parabola_l281_281454


namespace minimize_sum_of_distances_on_x_axis_l281_281417

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  ((p.1 - q.1)^2 + (p.2 - q.2)^2).sqrt

theorem minimize_sum_of_distances_on_x_axis 
  (A B : ℝ × ℝ) (k : ℝ) 
  (hA : A = (7, 4))
  (hB : B = (3, -2)) :
  (minimizes (λ k, distance A (k, 0) + distance B (k, 0)) 5) := 
sorry

end minimize_sum_of_distances_on_x_axis_l281_281417


namespace total_length_of_tape_l281_281482

theorem total_length_of_tape 
  (n : ℕ) (tape_length : ℕ) (overlap : ℕ) (remaining_tapes : ℕ)
  (length_first_piece : ℕ := tape_length)
  (length_each_subsequent_piece : ℕ := tape_length - overlap) :
  n = 15 → tape_length = 20 → overlap = 5 → total_length_of_tape = 230 :=
by
  intros n tape_length overlap hn htape hoverlap
  sorry

end total_length_of_tape_l281_281482


namespace conditional_prob_B_given_A_l281_281424

-- Define the events for students running specific legs
def A_runs_first_leg : Prop := sorry -- Set the actual probability space and conditions
def B_runs_second_leg : Prop := sorry -- Set the actual probability space and conditions

-- Define the condition that A runs the first leg
axiom A_runs_first : A_runs_first_leg

-- Define the total number of students
def total_students := 4

-- Define the number of students available for the second leg after A runs the first leg
def available_students := total_students - 1

-- Define the probability that B runs the second leg given A runs the first leg
def P_B_given_A : ℝ := 1 / (available_students : ℝ)

-- Prove that P(B|A) = 1/3
theorem conditional_prob_B_given_A : P(B_runs_second_leg | A_runs_first_leg) = 1 / 3 := by
  sorry

end conditional_prob_B_given_A_l281_281424


namespace ratio_of_squares_area_sum_l281_281999

theorem ratio_of_squares_area_sum (a b c : ℤ) (hr : (300 : ℚ) / 75 = 4) :
  let side_length_ratio := 2
  let a := 2
  let b := 1
  let c := 1
  a + b + c = 4 :=
by {
  have h1: (300 : ℚ) / 75 = 4 := hr,
  have h2: side_length_ratio = 2 := by norm_num,
  let a := 2,
  let b := 1,
  let c := 1,
  calc a + b + c = 2 + 1 + 1 : by norm_num
              ... = 4 : by norm_num 
}

end ratio_of_squares_area_sum_l281_281999


namespace package_contains_12_rolls_l281_281693

-- Define the problem conditions
def package_cost : ℝ := 9
def individual_roll_cost : ℝ := 1
def savings_percent : ℝ := 0.25

-- Define the price per roll in the package
def price_per_roll_in_package := individual_roll_cost * (1 - savings_percent)

-- Define the number of rolls in the package
def number_of_rolls_in_package := package_cost / price_per_roll_in_package

-- Question: How many rolls are in the package?
theorem package_contains_12_rolls : number_of_rolls_in_package = 12 := by
  sorry

end package_contains_12_rolls_l281_281693


namespace find_f_sqrt2_div_2_l281_281017

def f (x : ℝ) : ℝ :=
  if x ≥ 1 then log x / log 2 else f (2 * x)

theorem find_f_sqrt2_div_2 :
  f (real.sqrt 2 / 2) = 1 / 2 :=
sorry

end find_f_sqrt2_div_2_l281_281017


namespace cyclic_quad_APXY_l281_281913

variables (A B C M P X Y : Type*) [MetricSpace A B C M P X Y]
variables (hAB_AC : dist A B = dist A C)
variables (hM : midpoint M B C)
variables (hP_parallel_BC : parallel (line_segment P A) (line_segment B C))
variables (hPB_lt_PC : dist P B < dist P C)
variables (hB_on_PX : ∃ px_pos px_end, B = point_on PX px_pos px_end)
variables (hC_on_PY : ∃ py_pos py_end, C = point_on PY py_pos py_end)
variables (hPXM_eq_PYM : ∠ PXM = ∠ PYM)

theorem cyclic_quad_APXY :
  cyclic_quad A P X Y :=
sorry

end cyclic_quad_APXY_l281_281913


namespace num_ordered_triples_eq_ten_l281_281335

theorem num_ordered_triples_eq_ten :
  ∃ (a b c : ℕ), 
  (1 ≤ a ∧ a ≤ b ∧ b ≤ c) ∧ 
  2 * (a * b + b * c + c * a) = a * b * c ∧ 
  { (a, b, c) | 1 ≤ a ∧ a ≤ b ∧ b ≤ c ∧ 2 * (a * b + b * c + c * a) = a * b * c }.to_finset.card = 10 :=
sorry

end num_ordered_triples_eq_ten_l281_281335


namespace triangle_area_tangent_l281_281985

theorem triangle_area_tangent (x : ℝ) (y : ℝ) : 
  (x = 2) → 
  (y = Real.exp x) →
  (Real.exp' x = Real.exp x) →
  let slope := Real.exp 2
  let tangent_line := λ x, slope * (x - 2) + Real.exp 2
  let intercept_y := tangent_line 0
  let intercept_x := 0
  intercept_y = Real.exp 2 ∧ intercept_x = 1 →
  (1 / 2) * intercept_x * intercept_y = (1 / 2) * Real.exp 2 := 
by
  sorry

end triangle_area_tangent_l281_281985


namespace binomial_probability_example_l281_281438

noncomputable def binomialProbability (n k : ℕ) (p : ℚ) : ℚ :=
  (nat.choose n k) * (p^k) * ((1 - p)^(n - k))

theorem binomial_probability_example :
  binomialProbability 5 3 (1/3) = 40 / 243 :=
by
  sorry  -- This is where the proof would go.

end binomial_probability_example_l281_281438


namespace range_of_m_l281_281835

noncomputable def set_S (m : ℝ) : set ℝ := {x | 1 - m <= x ∧ x <= 1 + m}
noncomputable def set_P : set ℝ := {x | -2 <= x ∧ x <= 10}

theorem range_of_m (m : ℝ) (hS : ∃ x, x ∈ set_S m) :
  (∀ x, x ∈ set_S m → x ∈ set_P) → 0 <= m ∧ m <= 3 :=
by
  intro h
  sorry

end range_of_m_l281_281835


namespace triangle_is_right_angled_l281_281225

noncomputable def median (a b c : ℝ) : ℝ := (1 / 2) * (Real.sqrt (2 * b^2 + 2 * c^2 - a^2))

theorem triangle_is_right_angled (a b c : ℝ) (ha : median a b c = 5) (hb : median b c a = Real.sqrt 52) (hc : median c a b = Real.sqrt 73) :
  a^2 = b^2 + c^2 :=
sorry

end triangle_is_right_angled_l281_281225


namespace distinct_flavors_l281_281556

theorem distinct_flavors (blue_candies : Nat) (orange_candies : Nat) (h₁ : blue_candies = 5) (h₂ : orange_candies = 4) : 
  ∃ (distinct_flavors : Nat), distinct_flavors = 17 :=
by
  let y := 4
  let x := 5
  have h₁ : x = 5
  have h₂ : y = 4
  use 17
  sorry

end distinct_flavors_l281_281556


namespace angle_ADE_105_l281_281957

theorem angle_ADE_105 (A B C D E : Point) :
  (C ∈ Segment A E) ∧
  (triangle A B C ∧ is_equilateral A B C) ∧
  (triangle C D E ∧ is_isosceles_right C D E D) ∧
  (triangle B C D ∧ is_isosceles B C D C) →
  angle ADE = 105 :=
by sorry

end angle_ADE_105_l281_281957


namespace graph_passes_through_fixed_point_l281_281563

-- Define the linear function given in the conditions
def linearFunction (k x y : ℝ) : ℝ :=
  (2 * k - 1) * x - (k + 3) * y - (k - 11)

-- Define the fixed point (2, 3)
def fixedPoint : ℝ × ℝ :=
  (2, 3)

-- State the theorem that the graph of the linear function always passes through the fixed point 
theorem graph_passes_through_fixed_point :
  ∀ k : ℝ, linearFunction k fixedPoint.1 fixedPoint.2 = 0 :=
by sorry  -- proof skipped

end graph_passes_through_fixed_point_l281_281563


namespace solve_equation_l281_281146

noncomputable def diagonal_length (a b : ℚ) : ℚ := Real.sqrt (a^2 + b^2)

theorem solve_equation (a b : ℚ) (h₁ : 0 < a) (h₂ : 0 < b) (h : (∃ x : ℚ, (diagonal_length a b)^x = (a^3 / (diagonal_length a b)^2)^x + (b^3 / (diagonal_length a b)^2)^x)) :
  ∃ x : ℚ, x = 2/3 :=
begin
  use 2/3,
  sorry
end

end solve_equation_l281_281146


namespace lowest_degree_polynomial_l281_281292

-- Define the conditions
def polynomial_conditions (P : ℕ → ℤ) (b : ℤ): Prop :=
  (∃ c, c > b ∧ c ∈ set.range P) ∧ (∃ d, d < b ∧ d ∈ set.range P) ∧ (b ∉ set.range P)

-- The main statement
theorem lowest_degree_polynomial : ∃ P : ℕ → ℤ, polynomial_conditions P 4 ∧ (∀ Q : ℕ → ℤ, polynomial_conditions Q 4 → degree Q >= 4) :=
sorry

end lowest_degree_polynomial_l281_281292


namespace angle_sum_equal_l281_281690

theorem angle_sum_equal 
  (AB AC DE DF : ℝ)
  (h_AB_AC : AB = AC)
  (h_DE_DF : DE = DF)
  (angle_BAC angle_EDF : ℝ)
  (h_angle_BAC : angle_BAC = 40)
  (h_angle_EDF : angle_EDF = 50)
  (angle_DAC angle_ADE : ℝ)
  (h_angle_DAC : angle_DAC = 70)
  (h_angle_ADE : angle_ADE = 65) :
  angle_DAC + angle_ADE = 135 := 
sorry

end angle_sum_equal_l281_281690


namespace probability_3_tails_in_8_flips_unfair_coin_l281_281013

noncomputable def probability_exactly_3_tails_in_8_flips (P_heads P_tails : ℚ) : ℚ :=
  let probability_sequence := (P_tails ^ 3) * (P_heads ^ 5)
  let combinations := nat.choose 8 3
  combinations * probability_sequence

theorem probability_3_tails_in_8_flips_unfair_coin :
  let P_heads := 1 / 4
  let P_tails := 3 / 4
  probability_exactly_3_tails_in_8_flips P_heads P_tails = 189 / 512 :=
by
  sorry

end probability_3_tails_in_8_flips_unfair_coin_l281_281013


namespace inequality1_in_triangle_inequality2_in_triangle_l281_281522

theorem inequality1_in_triangle (a b c s : ℝ)
  (h1 : a + b + c = s) :
  (13 / 27) * s^2 ≤ a^2 + b^2 + c^2 + (4 / s) * a * b * c ∧ 
  a^2 + b^2 + c^2 + (4 / s) * a * b * c < s^2 / 2 :=
sorry

theorem inequality2_in_triangle (a b c s : ℝ)
  (h1 : a + b + c = s) :
  s^2 / 4 < a * b + b * c + c * a - (2 / s) * a * b * c ∧ 
  a * b + b * c + c * a - (2 / s) * a * b * c ≤ (7 / 27) * s^2 :=
sorry

end inequality1_in_triangle_inequality2_in_triangle_l281_281522


namespace articles_in_selling_price_l281_281496

theorem articles_in_selling_price (C : ℝ) (N : ℕ) 
  (h1 : 50 * C = N * (1.25 * C)) 
  (h2 : 0.25 * C = 25 / 100 * C) :
  N = 40 :=
by
  sorry

end articles_in_selling_price_l281_281496


namespace simplify_and_evaluate_expression_l281_281972

theorem simplify_and_evaluate_expression (a b : ℤ) (h_a : a = 2) (h_b : b = -1) : 
  2 * (-a^2 + 2 * a * b) - 3 * (a * b - a^2) = 2 :=
by 
  sorry

end simplify_and_evaluate_expression_l281_281972


namespace sum_of_palindromic_primes_lt_200_l281_281947

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_palindromic (n : ℕ) : Prop :=
  let str_n := n.toString
  str_n = str_n.reverse

def is_palindromic_prime (n : ℕ) : Prop :=
  is_prime n ∧ is_palindromic n ∧ is_prime (n.toString.reverse.toNat)

def palindromic_primes_less_than_200 : List ℕ :=
  [101, 113, 131, 151, 181, 191]

theorem sum_of_palindromic_primes_lt_200 : (palindromic_primes_less_than_200.foldl (· + ·) 0) = 868 := by
  sorry

end sum_of_palindromic_primes_lt_200_l281_281947


namespace freight_train_travel_distance_l281_281707

theorem freight_train_travel_distance
  (one_mile_two_minutes: ∀ (t: ℝ), t = 2 → 1 = t / 2 * 1 )
  (one_hour_sixty_minutes: 1 = 60 / 60)
  (two_hours: ∀ (t: ℝ), t = 2 → t * 60 = 120 ) :
  ∃ d : ℝ, d = 60 :=
by {
  -- Here is the precise statement of the proof.
  have speed := one_mile_two_minutes 2,
  have minutes := two_hours 2,
  exact ⟨ 60, minutes ▸ rfl⟩
}

end freight_train_travel_distance_l281_281707


namespace degrees_for_combined_research_l281_281704

-- Define the conditions as constants.
def microphotonics_percentage : ℝ := 0.10
def home_electronics_percentage : ℝ := 0.24
def food_additives_percentage : ℝ := 0.15
def gmo_percentage : ℝ := 0.29
def industrial_lubricants_percentage : ℝ := 0.08
def nanotechnology_percentage : ℝ := 0.07

noncomputable def remaining_percentage : ℝ :=
  1 - (microphotonics_percentage + home_electronics_percentage + food_additives_percentage +
    gmo_percentage + industrial_lubricants_percentage + nanotechnology_percentage)

noncomputable def total_percentage : ℝ :=
  remaining_percentage + nanotechnology_percentage

noncomputable def degrees_in_circle : ℝ := 360

noncomputable def degrees_representing_combined_research : ℝ :=
  total_percentage * degrees_in_circle

-- State the theorem to prove the correct answer
theorem degrees_for_combined_research : degrees_representing_combined_research = 50.4 :=
by
  -- Proof will go here
  sorry

end degrees_for_combined_research_l281_281704


namespace parallel_lines_angle_sum_l281_281870

theorem parallel_lines_angle_sum 
  (r s : Line) (A B H : Point)
  (parallel : parallel r s)
  (angle_A : angle r A = 40)
  (angle_B : angle r B = 70) :
  angle s H = 110 :=
by
  sorry

end parallel_lines_angle_sum_l281_281870


namespace lowest_degree_is_4_l281_281285

noncomputable def lowest_degree_polynomial (P : ℝ → ℝ) : ℕ :=
  if ∃ b : ℤ, (∀ coeff ∈ (P.coefficients), coeff < (b : ℝ) ∨ coeff > (b : ℝ)) ∧ (¬ ∃ coeff ∈ (P.coefficients), coeff = (b : ℝ))
  then Polynomial.natDegree P
  else 0

theorem lowest_degree_is_4 : ∀ (P : Polynomial ℝ), 
  (∃ b : ℤ, (∀ coeff ∈ P.coefficients, coeff < (b : ℝ) ∨ coeff > (b : ℝ)) ∧ (¬ ∃ coeff ∈ P.coefficients, coeff = (b : ℝ)))
  → lowest_degree_polynomial P = 4 :=
by
  sorry

end lowest_degree_is_4_l281_281285


namespace multiplication_by_strange_method_l281_281401

theorem multiplication_by_strange_method (a b : ℕ) 
  (h1 : a = 97) (h2 : b = 23) : 
  let cols := [(97, 23), (48, 46), (24, 92), (12, 184), (6, 368), (3, 736), (1, 1472)] in
  let filtered := filter (λ (pair : ℕ × ℕ), pair.fst % 2 = 1) cols in
  let result := filtered.foldl (λ acc pair, acc + pair.snd) 0 in
  result = 2231 :=
  by
    sorry

end multiplication_by_strange_method_l281_281401


namespace conditional_probabilities_l281_281967

-- Define the events A and B
def three_dice := ℕ × ℕ × ℕ

def event_A (d : three_dice) : Prop :=
  d.1 ≠ d.2 ∧ d.2 ≠ d.3 ∧ d.3 ≠ d.1

def event_B (d : three_dice) : Prop :=
  d.1 = 6 ∨ d.2 = 6 ∨ d.3 = 6

-- Calculate conditional probabilities
noncomputable def P_A_given_B : ℚ :=
  60 / 91

noncomputable def P_B_given_A : ℚ :=
  1 / 2

-- Statement of the problem
theorem conditional_probabilities :
  ((P_A_given_B = 60 / 91) ∧ (P_B_given_A = 1 / 2)) :=
by
  sorry

end conditional_probabilities_l281_281967


namespace remaining_volume_l281_281328

def side_length := 6    -- Define the side length of the cube
def radius := 3         -- Define the radius of the cylinder
def height := 6         -- Define the height of the cylinder

def volume_cube := side_length ^ 3 -- Volume of the cube
def volume_cylinder := π * (radius ^ 2) * height -- Volume of the cylinder

theorem remaining_volume : volume_cube - volume_cylinder = 216 - 54 * π := 
by 
  -- the proof will be inserted here
  sorry

end remaining_volume_l281_281328


namespace new_room_ratio_l281_281477

theorem new_room_ratio
  (bedroom_size : ℕ)
  (bathroom_size : ℕ)
  (new_room_size : ℕ)
  (h_bedroom : bedroom_size = 309)
  (h_bathroom : bathroom_size = 150)
  (h_new_room : new_room_size = 918) :
  new_room_size / (bedroom_size + bathroom_size) = 2 :=
by
  rw [h_bedroom, h_bathroom, h_new_room]
  simp
  sorry

end new_room_ratio_l281_281477


namespace pure_milk_cost_l281_281965

noncomputable def cost_of_pure_milk_litre (x : ℝ) : Prop :=
  let litres_of_pure_milk : ℝ := 25
  let litres_of_water : ℝ := 5
  let total_litres : ℝ := litres_of_pure_milk + litres_of_water
  let price_per_litre_mixture : ℝ := 3
  let total_revenue : ℝ := total_litres * price_per_litre_mixture
  total_revenue = 90 → litres_of_pure_milk * x = 90

theorem pure_milk_cost : cost_of_pure_milk_litre 3.6 := by
  let litres_of_pure_milk := 25 : ℝ
  let litres_of_water := 5 : ℝ
  let total_litres := litres_of_pure_milk + litres_of_water
  let price_per_litre_mixture := 3 : ℝ
  let total_revenue := total_litres * price_per_litre_mixture
  have : total_revenue = 90 := by norm_num

  sorry

end pure_milk_cost_l281_281965


namespace trigonometric_identity_solution_l281_281312

theorem trigonometric_identity_solution (x : ℝ) (k : ℤ) :
  8.435 * (sin(3 * x) ^ 10) + (cos(3 * x) ^ 10) = 
  4 * ((sin(3 * x) ^ 6 + cos(3 * x) ^ 6) / (4 * cos(6 * x) ^ 2 + sin(6 * x) ^ 2)) →
  x = (π * k) / 6 :=
by
  sorry

end trigonometric_identity_solution_l281_281312


namespace min_pq_value_l281_281557

theorem min_pq_value
  (P Q R S : ℕ)
  (hPQR : P ≠ Q) (hPQS : P ≠ S) (hPPR : P ≠ R) (hQQS : Q ≠ S) (hQRS : Q ≠ R) (hRRS : R ≠ S)
  (hP : P ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
  (hQ : Q ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
  (hR : R ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
  (hS : S ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
  (hDistinct : ∀ x y z w, x ≠ y → y ≠ z → z ≠ w → w ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
  (hInteger : ∃ k : ℕ, P + Q = (k * (R + S))) :
  P + Q = 1 :=
sorry

end min_pq_value_l281_281557


namespace area_covered_by_both_strips_is_correct_l281_281719

-- Definitions of lengths of the strips and areas
def length_total : ℝ := 16
def length_left : ℝ := 9
def length_right : ℝ := 7
def area_left_only : ℝ := 27
def area_right_only : ℝ := 18

noncomputable def width_strip : ℝ := sorry -- The width can be inferred from solution but is not the focus of the proof.

-- Definition of the area covered by both strips
def S : ℝ := 13.5

-- Proof statement
theorem area_covered_by_both_strips_is_correct :
  ∀ w : ℝ,
    length_left * w - S = area_left_only ∧ length_right * w - S = area_right_only →
    S = 13.5 := 
by
  sorry

end area_covered_by_both_strips_is_correct_l281_281719


namespace new_car_fuel_consumption_l281_281712

theorem new_car_fuel_consumption :
  ∃ x : ℝ, (100 / x = 100 / (x + 2) + 4.2) ∧ (x ≈ 5.97) := by
sorry

end new_car_fuel_consumption_l281_281712


namespace rowing_time_ratio_l281_281619

def V_b : ℕ := 57
def V_s : ℕ := 19
def V_up : ℕ := V_b - V_s
def V_down : ℕ := V_b + V_s

theorem rowing_time_ratio :
  ∀ (T_up T_down : ℕ), V_up * T_up = V_down * T_down → T_up = 2 * T_down :=
by
  intros T_up T_down h
  sorry

end rowing_time_ratio_l281_281619


namespace ratio_BQ_CQ_l281_281553

theorem ratio_BQ_CQ (A B C D E F P Q : Point)
  (h_rect : is_rectangle A B C D)
  (h_segment1 : E ∈ segment A B ∧ F ∈ segment A B)
  (h_div : dist A E = dist E F ∧ dist E F = dist F B)
  (h_intersect1 : P = line_intersect (line C E) (line A D))
  (h_intersect2 : Q = line_intersect (line P F) (line B C)) :
  ratio (dist B Q) (dist C Q) = 1 / 3 := by
  sorry

end ratio_BQ_CQ_l281_281553


namespace determine_true_propositions_l281_281807

def p (x y : ℝ) := x > y → -x < -y
def q (x y : ℝ) := (1/x > 1/y) → x < y

theorem determine_true_propositions (x y : ℝ) :
  (p x y ∨ q x y) ∧ (p x y ∧ ¬ q x y) :=
by
  sorry

end determine_true_propositions_l281_281807


namespace tangent_line_at_e_a_range_condition_l281_281467

noncomputable def f (x a : ℝ) : ℝ := 1/2 * (x - a) * log x - log (log x)

theorem tangent_line_at_e (a := Real.exp 1) :
  let e := Real.exp 1 in
  let fx := f x e in
  let dfdx := (1 / 2) * log x + (x - e) / (2 * x) - 1 / (x * log x) in
  ∀ y : ℝ, (x = e) → y = (1 / 2 - 1 / e) * (x - e) + f e e :=
by
  sorry

theorem a_range_condition :
  ∀ a : ℝ, (∀ x : ℝ, x > 0 → f x a ≥ 1 - log 2) ↔ a ≤ 2 :=
by
  sorry

end tangent_line_at_e_a_range_condition_l281_281467


namespace angle_C_is_120_l281_281498

-- Given condition in the problem.
variables {a b c : ℝ} (A B C : ℝ)
variable (h : a^2 + b^2 + a * b = c^2)

-- The theorem we need to prove.
theorem angle_C_is_120 (h : a^2 + b^2 + a * b = c^2) : C = 120 * (Math.pi / 180) :=
by sorry

end angle_C_is_120_l281_281498


namespace largest_divisor_of_three_consecutive_even_integers_is_sixteen_l281_281148

theorem largest_divisor_of_three_consecutive_even_integers_is_sixteen (n : ℕ) :
  ∃ d : ℕ, d = 16 ∧ 16 ∣ ((2 * n) * (2 * n + 2) * (2 * n + 4)) :=
by
  sorry

end largest_divisor_of_three_consecutive_even_integers_is_sixteen_l281_281148


namespace fabric_length_l281_281772

-- Define the width and area as given in the problem
def width : ℝ := 3
def area : ℝ := 24

-- Prove that the length is 8 cm
theorem fabric_length : (area / width) = 8 :=
by
  sorry

end fabric_length_l281_281772


namespace evaluate_expression_l281_281396

theorem evaluate_expression :
  (2^1 - 3 + 5^3 - 2)⁻¹ * 3 = (3 : ℚ) / 122 :=
by
  -- proof goes here
  sorry

end evaluate_expression_l281_281396


namespace lowest_degree_poly_meets_conditions_l281_281296

-- Define a predicate that checks if a polynomial P meets the conditions
def poly_meets_conditions (P : ℚ[X]) (b : ℚ) : Prop :=
  (∀ x, coeff P x ≠ b) ∧ 
  (∃ x y, coeff P x < b ∧ coeff P y > b)

-- Statement of the theorem we want to prove
theorem lowest_degree_poly_meets_conditions : ∀ (b : ℚ), 
  ∃ (P : ℚ[X]), poly_meets_conditions P b ∧ degree P = 4 :=
begin
  sorry
end

end lowest_degree_poly_meets_conditions_l281_281296


namespace eccentricity_of_hyperbola_l281_281021

variable (C : Type) [Center : ℝ] [SymmetryAxes : ℝ → ℝ] (π : ℝ)

theorem eccentricity_of_hyperbola (b a c e : ℝ)
  (h₁ : Center = 0)
  (h₂ : SymmetryAxes = λ x, -x)
  (h₃ : tan (π / 3) = sqrt 3) :
  e = 2 ∨ e = 2 * sqrt 3 / 3 :=
by 
  sorry

end eccentricity_of_hyperbola_l281_281021


namespace area_of_triangle_PQR_is_correct_l281_281341

noncomputable def calculate_area_of_triangle_PQR : ℝ := 
  let side_length := 4
  let altitude := 8
  let WO := (side_length * Real.sqrt 2) / 2
  let center_to_vertex_distance := Real.sqrt (WO^2 + altitude^2)
  let WP := (1 / 4) * WO
  let YQ := (1 / 2) * WO
  let XR := (3 / 4) * WO
  let area := (1 / 2) * (YQ - WP) * (XR - YQ)
  area

theorem area_of_triangle_PQR_is_correct :
  calculate_area_of_triangle_PQR = 2.25 := sorry

end area_of_triangle_PQR_is_correct_l281_281341


namespace positive_integer_expression_iff_l281_281490

theorem positive_integer_expression_iff (p : ℕ) : (0 < p) ∧ (∃ k : ℕ, 0 < k ∧ 4 * p + 35 = k * (3 * p - 8)) ↔ p = 3 :=
by
  sorry

end positive_integer_expression_iff_l281_281490


namespace slope_and_intercept_l281_281782

-- Define the points M (2, -1) and P (-1, 8)
structure Point where
  x : ℝ
  y : ℝ

def M : Point := ⟨2, -1⟩
def P : Point := ⟨-1, 8⟩

-- Define the slope and y-intercept based on the line passing through M and P
def slope (M P : Point) : ℝ :=
  (P.y - M.y) / (P.x - M.x)

def y_intercept (M P : Point) : ℝ :=
  M.y - (slope M P) * M.x

-- The proof problem: prove that the slope is -3 and the y-intercept is 5
theorem slope_and_intercept (M P : Point) (hM : M = (⟨2, -1⟩ : Point)) (hP : P = (⟨-1, 8⟩ : Point)) :
  slope M P = -3 ∧ y_intercept M P = 5 := by
  sorry

end slope_and_intercept_l281_281782


namespace exponential_inequality_l281_281544

theorem exponential_inequality (a x1 x2 : ℝ) (h1 : 1 < a) (h2 : x1 < x2) :
  |a ^ ((1 / 2) * (x1 + x2)) - a ^ x1| < |a ^ x2 - a ^ ((1 / 2) * (x1 + x2))| :=
by
  sorry

end exponential_inequality_l281_281544


namespace parabola_directrix_focus_exists_parallel_line_l281_281060

noncomputable def parabola_C : ℝ → ℝ → Prop := λ x y, y^2 = 2 * p * x
noncomputable def ellipse : ℝ → ℝ → Prop := λ x y, (x^2)/4 + (y^2)/3 = 1
noncomputable def direction_line (b : ℝ) : ℝ → ℝ := λ x, 2 * x + b

theorem parabola_directrix_focus (p : ℝ) (h : 0 < p) :
  (∃ x, ∃ y, ellipse x y ∧ parabola_C x y) ->
  (parabola_C = (λ x y, y^2 = 4 * x) ∧ 
  ∀ x, directrix x = -1 :=
begin
  sorry
end

theorem exists_parallel_line (l : ℝ → ℝ) (h_l : ∀ x, l x = 2 * x - 1) :
  ∃ line : ℝ → ℝ, 
  (∀ x, line x = (λ x, 2 * x + b) ∧
  (b = 1 ∨ b = -1) ∧
  b ≤ 1/2 ∧ 
  ∀ x y, parabola_C x y ∧ l ≠ OP :=
begin
  sorry
end

end parabola_directrix_focus_exists_parallel_line_l281_281060


namespace lowest_degree_polynomial_l281_281295

-- Define the conditions
def polynomial_conditions (P : ℕ → ℤ) (b : ℤ): Prop :=
  (∃ c, c > b ∧ c ∈ set.range P) ∧ (∃ d, d < b ∧ d ∈ set.range P) ∧ (b ∉ set.range P)

-- The main statement
theorem lowest_degree_polynomial : ∃ P : ℕ → ℤ, polynomial_conditions P 4 ∧ (∀ Q : ℕ → ℤ, polynomial_conditions Q 4 → degree Q >= 4) :=
sorry

end lowest_degree_polynomial_l281_281295


namespace hexagon_area_l281_281554

-- Define the triangle and its sides
def Triangle (A B C : ℝ) := (AB BC CA : ℝ)

-- Define the circumcircle and diametrically opposite points
def Circumcircle (ABC : Triangle) (A1 B1 C1 : ℝ) := ∃ (Omega : Set ℝ), 
  Omega.contains A ∧ Omega.contains B ∧ Omega.contains C ∧
  Omega.contains A1 ∧ Omega.contains B1 ∧ Omega.contains C1 ∧
  -- A1, B1, C1 are diametrically opposite points
  ∀ x ∈ Omega, ∃ y ∈ Omega, dist x y = 2 * circumradius ABC

-- Define the convex hexagon
def ConvexHexagonArea (ABC : Triangle) (A1 B1 C1 : ℝ) := 
  2 * (1 / 2) * BC * CA * sin_angle (Angle A) + CA * A1C

-- Prove the area of the convex hexagon
theorem hexagon_area (ABC : Triangle) (A1 B1 C1 : ℝ) (Omega : Circumcircle ABC A1 B1 C1) :
  ConvexHexagonArea ABC A1 B1 C1 = 1155 / 4 := by
  sorry

end hexagon_area_l281_281554


namespace steel_structure_triangle_count_l281_281739

-- Define the structure consisting of 11 steel beams forming triangles.
def steel_structure : Type := sorry
def number_of_triangles (structure : steel_structure) : Nat := sorry

-- Theorem stating the total number of triangles in the steel structure.
theorem steel_structure_triangle_count (s : steel_structure) 
  (eleven_beams : True) : number_of_triangles s = 34 := by
  sorry

end steel_structure_triangle_count_l281_281739


namespace quadratic_solution_1_quadratic_solution_2_quadratic_solution_3_quadratic_solution_4_l281_281205

-- Question 1: Proving solutions for (x-1)^2 = 4
theorem quadratic_solution_1 : (x - 1)^2 = 4 ↔ x = -1 ∨ x = 3 := by sorry

-- Question 2: Proving solutions for x^2 + 3x - 4 = 0
theorem quadratic_solution_2 : (x^2 + 3x - 4 = 0) ↔ (x = -4 ∨ x = 1) := by sorry

-- Question 3: Proving solutions for 4x(2x + 1) = 3(2x + 1)
theorem quadratic_solution_3 : (4 * x * (2 * x + 1) = 3 * (2 * x + 1)) ↔ (x = -1/2 ∨ x = 3/4) := by sorry

-- Question 4: Proving solutions for 2x^2 + 5x - 3 = 0
theorem quadratic_solution_4 : (2 * x^2 + 5 * x - 3 = 0) ↔ (x = 1/2 ∨ x = -3) := by sorry

end quadratic_solution_1_quadratic_solution_2_quadratic_solution_3_quadratic_solution_4_l281_281205


namespace counterexample_disproves_statement_l281_281318

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem counterexample_disproves_statement :
  ∃ n : ℕ, ¬ is_prime n ∧ is_prime (n + 3) :=
  by
    use 8
    -- Proof that 8 is not prime
    -- Proof that 11 (8 + 3) is prime
    sorry

end counterexample_disproves_statement_l281_281318


namespace ellipse_properties_l281_281028

-- Given definitions based on the problem conditions
def is_ellipse (a b c : ℝ) : Prop :=
  a^2 = b^2 + c^2 ∧ 2 * c = 2

def point_on_ellipse (a b x y: ℝ) : Prop :=
  (y^2 / a^2) + (x^2 / b^2) = 1

def slope_product_condition (a b x y: ℝ) : Prop :=
  ∃ P : ℝ × ℝ, P ≠ (-b, 0) ∧ P ≠ (b, 0) ∧ 
  let k_PA := y / (x + b)
  let k_PB := y / (x - b)
  k_PA * k_PB = -4 / 3

def intersect_conditions (E : ℝ → ℝ → Prop) (F : ℝ × ℝ) : Prop :=
  ∀ M N P Q S T: ℝ × ℝ,
  M = (F.1, (F.2 * M.1 + 1)) ∧ N = (F.1, (F.2 * N.1 + 1)) ∧
  P = (-F.1 / F.2, 1) ∧ Q = (F.1 / F.2, 1) ∧
  S = ((M.1 + N.1) / 2, (F.2 * ((M.1 + N.1) / 2) + 1)) ∧
  T = ((P.1 + Q.1) / 2, ((F.2 ^ 2) * ((P.1 + Q.1) / 2)) / (3 + 4 * (F.2 ^ 2))) ∧
  T = (0, 4 / 7)

-- Lean 4 statement
theorem ellipse_properties :
  ∃ a b c : ℝ, is_ellipse a b c ∧ point_on_ellipse a b b 0 ∧ slope_product_condition a b b 0 → (∀ x, point_on_ellipse a b x 0) ∧ 
  (∃ F: ℝ × ℝ, F = (0, 1) → intersect_conditions (point_on_ellipse a b) F := sorry

end ellipse_properties_l281_281028


namespace triangle_base_length_l281_281984

theorem triangle_base_length (A h : ℝ) (hA : A = 615) (hh : h = 10) : 
  let b := (2 * A) / h in 
  b = 123 := 
by 
  -- Definitions based on given conditions
  let b := (2 * A) / h
  -- Use the conditions to replace A and h
  have h1 : A = 615 := hA
  have h2 : h = 10 := hh
  -- Replace and simplify to get the result
  rw [h1, h2]
  -- Simplify to match correct answer
  suffices : (2 * 615) / 10 = 123
  · exact this
      
  norm_num -- Computes the result
  sorry

end triangle_base_length_l281_281984


namespace part1_part2_l281_281469

noncomputable def hyperbola (x y : ℝ) : Prop := x^2 - y^2 / 2 = 1
noncomputable def P : ℝ × ℝ := (0, real.sqrt 3)
noncomputable def F : ℝ × ℝ := (-real.sqrt 3, 0)
noncomputable def M : ℝ × ℝ := (0, -real.sqrt 3 / 4)

theorem part1 (x1 y1 x2 y2 : ℝ) (hA : hyperbola x1 y1) (hB : hyperbola x2 y2)
    (hx1 : y1 = x1 + real.sqrt 3) (hx2 : y2 = x2 + real.sqrt 3) 
    (xy_sum : x1 + x2 = 2 * real.sqrt 3) (xy_prod : x1 * x2 = -5):
    real.sqrt ((x1 + real.sqrt 3)^2 + y1^2) * real.sqrt ((x2 + real.sqrt 3)^2 + y2^2) = 8 := sorry

theorem part2 (x1 y1 x2 y2 : ℝ) (hA : hyperbola x1 y1) (hB : hyperbola x2 y2)
    (hx1 : y1 = x1 + real.sqrt 3) (hx2 : y2 = x2 + real.sqrt 3)
    (xy_sum : x1 + x2 = 2 * real.sqrt 3) (xy_prod : x1 * x2 = -5):
    (x1 * x2 + (y1 + real.sqrt 3 / 4) * (y2 + real.sqrt 3 / 4)) = 35 / 16 := sorry

end part1_part2_l281_281469


namespace mean_median_difference_is_zero_l281_281116

variable {students : List ℕ} 

def scores := [(60, 15), (75, 20), (80, 25), (88, 20), (92, 20)]

noncomputable def mean (l : List (ℕ × ℕ)) :=
  let totalStudents :=  (l.map (λ pair => pair.2)).sum
  let weightedSum := (l.map (λ pair => pair.1 * pair.2)).sum
  weightedSum / totalStudents

noncomputable def median (l : List (ℕ × ℕ)) :=
  let totalStudents := (l.map (λ pair => pair.2)).sum
  let orderedScores := l.bind (λ pair => List.replicate pair.2 pair.1)
  let sortedScores := orderedScores.qsort (λ x y => x < y)
  let mid := totalStudents / 2
  if totalStudents % 2 == 0 then
    (sortedScores.get! mid + sortedScores.get! (mid - 1)) / 2
  else
    sortedScores.get! mid

noncomputable def mean_median_difference (l : List (ℕ × ℕ)) :=
  (mean l) - (median l)

theorem mean_median_difference_is_zero : mean_median_difference scores = 0 := sorry

end mean_median_difference_is_zero_l281_281116


namespace fergus_entry_exit_l281_281329

theorem fergus_entry_exit (n : ℕ) (hn : n = 8) : 
  n * (n - 1) = 56 := 
by
  sorry

end fergus_entry_exit_l281_281329


namespace percentage_calculation_l281_281683

theorem percentage_calculation :
  let total_amt := 1600
  let pct_25 := 0.25 * total_amt
  let pct_5 := 0.05 * pct_25
  pct_5 = 20 := by
sorry

end percentage_calculation_l281_281683


namespace average_boxes_per_day_by_third_day_l281_281234

theorem average_boxes_per_day_by_third_day (day1 day2 day3_part1 day3_part2 : ℕ) :
  day1 = 318 →
  day2 = 312 →
  day3_part1 = 180 →
  day3_part2 = 162 →
  ((day1 + day2 + (day3_part1 + day3_part2)) / 3) = 324 :=
by
  intros h1 h2 h3 h4
  sorry

end average_boxes_per_day_by_third_day_l281_281234


namespace sum_of_B_coordinates_l281_281187

theorem sum_of_B_coordinates 
  (M : ℝ × ℝ)
  (A : ℝ × ℝ)
  (midpoint_condition : M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) 
  (coords_M : M = (2, 9))
  (coords_A : A = (5, 1)) : 
  (B.1 + B.2) = 16 := 
by sorry

end sum_of_B_coordinates_l281_281187


namespace complex_number_in_first_quadrant_l281_281105

theorem complex_number_in_first_quadrant
  (a b : ℝ)
  (h : (a + b * complex.I) + 2 * (a - b * complex.I) * complex.I = 3 + 2 * complex.I) :
  0 < a ∧ 0 < b :=
by {
  sorry
}

end complex_number_in_first_quadrant_l281_281105


namespace count_of_ordered_triples_l281_281334

theorem count_of_ordered_triples :
  { (a, b, c) : ℕ × ℕ × ℕ // 1 ≤ a ∧ a ≤ b ∧ b ≤ c ∧ (1 / a + 1 / b + 1 / c) = 1 / 2 }.card = 10 :=
sorry

end count_of_ordered_triples_l281_281334


namespace DK_KM_CK_KN_ratios_l281_281188

open EuclideanGeometry

variables {A B C D M N K : Point}
variable (parallelogramABCD : parallelogram A B C D)
variable (M_on_AB : M ∈ line_segment A B)
variable (N_on_AD : N ∈ line_segment A D)
variable (AM_ratio_MB : ratio A M M B = 1 / 3)
variable (AN_ratio_ND : ratio A N N D = 3 / 5)
variable (K_on_DM_CN : K ∈ intersection_line D M C N)

theorem DK_KM_CK_KN_ratios 
  (h1 : parallelogramABCD)
  (h2 : M_on_AB)
  (h3 : N_on_AD)
  (h4 : AM_ratio_MB)
  (h5 : AN_ratio_ND)
  (h6 : K_on_DM_CN) :
  ratio D K K M = 6 / 17 ∧ 
  ratio C K K N = 15 / 17 :=
sorry

end DK_KM_CK_KN_ratios_l281_281188


namespace number_of_correct_statements_l281_281354

-- Define the statements as booleans that represent their correctness.

def statement1 : Prop :=
  ∃(A : Angle), ∃(l : Line), (SymmetricFigure A l ∧ SymmetricAxis A l)

def statement2 : Prop :=
  ∃(T : IsoscelesTriangle), (AtLeastOneSymmetricAxis T ∧ AtMostThreeSymmetricAxes T)

def statement3 : Prop :=
  ∀(Δ1 Δ2 : Triangle), (SymmetricAboutLine Δ1 Δ2) → CongruentTriangles Δ1 Δ2

def statement4 : Prop :=
  ∀ (F1 F2 : Figure), (SymmetricAboutLine F1 F2) → PointsOnSymmetricLine F1 F2

-- The final theorem stating the number of correct statements
theorem number_of_correct_statements : 
  ({statement1, statement2, statement3, statement4}.filter (λ stmt, stmt)).length = 2 :=
sorry

end number_of_correct_statements_l281_281354


namespace cord_gcd_l281_281934

theorem cord_gcd (a b : ℕ) (h₁: a = 15) (h₂: b = 12) : Nat.gcd a b = 3 :=
by
  rw [h₁, h₂]
  exact Nat.gcd_comm 15 12
  sorry

end cord_gcd_l281_281934


namespace part1_part2_l281_281543

-- Definitions for conditions
def f (a x : ℝ) : ℝ := 1 + a * x
def g (a x : ℝ) : ℝ := a * x - a / x + 2 * real.log x
def p (a : ℝ) : Prop := ∀ x, 0 < x ∧ x ≤ 2 → f a x ≥ 0
def q (a : ℝ) : Prop := ∃ x : ℝ, x > 0 ∧ g a x = 0

-- Theorem statements
theorem part1 (a : ℝ) : p a → a ≥ -1/2 :=
sorry

theorem part2 (a : ℝ) : (p a ∨ q a) ∧ ¬(p a ∧ q a) → (a ∈ set.Ioo (-1) (-1/2) ∨ a ≥ 0) :=
sorry

end part1_part2_l281_281543


namespace lcm_upto_condition_l281_281912

noncomputable def lcm_upto (n : ℕ) : ℕ :=
  Nat.lcmList (List.range (n + 1))

theorem lcm_upto_condition (n : ℕ) :
  (lcm_upto (n - 1) = lcm_upto n) ↔ (¬ ∃ (p : ℕ) (k : ℕ), Nat.prime p ∧ n = p^k) := by
  sorry

end lcm_upto_condition_l281_281912


namespace problem1_problem2_l281_281025

def arithmetic_sequence (a : ℕ → ℤ) := ∃ d : ℤ, d ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n + d
def geometric_relation (a : ℕ → ℤ) (n1 n2 n3 : ℕ) := a n2 ^ 2 = a n1 * a n3

theorem problem1 (a : ℕ → ℤ) (a₁ a₂ a₅ : ℕ) (d : ℤ):
  (arithmetic_sequence a) →
  a 10 = 19 →
  a₁ = a 1 →
  a₂ = a 2 →
  a₅ = a 5 →
  geometric_relation a 1 2 5 →
  (∀ n : ℕ, a n = 2 * n - 1) :=
by
  sorry

theorem problem2 (a : ℕ → ℤ) (b : ℕ → ℤ) (a₁ a₂ a₅ : ℕ) (d : ℤ) (sum : ℕ → ℤ):
  (arithmetic_sequence a) →
  a 10 = 19 →
  a₁ = a 1 →
  a₂ = a 2 →
  a₅ = a 5 →
  geometric_relation a 1 2 5 →
  (∀ n : ℕ, a n = 2 * n - 1) →
  (∀ n : ℕ, b n = a n * 2 ^ n) →
  (∀ n : ℕ, sum n = Σ k in Finset.range n, b k) →
  (∀ n : ℕ, sum n = (2 * n - 3) * 2 ^ (n + 1) + 6) :=
by
  sorry

end problem1_problem2_l281_281025


namespace lowest_degree_required_l281_281268

noncomputable def smallest_degree_poly (b : ℤ) : ℕ :=
  if h : ∃ P : Polynomial ℝ, (∀ x, (P.eval x ≠ b)) ∧
    (∃ y, (P.eval y > b)) ∧ (∃ z, (P.eval z < b)) 
  then Nat.find h 
  else 0

theorem lowest_degree_required :
  ∃ b : ℤ, smallest_degree_poly b = 4 :=
by
  -- b is some integer that fits the described conditions
  use 0
  sorry

end lowest_degree_required_l281_281268


namespace inverse_negation_proposition_l281_281665

-- Definitions for quadrilateral, pairs of opposite sides, and parallelogram
variable (Q : Type) [quads : quadrilateral Q]
variable (has_equal_opposite_sides : ∀ (q : Q), (q.has_both_pairs_opposite_sides_equal))
variable (is_parallelogram : ∀ (q : Q), (q.is_parallelogram))

-- Proposition statement
theorem inverse_negation_proposition
  (q : Q) : ¬ is_parallelogram q → ¬ has_equal_opposite_sides q := 
sorry

end inverse_negation_proposition_l281_281665


namespace minimum_value_is_correct_l281_281916

noncomputable def minimum_value (x y : ℝ) : ℝ :=
  (x + 1/y) * (x + 1/y - 2024) + (y + 1/x) * (y + 1/x - 2024) + 2024

theorem minimum_value_is_correct (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (minimum_value x y) ≥ -2050208 := 
sorry

end minimum_value_is_correct_l281_281916


namespace combined_loading_time_l281_281732

theorem combined_loading_time 
  (rate1 rate2 : ℝ) 
  (h_rate1 : rate1 = 1 / 5) 
  (h_rate2 : rate2 = 1 / 8) : 
  1 / (rate1 + rate2) = 40 / 13 := 
by 
  have h_combined_rate : rate1 + rate2 = 1 / 5 + 1 / 8 := by rw [h_rate1, h_rate2]
  have h_common_denominator : 1 / 5 + 1 / 8 = 8 / 40 + 5 / 40 := by linarith
  have h_added_rates : 8 / 40 + 5 / 40 = 13 / 40 := by linarith
  have h_reciprocal_rate : 1 / (13 / 40) = 40 / 13 := by exact (one_div_div 13 40).symm
  rw [h_combined_rate, h_common_denominator, h_added_rates, h_reciprocal_rate]
  exact h_reciprocal_rate

end combined_loading_time_l281_281732


namespace period_of_f_is_4_f_of_3_is_0_f_of_0_given_sum_l281_281818

def f : ℝ → ℝ := sorry

-- Conditions
axiom domain_of_f : ∀ x, f x ∈ ℝ
axiom odd_function_f : ∀ x, f (2 * x + 1) = -f (1 - 2 * x)
axiom symmetry_f : ∀ x, f (4 - x) = f x

-- Assertions to prove
theorem period_of_f_is_4 : ∀ x, f (x + 4) = f x := 
sorry

theorem f_of_3_is_0 : f 3 = 0 := 
sorry

theorem f_of_0_given_sum : (∑ k in finset.range 2023, f (k + 1)) = 1 → f 0 = -1 :=
sorry

end period_of_f_is_4_f_of_3_is_0_f_of_0_given_sum_l281_281818


namespace grouping_schemes_count_l281_281629

/-- Number of possible grouping schemes where each group consists
    of either 2 or 3 students and the total number of students is 25 is 4.-/
theorem grouping_schemes_count : ∃ (x y : ℕ), 2 * x + 3 * y = 25 ∧ 
  (x = 11 ∧ y = 1 ∨ x = 8 ∧ y = 3 ∨ x = 5 ∧ y = 5 ∨ x = 2 ∧ y = 7) :=
sorry

end grouping_schemes_count_l281_281629


namespace max_k_sum_2013_in_sequence_l281_281876

theorem max_k_sum_2013_in_sequence : 
  ∃ (k : ℕ), k ≤ 51 ∧ (∃ (s : finset ℕ), (∀ x ∈ s, x ∈ finset.range 52 ∧ x % 2 = 1) ∧ s.card = k ∧ s.sum id = 2013) ∧ ∀ t, (t ≤ 51 ∧ (∃ u : finset ℕ, (∀ y ∈ u, y ∈ finset.range 52 ∧ y % 2 = 1) ∧ u.card = t ∧ u.sum id = 2013)) → t ≤ k :=
begin
  use 43,
  split,
  { exact dec_trivial, }, -- 43 ≤ 51
  split,
  { use finset.filter (λ x, (x : ℕ) ∈ (finset.range 52).filter (λ n, n % 2 = 1)) (finset.range 87),
    split,
    { intros x hx,
      rw finset.mem_filter at hx,
      exact ⟨hx.2, by norm_num at hx; exact hx⟩, },
    split,
    { exact dec_trivial, }, -- cardinality of filtered set is 43
    { exact sorry, }, -- sum of selected numbers
  },
  { intros t ht,
    cases ht with ht_le ht_ex,
    cases ht_ex with u hu,
    cases hu with hu_mem hu_sum,
    have t_le_43 : t ≤ 43,
    { sorry, }, -- proof that t cannot be greater than 43
    exact t_le_43,
  },
end

end max_k_sum_2013_in_sequence_l281_281876


namespace marble_ratio_l281_281755

-- Definitions based on conditions
def dan_marbles : ℕ := 5
def mary_marbles : ℕ := 10

-- Statement of the theorem to prove the ratio is 2:1
theorem marble_ratio : mary_marbles / dan_marbles = 2 := by
  sorry

end marble_ratio_l281_281755


namespace intersection_obtuse_triangle_l281_281574

/-- Representation of a regular tetrahedron in terms of its vertices --/
structure RegularTetrahedron :=
  (P P1 P2 T : Point)
  [instance : Vertex P Tetrahedron]
  [instance : Vertex P1 Tetrahedron]
  [instance : Vertex P2 Tetrahedron]
  (edge1 : LineSegment P T)
  (edge2 : LineSegment T P1)
  (edge3 : LineSegment T P2)

/-- Two given points Q and R such that Q is on the line TP1, R is on the line TP2 --/
noncomputable def IntersectionTriangle (T P P1 P2 Q R : Point) (lambda mu : Real) 
  (h1 : Q = T + lambda • (P1 - T)) (h2 : R = T + mu • (P2 - T)) (hlambda : 0 < lambda) (hmu : 0 < mu) : Prop :=
  ∃(Triangle : Triangle), 
  Triangle.angles.exists λ θ, obtuse θ ∧ θ < 120∘

/-- Statement: The intersection of a plane and a regular tetrahedron 
forms an obtuse-angled triangle, and the obtuse angle in any such 
triangle is always smaller than 120 degrees --/
theorem intersection_obtuse_triangle (T P P1 P2 Q R : Point) (lambda mu : Real)
  (h1 : Q = T + lambda • (P1 - T)) (h2 : R = T + mu • (P2 - T)) (hlambda : 0 < lambda) (hmu : 0 < mu) :
  IntersectionTriangle T P P1 P2 Q R lambda mu h1 h2 hlambda hmu :=
sorry

end intersection_obtuse_triangle_l281_281574


namespace how_many_peaches_l281_281883

-- Define the main problem statement and conditions.
theorem how_many_peaches (A P J_A J_P : ℕ) (h_person_apples: A = 16) (h_person_peaches: P = A + 1) (h_jake_apples: J_A = A + 8) (h_jake_peaches: J_P = P - 6) : P = 17 :=
by
  -- Since the proof is not required, we use sorry to skip it.
  sorry

end how_many_peaches_l281_281883


namespace extremum_point_is_three_l281_281858

noncomputable def f (x : ℝ) : ℝ := (x - 2) / Real.exp x

theorem extremum_point_is_three {x₀ : ℝ} (h : ∀ x, f x₀ ≤ f x) : x₀ = 3 :=
by
  -- proof goes here
  sorry

end extremum_point_is_three_l281_281858


namespace abs_S_eq_96768_l281_281166

-- Define the polynomial and the condition for b
def polynomial (b : ℤ) : polynomial ℤ :=
  polynomial.X^2 + polynomial.C b * polynomial.X + polynomial.C (2016 * b)

-- Define the sum S of all b for which the polynomial is factorable over the integers
noncomputable def S : ℤ :=
  ∑ b in {b : ℤ | polynomial b.factorable.integer_factorable}, b

-- Statement of the theorem
theorem abs_S_eq_96768 : abs S = 96768 :=
sorry

end abs_S_eq_96768_l281_281166


namespace blocks_differ_in_exactly_two_ways_l281_281699

theorem blocks_differ_in_exactly_two_ways :
  let materials := 2 in
  let sizes := 3 in
  let colors := 5 in
  let shapes := 4 in
  (materials - 1) * (sizes - 1) * colors * shapes +
  materials * (sizes - 1) * (colors - 1) * shapes + 
  materials * sizes * (colors - 1) * (shapes - 1) +
  materials * sizes * colors * (shapes - 1) = 28 :=
begin
  -- Since (1 + x) * (1 + 2x) * (1 + 4x) * (1 + 3x) = 1 + 10x + 28x^2 + 31x^3 + 12x^4,
  -- the coefficient of x^2 is 28.
  sorry
end

end blocks_differ_in_exactly_two_ways_l281_281699


namespace symmetry_of_function_l281_281826

noncomputable def f (ω x : ℝ) : ℝ := sin (2 * ω * x + π / 3)

theorem symmetry_of_function (ω : ℝ) (h_pos : ω > 0) (h_period : ∀ x, f ω (x + π) = f ω x) :
  ∀ x, f ω (2 * (13 * π / 12) - x) = f ω x :=
sorry

end symmetry_of_function_l281_281826


namespace find_P_coordinates_l281_281804

-- Definitions
structure Point where
  x : ℝ
  y : ℝ

def A : Point := ⟨4, -3⟩
def B : Point := ⟨2, -1⟩

def line_l (P : Point) : ℝ := 4 * P.x + 3 * P.y - 2
def distance_to_line (P : Point) : ℝ := (|line_l P|) / (Real.sqrt (4^2 + 3^2))

def midpoint (P1 P2 : Point) : Point := ⟨(P1.x + P2.x) / 2, (P1.y + P2.y) / 2⟩
def distance (P1 P2 : Point) : ℝ := Real.sqrt ((P1.x - P2.x)^2 + (P1.y - P2.y)^2)

def is_on_perpendicular_bisector (P : Point) : Prop :=
  let M := midpoint A B
  line_l P = 0 ∧ P.x - P.y - 5 = 0

-- Main Statement
theorem find_P_coordinates :
  ∃ P : Point, 
    (distance P A = distance P B) ∧ 
    (distance_to_line P = 2) ∧ 
    ((P = ⟨1, -4⟩) ∨ (P = ⟨27/7, -8/7⟩)) :=
  by sorry

end find_P_coordinates_l281_281804


namespace smallest_k_l281_281012

theorem smallest_k (M : Finset ℕ) (H : ∀ (a b c d : ℕ), a ∈ M → b ∈ M → c ∈ M → d ∈ M → a ≠ b → b ≠ c → c ≠ d → d ≠ a → 20 ∣ (a - b + c - d)) :
  ∃ k, k = 7 ∧ ∀ (M' : Finset ℕ), M'.card = k → ∀ (a b c d : ℕ), a ∈ M' → b ∈ M' → c ∈ M' → d ∈ M' → a ≠ b → b ≠ c → c ≠ d → d ≠ a → 20 ∣ (a - b + c - d) :=
sorry

end smallest_k_l281_281012


namespace maddie_total_payment_l281_281929

def makeup_cost := 3 * 15
def lipstick_cost := 4 * 2.50
def hair_color_cost := 3 * 4

def total_cost := makeup_cost + lipstick_cost + hair_color_cost

theorem maddie_total_payment : total_cost = 67 := by
  sorry

end maddie_total_payment_l281_281929


namespace g_zero_not_in_range_l281_281154

def g (x : ℝ) : ℤ :=
  if x > -3 then ⌈2 / (x + 3)⌉
  else ⌊2 / (x + 3)⌋

theorem g_zero_not_in_range :
  ¬ ∃ x : ℝ, x ≠ -3 ∧ g x = 0 := 
sorry

end g_zero_not_in_range_l281_281154


namespace verification_l281_281906

variables {a b : ℝ} (h_pos_a : 0 < a) (h_pos_b : 0 < b)

def arithmetic_mean (a b : ℝ) := (a + b) / 2
def geometric_mean (a b : ℝ) := real.sqrt (a * b)
def lehmer_mean (p a b : ℝ) := (a^p + b^p) / (a^(p-1) + b^(p-1))

theorem verification (h_pos_a : 0 < a) (h_pos_b : 0 < b) :
  lehmer_mean 0.5 a b ≤ arithmetic_mean a b ∧
  lehmer_mean 2 a b ≥ lehmer_mean 1 a b :=
by
  sorry

end verification_l281_281906


namespace min_expr_value_l281_281910

noncomputable def expr (a b c : ℝ) : ℝ :=
  (a - 2)^2 + (b / a - 1)^2 + (c / b - 1)^2 + (5 / c - 1)^2 

theorem min_expr_value :
  ∃ a b c : ℝ, 2 ≤ a ∧ a ≤ b ∧ b ≤ c ∧ c ≤ 5 ∧ expr a b c = 4 * (Real.root 4 5 - 5 / 4)^2 :=
by
  sorry

end min_expr_value_l281_281910


namespace parametric_to_ordinary_l281_281998

theorem parametric_to_ordinary (θ : ℝ) : 
  let x := cos θ ^ 2
  let y := 2 * sin θ ^ 2
  x ∈ set.Icc 0 1 → 2 * x + y = 2 :=
by
  intros h
  sorry

end parametric_to_ordinary_l281_281998


namespace total_combinations_meals_l281_281980

-- Define the total number of menu items
def menu_items : ℕ := 12

-- Define the function for computing the number of combinations of meals ordered by three people
def combinations_of_meals (n : ℕ) : ℕ := n * n * n

-- Theorem stating the total number of different combinations of meals is 1728
theorem total_combinations_meals : combinations_of_meals menu_items = 1728 :=
by
  -- Placeholder for actual proof
  sorry

end total_combinations_meals_l281_281980


namespace measure_of_angle_C_sin_A_plus_sin_B_l281_281133

-- Problem 1
theorem measure_of_angle_C (a b c : ℝ) (h1 : a^2 + b^2 - c^2 = 8) (h2 : (1/2) * a * b * Real.sin C = 2 * Real.sqrt 3) : C = Real.pi / 3 := 
sorry

-- Problem 2
theorem sin_A_plus_sin_B (a b c A B C : ℝ) (h1 : a^2 + b^2 - c^2 = 8) (h2 : (1/2) * a * b * Real.sin C = 2 * Real.sqrt 3) (h3 : c = 2 * Real.sqrt 3) : Real.sin A + Real.sin B = 3 / 2 := 
sorry

end measure_of_angle_C_sin_A_plus_sin_B_l281_281133


namespace cricket_team_new_winning_percentage_l281_281500

theorem cricket_team_new_winning_percentage (initial_matches_played : ℕ) (initial_win_percentage : ℚ) (additional_wins : ℕ) :
  initial_matches_played = 120 ∧ initial_win_percentage = 0.24 ∧ additional_wins = 70 →
  let initial_wins := round (initial_win_percentage * initial_matches_played : ℚ),
      total_wins := initial_wins + additional_wins,
      total_matches_played := initial_matches_played + additional_wins,
      new_win_percentage := (total_wins.toRat / total_matches_played) * 100 in
  abs (new_win_percentage - 52.11) < 0.01 :=
by
  intros h
  let initial_matches_played := 120
  let initial_win_percentage := (24 : ℚ)/100
  
  let additional_wins := 70
  let initial_wins: ℕ := 29
  let total_wins: ℚ := 99
  let total_matches_played := 190
  let new_wins := total_wins / total_matches_played * 100
  have h: abs (new_wins - 52.11) < 0.01 := by sorry
  exact h

end cricket_team_new_winning_percentage_l281_281500


namespace travel_time_proportion_l281_281692

theorem travel_time_proportion (D V : ℝ) (hV_pos : V > 0) :
  let Time1 := D / (16 * V)
  let Time2 := 3 * D / (4 * V)
  let TimeTotal := Time1 + Time2
  (Time1 / TimeTotal) = 1 / 13 :=
by
  sorry

end travel_time_proportion_l281_281692


namespace find_x_sets_l281_281408

theorem find_x_sets (n : ℕ) (x : Fin (n + 1) → ℝ) 
  (h1 : x 0 = x n)
  (h2 : ∀ k : Fin n, 2 * log 2 (x k) * log 2 (x (k + 1)) - (log 2 (x k)) ^ 2 = 9) :
  (∀ k : Fin (n + 1), x k = 8) ∨ (∀ k : Fin (n + 1), x k = 1 / 8) :=
begin
  sorry
end

end find_x_sets_l281_281408


namespace smallest_ellipse_area_l281_281415

theorem smallest_ellipse_area :
  ∃ (Γ : set (ℝ × ℝ)), 
  (∀ p ∈ [(2,0), (0,3), (0,7), (6,0)], p ∈ Γ) ∧
  (area_of_ellipse Γ = (56 * real.pi * real.sqrt 3) / 9) :=
sorry

end smallest_ellipse_area_l281_281415


namespace solution_set_of_inequality_range_of_k_l281_281681

-- Define the solution set for the inequality problem
theorem solution_set_of_inequality (x : ℝ) :
  (9 / (x + 4) ≤ 2) ↔ (x < -4 ∨ x ≥ 1 / 2) :=
by
  sorry

-- Define the range of k for the quadratic inequality problem
theorem range_of_k (k : ℝ) :
  (∀ x : ℝ, x^2 - 2 * x + k^2 - 1 > 0) ↔ (k < -real.sqrt 2 ∨ k > real.sqrt 2) :=
by
  sorry

end solution_set_of_inequality_range_of_k_l281_281681


namespace min_value_of_expression_l281_281430

theorem min_value_of_expression
  (a b : ℝ)
  (ha : a > 0)
  (hb : b > 0)
  (hlines : (∀ x y : ℝ, x + (a-4) * y + 1 = 0) ∧ (∀ x y : ℝ, 2 * b * x + y - 2 = 0) ∧ (∀ x y : ℝ, (x + (a-4) * y + 1 = 0) ∧ (2 * b * x + y - 2 = 0) → -1 * 1 / (a-4) * -2 * b = 1)) :
  ∃ (min_val : ℝ), min_val = (9/5) ∧ min_val = (a + 2)/(a + 1) + 1/(2 * b) :=
by
  sorry

end min_value_of_expression_l281_281430


namespace triangle_BQ_QF_ratio_l281_281534

theorem triangle_BQ_QF_ratio
  {A B C E F Q : Type}
  [Inhabited A] [Inhabited B] [Inhabited C]
  (on_segment_BC: B ∈ [B, C]) (on_segment_BE_EC: BE = 2 * EC)
  (F_midpoint_AC: F ∈ [A, C] ∧ F = midpoint A C)
  (BF_intersects_AE_at_Q: ∃ Q, Q ∈ [AE] ∧ Q = (BF ∩ AE)) :
  BQ / QF = 4 :=
sorry

end triangle_BQ_QF_ratio_l281_281534


namespace percentage_difference_l281_281632

theorem percentage_difference (X : ℝ) :
  let second_number := 0.63 * X in
  let fourth_number := 1.15 * X in
  (fourth_number - second_number) / second_number * 100 = 82.54 := 
sorry

end percentage_difference_l281_281632


namespace poly_comp_eq_l281_281918

variable {K : Type*} [Field K]

theorem poly_comp_eq {Q1 Q2 : Polynomial K} (P : Polynomial K) (hP : ¬P.degree = 0) :
  Q1.comp P = Q2.comp P → Q1 = Q2 :=
by
  intro h
  sorry

end poly_comp_eq_l281_281918


namespace sum_of_ages_l281_281239

theorem sum_of_ages (a b : ℕ) :
  let c1 := a
  let c2 := a + 2
  let c3 := a + 4
  let c4 := a + 6
  let coach1 := b
  let coach2 := b + 2
  c1^2 + c2^2 + c3^2 + c4^2 + coach1^2 + coach2^2 = 2796 →
  c1 + c2 + c3 + c4 + coach1 + coach2 = 106 :=
by
  intro h
  sorry

end sum_of_ages_l281_281239


namespace circumcenter_proof_l281_281443

variable {P D E O : Type}

def equilateral_triangle (A B C : Type) : Prop :=
∀ (a b c : ℝ), a = b ∧ b = c ∧ c = a

def on_side (P : Type) (B C : Type) : Prop :=
-- Some definition indicating that P is on side BC
sorry

def parallel_lines (line1 line2 : Type) : Prop :=
-- Some definition indicating that line1 is parallel to line2
sorry

def circumcircle (A E D : Type) (O : Type) : Prop :=
-- Some definition indicating that O is on the circumcircle of triangle AED
sorry

def bisector_intersects (angle_bisector : Type) (O : Type) : Prop :=
-- Some definition indicating the intersection of the angle bisector of ∠A with O
sorry

def is_circumcenter (O : Type) (A B C : Type) : Prop :=
-- Some definition indicating that O is the circumcenter of triangle ABC
sorry

theorem circumcenter_proof (A B C P D E O : Type)
  (h1 : equilateral_triangle A B C)
  (h2 : on_side P B C)
  (h3 : parallel_lines (line_through P parallel_to AB) AB)
  (h4 : parallel_lines (line_through P parallel_to AC) AC)
  (h5 : circumcircle A E D O)
  (h6 : bisector_intersects A O) :
  is_circumcenter O A B C :=
sorry

end circumcenter_proof_l281_281443


namespace complex_eq_triangle_l281_281374

theorem complex_eq_triangle (p q r : ℂ) (h1 : ∀ (a b : ℂ) (h : a ≠ b), (∃ (θ : ℂ) (H : Im θ = real.pi / 3), (b - a) = (r - a) * θ))
(h2 : complex.abs (p + q + r) = 48) : complex.abs (p * q + p * r + q * r) = 768 :=
sorry

end complex_eq_triangle_l281_281374


namespace distinct_numbers_at_least_n_div_4_l281_281572

-- Triangle definition
def is_triangle_of_natural_numbers (triangle : List (List ℕ)) : Prop :=
  ∃ n, ∀ (i j : ℕ), i < n → j ≤ i → j < (triangle.length) → i < (triangle.length) ∧ triangle.length = triangle.head.length ∧ ∀ x ∈ triangle, x.nat? 

-- Theorem statement
theorem distinct_numbers_at_least_n_div_4 (triangle : List (List ℕ)) (n : ℕ) 
  (h_triangle : is_triangle_of_natural_numbers triangle) : 
  (triangle.flatten.erase_dup.length) ≥ (n / 4) :=
sorry

end distinct_numbers_at_least_n_div_4_l281_281572


namespace circles_tangent_internally_l281_281229

theorem circles_tangent_internally
  (C1_eqn : ∀ x y : ℝ, x^2 + y^2 + 2*x + 4*y + 1 = 0)
  (C2_eqn : ∀ x y : ℝ, x^2 + y^2 - 4*x + 4*y - 17 = 0) :
  let C1_center := (-1, -2),
      C2_center := (2, -2),
      R1 := 2,
      R2 := 5,
      distance := dist C1_center C2_center in
  distance = abs (R2 - R1) :=
sorry

end circles_tangent_internally_l281_281229


namespace notebooks_last_days_l281_281892

namespace ProofProblem

noncomputable def total_pages (notebooks : ℕ) (pages_per_notebook : ℕ) : ℕ :=
  notebooks * pages_per_notebook

noncomputable def pages_per_week (weekdays : ℕ) (pages_weekday : ℕ) (weekends : ℕ) (pages_weekendday : ℕ) : ℕ :=
  (weekdays * pages_weekday) + (weekends * pages_weekendday)

noncomputable def weeks_of_use (total_pages : ℕ) (pages_per_week : ℕ) : ℚ :=
  total_pages / pages_per_week

noncomputable def remaining_days (weeks_of_use : ℚ) : ℚ :=
  (weeks_of_use - weeks_of_use.floor) * 7

noncomputable def pages_in_remaining_days (remaining_days : ℚ) (pages_weekendday : ℕ) : ℚ :=
  remaining_days * pages_weekendday

theorem notebooks_last_days (notebooks : ℕ) (pages_per_notebook : ℕ) (weekdays : ℕ) (pages_weekday : ℕ) (weekends : ℕ) (pages_weekendday : ℕ) :
    let total := total_pages notebooks pages_per_notebook
    total = 200 →
    let per_week := pages_per_week weekdays pages_weekday weekends pages_weekendday
    per_week = 32 →
    let weeks := total / per_week
    weeks = 6.25 →
    let remaining := remaining_days weeks
    remaining = 1.75 →
    let used_in_remaining := remaining * (pages_weekendday : ℚ)
    used_in_remaining.floor = 10 →
    7 * weeks.floor.to_nat + (used_in_remaining / pages_weekendday).floor.to_nat + 1 = 43 :=
begin
  intros,
  sorry
end

end ProofProblem

end notebooks_last_days_l281_281892


namespace cube_volume_given_face_area_l281_281587

theorem cube_volume_given_face_area (s : ℝ) (h : s^2 = 36) : s^3 = 216 := by
  sorry

end cube_volume_given_face_area_l281_281587


namespace find_difference_of_squares_l281_281086

variable (x y : ℝ)
variable (h1 : (x + y) ^ 2 = 81)
variable (h2 : x * y = 18)

theorem find_difference_of_squares : (x - y) ^ 2 = 9 := by
  sorry

end find_difference_of_squares_l281_281086


namespace number_called_2009th_position_l281_281861

theorem number_called_2009th_position :
  let sequence := [1, 2, 3, 4, 3, 2]
  ∃ n, n = 2009 → sequence[(2009 % 6) - 1] = 3 := 
by
  -- let sequence := [1, 2, 3, 4, 3, 2]
  -- 2009 % 6 = 5
  -- sequence[4] = 3
  sorry

end number_called_2009th_position_l281_281861


namespace smaller_fraction_is_l281_281617

theorem smaller_fraction_is
  (x y : ℝ)
  (h₁ : x + y = 7 / 8)
  (h₂ : x * y = 1 / 12) :
  min x y = (7 - Real.sqrt 17) / 16 :=
sorry

end smaller_fraction_is_l281_281617


namespace cat_roaming_area_l281_281571

theorem cat_roaming_area (r_tank r_rope : ℝ) (h_tank : r_tank = 20) (h_rope : r_rope = 10) : 
  let
    r_total := r_tank + r_rope,
    area_total := Real.pi * r_total^2,
    area_tank := Real.pi * r_tank^2
  in
  (area_total - area_tank = 500 * Real.pi) :=
by
  intro r_tank r_rope h_tank h_rope,
  let r_total := r_tank + r_rope,
  let area_total := Real.pi * r_total^2,
  let area_tank := Real.pi * r_tank^2,
  have h_goal : area_total - area_tank = 500 * Real.pi,
  sorry

end cat_roaming_area_l281_281571


namespace children_count_after_addition_l281_281564

theorem children_count_after_addition :
  ∀ (total_guests men guests children_added : ℕ),
    total_guests = 80 →
    men = 40 →
    guests = (men + men / 2) →
    children_added = 10 →
    total_guests - guests + children_added = 30 :=
by
  intros total_guests men guests children_added h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end children_count_after_addition_l281_281564


namespace problem_l281_281091

variable (x y : ℝ)

theorem problem (h1 : (x + y)^2 = 81) (h2 : x * y = 18) : (x - y)^2 = 9 :=
by
  sorry

end problem_l281_281091


namespace a_b_product_l281_281062

noncomputable def a_b_prod : ℤ :=
  let f (x : ℝ) (a b : ℤ) := (b + 2) * x^a
  let conditions := (f 2 3 (-1) = 8)
  3 * (-1)

theorem a_b_product (a b : ℤ) (h : ∀ x, f x a b = (b + 2) * x ^ a) : a * b = -3 :=
by
  have a_val : a = 3 := sorry
  have b_val : b = -1 := sorry
  show a * b = -3, from by
    rw [a_val, b_val]
    exact rfl

end a_b_product_l281_281062


namespace bookA_net_change_bookB_net_change_bookC_net_change_l281_281689

-- Define the price adjustments for Book A
def bookA_initial_price := 100.0
def bookA_after_first_adjustment := bookA_initial_price * (1 - 0.5)
def bookA_after_second_adjustment := bookA_after_first_adjustment * (1 + 0.6)
def bookA_final_price := bookA_after_second_adjustment * (1 + 0.1)
def bookA_net_percentage_change := (bookA_final_price - bookA_initial_price) / bookA_initial_price * 100

-- Define the price adjustments for Book B
def bookB_initial_price := 100.0
def bookB_after_first_adjustment := bookB_initial_price * (1 + 0.2)
def bookB_after_second_adjustment := bookB_after_first_adjustment * (1 - 0.3)
def bookB_final_price := bookB_after_second_adjustment * (1 + 0.25)
def bookB_net_percentage_change := (bookB_final_price - bookB_initial_price) / bookB_initial_price * 100

-- Define the price adjustments for Book C
def bookC_initial_price := 100.0
def bookC_after_first_adjustment := bookC_initial_price * (1 + 0.4)
def bookC_after_second_adjustment := bookC_after_first_adjustment * (1 - 0.1)
def bookC_final_price := bookC_after_second_adjustment * (1 - 0.05)
def bookC_net_percentage_change := (bookC_final_price - bookC_initial_price) / bookC_initial_price * 100

-- Statements to prove the net percentage changes
theorem bookA_net_change : bookA_net_percentage_change = -12 := by
  sorry

theorem bookB_net_change : bookB_net_percentage_change = 5 := by
  sorry

theorem bookC_net_change : bookC_net_percentage_change = 19.7 := by
  sorry

end bookA_net_change_bookB_net_change_bookC_net_change_l281_281689


namespace infinite_series_sum_l281_281383

theorem infinite_series_sum : 
  ∑' k : ℕ, (8 ^ k) / ((4 ^ k - 3 ^ k) * (4 ^ (k + 1) - 3 ^ (k + 1))) = 2 :=
by 
  sorry

end infinite_series_sum_l281_281383


namespace matrix_product_scaled_transpose_l281_281384

noncomputable def A : Matrix (Fin 2) (Fin 3) ℤ := ![
  ![2, 0, 1],
  ![3, -1, 0]
]

noncomputable def B : Matrix (Fin 3) (Fin 2) ℤ := ![
  ![1, -1],
  ![2, 0],
  ![0, 3]
]

noncomputable def k : ℤ := 2

theorem matrix_product_scaled_transpose :
  (A ⬝ (k • Bᵀ)) = ![
    ![4, 8, 0],
    ![8, 12, -6]
  ] :=
sorry

end matrix_product_scaled_transpose_l281_281384


namespace find_special_numbers_l281_281944

theorem find_special_numbers : 
  ∃ (n1 n2 : ℕ), 
    (100 ≤ n1 ∧ n1 < 1000 ∧ 
    (∀ (d1 d2 d3 : ℕ), n1 = 100 * d1 + 10 * d2 + d3 → d1 ≠ d2 ∧ d2 ≠ d3 ∧ d1 ≠ d3 ∧ d1 < d2 ∧ d2 < d3 ∧ 
    (String.head (String.words (to_string n1)).getD "" = String.head (String.words (to_string n1)).getD ""))) ∧
    (100 ≤ n2 ∧ n2 < 1000 ∧
    (∀ (d : ℕ), n2 = d * 111 → 
    let name_words := String.words (to_string n2) in
    (String.head name_words.head = "O") ∧ 
    (String.head name_words.tail.headD "" = "H") ∧ 
    (String.head (name_words.drop 2).headD "" = "E"))) ∧
    n1 = 147 ∧ n2 = 111 := by {
  sorry
}

end find_special_numbers_l281_281944


namespace max_value_of_function_l281_281222

/--
Statement:
Given a function y = a^(2 * x) + 2 * a^x - 1 where a > 0 and a ≠ 1,
if the function has a maximum value of 14 on the interval [-1, 1],
then the value of a must be either 3 or 1/3.
-/
theorem max_value_of_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) 
  (h3 : ∀ x : ℝ, x ∈ Icc (-1:ℝ) (1:ℝ) → a^(2*x) + 2*a^x - 1 ≤ 14) : 
  a = 3 ∨ a = 1/3 :=
sorry

end max_value_of_function_l281_281222


namespace max_touching_points_l281_281624

theorem max_touching_points (E1 E2 E3 E4 : Set ℝ^2) (line : Set ℝ^2) 
  (h1 : ∀ i ∈ {E1, E2, E3, E4}, IsEllipse i)
  (h2 : ∀ i j ∈ {E1, E2, E3, E4}, i ≠ j → i ∩ j = ∅)
  (h3 : ∀ i ∈ {E1, E2, E3, E4}, line ∩ i ≠ ∅) :
  ( ∀ i ∈ {E1, E2, E3, E4}, #(line ∩ i) ≤ 2 )  → #(line ∩ (E1 ∪ E2 ∪ E3 ∪ E4)) ≤ 8 :=
by
  intros h4
  sorry

end max_touching_points_l281_281624


namespace interval_for_x_l281_281319

noncomputable def satisfies_inequality (x : ℝ) : Prop :=
  8.66 * Real.sin (4 * x) + Real.cos (4 * x) * Real.cot (2 * x) > 1

theorem interval_for_x (x : ℝ) (n : ℤ) :
  satisfies_inequality x ∧ x ≠ (n : ℝ) * (Real.pi / 2) →
  ∃ (k : ℤ), x ∈ Ioo ((k : ℝ) * (Real.pi / 2)) ((4 * (k : ℝ) + 1) * (Real.pi / 8)) :=
sorry

end interval_for_x_l281_281319


namespace Joan_balloons_l281_281889

variable (J : ℕ) -- Joan's blue balloons

theorem Joan_balloons (h : J + 41 = 81) : J = 40 :=
by
  sorry

end Joan_balloons_l281_281889


namespace simplify_fraction_l281_281580

theorem simplify_fraction : (5^5 + 5^3) / (5^4 - 5^2 + 1) = 16250 / 601 :=
by sorry

end simplify_fraction_l281_281580


namespace perpendicular_bisector_eq_l281_281067

structure Point where
  x : ℝ
  y : ℝ

def A : Point := {x := 0, y := 1}
def B : Point := {x := 4, y := 3}

def is_perpendicular_bisector (L : ℝ → ℝ → Prop) (A B : Point) : Prop :=
  let M := {x := (A.x + B.x) / 2, y := (A.y + B.y) / 2}
  let slope_AB := (B.y - A.y) / (B.x - A.x)
  let slope_perpendicular := -1 / slope_AB
  ∀ x y : ℝ, L x y ↔ (y - M.y = slope_perpendicular * (x - M.x))

theorem perpendicular_bisector_eq :
  is_perpendicular_bisector (λ x y, 2 * x + y - 6 = 0) A B :=
sorry

end perpendicular_bisector_eq_l281_281067


namespace eventB_is_not_random_l281_281306

def eventA := "The sun rises in the east and it rains in the west"
def eventB := "It's not cold when it snows but cold when it melts"
def eventC := "It rains continuously during the Qingming festival"
def eventD := "It's sunny every day when the plums turn yellow"

def is_random_event (event : String) : Prop :=
  event = eventA ∨ event = eventC ∨ event = eventD

theorem eventB_is_not_random : ¬ is_random_event eventB :=
by
  unfold is_random_event
  sorry

end eventB_is_not_random_l281_281306


namespace hexagon_count_l281_281147

theorem hexagon_count (n : ℕ) (hn : 0 < n) :
  let hexagons_within_divided_hexagon := (n^2 + n)^2 / 4 in
  hexagons_within_divided_hexagon = (n^2 + n)^2 / 4 :=
sorry

end hexagon_count_l281_281147


namespace semi_focal_distance_of_hyperbola_l281_281823

def hyperbola_equation (a b: ℝ) := ∀ x y, (x^2 / a^2) - (y^2 / b^2) = 1

noncomputable def semi_focal_distance (a b: ℝ) : ℝ := real.sqrt(a^2 + b^2)

theorem semi_focal_distance_of_hyperbola :
  semi_focal_distance 20 5 = 5 :=
by
  sorry

end semi_focal_distance_of_hyperbola_l281_281823


namespace find_x_y_l281_281100

theorem find_x_y (x y : ℤ) (h1 : 3 * x - 482 = 2 * y) (h2 : 7 * x + 517 = 5 * y) :
  x = 3444 ∧ y = 4925 :=
by
  sorry

end find_x_y_l281_281100


namespace difference_triangle_areas_l281_281125
-- Import required libraries and definitions

-- Define the problem with appropriate types and statements
theorem difference_triangle_areas 
  (A B C D F : Type)
  (angle_FAB_right : is_right_angle (angle FAB))
  (angle_ABC_right : is_right_angle (angle ABC))
  (length_AB : Distance AB = 5)
  (length_BC : Distance BC = 3)
  (length_AF : Distance AF = 7)
  (intersection_AC_BF_D : intersects AC BF D) :
  area (triangle ADF) - area (triangle BDC) = 10 :=
sorry

end difference_triangle_areas_l281_281125


namespace number_of_digits_l281_281393

noncomputable def num_digits_base10 (x : ℝ) : ℕ :=
  ⌊Real.log10 x + 1⌋₊

theorem number_of_digits (x : ℝ) (y : ℝ) (log10_3 : ℝ) (log10_7 : ℝ) :
  x = 3 ∧ y = 7 ∧ log10_3 = 0.477 ∧ log10_7 = 0.845 → 
  num_digits_base10 (x^15 * y^24) = 28 :=
by
  sorry

end number_of_digits_l281_281393


namespace solve_for_n_l281_281845

theorem solve_for_n (n : ℕ) (h : sqrt (8 + n) = 9) : n = 73 :=
sorry

end solve_for_n_l281_281845


namespace highest_wave_height_l281_281365

-- Definitions of surfboard length and shortest wave conditions
def surfboard_length : ℕ := 7
def shortest_wave_height (H : ℕ) : ℕ := H + 4

-- Lean statement to be proved
theorem highest_wave_height (H : ℕ) (condition1 : H + 4 = surfboard_length + 3) : 
  4 * H + 2 = 26 :=
sorry

end highest_wave_height_l281_281365


namespace largest_hexagon_angle_l281_281391

theorem largest_hexagon_angle (x : ℝ) 
  (h : 2 * x + 3 * x + 3 * x + 4 * x + 4 * x + 5 * x = 720) : 
  5 * (720 / 21) = 1200 / 7 := 
by
  have hx : x = 720 / 21 := by sorry
  have largest_angle : 5 * x = 5 * (720 / 21) := by
    rw [hx]
  exact calc
    5 * (720 / 21) = 5 * (720 / 21) : by sorry
               ... = 1200 / 7      : by sorry

end largest_hexagon_angle_l281_281391


namespace trigonometric_relationship_l281_281812

noncomputable def a : ℝ := Real.sin (46 * Real.pi / 180)
noncomputable def b : ℝ := Real.cos (46 * Real.pi / 180)
noncomputable def c : ℝ := Real.tan (46 * Real.pi / 180)

theorem trigonometric_relationship : c > a ∧ a > b :=
by
  -- This is the statement part; the proof will be handled here
  sorry

end trigonometric_relationship_l281_281812


namespace system_of_equations_solutions_l281_281066

theorem system_of_equations_solutions (x y a b : ℝ) 
  (h1 : 2 * x + y = b) 
  (h2 : x - b * y = a) 
  (hx : x = 1)
  (hy : y = 0) : a - b = -1 :=
by 
  sorry

end system_of_equations_solutions_l281_281066


namespace identify_fake_bag_identify_fake_bags_l281_281349

-- Part (a) conditions
def bags_count : Nat := 10
def coins_per_bag : Nat := 10
def coin_weight_genuine : ℕ := 10
def coin_weight_fake : ℕ := 1

-- Part (a) statement
theorem identify_fake_bag (D W : ℕ) (bags_count : ℕ) (coins_per_bag : ℕ) 
    (coin_weight_genuine coin_weight_fake : ℕ) :
  bags_count = 10 → coins_per_bag = 10 → coin_weight_genuine = 10 → coin_weight_fake = 1 →
  let total_coins := (List.range bags_count).sum + bags_count in
  let expected_weight := total_coins * coin_weight_genuine in
  let discrepancy := expected_weight - W in
  ∃ (bag_index : ℕ), bag_index < bags_count ∧ discrepancy = bag_index + 1 := sorry

-- Part (b) conditions 
def powers_of_two (n : ℕ) : ℕ := 2^n

-- Part (b) statement
theorem identify_fake_bags (D W : ℕ) (bags_count : ℕ) (coin_weight_genuine coin_weight_fake : ℕ) :
  bags_count = 10 → coin_weight_genuine = 10 → coin_weight_fake = 1 →
  let total_coins := ((List.range bags_count).map powers_of_two).sum in
  let expected_weight := total_coins * coin_weight_genuine in
  let discrepancy := expected_weight - W in
  ∃ (fake_bags : List ℕ), (∀ b ∈ fake_bags, b < bags_count) ∧ 
  (fake_bags.map powers_of_two).sum = discrepancy := sorry

end identify_fake_bag_identify_fake_bags_l281_281349


namespace min_sum_abs_l281_281006

theorem min_sum_abs (n : ℕ) (h : n > 0) : 
  (∃ n_min : ℕ, (n_min > 0) ∧ (∀ n : ℕ, n > 0 → ∑ k in finset.range (100 + 1), |n - k| ≥ ∑ k in finset.range (100 + 1), |n_min - k|)) ∧ 
  (∑ k in finset.range (100 + 1), |50 - k| = 2500) :=
by {
  -- Proof goes here
  sorry
}

end min_sum_abs_l281_281006


namespace gerald_bars_l281_281789

theorem gerald_bars (G : ℕ) 
  (H1 : ∀ G, ∀ teacher_bars : ℕ, teacher_bars = 2 * G → total_bars = G + teacher_bars) 
  (H2 : ∀ total_bars : ℕ, total_squares = total_bars * 8 → total_squares_needed = 24 * 7) 
  (H3 : ∀ total_squares : ℕ, total_squares_needed = 24 * 7) 
  : G = 7 :=
by
  sorry

end gerald_bars_l281_281789


namespace arithmetic_operations_correct_l281_281186

theorem arithmetic_operations_correct :
  (3 + (3 / 3) = (77 / 7) - 7) :=
by
  sorry

end arithmetic_operations_correct_l281_281186


namespace false_proposition_among_given_statements_l281_281353

theorem false_proposition_among_given_statements : 
  ¬ (∀ (R : Type) [ring R] (a b c d : R), 
    (a ≠ b ∧ c ≠ d ∧ 
     (∃ (x y : R), x = y ∧ 
        (∃ (p q r s : R), 
            x + p = y + q ∧ p ≠ s
        )
     )
    )
  ) := 
begin
  sorry,
end

end false_proposition_among_given_statements_l281_281353


namespace units_digit_G_2000_l281_281562

-- Define the sequence G
def G (n : ℕ) : ℕ := 2 ^ (2 ^ n) + 5 ^ (5 ^ n)

-- The main goal is to show that the units digit of G 2000 is 1
theorem units_digit_G_2000 : (G 2000) % 10 = 1 :=
by
  sorry

end units_digit_G_2000_l281_281562


namespace shaded_area_calculation_l281_281872

def area_of_circle (r : ℝ) : ℝ := π * r^2

noncomputable def total_shaded_area : ℝ := 
  let R := 9 in -- The radius of the largest circle
  let large_shaded_area := area_of_circle R / 2 in
  let r := R / 2 in -- The radius of the smaller circle
  let small_shaded_area := area_of_circle r / 2 in
  large_shaded_area + small_shaded_area

theorem shaded_area_calculation :
  total_shaded_area = 50.625 * π :=
by
  sorry

end shaded_area_calculation_l281_281872


namespace angle_AEB_plus_angle_ADB_eq_45_l281_281896

noncomputable def isosceles_right_triangle (A B C : Point) : Prop :=
  (∠ABC = 90) ∧ (AB = AC)

def midpoint (D A C : Point) : Prop :=
  D = midpoint_of_line_segment A C

def E_on_AC (E A C : Point) : Prop :=
  E ∈ segment_line A C

def EC_eq_2AE (E A C : Point) : Prop :=
  distance E C = 2 * distance A E

theorem angle_AEB_plus_angle_ADB_eq_45 
  (A B C D E : Point)
  (h_triangle : isosceles_right_triangle A B C)
  (h_midpoint : midpoint D A C)
  (h_E_on_AC : E_on_AC E A C)
  (h_EC_eq_2AE : EC_eq_2AE E A C)
  : ∠AEB + ∠ADB = 45 := by 
  sorry

end angle_AEB_plus_angle_ADB_eq_45_l281_281896


namespace degrees_for_combined_research_l281_281703

-- Define the conditions as constants.
def microphotonics_percentage : ℝ := 0.10
def home_electronics_percentage : ℝ := 0.24
def food_additives_percentage : ℝ := 0.15
def gmo_percentage : ℝ := 0.29
def industrial_lubricants_percentage : ℝ := 0.08
def nanotechnology_percentage : ℝ := 0.07

noncomputable def remaining_percentage : ℝ :=
  1 - (microphotonics_percentage + home_electronics_percentage + food_additives_percentage +
    gmo_percentage + industrial_lubricants_percentage + nanotechnology_percentage)

noncomputable def total_percentage : ℝ :=
  remaining_percentage + nanotechnology_percentage

noncomputable def degrees_in_circle : ℝ := 360

noncomputable def degrees_representing_combined_research : ℝ :=
  total_percentage * degrees_in_circle

-- State the theorem to prove the correct answer
theorem degrees_for_combined_research : degrees_representing_combined_research = 50.4 :=
by
  -- Proof will go here
  sorry

end degrees_for_combined_research_l281_281703


namespace find_f_at_6_l281_281448

-- Definitions of the quadratic functions
variable (f g : ℝ → ℝ)
variable (A B C D : ℝ)

-- Conditions:
-- 1. f(x) and g(x) are quadratic functions with leading coefficient 1
-- Assume the general form of a quadratic function with leading coefficient 1
-- Given: g(6) = 35
def g_quadratic (x : ℝ) : Bool := g x = A * x^2 + B * x + C
def f_quadratic (x : ℝ) : Bool := f x = D * x^2 + B * x + C

axiom leading_coeff_g : A = 1
axiom leading_coeff_f : D = 1
axiom g_at_6 : g 6 = 35

-- Given: f(-1) / g(-1) = 21 / 20 and f(1) / g(1) = 21 / 20
axiom ratio_neg1 : f (-1) / g (-1) = 21 / 20
axiom ratio_pos1 : f 1 / g 1 = 21 / 20

-- The goal is to find f(6).
theorem find_f_at_6 : f 6 = 35 := by
  sorry

end find_f_at_6_l281_281448


namespace cube_volume_increase_l281_281107

variable (a : ℝ)

theorem cube_volume_increase (a : ℝ) : (2 * a)^3 - a^3 = 7 * a^3 :=
by
  sorry

end cube_volume_increase_l281_281107


namespace sum_palindromic_primes_eq_1383_l281_281952

def is_prime (n : ℕ) : Prop := sorry -- Definition of prime numbers can be imported from Mathlib

def is_palindromic (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

def is_palindromic_prime (n : ℕ) : Prop :=
  is_prime n ∧ is_palindromic n ∧ n < 200 ∧ 100 ≤ n

def sum_palindromic_primes : ℕ :=
  (List.range 1000).filter is_palindromic_prime |> List.sum

theorem sum_palindromic_primes_eq_1383 : sum_palindromic_primes = 1383 :=
  sorry

end sum_palindromic_primes_eq_1383_l281_281952


namespace binary_product_l281_281745

theorem binary_product (a b : ℕ) (h1 : a = nat.of_digits 2 [1, 0, 1, 0, 1, 1])
                                   (h2 : b = nat.of_digits 2 [1, 1, 0, 1]) :
  nat.of_digits 2 (nat.digits 2 (a * b)) = [1, 0, 0, 0, 1, 0, 1, 1, 1, 1] :=
by
  -- Here, we will later fill in the proof steps.
  sorry

end binary_product_l281_281745


namespace number_of_players_tournament_l281_281206

theorem number_of_players_tournament (n : ℕ) : 
  (2 * n * (n - 1) = 272) → n = 17 :=
by
  sorry

end number_of_players_tournament_l281_281206


namespace infinite_sum_l281_281377

theorem infinite_sum:
  ∑ k in (filter (λ n, n ≥ 1) (range (n + 1))) (8^k / ((4^k - 3^k) * (4^(k + 1) - 3^(k + 1)))) = 2 :=
sorry

end infinite_sum_l281_281377


namespace probability_of_picking_matching_pair_l281_281771

open Nat

def num_blue_socks : Nat := 12
def num_green_socks : Nat := 10
def num_red_socks : Nat := 9
def total_socks : Nat := num_blue_socks + num_green_socks + num_red_socks
def total_ways_to_pick_two : Nat := binomial total_socks 2
def ways_to_pick_two_blue : Nat := binomial num_blue_socks 2
def ways_to_pick_two_green : Nat := binomial num_green_socks 2
def ways_to_pick_two_red : Nat := binomial num_red_socks 2
def total_ways_to_pick_matching : Nat := ways_to_pick_two_blue + ways_to_pick_two_green + ways_to_pick_two_red
def probability_matching : ℚ := total_ways_to_pick_matching / total_ways_to_pick_two

theorem probability_of_picking_matching_pair :
  probability_matching = 147 / 465 :=
  by
  sorry

end probability_of_picking_matching_pair_l281_281771


namespace possible_values_of_k_l281_281510

-- Define vectors and operations
noncomputable def i : ℝ × ℝ := (1, 0)
noncomputable def j : ℝ × ℝ := (0, 1)

noncomputable def AB : ℝ × ℝ := (2, 1)
noncomputable def AC (k : ℝ) : ℝ × ℝ := (3, k)

-- Define dot product
noncomputable def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

noncomputable def BC (k : ℝ) : ℝ × ℝ :=
  (1, k - 1)

-- Define the theorem
theorem possible_values_of_k (k : ℝ) :
  (dot_product AB AC k = 0 ∨ dot_product AB (BC k) = 0 ∨ dot_product (BC k) (AC k) = 0) ↔
  (k = -6 ∨ k = -1) :=
sorry

end possible_values_of_k_l281_281510


namespace perimeter_WXY_is_correct_l281_281339

-- Given conditions
structure Prism :=
(height : ℝ)
(base_side_length : ℝ)
(midpoint_PQ : ℝ)
(midpoint_QR : ℝ)
(midpoint_RT : ℝ)

def PQRSTU : Prism :=
{ height := 20,
  base_side_length := 10,
  midpoint_PQ := 5,
  midpoint_QR := 5,
  midpoint_RT := 10 }

-- Lean statement to prove
def perimeter_WXY (prism : Prism) : ℝ :=
  prism.midpoint_PQ * 2 * Real.sqrt 5 + prism.base_side_length

theorem perimeter_WXY_is_correct :
  perimeter_WXY PQRSTU = 5 * (2 * Real.sqrt 5 + 1) :=
by
  sorry

end perimeter_WXY_is_correct_l281_281339


namespace min_value_x_plus_y_l281_281032

theorem min_value_x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = x * y) : x + y ≥ 4 :=
by
  sorry

end min_value_x_plus_y_l281_281032


namespace shiela_tradition_days_l281_281243

theorem shiela_tradition_days (total_socks : ℕ) (extra_socks : ℕ) 
  (cinnamon_balls_regular : ℕ) (cinnamon_balls_extra : ℕ) (total_cinnamon_balls : ℕ)
  (h1 : total_socks = 9)
  (h2 : extra_socks = 3)
  (h3 : cinnamon_balls_regular = 2)
  (h4 : cinnamon_balls_extra = 3)
  (h5 : total_cinnamon_balls = 75) :
  (total_cinnamon_balls / (6 * cinnamon_balls_regular + 3 * cinnamon_balls_extra) = 3) :=
by
  have h_regular_socks : ℕ := total_socks - extra_socks
  have h_cinnamon_balls_per_day : ℕ := h_regular_socks * cinnamon_balls_regular + extra_socks * cinnamon_balls_extra
  exact calc
    (total_cinnamon_balls / h_cinnamon_balls_per_day) = (total_cinnamon_balls / (6 * cinnamon_balls_regular + 3 * cinnamon_balls_extra)) : by rfl
    ... = (75 / 21) : by rw [h1, h2, h3, h4, h5]
    ... = 3 : by norm_num

end shiela_tradition_days_l281_281243


namespace reflection_ray_equation_l281_281717

theorem reflection_ray_equation (x y : ℝ) : (y = 2 * x + 1) → (∃ (x' y' : ℝ), y' = x ∧ y = 2 * x' + 1 ∧ x - 2 * y - 1 = 0) :=
by
  intro h
  sorry

end reflection_ray_equation_l281_281717


namespace simplify_expression_l281_281581

theorem simplify_expression :
  2 * Real.sqrt 8 + 3 * Real.sqrt 32 = 16 * Real.sqrt 2 :=
by
  have h1 : Real.sqrt 8 = 2 * Real.sqrt 2 := sorry
  have h2 : Real.sqrt 32 = 4 * Real.sqrt 2 := sorry
  calc
    2 * Real.sqrt 8 + 3 * Real.sqrt 32
      = 2 * (2 * Real.sqrt 2) + 3 * (4 * Real.sqrt 2) : by rw [h1, h2]
  ... = 4 * Real.sqrt 2 + 12 * Real.sqrt 2 : by ring
  ... = 16 * Real.sqrt 2 : by ring

end simplify_expression_l281_281581


namespace right_triangle_area_l281_281503

-- Define the lengths of the legs of the triangle
def leg1 : ℝ := 45
def leg2 : ℝ := 48

-- Define the formula for the area of a right triangle
def area_of_right_triangle (a b : ℝ) : ℝ := (1 / 2) * a * b

-- The theorem we want to prove:
theorem right_triangle_area : area_of_right_triangle leg1 leg2 = 1080 := by
  sorry

end right_triangle_area_l281_281503


namespace geometric_sequence_property_l281_281128

open Classical

noncomputable def geometric_sequence (a : ℕ → ℝ) (q : ℝ) := ∀ n, a (n + 1) = a n * q

theorem geometric_sequence_property :
  ∃ (a : ℕ → ℝ) (q : ℝ), q < 0 ∧ geometric_sequence a q ∧
    a 1 = 1 - a 0 ∧ a 3 = 4 - a 2 ∧ a 3 + a 4 = -8 :=
by
  sorry

end geometric_sequence_property_l281_281128


namespace disjoint_quadrilaterals_l281_281795

/-- Given 2016 points in the plane with no three points collinear, 
    prove that it is possible to construct 504 pairwise disjoint quadrilaterals 
    using these points. -/
theorem disjoint_quadrilaterals (points : Fin 2016 → ℝ × ℝ)
  (h_no_three_collinear : ∀ (i j k : Fin 2016), 
                          i ≠ j → j ≠ k → i ≠ k → 
                          ¬ (collinear (points i) (points j) (points k))) :
  ∃ quadrilaterals : Fin 504 → Finset (Fin 2016), 
    (∀ q : Fin 504, quadrilaterals q).card = 4 ∧
    pairwise (Disjoint on quadrilaterals) := sorry

end disjoint_quadrilaterals_l281_281795


namespace inverse_of_original_reverse_of_inverse_l281_281601

def original_function (x : ℝ) : ℝ := 3^(x + 1)

def inverse_function (y : ℝ) : ℝ := logBase 3 y - 1

theorem inverse_of_original :
  ∀ y : ℝ, 1 ≤ y ∧ y < 3 → (original_function (inverse_function y)) = y :=
by 
  intros y hy
  sorry

theorem reverse_of_inverse :
  ∀ x : ℝ, -1 ≤ x ∧ x < 0 → (inverse_function (original_function x)) = x :=
by 
  intros x hx
  sorry

end inverse_of_original_reverse_of_inverse_l281_281601


namespace discount_difference_l281_281342

theorem discount_difference (initial_discount : ℝ) (additional_discount : ℝ) (claimed_discount : ℝ) :
  initial_discount = 0.4 → additional_discount = 0.1 → claimed_discount = 0.5 →
  let true_discount := 1 - (1 - initial_discount) * (1 - additional_discount)
  in claimed_discount - true_discount = 0.04 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  let true_discount := 1 - (1 - 0.4) * (1 - 0.1)
  have true_discount_val : true_discount = 0.46 := by sorry
  have claimed_minus_true : 0.5 - true_discount = 0.5 - 0.46 := by rw true_discount_val
  show 0.5 - 0.46 = 0.04 from claimed_minus_true
  sorry

end discount_difference_l281_281342


namespace lowest_degree_required_l281_281273

noncomputable def smallest_degree_poly (b : ℤ) : ℕ :=
  if h : ∃ P : Polynomial ℝ, (∀ x, (P.eval x ≠ b)) ∧
    (∃ y, (P.eval y > b)) ∧ (∃ z, (P.eval z < b)) 
  then Nat.find h 
  else 0

theorem lowest_degree_required :
  ∃ b : ℤ, smallest_degree_poly b = 4 :=
by
  -- b is some integer that fits the described conditions
  use 0
  sorry

end lowest_degree_required_l281_281273


namespace function_pass_through_point_l281_281993

theorem function_pass_through_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  ∃ (x y : ℝ), y = a^(x-2) - 1 ∧ (x, y) = (2, 0) := 
by
  use 2
  use 0
  sorry

end function_pass_through_point_l281_281993


namespace find_magnitude_sub_vector_find_lambda_l281_281069

noncomputable def vec_a : ℝ × ℝ := (1, 0)
noncomputable def vec_b (m : ℝ) : ℝ × ℝ := (m, 1)
noncomputable def angle : ℝ := Real.pi / 4

theorem find_magnitude_sub_vector (m : ℝ) (h : cos angle = m / Real.sqrt (m^2 + 1)) :
  ‖(1,0) - 2 • (m,1)‖ = Real.sqrt 5 := sorry

theorem find_lambda (m : ℝ) (h : cos angle = m / Real.sqrt (m^2 + 1)) (λ : ℝ) 
  (h1 : (1, 0) + λ • (m, 1) ⬝ (m, 1) = 0) : λ = -1 / 2 := sorry

end find_magnitude_sub_vector_find_lambda_l281_281069


namespace product_of_pieces_with_three_cuts_l281_281338

theorem product_of_pieces_with_three_cuts (n : Nat) (h : n ∈ {4, 5, 6, 7}) :
  ∃ (product : Nat), product = 4 * 5 * 6 * 7 :=
by
  use 840
  sorry

end product_of_pieces_with_three_cuts_l281_281338


namespace number_base_addition_l281_281537

theorem number_base_addition (A B : ℕ) (h1: A = 2 * B) (h2: 2 * B^2 + 2 * B + 4 + 10 * B + 5 = (3 * B)^2 + 3 * (3 * B) + 4) : 
  A + B = 9 := 
by 
  sorry

end number_base_addition_l281_281537


namespace cannot_place_pieces_l281_281524

inductive Piece
| Rook
| Knight
| Bishop

def attacks : Piece → Piece → Prop
| Piece.Rook, Piece.Knight => True
| Piece.Knight, Piece.Bishop => True
| Piece.Bishop, Piece.Rook => True
| _, _ => False

def chessboard := Fin 8 × Fin 8 → Piece

theorem cannot_place_pieces (b : chessboard) :
  ∀ x y : Fin 8, (b x y = Piece.Rook → ∃ i : Fin 8, i ≠ y ∧ b x i = Piece.Knight) ∧
                 (b x y = Piece.Knight → ∃ i j : Fin 8, i ≠ x ∧ j ≠ y ∧ b i j = Piece.Bishop) ∧
                 (b x y = Piece.Bishop → ∃ i : Fin 8, i ≠ x ∧ b i y = Piece.Rook) →
  False := 
sorry

end cannot_place_pieces_l281_281524


namespace volume_double_in_m3_l281_281840

def length_original := 75 -- cm
def width_original := 80 -- cm
def height_original := 120 -- cm

def volume_original (length : ℕ) (width : ℕ) (height : ℕ) := length * width * height -- cm³

def length_double := 2 * length_original
def width_double := 2 * width_original
def height_double := 2 * height_original

def volume_double (length : ℕ) (width : ℕ) (height : ℕ) := length * width * height -- cm³

def cm3_to_m3 (volume_cm3 : ℕ) := volume_cm3 / 1000000 -- 1 m³ = 1,000,000 cm³ 

theorem volume_double_in_m3 :
  cm3_to_m3 (volume_double length_double width_double height_double) = 5.76 :=
by
  sorry

end volume_double_in_m3_l281_281840


namespace probability_roots_real_l281_281718

noncomputable def P (b : ℝ) (x : ℝ) : ℝ := x^4 + 3 * b * x^3 + (3 * b - 5) * x^2 + (-6 * b + 4) * x - 3

theorem probability_roots_real :
  let interval := set.Icc (-25 : ℝ) 22,
      condition (b : ℝ) := b ∈ interval,
      answer := 137 / 141 in
  (∀ b : ℝ, condition b → (∀ x : ℝ, P b x = 0 → is_real x)) ↔ (probability_event condition = answer) :=
sorry

end probability_roots_real_l281_281718


namespace fibonacci_divisible_by_m_spaced_eq_l281_281963

def fibonacci : ℕ → ℕ
| 0       := 0
| 1       := 1
| (n + 2) := fibonacci (n + 1) + fibonacci n

theorem fibonacci_divisible_by_m_spaced_eq (m : ℕ) (h : m > 0) :
  ∃ d : ℕ, ∀ k : ℕ, ∃ n : ℕ, n = k * d ∧ fibonacci n % m = 0 :=
sorry

end fibonacci_divisible_by_m_spaced_eq_l281_281963


namespace intersection_eq_l281_281495

open Set Real

variable A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}
variable B : Set ℝ := {x | x < 1}

theorem intersection_eq : A ∩ B = {x : ℝ | -2 ≤ x ∧ x < 1} :=
by
  sorry

end intersection_eq_l281_281495


namespace combined_angle_basic_astrophysics_nanotech_l281_281701

theorem combined_angle_basic_astrophysics_nanotech :
  let percentage_microphotonics : ℝ := 10
  let percentage_home_electronics : ℝ := 24
  let percentage_food_additives : ℝ := 15
  let percentage_gmo : ℝ := 29
  let percentage_industrial_lubricants : ℝ := 8
  let percentage_nanotechnology : ℝ := 7
  let total_percentage : ℝ := 100
  let percentage_basic_astrophysics := total_percentage - 
                                       (percentage_microphotonics + 
                                        percentage_home_electronics + 
                                        percentage_food_additives + 
                                        percentage_gmo + 
                                        percentage_industrial_lubricants + 
                                        percentage_nanotechnology)
  let combined_percentage := percentage_basic_astrophysics + 
                             percentage_nanotechnology
  let degrees_per_percentage : ℝ := 360 / total_percentage
  let combined_degrees := combined_percentage * degrees_per_percentage
  combined_degrees = 50.4 := by
begin
  sorry
end

end combined_angle_basic_astrophysics_nanotech_l281_281701


namespace equilateral_triangle_excircle_identity_l281_281200

theorem equilateral_triangle_excircle_identity 
  (a : ℝ) (rho : ℝ) (rho' : ℝ) 
  (h1 : rho = (sqrt 3 / 6) * a) 
  (h2 : rho' = (sqrt 3 / 2) * a) : 
  rho' * (rho + rho') = a^2 := 
by 
  sorry

end equilateral_triangle_excircle_identity_l281_281200


namespace blocks_before_jess_turn_l281_281139

def blocks_at_start : Nat := 54
def players : Nat := 5
def rounds : Nat := 5
def father_removes_block_in_6th_round : Nat := 1

theorem blocks_before_jess_turn :
    (blocks_at_start - (players * rounds + father_removes_block_in_6th_round)) = 28 :=
by 
    sorry

end blocks_before_jess_turn_l281_281139


namespace total_heads_l281_281330

def total_legs : ℕ := 45
def num_cats : ℕ := 7
def legs_per_cat : ℕ := 4
def captain_legs : ℕ := 1
def legs_humans := total_legs - (num_cats * legs_per_cat) - captain_legs
def num_humans := legs_humans / 2

theorem total_heads : (num_cats + (num_humans + 1)) = 15 := by
  sorry

end total_heads_l281_281330


namespace square_diff_l281_281078

theorem square_diff (x y : ℝ) (h1 : (x + y)^2 = 81) (h2 : x * y = 18) : (x - y)^2 = 9 :=
by 
  sorry

end square_diff_l281_281078


namespace more_soccer_balls_than_basketballs_l281_281241

theorem more_soccer_balls_than_basketballs :
  let soccer_boxes := 8
  let basketball_boxes := 5
  let balls_per_box := 12
  soccer_boxes * balls_per_box - basketball_boxes * balls_per_box = 36 := by
  sorry

end more_soccer_balls_than_basketballs_l281_281241


namespace will_clothing_loads_l281_281308

theorem will_clothing_loads :
  ∃ (pieces_per_load : ℕ), 
  (let total_clothing := 59 in
   let loaded_clothing := 32 in
   let remaining_clothing := total_clothing - loaded_clothing in
   let num_small_loads := 9 in
   pieces_per_load = remaining_clothing / num_small_loads) :=
begin
  use 3,
  have total_clothing := 59,
  have loaded_clothing := 32,
  have remaining_clothing := total_clothing - loaded_clothing,
  have num_small_loads := 9,
  have pieces_per_load := remaining_clothing / num_small_loads,
  simp only [total_clothing, loaded_clothing, remaining_clothing, num_small_loads],
  norm_num,
end

end will_clothing_loads_l281_281308


namespace missing_digit_divisible_by_9_l281_281610

theorem missing_digit_divisible_by_9 (x : ℕ) (h : 0 ≤ x ∧ x < 10) : (3 + 5 + 1 + 9 + 2 + x) % 9 = 0 ↔ x = 7 :=
by
  sorry

end missing_digit_divisible_by_9_l281_281610


namespace squared_difference_l281_281082

variable {x y : ℝ}

theorem squared_difference (h1 : (x + y)^2 = 81) (h2 : x * y = 18) :
  (x - y)^2 = 9 :=
by
  sorry

end squared_difference_l281_281082


namespace arithmetic_mean_is_one_l281_281215

theorem arithmetic_mean_is_one (x a : ℝ) (hx : x ≠ 0) : 
  (1 / 2) * ((x + a) / x + (x - a) / x) = 1 :=
by
  sorry

end arithmetic_mean_is_one_l281_281215


namespace BD_value_l281_281869

-- Definitions of the given conditions
variables {AB BC CD DA BD : ℕ}
def quadrilateral (AB BC CD DA BD : ℕ) : Prop :=
  AB = 6 ∧ BC = 12 ∧ CD = 6 ∧ DA = 8 ∧ (2 < BD) ∧ (BD < 18) ∧ (BD > 6) ∧ (BD < 14)

-- The statement we need to prove
theorem BD_value (h : quadrilateral AB BC CD DA BD) : BD = 12 :=
by {
  obtain ⟨hAB, hBC, hCD, hDA, hBD1, hBD2, hBD3, hBD4⟩ := h,
  -- Given all inequalities, we want to find if BD = 12 is a valid solution
  sorry -- Proof goes here
}

end BD_value_l281_281869


namespace range_of_m_l281_281468

noncomputable def f (x : ℝ) := Real.log (x^2 + 1)

noncomputable def g (x m : ℝ) := (1 / 2)^x - m

theorem range_of_m (m : ℝ) :
  (∀ x1 ∈ Set.Icc (0:ℝ) 3, ∃ x2 ∈ Set.Icc (1:ℝ) 2, f x1 ≤ g x2 m) ↔ m ≤ -1/2 :=
by
  sorry

end range_of_m_l281_281468


namespace reservoir_capacity_l281_281337

theorem reservoir_capacity (x : ℝ) (h1 : (3 / 8) * x - (1 / 4) * x = 100) : x = 800 :=
by
  sorry

end reservoir_capacity_l281_281337


namespace new_car_fuel_consumption_l281_281715

-- Definitions of the conditions
variable (x : ℝ)
variable (h1 : ∀ x, 100 / x = 100 / (x + 2) + 4.2)
variable (h2 : x + 2 > 0)

-- The expected outcome
theorem new_car_fuel_consumption : x ≈ 5.97 :=
by
  -- We use sorry as a placeholder to skip the proof
  sorry

end new_car_fuel_consumption_l281_281715


namespace triangle_inequality_l281_281191

theorem triangle_inequality (a b c S : ℝ) (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) (area_eq : S = Real.sqrt ((a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c))) :
    (a * b + b * c + c * a) / (4 * S) ≥ Real.cot (Real.pi / 6) :=
  sorry

end triangle_inequality_l281_281191


namespace new_car_fuel_consumption_l281_281713

theorem new_car_fuel_consumption :
  ∃ x : ℝ, (100 / x = 100 / (x + 2) + 4.2) ∧ (x ≈ 5.97) := by
sorry

end new_car_fuel_consumption_l281_281713


namespace part1_monotonicity_part2_range_k_l281_281825

noncomputable def f (x : ℝ) : ℝ := Real.log x - (x + 1)

theorem part1_monotonicity (a : ℝ) (h : a = 1) :
  (∀ x ∈ set.Ioo 0 1, f x < f (x + 1)) ∧ (∀ x ∈ set.Ioo 1 (Real.top), f (x + 1) < f x) :=
by
  sorry

theorem part2_range_k (x0 : ℝ) (h1 : 1 < x0)
    (h2 : ∀ x ∈ set.Ioo 1 x0, f x - x^2 / 2 + 2 * x + 1 / 2 > k * (x - 1)) :
  k ∈ set.Ioo (-Real.top) 1 :=
by
  sorry

end part1_monotonicity_part2_range_k_l281_281825


namespace solid_brick_height_l281_281254

theorem solid_brick_height (n c base_perimeter height : ℕ) 
  (h1 : n = 42) 
  (h2 : c = 1) 
  (h3 : base_perimeter = 18)
  (h4 : n % base_area = 0)
  (h5 : 2 * (length + width) = base_perimeter)
  (h6 : base_area * height = n) : 
  height = 3 :=
by sorry

end solid_brick_height_l281_281254


namespace min_sum_of_3_digit_numbers_l281_281256

def digits : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9]

def is_3_digit (n : ℕ) := 100 ≤ n ∧ n ≤ 999

theorem min_sum_of_3_digit_numbers : 
  ∃ (a b c : ℕ), 
    a ∈ digits.permutations.map (λ l => 100*l.head! + 10*(l.tail!.head!) + l.tail!.tail!.head!) ∧ 
    b ∈ digits.permutations.map (λ l => 100*l.head! + 10*(l.tail!.head!) + l.tail!.tail!.head!) ∧ 
    c ∈ digits.permutations.map (λ l => 100*l.head! + 10*(l.tail!.head!) + l.tail!.tail!.head!) ∧ 
    a + b = c ∧ 
    a + b + c = 459 := 
sorry

end min_sum_of_3_digit_numbers_l281_281256


namespace a_lt_minus_3_l281_281847

axiom x y a : ℝ
axiom h₁ : x > y
axiom h₂ : (a + 3) * x < (a + 3) * y

theorem a_lt_minus_3 : a < -3 :=
by
  sorry

end a_lt_minus_3_l281_281847


namespace cost_per_foot_of_fence_l281_281303

theorem cost_per_foot_of_fence 
  (area : ℝ) 
  (total_cost : ℝ) 
  (h_area : area = 289) 
  (h_total_cost : total_cost = 4080) 
  : total_cost / (4 * (Real.sqrt area)) = 60 := 
by
  sorry

end cost_per_foot_of_fence_l281_281303


namespace sum_of_triangle_areas_in_cube_l281_281612

theorem sum_of_triangle_areas_in_cube (m n p : ℕ) 
  (h_cube : ∀ V T,
    (V ⊆ {p : ℝ × ℝ × ℝ | p.1 = 0 ∨ p.1 = 2 ∨ p.2 = 0 ∨ p.2 = 2 ∨ p.3 = 0 ∨ p.3 = 2})
    ∧ (∀ t ∈ T, ∃ x₁ y₁ z₁ x₂ y₂ z₂ x₃ y₃ z₃, t = triangle (x₁, y₁, z₁) (x₂, y₂, z₂) (x₃, y₃, z₃))
    ∧ (∀ t ∈ T, area t = 48 + sqrt(2304) + sqrt(3072))) :
  m + n + p = 5424 :=
begin
  sorry
end

end sum_of_triangle_areas_in_cube_l281_281612


namespace triangle_area_is_one_l281_281014

variables (a b : ℝ × ℝ)
def vector_length (v : ℝ × ℝ) : ℝ := real.sqrt (v.1 ^ 2 + v.2 ^ 2)
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- Given conditions
def a := (real.cos (2 / 3 * real.pi), real.sin (2 / 3 * real.pi)) -- definition of vector a
def OA := (a.1 - b.1, a.2 - b.2) -- definition of vector OA
def OB := (a.1 + b.1, a.2 + b.2) -- definition of vector OB

-- Proof statement
theorem triangle_area_is_one (h_isosceles_right : dot_product OA OB = 0) 
                             (h_equals_length : vector_length a = vector_length b)
                             (h_right_vertex : vector_length OA = vector_length OB) : 
                             (1 / 2) * vector_length OA * vector_length OB = 1 :=
sorry

end triangle_area_is_one_l281_281014


namespace pipe_filling_time_l281_281568

theorem pipe_filling_time (T : ℝ) (h1 : T > 0) (h2 : 1/(3:ℝ) = 1/T - 1/(6:ℝ)) : T = 2 := 
by sorry

end pipe_filling_time_l281_281568


namespace sequence_differs_in_one_position_l281_281024

/-- Given a sequence of 10 elements consisting of +1 and -1, where we repeatedly change the signs of five elements at a time, it is possible to obtain, after a certain number of steps, a sequence that differs from the original in exactly one position. -/
theorem sequence_differs_in_one_position :
  ∃ (s : Fin 10 → ℤ) (k : ℕ), 
  (∀ i : Fin 10, s i = 1 ∨ s i = -1) ∧
  (∃ seqs : List (Fin 10 → ℤ), 
    (∀ t : ℕ, t < k → 
      ∃ (indices : Finset (Fin 10)), 
      indices.card = 5 ∧ 
      (seqs.get t).map (λ i, if i ∈ indices then -(s i) else s i) = seqs.get (t+1)) ∧
    ∃ final_seq : Fin 10 → ℤ, 
      seqs.head = s ∧
      seqs.get k = final_seq ∧
      ((Finset.univ.filter (λ i, s i ≠ final_seq i)).card = 1
      )
    )
  :=
sorry

end sequence_differs_in_one_position_l281_281024


namespace first_head_on_third_flip_l281_281652

def fair_coin_flip : ℙ Bool := 
  { p_head := 1 / 2, p_tail := 1 / 2 }

def prob_first_head_on_third_flip (coin_flip : ℙ Bool) : ℚ :=
  coin_flip.p_tail * coin_flip.p_tail * coin_flip.p_head

theorem first_head_on_third_flip : 
  prob_first_head_on_third_flip fair_coin_flip = 1 / 8 :=
by
  sorry

end first_head_on_third_flip_l281_281652


namespace round_repeating_decimal_l281_281579

theorem round_repeating_decimal (r : ℝ) (h₁ : r = 23.3636363636) (h₂ : ∀ n : ℕ, (23.3636363636 + ∑ i in finset.range n, (3 / 10^(2*(i+1))) + 6 / 10^(2*(i+2))) = 23.4)  : 
  Float.round (r * 10) / 10 = 23.4 :=
by
  sorry

end round_repeating_decimal_l281_281579


namespace num_zeros_in_interval_l281_281228

def f (x : ℝ) : ℝ := 2 * x ^ 3 - 6 * x ^ 2 + 7

theorem num_zeros_in_interval : 
    (∃ (a b : ℝ), a < b ∧ a = 0 ∧ b = 2 ∧
     (∀ x, f x = 0 → (0 < x ∧ x < 2)) ∧
     (∃! x, (0 < x ∧ x < 2) ∧ f x = 0)) :=
by
    sorry

end num_zeros_in_interval_l281_281228


namespace overall_percentage_change_l281_281578

theorem overall_percentage_change (S : ℝ) : 
  let after_first_decrease := 0.65 * S,
  after_first_increase := after_first_decrease * 1.2,
  after_second_decrease := after_first_increase * 0.85,
  final_salary := after_second_decrease * 1.25 in
  ((final_salary - S) / S) * 100 = -17.125 := 
by {
  -- Let bindings translated from the conditions
  let after_first_decrease := 0.65 * S,
  let after_first_increase := after_first_decrease * 1.2,
  let after_second_decrease := after_first_increase * 0.85,
  let final_salary := after_second_decrease * 1.25,

  -- Put the final statement
  sorry
}

end overall_percentage_change_l281_281578


namespace Monica_tiles_count_l281_281938

noncomputable def total_tiles (length width : ℕ) := 
  let double_border_tiles := (2 * ((length - 4) + (width - 4)) + 8)
  let inner_area := (length - 4) * (width - 4)
  let three_foot_tiles := (inner_area + 8) / 9
  double_border_tiles + three_foot_tiles

theorem Monica_tiles_count : total_tiles 18 24 = 183 := 
by
  sorry

end Monica_tiles_count_l281_281938


namespace initial_selling_price_l281_281399

theorem initial_selling_price (P : ℝ) : 
    (∀ (c_i c_m p_m r : ℝ),
        c_i = 3 ∧
        c_m = 20 ∧
        p_m = 4 ∧
        r = 50 ∧
        (15 * P + 5 * p_m - 20 * c_i = r)
    ) → 
    P = 6 := by 
    sorry

end initial_selling_price_l281_281399


namespace sqrt3_eq_sqrt_81_eq_abs_sub_eq_sqrt_fraction_eq_simplify_expression_eq_l281_281003

noncomputable def sqrt_3 : ℝ := Real.sqrt 3
theorem sqrt3_eq : sqrt_3 = Real.sqrt 3 := 
by sorry

theorem sqrt_81_eq : Real.sqrt 81 = 9 ∨ Real.sqrt 81 = -9 :=
by sorry

noncomputable def abs_x_y : ℝ := Real.sqrt 5 - 2
theorem abs_sub_eq : Real.abs (2 - Real.sqrt 5) = abs_x_y :=
by sorry

noncomputable def sqrt_fraction : ℝ := (2 : ℝ) / (11 : ℝ)
theorem sqrt_fraction_eq : Real.sqrt (4 / 121) = sqrt_fraction :=
by sorry

theorem simplify_expression_eq : 2 * Real.sqrt 3 - 5 * Real.sqrt 3 = -3 * Real.sqrt 3 :=
by sorry

end sqrt3_eq_sqrt_81_eq_abs_sub_eq_sqrt_fraction_eq_simplify_expression_eq_l281_281003


namespace min_tourism_income_l281_281344

noncomputable def f (t : ℕ) : ℝ := 4 + 1 / t
noncomputable def g (t : ℕ) : ℝ := 115 - |t - 15|
noncomputable def w (t : ℕ) : ℝ :=
  if 1 ≤ t ∧ t < 15 then (4 + 1 / t) * (t + 100)
  else if 15 ≤ t ∧ t ≤ 30 then (4 + 1 / t) * (130 - t)
  else 0

theorem min_tourism_income : ∃ t (ht : 1 ≤ t ∧ t ≤ 30 ∧ t ∈ ℕ), w(t) = (1210 / 3) :=
by
  sorry

end min_tourism_income_l281_281344


namespace find_n_in_sequence_l281_281065

theorem find_n_in_sequence (n : ℕ) (a : ℕ → ℕ) (S : ℕ → ℕ) 
    (h1 : a 1 = 2) 
    (h2 : ∀ n, a (n+1) = 2 * a n) 
    (h3 : S n = 126) 
    (h4 : S n = 2^(n+1) - 2) : 
  n = 6 :=
sorry

end find_n_in_sequence_l281_281065


namespace Horner_method_eval_l281_281255

noncomputable def f (x : ℝ) : ℝ := 3 * x^4 + 2 * x^2 + x + 4

theorem Horner_method_eval :
  let x := 10
  let v₀ := 3
  let v₁ := v₀ * x + 2
  v₁ = 32 :=
by
  let x := 10
  let v₀ := 3
  let v₁ := v₀ * x + 2
  show v₁ = 32
  sorry

end Horner_method_eval_l281_281255


namespace simplify_expression_l281_281201

theorem simplify_expression :
  (8 : ℝ)^(1/3) - (343 : ℝ)^(1/3) = -5 :=
by
  sorry

end simplify_expression_l281_281201


namespace lean_proof_l281_281488

theorem lean_proof (a : ℝ) (h : a = real.cos (2 * real.pi / 7)) : 2^(a - 1/2) < 2 * a :=
sorry

end lean_proof_l281_281488


namespace combined_purchase_saves_more_l281_281727

-- Define window costs and the store's offer
def window_cost : ℕ := 125
def free_windows_offer (purchased: ℕ) : ℕ := (purchased / 6) * 2

-- Define the total number of windows needed by Dave and Doug
def dave_windows : ℕ := 9
def doug_windows : ℕ := 11

-- Define separate and combined window purchases
def separate_purchases : ℕ := dave_windows + doug_windows
def combined_purchases : ℕ := separate_purchases

-- Calculate costs
def cost_no_offer (num_windows: ℕ) : ℕ := num_windows * window_cost
def cost_with_offer (num_windows: ℕ) : ℕ :=
  let purchased := num_windows - free_windows_offer(num_windows)
  purchased * window_cost

-- Separate purchase costs and savings
def dave_cost_no_offer : ℕ := cost_no_offer(dave_windows)
def dave_cost_with_offer : ℕ := cost_with_offer(dave_windows)
def dave_savings : ℕ := dave_cost_no_offer - dave_cost_with_offer

def doug_cost_no_offer : ℕ := cost_no_offer(doug_windows)
def doug_cost_with_offer : ℕ := cost_with_offer(doug_windows)
def doug_savings : ℕ := doug_cost_no_offer - doug_cost_with_offer

def total_savings_separate : ℕ := dave_savings + doug_savings

-- Combined purchase costs and savings
def combined_cost_no_offer : ℕ := cost_no_offer(combined_purchases)
def combined_cost_with_offer : ℕ := cost_with_offer(combined_purchases)
def combined_savings : ℕ := combined_cost_no_offer - combined_cost_with_offer

-- Additional savings when purchased together
def additional_savings : ℕ := combined_savings - total_savings_separate

theorem combined_purchase_saves_more :
  additional_savings = 250 :=
by
  sorry

end combined_purchase_saves_more_l281_281727


namespace solve_expression_l281_281649

theorem solve_expression : 
  (let a := 315; let b := 285 in (a^2 - b^2) / 30) = 600 := by
  sorry

end solve_expression_l281_281649


namespace prob_A_wins_match_is_correct_l281_281569

/-- Definitions -/

def prob_A_wins_game : ℝ := 0.6

def prob_B_wins_game : ℝ := 1 - prob_A_wins_game

def prob_A_wins_match (p: ℝ) : ℝ :=
  p * p * (1 - p) + p * (1 - p) * p + p * p

/-- Theorem -/

theorem prob_A_wins_match_is_correct : 
  prob_A_wins_match prob_A_wins_game = 0.648 :=
by
  sorry

end prob_A_wins_match_is_correct_l281_281569


namespace arrangement_exists_l281_281361

-- Definitions of pairwise coprimeness and gcd
def pairwise_coprime (a b c d : ℕ) : Prop :=
  Nat.gcd a b = 1 ∧ Nat.gcd a c = 1 ∧ Nat.gcd a d = 1 ∧ Nat.gcd b c = 1 ∧ Nat.gcd b d = 1 ∧ Nat.gcd c d = 1

def common_divisor (x y : ℕ) : Prop := ∃ d > 1, d ∣ x ∧ d ∣ y

def relatively_prime (x y : ℕ) : Prop := Nat.gcd x y = 1

-- Main theorem statement
theorem arrangement_exists :
  ∃ a b c d ab cd ad bc abcd : ℕ,
    pairwise_coprime a b c d ∧
    ab = a * b ∧ cd = c * d ∧ ad = a * d ∧ bc = b * c ∧ abcd = a * b * c * d ∧
    (common_divisor ab abcd ∧ common_divisor cd abcd ∧ common_divisor ad abcd ∧ common_divisor bc abcd) ∧
    (common_divisor ab ad ∧ common_divisor ab bc ∧ common_divisor cd ad ∧ common_divisor cd bc) ∧
    (relatively_prime ab cd ∧ relatively_prime ad bc) :=
by
  -- The proof will be filled here
  sorry

end arrangement_exists_l281_281361


namespace spider_human_legs_multiple_l281_281725

theorem spider_human_legs_multiple (human_legs : ℕ) (spider_legs : ℕ) (h1 : human_legs = 2) (h2 : spider_legs = 8) : spider_legs / human_legs = 4 :=
by
  rw [h1, h2]
  norm_num

end spider_human_legs_multiple_l281_281725


namespace acute_triangle_inequality_l281_281190

theorem acute_triangle_inequality 
  (a b c s_a s_b s_c t : ℝ)
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
  (h4 : s_a = √((2b^2 + 2c^2 - a^2) / 4))
  (h5 : s_b = √((2a^2 + 2c^2 - b^2) / 4))
  (h6 : s_c = √((2a^2 + 2b^2 - c^2) / 4))
  (h7 : t = (1 / 2) * a * b * c) :
  (1 / (s_a^2 - a^2 / 4)) + (1 / (s_b^2 - b^2 / 4)) + (1 / (s_c^2 - c^2 / 4)) 
  ≥ (3 * √3) / (2 * t) := 
sorry

end acute_triangle_inequality_l281_281190


namespace calculate_sum_of_perimeters_l281_281956

open EuclideanGeometry

noncomputable def sum_of_perimeters (A B C D : Point) : ℕ :=
if (dist A B = 12 ∧ dist B C = 18 ∧ dist A C = 30 ∧ dist A D = dist C D ∧ 
    ∃ x y : ℕ, dist A D = x ∧ dist B D = y ∧ 
    (x^2 - y^2 = 216))
then 320
else 0

theorem calculate_sum_of_perimeters (A B C D : Point) (h1 : dist A B = 12) 
  (h2 : dist B C = 18) (h3 : dist A C = 30) (h4 : dist A D = dist C D)
  (h5 : ∃ (x y : ℕ), dist A D = x ∧ dist B D = y ∧ x^2 - y^2 = 216) :
  sum_of_perimeters A B C D = 320 := by
    sorry

end calculate_sum_of_perimeters_l281_281956


namespace continuity_f1_discontinuity_f1_at_origin_continuity_f2_continuity_f3_discontinuity_f3_at_axes_l281_281881

-- Problem 1
def f1 (x y : ℝ) : ℝ := if (x ≠ 0 ∨ y ≠ 0) then 2 * x^2 * y^2 / (x^4 + y^4) else 0
theorem continuity_f1 (x y : ℝ) : (x ≠ 0 ∨ y ≠ 0) → continuous_at (λ (p : ℝ × ℝ), f1 p.1 p.2) (x, y) :=
sorry

theorem discontinuity_f1_at_origin : ¬ continuous_at (λ (p : ℝ × ℝ), f1 p.1 p.2) (0, 0) :=
sorry

-- Problem 2
def f2 (x y : ℝ) : ℝ := if (x ≠ 0 ∨ y ≠ 0) then (sin (x^2 + y^2)) / (x^2 + y^2) else 1
theorem continuity_f2 (x y : ℝ) : continuous_at (λ (p : ℝ × ℝ), f2 p.1 p.2) (x, y) :=
sorry

-- Problem 3
def f3 (x y : ℝ) : ℝ := if (x ≠ 0 ∧ y ≠ 0) then sin(3 / (x * y)) else 0
theorem continuity_f3 (x y : ℝ) : (x ≠ 0 ∧ y ≠ 0) → continuous_at (λ (p : ℝ × ℝ), f3 p.1 p.2) (x, y) :=
sorry

theorem discontinuity_f3_at_axes (x y : ℝ) : (x = 0 ∨ y = 0) → ¬ continuous_at (λ (p : ℝ × ℝ), f3 p.1 p.2) (x, y) :=
sorry

end continuity_f1_discontinuity_f1_at_origin_continuity_f2_continuity_f3_discontinuity_f3_at_axes_l281_281881


namespace problem1_problem2_problem3_l281_281875

-- Definitions based on the conditions and the sequence properties
def a_seq (c : ℝ) (a : ℕ+ → ℝ) : Prop :=
  (a 1 = 1) ∧ ∀ n : ℕ+, a (n + 1) = a n / (c * a n + 1)

def is_geometric_sequence (x y z : ℝ) : Prop :=
  y * y = x * z ∧ x ≠ z

def bn (c : ℝ) (a : ℕ+ → ℝ) (n : ℕ+) : ℝ :=
  a n * a (n + 1)

def Sn (c : ℝ) (a : ℕ+ → ℝ) (n : ℕ) : ℝ :=
  (1 / 2) * (1 - 1 / (2 * n + 1))

-- Proof problems in Lean 4
theorem problem1 (c : ℝ) (a : ℕ+ → ℝ) (h : a_seq c a) :
  ∃ d : ℝ, ∀ n : ℕ+, 1 / a (n + 1) - 1 / a n = d := sorry

theorem problem2 : ∃ c : ℝ, c = 2 := sorry

theorem problem3 (c : ℝ := 2) (a : ℕ+ → ℝ) (h : a_seq c a) (n : ℕ) :
  (∑ k in finset.range n, bn c a (⟨k + 1, nat.succ_pos k⟩)) = Sn c a n := sorry

end problem1_problem2_problem3_l281_281875


namespace symmetric_point_yOz_l281_281519

-- Given point A in 3D Cartesian system
def A : ℝ × ℝ × ℝ := (1, -3, 5)

-- Plane yOz where x = 0
def symmetric_yOz (point : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := point
  (-x, y, z)

-- Proof statement (without the actual proof)
theorem symmetric_point_yOz : symmetric_yOz A = (-1, -3, 5) :=
by sorry

end symmetric_point_yOz_l281_281519


namespace equation_of_ellipse_product_of_slopes_slope_and_area_l281_281026

-- Given definitions based on conditions in a)
def ellipse (a b : ℝ) := ∀ (x y : ℝ), (x^2 / a^2 + y^2 / b^2 = 1)
def focus (F : ℝ × ℝ) := F = (1, 0)
def short_axis_length (b : ℝ) := 2 * b = 2
def intersects (A B : ℝ × ℝ) (l : ℝ → ℝ) := ∃ (x y : ℝ), (l x = y) ∧ ((x, y) = A ∨ (x, y) = B)
def midpoint (A B M : ℝ × ℝ) := M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- 1. Equation of the ellipse
theorem equation_of_ellipse (a b : ℝ) (h1 : a > b ∧ b > 0) (h2 : (1, 0)) (h3 : 2 * b = 2) :
  ellipse (sqrt 2) 1 := by
  sorry

-- 2. Product of slopes of OM and l
theorem product_of_slopes (k : ℝ) (A B M : ℝ × ℝ) (h4 : ¬ (k = 0)) 
  (h5 : (k * (M.1 - 1)) = M.2) (h6 : midpoint A B M) :
  (M.2 / (M.1 - 0)) * k = -1 / 2 := by
  sorry

-- 3. Slope of l and area of quadrilateral OAPB
theorem slope_and_area (A B P : ℝ × ℝ) (l : ℝ → ℝ) (h7 : P = (4 * k^2 / (2 * k^2 + 1), -2 * k / (2 * k^2 + 1))) 
  (h8 : A = (A.1, k * (A.1 - 1))) (h9 : B = (B.1, k * (B.1 - 1))) (h10 : l = (k)) :
  l = ±(sqrt 2 / 2) ∧ (area l O A P B = sqrt 6 / 2) := by
  sorry

end equation_of_ellipse_product_of_slopes_slope_and_area_l281_281026


namespace find_a_l281_281050

variable (a : ℝ)

def augmented_matrix (a : ℝ) :=
  ([1, -1, -3], [a, 3, 4])

def solution := (-1, 2)

theorem find_a (hx : -1 - 2 = -3)
               (hy : a * (-1) + 3 * 2 = 4) :
               a = 2 :=
by
  sorry

end find_a_l281_281050


namespace number_of_elements_in_A_inter_Z_l281_281833

-- Define the sets A and Z
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}
def Z : Set ℤ := {z | True} -- ℤ is the set of all integers

-- Define the intersection of A and Z considered as ℝ and ℤ respectively
def A_inter_Z : Set ℤ := {z : ℤ | (z : ℝ) ∈ A}

-- State the proof problem as a theorem
theorem number_of_elements_in_A_inter_Z : (A_inter_Z.to_finset.card = 5) :=
by
  -- proof steps skipped
  sorry

end number_of_elements_in_A_inter_Z_l281_281833


namespace min_value_of_abc_l281_281106

noncomputable def minimum_value_abc (a b c : ℝ) : Prop :=
  (a > 0) ∧ (b > 0) ∧ (c > 0) ∧ ((a + c) * (a + b) = 6 - 2 * Real.sqrt 5) → (2 * a + b + c ≥ 2 * Real.sqrt 5 - 2)

theorem min_value_of_abc (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : (a + c) * (a + b) = 6 - 2 * Real.sqrt 5) : 
  2 * a + b + c ≥ 2 * Real.sqrt 5 - 2 :=
by 
  sorry

end min_value_of_abc_l281_281106


namespace infinite_sum_l281_281376

theorem infinite_sum:
  ∑ k in (filter (λ n, n ≥ 1) (range (n + 1))) (8^k / ((4^k - 3^k) * (4^(k + 1) - 3^(k + 1)))) = 2 :=
sorry

end infinite_sum_l281_281376


namespace crayons_selection_l281_281236

theorem crayons_selection (n k : ℕ) (h_n : n = 15) (h_k : k = 5) :
  Nat.choose n k = 3003 := 
by
  rw [h_n, h_k]
  rfl

end crayons_selection_l281_281236


namespace exists_another_point_with_equal_cevians_l281_281546

-- Definitions of the necessary objects and conditions
variables (A B C P : Point) (triangleABC : Triangle A B C) 
variables (is_non_equilateral : ¬(Triangle.isEquilateral triangleABC))
variables (lambda : ℝ)
variables (cevians_P_equal : ∀ (D E F : Point), (Cevian.from P D) = (Cevian.from P E) ∧ (Cevian.from P E) = (Cevian.from P F) ∧ (Cevian.from P D) < Segment.length A B)
variables (lambda_lt_sides : lambda < Segment.length A B ∧ lambda < Segment.length B C ∧ lambda < Segment.length C A)

-- The goal to be proven
theorem exists_another_point_with_equal_cevians (A B C : Point) (triangleABC : Triangle A B C)
  (is_non_equilateral : ¬(Triangle.isEquilateral triangleABC))
  (P : Point) (lambda : ℝ)
  (cevians_P_equal : ∀ (D E F : Point), 
    (Cevian.from P D).length = lambda ∧ 
    (Cevian.from P E).length = lambda ∧ 
    (Cevian.from P F).length = lambda ∧ 
    lambda < Segment.length A B)
  (lambda_lt_sides : lambda < Segment.length A B ∧ lambda < Segment.length B C ∧ lambda < Segment.length C A) :
  ∃ (P' : Point), P' ≠ P ∧ 
    ∃ (D' E' F' : Point), 
    (Cevian.from P' D').length = lambda ∧ 
    (Cevian.from P' E').length = lambda ∧ 
    (Cevian.from P' F').length = lambda :=
sorry

end exists_another_point_with_equal_cevians_l281_281546


namespace kindergarten_tissues_l281_281623

theorem kindergarten_tissues 
  (n1 n2 n3 n4 n5 : ℕ) 
  (tissue_per_box : ℕ) 
  (students_each_box : ∀ n, n = 1) 
  (total_tissues : ℕ)
  (h1 : n1 = 15)
  (h2 : n2 = 20)
  (h3 : n3 = 18)
  (h4 : n4 = 22)
  (h5 : n5 = 25)
  (h_tissue_per_box : tissue_per_box = 70) :
  total_tissues = (n1 + n2 + n3 + n4 + n5) * tissue_per_box := 
begin
  sorry
end

end kindergarten_tissues_l281_281623


namespace product_inequality_l281_281436

variables {n : ℕ}
variables {r s t u v : Fin n → ℝ}

def R := (∑ i, r i) / n
def S := (∑ i, s i) / n
def T := (∑ i, t i) / n
def U := (∑ i, u i) / n
def V := (∑ i, v i) / n

theorem product_inequality
  (hr : ∀ i, r i > 1) (hs : ∀ i, s i > 1) (ht : ∀ i, t i > 1) (hu : ∀ i, u i > 1) (hv : ∀ i, v i > 1) :
  (∏ i, (r i * s i * t i * u i * v i + 1) / (r i * s i * t i * u i * v i - 1)) ≥ 
  (RS * S * T * U * V + 1) / (R * S * T * U * V - 1) ^ n :=
sorry

end product_inequality_l281_281436


namespace harkamal_total_amount_l281_281838

-- Define the conditions as constants
def quantity_grapes : ℕ := 10
def rate_grapes : ℕ := 70
def quantity_mangoes : ℕ := 9
def rate_mangoes : ℕ := 55

-- Define the cost of grapes and mangoes based on the given conditions
def cost_grapes : ℕ := quantity_grapes * rate_grapes
def cost_mangoes : ℕ := quantity_mangoes * rate_mangoes

-- Define the total amount paid
def total_amount_paid : ℕ := cost_grapes + cost_mangoes

-- The theorem stating the problem and the solution
theorem harkamal_total_amount : total_amount_paid = 1195 := by
  -- Proof goes here (omitted)
  sorry

end harkamal_total_amount_l281_281838


namespace hyperbola_correct_equation_l281_281593

def hyperbola_equation (focus_x focus_y : ℝ) (asymptote1 asymptote2 : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), focus_y = 10 ∧
    asymptote1 = λ x, (4 / 3) * x ∧
    asymptote2 = λ x, -(4 / 3) * x ∧
    a = 8 ∧ b = 6 ∧ 
    (∀ x y : ℝ, (y^2 / a^2) - (x^2 / b^2) = 1 ↔ y^2 / 64 - x^2 / 36 = 1)

theorem hyperbola_correct_equation :
  hyperbola_equation 0 10 (λ x : ℝ, (4 / 3) * x) (λ x : ℝ, -(4 / 3) * x) :=
begin
  sorry
end

end hyperbola_correct_equation_l281_281593


namespace fully_factor_expression_l281_281426

theorem fully_factor_expression (x : ℝ) : 3 * x^2 - 75 = 3 * (x + 5) * (x - 5) := 
by
  -- pending proof, represented by sorry
  sorry

end fully_factor_expression_l281_281426


namespace vector_dot_product_l281_281476

-- Definitions of the vectors
def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (-1, 2)

-- Definition of the dot product for 2D vectors
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Main statement to prove
theorem vector_dot_product :
  dot_product (a.1 + b.1, a.2 + b.2) (a.1 - b.1, a.2 - b.2) = 0 :=
by
  sorry

end vector_dot_product_l281_281476


namespace sqrt_sequence_term_equals_5sqrt5_l281_281471

theorem sqrt_sequence_term_equals_5sqrt5 : 
  ∃ n : ℕ, (sqrt (6 * n - 1) = 5 * sqrt 5) ∧ n = 21 :=
by {
  use 21,
  split,
  {
    -- we need to use some properties about square root and basic arithmetic here
    sorry
  },
  {
    refl
  }
}

end sqrt_sequence_term_equals_5sqrt5_l281_281471


namespace integer_not_in_range_l281_281157

theorem integer_not_in_range (g : ℝ → ℤ) :
  (∀ x, x > -3 → g x = Int.ceil (2 / (x + 3))) ∧
  (∀ x, x < -3 → g x = Int.floor (2 / (x + 3))) →
  ∀ z : ℤ, (∃ x, g x = z) ↔ z ≠ 0 :=
by
  intros h z
  sorry

end integer_not_in_range_l281_281157


namespace slope_tangent_at_zero_l281_281611

noncomputable def f (x : ℝ) : ℝ := Real.exp x + x^2 - 2 * x

def derivative_f (x : ℝ) : ℝ := Real.exp x + 2 * x - 2

theorem slope_tangent_at_zero : derivative_f 0 = -1 :=
by
  -- Proof would go here
  sorry

end slope_tangent_at_zero_l281_281611


namespace find_difference_of_squares_l281_281087

variable (x y : ℝ)
variable (h1 : (x + y) ^ 2 = 81)
variable (h2 : x * y = 18)

theorem find_difference_of_squares : (x - y) ^ 2 = 9 := by
  sorry

end find_difference_of_squares_l281_281087


namespace distance_PQ_l281_281928

theorem distance_PQ (P : ℝ × ℝ) (inclination : ℝ) (l2 : ℝ → ℝ → Prop) :
  P = (3, 2) →
  inclination = Real.arctan (3 / 4) →
  (∀ x y, l2 x y ↔ x - 2 * y + 11 = 0) →
  ∃ Q : ℝ × ℝ, (∃ t : ℝ, Q = (3 + 4 / 5 * t, 2 + 3 / 5 * t) ∧ l2 Q.1 Q.2) ∧ 
  Real.dist P Q = 25 :=
by
  intros hP hIncl hL2
  sorry

end distance_PQ_l281_281928


namespace x_lt_0_is_necessary_but_not_sufficient_for_ln_x_plus_1_lt_0_l281_281677

theorem x_lt_0_is_necessary_but_not_sufficient_for_ln_x_plus_1_lt_0 (x : ℝ) :
  ln(x + 1) < 0 → (x < 0 ∧ ¬(x < 0 → ln(x + 1) < 0)) :=
by
  sorry

end x_lt_0_is_necessary_but_not_sufficient_for_ln_x_plus_1_lt_0_l281_281677


namespace correct_operation_l281_281656

theorem correct_operation (x : ℝ) : (2 * x ^ 3) ^ 2 = 4 * x ^ 6 := 
  sorry

end correct_operation_l281_281656


namespace ellipse_a_value_l281_281027

theorem ellipse_a_value
  (a : ℝ)
  (h1 : 0 < a)
  (h2 : ∀ x y : ℝ, x^2 / a^2 + y^2 / 5 = 1)
  (e : ℝ)
  (h3 : e = 2 / 3)
  : a = 3 :=
by
  sorry

end ellipse_a_value_l281_281027


namespace distance_to_y_axis_l281_281799

-- Define the point P is on the parabola y^2 = 8x
def on_parabola (P : ℝ × ℝ) : Prop :=
  ∃ y x : ℝ, P = (x, y) ∧ y^2 = 8 * x

-- Define the distance between two points in the plane
def distance (A B : ℝ × ℝ) : ℝ :=
  real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- The main theorem to prove
theorem distance_to_y_axis {P : ℝ × ℝ} 
  (h₁ : on_parabola P)
  (h₂ : distance P (2, 0) = 6) :
  P.1 = 4 :=
begin
  sorry
end

end distance_to_y_axis_l281_281799


namespace plus_minus_pairs_l281_281620

theorem plus_minus_pairs (a b p q : ℕ) (h_plus_pairs : p = a) (h_minus_pairs : q = b) : 
  a - b = p - q := 
by 
  sorry

end plus_minus_pairs_l281_281620


namespace exists_n_no_rational_root_l281_281716

theorem exists_n_no_rational_root (a b c : ℝ) (h : ∃ x : ℝ, a*x*x + b*x + c = 0) :
    ∃ n : ℕ, ∀ u : ℚ, a * (u.val / u.denom)^2 + b * (u.val / u.denom) + c ≠ 1 / n := 
by {
    sorry,
}

end exists_n_no_rational_root_l281_281716


namespace probability_point_in_ellipse_l281_281809

noncomputable def is_in_ellipse (x y : ℕ) : Prop :=
  (x^2 : ℝ) / 16 + (y^2 : ℝ) / 9 ≤ 1

theorem probability_point_in_ellipse :
  let a := [4, 2, 0, 1, 3] in
  (∑ p in (a.product a), if is_in_ellipse p.1 p.2 then 1 else 0) = 11 →
  let total_points := a.length * a.length in
  (∑ p in (a.product a), if is_in_ellipse p.1 p.2 then 1 else 0) / total_points = 11 / 25 := by 
  sorry

end probability_point_in_ellipse_l281_281809


namespace predict_y_at_5_l281_281047

noncomputable theory
open Classical

-- Conditions
def regression_eqn (b x : ℝ) : ℝ := Real.exp (b * x - 0.5)
def transform_eqn (y : ℝ) : ℝ := Real.log y
def transformed_reg_eqn (b x : ℝ) : ℝ := b * x - 0.5

-- Data points
def x_vals : List ℝ := [1, 2, 3, 4]
def y_vals : List ℝ := [Real.exp 1, Real.exp 3, Real.exp 4, Real.exp 6]
def z_vals : List ℝ := [1, 3, 4, 6]

-- Question: Prove the predicted value of y when x = 5
theorem predict_y_at_5 :
  ∃ b : ℝ, (∀ (i : ℕ) (h : i < 4), z_vals.get i h = b * x_vals.get i h - 0.5) →
  regression_eqn b 5 = Real.exp (15 / 2) := sorry

end predict_y_at_5_l281_281047


namespace horner_method_multiplications_horner_method_additions_l281_281635

def polynomial := 9 * x^6 + 12 * x^5 + 7 * x^4 + 54 * x^3 + 34 * x^2 + 9 * x + 1

theorem horner_method_multiplications : 
  ∀ x, (calc
    let x1 := 9 * x
    let x2 := x1 + 12
    let x3 := x2 * x
    let x4 := x3 + 7
    let x5 := x4 * x
    let x6 := x5 + 54
    let x7 := x6 * x
    let x8 := x7 + 34
    let x9 := x8 * x
    let x10 := x9 + 9
    let x11 := x10 * x
    let x12 := x11 + 1
    in x1 * x * x * x * x * x).prop = 6 := sorry

theorem horner_method_additions : 
  ∀ x, (calc
    let x1 := 9 * x
    let x2 := x1 + 12
    let x3 := x2 * x
    let x4 := x3 + 7
    let x5 := x4 * x
    let x6 := x5 + 54
    let x7 := x6 * x
    let x8 := x7 + 34
    let x9 := x8 * x
    let x10 := x9 + 9
    let x11 := x10 * x
    let x12 := x11 + 1
    in x2 + x4 + x6 + x8 + x10 + x12).prop = 6 := sorry

end horner_method_multiplications_horner_method_additions_l281_281635


namespace count_of_ordered_triples_l281_281333

theorem count_of_ordered_triples :
  { (a, b, c) : ℕ × ℕ × ℕ // 1 ≤ a ∧ a ≤ b ∧ b ≤ c ∧ (1 / a + 1 / b + 1 / c) = 1 / 2 }.card = 10 :=
sorry

end count_of_ordered_triples_l281_281333


namespace distance_between_foci_of_ellipse_l281_281778

-- Definitions based on conditions in the problem
def ellipse_eq (x y : ℝ) : Prop := (x^2) / 45 + (y^2) / 5 = 9

-- Proof statement
theorem distance_between_foci_of_ellipse : 
  ∀ (x y : ℝ), ellipse_eq x y → 2 * (√ (405 - 45)) = 12 * (√ 10) :=
by 
  sorry

end distance_between_foci_of_ellipse_l281_281778


namespace similar_triangles_length_LM_l281_281311

theorem similar_triangles_length_LM
  (H I J K L M : Type)
  (HI : ℝ) (IJ : ℝ) (KL : ℝ) (LM : ℝ)
  (h_sim : similar_triangles H I J K L M)
  (h_HI : HI = 9)
  (h_IJ : IJ = 12)
  (h_KL : KL = 6) :
  LM = 8 :=
by
  sorry

end similar_triangles_length_LM_l281_281311


namespace lowest_degree_polynomial_is_4_l281_281265

noncomputable def lowest_degree_polynomial : ℕ :=
  let exists_polynomial_of_degree_four_with_conditions := ∃ (P : Polynomial ℤ), P.degree = 4 ∧
    ∃ b : ℤ, ∀ coeff ∈ P.coeffs, (coeff < b ∨ coeff > b) ∧ coeff ≠ b
  let no_polynomial_of_degree_less_than_four_with_conditions := ∀ (d < 4), ¬∃ (P : Polynomial ℤ), P.degree = d ∧
    ∃ b : ℤ, ∀ coeff ∈ P.coeffs, (coeff < b ∨ coeff > b) ∧ coeff ≠ b
  if h₁ : exists_polynomial_of_degree_four_with_conditions ∧ no_polynomial_of_degree_less_than_four_with_conditions then 4 else 0

theorem lowest_degree_polynomial_is_4 :
  lowest_degree_polynomial = 4 :=
by
  sorry

end lowest_degree_polynomial_is_4_l281_281265


namespace largest_four_digit_perfect_cube_l281_281671

-- Define what it means for a number to be a perfect cube
def is_perfect_cube (n : ℕ) : Prop :=
  ∃ k : ℕ, k ^ 3 = n

-- Define what it means for a number to be a four-digit number
def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

-- State the problem as a theorem
theorem largest_four_digit_perfect_cube :
  ∃ n, is_four_digit n ∧ is_perfect_cube n ∧ ∀ m, is_four_digit m ∧ is_perfect_cube m → m ≤ n :=
begin
  use 9261,
  split,
  { -- 9261 is a four-digit number
    split,
    { exact Nat.le_of_lt (Nat.lt_of_lt_of_le (show 1000 < 9261 by norm_num) (show 9261 < 10000 by norm_num)) },
    { norm_num },
  },
  split,
  { -- 9261 is a perfect cube
    use 21,
    norm_num,
  },
  { -- 9261 is the largest such number
    intros m hm,
    rcases hm with ⟨h1, ⟨k, h2⟩⟩,
    rw [←h2, ←Nat.le_mul_iff_one_le_right_dec’ k (show k^2 * k > 0 by nlinarith)],
    norm_num [h1] using k,
  },
end
  

end largest_four_digit_perfect_cube_l281_281671


namespace sum_positive_implies_at_least_one_positive_l281_281109

variables {a b : ℝ}

theorem sum_positive_implies_at_least_one_positive (h : a + b > 0) : a > 0 ∨ b > 0 :=
sorry

end sum_positive_implies_at_least_one_positive_l281_281109


namespace matrix_cubed_values_l281_281532

theorem matrix_cubed_values (d e f : ℂ) (h_condition : (matrix 3 3 ℂ))
  (h_def : (det (matrix d e f e f d f d e) = -1))
  (h_identity : (matrix d e f e f d f d e) ^ 2 = 1) :
  d^3 + e^3 + f^3 = 2 ∨ d^3 + e^3 + f^3 = 4 :=
by
  sorry

end matrix_cubed_values_l281_281532


namespace joan_balloons_l281_281890

-- Defining the condition
def melanie_balloons : ℕ := 41
def total_balloons : ℕ := 81

-- Stating the theorem
theorem joan_balloons :
  ∃ (joan_balloons : ℕ), joan_balloons = total_balloons - melanie_balloons ∧ joan_balloons = 40 :=
by
  -- Placeholder for the proof
  sorry

end joan_balloons_l281_281890


namespace part_I_part_II_l281_281791

open Real

variable (a b : ℝ)

theorem part_I (h₁ : a > 0) (h₂ : b > 0) (h₃ : a + b = 1) : (1 / a^2) + (1 / b^2) ≥ 8 := 
sorry

theorem part_II (h₁ : a > 0) (h₂ : b > 0) (h₃ : a + b = 1) : (1 / a) + (1 / b) + (1 / (a * b)) ≥ 8 := 
sorry

end part_I_part_II_l281_281791


namespace lowest_degree_poly_meets_conditions_l281_281302

-- Define a predicate that checks if a polynomial P meets the conditions
def poly_meets_conditions (P : ℚ[X]) (b : ℚ) : Prop :=
  (∀ x, coeff P x ≠ b) ∧ 
  (∃ x y, coeff P x < b ∧ coeff P y > b)

-- Statement of the theorem we want to prove
theorem lowest_degree_poly_meets_conditions : ∀ (b : ℚ), 
  ∃ (P : ℚ[X]), poly_meets_conditions P b ∧ degree P = 4 :=
begin
  sorry
end

end lowest_degree_poly_meets_conditions_l281_281302


namespace nilpotent_of_invertibility_l281_281895

variables {A : Type*} [ring A] [fintype A]

theorem nilpotent_of_invertibility (m : ℕ) (n : ℕ) (hn : 2 ≤ n)
  (hA : fintype.card A = n) (a : A)
  (hinv : ∀ k ∈ (finset.range (m + n)).filter (λ k, m < k), is_unit (1 - a ^ k)) :
  ∃ l : ℕ, a ^ l = 0 :=
sorry

end nilpotent_of_invertibility_l281_281895


namespace limit_w_div_N_eq_one_div_e_l281_281750

noncomputable def w : ℕ → ℕ
| 0       => 0
| (N + 1) => (1 + N) * (Finset.sum (Finset.range (N + 2)) (fun i => (-1 : ℝ) ^ i / Real.fact i))

theorem limit_w_div_N_eq_one_div_e :
  Filter.Tendsto (fun N => w N / N) Filter.atTop (Filter.liminf (λ n, 1 / Real.exp 1)) :=
sorry

end limit_w_div_N_eq_one_div_e_l281_281750


namespace min_value_pos_x_y_l281_281856

noncomputable def x (y : ℝ) := sorry
noncomputable def y (x : ℝ) := sorry

theorem min_value_pos_x_y (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_eq : x^2 + 3*x*y - 2 = 0) : x + y ≥ 4 / 3 := 
sorry

end min_value_pos_x_y_l281_281856


namespace reciprocal_of_sum_l281_281606

-- Define the fractions
def a := (1: ℚ) / 2
def b := (1: ℚ) / 3

-- Define their sum
def c := a + b

-- Define the expected reciprocal
def reciprocal := (6: ℚ) / 5

-- The theorem we want to prove:
theorem reciprocal_of_sum : (c⁻¹ = reciprocal) :=
by 
  sorry

end reciprocal_of_sum_l281_281606


namespace lowest_degree_required_l281_281271

noncomputable def smallest_degree_poly (b : ℤ) : ℕ :=
  if h : ∃ P : Polynomial ℝ, (∀ x, (P.eval x ≠ b)) ∧
    (∃ y, (P.eval y > b)) ∧ (∃ z, (P.eval z < b)) 
  then Nat.find h 
  else 0

theorem lowest_degree_required :
  ∃ b : ℤ, smallest_degree_poly b = 4 :=
by
  -- b is some integer that fits the described conditions
  use 0
  sorry

end lowest_degree_required_l281_281271


namespace max_a5_a6_l281_281119

-- Definitions as per the conditions
def arithmetic_sequence (a : ℕ → ℝ) := ∃ (d : ℝ), ∀ n : ℕ, a (n + 1) = a n + d
def all_positive (a : ℕ → ℝ) := ∀ n : ℕ, a n > 0
def sum_first_ten_terms (a : ℕ → ℝ) := ∑ n in (finset.range 10), a n = 30 

-- The theorem we need to prove
theorem max_a5_a6 (a : ℕ → ℝ) (h_arith : arithmetic_sequence a) (h_pos : all_positive a) (h_sum : sum_first_ten_terms a) :
  (∃ max_val, max_val = 9 ∧ ∀ b5 b6, b5 + b6 = a 5 + a 6 → b5 * b6 ≤ max_val) :=
  sorry

end max_a5_a6_l281_281119


namespace foci_distance_l281_281775

variable (x y : ℝ)

def ellipse_eq : Prop := (x^2 / 45) + (y^2 / 5) = 9

theorem foci_distance : ellipse_eq x y → (distance_between_foci : ℝ) = 12 * Real.sqrt 10 :=
by
  sorry

end foci_distance_l281_281775


namespace distance_between_foci_of_ellipse_l281_281780

-- Definitions based on conditions in the problem
def ellipse_eq (x y : ℝ) : Prop := (x^2) / 45 + (y^2) / 5 = 9

-- Proof statement
theorem distance_between_foci_of_ellipse : 
  ∀ (x y : ℝ), ellipse_eq x y → 2 * (√ (405 - 45)) = 12 * (√ 10) :=
by 
  sorry

end distance_between_foci_of_ellipse_l281_281780


namespace lottery_ticket_random_event_l281_281990

-- Define the type of possible outcomes of buying a lottery ticket
inductive LotteryOutcome
| Win
| Lose

-- Define the random event condition
def is_random_event (outcome: LotteryOutcome) : Prop :=
  match outcome with
  | LotteryOutcome.Win => True
  | LotteryOutcome.Lose => True

-- The theorem to prove that buying 1 lottery ticket and winning is a random event
theorem lottery_ticket_random_event : is_random_event LotteryOutcome.Win :=
by
  sorry

end lottery_ticket_random_event_l281_281990


namespace cyclic_pentagon_quadrilaterals_l281_281121

noncomputable def isCyclic (A B C D : ℝ × ℝ) : Prop :=
  -- Placeholder definition of cyclic quadrilateral, to be replaced with appropriate geometry definition.
  sorry

/-- In a convex pentagon \(ABCDE\) with intersections of pairs of diagonals forming points \(A_1, B_1, C_1, D_1, E_1\),
if four of the quadrilaterals \(AB_1A_1B\), \(BC_1B_1C\), \(CD_1C_1D\), \(DE_1D_1E\), and \(EA_1E_1A\) are cyclic, then the fifth one is also cyclic. -/
theorem cyclic_pentagon_quadrilaterals
  (A B C D E A1 B1 C1 D1 E1 : ℝ × ℝ)
  (h1: A1 = intersection (BD, CE))
  (h2: B1 = intersection (CE, DA))
  (h3: C1 = intersection (DA, EB))
  (h4: D1 = intersection (EB, AC))
  (h5: E1 = intersection (AC, BD))
  (cyclic_ABA1B1: isCyclic A B A1 B1)
  (cyclic_BCB1C1: isCyclic B C B1 C1)
  (cyclic_CDC1D1: isCyclic C D C1 D1)
  (cyclic_DED1E1: isCyclic D E D1 E1):
  isCyclic E A E1 A1 :=
sorry

end cyclic_pentagon_quadrilaterals_l281_281121


namespace inequality_solution_sets_l281_281453

-- Definition for the conditions and the required theorem
theorem inequality_solution_sets (m : ℝ) (a: ℝ) (x : ℝ) (y : ℝ) 
  (h1 : 0 < m)
  (h2 : ∀ x, (x + 1/2) ≤ 2 * m → -2 ≤ x ∧ x ≤ 2)
  (h3 : ∀ y, ∀ x, f(x) ≤ 2 + (a / 2^y) + |2 * x + 3|) :
  m = 3 / 4 ∧ a ≤ 4 :=
by
  sorry

end inequality_solution_sets_l281_281453


namespace sum_palindromic_primes_l281_281950

/-- Definition of a palindromic prime -/
def is_palindromic_prime (p : ℕ) : Prop :=
  p < 200 ∧ p ≥ 100 ∧ p.isPrime ∧ (Nat.reverseDigits p).isPrime

/-- Sum of all palindromic primes less than 200 -/
theorem sum_palindromic_primes : 
  ∑ p in (Finset.filter is_palindromic_prime (Finset.range 200)).val, p = 1299 :=
sorry

end sum_palindromic_primes_l281_281950


namespace price_reduction_l281_281343

theorem price_reduction (P : ℝ) :
  let P1 := P * 0.88,
      P2 := P1 * 0.9,
      P3 := P2 * 0.92,
      P4 := P3 * 1.05 in
  P4 = P * 0.765072 :=
by
  sorry

end price_reduction_l281_281343


namespace num_common_points_l281_281458

noncomputable def curve (x : ℝ) : ℝ := 3 * x ^ 4 - 2 * x ^ 3 - 9 * x ^ 2 + 4

noncomputable def tangent_line (x : ℝ) : ℝ :=
  -12 * (x - 1) - 4

theorem num_common_points :
  ∃ (x1 x2 x3 : ℝ), curve x1 = tangent_line x1 ∧
                    curve x2 = tangent_line x2 ∧
                    curve x3 = tangent_line x3 ∧
                    (x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3) :=
sorry

end num_common_points_l281_281458


namespace complex_abs_value_example_l281_281002

def abs_sq (z : ℂ) : ℝ := z.re^2 + z.im^2

theorem complex_abs_value_example :
  abs ((5 : ℂ) + (real.sqrt 11 : ℂ) * I)^4 = 1296 :=
by
  let z := (5 : ℂ) + (real.sqrt 11 : ℂ) * I
  have h1 : abs z = real.sqrt (abs_sq z),
  sorry
  have h2 : abs z = 6,
  sorry
  have h3 : abs (z^4) = (abs z)^4,
  sorry
  show abs z ^ 4 = 1296,
  sorry

end complex_abs_value_example_l281_281002


namespace count_integer_pairs_l281_281508

theorem count_integer_pairs (s : Finset ℤ) :
  s = Finset.Icc (-100 : ℤ) 100 →
  (Finset.card {p : ℤ × ℤ | p.1 ∈ s ∧ p.2 ∈ s ∧ p.1 < p.2 ∧ p.1 + p.2 > p.1 * p.2} = 10199) :=
by
  intro hs
  rw hs
  sorry

end count_integer_pairs_l281_281508
