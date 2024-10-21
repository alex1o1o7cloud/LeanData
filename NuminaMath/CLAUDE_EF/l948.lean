import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_real_part_sum_l948_94829

def polynomial_zeros (n : ℕ) (r : ℝ) : ℂ → Prop :=
  λ z ↦ z^n = r^n

def w_selection (z : ℂ) : Set ℂ :=
  {z, Complex.I * z}

theorem max_real_part_sum (n : ℕ) (r : ℝ) :
  ∃ (zs : Fin n → ℂ) (ws : Fin n → ℂ),
    (∀ j, polynomial_zeros n r (zs j)) ∧
    (∀ j, ws j ∈ w_selection (zs j)) ∧
    (∀ (vs : Fin n → ℂ), (∀ j, vs j ∈ w_selection (zs j)) →
      (Finset.sum Finset.univ (λ j ↦ (ws j).re) : ℝ) ≥ 
      (Finset.sum Finset.univ (λ j ↦ (vs j).re) : ℝ)) ∧
    (Finset.sum Finset.univ (λ j ↦ (ws j).re) : ℝ) = 
      16 * (1 + 2 * Real.cos (π / 8) + 2 * Real.cos (π / 4) + 2 * Real.cos (3 * π / 8)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_real_part_sum_l948_94829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_frustum_volume_ratio_l948_94818

/-- Represents a square pyramid -/
structure SquarePyramid where
  baseEdge : ℝ
  altitude : ℝ

/-- Calculates the volume of a square pyramid -/
noncomputable def pyramidVolume (p : SquarePyramid) : ℝ :=
  (1 / 3) * p.baseEdge ^ 2 * p.altitude

/-- Represents a frustum of a square pyramid -/
structure PyramidFrustum where
  originalPyramid : SquarePyramid
  cutRatio : ℝ

/-- Calculates the volume of a pyramid frustum -/
noncomputable def frustumVolume (f : PyramidFrustum) : ℝ :=
  let originalVolume := pyramidVolume f.originalPyramid
  let cutVolume := (f.cutRatio ^ 3) * originalVolume
  originalVolume - cutVolume

theorem pyramid_frustum_volume_ratio (p : SquarePyramid) (f : PyramidFrustum) :
  p.baseEdge = 64 ∧ p.altitude = 18 ∧ f.originalPyramid = p ∧ f.cutRatio = 1/3 →
  frustumVolume f / pyramidVolume p = 263 / 272 := by
  sorry

#eval "Compilation successful!"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_frustum_volume_ratio_l948_94818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_when_tan_is_two_l948_94830

theorem sin_double_angle_when_tan_is_two (α : Real) (h : Real.tan α = 2) : Real.sin (2 * α) = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_when_tan_is_two_l948_94830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incenter_distance_to_perpendicular_l948_94878

/-- A right triangle with sides 6, 8, and 10, its incenter, and a perpendicular line -/
structure RightTriangleWithIncenter where
  -- The vertices of the triangle
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  -- The incenter of the triangle
  I : ℝ × ℝ
  -- The point where BI intersects AC
  K : ℝ × ℝ
  -- The foot of the perpendicular from K to AB
  H : ℝ × ℝ
  -- Conditions
  right_angle : (C.1 - A.1) * (B.1 - A.1) + (C.2 - A.2) * (B.2 - A.2) = 0
  side_BC : Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) = 6
  side_CA : Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2) = 8
  side_AB : Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 10
  I_is_incenter : ∀ (P : ℝ × ℝ), Real.sqrt ((P.1 - I.1)^2 + (P.2 - I.2)^2) = 2 ↔
    ((P.1 - A.1) * (B.2 - A.2) = (P.2 - A.2) * (B.1 - A.1) ∨
     (P.1 - B.1) * (C.2 - B.2) = (P.2 - B.2) * (C.1 - B.1) ∨
     (P.1 - C.1) * (A.2 - C.2) = (P.2 - C.2) * (A.1 - C.1))
  K_on_AC : (K.1 - A.1) * (C.2 - A.2) = (K.2 - A.2) * (C.1 - A.1)
  K_on_BI : (K.1 - B.1) * (I.2 - B.2) = (K.2 - B.2) * (I.1 - B.1)
  KH_perpendicular_AB : (H.1 - A.1) * (B.1 - A.1) + (H.2 - A.2) * (B.2 - A.2) = 0

/-- The distance from the incenter to the perpendicular line KH is 2 -/
theorem incenter_distance_to_perpendicular (t : RightTriangleWithIncenter) : 
  Real.sqrt ((t.I.1 - t.H.1)^2 + (t.I.2 - t.H.2)^2) = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_incenter_distance_to_perpendicular_l948_94878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_knight_return_even_moves_l948_94890

/-- Represents a position on a chessboard -/
structure Position :=
  (x : ℤ) (y : ℤ)

/-- Represents a knight's move on a chessboard -/
def knight_move (p : Position) : Position :=
  let dx := [1, 2, 2, 1, -1, -2, -2, -1]
  let dy := [2, 1, -1, -2, -2, -1, 1, 2]
  ⟨p.x + dx[0], p.y + dy[0]⟩  -- We only need to define one possible move

/-- Predicate to check if a position is the same as the starting position -/
def is_start_position (start : Position) (current : Position) : Prop :=
  start.x = current.x ∧ start.y = current.y

/-- Theorem stating that if a knight returns to its starting position after n moves, n is even -/
theorem knight_return_even_moves (start : Position) (n : ℕ) :
  (∃ (path : ℕ → Position), 
    path 0 = start ∧
    (∀ i : ℕ, i < n → path (i + 1) = knight_move (path i)) ∧
    is_start_position start (path n)) →
  Even n :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_knight_return_even_moves_l948_94890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_society_theorem_l948_94843

/-- A society with friendship and enmity relations -/
structure Society where
  members : Finset ℕ
  friends : Finset (ℕ × ℕ)
  enemies : Finset (ℕ × ℕ)

/-- Conditions for the society -/
def SocietyConditions (s : Society) (n : ℕ) (q : ℕ) : Prop :=
  -- There are n members
  (s.members.card = n) ∧
  -- Every two members are either friends or enemies
  (∀ x y, x ∈ s.members → y ∈ s.members → x ≠ y → ((x, y) ∈ s.friends ∨ (x, y) ∈ s.enemies)) ∧
  -- There are exactly q pairs of friends
  (s.friends.card = q) ∧
  -- In every set of three persons, there are two who are enemies
  (∀ x y z, x ∈ s.members → y ∈ s.members → z ∈ s.members → 
    x ≠ y ∧ y ≠ z ∧ x ≠ z →
    ((x, y) ∈ s.enemies ∨ (y, z) ∈ s.enemies ∨ (x, z) ∈ s.enemies))

/-- The number of friend pairs among a member's enemies -/
def EnemyFriendPairs (s : Society) (x : ℕ) : ℕ :=
  (s.friends.filter (λ p => (x, p.1) ∈ s.enemies ∧ (x, p.2) ∈ s.enemies)).card

/-- The main theorem -/
theorem society_theorem (s : Society) (n : ℕ) (q : ℕ) (h : SocietyConditions s n q) :
  ∃ x ∈ s.members, EnemyFriendPairs s x ≤ q * (1 - 4 * q / n^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_society_theorem_l948_94843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_equivalence_l948_94820

noncomputable def f (x : ℝ) : ℝ := 3 * x + Real.sin x

theorem inequality_equivalence (m : ℝ) : 
  (f (2 * m - 1) + f (3 - m) > 0) ↔ (m > -2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_equivalence_l948_94820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_integer_solutions_l948_94815

theorem infinite_integer_solutions
  (a b c k : ℤ)
  (D : ℤ)
  (h_D : D = b^2 - 4*a*c)
  (h_D_pos : D > 0)
  (h_D_nonsquare : ∀ m : ℤ, m^2 ≠ D)
  (h_k_nonzero : k ≠ 0)
  (x0 y0 : ℤ)
  (h_solution : a*x0^2 + b*x0*y0 + c*y0^2 = k) :
  ∃ S : Set (ℤ × ℤ), (Infinite S) ∧
    (∀ (x y : ℤ), (x, y) ∈ S → a*x^2 + b*x*y + c*y^2 = k) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_integer_solutions_l948_94815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exp_inequality_implies_log_inequality_l948_94893

theorem exp_inequality_implies_log_inequality (x y : ℝ) :
  (2:ℝ)^x - (2:ℝ)^y < (3:ℝ)^(-x) - (3:ℝ)^(-y) → Real.log (y - x + 1) > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exp_inequality_implies_log_inequality_l948_94893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_sum_l948_94813

theorem cubic_polynomial_sum (w : ℂ) :
  let Q (z : ℂ) := z^3 + p*z^2 + q*z + r
  ∀ (p q r : ℝ),
    (∃ (z₁ z₂ z₃ : ℂ), z₁ = w + 5*Complex.I ∧ z₂ = w + 15*Complex.I ∧ z₃ = 3*w - 6 ∧
      ∀ (z : ℂ), Q z = 0 ↔ z = z₁ ∨ z = z₂ ∨ z = z₃) →
    p + q + r = -1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_sum_l948_94813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplification_and_evaluation_l948_94873

theorem simplification_and_evaluation :
  (∀ x y : ℝ, x - (2*x - y) + (3*x - 2*y) = 2*x - y) ∧
  (let x : ℝ := -2/3;
   let y : ℝ := 3/2;
   2*x*y + (-3*x^3 + 5*x*y + 2) - 3*(2*x*y - x^3 + 1) = -2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplification_and_evaluation_l948_94873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_expression_l948_94812

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

noncomputable def f : ℝ → ℝ := sorry

axiom f_odd : is_odd_function f
axiom f_positive : ∀ x > 0, f x = 3^x + 1

theorem f_expression :
  ∀ x, f x = if x > 0 then 3^x + 1
            else if x = 0 then 0
            else -3^(-x) - 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_expression_l948_94812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_problem_l948_94802

theorem ratio_problem (second_part : ℝ) (percent : ℝ) 
  (h1 : second_part = 3)
  (h2 : percent = 200) :
  (percent / 100) * second_part = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_problem_l948_94802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l948_94862

-- Define the parabola C: y² = 2px (p > 0)
noncomputable def C (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

-- Define the focus F
noncomputable def F (p : ℝ) : ℝ × ℝ := (p/2, 0)

-- Define the directrix
noncomputable def directrix (p : ℝ) : ℝ → Prop := λ x => x = -p/2

-- Define the line perpendicular to x-axis passing through F
noncomputable def perp_line (p : ℝ) (x : ℝ) : Prop := x = p/2

-- Define the tangent line
noncomputable def tangent_line (x y : ℝ) : Prop := y = x + 1

theorem parabola_properties (p : ℝ) :
  C p (F p).1 (F p).2 ∧
  (∃ x, directrix p x) ∧
  (∃ m n : ℝ × ℝ, C p m.1 m.2 ∧ C p n.1 n.2 ∧ perp_line p m.1 ∧ perp_line p n.1 ∧ 
    Real.sqrt ((m.1 - n.1)^2 + (m.2 - n.2)^2) = 4) ∧
  (∃ x y : ℝ, C p x y ∧ tangent_line x y) →
  F p = (1, 0) ∧
  directrix p (-1) ∧
  (∃ m n : ℝ × ℝ, C p m.1 m.2 ∧ C p n.1 n.2 ∧ perp_line p m.1 ∧ perp_line p n.1 ∧ 
    Real.sqrt ((m.1 - n.1)^2 + (m.2 - n.2)^2) = 4) ∧
  (∃ x y : ℝ, C p x y ∧ tangent_line x y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l948_94862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_difference_l948_94807

/-- The function g(x) = 8^x -/
noncomputable def g (x : ℝ) : ℝ := 8^x

/-- Theorem: For the function g(x) = 8^x, g(x+1) - g(x) = 7g(x) for all real x -/
theorem g_difference (x : ℝ) : g (x + 1) - g x = 7 * g x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_difference_l948_94807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_real_roots_l948_94863

-- Define the polynomial f(x) with parameter p
def f (p : ℝ) (x : ℝ) : ℝ := x^4 + 6*p*x^3 + 3*x^2 + 6*p*x + 9

-- Define the set of p values
def P : Set ℝ := {p | p < -Real.sqrt (1/3) ∨ p > Real.sqrt (1/3)}

-- Theorem statement
theorem all_real_roots (p : ℝ) : 
  (∀ z : ℂ, Complex.re (f p (Complex.re z)) = 0 ∧ Complex.im (f p (Complex.re z)) = 0 → z.im = 0) ↔ p ∈ P := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_real_roots_l948_94863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_divisible_pair_l948_94851

-- Define the set of numbers from 1 to 200
def S : Set Nat := {n | 1 ≤ n ∧ n ≤ 200}

-- Define a property for a subset of S
def ValidSubset (T : Set Nat) : Prop :=
  T ⊆ S ∧ Finite T ∧ Nat.card T = 100 ∧ ∃ x ∈ T, x < 16

-- Define the divisibility relation
def Divides (a b : Nat) : Prop := ∃ k, b = a * k

-- The main theorem
theorem exists_divisible_pair (T : Set Nat) (h : ValidSubset T) :
  ∃ a b, a ∈ T ∧ b ∈ T ∧ a ≠ b ∧ Divides a b :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_divisible_pair_l948_94851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_proportions_l948_94803

/-- A function f(x) = x^(m^2) - 2 where m is a real number -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x^(m^2) - 2

/-- Definition of a direct proportion function -/
def is_direct_proportion (g : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, g x = k * x

/-- Definition of an inverse proportion function -/
def is_inverse_proportion (g : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, x ≠ 0 → g x = k / x

/-- Definition of a power function -/
def is_power_function (g : ℝ → ℝ) : Prop :=
  ∃ k n : ℝ, ∀ x : ℝ, g x = k * x^n

theorem f_proportions (m : ℝ) :
  (is_direct_proportion (f m) → m = 1 ∨ m = -1) ∧
  (is_inverse_proportion (f m) → m = -1) ∧
  (is_power_function (f m) → m = 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_proportions_l948_94803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_parabola_l948_94852

/-- The circle defined by the equation x^2 + y^2 - 12x + 31 = 0 -/
def Circle : Set (ℝ × ℝ) :=
  {p | p.1^2 + p.2^2 - 12*p.1 + 31 = 0}

/-- The parabola defined by the equation y^2 = 4x -/
def Parabola : Set (ℝ × ℝ) :=
  {p | p.2^2 = 4*p.1}

/-- The distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The theorem stating the minimum distance between the circle and parabola -/
theorem min_distance_circle_parabola :
  ∃ (d : ℝ), d = Real.sqrt 5 ∧
  ∀ (a b : ℝ × ℝ), a ∈ Circle → b ∈ Parabola →
  distance a b ≥ d := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_parabola_l948_94852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pure_imaginary_z_z_in_third_quadrant_l948_94826

def z (a : ℝ) : ℂ := (1 - Complex.I) * a^2 - 3 * a + 2 + Complex.I

theorem pure_imaginary_z :
  ∀ a : ℝ, (z a).re = 0 → (z a).im ≠ 0 → z a = -3 * Complex.I :=
by sorry

theorem z_in_third_quadrant :
  ∀ a : ℝ, (z a).re < 0 ∧ (z a).im < 0 → 1 < a ∧ a < 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pure_imaginary_z_z_in_third_quadrant_l948_94826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_is_39_seconds_l948_94886

/-- The time taken for a train to pass a jogger -/
noncomputable def train_passing_time (jogger_speed : ℝ) (train_speed : ℝ) (train_length : ℝ) (initial_distance : ℝ) : ℝ :=
  let jogger_speed_ms := jogger_speed * (1000 / 3600)
  let train_speed_ms := train_speed * (1000 / 3600)
  let relative_speed := train_speed_ms - jogger_speed_ms
  let total_distance := initial_distance + train_length
  total_distance / relative_speed

/-- Theorem stating that the time taken for the train to pass the jogger is 39 seconds -/
theorem train_passing_time_is_39_seconds :
  train_passing_time 9 45 120 270 = 39 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_is_39_seconds_l948_94886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xy_circle_radius_is_sqrt_28_l948_94881

noncomputable def sphere_center : ℝ × ℝ × ℝ := (3, 7, 5)

noncomputable def xz_circle_center : ℝ × ℝ × ℝ := (3, 0, 5)
noncomputable def xz_circle_radius : ℝ := 2

noncomputable def xy_circle_center : ℝ × ℝ × ℝ := (3, 7, 0)

noncomputable def sphere_radius : ℝ := Real.sqrt 53

theorem xy_circle_radius_is_sqrt_28 :
  let r : ℝ := Real.sqrt (sphere_radius^2 - 5^2)
  r = Real.sqrt 28 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_xy_circle_radius_is_sqrt_28_l948_94881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_theorem_l948_94871

/-- Represents an ellipse with given foci and eccentricity -/
structure Ellipse where
  foci : ℝ × ℝ × ℝ × ℝ -- (x₁, y₁, x₂, y₂) of F₁ and F₂
  eccentricity : ℝ

/-- Represents a line in the form y = mx + b -/
structure Line where
  m : ℝ
  b : ℝ

/-- The distance between two points -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

theorem ellipse_intersection_theorem (e : Ellipse) (l : Line) :
  e.foci = (-Real.sqrt 3, 0, Real.sqrt 3, 0) →
  e.eccentricity = Real.sqrt 3 / 2 →
  l.m = 1 / 2 →
  (∃ P Q : ℝ × ℝ, P ≠ Q ∧ 
    (P.1^2 / 4 + P.2^2 = 1) ∧ 
    (Q.1^2 / 4 + Q.2^2 = 1) ∧
    P.2 = l.m * P.1 + l.b ∧
    Q.2 = l.m * Q.1 + l.b ∧
    distance P.1 P.2 Q.1 Q.2 = 2) →
  l.b = Real.sqrt 30 / 5 ∨ l.b = -Real.sqrt 30 / 5 := by
  sorry

#check ellipse_intersection_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_theorem_l948_94871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_weight_of_children_l948_94819

theorem average_weight_of_children (num_boys num_girls : ℕ) 
  (avg_weight_boys avg_weight_girls : ℚ) :
  num_boys = 8 →
  num_girls = 3 →
  avg_weight_boys = 160 →
  avg_weight_girls = 110 →
  (num_boys * avg_weight_boys + num_girls * avg_weight_girls) / (num_boys + num_girls : ℚ) = 146 :=
by
  intro h_boys h_girls h_avg_boys h_avg_girls
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_weight_of_children_l948_94819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_pi_squared_abs_irrational_l948_94834

theorem nine_pi_squared_abs_irrational : ¬ (∃ (q : ℚ), |9 * Real.pi ^ 2| = (q : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_pi_squared_abs_irrational_l948_94834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l948_94896

/-- The parabola equation y^2 = 12x --/
def parabola (x y : ℝ) : Prop := y^2 = 12 * x

/-- The circle equation x^2 + y^2 - 4x - 6y + 3 = 0 --/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 6*y + 3 = 0

/-- The distance between two points in 2D space --/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

/-- Theorem: The distance between the intersection points of the parabola and circle is 4 --/
theorem intersection_distance : ∃ (x₁ y₁ x₂ y₂ : ℝ),
  parabola x₁ y₁ ∧ circle_eq x₁ y₁ ∧
  parabola x₂ y₂ ∧ circle_eq x₂ y₂ ∧
  (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧
  distance x₁ y₁ x₂ y₂ = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l948_94896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_intersection_eq_open_interval_l948_94864

-- Define sets A and B
def A : Set ℝ := {x | x ≤ 4}
def B : Set ℝ := {x | ∃ y, y = -x^2}

-- Theorem statement
theorem complement_intersection_eq_open_interval :
  (Set.univ : Set ℝ) \ (A ∩ B) = Set.Ioi 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_intersection_eq_open_interval_l948_94864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_chord_length_l948_94897

-- Define the circle
def circle_equation (x y : ℝ) : Prop := (x - 1)^2 + (y + 2)^2 = 20

-- Define the x-axis
def x_axis (y : ℝ) : Prop := y = 0

-- Define the chord length
def chord_length (x₁ x₂ : ℝ) : ℝ := |x₁ - x₂|

-- Theorem statement
theorem circle_chord_length :
  ∃ x₁ x₂ : ℝ,
    circle_equation x₁ 0 ∧
    circle_equation x₂ 0 ∧
    x_axis 0 ∧
    chord_length x₁ x₂ = 8 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_chord_length_l948_94897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l948_94876

noncomputable section

/-- The function f(x) as defined in the problem -/
def f (x : ℝ) : ℝ := Real.sin x ^ 2 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x + 
  Real.sin (x + Real.pi/4) * Real.sin (x - Real.pi/4)

/-- The smallest positive period of f(x) -/
def smallest_period : ℝ := Real.pi

/-- The lower bound of the range of f(x) -/
def range_lower : ℝ := -3/2

/-- The upper bound of the range of f(x) -/
def range_upper : ℝ := 5/2

theorem f_properties :
  (∀ x : ℝ, f (x + smallest_period) = f x) ∧ 
  (∀ x : ℝ, range_lower ≤ f x ∧ f x ≤ range_upper) ∧
  (∀ x₀ : ℝ, x₀ ∈ Set.Icc 0 (Real.pi/2) → f x₀ = 0 → 
    Real.sin (2 * x₀) = (Real.sqrt 15 - Real.sqrt 3) / 8) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l948_94876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l948_94855

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Theorem stating the properties of the given triangle -/
theorem triangle_properties (t : Triangle) 
  (h1 : t.b = 1) 
  (h2 : 2 * Real.cos t.C - 2 * t.a - t.c = 0) : 
  t.B = 2 * Real.pi / 3 ∧ 
  Real.sqrt ((1 / Real.sqrt 3)^2 - (t.b/2)^2) = Real.sqrt 3 / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l948_94855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_beginner_trig_probability_l948_94806

/-- Represents the number of students in calculus courses -/
def C : ℕ → ℕ := id

/-- The total number of students -/
def T (c : ℕ) : ℕ := 5 * c / 2

/-- The number of students in trigonometry courses -/
def trig_students (c : ℕ) : ℕ := 3 * c / 2

/-- The number of students in beginner calculus -/
def beginner_calc (c : ℕ) : ℕ := 7 * c / 10

/-- The number of students in beginner courses -/
def beginner_total (c : ℕ) : ℕ := 4 * T c / 5

/-- The number of students in beginner trigonometry -/
def beginner_trig (c : ℕ) : ℕ := beginner_total c - beginner_calc c

theorem beginner_trig_probability (c : ℕ) : 
  (beginner_trig c : ℚ) / (T c) = 13 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_beginner_trig_probability_l948_94806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mother_sold_150_rings_l948_94894

/-- Calculates the number of rings sold by the mother given the initial conditions --/
def rings_sold_by_mother (initial_purchase : ℕ) (mother_purchase : ℕ) (final_stock : ℕ) : ℕ :=
  let initial_stock := initial_purchase / 2
  let total_stock := initial_purchase + initial_stock
  let sold_before_mother := (3 * total_stock) / 4
  let remaining_before_mother := total_stock - sold_before_mother
  let new_total_stock := remaining_before_mother + mother_purchase
  new_total_stock - final_stock

/-- Theorem stating that given the conditions in the problem, the mother sold 150 rings --/
theorem mother_sold_150_rings :
  rings_sold_by_mother 200 300 225 = 150 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mother_sold_150_rings_l948_94894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_equation_solution_l948_94849

theorem cubic_equation_solution (x : ℝ) :
  (x ^ (1/3) + (30 - x) ^ (1/3) = 3) →
  (x = ((9 + Real.sqrt 93) / 6) ^ 3 ∨
   x = ((9 - Real.sqrt 93) / 6) ^ 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_equation_solution_l948_94849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vector_scalar_l948_94836

/-- Given vectors a and b in ℝ², prove that if there exists a real number lambda such that (a - lambda*b) ⊥ b, then lambda = 6/5 -/
theorem perpendicular_vector_scalar (a b : ℝ × ℝ) (h : a = (2, -7) ∧ b = (-2, -4)) :
  ∃ lambda : ℝ, (a.1 - lambda * b.1, a.2 - lambda * b.2) • b = 0 → lambda = 6/5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vector_scalar_l948_94836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_l948_94888

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- The slope of the asymptote of a hyperbola -/
noncomputable def asymptote_slope (h : Hyperbola) : ℝ :=
  h.b / h.a

/-- Theorem stating the range of eccentricity for a hyperbola 
    given specific conditions -/
theorem eccentricity_range (h : Hyperbola) 
  (h_slope : asymptote_slope h < 2)
  (h_intersect : ∃ (x y : ℝ), x > h.a ∧ 2 * (x - h.a) = y ∧ x^2 / h.a^2 - y^2 / h.b^2 = 1) :
  1 < eccentricity h ∧ eccentricity h < Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_l948_94888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_four_digit_number_with_special_property_l948_94800

theorem no_four_digit_number_with_special_property : 
  ¬ ∃ (n : ℕ), 
    (1000 ≤ n ∧ n ≤ 9999) ∧ 
    (let digits := [n / 1000, (n / 100) % 10, (n / 10) % 10, n % 10];
     (List.sum digits) * 25 = List.prod digits) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_four_digit_number_with_special_property_l948_94800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_zero_in_interval_l948_94842

noncomputable def f (x : ℝ) := (1/3) * x^3 - 2*x - 2

theorem f_has_zero_in_interval :
  ∃ c ∈ Set.Ioo 2 3, f c = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_zero_in_interval_l948_94842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_complex_fraction_l948_94874

theorem imaginary_part_of_complex_fraction :
  let z : ℂ := (1 + 4*Complex.I) / (1 - Complex.I)
  Complex.im z = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_complex_fraction_l948_94874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equiv_g_l948_94895

noncomputable section

-- Define the functions
def f (x : ℝ) : ℝ := x^2 / x
def g (x : ℝ) : ℝ := x

-- State the theorem
theorem f_equiv_g : ∀ (x : ℝ), x ≠ 0 → f x = g x := by
  intro x hx
  simp [f, g]
  field_simp [hx]
  ring

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equiv_g_l948_94895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hybrid_car_trip_theorem_l948_94867

/-- Represents the cost and distance details of a hybrid car trip -/
structure HybridCarTrip where
  fuel_cost : ℝ
  electricity_cost : ℝ
  fuel_electricity_diff : ℝ
  max_total_cost : ℝ

/-- Calculates the cost per kilometer for electricity -/
noncomputable def electricity_cost_per_km (trip : HybridCarTrip) : ℝ :=
  trip.electricity_cost / (trip.electricity_cost / (trip.fuel_cost / (trip.fuel_electricity_diff + trip.electricity_cost / (trip.fuel_cost / trip.fuel_electricity_diff))))

/-- Calculates the minimum kilometers traveled using electricity -/
noncomputable def min_km_electricity (trip : HybridCarTrip) (cost_per_km : ℝ) : ℝ :=
  (trip.max_total_cost - trip.fuel_cost - trip.electricity_cost) / (cost_per_km + trip.fuel_electricity_diff - (trip.fuel_cost + trip.electricity_cost) / (trip.electricity_cost / cost_per_km))

theorem hybrid_car_trip_theorem (trip : HybridCarTrip) 
  (h1 : trip.fuel_cost = 76)
  (h2 : trip.electricity_cost = 26)
  (h3 : trip.fuel_electricity_diff = 0.5)
  (h4 : trip.max_total_cost = 39) :
  electricity_cost_per_km trip = 0.26 ∧ 
  min_km_electricity trip (electricity_cost_per_km trip) = 74 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hybrid_car_trip_theorem_l948_94867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cistern_width_is_four_l948_94831

/-- Calculates the width of a cistern given its length, water depth, and total wet surface area. -/
noncomputable def cistern_width (length : ℝ) (depth : ℝ) (wet_surface_area : ℝ) : ℝ :=
  (wet_surface_area - 2 * depth * length) / (length + 2 * depth)

/-- Theorem stating that a cistern with given dimensions has a width of 4 meters. -/
theorem cistern_width_is_four :
  cistern_width 12 1.25 88 = 4 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval cistern_width 12 1.25 88

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cistern_width_is_four_l948_94831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trisha_cookie_count_l948_94879

/-- Represents the shape of a cookie -/
inductive CookieShape
  | Trapezoid (base1 : ℚ) (base2 : ℚ) (height : ℚ)
  | RightTriangle (leg1 : ℚ) (leg2 : ℚ)

/-- Calculates the area of a cookie based on its shape -/
def cookieArea (shape : CookieShape) : ℚ :=
  match shape with
  | CookieShape.Trapezoid b1 b2 h => (b1 + b2) * h / 2
  | CookieShape.RightTriangle l1 l2 => l1 * l2 / 2

/-- Represents a batch of cookies -/
structure CookieBatch where
  shape : CookieShape
  count : ℕ

/-- Calculates the total area of dough used in a batch -/
def batchArea (batch : CookieBatch) : ℚ :=
  (cookieArea batch.shape) * batch.count

theorem trisha_cookie_count (artBatch trishaBatch : CookieBatch) :
  artBatch.shape = CookieShape.Trapezoid 4 6 4 →
  artBatch.count = 10 →
  trishaBatch.shape = CookieShape.RightTriangle 4 3 →
  batchArea artBatch = batchArea trishaBatch →
  trishaBatch.count = 33 := by
  sorry

#check trisha_cookie_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trisha_cookie_count_l948_94879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonically_increasing_interval_l948_94805

noncomputable def f (a b x : ℝ) : ℝ := a * Real.log x + b * x^2 + x

def has_extreme_values_at (f : ℝ → ℝ) (x₁ x₂ : ℝ) : Prop :=
  (deriv f) x₁ = 0 ∧ (deriv f) x₂ = 0

def monotonically_increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ (x y : ℝ), x ∈ s → y ∈ s → x < y → f x < f y

theorem f_monotonically_increasing_interval 
  (a b : ℝ) (h : has_extreme_values_at (f a b) 1 2) :
  monotonically_increasing_on (f a b) (Set.Ioo 1 2) := by
  sorry

#check f_monotonically_increasing_interval

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonically_increasing_interval_l948_94805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_six_point_graph_exists_l948_94823

/-- A planar graph with 6 vertices and 4 edges per vertex --/
structure SixPointGraph where
  -- The set of vertices
  V : Finset ℕ
  -- The set of edges
  E : Finset (ℕ × ℕ)
  -- There are exactly 6 vertices
  vertex_count : V.card = 6
  -- Each vertex has degree 4
  degree_four : ∀ v, v ∈ V → (E.filter (λ e => e.1 = v ∨ e.2 = v)).card = 4
  -- The graph is planar (no intersecting edges)
  planar : ∀ e1 e2, e1 ∈ E → e2 ∈ E → e1 ≠ e2 → e1.1 ≠ e2.1 ∧ e1.1 ≠ e2.2 ∧ e1.2 ≠ e2.1 ∧ e1.2 ≠ e2.2

/-- There exists a planar graph with 6 vertices, each having degree 4 --/
theorem six_point_graph_exists : ∃ g : SixPointGraph, True := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_six_point_graph_exists_l948_94823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_sum_inverse_squares_constant_l948_94892

/-- A parabola in the xy-plane -/
structure Parabola where
  f : ℝ → ℝ
  is_parabola : f = fun x ↦ x^2

/-- A point in the xy-plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A chord of a parabola -/
structure Chord (p : Parabola) where
  a : Point
  b : Point
  on_parabola_a : p.f a.x = a.y
  on_parabola_b : p.f b.x = b.y

/-- The distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- The theorem to be proved -/
theorem chord_sum_inverse_squares_constant (p : Parabola) (c : Point) 
    (h_c : c.x = 0 ∧ c.y = 1) :
  ∃ s : ℝ, ∀ ab : Chord p, 
    (ab.a.x - c.x) * (ab.b.y - c.y) = (ab.b.x - c.x) * (ab.a.y - c.y) →
    1 / (distance ab.a c)^2 + 1 / (distance ab.b c)^2 = s ∧ s = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_sum_inverse_squares_constant_l948_94892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_equal_volume_cut_l948_94899

/-- Theorem: For a quadrilateral pyramid with base sides a and b, side edge c,
    the distance x from the apex to cut the pyramid into equal volumes is
    given by x = (√(4c² - (a² + b²))) / (2 * ∛2) -/
theorem pyramid_equal_volume_cut (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  let x := Real.sqrt (4 * c^2 - (a^2 + b^2)) / (2 * (2 : ℝ)^(1/3))
  ∃ (V : ℝ → ℝ), (V x = V (Real.sqrt (4 * c^2 - (a^2 + b^2)) / 2 - x)) ∧
                 (∀ y, 0 < y ∧ y < Real.sqrt (4 * c^2 - (a^2 + b^2)) / 2 → 
                   V y = (1/3) * a * b * y * (Real.sqrt (4 * c^2 - (a^2 + b^2)) / 2 - y)^2 / 
                         (Real.sqrt (4 * c^2 - (a^2 + b^2)) / 2)^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_equal_volume_cut_l948_94899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_charlyn_visible_area_l948_94860

/-- The area of the region visible to Charlyn during her walk around a rectangle -/
noncomputable def visible_area (length width visible_distance : ℝ) : ℝ :=
  let inside_area := length * width - (length - 2 * visible_distance) * (width - 2 * visible_distance)
  let outside_rectangles_area := 2 * (length * visible_distance) + 2 * (width * visible_distance)
  let outside_circles_area := 4 * (Real.pi * visible_distance^2 / 4)
  inside_area + outside_rectangles_area + outside_circles_area

/-- Theorem stating the area of the region Charlyn can see -/
theorem charlyn_visible_area :
  visible_area 8 6 1.5 = 75 + (9 * Real.pi / 4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_charlyn_visible_area_l948_94860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_equivalence_l948_94885

theorem inequality_equivalence (x : ℝ) : 
  (1/5 : ℝ) * (5 : ℝ)^(2*x) * (7 : ℝ)^(3*x+2) ≤ (25/7 : ℝ) * (7 : ℝ)^(2*x) * (5 : ℝ)^(3*x) ↔ x ≤ -3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_equivalence_l948_94885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_has_most_axes_l948_94856

-- Define a type for the shapes
inductive Shape
  | IsoscelesTriangle
  | Square
  | Circle
  | LineSegment

-- Define a function to count the axes of symmetry
def axesOfSymmetry (s : Shape) : ℕ ⊕ Unit :=
  match s with
  | Shape.IsoscelesTriangle => Sum.inl 1
  | Shape.Square => Sum.inl 4
  | Shape.Circle => Sum.inr ()
  | Shape.LineSegment => Sum.inl 2

-- Define a relation for "has more axes of symmetry than"
def hasMoreAxes (s1 s2 : Shape) : Prop :=
  match axesOfSymmetry s1, axesOfSymmetry s2 with
  | Sum.inr (), Sum.inr () => True
  | Sum.inr (), Sum.inl _ => True
  | Sum.inl _, Sum.inr () => False
  | Sum.inl n1, Sum.inl n2 => n1 > n2

-- Theorem statement
theorem circle_has_most_axes :
  ∀ s : Shape, s ≠ Shape.Circle → hasMoreAxes Shape.Circle s := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_has_most_axes_l948_94856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_QPO_l948_94814

-- Define Point type if not already defined in Mathlib
def Point := ℝ × ℝ

-- Define IsRectangle as a predicate
def IsRectangle (A B C D : Point) : Prop := sorry

-- Define Rectangle structure
structure Rectangle (A B C D : Point) where
  is_rectangle : IsRectangle A B C D

-- Define Trisects as a predicate
def Trisects (P D B C N : Point) : Prop := sorry

-- Define Trisection structure
structure Trisection (P D B C N : Point) where
  trisects : Trisects P D B C N

-- Define Intersects as a predicate
def Intersects (D P C Q O : Point) : Prop := sorry

-- Define Intersection structure
structure Intersection (D P C Q O : Point) where
  intersect : Intersects D P C Q O

-- Define area function
noncomputable def area (shape : Set Point) : ℝ := sorry

-- Define Triangle type
structure Triangle (P Q O : Point) where
  mk :: -- Use :: for constructors in Lean 4

theorem area_of_triangle_QPO 
  (A B C D P Q O N M : Point) 
  (rect : Rectangle A B C D) 
  (trisect_BC : Trisection P D B C N)
  (trisect_AD : Trisection Q C A D M)
  (intersect : Intersection D P C Q O)
  (h_area : area {x | IsRectangle A B C D} = 360) :
  area {x | ∃ t : Triangle P Q O, true} = 140 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_QPO_l948_94814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_and_minimum_l948_94877

/-- Parabola with equation y^2 = 2px and focus F(4, 0) -/
structure Parabola where
  p : ℝ
  hp : p > 0

/-- Line passing through F(4, 0) and intersecting the parabola at M and N -/
structure IntersectingLine (para : Parabola) where
  M : ℝ × ℝ
  N : ℝ × ℝ
  hM : M.2^2 = 2 * para.p * M.1
  hN : N.2^2 = 2 * para.p * N.1
  hF : (M.1 - 4) * N.2 = (N.1 - 4) * M.2

/-- Distance between two points -/
noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

theorem parabola_focus_and_minimum (para : Parabola) :
  para.p = 8 ∧
  ∀ l : IntersectingLine para,
    (distance l.N (4, 0) / 9 - 4 / distance l.M (4, 0)) ≥ 1/3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_and_minimum_l948_94877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_distance_l948_94835

/-- The distance between two points on a circle --/
noncomputable def chord_length (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

/-- The equation of the circle (x-2)² + y² = 4 --/
def on_circle (x y : ℝ) : Prop :=
  (x - 2)^2 + y^2 = 4

/-- The equation of the line y = x - m --/
def on_line (x y m : ℝ) : Prop :=
  y = x - m

/-- The main theorem --/
theorem intersection_points_distance (m : ℝ) :
  (∃ x₁ y₁ x₂ y₂ : ℝ,
    on_circle x₁ y₁ ∧
    on_circle x₂ y₂ ∧
    on_line x₁ y₁ m ∧
    on_line x₂ y₂ m ∧
    chord_length x₁ y₁ x₂ y₂ = Real.sqrt 14) →
  m = 1 ∨ m = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_distance_l948_94835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l948_94810

noncomputable def area_triangle (a b c : ℝ) : ℝ := 
  Real.sqrt ((a + b + c) * (-a + b + c) * (a - b + c) * (a + b - c)) / 4

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  A + B + C = π ∧
  a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C ∧
  2 * (Real.tan A + Real.tan B) = Real.tan A / Real.cos B + Real.tan B / Real.cos A →
  (a + b) / c = 2 ∧
  (c = 2 ∧ C = π / 3 → area_triangle a b c = Real.sqrt 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l948_94810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_difference_magnitude_l948_94847

/-- Given vectors a and b in a real inner product space satisfying certain conditions,
    prove that the magnitude of their difference is √10. -/
theorem vector_difference_magnitude (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] (a b : V) 
  (ha : ‖a‖ = 3)
  (hb : ‖b‖ = 2)
  (hab : ‖a + b‖ = 4) :
  ‖a - b‖ = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_difference_magnitude_l948_94847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_color_graph_edge_bound_l948_94859

/-- A graph with two-colored edges and no monochromatic cycles of length 3, 4, or 5 -/
structure TwoColorGraph (n : ℕ) where
  vertices : Fin n
  edges : Finset (Fin n × Fin n)
  coloring : (Fin n × Fin n) → Bool
  symm : ∀ (i j : Fin n), (i, j) ∈ edges ↔ (j, i) ∈ edges
  no_mono_cycles : ∀ (k : Fin 3), ∀ (cycle : Fin (k + 3) → Fin n),
    (∀ (i : Fin (k + 3)), (cycle i, cycle (i + 1)) ∈ edges) →
    ¬(∀ (i : Fin (k + 3)), coloring (cycle i, cycle (i + 1)) = coloring (cycle 0, cycle 1))

/-- The main theorem -/
theorem two_color_graph_edge_bound {n : ℕ} (hn : n ≥ 5) (G : TwoColorGraph n) :
  G.edges.card < n^2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_color_graph_edge_bound_l948_94859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lamp_purchase_solution_l948_94884

/-- Represents the cost and quantity of lamps and flashlights -/
structure LampPurchase where
  desk_lamp_cost : ℝ
  flashlight_cost : ℝ
  desk_lamp_quantity : ℕ
  flashlight_quantity : ℕ

/-- Defines the conditions of the lamp purchase problem -/
def lamp_purchase_conditions (p : LampPurchase) : Prop :=
  p.desk_lamp_cost = p.flashlight_cost + 20 ∧
  p.desk_lamp_cost * p.desk_lamp_quantity = 400 ∧
  p.flashlight_cost * p.flashlight_quantity = 160 ∧
  (p.desk_lamp_quantity : ℝ) = (p.flashlight_quantity : ℝ) / 2

/-- Defines the promotional conditions for the second part of the problem -/
def promotional_conditions (p : LampPurchase) (max_desk_lamps : ℕ) : Prop :=
  (2 * (max_desk_lamps : ℝ) + 8 : ℝ) = (max_desk_lamps : ℝ) + ((max_desk_lamps : ℝ) + 8) ∧
  p.desk_lamp_cost * (max_desk_lamps : ℝ) + p.flashlight_cost * ((max_desk_lamps : ℝ) + 8) ≤ 670

/-- Theorem stating the solution to the lamp purchase problem -/
theorem lamp_purchase_solution :
  ∃ (p : LampPurchase) (max_desk_lamps : ℕ),
    lamp_purchase_conditions p ∧
    promotional_conditions p max_desk_lamps ∧
    p.desk_lamp_cost = 25 ∧
    p.flashlight_cost = 5 ∧
    max_desk_lamps = 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lamp_purchase_solution_l948_94884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_increasing_condition_l948_94840

/-- A geometric sequence with first term a and common ratio q -/
def geometric_sequence (a q : ℝ) : ℕ → ℝ := λ n ↦ a * q^(n - 1)

/-- A sequence is increasing -/
def is_increasing (s : ℕ → ℝ) : Prop := ∀ n : ℕ, s (n + 1) > s n

theorem geometric_increasing_condition (a q : ℝ) :
  (geometric_sequence a q 2 > geometric_sequence a q 1) →
  (is_increasing (geometric_sequence a q) ∧
  ¬(∀ a' q' : ℝ, (geometric_sequence a' q' 2 > geometric_sequence a' q' 1) →
    is_increasing (geometric_sequence a' q'))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_increasing_condition_l948_94840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_projection_area_is_one_l948_94808

/-- Represents a tetrahedron with specific properties -/
structure Tetrahedron where
  /-- The length of the hypotenuse of the isosceles right triangular faces -/
  hypotenuse : ℝ
  /-- The dihedral angle between the two adjacent isosceles right triangular faces -/
  dihedral_angle : ℝ

/-- Calculates the maximum projection area of the tetrahedron -/
noncomputable def max_projection_area (t : Tetrahedron) : ℝ :=
  (t.hypotenuse^2) / 4

/-- Theorem stating the maximum projection area for a specific tetrahedron -/
theorem max_projection_area_is_one (t : Tetrahedron) 
  (h1 : t.hypotenuse = 2)
  (h2 : t.dihedral_angle = Real.pi / 3) :
  max_projection_area t = 1 := by
  sorry

#check max_projection_area_is_one

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_projection_area_is_one_l948_94808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l948_94833

-- Define the ellipse C
noncomputable def ellipse_C (x y : ℝ) : Prop := x^2 / 3 + y^2 = 1

-- Define the eccentricity
noncomputable def eccentricity : ℝ := Real.sqrt 6 / 3

-- Define the distance from right focus to the line x + y + √2 = 0
def focus_distance : ℝ := 2

-- Define the line y = kx + m
def line (k m x y : ℝ) : Prop := y = k * x + m

-- Define the lower vertex A
def vertex_A : ℝ × ℝ := (0, -1)

-- Theorem statement
theorem ellipse_properties :
  ∀ (k m : ℝ),
  k ≠ 0 →
  (∃ (M N : ℝ × ℝ),
    M ≠ N ∧
    ellipse_C M.1 M.2 ∧
    ellipse_C N.1 N.2 ∧
    line k m M.1 M.2 ∧
    line k m N.1 N.2 ∧
    (M.1 - vertex_A.1)^2 + (M.2 - vertex_A.2)^2 = (N.1 - vertex_A.1)^2 + (N.2 - vertex_A.2)^2) →
  1/2 < m ∧ m < 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l948_94833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_problem_l948_94850

noncomputable def arithmetic_sequence (a₁ : ℝ) (d : ℝ) : ℕ → ℝ :=
  λ n => a₁ + (n - 1 : ℝ) * d

noncomputable def S (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) * (2 * a₁ + (n - 1 : ℝ) * d) / 2

theorem arithmetic_sequence_problem (n : ℕ) :
  let a₁ : ℝ := 1
  let d : ℝ := 2
  S a₁ d (n + 2) - S a₁ d n = 36 → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_problem_l948_94850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_one_zero_l948_94853

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 * x - 1

-- Theorem stating that f(x) has exactly one zero
theorem f_has_one_zero : ∃! x : ℝ, f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_one_zero_l948_94853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_9_fourth_power_l948_94870

-- Define the functions h and k
def h : ℝ → ℝ := sorry
def k : ℝ → ℝ := sorry

-- State the theorem
theorem k_9_fourth_power :
  (∀ x ≥ 1, h (k x) = x^3) →
  (∀ x ≥ 1, k (h x) = x^4) →
  k 81 = 9 →
  (k 9)^4 = 81 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_9_fourth_power_l948_94870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_black_car_overtakes_red_car_l948_94848

/-- Represents the time (in hours) it takes for a faster car to overtake a slower car -/
noncomputable def overtake_time (initial_distance : ℝ) (speed_slower : ℝ) (speed_faster : ℝ) : ℝ :=
  initial_distance / (speed_faster - speed_slower)

/-- Theorem: The time for the black car to overtake the red car is 1 hour -/
theorem black_car_overtakes_red_car :
  let initial_distance : ℝ := 10
  let speed_red : ℝ := 40
  let speed_black : ℝ := 50
  overtake_time initial_distance speed_red speed_black = 1 := by
  -- Unfold the definition of overtake_time
  unfold overtake_time
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_black_car_overtakes_red_car_l948_94848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_greater_than_sum_minus_one_l948_94816

theorem product_greater_than_sum_minus_one
  (a b : ℝ)
  (ha : 0 < a ∧ a < 1)
  (hb : 0 < b ∧ b < 1) :
  a * b > a + b - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_greater_than_sum_minus_one_l948_94816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carmen_brush_length_in_mm_l948_94857

-- Define the length of Carla's brush in inches
noncomputable def carla_brush_length : ℝ := 12

-- Define the conversion factor from inches to millimeters
noncomputable def inches_to_mm : ℝ := 25.4

-- Define the percentage of Carmen's brush length compared to Carla's
noncomputable def carmen_percentage : ℝ := 125 / 100

-- Theorem statement
theorem carmen_brush_length_in_mm :
  let carmen_length_inches := (carmen_percentage * carla_brush_length) ^ 2
  carmen_length_inches * inches_to_mm = 381 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_carmen_brush_length_in_mm_l948_94857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_radius_circle_on_ellipse_l948_94861

/-- An ellipse in a 2D plane -/
structure Ellipse where
  center : ℝ × ℝ
  a : ℝ  -- semi-major axis
  b : ℝ  -- semi-minor axis

/-- A pair of conjugate diameters of an ellipse -/
structure ConjugateDiameters (E : Ellipse) where
  diameter1 : ℝ × ℝ → ℝ × ℝ
  diameter2 : ℝ × ℝ → ℝ × ℝ

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Predicate to check if a point is on an ellipse -/
def IsOnEllipse (E : Ellipse) (p : ℝ × ℝ) : Prop :=
  let (x, y) := p
  let (cx, cy) := E.center
  (x - cx)^2 / E.a^2 + (y - cy)^2 / E.b^2 = 1

/-- Predicate to check if a circle is tangent to a line segment -/
def IsTangent (C : Circle) (L : ℝ × ℝ → ℝ × ℝ) : Prop := sorry

/-- Theorem stating that the radius of a circle centered on an ellipse and tangent to conjugate diameters is constant -/
theorem constant_radius_circle_on_ellipse (E : Ellipse) :
  ∀ (CD1 CD2 : ConjugateDiameters E),
  ∀ (C : Circle),
  IsOnEllipse E C.center ∧
  IsTangent C CD1.diameter1 ∧
  IsTangent C CD1.diameter2 ∧
  IsTangent C CD2.diameter1 ∧
  IsTangent C CD2.diameter2 →
  ∃ (r : ℝ), C.radius = r := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_radius_circle_on_ellipse_l948_94861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_is_2_sqrt_5_l948_94801

/-- The distance from a point (x₀, y₀) to a line ax + by + c = 0 -/
noncomputable def distance_point_to_line (x₀ y₀ a b c : ℝ) : ℝ :=
  |a * x₀ + b * y₀ + c| / Real.sqrt (a^2 + b^2)

/-- The center of the circle -/
def P : ℝ × ℝ := (-4, 3)

/-- The coefficients of the line equation 2x + y - 5 = 0 -/
def line_coeffs : ℝ × ℝ × ℝ := (2, 1, -5)

theorem circle_radius_is_2_sqrt_5 :
  let (x₀, y₀) := P
  let (a, b, c) := line_coeffs
  distance_point_to_line x₀ y₀ a b c = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_is_2_sqrt_5_l948_94801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_center_distance_l948_94866

/-- Two unit squares with parallel sides and overlapping area of 1/16 -/
structure OverlappingSquares where
  square1 : Set (ℝ × ℝ)
  square2 : Set (ℝ × ℝ)
  side_length : ℝ
  parallel_sides : Bool
  overlap_area : ℝ

/-- The distance between the centers of two squares -/
noncomputable def center_distance (s : OverlappingSquares) : ℝ := sorry

/-- The theorem stating the minimum distance between the centers -/
theorem min_center_distance (s : OverlappingSquares) 
  (h1 : s.side_length = 1)
  (h2 : s.parallel_sides = true)
  (h3 : s.overlap_area = 1/16) :
  ∃ (d : ℝ), d = Real.sqrt 14 / 4 ∧ ∀ (d' : ℝ), center_distance s ≥ d' → d ≤ d' := by
  sorry

#check min_center_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_center_distance_l948_94866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_age_l948_94817

-- Define the ages as natural numbers
variable (Maya Drew Peter John Jacob : ℕ)

-- Define the conditions
axiom drew_maya : Drew = Maya + 5
axiom peter_drew : Peter = Drew + 4
axiom john_maya : John = 2 * Maya
axiom jacob_peter_future : Jacob + 2 = (Peter + 2) / 2
axiom jacob_age : Jacob = 11

-- Theorem to prove
theorem johns_age : John = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_age_l948_94817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l948_94869

-- Define the function representing the left side of the inequality
noncomputable def f (x : ℝ) : ℝ := (x^2 + 1)/(x-2) + (2*x + 3)/(2*x-1)

-- Define the set of solutions
def solution_set : Set ℝ := Set.Icc 0.5 2 ∪ Set.Ici 3

-- Theorem statement
theorem inequality_solution :
  ∀ x : ℝ, f x ≥ 4 ↔ x ∈ solution_set :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l948_94869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_barycentric_coordinate_vector_relation_l948_94882

/-- Given a triangle ABC and a point X with barycentric coordinates (α:β:γ),
    prove that ⃗AX = β⃗AB + γ⃗AC. -/
theorem barycentric_coordinate_vector_relation 
  {A B C X : EuclideanSpace ℝ (Fin 3)} 
  (α β γ : ℝ) 
  (h_barycentric : X = α • A + β • B + γ • C) 
  (h_sum : α + β + γ = 1) :
  X - A = β • (B - A) + γ • (C - A) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_barycentric_coordinate_vector_relation_l948_94882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l948_94832

/-- The eccentricity of an ellipse with specific properties -/
theorem ellipse_eccentricity (a b c : ℝ) (h1 : a > b) (h2 : b > 0) : 
  let e := c / a
  (∀ x y, x^2/a^2 + y^2/b^2 = 1 → 
    ∃ A C : ℝ × ℝ, 
      A.1 = -c ∧ A.2 = b^2/a ∧
      C.1 = 2*c ∧ C.2 = -b^2/(2*a) ∧
      (2*c, -b^2/a) = (2*(C.1 - c), 2*C.2)) →
  e = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l948_94832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_asymptote_asymptote_greater_than_one_l948_94891

/-- The function f(x) = (a*x^5 - 3x^3 + 7) / (x^5 - 2x^3 + x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a*x^5 - 3*x^3 + 7) / (x^5 - 2*x^3 + x)

/-- The horizontal asymptote of f(x) is y = a -/
theorem horizontal_asymptote (a : ℝ) : 
  ∀ ε > 0, ∃ M, ∀ x, x > M → |f a x - a| < ε := by
  sorry

/-- The asymptote is greater than 1 when a > 1 -/
theorem asymptote_greater_than_one (a : ℝ) (h : a > 1) : 
  ∃ M, ∀ x, x > M → f a x > 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_asymptote_asymptote_greater_than_one_l948_94891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_l948_94883

noncomputable def f (x : Real) : Real :=
  2 * Real.sin (180 * Real.pi / 180 - x) + Real.cos (-x) - Real.sin (450 * Real.pi / 180 - x) + Real.cos (90 * Real.pi / 180 + x)

theorem problem_1 (α : Real) (h1 : f α = 2/3) (h2 : 0 < α) (h3 : α < Real.pi) :
  Real.tan α = 2 * Real.sqrt 5 / 5 ∨ Real.tan α = -2 * Real.sqrt 5 / 5 := by
  sorry

theorem problem_2 (α : Real) (h : f α = 2 * Real.sin α - Real.cos α + 3/4) :
  Real.sin α * Real.cos α = 7/32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_l948_94883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_of_sums_lower_bound_l948_94898

/-- Definition of sum of elements in a finite set of natural numbers -/
def S (X : Finset ℕ+) : ℕ := X.sum (λ x => x.val)

/-- Definition of product of elements in a finite set of natural numbers -/
def P (X : Finset ℕ+) : ℕ := X.prod (λ x => x.val)

/-- Main theorem -/
theorem difference_of_sums_lower_bound 
  (A B : Finset ℕ+) 
  (h_card : A.card = B.card)
  (h_prod : P A = P B)
  (h_sum_neq : S A ≠ S B)
  (h_prime_power : ∀ (n : ℕ+) (p : ℕ) (hn : n ∈ A ∪ B) (hp : Nat.Prime p) (hdvd : p ∣ n),
    (p ^ 36 : ℕ) ∣ n ∧ ¬((p ^ 37 : ℕ) ∣ n)) :
  |((S A : ℤ) - (S B : ℤ))| > (1.9 * 10^6 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_of_sums_lower_bound_l948_94898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_squares_are_rectangles_l948_94846

-- Define the properties as predicates on Type
def is_parallelogram (x : Type) : Prop := sorry
def is_rectangle (x : Type) : Prop := sorry
def is_square (x : Type) : Prop := sorry
def is_rhombus (x : Type) : Prop := sorry

-- Define the sets using set comprehension notation
def A : Set Type := {x : Type | is_parallelogram x}
def B : Set Type := {x : Type | is_rectangle x}
def C : Set Type := {x : Type | is_square x}
def D : Set Type := {x : Type | is_rhombus x}

-- Theorem to prove
theorem squares_are_rectangles : C ⊆ B := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_squares_are_rectangles_l948_94846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_average_speed_l948_94827

/-- Represents the distance between towns B and C -/
noncomputable def D : ℝ := sorry

/-- The speed from R to B in miles per hour -/
def speed_RB : ℝ := 60

/-- The speed from B to C in miles per hour -/
def speed_BC : ℝ := 20

/-- The total distance of the journey -/
noncomputable def total_distance : ℝ := 3 * D

/-- The time taken to travel from R to B -/
noncomputable def time_RB : ℝ := (2 * D) / speed_RB

/-- The time taken to travel from B to C -/
noncomputable def time_BC : ℝ := D / speed_BC

/-- The total time of the journey -/
noncomputable def total_time : ℝ := time_RB + time_BC

/-- The average speed of the whole journey -/
noncomputable def average_speed : ℝ := total_distance / total_time

theorem journey_average_speed : average_speed = 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_average_speed_l948_94827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_increase_l948_94811

/-- Calculates the side length of the nth triangle in the sequence -/
noncomputable def side_length (n : ℕ) : ℝ :=
  3 * (1.25 ^ (n - 1))

/-- Calculates the perimeter of an equilateral triangle given its side length -/
noncomputable def perimeter (side : ℝ) : ℝ :=
  3 * side

/-- Calculates the percent increase between two values -/
noncomputable def percent_increase (initial : ℝ) (final : ℝ) : ℝ :=
  (final - initial) / initial * 100

theorem triangle_perimeter_increase :
  ∃ ε > 0, ε < 0.1 ∧ 
  |percent_increase (perimeter (side_length 1)) (perimeter (side_length 4)) - 95.3| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_increase_l948_94811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_sides_range_l948_94824

open Real

def f (m : ℝ) (x : ℝ) : ℝ := x^3 - 3*x + 2 + m

theorem right_triangle_sides_range (m : ℝ) (h_m : m > 0) :
  (∃ a b c : ℝ, a ∈ Set.Icc 0 2 ∧ b ∈ Set.Icc 0 2 ∧ c ∈ Set.Icc 0 2 ∧ 
   a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
   (f m a)^2 + (f m b)^2 = (f m c)^2) →
  m < 3 + 4 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_sides_range_l948_94824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_project_funding_shortfall_l948_94889

def total_students : ℕ := 45
def project_goal : ℚ := 3000
def full_payment : ℚ := 60
def full_payment_students : ℕ := 25
def high_merit_students : ℕ := 10
def high_merit_payment : ℚ := 40 * 12 / 10
def financial_need_students : ℕ := 7
def financial_need_payment : ℚ := 30 * 27 / 20
def special_discount_students : ℕ := 3
def special_discount_payment : ℚ := 68 * 4 / 5
def admin_fee : ℚ := 10000 * 9 / 1000

def total_collected : ℚ :=
  full_payment * full_payment_students +
  high_merit_payment * high_merit_students +
  financial_need_payment * financial_need_students +
  special_discount_payment * special_discount_students -
  admin_fee

theorem project_funding_shortfall :
  project_goal - total_collected = 5723 / 10 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_project_funding_shortfall_l948_94889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_10_l948_94872

noncomputable def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

noncomputable def sum_of_arithmetic_sequence (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  n * (a 1 + a n) / 2

theorem arithmetic_sequence_sum_10 (a : ℕ → ℝ) :
  arithmetic_sequence a → a 3 + a 8 = 12 → sum_of_arithmetic_sequence a 10 = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_10_l948_94872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_green_ducks_percentage_theorem_l948_94858

-- Define the number of ducks in each park
def ducks_in_park_A : ℕ := 200
def ducks_in_park_B : ℕ := 350
def ducks_in_park_C : ℕ := 120
def ducks_in_park_D : ℕ := 60
def ducks_in_park_E : ℕ := 500

-- Define the percentage of green ducks in each park
def green_ducks_percentage_A : ℚ := 25 / 100
def green_ducks_percentage_B : ℚ := 20 / 100
def green_ducks_percentage_C : ℚ := 50 / 100
def green_ducks_percentage_D : ℚ := 25 / 100
def green_ducks_percentage_E : ℚ := 30 / 100

-- Calculate the total number of ducks
def total_ducks : ℕ := ducks_in_park_A + ducks_in_park_B + ducks_in_park_C + ducks_in_park_D + ducks_in_park_E

-- Calculate the total number of green ducks
def total_green_ducks : ℚ := 
  ducks_in_park_A * green_ducks_percentage_A +
  ducks_in_park_B * green_ducks_percentage_B +
  ducks_in_park_C * green_ducks_percentage_C +
  ducks_in_park_D * green_ducks_percentage_D +
  ducks_in_park_E * green_ducks_percentage_E

-- Theorem: The total percentage of green ducks in the entire birdwatching reserve is approximately 28.05%
theorem green_ducks_percentage_theorem : 
  ∃ (ε : ℚ), ε > 0 ∧ ε < 0.01 ∧ abs ((total_green_ducks / total_ducks : ℚ) * 100 - 28.05) < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_green_ducks_percentage_theorem_l948_94858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_park_area_calculation_l948_94825

/-- Calculates the area of a rectangular park given cycling speed and time -/
noncomputable def park_area (cycling_speed : ℝ) (cycling_time : ℝ) : ℝ :=
  let perimeter := cycling_speed * cycling_time / 7.5
  let breadth := perimeter / 10
  let length := breadth / 4
  length * breadth

/-- Theorem stating the area of the park under given conditions -/
theorem park_area_calculation :
  park_area 12 (8/60) = 102400 := by
  -- Unfold the definition of park_area
  unfold park_area
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_park_area_calculation_l948_94825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_equation_l948_94841

-- Define the points M and N
def M : ℝ × ℝ := (-2, 0)
def N : ℝ × ℝ := (2, 0)

-- Define the vector MN
def MN : ℝ × ℝ := (N.1 - M.1, N.2 - M.2)

-- Define the length of MN
noncomputable def MN_length : ℝ := Real.sqrt ((N.1 - M.1)^2 + (N.2 - M.2)^2)

-- Define the condition for point P
def condition (P : ℝ × ℝ) : Prop :=
  MN_length * Real.sqrt ((P.1 - M.1)^2 + (P.2 - M.2)^2) +
  MN.1 * (P.1 - N.1) + MN.2 * (P.2 - N.2) = 0

-- Theorem statement
theorem trajectory_equation :
  ∀ P : ℝ × ℝ, condition P → P.2^2 = -8 * P.1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_equation_l948_94841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_aquarium_fill_time_l948_94822

/-- Represents the dimensions of a rectangular prism aquarium -/
structure AquariumDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a rectangular prism -/
noncomputable def volume (d : AquariumDimensions) : ℝ :=
  d.length * d.width * d.height

/-- Converts liters per minute to cubic meters per second -/
noncomputable def flowRate (litersPerMinute : ℝ) : ℝ :=
  litersPerMinute * (1 / 1000) * (1 / 60)

/-- Calculates the time to fill an aquarium given its dimensions and flow rate -/
noncomputable def timeToFill (d : AquariumDimensions) (rate : ℝ) : ℝ :=
  volume d / flowRate rate

/-- Theorem: An aquarium with given dimensions filled at 3 liters per minute takes 14400 seconds to fill -/
theorem aquarium_fill_time :
  let d : AquariumDimensions := {
    length := 2,
    width := 0.6,
    height := 0.6
  }
  timeToFill d 3 = 14400 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_aquarium_fill_time_l948_94822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_fourth_rod_l948_94837

/-- A function that checks if four lengths can form a quadrilateral with positive area -/
def can_form_quadrilateral (a b c d : ℕ) : Prop :=
  a + b + c > d ∧ a + b + d > c ∧ a + c + d > b ∧ b + c + d > a

/-- The set of available rod lengths -/
def available_lengths : Finset ℕ :=
  Finset.filter (λ x => x ≠ 4 ∧ x ≠ 9 ∧ x ≠ 18) (Finset.range 36 \ {0})

/-- Helper function to make can_form_quadrilateral decidable -/
def can_form_quadrilateral_dec (a b c d : ℕ) : Bool :=
  a + b + c > d && a + b + d > c && a + c + d > b && b + c + d > a

/-- The theorem to be proved -/
theorem count_valid_fourth_rod :
  (available_lengths.filter (λ d => can_form_quadrilateral_dec 4 9 18 d)).card = 23 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_fourth_rod_l948_94837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_main_theorem_l948_94809

/-- The function f satisfying the given conditions -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 1

/-- The function g as defined in the problem -/
noncomputable def g (x : ℝ) : ℝ := 1 - (2 : ℝ)^x

/-- Main theorem encapsulating all parts of the problem -/
theorem main_theorem (a b : ℝ) (ha : a ≠ 0) :
  (∀ x, f a b (1 + x) = f a b (1 - x)) →
  (∀ x, f a b x + 2 * x = f a b (-x) + 2 * (-x)) →
  (∀ x, f a b x = (x - 1)^2) ∧
  (∃! x, x ∈ Set.Icc 0 1 ∧ f a b x + g x = 0) ∧
  (∀ n, (∃ m, f a b m = g n) → n ≤ 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_main_theorem_l948_94809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_increasing_l948_94838

-- Define the function f(x) = 3^(x^2 - 2x)
noncomputable def f (x : ℝ) : ℝ := 3^(x^2 - 2*x)

-- State the theorem
theorem f_strictly_increasing :
  StrictMonoOn f (Set.Ioi 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_increasing_l948_94838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2017_equals_2_l948_94887

def sequence_a : ℕ → ℚ
  | 0 => 2  -- Add a case for 0 to cover all natural numbers
  | 1 => 2
  | n + 2 => 1 - 1 / sequence_a (n + 1)

theorem a_2017_equals_2 : sequence_a 2017 = 2 := by
  sorry

-- Helper lemma to prove the periodicity
lemma sequence_a_period_3 (n : ℕ) : sequence_a (n + 3) = sequence_a n := by
  sorry

-- Proof of the first few terms
lemma sequence_a_first_terms :
  sequence_a 1 = 2 ∧ sequence_a 2 = 1/2 ∧ sequence_a 3 = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2017_equals_2_l948_94887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_difference_l948_94839

/-- Calculates the future value of an investment with compound interest -/
noncomputable def futureValue (principal : ℝ) (rate : ℝ) (compoundingFrequency : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate / compoundingFrequency) ^ (compoundingFrequency * time)

/-- The initial investment in Fund A -/
def fundA_initial : ℝ := 2000

/-- The initial investment in Fund B -/
def fundB_initial : ℝ := 1000

/-- The annual interest rate for Fund A -/
def fundA_rate : ℝ := 0.12

/-- The annual interest rate for Fund B -/
def fundB_rate : ℝ := 0.30

/-- The compounding frequency for Fund A (monthly) -/
def fundA_compound : ℝ := 12

/-- The compounding frequency for Fund B (quarterly) -/
def fundB_compound : ℝ := 4

/-- The time period in years -/
def time : ℝ := 2

/-- The future value of Fund A after 2 years -/
noncomputable def fundA_future : ℝ := futureValue fundA_initial fundA_rate fundA_compound time

/-- The future value of Fund B after 2 years -/
noncomputable def fundB_future : ℝ := futureValue fundB_initial fundB_rate fundB_compound time

theorem investment_difference :
  ∃ ε > 0, abs (fundA_future - fundB_future - 724.87) < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_difference_l948_94839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_folded_rectangle_perimeter_sum_l948_94865

/-- Represents a folded rectangle with given measurements -/
structure FoldedRectangle where
  ae : ℝ
  be : ℝ
  cf : ℝ

/-- Calculates the perimeter of the folded rectangle -/
noncomputable def perimeter (r : FoldedRectangle) : ℝ :=
  2 * (25 + 70/3)

/-- Theorem stating the perimeter and the sum of numerator and denominator -/
theorem folded_rectangle_perimeter_sum (r : FoldedRectangle) 
  (h1 : r.ae = 8) (h2 : r.be = 17) (h3 : r.cf = 3) :
  ∃ (m n : ℕ), (perimeter r = m / n) ∧ (Nat.Coprime m n) ∧ (m + n = 293) := by
  sorry

#check folded_rectangle_perimeter_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_folded_rectangle_perimeter_sum_l948_94865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeated_ones_divisible_by_nine_l948_94845

/-- The number of ones in the sequence -/
def num_ones (n : ℕ) : ℕ := 3^(3^n)

/-- The number formed by repeating '1' num_ones times -/
def repeated_ones (n : ℕ) : ℕ := 
  (Finset.range (num_ones n)).sum (λ i => 10^i)

theorem repeated_ones_divisible_by_nine (n : ℕ) :
  (repeated_ones n) % 9 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeated_ones_divisible_by_nine_l948_94845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l948_94844

-- Define the vectors
def a : ℝ × ℝ := (-1, 2)
def b : ℝ × ℝ := (1, -4)

-- Define the perpendicularity condition
def is_perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

-- Define the angle between two vectors
noncomputable def angle (v w : ℝ × ℝ) : ℝ :=
  Real.arccos ((v.1 * w.1 + v.2 * w.2) / (Real.sqrt (v.1^2 + v.2^2) * Real.sqrt (w.1^2 + w.2^2)))

theorem vector_problem :
  (∃ k : ℝ, is_perpendicular (4 • a + b) (k • a - b) ∧ k = -13/5) ∧
  (let θ := angle (4 • a + b) (a + b); Real.tan θ = -3/4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l948_94844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_library_books_sold_l948_94821

/-- The number of books sold by a library -/
def books_sold (initial_books : ℕ) (remaining_books : ℕ) (total_books : ℕ) : ℕ :=
  initial_books - (initial_books * remaining_books / total_books)

/-- Theorem stating the number of books sold by the library -/
theorem library_books_sold :
  books_sold 37835 143 271 = 128 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_library_books_sold_l948_94821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_journey_first_part_speed_l948_94854

/-- A journey with two parts -/
structure Journey where
  total_distance : ℝ
  total_time : ℝ
  first_part_duration : ℝ
  second_part_speed : ℝ

/-- The speed during the first part of the journey -/
noncomputable def first_part_speed (j : Journey) : ℝ :=
  (j.total_distance - j.second_part_speed * (j.total_time - j.first_part_duration)) / j.first_part_duration

/-- Theorem stating that for a specific journey, the first part speed is 40 km/h -/
theorem specific_journey_first_part_speed :
  let j : Journey := {
    total_distance := 240
    total_time := 5
    first_part_duration := 3
    second_part_speed := 60
  }
  first_part_speed j = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_journey_first_part_speed_l948_94854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_on_concentric_spheres_l948_94828

theorem area_ratio_on_concentric_spheres 
  (R₁ R₂ A₁ : ℝ) 
  (h₁ : R₁ = 4) 
  (h₂ : R₂ = 6) 
  (h₃ : A₁ = 37) : 
  A₁ * (R₂ / R₁)^2 = 83.25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_on_concentric_spheres_l948_94828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_sum_alpha_gamma_l948_94868

/-- A complex-valued function f(z) = (5 + i)z^2 + αz + γ -/
def f (α γ z : ℂ) : ℂ := (5 + Complex.I) * z^2 + α * z + γ

/-- The theorem stating the minimum value of |α| + |γ| -/
theorem min_abs_sum_alpha_gamma :
  ∃ (α₀ γ₀ : ℂ), 
    (f α₀ γ₀ 1).im = 0 ∧ (f α₀ γ₀ Complex.I).im = 0 ∧
    (∀ α γ : ℂ, (f α γ 1).im = 0 ∧ (f α γ Complex.I).im = 0 → 
      Complex.abs α + Complex.abs γ ≥ Complex.abs α₀ + Complex.abs γ₀) ∧
    Complex.abs α₀ + Complex.abs γ₀ = Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_sum_alpha_gamma_l948_94868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_octagon_centrally_symmetric_not_always_true_for_hexagon_not_always_true_for_dodecagon_l948_94880

/-- A regular octagon with integer side lengths -/
structure RegularOctagon where
  side_length : ℕ
  is_regular : True  -- Placeholder for regularity condition

/-- Central symmetry of a polygon -/
def CentrallySymmetric (polygon : Set (ℝ × ℝ)) : Prop :=
  ∃ (center : ℝ × ℝ), ∀ (point : ℝ × ℝ), point ∈ polygon → 
    ∃ (opposite : ℝ × ℝ), opposite ∈ polygon ∧ 
      (opposite.1 - center.1 = center.1 - point.1) ∧ 
      (opposite.2 - center.2 = center.2 - point.2)

/-- Function to convert RegularOctagon to a set of points -/
def regularOctagonToSet (octagon : RegularOctagon) : Set (ℝ × ℝ) :=
  sorry  -- Implementation details omitted for brevity

/-- Theorem: A regular octagon with integer side lengths is centrally symmetric -/
theorem regular_octagon_centrally_symmetric (octagon : RegularOctagon) :
  CentrallySymmetric (regularOctagonToSet octagon) :=
sorry

/-- Counterexample for hexagon -/
def counterexampleHexagon : Set (ℝ × ℝ) :=
  sorry  -- Implementation of the counterexample

/-- Theorem: The statement is not always true for hexagons -/
theorem not_always_true_for_hexagon :
  ∃ (hexagon : Set (ℝ × ℝ)), ¬ CentrallySymmetric hexagon :=
sorry

/-- Counterexample for dodecagon -/
def counterexampleDodecagon : Set (ℝ × ℝ) :=
  sorry  -- Implementation of the counterexample

/-- Theorem: The statement is not always true for dodecagons -/
theorem not_always_true_for_dodecagon :
  ∃ (dodecagon : Set (ℝ × ℝ)), ¬ CentrallySymmetric dodecagon :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_octagon_centrally_symmetric_not_always_true_for_hexagon_not_always_true_for_dodecagon_l948_94880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_window_breaker_l948_94804

-- Define the children
inductive Child : Type
| BaoBao : Child
| KeKe : Child
| MaoMao : Child
| DuoDuo : Child

-- Define the culprit
variable (culprit : Child)

-- Define the statement of each child
def statement (c : Child) : Prop :=
  match c with
  | Child.BaoBao => Child.KeKe = culprit
  | Child.KeKe => Child.MaoMao = culprit
  | Child.MaoMao => ¬(Child.KeKe = culprit)
  | Child.DuoDuo => ¬(Child.DuoDuo = culprit)

-- Theorem statement
theorem window_breaker :
  (∃! c : Child, statement culprit c) →
  (culprit = Child.MaoMao ∧ statement culprit Child.DuoDuo) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_window_breaker_l948_94804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reservoir_fill_time_l948_94875

/-- Represents the rate at which one water pipe can fill the reservoir -/
noncomputable def fill_rate : ℝ → ℝ := sorry

/-- Represents the rate at which water is drained from the reservoir -/
noncomputable def drain_rate : ℝ → ℝ := sorry

/-- Represents the volume of the reservoir -/
noncomputable def reservoir_volume : ℝ → ℝ := sorry

/-- Represents the time it takes to fill the reservoir with a given number of pipes -/
noncomputable def fill_time (num_pipes : ℝ) (r : ℝ) : ℝ :=
  reservoir_volume r / (num_pipes * fill_rate r - drain_rate r)

theorem reservoir_fill_time (r : ℝ) :
  (fill_time 3 r = 30) →
  (fill_time 5 r = 10) →
  (fill_time 4 r = 15) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reservoir_fill_time_l948_94875
