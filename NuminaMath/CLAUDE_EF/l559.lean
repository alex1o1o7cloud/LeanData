import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_equations_l559_55977

noncomputable section

/-- Curve C₁ with parametric equation x = a cos φ, y = b sin φ -/
def curve_C₁ (a b : ℝ) (φ : ℝ) : ℝ × ℝ := (a * Real.cos φ, b * Real.sin φ)

/-- Point M on curve C₁ -/
def point_M : ℝ × ℝ := (2, Real.sqrt 3)

/-- Curve C₂ in polar coordinates -/
def curve_C₂ (R : ℝ) (θ : ℝ) : ℝ := 2 * R * Real.cos θ

/-- Point D on curve C₂ -/
def point_D : ℝ × ℝ := (Real.sqrt 2, Real.pi/4)

theorem curve_equations (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  (∃ φ, curve_C₁ a b φ = point_M ∧ φ = Real.pi/3) →
  (∃ R, curve_C₂ R (Real.pi/4) = Real.sqrt 2) →
  (∀ x y, x^2/16 + y^2/4 = 1 ↔ ∃ φ, (x, y) = curve_C₁ a b φ) ∧
  (∀ ρ θ, ρ = 2 * Real.cos θ ↔ ρ = curve_C₂ 1 θ) ∧
  (∀ ρ₁ ρ₂ θ, (∃ φ₁, (ρ₁ * Real.cos θ, ρ₁ * Real.sin θ) = curve_C₁ a b φ₁) →
               (∃ φ₂, (ρ₂ * Real.cos (θ + Real.pi/2), ρ₂ * Real.sin (θ + Real.pi/2)) = curve_C₁ a b φ₂) →
               1/ρ₁^2 + 1/ρ₂^2 = 5/16) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_equations_l559_55977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_ports_l559_55934

/-- The distance between two ports along a river -/
noncomputable def river_distance (ship_speed water_speed : ℝ) (time_difference : ℝ) : ℝ :=
  let downstream_speed := ship_speed + water_speed
  let upstream_speed := ship_speed - water_speed
  (time_difference * downstream_speed * upstream_speed) / (upstream_speed - downstream_speed)

/-- Theorem stating the distance between ports A and B -/
theorem distance_between_ports :
  let ship_speed : ℝ := 26
  let water_speed : ℝ := 2
  let time_difference : ℝ := 3
  river_distance ship_speed water_speed time_difference = 504 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval river_distance 26 2 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_ports_l559_55934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_intersection_points_l559_55943

-- Define the circle
def circle_equation (h k r : ℝ) (x y : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

-- Define the cosine function
def cosine_graph (x y : ℝ) : Prop :=
  y = Real.cos x

-- Define the interval
def interval (x : ℝ) : Prop :=
  -2 * Real.pi ≤ x ∧ x ≤ 2 * Real.pi

-- Theorem statement
theorem max_intersection_points (h k r : ℝ) :
  ∃ (n : ℕ), n ≤ 8 ∧
  ∀ (m : ℕ), (∃ (points : Finset (ℝ × ℝ)),
    (∀ (p : ℝ × ℝ), p ∈ points →
      circle_equation h k r p.1 p.2 ∧
      cosine_graph p.1 p.2 ∧
      interval p.1) →
    m = points.card) →
  m ≤ n :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_intersection_points_l559_55943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_composition_value_l559_55940

-- Define the functions f and g
noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

-- Define the condition that g has an inverse
def g_has_inverse : Function.Injective g := sorry

-- Define the given relationship between f and g
axiom fg_relation : ∀ x : ℝ, f⁻¹ (g x) = x^4 - 3

-- State the theorem to be proved
theorem inverse_composition_value :
  g⁻¹ (f 10) = (13 : ℝ)^(1/4) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_composition_value_l559_55940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_cross_section_area_l559_55968

theorem cone_cross_section_area (l R : ℝ) (h : l > 0) (h2 : R > 0) :
  (R / l ∈ Set.Icc (Real.sqrt 2 / 2) 1) ↔ 
  (∃ θ : ℝ, θ ∈ Set.Ioo (0 : ℝ) (π / 2) ∧ R * l * Real.sin θ = l^2 / 2) :=
by
  sorry

#check cone_cross_section_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_cross_section_area_l559_55968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ilya_has_winning_strategy_l559_55974

/-- Represents a game state with three piles of stones -/
structure GameState where
  pile1 : Nat
  pile2 : Nat
  pile3 : Nat
deriving Repr

/-- Represents a player in the game -/
inductive Player where
  | Ilya
  | Kostya
deriving Repr

/-- Represents a move in the game -/
inductive Move where
  | TakeFromPile1
  | TakeFromPile2
  | TakeFromPile3
deriving Repr

/-- Checks if a move is valid given the current state and previous move -/
def isValidMove (state : GameState) (move : Move) (prevMove : Option Move) : Prop :=
  match move, prevMove with
  | Move.TakeFromPile1, some Move.TakeFromPile1 => False
  | Move.TakeFromPile2, some Move.TakeFromPile2 => False
  | Move.TakeFromPile3, some Move.TakeFromPile3 => False
  | Move.TakeFromPile1, _ => state.pile1 > 0
  | Move.TakeFromPile2, _ => state.pile2 > 0
  | Move.TakeFromPile3, _ => state.pile3 > 0

/-- Applies a move to the current game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  match move with
  | Move.TakeFromPile1 => { state with pile1 := state.pile1 - 1 }
  | Move.TakeFromPile2 => { state with pile2 := state.pile2 - 1 }
  | Move.TakeFromPile3 => { state with pile3 := state.pile3 - 1 }

/-- Defines a winning strategy for a player -/
def hasWinningStrategy (player : Player) (initialState : GameState) : Prop :=
  ∃ (strategy : GameState → Move),
    ∀ (gameSequence : Nat → GameState),
      gameSequence 0 = initialState →
      (∀ n, isValidMove (gameSequence n) (strategy (gameSequence n)) none) →
      (∀ n, gameSequence (n + 1) = applyMove (gameSequence n) (strategy (gameSequence n))) →
      ∃ k, ¬∃ m, isValidMove (gameSequence k) m none

/-- The main theorem stating that Ilya has a winning strategy -/
theorem ilya_has_winning_strategy :
  hasWinningStrategy Player.Ilya { pile1 := 100, pile2 := 101, pile3 := 102 } := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ilya_has_winning_strategy_l559_55974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_remainder_l559_55951

theorem product_remainder (a b c : ℕ) 
  (ha : a % 7 = 2) 
  (hb : b % 7 = 3) 
  (hc : c % 7 = 4) : 
  (a * b * c) % 7 = 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_remainder_l559_55951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_x_axis_at_10_0_l559_55991

/-- A line passing through two points (x₁, y₁) and (x₂, y₂) -/
structure Line where
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ

/-- The slope of a line -/
noncomputable def Line.slope (l : Line) : ℝ :=
  (l.y₂ - l.y₁) / (l.x₂ - l.x₁)

/-- The y-intercept of a line -/
noncomputable def Line.yIntercept (l : Line) : ℝ :=
  l.y₁ - l.slope * l.x₁

/-- The x-coordinate of the intersection point with the x-axis -/
noncomputable def Line.xAxisIntersection (l : Line) : ℝ :=
  -l.yIntercept / l.slope

/-- The theorem stating that the given line intersects the x-axis at (10, 0) -/
theorem line_intersects_x_axis_at_10_0 :
  let l : Line := { x₁ := 9, y₁ := 1, x₂ := 5, y₂ := 5 }
  l.xAxisIntersection = 10 ∧ l.yIntercept + l.slope * l.xAxisIntersection = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_x_axis_at_10_0_l559_55991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_correct_l559_55944

/-- A line defined by parametric equations -/
def line (t : ℝ) : ℝ × ℝ := (-2 + t, 1 - t)

/-- The equation of the circle -/
def circle_eq (x y : ℝ) : Prop := (x - 3)^2 + (y + 1)^2 = 25

/-- The chord length intercepted by the circle on the line -/
noncomputable def chord_length : ℝ := Real.sqrt 82

/-- Theorem stating that the chord length is correct -/
theorem chord_length_is_correct : 
  ∃ t₁ t₂ : ℝ, t₁ ≠ t₂ ∧ 
  circle_eq (line t₁).1 (line t₁).2 ∧
  circle_eq (line t₂).1 (line t₂).2 ∧
  (line t₁).1 - (line t₂).1 = chord_length * (((line t₁).1 - (line t₂).1) / 
    Real.sqrt ((line t₁).1 - (line t₂).1)^2 + ((line t₁).2 - (line t₂).2)^2) ∧
  (line t₁).2 - (line t₂).2 = chord_length * (((line t₁).2 - (line t₂).2) / 
    Real.sqrt ((line t₁).1 - (line t₂).1)^2 + ((line t₁).2 - (line t₂).2)^2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_correct_l559_55944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_sum_l559_55979

/-- The combined area of the triangles formed by the intersection of the lines
    y = x, x = -6, y = -2x + 4, and the x-axis -/
theorem triangle_area_sum : ℝ := by
  /- Line equations -/
  let line1 (x y : ℝ) : Prop := y = x
  let line2 (x : ℝ) : Prop := x = -6
  let line3 (x y : ℝ) : Prop := y = -2*x + 4

  /- Points of intersection -/
  let point1 : ℝ × ℝ := (-6, -6)
  let point2 : ℝ × ℝ := (4/3, 4/3)
  let point3 : ℝ × ℝ := (-6, 16)

  /- Areas of individual triangles -/
  let area1 : ℝ := (1/2) * 6 * 6
  let area2 : ℝ := (1/2) * (4/3) * (4/3)
  let area3 : ℝ := (1/2) * 6 * 16

  /- Combined area -/
  let total_area : ℝ := area1 + area2 + area3

  /- Theorem statement -/
  have : total_area = 66 + 8/9 := by
    -- Proof goes here
    sorry

  exact 66 + 8/9


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_sum_l559_55979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_terms_are_perfect_squares_l559_55990

def intSequence (a₀ a₁ : ℤ) : ℕ → ℤ
  | 0 => a₀
  | 1 => a₁
  | (n + 2) => 3 * intSequence a₀ a₁ (n + 1) - 3 * intSequence a₀ a₁ n + intSequence a₀ a₁ (n - 1)

theorem all_terms_are_perfect_squares (a₀ a₁ : ℤ) :
  (2 * a₁ = a₀ + intSequence a₀ a₁ 2 - 2) →
  (∀ m : ℕ, ∃ k : ℕ, ∀ i : ℕ, i < m → ∃ n : ℕ, intSequence a₀ a₁ (k + i) = n ^ 2) →
  (∀ n : ℕ, ∃ k : ℕ, intSequence a₀ a₁ n = k ^ 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_terms_are_perfect_squares_l559_55990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l559_55946

-- Define the triangle
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- State the theorem
theorem triangle_problem (abc : Triangle) 
  (h1 : Real.cos abc.B = 3/5) 
  (h2 : abc.a * abc.c * Real.cos abc.B = -21) 
  (h3 : abc.a = 7) : 
  -- Part 1: Area of the triangle
  (1/2 * abc.a * abc.c * Real.sin abc.B = 14) ∧ 
  -- Part 2: Angle C
  (abc.C = Real.pi/4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l559_55946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_euler_char_sum_l559_55993

-- Define a surface
class Surface where

-- Define a circle on a surface
class Circle (S : Surface) where

-- Define a separating circle on a surface
class SeparatingCircle (S : Surface) extends Circle S where

-- Define the operation of cutting a surface along a circle and covering the holes
noncomputable def cut_and_cover (S : Surface) (C : SeparatingCircle S) : Surface × Surface :=
  sorry

-- Define Euler characteristic for a surface
noncomputable def euler_char (S : Surface) : ℤ :=
  sorry

-- Theorem statement
theorem euler_char_sum (S : Surface) (C : SeparatingCircle S) :
  let (S', S'') := cut_and_cover S C
  euler_char S = euler_char S' + euler_char S'' := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_euler_char_sum_l559_55993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_closed_form_l559_55971

/-- Recursive definition of x_k -/
noncomputable def x (n : ℕ) (c : ℝ) : ℕ → ℝ
  | 0 => 0
  | 1 => 1
  | (k+2) => (c * x n c (k+1) - (n - k : ℝ) * x n c k) / (k + 1 : ℝ)

/-- Theorem stating the closed form of x_k -/
theorem x_closed_form (n : ℕ) (k : ℕ) (h1 : n > 0) (h2 : 1 ≤ k) (h3 : k ≤ n) :
  x n (n - 1 : ℝ) k = (Nat.choose (n-1) (k-1) : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_closed_form_l559_55971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l559_55973

-- Define the function f
variable (f : ℝ → ℝ)

-- Define symmetry about x = 1
def symmetric_about_one (f : ℝ → ℝ) : Prop :=
  ∀ x, f (2 - x) = f x

-- Define monotonically decreasing on (-∞, 1]
def monotone_decreasing_to_one (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y ∧ y ≤ 1 → f y ≤ f x

-- Theorem statement
theorem function_inequality {f : ℝ → ℝ} 
  (h_sym : symmetric_about_one f) 
  (h_mono : monotone_decreasing_to_one f) :
  f 3 > f 0 ∧ f 0 > f (3/2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l559_55973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_A_l559_55969

-- Define the vectors m and n
noncomputable def m (x : ℝ) : ℝ × ℝ := (1, Real.cos x)
noncomputable def n (t x : ℝ) : ℝ × ℝ := (t, Real.sqrt 3 * Real.sin x - Real.cos x)

-- Define the function f
noncomputable def f (t x : ℝ) : ℝ := (m x).1 * (n t x).1 + (m x).2 * (n t x).2

-- State the theorem
theorem range_of_f_A (A B C : ℝ) (a b c : ℝ) : 
  (∀ x, f (1/2) x = Real.sin (2*x - π/6) - 1/2) →
  f (1/2) (π/12) = 0 →
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →
  A + B + C = π →
  a = (Real.cos B + b * Real.cos C) / (2 * Real.cos B) →
  (∃ y, y ∈ Set.Ioo (-1/2 : ℝ) 1 ∧ y = f (1/2) A) ∧ 
  (∀ y, y = f (1/2) A → y ≤ 1 ∧ y > -1/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_A_l559_55969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_is_200pi_l559_55900

/-- The volume of a structure with a cylindrical base and conical top -/
noncomputable def structure_volume (total_height radius base_height_ratio : ℝ) : ℝ :=
  let base_height := total_height * base_height_ratio
  let cone_height := total_height - base_height
  let cylinder_volume := Real.pi * radius^2 * base_height
  let cone_volume := (1/3) * Real.pi * radius^2 * cone_height
  cylinder_volume + cone_volume

/-- The total volume of the structure is 200π cubic feet -/
theorem volume_is_200pi :
  structure_volume 12 5 0.5 = 200 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_is_200pi_l559_55900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_twin_primes_characterization_l559_55932

theorem twin_primes_characterization (n : ℕ) (h_odd : Odd n) :
  (Prime n ∧ Prime (n + 2)) ↔ (n * (n + 2) ∣ 4 * (Nat.factorial (n - 1) + 1) + n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_twin_primes_characterization_l559_55932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_roots_sum_l559_55923

open Real Set

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := sin (2 * x + π / 6)

-- Define the domain
def domain : Set ℝ := Icc 0 (7 * π / 6)

-- State the theorem
theorem three_roots_sum (m : ℝ) (x₁ x₂ x₃ : ℝ) :
  x₁ ∈ domain → x₂ ∈ domain → x₃ ∈ domain →
  x₁ < x₂ → x₂ < x₃ →
  f x₁ = m → f x₂ = m → f x₃ = m →
  (∀ x ∈ domain, f x = m → x = x₁ ∨ x = x₂ ∨ x = x₃) →
  x₁ + 2 * x₂ + x₃ = 5 * π / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_roots_sum_l559_55923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_chord_through_M_l559_55950

-- Define the circle
def my_circle (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 2*y = 0

-- Define point M
def point_M : ℝ × ℝ := (1, 0)

-- Define the line equation
def line_equation (x y : ℝ) : Prop := x + y - 1 = 0

-- Theorem statement
theorem shortest_chord_through_M :
  ∀ (x y : ℝ),
  my_circle x y →
  (∃ (t : ℝ), x = 1 + t ∧ y = t) →
  line_equation x y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_chord_through_M_l559_55950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_tree_height_l559_55978

/-- Given five trees with specific height relationships, prove the height of the fifth tree -/
theorem fifth_tree_height (h1 h2 h3 h4 h5 : ℝ) : 
  h1 = 108 ∧ 
  h2 = h1 / 2 - 6 ∧ 
  h3 = h2 / 4 ∧ 
  h4 = h2 + h3 - 2 ∧ 
  h5 = 0.75 * (h1 + h2 + h3 + h4) → 
  h5 = 169.5 := by
  intro h
  sorry

#check fifth_tree_height

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_tree_height_l559_55978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l559_55996

-- Define set A
def A : Set ℝ := {x | x^2 - 3*x - 10 < 0}

-- Define set B
def B : Set ℝ := {x | Real.exp (x * Real.log 2) < 2}

-- Theorem statement
theorem intersection_A_B : A ∩ B = Set.Ioo (-2) 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l559_55996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_divisor_property_l559_55989

def sequence_a (n : ℕ) : ℕ :=
  match n with
  | 0 => 2
  | 1 => 1
  | n+2 => sequence_a (n+1) + sequence_a n

theorem prime_divisor_property (k : ℕ) (p : ℕ) (h_prime : Nat.Prime p) :
  p ∣ (sequence_a (2 * k) - 2) → p ∣ (sequence_a (2 * k + 1) - 1) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_divisor_property_l559_55989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_to_cube_volume_ratio_l559_55906

/-- The ratio of the volume of a regular tetrahedron formed by joining four non-coplanar vertices
    of a cube to the volume of the cube itself. -/
noncomputable def tetrahedronToCubeVolumeRatio (s : ℝ) : ℝ :=
  let cubeVolume := s^3
  let tetrahedronEdgeLength := s * Real.sqrt 3
  let tetrahedronVolume := (Real.sqrt 2 / 12) * tetrahedronEdgeLength^3
  tetrahedronVolume / cubeVolume

/-- Theorem stating that the ratio of the volume of a regular tetrahedron formed by joining
    four non-coplanar vertices of a cube to the volume of the cube is √6/4. -/
theorem tetrahedron_to_cube_volume_ratio (s : ℝ) (h : s > 0) :
  tetrahedronToCubeVolumeRatio s = Real.sqrt 6 / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_to_cube_volume_ratio_l559_55906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_peter_walking_speed_l559_55987

/-- Represents the walking speed in minutes per mile -/
noncomputable def walkingSpeed (totalDistance remainingTime remainingDistance : ℝ) : ℝ :=
  remainingTime / remainingDistance

theorem peter_walking_speed :
  let totalDistance : ℝ := 2.5
  let walkedDistance : ℝ := 1
  let remainingTime : ℝ := 30
  let remainingDistance : ℝ := totalDistance - walkedDistance
  walkingSpeed totalDistance remainingTime remainingDistance = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_peter_walking_speed_l559_55987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_touching_semicircles_distance_squared_l559_55918

theorem touching_semicircles_distance_squared (P Q R S : ℝ × ℝ) :
  let radius : ℝ := 1
  let semicircle1 := {(x, y) : ℝ × ℝ | (x - P.1)^2 + (y - P.2)^2 = radius^2 ∧ y ≥ P.2}
  let semicircle2 := {(x, y) : ℝ × ℝ | (x - R.1)^2 + (y - R.2)^2 = radius^2 ∧ y ≥ R.2}
  (∀ (p : ℝ × ℝ), p ∈ semicircle1 ∩ semicircle2 → p = (P.1 + radius, P.2)) →
  (Q.2 = P.2 ∧ S.2 = R.2) →
  (Q.1 - P.1 = 2 * radius ∧ S.1 - R.1 = 2 * radius) →
  (P.2 = R.2) →
  (S.1 - P.1)^2 + (S.2 - P.2)^2 = 8 + 4 * Real.sqrt 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_touching_semicircles_distance_squared_l559_55918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_A_l559_55967

/-- Given points A, B, C, and the conditions for A' and B', prove that |A'B'| = 32/7 -/
theorem length_of_A'B' (A B C A' B' : ℝ × ℝ) : 
  A = (0, 7) →
  B = (0, 11) →
  C = (3, 7) →
  (A'.1 = A'.2) →  -- A' is on the line y = x
  (B'.1 = B'.2) →  -- B' is on the line y = x
  (∃ t : ℝ, A + t • (A' - A) = C) →  -- AA' passes through C
  (∃ s : ℝ, B + s • (B' - B) = C) →  -- BB' passes through C
  ‖A' - B'‖ = 32 / 7 := by
  sorry

#check length_of_A'B'

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_A_l559_55967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_side_equality_l559_55930

theorem regular_polygon_side_equality :
  ∃ (n : ℕ), n > 2 ∧ 
  (60 : ℚ) / n = (67 : ℚ) / (n + 7) := by
  use 60
  constructor
  · norm_num
  · field_simp
    ring

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_side_equality_l559_55930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_reduction_l559_55999

theorem trigonometric_equation_reduction (a b c : ℕ+) :
  (∀ x : ℝ, (Real.cos x)^2 + (Real.cos (2*x))^2 + (Real.sin (3*x))^2 + (Real.sin (4*x))^2 = 3) →
  (∀ x : ℝ, Real.sin (↑a * x) * Real.sin (↑b * x) * Real.cos (↑c * x) = 0) →
  a + b + c = 7 := by
  sorry

#check trigonometric_equation_reduction

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_reduction_l559_55999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_transformation_l559_55982

noncomputable def f (x φ : ℝ) : ℝ := Real.cos (2 * x + φ)

theorem function_transformation (φ : ℝ) 
  (h1 : |φ| ≤ π/2) 
  (h2 : ∀ x, f x φ = -f (8*π/3 - x) φ) :
  ∀ x, f (x - π/6) φ = Real.sin (2*x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_transformation_l559_55982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coordinates_is_32_l559_55942

noncomputable def distance_point_to_line (x y : ℝ) : ℝ := |y - 10|

noncomputable def distance_between_points (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

theorem sum_of_coordinates_is_32 :
  ∃ (x1 y1 x2 y2 x3 y3 x4 y4 : ℝ),
    (distance_point_to_line x1 y1 = 4) ∧
    (distance_point_to_line x2 y2 = 4) ∧
    (distance_point_to_line x3 y3 = 4) ∧
    (distance_point_to_line x4 y4 = 4) ∧
    (distance_between_points x1 y1 3 10 = 15) ∧
    (distance_between_points x2 y2 3 10 = 15) ∧
    (distance_between_points x3 y3 3 10 = 15) ∧
    (distance_between_points x4 y4 3 10 = 15) ∧
    (x1 + y1 + x2 + y2 + x3 + y3 + x4 + y4 = 32) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coordinates_is_32_l559_55942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l559_55938

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions of the problem
def problem_conditions (t : Triangle) : Prop :=
  t.b = 2 ∧ 
  t.A = 2 * Real.pi / 3 ∧ -- 120° in radians
  1/2 * t.a * t.b * Real.sin t.A = 2 * Real.sqrt 3

-- Theorem statement
theorem triangle_side_length (t : Triangle) 
  (h : problem_conditions t) : t.a = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l559_55938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_has_difference_7992_in_58408_l559_55964

/-- Given a natural number n and a digit d, returns the list of local values of d in n -/
def localValues (n : ℕ) (d : ℕ) : List ℕ := sorry

/-- Returns true if the difference between the maximum and minimum local values of a digit in a number is equal to a given value -/
def hasDifferenceInLocalValues (n : ℕ) (d : ℕ) (diff : ℕ) : Prop :=
  let values := localValues n d
  values.length > 1 ∧ 
  (∃ max min, values.maximum? = some max ∧ values.minimum? = some min ∧ max - min = diff)

theorem eight_has_difference_7992_in_58408 :
  hasDifferenceInLocalValues 58408 8 7992 ∧
  ∀ d, d ≠ 8 → ¬(hasDifferenceInLocalValues 58408 d 7992) :=
sorry

#check eight_has_difference_7992_in_58408

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_has_difference_7992_in_58408_l559_55964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_time_implies_two_thirds_ratio_l559_55984

/-- Represents a point on a line --/
structure Point where
  x : ℝ

/-- Represents a person's movement capabilities --/
structure Person where
  walkSpeed : ℝ
  cycleSpeed : ℝ
  h_cycle_faster : cycleSpeed = 5 * walkSpeed

/-- The scenario with initial position, starting point, and destination --/
structure Scenario where
  initial : Point
  start : Point
  destination : Point
  h_between : start.x < initial.x ∧ initial.x < destination.x

/-- Time taken for a direct walk to the destination --/
noncomputable def directWalkTime (s : Scenario) (p : Person) : ℝ :=
  (s.destination.x - s.initial.x) / p.walkSpeed

/-- Time taken to walk back and then cycle to the destination --/
noncomputable def backAndCycleTime (s : Scenario) (p : Person) : ℝ :=
  (s.initial.x - s.start.x) / p.walkSpeed + (s.destination.x - s.start.x) / p.cycleSpeed

theorem equal_time_implies_two_thirds_ratio 
    (s : Scenario) (p : Person) 
    (h_equal_time : directWalkTime s p = backAndCycleTime s p) :
  (s.initial.x - s.start.x) / (s.destination.x - s.initial.x) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_time_implies_two_thirds_ratio_l559_55984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_cos_plus_sin_min_distance_x1_x2_l559_55931

open Real

-- Define the relationship between f and g
noncomputable def g (f : ℝ → ℝ) (x : ℝ) : ℝ := f x * f (x + π / 2)

-- Part 1
theorem g_cos_plus_sin (x : ℝ) : 
  g (fun x => cos x + sin x) x = cos (2 * x) := by sorry

-- Part 2
theorem min_distance_x1_x2 :
  ∃ x₁ x₂, (∀ x, g (fun x => |sin x| + cos x) x₁ ≤ g (fun x => |sin x| + cos x) x ∧ 
              g (fun x => |sin x| + cos x) x ≤ g (fun x => |sin x| + cos x) x₂) ∧
  (∀ y₁ y₂, (∀ x, g (fun x => |sin x| + cos x) y₁ ≤ g (fun x => |sin x| + cos x) x ∧ 
                    g (fun x => |sin x| + cos x) x ≤ g (fun x => |sin x| + cos x) y₂) → 
            |x₁ - x₂| ≤ |y₁ - y₂|) ∧
  |x₁ - x₂| = 3 * π / 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_cos_plus_sin_min_distance_x1_x2_l559_55931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_increasing_lambda_bound_l559_55945

theorem sequence_increasing_lambda_bound (lambda : ℝ) :
  (∀ n : ℕ+, ∃ a : ℝ, a = n^2 + lambda * n) →
  (∀ n : ℕ+, ∃ a b : ℝ, a = n^2 + lambda * n ∧ b = (n + 1)^2 + lambda * (n + 1) ∧ b > a) →
  lambda > -3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_increasing_lambda_bound_l559_55945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_specific_case_l559_55908

/-- Given two circles with radii r₁ and r₂, whose centers are d units apart,
    intersecting at point P, and a line through P creating equal chords QP and PR,
    this function calculates the square of the length of QP. -/
noncomputable def chord_length_squared (r₁ r₂ d : ℝ) : ℝ :=
  let cos_angle := (r₁^2 + r₂^2 - d^2) / (2 * r₁ * r₂)
  let sin_half_angle := Real.sqrt ((1 - cos_angle) / 2)
  2 * r₁^2 * (1 - sin_half_angle)

/-- The theorem states that for two circles with radii 10 and 7, whose centers
    are 15 units apart, the square of the length of the chord QP (where QP = PR)
    is equal to 200 - 200 * √(27/35). -/
theorem chord_length_specific_case :
  chord_length_squared 10 7 15 = 200 - 200 * Real.sqrt (27/35) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_specific_case_l559_55908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_statue_profit_percentage_l559_55958

/-- Calculate the percentage profit given the selling price and original cost -/
noncomputable def percentage_profit (selling_price : ℝ) (original_cost : ℝ) : ℝ :=
  ((selling_price - original_cost) / original_cost) * 100

/-- The statue problem -/
theorem statue_profit_percentage :
  let selling_price : ℝ := 550
  let original_cost : ℝ := 407.41
  abs (percentage_profit selling_price original_cost - 34.99) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_statue_profit_percentage_l559_55958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l559_55929

-- Define the polar curves
noncomputable def curve1 (θ : ℝ) : ℝ := 4 * Real.sin θ
noncomputable def curve2 (θ : ℝ) : ℝ := 1 / Real.cos θ

-- Define the intersection points in Cartesian coordinates
def intersection_points : Set (ℝ × ℝ) :=
  {p | ∃ θ : ℝ, 
    p.1 = curve1 θ * Real.cos θ ∧ 
    p.2 = curve1 θ * Real.sin θ ∧
    curve1 θ = curve2 θ}

-- Theorem statement
theorem intersection_distance : 
  ∃ A B : ℝ × ℝ, A ∈ intersection_points ∧ B ∈ intersection_points ∧ A ≠ B ∧ 
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l559_55929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetry_l559_55907

-- Define the function f
noncomputable def f (a b c x : ℝ) : ℝ := a * Real.tan x - b * x^5 + c * x - 3

-- State the theorem
theorem f_symmetry (a b c : ℝ) : f a b c (-3) = 7 → f a b c 3 = -13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetry_l559_55907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_progression_iff_equal_initial_terms_l559_55928

/-- A sequence is a geometric progression if the ratio of consecutive terms is constant. -/
def IsGeometricProgression (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The sequence a_n defined by the given recurrence relation. -/
def RecurrenceSequence (a b k : ℝ) : ℕ → ℝ
  | 0 => a  -- Add this case for n = 0
  | 1 => b
  | n + 2 => k * RecurrenceSequence a b k n * RecurrenceSequence a b k (n + 1)

theorem geometric_progression_iff_equal_initial_terms
  (a b k : ℝ) (ha : a > 0) (hb : b > 0) (hk : k > 0) :
  IsGeometricProgression (RecurrenceSequence a b k) ↔ a = b := by
  sorry

#check geometric_progression_iff_equal_initial_terms

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_progression_iff_equal_initial_terms_l559_55928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_l559_55998

/-- A function f with parameter ω -/
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := (2 * Real.cos (ω * x) - 1) * Real.sin (ω * x - Real.pi / 4)

/-- The theorem stating the range of ω given the conditions -/
theorem omega_range (ω : ℝ) :
  ω > 0 →
  (∃ (z : Finset ℝ), z.card = 6 ∧ (∀ x ∈ z, 0 < x ∧ x < 3 * Real.pi ∧ f ω x = 0)) →
  7 / 9 < ω ∧ ω ≤ 13 / 12 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_l559_55998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_vertex_ratio_l559_55903

/-- A parabola in 2D space -/
structure Parabola where
  f : ℝ → ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance between two points -/
noncomputable def distance (p q : Point) : ℝ := Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- The theorem to be proved -/
theorem parabola_focus_vertex_ratio 
  (P : Parabola)
  (V₁ F₁ V₂ F₂ A B : Point)
  (h₁ : ∀ x, P.f x = (x - 1)^2 - 3)
  (h₂ : V₁ = ⟨1, -3⟩)
  (h₃ : F₁.y = V₁.y + 1/4)
  (h₄ : P.f A.x = A.y ∧ P.f B.x = B.y)
  (h₅ : (A.x - V₁.x) * (B.x - V₁.x) + (A.y - V₁.y) * (B.y - V₁.y) = 0)
  (h₆ : V₂ = ⟨0, -1⟩)
  (h₇ : F₂ = ⟨0, -3/4⟩) :
  distance F₁ F₂ / distance V₁ V₂ = 1 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_vertex_ratio_l559_55903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cardinality_equality_l559_55963

/-- For any integer n ≥ 3, the cardinality of set G equals the cardinality of set L -/
theorem cardinality_equality (n : ℕ) (h : n ≥ 3) :
  let G := {p : ℕ × ℕ | 1 ≤ p.1 ∧ p.1 < p.2 ∧ p.2 ≤ n ∧ p.2 > 2 * p.1}
  let L := {p : ℕ × ℕ | 1 ≤ p.1 ∧ p.1 < p.2 ∧ p.2 ≤ n ∧ p.2 < 2 * p.1}
  Finset.card (Finset.filter (fun p => 1 ≤ p.1 ∧ p.1 < p.2 ∧ p.2 ≤ n ∧ p.2 > 2 * p.1) (Finset.range n ×ˢ Finset.range n)) =
  Finset.card (Finset.filter (fun p => 1 ≤ p.1 ∧ p.1 < p.2 ∧ p.2 ≤ n ∧ p.2 < 2 * p.1) (Finset.range n ×ˢ Finset.range n)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cardinality_equality_l559_55963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_equals_inverse_l559_55956

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (x - 3) / (x - 2)

-- Define the function g in terms of f and a
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f (x + a)

-- Theorem statement
theorem g_equals_inverse (a : ℝ) : 
  (∀ x, g a x = (Function.invFun (g a)) x) ↔ a = -1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_equals_inverse_l559_55956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_area_l559_55917

/-- Defines an isosceles triangle with two equal sides -/
def IsoscelesTriangle (a b c : ℝ) : Prop :=
  a = b ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a

/-- The area of an isosceles triangle with two sides of length 15 and a base of length 24 is 108 square units. -/
theorem isosceles_triangle_area : ∀ (a b c : ℝ), 
  a = 15 → b = 15 → c = 24 →
  IsoscelesTriangle a b c →
  (1/2 : ℝ) * c * Real.sqrt (a^2 - (c/2)^2) = 108 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_area_l559_55917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l559_55921

def prop_p (a : ℝ) : Prop :=
  ∃! x, x ∈ Set.Icc (-1 : ℝ) 1 ∧ a^2 * x^2 + a * x - 2 = 0

def prop_q (a : ℝ) : Prop :=
  ∃! x : ℝ, x^2 + 2*a*x + 2*a ≤ 0

theorem range_of_a : 
  {a : ℝ | ¬(prop_p a ∨ prop_q a)} = 
  {a : ℝ | a ≤ -2 ∨ (-1 < a ∧ a < 0) ∨ (0 < a ∧ a < 1) ∨ a > 2} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l559_55921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jude_buys_sixteen_vehicles_l559_55992

/-- The number of matchbox vehicles Jude buys given his bottle cap collection and trading rules. -/
def total_vehicles (initial_caps : ℕ) (car_cost : ℕ) (truck_cost : ℕ) (trucks_bought : ℕ) (car_spending_ratio : ℚ) : ℕ :=
  let remaining_caps := initial_caps - truck_cost * trucks_bought
  let caps_for_cars := (remaining_caps : ℚ) * car_spending_ratio
  let cars_bought := (caps_for_cars / car_cost).floor.toNat
  trucks_bought + cars_bought

/-- Theorem stating that Jude buys 16 matchbox vehicles in total. -/
theorem jude_buys_sixteen_vehicles :
  total_vehicles 100 5 6 10 (3/4) = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jude_buys_sixteen_vehicles_l559_55992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oblique_projection_properties_l559_55957

/-- Represents a geometric shape --/
inductive Shape
  | Triangle
  | Square
  | IsoscelesTrapezoid
  | Rhombus
  | Parallelogram

/-- Represents the oblique projection method --/
def obliqueProjection : Shape → Shape :=
  fun s => match s with
  | Shape.Triangle => Shape.Triangle
  | Shape.Square => Shape.Rhombus
  | Shape.IsoscelesTrapezoid => Shape.IsoscelesTrapezoid
  | Shape.Rhombus => Shape.Parallelogram
  | Shape.Parallelogram => Shape.Parallelogram

/-- The statement that the intuitive diagram of a triangle is definitely a triangle --/
def statement1 : Prop := ∀ t : Shape, t = Shape.Triangle → obliqueProjection t = Shape.Triangle

/-- The statement that the intuitive diagram of a square is definitely a rhombus --/
def statement2 : Prop := ∀ s : Shape, s = Shape.Square → obliqueProjection s = Shape.Rhombus

/-- The statement that the intuitive diagram of an isosceles trapezoid can be a parallelogram --/
def statement3 : Prop := ∃ t : Shape, t = Shape.IsoscelesTrapezoid ∧ obliqueProjection t = Shape.Parallelogram

/-- The statement that the intuitive diagram of a rhombus is definitely a rhombus --/
def statement4 : Prop := ∀ r : Shape, r = Shape.Rhombus → obliqueProjection r = Shape.Rhombus

/-- The theorem stating that only statement1 is true under the oblique projection method --/
theorem oblique_projection_properties :
  statement1 ∧ ¬statement2 ∧ ¬statement3 ∧ ¬statement4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_oblique_projection_properties_l559_55957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2x_minus_8cosx_range_l559_55912

theorem cos_2x_minus_8cosx_range :
  ∀ x : ℝ, ∃ y : ℝ, y = Real.cos (2 * x) - 8 * Real.cos x ∧ -7 ≤ y ∧ y ≤ 9 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2x_minus_8cosx_range_l559_55912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_non_divisible_by_three_l559_55904

theorem max_non_divisible_by_three (integers : Finset ℕ) : 
  integers.card = 7 → 
  (integers.prod id) % 3 = 0 → 
  (integers.filter (fun x => x % 3 ≠ 0)).card ≤ 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_non_divisible_by_three_l559_55904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_edge_bound_theorem_l559_55949

/-- A simple graph with n vertices and no cycles of length 4 -/
structure Graph (n : ℕ) where
  edges : Finset (Fin n × Fin n)
  symm : ∀ (i j : Fin n), (i, j) ∈ edges → (j, i) ∈ edges
  no_loops : ∀ (i : Fin n), (i, i) ∉ edges
  no_cycle4 : ∀ (a b c d : Fin n), a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a →
    ¬((a, b) ∈ edges ∧ (b, c) ∈ edges ∧ (c, d) ∈ edges ∧ (d, a) ∈ edges)

/-- The number of edges in a graph -/
def num_edges {n : ℕ} (G : Graph n) : ℕ := G.edges.card / 2

/-- Upper bound on the number of edges -/
noncomputable def edge_bound (n : ℕ) : ℕ := 
  ⌊(n : ℝ) / 4 * (1 + Real.sqrt (4 * n - 3))⌋.toNat

/-- Theorem: The number of edges in a graph with n vertices and no cycles of length 4
    is at most ⌊(n/4)(1+√(4n-3))⌋ -/
theorem edge_bound_theorem (n : ℕ) (G : Graph n) :
  num_edges G ≤ edge_bound n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_edge_bound_theorem_l559_55949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_150_degrees_l559_55970

theorem tan_150_degrees : 
  Real.tan (150 * π / 180) = -Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_150_degrees_l559_55970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monic_quadratic_polynomial_lower_bound_l559_55926

/-- A monic polynomial of degree 2 over the real numbers. -/
def MonicQuadraticPolynomial (a b : ℝ) : ℝ → ℝ := fun x ↦ x^2 + a*x + b

theorem monic_quadratic_polynomial_lower_bound
  (a b : ℝ) (x₁ x₂ : ℝ)
  (h_monic : MonicQuadraticPolynomial a b 1 ≥ MonicQuadraticPolynomial a b 0 + 3)
  (h_roots : x₁ * x₂ = b ∧ x₁ + x₂ = -a) :
  (x₁^2 + 1) * (x₂^2 + 1) ≥ 4 := by
  sorry

#check monic_quadratic_polynomial_lower_bound

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monic_quadratic_polynomial_lower_bound_l559_55926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_answer_is_C_l559_55937

inductive AnswerOption where
  | A
  | B
  | C
  | D

def correctAnswer : AnswerOption := AnswerOption.C

theorem correct_answer_is_C : correctAnswer = AnswerOption.C := by
  rfl

#check correct_answer_is_C

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_answer_is_C_l559_55937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2015_eq_neg_cos_l559_55920

open Real

noncomputable def f : ℕ → (ℝ → ℝ)
  | 0 => cos
  | (n + 1) => deriv (f n)

theorem f_2015_eq_neg_cos : f 2015 = λ x => -cos x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2015_eq_neg_cos_l559_55920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_two_l559_55988

noncomputable section

-- Define the sales volume function
noncomputable def sales_volume (x : ℝ) : ℝ :=
  if 1 < x ∧ x ≤ 3 then (x - 4)^2 + 6 / (x - 1)
  else if 3 < x ∧ x ≤ 5 then -x + 7
  else 0

-- Define the profit function
noncomputable def profit (x : ℝ) : ℝ := (sales_volume x) * (x - 1)

-- Theorem statement
theorem max_profit_at_two :
  ∀ x : ℝ, 1 < x ∧ x ≤ 5 → profit x ≤ profit 2 := by
  sorry

-- Additional conditions to ensure the problem is well-defined
axiom sales_volume_at_three : sales_volume 3 = 4
axiom min_sales_volume : ∀ x : ℝ, 3 < x ∧ x ≤ 5 → sales_volume x ≥ 2

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_two_l559_55988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_specific_triangle_l559_55975

/-- The radius of the inscribed circle of a triangle -/
noncomputable def inscribed_circle_radius (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  area / s

/-- Theorem: The radius of the inscribed circle in triangle ABC with side lengths 15, 20, and 25 is 5 -/
theorem inscribed_circle_radius_specific_triangle :
  inscribed_circle_radius 15 20 25 = 5 := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_specific_triangle_l559_55975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_sum_greater_than_two_l559_55941

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2*a*Real.log x + x^2 - 2*(a+1)*x

-- State the theorem
theorem zeros_sum_greater_than_two (a : ℝ) (x₁ x₂ : ℝ) 
  (ha : a < 0) 
  (ha' : -1/2 < a)
  (hx : 0 < x₁ ∧ x₁ < x₂) 
  (hf₁ : f a x₁ = 0) 
  (hf₂ : f a x₂ = 0) : 
  x₁ + x₂ > 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_sum_greater_than_two_l559_55941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_f_defined_l559_55976

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := 1 / x

-- State the theorem
theorem f_f_defined (x : ℝ) : 
  (∃ y : ℝ, f (f y) = y) ↔ x ≠ 0 :=
by
  -- We'll use sorry to skip the proof for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_f_defined_l559_55976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l559_55965

-- Define the functions f and g
def f (m x : ℝ) : ℝ := m * (x - 2 * m) * (x + m + 3)

noncomputable def g (x : ℝ) : ℝ := 2^x - 2

-- State the theorem
theorem range_of_m :
  (∀ m : ℝ,
    (∀ x : ℝ, f m x < 0 ∨ g x < 0) ∧
    (∃ x : ℝ, x < -4 ∧ f m x * g x < 0)) ↔
  (∀ m : ℝ, -4 < m ∧ m < -2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l559_55965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_output_for_eight_l559_55997

/-- The function representing the input-output relationship -/
noncomputable def f (a : ℝ) : ℝ := (2 * a) / (a^2 + 1)

/-- Theorem stating that f(8) = 16/65 -/
theorem output_for_eight : f 8 = 16 / 65 := by
  -- Unfold the definition of f
  unfold f
  -- Simplify the numerator and denominator
  simp [pow_two]
  -- Perform the division
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_output_for_eight_l559_55997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangentPointsDistance_eq_6_4_l559_55935

/-- An isosceles trapezoid with an inscribed circle -/
structure InscribedCircleTrapezoid where
  /-- Radius of the inscribed circle -/
  r : ℝ
  /-- Length of the shorter base of the trapezoid -/
  b : ℝ
  /-- The trapezoid is isosceles -/
  isIsosceles : Bool
  /-- The circle is inscribed in the trapezoid -/
  isInscribed : Bool
  /-- Condition: radius is 4 -/
  h_r : r = 4
  /-- Condition: shorter base is 4 -/
  h_b : b = 4
  /-- Condition: trapezoid is isosceles -/
  h_isosceles : isIsosceles = true
  /-- Condition: circle is inscribed -/
  h_inscribed : isInscribed = true

/-- The distance between the points where the circle touches the legs of the trapezoid -/
noncomputable def tangentPointsDistance (t : InscribedCircleTrapezoid) : ℝ := 
  2 * t.r * Real.sqrt 3 / 2

/-- Theorem: The distance between the points where the circle touches the legs of the trapezoid is 6.4 -/
theorem tangentPointsDistance_eq_6_4 (t : InscribedCircleTrapezoid) : 
  tangentPointsDistance t = 6.4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangentPointsDistance_eq_6_4_l559_55935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sqrt_2x_minus_x_squared_minus_x_l559_55913

theorem integral_sqrt_2x_minus_x_squared_minus_x : 
  (∫ (x : ℝ) in Set.Icc 0 1, (Real.sqrt (2 * x - x^2) - x)) = (Real.pi - 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sqrt_2x_minus_x_squared_minus_x_l559_55913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_theorem_l559_55927

-- Define the triangle ABC
def triangle (A B C : Real) (a b c : Real) : Prop :=
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C

-- Theorem statement
theorem triangle_area_theorem (A B C : Real) (a b c : Real) :
  triangle A B C a b c →
  Real.sin A + Real.sqrt 3 * Real.cos A = 2 →
  a = 2 →
  B = Real.pi / 4 →
  A = Real.pi / 6 ∧ 
  (1 / 2) * a * b * Real.sin C = 1 + Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_theorem_l559_55927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_algorithm_correct_l559_55924

def algorithm (x : Int) : Int :=
  if x < 0 then x + 1
  else if x = 0 then 0
  else x

theorem algorithm_correct (x : Int) :
  algorithm x = 
    if x < 0 then x + 1
    else if x = 0 then 0
    else x := by
  rfl

#eval algorithm (-2)
#eval algorithm 0
#eval algorithm 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_algorithm_correct_l559_55924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_max_volume_and_area_l559_55933

/-- Represents a tetrahedron with two skew edges of lengths a and b, 
    and the segment connecting their midpoints of length k -/
structure Tetrahedron where
  a : ℝ
  b : ℝ
  k : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  k_pos : 0 < k

/-- The maximum volume of the tetrahedron -/
noncomputable def max_volume (t : Tetrahedron) : ℝ :=
  (t.a * t.b * t.k) / 6

/-- The maximum surface area of the tetrahedron -/
noncomputable def max_surface_area (t : Tetrahedron) : ℝ :=
  t.a * Real.sqrt (t.k^2 + t.b^2/4) + t.b * Real.sqrt (t.k^2 + t.a^2/4)

theorem tetrahedron_max_volume_and_area (t : Tetrahedron) :
  (∀ (v : ℝ), v ≤ max_volume t) ∧
  (∀ (s : ℝ), s ≤ max_surface_area t) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_max_volume_and_area_l559_55933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_absolute_difference_l559_55953

theorem smallest_absolute_difference : 
  ∃ (k₀ l₀ : ℕ), |Int.ofNat (36^k₀) - Int.ofNat (5^l₀)| = 11 ∧ 
  ∀ (k l : ℕ), |Int.ofNat (36^k) - Int.ofNat (5^l)| ≥ 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_absolute_difference_l559_55953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_minus_q_equals_sum_of_squares_l559_55985

/-- Represents a four-digit number --/
structure FourDigitNumber where
  a : Nat
  b : Nat
  c : Nat
  d : Nat
  a_valid : a ≥ 1 ∧ a ≤ 9
  b_valid : b ≥ 0 ∧ b ≤ 9
  c_valid : c ≥ 0 ∧ c ≤ 9
  d_valid : d ≥ 0 ∧ d ≤ 9

/-- Defines P type numbers --/
def is_p_type (n : FourDigitNumber) : Prop :=
  n.a > n.b ∧ n.b < n.c ∧ n.c > n.d

/-- Defines Q type numbers --/
def is_q_type (n : FourDigitNumber) : Prop :=
  n.a < n.b ∧ n.b > n.c ∧ n.c < n.d

/-- Counts the number of P type numbers --/
def count_p_type : Nat :=
  sorry

/-- Counts the number of Q type numbers --/
def count_q_type : Nat :=
  sorry

/-- Sum of squares of first n natural numbers --/
def sum_of_squares (n : Nat) : Nat :=
  List.range n |>.map (fun k => (k + 1) ^ 2) |>.sum

theorem p_minus_q_equals_sum_of_squares :
  count_p_type - count_q_type = sum_of_squares 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_minus_q_equals_sum_of_squares_l559_55985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_solution_l559_55959

noncomputable def frac (x : ℝ) : ℝ := x - ⌊x⌋

theorem arithmetic_sequence_solution :
  ∃! (x : ℝ), x ≠ 0 ∧
  ∃ (d : ℝ), d ≠ 0 ∧
  frac x = (⌊x⌋ - d) ∧
  ⌊x⌋ = (frac x + d) ∧
  x = (⌊x⌋ + d) ∧
  (x + frac x) = (x + d) ∧
  x = (3/2 : ℝ) := by
  sorry

#check arithmetic_sequence_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_solution_l559_55959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_difference_magnitude_l559_55936

noncomputable def a : ℝ × ℝ := (Real.cos (10 * Real.pi / 180), Real.sin (10 * Real.pi / 180))
noncomputable def b : ℝ × ℝ := (Real.cos (70 * Real.pi / 180), Real.sin (70 * Real.pi / 180))

theorem vector_difference_magnitude : 
  Real.sqrt ((a.1 - 2 * b.1)^2 + (a.2 - 2 * b.2)^2) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_difference_magnitude_l559_55936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_theorem_l559_55962

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x + Real.sin x * Real.cos x

noncomputable def g (m : ℝ) (x : ℝ) : ℝ := m * Real.cos (2 * x - Real.pi / 6) - 2 * m + 3

theorem m_range_theorem (m : ℝ) (h_m : m > 0) :
  (∀ x₁ ∈ Set.Icc 0 (Real.pi / 4), ∃ x₂ ∈ Set.Icc 0 (Real.pi / 4), g m x₁ = f x₂) ↔
  m ∈ Set.Icc (5 / 2 - Real.sqrt 2) (4 / 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_theorem_l559_55962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_profit_percentage_approx_l559_55961

-- Define the initial conditions
def initial_cost : ℝ := 100
def profit_percentage : ℝ := 1.5
def ingredient_cost_percentage : ℝ := 0.35
def ingredient_cost_increase : ℝ := 0.12
def labor_cost_increase : ℝ := 0.05

-- Define the selling price
def selling_price : ℝ := initial_cost * (1 + profit_percentage)

-- Define the new costs after increases
def new_ingredient_cost : ℝ := initial_cost * ingredient_cost_percentage * (1 + ingredient_cost_increase)
def new_labor_cost : ℝ := initial_cost * (1 - ingredient_cost_percentage) * (1 + labor_cost_increase)

-- Define the new total cost
def new_total_cost : ℝ := new_ingredient_cost + new_labor_cost

-- Define the new profit
def new_profit : ℝ := selling_price - new_total_cost

-- Theorem to prove
theorem new_profit_percentage_approx :
  ∃ ε > 0, abs ((new_profit / selling_price) * 100 - 57.02) < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_profit_percentage_approx_l559_55961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_upper_bound_implies_m_range_l559_55960

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then -x^2 + x else Real.log x / Real.log (1/3)

-- State the theorem
theorem f_upper_bound_implies_m_range (m : ℝ) :
  (∀ x : ℝ, f x ≤ 5/4 * m - m^2) → m ∈ Set.Icc (1/4 : ℝ) 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_upper_bound_implies_m_range_l559_55960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_janes_height_l559_55911

/-- Given a tree and a person (Jane) casting shadows, this theorem calculates Jane's height
    using the principle of similar triangles. -/
theorem janes_height
  (tree_height : ℝ)
  (tree_shadow : ℝ)
  (jane_shadow : ℝ)
  (h1 : tree_height = 30)
  (h2 : tree_shadow = 10)
  (h3 : jane_shadow = 0.5)
  : tree_height / tree_shadow * jane_shadow = 1.5 := by
  -- Substitute the known values
  rw [h1, h2, h3]
  -- Simplify the expression
  norm_num
  -- The proof is complete
  done

#check janes_height

end NUMINAMATH_CALUDE_ERRORFEEDBACK_janes_height_l559_55911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_detectors_for_battleship_l559_55954

/-- Represents a detector on the playing field -/
structure Detector where
  position : Nat

/-- Represents a ship on the playing field -/
structure Ship where
  start_position : Nat
  length : Nat

/-- The playing field for the game -/
structure PlayingField where
  width : Nat
  height : Nat
  detectors : List Detector
  ship : Ship

/-- Checks if a detector is triggered by the ship -/
def is_detector_triggered (field : PlayingField) (d : Detector) : Bool :=
  d.position ≥ field.ship.start_position ∧ d.position < field.ship.start_position + field.ship.length

/-- Checks if the ship's position can be uniquely determined -/
def can_determine_ship_position (field : PlayingField) : Bool :=
  sorry

/-- The main theorem to prove -/
theorem min_detectors_for_battleship :
  ∀ (field : PlayingField),
    field.width = 1 ∧
    field.height = 203 ∧
    field.ship.length = 2 →
    (∀ (ship_position : Nat),
      ∃ (detectors : List Detector),
        detectors.length = 134 ∧
        can_determine_ship_position { width := field.width,
                                      height := field.height,
                                      ship := { start_position := ship_position, length := 2 },
                                      detectors := detectors }) ∧
    (∀ (detectors : List Detector),
      detectors.length < 134 →
      ∃ (ship_position1 ship_position2 : Nat),
        ship_position1 ≠ ship_position2 ∧
        ¬can_determine_ship_position { width := field.width,
                                       height := field.height,
                                       ship := { start_position := ship_position1, length := 2 },
                                       detectors := detectors } ∧
        ¬can_determine_ship_position { width := field.width,
                                       height := field.height,
                                       ship := { start_position := ship_position2, length := 2 },
                                       detectors := detectors }) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_detectors_for_battleship_l559_55954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magical_stack_size_l559_55955

/-- Represents a stack of cards -/
structure CardStack :=
  (n : ℕ)  -- Half the total number of cards

/-- Represents the position of a card in the stack -/
def CardPosition := ℕ

/-- Returns the new position of a card after restacking -/
def new_position (stack : CardStack) (original_pos : ℕ) : ℕ :=
  if original_pos ≤ stack.n then
    2 * (stack.n - original_pos + 1)
  else
    2 * (original_pos - stack.n) - 1

/-- Defines a magical stack -/
def is_magical (stack : CardStack) : Prop :=
  ∃ (card_a : ℕ) (card_b : ℕ),
    card_a ≤ stack.n ∧ card_b > stack.n ∧
    new_position stack card_a = card_a ∧
    new_position stack card_b = card_b

/-- Main theorem: A magical stack where card 131 retains its position has 130 cards -/
theorem magical_stack_size :
  ∀ (stack : CardStack),
    is_magical stack →
    new_position stack 131 = 131 →
    2 * stack.n = 130 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magical_stack_size_l559_55955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_multiple_2_or_5_in_50_l559_55909

def is_multiple_of_2_or_5 (n : ℕ) : Bool := n % 2 = 0 || n % 5 = 0

def count_multiples (n : ℕ) : ℕ := (Finset.range n).filter (fun x => is_multiple_of_2_or_5 (x + 1)) |>.card

theorem probability_multiple_2_or_5_in_50 : 
  (count_multiples 50 : ℚ) / 50 = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_multiple_2_or_5_in_50_l559_55909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_range_equivalence_l559_55994

-- Define the function f as noncomputable
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (Real.sqrt (a * x^2 + 2*x - 1))

-- State the theorem
theorem range_of_a (a : ℝ) : 
  (∀ y : ℝ, ∃ x : ℝ, f a x = y) → a ∈ Set.Ici (0 : ℝ) := by
  sorry

-- Define the set of valid 'a' values
def valid_a : Set ℝ := Set.Ici (0 : ℝ)

-- State the equivalence theorem
theorem range_equivalence : 
  ∀ a : ℝ, (∀ y : ℝ, ∃ x : ℝ, f a x = y) ↔ a ∈ valid_a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_range_equivalence_l559_55994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_andrea_carpet_coverage_l559_55910

/-- The percentage of Andrea's living room floor covered by the carpet -/
noncomputable def carpet_coverage_percentage (carpet_length : ℝ) (carpet_width : ℝ) (room_area : ℝ) : ℝ :=
  (carpet_length * carpet_width / room_area) * 100

/-- Theorem stating the percentage of Andrea's living room floor covered by the carpet -/
theorem andrea_carpet_coverage :
  carpet_coverage_percentage 4 9 180 = 20 := by
  -- Unfold the definition of carpet_coverage_percentage
  unfold carpet_coverage_percentage
  -- Simplify the arithmetic
  simp [mul_div_assoc, mul_comm]
  -- Check that the result is equal to 20
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_andrea_carpet_coverage_l559_55910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_integer_solution_l559_55952

theorem no_integer_solution : ¬∃ (n : ℤ), 
  let x : ℤ := 11  -- smaller part
  let y : ℤ := 24 - x  -- larger part
  7 * x + n * y = 146 ∧ x + y = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_integer_solution_l559_55952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_semicircles_area_ratio_l559_55919

/-- Given a circle with radius r and a rectangle inscribed across its center
    with width equal to r and length 4r, the ratio of the area of the rectangle
    to the combined area of the semicircles is 4:π -/
theorem rectangle_semicircles_area_ratio (r : ℝ) (hr : r > 0) :
  (4 * r * r) / (π * r^2) = 4 / π := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_semicircles_area_ratio_l559_55919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prove_inequality_l559_55922

-- Define the variables and conditions
theorem prove_inequality (m : ℝ) (a b : ℝ) 
  (h1 : (9 : ℝ)^m = 10)
  (h2 : a = (10 : ℝ)^m - 11)
  (h3 : b = (8 : ℝ)^m - 9) :
  a > 0 ∧ 0 > b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prove_inequality_l559_55922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_spade_result_l559_55925

/-- The ♠ operation for positive real numbers -/
noncomputable def spade (x y : ℝ) : ℝ := x - 1 / y

/-- Theorem stating the result of nested spade operations -/
theorem nested_spade_result : 
  spade 3 (spade 3 (spade 3 3)) = 55 / 21 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_spade_result_l559_55925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_sum_l559_55939

/-- In a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if cos A = 1/3, a = √3, and bc = 3/2, then b + c = √7 -/
theorem triangle_side_sum (a b c : ℝ) (A : ℝ) :
  Real.cos A = 1/3 → a = Real.sqrt 3 → b * c = 3/2 → b + c = Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_sum_l559_55939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_implies_a_l559_55905

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 - a * x^2 + 1

-- State the theorem
theorem max_value_implies_a (a : ℝ) :
  (∀ x : ℝ, f a x ≤ f a (-4)) → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_implies_a_l559_55905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_cards_in_box_l559_55947

theorem original_cards_in_box (cards_added : ℕ) (fraction_removed : ℚ) (final_cards : ℕ) :
  cards_added = 48 →
  fraction_removed = 1 / 6 →
  final_cards = 83 →
  ∃ original_cards : ℕ, 
    original_cards + cards_added - (fraction_removed * ↑cards_added).floor = final_cards ∧
    original_cards = 43 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_cards_in_box_l559_55947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lottery_probability_and_income_l559_55916

def lottery_size : ℕ := 90
def drawn_numbers : ℕ := 5
def payout_multiple : ℕ := 14

def probability_hit_one (n m : ℕ) : ℚ :=
  (Nat.choose m 1 * Nat.choose (n - m) (m - 1)) / Nat.choose n m

def state_income_percentage (prob : ℚ) (payout : ℕ) : ℚ :=
  (1 - payout * prob) * 100

theorem lottery_probability_and_income :
  probability_hit_one lottery_size drawn_numbers = 1 / 18 ∧
  state_income_percentage (probability_hit_one lottery_size drawn_numbers) payout_multiple = 2222 / 100 := by
  sorry

#eval probability_hit_one lottery_size drawn_numbers
#eval state_income_percentage (probability_hit_one lottery_size drawn_numbers) payout_multiple

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lottery_probability_and_income_l559_55916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_term_exceeding_2016_l559_55983

def sequenceA : ℕ → ℕ
  | 0 => 12
  | 1 => 19
  | (n + 2) => 
    let prev1 := sequenceA n
    let prev2 := sequenceA (n + 1)
    if (prev1 + prev2) % 2 = 1 then 
      prev1 + prev2
    else 
      max prev1 prev2 - min prev1 prev2

theorem first_term_exceeding_2016 : 
  (∀ k < 504, sequenceA k ≤ 2016) ∧ sequenceA 503 > 2016 := by
  sorry

#eval sequenceA 503

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_term_exceeding_2016_l559_55983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intervals_of_decrease_l559_55915

noncomputable def f (x : ℝ) : ℝ := Real.sin (-2 * x + Real.pi / 2)

theorem intervals_of_decrease (k : ℤ) :
  ∀ x ∈ Set.Icc (↑k * Real.pi) (↑k * Real.pi + Real.pi / 2),
    ∀ y ∈ Set.Icc (↑k * Real.pi) (↑k * Real.pi + Real.pi / 2),
      x ≤ y → f y ≤ f x :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intervals_of_decrease_l559_55915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_salary_increase_l559_55972

/-- Calculates the compound percentage increase given an initial salary and a list of raises --/
noncomputable def compound_percentage_increase (initial_salary : ℝ) (raises : List ℝ) : ℝ :=
  let final_salary := raises.foldl (fun acc r => acc * (1 + r / 100)) initial_salary
  (final_salary - initial_salary) / initial_salary * 100

/-- The compound percentage increase for John's salary --/
theorem johns_salary_increase :
  let initial_salary : ℝ := 60
  let raises : List ℝ := [10, 15, 12, 8]
  abs (compound_percentage_increase initial_salary raises - 53.01) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_salary_increase_l559_55972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l559_55986

noncomputable def slope_angle : ℝ := 45 * Real.pi / 180
def y_intercept : ℝ := 2

theorem line_equation (x y : ℝ) :
  (Real.tan slope_angle = 1) →
  (y = Real.tan slope_angle * x + y_intercept) ↔
  (y = x + 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l559_55986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_v_l559_55995

-- Define the function v(x)
noncomputable def v (x : ℝ) : ℝ := 1 / Real.sqrt (2 * x - 4)

-- State the theorem about the domain of v(x)
theorem domain_of_v :
  {x : ℝ | ∃ y, v x = y} = {x : ℝ | x > 2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_v_l559_55995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l559_55914

noncomputable section

-- Define the bounds of x
def x_lower : ℝ := 1
def x_upper : ℝ := 5

-- Define the functions that bound the region
noncomputable def f (x : ℝ) : ℝ := 4 - x
noncomputable def g (x : ℝ) : ℝ := (1/2) * (x - 2)^2 - 1

-- State the theorem
theorem area_of_region : 
  ∫ x in x_lower..x_upper, f x - g x = 4 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l559_55914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_repeating_decimal_nine_l559_55966

/-- The repeating decimal 0.4̄ as a real number -/
noncomputable def repeating_decimal : ℚ := 4/9

/-- Theorem: The product of 0.4̄ and 9 is equal to 4 -/
theorem product_repeating_decimal_nine : (repeating_decimal : ℝ) * 9 = 4 := by
  -- Convert the rational number to a real number
  have h1 : (4/9 : ℚ) = repeating_decimal := rfl
  -- Use the fact that (4/9) * 9 = 4
  have h2 : ((4/9 : ℚ) : ℝ) * 9 = 4 := by norm_num
  -- Rewrite using h1 and h2
  rw [← h1, h2]


end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_repeating_decimal_nine_l559_55966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_side_length_l559_55980

noncomputable def circle_radius : ℝ := 100 * Real.sqrt 2

noncomputable def known_side_length : ℝ := 100 * Real.sqrt 3

structure InscribedQuadrilateral :=
  (side1 : ℝ)
  (side2 : ℝ)
  (side3 : ℝ)
  (side4 : ℝ)
  (inscribed : Prop)
  (three_sides_equal : side1 = known_side_length ∧ side2 = known_side_length ∧ side3 = known_side_length)

theorem fourth_side_length (q : InscribedQuadrilateral) :
  q.inscribed → q.side4 = known_side_length :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_side_length_l559_55980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calendar_column_sum_divisibility_calendar_column_sum_not_four_l559_55902

theorem calendar_column_sum_divisibility (x : ℕ) : 
  ∃ k : ℕ, x + (x + 7) + (x + 14) = 3 * k :=
by
  use x + 7
  ring

theorem calendar_column_sum_not_four : 
  ∀ x : ℕ, x + (x + 7) + (x + 14) ≠ 4 :=
by
  intro x
  have h : ∃ k : ℕ, x + (x + 7) + (x + 14) = 3 * k := calendar_column_sum_divisibility x
  cases h with | intro k hk =>
  intro contra
  rw [contra] at hk
  have : 3 ∣ 4 := Dvd.intro k hk.symm
  exact absurd this (by norm_num)

#check calendar_column_sum_not_four

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calendar_column_sum_divisibility_calendar_column_sum_not_four_l559_55902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_x_value_l559_55981

-- Define the vectors
def a : Fin 3 → ℝ := ![2, -1, 3]
def b (x : ℝ) : Fin 3 → ℝ := ![x, 2, -6]

-- Define parallel vectors
def parallel (v w : Fin 3 → ℝ) : Prop :=
  ∃ (k : ℝ), ∀ i, v i = k * w i

-- Theorem statement
theorem parallel_vectors_x_value :
  ∀ x : ℝ, parallel a (b x) → x = -4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_x_value_l559_55981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_on_y_axis_l559_55948

-- Define the points C and D
def C : ℝ × ℝ := (-3, 0)
def D : ℝ × ℝ := (-2, 5)

-- Define the point P on the y-axis
def P : ℝ → ℝ × ℝ := λ y ↦ (0, y)

-- Define the distance function between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem equidistant_point_on_y_axis :
  ∃ y : ℝ, distance (P y) C = distance (P y) D ∧ y = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_on_y_axis_l559_55948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equality_l559_55901

theorem expression_equality : (243 : ℝ) ^ (-(2 : ℝ) ^ (-(3 : ℝ))) = 1 / (3 : ℝ) ^ (5/8) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equality_l559_55901
