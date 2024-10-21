import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_triangle_centers_l728_72872

structure Circle where
  center : ℝ × ℝ
  radius : ℝ

structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

def isOnCircle (p : ℝ × ℝ) (circle : Circle) : Prop :=
  let (x, y) := p
  let (cx, cy) := circle.center
  (x - cx)^2 + (y - cy)^2 = circle.radius^2

noncomputable def orthocenter (t : Triangle) : ℝ × ℝ := sorry

noncomputable def incenter (t : Triangle) : ℝ × ℝ := sorry

def isSymmetricCircle (c1 c2 : Circle) (line : Set (ℝ × ℝ)) : Prop := sorry

theorem locus_of_triangle_centers 
  (circle : Circle) (A B : ℝ × ℝ) 
  (hA : isOnCircle A circle) (hB : isOnCircle B circle) :
  ∃ (orthoLocus : Circle) (incenterLocus : Set (ℝ × ℝ)),
    (∀ C, isOnCircle C circle → C ≠ A → C ≠ B →
      let t := Triangle.mk A B C
      (isOnCircle (orthocenter t) orthoLocus ∧ orthocenter t ≠ A ∧ orthocenter t ≠ B) ∧
      (incenter t ∈ incenterLocus ∧ incenter t ≠ A ∧ incenter t ≠ B)) ∧
    isSymmetricCircle orthoLocus circle {p | p = A ∨ p = B} ∧
    (∃ arc1 arc2 : Set (ℝ × ℝ), 
      incenterLocus = arc1 ∪ arc2 ∧ 
      arc1 ⊆ {p | isOnCircle p circle} ∧
      arc2 ⊆ {p | isOnCircle p circle}) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_triangle_centers_l728_72872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_value_l728_72845

-- Define the function f
noncomputable def f (x : ℝ) (b : ℝ) : ℝ := 
  if x ≥ 0 then 4 * x + b else -(4 * (-x) + b)

-- State the theorem
theorem odd_function_value (b : ℝ) :
  (∀ x, f (-x) b = -(f x b)) →  -- f is an odd function
  f (-1) b = -3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_value_l728_72845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_100_greater_than_14_l728_72878

noncomputable def a : ℕ → ℝ
  | 0 => 1  -- Add case for 0
  | 1 => 1
  | n + 2 => a (n + 1) + 1 / a (n + 1)

theorem a_100_greater_than_14 : a 100 > 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_100_greater_than_14_l728_72878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_with_tan_2_l728_72885

theorem cos_double_angle_with_tan_2 (α : ℝ) (h : Real.tan α = 2) : 
  Real.cos (2 * α) = -3/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_with_tan_2_l728_72885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_strategy_always_succeeds_l728_72839

/-- Represents a box that may contain a coin -/
inductive Box
| Empty
| Coin

/-- Represents the state of all boxes -/
def BoxState := Fin 12 → Box

/-- Checks if a given set of indices contains both coins -/
def containsBothCoins (s : BoxState) (indices : Finset (Fin 12)) : Prop :=
  ∃ i j, i ∈ indices ∧ j ∈ indices ∧ i ≠ j ∧ s i = Box.Coin ∧ s j = Box.Coin

/-- The strategy function that selects 4 boxes to open -/
def strategy (k : Fin 12) : Finset (Fin 12) :=
  {(k + 1), (k + 2), (k + 5), (k + 7)}

/-- The main theorem stating that the strategy always works -/
theorem strategy_always_succeeds :
  ∀ (s : BoxState),
  (∃ i j : Fin 12, i ≠ j ∧ s i = Box.Coin ∧ s j = Box.Coin) →
  ∃ k : Fin 12, s k = Box.Empty ∧ containsBothCoins s (strategy k) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_strategy_always_succeeds_l728_72839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_divisibility_l728_72830

/-- An arithmetic sequence with positive integer terms -/
def ArithmeticSequence (a : ℕ → ℕ) : Prop :=
  ∃ (x c : ℕ), ∀ n, a n = x + n * c

/-- The sum of the first n terms of a sequence -/
def SequenceSum (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  Finset.sum (Finset.range n) a

/-- The statement to be proved -/
theorem arithmetic_sequence_sum_divisibility :
  ∀ (a : ℕ → ℕ), ArithmeticSequence a →
    (∀ k : ℕ, k > 15 → ¬(k ∣ SequenceSum a 15)) ∧
    (15 ∣ SequenceSum a 15) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_divisibility_l728_72830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_extrema_is_eight_l728_72817

/-- Given a > 0 and a ≠ 1, prove that for the function f(x) defined on [-1/4, 1/4],
    the sum of its maximum and minimum values is 8 -/
theorem sum_of_extrema_is_eight (a : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) :
  let f : ℝ → ℝ := λ x => (5 * a^x + 3) / (a^x + 1) + 4 * (Real.log a * Real.log ((1 + x) / (1 - x)))
  ∃ (max min : ℝ), (∀ x ∈ Set.Icc (-1/4 : ℝ) (1/4 : ℝ), f x ≤ max ∧ min ≤ f x) ∧
                   max + min = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_extrema_is_eight_l728_72817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_standard_equation_l728_72879

/-- Given an ellipse with vertices at (4, 0) and (0, 3), and foci on either the x-axis or y-axis,
    prove that its standard equation is either x^2/25 + y^2/9 = 1 or x^2/16 + y^2/25 = 1. -/
theorem ellipse_standard_equation 
  (E : Set (ℝ × ℝ)) -- The ellipse
  (v1 v2 : ℝ × ℝ) -- Vertices of the ellipse
  (h_vertices : v1 = (4, 0) ∧ v2 = (0, 3)) -- Vertices are at (4, 0) and (0, 3)
  (h_foci : (∃ c : ℝ, (c, 0) ∈ E ∧ (-c, 0) ∈ E) ∨ (∃ c : ℝ, (0, c) ∈ E ∧ (0, -c) ∈ E)) -- Foci are on x-axis or y-axis
  : (∀ (x y : ℝ), (x, y) ∈ E ↔ x^2/25 + y^2/9 = 1) ∨
    (∀ (x y : ℝ), (x, y) ∈ E ↔ x^2/16 + y^2/25 = 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_standard_equation_l728_72879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_level_is_two_feet_l728_72886

/-- Represents a cubical water tank -/
structure CubicalTank where
  capacity : ℝ
  waterVolume : ℝ
  fillRatio : ℝ

/-- Calculates the water level in a cubical tank -/
noncomputable def waterLevel (tank : CubicalTank) : ℝ :=
  let sideLength := (tank.capacity) ^ (1/3)
  let baseArea := sideLength ^ 2
  tank.waterVolume / baseArea

/-- Theorem: The water level in the given cubical tank is 2 feet -/
theorem water_level_is_two_feet 
  (tank : CubicalTank) 
  (h1 : tank.waterVolume = 50)
  (h2 : tank.fillRatio = 0.4)
  (h3 : tank.capacity = tank.waterVolume / tank.fillRatio) :
  waterLevel tank = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_level_is_two_feet_l728_72886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_circle_center_l728_72896

-- Define the fixed circle M
def circle_M (p : ℝ × ℝ) : Prop := (p.1 - 2)^2 + p.2^2 = 64

-- Define point A
def point_A : ℝ × ℝ := (-2, 0)

-- Define the property of being internally tangent
def internally_tangent (C M : (ℝ × ℝ) → Prop) : Prop :=
  ∃ (p : ℝ × ℝ), C p ∧ M p ∧
  ∀ (q : ℝ × ℝ), C q → M q

-- Define the moving circle C
def circle_C (center : ℝ × ℝ) (p : ℝ × ℝ) : Prop :=
  (p.1 - center.1)^2 + (p.2 - center.2)^2 = (p.1 - point_A.1)^2 + (p.2 - point_A.2)^2

-- Theorem statement
theorem trajectory_of_circle_center :
  ∀ (center : ℝ × ℝ),
  (circle_C center point_A) ∧
  (internally_tangent (circle_C center) circle_M) →
  center.1^2 / 16 + center.2^2 / 12 = 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_circle_center_l728_72896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wick_count_is_24_l728_72815

/-- The length of the string in inches -/
def string_length : ℕ := 25 * 12

/-- The lengths of the different wick sizes in inches -/
def wick_sizes : List ℕ := [6, 9, 12]

/-- The function to calculate the total number of wicks -/
def total_wicks (total_length : ℕ) (sizes : List ℕ) : ℕ :=
  (total_length / (sizes.foldl Nat.lcm 1)) * sizes.length

/-- Theorem stating that the total number of wicks is 24 -/
theorem wick_count_is_24 : 
  total_wicks string_length wick_sizes = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wick_count_is_24_l728_72815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_tangents_theorem_l728_72882

/-- A parabola defined by the equation x^2 = 4y -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 = 4 * p.2}

/-- The tangent line to the parabola at a point -/
def TangentLine (p : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {q : ℝ × ℝ | q.2 - p.2 = (p.1 / 2) * (q.1 - p.1)}

theorem perpendicular_tangents_theorem 
  (A B : ℝ × ℝ) 
  (hA : A ∈ Parabola) 
  (hB : B ∈ Parabola)
  (hPerp : (A.1 / 2) * (B.1 / 2) = -1)
  (a b : ℝ)
  (ha : a > 0) 
  (hb : b > 0)
  (hLen : a^2 + b^2 = (A.1 - B.1)^2 + (A.2 - B.2)^2) :
  a^4 * b^4 = (a^2 + b^2)^3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_tangents_theorem_l728_72882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2BC_plus_sin_squared_half_BC_max_area_l728_72847

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  acute : A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = Real.pi
  sides : a > 0 ∧ b > 0 ∧ c > 0
  law_of_sines : a / (Real.sin A) = b / (Real.sin B) ∧ b / (Real.sin B) = c / (Real.sin C)

-- Define the specific triangle with given conditions
noncomputable def special_triangle : Triangle where
  A := Real.arcsin ((4 * Real.sqrt 5) / 9)
  B := sorry
  C := sorry
  a := 4
  b := sorry
  c := sorry
  acute := by sorry
  sides := by sorry
  law_of_sines := by sorry

-- Theorem for part 1
theorem sin_2BC_plus_sin_squared_half_BC (t : Triangle) (h : t = special_triangle) :
  Real.sin (2 * (t.B + t.C)) + (Real.sin ((t.B + t.C) / 2))^2 = (45 - 8 * Real.sqrt 5) / 81 := by
  sorry

-- Theorem for part 2
theorem max_area (t : Triangle) (h : t = special_triangle) :
  ∃ (S : Real), S = 2 * Real.sqrt 5 ∧ 
  ∀ (S' : Real), S' = (1/2) * t.b * t.c * Real.sin t.A → S' ≤ S := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2BC_plus_sin_squared_half_BC_max_area_l728_72847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_on_unit_circle_l728_72867

theorem square_on_unit_circle (S : Set (ℝ × ℝ)) (c : ℝ × ℝ) (r : ℝ) :
  let circle := {p : ℝ × ℝ | (p.1 - c.1)^2 + (p.2 - c.2)^2 = r^2}
  r = 1 →
  (∃ (A B C D : ℝ × ℝ), S = {A, B, C, D} ∧
    A ∈ circle ∧ B ∈ circle ∧
    ((C.1 - D.1)^2 + (C.2 - D.2)^2 = r^2) ∧
    ((A.1 - B.1)^2 + (A.2 - B.2)^2 = (B.1 - C.1)^2 + (B.2 - C.2)^2) ∧
    ((B.1 - C.1)^2 + (B.2 - C.2)^2 = (C.1 - D.1)^2 + (C.2 - D.2)^2) ∧
    ((C.1 - D.1)^2 + (C.2 - D.2)^2 = (D.1 - A.1)^2 + (D.2 - A.2)^2)) →
  (∀ (p q : ℝ × ℝ), p ∈ S ∧ q ∈ S ∧ p ≠ q →
    (p.1 - q.1)^2 + (p.2 - q.2)^2 = (8/5)^2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_on_unit_circle_l728_72867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l728_72806

noncomputable section

-- Define the points in the plane
variable (A B C P : ℝ × ℝ)

-- Define the vectors
def AB (A B : ℝ × ℝ) : ℝ × ℝ := B - A
def AC (A C : ℝ × ℝ) : ℝ × ℝ := C - A
def AP (A P : ℝ × ℝ) : ℝ × ℝ := P - A

-- Define the conditions
axiom AB_length : ‖AB A B‖ = 1
axiom AC_length : ‖AC A C‖ = 1
axiom AB_AC_orthogonal : AB A B • AC A C = 0

-- Define the region
def in_region (A B C P : ℝ × ℝ) (l : ℝ) : Prop :=
  AP A P = l • AB A B + AC A C ∧ 1 ≤ l ∧ l ≤ 2

-- Define the area of the region
def region_area : ℝ := Real.pi * 3

-- Theorem statement
theorem area_of_region :
  region_area = Real.pi * 3 :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l728_72806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_term_l728_72846

theorem geometric_sequence_term (b : ℕ+ → ℝ) (m n : ℕ+) (c d : ℝ) 
  (h_geom : ∀ i j : ℕ+, b (i + 1) / b i = b (j + 1) / b j)
  (h_positive : ∀ i : ℕ+, b i > 0)
  (h_m : b m = c)
  (h_n : b n = d)
  (h_diff : n - m ≥ 1) :
  b (m + n) = (d^(n : ℝ) / c^(m : ℝ)) ^ (1 / ((n : ℝ) - (m : ℝ))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_term_l728_72846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_surface_area_from_prism_l728_72858

-- Define the right square prism
structure RightSquarePrism where
  height : ℝ
  volume : ℝ

-- Define the sphere
structure Sphere where
  radius : ℝ

-- Define a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define membership for a point in a sphere
def Point3D.inSphere (p : Point3D) (s : Sphere) : Prop :=
  (p.x^2 + p.y^2 + p.z^2) = s.radius^2

-- Define the theorem
theorem sphere_surface_area_from_prism (prism : RightSquarePrism) (sphere : Sphere) : 
  prism.height = 4 → 
  prism.volume = 16 → 
  (∀ (vertex : Point3D), vertex.inSphere sphere) → 
  4 * Real.pi * sphere.radius^2 = 24 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_surface_area_from_prism_l728_72858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_M₀_to_plane_l728_72893

-- Define the points
def M₀ : Fin 3 → ℝ := ![4, 3, 0]
def M₁ : Fin 3 → ℝ := ![1, 3, 0]
def M₂ : Fin 3 → ℝ := ![4, -1, 2]
def M₃ : Fin 3 → ℝ := ![3, 0, 1]

-- Define the plane passing through M₁, M₂, and M₃
def plane_equation (x y z : ℝ) : Prop :=
  2*x + y - z - 5 = 0

-- Define the distance function from a point to a plane
noncomputable def distance_to_plane (p : Fin 3 → ℝ) : ℝ :=
  |2*p 0 + p 1 - p 2 - 5| / Real.sqrt 6

-- Theorem statement
theorem distance_M₀_to_plane :
  distance_to_plane M₀ = Real.sqrt 6 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_M₀_to_plane_l728_72893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_series_expansion_of_1_plus_x_exp_x_l728_72810

/-- The power series expansion of (1+x)e^x -/
theorem power_series_expansion_of_1_plus_x_exp_x (x : ℝ) :
  (1 + x) * (Real.exp x) = ∑' n : ℕ, (n + 1 : ℝ) * x^n / n.factorial :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_series_expansion_of_1_plus_x_exp_x_l728_72810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_inside_circle_l728_72829

/-- A circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Distance between two points in a plane -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Predicate to check if a point is inside a circle -/
def is_inside (p : ℝ × ℝ) (c : Circle) : Prop :=
  distance p c.center < c.radius

theorem point_inside_circle (O : Circle) (P : ℝ × ℝ) 
  (h1 : O.radius = 5)
  (h2 : distance P O.center = 4) :
  is_inside P O := by
  sorry

#check point_inside_circle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_inside_circle_l728_72829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2014_value_l728_72805

def sequence_a : ℕ → ℚ
  | 0 => 3  -- Add this case for 0
  | 1 => 3
  | n + 2 => 1 / (sequence_a (n + 1) - 1) + 1

theorem a_2014_value : sequence_a 2014 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2014_value_l728_72805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_shift_even_function_l728_72802

/-- Given a function f and an angle φ, proves that if y = f(x - φ) is even, then φ = π/3 --/
theorem sin_shift_even_function (φ : ℝ) 
  (h1 : 0 < φ) (h2 : φ < π/2) :
  (∀ x : ℝ, Real.sin (2*(x - φ) + π/6) = Real.sin (2*(-x - φ) + π/6)) → φ = π/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_shift_even_function_l728_72802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_mass_frequency_l728_72842

def apple_masses : List ℚ := [125, 120, 122, 105, 130, 114, 116, 95, 120, 134]

def in_interval (x : ℚ) : Bool := 114.5 ≤ x ∧ x < 124.5

def count_in_interval (masses : List ℚ) : ℕ :=
  masses.filter in_interval |>.length

theorem apple_mass_frequency :
  (count_in_interval apple_masses : ℚ) / apple_masses.length = 4/10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_mass_frequency_l728_72842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_vector_c_l728_72883

def a : ℝ × ℝ := (2, -3)
def b : ℝ × ℝ := (-1, 1)

theorem unit_vector_c (c : ℝ × ℝ) : 
  (‖c‖ = 1) → -- c is a unit vector
  (∃ k : ℝ, k > 0 ∧ c = k • (a - b)) → -- c is in the same direction as a - b
  c = (3/5, -4/5) := by
  sorry

#check unit_vector_c

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_vector_c_l728_72883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_boxes_for_three_digit_cards_l728_72853

/-- Represents a card with a 3-digit number -/
def Card := Fin 1000

/-- Represents a box with a 2-digit number -/
def Box := Fin 100

/-- A card can be placed in a box if the box number can be obtained by deleting one digit from the card number -/
def can_place (c : Card) (b : Box) : Prop :=
  ∃ (i : Fin 3), ((c.val / (10 ^ i.val)) % 10 * (10 ^ i.val) + c.val % (10 ^ i.val)) = b.val

/-- A valid distribution is a function from Card to Box that satisfies the placement rule -/
def valid_distribution (f : Card → Box) : Prop :=
  ∀ c : Card, can_place c (f c)

/-- The theorem to be proved -/
theorem min_boxes_for_three_digit_cards :
  ∀ (f : Card → Box), valid_distribution f →
    ∃ (S : Finset Box), (∀ c : Card, f c ∈ S) ∧ S.card = 50 ∧
    ∀ (T : Finset Box), (∀ c : Card, f c ∈ T) → T.card ≥ 50 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_boxes_for_three_digit_cards_l728_72853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_burger_composition_theorem_l728_72856

/-- The percentage of a burger that is neither filler nor cheese -/
noncomputable def burger_percentage (total_weight filler_weight cheese_weight : ℝ) : ℝ :=
  (total_weight - filler_weight - cheese_weight) / total_weight * 100

/-- Theorem stating that the percentage of a 200-gram burger that is neither 40 grams of filler nor 30 grams of cheese is 65% -/
theorem burger_composition_theorem :
  burger_percentage 200 40 30 = 65 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_burger_composition_theorem_l728_72856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_G_relation_l728_72821

/-- The sequence G_n defined recursively -/
def G : ℕ → ℤ
  | 0 => 0  -- We define G_0 as 0 for completeness
  | 1 => 1
  | 2 => 2
  | (n + 3) => 2 * G (n + 2) + G (n + 1)

/-- The matrix power property -/
def matrix_power_property (n : ℕ+) : Prop :=
  ∃ (A : Matrix (Fin 2) (Fin 2) ℤ),
    A = !![1, 2; 2, 1] ∧
    A ^ (n : ℕ) = !![G (n + 1), 2 * G n; 2 * G n, G (n - 1)]

/-- The main theorem to prove -/
theorem G_relation (n : ℕ+) : G (n + 1) * G (n - 1) - 4 * G n ^ 2 = (-3 : ℤ) ^ (n : ℕ) := by
  sorry

#check G_relation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_G_relation_l728_72821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_sin_R_l728_72857

theorem right_triangle_sin_R (P Q R : Real) (sinP : ℝ) :
  -- PQR is a right triangle with Q as the right angle
  Q = 90 →
  -- sin P is given
  sinP = 3/5 →
  -- sin P is the sine of angle P
  Real.sin P = sinP →
  -- The theorem to prove
  Real.sin R = 4/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_sin_R_l728_72857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l728_72808

noncomputable def sequenceA (a : ℕ → ℝ) : Prop :=
  (∀ n, 2 * a (n + 2) + a n = 3 * a (n + 1)) ∧
  a 1 = 1 ∧ a 2 = 5

noncomputable def triangle_area (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  1/2 * (9 - a n) * n

theorem max_triangle_area (a : ℕ → ℝ) (h : sequenceA a) :
  ∃ M : ℝ, M = 4 ∧ ∀ n : ℕ, triangle_area a n ≤ M := by
  sorry

#check max_triangle_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l728_72808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_type_b_first_optimal_l728_72870

/-- Represents the types of questions in the competition -/
inductive QuestionType
| A
| B
deriving Repr, DecidableEq

/-- Represents the score for a correct answer based on question type -/
def score (q : QuestionType) : ℕ :=
  match q with
  | QuestionType.A => 20
  | QuestionType.B => 80

/-- Represents the probability of correctly answering a question of given type -/
def prob_correct (q : QuestionType) : ℝ :=
  match q with
  | QuestionType.A => 0.8
  | QuestionType.B => 0.6

/-- Calculates the expected score when choosing a given question type first -/
def expected_score (first : QuestionType) : ℝ :=
  let second := if first = QuestionType.A then QuestionType.B else QuestionType.A
  (prob_correct first) * (score first + (prob_correct second) * score second) +
  (prob_correct first) * (1 - prob_correct second) * score first

/-- Theorem: Choosing type B first yields a higher expected score -/
theorem type_b_first_optimal :
  expected_score QuestionType.B > expected_score QuestionType.A := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_type_b_first_optimal_l728_72870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_6_equals_51_l728_72832

def f : ℕ → ℕ
  | 0 => 1  -- Adding this case to cover Nat.zero
  | 1 => 1
  | 2 => 2
  | n+3 => f (n+2) - f (n+1) + (n+3)^2

theorem f_6_equals_51 : f 6 = 51 := by
  -- We'll use 'rfl' here to check the computation
  rfl

#eval f 6  -- This will evaluate f 6 and print the result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_6_equals_51_l728_72832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_statement_line_intersects_circle_with_chord_length_2_l728_72834

/-
Definition of the circle:
x^2 + y^2 - 2x - 4y + 4 = 0
-/
def is_on_circle (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 4*y + 4 = 0

/-
Definition of a point being on the line y = 2x
-/
def is_on_line (x y : ℝ) : Prop := y = 2*x

/-
Definition of the distance between two points
-/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

/-
Theorem statement
-/
theorem line_intersects_circle_with_chord_length_2 :
  ∃ (x1 y1 x2 y2 : ℝ),
    is_on_line x1 y1 ∧ is_on_line x2 y2 ∧  -- The points are on the line y = 2x
    is_on_circle x1 y1 ∧ is_on_circle x2 y2 ∧    -- The points are on the circle
    distance x1 y1 x2 y2 = 2 ∧       -- The distance between the points is 2
    (x1 = 0 ∨ x2 = 0) :=             -- One of the points is the origin
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_statement_line_intersects_circle_with_chord_length_2_l728_72834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_coordinates_X_W_l728_72888

-- Define the points as pairs of real numbers
noncomputable def X : ℝ × ℝ := sorry
def Y : ℝ × ℝ := (1, 7)
def Z : ℝ × ℝ := (-3, -7)
noncomputable def W : ℝ × ℝ := sorry

-- Define the ratios
noncomputable def ratio_XZ_XY : ℝ := 1/2
noncomputable def ratio_ZY_XY : ℝ := 1/2
noncomputable def ratio_XW_XZ : ℝ := 2

-- Define the theorem
theorem sum_coordinates_X_W :
  (ratio_XZ_XY = 1/2) →
  (ratio_ZY_XY = 1/2) →
  (ratio_XW_XZ = 2) →
  (Y = (1, 7)) →
  (Z = (-3, -7)) →
  (X.1 + X.2 + W.1 + W.2 = -20) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_coordinates_X_W_l728_72888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_odd_integer_in_sequence_l728_72894

theorem largest_odd_integer_in_sequence (seq : List Nat) : 
  seq.length = 30 ∧ 
  (∀ i j, i < j → i < seq.length → j < seq.length → seq.get! i < seq.get! j) ∧
  (∀ n, n ∈ seq → n % 2 = 1) ∧
  (∀ i, i < 29 → seq.get! (i+1) - seq.get! i = 2) ∧
  seq.sum = 7500 →
  seq.get! 29 = 279 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_odd_integer_in_sequence_l728_72894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_on_unit_circle_at_240_degrees_l728_72862

noncomputable def angle : Real := 240 * (Real.pi / 180)

theorem intersection_point_on_unit_circle_at_240_degrees :
  let x : Real := -1/2
  let y : Real := -Real.sqrt 3 / 2
  (x^2 + y^2 = 1) ∧ (x = Real.cos angle) ∧ (y = Real.sin angle) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_on_unit_circle_at_240_degrees_l728_72862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_radius_sum_l728_72895

/-- Definition of the circle C -/
def C (x y : ℝ) : Prop :=
  x^2 - 4*y - 18 = -y^2 + 6*x + 26

/-- Center of the circle -/
def center : ℝ × ℝ := (3, 2)

/-- Radius of the circle -/
noncomputable def radius : ℝ := Real.sqrt 57

/-- Theorem: The sum of the center coordinates and radius equals 5 + √57 -/
theorem circle_center_radius_sum :
  center.1 + center.2 + radius = 5 + Real.sqrt 57 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_radius_sum_l728_72895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_brendans_weekly_yard_cut_l728_72836

/-- Calculates the total yards of grass Brendan can cut in a week with his new lawnmower -/
def total_yards_cut (original_speed : ℝ) (improvement_percent : ℝ) (reduction_percent : ℝ) 
                    (flat_days : ℕ) (uneven_days : ℕ) : ℝ :=
  let improved_speed := original_speed * (1 + improvement_percent)
  let uneven_speed := improved_speed * (1 - reduction_percent)
  improved_speed * (flat_days : ℝ) + uneven_speed * (uneven_days : ℝ)

/-- Theorem stating that Brendan will cut 71.4 yards in a week -/
theorem brendans_weekly_yard_cut :
  total_yards_cut 8 0.5 0.35 4 3 = 71.4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_brendans_weekly_yard_cut_l728_72836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vertex_y_coordinate_l728_72820

/-- The y-coordinate of the vertex of the parabola y = -3x^2 - 30x - 81 is -6 -/
theorem parabola_vertex_y_coordinate :
  let f (x : ℝ) := -3 * x^2 - 30 * x - 81
  ∃ m : ℝ, (∀ x : ℝ, f x ≤ f m) ∧ f m = -6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vertex_y_coordinate_l728_72820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_job_completion_time_l728_72844

/-- The time taken to complete a job when multiple workers work together -/
noncomputable def time_to_complete (individual_times : List ℝ) : ℝ :=
  1 / (individual_times.map (λ t => 1 / t)).sum

/-- Theorem: Three workers with individual completion times of 15, 14, and 16 days
    can complete the job in 5 days when working together -/
theorem job_completion_time :
  time_to_complete [15, 14, 16] = 5 := by
  sorry

-- Use #eval only for computable functions
#check time_to_complete [15, 14, 16]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_job_completion_time_l728_72844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_power_magnitude_l728_72818

noncomputable def z : ℂ := 2 - 3 * Real.sqrt 2 * Complex.I

theorem complex_power_magnitude :
  Complex.abs (z^5) = (Real.sqrt 22) ^ 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_power_magnitude_l728_72818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conditional_probability_B_given_A_l728_72838

-- Define the sample space
def Ω : Type := Bool × Bool

-- Define the probability measure
noncomputable def P : Set Ω → ℝ := sorry

-- Define event A: tails on the first flip
def A : Set Ω := {ω : Ω | ω.1 = false}

-- Define event B: heads on the second flip
def B : Set Ω := {ω : Ω | ω.2 = true}

-- Axioms for probability measure
axiom P_nonneg : ∀ S : Set Ω, 0 ≤ P S
axiom P_le_one : ∀ S : Set Ω, P S ≤ 1
axiom P_total : P (Set.univ : Set Ω) = 1

-- Axioms for independent coin flips
axiom P_first_flip : P {ω : Ω | ω.1 = true} = 1/2
axiom P_second_flip : P {ω : Ω | ω.2 = true} = 1/2

-- Theorem: P(B|A) = 1/2
theorem conditional_probability_B_given_A : P (B ∩ A) / P A = 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_conditional_probability_B_given_A_l728_72838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xiao_ming_speed_equation_l728_72852

/-- Represents the distance between Xiao Ming's home and school in meters -/
def distance : ℝ := 2000

/-- Represents Xiao Ming's usual travel time in minutes -/
def x : ℝ := sorry

/-- Represents the increase in Xiao Ming's speed in meters per minute -/
def speed_increase : ℝ := 5

/-- Represents the reduction in travel time in minutes -/
def time_reduction : ℝ := 2

/-- Theorem stating that the given equation correctly represents the relationship
    between Xiao Ming's usual travel time and his speed on the day he was late -/
theorem xiao_ming_speed_equation (hx : x > 2) :
  distance / (x - time_reduction) - distance / x = speed_increase := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_xiao_ming_speed_equation_l728_72852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_equation_l728_72807

theorem solve_equation : ∃ n : ℚ, 3 * 7 * 4 * n = 6 * 5 * 4 * 3 * 2 * 1 ∧ n = 8 + 4/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_equation_l728_72807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_intersections_on_circle_l728_72887

/-- The curve in the problem -/
def curve (x : ℝ) : ℝ := x^2 - 6*x + 1

/-- A point lies on the curve if its y-coordinate equals the curve function at its x-coordinate -/
def lies_on_curve (p : ℝ × ℝ) : Prop := p.2 = curve p.1

/-- The circle C -/
def circle_C (c : ℝ × ℝ) (r : ℝ) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.1)^2 + (p.2 - c.2)^2 = r^2

theorem curve_intersections_on_circle :
  ∃ (c : ℝ × ℝ) (r : ℝ),
    (∀ p : ℝ × ℝ, lies_on_curve p → (p.1 = 0 ∨ p.2 = 0) → circle_C c r p) →
    c = (3, 1) ∧ r = 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_intersections_on_circle_l728_72887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_smallest_angle_l728_72864

theorem triangle_cosine_smallest_angle :
  ∀ (a b c : ℝ) (α β γ : ℝ),
    a = 8 ∧ b = 10 ∧ c = 12 →
    α + β + γ = Real.pi →
    Real.cos α * a = Real.sin β * b →
    Real.cos β * b = Real.sin γ * c →
    Real.cos γ * c = Real.sin α * a →
    γ = 1.5 * α →
    Real.cos α = 45 / 58 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_smallest_angle_l728_72864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l728_72859

noncomputable def f (x : ℝ) : ℝ := x^3 - 3*x^2 - 1

theorem f_properties :
  (∀ x < 0, StrictMono (f ∘ (fun y => (y : ℝ)) : Set.Iio 0 → ℝ)) ∧
  (∀ x > 2, StrictMono (f ∘ (fun y => (y : ℝ)) : Set.Ioi 2 → ℝ)) ∧
  (StrictAnti (f ∘ (fun y => (y : ℝ)) : Set.Icc 0 2 → ℝ)) ∧
  (f 0 = -1) ∧
  (f 2 = -5) ∧
  (∀ x y : ℝ, y = -3*x ↔ HasDerivAt f (-3) x ∧ f x = y) ∧
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
    HasDerivAt f ((f x₁) / x₁) x₁ ∧
    HasDerivAt f ((f x₂) / x₂) x₂ ∧
    3*x₁ + (f x₁) = 0 ∧
    15*x₂ + 4*(f x₂) = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l728_72859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_approximate_l728_72822

-- Define the speed of the train in km/hr
noncomputable def train_speed_kmh : ℝ := 126

-- Define the time taken to cross the pole in seconds
noncomputable def crossing_time : ℝ := 2.856914303998537

-- Define the conversion factor from km/hr to m/s
noncomputable def kmh_to_ms : ℝ := 1000 / 3600

-- Define the length of the train
noncomputable def train_length : ℝ := train_speed_kmh * kmh_to_ms * crossing_time

-- Theorem statement
theorem train_length_approximate :
  abs (train_length - 99.992) < 0.001 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_approximate_l728_72822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_B_is_21_l728_72855

/-- Represents the number of items of type A -/
def num_A : ℕ := sorry

/-- Represents the number of items of type B -/
def num_B : ℕ := sorry

/-- Represents the number of items of type C -/
def num_C : ℕ := sorry

/-- The total number of items bought -/
def total_items : ℕ := 100

/-- The total amount spent in yuan -/
def total_spent : ℕ := 100

/-- The price of item A in yuan per 8 pieces -/
def price_A : ℚ := 1 / 8

/-- The price of item B in yuan per piece -/
def price_B : ℚ := 1

/-- The price of item C in yuan per piece -/
def price_C : ℚ := 10

theorem num_B_is_21 :
  num_A + num_B + num_C = total_items ∧
  price_A * (num_A : ℚ) + price_B * (num_B : ℚ) + price_C * (num_C : ℚ) = total_spent →
  num_B = 21 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_B_is_21_l728_72855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_circle_l728_72881

/-- Given a circle and a line of symmetry, find the symmetric circle -/
theorem symmetric_circle (x y : ℝ) :
  let original_circle := (x + 1)^2 + (y - 4)^2 = 1
  let symmetry_line := y = x
  let symmetric_circle := (x - 4)^2 + (y + 1)^2 = 1
  symmetric_circle := by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_circle_l728_72881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_cost_is_6000_l728_72860

/-- Represents the cost and quantity information for sports equipment purchase --/
structure EquipmentPurchase where
  jumpRopePrice : ℚ
  shuttlecockPrice : ℚ
  totalItems : ℕ
  jumpRopeDiscount : ℚ
  shuttlecockDiscount : ℚ
  maxJumpRopes : ℕ

/-- Calculates the minimum cost for the sports equipment purchase --/
def minimumCost (purchase : EquipmentPurchase) : ℚ :=
  let jumpRopePrice := purchase.jumpRopePrice * (1 - purchase.jumpRopeDiscount)
  let shuttlecockPrice := purchase.shuttlecockPrice * (1 - purchase.shuttlecockDiscount)
  let minJumpRopes := (3 * purchase.totalItems) / 4
  jumpRopePrice * minJumpRopes + shuttlecockPrice * (purchase.totalItems - minJumpRopes)

/-- Theorem stating that the minimum cost for the given purchase scenario is 6000 --/
theorem minimum_cost_is_6000 (purchase : EquipmentPurchase) 
  (h1 : 5 * purchase.jumpRopePrice + 6 * purchase.shuttlecockPrice = 196)
  (h2 : 2 * purchase.jumpRopePrice + 5 * purchase.shuttlecockPrice = 120)
  (h3 : purchase.totalItems = 400)
  (h4 : purchase.jumpRopeDiscount = 1/5)
  (h5 : purchase.shuttlecockDiscount = 1/4)
  (h6 : purchase.maxJumpRopes = 310) :
  minimumCost purchase = 6000 := by
  sorry

#eval minimumCost {
  jumpRopePrice := 20,
  shuttlecockPrice := 16,
  totalItems := 400,
  jumpRopeDiscount := 1/5,
  shuttlecockDiscount := 1/4,
  maxJumpRopes := 310
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_cost_is_6000_l728_72860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_slope_l728_72890

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a parabola y^2 = 2px -/
structure Parabola where
  p : ℝ
  hpos : p > 0

/-- Represents a line passing through two points -/
structure Line where
  pointA : Point2D
  pointB : Point2D

/-- The focus of a parabola -/
noncomputable def focus (par : Parabola) : Point2D :=
  { x := par.p / 2, y := 0 }

/-- Check if a point lies on the parabola -/
def onParabola (point : Point2D) (par : Parabola) : Prop :=
  point.y^2 = 2 * par.p * point.x

/-- Calculate the slope of a line -/
noncomputable def slopeLine (line : Line) : ℝ :=
  (line.pointB.y - line.pointA.y) / (line.pointB.x - line.pointA.x)

/-- Calculate the area of a triangle given three points -/
noncomputable def triangleArea (p1 p2 p3 : Point2D) : ℝ :=
  abs ((p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y)) / 2)

/-- Main theorem -/
theorem parabola_line_intersection_slope (par : Parabola) (line : Line) :
  onParabola line.pointA par →
  onParabola line.pointB par →
  triangleArea (Point2D.mk 0 0) line.pointA (focus par) = 
    4 * triangleArea (Point2D.mk 0 0) line.pointB (focus par) →
  slopeLine line = 4/3 ∨ slopeLine line = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_slope_l728_72890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_standard_deviation_of_temperatures_l728_72816

noncomputable def temperatures : List ℝ := [8, -4, -1, 0, 2]

noncomputable def mean (xs : List ℝ) : ℝ := (xs.sum) / xs.length

noncomputable def variance (xs : List ℝ) : ℝ :=
  let m := mean xs
  (xs.map (fun x => (x - m) ^ 2)).sum / xs.length

noncomputable def standardDeviation (xs : List ℝ) : ℝ :=
  Real.sqrt (variance xs)

theorem standard_deviation_of_temperatures :
  standardDeviation temperatures = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_standard_deviation_of_temperatures_l728_72816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_l728_72819

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 2|

-- State the theorem
theorem range_of_x :
  ∀ a b : ℝ, a ≠ 0 → 
  (∀ x : ℝ, |a + b| + |a - b| ≥ |a| * f x) → 
  ∃ x : ℝ, 0 ≤ x ∧ x ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_l728_72819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_properties_l728_72871

/-- Tetrahedron with given vertices -/
structure Tetrahedron where
  A₁ : ℝ × ℝ × ℝ := (2, -1, -2)
  A₂ : ℝ × ℝ × ℝ := (1, 2, 1)
  A₃ : ℝ × ℝ × ℝ := (5, 0, -6)
  A₄ : ℝ × ℝ × ℝ := (-10, 9, -7)

/-- Calculate the volume of the tetrahedron -/
def volume (t : Tetrahedron) : ℝ :=
  sorry

/-- Calculate the height from A₄ to the face A₁A₂A₃ -/
def tetraHeight (t : Tetrahedron) : ℝ :=
  sorry

theorem tetrahedron_properties (t : Tetrahedron) :
  volume t = 140 / 3 ∧ tetraHeight t = 4 * Real.sqrt 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_properties_l728_72871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_angles_l728_72865

-- Define a triangle with the given properties
structure SpecialTriangle where
  A : ℝ
  B : ℝ
  C : ℝ
  angle_sum : A + B + C = 180
  c_divided : ∃ (m b h : ℝ), m > 0 ∧ b > 0 ∧ h > 0 ∧ C = m + b + h + h

-- Theorem statement
theorem special_triangle_angles (t : SpecialTriangle) :
  t.A = 22.5 ∧ t.B = 67.5 ∧ t.C = 90 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_angles_l728_72865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l728_72835

-- Define the set B
def B : Set ℝ := {x | Real.sqrt 2 ≤ x ∧ x ≤ 4}

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (Real.log x - Real.log 8) * (Real.log (2 * x))

-- Theorem statement
theorem problem_solution :
  ∀ x : ℝ, (2 * (Real.log x)^2 - 5 * (Real.log x) + 2 * Real.log 2 ≤ 0) →
  (B = {x | Real.sqrt 2 ≤ x ∧ x ≤ 4}) ∧
  (x ∈ B → (∃ (y : ℝ), y ∈ B ∧ f y = 5 * Real.log 2 ∧ ∀ z ∈ B, f z ≤ 5 * Real.log 2) ∧
           (∃ (y : ℝ), y ∈ B ∧ f y = -4 * Real.log 2 ∧ ∀ z ∈ B, f z ≥ -4 * Real.log 2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l728_72835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_min_surface_area_l728_72877

/-- Given a cylinder with fixed height and volume, this theorem states the radius that minimizes the total surface area. -/
theorem cylinder_min_surface_area (H V : ℝ) (h1 : H > 0) (h2 : V > 0) :
  let r := (1/2) * (4 * V / Real.pi) ^ (1/3)
  let surface_area := fun (r : ℝ) => 2 * Real.pi * r^2 + 2 * Real.pi * r * H
  let volume := fun (r : ℝ) => Real.pi * r^2 * H
  (∀ r' > 0, volume r' = V → surface_area r ≤ surface_area r') ∧ volume r = V :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_min_surface_area_l728_72877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l728_72824

theorem inequality_proof (a b c lambda : ℝ) (n : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hlambda : lambda > 0) (hn : n ≥ 2) 
  (h_sum : a^(n-1) + b^(n-1) + c^(n-1) = 1) :
  (a^n / (b + lambda*c)) + (b^n / (c + lambda*a)) + (c^n / (a + lambda*b)) ≥ 1 / (1 + lambda) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l728_72824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l728_72876

/-- Represents a hyperbola with parameters a and b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0
  equation : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1

/-- The focal length of a hyperbola -/
noncomputable def focal_length (h : Hyperbola) : ℝ := Real.sqrt (h.a^2 + h.b^2)

/-- The real axis length of a hyperbola -/
def real_axis_length (h : Hyperbola) : ℝ := 2 * h.a

/-- The asymptotes of a hyperbola -/
def asymptotes (h : Hyperbola) : Set (ℝ × ℝ) :=
  {(x, y) | y = h.b / h.a * x ∨ y = -h.b / h.a * x}

/-- Main theorem: If the real axis length is half the focal length, 
    then the asymptotes are y = ±√3 x -/
theorem hyperbola_asymptotes (h : Hyperbola) 
  (h_axis : real_axis_length h = (focal_length h) / 2) :
  asymptotes h = {(x, y) | y = Real.sqrt 3 * x ∨ y = -Real.sqrt 3 * x} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l728_72876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_l728_72868

def a : ℕ → ℚ
| 0 => 1
| 1 => 5
| n + 2 => (2 * a (n + 1)^2 - 3 * a (n + 1) - 9) / (2 * a n)

theorem a_formula (n : ℕ) : a n = 2^(n+2) - 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_l728_72868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_product_l728_72804

/-- Conic section C -/
def C (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

/-- Line L -/
noncomputable def L (x y : ℝ) : Prop := y = Real.sqrt 3 * (x + 1)

/-- Left focus F₁ -/
def F₁ : ℝ × ℝ := (-1, 0)

/-- Intersection points of C and L -/
def intersectionPoints : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | C p.1 p.2 ∧ L p.1 p.2}

/-- Distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Main theorem -/
theorem intersection_product (M N : ℝ × ℝ) 
  (hM : M ∈ intersectionPoints) (hN : N ∈ intersectionPoints) (hMN : M ≠ N) :
  distance F₁ M * distance F₁ N = 12 / 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_product_l728_72804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_center_of_g_l728_72866

noncomputable section

/-- The original function -/
def f (x : ℝ) : ℝ := 4 * Real.sin (4 * x + Real.pi / 6)

/-- The resulting function after transformations -/
def g (x : ℝ) : ℝ := 4 * Real.sin (2 * x - Real.pi / 6)

/-- Theorem stating that (7π/12, 0) is a symmetry center of g -/
theorem symmetry_center_of_g :
  ∀ x : ℝ, g (7 * Real.pi / 12 + x) = g (7 * Real.pi / 12 - x) :=
by
  intro x
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_center_of_g_l728_72866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_points_form_parabola_l728_72840

theorem points_form_parabola (t : ℝ) :
  let x := (3 : ℝ)^t - 4
  let y := (3 : ℝ)^(2*t) - 7 * (3 : ℝ)^t - 6
  y = x^2 + x - 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_points_form_parabola_l728_72840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_village_population_after_panic_l728_72813

theorem village_population_after_panic (original_population : ℕ) 
  (h1 : original_population = 7200) : 
  (original_population - (original_population / 10) - 
   (original_population - (original_population / 10)) / 4) = 4860 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_village_population_after_panic_l728_72813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_side_length_equality_l728_72831

/-- Given a square and an equilateral triangle on a plane where the area of each shape
    is numerically equal to the perimeter of the other, prove that the side length of
    the square is 2 * (2^(1/3)) * √3. -/
theorem square_side_length_equality (a b : ℝ) 
    (h1 : a > 0) 
    (h2 : b > 0)
    (h3 : (Real.sqrt 3 / 4) * a^2 = 4 * b)  -- Area of triangle equals perimeter of square
    (h4 : b^2 = 3 * a)             -- Area of square equals perimeter of triangle
    : b = 2 * (2^(1/3)) * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_side_length_equality_l728_72831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_angle_expression_l728_72828

-- Define the curve
noncomputable def f (x : ℝ) : ℝ := (2/3) * x^3

-- Define the derivative of f
noncomputable def f' (x : ℝ) : ℝ := 2 * x^2

-- Define the angle of inclination α
noncomputable def α : ℝ := Real.arctan (f' 1)

-- Theorem statement
theorem tangent_angle_expression :
  (Real.sin α)^2 - (Real.cos α)^2 = (3/5) * (2 * Real.sin α * Real.cos α + (Real.cos α)^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_angle_expression_l728_72828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_olga_cant_paint_l728_72823

/-- Represents the time taken for a round trip to the country house -/
def roundTripTime : ℝ → ℝ := sorry

/-- Represents the painting speed in boards per hour -/
def paintingSpeed : ℝ → ℝ := sorry

/-- Valera's painting speed -/
def valeraSpeed : ℝ := 5.5

/-- Olga's painting speed -/
def olgaSpeed : ℝ := 4

theorem olga_cant_paint (t : ℝ) (h1 : t = 1.5) :
  ∀ (x : ℝ), roundTripTime t + x / olgaSpeed > t → x = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_olga_cant_paint_l728_72823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_is_pi_over_two_l728_72841

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := 2 - Real.sqrt (4 - x^2)

-- Define the domain of g
def g_domain (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 0

-- Theorem statement
theorem enclosed_area_is_pi_over_two :
  ∃ (A : Set (ℝ × ℝ)), 
    (∀ (x y : ℝ), (x, y) ∈ A ↔ 
      ((g_domain x ∧ y = g x) ∨ 
       (g_domain y ∧ x = g y))) ∧
    MeasureTheory.volume A = π / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_is_pi_over_two_l728_72841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_x_above_line_eq_17_l728_72892

def points : List (ℚ × ℚ) := [(4, 15), (7, 28), (10, 40), (13, 44), (16, 53)]

def line (x : ℚ) : ℚ := 3 * x + 5

def is_above_line (point : ℚ × ℚ) : Bool :=
  point.2 > line point.1

def sum_x_above_line (points : List (ℚ × ℚ)) : ℚ :=
  (points.filter is_above_line).map Prod.fst |>.sum

theorem sum_x_above_line_eq_17 : sum_x_above_line points = 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_x_above_line_eq_17_l728_72892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_double_radius_quadruple_weight_weight_of_larger_sphere_l728_72863

/-- The weight of a hollow sphere given its radius and the weight of a reference sphere -/
noncomputable def sphere_weight (r : ℝ) (r_ref : ℝ) (w_ref : ℝ) : ℝ :=
  w_ref * (r / r_ref)^2

/-- Theorem: A hollow sphere with twice the radius of a reference sphere weighs four times as much -/
theorem double_radius_quadruple_weight (r : ℝ) (w : ℝ) (h : r > 0) :
  sphere_weight (2 * r) r w = 4 * w := by
  sorry

/-- The weight of a hollow sphere with twice the radius of a 0.15 cm sphere weighing 8 grams -/
theorem weight_of_larger_sphere :
  sphere_weight 0.3 0.15 8 = 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_double_radius_quadruple_weight_weight_of_larger_sphere_l728_72863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l728_72880

/-- The time it takes for two trains to cross each other -/
noncomputable def crossing_time (length1 length2 speed1 speed2 : ℝ) : ℝ :=
  (length1 + length2) / ((speed1 + speed2) * (5/18))

/-- Theorem stating the crossing time for the given train problem -/
theorem train_crossing_time :
  let length1 : ℝ := 110
  let length2 : ℝ := 160
  let speed1 : ℝ := 60
  let speed2 : ℝ := 40
  abs (crossing_time length1 length2 speed1 speed2 - 9.72) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l728_72880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_FAB_l728_72812

-- Define the parabola C
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the line l₂
def line_l2 (x y : ℝ) : Prop := x = y + 8

-- Define the focus F of the parabola
def focus : ℝ × ℝ := (2, 0)

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  parabola A.1 A.2 ∧ parabola B.1 B.2 ∧
  line_l2 A.1 A.2 ∧ line_l2 B.1 B.2 ∧
  A ≠ B

-- Define the area of a triangle
def area_triangle (A B C : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem area_of_triangle_FAB (A B : ℝ × ℝ) :
  intersection_points A B →
  area_triangle focus A B = 24 * Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_FAB_l728_72812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l728_72849

-- Define the circle
def myCircle (x y : ℝ) : Prop := x^2 + y^2 - 2 * Real.sqrt 3 * x - 2 * y + 3 = 0

-- Define the points A and B
def A (a : ℝ) : ℝ × ℝ := (a, 0)
def B (a : ℝ) : ℝ × ℝ := (-a, 0)

-- Define the right angle condition
def rightAngle (a m n : ℝ) : Prop := (m + a) * (m - a) + n^2 = 0

-- Main theorem
theorem range_of_a (a : ℝ) :
  (a > 0) →
  (∃ m n : ℝ, myCircle m n ∧ rightAngle a m n) →
  1 ≤ a ∧ a ≤ 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l728_72849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclists_meet_time_l728_72869

/-- The length of the circular track in meters -/
noncomputable def track_length : ℝ := 400

/-- The speed of Mona in kilometers per hour -/
noncomputable def speed_mona : ℝ := 18

/-- The speed of Sona in kilometers per hour -/
noncomputable def speed_sona : ℝ := 36

/-- The speed of Ravi in kilometers per hour -/
noncomputable def speed_ravi : ℝ := 24

/-- The speed of Nina in kilometers per hour -/
noncomputable def speed_nina : ℝ := 48

/-- Conversion factor from kilometers per hour to meters per minute -/
noncomputable def kmph_to_mpm : ℝ := 1000 / 60

/-- Function to calculate the time for one lap given a speed in kmph -/
noncomputable def lap_time (speed : ℝ) : ℝ := track_length / (speed * kmph_to_mpm)

/-- Theorem stating that the time for all cyclists to meet at the starting point is 4 minutes -/
theorem cyclists_meet_time : 
  let times := [lap_time speed_mona, lap_time speed_sona, lap_time speed_ravi, lap_time speed_nina]
  ∃ (lcm : ℝ), lcm > 0 ∧ (∀ t ∈ times, ∃ (n : ℕ), n * t = lcm) ∧ lcm = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclists_meet_time_l728_72869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_equation_altitude_equation_triangle_area_l728_72897

-- Define the coordinates of the triangle vertices
noncomputable def A : ℝ × ℝ := (-2, 1)
noncomputable def B : ℝ × ℝ := (2, 1)
noncomputable def C : ℝ × ℝ := (4, -3)

-- Define the midpoint D of AC
noncomputable def D : ℝ × ℝ := ((A.1 + C.1) / 2, (A.2 + C.2) / 2)

-- Define the equation of a line in the form ax + by + c = 0
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Theorem for the equation of median BD
theorem median_equation : Line.mk 2 (-1) (-3) = 
  let k := (D.2 - B.2) / (D.1 - B.1)
  let b := B.2 - k * B.1
  Line.mk k (-1) (-b) := by sorry

-- Theorem for the equation of altitude on BC
theorem altitude_equation : Line.mk 1 (-2) 4 = 
  let k_BC := (C.2 - B.2) / (C.1 - B.1)
  let k_AH := -1 / k_BC
  let b := A.2 - k_AH * A.1
  Line.mk k_AH (-1) (-b) := by sorry

-- Theorem for the area of triangle ABC
theorem triangle_area : 8 = 
  let k_BC := (C.2 - B.2) / (C.1 - B.1)
  let b_BC := B.2 - k_BC * B.1
  let d := |2 * A.1 + A.2 - 5| / Real.sqrt (2^2 + 1^2)
  let BC_length := Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2)
  (1/2) * BC_length * d := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_equation_altitude_equation_triangle_area_l728_72897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_product_equals_44_l728_72801

theorem floor_product_equals_44 (x : ℝ) :
  ⌊x * ⌊x⌋⌋ = 44 ↔ x ∈ (Set.Icc (-45/7) (-44/7) ∪ Set.Ico (44/6) (45/6)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_product_equals_44_l728_72801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_tan_B_in_special_triangle_l728_72803

theorem max_tan_B_in_special_triangle (A B C : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = Real.pi ∧ 
  (Real.sin B) / (Real.sin A) = Real.cos (A + B) →
  ∀ B' : ℝ, 0 < B' ∧ 
    ∃ A' C' : ℝ, 0 < A' ∧ 0 < C' ∧ 
    A' + B' + C' = Real.pi ∧ 
    (Real.sin B') / (Real.sin A') = Real.cos (A' + B') →
  Real.tan B ≤ Real.sqrt 2 / 4 ∧ 
  ∃ A₀ B₀ C₀ : ℝ, 0 < A₀ ∧ 0 < B₀ ∧ 0 < C₀ ∧ 
    A₀ + B₀ + C₀ = Real.pi ∧ 
    (Real.sin B₀) / (Real.sin A₀) = Real.cos (A₀ + B₀) ∧
    Real.tan B₀ = Real.sqrt 2 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_tan_B_in_special_triangle_l728_72803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frame_interior_edges_sum_l728_72837

/-- A rectangular picture frame -/
structure Frame where
  outer_edge : ℝ
  frame_area : ℝ
  frame_width : ℝ

/-- The sum of the lengths of the four interior edges of a frame -/
noncomputable def interior_edges_sum (f : Frame) : ℝ :=
  2 * (f.outer_edge - 2 * f.frame_width) + 2 * ((f.frame_area / (2 * f.frame_width)) - f.frame_width)

/-- Theorem: For a frame with outer edge 8 inches, frame area 32 square inches, 
    and frame width 1 inch, the sum of interior edges is 24 inches -/
theorem frame_interior_edges_sum :
  let f : Frame := { outer_edge := 8, frame_area := 32, frame_width := 1 }
  interior_edges_sum f = 24 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_frame_interior_edges_sum_l728_72837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribedRadius_correct_l728_72861

noncomputable section

open Real

/-- The radius of the inscribed circle in an isosceles triangle -/
def inscribedRadius (S α : ℝ) : ℝ :=
  Real.sqrt (S * tan α) * tan ((π / 4) - (α / 2))

theorem inscribedRadius_correct (S α : ℝ) (hS : S > 0) (hα : 0 < α ∧ α < π / 2) :
  inscribedRadius S α = Real.sqrt (S * tan α) * tan ((π / 4) - (α / 2)) := by
  unfold inscribedRadius
  -- The proof steps would go here
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribedRadius_correct_l728_72861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_period_problem_l728_72843

noncomputable def cosine_period (b : ℝ) : ℝ := 2 * Real.pi / b

noncomputable def f (a b c d : ℝ) (x : ℝ) : ℝ := a * Real.cos (b * x + c) + d

theorem cosine_period_problem (a b c d : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) 
  (h5 : cosine_period b * 2 = 3 * Real.pi) : b = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_period_problem_l728_72843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_cone_properties_l728_72850

/-- Represents a cone formed from a circular sector --/
structure SectorCone where
  sector_angle : ℝ  -- Angle of the sector in degrees
  sector_radius : ℝ  -- Radius of the original circle
  base_radius : ℝ    -- Radius of the cone's base
  slant_height : ℝ   -- Slant height of the cone

/-- Theorem stating the properties of a cone formed from a 300° sector of a circle with radius 10 --/
theorem sector_cone_properties :
  ∃ (cone : SectorCone),
    cone.sector_angle = 300 ∧
    cone.sector_radius = 10 ∧
    cone.slant_height = 10 ∧
    abs (cone.base_radius - 8) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_cone_properties_l728_72850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l728_72889

-- Define the interval [-1/2, 1]
def I : Set ℝ := {x | -1/2 ≤ x ∧ x ≤ 1}

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x - 2 * Real.cos x
noncomputable def g (x : ℝ) : ℝ := Real.exp (1 - 2 * x)

-- Define the tangent line l
def l (x : ℝ) : ℝ := 2 * x - 2

theorem function_properties :
  (∀ x ∈ I, f x ≥ l x) ∧
  (∀ x ∈ I, f x + g x ≥ 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l728_72889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_P_proper_subset_M_l728_72825

def M : Set ℤ := {x | -1 ≤ x ∧ x ≤ 1}

def P : Set ℤ := {y | ∃ x ∈ M, y = x^2}

theorem P_proper_subset_M : P ⊂ M := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_P_proper_subset_M_l728_72825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_side_length_l728_72827

/-- Represents a trapezium with given dimensions -/
structure Trapezium where
  side1 : ℝ
  side2 : ℝ
  height : ℝ

/-- Calculates the area of a trapezium -/
noncomputable def area (t : Trapezium) : ℝ :=
  (t.side1 + t.side2) * t.height / 2

/-- Theorem stating the length of the unknown side of the trapezium -/
theorem trapezium_side_length : 
  ∀ (t : Trapezium), 
    t.side2 = 18 ∧ 
    t.height = 5 ∧ 
    area t = 95 → 
    t.side1 = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_side_length_l728_72827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jenn_wins_iff_n_eq_6_l728_72898

/-- Represents a player in the game -/
inductive Player
| Bela
| Jenn

/-- Represents the game state -/
structure GameState where
  n : ℕ
  choices : List ℝ
  currentPlayer : Player

/-- Checks if a move is valid -/
def isValidMove (state : GameState) (move : ℝ) : Prop :=
  ∀ c ∈ state.choices, |move - c| > 2

/-- Defines the game rules and winning condition -/
def gameRules (n : ℕ) : Prop :=
  n > 6 ∧
  ∀ (state : GameState),
    (state.n = n ∧ state.choices = [0] ∧ state.currentPlayer = Player.Jenn) →
    (∃ (move : ℝ), isValidMove state move) →
    (∀ (move : ℝ), ¬isValidMove state move) →
    state.currentPlayer = Player.Bela

/-- Represents a strategy for a player -/
def Strategy := GameState → ℝ

/-- Defines what it means for Jenn to win -/
def JennWins (n : ℕ) (strategy : Strategy) : Prop :=
  ∀ (state : GameState),
    state.n = n →
    state.currentPlayer = Player.Jenn →
    isValidMove state (strategy state) ∧
    ¬∃ (move : ℝ), isValidMove (⟨n, state.choices ++ [strategy state], Player.Bela⟩) move

/-- Theorem stating the winning condition for Jenn -/
theorem jenn_wins_iff_n_eq_6 (n : ℕ) :
  gameRules n → (∃ (strategy : Strategy), JennWins n strategy ↔ n = 6) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jenn_wins_iff_n_eq_6_l728_72898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_trapezoid_KHLG_l728_72854

/-- Represents a point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle in a 2D plane -/
structure Rectangle where
  bottomLeft : Point
  topRight : Point

/-- Calculates the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ :=
  sorry

/-- Calculates the length of a line segment between two points -/
def segment_length (p1 p2 : Point) : ℝ :=
  sorry

/-- Calculates the area of a trapezoid given its four vertices -/
def area_of_trapezoid (p1 p2 p3 p4 : Point) : ℝ :=
  sorry

/-- Given a rectangle GHIJ with area 20 square units, where K divides IJ into segments of 2 and 8 units,
    and L divides GJ into segments of 1 and 4 units, the area of trapezoid KHLG is 22 square units. -/
theorem area_of_trapezoid_KHLG (GHIJ : Rectangle) (G H I J K L : Point) :
  GHIJ.area = 20 →
  segment_length I K = 2 →
  segment_length K J = 8 →
  segment_length G L = 1 →
  segment_length L J = 4 →
  area_of_trapezoid K H L G = 22 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_trapezoid_KHLG_l728_72854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oplus_minus_equality_range_of_a_range_of_m_l728_72848

-- Define the ⊕ operation
noncomputable def oplus (x y : ℝ) : ℝ := Real.log (10^x + 10^y)

-- Part (Ⅰ)
theorem oplus_minus_equality (a b c : ℝ) :
  (oplus a b) - c = oplus (a - c) (b - c) := by sorry

-- Part (Ⅱ)
def has_exactly_three_integer_solutions (a : ℝ) : Prop :=
  ∃! (x₁ x₂ x₃ : ℤ), (x₁ - 1)^2 > oplus (a^2 * x₁^2) (a^2 * x₁^2) - Real.log 2 ∧
                     (x₂ - 1)^2 > oplus (a^2 * x₂^2) (a^2 * x₂^2) - Real.log 2 ∧
                     (x₃ - 1)^2 > oplus (a^2 * x₃^2) (a^2 * x₃^2) - Real.log 2

theorem range_of_a :
  {a : ℝ | has_exactly_three_integer_solutions a} = 
  Set.Icc (-3/2) (-4/3) ∪ Set.Icc (4/3) (3/2) := by sorry

-- Part (Ⅲ)
noncomputable def f (x : ℝ) : ℝ := Real.log (oplus (x + 4) (x + 4) - Real.sqrt (2 * x + 3) - Real.log 2)

noncomputable def g (x : ℝ) : ℝ := oplus (oplus 1 x) (-x)

theorem range_of_m :
  {m : ℝ | ∀ x₁, ∃ x₂ ∈ Set.Ici (-3/2), g x₁ = Real.log (|3*m - 2|) + f x₂} =
  Set.Icc (-4/3) (2/3) ∪ Set.Ioc (2/3) (8/3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oplus_minus_equality_range_of_a_range_of_m_l728_72848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_rectangle_perimeter_difference_count_impossible_d_values_l728_72875

theorem hexagon_rectangle_perimeter_difference (d : ℕ) : 
  (∃ (h w : ℝ), 
    h > 0 ∧ w > 0 ∧
    6 * h = 6 * w + 2997 ∧
    h = w + d ∧
    6 * w > 0) ↔ d ≥ 500 :=
by sorry

theorem count_impossible_d_values : 
  Finset.card (Finset.range 500) = 499 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_rectangle_perimeter_difference_count_impossible_d_values_l728_72875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shift_for_symmetry_min_shift_is_pi_over_six_l728_72800

/-- The function f(x) = 3sin(x) + √3cos(x) -/
noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin x + Real.sqrt 3 * Real.cos x

/-- The shifted function g(x) = f(x - φ) -/
noncomputable def g (φ : ℝ) (x : ℝ) : ℝ := f (x - φ)

/-- Symmetry condition: g(x) = -g(-x) for all x -/
def is_symmetric (φ : ℝ) : Prop := ∀ x, g φ x = - g φ (-x)

theorem min_shift_for_symmetry :
  ∃ φ_min : ℝ, φ_min > 0 ∧ is_symmetric φ_min ∧ ∀ φ, φ > 0 → is_symmetric φ → φ_min ≤ φ := by
  sorry

theorem min_shift_is_pi_over_six :
  ∃ φ_min : ℝ, φ_min = Real.pi / 6 ∧ is_symmetric φ_min ∧ 
    ∀ φ, φ > 0 → is_symmetric φ → φ_min ≤ φ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shift_for_symmetry_min_shift_is_pi_over_six_l728_72800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_domain_function_proof_l728_72874

/-- A function with the same domain and range as [m,n] -/
def SameDomainFunction (f : ℝ → ℝ) (m n : ℝ) : Prop :=
  (∀ x, m ≤ x ∧ x ≤ n → m ≤ f x ∧ f x ≤ n) ∧
  (∀ y, m ≤ y ∧ y ≤ n → ∃ x, m ≤ x ∧ x ≤ n ∧ f x = y)

/-- The given function f(x) -/
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := Real.sqrt a * x^2 - 2 * Real.sqrt a * x + b + 1

/-- The function g(x) defined in terms of f(x) -/
noncomputable def g (k : ℝ) (f : ℝ → ℝ) (x : ℝ) : ℝ := k - Real.sqrt (f x - 3/2)

theorem same_domain_function_proof :
  ∀ a b : ℝ, SameDomainFunction (f a b) 1 3 →
  (∃ a' b', ∀ x, f a' b' x = 1/2 * x^2 - x + 3/2) ∧
  (∀ k : ℝ, k ≥ 0 →
    (∃ c d : ℝ, c < 0 ∧ d < 0 ∧ SameDomainFunction (g k (f a b)) c d) ↔
    (0 ≤ k ∧ k < 1 - Real.sqrt 2 / 2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_domain_function_proof_l728_72874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_country_Z_diploma_percentage_l728_72873

theorem country_Z_diploma_percentage :
  let total_population := 100
  let job_of_choice_no_diploma := 10
  let job_of_choice_total := 20
  let no_job_of_choice_with_diploma_percentage := 25
  
  job_of_choice_no_diploma < job_of_choice_total ∧
  job_of_choice_total < total_population ∧
  0 < no_job_of_choice_with_diploma_percentage ∧ no_job_of_choice_with_diploma_percentage < 100 →
  
  (job_of_choice_total - job_of_choice_no_diploma +
   (total_population - job_of_choice_total) * no_job_of_choice_with_diploma_percentage / 100 : ℝ) = 30 :=
by
  intro total_population job_of_choice_no_diploma job_of_choice_total no_job_of_choice_with_diploma_percentage h
  -- The proof would go here, but we'll use sorry for now
  sorry

#check country_Z_diploma_percentage

end NUMINAMATH_CALUDE_ERRORFEEDBACK_country_Z_diploma_percentage_l728_72873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_distance_theorem_l728_72899

/-- Calculates the distance traveled by a car in 30 minutes, given its speed relative to a train --/
noncomputable def car_distance (train_speed : ℝ) (car_speed_ratio : ℝ) : ℝ :=
  let car_speed := car_speed_ratio * train_speed
  let time_hours := 30 / 60
  car_speed * time_hours

theorem car_distance_theorem :
  car_distance 90 (5/6) = 37.5 := by
  -- Unfold the definition of car_distance
  unfold car_distance
  -- Perform the calculation
  simp [mul_assoc, mul_comm, mul_div_cancel']
  -- The proof is complete
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_distance_theorem_l728_72899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_radius_increase_l728_72809

-- Define constants
def original_radius : ℝ := 16
def original_distance : ℝ := 520
def new_distance : ℝ := 500

-- Define the function to calculate the actual distance given radius and odometer reading
noncomputable def actual_distance (radius : ℝ) (odometer_reading : ℝ) : ℝ :=
  odometer_reading * (2 * Real.pi * radius) / (2 * Real.pi * original_radius)

-- State the theorem
theorem wheel_radius_increase :
  ∃ (new_radius : ℝ),
    actual_distance new_radius new_distance = original_distance ∧
    new_radius - original_radius = 0.40 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_radius_increase_l728_72809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_slope_l728_72884

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 2 = 1

-- Define a line passing through (1,1) with slope k
def line (k : ℝ) (x y : ℝ) : Prop := y - 1 = k * (x - 1)

-- Define the intersection points of the line and the ellipse
def intersection (k : ℝ) (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  ellipse x₁ y₁ ∧ ellipse x₂ y₂ ∧ line k x₁ y₁ ∧ line k x₂ y₂

-- Define the midpoint condition
def is_midpoint (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  (x₁ + x₂) / 2 = 1 ∧ (y₁ + y₂) / 2 = 1

-- The main theorem
theorem ellipse_line_slope :
  ∀ k x₁ y₁ x₂ y₂ : ℝ,
  intersection k x₁ y₁ x₂ y₂ →
  is_midpoint x₁ y₁ x₂ y₂ →
  k = -1/2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_slope_l728_72884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_alloy_chromium_percentage_is_12_l728_72851

-- Define the percentages and masses
noncomputable def second_alloy_chromium_percentage : ℝ := 8
noncomputable def first_alloy_mass : ℝ := 10
noncomputable def second_alloy_mass : ℝ := 30
noncomputable def new_alloy_chromium_percentage : ℝ := 9

-- Define the function to calculate the chromium percentage in the first alloy
noncomputable def calculate_first_alloy_chromium_percentage : ℝ :=
  let total_mass := first_alloy_mass + second_alloy_mass
  let new_alloy_chromium_mass := (new_alloy_chromium_percentage / 100) * total_mass
  let second_alloy_chromium_mass := (second_alloy_chromium_percentage / 100) * second_alloy_mass
  let first_alloy_chromium_mass := new_alloy_chromium_mass - second_alloy_chromium_mass
  (first_alloy_chromium_mass / first_alloy_mass) * 100

-- Theorem statement
theorem first_alloy_chromium_percentage_is_12 :
  calculate_first_alloy_chromium_percentage = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_alloy_chromium_percentage_is_12_l728_72851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bee_travel_theorem_l728_72833

/-- Calculates the distance traveled by a bee flying between two people moving towards each other -/
noncomputable def beeTravelDistance (initialDistance : ℝ) (speedA : ℝ) (speedB : ℝ) (speedBee : ℝ) : ℝ :=
  let relativeSpeed := speedA + speedB
  let meetingTime := initialDistance / relativeSpeed
  speedBee * meetingTime

/-- Theorem stating the distance traveled by the bee in the given scenario -/
theorem bee_travel_theorem :
  beeTravelDistance 120 30 10 60 = 180 := by
  -- Unfold the definition of beeTravelDistance
  unfold beeTravelDistance
  -- Simplify the expression
  simp
  -- Check that the result is equal to 180
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bee_travel_theorem_l728_72833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_inequality_solution_set_l728_72826

theorem fraction_inequality_solution_set (x : ℝ) :
  (1 + x) / (2 - x) ≥ 0 ↔ x ∈ Set.Icc (-1 : ℝ) 2 ∧ x ≠ 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_inequality_solution_set_l728_72826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l728_72891

/-- A line in 2D space --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ
  eq : (x y : ℝ) → Prop := λ x y ↦ a * x + b * y + c = 0

/-- Two lines are parallel if their slopes are equal --/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l2.a * l1.b

theorem line_equation_proof (L L1 L2 : Line) :
  L.a = 1 ∧ L.b = -3 ∧ L.c = -1 ∧
  L1.a = 3 ∧ L1.b = 1 ∧ L1.c = -6 ∧
  L2.a = 3 ∧ L2.b = 1 ∧ L2.c = 3 ∧
  parallel L1 L2 ∧
  L.eq 1 0 →
  L.eq = (λ x y ↦ x - 3 * y - 1 = 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l728_72891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_eight_to_twelve_rounded_l728_72814

theorem ratio_eight_to_twelve_rounded (ratio : ℚ) (rounded : ℚ) : 
  ratio = 8 / 12 →
  rounded = (ratio * 10).floor / 10 + 0.1 →
  rounded = 0.7 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_eight_to_twelve_rounded_l728_72814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_prism_diagonal_length_l728_72811

/-- In a rectangular prism, given the distances from a body diagonal to three edges,
    this function calculates the length of that body diagonal. -/
noncomputable def body_diagonal_length (m n p : ℝ) : ℝ :=
  120 * Real.sqrt (2 / (m^2 + p^2 - n^2) + 2 / (m^2 - p^2 + n^2) + 2 / (-m^2 + p^2 + n^2))

/-- Theorem stating that in a rectangular prism where the distances from a body diagonal
    to three edges are 1/m, 1/n, and 1/p, the length of that body diagonal is equal to
    120 √(2/(m² + p² - n²) + 2/(m² - p² + n²) + 2/(-m² + p² + n²)). -/
theorem rectangular_prism_diagonal_length (m n p : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) (hp : p ≠ 0) :
  ∃ (a b c : ℝ),
    (a > 0 ∧ b > 0 ∧ c > 0) ∧
    (1 / a^2 + 1 / b^2 = m^2) ∧
    (1 / b^2 + 1 / c^2 = n^2) ∧
    (1 / c^2 + 1 / a^2 = p^2) ∧
    Real.sqrt (a^2 + b^2 + c^2) = body_diagonal_length m n p :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_prism_diagonal_length_l728_72811
