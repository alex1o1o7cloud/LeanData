import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_students_only_in_science_l67_6769

theorem students_only_in_science (total science art : ℕ)
  (h_total : total = 120)
  (h_science : science = 80)
  (h_art : art = 65)
  (h_all_enrolled : total ≤ science + art) :
  (science - (science + art - total)) = 55 := by
  sorry

#check students_only_in_science

end NUMINAMATH_CALUDE_ERRORFEEDBACK_students_only_in_science_l67_6769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pearl_finding_theorem_l67_6719

/-- Represents a circular cake -/
structure Cake where
  radius : ℝ

/-- Represents a pearl hidden in the cake -/
structure Pearl where
  radius : ℝ

/-- Represents a straight cut on the cake -/
structure Cut where
  -- We don't need to define the specifics of a cut for this problem

/-- Helper function to determine if a pearl is found given a list of cuts and its location -/
def pearlFound (cuts : List Cut) (pearlLocation : ℝ × ℝ) : Prop :=
  sorry  -- Definition omitted as it's not directly relevant to the theorem statement

/-- Function to determine if a pearl is definitely found after a given number of cuts -/
def isPearlDefinitelyFound (cake : Cake) (pearl : Pearl) (numCuts : ℕ) : Prop :=
  ∀ (pearlLocation : ℝ × ℝ), 
    pearlLocation.1^2 + pearlLocation.2^2 ≤ cake.radius^2 →
    ∃ (cuts : List Cut), cuts.length = numCuts ∧ pearlFound cuts pearlLocation

/-- Function to determine if it's possible that a pearl is not found after a given number of cuts -/
def isPearlPossiblyNotFound (cake : Cake) (pearl : Pearl) (numCuts : ℕ) : Prop :=
  ∃ (pearlLocation : ℝ × ℝ), 
    pearlLocation.1^2 + pearlLocation.2^2 ≤ cake.radius^2 ∧
    ∀ (cuts : List Cut), cuts.length = numCuts → ¬pearlFound cuts pearlLocation

/-- The main theorem to be proved -/
theorem pearl_finding_theorem (cake : Cake) (pearl : Pearl) 
    (h_cake_radius : cake.radius = 10)
    (h_pearl_radius : pearl.radius = 0.3) : 
    isPearlPossiblyNotFound cake pearl 32 ∧ 
    isPearlDefinitelyFound cake pearl 33 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pearl_finding_theorem_l67_6719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_theta_plus_seven_twelfths_pi_l67_6767

theorem cos_theta_plus_seven_twelfths_pi 
  (θ : ℝ) 
  (h1 : θ ∈ Set.Ioo (-3/2 * Real.pi) (-Real.pi)) 
  (h2 : Real.sin (θ + Real.pi/4) = 1/4) : 
  Real.cos (θ + 7/12 * Real.pi) = -(Real.sqrt 3 + Real.sqrt 15) / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_theta_plus_seven_twelfths_pi_l67_6767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blocks_needed_for_sculpture_l67_6759

/-- Represents the dimensions of a rectangular block -/
structure BlockDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents the dimensions of a cylindrical sculpture -/
structure SculptureDimensions where
  height : ℝ
  diameter : ℝ

/-- Calculates the volume of a rectangular block -/
def blockVolume (d : BlockDimensions) : ℝ :=
  d.length * d.width * d.height

/-- Calculates the volume of a cylindrical sculpture -/
noncomputable def sculptureVolume (d : SculptureDimensions) : ℝ :=
  Real.pi * (d.diameter / 2) ^ 2 * d.height

/-- Calculates the number of whole blocks needed for the sculpture -/
noncomputable def blocksNeeded (block : BlockDimensions) (sculpture : SculptureDimensions) : ℕ :=
  (Int.ceil (sculptureVolume sculpture / blockVolume block)).toNat

theorem blocks_needed_for_sculpture
  (block : BlockDimensions)
  (sculpture : SculptureDimensions)
  (h1 : block.length = 8)
  (h2 : block.width = 3)
  (h3 : block.height = 1.5)
  (h4 : sculpture.height = 9)
  (h5 : sculpture.diameter = 5) :
  blocksNeeded block sculpture = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_blocks_needed_for_sculpture_l67_6759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_m_values_l67_6706

-- Define the ellipse equation
def ellipse_equation (x y m : ℝ) : Prop := x^2 / 16 + y^2 / m = 1

-- Define the focal length
noncomputable def focal_length : ℝ := 2 * Real.sqrt 7

-- Theorem statement
theorem ellipse_m_values :
  ∃ (m : ℝ), (∀ x y, ellipse_equation x y m) ∧ 
  (m = 9 ∨ m = 23) ∧
  (focal_length = 2 * Real.sqrt (max (16 - m) (m - 16))) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_m_values_l67_6706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_joshua_profit_l67_6772

def orange_count : ℕ := 25
def apple_count : ℕ := 15
def banana_count : ℕ := 10

def orange_cost : ℚ := 1250 / 100  -- $12.50 in decimal form
def apple_cost : ℚ := 975 / 100    -- $9.75 in decimal form
def banana_cost : ℚ := 350 / 100   -- $3.50 in decimal form

def orange_price : ℕ := 60  -- 60 cents
def apple_price : ℕ := 75   -- 75 cents
def banana_price : ℕ := 45  -- 45 cents

def total_cost : ℚ := orange_cost + apple_cost + banana_cost

def total_revenue : ℕ := 
  orange_count * orange_price + 
  apple_count * apple_price + 
  banana_count * banana_price

theorem joshua_profit : 
  total_revenue - (total_cost * 100).floor = 500 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_joshua_profit_l67_6772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_two_extreme_points_l67_6708

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * (log x - 2 * a * x)

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := log x + 1 - 4 * a * x

theorem range_of_a_for_two_extreme_points :
  ∃ a : ℝ, 0 < a ∧ a < 1/4 ∧
  (∃ x₁ x₂ : ℝ, 0 < x₁ ∧ 0 < x₂ ∧ x₁ ≠ x₂ ∧ 
   g a x₁ = 0 ∧ g a x₂ = 0 ∧
   ∀ x, 0 < x → g a x ≤ 0 → x = x₁ ∨ x = x₂) := by
  sorry -- Proof to be filled in

#check range_of_a_for_two_extreme_points

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_two_extreme_points_l67_6708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_seven_pointed_star_angle_l67_6711

/-- The measure of the angle between two adjacent points in a regular seven-pointed star -/
noncomputable def angle_between_points_in_seven_pointed_star : ℝ :=
  5 * Real.pi / 7

/-- Theorem: In a regular seven-pointed star, the measure of the angle between two adjacent points is 5π/7 radians -/
theorem regular_seven_pointed_star_angle :
  angle_between_points_in_seven_pointed_star = 5 * Real.pi / 7 := by
  -- Unfold the definition of angle_between_points_in_seven_pointed_star
  unfold angle_between_points_in_seven_pointed_star
  -- The equality now holds by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_seven_pointed_star_angle_l67_6711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_between_specific_spheres_l67_6732

/-- The volume of a sphere with radius r -/
noncomputable def sphereVolume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

/-- The volume of the space between two concentric spheres -/
noncomputable def volumeBetweenSpheres (r₁ r₂ : ℝ) : ℝ := sphereVolume r₂ - sphereVolume r₁

theorem volume_between_specific_spheres :
  volumeBetweenSpheres 4 8 = (1792 / 3) * Real.pi := by
  -- Expand the definitions
  unfold volumeBetweenSpheres sphereVolume
  -- Simplify the expression
  simp [Real.pi]
  -- The proof is completed with sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_between_specific_spheres_l67_6732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_theorem_l67_6781

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle -/
structure Circle where
  center : Point
  radius : ℝ

/-- Checks if a point lies on a given circle -/
def Point.onCircle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

/-- Checks if a point lies on a line segment between two other points -/
def Point.onSegment (p q r : Point) : Prop :=
  (q.x ≤ p.x ∧ p.x ≤ r.x) ∨ (r.x ≤ p.x ∧ p.x ≤ q.x)

/-- Calculates the distance between two points -/
noncomputable def Point.distance (p q : Point) : ℝ :=
  ((p.x - q.x)^2 + (p.y - q.y)^2).sqrt

/-- Checks if two line segments intersect -/
def segmentsIntersect (p1 q1 p2 q2 : Point) : Prop :=
  ∃ (i : Point), i.onSegment p1 q1 ∧ i.onSegment p2 q2

/-- Calculates the angle between three points -/
noncomputable def angle (p q r : Point) : ℝ := sorry

theorem semicircle_theorem (A B C D E F : Point) (circle : Circle) :
  C.onCircle circle →
  circle.center = Point.mk ((A.x + B.x) / 2) ((A.y + B.y) / 2) →
  circle.radius = A.distance B / 2 →
  C.onSegment A B →
  D.onSegment B C →
  F.onSegment A C →
  segmentsIntersect B F A D →
  B.distance D = A.distance C →
  angle B E D = π / 4 →
  A.distance F = C.distance D := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_theorem_l67_6781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_problem_l67_6768

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem vector_magnitude_problem (a b : V) 
  (h1 : ‖a - b‖ = Real.sqrt 3)
  (h2 : ‖a + b‖ = ‖2 • a - b‖) :
  ‖b‖ = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_problem_l67_6768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_and_difference_min_l67_6712

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 2 * x + 1

-- Define the domain of f
def domain : Set ℝ := Set.Icc 1 3

-- Define M(a) as the maximum value of f(x) in the domain
noncomputable def M (a : ℝ) : ℝ := ⨆ x ∈ domain, f a x

-- Define N(a) as the minimum value of f(x) in the domain
noncomputable def N (a : ℝ) : ℝ := ⨅ x ∈ domain, f a x

theorem min_value_and_difference_min (a : ℝ) (h : 1/3 ≤ a ∧ a ≤ 1) :
  (∀ x ∈ domain, N a ≤ f a x) ∧
  N a = 1 - 1/a ∧
  ∃ (m : ℝ), m = ⨅ a ∈ Set.Icc (1/3) 1, (M a - N a) ∧ m = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_and_difference_min_l67_6712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_angle_between_vectors_perpendicular_vectors_k_values_l67_6795

def a : Fin 3 → ℝ := ![3, 2, -1]
def b : Fin 3 → ℝ := ![2, 1, 2]

def dot_product (v w : Fin 3 → ℝ) : ℝ :=
  (v 0) * (w 0) + (v 1) * (w 1) + (v 2) * (w 2)

noncomputable def magnitude (v : Fin 3 → ℝ) : ℝ :=
  Real.sqrt ((v 0)^2 + (v 1)^2 + (v 2)^2)

theorem cosine_angle_between_vectors :
  dot_product a b / (magnitude a * magnitude b) = 2 / Real.sqrt 14 := by sorry

theorem perpendicular_vectors_k_values :
  ∀ k : ℝ, dot_product (fun i => k * (a i) + (b i)) (fun i => (a i) - k * (b i)) = 0 ↔ k = 3/2 ∨ k = -2/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_angle_between_vectors_perpendicular_vectors_k_values_l67_6795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_diagonal_trapezoid_area_l67_6753

/-- A trapezoid with perpendicular diagonals -/
structure PerpendicularDiagonalTrapezoid where
  /-- One of the diagonals -/
  diagonal : ℝ
  /-- Height of the trapezoid -/
  height : ℝ
  /-- The diagonals are perpendicular -/
  diagonals_perpendicular : True

/-- The area of a trapezoid with perpendicular diagonals -/
noncomputable def area (t : PerpendicularDiagonalTrapezoid) : ℝ :=
  (1014 : ℝ) / 5

/-- Theorem: If a trapezoid has perpendicular diagonals, one diagonal is 13, and height is 12, then its area is 1014/5 -/
theorem perpendicular_diagonal_trapezoid_area 
  (t : PerpendicularDiagonalTrapezoid) 
  (h1 : t.diagonal = 13) 
  (h2 : t.height = 12) : 
  area t = 1014 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_diagonal_trapezoid_area_l67_6753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_first_ten_terms_l67_6710

def x : ℕ → ℚ
  | 0 => 2001  -- Define x for 0 to cover all natural numbers
  | 1 => 2001
  | n+2 => (n+2 : ℚ) / x (n+1)

theorem product_of_first_ten_terms : x 1 * x 2 * x 3 * x 4 * x 5 * x 6 * x 7 * x 8 * x 9 * x 10 = 3840 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_first_ten_terms_l67_6710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_perimeter_l67_6730

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculate the distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Definition of the polygon UVWXY -/
structure Polygon where
  U : Point
  V : Point
  W : Point
  X : Point
  Y : Point

/-- The conditions of the problem -/
def polygon_conditions (p : Polygon) : Prop :=
  distance p.U p.X = 7 ∧
  distance p.U p.V = 5 ∧
  distance p.X p.Y = 9 ∧
  (p.U.x - p.X.x) * (p.Y.x - p.X.x) + (p.U.y - p.X.y) * (p.Y.y - p.X.y) = 0 ∧
  (p.X.x - p.U.x) * (p.V.x - p.U.x) + (p.X.y - p.U.y) * (p.V.y - p.U.y) = 0 ∧
  (p.U.x - p.V.x) * (p.W.x - p.V.x) + (p.U.y - p.V.y) * (p.W.y - p.V.y) = 0 ∧
  (p.V.y - p.W.y) / (p.V.x - p.W.x) = (p.X.y - p.Y.y) / (p.X.x - p.Y.x)

/-- The theorem to be proved -/
theorem polygon_perimeter (p : Polygon) :
  polygon_conditions p →
  distance p.U p.V + distance p.V p.W + distance p.W p.Y +
  distance p.Y p.X + distance p.X p.U = 29 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_perimeter_l67_6730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_theorem_l67_6703

noncomputable section

/-- Parabola defined by y^2 = 4x -/
def parabola (p : ℝ × ℝ) : Prop := p.2^2 = 4 * p.1

/-- Focus of the parabola y^2 = 4x -/
def focus : ℝ × ℝ := (1, 0)

/-- Point B -/
def B : ℝ × ℝ := (3, 0)

/-- Distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem parabola_distance_theorem (A : ℝ × ℝ) :
  parabola A →
  distance A focus = distance B focus →
  distance A B = 2 * Real.sqrt 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_theorem_l67_6703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_sets_between_sets_between_count_l67_6778

theorem number_of_sets_between (A B : Finset Nat) (hAB : A ⊆ B) : 
  (Finset.filter (fun S => A ⊆ S ∧ S ⊆ B) (Finset.powerset B)).card = 2 ^ (B.card - A.card) :=
by sorry

theorem sets_between_count : 
  (Finset.filter (fun S => {1, 2} ⊆ S ∧ S ⊆ {1, 2, 3, 4}) (Finset.powerset {1, 2, 3, 4})).card = 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_sets_between_sets_between_count_l67_6778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shift_for_odd_sine_function_l67_6788

theorem min_shift_for_odd_sine_function (φ : ℝ) : 
  (∀ x, Real.sin (2 * x - 2 * φ + π / 3) = -Real.sin (-2 * x + 2 * φ - π / 3)) →
  φ ≥ π / 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shift_for_odd_sine_function_l67_6788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_prime_count_l67_6751

def polynomial (n : ℕ) : ℤ := n^3 - 6*n^2 + 17*n - 19

theorem polynomial_prime_count :
  ∃! (s : Finset ℕ), (∀ n ∈ s, Nat.Prime (Int.natAbs (polynomial n))) ∧ s.card = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_prime_count_l67_6751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_grid_filling_exists_l67_6718

-- Define the type for our grid function
def GridFunction := ℕ → ℕ → ℕ

-- Define what it means for a function to be bijective on rows and columns
def BijectiveOnRowsAndColumns (f : GridFunction) : Prop :=
  (∀ n : ℕ, Function.Bijective (λ m ↦ f n m)) ∧
  (∀ m : ℕ, Function.Bijective (λ n ↦ f n m))

-- State the theorem
theorem infinite_grid_filling_exists :
  ∃ f : GridFunction, BijectiveOnRowsAndColumns f := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_grid_filling_exists_l67_6718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_disjoint_subset_pairs_remainder_l67_6797

def S : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}

def disjointSubsetPairs (X : Finset ℕ) : ℕ :=
  (3^X.card - 2 * 2^X.card + 1) / 2

theorem disjoint_subset_pairs_remainder (n : ℕ) : 
  n = disjointSubsetPairs S → n % 1000 = 625 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_disjoint_subset_pairs_remainder_l67_6797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_color_plane_theorem_l67_6770

/-- A coloring function that assigns a color (0 or 1) to each point in ℝ² -/
def ColoringFunction := ℝ × ℝ → Fin 2

/-- Represents a line segment in ℝ² -/
structure LineSegment where
  start : ℝ × ℝ
  endpoint : ℝ × ℝ

/-- A point lies on a given line segment -/
def PointOnLineSegment (p : ℝ × ℝ) (seg : LineSegment) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ p = (1 - t) • seg.start + t • seg.endpoint

theorem two_color_plane_theorem :
  ∃ (f : ColoringFunction),
    ∀ (seg : LineSegment),
      ∃ (p q : ℝ × ℝ),
        PointOnLineSegment p seg ∧
        PointOnLineSegment q seg ∧
        f p ≠ f q :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_color_plane_theorem_l67_6770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_9_eq_one_l67_6756

/-- The sequence x_n defined by the problem -/
noncomputable def x (n : ℕ) : ℤ :=
  ⌊(n + 1 : ℝ) * Real.sqrt (2013 / 2014)⌋ - ⌊(n : ℝ) * Real.sqrt (2013 / 2014)⌋

/-- The main theorem stating that x_9 = 1 -/
theorem x_9_eq_one : x 9 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_9_eq_one_l67_6756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_length_is_two_l67_6746

/-- The circle with center (1, 1) and radius 1 -/
def my_circle (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 1

/-- The point P outside the circle -/
def P : ℝ × ℝ := (2, 3)

/-- The length of the tangent line from P to the circle -/
noncomputable def tangent_length : ℝ :=
  Real.sqrt ((P.1 - 1)^2 + (P.2 - 1)^2 - 1)

theorem tangent_length_is_two : tangent_length = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_length_is_two_l67_6746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_properties_l67_6760

/-- A rhombus with diagonals of length d₁ and d₂ -/
structure Rhombus where
  d₁ : ℝ
  d₂ : ℝ

/-- The area of a rhombus -/
noncomputable def area (r : Rhombus) : ℝ := r.d₁ * r.d₂ / 2

/-- Whether the diagonals of a rhombus are perpendicular -/
def perpendicular_diagonals (r : Rhombus) : Prop :=
  (r.d₁ / 2) ^ 2 + (r.d₂ / 2) ^ 2 = ((r.d₁ ^ 2 + r.d₂ ^ 2) / 4)

theorem rhombus_properties (r : Rhombus) (h₁ : r.d₁ = 18) (h₂ : r.d₂ = 26) :
  area r = 234 ∧ perpendicular_diagonals r := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_properties_l67_6760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_strategy_l67_6776

/-- The probability of a messenger being robbed -/
def p : ℝ := sorry

/-- Assumption that p is between 0 and 1 -/
axiom h_p_bounds : 0 < p ∧ p < 1

/-- Probability of success for 2 messengers strategy -/
def prob_2 : ℝ := 1 - p^2

/-- Probability of success for 3 messengers strategy -/
def prob_3 : ℝ := 1 - p^2 * (2 - p)

/-- Probability of success for 4 messengers strategy -/
def prob_4 : ℝ := 1 - p^3 * (4 - 3*p)

/-- Theorem stating the optimal strategy -/
theorem optimal_strategy :
  (0 < p ∧ p < 1/3 → prob_4 > max prob_2 prob_3) ∧
  (1/3 ≤ p ∧ p < 1 → prob_2 ≥ max prob_3 prob_4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_strategy_l67_6776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_g_zeros_range_l67_6729

noncomputable def f (a x : ℝ) : ℝ := -1/2 * a * x^2 + (1+a) * x - Real.log x

noncomputable def g (x k : ℝ) : ℝ := x * f 0 x - k * (x+2) + 2

def monotone_decreasing (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ x y, x ∈ s → y ∈ s → x < y → f y < f x

theorem f_monotonicity (a : ℝ) :
  (0 < a ∧ a < 1 → monotone_decreasing (f a) (Set.Ioo 0 1 ∪ Set.Ioi (1/a))) ∧
  (a = 1 → monotone_decreasing (f a) (Set.Ioi 0)) ∧
  (a > 1 → monotone_decreasing (f a) (Set.Ioo 0 (1/a) ∪ Set.Ioi 1)) :=
sorry

def has_two_zeros (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∃ x y, x ∈ s ∧ y ∈ s ∧ x ≠ y ∧ f x = 0 ∧ f y = 0

theorem g_zeros_range (k : ℝ) :
  has_two_zeros (fun x ↦ g x k) (Set.Ici (1/2)) ↔ k ∈ Set.Ioo 1 (9/10 + Real.log 2 / 5) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_g_zeros_range_l67_6729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_not_shaded_value_l67_6738

/-- Represents a rectangle in the 2 by 1003 grid -/
structure Rectangle where
  left : Nat
  right : Nat
  top : Nat
  bottom : Nat

/-- The total width of the rectangle -/
def grid_width : Nat := 1003

/-- The total height of the rectangle -/
def grid_height : Nat := 2

/-- The position of the shaded square in each row -/
def shaded_position : Nat := (grid_width + 1) / 2

/-- Checks if a rectangle contains a shaded square -/
def contains_shaded (r : Rectangle) : Prop :=
  (r.left ≤ shaded_position) ∧ (shaded_position ≤ r.right) ∧ (r.top ≤ 1) ∧ (r.bottom ≥ 1)

/-- The total number of possible rectangles -/
def total_rectangles : Nat := (grid_width + 1).choose 2 * 3

/-- The number of rectangles containing a shaded square -/
def shaded_rectangles : Nat := (shaded_position * (grid_width + 1 - shaded_position)) * 3

/-- The probability of choosing a rectangle that does not include a shaded square -/
noncomputable def probability_not_shaded : ℚ :=
  1 - (shaded_rectangles : ℚ) / (total_rectangles : ℚ)

theorem probability_not_shaded_value :
  probability_not_shaded = 501 / 1003 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_not_shaded_value_l67_6738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_sum_theorem_l67_6798

def is_valid_grid (grid : Matrix (Fin 3) (Fin 3) Nat) : Prop :=
  (∀ i j, grid i j ∈ Finset.range 9) ∧
  (∀ x, x ∈ Finset.range 9 → ∃ i j, grid i j = x)

def sum_of_four (grid : Matrix (Fin 3) (Fin 3) Nat) (i j k l : Fin 9) : Nat :=
  grid (i / 3) (i % 3) +
  grid (j / 3) (j % 3) +
  grid (k / 3) (k % 3) +
  grid (l / 3) (l % 3)

def five_digit_number (grid : Matrix (Fin 3) (Fin 3) Nat) : Nat :=
  grid 0 0 * 10000 +
  grid 0 2 * 1000 +
  grid 1 1 * 100 +
  grid 2 0 * 10 +
  grid 2 2

theorem grid_sum_theorem (grid : Matrix (Fin 3) (Fin 3) Nat) :
  is_valid_grid grid →
  sum_of_four grid 0 1 3 4 = 28 →
  five_digit_number grid = 71925 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_sum_theorem_l67_6798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_min_value_of_a_l67_6799

open Real

variable (A B C : ℝ) -- Angles of the triangle
variable (a b c : ℝ) -- Sides of the triangle opposite to A, B, C respectively

-- Define the conditions
def condition1 (A B C a b c : ℝ) : Prop := sqrt 3 * c * cos A = a * sin C
def condition2 (B C c : ℝ) : Prop := 4 * sin C = c^2 * sin B
def condition3 (b c : ℝ) : Prop := b * c * cos (π/3) = 4

-- Define the area of the triangle
noncomputable def triangle_area (A b c : ℝ) : ℝ := (1/2) * b * c * sin A

-- Theorem for the first question
theorem area_of_triangle {A B C a b c : ℝ} 
  (h1 : condition1 A B C a b c) (h2 : condition2 B C c) :
  triangle_area A b c = sqrt 3 := by sorry

-- Theorem for the second question
theorem min_value_of_a {A B C a b c : ℝ} 
  (h1 : condition1 A B C a b c) (h3 : condition3 b c) :
  ∃ (min_a : ℝ), min_a = 2 * sqrt 2 ∧ ∀ (a' : ℝ), a' ≥ min_a := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_min_value_of_a_l67_6799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_difference_to_halfway_point_l67_6702

/-- The time it takes Danny to reach Steve's house in minutes -/
noncomputable def danny_time : ℝ := 27

/-- The time it takes Steve to reach Danny's house in minutes -/
noncomputable def steve_time : ℝ := 2 * danny_time

/-- The time it takes Danny to reach the halfway point -/
noncomputable def danny_halfway_time : ℝ := danny_time / 2

/-- The time it takes Steve to reach the halfway point -/
noncomputable def steve_halfway_time : ℝ := steve_time / 2

theorem time_difference_to_halfway_point : 
  steve_halfway_time - danny_halfway_time = 13.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_difference_to_halfway_point_l67_6702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_three_implies_expression_value_l67_6713

theorem tan_three_implies_expression_value (x : ℝ) (h : Real.tan x = 3) :
  1 / (Real.sin x ^ 2 - 2 * Real.cos x ^ 2) = 10 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_three_implies_expression_value_l67_6713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_apples_calculation_average_apples_result_l67_6790

-- Define the number of apples eaten
noncomputable def apples_eaten : ℝ := 5.0

-- Define the number of hours spent eating
noncomputable def hours_spent : ℝ := 3.0

-- Define the average apples eaten per hour
noncomputable def average_apples_per_hour : ℝ := apples_eaten / hours_spent

-- Theorem statement
theorem average_apples_calculation :
  average_apples_per_hour = apples_eaten / hours_spent :=
by
  -- Unfold the definition of average_apples_per_hour
  unfold average_apples_per_hour
  -- The equality now holds by reflexivity
  rfl

-- Theorem to show the numerical result
theorem average_apples_result :
  average_apples_per_hour = 1.6666666666666667 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_apples_calculation_average_apples_result_l67_6790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_otimes_expression_equals_seven_twelfths_l67_6758

/-- The ⊗ operation for nonzero real numbers -/
noncomputable def otimes (a b : ℝ) : ℝ := a^3 / b

/-- Theorem stating the result of the given expression -/
theorem otimes_expression_equals_seven_twelfths :
  ∀ (a b c : ℝ), a ≠ 0 → b ≠ 0 → c ≠ 0 →
  (otimes (otimes a b) c) - (otimes a (otimes b c)) = 7/12 :=
by
  intros a b c ha hb hc
  -- The proof steps would go here
  sorry

#check otimes_expression_equals_seven_twelfths

end NUMINAMATH_CALUDE_ERRORFEEDBACK_otimes_expression_equals_seven_twelfths_l67_6758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chords_inequality_l67_6701

def semicircle_chords (a b c d : ℝ) : Prop :=
  ∃ (α β γ δ : ℝ),
    0 < α ∧ 0 < β ∧ 0 < γ ∧ 0 < δ ∧
    α + β + γ + δ = Real.pi ∧
    a = 2 * Real.sin α ∧
    b = 2 * Real.sin β ∧
    c = 2 * Real.sin γ ∧
    d = 2 * Real.sin δ

theorem chords_inequality (a b c d : ℝ) (h : semicircle_chords a b c d) :
  a^2 + b^2 + c^2 + d^2 + a*b*c + b*c*d < 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chords_inequality_l67_6701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_x_coordinate_l67_6744

noncomputable section

/-- A parabola with equation y² = (1/4)x -/
structure Parabola where
  equation : ∀ x y, y^2 = (1/4) * x

/-- The focus of the parabola -/
def focus : ℝ × ℝ := (1/16, 0)

/-- A point on the parabola -/
structure PointOnParabola (p : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : y^2 = (1/4) * x

/-- Distance between two points in ℝ² -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem parabola_point_x_coordinate 
  (p : Parabola) 
  (M : PointOnParabola p) 
  (h : distance (M.x, M.y) focus = 1) : 
  M.x = 15/16 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_x_coordinate_l67_6744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_beautiful_heel_height_theorem_l67_6749

/-- The golden ratio for beauty -/
def goldenRatioBeauty : ℝ := 0.618

/-- The girl's current height in cm -/
def currentHeight : ℝ := 157

/-- The girl's lower limbs length in cm -/
def lowerLimbsLength : ℝ := 95

/-- The height of high heels that makes the girl look most beautiful -/
def beautifulHeelHeight : ℝ := 5.3

theorem beautiful_heel_height_theorem :
  abs ((lowerLimbsLength + beautifulHeelHeight) / (currentHeight + beautifulHeelHeight) - goldenRatioBeauty) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_beautiful_heel_height_theorem_l67_6749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_team_average_age_is_24_l67_6777

/-- Represents a cricket team with given properties -/
structure CricketTeam where
  total_members : ℕ
  captain_age : ℕ
  wicket_keeper_age : ℕ
  average_age : ℚ
  remaining_average_age : ℚ

/-- The properties of the cricket team as described in the problem -/
def team (average_age : ℚ) : CricketTeam where
  total_members := 11
  captain_age := 27
  wicket_keeper_age := 30
  average_age := average_age
  remaining_average_age := average_age - 1

/-- Theorem stating that the average age of the team is 24 years -/
theorem team_average_age_is_24 :
  ∃ (average_age : ℚ), team average_age = { total_members := 11,
                                            captain_age := 27,
                                            wicket_keeper_age := 30,
                                            average_age := 24,
                                            remaining_average_age := 23 } := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_team_average_age_is_24_l67_6777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_increase_by_six_l67_6794

/-- Represents a cricket player's statistics -/
structure CricketPlayer where
  current_innings : ℕ
  current_average : ℚ
  next_innings_runs : ℕ

/-- Calculates the new average after playing an additional innings -/
def new_average (player : CricketPlayer) : ℚ :=
  (player.current_average * player.current_innings + player.next_innings_runs) / (player.current_innings + 1)

/-- Theorem: A player with 25 runs average in 15 innings who scores 121 in the next innings increases their average by 6 runs -/
theorem average_increase_by_six (player : CricketPlayer)
  (h1 : player.current_innings = 15)
  (h2 : player.current_average = 25)
  (h3 : player.next_innings_runs = 121) :
  new_average player - player.current_average = 6 := by
  sorry

#eval new_average { current_innings := 15, current_average := 25, next_innings_runs := 121 } - 25

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_increase_by_six_l67_6794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ladder_length_l67_6722

theorem ladder_length (θ : Real) (a : Real) (h : Real) :
  θ = 60 * Real.pi / 180 →  -- Convert 60° to radians
  a = 6.4 →  -- Adjacent side length
  Real.cos θ = a / h →  -- Cosine relation in right triangle
  h = 12.8 :=  -- Length of the ladder (hypotenuse)
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ladder_length_l67_6722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_decreasing_subsequences_l67_6741

/-- The sequence x_n defined recursively -/
noncomputable def x (a : ℝ) : ℕ → ℝ
  | 0 => a  -- Add this case for n = 0
  | 1 => a
  | n + 2 => a^(x a (n + 1))

/-- The statement to be proved -/
theorem x_decreasing_subsequences (a : ℝ) (ha : 0 < a) (ha' : a < 1) :
  (∀ n : ℕ, x a (2*n + 1) > x a (2*n + 3)) ∧ 
  (∀ n : ℕ, x a (2*n + 2) > x a (2*n + 4)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_decreasing_subsequences_l67_6741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_is_330_l67_6783

/-- The sum of the digits in (10^38) - 85 when written as a base 10 integer -/
def sum_of_digits : ℕ :=
  let n : ℕ := 10^38 - 85
  (n.digits 10).sum

/-- Theorem stating that the sum of the digits in (10^38) - 85 is 330 -/
theorem sum_of_digits_is_330 : sum_of_digits = 330 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_is_330_l67_6783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_comparison_power_comparison_l67_6763

-- Part 1
theorem log_comparison : Real.log 7 / Real.log 6 > Real.log 6 / Real.log 7 := by sorry

-- Part 2
theorem power_comparison :
  (1/2 : ℝ)^(1/2) > (1/2 : ℝ)^(3/4) ∧ (1/2 : ℝ)^(3/4) > (1/5 : ℝ)^(3/4) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_comparison_power_comparison_l67_6763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_op_commutative_special_op_not_associative_l67_6733

/-- A binary operation on integers satisfying specific axioms -/
def special_op : ℤ → ℤ → ℤ := sorry

/-- Axiom 1: x * (x * y) = y for all x, y ∈ ℤ -/
axiom axiom1 (x y : ℤ) : special_op x (special_op x y) = y

/-- Axiom 2: (x * y) * y = x for all x, y ∈ ℤ -/
axiom axiom2 (x y : ℤ) : special_op (special_op x y) y = x

/-- Theorem: The operation is commutative -/
theorem special_op_commutative : ∀ x y : ℤ, special_op x y = special_op y x := by
  sorry

/-- Theorem: The operation is not associative -/
theorem special_op_not_associative : ¬(∀ x y z : ℤ, special_op (special_op x y) z = special_op x (special_op y z)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_op_commutative_special_op_not_associative_l67_6733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_harmonic_mean_pairs_l67_6791

open BigOperators

def harmonic_mean (a b : ℚ) : ℚ :=
  2 * (a * b) / (a + b)

def is_valid_pair (p : ℚ × ℚ) : Prop :=
  p.1 < p.2 ∧ p.1 > 0 ∧ p.2 > 0 ∧ harmonic_mean p.1 p.2 = 5^10

def count_pairs : ℕ := sorry

theorem harmonic_mean_pairs :
  count_pairs = 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_harmonic_mean_pairs_l67_6791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_mass_density_relation_density_is_constant_mass_is_variable_volume_is_variable_l67_6721

/-- Density of a substance -/
noncomputable def ρ : ℝ := sorry

/-- Mass of a substance -/
noncomputable def m : ℝ → ℝ := sorry

/-- Volume of a substance -/
noncomputable def V : ℝ → ℝ := sorry

/-- The relationship between volume, mass, and density -/
theorem volume_mass_density_relation (x : ℝ) :
  V x = m x / ρ := by sorry

/-- Density is constant for a given substance -/
theorem density_is_constant (x y : ℝ) :
  ρ = ρ := by sorry

/-- Mass can vary -/
theorem mass_is_variable :
  ∃ x y, m x ≠ m y := by sorry

/-- Volume can vary -/
theorem volume_is_variable :
  ∃ x y, V x ≠ V y := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_mass_density_relation_density_is_constant_mass_is_variable_volume_is_variable_l67_6721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_perp_two_planes_implies_planes_parallel_l67_6750

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Plane → Prop)
variable (para : Plane → Plane → Prop)

-- Define a relation for lines being distinct from planes
variable (distinct : Line → Plane → Prop)

-- State the theorem
theorem line_perp_two_planes_implies_planes_parallel 
  (m : Line) (α β : Plane) :
  distinct m α → distinct m β → α ≠ β → 
  perp m α → perp m β → para α β :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_perp_two_planes_implies_planes_parallel_l67_6750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_food_fraction_is_one_fifth_l67_6731

-- Define the salary and the fractions
noncomputable def salary : ℝ := 190000
noncomputable def rent_fraction : ℝ := 1/10
noncomputable def clothes_fraction : ℝ := 3/5
noncomputable def remaining_amount : ℝ := 19000

-- Define the theorem
theorem food_fraction_is_one_fifth :
  ∃ (food_fraction : ℝ),
    food_fraction * salary + rent_fraction * salary + clothes_fraction * salary + remaining_amount = salary ∧
    food_fraction = 1/5 := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_food_fraction_is_one_fifth_l67_6731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_symmetry_l67_6736

noncomputable def f (x : ℝ) := Real.sin (x - Real.pi/4)

theorem sine_symmetry (k : ℤ) :
  (∀ x, f (3*Real.pi/4 + k*Real.pi + x) = f (3*Real.pi/4 + k*Real.pi - x)) ∧
  (∀ x, f (Real.pi/4 + k*Real.pi + x) = -f (Real.pi/4 + k*Real.pi - x)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_symmetry_l67_6736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_speed_for_dan_l67_6725

/-- Represents the scenario of two drivers traveling between two cities -/
structure TravelScenario where
  distance : ℚ  -- Distance between cities in miles
  cara_speed : ℚ  -- Cara's constant speed in miles per hour
  dan_delay : ℚ  -- Dan's delay in hours

/-- Calculates the minimum speed Dan needs to exceed to arrive before Cara -/
def min_speed_to_arrive_first (scenario : TravelScenario) : ℚ :=
  scenario.distance / (scenario.distance / scenario.cara_speed - scenario.dan_delay)

/-- Theorem stating the minimum speed Dan must exceed -/
theorem min_speed_for_dan (scenario : TravelScenario) 
  (h1 : scenario.distance = 120)
  (h2 : scenario.cara_speed = 30)
  (h3 : scenario.dan_delay = 1) :
  min_speed_to_arrive_first scenario > 40 := by
  sorry

/-- Evaluation of the minimum speed for the given scenario -/
def evaluate_min_speed : ℚ :=
  min_speed_to_arrive_first { distance := 120, cara_speed := 30, dan_delay := 1 }

#eval evaluate_min_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_speed_for_dan_l67_6725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_of_rounding_methods_l67_6726

/-- Round a number to the hundredths place -/
noncomputable def roundToHundredths (x : ℝ) : ℝ :=
  ⌊x * 100 + 0.5⌋ / 100

/-- Calculate 'a' by rounding each number and then summing -/
noncomputable def calculateA (x y z : ℝ) : ℝ :=
  roundToHundredths x + roundToHundredths y + roundToHundredths z

/-- Calculate 'b' by summing and then rounding -/
noncomputable def calculateB (x y z : ℝ) : ℝ :=
  roundToHundredths (x + y + z)

theorem difference_of_rounding_methods (x y z : ℝ) 
  (hx : x = 13.165) (hy : y = 7.686) (hz : z = 11.545) : 
  calculateA x y z - calculateB x y z = 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_of_rounding_methods_l67_6726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_divisors_of_29_l67_6757

theorem sum_of_divisors_of_29 (h : Nat.Prime 29) : 
  (Finset.filter (λ x => x ∣ 29) (Finset.range 30)).sum id = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_divisors_of_29_l67_6757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ordering_of_logarithms_and_roots_l67_6796

theorem ordering_of_logarithms_and_roots : 
  ∃ (a b c : ℝ), 
    a = Real.log 5 / Real.log (1/2) ∧
    b = (1/3) ^ (1/5) ∧
    c = 2 ^ (1/3) ∧
    a < b ∧ b < c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ordering_of_logarithms_and_roots_l67_6796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_for_f_l67_6714

noncomputable def f (x : ℝ) : ℝ := if x > 0 then 2 * x else x + 1

theorem unique_solution_for_f : ∀ a : ℝ, f a + f 1 = 0 ↔ a = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_for_f_l67_6714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_rotation_power_l67_6793

noncomputable def rotation_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![1/2, Real.sqrt 3 / 2],
    ![-Real.sqrt 3 / 2, 1/2]]

theorem smallest_rotation_power :
  (∃ (n : ℕ), n > 0 ∧ rotation_matrix ^ n = 1) ∧
  (∀ (m : ℕ), 0 < m ∧ m < 6 → rotation_matrix ^ m ≠ 1) ∧
  rotation_matrix ^ 6 = 1 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_rotation_power_l67_6793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_has_four_digits_l67_6737

/-- A nonzero digit is a natural number between 1 and 9. -/
def NonzeroDigit : Type := {n : ℕ // 1 ≤ n ∧ n ≤ 9}

/-- The sum of 8765, C43, and D2 for any nonzero digits C and D. -/
def SumDigits (C D : NonzeroDigit) : ℕ := 8765 + (C.val * 100 + 43) + (D.val * 10 + 2)

/-- The number of digits in a natural number. -/
def numDigits (n : ℕ) : ℕ := (Nat.log n 10) + 1

theorem sum_has_four_digits (C D : NonzeroDigit) : numDigits (SumDigits C D) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_has_four_digits_l67_6737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_rectangular_equivalence_l67_6723

/-- The polar coordinate equation ρ = (2 + 2cos θ) / (sin² θ) is equivalent to the rectangular coordinate equation y² = x + 1. -/
theorem polar_to_rectangular_equivalence :
  ∀ (x y : ℝ) (θ : ℝ) (ρ : ℝ),
    ρ > 0 →
    x = ρ * Real.cos θ →
    y = ρ * Real.sin θ →
    ρ = (2 + 2 * Real.cos θ) / (Real.sin θ)^2 →
    y^2 = x + 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_rectangular_equivalence_l67_6723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_conclusions_correct_l67_6761

-- Define the polynomial type
def MyPolynomial := ℤ → ℤ

-- Define the operation to generate new polynomials
def generate_new_polynomial (p q : MyPolynomial) : MyPolynomial :=
  fun x => q x - p x

-- Define the function to generate the nth polynomial string
def generate_nth_string : ℕ → List MyPolynomial
  | 0 => [(fun x => x), (fun x => x + 6), (fun x => x - 3)]
  | n + 1 => 
    let prev := generate_nth_string n
    List.join (List.zipWith (fun p q => [p, generate_new_polynomial p q, q]) prev (List.tail prev))

-- Define the function to sum all polynomials in a string
def sum_polynomials (ps : List MyPolynomial) : MyPolynomial :=
  fun x => ps.foldl (fun acc p => acc + p x) 0

-- State the theorem
theorem all_conclusions_correct :
  -- Conclusion 1
  generate_nth_string 2 = [
    (fun x => x),
    (fun x => 6 - x),
    (fun x => 6),
    (fun x => x),
    (fun x => x + 6),
    (fun x => -x - 15),
    (fun x => -9),
    (fun x => x + 6),
    (fun x => x - 3)
  ] ∧
  -- Conclusion 2
  ∀ x, (sum_polynomials (generate_nth_string 3)) x = 
       (sum_polynomials (generate_nth_string 2)) x - 3 ∧
  -- Conclusion 3
  (generate_nth_string 5).length = 65 ∧
  -- Conclusion 4
  ∀ x, (sum_polynomials (generate_nth_string 2024)) x = 3*x - 6069 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_conclusions_correct_l67_6761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_on_plane_and_nearest_l67_6773

/-- The plane equation -/
def plane_equation (x y z : ℝ) : Prop := 3 * x - 4 * y + 5 * z = 30

/-- The given point -/
def given_point : ℝ × ℝ × ℝ := (1, 2, 3)

/-- The claimed closest point -/
noncomputable def closest_point : ℝ × ℝ × ℝ := (11/5, 2/5, 5)

/-- Theorem stating that the closest_point is on the plane and is closest to the given_point -/
theorem closest_point_on_plane_and_nearest :
  plane_equation closest_point.1 closest_point.2.1 closest_point.2.2 ∧
  ∀ (p : ℝ × ℝ × ℝ), plane_equation p.1 p.2.1 p.2.2 →
    Real.sqrt ((p.1 - given_point.1)^2 + (p.2.1 - given_point.2.1)^2 + (p.2.2 - given_point.2.2)^2) ≥
    Real.sqrt ((closest_point.1 - given_point.1)^2 + (closest_point.2.1 - given_point.2.1)^2 + (closest_point.2.2 - given_point.2.2)^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_on_plane_and_nearest_l67_6773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l67_6745

noncomputable section

-- Define the linear function
def linear_function (k b : ℝ) (x : ℝ) : ℝ := k * x + b

-- Define the inverse proportion function
def inverse_proportion (m : ℝ) (x : ℝ) : ℝ := m / x

-- Define the conditions
def conditions (k b m : ℝ) : Prop :=
  ∃ a : ℝ,
    linear_function k b 2 = 2 ∧
    inverse_proportion m 2 = 2 ∧
    linear_function k b (-1) = a ∧
    inverse_proportion m (-1) = a

-- Theorem statement
theorem function_properties :
  ∀ k b m : ℝ,
    conditions k b m →
    (∀ x : ℝ, linear_function k b x = 2 * x - 2) ∧
    (∀ x : ℝ, inverse_proportion m x = 4 / x) ∧
    (∀ h : ℝ, (h > 2 ∨ (-1 < h ∧ h < 0)) ↔ linear_function k b h > inverse_proportion m h) ∧
    (∀ h : ℝ, linear_function k b h - inverse_proportion m h = 2 ↔ h = 1 + Real.sqrt 3 ∨ h = 1 - Real.sqrt 3) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l67_6745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_reflection_l67_6782

/-- Given a parabola with equation y = 2(x-3)^2 - 5, 
    prove that y = -2(x-3)^2 + 5 is its reflection about the x-axis -/
theorem parabola_reflection :
  ∀ x y : ℝ, y = 2*(x-3)^2 - 5 → 
  y = -2*(x-3)^2 + 5 ↔ y = -(2*(x-3)^2 - 5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_reflection_l67_6782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_rhombus_l67_6766

/-- The radius of the inscribed circle in a rhombus with given diagonals -/
theorem inscribed_circle_radius_rhombus (d1 d2 : ℝ) (h1 : d1 > 0) (h2 : d2 > 0) :
  (d1 = 10 ∧ d2 = 24) →
  (d1 * d2) / (4 * Real.sqrt (d1^2 + d2^2)) = 60 / 13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_rhombus_l67_6766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slower_speed_calculation_l67_6727

/-- Proves that given a person walks 40 km at 15 km/hr, if they had walked at a slower speed for the same time, they would have covered 20 km less distance, then the slower speed is 7.5 km/hr. -/
theorem slower_speed_calculation (actual_speed : ℝ) (actual_distance : ℝ) (distance_difference : ℝ) (slower_speed : ℝ) : 
  actual_speed = 15 →
  actual_distance = 40 →
  distance_difference = 20 →
  (actual_distance / actual_speed) * slower_speed = actual_distance - distance_difference →
  slower_speed = 7.5 := by
  intro h1 h2 h3 h4
  -- The proof steps would go here
  sorry

#check slower_speed_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slower_speed_calculation_l67_6727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_root_l67_6765

theorem equation_root : ∃ x : ℝ, (49 : ℝ)^x - 6 * (7 : ℝ)^x - 7 = 0 ∧ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_root_l67_6765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_complement_N_l67_6789

open Set

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set M
def M : Set ℝ := {x | x^2 > 9}

-- Define set N
def N : Set ℝ := {x | -1 < x ∧ x < 4}

-- Theorem statement
theorem intersection_M_complement_N :
  M ∩ (U \ N) = {x : ℝ | x < -3 ∨ x ≥ 4} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_complement_N_l67_6789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_course_students_count_l67_6720

theorem course_students_count (total : ℕ) : total = 135 :=
  by
  -- Define the conditions
  have grade_a : (2 : ℚ) / 9 * total = (total - 15 : ℚ) * (2 : ℚ) / 9 := by sorry
  have grade_b : (1 : ℚ) / 3 * total = (total - 15 : ℚ) * (1 : ℚ) / 3 := by sorry
  have grade_c : (2 : ℚ) / 9 * total = (total - 15 : ℚ) * (2 : ℚ) / 9 := by sorry
  have grade_d : (1 : ℚ) / 9 * total = (total - 15 : ℚ) * (1 : ℚ) / 9 := by sorry
  have grade_e : 15 = total - ((2 : ℚ) / 9 * total + (1 : ℚ) / 3 * total + (2 : ℚ) / 9 * total + (1 : ℚ) / 9 * total) := by sorry

  -- The proof
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_course_students_count_l67_6720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_division_theorem_l67_6704

/-- Represents a regular tetrahedron -/
structure RegularTetrahedron where
  -- Add necessary fields

/-- Represents a plane in 3D space -/
structure Plane where
  -- Add necessary fields

/-- Represents a point of division on an edge of the tetrahedron -/
structure DivisionPoint where
  -- Add necessary fields

/-- Given a regular tetrahedron, returns the list of all division points -/
def getDivisionPoints (t : RegularTetrahedron) : List DivisionPoint :=
  sorry

/-- Given a division point and a tetrahedron, returns the two planes parallel to faces not containing the point -/
def getPlanesForDivisionPoint (p : DivisionPoint) (t : RegularTetrahedron) : List Plane :=
  sorry

/-- Given a list of planes and a tetrahedron, returns the number of regions the tetrahedron is divided into -/
def countRegions (planes : List Plane) (t : RegularTetrahedron) : Nat :=
  sorry

theorem tetrahedron_division_theorem (t : RegularTetrahedron) :
  let divisionPoints := getDivisionPoints t
  let planes := divisionPoints.bind (fun p => getPlanesForDivisionPoint p t)
  countRegions planes t = 27 := by
  sorry

#eval "Theorem statement compiled successfully."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_division_theorem_l67_6704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_cube_volume_l67_6700

/-- A pyramid with a square base and equilateral triangular faces -/
structure Pyramid where
  baseSide : ℝ
  baseSide_pos : 0 < baseSide

/-- A cube inscribed in a pyramid -/
structure InscribedCube (p : Pyramid) where
  sideLength : ℝ
  sideLength_pos : 0 < sideLength
  fits_in_pyramid : sideLength ≤ p.baseSide
  touches_lateral_faces : True  -- This is a simplification, as we can't easily express this geometrically

/-- The volume of the inscribed cube -/
def cubeVolume (p : Pyramid) (c : InscribedCube p) : ℝ :=
  c.sideLength ^ 3

/-- The theorem stating the volume of the inscribed cube -/
theorem inscribed_cube_volume (p : Pyramid) (c : InscribedCube p)
    (h1 : p.baseSide = 2)
    (h2 : c.sideLength = Real.sqrt 6 / 2) :
    cubeVolume p c = 3 * Real.sqrt 6 / 8 := by
  sorry

#check inscribed_cube_volume

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_cube_volume_l67_6700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_tangent_line_equation_correct_l67_6743

/-- The function f(x) = e^x + sin x + 1 -/
noncomputable def f (x : ℝ) : ℝ := Real.exp x + Real.sin x + 1

/-- The tangent line equation at x=0 for the curve defined by f -/
def tangent_line (x : ℝ) : ℝ := 2 * x + 2

/-- Theorem stating that the tangent line equation at x=0 for f(x) = e^x + sin x + 1 is y = 2x + 2 -/
theorem tangent_line_at_zero :
  HasDerivAt f 2 0 := by sorry

/-- Theorem verifying that the tangent line equation is correct -/
theorem tangent_line_equation_correct :
  ∀ x : ℝ, f 0 + (tangent_line x - tangent_line 0) = tangent_line x := by
  intro x
  simp [f, tangent_line]
  ring

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_tangent_line_equation_correct_l67_6743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unreduced_fraction_count_l67_6774

theorem unreduced_fraction_count : 
  (Finset.filter (fun m : ℕ => Nat.gcd (m^2 + 7) (m + 4) > 1) 
    (Finset.range 1996)).card = 86 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unreduced_fraction_count_l67_6774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_tangent_line_at_one_l67_6764

-- Define the function
noncomputable def f (x : ℝ) : ℝ := x * Real.log x

-- Theorem for the derivative
theorem derivative_f : 
  deriv f = fun x => Real.log x + 1 := by sorry

-- Theorem for the tangent line
theorem tangent_line_at_one :
  (fun x => x - 1) = fun x => f 1 + deriv f 1 * (x - 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_tangent_line_at_one_l67_6764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reassembly_not_always_same_l67_6792

-- Define a point in 3D space
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a face as a set of points
def Face := Set Point

-- Define an edge as a pair of points
def Edge := Point × Point

-- Define a polyhedron as a structure with faces and edges
structure Polyhedron where
  faces : Set Face
  edges : Set Edge

-- Define a function to disassemble a polyhedron into its faces
def disassemble (p : Polyhedron) : Set Face := p.faces

-- Define a function to reassemble faces into a polyhedron
noncomputable def reassemble (faces : Set Face) : Polyhedron :=
  { faces := faces
  , edges := sorry }

-- Theorem stating that reassembly may not always produce the original polyhedron
theorem reassembly_not_always_same : 
  ∃ (p : Polyhedron), reassemble (disassemble p) ≠ p := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_reassembly_not_always_same_l67_6792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_terminal_side_315_neg45_l67_6786

/-- An angle in degrees -/
structure Angle where
  value : ℝ

/-- Two angles have the same terminal side if their difference is a multiple of 360° -/
def SameTerminalSide (α β : Angle) : Prop :=
  ∃ k : ℤ, α.value = k * 360 + β.value

/-- The theorem stating that -45° has the same terminal side as 315° -/
theorem same_terminal_side_315_neg45 :
  SameTerminalSide ⟨315⟩ ⟨-45⟩ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_terminal_side_315_neg45_l67_6786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_phi_l67_6771

-- Define the function
noncomputable def f (x φ : ℝ) : ℝ := 2 * Real.sin (2 * x + φ)

-- Theorem statement
theorem symmetry_implies_phi (φ : ℝ) 
  (h1 : -Real.pi / 2 < φ) (h2 : φ < Real.pi / 2)
  (h3 : ∀ x, f (Real.pi / 3 - x) φ = f (Real.pi / 3 + x) φ) :
  φ = Real.pi / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_phi_l67_6771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_at_pi_third_l67_6779

noncomputable def f (x : ℝ) : ℝ := Real.cos x + Real.sqrt 3 * Real.sin x

theorem f_derivative_at_pi_third : 
  deriv f (π / 3) = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_at_pi_third_l67_6779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extrema_sum_range_l67_6762

open Real

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 - m * x * log x

-- State the theorem
theorem extrema_sum_range (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ 
   (∀ x : ℝ, x > 0 → f m x ≤ f m x₁) ∧
   (∀ x : ℝ, x > 0 → f m x ≤ f m x₂) ∧
   x₁ ≠ x₂) →
  f m x₁ + f m x₂ < -3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extrema_sum_range_l67_6762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_arithmetic_sequence_l67_6784

def IsArithmeticSequence (a b c d : ℝ) : Prop :=
  ∃ r : ℝ, b - a = r ∧ c - b = r ∧ d - c = r

theorem roots_arithmetic_sequence (m n : ℝ) : 
  (∃ a b c d : ℝ, 
    (a = 1/4) ∧
    (IsArithmeticSequence a b c d) ∧
    ({x : ℝ | (x^2 - 2*x + m) * (x^2 - 2*x + n) = 0} = {a, b, c, d})) →
  |m - n| = 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_arithmetic_sequence_l67_6784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_min_value_l67_6740

noncomputable def g (x : ℝ) : ℝ := (3 * x^2 + 9 * x + 18) / (4 * (1 + x))

theorem g_min_value :
  (∀ x : ℝ, x ≥ 0 → g x ≥ 3 * Real.sqrt 11.25 / 2) ∧
  (∃ x : ℝ, x ≥ 0 ∧ g x = 3 * Real.sqrt 11.25 / 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_min_value_l67_6740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_indefinite_integral_proof_l67_6748

-- Define the function f(x) = (5x + 6)cos(2x)
noncomputable def f (x : ℝ) : ℝ := (5 * x + 6) * Real.cos (2 * x)

-- Define the function F(x) = (1/2)(5x + 6)sin(2x) + (5/4)cos(2x)
noncomputable def F (x : ℝ) : ℝ := (1/2) * (5 * x + 6) * Real.sin (2 * x) + (5/4) * Real.cos (2 * x)

-- Theorem statement
theorem indefinite_integral_proof : 
  ∀ x : ℝ, deriv F x = f x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_indefinite_integral_proof_l67_6748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_detour_percentage_increase_l67_6780

noncomputable def distance_to_friend : ℝ := 30
noncomputable def time_at_friend : ℝ := 0.5
noncomputable def speed : ℝ := 44
noncomputable def total_time_away : ℝ := 2

noncomputable def total_driving_time : ℝ := total_time_away - time_at_friend
noncomputable def total_distance : ℝ := speed * total_driving_time
noncomputable def detour_distance : ℝ := total_distance - distance_to_friend
noncomputable def percentage_increase : ℝ := (detour_distance - distance_to_friend) / distance_to_friend * 100

theorem detour_percentage_increase :
  percentage_increase = 20 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_detour_percentage_increase_l67_6780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_expression_l67_6717

theorem max_value_expression (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) 
  (h_sum : a + b + 2 * c = 1) :
  2 * a + Real.sqrt (2 * a * b) + (4 * a * b * c) ^ (1/3) ≤ 3 / 2 ∧
  (2 * a + Real.sqrt (2 * a * b) + (4 * a * b * c) ^ (1/3) = 3 / 2 ↔ 
    a = 1 / 4 ∧ b = 1 / 4 ∧ c = 1 / 8) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_expression_l67_6717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_l67_6735

/-- The area of a rectangle with its longer side on y = 6 and endpoints on y = x^2 + 4x + 3,
    where the shorter side is 3 units longer than the longer side. -/
theorem rectangle_area :
  ∃ (area longer_side shorter_side : ℝ),
    (∃ x, x^2 + 4*x + 3 = 6) ∧
    (shorter_side = longer_side + 3) ∧
    (area = longer_side * shorter_side) ∧
    area = 28 + 12 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_l67_6735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_side_approx_l67_6707

/-- The length of the longest side of a rectangle with perimeter 240 feet and area equal to twelve times its perimeter --/
noncomputable def longest_side : ℝ :=
  let perimeter := 240
  let area := 12 * perimeter
  let width := (perimeter + Real.sqrt (perimeter^2 - 16 * area)) / 4
  let length := perimeter / 2 - width
  max width length

/-- Theorem stating that the longest side of the rectangle is approximately 86.833 feet --/
theorem longest_side_approx :
  |longest_side - 86.833| < 0.001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_side_approx_l67_6707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_terminal_side_quadrant_l67_6787

-- Define the Quadrant type
inductive Quadrant
  | I
  | II
  | III
  | IV

-- Define the function to determine the quadrant
noncomputable def determine_quadrant (θ : Real) : Quadrant :=
  if Real.sin θ < 0 ∧ Real.cos θ > 0 then
    Quadrant.IV
  else
    sorry -- Other cases are not needed for this problem

-- Theorem statement
theorem terminal_side_quadrant (θ : Real) :
  Real.sin θ < 0 → Real.cos θ > 0 → determine_quadrant θ = Quadrant.IV :=
by
  sorry -- Proof is not required

end NUMINAMATH_CALUDE_ERRORFEEDBACK_terminal_side_quadrant_l67_6787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_functions_unit_circle_l67_6775

open Real

theorem trigonometric_functions_unit_circle (α : ℝ) :
  (∃ P : ℝ × ℝ, P = (-3/5, 4/5) ∧ P.1^2 + P.2^2 = 1 ∧ 
   Real.sin α = P.2 ∧ Real.cos α = P.1 ∧ Real.tan α = P.2 / P.1) →
  Real.sin α = 4/5 ∧ Real.cos α = -3/5 ∧ Real.tan α = -4/3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_functions_unit_circle_l67_6775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_second_quadrant_l67_6755

/-- The complex number z -/
def z (a b : ℝ) : ℂ := a + b * Complex.I

/-- The condition given in the problem -/
def condition (a b : ℝ) : Prop := (1 + a * Complex.I) * Complex.I = 1 + b * Complex.I

/-- A point is in the second quadrant if its real part is negative and imaginary part is positive -/
def in_second_quadrant (z : ℂ) : Prop := z.re < 0 ∧ z.im > 0

/-- The main theorem to be proved -/
theorem z_in_second_quadrant {a b : ℝ} (h : condition a b) : in_second_quadrant (z a b) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_second_quadrant_l67_6755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l67_6747

noncomputable def f (x : ℝ) : ℝ := (x + 1) * Real.log (x + 1)

theorem f_properties :
  (∃ (x : ℝ), ∀ (y : ℝ), f y ≥ f x) ∧
  f (Real.exp (-1) - 1) = -Real.exp (-1) ∧
  (∀ (a : ℝ), (∀ (x : ℝ), x ≥ 0 → f x ≥ a * x) ↔ a ≤ 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l67_6747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_and_m_range_l67_6785

open Real

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x / (exp x) - m * x

theorem f_monotonicity_and_m_range :
  -- Part 1: Monotonicity when m = 0
  (∀ x y, x < y ∧ y < 1 → f 0 x < f 0 y) ∧
  (∀ x y, 1 < x ∧ x < y → f 0 x > f 0 y) ∧
  -- Part 2: Condition for (f(b) - f(a))/(b - a) > 1
  (∀ m, (∀ a b, 0 < a ∧ a < b → (f m b - f m a) / (b - a) > 1) ↔ m ≤ -1 - exp (-2)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_and_m_range_l67_6785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l67_6716

-- Define the function f as noncomputable
noncomputable def f (φ : Real) (x : Real) : Real := Real.sqrt 5 * Real.sin (2 * x + φ)

-- State the theorem
theorem function_properties (φ : Real) 
  (h1 : 0 < φ ∧ φ < Real.pi)
  (h2 : ∀ x, f φ (Real.pi/3 - x) = f φ (Real.pi/3 + x)) :
  -- Part 1: Value of φ
  φ = 5 * Real.pi / 6 ∧
  -- Part 2: Maximum and minimum values
  (∀ x ∈ Set.Icc (-Real.pi/12) (Real.pi/2),
    f φ x ≤ Real.sqrt 15 / 2 ∧
    f φ x ≥ -Real.sqrt 5) ∧
  f φ (-Real.pi/12) = Real.sqrt 15 / 2 ∧
  f φ (Real.pi/3) = -Real.sqrt 5 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l67_6716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_factorial_ten_greater_than_factorial_nine_l67_6754

theorem divisors_of_factorial_ten_greater_than_factorial_nine : 
  (Finset.filter (λ d : ℕ => d > Nat.factorial 9 ∧ Nat.factorial 10 % d = 0) (Finset.range (Nat.factorial 10 + 1))).card = 9 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_factorial_ten_greater_than_factorial_nine_l67_6754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_geometric_sides_ratio_range_l67_6734

/-- The range of possible values for the common ratio of a geometric sequence
    formed by the sides of a triangle. -/
theorem triangle_geometric_sides_ratio_range :
  ∀ q : ℝ, q > 0 →
  (∃ a : ℝ, a > 0 ∧ 
    (a + q*a > q^2*a) ∧ 
    (a + q^2*a > q*a) ∧ 
    (q*a + q^2*a > a)) ↔ 
  ((Real.sqrt 5 - 1) / 2 < q ∧ q < (1 + Real.sqrt 5) / 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_geometric_sides_ratio_range_l67_6734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_sphere_radius_l67_6724

/-- The volume of a sphere given its radius -/
noncomputable def sphereVolume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r ^ 3

/-- The radius of the original sphere -/
def R : ℝ := 7

/-- The number of smaller spheres -/
def n : ℕ := 343

/-- The radius of each smaller sphere -/
def r : ℝ := 1

theorem original_sphere_radius :
  sphereVolume R = n * sphereVolume r := by
  -- Unfold definitions
  unfold sphereVolume R n r
  -- Simplify
  simp [Real.pi]
  -- The proof is completed with sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_sphere_radius_l67_6724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_moskvich_halfway_theorem_l67_6752

/-- Represents a car with a constant speed --/
structure Car where
  speed : ℝ

/-- Represents the state of the cars at a given time --/
structure CarState where
  time : ℝ
  moskvichPosition : ℝ
  zhiguliPosition : ℝ

/-- The initial state of the cars --/
def initialState (distance : ℝ) : CarState :=
  { time := 0, moskvichPosition := 0, zhiguliPosition := distance }

/-- Calculates the state of the cars after a given time --/
def stateAfterTime (moskvich : Car) (zhiguli : Car) (distance : ℝ) (t : ℝ) : CarState :=
  { time := t
  , moskvichPosition := t * moskvich.speed
  , zhiguliPosition := distance - t * zhiguli.speed }

/-- Theorem: If Moskvich is halfway to Zhiguli after 1 hour, it will be halfway from Zhiguli to B after 2 hours --/
theorem moskvich_halfway_theorem (moskvich : Car) (zhiguli : Car) (distance : ℝ) :
  let state1 := stateAfterTime moskvich zhiguli distance 1
  let state2 := stateAfterTime moskvich zhiguli distance 2
  state1.moskvichPosition = (state1.moskvichPosition + state1.zhiguliPosition) / 2 →
  moskvich.speed < 2 * zhiguli.speed →
  zhiguli.speed < 2 * moskvich.speed →
  state2.moskvichPosition = (state2.zhiguliPosition + distance) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_moskvich_halfway_theorem_l67_6752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_perimeter_theorem_l67_6742

/-- The perimeter of a quadrilateral formed by cutting a square with the line y = x/3, divided by the side length parameter. -/
noncomputable def quadrilateral_perimeter_ratio (b : ℝ) : ℝ :=
  4 * (5 + Real.sqrt 10) / 3

/-- The perimeter of the quadrilateral formed by cutting the square, divided by b, is equal to 4(5+√10)/3. -/
theorem quadrilateral_perimeter_theorem (b : ℝ) (h : b > 0) :
  quadrilateral_perimeter_ratio b = 4 * (5 + Real.sqrt 10) / 3 := by
  -- Unfold the definition of quadrilateral_perimeter_ratio
  unfold quadrilateral_perimeter_ratio
  -- The left-hand side is now exactly equal to the right-hand side
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_perimeter_theorem_l67_6742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_on_imaginary_axis_z_on_imaginary_axis_alt_l67_6739

-- Define the complex number z
noncomputable def z (a : ℝ) : ℂ := (-2 + a * Complex.I) / (1 + Complex.I)

-- Theorem statement
theorem z_on_imaginary_axis : 
  (∃ (a : ℝ), (z a).re = 0) ↔ (∃ (a : ℝ), a = 2) := by
  sorry

-- Alternative formulation using Set.range
theorem z_on_imaginary_axis_alt : 
  (∃ (a : ℝ), z a ∈ Set.range (λ y : ℝ => Complex.I * y)) ↔ (∃ (a : ℝ), a = 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_on_imaginary_axis_z_on_imaginary_axis_alt_l67_6739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_finite_l67_6715

open Set
open Function
open Filter
open Topology

/-- A function f: ℝ → ℝ that is positive and approaches 0 as x approaches infinity -/
structure PositiveDecreasingFunction where
  f : ℝ → ℝ
  positive : ∀ x, f x > 0
  limit_zero : Tendsto f atTop (𝓝 0)

/-- The set of positive integer triples (m, n, p) satisfying f(m) + f(n) + f(p) = 1 -/
def SolutionSet (f : ℝ → ℝ) : Set (ℕ × ℕ × ℕ) :=
  {t | f (t.1 : ℝ) + f (t.2.1 : ℝ) + f (t.2.2 : ℝ) = 1}

/-- Theorem: The solution set is finite for any positive decreasing function -/
theorem solution_set_finite (pdf : PositiveDecreasingFunction) :
    (SolutionSet pdf.f).Finite := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_finite_l67_6715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chocolate_sales_week2_l67_6705

/-- Represents the number of chocolates sold in each week --/
structure ChocolateSales where
  week1 : ℕ
  week2 : ℕ
  week3 : ℕ
  week4 : ℕ
  week5 : ℕ

/-- Calculates the mean of chocolate sales over 5 weeks --/
def mean (sales : ChocolateSales) : ℚ :=
  (sales.week1 + sales.week2 + sales.week3 + sales.week4 + sales.week5) / 5

theorem chocolate_sales_week2 (sales : ChocolateSales) :
  sales.week1 = 75 →
  sales.week3 = 75 →
  sales.week4 = 70 →
  sales.week5 = 68 →
  mean sales = 71 →
  sales.week2 = 67 := by
  sorry

-- Remove the #eval line as it's not necessary for the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chocolate_sales_week2_l67_6705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_loss_percentage_calculation_l67_6709

/-- Represents the selling price of a product in terms of units per currency unit -/
structure SellingPrice where
  units_per_currency : ℚ
  deriving Repr

/-- Calculates the price per unit given a SellingPrice -/
noncomputable def price_per_unit (sp : SellingPrice) : ℚ := 1 / sp.units_per_currency

/-- Theorem: If selling at 12 units per currency results in a loss,
    and selling at 7.5 units per currency results in a 44% gain,
    then the original loss percentage is 10%. -/
theorem loss_percentage_calculation
  (sp_loss : SellingPrice)
  (sp_gain : SellingPrice)
  (h_loss : sp_loss.units_per_currency = 12)
  (h_gain : sp_gain.units_per_currency = 15/2)
  (h_gain_percentage : price_per_unit sp_gain = 144/100 * price_per_unit sp_loss) :
  (price_per_unit sp_loss - price_per_unit sp_gain) / price_per_unit sp_gain * 100 = 10 := by
  sorry

#eval SellingPrice.mk 12
#eval SellingPrice.mk (15/2)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_loss_percentage_calculation_l67_6709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l67_6728

-- Define the equation of the region
def region_equation (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 8*y = -9

-- Define the area of the region
noncomputable def region_area : ℝ := 16 * Real.pi

-- Theorem statement
theorem area_of_region :
  ∃ (center_x center_y radius : ℝ),
    (∀ x y : ℝ, region_equation x y ↔ (x - center_x)^2 + (y - center_y)^2 = radius^2) ∧
    region_area = Real.pi * radius^2 := by
  -- Proof goes here
  sorry

-- Additional lemma to show that the region is indeed a circle
lemma region_is_circle :
  ∃ (center_x center_y radius : ℝ),
    ∀ x y : ℝ, region_equation x y ↔ (x - center_x)^2 + (y - center_y)^2 = radius^2 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l67_6728
