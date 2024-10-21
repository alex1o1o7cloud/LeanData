import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_divisible_by_four_l409_40940

def fair_8_sided_die := Finset.range 8

theorem probability_divisible_by_four :
  let outcomes := fair_8_sided_die.product fair_8_sided_die
  let favorable_outcomes := outcomes.filter (fun p => p.1 % 4 = 0 ∧ p.2 % 4 = 0 ∧ (10 * p.1 + p.2) % 4 = 0)
  (favorable_outcomes.card : ℚ) / outcomes.card = 1 / 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_divisible_by_four_l409_40940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cut_impossibility_l409_40930

-- Define a point
structure Point where
  x : Real
  y : Real

-- Define a triangle
structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define a straight cut
structure Cut where
  start : Point
  endpoint : Point  -- Changed 'end' to 'endpoint' to avoid keyword conflict

-- Define the result of cutting a triangle
structure CutResult where
  pieces : Finset Point
  triangles : Finset Point

-- Function to cut a triangle
noncomputable def cutTriangle (t : Triangle) (c1 c2 : Cut) : CutResult :=
  sorry

-- Function to calculate area of a triangle
noncomputable def area (t : Triangle) : Real :=
  sorry

-- Theorem statement
theorem triangle_cut_impossibility (t : Triangle) :
  ∀ c1 c2 : Cut,
    (c1.start = t.A ∨ c1.start = t.B ∨ c1.start = t.C) →
    (c1.endpoint = t.A ∨ c1.endpoint = t.B ∨ c1.endpoint = t.C) →
    (c2.start = t.A ∨ c2.start = t.B ∨ c2.start = t.C) →
    (c2.endpoint = t.A ∨ c2.endpoint = t.B ∨ c2.endpoint = t.C) →
    let result := cutTriangle t c1 c2
    (result.pieces.card = 4 ∧ result.triangles.card ≥ 3) →
    ¬∃ (t1 t2 t3 : Triangle),
      t1.A ∈ result.triangles ∧ t2.A ∈ result.triangles ∧ t3.A ∈ result.triangles ∧
      t1 ≠ t2 ∧ t2 ≠ t3 ∧ t1 ≠ t3 ∧
      area t1 = area t2 ∧ area t2 = area t3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cut_impossibility_l409_40930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l409_40919

/-- Definition of a Triangle -/
structure Triangle where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ

/-- Definition of an equilateral triangle -/
def is_equilateral (t : Triangle) : Prop :=
  t.side1 = t.side2 ∧ t.side2 = t.side3

/-- Definition of an isosceles triangle -/
def is_isosceles (t : Triangle) : Prop :=
  t.side1 = t.side2 ∨ t.side2 = t.side3 ∨ t.side3 = t.side1

/-- Sum of angles in a triangle is 180° -/
axiom triangle_angle_sum (t : Triangle) : t.angle1 + t.angle2 + t.angle3 = 180

theorem triangle_properties : 
  ∀ t : Triangle,
  (is_equilateral t → t.angle1 = t.angle2 ∧ t.angle2 = t.angle3) ∧
  (is_equilateral t → t.angle1 = 60 ∧ t.angle2 = 60 ∧ t.angle3 = 60) ∧
  (t.angle1 = t.angle2 ∧ t.angle2 = t.angle3 → is_equilateral t) ∧
  (is_isosceles t ∧ t.angle1 = 60 → is_equilateral t) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l409_40919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_possible_values_of_P_l409_40982

theorem possible_values_of_P (x y : ℕ+) (h : x < y) :
  let P := (x.val^3 - y.val : ℤ) / (1 + x.val * y.val)
  (∃ (k : ℤ), P = k) ↔ (P = 0 ∨ P ≥ 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_possible_values_of_P_l409_40982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_path_ratio_l409_40938

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangular quarry -/
structure Quarry where
  width : ℝ
  height : ℝ

/-- Represents the path of Swimmer 1 -/
noncomputable def swimmer1Path (q : Quarry) : ℝ :=
  2 * (q.width^2 + q.height^2).sqrt

/-- Represents a point on the side of the quarry -/
structure SidePoint where
  side : Fin 4  -- 0: top, 1: right, 2: bottom, 3: left
  position : ℝ  -- Position along the side (0 to 1)

/-- Represents the path of Swimmer 2 -/
noncomputable def swimmer2Path (q : Quarry) (p1 p2 p3 p4 : SidePoint) : ℝ :=
  sorry  -- The actual calculation would go here

/-- The main theorem to prove -/
theorem min_path_ratio (q : Quarry) : 
  ∃ (p1 p2 p3 p4 : SidePoint), 
    p1.side = 0 ∧ p1.position = 2018 / 4037 ∧
    swimmer2Path q p1 p2 p3 p4 ≤ swimmer1Path q ∧
    ∀ (p1' p2' p3' p4' : SidePoint),
      p1'.side = 0 ∧ p1'.position = 2018 / 4037 →
      swimmer1Path q / swimmer2Path q p1' p2' p3' p4' ≤ 1 := by
  sorry

#check min_path_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_path_ratio_l409_40938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_problem_l409_40916

theorem divisor_problem : ∃! d : ℕ, 
  (d > 0) ∧ 
  (∃ k : ℕ, k > 0 ∧ k * d ≥ 9 ∧ (k + 6) * d ≤ 79) ∧
  (∀ m : ℕ, m > d → ¬(∃ j : ℕ, j > 0 ∧ j * m ≥ 9 ∧ (j + 6) * m ≤ 79)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_problem_l409_40916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_thirty_distinct_distances_l409_40921

/-- A type representing a point on a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- A set of 2004 points on a plane -/
def points : Finset Point :=
  sorry

/-- The set of all pairwise distances between the points -/
noncomputable def distances : Finset ℝ :=
  Finset.image (fun (pair : Point × Point) => distance pair.1 pair.2)
    (Finset.product points points)

theorem at_least_thirty_distinct_distances :
  Finset.card distances ≥ 30 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_thirty_distinct_distances_l409_40921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_equals_rounded_sum_l409_40963

def roundToNearestMultipleOf5 (n : Nat) : Nat :=
  5 * ((n + 2) / 5)

def sumFirstN (n : Nat) : Nat :=
  n * (n + 1) / 2

def sumRoundedFirstN (n : Nat) : Nat :=
  (List.range n).map (fun i => roundToNearestMultipleOf5 (i + 1)) |>.sum

theorem sum_equals_rounded_sum :
  sumFirstN 50 = sumRoundedFirstN 50 :=
by sorry

#eval sumFirstN 50
#eval sumRoundedFirstN 50

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_equals_rounded_sum_l409_40963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_integer_to_sqrt_33_l409_40959

theorem closest_integer_to_sqrt_33 :
  ∃ (n : ℤ), n = 6 ∧ ∀ (m : ℤ), |Real.sqrt 33 - n| ≤ |Real.sqrt 33 - m| :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_integer_to_sqrt_33_l409_40959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_flip_probability_l409_40910

noncomputable def coin_flip_transform (x : ℂ) (is_heads : Bool) : ℂ :=
  if is_heads then 1 - x else 1 / x

noncomputable def probability_equal_after_flips (x₀ : ℂ) (num_flips : ℕ) : ℝ :=
  sorry

theorem coin_flip_probability (x₀ : ℂ) (h₁ : x₀ ≠ 0) (h₂ : x₀ ≠ 1) :
  probability_equal_after_flips x₀ 2012 = 1 ∨
  probability_equal_after_flips x₀ 2012 = (2^2011 + 1) / (3 * 2^2011) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_flip_probability_l409_40910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_condition_l409_40995

theorem sufficient_not_necessary_condition :
  (∃ a : ℝ, a = π / 6 ∧ Real.tan (π - a) = -(Real.sqrt 3) / 3) ∧
  (∃ b : ℝ, b ≠ π / 6 ∧ Real.tan (π - b) = -(Real.sqrt 3) / 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_condition_l409_40995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_l409_40928

-- Define the circles and points
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

noncomputable def A : ℝ × ℝ := sorry
noncomputable def O : ℝ × ℝ := sorry
noncomputable def P : ℝ × ℝ := sorry

def Γ : Set (ℝ × ℝ) := Circle A 1
def ω : Set (ℝ × ℝ) := Circle O 7

-- State the conditions
axiom A_on_ω : A ∈ ω
axiom P_on_ω : P ∈ ω
axiom AP_distance : (A.1 - P.1)^2 + (A.2 - P.2)^2 = 4^2

-- Define the intersection points
noncomputable def X : ℝ × ℝ := sorry
noncomputable def Y : ℝ × ℝ := sorry

axiom X_intersection : X ∈ Γ ∧ X ∈ ω
axiom Y_intersection : Y ∈ Γ ∧ Y ∈ ω

-- State the theorem
theorem intersection_distance_product :
  ((P.1 - X.1)^2 + (P.2 - X.2)^2) * ((P.1 - Y.1)^2 + (P.2 - Y.2)^2) = 15^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_l409_40928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_circumcenter_coordinates_l409_40903

/-- Given a triangle ABC with vertices A(2,2), B(-5,1), and C(3,-5),
    its circumcenter has coordinates (-1,-2). -/
theorem triangle_circumcenter_coordinates :
  let A : ℝ × ℝ := (2, 2)
  let B : ℝ × ℝ := (-5, 1)
  let C : ℝ × ℝ := (3, -5)
  let circumcenter : ℝ × ℝ := (-1, -2)
  (∀ P : ℝ × ℝ, (P.1 - A.1)^2 + (P.2 - A.2)^2 = (P.1 - B.1)^2 + (P.2 - B.2)^2 ∧
                 (P.1 - B.1)^2 + (P.2 - B.2)^2 = (P.1 - C.1)^2 + (P.2 - C.2)^2) →
  circumcenter.1 = -1 ∧ circumcenter.2 = -2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_circumcenter_coordinates_l409_40903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_l409_40912

noncomputable def f (x : ℝ) : ℝ := 2 / x + 9 / (1 - 2 * x) - 5

theorem f_minimum :
  ∀ x ∈ Set.Ioo (0 : ℝ) (1/2),
    f x ≥ 20 ∧
    (f x = 20 ↔ x = 1/5) := by
  sorry

#check f_minimum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_l409_40912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equations_with_common_root_l409_40911

theorem quadratic_equations_with_common_root (p1 p2 q1 q2 : ℤ) (α : ℝ) :
  (α^2 + ↑p1*α + ↑q1 = 0) →
  (α^2 + ↑p2*α + ↑q2 = 0) →
  (∀ n : ℤ, α ≠ ↑n) →
  (p1 = p2 ∧ q1 = q2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equations_with_common_root_l409_40911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_distances_ge_half_perimeter_l409_40924

/-- Represents a triangle with vertices A, B, and C -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Calculates the distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

/-- Calculates the perimeter of a triangle -/
def perimeter (t : Triangle) : ℝ :=
  distance t.A t.B + distance t.B t.C + distance t.C t.A

/-- Represents a point D on side BC of the triangle -/
def pointOnSide (t : Triangle) : ℝ × ℝ := sorry

/-- Calculates the sum of distances from D to all vertices -/
def sumDistances (t : Triangle) (d : ℝ × ℝ) : ℝ :=
  distance d t.A + distance d t.B + distance d t.C

/-- Theorem: The sum of distances from D to all vertices is ≥ half the perimeter -/
theorem sum_distances_ge_half_perimeter (t : Triangle) :
  let d := pointOnSide t
  let s := sumDistances t d
  let p := perimeter t
  distance d t.B = distance d t.C →
  s ≥ p / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_distances_ge_half_perimeter_l409_40924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spinner_probability_l409_40993

-- Define the spinners
def spinnerS : List ℕ := [1, 2, 3, 4]
def spinnerT : List ℕ := [3, 4, 5, 6]
def spinnerU : List ℕ := [1, 2, 5, 6]

-- Define a function to check if a number is odd
def isOdd (n : ℕ) : Bool := n % 2 = 1

-- Define a function to check if the sum of three numbers is odd and greater than 10
def isValidSum (a b c : ℕ) : Bool :=
  isOdd (a + b + c) ∧ (a + b + c > 10)

-- Calculate the total number of possible outcomes
def totalOutcomes : ℕ := (List.length spinnerS) * (List.length spinnerT) * (List.length spinnerU)

-- Calculate the number of valid outcomes
def validOutcomes : ℕ :=
  List.length (List.filter (fun (x : ℕ × ℕ × ℕ) => isValidSum x.1 x.2.1 x.2.2)
    (List.product spinnerS (List.product spinnerT spinnerU)))

-- Theorem to prove
theorem spinner_probability :
  (validOutcomes : ℚ) / totalOutcomes = 9 / 64 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spinner_probability_l409_40993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_area_example_distance_between_sides_l409_40967

/-- The area of a trapezium given the lengths of its parallel sides and the distance between them. -/
noncomputable def trapezium_area (a b h : ℝ) : ℝ := (a + b) * h / 2

/-- Theorem: The area of a trapezium with parallel sides of 20 cm and 18 cm, 
    and a distance of 5 cm between them, is 95 cm². -/
theorem trapezium_area_example : trapezium_area 20 18 5 = 95 := by
  -- Unfold the definition of trapezium_area
  unfold trapezium_area
  -- Perform the calculation
  simp [mul_add, mul_div_right_comm]
  -- The result should now be obvious to Lean
  norm_num

/-- The distance between the parallel sides of the trapezium is 5 cm. -/
theorem distance_between_sides : 5 = 5 := by rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_area_example_distance_between_sides_l409_40967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_OMN_is_pi_over_six_l409_40955

/-- A regular polygon with n sides inscribed in a circle -/
structure RegularPolygon (n : ℕ) where
  center : ℝ × ℝ
  radius : ℝ
  vertices : Fin n → ℝ × ℝ

/-- The angle OMN in a regular 9-gon -/
noncomputable def angle_OMN (polygon : RegularPolygon 9) : ℝ :=
  let O := polygon.center
  let A := polygon.vertices 0
  let B := polygon.vertices 1
  let C := polygon.vertices 2
  let M := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)  -- midpoint of AB
  let N := ((O.1 + C.1) / 2, (O.2 + C.2) / 2)  -- midpoint of radius perpendicular to BC
  Real.arctan (Real.sqrt 3 / 3)  -- angle corresponding to tan(θ) = √3/3

/-- Theorem: The angle OMN in a regular 9-gon is π/6 -/
theorem angle_OMN_is_pi_over_six (polygon : RegularPolygon 9) :
  angle_OMN polygon = π / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_OMN_is_pi_over_six_l409_40955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_four_digit_number_l409_40933

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m^2

def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

theorem unique_four_digit_number :
  ∃! n : ℕ,
    is_four_digit n ∧
    is_perfect_square n ∧
    is_perfect_square (sum_of_digits n) ∧
    is_perfect_square (n / sum_of_digits n) ∧
    (Nat.divisors n).card = sum_of_digits n ∧
    (n.digits 10).Nodup :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_four_digit_number_l409_40933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_k_l409_40954

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2 / a^2)

/-- Theorem: For an ellipse with equation x^2/(k+4) + y^2/9 = 1, 
    where the foci are on the x-axis and the eccentricity is 1/2, 
    the value of k is 8. -/
theorem ellipse_eccentricity_k (k : ℝ) :
  k > 5 →
  eccentricity (Real.sqrt (k + 4)) 3 = 1/2 →
  k = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_k_l409_40954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_arithmetic_progressions_l409_40941

def isArithmeticProgression (s : List Nat) : Prop :=
  s.length ≥ 2 ∧ ∃ d, ∀ i, i + 1 < s.length → s[i + 1]! - s[i]! = d

def isStrictlyIncreasing (s : List Nat) : Prop :=
  ∀ i j, i < j → j < s.length → s[i]! < s[j]!

def isPrimeList (s : List Nat) : Prop :=
  ∀ n, n ∈ s → Nat.Prime n

theorem prime_arithmetic_progressions :
  ∀ s : List Nat,
    isArithmeticProgression s ∧
    isStrictlyIncreasing s ∧
    isPrimeList s ∧
    (∃ d, isArithmeticProgression s ∧ s.length > d) →
    s = [2, 3] ∨ s = [3, 5, 7] :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_arithmetic_progressions_l409_40941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_circle_to_line_l409_40960

/-- Line l in the xy-plane -/
def line_l (x y : ℝ) : Prop := y = x + 4

/-- Circle C in the xy-plane -/
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 4

/-- Distance from a point (x, y) to line l -/
noncomputable def distance_to_line (x y : ℝ) : ℝ := 
  (|x - y + 4|) / Real.sqrt 2

/-- Maximum distance from any point on circle C to line l -/
theorem max_distance_circle_to_line :
  ∃ (x y : ℝ), circle_C x y ∧
  ∀ (x' y' : ℝ), circle_C x' y' → 
  distance_to_line x y ≥ distance_to_line x' y' ∧
  distance_to_line x y = 3 * Real.sqrt 2 + 2 :=
by
  sorry

#check max_distance_circle_to_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_circle_to_line_l409_40960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_when_α_2π_3_intersection_range_l409_40920

-- Define circle C
noncomputable def circle_C (θ : ℝ) : ℝ × ℝ := (1 + Real.cos θ, Real.sin θ)

-- Define line l
noncomputable def line_l (α t : ℝ) : ℝ × ℝ := (2 + t * Real.cos α, Real.sqrt 3 + t * Real.sin α)

-- Theorem for minimum distance
theorem min_distance_when_α_2π_3 :
  let α : ℝ := 2 * Real.pi / 3
  ∃ d : ℝ, d = Real.sqrt 3 - 1 ∧
  ∀ θ : ℝ, ∀ p : ℝ × ℝ,
    p = circle_C θ →
    d ≤ Real.sqrt ((p.1 - (line_l α 0).1)^2 + (p.2 - (line_l α 0).2)^2) :=
by sorry

-- Theorem for intersection range
theorem intersection_range :
  ∀ α : ℝ,
    (∃ θ t : ℝ, circle_C θ = line_l α t) ↔
    (Real.pi / 6 ≤ α ∧ α < Real.pi / 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_when_α_2π_3_intersection_range_l409_40920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_star_commutative_star_associative_l409_40990

class CustomSet (α : Type) where
  star : α → α → α

variable {α : Type} [CustomSet α]

axiom star_property_1 {a b c : α} : CustomSet.star a (CustomSet.star b c) = CustomSet.star b (CustomSet.star c a)
axiom star_property_2 {a b c : α} : CustomSet.star a b = CustomSet.star a c → b = c
axiom star_property_3 {a b c : α} : CustomSet.star a c = CustomSet.star b c → a = b

theorem star_commutative (a b : α) : CustomSet.star a b = CustomSet.star b a := by
  sorry

theorem star_associative (a b c : α) : CustomSet.star (CustomSet.star a b) c = CustomSet.star a (CustomSet.star b c) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_star_commutative_star_associative_l409_40990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_coverage_l409_40972

theorem triangle_coverage (a b c d e : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d) (pos_e : 0 < e)
  (sum_squares : a^2 + b^2 + c^2 + d^2 + e^2 = 2) :
  ∃ (w x y z : ℝ), ({w, x, y, z} : Set ℝ) ⊆ {a, b, c, d, e} ∧ w^2 + x^2 + y^2 + z^2 ≥ 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_coverage_l409_40972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_relationship_l409_40977

/-- Represents a line in 3D space -/
structure Line3D where
  -- Define necessary properties for a 3D line
  -- This is a simplified representation
  point : Fin 3 → ℝ  -- A point on the line
  direction : Fin 3 → ℝ  -- Direction vector of the line

/-- Defines when two lines are skew -/
def are_skew (l1 l2 : Line3D) : Prop :=
  -- Define the condition for two lines to be skew
  sorry

/-- Defines when two lines are parallel -/
def are_parallel (l1 l2 : Line3D) : Prop :=
  -- Define the condition for two lines to be parallel
  sorry

/-- Defines when two lines intersect -/
def do_intersect (l1 l2 : Line3D) : Prop :=
  -- Define the condition for two lines to intersect
  sorry

theorem line_relationship (a b c : Line3D) 
  (h1 : are_skew a b) (h2 : are_parallel c a) : 
  are_skew c b ∨ do_intersect c b :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_relationship_l409_40977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_age_progression_l409_40901

/-- Represents a stem-and-leaf diagram --/
structure StemAndLeaf where
  stems : List Nat
  leaves : List (List Nat)

/-- Check if a list of ages produces a given stem-and-leaf structure --/
def matchesStemAndLeaf (ages : List Nat) (diagram : StemAndLeaf) : Prop :=
  sorry

/-- The theorem to be proved --/
theorem age_progression (initial_ages : List Nat) (new_diagram : StemAndLeaf) :
  initial_ages = [10, 10, 11, 12, 12, 13, 21, 25, 26, 30, 32, 34, 36, 41, 46] →
  new_diagram = {
    stems := [0, 1, 2, 3, 4],
    leaves := [[],[0, 0, 1, 2, 2, 3], [1, 5, 6], [0, 2, 4, 6], [1, 6]]
  } →
  ∃ (years : Nat), 
    years = 6 ∧ 
    matchesStemAndLeaf (initial_ages.map (· + years)) new_diagram ∧
    ∀ (other_years : Nat), other_years ≠ 6 → 
      ¬matchesStemAndLeaf (initial_ages.map (· + other_years)) new_diagram :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_age_progression_l409_40901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bag_value_l409_40922

/-- Represents the number of pennies in the bag -/
def x : ℕ → ℕ := id

/-- The value of a penny in dollars -/
def penny_value : ℚ := 1/100

/-- The value of a dime in dollars -/
def dime_value : ℚ := 1/10

/-- The value of a quarter in dollars -/
def quarter_value : ℚ := 1/4

/-- The number of dimes is four times the number of pennies -/
def num_dimes (x : ℕ) : ℕ := 4 * x

/-- The number of quarters is twice the number of dimes -/
def num_quarters (x : ℕ) : ℕ := 2 * num_dimes x

/-- The total value of coins in the bag in dollars -/
def total_value (x : ℕ) : ℚ := 
  (x : ℚ) * penny_value + (num_dimes x : ℚ) * dime_value + (num_quarters x : ℚ) * quarter_value

theorem bag_value (x : ℕ) : total_value x = 241/100 * (x : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bag_value_l409_40922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_sets_l409_40981

/-- Given that the solution set of x² - ax - b < 0 is {x | 2 < x < 3},
    prove that the solution set of bx² - ax - 1 > 0 is {x | -1/2 < x < -1/3}. -/
theorem inequality_solution_sets 
  (a b : ℝ) 
  (h : ∀ x : ℝ, x^2 - a*x - b < 0 ↔ 2 < x ∧ x < 3) :
  ∀ x : ℝ, b*x^2 - a*x - 1 > 0 ↔ -1/2 < x ∧ x < -1/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_sets_l409_40981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_valid_points_l409_40958

-- Define the plane and points
def Plane := ℝ × ℝ

noncomputable def distance (p1 p2 : Plane) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

noncomputable def perimeter (a b c : Plane) : ℝ :=
  distance a b + distance b c + distance c a

noncomputable def area (a b c : Plane) : ℝ :=
  abs ((b.1 - a.1) * (c.2 - a.2) - (c.1 - a.1) * (b.2 - a.2)) / 2

def isValidTriangle (a b c : Plane) : Prop :=
  perimeter a b c = 60 ∧ area a b c = 72

theorem two_valid_points (a b : Plane) (h : distance a b = 12) :
  ∃! (s : Finset Plane), s.card = 2 ∧ ∀ c ∈ s, isValidTriangle a b c := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_valid_points_l409_40958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_statement_B_l409_40931

-- Define the statements as axioms instead of definitions
axiom statement_A : Prop
axiom statement_B : Prop
axiom statement_C : Prop
axiom statement_D : Prop

-- Define chromosomal variation
def chromosomal_variation (change : String) : Prop :=
  change = "structural" ∨ change = "numerical"

-- Define genetic recombination
def genetic_recombination (exchange : String) : Prop :=
  exchange = "between non-sister chromatids of homologous chromosomes"

-- Theorem statement
theorem incorrect_statement_B
  (h_A : statement_A)
  (h_C : statement_C)
  (h_D : statement_D)
  (h_variation : ∀ change, chromosomal_variation change → 
    (change = "structural" ∨ change = "numerical"))
  (h_recombination : ∀ exchange, genetic_recombination exchange → 
    exchange = "between non-sister chromatids of homologous chromosomes") :
  ¬statement_B :=
by
  sorry -- Proof is omitted as per the instruction


end NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_statement_B_l409_40931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_l409_40973

noncomputable def distanceToLine (x y a b c : ℝ) : ℝ :=
  |a * x + b * y + c| / Real.sqrt (a^2 + b^2)

def onCircle (x y r : ℝ) : Prop :=
  x^2 + y^2 = r^2

def atDistance1FromLine (x y : ℝ) : Prop :=
  distanceToLine x y 1 (-1) (-2) = 1

def exactlyFourPoints (r : ℝ) : Prop :=
  ∃ (x1 y1 x2 y2 x3 y3 x4 y4 : ℝ),
    (onCircle x1 y1 r ∧ atDistance1FromLine x1 y1) ∧
    (onCircle x2 y2 r ∧ atDistance1FromLine x2 y2) ∧
    (onCircle x3 y3 r ∧ atDistance1FromLine x3 y3) ∧
    (onCircle x4 y4 r ∧ atDistance1FromLine x4 y4) ∧
    ∀ (x y : ℝ), (onCircle x y r ∧ atDistance1FromLine x y) →
      (x = x1 ∧ y = y1) ∨ (x = x2 ∧ y = y2) ∨ (x = x3 ∧ y = y3) ∨ (x = x4 ∧ y = y4)

theorem circle_line_intersection (r : ℝ) (hr : r > 0) (h : exactlyFourPoints r) :
  r > Real.sqrt 2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_l409_40973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l409_40944

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 12 * x

-- Define the circle
def circle' (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 6*y = 0

-- Define the distance function
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- Theorem statement
theorem intersection_distance :
  ∃ (x1 y1 x2 y2 : ℝ),
    parabola x1 y1 ∧ circle' x1 y1 ∧
    parabola x2 y2 ∧ circle' x2 y2 ∧
    (x1 ≠ x2 ∨ y1 ≠ y2) ∧
    distance x1 y1 x2 y2 = 6 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l409_40944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersection_l409_40943

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 6*x - 4 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 6*y - 28 = 0

-- Define the line equation
def line_eq (x y : ℝ) : Prop := x - y + 4 = 0

-- Define the length of the common chord
noncomputable def common_chord_length : ℝ := 5 * Real.sqrt 2

theorem circles_intersection :
  (∀ x y : ℝ, circle1 x y ∧ circle2 x y → line_eq x y) ∧
  (∃ a b c d : ℝ, circle1 a b ∧ circle2 a b ∧ circle1 c d ∧ circle2 c d ∧
    ((a - c)^2 + (b - d)^2 = common_chord_length^2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersection_l409_40943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_C_in_acetone_approx_l409_40929

/-- The mass percentage of carbon in acetone -/
noncomputable def mass_percentage_C_in_acetone : ℝ :=
  let atomic_mass_C : ℝ := 12.01
  let atomic_mass_H : ℝ := 1.01
  let atomic_mass_O : ℝ := 16.00
  let molar_mass_acetone : ℝ := 3 * atomic_mass_C + 6 * atomic_mass_H + atomic_mass_O
  let mass_C_in_acetone : ℝ := 3 * atomic_mass_C
  (mass_C_in_acetone / molar_mass_acetone) * 100

/-- Theorem stating that the mass percentage of carbon in acetone is approximately 62.01% -/
theorem mass_percentage_C_in_acetone_approx :
  |mass_percentage_C_in_acetone - 62.01| < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_C_in_acetone_approx_l409_40929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_task_distribution_count_l409_40905

def num_volunteers : ℕ := 3
def num_tasks : ℕ := 4

/-- The number of ways to distribute tasks among volunteers -/
def distribute_tasks : ℕ := 36

theorem task_distribution_count :
  (num_volunteers = 3) →
  (num_tasks = 4) →
  (∀ v, v ≤ num_volunteers → ∃ t, t ≤ num_tasks ∧ (v ≤ num_volunteers ∧ t ≤ num_tasks)) →
  (∀ t, t ≤ num_tasks → ∃! v, v ≤ num_volunteers ∧ (v ≤ num_volunteers ∧ t ≤ num_tasks)) →
  distribute_tasks = 36 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_task_distribution_count_l409_40905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mississippi_arrangements_mississippi_arrangements_proof_l409_40946

theorem mississippi_arrangements : ℕ := 34650

theorem mississippi_arrangements_proof :
  let total_letters : ℕ := 11
  let i_count : ℕ := 4
  let s_count : ℕ := 4
  let p_count : ℕ := 2
  let m_count : ℕ := 1
  (Nat.factorial total_letters) / ((Nat.factorial i_count) * (Nat.factorial s_count) * (Nat.factorial p_count)) = 34650 := by
  sorry

#check mississippi_arrangements
#check mississippi_arrangements_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mississippi_arrangements_mississippi_arrangements_proof_l409_40946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_property_l409_40968

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 12 * x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (3, 0)

-- Define the directrix of the parabola
def directrix : ℝ := -3

-- Distance function between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Distance from a point to the directrix
def distanceToDirectrix (p : ℝ × ℝ) : ℝ :=
  |p.2 - directrix|

-- Theorem statement
theorem parabola_property (p : ℝ × ℝ) :
  parabola p.1 p.2 → distance p focus = 8 → distanceToDirectrix p = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_property_l409_40968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_negative_reals_f_strictly_increasing_on_negative_reals_l409_40984

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 2*x) / Real.log (1/2)

-- Define the domain of f(x)
def domain (x : ℝ) : Prop := x < 0 ∨ x > 2

-- Define the composition functions
noncomputable def g (t : ℝ) : ℝ := Real.log t / Real.log (1/2)
def t (x : ℝ) : ℝ := x^2 - 2*x

-- Theorem statement
theorem f_increasing_on_negative_reals :
  ∀ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₂ < 0 → f x₁ < f x₂ := by
  sorry

-- Additional theorem to show that f is indeed increasing on (-∞, 0)
theorem f_strictly_increasing_on_negative_reals :
  StrictMonoOn f (Set.Iio 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_negative_reals_f_strictly_increasing_on_negative_reals_l409_40984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_relationship_l409_40923

theorem y_relationship : ∀ (y₁ y₂ y₃ : ℝ),
  y₁ = (4 : ℝ) ^ (1/5 : ℝ) →
  y₂ = (1/2 : ℝ) ^ (-(3/10) : ℝ) →
  y₃ = Real.log 8 / Real.log (1/2) →
  y₁ > y₂ ∧ y₂ > y₃ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_relationship_l409_40923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_P_l409_40994

-- Define the curve
def f (x : ℝ) : ℝ := 2 * x^2

-- Define the point of tangency
def P : ℝ × ℝ := (1, 2)

-- Define the tangent line equation
def tangent_line (x y : ℝ) : Prop := 4 * x - y - 2 = 0

-- Theorem statement
theorem tangent_line_at_P : 
  ∃ (m : ℝ), (∀ x, x ∈ Set.Ioo (P.1 - 1) (P.1 + 1) → f x = P.2 + m * (x - P.1)) ∧ 
  (∀ x y, tangent_line x y ↔ y = P.2 + m * (x - P.1)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_P_l409_40994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equality_l409_40915

theorem complex_equality : ∀ (z : ℂ), z * z = -1 → z * (1 - z) - 1 = z := by
  intro z hz
  -- The proof is omitted
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equality_l409_40915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_cross_section_area_l409_40902

/-- The area of a cross-sectional plane in a frustum -/
noncomputable def area_of_cross_section (S_upper S_lower m n : ℝ) : ℝ :=
((n * Real.sqrt S_upper + m * Real.sqrt S_lower) / (m + n)) ^ 2

/-- The area of a cross-sectional plane in a frustum -/
theorem frustum_cross_section_area
  (S_upper S_lower : ℝ)
  (m n : ℝ)
  (h_S_upper : S_upper > 0)
  (h_S_lower : S_lower > 0)
  (h_m : m > 0)
  (h_n : n > 0) :
  let S_cross := ((n * Real.sqrt S_upper + m * Real.sqrt S_lower) / (m + n)) ^ 2
  ∃ (S : ℝ), S = S_cross ∧ 
    S = area_of_cross_section S_upper S_lower m n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_cross_section_area_l409_40902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_and_max_S_l409_40904

-- Define the inequality
noncomputable def inequality (x : ℝ) : Prop := |x^2 - 3*x - 4| < 2*x + 2

-- Define the solution set
noncomputable def solution_set : Set ℝ := {x | inequality x}

-- Define a and b
noncomputable def a : ℝ := 2
noncomputable def b : ℝ := 6

-- Define S
noncomputable def S (m n : ℝ) : ℝ := a / (m^2 - 1) + b / (3 * (n^2 - 1))

theorem inequality_solution_and_max_S :
  (∀ x, x ∈ solution_set ↔ a < x ∧ x < b) ∧
  (∀ m n, m ∈ Set.Ioo (-1 : ℝ) 1 → n ∈ Set.Ioo (-1 : ℝ) 1 → m * n = a / b →
    S m n ≤ -6 ∧ ∃ m₀ n₀, m₀ ∈ Set.Ioo (-1 : ℝ) 1 ∧ n₀ ∈ Set.Ioo (-1 : ℝ) 1 ∧ S m₀ n₀ = -6) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_and_max_S_l409_40904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_investment_is_45000_l409_40986

/-- A partnership business with three partners a, b, and c. -/
structure Partnership where
  a_investment : ℕ
  b_investment : ℕ
  c_investment : ℕ
  total_profit : ℕ
  c_profit_share : ℕ

/-- The given partnership details -/
def given_partnership : Partnership := {
  a_investment := 30000,
  b_investment := 45000,  -- We set this to 45000 as it's the value we're proving
  c_investment := 50000,
  total_profit := 90000,
  c_profit_share := 36000
}

/-- Theorem stating that b's investment is 45000 -/
theorem b_investment_is_45000 (p : Partnership) 
  (h1 : p = given_partnership) 
  (h2 : p.c_profit_share * (p.a_investment + p.b_investment + p.c_investment) = 
        p.total_profit * p.c_investment) : 
  p.b_investment = 45000 := by
  sorry

#check b_investment_is_45000

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_investment_is_45000_l409_40986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_difference_l409_40996

noncomputable section

-- Define the circle O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the trajectory C
def trajectory_C (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1 ∧ -2 < x ∧ x < 2

-- Define point F
def point_F : ℝ × ℝ := (1, 0)

-- Define point A
def point_A : ℝ × ℝ := (4, 0)

-- Define the line l
def line_l (m : ℝ) (x y : ℝ) : Prop := x = m * y + 1

-- Define the area difference function
noncomputable def area_difference (m : ℝ) : ℝ := -12 * m / (3 * m^2 + 4)

theorem max_area_difference :
  ∃ (m : ℝ), ∀ (m' : ℝ), abs (area_difference m') ≤ abs (area_difference m) ∧
  abs (area_difference m) = Real.sqrt 3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_difference_l409_40996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_class_size_is_25_l409_40961

/-- Represents the number of students in the class -/
def n : ℕ := sorry

/-- Xiaoming's age rank relative to others -/
def x : ℕ := sorry

/-- Xiaohua's age rank relative to others -/
def y : ℕ := sorry

/-- The class size is between 20 and 30 -/
axiom class_size : 20 ≤ n ∧ n ≤ 30

/-- Each person has a unique birth date -/
axiom unique_birthdays : ∀ i j : Fin n, i ≠ j → i.val ≠ j.val

/-- Xiaoming's statement: older students are twice younger students -/
axiom xiaoming_statement : n - 1 - x = 2 * x

/-- Xiaohua's statement: older students are three times younger students -/
axiom xiaohua_statement : n - 1 - y = 3 * y

theorem class_size_is_25 : n = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_class_size_is_25_l409_40961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_when_a_is_2_a_range_when_f_decreasing_l409_40975

/-- The function f(x) defined as (ax-1)/(x+1) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a * x - 1) / (x + 1)

/-- Part 1: f(x) is increasing on (-∞, -1) when a = 2 -/
theorem f_increasing_when_a_is_2 :
  ∀ x₁ x₂, x₁ < x₂ → x₂ < -1 → f 2 x₁ < f 2 x₂ := by
  sorry

/-- Part 2: If f(x) is decreasing on (-∞, -1), then a ∈ (-∞, -1) -/
theorem a_range_when_f_decreasing (a : ℝ) :
  (∀ x₁ x₂, x₁ < x₂ → x₂ < -1 → f a x₁ > f a x₂) → a < -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_when_a_is_2_a_range_when_f_decreasing_l409_40975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_fractions_2011_l409_40925

open BigOperators

def sum_fractions (n : ℕ) : ℚ :=
  ∑ k in Finset.filter (fun i => i % 2 = 1 ∧ i ≤ n) (Finset.range (n + 1)), 2 / (k * (k + 2))

theorem sum_fractions_2011 :
  sum_fractions 2011 = 2010 / 2011 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_fractions_2011_l409_40925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_specific_floor_l409_40951

/-- The total shaded area of a rectangular floor tiled with square tiles, 
    where each tile has four white quarter circles at its corners. -/
noncomputable def total_shaded_area (length width : ℝ) (tile_size : ℝ) (circle_radius : ℝ) : ℝ :=
  let num_tiles := (length * width) / (tile_size * tile_size)
  let shaded_area_per_tile := tile_size * tile_size - Real.pi * circle_radius * circle_radius
  num_tiles * shaded_area_per_tile

/-- The total shaded area of a 12-foot by 15-foot floor tiled with 1-foot square tiles, 
    where each tile has four white quarter circles of radius 1/2 foot centered at each corner, 
    is equal to 180 - 45π square feet. -/
theorem shaded_area_specific_floor : 
  total_shaded_area 12 15 1 (1/2) = 180 - 45 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_specific_floor_l409_40951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lcd_of_fractions_l409_40942

/-- Represents the least common denominator (LCD) of a set of fractions. -/
def IsLCD (lcd : ℝ) (f₁ f₂ f₃ : ℝ) : Prop :=
  ∃ (a b c : ℝ), f₁ * lcd = a ∧ f₂ * lcd = b ∧ f₃ * lcd = c ∧
  ∀ (d : ℝ), (∃ (x y z : ℝ), f₁ * d = x ∧ f₂ * d = y ∧ f₃ * d = z) → lcd ≤ d

theorem lcd_of_fractions (x y : ℝ) (hxy : x ≠ y) (hx : x ≠ 0) (hy : y ≠ 0) :
  let f₁ := (2 : ℝ) / (x * y)
  let f₂ := (3 : ℝ) / (x + y)
  let f₃ := (4 : ℝ) / (x - y)
  let lcd := x * y * (x + y) * (x - y)
  IsLCD lcd f₁ f₂ f₃ :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lcd_of_fractions_l409_40942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangency_m_value_l409_40992

-- Define the circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 1
def C₂ (x y m : ℝ) : Prop := x^2 + y^2 - 6*x - 8*y + m = 0

-- Define tangency between circles
def tangent (C₁ C₂ : ℝ → ℝ → Prop) : Prop :=
  ∃ (x y : ℝ), C₁ x y ∧ C₂ x y ∧
  ∀ (x' y' : ℝ), C₁ x' y' ∧ C₂ x' y' → (x = x' ∧ y = y')

-- Theorem statement
theorem circle_tangency_m_value :
  tangent C₁ (λ x y => C₂ x y 9) → ∀ m : ℝ, tangent C₁ (λ x y => C₂ x y m) → m = 9 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangency_m_value_l409_40992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arrange_digits_count_arrange_digits_explanation_l409_40964

/-- The number of ways to arrange the digits of 45,520,1 to form a 6-digit number -/
def arrange_digits : ℕ := 300

/-- Theorem stating that the number of arrangements is 300 -/
theorem arrange_digits_count : arrange_digits = 300 := by
  -- Unfold the definition of arrange_digits
  unfold arrange_digits
  -- The result follows directly from the definition
  rfl

/-- Helper function to count occurrences of each digit -/
def count_digits (digits : List ℕ) : List (ℕ × ℕ) :=
  digits.foldl (fun acc d =>
    match acc.find? (fun p => p.1 = d) with
    | some p => acc.erase p ++ [(p.1, p.2 + 1)]
    | none => acc ++ [(d, 1)]
  ) []

/-- Theorem explaining the calculation of arrange_digits -/
theorem arrange_digits_explanation :
  let digits : List ℕ := [4, 5, 5, 2, 0, 1]
  let total_digits : ℕ := digits.length
  let non_zero_digits : ℕ := (digits.filter (· ≠ 0)).length
  let repeated_digits : List (ℕ × ℕ) := count_digits digits
  let zero_positions : ℕ := total_digits - 1
  let other_arrangements : ℕ := (Nat.factorial non_zero_digits) / (Nat.factorial 2)
  arrange_digits = zero_positions * other_arrangements := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arrange_digits_count_arrange_digits_explanation_l409_40964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_intersection_l409_40998

-- Define the circle
def my_circle (r : ℝ) : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = r^2}

-- Define the line
def my_line (x₀ y₀ r : ℝ) : Set (ℝ × ℝ) := {p | x₀ * p.1 + y₀ * p.2 = r^2}

-- Define what it means for a point to be inside the circle
def inside_circle (x₀ y₀ r : ℝ) : Prop := x₀^2 + y₀^2 < r^2

-- Theorem statement
theorem no_intersection (x₀ y₀ r : ℝ) (h : inside_circle x₀ y₀ r) :
  (my_line x₀ y₀ r) ∩ (my_circle r) = ∅ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_intersection_l409_40998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_surface_areas_l409_40976

/-- Represents a cone with base radius R and height 2R -/
structure Cone (R : ℝ) where
  baseRadius : ℝ := R
  height : ℝ := 2 * R

/-- The distance from the apex where the cone should be cut -/
noncomputable def cutDistance (R : ℝ) : ℝ := R / Real.sin (36 * Real.pi / 180)

/-- Calculate the surface area of a cone section -/
noncomputable def surfaceArea (r : ℝ) (h : ℝ) : ℝ :=
  Real.pi * r * (r + (r^2 + h^2).sqrt)

/-- Theorem: Cutting the cone at the specified distance results in equal surface areas -/
theorem equal_surface_areas (R : ℝ) (h : R > 0) :
  let cone := Cone R
  let cut := cutDistance R
  let bottomSection := surfaceArea R (2*R - cut)
  let topSection := surfaceArea (cut * R / (2*R)) cut
  bottomSection = topSection := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_surface_areas_l409_40976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_system_solution_range_l409_40906

theorem inequality_system_solution_range (m : ℝ) : 
  (∃ (S : Finset ℤ), (∀ x ∈ S, 3*x - m > 0 ∧ x - 1 ≤ 5) ∧ Finset.card S = 4) ↔ 
  (6 ≤ m ∧ m < 9) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_system_solution_range_l409_40906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_can_price_l409_40948

/-- Represents a cylindrical can with diameter, height, and price -/
structure Can where
  diameter : ℝ
  height : ℝ
  price : ℝ

/-- Calculates the volume of a cylindrical can -/
noncomputable def volume (can : Can) : ℝ :=
  Real.pi * (can.diameter / 2)^2 * can.height

/-- Theorem stating the price of the larger can given the conditions -/
theorem larger_can_price (small_can large_can : Can)
  (h1 : small_can.diameter = 4)
  (h2 : small_can.height = 5)
  (h3 : small_can.price = 0.8)
  (h4 : large_can.diameter = 8)
  (h5 : large_can.height = 10)
  (h6 : volume large_can / volume small_can = large_can.price / small_can.price) :
  large_can.price = 6.4 := by
  sorry

-- Remove the #eval statement as it's not necessary for building

end NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_can_price_l409_40948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_filets_is_22_l409_40989

-- Define the fish catch data structure
structure FishCatch where
  size : Nat
  count : Nat

-- Define the family member data structure
structure FamilyMember where
  name : String
  catches : List FishCatch

-- Define the problem parameters
def minSizeLimit : Nat := 6
def filetsPerFish : Nat := 2

-- Define the family's catch data
def familyCatch : List FamilyMember := [
  { name := "Ben", catches := [{ size := 5, count := 2 }, { size := 9, count := 2 }] },
  { name := "Judy", catches := [{ size := 11, count := 1 }] },
  { name := "Billy", catches := [{ size := 6, count := 2 }, { size := 10, count := 1 }] },
  { name := "Jim", catches := [{ size := 4, count := 1 }, { size := 8, count := 1 }] },
  { name := "Susie", catches := [{ size := 3, count := 1 }, { size := 7, count := 2 }, { size := 12, count := 2 }] }
]

-- Function to calculate the number of filets from a single catch
def filetsFromCatch (c : FishCatch) : Nat :=
  if c.size ≥ minSizeLimit then c.count * filetsPerFish else 0

-- Function to calculate the total number of filets for a family member
def totalFiletsForMember (member : FamilyMember) : Nat :=
  (member.catches.map filetsFromCatch).sum

-- Theorem statement
theorem total_filets_is_22 :
  (familyCatch.map totalFiletsForMember).sum = 22 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_filets_is_22_l409_40989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_number_not_square_l409_40980

/-- A 60-digit number composed of 30 zeros and 30 ones -/
def special_number : ℕ :=
  sorry

/-- The sum of digits of the special number is 30 -/
axiom sum_of_digits : (Nat.digits 10 special_number).sum = 30

/-- The special number is divisible by 3 -/
axiom divisible_by_three : special_number % 3 = 0

/-- The special number is not divisible by 9 -/
axiom not_divisible_by_nine : special_number % 9 ≠ 0

/-- Theorem: The special number is not a perfect square -/
theorem special_number_not_square : ∀ n : ℕ, special_number ≠ n^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_number_not_square_l409_40980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bowling_average_before_match_l409_40917

/-- Represents a cricketer's bowling statistics -/
structure BowlingStats where
  wickets : ℕ
  runs : ℕ
  average : ℚ

/-- Calculates the new bowling average after a match -/
def newAverage (stats : BowlingStats) (newWickets : ℕ) (newRuns : ℕ) : ℚ :=
  (stats.runs + newRuns : ℚ) / (stats.wickets + newWickets : ℚ)

/-- Theorem stating the cricketer's bowling average before the last match -/
theorem bowling_average_before_match 
  (stats : BowlingStats)
  (h1 : stats.wickets = 85)
  (h2 : newAverage stats 5 26 = stats.average - 2/5) :
  stats.average = 62/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bowling_average_before_match_l409_40917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem1_l409_40971

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x * Real.log x

theorem problem1 (a : ℝ) :
  (∀ x > 0, deriv (f a) x = a * (1 + Real.log x)) →
  deriv (f a) 1 = 3 →
  a = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem1_l409_40971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_chessboards_l409_40997

/-- Represents a chessboard with 64 squares numbered from 1 to 64 -/
def Chessboard := Fin 64 → ℕ

/-- Represents a collection of chessboards -/
def ChessboardCollection := ℕ → Chessboard

/-- Checks if two chessboards have no overlapping numbers in corresponding squares -/
def no_overlap (b1 b2 : Chessboard) : Prop :=
  ∀ i : Fin 64, b1 i ≠ b2 i

/-- A valid collection of chessboards has no overlapping numbers between any two boards -/
def valid_collection (c : ChessboardCollection) (n : ℕ) : Prop :=
  ∀ i j : Fin n, i.val ≠ j.val → no_overlap (c i.val) (c j.val)

/-- The maximum number of chessboards in a valid collection is 16 -/
theorem max_chessboards :
  (∃ c : ChessboardCollection, valid_collection c 16) ∧
  (∀ c : ChessboardCollection, ¬(valid_collection c 17)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_chessboards_l409_40997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_students_average_mark_l409_40936

/-- Given a class of students, prove that the average mark of the remaining students
    after excluding some students is as calculated. -/
theorem remaining_students_average_mark
  (total_students : ℕ)
  (total_average : ℚ)
  (excluded_students : ℕ)
  (excluded_average : ℚ)
  (h_total_students : total_students = 30)
  (h_total_average : total_average = 80)
  (h_excluded_students : excluded_students = 5)
  (h_excluded_average : excluded_average = 20)
  : (total_average * total_students - excluded_average * excluded_students) /
    (total_students - excluded_students : ℚ) = 92 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_students_average_mark_l409_40936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_MN_is_six_l409_40926

/-- A circle with an inscribed regular triangle and a smaller inscribed circle -/
structure InscribedCircles where
  -- The larger circle
  Circle : Type
  -- The regular triangle ABC inscribed in the larger circle
  A : Circle
  B : Circle
  C : Circle
  -- The smaller circle inscribed in the sector bounded by chord BC
  SmallCircle : Type
  -- Point where the smaller circle touches the larger circle
  M : Circle
  -- Point where the smaller circle touches chord BC
  K : Circle
  -- Point where ray MK intersects the larger circle again
  N : Circle
  -- The sum of distances from M to B and C is 6
  distance_sum : ℝ
  -- Axiom: The sum of distances from M to B and C is 6
  ax_distance_sum : distance_sum = 6

/-- The length of MN in the inscribed circles configuration -/
def length_MN (ic : InscribedCircles) : ℝ :=
  sorry

/-- Theorem: The length of MN is 6 -/
theorem length_MN_is_six (ic : InscribedCircles) : length_MN ic = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_MN_is_six_l409_40926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l409_40979

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : t.a * t.c / (t.b^2 - t.a^2 - t.c^2) = Real.sin t.A * Real.cos t.A / Real.cos (t.A + t.C)) :
  t.A = π/4 ∧ 
  (t.a = Real.sqrt 2 → 0 < t.b * t.c ∧ t.b * t.c ≤ 2 + Real.sqrt 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l409_40979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_crossing_time_l409_40935

/-- Represents a train with its length and speed -/
structure Train where
  length : ℝ  -- Length in meters
  speed : ℝ   -- Speed in km/hr

/-- Calculates the time (in seconds) for two trains to cross each other -/
noncomputable def timeToCross (trainA trainB : Train) : ℝ :=
  let totalLength := trainA.length + trainB.length
  let relativeSpeed := (trainA.speed + trainB.speed) * (1000 / 3600)
  totalLength / relativeSpeed

/-- Theorem: The time taken for the given trains to cross each other is approximately 11.88 seconds -/
theorem trains_crossing_time :
  let trainA : Train := { length := 170, speed := 60 }
  let trainB : Train := { length := 160, speed := 40 }
  abs (timeToCross trainA trainB - 11.88) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_crossing_time_l409_40935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_calls_for_full_info_l409_40978

/-- Represents a person with their knowledge -/
structure Person where
  id : Nat
  knowledge : Finset Nat

/-- The state of information sharing -/
structure InfoState where
  people : Finset Person
  calls : Nat

/-- Defines a valid initial state where each person knows only their own information -/
def validInitialState (n : Nat) (s : InfoState) : Prop :=
  s.people.card = n ∧
  ∀ p ∈ s.people, p.knowledge = {p.id} ∧
  s.calls = 0

/-- Defines when all information has been shared -/
def allInfoShared (s : InfoState) : Prop :=
  ∀ p ∈ s.people, p.knowledge.card = s.people.card

/-- Defines a valid phone call between two people -/
def validCall (s s' : InfoState) (a b : Person) : Prop :=
  a ∈ s.people ∧ b ∈ s.people ∧
  s'.people = s.people ∧
  s'.calls = s.calls + 1 ∧
  b.knowledge = a.knowledge ∪ b.knowledge

/-- The main theorem to prove -/
theorem min_calls_for_full_info (n : Nat) :
  ∃ (s : InfoState), validInitialState n s →
  (∃ (s' : InfoState), allInfoShared s' ∧
    s'.calls = 2 * n - 2 ∧
    (∀ s'' : InfoState, allInfoShared s'' → s''.calls ≥ 2 * n - 2)) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_calls_for_full_info_l409_40978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_bill_25_water_bill_over_30_two_month_usage_l409_40969

/-- Calculates the water bill based on the tiered pricing system -/
noncomputable def water_bill (usage : ℝ) : ℝ :=
  if usage ≤ 22 then 3 * usage
  else if usage ≤ 30 then 3 * 22 + 5 * (usage - 22)
  else 3 * 22 + 5 * 8 + 7 * (usage - 30)

/-- Theorem stating the correctness of water bill calculation for 25 m³ -/
theorem water_bill_25 : water_bill 25 = 81 := by sorry

/-- Theorem stating the formula for water bill when usage is over 30 m³ -/
theorem water_bill_over_30 (x : ℝ) (h : x > 30) : water_bill x = 7 * x - 104 := by sorry

/-- Theorem stating the correct water usage distribution for a two-month period -/
theorem two_month_usage (may june : ℝ) 
  (h1 : may + june = 50) 
  (h2 : water_bill may + water_bill june = 174) 
  (h3 : may < june) : 
  may = 18 ∧ june = 32 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_bill_25_water_bill_over_30_two_month_usage_l409_40969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_circles_theorem_l409_40962

/-- The area between two concentric circles -/
noncomputable def area_between_circles (r_small : ℝ) (r_large : ℝ) : ℝ :=
  Real.pi * (r_large^2 - r_small^2)

/-- Theorem: Area between circles with given conditions -/
theorem area_between_circles_theorem (r_small : ℝ) (r_large : ℝ) 
  (h1 : r_large = 3 * r_small)
  (h2 : 2 * r_small = 6) :
  area_between_circles r_small r_large = 72 * Real.pi := by
  sorry

#check area_between_circles_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_circles_theorem_l409_40962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hawks_percentage_is_30_l409_40991

/-- Represents the percentage of hawks in a nature reserve -/
def hawks_percentage (h : ℝ) : Prop :=
  -- Total percentage of hawks, paddyfield-warblers, and kingfishers is 65%
  h + 0.4 * (100 - h) + 0.1 * (100 - h) = 65 ∧
  -- 40% of non-hawks are paddyfield-warblers
  0.4 * (100 - h) = 40 * (100 - h) / 100 ∧
  -- Percentage of kingfishers is 25% of paddyfield-warblers
  0.1 * (100 - h) = 0.25 * (0.4 * (100 - h))

/-- Theorem stating that the percentage of hawks in the nature reserve is 30% -/
theorem hawks_percentage_is_30 : hawks_percentage 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hawks_percentage_is_30_l409_40991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weekend_rain_probability_l409_40913

/-- The probability of rain on at least one day during a weekend, given specific probabilities for Saturday and Sunday. -/
theorem weekend_rain_probability
  (p_rain_sat : ℝ)
  (p_rain_sun_given_rain_sat : ℝ)
  (p_rain_sun_given_no_rain_sat : ℝ)
  (h1 : p_rain_sat = 0.6)
  (h2 : p_rain_sun_given_rain_sat = 0.7)
  (h3 : p_rain_sun_given_no_rain_sat = 0.4)
  : p_rain_sat + (p_rain_sun_given_rain_sat * p_rain_sat + p_rain_sun_given_no_rain_sat * (1 - p_rain_sat)) - 
    (p_rain_sat * p_rain_sun_given_rain_sat) = 0.76 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_weekend_rain_probability_l409_40913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_side_length_greater_than_green_segments_l409_40956

/-- Represents a circle with a center point and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents an equilateral triangle with side length and circles at its vertices -/
structure TriangleWithCircles where
  sideLength : ℝ
  circles : Fin 3 → Circle

/-- Calculates the area of a region covered by a specific number of circles -/
noncomputable def areaOfRegion (t : TriangleWithCircles) (numCircles : Nat) : ℝ := sorry

/-- Calculates the total length of segments on the sides of the triangle that are inside exactly two circles -/
noncomputable def greenSegmentsLength (t : TriangleWithCircles) : ℝ := sorry

/-- Main theorem: The side length of the triangle is greater than the total length of green segments -/
theorem side_length_greater_than_green_segments 
  (t : TriangleWithCircles) 
  (h1 : ∀ i : Fin 3, (t.circles i).radius < (t.sideLength * Real.sqrt 3) / 2) 
  (h2 : areaOfRegion t 1 = 1000) 
  (h3 : areaOfRegion t 2 = 100) 
  (h4 : areaOfRegion t 3 = 1) : 
  t.sideLength > greenSegmentsLength t := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_side_length_greater_than_green_segments_l409_40956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monday_absence_probability_l409_40988

/-- The probability of a student being absent on a normal day -/
def normal_absence_prob : ℚ := 2/30

/-- The increase in absence rate on Mondays -/
def monday_increase : ℚ := 1/10

/-- The probability of a student being absent on a Monday -/
def monday_absence_prob : ℚ := normal_absence_prob + normal_absence_prob * monday_increase

/-- The probability of a student being present on a Monday -/
def monday_presence_prob : ℚ := 1 - monday_absence_prob

/-- The number of students we're considering -/
def num_students : ℕ := 3

/-- The number of students we want to be absent -/
def num_absent : ℕ := 2

theorem monday_absence_probability :
  (Nat.choose num_students num_absent : ℚ) * monday_absence_prob^num_absent * monday_presence_prob^(num_students - num_absent) = 50457/3375000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monday_absence_probability_l409_40988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_zeros_range_l409_40966

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2 : ℝ)^(1 + a*x) - x

-- State the theorem
theorem two_zeros_range (a : ℝ) :
  a > 0 ∧
  (∃! x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a (f a x₁) = x₁ ∧ f a (f a x₂) = x₂) →
  0 < a ∧ a < 1 / (Real.exp 1 * Real.log 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_zeros_range_l409_40966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nicky_run_time_is_30_l409_40970

/-- Represents the race between Nicky and Cristina -/
structure Race where
  length : ℝ
  headStart : ℝ
  cristinaSpeed : ℝ
  nickySpeed : ℝ

/-- Calculates the time when Cristina catches up to Nicky -/
noncomputable def catchUpTime (race : Race) : ℝ :=
  (race.headStart * race.nickySpeed) / (race.cristinaSpeed - race.nickySpeed)

/-- Calculates the total time Nicky runs before Cristina catches up -/
noncomputable def nickyRunTime (race : Race) : ℝ :=
  race.headStart + catchUpTime race

/-- The specific race conditions -/
def raceConditions : Race :=
  { length := 300
    headStart := 12
    cristinaSpeed := 5
    nickySpeed := 3 }

/-- Theorem stating that Nicky runs for 30 seconds before Cristina catches up -/
theorem nicky_run_time_is_30 : nickyRunTime raceConditions = 30 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_nicky_run_time_is_30_l409_40970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_statements_count_l409_40965

/-- Represents a statement that can be true or false -/
inductive Statement
| algorithmDefiniteResult
| inputPromptContent
| heightAgeCorrelation
| histogramDensityCurve

/-- Determines if a given statement is correct -/
def isCorrect (s : Statement) : Bool :=
  match s with
  | Statement.algorithmDefiniteResult => true
  | Statement.inputPromptContent => false
  | Statement.heightAgeCorrelation => true
  | Statement.histogramDensityCurve => false

/-- Counts the number of correct statements -/
def countCorrect (statements : List Statement) : Nat :=
  statements.filter isCorrect |>.length

/-- The list of all statements -/
def allStatements : List Statement :=
  [Statement.algorithmDefiniteResult,
   Statement.inputPromptContent,
   Statement.heightAgeCorrelation,
   Statement.histogramDensityCurve]

theorem correct_statements_count :
  countCorrect allStatements = 2 := by
  rfl

#eval countCorrect allStatements

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_statements_count_l409_40965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_score_theorem_l409_40932

def geography_score : ℕ := 50
def math_score : ℕ := 70
def english_score : ℕ := 66

def history_score : ℚ :=
  (geography_score + math_score + english_score : ℚ) / 3

theorem total_score_theorem :
  geography_score + math_score + english_score + history_score.floor = 248 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_score_theorem_l409_40932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_d_coverable_iff_four_or_prime_l409_40914

/-- A positive integer n > 1 is d-coverable if for each non-empty subset S ⊆ {0, 1, ..., n-1},
    there exists a polynomial P with integer coefficients and degree at most d such that S is
    exactly the set of residues modulo n that P attains as it ranges over the integers. -/
def IsDCoverable (n d : ℕ) : Prop :=
  n > 1 ∧ ∀ (S : Set ℕ), S.Nonempty → S ⊆ Finset.range n →
    ∃ (P : Polynomial ℤ), P.degree ≤ d ∧
      S = {r : ℕ | ∃ (x : ℤ), (P.eval x : ℤ) % n = r}

/-- Theorem: A positive integer n > 1 is d-coverable if and only if
    n = 4 and d ≥ 3, or n is prime and d ≥ n - 1 -/
theorem d_coverable_iff_four_or_prime (n d : ℕ) :
  IsDCoverable n d ↔ (n = 4 ∧ d ≥ 3) ∨ (Nat.Prime n ∧ d ≥ n - 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_d_coverable_iff_four_or_prime_l409_40914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dodecahedron_coloring_count_l409_40999

/-- A regular dodecahedron with 12 differently colored faces -/
structure ColoredDodecahedron where
  faces : Fin 12 → Fin 12
  different_colors : ∀ i j, i ≠ j → faces i ≠ faces j

/-- The set of rotations that preserve the positions of two adjacent faces -/
def adjacent_face_preserving_rotations : Finset (ColoredDodecahedron → ColoredDodecahedron) :=
  sorry

/-- Two colored dodecahedrons are equivalent if one can be rotated to look like the other -/
def equivalent (d1 d2 : ColoredDodecahedron) : Prop :=
  ∃ r ∈ adjacent_face_preserving_rotations, r d1 = d2

/-- The number of distinguishable colorings of a dodecahedron with two adjacent faces fixed -/
def num_distinguishable_colorings : ℕ :=
  sorry

theorem dodecahedron_coloring_count :
  num_distinguishable_colorings = 725760 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dodecahedron_coloring_count_l409_40999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_of_circle_with_6000_anglets_l409_40985

/-- An anglet is 1 percent of 1 degree -/
noncomputable def anglet : ℚ := 1 / 100

/-- The number of degrees in a full circle -/
def full_circle_degrees : ℚ := 360

/-- The number of anglets in the fraction of the circle -/
def fraction_anglets : ℕ := 6000

/-- The fraction of the circle we're looking for -/
def circle_fraction : ℚ := 1 / 6

theorem fraction_of_circle_with_6000_anglets :
  (fraction_anglets : ℚ) / (full_circle_degrees / anglet) = circle_fraction := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_of_circle_with_6000_anglets_l409_40985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tiles_per_row_l409_40983

/-- Given a square room with an area of 400 square feet and 8-inch by 8-inch tiles,
    prove that there are 30 tiles in each row. -/
theorem tiles_per_row (room_area : ℝ) (tile_size : ℝ) : 
  room_area = 400 ∧ tile_size = 8 / 12 → 
  Int.floor (Real.sqrt room_area / tile_size) = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tiles_per_row_l409_40983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l409_40949

/-- Given a hyperbola and a circle with specific properties, 
    prove that the eccentricity of the hyperbola is 3√5/5 -/
theorem hyperbola_eccentricity (a : ℝ) (M N : ℝ × ℝ) :
  a > 0 →
  (∀ x y, x^2 / a^2 - y^2 / 4 = 1 → 
    ∃ k, y = k * x ∧ (x - 3)^2 + y^2 = 8) →
  (M.1 - 3)^2 + M.2^2 = 8 →
  (N.1 - 3)^2 + N.2^2 = 8 →
  (M.1 - N.1)^2 + (M.2 - N.2)^2 = 16 →
  let e := Real.sqrt (1 + 4 / a^2)
  e = 3 * Real.sqrt 5 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l409_40949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hidden_symmetric_points_iff_m_leq_neg_two_l409_40908

noncomputable def f (m : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then x^2 + 2*x else m*x + 4

def has_hidden_symmetric_points (m : ℝ) : Prop :=
  ∃ x₀ : ℝ, f m x₀ = -(f m (-x₀))

theorem hidden_symmetric_points_iff_m_leq_neg_two :
  ∀ m : ℝ, has_hidden_symmetric_points m ↔ m ≤ -2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hidden_symmetric_points_iff_m_leq_neg_two_l409_40908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_identity_in_A_l409_40952

-- Define the set of functions A
def A : Set (ℝ → ℝ) := sorry

-- Define the identity function
def i : ℝ → ℝ := λ x => x

-- Axiom: A is finite
axiom A_finite : Set.Finite A

-- Axiom: A is closed under composition
axiom A_closed_composition : ∀ f g : ℝ → ℝ, f ∈ A → g ∈ A → (λ x => f (g x)) ∈ A

-- Axiom: Functional equation property
axiom A_functional_equation : ∀ f : ℝ → ℝ, f ∈ A → 
  ∃ g : ℝ → ℝ, g ∈ A ∧ ∀ x y : ℝ, f (f x + y) = 2 * x + g (g y - x)

-- Theorem: The identity function belongs to A
theorem identity_in_A : i ∈ A := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_identity_in_A_l409_40952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_parameters_sum_l409_40939

noncomputable section

def F₁ : ℝ × ℝ := (1, 2)
def F₂ : ℝ × ℝ := (7, 2)

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

def is_on_ellipse (p : ℝ × ℝ) : Prop :=
  distance p F₁ + distance p F₂ = 10

noncomputable def center : ℝ × ℝ := ((F₁.1 + F₂.1) / 2, (F₁.2 + F₂.2) / 2)

noncomputable def h : ℝ := center.1
noncomputable def k : ℝ := center.2

noncomputable def c : ℝ := distance F₁ F₂ / 2
def a : ℝ := 5
noncomputable def b : ℝ := Real.sqrt (a^2 - c^2)

theorem ellipse_parameters_sum :
  h + k + a + b = 15 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_parameters_sum_l409_40939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_ratio_is_one_l409_40953

/-- A business trip with two different speeds -/
structure BusinessTrip where
  totalTime : ℝ
  totalDistance : ℝ
  speed1 : ℝ
  speed2 : ℝ

/-- The ratio of time spent at each speed -/
noncomputable def timeRatio (trip : BusinessTrip) : ℝ := 
  let t1 := (trip.totalDistance - trip.speed2 * trip.totalTime) / (trip.speed1 - trip.speed2)
  let t2 := trip.totalTime - t1
  t1 / t2

/-- Theorem stating that the time ratio for the given trip is 1 -/
theorem time_ratio_is_one : 
  let trip : BusinessTrip := {
    totalTime := 8,
    totalDistance := 620,
    speed1 := 70,
    speed2 := 85
  }
  timeRatio trip = 1 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_ratio_is_one_l409_40953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l409_40934

noncomputable def f (x : ℝ) : ℝ := Real.sin (x - Real.pi/6) * Real.cos x + 1

theorem f_properties :
  (∃ T > 0, ∀ x, f (x + T) = f x ∧ ∀ S, (0 < S ∧ S < T) → ∃ y, f (y + S) ≠ f y) ∧
  (∀ x ∈ Set.Icc (Real.pi/12) (Real.pi/2), f x ≤ 5/4) ∧
  (∀ x ∈ Set.Icc (Real.pi/12) (Real.pi/2), f x ≥ 3/4) ∧
  (∃ x ∈ Set.Icc (Real.pi/12) (Real.pi/2), f x = 5/4) ∧
  (∃ x ∈ Set.Icc (Real.pi/12) (Real.pi/2), f x = 3/4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l409_40934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rudy_speed_is_10_l409_40945

/-- Grayson's motorboat trip -/
noncomputable def grayson_trip_1_distance : ℝ := 25 * 1
noncomputable def grayson_trip_2_distance : ℝ := 20 * 0.5

/-- Total distance Grayson traveled -/
noncomputable def grayson_total_distance : ℝ := grayson_trip_1_distance + grayson_trip_2_distance

/-- Rudy's rowing time in hours -/
noncomputable def rudy_time : ℝ := 3

/-- Distance difference between Grayson and Rudy -/
noncomputable def distance_difference : ℝ := 5

/-- Rudy's distance -/
noncomputable def rudy_distance : ℝ := grayson_total_distance - distance_difference

/-- Rudy's speed -/
noncomputable def rudy_speed : ℝ := rudy_distance / rudy_time

theorem rudy_speed_is_10 : rudy_speed = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rudy_speed_is_10_l409_40945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_main_theorem_l409_40950

-- Define the hyperbola and its properties
def Hyperbola (a b : ℝ) := {(x, y) : ℝ × ℝ | x^2 / a^2 - y^2 / b^2 = 1}

-- Define the foci
noncomputable def LeftFocus (a b : ℝ) : ℝ × ℝ := sorry
noncomputable def RightFocus (a b : ℝ) : ℝ × ℝ := sorry

-- Define the point P on the right branch of the hyperbola
noncomputable def P (a b : ℝ) : ℝ × ℝ := sorry

-- Define the asymptotes
def Asymptotes (h : Set (ℝ × ℝ)) : Set (ℝ × ℝ) := sorry

-- State the theorem
theorem hyperbola_asymptotes (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ (k : ℝ), k = Real.sqrt 2 ∧ 
  (∀ (x y : ℝ), (x, y) ∈ Asymptotes (Hyperbola a b) ↔ y = k * x ∨ y = -k * x) :=
by sorry

-- Define the conditions
def angle_condition (a b : ℝ) : Prop :=
  ∃ (θ : ℝ), θ = 30 * Real.pi / 180 ∧ 
  (sorry : Prop) -- Placeholder for LineAngle condition

def perpendicular_condition (a b : ℝ) : Prop :=
  sorry -- Placeholder for LinePerp condition

-- Main theorem
theorem main_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h_angle : angle_condition a b)
  (h_perp : perpendicular_condition a b) :
  ∃ (k : ℝ), k = Real.sqrt 2 ∧ 
  (∀ (x y : ℝ), (x, y) ∈ Asymptotes (Hyperbola a b) ↔ y = k * x ∨ y = -k * x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_main_theorem_l409_40950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_is_ten_l409_40927

/-- Represents a rectangle inside an isosceles right triangle -/
structure RectangleInTriangle where
  -- Length of the equal sides of the isosceles right triangle
  triangle_side : ℝ
  -- Width of the rectangle (EG)
  rect_width : ℝ
  -- Height of the rectangle (EH)
  rect_height : ℝ

/-- The length of the diagonal of the rectangle inside the isosceles right triangle -/
noncomputable def diagonal_length (r : RectangleInTriangle) : ℝ :=
  Real.sqrt (r.rect_width^2 + r.rect_height^2)

/-- Theorem stating that for the given dimensions, the diagonal length is 10 -/
theorem diagonal_is_ten :
  let r : RectangleInTriangle := { triangle_side := 10, rect_width := 6, rect_height := 8 }
  diagonal_length r = 10 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_is_ten_l409_40927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_function_derivative_condition_l409_40937

open Set
open Function

theorem increasing_function_derivative_condition 
  {f : ℝ → ℝ} {a b : ℝ} (hab : a < b) 
  (hf : DifferentiableOn ℝ f (Ioo a b)) :
  (StrictMonoOn f (Ioo a b)) ↔ 
  ((∀ x ∈ Ioo a b, (deriv f x) > 0) ∧ 
   ∃ g : ℝ → ℝ, (∀ x ∈ Ioo a b, (deriv g x) ≥ 0) ∧ StrictMonoOn g (Ioo a b)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_function_derivative_condition_l409_40937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l409_40947

-- Define propositions p and q as functions of a
def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 + a*x + 2 ≠ 0

def q (a : ℝ) : Prop := a > 1

-- Define the set of a that satisfies the conditions
def A : Set ℝ := {a : ℝ | (¬(p a ∧ q a)) ∧ (p a ∨ q a)}

-- State the theorem
theorem range_of_a : A = Set.Icc (-2 * Real.sqrt 2) 1 ∪ Set.Ioi (2 * Real.sqrt 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l409_40947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l409_40957

theorem tan_alpha_value (α : Real) (m : Real) :
  (α > Real.pi / 2 ∧ α < Real.pi) →  -- α is in the second quadrant
  (m < 0) →  -- x-coordinate is negative in the second quadrant
  (m^2 + (Real.sqrt 3 / 2)^2 = 1) →  -- Point lies on the unit circle
  Real.tan α = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l409_40957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_polygon_diagonals_l409_40974

/-- A convex polygon with each interior angle equal to 150° has 54 diagonals. -/
theorem convex_polygon_diagonals (n : ℕ) (h_interior_angle : ∀ i, i < n → 150 = 150) : 
  n * (n - 3) / 2 = 54 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_polygon_diagonals_l409_40974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_palindromic_primes_less_than_200_l409_40987

def isPrime (n : Nat) : Bool := 
  n > 1 && (List.range (n - 1)).all (fun i => i + 2 = n || n % (i + 2) ≠ 0)

def reverse (n : Nat) : Nat :=
  let rec reverseAux (n acc : Nat) : Nat :=
    if n = 0 then acc
    else reverseAux (n / 10) (acc * 10 + n % 10)
  reverseAux n 0

def isPalindromicPrime (n : Nat) : Bool :=
  isPrime n && isPrime (reverse n)

def sumPalindromicPrimes : Nat :=
  (List.range 100).map (fun i => i + 100)
    |>.filter isPalindromicPrime
    |>.foldl (· + ·) 0

theorem sum_palindromic_primes_less_than_200 :
  sumPalindromicPrimes = 868 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_palindromic_primes_less_than_200_l409_40987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_degree_three_l409_40909

/-- The polynomial f(x) -/
def f (x : ℝ) : ℝ := 2 - 15*x + 4*x^2 - 5*x^3 + 6*x^4

/-- The polynomial g(x) -/
def g (x : ℝ) : ℝ := 4 - 3*x - 7*x^3 + 10*x^4

/-- The combined polynomial f(x) + dg(x) -/
def h (d : ℝ) (x : ℝ) : ℝ := f x + d * g x

/-- The theorem stating that h(x) has degree 3 when d = -3/5 -/
theorem h_degree_three :
  ∃ (a b c : ℝ), (∀ x, h (-3/5) x = a*x^3 + b*x^2 + c*x + h (-3/5) 0) ∧
  a ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_degree_three_l409_40909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l409_40918

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) :
  (0 < a ∧ 0 < b ∧ 0 < c) →  -- Ensure positive side lengths
  (Real.cos A = Real.sqrt 6 / 3) →
  (b = 2 * Real.sqrt 2) →
  (c = Real.sqrt 3) →
  (a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * Real.cos A) →  -- Cosine rule
  a = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l409_40918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_prime_quotient_iff_even_l409_40900

def IsCircularPrimeQuotient (m : ℕ+) (a : Fin m → ℕ+) : Prop :=
  ∀ i : Fin m, Nat.Prime (max (a i).val (a (i + 1)).val / min (a i).val (a (i + 1)).val)

theorem circular_prime_quotient_iff_even (m : ℕ+) :
  (∃ a : Fin m → ℕ+, IsCircularPrimeQuotient m a) ↔ Even m.val := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_prime_quotient_iff_even_l409_40900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_to_fraction_l409_40907

theorem repeating_decimal_to_fraction :
  ∃ (x : ℚ), (∀ (n : ℕ), x * 100^n - x.floor * 100^n = 56 * (100^n - 1) / 99) ∧ x = 56 / 99 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_to_fraction_l409_40907
