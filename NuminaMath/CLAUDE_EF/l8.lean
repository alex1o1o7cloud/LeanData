import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_triangle_OAB_l8_854

/-- The area of a triangle given three points in ℝ² -/
noncomputable def area_triangle (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  (1/2) * abs ((p2.1 - p1.1) * (p3.2 - p1.2) - (p3.1 - p1.1) * (p2.2 - p1.2))

/-- The maximum area of triangle OAB given the specified conditions -/
theorem max_area_triangle_OAB :
  let O : ℝ × ℝ := (0, 0)
  let P : ℝ × ℝ := (2, 1)
  ∃ (A B : ℝ × ℝ),
    (∃ a > 0, A = (a, 0)) ∧
    (∃ b > 0, B = (0, b)) ∧
    ((P.1 - A.1) * (P.2 - B.2) + (P.2 - A.2) * (B.1 - P.1) = 0) ∧
    (∀ A' B' : ℝ × ℝ,
      (∃ a' > 0, A' = (a', 0)) →
      (∃ b' > 0, B' = (0, b')) →
      ((P.1 - A'.1) * (P.2 - B'.2) + (P.2 - A'.2) * (B'.1 - P.1) = 0) →
      area_triangle O A' B' ≤ area_triangle O A B) ∧
    area_triangle O A B = 25/16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_triangle_OAB_l8_854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_intervals_l8_880

noncomputable def f (x : ℝ) : ℝ := Real.sin (1/2 * x + Real.pi/3)

theorem monotonicity_intervals (x : ℝ) (h : x ∈ Set.Icc (-2*Real.pi) (2*Real.pi)) :
  (∀ x₁ x₂, x₁ ∈ Set.Icc (-5*Real.pi/3) (Real.pi/3) → x₂ ∈ Set.Icc (-5*Real.pi/3) (Real.pi/3) → x₁ ≤ x₂ → f x₁ ≤ f x₂) ∧
  (∀ x₁ x₂, x₁ ∈ Set.Icc (-2*Real.pi) (-5*Real.pi/3) → x₂ ∈ Set.Icc (-2*Real.pi) (-5*Real.pi/3) → x₁ ≤ x₂ → f x₁ ≥ f x₂) ∧
  (∀ x₁ x₂, x₁ ∈ Set.Icc (Real.pi/3) (2*Real.pi) → x₂ ∈ Set.Icc (Real.pi/3) (2*Real.pi) → x₁ ≤ x₂ → f x₁ ≥ f x₂) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_intervals_l8_880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_height_theorem_l8_895

/-- The area of a trapezium given the lengths of its parallel sides and the distance between them. -/
noncomputable def trapezium_area (a b h : ℝ) : ℝ := (1/2) * (a + b) * h

/-- Theorem: In a trapezium with parallel sides of 20 cm and 18 cm and an area of 190 square centimeters, the distance between the parallel sides is 10 cm. -/
theorem trapezium_height_theorem :
  ∃ (h : ℝ), trapezium_area 20 18 h = 190 ∧ h = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_height_theorem_l8_895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_travel_time_proof_l8_802

/-- Given a speed in km/h and a distance in meters, calculate the time traveled in seconds -/
noncomputable def calculate_time (speed_kmh : ℝ) (distance_m : ℝ) : ℝ :=
  let speed_ms := speed_kmh * 1000 / 3600
  distance_m / speed_ms

theorem travel_time_proof (speed_kmh : ℝ) (distance_m : ℝ) 
    (h1 : speed_kmh = 63) 
    (h2 : distance_m = 437.535) : 
  calculate_time speed_kmh distance_m = 25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_travel_time_proof_l8_802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nGonCentersEquilateral_l8_853

-- Define a triangle as a triple of complex numbers
def Triangle := ℂ × ℂ × ℂ

-- Define the nth root of unity
noncomputable def nthRootOfUnity (n : ℕ) : ℂ :=
  Complex.exp (2 * Real.pi * Complex.I / n)

-- Define the center of a regular n-gon constructed on a side of a triangle
noncomputable def nGonCenter (a b : ℂ) (n : ℕ) : ℂ :=
  (a * nthRootOfUnity n - b) / (nthRootOfUnity n - 1)

-- Define the third root of unity
noncomputable def thirdRootOfUnity : ℂ :=
  Complex.exp (2 * Real.pi * Complex.I / 3)

-- Statement of the theorem
theorem nGonCentersEquilateral (t : Triangle) (n : ℕ) :
  let (a, b, c) := t
  let o₁ := nGonCenter a b n
  let o₂ := nGonCenter b c n
  let o₃ := nGonCenter c a n
  (o₁ + o₂ * thirdRootOfUnity + o₃ * thirdRootOfUnity^2 = 0) ↔ n = 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_nGonCentersEquilateral_l8_853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_definite_integral_problem_l8_800

open Real MeasureTheory

theorem definite_integral_problem : ∫ x in Set.Icc 0 1, (x^2 + Real.exp x - 1/3) = Real.exp 1 - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_definite_integral_problem_l8_800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_consecutive_composites_l8_822

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def is_two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n ≤ 99

theorem largest_consecutive_composites :
  ∃ (a : ℕ),
    (a = 36) ∧
    (∀ i : ℕ, i ∈ [a - 4, a - 3, a - 2, a - 1, a] → is_two_digit i) ∧
    (∀ i : ℕ, i ∈ [a - 4, a - 3, a - 2, a - 1, a] → i < 40) ∧
    (∀ i : ℕ, i ∈ [a - 4, a - 3, a - 2, a - 1, a] → ¬(is_prime i)) ∧
    (∀ b : ℕ, b > a →
      ¬(∀ i : ℕ, i ∈ [b - 4, b - 3, b - 2, b - 1, b] → is_two_digit i ∧ i < 40 ∧ ¬(is_prime i))) :=
by
  sorry

#check largest_consecutive_composites

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_consecutive_composites_l8_822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_up_for_specific_journey_l8_832

/-- Represents a round trip journey up and down a hill -/
structure HillJourney where
  totalTime : ℝ
  upTime : ℝ
  downTime : ℝ
  averageSpeed : ℝ

/-- Calculates the average speed while climbing up the hill -/
noncomputable def averageSpeedUp (journey : HillJourney) : ℝ :=
  (journey.averageSpeed * journey.totalTime) / (2 * journey.upTime)

/-- Theorem stating the average speed while climbing up for a specific journey -/
theorem average_speed_up_for_specific_journey :
  let journey : HillJourney := {
    totalTime := 6,
    upTime := 4,
    downTime := 2,
    averageSpeed := 1.5
  }
  averageSpeedUp journey = 1.125 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_up_for_specific_journey_l8_832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangency_radius_sum_p_q_l8_894

/-- Circle structure -/
structure Circle where
  center : Point
  radius : ℝ

/-- Point structure -/
structure Point where
  x : ℝ
  y : ℝ

/-- Internally tangent circles -/
def internally_tangent (c1 c2 : Circle) (p : Point) : Prop := sorry

/-- Externally tangent circles -/
def externally_tangent (c1 c2 : Circle) : Prop := sorry

/-- Circle tangent to diameter of another circle -/
def tangent_to_diameter (c1 c2 : Circle) (a b : Point) : Prop := sorry

/-- Given three circles C, D, and E with the following properties:
    - Circle C has a radius of 5
    - Circle D is internally tangent to circle C at point A
    - Circle E is internally tangent to circle C, externally tangent to circle D, and tangent to diameter AB of circle C
    - The radius of circle D is four times the radius of circle E
    Then the radius of circle D is √5325 - 135/47 -/
theorem circle_tangency_radius : 
  ∀ (C D E : Circle) (A B : Point),
  C.radius = 5 ∧ 
  internally_tangent D C A ∧
  internally_tangent E C A ∧
  externally_tangent E D ∧
  tangent_to_diameter E C A B ∧
  D.radius = 4 * E.radius →
  D.radius = Real.sqrt 5325 - 135 / 47 := by sorry

/-- The sum of p and q in the expression √p - q for the radius of circle D is 5460 -/
theorem sum_p_q : 
  ∀ (p q : ℕ),
  (∃ (C D E : Circle) (A B : Point),
    C.radius = 5 ∧ 
    internally_tangent D C A ∧
    internally_tangent E C A ∧
    externally_tangent E D ∧
    tangent_to_diameter E C A B ∧
    D.radius = 4 * E.radius ∧
    D.radius = Real.sqrt (p : ℝ) - (q : ℝ)) →
  p + q = 5460 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangency_radius_sum_p_q_l8_894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_certain_exponent_is_four_l8_892

-- Define the function that calculates the number of digits after the decimal point
noncomputable def digitsAfterDecimal (n : ℝ) : ℕ := sorry

-- Define the expression
noncomputable def expression (x : ℝ) : ℝ := (10^x * 3.456789)^10

-- Theorem statement
theorem certain_exponent_is_four :
  ∃ x : ℝ, digitsAfterDecimal (expression x) = 20 ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_certain_exponent_is_four_l8_892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_increase_effect_l8_809

theorem price_increase_effect (original_price original_quantity : ℝ) 
  (price_increase_percent : ℝ) (quantity_decrease_percent : ℝ) : 
  price_increase_percent = 25 →
  quantity_decrease_percent = 28 →
  (original_price * (1 + price_increase_percent / 100) * (original_quantity * (1 - quantity_decrease_percent / 100)) - 
   original_price * original_quantity) / (original_price * original_quantity) = -0.1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_increase_effect_l8_809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l8_837

noncomputable section

-- Define the function f
def f (a θ : ℝ) : ℝ := Real.sin θ ^ 3 + 4 / (3 * a * Real.sin θ ^ 2 - a ^ 3)

-- State the theorem
theorem max_value_of_f :
  ∀ a θ : ℝ,
  0 < a → a < Real.sqrt 3 * Real.sin θ →
  θ ∈ Set.Icc (π / 6) (Real.arcsin ((3 : ℝ) ^ (1/3) / 2)) →
  f a θ ≤ 3 * Real.sqrt 3 :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l8_837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_probability_l8_868

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  hp : p > 0

/-- Point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line segment between two points -/
def LineSegment (A B : Point) : Type := Unit

/-- Quadrilateral defined by four points -/
def Quadrilateral (A B C D : Point) : Type := Unit

/-- Triangle defined by three points -/
def Triangle (A B C : Point) : Type := Unit

/-- Area of a geometric shape -/
noncomputable def Area {α : Type} (shape : α) : ℝ := sorry

/-- The main theorem -/
theorem parabola_probability (para : Parabola) 
  (F A B A' B' : Point) 
  (l : LineSegment A' B') 
  (quad : Quadrilateral A A' B' B) 
  (tri : Triangle F A' B')
  (h1 : A.y^2 = 2 * para.p * A.x)
  (h2 : B.y^2 = 2 * para.p * B.x)
  (h3 : F.x = para.p / 2 ∧ F.y = 0)
  (h4 : A'.x = -para.p / 2 ∧ A'.y = A.y)
  (h5 : B'.x = -para.p / 2 ∧ B'.y = B.y)
  (h6 : ∃ (k : ℝ), A.y = k * (A.x - F.x) ∧ B.y = k * (B.x - F.x))
  (h7 : ∃ (d : ℝ), d = ((A.x - B.x)^2 + (A.y - B.y)^2).sqrt ∧ d = 3 * para.p) :
  Area tri / Area quad = 1 / 6 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_probability_l8_868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_segment_l8_882

-- Define the points
def A : ℝ × ℝ := (-3, 0)
def B : ℝ × ℝ := (0, 2)
def C : ℝ × ℝ := (3, 0)
def D : ℝ × ℝ := (0, -1)

-- Define the angles
def angle_ADB : ℝ := 60
def angle_CDB : ℝ := 60
def angle_ABD : ℝ := 30
def angle_CBD : ℝ := 90

-- Define the segments
noncomputable def AD : ℝ := Real.sqrt ((A.1 - D.1)^2 + (A.2 - D.2)^2)
noncomputable def AB : ℝ := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
noncomputable def BD : ℝ := Real.sqrt ((B.1 - D.1)^2 + (B.2 - D.2)^2)
noncomputable def BC : ℝ := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
noncomputable def CD : ℝ := Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2)

theorem shortest_segment :
  AD < AB ∧ AD < BD ∧ AD < BC ∧ AD < CD := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_segment_l8_882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_mapping_triangles_l8_891

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle in 2D space -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Clockwise rotation of a point around a center by an angle in radians -/
noncomputable def rotatePoint (center : Point) (angle : ℝ) (p : Point) : Point :=
  { x := center.x + (p.x - center.x) * Real.cos angle + (p.y - center.y) * Real.sin angle,
    y := center.y - (p.x - center.x) * Real.sin angle + (p.y - center.y) * Real.cos angle }

/-- Clockwise rotation of a triangle around a center by an angle in radians -/
noncomputable def rotateTriangle (center : Point) (angle : ℝ) (t : Triangle) : Triangle :=
  { A := rotatePoint center angle t.A,
    B := rotatePoint center angle t.B,
    C := rotatePoint center angle t.C }

theorem rotation_mapping_triangles (n p q : ℝ) : 
  0 < n → n < π →
  let ABC : Triangle := ⟨⟨0, 0⟩, ⟨4, 0⟩, ⟨0, 3⟩⟩
  let DEF : Triangle := ⟨⟨8, 10⟩, ⟨8, 6⟩, ⟨11, 10⟩⟩
  let center : Point := ⟨p, q⟩
  rotateTriangle center n ABC = DEF →
  n * 180 / π + p + q = 102 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_mapping_triangles_l8_891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_triangle_area_theorem_l8_805

/-- Triangle with points dividing sides in given ratio -/
structure RatioTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  P : ℝ × ℝ
  Q : ℝ × ℝ
  R : ℝ × ℝ
  t : ℝ
  h1 : P = ((1 - t) * B.1 + t * C.1, (1 - t) * B.2 + t * C.2)
  h2 : Q = ((1 - t) * C.1 + t * A.1, (1 - t) * C.2 + t * A.2)
  h3 : R = ((1 - t) * A.1 + t * B.1, (1 - t) * A.2 + t * B.2)

/-- Area of a triangle given its vertices -/
def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  0.5 * abs ((p2.1 - p1.1) * (p3.2 - p1.2) - (p3.1 - p1.1) * (p2.2 - p1.2))

/-- Theorem: The ratio of areas in a RatioTriangle -/
theorem ratio_triangle_area_theorem (rt : RatioTriangle) :
  let K := triangleArea rt.A rt.P rt.Q + triangleArea rt.B rt.Q rt.R + triangleArea rt.C rt.R rt.P
  let L := triangleArea rt.A rt.B rt.C
  K / L = rt.t^2 - rt.t + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_triangle_area_theorem_l8_805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mushroom_collection_exists_l8_871

/-- Represents the number of mushrooms collected by Petya and Vasya -/
structure MushroomCollection where
  petya : ℕ
  vasya : ℕ
  total_is_valid : petya + vasya = 25 ∨ petya + vasya = 300 ∨ petya + vasya = 525 ∨ petya + vasya = 1900 ∨ petya + vasya = 9900
  vasya_is_odd : Odd vasya
  petya_percentage : petya * (petya + vasya) = vasya * 100

/-- There exists a valid mushroom collection satisfying the problem conditions -/
theorem mushroom_collection_exists : ∃ (mc : MushroomCollection), True := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mushroom_collection_exists_l8_871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_S_8_value_l8_870

-- Define the sequence a_n
def a : ℕ → ℝ := sorry

-- Define the sum S_n
def S : ℕ → ℝ := sorry

-- Condition: S_{n+1} = S_n + a_n + 3
axiom S_recurrence (n : ℕ) : S (n + 1) = S n + a n + 3

-- Condition: a_4 + a_5 = 23
axiom a_sum : a 4 + a 5 = 23

-- Theorem to prove
theorem S_8_value : S 8 = 92 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_S_8_value_l8_870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_simplification_l8_887

open Real

theorem trigonometric_simplification :
  (∀ x : ℝ, sin (x - 2 * π) = sin x) ∧
  (∀ x : ℝ, cos (x + π) = -cos x) ∧
  (∀ x : ℝ, sin (x + 5 * π / 2) = cos x) →
  (sin (π / 6 - 2 * π) * cos (π / 4 + π) = -sqrt 2 / 4) ∧
  (sin (π / 4 + 5 * π / 2) = sqrt 2 / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_simplification_l8_887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_circles_l8_863

-- Define the two circles
def circle1 (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 1
def circle2 (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := 
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- Theorem statement
theorem min_distance_between_circles :
  ∃ (min_dist : ℝ), 
    (∀ (x1 y1 x2 y2 : ℝ), 
      circle1 x1 y1 → circle2 x2 y2 → 
      distance x1 y1 x2 y2 ≥ min_dist) ∧ 
    (∃ (x1 y1 x2 y2 : ℝ), 
      circle1 x1 y1 ∧ circle2 x2 y2 ∧ 
      distance x1 y1 x2 y2 = min_dist) ∧
    min_dist = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_circles_l8_863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_device_b_produces_1800_l8_811

/-- The total production of both devices -/
def total_production : ℕ := 4800

/-- The number of units produced by Device A -/
def device_a_production : ℕ → ℕ := λ x ↦ x

/-- The number of units produced by Device B -/
def device_b_production : ℕ → ℕ := λ x ↦ total_production - x

/-- The number of samples from Device A -/
def sample_a : ℕ := 50

/-- The total number of samples -/
def total_samples : ℕ := 80

/-- The proportion of samples matches the proportion in total production -/
def proportion_matches (x : ℕ) : Prop :=
  (sample_a : ℚ) / total_samples = (device_a_production x : ℚ) / total_production

theorem device_b_produces_1800 :
  ∃ x : ℕ, device_a_production x + device_b_production x = total_production ∧
           proportion_matches x ∧
           device_b_production x = 1800 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_device_b_produces_1800_l8_811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_table_filling_iff_even_l8_893

/-- A valid table filling for an n×n grid -/
def ValidTableFilling (n : ℕ) : Type :=
  Fin n → Fin n → Fin (n^2)

/-- Two cells share a side if they differ by 1 in exactly one coordinate -/
def SharesSide {n : ℕ} (c1 c2 : Fin n × Fin n) : Prop :=
  (c1.1 = c2.1 ∧ (c1.2.val + 1 = c2.2.val ∨ c2.2.val + 1 = c1.2.val)) ∨
  (c1.2 = c2.2 ∧ (c1.1.val + 1 = c2.1.val ∨ c2.1.val + 1 = c1.1.val))

/-- Theorem: A valid table filling exists if and only if n is even -/
theorem valid_table_filling_iff_even (n : ℕ) (h : n > 1) :
  (∃ f : ValidTableFilling n,
    (∀ i j : Fin n, ∀ k : Fin (n^2), f i j = k → 
      (∀ i' j' : Fin n, f i' j' = k.succ → SharesSide (i, j) (i', j'))) ∧
    (∀ i j i' j' : Fin n, f i j % n = f i' j' % n → 
      (i ≠ i' ∧ j ≠ j'))) ↔ 
  Even n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_table_filling_iff_even_l8_893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_c_time_to_finish_l8_855

/-- The time taken by worker c to finish the job alone -/
noncomputable def time_c : ℝ := 3.75

/-- The combined rate of workers a and b -/
noncomputable def rate_ab : ℝ := 1 / 15

/-- The combined rate of workers a, b, and c -/
noncomputable def rate_abc : ℝ := 1 / 3

/-- Theorem stating that given the conditions, c takes 3.75 days to finish the job alone -/
theorem c_time_to_finish (h1 : rate_ab = 1 / 15) (h2 : rate_abc = 1 / 3) : 
  time_c = 3.75 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_c_time_to_finish_l8_855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_A_members_l8_881

theorem set_A_members (U A B : Finset Nat) : 
  (U.card = 193) →
  (B.card = 49) →
  ((U \ (A ∪ B)).card = 59) →
  ((A ∩ B).card = 25) →
  (A ⊆ U) →
  (B ⊆ U) →
  (A.card = 110) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_A_members_l8_881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_field_run_time_l8_849

/-- The time taken to run around a square field -/
noncomputable def run_time (side_length : ℝ) (speed_kmh : ℝ) : ℝ :=
  let perimeter := 4 * side_length
  let speed_ms := speed_kmh * (1000 / 3600)
  perimeter / speed_ms

/-- Proof that running around a square field of side 35 meters at 9 km/h takes 56 seconds -/
theorem square_field_run_time :
  run_time 35 9 = 56 := by
  -- Unfold the definition of run_time
  unfold run_time
  -- Simplify the expression
  simp
  -- The exact proof would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_field_run_time_l8_849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_sqrt_30_squared_l8_829

theorem floor_sqrt_30_squared : ⌊Real.sqrt 30⌋^2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_sqrt_30_squared_l8_829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_prime_factors_l8_865

theorem max_prime_factors (a b c : ℕ+) 
  (h_gcd : (Nat.gcd (Nat.gcd a.val b.val) c.val).factors.length = 10)
  (h_lcm : (Nat.lcm a.val b.val).factors.length = 40)
  (h_fewer : a.val.factors.length < min b.val.factors.length c.val.factors.length) :
  a.val.factors.length ≤ 24 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_prime_factors_l8_865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_organization_growth_after_four_years_l8_819

/-- Calculates the number of members in an organization after a given number of years -/
def organizationGrowth (initialTotal : ℕ) (initialLeaders : ℕ) (years : ℕ) : ℕ :=
  let initialRegular := initialTotal - initialLeaders
  let regularGrowth := fun n => 4 * n
  let rec growthAfterYears : ℕ → ℕ
    | 0 => initialTotal
    | n + 1 => regularGrowth (growthAfterYears n - initialLeaders) + initialLeaders
  growthAfterYears years

/-- Theorem stating the number of members after 4 years given initial conditions -/
theorem organization_growth_after_four_years :
  organizationGrowth 20 6 4 = 3590 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_organization_growth_after_four_years_l8_819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_intersection_l8_807

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log (x + 3) / Real.log 2

-- Define the domain of f
def M : Set ℝ := {x | x > -3}

-- Define the domain of g (we don't know what g is, so we'll leave it as a variable)
variable (N : Set ℝ)

-- State the theorem
theorem domain_intersection :
  M ∩ N = {x : ℝ | x > -3} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_intersection_l8_807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l8_826

-- Define the hyperbola
def hyperbola (m : ℝ) (x y : ℝ) : Prop := m * x^2 - y^2 = 1

-- Define the focal distance and conjugate axis relationship
def focal_conjugate_relation (c b : ℝ) : Prop := c = 3 * b

-- Define the asymptote equation
def asymptote (k : ℝ) (x y : ℝ) : Prop := y = k * x ∨ y = -k * x

-- Theorem statement
theorem hyperbola_asymptotes 
  (m : ℝ) (h : m > 0) : 
  ∃ (b c : ℝ), 
    (∀ x y, hyperbola m x y → 
      focal_conjugate_relation c b → 
      asymptote (Real.sqrt 2 / 4) x y) :=
by
  sorry

#check hyperbola_asymptotes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l8_826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_special_configuration_l8_827

/-- A type representing a point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- A configuration of 12 points on a plane -/
def Configuration := Fin 12 → Point

/-- The property that the distance between any two points in the configuration does not exceed 3 -/
def ValidConfiguration (config : Configuration) : Prop :=
  ∀ i j, distance (config i) (config j) ≤ 3

/-- The property that it's impossible to select 4 points with pairwise distances not exceeding 2 -/
def NoFourPointsWithin2 (config : Configuration) : Prop :=
  ¬∃ (a b c d : Fin 12), (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧
    (distance (config a) (config b) ≤ 2 ∧
     distance (config a) (config c) ≤ 2 ∧
     distance (config a) (config d) ≤ 2 ∧
     distance (config b) (config c) ≤ 2 ∧
     distance (config b) (config d) ≤ 2 ∧
     distance (config c) (config d) ≤ 2)

theorem existence_of_special_configuration :
  ∃ (config : Configuration), ValidConfiguration config ∧ NoFourPointsWithin2 config := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_special_configuration_l8_827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_player_can_achieve_goal_l8_877

/-- Represents the state of the player's coins -/
structure CoinState where
  gold : ℕ
  silver : ℕ

/-- Represents a bet placement -/
structure Bet where
  red_gold : ℕ
  red_silver : ℕ
  black_gold : ℕ
  black_silver : ℕ

/-- Represents the possible outcomes of a round -/
inductive Outcome
| Red
| Black

/-- Function to update the coin state after a round -/
def updateState (state : CoinState) (bet : Bet) (outcome : Outcome) : CoinState :=
  match outcome with
  | Outcome.Red => 
    { gold := state.gold + bet.red_gold - bet.black_gold,
      silver := state.silver + bet.red_silver - bet.black_silver }
  | Outcome.Black => 
    { gold := state.gold + bet.black_gold - bet.red_gold,
      silver := state.silver + bet.black_silver - bet.red_silver }

/-- Predicate to check if the goal state is achieved -/
def isGoalState (state : CoinState) : Prop :=
  state.gold = 0 ∧ state.silver = 0 ∨ 
  state.gold = 3 * state.silver ∨ 
  state.silver = 3 * state.gold

/-- Main theorem statement -/
theorem player_can_achieve_goal (m n : ℕ) : 
  (m < 3 * n ∧ n < 3 * m) → 
  ∃ (strategy : CoinState → Bet), 
    ∀ (outcomes : List Outcome), 
      let final_state := outcomes.foldl 
        (fun state outcome ↦ updateState state (strategy state) outcome) 
        { gold := m, silver := n }
      isGoalState final_state := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_player_can_achieve_goal_l8_877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l8_834

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := (x + b) / (x^2 + a)

theorem odd_function_properties (a b : ℝ) :
  (∀ x, f a b x = -f a b (-x)) →  -- f is odd
  (∀ x, f a b x ≠ 0 → x^2 + a ≠ 0) →  -- domain is ℝ
  f a b 1 = 1/2 →
  (a = 1 ∧ b = 0) ∧
  (∀ x y, -1 < x ∧ x < y ∧ y < 1 → f a b x < f a b y) ∧
  (∀ t, -1 < t ∧ t < 1 ∧ f a b t + f a b (t-1) < 0 ↔ 0 < t ∧ t < 1/2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l8_834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_opposite_sides_not_imply_parallelogram_l8_866

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A quadrilateral in 3D space -/
structure Quadrilateral3D where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

/-- Distance between two points in 3D space -/
noncomputable def distance (p q : Point3D) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2 + (p.z - q.z)^2)

/-- Predicate for a quadrilateral with two pairs of opposite sides equal in length -/
def has_equal_opposite_sides (q : Quadrilateral3D) : Prop :=
  distance q.A q.B = distance q.C q.D ∧ distance q.A q.D = distance q.B q.C

/-- Predicate for a parallelogram in 3D space -/
def is_parallelogram (q : Quadrilateral3D) : Prop :=
  ∃ (v w : Point3D),
    q.B.x - q.A.x = v.x ∧ q.B.y - q.A.y = v.y ∧ q.B.z - q.A.z = v.z ∧
    q.C.x - q.D.x = v.x ∧ q.C.y - q.D.y = v.y ∧ q.C.z - q.D.z = v.z ∧
    q.D.x - q.A.x = w.x ∧ q.D.y - q.A.y = w.y ∧ q.D.z - q.A.z = w.z ∧
    q.C.x - q.B.x = w.x ∧ q.C.y - q.B.y = w.y ∧ q.C.z - q.B.z = w.z

theorem equal_opposite_sides_not_imply_parallelogram :
  ∃ (q : Quadrilateral3D), has_equal_opposite_sides q ∧ ¬is_parallelogram q := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_opposite_sides_not_imply_parallelogram_l8_866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vasya_missed_lessons_not_64_l8_841

/-- Represents the days of the week -/
inductive Weekday
  | monday
  | tuesday
  | wednesday
  | thursday
  | friday
  | saturday
  | sunday

/-- Returns the number of lessons Vasya misses on a given weekday -/
def missedLessons (day : Weekday) : Nat :=
  match day with
  | Weekday.monday => 1
  | Weekday.tuesday => 2
  | Weekday.wednesday => 3
  | Weekday.thursday => 4
  | Weekday.friday => 5
  | Weekday.saturday => 0
  | Weekday.sunday => 0

/-- Returns true if the given weekday is a school day -/
def isSchoolDay (day : Weekday) : Bool :=
  match day with
  | Weekday.saturday => false
  | Weekday.sunday => false
  | _ => true

/-- Represents the structure of September -/
structure September where
  days : Nat
  weekdays : List Weekday
  school_days : Nat

/-- The specific structure of September in the problem -/
def septemberStructure : September :=
  { days := 30
  , weekdays := [Weekday.monday, Weekday.tuesday, Weekday.wednesday, Weekday.thursday, Weekday.friday, Weekday.saturday, Weekday.sunday]
  , school_days := 22 }

/-- Theorem stating that Vasya cannot miss exactly 64 lessons in September -/
theorem vasya_missed_lessons_not_64 (sept : September := septemberStructure) :
  ¬ (List.sum (List.map missedLessons (List.filter isSchoolDay sept.weekdays))) = 64 := by
  sorry

#eval List.sum (List.map missedLessons (List.filter isSchoolDay septemberStructure.weekdays))

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vasya_missed_lessons_not_64_l8_841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_perimeter_of_divided_triangle_l8_873

/-- Represents an isosceles triangle with given base and height -/
structure IsoscelesTriangle where
  base : ℝ
  height : ℝ

/-- Calculates the perimeter of a piece of the divided triangle -/
noncomputable def piece_perimeter (triangle : IsoscelesTriangle) (k : ℕ) : ℝ :=
  2 + Real.sqrt (triangle.height^2 + (k * (triangle.base / 5))^2) +
      Real.sqrt (triangle.height^2 + ((k + 1) * (triangle.base / 5))^2)

/-- The main theorem statement -/
theorem max_perimeter_of_divided_triangle (triangle : IsoscelesTriangle)
    (h_base : triangle.base = 10)
    (h_height : triangle.height = 15) :
    ∃ (max_perimeter : ℝ),
      max_perimeter = piece_perimeter triangle 4 ∧
      max_perimeter = 2 + Real.sqrt 289 + Real.sqrt 325 ∧
      ∀ k, k < 5 → piece_perimeter triangle k ≤ max_perimeter :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_perimeter_of_divided_triangle_l8_873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_table_tennis_tournament_l8_856

def number_of_matches_among_withdrawn_players (n : ℕ) : ℕ :=
  6 - ((n - 3) * (n - 4) / 2 + 6 - 50)

theorem table_tennis_tournament (n : ℕ) : 
  n > 3 →
  (n - 3) * (n - 4) / 2 + 6 - 1 = 50 →
  ∃ (r : ℕ), r = 1 ∧ r = number_of_matches_among_withdrawn_players n :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_table_tennis_tournament_l8_856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correlation_analysis_l8_828

-- Define the types for our variables
structure Student where
  mk ::

structure Teacher where
  mk ::

structure Family where
  mk ::

-- Define the attributes
noncomputable def learning_attitude (s : Student) : ℝ := sorry
noncomputable def academic_performance (s : Student) : ℝ := sorry
noncomputable def teaching_level (t : Teacher) : ℝ := sorry
noncomputable def student_height (s : Student) : ℝ := sorry
noncomputable def economic_condition (f : Family) : ℝ := sorry

-- Define correlation
def correlated {α : Type*} (f g : α → ℝ) : Prop := sorry

-- State the theorem
theorem correlation_analysis :
  (∃ (s : Student), correlated (learning_attitude) (academic_performance)) ∧
  (∃ (t : Teacher), correlated (fun s => teaching_level t) academic_performance) ∧
  (∀ (s : Student), ¬ correlated (student_height) (academic_performance)) ∧
  (∀ (f : Family), ¬ correlated (fun s => economic_condition f) academic_performance) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correlation_analysis_l8_828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_to_hundredth_3_8963_l8_810

/-- Rounds a real number to the nearest hundredth -/
noncomputable def roundToHundredth (x : ℝ) : ℝ :=
  ⌊x * 100 + 0.5⌋ / 100

/-- The given number to be rounded -/
def givenNumber : ℝ := 3.8963

/-- Theorem stating that rounding the given number to the nearest hundredth results in 3.90 -/
theorem round_to_hundredth_3_8963 :
  roundToHundredth givenNumber = 3.90 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_to_hundredth_3_8963_l8_810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_sum_property_l8_848

theorem subset_sum_property (n : ℕ) (h : n > 16) :
  ∃ (S : Finset ℕ), (Finset.card S = n) ∧
    (∀ (A : Finset ℕ), A ⊆ S →
      (∀ a a' : ℕ, a ∈ A → a' ∈ A → a ≠ a' → a + a' ∉ S) →
        Finset.card A ≤ 4 * Real.sqrt (n : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_sum_property_l8_848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_side_b_value_l8_846

-- Define an acute-angled triangle
structure AcuteTriangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  acute_A : 0 < A ∧ A < Real.pi / 2
  acute_B : 0 < B ∧ B < Real.pi / 2
  acute_C : 0 < C ∧ C < Real.pi / 2
  angle_sum : A + B + C = Real.pi
  side_a : a > 0
  side_b : b > 0
  side_c : c > 0
  sine_law : a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C

-- State the theorem
theorem side_b_value (t : AcuteTriangle) 
  (h : Real.cos t.B / t.b + Real.cos t.C / t.c = Real.sin t.A / (Real.sqrt 3 * Real.sin t.C)) :
  t.b = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_side_b_value_l8_846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_disks_theorem_l8_859

/-- The number of labels -/
def n : ℕ := 60

/-- The minimum number of disks with the same label we want to guarantee -/
def k : ℕ := 15

/-- The total number of disks -/
def total_disks : ℕ := n * (n + 1) / 2

/-- The number of disks for each label i -/
def disks_per_label (i : ℕ) : ℕ := i

/-- The minimum number of disks that must be drawn to guarantee at least k disks with the same label -/
def min_disks_to_draw : ℕ := (k - 1) * (n - k + 1) + (k - 1) * k / 2 + 1

theorem min_disks_theorem :
  min_disks_to_draw = 750 ∧
  ∀ m : ℕ, m < min_disks_to_draw →
    ∃ f : Fin m → Fin n,
      ∀ i : Fin n, (Finset.filter (λ j : Fin m ↦ f j = i) Finset.univ).card < k :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_disks_theorem_l8_859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l8_813

noncomputable def f (x : ℝ) : ℝ := Real.sin (13 * Real.pi / 2 - x)

theorem f_properties :
  (∀ x, f x = f (-x)) ∧ 
  (∃ T > 0, ∀ x, f (x + T) = f x) ∧
  (∀ T > 0, (∀ x, f (x + T) = f x) → T ≥ 2 * Real.pi) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l8_813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marathon_time_is_three_hours_l8_844

/-- Represents the marathon race with given conditions -/
structure Marathon where
  total_distance : ℝ
  initial_distance : ℝ
  initial_time : ℝ
  pace_reduction : ℝ

/-- Calculates the total time to complete the marathon -/
noncomputable def marathon_time (m : Marathon) : ℝ :=
  let remaining_distance := m.total_distance - m.initial_distance
  let initial_pace := m.initial_distance / m.initial_time
  let reduced_pace := initial_pace * m.pace_reduction
  let remaining_time := remaining_distance / reduced_pace
  m.initial_time + remaining_time

/-- Theorem stating that the marathon time is 3 hours given the specified conditions -/
theorem marathon_time_is_three_hours :
  let m : Marathon := {
    total_distance := 26,
    initial_distance := 10,
    initial_time := 1,
    pace_reduction := 0.8
  }
  marathon_time m = 3 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_marathon_time_is_three_hours_l8_844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carmen_pet_difference_l8_876

/-- Represents the number of pets Carmen has -/
structure PetCount where
  cats : ℤ
  dogs : ℤ
  rabbits : ℤ
  parrots : ℤ

/-- Represents the changes in pet count for each round -/
structure PetChange where
  cats : ℤ
  dogs : ℤ
  rabbits : ℤ
  parrots : ℤ

def initial_pets : PetCount := {
  cats := 48,
  dogs := 36,
  rabbits := 10,
  parrots := 5
}

def round1 : PetChange := {
  cats := -6,
  dogs := 0,
  rabbits := -2,
  parrots := 0
}

def round2 : PetChange := {
  cats := -12,
  dogs := -8,
  rabbits := 0,
  parrots := -2
}

def round3 : PetChange := {
  cats := -8,
  dogs := 0,
  rabbits := -4,
  parrots := -1
}

def round4 : PetChange := {
  cats := 0,
  dogs := -5,
  rabbits := -2,
  parrots := 0
}

def apply_change (pets : PetCount) (change : PetChange) : PetCount := {
  cats := pets.cats + change.cats,
  dogs := pets.dogs + change.dogs,
  rabbits := pets.rabbits + change.rabbits,
  parrots := pets.parrots + change.parrots
}

def final_pets : PetCount :=
  apply_change (apply_change (apply_change (apply_change initial_pets round1) round2) round3) round4

theorem carmen_pet_difference :
  final_pets.cats - final_pets.dogs = -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_carmen_pet_difference_l8_876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_iff_a_in_set_l8_801

/-- The equation has no solution if and only if a is in the given set -/
theorem no_solution_iff_a_in_set (a : ℝ) :
  (∀ x : ℝ, 6 * |x - 4*a| + |x - a^2| + 5*x - 3*a ≠ 0) ↔ 
  (a < -13 ∨ a > 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_iff_a_in_set_l8_801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_sqrt5_l8_888

/-- A hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt ((h.a ^ 2 + h.b ^ 2) / h.a ^ 2)

/-- Theorem: If a hyperbola has an asymptote with equation 2x + y = 0, 
    then its eccentricity is √5 -/
theorem hyperbola_eccentricity_sqrt5 (h : Hyperbola) 
    (h_asymptote : h.b / h.a = 2) : eccentricity h = Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_sqrt5_l8_888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oil_truck_tank_height_l8_884

/-- Represents a right circular cylinder -/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- Calculates the volume of a cylinder -/
noncomputable def cylinderVolume (c : Cylinder) : ℝ := Real.pi * c.radius^2 * c.height

theorem oil_truck_tank_height :
  let stationaryTank : Cylinder := { radius := 100, height := 25 }
  let oilLevelDrop : ℝ := 0.03
  let truckTankRadius : ℝ := 5
  let pumpedVolume : ℝ := cylinderVolume { radius := stationaryTank.radius, height := oilLevelDrop }
  let truckTankHeight : ℝ := pumpedVolume / (Real.pi * truckTankRadius^2)
  truckTankHeight = 12 := by
    -- The proof goes here
    sorry

#eval "Oil truck tank height theorem defined."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oil_truck_tank_height_l8_884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_in_fourth_quadrant_l8_872

-- Define the point P
noncomputable def P : ℝ × ℝ := (Real.sin 2, Real.cos 2)

-- Define the fourth quadrant
def fourth_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 < 0

-- Theorem statement
theorem point_in_fourth_quadrant : fourth_quadrant P := by
  -- Unfold the definition of fourth_quadrant and P
  unfold fourth_quadrant P
  
  -- Split the conjunction into two goals
  apply And.intro
  
  -- Prove sin 2 > 0
  · sorry
  
  -- Prove cos 2 < 0
  · sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_in_fourth_quadrant_l8_872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exact_100_numbers_between_l8_812

theorem exact_100_numbers_between (n : ℕ) : 
  (∃ k : ℕ, k > 0 ∧ 
    (∀ m : ℕ, (n : ℝ) / 2 < m ∧ m < (3 * n : ℝ) / 5 ↔ k ≤ m ∧ m < k + 100)) ↔ 
  (997 ≤ n ∧ n ≤ 1010) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exact_100_numbers_between_l8_812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_distance_l8_878

/-- The parabola equation -/
def parabola (x : ℝ) : ℝ := x^2 + 3*x - 4

/-- The y-coordinate of point P -/
def y_P : ℝ := 3

/-- The y-coordinate of point Q -/
def y_Q : ℝ := -4

/-- The set of x-coordinates for point P -/
def x_P : Set ℝ := {x | parabola x = y_P}

/-- The set of x-coordinates for point Q -/
def x_Q : Set ℝ := {x | parabola x = y_Q}

/-- The horizontal distance between two points -/
def horizontal_distance (x1 x2 : ℝ) : ℝ := |x1 - x2|

theorem particle_distance :
  ∃ (p q : ℝ), p ∈ x_P ∧ q ∈ x_Q ∧
    (∀ (p' q' : ℝ), p' ∈ x_P → q' ∈ x_Q → horizontal_distance p q ≤ horizontal_distance p' q') ∧
    horizontal_distance p q = 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_distance_l8_878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_chord_length_l8_867

-- Define the circle
def my_circle (x y : ℝ) : Prop := x^2 + y^2 = 16

-- Define the line
def my_line (x y a b : ℝ) : Prop := (a - b) * x + (3 * b - 2 * a) * y - a = 0

-- Define the chord length
noncomputable def chord_length (x₁ y₁ x₂ y₂ : ℝ) : ℝ := Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

-- Theorem statement
theorem min_chord_length :
  ∀ a b : ℝ, (a ≠ 0 ∨ b ≠ 0) →
  ∃ min_length : ℝ,
    (∀ x₁ y₁ x₂ y₂ : ℝ,
      my_circle x₁ y₁ ∧ my_circle x₂ y₂ ∧
      my_line x₁ y₁ a b ∧ my_line x₂ y₂ a b →
      chord_length x₁ y₁ x₂ y₂ ≥ min_length) ∧
    min_length = 2 * Real.sqrt 6 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_chord_length_l8_867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_eq_one_sufficient_not_necessary_l8_852

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

/-- The slope of the line ax + (2-a)y + 3 = 0 -/
noncomputable def slope1 (a : ℝ) : ℝ := -a

/-- The slope of the line x - ay - 2 = 0 -/
noncomputable def slope2 (a : ℝ) : ℝ := 1 / a

/-- The condition a = 1 is sufficient but not necessary for perpendicularity -/
theorem a_eq_one_sufficient_not_necessary :
  (∀ a : ℝ, a ≠ 0 → perpendicular (slope1 a) (slope2 a)) ∧
  (∃ a : ℝ, a ≠ 1 ∧ perpendicular (slope1 a) (slope2 a)) ∧
  perpendicular (slope1 1) (slope2 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_eq_one_sufficient_not_necessary_l8_852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_next_larger_perfect_square_l8_817

theorem next_larger_perfect_square (x : ℕ) (h : ∃ k : ℕ, x = k^2) :
  ∃ n : ℕ, n > x ∧ (∃ m : ℕ, m > n ∧ m = x + 4 * Nat.sqrt x + 4 ∧ ∃ l : ℕ, m = l^2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_next_larger_perfect_square_l8_817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l8_847

-- Define the function f
def f (x a : ℝ) : ℝ := |x - 4| + |x - a|

-- State the theorem
theorem function_properties (a : ℝ) (h1 : a > 1) :
  (∃ (x : ℝ), ∀ (y : ℝ), f x a ≤ f y a) ∧ 
  (∃ (m : ℝ), m = 3 ∧ ∀ (x : ℝ), f x a ≥ m) →
  a = 7 ∧ 
  {x : ℝ | f x a ≤ 5} = Set.Icc 3 8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l8_847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_pass_bridge_time_l8_830

/-- The time (in seconds) it takes for a train to pass a bridge -/
noncomputable def train_pass_time (train_length bridge_length : ℝ) (train_speed_kmh : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  total_distance / train_speed_ms

/-- Theorem: A train of length 560 meters, traveling at 45 km/hour, 
    takes 56 seconds to pass a bridge of length 140 meters -/
theorem train_pass_bridge_time :
  train_pass_time 560 140 45 = 56 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_pass_bridge_time_l8_830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_expression_value_l8_836

def is_valid_assignment (a b c d : ℕ) : Prop :=
  Finset.toSet {a, b, c, d} = Finset.toSet {1, 2, 3, 4}

def expression_value (a b c d : ℕ) : ℤ :=
  (c : ℤ) * (a ^ b) - d

theorem max_expression_value :
  ∃ (a b c d : ℕ), is_valid_assignment a b c d ∧ 
    (∀ (a' b' c' d' : ℕ), is_valid_assignment a' b' c' d' →
      expression_value a b c d ≥ expression_value a' b' c' d') ∧
    expression_value a b c d = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_expression_value_l8_836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scaling_transformation_correct_l8_861

/-- The scaling transformation φ --/
noncomputable def φ (x y : ℝ) : ℝ × ℝ := (x / 2, y * Real.sqrt 3 / 3)

/-- The initial equation --/
def initial_eq (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

/-- The target equation --/
def target_eq (x y : ℝ) : Prop := x^2 + y^2 = 1

theorem scaling_transformation_correct :
  ∀ x y : ℝ, initial_eq x y ↔ target_eq (φ x y).1 (φ x y).2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_scaling_transformation_correct_l8_861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l8_899

noncomputable def f (x : ℝ) : ℝ := 
  if x > 0 ∧ x ≤ 2 then x^2 - 2*x + 2
  else if x < 0 ∧ x ≥ -2 then -((-x)^2 - 2*(-x) + 2)
  else 0

theorem min_value_of_f :
  (∀ x ∈ Set.Icc (-2) 2, f (-x) = -f x) →
  (∀ x ∈ Set.Ioo 0 2, f x = x^2 - 2*x + 2) →
  ∃ x ∈ Set.Icc (-2) 2, ∀ y ∈ Set.Icc (-2) 2, f x ≤ f y ∧ f x = -2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l8_899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_is_20_l8_850

-- Define the length of the platform
noncomputable def platform_length : ℝ := 250

-- Define the time taken to cross the platform
noncomputable def crossing_time : ℝ := 26

-- Define the length of the train
noncomputable def train_length : ℝ := 270

-- Define the speed of the train
noncomputable def train_speed : ℝ := (platform_length + train_length) / crossing_time

-- Theorem statement
theorem train_speed_is_20 : train_speed = 20 := by
  -- Unfold the definition of train_speed
  unfold train_speed
  -- Simplify the expression
  simp [platform_length, crossing_time, train_length]
  -- Perform the arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_is_20_l8_850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_increasing_on_interval_l8_820

-- Define the function f(x) = (x-3)e^x
noncomputable def f (x : ℝ) : ℝ := (x - 3) * Real.exp x

-- State the theorem
theorem f_strictly_increasing_on_interval :
  StrictMonoOn f (Set.Ioi 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_increasing_on_interval_l8_820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l8_889

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 - 2*x else 1/x

theorem solution_set (a : ℝ) : 
  (f 1 + f a = -2) ↔ (a = -1 ∨ a = 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l8_889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l8_823

theorem trig_identity (α β : ℝ) (h : Real.sin α ≠ 0) :
  2 * Real.cos (α - β) - Real.sin (2 * α - β) / Real.sin α = Real.sin β / Real.sin α := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l8_823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l8_806

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 
  if x ≤ 0 then x^2 + x else -((-x)^2 + (-x))

-- State the theorem
theorem tangent_line_at_one (h_odd : ∀ x, f (-x) = -f x) :
  ∃ a b c : ℝ, a = 1 ∧ b = 1 ∧ c = -1 ∧
  ∀ x y : ℝ, y = f x → (x = 1 → a * x + b * y + c = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l8_806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_towel_problem_l8_814

def towel_problem (unknown_rate : ℚ) : Prop :=
  let towels_100 := 3
  let towels_150 := 5
  let towels_unknown := 2
  let price_100 := 100
  let price_150 := 150
  let total_towels := towels_100 + towels_150 + towels_unknown
  let total_cost := towels_100 * price_100 + towels_150 * price_150 + towels_unknown * unknown_rate
  let average_price := 145
  total_cost / total_towels = average_price

theorem solve_towel_problem :
  ∃ (unknown_rate : ℚ), towel_problem unknown_rate ∧ unknown_rate = 200 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_towel_problem_l8_814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_minus_sector_area_l8_885

noncomputable section

def P : ℝ × ℝ := (1, 2)
def Q : ℝ × ℝ := (7, 6)
def central_angle : ℝ := Real.pi / 3

-- Distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- Radius of the circle
noncomputable def radius : ℝ := distance P Q

-- Area of the circle
noncomputable def circle_area : ℝ := Real.pi * radius^2

-- Area of the sector
noncomputable def sector_area : ℝ := (1/2) * radius^2 * central_angle

-- Theorem: The area of the circle minus the sector is 130π/3
theorem circle_minus_sector_area :
  circle_area - sector_area = (130 * Real.pi) / 3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_minus_sector_area_l8_885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frequency_die_roll_example_l8_879

/-- The frequency of an event in a series of trials -/
def frequency (event_occurrences : ℕ) (total_trials : ℕ) : ℚ :=
  event_occurrences / total_trials

/-- Rolling a die 100 times and getting 6 for 20 times -/
def die_roll_example : Prop :=
  ∃ (rolls : Fin 100 → ℕ),
    (∀ i, rolls i ≤ 6) ∧
    ((Finset.filter (λ i => rolls i = 6) Finset.univ).card = 20)

theorem frequency_die_roll_example :
  die_roll_example →
  frequency 20 100 = 1/5 := by
  intro h
  unfold frequency
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_frequency_die_roll_example_l8_879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_existence_condition_l8_816

/-- A quadrilateral with given side lengths and midpoint segment length -/
structure Quadrilateral (a b c d f : ℝ) where
  sides_positive : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d
  midpoint_positive : 0 < f
  midpoint_length : f = (1/2) * Real.sqrt ((a^2 + c^2) / 2 + b^2 + d^2 - (a * c + b * d) / 2)

/-- The existence of a quadrilateral with given side lengths implies a relation between those lengths -/
theorem quadrilateral_existence_condition {a b c d f : ℝ} (h : Quadrilateral a b c d f) :
  b + d ≥ 2 * f := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_existence_condition_l8_816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_product_l8_896

-- Define the curves C₁ and C₂
noncomputable def C₁ (t : ℝ) : ℝ × ℝ := (2 - Real.sqrt 2 * t, -1 + Real.sqrt 2 * t)

noncomputable def C₂ (θ : ℝ) : ℝ × ℝ := 
  let ρ := 2 / Real.sqrt (1 + 3 * Real.sin θ ^ 2)
  (ρ * Real.cos θ, ρ * Real.sin θ)

-- Define point M
def M : ℝ × ℝ := (2, -1)

-- Define the intersection points A and B (existence assumed)
axiom A : ℝ × ℝ
axiom B : ℝ × ℝ

-- Assume A and B are on both curves
axiom A_on_C₁ : ∃ t₁, C₁ t₁ = A
axiom A_on_C₂ : ∃ θ₁, C₂ θ₁ = A
axiom B_on_C₁ : ∃ t₂, C₁ t₂ = B
axiom B_on_C₂ : ∃ θ₂, C₂ θ₂ = B

-- The theorem to prove
theorem intersection_product :
  let MA := Real.sqrt ((A.1 - M.1)^2 + (A.2 - M.2)^2)
  let MB := Real.sqrt ((B.1 - M.1)^2 + (B.2 - M.2)^2)
  MA * MB = 8/5 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_product_l8_896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_solution_l8_824

theorem smallest_angle_solution (x : ℝ) : 
  (∀ y : ℝ, y > 0 ∧ y < x → ¬(Real.sin (4 * y) * Real.sin (6 * y) = Real.cos (4 * y) * Real.cos (6 * y))) ∧
  (Real.sin (4 * x) * Real.sin (6 * x) = Real.cos (4 * x) * Real.cos (6 * x)) ∧
  (x > 0) →
  x = 9 * π / 180 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_solution_l8_824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_different_radii_symmetry_concentric_circles_symmetry_same_radii_different_centers_symmetry_l8_835

-- Define a circle in a plane
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a system of two circles in a plane
structure TwoCircleSystem where
  circle1 : Circle
  circle2 : Circle

-- Define a line in a plane
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Function to check if a line is a symmetry line for a two-circle system
def isSymmetryLine (system : TwoCircleSystem) (line : Line) : Prop :=
  sorry

-- Function to count the number of symmetry lines
noncomputable def symmetryLineCount (system : TwoCircleSystem) : ℕ :=
  sorry

-- Theorem for the general case: Different radii circles
theorem different_radii_symmetry (system : TwoCircleSystem) 
  (h : system.circle1.radius ≠ system.circle2.radius) : 
  symmetryLineCount system = 1 := by
  sorry

-- Theorem for special case 1: Concentric circles
theorem concentric_circles_symmetry (system : TwoCircleSystem) 
  (h : system.circle1.center = system.circle2.center) : 
  symmetryLineCount system > 2 := by
  sorry

-- Theorem for special case 2: Same radii with different centers
theorem same_radii_different_centers_symmetry (system : TwoCircleSystem) 
  (h1 : system.circle1.radius = system.circle2.radius) 
  (h2 : system.circle1.center ≠ system.circle2.center) : 
  symmetryLineCount system = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_different_radii_symmetry_concentric_circles_symmetry_same_radii_different_centers_symmetry_l8_835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_progression_set_has_n_identical_numbers_l8_838

/-- A set of positive real numbers where any four pairwise distinct numbers form a geometric progression -/
def GeometricProgressionSet (S : Set ℝ) : Prop :=
  ∀ a b c d, a ∈ S → b ∈ S → c ∈ S → d ∈ S → a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
    ∃ r : ℝ, r > 0 ∧ {a, b, c, d} = {x | ∃ k : ℕ, x = r^k}

/-- The main theorem stating that in a set of 4n positive numbers with the geometric progression property,
    there must be at least n identical numbers -/
theorem geometric_progression_set_has_n_identical_numbers
  {n : ℕ} {S : Finset ℝ} (h_card : S.card = 4 * n) (h_pos : ∀ x ∈ S, x > 0)
  (h_gp : GeometricProgressionSet (S : Set ℝ)) :
  ∃ x ∈ S, (S.filter (· = x)).card ≥ n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_progression_set_has_n_identical_numbers_l8_838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_procgen_reward_is_240_l8_886

/-- The maximum ProcGen reward -/
noncomputable def max_procgen_reward : ℝ := 240

/-- Greg's PPO algorithm's performance on CoinRun as a percentage -/
noncomputable def greg_ppo_percentage : ℝ := 0.9

/-- Greg's PPO algorithm's reward on CoinRun -/
noncomputable def greg_ppo_reward : ℝ := 108

/-- The maximum CoinRun reward -/
noncomputable def max_coinrun_reward : ℝ := greg_ppo_reward / greg_ppo_percentage

/-- Theorem stating that the maximum ProcGen reward is 240 -/
theorem max_procgen_reward_is_240 :
  max_procgen_reward = 2 * max_coinrun_reward :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_procgen_reward_is_240_l8_886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_given_conditions_l8_857

/-- A circle C with center (h, k) and radius r -/
structure Circle where
  h : ℝ
  k : ℝ
  r : ℝ

/-- The line ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

def CircleEquation (c : Circle) (x y : ℝ) : Prop :=
  (x - c.h)^2 + (y - c.k)^2 = c.r^2

def PointOnLine (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y + l.c = 0

def LineTangentToCircle (l : Line) (c : Circle) : Prop :=
  ∃ x y, PointOnLine l x y ∧ CircleEquation c x y ∧
    ∀ x' y', PointOnLine l x' y' → (x' = x ∧ y' = y) ∨ ¬CircleEquation c x' y'

noncomputable def ChordLength (l : Line) (c : Circle) : ℝ :=
  2 * Real.sqrt (c.r^2 - (l.a * c.h + l.b * c.k + l.c)^2 / (l.a^2 + l.b^2))

theorem circle_equation_given_conditions (c : Circle) :
  PointOnLine ⟨3, -1, 0⟩ c.h c.k →
  LineTangentToCircle ⟨0, 1, 0⟩ c →
  ChordLength ⟨1, -1, 0⟩ c = 2 * Real.sqrt 7 →
  (CircleEquation c = fun x y => (x + 1)^2 + (y + 3)^2 = 9) ∨
  (CircleEquation c = fun x y => (x - 1)^2 + (y - 3)^2 = 9) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_given_conditions_l8_857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_reciprocal_problem_l8_804

theorem sum_reciprocal_problem : (12 : ℝ) * (1/3 + 1/4 + 1/6 + 1/12)⁻¹ = 72/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_reciprocal_problem_l8_804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_property_l8_875

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  seq_def : ∀ n, a (n + 1) = a n + d

/-- Sum of the first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (seq.a 1 + seq.a n) / 2

/-- Main theorem -/
theorem arithmetic_sequence_sum_property (seq : ArithmeticSequence) :
  S seq 3 + S seq 7 = 37 → 19 * seq.a 3 + seq.a 11 = 74 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_property_l8_875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mystical_mountain_creatures_l8_843

/-- Represents the number of Nine-Headed Birds -/
def num_birds : ℕ → Prop := sorry

/-- Represents the number of Nine-Tailed Foxes -/
def num_foxes : ℕ → Prop := sorry

/-- The condition for Nine-Headed Bird observation -/
def bird_condition (b f : ℕ) : Prop :=
  9 * f + (b - 1) = 4 * (9 * (b - 1) + f)

/-- The condition for Nine-Tailed Fox observation -/
def fox_condition (b f : ℕ) : Prop :=
  9 * (f - 1) + b = 3 * (9 * b + (f - 1))

/-- Theorem stating the existence of a solution to the mystical mountain creatures problem -/
theorem mystical_mountain_creatures :
  ∃ (b f : ℕ), num_birds b ∧ num_foxes f ∧ bird_condition b f ∧ fox_condition b f ∧ f = 14 := by
  sorry

#check mystical_mountain_creatures

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mystical_mountain_creatures_l8_843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_with_shared_focus_l8_821

/-- The equation of the asymptotes of a hyperbola that shares a focus with a parabola -/
theorem hyperbola_asymptotes_with_shared_focus (a : ℝ) :
  (∃ (x y : ℝ), x^2 / a^2 - y^2 = 1) →  -- Hyperbola equation
  (∃ (x y : ℝ), y^2 = -12*x) →          -- Parabola equation
  (∃ (c : ℝ), c = 3 ∧ a^2 = c^2 - 1) →  -- Shared focus condition
  (∀ (x y : ℝ), y = (Real.sqrt 2/4)*x ∨ y = -(Real.sqrt 2/4)*x) -- Asymptote equations
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_with_shared_focus_l8_821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_sum_l8_883

noncomputable section

-- Define the points
def A : ℝ × ℝ := (0, 8)
def B : ℝ × ℝ := (0, 0)
def C : ℝ × ℝ := (10, 0)

-- Define midpoints
noncomputable def D : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
noncomputable def E : ℝ × ℝ := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
noncomputable def F : ℝ × ℝ := ((A.1 + C.1) / 2, (A.2 + C.2) / 2)

-- Define lines AF and CE
noncomputable def line_AF (x : ℝ) : ℝ := (F.2 - A.2) / (F.1 - A.1) * (x - A.1) + A.2
noncomputable def line_CE (x : ℝ) : ℝ := (E.2 - C.2) / (E.1 - C.1) * (x - C.1) + C.2

-- Define the intersection point G
noncomputable def G : ℝ × ℝ := 
  let x := (line_CE 0 - line_AF 0) / ((F.2 - A.2) / (F.1 - A.1) - (E.2 - C.2) / (E.1 - C.1))
  (x, line_AF x)

-- Theorem statement
theorem intersection_point_sum : G.1 + G.2 = 10 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_sum_l8_883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chessboard_sign_configurations_not_always_possible_to_make_all_plus_l8_897

/-- Represents an 8x8 chessboard with signs -/
def Chessboard : Type := Fin 8 → Fin 8 → Bool

/-- The number of 3x3 squares on an 8x8 chessboard -/
def num_3x3_squares : ℕ := (8 - 3 + 1) * (8 - 3 + 1)

/-- The number of 4x4 squares on an 8x8 chessboard -/
def num_4x4_squares : ℕ := (8 - 4 + 1) * (8 - 4 + 1)

/-- The total number of squares that can be toggled -/
def total_toggleable_squares : ℕ := num_3x3_squares + num_4x4_squares

/-- The number of cells on the chessboard -/
def num_cells : ℕ := 8 * 8

theorem chessboard_sign_configurations :
  2^total_toggleable_squares < 2^num_cells :=
by sorry

/-- Represents an operation on the chessboard -/
inductive Operation
| Toggle3x3 (row col : Fin 8)
| Toggle4x4 (row col : Fin 8)

/-- Applies a list of operations to a chessboard -/
def applyOperations : List Operation → Chessboard → Chessboard
| [], b => b
| (op::ops), b => applyOperations ops (
  match op with
  | Operation.Toggle3x3 r c => sorry  -- Implementation omitted
  | Operation.Toggle4x4 r c => sorry  -- Implementation omitted
)

/-- Checks if all cells on the chessboard are positive -/
def allPositive : Chessboard → Prop
| b => ∀ (i j : Fin 8), b i j = true

/-- 
Theorem: It is not always possible to make all signs on the board "+" 
using the given operations.
-/
theorem not_always_possible_to_make_all_plus :
  ¬ ∀ (b : Chessboard), ∃ (operations : List Operation), 
    allPositive (applyOperations operations b) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chessboard_sign_configurations_not_always_possible_to_make_all_plus_l8_897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shadow_problem_l8_818

noncomputable def cube_edge : ℝ := 2
noncomputable def shadow_area : ℝ := 162

noncomputable def light_source_height (shadow_side : ℝ) : ℝ :=
  cube_edge * (shadow_side - cube_edge) / cube_edge

theorem shadow_problem :
  ∃ (shadow_side : ℝ),
    shadow_side^2 = shadow_area + cube_edge^2 ∧
    Int.floor (1000 * light_source_height shadow_side) = 11000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shadow_problem_l8_818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_parabola_l8_842

/-- The set of points (x, y) satisfying 5√((x-1)² + (y-2)²) = |3x + 4y + 12| forms a parabola -/
theorem trajectory_is_parabola :
  ∃ (a b c d e f : ℝ) (h : a ≠ 0),
    {p : ℝ × ℝ | 5 * Real.sqrt ((p.1 - 1)^2 + (p.2 - 2)^2) = |3 * p.1 + 4 * p.2 + 12|} =
    {p : ℝ × ℝ | a * p.1^2 + b * p.1 * p.2 + c * p.2^2 + d * p.1 + e * p.2 + f = 0} :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_parabola_l8_842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sgn_g_eq_neg_sgn_x_l8_815

-- Define the signum function
noncomputable def sgn (x : ℝ) : ℝ :=
  if x > 0 then 1
  else if x < 0 then -1
  else 0

-- Define the properties of f and g
def is_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

theorem sgn_g_eq_neg_sgn_x 
  (f : ℝ → ℝ) (a : ℝ) (h_inc : is_increasing f) (h_a : a > 1) :
  ∀ x, sgn (f x - f (a * x)) = -sgn x :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sgn_g_eq_neg_sgn_x_l8_815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_intersection_theorem_l8_864

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/4 = 1

/-- The line equation -/
def line_eq (x y : ℝ) : Prop := y = 4*x - 6

/-- The midpoint condition -/
def is_midpoint (mx my ax ay bx : ℝ) (by_ : ℝ) : Prop :=
  mx = (ax + bx)/2 ∧ my = (ay + by_)/2

/-- The main theorem -/
theorem hyperbola_intersection_theorem (ax ay bx : ℝ) (by_ : ℝ) :
  hyperbola ax ay ∧ 
  hyperbola bx by_ ∧
  is_midpoint 2 2 ax ay bx by_ →
  (∀ x y, line_eq x y ↔ (x - 2) = (y - 2)/4) ∧
  Real.sqrt ((ax - bx)^2 + (ay - by_)^2) = 2 * Real.sqrt 102 / 3 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_intersection_theorem_l8_864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l8_874

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

def arithmetic_sequence (b : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, b (n + 1) = b n + d

theorem sequence_properties (a b : ℕ → ℝ) (c : ℕ → ℝ) :
  geometric_sequence a →
  arithmetic_sequence b →
  a 1 = 3 →
  a 4 = 24 →
  b 2 = 4 →
  b 4 = a 3 →
  (∀ n, c n = a n - b n) →
  (∀ n, a n = 3 * 2^(n - 1)) ∧
  (∀ n, b n = 4 * n - 4) ∧
  (∀ n, Finset.sum (Finset.range n) c = 3 * 2^n - 3 - 2 * n^2 + 2 * n) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l8_874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_of_divisors_9999_l8_862

theorem median_of_divisors_9999 : 
  let divisors := (Finset.range 10000).filter (λ x => 9999 % x = 0)
  Finset.card divisors % 2 = 0 ∧ 
  (Finset.sort (λ a b => a ≤ b) divisors).nthLe ((Finset.card divisors) / 2 - 1) sorry +
  (Finset.sort (λ a b => a ≤ b) divisors).nthLe (Finset.card divisors / 2) sorry = 200 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_of_divisors_9999_l8_862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rice_bag_cost_effectiveness_l8_869

/-- Represents the size of a rice bag -/
inductive BagSize
  | Small
  | Medium
  | Large

/-- Represents the cost and quantity of rice in a bag -/
structure Bag where
  size : BagSize
  cost : ℚ
  quantity : ℚ

/-- Calculates the cost-effectiveness of a bag -/
def costEffectiveness (bag : Bag) : ℚ :=
  bag.quantity / bag.cost

/-- Checks if one bag is more cost-effective than another -/
def isMoreCostEffective (bag1 bag2 : Bag) : Prop :=
  costEffectiveness bag1 > costEffectiveness bag2

theorem rice_bag_cost_effectiveness 
  (s m l : Bag)
  (h_s_size : s.size = BagSize.Small)
  (h_m_size : m.size = BagSize.Medium)
  (h_l_size : l.size = BagSize.Large)
  (h_m_cost : m.cost = 2 * s.cost)
  (h_m_quantity : m.quantity = 7 * l.quantity / 10)
  (h_l_quantity : l.quantity = 3 * s.quantity)
  (h_l_cost : l.cost = 6 * s.cost / 5) :
  isMoreCostEffective l m ∧ isMoreCostEffective m s := by
  sorry

#check rice_bag_cost_effectiveness

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rice_bag_cost_effectiveness_l8_869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_net_rate_is_25_l8_803

/-- Calculates the net rate of pay for a driver --/
noncomputable def netRateOfPay (travelTime : ℝ) (speed : ℝ) (fuelEfficiency : ℝ) 
                 (paymentRate : ℝ) (gasolineCost : ℝ) : ℝ :=
  let distance := travelTime * speed
  let gasolineUsed := distance / fuelEfficiency
  let earnings := paymentRate * distance
  let gasolineExpense := gasolineCost * gasolineUsed
  let netEarnings := earnings - gasolineExpense
  netEarnings / travelTime

/-- Theorem stating that the net rate of pay is $25 per hour --/
theorem net_rate_is_25 :
  netRateOfPay 3 50 25 0.60 2.50 = 25 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval netRateOfPay 3 50 25 0.60 2.50

end NUMINAMATH_CALUDE_ERRORFEEDBACK_net_rate_is_25_l8_803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_negative_integers_l8_851

theorem max_negative_integers (integers : Finset ℤ) : 
  integers.card = 8 → 
  Even (integers.prod id) → 
  ∃ (negative_set : Finset ℤ), negative_set ⊆ integers ∧ 
    negative_set.card = 8 ∧ 
    ∀ i ∈ negative_set, i < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_negative_integers_l8_851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_news_probability_is_one_sixth_l8_858

/-- Represents the duration of a news broadcast in minutes -/
noncomputable def news_duration : ℝ := 5

/-- Represents the total cycle time between news broadcasts in minutes -/
noncomputable def total_cycle_time : ℝ := 30

/-- The probability of hearing the news when turning on the radio at a random time -/
noncomputable def probability_of_hearing_news : ℝ := news_duration / total_cycle_time

theorem news_probability_is_one_sixth :
  probability_of_hearing_news = 1 / 6 := by
  -- Unfold the definitions
  unfold probability_of_hearing_news news_duration total_cycle_time
  -- Simplify the fraction
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_news_probability_is_one_sixth_l8_858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_whitewashing_cost_is_5_l8_845

/-- Represents the dimensions of a rectangular prism -/
structure Dimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the whitewashing cost per square foot -/
noncomputable def whitewashing_cost_per_sqft (room : Dimensions) (door : Dimensions) (window : Dimensions) (num_windows : ℕ) (total_cost : ℝ) : ℝ :=
  let wall_area := 2 * (room.length + room.width) * room.height
  let door_area := door.length * door.width
  let window_area := num_windows * (window.length * window.width)
  let whitewash_area := wall_area - door_area - window_area
  total_cost / whitewash_area

/-- Theorem: The cost of whitewashing per square foot is 5 Rs. given the room dimensions and total cost -/
theorem whitewashing_cost_is_5 (room : Dimensions) (door : Dimensions) (window : Dimensions) :
  whitewashing_cost_per_sqft room door window 3 4530 = 5 :=
by
  have h_room : room = ⟨25, 15, 12⟩ := by sorry
  have h_door : door = ⟨6, 3, 0⟩ := by sorry
  have h_window : window = ⟨4, 3, 0⟩ := by sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_whitewashing_cost_is_5_l8_845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l8_890

-- Define the points
def point1 : ℝ × ℝ := (-3, 4)
def point2 : ℝ × ℝ := (5, -6)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- Theorem statement
theorem distance_between_points :
  distance point1 point2 = 2 * Real.sqrt 41 := by
  -- Unfold the definitions
  unfold distance point1 point2
  -- Simplify the expressions
  simp
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l8_890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inclination_angle_eq_l8_898

/-- Regular quadrilateral pyramid with lateral edge angle α -/
structure RegularQuadPyramid where
  α : ℝ  -- angle between lateral edge and base plane
  (α_pos : α > 0)
  (α_lt_pi_div_two : α < Real.pi / 2)

/-- The angle between the base plane and the constructed plane -/
noncomputable def inclinationAngle (p : RegularQuadPyramid) : ℝ :=
  Real.arctan (Real.tan p.α / 3)

/-- Theorem stating the relationship between the lateral edge angle and the inclination angle -/
theorem inclination_angle_eq (p : RegularQuadPyramid) :
  inclinationAngle p = Real.arctan (Real.tan p.α / 3) := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inclination_angle_eq_l8_898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_neither_even_nor_odd_l8_860

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := Real.log (x + Real.sqrt (4 + x^2))

-- Statement to prove
theorem g_neither_even_nor_odd :
  ¬(∀ x, g (-x) = g x) ∧ ¬(∀ x, g (-x) = -g x) := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_neither_even_nor_odd_l8_860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_exponential_sum_l8_831

theorem min_value_of_exponential_sum (x : ℝ) : 
  (2 : ℝ)^x + (2 : ℝ)^(2-x) ≥ 4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_exponential_sum_l8_831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_values_of_f_tangent_line_at_P_l8_825

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 + x^2 - x + 2

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 3*x^2 + 2*x - 1

-- Theorem for extreme values
theorem extreme_values_of_f :
  (∃ x : ℝ, f x = 3 ∧ ∀ y : ℝ, f y ≤ 3) ∧
  (∃ x : ℝ, f x = 49/27 ∧ ∀ y : ℝ, f y ≥ 49/27) :=
sorry

-- Theorem for tangent line equation at P(1,3)
theorem tangent_line_at_P :
  ∀ x y : ℝ, f 1 = 3 → (4*x - y - 1 = 0 ↔ y = f x + f' 1 * (x - 1)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_values_of_f_tangent_line_at_P_l8_825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_existence_l8_833

theorem smallest_k_existence (n : ℕ) (h_even : Even n) : 
  ∃ k₀ : ℕ, 
    k₀ > 0 ∧ 
    (∃ f g : Polynomial ℤ, k₀ = (f * (X + 1)^n + g * (X^n + 1)).eval 0) ∧
    (∃ a t : ℕ, n = 2^a * t ∧ Odd t ∧ k₀ = 2^t) ∧
    (∀ k : ℕ, k > 0 → (∃ f g : Polynomial ℤ, k = (f * (X + 1)^n + g * (X^n + 1)).eval 0) → k₀ ≤ k) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_existence_l8_833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_negative_functions_l8_839

-- Define the "inverse negative" property
def inverse_negative (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x ≠ 0 → f (1 / x) = -f x

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := x - 1 / x

noncomputable def g (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 1 then x
  else if x = 1 then 0
  else -1 / x

noncomputable def h (x : ℝ) : ℝ := -x^3 + 1 / x^3

-- State the theorem
theorem inverse_negative_functions :
  inverse_negative f ∧ inverse_negative g ∧ inverse_negative h := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_negative_functions_l8_839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ap_terms_count_l8_808

/-- An arithmetic progression with an even number of terms -/
structure EvenAP where
  n : ℕ
  a : ℝ
  d : ℝ
  even_terms : Even n

/-- The sum of odd-numbered terms in the arithmetic progression -/
noncomputable def sum_odd_terms (ap : EvenAP) : ℝ :=
  (ap.n / 2 : ℝ) * (ap.a + (ap.a + (ap.n - 2) * ap.d))

/-- The sum of even-numbered terms in the arithmetic progression -/
noncomputable def sum_even_terms (ap : EvenAP) : ℝ :=
  (ap.n / 2 : ℝ) * ((ap.a + ap.d) + (ap.a + (ap.n - 1) * ap.d))

/-- The difference between the last and first terms -/
noncomputable def last_first_diff (ap : EvenAP) : ℝ :=
  (ap.n - 1) * ap.d

theorem ap_terms_count (ap : EvenAP) 
  (h_odd_sum : sum_odd_terms ap = 24)
  (h_even_sum : sum_even_terms ap = 30)
  (h_diff : last_first_diff ap = 10.5) :
  ap.n = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ap_terms_count_l8_808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_l8_840

noncomputable def f (x : ℝ) := 1 / (Real.sin x * Real.cos x) + Real.sin x

theorem f_period : ∃ (p : ℝ), p > 0 ∧ (∀ (x : ℝ), f (x + p) = f x) ∧ 
  (∀ (q : ℝ), q > 0 → (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧ p = 2 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_l8_840
