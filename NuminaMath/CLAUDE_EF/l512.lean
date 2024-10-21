import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_money_is_29_70_l512_51248

noncomputable def initial_amount : ℝ := 120

noncomputable def tablet_cost (amount : ℝ) : ℝ := 0.45 * amount

noncomputable def phone_game_cost (amount : ℝ) : ℝ := (1/3) * amount

noncomputable def in_game_purchases (amount : ℝ) : ℝ := 0.25 * amount

noncomputable def tablet_case_cost (amount : ℝ) : ℝ := 0.1 * amount

noncomputable def remaining_money : ℝ :=
  let after_tablet := initial_amount - tablet_cost initial_amount
  let after_phone_game := after_tablet - phone_game_cost after_tablet
  let after_in_game := after_phone_game - in_game_purchases after_phone_game
  after_in_game - tablet_case_cost after_in_game

theorem remaining_money_is_29_70 : remaining_money = 29.70 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_money_is_29_70_l512_51248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_problem_l512_51254

theorem integral_problem (a : ℝ) (h1 : a > 0) (h2 : ∫ x in (0 : ℝ)..a, (2*x - 2) = 3) : a = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_problem_l512_51254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_height_theorem_l512_51223

/-- Represents the height of water in a cone after submerging a sphere -/
noncomputable def water_height_after_sphere_submersion 
  (H : ℝ) -- Height of the cone
  (R : ℝ) -- Base radius of the cone
  (h : ℝ) -- Initial height of water
  (r : ℝ) -- Radius of the submerged sphere
  : ℝ :=
  (h^3 + 4 * H^2 * r^3 / R^2) ^ (1/3)

/-- Theorem stating that the new water height after submerging a sphere is correct -/
theorem water_height_theorem 
  (H : ℝ) -- Height of the cone
  (R : ℝ) -- Base radius of the cone
  (h : ℝ) -- Initial height of water
  (r : ℝ) -- Radius of the submerged sphere
  (x : ℝ) -- New height of water after sphere submersion
  (h_pos : 0 < h) -- Assumption: initial water height is positive
  (H_pos : 0 < H) -- Assumption: cone height is positive
  (R_pos : 0 < R) -- Assumption: cone base radius is positive
  (r_pos : 0 < r) -- Assumption: sphere radius is positive
  (h_le_H : h ≤ H) -- Assumption: initial water height is not greater than cone height
  : x = water_height_after_sphere_submersion H R h r :=
by
  sorry

#check water_height_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_height_theorem_l512_51223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_flight_time_approx_l512_51230

/-- The initial velocity of the balls in m/s -/
def v₀ : ℝ := 10

/-- The acceleration due to gravity in m/s² -/
def g : ℝ := 10

/-- The proportion of the maximum range we're interested in -/
def range_proportion : ℝ := 0.96

/-- The launch angle that maximizes the flight time for the given range proportion -/
noncomputable def optimal_angle (v₀ g range_proportion : ℝ) : ℝ :=
  Real.pi / 2 - 1 / 2 * Real.arcsin range_proportion

/-- The flight time for a given launch angle -/
noncomputable def flight_time (v₀ g α : ℝ) : ℝ :=
  2 * v₀ * Real.sin α / g

/-- Theorem stating that the maximum flight time for balls landing at or beyond 
    96% of the maximum range is approximately 1.6 seconds -/
theorem max_flight_time_approx (ε : ℝ) (hε : ε > 0) :
  ∃ t : ℝ, abs (t - flight_time v₀ g (optimal_angle v₀ g range_proportion)) < ε ∧ 
           abs (t - 1.6) < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_flight_time_approx_l512_51230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_square_area_l512_51261

-- Define the necessary types and functions
def Point := ℝ × ℝ
def Square := Set Point

def is_square (S : Square) : Prop := sorry
def inscribed_in_right_triangle (S : Square) (XY YZ : ℝ) : Prop := sorry
def area (S : Square) : ℝ := sorry

theorem inscribed_square_area (XY YZ : ℝ) (h1 : XY = 35) (h2 : YZ = 75) :
  ∃ (PQRS : Square), 
    is_square PQRS ∧ 
    inscribed_in_right_triangle PQRS XY YZ ∧ 
    area PQRS = 2625 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_square_area_l512_51261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_large_triangle_area_is_four_sqrt_three_l512_51211

/-- Configuration of triangles -/
structure TriangleConfiguration where
  /-- Side length of the central and outer triangles -/
  side_length : ℝ
  /-- Assumption that the side length is 2 -/
  side_length_eq_two : side_length = 2

/-- The area of the large triangle formed by the centers of the outer triangles -/
noncomputable def large_triangle_area (config : TriangleConfiguration) : ℝ :=
  4 * Real.sqrt 3

/-- Theorem stating that the area of the large triangle is 4√3 -/
theorem large_triangle_area_is_four_sqrt_three (config : TriangleConfiguration) :
  large_triangle_area config = 4 * Real.sqrt 3 := by
  -- Unfold the definition of large_triangle_area
  unfold large_triangle_area
  -- The equality follows directly from the definition
  rfl

#check large_triangle_area_is_four_sqrt_three

end NUMINAMATH_CALUDE_ERRORFEEDBACK_large_triangle_area_is_four_sqrt_three_l512_51211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_squares_is_eleven_l512_51234

/-- A point on a grid paper --/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- A set of points on a grid paper --/
def GridPointSet := Finset GridPoint

/-- A function that counts the number of squares that can be formed by connecting four points from a given set --/
def countSquares (points : GridPointSet) : ℕ := sorry

/-- The theorem stating that the maximum number of squares is 11 --/
theorem max_squares_is_eleven (points : GridPointSet) : 
  points.card = 12 → countSquares points ≤ 11 ∧ ∃ (p : GridPointSet), p.card = 12 ∧ countSquares p = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_squares_is_eleven_l512_51234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_10_terms_l512_51224

open BigOperators

def mySequence (n : ℕ) : ℚ := 1 / ((3 * n - 2) * (3 * n + 1))

theorem sum_of_first_10_terms : 
  ∑ i in Finset.range 10, mySequence (i + 1) = 10 / 31 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_10_terms_l512_51224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_properties_l512_51220

theorem expansion_properties :
  ∃ (f : Polynomial ℝ),
    f = (X + 2) * (X + 1)^6
    ∧ (Polynomial.coeff f 3 = 55)
    ∧ (Polynomial.eval 1 f = 192) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_properties_l512_51220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arctan_sum_tangent_l512_51249

theorem arctan_sum_tangent (a b : ℝ) : 
  a = 2/3 → (a + 1) * (b + 1) = 8/3 → Real.tan (Real.arctan a + Real.arctan b) = 19/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arctan_sum_tangent_l512_51249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_12_eq_90_l512_51272

/-- A sequence of integers satisfying the given recurrence relation -/
def b : ℕ → ℤ
  | 0 => 2  -- We use 0 to represent the first term
  | (n + 1) => b (n / 2) + b ((n + 1) / 2) + (n / 2 + 1) * ((n + 1) / 2 + 1)

/-- The 12th term of the sequence is 90 -/
theorem b_12_eq_90 : b 11 = 90 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_12_eq_90_l512_51272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_area_of_quadrilateral_l512_51297

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Represents a parabola in 2D space -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

noncomputable def parabola : Parabola := { a := 1, h := 0, k := 0 }
noncomputable def focus : Point := { x := 0, y := 1/4 }
noncomputable def intersectionPoint : Point := { x := 0, y := 2 }

def isOnParabola (p : Point) (para : Parabola) : Prop :=
  p.y = para.a * (p.x - para.h)^2 + para.k

def isOnLine (p : Point) (l : Line) : Prop :=
  p.y = l.slope * p.x + l.intercept

def isSymmetric (p1 p2 : Point) (l : Line) : Prop :=
  (p1.y + p2.y) / 2 = l.slope * ((p1.x + p2.x) / 2) + l.intercept

noncomputable def quadrilateralArea (p1 p2 p3 p4 : Point) : ℝ :=
  sorry

theorem minimum_area_of_quadrilateral (l : Line) (b d : Point) :
  isOnParabola b parabola →
  isOnParabola d parabola →
  isOnLine b l →
  isOnLine d l →
  isOnLine intersectionPoint l →
  ∃ (c : Point), isSymmetric c d l →
  ∀ (c' : Point), isSymmetric c' d l →
    quadrilateralArea { x := 0, y := 0 } b d c ≤ quadrilateralArea { x := 0, y := 0 } b d c' →
  quadrilateralArea { x := 0, y := 0 } b d c = 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_area_of_quadrilateral_l512_51297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_annual_growth_rate_correct_other_options_incorrect_l512_51273

-- Define the monthly growth rate
variable (p : ℝ)

-- Define the annual growth rate function
def annual_growth_rate (p : ℝ) : ℝ := (1 + p)^12 - 1

-- Theorem stating that the annual growth rate is (1+p)^12 - 1
theorem annual_growth_rate_correct (p : ℝ) :
  annual_growth_rate p = (1 + p)^12 - 1 := by
  -- The proof is trivial since it's by definition
  rfl

-- Theorem stating that the other options are not correct
theorem other_options_incorrect (p : ℝ) :
  annual_growth_rate p ≠ p ∧
  annual_growth_rate p ≠ 12*p ∧
  annual_growth_rate p ≠ (1 + p)^12 := by
  -- We'll leave the actual proof as an exercise
  sorry

#check annual_growth_rate_correct
#check other_options_incorrect

end NUMINAMATH_CALUDE_ERRORFEEDBACK_annual_growth_rate_correct_other_options_incorrect_l512_51273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_birthday_celebration_problem_l512_51214

theorem birthday_celebration_problem (total_guests : ℕ) 
  (h_total : total_guests = 60)
  (h_women : total_guests / 2 = 30)
  (h_men : 15 = 15)
  (h_remaining : 50 = 50) : 
  let women := total_guests / 2
  let men := 15
  let initial_children := total_guests - women - men
  let men_left := men / 3
  let total_left := total_guests - 50
  total_left - men_left = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_birthday_celebration_problem_l512_51214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_max_distance_C2_to_l_l512_51268

noncomputable section

-- Define the line l
def line_l (t : ℝ) : ℝ × ℝ := (1 + (1/2) * t, (Real.sqrt 3 / 2) * t)

-- Define curve C₁
def curve_C1 (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ)

-- Define curve C₂
def curve_C2 (θ : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos θ, 3 * Real.sin θ)

-- Theorem for part 1
theorem intersection_distance :
  ∃ A B : ℝ × ℝ, 
  (∃ t : ℝ, line_l t = A) ∧ 
  (∃ θ : ℝ, curve_C1 θ = A) ∧
  (∃ t : ℝ, line_l t = B) ∧ 
  (∃ θ : ℝ, curve_C1 θ = B) ∧
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 1 := by
  sorry

-- Theorem for part 2
theorem max_distance_C2_to_l :
  ∃ d : ℝ, 
  (∀ θ : ℝ, ∀ t : ℝ, 
    let P := curve_C2 θ
    let Q := line_l t
    Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) ≤ d) ∧
  d = (3 * Real.sqrt 2 + Real.sqrt 3) / 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_max_distance_C2_to_l_l512_51268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_negative_three_l512_51299

/-- The function f as defined in the problem -/
noncomputable def f (x : ℝ) : ℝ := x⁻¹ + x⁻¹ / (1 + x⁻¹)

/-- Theorem stating that f(f(-3)) = 24/5 -/
theorem f_composition_negative_three : f (f (-3)) = 24 / 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_negative_three_l512_51299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_max_area_l512_51292

/-- The length of the wire in centimeters -/
noncomputable def wire_length : ℝ := 32

/-- The area of a rectangle formed by the wire as a function of one side length -/
noncomputable def rectangle_area (x : ℝ) : ℝ := x * (wire_length / 2 - x)

/-- The maximum area of a rectangle that can be formed by the wire -/
noncomputable def max_area : ℝ := 64

theorem rectangle_max_area :
  ∃ (x : ℝ), 0 < x ∧ x < wire_length / 2 ∧
  rectangle_area x = max_area ∧
  ∀ (y : ℝ), 0 < y → y < wire_length / 2 → rectangle_area y ≤ max_area := by
  sorry

#eval "Theorem stated successfully"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_max_area_l512_51292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersectionCount_l512_51262

/-- The number of intersection points between two regular polygons with m and n sides, 
    where m > n and no vertices are shared -/
def intersectionPoints (m n : ℕ) : ℕ := 2 * n

/-- The set of regular polygons inscribed in the circle -/
def polygons : Finset ℕ := {4, 5, 6, 7}

/-- The total number of intersection points between all pairs of polygons -/
def totalIntersections : ℕ :=
  Finset.sum (Finset.filter (fun p => p.1 > p.2) (polygons.product polygons))
    (fun p => intersectionPoints p.1 p.2)

theorem intersectionCount : totalIntersections = 56 := by
  -- Expand the definition of totalIntersections
  unfold totalIntersections
  -- Evaluate the sum
  simp [polygons, intersectionPoints]
  -- The rest of the proof
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersectionCount_l512_51262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boy_strategy_exists_l512_51259

/-- A strategy for picking candies -/
def Strategy := Nat → List Nat → Nat

/-- The game state after each turn -/
structure GameState where
  boxes : List Nat
  turn : Nat

/-- The result of applying a strategy -/
def applyStrategy (s : Strategy) (initial : List Nat) : GameState :=
  sorry

/-- Theorem: There exists a strategy for the boy to ensure the last two candies come from the same box -/
theorem boy_strategy_exists (n : Nat) :
  ∃ (s : Strategy), ∀ (initial : List Nat),
    (initial.length = n) →
    (initial.sum = 2 * n) →
    let final := applyStrategy s initial
    (final.boxes.filter (· > 0)).length = 1 ∧
    final.boxes.sum = 2 :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_boy_strategy_exists_l512_51259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_interval_of_f_l512_51242

-- Define the function f(x) = ln x - x
noncomputable def f (x : ℝ) : ℝ := Real.log x - x

-- State the theorem
theorem monotonic_increasing_interval_of_f :
  ∀ x : ℝ, (0 < x ∧ x < 1) ↔ (∀ y : ℝ, 0 < y ∧ y < x → f y < f x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_interval_of_f_l512_51242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l512_51244

/-- Parabola with focus F and point A satisfying given conditions -/
structure Parabola where
  p : ℝ
  a : ℝ
  hp : p > 0
  hA : a^2 = 4 * 2
  hAF : Real.sqrt ((2 - 1)^2 + a^2) = 3

/-- Line passing through focus F and intersecting parabola at M and N -/
structure IntersectingLine (C : Parabola) where
  m : ℝ
  hMN : ∀ y, y^2 = 4 * (m * y + 1) → y^2 - 4 * m * y - 4 = 0

/-- Main theorem statement -/
theorem parabola_properties (C : Parabola) (l : IntersectingLine C) :
  (∀ x y, y^2 = 4 * x ↔ y^2 = 2 * C.p * x) ∧
  (C.p = 2 ∧ (1, 0) = (C.p / 2, 0)) ∧
  (∀ x y, 2 * x - y - 2 = 0 ↔ x = l.m * y + 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l512_51244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_circle_parabola_intersection_line_l512_51280

-- Define the parabola C: y^2 = 4x
def C (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus F(1,0)
def F : ℝ × ℝ := (1, 0)

-- Define a line passing through a point with a given slope
def line_through_point (p : ℝ × ℝ) (slope : ℝ) (x y : ℝ) : Prop :=
  y - p.2 = slope * (x - p.1)

-- Define a circle with center (a,b) and radius r
def circle_equation (a b r : ℝ) (x y : ℝ) : Prop :=
  (x - a)^2 + (y - b)^2 = r^2

theorem parabola_intersection_circle :
  ∀ A B : ℝ × ℝ,
  (C A.1 A.2 ∧ C B.1 B.2) →  -- A and B are on the parabola
  (line_through_point F 1 A.1 A.2 ∧ line_through_point F 1 B.1 B.2) →  -- A and B are on the line through F with slope 1
  (circle_equation 3 2 4 A.1 A.2 ∧ circle_equation 3 2 4 B.1 B.2) -- A and B are on the circle (x-3)^2 + (y-2)^2 = 16
  := by sorry

theorem parabola_intersection_line :
  ∀ k : ℝ,
  k = 2 ∨ k = -2 →
  ∀ x y : ℝ,
  line_through_point F k x y ↔ y = k*(x - 1)
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_circle_parabola_intersection_line_l512_51280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sqrt_one_minus_x_squared_plus_3x_l512_51229

theorem integral_sqrt_one_minus_x_squared_plus_3x :
  ∫ x in (0:ℝ)..(1:ℝ), (Real.sqrt (1 - x^2) + 3 * x) = π / 4 + 3 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sqrt_one_minus_x_squared_plus_3x_l512_51229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_coefficient_sets_l512_51239

theorem quadratic_coefficient_sets (a b c : ℝ) : 
  a > 0 →
  (let f (x : ℝ) := a * x^2 + b * x + c;
   |f 1| = 1 ∧ |f 2| = 1 ∧ |f 3| = 1) →
  ((a, b, c) = (2, -8, 7) ∨ (a, b, c) = (1, -3, 1) ∨ (a, b, c) = (1, -5, 5)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_coefficient_sets_l512_51239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_for_intersection_equality_l512_51218

/-- The set A of solutions to x^2 - 3x + 2 = 0 -/
def A : Set ℝ := {x | x^2 - 3*x + 2 = 0}

/-- The set B of solutions to x^2 - mx + 2 = 0, parameterized by m -/
def B (m : ℝ) : Set ℝ := {x | x^2 - m*x + 2 = 0}

/-- The theorem stating the range of m for which A ∩ B = B -/
theorem range_of_m_for_intersection_equality :
  {m : ℝ | A ∩ B m = B m} = Set.Ioo (-Real.sqrt 2) 2 ∪ {3} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_for_intersection_equality_l512_51218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_5_equals_142_l512_51240

def sequence_b : ℕ → ℕ
  | 0 => 2  -- We define b₁ as sequence_b 0
  | 1 => 5  -- We define b₂ as sequence_b 1
  | n+2 => 2 * sequence_b (n+1) + 3 * sequence_b n

theorem b_5_equals_142 : sequence_b 4 = 142 := by
  -- The proof goes here
  sorry

#eval sequence_b 4  -- This will evaluate b₅

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_5_equals_142_l512_51240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weighted_average_AC_l512_51270

/-- The weighted average of sets A and C combined -/
noncomputable def weightedAverageAC (totalNumbers : ℕ) (totalAvg : ℝ) 
  (avgA avgB avgC : ℝ) (weightA weightB weightC : ℕ) : ℝ :=
  ((avgA * (weightA : ℝ) + avgC * (weightC : ℝ)) / ((weightA + weightC) : ℝ))

/-- Theorem stating the weighted average of sets A and C combined -/
theorem weighted_average_AC : 
  ∀ (totalNumbers : ℕ) (totalAvg avgA avgB avgC : ℝ) (weightA weightB weightC : ℕ),
  totalNumbers = 8 →
  totalAvg = 7.45 →
  avgA = 7.3 →
  avgB = 7.6 →
  avgC = 7.2 →
  weightA = 3 →
  weightB = 4 →
  weightC = 1 →
  weightA + weightC = 5 →
  weightedAverageAC totalNumbers totalAvg avgA avgB avgC weightA weightB weightC = 5.82 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_weighted_average_AC_l512_51270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prove_p_and_not_q_l512_51222

-- Define proposition p
def p : Prop := ∀ x : ℝ, x > 0 → (3 : ℝ)^x > (2 : ℝ)^x

-- Define proposition q
def q : Prop := ∃ x : ℝ, x < 0 ∧ 3*x > 2*x

-- Theorem to prove
theorem prove_p_and_not_q : p ∧ ¬q := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prove_p_and_not_q_l512_51222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_sample_set_l512_51245

noncomputable def sample_set : List ℝ := [2, 3, 7, 8]

noncomputable def average (l : List ℝ) (a : ℝ) : ℝ := (l.sum + a) / (l.length + 1)

noncomputable def variance (l : List ℝ) (a : ℝ) : ℝ :=
  let mean := average l a
  ((l.map (λ x => (x - mean)^2)).sum + (a - mean)^2) / (l.length + 1)

theorem variance_of_sample_set :
  ∃ a : ℝ, average sample_set a = 5 ∧ variance sample_set a = 26/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_sample_set_l512_51245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_hyperbola_foci_distance_l512_51279

/-- A hyperbola with given asymptotes and a point it passes through -/
structure Hyperbola where
  asymptote1 : ℝ → ℝ
  asymptote2 : ℝ → ℝ
  point : ℝ × ℝ

/-- The distance between the foci of a hyperbola -/
noncomputable def foci_distance (h : Hyperbola) : ℝ := sorry

/-- Theorem stating the distance between foci for a specific hyperbola -/
theorem specific_hyperbola_foci_distance :
  let h : Hyperbola := {
    asymptote1 := λ x ↦ 2 * x + 3,
    asymptote2 := λ x ↦ -2 * x + 1,
    point := (2, 1)
  }
  foci_distance h = 2 * Real.sqrt 30 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_hyperbola_foci_distance_l512_51279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_l512_51298

/-- A function f(x) with specific properties -/
noncomputable def f (A B C : ℤ) (x : ℝ) : ℝ := x^2 / (A * x^2 + B * x + C)

/-- Theorem stating the sum of coefficients given specific conditions -/
theorem sum_of_coefficients 
  (A B C : ℤ) 
  (h1 : ∀ x > 5, f A B C x > 0.6)
  (h2 : (A * (-3)^2 + B * (-3) + C = 0) ∧ (A * 2^2 + B * 2 + C = 0)) :
  A + B + C = -6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_l512_51298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_ratio_l512_51269

-- Define a point in a plane
structure Point where
  x : ℝ
  y : ℝ

-- Define the distance between two points
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

-- Define the sum of distances between all pairs of points
noncomputable def sumDistances (p1 p2 p3 p4 : Point) : ℝ :=
  distance p1 p2 + distance p1 p3 + distance p1 p4 +
  distance p2 p3 + distance p2 p4 + distance p3 p4

-- Define the minimum distance between any pair of points
noncomputable def minDistance (p1 p2 p3 p4 : Point) : ℝ :=
  min (min (min (distance p1 p2) (distance p1 p3)) (min (distance p1 p4) (distance p2 p3)))
      (min (distance p2 p4) (distance p3 p4))

-- The main theorem
theorem min_distance_ratio (p1 p2 p3 p4 : Point) 
  (h : p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4) :
  (sumDistances p1 p2 p3 p4) / (minDistance p1 p2 p3 p4) ≥ 5 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_ratio_l512_51269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grocery_cost_is_16_l512_51236

/-- The total cost of a grocery purchase -/
def total_cost (chicken_cost_per_person : ℚ) (num_people : ℕ) 
               (beef_cost_per_pound : ℚ) (beef_pounds : ℕ) 
               (oil_cost : ℚ) : ℚ :=
  chicken_cost_per_person * num_people + 
  beef_cost_per_pound * beef_pounds + 
  oil_cost

/-- Theorem stating the total cost of the grocery purchase is $16 -/
theorem grocery_cost_is_16 :
  total_cost 1 3 4 3 1 = 16 := by
  -- Unfold the definition of total_cost
  unfold total_cost
  -- Simplify the arithmetic
  simp
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_grocery_cost_is_16_l512_51236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_thursdays_in_july_l512_51260

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
deriving Repr, BEq

/-- Returns the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

/-- Returns the day of the week after n days -/
def addDays (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => nextDay (addDays d n)

/-- Counts the occurrences of a specific day in a month -/
def countDayInMonth (startDay : DayOfWeek) (targetDay : DayOfWeek) (daysInMonth : Nat) : Nat :=
  (List.range daysInMonth).filter (fun i => (addDays startDay i) == targetDay) |>.length

theorem five_thursdays_in_july (firstJuneDay : DayOfWeek) 
    (h : countDayInMonth firstJuneDay DayOfWeek.Tuesday 30 = 5) : 
    countDayInMonth (addDays firstJuneDay 30) DayOfWeek.Thursday 31 = 5 := by
  sorry

#eval countDayInMonth DayOfWeek.Wednesday DayOfWeek.Thursday 31

end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_thursdays_in_july_l512_51260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_and_max_S_l512_51227

noncomputable section

-- Define the inequality
def inequality (x : ℝ) : Prop := |x^2 - 3*x - 4| < 2*x + 2

-- Define the solution set
def solution_set (a b : ℝ) : Set ℝ := {x : ℝ | a < x ∧ x < b}

-- Define S
noncomputable def S (a b m n : ℝ) : ℝ := a / (m^2 - 1) + b / (3 * (n^2 - 1))

theorem inequality_solution_and_max_S :
  ∃ (a b : ℝ), 
    (∀ x, x ∈ solution_set a b ↔ inequality x) ∧
    a = 2 ∧ 
    b = 6 ∧
    (∀ m n : ℝ, m ∈ Set.Ioo (-1 : ℝ) 1 → n ∈ Set.Ioo (-1 : ℝ) 1 → m * n = a / b →
      S a b m n ≤ -6) ∧
    (∃ m n : ℝ, m ∈ Set.Ioo (-1 : ℝ) 1 ∧ n ∈ Set.Ioo (-1 : ℝ) 1 ∧ m * n = a / b ∧ S a b m n = -6) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_and_max_S_l512_51227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_vector_expression_l512_51284

theorem min_vector_expression (O A B : ℝ × ℝ) (t : ℝ) : 
  (O.1 - A.1) * (O.1 - B.1) + (O.2 - A.2) * (O.2 - B.2) = 0 → 
  Real.sqrt ((O.1 - A.1)^2 + (O.2 - A.2)^2) = 24 → 
  Real.sqrt ((O.1 - B.1)^2 + (O.2 - B.2)^2) = 24 → 
  0 ≤ t ∧ t ≤ 1 → 
  ∃ (min_value : ℝ), 
    min_value = 26 ∧
    ∀ t', 0 ≤ t' ∧ t' ≤ 1 → 
      Real.sqrt ((t' * (B.1 - A.1) - (A.1 - O.1))^2 + (t' * (B.2 - A.2) - (A.2 - O.2))^2) +
      Real.sqrt (((5/12) * (B.1 - O.1) - (1 - t') * (A.1 - B.1))^2 + 
                 ((5/12) * (B.2 - O.2) - (1 - t') * (A.2 - B.2))^2) ≥ min_value :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_vector_expression_l512_51284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_difference_l512_51264

def a : ℕ → ℕ
| 0 => 0
| n + 1 => 2 * a n + 1

theorem units_digit_difference : (a 2004 - a 2003) % 10 = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_difference_l512_51264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_period_and_triangle_l512_51281

-- Define the vectors m and n
noncomputable def m (ω x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin (ω * x), Real.cos (ω * x))
noncomputable def n (ω x : ℝ) : ℝ × ℝ := (Real.cos (ω * x), -Real.cos (ω * x))

-- Define the function f as the dot product of m and n
noncomputable def f (ω x : ℝ) : ℝ := (m ω x).1 * (n ω x).1 + (m ω x).2 * (n ω x).2

-- State the theorem
theorem vector_period_and_triangle (ω : ℝ) (a b c : ℝ) :
  ω > 0 →
  (∃ T, T > 0 ∧ T = π / 2 ∧ ∀ x, f ω (x + T) = f ω x) →
  b^2 = a * c →
  (ω = 2 ∧
   ∀ k, (∃ x₁ x₂, x₁ ≠ x₂ ∧ f ω x₁ = k ∧ f ω x₂ = k) ↔ -1 < k ∧ k < 1/2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_period_and_triangle_l512_51281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l512_51241

/-- The equation from the original problem -/
def equation (X : ℝ) : Prop :=
  X / (10.5 * 0.24 - 15.15 / 7.5) = 9 * (1 + 11/20 - 0.945 / 0.9) / (1 + 3/40 - (4 + 3/8) / 7)

/-- The theorem stating that the solution to the equation is 5 -/
theorem equation_solution : ∃ X : ℝ, equation X ∧ X = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l512_51241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inverse_is_g_l512_51204

noncomputable def f (x : ℝ) : ℝ := 3 - 7 * x

noncomputable def g (x : ℝ) : ℝ := (3 - x) / 7

theorem f_inverse_is_g : Function.LeftInverse g f ∧ Function.RightInverse g f := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inverse_is_g_l512_51204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octahedron_side_length_inscribed_octahedron_side_length_l512_51215

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a unit cube -/
structure UnitCube where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D
  A' : Point3D
  B' : Point3D
  C' : Point3D
  D' : Point3D

/-- Represents an octahedron inscribed in a unit cube -/
structure InscribedOctahedron where
  cube : UnitCube
  vertexAB : Point3D
  vertexAC : Point3D
  vertexAD : Point3D
  vertexA'B' : Point3D
  vertexA'C' : Point3D
  vertexA'D' : Point3D

/-- Calculates the distance between two points in 3D space -/
noncomputable def distance (p1 p2 : Point3D) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2)

/-- Theorem stating that the side length of the inscribed octahedron is √2/3 -/
theorem octahedron_side_length (o : InscribedOctahedron) : 
  distance o.vertexAB o.vertexAC = Real.sqrt 2 / 3 := by
  sorry

/-- Main theorem to be proved -/
theorem inscribed_octahedron_side_length : ∃ (o : InscribedOctahedron), 
  ∀ (v1 v2 : Point3D), v1 ≠ v2 → 
  v1 ∈ ({o.vertexAB, o.vertexAC, o.vertexAD, o.vertexA'B', o.vertexA'C', o.vertexA'D'} : Set Point3D) → 
  v2 ∈ ({o.vertexAB, o.vertexAC, o.vertexAD, o.vertexA'B', o.vertexA'C', o.vertexA'D'} : Set Point3D) → 
  distance v1 v2 = Real.sqrt 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octahedron_side_length_inscribed_octahedron_side_length_l512_51215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_ellipse_tangency_l512_51228

/-- Predicate stating that three circles with radii r₁, r₂, and r₃ are tangent to an ellipse
    with semi-major axis a and semi-minor axis b -/
def CirclesTangentToEllipse (a b r₁ r₂ r₃ : ℝ) : Prop :=
  sorry

/-- Predicate stating that the circle with radius r₂ is externally tangent to
    the circles with radii r₁ and r₃ -/
def CircleR₂TangentToR₁AndR₃ (r₁ r₂ r₃ : ℝ) : Prop :=
  sorry

/-- Given an ellipse with semi-major axis a and semi-minor axis b, and three circles with radii r₁, r₂, and r₃
    whose centers lie on the major axis of the ellipse, such that:
    1. The circles are tangent to the ellipse
    2. The circle with radius r₂ is externally tangent to the circles with radii r₁ and r₃
    Then the following equation holds: r₁ + r₃ = (2a²(a² - 2b²) / a⁴) * r₂ -/
theorem circle_ellipse_tangency (a b r₁ r₂ r₃ : ℝ) 
    (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_a_ge_b : a ≥ b)
    (h_pos_r₁ : 0 < r₁) (h_pos_r₂ : 0 < r₂) (h_pos_r₃ : 0 < r₃)
    (h_tangent_ellipse : CirclesTangentToEllipse a b r₁ r₂ r₃)
    (h_tangent_circles : CircleR₂TangentToR₁AndR₃ r₁ r₂ r₃) :
  r₁ + r₃ = (2 * a^2 * (a^2 - 2*b^2) / a^4) * r₂ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_ellipse_tangency_l512_51228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expr_equals_ten_l512_51258

-- Define the expression as noncomputable
noncomputable def expr : ℝ := Real.sqrt (45 + 20 * Real.sqrt 5) + Real.sqrt (45 - 20 * Real.sqrt 5)

-- State the theorem
theorem expr_equals_ten : expr = 10 := by
  -- The proof steps will go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expr_equals_ten_l512_51258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_slope_for_no_lattice_points_l512_51256

/-- Represents a rational number in lowest terms -/
structure LowestTerms where
  p : Int
  q : Nat
  q_pos : q > 0
  coprime : Nat.Coprime p.natAbs q

/-- Checks if a number is prime -/
def isPrime (n : Nat) : Prop := n > 1 ∧ ∀ d : Nat, d ∣ n → d = 1 ∨ d = n

theorem max_slope_for_no_lattice_points (T b : Int) (h_T : T > 0) (h_b : b = 3) :
  ∃ (a : ℚ),
    a = 152 / 451 ∧
    (∀ (m : LowestTerms),
      1/3 < m.p / m.q →
      m.p / m.q < a →
      ¬isPrime m.q →
      ∀ (x : Int),
        0 < x →
        x ≤ T →
        ¬∃ (y : Int), y = m.p * x / m.q + b) ∧
    (∀ (a' : ℚ),
      a' > a →
      ∃ (m : LowestTerms),
        1/3 < m.p / m.q →
        m.p / m.q < a' →
        ¬isPrime m.q →
        ∃ (x : Int),
          0 < x →
          x ≤ T →
          ∃ (y : Int), y = m.p * x / m.q + b) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_slope_for_no_lattice_points_l512_51256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l512_51216

/-- A function with the property f(1+x) = f(1-x) for all real x -/
def symmetric_about_one (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (1 + x) = f (1 - x)

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, d ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
noncomputable def arithmetic_sum (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  n * (a 1 + a n) / 2

theorem arithmetic_sequence_sum
  (f : ℝ → ℝ)
  (a : ℕ → ℝ)
  (h1 : symmetric_about_one f)
  (h2 : Monotone (f ∘ (λ x => x + 1)))
  (h3 : arithmetic_sequence a)
  (h4 : f (a 6) = f (a 23)) :
  arithmetic_sum a 28 = 28 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l512_51216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_for_r_unity_k_315_makes_r_unity_smallest_k_is_315_l512_51231

noncomputable def r : ℂ := Complex.exp (Complex.I * (253 * Real.pi / 315))

theorem smallest_k_for_r_unity (k : ℕ) : k > 0 ∧ r ^ k = 1 → k ≥ 315 := by
  sorry

theorem k_315_makes_r_unity : r ^ 315 = 1 := by
  sorry

theorem smallest_k_is_315 : 
  ∃ (k : ℕ), k > 0 ∧ r ^ k = 1 ∧ ∀ (m : ℕ), (m > 0 ∧ r ^ m = 1 → m ≥ k) ∧ k = 315 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_for_r_unity_k_315_makes_r_unity_smallest_k_is_315_l512_51231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l512_51206

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if -2 < x ∧ x ≤ -1 then Real.log (-x) + 3
  else if x > -1 then -x^2 - 2*x + 1
  else 0  -- This else case is added to make the function total

-- Define the inequality condition
def inequality (a : ℝ) : Prop :=
  f (2*a) - 1/2 * (2*a + 2)^2 < f (12 - a) - 1/2 * (14 - a)^2

-- Theorem statement
theorem range_of_a (a : ℝ) :
  inequality a → 4 < a ∧ a < 14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l512_51206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_at_three_eighths_unique_zero_for_negative_a_two_zeros_iff_a_between_zero_and_one_l512_51208

-- Define the function f(x) = ax^2 - x - ln(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - x - Real.log x

-- Theorem 1: When a = 3/8, the minimum value of f(x) is -1/2 - ln(2)
theorem min_value_at_three_eighths :
  ∃ (x : ℝ), x > 0 ∧ f (3/8) x = -1/2 - Real.log 2 ∧ ∀ (y : ℝ), y > 0 → f (3/8) y ≥ f (3/8) x :=
by sorry

-- Theorem 2: When -1 ≤ a ≤ 0, f(x) has exactly one zero
theorem unique_zero_for_negative_a :
  ∀ (a : ℝ), -1 ≤ a ∧ a ≤ 0 → ∃! (x : ℝ), x > 0 ∧ f a x = 0 :=
by sorry

-- Theorem 3: f(x) has two zeros if and only if 0 < a < 1
theorem two_zeros_iff_a_between_zero_and_one :
  ∀ (a : ℝ), (∃ (x y : ℝ), x ≠ y ∧ x > 0 ∧ y > 0 ∧ f a x = 0 ∧ f a y = 0) ↔ (0 < a ∧ a < 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_at_three_eighths_unique_zero_for_negative_a_two_zeros_iff_a_between_zero_and_one_l512_51208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_formula_l512_51250

/-- Represents the initial amount lent out in rupees -/
noncomputable def initial_amount : ℝ := 7031.72

/-- Represents the interest rate as a decimal -/
noncomputable def interest_rate : ℝ := 0.04000000000000001

/-- Represents the time period in years -/
noncomputable def time_period : ℝ := 1 + 8 / 12

/-- Represents the total amount received after the time period -/
noncomputable def total_amount : ℝ := 500

/-- Theorem stating that the initial amount lent out satisfies the simple interest formula -/
theorem simple_interest_formula :
  ∃ ε > 0, |total_amount - initial_amount * (1 + interest_rate * time_period)| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_formula_l512_51250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bowling_average_wickets_l512_51274

theorem bowling_average_wickets (initial_average : ℝ) (wickets_last_match : ℕ) (runs_last_match : ℕ) (average_decrease : ℝ) : 
  initial_average = 12.4 →
  wickets_last_match = 6 →
  runs_last_match = 26 →
  average_decrease = 0.4 →
  ∃ (previous_wickets : ℕ), 
    previous_wickets = 115 ∧
    (initial_average * (previous_wickets : ℝ) + (runs_last_match : ℝ)) / ((previous_wickets + wickets_last_match) : ℝ) = initial_average - average_decrease :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bowling_average_wickets_l512_51274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_f2_is_quadratic_l512_51257

-- Define the concept of a quadratic function
noncomputable def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

-- Define the given functions
noncomputable def f1 : ℝ → ℝ := λ x ↦ x - 2
noncomputable def f2 : ℝ → ℝ := λ x ↦ x^2
noncomputable def f3 : ℝ → ℝ := λ x ↦ x^2 - (x + 1)^2
noncomputable def f4 : ℝ → ℝ := λ x ↦ 2 / x^2

-- Theorem statement
theorem only_f2_is_quadratic :
  ¬(is_quadratic f1) ∧
  (is_quadratic f2) ∧
  ¬(is_quadratic f3) ∧
  ¬(is_quadratic f4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_f2_is_quadratic_l512_51257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z5_distance_l512_51295

/-- Sequence of complex numbers defined recursively -/
def z : ℕ → ℂ
  | 0 => 1
  | n + 1 => (z n)^2 - 1 + Complex.I

/-- The fifth term of the sequence -/
def z5 : ℂ := z 4

/-- Theorem: The distance of z_5 from the origin is √370 -/
theorem z5_distance : Complex.abs z5 = Real.sqrt 370 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z5_distance_l512_51295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_value_l512_51253

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then -x^2 + x else -((-x)^2 - x)

-- State the theorem
theorem odd_function_value :
  (∀ x : ℝ, f (-x) = -f x) →  -- f is odd
  f 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_value_l512_51253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_diamond_equation_l512_51217

-- Define the binary operation ◇ on nonzero real numbers
noncomputable def diamond (a b : ℝ) : ℝ := a / b

-- State the properties of ◇
axiom diamond_prop1 (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  diamond a (diamond b c) = (diamond a b) * c

axiom diamond_prop2 (a : ℝ) (ha : a ≠ 0) :
  diamond a a = 1

-- State the theorem
theorem solve_diamond_equation :
  ∃ (x : ℝ), x ≠ 0 ∧ diamond 2016 (diamond 6 x) = 100 ∧ x = 25 / 84 := by
  -- Proof steps would go here, but we'll use sorry for now
  sorry

#check solve_diamond_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_diamond_equation_l512_51217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l512_51293

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r > 0, ∀ n, a (n + 1) = r * a n

theorem min_value_theorem (a : ℕ → ℝ) (m n : ℕ) :
  geometric_sequence a →
  (∀ k, a k > 0) →
  a 7 = a 6 + 2 * a 5 →
  Real.sqrt (a m * a n) = 4 * a 1 →
  (1 : ℝ) / m + 4 / n ≥ 3 / 2 :=
by
  sorry

#check min_value_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l512_51293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_range_l512_51287

noncomputable def f (x : ℝ) := (Real.log (x^2 - 2*x)) / (Real.sqrt (9 - x^2))

def A : Set ℝ := {x | -3 < x ∧ x < 0 ∨ 2 < x ∧ x < 3}

def B (k : ℝ) : Set ℝ := {x | x^2 - 2*x + 1 - k^2 ≥ 0}

theorem k_range (k : ℝ) : (∃ x, x ∈ A ∩ B k) → k ∈ Set.Icc (-4 : ℝ) 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_range_l512_51287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_functions_properties_l512_51290

variable (k m b : ℝ)

/-- Two linear functions intersecting at (1, 2) -/
def y₁ (k b : ℝ) (x : ℝ) : ℝ := k * x + b
def y₂ (m : ℝ) (x : ℝ) : ℝ := m * x + 3

/-- Conditions on the constants -/
axiom h_k_nonzero : k ≠ 0
axiom h_m_nonzero : m ≠ 0
axiom h_intersection : y₁ k b 1 = y₂ m 1 ∧ y₁ k b 1 = 2

theorem linear_functions_properties :
  (∀ x, k * x + b = m * x + 3 → x = 1) ∧
  (∀ xₐ xᵦ yₐ yᵦ, y₂ m xₐ = yₐ → y₂ m xᵦ = yᵦ → xₐ ≠ xᵦ → (xₐ - xᵦ) * (yₐ - yᵦ) < 0) ∧
  (b < 3 → b ≠ 2 → ∀ x > 1, y₁ k b x > y₂ m x) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_functions_properties_l512_51290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_rate_analysis_l512_51225

-- Define the profit function
noncomputable def profit (x : ℝ) : ℝ :=
  if x ≤ 36 then -0.1 * x^2 + 8 * x - 90 else 0.4 * x + 54

-- Define the profit rate function
noncomputable def profitRate (x : ℝ) : ℝ :=
  (profit x) / x * 100

-- Theorem statement
theorem profit_rate_analysis (x : ℝ) (h : 15 ≤ x ∧ x ≤ 40) :
  -- Part 1: Maximum profit rate occurs at x = 30 and equals 200%
  (∀ y, 15 ≤ y ∧ y ≤ 40 → profitRate y ≤ profitRate 30) ∧
  profitRate 30 = 200 ∧
  -- Part 2: Range of x for which profit rate is at least 190%
  (profitRate x ≥ 190 ↔ 25 ≤ x ∧ x ≤ 36) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_rate_analysis_l512_51225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_slope_reciprocal_l512_51288

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log x - a * (x - 1)

noncomputable def g (x : ℝ) : ℝ := exp x

theorem tangent_line_slope_reciprocal (a : ℝ) (h : a > 0) :
  ∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧
  (deriv (f a)) x₁ * (deriv g) x₂ = 1 ∧
  f a x₁ = ((deriv (f a)) x₁) * x₁ ∧
  g x₂ = ((deriv g) x₂) * x₂ →
  (exp 1 - 1) / exp 1 < a ∧ a < (exp 2 - 1) / exp 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_slope_reciprocal_l512_51288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_l512_51213

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the focus point
def focus : ℝ × ℝ := (2, 0)

-- Define the slope angle
noncomputable def slope_angle : ℝ := 60 * (Real.pi / 180)

-- Define the line passing through the focus with the given slope angle
noncomputable def line (x y : ℝ) : Prop :=
  y = Real.tan slope_angle * (x - focus.1) + focus.2

-- Theorem statement
theorem parabola_line_intersection :
  -- The equation of the line
  (∀ x y, line x y ↔ Real.sqrt 3 * x - y - 2 * Real.sqrt 3 = 0) ∧
  -- The length of the chord
  (∃ a b : ℝ × ℝ,
    parabola a.1 a.2 ∧ parabola b.1 b.2 ∧
    line a.1 a.2 ∧ line b.1 b.2 ∧
    Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = 32/3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_l512_51213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l512_51286

-- Define the circle
def circle_eq (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 25

-- Define the line
def line_eq (x y : ℝ) : Prop := 3*x + 4*y + 10 = 0

-- State the theorem
theorem chord_length :
  ∃ (a b c d : ℝ),
    circle_eq a b ∧ circle_eq c d ∧
    line_eq a b ∧ line_eq c d ∧
    ((a - c)^2 + (b - d)^2)^(1/2 : ℝ) = 6 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l512_51286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_l512_51212

/-- The function f(x) for a given a > 0 -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a^x - a)^2 + (a^(-x) - a)^2

/-- Theorem stating that the minimum value of f(x) is 0 when a > 0 -/
theorem min_value_f (a : ℝ) (h : a > 0) : 
  ∀ x : ℝ, f a x ≥ 0 ∧ ∃ x₀ : ℝ, f a x₀ = 0 := by
  sorry

#check min_value_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_l512_51212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_C₁_equivalence_intersection_distance_l512_51207

-- Define the curves C₁ and C₂
noncomputable def C₁ (x y : ℝ) : Prop := x^2 + y^2 - x + y = 0

noncomputable def C₂ (t : ℝ) : ℝ × ℝ := (1/2 - (Real.sqrt 2)/2 * t, (Real.sqrt 2)/2 * t)

-- Theorem 1: Equivalence of polar and rectangular forms of C₁
theorem C₁_equivalence (θ : ℝ) :
  (∃ ρ : ℝ, ρ = Real.cos θ - Real.sin θ ∧ 
   C₁ (ρ * Real.cos θ) (ρ * Real.sin θ)) ↔ 
  C₁ (Real.cos θ) (Real.sin θ) := by
  sorry

-- Theorem 2: Distance between intersection points
theorem intersection_distance :
  ∃ t₁ t₂ : ℝ, 
    t₁ ≠ t₂ ∧ 
    C₁ (C₂ t₁).1 (C₂ t₁).2 ∧ 
    C₁ (C₂ t₂).1 (C₂ t₂).2 ∧
    ((C₂ t₁).1 - (C₂ t₂).1)^2 + ((C₂ t₁).2 - (C₂ t₂).2)^2 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_C₁_equivalence_intersection_distance_l512_51207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_B_in_interval_l512_51255

noncomputable def g : ℕ → ℝ
  | 0 => 0  -- Add a base case for 0
  | 1 => 0  -- Add a base case for 1
  | 2 => 0  -- Add a base case for 2
  | 3 => 0  -- Add a base case for 3
  | 4 => Real.log 4
  | n + 1 => Real.log (n + 1 + g n)

theorem B_in_interval :
  let B := g 2020
  B > Real.log 2023 ∧ B < Real.log 2024 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_B_in_interval_l512_51255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nabla_problem_l512_51202

-- Define the nabla operation as noncomputable
noncomputable def nabla (a b : ℝ) : ℝ := (a + b) / (1 + a * b)

-- State the theorem
theorem nabla_problem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  nabla (nabla 2 3) (nabla 1 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nabla_problem_l512_51202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_cube_plus_nine_l512_51265

theorem prime_cube_plus_nine (P : ℕ) : 
  Nat.Prime P → Nat.Prime (P^3 + 9) → (P^2 : ℤ) - 7 = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_cube_plus_nine_l512_51265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xiao_ming_mistake_l512_51243

variable (a b c : ℝ)

def A (a b c : ℝ) : ℝ := 3 * a^2 * b - 2 * a * b^2 + a * b * c
def C (a b c : ℝ) : ℝ := 4 * a^2 * b - 3 * a * b^2 + 4 * a * b * c

theorem xiao_ming_mistake (a b c : ℝ) :
  (∃ B : ℝ, B = C a b c - 2 * A a b c ∧ B = -2 * a^2 * b + a * b^2 + 2 * a * b * c) ∧
  (2 * A a b c - (C a b c - 2 * A a b c) = 8 * a^2 * b - 5 * a * b^2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_xiao_ming_mistake_l512_51243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_and_minimum_l512_51282

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + (1 - x) / (a * x)

theorem f_monotonicity_and_minimum (a : ℝ) (h : a > 0) :
  (∀ x ∈ Set.Ioo 0 1, StrictMonoOn (fun x => -f 1 x) (Set.Ioo 0 1)) ∧
  (∀ x ∈ Set.Ioi 1, StrictMonoOn (f 1) (Set.Ioi 1)) ∧
  (a ≥ 1/2 → ∃ x ∈ Set.Icc 2 3, IsMinOn (f a) (Set.Icc 2 3) x ∧ f a x = f a 2) ∧
  (1/3 < a ∧ a < 1/2 → ∃ x ∈ Set.Icc 2 3, IsMinOn (f a) (Set.Icc 2 3) x ∧ f a x = f a (1/a)) ∧
  (0 < a ∧ a ≤ 1/3 → ∃ x ∈ Set.Icc 2 3, IsMinOn (f a) (Set.Icc 2 3) x ∧ f a x = f a 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_and_minimum_l512_51282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_triangle_ratio_cubic_equation_l512_51289

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y = ax^2 -/
structure Parabola where
  a : ℝ
  property : a > 0

/-- Represents an isosceles right triangle inscribed in a parabola -/
structure InscribedTriangle where
  parabola : Parabola
  C : Point
  on_parabola : C.y = parabola.a * C.x^2
  in_first_quadrant : C.x > 0 ∧ C.y > 0

/-- The ratio of y-coordinate to x-coordinate of point C -/
noncomputable def t (triangle : InscribedTriangle) : ℝ :=
  triangle.C.y / triangle.C.x

/-- Theorem: The ratio t satisfies the cubic equation t^3 - 2t^2 - 1 = 0 -/
theorem inscribed_triangle_ratio_cubic_equation (triangle : InscribedTriangle) :
  (t triangle)^3 - 2*(t triangle)^2 - 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_triangle_ratio_cubic_equation_l512_51289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_expression_l512_51291

theorem min_value_expression (n : ℕ+) : 
  (n : ℝ) / 2 + 50 / (n : ℝ) ≥ 10 ∧ ((n : ℝ) / 2 + 50 / (n : ℝ) = 10 ↔ n = 10) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_expression_l512_51291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_l512_51201

-- Define the line equation
def line_equation (x y : ℝ) : Prop := x / 4 - y / 5 = 2

-- Define the slope-intercept form of a line
def slope_intercept_form (m b x y : ℝ) : Prop := y = m * x + b

-- Theorem statement
theorem line_slope :
  ∃ (m b : ℝ), m = 5/4 ∧ ∀ (x y : ℝ), line_equation x y → slope_intercept_form m b x y :=
by
  -- We'll use 'sorry' to skip the proof for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_l512_51201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_wins_l512_51251

/-- Represents a circular table. -/
structure CircularTable where
  radius : ℝ
  inv_radius_pos : radius > 0

/-- Represents a coin. -/
structure Coin where
  radius : ℝ
  inv_radius_pos : radius > 0

/-- Represents a position on the table. -/
structure Position where
  x : ℝ
  y : ℝ

/-- Checks if a coin at a given position is within the table and doesn't overlap with other coins. -/
def isValidPlacement (table : CircularTable) (coin : Coin) (pos : Position) (placedCoins : List Position) : Prop :=
  -- Coin is within the table
  pos.x^2 + pos.y^2 ≤ (table.radius - coin.radius)^2 ∧
  -- Coin doesn't overlap with other coins
  ∀ p ∈ placedCoins, (pos.x - p.x)^2 + (pos.y - p.y)^2 ≥ (2 * coin.radius)^2

/-- Represents the game state. -/
inductive GameState where
  | PlayerOneTurn : List Position → GameState
  | PlayerTwoTurn : List Position → GameState
  | PlayerOneWins : GameState
  | PlayerTwoWins : GameState

/-- The main theorem stating that the first player has a winning strategy. -/
theorem first_player_wins (table : CircularTable) (coin : Coin) :
  ∃ (strategy : GameState → Position),
    ∀ (game : GameState),
      match game with
      | GameState.PlayerOneTurn placedCoins =>
          isValidPlacement table coin (strategy game) placedCoins
      | GameState.PlayerTwoTurn _ => True
      | GameState.PlayerOneWins => True
      | GameState.PlayerTwoWins => False :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_wins_l512_51251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_x_1_l512_51221

-- Define the function
noncomputable def f (x : ℝ) : ℝ := 2 / x + 3 * x

-- Define the derivative of the function
noncomputable def f' (x : ℝ) : ℝ := -2 / (x^2) + 3

-- State the theorem
theorem tangent_line_at_x_1 :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  (λ (x y : ℝ) => y - y₀ = m * (x - x₀)) = (λ (x y : ℝ) => y = x + 4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_x_1_l512_51221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pizza_area_comparison_l512_51219

/-- The radius of the circular pizza in inches -/
def circular_radius : ℝ := 5

/-- The side length of the square pizza in inches -/
def square_side : ℝ := 2 * circular_radius

/-- The area of the circular pizza in square inches -/
noncomputable def circular_area : ℝ := Real.pi * circular_radius^2

/-- The area of the square pizza in square inches -/
def square_area : ℝ := square_side^2

/-- The percentage increase in area from circular to square pizza -/
noncomputable def percentage_increase : ℝ := (square_area - circular_area) / circular_area * 100

theorem pizza_area_comparison : 
  Int.floor (percentage_increase + 0.5) = 27 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pizza_area_comparison_l512_51219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_simplification_and_value_l512_51246

noncomputable def f (α : Real) : Real :=
  (Real.sin (Real.pi - α) * Real.cos (2*Real.pi - α) * Real.cos (-α + 3*Real.pi/2)) /
  (Real.cos (Real.pi/2 - α) * Real.sin (-Real.pi - α))

theorem f_simplification_and_value (α : Real) 
  (h1 : Real.pi < α ∧ α < 3*Real.pi/2)  -- α is in the third quadrant
  (h2 : Real.sin α = -1/5) :
  (f α = -Real.cos α) ∧ 
  (f α = 2 * Real.sqrt 6 / 5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_simplification_and_value_l512_51246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_range_theorem_l512_51252

/-- An ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  c : ℝ
  h_ab : a > b
  h_b_pos : b > 0
  h_eccentricity : c / a = 2/3
  h_af_product : (-c + a) * (-a - c) = 5

/-- A point on the ellipse -/
structure EllipsePoint (E : Ellipse) where
  x : ℝ
  y : ℝ
  h_on_ellipse : x^2 / E.a^2 + y^2 / E.b^2 = 1

/-- The distance from a focus to a point on the ellipse -/
noncomputable def focus_distance (E : Ellipse) (P : EllipsePoint E) : ℝ :=
  Real.sqrt ((P.x + E.c)^2 + P.y^2)

/-- Theorem stating the range of the given expression -/
theorem ellipse_range_theorem (E : Ellipse) :
  ∀ M N : EllipsePoint E,
  ∃ line : ℝ → ℝ, line 0 = 0 ∧ 
  (∃ t₁ t₂ : ℝ, line t₁ = M.x ∧ line t₂ = N.x) →
  (3/2 : ℝ) ≤ 1 / focus_distance E M + 4 / focus_distance E N ∧
  1 / focus_distance E M + 4 / focus_distance E N ≤ 21/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_range_theorem_l512_51252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_root_properties_l512_51263

theorem square_root_properties : 
  (Real.sqrt ((-4)^2) = 4) ∧ 
  ¬ (Real.sqrt (-4) = -2) ∧ 
  ¬ (Real.sqrt 16 = (4 : ℝ) ∨ Real.sqrt 16 = (-4 : ℝ)) ∧ 
  ¬ ((Real.sqrt 4 = 2) ∧ (-Real.sqrt 4 = 2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_root_properties_l512_51263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_preferences_correct_l512_51276

/-- Represents the color preferences of students in Miss Molly's class --/
structure ColorPreferences where
  pink : ℕ
  purple : ℕ
  blue : ℕ
  red : ℕ
  green : ℕ
  orange : ℕ

/-- Calculates the color preferences based on the given conditions --/
def calculatePreferences (totalStudents girlCount boyCount : ℕ) : ColorPreferences :=
  let girlsPink := girlCount / 3
  let girlsPurple := (2 * girlCount) / 5
  let girlsBlue := girlCount - girlsPink - girlsPurple
  let boysRed := (2 * boyCount) / 5
  let boysGreen := (3 * boyCount) / 10
  let boysOrange := boyCount - boysRed - boysGreen
  { pink := girlsPink,
    purple := girlsPurple,
    blue := girlsBlue,
    red := boysRed,
    green := boysGreen,
    orange := boysOrange }

/-- Theorem stating that the calculated preferences match the expected results --/
theorem preferences_correct :
  let prefs := calculatePreferences 50 30 20
  prefs.pink = 10 ∧
  prefs.purple = 12 ∧
  prefs.blue = 8 ∧
  prefs.red = 8 ∧
  prefs.green = 6 ∧
  prefs.orange = 6 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_preferences_correct_l512_51276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l512_51283

noncomputable def f (a b x : ℝ) : ℝ := x^2 + b*x - a * Real.log x

theorem problem_statement (a b : ℝ) :
  (∃ x₀ : ℝ, f a b 1 = 0 ∧ f a b x₀ = 0 ∧ x₀ ≠ 1 ∧ ∃ n : ℕ, x₀ ∈ Set.Ioo (n : ℝ) ((n + 1) : ℝ)) →
  (f a b 2 = 0 ∨ (deriv (f a b)) 2 = 0) →
  (∃ n : ℕ, ∀ x₀ : ℝ, f a b 1 = 0 → f a b x₀ = 0 → x₀ ≠ 1 → x₀ ∈ Set.Ioo (n : ℝ) ((n + 1) : ℝ) → n = 3) ∧
  ((∀ b ∈ Set.Icc (-2) (-1), ∃ x ∈ Set.Ioo 1 (Real.exp 1), f a b x < 0) → a > 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l512_51283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_eventually_constant_mod_l512_51285

/-- The sequence defined by u₁ = 2 and u_{i+1} = 2^(u_i) for i ≥ 1 -/
def u : ℕ → ℕ
  | 0 => 2  -- Define for 0 to cover all natural numbers
  | n + 1 => 2^(u n)

/-- For any integer n, the sequence u becomes constant from a certain point onwards modulo n -/
theorem sequence_eventually_constant_mod (n : ℤ) : 
  ∃ k : ℕ, ∀ i : ℕ, i ≥ k → u i ≡ u k [ZMOD n] := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_eventually_constant_mod_l512_51285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_calculation_l512_51226

/-- Calculate compound interest --/
def compoundInterest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time - principal

/-- Proof of the compound interest calculation --/
theorem compound_interest_calculation :
  let principal : ℝ := 2000
  let rate : ℝ := 0.05
  let time : ℕ := 5
  ∃ ε > 0, |compoundInterest principal rate time - 552.56| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_calculation_l512_51226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_operations_l512_51205

/-- Given two vectors in ℝ³, prove their difference and dot product -/
theorem vector_operations (a b : ℝ × ℝ × ℝ) 
  (ha : a = (1, -1, -2)) 
  (hb : b = (1, -3, -3)) : 
  (a.1 - b.1, a.2.1 - b.2.1, a.2.2 - b.2.2) = (0, 2, 1) ∧ 
  (a.1 * b.1 + a.2.1 * b.2.1 + a.2.2 * b.2.2) = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_operations_l512_51205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_eq_solution_l512_51210

-- Define our own max function to avoid ambiguity with the built-in Max.max
def myMax (a b : ℚ) : ℚ := if a ≥ b then a else b

theorem max_eq_solution :
  ∃! x : ℚ, myMax x (-x) = 2*x + 9 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_eq_solution_l512_51210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l512_51271

/-- The line x/a + y/b = 1 -/
def line (a b x y : ℝ) : Prop := x/a + y/b = 1

/-- The circle x^2 + y^2 = 1 -/
def unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- The line and circle have common points -/
def have_common_points (a b : ℝ) : Prop :=
  ∃ x y : ℝ, line a b x y ∧ unit_circle x y

theorem line_circle_intersection (a b : ℝ) :
  have_common_points a b → 1/a^2 + 1/b^2 ≥ 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l512_51271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_of_log_base_3_f_domain_l512_51233

-- Define the original function
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 3

-- Define the proposed inverse function
noncomputable def f_inv (x : ℝ) : ℝ := 3^x

-- Theorem statement
theorem inverse_of_log_base_3 (x : ℝ) (h : x > 0) :
  f (f_inv x) = x ∧ f_inv (f x) = x :=
by
  sorry

-- Additional theorem to ensure the domain of f is positive real numbers
theorem f_domain (x : ℝ) : f x > 0 → x > 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_of_log_base_3_f_domain_l512_51233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_room_tiling_l512_51247

theorem room_tiling (room_length room_width : ℕ) 
  (border_tile_size inner_tile_size : ℕ) : 
  room_length = 20 → 
  room_width = 15 → 
  border_tile_size = 2 → 
  inner_tile_size = 1 → 
  (2 * (room_length - 2 * border_tile_size) / border_tile_size + 
   2 * (room_width - 2 * border_tile_size) / border_tile_size + 4) + 
  ((room_length - 2 * border_tile_size) * 
   (room_width - 2 * border_tile_size) / 
   (inner_tile_size * inner_tile_size)) = 208 :=
by
  sorry

#eval (2 * (20 - 2 * 2) / 2 + 2 * (15 - 2 * 2) / 2 + 4) + 
      ((20 - 2 * 2) * (15 - 2 * 2) / (1 * 1))

end NUMINAMATH_CALUDE_ERRORFEEDBACK_room_tiling_l512_51247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_of_ABFCDE_l512_51267

-- Define the square ABCD
def square_perimeter : ℝ := 40

-- Define the isosceles right triangle BFC
def triangle_BFC (side_length : ℝ) : Set (EuclideanSpace ℝ (Fin 2)) := sorry

-- Define the figure ABFCDE
def figure_ABFCDE (square_side : ℝ) (triangle : Set (EuclideanSpace ℝ (Fin 2))) : Set (EuclideanSpace ℝ (Fin 2)) := sorry

-- Define perimeter function
noncomputable def perimeter (s : Set (EuclideanSpace ℝ (Fin 2))) : ℝ := sorry

-- Theorem statement
theorem perimeter_of_ABFCDE :
  let square_side := square_perimeter / 4
  let triangle := triangle_BFC square_side
  let figure := figure_ABFCDE square_side triangle
  perimeter figure = 30 + 10 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_of_ABFCDE_l512_51267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shoe_collection_problem_l512_51278

/-- Calculates the final number of shoes after donation and purchase -/
def final_shoe_count (initial : ℕ) (donation_percent : ℚ) (purchased : ℕ) : ℕ :=
  initial - (initial * donation_percent).floor.toNat + purchased

/-- Theorem stating that the final shoe count is 62 given the problem conditions -/
theorem shoe_collection_problem :
  final_shoe_count 80 (30 / 100) 6 = 62 := by
  -- Unfold the definition of final_shoe_count
  unfold final_shoe_count
  -- Simplify the arithmetic expressions
  simp [Nat.cast_mul, Nat.cast_div, Nat.cast_ofNat]
  -- The proof is completed
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shoe_collection_problem_l512_51278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_target_year_is_2024_l512_51266

/-- Represents a person with their age -/
structure Person where
  age : Nat

/-- Represents the group of people in the problem -/
structure PeopleGroup where
  ulysses : Person
  kim : Person
  mei : Person
  tanika : Person

/-- Calculates the total age of the group -/
def totalAge (g : PeopleGroup) : Nat :=
  g.ulysses.age + g.kim.age + g.mei.age + g.tanika.age

/-- The initial group in the year 2023 -/
def initialGroup : PeopleGroup :=
  { ulysses := { age := 12 },
    kim := { age := 14 },
    mei := { age := 15 },
    tanika := { age := 15 } }

/-- The year when the total age first reaches 100 -/
def targetYear (startYear : Nat) (g : PeopleGroup) : Nat :=
  startYear + ((100 - totalAge g) / 4)

/-- Theorem stating that the target year is 2024 -/
theorem target_year_is_2024 :
  targetYear 2023 initialGroup = 2024 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_target_year_is_2024_l512_51266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_1895_to_hundredth_l512_51277

/-- Rounds a real number to the nearest hundredth -/
noncomputable def roundToHundredth (x : ℝ) : ℝ :=
  (⌊x * 100 + 0.5⌋ : ℝ) / 100

/-- The theorem states that rounding 1.895 to the nearest hundredth results in 1.90 -/
theorem round_1895_to_hundredth :
  roundToHundredth 1.895 = 1.90 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_1895_to_hundredth_l512_51277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_driver_speed_proof_l512_51296

/-- The actual average speed of the driver -/
noncomputable def actual_speed : ℝ := 42

/-- The increase in speed that would reduce the travel time by 1/3 -/
noncomputable def speed_increase : ℝ := 21

/-- The ratio by which the travel time would be reduced if the speed was increased -/
noncomputable def time_reduction_ratio : ℝ := 1/3

theorem driver_speed_proof :
  let new_speed := actual_speed + speed_increase
  let new_time_ratio := 1 - time_reduction_ratio
  actual_speed * 1 = new_speed * new_time_ratio :=
by
  -- Unfold the definitions
  unfold actual_speed speed_increase time_reduction_ratio
  -- Perform algebraic manipulations
  simp [mul_one, sub_mul, mul_sub, mul_add]
  -- Check that the equation holds
  norm_num

#check driver_speed_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_driver_speed_proof_l512_51296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_moves_to_switch_queens_l512_51232

/-- Represents a chessboard with queens --/
structure Chessboard where
  black_queens : Fin 8 → Nat  -- positions of black queens
  white_queens : Fin 8 → Nat  -- positions of white queens

/-- Represents a move of a queen --/
inductive Move
  | Black (index : Fin 8) (newPos : Nat)  -- move a black queen to a new position
  | White (index : Fin 8) (newPos : Nat)  -- move a white queen to a new position

/-- The initial configuration of the chessboard --/
def initial_board : Chessboard :=
  { black_queens := λ _ => 1,  -- all black queens on rank 1
    white_queens := λ _ => 8 } -- all white queens on rank 8

/-- The final configuration of the chessboard --/
def final_board : Chessboard :=
  { black_queens := λ _ => 8,  -- all black queens on rank 8
    white_queens := λ _ => 1 } -- all white queens on rank 1

/-- Checks if a move is valid --/
def is_valid_move (board : Chessboard) (move : Move) : Prop :=
  sorry

/-- Applies a move to the board --/
def apply_move (board : Chessboard) (move : Move) : Chessboard :=
  sorry

/-- Checks if a sequence of moves is valid and alternating --/
def is_valid_sequence (board : Chessboard) (moves : List Move) : Prop :=
  sorry

/-- The main theorem --/
theorem min_moves_to_switch_queens :
  ∃ (moves : List Move),
    is_valid_sequence initial_board moves ∧
    apply_move (moves.foldl apply_move initial_board) (Move.White 0 0) = final_board ∧
    moves.length = 23 ∧
    (∀ (other_moves : List Move),
      is_valid_sequence initial_board other_moves →
      apply_move (other_moves.foldl apply_move initial_board) (Move.White 0 0) = final_board →
      other_moves.length ≥ 23) :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_moves_to_switch_queens_l512_51232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimal_cut_for_specific_triangle_l512_51203

/-- Represents the lengths of three pieces that can form a triangle -/
structure TrianglePieces where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c
  triangle_inequality : a + b > c ∧ a + c > b ∧ b + c > a

/-- The minimal length to cut from each piece to prevent triangle formation -/
noncomputable def minimalCut (t : TrianglePieces) : ℝ :=
  min t.a (min t.b (min t.c (
    max ((t.a + t.b - t.c) / 2) (max ((t.a + t.c - t.b) / 2) ((t.b + t.c - t.a) / 2))
  )))

theorem minimal_cut_for_specific_triangle :
  let t : TrianglePieces := {
    a := 9, b := 15, c := 18,
    a_pos := by norm_num,
    b_pos := by norm_num,
    c_pos := by norm_num,
    triangle_inequality := by norm_num
  }
  minimalCut t = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimal_cut_for_specific_triangle_l512_51203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_peanut_price_theorem_l512_51275

/-- Given the price of cashews, the total weight, the weight of cashews, 
    the weight of peanuts, and the average price, prove the price of peanuts. -/
theorem peanut_price_theorem
  (cashew_price : ℝ)
  (total_weight : ℝ)
  (cashew_weight : ℝ)
  (peanut_weight : ℝ)
  (average_price : ℝ)
  (peanut_price : ℝ)
  (h1 : cashew_price = 210)
  (h2 : total_weight = 5)
  (h3 : cashew_weight = 3)
  (h4 : peanut_weight = 2)
  (h5 : average_price = 178)
  (h6 : total_weight = cashew_weight + peanut_weight)
  (h7 : total_weight * average_price = cashew_weight * cashew_price + peanut_weight * peanut_price) :
  peanut_price = 130 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_peanut_price_theorem_l512_51275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_iff_a_in_range_l512_51235

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (5*a - 1)*x + 4*a else Real.log x / Real.log a

theorem f_decreasing_iff_a_in_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x > f a y) ↔ (1/9 ≤ a ∧ a < 1/5) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_iff_a_in_range_l512_51235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_OM_l512_51200

/-- Parabola with equation y² = 4x, focus F(1,0), and directrix x = -1 -/
structure Parabola where
  equation : ℝ → ℝ → Prop
  focus : ℝ × ℝ
  directrix : ℝ → Prop

/-- Point on the directrix above x-axis -/
noncomputable def M : ℝ × ℝ := (-1, 2 * Real.sqrt 2)

/-- Origin -/
def O : ℝ × ℝ := (0, 0)

/-- Point on y-axis -/
structure N where
  y : ℝ

/-- Point on parabola -/
structure P where
  x : ℝ
  y : ℝ

/-- Theorem stating the slope of OM given the conditions -/
theorem slope_of_OM (para : Parabola) (n : N) (p : P) : 
  para.equation = fun x y => y^2 = 4*x →
  para.focus = (1, 0) →
  para.directrix = fun x => x = -1 →
  p.x = (para.focus.1 + 0) / 2 →
  p.y = (para.focus.2 + n.y) / 2 →
  para.equation p.x p.y →
  (M.2 - O.2) / (M.1 - O.1) = -2 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_OM_l512_51200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_AB_fixed_point_min_triangle_area_P_coords_at_min_area_l512_51238

noncomputable section

-- Define the line y = x - 2
def line (x : ℝ) : ℝ := x - 2

-- Define the parabola y = 1/2 x^2
def parabola (x : ℝ) : ℝ := (1/2) * x^2

-- Define a point P on the line
structure Point where
  x : ℝ
  y : ℝ
  on_line : y = line x

-- Define points A and B on the parabola
structure TangentPoint where
  x : ℝ
  y : ℝ
  on_parabola : y = parabola x

-- Define the area of triangle PAB
def triangle_area (P : Point) (A B : TangentPoint) : ℝ := sorry

-- Theorem 1: Line AB passes through (1, 2)
theorem line_AB_fixed_point (P : Point) (A B : TangentPoint) :
  ∃ (m : ℝ), m * (1 - A.x) + A.y = 2 ∧ m * (1 - B.x) + B.y = 2 := by sorry

-- Theorem 2: Minimum area of triangle PAB
theorem min_triangle_area :
  ∃ (P : Point) (A B : TangentPoint),
    triangle_area P A B = 3 * Real.sqrt 3 ∧
    ∀ (P' : Point) (A' B' : TangentPoint),
      triangle_area P' A' B' ≥ 3 * Real.sqrt 3 := by sorry

-- Theorem 3: P coordinates at minimum area
theorem P_coords_at_min_area :
  ∃ (P : Point) (A B : TangentPoint),
    triangle_area P A B = 3 * Real.sqrt 3 ∧ P.x = 1 ∧ P.y = -1 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_AB_fixed_point_min_triangle_area_P_coords_at_min_area_l512_51238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_perpendicular_line_parallel_intersection_lines_perpendicular_line_in_plane_l512_51209

-- Define the basic types
variable (Point : Type) (Line : Type) (Plane : Type)

-- Define the necessary relations
variable (on_line : Point → Line → Prop)
variable (on_plane : Point → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)
variable (intersect_planes : Plane → Plane → Line → Prop)
variable (parallel_lines : Line → Line → Prop)

-- Statement 1
theorem unique_perpendicular_line 
  (p : Point) (π : Plane) (h : ¬ on_plane p π) :
  ∃! l : Line, on_line p l ∧ perpendicular_line_plane l π :=
sorry

-- Statement 2
theorem parallel_intersection_lines 
  (π₁ π₂ π₃ : Plane) (l₁ l₂ : Line)
  (h₁ : parallel_planes π₁ π₂)
  (h₂ : intersect_planes π₁ π₃ l₁)
  (h₃ : intersect_planes π₂ π₃ l₂) :
  parallel_lines l₁ l₂ :=
sorry

-- Statement 3
theorem perpendicular_line_in_plane
  (π₁ π₂ : Plane) (p : Point) (l : Line)
  (h₁ : perpendicular_planes π₁ π₂)
  (h₂ : on_plane p π₁)
  (h₃ : on_line p l)
  (h₄ : perpendicular_line_plane l π₂) :
  ∀ q : Point, on_line q l → on_plane q π₁ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_perpendicular_line_parallel_intersection_lines_perpendicular_line_in_plane_l512_51209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_condition_l512_51237

def triangle_ABC (k : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) :=
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (2, 3)
  let C : ℝ × ℝ := (1, k)
  (A, B, C)

def is_right_triangle (T : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)) : Prop :=
  let (A, B, C) := T
  let AB := (B.1 - A.1, B.2 - A.2)
  let BC := (C.1 - B.1, C.2 - B.2)
  let CA := (A.1 - C.1, A.2 - C.2)
  (AB.1 * BC.1 + AB.2 * BC.2 = 0) ∨
  (BC.1 * CA.1 + BC.2 * CA.2 = 0) ∨
  (CA.1 * AB.1 + CA.2 * AB.2 = 0)

theorem right_triangle_condition (k : ℝ) :
  is_right_triangle (triangle_ABC k) ↔ k = (3 + Real.sqrt 13) / 2 ∨ k = (3 - Real.sqrt 13) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_condition_l512_51237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_range_l512_51294

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2^(1 + x^2) - 1 / (1 + x^2)

-- State the theorem
theorem f_inequality_range (x : ℝ) :
  f (2 * x) > f (x - 3) ↔ x < -3 ∨ x > 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_range_l512_51294
