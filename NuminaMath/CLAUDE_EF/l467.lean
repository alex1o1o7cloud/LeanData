import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_diff_and_min_count_l467_46784

-- Define the function f_M
def f_M (M : Finset ℕ) (x : ℕ) : Int :=
  if x ∈ M then -1 else 1

-- Define the symmetric difference operation
def symmetricDiff (M N : Finset ℕ) : Finset ℕ :=
  (M ∪ N) \ (M ∩ N)

-- Given sets A and B
def A : Finset ℕ := {2, 4, 6, 8, 10}
def B : Finset ℕ := {1, 2, 4, 8, 16}

-- Theorem statement
theorem symmetric_diff_and_min_count :
  (symmetricDiff A B = {1, 6, 10, 16}) ∧
  (∃! (n : ℕ), n = (Finset.filter (fun X => 
    (Finset.card (symmetricDiff X A) + Finset.card (symmetricDiff X B)) =
    (Finset.filter (fun Y => 
      (Finset.card (symmetricDiff Y A) + Finset.card (symmetricDiff Y B)) ≤
      (Finset.card (symmetricDiff X A) + Finset.card (symmetricDiff X B)))
    (Finset.powerset (A ∪ B))).card
  ) (Finset.powerset (A ∪ B))).card ∧ n = 16) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_diff_and_min_count_l467_46784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_G_equals_2F_l467_46702

-- Define F as a noncomputable function of x
noncomputable def F (x : ℝ) : ℝ := Real.log ((1 + x) / (1 - x))

-- Define G as a noncomputable function of x
noncomputable def G (x : ℝ) : ℝ := Real.log ((1 + (2 * x) / (1 + x^2)) / (1 - (2 * x) / (1 + x^2)))

-- Theorem statement
theorem G_equals_2F (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) (h3 : 1 + x^2 ≠ 0) : G x = 2 * F x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_G_equals_2F_l467_46702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_triangle_weight_l467_46701

/-- Represents a right-angled triangle with given leg lengths and weight -/
structure RightTriangle where
  leg1 : ℝ
  leg2 : ℝ
  weight : ℝ

/-- Calculates the area of a right-angled triangle -/
noncomputable def triangleArea (t : RightTriangle) : ℝ :=
  t.leg1 * t.leg2 / 2

/-- Theorem: Weight of the second triangle given the properties of both triangles -/
theorem second_triangle_weight
  (t1 : RightTriangle)
  (t2 : RightTriangle)
  (h1 : t1.leg1 = 3 ∧ t1.leg2 = 4 ∧ t1.weight = 10)
  (h2 : t2.leg1 = 5 ∧ t2.leg2 = 12)
  (h_uniform : ∃ (k : ℝ), t1.weight = k * triangleArea t1 ∧ t2.weight = k * triangleArea t2) :
  t2.weight = 50 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_triangle_weight_l467_46701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_pairs_with_unit_product_solutions_l467_46786

theorem infinite_pairs_with_unit_product_solutions :
  ∃ f : ℕ → ℤ × ℤ,
    Function.Injective f ∧
    ∀ n : ℕ,
      ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ * x₂ = 1 ∧ 
        x₁^2012 = (f n).1 * x₁ + (f n).2 ∧ 
        x₂^2012 = (f n).1 * x₂ + (f n).2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_pairs_with_unit_product_solutions_l467_46786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_sum_l467_46744

-- Define the curves C1 and C2
noncomputable def C1 (t : ℝ) : ℝ × ℝ := (1 + (1/2) * t, (Real.sqrt 3 / 2) * t)

noncomputable def C2 (θ : ℝ) : ℝ × ℝ := 
  let ρ := Real.sqrt (12 / (3 + Real.sin θ * Real.sin θ))
  (ρ * Real.cos θ, ρ * Real.sin θ)

-- Define point F
def F : ℝ × ℝ := (1, 0)

-- Define the theorem
theorem intersection_distance_sum : 
  ∃ (t₁ t₂ : ℝ), 
    (C1 t₁ ∈ Set.range C2) ∧ 
    (C1 t₂ ∈ Set.range C2) ∧ 
    (t₁ ≠ t₂) →
    (1 / dist F (C1 t₁)) + (1 / dist F (C1 t₂)) = 4/3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_sum_l467_46744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seating_theorem_l467_46756

/-- The number of ways to arrange n distinct objects in a circle --/
def circularPermutations (n : ℕ) : ℕ := (n - 1).factorial

/-- The number of ways to seat 7 people around a round table with 2 specific people always next to each other --/
def seatingArrangements : ℕ :=
  let n : ℕ := 7  -- Total number of people
  let m : ℕ := 2  -- Number of people who must sit together
  (circularPermutations (n - m + 1)) * m.factorial

theorem seating_theorem : seatingArrangements = 240 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seating_theorem_l467_46756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l467_46799

/-- Given a parabola y² = 2px (p > 0), a line passing through its focus with slope -1
    intersects the parabola at points A and B. If the x-coordinate of the midpoint of AB is 3,
    then the equation of the directrix of this parabola is x = -1. -/
theorem parabola_directrix (p : ℝ) (A B : ℝ × ℝ) (h1 : p > 0) :
  (∀ x y : ℝ, y^2 = 2*p*x → (y = -(x - p/2) ∨ y = x - p/2)) →  -- parabola equation and line through focus
  (A.2^2 = 2*p*A.1 ∧ B.2^2 = 2*p*B.1) →                    -- A and B are on the parabola
  (A.2 = -(A.1 - p/2) ∧ B.2 = -(B.1 - p/2)) →              -- A and B are on the line
  ((A.1 + B.1) / 2 = 3) →                                  -- x-coordinate of midpoint is 3
  (λ x : ℝ ↦ x = -1) = (λ x : ℝ ↦ x = -p/2) :=                -- directrix equation
by sorry

#check parabola_directrix

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l467_46799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l467_46720

-- Define the length of the train in meters
noncomputable def train_length : ℝ := 110

-- Define the speed of the train in km/hr
noncomputable def train_speed_kmh : ℝ := 144

-- Define the conversion factor from km/hr to m/s
noncomputable def kmh_to_ms : ℝ := 1000 / 3600

-- Define the function to calculate the time to cross
noncomputable def time_to_cross (length : ℝ) (speed_kmh : ℝ) : ℝ :=
  length / (speed_kmh * kmh_to_ms)

-- Theorem statement
theorem train_crossing_time :
  time_to_cross train_length train_speed_kmh = 2.75 := by
  sorry -- Proof is omitted as per instructions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l467_46720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_article_cost_price_l467_46750

/-- Proves that the cost price of an article is approximately 51.07 given the conditions of the problem -/
theorem article_cost_price (C M : ℝ) : 
  (0.93 * M = 1.25 * C) →  -- Selling price after 7% deduction equals 25% profit on cost
  (abs (0.93 * M - 63.84) < 0.01) →     -- Selling price is approximately 63.84
  (abs (C - 51.07) < 0.01) :=           -- Cost price is approximately 51.07
by
  sorry

#check article_cost_price

end NUMINAMATH_CALUDE_ERRORFEEDBACK_article_cost_price_l467_46750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_characterization_of_f_l467_46715

/-- A positive integer constant -/
def C : ℕ+ := sorry

/-- A function from positive integers to positive integers -/
def f : ℕ+ → ℕ+ := sorry

/-- The condition that must hold for the function -/
def satisfies_condition (f : ℕ+ → ℕ+) : Prop :=
  ∀ a b : ℕ+, a + b > C → (a + f b) ∣ (a^2 + b * f a)

/-- The theorem stating the form of functions satisfying the condition -/
theorem characterization_of_f :
  satisfies_condition f →
  ∃ k : ℕ+, ∀ a : ℕ+, f a = k * a :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_characterization_of_f_l467_46715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_purely_imaginary_l467_46719

def z₁ : ℂ := 2 - Complex.I
def z₂ (m : ℝ) : ℂ := m + Complex.I

theorem product_purely_imaginary (m : ℝ) : 
  (z₁ * z₂ m).re = 0 ∧ (z₁ * z₂ m).im ≠ 0 ↔ m = -1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_purely_imaginary_l467_46719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_harvard_building_levels_l467_46787

/-- The number of levels in the building -/
noncomputable def n : ℕ := 13

/-- The radius of the k-th level from the top -/
def radius (k : ℕ) : ℝ := k

/-- The height of each level -/
def level_height : ℝ := 1

/-- The lateral surface area of the building -/
noncomputable def lateral_area : ℝ := n * (n + 1) * Real.pi

/-- The bottom surface area of the building -/
noncomputable def bottom_area : ℝ := n^2 * Real.pi

/-- The total surface area of the building -/
noncomputable def total_area : ℝ := lateral_area + 2 * bottom_area

/-- The proportion of lateral surface area to total surface area -/
def lateral_proportion : ℝ := 0.35

theorem harvard_building_levels :
  lateral_area = lateral_proportion * total_area → n = 13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_harvard_building_levels_l467_46787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_custom_operation_calculation_l467_46766

/-- Custom binary operation "*" -/
noncomputable def star (a b : ℝ) : ℝ :=
  if a > b then a
  else if a = b then 1
  else b

/-- Theorem stating the result of the calculation -/
theorem custom_operation_calculation :
  (star 1.1 (7/3) - star (1/3) 0.1) / (star (4/5) 0.8) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_custom_operation_calculation_l467_46766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_liu_xiang_performance_l467_46755

/-- Represents the 110m hurdles race configuration and performance metrics -/
structure HurdlesRace where
  totalDistance : ℝ
  numHurdles : ℕ
  startToFirstHurdle : ℝ
  lastHurdleToFinish : ℝ
  bestTimeToFirstHurdle : ℝ
  bestTimeFromLastHurdle : ℝ
  fastestHurdleCycleTime : ℝ

/-- Calculates the distance between consecutive hurdles and the theoretical best time -/
noncomputable def calculateHurdlesMetrics (race : HurdlesRace) : ℝ × ℝ :=
  let hurdlingDistance := race.totalDistance - race.startToFirstHurdle - race.lastHurdleToFinish
  let distanceBetweenHurdles := hurdlingDistance / (race.numHurdles - 1)
  let theoreticalBestTime := 
    race.bestTimeToFirstHurdle + 
    (race.fastestHurdleCycleTime * (race.numHurdles - 1)) + 
    race.bestTimeFromLastHurdle
  (distanceBetweenHurdles, theoreticalBestTime)

/-- The main theorem stating the expected results for Liu Xiang's performance -/
theorem liu_xiang_performance : 
  let race : HurdlesRace := {
    totalDistance := 110
    numHurdles := 10
    startToFirstHurdle := 13.72
    lastHurdleToFinish := 14.02
    bestTimeToFirstHurdle := 2.5
    bestTimeFromLastHurdle := 1.4
    fastestHurdleCycleTime := 0.96
  }
  let (distanceBetweenHurdles, theoreticalBestTime) := calculateHurdlesMetrics race
  abs (distanceBetweenHurdles - 9.14) < 0.01 ∧ abs (theoreticalBestTime - 12.54) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_liu_xiang_performance_l467_46755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_inequality_l467_46717

theorem integral_inequality : 
  (∫ x in (0:ℝ)..1, Real.sin x) < (∫ x in (0:ℝ)..1, Real.sqrt x) ∧ 
  (∫ x in (0:ℝ)..1, Real.sqrt x) < (∫ x in (0:ℝ)..1, x^(1/3 : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_inequality_l467_46717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_computation_l467_46781

theorem vector_computation :
  let v1 : Fin 2 → ℝ := ![3, -9]
  let v2 : Fin 2 → ℝ := ![-1, 6]
  let v3 : Fin 2 → ℝ := ![0, -2]
  (4 : ℝ) • v1 + (3 : ℝ) • v2 - (5 : ℝ) • v3 = ![9, -8] :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_computation_l467_46781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hiker_final_distance_l467_46778

/-- The final distance of a hiker from the starting point after a series of movements -/
noncomputable def final_distance (east west north south : ℝ) : ℝ :=
  Real.sqrt ((east - west)^2 + (north - south)^2)

/-- Theorem stating that the hiker's final distance is 6√5 miles -/
theorem hiker_final_distance :
  final_distance 15 3 8 2 = 6 * Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hiker_final_distance_l467_46778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l467_46777

/-- Point in 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line in 2D plane -/
structure Line where
  slope : ℝ
  point : Point

/-- Triangle formed by three lines -/
structure Triangle where
  l1 : Line
  l2 : Line
  l3 : Line

/-- Rotation of a line around a point -/
noncomputable def rotate (l : Line) (center : Point) (angle : ℝ) : Line :=
  sorry

/-- Area of a triangle -/
noncomputable def area (t : Triangle) : ℝ :=
  sorry

/-- The problem statement -/
theorem max_triangle_area :
  let A : Point := ⟨0, 0⟩
  let B : Point := ⟨14, 0⟩
  let C : Point := ⟨25, 0⟩
  let ℓA : Line := ⟨2, A⟩
  let ℓB : Line := ⟨0, B⟩  -- Vertical line represented by slope 0
  let ℓC : Line := ⟨-2, C⟩
  ∀ θ : ℝ,
    let rotated_triangle : Triangle := ⟨rotate ℓA A θ, rotate ℓB B θ, rotate ℓC C θ⟩
    (∀ φ : ℝ, area rotated_triangle ≤ area (Triangle.mk (rotate ℓA A φ) (rotate ℓB B φ) (rotate ℓC C φ))) →
    area rotated_triangle = 158.5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l467_46777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_exponential_range_l467_46731

noncomputable def f (a : ℝ) (x : ℝ) := (2 * a + 1) ^ x

theorem decreasing_exponential_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x > f a y) ↔ -1/2 < a ∧ a < 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_exponential_range_l467_46731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coach_initial_water_amount_l467_46763

theorem coach_initial_water_amount : 
  let num_players : ℕ := 30
  let water_per_player : ℕ := 200
  let spilled_water : ℕ := 250
  let leftover_water : ℕ := 1750
  (num_players * water_per_player + spilled_water + leftover_water) / 1000 = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coach_initial_water_amount_l467_46763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_transformation_and_intersection_l467_46775

-- Define the curves and transformations
noncomputable def curve_C1 (a : ℝ) : ℝ × ℝ := (3 + 3 * Real.cos a, 2 * Real.sin a)

noncomputable def scaling_transform (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 / 3, p.2 / 2)

noncomputable def curve_C2 (θ : ℝ) : ℝ := 2 * Real.cos θ

noncomputable def curve_C3 (θ : ℝ) : ℝ := 1 / Real.sin (Real.pi / 6 - θ)

-- Theorem statement
theorem curve_transformation_and_intersection :
  -- Part 1: The polar equation of C₂ is ρ = 2cos(θ)
  (∀ θ, curve_C2 θ = 2 * Real.cos θ) ∧
  -- Part 2: The distance between intersection points of C₂ and C₃ is √3
  (∃ P Q : ℝ × ℝ,
    (∃ θ₁, (curve_C2 θ₁ * Real.cos θ₁, curve_C2 θ₁ * Real.sin θ₁) = P) ∧
    (∃ θ₂, (curve_C3 θ₂ * Real.cos θ₂, curve_C3 θ₂ * Real.sin θ₂) = P) ∧
    (∃ θ₃, (curve_C2 θ₃ * Real.cos θ₃, curve_C2 θ₃ * Real.sin θ₃) = Q) ∧
    (∃ θ₄, (curve_C3 θ₄ * Real.cos θ₄, curve_C3 θ₄ * Real.sin θ₄) = Q) ∧
    Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = Real.sqrt 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_transformation_and_intersection_l467_46775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conveyance_is_ten_percent_l467_46779

noncomputable def monthly_salary : ℝ := 12500
noncomputable def savings : ℝ := 2500
noncomputable def food_percentage : ℝ := 40
noncomputable def rent_percentage : ℝ := 20
noncomputable def entertainment_percentage : ℝ := 10

noncomputable def conveyance_percentage : ℝ :=
  100 - (food_percentage + rent_percentage + entertainment_percentage + (savings / monthly_salary * 100))

theorem conveyance_is_ten_percent : conveyance_percentage = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_conveyance_is_ten_percent_l467_46779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sides_equal_longest_diagonal_l467_46796

/-- A convex polygon with n sides -/
structure ConvexPolygon (n : ℕ) where
  sides : Fin n → ℝ
  convex : ∀ i j k : Fin n, i < j → j < k → sides i + sides j > sides k
  n_gt_3 : n > 3

/-- The longest diagonal of a convex polygon -/
noncomputable def longestDiagonal (p : ConvexPolygon n) : ℝ :=
  sorry

/-- The number of sides equal to the longest diagonal -/
def numSidesEqualDiagonal (p : ConvexPolygon n) : ℕ :=
  sorry

/-- Theorem: In any convex polygon with more than 3 sides, 
    at most 2 sides can be equal to the longest diagonal -/
theorem max_sides_equal_longest_diagonal (n : ℕ) (p : ConvexPolygon n) :
  numSidesEqualDiagonal p ≤ 2 := by
  sorry

#check max_sides_equal_longest_diagonal

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sides_equal_longest_diagonal_l467_46796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_monotonicity_l467_46768

theorem exponential_monotonicity (a b : ℝ) (h : a > b) : (2 : ℝ)^a > (2 : ℝ)^b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_monotonicity_l467_46768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_double_arrangement_l467_46735

noncomputable def digits_of (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc else aux (m / 10) ((m % 10) :: acc)
  aux n []

theorem impossible_double_arrangement : ¬ ∃ (a b : ℕ) (digits : List ℕ),
  digits = [2, 3, 4, 5, 6, 7, 8, 9] ∧
  (∀ d, d ∈ digits → d ∈ (digits_of a ++ digits_of b)) ∧
  (∀ d, d ∈ (digits_of a ++ digits_of b) → d ∈ digits) ∧
  a = 2 * b :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_double_arrangement_l467_46735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_in_quadrant_II_l467_46730

/-- The function f(x) = -5/x -/
noncomputable def f (x : ℝ) : ℝ := -5 / x

/-- Theorem: When x < 0, the graph of f(x) = -5/x is in Quadrant II -/
theorem graph_in_quadrant_II (x : ℝ) (hx : x < 0) : x < 0 ∧ f x > 0 := by
  constructor
  · exact hx
  · sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_in_quadrant_II_l467_46730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_range_l467_46758

-- Define the ellipse
def ellipse (P : ℝ × ℝ) : Prop :=
  P.1^2 / 16 + P.2^2 / 12 = 1

-- Define the circle
def circleC (M : ℝ × ℝ) : Prop :=
  (M.1 - 2)^2 + M.2^2 = 1

-- Define the distance between two points
noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- Define the vector sum
def vectorSum (A B C : ℝ × ℝ) : ℝ × ℝ :=
  (B.1 - A.1 + C.1 - A.1, B.2 - A.2 + C.2 - A.2)

-- Define the magnitude of a vector
noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1^2 + v.2^2)

-- The main theorem
theorem vector_sum_range :
  ∀ (P M N : ℝ × ℝ),
    ellipse P →
    circleC M →
    circleC N →
    distance M N = Real.sqrt 3 →
    3 ≤ magnitude (vectorSum P M N) ∧ magnitude (vectorSum P M N) ≤ 13 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_range_l467_46758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_tiling_l467_46704

/-- A function that represents the interior angle of a regular polygon with n sides in radians -/
noncomputable def interior_angle (n : ℕ) : ℝ := (1 - 2 / (n : ℝ)) * Real.pi

/-- Three regular polygons can tile a plane at a vertex if the sum of their interior angles is 2π -/
def can_tile_vertex (x y z : ℕ) : Prop :=
  interior_angle x + interior_angle y + interior_angle z = 2 * Real.pi

theorem regular_polygon_tiling (x y z : ℕ) (h : can_tile_vertex x y z) :
  1 / (x : ℝ) + 1 / (y : ℝ) + 1 / (z : ℝ) = 1 / 2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_tiling_l467_46704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_tangent_r_values_l467_46776

-- Define the centers and radii of the circles
def C1_center : ℝ × ℝ := (-2, 2)
def C2_center : ℝ × ℝ := (2, 5)
def R : ℝ := 1

-- Define the distance between the centers
noncomputable def distance_between_centers : ℝ := 
  Real.sqrt ((C1_center.1 - C2_center.1)^2 + (C1_center.2 - C2_center.2)^2)

-- Define the condition for the circles to be tangent
def are_tangent (r : ℝ) : Prop := 
  (r + R = distance_between_centers) ∨ (r - R = distance_between_centers)

-- The theorem to prove
theorem circles_tangent_r_values : 
  ∀ r : ℝ, are_tangent r ↔ (r = 4 ∨ r = 6) := by
  sorry

-- Additional lemma to show that the distance between centers is indeed 5
lemma distance_is_five : distance_between_centers = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_tangent_r_values_l467_46776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_yi_speed_l467_46773

-- Define constants and variables
noncomputable def bus_interval : ℚ := 5 / 60 -- 5 minutes in hours
def jia_remaining_distance : ℚ := 21 -- km

-- Define the problem
theorem yi_speed : 
  ∀ (total_distance : ℚ) (bus_speed : ℚ) (jia_speed : ℚ) (yi_speed : ℚ),
  -- Conditions
  (bus_speed > 0) →
  (jia_speed > 0) →
  (yi_speed > 0) →
  (total_distance > 0) →
  (yi_speed = 3 * jia_speed) →
  (9 * bus_interval * bus_speed = 6 * bus_interval * yi_speed) →
  (8 * bus_interval * bus_speed = total_distance) →
  (total_distance - jia_remaining_distance = 3 * jia_speed * (8 * bus_interval)) →
  -- Conclusion
  yi_speed = 27 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_yi_speed_l467_46773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_f_min_value_exact_l467_46752

/-- The function f(x) defined for x > 0 -/
noncomputable def f (x : ℝ) : ℝ := x^2 + 1/x^2 + 1/(x^2 + 1/x^2)

/-- Theorem stating that f(x) ≥ 2.5 for all x > 0 -/
theorem f_min_value (x : ℝ) (hx : x > 0) : f x ≥ 2.5 := by
  sorry

/-- Theorem stating that the minimum value of f(x) for x > 0 is 2.5 -/
theorem f_min_value_exact : ∃ (x : ℝ), x > 0 ∧ f x = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_f_min_value_exact_l467_46752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cover_triangle_l467_46728

/-- The side length of the large equilateral triangle -/
noncomputable def large_side : ℝ := 15

/-- The side length of the small equilateral triangles -/
noncomputable def small_side : ℝ := 1

/-- The area of an equilateral triangle given its side length -/
noncomputable def equilateral_triangle_area (side : ℝ) : ℝ := (Real.sqrt 3 / 4) * side ^ 2

/-- The minimum number of small triangles needed to cover the large triangle -/
def min_triangles : ℕ := 225

theorem cover_triangle :
  (equilateral_triangle_area large_side) / (equilateral_triangle_area small_side) = min_triangles := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cover_triangle_l467_46728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_root_implies_coefficients_l467_46791

theorem polynomial_root_implies_coefficients :
  ∀ (a b : ℝ),
  (Complex.I : ℂ) ^ 2 = -1 →
  (fun x : ℂ ↦ x^3 + a*x^2 + 3*x + b) (2 - 3*Complex.I) = 0 →
  a = -3/2 ∧ b = 65/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_root_implies_coefficients_l467_46791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_point_theorem_l467_46740

/-- The distance from the top of the hill where Jack and Jill pass each other -/
noncomputable def meeting_distance : ℝ := 35 / 27

/-- The total distance of the run -/
noncomputable def total_distance : ℝ := 10

/-- The distance to the top of the hill -/
noncomputable def hill_distance : ℝ := 5

/-- Jack's head start in hours -/
noncomputable def head_start : ℝ := 1 / 6

/-- Jack's uphill speed in km/hr -/
noncomputable def jack_uphill_speed : ℝ := 15

/-- Jack's downhill speed in km/hr -/
noncomputable def jack_downhill_speed : ℝ := 20

/-- Jill's uphill speed in km/hr -/
noncomputable def jill_uphill_speed : ℝ := 16

/-- Jill's downhill speed in km/hr -/
noncomputable def jill_downhill_speed : ℝ := 22

theorem meeting_point_theorem :
  ∃ (t : ℝ), 
    t > head_start ∧
    t < head_start + hill_distance / jack_uphill_speed + hill_distance / jack_downhill_speed ∧
    hill_distance - jack_downhill_speed * (t - head_start - hill_distance / jack_uphill_speed) = 
    jill_uphill_speed * (t - head_start) ∧
    meeting_distance = hill_distance - jill_uphill_speed * (t - head_start) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_point_theorem_l467_46740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_contained_arc_l467_46785

/-- Represents a point on a circle --/
structure CirclePoint where
  angle : Real

/-- Represents an arc on a circle --/
structure Arc where
  start : CirclePoint
  finish : CirclePoint

/-- Given n points on a circle and a rotation angle, there exists a new arc entirely within an old arc --/
theorem exists_contained_arc (n k : ℕ) (original_points : Fin n → CirclePoint) 
  (h_distinct : ∀ i j, i ≠ j → original_points i ≠ original_points j) 
  (rotation_angle : Real) (h_rotation : rotation_angle = 2 * Real.pi * (k : Real) / (n : Real)) :
  ∃ (new_arc : Arc) (old_arc : Arc), 
    (∃ i j : Fin n, 
      old_arc.start = original_points i ∧ 
      old_arc.finish = original_points j) ∧
    (new_arc.start.angle ≥ old_arc.start.angle) ∧ 
    (new_arc.finish.angle ≤ old_arc.finish.angle) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_contained_arc_l467_46785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l467_46794

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sin x) * Real.log (Real.cos x)

-- Define the function g(x) = f(x + π/4)
noncomputable def g (x : ℝ) : ℝ := f (x + Real.pi / 4)

-- Theorem statement
theorem f_properties :
  (∀ k : ℤ, ∀ x : ℝ, f x ≠ 0 → x ∈ Set.Ioo (2 * k * Real.pi) (2 * k * Real.pi + Real.pi / 2)) ∧
  (∀ x : ℝ, g (-x) = g x) ∧
  (∃! x : ℝ, x ∈ Set.Ioo 0 (Real.pi / 2) ∧ ∀ y ∈ Set.Ioo 0 (Real.pi / 2), f y ≤ f x) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l467_46794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curvature_cubic_zero_exists_constant_curvature_parabola_curvature_bound_exp_curvature_bound_l467_46711

noncomputable def curvature (f : ℝ → ℝ) (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  let k_A := (deriv f) x₁
  let k_B := (deriv f) x₂
  let AB := Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)
  abs (k_A - k_B) / AB

-- Theorem 1
theorem curvature_cubic_zero :
  let f (x : ℝ) := x^3
  curvature f 1 (f 1) (-1) (f (-1)) = 0 := by sorry

-- Theorem 2
theorem exists_constant_curvature :
  ∃ f : ℝ → ℝ, ∀ x₁ y₁ x₂ y₂ : ℝ, x₁ ≠ x₂ → 
    curvature f x₁ y₁ x₂ y₂ = curvature f 0 (f 0) 1 (f 1) := by sorry

-- Theorem 3
theorem parabola_curvature_bound :
  let f (x : ℝ) := x^2 + 1
  ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → curvature f x₁ (f x₁) x₂ (f x₂) ≤ 2 := by sorry

-- Theorem 4
theorem exp_curvature_bound :
  let f (x : ℝ) := Real.exp x
  ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → curvature f x₁ (f x₁) x₂ (f x₂) < 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curvature_cubic_zero_exists_constant_curvature_parabola_curvature_bound_exp_curvature_bound_l467_46711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alternating_multiple_characterization_l467_46764

def is_alternating (n : ℕ) : Prop :=
  ∀ i : Fin (Nat.digits 10 n).length, i.val + 1 < (Nat.digits 10 n).length →
    (Nat.digits 10 n).get i % 2 ≠ (Nat.digits 10 n).get ⟨i.val + 1, by sorry⟩ % 2

theorem alternating_multiple_characterization (n : ℕ) :
  (∃ m : ℕ, m > 0 ∧ n ∣ m ∧ is_alternating m) ↔ ¬(20 ∣ n) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alternating_multiple_characterization_l467_46764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_contraction_half_satisfies_contraction_l467_46724

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 0
  | (n + 1) => ((1/2 : ℝ) * (sequence_a n)^2 - 1)^(1/3 : ℝ)

theorem sequence_contraction : 
  ∃ (q : ℝ), 0 < q ∧ q < 1 ∧ 
  ∀ (n : ℕ), n ≥ 1 → |sequence_a (n + 1) - sequence_a n| ≤ q * |sequence_a n - sequence_a (n - 1)| :=
by sorry

theorem half_satisfies_contraction : 
  ∀ (n : ℕ), n ≥ 1 → |sequence_a (n + 1) - sequence_a n| ≤ (1/2 : ℝ) * |sequence_a n - sequence_a (n - 1)| :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_contraction_half_satisfies_contraction_l467_46724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_internal_boundary_length_l467_46741

/-- Represents a square plot in the garden -/
structure SquarePlot where
  side : ℕ

/-- Represents the garden configuration -/
structure Garden where
  width : ℕ
  height : ℕ
  plots : List SquarePlot

def Garden.valid (g : Garden) : Prop :=
  g.width = 6 ∧ g.height = 7 ∧ g.plots.length = 5 ∧
  g.plots.all (λ p ↦ p.side > 0) ∧
  (g.plots.map (λ p ↦ p.side * p.side)).sum = g.width * g.height

def Garden.internalBoundaryLength (g : Garden) : ℕ :=
  (g.plots.map (λ p ↦ 4 * p.side)).sum / 2 - (g.width + g.height)

theorem garden_internal_boundary_length (g : Garden) :
  g.valid → g.internalBoundaryLength = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_internal_boundary_length_l467_46741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_eq_cos_implies_cos_2theta_zero_l467_46721

theorem sin_eq_cos_implies_cos_2theta_zero (θ : ℝ) :
  (Real.sin θ = Real.cos θ → Real.cos (2 * θ) = 0) ∧
  ¬(Real.cos (2 * θ) = 0 → Real.sin θ = Real.cos θ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_eq_cos_implies_cos_2theta_zero_l467_46721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wage_increase_percentage_l467_46703

/-- The percentage increase from an initial amount to a new amount -/
noncomputable def percentageIncrease (initial : ℝ) (new : ℝ) : ℝ :=
  ((new - initial) / initial) * 100

/-- Proof that the percentage increase from $50 to $75 is 50% -/
theorem wage_increase_percentage (initialWage newWage : ℝ) 
  (h1 : initialWage = 50) 
  (h2 : newWage = 75) : 
  percentageIncrease initialWage newWage = 50 := by
  -- Unfold the definition of percentageIncrease
  unfold percentageIncrease
  -- Substitute the known values
  rw [h1, h2]
  -- Simplify the arithmetic
  simp [div_eq_mul_inv]
  -- Perform the final calculation
  norm_num
  -- QED


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wage_increase_percentage_l467_46703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_M_is_ellipse_l467_46789

-- Define the circle
def my_circle (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 100

-- Define points A and C
def A : ℝ × ℝ := (1, 0)
def C : ℝ × ℝ := (-1, 0)

-- Define point Q on the circle
def Q : ℝ × ℝ → Prop := λ q => my_circle q.1 q.2

-- Define point M as the intersection of perpendicular bisector of AQ and line CQ
def M (q : ℝ × ℝ) : ℝ × ℝ → Prop :=
  λ m => ∃ (t : ℝ), m = (t * q.1 + (1 - t) * C.1, t * q.2 + (1 - t) * C.2) ∧
                    (m.1 - A.1)^2 + (m.2 - A.2)^2 = (m.1 - q.1)^2 + (m.2 - q.2)^2

-- Statement of the theorem
theorem locus_of_M_is_ellipse :
  ∀ (q : ℝ × ℝ) (m : ℝ × ℝ), Q q → M q m →
  m.1^2 / 25 + m.2^2 / 24 = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_M_is_ellipse_l467_46789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_social_studies_score_l467_46761

theorem social_studies_score (k e s ss : ℕ) (avg_three : (k + e + s) / 3 = 89) 
  (avg_four : (k + e + s + ss) / 4 = 90) : ss = 93 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_social_studies_score_l467_46761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_distribution_l467_46767

def robi_contribution : ℝ := 4000
def rachel_contribution : ℝ := 5000
def profit_percentage : ℝ := 0.20

theorem profit_distribution (rudy_contribution : ℝ) 
  (h1 : rudy_contribution = robi_contribution * 1.25)
  (total_contribution : ℝ) 
  (h2 : total_contribution = robi_contribution + rudy_contribution + rachel_contribution)
  (total_profit : ℝ)
  (h3 : total_profit = total_contribution * profit_percentage) :
  (robi_contribution / total_contribution) * total_profit = 800 ∧ 
  (rudy_contribution / total_contribution) * total_profit = 1000 ∧ 
  (rachel_contribution / total_contribution) * total_profit = 1000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_distribution_l467_46767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_powers_implies_real_l467_46774

/-- Three complex numbers are collinear if they lie on a straight line in the complex plane. -/
def collinear (a b c : ℂ) : Prop :=
  ∃ (t : ℝ), b - a = t • (c - a) ∨ c - a = t • (b - a)

/-- Theorem: If a complex number z and its powers z^2 and z^3 are collinear, then z is real. -/
theorem collinear_powers_implies_real (z : ℂ) 
  (h : collinear z (z^2) (z^3)) : z.im = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_powers_implies_real_l467_46774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_point_circle_radius_is_two_no_tangent_line_longest_chord_not_2root2_main_result_l467_46709

-- Define the line l
def line (k : ℝ) (x y : ℝ) : Prop := k * x - y - k = 0

-- Define the circle M
def circle_M (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 2*y + 1 = 0

-- Theorem 1: The line l always passes through (1,0)
theorem line_passes_through_point :
  ∀ k : ℝ, line k 1 0 := by sorry

-- Theorem 2: The radius of circle M is 2
theorem circle_radius_is_two :
  ∃ center : ℝ × ℝ, ∀ x y : ℝ, circle_M x y ↔ (x - center.1)^2 + (y - center.2)^2 = 4 := by sorry

-- Theorem 3: There does not exist a k such that the line l is tangent to circle M
theorem no_tangent_line :
  ¬∃ k : ℝ, ∀ x y : ℝ, line k x y → circle_M x y → 
    ∀ x' y', line k x' y' ∧ circle_M x' y' → (x = x' ∧ y = y') := by sorry

-- Theorem 4: The longest chord cut by the line l on circle M is not 2√2
theorem longest_chord_not_2root2 :
  ¬∃ k : ℝ, ∀ x₁ y₁ x₂ y₂ : ℝ, 
    line k x₁ y₁ ∧ circle_M x₁ y₁ ∧ line k x₂ y₂ ∧ circle_M x₂ y₂ →
    (x₁ - x₂)^2 + (y₁ - y₂)^2 ≤ 8 := by sorry

-- Main theorem combining all results
theorem main_result :
  (∀ k : ℝ, line k 1 0) ∧
  (∃ center : ℝ × ℝ, ∀ x y : ℝ, circle_M x y ↔ (x - center.1)^2 + (y - center.2)^2 = 4) ∧
  (¬∃ k : ℝ, ∀ x y : ℝ, line k x y → circle_M x y → 
    ∀ x' y', line k x' y' ∧ circle_M x' y' → (x = x' ∧ y = y')) ∧
  (¬∃ k : ℝ, ∀ x₁ y₁ x₂ y₂ : ℝ, 
    line k x₁ y₁ ∧ circle_M x₁ y₁ ∧ line k x₂ y₂ ∧ circle_M x₂ y₂ →
    (x₁ - x₂)^2 + (y₁ - y₂)^2 ≤ 8) := by
  exact ⟨line_passes_through_point, circle_radius_is_two, no_tangent_line, longest_chord_not_2root2⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_point_circle_radius_is_two_no_tangent_line_longest_chord_not_2root2_main_result_l467_46709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_cost_per_meter_l467_46712

/-- Represents a rectangular plot with given dimensions and fencing cost. -/
structure RectangularPlot where
  length : ℝ
  breadth : ℝ
  total_fencing_cost : ℝ

/-- Calculates the cost per meter of fencing for a given rectangular plot. -/
noncomputable def cost_per_meter (plot : RectangularPlot) : ℝ :=
  plot.total_fencing_cost / (2 * (plot.length + plot.breadth))

/-- Theorem stating that for a rectangular plot with given dimensions and total fencing cost,
    the cost per meter of fencing is 26.5. -/
theorem fencing_cost_per_meter (plot : RectangularPlot)
    (h1 : plot.length = 80)
    (h2 : plot.length = plot.breadth + 60)
    (h3 : plot.total_fencing_cost = 5300) :
    cost_per_meter plot = 26.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_cost_per_meter_l467_46712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_moles_of_koh_combined_l467_46748

-- Define the chemical reaction
noncomputable def reaction (hcl koh : ℚ) : ℚ × ℚ := (min hcl koh, min hcl koh)

-- Theorem statement
theorem moles_of_koh_combined 
  (hcl koh : ℚ) -- Amount of HCl and KOH in moles
  (h1 : hcl = 1) -- 1 mole of HCl is used
  (h2 : (reaction hcl koh).1 = 1) -- 1 mole of KCl is produced
  : koh = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_moles_of_koh_combined_l467_46748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l467_46743

/-- The length of a train in meters, given its speed in km/h and time to cross a pole in seconds -/
noncomputable def trainLength (speed : ℝ) (time : ℝ) : ℝ :=
  speed * (1000 / 3600) * time

/-- Theorem: A train traveling at 180 km/h that crosses a pole in 5 seconds has a length of 250 meters -/
theorem train_length_calculation :
  trainLength 180 5 = 250 := by
  -- Unfold the definition of trainLength
  unfold trainLength
  -- Simplify the arithmetic
  simp [mul_assoc, mul_comm, mul_left_comm]
  -- Check that the equality holds
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l467_46743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_eventually_decreasing_l467_46706

noncomputable def a (n : ℕ) : ℚ := (100 ^ n : ℚ) / n.factorial

theorem sequence_eventually_decreasing :
  ∃ N : ℕ, ∀ n ≥ N, a n > a (n + 1) :=
by
  -- We choose N = 100
  use 100
  intro n hn
  -- Prove that for n ≥ 100, a(n) > a(n+1)
  -- This involves comparing (100^n / n!) with (100^(n+1) / (n+1)!)
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_eventually_decreasing_l467_46706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_constants_is_zero_l467_46769

/-- The polynomial fraction and its partial fraction decomposition -/
noncomputable def polynomial_fraction (x y : ℝ) : ℝ := 1 / (x * (x + 1) * (x + 2) * (x + 3) * (y + 2))

/-- The partial fraction decomposition -/
noncomputable def partial_fraction_decomp (x y A B C D E : ℝ) : ℝ :=
  A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (y + 2)

/-- Theorem stating that the sum of constants in the partial fraction decomposition is zero -/
theorem sum_of_constants_is_zero (x y A B C D E : ℝ) 
  (h : ∀ x y, polynomial_fraction x y = partial_fraction_decomp x y A B C D E) :
  A + B + C + D + E = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_constants_is_zero_l467_46769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_hexagon_area_ratio_l467_46742

theorem equilateral_triangle_hexagon_area_ratio (s t : ℝ) (h1 : s > 0) (h2 : t > 0) :
  3 * s = 6 * t →
  (Real.sqrt 3 / 4 * s^2) / (6 * Real.sqrt 3 / 4 * t^2) = 2 / 3 :=
by
  intro h
  -- Proof steps go here
  sorry

#check equilateral_triangle_hexagon_area_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_hexagon_area_ratio_l467_46742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_football_shape_area_football_shape_approx_l467_46739

/-- Represents a square with side length 3 cm and arcs PXC and PYC drawn from centers S and R --/
structure FootballShape where
  side_length : ℝ
  side_length_eq : side_length = 3

/-- Calculates the area of regions II and III combined in the football shape --/
noncomputable def area_regions_II_and_III (shape : FootballShape) : ℝ :=
  (9 / 2) * Real.pi - 9

/-- Theorem stating that the area of regions II and III is (9/2)π - 9 square centimeters --/
theorem area_football_shape (shape : FootballShape) :
    area_regions_II_and_III shape = (9 / 2) * Real.pi - 9 := by
  -- Unfold the definition of area_regions_II_and_III
  unfold area_regions_II_and_III
  -- The equality holds by definition
  rfl

/-- Theorem stating that the area of regions II and III is approximately 5.1 square centimeters --/
theorem area_football_shape_approx (shape : FootballShape) :
    ∃ (ε : ℝ), ε > 0 ∧ ε < 0.05 ∧ |area_regions_II_and_III shape - 5.1| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_football_shape_area_football_shape_approx_l467_46739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_multiples_equal_squared_diff_zero_l467_46797

def a : ℕ := (Finset.filter (λ x : ℕ => x % 12 = 0 ∧ x < 60) (Finset.range 60)).card

def b : ℕ := (Finset.filter (λ x : ℕ => x % 4 = 0 ∧ x % 3 = 0 ∧ x < 60) (Finset.range 60)).card

theorem multiples_equal_squared_diff_zero : (a - b)^2 = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_multiples_equal_squared_diff_zero_l467_46797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_one_common_member_l467_46751

/-- Represents a club of students -/
structure Club where
  members : Finset Nat
  size_eq_three : members.card = 3

/-- The theorem statement -/
theorem existence_of_one_common_member
  (n : Nat)
  (h_n : n ≥ 4)
  (clubs : Finset Club)
  (h_club_count : clubs.card = n + 1)
  (h_unique_clubs : ∀ c1 c2 : Club, c1 ∈ clubs → c2 ∈ clubs → c1 ≠ c2 → c1.members ≠ c2.members)
  (h_student_range : ∀ c : Club, c ∈ clubs → ∀ s ∈ c.members, s ≤ n) :
  ∃ c1 c2 : Club, c1 ∈ clubs ∧ c2 ∈ clubs ∧ c1 ≠ c2 ∧ (c1.members ∩ c2.members).card = 1 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_one_common_member_l467_46751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_village_population_theorem_l467_46770

theorem village_population_theorem (P : ℝ) : 
  1.05 * (0.765 * P + 50) = 3213 ↔ (P ≥ 3934.5 ∧ P < 3935.5) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_village_population_theorem_l467_46770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_cubed_times_seven_to_w_equals_49_l467_46745

theorem seven_cubed_times_seven_to_w_equals_49 :
  ∃ w : ℝ, (7 : ℝ)^3 * (7 : ℝ)^w = 49 ∧ w = -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_cubed_times_seven_to_w_equals_49_l467_46745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_ratio_range_l467_46722

/-- Semi-focal distance of an ellipse -/
noncomputable def semi_focal_distance (a b : ℝ) : ℝ := Real.sqrt (a^2 - b^2)

/-- The theorem statement -/
theorem ellipse_ratio_range (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : b < a) :
  let c := semi_focal_distance a b
  1 < (b + c) / a ∧ (b + c) / a ≤ Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_ratio_range_l467_46722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_composite_after_five_steps_l467_46793

/-- Transformation function that adds the last digit plus one to the number -/
def transform (n : ℕ) : ℕ :=
  n + (n % 10 + 1)

/-- Predicate to check if a number is prime -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → n % d ≠ 0

/-- Theorem stating that for any prime number, after at most 5 iterations of the transformation,
    the resulting number will be composite -/
theorem composite_after_five_steps (p : ℕ) (h : is_prime p) :
  ∃ k : ℕ, k ≤ 5 ∧ ¬is_prime (Nat.iterate transform k p) := by
  sorry

#check composite_after_five_steps

end NUMINAMATH_CALUDE_ERRORFEEDBACK_composite_after_five_steps_l467_46793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_minus_x_l467_46747

-- Define the step function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < -2 then -3
  else if x < -1 then -2
  else if x < 0 then -1
  else if x < 1 then 0
  else if x < 2 then 1
  else if x < 3 then 2
  else 3

-- Define the domain of f
def domain : Set ℝ := Set.Icc (-3) 3

-- State the theorem
theorem range_of_f_minus_x :
  Set.range (fun x => f x - x) = Set.Ioo (-1) 0 ∪ {0} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_minus_x_l467_46747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_always_divisible_by_23_l467_46783

theorem not_always_divisible_by_23 : ∃ n : ℕ, ¬ (23 ∣ 2^(2^(10*n+1))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_always_divisible_by_23_l467_46783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_five_pi_minus_sixteen_approx_l467_46749

theorem abs_five_pi_minus_sixteen_approx : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.00001 ∧ abs (abs (5 * Real.pi - 16) - 0.29205) < ε :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_five_pi_minus_sixteen_approx_l467_46749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_period_sine_l467_46710

/-- A function with the given properties -/
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 6)

/-- The theorem statement -/
theorem smallest_period_sine (ω : ℝ) (h1 : ω > 0) 
  (h2 : ∀ (p : ℝ), p > 0 → (∀ (x : ℝ), f ω x = f ω (x + p)) → p ≥ Real.pi) :
  f ω (Real.pi / 3) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_period_sine_l467_46710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_barrel_cost_minimized_l467_46765

/-- Represents the cost function for a cylindrical barrel -/
noncomputable def cost (V : ℝ) (a : ℝ) (r : ℝ) : ℝ := a * (2 * V / r + 4 * Real.pi * r^2)

/-- Theorem: The cost of a cylindrical barrel is minimized when r/h = 1/4 -/
theorem barrel_cost_minimized (V : ℝ) (a : ℝ) (h r : ℝ) 
  (hV : V > 0) (ha : a > 0) (hr : r > 0) (hh : h > 0) 
  (hVol : V = Real.pi * r^2 * h) :
  (∀ x > 0, cost V a r ≤ cost V a x) → r / h = 1 / 4 := by
  sorry

#check barrel_cost_minimized

end NUMINAMATH_CALUDE_ERRORFEEDBACK_barrel_cost_minimized_l467_46765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_sin_acute_triangle_l467_46780

-- Define a convex function on an interval
def ConvexFunction (f : ℝ → ℝ) (D : Set ℝ) :=
  ∀ (x y : ℝ) (t : ℝ), x ∈ D → y ∈ D → 0 ≤ t → t ≤ 1 →
    f (t * x + (1 - t) * y) ≤ t * f x + (1 - t) * f y

-- Define an acute-angled triangle
def AcuteTriangle (A B C : ℝ) :=
  0 < A ∧ A < Real.pi/2 ∧ 0 < B ∧ B < Real.pi/2 ∧ 0 < C ∧ C < Real.pi/2 ∧ A + B + C = Real.pi

-- State the theorem
theorem max_sum_sin_acute_triangle :
  ConvexFunction Real.sin (Set.Icc 0 (Real.pi/2)) →
  ∀ (A B C : ℝ), AcuteTriangle A B C →
  Real.sin A + Real.sin B + Real.sin C ≤ 3 * Real.sqrt 3 / 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_sin_acute_triangle_l467_46780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dwarf_truth_count_l467_46737

/-- Represents a dwarf who either always tells the truth or always lies -/
inductive Dwarf
| truthful
| liar
deriving DecidableEq

/-- Represents the three types of ice cream -/
inductive IceCream
| vanilla
| chocolate
| fruit
deriving DecidableEq

/-- The main theorem to be proved -/
theorem dwarf_truth_count 
  (dwarfs : Finset Dwarf) 
  (ice_cream_preference : Dwarf → IceCream) 
  (h1 : dwarfs.card = 10)
  (h2 : ∀ d : Dwarf, d ∈ dwarfs → (ice_cream_preference d = IceCream.vanilla ∨ 
                                   ice_cream_preference d = IceCream.chocolate ∨ 
                                   ice_cream_preference d = IceCream.fruit))
  (h3 : (dwarfs.filter (λ d => d = Dwarf.truthful ∨ 
         (d = Dwarf.liar ∧ ice_cream_preference d ≠ IceCream.vanilla))).card = 10)
  (h4 : (dwarfs.filter (λ d => (d = Dwarf.truthful ∧ ice_cream_preference d = IceCream.chocolate) ∨ 
         (d = Dwarf.liar ∧ ice_cream_preference d ≠ IceCream.chocolate))).card = 5)
  (h5 : (dwarfs.filter (λ d => (d = Dwarf.truthful ∧ ice_cream_preference d = IceCream.fruit) ∨ 
         (d = Dwarf.liar ∧ ice_cream_preference d ≠ IceCream.fruit))).card = 1) :
  (dwarfs.filter (λ d => d = Dwarf.truthful)).card = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dwarf_truth_count_l467_46737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_P_to_l_is_one_l467_46726

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 + x

-- Define the property of f being an odd function
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define the line l
def line_l (x y : ℝ) : Prop := 3 * x + 4 * y - 6 = 0

-- Define the distance formula from a point to a line
noncomputable def distance_point_to_line (x₀ y₀ A B C : ℝ) : ℝ :=
  |A * x₀ + B * y₀ + C| / Real.sqrt (A^2 + B^2)

-- The main theorem
theorem distance_from_P_to_l_is_one (m : ℝ) :
  is_odd_function (f m) →
  distance_point_to_line m 2 3 4 (-6) = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_P_to_l_is_one_l467_46726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_crackers_eaten_proof_l467_46753

/-- Represents the number of animal crackers in a pack -/
inductive PackSize
  | eight : PackSize
  | ten : PackSize
  | twelve : PackSize
  | fifteen : PackSize

/-- Returns the number of crackers for a given PackSize -/
def crackers_in_pack (size : PackSize) : ℕ :=
  match size with
  | .eight => 8
  | .ten => 10
  | .twelve => 12
  | .fifteen => 15

/-- Represents the distribution of pack sizes -/
def pack_distribution : List (PackSize × ℕ) :=
  [(PackSize.eight, 5), (PackSize.ten, 10), (PackSize.twelve, 7), (PackSize.fifteen, 3)]

/-- Calculates the total number of crackers in all packs -/
def total_crackers : ℕ :=
  pack_distribution.foldr (fun (size, count) acc => acc + crackers_in_pack size * count) 0

/-- Calculates the number of crackers in uneaten packs -/
def uneaten_crackers : ℕ :=
  crackers_in_pack PackSize.eight +
  crackers_in_pack PackSize.ten +
  crackers_in_pack PackSize.twelve +
  crackers_in_pack PackSize.fifteen

theorem crackers_eaten_proof :
  total_crackers - uneaten_crackers = 224 := by
  sorry

#eval total_crackers - uneaten_crackers

end NUMINAMATH_CALUDE_ERRORFEEDBACK_crackers_eaten_proof_l467_46753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_douglas_county_x_percentage_l467_46732

/-- Represents the election results in counties X and Y -/
structure ElectionResults where
  totalVotePercentage : ℚ  -- Total vote percentage for Douglas in both counties
  voterRatio : ℚ  -- Ratio of voters in county X to county Y
  countyYPercentage : ℚ  -- Douglas's vote percentage in county Y

/-- Calculates the vote percentage for Douglas in county X -/
def countyXPercentage (results : ElectionResults) : ℚ :=
  (3 * results.totalVotePercentage - results.countyYPercentage) / 2

/-- Theorem stating that given the conditions, Douglas won 72% of the vote in county X -/
theorem douglas_county_x_percentage 
  (results : ElectionResults) 
  (h1 : results.totalVotePercentage = 6/10) 
  (h2 : results.voterRatio = 2) 
  (h3 : results.countyYPercentage = 36/100) : 
  countyXPercentage results = 72/100 := by
  sorry

#eval countyXPercentage { totalVotePercentage := 6/10, voterRatio := 2, countyYPercentage := 36/100 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_douglas_county_x_percentage_l467_46732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l467_46700

-- Define the function f(x) = log_5(x) + x - 3
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 5 + x - 3

-- State the theorem
theorem zero_in_interval :
  ∃ x : ℝ, 2 < x ∧ x < 3 ∧ f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l467_46700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_more_bad_labyrinths_l467_46762

/-- Represents a labyrinth on an 8x8 chessboard -/
def Labyrinth := Fin 8 → Fin 8 → Bool

/-- Two squares are adjacent if they share a row or column -/
def adjacent (p q : Fin 8 × Fin 8) : Prop :=
  (p.1 = q.1 ∧ p.2 ≠ q.2) ∨ (p.1 ≠ q.1 ∧ p.2 = q.2)

/-- A labyrinth is good if a rook can traverse all squares without jumping over walls -/
def is_good (l : Labyrinth) : Prop :=
  ∃ (path : List (Fin 8 × Fin 8)), 
    path.length = 64 ∧ 
    path.Nodup ∧
    (∀ (i j : Fin 8), (i, j) ∈ path) ∧
    (∀ (p q : Fin 8 × Fin 8), p ∈ path → q ∈ path → adjacent p q → l p.1 p.2 = false ∨ l q.1 q.2 = false)

/-- The set of all possible labyrinths -/
def AllLabyrinths : Set Labyrinth := Set.univ

/-- The set of good labyrinths -/
def GoodLabyrinths : Set Labyrinth := {l ∈ AllLabyrinths | is_good l}

/-- The set of bad labyrinths -/
def BadLabyrinths : Set Labyrinth := AllLabyrinths \ GoodLabyrinths

theorem more_bad_labyrinths : Set.ncard BadLabyrinths > Set.ncard GoodLabyrinths := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_more_bad_labyrinths_l467_46762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_in_rectangle_perimeter_l467_46746

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rhombus -/
structure Rhombus where
  e : Point
  f : Point
  g : Point
  h : Point

/-- Represents a rectangle -/
structure Rectangle where
  j : Point
  k : Point
  l : Point
  m : Point

/-- Check if a rhombus is inscribed in a rectangle -/
def isInscribed (r : Rhombus) (rect : Rectangle) : Prop :=
  sorry

/-- Calculate the perimeter of a rectangle -/
def perimeter (rect : Rectangle) : ℝ :=
  sorry

/-- Find the fraction representation of a real number -/
noncomputable def toFraction (x : ℝ) : ℚ :=
  sorry

theorem rhombus_in_rectangle_perimeter 
  (r : Rhombus) (rect : Rectangle) 
  (h_inscribed : isInscribed r rect)
  (h_je : |rect.j.x - r.e.x| = 12)
  (h_ef : |r.e.x - r.f.x| + |r.e.y - r.f.y| = 24)
  (h_fk : |r.f.x - rect.k.x| = 16)
  (h_eg : |r.e.x - r.g.x| + |r.e.y - r.g.y| = 32) :
  let p_q := toFraction (perimeter rect)
  p_q.num + p_q.den = 117 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_in_rectangle_perimeter_l467_46746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithm_of_fraction_equals_two_l467_46782

-- Define the star operation as noncomputable
noncomputable def star (a b : ℝ) : ℝ := Real.rpow a (Real.log b)

-- State the theorem
theorem logarithm_of_fraction_equals_two (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  Real.log ((star (a * b) (a * b)) / (star a a * star b b)) / Real.log (star a b) = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithm_of_fraction_equals_two_l467_46782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coffee_packing_solution_l467_46736

def coffee_packing (total_weight : ℕ) (bag_weight_1 bag_weight_2 : ℕ) : ℕ → ℕ → Prop :=
  fun x y ↦ bag_weight_1 * x + bag_weight_2 * y = total_weight

def minimize_smaller_bags (total_weight : ℕ) (bag_weight_1 bag_weight_2 : ℕ) : ℕ → ℕ → Prop :=
  fun x y ↦ ∀ x' y', coffee_packing total_weight bag_weight_1 bag_weight_2 x' y' → y ≤ y'

theorem coffee_packing_solution :
  ∃ x y : ℕ,
    coffee_packing 1998 15 8 x y ∧
    minimize_smaller_bags 1998 15 8 x y ∧
    x + y = 140 := by
  sorry

#check coffee_packing_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coffee_packing_solution_l467_46736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_parts_eq_neg_fourth_l467_46798

/-- The imaginary unit -/
noncomputable def i : ℂ := Complex.I

/-- The complex number we're analyzing -/
noncomputable def z : ℂ := i / (1 + i)

/-- The real part of z -/
noncomputable def real_part : ℝ := z.re

/-- The imaginary part of z -/
noncomputable def imag_part : ℝ := z.im

theorem product_of_parts_eq_neg_fourth : real_part * imag_part = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_parts_eq_neg_fourth_l467_46798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_condition_l467_46795

theorem quadratic_roots_condition (c : ℝ) : 
  (∀ x : ℝ, x^2 - 3*x + c = 0 ↔ x = (3 + Real.sqrt c)/2 ∨ x = (3 - Real.sqrt c)/2) → 
  c = 9/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_condition_l467_46795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_hexagon_interior_angle_proof_l467_46707

/-- The measure of an interior angle of a regular hexagon is 120 degrees. -/
def regular_hexagon_interior_angle : ℚ :=
  120

/-- A regular hexagon has 6 sides. -/
def regular_hexagon_sides : ℕ := 6

/-- The sum of interior angles of a polygon with n sides is (n-2) * 180 degrees. -/
def sum_of_interior_angles (n : ℕ) : ℚ :=
  (n - 2) * 180

/-- The measure of each interior angle in a regular polygon is the sum of interior angles divided by the number of sides. -/
def interior_angle_measure (n : ℕ) : ℚ :=
  sum_of_interior_angles n / n

/-- Proof that the measure of an interior angle of a regular hexagon is 120 degrees. -/
theorem regular_hexagon_interior_angle_proof :
  interior_angle_measure regular_hexagon_sides = regular_hexagon_interior_angle :=
by
  -- Unfold the definitions
  unfold interior_angle_measure
  unfold sum_of_interior_angles
  unfold regular_hexagon_sides
  unfold regular_hexagon_interior_angle
  -- Simplify the expression
  simp
  -- The proof is complete
  rfl

#check regular_hexagon_interior_angle_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_hexagon_interior_angle_proof_l467_46707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_urn_experiment_probabilities_l467_46718

/-- Two-urn ball transfer experiment -/
structure TwoUrnExperiment (a b : ℕ) where
  left : ℕ   -- number of balls in the left urn
  right : ℕ  -- number of balls in the right urn
  total : ℕ  -- total number of balls
  h1 : left + right = total
  h2 : total = a + b
  h3 : left = a
  h4 : right = b

/-- Probability of the left urn being emptied -/
def prob_left_empty (a b : ℕ) : ℚ := b / (a + b)

/-- Probability of the right urn being emptied -/
def prob_right_empty (a b : ℕ) : ℚ := a / (a + b)

/-- Probability that the experiment never ends -/
def prob_never_ends : ℚ := 0

theorem urn_experiment_probabilities (a b : ℕ) (h : a > 0 ∧ b > 0) :
  let exp := TwoUrnExperiment a b
  (prob_left_empty a b + prob_right_empty a b = 1) ∧
  (prob_never_ends = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_urn_experiment_probabilities_l467_46718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_values_count_l467_46760

theorem integer_values_count : 
  ∃ (S : Finset ℤ), (∀ n : ℤ, n ∈ S ↔ ∃ m : ℤ, 4800 * (3/4 : ℚ)^n = m) ∧ Finset.card S = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_values_count_l467_46760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equality_l467_46738

open Real

-- Define the integrand
noncomputable def f (x : ℝ) : ℝ := (3*x^3 + 9*x^2 + 10*x + 2) / ((x-1)*(x+1)^3)

-- Define the antiderivative
noncomputable def F (x : ℝ) : ℝ := 3 * log (abs (x - 1)) - 1 / (2 * (x + 1)^2)

-- State the theorem
theorem integral_equality (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) : 
  deriv F x = f x := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equality_l467_46738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_properties_l467_46727

/-- The sum of coefficients in the expansion of (x + x^(-1))^n -/
def sum_of_coefficients (n : ℕ) : ℕ := 2^n

/-- The coefficient of x^k in the expansion of (x + x^(-1))^n -/
def coefficient (n k : ℕ) : ℕ := Nat.choose n ((n - k) / 2)

theorem expansion_properties (n : ℕ) :
  sum_of_coefficients n = 128 →
  n = 7 ∧ coefficient n 3 = 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_properties_l467_46727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_f_value_l467_46725

/-- Given positive integers d, e, f where d < e < f, and a system of equations
    with exactly one solution, prove that the minimum value of f is 1006. -/
theorem min_f_value (d e f : ℕ) (hd : d < e) (he : e < f)
    (h_unique : ∃! (x y : ℝ), 2 * x + y = 2010 ∧ y = |x - d| + |x - e| + |x - f|) :
    f ≥ 1006 ∧ ∃ (d' e' : ℕ), d' < e' ∧ e' < 1006 ∧
    ∃! (x y : ℝ), 2 * x + y = 2010 ∧ y = |x - d'| + |x - e'| + |x - 1006| :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_f_value_l467_46725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_angle_A_is_pi_third_sin_2B_plus_A_triangle_perimeter_l467_46708

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem combining all parts of the problem -/
theorem triangle_properties (t : Triangle) 
  (h1 : t.c * Real.cos t.B + t.b * Real.cos t.C = t.a / (2 * Real.cos t.A))
  (h2 : Real.cos t.B = Real.sqrt 3 / 3)
  (h3 : (1/2) * t.b * t.c * Real.sin t.A = 4 * Real.sqrt 3 / 3)
  (h4 : t.a = 3) :
  t.A = π/3 ∧ 
  Real.sin (2*t.B + t.A) = (2*Real.sqrt 2 - Real.sqrt 3) / 6 ∧
  t.a + t.b + t.c = 8 := by
  sorry

/-- Helper theorem for part 1 -/
theorem angle_A_is_pi_third (t : Triangle) 
  (h : t.c * Real.cos t.B + t.b * Real.cos t.C = t.a / (2 * Real.cos t.A)) :
  t.A = π/3 := by
  sorry

/-- Helper theorem for part 2 -/
theorem sin_2B_plus_A (t : Triangle) 
  (h1 : t.A = π/3)
  (h2 : Real.cos t.B = Real.sqrt 3 / 3) :
  Real.sin (2*t.B + t.A) = (2*Real.sqrt 2 - Real.sqrt 3) / 6 := by
  sorry

/-- Helper theorem for part 3 -/
theorem triangle_perimeter (t : Triangle) 
  (h1 : (1/2) * t.b * t.c * Real.sin t.A = 4 * Real.sqrt 3 / 3)
  (h2 : t.a = 3)
  (h3 : t.A = π/3) :
  t.a + t.b + t.c = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_angle_A_is_pi_third_sin_2B_plus_A_triangle_perimeter_l467_46708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_bounds_l467_46759

/-- The function F as defined in the problem -/
noncomputable def F (a x y : ℝ) : ℝ := x + y - a * (2 * Real.sqrt (3 * x * y) + x)

/-- The theorem statement -/
theorem a_bounds (x₀ : ℝ) (h₁ : x₀ > 0) (h₂ : F a x₀ 3 = 3) : 0 < a ∧ a < 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_bounds_l467_46759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_difference_implies_x_squared_range_l467_46790

theorem cube_root_difference_implies_x_squared_range (x : ℝ) :
  (x + 9).rpow (1/3) - (x - 9).rpow (1/3) = 3 →
  75 < x^2 ∧ x^2 < 85 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_difference_implies_x_squared_range_l467_46790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l467_46734

open Real

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x : ℝ, x > 0 → f x + 2 * f (1 / x) = 3 * x + 6) →
  (∀ x : ℝ, x > 0 → f x = 2 / x - x + 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l467_46734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_profit_percentage_l467_46771

noncomputable def car_purchase_price : ℝ := 42000
noncomputable def repair_expenses : ℝ := 16500
noncomputable def selling_price : ℝ := 64900

noncomputable def total_cost : ℝ := car_purchase_price + repair_expenses
noncomputable def profit : ℝ := selling_price - total_cost
noncomputable def profit_percentage : ℝ := (profit / total_cost) * 100

theorem car_profit_percentage :
  |profit_percentage - 10.94| < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_profit_percentage_l467_46771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_length_l467_46729

/-- Curve C in the Cartesian plane -/
def curve_C (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

/-- Line l in the Cartesian plane -/
def line_l (x y : ℝ) : Prop := x - y - 1 = 0

/-- The length of the line segment AB formed by the intersection of curve C and line l -/
noncomputable def length_AB : ℝ := 24/7

/-- Theorem stating that the length of AB is 24/7 -/
theorem intersection_length :
  ∀ (A B : ℝ × ℝ),
  curve_C A.1 A.2 ∧ curve_C B.1 B.2 ∧
  line_l A.1 A.2 ∧ line_l B.1 B.2 ∧
  A ≠ B →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = length_AB :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_length_l467_46729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_l467_46713

def U : Set ℕ := {x | 1 ≤ x ∧ x ≤ 8}

def A : Set ℕ := {1, 2, 5, 7}

def B : Set ℕ := {2, 4, 6, 7}

theorem set_operations :
  (A ∩ B = {2, 7}) ∧
  ((U \ A) ∪ B = {2, 3, 4, 6, 7, 8}) ∧
  (A ∩ (U \ B) = {1, 5}) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_l467_46713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclist_rate_calculation_l467_46723

/-- The cyclist's constant traveling rate in km/h -/
noncomputable def cyclist_rate : ℝ := 8

/-- The hiker's constant walking rate in km/h -/
noncomputable def hiker_rate : ℝ := 4

/-- Time the cyclist travels before stopping, in hours -/
noncomputable def cyclist_travel_time : ℝ := 5 / 60

/-- Time the cyclist waits for the hiker, in hours -/
noncomputable def cyclist_wait_time : ℝ := 10.000000000000002 / 60

theorem cyclist_rate_calculation :
  cyclist_rate * cyclist_travel_time = hiker_rate * (cyclist_travel_time + cyclist_wait_time) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclist_rate_calculation_l467_46723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_section_area_l467_46788

-- Define the cube
structure Cube where
  edge_length : ℝ
  edge_positive : edge_length > 0

-- Define the cross-section
structure CrossSection (c : Cube) where
  area : ℝ
  area_positive : area > 0

-- Theorem statement
theorem cross_section_area (c : Cube) (h : c.edge_length = 2) :
  ∃ (cs : CrossSection c), cs.area = 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_section_area_l467_46788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_function_solution_l467_46772

/-- A function satisfying the given functional equation and condition -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  Continuous f ∧
  (∀ x y : ℝ, 3 * f (x + y) = f x * f y) ∧
  f 1 = 12

/-- The theorem stating that the only function satisfying the given conditions is f(x) = 3 * 4^x -/
theorem unique_function_solution (f : ℝ → ℝ) (h : FunctionalEquation f) :
  ∀ x : ℝ, f x = 3 * (4 : ℝ)^x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_function_solution_l467_46772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fewest_cookies_alice_cookies_l467_46733

-- Define the cookie shapes
inductive CookieShape
  | Rectangle
  | Parallelogram
  | Trapezoid
  | Circle

-- Define a function to get the area of each cookie shape
noncomputable def cookieArea (shape : CookieShape) : ℝ :=
  match shape with
  | CookieShape.Rectangle => 10
  | CookieShape.Parallelogram => 12
  | CookieShape.Trapezoid => 14
  | CookieShape.Circle => 16

-- Define a function to calculate the number of cookies for a given shape and dough volume
noncomputable def numberOfCookies (shape : CookieShape) (doughVolume : ℝ) : ℝ :=
  doughVolume / cookieArea shape

-- Theorem statement
theorem fewest_cookies (doughVolume : ℝ) (h : doughVolume > 0) :
  ∀ (shape : CookieShape), numberOfCookies CookieShape.Circle doughVolume ≤ numberOfCookies shape doughVolume :=
by sorry

-- Alice makes exactly 15 cookies
theorem alice_cookies :
  numberOfCookies CookieShape.Rectangle 150 = 15 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fewest_cookies_alice_cookies_l467_46733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_PQRS_equals_one_l467_46705

-- Define the variables as noncomputable
noncomputable def P : ℝ := Real.sqrt 2012 + Real.sqrt 2013
noncomputable def Q : ℝ := -Real.sqrt 2012 - Real.sqrt 2013
noncomputable def R : ℝ := Real.sqrt 2012 - Real.sqrt 2013
noncomputable def S : ℝ := Real.sqrt 2013 - Real.sqrt 2012

-- State the theorem
theorem product_PQRS_equals_one : P * Q * R * S = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_PQRS_equals_one_l467_46705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_girls_exceed_boys_by_six_l467_46757

/-- Represents a class with boys and girls -/
structure ClassComposition where
  boys : ℕ
  girls : ℕ

/-- The ratio of boys to girls is 3:4 -/
def ratio_condition (c : ClassComposition) : Prop :=
  4 * c.boys = 3 * c.girls

/-- The total number of students is 42 -/
def total_condition (c : ClassComposition) : Prop :=
  c.boys + c.girls = 42

/-- The theorem to be proved -/
theorem girls_exceed_boys_by_six (c : ClassComposition) 
  (h_ratio : ratio_condition c) (h_total : total_condition c) : 
  c.girls = c.boys + 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_girls_exceed_boys_by_six_l467_46757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_xy_plus_yz_l467_46754

theorem max_value_xy_plus_yz (x y z : ℝ) (h : x^2 + y^2 + z^2 = 1) :
  ∃ (max : ℝ), max = Real.sqrt 2 / 2 ∧ 
  ∀ (x' y' z' : ℝ), x'^2 + y'^2 + z'^2 = 1 → x' * y' + y' * z' ≤ max := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_xy_plus_yz_l467_46754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_after_three_hours_l467_46792

/-- Represents a swimming pool with water flow rates and capacity. -/
structure Pool where
  capacity : ℚ
  drainTime : ℚ
  fillTime : ℚ

/-- Calculates the amount of water remaining in the pool after a given time. -/
def waterRemaining (p : Pool) (time : ℚ) : ℚ :=
  p.capacity - (p.capacity / p.drainTime) * time + (p.capacity / p.fillTime) * time

/-- Theorem stating the amount of water remaining after 3 hours in the specific pool scenario. -/
theorem water_after_three_hours :
  let p : Pool := { capacity := 120, drainTime := 4, fillTime := 6 }
  waterRemaining p 3 = 90 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_after_three_hours_l467_46792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scaling_transformation_correct_l467_46716

/-- The scaling transformation that transforms the given ellipse to a unit circle -/
noncomputable def φ (x y : ℝ) : ℝ × ℝ := (x / 2, y / Real.sqrt 3)

/-- The original ellipse equation -/
def original_equation (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

/-- The target circle equation -/
def target_equation (x y : ℝ) : Prop := x^2 + y^2 = 1

theorem scaling_transformation_correct :
  ∀ x y : ℝ, original_equation x y ↔ target_equation (φ x y).1 (φ x y).2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scaling_transformation_correct_l467_46716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_frustum_volume_l467_46714

/-- The volume of a frustum with given dimensions -/
noncomputable def frustum_volume (r1 : ℝ) (r2 : ℝ) (s : ℝ) : ℝ :=
  (1/3) * Real.pi * (r1^2 + r2^2 + r1*r2) * Real.sqrt (s^2 - (r2 - r1)^2)

/-- Theorem stating that the volume of the specific frustum is 21π -/
theorem specific_frustum_volume :
  frustum_volume 1 4 (3 * Real.sqrt 2) = 21 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_frustum_volume_l467_46714
