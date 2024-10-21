import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_convergence_l558_55860

/-- The number of zeroes in the base 3 representation of a positive integer -/
def a (n : ℕ+) : ℕ := sorry

/-- The series in question -/
noncomputable def series (x : ℝ) : ℝ := ∑' n : ℕ+, x^(a n) / (n : ℝ)^3

/-- Theorem: The series converges if and only if x < 25 -/
theorem series_convergence (x : ℝ) (hx : x > 0) : 
  HasSum (λ n ↦ x^(a n) / (n : ℝ)^3) (series x) ↔ x < 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_convergence_l558_55860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meat_budget_percentage_l558_55872

noncomputable def broccoli_cost : ℚ := 3 * 4
noncomputable def oranges_cost : ℚ := 3 * (3/4)
noncomputable def cabbage_cost : ℚ := 15/4
noncomputable def bacon_cost : ℚ := 3
noncomputable def chicken_cost : ℚ := 2 * 3

noncomputable def total_vegetable_cost : ℚ := broccoli_cost + oranges_cost + cabbage_cost
noncomputable def total_meat_cost : ℚ := bacon_cost + chicken_cost
noncomputable def total_grocery_cost : ℚ := total_vegetable_cost + total_meat_cost

noncomputable def meat_percentage : ℚ := (total_meat_cost / total_grocery_cost) * 100

theorem meat_budget_percentage :
  Int.floor meat_percentage = 33 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_meat_budget_percentage_l558_55872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ning_farm_output_values_l558_55803

/-- Represents the annual output values of Ning Farm from 1999 to 2003 -/
def NingFarmOutput : Fin 5 → ℕ := sorry

/-- The output values are strictly increasing -/
axiom increasing_output : ∀ i j, i < j → NingFarmOutput i < NingFarmOutput j

/-- Average of first three years is 180 -/
axiom avg_first_three : (NingFarmOutput 0 + NingFarmOutput 1 + NingFarmOutput 2) / 3 = 180

/-- Average of last three years is 260 -/
axiom avg_last_three : (NingFarmOutput 2 + NingFarmOutput 3 + NingFarmOutput 4) / 3 = 260

/-- Difference between first two years is 70 -/
axiom diff_first_two : NingFarmOutput 1 - NingFarmOutput 0 = 70

/-- Difference between last two years is 50 -/
axiom diff_last_two : NingFarmOutput 4 - NingFarmOutput 3 = 50

/-- Average of highest and lowest values is 220 -/
axiom avg_highest_lowest : (NingFarmOutput 0 + NingFarmOutput 4) / 2 = 220

theorem ning_farm_output_values :
  NingFarmOutput 0 = 130 ∧
  NingFarmOutput 1 = 200 ∧
  NingFarmOutput 2 = 210 ∧
  NingFarmOutput 3 = 260 ∧
  NingFarmOutput 4 = 310 := by
  sorry

#check ning_farm_output_values

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ning_farm_output_values_l558_55803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l558_55844

-- Define the function f as noncomputable
noncomputable def f (x y : ℝ) : ℝ := (x^3 + y^3) / (x + y)^3

-- State the theorem
theorem range_of_f :
  ∀ x y : ℝ, x > 0 → y > 0 → x^2 + y^2 = 1 →
  ∃ z : ℝ, z ∈ Set.Icc (1/4 : ℝ) 1 ∧ f x y = z ∧
  ∀ w : ℝ, f x y = w → w ∈ Set.Icc (1/4 : ℝ) 1 :=
by
  -- The proof is omitted and replaced with sorry
  sorry

-- Note: Set.Icc a b represents the closed interval [a, b]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l558_55844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sarahs_calculation_l558_55886

theorem sarahs_calculation (f g h i j : ℝ) 
  (hf : f = 2) (hg : g = 4) (hh : h = 5) (hi : i = 10) :
  f - (g - (h * (i - j))) = f - g - h * i - j ↔ j = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sarahs_calculation_l558_55886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_dataset_l558_55869

noncomputable def dataSet : List ℝ := [11, 12, 15, 18, 13, 15]

noncomputable def mean (list : List ℝ) : ℝ := (list.sum) / list.length

noncomputable def variance (list : List ℝ) : ℝ :=
  (list.map (λ x => (x - mean list) ^ 2)).sum / list.length

noncomputable def mode (list : List ℝ) : ℝ :=
  list.foldl (λ acc x => if list.count x > list.count acc then x else acc) (list.head!)

theorem variance_of_dataset :
  mode dataSet = 15 → variance dataSet = 16/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_dataset_l558_55869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tourist_contact_probability_l558_55845

/-- Represents whether a tourist from the first group has the phone number of a tourist from the second group -/
def TouristContact (i : Fin 5) (j : Fin 8) : Prop := sorry

/-- Represents the probability of an event occurring -/
noncomputable def Prob (A : Prop) : ℝ := sorry

/-- The probability that two groups of tourists can contact each other -/
theorem tourist_contact_probability (p : ℝ) : 
  0 ≤ p → p ≤ 1 → 
  (1 - (1 - p) ^ (5 * 8)) = 
  Prob (∃ (i : Fin 5) (j : Fin 8), TouristContact i j) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tourist_contact_probability_l558_55845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_vertex_l558_55868

noncomputable def quadratic_function (a b c : ℝ) : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c

noncomputable def axis_of_symmetry (a b : ℝ) : ℝ := -b / (2 * a)

noncomputable def vertex (a b c : ℝ) : ℝ × ℝ := 
  let x := axis_of_symmetry a b
  (x, quadratic_function a b c x)

theorem quadratic_vertex (a b c : ℝ) :
  quadratic_function a b c 2 = 5 ∧ axis_of_symmetry a b = 2 → vertex a b c = (2, 5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_vertex_l558_55868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_RQ_l558_55810

/-- Circle C in polar coordinates -/
noncomputable def circle_C (θ : ℝ) : ℝ := 2 * Real.cos θ

/-- Line l in polar coordinates -/
def line_l (ρ θ : ℝ) : Prop := 2 * ρ * Real.sin (θ + Real.pi/3) = 3 * Real.sqrt 3

/-- Ray OM -/
noncomputable def ray_OM : ℝ := Real.pi/3

/-- Point R: intersection of circle C and ray OM -/
noncomputable def R : ℝ := circle_C ray_OM

/-- Point Q: intersection of line l and ray OM -/
noncomputable def Q : ℝ := 3

theorem length_RQ : |R - Q| = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_RQ_l558_55810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_skew_lines_common_perpendiculars_l558_55817

-- Define the concept of a line in 3D space
structure Line3D where
  point : Fin 3 → ℝ
  direction : Fin 3 → ℝ

-- Define the concept of skew lines
def areSkew (l1 l2 : Line3D) : Prop :=
  -- Two lines are skew if they are not parallel and do not intersect
  sorry

-- Define perpendicularity for lines
def perp (l1 l2 : Line3D) : Prop :=
  -- Two lines are perpendicular if their direction vectors are orthogonal
  sorry

theorem skew_lines_common_perpendiculars 
  (a b c : Line3D) 
  (a_prime b_prime c_prime : Line3D)
  (skew_ab : areSkew a b)
  (skew_bc : areSkew b c)
  (skew_ca : areSkew c a)
  (perp_a_prime_b : perp a_prime b)
  (perp_a_prime_c : perp a_prime c)
  (perp_b_prime_c : perp b_prime c)
  (perp_b_prime_a : perp b_prime a)
  (perp_c_prime_a : perp c_prime a)
  (perp_c_prime_b : perp c_prime b) :
  (perp a b_prime ∧ perp a c_prime) ∧
  (perp b c_prime ∧ perp b a_prime) ∧
  (perp c a_prime ∧ perp c b_prime) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_skew_lines_common_perpendiculars_l558_55817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_counterexample_l558_55842

-- Define the types for our geometric objects
structure Plane : Type
structure Line : Type

-- Define the relationships between geometric objects
axiom perpendicular_plane_plane : Plane → Plane → Prop
axiom parallel_plane_line : Plane → Line → Prop
axiom perpendicular_plane_line : Plane → Line → Prop

-- State the theorem
theorem geometric_counterexample : 
  ∃ (x y : Plane) (z : Line), 
    perpendicular_plane_plane x y ∧ 
    parallel_plane_line y z ∧ 
    ¬ perpendicular_plane_line x z := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_counterexample_l558_55842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_line_is_2sqrt5_div_5_l558_55812

/-- A circle passing through (2,1) and tangent to both coordinate axes -/
structure TangentCircle where
  center : ℝ × ℝ
  radius : ℝ
  passes_through : (center.1 - 2)^2 + (center.2 - 1)^2 = radius^2
  tangent_to_axes : center.1 = radius ∧ center.2 = radius
  in_first_quadrant : center.1 > 0 ∧ center.2 > 0

/-- The line 2x-y-3=0 -/
def target_line (x y : ℝ) : Prop := 2*x - y - 3 = 0

/-- Distance from a point to a line -/
noncomputable def distance_to_line (p : ℝ × ℝ) : ℝ :=
  |2*p.1 - p.2 - 3| / Real.sqrt 5

theorem distance_to_line_is_2sqrt5_div_5 (c : TangentCircle) :
  distance_to_line c.center = 2 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_line_is_2sqrt5_div_5_l558_55812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_debony_bike_time_l558_55802

/-- Represents the travel scenario for Debony --/
structure TravelScenario where
  normalDriveTime : ℚ  -- in minutes
  normalDriveSpeed : ℚ  -- in miles per hour
  bikeRouteReduction : ℚ  -- percentage reduction in distance
  minBikeSpeed : ℚ  -- in miles per hour
  maxBikeSpeed : ℚ  -- in miles per hour

/-- Calculates the extra time needed for biking --/
noncomputable def extraTimeBiking (scenario : TravelScenario) : ℚ :=
  let driveDistance := scenario.normalDriveSpeed * (scenario.normalDriveTime / 60)
  let bikeDistance := driveDistance * (1 - scenario.bikeRouteReduction / 100)
  let bikeTime := (bikeDistance / scenario.minBikeSpeed) * 60
  bikeTime - scenario.normalDriveTime

/-- Theorem stating that Debony needs to leave 75 minutes earlier when biking --/
theorem debony_bike_time (scenario : TravelScenario) 
    (h1 : scenario.normalDriveTime = 45)
    (h2 : scenario.normalDriveSpeed = 40)
    (h3 : scenario.bikeRouteReduction = 20)
    (h4 : scenario.minBikeSpeed = 12)
    (h5 : scenario.maxBikeSpeed = 16) :
    extraTimeBiking scenario = 75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_debony_bike_time_l558_55802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_difference_is_90_l558_55881

/-- Rounds a number to the nearest multiple of 5, rounding 5s up -/
def roundToNearestFive (n : ℕ) : ℕ :=
  ((n + 2) / 5) * 5

/-- Sums up all integers from 1 to n -/
def sumToN (n : ℕ) : ℕ :=
  n * (n + 1) / 2

/-- Sums up all integers from 1 to n after rounding each to the nearest multiple of 5 -/
def sumRoundedToN (n : ℕ) : ℕ :=
  (List.range n).map (fun i ↦ roundToNearestFive (i + 1)) |>.sum

theorem sum_difference_is_90 :
  sumRoundedToN 50 - sumToN 50 = 90 := by
  sorry

#eval sumRoundedToN 50 - sumToN 50

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_difference_is_90_l558_55881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_equality_l558_55825

-- Define the basic structures
variable (A B C D M : ℂ)

-- Define the conditions
def IsAcute (A B C : ℂ) : Prop := sorry
def OnCircumcircle (A B C : ℂ) : Prop := sorry
def IsTangentIntersection (A B C D : ℂ) : Prop := sorry
def IsMidpoint (M A B : ℂ) : Prop := sorry

-- Define angle measure
def angle_measure (A B C : ℂ) : ℝ := sorry

-- Axioms for the given conditions
axiom acute_triangle : IsAcute A B C
axiom on_circumcircle : OnCircumcircle A B C
axiom D_is_tangent_intersection : IsTangentIntersection A B C D
axiom M_is_midpoint : IsMidpoint M A B

-- Define the theorem
theorem angle_equality : angle_measure A C M = angle_measure B C D := by
  sorry

-- where:
-- IsAcute A B C means triangle ABC is acute
-- OnCircumcircle A B C means A and B are on the circumcircle of triangle ABC
-- IsTangentIntersection A B C D means D is the intersection of tangents at A and B
-- IsMidpoint M A B means M is the midpoint of AB
-- angle_measure A C M represents the angle ACM
-- angle_measure B C D represents the angle BCD

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_equality_l558_55825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_x_in_terms_of_a_and_b_l558_55882

theorem sin_x_in_terms_of_a_and_b (a b x : ℝ) 
  (h1 : Real.tan x = (3 * a * b) / (a^2 - b^2))
  (h2 : a > b)
  (h3 : b > 0)
  (h4 : 0 < x)
  (h5 : x < π/2) :
  Real.sin x = (3 * a * b) / Real.sqrt (a^4 + 7 * a^2 * b^2 + b^4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_x_in_terms_of_a_and_b_l558_55882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_negative_b_l558_55849

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then -2 * x else 3 * x - 50

-- State the theorem
theorem unique_negative_b :
  ∃! b : ℝ, b < 0 ∧ f (f 15) = f (f b) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_negative_b_l558_55849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_beetle_shortest_path_l558_55858

/-- Represents a right circular cone -/
structure Cone where
  baseRadius : ℝ
  height : ℝ

/-- Represents a point on the surface of a cone -/
structure ConePoint where
  distanceFromVertex : ℝ

/-- Calculates the shortest path between two points on a cone's surface -/
noncomputable def shortestPathOnCone (cone : Cone) (start : ConePoint) (finish : ConePoint) : ℝ :=
  sorry

/-- Theorem stating the shortest path for the given problem -/
theorem beetle_shortest_path :
  let cone : Cone := { baseRadius := 800, height := 300 * Real.sqrt 3 }
  let start : ConePoint := { distanceFromVertex := 150 }
  let finish : ConePoint := { distanceFromVertex := 450 * Real.sqrt 2 }
  abs (shortestPathOnCone cone start finish - 562.158) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_beetle_shortest_path_l558_55858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_swimmer_speed_theorem_l558_55815

/-- Represents the speed of a swimmer in still water given downstream and upstream distances and times. -/
noncomputable def swimmerSpeed (downstreamDistance : ℝ) (upstreamDistance : ℝ) (downstreamTime : ℝ) (upstreamTime : ℝ) : ℝ :=
  (downstreamDistance + upstreamDistance) / (downstreamTime + upstreamTime)

/-- Theorem stating that a swimmer who covers 48 km downstream and 18 km upstream, 
    both in 3 hours each, has a speed of 11 km/h in still water. -/
theorem swimmer_speed_theorem :
  swimmerSpeed 48 18 3 3 = 11 := by
  unfold swimmerSpeed
  norm_num

-- The following line is commented out as it's not computable
-- #eval swimmerSpeed 48 18 3 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_swimmer_speed_theorem_l558_55815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lilac_mixture_theorem_l558_55855

/-- Calculates the amount of white paint added to a lilac mixture -/
noncomputable def white_paint_amount (blue_percent : ℝ) (red_percent : ℝ) (blue_amount : ℝ) : ℝ :=
  let white_percent := 1 - blue_percent - red_percent
  (white_percent * blue_amount) / blue_percent

/-- Theorem: Given a lilac mixture with 70% blue paint, 20% red paint, 
    and 140 ounces of blue paint added, the amount of white paint added is 20 ounces -/
theorem lilac_mixture_theorem :
  white_paint_amount 0.7 0.2 140 = 20 := by
  -- Unfold the definition of white_paint_amount
  unfold white_paint_amount
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lilac_mixture_theorem_l558_55855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_is_17_l558_55884

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The square defined by four vertices -/
structure Square where
  P : Point
  Q : Point
  R : Point
  S : Point

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Calculate the area of a square given its side length -/
def squareArea (side : ℝ) : ℝ :=
  side^2

/-- The theorem stating that the area of the given square is 17 square units -/
theorem square_area_is_17 (square : Square) (h1 : square.P = ⟨1, 1⟩) (h2 : square.Q = ⟨-3, 2⟩)
    (h3 : square.R = ⟨-2, -3⟩) (h4 : square.S = ⟨2, -2⟩) :
    squareArea (distance square.P square.Q) = 17 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_is_17_l558_55884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_digit_periodic_l558_55896

theorem last_digit_periodic (n : ℕ) : 
  (n^n)^n % 10 = ((n + 10)^(n + 10))^(n + 10) % 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_digit_periodic_l558_55896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_number_with_remainder_l558_55834

theorem least_number_with_remainder (n : ℕ) : n ≥ 40 ∧ 
  (∀ d ∈ ({6, 9, 12, 18} : Set ℕ), n % d = 4) →
  ∀ m : ℕ, m < n → ∃ d ∈ ({6, 9, 12, 18} : Set ℕ), m % d ≠ 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_number_with_remainder_l558_55834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_ellipse_same_foci_eccentricity_l558_55891

-- Define the hyperbola and ellipse
def hyperbola (a : ℝ) (x y : ℝ) : Prop := x^2 / a - y^2 / 2 = 1
def ellipse (x y : ℝ) : Prop := x^2 / 6 + y^2 / 2 = 1

-- Define the eccentricity of a hyperbola
noncomputable def eccentricity (a : ℝ) : ℝ := Real.sqrt (1 + 2 / a)

-- Theorem statement
theorem hyperbola_ellipse_same_foci_eccentricity (a : ℝ) :
  (∃ c : ℝ, c^2 = 4 ∧
    (∀ x y : ℝ, hyperbola a x y ↔ x^2 - y^2 = a * (1 + y^2 / 2)) ∧
    (∀ x y : ℝ, ellipse x y ↔ x^2 + 3 * y^2 = 6)) →
  eccentricity a = Real.sqrt 2 :=
by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_ellipse_same_foci_eccentricity_l558_55891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_horizontal_asymptote_and_oblique_asymptote_l558_55835

noncomputable def f (x : ℝ) : ℝ := (15*x^5 + 12*x^4 + 4*x^3 + 9*x^2 + 5*x + 3) / (3*x^4 + 2*x^3 + 8*x^2 + 3*x + 1)

theorem no_horizontal_asymptote_and_oblique_asymptote (ε : ℝ) (h : ε > 0) :
  ∃ M : ℝ, ∀ x : ℝ, x > M → |f x - (5*x)| < ε * |x| :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_horizontal_asymptote_and_oblique_asymptote_l558_55835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_relation_l558_55888

-- Define the points A, B, and C
noncomputable def A (x₁ : ℝ) : ℝ × ℝ := (x₁, 1 / x₁)
noncomputable def B (x₂ : ℝ) : ℝ × ℝ := (x₂, 1 / x₂)
noncomputable def C (x₁ x₂ : ℝ) : ℝ × ℝ := ((x₁ + x₂) / 2, (x₁ + x₂) / (2 * x₁ * x₂))

-- Define the theorem
theorem angle_relation (x₁ x₂ : ℝ) (h₁ : 0 < x₁) (h₂ : x₁ < x₂) :
  Real.arctan (1 / x₁^2) = 3 * Real.arctan (1 / (x₁ * x₂)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_relation_l558_55888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_factorial_150_l558_55887

theorem units_digit_factorial_150 :
  ∃ k : ℕ, (Nat.factorial 150) = 10 * k :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_factorial_150_l558_55887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jacksons_chore_payment_l558_55809

/-- Jackson's chore payment problem -/
theorem jacksons_chore_payment (vacuum_time cleaning_time dish_time bathroom_time total_time total_pay : ℝ) 
  (h1 : vacuum_time = 2 * 2)
  (h2 : dish_time = 0.5)
  (h3 : bathroom_time = 3 * dish_time)
  (h4 : total_time = vacuum_time + dish_time + bathroom_time)
  (h5 : total_pay = 30) : 
  total_pay / total_time = 5 := by
  sorry

#check jacksons_chore_payment

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jacksons_chore_payment_l558_55809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_graph_shift_l558_55829

theorem cos_graph_shift (x : ℝ) : 
  Real.cos (2*x - π/2) = Real.cos (2*(x - π/4)) :=
by
  have h : 2*x - π/2 = 2*(x - π/4) := by
    ring
  rw [h]


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_graph_shift_l558_55829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l558_55862

noncomputable section

/-- The function f(x) = 2sin(π/6 - 2x) -/
def f (x : ℝ) : ℝ := 2 * Real.sin (Real.pi / 6 - 2 * x)

/-- The increasing interval of f(x) -/
def increasing_interval (k : ℤ) : Set ℝ := 
  Set.Icc (k * Real.pi + Real.pi / 3) (k * Real.pi + 5 * Real.pi / 6)

theorem f_increasing_on_interval (k : ℤ) :
  StrictMonoOn f (increasing_interval k) := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l558_55862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_study_hours_l558_55851

-- Define the investment variables
variable (a b c : ℝ)

-- Define the constraint
def total_investment (a b c : ℝ) : Prop := a + b + c = 5

-- Define the study hours function
def study_hours (a b c : ℝ) : ℝ := 5*a + 3*b + (11*c - c^2)

-- State the theorem
theorem max_study_hours :
  ∀ a b c : ℝ,
  total_investment a b c →
  study_hours a b c ≤ 34 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_study_hours_l558_55851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_and_positivity_l558_55894

noncomputable def f (a x : ℝ) : ℝ := a^2 * x^2 + a*x - 3 * Real.log x + 1

noncomputable def f' (a x : ℝ) : ℝ := 2*a^2*x + a - 3/x

theorem f_monotonicity_and_positivity (a : ℝ) (ha : a > 0) :
  (∀ x ∈ Set.Ioo (0 : ℝ) (1/a), HasDerivAt (f a) (f' a x) x ∧ f' a x < 0) ∧
  (∀ x ∈ Set.Ioi (1/a), HasDerivAt (f a) (f' a x) x ∧ f' a x > 0) ∧
  (∀ x > 0, f a x > 0 ↔ a ∈ Set.Ioi (Real.exp (-1))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_and_positivity_l558_55894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_AQC_is_45_or_90_l558_55854

structure CircleConfiguration where
  c : Set (ℝ × ℝ)  -- The main circle
  O : ℝ × ℝ      -- Center of circle c
  A : ℝ × ℝ      -- Point on circle c
  B : ℝ × ℝ      -- Point on circle c
  Q : ℝ × ℝ      -- Point where the smaller circle touches c
  C : ℝ × ℝ      -- Point where the smaller circle touches OA or OB
  D : ℝ × ℝ      -- Point where the smaller circle touches OB or OA

def is_valid_configuration (config : CircleConfiguration) : Prop :=
  -- O is the center of c
  config.O ∈ config.c ∧
  -- A and B are on c
  config.A ∈ config.c ∧ config.B ∈ config.c ∧
  -- OA and OB are perpendicular
  (config.A.1 - config.O.1) * (config.B.1 - config.O.1) + 
  (config.A.2 - config.O.2) * (config.B.2 - config.O.2) = 0 ∧
  -- Q is on c
  config.Q ∈ config.c ∧
  -- C is on OA or OB
  (∃ t : ℝ, config.C = (config.O.1 + t * (config.A.1 - config.O.1), 
                        config.O.2 + t * (config.A.2 - config.O.2))) ∨
  (∃ t : ℝ, config.C = (config.O.1 + t * (config.B.1 - config.O.1), 
                        config.O.2 + t * (config.B.2 - config.O.2))) ∧
  -- D is on OB or OA (whichever C is not on)
  (∃ t : ℝ, config.D = (config.O.1 + t * (config.B.1 - config.O.1), 
                        config.O.2 + t * (config.B.2 - config.O.2))) ∨
  (∃ t : ℝ, config.D = (config.O.1 + t * (config.A.1 - config.O.1), 
                        config.O.2 + t * (config.A.2 - config.O.2)))

noncomputable def angle (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

theorem angle_AQC_is_45_or_90 (config : CircleConfiguration) 
  (h : is_valid_configuration config) :
  angle config.A config.Q config.C = 45 ∨ 
  angle config.A config.Q config.C = 90 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_AQC_is_45_or_90_l558_55854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_x_value_l558_55897

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = (k * b.1, k * b.2)

/-- The problem statement -/
theorem parallel_vectors_x_value (x : ℝ) :
  let a : ℝ × ℝ := (3, 1)
  let b : ℝ × ℝ := (x, -3)
  are_parallel a b → x = -9 :=
by
  intro h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_x_value_l558_55897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_of_angles_l558_55898

theorem sin_sum_of_angles (α β : Real) : 
  0 < α ∧ α < Real.pi/2 →
  0 < β ∧ β < Real.pi/2 →
  Real.sin α = 5/13 →
  Real.cos β = 4/5 →
  Real.sin (α + β) = 56/65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_of_angles_l558_55898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_trees_divisible_by_43_l558_55839

/-- Represents the number of cones on each tree initially -/
def m : ℕ := sorry

/-- Represents the number of pine trees -/
def s : ℕ := sorry

/-- Represents the number of cedar trees -/
def k : ℕ := sorry

/-- Represents the number of larch trees -/
def l : ℕ := sorry

/-- The proportion of cones that fell from pine trees -/
def pine_fall_ratio : ℚ := 11 / 100

/-- The proportion of cones that fell from cedar trees -/
def cedar_fall_ratio : ℚ := 54 / 100

/-- The proportion of cones that fell from larch trees -/
def larch_fall_ratio : ℚ := 97 / 100

/-- The overall proportion of cones that fell from all trees -/
def total_fall_ratio : ℚ := 30 / 100

/-- The theorem stating that the total number of trees is divisible by 43 -/
theorem total_trees_divisible_by_43 (h : total_fall_ratio * (s + k + l : ℚ) = 
  pine_fall_ratio * s + cedar_fall_ratio * k + larch_fall_ratio * l) : 
  43 ∣ (s + k + l) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_trees_divisible_by_43_l558_55839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_points_at_distance_one_l558_55811

/-- The circle equation -/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2*y - 2 = 0

/-- The line equation -/
def line_eq (x y a : ℝ) : Prop := x + y + a = 0

/-- The distance from a point (x, y) to the line x + y + a = 0 -/
noncomputable def distance_to_line (x y a : ℝ) : ℝ := |x + y + a| / Real.sqrt 2

/-- The theorem stating the condition for exactly three points on the circle
    to have a distance of 1 from the line -/
theorem three_points_at_distance_one (a : ℝ) :
  (∃! (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ),
    circle_eq x₁ y₁ ∧ circle_eq x₂ y₂ ∧ circle_eq x₃ y₃ ∧
    distance_to_line x₁ y₁ a = 1 ∧
    distance_to_line x₂ y₂ a = 1 ∧
    distance_to_line x₃ y₃ a = 1) ↔
  (a = -2 - Real.sqrt 2 ∨ a = -2 + Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_points_at_distance_one_l558_55811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_salesman_remittance_l558_55846

/-- Calculates the amount remitted to the parent company given the total sales and commission structure. -/
noncomputable def amount_remitted (total_sales : ℝ) (commission_rate_low : ℝ) (commission_rate_high : ℝ) (threshold : ℝ) : ℝ :=
  let commission_low := commission_rate_low * min total_sales threshold
  let commission_high := commission_rate_high * max (total_sales - threshold) 0
  let total_commission := commission_low + commission_high
  total_sales - total_commission

/-- Theorem stating that the amount remitted to the parent company is 31100 given the specific sales and commission structure. -/
theorem salesman_remittance :
  amount_remitted 32500 0.05 0.04 10000 = 31100 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval amount_remitted 32500 0.05 0.04 10000

end NUMINAMATH_CALUDE_ERRORFEEDBACK_salesman_remittance_l558_55846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_school_journey_time_l558_55813

/-- Represents the time taken for Joe's journey to school -/
noncomputable def school_journey (total_distance : ℝ) (walking_speed : ℝ) : ℝ :=
  let walking_time := (2/3) * total_distance / walking_speed
  let waiting_time := 3
  let running_speed := 4 * walking_speed
  let running_time := (1/3) * total_distance / running_speed
  walking_time + waiting_time + running_time

/-- Theorem stating that Joe's journey to school takes 16.5 minutes -/
theorem school_journey_time : ∃ (d : ℝ) (w : ℝ), 
  d > 0 ∧ w > 0 ∧ (2/3) * d / w = 12 ∧ school_journey d w = 16.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_school_journey_time_l558_55813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_solution_set_implies_k_x_3_solution_implies_k_range_l558_55873

-- Define the inequality
def inequality (x k : ℝ) : Prop :=
  (x + 2) / k > 1 + (x - 3) / k^2

-- Part 1: Solution to the inequality
theorem inequality_solution (k : ℝ) (hk : k ≠ 0) :
  (∀ x, inequality x k ↔ 
    (k > 1 ∧ x > (k^2 - 2*k - 3) / (k - 1)) ∨
    (k < 1 ∧ x < (k^2 - 2*k - 3) / (k - 1)) ∨
    (k = 1 ∧ x ∈ Set.univ)) :=
by sorry

-- Part 2: Solution set (3, +∞) implies k = 5
theorem solution_set_implies_k (k : ℝ) (hk : k ≠ 0) :
  (∀ x, inequality x k ↔ x > 3) → k = 5 :=
by sorry

-- Part 3: x = 3 is a solution implies 0 < k < 5
theorem x_3_solution_implies_k_range (k : ℝ) (hk : k ≠ 0) :
  inequality 3 k → 0 < k ∧ k < 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_solution_set_implies_k_x_3_solution_implies_k_range_l558_55873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_kings_on_12x12_board_l558_55866

/-- Represents a chessboard --/
structure Chessboard :=
  (size : ℕ)

/-- Represents a king on the chessboard --/
structure King :=
  (x : ℕ)
  (y : ℕ)

/-- Two kings attack each other if their squares share at least one common vertex --/
def attacks (k1 k2 : King) : Prop :=
  (k1.x = k2.x ∨ k1.x = k2.x + 1 ∨ k1.x + 1 = k2.x) ∧
  (k1.y = k2.y ∨ k1.y = k2.y + 1 ∨ k1.y + 1 = k2.y)

/-- A valid placement of kings on the chessboard --/
def valid_placement (board : Chessboard) (kings : List King) : Prop :=
  (∀ k, k ∈ kings → k.x < board.size ∧ k.y < board.size) ∧
  (∀ k, k ∈ kings → ∃! k', k' ∈ kings ∧ k ≠ k' ∧ attacks k k')

/-- The maximum number of kings that can be placed on a 12x12 chessboard
    such that each king attacks exactly one other king is 56 --/
theorem max_kings_on_12x12_board :
  ∃ (kings : List King),
    valid_placement (Chessboard.mk 12) kings ∧
    kings.length = 56 ∧
    (∀ kings' : List King, valid_placement (Chessboard.mk 12) kings' → kings'.length ≤ 56) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_kings_on_12x12_board_l558_55866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_s_of_3_equals_11_l558_55875

-- Define the functions t and s
noncomputable def t (x : ℝ) : ℝ := 4 * x - 5
noncomputable def s (y : ℝ) : ℝ := 
  let x := (y + 5) / 4  -- Inverse of t(x)
  x^2 + 4*x - 1

-- Theorem to prove
theorem s_of_3_equals_11 : s 3 = 11 := by
  -- Unfold the definition of s
  unfold s
  -- Simplify the expression
  simp
  -- Prove the equality
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_s_of_3_equals_11_l558_55875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_equals_sqrt65_div_5_l558_55848

/-- The projection of vector a onto the direction of vector b -/
noncomputable def vector_projection (a b : ℝ × ℝ) : ℝ :=
  let a_dot_b := a.1 * b.1 + a.2 * b.2
  let magnitude_a := Real.sqrt (a.1^2 + a.2^2)
  let magnitude_b := Real.sqrt (b.1^2 + b.2^2)
  magnitude_a * (a_dot_b / (magnitude_a * magnitude_b))

/-- Theorem: The projection of vector (2, 3) onto the direction of vector (-4, 7) is √65 / 5 -/
theorem projection_equals_sqrt65_div_5 :
  vector_projection (2, 3) (-4, 7) = Real.sqrt 65 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_equals_sqrt65_div_5_l558_55848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l558_55889

/-- The speed of the train in km/hr -/
noncomputable def train_speed : ℝ := 40

/-- The length of the train in meters -/
noncomputable def train_length : ℝ := 190.0152

/-- Converts km/hr to m/s -/
noncomputable def km_hr_to_m_s (v : ℝ) : ℝ := v * 1000 / 3600

/-- Calculates the time taken for the train to cross a post -/
noncomputable def time_to_cross (speed : ℝ) (length : ℝ) : ℝ :=
  length / (km_hr_to_m_s speed)

/-- Theorem stating that the time taken for the train to cross a post is approximately 17.1 seconds -/
theorem train_crossing_time :
  abs (time_to_cross train_speed train_length - 17.1) < 0.1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l558_55889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_runner_speed_relation_l558_55880

/-- Represents a runner in the race -/
structure Runner where
  speed : ℝ
  initialPosition : ℝ

/-- Represents the race course -/
structure RaceCourse where
  straightPathLength : ℝ
  lapLength : ℝ

/-- Calculates the position of a runner at a given time -/
def position (runner : Runner) (course : RaceCourse) (time : ℝ) : ℝ :=
  runner.initialPosition + runner.speed * time

/-- Determines if one runner has overtaken another at a given time -/
def hasOvertaken (runner1 runner2 : Runner) (course : RaceCourse) (time : ℝ) : Prop :=
  ∃ n : ℤ, (n : ℝ) * course.lapLength < position runner1 course time - position runner2 course time ∧
            position runner1 course time - position runner2 course time ≤ (n + 1 : ℝ) * course.lapLength

theorem runner_speed_relation (runner1 runner2 : Runner) (course : RaceCourse) :
  runner1.initialPosition = runner2.initialPosition →
  runner1.speed > runner2.speed →
  (∃ t1 t2 : ℝ, t1 < t2 ∧
    hasOvertaken runner1 runner2 course t1 ∧
    hasOvertaken runner1 runner2 course t2) →
  runner1.speed ≥ 2 * runner2.speed := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_runner_speed_relation_l558_55880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_balance_difference_theorem_l558_55843

/-- Calculates the balance after compound interest --/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- Calculates the positive difference between two real numbers --/
def positive_difference (a b : ℝ) : ℝ :=
  abs (a - b)

/-- Rounds a real number to the nearest integer --/
noncomputable def round_to_nearest (x : ℝ) : ℤ :=
  Int.floor (x + 0.5)

theorem balance_difference_theorem (alice_principal charlie_principal : ℝ)
    (alice_rate charlie_rate : ℝ) (time : ℕ) :
    alice_principal = 6000 →
    charlie_principal = 8000 →
    alice_rate = 0.05 →
    charlie_rate = 0.045 →
    time = 15 →
    round_to_nearest (positive_difference
      (compound_interest alice_principal alice_rate time)
      (compound_interest charlie_principal charlie_rate time)) = 2945 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_balance_difference_theorem_l558_55843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l558_55831

noncomputable def f (x : ℝ) := (x^3 - 3*x^2 + 2*x + 5) / (x^3 - 6*x^2 + 11*x - 6)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x | x < 1 ∨ (1 < x ∧ x < 2) ∨ (2 < x ∧ x < 3) ∨ 3 < x} :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l558_55831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_equation_m_range_l558_55856

theorem sin_equation_m_range (m : ℝ) : 
  (∃ x : ℝ, Real.sin x ^ 2 + 2 * Real.sin x - 1 + m = 0) → 
  -2 ≤ m ∧ m ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_equation_m_range_l558_55856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_speed_theorem_l558_55808

-- Define constants
noncomputable def wheel_circumference : ℝ := 15  -- in feet
noncomputable def speed_increase : ℝ := 8  -- in miles per hour
noncomputable def time_decrease : ℝ := 1 / 5  -- in seconds

-- Define the original speed (to be proved)
noncomputable def original_speed : ℝ := 15  -- in miles per hour

-- Theorem statement
theorem wheel_speed_theorem :
  let circumference_miles := wheel_circumference / 5280
  let original_time := circumference_miles / original_speed * 3600
  let new_time := original_time - time_decrease / 3600
  let new_speed := original_speed + speed_increase
  circumference_miles = original_speed * original_time / 3600 ∧
  circumference_miles = new_speed * new_time / 3600 →
  original_speed = 15 := by
  sorry

#check wheel_speed_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_speed_theorem_l558_55808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_divisible_by_product_l558_55819

theorem not_divisible_by_product (n : ℕ) (a : ℕ) (a_list : List ℕ) 
  (h_pos : ∀ i, i ∈ a_list → i > 0)
  (h_a_gt_one : a > 1)
  (h_a_divisible : ∃ k : ℕ, a = k * (a_list.prod)) : 
  ¬ (∃ m : ℕ, a^(n+1) + a - 1 = m * ((a_list.map (λ x ↦ a + x - 1)).prod)) := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_divisible_by_product_l558_55819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_differential_form_is_exact_with_potential_l558_55838

open Real

-- Define the differential form
def P (x y : ℝ) : ℝ := 3 * x^2 * y + 4 * y^4 - 5
def Q (x y : ℝ) : ℝ := x^3 + 8 * x * y

-- Define the potential function
def f (x y : ℝ) : ℝ := x^3 * y + 4 * y^2 * x - 5 * x

-- Theorem statement
theorem differential_form_is_exact_with_potential (x y : ℝ) :
  (deriv (fun y => P x y) y) = (deriv (fun x => Q x y) x) ∧
  (deriv (fun x => f x y) x) = P x y ∧
  (deriv (fun y => f x y) y) = Q x y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_differential_form_is_exact_with_potential_l558_55838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_OP_l558_55820

-- Define the circle C
def C (x y : ℝ) : Prop := (x - 3)^2 + (y + 4)^2 = 4

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem max_distance_OP :
  ∃ (max_dist : ℝ), max_dist = 7 ∧
  (∀ (P : ℝ × ℝ), C P.1 P.2 → distance O P ≤ max_dist) ∧
  (∃ (P : ℝ × ℝ), C P.1 P.2 ∧ distance O P = max_dist) := by
  sorry

#check max_distance_OP

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_OP_l558_55820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l558_55824

noncomputable def f (x : ℝ) := Real.sin (8 * x + Real.pi / 4)

theorem problem_solution (α β : ℝ) 
  (h1 : ∀ x, f (x + α) = f x)  -- Period of f is α
  (h2 : Real.tan (α + β) = 1 / 3) :
  (1 - Real.cos (2 * β)) / Real.sin (2 * β) = -1 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l558_55824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_amount_in_paise_l558_55890

-- Define the conversion rate from rupees to paise
noncomputable def rupees_to_paise (rupees : ℝ) : ℝ := rupees * 100

-- Define the percentage calculation
noncomputable def percentage_of (percent : ℝ) (value : ℝ) : ℝ := (percent / 100) * value

-- Theorem statement
theorem amount_in_paise : 
  let a : ℝ := 170
  let percent : ℝ := 0.5
  rupees_to_paise (percentage_of percent a) = 85 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_amount_in_paise_l558_55890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radii_sum_l558_55821

/-- A circle represented by its center and radius. -/
structure Circle where
  center : EuclideanSpace ℝ (Fin 2)
  radius : ℝ

/-- A point in 2D Euclidean space. -/
abbrev Point := EuclideanSpace ℝ (Fin 2)

/-- Two circles touch internally at a point. -/
def TouchInternally (c₁ c₂ : Circle) (p : Point) : Prop :=
  dist p c₁.center = c₁.radius ∧
  dist p c₂.center = c₂.radius ∧
  dist c₁.center c₂.center = c₁.radius - c₂.radius

/-- A point is a member of a line segment defined by two other points. -/
def SegmentMember (p a b : Point) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ p = a + t • (b - a)

/-- Given circles S, S₁, and S₂, where S₁ and S₂ touch S internally at points A and B respectively,
    and one intersection point of S₁ and S₂ lies on segment AB, 
    prove that the radius of S is equal to the sum of the radii of S₁ and S₂. -/
theorem circle_radii_sum (S S₁ S₂ : Circle) (A B : Point) :
  TouchInternally S₁ S A →
  TouchInternally S₂ S B →
  (∃ C, dist C S₁.center = S₁.radius ∧ 
        dist C S₂.center = S₂.radius ∧ 
        SegmentMember C A B) →
  S.radius = S₁.radius + S₂.radius := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radii_sum_l558_55821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorizations_of_900_l558_55822

def factorizations (n : ℕ) : ℕ :=
  (Finset.filter (λ p : ℕ × ℕ => p.1 * p.2 = n ∧ p.1 ≤ p.2) (Finset.product (Finset.range (n+1)) (Finset.range (n+1)))).card

theorem factorizations_of_900 : factorizations 900 = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorizations_of_900_l558_55822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l558_55807

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin x + Real.cos x

theorem smallest_positive_period_of_f :
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  T = 2 * Real.pi :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l558_55807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_strictly_increasing_condition_local_maximum_at_negative_one_local_minimum_at_three_l558_55818

-- Define the function f(x) with parameter a
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 - x^2 + a*x - a

-- Theorem for part I
theorem strictly_increasing_condition (a : ℝ) :
  (∀ x : ℝ, StrictMono (f a)) ↔ a ≥ 1 := by
  sorry

-- Theorems for part II
theorem local_maximum_at_negative_one :
  f (-3) (-1) = 14/3 ∧ IsLocalMax (f (-3)) (-1) := by
  sorry

theorem local_minimum_at_three :
  f (-3) 3 = -6 ∧ IsLocalMin (f (-3)) 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_strictly_increasing_condition_local_maximum_at_negative_one_local_minimum_at_three_l558_55818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_sqrt_two_over_two_l558_55876

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- A point on the ellipse -/
structure PointOnEllipse (e : Ellipse) where
  x : ℝ
  y : ℝ
  h_on_ellipse : x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - e.b^2 / e.a^2)

theorem ellipse_eccentricity_sqrt_two_over_two 
  (e : Ellipse) 
  (P : PointOnEllipse e)
  (h_perpendicular : P.x = -Real.sqrt (e.a^2 - e.b^2)) 
  (h_parallel : P.y / P.x = -e.b / e.a) : 
  eccentricity e = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_sqrt_two_over_two_l558_55876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_sum_equation_l558_55893

theorem polynomial_sum_equation : ∃ n : ℕ, 2^(n + 1) = n + 1015 := by
  use 9
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_sum_equation_l558_55893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l558_55885

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sqrt (x^2 + 1) + x)

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : f (2*a - 2) + f b = 0) :
  ∃ (m : ℝ), m = (⨅ y ∈ {y | ∃ (a' b' : ℝ), a' > 0 ∧ b' > 0 ∧ f (2*a' - 2) + f b' = 0 ∧ y = (2*a' + b') / (a' * b')}, y) ∧ m = 4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l558_55885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2016_at_pi_div_3_l558_55859

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x

noncomputable def f_n : ℕ → (ℝ → ℝ)
| 0 => f
| n + 1 => deriv (f_n n)

theorem f_2016_at_pi_div_3 : 
  f_n 2016 (π / 3) = (Real.sqrt 3 + 1) / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2016_at_pi_div_3_l558_55859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_debate_team_arrangements_l558_55828

def debate_team_size : ℕ := 8
def male_members : ℕ := 3
def female_members : ℕ := 5

theorem debate_team_arrangements (n m f : ℕ) (h1 : n = m + f) (h2 : n = debate_team_size) 
  (h3 : m = male_members) (h4 : f = female_members) :
  (Nat.factorial f * (Nat.choose (f + 1) m * Nat.factorial m) = 14400) ∧
  ((Nat.choose n 2 * Nat.choose (n - 2) 2 * Nat.choose (n - 4) 2 * Nat.choose (n - 6) 2) / (Nat.factorial 4) * Nat.factorial 4 = 2520) ∧
  ((Nat.choose n 4 - Nat.choose f 4) * Nat.factorial 4 = 1560) :=
by sorry

#check debate_team_arrangements

end NUMINAMATH_CALUDE_ERRORFEEDBACK_debate_team_arrangements_l558_55828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_school_to_academy_l558_55833

/-- The distance between the school and the academy -/
def D : ℝ := sorry

/-- The speed to the academy in km/h -/
def speed_to_academy : ℝ := 15

/-- The speed from the academy to school in km/h -/
def speed_from_academy : ℝ := 10

/-- The additional time taken for the return trip in hours -/
def additional_time : ℝ := 0.5

theorem distance_school_to_academy :
  D = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_school_to_academy_l558_55833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricities_l558_55861

noncomputable def arithmetic_mean (x y : ℝ) : ℝ := (x + y) / 2

noncomputable def geometric_mean (x y : ℝ) : ℝ := Real.sqrt (x * y)

noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - (b^2 / a^2))

theorem ellipse_eccentricities :
  let a := arithmetic_mean 1 9
  let b := geometric_mean 1 9
  (eccentricity a b = Real.sqrt 10 / 5) ∧
  (eccentricity a (-b) = 2 * Real.sqrt 10 / 5) := by
  sorry

#check ellipse_eccentricities

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricities_l558_55861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_terminal_side_neg_527_l558_55857

/-- The set of angles with the same terminal side as a given angle -/
def SameTerminalSide (θ : ℝ) : Set ℝ :=
  {α | ∃ k : ℤ, α = θ + k * 360}

/-- Theorem: The set of angles with the same terminal side as -527° -/
theorem same_terminal_side_neg_527 :
  SameTerminalSide (-527) = {α : ℝ | ∃ k : ℤ, α = 193 + k * 360} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_terminal_side_neg_527_l558_55857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_journey_times_iff_20_minutes_l558_55852

/-- Represents a journey with a possible stop and speed change -/
structure Journey where
  distance : ℝ
  initial_speed : ℝ
  stop_time : ℝ
  speed_multiplier : ℝ

/-- Calculates the time taken for a journey -/
noncomputable def journey_time (j : Journey) : ℝ :=
  (j.distance / 2) / j.initial_speed + j.stop_time + (j.distance / 2) / (j.initial_speed * j.speed_multiplier)

/-- Theorem stating that two specific journeys take the same time if and only if that time is 20 minutes -/
theorem equal_journey_times_iff_20_minutes (d v : ℝ) (h_pos_d : d > 0) (h_pos_v : v > 0) :
  let j1 : Journey := { distance := d, initial_speed := v, stop_time := 0, speed_multiplier := 1 }
  let j2 : Journey := { distance := d, initial_speed := v, stop_time := 5, speed_multiplier := 2 }
  journey_time j1 = journey_time j2 ↔ journey_time j1 = 20 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_journey_times_iff_20_minutes_l558_55852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l558_55826

theorem problem_solution (a b c : ℝ) : 
  (((3*a + 2) : ℝ) ^ (1/3 : ℝ) = 2) →
  ((3*a + b - 1 : ℝ).sqrt = 3) →
  (c = Int.floor (Real.sqrt 2)) →
  (a = 2 ∧ b = 4 ∧ c = 1) ∧
  ((a + b - c : ℝ).sqrt = Real.sqrt 5 ∨ (a + b - c : ℝ).sqrt = -Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l558_55826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_interval_sum_l558_55830

noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 3 + x - 5

theorem root_interval_sum (a b : ℕ+) (x₀ : ℝ) : 
  (∃ x₀, f x₀ = 0 ∧ x₀ ∈ Set.Icc (a : ℝ) (b : ℝ)) →
  (b : ℝ) - (a : ℝ) = 1 →
  (a : ℕ) + (b : ℕ) = 7 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_interval_sum_l558_55830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l558_55841

/-- An ellipse with given foci and a point on its curve -/
structure Ellipse where
  foci : Fin 2 → ℝ × ℝ
  point_on_curve : ℝ × ℝ

/-- The standard form of an ellipse equation -/
def standard_equation (a b : ℝ) : ℝ → ℝ → Prop :=
  λ x y ↦ x^2 / a^2 + y^2 / b^2 = 1

/-- Theorem stating the standard equation of the ellipse -/
theorem ellipse_equation (e : Ellipse) 
  (h1 : e.foci 0 = (-3, 0))
  (h2 : e.foci 1 = (3, 0))
  (h3 : e.point_on_curve = (0, 2)) :
  standard_equation 13 4 = λ x y ↦ x^2 / 13 + y^2 / 4 = 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l558_55841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_form_l558_55895

theorem quadratic_root_form (a b c m n p : ℤ) : 
  (a = 3 ∧ b = -7 ∧ c = 1) →
  (∃ x : ℚ, a * x^2 + b * x + c = 0 ∧ 
    ∃ s : ℤ, x = (m + s * (n : ℚ).sqrt) / p ∨ x = (m - s * (n : ℚ).sqrt) / p) →
  (Int.gcd m (Int.gcd n p) = 1) →
  n = 37 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_form_l558_55895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_plane_relationships_l558_55806

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicularLineToPlane : Line → Plane → Prop)
variable (parallelPlanes : Plane → Plane → Prop)
variable (contains : Plane → Line → Prop)
variable (perpendicularLines : Line → Line → Prop)
variable (parallelLines : Line → Line → Prop)
variable (perpendicularPlanes : Plane → Plane → Prop)

-- State the theorem
theorem line_plane_relationships
  (m l : Line) (α β : Plane)
  (h1 : perpendicularLineToPlane m α)
  (h2 : contains β l) :
  (parallelPlanes α β → perpendicularLines m l) ∧
  (parallelLines m l → perpendicularPlanes α β) :=
by
  constructor
  · intro h_parallel
    sorry
  · intro h_parallel_lines
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_plane_relationships_l558_55806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2017_is_neg_sin_l558_55877

open Real

noncomputable def f (x : ℝ) : ℝ := cos x

noncomputable def f_n : ℕ → (ℝ → ℝ)
| 0 => f
| 1 => λ x => deriv (deriv f) x
| (n + 2) => λ x => deriv (deriv (f_n (n + 1))) x

theorem f_2017_is_neg_sin : f_n 2017 = λ x => -sin x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2017_is_neg_sin_l558_55877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_win_formula_prob_win_once_n_3_max_prob_at_n_2_l558_55837

-- Define the number of red balls
variable (n : ℕ)

-- Assume n is at least 2
axiom n_ge_two : n ≥ 2

-- Define the probability of winning in one draw
def prob_win (n : ℕ) : ℚ := (n^2 - n + 2) / (n^2 + 3*n + 2)

-- Define the probability of winning exactly once in three draws
def prob_win_once (p : ℚ) : ℚ := 3 * p * (1 - p)^2

-- Theorem 1: Probability of winning in one draw
theorem prob_win_formula : 
  ∀ n : ℕ, n ≥ 2 → prob_win n = (n^2 - n + 2) / (n^2 + 3*n + 2) := by
  sorry

-- Theorem 2: Probability of winning exactly once in three draws when n = 3
theorem prob_win_once_n_3 : 
  prob_win_once (prob_win 3) = 54 / 125 := by
  sorry

-- Theorem 3: Maximum probability occurs when n = 2
theorem max_prob_at_n_2 : 
  ∀ n : ℕ, n ≥ 2 → prob_win_once (prob_win n) ≤ prob_win_once (prob_win 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_win_formula_prob_win_once_n_3_max_prob_at_n_2_l558_55837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_in_fourth_quadrant_l558_55847

noncomputable def complex_number : ℂ := 1 / ((1 + Complex.I)^2 + 1) + Complex.I^4

theorem complex_number_in_fourth_quadrant :
  Real.sign (complex_number.re) = 1 ∧ Real.sign (complex_number.im) = -1 :=
by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_in_fourth_quadrant_l558_55847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_asphalt_cost_calculation_l558_55883

/-- Calculates the total cost of asphalt for a road, including sales tax. -/
noncomputable def total_asphalt_cost (road_length : ℝ) (road_width : ℝ) (coverage_per_truckload : ℝ) 
  (cost_per_truckload : ℝ) (tax_rate : ℝ) : ℝ :=
  let total_area := road_length * road_width
  let num_truckloads := total_area / coverage_per_truckload
  let cost_before_tax := num_truckloads * cost_per_truckload
  let tax_amount := cost_before_tax * tax_rate
  cost_before_tax + tax_amount

/-- Theorem stating the total cost of asphalt for the given road specifications. -/
theorem asphalt_cost_calculation :
  total_asphalt_cost 2000 20 800 75 0.2 = 4500 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_asphalt_cost_calculation_l558_55883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_taxi_fare_12km_l558_55879

/-- Taxi fare calculation function -/
noncomputable def taxi_fare (distance : ℝ) : ℝ :=
  let base_fare := 8
  let base_distance := 2
  let rate_per_km := 1.8
  if distance ≤ base_distance then
    base_fare
  else
    base_fare + (distance - base_distance) * rate_per_km

/-- Theorem stating that a 12 km taxi ride costs 26 yuan -/
theorem taxi_fare_12km : taxi_fare 12 = 26 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_taxi_fare_12km_l558_55879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_proof_l558_55832

/-- A moving straight line passing through (1,3) and intersecting y = x^2 -/
def movingLine (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 - 3 = k * (p.1 - 1)}

/-- The parabola y = x^2 -/
def parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = p.1^2}

/-- The circle x^2 + (y-2)^2 = 4 -/
def myCircle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + (p.2 - 2)^2 = 4}

/-- Point Q as a function of k -/
noncomputable def Q (k : ℝ) : ℝ × ℝ := (k/2, k-3)

/-- The minimum distance between Q and any point on the circle -/
noncomputable def minDistance (k : ℝ) : ℝ :=
  Real.sqrt 5 - 2

theorem min_distance_proof (k : ℝ) :
  ∀ p ∈ myCircle, dist (Q k) p ≥ minDistance k :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_proof_l558_55832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jean_money_approx_l558_55871

/-- Represents the exchange rate from euros to US dollars -/
noncomputable def exchange_rate : ℝ := 1.18

/-- Represents Jack's money in US dollars -/
noncomputable def jack_money : ℝ := 120

/-- Represents the total combined money in US dollars -/
noncomputable def total_money : ℝ := 256

/-- Represents Jane's money in US dollars -/
noncomputable def jane_money : ℝ := (total_money - jack_money) / (1 + 3 * exchange_rate)

/-- Represents Jean's money in euros -/
noncomputable def jean_money : ℝ := 3 * jane_money / exchange_rate

theorem jean_money_approx :
  ∃ ε > 0, |jean_money - 76.17| < ε :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jean_money_approx_l558_55871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_meet_time_l558_55864

/-- Calculates the time for two trains to meet given their lengths, initial distance, and speeds. -/
noncomputable def time_to_meet (length1 length2 initial_distance : ℝ) (speed1 speed2 : ℝ) : ℝ :=
  let speed1_ms := speed1 * 1000 / 3600
  let speed2_ms := speed2 * 1000 / 3600
  let relative_speed := speed1_ms + speed2_ms
  let total_distance := length1 + length2 + initial_distance
  total_distance / relative_speed

/-- Theorem stating that the time for two trains to meet is approximately 11.43 seconds. -/
theorem trains_meet_time :
  let length1 := (100 : ℝ)
  let length2 := (200 : ℝ)
  let initial_distance := (100 : ℝ)
  let speed1 := (54 : ℝ)
  let speed2 := (72 : ℝ)
  abs (time_to_meet length1 length2 initial_distance speed1 speed2 - 11.43) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_meet_time_l558_55864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l558_55805

theorem problem_statement (k m : ℕ) (n : ℕ) 
  (h1 : 18^k ∣ 624938)
  (h2 : 24^m ∣ 819304)
  (h3 : n = 2*k + m) :
  6^k - k^6 + 2^m - 4^m + n^3 - 3^n = 0 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l558_55805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shopkeeper_loss_percent_l558_55870

theorem shopkeeper_loss_percent 
  (initial_value : ℝ)
  (profit_rate : ℝ)
  (theft_rate : ℝ)
  (profit_rate_is_10_percent : profit_rate = 0.1)
  (theft_rate_is_70_percent : theft_rate = 0.7)
  (initial_value_positive : initial_value > 0) :
  let remaining_value := initial_value * (1 - theft_rate)
  let final_value := remaining_value * (1 + profit_rate)
  let loss := initial_value - final_value
  let loss_percent := (loss / initial_value) * 100
  loss_percent = 67 := by
  -- Proof steps would go here
  sorry

#check shopkeeper_loss_percent

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shopkeeper_loss_percent_l558_55870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_numbers_rational_l558_55899

theorem all_numbers_rational : 
  (∃ (a : ℚ), a^2 = 9) ∧ 
  (∃ (b : ℚ), b^3 = 512/1000) ∧ 
  (∃ (c : ℚ), c^4 = 1/16) ∧ 
  (∃ (d : ℚ), d = (-2) * 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_numbers_rational_l558_55899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l558_55804

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then x^2 - 4*x + 2
  else if x < 0 then -((-x)^2 - 4*(-x) + 2)
  else 0

-- State the theorem
theorem f_properties :
  (∀ x, f (-x) = -f x) →  -- f is odd
  (∀ x > 0, f x = x^2 - 4*x + 2) →  -- f(x) = x^2 - 4x + 2 for x > 0
  (∀ x y, -4 ≤ x ∧ x < y ∧ y ≤ -2 → f x < f y) ∧  -- f is monotonically increasing on [-4, -2]
  (∀ x, -4 ≤ x ∧ x ≤ -2 → f x ≤ 2) ∧  -- maximum value is 2
  (∀ x, -4 ≤ x ∧ x ≤ -2 → f x ≥ -2) ∧  -- minimum value is -2
  f (-2) = 2 ∧  -- maximum value occurs at -2
  f (-4) = -2  -- minimum value occurs at -4
:= by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l558_55804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_half_l558_55878

/-- Represents a regular tetrahedron -/
structure RegularTetrahedron where
  sideLength : ℝ
  sideLength_pos : sideLength > 0

/-- Volume of a regular tetrahedron -/
noncomputable def volume (t : RegularTetrahedron) : ℝ :=
  t.sideLength^3 * Real.sqrt 2 / 12

/-- Represents the four smaller tetrahedra formed within the original tetrahedron -/
structure SmallerTetrahedra (t : RegularTetrahedron) where
  count : Nat
  count_eq : count = 4
  sideLength : ℝ
  sideLength_eq : sideLength = t.sideLength / 2

/-- Total volume of the smaller tetrahedra -/
noncomputable def totalVolumeSmaller (t : RegularTetrahedron) (st : SmallerTetrahedra t) : ℝ :=
  st.count * (st.sideLength^3 * Real.sqrt 2 / 12)

/-- The theorem to be proved -/
theorem volume_ratio_half (t : RegularTetrahedron) (st : SmallerTetrahedra t) :
    totalVolumeSmaller t st / volume t = 1 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_half_l558_55878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_set_l558_55863

/-- An odd function f: ℝ → ℝ satisfying certain conditions -/
def f : ℝ → ℝ := sorry

/-- The derivative of f -/
def f' : ℝ → ℝ := sorry

/-- f is an odd function -/
axiom f_odd : ∀ x, f (-x) = -f x

/-- f(-2) = 0 -/
axiom f_neg_two_eq_zero : f (-2) = 0

/-- For x > 0, 3f(x) + xf'(x) > 0 -/
axiom f_condition : ∀ x, x > 0 → 3 * f x + x * f' x > 0

/-- The theorem to be proved -/
theorem f_positive_set : {x : ℝ | f x > 0} = Set.Ioo (-2) 0 ∪ Set.Ioi 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_set_l558_55863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l558_55865

-- Define the sets M and N
def M : Set ℝ := {x | ∃ y, y = Real.log x}
def N : Set ℝ := {x | ∃ y, y = Real.sqrt (1 - x)}

-- State the theorem
theorem intersection_M_N : M ∩ N = Set.Ioo 0 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l558_55865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_of_S_l558_55800

def S : Finset ℕ := {9, 99, 999, 9999, 99999}

theorem arithmetic_mean_of_S :
  (Finset.sum S id) / S.card = 22233 ∧
  ¬ (Nat.digits 10 22233).contains 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_of_S_l558_55800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_rook_free_square_l558_55850

/-- Represents a peaceful placement of rooks on an n × n chessboard -/
def PeacefulPlacement (n : ℕ) := Fin n → Fin n

/-- Checks if a k × k square starting at (row, col) is rook-free -/
def isRookFreeSquare (n k : ℕ) (placement : PeacefulPlacement n) (row col : Fin n) : Prop :=
  ∀ (i j : Fin k), placement (⟨(row.val + i.val) % n, by sorry⟩) ≠ ⟨(col.val + j.val) % n, by sorry⟩

/-- The main theorem statement -/
theorem largest_rook_free_square (n : ℕ) (hn : n ≥ 2) :
  let i := Nat.floor (Real.sqrt (n : ℝ))
  ∀ (k : ℕ), (∀ (placement : PeacefulPlacement n),
    ∃ (row col : Fin n), isRookFreeSquare n k placement row col) ↔ k ≤ i :=
by sorry

#check largest_rook_free_square

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_rook_free_square_l558_55850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_upper_bound_l558_55853

open Real

noncomputable def f (α : ℝ) (x : ℝ) : ℝ := x + α / x + log x

theorem f_upper_bound (l : ℝ) (m : ℝ) :
  (∀ α ∈ Set.Icc (1 / Real.exp 1) (2 * Real.exp 1 ^ 2),
    ∀ x ∈ Set.Icc l (Real.exp 1), f α x < m) →
  m > 1 + 2 * Real.exp 1 ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_upper_bound_l558_55853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_values_at_specific_points_l558_55823

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 
  (4 - x) * Real.sqrt x * (x - 1) ^ (1/3) * (2 * x - 1/2) ^ (1/6)

-- State the theorem
theorem f_values_at_specific_points :
  (∀ x : ℝ, x ≥ (1/4 : ℝ) → f x = (4 - x) * Real.sqrt x * (x - 1) ^ (1/3) * (2 * x - 1/2) ^ (1/6)) →
  f (2 + Real.sqrt 3) = 1 ∧ f (2 - Real.sqrt 3) = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_values_at_specific_points_l558_55823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_is_one_l558_55827

open Real Matrix

-- Define the matrix as a function of α, β, and φ
noncomputable def matrix (α β φ : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![cos (α + φ) * cos (β + φ), cos (α + φ) * sin (β + φ), -sin (α + φ)],
    ![-sin (β + φ), cos (β + φ), 0],
    ![sin (α + φ) * cos (β + φ), sin (α + φ) * sin (β + φ), cos (α + φ)]]

-- State the theorem
theorem determinant_is_one (α β φ : ℝ) :
  det (matrix α β φ) = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_is_one_l558_55827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_distinct_values_in_list_l558_55892

theorem least_distinct_values_in_list (n : ℕ) (mode_freq : ℕ) 
  (h1 : n = 2023) 
  (h2 : mode_freq = 15) : 
  (∃ (l : List ℕ), 
    (∀ x, x ∈ l → x > 0) ∧ 
    l.length = n ∧ 
    (∃! m, m ∈ l ∧ l.count m = mode_freq) ∧ 
    (∀ x, x ∈ l → x ≠ m → l.count x < mode_freq) ∧
    l.toFinset.card = 145) ∧ 
  (∀ (l : List ℕ), 
    (∀ x, x ∈ l → x > 0) → 
    l.length = n → 
    (∃! m, m ∈ l ∧ l.count m = mode_freq) → 
    (∀ x, x ∈ l → x ≠ m → l.count x < mode_freq) → 
    l.toFinset.card ≥ 145) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_distinct_values_in_list_l558_55892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_center_of_f_l558_55836

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 3 * Real.sin (ω * x + Real.pi / 6)

theorem symmetry_center_of_f (ω : ℝ) (h1 : ω > 0) (h2 : ∀ x, f ω (x + Real.pi / ω) = f ω x) :
  ∃ k : ℤ, f ω ((5 * Real.pi / 12) + x) = f ω ((5 * Real.pi / 12) - x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_center_of_f_l558_55836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_for_three_zeros_l558_55814

-- Define the function f
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.cos (ω * x) - 1

-- Define the property of having exactly 3 zeros in [0, 2π]
def has_exactly_three_zeros (ω : ℝ) : Prop :=
  ∃ (x₁ x₂ x₃ : ℝ), 0 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ ≤ 2 * Real.pi ∧
  f ω x₁ = 0 ∧ f ω x₂ = 0 ∧ f ω x₃ = 0 ∧
  ∀ x, 0 ≤ x ∧ x ≤ 2 * Real.pi ∧ f ω x = 0 → x = x₁ ∨ x = x₂ ∨ x = x₃

-- State the theorem
theorem omega_range_for_three_zeros :
  ∀ ω : ℝ, ω > 0 → has_exactly_three_zeros ω → 2 ≤ ω ∧ ω < 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_for_three_zeros_l558_55814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_special_triangle_l558_55874

/-- A triangle with consecutive natural number side lengths and one angle twice another -/
structure SpecialTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  consecutive : (a + 1 = b ∧ b + 1 = c) ∨ (a + 1 = c ∧ c + 1 = b) ∨ (b + 1 = a ∧ a + 1 = c)
  angle_condition : ∃ (x y : Real), x = 2 * y ∧ 
    (x = Real.arccos ((b^2 + c^2 - a^2) / (2 * b * c)) ∨
     x = Real.arccos ((a^2 + c^2 - b^2) / (2 * a * c)) ∨
     x = Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b)))

/-- The unique special triangle has side lengths (4, 5, 6) -/
theorem unique_special_triangle : 
  ∃! (t : SpecialTriangle), t.a = 4 ∧ t.b = 5 ∧ t.c = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_special_triangle_l558_55874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_arrangement_exists_l558_55801

def primes : List Nat := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41]

def inner_circle : List Nat := [41, 37, 11, 29]
def outer_circle : List Nat := [5, 13, 23, 7, 3, 19, 31, 17]

def triangle_sum (a b c : Nat) : Nat := a + b + c

theorem prime_arrangement_exists :
  ∃ (inner outer : List Nat),
    inner.length = 4 ∧
    outer.length = 8 ∧
    (inner ++ outer).toFinset = primes.toFinset ∧
    inner.sum = outer.sum ∧
    (∀ i j k, i ∈ inner → j ∈ outer → k ∈ outer →
      triangle_sum i j k = triangle_sum inner.head! outer.head! outer.tail.head!) :=
by
  -- Provide the explicit lists as evidence
  use inner_circle, outer_circle
  -- The proof is omitted for brevity
  sorry

#check prime_arrangement_exists

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_arrangement_exists_l558_55801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangle_perimeter_l558_55816

/-- Given an isosceles triangle with side lengths 12, 12, and 15, and a similar triangle
    with longest side 45, the perimeter of the larger triangle is 117. -/
theorem similar_triangle_perimeter 
  (small_equal : ℝ) 
  (small_base : ℝ) 
  (large_base : ℝ) 
  (perimeter : ℝ)
  (h1 : small_equal = 12)
  (h2 : small_base = 15)
  (h3 : large_base = 45)
  (h4 : perimeter = (small_equal * large_base / small_base) * 2 + large_base) :
  perimeter = 117 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangle_perimeter_l558_55816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_perfect_square_divisible_by_252_l558_55840

theorem smallest_perfect_square_divisible_by_252 : 
  ∃ n : ℕ, (∃ k : ℕ, n = k^2) ∧ 
           (252 ∣ n) ∧ 
           (∀ m : ℕ, m < n → (∃ k : ℕ, m = k^2) → ¬(252 ∣ m)) ∧
           n = 1764 :=
by
  -- Proof goes here
  sorry

#check smallest_perfect_square_divisible_by_252

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_perfect_square_divisible_by_252_l558_55840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_150_deg_angles_l558_55867

/-- The set of interior angles of a regular n-gon -/
def interior_angles (n : ℕ) : Set ℝ :=
  {θ | ∃ k, k ∈ Finset.range n ∧ θ = (n - 2) * 180 / n}

/-- A regular polygon with interior angles measuring 150° has 12 sides -/
theorem regular_polygon_150_deg_angles (n : ℕ) : 
  n > 2 → (∀ θ, θ ∈ interior_angles n → θ = 150) → n = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_150_deg_angles_l558_55867
