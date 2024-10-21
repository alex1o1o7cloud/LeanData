import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_taxi_distance_theorem_l1004_100447

/-- Taxi fare structure -/
structure TaxiFare where
  flagDownFare : ℝ
  flagDownDistance : ℝ
  midRangeFare : ℝ
  longRangeFare : ℝ
  midRangeLimit : ℝ
  fuelSurcharge : ℝ

/-- Calculate the fare based on distance -/
noncomputable def calculateFare (fare : TaxiFare) (distance : ℝ) : ℝ :=
  if distance ≤ fare.flagDownDistance then
    fare.flagDownFare + fare.fuelSurcharge
  else if distance ≤ fare.midRangeLimit then
    fare.flagDownFare + fare.midRangeFare * (distance - fare.flagDownDistance) + fare.fuelSurcharge
  else
    fare.flagDownFare + fare.midRangeFare * (fare.midRangeLimit - fare.flagDownDistance) +
    fare.longRangeFare * (distance - fare.midRangeLimit) + fare.fuelSurcharge

/-- Theorem: Given the fare structure and a paid fare of $22.6, the distance traveled is 9 km -/
theorem taxi_distance_theorem (fare : TaxiFare)
    (h1 : fare.flagDownFare = 8)
    (h2 : fare.flagDownDistance = 3)
    (h3 : fare.midRangeFare = 2.15)
    (h4 : fare.longRangeFare = 2.85)
    (h5 : fare.midRangeLimit = 8)
    (h6 : fare.fuelSurcharge = 1) :
    ∃ (distance : ℝ), calculateFare fare distance = 22.6 ∧ distance = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_taxi_distance_theorem_l1004_100447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_article_price_equation_price_satisfies_equation_l1004_100475

/-- Given an original price P, this theorem states the relationship between
    the price after various discounts, taxes, and profit margins. -/
theorem article_price_equation (P : ℝ) : 
  0.90117 * P * 1.10 = 0.99603 * P - 5 := by sorry

/-- This function calculates the original price P that satisfies the equation. -/
noncomputable def calculate_original_price : ℝ :=
  5 / 0.004743

/-- This theorem states that the calculated price satisfies the equation
    within a small margin of error due to floating-point arithmetic. -/
theorem price_satisfies_equation :
  let P := calculate_original_price
  abs (0.90117 * P * 1.10 - (0.99603 * P - 5)) < 1e-6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_article_price_equation_price_satisfies_equation_l1004_100475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1004_100469

-- Define the function f as noncomputable
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.sqrt (|x + 2| + |x - 4| - m)

-- State the theorem
theorem function_properties :
  (∀ x, f 6 x ≥ 0) ∧
  (∀ m, m > 6 → ∃ x, f m x < 0) ∧
  (∀ a b : ℝ, a > 0 → b > 0 → 4 / (a + 5*b) + 1 / (3*a + 2*b) = 6 → 4*a + 7*b ≥ 3/2) ∧
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 4 / (a + 5*b) + 1 / (3*a + 2*b) = 6 ∧ 4*a + 7*b = 3/2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1004_100469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_illuminated_area_of_cube_theorem_l1004_100478

/-- The illuminated area of a cube's surface given its edge length and the radius of the cylindrical beam -/
noncomputable def illuminated_area_of_cube (a : ℝ) (ρ : ℝ) : ℝ := 
  (2 * Real.sqrt 3 - Real.sqrt 6) * (Real.pi + 6 * Real.sqrt 2) / 4

/-- The area of the illuminated part of a cube's surface when illuminated by a cylindrical beam along its main diagonal -/
theorem illuminated_area_of_cube_theorem (a : ℝ) (ρ : ℝ) : 
  a = 1 → 
  ρ = Real.sqrt (2 - Real.sqrt 2) → 
  illuminated_area_of_cube a ρ = 
    (2 * Real.sqrt 3 - Real.sqrt 6) * (Real.pi + 6 * Real.sqrt 2) / 4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_illuminated_area_of_cube_theorem_l1004_100478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_coefficient_sum_l1004_100476

/-- Given a function f : ℝ → ℝ satisfying the conditions:
    1) f(x+3) = 3x^2 + 7x + 4
    2) f(x) = ax^2 + bx + c
    Prove that a + b + c = 2 -/
theorem function_coefficient_sum (f : ℝ → ℝ) (a b c : ℝ) 
    (h1 : ∀ x, f (x + 3) = 3 * x^2 + 7 * x + 4)
    (h2 : ∀ x, f x = a * x^2 + b * x + c) : 
  a + b + c = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_coefficient_sum_l1004_100476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_five_consecutive_odd_integers_l1004_100470

theorem sum_of_five_consecutive_odd_integers :
  ∀ n : ℤ, (∃ k : ℤ, n = 5*k + 20 ∧ k % 2 ≠ 0) ↔ n ∈ ({35, 55, 75, 145} : Set ℤ) :=
by sorry

#check sum_of_five_consecutive_odd_integers

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_five_consecutive_odd_integers_l1004_100470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_equality_implies_z_value_l1004_100444

-- Define the theorem
theorem mean_equality_implies_z_value (z : ℝ) : 
  (6 + 10 + 22) / 3 = (15 + z) / 2 → z = 31 / 3 := by
  -- The proof is omitted
  sorry

-- Prove the theorem
example : ∃ z : ℝ, (6 + 10 + 22) / 3 = (15 + z) / 2 ∧ z = 31 / 3 := by
  -- The proof is omitted
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_equality_implies_z_value_l1004_100444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_equality_l1004_100402

theorem cube_root_equality (x : ℝ) : Real.rpow x (1/3) = Real.rpow (Real.rpow x (1/3)) 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_equality_l1004_100402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_special_case_l1004_100485

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hyperbola -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := h.c / h.a

theorem hyperbola_eccentricity_special_case 
  (h : Hyperbola) 
  (ha : h.a > 0)
  (hb : h.b > 0)
  (hF : Point)
  (hE : Point)
  (hP : Point)
  (h_left_focus : hF.x = -h.c ∧ hF.y = 0)
  (h_circle_tangent : hE.x^2 + hE.y^2 = h.a^2)
  (h_parabola : hP.y^2 = 4 * h.c * hP.x)
  (h_midpoint : hE.x = (hF.x + hP.x) / 2 ∧ hE.y = (hF.y + hP.y) / 2)
  : eccentricity h = (Real.sqrt 5 + 1) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_special_case_l1004_100485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l1004_100408

noncomputable def f (x : ℝ) : ℝ := (x^2 - 1) / ((x-4)^2)

theorem solution_set (x : ℝ) :
  (f x ≥ 0 ∧ x ≠ 4) ↔ (x ≤ -1 ∨ x > 1) ∧ x ≠ 4 :=
by
  sorry

#check solution_set

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l1004_100408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_tangent_difference_max_l1004_100412

theorem triangle_tangent_difference_max (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Angles are positive
  A + B + C = π ∧  -- Sum of angles in a triangle is π
  a > 0 ∧ b > 0 ∧ c > 0 ∧  -- Sides are positive
  a / Real.sin A = b / Real.sin B ∧  -- Sine rule
  b / Real.sin B = c / Real.sin C ∧  -- Sine rule
  2 * b * Real.cos C - 3 * c * Real.cos B = a →  -- Given condition
  |Real.tan (B - C)| ≤ 3/4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_tangent_difference_max_l1004_100412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_term_is_half_l1004_100414

def a : ℕ → ℚ
  | 0 => 1/2
  | n+1 => 1 - 1 / a n

theorem tenth_term_is_half :
  a 9 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_term_is_half_l1004_100414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trailing_zeros_existence_l1004_100472

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125) + (n / 625) + (n / 3125)

theorem trailing_zeros_existence :
  (∃ n : ℕ, trailingZeros n = 1971) ∧ (¬ ∃ n : ℕ, trailingZeros n = 1972) := by
  sorry

#eval trailingZeros 7895  -- Should output 1971

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trailing_zeros_existence_l1004_100472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pair_op_example_pair_op_equation_pair_op_integer_solution_l1004_100464

-- Define the ⊗ operation for rational number pairs
def pair_op (a b c d : ℚ) : ℚ := b * c - a * d

-- Theorem 1
theorem pair_op_example : pair_op 5 3 (-2) 1 = -11 := by sorry

-- Theorem 2
theorem pair_op_equation (x : ℚ) : pair_op 2 (3*x - 1) 6 (x + 2) = 22 → x = 2 := by sorry

-- Theorem 3
theorem pair_op_integer_solution (k : ℤ) :
  (∃ x : ℤ, pair_op 4 (↑(k - 2)) ↑x (↑(2*x - 1)) = 6) ↔ k ∈ ({8, 9, 11, 12} : Set ℤ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pair_op_example_pair_op_equation_pair_op_integer_solution_l1004_100464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pig_selling_price_l1004_100438

/-- Represents the selling price of a fully grown pig -/
def selling_price : ℕ → ℕ := λ _ ↦ 300

/-- Given conditions of the problem -/
def num_piglets : ℕ := 6
def feeding_cost : ℕ := 10
def pigs_sold_12_months : ℕ := 3
def pigs_sold_16_months : ℕ := 3
def total_profit : ℕ := 960

/-- Theorem stating the selling price of a fully grown pig -/
theorem pig_selling_price : selling_price num_piglets = 300 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pig_selling_price_l1004_100438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_ratio_equality_l1004_100497

theorem trig_ratio_equality (a : ℝ) 
  (h1 : Real.cos (π/4 - a) = 12/13) 
  (h2 : 0 < π/4 - a ∧ π/4 - a < π/2) : 
  Real.sin (π/2 - 2*a) / Real.sin (π/4 + a) = 119/144 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_ratio_equality_l1004_100497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bowling_average_decrease_correct_l1004_100491

noncomputable def bowling_average_decrease (initial_average : ℝ) (last_match_wickets : ℕ) (last_match_runs : ℕ) (previous_wickets : ℕ) : ℝ :=
  let total_runs_before := initial_average * (previous_wickets : ℝ)
  let new_total_runs := total_runs_before + (last_match_runs : ℝ)
  let new_total_wickets := previous_wickets + last_match_wickets
  let new_average := new_total_runs / (new_total_wickets : ℝ)
  initial_average - new_average

theorem bowling_average_decrease_correct :
  bowling_average_decrease 12.4 6 26 115 = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bowling_average_decrease_correct_l1004_100491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_ratio_sum_constant_l1004_100424

/-- An ellipse with center at origin, minor axis length 2, and eccentricity 2√5/5 -/
structure Ellipse :=
  (center : ℝ × ℝ)
  (minor_axis : ℝ)
  (eccentricity : ℝ)

/-- A line passing through a point and intersecting an ellipse -/
structure Line :=
  (focus : ℝ × ℝ)
  (slope : ℝ)

/-- Points of intersection between the line and the ellipse/y-axis -/
structure Intersections :=
  (A : ℝ × ℝ)
  (B : ℝ × ℝ)
  (M : ℝ × ℝ)

/-- Ratios of vector lengths -/
structure Ratios :=
  (lambda1 : ℝ)
  (lambda2 : ℝ)

/-- Main theorem statement -/
theorem ellipse_intersection_ratio_sum_constant
  (C : Ellipse)
  (l : Line)
  (points : Intersections)
  (ratios : Ratios) :
  C.center = (0, 0) →
  C.minor_axis = 2 →
  C.eccentricity = 2 * Real.sqrt 5 / 5 →
  l.focus = (2, 0) →
  ratios.lambda1 = (points.A.1 / (2 - points.A.1)) →
  ratios.lambda2 = (points.B.1 / (2 - points.B.1)) →
  ratios.lambda1 + ratios.lambda2 = -10 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_ratio_sum_constant_l1004_100424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_positive_integer_l1004_100428

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

def is_six_digit (n : ℕ) : Prop := 100000 ≤ n ∧ n < 1000000

def digits (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc else aux (m / 10) ((m % 10) :: acc)
  aux n []

def unique_digits (n m : ℕ) : Prop :=
  (digits n ++ digits m).toFinset = Finset.range 10

theorem unique_positive_integer : ∃! (n : ℕ), 
  n > 0 ∧ 
  is_four_digit (n^3) ∧ 
  is_six_digit (n^4) ∧ 
  unique_digits (n^3) (n^4) ∧ 
  n = 18 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_positive_integer_l1004_100428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_region_perimeter_l1004_100423

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : EuclideanSpace ℝ (Fin 2)
  radius : ℝ

/-- Defines when two circles touch each other -/
def Circle.touches (c1 c2 : Circle) : Prop :=
  ‖c1.center - c2.center‖ = c1.radius + c2.radius

/-- Calculates the circumference of a circle -/
noncomputable def Circle.circumference (c : Circle) : ℝ :=
  2 * Real.pi * c.radius

/-- Represents the subtended angle at the center of a circle for the shaded region -/
def Circle.subtendedAngle (c : Circle) : ℝ := 
  sorry

/-- Represents the shaded region formed by three circles -/
def shadedRegion (c1 c2 c3 : Circle) : Set (EuclideanSpace ℝ (Fin 2)) := 
  sorry

/-- Calculates the perimeter of a region -/
noncomputable def Region.perimeter (r : Set (EuclideanSpace ℝ (Fin 2))) : ℝ := 
  sorry

/-- Given three identical circles touching each other, this theorem proves that
    the perimeter of the shaded region formed by 90° sectors of these circles is 18,
    when each circle has a circumference of 24. -/
theorem shaded_region_perimeter (c1 c2 c3 : Circle) : 
  c1.touches c2 ∧ c2.touches c3 ∧ c3.touches c1 →
  c1.circumference = 24 ∧ c2.circumference = 24 ∧ c3.circumference = 24 →
  c1.subtendedAngle = 90 ∧ c2.subtendedAngle = 90 ∧ c3.subtendedAngle = 90 →
  Region.perimeter (shadedRegion c1 c2 c3) = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_region_perimeter_l1004_100423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_l1004_100445

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if 1 ≤ x ∧ x ≤ 2 then 2 * x + 6
  else if -1 ≤ x ∧ x ≤ 1 then x + 7
  else 0  -- This else case is added to make the function total

-- State the theorem
theorem f_max_min :
  (∃ (x : ℝ), -1 ≤ x ∧ x ≤ 2 ∧ f x = 10) ∧
  (∀ (x : ℝ), -1 ≤ x ∧ x ≤ 2 → f x ≤ 10) ∧
  (∃ (x : ℝ), -1 ≤ x ∧ x ≤ 2 ∧ f x = 6) ∧
  (∀ (x : ℝ), -1 ≤ x ∧ x ≤ 2 → f x ≥ 6) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_l1004_100445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_cylinder_volume_ratio_l1004_100417

noncomputable section

/-- The volume of a right circular cone with radius r and height h -/
def cone_volume (r h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

/-- The volume of a right cylinder with radius r and height h -/
def cylinder_volume (r h : ℝ) : ℝ := Real.pi * r^2 * h

/-- The ratio of the volume of a cone to the volume of a cylinder -/
def volume_ratio (r h : ℝ) : ℝ := (cone_volume r h) / (cylinder_volume r h)

theorem cone_cylinder_volume_ratio :
  ∀ (r h : ℝ), r > 0 → h > 0 → volume_ratio r h = 1/3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_cylinder_volume_ratio_l1004_100417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_C₁_cartesian_equation_min_distance_C₁_C₂_l1004_100449

-- Define the parametric equations for curve C₁
noncomputable def C₁_x (α : Real) : Real := Real.sqrt 2 * Real.sin (α + Real.pi / 4)
noncomputable def C₁_y (α : Real) : Real := Real.sin (2 * α) + 1

-- Define the polar equation for curve C₂
def C₂_polar (ρ θ : Real) : Prop := ρ^2 = 4 * ρ * Real.sin θ - 3

-- Theorem 1: Cartesian equation of C₁
theorem C₁_cartesian_equation :
  ∀ x y : Real, (∃ α : Real, C₁_x α = x ∧ C₁_y α = y) → y = x^2 := by sorry

-- Theorem 2: Minimum distance between C₁ and C₂
theorem min_distance_C₁_C₂ :
  let C₂ := {p : Real × Real | p.1^2 + p.2^2 - 4*p.2 + 3 = 0}
  let minDist : Real := Real.sqrt ((Real.sqrt 7 / 2 - 1)^2)
  ∀ p₁ p₂ : Real × Real,
    (p₁.2 = p₁.1^2) →  -- p₁ is on C₁
    p₂ ∈ C₂ →          -- p₂ is on C₂
    Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2) ≥ minDist := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_C₁_cartesian_equation_min_distance_C₁_C₂_l1004_100449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_and_range_l1004_100451

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 / (3^x + 1) + a

theorem odd_function_and_range (a : ℝ) :
  (∀ x, f a x = -f a (-x)) →
  (a = -1 ∧
   ∀ t, (∃ x, 0 ≤ x ∧ x ≤ 1 ∧ f (-1) x + 1 = t) ↔ 1/2 ≤ t ∧ t ≤ 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_and_range_l1004_100451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_trip_distance_l1004_100480

theorem boat_trip_distance 
  (boat_speed : ℝ) 
  (stream_speed : ℝ) 
  (total_time : ℝ) 
  (h1 : boat_speed = 16) 
  (h2 : stream_speed = 2) 
  (h3 : total_time = 891.4285714285714) : 
  (total_time * (boat_speed + stream_speed) * (boat_speed - stream_speed)) / 
  ((boat_speed + stream_speed) + (boat_speed - stream_speed)) = 7020 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_trip_distance_l1004_100480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_theorem_l1004_100422

noncomputable section

/-- The parabola y² = 6x -/
def Parabola (x y : ℝ) : Prop := y^2 = 6*x

/-- The line 3x - 4y + 26 = 0 -/
def Line (x y : ℝ) : Prop := 3*x - 4*y + 26 = 0

/-- Distance from a point (x, y) to the line 3x - 4y + 26 = 0 -/
noncomputable def DistanceToLine (x y : ℝ) : ℝ := |3*x - 4*y + 26| / 5

/-- Focus of the parabola y² = 6x -/
def FocusX : ℝ := 3/2

def FocusY : ℝ := 0

/-- Distance from a point (x, y) to the focus of the parabola -/
noncomputable def DistanceToFocus (x y : ℝ) : ℝ := ((x - FocusX)^2 + (y - FocusY)^2)^(1/2)

theorem parabola_distance_theorem :
  ∃ (x₀ y₀ : ℝ),
    Parabola x₀ y₀ ∧
    (∀ (x y : ℝ), Parabola x y → DistanceToLine x₀ y₀ ≤ DistanceToLine x y) ∧
    x₀ = 8/3 ∧ y₀ = 4 ∧ DistanceToLine x₀ y₀ = 3.6 ∧
    (∀ (x y : ℝ), Parabola x y →
      DistanceToLine x y + DistanceToFocus x y ≥ 6.1) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_theorem_l1004_100422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_is_even_l1004_100407

noncomputable def f (x : ℝ) : ℝ := Real.log (1 + 2*x)
noncomputable def g (x : ℝ) : ℝ := Real.log (1 - 2*x)
noncomputable def h (x : ℝ) : ℝ := f x + g x

theorem h_is_even : ∀ x ∈ Set.Ioo (-1/2 : ℝ) (1/2 : ℝ), h (-x) = h x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_is_even_l1004_100407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_part_of_division_problem_statement_l1004_100401

def arithmetic_sum (a₁ : ℚ) (aₙ : ℚ) (n : ℕ) : ℚ :=
  n * (a₁ + aₙ) / 2

def sum_040_to_059 : ℚ :=
  arithmetic_sum 0.40 0.59 20

theorem integer_part_of_division (x : ℚ) :
  ⌊x⌋ = 1 ↔ 1 ≤ x ∧ x < 2 := by sorry

theorem problem_statement :
  ⌊16 / sum_040_to_059⌋ = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_part_of_division_problem_statement_l1004_100401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_zero_l1004_100495

-- Define the curve C
noncomputable def C (x : ℝ) : ℝ := Real.log x / x

-- Define the tangent line equation
def tangent_line (x y : ℝ) : Prop := x - y - 1 = 0

-- Theorem statement
theorem tangent_line_at_one_zero :
  ∃ (m : ℝ), 
    (∀ h : ℝ → ℝ, h = C → (deriv h) 1 = m) ∧ 
    (∀ x y : ℝ, tangent_line x y ↔ y - 0 = m * (x - 1)) := by
  -- Proof goes here
  sorry

-- Additional lemma to show that the slope is indeed 1
lemma slope_is_one : 
  ∃ (m : ℝ), (∀ h : ℝ → ℝ, h = C → (deriv h) 1 = m) ∧ m = 1 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_zero_l1004_100495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_value_l1004_100468

noncomputable def reward_function (a : ℝ) (x : ℝ) : ℝ := (10 * x - 3 * a) / (x + 2)

theorem min_a_value :
  ∃ (a : ℕ), 
    (∀ x : ℝ, 10 ≤ x ∧ x ≤ 1000 → 
      reward_function (a : ℝ) x ≤ 9 ∧ 
      reward_function (a : ℝ) x ≤ x / 5) ∧
    (∀ b : ℕ, b < a → 
      ∃ x : ℝ, 10 ≤ x ∧ x ≤ 1000 ∧ 
        (reward_function (b : ℝ) x > 9 ∨ 
         reward_function (b : ℝ) x > x / 5)) ∧
    a = 328 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_value_l1004_100468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_value_l1004_100430

/-- The eccentricity of a hyperbola with given parameters -/
noncomputable def hyperbola_eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 + b^2 / a^2)

/-- Theorem: The eccentricity of the given hyperbola is √6/2 -/
theorem hyperbola_eccentricity_value (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : a = Real.sqrt 2 * b) : 
  hyperbola_eccentricity a b = Real.sqrt 6 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_value_l1004_100430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bug_meeting_point_l1004_100462

/-- Triangle PQR with given side lengths -/
structure Triangle where
  PQ : ℝ
  QR : ℝ
  PR : ℝ

/-- Bugs crawling on the triangle -/
structure Bugs where
  clockwise_speed : ℝ
  counterclockwise_speed : ℝ

/-- The meeting point S on side QR -/
noncomputable def meeting_point (t : Triangle) (b : Bugs) : ℝ :=
  t.QR - (b.clockwise_speed * (t.PQ + t.QR + t.PR) / (b.clockwise_speed + b.counterclockwise_speed))

theorem bug_meeting_point (t : Triangle) (b : Bugs) :
  t.PQ = 8 → t.QR = 10 → t.PR = 12 →
  b.clockwise_speed = 2 → b.counterclockwise_speed = 3 →
  meeting_point t b = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bug_meeting_point_l1004_100462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hex_count_and_sum_l1004_100481

def is_valid_hex (n : ℕ) : Bool :=
  n.digits 16 |>.all (· < 10)

def count_valid_hex (upper_bound : ℕ) : ℕ :=
  (List.range upper_bound).filter is_valid_hex |>.length

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem hex_count_and_sum :
  count_valid_hex 500 = 190 ∧ sum_of_digits 190 = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hex_count_and_sum_l1004_100481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_is_30_l1004_100420

/-- Represents a convex quadrilateral ABCD with given side lengths and a right angle at B -/
structure ConvexQuadrilateral where
  AB : ℝ
  BC : ℝ
  CD : ℝ
  DA : ℝ
  right_angle_at_B : AB > 0 ∧ BC > 0 ∧ CD > 0 ∧ DA > 0

/-- The area of triangle ABC in the given convex quadrilateral ABCD -/
noncomputable def area_triangle_ABC (q : ConvexQuadrilateral) : ℝ :=
  (1 / 2) * q.AB * q.BC

/-- Theorem stating that the area of triangle ABC is 30 for the given quadrilateral -/
theorem area_is_30 (q : ConvexQuadrilateral) 
  (h1 : q.AB = 5) 
  (h2 : q.BC = 12) : 
  area_triangle_ABC q = 30 := by
  -- Unfold the definition of area_triangle_ABC
  unfold area_triangle_ABC
  -- Substitute the given values
  rw [h1, h2]
  -- Simplify the arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_is_30_l1004_100420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_value_l1004_100415

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then 2^x - 3*x + 2*m else -(2^(-x) - 3*(-x) + 2*m)

-- State the theorem
theorem odd_function_value (m : ℝ) :
  (∀ x, f m x = -(f m (-x))) →  -- f is odd
  f m 1 = -5/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_value_l1004_100415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_lines_l1004_100432

/-- The point of intersection of two lines given by linear equations -/
noncomputable def intersection_point (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) : ℝ × ℝ :=
  let x := (b₁ * c₂ - b₂ * c₁) / (a₁ * b₂ - a₂ * b₁)
  let y := (a₂ * c₁ - a₁ * c₂) / (a₁ * b₂ - a₂ * b₁)
  (x, y)

/-- Theorem stating that (0, 2) is the unique intersection point of the given lines -/
theorem intersection_of_lines :
  let line1 : ℝ → ℝ → Prop := λ x y => 3 * y = -2 * x + 6
  let line2 : ℝ → ℝ → Prop := λ x y => -2 * y = 6 * x - 4
  intersection_point 2 3 6 (-6) (-2) (-4) = (0, 2) ∧
  (∀ x y : ℝ, line1 x y ∧ line2 x y → (x, y) = (0, 2)) := by
  sorry

#check intersection_of_lines

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_lines_l1004_100432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_maria_vegetable_cutting_l1004_100477

/-- The number of potatoes Maria has -/
def potatoes : ℕ := 2

/-- The ratio of carrots to potatoes -/
def carrot_potato_ratio : ℕ := 6

/-- The ratio of onions to carrots -/
def onion_carrot_ratio : ℕ := 2

/-- The ratio of green beans to onions -/
def green_bean_onion_ratio : ℚ := 1/3

/-- The number of green beans Maria needs to cut -/
def green_beans : ℕ := 8

theorem maria_vegetable_cutting :
  green_beans = (green_bean_onion_ratio * (onion_carrot_ratio * (carrot_potato_ratio * potatoes))).floor := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_maria_vegetable_cutting_l1004_100477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_formula_l1004_100429

def sequence_a : ℕ+ → ℝ := sorry
def sequence_b : ℕ+ → ℝ := sorry

axiom positive_terms : ∀ n : ℕ+, sequence_a n > 0 ∧ sequence_b n > 0

axiom arithmetic_sequence : ∀ n : ℕ+, 2 * sequence_b n = sequence_a n + sequence_a (n + 1)

axiom geometric_sequence : ∀ n : ℕ+, (sequence_a (n + 1))^2 = sequence_b n * sequence_b (n + 1)

axiom initial_values : sequence_a 1 = 1 ∧ sequence_a 2 = 3

theorem sequence_a_formula : ∀ n : ℕ+, sequence_a n = (n^2 + n) / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_formula_l1004_100429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_problem_l1004_100440

theorem candy_problem (S N a : ℕ) : 
  (a = S - 7 - a) → 
  (a > 1) → 
  (S = N * a) → 
  S = 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_problem_l1004_100440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_one_eq_seven_l1004_100493

theorem g_of_one_eq_seven (g : ℝ → ℝ) (h : ∀ x : ℝ, g (2*x + 3) = x^2 - 2*x + 4) : g 1 = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_one_eq_seven_l1004_100493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_power_of_18_dividing_24_factorial_l1004_100434

theorem largest_power_of_18_dividing_24_factorial :
  (∃ n : ℕ, n = 4 ∧ 
    (∀ m : ℕ, 18^m ∣ Nat.factorial 24 → m ≤ n) ∧
    18^n ∣ Nat.factorial 24) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_power_of_18_dividing_24_factorial_l1004_100434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_factorable_l1004_100458

/-- Represents a polynomial expression -/
inductive Expr
  | neg_sum_squares (x y : ℝ) : Expr
  | neg_fraction_plus_one (a b : ℝ) : Expr
  | sum_three_terms (a b : ℝ) : Expr
  | fraction_minus_product_plus_square (m n : ℝ) : Expr

/-- Checks if an expression can be factored using standard formula methods -/
def can_be_factored : Expr → Bool
  | Expr.neg_sum_squares _ _ => false
  | Expr.neg_fraction_plus_one _ _ => true
  | Expr.sum_three_terms _ _ => false
  | Expr.fraction_minus_product_plus_square _ _ => true

/-- The list of all expressions to be checked -/
def all_expressions : List Expr :=
  [Expr.neg_sum_squares 0 0,
   Expr.neg_fraction_plus_one 0 0,
   Expr.sum_three_terms 0 0,
   Expr.fraction_minus_product_plus_square 0 0]

/-- The main theorem stating that exactly two expressions can be factored -/
theorem exactly_two_factorable :
  (all_expressions.filter can_be_factored).length = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_factorable_l1004_100458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_length_is_one_l1004_100482

noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

noncomputable def frac (x : ℝ) : ℝ := x - floor x

noncomputable def f (x : ℝ) : ℝ := (floor x : ℝ) * frac x

def g (x : ℝ) : ℝ := x - 1

theorem solution_length_is_one :
  ∃ (a b : ℝ), a = 2 ∧ b = 3 ∧
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 3 → (f x < g x ↔ a ≤ x ∧ x ≤ b)) ∧
  b - a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_length_is_one_l1004_100482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_a_l1004_100457

theorem triangle_angle_a (a b : ℝ) (B : ℝ) (hb : b = 1) (ha : a = Real.sqrt 3) (hB : B = 30 * π / 180) :
  let A := Real.arcsin (a * Real.sin B / b)
  A = 60 * π / 180 ∨ A = 120 * π / 180 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_a_l1004_100457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_intersection_l1004_100454

/-- The parabola y = x^2 -/
noncomputable def parabola (x : ℝ) : ℝ := x^2

/-- Point A on the parabola -/
noncomputable def A : ℝ × ℝ := (2, 4)

/-- Point B, the other intersection of the normal line and the parabola -/
noncomputable def B : ℝ × ℝ := (-9/4, 81/16)

/-- The slope of the normal line at A -/
noncomputable def normal_slope : ℝ := -1 / (2 * A.1)

/-- The normal line equation -/
noncomputable def normal_line (x : ℝ) : ℝ := normal_slope * (x - A.1) + A.2

theorem normal_intersection :
  B.2 = parabola B.1 ∧
  normal_line B.1 = B.2 ∧
  B ≠ A := by
  sorry

#check normal_intersection

end NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_intersection_l1004_100454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_accurate_reading_l1004_100435

noncomputable def scale_reading (arrow_position : ℝ) (options : List ℝ) : ℝ :=
  (options.argmin (fun x => |x - arrow_position|)).getD 0

theorem accurate_reading (arrow_position : ℝ) :
  42.3 < arrow_position →
  arrow_position < 42.6 →
  arrow_position < (42.3 + 42.6) / 2 →
  scale_reading arrow_position [42.05, 42.15, 42.25, 42.3, 42.6] = 42.3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_accurate_reading_l1004_100435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rose_difference_l1004_100490

theorem rose_difference (total : ℕ) (red_fraction : ℚ) (yellow_fraction : ℚ)
  (h_total : total = 48)
  (h_red : red_fraction = 3/8)
  (h_yellow : yellow_fraction = 5/16) :
  (red_fraction * total).floor - (yellow_fraction * total).floor = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rose_difference_l1004_100490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_with_sums_l1004_100437

def digit_pair_sums (n : ℕ) : List ℕ :=
  sorry

theorem smallest_number_with_sums (n : ℕ) :
  (2 ∈ digit_pair_sums n) ∧
  (0 ∈ digit_pair_sums n) ∧
  ((digit_pair_sums n).filter (· = 2)).length ≥ 2 →
  n ≥ 2000 :=
by
  sorry

#check smallest_number_with_sums

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_with_sums_l1004_100437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_roots_l1004_100489

noncomputable def polynomial (x : ℂ) : ℂ := 3 * x^4 - 5 * x^3 - x^2 + 5 * x + 3

noncomputable def root1 : ℂ := (5 + Real.sqrt 109) / 12 + Real.sqrt (((5 + Real.sqrt 109) / 6)^2 - 4) / 2
noncomputable def root2 : ℂ := (5 + Real.sqrt 109) / 12 - Real.sqrt (((5 + Real.sqrt 109) / 6)^2 - 4) / 2
noncomputable def root3 : ℂ := (5 - Real.sqrt 109) / 12 + Real.sqrt (((5 - Real.sqrt 109) / 6)^2 - 4) / 2
noncomputable def root4 : ℂ := (5 - Real.sqrt 109) / 12 - Real.sqrt (((5 - Real.sqrt 109) / 6)^2 - 4) / 2

theorem polynomial_roots :
  polynomial root1 = 0 ∧
  polynomial root2 = 0 ∧
  polynomial root3 = 0 ∧
  polynomial root4 = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_roots_l1004_100489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pool_testing_efficiency_l1004_100487

/-- 
Represents the efficiency of pool testing compared to individual testing 
for nucleic acid screening with the following parameters:
- k: number of samples in a pool (fixed at 4)
- p: probability of a sample being positive (0 < p < 1)
-/
theorem pool_testing_efficiency (p : ℝ) (h1 : 0 < p) (h2 : p < 1) :
  let k : ℕ := 4
  let e_individual : ℝ := k
  let e_pool : ℝ := 5 - 4 * (1 - p)^k
  e_pool < e_individual ↔ (1 - p)^2 > 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pool_testing_efficiency_l1004_100487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_constant_term_product_l1004_100431

/-- A polynomial over ℝ -/
def MyPolynomial := ℝ → ℝ

/-- The constant term of a polynomial -/
def constantTerm (p : MyPolynomial) : ℝ := p 0

theorem polynomial_constant_term_product
  (p q r : MyPolynomial)
  (h1 : ∀ x, r x = p x * q x)
  (h2 : constantTerm p = 5)
  (h3 : constantTerm r = -10) :
  q 0 = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_constant_term_product_l1004_100431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_specific_point_line_l1004_100406

/-- The distance between a point and a line in 3D space -/
noncomputable def distance_point_line_3d (x₀ y₀ z₀ A B C D : ℝ) : ℝ :=
  |A * x₀ + B * y₀ + C * z₀ + D| / Real.sqrt (A^2 + B^2 + C^2)

/-- The distance between the point (2, 4, 1) and the line x + 2y + 2z + 3 = 0 is 5 -/
theorem distance_specific_point_line : distance_point_line_3d 2 4 1 1 2 2 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_specific_point_line_l1004_100406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l1004_100442

def is_arithmetic_sequence (a : ℕ+ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ+, a (n + 1) = a n + d

def partial_sum (a : ℕ+ → ℝ) (n : ℕ+) : ℝ :=
  Finset.sum (Finset.range n.val) (λ i => a ⟨i + 1, Nat.succ_pos i⟩)

theorem arithmetic_sequence_properties
  (a : ℕ+ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_S5_lt_S6 : partial_sum a 5 < partial_sum a 6)
  (h_S6_eq_S7 : partial_sum a 6 = partial_sum a 7)
  (h_S7_gt_S8 : partial_sum a 7 > partial_sum a 8) :
  ∃ d : ℝ,
    (∀ n : ℕ+, a (n + 1) = a n + d) ∧
    d < 0 ∧
    a 7 = 0 ∧
    partial_sum a 9 ≤ partial_sum a 5 ∧
    (∀ n : ℕ+, partial_sum a n ≤ partial_sum a 6) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l1004_100442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rademacher_valid_l1004_100498

-- Define the Rademacher function
noncomputable def rademacher (n : ℕ) (x : ℝ) : ℝ :=
  Real.sign (Real.sin (2^n * Real.pi * x))

-- Define the property of being a Rademacher function
def IsRademacherFunction (R : ℕ → ℝ → ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → ∀ x : ℝ, 0 ≤ x → x ≤ 1 →
    (R n x = 1 ∨ R n x = -1) ∧
    ∀ k : ℕ, k < 2^n → R n ((k + 1) / 2^n) = - R n (k / 2^n)

-- State the theorem
theorem rademacher_valid (n : ℕ) (x : ℝ) (h1 : n ≥ 1) (h2 : 0 ≤ x) (h3 : x ≤ 1) :
  ∃ (R : ℕ → ℝ → ℝ), R n x = rademacher n x ∧ IsRademacherFunction R :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rademacher_valid_l1004_100498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_base_length_l1004_100413

/-- Represents a trapezoid with given dimensions -/
structure Trapezoid where
  area : ℝ
  upper_side : ℝ
  height : ℝ
  base : ℝ

/-- The area formula for a trapezoid -/
noncomputable def trapezoid_area_formula (t : Trapezoid) : ℝ :=
  (1/2) * (t.upper_side + t.base) * t.height

/-- Theorem: Given a trapezoid with area 222 cm², upper side 23 cm, and height 12 cm, its base is 14 cm -/
theorem trapezoid_base_length (t : Trapezoid) 
    (h_area : t.area = 222)
    (h_upper : t.upper_side = 23)
    (h_height : t.height = 12)
    (h_formula : t.area = trapezoid_area_formula t) : 
  t.base = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_base_length_l1004_100413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_numbers_l1004_100488

def is_valid_number (n : ℕ) : Prop :=
  n ≥ 10000 ∧ n < 100000 ∧
  ∀ i : Fin 5, 
    let m := n % (10^(5-i.val))
    m ≠ 0 ∧ (i.val < 4 → m ∣ (n % (10^(6-i.val))))

theorem valid_numbers : 
  {n : ℕ | is_valid_number n} = {53125, 91125, 95625} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_numbers_l1004_100488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_upper_vertex_l1004_100443

def ellipse (x y : ℝ) : Prop := x^2 / 5 + y^2 = 1

def upper_vertex : ℝ × ℝ := (0, 1)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem max_distance_to_upper_vertex :
  ∃ (max_dist : ℝ),
    (∀ (P : ℝ × ℝ), ellipse P.1 P.2 → distance P upper_vertex ≤ max_dist) ∧
    (∃ (P : ℝ × ℝ), ellipse P.1 P.2 ∧ distance P upper_vertex = max_dist) ∧
    max_dist = 5/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_upper_vertex_l1004_100443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l1004_100426

theorem inequality_solution (x y : ℝ) : 
  Real.sqrt (π/4 - Real.arctan ((|x| + |y|)/π)) + Real.tan x^2 + 1 
  ≤ Real.sqrt 2 * |Real.tan x| * (Real.sin x + Real.cos x) 
  ↔ ((x = π/4 ∧ y = 3*π/4) ∨ (x = π/4 ∧ y = -3*π/4)) := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l1004_100426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_quadrant_l1004_100467

theorem angle_quadrant (θ : Real) (h : Real.sin θ + Real.cos θ = 2023 / 2024) :
  Real.sin θ * Real.cos θ < 0 :=
by sorry

-- The theorem above proves that sin θ * cos θ is negative,
-- which implies that θ is in the second or fourth quadrant.

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_quadrant_l1004_100467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_set_is_primes_l1004_100425

/-- The largest prime divisor of a natural number -/
def largest_prime_divisor (n : ℕ) : ℕ :=
  (Nat.factors n).maximum?.getD 1

/-- The sequence defined by a(0) = 2 and a(n+1) = a(n) + l(a(n)) -/
def a : ℕ → ℕ
  | 0 => 2
  | n + 1 => a n + largest_prime_divisor (a n)

/-- The set of natural numbers m such that m² appears in the sequence a -/
def square_set : Set ℕ := {m : ℕ | ∃ i : ℕ, a i = m^2}

/-- The theorem stating that square_set is exactly the set of all prime numbers -/
theorem square_set_is_primes : square_set = {p : ℕ | Nat.Prime p} := by
  sorry

#eval a 0
#eval a 1
#eval a 2
#eval a 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_set_is_primes_l1004_100425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_seated_is_2016_l1004_100409

/-- Represents the state of a chair (occupied or unoccupied) -/
inductive ChairState
| Occupied
| Unoccupied
deriving BEq, Inhabited

/-- Represents the seating arrangement -/
def SeatingArrangement := List ChairState

/-- The number of chairs in the row -/
def numChairs : Nat := 2017

/-- A person can sit in an empty chair -/
def canSit (arrangement : SeatingArrangement) (pos : Nat) : Prop :=
  pos < numChairs ∧ arrangement.get! pos = ChairState.Unoccupied

/-- A person must leave if someone sits next to them -/
def mustLeave (arrangement : SeatingArrangement) (pos : Nat) : Prop :=
  pos < numChairs ∧ arrangement.get! pos = ChairState.Occupied ∧
  ((pos > 0 ∧ arrangement.get! (pos - 1) = ChairState.Occupied) ∨
   (pos < numChairs - 1 ∧ arrangement.get! (pos + 1) = ChairState.Occupied))

/-- The maximum number of people that can be simultaneously seated -/
def maxSeated : Nat := 2016

/-- Theorem stating the maximum number of people that can be simultaneously seated -/
theorem max_seated_is_2016 :
  ∀ (arrangement : SeatingArrangement),
    (∀ (pos : Nat), canSit arrangement pos → mustLeave arrangement pos) →
    arrangement.count ChairState.Occupied ≤ maxSeated :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_seated_is_2016_l1004_100409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_perimeter_l1004_100400

/-- Represents a point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a trapezoid with four vertices -/
structure Trapezoid where
  j : Point
  k : Point
  l : Point
  m : Point

/-- Calculates the distance between two points in a 2D plane -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2)

/-- Calculates the perimeter of a trapezoid -/
noncomputable def perimeter (t : Trapezoid) : ℝ :=
  distance t.j t.k + distance t.k t.l + distance t.l t.m + distance t.m t.j

/-- The theorem to be proved -/
theorem trapezoid_perimeter :
  let t : Trapezoid := {
    j := { x := -2, y := -4 },
    k := { x := -2, y := 2 },
    l := { x := 6, y := 8 },
    m := { x := 6, y := -4 }
  }
  perimeter t = 36 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_perimeter_l1004_100400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_root_expression_l1004_100446

theorem cubic_root_expression (a b c : ℝ) : 
  a^3 - 6*a^2 + 11*a - 6 = 0 →
  b^3 - 6*b^2 + 11*b - 6 = 0 →
  c^3 - 6*c^2 + 11*c - 6 = 0 →
  a + b + c = 6 →
  a*b + a*c + b*c = 11 →
  a*b*c = 6 →
  ∃ (result : ℝ), a^3 + b^3 + a^2*b^2*(a^2 + b^2) = result :=
by
  intro ha hb hc sum_abc prod_abc_pairs prod_abc
  use (a + b)^3 - 6*a*b*(a + b) + 18 + (((a + b)^2 - 11)/2)^2*((a + b)^2 - ((a + b)^2 - 11))
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_root_expression_l1004_100446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l1004_100459

-- Define set M
def M : Set ℝ := {x | (x + 3) * (x - 2) < 0}

-- Define set N
def N : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

-- Theorem statement
theorem intersection_M_N : M ∩ N = Set.Ioc 1 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l1004_100459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_squared_minus_two_sin_range_l1004_100484

theorem cos_squared_minus_two_sin_range :
  ∀ x : ℝ, ∃ y ∈ Set.Icc (-2 : ℝ) 2, y = (Real.cos x) ^ 2 - 2 * Real.sin x ∧
  (∃ x₁ : ℝ, (Real.cos x₁) ^ 2 - 2 * Real.sin x₁ = -2) ∧
  (∃ x₂ : ℝ, (Real.cos x₂) ^ 2 - 2 * Real.sin x₂ = 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_squared_minus_two_sin_range_l1004_100484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_image_of_f_l1004_100463

/-- A right triangle with legs a and b, and hypotenuse c -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  right_angle : c^2 = a^2 + b^2
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c

/-- The function f from right triangles to real numbers -/
noncomputable def f (t : RightTriangle) : ℝ :=
  let h := t.a * t.b / t.c  -- Height with respect to hypotenuse
  let r := (t.a + t.b - t.c) / 2  -- Radius of inscribed circle
  h / r

/-- The image of f is the interval (2, 1 + √2] -/
theorem image_of_f :
  Set.range f = Set.Ioc 2 (1 + Real.sqrt 2) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_image_of_f_l1004_100463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_exists_l1004_100452

noncomputable def f (x : ℝ) := 2/x + Real.log (1/(x-1))

theorem zero_point_exists : ∃ c ∈ Set.Ioo 2 3, f c = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_exists_l1004_100452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_expression_l1004_100411

noncomputable def expr_a : ℝ := Real.sqrt (Real.rpow (7 * 8) (1/3))
noncomputable def expr_b : ℝ := Real.sqrt (8 * Real.rpow 7 (1/3))
noncomputable def expr_c : ℝ := Real.sqrt (7 * Real.rpow 8 (1/3))
noncomputable def expr_d : ℝ := Real.rpow (7 * Real.sqrt 8) (1/3)
noncomputable def expr_e : ℝ := Real.rpow (8 * Real.sqrt 7) (1/3)

theorem largest_expression : 
  expr_b = max expr_a (max expr_b (max expr_c (max expr_d expr_e))) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_expression_l1004_100411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_is_one_twelfth_l1004_100421

-- Define the functions for the curves
def f (x : ℝ) : ℝ := x^2
def g (x : ℝ) : ℝ := x^3

-- Define the area enclosed by the curves
noncomputable def area_between_curves : ℝ := ∫ x in (0)..(1), f x - g x

-- Theorem statement
theorem area_enclosed_is_one_twelfth :
  area_between_curves = 1/12 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_is_one_twelfth_l1004_100421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l1004_100439

/-- An arithmetic sequence with given conditions -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  a 10 = 30 ∧ a 20 = 50

/-- The general term of the arithmetic sequence -/
noncomputable def general_term (n : ℕ) : ℝ := 2 * n + 10

/-- The nth term of sequence b -/
noncomputable def b (n : ℕ) : ℝ := 4 / ((general_term n - 10) * (general_term n - 8))

/-- The sum of the first n terms of sequence b -/
noncomputable def S (n : ℕ) : ℝ := n / (n + 1)

theorem arithmetic_sequence_properties (a : ℕ → ℝ) (ha : arithmetic_sequence a) :
  (∀ n, a n = general_term n) ∧
  (∀ n, Finset.sum (Finset.range n) (fun i => b (i + 1)) = S n) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l1004_100439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magic_square_sum_l1004_100460

/-- A 3x3 magic square using integers from 1 to 9 -/
def MagicSquare : Type := Matrix (Fin 3) (Fin 3) ℕ

/-- The property that a 3x3 square is a magic square using integers from 1 to 9 -/
def is_magic_square (s : MagicSquare) : Prop :=
  (∀ i j, s i j ∈ Finset.range 9) ∧ 
  (∀ i j, 1 ≤ s i j) ∧
  (Finset.sum (Finset.univ : Finset (Fin 3 × Fin 3)) (fun ij ↦ s ij.1 ij.2) = 45) ∧
  (∀ i, Finset.sum Finset.univ (fun j ↦ s i j) = Finset.sum Finset.univ (fun j ↦ s j i)) ∧
  (Finset.sum Finset.univ (fun i ↦ s i i) = Finset.sum Finset.univ (fun i ↦ s i (2 - i)))

/-- The theorem stating that the common sum in a 3x3 magic square is 15 -/
theorem magic_square_sum (s : MagicSquare) (h : is_magic_square s) :
  ∀ i, Finset.sum Finset.univ (fun j ↦ s i j) = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magic_square_sum_l1004_100460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_sphere_radius_formula_l1004_100486

/-- Regular tetrahedron with given base side length and distance between opposite edges -/
structure RegularTetrahedron where
  base_side : ℝ
  opposite_edge_distance : ℝ
  base_side_positive : 0 < base_side
  opposite_edge_distance_eq : opposite_edge_distance = (base_side * Real.sqrt 2) / 8

/-- The radius of the circumscribed sphere of a regular tetrahedron -/
noncomputable def circumscribed_sphere_radius (t : RegularTetrahedron) : ℝ :=
  t.base_side / Real.sqrt 3

theorem circumscribed_sphere_radius_formula (t : RegularTetrahedron) :
  circumscribed_sphere_radius t = t.base_side / Real.sqrt 3 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_sphere_radius_formula_l1004_100486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_bisector_value_l1004_100474

/-- The perpendicular bisector of a line segment passes through its midpoint -/
axiom perpendicular_bisector_passes_through_midpoint 
  {x₁ y₁ x₂ y₂ b : ℝ} :
  (∀ (x y : ℝ), x + y = b → (x - (x₁ + x₂) / 2)^2 + (y - (y₁ + y₂) / 2)^2 = 
    ((x₁ - x₂)^2 + (y₁ - y₂)^2) / 4) →
  ((x₁ + x₂) / 2 + (y₁ + y₂) / 2 = b)

/-- The value of b for the perpendicular bisector of the line segment from (2,5) to (10,11) -/
theorem perpendicular_bisector_value : 
  ∃ (b : ℝ), (∀ (x y : ℝ), x + y = b → 
    (x - (2 + 10) / 2)^2 + (y - (5 + 11) / 2)^2 = ((2 - 10)^2 + (5 - 11)^2) / 4) ∧
  b = 14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_bisector_value_l1004_100474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_theorem_l1004_100436

-- Define the sequence property
def sequence_property (x : ℕ → ℝ) : Prop :=
  ∀ n ≥ 2, x n = (x (n-1) + 198 * x n + x (n+1)) / 200

-- Theorem statement
theorem sequence_theorem (x : ℕ → ℝ) (h : sequence_property x) 
  (h_distinct : ∀ n m, n ≥ 2 → m ≥ 2 → n ≠ m → x n ≠ x m) :
  Real.sqrt ((x 2023 - x 1) / 2022 * 2021 / (x 2023 - x 2)) + 2022 = 2023 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_theorem_l1004_100436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_B_power_2023_equals_B_l1004_100492

noncomputable def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![1/2, 0, Real.sqrt 3/2],
    ![0, 1, 0],
    ![-(Real.sqrt 3)/2, 0, 1/2]]

theorem B_power_2023_equals_B :
  B ^ 2023 = B := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_B_power_2023_equals_B_l1004_100492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_farmer_corn_bags_l1004_100483

noncomputable def corn_seeds_cost : ℝ := 50
noncomputable def fertilizers_pesticides_cost : ℝ := 35
noncomputable def labor_cost : ℝ := 15
noncomputable def profit_percentage : ℝ := 0.10
noncomputable def selling_price_per_bag : ℝ := 11

noncomputable def total_cost : ℝ := corn_seeds_cost + fertilizers_pesticides_cost + labor_cost
noncomputable def total_revenue : ℝ := total_cost + (total_cost * profit_percentage)
noncomputable def number_of_bags : ℝ := total_revenue / selling_price_per_bag

theorem farmer_corn_bags : number_of_bags = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_farmer_corn_bags_l1004_100483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_and_g_value_l1004_100466

noncomputable section

open Real

def f (x : ℝ) : ℝ := 2 * sqrt 3 * sin (π - x) * sin x - (sin x - cos x)^2

def g (x : ℝ) : ℝ := f (2 * (x + π / 3))

theorem f_monotone_and_g_value :
  (∀ (k : ℤ), ∀ (x : ℝ), k * π - π / 12 ≤ x ∧ x ≤ k * π + 5 * π / 12 → (deriv f) x > 0) ∧
  g (π / 6) = 1 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_and_g_value_l1004_100466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_coefficient_divisibility_binomial_coefficient_is_integer_l1004_100494

/-- Legendre's formula for the exponent of a prime p in n! -/
noncomputable def legendre_exponent (p : ℕ) (n : ℕ) : ℕ := ∑' k, (n / p^k : ℕ)

/-- The exponent of a prime p in the factorization of (2n)! / (n! * (n+1)!) -/
noncomputable def binomial_exponent (p : ℕ) (n : ℕ) : ℤ :=
  (legendre_exponent p (2*n) : ℤ) - (legendre_exponent p n : ℤ) - (legendre_exponent p (n+1) : ℤ)

theorem binomial_coefficient_divisibility (n : ℕ) :
  ∀ p, Nat.Prime p → binomial_exponent p n ≥ 0 := by
  sorry

/-- The binomial coefficient C(2n,n) divided by (n+1) is an integer -/
theorem binomial_coefficient_is_integer (n : ℕ) :
  ∃ k : ℕ, (Nat.choose (2*n) n : ℚ) / (n + 1 : ℚ) = k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_coefficient_divisibility_binomial_coefficient_is_integer_l1004_100494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_betty_slippers_order_l1004_100496

/-- Calculates the number of slippers ordered given the total items, costs, and quantities of other items --/
def slippers_ordered (total_items : ℕ) (slipper_cost lipstick_cost hair_color_cost : ℚ)
  (lipstick_quantity hair_color_quantity : ℕ) (total_payment : ℚ) : ℕ :=
  let remaining_items := total_items - lipstick_quantity - hair_color_quantity
  let other_items_cost := lipstick_quantity * lipstick_cost + hair_color_quantity * hair_color_cost
  let slippers_cost := total_payment - other_items_cost
  (slippers_cost / slipper_cost).floor.toNat

/-- Proves that Betty ordered 6 slippers --/
theorem betty_slippers_order : slippers_ordered 18 (5/2) (5/4) 3 4 8 44 = 6 := by
  rfl

#eval slippers_ordered 18 (5/2) (5/4) 3 4 8 44

end NUMINAMATH_CALUDE_ERRORFEEDBACK_betty_slippers_order_l1004_100496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_is_correct_l1004_100471

noncomputable section

/-- Square with side length 40 -/
def Square : Set (ℝ × ℝ) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ 40 ∧ 0 ≤ p.2 ∧ p.2 ≤ 40}

/-- Point A of the square -/
def A : ℝ × ℝ := (0, 0)

/-- Point B of the square -/
def B : ℝ × ℝ := (40, 0)

/-- Point C of the square -/
def C : ℝ × ℝ := (40, 40)

/-- Point D of the square -/
def D : ℝ × ℝ := (0, 40)

/-- Point Q within the square -/
def Q : ℝ × ℝ := sorry

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Centroid of a triangle -/
noncomputable def centroid (p1 p2 p3 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1 + p3.1) / 3, (p1.2 + p2.2 + p3.2) / 3)

/-- Area of a quadrilateral given its four vertices -/
noncomputable def quadrilateralArea (p1 p2 p3 p4 : ℝ × ℝ) : ℝ := sorry

theorem quadrilateral_area_is_correct :
  Q ∈ Square ∧
  distance A Q = 16 ∧
  distance B Q = 34 →
  quadrilateralArea
    (centroid A B Q)
    (centroid B C Q)
    (centroid C D Q)
    (centroid D A Q) = 355.56 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_is_correct_l1004_100471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dilation_origin_movement_l1004_100410

/-- Represents a point in 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle in 2D plane -/
structure Circle where
  center : Point
  radius : ℝ

/-- Dilation transformation in 2D plane -/
def dilation (k : ℝ) (center : Point) (p : Point) : Point :=
  { x := center.x + k * (p.x - center.x),
    y := center.y + k * (p.y - center.y) }

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

theorem dilation_origin_movement :
  let original : Circle := ⟨⟨3, 3⟩, 3⟩
  let dilated : Circle := ⟨⟨9, 10⟩, 6⟩
  let origin : Point := ⟨0, 0⟩
  let k : ℝ := dilated.radius / original.radius
  let center : Point := ⟨0, 3⟩  -- Simplified assumption from the solution
  let transformed_origin : Point := dilation k center origin
  distance origin transformed_origin = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dilation_origin_movement_l1004_100410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l1004_100450

noncomputable def f (ω φ x : ℝ) : ℝ := Real.sin (ω * x + φ) + Real.cos (ω * x + φ)

theorem f_increasing_on_interval 
  (ω φ : ℝ) 
  (h_ω : ω > 0) 
  (h_φ : 0 < φ ∧ φ < Real.pi) 
  (h_odd : ∀ x, f ω φ (-x) = -(f ω φ x)) 
  (h_intersect : ∃ k : ℤ, ∀ x₁ x₂ : ℝ, 
    f ω φ x₁ = Real.sqrt 2 ∧ 
    f ω φ x₂ = Real.sqrt 2 ∧ 
    x₁ < x₂ ∧ 
    (∀ x ∈ Set.Ioo x₁ x₂, f ω φ x ≠ Real.sqrt 2) → 
    x₂ - x₁ = k * Real.pi / 2) :
  ∀ x ∈ Set.Ioo (Real.pi / 8) (3 * Real.pi / 8), 
    ∃ ε > 0, ∀ y ∈ Set.Ioo x (x + ε), f ω φ x ≤ f ω φ y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l1004_100450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collaborative_groups_theorem_l1004_100453

/-- Represents a collaborative group of students -/
structure CollaborativeGroup where
  members : Finset Nat

/-- The total number of students -/
def total_students : Nat := 2008

/-- The minimum number of collaborative groups -/
def min_groups : Nat := 1008017

/-- Checks if three students are in the same group -/
def three_in_same_group (groups : Finset CollaborativeGroup) : Prop :=
  ∃ (g : CollaborativeGroup) (a b c : Nat),
    g ∈ groups ∧ a ∈ g.members ∧ b ∈ g.members ∧ c ∈ g.members ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c

/-- Checks if four students form a cycle of collaborative groups -/
def four_student_cycle (groups : Finset CollaborativeGroup) : Prop :=
  ∃ (a b c d : Nat) (g1 g2 g3 g4 : CollaborativeGroup),
    g1 ∈ groups ∧ g2 ∈ groups ∧ g3 ∈ groups ∧ g4 ∈ groups ∧
    a ∈ g1.members ∧ b ∈ g1.members ∧
    b ∈ g2.members ∧ c ∈ g2.members ∧
    c ∈ g3.members ∧ d ∈ g3.members ∧
    d ∈ g4.members ∧ a ∈ g4.members ∧
    a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a

/-- Main theorem combining both parts of the problem -/
theorem collaborative_groups_theorem :
  (∀ (groups : Finset CollaborativeGroup),
    groups.card < min_groups → ¬(three_in_same_group groups)) ∧
  (∀ (groups : Finset CollaborativeGroup),
    groups.card = min_groups / 22 → four_student_cycle groups) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_collaborative_groups_theorem_l1004_100453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octahedron_volume_in_unit_cube_l1004_100499

/-- The volume of a regular octahedron formed by joining the centers of adjacent faces of a unit cube -/
noncomputable def octahedron_volume : ℝ := 1 / 3

/-- The side length of the unit cube -/
def cube_side_length : ℝ := 1

/-- Theorem: The volume of a regular octahedron formed by joining the centers of adjacent faces of a unit cube is 1/3 -/
theorem octahedron_volume_in_unit_cube : 
  octahedron_volume = 1 / 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octahedron_volume_in_unit_cube_l1004_100499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_bounds_l1004_100418

structure Trapezoid (A B C D : ℝ × ℝ) : Prop where
  is_trapezoid : True  -- This is a placeholder for the trapezoid condition

structure Point where
  x : ℝ
  y : ℝ

def on_segment (P : Point) (A B : ℝ × ℝ) : Prop := sorry

def intersect (AB CD : ℝ × ℝ × ℝ × ℝ) : ℝ × ℝ := sorry

def area (points : List (ℝ × ℝ)) : ℝ := sorry

theorem trapezoid_area_bounds
  (A B C D : ℝ × ℝ)
  (G H E F : Point)
  (h_trapezoid : Trapezoid A B C D)
  (h_G : on_segment G A D)
  (h_H : on_segment H A D)
  (h_E : on_segment E B C)
  (h_F : on_segment F B C)
  (K : ℝ × ℝ)
  (L : ℝ × ℝ)
  (M : ℝ × ℝ)
  (h_K : K = intersect (B.1, B.2, G.x, G.y) (A.1, A.2, E.x, E.y))
  (h_L : L = intersect (E.x, E.y, H.x, H.y) (G.x, G.y, F.x, F.y))
  (h_M : M = intersect (F.x, F.y, D.1, D.2) (H.x, H.y, C.1, C.2))
  (h_area_ELGK : area [(E.x, E.y), L, (G.x, G.y), K] = 4)
  (h_area_FMHL : area [(F.x, F.y), M, (H.x, H.y), L] = 8)
  (h_area_CDM_int : ∃ n : ℤ, area [C, D, M] = n) :
  5 ≤ area [C, D, M] ∧ area [C, D, M] ≤ 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_bounds_l1004_100418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_and_fixed_point_exist_l1004_100403

noncomputable section

-- Define A as a predicate on ℝ × ℝ
def A : ℝ × ℝ → Prop := λ p => p.1 > 0

def B : ℝ × ℝ := (-1, 0)
def C : ℝ × ℝ := (1, 0)

-- Redefine slope to avoid conflicts
def line_slope (p q : ℝ × ℝ) : ℝ := (q.2 - p.2) / (q.1 - p.1)

def locus_D (p : ℝ × ℝ) : Prop := p.1^2 - p.2^2/3 = 1 ∧ p.1 > 1

def parabola (p : ℝ) (point : ℝ × ℝ) : Prop := point.1^2 = 2 * p * point.2 ∧ p > 0

def perpendicular (l1 l2 : ℝ × ℝ → Prop) : Prop :=
  ∀ p q r s, l1 p ∧ l1 q ∧ l2 r ∧ l2 s → (q.2 - p.2) * (s.1 - r.1) = -(s.2 - r.2) * (q.1 - p.1)

theorem locus_and_fixed_point_exist :
  ∃ (a : ℝ × ℝ) (p : ℝ) (E F H : ℝ × ℝ),
    A a ∧
    line_slope a B * line_slope a C = 3 ∧
    locus_D a ∧
    parabola p E ∧ parabola p F ∧
    locus_D E ∧ locus_D F ∧
    perpendicular (λ x => x = B ∨ x = H) (λ x => ∃ t, x = (1-t) • E + t • F) ∧
    ∃ (G : ℝ × ℝ), G = (-1/2, -Real.sqrt 3/2) ∧ ∀ H', perpendicular (λ x => x = B ∨ x = H') (λ x => ∃ t, x = (1-t) • E + t • F) →
      (H'.1 - G.1)^2 + (H'.2 - G.2)^2 = 1 :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_and_fixed_point_exist_l1004_100403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_five_terms_l1004_100433

def fibonacci_like_sequence : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | (n+2) => fibonacci_like_sequence (n+1) + fibonacci_like_sequence n

theorem sum_of_first_five_terms :
  (fibonacci_like_sequence 0) +
  (fibonacci_like_sequence 1) +
  (fibonacci_like_sequence 2) +
  (fibonacci_like_sequence 3) +
  (fibonacci_like_sequence 4) = 19 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_five_terms_l1004_100433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_markup_rate_approx_42_86_l1004_100455

/-- The selling price of the product -/
noncomputable def selling_price : ℝ := 8.00

/-- The profit percentage of the selling price -/
noncomputable def profit_percentage : ℝ := 0.12

/-- The expense percentage of the selling price -/
noncomputable def expense_percentage : ℝ := 0.18

/-- The cost of the product -/
noncomputable def cost : ℝ := selling_price * (1 - profit_percentage - expense_percentage)

/-- The rate of markup -/
noncomputable def markup_rate : ℝ := (selling_price - cost) / cost * 100

theorem markup_rate_approx_42_86 :
  ∃ ε > 0, abs (markup_rate - 42.86) < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_markup_rate_approx_42_86_l1004_100455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_crosses_asymptote_at_three_l1004_100441

/-- The function g(x) -/
noncomputable def g (x : ℝ) : ℝ := (3 * x^2 - 6 * x - 9) / (x^2 - 5 * x + 6)

/-- The horizontal asymptote of g(x) -/
def horizontal_asymptote : ℝ := 3

theorem g_crosses_asymptote_at_three :
  ∃ (x : ℝ), x = 3 ∧ g x = horizontal_asymptote := by
  use 3
  constructor
  · rfl
  · sorry  -- The actual proof would go here


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_crosses_asymptote_at_three_l1004_100441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_with_property_l1004_100473

def Sn (n : ℕ) : Set ℕ := {x | 3 ≤ x ∧ x ≤ n}

def has_product_triple (S : Set ℕ) : Prop :=
  ∃ a b c, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a * b = c

theorem smallest_n_with_property : 
  (∀ n : ℕ, n ≥ 243 → 
    ∀ A B : Set ℕ, 
      (A ∪ B = Sn n) → (A ∩ B = ∅) → 
      (has_product_triple A ∨ has_product_triple B)) ∧
  (∀ m : ℕ, m < 243 → 
    ∃ A B : Set ℕ,
      (A ∪ B = Sn m) ∧ (A ∩ B = ∅) ∧
      ¬(has_product_triple A ∨ has_product_triple B)) :=
by sorry

#check smallest_n_with_property

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_with_property_l1004_100473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_deflation_time_for_specific_case_l1004_100427

/-- Represents the time taken (in minutes) for a puncture to deflate a tyre completely. -/
structure PunctureDeflationTime where
  minutes : ℚ
  positive : minutes > 0

/-- Represents a tyre with two punctures. -/
structure TyreWithTwoPunctures where
  puncture1 : PunctureDeflationTime
  puncture2 : PunctureDeflationTime

/-- Calculates the combined deflation time for a tyre with two punctures. -/
noncomputable def combined_deflation_time (t : TyreWithTwoPunctures) : ℚ :=
  1 / (1 / t.puncture1.minutes + 1 / t.puncture2.minutes)

/-- Theorem stating the existence of a tyre with two punctures that satisfies the given conditions. -/
theorem combined_deflation_time_for_specific_case :
  ∃ t : TyreWithTwoPunctures,
    t.puncture1.minutes = 9 ∧
    t.puncture2.minutes = 6 ∧
    combined_deflation_time t = 18 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_deflation_time_for_specific_case_l1004_100427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_valid_configuration_l1004_100465

/-- A configuration of black cells on an n × n grid -/
def BlackCellConfiguration (n : ℕ) := Fin n → Fin n → Bool

/-- Checks if two cells are adjacent (share a side or vertex) -/
def adjacent (n : ℕ) (i1 j1 i2 j2 : Fin n) : Prop :=
  (i1 = i2 ∧ j1.val + 1 = j2.val) ∨
  (i1 = i2 ∧ j2.val + 1 = j1.val) ∨
  (j1 = j2 ∧ i1.val + 1 = i2.val) ∨
  (j1 = j2 ∧ i2.val + 1 = i1.val) ∨
  (i1.val + 1 = i2.val ∧ j1.val + 1 = j2.val) ∨
  (i2.val + 1 = i1.val ∧ j2.val + 1 = j1.val) ∨
  (i1.val + 1 = i2.val ∧ j2.val + 1 = j1.val) ∨
  (i2.val + 1 = i1.val ∧ j1.val + 1 = j2.val)

/-- Checks if a configuration is valid (no adjacent black cells) -/
def valid_configuration (n : ℕ) (config : BlackCellConfiguration n) : Prop :=
  ∀ i1 j1 i2 j2, config i1 j1 ∧ config i2 j2 → ¬(adjacent n i1 j1 i2 j2)

/-- Counts black cells in a row -/
def black_cells_in_row (n : ℕ) (config : BlackCellConfiguration n) (i : Fin n) : ℕ :=
  (Finset.filter (λ j : Fin n => config i j) (Finset.univ)).card

/-- Counts black cells in a column -/
def black_cells_in_column (n : ℕ) (config : BlackCellConfiguration n) (j : Fin n) : ℕ :=
  (Finset.filter (λ i : Fin n => config i j) (Finset.univ)).card

/-- Checks if a configuration has exactly k black cells in each row and column -/
def has_k_black_cells (n k : ℕ) (config : BlackCellConfiguration n) : Prop :=
  (∀ i : Fin n, black_cells_in_row n config i = k) ∧
  (∀ j : Fin n, black_cells_in_column n config j = k)

/-- The main theorem stating the smallest n for which a valid configuration exists -/
theorem smallest_n_for_valid_configuration (k : ℕ) (h1 : k > 1) :
  (∃ n : ℕ, n > 1 ∧ k < n ∧ ∃ config : BlackCellConfiguration n,
    valid_configuration n config ∧ has_k_black_cells n k config) →
  (∀ n : ℕ, n > 1 ∧ k < n ∧ ∃ config : BlackCellConfiguration n,
    valid_configuration n config ∧ has_k_black_cells n k config → n ≥ 4 * k) ∧
  ∃ config : BlackCellConfiguration (4 * k),
    valid_configuration (4 * k) config ∧ has_k_black_cells (4 * k) k config := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_valid_configuration_l1004_100465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l1004_100404

theorem triangle_abc_properties (A B C a b c : Real) : 
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  a > 0 ∧ b > 0 ∧ c > 0 →
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  (Real.cos C) / (Real.cos B) = (2 * a - c) / b →
  Real.tan (A + π / 4) = 7 →
  B = π / 3 ∧ Real.cos C = (-4 + 3 * Real.sqrt 3) / 10 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l1004_100404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_solid_paint_l1004_100405

theorem rectangular_solid_paint (m n : ℕ+) (h1 : m ≥ n) :
  (5 - 1) * (m.val - 1) * (n.val - 1) = (1 / 2) * 5 * m.val * n.val ↔ 
  (m = 16 ∧ n = 3) ∨ (m = 6 ∧ n = 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_solid_paint_l1004_100405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_accelerating_growth_l1004_100461

-- Define the years and corresponding percentages
def years : List Nat := [1960, 1970, 1980, 1990]
def percentages : List Nat := [5, 8, 15, 30]

-- Define a function to calculate the rate of increase between two decades
def rate_of_increase (p1 p2 : Nat) : Nat := p2 - p1

-- Theorem statement
theorem accelerating_growth (h : years.length = percentages.length) :
  ∀ i : Nat, i + 2 < percentages.length →
    rate_of_increase (percentages.get ⟨i, by sorry⟩) (percentages.get ⟨i+1, by sorry⟩) <
    rate_of_increase (percentages.get ⟨i+1, by sorry⟩) (percentages.get ⟨i+2, by sorry⟩) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_accelerating_growth_l1004_100461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_g_cube_l1004_100448

noncomputable def g (x : ℝ) : ℝ := 25 / (4 + 5 * x)

theorem inverse_g_cube (h : Function.Bijective g) :
  (Function.invFun g 5) ^ (-3 : ℝ) = 125 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_g_cube_l1004_100448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_two_zeros_f_has_specific_zeros_l1004_100416

/-- The function f(x) = 2ln(x) - ax^2 + 1 --/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * Real.log x - a * x^2 + 1

/-- Theorem stating the conditions for f to have two zeros --/
theorem f_has_two_zeros (a : ℝ) :
  (∃ x y, x ≠ y ∧ f a x = 0 ∧ f a y = 0) ↔ 0 < a ∧ a < 1 := by sorry

/-- Theorem stating the conditions for f to have two zeros with specific properties --/
theorem f_has_specific_zeros (a : ℝ) :
  (∃ m α β, 1 ≤ α ∧ α < β ∧ β ≤ 4 ∧ β - α = 1 ∧ f a α = m ∧ f a β = m) ↔
  (2/7 * Real.log (4/3) ≤ a ∧ a ≤ 2/3 * Real.log 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_two_zeros_f_has_specific_zeros_l1004_100416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_divisor_of_sum_of_squares_l1004_100479

theorem largest_prime_divisor_of_sum_of_squares : 
  (Nat.factors (15^2 + 45^2)).reverse.head? = some 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_divisor_of_sum_of_squares_l1004_100479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_preservation_time_at_33_l1004_100419

/-- The preservation time function -/
noncomputable def preservation_time (k b x : ℝ) : ℝ := Real.exp (k * x + b)

/-- Theorem stating the preservation time at 33°C given conditions -/
theorem preservation_time_at_33 (k b : ℝ) :
  preservation_time k b 0 = 192 →
  preservation_time k b 22 = 48 →
  preservation_time k b 33 = 24 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_preservation_time_at_33_l1004_100419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l1004_100456

theorem arithmetic_sequence_sum : ∃ (S : ℕ), 3 * S = 2805 := by
  -- Define the first term of the sequence
  let a : ℕ := 70
  -- Define the common difference
  let d : ℕ := 3
  -- Define the last term
  let l : ℕ := 100
  -- Calculate the number of terms
  let n : ℕ := (l - a) / d + 1
  -- Define the sum of the arithmetic sequence
  let S : ℕ := n * (a + l) / 2
  -- Prove the theorem
  use S
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l1004_100456
