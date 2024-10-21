import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_distance_l642_64290

-- Define the hyperbola equation
def hyperbola (x y : ℝ) : Prop := y^2 / 25 - x^2 / 9 = 1

-- Define the distance between foci
noncomputable def distance_between_foci : ℝ := 2 * Real.sqrt 34

-- Theorem statement
theorem hyperbola_foci_distance :
  distance_between_foci = 2 * Real.sqrt 34 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_distance_l642_64290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_T_l642_64224

-- Define ω as a complex number
def ω : ℂ := Complex.I

-- Define the set T
def T : Set ℂ :=
  {z : ℂ | ∃ (a b c : ℝ), 0 ≤ a ∧ a ≤ 2 ∧ 0 ≤ b ∧ b ≤ 1 ∧ 0 ≤ c ∧ c ≤ 2 ∧
    z = a + b * ω + c * ω^2}

-- State the theorem
theorem area_of_T : MeasureTheory.volume T = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_T_l642_64224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bicycle_transactions_profit_l642_64280

/-- Represents a transaction with a profit or loss percentage -/
structure Transaction where
  profitPercentage : ℝ

/-- Calculates the final price after a sequence of transactions -/
noncomputable def finalPrice (initialPrice : ℝ) (transactions : List Transaction) : ℝ :=
  transactions.foldl (fun price t => price * (1 + t.profitPercentage / 100)) initialPrice

/-- Calculates the overall profit percentage -/
noncomputable def overallProfitPercentage (initialPrice : ℝ) (finalPrice : ℝ) : ℝ :=
  (finalPrice - initialPrice) / initialPrice * 100

/-- Theorem stating the overall profit percentage for the bicycle transactions -/
theorem bicycle_transactions_profit :
  let transactions : List Transaction := [
    { profitPercentage := 35 },
    { profitPercentage := -25 },
    { profitPercentage := 20 },
    { profitPercentage := -15 }
  ]
  let initialPrice := 100
  let finalPrice := finalPrice initialPrice transactions
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.001 ∧ abs (overallProfitPercentage initialPrice finalPrice - 3.275) < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bicycle_transactions_profit_l642_64280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_solution_set_l642_64242

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then 1 / x else (1 / 3) ^ x

-- Define the solution set
def solution_set : Set ℝ := Set.Icc (-3) 1

-- Theorem statement
theorem f_inequality_solution_set :
  { x : ℝ | |f x| ≥ 1/3 } = solution_set := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_solution_set_l642_64242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_proof_perpendicular_line_proof_l642_64277

-- Define the intersection point
def intersection_point : ℝ × ℝ := (1, 3)

-- Define the lines
def l₁ (x y : ℝ) : Prop := x + y - 4 = 0
def l₂ (x y : ℝ) : Prop := x - y + 2 = 0
def l_given (x y : ℝ) : Prop := 2*x - y - 1 = 0

-- Define the parallel and perpendicular lines
def l_parallel (x y : ℝ) : Prop := 2*x - y + 1 = 0
def l_perpendicular (x y : ℝ) : Prop := x + 2*y - 7 = 0

-- Theorem for the parallel case
theorem parallel_line_proof :
  (l₁ intersection_point.1 intersection_point.2 ∧ 
   l₂ intersection_point.1 intersection_point.2) →
  (l_parallel intersection_point.1 intersection_point.2 ∧
   ∃ (k : ℝ), ∀ (x y : ℝ), l_parallel x y ↔ l_given x y ∧ k = 0) :=
by sorry

-- Theorem for the perpendicular case
theorem perpendicular_line_proof :
  (l₁ intersection_point.1 intersection_point.2 ∧ 
   l₂ intersection_point.1 intersection_point.2) →
  (l_perpendicular intersection_point.1 intersection_point.2 ∧
   ∀ (x₁ y₁ x₂ y₂ : ℝ), 
     l_given x₁ y₁ ∧ l_given x₂ y₂ ∧ x₁ ≠ x₂ →
     (y₂ - y₁) / (x₂ - x₁) * ((y₂ - y₁) / (x₂ - x₁) + 2) = -1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_proof_perpendicular_line_proof_l642_64277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_comet_midpoint_distance_l642_64235

/-- Represents an elliptical orbit -/
structure EllipticalOrbit where
  perihelion : ℝ
  aphelion : ℝ

/-- The distance from the focus to the midpoint of an elliptical orbit -/
noncomputable def midpoint_distance (orbit : EllipticalOrbit) : ℝ :=
  (orbit.perihelion + orbit.aphelion) / 4

/-- Theorem stating that for the given orbit, the midpoint distance is 9 AU -/
theorem comet_midpoint_distance :
  let orbit : EllipticalOrbit := { perihelion := 3, aphelion := 15 }
  midpoint_distance orbit = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_comet_midpoint_distance_l642_64235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exist_pair_sum_le_two_l642_64207

theorem exist_pair_sum_le_two (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_squares : a^2 + b^2 + c^2 + d^2 = 4) :
  ∃ (i j : ℝ), i ∈ ({a, b, c, d} : Set ℝ) ∧ j ∈ ({a, b, c, d} : Set ℝ) ∧ i ≠ j ∧ i + j ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exist_pair_sum_le_two_l642_64207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_limit_is_infinity_l642_64266

noncomputable def sequence_limit (n : ℝ) : ℝ := ((n+1)^3 - (n-1)^3) / ((n+1)^2 - (n-1)^2)

theorem sequence_limit_is_infinity :
  Filter.Tendsto sequence_limit Filter.atTop Filter.atTop :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_limit_is_infinity_l642_64266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_equals_506_l642_64270

open BigOperators

def numerator : ℚ :=
  ∏ k in Finset.range 25, (1 + 21 / (↑k + 1))

def denominator : ℚ :=
  ∏ k in Finset.range 21, (1 + 23 / (↑k + 1))

theorem fraction_equals_506 : 
  numerator / denominator = 22 * 23 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_equals_506_l642_64270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l642_64238

theorem triangle_property (A B C a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Positive angles
  A + B + C = π ∧  -- Sum of angles in a triangle
  A < π/2 ∧ B < π/2 ∧ C < π/2 ∧  -- Acute triangle
  a > 0 ∧ b > 0 ∧ c > 0 ∧  -- Positive side lengths
  a * Real.sin ((A + C) / 2) = b * Real.sin A ∧  -- Given condition
  c = 1 →  -- Given condition
  B = π/3 ∧ 
  ∃ S : ℝ, S = (1/2) * a * b * Real.sin C ∧ 
              Real.sqrt 3/8 < S ∧ S < Real.sqrt 3/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l642_64238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_A_in_U_l642_64259

def U : Set ℕ := {x : ℕ | (Real.sqrt (x : ℝ)) ≤ 2}
def A : Set ℕ := {2, 3}

theorem complement_of_A_in_U : 
  (U \ A) = {0, 1, 4} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_A_in_U_l642_64259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_prime_cube_difference_l642_64246

theorem unique_prime_cube_difference (p : ℕ) : 
  (Prime p ∧ 
   ∃ q r : ℕ+, (¬(p ∣ q) ∧ ¬(3 ∣ q) ∧ p^3 = r^3 - q^2)) 
  ↔ p = 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_prime_cube_difference_l642_64246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_area_is_4pi_l642_64278

/-- A rectangle in 2D space --/
structure Rectangle where
  x1 : ℝ
  y1 : ℝ
  x2 : ℝ
  y2 : ℝ
  x3 : ℝ
  y3 : ℝ
  x4 : ℝ
  y4 : ℝ

/-- A circle in 2D space --/
structure Circle where
  center_x : ℝ
  center_y : ℝ
  radius : ℝ

/-- The area of intersection between a rectangle and a circle --/
noncomputable def intersectionArea (r : Rectangle) (c : Circle) : ℝ := 
  sorry

/-- The given rectangle --/
def givenRectangle : Rectangle := {
  x1 := 1, y1 := 5,
  x2 := 12, y2 := 5,
  x3 := 12, y3 := -4,
  x4 := 1, y4 := -4
}

/-- The given circle --/
def givenCircle : Circle := {
  center_x := 1,
  center_y := 4,
  radius := 4
}

theorem intersection_area_is_4pi :
  intersectionArea givenRectangle givenCircle = 4 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_area_is_4pi_l642_64278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_percentage_approx_72_63_l642_64206

/-- Calculates the percentage of water in a mixture of two liquids -/
noncomputable def water_percentage_in_mixture (volume1 : ℝ) (liquid_percent1 : ℝ) (volume2 : ℝ) (liquid_percent2 : ℝ) : ℝ :=
  let total_volume := volume1 + volume2
  let total_liquid := volume1 * (liquid_percent1 / 100) + volume2 * (liquid_percent2 / 100)
  let total_water := total_volume - total_liquid
  (total_water / total_volume) * 100

/-- Theorem stating that the water percentage in the mixture is approximately 72.63% -/
theorem water_percentage_approx_72_63 :
  ∃ ε > 0, abs (water_percentage_in_mixture 100 25 90 30 - 72.63) < ε :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_percentage_approx_72_63_l642_64206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_property_l642_64295

theorem quadratic_root_property (a b c : ℝ) (x₁ x₂ : ℂ) : 
  (∀ x : ℂ, a * x^2 + b * x + c = 0 ↔ x = x₁ ∨ x = x₂) →
  x₁ ≠ x₂ →
  (x₁^3).im = 0 →
  a ≠ 0 →
  b ≠ 0 →
  a * c / b^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_property_l642_64295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l642_64210

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 2*x - 3) / Real.log (1/2)

-- State the theorem
theorem f_increasing_on_interval :
  StrictMonoOn f (Set.Iio (-1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l642_64210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_union_l642_64251

def U : Set ℕ := {1, 2, 3, 4}
def A : Set ℕ := {1, 3}
def B : Set ℕ := {1, 4}

theorem complement_of_union : (U \ (A ∪ B)) = {2} := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_union_l642_64251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l642_64204

theorem simplify_expression : 
  ∃ (a b c : ℕ), 
    (Real.sqrt 2 - 1)^(1 - Real.sqrt 3) / (Real.sqrt 2 + 1)^(1 + Real.sqrt 3) = a - b * Real.sqrt c ∧
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    ∀ (p : ℕ), Nat.Prime p → ¬(p^2 ∣ c) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l642_64204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_l642_64230

-- Define the circle C
def circleC (r : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 = r^2}

-- Define the line l
def lineL : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 - p.2 + 2 = 0}

-- Define the chord length
noncomputable def chordLength (r : ℝ) : ℝ :=
  2 * Real.sqrt (r^2 - 2)

theorem circle_line_intersection (r : ℝ) (h1 : r > 0) :
  chordLength r = 2 * Real.sqrt 2 → r = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_l642_64230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_equal_dihedral_angles_l642_64248

/-- A structure representing a triangular pyramid -/
structure TriangularPyramid where
  /-- The volume of the pyramid -/
  volume : ℝ
  /-- The dihedral angles of the pyramid -/
  dihedralAngles : Set ℝ
  /-- Function to get the length of an edge -/
  edgeLength : Edge → ℝ

/-- An enumeration of the edges of a triangular pyramid -/
inductive Edge
  | AF
  | BF
  | CF
  | A₁F₁
  | B₁F₁
  | C₁F₁

/-- Given two triangular pyramids with equal dihedral angles, their volumes are proportional to 
    the products of the lengths of the three edges forming the equal dihedral angles. -/
theorem volume_ratio_equal_dihedral_angles 
  (FABC F₁A₁B₁C₁ : TriangularPyramid) 
  (h_equal_dihedral : FABC.dihedralAngles = F₁A₁B₁C₁.dihedralAngles) :
  FABC.volume / F₁A₁B₁C₁.volume = 
  (FABC.edgeLength Edge.AF * FABC.edgeLength Edge.BF * FABC.edgeLength Edge.CF) / 
  (F₁A₁B₁C₁.edgeLength Edge.A₁F₁ * F₁A₁B₁C₁.edgeLength Edge.B₁F₁ * F₁A₁B₁C₁.edgeLength Edge.C₁F₁) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_equal_dihedral_angles_l642_64248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ice_production_result_l642_64254

/-- Represents the ice cube production scenario -/
structure IceProduction where
  water_per_cube : ℚ
  cube_weight : ℚ
  cubes_per_hour : ℚ
  ice_maker_cost_per_hour : ℚ
  water_cost_per_ounce : ℚ
  total_cost : ℚ

/-- Calculates the amount of ice produced in pounds -/
def ice_amount (prod : IceProduction) : ℚ :=
  let cost_per_cube := prod.water_per_cube * prod.water_cost_per_ounce + 
                       prod.ice_maker_cost_per_hour / prod.cubes_per_hour
  let num_cubes := prod.total_cost / cost_per_cube
  num_cubes * prod.cube_weight

/-- Theorem stating that the ice production scenario results in 10 pounds of ice -/
theorem ice_production_result (prod : IceProduction) 
  (h1 : prod.water_per_cube = 2)
  (h2 : prod.cube_weight = 1/16)
  (h3 : prod.cubes_per_hour = 10)
  (h4 : prod.ice_maker_cost_per_hour = 3/2)
  (h5 : prod.water_cost_per_ounce = 1/10)
  (h6 : prod.total_cost = 56) :
  ice_amount prod = 10 := by
  sorry

#eval ice_amount {
  water_per_cube := 2,
  cube_weight := 1/16,
  cubes_per_hour := 10,
  ice_maker_cost_per_hour := 3/2,
  water_cost_per_ounce := 1/10,
  total_cost := 56
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ice_production_result_l642_64254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_reflections_is_15_l642_64241

/-- The maximum number of internal reflections in a prism -/
def max_reflections (angle_increment : ℚ) (max_angle : ℚ) : ℕ :=
  (max_angle / angle_increment).floor.toNat

/-- Theorem: The maximum number of internal reflections is 15 -/
theorem max_reflections_is_15 :
  max_reflections 6 90 = 15 := by
  -- Unfold the definition of max_reflections
  unfold max_reflections
  -- Evaluate the expression
  norm_num
  -- QED
  rfl

#eval max_reflections 6 90

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_reflections_is_15_l642_64241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_min_area_l642_64250

noncomputable section

-- Define the coordinates of points E' and F'
def E' : ℝ × ℝ := (0, Real.sqrt 3)
def F' : ℝ × ℝ := (0, -Real.sqrt 3)

-- Define the condition for point G
def slope_product (G : ℝ × ℝ) : ℝ := 
  let (x, y) := G
  ((y - E'.2) / x) * ((y - F'.2) / x)

-- Define the trajectory equation
def trajectory (G : ℝ × ℝ) : Prop :=
  let (x, y) := G
  x ≠ 0 ∧ x^2 / 4 + y^2 / 3 = 1

-- Define the area of a triangle
def area_triangle (A B C : ℝ × ℝ) : ℝ :=
  abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2)) / 2

-- Theorem statement
theorem trajectory_and_min_area :
  ∀ G : ℝ × ℝ, 
    slope_product G = -3/4 →
    (trajectory G ∧
     ∃ A B : ℝ × ℝ,
       trajectory A ∧ 
       trajectory B ∧
       (A.1 * B.1 + A.2 * B.2 = 0) ∧  -- OA ⊥ OB
       (∀ A' B' : ℝ × ℝ,
         trajectory A' ∧ 
         trajectory B' ∧
         (A'.1 * B'.1 + A'.2 * B'.2 = 0) →
         area_triangle (0, 0) A' B' ≥ 12/7)) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_min_area_l642_64250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_negative_25pi_over_3_l642_64236

open Real

-- Define the function f
noncomputable def f (α : ℝ) : ℝ :=
  (cos (π / 2 + α) * sin (3 * π / 2 - α)) /
  (cos (-π - α) * tan (π - α))

-- Theorem statement
theorem f_value_at_negative_25pi_over_3 :
  f (-25 * π / 3) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_negative_25pi_over_3_l642_64236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_line_equation_l642_64292

-- Define the circles C₁ and C₂
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 4
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 4*y + 4 = 0

-- Define symmetry about a line
def symmetric_about_line (C₁ C₂ : ℝ → ℝ → Prop) (l : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, C₁ x y ∧ C₂ x y → l x y

-- Theorem statement
theorem symmetry_line_equation :
  symmetric_about_line C₁ C₂ (fun x y ↦ y = x - 2) := by
  sorry

#check symmetry_line_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_line_equation_l642_64292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sum_special_angles_l642_64285

theorem cos_sum_special_angles (α β : Real) :
  0 < α ∧ α < π/2 →
  -π/2 < β ∧ β < 0 →
  Real.cos (π/4 + α) = 1/3 →
  Real.cos (π/4 - β) = Real.sqrt 3 / 3 →
  Real.cos (α + β) = 5 * Real.sqrt 3 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sum_special_angles_l642_64285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l642_64218

def f (x : ℕ) : ℝ := 1 - 2^x

def S (n : ℕ) : ℝ := 1 - 2^n

def a : ℕ → ℝ
| 0 => 0  -- Add case for 0
| 1 => -1
| (n+1) => S (n+1) - S n

theorem sequence_formula (n : ℕ) (h : n > 0) : 
  (∀ k, f k = S k) → a n = -2^(n-1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l642_64218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_two_balloons_l642_64258

-- Define the type for balloons
inductive Balloon : Type
| A | B | C | D | E | F | G | H | I | J | K | L

-- Define the circle of balloons
def balloonCircle : List Balloon :=
  [Balloon.A, Balloon.B, Balloon.C, Balloon.D, Balloon.E, Balloon.F,
   Balloon.G, Balloon.H, Balloon.I, Balloon.J, Balloon.K, Balloon.L]

-- Function to pop every third balloon
def popEveryThird (balloons : List Balloon) : List Balloon :=
  sorry

-- Theorem stating the final result
theorem final_two_balloons :
  popEveryThird balloonCircle = [Balloon.E, Balloon.J] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_two_balloons_l642_64258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_calculation_l642_64228

/-- Calculates the time (in seconds) for a train to cross a platform -/
noncomputable def train_crossing_time (train_speed_kmh : ℝ) (train_length_m : ℝ) (platform_length_m : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * (5/18)  -- Convert km/h to m/s
  let total_distance := train_length_m + platform_length_m
  total_distance / train_speed_ms

theorem train_crossing_time_calculation :
  train_crossing_time 72 250 270 = 26 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_calculation_l642_64228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rebate_percentage_calculation_l642_64284

theorem rebate_percentage_calculation 
  (initial_cost : ℝ) 
  (sales_tax_rate : ℝ) 
  (final_amount : ℝ) 
  (rebate_percentage : ℝ) : 
  initial_cost = 6650 →
  sales_tax_rate = 0.10 →
  final_amount = 6876.1 →
  ((initial_cost - initial_cost * rebate_percentage) * (1 + sales_tax_rate) = final_amount) →
  rebate_percentage = 0.06 := by
  intros h1 h2 h3 h4
  -- The proof goes here
  sorry

#check rebate_percentage_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rebate_percentage_calculation_l642_64284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_division_l642_64213

-- Define a structure for a kid
structure Kid where
  portion : ℚ
  large_piece : ℚ
  small_piece : ℚ

theorem apple_division (total_apples : ℚ) (num_kids : ℚ) 
  (h1 : total_apples = 5)
  (h2 : num_kids = 6)
  (h3 : ∀ k : Kid, k.portion = k.large_piece + k.small_piece)
  (h4 : ∀ k1 k2 : Kid, k1.portion = k2.portion) :
  (∀ k : Kid, k.large_piece = 2/3 ∧ k.small_piece = 1/6) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_division_l642_64213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l642_64293

noncomputable section

/-- Definition of the ellipse C -/
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

/-- Eccentricity of the ellipse -/
def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - b^2 / a^2)

/-- Length of the chord passing through the left focus and perpendicular to the major axis -/
def chord_length (a b : ℝ) : ℝ :=
  2 * b^2 / a

/-- Slope of line l -/
def line_slope : ℝ := 4 / 5

/-- Theorem stating the properties of the ellipse and the constant sum of squared distances -/
theorem ellipse_properties (a b : ℝ) (h1 : a > b) (h2 : b > 0)
  (h3 : eccentricity a b = 3 / 5) (h4 : chord_length a b = 32 / 5) :
  a = 5 ∧ b = 4 ∧
  ∀ m : ℝ, ∀ A B : ℝ × ℝ,
    ellipse a b A.1 A.2 →
    ellipse a b B.1 B.2 →
    (∃ k : ℝ, A.2 = line_slope * (A.1 - m) ∧ B.2 = line_slope * (B.1 - m)) →
    (A.1 - m)^2 + A.2^2 + (B.1 - m)^2 + B.2^2 = 41 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l642_64293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_formula_holds_for_given_pairs_l642_64261

theorem formula_holds_for_given_pairs : ∀ (x y : ℤ), 
  ((x = 1 ∧ y = -1) ∨ 
   (x = 2 ∧ y = 3) ∨ 
   (x = 3 ∧ y = 9) ∨ 
   (x = 4 ∧ y = 17) ∨ 
   (x = 5 ∧ y = 27)) → 
  y = x^2 + x - 3 := by
  intro x y h
  cases h with
  | inl h1 => 
    cases h1 with
    | intro hx hy => 
      rw [hx, hy]
      norm_num
  | inr h2 => 
    cases h2 with
    | inl h2 =>
      cases h2 with
      | intro hx hy =>
        rw [hx, hy]
        norm_num
    | inr h3 =>
      cases h3 with
      | inl h3 =>
        cases h3 with
        | intro hx hy =>
          rw [hx, hy]
          norm_num
      | inr h4 =>
        cases h4 with
        | inl h4 =>
          cases h4 with
          | intro hx hy =>
            rw [hx, hy]
            norm_num
        | inr h5 =>
          cases h5 with
          | intro hx hy =>
            rw [hx, hy]
            norm_num

#check formula_holds_for_given_pairs

end NUMINAMATH_CALUDE_ERRORFEEDBACK_formula_holds_for_given_pairs_l642_64261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_meeting_distances_l642_64232

-- Define the problem parameters
noncomputable def distance_between_stations : ℝ := 550
noncomputable def speed_A : ℝ := 70
noncomputable def speed_B : ℝ := 40
noncomputable def speed_C : ℝ := 50
noncomputable def stop_time_A : ℝ := 1/3  -- 20 minutes in hours

-- Theorem statement
theorem train_meeting_distances :
  let relative_speed := speed_A + speed_B
  let meeting_time := distance_between_stations / relative_speed
  let distance_A := speed_A * meeting_time
  let distance_C_during_stop := speed_C * stop_time_A
  let distance_C := distance_A + distance_C_during_stop
  (distance_A = 350 ∧ Int.floor distance_C = 367) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_meeting_distances_l642_64232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_sides_line_constant_range_l642_64299

/-- Given two points on opposite sides of a line, prove the range of the constant in the line equation -/
theorem opposite_sides_line_constant_range :
  let p1 : ℝ × ℝ := (3, 1)
  let p2 : ℝ × ℝ := (-4, 6)
  let line (x y : ℝ) (a : ℝ) := 3 * x - 2 * y + a
  ∀ a : ℝ, (line p1.1 p1.2 a) * (line p2.1 p2.2 a) < 0 → -7 < a ∧ a < 24 :=
by
  intro a h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_sides_line_constant_range_l642_64299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_t_in_radicals_l642_64212

theorem solve_t_in_radicals (t : ℝ) : 
  t = 1 / (1 - Real.rpow 4 (1/3)) → 
  t = -1/3 * (1 + Real.rpow 4 (1/3) + (Real.rpow 4 (1/3))^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_t_in_radicals_l642_64212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_identity_and_sum_l642_64205

theorem binomial_identity_and_sum (n k m : ℕ) : 
  (Nat.choose n k + Nat.choose n (k-1) = Nat.choose (n+1) k) ∧ 
  (Finset.sum (Finset.range 11) (λ i => Nat.choose (n-i) m) = Nat.choose (n+1) (m+1) - Nat.choose (n-10) (m+1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_identity_and_sum_l642_64205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_implies_a_l642_64283

/-- Represents an ellipse with semi-major axis a and semi-minor axis √5 -/
structure Ellipse where
  a : ℝ
  h_pos : a > 0

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := 
  Real.sqrt (e.a^2 - 5) / e.a

/-- Theorem: For an ellipse with semi-major axis a and semi-minor axis √5,
    if the eccentricity is 2/3, then a = 3 -/
theorem ellipse_eccentricity_implies_a (e : Ellipse) 
    (h_ecc : eccentricity e = 2/3) : e.a = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_implies_a_l642_64283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_line_l642_64271

/-- A line in 2D space represented by the equation ax + by + c = 0 --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Distance from a point to a line --/
noncomputable def distancePointToLine (x₀ y₀ : ℝ) (l : Line) : ℝ :=
  (abs (l.a * x₀ + l.b * y₀ + l.c)) / Real.sqrt (l.a^2 + l.b^2)

/-- Distance from the origin to a line --/
noncomputable def distanceOriginToLine (l : Line) : ℝ :=
  (abs l.c) / Real.sqrt (l.a^2 + l.b^2)

/-- Checks if a point lies on a line --/
def pointOnLine (x y : ℝ) (l : Line) : Prop :=
  l.a * x + l.b * y + l.c = 0

theorem max_distance_line :
  ∀ (l : Line),
    pointOnLine 1 2 l →
    distanceOriginToLine l ≤ distanceOriginToLine ⟨1, 3, -7⟩ := by
  sorry

#check max_distance_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_line_l642_64271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_root_abs_value_implies_power_l642_64247

theorem square_root_abs_value_implies_power (x y : ℝ) :
  Real.sqrt (x - 3) + abs (y - 2) = 0 →
  (y - x)^2023 = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_root_abs_value_implies_power_l642_64247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_of_2025th_local_min_l642_64262

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := x + 2 * Real.sin x

-- Define the property of being a local minimum point
def is_local_min (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∃ ε > 0, ∀ y, |y - x| < ε → f y ≥ f x

-- Define the sequence of local minimum points
noncomputable def local_min_seq (f : ℝ → ℝ) : ℕ → ℝ
| 0 => Real.pi * 4 / 3  -- First local minimum
| n + 1 => local_min_seq f n + 2 * Real.pi

-- State the theorem
theorem sin_of_2025th_local_min :
  Real.sin (local_min_seq f 2025) = -Real.sqrt 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_of_2025th_local_min_l642_64262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_cot_45_simplification_l642_64240

theorem tan_cot_45_simplification :
  (Real.tan (45 * π / 180))^3 + (1 / Real.tan (45 * π / 180))^3 =
  Real.tan (45 * π / 180) + (1 / Real.tan (45 * π / 180)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_cot_45_simplification_l642_64240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extremum_and_increasing_l642_64221

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (Real.sin x + a) / Real.exp x

theorem f_extremum_and_increasing (a : ℝ) :
  (∃ x : ℝ, x = 0 ∧ HasDerivAt (f a) 0 x) →
  a = 1 ∧
  (∀ x : ℝ, HasDerivAt (f a) (Real.cos x - Real.sin x - a) x) →
  a ≤ -Real.sqrt 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extremum_and_increasing_l642_64221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l642_64272

noncomputable def f (x : ℝ) : ℝ := (x^2 + 5*x + 6) / (x + 2)

theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, x ≠ -2 ∧ f x = y) ↔ y ∈ Set.Iio 1 ∪ Set.Ioi 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l642_64272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_parabola_line_l642_64269

/-- The parabola function -/
noncomputable def parabola (x : ℝ) : ℝ := x^2 - 4*x + 8

/-- The line function -/
noncomputable def line (x : ℝ) : ℝ := 2*x - 3

/-- Distance function from a point to a line -/
noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  |2*x - y + 3| / Real.sqrt 5

/-- The shortest distance between the parabola and the line -/
theorem shortest_distance_parabola_line :
  ∃ (x : ℝ), ∀ (t : ℝ), 
    distance_to_line x (parabola x) ≤ distance_to_line t (parabola t) ∧
    distance_to_line x (parabola x) = (4 * Real.sqrt 5) / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_parabola_line_l642_64269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_evaluate_expression_power_256_power_64_l642_64215

theorem evaluate_expression : 
  (256 : ℝ) ^ (16 / 100) * (256 : ℝ) ^ (9 / 100) * (64 : ℝ) ^ (1 / 4) = 8 := by
  sorry

-- Definitions based on conditions
theorem power_256 : (256 : ℝ) = 4 ^ 4 := by sorry
theorem power_64 : (64 : ℝ) = 4 ^ 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_evaluate_expression_power_256_power_64_l642_64215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_iff_m_eq_neg_one_l642_64289

/-- A power function parameterized by m -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m^2 - m - 1) * x^(m^2 + 2*m - 3)

/-- The theorem stating that f is decreasing on (0, +∞) if and only if m = -1 -/
theorem f_decreasing_iff_m_eq_neg_one :
  ∀ m : ℝ, (∀ x y : ℝ, 0 < x → x < y → f m y < f m x) ↔ m = -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_iff_m_eq_neg_one_l642_64289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_semicircle_radius_l642_64268

theorem inscribed_semicircle_radius (R : ℝ) (r : ℝ) : 
  R = 2 → (R - r)^2 + r^2 = R^2 → r = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_semicircle_radius_l642_64268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l642_64245

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := 1 / (3^x - 1) + 1/3

-- Theorem statement
theorem g_is_odd : ∀ x : ℝ, g (-x) = -g x := by
  intro x
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l642_64245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partner_a_profit_l642_64237

/-- Calculates the money received by partner A in a business partnership --/
theorem partner_a_profit (a_investment b_investment total_profit : ℚ) : 
  a_investment = 2000 →
  b_investment = 3000 →
  total_profit = 9600 →
  (10 : ℚ) / 100 * total_profit + 
  (a_investment / (a_investment + b_investment)) * 
  (total_profit - (10 : ℚ) / 100 * total_profit) = 4416 := by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_partner_a_profit_l642_64237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_perimeter_product_l642_64209

/-- A point on a 2D grid -/
structure Point where
  x : ℝ
  y : ℝ

/-- A square defined by four points -/
structure Square where
  e : Point
  f : Point
  g : Point
  h : Point

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Calculate the side length of a square -/
noncomputable def sideLength (s : Square) : ℝ :=
  distance s.e s.h

/-- Calculate the area of a square -/
noncomputable def area (s : Square) : ℝ :=
  (sideLength s)^2

/-- Calculate the perimeter of a square -/
noncomputable def perimeter (s : Square) : ℝ :=
  4 * sideLength s

/-- The main theorem -/
theorem square_area_perimeter_product (s : Square) :
  s.e = ⟨4, 5⟩ ∧ s.f = ⟨6, 2⟩ ∧ s.g = ⟨2, 0⟩ ∧ s.h = ⟨0, 3⟩ →
  area s * perimeter s = 160 * Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_perimeter_product_l642_64209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l642_64256

noncomputable def f (x : ℝ) := Real.sqrt (-Real.sin x) + Real.sqrt (25 - x^2)

theorem domain_of_f :
  ∀ x : ℝ, x ∈ Set.Ici (-Real.pi) ∩ Set.Iic 0 ∪ Set.Ici Real.pi ∩ Set.Iic 5 ↔ 
    (-Real.sin x ≥ 0) ∧ (25 - x^2 ≥ 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l642_64256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_surface_area_ratio_l642_64231

-- Define the cylinder's dimensions
def cylinder_height : ℝ := 6
def cylinder_diameter : ℝ := 6

-- Define the surface area of a cylinder
noncomputable def cylinder_surface_area (r h : ℝ) : ℝ := 2 * Real.pi * r * (r + h)

-- Define the surface area of a sphere
noncomputable def sphere_surface_area (r : ℝ) : ℝ := 4 * Real.pi * r^2

-- State the theorem
theorem sphere_surface_area_ratio :
  let r1 := Real.sqrt ((cylinder_surface_area (cylinder_diameter / 2) cylinder_height) / (4 * Real.pi))
  let r2 := 2 * r1
  (sphere_surface_area r1) / (sphere_surface_area r2) = 1 / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_surface_area_ratio_l642_64231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_inequality_l642_64214

theorem cosine_inequality : 
  Real.cos 6 > Real.cos 8 ∧ Real.cos 8 > Real.cos 2 ∧ Real.cos 2 > Real.cos 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_inequality_l642_64214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_students_wearing_both_l642_64288

theorem min_students_wearing_both (n : ℕ) (sunglasses caps both : ℕ) : 
  n > 0 ∧ 
  sunglasses = (3 * n) / 7 ∧ 
  caps = (4 * n) / 5 ∧ 
  both = sunglasses + caps - n → 
  both ≥ 8 := by
  sorry

#check min_students_wearing_both

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_students_wearing_both_l642_64288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inlet_rate_is_eight_litres_per_minute_l642_64297

/-- The rate at which the inlet pipe lets liquid into a tank. -/
noncomputable def inlet_rate (tank_capacity : ℝ) (outlet_time : ℝ) (combined_time : ℝ) : ℝ :=
  (tank_capacity / outlet_time - tank_capacity / combined_time) * (1 / 60)

/-- Theorem stating that under given conditions, the inlet rate is 8 litres per minute. -/
theorem inlet_rate_is_eight_litres_per_minute :
  inlet_rate 12800 10 16 = 8 := by
  -- Unfold the definition of inlet_rate
  unfold inlet_rate
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inlet_rate_is_eight_litres_per_minute_l642_64297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l642_64296

theorem simplify_expression : 
  (Real.rpow 64 (1/6) - Real.sqrt (8 + 1/2))^2 = (50 - 8 * Real.sqrt 34) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l642_64296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l642_64252

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the vectors
noncomputable def m (t : Triangle) : Real × Real := (t.a, t.b)
noncomputable def n (t : Triangle) : Real × Real := (Real.sin t.B, Real.sqrt 3 * Real.cos t.A)

-- Define perpendicularity
def perpendicular (v w : Real × Real) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

-- State the theorem
theorem triangle_problem (t : Triangle) 
  (h1 : perpendicular (m t) (n t))  -- m is perpendicular to n
  (h2 : t.a = Real.sqrt 7)  -- a = √7
  (h3 : (1/2) * t.b * t.c * Real.sin t.A = Real.sqrt 3 / 2)  -- Area of triangle
  : t.A = 2 * Real.pi / 3 ∧ t.a + t.b + t.c = Real.sqrt 7 + 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l642_64252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tip_is_twenty_percent_l642_64286

/-- Represents the cost of different ride options and the taxi ride details. -/
structure RideCosts where
  uber : ℚ
  lyft : ℚ
  taxi : ℚ
  total_taxi : ℚ

/-- Calculates the tip percentage given the ride costs. -/
def tip_percentage (costs : RideCosts) : ℚ :=
  ((costs.total_taxi - costs.taxi) / costs.taxi) * 100

/-- Theorem stating that the tip percentage is 20% given the problem conditions. -/
theorem tip_is_twenty_percent (costs : RideCosts) 
  (h1 : costs.uber = costs.lyft + 3)
  (h2 : costs.lyft = costs.taxi + 4)
  (h3 : costs.uber = 22)
  (h4 : costs.total_taxi = 18) :
  tip_percentage costs = 20 := by
  sorry

#eval tip_percentage { uber := 22, lyft := 19, taxi := 15, total_taxi := 18 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tip_is_twenty_percent_l642_64286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_preimage_of_nine_l642_64279

-- Define the mapping f: ℝ → ℝ
def f (x : ℝ) : ℝ := 2 * x - 3

-- Theorem: The preimage of 9 under f is 6
theorem preimage_of_nine : ∃ (x : ℝ), f x = 9 ∧ x = 6 :=
by
  -- We use 6 as our witness for x
  use 6
  constructor
  · -- Prove f 6 = 9
    simp [f]
    norm_num
  · -- Prove 6 = 6
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_preimage_of_nine_l642_64279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_sequence_count_l642_64239

theorem number_sequence_count (s : List ℝ) 
  (h1 : s.sum / s.length = 104)
  (h2 : (s.take 5).sum / 5 = 99)
  (h3 : (s.reverse.take 5).sum / 5 = 100)
  (h4 : s.length > 4)
  (h5 : s[4]! = 59) : 
  s.length = 9 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_sequence_count_l642_64239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_length_implies_k_value_l642_64234

/-- The line on which P(x,y) lies -/
def line (k : ℝ) (x y : ℝ) : Prop := k * x + y + 4 = 0

/-- The circle C -/
def circleC (x y : ℝ) : Prop := x^2 + y^2 - 2*y = 0

/-- The point P -/
structure Point (k : ℝ) where
  x : ℝ
  y : ℝ
  on_line : line k x y

/-- The point A (point of tangency) -/
structure TangentPoint where
  x : ℝ
  y : ℝ
  on_circle : circleC x y

theorem tangent_length_implies_k_value (k : ℝ) (P : Point k) (A : TangentPoint) 
  (h1 : k > 0)
  (h2 : ∃ (min_length : ℝ), min_length = 2 ∧ 
    ∀ (P' : Point k) (A' : TangentPoint), 
      Real.sqrt ((P'.x - A'.x)^2 + (P'.y - A'.y)^2) ≥ min_length) :
  k = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_length_implies_k_value_l642_64234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l642_64217

noncomputable def m (a θ : ℝ) : ℝ × ℝ := (a - Real.sin θ, -1/2)
noncomputable def n (θ : ℝ) : ℝ × ℝ := (1/2, Real.cos θ)

def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v = (k * w.1, k * w.2)

theorem vector_problem :
  (∀ θ : ℝ, perpendicular (m (Real.sqrt 2 / 2) θ) (n θ) → Real.sin (2 * θ) = -1/2) ∧
  (∀ θ : ℝ, parallel (m 0 θ) (n θ) → Real.tan θ = 2 + Real.sqrt 3 ∨ Real.tan θ = 2 - Real.sqrt 3) := by
  sorry

#check vector_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l642_64217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_to_fill_pool_l642_64298

-- Define the water sources
noncomputable def hose_flow_rate : ℝ := 100
noncomputable def hose_cost_rate : ℝ := 1 / 10
noncomputable def pump_flow_rate : ℝ := 150
noncomputable def pump_cost_rate : ℝ := 1 / 8

-- Define the filling times
noncomputable def day1_hours : ℝ := 30
noncomputable def day2_hours : ℝ := 20

-- Define the evaporation rate
noncomputable def evaporation_rate : ℝ := 0.02

-- Define the total flow rate
noncomputable def total_flow_rate : ℝ := hose_flow_rate + pump_flow_rate

-- Function to calculate water added in a day
noncomputable def water_added (hours : ℝ) : ℝ := hours * total_flow_rate

-- Function to calculate water remaining after evaporation
noncomputable def water_after_evaporation (initial_water : ℝ) : ℝ :=
  initial_water * (1 - evaporation_rate)

-- Function to calculate cost for a water source
noncomputable def cost_for_source (gallons : ℝ) (cost_rate : ℝ) : ℝ :=
  gallons * cost_rate

-- Theorem statement
theorem total_cost_to_fill_pool :
  let day1_water := water_added day1_hours
  let day1_remaining := water_after_evaporation day1_water
  let day2_water := water_added day2_hours
  let total_water := day1_remaining + day2_water
  let final_water := water_after_evaporation total_water
  let hose_cost := cost_for_source (day1_water + day2_water) hose_cost_rate
  let pump_cost := cost_for_source (day1_water + day2_water) pump_cost_rate
  hose_cost + pump_cost = 28.125 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_to_fill_pool_l642_64298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_divisor_of_least_number_l642_64282

theorem smallest_divisor_of_least_number (n : ℕ) : 
  n = 856 ∧ 
  (n + 8) % 32 = 0 ∧ 
  (n + 8) % 36 = 0 ∧ 
  (n + 8) % 54 = 0 → 
  (∀ d : ℕ, d ∈ ({32, 36, 54} : Set ℕ) → (n + 8) % d = 0) ∧
  32 ≤ 36 ∧
  32 ≤ 54 ∧
  (n + 8) % 32 = 0 :=
by
  intro h
  constructor
  · intro d hd
    simp at hd
    cases hd with
    | inl hd32 => rw [hd32]; exact h.2.1
    | inr hd =>
      cases hd with
      | inl hd36 => rw [hd36]; exact h.2.2.1
      | inr hd54 => rw [hd54]; exact h.2.2.2
  · constructor
    · exact Nat.le_of_lt (by norm_num)
    · constructor
      · exact Nat.le_of_lt (by norm_num)
      · exact h.2.1

#check smallest_divisor_of_least_number

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_divisor_of_least_number_l642_64282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossibility_of_x0_greater_than_c_l642_64276

noncomputable def f (x : ℝ) := Real.exp (-x) + Real.log x

theorem impossibility_of_x0_greater_than_c
  (a b c : ℝ)
  (h_pos_a : a > 0)
  (h_pos_b : b > 0)
  (h_pos_c : c > 0)
  (h_order : a < b ∧ b < c)
  (h_product : f a * f b * f c > 0)
  (x₀ : ℝ)
  (h_x₀_root : f x₀ = 0) :
  ¬(x₀ > c) := by
  sorry

#check impossibility_of_x0_greater_than_c

end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossibility_of_x0_greater_than_c_l642_64276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_of_quadrant_angle_l642_64229

theorem sine_of_quadrant_angle (α : ℝ) :
  (∃ (x y : ℝ), x = 3/5 ∧ y = 4/5 ∧ x^2 + y^2 = 1 ∧ y = Real.sin α) →
  Real.sin α = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_of_quadrant_angle_l642_64229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l642_64243

/-- Definition of a focus for a parabola -/
def is_focus (p : ℝ × ℝ) (parabola : ℝ × ℝ → Prop) : Prop := sorry

/-- Definition of a directrix for a parabola -/
def directrix (parabola : ℝ × ℝ → Prop) : Set (ℝ × ℝ) := sorry

/-- Given a parabola with equation x = -1/4 * y^2 + 2, its directrix is x = 3 -/
theorem parabola_directrix : 
  ∀ (x y : ℝ), x = -(1/4) * y^2 + 2 → 
    (∃ (p : ℝ × ℝ), is_focus p (λ q : ℝ × ℝ ↦ q.1 = -(1/4) * q.2^2 + 2) ∧ 
      directrix (λ q : ℝ × ℝ ↦ q.1 = -(1/4) * q.2^2 + 2) = {q : ℝ × ℝ | q.1 = 3}) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l642_64243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_intersect_N_equals_zero_one_two_l642_64263

-- Define set A
def A : Set ℝ := {x | (3 : ℝ)^x ≤ 10}

-- Define the theorem
theorem A_intersect_N_equals_zero_one_two : A ∩ Set.range (fun n : ℕ => (n : ℝ)) = {0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_intersect_N_equals_zero_one_two_l642_64263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_problem_l642_64220

open Real

theorem trig_problem (α : ℝ) (h1 : cos α = -4/5) (h2 : α ∈ Set.Ioo (π/2) π) :
  tan (2*α) = -24/7 ∧ cos (α + π/3) = (-4 - 3*Real.sqrt 3)/10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_problem_l642_64220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_ratio_sum_l642_64253

theorem lcm_ratio_sum (a b : ℕ) : 
  Nat.lcm a b = 48 → 
  a = 2 * (b / 3) → 
  a + b = 80 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_ratio_sum_l642_64253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_with_special_angle_relation_is_nonagon_l642_64291

/-- A convex polygon where an interior angle is 180° more than three times its exterior angle has 9 sides. -/
theorem polygon_with_special_angle_relation_is_nonagon (n : ℕ) 
  (h_convex : n ≥ 3)
  (h_angle_relation : ∃ (interior_angle exterior_angle : ℝ), 
    interior_angle = 3 * exterior_angle + 180 ∧ 
    interior_angle + exterior_angle = 180) :
  n = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_with_special_angle_relation_is_nonagon_l642_64291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_of_chips_is_three_l642_64222

/-- The cost of a single bag of chips given the following conditions:
  * Frank buys 5 chocolate bars and 2 bags of chips.
  * He pays $20 and receives $4 in change.
  * Each chocolate bar costs $2.
-/
def cost_of_chips (num_chocolate_bars : ℕ) (num_chip_bags : ℕ) 
  (cost_per_chocolate_bar : ℕ) (total_paid : ℕ) (change_received : ℕ) : ℕ :=
  let total_spent := total_paid - change_received
  let chocolate_cost := num_chocolate_bars * cost_per_chocolate_bar
  let chips_total_cost := total_spent - chocolate_cost
  chips_total_cost / num_chip_bags

theorem cost_of_chips_is_three :
  cost_of_chips 5 2 2 20 4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_of_chips_is_three_l642_64222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_true_statements_l642_64200

theorem four_true_statements : 
  (∀ m n : ℝ, (abs m > abs n → m^2 > n^2)) ∧ 
  (∀ m n : ℝ, (m^2 > n^2 → abs m > abs n)) ∧ 
  (∀ m n : ℝ, (abs m ≤ abs n → m^2 ≤ n^2)) ∧ 
  (∀ m n : ℝ, (m^2 ≤ n^2 → abs m ≤ abs n)) := by
  constructor
  · intro m n h
    sorry
  · constructor
    · intro m n h
      sorry
    · constructor
      · intro m n h
        sorry
      · intro m n h
        sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_true_statements_l642_64200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pokemon_card_friends_l642_64274

theorem pokemon_card_friends : 
  (432 : ℕ) / (9 * 12 : ℕ) = 4 := by
  -- Proof steps will go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pokemon_card_friends_l642_64274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_problem_l642_64211

theorem divisibility_problem (a : ℤ) (h1 : 0 ≤ a) (h2 : a ≤ 13) 
  (h3 : (51^2015 + a) % 13 = 0) : a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_problem_l642_64211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_solution_set_correct_l642_64225

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 1| + |2 * x + 4|

-- Theorem for the minimum value of f(x)
theorem min_value_f : 
  ∃ (m : ℝ), m = 3 ∧ ∀ x, f x ≥ m :=
sorry

-- Define the set of solutions for |f(x) - 6| ≤ 1
def solution_set : Set ℝ :=
  {x | x ∈ Set.Icc (-10/3) (-8/3) ∨ x ∈ Set.Icc 0 1 ∨ x ∈ Set.Ico 1 (4/3)}

-- Theorem for the solution set of |f(x) - 6| ≤ 1
theorem solution_set_correct :
  ∀ x, x ∈ solution_set ↔ |f x - 6| ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_solution_set_correct_l642_64225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_l642_64233

/-- The profit function for a company based on promotional cost x -/
noncomputable def profit (x : ℝ) : ℝ := 19 - 24 / (x + 2) - (3/2) * x

/-- The theorem stating the conditions for maximum profit -/
theorem max_profit (a : ℝ) (ha : 0 < a) :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ a → profit x ≤ profit (min 2 a)) ∧
  (a < 2 → ∀ x : ℝ, 0 ≤ x ∧ x < a → profit x < profit a) ∧
  (2 ≤ a → ∀ x : ℝ, 0 ≤ x ∧ x ≠ 2 ∧ x ≤ a → profit x < profit 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_l642_64233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decagon_symmetry_axes_l642_64219

/-- A regular polygon is a polygon with all sides and angles equal -/
structure RegularPolygon where
  sides : ℕ
  is_regular : Prop  -- Changed from True to Prop

/-- The number of axes of symmetry in a regular polygon -/
def axes_of_symmetry (p : RegularPolygon) : ℕ := p.sides

/-- Definition of a regular decagon -/
def regular_decagon : RegularPolygon :=
  { sides := 10,
    is_regular := True }  -- Now True is of type Prop, which is correct

/-- Theorem: A regular decagon has 10 axes of symmetry -/
theorem decagon_symmetry_axes : axes_of_symmetry regular_decagon = 10 := by
  -- Unfold the definition of axes_of_symmetry and regular_decagon
  unfold axes_of_symmetry regular_decagon
  -- The proof is now trivial as it reduces to 10 = 10
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_decagon_symmetry_axes_l642_64219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_coprime_sum_125_l642_64273

/-- A function that checks if a list of natural numbers are pairwise coprime -/
def isPairwiseCoprime (list : List Nat) : Prop :=
  ∀ i j, i ≠ j → i < list.length → j < list.length → 
    Nat.gcd (list.get ⟨i, by sorry⟩) (list.get ⟨j, by sorry⟩) = 1

/-- The theorem stating that the maximum number of pairwise coprime integers larger than 1 that sum to 125 is 10 -/
theorem max_coprime_sum_125 :
  ∀ (list : List Nat),
    list.sum = 125 →
    (∀ x ∈ list, x > 1) →
    isPairwiseCoprime list →
    list.length ≤ 10 :=
by sorry

#check max_coprime_sum_125

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_coprime_sum_125_l642_64273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_iron_heating_time_l642_64244

/-- The heating rate of the iron in degrees Celsius per second -/
noncomputable def heating_rate : ℝ := 9 / 20

/-- The cooling rate of the iron in degrees Celsius per second -/
noncomputable def cooling_rate : ℝ := 15 / 30

/-- The total cooling time in seconds -/
def total_cooling_time : ℝ := 3 * 60

/-- Theorem: The iron was turned on for 200 seconds -/
theorem iron_heating_time : ℝ := by
  -- Define the total temperature drop during cooling
  let total_temp_drop := cooling_rate * total_cooling_time
  
  -- Define the heating time required to achieve the total temperature drop
  let heating_time := total_temp_drop / heating_rate
  
  -- Prove that the heating time is equal to 200 seconds
  sorry

-- Remove the #eval statement as it's not necessary for the proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_iron_heating_time_l642_64244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_even_digit_sum_l642_64264

/-- The set of digits used to form the numbers -/
def digits : Finset Nat := {1, 2, 3, 4, 5, 6}

/-- A three-digit number without repetition -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  units : Nat
  hundreds_in_digits : hundreds ∈ digits
  tens_in_digits : tens ∈ digits
  units_in_digits : units ∈ digits
  all_different : hundreds ≠ tens ∧ hundreds ≠ units ∧ tens ≠ units

/-- The sum of digits of a three-digit number -/
def digitSum (n : ThreeDigitNumber) : Nat :=
  n.hundreds + n.tens + n.units

/-- Predicate for numbers with even digit sum -/
def hasEvenDigitSum (n : ThreeDigitNumber) : Prop :=
  digitSum n % 2 = 0

instance : DecidablePred hasEvenDigitSum := fun n =>
  decEq (digitSum n % 2) 0

/-- The set of all valid three-digit numbers -/
def allThreeDigitNumbers : Finset ThreeDigitNumber :=
  sorry

theorem count_even_digit_sum :
  (allThreeDigitNumbers.filter hasEvenDigitSum).card = 60 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_even_digit_sum_l642_64264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sudoku_competition_l642_64203

/-- Probability of A solving the puzzle correctly in each round -/
noncomputable def prob_A : ℝ := 0.8

/-- Probability of B solving the puzzle correctly in each round -/
noncomputable def prob_B : ℝ := 0.75

/-- Probability that B scores exactly 1 point after two rounds -/
noncomputable def prob_B_scores_1 : ℝ := 3/8

/-- Probability that A wins without a tiebreaker -/
noncomputable def prob_A_wins : ℝ := 3/10

theorem sudoku_competition (hA : 0 ≤ prob_A ∧ prob_A ≤ 1) (hB : 0 ≤ prob_B ∧ prob_B ≤ 1) :
  prob_B_scores_1 = 3/8 ∧ prob_A_wins = 3/10 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sudoku_competition_l642_64203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_statements_correct_l642_64227

def f (x : ℝ) := x^3 - 3*x^2

def statement1 : Prop := ∀ x y, x < y → f x < f y ∧ ∀ x, ¬(∀ ε > 0, ∃ δ > 0, ∀ y, |y - x| < δ → f y ≤ f x)

def statement2 : Prop := ∀ x y, x < y → f x > f y ∧ ∀ x, ¬(∀ ε > 0, ∃ δ > 0, ∀ y, |y - x| < δ → f y ≤ f x)

def statement3 : Prop := (∀ x y, x < y ∧ x < 0 ∧ y < 0 → f x < f y) ∧
                         (∀ x y, 2 < x ∧ x < y → f x < f y) ∧
                         (∀ x y, 0 < x ∧ x < y ∧ y < 2 → f x > f y)

def statement4 : Prop := f 0 = 0 ∧ f 2 = -4 ∧
                         (∀ x, f x ≤ f 0) ∧
                         (∀ x, f x ≥ f 2)

theorem two_statements_correct :
  (statement3 ∧ statement4) ∧ ¬statement1 ∧ ¬statement2 := by
  sorry

#check two_statements_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_statements_correct_l642_64227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_and_g_min_l642_64294

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * (3^x) + 1 / (3^(x-1))

noncomputable def g (m : ℝ) (x : ℝ) : ℝ := 9^x + 9^(-x) + m * f 3 x + m^2 - 1

theorem f_even_and_g_min (a : ℝ) :
  (∀ x, f a x = f a (-x)) →
  (a = 3) ∧
  (∀ m x, 
    (m < -4/3 → g m x ≥ -5/4 * m^2 - 3) ∧
    (m ≥ -4/3 → g m x ≥ m^2 + 6*m + 1)) := by
  sorry

#check f_even_and_g_min

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_and_g_min_l642_64294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bela_wins_iff_n_odd_l642_64201

/-- The game interval --/
def GameInterval (n : ℕ) : Set ℝ := Set.Icc 0 n

/-- A valid move in the game --/
def ValidMove (n : ℕ) (choices : Set ℝ) (x : ℝ) : Prop :=
  x ∈ GameInterval n ∧ ∀ y ∈ choices, |x - y| > 2

/-- The game ends when no valid moves are left --/
def GameOver (n : ℕ) (choices : Set ℝ) : Prop :=
  ∀ x, ¬ValidMove n choices x

/-- Bela wins if and only if n is odd --/
theorem bela_wins_iff_n_odd (n : ℕ) (h : n > 6) :
  (∃ strategy : ℕ → Set ℝ → ℝ,
    (∀ k choices, ValidMove n choices (strategy k choices)) ∧
    GameOver n (Set.range (λ k ↦ strategy k (Set.range (λ i ↦ strategy i ∅)))))
  ↔ n % 2 = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bela_wins_iff_n_odd_l642_64201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_approximation_l642_64257

/-- Represents the principal amount in Rupees -/
noncomputable def principal : ℝ := 16000

/-- The annual interest rate as a decimal -/
noncomputable def rate : ℝ := 0.20

/-- Number of times interest is compounded per year -/
noncomputable def compoundingFrequency : ℝ := 4

/-- Time period in years -/
noncomputable def time : ℝ := 9 / 12

/-- The compound interest amount in Rupees -/
noncomputable def compoundInterest : ℝ := 2522.0000000000036

/-- Theorem stating that the given principal approximates the correct amount 
    given the compound interest and other conditions -/
theorem principal_approximation :
  abs (compoundInterest - (principal * ((1 + rate / compoundingFrequency) ^ (compoundingFrequency * time) - 1))) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_approximation_l642_64257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_proportional_to_volume_l642_64202

/-- Represents a cylindrical container with height, diameter, and price. -/
structure Container where
  height : ℝ
  diameter : ℝ
  price : ℝ

/-- Calculates the volume of a cylindrical container. -/
noncomputable def volume (c : Container) : ℝ :=
  Real.pi * (c.diameter / 2)^2 * c.height

/-- The price per unit volume of a container. -/
noncomputable def pricePerUnitVolume (c : Container) : ℝ :=
  c.price / volume c

theorem price_proportional_to_volume 
  (c1 c2 : Container) 
  (h_volume : volume c2 = 8 * volume c1) 
  (h_price1 : c1.price = 0.80) 
  (h_same_rate : pricePerUnitVolume c1 = pricePerUnitVolume c2) : 
  c2.price = 6.40 := by
  sorry

#check price_proportional_to_volume

end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_proportional_to_volume_l642_64202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_exponent_l642_64275

theorem find_exponent (exponent : ℝ) : (12 : ℝ) * 6^exponent / 432 = 36 → exponent = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_exponent_l642_64275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l642_64287

-- Define the line on which point M lies
def line_eq (x y : ℝ) : Prop := 2 * x + y - 1 = 0

-- Define the circle passing through (3,0) and (0,1)
def circle_eq (cx cy r : ℝ) : Prop :=
  (3 - cx)^2 + cy^2 = r^2 ∧ cx^2 + (1 - cy)^2 = r^2

-- Theorem statement
theorem circle_equation :
  ∃ (cx cy : ℝ), line_eq cx cy ∧ ∃ (r : ℝ), circle_eq cx cy r ∧
  ∀ (x y : ℝ), (x - cx)^2 + (y - cy)^2 = r^2 ↔ (x - 1)^2 + (y + 1)^2 = 5 :=
by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l642_64287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_one_statement_correct_l642_64208

/-- Definition of an infinite decimal number -/
noncomputable def InfiniteDecimal (x : ℝ) : Prop := ∀ n : ℕ, ∃ d : ℕ, d < 10 ∧ (x * 10^n - ⌊x * 10^n⌋) * 10 = d

/-- Definition of a rational number -/
def IsRational (x : ℝ) : Prop := ∃ (p q : ℤ), q ≠ 0 ∧ x = p / q

/-- Definition of an irrational number -/
def IsIrrational (x : ℝ) : Prop := ¬IsRational x

/-- Statement 1: All infinite decimal numbers are irrational -/
def Statement1 : Prop := ∀ x : ℝ, InfiniteDecimal x → IsIrrational x

/-- Statement 2: If A has a rational square root, then A must have an irrational cube root -/
def Statement2 : Prop := ∀ A : ℝ, (∃ r : ℚ, r^2 = A) → IsIrrational (A^(1/3 : ℝ))

/-- Statement 3: For an irrational number a, there exists a positive integer n such that a^n is rational -/
def Statement3 : Prop := ∀ a : ℝ, IsIrrational a → ∃ n : ℕ+, IsRational (a^(n : ℝ))

/-- Statement 4: The reciprocal and the opposite of an irrational number are both irrational -/
def Statement4 : Prop := ∀ a : ℝ, IsIrrational a → IsIrrational (1/a) ∧ IsIrrational (-a)

/-- Theorem: Only one of the above statements is correct -/
theorem only_one_statement_correct : 
  (Statement1 = False) ∧ 
  (Statement2 = False) ∧ 
  (Statement3 = False) ∧ 
  (Statement4 = True) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_one_statement_correct_l642_64208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_parabola_intersection_l642_64249

-- Define the hyperbola and parabola
def hyperbola (a b x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

-- Define the focus F and intersection point P
def F : ℝ × ℝ := (1, 0)
noncomputable def P : ℝ × ℝ := (3/2, Real.sqrt 6)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Main theorem
theorem hyperbola_parabola_intersection
  (a b : ℝ)
  (ha : a > 0)
  (hb : b > 0)
  (h_common_focus : F = (1, 0))
  (h_intersection : hyperbola a b P.1 P.2 ∧ parabola P.1 P.2)
  (h_distance : distance P F = 5/2) :
  (P = (3/2, Real.sqrt 6) ∨ P = (3/2, -Real.sqrt 6)) ∧
  (∀ (x y : ℝ), Real.sqrt 3 * x + y = 0 ∨ Real.sqrt 3 * x - y = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_parabola_intersection_l642_64249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_product_equivalence_l642_64255

theorem cosine_sum_product_equivalence :
  ∃ (a b c d : ℕ+), 
    (∀ x : ℝ, (Real.cos x) + (Real.cos (3*x)) + (Real.cos (7*x)) + (Real.cos (9*x)) = 
      (a : ℝ) * (Real.cos (b*x)) * (Real.cos (c*x)) * (Real.cos (d*x))) ∧
    a + b + c + d = 13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_product_equivalence_l642_64255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l642_64226

theorem problem_solution (x y : ℝ) (h1 : x + y = x * y) (h2 : x > 0) (h3 : y > 0) :
  (x + 2 * y ≥ 3 + 2 * Real.sqrt 2) ∧
  ((2 : ℝ)^x + (2 : ℝ)^y ≥ 8) ∧
  (1 / Real.sqrt x + 1 / Real.sqrt y ≤ Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l642_64226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_and_derivative_diff_l642_64281

noncomputable def f (a b x : ℝ) : ℝ := a * Real.sin x + b * x^3 + 4

theorem f_sum_and_derivative_diff (a b : ℝ) : 
  f a b 2016 + f a b (-2016) + (deriv (f a b)) 2017 - (deriv (f a b)) (-2017) = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_and_derivative_diff_l642_64281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_translation_on_cosine_graphs_l642_64260

theorem point_translation_on_cosine_graphs :
  ∀ (t m : ℝ),
  t = Real.cos (2 * (π/4) + π/6) →
  t = Real.cos (2 * (π/4 + m)) →
  m > 0 →
  t = -1/2 ∧
  (∀ m' > 0, (∀ t', t' = Real.cos (2 * (π/4) + π/6) ∧ t' = Real.cos (2 * (π/4 + m')) → m' ≥ π/12)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_translation_on_cosine_graphs_l642_64260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_space_mission_contribution_l642_64267

/-- The cost of the space mission in billions of dollars -/
noncomputable def mission_cost : ℚ := 30

/-- The combined population of the U.S. and Canada in millions -/
noncomputable def population : ℚ := 350

/-- The per-person contribution in dollars -/
noncomputable def contribution : ℚ := (mission_cost * 1000) / population

theorem space_mission_contribution :
  ⌊(contribution : ℝ)⌋ = 86 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_space_mission_contribution_l642_64267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_cost_is_140_l642_64265

/-- Represents a rectangular park with given properties -/
structure RectangularPark where
  ratio_length : ℕ := 3
  ratio_width : ℕ := 2
  area : ℝ := 4704
  fencing_cost_paise : ℝ := 50

/-- Calculates the cost of fencing a rectangular park -/
noncomputable def fencing_cost (park : RectangularPark) : ℝ :=
  let x : ℝ := Real.sqrt (park.area / (park.ratio_length * park.ratio_width))
  let length : ℝ := park.ratio_length * x
  let width : ℝ := park.ratio_width * x
  let perimeter : ℝ := 2 * (length + width)
  (perimeter * park.fencing_cost_paise) / 100

theorem fencing_cost_is_140 (park : RectangularPark) :
  fencing_cost park = 140 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_cost_is_140_l642_64265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_surface_area_formula_l642_64216

/-- The surface area of a pyramid with square base of side length a -/
noncomputable def pyramid_surface_area (a : ℝ) : ℝ := 2 * Real.sqrt 3 * a^2

/-- Theorem: The surface area of a pyramid with square base of side length a is 2√3a² -/
theorem pyramid_surface_area_formula (a : ℝ) (h : a > 0) :
  pyramid_surface_area a = 2 * Real.sqrt 3 * a^2 :=
by
  -- Unfold the definition of pyramid_surface_area
  unfold pyramid_surface_area
  -- The equality holds by reflexivity
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_surface_area_formula_l642_64216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_digit_is_zero_l642_64223

def modifiedFibonacci : ℕ → ℕ
  | 0 => 3
  | 1 => 4
  | n + 2 => modifiedFibonacci n + modifiedFibonacci (n + 1)

def evenIndexedUnitsDigit (n : ℕ) : ℕ := modifiedFibonacci (2 * n + 2) % 10

def allDigitsAppearBefore (m : ℕ) : Prop :=
  ∀ d : ℕ, d < 10 → ∃ n : ℕ, n < m ∧ evenIndexedUnitsDigit n = d

theorem last_digit_is_zero :
  ∃ m : ℕ, allDigitsAppearBefore m ∧
    ∀ n : ℕ, n ≥ m → evenIndexedUnitsDigit n ≠ 0 := by
  sorry

#eval [evenIndexedUnitsDigit 0, evenIndexedUnitsDigit 1, evenIndexedUnitsDigit 2, evenIndexedUnitsDigit 3, evenIndexedUnitsDigit 4]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_digit_is_zero_l642_64223
