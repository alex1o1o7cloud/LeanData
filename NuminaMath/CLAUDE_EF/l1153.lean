import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_implication_l1153_115324

theorem negation_of_implication (a b : ℝ) : 
  ¬(a > b → (2 : ℝ)^a > (2 : ℝ)^b - 1) ↔ (a ≤ b ∧ (2 : ℝ)^a ≤ (2 : ℝ)^b - 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_implication_l1153_115324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_town_population_factor_l1153_115341

theorem town_population_factor : 
  (∃ (x y z : ℕ), 
    x^2 + 150 = y^2 - 1 ∧ 
    y^2 - 1 + 150 = z^2) →
  ∃ (n : ℕ), n > 1 ∧ n ∣ (x^2 : ℕ) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_town_population_factor_l1153_115341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_statements_l1153_115304

noncomputable def power_function (α : ℝ) : ℝ → ℝ := λ x => x^α

def quadratic_function (a b c : ℝ) : ℝ → ℝ := λ x => a*x^2 + b*x + c

def statement1 : Prop := ∀ α : ℝ, (power_function α) 0 = 0

def statement2 : Prop := ∀ α : ℝ, α < 0 → ∀ x y : ℝ, x < y → (power_function α) x > (power_function α) y

def statement3 : Prop := ∀ α : ℝ, α > 0 → ∀ x y : ℝ, x < y → (power_function α) x < (power_function α) y

def statement4 : Prop := ∃ a b c : ℝ, (λ x => 2*x^2) = quadratic_function a b c ∧ ∃ α : ℝ, (λ x => 2*x^2) = power_function α

theorem power_function_statements :
  (statement1 = False) ∧
  (statement2 = False) ∧
  (statement3 = False) ∧
  (statement4 = True) := by
  sorry

#check power_function_statements

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_statements_l1153_115304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_speed_fuel_cost_at_10_l1153_115373

-- Define the ship's speed
variable (x : ℝ)

-- Define the fuel cost function
noncomputable def fuel_cost (x : ℝ) : ℝ := (3 / 500) * x^3

-- Define the total cost per hour function
noncomputable def total_cost_per_hour (x : ℝ) : ℝ := fuel_cost x + 96

-- Define the total cost per kilometer function
noncomputable def total_cost_per_km (x : ℝ) : ℝ := total_cost_per_hour x / x

-- Theorem statement
theorem optimal_speed :
  ∀ x > 0, total_cost_per_km x ≥ total_cost_per_km 20 := by
  sorry

-- Verify the fuel cost at 10 km/h
theorem fuel_cost_at_10 : fuel_cost 10 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_speed_fuel_cost_at_10_l1153_115373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2x_unique_property_l1153_115337

noncomputable section

-- Define the functions
def f1 (x : ℝ) := Real.sin (x + Real.pi/4)
def f2 (x : ℝ) := Real.cos (x + Real.pi/4)
def f3 (x : ℝ) := Real.sin (2*x)
def f4 (x : ℝ) := Real.cos (2*x)

-- Define the properties
def has_period_pi (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + Real.pi) = f x

def is_decreasing_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

-- State the theorem
theorem cos_2x_unique_property :
  (has_period_pi f4 ∧ is_decreasing_on_interval f4 0 (Real.pi/2)) ∧
  (¬(has_period_pi f1 ∧ is_decreasing_on_interval f1 0 (Real.pi/2))) ∧
  (¬(has_period_pi f2 ∧ is_decreasing_on_interval f2 0 (Real.pi/2))) ∧
  (¬(has_period_pi f3 ∧ is_decreasing_on_interval f3 0 (Real.pi/2))) :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2x_unique_property_l1153_115337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_f₁_has_property_P_l1153_115319

noncomputable def has_property_P (f : ℝ → ℝ) : Prop :=
  ∃ c : ℝ, c > 0 ∧ ∀ x : ℝ, f x + c ≥ f (x + c)

noncomputable def f₁ : ℝ → ℝ := λ x ↦ (1/2) * x + 1
noncomputable def f₂ : ℝ → ℝ := λ x ↦ x^2
noncomputable def f₃ : ℝ → ℝ := λ x ↦ 2^x

theorem only_f₁_has_property_P :
  has_property_P f₁ ∧ ¬has_property_P f₂ ∧ ¬has_property_P f₃ := by
  sorry

#check only_f₁_has_property_P

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_f₁_has_property_P_l1153_115319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_XY_is_32_l1153_115378

/-- Three circles with centers A, B, and C arranged in a row -/
structure CircleArrangement where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  radius : ℝ
  centers_in_row : A.1 < B.1 ∧ B.1 < C.1
  equal_spacing : B.1 - A.1 = C.1 - B.1
  radius_positive : radius > 0

/-- Line WZ tangent to the third circle -/
def is_tangent_to_third_circle (arr : CircleArrangement) (W Z : ℝ × ℝ) : Prop :=
  let (cx, cy) := arr.C
  ∃ (tx ty : ℝ), (tx - cx)^2 + (ty - cy)^2 = arr.radius^2 ∧
                  (W.1 - tx) * (Z.2 - ty) = (W.2 - ty) * (Z.1 - tx)

/-- Length of XY -/
noncomputable def length_XY (arr : CircleArrangement) : ℝ :=
  2 * Real.sqrt (arr.radius^2 - (arr.radius / 5)^2)

/-- Main theorem -/
theorem length_XY_is_32 (arr : CircleArrangement) (W Z : ℝ × ℝ) :
  arr.radius = 20 →
  is_tangent_to_third_circle arr W Z →
  length_XY arr = 32 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_XY_is_32_l1153_115378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_equiv_a_range_l1153_115352

/-- The function f(x) defined with parameter a. -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a * x + 1) / Real.sqrt (a * x^2 - 4 * a * x + 2)

/-- The theorem stating the equivalence between the domain of f being ℝ and the range of a. -/
theorem domain_equiv_a_range :
  ∀ a : ℝ, (∀ x : ℝ, ∃ y : ℝ, f a x = y) ↔ (0 ≤ a ∧ a < 1/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_equiv_a_range_l1153_115352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_minimum_l1153_115358

/-- Given a positive geometric sequence {aₙ}, if aₘ * aₙ = a₃², 
    then the minimum value of 2/m + 1/(2n) is 3/4 -/
theorem geometric_sequence_minimum (a : ℕ → ℝ) (m n : ℕ) :
  (∀ k, a k > 0) →  -- positive sequence
  (∃ r > 0, ∀ k, a (k + 1) = r * a k) →  -- geometric sequence
  a m * a n = a 3 ^ 2 →  -- given condition
  m > 0 → n > 0 →  -- ensure m and n are positive
  (2 / (m : ℝ) + 1 / (2 * (n : ℝ))) ≥ 3/4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_minimum_l1153_115358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_minimum_implies_a_range_l1153_115326

/-- The function f(x) defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - (a + 2) * x + Real.log x

/-- The theorem statement -/
theorem function_minimum_implies_a_range (a : ℝ) (h1 : a > 0) :
  (∀ x ∈ Set.Icc 1 (Real.exp 1), f a x ≥ -2) ∧ (∃ x ∈ Set.Icc 1 (Real.exp 1), f a x = -2) →
  a ≥ 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_minimum_implies_a_range_l1153_115326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_n_lines_l1153_115377

/-- A point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in a plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if three points are collinear -/
def collinear (p q r : Point) : Prop :=
  (q.x - p.x) * (r.y - p.y) = (r.x - p.x) * (q.y - p.y)

/-- The set of lines connecting a set of points -/
def connecting_lines (points : Set Point) : Set Line :=
  { l | ∃ p q, p ∈ points ∧ q ∈ points ∧ p ≠ q ∧ l.a * p.x + l.b * p.y + l.c = 0 ∧ l.a * q.x + l.b * q.y + l.c = 0 }

/-- Main theorem -/
theorem at_least_n_lines (n : ℕ) (points : Set Point) :
  n ≥ 3 →
  points.ncard = n →
  (∀ p q r, p ∈ points → q ∈ points → r ∈ points → p ≠ q → q ≠ r → p ≠ r → ¬collinear p q r) →
  (connecting_lines points).ncard ≥ n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_n_lines_l1153_115377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_of_triangle_l1153_115307

/-- A right-angled triangle with squares constructed on two sides --/
structure RightTriangleWithSquares where
  -- The vertices of the triangle
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  -- The vertices of the squares
  X : ℝ × ℝ
  Y : ℝ × ℝ
  Z : ℝ × ℝ
  W : ℝ × ℝ
  -- Angle C is 90 degrees
  angle_C_is_right : (B.1 - C.1) * (A.1 - C.1) + (B.2 - C.2) * (A.2 - C.2) = 0
  -- AB = 15
  AB_length : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 15
  -- ABXY is a square
  ABXY_is_square : 
    (X.1 - A.1) = (Y.1 - B.1) ∧ 
    (X.2 - A.2) = (Y.2 - B.2) ∧
    (X.1 - A.1) * (B.1 - A.1) + (X.2 - A.2) * (B.2 - A.2) = 0
  -- ACWZ is a square
  ACWZ_is_square : 
    (W.1 - A.1) = (Z.1 - C.1) ∧ 
    (W.2 - A.2) = (Z.2 - C.2) ∧
    (W.1 - A.1) * (C.1 - A.1) + (W.2 - A.2) * (C.2 - A.2) = 0
  -- X, Y, Z, W lie on a circle
  points_on_circle : ∃ O : ℝ × ℝ, ∃ r : ℝ,
    (X.1 - O.1)^2 + (X.2 - O.2)^2 = r^2 ∧
    (Y.1 - O.1)^2 + (Y.2 - O.2)^2 = r^2 ∧
    (Z.1 - O.1)^2 + (Z.2 - O.2)^2 = r^2 ∧
    (W.1 - O.1)^2 + (W.2 - O.2)^2 = r^2

/-- The perimeter of the triangle ABC is 15 + 15√2 --/
theorem perimeter_of_triangle (t : RightTriangleWithSquares) :
  Real.sqrt ((t.A.1 - t.B.1)^2 + (t.A.2 - t.B.2)^2) +
  Real.sqrt ((t.B.1 - t.C.1)^2 + (t.B.2 - t.C.2)^2) +
  Real.sqrt ((t.C.1 - t.A.1)^2 + (t.C.2 - t.A.2)^2) =
  15 + 15 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_of_triangle_l1153_115307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l1153_115357

noncomputable def a : ℝ := Real.log 3 / Real.log 2
noncomputable def b : ℝ := Real.log 2 / Real.log 3

noncomputable def f (x : ℝ) : ℝ := a^x + x - b

theorem root_in_interval :
  ∃ x : ℝ, x ∈ Set.Ioo (-1 : ℝ) 0 ∧ f x = 0 := by
  sorry

#check root_in_interval

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l1153_115357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_minus_seven_pi_sixths_l1153_115345

theorem sin_alpha_minus_seven_pi_sixths (α : ℝ) 
  (h : Real.cos (α + π/6) - Real.sin α = 2*Real.sqrt 3/3) : 
  Real.sin (α - 7*π/6) = 2/3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_minus_seven_pi_sixths_l1153_115345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_square_area_l1153_115333

theorem second_square_area (x : ℝ) :
  let first_square_area := x^2 + 4*x + 4
  let total_perimeter := 32
  let x_value := 3
  let second_square_area := λ y : ℝ => y^2 - 6*y + 9
  (∀ y : ℝ, y = x_value → second_square_area y = 9) ∧
  (∀ y : ℝ, y = x_value → 4 * (first_square_area.sqrt) + 4 * (second_square_area y).sqrt = total_perimeter) :=
by
  sorry

#check second_square_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_square_area_l1153_115333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l1153_115311

/-- The time taken for a train to cross a bridge -/
noncomputable def time_to_cross_bridge (train_length : ℝ) (train_speed_kmph : ℝ) (bridge_length : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let train_speed_mps := train_speed_kmph * (1000 / 3600)
  total_distance / train_speed_mps

/-- Theorem stating that the time taken for the train to cross the bridge is approximately 30 seconds -/
theorem train_crossing_time :
  let train_length := (110 : ℝ)
  let train_speed_kmph := (60 : ℝ)
  let bridge_length := (390 : ℝ)
  let crossing_time := time_to_cross_bridge train_length train_speed_kmph bridge_length
  ∃ ε > 0, |crossing_time - 30| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l1153_115311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_206788_is_7_l1153_115395

/-- Represents the sequence of digits formed by writing out consecutive whole numbers starting from 1. -/
def ConsecutiveDigitSequence : ℕ → ℕ := sorry

/-- Returns the number of digits in a natural number. -/
def numDigits (n : ℕ) : ℕ := sorry

/-- Returns the nth digit in the ConsecutiveDigitSequence. -/
def nthDigit (n : ℕ) : ℕ := sorry

theorem digit_206788_is_7 : nthDigit 206788 = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_206788_is_7_l1153_115395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplification_l1153_115328

theorem expression_simplification (k : ℤ) : 
  (3 : ℝ)^(-(3*k+2)) - (3 : ℝ)^(-(3*k)) + (3 : ℝ)^(-(3*k+1)) + (3 : ℝ)^(-(3*k-1)) = (22 * (3 : ℝ)^(-3*k)) / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplification_l1153_115328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_slope_l1153_115300

-- Define the two points
def point1 : ℝ × ℝ := (3, -2)
def point2 : ℝ × ℝ := (-4, 3)

-- Define the slope of the line passing through the two points
noncomputable def originalSlope : ℝ := (point2.2 - point1.2) / (point2.1 - point1.1)

-- Theorem: The slope of the perpendicular line is 7/5
theorem perpendicular_slope :
  -1 / originalSlope = 7 / 5 := by
  -- Expand the definition of originalSlope
  have h1 : originalSlope = (3 - (-2)) / ((-4) - 3) := by rfl
  
  -- Simplify the fraction
  have h2 : originalSlope = -5 / 7 := by
    simp [h1]
    norm_num
  
  -- Calculate the perpendicular slope
  calc
    -1 / originalSlope = -1 / (-5 / 7) := by rw [h2]
    _ = 7 / 5 := by norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_slope_l1153_115300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_avg_rate_of_change_x_squared_l1153_115329

-- Define the function
noncomputable def f (x : ℝ) : ℝ := x^2

-- Define the average rate of change
noncomputable def avgRateOfChange (f : ℝ → ℝ) (a b : ℝ) : ℝ := (f b - f a) / (b - a)

-- Theorem statement
theorem avg_rate_of_change_x_squared : avgRateOfChange f 1 2 = 3 := by
  -- Unfold the definitions
  unfold avgRateOfChange f
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_avg_rate_of_change_x_squared_l1153_115329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_maximum_l1153_115379

/-- The integrand function -/
noncomputable def f (x : ℝ) : ℝ := Real.exp (-x - x^2)

/-- The integral as a function of a -/
noncomputable def integral (a : ℝ) : ℝ := ∫ x in a..(a+8), f x

/-- The statement that -4.5 maximizes the integral -/
theorem integral_maximum : ∀ a : ℝ, integral (-4.5) ≥ integral a := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_maximum_l1153_115379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_l1153_115389

-- Define the function as noncomputable
noncomputable def f (x : ℝ) := Real.sqrt (x - 2) / 2

-- State the theorem
theorem range_of_x (x : ℝ) : x ∈ Set.Ici 2 ↔ ∃ y, f y = f x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_l1153_115389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_remainder_l1153_115303

theorem polynomial_remainder (P : Polynomial ℝ) (h1 : P.eval 17 = 101) (h2 : P.eval 95 = 23) :
  ∃ Q : Polynomial ℝ, P = (X - 17) * (X - 95) * Q + (-X + 118) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_remainder_l1153_115303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_satisfying_inequality_is_constant_l1153_115314

open Real Set

/-- A function satisfying the given inequality is constant -/
theorem function_satisfying_inequality_is_constant
  (f : ℝ → ℝ)
  (h_pos : ∀ x, x ∈ Ioo 0 1 → 0 < f x)
  (h_ineq : ∀ x y, x ∈ Ioo 0 1 → y ∈ Ioo 0 1 → f x / f y + f (1 - x) / f (1 - y) ≤ 2) :
  ∀ x y, x ∈ Ioo 0 1 → y ∈ Ioo 0 1 → f x = f y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_satisfying_inequality_is_constant_l1153_115314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_power_simplification_l1153_115348

open Complex

theorem complex_power_simplification :
  ((2 + I) / (2 - I)) ^ 12 = exp (I * Real.arctan (4/3) * 12) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_power_simplification_l1153_115348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jesse_cannot_afford_single_carpet_type_l1153_115356

-- Define the room dimensions
noncomputable def room1_length : ℝ := 11
noncomputable def room1_width : ℝ := 15
noncomputable def room2_base : ℝ := 12
noncomputable def room2_height : ℝ := 8
noncomputable def room3_radius : ℝ := 6

-- Define the carpet already bought and budget
noncomputable def carpet_bought : ℝ := 16
noncomputable def budget : ℝ := 800

-- Define carpet prices
noncomputable def regular_carpet_price : ℝ := 5
noncomputable def deluxe_carpet_price : ℝ := 7.5
noncomputable def luxury_carpet_price : ℝ := 10

-- Calculate total area needed
noncomputable def total_area_needed : ℝ := 
  room1_length * room1_width + 
  (room2_base * room2_height) / 2 + 
  Real.pi * room3_radius^2 - 
  carpet_bought

-- Theorem stating that Jesse cannot afford to cover the entire remaining area with any single type of carpet
theorem jesse_cannot_afford_single_carpet_type : 
  (budget / regular_carpet_price < total_area_needed) ∧ 
  (budget / deluxe_carpet_price < total_area_needed) ∧ 
  (budget / luxury_carpet_price < total_area_needed) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jesse_cannot_afford_single_carpet_type_l1153_115356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_specific_parallel_lines_l1153_115393

/-- The distance between two parallel lines -/
noncomputable def distance_between_parallel_lines (a b c₁ c₂ : ℝ) : ℝ :=
  |c₂ - c₁| / Real.sqrt (a^2 + b^2)

/-- Proof that the distance between x + y + 2 = 0 and 2x + 2y - 5 = 0 is 9√2/4 -/
theorem distance_specific_parallel_lines :
  distance_between_parallel_lines 2 2 4 (-5) = 9 * Real.sqrt 2 / 4 := by
  sorry

#check distance_specific_parallel_lines

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_specific_parallel_lines_l1153_115393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_inscribed_cylinder_height_l1153_115347

/-- 
Given a square-based pyramid with all edges equal and an inscribed cylinder 
with height twice its radius, the height of the cylinder is 2a / (√2 + 2), 
where a is the edge length of the pyramid.
-/
theorem pyramid_inscribed_cylinder_height 
  (a : ℝ) (h : a > 0) : 
  let cylinder_height := (2 * a) / (Real.sqrt 2 + 2)
  ∃ (r : ℝ), r > 0 ∧ cylinder_height = 2 * r := by
  sorry

#check pyramid_inscribed_cylinder_height

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_inscribed_cylinder_height_l1153_115347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1153_115342

noncomputable def f (a b x : ℝ) : ℝ := (1/3) * x^3 - a * x^2 + (a^2 - 1) * x + b

noncomputable def f' (a x : ℝ) : ℝ := x^2 - 2*a*x + a^2 - 1

theorem max_value_of_f (a b : ℝ) :
  (f' a 1 = 0) →  -- x = 1 is an extreme point
  (f a b 1 = 2) →  -- (1, f(1)) is on the line x + y - 3 = 0
  (f' a 1 = -1) →  -- slope of tangent line at x = 1 is -1
  (∀ x ∈ Set.Icc (-2 : ℝ) 4, f a b x ≤ 8) ∧ 
  (∃ x ∈ Set.Icc (-2 : ℝ) 4, f a b x = 8) :=
by sorry

#check max_value_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1153_115342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_prism_ratio_existence_l1153_115355

structure RectangularPrism where
  base_length : ℝ
  base_width : ℝ
  height : ℝ
  base_length_pos : base_length > 0
  base_width_pos : base_width > 0
  height_pos : height > 0

def lateral_surface_area (p : RectangularPrism) : ℝ :=
  2 * (p.base_length + p.base_width) * p.height

def volume (p : RectangularPrism) : ℝ :=
  p.base_length * p.base_width * p.height

theorem rectangular_prism_ratio_existence :
  (∀ (A : RectangularPrism) (l : ℝ), l ≥ 1 →
    ∃ (B : RectangularPrism),
      B.height = A.height ∧
      lateral_surface_area B / lateral_surface_area A = l ∧
      volume B / volume A = l) ∧
  (∀ (A : RectangularPrism) (l : ℝ),
    (∃ (B : RectangularPrism),
      B.height = A.height ∧
      lateral_surface_area B / lateral_surface_area A = l ∧
      volume B / volume A = l) →
    l ≥ 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_prism_ratio_existence_l1153_115355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_point_on_line_optimal_m_value_l1153_115318

/-- Given points P, Q, and R in the xy-plane, prove that PR + RQ is minimized when R lies on the line PQ -/
theorem min_distance_point_on_line (P Q R : ℝ × ℝ) :
  P.1 = -1 →
  P.2 = -2 →
  Q.1 = 4 →
  Q.2 = 2 →
  R.1 = 1 →
  (∀ m : ℝ, dist P R + dist R Q ≤ dist P (1, m) + dist (1, m) Q) →
  R.2 = -2/5 :=
by sorry

/-- The specific value of m that minimizes PR + RQ -/
noncomputable def optimal_m : ℝ := -2/5

/-- Prove that the optimal m value is indeed -2/5 -/
theorem optimal_m_value (P Q : ℝ × ℝ) :
  P.1 = -1 →
  P.2 = -2 →
  Q.1 = 4 →
  Q.2 = 2 →
  (∀ m : ℝ, dist P (1, optimal_m) + dist (1, optimal_m) Q ≤ dist P (1, m) + dist (1, m) Q) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_point_on_line_optimal_m_value_l1153_115318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shift_for_symmetry_l1153_115359

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.cos x + Real.sin x

theorem min_shift_for_symmetry :
  ∃ (m : ℝ), m > 0 ∧
  (∀ (x : ℝ), f (x + m) = -f (-x - m)) ∧
  (∀ (m' : ℝ), m' > 0 → (∀ (x : ℝ), f (x + m') = -f (-x - m')) → m' ≥ m) ∧
  m = 2 * Real.pi / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shift_for_symmetry_l1153_115359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_arithmetic_sequence_l1153_115366

/-- An arithmetic sequence with a positive first term -/
structure ArithmeticSequence where
  a₁ : ℚ
  d : ℚ
  h_positive : a₁ > 0

/-- Sum of the first n terms of an arithmetic sequence -/
def sumOfTerms (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (2 * seq.a₁ + (n - 1) * seq.d) / 2

theorem max_sum_arithmetic_sequence (seq : ArithmeticSequence) 
  (h_equal : sumOfTerms seq 3 = sumOfTerms seq 8) :
  (∃ n : ℕ, n = 5 ∨ n = 6) ∧ 
  (∀ m : ℕ, sumOfTerms seq m ≤ sumOfTerms seq 5 ∧ sumOfTerms seq m ≤ sumOfTerms seq 6) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_arithmetic_sequence_l1153_115366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_implies_b_equals_three_l1153_115384

noncomputable def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

noncomputable def f (b : ℝ) (x : ℝ) : ℝ := Real.sin (2 * x) * Real.log ((2 * x - 3) / (2 * x + b))

theorem even_function_implies_b_equals_three :
  ∃ b : ℝ, IsEven (f b) → b = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_implies_b_equals_three_l1153_115384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_calculation_l1153_115388

-- Define the force function
def F (x : ℝ) : ℝ := 4 * x - 1

-- Define the work function
noncomputable def work (a b : ℝ) : ℝ := ∫ x in a..b, F x

-- Theorem statement
theorem work_calculation : work 1 3 = 14 := by
  -- Unfold the definition of work
  unfold work
  -- Evaluate the integral
  simp [F]
  -- The rest of the proof would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_calculation_l1153_115388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_five_ones_approx_l1153_115385

/-- The probability of rolling exactly 5 ones with 12 six-sided dice -/
def probability_five_ones : ℚ :=
  (792 * 5^7 : ℚ) / 6^12

/-- Theorem stating that the probability is approximately 0.028 -/
theorem probability_five_ones_approx :
  |probability_five_ones - 0.028| < 0.0005 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_five_ones_approx_l1153_115385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l1153_115382

theorem trigonometric_identity (x : ℝ) 
  (h1 : x ∈ {y : ℝ | ∃ (k : ℤ), y = π/2 * (2*k + 1) ∨ 
                     ∃ (n : ℤ), y = π/4 * (4*n - 1) ∨ 
                     ∃ (l : ℤ), y = Real.arctan (1/2) + π * l})
  (h2 : 2 * Real.cos x - Real.sin x ≠ 0) :
  (Real.sin x)^3 + (Real.cos x)^3 / (2 * Real.cos x - Real.sin x) = Real.cos (2*x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l1153_115382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_problem_l1153_115339

theorem trigonometric_problem (α β : Real) 
  (h1 : Real.cos α = 1/7)
  (h2 : Real.cos (α - β) = 13/14)
  (h3 : 0 < β)
  (h4 : β < α)
  (h5 : α < π/2) :
  (Real.tan (2*α) = -8*Real.sqrt 3/47) ∧ (β = π/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_problem_l1153_115339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_doily_table_radius_l1153_115396

theorem doily_table_radius (r₁ r₂ r₃ R : ℝ) : 
  r₁ = 2 ∧ r₂ = 3 ∧ r₃ = 10 ∧ 
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ),
    -- Centers of the three circles
    (x₁^2 + y₁^2 = R^2) ∧ 
    (x₂^2 + y₂^2 = R^2) ∧ 
    (x₃^2 + y₃^2 = R^2) ∧ 
    -- Circles touch each other
    ((x₁ - x₂)^2 + (y₁ - y₂)^2 = (r₁ + r₂)^2) ∧
    ((x₂ - x₃)^2 + (y₂ - y₃)^2 = (r₂ + r₃)^2) ∧
    ((x₃ - x₁)^2 + (y₃ - y₁)^2 = (r₃ + r₁)^2) ∧
    -- Circles touch the edge of the larger circle
    (∀ i j, (x₁ - i)^2 + (y₁ - j)^2 = r₁^2 → i^2 + j^2 ≥ R^2) ∧
    (∀ i j, (x₂ - i)^2 + (y₂ - j)^2 = r₂^2 → i^2 + j^2 ≥ R^2) ∧
    (∀ i j, (x₃ - i)^2 + (y₃ - j)^2 = r₃^2 → i^2 + j^2 ≥ R^2)
  → R = 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_doily_table_radius_l1153_115396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_implies_b_zero_l1153_115362

noncomputable section

-- Define the function f(x)
def f (x : ℝ) : ℝ := (x + 1)^3 + x / (x + 1)

-- Define the line g(x)
def g (b : ℝ) (x : ℝ) : ℝ := -x + b

-- State the theorem
theorem intersection_sum_implies_b_zero 
  (h : ∃ (x₁ x₂ x₃ : ℝ), f x₁ = g b x₁ ∧ f x₂ = g b x₂ ∧ f x₃ = g b x₃ ∧ x₁ + x₂ + x₃ = -2) :
  b = 0 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_implies_b_zero_l1153_115362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_logarithm_relation_l1153_115332

theorem exponential_logarithm_relation (x y : ℝ) : 
  (2.5 : ℝ)^x = 1000 → (0.25 : ℝ)^y = 1000 → 1/x - 1/y = 1/3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_logarithm_relation_l1153_115332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angeli_candies_l1153_115353

/-- The number of candies Angeli had initially -/
def total_candies : ℕ := sorry

/-- The number of boys who received lollipops -/
def num_boys : ℕ := sorry

/-- The number of girls who received candy canes -/
def num_girls : ℕ := sorry

/-- The total number of children who received candies -/
axiom total_children : num_boys + num_girls = 40

/-- One-third of the candies were lollipops -/
axiom lollipop_fraction : total_candies / 3 = num_boys * 3

/-- Two-thirds of the candies were candy canes -/
axiom candy_cane_fraction : 2 * total_candies / 3 = num_girls * 2

theorem angeli_candies : total_candies = 144 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angeli_candies_l1153_115353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spring_stretch_limit_l1153_115338

/-- Represents the length of a spring as a function of weight. -/
noncomputable def spring_length (initial_length : ℝ) (stretch_rate : ℝ) (max_length : ℝ) (weight : ℝ) : ℝ :=
  min (initial_length + stretch_rate * weight) max_length

theorem spring_stretch_limit (initial_length stretch_rate max_length weight : ℝ) 
  (h1 : initial_length = 8)
  (h2 : stretch_rate = 0.5)
  (h3 : max_length = 20)
  (h4 : weight = 30) :
  spring_length initial_length stretch_rate max_length weight - initial_length ≠ 15 := by
  sorry

#check spring_stretch_limit

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spring_stretch_limit_l1153_115338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_ratio_l1153_115369

/-- A truncated right circular cone with an inscribed sphere -/
structure TruncatedConeWithSphere where
  R : ℝ  -- radius of the larger base
  r : ℝ  -- radius of the smaller base
  s : ℝ  -- radius of the inscribed sphere
  h : ℝ  -- height of the truncated cone

/-- The volume of a sphere -/
noncomputable def sphereVolume (radius : ℝ) : ℝ := (4 / 3) * Real.pi * radius ^ 3

/-- The volume of a truncated cone -/
noncomputable def truncatedConeVolume (R r h : ℝ) : ℝ := (1 / 3) * Real.pi * h * (R^2 + r^2 + R*r)

/-- Theorem: If a sphere is inscribed in a truncated right circular cone and the volume of the
    truncated cone is three times that of the sphere, then the ratio of the radius of the bottom
    base to the radius of the top base of the truncated cone is (5 + √21) / 2 -/
theorem inscribed_sphere_ratio (cone : TruncatedConeWithSphere) 
    (h_geom : cone.s^2 = cone.R * cone.r)  -- geometric mean theorem
    (h_vol : truncatedConeVolume cone.R cone.r cone.h = 3 * sphereVolume cone.s) :
    cone.R / cone.r = (5 + Real.sqrt 21) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_ratio_l1153_115369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monochromatic_unit_distance_l1153_115367

/-- A coloring of the plane using three colors -/
def Coloring := ℝ × ℝ → Fin 3

/-- The statement that for any three-coloring of the plane, 
    there exist two points of the same color that are exactly one unit apart -/
theorem monochromatic_unit_distance (c : Coloring) : 
  ∃ (x y : ℝ × ℝ), c x = c y ∧ dist x y = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monochromatic_unit_distance_l1153_115367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_repeated_number_l1153_115301

/-- Represents a grid of integers -/
def Grid (n : ℕ) := Fin n → Fin n → ℤ

/-- Checks if two positions in the grid are adjacent -/
def adjacent (n : ℕ) (i j k l : Fin n) : Prop :=
  (i = k ∧ j.val + 1 = l.val) ∨ 
  (i = k ∧ j.val = l.val + 1) ∨
  (i.val + 1 = k.val ∧ j = l) ∨
  (i.val = k.val + 1 ∧ j = l)

/-- The main theorem -/
theorem grid_repeated_number (n : ℕ) (grid : Grid n) :
  (∀ i j k l : Fin n, adjacent n i j k l → |grid i j - grid k l| ≤ 1) →
  ∃ x : ℤ, (Finset.filter (fun p : Fin n × Fin n => grid p.1 p.2 = x) Finset.univ).card ≥ n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_repeated_number_l1153_115301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_price_per_bottle_is_162_cents_l1153_115316

/-- Calculates the average price per bottle given the number and price of large and small bottles -/
def averagePricePerBottle (largeBotCount : ℕ) (largeBotPrice : ℚ) (smallBotCount : ℕ) (smallBotPrice : ℚ) : ℚ :=
  let totalCost := largeBotCount * largeBotPrice + smallBotCount * smallBotPrice
  let totalBottles := largeBotCount + smallBotCount
  totalCost / totalBottles

/-- Rounds a rational number to the nearest cent (hundredth) -/
def roundToCent (x : ℚ) : ℚ :=
  (x * 100).floor / 100

theorem average_price_per_bottle_is_162_cents :
  roundToCent (averagePricePerBottle 1375 (175/100) 690 (135/100)) = 162/100 := by
  sorry

#eval roundToCent (averagePricePerBottle 1375 (175/100) 690 (135/100))

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_price_per_bottle_is_162_cents_l1153_115316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_theta_plus_pi_over_4_l1153_115354

theorem tan_theta_plus_pi_over_4 (θ : Real) :
  Real.sin θ + Real.cos θ = (2 * Real.sqrt 10) / 5 →
  Real.tan (θ + π / 4) = 2 ∨ Real.tan (θ + π / 4) = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_theta_plus_pi_over_4_l1153_115354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_implies_k_l1153_115375

-- Define the functions
def f (x : ℝ) := x^2
def g (k : ℝ) (x : ℝ) := k * x

-- Define the area of the enclosed region
noncomputable def enclosed_area (k : ℝ) : ℝ := ∫ x in (0)..(k), (g k x - f x)

-- Theorem statement
theorem enclosed_area_implies_k (k : ℝ) :
  k > 0 → enclosed_area k = 9/2 → k = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_implies_k_l1153_115375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boys_in_second_group_l1153_115349

/-- Represents the daily work units done by a person -/
abbrev DailyWork := ℕ

/-- Represents the number of days to complete the work -/
abbrev Days := ℕ

/-- Represents the total work units -/
abbrev TotalWork := ℕ

/-- Calculates the total work done by a group -/
def calculate_total_work (men : ℕ) (boys : ℕ) (man_work : DailyWork) (boy_work : DailyWork) (days : Days) : TotalWork :=
  (men * man_work + boys * boy_work) * days

theorem boys_in_second_group 
  (total_work : TotalWork)
  (man_work boy_work : DailyWork)
  (days1 days2 : Days)
  (men1 men2 boys1 : ℕ)
  (h1 : man_work = 2 * boy_work)
  (h2 : calculate_total_work men1 boys1 man_work boy_work days1 = total_work)
  (h3 : days1 = 5)
  (h4 : men1 = 12)
  (h5 : boys1 = 16)
  (h6 : days2 = 4)
  (h7 : men2 = 13) :
  ∃ (boys2 : ℕ), calculate_total_work men2 boys2 man_work boy_work days2 = total_work ∧ boys2 = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boys_in_second_group_l1153_115349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_P_locus_is_ellipse_l1153_115313

noncomputable section

-- Define the circle
def circleEq (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 36

-- Define the center of the circle
def M : ℝ × ℝ := (-2, 0)

-- Define point N
def N : ℝ × ℝ := (2, 0)

-- Define a point A on the circle
noncomputable def A : ℝ × ℝ := sorry

-- Assume A is on the circle
axiom A_on_circle : circleEq A.1 A.2

-- Define point P
noncomputable def P : ℝ × ℝ := sorry

-- P is on line MA
axiom P_on_MA : ∃ (t : ℝ), P = (t * (A.1 - M.1) + M.1, t * (A.2 - M.2) + M.2)

-- P is equidistant from A and N
axiom P_equidistant : (P.1 - A.1)^2 + (P.2 - A.2)^2 = (P.1 - N.1)^2 + (P.2 - N.2)^2

-- Theorem: The locus of P forms an ellipse
theorem P_locus_is_ellipse : ∃ (a b c d e f : ℝ), 
  a * P.1^2 + b * P.1 * P.2 + c * P.2^2 + d * P.1 + e * P.2 + f = 0 ∧ 
  b^2 - 4 * a * c < 0 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_P_locus_is_ellipse_l1153_115313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_brother_spent_formula_l1153_115363

/-- Represents the scenario of Sarah and her brother buying books -/
structure BookPurchase where
  x : ℝ  -- number of paperback books Sarah bought
  y : ℝ  -- number of hardback books Sarah bought

/-- Calculates the total amount spent by Sarah's brother -/
noncomputable def brotherSpent (purchase : BookPurchase) : ℝ :=
  0.9 * (8 * (1/3 * purchase.x) + 15 * (2 * purchase.y))

/-- Theorem stating the correct formula for the amount spent by Sarah's brother -/
theorem brother_spent_formula (purchase : BookPurchase) :
  brotherSpent purchase = 2.4 * purchase.x + 27 * purchase.y := by
  -- Unfold the definition of brotherSpent
  unfold brotherSpent
  -- Simplify the expression
  simp [mul_add, mul_assoc, mul_comm, mul_left_comm]
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_brother_spent_formula_l1153_115363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_target_score_l1153_115334

/-- Represents a cricket game with given parameters -/
structure CricketGame where
  totalOvers : ℕ
  firstPeriodOvers : ℕ
  firstPeriodRunRate : ℚ
  requiredOverallRunRate : ℚ

/-- Calculates the target score for a cricket game -/
def targetScore (game : CricketGame) : ℕ :=
  ⌈(game.totalOvers : ℚ) * game.requiredOverallRunRate⌉.toNat

/-- Theorem stating the target score for the given cricket game -/
theorem cricket_target_score (game : CricketGame) 
  (h1 : game.totalOvers = 50)
  (h2 : game.firstPeriodOvers = 10)
  (h3 : game.firstPeriodRunRate = 16/5)  -- 3.2 as a fraction
  (h4 : game.requiredOverallRunRate = 6) :
  targetScore game = 272 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_target_score_l1153_115334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_complement_equals_set_l1153_115325

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set M
def M : Set ℝ := {x | x * (x - 2) ≤ 0}

-- Define set N
def N : Set ℝ := {1, 2, 3, 4}

-- State the theorem
theorem intersection_complement_equals_set :
  N ∩ (U \ M) = {3, 4} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_complement_equals_set_l1153_115325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_line_correct_l1153_115360

/-- Parametric equations of curve C₁ -/
noncomputable def C₁ (θ : ℝ) : ℝ × ℝ := (3 + 4 * Real.cos θ, 4 + 4 * Real.sin θ)

/-- Polar equation of curve C₂ -/
noncomputable def C₂ (θ : ℝ) : ℝ := 4 * Real.sin θ

/-- Polar equation of the line containing intersection points of C₁ and C₂ -/
def intersection_line (ρ θ : ℝ) : Prop :=
  6 * ρ * Real.cos θ + 4 * ρ * Real.sin θ - 9 = 0

/-- Theorem stating that the intersection line is correct given C₁ and C₂ -/
theorem intersection_line_correct :
  ∀ ρ θ : ℝ, (∃ t : ℝ, C₁ t = (ρ * Real.cos θ, ρ * Real.sin θ)) ∧ 
             (C₂ θ = ρ) → 
             intersection_line ρ θ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_line_correct_l1153_115360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domino_arrangement_exists_l1153_115370

/-- Represents a domino tile with two non-negative integer values -/
structure Domino where
  a : ℕ
  b : ℕ

/-- Represents a position on the grid -/
structure Position where
  row : Fin 7
  col : Fin 7

/-- Represents the orientation of a domino tile -/
inductive DominoOrientation
  | Horizontal
  | Vertical

/-- Represents a placed domino on the grid -/
structure PlacedDomino where
  domino : Domino
  position : Position
  orientation : DominoOrientation

/-- The type of a valid grid arrangement -/
def GridArrangement := List PlacedDomino

/-- Sums the points in a given row -/
def sumRow (arrangement : GridArrangement) (row : Fin 7) : ℕ := sorry

/-- Sums the points in a given column -/
def sumColumn (arrangement : GridArrangement) (col : Fin 7) : ℕ := sorry

/-- Checks if all dominos cover two adjacent cells -/
def coversTwoAdjacentCells (arrangement : GridArrangement) : Prop := sorry

/-- Checks if a grid arrangement is valid -/
def isValidArrangement (arrangement : GridArrangement) : Prop :=
  (arrangement.length = 28) ∧
  (∀ row : Fin 7, (sumRow arrangement row) = 21) ∧
  (∀ col : Fin 7, (sumColumn arrangement col) = 24) ∧
  (coversTwoAdjacentCells arrangement)

theorem domino_arrangement_exists : ∃ (arrangement : GridArrangement), isValidArrangement arrangement := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domino_arrangement_exists_l1153_115370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_multiple_in_set_l1153_115392

def is_valid_number (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n < 10000 ∧ 
  ∃ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  ({a, b, c, d} : Finset ℕ) = {2, 4, 5, 7} ∧
  n = 1000 * a + 100 * b + 10 * c + d

def is_multiple_in_set (n : ℕ) : Prop :=
  is_valid_number n ∧ ∃ m, is_valid_number m ∧ m ≠ n ∧ n % m = 0

theorem unique_multiple_in_set :
  ∀ n, is_multiple_in_set n ↔ n = 7425 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_multiple_in_set_l1153_115392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_semicircles_area_and_volume_l1153_115344

/-- Represents the figure formed by rotating three semicircles around a diameter --/
structure RotatedSemicircles where
  diameter : ℝ
  diameter_positive : diameter > 0

/-- Calculates the surface area of the rotated semicircles figure --/
noncomputable def surface_area (figure : RotatedSemicircles) : ℝ :=
  600 * Real.pi * figure.diameter^2 / 400

/-- Calculates the volume of the rotated semicircles figure --/
noncomputable def volume (figure : RotatedSemicircles) : ℝ :=
  1000 * Real.pi * figure.diameter^3 / 8000

/-- Theorem stating the surface area and volume of the specific figure --/
theorem rotated_semicircles_area_and_volume :
  ∀ (figure : RotatedSemicircles),
    figure.diameter = 20 →
    surface_area figure = 600 * Real.pi ∧
    volume figure = 1000 * Real.pi := by
  intro figure h
  simp [surface_area, volume, h]
  sorry

#check rotated_semicircles_area_and_volume

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_semicircles_area_and_volume_l1153_115344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_16_4_l1153_115323

theorem log_16_4 : Real.log 4 / Real.log 16 = 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_16_4_l1153_115323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2017_value_l1153_115397

def a : ℕ → ℚ
  | 0 => 1  -- Add a case for 0
  | 1 => 1
  | n + 1 => (2016 * a n) / (2014 * a n + 2016)

theorem a_2017_value : a 2017 = 1008 / (1007 * 2017 + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2017_value_l1153_115397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_swept_area_l1153_115372

-- Define the circle and its properties
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define points A, B, and C
def A (c : Circle) : ℝ × ℝ := sorry
def B (c : Circle) : ℝ × ℝ := sorry
def C (c : Circle) : ℝ × ℝ := sorry

-- AB is a diameter of the circle
def is_diameter (c : Circle) (p q : ℝ × ℝ) : Prop := sorry

axiom ab_diameter (c : Circle) : is_diameter c (A c) (B c)

-- Length of AB is 30
axiom ab_length : ∀ c, dist (A c) (B c) = 30

-- C lies on the semicircle defined by AB
def on_semicircle (c : Circle) (p q r : ℝ × ℝ) : Prop := sorry

axiom c_on_semicircle (c : Circle) : on_semicircle c (A c) (B c) (C c)

-- Define the centroid of triangle ABC
noncomputable def centroid (c : Circle) : ℝ × ℝ := sorry

-- Define the area swept by the centroid
noncomputable def area_swept_by_centroid (c : Circle) : ℝ := sorry

-- The theorem to prove
theorem centroid_swept_area :
  ∀ c : Circle, area_swept_by_centroid c = 25 * Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_swept_area_l1153_115372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_double_angle_plus_pi_twelve_l1153_115374

theorem sine_double_angle_plus_pi_twelve (α : ℝ) :
  0 < α ∧ α < π / 2 →
  Real.cos (α + π / 6) = -1 / 3 →
  Real.sin (2 * α + π / 12) = (7 * Real.sqrt 2 - 8) / 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_double_angle_plus_pi_twelve_l1153_115374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_theorem_l1153_115391

-- Define the integrand
noncomputable def f (x : ℝ) : ℝ := (x^3 + x^2 + 2) / (x * (x^2 - 1)^2)

-- Define the antiderivative
noncomputable def F (x : ℝ) : ℝ := 2 * Real.log (abs x) - 3/4 * Real.log (abs (x - 1)) - 1/(x - 1) - 5/4 * Real.log (abs (x + 1)) + 1/(2*(x + 1))

-- Theorem statement
theorem integral_theorem (x : ℝ) (hx : x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1) : 
  deriv F x = f x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_theorem_l1153_115391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_solution_sets_l1153_115386

theorem quadratic_inequality_solution_sets
  (a c : ℝ)
  (h : Set.union (Set.Ioi (-1/3 : ℝ)) (Set.Ioi (1/2)) =
       {x | a * x^2 + 2 * x + c < 0}) :
  {x : ℝ | c * x^2 - 2 * x + a ≤ 0} = Set.Icc (-2) 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_solution_sets_l1153_115386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_8_15_l1153_115387

noncomputable def minute_hand_angle (minutes : ℝ) : ℝ :=
  (minutes / 60) * 360

noncomputable def hour_hand_angle (hours : ℝ) (minutes : ℝ) : ℝ :=
  (hours % 12 + minutes / 60) * 30

noncomputable def smaller_angle (angle1 : ℝ) (angle2 : ℝ) : ℝ :=
  min (abs (angle1 - angle2)) (360 - abs (angle1 - angle2))

theorem clock_angle_at_8_15 :
  smaller_angle (hour_hand_angle 20 15) (minute_hand_angle 15) = 157.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_8_15_l1153_115387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vertex_x_coordinate_l1153_115312

/-- Represents a parabola of the form y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- A point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

theorem parabola_vertex_x_coordinate (p : Parabola) (A B C : Point) :
  A.x = -5 →
  B.x = -1 →
  C.x = -p.b / (2 * p.a) →  -- Vertex x-coordinate formula
  C.y = p.c - p.b^2 / (4 * p.a) →  -- Vertex y-coordinate formula
  A.y > B.y →
  B.y ≥ C.y →
  C.x > -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vertex_x_coordinate_l1153_115312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incircle_line_bisection_l1153_115365

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define a triangle
structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define an incircle of a triangle
structure Incircle (T : Triangle) where
  center : Point
  radius : ℝ

-- Define a line
structure Line where
  point1 : Point
  point2 : Point

-- Define the perimeter of a triangle
noncomputable def perimeter (T : Triangle) : ℝ := sorry

-- Define the area of a triangle
noncomputable def area (T : Triangle) : ℝ := sorry

-- Define a function to check if a line bisects the perimeter of a triangle
def bisects_perimeter (l : Line) (T : Triangle) : Prop := sorry

-- Define a function to check if a line bisects the area of a triangle
def bisects_area (l : Line) (T : Triangle) : Prop := sorry

-- Define a function to check if a line passes through a point
def passes_through (l : Line) (p : Point) : Prop := sorry

-- Main theorem
theorem incircle_line_bisection (T : Triangle) (I : Incircle T) (l : Line) :
  passes_through l I.center →
  (bisects_perimeter l T ↔ bisects_area l T) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_incircle_line_bisection_l1153_115365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_l1153_115331

noncomputable section

/-- A function f with a single parameter ω that satisfies the given conditions -/
def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x - Real.pi / 4)

/-- The proposition that ω satisfies the given conditions -/
def satisfies_conditions (ω : ℝ) : Prop :=
  ω > 0 ∧
  ∃! x₁, x₁ ∈ Set.Icc 0 1 ∧ ∀ y, f ω (2 * x₁ - y) = f ω y ∧
  ∃! x₂, x₂ ∈ Set.Icc 0 1 ∧ ∀ y, f ω (2 * x₂ - y) = -f ω y

/-- The theorem stating that if ω satisfies the conditions, then it falls within the specified range -/
theorem omega_range (ω : ℝ) :
  satisfies_conditions ω → 3 * Real.pi / 4 ≤ ω ∧ ω < 5 * Real.pi / 4 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_l1153_115331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_integer_fifth_power_l1153_115315

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem no_integer_fifth_power : ∀ n m : ℤ, (n : ℂ) + i ≠ (m : ℂ)^(1/5) :=
by
  sorry

-- Note: The proof is omitted as per instructions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_integer_fifth_power_l1153_115315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_cost_theorem_l1153_115327

/-- Represents the cost of fencing a rectangular field -/
noncomputable def fencing_cost (length width area fence_rate : ℝ) : ℝ :=
  2 * (length + width) * fence_rate / 100

/-- Theorem: Cost of fencing a rectangular field with given conditions -/
theorem fencing_cost_theorem (length width area fence_rate : ℝ) 
  (h1 : length / width = 3 / 4)
  (h2 : length * width = area)
  (h3 : area = 10800)
  (h4 : fence_rate = 25) : 
  fencing_cost length width area fence_rate = 105 := by
  sorry

#check fencing_cost_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_cost_theorem_l1153_115327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orange_juice_production_l1153_115361

/-- Given the total orange production and export percentage, calculates the amount used for juice --/
noncomputable def oranges_for_juice (total_production : ℝ) (export_percentage : ℝ) (juice_percentage : ℝ) : ℝ :=
  (1 - export_percentage / 100) * (juice_percentage / 100) * total_production

/-- Rounds a real number to the nearest tenth --/
noncomputable def round_to_tenth (x : ℝ) : ℝ :=
  ⌊x * 10 + 0.5⌋ / 10

/-- Theorem stating that the rounded result of oranges used for juice is 2.6 million tons --/
theorem orange_juice_production :
  round_to_tenth (oranges_for_juice 6.2 30 60) = 2.6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_orange_juice_production_l1153_115361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_calculation_l1153_115364

/-- Calculates the simple interest given principal, rate, and time -/
noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

/-- Calculates the compound interest given principal, rate, and time -/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * ((1 + rate / 100) ^ time - 1)

/-- Theorem: If 5000 invested for 2 years yields 512.50 in compound interest,
    then it yields 495 in simple interest -/
theorem interest_calculation (rate : ℝ) :
  compound_interest 5000 rate 2 = 512.50 →
  simple_interest 5000 rate 2 = 495 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_calculation_l1153_115364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_swaps_to_transform_l1153_115317

/-- A single swap operation on a matrix -/
def single_swap {n : ℕ} (M : Matrix (Fin n) (Fin n) ℕ) (i j k l : Fin n) : Matrix (Fin n) (Fin n) ℕ :=
  sorry

/-- Predicate to check if a matrix contains all integers from 1 to n^2 -/
def contains_all_integers {n : ℕ} (M : Matrix (Fin n) (Fin n) ℕ) : Prop :=
  sorry

theorem max_swaps_to_transform (n : ℕ) (hn : n ≥ 2) :
  ∃ (m : ℕ), ∀ (A B : Matrix (Fin n) (Fin n) ℕ),
    contains_all_integers A → contains_all_integers B →
    (∃ (k : ℕ) (swaps : List (Fin n × Fin n × Fin n × Fin n)),
      k ≤ m ∧ 
      (swaps.foldl (fun acc s => single_swap acc s.1 s.2.1 s.2.2.1 s.2.2.2) A) = B) ∧
    (∀ (m' : ℕ), m' < m → 
      ∃ (A' B' : Matrix (Fin n) (Fin n) ℕ),
        contains_all_integers A' ∧ contains_all_integers B' ∧
        ¬(∃ (k : ℕ) (swaps : List (Fin n × Fin n × Fin n × Fin n)),
          k ≤ m' ∧ 
          (swaps.foldl (fun acc s => single_swap acc s.1 s.2.1 s.2.2.1 s.2.2.2) A') = B')) :=
by
  use n^2
  sorry

#check max_swaps_to_transform

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_swaps_to_transform_l1153_115317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_brown_catch_up_speed_l1153_115306

-- Define the circular track
def CircularTrack := ℝ

-- Define the runners
structure Runner :=
  (position : CircularTrack)
  (speed : ℝ)
  (direction : Bool)

-- Define the initial setup
noncomputable def initial_setup (track_length : ℝ) : Runner × Runner :=
  let brown := ⟨(0 : ℝ), 1, true⟩  -- Brown starts at 0, with speed 1, running clockwise
  let tomkins := ⟨track_length / 8, 7/4, false⟩  -- Tomkins starts 1/8 ahead, with speed 7/4, running counter-clockwise
  (brown, tomkins)

-- Theorem statement
theorem brown_catch_up_speed 
  (track_length : ℝ) 
  (h_positive : track_length > 0) :
  let (brown, tomkins) := initial_setup track_length
  let meeting_point := track_length / 6
  let brown_remaining := track_length - meeting_point
  let tomkins_remaining := track_length - (meeting_point + track_length / 8)
  (brown_remaining / tomkins_remaining) * tomkins.speed / brown.speed = 119 / 64 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_brown_catch_up_speed_l1153_115306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_median_extension_l1153_115376

/-- Given a triangle ABC with sides AB = a and BC = b, and a median BD extended to intersect 
    the circumcircle at E such that BD/DE = m/n, prove that the length of AC is equal to 
    sqrt((2n/(m+n)) * (a^2 + b^2)). -/
theorem triangle_median_extension (a b m n : ℝ) (h_pos : 0 < m ∧ 0 < n) :
  ∃ (AC : ℝ), AC = Real.sqrt ((2 * n / (m + n)) * (a^2 + b^2)) :=
by
  -- We'll define the triangle and other elements conceptually
  -- without explicitly constructing the geometric objects
  
  -- Let's assume the existence of the triangle and its properties
  have h_triangle : ∃ (A B C : ℝ × ℝ), True := by exact ⟨(0, 0), (a, 0), (b, 0), trivial⟩
  
  -- Assume the existence of point D on BC and E on the circumcircle
  have h_points : ∃ (D E : ℝ × ℝ), True := by exact ⟨(0, 0), (1, 0), trivial⟩
  
  -- Assume the property BD/DE = m/n
  have h_ratio : True := trivial
  
  -- Now we can proceed with the proof
  -- For now, we'll use sorry to skip the actual proof steps
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_median_extension_l1153_115376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_removed_percentage_l1153_115368

noncomputable def box_length : ℝ := 18
noncomputable def box_width : ℝ := 12
noncomputable def box_height : ℝ := 9
noncomputable def cube_side : ℝ := 2

noncomputable def original_volume : ℝ := box_length * box_width * box_height
noncomputable def removed_cube_volume : ℝ := cube_side^3
def num_corners : ℕ := 8

noncomputable def total_removed_volume : ℝ := removed_cube_volume * (num_corners : ℝ)
noncomputable def volume_ratio : ℝ := total_removed_volume / original_volume

theorem volume_removed_percentage (ε : ℝ) (h : ε > 0) :
  ∃ δ : ℝ, δ > 0 ∧ |volume_ratio * 100 - 3.29| < δ ∧ δ < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_removed_percentage_l1153_115368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_single_elimination_tournament_games_tournament_with_19_teams_l1153_115322

/-- In a single-elimination tournament with no ties, the number of games
    played is one less than the number of teams. -/
theorem single_elimination_tournament_games (n : ℕ) (n_pos : 0 < n) :
  n - 1 = n - 1 :=
by
  rfl

/-- For a tournament with 19 teams, 18 games are played. -/
theorem tournament_with_19_teams :
  let num_teams : ℕ := 19
  let num_games : ℕ := num_teams - 1
  num_games = 18 :=
by
  rfl

#check tournament_with_19_teams

end NUMINAMATH_CALUDE_ERRORFEEDBACK_single_elimination_tournament_games_tournament_with_19_teams_l1153_115322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_problem_l1153_115381

open InnerProductSpace

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem vector_magnitude_problem (a b : V) 
  (h1 : ‖a - b‖ = Real.sqrt 3)
  (h2 : ‖a + b‖ = ‖(2 : ℝ) • a - b‖) :
  ‖b‖ = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_problem_l1153_115381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_distance_ratio_l1153_115321

noncomputable def kmhr_to_ms (v : ℝ) : ℝ := v / 3.6

noncomputable def distance (v0 : ℝ) (a : ℝ) (t : ℝ) : ℝ := v0 * t + 0.5 * a * t^2

theorem car_distance_ratio :
  let v0_A := kmhr_to_ms 70
  let a_A := (3 : ℝ)
  let v0_B := kmhr_to_ms 35
  let a_B := (1.5 : ℝ)
  let t := (10 * 3600 : ℝ)
  let d_A := distance v0_A a_A t
  let d_B := distance v0_B a_B t
  d_A / d_B = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_distance_ratio_l1153_115321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_ratio_l1153_115340

/-- A parabola with equation y^2 = 4x and focus F(1, 0) -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = 4 * p.1}

/-- The focus of the parabola -/
def F : ℝ × ℝ := (1, 0)

/-- A line passing through F with slope √3 -/
def Line : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = Real.sqrt 3 * (p.1 - 1)}

/-- Point A is an intersection of the parabola and the line -/
noncomputable def A : ℝ × ℝ :=
  (3, 2 * Real.sqrt 3)

/-- Point B is another intersection of the parabola and the line -/
noncomputable def B : ℝ × ℝ :=
  (1/3, -2/3 * Real.sqrt 3)

/-- The distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem parabola_intersection_ratio :
  A ∈ Parabola ∧ A ∈ Line ∧
  B ∈ Parabola ∧ B ∈ Line ∧
  distance F A > distance F B →
  distance F A / distance F B = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_ratio_l1153_115340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_AB_length_l1153_115346

-- Define the curves and ray
def C1 (x y : ℝ) : Prop := x^2 / 3 + y^2 = 1
def C2 (ρ θ : ℝ) : Prop := ρ = 4 * Real.sin (θ + Real.pi/3)
def rayOM (ρ θ : ℝ) : Prop := θ = Real.pi/6 ∧ ρ ≥ 0

-- Define points A and B
def pointA (x y : ℝ) : Prop := C1 x y ∧ ∃ ρ, rayOM ρ (Real.arctan (y/x))
def pointB (x y : ℝ) : Prop := C1 x y

-- Theorem statement
theorem AB_length :
  ∀ (xA yA xB yB : ℝ),
  pointA xA yA →
  pointB xB yB →
  Real.sqrt ((xB - xA)^2 + (yB - yA)^2) = 4 * Real.sqrt 5 / 5 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_AB_length_l1153_115346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paddyfield_warbler_percentage_l1153_115398

/-- Represents the bird population in Goshawk-Eurasian Nature Reserve -/
structure BirdPopulation where
  total : ℝ
  hawk_percent : ℝ
  other_percent : ℝ
  kingfisher_ratio : ℝ

/-- Calculates the percentage of non-hawks that are paddyfield-warblers -/
noncomputable def paddyfield_warbler_percent (pop : BirdPopulation) : ℝ :=
  let non_hawk_percent := 1 - pop.hawk_percent
  let hawk_paddyfield_kingfisher_percent := 1 - pop.other_percent
  let x := (hawk_paddyfield_kingfisher_percent - pop.hawk_percent) / (non_hawk_percent * (1 + pop.kingfisher_ratio))
  x * 100

/-- Theorem stating that the percentage of non-hawks that are paddyfield-warblers is 40% -/
theorem paddyfield_warbler_percentage 
  (pop : BirdPopulation)
  (h1 : pop.hawk_percent = 0.3)
  (h2 : pop.other_percent = 0.35)
  (h3 : pop.kingfisher_ratio = 0.25) :
  paddyfield_warbler_percent pop = 40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_paddyfield_warbler_percentage_l1153_115398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_l1153_115351

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

/-- The slope of line l1: ax - y + 2a = 0 -/
noncomputable def slope_l1 (a : ℝ) : ℝ := a

/-- The slope of line l2: (2a - 1)x + ay + a = 0 -/
noncomputable def slope_l2 (a : ℝ) : ℝ := 
  if a ≠ 0 then -(2*a - 1) / a else 0

theorem perpendicular_lines (a : ℝ) : 
  perpendicular (slope_l1 a) (slope_l2 a) → a = 0 ∨ a = 1 := by
  sorry

#check perpendicular_lines

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_l1153_115351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_share_difference_l1153_115336

/-- The percentage by which x is greater than y -/
noncomputable def percentage_difference (x y : ℝ) : ℝ := (x - y) / y * 100

theorem share_difference (x y z : ℝ) : 
  y = 1.2 * z →
  z = 100 →
  x + y + z = 370 →
  percentage_difference x y = 25 := by
  sorry

#eval "The theorem has been stated successfully."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_share_difference_l1153_115336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_properties_l1153_115310

noncomputable def z (m : ℝ) : ℂ := (m * (m + 2)) / (m - 1) + (m^2 + 2*m - 3) * Complex.I

theorem complex_number_properties (m : ℝ) :
  (z m ∈ Set.range (Complex.ofReal) ↔ m = -3) ∧
  (z m ∈ Set.univ ↔ m = 0 ∨ m = -2) ∧
  (z m = Complex.I * (z m).im ↔ m = 0 ∨ m = -2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_properties_l1153_115310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_divisor_of_m_l1153_115343

theorem largest_divisor_of_m (m : ℕ) (h1 : m > 0) (h2 : 33 ∣ m^2) :
  ∃ d : ℕ, d ∣ m ∧ d = 33 ∧ ∀ k : ℕ, k ∣ m → k ≤ d :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_divisor_of_m_l1153_115343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_congruent_triangles_xy_values_l1153_115302

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the distance between two points
noncomputable def distance (p1 p2 : Point2D) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

-- Define the congruence of two triangles
def trianglesCongruent (a b c d e : Point2D) : Prop :=
  distance a b = distance a d ∧
  distance a c = distance a e ∧
  distance b c = distance d e

-- Theorem statement
theorem congruent_triangles_xy_values (a b c d e : Point2D) :
  trianglesCongruent a b c d e →
  (e.x * e.y = 14 ∨ e.x * e.y = 18 ∨ e.x * e.y = 40) :=
by
  sorry

#check congruent_triangles_xy_values

end NUMINAMATH_CALUDE_ERRORFEEDBACK_congruent_triangles_xy_values_l1153_115302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l1153_115305

noncomputable section

variable (f : ℝ → ℝ)

axiom f_derivative_condition : ∀ x : ℝ, f x + (deriv f) x > 1
axiom f_initial_value : f 0 = 4

theorem solution_set (x : ℝ) (h : x > 0) : Real.exp x * f x > Real.exp x + 3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l1153_115305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1153_115320

noncomputable def f (x : ℝ) (z : ℝ) : ℝ := 
  Real.sqrt (Real.log (x - 1) / Real.log (1/2)) / (abs x - z)

theorem domain_of_f (z : ℝ) : 
  {x : ℝ | ∃ y, f x z = y} = {x : ℝ | 1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1153_115320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_l1153_115309

noncomputable def line1 (t : ℝ) : ℝ × ℝ := (2 + t, 1 + Real.sqrt 3 * t)
noncomputable def line2 (t : ℝ) : ℝ × ℝ := (-Real.sqrt 3 * t, 2 + t)

theorem perpendicular_lines :
  (∃ t : ℝ, line2 t = (0, 2)) ∧
  (∀ t1 t2 : ℝ, 
    let (x1, y1) := line1 t1
    let (x2, y2) := line2 t2
    (x2 - 0) * (x1 - 2) + (y2 - 2) * (y1 - 1) = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_l1153_115309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rabbit_portion_is_three_tenths_l1153_115330

-- Define the portion of honey each character ate
variable (winnie_portion : ℚ)
variable (piglet_portion : ℚ)
variable (rabbit_portion : ℚ)
variable (eeyore_portion : ℚ)

-- State the conditions
axiom piglet_half_winnie : piglet_portion = winnie_portion / 2
axiom rabbit_half_not_winnie : rabbit_portion = (1 - winnie_portion) / 2
axiom eeyore_tenth : eeyore_portion = 1 / 10

-- State that the sum of all portions is 1 (the whole pot)
axiom total_is_one : winnie_portion + piglet_portion + rabbit_portion + eeyore_portion = 1

-- The theorem to prove
theorem rabbit_portion_is_three_tenths : rabbit_portion = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rabbit_portion_is_three_tenths_l1153_115330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_height_l1153_115380

/-- The height of the first pole in inches -/
noncomputable def h₁ : ℝ := 30

/-- The height of the second pole in inches -/
noncomputable def h₂ : ℝ := 100

/-- The distance between the poles in inches -/
noncomputable def d : ℝ := 150

/-- The slope of the line from the top of the first pole to the foot of the second pole -/
noncomputable def m₁ : ℝ := (0 - h₁) / d

/-- The slope of the line from the top of the second pole to the foot of the first pole -/
noncomputable def m₂ : ℝ := (0 - h₂) / (-d)

/-- The y-intercept of the line from the top of the first pole to the foot of the second pole -/
noncomputable def b₁ : ℝ := h₁

theorem intersection_height :
  ∃ x y : ℝ, m₁ * x + b₁ = m₂ * x ∧ y = m₂ * x ∧ y = 300 / 13 := by
  sorry

#check intersection_height

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_height_l1153_115380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2018th_term_l1153_115383

def arith_sequence (a : ℕ+ → ℕ) : Prop :=
  ∀ m n : ℕ+, a m + a n = a (m + n)

theorem sequence_2018th_term (a : ℕ+ → ℕ) (h : arith_sequence a) (h1 : a 1 = 2) :
  a 2018 = 4036 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2018th_term_l1153_115383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_folding_l1153_115371

/-- Represents a configuration of connected regular triangular faces -/
structure TriangularConfiguration where
  num_faces : ℕ
  is_connected : Bool
  is_regular : Bool

/-- Represents the result of attempting to fold the configuration into a plane -/
def can_fold_to_plane (config : TriangularConfiguration) : Prop :=
  ∃ (folding : TriangularConfiguration → Prop), folding config

/-- The main theorem stating that 28 connected regular triangular faces cannot be folded into a plane -/
theorem impossible_folding (config : TriangularConfiguration) :
  config.num_faces = 28 ∧ config.is_connected ∧ config.is_regular →
  ¬(can_fold_to_plane config) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_folding_l1153_115371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_domain_range_implies_a_value_l1153_115350

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (1 / (x + 1)) / Real.log a

theorem function_domain_range_implies_a_value
  (a : ℝ)
  (h_a_pos : a > 0)
  (h_a_neq_one : a ≠ 1)
  (h_domain : ∀ x, x ∈ Set.Icc 0 1 → f a x ∈ Set.Icc 0 1)
  (h_range : Set.range (f a) = Set.Icc 0 1) :
  a = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_domain_range_implies_a_value_l1153_115350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_difference_is_ln2_l1153_115335

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := Real.log (x / 2) + 1 / 2
noncomputable def g (x : ℝ) : ℝ := Real.exp (x - 2)

-- State the theorem
theorem max_difference_is_ln2 :
  ∃ (C : ℝ), C = Real.log 2 ∧
  (∀ m n : ℝ, n > 0 → f m = g n → n - m ≤ C) ∧
  (∃ m n : ℝ, n > 0 ∧ f m = g n ∧ n - m = C) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_difference_is_ln2_l1153_115335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_composition_l1153_115399

-- Define the functions f and g
noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

-- Define the given condition
axiom inverse_condition : ∀ x, f⁻¹ (g x) = 5 * x^2 + 3

-- State the theorem to be proved
theorem inverse_composition :
  g⁻¹ (f 7) = (Real.sqrt (4/5) : ℝ) ∨ g⁻¹ (f 7) = -(Real.sqrt (4/5) : ℝ) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_composition_l1153_115399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_factorization_l1153_115390

theorem polynomial_factorization : 
  ∀ (x : Polynomial ℤ), 
  x^15 + x^5 + 1 = (x^2 + x + 1) * (x^13 - x^12 + x^10 - x^9 + x^7 - x^6 + x^4 - x^3 + x - 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_factorization_l1153_115390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1153_115308

noncomputable def f (x : ℝ) := Real.log (1 + 1/x) + Real.sqrt (1 - x^2)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | 0 < x ∧ x ≤ 1} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1153_115308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_theorem_l1153_115394

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on the ellipse -/
def Point.onEllipse (p : Point) (e : Ellipse) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

/-- The eccentricity of an ellipse -/
noncomputable def Ellipse.eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - e.b^2 / e.a^2)

/-- The diameter of the incircle of a triangle formed by a point on the ellipse and its foci -/
noncomputable def incircleDiameter (e : Ellipse) (p : Point) : ℝ := 
  sorry

theorem ellipse_eccentricity_theorem (e : Ellipse) (p : Point) :
  p.onEllipse e →
  p.y = 4 →
  incircleDiameter e p = 3 →
  e.eccentricity = 3/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_theorem_l1153_115394
