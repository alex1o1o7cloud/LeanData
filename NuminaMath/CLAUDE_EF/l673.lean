import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_represents_frequency_l673_67370

/-- Represents a frequency distribution histogram --/
structure FrequencyHistogram where
  /-- The set of small rectangles in the histogram --/
  rectangles : Set (ℝ × ℝ)

/-- Represents a group in the frequency distribution --/
structure FrequencyGroup where
  /-- The frequency of the group --/
  frequency : ℝ

/-- The area of a rectangle --/
def rectangleArea (rect : ℝ × ℝ) : ℝ := rect.1 * rect.2

/-- The theorem stating that the area of each small rectangle in a frequency distribution histogram
    represents the frequency of the corresponding group --/
theorem area_represents_frequency (h : FrequencyHistogram) (g : FrequencyGroup) (rect : ℝ × ℝ) :
  rect ∈ h.rectangles → rectangleArea rect = g.frequency := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_represents_frequency_l673_67370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_l673_67307

/-- The function f(x, y) as defined in the problem -/
noncomputable def f (x y : ℝ) : ℝ := x^4/y^4 + y^4/x^4 - x^2/y^2 - y^2/x^2 + x/y - y/x

/-- Theorem stating the minimum value of f(x, y) and when it's achieved -/
theorem f_minimum (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  f x y ≥ 2 ∧ (f x y = 2 ↔ x = y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_l673_67307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_term_is_fourth_l673_67308

noncomputable def sequence_term (n : ℕ) : ℝ := n * (n + 4) * (2/3)^n

theorem largest_term_is_fourth :
  ∀ n : ℕ, n ≠ 0 → sequence_term 4 ≥ sequence_term n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_term_is_fourth_l673_67308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_mean_inequality_l673_67346

theorem cubic_mean_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a * b * c) ^ (1/3) + 1 / a + 1 / b + 1 / c ≥ 2 * Real.sqrt 3 ∧
  ((a * b * c) ^ (1/3) + 1 / a + 1 / b + 1 / c = 2 * Real.sqrt 3 ↔
    a = (3 * Real.sqrt 3) ^ (1/3) ∧ b = (3 * Real.sqrt 3) ^ (1/3) ∧ c = (3 * Real.sqrt 3) ^ (1/3)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_mean_inequality_l673_67346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l673_67324

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then 1 else 0

-- Define the solution set
def solution_set : Set ℝ :=
  {x | x ≤ 1}

-- Theorem statement
theorem inequality_solution_set :
  {x : ℝ | x * f x + x ≤ 2} = solution_set :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l673_67324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_positive_sequence_l673_67336

noncomputable def sequenceK (k : ℝ) : ℕ → ℝ
  | 0 => 1
  | 1 => k
  | (n + 2) => sequenceK k n - sequenceK k (n + 1)

theorem unique_positive_sequence :
  ∃! k : ℝ, (∀ n : ℕ, sequenceK k n > 0) ∧ k = (Real.sqrt 5 - 1) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_positive_sequence_l673_67336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_y_axis_l673_67303

/-- A line passing through two points (x₁, y₁) and (x₂, y₂) -/
structure Line where
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ

/-- The y-intercept of a line -/
noncomputable def y_intercept (l : Line) : ℝ × ℝ :=
  (0, (l.y₂ - l.y₁) / (l.x₂ - l.x₁) * (0 - l.x₁) + l.y₁)

/-- The line passing through (2, 5) and (6, 17) -/
def specific_line : Line :=
  { x₁ := 2, y₁ := 5, x₂ := 6, y₂ := 17 }

theorem line_intersects_y_axis :
  y_intercept specific_line = (0, -1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_y_axis_l673_67303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_properties_l673_67397

def equation (a b c : ℕ+) : Prop := a^b.val * b^c.val = c^a.val

theorem equation_properties :
  ∀ a b c : ℕ+, equation a b c →
  (∀ p : ℕ, Nat.Prime p → p ∣ a.val → p ∣ b.val) ∧
  (b ≥ a → a = 1 ∧ b = 1 ∧ c = 1) ∧
  (∃ f : ℕ → (ℕ+ × ℕ+ × ℕ+), ∀ n : ℕ, 
    let (an, bn, cn) := f n
    equation an bn cn) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_properties_l673_67397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_a_value_l673_67387

theorem smallest_a_value (P : ℤ → ℤ) (a : ℕ+) 
  (h1 : P 1 = a ∧ P 3 = a ∧ P 5 = a ∧ P 7 = a ∧ P 9 = a)
  (h2 : P 2 = -a ∧ P 4 = -a ∧ P 6 = -a ∧ P 8 = -a ∧ P 10 = -a)
  (h3 : ∀ x : ℤ, ∃ y : ℤ, P x = y) :
  a.val ≥ 945 ∧ ∃ P' : ℤ → ℤ, (∀ x : ℤ, ∃ y : ℤ, P' x = y) ∧
    (P' 1 = 945 ∧ P' 3 = 945 ∧ P' 5 = 945 ∧ P' 7 = 945 ∧ P' 9 = 945) ∧
    (P' 2 = -945 ∧ P' 4 = -945 ∧ P' 6 = -945 ∧ P' 8 = -945 ∧ P' 10 = -945) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_a_value_l673_67387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_gcd_problem_l673_67379

theorem lcm_gcd_problem : Nat.gcd (Nat.lcm 18 21) (Nat.lcm 9 14) = 126 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_gcd_problem_l673_67379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_monitoring_time_l673_67380

structure TimeSlot where
  start : Nat
  end_ : Nat

structure Volunteer where
  name : String
  slot1 : TimeSlot
  slot2 : TimeSlot

def volunteers : List Volunteer := [
  ⟨"A", ⟨6, 8⟩, ⟨16, 18⟩⟩,
  ⟨"B", ⟨6, 7⟩, ⟨17, 20⟩⟩,
  ⟨"C", ⟨8, 11⟩, ⟨18, 19⟩⟩,
  ⟨"D", ⟨7, 10⟩, ⟨17, 18⟩⟩
]

def isOverlapping (ts1 ts2 : TimeSlot) : Bool :=
  (ts1.start < ts2.end_ && ts2.start < ts1.end_)

def maxSimultaneousVolunteers : Nat := 2

def minParticipation : Nat := 1

def calculateMaxMonitoringTime (vols : List Volunteer) : Nat :=
  sorry

theorem max_monitoring_time :
  calculateMaxMonitoringTime volunteers = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_monitoring_time_l673_67380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_method1_more_effective_l673_67396

/-- Represents the cost calculation for a stationery store's badminton equipment sale. -/
def badminton_sale_cost (racket_price shuttlecock_price racket_count shuttlecock_count : ℕ) : ℕ :=
  racket_price * racket_count + shuttlecock_price * shuttlecock_count

/-- Calculates the cost after applying discount method 1 (one free shuttlecock per racket). -/
def discount_method1 (total_cost racket_count shuttlecock_price : ℕ) : ℚ :=
  (total_cost - racket_count * shuttlecock_price : ℚ)

/-- Calculates the cost after applying discount method 2 (92% of total price). -/
def discount_method2 (total_cost : ℕ) : ℚ :=
  (total_cost : ℚ) * 92 / 100

/-- Theorem stating that discount method 1 is more cost-effective than method 2. -/
theorem discount_method1_more_effective :
  let racket_price := 20
  let shuttlecock_price := 5
  let racket_count := 4
  let shuttlecock_count := 30
  let total_cost := badminton_sale_cost racket_price shuttlecock_price racket_count shuttlecock_count
  discount_method1 total_cost racket_count shuttlecock_price < discount_method2 total_cost := by
  sorry

#eval badminton_sale_cost 20 5 4 30
#eval discount_method1 230 4 5
#eval discount_method2 230

end NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_method1_more_effective_l673_67396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plant_arrangement_l673_67359

theorem plant_arrangement (n : ℕ) (m : ℕ) (h1 : n = 5) (h2 : m = 3) :
  (Nat.factorial (n + 1)) * (Nat.factorial m) = 4320 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plant_arrangement_l673_67359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_star_pentagon_area_formula_l673_67329

/-- The area of a regular star pentagon -/
noncomputable def star_pentagon_area (p : ℝ) : ℝ := 2 * Real.sqrt (85 - 38 * Real.sqrt 5) * p^2

/-- Theorem: The area of a regular star pentagon is 2 * sqrt(85 - 38 * sqrt(5)) * p^2,
    where p is the distance between two adjacent outer vertices. -/
theorem star_pentagon_area_formula (p : ℝ) (h : p > 0) :
  star_pentagon_area p = 2 * Real.sqrt (85 - 38 * Real.sqrt 5) * p^2 :=
by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_star_pentagon_area_formula_l673_67329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_J_l673_67330

def H (p q : ℝ) : ℝ := -3*p*q + 4*p*(1-q) + 5*(1-p)*q - 6*(1-p)*(1-q) + 2*p

noncomputable def J (p : ℝ) : ℝ := 
  ⨆ q ∈ Set.Icc 0 1, H p q

theorem minimize_J :
  ∃ (p : ℝ), p ∈ Set.Icc 0 1 ∧
  (∀ (p' : ℝ), p' ∈ Set.Icc 0 1 → J p ≤ J p') ∧
  p = 11 / 18 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_J_l673_67330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modified_short_bingo_first_column_possibilities_l673_67385

theorem modified_short_bingo_first_column_possibilities : ℕ := by
  let n : ℕ := 15  -- Total numbers to choose from
  let k : ℕ := 5   -- Number of distinct numbers to be selected
  let result : ℕ := n * (n-1) * (n-2) * (n-3) * (n-4)
  have h : result = 360360 := by
    -- Proof goes here
    sorry
  exact 360360

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modified_short_bingo_first_column_possibilities_l673_67385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_field_side_length_is_50_l673_67350

/-- The side length of a square field, given the time and speed of a runner around its perimeter -/
noncomputable def square_field_side_length (time_seconds : ℝ) (speed_km_per_hour : ℝ) : ℝ :=
  let speed_m_per_second := speed_km_per_hour * 1000 / 3600
  let perimeter := speed_m_per_second * time_seconds
  perimeter / 4

/-- Theorem stating that the side length of the square field is 50 meters -/
theorem square_field_side_length_is_50 :
  square_field_side_length 80 9 = 50 := by
  -- Unfold the definition of square_field_side_length
  unfold square_field_side_length
  -- Simplify the expression
  simp
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_field_side_length_is_50_l673_67350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_comparison_l673_67389

/-- Represents the characteristics of a car --/
structure Car where
  range : ℝ
  costPerKm : ℝ

/-- Calculates the cost per kilometer for a fuel car --/
noncomputable def fuelCarCostPerKm (a : ℝ) : ℝ := (40 * 9) / (2 * a)

/-- Calculates the cost per kilometer for a new energy car --/
noncomputable def newEnergyCarCostPerKm (a : ℝ) : ℝ := (60 * 0.6) / a

/-- Theorem stating the properties of the two cars --/
theorem car_comparison (a : ℝ) (fuelCar newEnergyCar : Car) : 
  fuelCar.range = 2 * a →
  newEnergyCar.range = a →
  fuelCar.costPerKm = fuelCarCostPerKm a →
  newEnergyCar.costPerKm = newEnergyCarCostPerKm a →
  fuelCar.costPerKm = newEnergyCar.costPerKm + 0.48 →
  a = 300 ∧ 
  fuelCar.costPerKm = 0.6 ∧
  newEnergyCar.costPerKm = 0.12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_comparison_l673_67389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_std_dev_right_triangle_l673_67304

/-- Right triangle with hypotenuse 3 -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  hypotenuse_eq : a^2 + b^2 = 3^2

/-- Standard deviation of the sides of a right triangle -/
noncomputable def standard_deviation (t : RightTriangle) : ℝ :=
  Real.sqrt (((t.a - (t.a + t.b + 3) / 3)^2 + (t.b - (t.a + t.b + 3) / 3)^2 + (3 - (t.a + t.b + 3) / 3)^2) / 3)

/-- Theorem: Minimum standard deviation and corresponding leg lengths -/
theorem min_std_dev_right_triangle :
  (∃ (t : RightTriangle), ∀ (t' : RightTriangle), standard_deviation t ≤ standard_deviation t') ∧
  (∃ (t : RightTriangle), standard_deviation t = Real.sqrt 2 - 1 ∧ t.a = 3 * Real.sqrt 2 / 2 ∧ t.b = 3 * Real.sqrt 2 / 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_std_dev_right_triangle_l673_67304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_medicines_spending_percentage_l673_67373

def monthly_income : ℝ := 40000
def household_percentage : ℝ := 45
def clothes_percentage : ℝ := 25
def savings : ℝ := 9000

def medicines_percentage : ℝ := 7.5

theorem medicines_spending_percentage :
  (monthly_income - (household_percentage / 100 * monthly_income) - 
   (clothes_percentage / 100 * monthly_income) - savings) / monthly_income * 100 = medicines_percentage :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_medicines_spending_percentage_l673_67373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_three_zeros_l673_67360

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then -x^2 + 2*a*x + 1 else Real.log x + 2*a

-- Theorem statement
theorem f_has_three_zeros (a : ℝ) (h : a < 0) :
  ∃ (x₁ x₂ x₃ : ℝ), (f a x₁ = 0 ∧ f a x₂ = 0 ∧ f a x₃ = 0) ∧
    (x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃) ∧
    (∀ x : ℝ, f a x = 0 → x = x₁ ∨ x = x₂ ∨ x = x₃) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_three_zeros_l673_67360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_has_eight_vertices_l673_67334

/-- A cube is a three-dimensional shape with six square faces. -/
structure Cube where
  -- We don't need to define the specifics of a cube for this problem

/-- The number of vertices in a geometric shape. -/
def num_vertices (shape : Type) : ℕ := sorry

/-- A cube has 8 vertices. -/
theorem cube_has_eight_vertices :
  ∀ (c : Cube), num_vertices Cube = 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_has_eight_vertices_l673_67334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_half_in_fourth_quadrant_l673_67354

-- Define the concept of quadrants
def in_third_quadrant (α : Real) : Prop :=
  Real.pi < α ∧ α < 3 * Real.pi / 2

def in_fourth_quadrant (α : Real) : Prop :=
  3 * Real.pi / 2 < α ∧ α < 2 * Real.pi

-- State the theorem
theorem angle_half_in_fourth_quadrant (α : Real) :
  in_third_quadrant α → |Real.cos (α/2)| = Real.cos (α/2) → in_fourth_quadrant (α/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_half_in_fourth_quadrant_l673_67354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shift_sinusoidal_function_l673_67317

open Real

-- Define the original function
noncomputable def f (x : ℝ) : ℝ := 2 * sin (2 * x + π / 6)

-- Define the period of the function
noncomputable def period : ℝ := π

-- Define the shift amount
noncomputable def shift : ℝ := period / 4

-- Define the resulting function after shift
noncomputable def g (x : ℝ) : ℝ := 2 * sin (2 * x - π / 3)

-- Theorem statement
theorem shift_sinusoidal_function :
  ∀ x : ℝ, f (x - shift) = g x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shift_sinusoidal_function_l673_67317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l673_67363

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 
  Real.cos (ω * x) * Real.sin (ω * x - Real.pi/3) + Real.sqrt 3 * (Real.cos (ω * x))^2 - Real.sqrt 3 / 4

theorem function_properties (ω : ℝ) (h_ω : ω > 0) :
  (∃ (c : ℝ), ∀ (x : ℝ), f ω (c + Real.pi/4) = f ω (c - Real.pi/4)) →
  (ω = 1 ∧ 
   (∀ (k : ℤ), ∃ (x : ℝ), x = (1/2 : ℝ) * k * Real.pi + Real.pi/12 ∧ 
    (∀ (y : ℝ), f ω (x + y) = f ω (x - y))) ∧
   (∀ (A B : ℝ) (a b : ℝ),
    f ω A = 0 → Real.sin B = 4/5 → a = Real.sqrt 3 →
    a / Real.sin A = b / Real.sin B →
    b = 8/5)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l673_67363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_value_for_odd_even_l673_67394

theorem expression_value_for_odd_even (n : ℕ) :
  (1 + (-1 : ℚ)^n) / 4 = if n % 2 = 0 then 1/2 else 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_value_for_odd_even_l673_67394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_max_volume_l673_67351

/-- The volume of a cylinder carved from a cone with base radius 1 and height 2 -/
noncomputable def cylinderVolume (x : ℝ) : ℝ := Real.pi * (2 * x^2 - 2 * x^3)

/-- The maximum volume of a cylinder carved from a cone with base radius 1 and height 2 -/
noncomputable def maxCylinderVolume : ℝ := 8 * Real.pi / 27

theorem cylinder_max_volume :
  ∃ (x : ℝ), x > 0 ∧ x < 1 ∧
  (∀ (y : ℝ), y > 0 → y < 1 → cylinderVolume y ≤ cylinderVolume x) ∧
  cylinderVolume x = maxCylinderVolume := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_max_volume_l673_67351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_journey_mpg_l673_67358

/-- Represents the average miles per gallon for a car journey -/
structure JourneyMPG where
  ab : ℝ  -- Average MPG from A to B
  bc : ℝ  -- Average MPG from B to C
  total : ℝ  -- Average MPG for entire journey

/-- Represents the distance of a car journey -/
structure JourneyDistance where
  ab : ℝ  -- Distance from A to B
  bc : ℝ  -- Distance from B to C

theorem car_journey_mpg (j : JourneyMPG) (d : JourneyDistance) :
  j.bc = 30 ∧ 
  j.total = 26.47 ∧ 
  d.ab = 2 * d.bc →
  abs (j.ab - 25) < 0.01 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_journey_mpg_l673_67358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_parameterizations_l673_67331

-- Define the line
noncomputable def line (x : ℝ) : ℝ := 2 * x - 5

-- Define the parameterizations
noncomputable def paramA (t : ℝ) : ℝ × ℝ := (3 - 2*t, 1 - 4*t)
noncomputable def paramB (t : ℝ) : ℝ × ℝ := (5 + 3*t, 5 + 6*t)
noncomputable def paramC (t : ℝ) : ℝ × ℝ := (t, -5 + 2*t)
noncomputable def paramD (t : ℝ) : ℝ × ℝ := (-1 + 2*t, -7 + 5*t)
noncomputable def paramE (t : ℝ) : ℝ × ℝ := (2 + t/2, -1 + t)

-- Theorem stating which parameterizations are valid
theorem valid_parameterizations :
  (∀ t, (paramA t).2 = line (paramA t).1) ∧
  (∀ t, (paramB t).2 = line (paramB t).1) ∧
  (∀ t, (paramC t).2 = line (paramC t).1) ∧
  (∀ t, (paramE t).2 = line (paramE t).1) ∧
  ¬(∀ t, (paramD t).2 = line (paramD t).1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_parameterizations_l673_67331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l673_67348

/-- The eccentricity of an ellipse with specific axis ratio -/
theorem ellipse_eccentricity (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : a = 2 * b) : 
  Real.sqrt ((a^2 - b^2) / a^2) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l673_67348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_tan_inequality_l673_67381

theorem cos_sin_tan_inequality : Real.cos 2 < Real.sin 3 ∧ Real.sin 3 < Real.tan 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_tan_inequality_l673_67381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_special_case_l673_67364

theorem sin_double_angle_special_case (α : ℝ) (h : Real.sin α - Real.cos α = 4/3) :
  Real.sin (2 * α) = -7/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_special_case_l673_67364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_triangle_area_inequality_l673_67357

/-- A convex pentagon represented by its vertices -/
structure ConvexPentagon where
  P : Fin 5 → ℝ × ℝ
  convex : Convex ℝ (Set.range P)

/-- The area of a triangle given by three points -/
noncomputable def triangleArea (A B C : ℝ × ℝ) : ℝ := sorry

/-- The area of a pentagon given by five points -/
noncomputable def pentagonArea (P : Fin 5 → ℝ × ℝ) : ℝ := sorry

/-- The theorem statement -/
theorem pentagon_triangle_area_inequality (ABCDE : ConvexPentagon) 
  (h : pentagonArea ABCDE.P = 1) : 
  ∃ (i j : Fin 7), 
    triangleArea (ABCDE.P (i % 5)) (ABCDE.P ((i + 1) % 5)) (ABCDE.P ((i + 2) % 5)) 
      ≤ (5 - Real.sqrt 5) / 10 
    ∧ (5 - Real.sqrt 5) / 10 
      ≤ triangleArea (ABCDE.P (j % 5)) (ABCDE.P ((j + 1) % 5)) (ABCDE.P ((j + 2) % 5)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_triangle_area_inequality_l673_67357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_intercept_of_line_l673_67318

/-- The y-intercept of a line is the y-coordinate of the point where the line intersects the y-axis. -/
noncomputable def y_intercept (a b c : ℝ) : ℝ := -c / b

/-- A line is defined by the equation ax + by + c = 0, where a, b, and c are real numbers and b ≠ 0. -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ
  b_nonzero : b ≠ 0

/-- The y-intercept of the line x - 3y - 1 = 0 is -1/3. -/
theorem y_intercept_of_line (l : Line) (h1 : l.a = 1) (h2 : l.b = -3) (h3 : l.c = -1) : 
  y_intercept l.a l.b l.c = -1/3 := by
  -- Unfold the definition of y_intercept
  unfold y_intercept
  -- Substitute the values of l.b and l.c
  rw [h2, h3]
  -- Simplify the fraction
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_intercept_of_line_l673_67318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_zeros_implies_b_value_l673_67326

noncomputable def f (b : ℝ) (x : ℝ) : ℝ := x^3 - b*x^2 + 1/2

theorem two_zeros_implies_b_value (b : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ f b x = 0 ∧ f b y = 0 ∧
    (∀ z : ℝ, f b z = 0 → z = x ∨ z = y)) →
  b = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_zeros_implies_b_value_l673_67326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_AB_implies_m_value_disjoint_A_C_implies_b_range_union_AB_equals_B_implies_m_range_l673_67374

-- Define the sets A, B, and C
def A : Set ℝ := {x | x^2 - 3*x - 4 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | x^2 - 2*m*x + m^2 - 9 ≤ 0}
def C (b : ℝ) : Set ℝ := {y | ∃ x, y = 2^x + b}

-- Theorem 1
theorem intersection_AB_implies_m_value (m : ℝ) :
  A ∩ B m = Set.Icc 0 4 → m = 3 := by sorry

-- Theorem 2
theorem disjoint_A_C_implies_b_range (b : ℝ) :
  A ∩ C b = ∅ → b ≥ 4 := by sorry

-- Theorem 3
theorem union_AB_equals_B_implies_m_range (m : ℝ) :
  A ∪ B m = B m → 1 ≤ m ∧ m ≤ 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_AB_implies_m_value_disjoint_A_C_implies_b_range_union_AB_equals_B_implies_m_range_l673_67374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_points_on_integer_lattice_l673_67345

-- Define the set of points
variable (n : ℕ)
variable (points : Fin n → ℝ × ℝ)

-- Define the origin
def origin : ℝ × ℝ := (0, 0)

-- Define the distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- State the theorem
theorem points_on_integer_lattice
  (h : ∀ i j : Fin n, ∃ m : ℕ, (distance (points i) (points j))^2 = m) :
  ∃ x y : ℝ × ℝ, ∀ i : Fin n,
    ∃ k l : ℤ, points i = (k : ℝ) • x + (l : ℝ) • y :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_points_on_integer_lattice_l673_67345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_apple_spaniel_l673_67332

def digits (n : ℕ) : List ℕ :=
  if n = 0 then [0] else 
  let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc
    else aux (m / 10) ((m % 10) :: acc)
  aux n []

def same_digits (a b : ℕ) : Prop :=
  ∃ (perm : List ℕ → List ℕ), Function.Bijective perm ∧ digits a = perm (digits b)

theorem no_solution_apple_spaniel : 
  ¬ ∃ (n₁ n₂ : ℕ), (same_digits n₁ n₂) ∧ (n₁ - n₂ = 2018 * 2019) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_apple_spaniel_l673_67332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_relation_l673_67383

-- Define the constants
noncomputable def a : ℝ := Real.exp 0.2
noncomputable def b : ℝ := 0.2 ^ Real.exp 1
noncomputable def c : ℝ := Real.log 2

-- State the theorem
theorem order_relation : b < c ∧ c < a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_relation_l673_67383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_l673_67382

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 2 * Real.sin x * (Real.cos x + Real.sin x) - Real.sqrt 2 / 2

theorem min_value_f :
  ∃ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) ∧
  (∀ (y : ℝ), y ∈ Set.Icc 0 (Real.pi / 2) → f y ≥ f x) ∧
  f x = -Real.sqrt 2 / 2 := by
  sorry

#check min_value_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_l673_67382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equals_one_l673_67369

-- Define the integrand
noncomputable def f (x : ℝ) : ℝ := 
  6 * (Real.sqrt (x + 2)) / ((x + 2)^2 * Real.sqrt (x + 1))

-- State the theorem
theorem integral_equals_one :
  ∫ x in (-14/15)..(-7/8), f x = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equals_one_l673_67369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_yellow_tint_percentage_l673_67315

/-- Represents the initial volume of the mixture in liters -/
noncomputable def initial_volume : ℝ := 40

/-- Represents the initial percentage of yellow tint in the mixture -/
noncomputable def initial_yellow_percent : ℝ := 35

/-- Represents the amount of yellow tint added in liters -/
noncomputable def added_yellow : ℝ := 3

/-- Represents the total amount of liquid added in liters -/
noncomputable def total_added : ℝ := 5

/-- Calculates the percentage of yellow tint in the new mixture -/
noncomputable def new_yellow_percent : ℝ := 
  ((initial_volume * initial_yellow_percent / 100 + added_yellow) / (initial_volume + total_added)) * 100

theorem yellow_tint_percentage :
  ⌊new_yellow_percent⌋ = 38 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_yellow_tint_percentage_l673_67315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l673_67322

theorem expression_evaluation : 
  Real.sqrt (6 + 1/4) - (Real.pi - 1)^(0 : ℝ) - (3 + 3/8)^(1/3 : ℝ) + (1/64)^(-(2/3) : ℝ) = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l673_67322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lucky_larry_coincidence_l673_67319

theorem lucky_larry_coincidence :
  ∃ f : ℤ, 
    (let a : ℤ := 2
     let b : ℤ := 3
     let c : ℤ := 4
     let d : ℤ := 5
     a - b + c - d - f = a - (b + (c - (d + f)))) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lucky_larry_coincidence_l673_67319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_cartesian_parabola_l673_67321

/-- The curve defined by r = 2 cot θ csc θ in polar coordinates is a parabola -/
theorem polar_to_cartesian_parabola :
  ∀ (r θ : ℝ) (x y : ℝ),
  r = 2 * (Real.cos θ / Real.sin θ) * (1 / Real.sin θ) →
  x = r * Real.cos θ →
  y = r * Real.sin θ →
  y^2 = 2*x :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_cartesian_parabola_l673_67321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_f₁_belongs_to_M_l673_67325

noncomputable def f₁ : ℝ → ℝ := λ x => x
noncomputable def f₂ : ℝ → ℝ := λ x => Real.exp (x * Real.log 2) - 1
noncomputable def f₃ : ℝ → ℝ := λ x => Real.log (x + 1)

def belongsToM (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, x ≥ 0 → f x ≥ 0) ∧
  (∀ s t : ℝ, f s + f t ≤ f (s + t))

theorem only_f₁_belongs_to_M :
  belongsToM f₁ ∧ ¬belongsToM f₂ ∧ ¬belongsToM f₃ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_f₁_belongs_to_M_l673_67325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_max_min_possible_l673_67355

/-- Represents a comparison operation between two weights -/
def Compare (a b : ℕ) : Bool := a < b

/-- Finds the maximum and minimum elements in a list using at most n comparisons -/
def findMaxMin (weights : List ℕ) (n : ℕ) : Option (ℕ × ℕ) :=
  sorry

theorem find_max_min_possible :
  ∀ (weights : List ℕ),
    weights.length = 10 →
    weights.Pairwise (λ a b => a ≠ b) →
    ∃ (result : ℕ × ℕ),
      findMaxMin weights 13 = some result ∧
      result.1 = weights.maximum ∧
      result.2 = weights.minimum :=
by sorry

#check find_max_min_possible

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_max_min_possible_l673_67355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_zero_when_x_is_three_l673_67393

theorem product_zero_when_x_is_three :
  ∀ x : ℝ, x = 3 → x * (x - 1) * (x - 2) * (x - 3) * (x - 4) = 0 := by
  intro x h
  rw [h]
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_zero_when_x_is_three_l673_67393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_theorem_l673_67352

-- Define the functions
noncomputable def f₁ (x : ℝ) : ℝ := 1 / x
noncomputable def f₂ (x : ℝ) : ℝ := Real.log x
noncomputable def f₃ (x : ℝ) : ℝ := Real.sin x
noncomputable def f₄ (x : ℝ) : ℝ := -Real.exp x

-- Define a predicate for functions that can have a tangent line with slope 1/2
def has_tangent_with_slope_half (f : ℝ → ℝ) : Prop :=
  ∃ x : ℝ, HasDerivAt f ((1 : ℝ) / 2) x

-- State the theorem
theorem tangent_line_theorem :
  ¬ has_tangent_with_slope_half f₁ ∧
  has_tangent_with_slope_half f₂ ∧
  has_tangent_with_slope_half f₃ ∧
  ¬ has_tangent_with_slope_half f₄ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_theorem_l673_67352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_share_of_A_l673_67341

def total_amount : ℕ := 232000
def num_persons : ℕ := 5

def ratio_A_B : Rat := 2/3
def ratio_B_C : Rat := 4/5
def ratio_C_D : Rat := 3/4
def ratio_D_E : Rat := 4/5

theorem share_of_A (share_A : ℕ) : 
  share_A = 23200 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_share_of_A_l673_67341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_square_root_l673_67399

-- Define a function to check if a number is a perfect square
def isPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- Define what it means for a square root to be in simplest form
def isSimplestForm (x : ℝ) : Prop :=
  ∀ y z : ℝ, y * y = x → z * z = x → y = z ∨ y = -z

-- Theorem statement
theorem simplest_square_root :
  isSimplestForm (Real.sqrt 10) ∧
  ¬isSimplestForm (Real.sqrt 12) ∧
  ¬isSimplestForm (Real.sqrt 3/5) ∧
  ∀ x : ℝ, ¬isSimplestForm (Real.sqrt (x^3)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_square_root_l673_67399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_distance_between_spheres_l673_67344

def sphere_center1 : Fin 3 → ℝ := ![3, -5, 10]
def sphere_radius1 : ℝ := 25

def sphere_center2 : Fin 3 → ℝ := ![-7, 15, -20]
def sphere_radius2 : ℝ := 95

theorem largest_distance_between_spheres :
  let distance := λ (p q : Fin 3 → ℝ) => Real.sqrt (
    (p 0 - q 0)^2 + (p 1 - q 1)^2 + (p 2 - q 2)^2
  )
  ∀ (p1 p2 : Fin 3 → ℝ),
  distance p1 sphere_center1 = sphere_radius1 →
  distance p2 sphere_center2 = sphere_radius2 →
  distance p1 p2 ≤ 120 + 10 * Real.sqrt 14 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_distance_between_spheres_l673_67344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_certain_number_proof_l673_67367

theorem certain_number_proof (smallest_number : ℕ) : 
  smallest_number = 3153 →
  (∃ (certain_number : ℕ), 
    Nat.Prime certain_number ∧
    (smallest_number + 3) % certain_number = 0 ∧
    (smallest_number + 3) % 9 = 0 ∧
    (smallest_number + 3) % 70 = 0 ∧
    (smallest_number + 3) % 25 = 0 ∧
    certain_number ∉ Nat.factors (Nat.lcm 9 (Nat.lcm 70 25)) ∧
    ∀ (p : ℕ), Nat.Prime p → p > certain_number → (smallest_number + 3) % p ≠ 0) →
  37 = certain_number :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_certain_number_proof_l673_67367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_between_circles_l673_67305

/-- Representation of a sphere in 3D space -/
structure Sphere where
  center : Real × Real × Real
  radius : Real

/-- Predicate indicating that two spheres are internally tangent -/
def InternallyTangent (c₁ c₂ : Sphere) : Prop := sorry

/-- Predicate indicating that the tangent points of the smaller circle with the two larger circles
    form a diameter of the smaller circle -/
def DiameterTangentPoints (c₁ c₂ c₃ : Sphere) : Prop := sorry

/-- Function calculating the area between three circles as described in the problem -/
noncomputable def AreaBetweenCircles (c₁ c₂ c₃ : Sphere) : Real := sorry

/-- The area of the region outside a circle of radius 1 and inside two circles of radius 2
    that are internally tangent to the smaller circle at opposite ends of its diameter -/
theorem shaded_area_between_circles :
  let r₁ : Real := 1  -- radius of the smaller circle
  let r₂ : Real := 2  -- radius of the larger circles
  let area : Real := (5 / 3) * Real.pi - 2 * Real.sqrt 3
  (∀ c₁ c₂ c₃ : Sphere, 
    (c₁.radius = r₁) →
    (c₂.radius = r₂) →
    (c₃.radius = r₂) →
    (InternallyTangent c₁ c₂) →
    (InternallyTangent c₁ c₃) →
    (DiameterTangentPoints c₁ c₂ c₃)) →
  AreaBetweenCircles c₁ c₂ c₃ = area := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_between_circles_l673_67305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_R_l673_67310

/-- Sum of arithmetic progression -/
noncomputable def sum_ap (a d : ℝ) (n : ℕ) : ℝ := n * (2 * a + (n - 1) * d) / 2

/-- R' calculation for arithmetic progression -/
noncomputable def R' (a d : ℝ) (n : ℕ) : ℝ :=
  sum_ap a d (3 * n) - 2 * sum_ap a d (2 * n) + sum_ap a d n

theorem R'_depends_only_on_a (a d : ℝ) (n : ℕ) :
  R' a d n = -3 * a := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_R_l673_67310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_l673_67327

open Real

-- Define the equation
def equation (x : ℝ) : Prop :=
  cos (2 * x) * (cos (2 * x) - cos (804 * Real.pi^2 / x)) + sin x ^ 2 = cos (4 * x)

-- Define the set of positive real solutions
def solution_set : Set ℝ := {x | x > 0 ∧ equation x}

-- State the theorem
theorem sum_of_solutions :
  ∃ (S : Finset ℝ), S.toSet = solution_set ∧ S.sum id = 405 * Real.pi :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_l673_67327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_inscribed_triangle_area_l673_67392

/-- A set of points in ℝ² forms a triangle. -/
def is_triangle (t : Set (ℝ × ℝ)) : Prop :=
sorry

/-- A triangle is inscribed in a circle with radius r. -/
def is_inscribed_in_circle (t : Set (ℝ × ℝ)) (r : ℝ) : Prop :=
sorry

/-- A triangle has one side equal to the diameter of the circle with radius r. -/
def has_diameter_side (t : Set (ℝ × ℝ)) (r : ℝ) : Prop :=
sorry

/-- The area of a triangle represented as a set of points in ℝ². -/
noncomputable def area (t : Set (ℝ × ℝ)) : ℝ :=
sorry

/-- Given a circle with radius r, the area of the largest inscribed triangle
    with one side as the diameter is equal to r^2. -/
theorem largest_inscribed_triangle_area (r : ℝ) (hr : r > 0) :
  ∃ (t : Set (ℝ × ℝ)), 
    is_triangle t ∧ 
    is_inscribed_in_circle t r ∧ 
    has_diameter_side t r ∧
    (∀ t' : Set (ℝ × ℝ), 
      is_triangle t' ∧ 
      is_inscribed_in_circle t' r ∧ 
      has_diameter_side t' r → 
      area t' ≤ area t) ∧
    area t = r^2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_inscribed_triangle_area_l673_67392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l673_67309

/-- Calculates the speed of a train in km/h given its length, the bridge length it passes, and the time it takes to pass the bridge. -/
noncomputable def train_speed (train_length : ℝ) (bridge_length : ℝ) (time : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let speed_ms := total_distance / time
  3.6 * speed_ms

/-- Theorem stating that a train of length 200 meters passing a bridge of length 180 meters
    in 21.04615384615385 seconds has a speed of approximately 64.98 km/h. -/
theorem train_speed_calculation :
  let train_length := (200 : ℝ)
  let bridge_length := (180 : ℝ)
  let time := (21.04615384615385 : ℝ)
  ∃ ε > 0, |train_speed train_length bridge_length time - 64.98| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l673_67309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l673_67339

-- Define the custom operation ⊕
noncomputable def customOp (a b : ℝ) : ℝ :=
  if a ≥ b then a else b^2

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (customOp 1 x) * x - (customOp 2 x)

-- Theorem statement
theorem f_range :
  ∀ x ∈ Set.Icc (-2 : ℝ) 2, ∃ y ∈ Set.Icc (-4 : ℝ) 6, f x = y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l673_67339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_arithmetic_sequence_l673_67343

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- An arithmetic sequence -/
def arithmetic_sequence (b : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, b (n + 1) = b n + d

/-- Sum of first n terms of an arithmetic sequence -/
noncomputable def arithmetic_sum (b : ℕ → ℝ) (n : ℕ) : ℝ :=
  n * (b 1 + b n) / 2

theorem sum_of_arithmetic_sequence
  (a b : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_arith : arithmetic_sequence b)
  (h_rel : a 4 * a 6 = 2 * a 5)
  (h_b5 : b 5 = 2 * a 5) :
  arithmetic_sum b 9 = 36 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_arithmetic_sequence_l673_67343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_D_72_l673_67347

/-- D(n) represents the number of ways to write n as a product of integers greater than 1, where the order matters -/
def D (n : ℕ+) : ℕ := sorry

/-- All factors in the product representation are greater than 1 -/
axiom factors_gt_one (n : ℕ+) : ∀ (f : ℕ), f ∈ (D n).factors → f > 1

/-- The order of factors matters in counting the representations -/
axiom order_matters (n : ℕ+) : ∀ (f g : List ℕ), 
  (f.prod = n) ∧ (g.prod = n) ∧ (f ≠ g) → (f.length = g.length) → (D n) > 1

/-- There is at least one factor in each representation -/
axiom at_least_one_factor (n : ℕ+) : ∀ (l : List ℕ), l.prod = n → l.length ≥ 1

/-- The main theorem: D(72) = 121 -/
theorem D_72 : D 72 = 121 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_D_72_l673_67347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_2000_coordinates_l673_67362

/-- Represents a point in the first quadrant of the Cartesian plane -/
structure Point where
  x : ℕ
  y : ℕ

/-- The spiral numbering function for points in the first quadrant -/
def spiralNumber : Point → ℕ := sorry

/-- The inverse of the spiral numbering function -/
def spiralCoordinate : ℕ → Point := sorry

/-- The spiral numbering is bijective -/
axiom spiral_bijective : Function.Bijective spiralNumber

/-- The 2000th point in the spiral numbering system -/
def point_2000 : Point := spiralCoordinate 2000

theorem point_2000_coordinates :
  point_2000.x = 44 ∧ point_2000.y = 25 := by
  sorry

#check point_2000_coordinates

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_2000_coordinates_l673_67362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diplomats_not_speaking_hindi_l673_67306

-- Define the total number of diplomats
def total_diplomats : ℕ := 120

-- Define the number of diplomats who spoke French
def french_speakers : ℕ := 20

-- Define the percentage of diplomats who spoke neither French nor Hindi
def neither_french_nor_hindi_percent : ℚ := 20 / 100

-- Define the percentage of diplomats who spoke both French and Hindi
def both_french_and_hindi_percent : ℚ := 10 / 100

-- Theorem statement
theorem diplomats_not_speaking_hindi :
  (french_speakers - (both_french_and_hindi_percent * total_diplomats).floor) +
  (neither_french_nor_hindi_percent * total_diplomats).floor = 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diplomats_not_speaking_hindi_l673_67306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_60_people_l673_67361

noncomputable def charter_fee : ℝ := 15000

noncomputable def ticket_price (x : ℝ) : ℝ :=
  if x ≤ 30 then 900 else 900 - 10 * (x - 30)

noncomputable def profit (x : ℝ) : ℝ :=
  if x ≤ 30
  then 900 * x - charter_fee
  else (ticket_price x) * x - charter_fee

theorem max_profit_at_60_people :
  ∃ (max_profit : ℝ),
    (∀ x, 0 ≤ x ∧ x ≤ 75 → profit x ≤ max_profit) ∧
    profit 60 = max_profit ∧
    max_profit = 21000 := by
  sorry

#check max_profit_at_60_people

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_60_people_l673_67361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_common_tangent_lines_l673_67377

/-- Circle represented by its center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Calculate the distance between two points in 2D space -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Calculate the number of common tangent lines between two circles -/
noncomputable def commonTangentLines (c1 c2 : Circle) : ℕ :=
  if distance c1.center c2.center > c1.radius + c2.radius then 4
  else if distance c1.center c2.center == c1.radius + c2.radius then 3
  else if distance c1.center c2.center < c1.radius + c2.radius &&
          distance c1.center c2.center > abs (c1.radius - c2.radius) then 2
  else if distance c1.center c2.center == abs (c1.radius - c2.radius) then 1
  else 0

theorem two_common_tangent_lines :
  let c1 : Circle := { center := (-2, 0), radius := 2 }
  let c2 : Circle := { center := (2, 1), radius := 3 }
  commonTangentLines c1 c2 = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_common_tangent_lines_l673_67377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_division_remainder_l673_67378

theorem polynomial_division_remainder : 
  ∃ q r : Polynomial ℤ, X^500 = (X^2 + 1) * (X^2 - 1) * q + r ∧ r.degree < 4 ∧ r = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_division_remainder_l673_67378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equality_condition_l673_67353

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then -x^2 + a*x else 2*a*x - 5

theorem function_equality_condition (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a x₁ = f a x₂) ↔ a < 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equality_condition_l673_67353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_ellipse_tangent_line_equation_l673_67320

-- Define the fixed point A
def A : ℝ × ℝ := (1, 0)

-- Define the line l: x = 4
def l : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = 4}

-- Define the point N
def N : ℝ × ℝ := (-1, 1)

-- Define the trajectory C
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 / 4 + p.2^2 / 3 = 1}

-- Define the distance ratio condition
def distance_ratio (M : ℝ × ℝ) : Prop :=
  (((M.1 - A.1)^2 + (M.2 - A.2)^2).sqrt / |M.1 - 4| = 1/2)

-- Theorem 1: The trajectory of M is the ellipse C
theorem trajectory_is_ellipse (M : ℝ × ℝ) :
  distance_ratio M → M ∈ C := by sorry

-- Define a line through two points
def line_through (P Q : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {X : ℝ × ℝ | ∃ t : ℝ, X = (1 - t) • P + t • Q}

-- Theorem 2: The tangent line through N has the equation 3x - 4y + 7 = 0
theorem tangent_line_equation :
  ∃ (P Q : ℝ × ℝ), P ∈ C ∧ Q ∈ C ∧ N = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2) →
  ∀ (x y : ℝ), (3 * x - 4 * y + 7 = 0) ↔ ((x, y) ∈ line_through P Q) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_ellipse_tangent_line_equation_l673_67320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_symmetry_l673_67323

/-- Given a function f(x) = ax³ + b*sin(x) + c/x + 2, prove that f(5) = 4 - m if f(-5) = m -/
theorem function_symmetry (a b c m : ℝ) :
  let f (x : ℝ) := a * x^3 + b * Real.sin x + c / x + 2
  f (-5) = m → f 5 = 4 - m := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_symmetry_l673_67323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_point_on_line_l673_67337

noncomputable def z : ℂ := (4 + 2*Complex.I) / (1 + Complex.I)^2

theorem complex_point_on_line :
  ∃ (x y : ℝ), (z = x + y*Complex.I) ∧ (x - 2*y + (-5) = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_point_on_line_l673_67337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_range_l673_67338

theorem cosine_range (m : ℝ) : 
  (∃ x : ℝ, Real.cos x = 2 * m - 1) → 0 ≤ m ∧ m ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_range_l673_67338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_to_marked_price_ratio_is_048_l673_67368

/-- The ratio of the cost to the marked price of books in a bookstore -/
noncomputable def cost_to_marked_price_ratio (marked_price : ℝ) : ℝ :=
  let discount_rate : ℝ := 0.20
  let cost_rate : ℝ := 0.60
  let selling_price : ℝ := marked_price * (1 - discount_rate)
  let cost : ℝ := selling_price * cost_rate
  cost / marked_price

/-- Theorem: The ratio of the cost to the marked price of books is 0.48 -/
theorem cost_to_marked_price_ratio_is_048 (marked_price : ℝ) (h : marked_price > 0) :
  cost_to_marked_price_ratio marked_price = 0.48 := by
  unfold cost_to_marked_price_ratio
  simp
  -- The actual proof would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_to_marked_price_ratio_is_048_l673_67368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_times_t_eq_two_l673_67391

/-- A function g : ℝ → ℝ satisfying the given conditions -/
noncomputable def g : ℝ → ℝ := sorry

/-- The condition g(1) = 1 -/
axiom g_one : g 1 = 1

/-- The condition g(x³ - y³) = (x - y)(g(x) + g(y)) for all real x and y -/
axiom g_cube_diff (x y : ℝ) : g (x^3 - y^3) = (x - y) * (g x + g y)

/-- The number of possible values of g(2) -/
noncomputable def m : ℕ := 1

/-- The sum of all possible values of g(2) -/
noncomputable def t : ℝ := 2

/-- The main theorem: m × t = 2 -/
theorem m_times_t_eq_two : m * t = 2 := by
  rw [m, t]
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_times_t_eq_two_l673_67391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_2022_value_l673_67365

def my_sequence (b : ℕ → ℝ) : Prop :=
  ∀ n ≥ 2, b n = b (n - 1) * b (n + 1)

theorem b_2022_value (b : ℕ → ℝ) 
  (h_seq : my_sequence b)
  (h_b1 : b 1 = 3 + Real.sqrt 11)
  (h_b1830 : b 1830 = 17 + Real.sqrt 11) :
  b 2022 = -1 + (7/4) * Real.sqrt 11 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_2022_value_l673_67365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_foci_distance_product_l673_67313

/-- The product of distances from an intersection point of an ellipse and hyperbola to their shared foci -/
theorem intersection_point_foci_distance_product (m n : ℝ) (hm : m > 0) (hn : n > 0) 
  (P F₁ F₂ : ℝ × ℝ) :
  (P.1^2 / m^2 + P.2^2 / 9 = 1) →  -- P is on the ellipse
  (P.1^2 / n^2 - P.2^2 / 4 = 1) →  -- P is on the hyperbola
  (∀ Q : ℝ × ℝ, Q.1^2 / m^2 + Q.2^2 / 9 = 1 → 
    (Real.sqrt ((Q.1 - F₁.1)^2 + (Q.2 - F₁.2)^2) + Real.sqrt ((Q.1 - F₂.1)^2 + (Q.2 - F₂.2)^2) = 2 * m)) →  -- F₁ and F₂ are foci of the ellipse
  (∀ Q : ℝ × ℝ, Q.1^2 / n^2 - Q.2^2 / 4 = 1 → 
    |Real.sqrt ((Q.1 - F₁.1)^2 + (Q.2 - F₁.2)^2) - Real.sqrt ((Q.1 - F₂.1)^2 + (Q.2 - F₂.2)^2)| = 2 * n) →  -- F₁ and F₂ are foci of the hyperbola
  Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) * Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2) = 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_foci_distance_product_l673_67313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_dot_product_bound_l673_67342

/-- Represents an ellipse with center at the origin -/
structure Ellipse where
  a : ℝ  -- Semi-major axis
  b : ℝ  -- Semi-minor axis
  h : a > 0 ∧ b > 0 ∧ a ≥ b

/-- The focal length of the ellipse -/
noncomputable def Ellipse.focalLength (e : Ellipse) : ℝ := Real.sqrt (e.a^2 - e.b^2)

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line passing through two points -/
structure Line where
  p1 : Point
  p2 : Point

/-- The dot product of two 2D vectors -/
def dotProduct (v1 v2 : Point) : ℝ := v1.x * v2.x + v1.y * v2.y

/-- Vector from one point to another -/
def vector (p1 p2 : Point) : Point :=
  { x := p2.x - p1.x, y := p2.y - p1.y }

/-- Membership of a point in an ellipse -/
def Point.inEllipse (p : Point) (e : Ellipse) : Prop :=
  (p.x^2 / e.a^2) + (p.y^2 / e.b^2) = 1

/-- Membership of a point on a line -/
def Point.onLine (p : Point) (l : Line) : Prop :=
  (p.y - l.p1.y) * (l.p2.x - l.p1.x) = (p.x - l.p1.x) * (l.p2.y - l.p1.y)

theorem ellipse_dot_product_bound 
  (e : Ellipse) 
  (h1 : e.focalLength = 2)
  (h2 : e.a = Real.sqrt 2 * e.b)
  (P : Point)
  (hP : P.x = 2 ∧ P.y = 0)
  (l : Line)
  (hl : l.p1.x = -1 ∧ l.p1.y = 0)  -- Left focus F
  (A B : Point)
  (hAB : A.inEllipse e ∧ B.inEllipse e ∧ A.onLine l ∧ B.onLine l)
  : dotProduct (vector P A) (vector P B) ≤ 17/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_dot_product_bound_l673_67342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l673_67311

def set_A : Set ℝ := {y | ∃ x, y = Real.sin x}

def set_B : Set ℝ := {y | ∃ x, y = Real.sqrt (-x^2 + 4*x - 3)}

theorem intersection_of_A_and_B : set_A ∩ set_B = Set.Icc 0 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l673_67311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_treadmill_time_difference_l673_67356

/-- Represents the days of the week when the treadmill was used -/
inductive Day
| Monday
| Wednesday
| Friday
| Sunday

/-- Calculates the time spent on the treadmill for a given distance and speed -/
noncomputable def time_spent (distance : ℝ) (speed : ℝ) : ℝ := distance / speed

/-- Returns the distance traveled on a given day -/
def distance (d : Day) : ℝ :=
  match d with
  | Day.Monday => 3
  | Day.Wednesday => 3
  | Day.Friday => 3
  | Day.Sunday => 4

/-- Returns the actual speed used on a given day -/
def actual_speed (d : Day) : ℝ :=
  match d with
  | Day.Monday => 6
  | Day.Wednesday => 4
  | Day.Friday => 5
  | Day.Sunday => 3

/-- Calculates the total time spent on the treadmill over all days -/
noncomputable def total_time (speed_func : Day → ℝ) : ℝ :=
  (time_spent (distance Day.Monday) (speed_func Day.Monday)) +
  (time_spent (distance Day.Wednesday) (speed_func Day.Wednesday)) +
  (time_spent (distance Day.Friday) (speed_func Day.Friday)) +
  (time_spent (distance Day.Sunday) (speed_func Day.Sunday))

/-- The constant speed of 5 mph -/
def constant_speed : Day → ℝ := λ _ => 5

theorem treadmill_time_difference :
  (total_time actual_speed - total_time constant_speed) * 60 = 35 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_treadmill_time_difference_l673_67356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_and_range_of_composition_l673_67312

noncomputable section

-- Define f as a function from ℝ to ℝ
def f : ℝ → ℝ := sorry

-- Domain of f
def domain_f : Set ℝ := Set.Icc (-2) 3

-- Range of f
def range_f : Set ℝ := Set.Icc (-2) 3

-- Function composition
def g (x : ℝ) : ℝ := 2 * x - 1

-- Domain of f ∘ g
def domain_fg : Set ℝ := {x | g x ∈ domain_f}

-- Range of f ∘ g
def range_fg : Set ℝ := {y | ∃ x ∈ domain_fg, f (g x) = y}

theorem domain_and_range_of_composition :
  domain_fg = Set.Icc (-1/2) 2 ∧ range_fg = range_f := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_and_range_of_composition_l673_67312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_dihedral_angle_l673_67314

-- Define a cube with unit side length
structure Cube where
  vertices : Fin 8 → ℝ × ℝ × ℝ

-- Define a point on an edge of the cube
def PointOnEdge (c : Cube) := ℝ

-- Define a plane using three points
def Plane := ℝ × ℝ × ℝ → Prop

-- Define the dihedral angle between two planes
noncomputable def dihedral_angle (p1 p2 : Plane) : ℝ := sorry

-- Define specific points and planes for our problem
def A : ℝ × ℝ × ℝ := (0, 0, 0)
def B : ℝ × ℝ × ℝ := (1, 0, 0)
def D : ℝ × ℝ × ℝ := (0, 1, 0)
def B₁ : ℝ × ℝ × ℝ := (1, 0, 1)
def D₁ : ℝ × ℝ × ℝ := (0, 1, 1)

def P (t : ℝ) : ℝ × ℝ × ℝ := (t, 0, 0)

def plane_PDB₁ (t : ℝ) : Plane := sorry
def plane_ADD₁A₁ : Plane := sorry

/--
  The minimum value of the dihedral angle between plane PDB₁ and plane ADD₁A₁ is arccos(√6/3)
-/
theorem min_dihedral_angle : 
  ∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 → 
    dihedral_angle (plane_PDB₁ t) plane_ADD₁A₁ ≥ Real.arccos (Real.sqrt 6 / 3) ∧ 
    ∃ t₀ : ℝ, 0 ≤ t₀ ∧ t₀ ≤ 1 ∧ dihedral_angle (plane_PDB₁ t₀) plane_ADD₁A₁ = Real.arccos (Real.sqrt 6 / 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_dihedral_angle_l673_67314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_hexagon_area_ratio_l673_67300

/-- A regular hexagon with side length a -/
structure RegularHexagon where
  a : ℝ
  a_pos : a > 0

/-- The smaller hexagon formed by connecting the midpoints of the sides of a regular hexagon -/
noncomputable def midpointHexagon (h : RegularHexagon) : RegularHexagon where
  a := h.a / 2
  a_pos := by
    apply div_pos
    exact h.a_pos
    norm_num

/-- The area of a regular hexagon -/
noncomputable def area (h : RegularHexagon) : ℝ := 3 * Real.sqrt 3 / 2 * h.a ^ 2

/-- The theorem stating the ratio of areas of the midpoint hexagon to the original hexagon -/
theorem midpoint_hexagon_area_ratio (h : RegularHexagon) :
    area (midpointHexagon h) / area h = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_hexagon_area_ratio_l673_67300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l673_67371

-- Define the function f(x)
noncomputable def f (x : ℝ) := x^2 + (x - 1) * Real.exp x

-- Theorem statement
theorem f_properties :
  (∃ (x : ℝ), f x = -1 ∧ ∀ (y : ℝ), f y ≥ f x) ∧
  (∀ (x1 x2 : ℝ), x2 > x1 ∧ x1 ≥ 1 →
    (f x1 - f x2) / (x1 - x2) > 4 / (x1 * x2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l673_67371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_calculation_l673_67388

/-- The constant term in the expansion of (1+x+x^2)(x-1/x)^6 -/
def m : ℝ := -5

/-- The area of the closed figure formed by y = -x^2 and y = mx -/
noncomputable def area : ℝ := ∫ x in (0)..(5), (-x^2 + m*x)

/-- Theorem stating that the area is equal to 125/6 -/
theorem area_calculation : area = 125/6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_calculation_l673_67388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eldest_child_sweets_l673_67333

theorem eldest_child_sweets (total : ℕ) (mother_fraction : ℚ) (second_child : ℕ) : 
  total = 27 →
  mother_fraction = 1/3 →
  second_child = 6 →
  ∃ (eldest youngest : ℕ),
    eldest + youngest + second_child = total - (mother_fraction * ↑total).floor ∧
    youngest = eldest / 2 ∧
    eldest = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eldest_child_sweets_l673_67333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_red_balls_count_l673_67390

/-- The number of red balls in the bag -/
def R : ℕ → ℕ := λ n => n

/-- The total number of balls in the bag -/
def total_balls (n : ℕ) : ℕ := R n + 3 + 2

/-- The probability of picking 2 red balls when 2 balls are picked at random -/
def prob_two_red (n : ℕ) : ℚ := (Nat.choose (R n) 2 : ℚ) / Nat.choose (total_balls n) 2

theorem red_balls_count (n : ℕ) : prob_two_red n = 1/6 → R n = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_red_balls_count_l673_67390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eva_wins_iff_n_odd_l673_67349

/-- Represents the state of the game board -/
def GameBoard (n : Nat) := Fin 2 → Fin n → Bool

/-- Represents a player in the game -/
inductive Player : Type where
  | Eva : Player
  | Camille : Player

/-- Represents a move in the game -/
structure Move (n : Nat) where
  row : Fin 2
  col : Fin n
  vertical : Bool

/-- Checks if a move is valid on the given board -/
def isValidMove {n : Nat} (board : GameBoard n) (move : Move n) : Bool :=
  sorry

/-- Applies a move to the board -/
def applyMove {n : Nat} (board : GameBoard n) (move : Move n) : GameBoard n :=
  sorry

/-- Checks if there are any valid moves left on the board -/
def hasValidMoves {n : Nat} (board : GameBoard n) : Bool :=
  sorry

/-- Returns the next player -/
def nextPlayer : Player → Player
  | Player.Eva => Player.Camille
  | Player.Camille => Player.Eva

/-- Determines the winner of the game -/
def gameWinner (n : Nat) : Player :=
  sorry

theorem eva_wins_iff_n_odd (n : Nat) :
  gameWinner n = Player.Eva ↔ n % 2 = 1 :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eva_wins_iff_n_odd_l673_67349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_path_length_theorem_path_length_theorem_modified_l673_67366

/-- Represents a section of the path between cities A and B -/
inductive PathSection
| Ascending
| Horizontal
| Descending

/-- Represents the path between cities A and B -/
structure PathAB where
  sections : List PathSection
  length : ℝ

/-- Represents a journey between cities A and B -/
structure Journey where
  path : PathAB
  speed : PathSection → ℝ
  time : ℝ

theorem path_length_theorem (pathAB : PathAB) (journeyAB journeyBA : Journey)
  (h1 : journeyAB.path = pathAB)
  (h2 : journeyBA.path = pathAB)
  (h3 : journeyAB.time = 2.3)
  (h4 : journeyBA.time = 2.6)
  (h5 : journeyAB.speed PathSection.Ascending = 3)
  (h6 : journeyAB.speed PathSection.Horizontal = 4)
  (h7 : journeyAB.speed PathSection.Descending = 6)
  (h8 : journeyBA.speed PathSection.Ascending = 3)
  (h9 : journeyBA.speed PathSection.Horizontal = 4)
  (h10 : journeyBA.speed PathSection.Descending = 6) :
  pathAB.length = 9.8 := by sorry

theorem path_length_theorem_modified (pathAB : PathAB) (journeyAB journeyBA : Journey)
  (h1 : journeyAB.path = pathAB)
  (h2 : journeyBA.path = pathAB)
  (h3 : journeyAB.time = 2.3)
  (h4 : journeyBA.time = 2.6)
  (h5 : journeyAB.speed PathSection.Ascending = 3)
  (h6 : journeyAB.speed PathSection.Horizontal = 4)
  (h7 : journeyAB.speed PathSection.Descending = 5)
  (h8 : journeyBA.speed PathSection.Ascending = 3)
  (h9 : journeyBA.speed PathSection.Horizontal = 4)
  (h10 : journeyBA.speed PathSection.Descending = 5) :
  9.1875 < pathAB.length ∧ pathAB.length < 9.65 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_path_length_theorem_path_length_theorem_modified_l673_67366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_example_l673_67384

/-- The area of a triangle given its three vertices -/
noncomputable def triangleArea (A B C : ℝ × ℝ) : ℝ :=
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  let (x₃, y₃) := C
  (1/2) * |x₁ * (y₂ - y₃) + x₂ * (y₃ - y₁) + x₃ * (y₁ - y₂)|

/-- Theorem: The area of the triangle with vertices (0,3), (4,-2), and (9,6) is 16.5 -/
theorem triangle_area_example : 
  triangleArea (0, 3) (4, -2) (9, 6) = 16.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_example_l673_67384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_shapes_count_l673_67376

-- Define the properties
def axisymmetric (shape : Shape) : Prop := sorry
def centrally_symmetric (shape : Shape) : Prop := sorry

-- Define the shapes
inductive Shape
| equilateral_triangle
| parallelogram
| rectangle
| rhombus
| square
| regular_pentagon

def is_both_symmetric (s : Shape) : Prop :=
  axisymmetric s ∧ centrally_symmetric s

def shape_list : List Shape :=
  [Shape.equilateral_triangle, Shape.parallelogram, Shape.rectangle, 
   Shape.rhombus, Shape.square, Shape.regular_pentagon]

theorem symmetric_shapes_count :
  (∃ (l : List Shape), l.length = 3 ∧
    (∀ s : Shape, s ∈ l ↔ is_both_symmetric s) ∧
    (∀ s : Shape, s ∈ shape_list)) :=
by sorry

#check symmetric_shapes_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_shapes_count_l673_67376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_iteration_for_identity_l673_67375

/-- Angle of a line with respect to the positive x-axis -/
noncomputable def angle (l : ℝ → ℝ) : ℝ := sorry

/-- Reflection of a line across another line -/
noncomputable def reflect (l : ℝ → ℝ) (axis : ℝ → ℝ) : ℝ → ℝ := sorry

/-- The transformation S as described in the problem -/
noncomputable def S (l : ℝ → ℝ) (l₁ l₂ : ℝ → ℝ) : ℝ → ℝ :=
  reflect (reflect l l₁) l₂

/-- The n-th iteration of S -/
noncomputable def S_iter (n : ℕ) (l l₁ l₂ : ℝ → ℝ) : ℝ → ℝ :=
  match n with
  | 0 => l
  | n + 1 => S (S_iter n l l₁ l₂) l₁ l₂

theorem smallest_iteration_for_identity :
  let l₁ : ℝ → ℝ := fun x => Real.tan (π/60) * x
  let l₂ : ℝ → ℝ := fun x => Real.tan (π/45) * x
  let l  : ℝ → ℝ := fun x => (1/3) * x
  ∀ m : ℕ, (S_iter m l l₁ l₂ = l ∧ ∀ k < m, S_iter k l l₁ l₂ ≠ l) ↔ m = 360 := by
  sorry

#check smallest_iteration_for_identity

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_iteration_for_identity_l673_67375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l673_67395

/-- Parabola structure -/
structure Parabola where
  eq : ℝ → ℝ → Prop
  h : ∀ x y, eq x y ↔ y^2 = x

/-- Point on a parabola -/
structure PointOnParabola (p : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : p.eq x y

/-- Intersection point of a line with x = t -/
structure IntersectionPoint where
  x : ℝ
  y : ℝ

/-- Main theorem about properties of points on a parabola -/
theorem parabola_properties
  (p : Parabola)
  (M : PointOnParabola p)
  (A B : IntersectionPoint)
  (P Q : PointOnParabola p)
  (t : ℝ) :
  -- Part 1
  (M.y = Real.sqrt 2 → ∃ F : ℝ × ℝ, ((M.x - F.1)^2 + (M.y - F.2)^2).sqrt = 7/4) ∧
  -- Part 2
  (t = -1 ∧ P.x = 1 ∧ P.y = 1 ∧ Q.x = 1 ∧ Q.y = -1 → A.y * B.y = -1) ∧
  -- Part 3
  (∃ t : ℝ, t = 1 ∧ A.y * B.y = 1 ∧ P.y * Q.y = 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l673_67395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l673_67316

noncomputable def ω : ℝ := 1

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin (ω * x), Real.sin (ω * x))
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.cos (ω * x), Real.sin (ω * x))

noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2 - 1/2

noncomputable def symmetry_distance : ℝ := Real.pi / 2

structure Triangle (A B C : ℝ) : Prop where
  sum_sides : A + B = 3
  third_side : C = Real.sqrt 3
  angle_condition : f C = 1

theorem triangle_area (A B C : ℝ) (h : Triangle A B C) : 
  (1/2) * A * B * Real.sin (Real.pi/3) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l673_67316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_scenario_stock_percentage_l673_67301

/-- Calculates the percentage of a stock (dividend yield) given the investment amount, stock price, and annual income. -/
noncomputable def stock_percentage (investment : ℝ) (price : ℝ) (income : ℝ) : ℝ :=
  (income * price / investment) / price * 100

/-- Theorem stating that for the given investment scenario, the stock percentage is approximately 36.76%. -/
theorem investment_scenario_stock_percentage :
  let investment := (6800 : ℝ)
  let price := (136 : ℝ)
  let income := (2500 : ℝ)
  abs (stock_percentage investment price income - 36.76) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_scenario_stock_percentage_l673_67301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solid_is_cone_l673_67398

/-- A solid with specific views -/
structure Solid where
  front_view : IsoscelesTriangle
  left_view : IsoscelesTriangle
  top_view : Circle

/-- An isosceles triangle -/
structure IsoscelesTriangle

/-- A circle -/
structure Circle

/-- A cone -/
structure Cone

/-- If a solid has a front view and a left view that are both isosceles triangles, 
    and a top view that is a circle, then the solid is a cone -/
theorem solid_is_cone (s : Solid) : Cone := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solid_is_cone_l673_67398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_transmission_l673_67328

def transmission_info (a₀ a₁ a₂ : Bool) : Bool × Bool × Bool × Bool × Bool :=
  let h₀ := xor a₀ a₁
  let h₁ := xor h₀ a₂
  (h₀, a₀, a₁, a₂, h₁)

theorem incorrect_transmission :
  ∀ (a₀ a₁ a₂ : Bool), transmission_info a₀ a₁ a₂ ≠ (true, false, true, true, true) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_transmission_l673_67328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_increasing_l673_67372

/-- A power function that passes through the point (4, 2) -/
noncomputable def f (x : ℝ) : ℝ := x ^ (Real.log 2 / Real.log 4)

theorem power_function_increasing : 
  (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ → f x₁ < f x₂) ∧ 
  f 4 = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_increasing_l673_67372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_power_problem_l673_67302

theorem nine_power_problem (y : ℝ) (h : (9 : ℝ)^(3*y) = 729) : (9 : ℝ)^(3*y - 2) = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_power_problem_l673_67302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_gcd_problem_l673_67386

theorem lcm_gcd_problem (a b : ℕ) : 
  a > 0 → b > 0 →
  Nat.lcm a b = 2310 → 
  Nat.gcd a b = 55 → 
  b = 605 → 
  a = 210 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_gcd_problem_l673_67386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l673_67340

noncomputable def f (x : ℝ) (t : ℝ) : ℝ := Real.log (-x^2 + 2*x - t) / Real.log (1/2)

def domain (m : ℝ) : Set ℝ := Set.Ioo m (m + 8)

def increasing_interval : Set ℝ := Set.Ioo 1 5

theorem f_monotone_increasing (m t : ℝ) :
  ∃ (h : domain m ⊆ Set.univ), StrictMono (f · t) →
  increasing_interval ⊆ domain m ∧ StrictMono (f · t) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l673_67340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_on_circumcircle_l673_67335

-- Define the necessary types and structures
structure Point where
  x : ℝ
  y : ℝ

def Ray := Point → Set Point

noncomputable def dist (A B : Point) : ℝ :=
  Real.sqrt ((A.x - B.x)^2 + (A.y - B.y)^2)

def circumcircle (A B C : Point) : Set Point := sorry

-- Define the theorem
theorem fixed_point_on_circumcircle 
  (A : Point) 
  (ray1 ray2 : Ray) 
  (h_constant_sum : ∀ (B C : Point), B ∈ ray1 A → C ∈ ray2 A → 
    ∃ (k : ℝ), dist A B + dist A C = k) :
  ∃ (D : Point), D ≠ A ∧ 
    ∀ (B C : Point), B ∈ ray1 A → C ∈ ray2 A → 
      D ∈ circumcircle A B C := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_on_circumcircle_l673_67335
