import Mathlib

namespace NUMINAMATH_CALUDE_jellybean_problem_l187_18775

theorem jellybean_problem :
  ∃ (n : ℕ), 
    n ≥ 150 ∧ 
    n % 17 = 9 ∧ 
    (∀ m : ℕ, m ≥ 150 ∧ m % 17 = 9 → m ≥ n) ∧
    n = 162 := by
  sorry

end NUMINAMATH_CALUDE_jellybean_problem_l187_18775


namespace NUMINAMATH_CALUDE_at_least_13_blondes_identifiable_l187_18700

/-- Represents a woman in the factory -/
inductive Woman
| Blonde
| Brunette

/-- The total number of women in the factory -/
def total_women : ℕ := 217

/-- The number of brunettes in the factory -/
def num_brunettes : ℕ := 17

/-- The number of blondes in the factory -/
def num_blondes : ℕ := 200

/-- The number of women each woman lists as blonde -/
def list_size : ℕ := 200

/-- A function representing a woman's list of supposed blondes -/
def list_blondes (w : Woman) : Finset Woman := sorry

theorem at_least_13_blondes_identifiable :
  ∃ (identified_blondes : Finset Woman),
    (∀ w ∈ identified_blondes, w = Woman.Blonde) ∧
    identified_blondes.card ≥ 13 := by sorry

end NUMINAMATH_CALUDE_at_least_13_blondes_identifiable_l187_18700


namespace NUMINAMATH_CALUDE_exponent_monotonicity_l187_18798

theorem exponent_monotonicity (a x₁ x₂ : ℝ) :
  (a > 1 ∧ x₁ > x₂ → a^x₁ > a^x₂) ∧
  (0 < a ∧ a < 1 ∧ x₁ > x₂ → a^x₁ < a^x₂) := by
  sorry

end NUMINAMATH_CALUDE_exponent_monotonicity_l187_18798


namespace NUMINAMATH_CALUDE_total_amount_is_105_l187_18787

/-- Represents the share distribution among x, y, and z -/
structure ShareDistribution where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The conditions of the problem -/
def problem_conditions (s : ShareDistribution) : Prop :=
  s.y = 0.45 * s.x ∧ s.z = 0.30 * s.x ∧ s.y = 27

/-- The theorem to prove -/
theorem total_amount_is_105 (s : ShareDistribution) :
  problem_conditions s → s.x + s.y + s.z = 105 := by sorry

end NUMINAMATH_CALUDE_total_amount_is_105_l187_18787


namespace NUMINAMATH_CALUDE_max_value_theorem_l187_18786

theorem max_value_theorem (a b c : ℝ) (ha : 0 ≤ a ∧ a ≤ 2) (hb : 0 ≤ b ∧ b ≤ 2) (hc : 0 ≤ c ∧ c ≤ 2) :
  2 * Real.sqrt (a * b * c / 8) + Real.sqrt ((2 - a) * (2 - b) * (2 - c)) ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l187_18786


namespace NUMINAMATH_CALUDE_mary_lamb_count_l187_18734

/-- The number of lambs Mary has after a series of events --/
def final_lamb_count (initial_lambs : ℕ) (lambs_with_babies : ℕ) (babies_per_lamb : ℕ) 
                     (lambs_traded : ℕ) (extra_lambs_found : ℕ) : ℕ :=
  initial_lambs + lambs_with_babies * babies_per_lamb - lambs_traded + extra_lambs_found

/-- Theorem stating that Mary ends up with 14 lambs --/
theorem mary_lamb_count : 
  final_lamb_count 6 2 2 3 7 = 14 := by
  sorry

end NUMINAMATH_CALUDE_mary_lamb_count_l187_18734


namespace NUMINAMATH_CALUDE_tangent_equations_not_equivalent_l187_18712

open Real

theorem tangent_equations_not_equivalent :
  ¬(∀ x : ℝ, (tan (2 * x) - (1 / tan x) = 0) ↔ ((2 * tan x) / (1 - tan x ^ 2) - 1 / tan x = 0)) :=
by sorry

end NUMINAMATH_CALUDE_tangent_equations_not_equivalent_l187_18712


namespace NUMINAMATH_CALUDE_min_diff_composite_sum_96_l187_18776

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

theorem min_diff_composite_sum_96 :
  ∃ (a b : ℕ), is_composite a ∧ is_composite b ∧ a + b = 96 ∧
  ∀ (c d : ℕ), is_composite c → is_composite d → c + d = 96 → c ≠ d →
  (max c d - min c d) ≥ (max a b - min a b) ∧ (max a b - min a b) = 4 :=
sorry

end NUMINAMATH_CALUDE_min_diff_composite_sum_96_l187_18776


namespace NUMINAMATH_CALUDE_flower_difference_l187_18740

def white_flowers : ℕ := 555
def red_flowers : ℕ := 347
def blue_flowers : ℕ := 498
def yellow_flowers : ℕ := 425

theorem flower_difference : 
  (red_flowers + blue_flowers + yellow_flowers) - white_flowers = 715 := by
  sorry

end NUMINAMATH_CALUDE_flower_difference_l187_18740


namespace NUMINAMATH_CALUDE_right_triangle_consecutive_sides_l187_18728

theorem right_triangle_consecutive_sides : 
  ∀ (a b c : ℕ), 
  (a * a + b * b = c * c) →  -- Pythagorean theorem for right-angled triangle
  (b = a + 1 ∧ c = b + 1) →  -- Sides are consecutive natural numbers
  (a = 3 ∧ b = 4) →          -- Two sides are 3 and 4
  c = 5 := by               -- The third side is 5
sorry

end NUMINAMATH_CALUDE_right_triangle_consecutive_sides_l187_18728


namespace NUMINAMATH_CALUDE_sampling_methods_are_appropriate_l187_18719

/-- Represents a sampling method -/
inductive SamplingMethod
  | StratifiedSampling
  | SimpleRandomSampling
  | SystematicSampling

/-- Represents a region with sales points -/
structure Region where
  name : String
  salesPoints : Nat

/-- Represents a company with multiple regions -/
structure Company where
  regions : List Region
  totalSalesPoints : Nat

/-- Represents an investigation -/
structure Investigation where
  sampleSize : Nat
  samplingMethod : SamplingMethod

/-- The company in the problem -/
def problemCompany : Company :=
  { regions := [
      { name := "A", salesPoints := 150 },
      { name := "B", salesPoints := 120 },
      { name := "C", salesPoints := 180 },
      { name := "D", salesPoints := 150 }
    ],
    totalSalesPoints := 600
  }

/-- The first investigation in the problem -/
def investigation1 : Investigation :=
  { sampleSize := 100,
    samplingMethod := SamplingMethod.StratifiedSampling
  }

/-- The second investigation in the problem -/
def investigation2 : Investigation :=
  { sampleSize := 7,
    samplingMethod := SamplingMethod.SimpleRandomSampling
  }

/-- Checks if stratified sampling is appropriate for the given company and investigation -/
def isStratifiedSamplingAppropriate (company : Company) (investigation : Investigation) : Prop :=
  investigation.samplingMethod = SamplingMethod.StratifiedSampling ∧
  company.regions.length > 1 ∧
  investigation.sampleSize < company.totalSalesPoints

/-- Checks if simple random sampling is appropriate for the given sample size and population -/
def isSimpleRandomSamplingAppropriate (sampleSize : Nat) (populationSize : Nat) : Prop :=
  sampleSize < populationSize

/-- Theorem stating that the sampling methods are appropriate for the given investigations -/
theorem sampling_methods_are_appropriate :
  isStratifiedSamplingAppropriate problemCompany investigation1 ∧
  isSimpleRandomSamplingAppropriate investigation2.sampleSize 20 :=
  sorry

end NUMINAMATH_CALUDE_sampling_methods_are_appropriate_l187_18719


namespace NUMINAMATH_CALUDE_f_properties_l187_18777

noncomputable def f (x : ℝ) := Real.exp x * Real.cos x - x

theorem f_properties :
  let a := 0
  let b := Real.pi / 2
  ∃ (tangent_line : ℝ → ℝ),
    (∀ x, tangent_line x = 1) ∧
    (∀ x ∈ Set.Icc a b, f x ≤ f a) ∧
    (∀ x ∈ Set.Icc a b, f b ≤ f x) ∧
    f a = 1 ∧
    f b = -Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l187_18777


namespace NUMINAMATH_CALUDE_quadratic_inequality_empty_solution_l187_18708

theorem quadratic_inequality_empty_solution : 
  {x : ℝ | -x^2 + 2*x - 3 > 0} = ∅ := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_empty_solution_l187_18708


namespace NUMINAMATH_CALUDE_complex_magnitude_l187_18747

theorem complex_magnitude (z : ℂ) (h : z * (1 + Complex.I) = 3 + Complex.I) :
  Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l187_18747


namespace NUMINAMATH_CALUDE_logarithm_sum_equation_l187_18774

theorem logarithm_sum_equation (x : ℝ) (h : x > 0) :
  (1 / Real.log x / Real.log 3) + (1 / Real.log x / Real.log 4) + (1 / Real.log x / Real.log 5) = 1 →
  x = 60 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_sum_equation_l187_18774


namespace NUMINAMATH_CALUDE_subtraction_multiplication_equality_l187_18759

theorem subtraction_multiplication_equality : ((3.54 - 1.32) * 2) = 4.44 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_multiplication_equality_l187_18759


namespace NUMINAMATH_CALUDE_abs_diff_ge_one_l187_18716

theorem abs_diff_ge_one (a b c : ℝ) 
  (sum_eq : a + b + c = 2) 
  (sum_sq_eq : a^2 + b^2 + c^2 = 2) : 
  max (|a - b|) (max (|b - c|) (|c - a|)) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_abs_diff_ge_one_l187_18716


namespace NUMINAMATH_CALUDE_inequality_proof_l187_18799

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 + 2) * (b^2 + 2) * (c^2 + 2) ≥ 9 * (a*b + b*c + c*a) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l187_18799


namespace NUMINAMATH_CALUDE_different_counting_units_for_equal_decimals_l187_18764

-- Define the concept of a decimal number
structure Decimal where
  value : ℚ
  decimalPlaces : ℕ

-- Define the concept of a counting unit
def countingUnit (d : Decimal) : ℚ := 1 / (10 ^ d.decimalPlaces)

-- Define equality for decimals based on their value
def decimalEqual (d1 d2 : Decimal) : Prop := d1.value = d2.value

-- Theorem statement
theorem different_counting_units_for_equal_decimals :
  ∃ (d1 d2 : Decimal), decimalEqual d1 d2 ∧ countingUnit d1 ≠ countingUnit d2 := by
  sorry

end NUMINAMATH_CALUDE_different_counting_units_for_equal_decimals_l187_18764


namespace NUMINAMATH_CALUDE_second_polygon_sides_l187_18750

/-- Given two regular polygons with the same perimeter, where one has 50 sides
    and a side length three times as long as the other, prove that the number
    of sides of the second polygon is 150. -/
theorem second_polygon_sides (s : ℝ) (n : ℕ) :
  s > 0 →
  50 * (3 * s) = n * s →
  n = 150 := by
  sorry

end NUMINAMATH_CALUDE_second_polygon_sides_l187_18750


namespace NUMINAMATH_CALUDE_angle_inequality_equivalence_l187_18749

theorem angle_inequality_equivalence (θ : Real) (h : 0 ≤ θ ∧ θ ≤ 2 * Real.pi) :
  (∀ x : Real, 0 ≤ x ∧ x ≤ 1 →
    x^2 * Real.sin θ - x * (1 - 2*x) + (1 - 3*x)^2 * Real.cos θ > 0) ↔
  (0 < θ ∧ θ < Real.pi / 2) :=
by sorry

end NUMINAMATH_CALUDE_angle_inequality_equivalence_l187_18749


namespace NUMINAMATH_CALUDE_west_distance_calculation_l187_18761

-- Define the given constants
def total_distance : ℝ := 150
def north_distance : ℝ := 55

-- Theorem statement
theorem west_distance_calculation :
  total_distance - north_distance = 95 := by sorry

end NUMINAMATH_CALUDE_west_distance_calculation_l187_18761


namespace NUMINAMATH_CALUDE_rotated_point_x_coordinate_l187_18720

/-- Given a point P(1,2) in the Cartesian plane, prove that when the vector OP
    is rotated counterclockwise by 5π/6 around the origin O to obtain vector OQ,
    the x-coordinate of Q is -√3/2 - 2√5. -/
theorem rotated_point_x_coordinate (P Q : ℝ × ℝ) (h1 : P = (1, 2)) :
  (∃ θ : ℝ, θ = 5 * π / 6 ∧
   Q.1 = P.1 * Real.cos θ - P.2 * Real.sin θ ∧
   Q.2 = P.1 * Real.sin θ + P.2 * Real.cos θ) →
  Q.1 = -Real.sqrt 3 / 2 - 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_rotated_point_x_coordinate_l187_18720


namespace NUMINAMATH_CALUDE_rectangular_field_area_l187_18743

theorem rectangular_field_area (a b c : ℝ) (h1 : a = 15) (h2 : c = 17) (h3 : a^2 + b^2 = c^2) : a * b = 120 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l187_18743


namespace NUMINAMATH_CALUDE_probability_of_observing_change_l187_18779

/-- Represents the duration of the traffic light cycle in seconds -/
def cycle_duration : ℕ := 63

/-- Represents the points in the cycle where color changes occur -/
def change_points : List ℕ := [30, 33, 63]

/-- Represents the duration of the observation interval in seconds -/
def observation_duration : ℕ := 4

/-- Calculates the total duration of intervals where a change can be observed -/
def total_change_duration (cycle : ℕ) (changes : List ℕ) (obs : ℕ) : ℕ :=
  sorry

/-- The main theorem stating the probability of observing a color change -/
theorem probability_of_observing_change :
  (total_change_duration cycle_duration change_points observation_duration : ℚ) / cycle_duration = 5 / 21 :=
sorry

end NUMINAMATH_CALUDE_probability_of_observing_change_l187_18779


namespace NUMINAMATH_CALUDE_shorter_diagonal_length_l187_18792

/-- Given two vectors in a 2D plane satisfying specific conditions, 
    prove that the length of the shorter diagonal of the parallelogram 
    formed by these vectors is √3. -/
theorem shorter_diagonal_length (a b : ℝ × ℝ) : 
  ‖a‖ = 1 →
  ‖b‖ = 2 →
  a • b = 1 →  -- This represents cos(π/3) = 1/2, as |a||b|cos(π/3) = 1
  Real.sqrt 3 = min (‖a + b‖) (‖a - b‖) := by
  sorry


end NUMINAMATH_CALUDE_shorter_diagonal_length_l187_18792


namespace NUMINAMATH_CALUDE_problem_solution_l187_18738

open Real

theorem problem_solution (α β : ℝ) (h1 : tan α = -1/3) (h2 : cos β = sqrt 5 / 5)
  (h3 : 0 < α) (h4 : α < π) (h5 : 0 < β) (h6 : β < π) :
  (tan (α + β) = 1) ∧
  (∃ (x : ℝ), sqrt 2 * sin (x - α) + cos (x + β) = sqrt 5) ∧
  (∃ (x : ℝ), sqrt 2 * sin (x - α) + cos (x + β) = -sqrt 5) ∧
  (∀ (x : ℝ), sqrt 2 * sin (x - α) + cos (x + β) ≤ sqrt 5) ∧
  (∀ (x : ℝ), -sqrt 5 ≤ sqrt 2 * sin (x - α) + cos (x + β)) := by
  sorry


end NUMINAMATH_CALUDE_problem_solution_l187_18738


namespace NUMINAMATH_CALUDE_alcohol_solution_proof_l187_18781

/-- Proves that adding 2.4 liters of pure alcohol to a 6-liter solution that is 30% alcohol 
    results in a 50% alcohol solution -/
theorem alcohol_solution_proof :
  let initial_volume : ℝ := 6
  let initial_percentage : ℝ := 0.30
  let target_percentage : ℝ := 0.50
  let added_alcohol : ℝ := 2.4
  let final_volume : ℝ := initial_volume + added_alcohol
  let final_alcohol_volume : ℝ := initial_volume * initial_percentage + added_alcohol
  final_alcohol_volume / final_volume = target_percentage :=
by sorry

end NUMINAMATH_CALUDE_alcohol_solution_proof_l187_18781


namespace NUMINAMATH_CALUDE_factorization_equality_l187_18729

theorem factorization_equality (m n : ℝ) : 
  m^2 - n^2 + 2*m - 2*n = (m - n)*(m + n + 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l187_18729


namespace NUMINAMATH_CALUDE_square_rectangle_area_problem_l187_18752

theorem square_rectangle_area_problem :
  ∃ (x₁ x₂ : ℝ),
    (∀ x : ℝ, (x - 3) * (x + 4) = 2 * (x - 2)^2 → x = x₁ ∨ x = x₂) ∧
    x₁ + x₂ = 9 := by
  sorry

end NUMINAMATH_CALUDE_square_rectangle_area_problem_l187_18752


namespace NUMINAMATH_CALUDE_polynomial_roots_in_arithmetic_progression_l187_18727

theorem polynomial_roots_in_arithmetic_progression (j k : ℝ) : 
  (∃ (b d : ℝ), d ≠ 0 ∧ 
    (∀ x : ℝ, x^4 + j*x^2 + k*x + 400 = 0 ↔ 
      (x = b ∨ x = b + d ∨ x = b + 2*d ∨ x = b + 3*d)) ∧
    (b ≠ b + d) ∧ (b + d ≠ b + 2*d) ∧ (b + 2*d ≠ b + 3*d))
  → j = -200 := by
sorry

end NUMINAMATH_CALUDE_polynomial_roots_in_arithmetic_progression_l187_18727


namespace NUMINAMATH_CALUDE_oil_drilling_probability_l187_18732

/-- The probability of drilling into an oil layer in a sea area -/
theorem oil_drilling_probability (total_area oil_area : ℝ) (h1 : total_area = 10000) (h2 : oil_area = 40) :
  oil_area / total_area = 0.004 := by
sorry

end NUMINAMATH_CALUDE_oil_drilling_probability_l187_18732


namespace NUMINAMATH_CALUDE_delta_y_value_l187_18763

/-- The function f(x) = x² + 1 -/
def f (x : ℝ) : ℝ := x^2 + 1

/-- Theorem: For f(x) = x² + 1, when x = 2 and Δx = 0.1, Δy = 0.41 -/
theorem delta_y_value (x : ℝ) (Δx : ℝ) (h1 : x = 2) (h2 : Δx = 0.1) :
  f (x + Δx) - f x = 0.41 := by
  sorry


end NUMINAMATH_CALUDE_delta_y_value_l187_18763


namespace NUMINAMATH_CALUDE_parabola_properties_l187_18768

/-- A parabola with equation y = x² - 2mx + m² - 9 where m is a constant -/
def parabola (m : ℝ) (x : ℝ) : ℝ := x^2 - 2*m*x + m^2 - 9

/-- The x-coordinates of the intersection points of the parabola with the x-axis -/
def roots (m : ℝ) : Set ℝ := {x : ℝ | parabola m x = 0}

/-- The y-coordinate of a point on the parabola given its x-coordinate -/
def y_coord (m : ℝ) (x : ℝ) : ℝ := parabola m x

theorem parabola_properties (m : ℝ) :
  (∃ (A B : ℝ), A ∈ roots m ∧ B ∈ roots m ∧ A ≠ B) →
  (∀ x, y_coord m x ≥ -9) ∧
  (∃ (A B : ℝ), A ∈ roots m ∧ B ∈ roots m ∧ |A - B| = 6) ∧
  (∀ x₁ x₂, x₁ < x₂ ∧ x₂ < m - 1 → y_coord m x₁ > y_coord m x₂) ∧
  (y_coord m (m + 1) < y_coord m (m - 3)) :=
by sorry

end NUMINAMATH_CALUDE_parabola_properties_l187_18768


namespace NUMINAMATH_CALUDE_all_subtracting_not_purple_not_all_happy_are_purple_some_happy_cant_subtract_l187_18745

-- Define the universe of snakes
variable (Snake : Type)

-- Define properties of snakes
variable (purple happy can_add can_subtract : Snake → Prop)

-- Define the number of snakes
variable (total_snakes : ℕ)
variable (purple_snakes : ℕ)
variable (happy_snakes : ℕ)

-- State the given conditions
variable (h1 : total_snakes = 20)
variable (h2 : purple_snakes = 6)
variable (h3 : happy_snakes = 8)
variable (h4 : ∃ s, happy s ∧ can_add s)
variable (h5 : ∀ s, purple s → ¬can_subtract s)
variable (h6 : ∀ s, ¬can_subtract s → ¬can_add s)

-- State the theorems to be proved
theorem all_subtracting_not_purple : ∀ s, can_subtract s → ¬purple s := by sorry

theorem not_all_happy_are_purple : ¬(∀ s, happy s → purple s) := by sorry

theorem some_happy_cant_subtract : ∃ s, happy s ∧ ¬can_subtract s := by sorry

end NUMINAMATH_CALUDE_all_subtracting_not_purple_not_all_happy_are_purple_some_happy_cant_subtract_l187_18745


namespace NUMINAMATH_CALUDE_decimal_33_is_quaternary_201_l187_18718

-- Define a function to convert decimal to quaternary
def decimalToQuaternary (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec convert (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else convert (m / 4) ((m % 4) :: acc)
    convert n []

-- Theorem statement
theorem decimal_33_is_quaternary_201 :
  decimalToQuaternary 33 = [2, 0, 1] := by
  sorry


end NUMINAMATH_CALUDE_decimal_33_is_quaternary_201_l187_18718


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l187_18713

theorem simplify_fraction_product : (225 : ℚ) / 10125 * 45 = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l187_18713


namespace NUMINAMATH_CALUDE_slope_relation_l187_18780

theorem slope_relation (k : ℝ) : 
  (∃ α : ℝ, α = (2 : ℝ) ∧ (2 : ℝ) * α = k) → k = -(4 : ℝ) / 3 := by
  sorry

end NUMINAMATH_CALUDE_slope_relation_l187_18780


namespace NUMINAMATH_CALUDE_quadratic_sum_l187_18725

theorem quadratic_sum (x : ℝ) : ∃ (a b c : ℝ),
  (8 * x^2 + 64 * x + 512 = a * (x + b)^2 + c) ∧ (a + b + c = 396) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l187_18725


namespace NUMINAMATH_CALUDE_speed_of_boat_in_still_water_l187_18766

/-- Theorem: Speed of boat in still water
Given:
- The rate of the current is 15 km/hr
- The boat traveled downstream for 25 minutes
- The boat covered a distance of 33.33 km downstream

Prove that the speed of the boat in still water is approximately 64.992 km/hr
-/
theorem speed_of_boat_in_still_water
  (current_speed : ℝ)
  (travel_time : ℝ)
  (distance_covered : ℝ)
  (h1 : current_speed = 15)
  (h2 : travel_time = 25 / 60)
  (h3 : distance_covered = 33.33) :
  ∃ (boat_speed : ℝ), abs (boat_speed - 64.992) < 0.001 ∧
    distance_covered = (boat_speed + current_speed) * travel_time :=
by sorry

end NUMINAMATH_CALUDE_speed_of_boat_in_still_water_l187_18766


namespace NUMINAMATH_CALUDE_binary_1101001101_equals_base4_12021_l187_18710

/-- Converts a binary (base 2) number to its decimal representation -/
def binary_to_decimal (binary : List Bool) : ℕ :=
  binary.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

/-- Converts a decimal number to its base 4 representation -/
def decimal_to_base4 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
    aux n []

theorem binary_1101001101_equals_base4_12021 :
  let binary : List Bool := [true, true, false, true, false, false, true, true, false, true]
  let decimal := binary_to_decimal binary
  let base4 := decimal_to_base4 decimal
  base4 = [1, 2, 0, 2, 1] := by sorry

end NUMINAMATH_CALUDE_binary_1101001101_equals_base4_12021_l187_18710


namespace NUMINAMATH_CALUDE_only_frustum_has_two_parallel_surfaces_l187_18754

-- Define the geometric bodies
inductive GeometricBody
  | Pyramid
  | Prism
  | Frustum
  | Cuboid

-- Define a function to count parallel surfaces
def parallelSurfaceCount (body : GeometricBody) : Nat :=
  match body with
  | GeometricBody.Pyramid => 0
  | GeometricBody.Prism => 6
  | GeometricBody.Frustum => 2
  | GeometricBody.Cuboid => 6

-- Theorem statement
theorem only_frustum_has_two_parallel_surfaces :
  ∀ (body : GeometricBody),
    parallelSurfaceCount body = 2 ↔ body = GeometricBody.Frustum :=
by
  sorry


end NUMINAMATH_CALUDE_only_frustum_has_two_parallel_surfaces_l187_18754


namespace NUMINAMATH_CALUDE_bus_problem_l187_18793

/-- The number of people who got off at the second stop of a bus route -/
def people_off_second_stop (initial : ℕ) (first_off : ℕ) (second_on : ℕ) (third_off : ℕ) (third_on : ℕ) (final : ℕ) : ℕ :=
  initial - first_off - final + second_on - third_off + third_on

theorem bus_problem : people_off_second_stop 50 15 2 4 3 28 = 8 := by
  sorry

end NUMINAMATH_CALUDE_bus_problem_l187_18793


namespace NUMINAMATH_CALUDE_crackers_per_person_l187_18782

theorem crackers_per_person 
  (num_friends : ℕ)
  (cracker_ratio cake_ratio : ℕ)
  (initial_crackers initial_cakes : ℕ)
  (h1 : num_friends = 6)
  (h2 : cracker_ratio = 3)
  (h3 : cake_ratio = 5)
  (h4 : initial_crackers = 72)
  (h5 : initial_cakes = 180) :
  initial_crackers / (cracker_ratio * num_friends) = 12 :=
by sorry

end NUMINAMATH_CALUDE_crackers_per_person_l187_18782


namespace NUMINAMATH_CALUDE_new_person_weight_new_person_weight_is_87_l187_18783

/-- The weight of a new person joining a group, given specific conditions -/
theorem new_person_weight (initial_count : ℕ) (leaving_weight : ℝ) (avg_increase : ℝ) : ℝ :=
  let total_increase := initial_count * avg_increase
  leaving_weight + total_increase

/-- Proof that the new person's weight is 87 kg under given conditions -/
theorem new_person_weight_is_87 :
  new_person_weight 8 67 2.5 = 87 := by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_new_person_weight_is_87_l187_18783


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l187_18737

theorem polynomial_coefficient_sum :
  ∀ (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ),
  (∀ x : ℝ, x + x^10 = a + a₁*(x+1) + a₂*(x+1)^2 + a₃*(x+1)^3 + a₄*(x+1)^4 + 
                     a₅*(x+1)^5 + a₆*(x+1)^6 + a₇*(x+1)^7 + a₈*(x+1)^8 + 
                     a₉*(x+1)^9 + a₁₀*(x+1)^10) →
  a + a₂ + a₃ + a₄ + a₅ + a₆ + a₈ = 510 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l187_18737


namespace NUMINAMATH_CALUDE_swimmer_downstream_distance_l187_18762

/-- Proves that a swimmer travels 32 km downstream given specific conditions -/
theorem swimmer_downstream_distance 
  (upstream_distance : ℝ) 
  (time : ℝ) 
  (still_water_speed : ℝ) 
  (h1 : upstream_distance = 24) 
  (h2 : time = 4) 
  (h3 : still_water_speed = 7) : 
  ∃ (downstream_distance : ℝ), downstream_distance = 32 := by
  sorry

end NUMINAMATH_CALUDE_swimmer_downstream_distance_l187_18762


namespace NUMINAMATH_CALUDE_vet_clinic_dog_treatment_cost_l187_18760

theorem vet_clinic_dog_treatment_cost (cat_cost : ℕ) (num_dogs num_cats total_cost : ℕ) :
  cat_cost = 40 →
  num_dogs = 20 →
  num_cats = 60 →
  total_cost = 3600 →
  ∃ (dog_cost : ℕ), dog_cost * num_dogs + cat_cost * num_cats = total_cost ∧ dog_cost = 60 :=
by sorry

end NUMINAMATH_CALUDE_vet_clinic_dog_treatment_cost_l187_18760


namespace NUMINAMATH_CALUDE_team_formations_count_l187_18756

/-- The number of ways to select k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of ways to form a team of 3 teachers from 4 female and 5 male teachers,
    with the condition that the team must include both male and female teachers -/
def teamFormations : ℕ :=
  choose 5 1 * choose 4 2 + choose 5 2 * choose 4 1

theorem team_formations_count :
  teamFormations = 70 := by sorry

end NUMINAMATH_CALUDE_team_formations_count_l187_18756


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l187_18778

-- Define a geometric sequence
def isGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- Define the problem statement
theorem geometric_sequence_problem (a : ℕ → ℝ) :
  isGeometricSequence a →
  (a 4 + a 8 = -11) →
  (a 4 * a 8 = 9) →
  a 6 = -3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l187_18778


namespace NUMINAMATH_CALUDE_school_distance_l187_18757

/-- The distance between a student's house and school, given travel time conditions. -/
theorem school_distance (t : ℝ) : 
  (t + 1/3 = 24/9) → (t - 1/3 = 24/12) → 24 = 24 := by
  sorry

end NUMINAMATH_CALUDE_school_distance_l187_18757


namespace NUMINAMATH_CALUDE_remainder_proof_l187_18705

theorem remainder_proof : 123456789012 % 252 = 84 := by
  sorry

end NUMINAMATH_CALUDE_remainder_proof_l187_18705


namespace NUMINAMATH_CALUDE_coins_in_pockets_l187_18767

/-- The number of ways to place n identical objects into k distinct containers -/
def stars_and_bars (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The problem of placing 5 identical coins into 3 different pockets -/
theorem coins_in_pockets : stars_and_bars 5 3 = 21 := by sorry

end NUMINAMATH_CALUDE_coins_in_pockets_l187_18767


namespace NUMINAMATH_CALUDE_labor_cost_increase_l187_18721

/-- Represents the cost components of manufacturing a car --/
structure CarCost where
  raw_material : ℝ
  labor : ℝ
  overhead : ℝ

/-- The initial cost ratio --/
def initial_ratio : CarCost := ⟨4, 3, 2⟩

/-- The percentage changes in costs --/
structure CostChanges where
  raw_material : ℝ := 0.10  -- 10% increase
  labor : ℝ                 -- Unknown, to be calculated
  overhead : ℝ := -0.05     -- 5% decrease
  total : ℝ := 0.06         -- 6% increase

/-- Calculates the new cost based on the initial cost and percentage change --/
def new_cost (initial : ℝ) (change : ℝ) : ℝ :=
  initial * (1 + change)

/-- Theorem stating that the labor cost increased by 8% --/
theorem labor_cost_increase (c : CostChanges) :
  c.labor = 0.08 := by sorry

end NUMINAMATH_CALUDE_labor_cost_increase_l187_18721


namespace NUMINAMATH_CALUDE_factor_polynomial_l187_18704

theorem factor_polynomial (x : ℝ) : 45 * x^3 - 135 * x^7 = 45 * x^3 * (1 - 3 * x^4) := by
  sorry

end NUMINAMATH_CALUDE_factor_polynomial_l187_18704


namespace NUMINAMATH_CALUDE_no_large_squares_in_H_l187_18758

/-- The set of points (x,y) with integer coordinates satisfying 2 ≤ |x| ≤ 6 and 2 ≤ |y| ≤ 6 -/
def H : Set (ℤ × ℤ) :=
  {p | 2 ≤ |p.1| ∧ |p.1| ≤ 6 ∧ 2 ≤ |p.2| ∧ |p.2| ≤ 6}

/-- A square with side length at least 8 -/
def IsValidSquare (s : Set (ℤ × ℤ)) : Prop :=
  ∃ (a b c d : ℤ × ℤ), s = {a, b, c, d} ∧
  (a.1 - b.1)^2 + (a.2 - b.2)^2 ≥ 64 ∧
  (b.1 - c.1)^2 + (b.2 - c.2)^2 ≥ 64 ∧
  (c.1 - d.1)^2 + (c.2 - d.2)^2 ≥ 64 ∧
  (d.1 - a.1)^2 + (d.2 - a.2)^2 ≥ 64

theorem no_large_squares_in_H :
  ¬∃ s : Set (ℤ × ℤ), (∀ p ∈ s, p ∈ H) ∧ IsValidSquare s := by
  sorry

end NUMINAMATH_CALUDE_no_large_squares_in_H_l187_18758


namespace NUMINAMATH_CALUDE_car_not_sold_probability_l187_18790

/-- Given the odds of selling a car on a given day are 5:6, 
    the probability that the car is not sold on that day is 6/11 -/
theorem car_not_sold_probability (odds_success : ℚ) (odds_failure : ℚ) :
  odds_success = 5/6 → odds_failure = 6/5 →
  (odds_failure / (odds_success + 1)) = 6/11 := by
  sorry

end NUMINAMATH_CALUDE_car_not_sold_probability_l187_18790


namespace NUMINAMATH_CALUDE_frustum_slant_height_l187_18742

/-- 
Given a cone cut by a plane parallel to its base forming a frustum:
- r: radius of the upper base of the frustum
- 4r: radius of the lower base of the frustum
- 3: slant height of the removed cone
- h: slant height of the frustum

Prove that h = 9
-/
theorem frustum_slant_height (r : ℝ) (h : ℝ) : h / (4 * r) = (h + 3) / (5 * r) → h = 9 := by
  sorry

end NUMINAMATH_CALUDE_frustum_slant_height_l187_18742


namespace NUMINAMATH_CALUDE_original_sales_tax_percentage_l187_18784

/-- Proves that the original sales tax percentage was 3.5% given the conditions -/
theorem original_sales_tax_percentage
  (new_tax_rate : ℚ)
  (market_price : ℚ)
  (tax_difference : ℚ)
  (h1 : new_tax_rate = 10 / 3)
  (h2 : market_price = 6600)
  (h3 : tax_difference = 10.999999999999991)
  : ∃ (original_tax_rate : ℚ), original_tax_rate = 7 / 2 :=
sorry

end NUMINAMATH_CALUDE_original_sales_tax_percentage_l187_18784


namespace NUMINAMATH_CALUDE_sarahs_initial_trucks_l187_18715

/-- Given that Sarah gave away 13 trucks and has 38 trucks remaining,
    prove that she initially had 51 trucks. -/
theorem sarahs_initial_trucks :
  ∀ (initial_trucks given_trucks remaining_trucks : ℕ),
    given_trucks = 13 →
    remaining_trucks = 38 →
    initial_trucks = given_trucks + remaining_trucks →
    initial_trucks = 51 :=
by sorry

end NUMINAMATH_CALUDE_sarahs_initial_trucks_l187_18715


namespace NUMINAMATH_CALUDE_pie_distribution_l187_18722

theorem pie_distribution (T R B S : ℕ) : 
  R = T / 2 →
  B = R - 14 →
  S = (R + B) / 2 →
  T = R + B + S →
  (T = 42 ∧ R = 21 ∧ B = 7 ∧ S = 14) :=
by sorry

end NUMINAMATH_CALUDE_pie_distribution_l187_18722


namespace NUMINAMATH_CALUDE_subset_of_any_set_implies_zero_l187_18753

theorem subset_of_any_set_implies_zero (a : ℝ) :
  (∀ S : Set ℝ, {x : ℝ | a * x = 1} ⊆ S) → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_subset_of_any_set_implies_zero_l187_18753


namespace NUMINAMATH_CALUDE_first_row_dots_l187_18794

def green_dots_sequence (n : ℕ) : ℕ := 3 * n + 3

theorem first_row_dots : green_dots_sequence 0 = 3 := by sorry

end NUMINAMATH_CALUDE_first_row_dots_l187_18794


namespace NUMINAMATH_CALUDE_even_function_property_l187_18702

def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def MonoDecreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x > f y

theorem even_function_property (f : ℝ → ℝ) :
  (∀ x ∈ Set.Icc (-6) 6, HasDerivAt f (f x) x) →
  IsEven f →
  MonoDecreasing f (-6) 0 →
  f 4 - f 1 > 0 :=
sorry

end NUMINAMATH_CALUDE_even_function_property_l187_18702


namespace NUMINAMATH_CALUDE_triangle_angle_determination_l187_18744

theorem triangle_angle_determination (a b c A B C : ℝ) : 
  a = Real.sqrt 3 → 
  b = Real.sqrt 2 → 
  B = π / 4 → 
  (a = 2 * Real.sin (A / 2)) → 
  (b = 2 * Real.sin (B / 2)) → 
  (c = 2 * Real.sin (C / 2)) → 
  A + B + C = π → 
  (A = π / 3 ∨ A = 2 * π / 3) := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_determination_l187_18744


namespace NUMINAMATH_CALUDE_cookie_sharing_proof_l187_18706

/-- Given a total number of cookies and the number of cookies each person gets,
    calculate the number of people sharing the cookies. -/
def number_of_people (total_cookies : ℕ) (cookies_per_person : ℕ) : ℕ :=
  total_cookies / cookies_per_person

/-- Prove that when sharing 24 cookies equally among people,
    with each person getting 4 cookies, the number of people is 6. -/
theorem cookie_sharing_proof :
  number_of_people 24 4 = 6 := by
  sorry

end NUMINAMATH_CALUDE_cookie_sharing_proof_l187_18706


namespace NUMINAMATH_CALUDE_group_size_l187_18717

theorem group_size (n : ℕ) 
  (avg_increase : ℝ) 
  (old_weight new_weight : ℝ) 
  (h1 : avg_increase = 1.5)
  (h2 : old_weight = 65)
  (h3 : new_weight = 74)
  (h4 : n * avg_increase = new_weight - old_weight) : 
  n = 6 := by
sorry

end NUMINAMATH_CALUDE_group_size_l187_18717


namespace NUMINAMATH_CALUDE_regular_polygon_area_l187_18751

theorem regular_polygon_area (n : ℕ) (R : ℝ) : 
  n > 2 → 
  R > 0 → 
  (1 / 2 : ℝ) * n * R^2 * Real.sin ((2 * Real.pi) / n) = 2 * R^2 → 
  n = 12 := by
sorry

end NUMINAMATH_CALUDE_regular_polygon_area_l187_18751


namespace NUMINAMATH_CALUDE_lost_sea_creatures_l187_18773

/-- Represents the count of sea creatures Harry collected --/
structure SeaCreatures where
  seaStars : ℕ
  seashells : ℕ
  snails : ℕ
  crabs : ℕ

/-- Represents the number of each type of sea creature that reproduced --/
structure Reproduction where
  seaStars : ℕ
  seashells : ℕ
  snails : ℕ

def initialCount : SeaCreatures :=
  { seaStars := 34, seashells := 21, snails := 29, crabs := 17 }

def reproductionCount : Reproduction :=
  { seaStars := 5, seashells := 3, snails := 4 }

def finalCount : ℕ := 105

def totalAfterReproduction (initial : SeaCreatures) (reproduction : Reproduction) : ℕ :=
  (initial.seaStars + reproduction.seaStars) +
  (initial.seashells + reproduction.seashells) +
  (initial.snails + reproduction.snails) +
  initial.crabs

theorem lost_sea_creatures : 
  totalAfterReproduction initialCount reproductionCount - finalCount = 8 := by
  sorry

end NUMINAMATH_CALUDE_lost_sea_creatures_l187_18773


namespace NUMINAMATH_CALUDE_five_seventeenths_repetend_l187_18791

/-- The repetend of a rational number a/b is the repeating sequence of digits in its decimal expansion. -/
def repetend (a b : ℕ) : List ℕ := sorry

/-- Returns the first n digits of a list. -/
def firstNDigits (n : ℕ) (l : List ℕ) : List ℕ := sorry

theorem five_seventeenths_repetend :
  firstNDigits 6 (repetend 5 17) = [2, 9, 4, 1, 1, 7] := by sorry

end NUMINAMATH_CALUDE_five_seventeenths_repetend_l187_18791


namespace NUMINAMATH_CALUDE_trig_identity_l187_18723

theorem trig_identity (α : ℝ) : 
  Real.sin (π + α)^2 - Real.cos (π + α) * Real.cos (-α) + 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l187_18723


namespace NUMINAMATH_CALUDE_cave_door_weight_theorem_l187_18769

/-- The weight already on the switch, in pounds. -/
def weight_on_switch : ℕ := 234

/-- The total weight needed, in pounds. -/
def total_weight_needed : ℕ := 712

/-- The additional weight needed to open the cave doors, in pounds. -/
def additional_weight_needed : ℕ := total_weight_needed - weight_on_switch

theorem cave_door_weight_theorem : additional_weight_needed = 478 := by
  sorry

end NUMINAMATH_CALUDE_cave_door_weight_theorem_l187_18769


namespace NUMINAMATH_CALUDE_fifteen_percent_of_900_is_135_l187_18736

theorem fifteen_percent_of_900_is_135 : ∃ x : ℝ, x * 0.15 = 135 ∧ x = 900 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_percent_of_900_is_135_l187_18736


namespace NUMINAMATH_CALUDE_inequality_solution_set_l187_18755

theorem inequality_solution_set : 
  ∀ x : ℝ, (3/20 : ℝ) + |x - 7/40| < (11/40 : ℝ) ↔ x ∈ Set.Ioo (1/20 : ℝ) (3/10 : ℝ) := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l187_18755


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l187_18765

-- Define an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- Define a geometric sequence
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, b (n + 1) = r * b n

theorem arithmetic_sequence_problem (a : ℕ → ℝ) (d : ℝ) :
  d ≠ 0 →
  arithmetic_sequence a d →
  a 3 = 7 →
  geometric_sequence (λ n => a n - 1) →
  a 10 = 21 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l187_18765


namespace NUMINAMATH_CALUDE_frog_jumped_farther_l187_18771

/-- The distance the grasshopper jumped in inches -/
def grasshopper_jump : ℕ := 36

/-- The distance the frog jumped in inches -/
def frog_jump : ℕ := 53

/-- The difference between the frog's jump and the grasshopper's jump -/
def jump_difference : ℕ := frog_jump - grasshopper_jump

theorem frog_jumped_farther : jump_difference = 17 := by
  sorry

end NUMINAMATH_CALUDE_frog_jumped_farther_l187_18771


namespace NUMINAMATH_CALUDE_smallest_common_multiple_12_9_l187_18726

theorem smallest_common_multiple_12_9 : ∃ n : ℕ, n > 0 ∧ 12 ∣ n ∧ 9 ∣ n ∧ ∀ m : ℕ, (m > 0 ∧ 12 ∣ m ∧ 9 ∣ m) → n ≤ m := by
  sorry

end NUMINAMATH_CALUDE_smallest_common_multiple_12_9_l187_18726


namespace NUMINAMATH_CALUDE_point_on_angle_terminal_side_l187_18796

theorem point_on_angle_terminal_side (P : ℝ × ℝ) (θ : ℝ) (h1 : θ = 2 * π / 3) (h2 : P.1 = -1) :
  P.2 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_point_on_angle_terminal_side_l187_18796


namespace NUMINAMATH_CALUDE_dice_probabilities_l187_18709

-- Define the type for a die
def Die : Type := Fin 6

-- Define the sample space
def SampleSpace : Type := Die × Die

-- Define the probability measure
noncomputable def P : Set SampleSpace → ℝ := sorry

-- Define the event of rolling the same number on both dice
def SameNumber : Set SampleSpace :=
  {p : SampleSpace | p.1 = p.2}

-- Define the event of rolling a sum less than 7
def SumLessThan7 : Set SampleSpace :=
  {p : SampleSpace | p.1.val + p.2.val + 2 < 7}

-- Define the event of rolling a sum equal to or greater than 11
def SumGreaterEqual11 : Set SampleSpace :=
  {p : SampleSpace | p.1.val + p.2.val + 2 ≥ 11}

theorem dice_probabilities :
  P SameNumber = 1/6 ∧
  P SumLessThan7 = 5/12 ∧
  P SumGreaterEqual11 = 1/12 := by sorry

end NUMINAMATH_CALUDE_dice_probabilities_l187_18709


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_9_6_l187_18772

theorem sqrt_sum_equals_9_6 (y : ℝ) 
  (h : Real.sqrt (64 - y^2) - Real.sqrt (16 - y^2) = 5) : 
  Real.sqrt (64 - y^2) + Real.sqrt (16 - y^2) = 9.6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_9_6_l187_18772


namespace NUMINAMATH_CALUDE_animal_mortality_probability_l187_18730

/-- The probability of an animal dying in each of the first 3 months, given survival data -/
theorem animal_mortality_probability (total : ℕ) (survivors : ℝ) (p : ℝ) 
  (h_total : total = 400)
  (h_survivors : survivors = 291.6)
  (h_survival_equation : survivors = total * (1 - p)^3) :
  p = 0.1 := by
sorry

end NUMINAMATH_CALUDE_animal_mortality_probability_l187_18730


namespace NUMINAMATH_CALUDE_largest_integer_with_conditions_l187_18788

theorem largest_integer_with_conditions : 
  ∃ (n : ℕ), n = 243 ∧ 
  (∀ m : ℕ, (200 < m ∧ m < 250 ∧ ∃ k : ℕ, 12 * m = k^2) → m ≤ n) :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_with_conditions_l187_18788


namespace NUMINAMATH_CALUDE_relationship_abc_l187_18733

theorem relationship_abc (a b c : ℝ) : 
  a = Real.sqrt 0.6 → 
  b = Real.rpow 0.6 (1/3) → 
  c = Real.log 3 / Real.log 0.6 → 
  c < a ∧ a < b := by
  sorry

end NUMINAMATH_CALUDE_relationship_abc_l187_18733


namespace NUMINAMATH_CALUDE_log_relationship_l187_18739

theorem log_relationship (a b x : ℝ) (h : 0 < a ∧ a ≠ 1 ∧ 0 < b ∧ b ≠ 1 ∧ 0 < x) :
  5 * (Real.log x / Real.log a)^2 + 2 * (Real.log x / Real.log b)^2 = 15 * (Real.log x)^2 / (Real.log a * Real.log b) →
  b = a^((3 + Real.sqrt 37) / 2) ∨ b = a^((3 - Real.sqrt 37) / 2) := by
sorry

end NUMINAMATH_CALUDE_log_relationship_l187_18739


namespace NUMINAMATH_CALUDE_underground_ticket_cost_l187_18735

/-- The cost of one ticket to the underground. -/
def ticket_cost (tickets_per_minute : ℕ) (total_minutes : ℕ) (total_earnings : ℕ) : ℚ :=
  total_earnings / (tickets_per_minute * total_minutes)

/-- Theorem stating that the cost of one ticket is $3. -/
theorem underground_ticket_cost :
  ticket_cost 5 6 90 = 3 := by
  sorry

end NUMINAMATH_CALUDE_underground_ticket_cost_l187_18735


namespace NUMINAMATH_CALUDE_min_value_ab_l187_18748

theorem min_value_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a * b = a + b + 3) :
  a * b ≥ 9 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ * b₀ = a₀ + b₀ + 3 ∧ a₀ * b₀ = 9 :=
sorry

end NUMINAMATH_CALUDE_min_value_ab_l187_18748


namespace NUMINAMATH_CALUDE_impossibility_of_tiling_l187_18770

/-- Represents a tetromino shape -/
inductive TetrominoShape
  | T
  | L
  | I

/-- Represents a 10x10 chessboard -/
def Chessboard := Fin 10 → Fin 10 → Bool

/-- Checks if a given tetromino shape can tile the chessboard -/
def can_tile (shape : TetrominoShape) (board : Chessboard) : Prop :=
  ∃ (tiling : Nat → Nat → Nat → Nat → Bool),
    ∀ (i j : Fin 10), board i j = true ↔ 
      ∃ (x y : Nat), tiling x y i j = true

theorem impossibility_of_tiling (shape : TetrominoShape) :
  ¬∃ (board : Chessboard), can_tile shape board := by
  sorry

#check impossibility_of_tiling TetrominoShape.T
#check impossibility_of_tiling TetrominoShape.L
#check impossibility_of_tiling TetrominoShape.I

end NUMINAMATH_CALUDE_impossibility_of_tiling_l187_18770


namespace NUMINAMATH_CALUDE_triangle_angle_A_l187_18785

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem triangle_angle_A (t : Triangle) : 
  t.a = 4 * Real.sqrt 3 → 
  t.c = 12 → 
  t.C = π / 3 → 
  t.A = π / 6 := by
  sorry


end NUMINAMATH_CALUDE_triangle_angle_A_l187_18785


namespace NUMINAMATH_CALUDE_square_plus_linear_plus_one_l187_18797

theorem square_plus_linear_plus_one (a : ℝ) : 
  a^2 + a - 5 = 0 → a^2 + a + 1 = 6 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_linear_plus_one_l187_18797


namespace NUMINAMATH_CALUDE_marbles_gcd_l187_18707

theorem marbles_gcd (blue : Nat) (white : Nat) (red : Nat) (green : Nat) (yellow : Nat)
  (h_blue : blue = 24)
  (h_white : white = 17)
  (h_red : red = 13)
  (h_green : green = 7)
  (h_yellow : yellow = 5) :
  Nat.gcd blue (Nat.gcd white (Nat.gcd red (Nat.gcd green yellow))) = 1 := by
  sorry

end NUMINAMATH_CALUDE_marbles_gcd_l187_18707


namespace NUMINAMATH_CALUDE_exponential_inequality_l187_18701

theorem exponential_inequality (x y a b : ℝ) (h1 : x > y) (h2 : y > a) (h3 : a > b) (h4 : b > 1) :
  a^x > b^y := by sorry

end NUMINAMATH_CALUDE_exponential_inequality_l187_18701


namespace NUMINAMATH_CALUDE_hyperbola_and_line_equations_l187_18789

/-- Given a hyperbola with specified properties, prove its equation and the equation of a line intersecting it. -/
theorem hyperbola_and_line_equations
  (a b : ℝ)
  (h_a : a > 0)
  (h_b : b > 0)
  (h_asymptote : ∀ x y : ℝ, y = 2 * x → (∃ t : ℝ, y = t * x ∧ y^2 / a^2 - x^2 / b^2 = 1))
  (h_focus_distance : ∃ F : ℝ × ℝ, ∀ x y : ℝ, y = 2 * x → Real.sqrt ((F.1 - x)^2 + (F.2 - y)^2) = 1)
  (h_midpoint : ∃ A B : ℝ × ℝ, A.1 ≠ B.1 ∧ A.2 ≠ B.2 ∧
    (A.2^2 / a^2 - A.1^2 / b^2 = 1) ∧
    (B.2^2 / a^2 - B.1^2 / b^2 = 1) ∧
    ((A.1 + B.1) / 2 = 1) ∧
    ((A.2 + B.2) / 2 = 4)) :
  (a = 2 ∧ b = 1) ∧
  (∀ x y : ℝ, y^2 / 4 - x^2 = 1 ↔ y^2 / a^2 - x^2 / b^2 = 1) ∧
  (∃ k m : ℝ, k = 1 ∧ m = 3 ∧ ∀ x y : ℝ, y^2 / 4 - x^2 = 1 → (x - y + m = 0 ↔ ∃ t : ℝ, x = 1 + t ∧ y = 4 + t)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_and_line_equations_l187_18789


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l187_18795

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x, y = 2^x}
def N : Set ℝ := {y | ∃ x, y = 2 * Real.sin x}

-- State the theorem
theorem intersection_of_M_and_N :
  M ∩ N = {y | 0 < y ∧ y ≤ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l187_18795


namespace NUMINAMATH_CALUDE_stamp_problem_l187_18731

/-- Returns true if postage can be formed with given denominations -/
def can_form_postage (d1 d2 d3 amount : ℕ) : Prop :=
  ∃ (x y z : ℕ), d1 * x + d2 * y + d3 * z = amount

/-- Returns true if n satisfies the stamp problem conditions -/
def satisfies_conditions (n : ℕ) : Prop :=
  n > 0 ∧
  (∀ m : ℕ, m > 70 → can_form_postage 3 n (n+1) m) ∧
  ¬(can_form_postage 3 n (n+1) 70)

theorem stamp_problem :
  ∃! (n : ℕ), satisfies_conditions n ∧ n = 37 :=
sorry

end NUMINAMATH_CALUDE_stamp_problem_l187_18731


namespace NUMINAMATH_CALUDE_max_third_altitude_is_six_l187_18741

/-- A scalene triangle with two known altitudes -/
structure ScaleneTriangle where
  /-- The length of the first known altitude -/
  altitude1 : ℝ
  /-- The length of the second known altitude -/
  altitude2 : ℝ
  /-- The triangle is scalene -/
  scalene : altitude1 ≠ altitude2
  /-- The altitudes are positive -/
  positive1 : altitude1 > 0
  positive2 : altitude2 > 0

/-- The maximum possible integer length of the third altitude -/
def max_third_altitude (t : ScaleneTriangle) : ℕ :=
  6

/-- Theorem stating that the maximum possible integer length of the third altitude is 6 -/
theorem max_third_altitude_is_six (t : ScaleneTriangle) 
  (h1 : t.altitude1 = 6 ∨ t.altitude2 = 6) 
  (h2 : t.altitude1 = 18 ∨ t.altitude2 = 18) : 
  max_third_altitude t = 6 := by
  sorry

#check max_third_altitude_is_six

end NUMINAMATH_CALUDE_max_third_altitude_is_six_l187_18741


namespace NUMINAMATH_CALUDE_pasture_rent_l187_18724

/-- Represents a milkman's grazing details -/
structure MilkmanGrazing where
  cows : ℕ
  months : ℕ

/-- Calculates the total rent of a pasture given the grazing details of milkmen -/
def totalRent (milkmen : List MilkmanGrazing) (aShare : ℕ) : ℕ :=
  let totalCowMonths := milkmen.foldl (fun acc m => acc + m.cows * m.months) 0
  let aMonths := (milkmen.head?).map (fun m => m.cows * m.months)
  match aMonths with
  | some months => (totalCowMonths * aShare) / months
  | none => 0

/-- Theorem stating that the total rent of the pasture is 3250 -/
theorem pasture_rent :
  let milkmen := [
    MilkmanGrazing.mk 24 3,  -- A
    MilkmanGrazing.mk 10 5,  -- B
    MilkmanGrazing.mk 35 4,  -- C
    MilkmanGrazing.mk 21 3   -- D
  ]
  totalRent milkmen 720 = 3250 := by
    sorry

end NUMINAMATH_CALUDE_pasture_rent_l187_18724


namespace NUMINAMATH_CALUDE_no_natural_solutions_l187_18703

theorem no_natural_solutions :
  (∀ x y z : ℕ, x^2 + y^2 + z^2 ≠ 2*x*y*z) ∧
  (∀ x y z u : ℕ, x^2 + y^2 + z^2 + u^2 ≠ 2*x*y*z*u) := by
  sorry

end NUMINAMATH_CALUDE_no_natural_solutions_l187_18703


namespace NUMINAMATH_CALUDE_event_arrangements_l187_18746

def number_of_arrangements (n : ℕ) (k : ℕ) : ℕ :=
  (Nat.choose n k) * k * k

theorem event_arrangements : number_of_arrangements 6 3 = 180 := by
  sorry

end NUMINAMATH_CALUDE_event_arrangements_l187_18746


namespace NUMINAMATH_CALUDE_distance_between_points_l187_18711

/-- The distance between two points (-3, 5) and (4, -9) is √245 -/
theorem distance_between_points : Real.sqrt 245 = Real.sqrt ((4 - (-3))^2 + (-9 - 5)^2) := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l187_18711


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l187_18714

theorem absolute_value_equation_solution :
  let f : ℝ → ℝ := λ x => |x^2 - 4*x + 4| - (3 - x)
  ∃ x₁ x₂ : ℝ, x₁ = (3 + Real.sqrt 5) / 2 ∧
              x₂ = (3 - Real.sqrt 5) / 2 ∧
              f x₁ = 0 ∧ f x₂ = 0 ∧
              ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l187_18714
