import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_interest_calculation_l57_5749

/-- Calculate simple interest -/
noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

theorem total_interest_calculation (loan_B : ℝ) (loan_C : ℝ) (time_B : ℝ) (time_C : ℝ) (rate : ℝ) :
  loan_B = 4000 →
  loan_C = 2000 →
  time_B = 2 →
  time_C = 4 →
  rate = 13.75 →
  simple_interest loan_B rate time_B + simple_interest loan_C rate time_C = 2200 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_interest_calculation_l57_5749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_equation_l57_5788

-- Define the custom operation
noncomputable def circle_op (a b : ℝ) : ℝ := (Real.sqrt (3 * a + 2 * b)) ^ 5

-- State the theorem
theorem solve_equation (y : ℝ) :
  circle_op 6 y = 243 → y = -4.5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_equation_l57_5788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_product_of_logarithmic_equation_l57_5770

open Real

theorem root_product_of_logarithmic_equation :
  let f : ℝ → ℝ := λ x => (log x)^2 + (log 2 + log 3) * (log x) + (log 2) * (log 3)
  ∃ x₁ x₂ : ℝ, (x₁ > 0 ∧ x₂ > 0) ∧ (f x₁ = 0 ∧ f x₂ = 0) ∧ (x₁ * x₂ = 1/6) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_product_of_logarithmic_equation_l57_5770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_x_150_divided_by_x_plus_1_cubed_l57_5727

theorem remainder_x_150_divided_by_x_plus_1_cubed (x : ℝ) :
  ∃ q : Polynomial ℝ, (Polynomial.X : Polynomial ℝ)^150 = 
    (Polynomial.X + 1)^3 * q + (11175 * Polynomial.X^2 + 22200 * Polynomial.X + 11026) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_x_150_divided_by_x_plus_1_cubed_l57_5727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_integers_from_multiple_condition_l57_5765

theorem equal_integers_from_multiple_condition (a b : ℕ) 
  (ha : a > 0) (hb : b > 0)
  (h : ∀ n : ℕ, n > 0 → ∃ k : ℕ, b ^ n + n = k * (a ^ n + n)) : 
  a = b := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_integers_from_multiple_condition_l57_5765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_is_five_eightyone_l57_5778

/-- Regular tetrahedron with inscribed and circumscribed spheres -/
structure RegularTetrahedron where
  h : ℝ  -- height of the tetrahedron
  r : ℝ  -- inradius
  R : ℝ  -- circumradius
  spheres : Fin 5 → Sphere  -- 5 spheres: 1 inscribed, 4 tangent to faces

/-- The probability that a random point in the circumscribed sphere of a regular tetrahedron
    lies in one of the five smaller spheres (inscribed sphere and four spheres tangent to faces) -/
noncomputable def probability_in_small_spheres (t : RegularTetrahedron) : ℝ :=
  let circumscribed_volume := (4/3) * Real.pi * t.R^3
  let small_spheres_volume := (5/3) * Real.pi * (t.h/4)^3
  small_spheres_volume / circumscribed_volume

/-- Theorem stating the probability is 5/81 -/
theorem probability_is_five_eightyone (t : RegularTetrahedron) :
  probability_in_small_spheres t = 5/81 := by
  sorry

#check probability_is_five_eightyone

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_is_five_eightyone_l57_5778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_renne_savings_time_l57_5780

noncomputable def months_to_save (monthly_earnings : ℝ) (savings_percentage : ℝ) 
                   (monthly_expenses : ℝ) (vehicle_cost : ℝ) : ℝ :=
  vehicle_cost / ((monthly_earnings * savings_percentage) - monthly_expenses)

theorem renne_savings_time : 
  let monthly_earnings : ℝ := 4000
  let savings_percentage : ℝ := 0.5
  let monthly_expenses : ℝ := 1500
  let vehicle_cost : ℝ := 25000
  months_to_save monthly_earnings savings_percentage monthly_expenses vehicle_cost = 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_renne_savings_time_l57_5780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_common_points_range_two_intersections_product_l57_5716

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := Real.log x
def g (a : ℝ) (x : ℝ) : ℝ := a * x

-- Part I: No common points between f and g
theorem no_common_points_range (a : ℝ) :
  (∀ x > 0, f x ≠ g a x) → a > 1 / Real.exp 1 := by
  sorry

-- Part II: Two intersection points imply x₁x₂ > e²
theorem two_intersections_product (a x₁ x₂ : ℝ) :
  x₁ ≠ x₂ → x₁ > 0 → x₂ > 0 → f x₁ = g a x₁ → f x₂ = g a x₂ → x₁ * x₂ > Real.exp 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_common_points_range_two_intersections_product_l57_5716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kennedy_airport_fraction_approx_one_third_l57_5751

/-- The total number of airline passengers traveling to or from the United States in 1979 -/
noncomputable def total_passengers : ℝ := 37.3

/-- The number of passengers that used Logan Airport -/
noncomputable def logan_passengers : ℝ := 1.036111111111111

/-- The number of passengers that used Kennedy Airport -/
noncomputable def kennedy_passengers : ℝ := 4 * 3 * logan_passengers

/-- The fraction of passengers that used Kennedy Airport -/
noncomputable def kennedy_fraction : ℝ := kennedy_passengers / total_passengers

/-- Theorem stating that the fraction of passengers using Kennedy Airport is approximately 1/3 -/
theorem kennedy_airport_fraction_approx_one_third :
  |kennedy_fraction - 1/3| < 0.001 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kennedy_airport_fraction_approx_one_third_l57_5751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_equation_roots_l57_5771

open Real

theorem trig_equation_roots (θ : ℝ) (a : ℝ) 
  (h1 : sin θ ^ 2 - a * sin θ + a = 0)
  (h2 : cos θ ^ 2 - a * cos θ + a = 0) :
  (cos (π/2 + θ) + sin (3*π/2 + θ) = Real.sqrt 2 - 1) ∧
  (tan (π - θ) - 1 / tan θ = Real.sqrt 2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_equation_roots_l57_5771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_number_l57_5791

theorem largest_number (a b c d : ℝ) (h1 : a = Real.sqrt 2) (h2 : b = 0) (h3 : c = -1) (h4 : d = 2) :
  max (max (max a b) c) d = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_number_l57_5791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_constant_sum_l57_5768

/-- A moving line passing through (0,m) intersects the parabola x^2 = -16y at A and B. -/
def intersectionPoints (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ t : ℝ, p.2 = t * p.1 + m ∧ p.1^2 = -16 * p.2}

/-- The distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The sum of inverse squares of distances from (0,m) to intersection points is constant -/
def inverseSquareSum (m : ℝ) : ℝ → Prop :=
  fun k => ∀ A B : ℝ × ℝ, A ∈ intersectionPoints m → B ∈ intersectionPoints m → A ≠ B →
    1 / (distance A (0, m))^2 + 1 / (distance B (0, m))^2 = k

theorem intersection_constant_sum (m : ℝ) :
  (∃ k : ℝ, inverseSquareSum m k) → m = -8 := by
  sorry

#check intersection_constant_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_constant_sum_l57_5768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_platform_crossing_time_l57_5712

/-- Calculates the time taken for a train to cross a platform -/
noncomputable def time_to_cross_platform (train_length platform_length : ℝ) (time_to_cross_pole : ℝ) : ℝ :=
  let train_speed := train_length / time_to_cross_pole
  let total_distance := train_length + platform_length
  total_distance / train_speed

/-- Theorem: Given the specified conditions, the train takes 39 seconds to cross the platform -/
theorem train_platform_crossing_time :
  time_to_cross_platform 300 350 18 = 39 := by
  -- Unfold the definition of time_to_cross_platform
  unfold time_to_cross_platform
  -- Simplify the expression
  simp
  -- The proof steps would go here, but we'll use sorry for now
  sorry

-- We can't use #eval with noncomputable definitions, so we'll use #check instead
#check time_to_cross_platform 300 350 18

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_platform_crossing_time_l57_5712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_result_l57_5703

noncomputable def line_equation (x y : ℝ) : Prop := y = 3 * x + 2

noncomputable def vector_on_line (v : ℝ × ℝ) : Prop :=
  line_equation v.1 v.2

noncomputable def projection (v w : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := v.1 * w.1 + v.2 * w.2
  let magnitude_squared := w.1 * w.1 + w.2 * w.2
  let scalar := dot_product / magnitude_squared
  (scalar * w.1, scalar * w.2)

theorem projection_result (w : ℝ × ℝ) :
  ∀ v : ℝ × ℝ, vector_on_line v → projection v w = (-3/5, 1/5) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_result_l57_5703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_difference_proof_optimal_pair_proof_l57_5796

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  is_valid : hundreds ≥ 1 ∧ hundreds ≤ 9 ∧ tens ≥ 0 ∧ tens ≤ 9 ∧ ones ≥ 0 ∧ ones ≤ 9

/-- Converts a ThreeDigitNumber to a natural number -/
def threeDigitToNat (n : ThreeDigitNumber) : Nat :=
  n.hundreds * 100 + n.tens * 10 + n.ones

/-- Checks if a digit is valid for a specific position based on the malfunctioning display -/
def isValidDigit (position : String) (digit : Nat) : Bool :=
  match position with
  | "hundreds_minuend" => digit ∈ [3, 5, 9]
  | "tens_minuend" => digit ∈ [2, 3, 7]
  | "ones_minuend" => digit ∈ [3, 4, 8, 9]
  | "hundreds_subtrahend" => digit ∈ [2, 3, 7]
  | "tens_subtrahend" => digit ∈ [3, 5, 9]
  | "ones_subtrahend" => digit ∈ [1, 4, 7]
  | _ => false

/-- Checks if a ThreeDigitNumber is valid based on the malfunctioning display -/
def isValidNumber (n : ThreeDigitNumber) (numberType : String) : Bool :=
  isValidDigit (numberType ++ "_hundreds") n.hundreds &&
  isValidDigit (numberType ++ "_tens") n.tens &&
  isValidDigit (numberType ++ "_ones") n.ones

theorem max_difference_proof :
  ∀ (minuend subtrahend : ThreeDigitNumber),
    isValidNumber minuend "minuend" →
    isValidNumber subtrahend "subtrahend" →
    threeDigitToNat minuend - threeDigitToNat subtrahend ≤ 529 :=
by sorry

theorem optimal_pair_proof :
  ∃ (minuend subtrahend : ThreeDigitNumber),
    isValidNumber minuend "minuend" ∧
    isValidNumber subtrahend "subtrahend" ∧
    threeDigitToNat minuend - threeDigitToNat subtrahend = 529 ∧
    threeDigitToNat minuend = 923 ∧
    threeDigitToNat subtrahend = 394 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_difference_proof_optimal_pair_proof_l57_5796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l57_5794

-- Define the curves C₁ and C₂
noncomputable def C₁ (α : ℝ) : ℝ × ℝ := (2 + 2 * Real.cos α, 2 * Real.sin α)

noncomputable def C₂ (θ : ℝ) : ℝ × ℝ := (2 * Real.cos θ * Real.cos θ, 2 * Real.cos θ * Real.sin θ)

-- Define the area of triangle POQ
noncomputable def triangle_area (P Q : ℝ × ℝ) : ℝ :=
  abs ((P.1 * Q.2 - P.2 * Q.1) / 2)

-- Theorem statement
theorem max_triangle_area :
  ∃ (α θ : ℝ),
    let P := C₁ α
    let Q := C₂ θ
    (P.1 * Q.1 + P.2 * Q.2) = 0 ∧ 
    ∀ (β γ : ℝ),
      let R := C₁ β
      let S := C₂ γ
      (R.1 * S.1 + R.2 * S.2) = 0 →
      triangle_area P Q ≥ triangle_area R S ∧
      triangle_area P Q = 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l57_5794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_l57_5717

-- Define the length of the train in meters
noncomputable def train_length : ℝ := 270

-- Define the speed of the train in km/hr
noncomputable def train_speed_kmh : ℝ := 108

-- Convert km/hr to m/s
noncomputable def km_per_hr_to_m_per_s (speed : ℝ) : ℝ := speed * (5/18)

-- Calculate the speed in m/s
noncomputable def train_speed_ms : ℝ := km_per_hr_to_m_per_s train_speed_kmh

-- Theorem statement
theorem train_passing_time :
  train_length / train_speed_ms = 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_l57_5717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sarah_reading_homework_pages_l57_5782

/-- Given Sarah's homework information, prove the number of reading homework pages -/
theorem sarah_reading_homework_pages 
  (math_pages : ℕ) 
  (problems_per_page : ℕ) 
  (total_problems : ℕ) 
  (h1 : math_pages = 4)
  (h2 : problems_per_page = 4)
  (h3 : total_problems = 40) :
  (total_problems - math_pages * problems_per_page) / problems_per_page = 6 := by
  -- Proof steps will go here
  sorry

#check sarah_reading_homework_pages

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sarah_reading_homework_pages_l57_5782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_manufacturer_cost_is_seventeen_l57_5750

/-- Represents the cost and profit structure of a glasses supply chain --/
structure GlassesSupplyChain where
  manufacturer_cost : ℝ
  manufacturer_profit_rate : ℝ
  wholesaler_profit_rate : ℝ
  retailer_profit_rate : ℝ
  customer_price : ℝ

/-- Calculates the manufacturer's cost price given the supply chain structure --/
noncomputable def calculate_manufacturer_cost (chain : GlassesSupplyChain) : ℝ :=
  chain.customer_price / 
  ((1 + chain.retailer_profit_rate) * 
   (1 + chain.wholesaler_profit_rate) * 
   (1 + chain.manufacturer_profit_rate))

/-- Theorem stating that the manufacturer's cost is approximately 17 --/
theorem manufacturer_cost_is_seventeen (chain : GlassesSupplyChain) 
  (h1 : chain.manufacturer_profit_rate = 0.18)
  (h2 : chain.wholesaler_profit_rate = 0.20)
  (h3 : chain.retailer_profit_rate = 0.25)
  (h4 : chain.customer_price = 30.09) :
  ∃ ε > 0, |calculate_manufacturer_cost chain - 17| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_manufacturer_cost_is_seventeen_l57_5750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_theorem_l57_5798

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola defined by y^2 = 8x -/
def Parabola := {p : Point | p.y^2 = 8 * p.x}

/-- The focus of the parabola -/
def focus : Point := ⟨4, 0⟩

/-- The origin -/
def origin : Point := ⟨0, 0⟩

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

theorem parabola_distance_theorem :
  ∀ M : Point,
  M ∈ Parabola →
  distance M focus = 4 →
  distance M origin = 2 * Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_theorem_l57_5798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_range_l57_5723

-- Define the function h as noncomputable
noncomputable def h (t : ℝ) : ℝ := (t^2 + 2*t) / (t^2 + 2*t + 3)

-- State the theorem
theorem h_range :
  ∀ y : ℝ, (∃ t : ℝ, h t = y) ↔ -1/2 ≤ y ∧ y ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_range_l57_5723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l57_5736

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  m : ℝ  -- slope
  c : ℝ  -- y-intercept

noncomputable def Ellipse.rightFocus (e : Ellipse) : Point :=
  { x := (e.a^2 - e.b^2).sqrt, y := 0 }

def Ellipse.leftVertex (e : Ellipse) : Point :=
  { x := -e.a, y := 0 }

def Ellipse.rightVertex (e : Ellipse) : Point :=
  { x := e.a, y := 0 }

noncomputable def distance (p1 p2 : Point) : ℝ :=
  ((p1.x - p2.x)^2 + (p1.y - p2.y)^2).sqrt

/-- The main theorem -/
theorem ellipse_theorem (e : Ellipse) 
  (h1 : e.rightFocus.x = 1) -- Right focus lies on 2x - y - 2 = 0
  (h2 : distance e.leftVertex e.rightFocus = 3 * distance e.rightVertex e.rightFocus) :
  (∃ l : Line, 
    l.m = -1/4 ∧ 
    (∃ p q n : Point, 
      p.x^2 / 4 + p.y^2 / 3 = 1 ∧
      q.x^2 / 4 + q.y^2 / 3 = 1 ∧
      n.x = (p.x + q.x) / 2 ∧
      n.y = (p.y + q.y) / 2 ∧
      l.m * (n.x - (-2)) + 0 = n.y ∧
      (n.y - 0) / (n.x - (-2)) = 2/5 ∧
      l.m * (4 - p.x) = q.y - p.y)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l57_5736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_solid_revolution_l57_5709

noncomputable section

-- Define the functions that bound the region
def f (x : ℝ) : ℝ := x^3
noncomputable def g (x : ℝ) : ℝ := Real.sqrt x

-- Define the volume of the solid of revolution
noncomputable def volume : ℝ := Real.pi * ∫ x in (0)..(1), (g x)^2 - (f x)^2

-- Theorem statement
theorem volume_of_solid_revolution : volume = (5 * Real.pi) / 14 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_solid_revolution_l57_5709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_property_l57_5790

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
noncomputable def sum_arithmetic (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) * (a 1 + a n) / 2

theorem arithmetic_sequence_sum_property (a : ℕ → ℝ) :
  arithmetic_sequence a →
  sum_arithmetic a 10 = 120 →
  a 2 + a 9 = 24 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_property_l57_5790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_range_l57_5775

open Real

theorem triangle_ratio_range (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →  -- Acute triangle condition
  0 < a ∧ 0 < b ∧ 0 < c →  -- Positive side lengths
  a = 2 →  -- Given condition
  tan A = (cos A + cos C) / (sin A + sin C) →  -- Given condition
  a / sin A = b / sin B →  -- Sine law
  a / sin A = c / sin C →  -- Sine law
  (4 * Real.sqrt 3) / 3 < (b + c) / (sin B + sin C) ∧ 
  (b + c) / (sin B + sin C) < 4 := by
  sorry

#check triangle_ratio_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_range_l57_5775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phirme_characterization_l57_5748

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- Phirme sequence property -/
def is_phirme (a : ℕ → ℤ) (k : ℕ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → a n + a (n + 1) = fib (n + k)

/-- Main theorem: characterization of phirme sequences -/
theorem phirme_characterization (a : ℕ → ℤ) (k : ℕ) :
  is_phirme a k →
  ∃ d : ℤ, ∀ n : ℕ, n ≥ 1 → a n = (fib (n + k - 2) : ℤ) + (-1 : ℤ) ^ (n - 1) * d :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_phirme_characterization_l57_5748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_O_percentage_in_HClO2_l57_5707

/-- Represents the mass percentage of an element in a compound -/
noncomputable def MassPercentage (elementMass : ℝ) (totalMass : ℝ) : ℝ :=
  (elementMass / totalMass) * 100

/-- Molar mass of Hydrogen in g/mol -/
def H_mass : ℝ := 1.01

/-- Molar mass of Chlorine in g/mol -/
def Cl_mass : ℝ := 35.45

/-- Molar mass of Oxygen in g/mol -/
def O_mass : ℝ := 16.00

/-- Molar mass of Chlorous acid (HClO2) in g/mol -/
def HClO2_mass : ℝ := H_mass + Cl_mass + 2 * O_mass

/-- Mass of Oxygen in Chlorous acid (HClO2) in g/mol -/
def O_in_HClO2_mass : ℝ := 2 * O_mass

/-- Theorem: The mass percentage of Oxygen in Chlorous acid is approximately 46.75% -/
theorem O_percentage_in_HClO2 : 
  ∃ (x : ℝ), abs (MassPercentage O_in_HClO2_mass HClO2_mass - x) < 0.01 ∧ x = 46.75 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_O_percentage_in_HClO2_l57_5707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l57_5776

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := (2 : ℝ)^x - (2 : ℝ)^(-x)
noncomputable def g (x : ℝ) : ℝ := (2 : ℝ)^x + (2 : ℝ)^(-x)

-- Define the propositions
def p : Prop := ∀ x y : ℝ, x < y → f y < f x
def q : Prop := ∀ x y : ℝ, x < y → g x < g y

-- State the theorem
theorem problem_solution :
  (¬p ∨ q) ∧ ¬(p ∧ q) ∧ ¬(p ∨ q) ∧ ¬(¬p ∧ q) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l57_5776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_specific_triangle_l57_5710

/-- The distance from the center of a sphere to the plane of an inscribed triangle -/
noncomputable def distance_to_triangle_plane (r : ℝ) (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let circumradius := (a * b * c) / (4 * area)
  Real.sqrt (r^2 - circumradius^2)

/-- Theorem: The distance from the center of a sphere with radius 15 to the plane of an inscribed triangle with side lengths 9, 10, and 11 is 15√15/4 -/
theorem distance_to_specific_triangle :
  distance_to_triangle_plane 15 9 10 11 = 15 * Real.sqrt 15 / 4 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_specific_triangle_l57_5710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_E_and_max_area_l57_5741

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 8

-- Define point A
def point_A : ℝ × ℝ := (1, 0)

-- Define the curve E
def curve_E (x y : ℝ) : Prop := x^2/2 + y^2 = 1

-- Define a line
def line (k m x y : ℝ) : Prop := y = k * x + m

-- Helper function to calculate triangle area
noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem curve_E_and_max_area :
  ∃ (E : Set (ℝ × ℝ)),
    (∀ x y, (x, y) ∈ E ↔ curve_E x y) ∧
    (∀ k m,
      let M_N := {(x, y) | curve_E x y ∧ line k m x y}
      let area := Real.sqrt 2 / 2
      ∀ M N, M ∈ M_N → N ∈ M_N → M ≠ N →
        triangle_area (0, 0) M N ≤ area ∧
        ∃ M' N', M' ∈ M_N ∧ N' ∈ M_N ∧ M' ≠ N' ∧ triangle_area (0, 0) M' N' = area) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_E_and_max_area_l57_5741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l57_5713

noncomputable def f (x : ℝ) := 3 * Real.sin (x / 2 + Real.pi / 6) + 3

theorem f_properties :
  -- Minimum positive period
  (∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, f (x + T) = f x ∧ ∀ T' : ℝ, 0 < T' ∧ T' < T → ∃ x : ℝ, f (x + T') ≠ f x) ∧
  -- Period value
  (let T := 4 * Real.pi; ∀ x : ℝ, f (x + T) = f x) ∧
  -- Monotonically increasing interval
  (∀ k : ℤ, ∀ x y : ℝ, -4*Real.pi/3 + 4*k*Real.pi ≤ x ∧ x < y ∧ y ≤ 2*Real.pi/3 + 4*k*Real.pi → f x < f y) ∧
  -- Minimum and maximum values for x ∈ [π/3, 4π/3]
  (∀ x : ℝ, Real.pi/3 ≤ x ∧ x ≤ 4*Real.pi/3 → f x ≥ 9/2 ∧ f x ≤ 6) ∧
  f (4*Real.pi/3) = 9/2 ∧
  f (2*Real.pi/3) = 6 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l57_5713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_difference_l57_5715

open Real Set

noncomputable def f (x : ℝ) : ℝ := (2 * sin x * cos x) / (1 + sin x + cos x)

def I : Set ℝ := Ioo 0 (π / 2)

theorem f_max_min_difference :
  ∃ (M N : ℝ), (∀ x ∈ I, f x ≤ M ∧ N ≤ f x) ∧ (M - N = Real.sqrt 2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_difference_l57_5715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l57_5762

noncomputable section

/-- The curve f(x) = x^2 -/
def f (x : ℝ) : ℝ := x^2

/-- Point P on the curve -/
def P : ℝ × ℝ := (-1, 1)

/-- Point Q on the curve -/
def Q : ℝ × ℝ := (2, 4)

/-- Slope of line PQ -/
noncomputable def m_PQ : ℝ := (Q.2 - P.2) / (Q.1 - P.1)

/-- Derivative of f -/
def f' (x : ℝ) : ℝ := 2 * x

/-- Point of tangency -/
def tangent_point : ℝ × ℝ := (1/2, 1/4)

theorem tangent_line_equation :
  ∀ (x y : ℝ), (f P.1 = P.2) ∧ (f Q.1 = Q.2) →
  (∃ (t : ℝ), f' t = m_PQ ∧ f t = tangent_point.2) →
  (4 * x - 4 * y - 1 = 0 ↔ y = tangent_point.2 + m_PQ * (x - tangent_point.1)) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l57_5762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_rotation_sum_l57_5747

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A triangle defined by three points -/
structure Triangle where
  a : Point2D
  b : Point2D
  c : Point2D

/-- Rotation parameters -/
structure RotationParams where
  n : ℝ  -- angle in degrees
  u : ℝ  -- x-coordinate of rotation center
  v : ℝ  -- y-coordinate of rotation center

/-- The problem statement -/
theorem triangle_rotation_sum (def1 : Triangle) (def2 : Triangle) (rot : RotationParams) : 
  def1.a = Point2D.mk 0 0 ∧ 
  def1.b = Point2D.mk 0 10 ∧ 
  def1.c = Point2D.mk 14 0 ∧
  def2.a = Point2D.mk 20 15 ∧
  def2.b = Point2D.mk 30 15 ∧
  def2.c = Point2D.mk 20 5 ∧
  0 < rot.n ∧ rot.n < 180 ∧
  -- Assume there exists a rotation function that transforms def1 to def2
  (∃ (rotate : Triangle → RotationParams → Triangle), rotate def1 rot = def2) →
  rot.n + rot.u + rot.v = 110 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_rotation_sum_l57_5747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_divisors_count_l57_5708

theorem odd_divisors_count (n : Nat) (h : n = 2^3 * 5^2 * 11^1) :
  (Finset.filter (fun d => d % 2 = 1) (Nat.divisors n)).card = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_divisors_count_l57_5708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_planes_normal_vectors_l57_5756

/-- Given two planes α and β with normal vectors, prove that if they are parallel, then x = 1/2 -/
theorem parallel_planes_normal_vectors (x : ℝ) : 
  let a : Fin 3 → ℝ := ![(-1 : ℝ), 2, 4]  -- normal vector of plane α
  let b : Fin 3 → ℝ := ![x, -1, -2] -- normal vector of plane β
  (∃ (k : ℝ), a = k • b) →        -- planes are parallel if their normal vectors are scalar multiples
  x = 1/2 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_planes_normal_vectors_l57_5756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_specific_equation_l57_5789

/-- Represents a hyperbola with semi-major axis a, semi-minor axis b, and focal distance c -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := h.c / h.a

/-- The left focus of a hyperbola -/
def leftFocus (h : Hyperbola) : ℝ × ℝ := (-h.c, 0)

/-- The equation of a hyperbola in standard form -/
def hyperbola_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / h.a^2 - y^2 / h.b^2 = 1

theorem hyperbola_specific_equation :
  ∀ h : Hyperbola,
    eccentricity h = 5/3 →
    leftFocus h = (-5, 0) →
    ∀ x y : ℝ,
      hyperbola_equation h x y ↔ x^2 / 9 - y^2 / 16 = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_specific_equation_l57_5789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_permutation_sum_to_all_nines_l57_5759

def digits (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec go (m : ℕ) : List ℕ :=
    if m = 0 then [] else (m % 10) :: go (m / 10)
  go n |>.reverse

def is_permutation (x y : ℕ) : Prop :=
  ∃ (perm : List ℕ → List ℕ), perm (digits x) = digits y

def sum_of_1111_nines : ℕ :=
  (10^1111 - 1) / 9

theorem no_permutation_sum_to_all_nines :
  ¬ ∃ (x y : ℕ), is_permutation x y ∧ x + y = sum_of_1111_nines :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_permutation_sum_to_all_nines_l57_5759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_intersects_median_once_l57_5729

/-- Given complex points A, B, C represented by z₀ = ai, z₁ = 1/2 + bi, z₂ = 1 + ci,
    where a, b, c are real numbers and A, B, C are not collinear,
    prove that the curve z = z₀ cos⁴t + 2z₁ cos²t sin²t + z₂ sin⁴t
    intersects the median parallel to AC in triangle ABC at only one point. -/
theorem curve_intersects_median_once
  (a b c : ℝ) 
  (h_not_collinear : a + c - 2*b ≠ 0) :
  ∃! p : ℂ,
    (∃ t : ℝ, p = Complex.I * a * (Real.cos t)^4 + 
              (1/2 + Complex.I * b) * 2 * (Real.cos t)^2 * (Real.sin t)^2 + 
              (1 + Complex.I * c) * (Real.sin t)^4) ∧
    (p.im = (c - a) * p.re + (3*a + 2*b - c)/4) ∧
    p = Complex.ofReal (1/2) + Complex.I * ((a + c + 2*b)/4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_intersects_median_once_l57_5729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_road_travel_cost_l57_5772

/-- Calculate the cost of traveling two intersecting roads in a rectangular lawn. -/
theorem road_travel_cost (lawn_length lawn_width road_width cost_per_sqm : ℝ) :
  lawn_length = 70 ∧
  lawn_width = 60 ∧
  road_width = 10 ∧
  cost_per_sqm = 3 →
  (road_width * lawn_width + road_width * lawn_length - road_width * road_width) * cost_per_sqm = 3600 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_road_travel_cost_l57_5772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_target_function_properties_l57_5746

/-- Represents a quadratic function of the form ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Evaluates the quadratic function at a given x -/
noncomputable def QuadraticFunction.evaluate (f : QuadraticFunction) (x : ℝ) : ℝ :=
  f.a * x^2 + f.b * x + f.c

/-- Calculates the vertex of a quadratic function -/
noncomputable def QuadraticFunction.vertex (f : QuadraticFunction) : ℝ × ℝ :=
  let x := -f.b / (2 * f.a)
  (x, f.evaluate x)

/-- Checks if a quadratic function has a vertical axis of symmetry -/
def QuadraticFunction.hasVerticalAxisOfSymmetry (f : QuadraticFunction) : Prop :=
  f.a ≠ 0

/-- The specific quadratic function we're proving about -/
def targetFunction : QuadraticFunction :=
  { a := -3, b := 18, c := -22 }

/-- Main theorem stating the properties of the target function -/
theorem target_function_properties :
  (targetFunction.vertex = (3, 5)) ∧
  targetFunction.hasVerticalAxisOfSymmetry ∧
  (targetFunction.evaluate 4 = 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_target_function_properties_l57_5746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circle_equation_l57_5793

/-- A circle with radius 6 that is tangent to the x-axis and internally tangent to another circle -/
structure TangentCircle where
  center : ℝ × ℝ
  radius : ℝ
  tangent_to_x_axis : (center.2) = radius
  internally_tangent : (center.1 - 0)^2 + (center.2 - 3)^2 = (1 + radius)^2

/-- The equation of the tangent circle -/
def circle_equation (c : TangentCircle) : ℝ → ℝ → Prop :=
  λ x y => (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

theorem tangent_circle_equation :
  ∀ c : TangentCircle,
  c.radius = 6 →
  (circle_equation c = λ x y => (x - 4)^2 + (y - 6)^2 = 36) ∨
  (circle_equation c = λ x y => (x + 4)^2 + (y - 6)^2 = 36) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circle_equation_l57_5793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_answer_l57_5784

def diagram_type : String := "Knowledge Structure Diagram"

theorem correct_answer : diagram_type = "Knowledge Structure Diagram" := by
  rfl

#check correct_answer

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_answer_l57_5784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_l57_5799

noncomputable section

/-- The curve function -/
def f (x : ℝ) : ℝ := Real.log (2 * x - 1)

/-- The line function -/
def g (x y : ℝ) : ℝ := 2 * x - y + 3

/-- The distance function between a point (x, f(x)) and the line -/
noncomputable def distance (x : ℝ) : ℝ := 
  |g x (f x)| / Real.sqrt (2^2 + (-1)^2)

theorem shortest_distance :
  ∃ (x : ℝ), ∀ (y : ℝ), distance x ≤ distance y ∧ distance x = Real.sqrt 5 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_l57_5799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_value_l57_5766

/-- The value of φ that satisfies the given conditions -/
noncomputable def phi : ℝ := 13 * Real.pi / 24

/-- The original function -/
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 2 * Real.sin (2 * x + Real.pi / 3)

/-- The translated function -/
noncomputable def g (x : ℝ) : ℝ := f (x - phi)

/-- The target function -/
noncomputable def h (x : ℝ) : ℝ := 2 * Real.sin x * (Real.sin x - Real.cos x) - 1

theorem phi_value :
  0 < phi ∧ phi < Real.pi ∧ 
  (∀ x, g x = h x) →
  phi = 13 * Real.pi / 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_value_l57_5766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_return_to_original_l57_5785

theorem stock_return_to_original (x : ℝ) (h : x > 0) : 
  x = (1 - 3/13) * (1.3 * x) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_return_to_original_l57_5785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_eccentricity_l57_5700

-- Define the geometric sequence condition
def is_geometric_sequence (a b c : ℝ) : Prop :=
  b * b = a * c

-- Define the eccentricity of a conic section
noncomputable def eccentricity (m : ℝ) : ℝ :=
  if m > 0 then
    Real.sqrt (5 / 6)
  else if m < 0 then
    Real.sqrt 7
  else
    0  -- undefined for m = 0

-- Theorem statement
theorem conic_eccentricity (m : ℝ) :
  is_geometric_sequence 4 m 9 →
  (eccentricity m = Real.sqrt 30 / 6 ∨ eccentricity m = Real.sqrt 7) :=
by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_eccentricity_l57_5700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shop_annual_rent_per_sqft_l57_5783

/-- Given a shop with dimensions 20 feet × 18 feet and a monthly rent of Rs. 1440,
    the annual rent per square foot is Rs. 48. -/
theorem shop_annual_rent_per_sqft :
  let shop_length : ℝ := 20
  let shop_width : ℝ := 18
  let monthly_rent : ℝ := 1440
  let shop_area : ℝ := shop_length * shop_width
  let annual_rent : ℝ := monthly_rent * 12
  annual_rent / shop_area = 48 := by
  sorry

-- Remove the #eval statement as it's not necessary for the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shop_annual_rent_per_sqft_l57_5783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_slope_ln_x_l57_5711

open Real

theorem tangent_line_slope_ln_x : 
  ∃ (a : ℝ), a > 0 ∧ 
  (∀ t : ℝ, t * (1/a) = Real.log t - Real.log a → t = 0) ∧ 
  (1/a) = 1/Real.exp 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_slope_ln_x_l57_5711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_vertex_distance_l57_5738

/-- Given a hyperbola with equation x²/36 - y²/16 = 1, 
    the distance between its vertices is 12. -/
theorem hyperbola_vertex_distance : 
  ∃ (d : ℝ), d = 12 ∧ 
  ∀ (x y : ℝ), x^2/36 - y^2/16 = 1 → 
  d = 2 * (Real.sqrt 36) :=
by
  -- We use 'sorry' to skip the proof for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_vertex_distance_l57_5738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_triangle_properties_l57_5742

-- Define an acute triangle
structure AcuteTriangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  acute : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π

-- Define the theorem
theorem acute_triangle_properties (t : AcuteTriangle) 
  (h1 : Real.sqrt 3 * t.a = 2 * t.c * Real.sin t.A)
  (h2 : t.c = Real.sqrt 13)
  (h3 : (1/2) * t.a * t.b * Real.sin t.C = 3 * Real.sqrt 3) :
  t.C = π/3 ∧ t.a + t.b = 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_triangle_properties_l57_5742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_l57_5758

theorem range_of_x (x : ℝ) : 
  Real.sqrt ((2 - 3 * abs x) ^ 2) = 2 + 3 * x → 
  -2/3 ≤ x ∧ x ≤ 0 := by 
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_l57_5758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cave_depth_calculation_l57_5795

/-- The depth of a cave, given the time for a stone to fall and its sound to return --/
noncomputable def cave_depth (t c g : ℝ) : ℝ :=
  (c / g) * (c + g * t - Real.sqrt (c^2 + 2 * g * c * t))

/-- Theorem stating the depth of the cave --/
theorem cave_depth_calculation :
  let t : ℝ := 25  -- Time in seconds
  let c : ℝ := 343  -- Speed of sound in m/s
  let g : ℝ := 9.81  -- Acceleration due to gravity in m/s²
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ 
    |cave_depth t c g - 1867| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cave_depth_calculation_l57_5795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_term_is_negative_three_sixteenths_l57_5724

/-- An arithmetic sequence with the given first four terms -/
noncomputable def ArithmeticSequence (x y : ℝ) : ℕ → ℝ
  | 0 => x^2 + y^2
  | 1 => x^2 - y^2
  | 2 => x^2 * y
  | 3 => x^2 / y
  | n + 4 => ArithmeticSequence x y (n + 3) - ArithmeticSequence x y (n + 2)

/-- Theorem stating that the fifth term of the sequence is -3/16 -/
theorem fifth_term_is_negative_three_sixteenths (x y : ℝ) (h1 : y ≠ 0) (h2 : y ≠ 1) :
  ArithmeticSequence x y 4 = -3/16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_term_is_negative_three_sixteenths_l57_5724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_house_cleaning_time_l57_5760

/-- The time it takes for Bruce and Anne to clean the house together -/
noncomputable def clean_time : ℝ := 4

/-- Anne's cleaning rate (fraction of house per hour) -/
noncomputable def anne_rate : ℝ := 1 / 12

/-- Combined cleaning rate when Anne's speed is doubled (fraction of house per hour) -/
noncomputable def combined_rate_double : ℝ := 1 / 3

theorem house_cleaning_time : 
  (1 / clean_time) = (1 / (1 / anne_rate)) + anne_rate := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_house_cleaning_time_l57_5760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l57_5725

open Set

def solution_set (a : ℝ) : Set ℝ :=
  if a = 0 then Iic (-1)
  else if a > 0 then Iic (-1) ∪ Ioi (1 / a)
  else if a = -1 then ∅
  else if a < -1 then Ioo (-1) (1 / a)
  else Ioo (1 / a) (-1)

theorem inequality_solution (a : ℝ) (x : ℝ) :
  (a * x - 1) / (x + 1) > 0 ↔ x ∈ solution_set a := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l57_5725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_binomial_coefficient_in_expansion_l57_5705

theorem largest_binomial_coefficient_in_expansion (x y : ℝ) :
  let n : ℕ := 11
  let expansion := (x - y) ^ n
  let sixth_term := (n.choose 5) * x^6 * (-y)^5
  let seventh_term := (n.choose 6) * x^5 * (-y)^6
  ∀ k, 0 ≤ k ∧ k ≤ n →
    (n.choose k) ≤ (n.choose 5) ∧ (n.choose k) ≤ (n.choose 6) :=
by
  intro k hk
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_binomial_coefficient_in_expansion_l57_5705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_indefinite_integral_proof_l57_5787

theorem indefinite_integral_proof (x : ℝ) :
  (deriv (fun x => -1/5 * (4*x + 3) * Real.cos (5*x) + 4/25 * Real.sin (5*x))) x
  = (4*x + 3) * Real.sin (5*x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_indefinite_integral_proof_l57_5787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_revolution_approx_l57_5719

/-- The number of revolutions a wheel makes given its radius and distance covered -/
noncomputable def wheel_revolutions (radius : ℝ) (distance : ℝ) : ℝ :=
  distance / (2 * Real.pi * radius)

/-- Theorem stating that a wheel with radius 22.4 cm covering 703.9999999999999 cm makes approximately 5 revolutions -/
theorem wheel_revolution_approx :
  let radius : ℝ := 22.4
  let distance : ℝ := 703.9999999999999
  abs (wheel_revolutions radius distance - 5) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_revolution_approx_l57_5719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_lambda_range_l57_5781

noncomputable def ellipse_equation (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

theorem ellipse_and_lambda_range :
  ∀ (a b : ℝ) (l : Set (ℝ × ℝ)) (C : Set (ℝ × ℝ)) (D M N : ℝ × ℝ) (lambda : ℝ),
    a > 0 →
    b > 0 →
    (∀ (x y : ℝ), (x, y) ∈ l ↔ y + 2 * Real.sqrt 3 = Real.sqrt 3 * x) →
    (0, -2 * Real.sqrt 3) ∈ l →
    (∀ (x y : ℝ), (x, y) ∈ C ↔ ellipse_equation a b x y) →
    (Real.sqrt 3, 1) ∈ C →
    D = (3, 0) →
    M ∈ C →
    N ∈ C →
    M ≠ N →
    (M.1 - D.1, M.2 - D.2) = lambda • (N.1 - D.1, N.2 - D.2) →
    (ellipse_equation 6 2 = ellipse_equation a b ∧
     5 - 2 * Real.sqrt 6 ≤ lambda ∧ lambda ≤ 5 + 2 * Real.sqrt 6 ∧ lambda ≠ 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_lambda_range_l57_5781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_sqrt10_l57_5702

/-- The eccentricity of a hyperbola with equation x²/a² - y²/b² = 1 and asymptotes y = ±kx -/
noncomputable def hyperbola_eccentricity (a b k : ℝ) : ℝ :=
  Real.sqrt (1 + k^2)

/-- Theorem: The eccentricity of the hyperbola x²/a² - y²/b² = 1 with asymptotes y = ±3x is √10 -/
theorem hyperbola_eccentricity_sqrt10 (a b : ℝ) :
  hyperbola_eccentricity a b 3 = Real.sqrt 10 := by
  unfold hyperbola_eccentricity
  -- The proof steps would go here, but we'll use sorry for now
  sorry

#check hyperbola_eccentricity_sqrt10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_sqrt10_l57_5702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_for_small_statues_l57_5720

/-- Amount of paint needed for a statue -/
noncomputable def paint_needed (height : ℝ) : ℝ := 2 * (height / 8) ^ 2

/-- The problem statement -/
theorem paint_for_small_statues :
  let original_height : ℝ := 8
  let small_height : ℝ := 2
  let num_statues : ℕ := 800
  paint_needed small_height * num_statues = 100 := by
  -- Unfold the definition of paint_needed
  unfold paint_needed
  -- Simplify the expression
  simp [pow_two]
  -- Perform the calculation
  norm_num
  -- QED
  done


end NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_for_small_statues_l57_5720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lemon_drink_increase_l57_5755

/-- Represents the percentage of lemon juice in the drink -/
def LemonJuicePercentage := ℝ

/-- Represents the quantity of lemon drink produced -/
def DrinkQuantity := ℝ

/-- 
Given:
- initial_percentage: The initial percentage of lemon juice in the drink
- final_percentage: The final percentage of lemon juice in the drink
- initial_quantity: The initial quantity of lemon drink produced

Proves that the percentage increase in drink quantity is 50% when the lemon juice
percentage is reduced from initial_percentage to final_percentage, assuming the
total amount of lemon juice remains constant.
-/
theorem lemon_drink_increase 
  (initial_percentage final_percentage : ℝ)
  (initial_quantity : ℝ)
  (h1 : initial_percentage = 15)
  (h2 : final_percentage = 10)
  (h3 : initial_percentage > 0)
  (h4 : final_percentage > 0) :
  let final_quantity := initial_quantity * initial_percentage / final_percentage
  (final_quantity - initial_quantity) / initial_quantity = 0.5 := by
  sorry

#check lemon_drink_increase

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lemon_drink_increase_l57_5755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_indefinite_integral_correct_l57_5731

open Real

-- Define the integrand function
noncomputable def f (x : ℝ) : ℝ :=
  (x^3 - 6*x^2 + 13*x - 7) / ((x+1)*(x-2)^3)

-- Define the antiderivative function
noncomputable def F (x : ℝ) : ℝ :=
  log (abs (x + 1)) - 1 / (2 * (x - 2)^2)

-- State the theorem
theorem indefinite_integral_correct :
  ∀ x : ℝ, x ≠ -1 ∧ x ≠ 2 → deriv F x = f x :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_indefinite_integral_correct_l57_5731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_quarter_ln_2_l57_5763

/-- The series term for the nth position -/
noncomputable def series_term (n : ℕ) : ℝ := 1 / ((4 * n - 3) * (4 * n - 2) * (4 * n - 1))

/-- The series sum -/
noncomputable def series_sum : ℝ := ∑' n, series_term n

/-- Theorem stating that the series sum equals (1/4) ln 2 -/
theorem series_sum_equals_quarter_ln_2 : series_sum = (1/4) * Real.log 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_quarter_ln_2_l57_5763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_intersection_distance_l57_5773

/-- Line l parameterized by t and α -/
noncomputable def line_l (α : ℝ) (t : ℝ) : ℝ × ℝ :=
  (t * Real.cos α, 2 + t * Real.sin α)

/-- Curve C in polar coordinates -/
def curve_C (θ : ℝ) (ρ : ℝ) : Prop :=
  ρ * (Real.cos θ)^2 = 4 * Real.sin θ

/-- Curve C in rectangular coordinates -/
def curve_C_rect (x y : ℝ) : Prop :=
  x^2 = 4 * y

/-- The intersection points of line l and curve C -/
def intersection (α : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ t, line_l α t = p ∧ curve_C_rect p.1 p.2}

/-- The distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The theorem to be proved -/
theorem min_intersection_distance :
    ∀ α, 0 ≤ α → α < π →
    (∃ A B, A ∈ intersection α ∧ B ∈ intersection α ∧ A ≠ B ∧
      ∀ p q, p ∈ intersection α → q ∈ intersection α → distance A B ≤ distance p q) →
    (∃ α₀, 0 ≤ α₀ ∧ α₀ < π ∧
      ∀ α, 0 ≤ α → α < π →
        ∃ A B, A ∈ intersection α ∧ B ∈ intersection α ∧
          distance A B ≥ 4 * Real.sqrt 2 ∧
          (∃ A₀ B₀, A₀ ∈ intersection α₀ ∧ B₀ ∈ intersection α₀ ∧ distance A₀ B₀ = 4 * Real.sqrt 2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_intersection_distance_l57_5773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_trajectory_is_straight_line_l57_5732

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube in 3D space -/
structure Cube where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D
  A1 : Point3D
  B1 : Point3D
  C1 : Point3D
  D1 : Point3D

/-- Represents a line in 3D space -/
structure Line3D where
  point : Point3D
  direction : Point3D

/-- Distance between two points in 3D space -/
noncomputable def distance (p1 p2 : Point3D) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2)

/-- Distance from a point to a line in 3D space -/
noncomputable def distanceToLine (p : Point3D) (l : Line3D) : ℝ :=
  sorry -- Definition of distance from point to line

/-- Predicate to check if a point is on a line -/
def onLine (p : Point3D) (l : Line3D) : Prop :=
  sorry -- Definition of a point being on a line

/-- Theorem: If a point P on the base of a cube has equal distance to A₁ and line CC₁, 
    then P lies on a straight line -/
theorem point_trajectory_is_straight_line (cube : Cube) :
  ∀ P : Point3D, 
    (P.z = cube.A.z) → -- P is on the base ABCD
    (distance P cube.A1 = distanceToLine P (Line3D.mk cube.C (Point3D.mk 0 0 1))) →
    ∃ l : Line3D, onLine P l := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_trajectory_is_straight_line_l57_5732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_formula_l57_5726

def sequence_a : ℕ → ℚ
  | 0 => 1  -- Add a case for 0 to avoid missing case error
  | 1 => 1
  | n + 1 => sequence_a n / (1 + sequence_a n)

theorem sequence_a_formula : ∀ n : ℕ, n ≥ 1 → sequence_a n = 1 / n := by
  intro n hn
  sorry  -- Placeholder for the actual proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_formula_l57_5726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_PQR_l57_5754

-- Define the vertices of the triangle
def P : ℚ × ℚ := (-3, 5)
def Q : ℚ × ℚ := (4, -2)
def R : ℚ × ℚ := (9, 2)

-- Define the function to calculate the area of a triangle given its vertices
def triangleArea (p q r : ℚ × ℚ) : ℚ :=
  let u := (p.1 - r.1, p.2 - r.2)
  let v := (q.1 - r.1, q.2 - r.2)
  (1/2) * abs (u.1 * v.2 - u.2 * v.1)

-- Theorem stating that the area of triangle PQR is 31.5
theorem area_of_triangle_PQR : triangleArea P Q R = 63/2 := by
  -- Unfold the definition of triangleArea
  unfold triangleArea
  -- Simplify the expression
  simp
  -- The proof is completed
  rfl

#eval triangleArea P Q R

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_PQR_l57_5754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_defective_percentage_approximation_l57_5774

noncomputable def total_meters : ℝ := 3333.3333333333335
def defective_meters : ℕ := 2

noncomputable def percentage_defective : ℝ := (defective_meters : ℝ) / total_meters * 100

theorem defective_percentage_approximation :
  ∃ ε > 0, |percentage_defective - 0.06000600060006| < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_defective_percentage_approximation_l57_5774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hat_promotion_savings_l57_5745

/-- Represents the promotion details for hat purchases -/
structure HatPromotion where
  regularPrice : ℚ
  secondHatDiscount : ℚ
  thirdHatDiscount : ℚ

/-- Calculates the total cost of three hats under the promotion -/
def promotionCost (p : HatPromotion) : ℚ :=
  p.regularPrice + (1 - p.secondHatDiscount) * p.regularPrice + (1 - p.thirdHatDiscount) * p.regularPrice

/-- Calculates the percentage saved under the promotion -/
def percentageSaved (p : HatPromotion) : ℚ :=
  (3 * p.regularPrice - promotionCost p) / (3 * p.regularPrice) * 100

/-- Theorem stating that the percentage saved is 30% -/
theorem hat_promotion_savings :
  let p : HatPromotion := ⟨40, 3/10, 6/10⟩
  percentageSaved p = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hat_promotion_savings_l57_5745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_side_length_l57_5792

/-- The quadrilateral region defined by the system of inequalities -/
def QuadrilateralRegion : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 + p.2 ≤ 4 ∧ 2 * p.1 + p.2 ≥ 1 ∧ p.1 ≥ 0 ∧ p.2 ≥ 0}

/-- The length of a side given by two points -/
noncomputable def SideLength (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- The theorem stating that the longest side of the quadrilateral has length 7√2/2 -/
theorem longest_side_length :
  ∃ (p1 p2 : ℝ × ℝ), p1 ∈ QuadrilateralRegion ∧ p2 ∈ QuadrilateralRegion ∧
    SideLength p1 p2 = 7 * Real.sqrt 2 / 2 ∧
    ∀ (q1 q2 : ℝ × ℝ), q1 ∈ QuadrilateralRegion → q2 ∈ QuadrilateralRegion →
      SideLength q1 q2 ≤ 7 * Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_side_length_l57_5792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carlos_trip_equation_correct_l57_5734

/-- Represents a driving trip with a stop -/
structure DrivingTrip where
  speed_before_stop : ℚ
  speed_after_stop : ℚ
  stop_duration : ℚ
  total_distance : ℚ
  total_time : ℚ

/-- The equation for the driving trip is correct -/
def correct_equation (trip : DrivingTrip) (t : ℚ) : Prop :=
  trip.speed_before_stop * t + trip.speed_after_stop * (trip.total_time - trip.stop_duration - t) = trip.total_distance

/-- Carlos's driving trip -/
def carlos_trip : DrivingTrip :=
  { speed_before_stop := 60
  , speed_after_stop := 80
  , stop_duration := 1/2
  , total_distance := 180
  , total_time := 4 }

theorem carlos_trip_equation_correct :
  ∃ t : ℚ, correct_equation carlos_trip t := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_carlos_trip_equation_correct_l57_5734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_southeast_northwest_angle_l57_5797

/-- The number of chairs around the circular table -/
def num_chairs : ℕ := 12

/-- The angle between two adjacent chairs in degrees -/
noncomputable def angle_between_chairs : ℝ := 360 / num_chairs

/-- The position of the southeast-facing chair (number of chairs clockwise from north) -/
def southeast_position : ℕ := 4

/-- The position of the northwest-facing chair (number of chairs clockwise from north) -/
def northwest_position : ℕ := 10

/-- The smaller angle between the southeast-facing and northwest-facing chairs -/
noncomputable def smaller_angle : ℝ := angle_between_chairs * (northwest_position - southeast_position)

theorem southeast_northwest_angle :
  smaller_angle = 180 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_southeast_northwest_angle_l57_5797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_ratio_one_half_l57_5718

/-- A rectangle with given length and perimeter -/
structure Rectangle where
  length : ℚ
  perimeter : ℚ

/-- The width of a rectangle -/
def Rectangle.width (r : Rectangle) : ℚ :=
  (r.perimeter - 2 * r.length) / 2

/-- The ratio of width to length of a rectangle -/
def Rectangle.widthLengthRatio (r : Rectangle) : ℚ :=
  r.width / r.length

/-- Theorem: For a rectangle with length 8 and perimeter 24, the ratio of width to length is 1:2 -/
theorem rectangle_ratio_one_half :
  let r := Rectangle.mk 8 24
  r.widthLengthRatio = 1 / 2 := by
  -- Proof goes here
  sorry

#eval (Rectangle.mk 8 24).widthLengthRatio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_ratio_one_half_l57_5718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_G_is_perfect_square_iff_m_equals_25_8_l57_5786

noncomputable def G (x m : ℝ) : ℝ := (8*x^2 + 20*x + 4*m) / 8

def perfectSquare (a b c : ℝ) : Prop := ∃ (k l : ℝ), ∀ x : ℝ, a*x^2 + b*x + c = (k*x + l)^2

theorem G_is_perfect_square_iff_m_equals_25_8 :
  ∀ m : ℝ, (∀ x : ℝ, perfectSquare 1 (5/2) (m/2)) ↔ m = 25/8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_G_is_perfect_square_iff_m_equals_25_8_l57_5786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_equation_max_value_g_max_x3_l57_5730

noncomputable def f (x : ℝ) := Real.exp (x * Real.log 2)

-- Part I
theorem solve_equation : ∃ x : ℝ, f (2*x) - f (x+1) = 8 ∧ x = 2 := by sorry

-- Part II
noncomputable def g (a x : ℝ) := f x + a * (f x)^2

noncomputable def M (a : ℝ) : ℝ :=
  if a > -1/4 then 4*a + 2
  else if a < -1/2 then a + 1
  else -1/(4*a)

theorem max_value_g : ∀ a : ℝ, ∀ x ∈ Set.Icc 0 1, g a x ≤ M a := by sorry

-- Part III
theorem max_x3 : 
  ∀ x1 x2 x3 : ℝ, 
    (f x1 + f x2 = f (x1 + x2)) → 
    (f x1 + f x2 + f x3 = f (x1 + x2 + x3)) →
    x3 ≤ 2 - (Real.log 3) / (Real.log 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_equation_max_value_g_max_x3_l57_5730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_harmonic_mean_of_2_3_6_5_l57_5752

noncomputable def harmonic_mean (x y z : ℝ) : ℝ := 3 / (1/x + 1/y + 1/z)

theorem harmonic_mean_of_2_3_6_5 :
  harmonic_mean 2 3 (13/2) = 234/77 := by
  -- Expand the definition of harmonic_mean
  unfold harmonic_mean
  -- Simplify the fraction
  simp [div_eq_mul_inv]
  -- Perform algebraic manipulations
  ring
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_harmonic_mean_of_2_3_6_5_l57_5752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotten_oranges_percentage_l57_5706

theorem rotten_oranges_percentage
  (total_oranges : ℕ)
  (total_bananas : ℕ)
  (rotten_bananas_percentage : ℚ)
  (good_fruits_percentage : ℚ)
  (h1 : total_oranges = 600)
  (h2 : total_bananas = 400)
  (h3 : rotten_bananas_percentage = 3 / 100)
  (h4 : good_fruits_percentage = 898 / 1000) :
  (total_oranges - (good_fruits_percentage * (total_oranges + total_bananas) -
    total_bananas + rotten_bananas_percentage * total_bananas).floor) /
    total_oranges = 15 / 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotten_oranges_percentage_l57_5706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_earth_fall_time_l57_5733

/-- The gravitational constant -/
noncomputable def γ : ℝ := sorry

/-- Mass of the Sun -/
noncomputable def M : ℝ := sorry

/-- Original distance between Earth and Sun -/
noncomputable def R : ℝ := sorry

/-- Time for Earth to fall into the Sun -/
noncomputable def fall_time : ℝ := (Real.pi / 2) * Real.sqrt (R^3 / (2 * γ * M))

/-- Theorem stating the time it takes for Earth to fall into the Sun -/
theorem earth_fall_time :
  ∀ (stop_orbiting : Prop) (fall_to_sun : Prop),
  stop_orbiting → fall_to_sun →
  ∃ (t : ℝ), t = fall_time := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_earth_fall_time_l57_5733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_l57_5743

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.log (4 + 3*x - x^2)

-- State the theorem
theorem monotonic_decreasing_interval :
  ∀ x ∈ Set.Icc (3/2 : ℝ) 4,
    ∀ y ∈ Set.Icc (3/2 : ℝ) 4,
      x ≤ y → f x ≥ f y :=
by
  -- Proof placeholder
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_l57_5743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_quadrilateral_area_l57_5737

/-- Regular octagon with apothem 3 -/
structure RegularOctagon where
  apothem : ℝ
  apothem_eq : apothem = 3

/-- Point on the octagon -/
def OctagonPoint (P : RegularOctagon) := ℝ × ℝ

/-- Midpoint of a side of the octagon -/
def midpoint_of_side (P : RegularOctagon) (i : Fin 8) : OctagonPoint P := sorry

/-- Quadrilateral formed by four consecutive midpoints -/
def midpoint_quadrilateral (P : RegularOctagon) : Type :=
  { Q : Fin 4 → OctagonPoint P // ∀ (i : Fin 4), ∃ (j : Fin 8), Q i = midpoint_of_side P j }

/-- Area of a quadrilateral -/
noncomputable def area (P : RegularOctagon) (Q : midpoint_quadrilateral P) : ℝ := sorry

/-- Theorem: Area of the midpoint quadrilateral in a regular octagon with apothem 3 -/
theorem midpoint_quadrilateral_area (P : RegularOctagon) (Q : midpoint_quadrilateral P) :
  area P Q = 9 * (3 - 2 * Real.sqrt 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_quadrilateral_area_l57_5737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_power_six_l57_5779

noncomputable def z : ℂ := (Real.sqrt 3 - Complex.I) / 2

theorem z_power_six : z^6 = -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_power_six_l57_5779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_money_proof_l57_5767

def beth_money : ℚ := 100
def jan_money : ℚ := 120
def tom_money : ℚ := 315

theorem total_money_proof :
  beth_money + 50 = 150 ∧
  jan_money - 20 = beth_money ∧
  tom_money = 3.5 * (jan_money - 30) ∧
  beth_money + jan_money + tom_money = 535 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_money_proof_l57_5767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pet_owners_calculation_l57_5757

theorem pet_owners_calculation :
  let total_pet_owners : ℕ := 89
  let only_dog_owners : ℕ := 15
  let only_cat_owners : ℕ := 10
  let cat_dog_snake_owners : ℕ := 3
  let total_snakes : ℕ := 59
  let only_snake_owners : ℕ := total_snakes - cat_dog_snake_owners
  let only_cat_and_dog_owners : ℕ := total_pet_owners - (only_dog_owners + only_cat_owners + cat_dog_snake_owners + only_snake_owners)
  only_cat_and_dog_owners = 5 := by
  sorry

#check pet_owners_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pet_owners_calculation_l57_5757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_l57_5722

noncomputable def f (x : ℝ) : ℝ := (1/2)^x

theorem f_monotone_decreasing :
  ∀ x₁ x₂ : ℝ, 0 < x₁ → 0 < x₂ → x₁ < x₂ → f x₂ < f x₁ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_l57_5722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_right_triangle_vertex_l57_5777

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Represents a right angle -/
structure RightAngle where
  vertex : Point
  side1 : Point
  side2 : Point

/-- Checks if a triangle is right-angled at vertex A -/
def isRightTriangle (t : Triangle) : Prop :=
  (t.B.x - t.A.x) * (t.C.x - t.A.x) + (t.B.y - t.A.y) * (t.C.y - t.A.y) = 0

/-- Checks if a point is on a line segment -/
def isOnSegment (p : Point) (a : Point) (b : Point) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ p.x = a.x + t * (b.x - a.x) ∧ p.y = a.y + t * (b.y - a.y)

/-- Calculates the distance between two points -/
noncomputable def distance (a : Point) (b : Point) : ℝ :=
  Real.sqrt ((a.x - b.x)^2 + (a.y - b.y)^2)

/-- The main theorem -/
theorem locus_of_right_triangle_vertex (ABC : Triangle) (P : RightAngle) :
  isRightTriangle ABC →
  (∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 →
    ∃ B' C', isOnSegment B' P.vertex P.side1 ∧ isOnSegment C' P.vertex P.side2 ∧
    isRightTriangle { A := ABC.A, B := B', C := C' }) →
  ∃ A₁ A₂ : Point,
    (∀ A' : Point, (∃ B' C', isOnSegment B' P.vertex P.side1 ∧ isOnSegment C' P.vertex P.side2 ∧
      isRightTriangle { A := A', B := B', C := C' }) ↔ isOnSegment A' A₁ A₂) ∧
    distance A₁ A₂ = distance ABC.B ABC.C - distance ABC.A ABC.B :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_right_triangle_vertex_l57_5777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_base_length_l57_5739

/-- Given an equilateral triangle with perimeter 60 and an isosceles triangle with perimeter 65,
    where one side of the equilateral triangle is equal to one of the equal sides of the isosceles triangle,
    the base of the isosceles triangle is 25 units long. -/
theorem isosceles_triangle_base_length 
  (equilateral_perimeter : ℝ) 
  (isosceles_perimeter : ℝ) 
  (h1 : equilateral_perimeter = 60) 
  (h2 : isosceles_perimeter = 65) 
  (h3 : equilateral_perimeter / 3 = isosceles_side) 
  (isosceles_side : ℝ) : 
  isosceles_perimeter - 2 * isosceles_side = 25 :=
by
  sorry

#check isosceles_triangle_base_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_base_length_l57_5739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_shift_equivalence_l57_5744

-- Define the function f as noncomputable
noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + φ)

-- State the theorem
theorem function_shift_equivalence 
  (ω φ : ℝ) 
  (h1 : ω > 0) 
  (h2 : 0 < φ ∧ φ < Real.pi) 
  (h3 : ∀ x, f ω φ (2 * Real.pi / 3 - x) = f ω φ (Real.pi / 3 + x)) 
  (h4 : Real.pi / 4 = Real.pi / (4 * ω)) :
  ∀ x, f ω φ (x + Real.pi / 12) = Real.cos (2 * x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_shift_equivalence_l57_5744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_sample_not_23_l57_5701

def systematic_sample : ℕ → ℕ → ℕ
  | _, 0 => 0  -- Handle the case when n = 0
  | first_sample, n => first_sample + 10 * (n - 1)

theorem second_sample_not_23 (first_sample : ℕ) (h : 1 ≤ first_sample ∧ first_sample ≤ 10) :
  systematic_sample first_sample 2 ≠ 23 := by
  unfold systematic_sample
  simp
  linarith

#eval systematic_sample 1 2  -- Example evaluation
#eval systematic_sample 10 2  -- Example evaluation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_sample_not_23_l57_5701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_integer_values_l57_5740

theorem polynomial_integer_values (a b c d : ℚ) : 
  (∀ x ∈ ({-1, 0, 1, 2} : Set ℤ), ∃ n : ℤ, a * x^3 + b * x^2 + c * x + d = n) →
  (∀ x : ℤ, ∃ n : ℤ, a * x^3 + b * x^2 + c * x + d = n) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_integer_values_l57_5740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_joe_fraction_l57_5761

theorem joe_fraction (initial_crayons : ℕ) (kiley_fraction : ℚ) (remaining_crayons : ℕ) : 
  initial_crayons = 48 →
  kiley_fraction = 1/4 →
  remaining_crayons = 18 →
  let crayons_after_kiley := initial_crayons - (kiley_fraction * ↑initial_crayons).floor
  (crayons_after_kiley - remaining_crayons) / crayons_after_kiley = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_joe_fraction_l57_5761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_PQR_eq_area_JKL_l57_5764

-- Define the triangle JKL and its area
variable (S : ℝ)
variable (triangle_JKL : Fin 3 → ℝ × ℝ)
variable (area_JKL : ℝ)
axiom area_JKL_eq_S : area_JKL = S

-- Define point M as the midpoint of KL
variable (M : ℝ × ℝ)
axiom M_is_midpoint_KL : M = (triangle_JKL 1 + triangle_JKL 2) / 2

-- Define points P, Q, R on extended lines
variable (P Q R : ℝ × ℝ)

-- Define the ratios for P, Q, R
axiom JP_ratio : ‖P - triangle_JKL 0‖ = 2 * ‖triangle_JKL 1 - triangle_JKL 0‖
axiom JQ_ratio : ‖Q - triangle_JKL 0‖ = 3 * ‖M - triangle_JKL 0‖
axiom JR_ratio : ‖R - triangle_JKL 0‖ = 4 * ‖triangle_JKL 2 - triangle_JKL 0‖

-- Define triangle PQR and its area
def triangle_PQR : Fin 3 → ℝ × ℝ := fun i => match i with
  | 0 => P
  | 1 => Q
  | 2 => R

noncomputable def area_PQR : ℝ := sorry

-- Theorem to prove
theorem area_PQR_eq_area_JKL : area_PQR = area_JKL := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_PQR_eq_area_JKL_l57_5764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l57_5728

noncomputable def f (x : Real) : Real := Real.sin (2 * x - Real.pi / 6)

theorem f_properties :
  (∃ T : Real, T > 0 ∧ T = Real.pi ∧ ∀ x, f (x + T) = f x) ∧
  (∀ x, f (2 * Real.pi / 3 - x) = f (2 * Real.pi / 3 + x)) ∧
  (∀ x y, -Real.pi / 6 ≤ x ∧ x < y ∧ y ≤ Real.pi / 3 → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l57_5728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_f_negative_solution_set_l57_5721

noncomputable def f (x : ℝ) : ℝ := 2^x

theorem inverse_f_negative_solution_set 
  (f_inv : ℝ → ℝ) 
  (h_inv : ∀ x, f (f_inv x) = x ∧ f_inv (f x) = x) :
  {x : ℝ | f_inv x < 0} = Set.Ioo 0 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_f_negative_solution_set_l57_5721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l57_5704

/-- Given a geometric sequence {a_n}, if (a₂, 2) and (a₃, 3) are parallel vectors,
    then (a₂ + a₄) / (a₃ + a₅) = 2/3 -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) :
  (∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n) →  -- a_n is a geometric sequence
  (∃ k : ℝ, k * a 2 = a 3 ∧ k * 2 = 3) →     -- (a₂, 2) and (a₃, 3) are parallel
  (a 2 + a 4) / (a 3 + a 5) = 2 / 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l57_5704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_five_shots_correct_expected_shots_correct_l57_5714

/-- The probability of hitting a target with a single shot -/
def p : ℝ := sorry

/-- Assumption: p is a probability, so it's between 0 and 1 -/
axiom h : 0 < p ∧ p < 1

/-- The probability of hitting all three targets in exactly 5 shots -/
noncomputable def prob_five_shots : ℝ := 6 * p^3 * (1 - p)^2

/-- The expected number of shots needed to hit all three targets -/
noncomputable def expected_shots : ℝ := 3 / p

/-- Theorem: The probability of hitting all three targets in exactly 5 shots is 6p³(1-p)² -/
theorem prob_five_shots_correct : 
  prob_five_shots = 6 * p^3 * (1 - p)^2 := by sorry

/-- Theorem: The expected number of shots needed to hit all three targets is 3/p -/
theorem expected_shots_correct : 
  expected_shots = 3 / p := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_five_shots_correct_expected_shots_correct_l57_5714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_proof_l57_5735

theorem integral_proof (x : ℝ) (h : x ≠ 0 ∧ x + 2 ≠ 0) : 
  deriv (λ x => 2 * Real.log (|x|) + Real.log (|x + 2|) + 10 / (x + 2)) x = 
  (3 * x^2 + 8) / (x * (x + 2)^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_proof_l57_5735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_funnel_height_l57_5769

-- Define the radius and volume
def r : ℝ := 4
def V : ℝ := 150

-- Define the height function
noncomputable def h : ℝ := (3 * V) / (Real.pi * r^2)

-- Define the rounding function
noncomputable def round_to_nearest (x : ℝ) : ℤ :=
  ⌊x + 0.5⌋

-- Theorem statement
theorem funnel_height :
  round_to_nearest h = 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_funnel_height_l57_5769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plot_size_in_acres_l57_5753

/-- Represents the scale of the land plot -/
def scale : ℝ := 3

/-- Represents the number of acres in a square mile -/
def acres_per_square_mile : ℝ := 640

/-- Represents the bottom edge of the trapezoid in cm -/
def bottom_edge : ℝ := 15

/-- Represents the top edge of the trapezoid in cm -/
def top_edge : ℝ := 10

/-- Represents the height of the trapezoid in cm -/
def trapezoid_height : ℝ := 10

/-- Calculates the area of a trapezoid -/
noncomputable def trapezoid_area (a b h : ℝ) : ℝ := (a + b) * h / 2

/-- Theorem: The size of the actual plot in acres is 720000 -/
theorem plot_size_in_acres : 
  trapezoid_area bottom_edge top_edge trapezoid_height * scale^2 * acres_per_square_mile = 720000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_plot_size_in_acres_l57_5753
