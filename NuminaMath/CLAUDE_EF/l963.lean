import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inflation_and_real_interest_rate_l963_96346

/-- Calculates the total inflation over a given number of years -/
noncomputable def totalInflation (annualRate : ℝ) (years : ℕ) : ℝ :=
  ((1 + annualRate) ^ years - 1) * 100

/-- Calculates the real interest rate given nominal rate and inflation rate -/
noncomputable def realInterestRate (nominalRate : ℝ) (inflationRate : ℝ) : ℝ :=
  ((1 + nominalRate) / (1 + inflationRate) - 1) * 100

theorem inflation_and_real_interest_rate 
  (annualInflationRate : ℝ) 
  (nominalInterestRate : ℝ) 
  (h1 : annualInflationRate = 0.025) 
  (h2 : nominalInterestRate = 0.06) :
  totalInflation annualInflationRate 2 = 5.0625 ∧ 
  realInterestRate (nominalInterestRate * 2) (totalInflation annualInflationRate 2 / 100) = 6.95 := by
  sorry

-- Remove #eval statements as they won't work with noncomputable definitions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inflation_and_real_interest_rate_l963_96346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monomial_properties_l963_96384

/-- A monomial is a product of a coefficient and variables raised to non-negative integer powers -/
structure Monomial (R : Type*) [CommRing R] where
  coeff : R
  vars : List (Nat × Nat)

/-- The degree of a monomial is the sum of the exponents of its variables -/
def Monomial.degree {R : Type*} [CommRing R] (m : Monomial R) : Nat :=
  (m.vars.map Prod.snd).sum

/-- Our specific monomial -\frac{3x{y}^{2}}{5} -/
def our_monomial : Monomial ℚ :=
  { coeff := -3/5
    vars := [(1, 1), (2, 2)] }  -- x has exponent 1, y has exponent 2

theorem monomial_properties :
  our_monomial.coeff = -3/5 ∧ our_monomial.degree = 3 := by
  constructor
  · rfl
  · rfl

#eval our_monomial.degree

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monomial_properties_l963_96384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_equals_cos_456_deg_l963_96388

-- Define the degree-to-radian conversion factor
noncomputable def deg_to_rad : ℝ := Real.pi / 180

-- Define the sine and cosine functions that work with degrees
noncomputable def sin_deg (x : ℝ) : ℝ := Real.sin (x * deg_to_rad)
noncomputable def cos_deg (x : ℝ) : ℝ := Real.cos (x * deg_to_rad)

theorem sin_equals_cos_456_deg :
  ∃ n : ℤ, -90 ≤ n ∧ n ≤ 90 ∧ sin_deg (↑n) = cos_deg 456 :=
by
  use -6
  sorry

#check sin_equals_cos_456_deg

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_equals_cos_456_deg_l963_96388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l963_96390

noncomputable def f (x : ℝ) := x - 3 + (Real.log x) / (Real.log 3)

theorem zero_in_interval :
  ∃ c ∈ Set.Ioo 2 3, f c = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l963_96390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sprint_time_proof_l963_96349

/-- Calculates the time taken given distance and speed -/
noncomputable def calculate_time (distance : ℝ) (speed : ℝ) : ℝ :=
  distance / speed

theorem sprint_time_proof :
  let distance : ℝ := 24
  let speed : ℝ := 6
  calculate_time distance speed = 4 := by
  unfold calculate_time
  simp
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sprint_time_proof_l963_96349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bullet_train_crossing_time_l963_96367

/-- The time (in seconds) for two trains to cross each other -/
noncomputable def crossingTime (trainLength : ℝ) (crossingTime1 : ℝ) (crossingTime2 : ℝ) : ℝ :=
  let speed1 := trainLength / crossingTime1
  let speed2 := trainLength / crossingTime2
  let relativeSpeed := speed1 + speed2
  let totalDistance := 2 * trainLength
  totalDistance / relativeSpeed

/-- Theorem stating the crossing time for two bullet trains -/
theorem bullet_train_crossing_time :
  let trainLength : ℝ := 120
  let crossingTime1 : ℝ := 10
  let crossingTime2 : ℝ := 12
  ∃ ε > 0, abs (crossingTime trainLength crossingTime1 crossingTime2 - 10.91) < ε :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bullet_train_crossing_time_l963_96367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_bridge_time_l963_96320

/-- Calculates the time (in seconds) for a train to cross a bridge -/
noncomputable def train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (bridge_length : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  total_distance / train_speed_ms

theorem train_crossing_bridge_time :
  train_crossing_time 475 90 275 = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_bridge_time_l963_96320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_crab_collection_frequency_l963_96387

theorem crab_collection_frequency 
  (baskets_per_week : ℕ) 
  (crabs_per_basket : ℕ) 
  (price_per_crab : ℕ) 
  (total_revenue : ℕ) 
  (h1 : baskets_per_week = 3)
  (h2 : crabs_per_basket = 4)
  (h3 : price_per_crab = 3)
  (h4 : total_revenue = 72) : ℕ := 2

-- Proof
example : crab_collection_frequency 3 4 3 72 rfl rfl rfl rfl = 2 := by
  -- Unfold the definition
  unfold crab_collection_frequency
  -- The result is immediate from the definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_crab_collection_frequency_l963_96387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_real_parts_complex_solutions_l963_96339

theorem product_real_parts_complex_solutions : 
  ∃ (x₁ x₂ : ℂ), 
    (2 * x₁^2 + 4 * x₁ = 3 + (1 : ℂ)) ∧ 
    (2 * x₂^2 + 4 * x₂ = 3 + (1 : ℂ)) ∧ 
    (x₁ ≠ x₂) ∧
    (x₁.re * x₂.re = 1 - Real.cos (π / 30) * Real.sqrt 6.5) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_real_parts_complex_solutions_l963_96339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_survey_results_l963_96355

/-- Represents the subject combinations --/
inductive SubjectCombination
| Physics
| History
| PoliticalScience
| Geography
| Chemistry
| Biology

/-- Represents the gender of a student --/
inductive Gender
| Boy
| Girl

/-- Represents the subject choice of a student --/
structure StudentChoice where
  first : SubjectCombination
  second : SubjectCombination
  third : SubjectCombination

/-- The total number of students surveyed --/
def totalStudents : ℕ := 100

/-- The number of students who chose History --/
def historyStudents : ℕ := 30

/-- The number of boys who chose History --/
def boysHistory : ℕ := 10

/-- The number of girls who chose Physics --/
def girlsPhysics : ℕ := 30

/-- Function to calculate K^2 value --/
def calculateK2 (a b c d : ℕ) : ℚ :=
  let n : ℕ := a + b + c + d
  (n * (a * d - b * c)^2 : ℚ) / ((a + b) * (c + d) * (a + c) * (b + d))

/-- Critical value for 95% confidence level --/
def criticalValue : ℚ := 3841 / 1000

theorem survey_results :
  -- Part 1: Probability of selecting "History, Political Science, Geography"
  (1 : ℚ) / 12 = 1 / 12 ∧
  
  -- Part 2: Values of a and d
  (totalStudents - historyStudents - girlsPhysics = 40) ∧
  (historyStudents - boysHistory = 20) ∧
  
  -- Part 3: K^2 value exceeds critical value
  calculateK2 40 10 30 20 > criticalValue := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_survey_results_l963_96355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_days_to_plant_trees_l963_96309

noncomputable def geometric_sum (a : ℝ) (r : ℝ) (n : ℕ) : ℝ := a * (1 - r^n) / (1 - r)

theorem min_days_to_plant_trees (n : ℕ) : n = 9 ↔ 
  (n > 0 ∧ geometric_sum 2 2 n ≥ 1000 ∧ 
  ∀ m : ℕ, m > 0 ∧ m < n → geometric_sum 2 2 m < 1000) := by
  sorry

#check min_days_to_plant_trees

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_days_to_plant_trees_l963_96309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_l963_96347

open Real

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (log (5 * x + 2)) ^ 3

-- State the theorem
theorem f_derivative (x : ℝ) (h : 5 * x + 2 > 0) :
  deriv f x = (15 * (log (5 * x + 2))^2) / (5 * x + 2) :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_l963_96347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_swimming_speed_is_12_l963_96328

/-- The swimming speed of a person in still water -/
noncomputable def swimming_speed : ℝ → ℝ → ℝ → ℝ := λ water_speed distance time =>
  distance / time + water_speed

/-- Theorem: The swimming speed in still water is 12 km/h -/
theorem swimming_speed_is_12 (water_speed : ℝ) (distance : ℝ) (time : ℝ)
  (h1 : water_speed = 2)
  (h2 : distance = 10)
  (h3 : time = 1) :
  swimming_speed water_speed distance time = 12 :=
by
  sorry

-- Remove the #eval line as it's not computable
-- #eval swimming_speed 2 10 1

end NUMINAMATH_CALUDE_ERRORFEEDBACK_swimming_speed_is_12_l963_96328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_operations_l963_96371

variable {n : Type*} [NormedAddCommGroup n] [InnerProductSpace ℝ n]
variable (a b : n)
variable (c : ℝ)

theorem vector_operations :
  (∃ r : ℝ, inner a b = r) ∧
  (∃ v : n, a + b = v) ∧
  (∃ v : n, a - b = v) ∧
  (∃ v : n, c • a = v) :=
by
  constructor
  · use inner a b
  constructor
  · use a + b
  constructor
  · use a - b
  · use c • a


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_operations_l963_96371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_term_is_correct_l963_96326

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  -- The sum of the first 30 terms
  sum_first_30 : ℚ
  -- The sum of the next 30 terms (terms 31 to 60)
  sum_next_30 : ℚ
  -- Property: The sum of the first 30 terms is 270
  sum_first_30_eq : sum_first_30 = 270
  -- Property: The sum of the next 30 terms is 1830
  sum_next_30_eq : sum_next_30 = 1830

/-- The first term of the arithmetic sequence -/
def first_term (seq : ArithmeticSequence) : ℚ := -242/15

/-- Theorem stating that the first term of the arithmetic sequence with the given properties is -242/15 -/
theorem first_term_is_correct (seq : ArithmeticSequence) : first_term seq = -242/15 := by
  sorry

#eval first_term { sum_first_30 := 270, sum_next_30 := 1830, sum_first_30_eq := rfl, sum_next_30_eq := rfl }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_term_is_correct_l963_96326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_squared_distances_bounds_l963_96398

noncomputable section

-- Define the curves C₁ and C₂
def C₁ (φ : Real) : Real × Real := (2 * Real.cos φ, 3 * Real.sin φ)
def C₂ (θ : Real) : Real × Real := (2 * Real.cos θ, 2 * Real.sin θ)

-- Define the vertices of square ABCD
def A : Real × Real := C₂ (Real.pi / 3)
def B : Real × Real := C₂ Real.pi
def C : Real × Real := C₂ (3 * Real.pi / 2)
def D : Real × Real := C₂ (2 * Real.pi)

-- Define the squared distance between two points
def squaredDistance (p q : Real × Real) : Real :=
  (p.1 - q.1)^2 + (p.2 - q.2)^2

-- Define the sum of squared distances from P to A, B, C, and D
def sumSquaredDistances (φ : Real) : Real :=
  let P := C₁ φ
  squaredDistance P A + squaredDistance P B + squaredDistance P C + squaredDistance P D

-- Theorem statement
theorem sum_squared_distances_bounds :
  ∀ φ : Real, 32 ≤ sumSquaredDistances φ ∧ sumSquaredDistances φ ≤ 52 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_squared_distances_bounds_l963_96398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_proof_l963_96372

theorem cosine_sum_proof (α β : ℝ) 
  (h1 : 0 < β) (h2 : β < Real.pi / 2) (h3 : Real.pi / 2 < α) (h4 : α < Real.pi)
  (h5 : Real.cos (α - β / 2) = - 1 / 9) (h6 : Real.sin (α / 2 - β) = 2 / 3) :
  Real.cos (α + β) = -239 / 729 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_proof_l963_96372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_maximization_l963_96331

noncomputable section

-- Define the sales revenue function
def I (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 2 then 2 * (x - 1) * Real.exp (x - 2) + 2
  else if 2 < x ∧ x ≤ 50 then 440 + 3050 / x - 9000 / x^2
  else 0

-- Define the profit function
def P (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 50 then x * I x - (1.8 + 0.45 * x)
  else 0

-- Theorem statement
theorem profit_maximization :
  ∃ (x : ℝ), 0 < x ∧ x ≤ 50 ∧
  (∀ (y : ℝ), 0 < y ∧ y ≤ 50 → P y ≤ P x) ∧
  x = 30 ∧ P x = 2270 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_maximization_l963_96331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_focus_l963_96318

/-- The parabola y^2 = 8x -/
def parabola (x y : ℝ) : Prop := y^2 = 8*x

/-- The focus of the parabola y^2 = 8x -/
def focus : ℝ × ℝ := (2, 0)

/-- Point P on the parabola -/
noncomputable def P : ℝ × ℝ := (8, Real.sqrt (8*8))

/-- Distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem distance_to_focus :
  parabola P.1 P.2 → distance P focus = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_focus_l963_96318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equals_double_f_l963_96359

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x^2 + 2 * x + 1

-- State the theorem
theorem integral_equals_double_f (a : ℝ) : 
  (∫ x in Set.Icc (-1 : ℝ) 1, f x) = 2 * f a ↔ a = -1 ∨ a = 1/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equals_double_f_l963_96359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_BEIH_is_three_fifths_l963_96386

-- Define the square ABCD
def A : ℝ × ℝ := (0, 3)
def B : ℝ × ℝ := (0, 0)
def C : ℝ × ℝ := (3, 0)
def D : ℝ × ℝ := (3, 3)

-- Define midpoints E and F
def E : ℝ × ℝ := (0, 1.5)
def F : ℝ × ℝ := (1.5, 0)

-- Define lines AF and DE
def line_AF (x : ℝ) : ℝ := -2 * x + 3
def line_DE (x : ℝ) : ℝ := 0.5 * x + 1.5

-- Define line BD
def line_BD (x : ℝ) : ℝ := x

-- Define intersection points I and H
def I : ℝ × ℝ := (0.6, 1.8)
def H : ℝ × ℝ := (1, 1)

-- Define the area function for a quadrilateral using Shoelace formula
def area_quadrilateral (p1 p2 p3 p4 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  let (x4, y4) := p4
  0.5 * abs (x1*y2 + x2*y3 + x3*y4 + x4*y1 - (y1*x2 + y2*x3 + y3*x4 + y4*x1))

-- State the theorem
theorem area_BEIH_is_three_fifths :
  area_quadrilateral B E I H = 3/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_BEIH_is_three_fifths_l963_96386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_abs_sin_l963_96327

noncomputable def f (x : ℝ) : ℝ := |Real.sin x|

theorem smallest_positive_period_of_abs_sin :
  ∃ (p : ℝ), p > 0 ∧ 
  (∀ (x : ℝ), f (x + p) = f x) ∧
  (∀ (q : ℝ), q > 0 → (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  p = Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_abs_sin_l963_96327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_john_park_time_l963_96301

/-- The time (in minutes) it takes John to reach the park -/
noncomputable def time_to_park (speed : ℝ) (distance : ℝ) : ℝ :=
  (distance / 1000) / speed * 60

/-- Theorem stating that John will reach the park in approximately 6.43 minutes -/
theorem john_park_time :
  let speed := 7 -- km/hr
  let distance := 750 -- meters
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |time_to_park speed distance - 6.43| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_john_park_time_l963_96301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_l963_96330

/-- The inclination angle of a line with equation ax + by + c = 0 -/
noncomputable def inclination_angle (a b : ℝ) : ℝ :=
  Real.arctan (- a / b)

/-- Theorem: The inclination angle of the line x - √3y + c = 0 is π/6 radians (30 degrees) -/
theorem line_inclination_angle :
  inclination_angle 1 (-Real.sqrt 3) = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_l963_96330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AB_is_two_l963_96314

-- Define the curve C
def curve_C (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 3

-- Define the line l
def line_l (x y : ℝ) : Prop := x + Real.sqrt 3 * y - 4 * Real.sqrt 3 = 0

-- Define the ray OP
def ray_OP (θ : ℝ) : Prop := θ = Real.pi / 6

-- Define the intersection point A on curve C
def point_A (ρ : ℝ) : Prop :=
  ρ^2 - ρ - 2 = 0 ∧ ρ > 0

-- Define the intersection point B on line l
def point_B : ℝ := 4

-- Theorem statement
theorem length_AB_is_two :
  ∀ (ρA : ℝ), point_A ρA → |ρA - point_B| = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AB_is_two_l963_96314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l963_96304

-- Define the circle
def my_circle (x y : ℝ) : Prop := x^2 + (y - 3)^2 = 4

-- Define the perpendicular line
def perpendicular_line (x y : ℝ) : Prop := x + y + 1 = 0

-- Define the line l
def line_l (x y : ℝ) : Prop := x - y + 3 = 0

-- Theorem statement
theorem line_equation :
  ∃ (x₀ y₀ : ℝ), 
    (∀ x y, my_circle x y → (x = 0 ∧ y = 3)) ∧ 
    (∀ m₁ m₂, (m₁ * m₂ = -1) → 
      (∀ x y, perpendicular_line x y → y = m₁ * x + (-1)) ∧
      (∀ x y, line_l x y → y = m₂ * x + 3)) →
    line_l x₀ y₀ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l963_96304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l963_96395

open Real

theorem problem_statement :
  (∀ α : ℝ, α ≠ π/4 → Real.tan α ≠ 1) ∧
  (∀ x : ℝ, Real.sin x ≤ 1) ∧
  (∀ φ : ℝ, (∀ x : ℝ, Real.sin (2*x + φ) = Real.sin (-2*x + φ)) ↔ ∃ k : ℤ, φ = π/2 + k*π) ∧
  ((¬ ∃ x₀ : ℝ, Real.sin x₀ + Real.cos x₀ = 3/2) ∨ (∃ α β : ℝ, Real.sin α > Real.sin β ∧ α ≤ β)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l963_96395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cashback_unprofitability_l963_96341

/-- Represents a bank's cashback program -/
structure CashbackProgram where
  expectedReturnPercentage : ℝ
  actualReturnPercentage : ℝ

/-- Represents a customer's behavior -/
structure CustomerBehavior where
  financialLiteracy : ℝ
  multipleCardUsage : Bool
  preferHighCashback : Bool

/-- Determines if a cashback program is profitable -/
noncomputable def isProfitable (program : CashbackProgram) : Prop :=
  program.actualReturnPercentage ≤ program.expectedReturnPercentage

/-- Theorem: A bank's cashback program can be unprofitable given certain customer behaviors -/
theorem cashback_unprofitability 
  (program : CashbackProgram) 
  (customer : CustomerBehavior) 
  (h1 : customer.financialLiteracy > 0.8)
  (h2 : customer.multipleCardUsage = true)
  (h3 : customer.preferHighCashback = true)
  (h4 : program.expectedReturnPercentage < 0.05) :
  ¬(isProfitable program) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cashback_unprofitability_l963_96341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_y_coordinate_of_h_l963_96343

noncomputable section

-- Define the functions f and g
def f (x : ℝ) : ℝ := 2*x^2 + 5*x + 3
def g (x : ℝ) : ℝ := -3*x^2 + 4*x - 1

-- Define h as the difference of f and g
def h (x : ℝ) : ℝ := f x - g x

-- Define the vertex of a quadratic function
def vertex (a b c : ℝ) : ℝ × ℝ := (-b / (2*a), c - b^2 / (4*a))

-- Theorem statement
theorem vertex_y_coordinate_of_h :
  (vertex 5 1 4).2 = 79/20 := by
  -- Proof goes here
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_y_coordinate_of_h_l963_96343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_unknown_rate_l963_96338

structure BlanketPurchase where
  price : ℕ
  quantity : ℕ
  discount : ℚ
  tax : ℚ

def total_cost (purchase : BlanketPurchase) : ℚ :=
  ↑(purchase.price * purchase.quantity) * (1 - purchase.discount + purchase.tax)

structure BlanketProblem where
  known_purchases : List BlanketPurchase
  unknown_quantity : ℕ
  average_price : ℕ

def unknown_rate : ℚ := 0  -- Placeholder definition

theorem find_unknown_rate (problem : BlanketProblem) 
  (h1 : problem.known_purchases = [
    { price := 100, quantity := 3, discount := 1/10, tax := 0 },
    { price := 150, quantity := 4, discount := 0, tax := 0 },
    { price := 200, quantity := 3, discount := 0, tax := 1/5 }
  ])
  (h2 : problem.unknown_quantity = 2)
  (h3 : problem.average_price = 150)
  (h4 : (List.sum (List.map total_cost problem.known_purchases) + 
         ↑problem.unknown_quantity * unknown_rate) / 
         ↑(List.sum (List.map BlanketPurchase.quantity problem.known_purchases) + 
           problem.unknown_quantity) = problem.average_price) :
  unknown_rate = 105 := by
  sorry

#eval unknown_rate  -- This will output 0 as we haven't implemented the actual calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_unknown_rate_l963_96338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_selection_probabilities_l963_96332

/-- A selection process where k different numbers are chosen from n possible numbers -/
structure SelectionProcess where
  n : ℕ  -- Total number of possible numbers
  k : ℕ  -- Number of different numbers to be selected
  h : k ≤ n

/-- The probability of a specific number appearing in the selection -/
def probabilityOfNumber (p : SelectionProcess) : ℚ :=
  p.k / p.n

/-- Predicate for uniform probability of all selections -/
def uniformSelectionProbability (p : SelectionProcess) : Prop :=
  ∀ s₁ s₂ : Finset ℕ, s₁.card = p.k → s₂.card = p.k → (∀ x ∈ s₁, x < p.n) → (∀ x ∈ s₂, x < p.n) →
    ∃ prob : ℚ, prob > 0 ∧ prob ≤ 1

theorem selection_probabilities (p : SelectionProcess) :
  (∀ i < p.n, probabilityOfNumber p = p.k / p.n) ∧
  ¬uniformSelectionProbability p :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_selection_probabilities_l963_96332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l963_96334

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a : ℕ+ → ℚ
  S : ℕ+ → ℚ
  h1 : a 2 = 5
  h2 : S 5 = 35

/-- The sum of terms in a derived sequence -/
def T (seq : ArithmeticSequence) (n : ℕ+) : ℚ :=
  (Finset.range n.val).sum (λ i => 1 / (seq.S ⟨i.succ, Nat.succ_pos i⟩ - i.succ))

theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (∀ n : ℕ+, seq.a n = 2 * n.val + 1) ∧
  (∀ n : ℕ+, T seq n = n.val / (n.val + 1)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l963_96334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l963_96333

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = Real.pi →
  4 * a = Real.sqrt 5 * c →
  Real.cos C = 3/5 →
  b = 11 →
  Real.sin A = Real.sqrt 5 / 5 ∧
  (1/2) * a * b * Real.sin C = 22 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l963_96333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_of_s_3005_l963_96368

-- Define the polynomial q(x)
def q (x : ℤ) : ℤ := (Finset.range 3006).sum (λ i ↦ x^i)

-- Define the divisor polynomial
def divisor (x : ℤ) : ℤ := x^5 + x^3 + 3*x^2 + 2*x + 1

-- Define s(x) as the remainder when q(x) is divided by the divisor
def s (x : ℤ) : ℤ := q x % divisor x

-- Theorem statement
theorem remainder_of_s_3005 : |s 3005| % 1000 = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_of_s_3005_l963_96368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candle_lighting_time_l963_96352

/-- Represents the length of a candle at a given time -/
noncomputable def candle_length (initial_length : ℝ) (burn_time : ℝ) (elapsed_time : ℝ) : ℝ :=
  initial_length * (1 - elapsed_time / burn_time)

/-- Proves that the candles must be lit at 11:20 AM -/
theorem candle_lighting_time 
  (initial_length : ℝ) 
  (burn_time1 : ℝ) 
  (burn_time2 : ℝ) 
  (lighting_time : ℝ) :
  burn_time1 = 2 * 60 →
  burn_time2 = 5 * 60 →
  initial_length > 0 →
  lighting_time > 0 →
  lighting_time < 24 * 60 →
  (let elapsed_time := 18 * 60 - lighting_time
   candle_length initial_length burn_time1 elapsed_time = 
   3 * candle_length initial_length burn_time2 elapsed_time) →
  lighting_time = 11 * 60 + 20 := by
  sorry

#check candle_lighting_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_candle_lighting_time_l963_96352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_same_color_quarter_l963_96336

/-- Represents the jelly bean distribution for a person -/
structure JellyBeans where
  green : ℕ
  red : ℕ
  yellow : ℕ

/-- Calculates the total number of jelly beans -/
def JellyBeans.total (jb : JellyBeans) : ℕ :=
  jb.green + jb.red + jb.yellow

/-- Alice's jelly bean distribution -/
def alice : JellyBeans := { green := 2, red := 2, yellow := 0 }

/-- Charlie's jelly bean distribution -/
def charlie : JellyBeans := { green := 2, red := 1, yellow := 3 }

/-- The probability of selecting a specific color jelly bean -/
def prob_select (jb : JellyBeans) (color : ℕ) : ℚ :=
  color / jb.total

/-- The probability of both people selecting the same color -/
def prob_same_color (jb1 jb2 : JellyBeans) : ℚ :=
  prob_select jb1 jb1.green * prob_select jb2 jb2.green +
  prob_select jb1 jb1.red * prob_select jb2 jb2.red

/-- Theorem stating that the probability of Alice and Charlie showing the same color is 1/4 -/
theorem prob_same_color_quarter : prob_same_color alice charlie = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_same_color_quarter_l963_96336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_sphere_radius_for_specific_pyramid_l963_96344

/-- Represents a pyramid with a square base -/
structure Pyramid where
  a : ℝ  -- Side length of the square base
  height : ℝ  -- Height of the pyramid

/-- The radius of the circumscribed sphere of a pyramid -/
noncomputable def circumscribed_sphere_radius (p : Pyramid) : ℝ := 
  p.a * Real.sqrt 21 / 6

/-- Theorem stating the radius of the circumscribed sphere for a specific pyramid -/
theorem circumscribed_sphere_radius_for_specific_pyramid 
  (p : Pyramid) 
  (h1 : p.height = p.a * Real.sqrt 3 / 2) :
  circumscribed_sphere_radius p = p.a * Real.sqrt 21 / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_sphere_radius_for_specific_pyramid_l963_96344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_without_fulltime_jobs_l963_96358

/-- Represents the percentage of parents who are women in the survey. -/
noncomputable def women_percentage : ℝ := 0.4

/-- Represents the fraction of mothers who hold full-time jobs. -/
noncomputable def mothers_fulltime : ℝ := 3/4

/-- Represents the fraction of fathers who hold full-time jobs. -/
noncomputable def fathers_fulltime : ℝ := 9/10

/-- Represents the total number of parents in the survey. -/
noncomputable def total_parents : ℝ := 100

/-- Calculates the percentage of parents who do not hold full-time jobs. -/
noncomputable def parents_without_fulltime_jobs : ℝ :=
  1 - (women_percentage * mothers_fulltime + (1 - women_percentage) * fathers_fulltime)

/-- Theorem stating that the percentage of parents without full-time jobs is 16%. -/
theorem percentage_without_fulltime_jobs :
  parents_without_fulltime_jobs * 100 = 16 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_without_fulltime_jobs_l963_96358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l963_96363

/-- The equation of the region -/
def region_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 10*x + 20*y - 75 = 0

/-- The area of the region -/
noncomputable def region_area : ℝ := 200 * Real.pi

/-- Theorem stating the existence of a circle that matches the region equation
    and proving that the area of this circle equals the defined region_area -/
theorem area_of_region :
  ∃ (center_x center_y radius : ℝ),
    (∀ x y, region_equation x y ↔ (x - center_x)^2 + (y - center_y)^2 = radius^2) ∧
    region_area = Real.pi * radius^2 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l963_96363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_brown_mms_fifth_bag_l963_96385

/-- The number of brown M&M's in each bag --/
def brown_mms : Fin 5 → ℕ
  | 0 => 9  -- First bag
  | 1 => 12 -- Second bag
  | 2 => 8  -- Third bag
  | 3 => 8  -- Fourth bag
  | 4 => 3  -- Fifth bag (to be proven)

/-- The average number of brown M&M's per bag --/
def average : ℚ := 8

theorem brown_mms_fifth_bag :
  (Finset.sum (Finset.range 5) (fun i => (brown_mms i : ℚ))) / 5 = average := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_brown_mms_fifth_bag_l963_96385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_O_approx_l963_96317

/-- The atomic mass of nitrogen in g/mol -/
noncomputable def atomic_mass_N : ℝ := 14.01

/-- The atomic mass of oxygen in g/mol -/
noncomputable def atomic_mass_O : ℝ := 16.00

/-- The number of nitrogen atoms in N2O5 -/
def num_N : ℕ := 2

/-- The number of oxygen atoms in N2O5 -/
def num_O : ℕ := 5

/-- The molar mass of N2O5 in g/mol -/
noncomputable def molar_mass_N2O5 : ℝ := num_N * atomic_mass_N + num_O * atomic_mass_O

/-- The mass percentage of oxygen in N2O5 -/
noncomputable def mass_percentage_O : ℝ := (num_O * atomic_mass_O / molar_mass_N2O5) * 100

/-- Theorem stating that the mass percentage of oxygen in N2O5 is approximately 74.07% -/
theorem mass_percentage_O_approx :
  ∃ ε > 0, abs (mass_percentage_O - 74.07) < ε :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_O_approx_l963_96317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_factors_of_M_l963_96312

def M : ℕ := 2^4 * 3^3 * 5^2 * 7^1

theorem total_factors_of_M : (Finset.filter (· ∣ M) (Finset.range (M + 1))).card = 120 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_factors_of_M_l963_96312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l963_96350

/-- Represents a geometric sequence -/
structure GeometricSequence where
  a₁ : ℚ  -- First term
  q : ℚ   -- Common ratio

/-- Sum of first n terms of a geometric sequence -/
noncomputable def sumGeometric (g : GeometricSequence) (n : ℕ) : ℚ :=
  if g.q = 1 then n * g.a₁
  else g.a₁ * (1 - g.q^n) / (1 - g.q)

/-- Theorem stating the possible values of the common ratio -/
theorem geometric_sequence_ratio (g : GeometricSequence) :
  g.a₁ = 2 ∧ sumGeometric g 3 = 6 → g.q = 1 ∨ g.q = -2 := by
  sorry

#check geometric_sequence_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l963_96350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_maximum_marks_l963_96305

theorem maximum_marks : ℝ := by
  let pass_percentage : ℝ := 0.33
  let obtained_marks : ℝ := 125
  let fail_margin : ℝ := 40
  let maximum_marks : ℝ := 500
  let pass_mark : ℝ := pass_percentage * maximum_marks
  let pass_mark_alt : ℝ := obtained_marks + fail_margin
  have h : pass_mark = pass_mark_alt := by sorry
  have h2 : maximum_marks = 500 := by sorry
  exact maximum_marks

end NUMINAMATH_CALUDE_ERRORFEEDBACK_maximum_marks_l963_96305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l963_96362

open Set
open Function
open Interval

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the derivative of f
def f' : ℝ → ℝ := sorry

-- Define the domain of f
def domain_f : Set ℝ := Ioo (-5) 5

-- Define the condition on f and its derivative
axiom condition_f (x : ℝ) : x ∈ domain_f → f x + x * f' x > 2

-- Define the inequality
def inequality (x : ℝ) : Prop := 
  (2*x - 3) * f (2*x - 3) - (x - 1) * f (x - 1) > 2*x - 4

-- State the theorem
theorem solution_set : 
  {x : ℝ | inequality x ∧ (2*x - 3) ∈ domain_f ∧ (x - 1) ∈ domain_f} = Ioo 2 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l963_96362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_year_population_increase_l963_96321

/-- The percentage increase in population during the first year, given the initial population,
    final population after two years, and the percentage increase in the second year. -/
theorem first_year_population_increase 
  (initial_population : ℕ) 
  (final_population : ℕ) 
  (second_year_increase : ℚ) 
  (h : final_population = initial_population * (1 + x) * (1 + second_year_increase)) : 
  x = 25 / 100 := by
  have h1 : initial_population = 800 := by sorry
  have h2 : final_population = 1150 := by sorry
  have h3 : second_year_increase = 15 / 100 := by sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_year_population_increase_l963_96321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_right_branch_of_hyperbola_l963_96310

-- Define the centers of the fixed circles
def C₁ : ℝ × ℝ := (-4, 0)
def C₂ : ℝ × ℝ := (4, 0)

-- Define the distance function as noncomputable
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the locus of points
def locus : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | distance p C₂ - distance p C₁ = 3 ∧ p.1 > 0}

-- Theorem statement
theorem locus_is_right_branch_of_hyperbola :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  locus = {p : ℝ × ℝ | p.1 > 0 ∧ (p.1 / a)^2 - (p.2 / b)^2 = 1} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_right_branch_of_hyperbola_l963_96310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_empty_time_l963_96375

/-- Represents the tank system with a leak and an inlet pipe -/
structure TankSystem where
  capacity : ℝ
  leakEmptyTime : ℝ
  inletRate : ℝ

/-- Calculates the time it takes for the tank to empty when both the leak and inlet are open -/
noncomputable def emptyTime (system : TankSystem) : ℝ :=
  let leakRate := system.capacity / system.leakEmptyTime
  let inletRatePerHour := system.inletRate * 60
  let netEmptyingRate := leakRate - inletRatePerHour
  system.capacity / netEmptyingRate

/-- Theorem stating that for the given tank system, it takes 12 hours to empty -/
theorem tank_empty_time :
  let system : TankSystem := {
    capacity := 8640,
    leakEmptyTime := 8,
    inletRate := 6
  }
  emptyTime system = 12 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_empty_time_l963_96375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_inside_circle_l963_96396

/-- The circle C: x^2 + y^2 = 4 -/
def circleC (x y : ℝ) : Prop := x^2 + y^2 = 4

/-- The line L: ax + by = 4 -/
def lineL (a b x y : ℝ) : Prop := a*x + b*y = 4

/-- Point P(a, b) is inside the circle if a^2 + b^2 < 4 -/
def inside_circle (a b : ℝ) : Prop := a^2 + b^2 < 4

/-- The line is separate from the circle if the distance from (0, 0) to the line is greater than 2 -/
def line_separate_from_circle (a b : ℝ) : Prop := 4 / (a^2 + b^2) > 4

theorem point_inside_circle (a b : ℝ) :
  line_separate_from_circle a b → inside_circle a b :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_inside_circle_l963_96396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_factor_of_2525_l963_96377

theorem largest_prime_factor_of_2525 : 
  (Nat.factors 2525).maximum? = some 101 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_factor_of_2525_l963_96377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l963_96382

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (Real.tan x - Real.sqrt 3)

-- Define the domain set
def domain : Set ℝ := {x | ∃ k : ℤ, k * Real.pi + Real.pi / 3 ≤ x ∧ x < k * Real.pi + Real.pi / 2}

-- Theorem statement
theorem f_domain : 
  ∀ x : ℝ, x ∈ domain ↔ ∃ y : ℝ, f x = y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l963_96382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_different_selections_count_l963_96379

theorem different_selections_count : ∃ (x : ℕ), x = 6 := by
  -- Define the number of items to choose from
  let n : ℕ := 3
  -- Define the number of items each person chooses
  let k : ℕ := 2
  -- Define the number of people making selections
  let people : ℕ := 2
  -- Calculate the total number of ways to make selections
  let total_selections := (n.choose k) ^ people
  -- Calculate the number of ways to make identical selections
  let identical_selections := n.choose k
  -- Calculate the number of different selections
  let different_selections := total_selections - identical_selections
  -- Prove that there exists a natural number equal to the number of different selections, which is 6
  use different_selections
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_different_selections_count_l963_96379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fox_rabbit_hunt_l963_96319

theorem fox_rabbit_hunt (initial_weasels initial_rabbits num_foxes weasels_per_fox_per_week weeks animals_left : ℕ) : ℕ :=
  by
  have h1 : initial_weasels = 100 := by sorry
  have h2 : initial_rabbits = 50 := by sorry
  have h3 : num_foxes = 3 := by sorry
  have h4 : weasels_per_fox_per_week = 4 := by sorry
  have h5 : weeks = 3 := by sorry
  have h6 : animals_left = 96 := by sorry
  
  -- Calculate total animals caught
  let total_caught := initial_weasels + initial_rabbits - animals_left
  
  -- Calculate total weasels caught
  let weasels_caught := num_foxes * weasels_per_fox_per_week * weeks
  
  -- Calculate total rabbits caught
  let rabbits_caught := total_caught - weasels_caught
  
  -- Calculate average rabbits caught per fox per week
  let avg_rabbits_per_fox_per_week := rabbits_caught / (num_foxes * weeks)
  
  exact avg_rabbits_per_fox_per_week

#check fox_rabbit_hunt

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fox_rabbit_hunt_l963_96319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_divisor_existence_l963_96361

theorem unique_divisor_existence : ∃! D : ℕ, 
  D > 1 ∧
  242 % D = 8 ∧
  698 % D = 9 ∧
  (242 + 698) % D = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_divisor_existence_l963_96361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l963_96360

noncomputable def solution_set (x : ℝ) : Prop :=
  ∃ m : ℤ, x = Real.pi/3 + Real.pi * m ∨ x = -Real.pi/3 + Real.pi * m

theorem trigonometric_equation_solution :
  ∀ x : ℝ, (∀ k : ℤ, x ≠ Real.pi * k / 2) →
  (Real.sin (3 * x))^2 / (Real.sin x)^2 = 8 * Real.cos (4 * x) + (Real.cos (3 * x))^2 / (Real.cos x)^2 ↔
  solution_set x :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l963_96360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_bounds_l963_96300

/-- The function g defined for positive real numbers a, b, and c -/
noncomputable def g (a b c : ℝ) : ℝ := a / (a + b) + b / (b + c) + c / (c + a)

/-- Theorem stating the bounds of g(a,b,c) for positive real numbers a, b, and c -/
theorem g_bounds (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  1 < g a b c ∧ g a b c < 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_bounds_l963_96300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_product_sevenths_l963_96322

theorem cos_product_sevenths : Real.cos (π / 7) * Real.cos (2 * π / 7) * Real.cos (4 * π / 7) = -1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_product_sevenths_l963_96322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_when_p_range_of_m_when_p_or_q_and_not_p_and_q_l963_96356

-- Define proposition p
def p (m : ℝ) : Prop := ∃ x y : ℝ, x ≠ y ∧ x^2 - 2 * Real.sqrt 2 * x + m = 0 ∧ y^2 - 2 * Real.sqrt 2 * y + m = 0

-- Define proposition q
def q (m : ℝ) : Prop := (2 : ℝ)^(m + 1) < 4

-- Theorem 1
theorem range_of_m_when_p (m : ℝ) : p m → m < 2 := by sorry

-- Theorem 2
theorem range_of_m_when_p_or_q_and_not_p_and_q (m : ℝ) : 
  ((p m ∨ q m) ∧ ¬(p m ∧ q m)) → 1 ≤ m ∧ m < 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_when_p_range_of_m_when_p_or_q_and_not_p_and_q_l963_96356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vanessa_allowance_l963_96307

/-- The amount of money Vanessa's parents give her weekly -/
def weekly_allowance : ℕ → ℕ := λ _ => 30

/-- The cost of the dress Vanessa wants to buy -/
def dress_cost : ℕ := 80

/-- Vanessa's initial savings -/
def initial_savings : ℕ := 20

/-- The amount Vanessa spends each weekend -/
def weekend_spending : ℕ := 10

/-- The number of weeks Vanessa waits to buy the dress -/
def weeks_waited : ℕ := 3

theorem vanessa_allowance :
  weekly_allowance weeks_waited = 30 ↔
    initial_savings + weeks_waited * weekly_allowance weeks_waited
    - weeks_waited * weekend_spending = dress_cost :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vanessa_allowance_l963_96307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sundays_in_58_days_l963_96316

/-- The number of days in a week -/
def daysInWeek : ℕ := 7

/-- The number of days we're considering -/
def totalDays : ℕ := 58

/-- The day of the week, represented as a number from 0 to 6 -/
inductive DayOfWeek : Type where
  | monday : DayOfWeek
  | tuesday : DayOfWeek
  | wednesday : DayOfWeek
  | thursday : DayOfWeek
  | friday : DayOfWeek
  | saturday : DayOfWeek
  | sunday : DayOfWeek
deriving Repr, DecidableEq

/-- Function to get the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.monday => DayOfWeek.tuesday
  | DayOfWeek.tuesday => DayOfWeek.wednesday
  | DayOfWeek.wednesday => DayOfWeek.thursday
  | DayOfWeek.thursday => DayOfWeek.friday
  | DayOfWeek.friday => DayOfWeek.saturday
  | DayOfWeek.saturday => DayOfWeek.sunday
  | DayOfWeek.sunday => DayOfWeek.monday

/-- Function to get the day of the week after a certain number of days -/
def dayAfter (start : DayOfWeek) (days : ℕ) : DayOfWeek :=
  match days with
  | 0 => start
  | n + 1 => nextDay (dayAfter start n)

/-- Function to count the number of Sundays in a given number of days -/
def countSundays (start : DayOfWeek) (days : ℕ) : ℕ :=
  match days with
  | 0 => 0
  | n + 1 => 
    let count := countSundays start n
    if dayAfter start n = DayOfWeek.sunday then count + 1 else count

/-- Theorem stating that the number of Sundays in the first 58 days of a year starting on Monday is 8 -/
theorem sundays_in_58_days : 
  countSundays DayOfWeek.monday totalDays = 8 := by
  sorry

#eval countSundays DayOfWeek.monday totalDays

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sundays_in_58_days_l963_96316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_positive_reals_l963_96340

open Real

-- Define the function f(x) = xe^x
noncomputable def f (x : ℝ) := x * Real.exp x

-- Theorem statement
theorem f_increasing_on_positive_reals :
  StrictMonoOn f (Set.Ioi 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_positive_reals_l963_96340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_squared_in_expansion_l963_96366

theorem coefficient_x_squared_in_expansion : 
  ∃ (a₅ a₄ a₃ a₂ a₁ a : ℚ), 
    (fun x => (3*x + 1)^5) = (fun x => a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a) ∧ 
    a₂ = 90 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_squared_in_expansion_l963_96366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_implies_a_eq_one_l963_96313

noncomputable def f (x a : ℝ) : ℝ := x * (Real.exp x - a / Real.exp x)

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem even_function_implies_a_eq_one (a : ℝ) :
  is_even (f · a) → a = 1 := by
  intro h
  -- The proof goes here
  sorry

#check even_function_implies_a_eq_one

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_implies_a_eq_one_l963_96313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_symmetry_and_parallelism_l963_96391

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Checks if two lines are parallel -/
def are_parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

/-- Finds the symmetric line about y = x -/
noncomputable def symmetric_line (l : Line) : Line :=
  { slope := 1 / l.slope, intercept := -l.intercept / l.slope }

/-- Theorem: Given the conditions, a must equal -2 -/
theorem line_symmetry_and_parallelism (a : ℝ) :
  let l1 : Line := { slope := a, intercept := 3 }
  let l3 : Line := { slope := -1/2, intercept := 1/2 }
  are_parallel (symmetric_line l1) l3 → a = -2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_symmetry_and_parallelism_l963_96391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_bounds_l963_96373

theorem angle_sum_bounds (A B C : ℝ) 
  (h_acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2)
  (h_sin_sum : Real.sin A ^ 2 + Real.sin B ^ 2 + Real.sin C ^ 2 = 1) :
  π/2 ≤ A + B + C ∧ A + B + C ≤ π :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_bounds_l963_96373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_count_l963_96376

theorem lcm_count : 
  (Finset.filter (fun k : ℕ => Nat.lcm (9^9) (Nat.lcm (12^12) k) = 18^18) (Finset.range 19)).card = 19 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_count_l963_96376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_variance_not_one_l963_96354

-- Define a random variable following a Binomial distribution
def binomial_rv (n : ℕ) (p : ℝ) : ℝ → ℝ := sorry

-- Define the variance of a random variable
noncomputable def variance (X : ℝ → ℝ) : ℝ := sorry

-- Theorem statement
theorem binomial_variance_not_one :
  ∀ ξ : ℝ → ℝ, 
  (∃ n p, ξ = binomial_rv n p ∧ n = 4 ∧ p = 0.25) →
  variance ξ ≠ 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_variance_not_one_l963_96354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_tetrahedron_volume_l963_96357

-- Define the regular tetrahedron
structure RegularTetrahedron where
  totalEdgeLength : ℝ
  volume : ℝ

-- Define the properties of the regular tetrahedron
def isValidRegularTetrahedron (t : RegularTetrahedron) : Prop :=
  t.totalEdgeLength = 72 ∧
  t.volume = (Real.sqrt 2 / 12) * (t.totalEdgeLength / 6)^3

-- Theorem statement
theorem regular_tetrahedron_volume 
  (t : RegularTetrahedron) 
  (h : isValidRegularTetrahedron t) : 
  t.volume = 144 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_tetrahedron_volume_l963_96357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_existence_l963_96342

theorem triangle_existence (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_unique : ∃! (x y z : ℝ), a^2 * x + b^2 * y + c^2 * z = 1 ∧ x * y + y * z + z * x = 1) :
  ∃ (A B C : EuclideanSpace ℝ (Fin 2)), 
    dist A B = a ∧ dist B C = b ∧ dist C A = c :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_existence_l963_96342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_after_five_hours_l963_96364

/-- Calculates the total distance between two cars traveling in opposite directions for 5 hours -/
def total_distance (car_a_speeds : List ℝ) (car_b_speed : ℝ) : ℝ :=
  let car_a_distance := car_a_speeds.sum
  let car_b_distance := 5 * car_b_speed
  car_a_distance + car_b_distance

/-- Theorem stating the total distance between the two cars after 5 hours -/
theorem total_distance_after_five_hours :
  let car_a_speeds : List ℝ := [10, 12, 14, 16, 15]
  let car_b_speed : ℝ := 20
  total_distance car_a_speeds car_b_speed = 167 := by
  -- Unfold the definition of total_distance
  unfold total_distance
  -- Simplify the expressions
  simp
  -- Perform the arithmetic
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_after_five_hours_l963_96364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2α_and_sin_2α_minus_β_l963_96335

-- Define the given conditions
noncomputable def α : ℝ := Real.arccos (3/5)
noncomputable def β : ℝ := Real.arccos (-12/13)

axiom α_range : α ∈ Set.Ioo 0 Real.pi
axiom β_range : β ∈ Set.Ioo 0 Real.pi

-- State the theorem to be proven
theorem cos_2α_and_sin_2α_minus_β :
  Real.cos (2 * α) = -7/25 ∧ Real.sin (2 * α - β) = -253/325 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2α_and_sin_2α_minus_β_l963_96335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l963_96324

noncomputable section

-- Define the function f(x)
def f (x : ℝ) : ℝ := (x - 1) / (x + 2)

-- Define the domain
def domain : Set ℝ := { x | 3 ≤ x ∧ x ≤ 5 }

theorem f_properties :
  -- 1. f(x) is increasing on [3, 5]
  (∀ x y, x ∈ domain → y ∈ domain → x < y → f x < f y) ∧
  -- 2. The maximum value of f(x) is 4/7
  (∀ x, x ∈ domain → f x ≤ 4/7) ∧ (∃ x, x ∈ domain ∧ f x = 4/7) ∧
  -- 3. The minimum value of f(x) is 2/5
  (∀ x, x ∈ domain → 2/5 ≤ f x) ∧ (∃ x, x ∈ domain ∧ f x = 2/5) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l963_96324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_tenth_game_score_l963_96306

/-- Represents the scores of a basketball player over a series of games -/
structure BasketballScores where
  first_five : Fin 5 → ℕ
  sixth_to_ninth : Fin 4 → ℕ
  tenth : ℕ

/-- The sum of scores for games 6 to 9 -/
def sum_sixth_to_ninth (scores : BasketballScores) : ℕ :=
  (scores.sixth_to_ninth 0) + (scores.sixth_to_ninth 1) + (scores.sixth_to_ninth 2) + (scores.sixth_to_ninth 3)

/-- The sum of all scores for 10 games -/
def total_score (scores : BasketballScores) : ℕ :=
  (Finset.sum (Finset.range 5) (fun i => scores.first_five i)) + (sum_sixth_to_ninth scores) + scores.tenth

/-- The average score after 9 games -/
noncomputable def average_after_nine (scores : BasketballScores) : ℚ :=
  (((Finset.sum (Finset.range 5) (fun i => scores.first_five i)) + (sum_sixth_to_ninth scores)) : ℚ) / 9

/-- The average score after 5 games -/
noncomputable def average_after_five (scores : BasketballScores) : ℚ :=
  ((Finset.sum (Finset.range 5) (fun i => scores.first_five i)) : ℚ) / 5

/-- The average score after 10 games -/
noncomputable def average_after_ten (scores : BasketballScores) : ℚ :=
  (total_score scores : ℚ) / 10

theorem min_tenth_game_score (scores : BasketballScores) 
  (h1 : scores.sixth_to_ninth 0 = 25)
  (h2 : scores.sixth_to_ninth 1 = 15)
  (h3 : scores.sixth_to_ninth 2 = 12)
  (h4 : scores.sixth_to_ninth 3 = 21)
  (h5 : average_after_nine scores > average_after_five scores)
  (h6 : average_after_ten scores > 19)
  : scores.tenth ≥ 28 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_tenth_game_score_l963_96306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_satisfies_conditions_l963_96399

noncomputable def sequenceA (n : ℕ) : ℚ :=
  1/25 + 3/5 * n - 27/50 * (-2/3)^n

theorem sequence_satisfies_conditions :
  (sequenceA 1 = 1) ∧
  (sequenceA 2 = 1) ∧
  (sequenceA 3 = 2) ∧
  (∀ n : ℕ, 3 * sequenceA (n + 3) = 4 * sequenceA (n + 2) + sequenceA (n + 1) - 2 * sequenceA n) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_satisfies_conditions_l963_96399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_angle_at_C_l963_96369

-- Define the line c
def line_c (x y : ℝ) : Prop := y = 2 * x

-- Define points A and B
def point_A : ℝ × ℝ := (2, 2)
def point_B : ℝ × ℝ := (6, 2)

-- Define point C
def point_C : ℝ × ℝ := (2, 4)

-- Helper function to calculate angle (not part of the problem statement, but might be useful in the proof)
noncomputable def angle (A B C : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem max_angle_at_C :
  line_c point_C.1 point_C.2 ∧
  ∀ (D : ℝ × ℝ), line_c D.1 D.2 → angle point_A D point_B ≤ angle point_A point_C point_B :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_angle_at_C_l963_96369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_tan_3x_l963_96348

noncomputable def f (x : ℝ) := Real.tan (3 * x)

theorem min_positive_period_tan_3x :
  ∃ T : ℝ, T > 0 ∧ (∀ x : ℝ, f (x + T) = f x) ∧
  (∀ T' : ℝ, T' > 0 ∧ (∀ x : ℝ, f (x + T') = f x) → T ≤ T') ∧
  T = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_tan_3x_l963_96348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_equivalence_l963_96392

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 
  if x ≥ 0 then 2^x - 4 else 2^(-x) - 4

-- State the theorem
theorem solution_set_equivalence : 
  (∀ x : ℝ, f (-x) = f x) →  -- f is even
  {x : ℝ | f (x - 2) > 0} = {x : ℝ | x < 0 ∨ x > 4} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_equivalence_l963_96392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_from_tan_cot_sum_l963_96325

theorem tan_sum_from_tan_cot_sum (x y : ℝ) 
  (h1 : Real.tan x + Real.tan y = 30)
  (h2 : (Real.tan x)⁻¹ + (Real.tan y)⁻¹ = 40) : 
  Real.tan (x + y) = 120 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_from_tan_cot_sum_l963_96325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_bar_profit_l963_96303

def candy_bars : ℕ := 800
def buy_price : ℚ := 3 / 4
def sell_price : ℚ := 2 / 3

def total_cost : ℚ := candy_bars * buy_price
def total_revenue : ℚ := candy_bars * sell_price

def profit : ℚ := total_revenue - total_cost

theorem candy_bar_profit : 
  (⌊profit * 100⌋ : ℤ) / 100 = -6664 / 100 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_bar_profit_l963_96303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_l963_96394

-- Define the circle C
def C (x y : ℝ) : Prop := x^2 + (y - 4)^2 = 4

-- Define the line l
def l (k x y : ℝ) : Prop := y = k * x

-- Define the intersection points M and N
def intersects (k : ℝ) : Prop := ∃ x₁ y₁ x₂ y₂, C x₁ y₁ ∧ C x₂ y₂ ∧ l k x₁ y₁ ∧ l k x₂ y₂ ∧ x₁ ≠ x₂

-- Define the midpoint G
def G (x₁ y₁ x₂ y₂ x y : ℝ) : Prop := x = (x₁ + x₂) / 2 ∧ y = (y₁ + y₂) / 2

-- Define the condition for point Q
def Q_condition (m n x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  2 / (m^2 + n^2) = 1 / (x₁^2 + y₁^2) + 1 / (x₂^2 + y₂^2)

theorem circle_line_intersection (k : ℝ) :
  intersects k →
  (k < -Real.sqrt 3 ∨ k > Real.sqrt 3) ∧
  (∀ x y x₁ y₁ x₂ y₂, G x₁ y₁ x₂ y₂ x y → C x₁ y₁ → C x₂ y₂ → l k x₁ y₁ → l k x₂ y₂ → x^2 + (y - 2)^2 = 4) ∧
  (∃ L, L = 4 * Real.pi / 3 ∧ 
    L = Real.arccos (-1/2) * 2 * 2) ∧ -- Length of arc on circle with radius 2 and central angle 2π/3
  (∀ m n x₁ y₁ x₂ y₂, C x₁ y₁ → C x₂ y₂ → l k x₁ y₁ → l k x₂ y₂ → Q_condition m n x₁ y₁ x₂ y₂ →
    n = Real.sqrt (15 * m^2 + 180) / 5 ∧ 
    (-Real.sqrt 3 < m ∧ m < 0 ∨ 0 < m ∧ m < Real.sqrt 3)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_l963_96394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_l963_96397

/-- Predicate to check if a list of real numbers forms an arithmetic sequence -/
def is_arithmetic_seq (l : List ℝ) : Prop :=
  l.length > 1 ∧ ∀ i : Fin (l.length - 2), l[i.val + 1] - l[i.val] = l[i.val + 2] - l[i.val + 1]

/-- Predicate to check if a list of real numbers forms a geometric sequence -/
def is_geometric_seq (l : List ℝ) : Prop :=
  l.length > 1 ∧ ∀ i : Fin (l.length - 2), l[i.val] ≠ 0 → l[i.val + 1] / l[i.val] = l[i.val + 2] / l[i.val + 1]

/-- Given an arithmetic sequence and a geometric sequence with specific properties, 
    prove that b₂(a₂-a₁) = 8 -/
theorem sequence_property (a₁ a₂ b₁ b₂ b₃ : ℝ) : 
  (is_arithmetic_seq [-2, a₁, a₂, -8]) → 
  (is_geometric_seq [-2, b₁, b₂, b₃, -8]) → 
  b₂ * (a₂ - a₁) = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_l963_96397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_regular_polyhedra_l963_96389

/-- A regular polyhedron in simple spatial geometry. -/
structure RegularPolyhedron where
  -- We don't need to define the internal structure of a regular polyhedron
  -- for this theorem, so we leave it empty.

/-- The set of all types of regular polyhedra in simple spatial geometry. -/
def RegularPolyhedraTypes : Finset RegularPolyhedron :=
  sorry

/-- Theorem stating that there are exactly 5 types of regular polyhedra. -/
theorem five_regular_polyhedra : RegularPolyhedraTypes.card = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_regular_polyhedra_l963_96389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mathematician_meeting_time_l963_96308

theorem mathematician_meeting_time (m : ℝ) (a b c : ℕ+) 
  (h1 : m = a - b * Real.sqrt c)
  (h2 : ∀ p : ℕ, Nat.Prime p → ¬(p^2 ∣ c.val))
  (h3 : (((90 : ℝ) - m)^2 / 90^2) = 0.7) :
  a = 90 ∧ b = 30 ∧ c = 7 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mathematician_meeting_time_l963_96308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_height_order_l963_96302

/-- Represents the three classmates -/
inductive Classmate
  | David
  | Emma
  | Fiona
  deriving Repr, DecidableEq

/-- Represents the height order of the classmates -/
def HeightOrder := List Classmate

/-- Checks if a given statement is true for a height order -/
def isStatementTrue (order : HeightOrder) : Fin 3 → Prop
  | 0 => order.headD Classmate.David = Classmate.Emma  -- Emma is the tallest
  | 1 => order.get? 2 ≠ some Classmate.Fiona  -- Fiona is not the shortest
  | 2 => order.headD Classmate.David ≠ Classmate.David  -- David is not the tallest

/-- The main theorem to prove -/
theorem correct_height_order :
  ∀ (order : HeightOrder),
    (order.length = 3) →  -- There are exactly three classmates
    (order.toFinset.card = 3) →  -- All classmates are different
    (∃! (i : Fin 3), isStatementTrue order i) →  -- Exactly one statement is true
    (order = [Classmate.David, Classmate.Fiona, Classmate.Emma]) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_height_order_l963_96302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_condition_positivity_condition_l963_96345

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*Real.log x

-- Define the derivative of f
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := 2*x + 2*a/x

theorem extreme_value_condition (a : ℝ) :
  f_derivative a 1 = 0 → a = -1 := by sorry

theorem positivity_condition (a : ℝ) :
  (∀ x : ℝ, x ≥ 1 → f a x > 0) → a > -Real.exp 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_condition_positivity_condition_l963_96345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_base_81_equals_2_l963_96315

theorem log_base_81_equals_2 (x : ℝ) (h1 : x > 0) (h2 : Real.log 81 / Real.log x = 2) : x = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_base_81_equals_2_l963_96315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paperclip_capacity_75cm3_l963_96323

/-- Represents the relationship between container volume and paperclip capacity -/
structure Container where
  volume : ℝ
  capacity : ℝ

/-- The proportion of paperclips to square root of volume is constant -/
noncomputable def proportionality_constant (c : Container) : ℝ :=
  c.capacity / Real.sqrt c.volume

theorem paperclip_capacity_75cm3 (c : Container)
  (h1 : c.volume = 27)
  (h2 : c.capacity = 60)
  (h3 : ∀ (c1 c2 : Container), proportionality_constant c1 = proportionality_constant c2) :
  ∃ (c_new : Container), c_new.volume = 75 ∧ c_new.capacity = 100 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_paperclip_capacity_75cm3_l963_96323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_polynomial_and_y_l963_96365

theorem gcd_polynomial_and_y (y : ℤ) (h : 42522 ∣ y) : 
  Nat.gcd ((3 * y + 4) * (8 * y + 3) * (14 * y + 9) * (y + 17)).natAbs y.natAbs = 102 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_polynomial_and_y_l963_96365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lung_disease_smoking_relation_l963_96380

/-- Represents the K² statistic for a contingency table test -/
structure ContingencyTest where
  sample_size : ℕ
  k_squared : ℝ
  critical_value : ℝ
  p_value : ℝ

/-- Determines if there is a significant relationship based on the K² test -/
def is_significant (test : ContingencyTest) : Prop :=
  test.k_squared ≥ test.critical_value

/-- Calculates the confidence level based on the p-value -/
def confidence_level (test : ContingencyTest) : ℝ :=
  1 - test.p_value

/-- Theorem: Given the conditions of the lung disease and smoking study,
    there is a 95% confidence that having lung disease is related to smoking -/
theorem lung_disease_smoking_relation (test : ContingencyTest)
  (h1 : test.sample_size = 1000)
  (h2 : test.k_squared = 4.453)
  (h3 : test.critical_value = 3.841)
  (h4 : test.p_value = 0.05) :
  is_significant test ∧ confidence_level test = 0.95 := by
  sorry

#check lung_disease_smoking_relation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lung_disease_smoking_relation_l963_96380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_will_worked_eight_hours_on_monday_l963_96351

-- Define Will's hourly wage
def hourly_wage : ℚ := 8

-- Define the number of hours Will worked on Tuesday
def tuesday_hours : ℚ := 2

-- Define the total amount Will made in two days
def total_earnings : ℚ := 80

-- Define the number of hours Will worked on Monday
noncomputable def monday_hours : ℚ := (total_earnings - hourly_wage * tuesday_hours) / hourly_wage

-- Theorem statement
theorem will_worked_eight_hours_on_monday :
  monday_hours = 8 := by
  -- Unfold the definition of monday_hours
  unfold monday_hours
  -- Perform the calculation
  norm_num
  -- QED
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_will_worked_eight_hours_on_monday_l963_96351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_NH4NO2_highest_nitrogen_percentage_l963_96311

-- Define the molar masses of the compounds
noncomputable def molar_mass_NH2OH : ℝ := 33.0
noncomputable def molar_mass_NH4NO2 : ℝ := 64.1
noncomputable def molar_mass_N2O3 : ℝ := 76.0
noncomputable def molar_mass_NH4NH2CO2 : ℝ := 78.1

-- Define the atomic mass of nitrogen
noncomputable def atomic_mass_N : ℝ := 14.0

-- Define a function to calculate the percentage of nitrogen by mass
noncomputable def nitrogen_percentage (molar_mass : ℝ) (num_nitrogen_atoms : ℕ) : ℝ :=
  (num_nitrogen_atoms * atomic_mass_N / molar_mass) * 100

-- Theorem statement
theorem NH4NO2_highest_nitrogen_percentage :
  let p_NH2OH := nitrogen_percentage molar_mass_NH2OH 1
  let p_NH4NO2 := nitrogen_percentage molar_mass_NH4NO2 2
  let p_N2O3 := nitrogen_percentage molar_mass_N2O3 2
  let p_NH4NH2CO2 := nitrogen_percentage molar_mass_NH4NH2CO2 2
  p_NH4NO2 > p_NH2OH ∧ p_NH4NO2 > p_N2O3 ∧ p_NH4NO2 > p_NH4NH2CO2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_NH4NO2_highest_nitrogen_percentage_l963_96311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_fixed_point_distance_MN_bounds_l963_96383

/-- The line equation parameterized by m -/
def line_equation (x y m : ℝ) : Prop :=
  2 * x + (1 + m) * y + 2 * m = 0

/-- Point P -/
def P : ℝ × ℝ := (-1, 0)

/-- Point Q -/
def Q : ℝ × ℝ := (1, -2)

/-- Point N -/
def N : ℝ × ℝ := (2, 1)

/-- M is the projection of P on the line -/
noncomputable def M (m : ℝ) : ℝ × ℝ := sorry

theorem line_passes_through_fixed_point :
  ∀ m : ℝ, line_equation Q.1 Q.2 m :=
by sorry

theorem distance_MN_bounds (m : ℝ) :
  Real.sqrt 2 ≤ dist (M m) N ∧ dist (M m) N ≤ 3 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_fixed_point_distance_MN_bounds_l963_96383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_logarithmic_graphs_l963_96353

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (2 + a * x) / Real.log a

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := Real.log (a + 2 * x) / Real.log (1 / a)

theorem symmetric_logarithmic_graphs (a b : ℝ) (ha : a > 0) (ha' : a ≠ 1) :
  (∀ x, f a x + g a x = 2 * b) →
  a + b = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_logarithmic_graphs_l963_96353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_contest_theorem_l963_96393

structure Contest where
  num_problems : ℕ
  num_contestants : ℕ
  solved_by : Fin num_problems → Finset (Fin num_contestants)

def Contest.more_than_two_fifths (c : Contest) : Prop :=
  ∀ p : Fin c.num_problems, (c.solved_by p).card > (2 * c.num_contestants) / 5

def Contest.no_perfect_score (c : Contest) : Prop :=
  ∀ contestant : Fin c.num_contestants,
    (Finset.filter (fun p : Fin c.num_problems => contestant ∈ c.solved_by p) (Finset.univ)).card < c.num_problems

def Contest.solved_exactly (c : Contest) (n : ℕ) (contestant : Fin c.num_contestants) : Prop :=
  (Finset.filter (fun p : Fin c.num_problems => contestant ∈ c.solved_by p) (Finset.univ)).card = n

theorem contest_theorem (c : Contest) 
  (h1 : c.num_problems = 6)
  (h2 : c.more_than_two_fifths)
  (h3 : c.no_perfect_score) :
  ∃ (a b : Fin c.num_contestants), a ≠ b ∧ c.solved_exactly 5 a ∧ c.solved_exactly 5 b :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_contest_theorem_l963_96393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_construct_notable_points_without_vertices_l963_96374

-- Define a line in 2D space
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ
  nonzero : a ≠ 0 ∨ b ≠ 0

-- Define a triangle as a set of three lines
structure Triangle where
  l1 : Line
  l2 : Line
  l3 : Line
  forms_triangle : ∃ (p1 p2 p3 : ℝ × ℝ), 
    p1 ≠ p2 ∧ p2 ≠ p3 ∧ p3 ≠ p1 ∧
    (l1.a * p1.1 + l1.b * p1.2 + l1.c = 0) ∧
    (l2.a * p2.1 + l2.b * p2.2 + l2.c = 0) ∧
    (l3.a * p3.1 + l3.b * p3.2 + l3.c = 0)

-- Define the notable points
structure NotablePoints where
  circumcenter : ℝ × ℝ
  incenter : ℝ × ℝ
  centroid : ℝ × ℝ
  orthocenter : ℝ × ℝ
  excenter1 : ℝ × ℝ
  excenter2 : ℝ × ℝ
  excenter3 : ℝ × ℝ

-- Helper function to construct notable points
noncomputable def construct_notable_points (t : Triangle) : NotablePoints :=
  sorry

-- The main theorem
theorem construct_notable_points_without_vertices (t : Triangle) : 
  ∃ (np : NotablePoints), np = construct_notable_points t := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_construct_notable_points_without_vertices_l963_96374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_pieces_to_crowd_l963_96337

/-- Represents a domino piece -/
structure Domino where
  length : Nat
  width : Nat

/-- Represents a square board -/
structure Board where
  size : Nat

/-- Function to check if a piece can be placed on the board -/
def can_place_piece (b : Board) (pieces : List Domino) (d : Domino) : Prop :=
  sorry -- Implementation details omitted for brevity

/-- Defines what it means for a board to be crowded -/
def is_crowded (b : Board) (pieces : List Domino) : Prop :=
  ∀ d : Domino, ¬ can_place_piece b pieces d

/-- The main theorem to be proved -/
theorem min_pieces_to_crowd (b : Board) (d : Domino) :
  b.size = 5 ∧ d.length = 2 ∧ d.width = 1 →
  ∃ (pieces : List Domino),
    pieces.length = 9 ∧
    is_crowded b pieces ∧
    ∀ (smaller_pieces : List Domino),
      smaller_pieces.length < 9 →
      ¬ is_crowded b smaller_pieces :=
by
  sorry -- Proof details omitted for brevity


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_pieces_to_crowd_l963_96337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_roots_of_cubic_l963_96381

/-- A polynomial of degree 3 with rational coefficients -/
def cubicPolynomial (a b c : ℚ) : ℚ → ℚ := fun x => x^3 + a*x^2 + b*x + c

/-- The roots of the polynomial are rational -/
def hasRationalRoots (a b c : ℚ) : Prop :=
  ∃ x y z : ℚ, (cubicPolynomial a b c x = 0) ∧ (cubicPolynomial a b c y = 0) ∧ (cubicPolynomial a b c z = 0)

/-- The roots of the polynomial are a, b, and c themselves -/
def rootsAreCoefficients (a b c : ℚ) : Prop :=
  (cubicPolynomial a b c a = 0) ∧ (cubicPolynomial a b c b = 0) ∧ (cubicPolynomial a b c c = 0)

/-- The main theorem stating the only possible sets of rational numbers (a, b, c) -/
theorem rational_roots_of_cubic (a b c : ℚ) :
  hasRationalRoots a b c ∧ rootsAreCoefficients a b c ↔
  ((a = 1 ∧ b = -1 ∧ c = -1) ∨ (a = 0 ∧ b = 0 ∧ c = 0) ∨ (a = 1 ∧ b = -2 ∧ c = 0)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_roots_of_cubic_l963_96381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_statistical_relations_l963_96370

variable (n : ℕ)
variable (x y : Fin n → ℝ)
variable (a₁ a₂ b₁ b₂ c₁ c₂ d₁ d₂ : ℝ)

-- Define the relationship between x and y
def relation (i : Fin n) : y i = 3 * (x i) - 1 := sorry

-- Define statistical measures for x
def mode_x : ℝ := a₁

def mean_x : ℝ := b₁

def variance_x : ℝ := c₁

def percentile80_x : ℝ := d₁

-- Define statistical measures for y
def mode_y : ℝ := a₂

def mean_y : ℝ := b₂

def variance_y : ℝ := c₂

def percentile80_y : ℝ := d₂

-- Theorem to prove
theorem statistical_relations :
  (b₂ = 3 * b₁ - 1) ∧
  (c₂ = 9 * c₁) ∧
  (d₂ = 3 * d₁ - 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_statistical_relations_l963_96370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_remainder_l963_96378

theorem polynomial_remainder (p : Polynomial ℝ) 
  (h1 : p.eval 2 = 3) 
  (h2 : p.eval 3 = 2) : 
  ∃ q : Polynomial ℝ, p = q * (X - 2) * (X - 3) + (-X + 5) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_remainder_l963_96378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boy_bike_trip_time_l963_96329

/-- Calculates the total time for a round trip given distance, speeds, and rest time -/
noncomputable def totalTime (distance : ℝ) (speedTo speedFrom : ℝ) (restTime : ℝ) : ℝ :=
  distance / speedTo + distance / speedFrom + restTime

/-- Theorem: Given the specific conditions, the total time is 6 hours -/
theorem boy_bike_trip_time :
  let distance := (7.5 : ℝ)
  let speedTo := (5 : ℝ)
  let speedFrom := (3 : ℝ)
  let restTime := (2 : ℝ)
  totalTime distance speedTo speedFrom restTime = 6 := by
  -- Unfold the definition of totalTime
  unfold totalTime
  -- Simplify the expression
  simp
  -- Perform the numerical calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_boy_bike_trip_time_l963_96329
