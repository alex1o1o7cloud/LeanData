import Mathlib

namespace NUMINAMATH_CALUDE_second_day_distance_l1231_123125

-- Define the constants
def first_day_distance : ℝ := 250
def average_speed : ℝ := 33.333333333333336
def time_difference : ℝ := 3

-- Define the theorem
theorem second_day_distance :
  let first_day_time := first_day_distance / average_speed
  let second_day_time := first_day_time + time_difference
  second_day_time * average_speed = 350 := by
  sorry

end NUMINAMATH_CALUDE_second_day_distance_l1231_123125


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l1231_123127

theorem regular_polygon_sides (n : ℕ) : n ≥ 3 →
  (n : ℝ) - (n * (n - 3) / 2) = 2 → n = 4 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l1231_123127


namespace NUMINAMATH_CALUDE_odd_function_root_property_l1231_123155

/-- A function f : ℝ → ℝ is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- x₀ is a root of f(x) + exp(x) = 0 -/
def IsRootOf (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  f x₀ + Real.exp x₀ = 0

theorem odd_function_root_property (f : ℝ → ℝ) (x₀ : ℝ) 
    (h_odd : IsOdd f) (h_root : IsRootOf f x₀) :
    Real.exp (-x₀) * f (-x₀) - 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_root_property_l1231_123155


namespace NUMINAMATH_CALUDE_vector_dot_product_l1231_123191

/-- Given two vectors a and b in ℝ², prove that their dot product is -29 -/
theorem vector_dot_product (a b : ℝ × ℝ) 
  (h1 : a.1 + b.1 = 2 ∧ a.2 + b.2 = -4)
  (h2 : 3 * a.1 - b.1 = -10 ∧ 3 * a.2 - b.2 = 16) :
  a.1 * b.1 + a.2 * b.2 = -29 := by
  sorry

end NUMINAMATH_CALUDE_vector_dot_product_l1231_123191


namespace NUMINAMATH_CALUDE_quadratic_linear_third_quadrant_l1231_123198

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a quadratic equation ax^2 + bx + c = 0 -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a linear function y = mx + b -/
structure LinearFunction where
  m : ℝ
  b : ℝ

/-- Checks if a point is in the third quadrant -/
def isInThirdQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y < 0

/-- Checks if a quadratic equation has no real roots -/
def hasNoRealRoots (eq : QuadraticEquation) : Prop :=
  eq.b^2 - 4*eq.a*eq.c < 0

/-- Checks if a linear function passes through a point -/
def passesThrough (f : LinearFunction) (p : Point) : Prop :=
  p.y = f.m * p.x + f.b

theorem quadratic_linear_third_quadrant 
  (b : ℝ) 
  (quad : QuadraticEquation) 
  (lin : LinearFunction) :
  quad = QuadraticEquation.mk 1 2 (b - 3) →
  hasNoRealRoots quad →
  lin = LinearFunction.mk (-2) b →
  ¬ ∃ (p : Point), isInThirdQuadrant p ∧ passesThrough lin p :=
sorry

end NUMINAMATH_CALUDE_quadratic_linear_third_quadrant_l1231_123198


namespace NUMINAMATH_CALUDE_farmer_brown_animals_legs_l1231_123182

/-- The number of legs for each animal type -/
def chicken_legs : ℕ := 2
def sheep_legs : ℕ := 4
def grasshopper_legs : ℕ := 6
def spider_legs : ℕ := 8

/-- The number of each animal type -/
def num_chickens : ℕ := 7
def num_sheep : ℕ := 5
def num_grasshoppers : ℕ := 10
def num_spiders : ℕ := 3

/-- The total number of legs -/
def total_legs : ℕ := 
  num_chickens * chicken_legs + 
  num_sheep * sheep_legs + 
  num_grasshoppers * grasshopper_legs + 
  num_spiders * spider_legs

theorem farmer_brown_animals_legs : total_legs = 118 := by
  sorry

end NUMINAMATH_CALUDE_farmer_brown_animals_legs_l1231_123182


namespace NUMINAMATH_CALUDE_work_completion_l1231_123112

/-- Represents the number of days it takes to complete the entire work -/
def total_days : ℕ := 40

/-- Represents the number of days y takes to finish the remaining work -/
def remaining_days : ℕ := 32

/-- Represents the fraction of work completed in one day -/
def daily_work_rate : ℚ := 1 / total_days

theorem work_completion (x_days : ℕ) : 
  x_days * daily_work_rate + remaining_days * daily_work_rate = 1 → 
  x_days = 8 := by sorry

end NUMINAMATH_CALUDE_work_completion_l1231_123112


namespace NUMINAMATH_CALUDE_geometric_sequence_third_term_l1231_123135

theorem geometric_sequence_third_term (a : ℕ → ℝ) (q : ℝ) (S₄ : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- Geometric sequence condition
  q = 2 →                       -- Common ratio
  S₄ = 60 →                     -- Sum of first 4 terms
  (a 0 * (1 - q^4)) / (1 - q) = S₄ →  -- Sum formula for geometric sequence
  a 2 = 16 := by               -- Third term (index 2 in 0-based indexing)
sorry

end NUMINAMATH_CALUDE_geometric_sequence_third_term_l1231_123135


namespace NUMINAMATH_CALUDE_yoque_borrowed_amount_l1231_123172

/-- The amount Yoque borrowed -/
def borrowed_amount : ℝ := 150

/-- The number of months for repayment -/
def repayment_period : ℕ := 11

/-- The monthly payment amount -/
def monthly_payment : ℝ := 15

/-- The interest rate as a decimal -/
def interest_rate : ℝ := 0.1

theorem yoque_borrowed_amount :
  borrowed_amount = (monthly_payment * repayment_period) / (1 + interest_rate) :=
by sorry

end NUMINAMATH_CALUDE_yoque_borrowed_amount_l1231_123172


namespace NUMINAMATH_CALUDE_arithmetic_simplification_l1231_123169

theorem arithmetic_simplification : 4 * (8 - 3 + 2) / 2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_simplification_l1231_123169


namespace NUMINAMATH_CALUDE_lowest_cost_per_pack_10_plus_cartons_cost_10_plus_lower_than_5_to_9_l1231_123116

/-- Represents the number of boxes per carton -/
def boxes_per_carton : ℕ := 15

/-- Represents the number of packs per box -/
def packs_per_box : ℕ := 12

/-- Represents the total cost for 12 cartons before discounts -/
def total_cost_12_cartons : ℝ := 3000

/-- Represents the quantity discount for 5 or more cartons -/
def quantity_discount_5_plus : ℝ := 0.10

/-- Represents the quantity discount for 10 or more cartons -/
def quantity_discount_10_plus : ℝ := 0.15

/-- Represents the gold tier membership discount -/
def gold_tier_discount : ℝ := 0.10

/-- Represents the seasonal promotion discount -/
def seasonal_discount : ℝ := 0.03

/-- Theorem stating that purchasing 10 or more cartons results in the lowest cost per pack -/
theorem lowest_cost_per_pack_10_plus_cartons :
  let cost_per_carton := total_cost_12_cartons / 12
  let packs_per_carton := boxes_per_carton * packs_per_box
  let total_discount := quantity_discount_10_plus + gold_tier_discount + seasonal_discount
  let cost_per_carton_after_discount := cost_per_carton * (1 - total_discount)
  cost_per_carton_after_discount / packs_per_carton = 1 :=
sorry

/-- Theorem stating that the cost per pack for 10 or more cartons is lower than for 5-9 cartons -/
theorem cost_10_plus_lower_than_5_to_9 :
  let cost_per_carton := total_cost_12_cartons / 12
  let packs_per_carton := boxes_per_carton * packs_per_box
  let total_discount_10_plus := quantity_discount_10_plus + gold_tier_discount + seasonal_discount
  let total_discount_5_to_9 := quantity_discount_5_plus + gold_tier_discount + seasonal_discount
  let cost_per_pack_10_plus := (cost_per_carton * (1 - total_discount_10_plus)) / packs_per_carton
  let cost_per_pack_5_to_9 := (cost_per_carton * (1 - total_discount_5_to_9)) / packs_per_carton
  cost_per_pack_10_plus < cost_per_pack_5_to_9 :=
sorry

end NUMINAMATH_CALUDE_lowest_cost_per_pack_10_plus_cartons_cost_10_plus_lower_than_5_to_9_l1231_123116


namespace NUMINAMATH_CALUDE_max_acute_angles_octagon_l1231_123101

/-- A convex octagon is a polygon with 8 sides where all interior angles are less than 180 degrees. -/
def ConvexOctagon : Type := Unit

/-- An acute angle is an angle less than 90 degrees. -/
def AcuteAngle : Type := Unit

/-- The number of acute angles in a convex octagon. -/
def num_acute_angles (octagon : ConvexOctagon) : ℕ := sorry

/-- The theorem stating that the maximum number of acute angles in a convex octagon is 4. -/
theorem max_acute_angles_octagon :
  ∀ (octagon : ConvexOctagon), num_acute_angles octagon ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_max_acute_angles_octagon_l1231_123101


namespace NUMINAMATH_CALUDE_missing_angle_in_polygon_l1231_123161

theorem missing_angle_in_polygon (n : ℕ) (sum_angles : ℝ) (common_angle : ℝ) : 
  sum_angles = 3420 →
  common_angle = 150 →
  n > 2 →
  (n - 1) * common_angle + (sum_angles - (n - 1) * common_angle) = sum_angles →
  sum_angles - (n - 1) * common_angle = 420 :=
by sorry

end NUMINAMATH_CALUDE_missing_angle_in_polygon_l1231_123161


namespace NUMINAMATH_CALUDE_heartsuit_three_eight_l1231_123188

-- Define the operation ⊛
def heartsuit (x y : ℝ) : ℝ := 4 * x + 6 * y

-- Theorem statement
theorem heartsuit_three_eight : heartsuit 3 8 = 60 := by
  sorry

end NUMINAMATH_CALUDE_heartsuit_three_eight_l1231_123188


namespace NUMINAMATH_CALUDE_square_root_of_64_l1231_123120

theorem square_root_of_64 : {x : ℝ | x^2 = 64} = {8, -8} := by sorry

end NUMINAMATH_CALUDE_square_root_of_64_l1231_123120


namespace NUMINAMATH_CALUDE_systematic_sampling_interval_count_l1231_123170

theorem systematic_sampling_interval_count :
  let total_employees : ℕ := 840
  let sample_size : ℕ := 42
  let interval_start : ℕ := 481
  let interval_end : ℕ := 720
  let interval_size : ℕ := interval_end - interval_start + 1
  let sampling_interval : ℕ := total_employees / sample_size
  (interval_size : ℚ) / (total_employees : ℚ) * (sample_size : ℚ) = 12 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_interval_count_l1231_123170


namespace NUMINAMATH_CALUDE_eggs_sold_equals_450_l1231_123194

/-- The number of eggs in one tray -/
def eggs_per_tray : ℕ := 30

/-- The initial number of trays to be collected -/
def initial_trays : ℕ := 10

/-- The number of trays dropped (lost) -/
def dropped_trays : ℕ := 2

/-- The number of additional trays added after the accident -/
def additional_trays : ℕ := 7

/-- The total number of eggs sold -/
def eggs_sold : ℕ := (initial_trays - dropped_trays + additional_trays) * eggs_per_tray

theorem eggs_sold_equals_450 : eggs_sold = 450 := by
  sorry

end NUMINAMATH_CALUDE_eggs_sold_equals_450_l1231_123194


namespace NUMINAMATH_CALUDE_range_of_m_l1231_123137

theorem range_of_m (m : ℝ) : 
  (∀ x y : ℝ, x > 0 → y > 0 → (2 * y / x + 9 * x / (2 * y) ≥ m^2 + m)) → 
  (-3 ≤ m ∧ m ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l1231_123137


namespace NUMINAMATH_CALUDE_min_value_of_x_plus_3y_min_value_is_16_min_value_achieved_l1231_123146

theorem min_value_of_x_plus_3y (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x * y = 3 * x + y) :
  ∀ a b : ℝ, a > 0 → b > 0 → a * b = 3 * a + b → x + 3 * y ≤ a + 3 * b :=
by sorry

theorem min_value_is_16 (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x * y = 3 * x + y) :
  x + 3 * y ≥ 16 :=
by sorry

theorem min_value_achieved (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x * y = 3 * x + y) :
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a * b = 3 * a + b ∧ a + 3 * b = 16 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_x_plus_3y_min_value_is_16_min_value_achieved_l1231_123146


namespace NUMINAMATH_CALUDE_investment_growth_l1231_123190

/-- Given an initial investment that grows to $400 after 4 years at 25% simple interest per year,
    prove that the value after 6 years is $500. -/
theorem investment_growth (P : ℝ) : 
  P + P * 0.25 * 4 = 400 → 
  P + P * 0.25 * 6 = 500 := by
  sorry

end NUMINAMATH_CALUDE_investment_growth_l1231_123190


namespace NUMINAMATH_CALUDE_cell_growth_theorem_l1231_123197

def cell_growth (initial : ℕ) (hours : ℕ) : ℕ :=
  if hours = 0 then
    initial
  else
    2 * (cell_growth initial (hours - 1) - 2)

theorem cell_growth_theorem :
  cell_growth 9 8 = 1284 := by
  sorry

end NUMINAMATH_CALUDE_cell_growth_theorem_l1231_123197


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1231_123119

theorem quadratic_inequality_solution (a b : ℝ) :
  (∀ x : ℝ, ax^2 + x + b > 0 ↔ 1 < x ∧ x < 2) →
  a + b = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1231_123119


namespace NUMINAMATH_CALUDE_gain_percent_proof_l1231_123102

/-- Given that the cost of 20 articles equals the selling price of 10 articles,
    prove that the gain percent is 100%. -/
theorem gain_percent_proof (cost : ℝ) (sell_price : ℝ) : 
  (20 * cost = 10 * sell_price) → (sell_price - cost) / cost * 100 = 100 := by
  sorry

end NUMINAMATH_CALUDE_gain_percent_proof_l1231_123102


namespace NUMINAMATH_CALUDE_largest_prime_and_composite_under_20_l1231_123118

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def is_composite (n : ℕ) : Prop := n > 1 ∧ ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

def is_two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n < 100

theorem largest_prime_and_composite_under_20 :
  (∀ n : ℕ, is_two_digit n → n < 20 → is_prime n → n ≤ 19) ∧
  (is_prime 19) ∧
  (∀ n : ℕ, is_two_digit n → n < 20 → is_composite n → n ≤ 18) ∧
  (is_composite 18) :=
sorry

end NUMINAMATH_CALUDE_largest_prime_and_composite_under_20_l1231_123118


namespace NUMINAMATH_CALUDE_trigonometric_sum_zero_l1231_123110

theorem trigonometric_sum_zero : 
  Real.sin (29/6 * Real.pi) + Real.cos (-29/3 * Real.pi) + Real.tan (-25/4 * Real.pi) = 0 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_sum_zero_l1231_123110


namespace NUMINAMATH_CALUDE_largest_power_of_two_dividing_seven_power_minus_one_l1231_123129

theorem largest_power_of_two_dividing_seven_power_minus_one :
  (∀ k : ℕ, k > 14 → ¬(2^k ∣ 7^2048 - 1)) ∧
  (2^14 ∣ 7^2048 - 1) :=
sorry

end NUMINAMATH_CALUDE_largest_power_of_two_dividing_seven_power_minus_one_l1231_123129


namespace NUMINAMATH_CALUDE_trapezoid_total_area_l1231_123108

/-- Represents a trapezoid with given side lengths -/
structure Trapezoid where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ

/-- Calculates the total possible area of the trapezoid with different configurations -/
def totalPossibleArea (t : Trapezoid) : ℝ :=
  sorry

/-- The theorem to be proved -/
theorem trapezoid_total_area :
  let t := Trapezoid.mk 4 6 8 10
  totalPossibleArea t = 48 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_total_area_l1231_123108


namespace NUMINAMATH_CALUDE_total_annual_insurance_cost_l1231_123159

def car_insurance_quarterly : ℕ := 378
def home_insurance_monthly : ℕ := 125
def health_insurance_annual : ℕ := 5045

theorem total_annual_insurance_cost :
  car_insurance_quarterly * 4 + home_insurance_monthly * 12 + health_insurance_annual = 8057 := by
  sorry

end NUMINAMATH_CALUDE_total_annual_insurance_cost_l1231_123159


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l1231_123151

theorem solve_exponential_equation :
  ∀ x : ℝ, (64 : ℝ)^(3*x + 1) = (16 : ℝ)^(4*x - 5) ↔ x = -13 :=
by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l1231_123151


namespace NUMINAMATH_CALUDE_age_difference_l1231_123185

-- Define the ages of A and B
def A : ℕ := sorry
def B : ℕ := 95

-- State the theorem
theorem age_difference : A - B = 5 := by
  -- The condition that in 30 years, A will be twice as old as B was 30 years ago
  have h : A + 30 = 2 * (B - 30) := by sorry
  sorry

end NUMINAMATH_CALUDE_age_difference_l1231_123185


namespace NUMINAMATH_CALUDE_curve_C_equation_min_distance_QM_l1231_123132

-- Define points A and B
def A : ℝ × ℝ := (-1, 2)
def B : ℝ × ℝ := (0, 1)

-- Define the distance condition for point P
def distance_condition (P : ℝ × ℝ) : Prop :=
  (P.1 + 1)^2 + (P.2 - 2)^2 = 2 * (P.1^2 + (P.2 - 1)^2)

-- Define curve C
def C : Set (ℝ × ℝ) := {P | distance_condition P}

-- Define line l₁
def l₁ : Set (ℝ × ℝ) := {Q | 3 * Q.1 - 4 * Q.2 + 12 = 0}

-- Theorem for the equation of curve C
theorem curve_C_equation : C = {P : ℝ × ℝ | (P.1 - 1)^2 + P.2^2 = 4} := by sorry

-- Theorem for the minimum distance
theorem min_distance_QM : 
  ∀ Q ∈ l₁, ∃ M ∈ C, ∀ M' ∈ C, dist Q M ≤ dist Q M' ∧ dist Q M = Real.sqrt 5 := by sorry


end NUMINAMATH_CALUDE_curve_C_equation_min_distance_QM_l1231_123132


namespace NUMINAMATH_CALUDE_sum_of_solutions_eq_seventeen_sixths_l1231_123193

theorem sum_of_solutions_eq_seventeen_sixths :
  let f : ℝ → ℝ := λ x => (3*x + 5) * (2*x - 9)
  (∃ x₁ x₂ : ℝ, f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ ≠ x₂) →
  (∃ x₁ x₂ : ℝ, f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ + x₂ = 17/6) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_eq_seventeen_sixths_l1231_123193


namespace NUMINAMATH_CALUDE_exists_special_function_l1231_123162

/-- A function satisfying specific properties --/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f x = f (-x)) ∧ 
  (f 0 = 1) ∧
  (∀ x, f (x + 3) = -f (-(x + 3))) ∧
  (f (-9) = 0) ∧
  (f 18 = -1) ∧
  (f 24 = 1)

/-- Theorem stating the existence of a function with the specified properties --/
theorem exists_special_function : ∃ f : ℝ → ℝ, special_function f := by
  sorry

end NUMINAMATH_CALUDE_exists_special_function_l1231_123162


namespace NUMINAMATH_CALUDE_complex_arithmetic_l1231_123158

theorem complex_arithmetic (B N T Q : ℂ) : 
  B = 5 - 2*I ∧ N = -5 + 2*I ∧ T = 3*I ∧ Q = 3 →
  B - N + T - Q = 7 - I :=
by sorry

end NUMINAMATH_CALUDE_complex_arithmetic_l1231_123158


namespace NUMINAMATH_CALUDE_inequality_proof_l1231_123183

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  1 / (a + b + c) + 1 / (b + c + d) + 1 / (c + d + a) + 1 / (a + b + d) ≥ 4 / (3 * (a + b + c + d)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1231_123183


namespace NUMINAMATH_CALUDE_negation_equivalence_l1231_123163

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x < 1 ∧ x^2 + 2*x + 1 ≤ 0) ↔ (∀ x : ℝ, x < 1 → x^2 + 2*x + 1 > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1231_123163


namespace NUMINAMATH_CALUDE_chord_length_implies_a_value_l1231_123113

/-- Given a polar coordinate system with a line θ = π/3 and a circle ρ = 2a * sin(θ),
    where the chord length intercepted by the line on the circle is 2√3,
    prove that a = 2. -/
theorem chord_length_implies_a_value (a : ℝ) (h1 : a > 0) : 
  (∃ (ρ : ℝ → ℝ) (θ : ℝ), 
    (θ = π / 3) ∧ 
    (ρ θ = 2 * a * Real.sin θ) ∧
    (∃ (chord_length : ℝ), chord_length = 2 * Real.sqrt 3)) → 
  a = 2 := by
  sorry

end NUMINAMATH_CALUDE_chord_length_implies_a_value_l1231_123113


namespace NUMINAMATH_CALUDE_beverlys_bottle_caps_l1231_123189

def bottle_caps_per_box : ℝ := 35.0
def total_bottle_caps : ℕ := 245

theorem beverlys_bottle_caps :
  (total_bottle_caps : ℝ) / bottle_caps_per_box = 7 :=
sorry

end NUMINAMATH_CALUDE_beverlys_bottle_caps_l1231_123189


namespace NUMINAMATH_CALUDE_total_students_is_540_l1231_123107

/-- Represents the student population of a high school. -/
structure StudentPopulation where
  freshmen : ℕ
  sophomores : ℕ
  juniors : ℕ
  seniors : ℕ

/-- The conditions of the student population problem. -/
def studentPopulationProblem (p : StudentPopulation) : Prop :=
  p.sophomores = 144 ∧
  p.freshmen = (125 * p.juniors) / 100 ∧
  p.sophomores = (90 * p.freshmen) / 100 ∧
  p.seniors = (20 * (p.freshmen + p.sophomores + p.juniors + p.seniors)) / 100

/-- The theorem stating that the total number of students is 540. -/
theorem total_students_is_540 (p : StudentPopulation) 
  (h : studentPopulationProblem p) : 
  p.freshmen + p.sophomores + p.juniors + p.seniors = 540 := by
  sorry


end NUMINAMATH_CALUDE_total_students_is_540_l1231_123107


namespace NUMINAMATH_CALUDE_min_distance_between_points_l1231_123176

/-- Given four points P, Q, R, and S in a metric space, with distances PQ = 12, QR = 5, and RS = 8,
    the minimum possible distance between P and S is 1. -/
theorem min_distance_between_points (X : Type*) [MetricSpace X] 
  (P Q R S : X) 
  (h_PQ : dist P Q = 12)
  (h_QR : dist Q R = 5)
  (h_RS : dist R S = 8) : 
  ∃ (configuration : X → X), dist (configuration P) (configuration S) = 1 ∧ 
    (∀ (config : X → X), dist (config P) (config S) ≥ 1) :=
sorry

end NUMINAMATH_CALUDE_min_distance_between_points_l1231_123176


namespace NUMINAMATH_CALUDE_gourmet_smores_cost_l1231_123165

/-- Represents the cost and pack information for an ingredient --/
structure IngredientInfo where
  single_cost : ℚ
  pack_size : ℕ
  pack_cost : ℚ

/-- Calculates the minimum cost to buy a certain quantity of an ingredient --/
def min_cost (info : IngredientInfo) (quantity : ℕ) : ℚ :=
  let packs_needed := (quantity + info.pack_size - 1) / info.pack_size
  packs_needed * info.pack_cost

/-- Calculates the total cost for all ingredients --/
def total_cost (people : ℕ) (smores_per_person : ℕ) : ℚ :=
  let graham_crackers := min_cost ⟨0.1, 20, 1.8⟩ (people * smores_per_person * 1)
  let marshmallows := min_cost ⟨0.15, 15, 2.0⟩ (people * smores_per_person * 1)
  let chocolate := min_cost ⟨0.25, 10, 2.0⟩ (people * smores_per_person * 1)
  let caramel := min_cost ⟨0.2, 25, 4.5⟩ (people * smores_per_person * 2)
  let toffee := min_cost ⟨0.05, 50, 2.0⟩ (people * smores_per_person * 4)
  graham_crackers + marshmallows + chocolate + caramel + toffee

theorem gourmet_smores_cost : total_cost 8 3 = 26.6 := by
  sorry

end NUMINAMATH_CALUDE_gourmet_smores_cost_l1231_123165


namespace NUMINAMATH_CALUDE_weight_of_replaced_person_l1231_123104

/-- Prove that in a group of 8 persons, if the average weight increases by 2.5 kg
    when a new person weighing 90 kg replaces one of them,
    then the weight of the replaced person is 70 kg. -/
theorem weight_of_replaced_person
  (original_group_size : ℕ)
  (weight_increase : ℝ)
  (new_person_weight : ℝ)
  (h1 : original_group_size = 8)
  (h2 : weight_increase = 2.5)
  (h3 : new_person_weight = 90)
  : ℝ :=
by
  sorry

#check weight_of_replaced_person

end NUMINAMATH_CALUDE_weight_of_replaced_person_l1231_123104


namespace NUMINAMATH_CALUDE_equation_solution_l1231_123147

theorem equation_solution : 
  let f : ℝ → ℝ := λ x => 1/((x - 3)*(x - 4)) + 1/((x - 4)*(x - 5)) + 1/((x - 5)*(x - 6))
  ∀ x : ℝ, f x = 1/8 ↔ x = (9 + Real.sqrt 57)/2 ∨ x = (9 - Real.sqrt 57)/2 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1231_123147


namespace NUMINAMATH_CALUDE_weight_division_l1231_123173

theorem weight_division (n : ℕ) : 
  (∃ (a b c : ℕ), a + b + c = n * (n + 1) / 2 ∧ a = b ∧ b = c) ↔ 
  (n > 3 ∧ (n % 3 = 0 ∨ n % 3 = 2)) :=
by sorry

end NUMINAMATH_CALUDE_weight_division_l1231_123173


namespace NUMINAMATH_CALUDE_builder_nuts_boxes_l1231_123121

/-- Represents the number of boxes of nuts purchased by the builder. -/
def boxes_of_nuts : ℕ := sorry

/-- Represents the number of boxes of bolts purchased by the builder. -/
def boxes_of_bolts : ℕ := 7

/-- Represents the number of bolts in each box. -/
def bolts_per_box : ℕ := 11

/-- Represents the number of nuts in each box. -/
def nuts_per_box : ℕ := 15

/-- Represents the number of bolts left over after the project. -/
def bolts_leftover : ℕ := 3

/-- Represents the number of nuts left over after the project. -/
def nuts_leftover : ℕ := 6

/-- Represents the total number of bolts and nuts used in the project. -/
def total_used : ℕ := 113

theorem builder_nuts_boxes : 
  boxes_of_nuts = 3 ∧
  boxes_of_bolts * bolts_per_box - bolts_leftover + 
  boxes_of_nuts * nuts_per_box - nuts_leftover = total_used :=
sorry

end NUMINAMATH_CALUDE_builder_nuts_boxes_l1231_123121


namespace NUMINAMATH_CALUDE_roots_sum_bound_l1231_123178

theorem roots_sum_bound (z x : ℂ) : 
  z ≠ x → 
  z^2017 = 1 → 
  x^2017 = 1 → 
  Complex.abs (z + x) < Real.sqrt (2 + Real.sqrt 5) :=
sorry

end NUMINAMATH_CALUDE_roots_sum_bound_l1231_123178


namespace NUMINAMATH_CALUDE_zahar_process_terminates_l1231_123115

/-- Represents the state of the notebooks -/
def NotebookState := List Nat

/-- Represents a single operation in Zahar's process -/
def ZaharOperation (state : NotebookState) : Option NotebookState := sorry

/-- Predicate to check if the notebooks are in ascending order -/
def IsAscendingOrder (state : NotebookState) : Prop := sorry

/-- Predicate to check if a state is valid (contains numbers 1 to n) -/
def IsValidState (state : NotebookState) : Prop := sorry

/-- The main theorem stating that Zahar's process will terminate -/
theorem zahar_process_terminates (n : Nat) (initial_state : NotebookState) :
  n ≥ 1 →
  IsValidState initial_state →
  ∃ (final_state : NotebookState) (steps : Nat),
    (∀ k : Nat, k < steps → ∃ intermediate_state, ZaharOperation (intermediate_state) ≠ none) ∧
    ZaharOperation final_state = none ∧
    IsAscendingOrder final_state :=
  sorry

end NUMINAMATH_CALUDE_zahar_process_terminates_l1231_123115


namespace NUMINAMATH_CALUDE_pie_remainder_l1231_123103

theorem pie_remainder (carlos_share : ℝ) (maria_fraction : ℝ) : 
  carlos_share = 0.6 → 
  maria_fraction = 0.5 → 
  (1 - carlos_share) * (1 - maria_fraction) = 0.2 := by
sorry

end NUMINAMATH_CALUDE_pie_remainder_l1231_123103


namespace NUMINAMATH_CALUDE_grains_per_teaspoon_l1231_123167

/-- Represents the number of grains of rice in a cup -/
def grains_per_cup : ℕ := 480

/-- Represents the number of tablespoons in half a cup -/
def tablespoons_per_half_cup : ℕ := 8

/-- Represents the number of teaspoons in a tablespoon -/
def teaspoons_per_tablespoon : ℕ := 3

/-- Theorem stating that there are 10 grains of rice in a teaspoon -/
theorem grains_per_teaspoon : 
  (grains_per_cup : ℚ) / ((2 * tablespoons_per_half_cup) * teaspoons_per_tablespoon) = 10 := by
  sorry

end NUMINAMATH_CALUDE_grains_per_teaspoon_l1231_123167


namespace NUMINAMATH_CALUDE_remainder_problem_l1231_123153

theorem remainder_problem (k : ℕ) 
  (h1 : k % 6 = 5) 
  (h2 : k % 7 = 3) 
  (h3 : k < 41) : 
  k % 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l1231_123153


namespace NUMINAMATH_CALUDE_range_of_sin_minus_cos_l1231_123184

open Real

theorem range_of_sin_minus_cos (x : ℝ) : 
  -Real.sqrt 3 ≤ sin (x + 18 * π / 180) - cos (x + 48 * π / 180) ∧
  sin (x + 18 * π / 180) - cos (x + 48 * π / 180) ≤ Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_sin_minus_cos_l1231_123184


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l1231_123175

/-- Arithmetic sequence with first term 20 and common difference -2 -/
def arithmetic_sequence (n : ℕ) : ℤ := 20 - 2 * (n - 1)

/-- Sum of first n terms of the arithmetic sequence -/
def sum_arithmetic_sequence (n : ℕ) : ℤ := -n^2 + 21*n

theorem arithmetic_sequence_properties :
  ∀ n : ℕ,
  (arithmetic_sequence n = -2*n + 22) ∧
  (sum_arithmetic_sequence n = -n^2 + 21*n) ∧
  (∀ k : ℕ, sum_arithmetic_sequence k ≤ 110) ∧
  (∃ m : ℕ, sum_arithmetic_sequence m = 110) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l1231_123175


namespace NUMINAMATH_CALUDE_second_bag_kernels_l1231_123177

/-- Represents the number of kernels in a bag of popcorn -/
structure PopcornBag where
  total : ℕ
  popped : ℕ

/-- Calculates the percentage of popped kernels in a bag -/
def poppedPercentage (bag : PopcornBag) : ℚ :=
  (bag.popped : ℚ) / (bag.total : ℚ) * 100

theorem second_bag_kernels (bag1 bag2 bag3 : PopcornBag)
  (h1 : bag1.total = 75 ∧ bag1.popped = 60)
  (h2 : bag2.popped = 42)
  (h3 : bag3.total = 100 ∧ bag3.popped = 82)
  (h_avg : (poppedPercentage bag1 + poppedPercentage bag2 + poppedPercentage bag3) / 3 = 82) :
  bag2.total = 50 := by
  sorry


end NUMINAMATH_CALUDE_second_bag_kernels_l1231_123177


namespace NUMINAMATH_CALUDE_problem_solution_l1231_123111

-- Define the conditions
def conditions (a b t : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ a + b = 1 ∧ t = a * b

-- Theorem statement
theorem problem_solution (a b t : ℝ) (h : conditions a b t) :
  (0 < a ∧ a < 1) ∧
  (0 < t ∧ t ≤ 1/4) ∧
  ((a + 1/a) * (b + 1/b) ≥ 25/4) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1231_123111


namespace NUMINAMATH_CALUDE_circle_cutting_terminates_l1231_123148

-- Define the circle-cutting process
def circle_cutting_process (m : ℕ) (n : ℕ) : Prop :=
  m ≥ 2 ∧ ∃ (remaining_area : ℝ), 
    remaining_area > 0 ∧
    remaining_area < (1 - 1/m)^n

-- Theorem statement
theorem circle_cutting_terminates (m : ℕ) :
  m ≥ 2 → ∃ n : ℕ, ∀ k : ℕ, k ≥ n → ¬(circle_cutting_process m k) :=
sorry

end NUMINAMATH_CALUDE_circle_cutting_terminates_l1231_123148


namespace NUMINAMATH_CALUDE_triangle_lines_theorem_l1231_123152

-- Define the triangle vertices
def A : ℝ × ℝ := (4, 0)
def B : ℝ × ℝ := (6, 7)
def C : ℝ × ℝ := (0, 3)

-- Define the line equation type
def LineEquation := ℝ → ℝ → ℝ

-- Define the line AC
def line_AC : LineEquation := fun x y => 3 * x + 4 * y - 12

-- Define the altitude from B to AB
def altitude_B : LineEquation := fun x y => 2 * x + 7 * y - 21

-- Theorem statement
theorem triangle_lines_theorem :
  (∀ x y, line_AC x y = 0 ↔ (x - A.1) * (C.2 - A.2) = (y - A.2) * (C.1 - A.1)) ∧
  (∀ x y, altitude_B x y = 0 ↔ (x - B.1) * (B.1 - A.1) + (y - B.2) * (B.2 - A.2) = 0) :=
sorry

end NUMINAMATH_CALUDE_triangle_lines_theorem_l1231_123152


namespace NUMINAMATH_CALUDE_tangent_parallel_to_x_axis_y_coordinates_l1231_123154

-- Define the function
def f (x : ℝ) : ℝ := x * (x - 4)^3

-- Define the derivative of the function
def f' (x : ℝ) : ℝ := 4 * (x - 4)^2 * (x - 1)

-- Theorem statement
theorem tangent_parallel_to_x_axis :
  ∀ x : ℝ, f' x = 0 ↔ x = 4 ∨ x = 1 :=
sorry

-- Verify the y-coordinates
theorem y_coordinates :
  f 4 = 0 ∧ f 1 = -27 :=
sorry

end NUMINAMATH_CALUDE_tangent_parallel_to_x_axis_y_coordinates_l1231_123154


namespace NUMINAMATH_CALUDE_necessary_condition_for_inequality_l1231_123164

theorem necessary_condition_for_inequality (a : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc 2 3 → x^2 - a ≤ 0) → a ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_necessary_condition_for_inequality_l1231_123164


namespace NUMINAMATH_CALUDE_matchsticks_20th_stage_l1231_123141

def matchsticks (n : ℕ) : ℕ :=
  5 + 3 * (n - 1) + (n - 1) / 5

theorem matchsticks_20th_stage :
  matchsticks 20 = 66 := by
  sorry

end NUMINAMATH_CALUDE_matchsticks_20th_stage_l1231_123141


namespace NUMINAMATH_CALUDE_oranges_per_bag_l1231_123140

theorem oranges_per_bag (total_bags : ℕ) (rotten_oranges : ℕ) (juice_oranges : ℕ) (sold_oranges : ℕ)
  (h1 : total_bags = 10)
  (h2 : rotten_oranges = 50)
  (h3 : juice_oranges = 30)
  (h4 : sold_oranges = 220) :
  (rotten_oranges + juice_oranges + sold_oranges) / total_bags = 30 :=
by sorry

end NUMINAMATH_CALUDE_oranges_per_bag_l1231_123140


namespace NUMINAMATH_CALUDE_polynomial_multiplication_l1231_123133

theorem polynomial_multiplication (t : ℝ) :
  (3 * t^3 - 2 * t^2 + 4 * t - 1) * (2 * t^2 - 5 * t + 3) =
  6 * t^5 - 19 * t^4 + 27 * t^3 - 28 * t^2 + 17 * t - 3 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_multiplication_l1231_123133


namespace NUMINAMATH_CALUDE_fraction_equality_l1231_123149

theorem fraction_equality : 
  (12 - 11 + 10 - 9 + 8 - 7 + 6 - 5 + 4 - 3 + 2 - 1) / 
  (1 - 2 + 3 - 4 + 5 - 6 + 7 - 8 + 9 - 10 + 11) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1231_123149


namespace NUMINAMATH_CALUDE_inequality_proof_l1231_123174

theorem inequality_proof (x y z t : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (ht : t > 0) :
  (x + y + z + t) / 2 + 4 / (x*y + y*z + z*t + t*x) ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1231_123174


namespace NUMINAMATH_CALUDE_rotten_bananas_percentage_l1231_123139

theorem rotten_bananas_percentage (total_oranges total_bananas : ℕ)
  (rotten_oranges_percent good_fruits_percent : ℚ) :
  total_oranges = 600 →
  total_bananas = 400 →
  rotten_oranges_percent = 15 / 100 →
  good_fruits_percent = 898 / 1000 →
  (total_bananas - (good_fruits_percent * (total_oranges + total_bananas : ℚ) -
    ((1 - rotten_oranges_percent) * total_oranges))) / total_bananas = 3 / 100 := by
  sorry

end NUMINAMATH_CALUDE_rotten_bananas_percentage_l1231_123139


namespace NUMINAMATH_CALUDE_max_distance_with_turns_l1231_123181

theorem max_distance_with_turns (total_distance : ℕ) (num_turns : ℕ) 
  (h1 : total_distance = 500) (h2 : num_turns = 300) :
  ∃ (d : ℝ), d ≤ Real.sqrt 145000 ∧ 
  (∀ (a b : ℕ), a + b = total_distance → a ≥ num_turns / 2 → b ≥ num_turns / 2 → 
    Real.sqrt (a^2 + b^2 : ℝ) ≤ d) ∧
  ⌊d⌋ = 380 := by
  sorry

end NUMINAMATH_CALUDE_max_distance_with_turns_l1231_123181


namespace NUMINAMATH_CALUDE_vector_magnitude_proof_l1231_123157

theorem vector_magnitude_proof (a b : ℝ × ℝ) : 
  a = (-1, 2) → b = (1, 3) → ‖(2 • a) - b‖ = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_proof_l1231_123157


namespace NUMINAMATH_CALUDE_cube_sum_preceding_integers_l1231_123142

theorem cube_sum_preceding_integers : ∃ n : ℤ, n = 6 ∧ n^3 = (n-1)^3 + (n-2)^3 + (n-3)^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_preceding_integers_l1231_123142


namespace NUMINAMATH_CALUDE_cookie_chip_ratio_l1231_123145

/-- Proves that the ratio of cookie tins to chip bags is 4:1 given the problem conditions -/
theorem cookie_chip_ratio :
  let chip_weight : ℕ := 20  -- weight of a bag of chips in ounces
  let cookie_weight : ℕ := 9  -- weight of a tin of cookies in ounces
  let chip_bags : ℕ := 6  -- number of bags of chips Jasmine buys
  let total_weight : ℕ := 21 * 16  -- total weight Jasmine carries in ounces

  let cookie_tins : ℕ := (total_weight - chip_weight * chip_bags) / cookie_weight

  (cookie_tins : ℚ) / chip_bags = 4 / 1 :=
by sorry

end NUMINAMATH_CALUDE_cookie_chip_ratio_l1231_123145


namespace NUMINAMATH_CALUDE_range_of_f_l1231_123130

-- Define the linear function
def f (x : ℝ) : ℝ := -2 * x + 5

-- Define the domain
def domain : Set ℝ := {x | -1 < x ∧ x < 1}

-- Theorem statement
theorem range_of_f :
  {y | ∃ x ∈ domain, f x = y} = {y | 3 < y ∧ y < 7} := by sorry

end NUMINAMATH_CALUDE_range_of_f_l1231_123130


namespace NUMINAMATH_CALUDE_distance_equals_abs_l1231_123114

theorem distance_equals_abs (x : ℝ) : |x - 0| = |x| := by
  sorry

end NUMINAMATH_CALUDE_distance_equals_abs_l1231_123114


namespace NUMINAMATH_CALUDE_symmetric_point_polar_axis_l1231_123109

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  r : ℝ
  θ : ℝ

/-- Reflects a polar point about the polar axis -/
def reflectAboutPolarAxis (p : PolarPoint) : PolarPoint :=
  { r := p.r, θ := -p.θ }

theorem symmetric_point_polar_axis (A : PolarPoint) (h : A = { r := 1, θ := π/3 }) :
  reflectAboutPolarAxis A = { r := 1, θ := -π/3 } := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_polar_axis_l1231_123109


namespace NUMINAMATH_CALUDE_ski_race_minimum_participants_l1231_123150

theorem ski_race_minimum_participants : ∀ n : ℕ,
  (∃ k : ℕ, 
    (k : ℝ) / n ≥ 0.035 ∧ 
    (k : ℝ) / n ≤ 0.045 ∧ 
    k > 0) →
  n ≥ 23 :=
by sorry

end NUMINAMATH_CALUDE_ski_race_minimum_participants_l1231_123150


namespace NUMINAMATH_CALUDE_stratified_sampling_ratio_l1231_123196

theorem stratified_sampling_ratio 
  (total_first : ℕ) 
  (total_second : ℕ) 
  (sample_first : ℕ) 
  (sample_second : ℕ) 
  (h1 : total_first = 400) 
  (h2 : total_second = 360) 
  (h3 : sample_first = 60) : 
  (sample_first : ℚ) / total_first = (sample_second : ℚ) / total_second → 
  sample_second = 54 := by
sorry

end NUMINAMATH_CALUDE_stratified_sampling_ratio_l1231_123196


namespace NUMINAMATH_CALUDE_number_exceeding_percentage_l1231_123100

theorem number_exceeding_percentage : ∃ x : ℝ, x = 0.16 * x + 21 ∧ x = 25 := by
  sorry

end NUMINAMATH_CALUDE_number_exceeding_percentage_l1231_123100


namespace NUMINAMATH_CALUDE_parallel_vectors_l1231_123171

/-- Given vectors in R² -/
def a : ℝ × ℝ := (3, 2)
def b : ℝ × ℝ := (-1, 2)
def c : ℝ × ℝ := (1, 6)

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), v.1 * w.2 = t * v.2 * w.1

/-- The main theorem -/
theorem parallel_vectors (k : ℝ) :
  are_parallel (a.1 + k * c.1, a.2 + k * c.2) (a.1 + b.1, a.2 + b.2) ↔ k = 1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_l1231_123171


namespace NUMINAMATH_CALUDE_rollercoaster_interval_l1231_123199

/-- Given that 7 students ride a rollercoaster every certain minutes,
    and 21 students rode the rollercoaster in 15 minutes,
    prove that the time interval for 7 students to ride the rollercoaster is 5 minutes. -/
theorem rollercoaster_interval (students_per_ride : ℕ) (total_students : ℕ) (total_time : ℕ) :
  students_per_ride = 7 →
  total_students = 21 →
  total_time = 15 →
  (total_time / (total_students / students_per_ride) : ℚ) = 5 :=
by sorry

end NUMINAMATH_CALUDE_rollercoaster_interval_l1231_123199


namespace NUMINAMATH_CALUDE_sol_earnings_l1231_123180

/-- Calculates the earnings from candy bar sales over a week -/
def candy_bar_earnings (initial_sales : ℕ) (daily_increase : ℕ) (days : ℕ) (price_cents : ℕ) : ℚ :=
  let total_sales := (List.range days).map (fun i => initial_sales + i * daily_increase) |>.sum
  (total_sales * price_cents : ℕ) / 100

/-- Theorem: Sol's earnings from candy bar sales over a week -/
theorem sol_earnings : candy_bar_earnings 10 4 6 10 = 12 := by
  sorry

end NUMINAMATH_CALUDE_sol_earnings_l1231_123180


namespace NUMINAMATH_CALUDE_cost_per_side_l1231_123187

-- Define the park as a square
structure SquarePark where
  side_cost : ℝ
  total_cost : ℝ

-- Define the properties of the square park
def is_valid_square_park (park : SquarePark) : Prop :=
  park.total_cost = 224 ∧ park.total_cost = 4 * park.side_cost

-- Theorem statement
theorem cost_per_side (park : SquarePark) (h : is_valid_square_park park) : 
  park.side_cost = 56 := by
  sorry

end NUMINAMATH_CALUDE_cost_per_side_l1231_123187


namespace NUMINAMATH_CALUDE_probability_even_sum_l1231_123166

def card_set : Set Nat := {1, 2, 3, 4, 5, 6, 7, 8, 9}

def is_even_sum (pair : Nat × Nat) : Bool :=
  (pair.1 + pair.2) % 2 == 0

def total_combinations : Nat :=
  Nat.choose 9 2

def even_sum_combinations : Nat :=
  Nat.choose 4 2 + Nat.choose 5 2

theorem probability_even_sum :
  (even_sum_combinations : ℚ) / total_combinations = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_probability_even_sum_l1231_123166


namespace NUMINAMATH_CALUDE_farmers_wheat_harvest_l1231_123138

/-- The farmer's wheat harvest problem -/
theorem farmers_wheat_harvest 
  (estimated_harvest : ℕ) 
  (additional_harvest : ℕ) 
  (h1 : estimated_harvest = 48097)
  (h2 : additional_harvest = 684) :
  estimated_harvest + additional_harvest = 48781 :=
by sorry

end NUMINAMATH_CALUDE_farmers_wheat_harvest_l1231_123138


namespace NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l1231_123124

/-- Given two vectors a and b in ℝ², where a = (3, -2) and b = (x, 1),
    prove that if a ⊥ b, then x = 2/3 -/
theorem perpendicular_vectors_x_value (x : ℝ) : 
  let a : Fin 2 → ℝ := ![3, -2]
  let b : Fin 2 → ℝ := ![x, 1]
  (∀ i j, i ≠ j → a i * a j + b i * b j = 0) →
  x = 2/3 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l1231_123124


namespace NUMINAMATH_CALUDE_square_difference_l1231_123117

theorem square_difference (a b : ℝ) (h1 : a + b = 6) (h2 : a - b = 2) : a^2 - b^2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l1231_123117


namespace NUMINAMATH_CALUDE_misha_second_round_score_l1231_123122

/-- Represents the points scored in each round of dart throwing -/
structure DartScores where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Defines the conditions of Misha's dart game -/
def valid_dart_game (scores : DartScores) : Prop :=
  scores.second = 2 * scores.first ∧
  scores.third = (3 * scores.second) / 2 ∧
  scores.first ≥ 24 ∧
  scores.third ≤ 72

/-- Theorem stating that Misha must have scored 48 points in the second round -/
theorem misha_second_round_score (scores : DartScores) 
  (h : valid_dart_game scores) : scores.second = 48 := by
  sorry

end NUMINAMATH_CALUDE_misha_second_round_score_l1231_123122


namespace NUMINAMATH_CALUDE_difference_of_squares_70_30_l1231_123156

theorem difference_of_squares_70_30 : 70^2 - 30^2 = 4000 := by sorry

end NUMINAMATH_CALUDE_difference_of_squares_70_30_l1231_123156


namespace NUMINAMATH_CALUDE_largest_number_is_sqrt5_l1231_123126

theorem largest_number_is_sqrt5 (x y z : ℝ) 
  (sum_eq : x + y + z = 3)
  (sum_prod_eq : x*y + x*z + y*z = -11)
  (prod_eq : x*y*z = 15) :
  max x (max y z) = Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_largest_number_is_sqrt5_l1231_123126


namespace NUMINAMATH_CALUDE_equation_system_solution_l1231_123143

theorem equation_system_solution (a b c d : ℝ) :
  (a + b = c + d) →
  (a^3 + b^3 = c^3 + d^3) →
  ((a = c ∧ b = d) ∨ (a = d ∧ b = c)) :=
by sorry

end NUMINAMATH_CALUDE_equation_system_solution_l1231_123143


namespace NUMINAMATH_CALUDE_increasing_function_inequality_l1231_123160

theorem increasing_function_inequality (f : ℝ → ℝ) (a b : ℝ) 
  (h_increasing : ∀ x y, x < y → f x < f y) 
  (h_sum_positive : a + b > 0) : 
  f a + f b > f (-a) + f (-b) := by
  sorry

end NUMINAMATH_CALUDE_increasing_function_inequality_l1231_123160


namespace NUMINAMATH_CALUDE_average_books_theorem_l1231_123123

/-- Represents the number of books borrowed by a student -/
structure BooksBorrowed where
  count : ℕ
  is_valid : count ≤ 6

/-- Represents the distribution of books borrowed in the class -/
structure ClassDistribution where
  total_students : ℕ
  zero_books : ℕ
  one_book : ℕ
  two_books : ℕ
  at_least_three : ℕ
  is_valid : total_students = zero_books + one_book + two_books + at_least_three

def average_books (dist : ClassDistribution) : ℚ :=
  let total_books := dist.one_book + 2 * dist.two_books + 3 * dist.at_least_three
  total_books / dist.total_students

theorem average_books_theorem (dist : ClassDistribution) 
  (h1 : dist.total_students = 40)
  (h2 : dist.zero_books = 2)
  (h3 : dist.one_book = 12)
  (h4 : dist.two_books = 13)
  (h5 : dist.at_least_three = dist.total_students - (dist.zero_books + dist.one_book + dist.two_books)) :
  average_books dist = 77 / 40 := by
  sorry

#eval (77 : ℚ) / 40

end NUMINAMATH_CALUDE_average_books_theorem_l1231_123123


namespace NUMINAMATH_CALUDE_stock_investment_percentage_l1231_123168

theorem stock_investment_percentage (investment : ℝ) (earnings : ℝ) (percentage : ℝ) :
  investment = 5760 →
  earnings = 1900 →
  percentage = (earnings * 100) / investment →
  percentage = 33 :=
by sorry

end NUMINAMATH_CALUDE_stock_investment_percentage_l1231_123168


namespace NUMINAMATH_CALUDE_class_average_approx_76_percent_l1231_123134

def class_average (group1_percent : ℝ) (group1_score : ℝ) 
                  (group2_percent : ℝ) (group2_score : ℝ) 
                  (group3_percent : ℝ) (group3_score : ℝ) : ℝ :=
  group1_percent * group1_score + group2_percent * group2_score + group3_percent * group3_score

theorem class_average_approx_76_percent :
  let group1_percent : ℝ := 0.15
  let group1_score : ℝ := 100
  let group2_percent : ℝ := 0.50
  let group2_score : ℝ := 78
  let group3_percent : ℝ := 0.35
  let group3_score : ℝ := 63
  let average := class_average group1_percent group1_score group2_percent group2_score group3_percent group3_score
  ∃ ε > 0, |average - 76| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_class_average_approx_76_percent_l1231_123134


namespace NUMINAMATH_CALUDE_circle_radius_from_area_circumference_ratio_l1231_123192

theorem circle_radius_from_area_circumference_ratio 
  (M N : ℝ) (h : M / N = 25) : 
  ∃ (r : ℝ), r > 0 ∧ M = π * r^2 ∧ N = 2 * π * r ∧ r = 50 :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_from_area_circumference_ratio_l1231_123192


namespace NUMINAMATH_CALUDE_gold_award_middle_sum_l1231_123106

/-- Represents the sequence of gold awards --/
def gold_sequence (n : ℕ) : ℚ := sorry

theorem gold_award_middle_sum :
  (∀ i j : ℕ, i < j → i < 10 → j < 10 → gold_sequence j - gold_sequence i = (j - i) * (gold_sequence 1 - gold_sequence 0)) →
  gold_sequence 7 + gold_sequence 8 + gold_sequence 9 = 12 →
  gold_sequence 0 + gold_sequence 1 + gold_sequence 2 + gold_sequence 3 = 12 →
  gold_sequence 4 + gold_sequence 5 + gold_sequence 6 = 83/26 := by
  sorry

end NUMINAMATH_CALUDE_gold_award_middle_sum_l1231_123106


namespace NUMINAMATH_CALUDE_savings_calculation_l1231_123179

theorem savings_calculation (income : ℕ) (ratio_income : ℕ) (ratio_expenditure : ℕ) :
  income = 21000 →
  ratio_income = 7 →
  ratio_expenditure = 6 →
  income - (income * ratio_expenditure / ratio_income) = 3000 := by
  sorry

end NUMINAMATH_CALUDE_savings_calculation_l1231_123179


namespace NUMINAMATH_CALUDE_zoo_animals_count_l1231_123186

/-- Represents the number of four-legged birds -/
def num_birds : ℕ := 14

/-- Represents the number of six-legged calves -/
def num_calves : ℕ := 22

/-- The total number of heads -/
def total_heads : ℕ := 36

/-- The total number of legs -/
def total_legs : ℕ := 100

/-- The number of legs each bird has -/
def bird_legs : ℕ := 4

/-- The number of legs each calf has -/
def calf_legs : ℕ := 6

theorem zoo_animals_count :
  (num_birds + num_calves = total_heads) ∧
  (num_birds * bird_legs + num_calves * calf_legs = total_legs) := by
  sorry

end NUMINAMATH_CALUDE_zoo_animals_count_l1231_123186


namespace NUMINAMATH_CALUDE_pages_per_day_l1231_123128

theorem pages_per_day (book1_pages book2_pages : ℕ) (days : ℕ) 
  (h1 : book1_pages = 180) 
  (h2 : book2_pages = 100) 
  (h3 : days = 14) : 
  (book1_pages + book2_pages) / days = 20 := by
  sorry

end NUMINAMATH_CALUDE_pages_per_day_l1231_123128


namespace NUMINAMATH_CALUDE_free_flowers_per_dozen_l1231_123105

def flowers_per_dozen : ℕ := 12

theorem free_flowers_per_dozen 
  (bought_dozens : ℕ) 
  (total_flowers : ℕ) 
  (h1 : bought_dozens = 3) 
  (h2 : total_flowers = 42) : ℕ := by
  sorry

#check free_flowers_per_dozen

end NUMINAMATH_CALUDE_free_flowers_per_dozen_l1231_123105


namespace NUMINAMATH_CALUDE_perpendicular_lines_k_l1231_123131

/-- Two lines in the plane given by their equations -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The condition for two lines to be perpendicular -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

theorem perpendicular_lines_k (k : ℝ) :
  let l1 : Line := { a := k, b := -1, c := -3 }
  let l2 : Line := { a := 1, b := 2*k+3, c := -2 }
  perpendicular l1 l2 → k = -3 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_k_l1231_123131


namespace NUMINAMATH_CALUDE_mean_proportional_of_segments_l1231_123144

theorem mean_proportional_of_segments (a b : ℝ) (ha : a = 2) (hb : b = 6) :
  ∃ c : ℝ, c > 0 ∧ c^2 = a * b ∧ c = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_mean_proportional_of_segments_l1231_123144


namespace NUMINAMATH_CALUDE_min_sum_squares_l1231_123136

theorem min_sum_squares (x y z : ℝ) (h : x^2 + y^2 + z^2 - 3*x*y*z = 1) :
  ∀ a b c : ℝ, a^2 + b^2 + c^2 - 3*a*b*c = 1 → x^2 + y^2 + z^2 ≤ a^2 + b^2 + c^2 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_squares_l1231_123136


namespace NUMINAMATH_CALUDE_plate_arrangement_theorem_l1231_123195

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def circular_permutations (n : ℕ) (groups : List ℕ) : ℕ :=
  factorial (n - 1) / (groups.map factorial).prod

theorem plate_arrangement_theorem : 
  let total_plates := 14
  let blue_plates := 6
  let red_plates := 3
  let green_plates := 3
  let orange_plates := 2
  let total_arrangements := circular_permutations total_plates [blue_plates, red_plates, green_plates, orange_plates]
  let adjacent_green_arrangements := circular_permutations (total_plates - green_plates + 1) [blue_plates, red_plates, 1, orange_plates]
  total_arrangements - adjacent_green_arrangements = 1349070 := by
  sorry

end NUMINAMATH_CALUDE_plate_arrangement_theorem_l1231_123195
