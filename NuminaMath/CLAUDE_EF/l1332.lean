import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_triangle_area_theorem_l1332_133274

/-- The area of a spherical triangle on a sphere with radius R, formed by the intersection of three lunes with angles α, β, and γ. -/
noncomputable def spherical_triangle_area (R α β γ : ℝ) : ℝ := (α + β + γ - Real.pi) * R^2

/-- Theorem stating that the area of a spherical triangle ABC on a sphere with radius R, 
    formed by the intersection of three lunes with angles α, β, and γ, 
    is equal to (α + β + γ - π) * R². -/
theorem spherical_triangle_area_theorem (R α β γ : ℝ) (h_R : R > 0) :
  spherical_triangle_area R α β γ = (α + β + γ - Real.pi) * R^2 := by
  -- Unfold the definition of spherical_triangle_area
  unfold spherical_triangle_area
  -- The result follows directly from the definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_triangle_area_theorem_l1332_133274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_sin_cos_product_l1332_133264

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.sin x * Real.cos x

-- State the theorem
theorem min_positive_period_sin_cos_product :
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T' ≥ T) ∧
  T = Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_sin_cos_product_l1332_133264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y1_equivalent_l1332_133271

noncomputable def y1 (x : ℝ) : ℝ :=
  if x < 1 then 1 - 2*x
  else if x < 2 then -1
  else if x < 3 then 3 - 2*x
  else 2*x - 9

theorem y1_equivalent (x : ℝ) :
  y1 x = |x - 1| - |x - 2| + 2*|x - 3| - 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y1_equivalent_l1332_133271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1332_133258

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the area function
noncomputable def area (t : Triangle) : ℝ := 
  1 / 2 * t.b * t.a * Real.sin t.B

-- State the theorem
theorem triangle_problem (t : Triangle) 
  (h1 : t.a * Real.sin t.B = Real.sqrt 3 * t.b * Real.cos t.A)
  (h2 : t.a = 3)
  (h3 : t.b = 2 * t.c) :
  t.A = π / 3 ∧ area t = 3 * Real.sqrt 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1332_133258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_independence_check_l1332_133255

-- Define the concept of linear independence for two functions
def LinearlyIndependent (f g : ℝ → ℝ) : Prop :=
  ∀ a b : ℝ, (∀ x : ℝ, a * (f x) + b * (g x) = 0) → a = 0 ∧ b = 0

-- Define the concept of linear dependence for two functions
def LinearlyDependent (f g : ℝ → ℝ) : Prop :=
  ¬(LinearlyIndependent f g)

theorem linear_independence_check :
  (LinearlyIndependent (λ x : ℝ => x) (λ x : ℝ => x^2)) ∧
  (LinearlyIndependent (λ x : ℝ => 1) (λ x : ℝ => x)) ∧
  (LinearlyDependent (λ x : ℝ => x) (λ x : ℝ => 2*x)) ∧
  (∀ C : ℝ, C ≠ 0 → LinearlyDependent (λ x : ℝ => Real.cos x) (λ x : ℝ => C * Real.cos x)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_independence_check_l1332_133255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_distance_proof_monthly_gasoline_cost_proof_l1332_133201

-- Define the standard distance
noncomputable def standard_distance : ℝ := 50

-- Define the daily distance differences
def daily_differences : List ℝ := [-8, -11, -14, 0, -16, 41, 8]

-- Define gasoline consumption rate
noncomputable def gasoline_rate : ℝ := 6 / 100

-- Define gasoline price
noncomputable def gasoline_price : ℝ := 6.2

-- Define number of days in a month
def days_in_month : ℕ := 30

-- Theorem for average distance
theorem average_distance_proof :
  (standard_distance + (daily_differences.sum / daily_differences.length : ℝ)) = 50 := by
  sorry

-- Theorem for monthly gasoline cost
theorem monthly_gasoline_cost_proof :
  (standard_distance * gasoline_rate * gasoline_price * days_in_month) = 558 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_distance_proof_monthly_gasoline_cost_proof_l1332_133201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_of_line_l1332_133226

/-- The fixed point of the line (2+λ)x+(λ-1)y-2λ=10 is (1,1) for all λ. -/
theorem fixed_point_of_line (lambda : ℝ) : 
  (2 + lambda) * 1 + (lambda - 1) * 1 - 2 * lambda = 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_of_line_l1332_133226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_square_side_length_l1332_133260

/-- The length of a square inscribed in a regular hexagon -/
noncomputable def squareSideLength (hexagonSideLength : ℝ) : ℝ :=
  3 * hexagonSideLength - Real.sqrt (3 * hexagonSideLength ^ 2)

/-- Theorem: The length of a square inscribed in a regular hexagon with side length 1,
    such that two sides of the square are parallel to two sides of the hexagon,
    is equal to 3 - √3. -/
theorem inscribed_square_side_length :
  squareSideLength 1 = 3 - Real.sqrt 3 := by
  sorry

-- This line is commented out as it's not computable
-- #eval squareSideLength 1

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_square_side_length_l1332_133260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_closer_to_longest_side_l1332_133202

/-- Represents a trapezoid with given properties -/
structure Trapezoid where
  AB : ℝ
  AD : ℝ
  BC : ℝ
  angleA : ℝ
  angleB : ℝ
  AB_eq : AB = 150
  AD_eq : AD = 300
  BC_eq : BC = 150
  angleA_eq : angleA = 75
  angleB_eq : angleB = 105
  AD_longest : AD ≥ BC ∧ AD ≥ AB

/-- The fraction of the trapezoid's area closer to the longest side -/
noncomputable def fractionCloserToLongestSide (t : Trapezoid) : ℝ := 1 / 2

/-- Theorem stating that the fraction of the area closer to the longest side is 1/2 -/
theorem fraction_closer_to_longest_side (t : Trapezoid) :
  fractionCloserToLongestSide t = 1 / 2 := by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_closer_to_longest_side_l1332_133202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_f_max_at_14_l1332_133233

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x + 35) + Real.sqrt (23 - x) + 2 * Real.sqrt x

-- State the theorem
theorem f_max_value :
  ∃ (x : ℝ), 0 ≤ x ∧ x ≤ 23 ∧ f x = 15 ∧
  ∀ (y : ℝ), 0 ≤ y ∧ y ≤ 23 → f y ≤ 15 :=
by
  -- The proof would go here
  sorry

-- State that the maximum occurs at x = 14
theorem f_max_at_14 :
  f 14 = 15 ∧
  ∀ (y : ℝ), 0 ≤ y ∧ y ≤ 23 → f y ≤ f 14 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_f_max_at_14_l1332_133233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sin_B_l1332_133286

noncomputable def B (a b A : ℝ) : ℝ := Real.arcsin ((b * Real.sin A) / a)

theorem triangle_sin_B (a b A : ℝ) :
  a = 3 * Real.sqrt 3 →
  b = 4 →
  A = 30 * π / 180 →
  Real.sin (B a b A) = (2 * Real.sqrt 3) / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sin_B_l1332_133286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_pit_problem_l1332_133268

theorem ball_pit_problem (total : ℕ) (red_fraction : ℚ) (blue_fraction : ℚ)
  (h_total : total = 360)
  (h_red : red_fraction = 1 / 4)
  (h_blue : blue_fraction = 1 / 5) :
  total - (red_fraction * ↑total).floor - (blue_fraction * ↑(total - (red_fraction * ↑total).floor)).floor = 216 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_pit_problem_l1332_133268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_93gons_constant_ratio_l1332_133208

/-- Two inscribed 93-gons with parallel corresponding sides have a constant ratio of side lengths -/
theorem inscribed_93gons_constant_ratio 
  (A B : Fin 93 → ℝ × ℝ) 
  (inscribed_A : ∀ i, (A i).1^2 + (A i).2^2 = 1)
  (inscribed_B : ∀ i, (B i).1^2 + (B i).2^2 = 1)
  (parallel_sides : ∀ i : Fin 93, 
    (A (i+1) - A i).1 * (B (i+1) - B i).2 = (A (i+1) - A i).2 * (B (i+1) - B i).1)
  : ∃ c : ℝ, ∀ i : Fin 93, 
    ((A (i+1) - A i).1^2 + (A (i+1) - A i).2^2) = 
    c^2 * ((B (i+1) - B i).1^2 + (B (i+1) - B i).2^2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_93gons_constant_ratio_l1332_133208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_business_metrics_correct_l1332_133290

/-- The business model for a mineral water production company --/
structure WaterBusiness where
  initial_investment : ℚ
  equipment_cost : ℚ
  max_capacity : ℚ
  july_production : ℚ
  august_production : ℚ
  september_production : ℚ
  desired_profit : ℚ

/-- Calculate the total depreciation, residual value, and required sales price --/
def calculate_business_metrics (b : WaterBusiness) : ℚ × ℚ × ℚ :=
  let depreciation_per_bottle := b.equipment_cost / b.max_capacity
  let total_production := b.july_production + b.august_production + b.september_production
  let total_depreciation := depreciation_per_bottle * total_production
  let residual_value := b.equipment_cost - total_depreciation
  let required_sales_price := residual_value + b.desired_profit
  (total_depreciation, residual_value, required_sales_price)

/-- The main theorem to prove --/
theorem water_business_metrics_correct (b : WaterBusiness) 
  (h1 : b.initial_investment = 1500000)
  (h2 : b.equipment_cost = 500000)
  (h3 : b.max_capacity = 100000)
  (h4 : b.july_production = 200)
  (h5 : b.august_production = 15000)
  (h6 : b.september_production = 12300)
  (h7 : b.desired_profit = 10000) :
  calculate_business_metrics b = (137500, 362500, 372500) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_business_metrics_correct_l1332_133290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_can_buy_22_candies_can_buy_25_candies_l1332_133261

/-- Represents the prices of candies in cents -/
def CandyPrices := List ℕ

/-- The total number of different candies -/
def totalCandies : ℕ := 44

/-- Anna's budget in cents -/
def annaBudget : ℕ := 75

/-- The total cost of all candies in cents -/
def totalCost : ℕ := 151

/-- Theorem stating that Anna can buy at least 22 different candies -/
theorem can_buy_22_candies (prices : CandyPrices) 
  (h1 : prices.length = totalCandies)
  (h2 : prices.sum = totalCost) :
  ∃ (subset : List ℕ), subset.length ≥ 22 ∧ subset.sum ≤ annaBudget ∧ subset.toFinset ⊆ prices.toFinset := by
  sorry

/-- Theorem stating that Anna can buy at least 25 different candies -/
theorem can_buy_25_candies (prices : CandyPrices)
  (h1 : prices.length = totalCandies)
  (h2 : prices.sum = totalCost) :
  ∃ (subset : List ℕ), subset.length ≥ 25 ∧ subset.sum ≤ annaBudget ∧ subset.toFinset ⊆ prices.toFinset := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_can_buy_22_candies_can_buy_25_candies_l1332_133261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_on_circle_l1332_133252

-- Define the complex equation
def complex_equation (z : ℂ) : Prop :=
  (z + 2) ^ 6 = 64 * z ^ 6

-- Define the circle
def circle_equation (z : ℂ) : Prop :=
  (z.re - 2/3) ^ 2 + z.im ^ 2 = (4/3) ^ 2

-- Theorem statement
theorem roots_on_circle :
  ∀ z : ℂ, complex_equation z → circle_equation z := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_on_circle_l1332_133252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lending_period_is_four_years_l1332_133211

/-- Calculates the number of years for a lending period given the principal amount,
    interest rates, and total gain. -/
noncomputable def lendingPeriod (principal : ℝ) (rateAB rateBC : ℝ) (totalGain : ℝ) : ℝ :=
  let annualInterestAB := principal * rateAB
  let annualInterestBC := principal * rateBC
  let annualGain := annualInterestBC - annualInterestAB
  totalGain / annualGain

/-- Theorem stating that under the given conditions, the lending period is 4 years. -/
theorem lending_period_is_four_years :
  let principal := (2000 : ℝ)
  let rateAB := (0.15 : ℝ)
  let rateBC := (0.17 : ℝ)
  let totalGain := (160 : ℝ)
  lendingPeriod principal rateAB rateBC totalGain = 4 := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lending_period_is_four_years_l1332_133211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rainbow_preschool_full_day_students_l1332_133247

/-- The number of full-day students at Rainbow Preschool -/
def full_day_students (total_students : ℕ) (half_day_percentage : ℚ) : ℕ :=
  total_students - Int.toNat ((↑total_students * half_day_percentage).floor)

/-- Theorem stating the number of full-day students at Rainbow Preschool -/
theorem rainbow_preschool_full_day_students :
  full_day_students 80 (1/4) = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rainbow_preschool_full_day_students_l1332_133247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_triangle_selection_sixteen_sufficient_l1332_133291

/-- A function that checks if three numbers can form a triangle --/
def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- The set of numbers from which we choose --/
def number_set : Set ℕ := {n : ℕ | 1 ≤ n ∧ n ≤ 1000}

/-- The main theorem stating that 16 is the minimum number --/
theorem min_triangle_selection :
  ∀ (S : Finset ℕ), S.toSet ⊆ number_set → 
  (∀ (a b c : ℕ), a ∈ S → b ∈ S → c ∈ S → a ≠ b → b ≠ c → a ≠ c → is_triangle a b c) → 
  S.card ≥ 16 :=
sorry

/-- Theorem stating that 16 is indeed sufficient --/
theorem sixteen_sufficient :
  ∃ (S : Finset ℕ), S.toSet ⊆ number_set ∧ S.card = 16 ∧
  (∀ (a b c : ℕ), a ∈ S → b ∈ S → c ∈ S → a ≠ b → b ≠ c → a ≠ c → is_triangle a b c) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_triangle_selection_sixteen_sufficient_l1332_133291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exact_time_proof_l1332_133240

/-- Represents the position of a clock hand in degrees -/
def ClockPosition := ℝ

/-- Calculates the position of the minute hand at time t (in minutes after 3:00) -/
def minuteHandPosition (t : ℝ) : ClockPosition :=
  6 * t

/-- Calculates the position of the hour hand at time t (in minutes after 3:00) -/
def hourHandPosition (t : ℝ) : ClockPosition :=
  90 + 0.5 * t

/-- The time is between 3:00 and 4:00 -/
def time_range (t : ℝ) : Prop := 0 ≤ t ∧ t < 60

theorem exact_time_proof :
  ∃ t : ℝ, time_range t ∧
    minuteHandPosition (t + 5) = hourHandPosition (t - 2) ∧
    t = 10 + 4/5 :=
by
  -- The proof goes here
  sorry

#check exact_time_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exact_time_proof_l1332_133240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_outside_triangle_l1332_133259

/-- Represents a right triangle with vertices P, Q, and R. -/
def RightTriangle (P Q R : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

/-- Represents a circle in a 2D Euclidean space. -/
def Circle : Type := EuclideanSpace ℝ (Fin 2) × ℝ

/-- Represents the distance between two points. -/
def distance (P Q : EuclideanSpace ℝ (Fin 2)) : ℝ := sorry

/-- Represents a circle being tangent to two sides of a triangle. -/
def CircleTangentToSides (circle : Circle) (P Q R : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

/-- Represents the property that the points diametrically opposite to the 
    tangent points lie on a specific side of the triangle. -/
def DiametricallyOppositePointsOnSide (circle : Circle) (Q R : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

/-- Calculates the area of the portion of a circle that lies outside a triangle. -/
noncomputable def AreaOfCircleOutsideTriangle (circle : Circle) (P Q R : EuclideanSpace ℝ (Fin 2)) : ℝ := sorry

/-- Given a right triangle PQR with PQ = 9 and a circle tangent to PQ and PR 
    with diametrically opposite points on QR, the area of the circle outside 
    the triangle is 9(π - 2)/4. -/
theorem circle_area_outside_triangle (P Q R : EuclideanSpace ℝ (Fin 2)) (circle : Circle) :
  RightTriangle P Q R →
  distance P Q = 9 →
  CircleTangentToSides circle P Q R →
  DiametricallyOppositePointsOnSide circle Q R →
  AreaOfCircleOutsideTriangle circle P Q R = 9 * (Real.pi - 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_outside_triangle_l1332_133259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_junhyung_trip_distance_l1332_133269

/-- The total distance of Junhyung's trip in kilometers -/
noncomputable def D : ℝ := 7.5

/-- The distance traveled on the first day -/
noncomputable def first_day : ℝ := D / 5

/-- The distance traveled on the second day -/
noncomputable def second_day : ℝ := 3 / 4 * (D - first_day)

/-- The distance traveled on the third day in kilometers -/
noncomputable def third_day : ℝ := 1.5

theorem junhyung_trip_distance :
  first_day + second_day + third_day = D ∧ D = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_junhyung_trip_distance_l1332_133269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_horizontal_distance_l1332_133250

/-- Calculates the horizontal distance traveled given an elevation change and vertical-to-horizontal ratio --/
noncomputable def horizontalDistance (initialElevation finalElevation : ℝ) (verticalToHorizontalRatio : ℝ) : ℝ :=
  (finalElevation - initialElevation) / verticalToHorizontalRatio

/-- Theorem: John's horizontal distance traveled is 2700 feet --/
theorem johns_horizontal_distance :
  let initialElevation : ℝ := 100
  let finalElevation : ℝ := 1450
  let verticalToHorizontalRatio : ℝ := 1 / 2
  horizontalDistance initialElevation finalElevation verticalToHorizontalRatio = 2700 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_horizontal_distance_l1332_133250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_orthogonal_when_z_is_two_l1332_133239

/-- Two vectors in ℝ³ are orthogonal if their dot product is zero. -/
def orthogonal (v w : Fin 3 → ℝ) : Prop :=
  (v 0) * (w 0) + (v 1) * (w 1) + (v 2) * (w 2) = 0

/-- The first vector -/
def v : Fin 3 → ℝ := ![3, -1, 5]

/-- The second vector -/
def w (z : ℝ) : Fin 3 → ℝ := ![4, z, -2]

/-- Theorem: The vectors v and w are orthogonal when z = 2 -/
theorem vectors_orthogonal_when_z_is_two : orthogonal v (w 2) := by
  unfold orthogonal v w
  simp
  ring


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_orthogonal_when_z_is_two_l1332_133239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_difference_is_20_l1332_133263

-- Define the given conditions
noncomputable def maya_distance : ℝ := 5 -- miles
noncomputable def kai_distance : ℝ := 10 -- miles
noncomputable def travel_time : ℝ := 15 / 60 -- hours (15 minutes converted to hours)

-- Define the average speed calculation function
noncomputable def average_speed (distance : ℝ) (time : ℝ) : ℝ := distance / time

-- Theorem statement
theorem speed_difference_is_20 :
  average_speed kai_distance travel_time - average_speed maya_distance travel_time = 20 := by
  -- Expand the definition of average_speed
  unfold average_speed
  -- Simplify the expression
  simp [kai_distance, maya_distance, travel_time]
  -- The proof steps would go here, but we'll use sorry to skip the proof
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_difference_is_20_l1332_133263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conical_pendulum_ceiling_height_l1332_133220

/-- Represents a conical pendulum system -/
structure ConicalPendulum where
  rope_length : ℝ
  ball_mass : ℝ
  revolution_period : ℝ

/-- Calculates the height of the ceiling above the horizontal plane for a conical pendulum -/
noncomputable def ceiling_height (p : ConicalPendulum) : ℝ :=
  let g : ℝ := 9.8
  let ω : ℝ := 2 * Real.pi / p.revolution_period
  let cos_θ : ℝ := g / (ω^2 * p.rope_length)
  p.rope_length * cos_θ

/-- Theorem stating that for the given conical pendulum, the ceiling height is approximately 6.2 cm -/
theorem conical_pendulum_ceiling_height :
  let p : ConicalPendulum := ⟨0.50, 0.003, 1.0⟩
  ∃ ε > 0, |ceiling_height p - 0.062| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_conical_pendulum_ceiling_height_l1332_133220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l1332_133225

open Real

theorem triangle_abc_properties (a b c : ℝ) (A B C : ℝ) :
  a = 3 → A = π / 3 →
  (∀ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 → x / sin x = y / sin y → x / sin x = z / sin z) →
  (∀ x y z : ℝ, cos x = (y^2 + z^2 - x^2) / (2 * y * z)) →
  (b = 2 → cos B = sqrt 6 / 3) ∧
  (∀ S : ℝ, S = 1/2 * b * c * sin A → S ≤ 9 * sqrt 3 / 4) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l1332_133225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_points_collinear_iff_product_one_l1332_133277

-- Define the vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define the vectors and scalars
variable (a b : V) (l m : ℝ)

-- Define the condition that a and b are not collinear
def not_collinear (a b : V) : Prop := ∀ (r : ℝ), a ≠ r • b

-- Define the vectors AB and AC
def AB (a b : V) (l : ℝ) : V := l • a + b
def AC (a b : V) (m : ℝ) : V := a + m • b

-- Define collinearity of three points
def collinear (A B C : V) : Prop := ∃ (t : ℝ), B - A = t • (C - A) ∨ C - A = t • (B - A)

-- State the theorem
theorem points_collinear_iff_product_one
  (h_not_collinear : not_collinear a b) :
  collinear (0 : V) (AB a b l) (AC a b m) ↔ l * m = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_points_collinear_iff_product_one_l1332_133277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_multiple_of_five_l1332_133256

def factorial (n : ℕ) : ℕ := Nat.factorial n

theorem expression_multiple_of_five (n : ℤ) (h : n ≥ 10) :
  ∃ k : ℤ, (factorial (n.toNat + 3) - factorial (n.toNat + 2)) / factorial n.toNat = 5 * k :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_multiple_of_five_l1332_133256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_EF_l1332_133221

-- Define the triangle ABC
structure RightTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  right_angle : A.1 = 0 ∧ A.2 = 0 ∧ B.2 = 0 -- A is at origin, B on x-axis
  ab_length : (B.1 - A.1)^2 + (B.2 - A.2)^2 = 1
  ac_length : (C.1 - A.1)^2 + (C.2 - A.2)^2 = 4

-- Define the perpendicular line ℓ
noncomputable def perpendicular_line (t : RightTriangle) : ℝ → ℝ :=
  λ x => (1 / (t.C.2 / t.C.1)) * x

-- Define points E and F
noncomputable def point_E (t : RightTriangle) : ℝ × ℝ :=
  (1/2, perpendicular_line t (1/2))

noncomputable def point_F (t : RightTriangle) : ℝ × ℝ :=
  (2 * t.C.2 / t.C.1, t.C.2)

-- Theorem statement
theorem length_EF (t : RightTriangle) :
  let E := point_E t
  let F := point_F t
  ((F.1 - E.1)^2 + (F.2 - E.2)^2) = (3 * Real.sqrt 5 / 4)^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_EF_l1332_133221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_roots_l1332_133203

/-- The quadratic function f(x) = x^2 - ax + a - 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x + a - 2

/-- The distance between roots of f(x) -/
noncomputable def distance_between_roots (a : ℝ) : ℝ := 
  Real.sqrt ((a - 2)^2 + 4)

theorem min_distance_between_roots :
  ∀ a : ℝ, (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0) →
  ∃ a₀ : ℝ, ∀ a : ℝ, distance_between_roots a ≥ distance_between_roots a₀ ∧ 
  distance_between_roots a₀ = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_roots_l1332_133203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_theta_value_l1332_133231

theorem cos_theta_value (θ : ℝ) (h1 : θ ∈ Set.Ioo (π/2) π) (h2 : Real.sin θ = 1/3) : 
  Real.cos θ = -(2 * Real.sqrt 2)/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_theta_value_l1332_133231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_events_A_C_mutually_exclusive_l1332_133205

-- Define the sample space
variable (Ω : Type)
-- Define the probability measure
variable (P : Set Ω → ℝ)

-- Define the events
def A (Ω : Type) : Set Ω := {ω : Ω | true} -- Placeholder for "all three items are non-defective"
def B (Ω : Type) : Set Ω := {ω : Ω | false} -- Placeholder for "all three items are defective"
def C (Ω : Type) : Set Ω := {ω : Ω | true} -- Placeholder for "at least one of the three items is defective"

-- Theorem to prove
theorem events_A_C_mutually_exclusive (Ω : Type) : A Ω ∩ C Ω = ∅ := by
  sorry

-- Note: The actual implementations of A, B, and C would require more detailed definitions,
-- which are not provided in the original problem. The given definitions are placeholders.

end NUMINAMATH_CALUDE_ERRORFEEDBACK_events_A_C_mutually_exclusive_l1332_133205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_ratio_is_one_l1332_133238

/-- A square with side length 2 -/
structure Square where
  side : ℝ
  is_two : side = 2

/-- A circle outside the square -/
structure CircleOutside (s : Square) where
  radius : ℝ
  tangent_to_side : Bool
  tangent_to_extensions : Bool

/-- The area of a circle -/
noncomputable def circle_area (c : CircleOutside s) : ℝ := Real.pi * c.radius^2

/-- Theorem: The ratio of areas of two circles outside a square is 1 -/
theorem circle_area_ratio_is_one (s : Square) 
  (c1 c2 : CircleOutside s) 
  (h1 : c1.tangent_to_side = true) 
  (h2 : c2.tangent_to_side = true) 
  (h3 : c1.tangent_to_extensions = true) 
  (h4 : c2.tangent_to_extensions = true) : 
  circle_area c1 / circle_area c2 = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_ratio_is_one_l1332_133238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cannot_reach_top_floor_l1332_133214

/-- Represents a person living in a building --/
structure Resident where
  height : ℝ
  floor : ℕ
  can_reach : ℕ → Prop

/-- Represents an elevator in a building --/
structure Elevator where
  top_floor : ℕ
  button_height : ℝ

/-- The building where the resident lives --/
def building : Elevator :=
  { top_floor := 30
  , button_height := 1.5 }

/-- The resident in question --/
def person : Resident :=
  { height := 1.4
  , floor := 25
  , can_reach := λ n => 1.4 ≥ 1.5 * (n : ℝ) / 30 }

/-- Theorem stating that the person cannot reach the 25th-floor button --/
theorem cannot_reach_top_floor :
  ¬(person.can_reach person.floor) ∧ (person.can_reach (person.floor - 1)) := by
  sorry

#check cannot_reach_top_floor

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cannot_reach_top_floor_l1332_133214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_property_l1332_133223

-- Define odd function
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define F function
def F (f g : ℝ → ℝ) (x : ℝ) : ℝ := f x + 3 * g x + 5

-- Main theorem
theorem odd_function_property (f g : ℝ → ℝ) (a b : ℝ) 
  (hf : IsOdd f) (hg : IsOdd g) (hFa : F f g a = b) : 
  F f g (-a) = -b + 10 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_property_l1332_133223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1332_133237

noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - (1/2) * x^2 + 1

theorem f_properties :
  (∀ x : ℝ, f x ≤ 1) ∧
  (∀ x : ℝ, f x ≥ 5/6) ∧
  (f 0 = 1) ∧
  (f 1 = 5/6) ∧
  ((∀ x : ℝ, f x = 1 → x = 0 ∨ x = 3/2) ∨
   (∀ x : ℝ, f x = (3/4) * x - 1/8 → x = 3/2)) ∧
  (∫ x in (0 : ℝ)..(3/2 : ℝ), (1 - f x) = 9/64) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1332_133237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transaction_violates_legislation_l1332_133243

/-- Represents a resident of Russia -/
structure RussianResident where
  name : String

/-- Represents a currency -/
inductive Currency
| Ruble
| Euro

/-- Represents a transaction between two parties -/
structure Transaction where
  party1 : RussianResident
  party2 : RussianResident
  amount : ℕ
  currency : Currency

/-- Predicate to indicate an illegal transaction -/
def IllegalTransaction (t : Transaction) : Prop := sorry

/-- Predicate to indicate that a currency is not legal tender -/
def NotLegalTender (c : Currency) : Prop := sorry

/-- Represents the Federal Law 173-FZ -/
axiom federal_law_173_FZ : 
  ∀ (t : Transaction), t.currency ≠ Currency.Ruble → IllegalTransaction t

/-- The Russian ruble is the only legal tender in Russia -/
axiom ruble_only_legal_tender :
  ∀ (t : Transaction), t.currency ≠ Currency.Ruble → NotLegalTender t.currency

/-- Theorem: A transaction between Russian residents in euros violates Russian legislation -/
theorem transaction_violates_legislation 
  (mikhail valentin : RussianResident)
  (h : Transaction) 
  (h_parties : h.party1 = mikhail ∧ h.party2 = valentin)
  (h_currency : h.currency = Currency.Euro) :
  IllegalTransaction h ∧ NotLegalTender h.currency := by
  constructor
  · apply federal_law_173_FZ h
    rw [h_currency]
    simp
  · apply ruble_only_legal_tender h
    rw [h_currency]
    simp


end NUMINAMATH_CALUDE_ERRORFEEDBACK_transaction_violates_legislation_l1332_133243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_U_sum_property_l1332_133216

def U : ℕ → ℤ
  | 0 => 0  -- Add a case for 0
  | 1 => 1
  | k + 1 => U k + (k + 1)

theorem U_sum_property (n : ℕ) : U n + U (n + 1) = (n + 1)^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_U_sum_property_l1332_133216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exist_distinct_prime_divisors_l1332_133289

/-- Given positive integers m and n where m > n^(n-1) and all numbers m+1, m+2, ..., m+n are composite,
    there exist distinct primes p₁, p₂, ..., pₙ such that pₖ divides m+k for k = 1, 2, ..., n. -/
theorem exist_distinct_prime_divisors (m n : ℕ) 
  (h_m_pos : m > 0)
  (h_n_pos : n > 0)
  (h_m_gt_n_pow : m > n^(n-1))
  (h_composite : ∀ k ∈ Finset.range n, ¬Nat.Prime (m + k + 1)) :
  ∃ p : Fin n → ℕ, 
    (∀ i : Fin n, Nat.Prime (p i)) ∧ 
    (∀ i j : Fin n, i ≠ j → p i ≠ p j) ∧
    (∀ i : Fin n, (p i) ∣ (m + i.val + 1)) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exist_distinct_prime_divisors_l1332_133289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_characterization_l1332_133285

/-- A function satisfying the given conditions -/
structure SpecialFunction where
  f : ℝ → ℝ
  diff : Differentiable ℝ f
  domain : ∀ x, x > 0 → f x ≠ 0
  condition : ∀ x, x > 0 → x * (deriv (deriv f) x) + 2 * f x > 0

/-- The solution set of the inequality -/
def SolutionSet (f : SpecialFunction) : Set ℝ :=
  {x | (x + 2017) * f.f (x + 2017) / 5 < 5 * f.f 5 / (x + 2017)}

/-- The main theorem -/
theorem solution_set_characterization (f : SpecialFunction) :
  SolutionSet f = {x | -2017 < x ∧ x < -2012} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_characterization_l1332_133285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_RXYZ_approx_l1332_133276

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a right prism with triangular bases -/
structure RightPrism where
  height : ℝ
  baseLength : ℝ
  baseWidth : ℝ

/-- Represents a slice of the prism -/
structure PrismSlice where
  prism : RightPrism
  x : ℝ  -- Relative position of X on PQ
  y : ℝ  -- Relative position of Y on QR
  z : ℝ  -- Relative position of Z on RV

/-- Calculates the surface area of the sliced part RXYZ -/
noncomputable def surfaceAreaRXYZ (slice : PrismSlice) : ℝ :=
  sorry

/-- The theorem stating the surface area of RXYZ is approximately 120.39 -/
theorem surface_area_RXYZ_approx (slice : PrismSlice) 
  (h1 : slice.prism.height = 20)
  (h2 : slice.prism.baseLength = 15)
  (h3 : slice.prism.baseWidth = 20)
  (h4 : slice.x = 3/5)
  (h5 : slice.y = 1/4)
  (h6 : slice.z = 2/3)  -- Note: 2/3 because ZV = 1/3 * RV, so RZ = 2/3 * RV
  : ∃ (ε : ℝ), abs (surfaceAreaRXYZ slice - 120.39) < ε ∧ ε > 0 := by
  sorry

#check surface_area_RXYZ_approx

end NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_RXYZ_approx_l1332_133276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_given_log_constraint_l1332_133204

theorem min_sum_given_log_constraint (x y : ℝ) (h : Real.log x + Real.log y = Real.log 2) 
  (hx : x > 0) (hy : y > 0) : 
  ∃ (min : ℝ), (∀ a b : ℝ, a > 0 → b > 0 → Real.log a + Real.log b = Real.log 2 → a + b ≥ min) ∧ 
  min = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_given_log_constraint_l1332_133204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_no_advice_theorem_l1332_133229

/-- The expected number of explorers who don't receive advice -/
noncomputable def expected_no_advice (n : ℕ) (p : ℝ) : ℝ :=
  if p = 1 then 1 else (1 - (1 - p) ^ n) / p

/-- Theorem: The expected number of explorers who don't receive advice -/
theorem expected_no_advice_theorem (n : ℕ) (p : ℝ) 
  (h1 : n > 0) (h2 : 0 ≤ p ∧ p ≤ 1) :
  expected_no_advice n p = 
    if p = 1 then 1 else (1 - (1 - p) ^ n) / p :=
by
  -- The proof is skipped for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_no_advice_theorem_l1332_133229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1332_133292

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 1 then x else a * x^2 + 2 * x

-- Define the range of f
def is_in_range (y : ℝ) (a : ℝ) : Prop :=
  ∃ x, f a x = y

-- The theorem statement
theorem range_of_a :
  ∀ a : ℝ, (∀ y : ℝ, is_in_range y a) ↔ a ∈ Set.Icc (-1) 0 :=
by
  sorry

#check range_of_a

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1332_133292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_curve_arc_length_l1332_133207

noncomputable def arcLength (ρ : Real → Real) (φ₀ φ₁ : Real) : Real :=
  ∫ x in φ₀..φ₁, Real.sqrt ((ρ x)^2 + (deriv ρ x)^2)

theorem polar_curve_arc_length :
  let ρ : Real → Real := fun φ ↦ 3 * φ
  let φ₀ : Real := 0
  let φ₁ : Real := 4/3
  arcLength ρ φ₀ φ₁ = 10/3 + (3/2) * Real.log 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_curve_arc_length_l1332_133207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_man_rowing_distance_l1332_133251

/-- Calculates the total distance traveled by a man rowing in a river -/
noncomputable def total_distance_traveled (man_speed : ℝ) (river_speed : ℝ) (round_trip_time : ℝ) : ℝ :=
  let upstream_speed := man_speed - river_speed
  let downstream_speed := man_speed + river_speed
  let one_way_distance := (upstream_speed * downstream_speed * round_trip_time) / (2 * (upstream_speed + downstream_speed))
  2 * one_way_distance

/-- Theorem stating the total distance traveled by the man -/
theorem man_rowing_distance :
  let man_speed : ℝ := 10
  let river_speed : ℝ := 1.2
  let round_trip_time : ℝ := 1
  abs (total_distance_traveled man_speed river_speed round_trip_time - 9.856) < 0.001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_man_rowing_distance_l1332_133251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_y_l1332_133228

-- Define the function y as noncomputable
noncomputable def y (x : ℝ) : ℝ := Real.cos (2 * x^2 + x)

-- State the theorem
theorem derivative_of_y (x : ℝ) : 
  deriv y x = -(4 * x + 1) * Real.sin (2 * x^2 + x) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_y_l1332_133228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l1332_133275

theorem trigonometric_identities (x : Real) 
  (h1 : Real.sin x + Real.cos x = 1/5) 
  (h2 : 0 < x) 
  (h3 : x < Real.pi) : 
  (Real.sin (2*x) = -24/25) ∧ 
  (Real.sin x - Real.cos x = 7/5) ∧ 
  (Real.sin x^3 - Real.cos x^3 = 91/125) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l1332_133275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1332_133246

-- Define the ellipse equation
def is_ellipse (m : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (12 - m) + y^2 / (m - 4) = 1 ∧ m ≠ 8 ∧ 4 < m ∧ m < 12

-- Define when foci are on y-axis
def foci_on_y_axis (m : ℝ) : Prop :=
  m - 4 > 12 - m

-- Define focal length
noncomputable def focal_length (m : ℝ) : ℝ :=
  2 * Real.sqrt (12 - m - (m - 4))

-- Theorem statements
theorem ellipse_properties :
  (∀ m : ℝ, is_ellipse m → (foci_on_y_axis m → 8 < m ∧ m < 12)) ∧
  (is_ellipse 6 → focal_length 6 = 4) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1332_133246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_bounds_inequality_solution_a_inequality_solution_b_l1332_133200

-- Define the curve C
def C (x y : ℝ) : Prop := (4 * x^2) / 9 + y^2 / 16 = 1

-- Define the line l
def l (x y t : ℝ) : Prop := x = 3 + t ∧ y = 5 - 2*t

-- Define the distance function
noncomputable def distance (x y : ℝ) : ℝ := |3*x + 4*y - 11| / Real.sqrt 5

-- Theorem statement
theorem distance_bounds :
  ∀ x y : ℝ, C x y →
  (6 : ℝ) / 5 ≤ distance x y ∧ distance x y ≤ (16 : ℝ) / 5 :=
by
  sorry

-- Part 2(a)
def f (x a : ℝ) : ℝ := 2 * |x + 1| - a
def g (x : ℝ) : ℝ := |x|

theorem inequality_solution_a (x : ℝ) :
  f x 0 ≥ g x ↔ x ≤ -2 ∨ x ≥ -(2/3) :=
by
  sorry

-- Part 2(b)
theorem inequality_solution_b :
  (∃ x, f x a ≥ 2 * g x) → a ≤ 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_bounds_inequality_solution_a_inequality_solution_b_l1332_133200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_half_minus_two_alpha_l1332_133241

theorem cos_pi_half_minus_two_alpha (α : Real) (h1 : 0 < α ∧ α < π / 2) (h2 : Real.tan α = 2) :
  Real.cos (π / 2 - 2 * α) = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_half_minus_two_alpha_l1332_133241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_and_tangency_l1332_133272

/-- Circle M with center (-cos q, sin q) and radius 1 -/
def circleM (q : ℝ) (x y : ℝ) : Prop :=
  (x + Real.cos q)^2 + (y - Real.sin q)^2 = 1

/-- Line l with slope k -/
def lineL (k : ℝ) (x y : ℝ) : Prop :=
  y = k * x

/-- Distance from point (h, j) to line y = kx -/
noncomputable def distance_to_line (k h j : ℝ) : ℝ :=
  |j - k * h| / Real.sqrt (1 + k^2)

theorem circle_line_intersection_and_tangency :
  (∀ k q : ℝ, ∃ x y : ℝ, circleM q x y ∧ lineL k x y) ∧
  (∀ k : ℝ, ∃ q : ℝ, distance_to_line k (-Real.cos q) (Real.sin q) = 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_and_tangency_l1332_133272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_properties_l1332_133288

noncomputable def f (ω φ x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem sine_function_properties (ω φ : ℝ) 
  (h_ω_pos : ω > 0) 
  (h_φ_bound : |φ| < π / 2) 
  (h_period : ∀ x, f ω φ (x + π) = f ω φ x) 
  (h_symmetry : ∀ x, f ω φ (x + π / 3) = f ω φ (-x + π / 3)) :
  ω = 2 ∧ φ = -π / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_properties_l1332_133288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_l1332_133284

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := Real.cos (ω * x + φ)

theorem min_omega (ω φ : ℝ) (h1 : ω > 0) (h2 : 0 < φ) (h3 : φ < π) :
  let T := 2 * π / ω
  (f ω φ T = 1/2) →
  (∃ k : ℤ, ω * (7 * π / 3) + φ = k * π) →
  (∀ ω' > 0, (∃ k' : ℤ, ω' * (7 * π / 3) + φ = k' * π) → ω' ≥ 2/7) →
  ω = 2/7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_l1332_133284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_value_at_one_l1332_133283

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a - 2 / (2^x + 1)

-- State the theorem
theorem odd_function_value_at_one (a : ℝ) :
  (∀ x, f a x = -f a (-x)) → f a 1 = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_value_at_one_l1332_133283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_b_value_l1332_133280

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_positive_a : a > 0
  h_positive_b : b > 0

/-- The eccentricity of the hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := Real.sqrt 5

/-- The area of the quadrilateral formed by the intersection of asymptotes 
    with the circle having foci as diameter -/
def quadrilateral_area : ℝ := 4

/-- Theorem: For a hyperbola with eccentricity √5 and quadrilateral area 4, b = 2 -/
theorem hyperbola_b_value (h : Hyperbola) 
  (h_eccentricity : eccentricity h = Real.sqrt 5)
  (h_area : quadrilateral_area = 4) : h.b = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_b_value_l1332_133280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_theorem_l1332_133244

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 3 + y^2 = 1

-- Define the line
def line (x y m : ℝ) : Prop := y = x + m

-- Define the foci
noncomputable def left_focus : ℝ × ℝ := (-Real.sqrt 2, 0)
noncomputable def right_focus : ℝ × ℝ := (Real.sqrt 2, 0)

-- Define the intersection points
def intersection_points (m : ℝ) : Prop :=
  ∃ (A B : ℝ × ℝ), ellipse A.1 A.2 ∧ ellipse B.1 B.2 ∧ 
    line A.1 A.2 m ∧ line B.1 B.2 m

-- Define the area ratio condition
def area_ratio_condition (A B : ℝ × ℝ) : Prop :=
  ∃ (area_triangle : (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ) → ℝ),
    area_triangle left_focus A B = 2 * area_triangle right_focus A B

theorem ellipse_intersection_theorem (m : ℝ) :
  intersection_points m →
  (∃ (A B : ℝ × ℝ), area_ratio_condition A B) →
  m = -Real.sqrt 2 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_theorem_l1332_133244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_light_path_length_in_cube_l1332_133265

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculates the distance between two points in 3D space -/
noncomputable def distance (p1 p2 : Point3D) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2)

/-- Theorem: Light path length in a cube -/
theorem light_path_length_in_cube (cube_side : ℝ) (reflection_point : Point3D) : 
  cube_side = 10 →
  reflection_point = ⟨3, 1, 10⟩ →
  ∃ (n : ℕ), (n : ℝ) * distance ⟨0, 0, 0⟩ reflection_point = 10 * Real.sqrt 110 ∧ 
             (n : ℝ) * reflection_point.x % cube_side = 0 ∧
             (n : ℝ) * reflection_point.y % cube_side = 0 ∧
             (n : ℝ) * reflection_point.z % cube_side = 0 :=
by sorry

#check light_path_length_in_cube

end NUMINAMATH_CALUDE_ERRORFEEDBACK_light_path_length_in_cube_l1332_133265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l1332_133253

theorem equation_solution :
  ∃! x : ℝ, (9 : ℝ)^(-x) - 2 * (3 : ℝ)^(1-x) = 27 :=
by
  use -2
  constructor
  · simp
    -- Proof steps would go here
    sorry
  · intro y h
    -- Uniqueness proof would go here
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l1332_133253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_definite_integral_exp_plus_2x_l1332_133248

theorem definite_integral_exp_plus_2x : ∫ x in (0:ℝ)..(1:ℝ), (Real.exp x + 2*x) = Real.exp 1 - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_definite_integral_exp_plus_2x_l1332_133248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_X_star_Y_equals_l1332_133262

def set_diff (A B : Set Int) : Set Int := { x | x ∈ A ∧ x ∉ B }

def set_star (A B : Set Int) : Set Int := (set_diff A B) ∪ (set_diff B A)

def X : Set Int := {1, 3, 5, 7}

def Y : Set Int := { x : Int | x < 4 }

theorem X_star_Y_equals : set_star X Y = {-3, -2, -1, 0, 2, 5, 7} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_X_star_Y_equals_l1332_133262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_solution_l1332_133266

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def IsPerp (m₁ m₂ : ℝ) : Prop := m₁ * m₂ = -1

axiom perpendicular_lines (m₁ m₂ : ℝ) : IsPerp m₁ m₂ ↔ m₁ * m₂ = -1

/-- Definition of line l₁ -/
def l₁ (m : ℝ) : ℝ → ℝ → Prop :=
  fun x y => (m + 3) * x + (m - 1) * y - 5 = 0

/-- Definition of line l₂ -/
def l₂ (m : ℝ) : ℝ → ℝ → Prop :=
  fun x y => (m - 1) * x + (3 * m + 9) * y - 1 = 0

/-- Slope of line l₁ -/
noncomputable def slope_l₁ (m : ℝ) : ℝ := -(m + 3) / (m - 1)

/-- Slope of line l₂ -/
noncomputable def slope_l₂ (m : ℝ) : ℝ := -(m - 1) / (3 * m + 9)

theorem perpendicular_lines_solution (m : ℝ) :
  IsPerp (slope_l₁ m) (slope_l₂ m) → m = 1 ∨ m = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_solution_l1332_133266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1332_133245

noncomputable section

def Hyperbola (a b : ℝ) := {p : ℝ × ℝ | (p.1^2 / a^2) - (p.2^2 / b^2) = 1}

def RightFocus : ℝ × ℝ := (2, 0)

def SymmetricPoints (A B : ℝ × ℝ) := B = (-A.1, -A.2)

def Eccentricity (a c : ℝ) := c / a

theorem hyperbola_eccentricity 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (A B : ℝ × ℝ) 
  (hAB : SymmetricPoints A B) 
  (hHyp : A ∈ Hyperbola a b ∧ B ∈ Hyperbola a b) 
  (hDotProduct : (A.1 - 2) * (-A.1 - 2) - A.2^2 = 0) 
  (hSlope : A.2 / A.1 = Real.sqrt 3) :
  Eccentricity a 2 = Real.sqrt 3 + 1 := by
  sorry

#check hyperbola_eccentricity

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1332_133245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_fill_time_l1332_133242

/-- Represents the time in minutes to fill the tank completely -/
noncomputable def fill_time (fill_rate : ℝ) (empty_rate : ℝ) (initial_fill : ℝ) : ℝ :=
  (1 - initial_fill) / (fill_rate - empty_rate)

/-- Theorem stating the time to fill the tank under given conditions -/
theorem tank_fill_time :
  let fill_rate : ℝ := 1 / 25
  let empty_rate : ℝ := 1 / 50
  let initial_fill : ℝ := 1 / 2
  fill_time fill_rate empty_rate initial_fill = 25 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_fill_time_l1332_133242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_line_with_two_rational_points_l1332_133219

-- Define a rational point as a pair of rational numbers
def RationalPoint := ℚ × ℚ

-- Define a line passing through (a, 0) and another point
def Line (a : ℝ) (p : RationalPoint) :=
  {(x, y) : ℝ × ℝ | ∃ (t : ℝ), x = a + t * (p.1 - a) ∧ y = t * p.2}

-- Define a coercion from RationalPoint to ℝ × ℝ
instance : Coe RationalPoint (ℝ × ℝ) :=
  ⟨fun p => (p.1, p.2)⟩

theorem unique_line_with_two_rational_points (a : ℝ) (h : Irrational a) :
  ∃! (l : Set (ℝ × ℝ)), ∃ (p q : RationalPoint), p ≠ q ∧ (p : ℝ × ℝ) ∈ l ∧ (q : ℝ × ℝ) ∈ l ∧
    ∃ (r : RationalPoint), l = Line a r :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_line_with_two_rational_points_l1332_133219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ali_remaining_money_l1332_133249

noncomputable def initial_amount : ℚ := 480

noncomputable def food_expense : ℚ := initial_amount / 2

noncomputable def remaining_after_food : ℚ := initial_amount - food_expense

noncomputable def glasses_expense : ℚ := remaining_after_food / 3

noncomputable def final_amount : ℚ := remaining_after_food - glasses_expense

theorem ali_remaining_money :
  final_amount = 160 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ali_remaining_money_l1332_133249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixty_fifth_term_value_l1332_133212

def next_term (n : ℕ) : ℕ :=
  if n < 10 then n * 9
  else if n % 2 = 0 then n / 2
  else n * 2 - 1

def sequence_term (start : ℕ) : ℕ → ℕ
  | 0 => start
  | n + 1 => next_term (sequence_term start n)

theorem sixty_fifth_term_value (n : ℕ) : sequence_term 87 64 = n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixty_fifth_term_value_l1332_133212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_difference_l1332_133236

noncomputable section

/-- Line C1 with parametric equation x = 1 + (1/2)t, y = (√3/2)t -/
def C1 (t : ℝ) : ℝ × ℝ := (1 + (1/2) * t, (Real.sqrt 3 / 2) * t)

/-- Curve C2 with Cartesian equation x²/3 + y² = 1 -/
def C2 (x y : ℝ) : Prop := x^2 / 3 + y^2 = 1

/-- Point M -/
def M : ℝ × ℝ := (1, 0)

/-- A and B are intersection points of C1 and C2 -/
def intersection_points (A B : ℝ × ℝ) : Prop :=
  ∃ t1 t2 : ℝ, C1 t1 = A ∧ C1 t2 = B ∧ C2 A.1 A.2 ∧ C2 B.1 B.2

/-- Main theorem -/
theorem intersection_distance_difference (A B : ℝ × ℝ) 
  (h : intersection_points A B) : 
  |dist M A - dist M B| = 2/5 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_difference_l1332_133236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_point_line_forming_triangle_l1332_133297

-- Define a line type
structure Line where
  slope : ℚ
  yIntercept : ℚ

-- Define a point type
structure Point where
  x : ℚ
  y : ℚ

-- Define a function to check if a point is on a line
def pointOnLine (l : Line) (p : Point) : Prop :=
  p.y = l.slope * p.x + l.yIntercept

-- Define a function to calculate the area of a triangle formed by a line and coordinate axes
def triangleArea (l : Line) : ℚ :=
  (l.yIntercept * l.yIntercept) / 2

-- Theorem for part (1)
theorem line_through_point (l : Line) (p : Point) :
  l.slope = -1 ∧ p = ⟨2, 2⟩ ∧ pointOnLine l p →
  l.yIntercept = 4 := by
  sorry

-- Theorem for part (2)
theorem line_forming_triangle (l : Line) :
  l.slope = -1 ∧ triangleArea l = 12 →
  l.yIntercept = 2 * (12 : ℚ).sqrt ∨ l.yIntercept = -2 * (12 : ℚ).sqrt := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_point_line_forming_triangle_l1332_133297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_cartesian_l1332_133213

/-- If the terminal side of angle α passes through point P(1, -2) in the Cartesian coordinate system, then sin(2α) = -4/5 -/
theorem sin_double_angle_cartesian (α : ℝ) : 
  (∃ (r : ℝ), r > 0 ∧ r * Real.cos α = 1 ∧ r * Real.sin α = -2) → 
  Real.sin (2 * α) = -4/5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_cartesian_l1332_133213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_f1_domain_f2_l1332_133206

-- Function 1
noncomputable def domain1 (x : ℝ) : Prop :=
  x > -2 ∧ x ≠ 1

noncomputable def f1 (x : ℝ) : ℝ := 
  (x - 1)^0 / Real.sqrt (x + 2)

theorem domain_f1 : 
  ∀ x : ℝ, (∃ y : ℝ, f1 x = y) ↔ domain1 x :=
by
  sorry

-- Function 2
noncomputable def domain2 (x : ℝ) : Prop :=
  x < 0

noncomputable def f2 (x : ℝ) : ℝ :=
  (x + 2) / Real.sqrt (|x| - x)

theorem domain_f2 :
  ∀ x : ℝ, (∃ y : ℝ, f2 x = y) ↔ domain2 x :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_f1_domain_f2_l1332_133206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_rectangular_example_l1332_133254

noncomputable def polar_to_rectangular (r : ℝ) (θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

theorem polar_to_rectangular_example :
  polar_to_rectangular 5 (5 * π / 4) = (-5 * Real.sqrt 2 / 2, -5 * Real.sqrt 2 / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_rectangular_example_l1332_133254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_equation_of_line_l1332_133230

theorem polar_equation_of_line :
  let line_through_pole : Set (ℝ × ℝ) := {(r, φ) | φ = π/3 ∨ φ = 4*π/3 ∧ r ≥ 0}
  let inclination_angle : ℝ := π/3
  ∀ (r φ : ℝ), (r, φ) ∈ line_through_pole ↔ 
    (φ = inclination_angle ∨ φ = inclination_angle + π) ∧ r ≥ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_equation_of_line_l1332_133230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_pyramid_volume_l1332_133215

/-- The volume of a pyramid with a square base and two adjacent triangular faces --/
noncomputable def pyramidVolume (baseArea : ℝ) (face1Area : ℝ) (face2Area : ℝ) : ℝ :=
  let s := Real.sqrt baseArea
  let h1 := 2 * face1Area / s
  let h2 := 2 * face2Area / s
  let a := (s^2 + h2^2 - h1^2) / (2 * s)
  let h := Real.sqrt (h1^2 - a^2)
  (1 / 3) * baseArea * h

/-- The theorem stating the volume of the specific pyramid --/
theorem specific_pyramid_volume :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |pyramidVolume 256 120 108 - 1156.61| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_pyramid_volume_l1332_133215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_inscribed_circle_radius_is_2_sqrt_10_l1332_133209

/-- A quadrilateral with side lengths 15, 10, 8, and 13 -/
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)
  (AB_length : dist A B = 15)
  (BC_length : dist B C = 10)
  (CD_length : dist C D = 8)
  (DA_length : dist D A = 13)

/-- The radius of the largest inscribed circle in the quadrilateral -/
noncomputable def largest_inscribed_circle_radius (q : Quadrilateral) : ℝ := 2 * Real.sqrt 10

/-- Theorem stating that the largest inscribed circle has radius 2√10 -/
theorem largest_inscribed_circle_radius_is_2_sqrt_10 (q : Quadrilateral) :
  largest_inscribed_circle_radius q = 2 * Real.sqrt 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_inscribed_circle_radius_is_2_sqrt_10_l1332_133209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interval_of_c_over_a_l1332_133270

theorem interval_of_c_over_a (a b c : ℝ) 
  (h1 : b / a ∈ Set.Ioo (-0.9) (-0.8)) 
  (h2 : b / c ∈ Set.Ioo (-0.9) (-0.8)) : 
  c / a ∈ Set.Ioo (8/9) (9/8) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interval_of_c_over_a_l1332_133270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l1332_133227

-- Define set M
def M : Set ℝ := {x : ℝ | -1 < x ∧ x < 1}

-- Define set N
def N : Set ℤ := {x : ℤ | x^2 < 2}

-- Embed N into ℝ
def N_real : Set ℝ := {x : ℝ | ∃ n : ℤ, n ∈ N ∧ x = n}

-- Theorem statement
theorem intersection_M_N : M ∩ N_real = {0} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l1332_133227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blair_17th_turn_l1332_133294

def blairSequence : ℕ → ℕ
  | 0 => 5
  | n + 1 => if n % 2 = 0 then blairSequence n + 2 else blairSequence n + 1

theorem blair_17th_turn : blairSequence 32 = 55 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_blair_17th_turn_l1332_133294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_volume_l1332_133296

/-- Calculates the volume of a square pyramid --/
noncomputable def pyramid_volume (base_edge : ℝ) (height : ℝ) : ℝ :=
  (1 / 3) * base_edge^2 * height

/-- Represents the dimensions of a square pyramid --/
structure SquarePyramid where
  base_edge : ℝ
  altitude : ℝ

/-- Represents the dimensions of a frustum formed by cutting a square pyramid --/
structure Frustum where
  original : SquarePyramid
  smaller : SquarePyramid

/-- Theorem stating the volume of the frustum --/
theorem frustum_volume (f : Frustum) 
  (h1 : f.original.base_edge = 15)
  (h2 : f.original.altitude = 10)
  (h3 : f.smaller.base_edge = 9)
  (h4 : f.smaller.altitude = 6) :
  pyramid_volume f.original.base_edge f.original.altitude - 
  pyramid_volume f.smaller.base_edge f.smaller.altitude = 588 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_volume_l1332_133296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_25_property_l1332_133210

/-- A function from nonnegative integers to nonnegative integers satisfying the given property -/
def FunctionF (f : ℕ → ℕ) : Prop :=
  ∀ a b : ℕ, 2 * f (a^2 + b^2) = (f a)^2 + (f b)^2

/-- The set of possible values for f(25) -/
def PossibleValues (f : ℕ → ℕ) : Set ℕ :=
  {x : ℕ | ∃ (g : ℕ → ℕ), FunctionF g ∧ g 25 = x}

/-- The theorem to be proved -/
theorem f_25_property :
  ∃ (S : Finset ℕ), S.card * S.sum id = 153 ∧
    ∀ f : ℕ → ℕ, FunctionF f → (f 25) ∈ S := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_25_property_l1332_133210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_annika_hike_distance_l1332_133235

/-- Represents the hiking scenario for Annika --/
structure HikingScenario where
  rate : ℝ  -- hiking rate in minutes per kilometer
  initialDistance : ℝ  -- initial distance hiked in kilometers
  totalTime : ℝ  -- total time available in minutes

/-- Calculates the total distance hiked east given a hiking scenario --/
noncomputable def totalDistanceEast (scenario : HikingScenario) : ℝ :=
  scenario.initialDistance + (scenario.totalTime - scenario.rate * scenario.initialDistance) / (2 * scenario.rate)

/-- Theorem stating that Annika will hike 3.5 kilometers east in total --/
theorem annika_hike_distance (scenario : HikingScenario) 
  (h1 : scenario.rate = 10)
  (h2 : scenario.initialDistance = 2.5)
  (h3 : scenario.totalTime = 35) :
  totalDistanceEast scenario = 3.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_annika_hike_distance_l1332_133235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_class_mean_score_l1332_133279

theorem class_mean_score (total_students : ℕ) (group1_students : ℕ) (group2_students : ℕ) 
  (group1_average : ℚ) (group2_average : ℚ) : 
  total_students = group1_students + group2_students →
  group1_students = 40 →
  group2_students = 10 →
  group1_average = 85/100 →
  group2_average = 75/100 →
  (group1_students * group1_average + group2_students * group2_average) / total_students = 83/100 := by
  sorry

-- Remove the #eval line as it's not necessary for the theorem proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_class_mean_score_l1332_133279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_leak_time_period_l1332_133273

/-- Represents the leak rate of the largest hole in ounces per minute -/
noncomputable def largest_hole_rate : ℝ := 3

/-- Represents the leak rate of the medium-sized hole in ounces per minute -/
noncomputable def medium_hole_rate : ℝ := largest_hole_rate / 2

/-- Represents the leak rate of the smallest hole in ounces per minute -/
noncomputable def smallest_hole_rate : ℝ := medium_hole_rate / 3

/-- Represents the combined leak rate of all three holes in ounces per minute -/
noncomputable def combined_rate : ℝ := largest_hole_rate + medium_hole_rate + smallest_hole_rate

/-- Represents the total amount of water leaked in ounces -/
def total_water : ℝ := 600

/-- Theorem stating that the time period for 600 ounces of water to leak is 120 minutes -/
theorem leak_time_period : total_water / combined_rate = 120 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_leak_time_period_l1332_133273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_sum_of_coordinates_l1332_133224

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  let (p, q) := t.A
  let (x₁, y₁) := t.B
  let (x₂, y₂) := t.C
  -- Area of triangle ABC is 70
  abs ((p * (y₁ - y₂) + x₁ * (y₂ - q) + x₂ * (q - y₁)) / 2) = 70
  ∧ -- Coordinates of B are (12, 19)
  t.B = (12, 19)
  ∧ -- Coordinates of C are (23, 20)
  t.C = (23, 20)
  ∧ -- The line containing the median to side BC has slope -5
  (q - (y₁ + y₂) / 2) / (p - (x₁ + x₂) / 2) = -5

-- Theorem statement
theorem largest_sum_of_coordinates (t : Triangle) :
  triangle_conditions t → (let (p, q) := t.A; p + q ≤ 47) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_sum_of_coordinates_l1332_133224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_students_only_in_science_l1332_133298

theorem students_only_in_science (total science drama : ℕ) 
  (h1 : total = 88) 
  (h2 : science = 75) 
  (h3 : drama = 65) 
  (h4 : ∀ s, s ≤ total → s ≤ science ∨ s ≤ drama) :
  science - (science + drama - total) = 23 := by
  sorry

#check students_only_in_science

end NUMINAMATH_CALUDE_ERRORFEEDBACK_students_only_in_science_l1332_133298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_sufficient_condition_l1332_133267

-- Define structures for Plane and Line
structure Plane where

structure Line where

-- Define predicates for perpendicular and parallel
def perpendicular (l : Line) (p : Plane) : Prop := sorry

def parallel (p1 p2 : Plane) : Prop := sorry

-- Declare variables
variable (α β γ : Plane)
variable (m n l : Line)

-- Axioms for distinctness
axiom planes_distinct : α ≠ β ∧ β ≠ γ ∧ α ≠ γ

axiom lines_distinct : m ≠ n ∧ n ≠ l ∧ m ≠ l

-- Theorem for the sufficient condition
theorem perpendicular_sufficient_condition :
  perpendicular n α → perpendicular n β → perpendicular m α → perpendicular m β :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_sufficient_condition_l1332_133267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_color_unit_distance_l1332_133257

-- Define a type for colors
inductive Color
  | White
  | Black

-- Define a point in a plane
structure Point where
  x : ℝ
  y : ℝ

-- Define the distance between two points
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

-- Define a coloring of the plane
def Coloring := Point → Color

-- The theorem to prove
theorem same_color_unit_distance (c : Coloring) :
  ∃ (p1 p2 : Point), c p1 = c p2 ∧ distance p1 p2 = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_color_unit_distance_l1332_133257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_equation_l1332_133282

/-- Given an ellipse and a hyperbola sharing a common focus, 
    prove that the asymptote of the hyperbola has a specific equation. -/
theorem hyperbola_asymptote_equation (m n : ℝ) (h : m > 0) (k : n > 0) :
  let ellipse := fun (x y : ℝ) ↦ x^2 / (3 * m^2) + y^2 / (5 * n^2) = 1
  let hyperbola := fun (x y : ℝ) ↦ x^2 / (2 * m^2) - y^2 / (3 * n^2) = 1
  let common_focus := ∃ (c : ℝ), c^2 = 3 * m^2 - 5 * n^2 ∧ c^2 = 2 * m^2 + 3 * n^2
  let asymptote := fun (x y : ℝ) ↦ y = (Real.sqrt 3 / 4) * x ∨ y = -(Real.sqrt 3 / 4) * x
  common_focus → (∀ x y, hyperbola x y → asymptote x y) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_equation_l1332_133282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_theorem_l1332_133281

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define a point on the parabola
def point_on_parabola (M : ℝ × ℝ) : Prop :=
  parabola M.1 M.2

-- Define the distance from a point to the line x = -3
def distance_to_line (M : ℝ × ℝ) : ℝ :=
  M.1 + 3

-- Define the distance between two points
noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- Theorem statement
theorem parabola_distance_theorem (M : ℝ × ℝ) :
  point_on_parabola M →
  distance_to_line M = 7 →
  distance M focus = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_theorem_l1332_133281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2013_pi_third_l1332_133278

open Real

/-- Recursive definition of the function sequence -/
noncomputable def f (n : ℕ) : ℝ → ℝ :=
  match n with
  | 0 => λ x => sin x - cos x
  | n + 1 => deriv (f n)

/-- The main theorem to prove -/
theorem f_2013_pi_third : f 2013 (π / 3) = (1 + Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2013_pi_third_l1332_133278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_to_plane_existence_of_non_transitive_skew_lines_parallel_line_to_plane_l1332_133295

-- Define the necessary structures
structure Line
structure Plane

-- Define the relationships between lines and planes
axiom perpendicular : Line → Plane → Prop
axiom perpendicular_line : Line → Line → Prop
axiom skew : Line → Line → Prop
axiom parallel : Plane → Plane → Prop
axiom parallel_line_plane : Line → Plane → Prop
axiom contained_in : Line → Plane → Prop

-- Theorem 1
theorem perpendicular_to_plane (l : Line) (p : Plane) (l1 l2 : Line) :
  contained_in l1 p → contained_in l2 p → perpendicular_line l l1 → perpendicular_line l l2 →
  perpendicular l p :=
sorry

-- Theorem 2
theorem existence_of_non_transitive_skew_lines :
  ∃ (m n l : Line), skew m n ∧ skew n l ∧ ¬skew m l :=
sorry

-- Theorem 3
theorem parallel_line_to_plane (m : Line) (α β : Plane) :
  parallel α β → contained_in m α → parallel_line_plane m β :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_to_plane_existence_of_non_transitive_skew_lines_parallel_line_to_plane_l1332_133295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_difference_l1332_133287

theorem percentage_difference (x y z : ℝ) (h1 : y = 1.6 * x) (h2 : z = 0.6 * y) :
  ∃ ε > 0, abs ((x - z) / z * 100 - 4.17) < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_difference_l1332_133287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plug_pairs_count_l1332_133218

/-- Prove that given the conditions in the problem, the number of pairs of Type X, Type Y, and Type Z plugs are 100, 50, and 50 respectively. -/
theorem plug_pairs_count (typeX_plugs typeY_plugs : ℕ) :
  let typeA_mittens := 100
  let typeB_mittens := 50
  let total_mittens := typeA_mittens + typeB_mittens
  let initial_total_plugs := total_mittens + 20
  let typeZ_plugs := typeA_mittens / 2
  typeX_plugs + typeY_plugs + typeZ_plugs = initial_total_plugs →
  typeX_plugs + 30 = 2 * typeY_plugs →
  (typeX_plugs + 30 = 100 ∧ typeY_plugs = 50 ∧ typeZ_plugs = 50) :=
by
  intro h1 h2
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_plug_pairs_count_l1332_133218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1332_133293

def my_sequence (a : ℕ → ℤ) : Prop :=
  a 1 = 0 ∧ a 2 = 2 ∧ ∀ n ≥ 2, a (n + 1) + a (n - 1) = 2 * (a n + 1)

theorem sequence_properties (a : ℕ → ℤ) (h : my_sequence a) :
  (∀ n : ℕ, n ≥ 1 → a (n + 1) - a n = 2 * n) ∧
  (∀ n : ℕ, n ≥ 1 → a n = n^2 - n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1332_133293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetry_l1332_133234

noncomputable def f (x : ℝ) : ℝ := Real.sin x * Real.cos x - (Real.sqrt 3 / 2) * Real.cos (2 * x)

noncomputable def symmetry_point : ℝ := 5 * Real.pi / 12

theorem f_symmetry : 
  ∀ (x : ℝ), f (symmetry_point + x) = f (symmetry_point - x) := by
  intro x
  -- Expand the definition of f
  simp [f, symmetry_point]
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetry_l1332_133234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_double_angle_inequality_in_acute_triangle_l1332_133222

theorem sine_double_angle_inequality_in_acute_triangle 
  (α β γ : ℝ) 
  (h_acute : 0 < α ∧ 0 < β ∧ 0 < γ)
  (h_triangle : α + β + γ = Real.pi)
  (h_order : α < β ∧ β < γ) : 
  Real.sin (2 * α) > Real.sin (2 * β) ∧ Real.sin (2 * β) > Real.sin (2 * γ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_double_angle_inequality_in_acute_triangle_l1332_133222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_range_l1332_133217

-- Define the line l: x - y + m = 0
def line (m : ℝ) (x y : ℝ) : Prop := x - y + m = 0

-- Define the circle C: x^2 + y^2 - 4x - 2y + 1 = 0
def circleC (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 2*y + 1 = 0

-- Theorem statement
theorem line_circle_intersection_range (m : ℝ) :
  (∀ x y : ℝ, line m x y → circleC x y) →
  m ∈ Set.Icc (-2 * Real.sqrt 2 - 1) (2 * Real.sqrt 2 - 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_range_l1332_133217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_multiple_of_4_l1332_133232

def is_divisible_by_4 (n : ℕ) : Bool := n % 4 = 0

def count_divisible_by_4 (f : ℕ → ℕ) (range : List ℕ) : ℕ :=
  (range.filter (λ x => is_divisible_by_4 (f x))).length

theorem three_digit_multiple_of_4 :
  count_divisible_by_4 (λ B => 320 + B) (List.range 10) = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_multiple_of_4_l1332_133232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_sin_2x_squared_plus_x_l1332_133299

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x^2 + x)

-- State the theorem
theorem derivative_of_sin_2x_squared_plus_x :
  deriv f = λ x => (4 * x + 1) * Real.cos (2 * x^2 + x) := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_sin_2x_squared_plus_x_l1332_133299
