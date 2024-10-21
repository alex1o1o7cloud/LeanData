import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transformed_cosine_l320_32083

-- Define the original function
noncomputable def f (x : ℝ) : ℝ := Real.cos x

-- Define the transformed function
noncomputable def g (x : ℝ) : ℝ := f (2 * (x + Real.pi/4))

-- Theorem statement
theorem transformed_cosine :
  ∀ x : ℝ, g x = Real.cos (2 * x + Real.pi/2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_transformed_cosine_l320_32083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_rectangle_area_relation_l320_32056

theorem square_rectangle_area_relation : 
  ∃ x₁ x₂ : ℝ, 
    (∀ x : ℝ, 
      let square_side : ℝ := x - 3
      let rect_length : ℝ := x - 5
      let rect_width : ℝ := x + 3
      let square_area : ℝ := square_side ^ 2
      let rect_area : ℝ := rect_length * rect_width
      rect_area = 3 * square_area →
      (x = x₁ ∨ x = x₂)) ∧
    x₁ * x₂ = 21 := by
  sorry

#check square_rectangle_area_relation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_rectangle_area_relation_l320_32056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_expression_equals_sqrt_two_l320_32005

theorem trig_expression_equals_sqrt_two :
  (Real.cos (-585 : ℝ)) / (Real.tan 495 + Real.sin (-690 : ℝ)) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_expression_equals_sqrt_two_l320_32005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_is_eight_cents_l320_32070

/-- Calculates the interest earned in cents -/
def interest_earned (principal : ℚ) (rate : ℚ) (time : ℚ) (final_amount : ℚ) : ℕ :=
  ((final_amount - principal) * 100).floor.toNat

/-- Theorem stating that the interest earned is 8 cents -/
theorem interest_is_eight_cents (principal : ℚ) :
  let rate : ℚ := 4 / 100  -- 4% annual rate
  let time : ℚ := 1 / 4    -- 3 months = 1/4 year
  let final_amount : ℚ := 310.62
  principal * (1 + rate * time) = final_amount →
  interest_earned principal rate time final_amount = 8 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_is_eight_cents_l320_32070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_intercept_of_perpendicular_line_l320_32045

-- Define the slope of the given line
noncomputable def m₁ : ℝ := 3 / 4

-- Define the perpendicular line
def perpendicular_line (k : ℝ) : ℝ → ℝ → Prop :=
  λ x y ↦ 4 * x + 3 * y + k = 0

-- Define the area of the triangle formed by the line and coordinate axes
noncomputable def triangle_area (k : ℝ) : ℝ :=
  k^2 / 24

-- The theorem to prove
theorem x_intercept_of_perpendicular_line :
  ∃ k : ℝ, triangle_area k = 6 ∧
  (perpendicular_line k (-k/4) 0 ∨ perpendicular_line k (k/4) 0) := by
  sorry

#check x_intercept_of_perpendicular_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_intercept_of_perpendicular_line_l320_32045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sampling_properties_l320_32066

/-- Represents a sampling method -/
inductive SamplingMethod
  | Systematic
  | Random
  | Stratified

/-- Represents a group with boys and girls -/
structure HobbyGroup where
  boys : Nat
  girls : Nat

/-- Represents a sample drawn from a group -/
structure Sample where
  size : Nat
  boys : Nat
  girls : Nat

/-- Checks if a sampling method is possible for given group and sample -/
def isPossibleSampling (g : HobbyGroup) (s : Sample) (m : SamplingMethod) : Prop :=
  match m with
  | SamplingMethod.Systematic => s.size ∣ (g.boys + g.girls)
  | SamplingMethod.Random => true
  | SamplingMethod.Stratified => s.boys * g.girls = s.girls * g.boys

/-- Calculates the probability of selection for boys and girls -/
def selectionProbability (g : HobbyGroup) (s : Sample) : (Rat × Rat) :=
  (s.boys / g.boys, s.girls / g.girls)

theorem sampling_properties (g : HobbyGroup) (s : Sample)
  (h_boys : g.boys = 20)
  (h_girls : g.girls = 10)
  (h_sample_size : s.size = 5)
  (h_sample_boys : s.boys = 2)
  (h_sample_girls : s.girls = 3) :
  isPossibleSampling g s SamplingMethod.Systematic ∧
  isPossibleSampling g s SamplingMethod.Random ∧
  ¬isPossibleSampling g s SamplingMethod.Stratified ∧
  (let (p_boys, p_girls) := selectionProbability g s
   p_boys ≠ p_girls) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sampling_properties_l320_32066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_amount_satisfies_conditions_principal_amount_approximation_l320_32082

/-- The principal amount that satisfies the given conditions -/
noncomputable def principal_amount : ℝ :=
  25 / ((1 + 0.05/2)^(2*3) - 1 - 0.05*3)

/-- Theorem stating that the principal amount satisfies the given conditions -/
theorem principal_amount_satisfies_conditions :
  let P := principal_amount
  let r := 0.05  -- 5% annual interest rate
  let t := 3     -- 3 years
  let n := 2     -- compounded semi-annually
  P * ((1 + r/n)^(n*t) - 1 - r*t) = 25 := by
  sorry

/-- Theorem stating that the principal amount is approximately 2580.39 -/
theorem principal_amount_approximation :
  ∃ ε > 0, |principal_amount - 2580.39| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_amount_satisfies_conditions_principal_amount_approximation_l320_32082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_ratio_theorem_l320_32079

theorem tan_ratio_theorem (α β : Real) 
  (h1 : α ∈ Set.Ioo 0 (π/2)) 
  (h2 : β ∈ Set.Ioo 0 (π/2)) 
  (h3 : Real.sin (α + β) = 3 * Real.sin (π - α + β)) : 
  Real.tan α / Real.tan β = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_ratio_theorem_l320_32079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vehicle_acceleration_exceeds_threshold_l320_32040

-- Define the vehicle's journey parameters
def total_distance : ℝ := 5280  -- feet
def max_time : ℝ := 60  -- seconds
def max_speed : ℝ := 132  -- feet per second (90 mph)
def acceleration_threshold : ℝ := 6.6  -- feet per second squared

-- Define the vehicle's journey characteristics
structure Vehicle where
  distance : ℝ
  time : ℝ
  start_velocity : ℝ
  end_velocity : ℝ
  max_velocity : ℝ
  max_acceleration : ℝ

-- State the theorem
theorem vehicle_acceleration_exceeds_threshold 
  (v : Vehicle) 
  (h1 : v.distance = total_distance)
  (h2 : v.time < max_time)
  (h3 : v.start_velocity = 0)
  (h4 : v.end_velocity = 0)
  (h5 : v.max_velocity ≤ max_speed) :
  v.max_acceleration > acceleration_threshold :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vehicle_acceleration_exceeds_threshold_l320_32040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_zero_l320_32076

-- Define the real number type
variable (x : ℝ)

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define the properties of f and g
axiom f_even : ∀ x, f (-x) = f x
axiom g_odd : ∀ x, g (-x) = -g x
axiom g_def : ∀ x, g x = f (x - 1)

-- State the theorem
theorem f_sum_zero : f 2017 + f 2019 = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_zero_l320_32076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bowling_ball_volume_l320_32024

noncomputable def sphere_volume (d : ℝ) : ℝ := (4 / 3) * Real.pi * (d / 2) ^ 3

noncomputable def cylinder_volume (r : ℝ) (h : ℝ) : ℝ := Real.pi * r ^ 2 * h

theorem bowling_ball_volume :
  let ball_diameter : ℝ := 24
  let hole_depth : ℝ := 6
  let hole1_diameter : ℝ := 2
  let hole2_diameter : ℝ := 2.5
  let hole3_diameter : ℝ := 4
  sphere_volume ball_diameter - 
    (cylinder_volume (hole1_diameter / 2) hole_depth +
     cylinder_volume (hole2_diameter / 2) hole_depth +
     cylinder_volume (hole3_diameter / 2) hole_depth) = 2264.625 * Real.pi :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bowling_ball_volume_l320_32024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jonathan_exercise_time_l320_32020

/-- Represents Jonathan's exercise routines --/
structure ExerciseRoutine where
  monday_speed : ℚ
  wednesday_speed : ℚ
  friday_speed : ℚ
  distance_per_day : ℚ

/-- Calculates the total exercise time in a week --/
def total_exercise_time (routine : ExerciseRoutine) : ℚ :=
  routine.distance_per_day / routine.monday_speed +
  routine.distance_per_day / routine.wednesday_speed +
  routine.distance_per_day / routine.friday_speed

/-- Theorem stating that Jonathan's total exercise time in a week is 6 hours --/
theorem jonathan_exercise_time :
  let routine : ExerciseRoutine := {
    monday_speed := 2,
    wednesday_speed := 3,
    friday_speed := 6,
    distance_per_day := 6
  }
  total_exercise_time routine = 6 := by
  -- Unfold the definition and simplify
  unfold total_exercise_time
  simp
  -- Perform the arithmetic
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jonathan_exercise_time_l320_32020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l320_32019

/-- The function f(x) = 1/2 * x^2 - 2x + 3 -/
noncomputable def f (x : ℝ) : ℝ := 1/2 * x^2 - 2*x + 3

/-- Theorem: If |x₁ - 2| > |x₂ - 2|, then f(x₁) > f(x₂) -/
theorem f_inequality (x₁ x₂ : ℝ) (h : |x₁ - 2| > |x₂ - 2|) : f x₁ > f x₂ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l320_32019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l320_32057

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 
  if x < 0 then -x + 3*a else a^x

theorem range_of_a (a : ℝ) : 
  (a > 0) → 
  (a ≠ 1) → 
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) < 0) → 
  a ∈ Set.Icc (1/3) 1 ∧ a < 1 := by
  sorry

#check range_of_a

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l320_32057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_integer_a_for_positive_f_l320_32025

/-- The function f(x) defined on the positive real numbers. -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 + (2 - a)*x - a*Real.log x

/-- The theorem stating that 2 is the maximum integer value of a for which f(x) > 0 when x ≥ 1. -/
theorem max_integer_a_for_positive_f :
  ∃ (a₀ : ℝ), 2 < a₀ ∧ a₀ < 3 ∧
  (∀ (a : ℝ), (∀ (x : ℝ), x ≥ 1 → f a x > 0) ↔ a ≤ a₀) ∧
  (∀ (a : ℤ), (∀ (x : ℝ), x ≥ 1 → f (a : ℝ) x > 0) ↔ a ≤ 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_integer_a_for_positive_f_l320_32025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_market_value_l320_32041

/-- Represents a stock with its face value, dividend yield, and current yield -/
structure Stock where
  face_value : ℝ
  dividend_yield : ℝ
  current_yield : ℝ

/-- Calculates the market value of a stock -/
noncomputable def market_value (s : Stock) : ℝ :=
  (s.face_value * s.dividend_yield) / s.current_yield

theorem stock_market_value (s : Stock) 
  (h1 : s.face_value = 100)
  (h2 : s.dividend_yield = 0.06)
  (h3 : s.current_yield = 0.08) :
  market_value s = 75 := by
  sorry

#eval 75

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_market_value_l320_32041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_l320_32029

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Define our function f
noncomputable def f (x : ℝ) : ℤ := 
  floor x + floor (2*x) + floor (4*x) + floor (8*x) + floor (16*x) + floor (32*x)

-- State the theorem
theorem no_solution : ∀ x : ℝ, f x ≠ 12345 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_l320_32029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_extended_quadrilateral_l320_32088

/-- Represents a quadrilateral with extended sides -/
structure ExtendedQuadrilateral where
  -- Original quadrilateral vertices
  E : ℝ × ℝ
  F : ℝ × ℝ
  G : ℝ × ℝ
  H : ℝ × ℝ
  -- Extended points
  E' : ℝ × ℝ
  F' : ℝ × ℝ
  G' : ℝ × ℝ
  H' : ℝ × ℝ

/-- Calculates the area of a quadrilateral given its vertices -/
noncomputable def quadrilateralArea (q : ExtendedQuadrilateral) : ℝ := sorry

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

/-- Theorem stating the area of E'F'G'H' is 36 given the conditions -/
theorem area_of_extended_quadrilateral (q : ExtendedQuadrilateral) :
  quadrilateralArea q = 12 →
  distance q.E q.F = 5 →
  distance q.F q.E' = 5 →
  distance q.F q.G = 6 →
  distance q.G q.G' = 6 →
  distance q.G q.H = 7 →
  distance q.H q.H' = 7 →
  distance q.H q.E = 8 →
  distance q.E q.E' = 8 →
  quadrilateralArea { E := q.E', F := q.F', G := q.G', H := q.H', E' := q.E', F' := q.F', G' := q.G', H' := q.H' } = 36 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_extended_quadrilateral_l320_32088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pancake_mix_count_l320_32053

def pancake_problem (total : ℕ) (blueberry_percent : ℚ) (banana_percent : ℚ) 
  (chocolate_percent : ℚ) (strawberry_percent : ℚ) : ℕ :=
  total - (↑total * (blueberry_percent + banana_percent + chocolate_percent + strawberry_percent)).floor.toNat

theorem pancake_mix_count :
  pancake_problem 280 (25/100) (30/100) (15/100) (10/100) = 56 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pancake_mix_count_l320_32053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_reciprocals_greater_than_one_l320_32099

theorem sum_of_reciprocals_greater_than_one 
  (m n : ℕ) 
  (hm : m > 1) 
  (hn : n > 1) : 
  (m + 1 : ℝ) ^ (-(1 : ℝ) / n) + (n + 1 : ℝ) ^ (-(1 : ℝ) / m) > 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_reciprocals_greater_than_one_l320_32099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_towel_area_decrease_approx_l320_32054

/-- Represents the percentage decrease in area of a towel after two washes -/
def towel_area_decrease : ℝ :=
  let first_wash_length_factor := 1 - 0.15
  let first_wash_breadth_factor := 1 - 0.12
  let second_wash_length_factor := 1 - 0.10
  let second_wash_breadth_factor := 1 - 0.05
  let total_factor := first_wash_length_factor * first_wash_breadth_factor * second_wash_length_factor * second_wash_breadth_factor
  (1 - total_factor) * 100

/-- Theorem stating that the percentage decrease in area of the towel after two washes is approximately 36.046% -/
theorem towel_area_decrease_approx :
  |towel_area_decrease - 36.046| < 0.001 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_towel_area_decrease_approx_l320_32054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_house_count_l320_32030

/-- Represents the cost function for building houses -/
noncomputable def cost_function (p₁ p₂ p₃ n : ℝ) : ℝ :=
  100 * 50^2 * (p₁ * (50 / n^(1/2)) + p₂ + p₃ * (n^(1/2) / 50))

/-- Theorem stating the optimal number of houses to minimize cost -/
theorem optimal_house_count (p₁ p₂ p₃ : ℝ) : 
  p₁ > 0 → p₂ > 0 → p₃ > 0 →
  p₁ * p₃ = p₂^2 →
  p₁ + p₂ + p₃ = 21 →
  p₁ * p₂ * p₃ = 64 →
  p₁ * (50 / Real.sqrt 63) < p₂ + p₃ * (Real.sqrt 63 / 50) →
  ∃ (n : ℕ), n = 156 ∧ 
    ∀ (m : ℕ), m > 0 → cost_function p₁ p₂ p₃ (m : ℝ) ≥ cost_function p₁ p₂ p₃ 156 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_house_count_l320_32030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_equations_l320_32095

/-- Helper function to represent a line through two points -/
def line_through (p q : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {r : ℝ × ℝ | ∃ t : ℝ, r = (1 - t) • p + t • q}

/-- Given three points A, B, and C in a 2D plane, this theorem proves that the lines
    forming the sides of triangle ABC have specific equations. -/
theorem triangle_side_equations (A B C : ℝ × ℝ) 
    (hA : A = (-2, 2)) (hB : B = (-2, -2)) (hC : C = (6, 6)) :
    (∀ x y : ℝ, (x, y) ∈ line_through A B ↔ x + 2 = 0) ∧
    (∀ x y : ℝ, (x, y) ∈ line_through A C ↔ x - 2*y + 6 = 0) ∧
    (∀ x y : ℝ, (x, y) ∈ line_through B C ↔ x - y = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_equations_l320_32095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_glycine_isoelectric_point_l320_32013

/-- The charge of glycine as a function of pH -/
noncomputable def charge (pH : ℝ) : ℝ := sorry

/-- The isoelectric point of glycine -/
noncomputable def isoelectric_point : ℝ := sorry

/-- Approximate equality for real numbers -/
def approx_eq (x y : ℝ) : Prop := ∃ (ε : ℝ), ε > 0 ∧ |x - y| < ε

notation:50 x " ≈ " y => approx_eq x y

theorem glycine_isoelectric_point :
  (∀ pH₁ pH₂ : ℝ, charge pH₁ - charge pH₂ = (pH₁ - pH₂) * ((1/2 - (-1/3)) / (9.6 - 3.55))) ∧
  charge 3.55 = -1/3 ∧
  charge 9.6 = 1/2 ∧
  charge isoelectric_point = 0 →
  isoelectric_point ≈ 5.97 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_glycine_isoelectric_point_l320_32013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_similar_pyramids_volume_ratio_cut_pyramid_l320_32068

/-- A pyramid with a rectangular base -/
structure RectangularPyramid where
  base_length : ℝ
  base_width : ℝ
  height : ℝ

/-- The volume of a rectangular pyramid -/
noncomputable def volume (p : RectangularPyramid) : ℝ :=
  (1/3) * p.base_length * p.base_width * p.height

/-- A theorem stating that the ratio of volumes of two similar pyramids
    is equal to the cube of the ratio of their heights -/
theorem volume_ratio_similar_pyramids (p q : RectangularPyramid) 
    (h : p.base_length / q.base_length = p.base_width / q.base_width) :
    volume p / volume q = (p.height / q.height)^3 := by sorry

theorem volume_ratio_cut_pyramid (p : RectangularPyramid) :
  let p' : RectangularPyramid := { base_length := p.base_length * (1/3), base_width := p.base_width * (1/3), height := p.height / 3 }
  volume p / volume p' = 27 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_similar_pyramids_volume_ratio_cut_pyramid_l320_32068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_to_line_l320_32002

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  (x - 2)^2 + (y + 1)^2 = 8

-- Define the line equation
def line_equation (x y : ℝ) : Prop :=
  x - y + 1 = 0

-- Define the distance function from a point to a line
noncomputable def distance_point_to_line (x₀ y₀ : ℝ) : ℝ :=
  |x₀ - y₀ + 1| / Real.sqrt 2

-- Theorem statement
theorem circle_tangent_to_line :
  ∀ x y : ℝ,
  circle_equation x y →
  (∃ x₀ y₀ : ℝ, line_equation x₀ y₀ ∧
    (x - x₀)^2 + (y - y₀)^2 = (distance_point_to_line 2 (-1))^2) :=
by
  sorry

#check circle_tangent_to_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_to_line_l320_32002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_synodic_month_calculation_l320_32028

/-- The sidereal month of the Moon in days -/
noncomputable def sidereal_month : ℝ := 27 + 7/24 + 43/(24*60)

/-- The sidereal year of the Earth in days -/
noncomputable def sidereal_year : ℝ := 365 + 6/24 + 9/(24*60)

/-- Calculate the synodic month given the sidereal month and sidereal year -/
noncomputable def synodic_month (T_H T_F : ℝ) : ℝ := (T_H * T_F) / (T_F - T_H)

/-- The calculated synodic month in days -/
noncomputable def calculated_synodic_month : ℝ := synodic_month sidereal_month sidereal_year

/-- Theorem stating that the calculated synodic month is close to 29 days 12 hours 44 minutes -/
theorem synodic_month_calculation :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.0001 ∧ 
  (abs (calculated_synodic_month - (29 + 12/24 + 44/(24*60))) < ε) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_synodic_month_calculation_l320_32028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_A_satisfies_conditions_l320_32078

theorem matrix_A_satisfies_conditions : ∃ (A : Matrix (Fin 2) (Fin 2) ℝ),
  (A.vecMul (![4, 0] : Fin 2 → ℝ) = ![12, 0]) ∧
  (A.vecMul (![2, -3] : Fin 2 → ℝ) = ![6, 9]) ∧
  (A = !![3, 0; 0, -3]) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_A_satisfies_conditions_l320_32078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_relationship_l320_32060

variable (a b r : ℝ)

/-- The distance from (0,0) to the line ax + by - r² = 0 -/
noncomputable def distance_to_line (a b r : ℝ) : ℝ := 
  |r^2| / Real.sqrt (a^2 + b^2)

theorem line_circle_relationship (a b r : ℝ) :
  (a^2 + b^2 = r^2 → distance_to_line a b r = r) ∧
  (a^2 + b^2 < r^2 → distance_to_line a b r > r) ∧
  (a^2 + b^2 = r^2 → distance_to_line a b r = r) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_relationship_l320_32060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_height_in_cylinder_l320_32017

-- Define the parameters
noncomputable def frustum_bottom_diameter : ℝ := 12
noncomputable def frustum_height : ℝ := 18
noncomputable def cylinder_diameter : ℝ := 24

-- Define the volume of the frustum
noncomputable def frustum_volume : ℝ := (1/3) * Real.pi * (frustum_bottom_diameter/2)^2 * frustum_height

-- Define the height of water in the cylinder
noncomputable def cylinder_water_height : ℝ := frustum_volume / (Real.pi * (cylinder_diameter/2)^2)

-- Theorem statement
theorem water_height_in_cylinder :
  cylinder_water_height = 1.5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_height_in_cylinder_l320_32017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_transformation_l320_32091

noncomputable def original_function (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 3)

noncomputable def shifted_function (x : ℝ) : ℝ := original_function (x + Real.pi / 3)

noncomputable def final_function (x : ℝ) : ℝ := shifted_function (2 * x)

theorem graph_transformation :
  ∀ x : ℝ, final_function x = Real.sin (4 * x + Real.pi / 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_transformation_l320_32091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_cubed_l320_32015

noncomputable def expansion (x : ℝ) : ℝ := (x - 1/x + 1) * (x - 1)^4

theorem coefficient_of_x_cubed : 
  (deriv (deriv (deriv expansion))) 0 / 6 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_cubed_l320_32015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_karlsson_has_winning_strategy_l320_32006

/-- Represents a rectangular piece of chocolate -/
structure ChocolatePiece where
  width : ℕ
  height : ℕ

/-- The game state -/
structure GameState where
  pieces : List ChocolatePiece
  moves : ℕ

/-- Represents a player's strategy -/
def Strategy := GameState → GameState

/-- Checks if a piece can be divided -/
def isDivisible (p : ChocolatePiece) : Bool :=
  p.width > 1 ∨ p.height > 1

/-- Checks if the game is over -/
def isGameOver (state : GameState) : Bool :=
  state.pieces.all (λ p => ¬isDivisible p)

/-- The winning condition for Karlsson -/
def karlssonWins (state : GameState) : Bool :=
  isGameOver state ∧ Odd state.moves

/-- Simulates the game for a given number of turns -/
def playGame (karlssonStrategy malyshStrategy : Strategy) (initialState : GameState) (turns : ℕ) : GameState :=
  match turns with
  | 0 => initialState
  | n + 1 =>
    let afterKarlsson := karlssonStrategy initialState
    if isGameOver afterKarlsson then
      afterKarlsson
    else
      playGame karlssonStrategy malyshStrategy (malyshStrategy afterKarlsson) n

theorem karlsson_has_winning_strategy :
  ∃ (karlssonStrategy : Strategy),
    ∀ (malyshStrategy : Strategy),
      karlssonWins (playGame karlssonStrategy malyshStrategy
        { pieces := [{ width := 2019, height := 2019 }], moves := 0 }
        (2019 * 2019)) := by
  sorry

#check karlsson_has_winning_strategy

end NUMINAMATH_CALUDE_ERRORFEEDBACK_karlsson_has_winning_strategy_l320_32006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_and_area_l320_32089

/-- Triangle properties and area theorem -/
theorem triangle_properties_and_area (a b c : ℝ) (A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →  -- Triangle inequality
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →  -- Angle ranges
  A + B + C = π →  -- Angle sum in a triangle
  a / (Real.sin A) = b / (Real.sin B) →  -- Law of sines
  (2 * c - b) / a = (Real.cos B) / (Real.cos A) →  -- Given condition
  a = 2 * Real.sqrt 5 →  -- Given side length
  A = π / 3 ∧  -- Angle A is π/3
  (∃ (S : ℝ), S ≤ 5 * Real.sqrt 3 ∧  -- Maximum area
    ∀ (S' : ℝ), S' = 1/2 * b * c * Real.sin A → S' ≤ S) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_and_area_l320_32089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_inequality_implies_logarithm_positive_l320_32064

theorem exponential_inequality_implies_logarithm_positive (x y : ℝ) :
  (2 : ℝ)^x - (2 : ℝ)^y < (3 : ℝ)^(-x) - (3 : ℝ)^(-y) → Real.log (y - x + 1) > 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_inequality_implies_logarithm_positive_l320_32064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_steiner_double_system_modulo_l320_32062

/-- A Steiner double system is a set of 3-element subsets (triples) of a set with n elements,
    such that every pair of elements appears in exactly one triple. -/
structure SteinerDoubleSystem (n : ℕ) where
  triples : Set (Finset (Fin n))
  triple_size : ∀ t ∈ triples, t.card = 3
  pair_uniqueness : ∀ (i j : Fin n), i ≠ j → ∃! t, t ∈ triples ∧ i ∈ t ∧ j ∈ t

/-- If a Steiner double system exists for n elements, then n ≡ 1 or 3 (mod 6) -/
theorem steiner_double_system_modulo (n : ℕ) (S : SteinerDoubleSystem n) :
  n % 6 = 1 ∨ n % 6 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_steiner_double_system_modulo_l320_32062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_approx_l320_32096

-- Constants
noncomputable def inch_to_cm : ℝ := 2.54
noncomputable def circle_area : ℝ := 39424

-- Definitions
noncomputable def circle_radius (area : ℝ) : ℝ := Real.sqrt (area / Real.pi)
noncomputable def circle_radius_inches (radius_cm : ℝ) : ℝ := radius_cm / inch_to_cm
noncomputable def square_side_length (perimeter : ℝ) : ℝ := perimeter / 4
noncomputable def square_area (side : ℝ) : ℝ := side ^ 2

-- Theorem
theorem square_area_approx :
  let radius_cm := circle_radius circle_area
  let radius_inches := circle_radius_inches radius_cm
  let square_side := square_side_length radius_inches
  let area := square_area square_side
  ∃ ε > 0, abs (area - 121.44) < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_approx_l320_32096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2A_value_l320_32032

theorem sin_2A_value (A B : ℝ) (h1 : 0 < A ∧ A < π/2) (h2 : 0 < B ∧ B < π/2)
  (h3 : Real.tan A - 1 / (Real.sin (2 * A)) = Real.tan B) 
  (h4 : (Real.cos (B / 2)) ^ 2 = Real.sqrt 6 / 3) :
  Real.sin (2 * A) = (2 * Real.sqrt 6 - 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2A_value_l320_32032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_tangent_ratio_l320_32023

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem triangle_tangent_ratio (t : Triangle) 
  (h : t.a * Real.cos t.B - t.b * Real.cos t.A + 2 * t.c = 0) : 
  Real.tan t.A / Real.tan t.B = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_tangent_ratio_l320_32023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_AB_is_two_l320_32044

-- Define the line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (6 + (Real.sqrt 2 / 2) * t, (Real.sqrt 2 / 2) * t)

-- Define the curve C
noncomputable def curve_C (θ : ℝ) : ℝ × ℝ := (6 * Real.cos θ * Real.cos θ, 6 * Real.cos θ * Real.sin θ)

-- Define point M
def point_M : ℝ × ℝ := (-1, 0)

-- Define line l1 (parallel to l and passing through M)
noncomputable def line_l1 (t : ℝ) : ℝ × ℝ := (-1 + (Real.sqrt 2 / 2) * t, (Real.sqrt 2 / 2) * t)

-- Theorem statement
theorem length_of_AB_is_two :
  ∃ (t1 t2 : ℝ), t1 ≠ t2 ∧ 
  (∃ (θ1 θ2 : ℝ), line_l1 t1 = curve_C θ1 ∧ line_l1 t2 = curve_C θ2) ∧
  Real.sqrt ((t1 - t2)^2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_AB_is_two_l320_32044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tea_cost_is_111_l320_32010

/-- The cost of one cup of tea in kopecks -/
def tea_cost : ℕ := sorry

/-- 9 cups of tea cost less than 1000 kopecks -/
axiom nine_cups_cost : 9 * tea_cost < 1000

/-- 10 cups of tea cost more than 1100 kopecks -/
axiom ten_cups_cost : 10 * tea_cost > 1100

/-- The cost of one cup of tea is 111 kopecks -/
theorem tea_cost_is_111 : tea_cost = 111 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tea_cost_is_111_l320_32010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dakota_medical_bill_l320_32075

/-- Calculates the total medical bill for Dakota's hospital stay --/
def total_medical_bill (
  days_in_hospital : ℕ
) (bed_cost_per_day : ℚ
) (specialist_cost_per_hour : ℚ
) (specialist_time_minutes : ℕ
) (num_specialists : ℕ
) (ambulance_cost : ℚ
) : ℚ :=
  let bed_total := days_in_hospital * bed_cost_per_day
  let specialist_total := (specialist_cost_per_hour / 60) * specialist_time_minutes * num_specialists
  bed_total + specialist_total + ambulance_cost

theorem dakota_medical_bill :
  total_medical_bill 3 900 250 15 2 1800 = 4625 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dakota_medical_bill_l320_32075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_divisors_of_3003_l320_32081

def n : ℕ := 3003

theorem number_of_divisors_of_3003 : 
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).card = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_divisors_of_3003_l320_32081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_is_pi_over_three_l320_32000

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  S : ℝ

-- State the theorem
theorem angle_B_is_pi_over_three (t : Triangle) 
  (h1 : 2 * (Real.sin t.A) ^ 2 + t.c * (Real.sin t.C - Real.sin t.A) = 2 * (Real.sin t.B) ^ 2)
  (h2 : t.S = (1/4) * t.a * t.b * t.c) :
  t.B = π/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_is_pi_over_three_l320_32000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_root_between_3_and_4_l320_32008

-- Define the function f(x) = ln x + 2x - 8
noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 * x - 8

-- State the theorem
theorem f_has_root_between_3_and_4 :
  ∃ x ∈ Set.Ioo 3 4, f x = 0 :=
by
  -- The proof would go here, but we'll skip it
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_root_between_3_and_4_l320_32008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_descartes_folium_properties_l320_32039

/-- The Descartes folium equation -/
def descartes_folium (a : ℝ) (x y : ℝ) : Prop := x^3 + y^3 = 3*a*x*y

theorem descartes_folium_properties (a : ℝ) (h : a ≠ 0) :
  /- 1. Symmetry about y = x -/
  (∀ x y : ℝ, descartes_folium a x y ↔ descartes_folium a y x) ∧
  /- 2. No points in third quadrant when a > 0 -/
  (a > 0 → ∀ x y : ℝ, descartes_folium a x y → ¬(x < 0 ∧ y < 0)) ∧
  /- 3. Maximum value of y when a = 1 -/
  (a = 1 → ∀ x y : ℝ, x > 0 → y > 0 → descartes_folium a x y → y ≤ (4 : ℝ)^(1/3)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_descartes_folium_properties_l320_32039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_and_line_equations_l320_32097

noncomputable def a_ellipse_squared : ℝ := 49
noncomputable def b_ellipse_squared : ℝ := 24

noncomputable def asymptote_slope : ℝ := 4/3

noncomputable def a_hyperbola_squared : ℝ := 9
noncomputable def b_hyperbola_squared : ℝ := 16

noncomputable def line_slope : ℝ := Real.sqrt 3
noncomputable def line_intercept : ℝ := 5 * Real.sqrt 3

theorem hyperbola_and_line_equations :
  let c_squared := a_ellipse_squared - b_ellipse_squared
  let right_focus := (Real.sqrt c_squared, 0)
  ∃ (a b : ℝ),
    (a > 0 ∧ b > 0) ∧
    (b / a = asymptote_slope) ∧
    (a^2 - b^2 = c_squared) ∧
    (a^2 = a_hyperbola_squared ∧ b^2 = b_hyperbola_squared) ∧
    (∀ (x y : ℝ), y = line_slope * (x - right_focus.1) ↔ line_slope * x - y - line_intercept = 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_and_line_equations_l320_32097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_three_lines_l320_32063

/-- Definition of a line in ℝ² -/
def IsLine (s : Set (ℝ × ℝ)) : Prop :=
  ∃ a b c, (a ≠ 0 ∨ b ≠ 0) ∧ ∀ x y, (x, y) ∈ s ↔ a * x + b * y + c = 0

/-- The solution set of the equation x²(x+y+2) = y²(x+y+2) consists of three lines that do not all pass through a common point. -/
theorem solution_set_three_lines :
  ∃ (l₁ l₂ l₃ : Set (ℝ × ℝ)),
    (∀ x y, x^2 * (x + y + 2) = y^2 * (x + y + 2) ↔ (x, y) ∈ l₁ ∪ l₂ ∪ l₃) ∧
    IsLine l₁ ∧ IsLine l₂ ∧ IsLine l₃ ∧
    ¬(∃ p, p ∈ l₁ ∧ p ∈ l₂ ∧ p ∈ l₃) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_three_lines_l320_32063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_and_range_l320_32007

-- Define the function f as noncomputable due to the use of Real.sqrt
noncomputable def f (x : Real) : Real := Real.sqrt 3 * (Real.sin x ^ 2 - Real.cos x ^ 2) + 2 * Real.sin x * Real.cos x

-- State the theorem about the period and range of f
theorem f_period_and_range :
  (∃ (p : Real), p > 0 ∧ ∀ (x : Real), f (x + p) = f x ∧ 
    ∀ (q : Real), q > 0 ∧ (∀ (x : Real), f (x + q) = f x) → p ≤ q) ∧
  (∀ (x : Real), 0 ≤ x ∧ x ≤ Real.pi / 3 → 
    -Real.sqrt 3 ≤ f x ∧ f x ≤ Real.sqrt 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_and_range_l320_32007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_complement_B_l320_32058

-- Define the universal set R as the real numbers
def R := ℝ

-- Define set A
def A : Set ℝ := {x | 0 < x ∧ x < 4}

-- Define set B
def B : Set ℝ := {x | x > 2}

-- State the theorem
theorem intersection_A_complement_B :
  A ∩ (Set.univ \ B) = {x : ℝ | 0 < x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_complement_B_l320_32058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_slope_l320_32022

-- Define the points
def point1 : ℝ × ℝ := (3, -6)
def point2 : ℝ × ℝ := (-4, 2)

-- Define the slope of the line containing the given points
noncomputable def line_slope : ℝ := (point2.2 - point1.2) / (point2.1 - point1.1)

-- Theorem: The slope of a line perpendicular to the line containing the given points is 7/8
theorem perpendicular_slope :
  let m := -1 / line_slope
  m = 7/8 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_slope_l320_32022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l320_32077

-- Define the function f(x) = 2xe^x
noncomputable def f (x : ℝ) : ℝ := 2 * x * Real.exp x

-- State the theorem
theorem min_value_of_f :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x_min ≤ f x ∧ f x_min = -2 / Real.exp 1 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l320_32077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_train_speed_l320_32085

/-- The speed of a train given the lengths of two trains, the speed of one train, and the time they take to cross each other when moving in opposite directions. -/
noncomputable def train_speed (length1 length2 speed2 cross_time : ℝ) : ℝ :=
  let total_distance := (length1 + length2) / 1000  -- Convert to km
  let cross_time_hours := cross_time / 3600  -- Convert to hours
  let relative_speed := total_distance / cross_time_hours
  relative_speed - speed2

/-- Theorem stating that under the given conditions, the speed of the first train is approximately 120.016 kmph. -/
theorem first_train_speed :
  let length1 : ℝ := 270
  let length2 : ℝ := 230.04
  let speed2 : ℝ := 80
  let cross_time : ℝ := 9
  abs (train_speed length1 length2 speed2 cross_time - 120.016) < 0.001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_train_speed_l320_32085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_intersection_theorem_l320_32014

-- Define the circles
variable (α β γ : Set (EuclideanSpace ℝ (Fin 2)))

-- Define the tangent lines
variable (l₁ l₂ m₁ m₂ n₁ n₂ : Set (EuclideanSpace ℝ (Fin 2)))

-- Define the property of being a common internal tangent
def is_common_internal_tangent (l : Set (EuclideanSpace ℝ (Fin 2))) (c₁ c₂ : Set (EuclideanSpace ℝ (Fin 2))) : Prop :=
  sorry  -- Placeholder for the actual definition

-- Define the property of lines intersecting at a single point
def intersect_at_single_point (l₁ l₂ l₃ : Set (EuclideanSpace ℝ (Fin 2))) : Prop :=
  sorry  -- Placeholder for the actual definition

-- State the theorem
theorem tangent_intersection_theorem
  (h₁ : is_common_internal_tangent l₁ α β)
  (h₂ : is_common_internal_tangent l₂ α β)
  (h₃ : is_common_internal_tangent m₁ β γ)
  (h₄ : is_common_internal_tangent m₂ β γ)
  (h₅ : is_common_internal_tangent n₁ γ α)
  (h₆ : is_common_internal_tangent n₂ γ α)
  (h₇ : intersect_at_single_point l₁ m₁ n₁) :
  intersect_at_single_point l₂ m₂ n₂ :=
by
  sorry  -- Placeholder for the actual proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_intersection_theorem_l320_32014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_n_good_not_n_plus_one_good_l320_32018

/-- Sum of digits of a positive integer -/
def S (k : ℕ+) : ℕ := sorry

/-- Sequence satisfying the recurrence relation -/
def IsValidSequence (a : List ℕ+) : Prop :=
  ∀ i, i + 1 < a.length → a[i+1]! = a[i]! - S a[i]!

/-- n-good property -/
def IsNGood (n : ℕ) (a : ℕ+) : Prop :=
  ∃ (seq : List ℕ+), seq.length = n + 1 ∧ IsValidSequence seq ∧ seq.getLast! = a

/-- Main theorem -/
theorem exists_n_good_not_n_plus_one_good :
  ∀ n : ℕ, ∃ a : ℕ+, IsNGood n a ∧ ¬IsNGood (n + 1) a := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_n_good_not_n_plus_one_good_l320_32018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_volume_ratio_l320_32027

/-- Represents a cone with height h and radius r -/
structure Cone where
  h : ℝ
  r : ℝ

/-- Calculates the volume of a cone -/
noncomputable def coneVolume (c : Cone) : ℝ := (1/3) * Real.pi * c.r^2 * c.h

/-- Represents a cone partially filled with water -/
structure FilledCone extends Cone where
  fillHeight : ℝ
  fillHeightRatio : fillHeight = (2/3) * h

theorem water_volume_ratio (c : FilledCone) : 
  let waterVolume := coneVolume { h := c.fillHeight, r := (c.fillHeight / c.h) * c.r }
  waterVolume / coneVolume c.toCone = 8/27 := by
  sorry

#check water_volume_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_volume_ratio_l320_32027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monge_circle_tangency_l320_32067

/-- The Monge circle radius for an ellipse -/
noncomputable def monge_circle_radius (a b : ℝ) : ℝ := Real.sqrt (a^2 + b^2)

/-- The distance between two points in 2D space -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

/-- The theorem about the tangency of the circle and the Monge circle -/
theorem monge_circle_tangency (b : ℝ) : 
  let a := Real.sqrt 3
  let monge_radius := monge_circle_radius a 1
  let circle_radius := 3
  let circle_center_distance := distance 3 b 0 0
  (circle_center_distance = monge_radius + circle_radius) → b = 4 := by
  sorry

#check monge_circle_tangency

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monge_circle_tangency_l320_32067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_range_theorem_l320_32042

-- Define the function f
noncomputable def f (x : ℝ) := 2 * x + Real.sin x

-- State the theorem
theorem x_range_theorem (x : ℝ) :
  (∀ m : ℝ, m ∈ Set.Icc (-2) 2 → f (m * x - 3) + f x < 0) →
  x ∈ Set.Ioo (-3) 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_range_theorem_l320_32042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_four_digit_divisible_by_11_with_two_even_two_odd_answer_satisfies_conditions_l320_32065

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def has_two_even_two_odd_digits (n : ℕ) : Prop :=
  let digits := n.digits 10
  (digits.filter (λ d => d % 2 = 0)).length = 2 ∧ 
  (digits.filter (λ d => d % 2 = 1)).length = 2

theorem smallest_four_digit_divisible_by_11_with_two_even_two_odd :
  ∀ n : ℕ, is_four_digit n →
           n % 11 = 0 →
           has_two_even_two_odd_digits n →
           1056 ≤ n :=
by sorry

theorem answer_satisfies_conditions :
  is_four_digit 1056 ∧
  1056 % 11 = 0 ∧
  has_two_even_two_odd_digits 1056 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_four_digit_divisible_by_11_with_two_even_two_odd_answer_satisfies_conditions_l320_32065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_triple_second_at_calculated_time_l320_32033

/-- The time (in hours) at which the height of the first candle is three times the height of the second candle. -/
noncomputable def time_when_first_is_triple_second : ℝ :=
  40 / 11

/-- The burn rate of the first candle (units per hour). -/
noncomputable def burn_rate_first : ℝ :=
  1 / 5

/-- The burn rate of the second candle (units per hour). -/
noncomputable def burn_rate_second : ℝ :=
  1 / 4

/-- The height of the first candle after t hours. -/
noncomputable def height_first (t : ℝ) : ℝ :=
  1 - burn_rate_first * t

/-- The height of the second candle after t hours. -/
noncomputable def height_second (t : ℝ) : ℝ :=
  1 - burn_rate_second * t

theorem first_triple_second_at_calculated_time :
  height_first time_when_first_is_triple_second = 3 * height_second time_when_first_is_triple_second :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_triple_second_at_calculated_time_l320_32033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_properties_log_power_comparison_l320_32069

noncomputable def f (x : ℝ) := x ^ (-4 : ℤ)

theorem power_function_properties :
  ∀ (k : ℕ+),
  (∀ (x : ℝ), x > 0 → f x = x ^ (k^2 - 2*k - 3 : ℤ)) →
  (∀ (x : ℝ), x ≠ 0 → f x = f (-x)) →
  (∀ (x y : ℝ), 0 < x ∧ x < y → f y < f x) →
  k = 1 :=
by sorry

theorem log_power_comparison (a : ℝ) (h : a > 1) :
  (a < Real.exp 1 → (Real.log a)^(0.7 : ℝ) < (Real.log a)^(0.6 : ℝ)) ∧
  (a = Real.exp 1 → (Real.log a)^(0.7 : ℝ) = (Real.log a)^(0.6 : ℝ)) ∧
  (a > Real.exp 1 → (Real.log a)^(0.7 : ℝ) > (Real.log a)^(0.6 : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_properties_log_power_comparison_l320_32069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_divisibility_l320_32052

def g : ℕ → ℕ
  | 0 => 1
  | n + 1 => g n ^ 2 + g n + 1

theorem g_divisibility (n : ℕ) :
  ∃ k : ℕ, g (n + 1) ^ 2 + 1 = (g n ^ 2 + 1) * k :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_divisibility_l320_32052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_divisible_by_each_l320_32031

theorem sum_divisible_by_each (S : Finset ℕ) : 
  S = {1, 2, 3, 6, 12, 24, 48, 96, 192, 384} →
  S.card = 10 ∧ 
  (∀ x, x ∈ S → x > 0) ∧
  (∀ x y, x ∈ S → y ∈ S → x ≠ y → x ≠ y) ∧
  (∀ x, x ∈ S → (S.sum id) % x = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_divisible_by_each_l320_32031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_general_term_l320_32059

-- Define the sequence a_n and its partial sum S_n
def a : ℕ → ℚ := sorry
def S : ℕ → ℚ := sorry

-- Define the condition for S_n + a_n
axiom condition (n : ℕ) : n ≥ 1 → S n + a n = (n - 1 : ℚ) / (n * (n + 1))

-- The theorem to prove
theorem general_term (n : ℕ) (h : n ≥ 1) : 
  a n = 1 / (2^n) - 1 / (n * (n + 1)) := by
  sorry

-- Additional lemma that might be useful for the proof
lemma partial_sum_relation (n : ℕ) (h : n ≥ 1) :
  S (n + 1) = S n + a (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_general_term_l320_32059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_difference_S_l320_32047

def a : ℕ → ℤ
  | 0 => -8
  | n + 1 => ((3 * n.succ - 2) * a n - 9 * n.succ^2 + 2 * n.succ - 10) / (3 * n.succ - 5)

def S (n : ℕ) : ℤ := (Finset.range n).sum (λ i => a i)

theorem max_difference_S :
  (∀ n m : ℕ, n > m → S n - S m ≤ 18) ∧ (∃ p q : ℕ, p > q ∧ S p - S q = 18) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_difference_S_l320_32047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l320_32050

/-- The parabola y² = 4x -/
def parabola (x y : ℝ) : Prop := y^2 = 4*x

/-- Point A is fixed at (0, 2√2) -/
noncomputable def point_A : ℝ × ℝ := (0, 2 * Real.sqrt 2)

/-- Q is the intersection of the perpendicular line from P to the y-axis -/
def point_Q (P : ℝ × ℝ) : ℝ × ℝ := (0, P.2)

/-- The distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- The theorem stating the minimum value of |PA| + |PQ| -/
theorem min_distance_sum :
  ∃ (min : ℝ), min = 2 ∧
  ∀ (P : ℝ × ℝ), parabola P.1 P.2 →
    distance P point_A + distance P (point_Q P) ≥ min := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l320_32050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_iff_m_in_range_l320_32012

/-- The function f(x) = mx^2 - 2x + 3 -/
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 - 2 * x + 3

/-- The domain of x is [-2, +∞) -/
def domain : Set ℝ := { x | x ≥ -2 }

/-- The theorem stating the equivalence between the condition and the range of m -/
theorem f_decreasing_iff_m_in_range (m : ℝ) : 
  (∀ (x₁ x₂ : ℝ), x₁ ∈ domain → x₂ ∈ domain → x₁ ≠ x₂ → (f m x₁ - f m x₂) / (x₁ - x₂) < 0) ↔ 
  m ∈ Set.Icc (-1/2) 0 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_iff_m_in_range_l320_32012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_related_function_l320_32034

open Real

theorem max_value_of_related_function 
  (a b : ℝ) 
  (f g : ℝ → ℝ) 
  (hf : f = λ x ↦ a * cos x + b) 
  (hg : g = λ x ↦ 3 + a * b * sin x) 
  (hmax : ∀ x, f x ≤ 1) 
  (hmin : ∀ x, f x ≥ -7) 
  (hmax_achieved : ∃ x, f x = 1) 
  (hmin_achieved : ∃ x, f x = -7) : 
  ∀ x, g x ≤ 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_related_function_l320_32034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_when_a_is_neg_three_range_of_a_for_all_x_l320_32037

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + a| + |x - 2|

-- Theorem 1
theorem solution_set_when_a_is_neg_three :
  {x : ℝ | f (-3) x ≥ 3} = Set.Iic 1 ∪ Set.Ici 4 := by sorry

-- Theorem 2
theorem range_of_a_for_all_x (a : ℝ) :
  (∀ x : ℝ, f a x ≥ 3) ↔ (a ≥ 1 ∨ a ≤ -5) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_when_a_is_neg_three_range_of_a_for_all_x_l320_32037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_of_triangle_l320_32093

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : a > 0
  pos_b : b > 0
  pos_c : c > 0
  triangle_ineq : a + b > c ∧ b + c > a ∧ c + a > b

/-- Calculate the area of a triangle using Heron's formula -/
noncomputable def area (t : Triangle) : ℝ :=
  let s := (t.a + t.b + t.c) / 2
  Real.sqrt (s * (s - t.a) * (s - t.b) * (s - t.c))

/-- The theorem stating the maximum area of the triangle -/
theorem max_area_of_triangle :
  ∃ (t : Triangle),
    t.a = 7 ∧
    t.b / t.c = 30 / 31 ∧
    ∀ (t' : Triangle),
      t'.a = 7 →
      t'.b / t'.c = 30 / 31 →
      area t' ≤ 1200.5 ∧
      area t = 1200.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_of_triangle_l320_32093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_n_b_n_odd_l320_32026

theorem a_n_b_n_odd (n : ℕ) : 
  let a_n := (2 + Real.sqrt 7) ^ (2 * n.succ + 1)
  let b_n := a_n - ⌊a_n⌋
  ∃ k : ℕ, a_n * b_n = 2 * k + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_n_b_n_odd_l320_32026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_even_l320_32094

noncomputable def g (x : ℝ) : ℝ := 5 / (3 * x^4 - 7)

theorem g_is_even : ∀ x, g (-x) = g x := by
  intro x
  simp [g]
  ring


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_even_l320_32094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_theorem_l320_32087

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := x * Real.exp x
noncomputable def g (x : ℝ) : ℝ := -Real.log x / x

-- Define the main theorem
theorem max_value_theorem (x₁ x₂ t : ℝ) (h₁ : t > 0) (h₂ : f x₁ = t) (h₃ : g x₂ = t) :
  ∃ (M : ℝ), M = 1 / Real.exp 1 ∧ ∀ (y₁ y₂ s : ℝ), s > 0 → f y₁ = s → g y₂ = s →
    y₁ / (y₂ * Real.exp s) ≤ M := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_theorem_l320_32087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_locus_and_min_distance_l320_32003

-- Define the parametric equations of lines l₁ and l₂
noncomputable def l₁ (t k : ℝ) : ℝ × ℝ := (t - Real.sqrt 3, k * t)
noncomputable def l₂ (m k : ℝ) : ℝ × ℝ := (Real.sqrt 3 - m, m / (3 * k))

-- Define the curve C₁
def C₁ : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 / 3 + p.2^2 = 1 ∧ p.2 ≠ 0}

-- Define the line C₂
def C₂ : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 + p.2 = 8}

theorem intersection_locus_and_min_distance :
  (∀ k : ℝ, k ≠ 0 → ∃ t m : ℝ, l₁ t k = l₂ m k ∧ l₁ t k ∈ C₁) ∧
  (∀ p : ℝ × ℝ, p ∈ C₁ → Real.sqrt ((p.1 - 8)^2 + (p.2 - 8)^2) ≥ 3 * Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_locus_and_min_distance_l320_32003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_to_equation_l320_32072

-- Define the ⊕ operation
noncomputable def oplus (a b : ℝ) : ℝ := 1 / a + 1 / b

-- State the theorem
theorem solution_to_equation :
  ∃ x : ℝ, x * (x + 1) / (oplus x (x + 1)) = 1 / 3 ∧ x = 1 :=
by
  -- Provide the existence of x
  use 1
  -- Split the goal into two parts
  apply And.intro
  -- Prove the equation
  · sorry
  -- Prove x = 1
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_to_equation_l320_32072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_area_l320_32090

-- Define the curve
noncomputable def f (x : ℝ) : ℝ := 1 / Real.exp x

-- Define the tangent line
def tangent_line (t : ℝ) (x y : ℝ) : Prop :=
  x + Real.exp t * y - t - 1 = 0

-- Define the area function
noncomputable def S (t : ℝ) : ℝ := (t + 1)^2 / (2 * Real.exp t)

-- Theorem statement
theorem tangent_line_area (t : ℝ) (h : t > 0) :
  let M : ℝ × ℝ := (t + 1, 0)
  let N : ℝ × ℝ := (0, (t + 1) / Real.exp t)
  tangent_line t M.1 M.2 ∧
  tangent_line t N.1 N.2 ∧
  S t = (1/2) * M.1 * N.2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_area_l320_32090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_combinability_with_sqrt_2_l320_32035

theorem sqrt_combinability_with_sqrt_2 :
  (∃ (q : ℚ), Real.sqrt (1/2 : ℝ) = q * Real.sqrt 2) ∧
  (∃ (q : ℚ), Real.sqrt (8 : ℝ) = q * Real.sqrt 2) ∧
  (∀ (q : ℚ), Real.sqrt (12 : ℝ) ≠ q * Real.sqrt 2) ∧
  (∃ (q : ℚ), Real.sqrt (18 : ℝ) = q * Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_combinability_with_sqrt_2_l320_32035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_A_is_pi_third_max_area_equilateral_l320_32074

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given condition from the problem -/
def condition (t : Triangle) : Prop :=
  (2 * t.b - t.c) * Real.cos t.A - t.a * Real.cos t.C = 0

/-- Area of a triangle -/
noncomputable def area (t : Triangle) : ℝ :=
  1 / 2 * t.a * t.b * Real.sin t.C

/-- Theorem 1: If the condition holds, then angle A is π/3 -/
theorem angle_A_is_pi_third (t : Triangle) (h : condition t) : t.A = π / 3 := by
  sorry

/-- Theorem 2: If a = √3 and the condition holds, the area is maximized when the triangle is equilateral -/
theorem max_area_equilateral (t : Triangle) (h1 : t.a = Real.sqrt 3) (h2 : condition t) :
  (∀ s : Triangle, s.a = t.a → area s ≤ area t) → (t.a = t.b ∧ t.b = t.c) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_A_is_pi_third_max_area_equilateral_l320_32074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_values_l320_32001

noncomputable def f (a b x : ℝ) : ℝ := 2 * a * Real.sin (2 * x - Real.pi / 3) + b

theorem function_values (a b : ℝ) : 
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f a b x ≤ 1) ∧ 
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f a b x ≥ -5) ∧ 
  (∃ x ∈ Set.Icc 0 (Real.pi / 2), f a b x = 1) ∧ 
  (∃ x ∈ Set.Icc 0 (Real.pi / 2), f a b x = -5) →
  a = 12 - 6 * Real.sqrt 3 ∧ b = -23 + 12 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_values_l320_32001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_comparison_l320_32021

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define the properties of f
axiom f_even : ∀ x, f (-x) = f x
axiom f_inverse : ∀ x, f (x + 1) = 1 / f x
axiom f_decreasing : ∀ x ∈ Set.Icc (-1) 0, ∀ y ∈ Set.Icc (-1) 0, x < y → f x > f y

-- Define a, b, and c
noncomputable def a := f (Real.log 2 / Real.log 5)
noncomputable def b := f (Real.log 4 / Real.log 2)
noncomputable def c := f (Real.sqrt 2)

-- State the theorem
theorem f_comparison : a > c ∧ c > b :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_comparison_l320_32021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rope_cutting_probability_l320_32016

/-- The length of the rope in meters -/
noncomputable def rope_length : ℝ := 5

/-- The minimum length of each segment in meters -/
noncomputable def min_segment_length : ℝ := 1.5

/-- The probability of cutting the rope such that both segments are at least min_segment_length long -/
noncomputable def probability_both_segments_long : ℝ := 2 / 5

/-- Theorem stating that the probability of cutting a rope of length rope_length
    such that both resulting segments are at least min_segment_length long
    is equal to probability_both_segments_long -/
theorem rope_cutting_probability :
  (2 * min_segment_length ≤ rope_length) →
  (probability_both_segments_long = (rope_length - 2 * min_segment_length) / rope_length) :=
by sorry

#check rope_cutting_probability

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rope_cutting_probability_l320_32016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_is_1740_l320_32080

/-- Represents the cost structure for travel -/
structure TravelCost where
  busCostPerKm : ℝ
  airplaneCostPerKm : ℝ
  airplaneBookingFee : ℝ

/-- Represents a right-angled triangle with cities at its vertices -/
structure CityTriangle where
  de : ℝ
  df : ℝ

/-- Calculates the cost of airplane travel between two points -/
def airplaneCost (distance : ℝ) (cost : TravelCost) : ℝ :=
  cost.airplaneBookingFee + cost.airplaneCostPerKm * distance

/-- Calculates the total travel cost for the triangle using airplanes -/
noncomputable def totalAirplaneCost (triangle : CityTriangle) (cost : TravelCost) : ℝ :=
  let ef := (triangle.de ^ 2 - triangle.df ^ 2).sqrt
  airplaneCost triangle.de cost + airplaneCost ef cost + airplaneCost triangle.df cost

/-- Theorem stating that the total airplane cost for the given problem is $1740 -/
theorem total_cost_is_1740 (triangle : CityTriangle) (cost : TravelCost) :
  triangle.de = 5000 →
  triangle.df = 4000 →
  cost.busCostPerKm = 0.2 →
  cost.airplaneCostPerKm = 0.12 →
  cost.airplaneBookingFee = 120 →
  totalAirplaneCost triangle cost = 1740 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_is_1740_l320_32080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_value_l320_32073

-- Define the power function as noncomputable
noncomputable def power_function (α : ℝ) (x : ℝ) : ℝ := x^α

-- State the theorem
theorem power_function_value (α : ℝ) :
  (power_function α 2 = Real.sqrt 2 / 2) → (power_function α 16 = 1 / 4) :=
by
  intro h
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_value_l320_32073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_A_intersect_B_equals_zero_two_open_l320_32011

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set B
def B : Set ℝ := {x : ℝ | Real.rpow (1/2) x ≤ 1}

-- Define set A
def A : Set ℝ := {x : ℝ | x ≥ 2}

-- State the theorem
theorem complement_A_intersect_B_equals_zero_two_open :
  (Set.univ \ A) ∩ B = Set.Ioc 0 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_A_intersect_B_equals_zero_two_open_l320_32011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_area_l320_32049

/-- Given two squares and an isosceles right triangle, prove the area of the triangle -/
theorem isosceles_right_triangle_area
  (small_square_area large_square_area : ℝ)
  (h_small_area : small_square_area = 64)
  (h_large_area : large_square_area = 256)
  (leg hypotenuse : ℝ)
  (h_isosceles : leg = Real.sqrt small_square_area)
  (h_right : 2 * leg^2 = hypotenuse^2) :
  (1/2 : ℝ) * leg * leg = 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_area_l320_32049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_closest_to_nine_l320_32061

theorem shaded_area_closest_to_nine (rectangle_length : ℝ) (rectangle_width : ℝ) (circle_diameter : ℝ) : 
  rectangle_length = 3 ∧ rectangle_width = 4 ∧ circle_diameter = 2 →
  ∃ (shaded_area : ℝ), 
    shaded_area = rectangle_length * rectangle_width - Real.pi * (circle_diameter / 2)^2 ∧
    ∀ (n : ℕ), n ≠ 9 → abs (shaded_area - 9) < abs (shaded_area - n) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_closest_to_nine_l320_32061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_store_item_cost_l320_32043

noncomputable section

/-- The cost of an item of type B in yuan -/
noncomputable def cost_B : ℝ := 40

/-- The cost of an item of type A in yuan -/
noncomputable def cost_A : ℝ := cost_B + 10

/-- The number of items of type A that can be purchased with 1000 yuan -/
noncomputable def num_A : ℝ := 1000 / cost_A

/-- The number of items of type B that can be purchased with 800 yuan -/
noncomputable def num_B : ℝ := 800 / cost_B

theorem store_item_cost :
  num_A = num_B ∧ cost_A = cost_B + 10 → cost_B = 40 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_store_item_cost_l320_32043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_Y_approaches_M_l320_32046

-- Define the necessary structures
structure Point where
  x : ℝ
  y : ℝ

-- Define the line segment [AB]
def A : Point := ⟨0, 0⟩
def B : Point := ⟨1, 0⟩

-- Define the midpoint M
def M : Point := ⟨0.5, 0⟩

-- Define the perpendicular bisector m
def m (t : ℝ) : Point := ⟨0.5, t⟩

-- Define X as a point on m, parameterized by t
def X (t : ℝ) : Point := m t

-- Define Y as the intersection of BX and the angle bisector of ∠BAX
noncomputable def Y (t : ℝ) : Point := sorry

-- State the theorem
theorem Y_approaches_M :
  ∀ ε > 0, ∃ δ > 0, ∀ t : ℝ, 0 < |t| ∧ |t| < δ → 
    Real.sqrt ((Y t).x - M.x)^2 + ((Y t).y - M.y)^2 < ε :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_Y_approaches_M_l320_32046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_I_1_part_I_2_part_II_l320_32038

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := a * Real.log x - b * x^2

-- Part I.1
theorem part_I_1 (a b : ℝ) :
  (∀ x, f a b x = -1/2 ↔ x = 1) ∧ 
  (∃ x, f a b x = -1/2 ∧ (deriv (f a b)) x = 0) →
  a = 1 ∧ b = 1/2 :=
sorry

-- Part I.2
theorem part_I_2 :
  ∀ x ∈ Set.Icc (1/Real.exp 1) (Real.exp 1), f 1 (1/2) x ≤ -1/2 :=
sorry

-- Part II
theorem part_II :
  ∀ m : ℝ, (∀ a ∈ Set.Icc 0 (3/2), ∀ x ∈ Set.Ioo 1 ((Real.exp 1)^2),
    f a 0 x ≥ m + x) →
  m ≤ -(Real.exp 1)^2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_I_1_part_I_2_part_II_l320_32038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squares_arithmetic_sequence_l320_32071

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ → ℕ := 
  λ k => a₁ + (k - 1) * d

def sum_of_squares (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  (n / 6) * (2 * a₁ + (n - 1) * d) * (2 * a₁ + (n - 1) * d + 3 * d)

theorem sum_of_squares_arithmetic_sequence :
  sum_of_squares 1 3 45 = 143565 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squares_arithmetic_sequence_l320_32071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_legged_is_outlier_l320_32009

-- Define the type for dragon attributes
inductive DragonAttribute
| OneEyed
| TwoEared
| ThreeTailed
| FourLegged
| FiveSpiked

-- Define a function to check if an attribute has doubled letters in Russian
def hasDoubledLetters : DragonAttribute → Bool
  | DragonAttribute.OneEyed => false    -- одноокий
  | DragonAttribute.TwoEared => true    -- двуухий
  | DragonAttribute.ThreeTailed => true -- треххвостый
  | DragonAttribute.FourLegged => false -- четырехлапый
  | DragonAttribute.FiveSpiked => true  -- пятиглый

-- Define the set of all dragon attributes
def allAttributes : List DragonAttribute :=
  [DragonAttribute.OneEyed, DragonAttribute.TwoEared, DragonAttribute.ThreeTailed, 
   DragonAttribute.FourLegged, DragonAttribute.FiveSpiked]

-- Theorem: The attribute without doubled letters is FourLegged
theorem four_legged_is_outlier :
  ∃! attr, attr ∈ allAttributes ∧ ¬(hasDoubledLetters attr) ∧ attr = DragonAttribute.FourLegged :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_legged_is_outlier_l320_32009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_of_cyclic_matrix_with_polynomial_roots_l320_32036

theorem determinant_of_cyclic_matrix_with_polynomial_roots (p q r : ℝ) (a b c d : ℝ) 
  (h1 : a^4 + p*a^2 + q*a + r = 0)
  (h2 : b^4 + p*b^2 + q*b + r = 0)
  (h3 : c^4 + p*c^2 + q*c + r = 0)
  (h4 : d^4 + p*d^2 + q*d + r = 0) :
  Matrix.det (![
    ![a, b, c, d], 
    ![b, c, d, a], 
    ![c, d, a, b], 
    ![d, a, b, c]
  ]) = 0 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_of_cyclic_matrix_with_polynomial_roots_l320_32036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_four_digit_divisible_by_11_with_two_even_two_odd_l320_32055

/-- A function that checks if a number has two even and two odd digits -/
def hasTwoEvenTwoOddDigits (n : ℕ) : Prop :=
  let digits := n.digits 10
  (digits.filter (fun d => d % 2 = 0)).length = 2 ∧ (digits.filter (fun d => d % 2 ≠ 0)).length = 2

/-- A function that checks if a number is a four-digit number -/
def isFourDigitNumber (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

theorem smallest_four_digit_divisible_by_11_with_two_even_two_odd :
  ∀ n : ℕ, isFourDigitNumber n → hasTwoEvenTwoOddDigits n → n % 11 = 0 → 1469 ≤ n :=
by
  sorry

#check smallest_four_digit_divisible_by_11_with_two_even_two_odd

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_four_digit_divisible_by_11_with_two_even_two_odd_l320_32055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_distance_l320_32092

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line2D where
  slope : ℝ
  yIntercept : ℝ

/-- Represents a parabola in 2D space -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

noncomputable def distance (p1 p2 : Point2D) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

def isOnParabola (p : Point2D) (parab : Parabola) : Prop :=
  (p.y - parab.k)^2 = 4 * parab.a * (p.x - parab.h)

def isPerpendicularTo (l1 l2 : Line2D) : Prop :=
  l1.slope * l2.slope = -1

theorem parabola_focus_distance (parab : Parabola) (F P A : Point2D) (l : Line2D) : 
  parab.a = 4 →
  parab.h = 4 →
  parab.k = 0 →
  F = ⟨4, 0⟩ →
  l.slope = 0 →
  l.yIntercept = -4 →
  isOnParabola P parab →
  A.x = -4 →
  P.y = A.y →
  isPerpendicularTo ⟨(P.y - A.y) / (P.x - A.x), 0⟩ l →
  (F.y - A.y) / (F.x - A.x) = -1 →
  distance P F = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_distance_l320_32092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_l320_32086

/-- Given a train that crosses two platforms of different lengths at different times,
    this theorem proves the length of the train. -/
theorem train_length
  (platform1_length : ℝ) (platform2_length : ℝ)
  (time1 : ℝ) (time2 : ℝ)
  (train_length : ℝ)
  (h1 : platform1_length = 150)
  (h2 : platform2_length = 250)
  (h3 : time1 = 15)
  (h4 : time2 = 20)
  (h5 : (train_length + platform1_length) / time1 = (train_length + platform2_length) / time2)
  : train_length = 150 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_l320_32086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dice_roll_probabilities_l320_32048

def dice_roll := Finset.range 6

def is_square_sum_25 (a b : ℕ) : Bool := a^2 + b^2 = 25

def can_form_isosceles (a b : ℕ) : Bool :=
  a = b || a = 5 || b = 5

theorem dice_roll_probabilities :
  (Finset.filter (λ (p : ℕ × ℕ) => is_square_sum_25 (p.1 + 1) (p.2 + 1)) 
    (dice_roll.product dice_roll)).card / (dice_roll.product dice_roll).card = 1 / 18 ∧
  (Finset.filter (λ (p : ℕ × ℕ) => can_form_isosceles (p.1 + 1) (p.2 + 1)) 
    (dice_roll.product dice_roll)).card / (dice_roll.product dice_roll).card = 7 / 18 :=
by sorry

#eval (Finset.filter (λ (p : ℕ × ℕ) => is_square_sum_25 (p.1 + 1) (p.2 + 1)) 
  (dice_roll.product dice_roll)).card

#eval (Finset.filter (λ (p : ℕ × ℕ) => can_form_isosceles (p.1 + 1) (p.2 + 1)) 
  (dice_roll.product dice_roll)).card

#eval (dice_roll.product dice_roll).card

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dice_roll_probabilities_l320_32048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_are_orthogonal_l320_32084

/-- Two vectors in R³ are orthogonal if their dot product is zero -/
def orthogonal (v w : Fin 3 → ℝ) : Prop :=
  (v 0) * (w 0) + (v 1) * (w 1) + (v 2) * (w 2) = 0

/-- The vectors (1, 2, 5) and (3, -4, 1) are orthogonal -/
theorem vectors_are_orthogonal : 
  orthogonal (![1, 2, 5]) (![3, -4, 1]) := by
  -- Unfold the definition of orthogonal
  unfold orthogonal
  -- Simplify the left-hand side of the equation
  simp
  -- The result is true by computation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_are_orthogonal_l320_32084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_area_decrease_l320_32004

/-- The area of an equilateral triangle with side length s -/
noncomputable def equilateral_triangle_area (s : ℝ) : ℝ := s^2 * Real.sqrt 3 / 4

theorem equilateral_triangle_area_decrease 
  (original_area : ℝ) 
  (side_decrease : ℝ) :
  original_area = 81 * Real.sqrt 3 →
  side_decrease = 6 →
  ∃ (original_side : ℝ),
    equilateral_triangle_area original_side = original_area ∧
    original_area - equilateral_triangle_area (original_side - side_decrease) = 45 * Real.sqrt 3 := by
  intro h_area h_decrease
  let original_side := Real.sqrt (4 * original_area / Real.sqrt 3)
  use original_side
  constructor
  · -- Prove that equilateral_triangle_area original_side = original_area
    sorry
  · -- Prove that original_area - equilateral_triangle_area (original_side - side_decrease) = 45 * Real.sqrt 3
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_area_decrease_l320_32004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_a₀_for_integer_ratio_l320_32098

noncomputable def sequenceA (a₀ : ℚ) : ℕ → ℚ
  | 0 => a₀
  | n + 1 => sequenceA a₀ n + 2 * 3^n

theorem unique_a₀_for_integer_ratio :
  ∃! a₀ : ℚ, ∀ j k : ℕ, 0 < j → j < k →
    ∃ m : ℤ, (sequenceA a₀ k)^j = m * (sequenceA a₀ j)^k :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_a₀_for_integer_ratio_l320_32098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_floor_l320_32051

/-- Represents the transformation of a number after one coin flip -/
noncomputable def transform (x : ℝ) : ℝ := (x + 1 + x⁻¹) / 2

/-- Calculates the expected value after n transformations -/
noncomputable def expectedValue : ℕ → ℝ
  | 0 => 1000
  | n + 1 => transform (expectedValue n)

/-- The main theorem to prove -/
theorem expected_value_floor : 
  ⌊expectedValue 8 / 10⌋ = 13 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_floor_l320_32051
