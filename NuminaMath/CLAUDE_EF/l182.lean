import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_l182_18250

theorem tan_double_angle (α : ℝ) (h : Real.tan (α / 2) = 2) : Real.tan α = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_l182_18250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_1_expression_evaluation_2_l182_18293

-- Part 1
theorem expression_evaluation_1 :
  (1 : ℝ) * (0.001 : ℝ)^(-(1/3) : ℝ) + 27^((2/3) : ℝ) + (1/4 : ℝ)^(-(1/2) : ℝ) - (1/9 : ℝ)^(-(3/2) : ℝ) = -6 :=
by sorry

-- Part 2
theorem expression_evaluation_2 :
  (1/2 : ℝ) * Real.log 25 / Real.log 10 + Real.log 2 / Real.log 10 - Real.log (Real.sqrt 0.1) / Real.log 10 - 
  (Real.log 9 / Real.log 2) * (Real.log 2 / Real.log 3) = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_1_expression_evaluation_2_l182_18293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_distance_l182_18290

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  center : Point
  a : ℝ
  b : ℝ

/-- Checks if a point lies on the ellipse -/
def isOnEllipse (e : Ellipse) (p : Point) : Prop :=
  (p.x - e.center.x)^2 / e.a^2 + (p.y - e.center.y)^2 / e.b^2 = 1

/-- The distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- The main theorem -/
theorem ellipse_intersection_distance (e : Ellipse) (a b p m n : Point) : 
  e.center = ⟨0, 0⟩ → 
  e.a = 2 → 
  e.b = 1 → 
  a = ⟨0, 1⟩ → 
  b = ⟨-2, 0⟩ → 
  isOnEllipse e p → 
  p.x > 0 → 
  p.y < 0 → 
  m.y = 0 → 
  n.x = 0 → 
  distance b m = 9/4 * distance a n → 
  distance m n = Real.sqrt 10 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_distance_l182_18290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_Ca_approx_40_calculated_close_to_given_l182_18232

/-- The molar mass of calcium (Ca) in g/mol -/
noncomputable def molar_mass_Ca : ℝ := 40.08

/-- The molar mass of carbon (C) in g/mol -/
noncomputable def molar_mass_C : ℝ := 12.01

/-- The molar mass of oxygen (O) in g/mol -/
noncomputable def molar_mass_O : ℝ := 16.00

/-- The molar mass of calcium carbonate (CaCO3) in g/mol -/
noncomputable def molar_mass_CaCO3 : ℝ := molar_mass_Ca + molar_mass_C + 3 * molar_mass_O

/-- The mass percentage of calcium (Ca) in calcium carbonate (CaCO3) -/
noncomputable def mass_percentage_Ca : ℝ := (molar_mass_Ca / molar_mass_CaCO3) * 100

/-- Theorem stating that the mass percentage of Ca in CaCO3 is approximately 40% -/
theorem mass_percentage_Ca_approx_40 :
  ∃ ε > 0, |mass_percentage_Ca - 40| < ε :=
by sorry

/-- The given mass percentage of Ca in the problem -/
noncomputable def given_mass_percentage : ℝ := 40

/-- Theorem stating that the calculated mass percentage is close to the given percentage -/
theorem calculated_close_to_given :
  ∃ δ > 0, |mass_percentage_Ca - given_mass_percentage| < δ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_Ca_approx_40_calculated_close_to_given_l182_18232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_speed_is_70_l182_18253

/-- A journey with a change in speed -/
structure Journey where
  D : ℝ  -- Total distance
  T : ℝ  -- Total time
  h1 : D > 0
  h2 : T > 0

/-- The initial speed of the journey -/
noncomputable def initialSpeed (j : Journey) : ℝ := (2 * j.D) / j.T

/-- The speed for the remaining part of the journey -/
noncomputable def remainingSpeed (j : Journey) : ℝ := j.D / (2 * j.T)

/-- Theorem stating the initial speed is 70 kmph given the conditions -/
theorem initial_speed_is_70 (j : Journey) 
  (h3 : remainingSpeed j = 35) : initialSpeed j = 70 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_speed_is_70_l182_18253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_overtake_l182_18251

/-- The time (in minutes) it takes for a faster plane to overtake a slower plane -/
noncomputable def overtake_time (speed_a speed_b : ℝ) (head_start : ℝ) : ℝ :=
  (speed_b * head_start) / (speed_b - speed_a)

/-- The total time (in minutes) from the slower plane's takeoff until overtaken -/
noncomputable def total_time (speed_a speed_b : ℝ) (head_start : ℝ) : ℝ :=
  overtake_time speed_a speed_b head_start + head_start

theorem plane_overtake :
  let speed_a : ℝ := 200 / 60  -- 200 mph converted to miles per minute
  let speed_b : ℝ := 300 / 60  -- 300 mph converted to miles per minute
  let head_start : ℝ := 40     -- 40 minutes head start
  total_time speed_a speed_b head_start = 120 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_overtake_l182_18251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scooter_safety_pad_cost_increase_l182_18282

/-- The percent increase in the combined cost of a scooter and safety pad set -/
theorem scooter_safety_pad_cost_increase :
  let initial_scooter_cost : ℝ := 120
  let initial_safety_pad_cost : ℝ := 30
  let scooter_increase_percent : ℝ := 8
  let safety_pad_increase_percent : ℝ := 15
  let new_scooter_cost : ℝ := initial_scooter_cost * (1 + scooter_increase_percent / 100)
  let new_safety_pad_cost : ℝ := initial_safety_pad_cost * (1 + safety_pad_increase_percent / 100)
  let initial_total_cost : ℝ := initial_scooter_cost + initial_safety_pad_cost
  let new_total_cost : ℝ := new_scooter_cost + new_safety_pad_cost
  let total_increase : ℝ := new_total_cost - initial_total_cost
  let percent_increase : ℝ := (total_increase / initial_total_cost) * 100
  ∃ (ε : ℝ), abs (percent_increase - 9.4) < ε ∧ ε > 0 := by
  sorry

#check scooter_safety_pad_cost_increase

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scooter_safety_pad_cost_increase_l182_18282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l182_18215

-- Define the function f
noncomputable def f (b : ℝ) (x : ℝ) : ℝ := (-2^x + b) / (2^(x+1) + 2)

-- State the theorem
theorem odd_function_properties (b : ℝ) :
  -- f is an odd function
  (∀ x : ℝ, f b (-x) = -(f b x)) →
  -- b = 1
  (b = 1) ∧
  -- f is monotonically decreasing
  (∀ x y : ℝ, x < y → f b x > f b y) ∧
  -- Condition for inequality
  (∀ k : ℝ, (∀ t : ℝ, f b (t^2 - 2*t) + f b (2*t^2 - k) < 0) ↔ k < -1/3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l182_18215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_calculations_l182_18257

theorem arithmetic_calculations : 
  (78 * 4 + 488 = 800) ∧
  (1903 - 475 * 4 = 3) ∧
  (350 * (12 + 342 / 9) = 17500) ∧
  (480 / (125 - 117) = 60) ∧
  ((3600 - 18 * 200) / 253 = 0) ∧
  ((243 - 162) / 27 * 380 = 1140) := by
  apply And.intro
  · norm_num
  apply And.intro
  · norm_num
  apply And.intro
  · norm_num
  apply And.intro
  · norm_num
  apply And.intro
  · norm_num
  · norm_num

#check arithmetic_calculations

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_calculations_l182_18257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2020_equals_quarter_l182_18203

def a : ℕ → ℚ
  | 0 => 1/4
  | n + 1 => 1 - 1 / a n

theorem a_2020_equals_quarter : a 2019 = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2020_equals_quarter_l182_18203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_common_difference_l182_18287

/-- The sum of the first n terms of an arithmetic sequence -/
noncomputable def S (n : ℕ) (a d : ℝ) : ℝ := (n / 2 : ℝ) * (2 * a + (n - 1 : ℝ) * d)

/-- Theorem: If (S12 / 12) - (S10 / 10) = -2 for an arithmetic sequence, then its common difference is -2 -/
theorem arithmetic_sequence_common_difference 
  (a : ℝ) -- first term of the sequence
  (d : ℝ) -- common difference of the sequence
  (h : (S 12 a d / 12) - (S 10 a d / 10) = -2) : 
  d = -2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_common_difference_l182_18287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_store_loss_l182_18260

/-- Represents the sale price of each item -/
noncomputable def sale_price : ℝ := 135

/-- Represents the profit margin for the profitable item -/
noncomputable def profit_margin : ℝ := 0.25

/-- Represents the loss margin for the loss-making item -/
noncomputable def loss_margin : ℝ := 0.25

/-- Calculates the cost price of the profitable item -/
noncomputable def cost_price_profit : ℝ := sale_price / (1 + profit_margin)

/-- Calculates the cost price of the loss-making item -/
noncomputable def cost_price_loss : ℝ := sale_price / (1 - loss_margin)

/-- Theorem stating that the store incurs a loss of 18 yuan -/
theorem store_loss : 
  (sale_price - cost_price_profit) + (sale_price - cost_price_loss) = -18 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_store_loss_l182_18260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_A_B_equals_result_l182_18292

open Finset

def U : Finset ℕ := {0, 1, 2, 3, 4}

def complementA : Finset ℕ := {1, 2}

def B : Finset ℕ := {1, 3}

theorem union_A_B_equals_result :
  let A := U \ complementA
  (A ∪ B) = {0, 1, 3, 4} := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_A_B_equals_result_l182_18292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_and_sin_x_l182_18281

noncomputable def m (x : Real) : Real × Real := (Real.cos (x/2), -1)

noncomputable def n (x : Real) : Real × Real := (Real.sqrt 3 * Real.sin (x/2), Real.cos (x/2) ^ 2)

def dot_product (v w : Real × Real) : Real :=
  v.1 * w.1 + v.2 * w.2

noncomputable def f (x : Real) : Real :=
  dot_product (m x) (n x) + 1

theorem min_value_and_sin_x :
  (∀ x ∈ Set.Icc (Real.pi / 2) Real.pi, f x ≥ 1 ∧ f Real.pi = 1) ∧
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x = 11/10 → Real.sin x = (3 * Real.sqrt 3 + 4) / 10) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_and_sin_x_l182_18281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l182_18221

/-- Given a hyperbola with equation x^2 - y^2/m = 1, its eccentricity is greater than √2 if and only if m > 1 -/
theorem hyperbola_eccentricity (m : ℝ) :
  (∀ x y : ℝ, x^2 - y^2/m = 1 → ∃ e : ℝ, e = Real.sqrt (1 + m) ∧ e > Real.sqrt 2) ↔ m > 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l182_18221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_operation_l182_18261

/-- The correct operation that should have been applied to x -/
noncomputable def f (x : ℝ) : ℝ := 10 * x

/-- The erroneous operation that was actually applied to x -/
noncomputable def g (x : ℝ) : ℝ := x / 10

theorem correct_operation (x : ℝ) (h : x ≠ 0) : 
  g x = (1 - 0.99) * f x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_operation_l182_18261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_2_simplest_l182_18228

-- Define the square roots
noncomputable def a : ℝ := Real.sqrt 0.1
noncomputable def b : ℝ := 1 / Real.sqrt 2
noncomputable def c : ℝ := Real.sqrt 2
noncomputable def d : ℝ := Real.sqrt 8

-- Define a function to check if a number is in its simplest form
def is_simplest_form (x : ℝ) : Prop :=
  ∀ y : ℝ, y ≠ x → (Real.sqrt y = x → ¬(∃ q : ℚ, ↑q = y)) ∧
                   (y / Real.sqrt (y * y) = x → ¬(∃ q : ℚ, ↑q = y))

-- Theorem statement
theorem sqrt_2_simplest : 
  is_simplest_form c ∧ 
  ¬(is_simplest_form a) ∧ 
  ¬(is_simplest_form b) ∧ 
  ¬(is_simplest_form d) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_2_simplest_l182_18228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_cosine_intersection_l182_18214

-- Define the cosine function
noncomputable def cosine_graph : ℝ → ℝ := λ x ↦ Real.cos x

-- Define a circle in 2D plane
def circle_set (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Theorem statement
theorem circle_cosine_intersection :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    ∃ (points : Finset (ℝ × ℝ)),
      (∀ p ∈ points, p ∈ circle_set center radius ∧ p.2 = cosine_graph p.1) ∧
      points.card > 16 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_cosine_intersection_l182_18214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_congruent_faces_l182_18243

/-- The volume of a tetrahedron with congruent faces -/
noncomputable def tetrahedron_volume (a b c : ℝ) : ℝ :=
  (1 / 24) * Real.sqrt ((a^2 + b^2 - c^2) * (a^2 + c^2 - b^2) * (b^2 + c^2 - a^2))

/-- Theorem: The volume of a tetrahedron with congruent faces, each having side lengths a, b, and c -/
theorem tetrahedron_volume_congruent_faces (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃ (V : ℝ), V = tetrahedron_volume a b c ∧ V > 0 := by
  -- We use 'sorry' to skip the proof for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_congruent_faces_l182_18243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_everett_worked_four_weeks_l182_18297

/-- Represents the work schedule and pay information for Everett --/
structure WorkInfo where
  weekdayHours : ℚ
  weekendHours : ℚ
  totalHours : ℚ
  overtimePay : ℚ
  normalRate : ℚ

/-- Calculates the number of weeks worked based on the given work information --/
noncomputable def weeksWorked (info : WorkInfo) : ℚ :=
  let overtimeRate := 2 * info.normalRate
  let overtimeHours := info.overtimePay / overtimeRate
  let regularHours := info.totalHours - overtimeHours
  let weeklyHours := 5 * info.weekdayHours + 2 * info.weekendHours
  regularHours / weeklyHours

/-- Theorem stating that given Everett's work information, he worked approximately 4 weeks --/
theorem everett_worked_four_weeks (info : WorkInfo)
  (h1 : info.weekdayHours = 5)
  (h2 : info.weekendHours = 6)
  (h3 : info.totalHours = 140)
  (h4 : info.overtimePay = 300)
  (h5 : info.normalRate = 15) :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1/2 ∧ |weeksWorked info - 4| < ε :=
by
  sorry

-- This evaluation will not work due to noncomputable definition
-- #eval weeksWorked ⟨5, 6, 140, 300, 15⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_everett_worked_four_weeks_l182_18297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_interval_l182_18254

-- Define the function
noncomputable def f (x : ℝ) : ℝ := 2^(-x^2 + 2*x)

-- Define the monotonicity property
def is_monotone_decreasing_on (f : ℝ → ℝ) (s : Set ℝ) :=
  ∀ {x y}, x ∈ s → y ∈ s → x ≤ y → f y ≤ f x

-- State the theorem
theorem f_monotone_decreasing_interval :
  is_monotone_decreasing_on f (Set.Ici 1) := by
  sorry

#check f_monotone_decreasing_interval

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_interval_l182_18254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factors_of_M_l182_18208

theorem factors_of_M (M : Nat) : 
  M = 2^5 * 3^3 * 5^2 * 7^1 → 
  (Finset.filter (fun n => M % n = 0) (Finset.range (M + 1))).card = 144 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factors_of_M_l182_18208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_cylinder_radius_l182_18269

structure Crate where
  length : ℝ
  width : ℝ
  height : ℝ

structure Cylinder where
  radius : ℝ
  height : ℝ

def fits_in_crate (cylinder : Cylinder) (crate : Crate) : Prop :=
  2 * cylinder.radius ≤ min crate.length crate.width ∧ cylinder.height ≤ crate.height

noncomputable def cylinder_volume (cylinder : Cylinder) : ℝ :=
  Real.pi * cylinder.radius^2 * cylinder.height

theorem largest_cylinder_radius (crate : Crate) :
  crate.length = 2 ∧ crate.width = 8 ∧ crate.height = 12 →
  ∃ (max_cylinder : Cylinder),
    fits_in_crate max_cylinder crate ∧
    max_cylinder.radius = 1 ∧
    ∀ (c : Cylinder), fits_in_crate c crate →
      cylinder_volume c ≤ cylinder_volume max_cylinder :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_cylinder_radius_l182_18269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_student_work_hours_is_25_student_earnings_sufficient_l182_18258

/-- A student's work schedule optimized for earnings and study time --/
def student_work_hours : ℚ :=
  let library_wage : ℚ := 8
  let construction_wage : ℚ := 15
  let min_weekly_earnings : ℚ := 300
  let library_hours : ℚ := 10
  let library_earnings : ℚ := library_wage * library_hours
  let remaining_earnings : ℚ := min_weekly_earnings - library_earnings
  let construction_hours : ℚ := (remaining_earnings / construction_wage).ceil
  library_hours + construction_hours

#eval student_work_hours

/-- Proof that the student needs to work 25 hours per week --/
theorem student_work_hours_is_25 : student_work_hours = 25 := by
  sorry

/-- Proof that the student earns at least $300 per week --/
theorem student_earnings_sufficient :
  let library_wage : ℚ := 8
  let construction_wage : ℚ := 15
  let min_weekly_earnings : ℚ := 300
  library_wage * 10 + construction_wage * (student_work_hours - 10) ≥ min_weekly_earnings := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_student_work_hours_is_25_student_earnings_sufficient_l182_18258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_f_l182_18273

noncomputable def f (x : ℝ) : ℝ := |Real.tan (x - 2011)|

theorem min_positive_period_of_f : ∃ p : ℝ, p > 0 ∧ 
  (∀ x : ℝ, f (x + p) = f x) ∧ 
  (∀ q : ℝ, q > 0 ∧ (∀ x : ℝ, f (x + q) = f x) → p ≤ q) ∧ 
  p = Real.pi :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_f_l182_18273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equals_negative_two_sin_four_angle_condition_l182_18265

-- Define the expression
noncomputable def expression (x : ℝ) : ℝ := 2 * Real.sqrt (1 - Real.sin x) + Real.sqrt (2 + 2 * Real.cos x)

-- State the theorem
theorem expression_equals_negative_two_sin_four :
  expression 8 = -2 * Real.sin 4 :=
by
  sorry

-- Define the condition
theorem angle_condition : π < (5 * π) / 4 ∧ (5 * π) / 4 < 4 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equals_negative_two_sin_four_angle_condition_l182_18265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_luisa_distance_back_home_l182_18231

/-- Represents the distance driven to each location in miles -/
structure DistanceDriven where
  grocery : ℚ
  mall : ℚ
  petStore : ℚ

/-- Represents the parameters of the problem -/
structure ProblemParameters where
  distanceDriven : DistanceDriven
  milesPerGallon : ℚ
  costPerGallon : ℚ
  totalCost : ℚ

/-- Calculates the distance driven back home given the problem parameters -/
def distanceDrivenBackHome (params : ProblemParameters) : ℚ :=
  let totalMilesDriven := params.milesPerGallon * (params.totalCost / params.costPerGallon)
  totalMilesDriven - (params.distanceDriven.grocery + params.distanceDriven.mall + params.distanceDriven.petStore)

/-- Theorem stating that given the problem parameters, Luisa drove 9 miles back home -/
theorem luisa_distance_back_home :
  let params : ProblemParameters := {
    distanceDriven := { grocery := 10, mall := 6, petStore := 5 },
    milesPerGallon := 15,
    costPerGallon := 7/2,
    totalCost := 7
  }
  distanceDrivenBackHome params = 9 := by
  sorry

#eval distanceDrivenBackHome {
  distanceDriven := { grocery := 10, mall := 6, petStore := 5 },
  milesPerGallon := 15,
  costPerGallon := 7/2,
  totalCost := 7
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_luisa_distance_back_home_l182_18231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_correct_l182_18299

/-- The focus of the parabola y = 4x^2 -/
noncomputable def parabola_focus : ℝ × ℝ := (0, 1/16)

/-- The equation of the parabola -/
def parabola_equation (x y : ℝ) : Prop := y = 4 * x^2

theorem parabola_focus_correct :
  let (f_x, f_y) := parabola_focus
  -- The focus is on the y-axis
  f_x = 0 ∧
  -- For any point (x, y) on the parabola
  ∀ x y : ℝ, parabola_equation x y →
    -- The distance from (x, y) to the focus is equal to
    -- the distance from (x, y) to the directrix y = -f_y
    (x - f_x)^2 + (y - f_y)^2 = (y + f_y)^2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_correct_l182_18299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nabla_calculation_l182_18285

-- Define the nabla operation
noncomputable def nabla (a b : ℝ) : ℝ := (a + b) / (1 + a * b)

-- Theorem statement
theorem nabla_calculation :
  nabla (nabla (nabla 2 3) 4) 5 = 7/8 :=
by
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_nabla_calculation_l182_18285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_is_zero_l182_18202

/-- Represents a digit in base 8 --/
def Base8Digit := Fin 8

/-- Represents the addition problem in base 8 --/
structure Base8Addition where
  a : Base8Digit × Base8Digit × Base8Digit × Base8Digit
  b : Base8Digit × Base8Digit × Base8Digit
  c : Base8Digit × Base8Digit
  result : Base8Digit × Base8Digit × Base8Digit × Base8Digit

/-- The specific addition problem given --/
def givenAddition : Base8Addition := {
  a := (⟨5, by norm_num⟩, ⟨4, by norm_num⟩, ⟨3, by norm_num⟩, ⟨0, by norm_num⟩)
  b := (⟨0, by norm_num⟩, ⟨6, by norm_num⟩, ⟨7, by norm_num⟩)
  c := (⟨0, by norm_num⟩, ⟨4, by norm_num⟩)
  result := (⟨6, by norm_num⟩, ⟨5, by norm_num⟩, ⟨0, by norm_num⟩, ⟨3, by norm_num⟩)
}

/-- Function to check if the addition is valid in base 8 --/
def isValidBase8Addition (add : Base8Addition) : Prop :=
  -- Implementation details omitted
  sorry

/-- Theorem stating that the given addition problem is valid when ⬜ = 0 --/
theorem square_is_zero (add : Base8Addition) (h : add = givenAddition) :
  isValidBase8Addition add := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_is_zero_l182_18202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_W_555_2_mod_1000_l182_18224

/-- Recursive definition of W(n,k) -/
def W : ℕ+ → ℕ → ℕ
  | n, 0 => n.val ^ n.val
  | n, k + 1 => W ⟨W n k, by sorry⟩ k

/-- Theorem: W(555,2) is congruent to 375 modulo 1000 -/
theorem W_555_2_mod_1000 :
  W ⟨555, by norm_num⟩ 2 ≡ 375 [ZMOD 1000] := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_W_555_2_mod_1000_l182_18224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_pyramid_cross_section_l182_18239

/-- The area of a cross-section that divides a truncated pyramid into two equal volumes -/
noncomputable def cross_section_area (A B : ℝ) : ℝ :=
  ((A * Real.sqrt A + B * Real.sqrt B) / 2) ^ (2/3)

/-- Theorem: For a truncated pyramid with base area 8 cm² and top area 1 cm²,
    the area of the cross-section that divides it into two equal volumes is approximately 5.19 cm² -/
theorem truncated_pyramid_cross_section :
  let A : ℝ := 8
  let B : ℝ := 1
  let C : ℝ := cross_section_area A B
  ‖C - 5.19‖ < 0.01 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_pyramid_cross_section_l182_18239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_squared_range_l182_18205

theorem cos_sin_squared_range :
  ∀ x : ℝ, (3 / 4 : ℝ) ≤ Real.cos x - (Real.sin x)^2 + 2 ∧
  (∃ x₁ : ℝ, Real.cos x₁ - (Real.sin x₁)^2 + 2 = 3 / 4) ∧
  (∃ x₂ : ℝ, Real.cos x₂ - (Real.sin x₂)^2 + 2 = 3) ∧
  ∀ y : ℝ, (∃ x : ℝ, y = Real.cos x - (Real.sin x)^2 + 2) → 3 / 4 ≤ y ∧ y ≤ 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_squared_range_l182_18205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_container_l182_18262

-- Define the total length of the steel strip
def total_length : ℝ := 14.8

-- Define the function for the volume of the container
def volume (x : ℝ) : ℝ := x * (x + 0.5) * (3.45 - x)

-- Define the height function
def container_height (x : ℝ) : ℝ := 3.45 - x

-- State the theorem
theorem max_volume_container :
  ∃ (x : ℝ), 
    0 < x ∧ 
    x < 3.45 ∧ 
    (∀ y, 0 < y ∧ y < 3.45 → volume y ≤ volume x) ∧
    container_height x = 2.45 ∧
    volume x = 3.675 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_container_l182_18262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_coordinates_l182_18246

noncomputable def equal_angles (a : ℝ × ℝ × ℝ) : Prop :=
  let (x, y, z) := a
  x^2 = y^2 ∧ y^2 = z^2

noncomputable def magnitude (a : ℝ × ℝ × ℝ) : ℝ :=
  let (x, y, z) := a
  Real.sqrt (x^2 + y^2 + z^2)

theorem vector_coordinates (a : ℝ × ℝ × ℝ) :
  equal_angles a → magnitude a = Real.sqrt 3 →
  a = (1, 1, 1) ∨ a = (-1, -1, -1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_coordinates_l182_18246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_odd_integers_sum_l182_18268

/-- Two consecutive odd integers where one is 61 have a sum of 124. -/
theorem consecutive_odd_integers_sum (n : ℤ) : 
  (n % 2 = 1 ∧ n = 61) → (n + (n + 2) = 124) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_odd_integers_sum_l182_18268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_choir_size_l182_18227

/-- Represents the number of choir members -/
def n (x c : ℕ) : ℕ := 
  if x * x + 11 = c * (c + 5) then x * x + 11 else 0

/-- The maximum number of choir members satisfying the conditions -/
def max_choir_members : ℕ := 150

/-- Theorem stating that 150 is the maximum number of choir members satisfying the conditions -/
theorem max_choir_size : 
  (∃ x c : ℕ, n x c = max_choir_members) ∧ 
  (∀ m : ℕ, m > max_choir_members → ¬∃ x c : ℕ, n x c = m) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_choir_size_l182_18227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_bound_l182_18230

theorem polynomial_bound (n : ℕ) (p : ℝ → ℝ) :
  (∀ x, ∃ (q : Polynomial ℝ), (q.eval x = p x) ∧ (q.natDegree ≤ 2*n)) →
  (∀ k : ℤ, k ∈ Finset.Icc (-n : ℤ) n → |p k| ≤ 1) →
  ∀ x ∈ Set.Icc (-n : ℝ) n, |p x| ≤ 2^(2*n) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_bound_l182_18230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_specific_lines_l182_18267

/-- The distance between two parallel lines in ℝ² --/
noncomputable def distance_parallel_lines (a b d : ℝ × ℝ) : ℝ :=
  let v := (b.1 - a.1, b.2 - a.2)
  let proj_v_d := ((v.1 * d.1 + v.2 * d.2) / (d.1 * d.1 + d.2 * d.2)) • d
  let w := (v.1 - proj_v_d.1, v.2 - proj_v_d.2)
  Real.sqrt (w.1 * w.1 + w.2 * w.2)

/-- Theorem: The distance between the given parallel lines is 33/13 --/
theorem distance_specific_lines :
  distance_parallel_lines (4, 1) (5, 4) (2, -3) = 33 / 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_specific_lines_l182_18267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_in_range_l182_18216

theorem count_integers_in_range : 
  (Finset.filter (fun x : ℕ => 24 ≤ (x : ℝ)^2 + 4*(x : ℝ) + 4 ∧ (x : ℝ)^2 + 4*(x : ℝ) + 4 ≤ 64) (Finset.range 7)).card = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_in_range_l182_18216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tourists_same_side_intervals_l182_18219

noncomputable def motion_tourist1 (t : ℝ) : ℝ := Real.sqrt (1 + 6 * t) - 1

noncomputable def motion_tourist2 (t : ℝ) : ℝ := 
  if t ≥ 1/6 then 6 * (t - 1/6) else 0

def same_side (t : ℝ) : Prop :=
  (motion_tourist1 t ≤ 2 ∧ motion_tourist2 t ≤ 2) ∨ 
  (motion_tourist1 t ≥ 2 ∧ motion_tourist2 t ≥ 2)

theorem tourists_same_side_intervals :
  ∀ t : ℝ, same_side t ↔ t ∈ Set.Ici 0 ∩ Set.Iic (1/2) ∪ Set.Ici (4/3) :=
by
  sorry

#check tourists_same_side_intervals

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tourists_same_side_intervals_l182_18219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_bound_l182_18276

/-- A type representing a closed segment on a line -/
structure Segment where
  start : ℝ
  stop : ℝ
  is_valid : start ≤ stop

/-- A type representing a set of disjoint segments -/
structure DisjointSegments where
  segments : List Segment
  are_disjoint : ∀ i j, i ≠ j → 
    (segments.get? i).map (λ s₁ => (segments.get? j).map (λ s₂ => 
      s₁.stop < s₂.start ∨ s₂.stop < s₁.start)) = some (some True)

/-- A function that returns the intersection of a list of DisjointSegments -/
noncomputable def intersection (sets : List DisjointSegments) : DisjointSegments :=
  sorry

/-- The main theorem -/
theorem intersection_bound (sets : List DisjointSegments) :
  (sets.length = 100) →
  (∀ s ∈ sets, s.segments.length = 100) →
  (intersection sets).segments.length ≤ 9901 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_bound_l182_18276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l182_18223

noncomputable def f (x : ℝ) : ℝ := (x^4 - 4*x^3 + 6*x^2 - 4*x + 1) / (x^2 - 9)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x < -3 ∨ (-3 < x ∧ x < 3) ∨ 3 < x} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l182_18223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_distance_range_l182_18234

/-- The ellipse with equation x^2/2 + y^2 = 1 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / 2) + p.2^2 = 1}

/-- The center of the ellipse -/
def O : ℝ × ℝ := (0, 0)

/-- The left focus of the ellipse -/
def F : ℝ × ℝ := (-1, 0)

/-- The squared distance between two points -/
def dist_squared (p q : ℝ × ℝ) : ℝ :=
  (p.1 - q.1)^2 + (p.2 - q.2)^2

theorem ellipse_distance_range :
  ∀ P ∈ Ellipse, 2 ≤ dist_squared O P + dist_squared P F ∧
                 dist_squared O P + dist_squared P F ≤ 5 + 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_distance_range_l182_18234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_properties_l182_18288

-- Define the propositions
def p : Prop := ∀ a b : ℝ × ℝ, (a.1 * b.1 + a.2 * b.2 > 0) → 
  (Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))) < Real.pi / 2)

def q : Prop := ∀ a x : ℝ, a < -1 → a^2 * x^2 - 2 * x + 1 > 0

-- State the theorem
theorem proposition_properties : (p ∨ q) ∧ ¬(p ∧ q) ∧ ¬p := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_properties_l182_18288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_B_length_l182_18291

/-- Calculates the length of a train given the parameters of two trains passing each other. -/
noncomputable def train_length_calculation (length_A : ℝ) (speed_A : ℝ) (speed_B : ℝ) (crossing_time : ℝ) : ℝ :=
  let relative_speed := (speed_A + speed_B) * (1000 / 3600)  -- Convert km/h to m/s
  let total_distance := relative_speed * crossing_time
  total_distance - length_A

/-- Theorem stating the length of Train B given the specified conditions. -/
theorem train_B_length :
  let length_A : ℝ := 90
  let speed_A : ℝ := 120
  let speed_B : ℝ := 80
  let crossing_time : ℝ := 9
  train_length_calculation length_A speed_A speed_B crossing_time = 410 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval train_length_calculation 90 120 80 9

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_B_length_l182_18291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_properties_l182_18236

theorem angle_properties (α β : Real) 
  (h1 : π < α ∧ α < 3*π/2)
  (h2 : π < β ∧ β < 3*π/2)
  (h3 : Real.sin α = -Real.sqrt 5 / 5)
  (h4 : Real.cos β = -Real.sqrt 10 / 10) :
  (α - β = -π/4) ∧ (Real.tan (2*α - β) = -1/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_properties_l182_18236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_alcohol_percentage_l182_18222

/-- Represents an alcohol mixture --/
structure AlcoholMixture where
  volume : ℝ
  percentage : ℝ

/-- Calculates the volume of pure alcohol in a mixture --/
noncomputable def pureAlcoholVolume (mixture : AlcoholMixture) : ℝ :=
  mixture.volume * (mixture.percentage / 100)

/-- Theorem: The final alcohol percentage is 45% --/
theorem final_alcohol_percentage
  (mixture30 : AlcoholMixture)
  (mixture50 : AlcoholMixture)
  (h1 : mixture30.percentage = 30)
  (h2 : mixture50.percentage = 50)
  (h3 : mixture30.volume = 2.5)
  (h4 : mixture30.volume + mixture50.volume = 10) :
  let finalMixture : AlcoholMixture := {
    volume := 10,
    percentage := (pureAlcoholVolume mixture30 + pureAlcoholVolume mixture50) / 10 * 100
  }
  finalMixture.percentage = 45 := by
  sorry

#check final_alcohol_percentage

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_alcohol_percentage_l182_18222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_inequality_l182_18272

noncomputable def a : ℕ → ℝ
  | 0 => Real.sqrt 2 / 2
  | n + 1 => Real.sqrt 2 / 2 * Real.sqrt (1 - Real.sqrt (1 - a n ^ 2))

noncomputable def b : ℕ → ℝ
  | 0 => 1
  | n + 1 => (Real.sqrt (1 + b n ^ 2) - 1) / b n

theorem sequence_inequality (n : ℕ) : 2^(n+2) * a n < Real.pi ∧ Real.pi < 2^(n+2) * b n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_inequality_l182_18272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_roots_of_equation_l182_18277

-- Define the equation
def equation (a b c x : ℂ) : Prop :=
  a * x^2 + b * Complex.abs x + c = 0

-- State the theorem
theorem max_roots_of_equation :
  ∀ (a b c : ℝ), a ≠ 0 →
  ∃ (n : ℕ), n ≤ 8 ∧
  (∀ (S : Finset ℂ), (∀ x ∈ S, equation a b c x) → S.card ≤ n) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_roots_of_equation_l182_18277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x3y3_is_40_l182_18200

/-- The coefficient of x³y³ in the expansion of (x+y)(2x-y)⁵ -/
def coefficient_x3y3 : ℤ :=
  let binomial_coeff (n k : ℕ) := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))
  let term1 : ℤ := (binomial_coeff 5 3 * 2^2 * (-1)^3 : ℤ)
  let term2 : ℤ := (binomial_coeff 5 2 * 2^3 * (-1)^2 : ℤ)
  (-term1 + term2).natAbs

theorem coefficient_x3y3_is_40 : coefficient_x3y3 = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x3y3_is_40_l182_18200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_product_not_unique_l182_18235

-- Define a point in a plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a line in a plane
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Function to calculate the distance from a point to a line
noncomputable def distancePointToLine (p : Point) (l : Line) : ℝ :=
  (abs (l.a * p.x + l.b * p.y + l.c)) / Real.sqrt (l.a^2 + l.b^2)

-- Function to calculate the angle between three points
noncomputable def angle (A B C : Point) : ℝ :=
  sorry -- Definition of angle calculation

-- Theorem statement
theorem max_distance_product_not_unique (A B C : Point) :
  ∃ (l1 l2 : Line), l1 ≠ l2 ∧
    (∀ (l : Line), l.a * C.x + l.b * C.y + l.c = 0 →
      distancePointToLine A l * distancePointToLine B l ≤
      distancePointToLine A l1 * distancePointToLine B l1) ∧
    (distancePointToLine A l1 * distancePointToLine B l1 =
     distancePointToLine A l2 * distancePointToLine B l2) ↔
  angle A C B = Real.pi / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_product_not_unique_l182_18235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_ages_l182_18211

/- Define Jessica's and Thomas's ages -/
variable (jessica_age : ℝ)
variable (thomas_age : ℝ)

/- Jessica is 8 years older than Thomas -/
axiom age_difference : jessica_age = thomas_age + 8

/- In 5 years, Jessica will be three times as old as Thomas was 2 years ago -/
axiom future_age_relation : jessica_age + 5 = 3 * (thomas_age - 2)

/- Theorem: The sum of their current ages is 27 -/
theorem sum_of_ages : jessica_age + thomas_age = 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_ages_l182_18211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_implies_conditions_l182_18271

theorem divisibility_implies_conditions (n m k l : ℕ) (h_n : n ≠ 1) 
  (h_div : (n^k + m*n^l + 1) ∣ (n^(k+l) - 1)) :
  (m = 1 ∧ l = 2*k) ∨ (l ∣ k ∧ m = (n^(k-l) - 1) / (n^l - 1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_implies_conditions_l182_18271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_finite_value_sequences_l182_18264

-- Define the function f
def f (x : ℝ) : ℝ := 4 * x - x^2

-- Define the sequence x_n
def seq (x₀ : ℝ) : ℕ → ℝ
  | 0 => x₀
  | n + 1 => f (seq x₀ n)

-- Define a predicate for sequences with finite distinct values
def has_finite_distinct_values (x₀ : ℝ) : Prop :=
  ∃ (S : Finset ℝ), ∀ n, seq x₀ n ∈ S

-- State the theorem
theorem infinitely_many_finite_value_sequences :
  ∃ (S : Set ℝ), (S.Infinite) ∧ (∀ x₀ ∈ S, x₀ ∈ Set.Icc 0 4 ∧ has_finite_distinct_values x₀) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_finite_value_sequences_l182_18264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_capacity_is_2000_l182_18213

/-- Represents the filling process of a tank with three pipes -/
structure TankFilling where
  pipe_a_rate : ℚ  -- filling rate of Pipe A in L/min
  pipe_b_rate : ℚ  -- filling rate of Pipe B in L/min
  pipe_c_rate : ℚ  -- draining rate of Pipe C in L/min
  cycle_time : ℚ   -- time for one complete cycle in minutes
  total_time : ℚ   -- total time to fill the tank in minutes

/-- Calculates the net amount of water added in one cycle -/
def net_water_per_cycle (tf : TankFilling) : ℚ :=
  tf.pipe_a_rate * 1 + tf.pipe_b_rate * 2 - tf.pipe_c_rate * 2

/-- Calculates the total capacity of the tank -/
def tank_capacity (tf : TankFilling) : ℚ :=
  (tf.total_time / tf.cycle_time) * net_water_per_cycle tf

/-- Theorem stating that under given conditions, the tank capacity is 2000 liters -/
theorem tank_capacity_is_2000 (tf : TankFilling) 
  (h1 : tf.pipe_a_rate = 200)
  (h2 : tf.pipe_b_rate = 50)
  (h3 : tf.pipe_c_rate = 25)
  (h4 : tf.cycle_time = 5)
  (h5 : tf.total_time = 40) :
  tank_capacity tf = 2000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_capacity_is_2000_l182_18213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cowboy_shortest_path_l182_18245

/-- Represents a 2D point with x and y coordinates -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the Euclidean distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- The shortest path for the cowboy's journey -/
noncomputable def shortest_path (cowboy_start cabin creek_crossing : Point) : ℝ :=
  distance cowboy_start creek_crossing +
  distance creek_crossing cabin +
  distance cabin creek_crossing +
  distance creek_crossing cowboy_start

theorem cowboy_shortest_path :
  let cowboy_start : Point := ⟨0, -5⟩
  let cabin : Point := ⟨-5, 5⟩
  let creek_crossing : Point := ⟨0, 0⟩
  shortest_path cowboy_start cabin creek_crossing = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cowboy_shortest_path_l182_18245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l182_18242

-- Define the function f
noncomputable def f (x : ℝ) := (1/4 : ℝ)^x - 3 * (1/2 : ℝ)^x + 2

-- State the theorem
theorem f_range :
  ∀ y ∈ Set.range f,
  (-1/4 : ℝ) ≤ y ∧ y ≤ 6 ∧
  (∃ x : ℝ, -2 ≤ x ∧ x ≤ 2 ∧ f x = y) ∧
  (∃ x₁ x₂ : ℝ, -2 ≤ x₁ ∧ x₁ ≤ 2 ∧ -2 ≤ x₂ ∧ x₂ ≤ 2 ∧ f x₁ = (-1/4 : ℝ) ∧ f x₂ = 6) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l182_18242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_strategy_removes_twenty_percent_l182_18249

/-- Represents Van Helsing's vampire hunting scenario -/
structure VampireHunt where
  vampire_pay : ℕ := 5
  werewolf_pay : ℕ := 10
  total_nights : ℕ := 7
  vampires_per_night : ℕ := 2
  werewolves_per_night : ℕ := 1
  vampires_removed : ℚ
  werewolves_removed : ℕ
  total_earnings : ℕ
  initial_werewolf_ratio : ℕ := 4

/-- Calculates the percentage of werewolves removed -/
def werewolves_removed_percentage (hunt : VampireHunt) : ℚ :=
  (hunt.werewolves_removed : ℚ) / (hunt.initial_werewolf_ratio * hunt.vampires_removed * 2) * 100

/-- Theorem stating that the optimal strategy removes 20% of werewolves -/
theorem optimal_strategy_removes_twenty_percent (hunt : VampireHunt)
  (h1 : hunt.vampires_removed = 5)
  (h2 : hunt.werewolves_removed = 8)
  (h3 : hunt.total_earnings = hunt.vampire_pay * hunt.vampires_removed + hunt.werewolf_pay * hunt.werewolves_removed)
  (h4 : hunt.total_earnings = 105) :
  werewolves_removed_percentage hunt = 20 := by
  sorry

def example_hunt : VampireHunt := {
  vampire_pay := 5,
  werewolf_pay := 10,
  total_nights := 7,
  vampires_per_night := 2,
  werewolves_per_night := 1,
  vampires_removed := 5,
  werewolves_removed := 8,
  total_earnings := 105,
  initial_werewolf_ratio := 4
}

#eval werewolves_removed_percentage example_hunt

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_strategy_removes_twenty_percent_l182_18249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_two_greater_than_one_l182_18241

theorem only_two_greater_than_one : 
  ∀ x : ℝ, x ∈ ({0, 2, -1, -3} : Set ℝ) → (x > 1 ↔ x = 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_two_greater_than_one_l182_18241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_sequence_property_l182_18252

/-- Represents the sequence of digits formed by writing natural numbers in ascending order starting from 1 -/
def digitSequence : ℕ → ℕ := sorry

/-- Returns the nth digit in the digitSequence -/
def nthDigit (n : ℕ) : ℕ := digitSequence n

theorem digit_sequence_property :
  (nthDigit 100 = 5) ∧ (nthDigit 1000 = 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_sequence_property_l182_18252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_M_and_N_l182_18283

def M : Set ℝ := {x | -1/2 < x ∧ x < 1/2}
def N : Set ℝ := {x | x^2 ≤ x}

theorem intersection_of_M_and_N : M ∩ N = Set.Icc 0 (1/2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_M_and_N_l182_18283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_categorization_l182_18275

noncomputable def given_numbers : List ℚ := [83/10, -4, -8/10, -1/5, 9/10, -34/3, 24]

def is_negative (x : ℚ) : Prop := x < 0
def is_integer (x : ℚ) : Prop := ∃ n : ℤ, x = n
def is_negative_integer (x : ℚ) : Prop := is_negative x ∧ is_integer x
def is_positive_fraction (x : ℚ) : Prop := x > 0 ∧ ¬(is_integer x)

theorem number_categorization :
  (∀ x ∈ [-4, -8/10, -1/5, -34/3], is_negative x) ∧
  (∀ x ∈ [-4, 24], is_integer x) ∧
  (∀ x ∈ [-4], is_negative_integer x) ∧
  (∀ x ∈ [83/10, 9/10], is_positive_fraction x) :=
by
  sorry

#check number_categorization

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_categorization_l182_18275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_problem_l182_18233

theorem remainder_problem (x : ℕ) (h : (13 * x) % 31 = 3) : (17 + x) % 31 = 22 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_problem_l182_18233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_proof_l182_18247

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := sin x - cos x

-- Define the domain of x
def domain (x : ℝ) : Prop := x > -π/2 ∧ x < π/2

-- Define the tangent line equation
def tangent_line (x y : ℝ) : Prop := x - y - 1 = 0

-- State the theorem
theorem tangent_line_proof :
  ∃ (x₀ y₀ : ℝ), domain x₀ ∧ 
  f x₀ = y₀ ∧
  (deriv f x₀ = 1) ∧
  tangent_line x₀ y₀ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_proof_l182_18247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_F_equals_closed_form_l182_18263

/-- The constant α defined as (3 - √5) / 2 -/
noncomputable def α : ℝ := (3 - Real.sqrt 5) / 2

/-- The function f(n) defined as ⌊αn⌋ -/
noncomputable def f (n : ℕ) : ℤ := ⌊α * n⌋

/-- The k-th iterate of F -/
def F : ℕ → ℕ
| 0 => 1
| 1 => 3
| (k+2) => 3 * F (k+1) - F k

/-- The closed form expression for F(k) -/
noncomputable def F_closed (k : ℕ) : ℝ := 
  (1 / Real.sqrt 5) * ((3 + Real.sqrt 5) / 2) ^ (k + 1) - 
  (1 / Real.sqrt 5) * ((3 - Real.sqrt 5) / 2) ^ (k + 1)

/-- Theorem stating that F(k) equals the closed form expression -/
theorem F_equals_closed_form : ∀ k : ℕ, (F k : ℝ) = F_closed k := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_F_equals_closed_form_l182_18263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l182_18286

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then 2 * x - 4 else -2 * x - 4

-- State the theorem
theorem solution_set_of_inequality :
  (∀ x : ℝ, f x = f (-x)) →  -- f is even
  (∀ x : ℝ, x ≥ 0 → f x = 2 * x - 4) →  -- f(x) = 2x - 4 for x ≥ 0
  {x : ℝ | f (x - 2) > 0} = {x : ℝ | x < 0 ∨ x > 4} :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l182_18286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_to_exp_conversion_l182_18240

theorem log_to_exp_conversion (a : ℝ) : 
  Real.log 20 / Real.log 5 = a ↔ (5 : ℝ)^a = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_to_exp_conversion_l182_18240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_early_arrival_l182_18207

/-- A boy's journey to school -/
noncomputable def school_journey (usual_time : ℝ) (rate_increase : ℝ) : ℝ :=
  usual_time * (1 / rate_increase)

/-- Theorem: Boy arrives 4 minutes early when increasing walking rate -/
theorem early_arrival (usual_time : ℝ) (rate_increase : ℝ) 
  (h1 : usual_time = 36)
  (h2 : rate_increase = 9/8) : 
  usual_time - school_journey usual_time rate_increase = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_early_arrival_l182_18207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_class_size_calculation_l182_18225

theorem class_size_calculation 
  (average_all : ℝ) 
  (excluded_count : ℕ) 
  (average_excluded : ℝ) 
  (average_remaining : ℝ) 
  (h1 : average_all = 65)
  (h2 : excluded_count = 5)
  (h3 : average_excluded = 20)
  (h4 : average_remaining = 90)
  : ∃ (total_students : ℕ), total_students = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_class_size_calculation_l182_18225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_length_l182_18295

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define the left focus
def left_focus : ℝ × ℝ := (-2, 0)

-- Define the line passing through the left focus at 45°
def line (x : ℝ) : ℝ := x + 2

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ellipse p.1 p.2 ∧ p.2 = line p.1}

-- State the theorem
theorem ellipse_intersection_length :
  ∃ (A B : ℝ × ℝ), A ∈ intersection_points ∧ B ∈ intersection_points ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = (12 * Real.sqrt 2 / 7)^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_length_l182_18295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_estimate_student_population_l182_18248

theorem estimate_student_population (N : ℝ) : 
  N > 0 → 
  (1 - (N - 90) / N * (N - 100) / N = 20 / N) → 
  N = 450 := by
  sorry

#check estimate_student_population

end NUMINAMATH_CALUDE_ERRORFEEDBACK_estimate_student_population_l182_18248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kite_perimeter_l182_18229

/-- The perimeter of a kite given its diagonal lengths -/
theorem kite_perimeter (d1 d2 : ℝ) (h1 : d1 = 20) (h2 : d2 = 16) : 
  4 * Real.sqrt ((d1/2)^2 + (d2/2)^2) = 8 * Real.sqrt 41 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kite_perimeter_l182_18229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_range_l182_18280

noncomputable def f (x : ℝ) : ℝ := Real.sin (x/4)^2 - 2*Real.cos (x/4)^2 + Real.sqrt 3 * Real.sin (x/4) * Real.cos (x/4)

theorem triangle_perimeter_range (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  b = Real.sqrt 3 ∧
  f B = -1/2 ∧
  a * Real.sin B = b * Real.sin A ∧
  b * Real.sin C = c * Real.sin B ∧
  c * Real.sin A = a * Real.sin C →
  2 * Real.sqrt 3 < a + b + c ∧ a + b + c ≤ 2 + Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_range_l182_18280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_condition_l182_18201

theorem log_condition (a b : ℝ) (ha : a > 0) (hb : b > 0) (hna : a ≠ 1) :
  (Real.log b / Real.log a > 0) ↔ ((a - 1) * (b - 1) > 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_condition_l182_18201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_at_one_l182_18298

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then Real.sin (Real.pi * x / 6) else 1 - 2 * x

theorem f_composition_at_one : f (f 1) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_at_one_l182_18298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_max_slope_l182_18238

/-- The maximum slope of OM for a parabola y^2 = 2px -/
theorem parabola_max_slope (p : ℝ) (hp : p > 0) :
  let parabola := {P : ℝ × ℝ | P.2^2 = 2 * p * P.1}
  let F := (p / 2, 0)
  ∀ P ∈ parabola,
    let M := ((P.1 + F.1) / 2, (P.2 + F.2) / 2)
    let slope := M.2 / M.1
    slope ≤ 1 ∧ ∃ P₀ ∈ parabola, let M₀ := ((P₀.1 + F.1) / 2, (P₀.2 + F.2) / 2); M₀.2 / M₀.1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_max_slope_l182_18238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_equals_negative_21_l182_18218

theorem sum_of_coefficients_equals_negative_21 (d : ℝ) : 
  let expanded := -d^2 + 12*d - 32
  (-1 + 12 + -32 = -21) := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_equals_negative_21_l182_18218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_problem_l182_18296

theorem trig_problem (α β : Real) 
  (h1 : Real.tan α = 4 * Real.sqrt 3)
  (h2 : Real.cos (β - α) = 13 / 14)
  (h3 : 0 < β) (h4 : β < α) (h5 : α < π / 2) :
  Real.cos α = 1 / 7 ∧ β = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_problem_l182_18296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_on_tangent_line_min_value_exists_l182_18294

-- Define the function f(x) = e^x
noncomputable def f (x : ℝ) : ℝ := Real.exp x

-- Define the tangent line l at (0, f(0))
def l (x : ℝ) : ℝ := x + 1

-- Theorem statement
theorem min_value_on_tangent_line :
  ∀ a b : ℝ, b = l a → 2^a + 2^(-b) ≥ Real.sqrt 2 := by
  sorry

-- Theorem for the existence of the minimum value
theorem min_value_exists :
  ∃ a b : ℝ, b = l a ∧ 2^a + 2^(-b) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_on_tangent_line_min_value_exists_l182_18294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_division_remainder_l182_18266

theorem polynomial_division_remainder : ∃ q : Polynomial ℤ, 
  (3 * X^2 - 23 * X + 68 : Polynomial ℤ) = (X - 7) * q + 54 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_division_remainder_l182_18266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_five_iff_power_two_l182_18256

theorem power_five_iff_power_two (a b : ℝ) : (a : ℝ)^5 < (b : ℝ)^5 ↔ (2 : ℝ)^a < (2 : ℝ)^b := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_five_iff_power_two_l182_18256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wire_length_is_144_l182_18284

/-- The radius of the original sphere in centimeters -/
noncomputable def sphere_radius : ℝ := 24

/-- The semi-major axis of the elliptical cross-section in centimeters -/
noncomputable def semi_major_axis : ℝ := 16

/-- The semi-minor axis of the elliptical cross-section in centimeters -/
noncomputable def semi_minor_axis : ℝ := 8

/-- The volume of the sphere -/
noncomputable def sphere_volume : ℝ := (4 / 3) * Real.pi * (sphere_radius ^ 3)

/-- The area of the elliptical cross-section -/
noncomputable def ellipse_area : ℝ := Real.pi * semi_major_axis * semi_minor_axis

/-- The theorem stating that the length of the wire is 144 cm -/
theorem wire_length_is_144 : 
  sphere_volume / ellipse_area = 144 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wire_length_is_144_l182_18284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l182_18217

/-- Two-dimensional vector -/
def Vector2D := ℝ × ℝ

/-- Given vectors a and b -/
def a (x : ℝ) : Vector2D := (1, x)
def b (x : ℝ) : Vector2D := (2*x + 3, -x)

/-- Dot product of two vectors -/
def dot_product (v w : Vector2D) : ℝ := v.1 * w.1 + v.2 * w.2

/-- Perpendicular vectors have zero dot product -/
def perpendicular (v w : Vector2D) : Prop := dot_product v w = 0

/-- Parallel vectors are scalar multiples of each other -/
def parallel (v w : Vector2D) : Prop := ∃ k : ℝ, v = (k * w.1, k * w.2)

/-- Magnitude of a vector -/
noncomputable def magnitude (v : Vector2D) : ℝ := Real.sqrt (v.1^2 + v.2^2)

theorem vector_problem :
  (∀ x : ℝ, perpendicular (a x) (b x) → x = -1 ∨ x = 3) ∧
  (∀ x : ℝ, parallel (a x) (b x) → magnitude ((a x).1 - (b x).1, (a x).2 - (b x).2) = 2 * Real.sqrt 5 ∨ 
                                    magnitude ((a x).1 - (b x).1, (a x).2 - (b x).2) = 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l182_18217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_f_homomorphism_f_unique_l182_18270

/-- The exponential function with base 3 -/
noncomputable def f (x : ℝ) : ℝ := 3^x

/-- The exponential function with base 3 is monotonically increasing -/
theorem f_increasing : Monotone f := by sorry

/-- The exponential function with base 3 satisfies f(x+y) = f(x)f(y) -/
theorem f_homomorphism (x y : ℝ) : f (x + y) = f x * f y := by sorry

/-- The exponential function with base 3 is the unique function satisfying
    both monotonicity and the homomorphism property -/
theorem f_unique (g : ℝ → ℝ) (h_mono : Monotone g) (h_homo : ∀ x y, g (x + y) = g x * g y) :
  g = f := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_f_homomorphism_f_unique_l182_18270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_difference_of_four_factors_of_8_factorial_l182_18212

theorem smallest_difference_of_four_factors_of_8_factorial :
  ∃ (w x y z : ℕ+),
    w * x * y * z = Nat.factorial 8 ∧
    w < x ∧ x < y ∧ y < z ∧
    ∀ (a b c d : ℕ+),
      a * b * c * d = Nat.factorial 8 →
      a < b → b < c → c < d →
      d - a ≥ 16 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_difference_of_four_factors_of_8_factorial_l182_18212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_rationals_in_list_l182_18244

/-- Represents the special decimal number 0.1010010001... with increasing zeros between ones -/
noncomputable def special_decimal : ℝ := sorry

/-- Checks if a real number is rational -/
def is_rational (x : ℝ) : Prop := ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

/-- The list of numbers from the problem -/
noncomputable def number_list : List ℝ := [-1, 0, 22/7, Real.pi/2, 0.383838, special_decimal]

/-- Counts the number of rational numbers in a list -/
def count_rational (l : List ℝ) : ℕ := sorry

/-- The main theorem stating that there are exactly 4 rational numbers in the given list -/
theorem four_rationals_in_list : count_rational number_list = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_rationals_in_list_l182_18244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_sum_l182_18237

/-- A cubic polynomial Q with specific properties -/
noncomputable def Q (m : ℝ) : ℝ → ℝ := 
  fun x => (5/2 * m) * x^3 + (5/2 * m) * x^2 + (-3 * m) * x + m

/-- Theorem stating the properties of Q and the sum Q(2) + Q(-2) -/
theorem cubic_polynomial_sum (m : ℝ) : 
  Q m 0 = m ∧ Q m 1 = 3 * m ∧ Q m (-1) = 4 * m ∧ Q m 2 + Q m (-2) = 22 * m := by
  sorry

-- Remove the #eval line as it's not necessary for the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_sum_l182_18237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_characterization_l182_18220

/-- Given a complex number Z = (m^2 + 5m + 6) + (m^2 - 2m - 15)i, this theorem characterizes
    the nature of Z based on different values of the real parameter m. -/
theorem complex_number_characterization (m : ℝ) :
  (Complex.im (Complex.mk (m^2 + 5*m + 6) (m^2 - 2*m - 15)) = 0 ↔ m = -3 ∨ m = 5) ∧
  (Complex.im (Complex.mk (m^2 + 5*m + 6) (m^2 - 2*m - 15)) ≠ 0 ↔ m ≠ -3 ∧ m ≠ 5) ∧
  (Complex.re (Complex.mk (m^2 + 5*m + 6) (m^2 - 2*m - 15)) = 0 ∧ 
   Complex.im (Complex.mk (m^2 + 5*m + 6) (m^2 - 2*m - 15)) ≠ 0 ↔ m = -2) ∧
  (Complex.re (Complex.mk (m^2 + 5*m + 6) (m^2 - 2*m - 15)) > 0 ∧ 
   Complex.im (Complex.mk (m^2 + 5*m + 6) (m^2 - 2*m - 15)) < 0 ↔ -2 < m ∧ m < 5) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_characterization_l182_18220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_meeting_lap_l182_18259

/-- Represents the track configuration and runner speeds -/
structure TrackConfig where
  trackLength : ℝ
  abLength : ℝ
  abStraightLength : ℝ
  fatherSpeed : ℝ
  sonSpeed : ℝ

/-- Calculates the lap time for the son -/
noncomputable def sonLapTime (config : TrackConfig) : ℝ :=
  config.trackLength / config.sonSpeed

/-- Calculates the lap time for the father -/
noncomputable def fatherLapTime (config : TrackConfig) : ℝ :=
  config.abLength / config.fatherSpeed + config.abStraightLength / config.fatherSpeed

/-- Determines the lap number when the son first meets the father again -/
noncomputable def meetingLap (config : TrackConfig) : ℕ :=
  sorry

/-- The main theorem stating when the son first meets the father again -/
theorem first_meeting_lap (config : TrackConfig) 
  (h1 : config.trackLength = 400)
  (h2 : config.abLength = 200)
  (h3 : config.abStraightLength = 50)
  (h4 : config.fatherSpeed = 5)  -- 100 meters / 20 seconds
  (h5 : config.sonSpeed = 100/19) :
  meetingLap config = 3 := by
  sorry

#check first_meeting_lap

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_meeting_lap_l182_18259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_theorem_l182_18210

noncomputable def v : ℝ × ℝ := (1, -1/2)

noncomputable def projection (u : ℝ × ℝ) (w : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := u.1 * w.1 + u.2 * w.2
  let magnitude_squared := w.1 * w.1 + w.2 * w.2
  (dot_product / magnitude_squared * w.1, dot_product / magnitude_squared * w.2)

theorem projection_theorem :
  projection (2, -1) v = (1, -1/2) →
  projection (3, 5) (v.1 + 1, v.2 + 1) = (104/17, 26/17) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_theorem_l182_18210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_log2_derivative_l182_18274

-- Define the derivative of log_2(x)
noncomputable def log2_derivative (x : ℝ) : ℝ := 1 / (x * Real.log 2)

-- Theorem statement
theorem correct_log2_derivative :
  ∀ x : ℝ, x > 0 → deriv (λ y => Real.log y / Real.log 2) x = log2_derivative x :=
by
  -- The proof is skipped for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_log2_derivative_l182_18274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_grid_with_monochromatic_rectangle_l182_18204

/-- A color type representing black and white --/
inductive Color
| black
| white

/-- A type representing a grid coloring --/
def GridColoring (n : ℕ) := Fin n → Fin n → Color

/-- A type representing a rectangle in the grid --/
structure Rectangle (n : ℕ) where
  x1 : Fin n
  y1 : Fin n
  x2 : Fin n
  y2 : Fin n
  different_x : x1 ≠ x2
  different_y : y1 ≠ y2

/-- A predicate that checks if a rectangle has all corners of the same color --/
def hasMonochromaticRectangle (n : ℕ) (coloring : GridColoring n) : Prop :=
  ∃ (rect : Rectangle n), 
    coloring rect.x1 rect.y1 = coloring rect.x2 rect.y1 ∧
    coloring rect.x1 rect.y1 = coloring rect.x1 rect.y2 ∧
    coloring rect.x1 rect.y1 = coloring rect.x2 rect.y2

/-- The main theorem stating that 5 is the smallest grid size that guarantees a monochromatic rectangle --/
theorem smallest_grid_with_monochromatic_rectangle :
  (∀ (coloring : GridColoring 5), hasMonochromaticRectangle 5 coloring) ∧
  (∀ (n : ℕ), n < 5 → ∃ (coloring : GridColoring n), ¬hasMonochromaticRectangle n coloring) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_grid_with_monochromatic_rectangle_l182_18204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_right_triangle_l182_18206

/-- The radius of the inscribed circle in a right triangle with legs 9 and 12 is 3 -/
theorem inscribed_circle_radius_right_triangle :
  ∀ (P Q R : ℝ × ℝ) (r : ℝ),
    (Q.1 - P.1)^2 + (Q.2 - P.2)^2 = 9^2 →
    (R.1 - P.1)^2 + (R.2 - P.2)^2 = 12^2 →
    (R.1 - Q.1)^2 + (R.2 - Q.2)^2 = (Q.1 - P.1)^2 + (Q.2 - P.2)^2 + (R.1 - P.1)^2 + (R.2 - P.2)^2 →
    r = (9 + 12 - Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2 + (R.1 - P.1)^2 + (R.2 - P.2)^2)) / 2 →
    r = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_right_triangle_l182_18206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_proposition_l182_18289

theorem negation_of_proposition :
  (¬(∀ x : ℝ, x^2 + x + 2 > 0)) ↔ (∃ x₀ : ℝ, x₀^2 + x₀ + 2 ≤ 0) :=
by
  apply Iff.intro
  · intro h
    push_neg at h
    exact h
  · intro h
    push_neg
    exact h

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_proposition_l182_18289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_of_digits_l182_18255

def digits : List Nat := [7, 3, 1, 4, 9]

def largest_number (digits : List Nat) : Nat :=
  digits.toArray.qsort (· > ·) |>.toList.foldl (fun acc d => acc * 10 + d) 0

def smallest_number (digits : List Nat) : Nat :=
  digits.toArray.qsort (· < ·) |>.toList.foldl (fun acc d => acc * 10 + d) 0

def difference (digits : List Nat) : Nat :=
  largest_number digits - smallest_number digits

theorem difference_of_digits :
  difference digits = 83952 ∧ difference digits % 3 = 0 := by
  sorry

#eval difference digits

end NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_of_digits_l182_18255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_equivalence_l182_18278

-- Define the system of equations
def system (a x y : ℝ) : Prop :=
  a * x + y = 2 * a + 3 ∧ x - a * y = a + 4

-- Define the circle
def circle_eq (x y : ℝ) : Prop :=
  (x - 3)^2 + (y - 1)^2 = 5

-- Statement of the theorem
theorem solution_set_equivalence :
  ∀ x y : ℝ, (∃ a : ℝ, system a x y) ↔ (circle_eq x y ∧ ¬(x = 2 ∧ y = -1)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_equivalence_l182_18278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_1414_l182_18226

/-- The floor function -/
noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

/-- The infinite series -/
noncomputable def series_sum : ℝ := ∑' k, floor ((1 + Real.sqrt (2000000 / 4^k)) / 2)

/-- Theorem stating that the sum of the series equals 1414 -/
theorem series_sum_equals_1414 : series_sum = 1414 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_1414_l182_18226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_cost_price_l182_18279

/-- The cost price of a book given its selling price and profit percentage -/
noncomputable def cost_price (selling_price : ℝ) (profit_percentage : ℝ) : ℝ :=
  selling_price / (1 + profit_percentage / 100)

/-- Theorem stating that the cost price of a book sold for $260 with 20% profit is approximately $216.67 -/
theorem book_cost_price : 
  let selling_price := (260 : ℝ)
  let profit_percentage := (20 : ℝ)
  abs (cost_price selling_price profit_percentage - 216.67) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_cost_price_l182_18279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_expression_l182_18209

/-- An arithmetic sequence with positive terms -/
structure PositiveArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  pos : ∀ n, a n > 0
  arith : ∀ n, a (n + 1) = a n + d

/-- Sum of the first n terms of an arithmetic sequence -/
def SumOfTerms (seq : PositiveArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (seq.a 1 + seq.a n) / 2

theorem min_value_of_expression (seq : PositiveArithmeticSequence) 
    (h1 : seq.a 3 = 5)
    (h2 : SumOfTerms seq 3 = seq.a 1 * seq.a 5) :
  ∃ n : ℕ, ∀ m : ℕ, (m : ℚ) * (2 * seq.a m - 10)^2 ≥ 0 ∧ 
    (n : ℚ) * (2 * seq.a n - 10)^2 = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_expression_l182_18209
