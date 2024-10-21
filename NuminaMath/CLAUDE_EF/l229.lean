import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sinusoidal_function_form_l229_22922

/-- Given a sinusoidal function f(x) = M * sin(ω * x + φ) with specific properties,
    prove that it has the form f(x) = √3 * sin(π/6 * x) -/
theorem sinusoidal_function_form (M ω φ : ℝ) (hM : M > 0) (hω : ω > 0) (hφ : |φ| < π/2) 
  (f : ℝ → ℝ) (hf : ∀ x, f x = M * Real.sin (ω * x + φ)) :
  f 0 = 0 →
  f 6 = 0 →
  (∃ C, ∀ x, f x ≤ f C) →
  (∃ a b c : ℝ, (a + c) * (Real.sin (f C) - Real.sin (f 0)) = (a + b) * Real.sin (f 6)) →
  ∀ x, f x = Real.sqrt 3 * Real.sin (π/6 * x) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sinusoidal_function_form_l229_22922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_possible_medians_l229_22960

/-- A set of ten distinct integers containing 5, 7, 12, 15, 18, and 21 -/
def S : Finset ℤ := sorry

/-- The median of a set of integers -/
def median (s : Finset ℤ) : ℚ := sorry

/-- The set of all possible medians for S -/
def possible_medians : Finset ℚ := sorry

theorem count_possible_medians :
  (S.card = 10) →
  (∀ x, x ∈ S → x ∈ ({5, 7, 12, 15, 18, 21} : Finset ℤ)) →
  (∀ x y, x ∈ S → y ∈ S → x ≠ y) →
  possible_medians.card = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_possible_medians_l229_22960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_safe_driving_time_theorem_safe_after_four_hours_l229_22934

/-- Represents the blood alcohol content (in mg/mL) after x hours of not drinking -/
noncomputable def blood_alcohol_content (x : ℝ) : ℝ := 0.3 * (0.5 ^ x)

/-- The safe driving limit for blood alcohol content (in mg/mL) -/
def safe_driving_limit : ℝ := 0.02

/-- Theorem stating that it takes at least 4 hours to reach a safe driving level -/
theorem safe_driving_time_theorem :
  ∀ x : ℝ, x < 4 → blood_alcohol_content x > safe_driving_limit :=
by sorry

/-- Theorem stating that after 4 hours, the blood alcohol content is at or below the safe driving limit -/
theorem safe_after_four_hours :
  blood_alcohol_content 4 ≤ safe_driving_limit :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_safe_driving_time_theorem_safe_after_four_hours_l229_22934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_l229_22966

def is_valid (n : Nat) : Bool :=
  1000 ≤ n && n ≤ 3000 &&
  n % 10 = (n / 1000) + ((n / 100) % 10) + ((n / 10) % 10)

theorem count_valid_numbers : 
  (Finset.filter (fun n => is_valid n) (Finset.range 2001)).card = 109 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_l229_22966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_of_special_square_l229_22949

/-- A square region with quarter-circular arcs at each corner -/
structure QuarterCircleSquare where
  side : ℝ
  side_positive : side > 0

/-- The perimeter of a QuarterCircleSquare -/
noncomputable def perimeter (q : QuarterCircleSquare) : ℝ :=
  4 * (Real.pi / 4) * q.side

theorem perimeter_of_special_square :
  ∀ q : QuarterCircleSquare, q.side = 4 / Real.pi → perimeter q = 4 := by
  sorry

#eval "Proof completed"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_of_special_square_l229_22949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_drainage_rate_is_11_inches_l229_22910

/-- Represents the rainfall and drainage scenario --/
structure RainfallScenario where
  capacity : ℚ  -- Total capacity in inches
  day1_rain : ℚ  -- Rainfall on day 1 in inches
  day2_rain : ℚ  -- Rainfall on day 2 in inches
  day3_rain : ℚ  -- Rainfall on day 3 in inches
  day4_rain : ℚ  -- Minimum rainfall on day 4 in inches
  overflow_day : ℕ  -- Day when overflow occurs

/-- Calculates the minimum drainage rate per day --/
def minimum_drainage_rate (scenario : RainfallScenario) : ℚ :=
  let total_rain_3_days := scenario.day1_rain + scenario.day2_rain + scenario.day3_rain
  let remaining_capacity := scenario.capacity - total_rain_3_days
  let overflow_amount := remaining_capacity + scenario.day4_rain
  overflow_amount / 3

/-- Theorem stating the minimum drainage rate is 11 inches per day --/
theorem drainage_rate_is_11_inches 
  (scenario : RainfallScenario)
  (h1 : scenario.capacity = 72)
  (h2 : scenario.day1_rain = 10)
  (h3 : scenario.day2_rain = 2 * scenario.day1_rain)
  (h4 : scenario.day3_rain = 3/2 * scenario.day2_rain)
  (h5 : scenario.day4_rain = 21)
  (h6 : scenario.overflow_day = 4) :
  minimum_drainage_rate scenario = 11 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_drainage_rate_is_11_inches_l229_22910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_domain_interval_notation_l229_22995

noncomputable def f (x : ℝ) : ℝ := 1 / ((x - 3) * (x - 7))

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x ≠ 3 ∧ x ≠ 7} :=
by sorry

theorem domain_interval_notation :
  {x : ℝ | x ≠ 3 ∧ x ≠ 7} = Set.Iio 3 ∪ Set.Ioo 3 7 ∪ Set.Ioi 7 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_domain_interval_notation_l229_22995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flight_theorem_l229_22986

/-- Represents the flight scenario between two cities -/
structure FlightScenario where
  d : ℝ  -- distance between cities
  p : ℝ  -- plane speed in still air
  w : ℝ  -- wind speed
  time_against_wind : ℝ  -- time for trip against wind
  time_diff : ℝ  -- time difference between still air and with wind

/-- The flight scenario satisfies the given conditions -/
def satisfies_conditions (fs : FlightScenario) : Prop :=
  fs.time_against_wind = 100 ∧
  fs.d = 100 * (fs.p - fs.w) ∧
  fs.d / (fs.p + fs.w) = fs.d / fs.p - fs.time_diff ∧
  fs.time_diff = 15

/-- The return trip time is either 15 or approximately 67 minutes -/
def return_trip_time (fs : FlightScenario) : Prop :=
  fs.d / (fs.p + fs.w) = 15 ∨ 
  (67 ≤ fs.d / (fs.p + fs.w) ∧ fs.d / (fs.p + fs.w) < 68)

theorem flight_theorem (fs : FlightScenario) :
  satisfies_conditions fs → return_trip_time fs :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_flight_theorem_l229_22986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quiz_point_difference_l229_22990

/-- Represents a quiz with arithmetic progression of question points -/
structure Quiz where
  n : ℕ -- number of questions
  x : ℚ -- points for the first question
  d : ℚ -- common difference between consecutive questions

/-- Total points of the quiz -/
def Quiz.totalPoints (q : Quiz) : ℚ :=
  q.n * q.x + q.n * (q.n - 1) / 2 * q.d

/-- Points for the k-th question (1-indexed) -/
def Quiz.pointsForQuestion (q : Quiz) (k : ℕ) : ℚ :=
  q.x + (k - 1) * q.d

theorem quiz_point_difference (q : Quiz) : 
  q.n = 8 ∧ 
  q.totalPoints = 360 ∧ 
  q.pointsForQuestion 3 = 39 → 
  q.d = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quiz_point_difference_l229_22990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log28_not_direct_others_direct_l229_22956

-- Define the given logarithm values
noncomputable def log7 : ℝ := 0.8451
noncomputable def log8 : ℝ := 0.9031

-- Define a function to check if a logarithm can be computed directly
def can_compute_directly (x : ℝ) : Prop :=
  ∃ (a b : ℝ), x = a * log7 + b * log8 ∨ x = a * log7 - b * log8 ∨ x = a + b * log8

-- Theorem statement
theorem log28_not_direct_others_direct :
  ¬(can_compute_directly (Real.log 28)) ∧
  (can_compute_directly (Real.log (Real.sqrt 56))) ∧
  (can_compute_directly (Real.log (7/8))) ∧
  (can_compute_directly (Real.log 800)) ∧
  (can_compute_directly (Real.log 0.875)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log28_not_direct_others_direct_l229_22956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_reciprocal_constant_l229_22948

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x^2 / (1 + x^2)

-- Theorem statement
theorem f_sum_reciprocal_constant (x : ℝ) (h : x ≠ 0) : f x + f (1/x) = 1 := by
  -- Expand the definition of f
  unfold f
  -- Simplify the expression
  simp [h]
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_reciprocal_constant_l229_22948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_sqrt2_line_equation_midpoint_l229_22974

-- Define the circle C
def C (x y : ℝ) : Prop := x^2 + y^2 - 6*y + 4 = 0

-- Define a line passing through the origin
def line_through_origin (m : ℝ) (x y : ℝ) : Prop := y = m * x

-- Define the chord length
noncomputable def chord_length (m : ℝ) : ℝ := 2 * Real.sqrt 2

-- Define the midpoint condition
def is_midpoint (x₁ y₁ x₂ y₂ : ℝ) : Prop := x₁ = x₂/2 ∧ y₁ = y₂/2

-- Theorem for part (I)
theorem chord_length_sqrt2 :
  ∀ x y : ℝ, C x y ∧ line_through_origin (Real.sqrt 2) x y →
  ∃ x' y', C x' y' ∧ line_through_origin (Real.sqrt 2) x' y' ∧
  Real.sqrt ((x - x')^2 + (y - y')^2) = chord_length (Real.sqrt 2) :=
by sorry

-- Theorem for part (II)
theorem line_equation_midpoint :
  ∀ x₁ y₁ x₂ y₂ : ℝ, C x₁ y₁ ∧ C x₂ y₂ ∧ is_midpoint x₁ y₁ x₂ y₂ →
  (∀ x y : ℝ, line_through_origin 1 x y ∨ line_through_origin (-1) x y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_sqrt2_line_equation_midpoint_l229_22974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_worker_travel_time_l229_22908

/-- The usual time taken by a worker to cover the distance between home and office -/
noncomputable def usual_time : ℝ := 60

/-- The worker's speed relative to her normal speed when she's late -/
noncomputable def relative_speed : ℝ := 4 / 5

/-- The additional time taken when walking at the relative speed -/
noncomputable def additional_time : ℝ := 15

theorem worker_travel_time :
  ∀ (normal_speed : ℝ) (distance : ℝ),
  normal_speed > 0 → distance > 0 →
  distance = normal_speed * usual_time ∧
  distance = (relative_speed * normal_speed) * (usual_time + additional_time) →
  usual_time = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_worker_travel_time_l229_22908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inradius_of_scalene_triangle_l229_22962

/-- A scalene triangle ABC with given side lengths and incenter properties -/
structure ScaleneTriangle where
  -- Side lengths
  AB : ℝ
  BC : ℝ
  AC : ℝ
  -- Incenter distance to vertex C
  IC : ℝ
  -- Conditions
  scalene : AB ≠ BC ∧ BC ≠ AC ∧ AC ≠ AB
  side_lengths : AB = 32 ∧ BC = 40 ∧ AC = 24
  incenter_distance : IC = 18

/-- The inradius of a scalene triangle with given properties -/
noncomputable def inradius (t : ScaleneTriangle) : ℝ := 2 * Real.sqrt 17

/-- Theorem: The inradius of the given scalene triangle is 2√17 -/
theorem inradius_of_scalene_triangle (t : ScaleneTriangle) :
  inradius t = 2 * Real.sqrt 17 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inradius_of_scalene_triangle_l229_22962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_deposit_proof_l229_22918

/-- Represents the compound interest calculation --/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate) ^ time

/-- Proves that the initial deposit is $6000 given the problem conditions --/
theorem initial_deposit_proof (final_amount : ℝ) (rate : ℝ) (time : ℝ)
  (h1 : final_amount = 6615)
  (h2 : rate = 0.05)
  (h3 : time = 2)
  (h4 : compound_interest 6000 rate time = final_amount) :
  6000 = (final_amount / ((1 + rate) ^ time)) := by
  sorry

#check initial_deposit_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_deposit_proof_l229_22918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linda_travel_theorem_l229_22987

/-- Represents the number of minutes it takes Linda to travel one mile on the first day -/
def initial_minutes_per_mile : ℕ := 4

/-- Calculates the number of minutes it takes Linda to travel one mile on a given day -/
def minutes_per_mile (day : ℕ) : ℕ :=
  initial_minutes_per_mile + 4 * (day - 1)

/-- Calculates the distance Linda travels on a given day -/
def distance_traveled (day : ℕ) : ℕ :=
  60 / minutes_per_mile day

/-- The total distance Linda traveled over the five days -/
def total_distance : ℕ :=
  (distance_traveled 1) + (distance_traveled 2) + (distance_traveled 3) + (distance_traveled 4) + (distance_traveled 5)

theorem linda_travel_theorem :
  (∀ (day : ℕ), day ≤ 5 → minutes_per_mile day ∣ 60) ∧
  (∀ (day : ℕ), day ≤ 5 → distance_traveled day > 0) →
  total_distance = 33 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linda_travel_theorem_l229_22987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_m_for_sets_l229_22961

theorem smallest_m_for_sets (n : ℕ) (h : n ≥ 5) :
  ∃ m : ℕ, m = 3 * n - 3 ∧
  (∃ A B : Finset ℤ,
    Finset.card A = n ∧
    Finset.card B = m ∧
    A ⊆ B ∧
    (∀ x y : ℤ, x ∈ B → y ∈ B → x ≠ y →
      (x + y ∈ B ↔ x ∈ A ∧ y ∈ A))) ∧
  (∀ k : ℕ, k < m →
    ¬∃ A B : Finset ℤ,
      Finset.card A = n ∧
      Finset.card B = k ∧
      A ⊆ B ∧
      (∀ x y : ℤ, x ∈ B → y ∈ B → x ≠ y →
        (x + y ∈ B ↔ x ∈ A ∧ y ∈ A))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_m_for_sets_l229_22961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_function_evaluation_l229_22971

def r (θ : ℚ) : ℚ := 1 / (1 + θ)

def s (θ : ℚ) : ℚ := θ + 1

theorem nested_function_evaluation :
  s (r (s (r (s (r 2))))) = 24/17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_function_evaluation_l229_22971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_A_to_B_l229_22968

/-- The distance between two points in 3D space -/
noncomputable def distance3D (a b : ℝ × ℝ × ℝ) : ℝ :=
  let (x₁, y₁, z₁) := a
  let (x₂, y₂, z₂) := b
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2 + (z₁ - z₂)^2)

/-- Point A in 3D space -/
def A : ℝ × ℝ × ℝ := (2, -3, 3)

/-- Point B in 3D space -/
def B : ℝ × ℝ × ℝ := (2, 1, 0)

/-- The distance between points A and B is 5 -/
theorem distance_A_to_B : distance3D A B = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_A_to_B_l229_22968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_problem_l229_22907

theorem cosine_problem (α : ℝ) 
  (h1 : Real.cos (α - π/12) = 3/5) 
  (h2 : α ∈ Set.Ioo 0 (π/2)) : 
  Real.cos (2*α + π/3) = -24/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_problem_l229_22907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_center_of_symmetry_l229_22911

/-- The function g(x) resulting from transforming cos(x) -/
noncomputable def g (x : ℝ) : ℝ := Real.cos (1/2 * (x - Real.pi/3))

/-- The x-coordinate of the center of symmetry of g(x) -/
noncomputable def center_of_symmetry (k : ℤ) : ℝ := 2 * ↑k * Real.pi + 4 * Real.pi / 3

theorem g_center_of_symmetry (k : ℤ) :
  g (center_of_symmetry k) = g (2 * center_of_symmetry k - (center_of_symmetry k)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_center_of_symmetry_l229_22911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_shape_l229_22909

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the property of being isosceles
def isIsosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.c = t.a

-- Define the property of being right-angled
def isRightAngled (t : Triangle) : Prop :=
  t.A = Real.pi / 2 ∨ t.B = Real.pi / 2 ∨ t.C = Real.pi / 2

-- Theorem statement
theorem triangle_shape (t : Triangle) (h : t.a * Real.cos t.A = t.b * Real.cos t.B) :
  isIsosceles t ∨ isRightAngled t := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_shape_l229_22909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l229_22964

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 1 / (1 - x * (1 - x))

-- State the theorem
theorem max_value_of_f :
  ∃ (M : ℝ), M = 4/3 ∧ ∀ (x : ℝ), 1 - x * (1 - x) ≠ 0 → f x ≤ M := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l229_22964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_project_completion_time_l229_22953

/-- The number of days it takes for A to complete the project alone -/
noncomputable def a_days : ℝ := 10

/-- The number of days it takes for A and B to complete the project together,
    with A quitting 10 days before completion -/
noncomputable def total_days : ℝ := 15

/-- The number of days A works before quitting -/
noncomputable def a_work_days : ℝ := total_days - 10

/-- The rate at which A completes the project per day -/
noncomputable def a_rate : ℝ := 1 / a_days

/-- The number of days it takes for B to complete the project alone -/
noncomputable def b_days : ℝ := 30

/-- The rate at which B completes the project per day -/
noncomputable def b_rate : ℝ := 1 / b_days

theorem project_completion_time :
  a_work_days * (a_rate + b_rate) + (total_days - a_work_days) * b_rate = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_project_completion_time_l229_22953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_calculation_l229_22933

/-- The length of a platform given train parameters -/
noncomputable def platform_length (train_length : ℝ) (train_speed_kmh : ℝ) (time_to_pass : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  let total_distance := train_speed_ms * time_to_pass
  total_distance - train_length

/-- Theorem stating the platform length for given parameters -/
theorem platform_length_calculation :
  platform_length 360 45 39.2 = 130 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_calculation_l229_22933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_property_l229_22972

open Real

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log x - a * x

-- State the theorem
theorem zeros_property (a : ℝ) (x₁ x₂ x₃ : ℝ) :
  (∃ (x₁ x₂ : ℝ), x₁ < x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0) →
  (0 < a ∧ a < 1 / exp 1) ∧
  (x₃ > 1 ∧ a = x₃ / (exp x₃) → exp x₃ = x₂) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_property_l229_22972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_square_root_l229_22980

theorem factorial_square_root : (Real.sqrt ((4 * 3 * 2 * 1) * (4 * 3 * 2 * 1)) : ℝ) = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_square_root_l229_22980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l229_22913

/-- Definition of the ellipse Γ -/
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

/-- Definition of a point on the ellipse -/
def point_on_ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  ellipse a b x y

/-- Definition of the focus of the ellipse -/
def focus (x y : ℝ) : Prop :=
  x = -2 * Real.sqrt 3 ∧ y = 0

/-- Definition of a line intersecting the ellipse -/
def line_intersects_ellipse (a b k m : ℝ) : Prop :=
  ∃ x₁ y₁ x₂ y₂, ellipse a b x₁ y₁ ∧ ellipse a b x₂ y₂ ∧
  y₁ = k * x₁ + m ∧ y₂ = k * x₂ + m

/-- Definition of the condition MP = (1/2)PN -/
def midpoint_condition (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₂ = -2 * x₁ ∧ y₂ = -2 * y₁ + 3

/-- Definition of the condition |OM + ON| = 4 -/
def sum_vectors_condition (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  (x₁ + x₂)^2 + (y₁ + y₂)^2 = 16

/-- Main theorem -/
theorem ellipse_properties (a b : ℝ) (h₁ : a > b) (h₂ : b > 0) :
  point_on_ellipse a b (Real.sqrt 3) (Real.sqrt 13 / 2) →
  focus (-2 * Real.sqrt 3) 0 →
  (a^2 = 16 ∧ b^2 = 4) ∧
  (∀ k m x₁ y₁ x₂ y₂, 
    line_intersects_ellipse a b k m →
    midpoint_condition x₁ y₁ x₂ y₂ →
    k^2 = 3/20) ∧
  (∀ x₁ y₁ x₂ y₂,
    ellipse a b x₁ y₁ →
    ellipse a b x₂ y₂ →
    sum_vectors_condition x₁ y₁ x₂ y₂ →
    abs (x₁ * y₂ - x₂ * y₁) ≤ 4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l229_22913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_five_identity_l229_22993

theorem sqrt_five_identity : Real.sqrt 5 * (Real.sqrt 5 - 1 / Real.sqrt 5) = 4 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_five_identity_l229_22993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_when_k_zero_f_nonnegative_iff_k_nonpositive_extreme_value_exists_and_independent_of_a_l229_22930

noncomputable section

variable (a : ℝ) (ha : a > 0)

def f (a : ℝ) (x k : ℝ) : ℝ := 2 * Real.log x + a / x - 2 * Real.log a - k * x / a

theorem f_positive_when_k_zero (a : ℝ) (ha : a > 0) (x : ℝ) (hx : x > 0) :
  f a x 0 > 0 := by sorry

theorem f_nonnegative_iff_k_nonpositive (a : ℝ) (ha : a > 0) :
  (∀ x > 0, f a x k ≥ 0) ↔ k ≤ 0 := by sorry

theorem extreme_value_exists_and_independent_of_a (k : ℝ) (hk : k ≤ 0) :
  ∃ (e : ℝ), ∀ (a : ℝ), a > 0 → 
    (∀ (x : ℝ), x > 0 → f a x k ≥ e) ∧ 
    (∃ (x₀ : ℝ), x₀ > 0 ∧ f a x₀ k = e) := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_when_k_zero_f_nonnegative_iff_k_nonpositive_extreme_value_exists_and_independent_of_a_l229_22930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_border_length_l229_22927

theorem garden_border_length :
  let garden_width : ℕ := 6
  let garden_length : ℕ := 7
  let num_beds : ℕ := 5
  let bed_sides : List ℕ := [4, 3, 3, 2, 2]
  bed_sides.length = num_beds ∧
  (List.sum (bed_sides.map (λ x => x * x))) = garden_width * garden_length ∧
  (List.sum (bed_sides.map (λ x => 4 * x)) - 2 * (garden_width + garden_length)) / 2 = 15
  := by
  -- Introduce the local variables
  intro garden_width garden_length num_beds bed_sides
  -- Prove the conditions
  have h1 : bed_sides.length = num_beds := by rfl
  have h2 : (List.sum (bed_sides.map (λ x => x * x))) = garden_width * garden_length := by
    simp [List.sum, List.map]
    norm_num
  have h3 : (List.sum (bed_sides.map (λ x => 4 * x)) - 2 * (garden_width + garden_length)) / 2 = 15 := by
    simp [List.sum, List.map]
    norm_num
  -- Combine the proofs
  exact ⟨h1, h2, h3⟩

#check garden_border_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_border_length_l229_22927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_value_l229_22988

theorem sin_cos_value (x : ℝ) (h : Real.sin x = 2 * Real.cos x) : Real.sin x * Real.cos x = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_value_l229_22988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimization_problem_l229_22955

/-- The problem statement --/
theorem optimization_problem (x y k : ℝ) :
  (x - 4 * y + 3 ≤ 0) →
  (3 * x + 5 * y - 25 ≤ 0) →
  (x ≥ 1) →
  (∃ (z : ℝ → ℝ), z = λ t ↦ k * t + y) →
  (∀ (x' y' : ℝ), x' - 4 * y' + 3 ≤ 0 → 3 * x' + 5 * y' - 25 ≤ 0 → x' ≥ 1 → k * x' + y' ≤ 12) →
  (∀ (x' y' : ℝ), x' - 4 * y' + 3 ≤ 0 → 3 * x' + 5 * y' - 25 ≤ 0 → x' ≥ 1 → k * x' + y' ≥ 3) →
  (∃ (x'' y'' : ℝ), x'' - 4 * y'' + 3 ≤ 0 ∧ 3 * x'' + 5 * y'' - 25 ≤ 0 ∧ x'' ≥ 1 ∧ k * x'' + y'' = 12) →
  (∃ (x''' y''' : ℝ), x''' - 4 * y''' + 3 ≤ 0 ∧ 3 * x''' + 5 * y''' - 25 ≤ 0 ∧ x''' ≥ 1 ∧ k * x''' + y''' = 3) →
  k = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimization_problem_l229_22955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l229_22959

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then -2017 * x^2 + 2018 * x else Real.log (x + 1)

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f (1 - 2^x + a * 4^x) ≥ 0) → a ≥ 1/4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l229_22959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_measure_l229_22942

theorem triangle_angle_measure (a b c : ℝ) (h : (a + b + c) * (b + c - a) = 3 * b * c) :
  (b^2 + c^2 - a^2) / (2 * b * c) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_measure_l229_22942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radii_sum_l229_22983

open Real

/-- A circle with radius r is inscribed in a triangle. Tangents to this circle, parallel to the sides
    of the triangle, cut off three smaller triangles from it. Let r₁, r₂, r₃ be the radii of the
    inscribed circles in these smaller triangles. This theorem states that r₁ + r₂ + r₃ = r. -/
theorem inscribed_circle_radii_sum (r r₁ r₂ r₃ : ℝ) 
    (h : r > 0 ∧ r₁ > 0 ∧ r₂ > 0 ∧ r₃ > 0) : r₁ + r₂ + r₃ = r := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radii_sum_l229_22983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_equation_l229_22950

/-- Given an ellipse and a hyperbola sharing the same foci, 
    prove that the hyperbola's asymptotes have a specific equation. -/
theorem hyperbola_asymptotes_equation 
  (a b : ℝ) 
  (h1 : a > b) 
  (h2 : b > 0) 
  (h3 : ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 → 
                   x^2 / a^2 - y^2 / b^2 = 1/2) :
  ∃ k : ℝ, k = Real.sqrt 3 / 3 ∧ 
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1/2 → 
              y = k * x ∨ y = -k * x) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_equation_l229_22950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_has_one_zero_l229_22921

-- Define the function g(x)
noncomputable def g (x : ℝ) : ℝ := Real.sin (Real.log x)

-- Define the interval (1, e^π)
def interval : Set ℝ := {x | 1 < x ∧ x < Real.exp Real.pi}

-- Theorem statement
theorem g_has_one_zero : ∃! x, x ∈ interval ∧ g x = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_has_one_zero_l229_22921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_t_is_smallest_l229_22998

/-- The smallest value of t for which the graph of r = sin θ encompasses the entire circle when 0 ≤ θ ≤ t -/
noncomputable def smallest_t : ℝ := Real.pi

/-- The graph of r = sin θ is a circle -/
axiom is_circle : ∀ θ : ℝ, ∃ r : ℝ, r = Real.sin θ ∧ r^2 + θ^2 = 1

/-- The resulting graph encompasses the entire circle when θ is in [0, t] -/
def encompasses_circle (t : ℝ) : Prop :=
  ∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ t → ∃ r : ℝ, r = Real.sin θ ∧ r^2 + θ^2 ≤ 1

/-- The theorem stating that smallest_t is the smallest value for which the graph encompasses the entire circle -/
theorem smallest_t_is_smallest :
  encompasses_circle smallest_t ∧
  ∀ t : ℝ, t < smallest_t → ¬(encompasses_circle t) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_t_is_smallest_l229_22998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_angle_sum_l229_22996

theorem tan_angle_sum (θ : Real) : 
  (π < θ ∧ θ < 3*π/2) →  -- θ is in the third quadrant
  Real.sin (θ - π/4) = 3/5 → 
  Real.tan (θ + π/4) = 4/3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_angle_sum_l229_22996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_on_line_formula_l229_22969

/-- The distance between two points on a line with slope m -/
noncomputable def distance_on_line (a c m : ℝ) : ℝ :=
  |a - c| * (1 + m^2).sqrt

/-- Theorem: The distance between two points (a, b) and (c, d) on the line y = mx + k
    is equal to |a-c| √(1+m²) -/
theorem distance_on_line_formula (a b c d m k : ℝ) :
  (b = m * a + k) →
  (d = m * c + k) →
  ((a - c)^2 + (b - d)^2).sqrt = distance_on_line a c m :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_on_line_formula_l229_22969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l229_22904

/-- 
Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
prove that if (a-c)sin A + c sin C - b sin B = 0, then:
1. B = π/3
2. The maximum value of sin A + sin C is √3, occurring when A = C = π/3
-/
theorem triangle_property (a b c A B C : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  (a - c) * Real.sin A + c * Real.sin C = b * Real.sin B →
  (B = π / 3) ∧ 
  (∃ (max : ℝ), max = Real.sqrt 3 ∧ 
    (∀ A' C', 0 < A' ∧ 0 < C' ∧ A' + C' = 2*π/3 → 
      Real.sin A' + Real.sin C' ≤ max)) ∧
  (Real.sin (π/3) + Real.sin (π/3) = Real.sqrt 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l229_22904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l229_22928

theorem equation_solution :
  ∃ x : ℝ, (x - 5)^3 = (1/16)^(-(1/2 : ℝ)) ↔ x = 5 + (4 : ℝ)^(1/3 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l229_22928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_APO_is_right_l229_22941

/-- Predicate to check if a point is inside a triangle -/
def IsInTriangle (P A B C : ℂ) : Prop := sorry

/-- Predicate to check if a point is the circumcenter of a triangle -/
def IsCircumcenter (O A B C : ℂ) : Prop := sorry

/-- Given a triangle ABC with a point P inside it such that ∠PAB = ∠PCA and ∠PAC = ∠PBA,
    and O ≠ P is the circumcenter of triangle ABC, then ∠APO is a right angle. -/
theorem angle_APO_is_right (A B C P O : ℂ) 
  (h_in : IsInTriangle P A B C)
  (h_ab : A ≠ B) (h_bc : B ≠ C) (h_ca : C ≠ A)
  (h_po : P ≠ O)
  (h_ang1 : (A - P).arg = (C - P).arg + (B - A).arg)
  (h_ang2 : (A - P).arg = (B - P).arg + (C - A).arg)
  (h_circ : IsCircumcenter O A B C) :
  (O - P).arg - (A - P).arg = π / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_APO_is_right_l229_22941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_lower_bound_f_inequality_iff_a_leq_one_l229_22944

-- Define the function f
noncomputable def f (a x : ℝ) : ℝ := (2 * x - 2 * a + 1) * Real.exp (2 * x - a)

-- Statement 1
theorem f_derivative_lower_bound (x : ℝ) (hx : x ≥ 1) :
  let a := 2
  HasDerivAt (f a) ((deriv (f a)) x) x → (deriv (f a)) x ≥ 4 * (x - 1) * x^2 := by sorry

-- Statement 2
theorem f_inequality_iff_a_leq_one (a : ℝ) :
  (∀ x, f a x - 2 * x + 1 ≥ 0) ↔ a ≤ 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_lower_bound_f_inequality_iff_a_leq_one_l229_22944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l229_22994

theorem range_of_a : 
  (∀ x : ℝ, 0 < x ∧ x < 4 → |x - a| < 3) ↔ (1 ≤ a ∧ a ≤ 3) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l229_22994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focal_length_theorem_l229_22985

/-- Definition of an ellipse with semi-major axis a and semi-minor axis b -/
def is_ellipse (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ ∃ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1

/-- Definition of focal length for an ellipse -/
noncomputable def focal_length (a b : ℝ) : ℝ :=
  2 * Real.sqrt (a^2 - b^2)

/-- Theorem: For an ellipse with equation x^2/a^2 + y^2/8 = 1 and focal length 4, a = 2√3 or a = 2 -/
theorem ellipse_focal_length_theorem (a : ℝ) :
  is_ellipse a (Real.sqrt 8) ∧ focal_length a (Real.sqrt 8) = 4 →
  a = 2 * Real.sqrt 3 ∨ a = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focal_length_theorem_l229_22985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_computation_l229_22905

theorem log_computation (c d : ℝ) (hc : c = Real.log 16 / Real.log 3) (hd : d = Real.log 81 / Real.log 4) :
  (9 : ℝ)^(c/d) + (4 : ℝ)^(d/c) = 528 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_computation_l229_22905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xyz_pattern_length_l229_22954

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Represents the "XYZ" pattern -/
structure XYZPattern where
  x1 : Point := ⟨0, 0⟩
  x2 : Point := ⟨2, 2⟩
  x3 : Point := ⟨2, 0⟩
  x4 : Point := ⟨0, 2⟩
  y1 : Point := ⟨3, 0⟩
  y2 : Point := ⟨3, 2⟩
  y3 : Point := ⟨4, 0⟩
  y4 : Point := ⟨2, 0⟩
  z1 : Point := ⟨5, 0⟩
  z2 : Point := ⟨7, 0⟩
  z3 : Point := ⟨5, 2⟩
  z4 : Point := ⟨7, 2⟩

/-- Calculates the total length of the XYZ pattern -/
noncomputable def totalLength (pattern : XYZPattern) : ℝ :=
  distance pattern.x1 pattern.x2 +
  distance pattern.x3 pattern.x4 +
  distance pattern.y1 pattern.y2 +
  distance pattern.y2 pattern.y3 +
  distance pattern.y2 pattern.y4 +
  distance pattern.z1 pattern.z2 +
  distance pattern.z3 pattern.z4 +
  distance pattern.z2 pattern.z3

theorem xyz_pattern_length :
  ∀ (pattern : XYZPattern), totalLength pattern = 7 + 7 * Real.sqrt 2 := by
  sorry

#eval "Compilation successful!"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_xyz_pattern_length_l229_22954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l229_22939

/-- The fractional part of a real number -/
noncomputable def frac (x : ℝ) : ℝ := x - ⌊x⌋

/-- The region described by the conditions -/
def Region : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 ≥ 0 ∧ p.2 ≥ 0 ∧ 50 * (frac p.1) ≥ ↑⌊p.1⌋ + 2 * ↑⌊p.2⌋}

/-- The area of the region -/
noncomputable def area : ℝ := 0.49

/-- Theorem stating that the area of the region is 0.49 -/
theorem area_of_region : area = 0.49 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l229_22939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equation_solution_l229_22932

theorem log_equation_solution (p y : ℝ) (hp : p > 0) (hy : y > 0) :
  (Real.log y / Real.log p) * (Real.log p / Real.log 3) = 4 → y = 81 := by
  intro h
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equation_solution_l229_22932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_proof_l229_22923

-- Define the arithmetic sequence a_n
noncomputable def a (n : ℕ) : ℝ := 2 - n

-- Define the sequence b_n
noncomputable def b (n : ℕ) : ℝ := 2^(a n)

-- Define the sum of first n terms of b_n
noncomputable def T (n : ℕ) : ℝ := 8 - 2^(2 - n)

-- Define the sum of first n terms of a_n
noncomputable def S (n : ℕ) : ℝ := n * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2

theorem arithmetic_sequence_proof :
  (a 2 = 0) ∧ 
  (S 5 = 2 * a 4 - 1) ∧
  (∀ n : ℕ, a (n + 1) - a n = a 2 - a 1) →
  (∀ n : ℕ, a n = 2 - n) ∧
  (∀ n : ℕ, T n = 8 - 2^(2 - n)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_proof_l229_22923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meal_cost_calculation_l229_22919

theorem meal_cost_calculation (initial_friends : ℚ) (additional_friends : ℚ) (cost_decrease : ℚ) (total_cost : ℚ) : 
  initial_friends = 4 →
  additional_friends = 3 →
  cost_decrease = 15 →
  (total_cost / initial_friends) - (total_cost / (initial_friends + additional_friends)) = cost_decrease →
  total_cost = 140 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_meal_cost_calculation_l229_22919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_problem_l229_22997

theorem tan_sum_problem (α β : ℝ) 
  (h1 : Real.tan α + Real.tan β = 25)
  (h2 : (Real.tan α)⁻¹ + (Real.tan β)⁻¹ = 30) : 
  Real.tan (α + β) = 150 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_problem_l229_22997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l229_22931

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi →
  -- Side lengths are positive
  0 < a ∧ 0 < b ∧ 0 < c →
  -- Law of Sines
  a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C →
  -- Given conditions
  Real.cos (2 * A) - 3 * Real.cos (B + C) = 1 →
  (1 / 2) * b * c * Real.sin A = 5 * Real.sqrt 3 →
  b = 5 →
  -- Conclusions
  A = Real.pi / 3 ∧ Real.sin B * Real.sin C = 5 / 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l229_22931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_values_when_a_is_one_monotonically_decreasing_condition_l229_22947

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a * x - 1) / (x + 1)

-- Theorem for part 1
theorem max_min_values_when_a_is_one :
  let f₁ := f 1
  ∀ x ∈ Set.Icc 0 3, f₁ x ≤ 1/2 ∧ f₁ x ≥ -1 ∧
  (∃ x₁ ∈ Set.Icc 0 3, f₁ x₁ = 1/2) ∧
  (∃ x₂ ∈ Set.Icc 0 3, f₁ x₂ = -1) := by
  sorry

-- Theorem for part 2
theorem monotonically_decreasing_condition (a : ℝ) :
  (∀ x₁ x₂, x₁ > x₂ ∧ x₁ > 0 ∧ x₂ > 0 → f a x₁ < f a x₂) ↔ a < -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_values_when_a_is_one_monotonically_decreasing_condition_l229_22947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cheese_division_l229_22958

theorem cheese_division (m : Fin 9 → ℝ) (h_distinct : ∀ i j, i ≠ j → m i ≠ m j) (h_positive : ∀ i, m i > 0) :
  ∃ (k : Fin 9) (x : ℝ), 0 < x ∧ x < m k ∧
    ∃ (S₁ S₂ : Finset (Fin 10)),
      S₁.card = 5 ∧ S₂.card = 5 ∧ S₁ ∩ S₂ = ∅ ∧ S₁ ∪ S₂ = Finset.univ ∧
      S₁.sum (λ i ↦ if i.val < 9 then m i else if i = ⟨9, by norm_num⟩ then x else m k - x) =
      S₂.sum (λ i ↦ if i.val < 9 then m i else if i = ⟨9, by norm_num⟩ then x else m k - x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cheese_division_l229_22958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_analysis_l229_22982

def standard_weight : ℕ := 500
def sample_size : ℕ := 20

structure WeightDifference :=
  (diff : ℤ)
  (count : ℕ)

def weight_differences : List WeightDifference := [
  ⟨-20, 4⟩, ⟨-5, 1⟩, ⟨0, 3⟩, ⟨2, 4⟩, ⟨3, 5⟩, ⟨10, 3⟩
]

theorem weight_analysis :
  (∃ (max min : ℤ), 
    max ∈ (weight_differences.map (λ w => w.diff)) ∧
    min ∈ (weight_differences.map (λ w => w.diff)) ∧
    max - min = 30) ∧
  (weight_differences.foldl (λ acc w => acc + w.diff * (w.count : ℤ)) 0 + 
   (sample_size * standard_weight : ℤ) = 9968) := by
  sorry

#eval weight_differences.foldl (λ acc w => acc + w.diff * (w.count : ℤ)) 0 + 
  (sample_size * standard_weight : ℤ)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_analysis_l229_22982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transfer_increases_averages_l229_22912

/-- Represents a group of students with their total score and count -/
structure StudentGroup where
  totalScore : ℚ
  count : ℕ

/-- Calculates the average score of a student group -/
def averageScore (group : StudentGroup) : ℚ :=
  group.totalScore / group.count

/-- Theorem: Transferring two students with scores between the initial averages 
    increases both groups' average scores -/
theorem transfer_increases_averages 
  (groupA groupB : StudentGroup) 
  (scoreL scoreF : ℚ) 
  (hA : groupA.count = 10)
  (hB : groupB.count = 10)
  (hAvgA : averageScore groupA = 472/10)
  (hAvgB : averageScore groupB = 418/10)
  (hScoreL : 418/10 < scoreL ∧ scoreL < 472/10)
  (hScoreF : 418/10 < scoreF ∧ scoreF < 472/10) :
  let newGroupA : StudentGroup := ⟨groupA.totalScore - scoreL - scoreF, groupA.count - 2⟩
  let newGroupB : StudentGroup := ⟨groupB.totalScore + scoreL + scoreF, groupB.count + 2⟩
  averageScore newGroupA > averageScore groupA ∧ 
  averageScore newGroupB > averageScore groupB := by
  sorry

#check transfer_increases_averages

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transfer_increases_averages_l229_22912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_999_value_l229_22984

def sequence_a : ℕ → ℚ
  | 0 => 1
  | (n + 1) => sequence_a n + (2 * sequence_a n) / (n + 1)

theorem a_999_value : sequence_a 999 = 499500 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_999_value_l229_22984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_condition_l229_22951

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := 3^(x + m) - 3 * Real.sqrt 3

-- Define the property of having no zeros in [1,+∞)
def has_no_zeros (m : ℝ) : Prop :=
  ∀ x : ℝ, x ≥ 1 → f m x ≠ 0

-- State the theorem
theorem sufficient_not_necessary_condition :
  (∀ m : ℝ, m > 1 → has_no_zeros m) ∧
  (∃ m : ℝ, m ≤ 1 ∧ has_no_zeros m) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_condition_l229_22951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_computable_values_l229_22929

/-- The expression 3^(3^(3^3)) -/
def original_expression (a b c : ℕ) : ℕ := a^(b^(c^3))

/-- The set of all possible parenthesizations of 3^(3^(3^3)) -/
def parenthesizations : List (ℕ → ℕ → ℕ → ℕ) := [
  original_expression,
  (λ a b c ↦ a^((b^c)^3)),
  (λ a b c ↦ (a^b)^(c^3)),
  (λ a b c ↦ (a^(b^c))^3),
  (λ a b c ↦ (a^b)^(c^3))
]

/-- A function is considered computable if it can be evaluated for a = b = c = 3 -/
def is_computable (f : ℕ → ℕ → ℕ → ℕ) : Bool :=
  match f 3 3 3 with
  | 0 => true  -- We use 0 as a placeholder for "computable"
  | _ => false

theorem distinct_computable_values :
  (parenthesizations.filter is_computable).length - 1 = 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_computable_values_l229_22929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_one_ninth_l229_22965

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log 3
  else 3^x

-- State the theorem
theorem f_composition_one_ninth : f (f (1/9)) = 1/9 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_one_ninth_l229_22965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_determines_p_l229_22970

/-- Triangle ABC with vertices A(2, 12), B(12, 0), and C(0, p) has area 27 if and only if p = 9 -/
theorem triangle_area_determines_p :
  ∀ p : ℝ,
  let A : ℝ × ℝ := (2, 12)
  let B : ℝ × ℝ := (12, 0)
  let C : ℝ × ℝ := (0, p)
  let triangle_area := abs ((A.1 - C.1) * (B.2 - C.2) - (B.1 - C.1) * (A.2 - C.2)) / 2
  triangle_area = 27 ↔ p = 9 := by
  sorry

#check triangle_area_determines_p

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_determines_p_l229_22970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_hyperbola_properties_l229_22976

/-- Represents a parabola with equation y² = 2px -/
structure Parabola where
  p : ℝ

/-- Represents a hyperbola with equation x²/3 - y² = 1 -/
structure Hyperbola where

/-- The shared right focus of the parabola and hyperbola -/
def shared_focus (par : Parabola) (hyp : Hyperbola) : Prop :=
  par.p / 2 = 2

/-- The equation of the parabola -/
def parabola_equation (par : Parabola) : Prop :=
  ∀ x y : ℝ, y^2 = 2 * par.p * x

/-- The area of the triangle formed by the latus rectum of the parabola
    and the asymptotes of the hyperbola -/
noncomputable def triangle_area (par : Parabola) (hyp : Hyperbola) : ℝ :=
  4 * Real.sqrt 3 / 3

theorem parabola_hyperbola_properties
  (par : Parabola) (hyp : Hyperbola)
  (h : shared_focus par hyp) :
  parabola_equation { p := 4 : Parabola } ∧
  triangle_area par hyp = 4 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_hyperbola_properties_l229_22976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_missing_digit_in_2_pow_29_l229_22920

theorem missing_digit_in_2_pow_29 : ∃ (digits : Finset Nat) (n : Nat),
  -- 2^29 has a 9-digit decimal representation
  10^8 ≤ 2^29 ∧ 2^29 < 10^9 ∧
  -- The decimal representation contains 9 distinct digits
  digits.card = 9 ∧
  -- All digits in the representation are less than 10
  ∀ d ∈ digits, d < 10 ∧
  -- The sum of the digits is congruent to 2^29 modulo 9
  (digits.sum id) % 9 = 2^29 % 9 ∧
  -- There exists a number n less than 10 that is not in the set of digits
  n < 10 ∧ n ∉ digits ∧
  -- The missing digit is 4
  n = 4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_missing_digit_in_2_pow_29_l229_22920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l229_22979

open Real

-- Define necessary functions
def distance (P Q : Point) : ℝ := sorry
def area_triangle (P Q R : Point) : ℝ := sorry
def height_from_side (P Q R : Point) : ℝ := sorry

theorem triangle_abc_properties (A B C : Point) (A_angle : ℝ) :
  -- A is an acute angle
  0 < A_angle ∧ A_angle < π / 2 →
  -- Given equation
  4 * sin (5 * π - A_angle) * (cos (A_angle / 2 - π / 4))^2 = Real.sqrt 3 * (sin (A_angle / 2) + cos (A_angle / 2))^2 →
  -- AC = 1
  distance A C = 1 →
  -- Area of triangle ABC is √3
  area_triangle A B C = Real.sqrt 3 →
  -- Conclusions
  A_angle = π / 3 ∧
  height_from_side A B C = (2 * Real.sqrt 39) / 13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l229_22979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_is_integer_l229_22925

noncomputable def frac (x : ℝ) : ℝ := x - ⌊x⌋

noncomputable def fracSeq (α : ℝ) (n : ℕ) : ℝ := frac (α ^ n)

def hasFinitelyManyDistinctValues (s : ℕ → ℝ) : Prop :=
  ∃ (k : ℕ), ∀ (n : ℕ), ∃ (m : ℕ), m < k ∧ s n = s m

theorem alpha_is_integer (α : ℝ) 
  (h : hasFinitelyManyDistinctValues (fracSeq α)) : 
  ∃ (n : ℤ), α = n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_is_integer_l229_22925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_a_equals_two_l229_22901

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1 / (x - a - 1)

-- Define the symmetry property
def isSymmetricAbout (g : ℝ → ℝ) (p : ℝ × ℝ) : Prop :=
  ∀ x, g (p.1 - (x - p.1)) = g x

-- Theorem statement
theorem symmetry_implies_a_equals_two (a : ℝ) :
  isSymmetricAbout (f a) (3, 0) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_a_equals_two_l229_22901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_points_perpendicular_bisector_property_l229_22900

/-- A point in a 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Checks if a point is on the perpendicular bisector of a line segment -/
def isOnPerpendicularBisector (p q r : Point2D) : Prop :=
  (p.x - q.x) * (r.x - (p.x + q.x) / 2) + (p.y - q.y) * (r.y - (p.y + q.y) / 2) = 0

/-- The main theorem stating the existence of 8 points with the desired property -/
theorem eight_points_perpendicular_bisector_property :
  ∃ (points : Finset Point2D),
    points.card = 8 ∧
    ∀ p q, p ∈ points → q ∈ points → p ≠ q →
      ∃ r s, r ∈ points ∧ s ∈ points ∧ r ≠ s ∧ r ≠ p ∧ r ≠ q ∧ s ≠ p ∧ s ≠ q ∧
        isOnPerpendicularBisector p q r ∧ isOnPerpendicularBisector p q s :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_points_perpendicular_bisector_property_l229_22900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_properties_l229_22902

open Real

-- Define α as a real number
variable (α : ℝ)

-- Define the conditions
def second_quadrant (α : ℝ) : Prop := π/2 < α ∧ α < π
noncomputable def sin_alpha : ℝ := 4/5

-- Define the theorem
theorem angle_properties (h1 : second_quadrant α) (h2 : sin α = sin_alpha) :
  (tan α = -4/3) ∧ 
  ((sin (π + α) - 2 * cos (π/2 + α)) / (-sin (-α) + cos (π - α)) = 4/7) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_properties_l229_22902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l229_22973

/-- Calculates the length of a train given its speed, the speed of a man walking in the opposite direction, and the time it takes for the train to pass the man. -/
noncomputable def train_length (train_speed : ℝ) (man_speed : ℝ) (passing_time : ℝ) : ℝ :=
  let relative_speed := train_speed + man_speed
  let relative_speed_mps := relative_speed * (5 / 18)
  relative_speed_mps * passing_time

/-- Theorem stating that a train with the given parameters has a length of approximately 129.99 meters. -/
theorem train_length_calculation :
  let train_speed := 72.99376049916008
  let man_speed := 5
  let passing_time := 6
  ∃ ε > 0, |train_length train_speed man_speed passing_time - 129.99| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l229_22973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_50_value_l229_22937

def b : ℕ → ℕ
  | 0 => 5  -- Add this case to cover Nat.zero
  | 1 => 5
  | n + 1 => b n + 3 * n + 1

theorem b_50_value : b 50 = 3729 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_50_value_l229_22937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sum_1000_l229_22924

theorem arithmetic_sum_1000 :
  {(m, n) : ℕ × ℕ | m > 0 ∧ n > 0 ∧ (m + 1) * n + m * (m + 1) / 2 = 1000} =
  {(15, 55), (24, 28), (4, 198)} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sum_1000_l229_22924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_trapezoid_DBCE_l229_22940

/-- Represents a nested triangle structure -/
structure NestedTriangle where
  level : ℕ
  area : ℝ

/-- The problem setup -/
structure NestedTriangleProblem where
  levels : ℕ
  smallestArea : ℝ
  areaRatio : ℝ
  adeLevel : ℕ

/-- Calculate the area of a triangle at a given level -/
def triangleArea (problem : NestedTriangleProblem) (level : ℕ) : ℝ :=
  problem.smallestArea * problem.areaRatio ^ (problem.levels - level)

/-- Calculate the sum of areas of triangles from a given level to the deepest level -/
def sumAreasFromLevel (problem : NestedTriangleProblem) (startLevel : ℕ) : ℝ :=
  (List.range (problem.levels - startLevel + 1)).map (fun i => triangleArea problem (startLevel + i))
    |> List.sum

/-- The main theorem to prove -/
theorem area_of_trapezoid_DBCE (problem : NestedTriangleProblem)
    (h1 : problem.levels = 4)
    (h2 : problem.smallestArea = 1)
    (h3 : problem.areaRatio = 2)
    (h4 : problem.adeLevel = 2) :
    triangleArea problem 0 - sumAreasFromLevel problem problem.adeLevel = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_trapezoid_DBCE_l229_22940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_ratio_l229_22926

open BigOperators

/-- Given arithmetic sequences a_n and b_n with S_n and T_n as their respective sums of first n terms,
    such that S_n / T_n = (2n + 1) / (4n - 2) for n ∈ ℕ*, 
    prove that a_10 / (b_3 + b_18) + a_11 / (b_6 + b_15) = 41 / 78 -/
theorem arithmetic_sequence_sum_ratio 
  (a b : ℕ → ℝ) 
  (h : ∀ n : ℕ, n > 0 → (∑ i in Finset.range n, a i) / (∑ i in Finset.range n, b i) = (2 * n + 1) / (4 * n - 2)) :
  a 9 / (b 2 + b 17) + a 10 / (b 5 + b 14) = 41 / 78 := by
  sorry -- Proof steps would go here


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_ratio_l229_22926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equations_equivalence_equation_II_eq_III_equation_I_neq_II_l229_22977

-- Define the three equations
def equation_I (x y : ℝ) : Prop := y = x - 2
def equation_II (x y : ℝ) : Prop := y = Real.sin (x^2 - 4) / (x + 2)
def equation_III (x y : ℝ) : Prop := (x + 2) * y = Real.sin (x^2 - 4)

-- Theorem stating that II and III are equivalent, while I is different
theorem equations_equivalence :
  (∀ x y : ℝ, equation_II x y ↔ equation_III x y) ∧
  (∃ x y : ℝ, equation_I x y ≠ equation_II x y) := by
  sorry

-- Additional helper theorem to show that II and III are indeed equivalent
theorem equation_II_eq_III (x y : ℝ) : equation_II x y ↔ equation_III x y := by
  sorry

-- Theorem to show that I is different from II (and thus from III)
theorem equation_I_neq_II : ∃ x y : ℝ, equation_I x y ≠ equation_II x y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equations_equivalence_equation_II_eq_III_equation_I_neq_II_l229_22977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_point_is_orthocenter_l229_22917

/-- Triangle ABC on a coordinate plane -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Point U with the special property -/
noncomputable def special_point (t : Triangle) : ℝ × ℝ := sorry

/-- Perpendicular foot from a point to a line -/
noncomputable def perpendicular_foot (P : ℝ × ℝ) (line : (ℝ × ℝ) × (ℝ × ℝ)) : ℝ × ℝ := sorry

/-- Distance between two points -/
noncomputable def distance (P Q : ℝ × ℝ) : ℝ := sorry

/-- The special property condition -/
def satisfies_condition (t : Triangle) (U : ℝ × ℝ) : Prop :=
  ∃ (l m n k : ℝ), (l ≠ 0 ∨ m ≠ 0 ∨ n ≠ 0 ∨ k ≠ 0) ∧
    ∀ (P : ℝ × ℝ), ∃ (c : ℝ),
      let L := perpendicular_foot P (t.B, t.C)
      let M := perpendicular_foot P (t.C, t.A)
      let N := perpendicular_foot P (t.A, t.B)
      l * (distance P L)^2 + m * (distance P M)^2 + n * (distance P N)^2 - k * (distance P U)^2 = c

/-- Orthocenter of a triangle -/
noncomputable def orthocenter (t : Triangle) : ℝ × ℝ := sorry

/-- Main theorem -/
theorem special_point_is_orthocenter (t : Triangle) :
  let U := special_point t
  satisfies_condition t U ∧ U = orthocenter t ∧
  U.1 = Real.cos t.A.1 + Real.cos t.B.1 + Real.cos t.C.1 ∧
  U.2 = Real.sin t.A.2 + Real.sin t.B.2 + Real.sin t.C.2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_point_is_orthocenter_l229_22917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_equation_solutions_l229_22975

theorem floor_equation_solutions : 
  (Finset.filter (fun x : ℕ => 
    x > 0 ∧ x < 220 ∧ 
    (Int.floor (x / 10 : ℚ) = Int.floor (x / 11 : ℚ) + 1)) 
    (Finset.range 220)).card = 110 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_equation_solutions_l229_22975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alloy_cut_weight_l229_22935

/-- Given two alloys with different copper percentages, prove that the weight cut from each to equalize the copper percentages after melting is 2.1 kg. -/
theorem alloy_cut_weight (x y : ℝ) (hxy : x ≠ y) : 
  ∃ (z : ℝ), z > 0 ∧ z < 3 ∧ z < 7 ∧
  (z * x + (3 - z) * y) / 3 = ((7 - z) * x + z * y) / 7 ∧
  z = 2.1 := by
  sorry

#check alloy_cut_weight

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alloy_cut_weight_l229_22935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_is_11_l229_22943

/-- The time taken to complete the work when a, b, and c work together, with c leaving 4 days before completion -/
noncomputable def workCompletionTime (a b c : ℝ) : ℝ :=
  let totalWorkPerDay := 1 / a + 1 / b + 1 / c
  let workDoneWithoutC := 1 / a + 1 / b
  ((1 - 4 * workDoneWithoutC) / totalWorkPerDay) + 4

theorem work_completion_time_is_11 :
  workCompletionTime 24 30 40 = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_is_11_l229_22943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_arithmetic_sequence_l229_22938

noncomputable def arithmeticSequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1) * d

noncomputable def sumArithmeticSequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem min_sum_arithmetic_sequence :
  ∀ (a₁ d : ℝ) (n : ℕ),
    a₁ = -10 →
    arithmeticSequence a₁ d 4 + arithmeticSequence a₁ d 6 = -4 →
    (∃ (m : ℕ), m = 5 ∨ m = 6) ∧
    (∀ (k : ℕ), sumArithmeticSequence a₁ d m ≤ sumArithmeticSequence a₁ d k) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_arithmetic_sequence_l229_22938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_difference_bound_l229_22999

/-- A structure representing a finite, non-empty set of real numbers -/
structure FiniteRealSet where
  values : List ℝ
  nonempty : values.length > 0

/-- The direct sum of two FiniteRealSets -/
def directSum (X Y : FiniteRealSet) : FiniteRealSet :=
  { values := X.values.bind (λ x => Y.values.map (λ y => x + y)),
    nonempty := by sorry }

/-- The median of a FiniteRealSet -/
noncomputable def median (S : FiniteRealSet) : ℝ := sorry

/-- Main theorem: The difference between the median of the direct sum and the sum of individual medians is ≤ 4 -/
theorem median_difference_bound (X Y : FiniteRealSet)
  (h_min : ∀ y ∈ Y.values, y ≥ 1)
  (h_max : ∀ y ∈ Y.values, y ≤ 5) :
  median (directSum X Y) - (median X + median Y) ≤ 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_difference_bound_l229_22999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_equation_l229_22963

/-- The intersection point of two lines -/
def intersection_point (l1 l2 : ℝ → ℝ → Prop) : ℝ × ℝ :=
  sorry

/-- The slope of a line -/
noncomputable def line_slope (l : ℝ → ℝ → Prop) : ℝ :=
  sorry

/-- Check if two lines are perpendicular -/
def perpendicular (l1 l2 : ℝ → ℝ → Prop) : Prop :=
  sorry

/-- The equation of a line passing through a point with a given slope -/
def line_equation (p : ℝ × ℝ) (m : ℝ) : ℝ → ℝ → Prop :=
  sorry

theorem perpendicular_line_equation :
  let l1 : ℝ → ℝ → Prop := λ x y ↦ x + y - 3 = 0
  let l2 : ℝ → ℝ → Prop := λ x y ↦ 2*x - y = 0
  let l3 : ℝ → ℝ → Prop := λ x y ↦ 2*x + y - 5 = 0
  let p := intersection_point l1 l2
  let m := -1 / (line_slope l3)
  let l4 := line_equation p m
  perpendicular l3 l4 ∧ l4 = (λ x y ↦ x - 2*y + 3 = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_equation_l229_22963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_n_greater_than_10_pow_10_l229_22936

def sequence' (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => Nat.factorial (sequence' n) + 1

theorem least_n_greater_than_10_pow_10 :
  (∃ n, sequence' n > 10^10) ∧
  (∀ k, k < 6 → sequence' k ≤ 10^10) ∧
  sequence' 6 > 10^10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_n_greater_than_10_pow_10_l229_22936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_l229_22952

/-- The function f(x) = sin x + cos x -/
noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x

/-- The point of tangency -/
noncomputable def point : ℝ × ℝ := (0, f 0)

/-- The equation of the tangent line -/
def tangent_line (x y : ℝ) : Prop := x - y + 1 = 0

theorem tangent_line_at_zero :
  ∀ x y : ℝ, (x, y) ∈ ({(x, y) : ℝ × ℝ | tangent_line x y} : Set (ℝ × ℝ)) ↔
  y = (deriv f point.fst) * (x - point.fst) + f point.fst :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_l229_22952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_3_20_l229_22957

noncomputable def hour_hand_degrees_per_hour : ℝ := 30
noncomputable def minute_hand_degrees_per_minute : ℝ := 6

noncomputable def hour_hand_position (hours : ℕ) (minutes : ℕ) : ℝ :=
  (hours % 12 : ℝ) * hour_hand_degrees_per_hour + 
  (minutes : ℝ) * hour_hand_degrees_per_hour / 60

noncomputable def minute_hand_position (minutes : ℕ) : ℝ :=
  (minutes : ℝ) * minute_hand_degrees_per_minute

noncomputable def angle_between_hands (hours : ℕ) (minutes : ℕ) : ℝ :=
  let hour_pos := hour_hand_position hours minutes
  let minute_pos := minute_hand_position minutes
  abs (minute_pos - hour_pos)

theorem clock_angle_at_3_20 : 
  angle_between_hands 3 20 = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_3_20_l229_22957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l229_22915

theorem tan_alpha_value (α : ℝ) (h1 : α ∈ Set.Ioo 0 π) 
  (h2 : Real.cos (2 * α) / Real.sin (α - π / 4) = Real.sqrt 2 / 5) : 
  Real.tan α = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l229_22915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_square_center_l229_22967

/-- Given a right triangle ABC with BC = a and AC = b, the distance from C to the center
    of a square constructed on the hypotenuse AB is √((a² + b²)/2). -/
theorem distance_to_square_center (a b : ℝ) (h : 0 < a ∧ 0 < b) :
  Real.sqrt ((a^2 + b^2) / 2) = Real.sqrt (a^2 + b^2) / Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_square_center_l229_22967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_james_clothing_cost_l229_22992

/-- Calculates the total cost of fixing clothing items --/
def calculate_total_cost (num_shirts num_pants num_jackets num_ties : ℕ)
  (time_shirt : ℚ) (time_jacket time_tie : ℚ)
  (rate_shirt_pants rate_jacket rate_tie : ℕ) : ℤ :=
  let time_pants := 2 * time_shirt
  let total_time_shirts := num_shirts * time_shirt
  let total_time_pants := num_pants * time_pants
  let total_time_jackets := num_jackets * time_jacket
  let total_time_ties := num_ties * time_tie
  let cost_shirts := (total_time_shirts * rate_shirt_pants).floor
  let cost_pants := (total_time_pants * rate_shirt_pants).floor
  let cost_jackets := (total_time_jackets * rate_jacket).floor
  let cost_ties := (total_time_ties * rate_tie).floor
  cost_shirts + cost_pants + cost_jackets + cost_ties

/-- Theorem stating the total cost of fixing James' clothing items --/
theorem james_clothing_cost :
  calculate_total_cost 10 12 8 5 (3/2) (5/2) (1/2) 30 40 20 = 2380 := by
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_james_clothing_cost_l229_22992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_C₁_C₂_l229_22978

-- Define the curves C₁ and C₂
def C₁ (x y : ℝ) : Prop := y = x^2 - 1 ∧ x^2 ≤ 2
def C₂ (x y : ℝ) : Prop := x + y + 2 = 0

-- Define the distance function between two points
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

-- State the theorem
theorem min_distance_C₁_C₂ :
  ∃ (d : ℝ), d = 3 * Real.sqrt 2 / 8 ∧
  ∀ (x₁ y₁ x₂ y₂ : ℝ), C₁ x₁ y₁ → C₂ x₂ y₂ →
  distance x₁ y₁ x₂ y₂ ≥ d := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_C₁_C₂_l229_22978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_value_when_y_is_15_l229_22991

/-- The constant ratio between (5x - 6) and (2y + 20) -/
noncomputable def k (x y : ℝ) : ℝ := (5 * x - 6) / (2 * y + 20)

/-- The theorem stating the relationship between x and y -/
theorem x_value_when_y_is_15 (h1 : k 3 5 = k x 15) (h2 : k 3 5 = 3 / 10) : x = 21 / 5 := by
  sorry

#check x_value_when_y_is_15

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_value_when_y_is_15_l229_22991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_yes_men_l229_22981

/-- Represents the types of inhabitants on the island -/
inductive Inhabitant
  | Knight
  | Liar
  | YesMan

/-- Represents a possible answer to the question -/
inductive Answer
  | Yes
  | No

/-- The total number of inhabitants -/
def total_inhabitants : Nat := 2018

/-- The number of 'Yes' answers -/
def yes_answers : Nat := 1009

/-- Function to determine if there are more knights than liars -/
def more_knights_than_liars (knights liars : Nat) : Prop :=
  knights > liars

/-- Function to determine a yes-man's answer based on previous answers -/
def yes_man_answer (previous_yes previous_no : Nat) : Answer :=
  if previous_yes ≥ previous_no then Answer.Yes else Answer.No

/-- Theorem stating the maximum number of yes-men possible -/
theorem max_yes_men :
  ∃ (knights liars yes_men : Nat),
    knights + liars + yes_men = total_inhabitants ∧
    yes_men = 1009 ∧
    (∀ (k l ym : Nat), k + l + ym = total_inhabitants →
                       ym ≤ yes_men) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_yes_men_l229_22981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_change_l229_22946

theorem rectangle_area_change (L W : ℝ) (h1 : L > 0) (h2 : W > 0) :
  (1.2 * L * 0.8 * W - L * W) / (L * W) = -0.04 := by
  -- Simplify the expression
  have h3 : 1.2 * L * 0.8 * W = 0.96 * L * W := by
    ring
  
  -- Substitute the simplified expression
  rw [h3]
  
  -- Simplify the numerator
  have h4 : 0.96 * L * W - L * W = -0.04 * L * W := by
    ring
  
  -- Substitute the simplified numerator
  rw [h4]
  
  -- Simplify the fraction
  field_simp
  ring
  
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_change_l229_22946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_numbers_in_set_with_mean_l229_22945

theorem equal_numbers_in_set_with_mean (S : Finset ℕ) (h : S.Nonempty) :
  (S.sum (λ x => (x : ℚ))) / S.card = 1011 / 50 →
  ∃ a b : ℕ, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ a = b :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_numbers_in_set_with_mean_l229_22945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_is_zero_l229_22914

-- Define the integer part function
noncomputable def integerPart (x : ℝ) : ℤ :=
  Int.floor x

-- State the theorem
theorem unique_solution_is_zero :
  ∀ x : ℝ, (integerPart ((x + 3) / 2))^2 - x = 1 ↔ x = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_is_zero_l229_22914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_other_x_intercept_l229_22903

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  focus1 : Point
  focus2 : Point

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: The other x-intercept of the ellipse -/
theorem ellipse_other_x_intercept 
  (e : Ellipse)
  (h1 : e.focus1 = ⟨1, 2⟩)
  (h2 : e.focus2 = ⟨4, 0⟩)
  (h3 : ∃ (p : Point), p.x = 1 ∧ p.y = 0 ∧ 
       distance p e.focus1 + distance p e.focus2 = 
       distance ⟨2, 0⟩ e.focus1 + distance ⟨2, 0⟩ e.focus2) :
  ∃ (q : Point), q.x = 2 ∧ q.y = 0 ∧
    distance q e.focus1 + distance q e.focus2 = 
    distance ⟨1, 0⟩ e.focus1 + distance ⟨1, 0⟩ e.focus2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_other_x_intercept_l229_22903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_y_l229_22916

noncomputable def y (n : ℕ) : ℝ :=
  match n with
  | 0 => 4^(1/4)
  | n+1 => (y n)^(4^(1/4))

theorem smallest_integer_y : 
  (∀ k < 8, ¬ (∃ m : ℤ, y k = m)) ∧ (∃ m : ℤ, y 8 = m) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_y_l229_22916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jack_diet_duration_l229_22906

/-- Represents the weight loss problem for Jack --/
structure WeightLossProblem where
  initial_weight : ℚ
  current_weight : ℚ
  future_weight : ℚ
  months_to_future : ℚ
  weight_loss_rate : ℚ

/-- Calculates the number of months since Jack started his diet --/
def months_since_start (p : WeightLossProblem) : ℚ :=
  (p.initial_weight - p.current_weight) / p.weight_loss_rate

/-- Theorem stating that the calculated time since Jack started his diet is approximately 26 months --/
theorem jack_diet_duration (p : WeightLossProblem) 
  (h1 : p.initial_weight = 222)
  (h2 : p.current_weight = 198)
  (h3 : p.future_weight = 180)
  (h4 : p.months_to_future = 45)
  (h5 : p.weight_loss_rate = (p.initial_weight - p.future_weight) / (months_since_start p + p.months_to_future)) :
  ∃ ε : ℚ, ε > 0 ∧ |months_since_start p - 26| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jack_diet_duration_l229_22906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_AGE_l229_22989

/-- Square ABCD with side length 5 -/
structure Square :=
  (A B C D : ℝ × ℝ)
  (is_square : A = (0, 0) ∧ B = (5, 0) ∧ C = (5, 5) ∧ D = (0, 5))

/-- Point E on side BC -/
noncomputable def E : ℝ × ℝ := (5, 2)

/-- Point G on diagonal BD -/
noncomputable def G : ℝ × ℝ := (3, 2)

/-- Area of a triangle given three points -/
noncomputable def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (1/2) * abs (x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))

/-- Theorem: The area of triangle AGE is 7.5 -/
theorem area_of_triangle_AGE (sq : Square) : 
  triangle_area sq.A G E = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_AGE_l229_22989
