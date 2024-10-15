import Mathlib

namespace NUMINAMATH_CALUDE_solution_equation1_solution_equation2_l3772_377256

-- Define the equations
def equation1 (x : ℝ) : Prop := 2 * (2 * x + 1) - (3 * x - 4) = 2
def equation2 (y : ℝ) : Prop := (3 * y - 1) / 4 - 1 = (5 * y - 7) / 6

-- Theorem statements
theorem solution_equation1 : ∃ x : ℝ, equation1 x ∧ x = -4 := by sorry

theorem solution_equation2 : ∃ y : ℝ, equation2 y ∧ y = -1 := by sorry

end NUMINAMATH_CALUDE_solution_equation1_solution_equation2_l3772_377256


namespace NUMINAMATH_CALUDE_constant_kill_time_l3772_377257

/-- Represents the time taken for lions to kill deers -/
def killTime (numLions : ℕ) : ℕ := 13

/-- The assumption that 13 lions can kill 13 deers in 13 minutes -/
axiom base_case : killTime 13 = 13

/-- Theorem stating that for any number of lions (equal to deers), 
    the time taken to kill all deers is always 13 minutes -/
theorem constant_kill_time (n : ℕ) : killTime n = 13 := by
  sorry

end NUMINAMATH_CALUDE_constant_kill_time_l3772_377257


namespace NUMINAMATH_CALUDE_point_movement_l3772_377228

/-- Represents a point on a number line -/
structure Point where
  value : ℤ

/-- Moves a point on the number line -/
def movePoint (p : Point) (distance : ℤ) : Point :=
  { value := p.value + distance }

theorem point_movement :
  let a : Point := { value := -3 }
  let b : Point := movePoint a 7
  b.value = 4 := by sorry

end NUMINAMATH_CALUDE_point_movement_l3772_377228


namespace NUMINAMATH_CALUDE_overtime_calculation_l3772_377273

/-- Calculates the number of overtime hours worked given the total gross pay, regular hourly rate, overtime hourly rate, and regular hours limit. -/
def overtime_hours (gross_pay : ℚ) (regular_rate : ℚ) (overtime_rate : ℚ) (regular_hours_limit : ℕ) : ℕ :=
  sorry

/-- The number of overtime hours worked is 10 given the specified conditions. -/
theorem overtime_calculation :
  let gross_pay : ℚ := 622
  let regular_rate : ℚ := 11.25
  let overtime_rate : ℚ := 16
  let regular_hours_limit : ℕ := 40
  overtime_hours gross_pay regular_rate overtime_rate regular_hours_limit = 10 := by
  sorry

end NUMINAMATH_CALUDE_overtime_calculation_l3772_377273


namespace NUMINAMATH_CALUDE_equation_solutions_l3772_377221

theorem equation_solutions : 
  ∃ (x₁ x₂ x₃ x₄ : ℝ), 
    (x₁ = (-1 + Real.sqrt 6) / 5 ∧ 
     x₂ = (-1 - Real.sqrt 6) / 5 ∧ 
     5 * x₁^2 + 2 * x₁ - 1 = 0 ∧ 
     5 * x₂^2 + 2 * x₂ - 1 = 0) ∧
    (x₃ = 3 ∧ 
     x₄ = -4 ∧ 
     x₃ * (x₃ - 3) - 4 * (3 - x₃) = 0 ∧ 
     x₄ * (x₄ - 3) - 4 * (3 - x₄) = 0) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l3772_377221


namespace NUMINAMATH_CALUDE_sphere_surface_area_relation_l3772_377265

theorem sphere_surface_area_relation (R₁ R₂ R₃ S₁ S₂ S₃ : ℝ) 
  (h₁ : R₁ + 2 * R₂ = 3 * R₃)
  (h₂ : S₁ = 4 * Real.pi * R₁^2)
  (h₃ : S₂ = 4 * Real.pi * R₂^2)
  (h₄ : S₃ = 4 * Real.pi * R₃^2) :
  Real.sqrt S₁ + 2 * Real.sqrt S₂ = 3 * Real.sqrt S₃ := by
  sorry

#check sphere_surface_area_relation

end NUMINAMATH_CALUDE_sphere_surface_area_relation_l3772_377265


namespace NUMINAMATH_CALUDE_license_plate_difference_l3772_377284

/-- The number of possible license plates for Sunland -/
def sunland_plates : ℕ := 1 * (10^3) * (26^2)

/-- The number of possible license plates for Moonland -/
def moonland_plates : ℕ := (10^2) * (26^2) * (10^2)

/-- The theorem stating the difference in the number of license plates -/
theorem license_plate_difference : moonland_plates - sunland_plates = 6084000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_difference_l3772_377284


namespace NUMINAMATH_CALUDE_catherine_pens_problem_l3772_377210

theorem catherine_pens_problem (initial_pens initial_pencils : ℕ) :
  initial_pens = initial_pencils →
  initial_pens - 7 * 8 + initial_pencils - 7 * 6 = 22 →
  initial_pens = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_catherine_pens_problem_l3772_377210


namespace NUMINAMATH_CALUDE_triangle_inequality_l3772_377206

/-- A triangle with sides x, y, and z satisfies the inequality
    (x+y+z)(x+y-z)(x+z-y)(z+y-x) ≤ 4x²y² -/
theorem triangle_inequality (x y z : ℝ) (h : 0 < x ∧ 0 < y ∧ 0 < z ∧ x + y > z ∧ x + z > y ∧ y + z > x) :
  (x + y + z) * (x + y - z) * (x + z - y) * (z + y - x) ≤ 4 * x^2 * y^2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l3772_377206


namespace NUMINAMATH_CALUDE_regular_pay_is_three_l3772_377246

/-- Calculates the regular hourly pay rate given total pay, regular hours, overtime hours, and overtime pay rate multiplier. -/
def regularHourlyPay (totalPay : ℚ) (regularHours : ℚ) (overtimeHours : ℚ) (overtimeMultiplier : ℚ) : ℚ :=
  totalPay / (regularHours + overtimeHours * overtimeMultiplier)

/-- Proves that the regular hourly pay is $3 given the problem conditions. -/
theorem regular_pay_is_three :
  let totalPay : ℚ := 192
  let regularHours : ℚ := 40
  let overtimeHours : ℚ := 12
  let overtimeMultiplier : ℚ := 2
  regularHourlyPay totalPay regularHours overtimeHours overtimeMultiplier = 3 := by
  sorry

#eval regularHourlyPay 192 40 12 2

end NUMINAMATH_CALUDE_regular_pay_is_three_l3772_377246


namespace NUMINAMATH_CALUDE_symmetry_properties_l3772_377266

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def symmetry_x_axis (p : Point3D) : Point3D :=
  ⟨p.x, -p.y, -p.z⟩

def symmetry_yOz_plane (p : Point3D) : Point3D :=
  ⟨-p.x, p.y, p.z⟩

def symmetry_y_axis (p : Point3D) : Point3D :=
  ⟨-p.x, p.y, -p.z⟩

def symmetry_origin (p : Point3D) : Point3D :=
  ⟨-p.x, -p.y, -p.z⟩

theorem symmetry_properties (p : Point3D) :
  (symmetry_x_axis p = ⟨p.x, -p.y, -p.z⟩) ∧
  (symmetry_yOz_plane p = ⟨-p.x, p.y, p.z⟩) ∧
  (symmetry_y_axis p = ⟨-p.x, p.y, -p.z⟩) ∧
  (symmetry_origin p = ⟨-p.x, -p.y, -p.z⟩) := by
  sorry

end NUMINAMATH_CALUDE_symmetry_properties_l3772_377266


namespace NUMINAMATH_CALUDE_line_point_sum_l3772_377219

/-- The line equation y = -5/3x + 15 -/
def line_equation (x y : ℝ) : Prop := y = -5/3 * x + 15

/-- Point P is where the line crosses the x-axis -/
def point_P : ℝ × ℝ := (9, 0)

/-- Point Q is where the line crosses the y-axis -/
def point_Q : ℝ × ℝ := (0, 15)

/-- Point T is on the line segment PQ -/
def point_T_on_PQ (r s : ℝ) : Prop :=
  ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧
  r = t * point_P.1 + (1 - t) * point_Q.1 ∧
  s = t * point_P.2 + (1 - t) * point_Q.2

/-- The area of triangle POQ is twice the area of triangle TOP -/
def area_condition (r s : ℝ) : Prop :=
  abs ((point_P.1 - 0) * (point_Q.2 - 0) - (point_Q.1 - 0) * (point_P.2 - 0)) / 2 =
  2 * abs ((point_P.1 - 0) * (s - 0) - (r - 0) * (point_P.2 - 0)) / 2

theorem line_point_sum (r s : ℝ) :
  line_equation r s →
  point_T_on_PQ r s →
  area_condition r s →
  r + s = 12 := by sorry

end NUMINAMATH_CALUDE_line_point_sum_l3772_377219


namespace NUMINAMATH_CALUDE_total_marbles_after_exchanges_l3772_377249

def initial_green : ℕ := 32
def initial_violet : ℕ := 38
def initial_blue : ℕ := 46

def mike_takes_green : ℕ := 23
def mike_gives_red : ℕ := 15
def alice_takes_violet : ℕ := 15
def alice_gives_yellow : ℕ := 20
def bob_takes_blue : ℕ := 31
def bob_gives_white : ℕ := 12

def mike_returns_green : ℕ := 10
def mike_takes_red : ℕ := 7
def alice_returns_violet : ℕ := 8
def alice_takes_yellow : ℕ := 9
def bob_returns_blue : ℕ := 17
def bob_takes_white : ℕ := 5

def final_green : ℕ := initial_green - mike_takes_green + mike_returns_green
def final_violet : ℕ := initial_violet - alice_takes_violet + alice_returns_violet
def final_blue : ℕ := initial_blue - bob_takes_blue + bob_returns_blue
def final_red : ℕ := mike_gives_red - mike_takes_red
def final_yellow : ℕ := alice_gives_yellow - alice_takes_yellow
def final_white : ℕ := bob_gives_white - bob_takes_white

theorem total_marbles_after_exchanges :
  final_green + final_violet + final_blue + final_red + final_yellow + final_white = 108 :=
by sorry

end NUMINAMATH_CALUDE_total_marbles_after_exchanges_l3772_377249


namespace NUMINAMATH_CALUDE_probability_sum_greater_than_five_l3772_377294

def A : Finset ℕ := {1, 2, 3}
def B : Finset ℕ := {2, 3, 4}

def sample_space : Finset (ℕ × ℕ) := A.product B

def favorable_outcomes : Finset (ℕ × ℕ) := sample_space.filter (fun p => p.1 + p.2 > 5)

theorem probability_sum_greater_than_five :
  (favorable_outcomes.card : ℚ) / sample_space.card = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_probability_sum_greater_than_five_l3772_377294


namespace NUMINAMATH_CALUDE_jeffs_score_l3772_377299

theorem jeffs_score (jeff tim : ℕ) (h1 : jeff = tim + 60) (h2 : (jeff + tim) / 2 = 112) : jeff = 142 := by
  sorry

end NUMINAMATH_CALUDE_jeffs_score_l3772_377299


namespace NUMINAMATH_CALUDE_square_side_length_l3772_377286

/-- A square with perimeter 24 meters has sides of length 6 meters. -/
theorem square_side_length (s : ℝ) (h1 : s > 0) (h2 : 4 * s = 24) : s = 6 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l3772_377286


namespace NUMINAMATH_CALUDE_clara_older_than_alice_l3772_377297

/-- Represents a person with their age and number of pens -/
structure Person where
  age : ℕ
  pens : ℕ

/-- The problem statement -/
theorem clara_older_than_alice (alice clara : Person)
  (h1 : alice.pens = 60)
  (h2 : clara.pens = 2 * alice.pens / 5)
  (h3 : alice.pens - clara.pens = alice.age - clara.age)
  (h4 : alice.age = 20)
  (h5 : clara.age + 5 = 61) :
  clara.age > alice.age := by
  sorry

#check clara_older_than_alice

end NUMINAMATH_CALUDE_clara_older_than_alice_l3772_377297


namespace NUMINAMATH_CALUDE_integral_equality_l3772_377271

theorem integral_equality : ∫ (x : ℝ) in (0 : ℝ)..(1 : ℝ), (Real.sqrt (1 - (x - 1)^2) - x) = (Real.pi - 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_integral_equality_l3772_377271


namespace NUMINAMATH_CALUDE_triangle_problem_l3772_377224

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The law of sines for a triangle -/
def lawOfSines (t : Triangle) : Prop :=
  t.a / (Real.sin t.A) = t.b / (Real.sin t.B) ∧ 
  t.b / (Real.sin t.B) = t.c / (Real.sin t.C)

/-- The law of cosines for a triangle -/
def lawOfCosines (t : Triangle) : Prop :=
  t.a^2 = t.b^2 + t.c^2 - 2*t.b*t.c*(Real.cos t.A) ∧
  t.b^2 = t.a^2 + t.c^2 - 2*t.a*t.c*(Real.cos t.B) ∧
  t.c^2 = t.a^2 + t.b^2 - 2*t.a*t.b*(Real.cos t.C)

theorem triangle_problem (t : Triangle) 
  (h1 : t.a = 7)
  (h2 : t.c = 3)
  (h3 : Real.sin t.C / Real.sin t.B = 3/5)
  (h4 : lawOfSines t)
  (h5 : lawOfCosines t) :
  t.b = 5 ∧ Real.cos t.A = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l3772_377224


namespace NUMINAMATH_CALUDE_z_range_is_closed_interval_l3772_377226

-- Define the ellipse equation
def ellipse_equation (x y : ℝ) : Prop := x^2/16 + y^2/9 = 1

-- Define z as a function of x and y
def z (x y : ℝ) : ℝ := x + y

-- Theorem statement
theorem z_range_is_closed_interval :
  ∃ (a b : ℝ), a = -5 ∧ b = 5 ∧
  (∀ (x y : ℝ), ellipse_equation x y → a ≤ z x y ∧ z x y ≤ b) ∧
  (∀ t : ℝ, a ≤ t ∧ t ≤ b → ∃ (x y : ℝ), ellipse_equation x y ∧ z x y = t) :=
sorry

end NUMINAMATH_CALUDE_z_range_is_closed_interval_l3772_377226


namespace NUMINAMATH_CALUDE_E_is_top_leftmost_l3772_377233

-- Define the structure for a rectangle
structure Rectangle where
  w : Int
  x : Int
  y : Int
  z : Int

-- Define the five rectangles
def A : Rectangle := { w := 4, x := 1, y := 6, z := 9 }
def B : Rectangle := { w := 1, x := 0, y := 3, z := 6 }
def C : Rectangle := { w := 3, x := 8, y := 5, z := 2 }
def D : Rectangle := { w := 7, x := 5, y := 4, z := 8 }
def E : Rectangle := { w := 9, x := 2, y := 7, z := 0 }

-- Define the placement rules
def isLeftmost (r : Rectangle) : Bool :=
  r.w = 1 ∨ r.w = 9

def isRightmost (r : Rectangle) : Bool :=
  r.y = 6 ∨ r.y = 5

def isCenter (r : Rectangle) : Bool :=
  ¬(isLeftmost r) ∧ ¬(isRightmost r)

-- Theorem to prove
theorem E_is_top_leftmost :
  isLeftmost E ∧ 
  isRightmost A ∧ 
  isRightmost C ∧ 
  isLeftmost B ∧ 
  isCenter D :=
sorry

end NUMINAMATH_CALUDE_E_is_top_leftmost_l3772_377233


namespace NUMINAMATH_CALUDE_share_calculation_l3772_377235

/-- The amount y gets for each rupee x gets -/
def a : ℝ := 0.45

/-- The share of y in rupees -/
def y : ℝ := 63

/-- The total amount in rupees -/
def total : ℝ := 273

theorem share_calculation (x : ℝ) :
  x > 0 →
  x + a * x + 0.5 * x = total ∧
  a * x = y →
  a = 0.45 := by
  sorry

end NUMINAMATH_CALUDE_share_calculation_l3772_377235


namespace NUMINAMATH_CALUDE_lesser_number_l3772_377250

theorem lesser_number (x y : ℝ) (sum : x + y = 60) (diff : x - y = 8) : 
  min x y = 26 := by
sorry

end NUMINAMATH_CALUDE_lesser_number_l3772_377250


namespace NUMINAMATH_CALUDE_packs_per_box_is_40_l3772_377225

/-- Represents Meadow's diaper business --/
structure DiaperBusiness where
  boxes_per_week : ℕ
  diapers_per_pack : ℕ
  price_per_diaper : ℕ
  total_revenue : ℕ

/-- Calculates the number of packs in each box --/
def packs_per_box (business : DiaperBusiness) : ℕ :=
  (business.total_revenue / business.price_per_diaper) / 
  (business.diapers_per_pack * business.boxes_per_week)

/-- Theorem stating that the number of packs in each box is 40 --/
theorem packs_per_box_is_40 (business : DiaperBusiness) 
  (h1 : business.boxes_per_week = 30)
  (h2 : business.diapers_per_pack = 160)
  (h3 : business.price_per_diaper = 5)
  (h4 : business.total_revenue = 960000) :
  packs_per_box business = 40 := by
  sorry

end NUMINAMATH_CALUDE_packs_per_box_is_40_l3772_377225


namespace NUMINAMATH_CALUDE_min_value_x_plus_four_over_x_l3772_377215

theorem min_value_x_plus_four_over_x :
  ∃ (min : ℝ), min > 0 ∧
  (∀ x : ℝ, x > 0 → x + 4 / x ≥ min) ∧
  (∃ x : ℝ, x > 0 ∧ x + 4 / x = min) :=
by sorry

end NUMINAMATH_CALUDE_min_value_x_plus_four_over_x_l3772_377215


namespace NUMINAMATH_CALUDE_workday_meeting_percentage_l3772_377200

def workday_hours : ℕ := 8
def minutes_per_hour : ℕ := 60
def first_meeting_duration : ℕ := 40
def third_meeting_duration : ℕ := 30
def overlap_duration : ℕ := 10

def total_workday_minutes : ℕ := workday_hours * minutes_per_hour

def second_meeting_duration : ℕ := 2 * first_meeting_duration

def effective_second_meeting_duration : ℕ := second_meeting_duration - overlap_duration

def total_meeting_time : ℕ := first_meeting_duration + effective_second_meeting_duration + third_meeting_duration

def meeting_percentage : ℚ := (total_meeting_time : ℚ) / (total_workday_minutes : ℚ) * 100

theorem workday_meeting_percentage :
  ∃ (x : ℚ), abs (meeting_percentage - x) < 1 ∧ ⌊x⌋ = 29 := by sorry

end NUMINAMATH_CALUDE_workday_meeting_percentage_l3772_377200


namespace NUMINAMATH_CALUDE_triangle_inequality_l3772_377289

theorem triangle_inequality (a b c : ℝ) (C : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hC : 0 < C ∧ C < π) :
  c ≥ (a + b) * Real.sin (C / 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l3772_377289


namespace NUMINAMATH_CALUDE_inequality_solution_l3772_377274

theorem inequality_solution (a : ℝ) :
  (a = 1/2 → ∀ x, (x - a) * (x + a - 1) > 0 ↔ x ≠ 1/2) ∧
  (a < 1/2 → ∀ x, (x - a) * (x + a - 1) > 0 ↔ x > a ∨ x < 1 - a) ∧
  (a > 1/2 → ∀ x, (x - a) * (x + a - 1) > 0 ↔ x > a ∨ x < 1 - a) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3772_377274


namespace NUMINAMATH_CALUDE_original_number_is_200_l3772_377259

theorem original_number_is_200 : 
  ∃ x : ℝ, (x - 25 = 0.75 * x + 25) ∧ x = 200 := by
  sorry

end NUMINAMATH_CALUDE_original_number_is_200_l3772_377259


namespace NUMINAMATH_CALUDE_arithmetic_geometric_properties_l3772_377236

-- Define the arithmetic-geometric sequence
def arithmetic_geometric (a b : ℝ) (u : ℕ → ℝ) : Prop :=
  ∀ n, u (n + 1) = a * u n + b

-- Define another sequence satisfying the same recurrence relation
def same_recurrence (a b : ℝ) (v : ℕ → ℝ) : Prop :=
  ∀ n, v (n + 1) = a * v n + b

-- Define the sequence w as the difference of u and v
def w (u v : ℕ → ℝ) : ℕ → ℝ :=
  λ n => u n - v n

-- State the theorem
theorem arithmetic_geometric_properties
  (a b : ℝ)
  (u v : ℕ → ℝ)
  (hu : arithmetic_geometric a b u)
  (hv : same_recurrence a b v)
  (ha : a ≠ 1) :
  (∀ n, w u v (n + 1) = a * w u v n) ∧
  (∃ c : ℝ, ∀ n, v n = c ∧ c = b / (1 - a)) ∧
  (∀ n, u n = a^n * (u 0 - b/(1-a)) + b/(1-a)) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_properties_l3772_377236


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3772_377261

theorem complex_equation_solution (z : ℂ) : 
  (1 + 2*I)*z = 4 + 3*I → z = 2 - I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3772_377261


namespace NUMINAMATH_CALUDE_quadratic_function_equality_l3772_377229

theorem quadratic_function_equality (a b c d : ℝ) : 
  (∀ x, (x^2 + a*x + b) = ((2*x + 1)^2 + a*(2*x + 1) + b)) → 
  (∀ x, 4*(x^2 + c*x + d) = ((2*x + 1)^2 + a*(2*x + 1) + b)) → 
  (∀ x, 2*x + a = 2*x + c) → 
  (5^2 + 5*a + b = 30) → 
  (a = 2 ∧ b = -5 ∧ c = 2 ∧ d = -1/2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_equality_l3772_377229


namespace NUMINAMATH_CALUDE_negation_of_all_integers_squared_geq_one_l3772_377208

theorem negation_of_all_integers_squared_geq_one :
  (¬ ∀ (x : ℤ), x^2 ≥ 1) ↔ (∃ (x : ℤ), x^2 < 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_all_integers_squared_geq_one_l3772_377208


namespace NUMINAMATH_CALUDE_cube_digit_sum_l3772_377255

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Predicate for nine-digit numbers -/
def is_nine_digit (n : ℕ) : Prop := sorry

theorem cube_digit_sum (N : ℕ) (h1 : is_nine_digit N) (h2 : sum_of_digits N = 3) :
  sum_of_digits (N^3) = 9 ∨ sum_of_digits (N^3) = 18 ∨ sum_of_digits (N^3) = 27 := by sorry

end NUMINAMATH_CALUDE_cube_digit_sum_l3772_377255


namespace NUMINAMATH_CALUDE_granola_net_profit_l3772_377252

/-- Calculates the net profit from selling granola bags --/
theorem granola_net_profit
  (cost_per_bag : ℝ)
  (total_bags : ℕ)
  (full_price : ℝ)
  (discounted_price : ℝ)
  (bags_sold_full : ℕ)
  (bags_sold_discounted : ℕ)
  (h1 : cost_per_bag = 3)
  (h2 : total_bags = 20)
  (h3 : full_price = 6)
  (h4 : discounted_price = 4)
  (h5 : bags_sold_full = 15)
  (h6 : bags_sold_discounted = 5)
  (h7 : bags_sold_full + bags_sold_discounted = total_bags) :
  (full_price * bags_sold_full + discounted_price * bags_sold_discounted) - (cost_per_bag * total_bags) = 50 := by
  sorry

#check granola_net_profit

end NUMINAMATH_CALUDE_granola_net_profit_l3772_377252


namespace NUMINAMATH_CALUDE_circle_center_l3772_377280

/-- Given a circle with diameter endpoints (3, -3) and (13, 17), its center is (8, 7) -/
theorem circle_center (Q : Set (ℝ × ℝ)) (p₁ p₂ : ℝ × ℝ) (h₁ : p₁ = (3, -3)) (h₂ : p₂ = (13, 17)) 
    (h₃ : ∀ x ∈ Q, ∃ y ∈ Q, (x.1 - y.1)^2 + (x.2 - y.2)^2 = (p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2) :
  ∃ c : ℝ × ℝ, c = (8, 7) ∧ ∀ x ∈ Q, (x.1 - c.1)^2 + (x.2 - c.2)^2 = ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2) / 4 :=
by
  sorry

end NUMINAMATH_CALUDE_circle_center_l3772_377280


namespace NUMINAMATH_CALUDE_project_hours_l3772_377222

theorem project_hours (total_hours : ℕ) (kate_hours : ℕ) : 
  total_hours = 135 → 
  2 * kate_hours + kate_hours + 6 * kate_hours = total_hours →
  6 * kate_hours - kate_hours = 75 :=
by
  sorry

end NUMINAMATH_CALUDE_project_hours_l3772_377222


namespace NUMINAMATH_CALUDE_six_balls_three_boxes_l3772_377296

/-- Number of partitions of n into at most k parts -/
def num_partitions (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of ways to put 6 indistinguishable balls into 3 indistinguishable boxes -/
theorem six_balls_three_boxes : num_partitions 6 3 = 7 := by sorry

end NUMINAMATH_CALUDE_six_balls_three_boxes_l3772_377296


namespace NUMINAMATH_CALUDE_exists_cousin_180_problems_l3772_377211

/-- Represents the homework scenario for me and my cousin -/
structure HomeworkScenario where
  p : ℕ+  -- My rate (problems per hour)
  t : ℕ+  -- Time I take to finish homework (hours)
  n : ℕ   -- Number of problems I complete

/-- Calculates the number of problems my cousin does -/
def cousin_problems (s : HomeworkScenario) : ℕ :=
  ((3 * s.p.val - 5) * (s.t.val + 3)) / 2

/-- Theorem stating that there exists a scenario where my cousin does 180 problems -/
theorem exists_cousin_180_problems :
  ∃ (s : HomeworkScenario), 
    s.p ≥ 15 ∧ 
    s.n = s.p.val * s.t.val ∧ 
    cousin_problems s = 180 := by
  sorry

end NUMINAMATH_CALUDE_exists_cousin_180_problems_l3772_377211


namespace NUMINAMATH_CALUDE_minimum_score_for_raised_average_l3772_377253

def current_scores : List ℝ := [88, 92, 75, 85, 80]
def raise_average : ℝ := 5

theorem minimum_score_for_raised_average 
  (scores : List ℝ) 
  (raise : ℝ) 
  (h1 : scores = current_scores) 
  (h2 : raise = raise_average) : 
  (scores.sum + (scores.length + 1) * (scores.sum / scores.length + raise) - scores.sum) = 114 :=
sorry

end NUMINAMATH_CALUDE_minimum_score_for_raised_average_l3772_377253


namespace NUMINAMATH_CALUDE_fifteenth_student_age_l3772_377203

theorem fifteenth_student_age
  (total_students : Nat)
  (average_age : ℕ)
  (group1_students : Nat)
  (group1_average : ℕ)
  (group2_students : Nat)
  (group2_average : ℕ)
  (h1 : total_students = 15)
  (h2 : average_age = 15)
  (h3 : group1_students = 6)
  (h4 : group1_average = 14)
  (h5 : group2_students = 8)
  (h6 : group2_average = 16)
  (h7 : group1_students + group2_students + 1 = total_students) :
  total_students * average_age - (group1_students * group1_average + group2_students * group2_average) = 13 :=
by sorry

end NUMINAMATH_CALUDE_fifteenth_student_age_l3772_377203


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_closed_form_l3772_377293

/-- An arithmetic-geometric sequence -/
def ArithmeticGeometricSequence (a b : ℝ) (u₀ : ℝ) : ℕ → ℝ
  | 0 => u₀
  | n + 1 => a * ArithmeticGeometricSequence a b u₀ n + b

/-- Theorem for the closed form of an arithmetic-geometric sequence -/
theorem arithmetic_geometric_sequence_closed_form (a b u₀ : ℝ) (ha : a ≠ 1) :
  ∀ n : ℕ, ArithmeticGeometricSequence a b u₀ n = a^n * u₀ + b * (a^n - 1) / (a - 1) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_closed_form_l3772_377293


namespace NUMINAMATH_CALUDE_problem_statement_l3772_377240

theorem problem_statement (x y : ℝ) (hx : x = 7) (hy : y = -2) :
  (x - 2*y)^y = 1/121 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l3772_377240


namespace NUMINAMATH_CALUDE_isosceles_triangle_quadratic_roots_l3772_377238

theorem isosceles_triangle_quadratic_roots (a b k : ℝ) : 
  (∃ c : ℝ, c = 4 ∧ 
   (a = b ∧ (a + b = c ∨ a + c = b ∨ b + c = a)) ∧
   a^2 - 12*a + k + 2 = 0 ∧
   b^2 - 12*b + k + 2 = 0) →
  k = 34 ∨ k = 30 := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_quadratic_roots_l3772_377238


namespace NUMINAMATH_CALUDE_total_amount_l3772_377202

theorem total_amount (z : ℚ) (y : ℚ) (x : ℚ) 
  (hz : z = 200)
  (hy : y = 1.2 * z)
  (hx : x = 1.25 * y) :
  x + y + z = 740 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_l3772_377202


namespace NUMINAMATH_CALUDE_mary_seashells_count_l3772_377276

/-- The number of seashells found by Mary and Jessica together -/
def total_seashells : ℕ := 59

/-- The number of seashells found by Jessica -/
def jessica_seashells : ℕ := 41

/-- The number of seashells found by Mary -/
def mary_seashells : ℕ := total_seashells - jessica_seashells

theorem mary_seashells_count : mary_seashells = 18 := by
  sorry

end NUMINAMATH_CALUDE_mary_seashells_count_l3772_377276


namespace NUMINAMATH_CALUDE_coeff_x2y2_is_168_l3772_377218

/-- The coefficient of x^2y^2 in the expansion of ((1+x)^8(1+y)^4) -/
def coeff_x2y2 : ℕ :=
  (Nat.choose 8 2) * (Nat.choose 4 2)

/-- Theorem stating that the coefficient of x^2y^2 in ((1+x)^8(1+y)^4) is 168 -/
theorem coeff_x2y2_is_168 : coeff_x2y2 = 168 := by
  sorry

end NUMINAMATH_CALUDE_coeff_x2y2_is_168_l3772_377218


namespace NUMINAMATH_CALUDE_paper_clip_count_l3772_377231

theorem paper_clip_count (num_boxes : ℕ) (clips_per_box : ℕ) 
  (h1 : num_boxes = 9) (h2 : clips_per_box = 9) : 
  num_boxes * clips_per_box = 81 := by
  sorry

end NUMINAMATH_CALUDE_paper_clip_count_l3772_377231


namespace NUMINAMATH_CALUDE_smallest_covering_circle_l3772_377277

-- Define the plane region
def plane_region (x y : ℝ) : Prop :=
  x ≥ 0 ∧ y ≥ 0 ∧ x + 2*y - 4 ≤ 0

-- Define the circle equation
def circle_equation (x y a b r : ℝ) : Prop :=
  (x - a)^2 + (y - b)^2 = r^2

-- Theorem statement
theorem smallest_covering_circle :
  ∃ (a b r : ℝ), 
    (∀ x y, plane_region x y → circle_equation x y a b r) ∧
    (∀ a' b' r', (∀ x y, plane_region x y → circle_equation x y a' b' r') → r' ≥ r) ∧
    a = 2 ∧ b = 1 ∧ r^2 = 5 :=
sorry

end NUMINAMATH_CALUDE_smallest_covering_circle_l3772_377277


namespace NUMINAMATH_CALUDE_new_average_weight_l3772_377270

/-- Given 19 students with an average weight of 15 kg and a new student weighing 11 kg,
    the new average weight of all 20 students is 14.8 kg. -/
theorem new_average_weight (initial_students : ℕ) (initial_avg_weight : ℝ) 
  (new_student_weight : ℝ) : 
  initial_students = 19 → 
  initial_avg_weight = 15 → 
  new_student_weight = 11 → 
  (initial_students * initial_avg_weight + new_student_weight) / (initial_students + 1) = 14.8 := by
  sorry

end NUMINAMATH_CALUDE_new_average_weight_l3772_377270


namespace NUMINAMATH_CALUDE_closure_of_M_union_N_l3772_377278

-- Define the sets M and N
def M : Set ℝ := {x | (x + 3) * (x - 1) < 0}
def N : Set ℝ := {x | x ≤ -3}

-- State the theorem
theorem closure_of_M_union_N :
  closure (M ∪ N) = {x : ℝ | x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_closure_of_M_union_N_l3772_377278


namespace NUMINAMATH_CALUDE_triangle_side_lengths_l3772_377285

theorem triangle_side_lengths 
  (A B C : Real) 
  (a b c : Real) 
  (h1 : c = 10)
  (h2 : Real.cos A / Real.cos B = b / a)
  (h3 : b / a = 4 / 3) :
  a = 6 ∧ b = 8 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_lengths_l3772_377285


namespace NUMINAMATH_CALUDE_touching_circles_perimeter_l3772_377281

/-- Given a circle with center O and radius R, and two smaller circles with centers O₁ and O₂
    that touch each other and internally touch the larger circle,
    the perimeter of triangle OO₁O₂ is 2R. -/
theorem touching_circles_perimeter (O O₁ O₂ : ℝ × ℝ) (R : ℝ) :
  (∃ R₁ R₂ : ℝ, 
    R₁ > 0 ∧ R₂ > 0 ∧
    dist O O₁ = R - R₁ ∧
    dist O O₂ = R - R₂ ∧
    dist O₁ O₂ = R₁ + R₂) →
  dist O O₁ + dist O O₂ + dist O₁ O₂ = 2 * R :=
by sorry


end NUMINAMATH_CALUDE_touching_circles_perimeter_l3772_377281


namespace NUMINAMATH_CALUDE_imaginary_unit_power_l3772_377282

theorem imaginary_unit_power (i : ℂ) : i^2 = -1 → i^2016 = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_unit_power_l3772_377282


namespace NUMINAMATH_CALUDE_minimum_value_reciprocal_sum_l3772_377287

theorem minimum_value_reciprocal_sum (m n : ℝ) (hm : m > 0) (hn : n > 0) (h_sum : m + 2*n = 1) :
  (1/m + 1/n) ≥ 3 + 2*Real.sqrt 2 ∧ ∃ m n : ℝ, m > 0 ∧ n > 0 ∧ m + 2*n = 1 ∧ 1/m + 1/n = 3 + 2*Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_minimum_value_reciprocal_sum_l3772_377287


namespace NUMINAMATH_CALUDE_geometric_sequence_a11_l3772_377214

/-- A geometric sequence with a_3 = 3 and a_7 = 6 -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ 
  (∀ n : ℕ, a (n + 1) = a n * q) ∧
  a 3 = 3 ∧ 
  a 7 = 6

theorem geometric_sequence_a11 (a : ℕ → ℝ) (h : geometric_sequence a) : 
  a 11 = 12 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a11_l3772_377214


namespace NUMINAMATH_CALUDE_b_100_mod_81_l3772_377288

def b (n : ℕ) : ℕ := 7^n + 9^n

theorem b_100_mod_81 : b 100 ≡ 38 [ZMOD 81] := by sorry

end NUMINAMATH_CALUDE_b_100_mod_81_l3772_377288


namespace NUMINAMATH_CALUDE_constant_function_shift_l3772_377201

/-- Given a function f that is constant 2 for all real numbers, 
    prove that f(x + 2) = 2 for all real numbers x. -/
theorem constant_function_shift (f : ℝ → ℝ) (h : ∀ x : ℝ, f x = 2) :
  ∀ x : ℝ, f (x + 2) = 2 := by
sorry

end NUMINAMATH_CALUDE_constant_function_shift_l3772_377201


namespace NUMINAMATH_CALUDE_line_segment_proportions_l3772_377216

theorem line_segment_proportions (a b x : ℝ) : 
  (a / b = 3 / 2) → 
  (a + 2 * b = 28) → 
  (x^2 = a * b) →
  (a = 12 ∧ b = 8 ∧ x = 4 * Real.sqrt 6) := by
sorry

end NUMINAMATH_CALUDE_line_segment_proportions_l3772_377216


namespace NUMINAMATH_CALUDE_hyperbola_vertices_distance_l3772_377268

/-- The distance between vertices of a hyperbola -/
def distance_between_vertices (a b : ℝ) : ℝ := 2 * a

/-- The equation of the hyperbola -/
def is_hyperbola (x y a b : ℝ) : Prop :=
  x^2 / (a^2) - y^2 / (b^2) = 1

theorem hyperbola_vertices_distance :
  ∀ x y : ℝ, is_hyperbola x y 4 2 → distance_between_vertices 4 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_vertices_distance_l3772_377268


namespace NUMINAMATH_CALUDE_meeting_time_and_distance_l3772_377232

/-- Represents the time in hours since 7:45 AM -/
def time_since_start : ℝ → ℝ := λ t => t

/-- Samantha's speed in miles per hour -/
def samantha_speed : ℝ := 15

/-- Adam's speed in miles per hour -/
def adam_speed : ℝ := 20

/-- Time difference between Samantha's and Adam's start times in hours -/
def start_time_diff : ℝ := 0.5

/-- Total distance between Town A and Town B in miles -/
def total_distance : ℝ := 75

/-- Calculates Samantha's traveled distance at time t -/
def samantha_distance (t : ℝ) : ℝ := samantha_speed * t

/-- Calculates Adam's traveled distance at time t -/
def adam_distance (t : ℝ) : ℝ := adam_speed * (t - start_time_diff)

/-- Theorem stating the meeting time and Samantha's traveled distance -/
theorem meeting_time_and_distance :
  ∃ t : ℝ, 
    samantha_distance t + adam_distance t = total_distance ∧
    time_since_start t = 2.4333333333333 ∧ 
    samantha_distance t = 36 := by
  sorry

end NUMINAMATH_CALUDE_meeting_time_and_distance_l3772_377232


namespace NUMINAMATH_CALUDE_lucas_cleaning_days_l3772_377260

/-- Calculates the number of days Lucas took to clean windows -/
def days_to_clean (floors : ℕ) (windows_per_floor : ℕ) (payment_per_window : ℕ) 
                  (deduction_per_period : ℕ) (days_per_period : ℕ) (final_payment : ℕ) : ℕ :=
  let total_windows := floors * windows_per_floor
  let total_payment := total_windows * payment_per_window
  let deduction := total_payment - final_payment
  let periods := deduction / deduction_per_period
  periods * days_per_period

/-- Theorem stating that Lucas took 6 days to clean all windows -/
theorem lucas_cleaning_days : 
  days_to_clean 3 3 2 1 3 16 = 6 := by sorry

end NUMINAMATH_CALUDE_lucas_cleaning_days_l3772_377260


namespace NUMINAMATH_CALUDE_prism_with_27_edges_has_11_faces_l3772_377263

/-- A prism is a polyhedron with two congruent and parallel faces (bases) connected by lateral faces. -/
structure Prism where
  edges : ℕ
  lateral_faces : ℕ

/-- The number of edges in a prism is three times the number of lateral faces. -/
axiom prism_edge_count (p : Prism) : p.edges = 3 * p.lateral_faces

/-- The total number of faces in a prism is the number of lateral faces plus two (for the bases). -/
def total_faces (p : Prism) : ℕ := p.lateral_faces + 2

/-- Theorem: A prism with 27 edges has 11 faces. -/
theorem prism_with_27_edges_has_11_faces (p : Prism) (h : p.edges = 27) : total_faces p = 11 := by
  sorry


end NUMINAMATH_CALUDE_prism_with_27_edges_has_11_faces_l3772_377263


namespace NUMINAMATH_CALUDE_g_composition_of_3_l3772_377269

def g (x : ℤ) : ℤ :=
  if x % 2 = 0 then x / 2 else 5 * x + 3

theorem g_composition_of_3 : g (g (g (g 3))) = 24 := by
  sorry

end NUMINAMATH_CALUDE_g_composition_of_3_l3772_377269


namespace NUMINAMATH_CALUDE_ceiling_sqrt_200_l3772_377254

theorem ceiling_sqrt_200 : ⌈Real.sqrt 200⌉ = 15 := by sorry

end NUMINAMATH_CALUDE_ceiling_sqrt_200_l3772_377254


namespace NUMINAMATH_CALUDE_fraction_inequality_solution_l3772_377290

theorem fraction_inequality_solution (x : ℝ) : 
  (x ≠ 5) → (x / (x - 5) ≥ 0 ↔ x ∈ Set.Ici 5 ∪ Set.Iic 0) :=
by sorry

end NUMINAMATH_CALUDE_fraction_inequality_solution_l3772_377290


namespace NUMINAMATH_CALUDE_m_plus_n_values_l3772_377247

theorem m_plus_n_values (m n : ℤ) 
  (h1 : |m - n| = n - m) 
  (h2 : |m| = 4) 
  (h3 : |n| = 3) : 
  m + n = -1 ∨ m + n = -7 := by
sorry

end NUMINAMATH_CALUDE_m_plus_n_values_l3772_377247


namespace NUMINAMATH_CALUDE_hockey_arena_rows_l3772_377262

/-- The minimum number of rows required in a hockey arena -/
def min_rows (seats_per_row : ℕ) (total_students : ℕ) (max_students_per_school : ℕ) : ℕ :=
  let schools_per_row := seats_per_row / max_students_per_school
  let total_schools := (total_students + max_students_per_school - 1) / max_students_per_school
  (total_schools + schools_per_row - 1) / schools_per_row

/-- Theorem stating the minimum number of rows required for the given conditions -/
theorem hockey_arena_rows :
  min_rows 168 2016 45 = 16 := by
  sorry

#eval min_rows 168 2016 45

end NUMINAMATH_CALUDE_hockey_arena_rows_l3772_377262


namespace NUMINAMATH_CALUDE_expand_product_l3772_377244

theorem expand_product (x : ℝ) : (x + 3) * (x - 8) = x^2 - 5*x - 24 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l3772_377244


namespace NUMINAMATH_CALUDE_savings_percentage_l3772_377243

-- Define the original prices and discount rates
def coat_price : ℝ := 120
def hat_price : ℝ := 30
def gloves_price : ℝ := 50

def coat_discount : ℝ := 0.20
def hat_discount : ℝ := 0.40
def gloves_discount : ℝ := 0.30

-- Define the total original cost
def total_original_cost : ℝ := coat_price + hat_price + gloves_price

-- Define the savings for each item
def coat_savings : ℝ := coat_price * coat_discount
def hat_savings : ℝ := hat_price * hat_discount
def gloves_savings : ℝ := gloves_price * gloves_discount

-- Define the total savings
def total_savings : ℝ := coat_savings + hat_savings + gloves_savings

-- Theorem to prove
theorem savings_percentage :
  (total_savings / total_original_cost) * 100 = 25.5 := by
  sorry

end NUMINAMATH_CALUDE_savings_percentage_l3772_377243


namespace NUMINAMATH_CALUDE_equation_implies_conditions_l3772_377220

theorem equation_implies_conditions (a b c d : ℝ) 
  (h : (a^2 + b^2) / (b^2 + c^2) = (c^2 + d^2) / (d^2 + a^2)) :
  a = c ∨ a = -c ∨ a^2 - c^2 + d^2 = b^2 := by
sorry

end NUMINAMATH_CALUDE_equation_implies_conditions_l3772_377220


namespace NUMINAMATH_CALUDE_ladder_velocity_l3772_377239

theorem ladder_velocity (l a τ : ℝ) (hl : l > 0) (ha : a > 0) (hτ : τ > 0) :
  let α := Real.arcsin (a * τ^2 / (2 * l))
  let v₁ := a * τ
  let v₂ := (a^2 * τ^3) / Real.sqrt (4 * l^2 - a^2 * τ^4)
  v₁ * Real.sin α = v₂ * Real.cos α :=
by sorry

end NUMINAMATH_CALUDE_ladder_velocity_l3772_377239


namespace NUMINAMATH_CALUDE_ali_fish_weight_l3772_377291

theorem ali_fish_weight (peter_weight joey_weight ali_weight : ℝ) 
  (h1 : ali_weight = 2 * peter_weight)
  (h2 : joey_weight = peter_weight + 1)
  (h3 : peter_weight + joey_weight + ali_weight = 25) :
  ali_weight = 12 := by
sorry

end NUMINAMATH_CALUDE_ali_fish_weight_l3772_377291


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_specific_hyperbola_eccentricity_l3772_377292

/-- The eccentricity of a hyperbola with equation x²/a² - y²/b² = 1 is √(a² + b²) / a -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let e := Real.sqrt (a^2 + b^2) / a
  let hyperbola := fun (x y : ℝ) ↦ x^2 / a^2 - y^2 / b^2 = 1
  ∀ x y, hyperbola x y → e = Real.sqrt 5 / 2 :=
by
  sorry

/-- The eccentricity of the hyperbola x²/4 - y² = 1 is √5/2 -/
theorem specific_hyperbola_eccentricity :
  let e := Real.sqrt 5 / 2
  let hyperbola := fun (x y : ℝ) ↦ x^2 / 4 - y^2 = 1
  ∀ x y, hyperbola x y → e = Real.sqrt 5 / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_specific_hyperbola_eccentricity_l3772_377292


namespace NUMINAMATH_CALUDE_inequality_proof_l3772_377242

theorem inequality_proof (a b c : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (sum_one : a + b + c = 1) : 
  (a * b + b * c + c * a ≤ 1 / 3) ∧ 
  (a^2 / b + b^2 / c + c^2 / a ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3772_377242


namespace NUMINAMATH_CALUDE_cycle_original_price_l3772_377234

/-- Given a cycle sold at a 25% loss for Rs. 1350, prove its original price was Rs. 1800. -/
theorem cycle_original_price (selling_price : ℝ) (loss_percentage : ℝ) :
  selling_price = 1350 →
  loss_percentage = 25 →
  ∃ (original_price : ℝ), 
    original_price * (1 - loss_percentage / 100) = selling_price ∧
    original_price = 1800 :=
by sorry

end NUMINAMATH_CALUDE_cycle_original_price_l3772_377234


namespace NUMINAMATH_CALUDE_power_of_64_two_thirds_l3772_377272

theorem power_of_64_two_thirds : (64 : ℝ) ^ (2/3) = 16 := by
  sorry

end NUMINAMATH_CALUDE_power_of_64_two_thirds_l3772_377272


namespace NUMINAMATH_CALUDE_six_digit_numbers_with_zero_l3772_377264

theorem six_digit_numbers_with_zero (total_six_digit : Nat) (six_digit_no_zero : Nat) :
  total_six_digit = 900000 →
  six_digit_no_zero = 531441 →
  total_six_digit - six_digit_no_zero = 368559 := by
  sorry

end NUMINAMATH_CALUDE_six_digit_numbers_with_zero_l3772_377264


namespace NUMINAMATH_CALUDE_applicants_with_experience_and_degree_l3772_377267

theorem applicants_with_experience_and_degree 
  (total : ℕ) 
  (experienced : ℕ) 
  (degreed : ℕ) 
  (inexperienced_no_degree : ℕ) 
  (h1 : total = 30)
  (h2 : experienced = 10)
  (h3 : degreed = 18)
  (h4 : inexperienced_no_degree = 3) :
  total - (experienced + degreed - (total - inexperienced_no_degree)) = 1 := by
sorry

end NUMINAMATH_CALUDE_applicants_with_experience_and_degree_l3772_377267


namespace NUMINAMATH_CALUDE_arithmetic_not_geometric_l3772_377227

/-- An arithmetic sequence containing 1 and √2 -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ) (k l : ℕ), 
    (∀ n, a (n + 1) = a n + r) ∧ 
    a k = 1 ∧ 
    a l = Real.sqrt 2

/-- Three terms form a geometric sequence -/
def IsGeometric (x y z : ℝ) : Prop :=
  y * y = x * z

theorem arithmetic_not_geometric (a : ℕ → ℝ) (h : ArithmeticSequence a) :
  ¬ ∃ (m n p : ℕ), m ≠ n ∧ n ≠ p ∧ m ≠ p ∧ IsGeometric (a m) (a n) (a p) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_not_geometric_l3772_377227


namespace NUMINAMATH_CALUDE_cubic_factorization_l3772_377204

theorem cubic_factorization (x : ℝ) : x^3 + 3*x^2 - 4 = (x-1)*(x+2)^2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l3772_377204


namespace NUMINAMATH_CALUDE_decimal_equivalent_one_fourth_power_one_l3772_377258

theorem decimal_equivalent_one_fourth_power_one :
  (1 / 4 : ℚ) ^ (1 : ℕ) = 0.25 := by sorry

end NUMINAMATH_CALUDE_decimal_equivalent_one_fourth_power_one_l3772_377258


namespace NUMINAMATH_CALUDE_pen_cost_is_47_l3772_377209

/-- The cost of a pen in cents -/
def pen_cost : ℕ := 47

/-- The cost of a pencil in cents -/
def pencil_cost : ℕ := sorry

/-- Six pens and five pencils cost 380 cents -/
axiom condition1 : 6 * pen_cost + 5 * pencil_cost = 380

/-- Three pens and eight pencils cost 298 cents -/
axiom condition2 : 3 * pen_cost + 8 * pencil_cost = 298

/-- The cost of a pen is 47 cents -/
theorem pen_cost_is_47 : pen_cost = 47 := by sorry

end NUMINAMATH_CALUDE_pen_cost_is_47_l3772_377209


namespace NUMINAMATH_CALUDE_karlson_candies_theorem_l3772_377248

/-- The number of ones initially on the board -/
def initial_ones : ℕ := 29

/-- The number of minutes the process continues -/
def total_minutes : ℕ := 29

/-- Calculates the number of edges in a complete graph with n vertices -/
def complete_graph_edges (n : ℕ) : ℕ :=
  n * (n - 1) / 2

/-- The maximum number of candies Karlson could eat -/
def max_candies : ℕ := complete_graph_edges initial_ones

theorem karlson_candies_theorem :
  max_candies = 406 :=
sorry

end NUMINAMATH_CALUDE_karlson_candies_theorem_l3772_377248


namespace NUMINAMATH_CALUDE_inequality_proof_l3772_377275

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + 1) / (y + 1) + (y + 1) / (z + 1) + (z + 1) / (x + 1) ≤ x / y + y / z + z / x :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3772_377275


namespace NUMINAMATH_CALUDE_prob_adjacent_knights_l3772_377223

/-- The number of knights at the round table -/
def n : ℕ := 30

/-- The number of knights chosen for the quest -/
def k : ℕ := 4

/-- The probability of choosing k knights from n such that at least two are adjacent -/
def prob_adjacent (n k : ℕ) : ℚ :=
  1 - (n * (n - 3) * (n - 4) * (n - 5) : ℚ) / (n.choose k : ℚ)

/-- The theorem stating the probability of choosing 4 knights from 30 such that at least two are adjacent -/
theorem prob_adjacent_knights : prob_adjacent n k = 53 / 183 := by sorry

end NUMINAMATH_CALUDE_prob_adjacent_knights_l3772_377223


namespace NUMINAMATH_CALUDE_fraction_simplification_l3772_377213

theorem fraction_simplification (x y : ℚ) (hx : x = 4) (hy : y = 5) :
  (1 / (x + y)) / (1 / (x - y)) = -1/9 := by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3772_377213


namespace NUMINAMATH_CALUDE_complement_M_intersect_N_l3772_377212

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set M
def M : Set ℝ := {y | ∃ x, y = 2^x}

-- Define set N
def N : Set ℝ := {y | ∃ x, y = Real.log (3 - x)}

-- Theorem statement
theorem complement_M_intersect_N : 
  (U \ M) ∩ N = {y | y ≤ 0} := by sorry

end NUMINAMATH_CALUDE_complement_M_intersect_N_l3772_377212


namespace NUMINAMATH_CALUDE_product_of_sum_and_cube_sum_l3772_377295

theorem product_of_sum_and_cube_sum (a b : ℝ) 
  (h1 : a + b = 8) 
  (h2 : a^3 + b^3 = 172) : 
  a * b = 85/6 := by
sorry

end NUMINAMATH_CALUDE_product_of_sum_and_cube_sum_l3772_377295


namespace NUMINAMATH_CALUDE_transmitter_find_probability_l3772_377230

/-- Represents a license plate format for government vehicles in Kerrania -/
structure LicensePlate :=
  (first_two : Fin 100)
  (second : Fin 10)
  (last_two : Fin 100)
  (letters : Fin 3 × Fin 3)

/-- Conditions for a valid government license plate -/
def is_valid_plate (plate : LicensePlate) : Prop :=
  plate.first_two = 79 ∧
  (plate.second = 3 ∨ plate.second = 5) ∧
  (plate.last_two / 10 = plate.last_two % 10)

/-- Number of vehicles police can inspect in 3 hours -/
def inspected_vehicles : ℕ := 18

/-- Total number of possible valid license plates -/
def total_valid_plates : ℕ := 180

/-- Probability of finding the transmitter within 3 hours -/
def find_probability : ℚ := 1 / 10

theorem transmitter_find_probability :
  (inspected_vehicles : ℚ) / total_valid_plates = find_probability :=
sorry

end NUMINAMATH_CALUDE_transmitter_find_probability_l3772_377230


namespace NUMINAMATH_CALUDE_childrens_ticket_cost_l3772_377279

/-- Given information about ticket sales for a show, prove the cost of a children's ticket. -/
theorem childrens_ticket_cost
  (adult_ticket_cost : ℝ)
  (adult_count : ℕ)
  (total_receipts : ℝ)
  (h1 : adult_ticket_cost = 5.50)
  (h2 : adult_count = 152)
  (h3 : total_receipts = 1026)
  (h4 : adult_count = 2 * (adult_count / 2)) :
  ∃ (childrens_ticket_cost : ℝ),
    childrens_ticket_cost = 2.50 ∧
    total_receipts = adult_count * adult_ticket_cost + (adult_count / 2) * childrens_ticket_cost :=
by sorry

end NUMINAMATH_CALUDE_childrens_ticket_cost_l3772_377279


namespace NUMINAMATH_CALUDE_system_solution_l3772_377207

theorem system_solution (x y z : ℕ) : 
  x + y + z = 12 → 
  4 * x + 3 * y + 2 * z = 36 → 
  x ∈ ({0, 1, 2, 3, 4, 5, 6} : Set ℕ) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l3772_377207


namespace NUMINAMATH_CALUDE_problem_statement_l3772_377237

theorem problem_statement (a b x y : ℝ) 
  (eq1 : a * x + b * y = 3)
  (eq2 : a * x^2 + b * y^2 = 7)
  (eq3 : a * x^3 + b * y^3 = 16)
  (eq4 : a * x^4 + b * y^4 = 42) :
  a * x^5 + b * y^5 = 20 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3772_377237


namespace NUMINAMATH_CALUDE_triangle_exists_but_not_isosceles_l3772_377205

def stick_lengths : List ℝ := [1, 1.9, 1.9^2, 1.9^3, 1.9^4, 1.9^5, 1.9^6, 1.9^7, 1.9^8, 1.9^9]

def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def is_isosceles_triangle (a b c : ℝ) : Prop :=
  (a = b ∧ a + b > c) ∨ (b = c ∧ b + c > a) ∨ (c = a ∧ c + a > b)

theorem triangle_exists_but_not_isosceles :
  (∃ (a b c : ℝ), a ∈ stick_lengths ∧ b ∈ stick_lengths ∧ c ∈ stick_lengths ∧ is_triangle a b c) ∧
  (¬ ∃ (a b c : ℝ), a ∈ stick_lengths ∧ b ∈ stick_lengths ∧ c ∈ stick_lengths ∧ is_isosceles_triangle a b c) :=
by sorry

end NUMINAMATH_CALUDE_triangle_exists_but_not_isosceles_l3772_377205


namespace NUMINAMATH_CALUDE_infinitely_many_inequality_holds_l3772_377298

theorem infinitely_many_inequality_holds (a : ℕ → ℝ) (h : ∀ n, a n > 0) :
  ∃ S : Set ℕ, Set.Infinite S ∧ ∀ n ∈ S, 1 + a n > a (n - 1) * (2 : ℝ) ^ (1 / n) :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_inequality_holds_l3772_377298


namespace NUMINAMATH_CALUDE_angle_properties_l3772_377217

theorem angle_properties (α : Real) 
  (h1 : 0 < α) (h2 : α < π) (h3 : Real.sin α + Real.cos α = 1/5) : 
  (Real.tan α = -4/3) ∧ 
  (Real.sin α ^ 2 - 3 * Real.sin α * Real.cos α - 4 * Real.cos α ^ 2 = 16/25) := by
  sorry

end NUMINAMATH_CALUDE_angle_properties_l3772_377217


namespace NUMINAMATH_CALUDE_optimal_avocado_buying_strategy_l3772_377241

/-- Represents the optimal buying strategy for avocados -/
theorem optimal_avocado_buying_strategy 
  (recipe_need : ℕ) 
  (initial_count : ℕ) 
  (price_less_than_10 : ℝ) 
  (price_10_or_more : ℝ) 
  (h1 : recipe_need = 3) 
  (h2 : initial_count = 5) 
  (h3 : price_10_or_more < price_less_than_10) : 
  let additional_buy := 5
  let total_count := initial_count + additional_buy
  let total_cost := additional_buy * price_10_or_more
  (∀ n : ℕ, 
    let alt_total_count := initial_count + n
    let alt_total_cost := if alt_total_count < 10 then n * price_less_than_10 else n * price_10_or_more
    (alt_total_count ≥ recipe_need → total_cost ≤ alt_total_cost) ∧ 
    (alt_total_cost = total_cost → total_count ≥ alt_total_count)) :=
by sorry

end NUMINAMATH_CALUDE_optimal_avocado_buying_strategy_l3772_377241


namespace NUMINAMATH_CALUDE_f_2021_2_l3772_377245

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem f_2021_2 (f : ℝ → ℝ) 
  (h1 : is_even_function f)
  (h2 : ∀ x, f (x + 2) = -f x)
  (h3 : ∀ x ∈ Set.Ioo 1 2, f x = 2^x) :
  f (2021/2) = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_f_2021_2_l3772_377245


namespace NUMINAMATH_CALUDE_union_complement_equality_l3772_377251

def U : Set Nat := {0,1,2,3,4,5,6}
def A : Set Nat := {2,4,5}
def B : Set Nat := {0,1,3,5}

theorem union_complement_equality : A ∪ (U \ B) = {2,4,5,6} := by sorry

end NUMINAMATH_CALUDE_union_complement_equality_l3772_377251


namespace NUMINAMATH_CALUDE_tangent_through_origin_l3772_377283

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x

theorem tangent_through_origin (x₀ : ℝ) (h₁ : x₀ > 0) :
  (∃ k : ℝ, k * x₀ = f x₀ ∧ ∀ x : ℝ, f x₀ + k * (x - x₀) = k * x) →
  x₀ = Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_through_origin_l3772_377283
