import Mathlib

namespace NUMINAMATH_CALUDE_symmetric_point_example_l2657_265748

/-- Given a point P in a Cartesian coordinate system, this function returns its symmetric point with respect to the origin. -/
def symmetricPoint (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, -p.2)

/-- Theorem stating that the symmetric point of (2, -3) with respect to the origin is (-2, 3). -/
theorem symmetric_point_example : symmetricPoint (2, -3) = (-2, 3) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_example_l2657_265748


namespace NUMINAMATH_CALUDE_table_sum_zero_l2657_265709

structure Table :=
  (a b c d : ℝ)

def distinct (t : Table) : Prop :=
  t.a ≠ t.b ∧ t.a ≠ t.c ∧ t.a ≠ t.d ∧ t.b ≠ t.c ∧ t.b ≠ t.d ∧ t.c ≠ t.d

def row_sum_equal (t : Table) : Prop :=
  t.a + t.b = t.c + t.d

def column_product_equal (t : Table) : Prop :=
  t.a * t.c = t.b * t.d

theorem table_sum_zero (t : Table) 
  (h1 : distinct t) 
  (h2 : row_sum_equal t) 
  (h3 : column_product_equal t) : 
  t.a + t.b + t.c + t.d = 0 := by
  sorry

end NUMINAMATH_CALUDE_table_sum_zero_l2657_265709


namespace NUMINAMATH_CALUDE_solve_for_q_l2657_265774

theorem solve_for_q (p q : ℝ) (h1 : p > 1) (h2 : q > 1) (h3 : 1/p + 1/q = 1) (h4 : p*q = 9) :
  q = (9 + 3*Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_q_l2657_265774


namespace NUMINAMATH_CALUDE_fuel_cost_per_liter_l2657_265755

/-- Calculates the cost per liter of fuel given the conditions of the problem -/
theorem fuel_cost_per_liter 
  (service_cost : ℝ) 
  (minivan_count : ℕ) 
  (truck_count : ℕ) 
  (total_cost : ℝ) 
  (minivan_tank : ℝ) 
  (truck_tank_multiplier : ℝ) :
  service_cost = 2.20 →
  minivan_count = 3 →
  truck_count = 2 →
  total_cost = 347.7 →
  minivan_tank = 65 →
  truck_tank_multiplier = 2.2 →
  (total_cost - (service_cost * (minivan_count + truck_count))) / 
  (minivan_count * minivan_tank + truck_count * (minivan_tank * truck_tank_multiplier)) = 0.70 :=
by sorry

end NUMINAMATH_CALUDE_fuel_cost_per_liter_l2657_265755


namespace NUMINAMATH_CALUDE_rebus_no_solution_l2657_265761

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents a four-digit number -/
def FourDigitNumber := Fin 10000

/-- Represents a five-digit number -/
def FiveDigitNumber := Fin 100000

/-- Converts a four-digit number to its decimal representation -/
def toDecimal (n : FourDigitNumber) : ℕ := n.val

/-- Converts a five-digit number to its decimal representation -/
def toDecimalFive (n : FiveDigitNumber) : ℕ := n.val

/-- Constructs a four-digit number from individual digits -/
def makeNumber (k u s y : Digit) : FourDigitNumber :=
  ⟨k.val * 1000 + u.val * 100 + s.val * 10 + y.val, by sorry⟩

/-- Constructs a five-digit number from individual digits -/
def makeNumberFive (u k s y u' s' : Digit) : FiveDigitNumber :=
  ⟨u.val * 10000 + k.val * 1000 + s.val * 100 + y.val * 10 + u'.val, by sorry⟩

/-- The main theorem stating that the rebus has no solution -/
theorem rebus_no_solution :
  ¬∃ (k u s y : Digit),
    k ≠ u ∧ k ≠ s ∧ k ≠ y ∧ u ≠ s ∧ u ≠ y ∧ s ≠ y ∧
    toDecimal (makeNumber k u s y) + toDecimal (makeNumber u k s y) =
    toDecimalFive (makeNumberFive u k s y u s) :=
by sorry

end NUMINAMATH_CALUDE_rebus_no_solution_l2657_265761


namespace NUMINAMATH_CALUDE_prob_A_nth_day_l2657_265798

/-- The probability of switching restaurants each day -/
def switch_prob : ℝ := 0.6

/-- The probability of choosing restaurant A on the n-th day -/
def prob_A (n : ℕ) : ℝ := 0.5 + 0.5 * (-0.2)^(n - 1)

/-- Theorem stating the probability of choosing restaurant A on the n-th day -/
theorem prob_A_nth_day (n : ℕ) :
  prob_A n = 0.5 + 0.5 * (-0.2)^(n - 1) :=
by sorry

end NUMINAMATH_CALUDE_prob_A_nth_day_l2657_265798


namespace NUMINAMATH_CALUDE_new_pyramid_volume_l2657_265729

/-- Represents the volume change of a pyramid -/
def pyramid_volume_change (initial_volume : ℝ) (length_scale : ℝ) (width_scale : ℝ) (height_scale : ℝ) : ℝ :=
  initial_volume * length_scale * width_scale * height_scale

/-- Theorem: New volume of the pyramid after scaling -/
theorem new_pyramid_volume :
  let initial_volume : ℝ := 100
  let length_scale : ℝ := 3
  let width_scale : ℝ := 2
  let height_scale : ℝ := 1.2
  pyramid_volume_change initial_volume length_scale width_scale height_scale = 720 := by
  sorry


end NUMINAMATH_CALUDE_new_pyramid_volume_l2657_265729


namespace NUMINAMATH_CALUDE_equation_solution_l2657_265716

theorem equation_solution : 
  ∀ x : ℝ, x ≠ 1 ∧ x ≠ (1/2) → 
  (((x^2 - 5*x + 4) / (x - 1)) + ((2*x^2 + 7*x - 4) / (2*x - 1)) = 4) → 
  x = 2 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l2657_265716


namespace NUMINAMATH_CALUDE_circle_equation_l2657_265737

/-- Represents a parabola in the form y^2 = 4x -/
def Parabola := { p : ℝ × ℝ | p.2^2 = 4 * p.1 }

/-- The focus of the parabola y^2 = 4x -/
def focus : ℝ × ℝ := (1, 0)

/-- The directrix of the parabola y^2 = 4x -/
def directrix : ℝ → ℝ := fun x ↦ -1

/-- Represents a circle with center (h, k) and radius r -/
def Circle (h k r : ℝ) := { p : ℝ × ℝ | (p.1 - h)^2 + (p.2 - k)^2 = r^2 }

/-- The theorem stating that the circle with the focus as its center and tangent to the directrix
    has the equation (x - 1)^2 + y^2 = 4 -/
theorem circle_equation : 
  ∃ (c : Set (ℝ × ℝ)), c = Circle focus.1 focus.2 2 ∧ 
  (∀ p ∈ c, p.1 ≠ -1) ∧
  (∃ p ∈ c, p.1 = -1) ∧
  c = { p : ℝ × ℝ | (p.1 - 1)^2 + p.2^2 = 4 } :=
sorry

end NUMINAMATH_CALUDE_circle_equation_l2657_265737


namespace NUMINAMATH_CALUDE_volleyball_team_size_l2657_265746

/-- The number of people on each team in a volleyball game -/
def peoplePerTeam (managers : ℕ) (employees : ℕ) (teams : ℕ) : ℕ :=
  (managers + employees) / teams

/-- Theorem: In a volleyball game with 3 managers, 3 employees, and 3 teams, there are 2 people per team -/
theorem volleyball_team_size : peoplePerTeam 3 3 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_team_size_l2657_265746


namespace NUMINAMATH_CALUDE_alex_bike_trip_downhill_speed_alex_bike_trip_downhill_speed_is_24_l2657_265745

/-- Calculates the average speed on the downhill section of Alex's bike trip --/
theorem alex_bike_trip_downhill_speed : ℝ :=
  let total_distance : ℝ := 164
  let flat_time : ℝ := 4.5
  let flat_speed : ℝ := 20
  let uphill_time : ℝ := 2.5
  let uphill_speed : ℝ := 12
  let downhill_time : ℝ := 1.5
  let walking_distance : ℝ := 8
  let flat_distance : ℝ := flat_time * flat_speed
  let uphill_distance : ℝ := uphill_time * uphill_speed
  let distance_before_puncture : ℝ := total_distance - walking_distance
  let downhill_distance : ℝ := distance_before_puncture - flat_distance - uphill_distance
  let downhill_speed : ℝ := downhill_distance / downhill_time
  downhill_speed

theorem alex_bike_trip_downhill_speed_is_24 : alex_bike_trip_downhill_speed = 24 := by
  sorry

end NUMINAMATH_CALUDE_alex_bike_trip_downhill_speed_alex_bike_trip_downhill_speed_is_24_l2657_265745


namespace NUMINAMATH_CALUDE_area_of_trapezoid_DBCE_l2657_265724

/-- A structure representing a triangle in our problem -/
structure Triangle where
  area : ℝ

/-- A structure representing the trapezoid DBCE -/
structure Trapezoid where
  area : ℝ

/-- The isosceles triangle ABC -/
def ABC : Triangle := { area := 36 }

/-- One of the smallest triangles -/
def smallTriangle : Triangle := { area := 1 }

/-- The number of smallest triangles -/
def numSmallTriangles : ℕ := 5

/-- Triangle ADE, composed of 3 smallest triangles -/
def ADE : Triangle := { area := 3 }

/-- The trapezoid DBCE -/
def DBCE : Trapezoid := { area := ABC.area - ADE.area }

/-- The theorem to be proved -/
theorem area_of_trapezoid_DBCE : DBCE.area = 33 := by
  sorry

end NUMINAMATH_CALUDE_area_of_trapezoid_DBCE_l2657_265724


namespace NUMINAMATH_CALUDE_pyramid_base_side_length_l2657_265743

/-- The side length of the square base of a right pyramid, given the area of a lateral face and the slant height. -/
theorem pyramid_base_side_length (lateral_face_area : ℝ) (slant_height : ℝ) :
  lateral_face_area = 120 →
  slant_height = 40 →
  (lateral_face_area / slant_height) / 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_base_side_length_l2657_265743


namespace NUMINAMATH_CALUDE_minimum_value_sqrt_sum_l2657_265723

theorem minimum_value_sqrt_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + y = 1 → Real.sqrt x + Real.sqrt y ≤ Real.sqrt 2) ∧
  (∀ ε > 0, ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 1 ∧ Real.sqrt x + Real.sqrt y > Real.sqrt 2 - ε) :=
by sorry

end NUMINAMATH_CALUDE_minimum_value_sqrt_sum_l2657_265723


namespace NUMINAMATH_CALUDE_circle_symmetry_about_y_axis_l2657_265717

/-- Given two circles in the xy-plane, this theorem states that they are symmetric about the y-axis
    if and only if their equations are identical when x is replaced by -x in one of them. -/
theorem circle_symmetry_about_y_axis (a b : ℝ) :
  (∀ x y, x^2 + y^2 + a*x = 0 ↔ (-x)^2 + y^2 + b*(-x) = 0) ↔
  a = -b :=
sorry

end NUMINAMATH_CALUDE_circle_symmetry_about_y_axis_l2657_265717


namespace NUMINAMATH_CALUDE_remainder_three_to_ninth_mod_five_l2657_265756

theorem remainder_three_to_ninth_mod_five : 3^9 ≡ 3 [MOD 5] := by
  sorry

end NUMINAMATH_CALUDE_remainder_three_to_ninth_mod_five_l2657_265756


namespace NUMINAMATH_CALUDE_student_count_l2657_265751

theorem student_count : ∃ n : ℕ, n < 40 ∧ n % 7 = 3 ∧ n % 6 = 1 ∧ n = 31 := by
  sorry

end NUMINAMATH_CALUDE_student_count_l2657_265751


namespace NUMINAMATH_CALUDE_line_slope_product_l2657_265794

theorem line_slope_product (m n : ℝ) (h1 : m ≠ 0) (h2 : m = 4 * n) 
  (h3 : Real.arctan m = 2 * Real.arctan n) : m * n = 2 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_product_l2657_265794


namespace NUMINAMATH_CALUDE_third_circle_radius_l2657_265777

/-- Two externally tangent circles with a third circle tangent to both and their common external tangent -/
structure TangentCircles where
  /-- Center of the first circle -/
  A : ℝ × ℝ
  /-- Center of the second circle -/
  B : ℝ × ℝ
  /-- Radius of the first circle -/
  r1 : ℝ
  /-- Radius of the second circle -/
  r2 : ℝ
  /-- Radius of the third circle -/
  r3 : ℝ
  /-- The first two circles are externally tangent -/
  externally_tangent : dist A B = r1 + r2
  /-- The third circle is tangent to the first circle -/
  tangent_to_first : ∃ P : ℝ × ℝ, dist P A = r1 + r3 ∧ dist P B = r2 + r3
  /-- The third circle is tangent to the second circle -/
  tangent_to_second : ∃ Q : ℝ × ℝ, dist Q A = r1 + r3 ∧ dist Q B = r2 + r3
  /-- The third circle is tangent to the common external tangent of the first two circles -/
  tangent_to_external : ∃ T : ℝ × ℝ, dist T A = r1 ∧ dist T B = r2 ∧ 
    ∃ C : ℝ × ℝ, dist C A = r1 + r3 ∧ dist C B = r2 + r3 ∧ dist C T = r3

/-- The radius of the third circle is 1 -/
theorem third_circle_radius (tc : TangentCircles) (h1 : tc.r1 = 2) (h2 : tc.r2 = 5) : tc.r3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_third_circle_radius_l2657_265777


namespace NUMINAMATH_CALUDE_range_of_z_l2657_265795

theorem range_of_z (x y : ℝ) (h : x^2 + 2*x*y + 4*y^2 = 6) :
  4 ≤ x^2 + 4*y^2 ∧ x^2 + 4*y^2 ≤ 12 := by
  sorry

end NUMINAMATH_CALUDE_range_of_z_l2657_265795


namespace NUMINAMATH_CALUDE_circles_externally_tangent_l2657_265799

/-- Two circles are externally tangent if the distance between their centers
    is equal to the sum of their radii -/
def externally_tangent (c₁ c₂ : ℝ × ℝ) (r₁ r₂ : ℝ) : Prop :=
  Real.sqrt ((c₁.1 - c₂.1)^2 + (c₁.2 - c₂.2)^2) = r₁ + r₂

theorem circles_externally_tangent :
  let c₁ : ℝ × ℝ := (0, 8)
  let c₂ : ℝ × ℝ := (-6, 0)
  let r₁ : ℝ := 6
  let r₂ : ℝ := 2
  externally_tangent c₁ c₂ r₁ r₂ := by
  sorry

#check circles_externally_tangent

end NUMINAMATH_CALUDE_circles_externally_tangent_l2657_265799


namespace NUMINAMATH_CALUDE_four_divides_sum_of_squares_l2657_265701

theorem four_divides_sum_of_squares (a b c : ℕ+) :
  4 ∣ (a^2 + b^2 + c^2) ↔ (2 ∣ a ∧ 2 ∣ b ∧ 2 ∣ c) := by
  sorry

end NUMINAMATH_CALUDE_four_divides_sum_of_squares_l2657_265701


namespace NUMINAMATH_CALUDE_triangle_containing_all_points_l2657_265776

/-- A point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculate the area of a triangle given three points -/
def triangleArea (p1 p2 p3 : Point) : ℝ := sorry

/-- Check if a point is inside a triangle -/
def isInside (p : Point) (t1 t2 t3 : Point) : Prop := sorry

theorem triangle_containing_all_points 
  (n : ℕ) 
  (points : Fin n → Point) 
  (h : ∀ (i j k : Fin n), triangleArea (points i) (points j) (points k) ≤ 1) :
  ∃ (t1 t2 t3 : Point), 
    (triangleArea t1 t2 t3 ≤ 4) ∧ 
    (∀ (i : Fin n), isInside (points i) t1 t2 t3) := by
  sorry

end NUMINAMATH_CALUDE_triangle_containing_all_points_l2657_265776


namespace NUMINAMATH_CALUDE_brad_lemonade_profit_l2657_265735

/-- Calculates the net profit from a lemonade stand operation -/
def lemonade_stand_profit (
  glasses_per_gallon : ℕ)
  (cost_per_gallon : ℚ)
  (gallons_made : ℕ)
  (price_per_glass : ℚ)
  (glasses_drunk : ℕ)
  (glasses_unsold : ℕ) : ℚ :=
  let total_glasses := glasses_per_gallon * gallons_made
  let glasses_for_sale := total_glasses - glasses_drunk
  let glasses_sold := glasses_for_sale - glasses_unsold
  let revenue := glasses_sold * price_per_glass
  let cost := gallons_made * cost_per_gallon
  revenue - cost

/-- Theorem stating that Brad's net profit is $14.00 -/
theorem brad_lemonade_profit :
  lemonade_stand_profit 16 3.5 2 1 5 6 = 14 := by
  sorry

end NUMINAMATH_CALUDE_brad_lemonade_profit_l2657_265735


namespace NUMINAMATH_CALUDE_grasshopper_cannot_return_l2657_265732

def jump_sequence (n : ℕ) : ℕ := n

theorem grasshopper_cannot_return : 
  ∀ (x₀ y₀ x₂₂₂₂ y₂₂₂₂ : ℤ),
  (x₀ + y₀) % 2 = 0 →
  (∀ n : ℕ, n ≤ 2222 → ∃ xₙ yₙ : ℤ, 
    (xₙ - x₀)^2 + (yₙ - y₀)^2 = (jump_sequence n)^2) →
  (x₂₂₂₂ + y₂₂₂₂) % 2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_grasshopper_cannot_return_l2657_265732


namespace NUMINAMATH_CALUDE_continued_fraction_solution_l2657_265753

theorem continued_fraction_solution :
  ∃ y : ℝ, y = 3 + 6 / (2 + 6 / y) ∧ y = 2 + Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_continued_fraction_solution_l2657_265753


namespace NUMINAMATH_CALUDE_max_sum_of_squares_l2657_265793

theorem max_sum_of_squares (a b : ℝ) 
  (h : Real.sqrt ((a - 1)^2) + Real.sqrt ((a - 6)^2) = 10 - |b + 3| - |b - 2|) : 
  (∀ x y : ℝ, Real.sqrt ((x - 1)^2) + Real.sqrt ((x - 6)^2) = 10 - |y + 3| - |y - 2| → 
    x^2 + y^2 ≤ a^2 + b^2) → 
  a^2 + b^2 = 45 :=
sorry

end NUMINAMATH_CALUDE_max_sum_of_squares_l2657_265793


namespace NUMINAMATH_CALUDE_soda_price_after_increase_l2657_265720

theorem soda_price_after_increase (candy_price_new : ℝ) (candy_increase : ℝ) (soda_increase : ℝ) (combined_price_old : ℝ) 
  (h1 : candy_price_new = 15)
  (h2 : candy_increase = 0.25)
  (h3 : soda_increase = 0.5)
  (h4 : combined_price_old = 16) :
  ∃ (soda_price_new : ℝ), soda_price_new = 6 := by
  sorry

#check soda_price_after_increase

end NUMINAMATH_CALUDE_soda_price_after_increase_l2657_265720


namespace NUMINAMATH_CALUDE_sum_of_A_and_B_is_13_l2657_265734

theorem sum_of_A_and_B_is_13 (A B : ℕ) : 
  A ≠ B → 
  A < 10 → 
  B < 10 → 
  70 + A - (10 * B + 5) = 34 → 
  A + B = 13 := by
sorry

end NUMINAMATH_CALUDE_sum_of_A_and_B_is_13_l2657_265734


namespace NUMINAMATH_CALUDE_ellipse_m_value_l2657_265741

/-- An ellipse with equation x² + my² = 1, where m > 0 -/
structure Ellipse (m : ℝ) :=
  (eq : ∀ x y : ℝ, x^2 + m*y^2 = 1)
  (m_pos : m > 0)

/-- The focus of the ellipse is on the y-axis -/
def focus_on_y_axis (m : ℝ) : Prop := 0 < m ∧ m < 1

/-- The length of the major axis is twice that of the minor axis -/
def major_twice_minor (m : ℝ) : Prop := Real.sqrt (1/m) = 2

/-- Theorem: For an ellipse with equation x² + my² = 1 (m > 0), 
    if its focus is on the y-axis and the length of its major axis 
    is twice that of its minor axis, then m = 1/4 -/
theorem ellipse_m_value (m : ℝ) (e : Ellipse m) 
  (h1 : focus_on_y_axis m) (h2 : major_twice_minor m) : m = 1/4 :=
sorry

end NUMINAMATH_CALUDE_ellipse_m_value_l2657_265741


namespace NUMINAMATH_CALUDE_trig_identity_l2657_265728

theorem trig_identity (α β : Real) 
  (h1 : Real.cos (α + β) = 1/3)
  (h2 : Real.sin α * Real.sin β = 1/4) :
  (Real.cos α * Real.cos β = 7/12) ∧
  (Real.cos (2*α - 2*β) = 7/18) := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l2657_265728


namespace NUMINAMATH_CALUDE_exists_solution_l2657_265742

theorem exists_solution : ∃ (a b : ℤ), a ≠ b ∧ 
  (a : ℚ) / 2015 + (b : ℚ) / 2016 = (2015 + 2016 : ℚ) / (2015 * 2016) := by
  sorry

end NUMINAMATH_CALUDE_exists_solution_l2657_265742


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l2657_265783

theorem inequality_system_solution_set :
  let S := {x : ℝ | x - 3 < 0 ∧ x + 1 ≥ 0}
  S = {x : ℝ | -1 ≤ x ∧ x < 3} := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l2657_265783


namespace NUMINAMATH_CALUDE_inequality_representation_l2657_265715

theorem inequality_representation (x y : ℝ) : 
  abs x + abs y ≤ Real.sqrt (2 * (x^2 + y^2)) ∧ 
  Real.sqrt (2 * (x^2 + y^2)) ≤ 2 * max (abs x) (abs y) := by
  sorry

end NUMINAMATH_CALUDE_inequality_representation_l2657_265715


namespace NUMINAMATH_CALUDE_count_numbers_with_ten_digit_square_and_cube_l2657_265766

-- Define a function to count the number of digits in a natural number
def countDigits (n : ℕ) : ℕ :=
  if n < 10 then 1 else 1 + countDigits (n / 10)

-- Define the condition for a number to satisfy the problem requirement
def satisfiesCondition (n : ℕ) : Prop :=
  countDigits (n^2) + countDigits (n^3) = 10

-- Theorem statement
theorem count_numbers_with_ten_digit_square_and_cube :
  ∃ (S : Finset ℕ), (∀ n ∈ S, satisfiesCondition n) ∧ S.card = 53 :=
sorry

end NUMINAMATH_CALUDE_count_numbers_with_ten_digit_square_and_cube_l2657_265766


namespace NUMINAMATH_CALUDE_cosine_sum_17th_roots_l2657_265727

theorem cosine_sum_17th_roots : 
  Real.cos (2 * Real.pi / 17) + Real.cos (6 * Real.pi / 17) + Real.cos (8 * Real.pi / 17) = (Real.sqrt 13 - 1) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sum_17th_roots_l2657_265727


namespace NUMINAMATH_CALUDE_possible_values_of_a_l2657_265749

def A (a : ℝ) : Set ℝ := {2, 4, a^3 - 2*a^2 - a + 7}

def B (a : ℝ) : Set ℝ := {-4, a + 3, a^2 - 2*a + 2, a^3 + a^2 + 3*a + 7}

theorem possible_values_of_a :
  ∃ S : Set ℝ, S = {a : ℝ | A a ∩ B a = {2, 5}} ∧ S = {-1, 2} := by
  sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l2657_265749


namespace NUMINAMATH_CALUDE_percentage_both_correct_l2657_265790

theorem percentage_both_correct (p_first : ℝ) (p_second : ℝ) (p_neither : ℝ) : 
  p_first = 0.63 → p_second = 0.50 → p_neither = 0.20 → 
  p_first + p_second - (1 - p_neither) = 0.33 := by
  sorry

end NUMINAMATH_CALUDE_percentage_both_correct_l2657_265790


namespace NUMINAMATH_CALUDE_symmetric_lines_line_symmetry_l2657_265757

/-- Given two lines in a plane and a point, this theorem states that these lines are symmetric with respect to the given point. -/
theorem symmetric_lines (l₁ l₂ : ℝ → ℝ) (M : ℝ × ℝ) : Prop :=
  ∀ x y : ℝ, l₁ y = x → l₂ (4 - y) = 6 - x

/-- The main theorem proving that y = 3x - 17 is symmetric to y = 3x + 3 with respect to (3, 2) -/
theorem line_symmetry : 
  symmetric_lines (λ y => (y + 17) / 3) (λ y => (y - 3) / 3) (3, 2) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_lines_line_symmetry_l2657_265757


namespace NUMINAMATH_CALUDE_correct_product_l2657_265764

theorem correct_product (a b : ℕ+) 
  (h1 : (a - 6) * b = 255) 
  (h2 : (a + 10) * b = 335) : 
  a * b = 285 := by sorry

end NUMINAMATH_CALUDE_correct_product_l2657_265764


namespace NUMINAMATH_CALUDE_min_value_sum_l2657_265726

theorem min_value_sum (a b x y : ℝ) (ha : 0 < a) (hb : 0 < b) (hx : 0 < x) (hy : 0 < y)
  (h : a / x + b / y = 2) : 
  x + y ≥ (a + b) / 2 + Real.sqrt (a * b) :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_l2657_265726


namespace NUMINAMATH_CALUDE_trapezoid_perimeter_l2657_265789

/-- Represents a trapezoid EFGH with point X dividing the longer base EH -/
structure Trapezoid where
  EF : ℝ
  FG : ℝ
  GH : ℝ
  EX : ℝ
  XH : ℝ

/-- The perimeter of the trapezoid -/
def perimeter (t : Trapezoid) : ℝ :=
  t.EF + t.FG + t.GH + (t.EX + t.XH)

/-- Theorem stating that the perimeter of the given trapezoid is 165 units -/
theorem trapezoid_perimeter : 
  ∀ t : Trapezoid, 
    t.EF = 45 ∧ 
    t.FG = 40 ∧ 
    t.GH = 35 ∧ 
    t.EX = 30 ∧ 
    t.XH = 15 → 
    perimeter t = 165 := by
  sorry


end NUMINAMATH_CALUDE_trapezoid_perimeter_l2657_265789


namespace NUMINAMATH_CALUDE_smallest_integer_a_l2657_265705

theorem smallest_integer_a : ∃ (a : ℕ), (∀ (x y : ℝ), x > 0 → y > 0 → x + Real.sqrt (3 * x * y) ≤ a * (x + y)) ∧ 
  (∀ (b : ℕ), b < a → ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + Real.sqrt (3 * x * y) > b * (x + y)) ∧ 
  a = 2 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_a_l2657_265705


namespace NUMINAMATH_CALUDE_largest_five_digit_palindrome_divisible_by_127_l2657_265736

/-- A function that checks if a number is a 5-digit palindrome -/
def is_five_digit_palindrome (n : ℕ) : Prop :=
  10000 ≤ n ∧ n ≤ 99999 ∧ 
  (n / 10000 = n % 10) ∧
  ((n / 1000) % 10 = (n / 10) % 10)

/-- The largest 5-digit palindrome divisible by 127 -/
def largest_palindrome : ℕ := 99399

theorem largest_five_digit_palindrome_divisible_by_127 :
  is_five_digit_palindrome largest_palindrome ∧
  largest_palindrome % 127 = 0 ∧
  ∀ n : ℕ, is_five_digit_palindrome n ∧ n % 127 = 0 → n ≤ largest_palindrome :=
by sorry

end NUMINAMATH_CALUDE_largest_five_digit_palindrome_divisible_by_127_l2657_265736


namespace NUMINAMATH_CALUDE_choose_captains_l2657_265786

theorem choose_captains (n k : ℕ) (hn : n = 15) (hk : k = 4) :
  Nat.choose n k = 1365 := by
  sorry

end NUMINAMATH_CALUDE_choose_captains_l2657_265786


namespace NUMINAMATH_CALUDE_triangle_rotation_path_length_l2657_265704

/-- The length of the path traversed by a vertex of an equilateral triangle rotating inside a square -/
theorem triangle_rotation_path_length 
  (triangle_side : ℝ) 
  (square_side : ℝ) 
  (rotations_per_corner : ℕ) 
  (num_corners : ℕ) 
  (h1 : triangle_side = 3) 
  (h2 : square_side = 6) 
  (h3 : rotations_per_corner = 2) 
  (h4 : num_corners = 4) : 
  (rotations_per_corner * num_corners * triangle_side * (2 * Real.pi / 3)) = 16 * Real.pi :=
sorry

end NUMINAMATH_CALUDE_triangle_rotation_path_length_l2657_265704


namespace NUMINAMATH_CALUDE_unique_A_value_l2657_265714

-- Define the ♣ operation
def clubsuit (A B : ℝ) : ℝ := 3 * A + 2 * B^2 + 5

-- Theorem statement
theorem unique_A_value : ∃! A : ℝ, clubsuit A 3 = 73 ∧ A = 50/3 := by
  sorry

end NUMINAMATH_CALUDE_unique_A_value_l2657_265714


namespace NUMINAMATH_CALUDE_elevation_change_proof_l2657_265765

def initial_elevation : ℝ := 400

def stage1_rate : ℝ := 10
def stage1_time : ℝ := 5

def stage2_rate : ℝ := 15
def stage2_time : ℝ := 3

def stage3_rate : ℝ := 12
def stage3_time : ℝ := 6

def stage4_rate : ℝ := 8
def stage4_time : ℝ := 4

def stage5_rate : ℝ := 5
def stage5_time : ℝ := 2

def final_elevation : ℝ := initial_elevation - 
  (stage1_rate * stage1_time + 
   stage2_rate * stage2_time + 
   stage3_rate * stage3_time - 
   stage4_rate * stage4_time + 
   stage5_rate * stage5_time)

theorem elevation_change_proof : final_elevation = 255 := by sorry

end NUMINAMATH_CALUDE_elevation_change_proof_l2657_265765


namespace NUMINAMATH_CALUDE_weston_penalty_kicks_l2657_265797

/-- Calculates the number of penalty kicks required for a football drill -/
def penalty_kicks (total_players : ℕ) (goalies : ℕ) : ℕ :=
  goalies * (total_players - 1)

/-- Theorem: The number of penalty kicks for Weston Junior Football Club's drill is 92 -/
theorem weston_penalty_kicks :
  penalty_kicks 24 4 = 92 := by
  sorry

end NUMINAMATH_CALUDE_weston_penalty_kicks_l2657_265797


namespace NUMINAMATH_CALUDE_equalize_expenses_l2657_265710

/-- The amount LeRoy paid initially -/
def leroy_paid : ℝ := 240

/-- The amount Bernardo paid initially -/
def bernardo_paid : ℝ := 360

/-- The total discount received -/
def discount : ℝ := 60

/-- The amount LeRoy should pay Bernardo to equalize expenses -/
def payment_to_equalize : ℝ := 30

theorem equalize_expenses : 
  let total_cost := leroy_paid + bernardo_paid - discount
  let each_share := total_cost / 2
  payment_to_equalize = each_share - leroy_paid :=
by sorry

end NUMINAMATH_CALUDE_equalize_expenses_l2657_265710


namespace NUMINAMATH_CALUDE_fifth_month_sales_l2657_265779

def sales_1 : ℕ := 5435
def sales_2 : ℕ := 5927
def sales_3 : ℕ := 5855
def sales_4 : ℕ := 6230
def sales_6 : ℕ := 3991
def average_sale : ℕ := 5500
def num_months : ℕ := 6

theorem fifth_month_sales :
  ∃ (sales_5 : ℕ),
    (sales_1 + sales_2 + sales_3 + sales_4 + sales_5 + sales_6) / num_months = average_sale ∧
    sales_5 = 5562 :=
by sorry

end NUMINAMATH_CALUDE_fifth_month_sales_l2657_265779


namespace NUMINAMATH_CALUDE_wedge_volume_l2657_265719

/-- The volume of a wedge cut from a cylindrical log --/
theorem wedge_volume (d h : ℝ) (α : ℝ) : 
  d = 10 → α = 60 → (d / 2)^2 * h * π / 6 = 250 * π / 6 := by sorry

end NUMINAMATH_CALUDE_wedge_volume_l2657_265719


namespace NUMINAMATH_CALUDE_median_line_property_l2657_265763

-- Define the plane α
variable (α : Plane)

-- Define points A, B, and C
variable (A B C : Point)

-- Define the property of being non-collinear
def NonCollinear (A B C : Point) : Prop := sorry

-- Define the property of a point being outside a plane
def OutsidePlane (P : Point) (π : Plane) : Prop := sorry

-- Define the property of a point being equidistant from a plane
def EquidistantFromPlane (P : Point) (π : Plane) : Prop := sorry

-- Define a median line of a triangle
def MedianLine (M : Line) (A B C : Point) : Prop := sorry

-- Define the property of a line being parallel to a plane
def ParallelToPlane (L : Line) (π : Plane) : Prop := sorry

-- Define the property of a line lying within a plane
def LiesWithinPlane (L : Line) (π : Plane) : Prop := sorry

-- The theorem statement
theorem median_line_property 
  (h1 : NonCollinear A B C)
  (h2 : OutsidePlane A α ∧ OutsidePlane B α ∧ OutsidePlane C α)
  (h3 : EquidistantFromPlane A α ∧ EquidistantFromPlane B α ∧ EquidistantFromPlane C α) :
  ∃ (M : Line), MedianLine M A B C ∧ (ParallelToPlane M α ∨ LiesWithinPlane M α) :=
sorry

end NUMINAMATH_CALUDE_median_line_property_l2657_265763


namespace NUMINAMATH_CALUDE_integer_roots_of_cubic_polynomial_l2657_265754

theorem integer_roots_of_cubic_polynomial (a₂ a₁ : ℤ) :
  ∀ r : ℤ, r^3 + a₂ * r^2 + a₁ * r + 24 = 0 → r ∣ 24 := by
sorry

end NUMINAMATH_CALUDE_integer_roots_of_cubic_polynomial_l2657_265754


namespace NUMINAMATH_CALUDE_custard_pie_price_per_slice_l2657_265731

/-- The price per slice of custard pie given the number of pies, slices per pie, and total earnings -/
def price_per_slice (num_pies : ℕ) (slices_per_pie : ℕ) (total_earnings : ℚ) : ℚ :=
  total_earnings / (num_pies * slices_per_pie)

/-- Theorem stating that the price per slice of custard pie is $3 under given conditions -/
theorem custard_pie_price_per_slice :
  let num_pies : ℕ := 6
  let slices_per_pie : ℕ := 10
  let total_earnings : ℚ := 180
  price_per_slice num_pies slices_per_pie total_earnings = 3 := by
  sorry

end NUMINAMATH_CALUDE_custard_pie_price_per_slice_l2657_265731


namespace NUMINAMATH_CALUDE_triangle_properties_l2657_265784

/-- Given a triangle ABC with the following properties:
  - a = 2√2
  - sin C = √2 * sin A
  - cos C = √2/4
  Prove that:
  - c = 4
  - The area of the triangle is 2√7
-/
theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  a = 2 * Real.sqrt 2 →
  Real.sin C = Real.sqrt 2 * Real.sin A →
  Real.cos C = Real.sqrt 2 / 4 →
  c = 4 ∧
  (1/2) * a * b * Real.sin C = 2 * Real.sqrt 7 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l2657_265784


namespace NUMINAMATH_CALUDE_geometric_sequence_general_term_l2657_265796

/-- A geometric sequence with positive terms, where a₁ = 1 and a₁ + a₂ + a₃ = 7 -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧
  (∃ q : ℝ, q > 0 ∧ ∀ n, a (n + 1) = q * a n) ∧
  a 1 = 1 ∧
  a 1 + a 2 + a 3 = 7

/-- The general term of the geometric sequence is 2^(n-1) -/
theorem geometric_sequence_general_term (a : ℕ → ℝ) 
  (h : geometric_sequence a) : 
  ∀ n : ℕ, n ≥ 1 → a n = 2^(n - 1) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_general_term_l2657_265796


namespace NUMINAMATH_CALUDE_z_in_fourth_quadrant_iff_l2657_265744

/-- The complex number z defined in terms of a real number a -/
def z (a : ℝ) : ℂ := Complex.mk (|a| - 1) (a + 1)

/-- A point is in the fourth quadrant if its real part is positive and its imaginary part is negative -/
def in_fourth_quadrant (w : ℂ) : Prop := 0 < w.re ∧ w.im < 0

/-- Theorem stating the necessary and sufficient condition for z to be in the fourth quadrant -/
theorem z_in_fourth_quadrant_iff (a : ℝ) : in_fourth_quadrant (z a) ↔ a < -1 := by
  sorry

end NUMINAMATH_CALUDE_z_in_fourth_quadrant_iff_l2657_265744


namespace NUMINAMATH_CALUDE_triangle_acute_obtuse_characterization_l2657_265787

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  sum_angles : A + B + C = Real.pi

def Triangle.isAcute (t : Triangle) : Prop :=
  t.A < Real.pi / 2 ∧ t.B < Real.pi / 2 ∧ t.C < Real.pi / 2

def Triangle.isObtuse (t : Triangle) : Prop :=
  t.A > Real.pi / 2 ∨ t.B > Real.pi / 2 ∨ t.C > Real.pi / 2

theorem triangle_acute_obtuse_characterization (t : Triangle) :
  (t.isAcute ↔ Real.cos t.A ^ 2 + Real.cos t.B ^ 2 + Real.cos t.C ^ 2 < 1) ∧
  (t.isObtuse ↔ Real.cos t.A ^ 2 + Real.cos t.B ^ 2 + Real.cos t.C ^ 2 > 1) :=
sorry

end NUMINAMATH_CALUDE_triangle_acute_obtuse_characterization_l2657_265787


namespace NUMINAMATH_CALUDE_range_of_a_l2657_265711

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x, ∃ y, y = Real.log (a * x^2 - x + 1/4 * a)

def q (a : ℝ) : Prop := ∀ x > 0, 3^x - 9^x < a

-- Define the theorem
theorem range_of_a (a : ℝ) : 
  ((p a ∨ q a) ∧ ¬(p a ∧ q a)) → 0 ≤ a ∧ a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2657_265711


namespace NUMINAMATH_CALUDE_haley_deleted_files_l2657_265780

/-- The number of files deleted from a flash drive -/
def files_deleted (initial_music : ℕ) (initial_video : ℕ) (files_left : ℕ) : ℕ :=
  initial_music + initial_video - files_left

/-- Proof that 11 files were deleted from Haley's flash drive -/
theorem haley_deleted_files : files_deleted 27 42 58 = 11 := by
  sorry

end NUMINAMATH_CALUDE_haley_deleted_files_l2657_265780


namespace NUMINAMATH_CALUDE_two_digit_integers_with_remainder_three_l2657_265771

theorem two_digit_integers_with_remainder_three : 
  (Finset.filter 
    (fun n => n ≥ 10 ∧ n < 100 ∧ n % 7 = 3) 
    (Finset.range 100)).card = 13 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_integers_with_remainder_three_l2657_265771


namespace NUMINAMATH_CALUDE_polynomial_equality_l2657_265739

theorem polynomial_equality (a b c d e : ℝ) : 
  (∀ x : ℝ, (3*x + 1)^4 = a*x^4 + b*x^3 + c*x^2 + d*x + e) → 
  a - b + c - d + e = 16 := by
sorry

end NUMINAMATH_CALUDE_polynomial_equality_l2657_265739


namespace NUMINAMATH_CALUDE_min_moves_for_chess_like_coloring_l2657_265730

/-- Represents a cell in the 5x5 grid -/
inductive Cell
| white
| black

/-- Represents the 5x5 grid -/
def Grid := Fin 5 → Fin 5 → Cell

/-- Checks if two cells are neighbors -/
def are_neighbors (a b : Fin 5 × Fin 5) : Prop :=
  (a.1 = b.1 ∧ (a.2 = b.2 + 1 ∨ a.2 + 1 = b.2)) ∨
  (a.2 = b.2 ∧ (a.1 = b.1 + 1 ∨ a.1 + 1 = b.1))

/-- Represents a move (changing colors of two neighboring cells) -/
structure Move where
  cell1 : Fin 5 × Fin 5
  cell2 : Fin 5 × Fin 5
  are_neighbors : are_neighbors cell1 cell2

/-- Applies a move to a grid -/
def apply_move (g : Grid) (m : Move) : Grid :=
  sorry

/-- Checks if a grid has a chess-like coloring -/
def is_chess_like (g : Grid) : Prop :=
  sorry

/-- The main theorem to prove -/
theorem min_moves_for_chess_like_coloring :
  ∃ (moves : List Move),
    moves.length = 12 ∧
    (∀ g : Grid, (∀ i j, g i j = Cell.white) →
      is_chess_like (moves.foldl apply_move g)) ∧
    (∀ (moves' : List Move),
      moves'.length < 12 →
      ¬∃ g : Grid, (∀ i j, g i j = Cell.white) ∧
        is_chess_like (moves'.foldl apply_move g)) :=
  sorry

end NUMINAMATH_CALUDE_min_moves_for_chess_like_coloring_l2657_265730


namespace NUMINAMATH_CALUDE_best_fitting_model_has_highest_r_squared_model1_has_best_fitting_effect_l2657_265782

/-- Represents a regression model with its R² value -/
structure RegressionModel where
  name : String
  r_squared : ℝ
  r_squared_nonneg : 0 ≤ r_squared
  r_squared_le_one : r_squared ≤ 1

/-- Determines if a model has the best fitting effect among a list of models -/
def has_best_fitting_effect (model : RegressionModel) (models : List RegressionModel) : Prop :=
  ∀ m ∈ models, m.r_squared ≤ model.r_squared

/-- The theorem stating that the model with the highest R² value has the best fitting effect -/
theorem best_fitting_model_has_highest_r_squared 
  (models : List RegressionModel) (model : RegressionModel) 
  (h_model_in_list : model ∈ models) 
  (h_nonempty : models ≠ []) :
  has_best_fitting_effect model models ↔ 
  ∀ m ∈ models, m.r_squared ≤ model.r_squared :=
sorry

/-- The specific problem instance -/
def model1 : RegressionModel := ⟨"Model 1", 0.98, by norm_num, by norm_num⟩
def model2 : RegressionModel := ⟨"Model 2", 0.80, by norm_num, by norm_num⟩
def model3 : RegressionModel := ⟨"Model 3", 0.54, by norm_num, by norm_num⟩
def model4 : RegressionModel := ⟨"Model 4", 0.35, by norm_num, by norm_num⟩

def problem_models : List RegressionModel := [model1, model2, model3, model4]

theorem model1_has_best_fitting_effect : 
  has_best_fitting_effect model1 problem_models :=
sorry

end NUMINAMATH_CALUDE_best_fitting_model_has_highest_r_squared_model1_has_best_fitting_effect_l2657_265782


namespace NUMINAMATH_CALUDE_set_intersection_theorem_l2657_265747

def A : Set ℝ := {x | x - 1 ≥ 0}
def B : Set ℝ := {x | |x| ≤ 2}

theorem set_intersection_theorem : A ∩ B = {x : ℝ | 1 ≤ x ∧ x ≤ 2} := by
  sorry

end NUMINAMATH_CALUDE_set_intersection_theorem_l2657_265747


namespace NUMINAMATH_CALUDE_area_not_unique_l2657_265781

/-- A land plot with a side of length 10 units -/
structure LandPlot where
  side : ℝ
  side_positive : side > 0

/-- The area of a land plot -/
noncomputable def area (plot : LandPlot) : ℝ := sorry

/-- Theorem: The area of a land plot cannot be uniquely determined given only the length of one side -/
theorem area_not_unique (plot1 plot2 : LandPlot) 
  (h : plot1.side = plot2.side) (h_side : plot1.side = 10) : 
  ¬ (∀ (p1 p2 : LandPlot), p1.side = p2.side → area p1 = area p2) := by
  sorry

end NUMINAMATH_CALUDE_area_not_unique_l2657_265781


namespace NUMINAMATH_CALUDE_grid_has_ten_rows_l2657_265762

/-- Represents a grid of colored squares. -/
structure ColoredGrid where
  squares_per_row : ℕ
  red_squares : ℕ
  blue_squares : ℕ
  green_squares : ℕ

/-- Calculates the number of rows in the grid. -/
def number_of_rows (grid : ColoredGrid) : ℕ :=
  (grid.red_squares + grid.blue_squares + grid.green_squares) / grid.squares_per_row

/-- Theorem stating that a grid with the given properties has 10 rows. -/
theorem grid_has_ten_rows (grid : ColoredGrid) 
  (h1 : grid.squares_per_row = 15)
  (h2 : grid.red_squares = 24)
  (h3 : grid.blue_squares = 60)
  (h4 : grid.green_squares = 66) : 
  number_of_rows grid = 10 := by
  sorry

end NUMINAMATH_CALUDE_grid_has_ten_rows_l2657_265762


namespace NUMINAMATH_CALUDE_smallest_number_divisible_by_all_l2657_265767

def is_divisible_by_all (n : ℕ) : Prop :=
  (n + 5) % 19 = 0 ∧
  (n + 5) % 73 = 0 ∧
  (n + 5) % 101 = 0 ∧
  (n + 5) % 89 = 0

theorem smallest_number_divisible_by_all :
  ∃! n : ℕ, is_divisible_by_all n ∧ ∀ m : ℕ, m < n → ¬is_divisible_by_all m :=
by
  use 1113805958
  sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_by_all_l2657_265767


namespace NUMINAMATH_CALUDE_destination_distance_l2657_265708

/-- The distance to the destination in nautical miles -/
def distance : ℝ := sorry

/-- Theon's ship speed in nautical miles per hour -/
def theon_speed : ℝ := 15

/-- Yara's ship speed in nautical miles per hour -/
def yara_speed : ℝ := 30

/-- The time difference between Yara and Theon's arrivals in hours -/
def time_difference : ℝ := 3

theorem destination_distance : 
  distance = 90 ∧ 
  yara_speed = 2 * theon_speed ∧
  distance / yara_speed + time_difference = distance / theon_speed :=
by sorry

end NUMINAMATH_CALUDE_destination_distance_l2657_265708


namespace NUMINAMATH_CALUDE_ellipse_condition_l2657_265758

/-- Represents a curve defined by the equation ax^2 + by^2 = 1 -/
structure Curve where
  a : ℝ
  b : ℝ

/-- Predicate to check if a curve is an ellipse -/
def is_ellipse (c : Curve) : Prop :=
  c.a > 0 ∧ c.b > 0 ∧ c.a ≠ c.b

theorem ellipse_condition (c : Curve) :
  (is_ellipse c → c.a > 0 ∧ c.b > 0) ∧
  (∃ c : Curve, c.a > 0 ∧ c.b > 0 ∧ ¬is_ellipse c) :=
sorry

end NUMINAMATH_CALUDE_ellipse_condition_l2657_265758


namespace NUMINAMATH_CALUDE_elevator_probability_l2657_265722

/-- The number of floors in the building -/
def num_floors : ℕ := 6

/-- The number of floors where people can exit (excluding ground floor) -/
def exit_floors : ℕ := num_floors - 1

/-- The probability of two people leaving the elevator on different floors -/
def prob_different_floors : ℚ := 4/5

theorem elevator_probability :
  prob_different_floors = 1 - (1 : ℚ) / exit_floors :=
by sorry

end NUMINAMATH_CALUDE_elevator_probability_l2657_265722


namespace NUMINAMATH_CALUDE_cube_inequality_iff_l2657_265725

theorem cube_inequality_iff (a b : ℝ) : a < b ↔ a^3 < b^3 := by sorry

end NUMINAMATH_CALUDE_cube_inequality_iff_l2657_265725


namespace NUMINAMATH_CALUDE_muffin_mix_buyers_l2657_265759

theorem muffin_mix_buyers (total_buyers : ℕ) (cake_mix_buyers : ℕ) (both_mix_buyers : ℕ) 
  (neither_mix_prob : ℚ) (h1 : total_buyers = 100) (h2 : cake_mix_buyers = 50) 
  (h3 : both_mix_buyers = 15) (h4 : neither_mix_prob = 1/4) : 
  ∃ muffin_mix_buyers : ℕ, muffin_mix_buyers = 40 ∧ 
  muffin_mix_buyers = total_buyers - (cake_mix_buyers - both_mix_buyers) - 
    (neither_mix_prob * total_buyers) :=
by
  sorry

#check muffin_mix_buyers

end NUMINAMATH_CALUDE_muffin_mix_buyers_l2657_265759


namespace NUMINAMATH_CALUDE_two_digit_number_difference_l2657_265785

/-- Given a two-digit number where the difference between its digits is 9,
    prove that the difference between the original number and the number
    with interchanged digits is always 81. -/
theorem two_digit_number_difference (x y : ℕ) : 
  x ≥ 1 ∧ x ≤ 9 ∧ y ≥ 0 ∧ y ≤ 9 ∧ x - y = 9 →
  (10 * x + y) - (10 * y + x) = 81 := by
sorry

end NUMINAMATH_CALUDE_two_digit_number_difference_l2657_265785


namespace NUMINAMATH_CALUDE_count_satisfying_pairs_l2657_265713

def satisfies_inequalities (a b : ℤ) : Prop :=
  (a^2 + b^2 < 25) ∧ ((a - 3)^2 + b^2 < 20) ∧ (a^2 + (b - 3)^2 < 20)

theorem count_satisfying_pairs :
  ∃! (s : Finset (ℤ × ℤ)), 
    (∀ (p : ℤ × ℤ), p ∈ s ↔ satisfies_inequalities p.1 p.2) ∧
    s.card = 7 :=
sorry

end NUMINAMATH_CALUDE_count_satisfying_pairs_l2657_265713


namespace NUMINAMATH_CALUDE_wholesale_price_calculation_l2657_265791

/-- Proves that the wholesale price of a machine is $90 given the specified conditions -/
theorem wholesale_price_calculation (retail_price : ℝ) (discount_rate : ℝ) (profit_rate : ℝ) :
  retail_price = 120 →
  discount_rate = 0.1 →
  profit_rate = 0.2 →
  ∃ (wholesale_price : ℝ),
    wholesale_price = 90 ∧
    retail_price * (1 - discount_rate) = wholesale_price * (1 + profit_rate) :=
by sorry

end NUMINAMATH_CALUDE_wholesale_price_calculation_l2657_265791


namespace NUMINAMATH_CALUDE_function_root_implies_m_range_l2657_265702

theorem function_root_implies_m_range (m : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Icc (-2) 1 ∧ 2 * m * x + 4 = 0) → 
  (m ≤ -2 ∨ m ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_function_root_implies_m_range_l2657_265702


namespace NUMINAMATH_CALUDE_triangle_area_with_60_degree_angle_l2657_265750

/-- The area of a triangle with one angle of 60 degrees and adjacent sides of 15 cm and 12 cm is 45√3 cm² -/
theorem triangle_area_with_60_degree_angle (a b : ℝ) (h1 : a = 15) (h2 : b = 12) :
  (1/2) * a * b * Real.sqrt 3 = 45 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_with_60_degree_angle_l2657_265750


namespace NUMINAMATH_CALUDE_election_winner_percentage_l2657_265703

/-- Given an election with two candidates where:
  - The winner received 1054 votes
  - The winner won by 408 votes
Prove that the percentage of votes the winner received is
(1054 / (1054 + (1054 - 408))) * 100 -/
theorem election_winner_percentage (winner_votes : ℕ) (winning_margin : ℕ) :
  winner_votes = 1054 →
  winning_margin = 408 →
  (winner_votes : ℝ) / (winner_votes + (winner_votes - winning_margin)) * 100 =
    1054 / 1700 * 100 :=
by sorry

end NUMINAMATH_CALUDE_election_winner_percentage_l2657_265703


namespace NUMINAMATH_CALUDE_job_farm_reserved_land_l2657_265740

/-- Represents the land allocation of a farm in hectares -/
structure FarmLand where
  total : ℕ
  house_and_machinery : ℕ
  cattle : ℕ
  crops : ℕ

/-- Calculates the land reserved for future expansion -/
def reserved_land (farm : FarmLand) : ℕ :=
  farm.total - (farm.house_and_machinery + farm.cattle + farm.crops)

/-- Theorem stating that the reserved land for Job's farm is 15 hectares -/
theorem job_farm_reserved_land :
  let job_farm : FarmLand := {
    total := 150,
    house_and_machinery := 25,
    cattle := 40,
    crops := 70
  }
  reserved_land job_farm = 15 := by
  sorry


end NUMINAMATH_CALUDE_job_farm_reserved_land_l2657_265740


namespace NUMINAMATH_CALUDE_massager_vibration_increase_l2657_265712

/-- Given a massager with a lowest setting of 1600 vibrations per second
    and a highest setting that produces 768,000 vibrations in 5 minutes,
    prove that the percentage increase from lowest to highest setting is 60% -/
theorem massager_vibration_increase (lowest : ℝ) (highest_total : ℝ) (duration : ℝ) :
  lowest = 1600 →
  highest_total = 768000 →
  duration = 5 * 60 →
  (highest_total / duration - lowest) / lowest * 100 = 60 := by
sorry

end NUMINAMATH_CALUDE_massager_vibration_increase_l2657_265712


namespace NUMINAMATH_CALUDE_divisibility_arithmetic_progression_l2657_265760

theorem divisibility_arithmetic_progression (K : ℕ) :
  (∃ n : ℕ, K = 30 * n - 1) ↔ (K^K + 1) % 30 = 0 :=
by sorry

end NUMINAMATH_CALUDE_divisibility_arithmetic_progression_l2657_265760


namespace NUMINAMATH_CALUDE_gasoline_price_increase_l2657_265718

theorem gasoline_price_increase (original_price original_quantity : ℝ) 
  (h1 : original_price > 0) (h2 : original_quantity > 0) : 
  ∃ (price_increase : ℝ),
    (original_price * (1 + price_increase / 100) * (original_quantity * 0.95) = 
     original_price * original_quantity * 1.14) ∧ 
    (price_increase = 20) := by
  sorry

end NUMINAMATH_CALUDE_gasoline_price_increase_l2657_265718


namespace NUMINAMATH_CALUDE_problem1_l2657_265738

theorem problem1 (x y : ℝ) (h1 : x + y = 6) (h2 : x^2 + y^2 = 30) : x * y = 3 := by
  sorry

end NUMINAMATH_CALUDE_problem1_l2657_265738


namespace NUMINAMATH_CALUDE_christophers_age_l2657_265788

/-- Proves Christopher's age given the conditions of the problem -/
theorem christophers_age (christopher george ford : ℕ) 
  (h1 : george = christopher + 8)
  (h2 : ford = christopher - 2)
  (h3 : christopher + george + ford = 60) :
  christopher = 18 := by
  sorry

end NUMINAMATH_CALUDE_christophers_age_l2657_265788


namespace NUMINAMATH_CALUDE_expected_adjacent_red_pairs_l2657_265775

/-- The number of cards in a standard deck -/
def standardDeckSize : ℕ := 52

/-- The number of red cards in a standard deck -/
def redCardsCount : ℕ := 26

/-- The probability of a red card being followed by another red card -/
def probRedFollowedByRed : ℚ := 25 / 51

theorem expected_adjacent_red_pairs (deck_size : ℕ) (red_count : ℕ) (prob_red_red : ℚ) :
  deck_size = standardDeckSize →
  red_count = redCardsCount →
  prob_red_red = probRedFollowedByRed →
  (red_count : ℚ) * prob_red_red = 650 / 51 := by
  sorry

#check expected_adjacent_red_pairs

end NUMINAMATH_CALUDE_expected_adjacent_red_pairs_l2657_265775


namespace NUMINAMATH_CALUDE_harry_apples_l2657_265778

theorem harry_apples (martha_apples : ℕ) (tim_less : ℕ) (harry_ratio : ℕ) :
  martha_apples = 68 →
  tim_less = 30 →
  harry_ratio = 2 →
  (martha_apples - tim_less) / harry_ratio = 19 :=
by
  sorry

end NUMINAMATH_CALUDE_harry_apples_l2657_265778


namespace NUMINAMATH_CALUDE_hidden_dots_four_dice_l2657_265768

/-- The sum of dots on a standard six-sided die -/
def standard_die_sum : ℕ := 21

/-- The total number of dots on four standard six-sided dice -/
def total_dots (n : ℕ) : ℕ := n * standard_die_sum

/-- The sum of visible dots on the stacked dice -/
def visible_dots : ℕ := 1 + 2 + 2 + 3 + 4 + 5 + 6 + 6

/-- The number of hidden dots on four stacked dice -/
def hidden_dots (n : ℕ) : ℕ := total_dots n - visible_dots

theorem hidden_dots_four_dice : 
  hidden_dots 4 = 55 := by sorry

end NUMINAMATH_CALUDE_hidden_dots_four_dice_l2657_265768


namespace NUMINAMATH_CALUDE_intersection_sum_l2657_265772

/-- The quadratic function h(x) = -x^2 - 4x + 1 -/
def h (x : ℝ) : ℝ := -x^2 - 4*x + 1

/-- The function j(x) = -h(x) -/
def j (x : ℝ) : ℝ := -h x

/-- The function k(x) = h(-x) -/
def k (x : ℝ) : ℝ := h (-x)

/-- The number of intersection points between y = h(x) and y = j(x) -/
def c : ℕ := 2

/-- The number of intersection points between y = h(x) and y = k(x) -/
def d : ℕ := 1

/-- Theorem: Given the functions h, j, k, and the intersection counts c and d, 10c + d = 21 -/
theorem intersection_sum : 10 * c + d = 21 := by sorry

end NUMINAMATH_CALUDE_intersection_sum_l2657_265772


namespace NUMINAMATH_CALUDE_trig_identity_l2657_265733

theorem trig_identity (α : Real) (h : Real.tan α = 2) : 
  7 * (Real.sin α)^2 + 3 * (Real.cos α)^2 = 31/5 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l2657_265733


namespace NUMINAMATH_CALUDE_book_ratio_is_one_fifth_l2657_265792

/-- The ratio of Queen's extra books to Alannah's books -/
def book_ratio (beatrix alannah queen total : ℕ) : ℚ :=
  let queen_extra := queen - alannah
  ↑queen_extra / ↑alannah

theorem book_ratio_is_one_fifth 
  (beatrix alannah queen total : ℕ) 
  (h1 : beatrix = 30)
  (h2 : alannah = beatrix + 20)
  (h3 : total = beatrix + alannah + queen)
  (h4 : total = 140) :
  book_ratio beatrix alannah queen total = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_book_ratio_is_one_fifth_l2657_265792


namespace NUMINAMATH_CALUDE_first_number_in_expression_l2657_265706

theorem first_number_in_expression (x : ℝ) : x = 0.3 → x * 0.8 + 0.1 * 0.5 = 0.29 := by sorry

end NUMINAMATH_CALUDE_first_number_in_expression_l2657_265706


namespace NUMINAMATH_CALUDE_negation_of_absolute_value_non_negative_l2657_265773

theorem negation_of_absolute_value_non_negative :
  (¬ ∀ x : ℝ, |x| ≥ 0) ↔ (∃ x : ℝ, |x| < 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_absolute_value_non_negative_l2657_265773


namespace NUMINAMATH_CALUDE_no_real_roots_composite_l2657_265769

-- Define the quadratic function f
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem no_real_roots_composite (a b c : ℝ) :
  (∀ x : ℝ, f a b c x ≠ x) →
  (∀ x : ℝ, f a b c (f a b c x) ≠ x) :=
by sorry

end NUMINAMATH_CALUDE_no_real_roots_composite_l2657_265769


namespace NUMINAMATH_CALUDE_expansion_equals_fifth_power_l2657_265770

theorem expansion_equals_fifth_power (y : ℝ) : 
  (y - 1)^5 + 5*(y - 1)^4 + 10*(y - 1)^3 + 10*(y - 1)^2 + 5*(y - 1) + 1 = y^5 := by
  sorry

end NUMINAMATH_CALUDE_expansion_equals_fifth_power_l2657_265770


namespace NUMINAMATH_CALUDE_k_max_is_closest_to_expected_l2657_265721

/-- The probability of rolling a one on a fair die -/
def p : ℚ := 1 / 6

/-- The number of times the die is tossed -/
def n : ℕ := 20

/-- The expected number of ones when tossing a fair die n times -/
def expected_ones : ℚ := n * p

/-- The probability of rolling k ones in n tosses of a fair die -/
noncomputable def P (k : ℕ) : ℝ := sorry

/-- The value of k that maximizes P(k) -/
noncomputable def k_max : ℕ := sorry

/-- Theorem stating that k_max is the integer closest to the expected number of ones -/
theorem k_max_is_closest_to_expected : 
  k_max = round expected_ones := by sorry

end NUMINAMATH_CALUDE_k_max_is_closest_to_expected_l2657_265721


namespace NUMINAMATH_CALUDE_sara_quarters_l2657_265700

def initial_quarters : ℕ := 21
def dad_gave : ℕ := 49
def spent : ℕ := 15
def mom_gave_dollars : ℕ := 2
def quarters_per_dollar : ℕ := 4

theorem sara_quarters (x : ℕ) : 
  initial_quarters + dad_gave - spent + (mom_gave_dollars * quarters_per_dollar) + x = 63 + x := by
  sorry

end NUMINAMATH_CALUDE_sara_quarters_l2657_265700


namespace NUMINAMATH_CALUDE_unoccupied_chair_fraction_l2657_265752

theorem unoccupied_chair_fraction :
  let total_chairs : ℕ := 40
  let chair_capacity : ℕ := 2
  let total_members : ℕ := total_chairs * chair_capacity
  let attending_members : ℕ := 48
  let unoccupied_chairs : ℕ := (total_members - attending_members) / chair_capacity
  (unoccupied_chairs : ℚ) / total_chairs = 2 / 5 :=
by sorry

end NUMINAMATH_CALUDE_unoccupied_chair_fraction_l2657_265752


namespace NUMINAMATH_CALUDE_sacks_to_eliminate_l2657_265707

/-- The number of sacks containing at least $65536 -/
def sacks_with_target : ℕ := 6

/-- The total number of sacks -/
def total_sacks : ℕ := 30

/-- The desired probability of selecting a sack with at least $65536 -/
def target_probability : ℚ := 2/5

theorem sacks_to_eliminate :
  ∃ (n : ℕ), n ≥ 15 ∧
  (sacks_with_target : ℚ) / (total_sacks - n : ℚ) ≥ target_probability ∧
  ∀ (m : ℕ), m < n →
    (sacks_with_target : ℚ) / (total_sacks - m : ℚ) < target_probability :=
by sorry

end NUMINAMATH_CALUDE_sacks_to_eliminate_l2657_265707
