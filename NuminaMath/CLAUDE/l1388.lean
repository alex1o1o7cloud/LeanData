import Mathlib

namespace NUMINAMATH_CALUDE_max_profit_is_2180_l1388_138879

/-- Represents the production plan for items A and B -/
structure ProductionPlan where
  itemA : ℕ
  itemB : ℕ

/-- Calculates the profit for a given production plan -/
def profit (plan : ProductionPlan) : ℕ :=
  80 * plan.itemA + 100 * plan.itemB

/-- Checks if a production plan is feasible given the resource constraints -/
def isFeasible (plan : ProductionPlan) : Prop :=
  10 * plan.itemA + 70 * plan.itemB ≤ 700 ∧
  23 * plan.itemA + 40 * plan.itemB ≤ 642

/-- Theorem stating that the maximum profit is 2180 thousand rubles -/
theorem max_profit_is_2180 :
  ∃ (optimalPlan : ProductionPlan),
    isFeasible optimalPlan ∧
    profit optimalPlan = 2180 ∧
    ∀ (plan : ProductionPlan), isFeasible plan → profit plan ≤ 2180 := by
  sorry

end NUMINAMATH_CALUDE_max_profit_is_2180_l1388_138879


namespace NUMINAMATH_CALUDE_integer_pairs_satisfying_equation_l1388_138830

theorem integer_pairs_satisfying_equation : 
  {(x, y) : ℤ × ℤ | (y - 2) * x^2 + (y^2 - 6*y + 8) * x = y^2 - 5*y + 62} = 
  {(8, 3), (2, 9), (-7, 9), (-7, 3), (2, -6), (8, -6)} := by
  sorry

end NUMINAMATH_CALUDE_integer_pairs_satisfying_equation_l1388_138830


namespace NUMINAMATH_CALUDE_student_ticket_cost_l1388_138860

theorem student_ticket_cost (num_students : ℕ) (num_teachers : ℕ) (adult_ticket_cost : ℚ) (total_cost : ℚ) :
  num_students = 12 →
  num_teachers = 4 →
  adult_ticket_cost = 3 →
  total_cost = 24 →
  ∃ (student_ticket_cost : ℚ),
    student_ticket_cost * num_students + adult_ticket_cost * num_teachers = total_cost ∧
    student_ticket_cost = 1 :=
by sorry

end NUMINAMATH_CALUDE_student_ticket_cost_l1388_138860


namespace NUMINAMATH_CALUDE_function_max_min_on_interval_l1388_138837

def f (x : ℝ) : ℝ := x^2 + 2*x + 3

theorem function_max_min_on_interval (m : ℝ) :
  (∀ x ∈ Set.Icc m 0, f x ≤ 3) ∧
  (∃ x ∈ Set.Icc m 0, f x = 3) ∧
  (∀ x ∈ Set.Icc m 0, f x ≥ 2) ∧
  (∃ x ∈ Set.Icc m 0, f x = 2) ↔
  m ∈ Set.Icc (-2) (-1) :=
by sorry

end NUMINAMATH_CALUDE_function_max_min_on_interval_l1388_138837


namespace NUMINAMATH_CALUDE_second_term_of_geometric_series_l1388_138893

theorem second_term_of_geometric_series 
  (r : ℝ) (S : ℝ) (a : ℝ) (h1 : r = (1 : ℝ) / 4)
  (h2 : S = 48) (h3 : S = a / (1 - r)) :
  a * r = 9 := by
sorry

end NUMINAMATH_CALUDE_second_term_of_geometric_series_l1388_138893


namespace NUMINAMATH_CALUDE_complex_magnitude_equals_five_sqrt_five_l1388_138881

theorem complex_magnitude_equals_five_sqrt_five (t : ℝ) :
  t > 0 → (Complex.abs (-5 + t * Complex.I) = 5 * Real.sqrt 5 ↔ t = 10) := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_equals_five_sqrt_five_l1388_138881


namespace NUMINAMATH_CALUDE_chocolate_price_after_discount_l1388_138832

/-- The final price of a chocolate after discount -/
def final_price (original_cost discount : ℚ) : ℚ :=
  original_cost - discount

/-- Theorem: The final price of a chocolate with original cost $2 and discount $0.57 is $1.43 -/
theorem chocolate_price_after_discount :
  final_price 2 0.57 = 1.43 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_price_after_discount_l1388_138832


namespace NUMINAMATH_CALUDE_alexa_vacation_fraction_is_three_fourths_l1388_138818

/-- The number of days Alexa spent on vacation -/
def alexa_vacation_days : ℕ := 7 + 2

/-- The number of days it took Joey to learn swimming -/
def joey_swimming_days : ℕ := 6

/-- The number of days it took Ethan to learn fencing tricks -/
def ethan_fencing_days : ℕ := 2 * joey_swimming_days

/-- The fraction of time Alexa spent on vacation compared to Ethan's fencing learning time -/
def alexa_vacation_fraction : ℚ := alexa_vacation_days / ethan_fencing_days

theorem alexa_vacation_fraction_is_three_fourths :
  alexa_vacation_fraction = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_alexa_vacation_fraction_is_three_fourths_l1388_138818


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l1388_138859

/-- If 4x^2 - (a-b)x + 9 is a perfect square trinomial, then 2a-2b = ±24 -/
theorem perfect_square_trinomial (a b : ℝ) :
  (∃ c : ℝ, ∀ x : ℝ, 4*x^2 - (a-b)*x + 9 = (2*x - c)^2) →
  (2*a - 2*b = 24 ∨ 2*a - 2*b = -24) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l1388_138859


namespace NUMINAMATH_CALUDE_min_people_theorem_l1388_138866

/-- Represents a group of people consisting of married couples -/
structure CoupleGroup :=
  (num_couples : ℕ)
  (total_people : ℕ)
  (h_total : total_people = 2 * num_couples)

/-- The minimum number of people required to guarantee at least one married couple -/
def min_for_couple (group : CoupleGroup) : ℕ :=
  group.num_couples + 3

/-- The minimum number of people required to guarantee at least two people of the same gender -/
def min_for_same_gender (group : CoupleGroup) : ℕ := 3

/-- Theorem stating the minimum number of people required for both conditions -/
theorem min_people_theorem (group : CoupleGroup) 
  (h_group : group.num_couples = 10) : 
  min_for_couple group = 13 ∧ min_for_same_gender group = 3 := by
  sorry

#eval min_for_couple ⟨10, 20, rfl⟩
#eval min_for_same_gender ⟨10, 20, rfl⟩

end NUMINAMATH_CALUDE_min_people_theorem_l1388_138866


namespace NUMINAMATH_CALUDE_triangle_inequality_l1388_138865

theorem triangle_inequality (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a + b > c) (hbc : b + c > a) (hca : c + a > b) :
  Real.sqrt (3 * (Real.sqrt (a * b) + Real.sqrt (b * c) + Real.sqrt (c * a))) ≥ 
  Real.sqrt (a + b - c) + Real.sqrt (b + c - a) + Real.sqrt (c + a - b) := by
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1388_138865


namespace NUMINAMATH_CALUDE_inequality_proof_l1388_138885

theorem inequality_proof (x : ℝ) :
  x ≥ Real.rpow 7 (1/3) / Real.rpow 2 (1/3) ∧
  x < Real.rpow 373 (1/3) / Real.rpow 72 (1/3) →
  Real.sqrt (2*x + 7/x^2) + Real.sqrt (2*x - 7/x^2) < 6/x :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1388_138885


namespace NUMINAMATH_CALUDE_range_of_g_l1388_138884

-- Define the function f
def f (x : ℝ) : ℝ := 4 * x - 3

-- Define the function g as a composition of f five times
def g (x : ℝ) : ℝ := f (f (f (f (f x))))

-- State the theorem
theorem range_of_g :
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ 3 →
  ∃ y : ℝ, g x = y ∧ -1023 ≤ y ∧ y ≤ 2049 :=
sorry

end NUMINAMATH_CALUDE_range_of_g_l1388_138884


namespace NUMINAMATH_CALUDE_triangle_properties_l1388_138861

open Real

/-- Triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem stating properties of the triangle -/
theorem triangle_properties (t : Triangle) 
  (h1 : t.A ≠ π / 2)
  (h2 : 3 * sin t.A * cos t.B + (1/2) * t.b * sin (2 * t.A) = 3 * sin t.C) :
  t.a = 3 ∧ 
  (t.A = 2 * π / 3 → 
    ∃ (p : ℝ), p ≤ t.a + t.b + t.c ∧ p = 3 + 2 * Real.sqrt 3) := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l1388_138861


namespace NUMINAMATH_CALUDE_canteen_distance_l1388_138806

/-- Given a right triangle with one leg of length 400 rods and hypotenuse of length 700 rods,
    the point on the other leg that is equidistant from both endpoints of the hypotenuse
    is approximately 1711 rods from each endpoint. -/
theorem canteen_distance (a b c : ℝ) (h1 : a = 400) (h2 : c = 700) (h3 : a^2 + b^2 = c^2) :
  let x := (2 * a^2 + 2 * b^2) / (2 * b)
  ∃ ε > 0, abs (x - 1711) < ε :=
sorry

end NUMINAMATH_CALUDE_canteen_distance_l1388_138806


namespace NUMINAMATH_CALUDE_odot_four_three_l1388_138843

-- Define the binary operation ⊙
def odot (a b : ℝ) : ℝ := 5 * a + 2 * b

-- Theorem statement
theorem odot_four_three : odot 4 3 = 26 := by
  sorry

end NUMINAMATH_CALUDE_odot_four_three_l1388_138843


namespace NUMINAMATH_CALUDE_science_fair_participants_l1388_138851

theorem science_fair_participants (total : ℕ) (j s : ℕ) : 
  total = 240 →
  j + s = total →
  (3 * j) / 4 = s / 2 →
  (3 * j) / 4 + s / 2 = 144 :=
by sorry

end NUMINAMATH_CALUDE_science_fair_participants_l1388_138851


namespace NUMINAMATH_CALUDE_total_campers_rowing_l1388_138890

theorem total_campers_rowing (morning_campers afternoon_campers : ℕ) 
  (h1 : morning_campers = 35) 
  (h2 : afternoon_campers = 27) : 
  morning_campers + afternoon_campers = 62 := by
  sorry

end NUMINAMATH_CALUDE_total_campers_rowing_l1388_138890


namespace NUMINAMATH_CALUDE_patio_length_l1388_138831

theorem patio_length (w l : ℝ) (h1 : l = 4 * w) (h2 : 2 * l + 2 * w = 100) : l = 40 := by
  sorry

end NUMINAMATH_CALUDE_patio_length_l1388_138831


namespace NUMINAMATH_CALUDE_large_circle_diameter_l1388_138807

/-- The diameter of a circle that encompasses six smaller tangent circles -/
theorem large_circle_diameter (r : ℝ) (offset : ℝ) : 
  r = 4 ∧ 
  offset = 1 → 
  2 * (Real.sqrt 17 + 4) = 
    2 * (Real.sqrt ((r - offset)^2 + (2*r/2)^2) + r) :=
by sorry

end NUMINAMATH_CALUDE_large_circle_diameter_l1388_138807


namespace NUMINAMATH_CALUDE_unique_a_value_l1388_138826

/-- Converts a number from base 53 to base 10 -/
def base53ToBase10 (n : ℕ) : ℕ := sorry

/-- Theorem: If a is an integer between 0 and 20 (inclusive) and 4254253₅₃ - a is a multiple of 17, then a = 3 -/
theorem unique_a_value (a : ℤ) (h1 : 0 ≤ a) (h2 : a ≤ 20) 
  (h3 : (base53ToBase10 4254253 - a) % 17 = 0) : a = 3 := by sorry

end NUMINAMATH_CALUDE_unique_a_value_l1388_138826


namespace NUMINAMATH_CALUDE_janice_earnings_l1388_138836

/-- Janice's weekly earnings calculation --/
theorem janice_earnings 
  (days_per_week : ℕ) 
  (overtime_shifts : ℕ) 
  (overtime_pay : ℝ) 
  (total_earnings : ℝ) 
  (h1 : days_per_week = 5)
  (h2 : overtime_shifts = 3)
  (h3 : overtime_pay = 15)
  (h4 : total_earnings = 195) :
  ∃ (daily_earnings : ℝ), 
    daily_earnings * days_per_week + overtime_pay * overtime_shifts = total_earnings ∧ 
    daily_earnings = 30 := by
  sorry


end NUMINAMATH_CALUDE_janice_earnings_l1388_138836


namespace NUMINAMATH_CALUDE_linear_function_not_in_third_quadrant_l1388_138805

/-- A linear function y = kx - k where k ≠ 0 and k < 0 does not pass through the third quadrant -/
theorem linear_function_not_in_third_quadrant (k : ℝ) (h1 : k ≠ 0) (h2 : k < 0) :
  ∀ x y : ℝ, y = k * x - k → ¬(x < 0 ∧ y < 0) :=
by sorry

end NUMINAMATH_CALUDE_linear_function_not_in_third_quadrant_l1388_138805


namespace NUMINAMATH_CALUDE_plane_equation_l1388_138898

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Check if a point lies on a plane -/
def pointOnPlane (pt : Point3D) (pl : Plane) : Prop :=
  pl.a * pt.x + pl.b * pt.y + pl.c * pt.z + pl.d = 0

/-- Check if two planes are perpendicular -/
def planesArePerpendicular (pl1 pl2 : Plane) : Prop :=
  pl1.a * pl2.a + pl1.b * pl2.b + pl1.c * pl2.c = 0

/-- The main theorem -/
theorem plane_equation : ∃ (pl : Plane),
  pointOnPlane ⟨0, 2, 1⟩ pl ∧
  pointOnPlane ⟨2, 0, 1⟩ pl ∧
  planesArePerpendicular pl ⟨2, -1, 3, -4⟩ ∧
  pl.a > 0 ∧
  Int.gcd (Int.natAbs (Int.floor pl.a)) (Int.gcd (Int.natAbs (Int.floor pl.b)) (Int.gcd (Int.natAbs (Int.floor pl.c)) (Int.natAbs (Int.floor pl.d)))) = 1 ∧
  pl = ⟨1, 1, -1, -1⟩ :=
by
  sorry

end NUMINAMATH_CALUDE_plane_equation_l1388_138898


namespace NUMINAMATH_CALUDE_james_writing_time_l1388_138863

/-- James' writing scenario -/
structure WritingScenario where
  pages_per_hour : ℕ
  total_pages : ℕ
  total_weeks : ℕ

/-- Calculate the hours James writes per night -/
def hours_per_night (s : WritingScenario) : ℚ :=
  (s.total_pages : ℚ) / (s.total_weeks * 7 * s.pages_per_hour)

/-- Theorem stating that James writes for 3 hours every night -/
theorem james_writing_time (s : WritingScenario)
  (h1 : s.pages_per_hour = 5)
  (h2 : s.total_pages = 735)
  (h3 : s.total_weeks = 7) :
  hours_per_night s = 3 := by
  sorry

#eval hours_per_night ⟨5, 735, 7⟩

end NUMINAMATH_CALUDE_james_writing_time_l1388_138863


namespace NUMINAMATH_CALUDE_meeting_time_and_bridge_location_l1388_138873

/-- Represents a time of day in hours and minutes -/
structure TimeOfDay where
  hours : Nat
  minutes : Nat
  deriving Repr

/-- Represents a journey between two villages -/
structure Journey where
  startTime : TimeOfDay
  endTime : TimeOfDay
  deriving Repr

/-- Calculates the duration of a journey in minutes -/
def journeyDuration (j : Journey) : Nat :=
  (j.endTime.hours - j.startTime.hours) * 60 + j.endTime.minutes - j.startTime.minutes

/-- Theorem: Meeting time and bridge location -/
theorem meeting_time_and_bridge_location
  (womanJourney : Journey)
  (manJourney : Journey)
  (hWoman : womanJourney = ⟨⟨10, 31⟩, ⟨13, 43⟩⟩)
  (hMan : manJourney = ⟨⟨9, 13⟩, ⟨11, 53⟩⟩)
  (hSameRoad : True)  -- They travel on the same road
  (hConstantSpeed : True)  -- Both travel at constant speeds
  (hBridgeCrossing : True)  -- Woman crosses bridge 1 minute later than man
  : ∃ (meetingTime : TimeOfDay) (bridgeFromA bridgeFromB : Nat),
    meetingTime = ⟨11, 13⟩ ∧
    bridgeFromA = 7 ∧
    bridgeFromB = 24 :=
by sorry

end NUMINAMATH_CALUDE_meeting_time_and_bridge_location_l1388_138873


namespace NUMINAMATH_CALUDE_cubic_polynomial_roots_l1388_138854

theorem cubic_polynomial_roots (P : ℝ → ℝ) (x y z : ℝ) :
  P = (fun t ↦ t^3 - 2*t^2 - 10*t - 3) →
  (∃ a b c : ℝ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ P a = 0 ∧ P b = 0 ∧ P c = 0) →
  x + y + z = 2 →
  x*y + x*z + y*z = -10 →
  x*y*z = 3 →
  let u := x^2 * y^2 * z
  let v := x^2 * z^2 * y
  let w := y^2 * z^2 * x
  let R := fun t ↦ t^3 - (u + v + w)*t^2 + (u*v + u*w + v*w)*t - u*v*w
  R = fun t ↦ t^3 + 30*t^2 + 54*t - 243 := by sorry

end NUMINAMATH_CALUDE_cubic_polynomial_roots_l1388_138854


namespace NUMINAMATH_CALUDE_product_of_sums_powers_l1388_138844

theorem product_of_sums_powers : (5 + 7) * (5^2 + 7^2) * (5^4 + 7^4) * (5^8 + 7^8) * (5^16 + 7^16) = 63403380965376 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sums_powers_l1388_138844


namespace NUMINAMATH_CALUDE_square_nine_implies_fourth_power_eightyone_l1388_138853

theorem square_nine_implies_fourth_power_eightyone (a : ℝ) : a^2 = 9 → a^4 = 81 := by
  sorry

end NUMINAMATH_CALUDE_square_nine_implies_fourth_power_eightyone_l1388_138853


namespace NUMINAMATH_CALUDE_smallest_greater_than_1_1_l1388_138810

def S : Set ℚ := {1.4, 9/10, 1.2, 0.5, 13/10}

theorem smallest_greater_than_1_1 : 
  ∃ x ∈ S, x > 1.1 ∧ ∀ y ∈ S, y > 1.1 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_smallest_greater_than_1_1_l1388_138810


namespace NUMINAMATH_CALUDE_imaginary_part_of_i_l1388_138876

theorem imaginary_part_of_i : Complex.im i = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_i_l1388_138876


namespace NUMINAMATH_CALUDE_sin_cos_sum_one_l1388_138811

theorem sin_cos_sum_one : 
  Real.sin (15 * π / 180) * Real.cos (75 * π / 180) + 
  Real.cos (15 * π / 180) * Real.sin (105 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_one_l1388_138811


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_not_regular_polygon_l1388_138862

-- Define an isosceles right triangle
structure IsoscelesRightTriangle where
  side : ℝ
  side_positive : side > 0

-- Define a regular polygon
structure RegularPolygon where
  sides : ℕ
  side_length : ℝ
  side_positive : side_length > 0

-- Theorem: Isosceles right triangles are not regular polygons
theorem isosceles_right_triangle_not_regular_polygon :
  ∀ (t : IsoscelesRightTriangle), ¬∃ (p : RegularPolygon), true :=
sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_not_regular_polygon_l1388_138862


namespace NUMINAMATH_CALUDE_prism_coloring_iff_divisible_by_three_l1388_138821

/-- A prism with an n-gon base -/
structure Prism (n : ℕ) where
  base : Fin n → Fin 3  -- coloring of the base
  top : Fin n → Fin 3   -- coloring of the top

/-- Check if a coloring is valid for a prism -/
def is_valid_coloring (n : ℕ) (p : Prism n) : Prop :=
  ∀ (i : Fin n),
    -- Each vertex is connected to all three colors
    (∃ j, p.base j ≠ p.base i ∧ p.base j ≠ p.top i) ∧
    (∃ j, p.top j ≠ p.base i ∧ p.top j ≠ p.top i) ∧
    p.base i ≠ p.top i

theorem prism_coloring_iff_divisible_by_three (n : ℕ) :
  (∃ p : Prism n, is_valid_coloring n p) ↔ 3 ∣ n :=
sorry

end NUMINAMATH_CALUDE_prism_coloring_iff_divisible_by_three_l1388_138821


namespace NUMINAMATH_CALUDE_least_valid_number_l1388_138871

def is_valid_number (n : ℕ) : Prop :=
  n ≥ 100 ∧ (n % 100 = n / 10)

theorem least_valid_number : 
  (∃ (n : ℕ), is_valid_number n) ∧ 
  (∀ (m : ℕ), is_valid_number m → m ≥ 900) :=
sorry

end NUMINAMATH_CALUDE_least_valid_number_l1388_138871


namespace NUMINAMATH_CALUDE_equation_substitution_l1388_138809

theorem equation_substitution (x y : ℝ) :
  (y = 2 * x - 1) → (2 * x - 3 * y = 5) → (2 * x - 6 * x + 3 = 5) :=
by
  sorry

end NUMINAMATH_CALUDE_equation_substitution_l1388_138809


namespace NUMINAMATH_CALUDE_APMS_is_parallelogram_l1388_138883

-- Define the points
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the quadrilateral APMS
def APMS (P Q M S A : Point2D) : Prop :=
  M.x = (P.x + Q.x) / 2 ∧
  M.y = (P.y + Q.y) / 2 ∧
  S.x = M.x ∧
  S.y ≠ M.y

-- Define what it means for a quadrilateral to be a parallelogram
def IsParallelogram (A P M S : Point2D) : Prop :=
  (P.x - A.x = M.x - S.x ∧ P.y - A.y = M.y - S.y) ∧
  (M.x - A.x = S.x - P.x ∧ M.y - A.y = S.y - P.y)

-- Theorem statement
theorem APMS_is_parallelogram 
  (P Q M S A : Point2D) 
  (h_distinct : P ≠ Q) 
  (h_APMS : APMS P Q M S A) : 
  IsParallelogram A P M S :=
sorry

end NUMINAMATH_CALUDE_APMS_is_parallelogram_l1388_138883


namespace NUMINAMATH_CALUDE_harmonious_example_harmonious_rational_sum_harmonious_rational_ratio_l1388_138847

/-- A pair of real numbers (a, b) is harmonious if a^2 + b and a + b^2 are both rational. -/
def Harmonious (a b : ℝ) : Prop :=
  (∃ q₁ : ℚ, a^2 + b = q₁) ∧ (∃ q₂ : ℚ, a + b^2 = q₂)

theorem harmonious_example :
  Harmonious (Real.sqrt 2 + 1/2) (1/2 - Real.sqrt 2) := by sorry

theorem harmonious_rational_sum {a b : ℝ} (h : Harmonious a b) (hs : ∃ q : ℚ, a + b = q) (hne : a + b ≠ 1) :
  ∃ (q₁ q₂ : ℚ), a = q₁ ∧ b = q₂ := by sorry

theorem harmonious_rational_ratio {a b : ℝ} (h : Harmonious a b) (hr : ∃ q : ℚ, a = q * b) :
  ∃ (q₁ q₂ : ℚ), a = q₁ ∧ b = q₂ := by sorry

end NUMINAMATH_CALUDE_harmonious_example_harmonious_rational_sum_harmonious_rational_ratio_l1388_138847


namespace NUMINAMATH_CALUDE_quadratic_root_range_l1388_138803

theorem quadratic_root_range (a : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x > 1 ∧ y < 1 ∧ 
   x^2 + (a^2 - 1)*x + a - 2 = 0 ∧
   y^2 + (a^2 - 1)*y + a - 2 = 0) →
  a > -2 ∧ a < 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_range_l1388_138803


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l1388_138877

theorem simplify_and_evaluate : ∀ x : ℤ, 
  -1 < x → x < 3 → x ≠ 1 → x ≠ 2 →
  (3 / (x - 1) - x - 1) * ((x - 1) / (x^2 - 4*x + 4)) = (2 + x) / (2 - x) ∧
  (0 : ℤ) ∈ {y : ℤ | -1 < y ∧ y < 3 ∧ y ≠ 1 ∧ y ≠ 2} ∧
  (2 + 0) / (2 - 0) = 1 := by
sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l1388_138877


namespace NUMINAMATH_CALUDE_min_value_theorem_l1388_138872

theorem min_value_theorem (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a * (a + b + c) + b * c = 4 - 2 * Real.sqrt 3) :
  ∃ (x : ℝ), x = 2 * Real.sqrt 3 - 2 ∧ ∀ (y : ℝ), 2 * a + b + c ≥ y :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1388_138872


namespace NUMINAMATH_CALUDE_quadratic_factorization_l1388_138867

theorem quadratic_factorization (d e f : ℤ) :
  (∀ x, x^2 + 17*x + 72 = (x + d) * (x + e)) →
  (∀ x, x^2 + 7*x - 60 = (x + e) * (x - f)) →
  d + e + f = 29 := by
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l1388_138867


namespace NUMINAMATH_CALUDE_polynomial_factor_implies_coefficients_l1388_138802

/-- Given a polynomial px^4 + qx^3 + 40x^2 - 20x + 8 with a factor of 4x^2 - 3x + 2,
    prove that p = 0 and q = -32 -/
theorem polynomial_factor_implies_coefficients
  (p q : ℚ)
  (h : ∃ (r s : ℚ), px^4 + qx^3 + 40*x^2 - 20*x + 8 = (4*x^2 - 3*x + 2) * (r*x^2 + s*x + 4)) :
  p = 0 ∧ q = -32 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factor_implies_coefficients_l1388_138802


namespace NUMINAMATH_CALUDE_larger_number_is_72_l1388_138895

theorem larger_number_is_72 (x y : ℝ) : 
  5 * y = 6 * x → y - x = 12 → y = 72 := by sorry

end NUMINAMATH_CALUDE_larger_number_is_72_l1388_138895


namespace NUMINAMATH_CALUDE_root_equation_solution_l1388_138849

theorem root_equation_solution (a b c : ℕ) (ha : a > 1) (hb : b > 1) (hc : c > 1)
  (h : ∀ (N : ℝ), N ≠ 1 → (N^2 * (N^3 * N^(4/c))^(1/b))^(1/a) = N^(17/24)) :
  b = 4 := by sorry

end NUMINAMATH_CALUDE_root_equation_solution_l1388_138849


namespace NUMINAMATH_CALUDE_lcm_140_225_l1388_138800

theorem lcm_140_225 : Nat.lcm 140 225 = 6300 := by
  sorry

end NUMINAMATH_CALUDE_lcm_140_225_l1388_138800


namespace NUMINAMATH_CALUDE_cubic_system_product_l1388_138845

theorem cubic_system_product (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) 
  (h₁ : x₁^3 - 3*x₁*y₁^2 = 2010 ∧ y₁^3 - 3*x₁^2*y₁ = 2000)
  (h₂ : x₂^3 - 3*x₂*y₂^2 = 2010 ∧ y₂^3 - 3*x₂^2*y₂ = 2000)
  (h₃ : x₃^3 - 3*x₃*y₃^2 = 2010 ∧ y₃^3 - 3*x₃^2*y₃ = 2000) :
  (1 - x₁/y₁) * (1 - x₂/y₂) * (1 - x₃/y₃) = 1/100 := by
sorry

end NUMINAMATH_CALUDE_cubic_system_product_l1388_138845


namespace NUMINAMATH_CALUDE_is_systematic_sampling_l1388_138850

/-- Represents a sampling method -/
inductive SamplingMethod
  | Lottery
  | RandomNumberTable
  | Stratified
  | Systematic

/-- Represents the auditorium setup and sampling process -/
structure AuditoriumSampling where
  rows : Nat
  seatsPerRow : Nat
  selectedSeatNumber : Nat

/-- Determines the sampling method based on the auditorium setup and sampling process -/
def determineSamplingMethod (setup : AuditoriumSampling) : SamplingMethod := sorry

/-- Theorem stating that the given sampling process is systematic sampling -/
theorem is_systematic_sampling (setup : AuditoriumSampling) 
  (h1 : setup.rows = 40)
  (h2 : setup.seatsPerRow = 25)
  (h3 : setup.selectedSeatNumber = 18) :
  determineSamplingMethod setup = SamplingMethod.Systematic := by sorry

end NUMINAMATH_CALUDE_is_systematic_sampling_l1388_138850


namespace NUMINAMATH_CALUDE_least_integer_with_12_factors_l1388_138816

/-- The number of positive factors of a positive integer -/
def num_factors (n : ℕ+) : ℕ := sorry

/-- Checks if a positive integer has exactly 12 factors -/
def has_12_factors (n : ℕ+) : Prop := num_factors n = 12

theorem least_integer_with_12_factors :
  ∃ (k : ℕ+), has_12_factors k ∧ ∀ (m : ℕ+), has_12_factors m → k ≤ m :=
by
  use 108
  sorry

end NUMINAMATH_CALUDE_least_integer_with_12_factors_l1388_138816


namespace NUMINAMATH_CALUDE_greatest_common_divisor_540_462_l1388_138889

theorem greatest_common_divisor_540_462 : Nat.gcd 540 462 = 6 := by
  sorry

end NUMINAMATH_CALUDE_greatest_common_divisor_540_462_l1388_138889


namespace NUMINAMATH_CALUDE_deepthi_material_usage_l1388_138815

theorem deepthi_material_usage
  (material1 : ℚ)
  (material2 : ℚ)
  (leftover : ℚ)
  (h1 : material1 = 4 / 17)
  (h2 : material2 = 3 / 10)
  (h3 : leftover = 9 / 30)
  : material1 + material2 - leftover = 4 / 17 := by
  sorry

end NUMINAMATH_CALUDE_deepthi_material_usage_l1388_138815


namespace NUMINAMATH_CALUDE_cyclic_quadrilaterals_count_l1388_138822

/-- A quadrilateral is cyclic if it can be inscribed in a circle. -/
def is_cyclic (q : Quadrilateral) : Prop := sorry

/-- A square is a quadrilateral with all sides equal and all angles right angles. -/
def is_square (q : Quadrilateral) : Prop := sorry

/-- A rectangle is a quadrilateral with all angles right angles. -/
def is_rectangle (q : Quadrilateral) : Prop := sorry

/-- A rhombus is a quadrilateral with all sides equal. -/
def is_rhombus (q : Quadrilateral) : Prop := sorry

/-- A parallelogram is a quadrilateral with opposite sides parallel. -/
def is_parallelogram (q : Quadrilateral) : Prop := sorry

/-- An isosceles trapezoid is a trapezoid with the non-parallel sides equal. -/
def is_isosceles_trapezoid (q : Quadrilateral) : Prop := sorry

theorem cyclic_quadrilaterals_count :
  ∃ (s r h p t : Quadrilateral),
    is_square s ∧
    is_rectangle r ∧ ¬ is_square r ∧
    is_rhombus h ∧ ¬ is_square h ∧
    is_parallelogram p ∧ ¬ is_rectangle p ∧ ¬ is_rhombus p ∧
    is_isosceles_trapezoid t ∧ ¬ is_parallelogram t ∧
    (is_cyclic s ∧ is_cyclic r ∧ is_cyclic t ∧
     ¬ is_cyclic h ∧ ¬ is_cyclic p) :=
by sorry

end NUMINAMATH_CALUDE_cyclic_quadrilaterals_count_l1388_138822


namespace NUMINAMATH_CALUDE_lucky_set_guaranteed_l1388_138841

/-- The number of cards in the deck -/
def deck_size : ℕ := 52

/-- The maximum sum of digits possible for any card in the deck -/
def max_sum : ℕ := 13

/-- Function to calculate the sum of digits for a given number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

/-- Theorem stating that drawing 26 cards guarantees a "lucky" set -/
theorem lucky_set_guaranteed (drawn : ℕ) (h : drawn = 26) :
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  a ≤ deck_size ∧ b ≤ deck_size ∧ c ≤ deck_size ∧
  sum_of_digits a = sum_of_digits b ∧ sum_of_digits b = sum_of_digits c :=
by sorry

end NUMINAMATH_CALUDE_lucky_set_guaranteed_l1388_138841


namespace NUMINAMATH_CALUDE_base_equation_solution_l1388_138864

/-- Convert a number from base b to base 10 -/
def toBase10 (digits : List Nat) (b : Nat) : Nat :=
  digits.foldr (fun d acc => d + b * acc) 0

/-- Given positive integers C and D where D = C + 2, 
    and the equation 253_C - 75_D = 124_(C+D) holds, 
    prove that C + D = 26 -/
theorem base_equation_solution (C D : Nat) 
  (h1 : C > 0) 
  (h2 : D > 0) 
  (h3 : D = C + 2) 
  (h4 : toBase10 [2, 5, 3] C - toBase10 [7, 5] D = toBase10 [1, 2, 4] (C + D)) :
  C + D = 26 := by
  sorry

end NUMINAMATH_CALUDE_base_equation_solution_l1388_138864


namespace NUMINAMATH_CALUDE_flagpole_distance_l1388_138852

/-- Given a street of length 11.5 meters with 6 flagpoles placed at regular intervals,
    including both ends, the distance between adjacent flagpoles is 2.3 meters. -/
theorem flagpole_distance (street_length : ℝ) (num_flagpoles : ℕ) :
  street_length = 11.5 ∧ num_flagpoles = 6 →
  (street_length / (num_flagpoles - 1 : ℝ)) = 2.3 := by
  sorry

end NUMINAMATH_CALUDE_flagpole_distance_l1388_138852


namespace NUMINAMATH_CALUDE_negative_cube_squared_l1388_138819

theorem negative_cube_squared (a : ℝ) : (-a^3)^2 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_negative_cube_squared_l1388_138819


namespace NUMINAMATH_CALUDE_chord_passes_through_fixed_point_min_distance_perpendicular_chords_l1388_138824

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 4*y

-- Define the line l
def line_l (x y : ℝ) : Prop := y = -1

-- Define a point on the parabola
def point_on_parabola (x y : ℝ) : Prop := parabola x y

-- Define a point on line l
def point_on_line_l (x y : ℝ) : Prop := line_l x y

-- Define the chord of tangent points
def chord_of_tangent_points (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  point_on_parabola x₁ y₁ ∧ point_on_parabola x₂ y₂

-- Theorem 1: The chord of tangent points passes through (0, 1)
theorem chord_passes_through_fixed_point :
  ∀ x₀ y₀ x₁ y₁ x₂ y₂ : ℝ,
  point_on_line_l x₀ y₀ →
  chord_of_tangent_points x₁ y₁ x₂ y₂ →
  ∃ t : ℝ, t * x₁ + (1 - t) * x₂ = 0 ∧ t * y₁ + (1 - t) * y₂ = 1 :=
sorry

-- Theorem 2: Minimum distance between P and Q when chords are perpendicular
theorem min_distance_perpendicular_chords :
  ∃ xP yP xQ yQ : ℝ,
  point_on_line_l xP yP ∧ point_on_line_l xQ yQ ∧
  (∀ x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ,
    chord_of_tangent_points x₁ y₁ x₂ y₂ ∧
    chord_of_tangent_points x₃ y₃ x₄ y₄ →
    (x₂ - x₁) * (x₄ - x₃) + (y₂ - y₁) * (y₄ - y₃) = 0 →
    (xQ - xP)^2 + (yQ - yP)^2 ≤ (x - xP)^2 + (y - yP)^2) ∧
  xP = -2 ∧ yP = -1 ∧ xQ = 2 ∧ yQ = -1 ∧
  (xQ - xP)^2 + (yQ - yP)^2 = 16 :=
sorry

end NUMINAMATH_CALUDE_chord_passes_through_fixed_point_min_distance_perpendicular_chords_l1388_138824


namespace NUMINAMATH_CALUDE_max_difference_intersection_points_l1388_138887

/-- The first function f(x) = 2 - x^2 + 2x^3 -/
def f (x : ℝ) : ℝ := 2 - x^2 + 2*x^3

/-- The second function g(x) = 3 + 2x^2 + 2x^3 -/
def g (x : ℝ) : ℝ := 3 + 2*x^2 + 2*x^3

/-- Theorem stating that the maximum difference between y-coordinates of intersection points is 4√3/9 -/
theorem max_difference_intersection_points :
  ∃ (x₁ x₂ : ℝ), f x₁ = g x₁ ∧ f x₂ = g x₂ ∧ 
  ∀ (y₁ y₂ : ℝ), (∃ (x : ℝ), f x = g x ∧ (y₁ = f x ∨ y₁ = g x)) →
                 (∃ (x : ℝ), f x = g x ∧ (y₂ = f x ∨ y₂ = g x)) →
                 |y₁ - y₂| ≤ 4 * Real.sqrt 3 / 9 :=
by sorry

end NUMINAMATH_CALUDE_max_difference_intersection_points_l1388_138887


namespace NUMINAMATH_CALUDE_sum_of_fourth_powers_l1388_138804

theorem sum_of_fourth_powers (a b c : ℝ) 
  (sum_zero : a + b + c = 0)
  (sum_squares : a^2 + b^2 + c^2 = 3)
  (sum_cubes : a^3 + b^3 + c^3 = 6) :
  a^4 + b^4 + c^4 = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fourth_powers_l1388_138804


namespace NUMINAMATH_CALUDE_defective_item_testing_methods_l1388_138888

theorem defective_item_testing_methods :
  let genuine_items : ℕ := 6
  let defective_items : ℕ := 4
  let total_tests : ℕ := 5
  let last_test_defective : ℕ := 1
  let genuine_in_first_four : ℕ := 1
  let defective_in_first_four : ℕ := 3

  (Nat.choose defective_items last_test_defective) *
  (Nat.choose genuine_items genuine_in_first_four) *
  (Nat.choose defective_in_first_four defective_in_first_four) *
  (Nat.factorial defective_in_first_four) = 576 :=
by
  sorry

end NUMINAMATH_CALUDE_defective_item_testing_methods_l1388_138888


namespace NUMINAMATH_CALUDE_village_b_largest_population_l1388_138892

/-- Calculate the population after n years given initial population and growth rate -/
def futurePopulation (initialPop : ℝ) (growthRate : ℝ) (years : ℕ) : ℝ :=
  initialPop * (1 + growthRate) ^ years

/-- Theorem: Village B has the largest population after 3 years -/
theorem village_b_largest_population :
  let villageA := futurePopulation 12000 0.24 3
  let villageB := futurePopulation 15000 0.18 3
  let villageC := futurePopulation 18000 (-0.12) 3
  villageB > villageA ∧ villageB > villageC := by sorry

end NUMINAMATH_CALUDE_village_b_largest_population_l1388_138892


namespace NUMINAMATH_CALUDE_sin_arccos_eight_seventeenths_l1388_138827

theorem sin_arccos_eight_seventeenths : 
  Real.sin (Real.arccos (8 / 17)) = 15 / 17 := by
  sorry

end NUMINAMATH_CALUDE_sin_arccos_eight_seventeenths_l1388_138827


namespace NUMINAMATH_CALUDE_value_of_3b_plus_4c_l1388_138848

-- Define the function g
def g (x : ℝ) : ℝ := 3 * x + 2

-- Define the function f
def f (b c x : ℝ) : ℝ := b * x + c

-- State the theorem
theorem value_of_3b_plus_4c (b c : ℝ) :
  (∃ f_inv : ℝ → ℝ, 
    (∀ x, f b c (f_inv x) = x ∧ f_inv (f b c x) = x) ∧ 
    (∀ x, g x = 2 * f_inv x + 4)) →
  3 * b + 4 * c = 14/3 :=
by sorry

end NUMINAMATH_CALUDE_value_of_3b_plus_4c_l1388_138848


namespace NUMINAMATH_CALUDE_polynomial_equality_implies_sum_l1388_138839

theorem polynomial_equality_implies_sum (a b c d e f : ℝ) :
  (∀ x : ℝ, (3*x + 1)^5 = a*x^5 + b*x^4 + c*x^3 + d*x^2 + e*x + f) →
  a - b + c - d + e - f = 32 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_implies_sum_l1388_138839


namespace NUMINAMATH_CALUDE_board_numbers_theorem_l1388_138820

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def proper_divisors (n : ℕ) : Set ℕ :=
  {d : ℕ | d ∣ n ∧ 1 < d ∧ d < n}

theorem board_numbers_theorem (n : ℕ) (hn : is_composite n) :
  (∃ m : ℕ, proper_divisors m = {d + 1 | d ∈ proper_divisors n}) ↔ n = 4 ∨ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_board_numbers_theorem_l1388_138820


namespace NUMINAMATH_CALUDE_at_least_one_chinese_book_l1388_138846

def total_books : ℕ := 12
def chinese_books : ℕ := 10
def math_books : ℕ := 2
def drawn_books : ℕ := 3

theorem at_least_one_chinese_book :
  ∀ (selection : Finset ℕ),
  selection.card = drawn_books →
  (∀ i ∈ selection, i < total_books) →
  ∃ i ∈ selection, i < chinese_books :=
by sorry

end NUMINAMATH_CALUDE_at_least_one_chinese_book_l1388_138846


namespace NUMINAMATH_CALUDE_waiter_customers_l1388_138896

/-- Given a waiter with tables, each having a certain number of women and men,
    calculate the total number of customers. -/
theorem waiter_customers (tables women_per_table men_per_table : ℕ) :
  tables = 5 →
  women_per_table = 5 →
  men_per_table = 3 →
  tables * (women_per_table + men_per_table) = 40 := by
sorry

end NUMINAMATH_CALUDE_waiter_customers_l1388_138896


namespace NUMINAMATH_CALUDE_swim_team_girls_count_l1388_138823

/-- Proves that the number of girls on a swim team is 80, given the specified conditions -/
theorem swim_team_girls_count : 
  ∀ (boys girls : ℕ), 
  girls = 5 * boys → 
  boys + girls = 96 → 
  girls = 80 := by
sorry

end NUMINAMATH_CALUDE_swim_team_girls_count_l1388_138823


namespace NUMINAMATH_CALUDE_shekar_marks_problem_l1388_138840

/-- Represents a student's marks in various subjects -/
structure StudentMarks where
  science : ℕ
  socialStudies : ℕ
  english : ℕ
  biology : ℕ
  mathematics : ℕ

/-- Calculates the average marks of a student -/
def averageMarks (marks : StudentMarks) : ℚ :=
  (marks.science + marks.socialStudies + marks.english + marks.biology + marks.mathematics) / 5

/-- Shekar's marks problem -/
theorem shekar_marks_problem (shekar : StudentMarks)
    (h1 : shekar.science = 65)
    (h2 : shekar.socialStudies = 82)
    (h3 : shekar.english = 67)
    (h4 : shekar.biology = 95)
    (h5 : averageMarks shekar = 77) :
    shekar.mathematics = 76 := by
  sorry

end NUMINAMATH_CALUDE_shekar_marks_problem_l1388_138840


namespace NUMINAMATH_CALUDE_negation_equivalence_l1388_138838

-- Define the universe of discourse
variable (Student : Type)

-- Define the property of being patient
variable (isPatient : Student → Prop)

-- Statement (6): All students are patient
def allStudentsPatient : Prop := ∀ s : Student, isPatient s

-- Statement (5): At least one student is impatient
def oneStudentImpatient : Prop := ∃ s : Student, ¬(isPatient s)

-- Theorem: Statement (5) is equivalent to the negation of statement (6)
theorem negation_equivalence : oneStudentImpatient Student isPatient ↔ ¬(allStudentsPatient Student isPatient) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1388_138838


namespace NUMINAMATH_CALUDE_range_of_a_l1388_138829

-- Define the propositions p and q
def p (x : ℝ) : Prop := x ≤ -1
def q (a x : ℝ) : Prop := a ≤ x ∧ x < a + 2

-- Define the sufficient but not necessary condition
def sufficient_not_necessary (a : ℝ) : Prop :=
  (∀ x, q a x → p x) ∧ ¬(∀ x, p x → q a x)

-- Theorem statement
theorem range_of_a : 
  ∀ a : ℝ, sufficient_not_necessary a ↔ a ≤ -3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1388_138829


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l1388_138808

theorem complex_number_quadrant (z : ℂ) (h : z * (2 - Complex.I) = 1) :
  0 < z.re ∧ 0 < z.im := by sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l1388_138808


namespace NUMINAMATH_CALUDE_freddy_age_l1388_138817

theorem freddy_age (F M R : ℕ) 
  (sum_ages : F + M + R = 35)
  (matthew_rebecca : M = R + 2)
  (freddy_matthew : F = M + 4) :
  F = 15 := by
sorry

end NUMINAMATH_CALUDE_freddy_age_l1388_138817


namespace NUMINAMATH_CALUDE_sphere_radius_from_hole_l1388_138874

/-- Given a spherical hole in ice with a diameter of 30 cm at the surface and a depth of 10 cm,
    the radius of the sphere that created this hole is 16.25 cm. -/
theorem sphere_radius_from_hole (diameter : ℝ) (depth : ℝ) (radius : ℝ) :
  diameter = 30 ∧ depth = 10 ∧ radius = (diameter / 2)^2 / (4 * depth) + depth / 4 →
  radius = 16.25 :=
by sorry

end NUMINAMATH_CALUDE_sphere_radius_from_hole_l1388_138874


namespace NUMINAMATH_CALUDE_train_speed_calculation_l1388_138878

/-- Proves that given the specified conditions, the train's speed is 36 kmph -/
theorem train_speed_calculation (jogger_speed : ℝ) (jogger_ahead : ℝ) (train_length : ℝ) (pass_time : ℝ) :
  jogger_speed = 9 →
  jogger_ahead = 240 →
  train_length = 120 →
  pass_time = 35.99712023038157 →
  (jogger_ahead + train_length) / pass_time * 3.6 = 36 := by
  sorry

#eval (240 + 120) / 35.99712023038157 * 3.6

end NUMINAMATH_CALUDE_train_speed_calculation_l1388_138878


namespace NUMINAMATH_CALUDE_intersection_range_l1388_138835

/-- The function f(x) = x³ - 3x - 1 --/
def f (x : ℝ) : ℝ := x^3 - 3*x - 1

/-- Theorem: If the line y = m intersects the graph of f(x) = x³ - 3x - 1
    at three distinct points, then m is in the open interval (-3, 1) --/
theorem intersection_range (m : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ 
    f x₁ = m ∧ f x₂ = m ∧ f x₃ = m) →
  m > -3 ∧ m < 1 :=
sorry

end NUMINAMATH_CALUDE_intersection_range_l1388_138835


namespace NUMINAMATH_CALUDE_triangle_side_length_l1388_138842

-- Define the triangle and circle
structure Triangle :=
  (A B C O : ℝ × ℝ)
  (circumscribed : Bool)

-- Define the properties of the triangle
def is_valid_triangle (t : Triangle) : Prop :=
  t.circumscribed ∧ 
  dist t.B t.C = 5 ∧
  dist t.A t.B = 4 ∧
  norm (3 • (t.A - t.O) - 4 • (t.B - t.O) + (t.C - t.O)) = 10

-- Theorem statement
theorem triangle_side_length (t : Triangle) :
  is_valid_triangle t → dist t.A t.C = 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1388_138842


namespace NUMINAMATH_CALUDE_mel_age_when_katherine_is_two_dozen_l1388_138858

/-- The age difference between Katherine and Mel -/
def age_difference : ℕ := 3

/-- Katherine's age when she is two dozen years old -/
def katherine_age : ℕ := 24

/-- Mel's age when Katherine is two dozen years old -/
def mel_age : ℕ := katherine_age - age_difference

theorem mel_age_when_katherine_is_two_dozen : mel_age = 21 := by
  sorry

end NUMINAMATH_CALUDE_mel_age_when_katherine_is_two_dozen_l1388_138858


namespace NUMINAMATH_CALUDE_melted_ice_cream_height_l1388_138813

/-- Given a sphere of ice cream with radius 3 inches that melts into a cylinder
    with radius 12 inches while maintaining constant density, the height of the
    resulting cylinder is 1/4 inch. -/
theorem melted_ice_cream_height (r_sphere r_cylinder : ℝ) (h : ℝ) : 
  r_sphere = 3 →
  r_cylinder = 12 →
  (4 / 3) * π * r_sphere^3 = π * r_cylinder^2 * h →
  h = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_melted_ice_cream_height_l1388_138813


namespace NUMINAMATH_CALUDE_notebook_cost_l1388_138882

/-- The cost of a notebook and pencil, given their relationship -/
theorem notebook_cost (notebook_cost pencil_cost : ℝ)
  (total_cost : notebook_cost + pencil_cost = 3.20)
  (cost_difference : notebook_cost = pencil_cost + 2.50) :
  notebook_cost = 2.85 := by
  sorry

end NUMINAMATH_CALUDE_notebook_cost_l1388_138882


namespace NUMINAMATH_CALUDE_password_recovery_l1388_138875

def alphabet_size : Nat := 32

def encode (c : Char) : Nat := 
  sorry

def decode (n : Nat) : Char := 
  sorry

def generate_x (a b x : Nat) : Nat :=
  (a * x + b) % 10

def generate_c (x y : Nat) : Nat :=
  (x + y) % 10

def is_valid_sequence (s : List Nat) (password : String) (a b : Nat) : Prop :=
  sorry

theorem password_recovery (a b : Nat) : 
  ∃ (password : String),
    password.length = 4 ∧ 
    is_valid_sequence [2, 8, 5, 2, 8, 3, 1, 9, 8, 4, 1, 8, 4, 9, 7] (password ++ password) a b ∧
    password = "яхта" :=
  sorry

end NUMINAMATH_CALUDE_password_recovery_l1388_138875


namespace NUMINAMATH_CALUDE_two_digit_number_property_l1388_138812

theorem two_digit_number_property (N : ℕ) : 
  (N ≥ 10 ∧ N ≤ 99) →
  (4 * (N / 10) + 2 * (N % 10) = N / 2) →
  (N = 32 ∨ N = 64 ∨ N = 96) :=
by sorry

end NUMINAMATH_CALUDE_two_digit_number_property_l1388_138812


namespace NUMINAMATH_CALUDE_parallelogram_area_equality_l1388_138891

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a triangle -/
structure Triangle :=
  (A B C : Point)

/-- Represents a parallelogram -/
structure Parallelogram :=
  (A B C D : Point)

/-- Calculates the area of a parallelogram -/
def area (p : Parallelogram) : ℝ := sorry

/-- Given a triangle ABC, constructs parallelogram ACDE on side AC -/
def constructParallelogramOnAC (t : Triangle) : Parallelogram := sorry

/-- Given a triangle ABC, constructs parallelogram BCFG on side BC -/
def constructParallelogramOnBC (t : Triangle) : Parallelogram := sorry

/-- Given a triangle ABC and point H, constructs parallelogram ABML on side AB 
    such that AL and BM are equal and parallel to HC -/
def constructParallelogramOnAB (t : Triangle) (H : Point) : Parallelogram := sorry

/-- Main theorem statement -/
theorem parallelogram_area_equality 
  (t : Triangle) 
  (H : Point) 
  (ACDE : Parallelogram) 
  (BCFG : Parallelogram) 
  (ABML : Parallelogram) 
  (h1 : ACDE = constructParallelogramOnAC t) 
  (h2 : BCFG = constructParallelogramOnBC t) 
  (h3 : ABML = constructParallelogramOnAB t H) :
  area ABML = area ACDE + area BCFG := by sorry

end NUMINAMATH_CALUDE_parallelogram_area_equality_l1388_138891


namespace NUMINAMATH_CALUDE_no_real_solutions_for_f_iteration_l1388_138801

def f (x : ℝ) : ℝ := x^2 + 2*x

theorem no_real_solutions_for_f_iteration :
  ¬ ∃ c : ℝ, f (f (f (f c))) = -4 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_for_f_iteration_l1388_138801


namespace NUMINAMATH_CALUDE_brownie_pieces_l1388_138814

theorem brownie_pieces (pan_length pan_width piece_length piece_width : ℕ) 
  (h1 : pan_length = 30)
  (h2 : pan_width = 24)
  (h3 : piece_length = 3)
  (h4 : piece_width = 4) :
  (pan_length * pan_width) / (piece_length * piece_width) = 60 := by
  sorry

end NUMINAMATH_CALUDE_brownie_pieces_l1388_138814


namespace NUMINAMATH_CALUDE_quadratic_increasing_negative_l1388_138857

-- Define the quadratic function
def f (x : ℝ) : ℝ := 2 * x^2

-- Theorem statement
theorem quadratic_increasing_negative (x₁ x₂ : ℝ) :
  x₁ < x₂ ∧ x₂ < 0 → f x₁ < f x₂ := by
  sorry

end NUMINAMATH_CALUDE_quadratic_increasing_negative_l1388_138857


namespace NUMINAMATH_CALUDE_line_through_point_with_equal_intercepts_l1388_138855

-- Define a line in 2D space
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ
  nonzero : a ≠ 0 ∨ b ≠ 0

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Function to check if a point lies on a line
def pointOnLine (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Function to check if a line has equal intercepts on both axes
def equalIntercepts (l : Line2D) : Prop :=
  l.a ≠ 0 ∧ l.b ≠ 0 ∧ l.c / l.a = l.c / l.b

-- Theorem statement
theorem line_through_point_with_equal_intercepts :
  ∀ (l : Line2D),
    pointOnLine { x := 1, y := 2 } l →
    equalIntercepts l →
    (l.a = 2 ∧ l.b = -1 ∧ l.c = 0) ∨ (l.a = 1 ∧ l.b = 1 ∧ l.c = -3) :=
sorry

end NUMINAMATH_CALUDE_line_through_point_with_equal_intercepts_l1388_138855


namespace NUMINAMATH_CALUDE_union_covers_reals_l1388_138886

open Set Real

theorem union_covers_reals (A B : Set ℝ) (a : ℝ) :
  A = Iic 0 ∧ B = Ioi a ∧ A ∪ B = univ ↔ a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_union_covers_reals_l1388_138886


namespace NUMINAMATH_CALUDE_lily_shopping_ratio_l1388_138894

theorem lily_shopping_ratio (initial_balance shirt_cost final_balance : ℕ) 
  (h1 : initial_balance = 55)
  (h2 : shirt_cost = 7)
  (h3 : final_balance = 27) :
  (initial_balance - shirt_cost - final_balance) / shirt_cost = 3 := by
sorry

end NUMINAMATH_CALUDE_lily_shopping_ratio_l1388_138894


namespace NUMINAMATH_CALUDE_symmetric_line_across_x_axis_l1388_138897

/-- A line in the 2D plane represented by ax + by + c = 0 -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Reflects a line across the x-axis -/
def reflectLineAcrossXAxis (l : Line2D) : Line2D :=
  { a := l.a, b := -l.b, c := l.c }

theorem symmetric_line_across_x_axis :
  let originalLine := Line2D.mk 2 (-3) 2
  let symmetricLine := Line2D.mk 2 3 2
  reflectLineAcrossXAxis originalLine = symmetricLine := by sorry

end NUMINAMATH_CALUDE_symmetric_line_across_x_axis_l1388_138897


namespace NUMINAMATH_CALUDE_remainder_problem_l1388_138856

theorem remainder_problem (N : ℤ) : 
  ∃ k : ℤ, N = 761 * k + 173 → N % 29 = 28 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l1388_138856


namespace NUMINAMATH_CALUDE_new_salary_calculation_l1388_138825

def current_salary : ℝ := 10000
def increase_percentage : ℝ := 0.02

theorem new_salary_calculation :
  current_salary * (1 + increase_percentage) = 10200 := by
  sorry

end NUMINAMATH_CALUDE_new_salary_calculation_l1388_138825


namespace NUMINAMATH_CALUDE_sequence_q_value_max_q_value_l1388_138880

-- Define the arithmetic sequence a_n
def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + d * (n - 1)

-- Define the geometric sequence b_n
def geometric_sequence (b₁ q : ℝ) (n : ℕ) : ℝ := b₁ * q^(n - 1)

-- Define the set E
structure E where
  m : ℕ+
  p : ℕ+
  r : ℕ+
  h_order : m < p ∧ p < r

theorem sequence_q_value
  (a₁ d b₁ q : ℝ)
  (hq : q ≠ 1 ∧ q ≠ -1)
  (h_equality : arithmetic_sequence a₁ d 1 + geometric_sequence b₁ q 2 =
                arithmetic_sequence a₁ d 2 + geometric_sequence b₁ q 3 ∧
                arithmetic_sequence a₁ d 2 + geometric_sequence b₁ q 3 =
                arithmetic_sequence a₁ d 3 + geometric_sequence b₁ q 1) :
  q = -1/2 :=
sorry

theorem max_q_value
  (a₁ d b₁ q : ℝ)
  (e : E)
  (hq : q ≠ 1 ∧ q ≠ -1)
  (h_arithmetic : ∃ (k : ℝ), k > 1 ∧ e.p = e.m + k ∧ e.r = e.p + k)
  (h_equality : arithmetic_sequence a₁ d e.m + geometric_sequence b₁ q e.p =
                arithmetic_sequence a₁ d e.p + geometric_sequence b₁ q e.r ∧
                arithmetic_sequence a₁ d e.p + geometric_sequence b₁ q e.r =
                arithmetic_sequence a₁ d e.r + geometric_sequence b₁ q e.m) :
  q ≤ -(1/2)^(1/3) :=
sorry

end NUMINAMATH_CALUDE_sequence_q_value_max_q_value_l1388_138880


namespace NUMINAMATH_CALUDE_perpendicular_lines_parameter_l1388_138833

/-- Given two lines ax + y - 1 = 0 and 4x + (a - 5)y - 2 = 0 that are perpendicular,
    prove that a = 1 -/
theorem perpendicular_lines_parameter (a : ℝ) :
  (∃ x y, a * x + y - 1 = 0 ∧ 4 * x + (a - 5) * y - 2 = 0) →
  (∀ x₁ y₁ x₂ y₂, 
    (a * x₁ + y₁ - 1 = 0 ∧ 4 * x₁ + (a - 5) * y₁ - 2 = 0) →
    (a * x₂ + y₂ - 1 = 0 ∧ 4 * x₂ + (a - 5) * y₂ - 2 = 0) →
    (x₂ - x₁) * (x₂ - x₁) + (y₂ - y₁) * (y₂ - y₁) ≠ 0 →
    (a * (x₂ - x₁) + (y₂ - y₁)) * (4 * (x₂ - x₁) + (a - 5) * (y₂ - y₁)) = 0) →
  a = 1 :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_parameter_l1388_138833


namespace NUMINAMATH_CALUDE_sin_ten_degrees_root_l1388_138834

theorem sin_ten_degrees_root : ∃ x : ℝ, 
  (x = Real.sin (10 * π / 180)) ∧ 
  (8 * x^3 - 6 * x + 1 = 0) := by
  sorry

end NUMINAMATH_CALUDE_sin_ten_degrees_root_l1388_138834


namespace NUMINAMATH_CALUDE_circles_externally_tangent_l1388_138870

theorem circles_externally_tangent (x y : ℝ) : 
  let circle1 := {(x, y) : ℝ × ℝ | x^2 + y^2 - 6*x = 0}
  let circle2 := {(x, y) : ℝ × ℝ | x^2 + y^2 + 8*y + 12 = 0}
  let center1 := (3, 0)
  let center2 := (0, -4)
  let radius1 := 3
  let radius2 := 2
  (∀ (p : ℝ × ℝ), p ∈ circle1 ↔ (p.1 - center1.1)^2 + (p.2 - center1.2)^2 = radius1^2) ∧
  (∀ (p : ℝ × ℝ), p ∈ circle2 ↔ (p.1 - center2.1)^2 + (p.2 - center2.2)^2 = radius2^2) ∧
  (center1.1 - center2.1)^2 + (center1.2 - center2.2)^2 = (radius1 + radius2)^2 :=
by
  sorry

end NUMINAMATH_CALUDE_circles_externally_tangent_l1388_138870


namespace NUMINAMATH_CALUDE_max_common_ratio_arithmetic_geometric_l1388_138869

theorem max_common_ratio_arithmetic_geometric (a : ℕ → ℝ) (d q : ℝ) (k : ℕ) :
  (∀ n, a (n + 1) - a n = d) →  -- arithmetic sequence condition
  d ≠ 0 →  -- non-zero common difference
  k ≥ 2 →  -- k condition
  a k / a 1 = q →  -- geometric sequence condition for a_1 and a_k
  a (2 * k) / a k = q →  -- geometric sequence condition for a_k and a_2k
  q ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_max_common_ratio_arithmetic_geometric_l1388_138869


namespace NUMINAMATH_CALUDE_vehicle_inspection_is_systematic_sampling_l1388_138868

/-- Represents a sampling method --/
structure SamplingMethod where
  name : String
  selectionCriteria : String
  isFixedInterval : Bool

/-- Defines systematic sampling --/
def systematicSampling : SamplingMethod where
  name := "Systematic Sampling"
  selectionCriteria := "Fixed periodic interval"
  isFixedInterval := true

/-- Represents the vehicle emission inspection method --/
def vehicleInspectionMethod : SamplingMethod where
  name := "Vehicle Inspection Method"
  selectionCriteria := "License plates ending in 8"
  isFixedInterval := true

/-- Theorem stating that the vehicle inspection method is systematic sampling --/
theorem vehicle_inspection_is_systematic_sampling :
  vehicleInspectionMethod = systematicSampling :=
by sorry

end NUMINAMATH_CALUDE_vehicle_inspection_is_systematic_sampling_l1388_138868


namespace NUMINAMATH_CALUDE_not_always_int_greater_than_decimal_l1388_138828

-- Define a decimal as a structure with an integer part and a fractional part
structure Decimal where
  integerPart : Int
  fractionalPart : Rat
  fractionalPart_lt_one : fractionalPart < 1

-- Define the comparison between an integer and a decimal
def intGreaterThanDecimal (n : Int) (d : Decimal) : Prop :=
  n > d.integerPart + d.fractionalPart

-- Theorem statement
theorem not_always_int_greater_than_decimal :
  ¬ ∀ (n : Int) (d : Decimal), intGreaterThanDecimal n d :=
sorry

end NUMINAMATH_CALUDE_not_always_int_greater_than_decimal_l1388_138828


namespace NUMINAMATH_CALUDE_complement_union_theorem_l1388_138899

def I : Set Nat := {1, 2, 3, 4}
def S : Set Nat := {1, 3}
def T : Set Nat := {4}

theorem complement_union_theorem :
  (I \ S) ∪ T = {2, 4} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l1388_138899
