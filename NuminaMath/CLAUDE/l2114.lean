import Mathlib

namespace NUMINAMATH_CALUDE_world_cup_souvenir_production_l2114_211487

def planned_daily_production : ℕ := 10000

def production_deviations : List ℤ := [41, -34, -52, 127, -72, 36, -29]

def production_cost : ℕ := 35

def selling_price : ℕ := 40

theorem world_cup_souvenir_production 
  (planned_daily_production : ℕ)
  (production_deviations : List ℤ)
  (production_cost selling_price : ℕ)
  (h1 : planned_daily_production = 10000)
  (h2 : production_deviations = [41, -34, -52, 127, -72, 36, -29])
  (h3 : production_cost = 35)
  (h4 : selling_price = 40) :
  (∃ (max min : ℤ), max ∈ production_deviations ∧ 
                    min ∈ production_deviations ∧ 
                    max - min = 199) ∧
  (production_deviations.sum = 17) ∧
  ((7 * planned_daily_production + production_deviations.sum) * 
   (selling_price - production_cost) = 350085) :=
by sorry

end NUMINAMATH_CALUDE_world_cup_souvenir_production_l2114_211487


namespace NUMINAMATH_CALUDE_polynomial_simplification_l2114_211422

theorem polynomial_simplification (x : ℝ) :
  (2 * x^6 + 3 * x^5 + x^2 + 15) - (x^6 + 4 * x^5 - 2 * x^3 + 20) = x^6 - x^5 + 2 * x^3 - 5 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2114_211422


namespace NUMINAMATH_CALUDE_box_velvet_problem_l2114_211477

theorem box_velvet_problem (long_side_length long_side_width short_side_length short_side_width total_velvet : ℕ) 
  (h1 : long_side_length = 8)
  (h2 : long_side_width = 6)
  (h3 : short_side_length = 5)
  (h4 : short_side_width = 6)
  (h5 : total_velvet = 236) :
  let side_area := 2 * (long_side_length * long_side_width + short_side_length * short_side_width)
  let remaining_area := total_velvet - side_area
  (remaining_area / 2 : ℕ) = 40 := by
  sorry

end NUMINAMATH_CALUDE_box_velvet_problem_l2114_211477


namespace NUMINAMATH_CALUDE_jasons_quarters_l2114_211492

theorem jasons_quarters (initial final given : ℕ) 
  (h1 : initial = 49)
  (h2 : final = 74)
  (h3 : final = initial + given) :
  given = 25 := by
  sorry

end NUMINAMATH_CALUDE_jasons_quarters_l2114_211492


namespace NUMINAMATH_CALUDE_max_value_sin_cos_l2114_211493

theorem max_value_sin_cos (θ : Real) (h : 0 < θ ∧ θ < π) :
  ∃ (max : Real), max = (4 * Real.sqrt 3) / 9 ∧
  ∀ (x : Real), 0 < x ∧ x < π →
    Real.sin (x / 2) * (1 + Real.cos x) ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_sin_cos_l2114_211493


namespace NUMINAMATH_CALUDE_quadratic_polynomial_existence_l2114_211481

/-- A quadratic polynomial with real coefficients -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Evaluate a quadratic polynomial at a complex number -/
def evaluate (p : QuadraticPolynomial) (z : ℂ) : ℂ :=
  p.a * z^2 + p.b * z + p.c

theorem quadratic_polynomial_existence : ∃ (p : QuadraticPolynomial),
  (evaluate p (-3 - 4*I) = 0) ∧ 
  (p.b = -10) ∧
  (p.a = -5/3) ∧ 
  (p.c = -125/3) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_existence_l2114_211481


namespace NUMINAMATH_CALUDE_orange_harvest_per_day_l2114_211463

theorem orange_harvest_per_day (total_sacks : ℕ) (total_days : ℕ) (sacks_per_day : ℕ) :
  total_sacks = 24 →
  total_days = 3 →
  sacks_per_day = total_sacks / total_days →
  sacks_per_day = 8 := by
sorry

end NUMINAMATH_CALUDE_orange_harvest_per_day_l2114_211463


namespace NUMINAMATH_CALUDE_interest_calculation_years_l2114_211490

theorem interest_calculation_years (P r : ℝ) (h1 : P = 625) (h2 : r = 0.04) : 
  ∃ n : ℕ, n = 2 ∧ P * ((1 + r)^n - 1) - P * r * n = 1 := by
  sorry

end NUMINAMATH_CALUDE_interest_calculation_years_l2114_211490


namespace NUMINAMATH_CALUDE_ceiling_floor_product_range_l2114_211471

theorem ceiling_floor_product_range (y : ℝ) :
  y < 0 → ⌈y⌉ * ⌊y⌋ = 72 → -9 < y ∧ y < -8 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_product_range_l2114_211471


namespace NUMINAMATH_CALUDE_tan_theta_right_triangle_l2114_211486

theorem tan_theta_right_triangle (BC AC BA : ℝ) (h1 : BC = 25) (h2 : AC = 20) 
  (h3 : BA^2 + AC^2 = BC^2) : 
  Real.tan (Real.arcsin (BA / BC)) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_tan_theta_right_triangle_l2114_211486


namespace NUMINAMATH_CALUDE_max_triangle_area_l2114_211431

noncomputable section

def ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

def focal_distance : ℝ := 2

def eccentricity : ℝ := Real.sqrt 2 / 2

def right_focus : ℝ × ℝ := (1, 0)

def point_k : ℝ × ℝ := (2, 0)

def line_intersects_ellipse (k : ℝ) (x y : ℝ) : Prop :=
  y = k * (x - 2) ∧ ellipse x y

def triangle_area (F P Q : ℝ × ℝ) : ℝ :=
  abs ((P.1 - F.1) * (Q.2 - F.2) - (Q.1 - F.1) * (P.2 - F.2)) / 2

theorem max_triangle_area :
  ∃ (max_area : ℝ),
    max_area = Real.sqrt 2 / 4 ∧
    ∀ (k : ℝ) (P Q : ℝ × ℝ),
      k ≠ 0 →
      line_intersects_ellipse k P.1 P.2 →
      line_intersects_ellipse k Q.1 Q.2 →
      P ≠ Q →
      triangle_area right_focus P Q ≤ max_area :=
sorry

end

end NUMINAMATH_CALUDE_max_triangle_area_l2114_211431


namespace NUMINAMATH_CALUDE_oplus_problem_l2114_211468

def oplus (a b : ℚ) : ℚ := a^3 / b

theorem oplus_problem : 
  let x := oplus (oplus 2 4) 6
  let y := oplus 2 (oplus 4 6)
  x - y = 7/12 := by sorry

end NUMINAMATH_CALUDE_oplus_problem_l2114_211468


namespace NUMINAMATH_CALUDE_difference_of_squares_example_l2114_211412

theorem difference_of_squares_example : (25 + 15)^2 - (25 - 15)^2 = 1500 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_example_l2114_211412


namespace NUMINAMATH_CALUDE_distance_major_minor_endpoints_l2114_211435

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := 4 * (x + 2)^2 + 16 * y^2 = 64

-- Define the center of the ellipse
def center : ℝ × ℝ := (-2, 0)

-- Define the semi-major axis length
def semi_major : ℝ := 4

-- Define the semi-minor axis length
def semi_minor : ℝ := 1

-- Define an endpoint of the major axis
def major_endpoint : ℝ × ℝ := (center.1 + semi_major, center.2)

-- Define an endpoint of the minor axis
def minor_endpoint : ℝ × ℝ := (center.1, center.2 + semi_minor)

-- Theorem statement
theorem distance_major_minor_endpoints : 
  Real.sqrt ((major_endpoint.1 - minor_endpoint.1)^2 + (major_endpoint.2 - minor_endpoint.2)^2) = Real.sqrt 17 := by
  sorry

end NUMINAMATH_CALUDE_distance_major_minor_endpoints_l2114_211435


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2114_211499

theorem inequality_solution_set (x : ℝ) : x^2 - |x| - 6 < 0 ↔ -3 < x ∧ x < 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2114_211499


namespace NUMINAMATH_CALUDE_oil_leak_calculation_l2114_211466

theorem oil_leak_calculation (total_leaked : ℕ) (leaked_before : ℕ) 
  (h1 : total_leaked = 6206)
  (h2 : leaked_before = 2475) :
  total_leaked - leaked_before = 3731 :=
by sorry

end NUMINAMATH_CALUDE_oil_leak_calculation_l2114_211466


namespace NUMINAMATH_CALUDE_problem_proof_l2114_211443

theorem problem_proof : 
  (14^2 * 5^3) / 568 = 43.13380281690141 := by sorry

end NUMINAMATH_CALUDE_problem_proof_l2114_211443


namespace NUMINAMATH_CALUDE_set_234_not_right_triangle_l2114_211413

/-- A function that checks if three numbers can form a right triangle -/
def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

/-- Theorem stating that the set {2, 3, 4} cannot form a right triangle -/
theorem set_234_not_right_triangle :
  ¬ (is_right_triangle 2 3 4) := by
  sorry

end NUMINAMATH_CALUDE_set_234_not_right_triangle_l2114_211413


namespace NUMINAMATH_CALUDE_hyperbola_m_range_l2114_211469

-- Define the equation
def is_hyperbola (m : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (m + 1) + y^2 / (m + 2) = 1 ∧ (m + 1) * (m + 2) < 0

-- State the theorem
theorem hyperbola_m_range :
  ∀ m : ℝ, is_hyperbola m → -2 < m ∧ m < -1 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_m_range_l2114_211469


namespace NUMINAMATH_CALUDE_smarties_remainder_l2114_211495

theorem smarties_remainder (m : ℕ) (h : m % 11 = 5) : (2 * m) % 11 = 10 := by
  sorry

end NUMINAMATH_CALUDE_smarties_remainder_l2114_211495


namespace NUMINAMATH_CALUDE_root_squared_plus_double_eq_three_l2114_211453

theorem root_squared_plus_double_eq_three (m : ℝ) : 
  m^2 + 2*m - 3 = 0 → m^2 + 2*m = 3 := by
  sorry

end NUMINAMATH_CALUDE_root_squared_plus_double_eq_three_l2114_211453


namespace NUMINAMATH_CALUDE_calculation_proof_l2114_211446

theorem calculation_proof : (0.08 / 0.002) * 0.5 = 20 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2114_211446


namespace NUMINAMATH_CALUDE_ticket_sales_l2114_211415

theorem ticket_sales (adult_price children_price senior_price discount : ℕ)
  (total_receipts total_attendance : ℕ)
  (discounted_adults discounted_children : ℕ)
  (h1 : adult_price = 25)
  (h2 : children_price = 15)
  (h3 : senior_price = 20)
  (h4 : discount = 5)
  (h5 : discounted_adults = 50)
  (h6 : discounted_children = 30)
  (h7 : total_receipts = 7200)
  (h8 : total_attendance = 400) :
  ∃ (regular_adults regular_children senior : ℕ),
    regular_adults + discounted_adults = 2 * senior ∧
    regular_adults + discounted_adults + regular_children + discounted_children + senior = total_attendance ∧
    regular_adults * adult_price + discounted_adults * (adult_price - discount) +
    regular_children * children_price + discounted_children * (children_price - discount) +
    senior * senior_price = total_receipts ∧
    regular_adults = 102 ∧
    regular_children = 142 ∧
    senior = 76 := by
  sorry

end NUMINAMATH_CALUDE_ticket_sales_l2114_211415


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_implies_c_less_than_one_l2114_211432

theorem quadratic_distinct_roots_implies_c_less_than_one :
  ∀ c : ℝ, (∃ x y : ℝ, x ≠ y ∧ x^2 - 2*x + c = 0 ∧ y^2 - 2*y + c = 0) → c < 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_implies_c_less_than_one_l2114_211432


namespace NUMINAMATH_CALUDE_actual_average_height_l2114_211482

/-- Calculates the actual average height of students given initial incorrect data and correction --/
theorem actual_average_height
  (num_students : ℕ)
  (initial_average : ℝ)
  (incorrect_height : ℝ)
  (actual_height : ℝ)
  (h_num_students : num_students = 20)
  (h_initial_average : initial_average = 175)
  (h_incorrect_height : incorrect_height = 151)
  (h_actual_height : actual_height = 136) :
  (num_students * initial_average - (incorrect_height - actual_height)) / num_students = 174.25 := by
  sorry

end NUMINAMATH_CALUDE_actual_average_height_l2114_211482


namespace NUMINAMATH_CALUDE_rocky_training_ratio_l2114_211461

/-- Rocky's training schedule over three days -/
structure TrainingSchedule where
  day1 : ℕ
  day2 : ℕ
  day3 : ℕ

/-- Conditions for Rocky's training -/
def validSchedule (s : TrainingSchedule) : Prop :=
  s.day1 = 4 ∧ 
  s.day2 = 2 * s.day1 ∧ 
  s.day3 > s.day2 ∧
  s.day1 + s.day2 + s.day3 = 36

/-- The ratio of miles run on day 3 to day 2 is 3 -/
theorem rocky_training_ratio (s : TrainingSchedule) 
  (h : validSchedule s) : s.day3 / s.day2 = 3 := by
  sorry


end NUMINAMATH_CALUDE_rocky_training_ratio_l2114_211461


namespace NUMINAMATH_CALUDE_heros_formula_triangle_area_l2114_211475

theorem heros_formula_triangle_area :
  let a : ℝ := 2
  let b : ℝ := 3
  let c : ℝ := 4
  let p : ℝ := (a + b + c) / 2
  let S : ℝ := Real.sqrt (p * (p - a) * (p - b) * (p - c))
  S = (3 / 4) * Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_heros_formula_triangle_area_l2114_211475


namespace NUMINAMATH_CALUDE_line_y_coordinate_at_x_10_l2114_211467

/-- Given a line passing through points (4, 0) and (-8, -6), 
    prove that the y-coordinate of the point on this line with x-coordinate 10 is 3. -/
theorem line_y_coordinate_at_x_10 :
  let m : ℚ := (0 - (-6)) / (4 - (-8))  -- Slope of the line
  let b : ℚ := 0 - m * 4                -- y-intercept of the line
  m * 10 + b = 3 := by sorry

end NUMINAMATH_CALUDE_line_y_coordinate_at_x_10_l2114_211467


namespace NUMINAMATH_CALUDE_value_of_a_minus_2b_l2114_211478

theorem value_of_a_minus_2b (a b : ℝ) (h : |a + b + 2| + |b - 3| = 0) : a - 2*b = -11 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_minus_2b_l2114_211478


namespace NUMINAMATH_CALUDE_x_squared_range_l2114_211488

theorem x_squared_range (x : ℝ) : 
  (Real.rpow (x + 9) (1/3) - Real.rpow (x - 9) (1/3) = 3) → 
  75 < x^2 ∧ x^2 < 85 := by
sorry

end NUMINAMATH_CALUDE_x_squared_range_l2114_211488


namespace NUMINAMATH_CALUDE_union_when_m_is_neg_half_subset_iff_m_geq_zero_l2114_211485

-- Define sets A and B
def A : Set ℝ := {x | x^2 + x - 2 < 0}
def B (m : ℝ) : Set ℝ := {x | 2*m < x ∧ x < 1 - m}

-- Theorem 1: When m = -1/2, A ∪ B = {x | -2 < x < 3/2}
theorem union_when_m_is_neg_half :
  A ∪ B (-1/2) = {x : ℝ | -2 < x ∧ x < 3/2} := by sorry

-- Theorem 2: B ⊆ A if and only if m ≥ 0
theorem subset_iff_m_geq_zero :
  ∀ m : ℝ, B m ⊆ A ↔ m ≥ 0 := by sorry

end NUMINAMATH_CALUDE_union_when_m_is_neg_half_subset_iff_m_geq_zero_l2114_211485


namespace NUMINAMATH_CALUDE_rhombus_rectangle_diagonals_bisect_l2114_211472

-- Define a quadrilateral
class Quadrilateral :=
  (diagonals_bisect : Bool)
  (diagonals_perpendicular : Bool)
  (diagonals_equal : Bool)

-- Define a rhombus
def Rhombus : Quadrilateral :=
{ diagonals_bisect := true,
  diagonals_perpendicular := true,
  diagonals_equal := false }

-- Define a rectangle
def Rectangle : Quadrilateral :=
{ diagonals_bisect := true,
  diagonals_perpendicular := false,
  diagonals_equal := true }

-- Theorem: Both rhombuses and rectangles have diagonals that bisect each other
theorem rhombus_rectangle_diagonals_bisect :
  Rhombus.diagonals_bisect ∧ Rectangle.diagonals_bisect :=
sorry

end NUMINAMATH_CALUDE_rhombus_rectangle_diagonals_bisect_l2114_211472


namespace NUMINAMATH_CALUDE_juanitas_dessert_cost_l2114_211438

/-- Represents the cost of a brownie dessert with various toppings -/
def brownieDessertCost (brownieCost iceCreamCost syrupCost nutsCost : ℚ)
  (iceCreamScoops syrupServings : ℕ) (includeNuts : Bool) : ℚ :=
  brownieCost +
  iceCreamCost * iceCreamScoops +
  syrupCost * syrupServings +
  (if includeNuts then nutsCost else 0)

/-- Proves that Juanita's dessert costs $7.00 given the prices and her order -/
theorem juanitas_dessert_cost :
  let brownieCost : ℚ := 5/2
  let iceCreamCost : ℚ := 1
  let syrupCost : ℚ := 1/2
  let nutsCost : ℚ := 3/2
  let iceCreamScoops : ℕ := 2
  let syrupServings : ℕ := 2
  let includeNuts : Bool := true
  brownieDessertCost brownieCost iceCreamCost syrupCost nutsCost
    iceCreamScoops syrupServings includeNuts = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_juanitas_dessert_cost_l2114_211438


namespace NUMINAMATH_CALUDE_smallest_four_digit_2_mod_5_l2114_211405

theorem smallest_four_digit_2_mod_5 : 
  ∀ n : ℕ, n ≥ 1000 ∧ n % 5 = 2 → n ≥ 1002 :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_2_mod_5_l2114_211405


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2114_211411

theorem complex_equation_solution :
  ∀ z : ℂ, (z - Complex.I) * (2 - Complex.I) = 5 → z = 2 + 2 * Complex.I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2114_211411


namespace NUMINAMATH_CALUDE_no_digit_product_6552_l2114_211473

theorem no_digit_product_6552 : ¬ ∃ (s : List ℕ), (∀ n ∈ s, n ≤ 9) ∧ s.prod = 6552 := by
  sorry

end NUMINAMATH_CALUDE_no_digit_product_6552_l2114_211473


namespace NUMINAMATH_CALUDE_young_in_sample_is_seven_l2114_211404

/-- Represents the number of employees in each age group and the sample size --/
structure EmployeeData where
  total : ℕ
  young : ℕ
  middleAged : ℕ
  elderly : ℕ
  sampleSize : ℕ

/-- Calculates the number of young employees in a stratified sample --/
def youngInSample (data : EmployeeData) : ℕ :=
  (data.young * data.sampleSize) / data.total

/-- Theorem stating that for the given employee data, the number of young employees in the sample is 7 --/
theorem young_in_sample_is_seven (data : EmployeeData)
  (h1 : data.total = 750)
  (h2 : data.young = 350)
  (h3 : data.middleAged = 250)
  (h4 : data.elderly = 150)
  (h5 : data.sampleSize = 15) :
  youngInSample data = 7 := by
  sorry


end NUMINAMATH_CALUDE_young_in_sample_is_seven_l2114_211404


namespace NUMINAMATH_CALUDE_slope_angle_30_implies_m_equals_neg_sqrt3_l2114_211497

/-- Given a line with equation x + my - 2 = 0 and slope angle 30°, m equals -√3 --/
theorem slope_angle_30_implies_m_equals_neg_sqrt3 (m : ℝ) : 
  (∃ x y, x + m * y - 2 = 0) →  -- Line equation
  (Real.tan (30 * π / 180) = -1 / m) →  -- Slope angle is 30°
  m = -Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_slope_angle_30_implies_m_equals_neg_sqrt3_l2114_211497


namespace NUMINAMATH_CALUDE_find_a_value_l2114_211464

theorem find_a_value (x y : ℝ) (a : ℝ) 
  (h1 : x = 2)
  (h2 : y = 1)
  (h3 : a * x - y = 3) :
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_find_a_value_l2114_211464


namespace NUMINAMATH_CALUDE_second_visit_date_l2114_211416

/-- Represents the bill amount for a single person --/
structure Bill :=
  (base : ℕ)
  (date : ℕ)

/-- The restaurant scenario --/
structure RestaurantScenario :=
  (first_visit : Bill)
  (second_visit : Bill)
  (num_friends : ℕ)
  (days_between : ℕ)

/-- The conditions of the problem --/
def problem_conditions (scenario : RestaurantScenario) : Prop :=
  scenario.num_friends = 3 ∧
  scenario.days_between = 4 ∧
  scenario.first_visit.base + scenario.first_visit.date = 168 ∧
  scenario.num_friends * scenario.second_visit.base + scenario.second_visit.date = 486 ∧
  scenario.first_visit.base = scenario.second_visit.base ∧
  scenario.second_visit.date = scenario.first_visit.date + scenario.days_between

/-- The theorem to prove --/
theorem second_visit_date (scenario : RestaurantScenario) :
  problem_conditions scenario → scenario.second_visit.date = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_second_visit_date_l2114_211416


namespace NUMINAMATH_CALUDE_curve_properties_l2114_211414

-- Define the function f(x) = x^3 - x
def f (x : ℝ) : ℝ := x^3 - x

-- State the theorem
theorem curve_properties :
  -- Part I: f'(1) = 2
  (deriv f) 1 = 2 ∧
  -- Part II: Tangent line equation at P(1, f(1)) is 2x - y - 2 = 0
  (∃ (m b : ℝ), m = 2 ∧ b = -2 ∧ ∀ x y, y = m * (x - 1) + f 1 ↔ 2 * x - y - 2 = 0) ∧
  -- Part III: Extreme values
  (∃ (x1 x2 : ℝ), 
    x1 = -Real.sqrt 3 / 3 ∧ 
    x2 = Real.sqrt 3 / 3 ∧
    f x1 = -2 * Real.sqrt 3 / 9 ∧
    f x2 = -2 * Real.sqrt 3 / 9 ∧
    (∀ x, f x ≥ -2 * Real.sqrt 3 / 9) ∧
    (∀ x, (deriv f) x = 0 → x = x1 ∨ x = x2)) :=
by sorry

end NUMINAMATH_CALUDE_curve_properties_l2114_211414


namespace NUMINAMATH_CALUDE_february_average_rainfall_l2114_211418

-- Define the given conditions
def total_rainfall : ℝ := 280
def days_in_february : ℕ := 28
def hours_per_day : ℕ := 24

-- Define the theorem
theorem february_average_rainfall :
  total_rainfall / (days_in_february * hours_per_day : ℝ) = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_february_average_rainfall_l2114_211418


namespace NUMINAMATH_CALUDE_angle_measure_l2114_211458

theorem angle_measure (x : ℝ) : 
  (180 - x = 3 * x - 10) → x = 47.5 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_l2114_211458


namespace NUMINAMATH_CALUDE_f_increasing_and_range_l2114_211419

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a - 1 / (2^x + 1)

theorem f_increasing_and_range :
  (∀ a : ℝ, Monotone (f a)) ∧
  (∀ a : ℝ, (∀ x : ℝ, f a x = -(f a (-x))) →
    Set.range (f a) = Set.Ioo (-1/2 : ℝ) (1/2 : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_f_increasing_and_range_l2114_211419


namespace NUMINAMATH_CALUDE_blue_face_probability_l2114_211454

structure Octahedron :=
  (total_faces : ℕ)
  (blue_faces : ℕ)
  (red_faces : ℕ)
  (green_faces : ℕ)
  (face_sum : blue_faces + red_faces + green_faces = total_faces)

def roll_probability (o : Octahedron) : ℚ :=
  o.blue_faces / o.total_faces

theorem blue_face_probability (o : Octahedron) 
  (h1 : o.total_faces = 8)
  (h2 : o.blue_faces = 4)
  (h3 : o.red_faces = 3)
  (h4 : o.green_faces = 1) :
  roll_probability o = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_blue_face_probability_l2114_211454


namespace NUMINAMATH_CALUDE_hexagon_triangle_angle_sum_l2114_211436

theorem hexagon_triangle_angle_sum : ∀ (P Q R s t : ℝ),
  P = 40 ∧ Q = 88 ∧ R = 30 →
  (720 : ℝ) = P + Q + R + (120 - t) + (130 - s) + s + t →
  s + t = 312 := by
sorry

end NUMINAMATH_CALUDE_hexagon_triangle_angle_sum_l2114_211436


namespace NUMINAMATH_CALUDE_sum_of_powers_equals_reciprocal_l2114_211420

theorem sum_of_powers_equals_reciprocal (m : ℕ) (h_m_odd : Odd m) (h_m_gt_1 : m > 1) :
  let n := 2 * m
  let θ := Complex.exp (2 * Real.pi * Complex.I / n)
  (Finset.sum (Finset.range ((m - 1) / 2)) (fun i => θ^(2 * i + 1))) = 1 / (1 - θ) := by
sorry

end NUMINAMATH_CALUDE_sum_of_powers_equals_reciprocal_l2114_211420


namespace NUMINAMATH_CALUDE_spheres_radius_is_half_l2114_211440

/-- A cube with side length 2 containing eight congruent spheres --/
structure SpheresInCube where
  -- The side length of the cube
  cube_side : ℝ
  -- The radius of each sphere
  sphere_radius : ℝ
  -- The number of spheres
  num_spheres : ℕ
  -- Condition that the cube side length is 2
  cube_side_is_two : cube_side = 2
  -- Condition that there are 8 spheres
  eight_spheres : num_spheres = 8
  -- Condition that spheres are tangent to three faces and neighboring spheres
  spheres_tangent : True  -- This is a simplification, as we can't easily express this geometric condition

/-- Theorem stating that the radius of each sphere is 1/2 --/
theorem spheres_radius_is_half (s : SpheresInCube) : s.sphere_radius = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_spheres_radius_is_half_l2114_211440


namespace NUMINAMATH_CALUDE_lcm_sum_bound_l2114_211484

theorem lcm_sum_bound (a b c d e : ℕ) (h1 : a > b) (h2 : b > c) (h3 : c > d) (h4 : d > e) (h5 : e > 1) :
  (1 : ℚ) / Nat.lcm a b + (1 : ℚ) / Nat.lcm b c + (1 : ℚ) / Nat.lcm c d + (1 : ℚ) / Nat.lcm d e ≤ 15 / 16 := by
  sorry

end NUMINAMATH_CALUDE_lcm_sum_bound_l2114_211484


namespace NUMINAMATH_CALUDE_pairwise_coprime_fraction_squares_l2114_211427

theorem pairwise_coprime_fraction_squares (x y z : ℕ+) 
  (h_coprime_xy : Nat.Coprime x.val y.val)
  (h_coprime_yz : Nat.Coprime y.val z.val)
  (h_coprime_xz : Nat.Coprime x.val z.val)
  (h_eq : (1 : ℚ) / x.val + (1 : ℚ) / y.val = (1 : ℚ) / z.val) :
  ∃ (a b c : ℕ), 
    (x.val + y.val = a ^ 2) ∧ 
    (x.val - z.val = b ^ 2) ∧ 
    (y.val - z.val = c ^ 2) := by
  sorry

end NUMINAMATH_CALUDE_pairwise_coprime_fraction_squares_l2114_211427


namespace NUMINAMATH_CALUDE_sandras_sock_purchase_l2114_211444

/-- Represents the number of pairs of socks at each price point --/
structure SockPurchase where
  two_dollar : ℕ
  three_dollar : ℕ
  five_dollar : ℕ

/-- Checks if the purchase satisfies the given conditions --/
def is_valid_purchase (p : SockPurchase) : Prop :=
  p.two_dollar + p.three_dollar + p.five_dollar = 15 ∧
  2 * p.two_dollar + 3 * p.three_dollar + 5 * p.five_dollar = 36 ∧
  p.two_dollar ≤ 6 ∧ p.three_dollar ≤ 6 ∧ p.five_dollar ≤ 6

/-- Theorem stating that the only valid purchase has 11 pairs of $2 socks --/
theorem sandras_sock_purchase :
  ∀ p : SockPurchase, is_valid_purchase p → p.two_dollar = 11 := by
  sorry

end NUMINAMATH_CALUDE_sandras_sock_purchase_l2114_211444


namespace NUMINAMATH_CALUDE_cube_string_length_l2114_211428

theorem cube_string_length (volume : ℝ) (edge_length : ℝ) (string_length : ℝ) : 
  volume = 3375 → 
  edge_length ^ 3 = volume →
  string_length = 12 * edge_length →
  string_length = 180 := by sorry

end NUMINAMATH_CALUDE_cube_string_length_l2114_211428


namespace NUMINAMATH_CALUDE_johnny_savings_l2114_211437

theorem johnny_savings (september : ℕ) (october : ℕ) (spent : ℕ) (left : ℕ) :
  september = 30 →
  october = 49 →
  spent = 58 →
  left = 67 →
  ∃ november : ℕ, november = 46 ∧ september + october + november - spent = left :=
by sorry

end NUMINAMATH_CALUDE_johnny_savings_l2114_211437


namespace NUMINAMATH_CALUDE_ball_transfer_equality_l2114_211462

/-- Represents a box containing balls of different colors -/
structure Box where
  black : ℕ
  white : ℕ

/-- Transfers balls between boxes -/
def transfer (a b : Box) (n : ℕ) : Box × Box :=
  let blackToB := min n a.black
  let whiteToA := min (n - blackToB) b.white
  let blackToA := n - whiteToA
  ({ black := a.black - blackToB + blackToA,
     white := a.white + whiteToA },
   { black := b.black + blackToB - blackToA,
     white := b.white - whiteToA })

theorem ball_transfer_equality (a b : Box) (n : ℕ) :
  let (a', b') := transfer a b n
  a'.white = b'.black := by sorry

end NUMINAMATH_CALUDE_ball_transfer_equality_l2114_211462


namespace NUMINAMATH_CALUDE_competition_sequences_count_l2114_211407

/-- The number of possible competition sequences for two teams with 7 members each -/
def competition_sequences : ℕ :=
  Nat.choose 14 7

/-- Theorem stating that the number of competition sequences is 3432 -/
theorem competition_sequences_count : competition_sequences = 3432 := by
  sorry

end NUMINAMATH_CALUDE_competition_sequences_count_l2114_211407


namespace NUMINAMATH_CALUDE_ellipse_tangent_inequality_l2114_211457

/-- Represents an ellipse with foci A and B, and semi-major and semi-minor axes a and b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  A : ℝ × ℝ
  B : ℝ × ℝ
  h_a_pos : 0 < a
  h_b_pos : 0 < b
  h_a_ge_b : a ≥ b

/-- Represents a point in 2D space -/
def Point := ℝ × ℝ

/-- Checks if a point is outside an ellipse -/
def is_outside (ε : Ellipse) (T : Point) : Prop := sorry

/-- Represents a tangent line from a point to an ellipse -/
def tangent_line (ε : Ellipse) (T : Point) : Type := sorry

/-- The length of a tangent line -/
def tangent_length (l : tangent_line ε T) : ℝ := sorry

theorem ellipse_tangent_inequality (ε : Ellipse) (T : Point) 
  (h_outside : is_outside ε T) 
  (TP TQ : tangent_line ε T) : 
  (tangent_length TP) / (tangent_length TQ) ≥ ε.b / ε.a := by sorry

end NUMINAMATH_CALUDE_ellipse_tangent_inequality_l2114_211457


namespace NUMINAMATH_CALUDE_range_of_a_l2114_211480

theorem range_of_a (a : ℝ) : 
  let M : Set ℝ := {a}
  let P : Set ℝ := {x | -1 < x ∧ x < 1}
  M ⊆ P → a ∈ P := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l2114_211480


namespace NUMINAMATH_CALUDE_mary_books_checked_out_l2114_211421

/-- Calculates the number of books Mary has checked out after a series of transactions --/
def books_checked_out (initial : ℕ) 
  (return1 checkout1 : ℕ) 
  (return2 checkout2 : ℕ) 
  (return3 checkout3 : ℕ) 
  (return4 checkout4 : ℕ) : ℕ :=
  initial - return1 + checkout1 - return2 + checkout2 - return3 + checkout3 - return4 + checkout4

/-- Proves that Mary has 22 books checked out given the problem conditions --/
theorem mary_books_checked_out : 
  books_checked_out 10 5 6 3 4 2 9 5 8 = 22 := by
  sorry

end NUMINAMATH_CALUDE_mary_books_checked_out_l2114_211421


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2114_211489

-- Define the quadratic function
def f (a b x : ℝ) := a * x^2 - x - b

-- Define the solution set condition
def solution_set_condition (a b : ℝ) :=
  ∀ x, f a b x > 0 ↔ (x > 2 ∨ x < -1)

-- Theorem statement
theorem quadratic_inequality_solution :
  ∀ a b : ℝ, solution_set_condition a b →
    (a = 1 ∧ b = 2) ∧
    (∀ c : ℝ,
      (c > 1 → ∀ x, x^2 - (c+1)*x + c < 0 ↔ 1 < x ∧ x < c) ∧
      (c = 1 → ∀ x, ¬(x^2 - (c+1)*x + c < 0)) ∧
      (c < 1 → ∀ x, x^2 - (c+1)*x + c < 0 ↔ c < x ∧ x < 1)) :=
by sorry


end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2114_211489


namespace NUMINAMATH_CALUDE_total_pizzas_is_fifteen_l2114_211449

/-- The number of pizzas served during lunch -/
def lunch_pizzas : ℕ := 9

/-- The number of pizzas served during dinner -/
def dinner_pizzas : ℕ := 6

/-- The total number of pizzas served today -/
def total_pizzas : ℕ := lunch_pizzas + dinner_pizzas

/-- Theorem: The total number of pizzas served today is 15 -/
theorem total_pizzas_is_fifteen : total_pizzas = 15 := by sorry

end NUMINAMATH_CALUDE_total_pizzas_is_fifteen_l2114_211449


namespace NUMINAMATH_CALUDE_phi_value_l2114_211476

/-- Given a function f and constants ω and φ, proves that φ = π/6 under certain conditions. -/
theorem phi_value (f g : ℝ → ℝ) (ω φ : ℝ) : 
  (∀ x, f x = 2 * Real.sin (ω * x + φ)) →
  ω > 0 →
  |φ| < π / 2 →
  (∀ x, f (x + π) = f x) →
  (∀ x, f (x - π / 6) = g x) →
  (∀ x, g (x + π / 3) = g (π / 3 - x)) →
  φ = π / 6 := by
sorry


end NUMINAMATH_CALUDE_phi_value_l2114_211476


namespace NUMINAMATH_CALUDE_unbounded_sequence_l2114_211496

def is_strictly_increasing (a : ℕ → ℝ) : Prop :=
  ∀ n, a n < a (n + 1)

def sequence_property (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 2) = (a (n + 1) - a n) ^ (Real.sqrt n) + n ^ (-(Real.sqrt n))

theorem unbounded_sequence
  (a : ℕ → ℝ)
  (h_pos : ∀ n, 0 < a n)
  (h_incr : is_strictly_increasing a)
  (h_prop : sequence_property a) :
  ∀ C, ∃ m, C < a m :=
sorry

end NUMINAMATH_CALUDE_unbounded_sequence_l2114_211496


namespace NUMINAMATH_CALUDE_garage_sale_pricing_l2114_211459

theorem garage_sale_pricing (total_items : ℕ) (highest_rank : ℕ) (lowest_rank : ℕ) 
  (h1 : total_items = 36)
  (h2 : highest_rank = 15)
  (h3 : lowest_rank + highest_rank = total_items + 1) : 
  lowest_rank = 22 := by
sorry

end NUMINAMATH_CALUDE_garage_sale_pricing_l2114_211459


namespace NUMINAMATH_CALUDE_min_volume_to_prevent_explosion_l2114_211424

/-- Represents the relationship between pressure and volume for a balloon -/
structure Balloon where
  k : ℝ
  pressure : ℝ → ℝ
  volume : ℝ → ℝ
  h1 : ∀ v, pressure v = k / v
  h2 : pressure 3 = 8000
  h3 : ∀ v, pressure v > 40000 → volume v < v

/-- The minimum volume to prevent the balloon from exploding is 0.6 m³ -/
theorem min_volume_to_prevent_explosion (b : Balloon) : 
  ∀ v, v ≥ 0.6 → b.pressure v ≤ 40000 :=
sorry

#check min_volume_to_prevent_explosion

end NUMINAMATH_CALUDE_min_volume_to_prevent_explosion_l2114_211424


namespace NUMINAMATH_CALUDE_total_money_after_redistribution_l2114_211410

/-- Represents the money redistribution problem among three friends --/
def money_redistribution (a j t : ℕ) : Prop :=
  let a1 := a - 2*(t + j)
  let j1 := 3*j
  let t1 := 3*t
  let a2 := 2*a1
  let j2 := j1 - (a1 + t1)
  let t2 := 2*t1
  let a3 := 2*a2
  let j3 := 2*j2
  let t3 := t2 - (a2 + j2)
  (t = 48) ∧ (t3 = 48) ∧ (a3 + j3 + t3 = 528)

/-- Theorem stating the total amount of money after redistribution --/
theorem total_money_after_redistribution :
  ∃ (a j : ℕ), money_redistribution a j 48 :=
sorry

end NUMINAMATH_CALUDE_total_money_after_redistribution_l2114_211410


namespace NUMINAMATH_CALUDE_edward_remaining_money_l2114_211403

/-- Given Edward's initial amount and the amount he spent, calculate how much he has left. -/
theorem edward_remaining_money (initial : ℕ) (spent : ℕ) (remaining : ℕ) 
  (h1 : initial = 19) 
  (h2 : spent = 13) 
  (h3 : remaining = initial - spent) : 
  remaining = 6 := by
  sorry

end NUMINAMATH_CALUDE_edward_remaining_money_l2114_211403


namespace NUMINAMATH_CALUDE_lauras_garden_tulips_l2114_211429

/-- Represents a garden with tulips and lilies -/
structure Garden where
  tulips : ℕ
  lilies : ℕ

/-- Calculates the number of tulips needed to maintain a 3:4 ratio with the given number of lilies -/
def tulipsForRatio (lilies : ℕ) : ℕ :=
  (3 * lilies) / 4

/-- Represents Laura's garden before and after adding flowers -/
def lauras_garden : Garden × Garden :=
  let initial := Garden.mk (tulipsForRatio 32) 32
  let final := Garden.mk (tulipsForRatio (32 + 24)) (32 + 24)
  (initial, final)

/-- Theorem stating that after adding 24 lilies and maintaining the 3:4 ratio, 
    Laura will have 42 tulips in total -/
theorem lauras_garden_tulips : 
  (lauras_garden.2).tulips = 42 := by sorry

end NUMINAMATH_CALUDE_lauras_garden_tulips_l2114_211429


namespace NUMINAMATH_CALUDE_congruence_problem_l2114_211483

theorem congruence_problem : ∃! n : ℕ, n ≤ 14 ∧ n ≡ 8657 [ZMOD 15] ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l2114_211483


namespace NUMINAMATH_CALUDE_joan_found_six_shells_l2114_211447

/-- The number of seashells Jessica found -/
def jessica_shells : ℕ := 8

/-- The total number of seashells Joan and Jessica found together -/
def total_shells : ℕ := 14

/-- The number of seashells Joan found -/
def joan_shells : ℕ := total_shells - jessica_shells

theorem joan_found_six_shells : joan_shells = 6 := by
  sorry

end NUMINAMATH_CALUDE_joan_found_six_shells_l2114_211447


namespace NUMINAMATH_CALUDE_triangle_inequality_l2114_211439

theorem triangle_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  |((a - b) / (a + b) + (b - c) / (b + c) + (c - a) / (c + a))| < 1 :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2114_211439


namespace NUMINAMATH_CALUDE_tetrahedron_inequality_l2114_211433

/-- Represents a tetrahedron -/
structure Tetrahedron where
  /-- The minimum distance between opposite edges -/
  d : ℝ
  /-- The length of the shortest height -/
  h : ℝ
  /-- d is positive -/
  d_pos : d > 0
  /-- h is positive -/
  h_pos : h > 0

/-- 
For any tetrahedron, twice the minimum distance between 
opposite edges is greater than the length of the shortest height
-/
theorem tetrahedron_inequality (t : Tetrahedron) : 2 * t.d > t.h := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_inequality_l2114_211433


namespace NUMINAMATH_CALUDE_square_divisibility_l2114_211456

theorem square_divisibility (n : ℤ) : (∃ k : ℤ, n^2 = 9*k) ∨ (∃ m : ℤ, n^2 = 3*m + 1) := by
  sorry

end NUMINAMATH_CALUDE_square_divisibility_l2114_211456


namespace NUMINAMATH_CALUDE_common_chord_equation_l2114_211402

/-- The equation of the first circle -/
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 8 = 0

/-- The equation of the second circle -/
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y - 4 = 0

/-- The equation of the line on which the common chord lies -/
def common_chord (x y : ℝ) : Prop := x - y + 1 = 0

/-- Theorem stating that the common chord of the two circles lies on the line x - y + 1 = 0 -/
theorem common_chord_equation :
  ∀ x y : ℝ, (circle1 x y ∧ circle2 x y) → common_chord x y :=
sorry

end NUMINAMATH_CALUDE_common_chord_equation_l2114_211402


namespace NUMINAMATH_CALUDE_max_diagonal_of_rectangle_l2114_211409

/-- The maximum diagonal of a rectangle with perimeter 40 --/
theorem max_diagonal_of_rectangle (l w : ℝ) : 
  l > 0 → w > 0 → l + w = 20 → 
  ∀ (x y : ℝ), x > 0 → y > 0 → x + y = 20 → 
  Real.sqrt (l^2 + w^2) ≤ 20 :=
by sorry

end NUMINAMATH_CALUDE_max_diagonal_of_rectangle_l2114_211409


namespace NUMINAMATH_CALUDE_max_value_complex_expression_l2114_211455

theorem max_value_complex_expression (z : ℂ) (h : Complex.abs z = 2) :
  Complex.abs ((z - 2)^2 * (z + 2)) ≤ 16 * Real.sqrt 2 ∧
  ∃ w : ℂ, Complex.abs w = 2 ∧ Complex.abs ((w - 2)^2 * (w + 2)) = 16 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_complex_expression_l2114_211455


namespace NUMINAMATH_CALUDE_walking_speed_problem_l2114_211434

/-- 
Given two people walking in opposite directions for 12 hours, 
with one walking at 3 km/hr and the distance between them after 12 hours being 120 km, 
prove that the speed of the other person is 7 km/hr.
-/
theorem walking_speed_problem (v : ℝ) : 
  v > 0 → -- Assuming positive speed
  (v + 3) * 12 = 120 → 
  v = 7 := by
sorry

end NUMINAMATH_CALUDE_walking_speed_problem_l2114_211434


namespace NUMINAMATH_CALUDE_cos_pi_sixth_eq_sin_shifted_l2114_211441

theorem cos_pi_sixth_eq_sin_shifted (x : ℝ) : 
  Real.cos x + π/6 = Real.sin (x + 2*π/3) := by sorry

end NUMINAMATH_CALUDE_cos_pi_sixth_eq_sin_shifted_l2114_211441


namespace NUMINAMATH_CALUDE_steve_take_home_pay_l2114_211448

/-- Calculates the take-home pay given annual salary and deductions --/
def takeHomePay (annualSalary : ℝ) (taxRate : ℝ) (healthcareRate : ℝ) (unionDues : ℝ) : ℝ :=
  annualSalary - (annualSalary * taxRate + annualSalary * healthcareRate + unionDues)

/-- Proves that Steve's take-home pay is $27,200 --/
theorem steve_take_home_pay :
  takeHomePay 40000 0.20 0.10 800 = 27200 := by
  sorry

end NUMINAMATH_CALUDE_steve_take_home_pay_l2114_211448


namespace NUMINAMATH_CALUDE_jacket_price_reduction_l2114_211479

theorem jacket_price_reduction (P : ℝ) (x : ℝ) : 
  P > 0 → 
  0 ≤ x → x ≤ 100 →
  P * (1 - x / 100) * (1 - 0.25) * (1 + 0.7778) = P →
  x = 25 := by
sorry

end NUMINAMATH_CALUDE_jacket_price_reduction_l2114_211479


namespace NUMINAMATH_CALUDE_multiply_and_add_l2114_211400

theorem multiply_and_add : 45 * 55 + 45 * 45 = 4500 := by
  sorry

end NUMINAMATH_CALUDE_multiply_and_add_l2114_211400


namespace NUMINAMATH_CALUDE_ivan_travel_theorem_l2114_211442

/-- Represents the travel scenario of Ivan Semenovich -/
structure TravelScenario where
  usual_travel_time : ℝ
  usual_arrival_time : ℝ
  late_departure : ℝ
  speed_increase : ℝ
  new_arrival_time : ℝ

/-- The theorem to be proved -/
theorem ivan_travel_theorem (scenario : TravelScenario) 
  (h1 : scenario.usual_arrival_time = 9 * 60)  -- 9:00 AM in minutes
  (h2 : scenario.late_departure = 40)
  (h3 : scenario.speed_increase = 0.6)
  (h4 : scenario.new_arrival_time = 8 * 60 + 35)  -- 8:35 AM in minutes
  : ∃ (optimal_increase : ℝ),
    optimal_increase = 0.3 ∧
    scenario.usual_arrival_time = 
      scenario.usual_travel_time * (1 - scenario.late_departure / scenario.usual_travel_time) / (1 + optimal_increase) + 
      scenario.late_departure :=
by sorry

end NUMINAMATH_CALUDE_ivan_travel_theorem_l2114_211442


namespace NUMINAMATH_CALUDE_population_growth_l2114_211452

theorem population_growth (p : ℕ) : 
  p > 0 →                           -- p is positive
  (p^2 + 121 = q^2 + 16) →          -- 2005 population condition
  (p^2 + 346 = r^2) →               -- 2015 population condition
  ∃ (growth : ℝ), 
    growth = ((p^2 + 346 - p^2) / p^2) * 100 ∧ 
    abs (growth - 111) < abs (growth - 100) ∧ 
    abs (growth - 111) < abs (growth - 105) ∧ 
    abs (growth - 111) < abs (growth - 110) ∧ 
    abs (growth - 111) < abs (growth - 115) :=
by sorry

end NUMINAMATH_CALUDE_population_growth_l2114_211452


namespace NUMINAMATH_CALUDE_correct_operation_l2114_211460

theorem correct_operation (a : ℝ) : 2 * a^2 * a = 2 * a^3 := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l2114_211460


namespace NUMINAMATH_CALUDE_exam_marks_calculation_l2114_211426

def math_passing_percentage : ℝ := 0.30
def science_passing_percentage : ℝ := 0.50
def english_passing_percentage : ℝ := 0.40

def math_marks_obtained : ℕ := 80
def math_marks_short : ℕ := 100

def science_marks_obtained : ℕ := 120
def science_marks_short : ℕ := 80

def english_marks_obtained : ℕ := 60
def english_marks_short : ℕ := 60

def math_max_marks : ℕ := 600
def science_max_marks : ℕ := 400
def english_max_marks : ℕ := 300

theorem exam_marks_calculation :
  (math_passing_percentage * math_max_marks : ℝ) = (math_marks_obtained + math_marks_short : ℝ) ∧
  (science_passing_percentage * science_max_marks : ℝ) = (science_marks_obtained + science_marks_short : ℝ) ∧
  (english_passing_percentage * english_max_marks : ℝ) = (english_marks_obtained + english_marks_short : ℝ) ∧
  math_max_marks + science_max_marks + english_max_marks = 1300 :=
by sorry

end NUMINAMATH_CALUDE_exam_marks_calculation_l2114_211426


namespace NUMINAMATH_CALUDE_trig_problem_l2114_211401

theorem trig_problem (x : ℝ) 
  (h1 : -π/2 < x ∧ x < 0) 
  (h2 : Real.sin x + Real.cos x = 1/5) : 
  (Real.sin x - Real.cos x = -7/5) ∧ 
  (4 * Real.sin x * Real.cos x - Real.cos x^2 = -64/25) := by
sorry

end NUMINAMATH_CALUDE_trig_problem_l2114_211401


namespace NUMINAMATH_CALUDE_decimal_difference_l2114_211445

-- Define the repeating decimal 0.3̄6
def repeating_decimal : ℚ := 4 / 11

-- Define the terminating decimal 0.36
def terminating_decimal : ℚ := 36 / 100

-- Theorem statement
theorem decimal_difference : 
  repeating_decimal - terminating_decimal = 4 / 1100 := by
  sorry

end NUMINAMATH_CALUDE_decimal_difference_l2114_211445


namespace NUMINAMATH_CALUDE_mikails_age_correct_l2114_211406

/-- Mikail's age on his birthday -/
def mikails_age : ℕ := 9

/-- Amount of money Mikail receives per year of age -/
def money_per_year : ℕ := 5

/-- Total amount of money Mikail receives on his birthday -/
def total_money : ℕ := 45

/-- Theorem: Mikail's age is correct given the money he receives -/
theorem mikails_age_correct : mikails_age = total_money / money_per_year := by
  sorry

end NUMINAMATH_CALUDE_mikails_age_correct_l2114_211406


namespace NUMINAMATH_CALUDE_AB_value_l2114_211425

-- Define the points
variable (A B C D E : ℝ × ℝ)

-- Define the conditions
axiom on_line : ∃ (y : ℝ), A.2 = y ∧ B.2 = y ∧ C.2 = y ∧ D.2 = y
axiom order : A.1 < B.1 ∧ B.1 < C.1 ∧ C.1 < D.1
axiom AB_eq_CD : dist A B = dist C D
axiom BC_eq_16 : dist B C = 16
axiom E_not_on_line : E.2 ≠ A.2
axiom BE_eq_CE : dist B E = dist C E
axiom BE_eq_13 : dist B E = 13

-- Define the perimeter function
def perimeter (X Y Z : ℝ × ℝ) : ℝ := dist X Y + dist Y Z + dist Z X

-- Define the main theorem
theorem AB_value : 
  perimeter A E D = 3 * perimeter B E C → dist A B = 34/3 :=
sorry

end NUMINAMATH_CALUDE_AB_value_l2114_211425


namespace NUMINAMATH_CALUDE_f_minus_one_indeterminate_l2114_211470

/-- A function f(x) with unknown coefficients a, b, and c. -/
noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ :=
  a * (Real.tan x)^3 - b * Real.sin (3 * x) + c * x + 7

/-- Theorem stating that f(-1) cannot be uniquely determined given only f(1) = 14. -/
theorem f_minus_one_indeterminate (a b c : ℝ) :
  f a b c 1 = 14 → ¬∃!y, f a b c (-1) = y :=
by
  sorry

#check f_minus_one_indeterminate

end NUMINAMATH_CALUDE_f_minus_one_indeterminate_l2114_211470


namespace NUMINAMATH_CALUDE_sqrt_x_minus_2_meaningful_l2114_211494

theorem sqrt_x_minus_2_meaningful (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 2) ↔ x ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_2_meaningful_l2114_211494


namespace NUMINAMATH_CALUDE_swim_club_prep_course_count_l2114_211408

/-- Represents a swim club with members, some of whom have passed a lifesaving test
    and some of whom have taken a preparatory course. -/
structure SwimClub where
  totalMembers : ℕ
  passedTest : ℕ
  notPassedNotTakenCourse : ℕ

/-- Calculates the number of members who have taken the preparatory course
    but not passed the test in a given swim club. -/
def membersInPreparatoryNotPassed (club : SwimClub) : ℕ :=
  club.totalMembers - club.passedTest - club.notPassedNotTakenCourse

/-- Theorem stating that in a swim club with 50 members, where 30% have passed
    the lifesaving test and 30 of those who haven't passed haven't taken the
    preparatory course, the number of members who have taken the preparatory
    course but not passed the test is 5. -/
theorem swim_club_prep_course_count :
  let club : SwimClub := {
    totalMembers := 50,
    passedTest := 15,  -- 30% of 50
    notPassedNotTakenCourse := 30
  }
  membersInPreparatoryNotPassed club = 5 := by
  sorry


end NUMINAMATH_CALUDE_swim_club_prep_course_count_l2114_211408


namespace NUMINAMATH_CALUDE_rectangular_to_polar_conversion_l2114_211423

theorem rectangular_to_polar_conversion :
  let x : ℝ := -1
  let y : ℝ := Real.sqrt 3
  let r : ℝ := Real.sqrt (x^2 + y^2)
  let θ : ℝ := 2 * Real.pi / 3
  r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧ r = 2 ∧ x = -r * Real.cos θ ∧ y = r * Real.sin θ := by
  sorry

end NUMINAMATH_CALUDE_rectangular_to_polar_conversion_l2114_211423


namespace NUMINAMATH_CALUDE_figure_colorings_l2114_211430

/-- Represents the number of ways to color a single equilateral triangle --/
def triangle_colorings : ℕ := 6

/-- Represents the number of ways to color each subsequent triangle --/
def subsequent_triangle_colorings : ℕ := 3

/-- Represents the number of ways to color the additional dot --/
def additional_dot_colorings : ℕ := 2

/-- The total number of dots in the figure --/
def total_dots : ℕ := 10

/-- The number of triangles in the figure --/
def num_triangles : ℕ := 3

theorem figure_colorings :
  triangle_colorings * subsequent_triangle_colorings ^ (num_triangles - 1) * additional_dot_colorings = 108 := by
  sorry

end NUMINAMATH_CALUDE_figure_colorings_l2114_211430


namespace NUMINAMATH_CALUDE_num_factors_of_m_l2114_211451

/-- The number of natural-number factors of m = 2^3 * 3^3 * 5^4 * 6^5 -/
def num_factors (m : ℕ) : ℕ := sorry

/-- m is defined as 2^3 * 3^3 * 5^4 * 6^5 -/
def m : ℕ := 2^3 * 3^3 * 5^4 * 6^5

theorem num_factors_of_m :
  num_factors m = 405 := by sorry

end NUMINAMATH_CALUDE_num_factors_of_m_l2114_211451


namespace NUMINAMATH_CALUDE_number_puzzle_solution_l2114_211498

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem number_puzzle_solution :
  ∃ (a b : ℕ),
    1 ≤ a ∧ a ≤ 60 ∧
    1 ≤ b ∧ b ≤ 60 ∧
    a ≠ b ∧
    ∀ k : ℕ, k < 5 → ¬((a + b) % k = 0) ∧
    is_prime b ∧
    b > 10 ∧
    ∃ (m : ℕ), 150 * b + a = m * m ∧
    a + b = 42 :=
by sorry

end NUMINAMATH_CALUDE_number_puzzle_solution_l2114_211498


namespace NUMINAMATH_CALUDE_max_value_of_function_l2114_211450

theorem max_value_of_function : 
  let f : ℝ → ℝ := λ x ↦ Real.sin (2 * x) - Real.cos (2 * x + π / 6)
  ∃ (M : ℝ), M = Real.sqrt 3 ∧ ∀ x, f x ≤ M ∧ ∃ x₀, f x₀ = M :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_function_l2114_211450


namespace NUMINAMATH_CALUDE_x_value_l2114_211417

-- Define the ⊕ operation
def oplus (a b : ℝ) : ℝ := 3 * a - b

-- State the theorem
theorem x_value : ∃ x : ℝ, oplus x (oplus 2 3) = 1 ∧ x = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l2114_211417


namespace NUMINAMATH_CALUDE_intersection_point_sum_l2114_211491

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- State the theorem
theorem intersection_point_sum :
  ∃ (a b : ℝ), f a = f (a - 4) ∧ a + b = 6 := by sorry

end NUMINAMATH_CALUDE_intersection_point_sum_l2114_211491


namespace NUMINAMATH_CALUDE_exactly_two_subsets_implies_a_values_l2114_211465

def A (a : ℝ) : Set ℝ := {x | a * x^2 + 2 * x + a = 0}

theorem exactly_two_subsets_implies_a_values (a : ℝ) :
  (∀ S : Set ℝ, S ⊆ A a → (S = ∅ ∨ S = A a)) →
  a = -1 ∨ a = 0 ∨ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_exactly_two_subsets_implies_a_values_l2114_211465


namespace NUMINAMATH_CALUDE_largest_prime_factor_is_57_l2114_211474

def numbers : List Nat := [57, 65, 91, 143, 169]

/-- The largest prime factor of a natural number -/
def largestPrimeFactor (n : Nat) : Nat :=
  (Nat.factors n).maximum?  -- Use Nat.factors to get prime factors
    |>.getD 1  -- Default to 1 if the list is empty

/-- Theorem: 57 has the largest prime factor among the given numbers -/
theorem largest_prime_factor_is_57 :
  ∀ n ∈ numbers, largestPrimeFactor 57 ≥ largestPrimeFactor n :=
by
  sorry

#eval largestPrimeFactor 57  -- Should output 19

end NUMINAMATH_CALUDE_largest_prime_factor_is_57_l2114_211474
