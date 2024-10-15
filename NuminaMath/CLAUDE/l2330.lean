import Mathlib

namespace NUMINAMATH_CALUDE_chord_length_concentric_circles_l2330_233054

/-- Given two concentric circles with radii a and b (a > b), 
    if the area of the ring between them is 12½π square inches,
    then the length of a chord of the larger circle tangent to the smaller circle is 5√2 inches. -/
theorem chord_length_concentric_circles (a b : ℝ) (h1 : a > b) 
  (h2 : π * a^2 - π * b^2 = 25/2 * π) : 
  ∃ (c : ℝ), c^2 = 50 ∧ c = (2 * a^2 - 2 * b^2).sqrt := by
  sorry

end NUMINAMATH_CALUDE_chord_length_concentric_circles_l2330_233054


namespace NUMINAMATH_CALUDE_complex_modulus_equality_l2330_233023

theorem complex_modulus_equality (n : ℝ) (hn : n > 0) :
  Complex.abs (5 + n * Complex.I) = 5 * Real.sqrt 13 → n = 10 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_equality_l2330_233023


namespace NUMINAMATH_CALUDE_special_quadrilateral_is_square_special_quadrilateral_is_not_always_square_l2330_233072

/-- A quadrilateral with perpendicular and equal diagonals --/
structure SpecialQuadrilateral where
  /-- The diagonals are perpendicular --/
  diagonals_perpendicular : Bool
  /-- The diagonals are equal in length --/
  diagonals_equal : Bool

/-- Definition of a square --/
def is_square (q : SpecialQuadrilateral) : Prop :=
  q.diagonals_perpendicular ∧ q.diagonals_equal

/-- The statement to be proven false --/
theorem special_quadrilateral_is_square (q : SpecialQuadrilateral) :
  q.diagonals_perpendicular ∧ q.diagonals_equal → is_square q :=
by
  sorry

/-- The theorem stating that the above statement is false --/
theorem special_quadrilateral_is_not_always_square :
  ¬ (∀ q : SpecialQuadrilateral, q.diagonals_perpendicular ∧ q.diagonals_equal → is_square q) :=
by
  sorry

end NUMINAMATH_CALUDE_special_quadrilateral_is_square_special_quadrilateral_is_not_always_square_l2330_233072


namespace NUMINAMATH_CALUDE_parking_space_area_l2330_233012

theorem parking_space_area (l w : ℝ) (h1 : l = 9) (h2 : 2 * w + l = 37) : l * w = 126 := by
  sorry

end NUMINAMATH_CALUDE_parking_space_area_l2330_233012


namespace NUMINAMATH_CALUDE_total_cost_calculation_l2330_233033

def rental_cost : ℝ := 150
def gas_needed : ℝ := 8
def gas_price : ℝ := 3.50
def mileage_expense : ℝ := 0.50
def distance_driven : ℝ := 320

theorem total_cost_calculation :
  rental_cost + gas_needed * gas_price + distance_driven * mileage_expense = 338 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_calculation_l2330_233033


namespace NUMINAMATH_CALUDE_pie_chart_central_angle_l2330_233003

theorem pie_chart_central_angle 
  (total_data : ℕ) 
  (group_frequency : ℕ) 
  (h1 : total_data = 60) 
  (h2 : group_frequency = 15) : 
  (group_frequency : ℝ) / (total_data : ℝ) * 360 = 90 := by
sorry

end NUMINAMATH_CALUDE_pie_chart_central_angle_l2330_233003


namespace NUMINAMATH_CALUDE_not_in_range_iff_b_in_interval_l2330_233051

-- Define the function f
def f (b x : ℝ) : ℝ := x^2 + b*x + 2

-- Theorem statement
theorem not_in_range_iff_b_in_interval (b : ℝ) :
  (∀ x : ℝ, f b x ≠ -2) ↔ b ∈ Set.Ioo (-4 : ℝ) 4 := by
  sorry

end NUMINAMATH_CALUDE_not_in_range_iff_b_in_interval_l2330_233051


namespace NUMINAMATH_CALUDE_employee_salary_problem_l2330_233070

/-- Proves that given the conditions of the problem, employee N's salary is $265 per week -/
theorem employee_salary_problem (total_salary m_salary n_salary : ℝ) : 
  total_salary = 583 →
  m_salary = 1.2 * n_salary →
  total_salary = m_salary + n_salary →
  n_salary = 265 := by
  sorry

end NUMINAMATH_CALUDE_employee_salary_problem_l2330_233070


namespace NUMINAMATH_CALUDE_planes_parallel_condition_l2330_233019

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (parallel_planes : Plane → Plane → Prop)

-- State the theorem
theorem planes_parallel_condition 
  (m n : Line) (α β : Plane) 
  (h1 : m ≠ n) (h2 : α ≠ β)
  (h3 : perpendicular m α) 
  (h4 : perpendicular n β) 
  (h5 : parallel_lines m n) : 
  parallel_planes α β :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_condition_l2330_233019


namespace NUMINAMATH_CALUDE_sphere_circle_paint_equivalence_l2330_233056

theorem sphere_circle_paint_equivalence (r_sphere r_circle : ℝ) : 
  r_sphere = 3 → 
  4 * π * r_sphere^2 = π * r_circle^2 → 
  r_circle = 6 := by
  sorry

end NUMINAMATH_CALUDE_sphere_circle_paint_equivalence_l2330_233056


namespace NUMINAMATH_CALUDE_fraction_equality_l2330_233094

theorem fraction_equality (a b c d e f : ℝ) 
  (h1 : a / b = 1 / 2) 
  (h2 : c / d = 1 / 2) 
  (h3 : e / f = 1 / 2) 
  (h4 : 3 * b - 2 * d + f ≠ 0) : 
  (3 * a - 2 * c + e) / (3 * b - 2 * d + f) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2330_233094


namespace NUMINAMATH_CALUDE_sqrt_product_simplification_l2330_233086

theorem sqrt_product_simplification (x : ℝ) (hx : x ≥ 0) :
  Real.sqrt (45 * x^2) * Real.sqrt (8 * x^3) * Real.sqrt (22 * x) = 60 * x^3 * Real.sqrt 55 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_simplification_l2330_233086


namespace NUMINAMATH_CALUDE_right_triangle_cone_volume_l2330_233026

theorem right_triangle_cone_volume (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (1 / 3 : ℝ) * π * y^2 * x = 1500 * π ∧
  (1 / 3 : ℝ) * π * x^2 * y = 540 * π →
  Real.sqrt (x^2 + y^2) = 5 * Real.sqrt 34 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_cone_volume_l2330_233026


namespace NUMINAMATH_CALUDE_rectangle_area_l2330_233042

/-- Given a rectangle with a length to width ratio of 0.875 and a width of 24 centimeters,
    its area is 504 square centimeters. -/
theorem rectangle_area (ratio : ℝ) (width : ℝ) (h1 : ratio = 0.875) (h2 : width = 24) :
  ratio * width * width = 504 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l2330_233042


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l2330_233079

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space using the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point lies on a line -/
def Point.liesOn (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are parallel -/
def Line.isParallelTo (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- The main theorem -/
theorem parallel_line_through_point (A B C : Point) : 
  let AC : Line := { a := C.y - A.y, b := A.x - C.x, c := A.y * C.x - A.x * C.y }
  let L : Line := { a := 1, b := -2, c := -7 }
  A.x = 5 ∧ A.y = 2 ∧ B.x = -1 ∧ B.y = -4 ∧ C.x = -5 ∧ C.y = -3 →
  B.liesOn L ∧ L.isParallelTo AC := by
  sorry


end NUMINAMATH_CALUDE_parallel_line_through_point_l2330_233079


namespace NUMINAMATH_CALUDE_remainder_3123_div_28_l2330_233011

theorem remainder_3123_div_28 : 3123 % 28 = 15 := by
  sorry

end NUMINAMATH_CALUDE_remainder_3123_div_28_l2330_233011


namespace NUMINAMATH_CALUDE_bears_in_stock_calculation_l2330_233073

/-- Calculates the number of bears in stock before a new shipment arrived -/
def bears_in_stock_before_shipment (new_shipment : ℕ) (bears_per_shelf : ℕ) (num_shelves : ℕ) : ℕ :=
  num_shelves * bears_per_shelf - new_shipment

theorem bears_in_stock_calculation (new_shipment : ℕ) (bears_per_shelf : ℕ) (num_shelves : ℕ) :
  bears_in_stock_before_shipment new_shipment bears_per_shelf num_shelves =
  num_shelves * bears_per_shelf - new_shipment :=
by
  sorry

end NUMINAMATH_CALUDE_bears_in_stock_calculation_l2330_233073


namespace NUMINAMATH_CALUDE_inequality_implies_a_zero_l2330_233099

theorem inequality_implies_a_zero (a : ℝ) :
  (∀ x : ℝ, a * (Real.sin x)^2 + Real.cos x ≥ a^2 - 1) → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implies_a_zero_l2330_233099


namespace NUMINAMATH_CALUDE_no_valid_assignment_l2330_233043

/-- Represents a vertex of the hexagon or its center -/
inductive Vertex
| A | B | C | D | E | F | G

/-- Represents a triangle formed by the center and two adjacent vertices -/
structure Triangle where
  v1 : Vertex
  v2 : Vertex
  v3 : Vertex

/-- The set of all triangles in the hexagon -/
def hexagonTriangles : List Triangle := [
  ⟨Vertex.A, Vertex.B, Vertex.G⟩,
  ⟨Vertex.B, Vertex.C, Vertex.G⟩,
  ⟨Vertex.C, Vertex.D, Vertex.G⟩,
  ⟨Vertex.D, Vertex.E, Vertex.G⟩,
  ⟨Vertex.E, Vertex.F, Vertex.G⟩,
  ⟨Vertex.F, Vertex.A, Vertex.G⟩
]

/-- A function that assigns an integer to each vertex -/
def VertexAssignment := Vertex → Int

/-- Checks if the integers assigned to a triangle are in ascending order clockwise -/
def isAscendingClockwise (assignment : VertexAssignment) (t : Triangle) : Prop :=
  assignment t.v1 < assignment t.v2 ∧ assignment t.v2 < assignment t.v3

/-- The main theorem stating that no valid assignment exists -/
theorem no_valid_assignment :
  ¬∃ (assignment : VertexAssignment),
    (∀ v1 v2 : Vertex, v1 ≠ v2 → assignment v1 ≠ assignment v2) ∧
    (∀ t ∈ hexagonTriangles, isAscendingClockwise assignment t) :=
sorry


end NUMINAMATH_CALUDE_no_valid_assignment_l2330_233043


namespace NUMINAMATH_CALUDE_combine_like_terms_1_combine_like_terms_2_l2330_233092

-- Problem 1
theorem combine_like_terms_1 (a : ℝ) :
  2*a^2 - 3*a - 5 + 4*a + a^2 = 3*a^2 + a - 5 := by sorry

-- Problem 2
theorem combine_like_terms_2 (m n : ℝ) :
  2*m^2 + 5/2*n^2 - 1/3*(m^2 - 6*n^2) = 5/3*m^2 + 9/2*n^2 := by sorry

end NUMINAMATH_CALUDE_combine_like_terms_1_combine_like_terms_2_l2330_233092


namespace NUMINAMATH_CALUDE_system_solution_l2330_233027

theorem system_solution (a b : ℝ) : 
  (∃ x y : ℝ, x + y = a ∧ 2 * x + y = 16 ∧ x = 6 ∧ y = b) → 
  a = 10 ∧ b = 4 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2330_233027


namespace NUMINAMATH_CALUDE_expression_evaluation_l2330_233098

theorem expression_evaluation : 
  (120^2 - 13^2) / (80^2 - 17^2) * ((80 - 17)*(80 + 17)) / ((120 - 13)*(120 + 13)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2330_233098


namespace NUMINAMATH_CALUDE_range_of_a_l2330_233044

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x - a| - |x + 2| ≤ 3) → -5 ≤ a ∧ a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2330_233044


namespace NUMINAMATH_CALUDE_distance_between_specific_planes_l2330_233060

/-- The distance between two planes given by their equations -/
def distance_between_planes (a₁ b₁ c₁ d₁ a₂ b₂ c₂ d₂ : ℝ) : ℝ := sorry

/-- Theorem: The distance between the planes 2x - 4y + 4z = 10 and 4x - 8y + 8z = 20 is 0 -/
theorem distance_between_specific_planes :
  distance_between_planes 2 (-4) 4 10 4 (-8) 8 20 = 0 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_specific_planes_l2330_233060


namespace NUMINAMATH_CALUDE_difference_of_numbers_l2330_233050

theorem difference_of_numbers (x y : ℝ) : 
  x + y = 20 → x^2 - y^2 = 160 → x - y = 8 := by sorry

end NUMINAMATH_CALUDE_difference_of_numbers_l2330_233050


namespace NUMINAMATH_CALUDE_monkey_fruit_ratio_l2330_233000

theorem monkey_fruit_ratio (a b x y z : ℝ) : 
  a > 0 ∧ b > 0 ∧ x > 0 ∧ y > 0 ∧ z > 0 →
  x = 1.4 * a →
  y + 0.25 * y = 1.25 * y →
  b = 2 * z →
  a + b = x + y →
  a + b = z + 1.4 * a →
  a / b = 1 / 2 := by
sorry


end NUMINAMATH_CALUDE_monkey_fruit_ratio_l2330_233000


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2330_233068

theorem purely_imaginary_complex_number (m : ℝ) : 
  (((m^2 - 5*m + 6) : ℂ) + (m^2 - 3*m)*I = (0 : ℂ) + ((m^2 - 3*m) : ℝ)*I) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2330_233068


namespace NUMINAMATH_CALUDE_quadratic_real_root_l2330_233049

theorem quadratic_real_root (b : ℝ) : 
  (∃ x : ℝ, x^2 + b*x + 25 = 0) ↔ b ≤ -10 ∨ b ≥ 10 := by sorry

end NUMINAMATH_CALUDE_quadratic_real_root_l2330_233049


namespace NUMINAMATH_CALUDE_train_seats_problem_l2330_233064

theorem train_seats_problem (total_cars : ℕ) 
  (half_free : ℕ) (third_free : ℕ) (all_occupied : ℕ)
  (h1 : total_cars = 18)
  (h2 : half_free + third_free + all_occupied = total_cars)
  (h3 : (half_free * 6 + third_free * 4) * 2 = total_cars * 4) :
  all_occupied = 13 := by
  sorry

end NUMINAMATH_CALUDE_train_seats_problem_l2330_233064


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_l2330_233016

theorem least_addition_for_divisibility : ∃! n : ℕ,
  (∀ m : ℕ, m < n → ¬((1789 + m) % 5 = 0 ∧ (1789 + m) % 7 = 0 ∧ (1789 + m) % 11 = 0 ∧ (1789 + m) % 13 = 0)) ∧
  ((1789 + n) % 5 = 0 ∧ (1789 + n) % 7 = 0 ∧ (1789 + n) % 11 = 0 ∧ (1789 + n) % 13 = 0) ∧
  n = 3216 :=
by sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_l2330_233016


namespace NUMINAMATH_CALUDE_derivative_at_negative_one_l2330_233071

def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^4 + b * x^2 + c

theorem derivative_at_negative_one 
  (a b c : ℝ) 
  (h : (4 * a + 2 * b) = 2) : 
  (4 * a * (-1)^3 + 2 * b * (-1)) = -2 := by sorry

end NUMINAMATH_CALUDE_derivative_at_negative_one_l2330_233071


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l2330_233053

theorem complex_modulus_problem (z : ℂ) : z * Complex.I ^ 2018 = 3 + 4 * Complex.I → Complex.abs z = 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l2330_233053


namespace NUMINAMATH_CALUDE_equation_solutions_function_property_l2330_233081

-- Part a
theorem equation_solutions (x : ℝ) : 2^x = x + 1 ↔ x = 0 ∨ x = 1 := by sorry

-- Part b
theorem function_property (f : ℝ → ℝ) (h : ∀ x, (f ∘ f) x = 2^x - 1) : f 0 + f 1 = 1 := by sorry

end NUMINAMATH_CALUDE_equation_solutions_function_property_l2330_233081


namespace NUMINAMATH_CALUDE_parabola_b_value_l2330_233069

-- Define the parabola equation
def parabola (a b x : ℝ) : ℝ := x^2 + a*x + b

-- State the theorem
theorem parabola_b_value :
  ∀ a b : ℝ,
  (parabola a b 2 = 10) →
  (parabola a b (-2) = 6) →
  b = 4 := by
sorry

end NUMINAMATH_CALUDE_parabola_b_value_l2330_233069


namespace NUMINAMATH_CALUDE_city_population_problem_l2330_233039

theorem city_population_problem :
  ∃ (N : ℕ),
    (∃ (x : ℕ), N = x^2) ∧
    (∃ (y : ℕ), N + 100 = y^2 + 1) ∧
    (∃ (z : ℕ), N + 200 = z^2) ∧
    (∃ (k : ℕ), N = 7 * k) :=
by
  sorry

end NUMINAMATH_CALUDE_city_population_problem_l2330_233039


namespace NUMINAMATH_CALUDE_tan_1450_degrees_solution_l2330_233030

theorem tan_1450_degrees_solution (n : ℤ) : 
  -180 < n ∧ n < 180 ∧ Real.tan (n * π / 180) = Real.tan (1450 * π / 180) →
  n = 10 ∨ n = -170 := by
sorry

end NUMINAMATH_CALUDE_tan_1450_degrees_solution_l2330_233030


namespace NUMINAMATH_CALUDE_half_abs_diff_squares_21_17_l2330_233052

theorem half_abs_diff_squares_21_17 : (1/2 : ℚ) * |21^2 - 17^2| = 76 := by
  sorry

end NUMINAMATH_CALUDE_half_abs_diff_squares_21_17_l2330_233052


namespace NUMINAMATH_CALUDE_james_tylenol_dosage_l2330_233095

/-- Represents the dosage schedule and total daily intake of Tylenol tablets -/
structure TylenolDosage where
  tablets_per_dose : ℕ
  hours_between_doses : ℕ
  total_daily_mg : ℕ

/-- Calculates the mg per tablet given a TylenolDosage -/
def mg_per_tablet (dosage : TylenolDosage) : ℕ :=
  let doses_per_day := 24 / dosage.hours_between_doses
  let tablets_per_day := doses_per_day * dosage.tablets_per_dose
  dosage.total_daily_mg / tablets_per_day

/-- Theorem: Given James' Tylenol dosage schedule, each tablet contains 375 mg -/
theorem james_tylenol_dosage :
  let james_dosage : TylenolDosage := {
    tablets_per_dose := 2,
    hours_between_doses := 6,
    total_daily_mg := 3000
  }
  mg_per_tablet james_dosage = 375 := by
  sorry

end NUMINAMATH_CALUDE_james_tylenol_dosage_l2330_233095


namespace NUMINAMATH_CALUDE_functions_properties_l2330_233009

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := (Real.exp x - Real.exp (-x)) / 2
noncomputable def g (x : ℝ) : ℝ := (Real.exp x + Real.exp (-x)) / 2

-- Theorem statement
theorem functions_properties :
  (∀ x : ℝ, f x < g x) ∧
  (∃ x : ℝ, f x ^ 2 + g x ^ 2 ≥ 1) ∧
  (∀ x : ℝ, f (2 * x) = 2 * f x * g x) := by
  sorry

end NUMINAMATH_CALUDE_functions_properties_l2330_233009


namespace NUMINAMATH_CALUDE_adult_ticket_cost_l2330_233014

theorem adult_ticket_cost (child_ticket_cost : ℕ) (total_tickets : ℕ) (total_revenue : ℕ) (adult_tickets : ℕ) :
  child_ticket_cost = 4 →
  total_tickets = 900 →
  total_revenue = 5100 →
  adult_tickets = 500 →
  ∃ (adult_ticket_cost : ℕ), adult_ticket_cost = 7 ∧
    adult_ticket_cost * adult_tickets + child_ticket_cost * (total_tickets - adult_tickets) = total_revenue :=
by sorry

end NUMINAMATH_CALUDE_adult_ticket_cost_l2330_233014


namespace NUMINAMATH_CALUDE_school_fee_calculation_l2330_233032

/-- Represents the number of bills of each denomination given by a parent -/
structure BillCount where
  fifty : Nat
  twenty : Nat
  ten : Nat

/-- Calculates the total value of bills given by a parent -/
def totalValue (bills : BillCount) : Nat :=
  50 * bills.fifty + 20 * bills.twenty + 10 * bills.ten

theorem school_fee_calculation (mother father : BillCount)
    (h_mother : mother = { fifty := 1, twenty := 2, ten := 3 })
    (h_father : father = { fifty := 4, twenty := 1, ten := 1 }) :
    totalValue mother + totalValue father = 350 := by
  sorry

end NUMINAMATH_CALUDE_school_fee_calculation_l2330_233032


namespace NUMINAMATH_CALUDE_odd_number_as_difference_of_squares_l2330_233097

theorem odd_number_as_difference_of_squares (n : ℕ) (hn : n > 0) :
  ∃! (x y : ℕ), (2 * n + 1 : ℕ) = x^2 - y^2 ∧ x = n + 1 ∧ y = n :=
by sorry

end NUMINAMATH_CALUDE_odd_number_as_difference_of_squares_l2330_233097


namespace NUMINAMATH_CALUDE_quiz_correct_answers_l2330_233010

theorem quiz_correct_answers
  (wendy_correct : ℕ)
  (campbell_correct : ℕ)
  (kelsey_correct : ℕ)
  (martin_correct : ℕ)
  (h1 : wendy_correct = 20)
  (h2 : campbell_correct = 2 * wendy_correct)
  (h3 : kelsey_correct = campbell_correct + 8)
  (h4 : martin_correct = kelsey_correct - 3) :
  martin_correct = 45 := by
  sorry

end NUMINAMATH_CALUDE_quiz_correct_answers_l2330_233010


namespace NUMINAMATH_CALUDE_sara_flowers_l2330_233025

/-- Given the number of red flowers and the number of bouquets, 
    calculate the number of yellow flowers needed to create bouquets 
    with an equal number of red and yellow flowers in each. -/
def yellow_flowers (red_flowers : ℕ) (num_bouquets : ℕ) : ℕ :=
  (red_flowers / num_bouquets) * num_bouquets

/-- Theorem stating that given 16 red flowers and 8 bouquets,
    the number of yellow flowers needed is 16. -/
theorem sara_flowers : yellow_flowers 16 8 = 16 := by
  sorry

end NUMINAMATH_CALUDE_sara_flowers_l2330_233025


namespace NUMINAMATH_CALUDE_count_eight_digit_numbers_with_product_4900_l2330_233034

/-- The number of eight-digit numbers whose digits' product equals 4900 -/
def eight_digit_numbers_with_product_4900 : ℕ := 4200

/-- Theorem stating that the number of eight-digit numbers whose digits' product equals 4900 is 4200 -/
theorem count_eight_digit_numbers_with_product_4900 :
  eight_digit_numbers_with_product_4900 = 4200 := by
  sorry

end NUMINAMATH_CALUDE_count_eight_digit_numbers_with_product_4900_l2330_233034


namespace NUMINAMATH_CALUDE_arithmetic_sqrt_one_fourth_l2330_233007

-- Define the arithmetic square root function
noncomputable def arithmetic_sqrt (x : ℝ) : ℝ := Real.sqrt x

-- State the theorem
theorem arithmetic_sqrt_one_fourth : arithmetic_sqrt (1/4) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sqrt_one_fourth_l2330_233007


namespace NUMINAMATH_CALUDE_largest_constant_inequality_two_is_largest_constant_l2330_233008

theorem largest_constant_inequality (a b c d e : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0) :
  Real.sqrt (a / (b + c + d + e)) + Real.sqrt (b / (a + c + d + e)) +
  Real.sqrt (c / (a + b + d + e)) + Real.sqrt (d / (a + b + c + e)) +
  Real.sqrt (e / (a + b + c + d)) > 2 :=
sorry

theorem two_is_largest_constant :
  ∀ ε > 0, ∃ a b c d e : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧
    Real.sqrt (a / (b + c + d + e)) + Real.sqrt (b / (a + c + d + e)) +
    Real.sqrt (c / (a + b + d + e)) + Real.sqrt (d / (a + b + c + e)) +
    Real.sqrt (e / (a + b + c + d)) < 2 + ε :=
sorry

end NUMINAMATH_CALUDE_largest_constant_inequality_two_is_largest_constant_l2330_233008


namespace NUMINAMATH_CALUDE_pen_pencil_ratio_l2330_233024

theorem pen_pencil_ratio (num_pencils : ℕ) (num_pens : ℕ) : 
  num_pencils = 36 → 
  num_pencils = num_pens + 6 → 
  (num_pens : ℚ) / (num_pencils : ℚ) = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_pen_pencil_ratio_l2330_233024


namespace NUMINAMATH_CALUDE_down_payment_equals_108000_l2330_233005

/-- The amount of money needed for a down payment on a house -/
def down_payment (richard_monthly_savings : ℕ) (sarah_monthly_savings : ℕ) (years : ℕ) : ℕ :=
  (richard_monthly_savings + sarah_monthly_savings) * years * 12

/-- Theorem stating that Richard and Sarah's savings over 3 years equal $108,000 -/
theorem down_payment_equals_108000 :
  down_payment 1500 1500 3 = 108000 := by
  sorry

end NUMINAMATH_CALUDE_down_payment_equals_108000_l2330_233005


namespace NUMINAMATH_CALUDE_x_minus_y_value_l2330_233013

theorem x_minus_y_value (x y : ℤ) (hx : x = -3) (hy : |y| = 4) :
  x - y = 1 ∨ x - y = -7 := by
  sorry

end NUMINAMATH_CALUDE_x_minus_y_value_l2330_233013


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2330_233040

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the given equation
def given_equation (z : ℂ) (a : ℝ) : Prop :=
  z / (a + 2 * i) = i

-- Define the condition that real part equals imaginary part
def real_equals_imag (z : ℂ) : Prop :=
  z.re = z.im

-- The theorem to prove
theorem complex_equation_solution (z : ℂ) (a : ℝ) 
  (h1 : given_equation z a) 
  (h2 : real_equals_imag (z / (a + 2 * i))) : 
  a = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2330_233040


namespace NUMINAMATH_CALUDE_sum_of_b_values_l2330_233022

/-- The sum of the two values of b for which the equation 3x^2 + bx + 6x + 7 = 0 has only one solution for x -/
theorem sum_of_b_values (b₁ b₂ : ℝ) : 
  (∃! x, 3 * x^2 + b₁ * x + 6 * x + 7 = 0) →
  (∃! x, 3 * x^2 + b₂ * x + 6 * x + 7 = 0) →
  b₁ + b₂ = -12 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_b_values_l2330_233022


namespace NUMINAMATH_CALUDE_leo_current_weight_l2330_233090

/-- Leo's current weight in pounds -/
def leo_weight : ℝ := 92

/-- Kendra's current weight in pounds -/
def kendra_weight : ℝ := 160 - leo_weight

/-- The combined weight of Leo and Kendra in pounds -/
def combined_weight : ℝ := 160

theorem leo_current_weight :
  (leo_weight + 10 = 1.5 * kendra_weight) ∧
  (leo_weight + kendra_weight = combined_weight) ∧
  (leo_weight = 92) := by
sorry

end NUMINAMATH_CALUDE_leo_current_weight_l2330_233090


namespace NUMINAMATH_CALUDE_smallest_n_for_integer_sum_l2330_233096

theorem smallest_n_for_integer_sum : 
  ∃ (n : ℕ), n > 0 ∧ 
  (1/3 + 1/4 + 1/8 + 1/n : ℚ).isInt ∧ 
  (∀ m : ℕ, m > 0 ∧ (1/3 + 1/4 + 1/8 + 1/m : ℚ).isInt → n ≤ m) ∧ 
  n = 24 := by
sorry

end NUMINAMATH_CALUDE_smallest_n_for_integer_sum_l2330_233096


namespace NUMINAMATH_CALUDE_star_self_inverse_l2330_233021

/-- The star operation for real numbers -/
def star (a b : ℝ) : ℝ := (a^2 - b^2)^2

/-- Theorem: The star operation of (x^2 - y^2) and (y^2 - x^2) is zero -/
theorem star_self_inverse (x y : ℝ) : star (x^2 - y^2) (y^2 - x^2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_star_self_inverse_l2330_233021


namespace NUMINAMATH_CALUDE_workshop_workers_l2330_233067

/-- Represents the total number of workers in a workshop -/
def total_workers : ℕ := 20

/-- Represents the number of technicians -/
def technicians : ℕ := 5

/-- Represents the average salary of all workers -/
def avg_salary_all : ℕ := 750

/-- Represents the average salary of technicians -/
def avg_salary_technicians : ℕ := 900

/-- Represents the average salary of non-technician workers -/
def avg_salary_others : ℕ := 700

/-- Theorem stating that given the conditions, the total number of workers is 20 -/
theorem workshop_workers : 
  (total_workers * avg_salary_all = technicians * avg_salary_technicians + 
   (total_workers - technicians) * avg_salary_others) → 
  total_workers = 20 :=
by sorry

end NUMINAMATH_CALUDE_workshop_workers_l2330_233067


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2330_233038

/-- The eccentricity of a hyperbola with specific conditions -/
theorem hyperbola_eccentricity : ∀ (a b : ℝ) (P : ℝ × ℝ),
  a > 0 → b > 0 →
  -- Hyperbola equation
  P.1^2 / a^2 - P.2^2 / b^2 = 1 →
  -- P is on the curve y = √x
  P.2 = Real.sqrt P.1 →
  -- Tangent line passes through the left focus (-1, 0)
  (Real.sqrt P.1 - 0) / (P.1 - (-1)) = 1 / (2 * Real.sqrt P.1) →
  -- The eccentricity is (√5 + 1) / 2
  a / Real.sqrt (a^2 + b^2) = (Real.sqrt 5 + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2330_233038


namespace NUMINAMATH_CALUDE_triangle_angle_values_l2330_233031

theorem triangle_angle_values (A B C : Real) (AB AC : Real) :
  AB = 2 →
  AC = Real.sqrt 2 →
  B = 30 * Real.pi / 180 →
  A = 105 * Real.pi / 180 ∨ A = 15 * Real.pi / 180 :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_values_l2330_233031


namespace NUMINAMATH_CALUDE_unique_integral_solution_l2330_233085

theorem unique_integral_solution (x y z n : ℤ) 
  (h1 : x * y + y * z + z * x = 3 * n^2 - 1)
  (h2 : x + y + z = 3 * n)
  (h3 : x ≥ y ∧ y ≥ z) :
  x = n + 1 ∧ y = n ∧ z = n - 1 :=
by sorry

end NUMINAMATH_CALUDE_unique_integral_solution_l2330_233085


namespace NUMINAMATH_CALUDE_symmetric_point_x_axis_l2330_233028

/-- A point in 3D Cartesian space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Symmetry with respect to the x-axis -/
def symmetricPointXAxis (p : Point3D) : Point3D :=
  ⟨p.x, -p.y, -p.z⟩

theorem symmetric_point_x_axis :
  let original := Point3D.mk (-2) 1 4
  symmetricPointXAxis original = Point3D.mk (-2) (-1) (-4) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_x_axis_l2330_233028


namespace NUMINAMATH_CALUDE_oliver_seashells_l2330_233065

/-- The number of seashells Oliver collected. -/
def total_seashells : ℕ := 4

/-- The number of seashells Oliver collected on Tuesday. -/
def tuesday_seashells : ℕ := 2

/-- The number of seashells Oliver collected on Monday. -/
def monday_seashells : ℕ := total_seashells - tuesday_seashells

theorem oliver_seashells :
  monday_seashells = total_seashells - tuesday_seashells :=
by sorry

end NUMINAMATH_CALUDE_oliver_seashells_l2330_233065


namespace NUMINAMATH_CALUDE_divisibility_implies_equality_l2330_233020

theorem divisibility_implies_equality (a b : ℕ) (h : (a^2 + b^2) ∣ (a * b)) : a = b := by
  sorry

end NUMINAMATH_CALUDE_divisibility_implies_equality_l2330_233020


namespace NUMINAMATH_CALUDE_count_goats_l2330_233035

/-- Given a field with animals, prove the number of goats -/
theorem count_goats (total : ℕ) (cows : ℕ) (sheep_and_goats : ℕ) 
  (h1 : total = 200)
  (h2 : cows = 40)
  (h3 : sheep_and_goats = 56)
  : total - cows - sheep_and_goats = 104 := by
  sorry

end NUMINAMATH_CALUDE_count_goats_l2330_233035


namespace NUMINAMATH_CALUDE_find_y_value_l2330_233047

theorem find_y_value (x y : ℝ) (h1 : 1.5 * x = 0.75 * y) (h2 : x = 20) : y = 40 := by
  sorry

end NUMINAMATH_CALUDE_find_y_value_l2330_233047


namespace NUMINAMATH_CALUDE_positive_c_in_quadratic_with_no_roots_l2330_233091

/-- A quadratic trinomial with no roots and positive sum of coefficients has a positive constant term. -/
theorem positive_c_in_quadratic_with_no_roots 
  (a b c : ℝ) 
  (no_roots : ∀ x : ℝ, a * x^2 + b * x + c ≠ 0) 
  (sum_positive : a + b + c > 0) : 
  c > 0 := by
  sorry

end NUMINAMATH_CALUDE_positive_c_in_quadratic_with_no_roots_l2330_233091


namespace NUMINAMATH_CALUDE_committee_probability_l2330_233066

def total_members : ℕ := 30
def num_boys : ℕ := 12
def num_girls : ℕ := 18
def committee_size : ℕ := 6

theorem committee_probability :
  let total_ways := Nat.choose total_members committee_size
  let unwanted_cases := Nat.choose num_girls committee_size +
                        num_boys * Nat.choose num_girls (committee_size - 1) +
                        Nat.choose num_boys committee_size +
                        num_girls * Nat.choose num_boys (committee_size - 1)
  (total_ways - unwanted_cases : ℚ) / total_ways = 457215 / 593775 := by
  sorry

end NUMINAMATH_CALUDE_committee_probability_l2330_233066


namespace NUMINAMATH_CALUDE_dodecahedron_edge_probability_l2330_233018

/-- A regular dodecahedron -/
structure RegularDodecahedron where
  vertices : Finset (Fin 20)
  edges : Finset (Fin 20 × Fin 20)
  vertex_degree : ∀ v : Fin 20, (edges.filter (λ e => e.1 = v ∨ e.2 = v)).card = 3

/-- The probability of selecting two vertices that form an edge in a regular dodecahedron -/
def edge_selection_probability (d : RegularDodecahedron) : ℚ :=
  3 / 19

/-- Theorem: The probability of randomly selecting two vertices that are endpoints 
    of the same edge in a regular dodecahedron is 3/19 -/
theorem dodecahedron_edge_probability (d : RegularDodecahedron) : 
  edge_selection_probability d = 3 / 19 := by
  sorry

end NUMINAMATH_CALUDE_dodecahedron_edge_probability_l2330_233018


namespace NUMINAMATH_CALUDE_max_value_rational_function_l2330_233075

theorem max_value_rational_function (x : ℝ) (h : x < -1) :
  (x^2 + 7*x + 10) / (x + 1) ≤ 1 ∧
  (x^2 + 7*x + 10) / (x + 1) = 1 ↔ x = -3 :=
by sorry

end NUMINAMATH_CALUDE_max_value_rational_function_l2330_233075


namespace NUMINAMATH_CALUDE_apple_bags_theorem_l2330_233080

def is_valid_total (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = 6 * a + 12 * b ∧ 70 ≤ n ∧ n ≤ 80

theorem apple_bags_theorem : 
  {n : ℕ | is_valid_total n} = {72, 78} :=
sorry

end NUMINAMATH_CALUDE_apple_bags_theorem_l2330_233080


namespace NUMINAMATH_CALUDE_pond_area_is_292_l2330_233041

/-- The total surface area of a cuboid-shaped pond, excluding the top surface. -/
def pondSurfaceArea (length width height : ℝ) : ℝ :=
  length * width + 2 * length * height + 2 * width * height

/-- Theorem: The surface area of a pond with given dimensions is 292 square meters. -/
theorem pond_area_is_292 :
  pondSurfaceArea 18 10 2 = 292 := by
  sorry

#eval pondSurfaceArea 18 10 2

end NUMINAMATH_CALUDE_pond_area_is_292_l2330_233041


namespace NUMINAMATH_CALUDE_distinct_centroids_count_l2330_233058

/-- Represents a point on the perimeter of the square -/
structure PerimeterPoint where
  x : Fin 11
  y : Fin 11
  on_perimeter : (x = 0 ∨ x = 10) ∨ (y = 0 ∨ y = 10)

/-- The set of 40 equally spaced points on the square's perimeter -/
def perimeterPoints : Finset PerimeterPoint :=
  sorry

/-- Represents the centroid of a triangle -/
structure Centroid where
  x : Rat
  y : Rat
  inside_square : 0 < x ∧ x < 10 ∧ 0 < y ∧ y < 10

/-- Function to calculate the centroid given three points -/
def calculateCentroid (p q r : PerimeterPoint) : Centroid :=
  sorry

/-- The set of all possible centroids -/
def allCentroids : Finset Centroid :=
  sorry

/-- Main theorem: The number of distinct centroids is 841 -/
theorem distinct_centroids_count : Finset.card allCentroids = 841 :=
  sorry

end NUMINAMATH_CALUDE_distinct_centroids_count_l2330_233058


namespace NUMINAMATH_CALUDE_sum_of_exponents_of_sqrt_largest_perfect_square_15_factorial_l2330_233076

-- Define the factorial function
def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

-- Define a function to calculate the exponent of a prime factor in n!
def primeExponentInFactorial (n : ℕ) (p : ℕ) : ℕ :=
  if p.Prime then
    (List.range (n + 1)).foldl (λ acc k => acc + k / p^k) 0
  else
    0

-- Define a function to get the largest even number not exceeding n
def largestEvenNotExceeding (n : ℕ) : ℕ :=
  if n % 2 = 0 then n else n - 1

-- Define the main theorem
theorem sum_of_exponents_of_sqrt_largest_perfect_square_15_factorial :
  (let n := 15
   let primes := [2, 3, 5, 7]
   let exponents := primes.map (λ p => largestEvenNotExceeding (primeExponentInFactorial n p) / 2)
   exponents.sum) = 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_exponents_of_sqrt_largest_perfect_square_15_factorial_l2330_233076


namespace NUMINAMATH_CALUDE_evaluate_expression_l2330_233017

theorem evaluate_expression : -(16 / 4 * 11 - 70 + 5^2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2330_233017


namespace NUMINAMATH_CALUDE_fraction_reciprocal_product_l2330_233063

theorem fraction_reciprocal_product : (1 / (5 / 3)) * (5 / 3) = 1 := by sorry

end NUMINAMATH_CALUDE_fraction_reciprocal_product_l2330_233063


namespace NUMINAMATH_CALUDE_gcd_lcm_product_24_60_l2330_233093

theorem gcd_lcm_product_24_60 : Nat.gcd 24 60 * Nat.lcm 24 60 = 1440 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_24_60_l2330_233093


namespace NUMINAMATH_CALUDE_father_son_meeting_point_father_son_meeting_point_specific_l2330_233029

/-- The meeting point of a father and son in a hallway -/
theorem father_son_meeting_point (hallway_length : ℝ) (speed_ratio : ℝ) : 
  hallway_length > 0 → 
  speed_ratio > 1 → 
  (speed_ratio * hallway_length) / (speed_ratio + 1) = 
    hallway_length - hallway_length / (speed_ratio + 1) :=
by
  sorry

/-- The specific case of a 16m hallway and 3:1 speed ratio -/
theorem father_son_meeting_point_specific : 
  (16 : ℝ) - 16 / (3 + 1) = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_father_son_meeting_point_father_son_meeting_point_specific_l2330_233029


namespace NUMINAMATH_CALUDE_smallest_solution_congruence_l2330_233083

theorem smallest_solution_congruence :
  ∃ (x : ℕ), x > 0 ∧ (5 * x) % 31 = 17 % 31 ∧
  ∀ (y : ℕ), y > 0 → (5 * y) % 31 = 17 % 31 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_congruence_l2330_233083


namespace NUMINAMATH_CALUDE_waiter_tips_ratio_l2330_233045

theorem waiter_tips_ratio (salary tips : ℝ) 
  (h : tips / (salary + tips) = 0.6363636363636364) :
  tips / salary = 1.75 := by
sorry

end NUMINAMATH_CALUDE_waiter_tips_ratio_l2330_233045


namespace NUMINAMATH_CALUDE_largest_four_digit_congruent_to_17_mod_24_l2330_233057

theorem largest_four_digit_congruent_to_17_mod_24 : ∃ n : ℕ, 
  (n ≡ 17 [ZMOD 24]) ∧ 
  (n < 10000) ∧ 
  (1000 ≤ n) ∧ 
  (∀ m : ℕ, (m ≡ 17 [ZMOD 24]) → (1000 ≤ m) → (m < 10000) → m ≤ n) ∧ 
  n = 9977 :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_congruent_to_17_mod_24_l2330_233057


namespace NUMINAMATH_CALUDE_frog_corner_probability_l2330_233082

/-- Represents a position on the 4x4 grid -/
structure Position :=
  (x : Fin 4)
  (y : Fin 4)

/-- Represents a direction of hop -/
inductive Direction
  | Up
  | Down
  | Left
  | Right

/-- The grid on which the frog hops -/
def Grid := Fin 4 → Fin 4 → Bool

/-- Checks if a position is a corner -/
def isCorner (p : Position) : Bool :=
  (p.x = 0 ∧ p.y = 0) ∨ (p.x = 0 ∧ p.y = 3) ∨ (p.x = 3 ∧ p.y = 0) ∨ (p.x = 3 ∧ p.y = 3)

/-- Performs a single hop in the given direction with wrap-around -/
def hop (p : Position) (d : Direction) : Position :=
  match d with
  | Direction.Up    => ⟨p.x, (p.y + 1) % 4⟩
  | Direction.Down  => ⟨p.x, (p.y - 1 + 4) % 4⟩
  | Direction.Left  => ⟨(p.x - 1 + 4) % 4, p.y⟩
  | Direction.Right => ⟨(p.x + 1) % 4, p.y⟩

/-- Calculates the probability of reaching a corner within n hops -/
def probReachCorner (start : Position) (n : Nat) : Rat :=
  sorry

/-- The main theorem to prove -/
theorem frog_corner_probability :
  probReachCorner ⟨1, 1⟩ 5 = 15/16 :=
sorry

end NUMINAMATH_CALUDE_frog_corner_probability_l2330_233082


namespace NUMINAMATH_CALUDE_sqrt_product_sqrt_l2330_233062

theorem sqrt_product_sqrt : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_sqrt_l2330_233062


namespace NUMINAMATH_CALUDE_sum_of_matching_indices_l2330_233004

def sequence_length : ℕ := 1011

def is_odd (n : ℕ) : Prop := n % 2 = 1

def sum_of_odds (n : ℕ) : ℕ := ((n + 1) / 2) ^ 2

theorem sum_of_matching_indices :
  sum_of_odds sequence_length = 256036 :=
sorry

end NUMINAMATH_CALUDE_sum_of_matching_indices_l2330_233004


namespace NUMINAMATH_CALUDE_range_of_fraction_l2330_233061

theorem range_of_fraction (x y : ℝ) (h : (x - 1)^2 + y^2 = 1) :
  ∃ (k : ℝ), y / (x + 1) = k ∧ -Real.sqrt 3 / 3 ≤ k ∧ k ≤ Real.sqrt 3 / 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_fraction_l2330_233061


namespace NUMINAMATH_CALUDE_cos_330_degrees_l2330_233059

theorem cos_330_degrees : Real.cos (330 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_330_degrees_l2330_233059


namespace NUMINAMATH_CALUDE_centroid_on_line_segment_l2330_233088

/-- Given a triangle ABC with points M on AB and N on AC, if BM/MA + CN/NA = 1,
    then the centroid of triangle ABC is collinear with M and N. -/
theorem centroid_on_line_segment (A B C M N : EuclideanSpace ℝ (Fin 2)) :
  (∃ s t : ℝ, 0 < s ∧ s < 1 ∧ 0 < t ∧ t < 1 ∧
   M = (1 - s) • A + s • B ∧
   N = (1 - t) • A + t • C ∧
   s / (1 - s) + t / (1 - t) = 1) →
  ∃ u : ℝ, (1/3 : ℝ) • (A + B + C) = (1 - u) • M + u • N :=
by sorry

end NUMINAMATH_CALUDE_centroid_on_line_segment_l2330_233088


namespace NUMINAMATH_CALUDE_mixture_volume_l2330_233046

/-- Given a mixture of two liquids p and q with an initial ratio of 5:3,
    if adding 15 liters of liquid q changes the ratio to 5:6,
    then the initial volume of the mixture was 40 liters. -/
theorem mixture_volume (p q : ℝ) (h1 : p / q = 5 / 3) 
    (h2 : p / (q + 15) = 5 / 6) : p + q = 40 := by
  sorry

end NUMINAMATH_CALUDE_mixture_volume_l2330_233046


namespace NUMINAMATH_CALUDE_selection_schemes_l2330_233037

theorem selection_schemes (num_boys num_girls : ℕ) (h1 : num_boys = 4) (h2 : num_girls = 2) :
  (num_boys : ℕ) * (num_girls : ℕ) = 8 := by
  sorry

end NUMINAMATH_CALUDE_selection_schemes_l2330_233037


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2330_233055

-- Define the sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 1}
def B : Set ℝ := {x | 0 ≤ x ∧ x ≤ 3}

-- Define the intersection of A and B
def A_intersect_B : Set ℝ := A ∩ B

-- Theorem statement
theorem intersection_of_A_and_B :
  A_intersect_B = {x | 0 ≤ x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2330_233055


namespace NUMINAMATH_CALUDE_abigail_lost_money_l2330_233074

def money_lost (initial_amount spent_amount remaining_amount : ℕ) : ℕ :=
  initial_amount - spent_amount - remaining_amount

theorem abigail_lost_money : money_lost 11 2 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_abigail_lost_money_l2330_233074


namespace NUMINAMATH_CALUDE_fifth_diagram_shaded_fraction_l2330_233087

/-- Represents the number of shaded triangles in the n-th diagram -/
def shadedTriangles (n : ℕ) : ℕ := 2^(n - 1)

/-- Represents the total number of triangles in the n-th diagram -/
def totalTriangles (n : ℕ) : ℕ := n^2

/-- The fraction of shaded triangles in the n-th diagram -/
def shadedFraction (n : ℕ) : ℚ :=
  (shadedTriangles n : ℚ) / (totalTriangles n : ℚ)

theorem fifth_diagram_shaded_fraction :
  shadedFraction 5 = 16 / 25 := by
  sorry

end NUMINAMATH_CALUDE_fifth_diagram_shaded_fraction_l2330_233087


namespace NUMINAMATH_CALUDE_staircase_theorem_l2330_233002

def staircase_problem (first_staircase : ℕ) (step_height : ℚ) : ℚ :=
  let second_staircase := 2 * first_staircase
  let third_staircase := second_staircase - 10
  let total_steps := first_staircase + second_staircase + third_staircase
  total_steps * step_height

theorem staircase_theorem :
  staircase_problem 20 (1/2) = 45 := by
  sorry

end NUMINAMATH_CALUDE_staircase_theorem_l2330_233002


namespace NUMINAMATH_CALUDE_binary_difference_digits_l2330_233006

theorem binary_difference_digits : ∃ (b : ℕ → Bool), 
  (Nat.castRingHom ℕ).toFun ((Nat.digits 2 1500).foldl (λ acc d => 2 * acc + d) 0 - 
                              (Nat.digits 2 300).foldl (λ acc d => 2 * acc + d) 0) = 
  (Nat.digits 2 1200).foldl (λ acc d => 2 * acc + d) 0 ∧
  (Nat.digits 2 1200).length = 11 :=
by sorry

end NUMINAMATH_CALUDE_binary_difference_digits_l2330_233006


namespace NUMINAMATH_CALUDE_cricket_average_l2330_233089

theorem cricket_average (initial_average : ℚ) : 
  (10 * initial_average + 65 = 11 * (initial_average + 3)) → initial_average = 32 := by
sorry

end NUMINAMATH_CALUDE_cricket_average_l2330_233089


namespace NUMINAMATH_CALUDE_arithmetic_mean_squares_l2330_233077

theorem arithmetic_mean_squares (x a : ℝ) (hx : x ≠ 0) (ha : a ≠ 0) :
  ((((x + a)^2) / x + ((x - a)^2) / x) / 2) = x + a^2 / x :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_squares_l2330_233077


namespace NUMINAMATH_CALUDE_sports_club_size_l2330_233048

/-- The number of members in a sports club -/
def sports_club_members (badminton tennis both neither : ℕ) : ℕ :=
  badminton + tennis - both + neither

/-- Theorem: The sports club has 40 members -/
theorem sports_club_size :
  sports_club_members 20 18 3 5 = 40 := by
  sorry

end NUMINAMATH_CALUDE_sports_club_size_l2330_233048


namespace NUMINAMATH_CALUDE_outfit_problem_l2330_233084

/-- The number of possible outfits given shirts, pants, and restrictions -/
def num_outfits (shirts : ℕ) (pants : ℕ) (restricted_shirts : ℕ) (restricted_pants : ℕ) : ℕ :=
  (shirts - restricted_shirts) * pants + restricted_shirts * (pants - restricted_pants)

/-- Theorem stating the number of outfits for the given problem -/
theorem outfit_problem :
  num_outfits 5 4 2 1 = 18 := by
  sorry


end NUMINAMATH_CALUDE_outfit_problem_l2330_233084


namespace NUMINAMATH_CALUDE_eventually_constant_l2330_233015

/-- S(n) is defined as n - m^2, where m is the greatest integer with m^2 ≤ n -/
def S (n : ℕ) : ℕ :=
  n - (Nat.sqrt n) ^ 2

/-- The sequence a_k is defined recursively -/
def a (A : ℕ) : ℕ → ℕ
  | 0 => A
  | k + 1 => a A k + S (a A k)

/-- The main theorem stating the condition for the sequence to be eventually constant -/
theorem eventually_constant (A : ℕ) :
  (∃ k : ℕ, ∀ n ≥ k, a A n = a A k) ↔ ∃ m : ℕ, A = m ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_eventually_constant_l2330_233015


namespace NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l2330_233036

/-- The complex number z = (2 + 3i) / (1 + 2i) is located in the fourth quadrant of the complex plane. -/
theorem complex_number_in_fourth_quadrant :
  let z : ℂ := (2 + 3*I) / (1 + 2*I)
  (z.re > 0) ∧ (z.im < 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l2330_233036


namespace NUMINAMATH_CALUDE_four_dice_same_number_l2330_233078

-- Define a standard six-sided die
def standard_die := Finset.range 6

-- Define the probability of getting the same number on all four dice
def same_number_probability : ℚ :=
  (1 : ℚ) / (standard_die.card ^ 4)

-- Theorem statement
theorem four_dice_same_number :
  same_number_probability = 1 / 216 := by
  sorry

end NUMINAMATH_CALUDE_four_dice_same_number_l2330_233078


namespace NUMINAMATH_CALUDE_factorization_problem_1_l2330_233001

theorem factorization_problem_1 (x : ℝ) :
  x^4 - 8*x^2 + 4 = (x^2 + 2*x - 2) * (x^2 - 2*x - 2) := by
sorry

end NUMINAMATH_CALUDE_factorization_problem_1_l2330_233001
