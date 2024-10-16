import Mathlib

namespace NUMINAMATH_CALUDE_square_sum_implies_product_l1775_177501

theorem square_sum_implies_product (x : ℝ) :
  Real.sqrt (10 + x) + Real.sqrt (15 - x) = 6 →
  (10 + x) * (15 - x) = 121 / 4 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_implies_product_l1775_177501


namespace NUMINAMATH_CALUDE_sum_10_terms_l1775_177514

/-- An arithmetic sequence with a₂ = 3 and a₉ = 17 -/
def arithmetic_seq (n : ℕ) : ℝ :=
  sorry

/-- The sum of the first n terms of the arithmetic sequence -/
def S (n : ℕ) : ℝ :=
  sorry

/-- Theorem: The sum of the first 10 terms of the arithmetic sequence is 100 -/
theorem sum_10_terms : S 10 = 100 :=
  sorry

end NUMINAMATH_CALUDE_sum_10_terms_l1775_177514


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_right_triangle_l1775_177592

/-- 
Given an isosceles trapezoid with parallel sides a and c, non-parallel sides (legs) b, 
and diagonals e, prove that e² = b² + ac, which implies that the triangle formed by 
e, b, and √(ac) is a right triangle.
-/
theorem isosceles_trapezoid_right_triangle 
  (a c b e : ℝ) 
  (h_positive : a > 0 ∧ c > 0 ∧ b > 0 ∧ e > 0)
  (h_isosceles : ∃ m : ℝ, b^2 = ((a - c)/2)^2 + m^2 ∧ e^2 = ((a + c)/2)^2 + m^2) :
  e^2 = b^2 + a*c :=
sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_right_triangle_l1775_177592


namespace NUMINAMATH_CALUDE_abs_inequality_implies_quadratic_inequality_l1775_177503

theorem abs_inequality_implies_quadratic_inequality :
  {x : ℝ | |x - 1| < 2} ⊂ {x : ℝ | (x + 2) * (x - 3) < 0} ∧
  {x : ℝ | |x - 1| < 2} ≠ {x : ℝ | (x + 2) * (x - 3) < 0} :=
sorry

end NUMINAMATH_CALUDE_abs_inequality_implies_quadratic_inequality_l1775_177503


namespace NUMINAMATH_CALUDE_sphere_diameter_from_cylinder_l1775_177564

/-- Given a cylinder with diameter 6 cm and height 6 cm, if spheres of equal volume are made from the same material, the diameter of each sphere is equal to the cube root of (162 * π) cm. -/
theorem sphere_diameter_from_cylinder (π : ℝ) (h : π > 0) :
  let cylinder_diameter : ℝ := 6
  let cylinder_height : ℝ := 6
  let cylinder_volume : ℝ := π * (cylinder_diameter / 2)^2 * cylinder_height
  let sphere_volume : ℝ := cylinder_volume
  let sphere_diameter : ℝ := 2 * (3 * sphere_volume / (4 * π))^(1/3)
  sphere_diameter = (162 * π)^(1/3) :=
by sorry

end NUMINAMATH_CALUDE_sphere_diameter_from_cylinder_l1775_177564


namespace NUMINAMATH_CALUDE_function_property_l1775_177574

/-- Given a function f(x) = (x^2 + ax + b)(e^x - e), where a and b are real numbers,
    and f(x) ≥ 0 for all x > 0, then a ≥ -1. -/
theorem function_property (a b : ℝ) :
  (∀ x > 0, (x^2 + a*x + b) * (Real.exp x - Real.exp 1) ≥ 0) →
  a ≥ -1 :=
by sorry

end NUMINAMATH_CALUDE_function_property_l1775_177574


namespace NUMINAMATH_CALUDE_seven_digit_numbers_existence_l1775_177567

theorem seven_digit_numbers_existence :
  ∃ (x y : ℕ),
    (10^6 ≤ x ∧ x < 10^7) ∧
    (10^6 ≤ y ∧ y < 10^7) ∧
    (3 * x * y = 10^7 * x + y) ∧
    (x = 166667 ∧ y = 333334) := by
  sorry

end NUMINAMATH_CALUDE_seven_digit_numbers_existence_l1775_177567


namespace NUMINAMATH_CALUDE_min_perimeter_rectangle_sphere_area_l1775_177510

/-- Given a rectangle ABCD with area 8, when its perimeter is minimized
    and triangle ACD is folded along diagonal AC to form a pyramid D-ABC,
    the surface area of the circumscribed sphere of this pyramid is 16π. -/
theorem min_perimeter_rectangle_sphere_area :
  ∀ (x y : ℝ),
  x > 0 → y > 0 →
  x * y = 8 →
  (∀ a b : ℝ, a > 0 → b > 0 → a * b = 8 → 2*(x + y) ≤ 2*(a + b)) →
  16 * Real.pi = 4 * Real.pi * (2 : ℝ)^2 := by
sorry

end NUMINAMATH_CALUDE_min_perimeter_rectangle_sphere_area_l1775_177510


namespace NUMINAMATH_CALUDE_triangle_angle_calculation_l1775_177596

theorem triangle_angle_calculation (a b c : ℝ) (A B C : ℝ) :
  a = 3 * Real.sqrt 2 →
  c = 3 →
  C = π / 6 →
  A = π / 4 ∨ A = 3 * π / 4 :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_calculation_l1775_177596


namespace NUMINAMATH_CALUDE_max_tax_revenue_l1775_177547

-- Define the market conditions
def supply (P : ℝ) : ℝ := 6 * P - 312
def demand_slope : ℝ := 4
def tax_rate : ℝ := 30
def consumer_price : ℝ := 118

-- Define the demand function
def demand (P : ℝ) : ℝ := 688 - demand_slope * P

-- Define the tax revenue function
def tax_revenue (t : ℝ) : ℝ := (288 - 2.4 * t) * t

-- Theorem statement
theorem max_tax_revenue :
  ∃ (t : ℝ), ∀ (t' : ℝ), tax_revenue t ≥ tax_revenue t' ∧ tax_revenue t = 8640 :=
sorry

end NUMINAMATH_CALUDE_max_tax_revenue_l1775_177547


namespace NUMINAMATH_CALUDE_fraction_relation_l1775_177511

theorem fraction_relation (x y z : ℚ) 
  (h1 : x / y = 3)
  (h2 : y / z = 5 / 2) :
  z / x = 2 / 15 := by
  sorry

end NUMINAMATH_CALUDE_fraction_relation_l1775_177511


namespace NUMINAMATH_CALUDE_log_cutting_problem_l1775_177563

/-- Represents the number of cuts needed to divide logs into 1-meter pieces -/
def num_cuts (x y : ℕ) : ℕ := 2 * x + 3 * y

theorem log_cutting_problem :
  ∃ (x y : ℕ),
    x + y = 30 ∧
    3 * x + 4 * y = 100 ∧
    num_cuts x y = 70 :=
by sorry

end NUMINAMATH_CALUDE_log_cutting_problem_l1775_177563


namespace NUMINAMATH_CALUDE_ratio_of_numbers_l1775_177556

theorem ratio_of_numbers (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) 
  (sum_diff : a + b = 7 * (a - b)) (product : a * b = 24) : a / b = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_numbers_l1775_177556


namespace NUMINAMATH_CALUDE_parrot_count_theorem_l1775_177573

/-- Represents the types of parrots -/
inductive ParrotType
  | Green
  | Yellow
  | Mottled

/-- Represents the behavior of parrots -/
def ParrotBehavior : ParrotType → Bool → Bool
  | ParrotType.Green, _ => true
  | ParrotType.Yellow, _ => false
  | ParrotType.Mottled, b => b

theorem parrot_count_theorem 
  (total_parrots : Nat)
  (green_count : Nat)
  (yellow_count : Nat)
  (mottled_count : Nat)
  (h_total : total_parrots = 100)
  (h_sum : green_count + yellow_count + mottled_count = total_parrots)
  (h_first_statement : green_count + (mottled_count / 2) = 50)
  (h_second_statement : yellow_count + (mottled_count / 2) = 50)
  : yellow_count = green_count :=
by sorry

end NUMINAMATH_CALUDE_parrot_count_theorem_l1775_177573


namespace NUMINAMATH_CALUDE_predicted_weight_for_178cm_l1775_177555

/-- Regression equation for weight prediction based on height -/
def weight_prediction (height : ℝ) : ℝ := 0.72 * height - 58.2

/-- Theorem: The predicted weight for a person with height 178 cm is 69.96 kg -/
theorem predicted_weight_for_178cm :
  weight_prediction 178 = 69.96 := by sorry

end NUMINAMATH_CALUDE_predicted_weight_for_178cm_l1775_177555


namespace NUMINAMATH_CALUDE_cylinder_volume_ratio_l1775_177590

/-- The ratio of the larger volume to the smaller volume of cylinders formed by rolling a 6 × 10 rectangle -/
theorem cylinder_volume_ratio : 
  let rectangle_width : ℝ := 6
  let rectangle_length : ℝ := 10
  let cylinder1_volume := π * (rectangle_width / (2 * π))^2 * rectangle_length
  let cylinder2_volume := π * (rectangle_length / (2 * π))^2 * rectangle_width
  max cylinder1_volume cylinder2_volume / min cylinder1_volume cylinder2_volume = 5 / 3 := by
sorry

end NUMINAMATH_CALUDE_cylinder_volume_ratio_l1775_177590


namespace NUMINAMATH_CALUDE_clara_number_problem_l1775_177550

theorem clara_number_problem (x : ℝ) : 2 * x + 3 = 23 → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_clara_number_problem_l1775_177550


namespace NUMINAMATH_CALUDE_other_x_intercept_of_quadratic_l1775_177509

/-- Given a quadratic function with vertex (4, -2) and one x-intercept at (1, 0),
    the x-coordinate of the other x-intercept is 7. -/
theorem other_x_intercept_of_quadratic (a b c : ℝ) : 
  (∀ x, a * x^2 + b * x + c = -2 ↔ x = 4) →  -- vertex condition
  a * 1^2 + b * 1 + c = 0 →                 -- x-intercept condition
  ∃ x, x ≠ 1 ∧ a * x^2 + b * x + c = 0 ∧ x = 7 :=
by sorry

end NUMINAMATH_CALUDE_other_x_intercept_of_quadratic_l1775_177509


namespace NUMINAMATH_CALUDE_school_trip_theorem_l1775_177571

/-- Represents the number of people initially planned per bus -/
def initial_people_per_bus : ℕ := 28

/-- Represents the number of students who couldn't get on the buses -/
def students_left_behind : ℕ := 13

/-- Represents the final number of people per bus -/
def final_people_per_bus : ℕ := 32

/-- Represents the number of empty seats per bus after redistribution -/
def empty_seats_per_bus : ℕ := 3

/-- Proves that the number of third-grade students is 125 and the number of buses rented is 4 -/
theorem school_trip_theorem :
  ∃ (num_students num_buses : ℕ),
    num_students = 125 ∧
    num_buses = 4 ∧
    num_students = initial_people_per_bus * num_buses + students_left_behind ∧
    num_students = final_people_per_bus * num_buses - empty_seats_per_bus * num_buses :=
by
  sorry

end NUMINAMATH_CALUDE_school_trip_theorem_l1775_177571


namespace NUMINAMATH_CALUDE_fencing_cost_per_meter_l1775_177589

/-- Proves that for a rectangular plot with given dimensions and total fencing cost,
    the cost per meter of fencing is as calculated. -/
theorem fencing_cost_per_meter
  (length : ℝ) (breadth : ℝ) (total_cost : ℝ)
  (h1 : length = 56)
  (h2 : breadth = 44)
  (h3 : total_cost = 5300) :
  total_cost / (2 * (length + breadth)) = 26.5 := by
  sorry

end NUMINAMATH_CALUDE_fencing_cost_per_meter_l1775_177589


namespace NUMINAMATH_CALUDE_percentage_problem_l1775_177532

theorem percentage_problem (p : ℝ) (h1 : 0.25 * 820 = (p/100) * 1500 - 20) : p = 15 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l1775_177532


namespace NUMINAMATH_CALUDE_blue_marble_difference_l1775_177580

theorem blue_marble_difference (jar1_total jar2_total : ℕ) : 
  jar1_total = jar2_total →                             -- Same number of marbles in each jar
  jar1_total = 10 * (jar1_total / 10) →                 -- Jar 1 ratio is 9:1 (blue:green)
  jar2_total = 9 * (jar2_total / 9) →                   -- Jar 2 ratio is 8:1 (blue:green)
  (jar1_total / 10) + (jar2_total / 9) = 95 →           -- Total green marbles is 95
  9 * (jar1_total / 10) - 8 * (jar2_total / 9) = 5 :=   -- Difference in blue marbles is 5
by sorry

end NUMINAMATH_CALUDE_blue_marble_difference_l1775_177580


namespace NUMINAMATH_CALUDE_inequality_solution_set_inequality_proof_l1775_177540

-- Define the function f(x) = |x-1|
def f (x : ℝ) : ℝ := |x - 1|

-- Theorem for part (1)
theorem inequality_solution_set (x : ℝ) :
  f x + f (x + 4) ≥ 8 ↔ x ≤ -5 ∨ x ≥ 3 := by
  sorry

-- Theorem for part (2)
theorem inequality_proof (a b : ℝ) (ha : a ≠ 0) (h1 : f (a + 1) < 1) (h2 : f (b + 1) < 1) :
  f (a * b) / |a| > f (b / a) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_inequality_proof_l1775_177540


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l1775_177559

theorem perfect_square_trinomial (x y : ℝ) :
  x^2 - x*y + (1/4)*y^2 = (x - (1/2)*y)^2 := by sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l1775_177559


namespace NUMINAMATH_CALUDE_simplify_expression_l1775_177566

theorem simplify_expression (x : ℝ) : (2*x - 5)*(x + 6) - (x + 4)*(2*x - 1) = -26 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1775_177566


namespace NUMINAMATH_CALUDE_symmetric_difference_of_A_and_B_l1775_177537

-- Define the sets A and B
def A : Set ℝ := {x | x ≥ -9/4}
def B : Set ℝ := {x | x < 0}

-- Define set difference
def setDifference (M N : Set ℝ) : Set ℝ := {x | x ∈ M ∧ x ∉ N}

-- Define symmetric difference
def symmetricDifference (M N : Set ℝ) : Set ℝ := 
  (setDifference M N) ∪ (setDifference N M)

-- Theorem statement
theorem symmetric_difference_of_A_and_B :
  symmetricDifference A B = {x | x ≥ 0 ∨ x < -9/4} := by
  sorry

end NUMINAMATH_CALUDE_symmetric_difference_of_A_and_B_l1775_177537


namespace NUMINAMATH_CALUDE_y_intercept_of_line_l1775_177516

def line (x : ℝ) : ℝ := x - 2

theorem y_intercept_of_line :
  line 0 = -2 := by sorry

end NUMINAMATH_CALUDE_y_intercept_of_line_l1775_177516


namespace NUMINAMATH_CALUDE_steak_meal_cost_l1775_177560

def tip : ℝ := 99
def num_steak_meals : ℕ := 2
def num_burgers : ℕ := 2
def burger_cost : ℝ := 3.5
def num_ice_cream : ℕ := 3
def ice_cream_cost : ℝ := 2
def remaining_money : ℝ := 38

theorem steak_meal_cost (steak_cost : ℝ) : 
  tip - (num_steak_meals * steak_cost + num_burgers * burger_cost + num_ice_cream * ice_cream_cost) = remaining_money → 
  steak_cost = 24 := by
sorry

end NUMINAMATH_CALUDE_steak_meal_cost_l1775_177560


namespace NUMINAMATH_CALUDE_triangle_count_relation_l1775_177518

/-- The number of non-overlapping triangles formed from 6 points when no three points are collinear -/
def n₀ : ℕ := 20

/-- The number of non-overlapping triangles formed from 6 points when exactly three points are collinear -/
def n₁ : ℕ := 19

/-- The number of non-overlapping triangles formed from 6 points when exactly four points are collinear -/
def n₂ : ℕ := 18

/-- Theorem stating the relationship between n₀, n₁, and n₂ -/
theorem triangle_count_relation : n₀ > n₁ ∧ n₁ > n₂ :=
by sorry

end NUMINAMATH_CALUDE_triangle_count_relation_l1775_177518


namespace NUMINAMATH_CALUDE_shorter_diagonal_length_l1775_177525

/-- Represents a rhombus with given properties -/
structure Rhombus where
  area : ℝ
  diagonal_ratio : ℝ × ℝ
  diagonal_side_relation : Bool

/-- Theorem: In a rhombus with area 150 and diagonal ratio 5:3, the shorter diagonal is 6√5 -/
theorem shorter_diagonal_length (R : Rhombus) 
    (h_area : R.area = 150)
    (h_ratio : R.diagonal_ratio = (5, 3))
    (h_relation : R.diagonal_side_relation = true) : 
  ∃ (d : ℝ), d = 6 * Real.sqrt 5 ∧ d = min (5 * (2 * Real.sqrt 5)) (3 * (2 * Real.sqrt 5)) :=
by sorry

end NUMINAMATH_CALUDE_shorter_diagonal_length_l1775_177525


namespace NUMINAMATH_CALUDE_complement_A_union_B_eq_B_l1775_177579

-- Define the sets A and B
def A : Set ℝ := {x | x < -1 ∨ x > 1}
def B : Set ℝ := {x | x ≥ -1}

-- State the theorem
theorem complement_A_union_B_eq_B : (Aᶜ ∪ B) = B := by sorry

end NUMINAMATH_CALUDE_complement_A_union_B_eq_B_l1775_177579


namespace NUMINAMATH_CALUDE_ratio_in_interval_l1775_177542

theorem ratio_in_interval (a : Fin 10 → ℕ) (h : ∀ i, a i ≤ 91) :
  ∃ i j, i ≠ j ∧ 2/3 ≤ (a i : ℚ) / (a j : ℚ) ∧ (a i : ℚ) / (a j : ℚ) ≤ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_in_interval_l1775_177542


namespace NUMINAMATH_CALUDE_m_range_theorem_l1775_177594

/-- Proposition p: x^2 - mx + 1 = 0 has no real solutions -/
def p (m : ℝ) : Prop := ∀ x : ℝ, x^2 - m*x + 1 ≠ 0

/-- Proposition q: x^2/m + y^2 = 1 has its foci on the x-axis -/
def q (m : ℝ) : Prop := m > 1

/-- The range of real values for m given the conditions -/
def m_range (m : ℝ) : Prop := (-2 < m ∧ m ≤ 1) ∨ m ≥ 2

/-- Theorem stating the range of m given the conditions -/
theorem m_range_theorem (m : ℝ) (h1 : p m ∨ q m) (h2 : ¬(p m ∧ q m)) : m_range m := by
  sorry

end NUMINAMATH_CALUDE_m_range_theorem_l1775_177594


namespace NUMINAMATH_CALUDE_sum_of_reciprocal_roots_l1775_177533

theorem sum_of_reciprocal_roots (a b : ℝ) : 
  (6 * a^2 + 5 * a + 7 = 0) →
  (6 * b^2 + 5 * b + 7 = 0) →
  a ≠ b →
  a ≠ 0 →
  b ≠ 0 →
  (1 / a) + (1 / b) = -5 / 7 := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocal_roots_l1775_177533


namespace NUMINAMATH_CALUDE_gumball_machine_total_l1775_177541

/-- Represents the number of gumballs of each color in the machine -/
structure GumballMachine where
  red : ℕ
  blue : ℕ
  green : ℕ

/-- Defines the properties of the gumball machine -/
def validGumballMachine (m : GumballMachine) : Prop :=
  m.red = 16 ∧
  m.blue = m.red / 2 ∧
  m.green = 4 * m.blue

/-- Calculates the total number of gumballs in the machine -/
def totalGumballs (m : GumballMachine) : ℕ :=
  m.red + m.blue + m.green

/-- Theorem stating that a valid gumball machine has 56 gumballs in total -/
theorem gumball_machine_total (m : GumballMachine) 
  (h : validGumballMachine m) : totalGumballs m = 56 := by
  sorry

end NUMINAMATH_CALUDE_gumball_machine_total_l1775_177541


namespace NUMINAMATH_CALUDE_binary_101111011_equals_379_l1775_177586

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

def binary_101111011 : List Bool := [true, true, false, true, true, true, true, false, true]

theorem binary_101111011_equals_379 :
  binary_to_decimal binary_101111011 = 379 := by
  sorry

end NUMINAMATH_CALUDE_binary_101111011_equals_379_l1775_177586


namespace NUMINAMATH_CALUDE_exactly_one_correct_l1775_177539

/-- Represents a geometric statement --/
inductive GeometricStatement
  | complement_acute : GeometricStatement
  | equal_vertical : GeometricStatement
  | unique_parallel : GeometricStatement
  | perpendicular_distance : GeometricStatement
  | corresponding_angles : GeometricStatement

/-- Checks if a geometric statement is correct --/
def is_correct (s : GeometricStatement) : Prop :=
  match s with
  | GeometricStatement.complement_acute => True
  | _ => False

/-- The list of all geometric statements --/
def all_statements : List GeometricStatement :=
  [GeometricStatement.complement_acute,
   GeometricStatement.equal_vertical,
   GeometricStatement.unique_parallel,
   GeometricStatement.perpendicular_distance,
   GeometricStatement.corresponding_angles]

/-- Theorem stating that exactly one statement is correct --/
theorem exactly_one_correct :
  ∃! (s : GeometricStatement), s ∈ all_statements ∧ is_correct s :=
sorry

end NUMINAMATH_CALUDE_exactly_one_correct_l1775_177539


namespace NUMINAMATH_CALUDE_sqrt_ax_cube_l1775_177595

theorem sqrt_ax_cube (a x : ℝ) (ha : a < 0) : 
  Real.sqrt (a * x^3) = -x * Real.sqrt (a * x) :=
sorry

end NUMINAMATH_CALUDE_sqrt_ax_cube_l1775_177595


namespace NUMINAMATH_CALUDE_tax_free_items_cost_l1775_177582

theorem tax_free_items_cost 
  (total_paid : ℝ)
  (sales_tax : ℝ)
  (tax_rate : ℝ)
  (h1 : total_paid = 30)
  (h2 : sales_tax = 1.28)
  (h3 : tax_rate = 0.08)
  : ∃ (tax_free_cost : ℝ), tax_free_cost = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_tax_free_items_cost_l1775_177582


namespace NUMINAMATH_CALUDE_sweetsies_leftover_l1775_177527

theorem sweetsies_leftover (m : ℕ) : 
  (∃ k : ℕ, m = 8 * k + 5) →  -- One bag leaves 5 when divided by 8
  (∃ l : ℕ, 4 * m = 8 * l + 4) -- Four bags leave 4 when divided by 8
  := by sorry

end NUMINAMATH_CALUDE_sweetsies_leftover_l1775_177527


namespace NUMINAMATH_CALUDE_min_fence_length_l1775_177512

theorem min_fence_length (w : ℝ) (l : ℝ) (area : ℝ) (perimeter : ℝ) : 
  w > 0 →
  l = 2 * w →
  area = l * w →
  area ≥ 500 →
  perimeter = 2 * (l + w) →
  perimeter ≥ 96 :=
by sorry

end NUMINAMATH_CALUDE_min_fence_length_l1775_177512


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l1775_177506

theorem absolute_value_inequality (x : ℝ) : 
  (|5 - x| < 6) ↔ (-1 < x ∧ x < 11) := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l1775_177506


namespace NUMINAMATH_CALUDE_work_completion_time_l1775_177543

theorem work_completion_time (b a_and_b : ℝ) (hb : b = 12) (hab : a_and_b = 3) : 
  let a := (1 / a_and_b - 1 / b)⁻¹
  a = 4 := by sorry

end NUMINAMATH_CALUDE_work_completion_time_l1775_177543


namespace NUMINAMATH_CALUDE_two_fifths_in_nine_thirds_l1775_177535

theorem two_fifths_in_nine_thirds : (9 / 3) / (2 / 5) = 15 / 2 := by
  sorry

end NUMINAMATH_CALUDE_two_fifths_in_nine_thirds_l1775_177535


namespace NUMINAMATH_CALUDE_inequality_proof_l1775_177548

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (1 / (x^2 + y^2)) + (1 / x^2) + (1 / y^2) ≥ 10 / ((x + y)^2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1775_177548


namespace NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l1775_177517

theorem smallest_prime_divisor_of_sum (n : ℕ) (m : ℕ) :
  2 = Nat.minFac (3^25 + 11^19) := by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l1775_177517


namespace NUMINAMATH_CALUDE_negation_existential_quadratic_l1775_177531

theorem negation_existential_quadratic (x : ℝ) :
  (¬ ∃ x, x^2 - x > 0) ↔ (∀ x, x^2 - x ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_existential_quadratic_l1775_177531


namespace NUMINAMATH_CALUDE_nested_subtraction_simplification_l1775_177538

theorem nested_subtraction_simplification (x : ℝ) :
  1 - (2 - (3 - (4 - (5 - x)))) = 3 - x := by
  sorry

end NUMINAMATH_CALUDE_nested_subtraction_simplification_l1775_177538


namespace NUMINAMATH_CALUDE_calculation_difference_l1775_177584

def correct_calculation : ℤ := 12 - (3 + 2) * 2

def incorrect_calculation : ℤ := 12 - 3 + 2 * 2

theorem calculation_difference :
  correct_calculation - incorrect_calculation = -11 := by
  sorry

end NUMINAMATH_CALUDE_calculation_difference_l1775_177584


namespace NUMINAMATH_CALUDE_one_common_sale_day_in_july_l1775_177554

def is_bookstore_sale_day (day : Nat) : Prop :=
  day ≤ 31 ∧ day % 5 = 0

def is_shoe_store_sale_day (day : Nat) : Prop :=
  day ≤ 31 ∧ ∃ k : Nat, day = 3 + 7 * k

def both_stores_sale_day (day : Nat) : Prop :=
  is_bookstore_sale_day day ∧ is_shoe_store_sale_day day

theorem one_common_sale_day_in_july :
  ∃! day : Nat, both_stores_sale_day day :=
sorry

end NUMINAMATH_CALUDE_one_common_sale_day_in_july_l1775_177554


namespace NUMINAMATH_CALUDE_split_sum_equals_capacity_l1775_177569

/-- The capacity of a pile with n stones -/
def capacity (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The sum of products obtained by splitting n stones -/
def split_sum (n : ℕ) : ℕ := sorry

theorem split_sum_equals_capacity :
  split_sum 2019 = capacity 2019 :=
sorry

end NUMINAMATH_CALUDE_split_sum_equals_capacity_l1775_177569


namespace NUMINAMATH_CALUDE_rectangular_field_area_l1775_177545

/-- Theorem: Area of a rectangular field -/
theorem rectangular_field_area (width : ℝ) : 
  (width ≥ 0) →
  (16 * width + 54 = 22 * width) → 
  (16 * width = 144) :=
by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l1775_177545


namespace NUMINAMATH_CALUDE_initial_money_calculation_l1775_177530

theorem initial_money_calculation (initial_money : ℚ) : 
  (initial_money * (1 - 1/3) * (1 - 1/5) * (1 - 1/4) = 100) → 
  initial_money = 250 := by
sorry

end NUMINAMATH_CALUDE_initial_money_calculation_l1775_177530


namespace NUMINAMATH_CALUDE_no_nontrivial_integer_solution_l1775_177578

theorem no_nontrivial_integer_solution (a b c d : ℤ) :
  6 * (6 * a ^ 2 + 3 * b ^ 2 + c ^ 2) = 5 * d ^ 2 → a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_nontrivial_integer_solution_l1775_177578


namespace NUMINAMATH_CALUDE_smallest_multiple_of_6_and_15_l1775_177577

theorem smallest_multiple_of_6_and_15 : ∃ (a : ℕ), a > 0 ∧ 6 ∣ a ∧ 15 ∣ a ∧ ∀ (b : ℕ), b > 0 → 6 ∣ b → 15 ∣ b → a ≤ b := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_6_and_15_l1775_177577


namespace NUMINAMATH_CALUDE_complex_in_first_quadrant_l1775_177581

theorem complex_in_first_quadrant (z : ℂ) : z = Complex.mk (Real.sqrt 3) 2 → z.re > 0 ∧ z.im > 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_in_first_quadrant_l1775_177581


namespace NUMINAMATH_CALUDE_fish_given_by_sister_l1775_177598

/-- Given that Mrs. Sheridan initially had 22 fish and now has 69 fish
    after receiving fish from her sister, prove that the number of fish
    her sister gave her is 47. -/
theorem fish_given_by_sister
  (initial_fish : ℕ)
  (final_fish : ℕ)
  (h1 : initial_fish = 22)
  (h2 : final_fish = 69) :
  final_fish - initial_fish = 47 := by
  sorry

end NUMINAMATH_CALUDE_fish_given_by_sister_l1775_177598


namespace NUMINAMATH_CALUDE_shortest_ant_path_l1775_177553

/-- Represents a grid of square tiles -/
structure TileGrid where
  rows : ℕ
  columns : ℕ
  tileSize : ℝ

/-- Represents the path of an ant on a tile grid -/
def antPath (grid : TileGrid) : ℝ :=
  grid.tileSize * (grid.rows + grid.columns - 2)

/-- Theorem stating the shortest path for an ant on a 5x3 grid with tile size 10 -/
theorem shortest_ant_path :
  let grid : TileGrid := ⟨5, 3, 10⟩
  antPath grid = 80 := by
  sorry

#check shortest_ant_path

end NUMINAMATH_CALUDE_shortest_ant_path_l1775_177553


namespace NUMINAMATH_CALUDE_little_red_journey_l1775_177565

-- Define the parameters
def total_distance : ℝ := 1500  -- in meters
def total_time : ℝ := 18  -- in minutes
def uphill_speed : ℝ := 2  -- in km/h
def downhill_speed : ℝ := 3  -- in km/h

-- Define variables for uphill and downhill time
variable (x y : ℝ)

-- Theorem statement
theorem little_red_journey :
  (x + y = total_time) ∧
  ((uphill_speed / 60) * x + (downhill_speed / 60) * y = total_distance / 1000) :=
by sorry

end NUMINAMATH_CALUDE_little_red_journey_l1775_177565


namespace NUMINAMATH_CALUDE_ceiling_sum_sqrt_l1775_177524

theorem ceiling_sum_sqrt : ⌈Real.sqrt 3⌉ + ⌈Real.sqrt 33⌉ + ⌈Real.sqrt 333⌉ = 27 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_sum_sqrt_l1775_177524


namespace NUMINAMATH_CALUDE_part_one_part_two_l1775_177521

-- Define the new operation *
def star (a b : ℚ) : ℚ := 4 * a * b

-- Theorem for part (1)
theorem part_one : star 3 (-4) = -48 := by sorry

-- Theorem for part (2)
theorem part_two : star (-2) (star 6 3) = -576 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1775_177521


namespace NUMINAMATH_CALUDE_cup_saucer_prices_l1775_177557

/-- The price of a cup in rubles -/
def cup_price : ℕ := 1370

/-- The price of a saucer in rubles -/
def saucer_price : ℕ := 1130

/-- The sum of the prices of one cup and one saucer is 2500 rubles -/
axiom sum_cup_saucer : cup_price + saucer_price = 2500

/-- The sum of the prices of four cups and three saucers is 8870 rubles -/
axiom sum_four_cups_three_saucers : 4 * cup_price + 3 * saucer_price = 8870

/-- 
Theorem: The price of a cup is 1370 rubles and the price of a saucer is 1130 rubles
-/
theorem cup_saucer_prices : cup_price = 1370 ∧ saucer_price = 1130 := by
  sorry

end NUMINAMATH_CALUDE_cup_saucer_prices_l1775_177557


namespace NUMINAMATH_CALUDE_book_cost_solution_l1775_177576

/-- Represents the cost of books problem --/
def BookCostProblem (initial_budget : ℚ) (books_per_series : ℕ) (series_bought : ℕ) (money_left : ℚ) (tax_rate : ℚ) : Prop :=
  let total_books := books_per_series * series_bought
  let money_spent := initial_budget - money_left
  let pre_tax_total := money_spent / (1 + tax_rate)
  let book_cost := pre_tax_total / total_books
  book_cost = 60 / 11

/-- Theorem stating the solution to the book cost problem --/
theorem book_cost_solution :
  BookCostProblem 200 8 3 56 (1/10) :=
sorry

end NUMINAMATH_CALUDE_book_cost_solution_l1775_177576


namespace NUMINAMATH_CALUDE_cosine_sum_120_l1775_177544

theorem cosine_sum_120 (α : ℝ) : 
  Real.cos (α - 120 * π / 180) + Real.cos α + Real.cos (α + 120 * π / 180) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sum_120_l1775_177544


namespace NUMINAMATH_CALUDE_shopping_theorem_l1775_177502

def shopping_problem (initial_amount discount_rate tax_rate: ℝ)
  (sweater t_shirt shoes jeans scarf: ℝ) : Prop :=
  let discounted_t_shirt := t_shirt * (1 - discount_rate)
  let subtotal := sweater + discounted_t_shirt + shoes + jeans + scarf
  let total_with_tax := subtotal * (1 + tax_rate)
  let remaining := initial_amount - total_with_tax
  remaining = 30.11

theorem shopping_theorem :
  shopping_problem 200 0.1 0.05 36 12 45 52 18 :=
by sorry

end NUMINAMATH_CALUDE_shopping_theorem_l1775_177502


namespace NUMINAMATH_CALUDE_triangle_area_l1775_177562

theorem triangle_area (a b c : ℝ) (h1 : a = 13) (h2 : b = 84) (h3 : c = 85) :
  (1/2) * a * b = 546 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l1775_177562


namespace NUMINAMATH_CALUDE_orchestra_females_count_l1775_177585

/-- The number of females in the orchestra -/
def females_in_orchestra : ℕ := 12

theorem orchestra_females_count :
  let males_in_orchestra : ℕ := 11
  let choir_size : ℕ := 12 + 17
  let total_musicians : ℕ := 98
  females_in_orchestra = 
    (total_musicians - choir_size - males_in_orchestra - 2 * males_in_orchestra) / 3 :=
by sorry

end NUMINAMATH_CALUDE_orchestra_females_count_l1775_177585


namespace NUMINAMATH_CALUDE_divisors_product_prime_factors_l1775_177570

theorem divisors_product_prime_factors :
  let divisors : List Nat := [1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60]
  let A : Nat := divisors.prod
  (Nat.factors A).toFinset.card = 3 := by
  sorry

end NUMINAMATH_CALUDE_divisors_product_prime_factors_l1775_177570


namespace NUMINAMATH_CALUDE_intersection_point_inside_circle_l1775_177523

theorem intersection_point_inside_circle (a : ℝ) : 
  let line1 : ℝ → ℝ := λ x => x + 2 * a
  let line2 : ℝ → ℝ := λ x => 2 * x + a + 1
  let P : ℝ × ℝ := (a - 1, 3 * a - 1)
  (∀ x y, y = line1 x ∧ y = line2 x → (x, y) = P) →
  P.1^2 + P.2^2 < 4 →
  -1/5 < a ∧ a < 1 :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_inside_circle_l1775_177523


namespace NUMINAMATH_CALUDE_simplify_expression_l1775_177593

theorem simplify_expression (a b : ℝ) : 
  (30*a + 70*b) + (15*a + 40*b) - (12*a + 55*b) + (5*a - 10*b) = 38*a + 45*b := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1775_177593


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l1775_177534

/-- A quadratic function with vertex at (-3, 0) passing through (2, -64) has a = -64/25 -/
theorem quadratic_coefficient (a b c : ℝ) : 
  (∀ x y, y = a * x^2 + b * x + c) → -- quadratic function
  (0 = a * (-3)^2 + b * (-3) + c) → -- vertex at (-3, 0)
  (-64 = a * 2^2 + b * 2 + c) → -- passes through (2, -64)
  a = -64/25 := by
sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l1775_177534


namespace NUMINAMATH_CALUDE_smallest_in_consecutive_odd_integers_l1775_177507

/-- A set of consecutive odd integers -/
def ConsecutiveOddIntegers := Set ℤ

/-- The median of a set of integers -/
def median (s : Set ℤ) : ℤ := sorry

/-- The smallest element in a set of integers -/
def smallest (s : Set ℤ) : ℤ := sorry

/-- The largest element in a set of integers -/
def largest (s : Set ℤ) : ℤ := sorry

theorem smallest_in_consecutive_odd_integers 
  (S : ConsecutiveOddIntegers) 
  (h_median : median S = 152) 
  (h_largest : largest S = 163) : 
  smallest S = 138 := by sorry

end NUMINAMATH_CALUDE_smallest_in_consecutive_odd_integers_l1775_177507


namespace NUMINAMATH_CALUDE_f_monotone_range_of_a_l1775_177508

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + 5*x + 6

-- Define the property of being monotonically increasing on an interval
def MonotonicallyIncreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

-- State the theorem
theorem f_monotone_range_of_a :
  {a : ℝ | MonotonicallyIncreasing (f a) 1 3} = {a | a ≤ -3 ∨ a ≥ -3} :=
by sorry

end NUMINAMATH_CALUDE_f_monotone_range_of_a_l1775_177508


namespace NUMINAMATH_CALUDE_dog_treat_cost_is_six_l1775_177546

/-- The cost of dog treats for a month -/
def dog_treat_cost (treats_per_day : ℕ) (cost_per_treat : ℚ) (days_in_month : ℕ) : ℚ :=
  (treats_per_day : ℚ) * cost_per_treat * (days_in_month : ℚ)

/-- Theorem: The cost of dog treats for a month under given conditions is $6 -/
theorem dog_treat_cost_is_six :
  dog_treat_cost 2 (1/10) 30 = 6 := by
sorry

end NUMINAMATH_CALUDE_dog_treat_cost_is_six_l1775_177546


namespace NUMINAMATH_CALUDE_specific_arrangement_probability_l1775_177597

def num_red_lamps : ℕ := 4
def num_blue_lamps : ℕ := 4
def num_lamps_on : ℕ := 4

def probability_specific_arrangement : ℚ := 3/49

theorem specific_arrangement_probability :
  let total_lamps := num_red_lamps + num_blue_lamps
  let total_arrangements := (total_lamps.choose num_red_lamps) * (total_lamps.choose num_lamps_on)
  let favorable_outcomes := (total_lamps - 2).choose (num_red_lamps - 1) * (total_lamps - 2).choose (num_lamps_on - 1)
  (favorable_outcomes : ℚ) / total_arrangements = probability_specific_arrangement :=
sorry

end NUMINAMATH_CALUDE_specific_arrangement_probability_l1775_177597


namespace NUMINAMATH_CALUDE_employee_salaries_exist_l1775_177504

/-- Proves the existence of salaries for three employees satisfying given conditions --/
theorem employee_salaries_exist :
  ∃ (m n p : ℝ),
    (m + n + p = 1750) ∧
    (m = 1.3 * n) ∧
    (p = 0.9 * (m + n)) :=
by sorry

end NUMINAMATH_CALUDE_employee_salaries_exist_l1775_177504


namespace NUMINAMATH_CALUDE_unique_solution_logarithmic_equation_l1775_177513

theorem unique_solution_logarithmic_equation (a b x : ℝ) :
  a > 0 ∧ b > 0 ∧ x > 1 ∧
  9 * (Real.log x / Real.log a)^2 + 5 * (Real.log x / Real.log b)^2 = 17 ∧
  (Real.log b / Real.log a) * (Real.log a / Real.log b) = 2 →
  a = Real.exp (Real.sqrt 2 * Real.log 10) ∧ b = 10 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_logarithmic_equation_l1775_177513


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_product_l1775_177522

theorem arithmetic_geometric_sequence_product (a₁ a₂ b₁ b₂ b₃ : ℝ) : 
  ((-9 : ℝ) < a₁ ∧ a₁ < a₂ ∧ a₂ < (-1 : ℝ)) →  -- Arithmetic sequence condition
  ((-9 : ℝ) < b₁ ∧ b₁ < b₂ ∧ b₂ < b₃ ∧ b₃ < (-1 : ℝ)) →  -- Geometric sequence condition
  (a₂ - a₁ = a₁ - (-9 : ℝ)) →  -- Arithmetic sequence property
  (b₂ * b₂ = b₁ * b₃) →  -- Geometric sequence property
  (b₂ * (a₂ - a₁) = (-8 : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_product_l1775_177522


namespace NUMINAMATH_CALUDE_sum_of_first_four_terms_l1775_177599

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem sum_of_first_four_terms 
  (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a) 
  (h_a2 : a 2 = 9) 
  (h_a5 : a 5 = 243) : 
  (a 1 + a 2 + a 3 + a 4 = 120) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_first_four_terms_l1775_177599


namespace NUMINAMATH_CALUDE_digit_five_minus_nine_in_book_pages_l1775_177575

/-- Counts the occurrences of a digit in a number -/
def countDigit (d : Nat) (n : Nat) : Nat :=
  sorry

/-- Counts the occurrences of a digit in a range of numbers -/
def countDigitInRange (d : Nat) (start finish : Nat) : Nat :=
  sorry

theorem digit_five_minus_nine_in_book_pages : 
  ∀ (n : Nat), n = 599 →
  (countDigitInRange 5 1 n) - (countDigitInRange 9 1 n) = 100 := by
  sorry

end NUMINAMATH_CALUDE_digit_five_minus_nine_in_book_pages_l1775_177575


namespace NUMINAMATH_CALUDE_algebraic_equality_l1775_177572

theorem algebraic_equality (a b c k m n : ℝ) 
  (h1 : b^2 - n^2 = a^2 - k^2) 
  (h2 : a^2 - k^2 = c^2 - m^2) : 
  (b*m - c*n)/(a - k) + (c*k - a*m)/(b - n) + (a*n - b*k)/(c - m) = 0 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_equality_l1775_177572


namespace NUMINAMATH_CALUDE_work_completion_time_l1775_177505

/-- The number of days it takes B to complete the work alone -/
def b_days : ℝ := 20

/-- The fraction of work left after A and B work together for 3 days -/
def work_left : ℝ := 0.65

/-- The number of days A and B work together -/
def days_together : ℝ := 3

/-- The number of days it takes A to complete the work alone -/
def a_days : ℝ := 15

theorem work_completion_time :
  ∃ (x : ℝ), x > 0 ∧ 
  days_together * (1 / x + 1 / b_days) = 1 - work_left ∧
  x = a_days := by sorry

end NUMINAMATH_CALUDE_work_completion_time_l1775_177505


namespace NUMINAMATH_CALUDE_f_properties_l1775_177536

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a / (x + 1)

def tangent_slope (a : ℝ) (x : ℝ) : ℝ :=
  1 / x - a / ((x + 1) ^ 2)

def critical_point (a : ℝ) : ℝ :=
  (a - 2 + Real.sqrt ((a - 2) ^ 2 + 4)) / 2

theorem f_properties (a : ℝ) (h : a ≥ 0) :
  (tangent_slope 3 1 = 1/4) ∧
  (∀ x > 0, f a x ≤ (2016 - a) * x^3 + (x^2 + a - 1) / (x + 1) →
    (∃ x > 0, (tangent_slope a x = 0) → 4 < a ∧ a ≤ 2016)) :=
sorry

end

end NUMINAMATH_CALUDE_f_properties_l1775_177536


namespace NUMINAMATH_CALUDE_trigonometric_expression_equality_l1775_177519

theorem trigonometric_expression_equality (θ : Real) (h : Real.tan θ = 3) :
  (1 + Real.cos θ) / Real.sin θ + Real.sin θ / (1 - Real.cos θ) = 17 * (Real.sqrt 10 + 1) / 24 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_equality_l1775_177519


namespace NUMINAMATH_CALUDE_polynomial_division_l1775_177558

theorem polynomial_division (x : ℝ) (h : x ≠ 0) :
  (x^4 - 3*x^2) / x^2 = x^2 - 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_l1775_177558


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l1775_177588

theorem absolute_value_inequality (x y ε : ℝ) (h_pos : ε > 0) 
  (hx : |x - 2| < ε) (hy : |y - 2| < ε) : |x - y| < 2 * ε := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l1775_177588


namespace NUMINAMATH_CALUDE_complex_numbers_on_circle_l1775_177561

theorem complex_numbers_on_circle 
  (a₁ a₂ a₃ a₄ a₅ : ℂ) 
  (h_nonzero : a₁ ≠ 0 ∧ a₂ ≠ 0 ∧ a₃ ≠ 0 ∧ a₄ ≠ 0 ∧ a₅ ≠ 0)
  (h_ratio : a₂ / a₁ = a₃ / a₂ ∧ a₃ / a₂ = a₄ / a₃ ∧ a₄ / a₃ = a₅ / a₄)
  (S : ℝ)
  (h_sum : a₁ + a₂ + a₃ + a₄ + a₅ - 4 * (1 / a₁ + 1 / a₂ + 1 / a₃ + 1 / a₄ + 1 / a₅) = S)
  (h_S_bound : abs S ≤ 2) :
  ∃ (r : ℝ), r > 0 ∧ Complex.abs a₁ = r ∧ Complex.abs a₂ = r ∧ Complex.abs a₃ = r ∧ Complex.abs a₄ = r ∧ Complex.abs a₅ = r :=
by sorry

end NUMINAMATH_CALUDE_complex_numbers_on_circle_l1775_177561


namespace NUMINAMATH_CALUDE_gcd_2_powers_l1775_177587

theorem gcd_2_powers : 
  Nat.gcd (2^2025 - 1) (2^2016 - 1) = 2^9 - 1 := by
sorry

end NUMINAMATH_CALUDE_gcd_2_powers_l1775_177587


namespace NUMINAMATH_CALUDE_parabola_x_intercepts_l1775_177591

/-- The number of x-intercepts for the parabola x = -3y^2 + 2y + 3 -/
theorem parabola_x_intercepts : ∃! x : ℝ, ∃ y : ℝ, x = -3 * y^2 + 2 * y + 3 ∧ y = 0 := by
  sorry

end NUMINAMATH_CALUDE_parabola_x_intercepts_l1775_177591


namespace NUMINAMATH_CALUDE_max_xy_constraint_l1775_177500

theorem max_xy_constraint (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 5 * x + 8 * y = 65) :
  x * y ≤ 25 ∧ ∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ 5 * x₀ + 8 * y₀ = 65 ∧ x₀ * y₀ = 25 := by
  sorry

end NUMINAMATH_CALUDE_max_xy_constraint_l1775_177500


namespace NUMINAMATH_CALUDE_rain_gauge_calculation_l1775_177583

theorem rain_gauge_calculation : 
  let initial_water : ℝ := 2
  let rate_2pm_to_4pm : ℝ := 4
  let rate_4pm_to_7pm : ℝ := 3
  let rate_7pm_to_9pm : ℝ := 0.5
  let duration_2pm_to_4pm : ℝ := 2
  let duration_4pm_to_7pm : ℝ := 3
  let duration_7pm_to_9pm : ℝ := 2
  
  initial_water + 
  (rate_2pm_to_4pm * duration_2pm_to_4pm) + 
  (rate_4pm_to_7pm * duration_4pm_to_7pm) + 
  (rate_7pm_to_9pm * duration_7pm_to_9pm) = 20 := by
sorry

end NUMINAMATH_CALUDE_rain_gauge_calculation_l1775_177583


namespace NUMINAMATH_CALUDE_connie_marbles_proof_l1775_177551

/-- The number of marbles Connie started with -/
def initial_marbles : ℕ := 776

/-- The number of marbles Connie gave to Juan -/
def marbles_given : ℕ := 183

/-- The number of marbles Connie has left -/
def remaining_marbles : ℕ := initial_marbles - marbles_given

theorem connie_marbles_proof : remaining_marbles = 593 := by
  sorry

end NUMINAMATH_CALUDE_connie_marbles_proof_l1775_177551


namespace NUMINAMATH_CALUDE_vegetarian_eaters_l1775_177528

theorem vegetarian_eaters (only_veg : ℕ) (only_non_veg : ℕ) (both : ℕ) :
  only_veg = 13 →
  only_non_veg = 8 →
  both = 6 →
  only_veg + both = 19 :=
by
  sorry

end NUMINAMATH_CALUDE_vegetarian_eaters_l1775_177528


namespace NUMINAMATH_CALUDE_car_hire_cost_for_b_l1775_177520

theorem car_hire_cost_for_b (total_cost : ℚ) (time_a time_b time_c : ℚ) : 
  total_cost = 520 →
  time_a = 7 →
  time_b = 8 →
  time_c = 11 →
  time_b / (time_a + time_b + time_c) * total_cost = 160 := by
  sorry

end NUMINAMATH_CALUDE_car_hire_cost_for_b_l1775_177520


namespace NUMINAMATH_CALUDE_no_solution_absolute_value_equation_l1775_177568

theorem no_solution_absolute_value_equation :
  ∀ x : ℝ, |-3 * x| + 5 ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_no_solution_absolute_value_equation_l1775_177568


namespace NUMINAMATH_CALUDE_both_companies_participate_both_will_participate_l1775_177549

/-- Represents a company in country A --/
structure Company where
  expectedIncome : ℝ
  investmentCost : ℝ

/-- The market conditions for the new technology development --/
structure MarketConditions where
  V : ℝ  -- Income if developed alone
  α : ℝ  -- Probability of success
  IC : ℝ  -- Investment cost
  h1 : 0 < α
  h2 : α < 1

/-- Calculate the expected income for a company when both participate --/
def expectedIncomeBothParticipate (m : MarketConditions) : ℝ :=
  m.α * (1 - m.α) * m.V + 0.5 * m.α^2 * m.V

/-- Theorem: Condition for both companies to participate --/
theorem both_companies_participate (m : MarketConditions) :
  expectedIncomeBothParticipate m - m.IC ≥ 0 ↔
  m.α * (1 - m.α) * m.V + 0.5 * m.α^2 * m.V - m.IC ≥ 0 := by
  sorry

/-- Function to determine if a company will participate --/
def willParticipate (c : Company) (m : MarketConditions) : Prop :=
  c.expectedIncome - c.investmentCost ≥ 0

/-- Theorem: Both companies will participate if the condition is met --/
theorem both_will_participate (c1 c2 : Company) (m : MarketConditions)
  (h : expectedIncomeBothParticipate m - m.IC ≥ 0) :
  willParticipate c1 m ∧ willParticipate c2 m := by
  sorry

end NUMINAMATH_CALUDE_both_companies_participate_both_will_participate_l1775_177549


namespace NUMINAMATH_CALUDE_lines_parallel_iff_same_slope_diff_intercept_l1775_177515

/-- Two lines in slope-intercept form are parallel if and only if they have the same slope and different y-intercepts -/
theorem lines_parallel_iff_same_slope_diff_intercept (k₁ k₂ l₁ l₂ : ℝ) :
  (∀ x y : ℝ, y = k₁ * x + l₁ ∨ y = k₂ * x + l₂) →
  (∀ x y : ℝ, y = k₁ * x + l₁ → y = k₂ * x + l₂ → False) ↔ k₁ = k₂ ∧ l₁ ≠ l₂ :=
by sorry

end NUMINAMATH_CALUDE_lines_parallel_iff_same_slope_diff_intercept_l1775_177515


namespace NUMINAMATH_CALUDE_line_intercepts_l1775_177526

/-- The equation of the line -/
def line_equation (x y : ℝ) : Prop := 3 * x - y + 6 = 0

/-- The y-intercept of the line -/
def y_intercept : ℝ := 6

/-- The x-intercept of the line -/
def x_intercept : ℝ := -2

/-- Theorem stating that the y-intercept and x-intercept are correct for the given line equation -/
theorem line_intercepts :
  line_equation 0 y_intercept ∧ line_equation x_intercept 0 :=
sorry

end NUMINAMATH_CALUDE_line_intercepts_l1775_177526


namespace NUMINAMATH_CALUDE_triangle_problem_l1775_177529

theorem triangle_problem (a b c A B C : Real) (h1 : (2 * a - b) / c = Real.cos B / Real.cos C) :
  let f := fun x => 2 * Real.sin x * Real.cos x * Real.cos C + 2 * Real.sin x * Real.sin x * Real.sin C - Real.sqrt 3 / 2
  C = π / 3 ∧ Set.Icc (f 0) (f (π / 2)) = Set.Icc (-(Real.sqrt 3) / 2) 1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l1775_177529


namespace NUMINAMATH_CALUDE_prime_pair_sum_cube_difference_l1775_177552

theorem prime_pair_sum_cube_difference (p q : ℕ) : 
  Prime p ∧ Prime q ∧ p + q = (p - q)^3 → (p = 5 ∧ q = 3) := by
  sorry

end NUMINAMATH_CALUDE_prime_pair_sum_cube_difference_l1775_177552
