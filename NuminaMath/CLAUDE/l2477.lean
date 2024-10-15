import Mathlib

namespace NUMINAMATH_CALUDE_unique_function_exists_l2477_247709

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + f y + 1) = x + y

/-- Theorem stating that there exists a unique function satisfying the equation -/
theorem unique_function_exists : ∃! f : ℝ → ℝ, SatisfiesEquation f := by
  sorry

end NUMINAMATH_CALUDE_unique_function_exists_l2477_247709


namespace NUMINAMATH_CALUDE_max_cross_section_area_l2477_247718

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

/-- Represents a square prism -/
structure SquarePrism where
  sideLength : ℝ
  baseVertices : List Point3D

/-- Calculates the area of the cross-section when a plane intersects a square prism -/
def crossSectionArea (prism : SquarePrism) (plane : Plane) : ℝ := sorry

/-- The main theorem stating the maximum area of the cross-section -/
theorem max_cross_section_area :
  let prism : SquarePrism := {
    sideLength := 12,
    baseVertices := [
      {x := 6, y := 6, z := 0},
      {x := -6, y := 6, z := 0},
      {x := -6, y := -6, z := 0},
      {x := 6, y := -6, z := 0}
    ]
  }
  let plane : Plane := {a := 5, b := -8, c := 3, d := 30}
  crossSectionArea prism plane = 252 := by sorry

end NUMINAMATH_CALUDE_max_cross_section_area_l2477_247718


namespace NUMINAMATH_CALUDE_antoinette_weight_l2477_247743

theorem antoinette_weight (rupert_weight antoinette_weight : ℝ) : 
  antoinette_weight = 2 * rupert_weight - 7 →
  antoinette_weight + rupert_weight = 98 →
  antoinette_weight = 63 := by
  sorry

end NUMINAMATH_CALUDE_antoinette_weight_l2477_247743


namespace NUMINAMATH_CALUDE_troy_computer_purchase_l2477_247795

/-- The problem of Troy buying a new computer -/
theorem troy_computer_purchase (new_computer_cost initial_savings old_computer_value : ℕ)
  (h1 : new_computer_cost = 80)
  (h2 : initial_savings = 50)
  (h3 : old_computer_value = 20) :
  new_computer_cost - (initial_savings + old_computer_value) = 10 := by
  sorry

end NUMINAMATH_CALUDE_troy_computer_purchase_l2477_247795


namespace NUMINAMATH_CALUDE_trapezoid_shorter_base_l2477_247773

/-- Represents a trapezoid -/
structure Trapezoid where
  longer_base : ℝ
  shorter_base : ℝ
  midpoint_segment : ℝ

/-- 
Theorem: In a trapezoid where the line segment joining the midpoints of the diagonals 
has length 4 and the longer base is 100, the length of the shorter base is 92.
-/
theorem trapezoid_shorter_base 
  (T : Trapezoid) 
  (h1 : T.longer_base = 100) 
  (h2 : T.midpoint_segment = 4) : 
  T.shorter_base = 92 := by
  sorry


end NUMINAMATH_CALUDE_trapezoid_shorter_base_l2477_247773


namespace NUMINAMATH_CALUDE_pie_division_l2477_247774

theorem pie_division (apple_pie : ℚ) (cherry_pie : ℚ) (people : ℕ) :
  apple_pie = 3/4 ∧ cherry_pie = 5/6 ∧ people = 3 →
  apple_pie / people = 1/4 ∧ cherry_pie / people = 5/18 := by
sorry

end NUMINAMATH_CALUDE_pie_division_l2477_247774


namespace NUMINAMATH_CALUDE_bridget_apples_bridget_apples_solution_l2477_247708

theorem bridget_apples : ℕ → Prop :=
  fun total_apples =>
    let apples_after_ann := (2 * total_apples) / 3
    let apples_after_cassie := apples_after_ann - 5
    apples_after_cassie = 7 → total_apples = 18

theorem bridget_apples_solution : bridget_apples 18 := by
  sorry

end NUMINAMATH_CALUDE_bridget_apples_bridget_apples_solution_l2477_247708


namespace NUMINAMATH_CALUDE_bug_position_after_2023_jumps_l2477_247751

/-- Represents the points on the circle --/
inductive Point
| One
| Two
| Three
| Four
| Five
| Six

/-- Defines the next point for a jump --/
def nextPoint (p : Point) : Point :=
  match p with
  | Point.One => Point.Three
  | Point.Two => Point.Three
  | Point.Three => Point.Five
  | Point.Four => Point.Five
  | Point.Five => Point.One
  | Point.Six => Point.One

/-- Calculates the position after n jumps --/
def positionAfterJumps (start : Point) (n : Nat) : Point :=
  match n with
  | 0 => start
  | Nat.succ m => nextPoint (positionAfterJumps start m)

theorem bug_position_after_2023_jumps :
  positionAfterJumps Point.Six 2023 = Point.One := by
  sorry


end NUMINAMATH_CALUDE_bug_position_after_2023_jumps_l2477_247751


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_unique_solution_of_system_l2477_247772

-- Problem 1
theorem solution_set_of_inequality (x : ℝ) :
  9 * (x - 2)^2 ≤ 25 ↔ x = 11/3 ∨ x = 1/3 :=
sorry

-- Problem 2
theorem unique_solution_of_system (x y : ℝ) :
  (x + 1) / 3 = 2 * y ∧ 2 * (x + 1) - y = 11 ↔ x = 5 ∧ y = 1 :=
sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_unique_solution_of_system_l2477_247772


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2477_247798

theorem complex_equation_solution (z : ℂ) : (Complex.I / (z + Complex.I) = 2 - Complex.I) → z = -1/5 - 3/5 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2477_247798


namespace NUMINAMATH_CALUDE_bird_flight_theorem_l2477_247703

/-- The height of the church tower in feet -/
def church_height : ℝ := 150

/-- The height of the catholic tower in feet -/
def catholic_height : ℝ := 200

/-- The distance between the two towers in feet -/
def tower_distance : ℝ := 350

/-- The distance of the grain from the church tower in feet -/
def grain_distance : ℝ := 200

theorem bird_flight_theorem :
  ∀ (x : ℝ),
  (x^2 + church_height^2 = (tower_distance - x)^2 + catholic_height^2) →
  x = grain_distance :=
by sorry

end NUMINAMATH_CALUDE_bird_flight_theorem_l2477_247703


namespace NUMINAMATH_CALUDE_discount_sum_is_22_percent_l2477_247762

/-- The discount rate for Pony jeans -/
def pony_discount : ℝ := 10.999999999999996

/-- The regular price of Fox jeans -/
def fox_price : ℝ := 15

/-- The regular price of Pony jeans -/
def pony_price : ℝ := 18

/-- The number of Fox jeans purchased -/
def fox_quantity : ℕ := 3

/-- The number of Pony jeans purchased -/
def pony_quantity : ℕ := 2

/-- The total savings from the discounts -/
def total_savings : ℝ := 8.91

/-- Theorem stating that the sum of discount rates for Fox and Pony jeans is 22% -/
theorem discount_sum_is_22_percent :
  ∃ (fox_discount : ℝ),
    fox_discount ≥ 0 ∧
    fox_discount ≤ 100 ∧
    fox_discount + pony_discount = 22 ∧
    (fox_price * fox_quantity * fox_discount / 100 + 
     pony_price * pony_quantity * pony_discount / 100 = total_savings) :=
by sorry

end NUMINAMATH_CALUDE_discount_sum_is_22_percent_l2477_247762


namespace NUMINAMATH_CALUDE_students_in_one_activity_l2477_247737

/-- Given a school with students participating in two elective courses, 
    prove the number of students in exactly one course. -/
theorem students_in_one_activity 
  (total : ℕ) 
  (both : ℕ) 
  (none : ℕ) 
  (h1 : total = 317) 
  (h2 : both = 30) 
  (h3 : none = 20) : 
  total - both - none = 267 := by
  sorry

#check students_in_one_activity

end NUMINAMATH_CALUDE_students_in_one_activity_l2477_247737


namespace NUMINAMATH_CALUDE_value_of_p_l2477_247783

variables (A B C p q r s : ℝ) 

/-- The roots of the first quadratic equation -/
def roots_eq1 : Prop := A * r^2 + B * r + C = 0 ∧ A * s^2 + B * s + C = 0

/-- The roots of the second quadratic equation -/
def roots_eq2 : Prop := r^2 + p * r + q = 0 ∧ s^2 + p * s + q = 0

/-- The theorem stating the value of p -/
theorem value_of_p (hA : A ≠ 0) (h1 : roots_eq1 A B C r s) (h2 : roots_eq2 p q r s) : 
  p = (2 * A * C - B^2) / A^2 := by
  sorry

end NUMINAMATH_CALUDE_value_of_p_l2477_247783


namespace NUMINAMATH_CALUDE_intersection_point_sum_l2477_247721

/-- Given a function g: ℝ → ℝ, if g(-2) = g(2) = 3, then (-2, 3) is the unique
    intersection point of y = g(x) and y = g(x+4), and the sum of its coordinates is 1. -/
theorem intersection_point_sum (g : ℝ → ℝ) (h1 : g (-2) = 3) (h2 : g 2 = 3) :
  ∃! p : ℝ × ℝ, (g p.1 = p.2 ∧ g (p.1 + 4) = p.2) ∧ p = (-2, 3) ∧ p.1 + p.2 = 1 := by
sorry

end NUMINAMATH_CALUDE_intersection_point_sum_l2477_247721


namespace NUMINAMATH_CALUDE_leak_emptying_time_l2477_247761

/-- Given a tank with specified capacity and inlet rate, proves the time taken for a leak to empty the tank. -/
theorem leak_emptying_time (tank_capacity : ℝ) (inlet_rate_per_minute : ℝ) (emptying_time_with_inlet : ℝ) 
  (h1 : tank_capacity = 1440)
  (h2 : inlet_rate_per_minute = 6)
  (h3 : emptying_time_with_inlet = 12) :
  (tank_capacity / (inlet_rate_per_minute * 60 + tank_capacity / emptying_time_with_inlet)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_leak_emptying_time_l2477_247761


namespace NUMINAMATH_CALUDE_power_base_property_l2477_247785

theorem power_base_property (k : ℝ) (h : k > 1) :
  let x := k^(1/(k-1))
  ∀ y : ℝ, (k*x)^(y/k) = x^y := by
sorry

end NUMINAMATH_CALUDE_power_base_property_l2477_247785


namespace NUMINAMATH_CALUDE_least_number_with_12_factors_l2477_247732

/-- The number of positive factors of a positive integer -/
def num_factors (n : ℕ+) : ℕ := sorry

/-- Check if a number is the least positive integer with exactly 12 factors -/
def is_least_with_12_factors (n : ℕ+) : Prop :=
  (num_factors n = 12) ∧ ∀ m : ℕ+, m < n → num_factors m ≠ 12

theorem least_number_with_12_factors :
  is_least_with_12_factors 96 := by sorry

end NUMINAMATH_CALUDE_least_number_with_12_factors_l2477_247732


namespace NUMINAMATH_CALUDE_cone_height_ratio_l2477_247712

/-- Theorem: Ratio of shortened height to original height of a cone -/
theorem cone_height_ratio (r : ℝ) (h₀ : ℝ) (h : ℝ) :
  r > 0 ∧ h₀ > 0 ∧ h > 0 →
  2 * Real.pi * r = 20 * Real.pi →
  h₀ = 50 →
  (1/3) * Real.pi * r^2 * h = 500 * Real.pi →
  h / h₀ = 3/10 := by sorry

end NUMINAMATH_CALUDE_cone_height_ratio_l2477_247712


namespace NUMINAMATH_CALUDE_angle_C_in_triangle_l2477_247790

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem angle_C_in_triangle (t : Triangle) :
  t.a = Real.sqrt 2 ∧ 
  t.b = Real.sqrt 3 ∧ 
  t.A = 45 * (π / 180) →
  t.C = 75 * (π / 180) ∨ t.C = 15 * (π / 180) := by
  sorry


end NUMINAMATH_CALUDE_angle_C_in_triangle_l2477_247790


namespace NUMINAMATH_CALUDE_prob_fewer_heads_12_coins_l2477_247711

/-- The number of coin flips -/
def n : ℕ := 12

/-- The probability of getting heads on a single fair coin flip -/
def p : ℚ := 1/2

/-- The probability of getting fewer heads than tails in n fair coin flips -/
def prob_fewer_heads (n : ℕ) (p : ℚ) : ℚ :=
  1/2 * (1 - (n.choose (n/2)) / (2^n : ℚ))

theorem prob_fewer_heads_12_coins : 
  prob_fewer_heads n p = 793/2048 := by sorry

end NUMINAMATH_CALUDE_prob_fewer_heads_12_coins_l2477_247711


namespace NUMINAMATH_CALUDE_function_bounded_by_square_l2477_247716

/-- A function satisfying the given inequality is bounded by x² -/
theorem function_bounded_by_square {f : ℝ → ℝ} (hf_nonneg : ∀ x ≥ 0, f x ≥ 0)
  (hf_bounded : ∃ M > 0, ∀ x ∈ Set.Icc 0 1, f x ≤ M)
  (h_ineq : ∀ x y, x ≥ 0 → y ≥ 0 → f x * f y ≤ x^2 * f (y/2) + y^2 * f (x/2)) :
  ∀ x ≥ 0, f x ≤ x^2 := by
  sorry

end NUMINAMATH_CALUDE_function_bounded_by_square_l2477_247716


namespace NUMINAMATH_CALUDE_slope_of_solutions_l2477_247719

theorem slope_of_solutions (x₁ x₂ y₁ y₂ : ℝ) (h₁ : x₁ ≠ x₂) 
  (h₂ : 3 / x₁ + 4 / y₁ = 0) (h₃ : 3 / x₂ + 4 / y₂ = 0) : 
  (y₂ - y₁) / (x₂ - x₁) = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_slope_of_solutions_l2477_247719


namespace NUMINAMATH_CALUDE_rope_length_proof_l2477_247741

/-- The length of the rope in meters -/
def rope_length : ℝ := 35

/-- The number of steps Xiaoming takes when walking in the same direction as the tractor -/
def steps_same_direction : ℕ := 140

/-- The number of steps Xiaoming takes when walking in the opposite direction of the tractor -/
def steps_opposite_direction : ℕ := 20

/-- The length of each of Xiaoming's steps in meters -/
def step_length : ℝ := 1

theorem rope_length_proof :
  ∃ (tractor_speed : ℝ),
    tractor_speed > 0 ∧
    rope_length + tractor_speed * steps_same_direction * step_length = steps_same_direction * step_length ∧
    rope_length - tractor_speed * steps_opposite_direction * step_length = steps_opposite_direction * step_length :=
by
  sorry

#check rope_length_proof

end NUMINAMATH_CALUDE_rope_length_proof_l2477_247741


namespace NUMINAMATH_CALUDE_day_of_week_problem_l2477_247739

/-- Represents days of the week -/
inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

/-- Represents a year -/
structure Year :=
  (number : ℤ)

/-- Represents a day in a year -/
structure DayInYear :=
  (year : Year)
  (dayNumber : ℕ)
  (dayOfWeek : DayOfWeek)

def isLeapYear (y : Year) : Prop := sorry

theorem day_of_week_problem 
  (M : Year)
  (day250_M : DayInYear)
  (day150_M_plus_2 : DayInYear)
  (day50_M_minus_1 : DayInYear)
  (h1 : day250_M.year = M)
  (h2 : day250_M.dayNumber = 250)
  (h3 : day250_M.dayOfWeek = DayOfWeek.Friday)
  (h4 : ¬ isLeapYear M)
  (h5 : day150_M_plus_2.year.number = M.number + 2)
  (h6 : day150_M_plus_2.dayNumber = 150)
  (h7 : day150_M_plus_2.dayOfWeek = DayOfWeek.Friday)
  (h8 : day50_M_minus_1.year.number = M.number - 1)
  (h9 : day50_M_minus_1.dayNumber = 50) :
  day50_M_minus_1.dayOfWeek = DayOfWeek.Sunday :=
sorry

end NUMINAMATH_CALUDE_day_of_week_problem_l2477_247739


namespace NUMINAMATH_CALUDE_bread_inventory_l2477_247792

/-- The number of loaves of bread sold during the day -/
def loaves_sold : ℕ := 629

/-- The number of loaves of bread delivered in the evening -/
def loaves_delivered : ℕ := 489

/-- The number of loaves of bread at the end of the day -/
def loaves_end : ℕ := 2215

/-- The number of loaves of bread at the start of the day -/
def loaves_start : ℕ := 2355

theorem bread_inventory : loaves_start - loaves_sold + loaves_delivered = loaves_end := by
  sorry

end NUMINAMATH_CALUDE_bread_inventory_l2477_247792


namespace NUMINAMATH_CALUDE_product_of_real_parts_of_roots_l2477_247750

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the polynomial
def f (z : ℂ) : ℂ := z^2 - z - (10 - 6*i)

-- State the theorem
theorem product_of_real_parts_of_roots :
  ∃ (z₁ z₂ : ℂ), f z₁ = 0 ∧ f z₂ = 0 ∧ (z₁.re * z₂.re = -47/4) :=
sorry

end NUMINAMATH_CALUDE_product_of_real_parts_of_roots_l2477_247750


namespace NUMINAMATH_CALUDE_polar_to_cartesian_l2477_247775

theorem polar_to_cartesian :
  let r : ℝ := 2
  let θ : ℝ := π / 6
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  (x = Real.sqrt 3 ∧ y = 1) := by sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_l2477_247775


namespace NUMINAMATH_CALUDE_smallest_factors_for_square_and_cube_l2477_247764

theorem smallest_factors_for_square_and_cube (n : ℕ) (hn : n = 450) :
  ∃ (x y : ℕ),
    (∀ x' : ℕ, x' < x → ¬∃ k : ℕ, n * x' = k^2) ∧
    (∀ y' : ℕ, y' < y → ¬∃ k : ℕ, n * y' = k^3) ∧
    (∃ k : ℕ, n * x = k^2) ∧
    (∃ k : ℕ, n * y = k^3) ∧
    x = 2 ∧ y = 60 ∧ x + y = 62 :=
by sorry

end NUMINAMATH_CALUDE_smallest_factors_for_square_and_cube_l2477_247764


namespace NUMINAMATH_CALUDE_factorization_problems_l2477_247714

variables (a b m n : ℝ)

theorem factorization_problems :
  (m^2 * (a - b) + 4 * n^2 * (b - a) = (a - b) * (m + 2*n) * (m - 2*n)) ∧
  (-a^3 + 2*a^2*b - a*b^2 = -a * (a - b)^2) := by sorry

end NUMINAMATH_CALUDE_factorization_problems_l2477_247714


namespace NUMINAMATH_CALUDE_angle_trigonometry_l2477_247747

theorem angle_trigonometry (α : Real) :
  (∃ (x y : Real), x = -1 ∧ y = Real.sqrt 2 ∧ 
   x = Real.cos α * Real.sqrt (x^2 + y^2) ∧
   y = Real.sin α * Real.sqrt (x^2 + y^2)) →
  Real.tan α = -Real.sqrt 2 ∧ Real.cos (2 * α) = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_angle_trigonometry_l2477_247747


namespace NUMINAMATH_CALUDE_exists_valid_nail_configuration_l2477_247736

/-- A point on the chessboard -/
structure Point where
  x : Fin 8
  y : Fin 8

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.x - p1.x) * (p3.y - p1.y) = (p3.x - p1.x) * (p2.y - p1.y)

/-- A set of 16 points on the chessboard -/
def NailConfiguration := Fin 16 → Point

/-- A valid nail configuration has no three collinear points -/
def valid_configuration (config : NailConfiguration) : Prop :=
  ∀ i j k, i ≠ j → j ≠ k → i ≠ k → ¬collinear (config i) (config j) (config k)

/-- There exists a valid configuration of 16 nails on an 8x8 chessboard -/
theorem exists_valid_nail_configuration : ∃ config : NailConfiguration, valid_configuration config := by
  sorry

end NUMINAMATH_CALUDE_exists_valid_nail_configuration_l2477_247736


namespace NUMINAMATH_CALUDE_complex_number_equivalence_l2477_247771

theorem complex_number_equivalence : (10 * Complex.I) / (1 - 2 * Complex.I) = -4 + 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_number_equivalence_l2477_247771


namespace NUMINAMATH_CALUDE_vector_magnitude_l2477_247744

theorem vector_magnitude (a b : ℝ × ℝ) :
  a = (Real.cos (10 * π / 180), Real.sin (10 * π / 180)) →
  b = (Real.cos (70 * π / 180), Real.sin (70 * π / 180)) →
  ‖a - 2 • b‖ = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_l2477_247744


namespace NUMINAMATH_CALUDE_pet_store_dogs_l2477_247715

theorem pet_store_dogs (initial_dogs : ℕ) : 
  initial_dogs + 5 + 3 = 10 → initial_dogs = 2 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_dogs_l2477_247715


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l2477_247770

/-- Two vectors are parallel if and only if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 * b.2 = k * a.2 * b.1

/-- Given vectors a and b, if they are parallel, then the x-coordinate of b is 2 -/
theorem parallel_vectors_x_value (a b : ℝ × ℝ) (h : parallel a b) :
    a = (1, 2) → b.1 = x → b.2 = 4 → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l2477_247770


namespace NUMINAMATH_CALUDE_min_sum_of_squares_on_line_l2477_247753

theorem min_sum_of_squares_on_line :
  ∀ (x y : ℝ), x + y = 4 → ∀ (a b : ℝ), a + b = 4 → x^2 + y^2 ≤ a^2 + b^2 ∧ ∃ (x₀ y₀ : ℝ), x₀ + y₀ = 4 ∧ x₀^2 + y₀^2 = 8 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_squares_on_line_l2477_247753


namespace NUMINAMATH_CALUDE_square_difference_of_quadratic_solutions_l2477_247777

theorem square_difference_of_quadratic_solutions : 
  ∀ α β : ℝ, 
  (α^2 = 2*α + 1) → 
  (β^2 = 2*β + 1) → 
  (α ≠ β) → 
  (α - β)^2 = 8 := by
sorry

end NUMINAMATH_CALUDE_square_difference_of_quadratic_solutions_l2477_247777


namespace NUMINAMATH_CALUDE_scientific_notation_of_104000000_l2477_247781

theorem scientific_notation_of_104000000 :
  (104000000 : ℝ) = 1.04 * (10 ^ 8) := by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_104000000_l2477_247781


namespace NUMINAMATH_CALUDE_vegetable_vendor_problem_l2477_247746

/-- Represents the vegetable vendor's purchase and profit calculation -/
theorem vegetable_vendor_problem :
  ∀ (x y : ℝ),
  -- Total weight condition
  x + y = 40 →
  -- Total cost condition
  3 * x + 2.4 * y = 114 →
  -- Prove the weights of cucumbers and potatoes
  x = 30 ∧ y = 10 ∧
  -- Prove the total profit
  (5 - 3) * x + (4 - 2.4) * y = 76 :=
by
  sorry


end NUMINAMATH_CALUDE_vegetable_vendor_problem_l2477_247746


namespace NUMINAMATH_CALUDE_giraffe_difference_l2477_247766

/- Define the number of lions in Safari National Park -/
def safari_lions : ℕ := 100

/- Define the number of snakes in Safari National Park -/
def safari_snakes : ℕ := safari_lions / 2

/- Define the number of giraffes in Safari National Park -/
def safari_giraffes : ℕ := safari_snakes - 10

/- Define the number of lions in Savanna National Park -/
def savanna_lions : ℕ := 2 * safari_lions

/- Define the number of snakes in Savanna National Park -/
def savanna_snakes : ℕ := 3 * safari_snakes

/- Define the total number of animals in Savanna National Park -/
def savanna_total : ℕ := 410

/- Theorem: The difference in the number of giraffes between Savanna and Safari National Parks is 20 -/
theorem giraffe_difference : 
  savanna_total - savanna_lions - savanna_snakes - safari_giraffes = 20 := by
  sorry

end NUMINAMATH_CALUDE_giraffe_difference_l2477_247766


namespace NUMINAMATH_CALUDE_ceiling_floor_sum_l2477_247704

theorem ceiling_floor_sum : ⌈(7 : ℝ) / 3⌉ + ⌊-(7 : ℝ) / 3⌋ = 0 := by sorry

end NUMINAMATH_CALUDE_ceiling_floor_sum_l2477_247704


namespace NUMINAMATH_CALUDE_carrie_vegetable_revenue_l2477_247796

/-- Represents the revenue calculation for Carrie's vegetable sales --/
theorem carrie_vegetable_revenue : 
  let tomatoes := 200
  let carrots := 350
  let eggplants := 120
  let cucumbers := 75
  let tomato_price := 1
  let carrot_price := 1.5
  let eggplant_price := 2.5
  let cucumber_price := 1.75
  let tomato_discount := 0.05
  let carrot_discount_price := 1.25
  let eggplant_free_per := 10
  let cucumber_discount := 0.1
  
  let tomato_revenue := tomatoes * (tomato_price * (1 - tomato_discount))
  let carrot_revenue := carrots * carrot_discount_price
  let eggplant_revenue := (eggplants - (eggplants / eggplant_free_per)) * eggplant_price
  let cucumber_revenue := cucumbers * (cucumber_price * (1 - cucumber_discount))
  
  tomato_revenue + carrot_revenue + eggplant_revenue + cucumber_revenue = 1015.625 := by
  sorry


end NUMINAMATH_CALUDE_carrie_vegetable_revenue_l2477_247796


namespace NUMINAMATH_CALUDE_gillians_total_spending_l2477_247782

def sandis_initial_amount : ℕ := 600
def gillians_additional_spending : ℕ := 150

def sandis_market_spending (initial_amount : ℕ) : ℕ :=
  initial_amount / 2

def gillians_market_spending (sandis_spending : ℕ) : ℕ :=
  3 * sandis_spending + gillians_additional_spending

theorem gillians_total_spending :
  gillians_market_spending (sandis_market_spending sandis_initial_amount) = 1050 := by
  sorry

end NUMINAMATH_CALUDE_gillians_total_spending_l2477_247782


namespace NUMINAMATH_CALUDE_not_divisible_by_four_l2477_247793

def a : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => a n * a (n + 1) + 1

theorem not_divisible_by_four : ¬ (4 ∣ a 2008) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_four_l2477_247793


namespace NUMINAMATH_CALUDE_value_of_a_l2477_247723

theorem value_of_a (A B : Set ℝ) (a : ℝ) : 
  A = {1, 3, a} → 
  B = {1, a^2} → 
  A ∩ B = {1, a} → 
  a = 0 := by
sorry

end NUMINAMATH_CALUDE_value_of_a_l2477_247723


namespace NUMINAMATH_CALUDE_A_intersect_B_equals_C_l2477_247738

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | ∃ y : ℝ, y = Real.log (4 - x^2)}
def B : Set ℝ := {x : ℝ | 2*x < 1}

-- State the theorem
theorem A_intersect_B_equals_C : A ∩ B = {x : ℝ | -2 < x ∧ x < 0} := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_equals_C_l2477_247738


namespace NUMINAMATH_CALUDE_tan_fifteen_simplification_l2477_247745

theorem tan_fifteen_simplification :
  (1 + Real.tan (15 * π / 180)) / (1 - Real.tan (15 * π / 180)) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_fifteen_simplification_l2477_247745


namespace NUMINAMATH_CALUDE_intersection_problem_l2477_247725

theorem intersection_problem (a : ℝ) : 
  let A : Set ℝ := {-1, 1, 3}
  let B : Set ℝ := {a - 1, a^2 + 3}
  A ∩ B = {3} → a = 4 ∨ a = 0 := by
sorry

end NUMINAMATH_CALUDE_intersection_problem_l2477_247725


namespace NUMINAMATH_CALUDE_measure_15_minutes_l2477_247768

/-- Represents an hourglass with a given duration in minutes -/
structure Hourglass where
  duration : ℕ

/-- Represents the state of measuring time with two hourglasses -/
structure MeasurementState where
  time_elapsed : ℕ
  hourglass1 : Hourglass
  hourglass2 : Hourglass

/-- Checks if it's possible to measure the target time using two hourglasses -/
def can_measure_time (target : ℕ) (h1 : Hourglass) (h2 : Hourglass) : Prop :=
  ∃ (steps : ℕ) (final_state : MeasurementState),
    final_state.time_elapsed = target ∧
    final_state.hourglass1 = h1 ∧
    final_state.hourglass2 = h2

theorem measure_15_minutes :
  can_measure_time 15 (Hourglass.mk 7) (Hourglass.mk 11) :=
sorry

end NUMINAMATH_CALUDE_measure_15_minutes_l2477_247768


namespace NUMINAMATH_CALUDE_line_parallel_to_plane_no_common_points_l2477_247758

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation between a line and a plane
variable (parallel : Line → Plane → Prop)

-- Define the "contained within" relation for a line in a plane
variable (contained_in : Line → Plane → Prop)

-- Define the "no common points" relation between two lines
variable (no_common_points : Line → Line → Prop)

-- State the theorem
theorem line_parallel_to_plane_no_common_points 
  (l a : Line) (α : Plane) 
  (h1 : parallel l α) 
  (h2 : contained_in a α) : 
  no_common_points l a := by
  sorry

end NUMINAMATH_CALUDE_line_parallel_to_plane_no_common_points_l2477_247758


namespace NUMINAMATH_CALUDE_grid_shading_contradiction_l2477_247707

theorem grid_shading_contradiction (k n x y : ℕ) (hk : k > 0) (hn : n > 0) :
  ¬(k * (x + 5) = n * y ∧ k * x = n * (y - 3)) :=
by sorry

end NUMINAMATH_CALUDE_grid_shading_contradiction_l2477_247707


namespace NUMINAMATH_CALUDE_miles_french_horns_l2477_247778

/-- Represents the number of musical instruments Miles owns -/
structure MilesInstruments where
  trumpets : ℕ
  guitars : ℕ
  trombones : ℕ
  frenchHorns : ℕ

/-- Represents Miles' body parts relevant to the problem -/
structure MilesAnatomy where
  fingers : ℕ
  hands : ℕ
  head : ℕ

theorem miles_french_horns 
  (anatomy : MilesAnatomy)
  (instruments : MilesInstruments)
  (h1 : anatomy.fingers = 10)
  (h2 : anatomy.hands = 2)
  (h3 : anatomy.head = 1)
  (h4 : instruments.trumpets = anatomy.fingers - 3)
  (h5 : instruments.guitars = anatomy.hands + 2)
  (h6 : instruments.trombones = anatomy.head + 2)
  (h7 : instruments.trumpets + instruments.guitars + instruments.trombones + instruments.frenchHorns = 17)
  (h8 : instruments.frenchHorns = instruments.guitars - 1) :
  instruments.frenchHorns = 3 := by
  sorry

#check miles_french_horns

end NUMINAMATH_CALUDE_miles_french_horns_l2477_247778


namespace NUMINAMATH_CALUDE_number_puzzle_l2477_247799

theorem number_puzzle : ∃! x : ℝ, x / 5 + 7 = x / 4 - 7 := by sorry

end NUMINAMATH_CALUDE_number_puzzle_l2477_247799


namespace NUMINAMATH_CALUDE_y_squared_value_l2477_247722

theorem y_squared_value (y : ℝ) (hy : y > 0) (h : Real.tan (Real.arccos y) = 2 * y) :
  y^2 = (-1 + Real.sqrt 17) / 8 := by
  sorry

end NUMINAMATH_CALUDE_y_squared_value_l2477_247722


namespace NUMINAMATH_CALUDE_exists_palindrome_multiple_l2477_247780

/-- A number is a decimal palindrome if its decimal representation is mirror symmetric. -/
def IsDecimalPalindrome (n : ℕ) : Prop :=
  ∃ (digits : List ℕ), digits.reverse = digits ∧ n = digits.foldl (fun acc d => acc * 10 + d) 0

/-- Main theorem: For any positive integer not divisible by 10, 
    there exists a positive multiple that is a decimal palindrome. -/
theorem exists_palindrome_multiple {n : ℕ} (hn : n > 0) (hndiv : ¬ 10 ∣ n) :
  ∃ (m : ℕ), m > 0 ∧ n ∣ m ∧ IsDecimalPalindrome m := by
  sorry

end NUMINAMATH_CALUDE_exists_palindrome_multiple_l2477_247780


namespace NUMINAMATH_CALUDE_two_digit_sum_square_property_l2477_247776

/-- Returns the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

/-- Checks if a number is two-digit -/
def isTwoDigit (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

/-- The set of numbers satisfying the condition -/
def validSet : Set ℕ :=
  {10, 20, 11, 30, 21, 12, 31, 22, 13}

/-- The main theorem -/
theorem two_digit_sum_square_property (A : ℕ) :
  isTwoDigit A →
  ((sumOfDigits A)^2 = sumOfDigits (A^2)) ↔ A ∈ validSet :=
sorry

end NUMINAMATH_CALUDE_two_digit_sum_square_property_l2477_247776


namespace NUMINAMATH_CALUDE_number_wall_problem_l2477_247700

/-- Represents a row in the Number Wall -/
structure NumberWallRow :=
  (a b c d : ℕ)

/-- Calculates the next row in the Number Wall -/
def nextRow (row : NumberWallRow) : NumberWallRow :=
  ⟨row.a + row.b, row.b + row.c, row.c + row.d, 0⟩

/-- The Number Wall problem -/
theorem number_wall_problem (m : ℕ) : m = 2 :=
  let row1 := NumberWallRow.mk m 5 9 6
  let row2 := nextRow row1
  let row3 := nextRow row2
  let row4 := nextRow row3
  have h1 : row2.c = 18 := by sorry
  have h2 : row4.a = 55 := by sorry
  sorry

end NUMINAMATH_CALUDE_number_wall_problem_l2477_247700


namespace NUMINAMATH_CALUDE_ice_cube_volume_l2477_247794

theorem ice_cube_volume (original_volume : ℝ) : 
  (original_volume > 0) →
  (original_volume * (1/4) * (1/4) = 0.4) →
  original_volume = 6.4 := by
sorry

end NUMINAMATH_CALUDE_ice_cube_volume_l2477_247794


namespace NUMINAMATH_CALUDE_milk_powder_sampling_l2477_247734

theorem milk_powder_sampling (total : ℕ) (sample_size : ℕ) (b : ℕ) :
  total = 240 →
  sample_size = 60 →
  (∃ (a d : ℕ), b = a ∧ total = (a - d) + a + (a + d)) →
  b * sample_size / total = 20 :=
by sorry

end NUMINAMATH_CALUDE_milk_powder_sampling_l2477_247734


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l2477_247729

/-- The repeating decimal 0.777... -/
def repeating_decimal : ℚ := 0.7777777

/-- The fraction 7/9 -/
def fraction : ℚ := 7/9

/-- Theorem stating that the repeating decimal 0.777... is equal to the fraction 7/9 -/
theorem repeating_decimal_equals_fraction : repeating_decimal = fraction := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l2477_247729


namespace NUMINAMATH_CALUDE_students_neither_music_nor_art_l2477_247710

theorem students_neither_music_nor_art 
  (total : ℕ) 
  (music : ℕ) 
  (art : ℕ) 
  (both : ℕ) 
  (h1 : total = 500) 
  (h2 : music = 50) 
  (h3 : art = 20) 
  (h4 : both = 10) : 
  total - (music + art - both) = 440 := by
sorry

end NUMINAMATH_CALUDE_students_neither_music_nor_art_l2477_247710


namespace NUMINAMATH_CALUDE_billy_candy_per_house_l2477_247726

/-- 
Given:
- Anna gets 14 pieces of candy per house
- Anna visits 60 houses
- Billy visits 75 houses
- Anna gets 15 more pieces of candy than Billy

Prove that Billy gets 11 pieces of candy per house
-/
theorem billy_candy_per_house :
  let anna_candy_per_house : ℕ := 14
  let anna_houses : ℕ := 60
  let billy_houses : ℕ := 75
  let candy_difference : ℕ := 15
  ∃ billy_candy_per_house : ℕ,
    anna_candy_per_house * anna_houses = billy_candy_per_house * billy_houses + candy_difference ∧
    billy_candy_per_house = 11 :=
by sorry

end NUMINAMATH_CALUDE_billy_candy_per_house_l2477_247726


namespace NUMINAMATH_CALUDE_power_equation_solutions_l2477_247754

theorem power_equation_solutions :
  ∀ x y : ℕ, 2^x = 3^y + 5 ↔ (x = 3 ∧ y = 1) ∨ (x = 5 ∧ y = 3) :=
by sorry

end NUMINAMATH_CALUDE_power_equation_solutions_l2477_247754


namespace NUMINAMATH_CALUDE_exists_index_sum_inequality_l2477_247749

theorem exists_index_sum_inequality (a : Fin 100 → ℝ) 
  (h_distinct : ∀ i j : Fin 100, i ≠ j → a i ≠ a j) :
  ∃ i : Fin 100, a i + a ((i + 3) % 100) > a ((i + 1) % 100) + a ((i + 2) % 100) := by
  sorry

end NUMINAMATH_CALUDE_exists_index_sum_inequality_l2477_247749


namespace NUMINAMATH_CALUDE_quadratic_inequality_l2477_247755

/-- A quadratic function f(x) = mx² - 2mx + 1 with m > 0 -/
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 - 2 * m * x + 1

/-- Theorem stating the relationship between f(6), f(-1), and f(5/2) -/
theorem quadratic_inequality (m : ℝ) (h : m > 0) : 
  f m 6 > f m (-1) ∧ f m (-1) > f m (5/2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l2477_247755


namespace NUMINAMATH_CALUDE_june_population_estimate_l2477_247784

/-- Represents the number of rabbits tagged on June 1 -/
def tagged_rabbits : ℕ := 50

/-- Represents the number of rabbits captured on October 1 -/
def captured_rabbits : ℕ := 80

/-- Represents the number of tagged rabbits found in the October capture -/
def tagged_captured : ℕ := 4

/-- Represents the percentage of original population no longer in the forest by October -/
def predation_rate : ℚ := 30 / 100

/-- Represents the percentage of October rabbits that were not in the forest in June -/
def new_birth_rate : ℚ := 50 / 100

/-- Estimates the number of rabbits in the forest on June 1 -/
def estimate_june_population : ℕ := 500

theorem june_population_estimate :
  tagged_rabbits * (captured_rabbits * (1 - new_birth_rate)) / tagged_captured = estimate_june_population :=
sorry

end NUMINAMATH_CALUDE_june_population_estimate_l2477_247784


namespace NUMINAMATH_CALUDE_system_solutions_l2477_247788

/-- The system of equations -/
def system (x y z : ℝ) : Prop :=
  x^2 - y = z^2 ∧ y^2 - z = x^2 ∧ z^2 - x = y^2

/-- The set of solutions -/
def solutions : Set (ℝ × ℝ × ℝ) :=
  {(0, 0, 0), (1, 0, -1), (0, -1, 1), (-1, 1, 0)}

/-- Theorem stating that the solutions are correct and complete -/
theorem system_solutions :
  ∀ x y z : ℝ, system x y z ↔ (x, y, z) ∈ solutions := by
  sorry

end NUMINAMATH_CALUDE_system_solutions_l2477_247788


namespace NUMINAMATH_CALUDE_coin_flip_probability_l2477_247786

def num_coins : ℕ := 6

def all_outcomes : ℕ := 2^num_coins

def favorable_outcomes : ℕ := 2 + 2 * (num_coins.choose 1)

theorem coin_flip_probability : 
  (favorable_outcomes : ℚ) / all_outcomes = 7 / 32 :=
sorry

end NUMINAMATH_CALUDE_coin_flip_probability_l2477_247786


namespace NUMINAMATH_CALUDE_gcd_372_684_l2477_247789

theorem gcd_372_684 : Nat.gcd 372 684 = 12 := by
  sorry

end NUMINAMATH_CALUDE_gcd_372_684_l2477_247789


namespace NUMINAMATH_CALUDE_intersection_A_B_when_m_3_range_of_m_when_A_subset_B_l2477_247752

-- Define set A
def A : Set ℝ := {x | ∃ y, y = Real.sqrt (6 + 5*x - x^2)}

-- Define set B
def B (m : ℝ) : Set ℝ := {x | (x - 1 + m) * (x - 1 - m) ≤ 0}

-- Theorem for part (1)
theorem intersection_A_B_when_m_3 : 
  A ∩ B 3 = {x | -1 ≤ x ∧ x ≤ 4} := by sorry

-- Theorem for part (2)
theorem range_of_m_when_A_subset_B : 
  ∀ m : ℝ, m > 0 → (A ⊆ B m) → m ≥ 5 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_when_m_3_range_of_m_when_A_subset_B_l2477_247752


namespace NUMINAMATH_CALUDE_unique_intersection_l2477_247735

noncomputable section

def f (m : ℝ) (x : ℝ) : ℝ := x^3 + 3*x^2 + m*x

def g (n : ℝ) (x : ℝ) : ℝ := Real.log (x + 1) + n*x

theorem unique_intersection (m n : ℝ) (h_n : n > 0) :
  (∀ x : ℝ, f m (-1 - x) = -1 - f m x) →
  (∃! x : ℝ, f m x = g n x) →
  m + n = 5 := by sorry

end NUMINAMATH_CALUDE_unique_intersection_l2477_247735


namespace NUMINAMATH_CALUDE_rotated_line_equation_l2477_247763

/-- The y-coordinate of point P on the original line -/
def y : ℝ := 4

/-- The slope of the original line -/
def m₁ : ℝ := 1

/-- The slope of line l -/
def m₂ : ℝ := -1

/-- Point P on the original line -/
def P : ℝ × ℝ := (3, y)

/-- The equation of the original line -/
def original_line (x y : ℝ) : Prop := x - y + 1 = 0

/-- The equation of line l after rotation -/
def line_l (x y : ℝ) : Prop := x + y - 7 = 0

theorem rotated_line_equation : 
  ∀ x y : ℝ, original_line P.1 P.2 → (m₁ * m₂ = -1) → line_l x y :=
sorry

end NUMINAMATH_CALUDE_rotated_line_equation_l2477_247763


namespace NUMINAMATH_CALUDE_households_using_both_is_15_l2477_247705

/-- Given information about soap brand usage in surveyed households -/
structure SoapSurvey where
  total : ℕ
  neither : ℕ
  only_E : ℕ
  both_to_only_B_ratio : ℕ
  h_total : total = 200
  h_neither : neither = 80
  h_only_E : only_E = 60
  h_ratio : both_to_only_B_ratio = 3

/-- The number of households using both brand E and brand B soap -/
def households_using_both (s : SoapSurvey) : ℕ := 15

/-- Theorem stating that the number of households using both brands is 15 -/
theorem households_using_both_is_15 (s : SoapSurvey) : 
  households_using_both s = 15 := by sorry

end NUMINAMATH_CALUDE_households_using_both_is_15_l2477_247705


namespace NUMINAMATH_CALUDE_minimal_polynomial_reciprocal_l2477_247713

theorem minimal_polynomial_reciprocal (x : ℂ) 
  (h1 : x^9 = 1) 
  (h2 : x^3 ≠ 1) : 
  x^5 - x^4 + x^3 = 1 / (1 + x) := by
  sorry

end NUMINAMATH_CALUDE_minimal_polynomial_reciprocal_l2477_247713


namespace NUMINAMATH_CALUDE_second_caterer_cheaper_at_34_min_people_for_second_caterer_cheaper_l2477_247727

/-- Represents the pricing model of a caterer -/
structure CatererPrice where
  basicFee : ℕ
  perPersonFee : ℕ

/-- Calculates the total price for a given number of people -/
def totalPrice (price : CatererPrice) (people : ℕ) : ℕ :=
  price.basicFee + price.perPersonFee * people

/-- The first caterer's pricing model -/
def caterer1 : CatererPrice :=
  { basicFee := 150, perPersonFee := 18 }

/-- The second caterer's pricing model -/
def caterer2 : CatererPrice :=
  { basicFee := 250, perPersonFee := 15 }

/-- Theorem stating that 34 is the minimum number of people for which
    the second caterer becomes cheaper than the first caterer -/
theorem second_caterer_cheaper_at_34 :
  (∀ n : ℕ, n < 34 → totalPrice caterer1 n ≤ totalPrice caterer2 n) ∧
  (totalPrice caterer1 34 > totalPrice caterer2 34) :=
by sorry

/-- Theorem stating that 34 is indeed the minimum such number -/
theorem min_people_for_second_caterer_cheaper :
  ∀ m : ℕ, m < 34 → ¬(∀ n : ℕ, n < m → totalPrice caterer1 n ≤ totalPrice caterer2 n) ∧
                    (totalPrice caterer1 m > totalPrice caterer2 m) :=
by sorry

end NUMINAMATH_CALUDE_second_caterer_cheaper_at_34_min_people_for_second_caterer_cheaper_l2477_247727


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l2477_247787

/-- Two vectors are parallel if their coordinates are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (2*x, -3)
  are_parallel a b → x = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l2477_247787


namespace NUMINAMATH_CALUDE_infinite_occurrence_in_digit_sum_sequence_l2477_247706

/-- A polynomial with integer coefficients -/
def IntPolynomial := Polynomial ℤ

/-- The sum of digits of an integer -/
def sumOfDigits (n : ℤ) : ℕ :=
  (n.natAbs.digits 10).sum

/-- The sequence of sums of digits of polynomial values -/
def digitSumSequence (w : IntPolynomial) : ℕ → ℕ := fun n ↦ sumOfDigits (w.eval n)

/-- There exists a value that occurs infinitely often in the digit sum sequence -/
theorem infinite_occurrence_in_digit_sum_sequence (w : IntPolynomial) :
  ∃ k : ℕ, Set.Infinite {n : ℕ | digitSumSequence w n = k} :=
sorry

end NUMINAMATH_CALUDE_infinite_occurrence_in_digit_sum_sequence_l2477_247706


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l2477_247733

theorem sqrt_product_equality : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l2477_247733


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l2477_247717

def U : Finset Nat := {1, 2, 3, 4, 5, 6}
def A : Finset Nat := {1, 2, 3}
def B : Finset Nat := {2, 5, 6}

theorem intersection_complement_equality :
  A ∩ (U \ B) = {1, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l2477_247717


namespace NUMINAMATH_CALUDE_inequality_proof_l2477_247740

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^2 / b + b^2 / c + c^2 / a ≥ a + b + c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2477_247740


namespace NUMINAMATH_CALUDE_longest_tape_measure_l2477_247791

theorem longest_tape_measure (a b c : ℕ) 
  (ha : a = 315) 
  (hb : b = 458) 
  (hc : c = 1112) : 
  Nat.gcd a (Nat.gcd b c) = 1 := by
  sorry

end NUMINAMATH_CALUDE_longest_tape_measure_l2477_247791


namespace NUMINAMATH_CALUDE_estimate_husk_amount_l2477_247765

/-- Estimate the amount of husk in a batch of rice given a sample -/
theorem estimate_husk_amount (total_rice : ℕ) (sample_size : ℕ) (husk_in_sample : ℕ) 
  (h1 : total_rice = 1520)
  (h2 : sample_size = 144)
  (h3 : husk_in_sample = 18) :
  (total_rice : ℚ) * (husk_in_sample : ℚ) / (sample_size : ℚ) = 190 := by
  sorry

#check estimate_husk_amount

end NUMINAMATH_CALUDE_estimate_husk_amount_l2477_247765


namespace NUMINAMATH_CALUDE_expression_always_positive_l2477_247701

theorem expression_always_positive (x y : ℝ) : x^2 - 4*x*y + 6*y^2 - 4*y + 3 > 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_always_positive_l2477_247701


namespace NUMINAMATH_CALUDE_parabola_properties_l2477_247730

-- Define the parabola
def parabola (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the conditions
theorem parabola_properties (a b c m : ℝ) 
  (h_vertex : -b / (2 * a) = -1/2)
  (h_m_pos : m > 0)
  (h_vertex_y : parabola a b c (-1/2) = m)
  (h_intercept : ∃ x, 0 < x ∧ x < 1 ∧ parabola a b c x = 0) :
  (b < 0) ∧ 
  (∀ y₁ y₂, parabola a b c (-2) = y₁ → parabola a b c 2 = y₂ → y₁ > y₂) :=
by sorry

end NUMINAMATH_CALUDE_parabola_properties_l2477_247730


namespace NUMINAMATH_CALUDE_pool_perimeter_l2477_247742

/-- The perimeter of a rectangular pool in a garden with specific conditions -/
theorem pool_perimeter (garden_length : ℝ) (square_area : ℝ) (num_squares : ℕ) :
  garden_length = 10 →
  square_area = 20 →
  num_squares = 4 →
  ∃ (pool_length pool_width : ℝ),
    0 < pool_length ∧ 0 < pool_width ∧
    2 * (pool_length + pool_width) = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_pool_perimeter_l2477_247742


namespace NUMINAMATH_CALUDE_correct_squares_form_cube_net_l2477_247757

-- Define the grid paper with 5 squares
structure GridPaper :=
  (squares : Fin 5 → Bool)
  (shaded : Set (Fin 5))

-- Define a cube net
def is_cube_net (gp : GridPaper) : Prop :=
  ∃ (s1 s2 : Fin 5), s1 ≠ s2 ∧ 
    gp.shaded = {s1, s2} ∧
    -- Additional conditions to ensure it forms a valid cube net
    true  -- Placeholder for the specific geometric conditions

-- Theorem statement
theorem correct_squares_form_cube_net (gp : GridPaper) :
  is_cube_net {squares := gp.squares, shaded := {4, 5}} :=
sorry

end NUMINAMATH_CALUDE_correct_squares_form_cube_net_l2477_247757


namespace NUMINAMATH_CALUDE_sarah_toy_cars_l2477_247748

def initial_amount : ℕ := 53
def toy_car_cost : ℕ := 11
def scarf_cost : ℕ := 10
def beanie_cost : ℕ := 14
def remaining_amount : ℕ := 7

theorem sarah_toy_cars :
  ∃ (num_cars : ℕ),
    num_cars * toy_car_cost + scarf_cost + beanie_cost = initial_amount - remaining_amount ∧
    num_cars = 2 :=
by sorry

end NUMINAMATH_CALUDE_sarah_toy_cars_l2477_247748


namespace NUMINAMATH_CALUDE_equation_solution_l2477_247769

theorem equation_solution : ∃ x : ℚ, (x - 7) / 3 - (1 + x) / 2 = 1 ∧ x = -23 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2477_247769


namespace NUMINAMATH_CALUDE_margaret_age_in_twelve_years_l2477_247756

/-- Given the ages of Brian, Christian, and Margaret, prove Margaret's age in 12 years -/
theorem margaret_age_in_twelve_years
  (brian_age : ℝ)
  (christian_age : ℝ)
  (margaret_age : ℝ)
  (h1 : christian_age = 3.5 * brian_age)
  (h2 : brian_age + 12 = 45)
  (h3 : margaret_age = christian_age - 10) :
  margaret_age + 12 = 117.5 := by
  sorry

end NUMINAMATH_CALUDE_margaret_age_in_twelve_years_l2477_247756


namespace NUMINAMATH_CALUDE_xiao_yun_age_l2477_247760

theorem xiao_yun_age :
  ∀ (x : ℕ),
  (∃ (f : ℕ), f = x + 25) →
  (x + 25 + 5 = 2 * (x + 5) - 10) →
  x = 30 := by
sorry

end NUMINAMATH_CALUDE_xiao_yun_age_l2477_247760


namespace NUMINAMATH_CALUDE_integer_parabola_coeff_sum_l2477_247702

/-- A parabola with integer coefficients passing through specific points -/
structure IntegerParabola where
  a : ℤ
  b : ℤ
  c : ℤ
  passes_through_origin : 1 = a * 0^2 + b * 0 + c
  passes_through_two_nine : 9 = a * 2^2 + b * 2 + c
  vertex_at_one_four : 4 = a * 1^2 + b * 1 + c

/-- The sum of coefficients of the integer parabola is 4 -/
theorem integer_parabola_coeff_sum (p : IntegerParabola) : p.a + p.b + p.c = 4 := by
  sorry


end NUMINAMATH_CALUDE_integer_parabola_coeff_sum_l2477_247702


namespace NUMINAMATH_CALUDE_equation_solution_l2477_247724

theorem equation_solution : 
  ∃ x : ℚ, (8 * x^2 + 80 * x + 4) / (4 * x + 45) = 2 * x + 3 ∧ x = -131/22 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2477_247724


namespace NUMINAMATH_CALUDE_three_a_in_S_implies_a_in_S_l2477_247779

def S : Set ℤ := {n | ∃ x y : ℤ, n = x^2 + 2*y^2}

theorem three_a_in_S_implies_a_in_S (a : ℤ) (h : (3 * a) ∈ S) : a ∈ S := by
  sorry

end NUMINAMATH_CALUDE_three_a_in_S_implies_a_in_S_l2477_247779


namespace NUMINAMATH_CALUDE_part_one_part_two_l2477_247767

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x - 3| - 2
def g (x : ℝ) : ℝ := 4 - |x + 1|

-- Part I
theorem part_one : 
  {x : ℝ | f x ≥ g x} = {x : ℝ | x ≥ 4 ∨ x ≤ -2} :=
by sorry

-- Part II
theorem part_two :
  {a : ℝ | ∀ x, f x - g x ≥ a^2 - 3*a} = {a : ℝ | 1 ≤ a ∧ a ≤ 2} :=
by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2477_247767


namespace NUMINAMATH_CALUDE_snow_leopard_arrangement_l2477_247728

theorem snow_leopard_arrangement (n : ℕ) (k : ℕ) (h1 : n = 10) (h2 : k = 3) :
  (k.factorial) * ((n - k).factorial) = 30240 :=
sorry

end NUMINAMATH_CALUDE_snow_leopard_arrangement_l2477_247728


namespace NUMINAMATH_CALUDE_initial_native_trees_l2477_247731

theorem initial_native_trees (N : ℕ) : 
  (3 * N - N) + (3 * N - N) / 3 = 80 → N = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_initial_native_trees_l2477_247731


namespace NUMINAMATH_CALUDE_unique_number_property_l2477_247759

theorem unique_number_property : ∃! x : ℝ, x / 3 = x - 5 := by sorry

end NUMINAMATH_CALUDE_unique_number_property_l2477_247759


namespace NUMINAMATH_CALUDE_second_discount_calculation_l2477_247720

/-- Given an initial price increase, two successive discounts, and the overall gain/loss,
    this theorem proves the relationship between these values. -/
theorem second_discount_calculation (initial_increase : ℝ) (first_discount : ℝ) 
    (overall_factor : ℝ) (second_discount : ℝ) : 
    initial_increase = 0.32 → first_discount = 0.10 → overall_factor = 0.98 →
    overall_factor = (1 - second_discount) * (1 + initial_increase) * (1 - first_discount) := by
  sorry

end NUMINAMATH_CALUDE_second_discount_calculation_l2477_247720


namespace NUMINAMATH_CALUDE_circle_tangency_problem_l2477_247797

theorem circle_tangency_problem (r : ℕ) : 
  (0 < r ∧ r < 60 ∧ 120 % r = 0) → 
  (∃ (S : Finset ℕ), S = {x : ℕ | 0 < x ∧ x < 60 ∧ 120 % x = 0} ∧ Finset.card S = 14) :=
by sorry

end NUMINAMATH_CALUDE_circle_tangency_problem_l2477_247797
