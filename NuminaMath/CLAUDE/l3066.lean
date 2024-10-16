import Mathlib

namespace NUMINAMATH_CALUDE_number_operations_l3066_306670

theorem number_operations (N : ℕ) (h1 : N % 5 = 0) (h2 : N / 5 = 4) :
  ((N - 10) * 3) - 18 = 12 := by
  sorry

end NUMINAMATH_CALUDE_number_operations_l3066_306670


namespace NUMINAMATH_CALUDE_max_product_constraint_l3066_306600

theorem max_product_constraint (a b : ℝ) (h : a^2 + b^2 = 6) : a * b ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_max_product_constraint_l3066_306600


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3066_306641

theorem complex_fraction_simplification :
  let a : ℂ := 4 + 6*I
  let b : ℂ := 4 - 6*I
  (a/b) * (b/a) + (b/a) * (a/b) = 2 := by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3066_306641


namespace NUMINAMATH_CALUDE_convenient_denominator_sum_or_diff_integer_l3066_306695

/-- A positive integer q is a convenient denominator for a real number α if 
    |α - p/q| < 1/(10q) for some integer p -/
def ConvenientDenominator (α : ℝ) (q : ℕ+) : Prop :=
  ∃ p : ℤ, |α - (p : ℝ) / q| < 1 / (10 * q)

theorem convenient_denominator_sum_or_diff_integer 
  (α β : ℝ) (hα : Irrational α) (hβ : Irrational β) :
  (∀ q : ℕ+, ConvenientDenominator α q ↔ ConvenientDenominator β q) →
  (∃ n : ℤ, α + β = n) ∨ (∃ n : ℤ, α - β = n) := by
  sorry

end NUMINAMATH_CALUDE_convenient_denominator_sum_or_diff_integer_l3066_306695


namespace NUMINAMATH_CALUDE_increasing_function_conditions_l3066_306648

-- Define the piecewise function f
noncomputable def f (a b : ℝ) : ℝ → ℝ := fun x =>
  if x ≥ 0 then x^2 + 3 else a*x + b

-- State the theorem
theorem increasing_function_conditions (a b : ℝ) :
  (∀ x y : ℝ, x < y → f a b x < f a b y) →
  (a > 0 ∧ b ≤ 3) :=
by sorry

end NUMINAMATH_CALUDE_increasing_function_conditions_l3066_306648


namespace NUMINAMATH_CALUDE_parabola_circle_tangent_line_l3066_306697

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 4*y

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the condition that a circle's center is on the parabola
def circle_center_on_parabola (c : Circle) : Prop :=
  parabola c.center.1 c.center.2

-- Define the condition that a circle passes through a point
def circle_passes_through (c : Circle) (p : ℝ × ℝ) : Prop :=
  (c.center.1 - p.1)^2 + (c.center.2 - p.2)^2 = c.radius^2

-- Define a line by its equation y = mx + b
structure Line where
  m : ℝ
  b : ℝ

-- Define the condition that a circle is tangent to a line
def circle_tangent_to_line (c : Circle) (l : Line) : Prop :=
  ∃ (x y : ℝ), y = l.m * x + l.b ∧
  (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2 ∧
  (c.center.1 - x)^2 + (c.center.2 - y)^2 = c.radius^2

theorem parabola_circle_tangent_line :
  ∀ (c : Circle) (l : Line),
  circle_center_on_parabola c →
  circle_passes_through c (0, 1) →
  circle_tangent_to_line c l →
  l.m = 0 ∧ l.b = -1 :=
sorry

end NUMINAMATH_CALUDE_parabola_circle_tangent_line_l3066_306697


namespace NUMINAMATH_CALUDE_square_sum_quadruple_l3066_306667

theorem square_sum_quadruple (n : ℕ) (h : n ≥ 8) :
  ∃ (a b c d : ℕ),
    a = 3*n^2 - 18*n - 39 ∧
    b = 3*n^2 + 6 ∧
    c = 3*n^2 + 18*n + 33 ∧
    d = 3*n^2 + 36*n + 42 ∧
    a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    ∃ (w x y z : ℕ),
      a + b + c = w^2 ∧
      a + b + d = x^2 ∧
      a + c + d = y^2 ∧
      b + c + d = z^2 :=
by sorry

end NUMINAMATH_CALUDE_square_sum_quadruple_l3066_306667


namespace NUMINAMATH_CALUDE_female_students_count_l3066_306675

theorem female_students_count (x : ℕ) : 
  (8 * x < 200) → 
  (9 * x > 200) → 
  (11 * (x + 4) > 300) → 
  x = 24 := by
sorry

end NUMINAMATH_CALUDE_female_students_count_l3066_306675


namespace NUMINAMATH_CALUDE_original_price_correct_l3066_306669

/-- The original price of the dish -/
def original_price : ℝ := 84

/-- The amount John paid -/
def john_paid (price : ℝ) : ℝ := 0.9 * price + 0.15 * price

/-- The amount Jane paid -/
def jane_paid (price : ℝ) : ℝ := 0.9 * price + 0.15 * (0.9 * price)

/-- The theorem stating that the original price satisfies the given conditions -/
theorem original_price_correct : 
  john_paid original_price = jane_paid original_price + 1.26 := by sorry

end NUMINAMATH_CALUDE_original_price_correct_l3066_306669


namespace NUMINAMATH_CALUDE_rice_containers_l3066_306601

theorem rice_containers (total_weight : ℚ) (container_weight : ℕ) (pound_to_ounce : ℕ) :
  total_weight = 35 / 2 →
  container_weight = 70 →
  pound_to_ounce = 16 →
  (total_weight * pound_to_ounce : ℚ) / container_weight = 4 :=
by sorry

end NUMINAMATH_CALUDE_rice_containers_l3066_306601


namespace NUMINAMATH_CALUDE_largest_n_for_equation_l3066_306655

theorem largest_n_for_equation : ∃ (x y z : ℕ+), 
  10^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 5*x + 5*y + 5*z - 10 ∧
  ∀ (n : ℕ+), n > 10 → ¬∃ (a b c : ℕ+), 
    n^2 = a^2 + b^2 + c^2 + 2*a*b + 2*b*c + 2*c*a + 5*a + 5*b + 5*c - 10 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_for_equation_l3066_306655


namespace NUMINAMATH_CALUDE_max_value_of_product_sum_l3066_306652

theorem max_value_of_product_sum (w x y z : ℝ) 
  (nonneg_w : w ≥ 0) (nonneg_x : x ≥ 0) (nonneg_y : y ≥ 0) (nonneg_z : z ≥ 0)
  (sum_eq_200 : w + x + y + z = 200) : 
  wx + xy + yz ≤ 10000 := by
sorry

end NUMINAMATH_CALUDE_max_value_of_product_sum_l3066_306652


namespace NUMINAMATH_CALUDE_x_plus_y_value_l3066_306624

theorem x_plus_y_value (x y : ℝ) 
  (eq1 : x + Real.cos y = 2010)
  (eq2 : x + 2010 * Real.sin y = 2009)
  (y_range : π / 2 ≤ y ∧ y ≤ π) :
  x + y = 2011 + π := by
  sorry

end NUMINAMATH_CALUDE_x_plus_y_value_l3066_306624


namespace NUMINAMATH_CALUDE_exists_same_color_rectangle_l3066_306604

/-- A color type representing red, black, and blue -/
inductive Color
  | Red
  | Black
  | Blue

/-- A type representing a point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A function that assigns a color to each point in the plane -/
def colorFunction : Point → Color := sorry

/-- A type representing a rectangle in the plane -/
structure Rectangle where
  topLeft : Point
  topRight : Point
  bottomLeft : Point
  bottomRight : Point

/-- A predicate that checks if all vertices of a rectangle have the same color -/
def sameColorVertices (rect : Rectangle) : Prop :=
  colorFunction rect.topLeft = colorFunction rect.topRight ∧
  colorFunction rect.topLeft = colorFunction rect.bottomLeft ∧
  colorFunction rect.topLeft = colorFunction rect.bottomRight

/-- Theorem stating that there exists a rectangle with vertices of the same color -/
theorem exists_same_color_rectangle : ∃ (rect : Rectangle), sameColorVertices rect := by
  sorry


end NUMINAMATH_CALUDE_exists_same_color_rectangle_l3066_306604


namespace NUMINAMATH_CALUDE_hcl_concentration_in_mixed_solution_l3066_306660

/-- Calculates the concentration of HCl in a mixed solution -/
theorem hcl_concentration_in_mixed_solution 
  (volume1 : ℝ) (concentration1 : ℝ) 
  (volume2 : ℝ) (concentration2 : ℝ) :
  volume1 = 60 →
  concentration1 = 0.4 →
  volume2 = 90 →
  concentration2 = 0.15 →
  (volume1 * concentration1 + volume2 * concentration2) / (volume1 + volume2) = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_hcl_concentration_in_mixed_solution_l3066_306660


namespace NUMINAMATH_CALUDE_two_thousand_two_in_sequence_l3066_306662

def next_in_sequence (b : ℕ) : ℕ :=
  b + (Nat.factors b).reverse.head!

def is_in_sequence (a : ℕ) (n : ℕ) : Prop :=
  ∃ k : ℕ, Nat.iterate next_in_sequence k a = n

theorem two_thousand_two_in_sequence (a : ℕ) :
  a > 1 → (is_in_sequence a 2002 ↔ a = 1859 ∨ a = 1991) :=
sorry

end NUMINAMATH_CALUDE_two_thousand_two_in_sequence_l3066_306662


namespace NUMINAMATH_CALUDE_quadratic_equation_condition_l3066_306653

theorem quadratic_equation_condition (m : ℝ) : 
  (m^2 - 2 = 2 ∧ m + 2 ≠ 0) ↔ m = 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_condition_l3066_306653


namespace NUMINAMATH_CALUDE_solution_is_negative_two_l3066_306657

/-- The equation we want to solve -/
def equation (x : ℝ) : Prop := 2 / x = 1 / (x + 1)

/-- The theorem stating that -2 is the solution to the equation -/
theorem solution_is_negative_two : ∃ x : ℝ, equation x ∧ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_solution_is_negative_two_l3066_306657


namespace NUMINAMATH_CALUDE_cost_of_dozen_pens_l3066_306666

/-- Given the cost of 3 pens and 5 pencils, and the cost ratio of pen to pencil,
    prove the cost of one dozen pens -/
theorem cost_of_dozen_pens (cost_3pens_5pencils : ℕ) (ratio_pen_pencil : ℚ) :
  cost_3pens_5pencils = 240 →
  ratio_pen_pencil = 5 / 1 →
  (12 : ℕ) * (5 * (cost_3pens_5pencils / (3 * 5 + 5))) = 720 := by
sorry

end NUMINAMATH_CALUDE_cost_of_dozen_pens_l3066_306666


namespace NUMINAMATH_CALUDE_parabola_values_l3066_306647

/-- A parabola passing through specific points -/
structure Parabola where
  a : ℝ
  b : ℝ
  eq : ℝ → ℝ := λ x => x^2 + a * x + b
  point1 : eq 2 = 20
  point2 : eq (-2) = 0
  point3 : eq 0 = b

/-- The values of a and b for the given parabola -/
theorem parabola_values (p : Parabola) : p.a = 5 ∧ p.b = 6 := by
  sorry

end NUMINAMATH_CALUDE_parabola_values_l3066_306647


namespace NUMINAMATH_CALUDE_pen_price_calculation_l3066_306610

theorem pen_price_calculation (total_cost : ℝ) (num_pens : ℕ) (num_pencils : ℕ) (pencil_price : ℝ) :
  total_cost = 630 ∧ num_pens = 30 ∧ num_pencils = 75 ∧ pencil_price = 2 →
  (total_cost - num_pencils * pencil_price) / num_pens = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_pen_price_calculation_l3066_306610


namespace NUMINAMATH_CALUDE_spinner_direction_l3066_306632

-- Define the directions
inductive Direction
  | North
  | East
  | South
  | West

-- Define the rotation function
def rotate (initial : Direction) (clockwise : Rat) (counterclockwise : Rat) : Direction :=
  sorry

-- Theorem statement
theorem spinner_direction :
  let initial_direction := Direction.North
  let clockwise_rotation := 3 + 3/4
  let counterclockwise_rotation := 2 + 2/4
  rotate initial_direction clockwise_rotation counterclockwise_rotation = Direction.East :=
sorry

end NUMINAMATH_CALUDE_spinner_direction_l3066_306632


namespace NUMINAMATH_CALUDE_point_transformation_l3066_306664

def rotate90CCW (x y : ℝ) : ℝ × ℝ := (-y, x)

def reflectAboutYeqX (x y : ℝ) : ℝ × ℝ := (y, x)

theorem point_transformation (a b : ℝ) :
  let (x₁, y₁) := rotate90CCW a b
  let (x₂, y₂) := reflectAboutYeqX x₁ y₁
  (x₂ = 3 ∧ y₂ = -7) → b - a = 4 := by sorry

end NUMINAMATH_CALUDE_point_transformation_l3066_306664


namespace NUMINAMATH_CALUDE_sqrt_D_irrational_l3066_306682

def D (x : ℝ) : ℝ := 6 * x^2 + 4 * x + 4

theorem sqrt_D_irrational : ∀ x : ℝ, Irrational (Real.sqrt (D x)) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_D_irrational_l3066_306682


namespace NUMINAMATH_CALUDE_quadratic_factorization_l3066_306688

theorem quadratic_factorization (x : ℝ) : 
  (x^2 - 6*x - 11 = 0) ↔ ((x - 3)^2 = 20) := by
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l3066_306688


namespace NUMINAMATH_CALUDE_specific_cylinder_properties_l3066_306698

/-- Represents a cylinder with height and surface area as parameters. -/
structure Cylinder where
  height : ℝ
  surfaceArea : ℝ

/-- Calculates the radius of the base circle of a cylinder. -/
def baseRadius (c : Cylinder) : ℝ :=
  sorry

/-- Calculates the volume of a cylinder. -/
def volume (c : Cylinder) : ℝ :=
  sorry

/-- Theorem stating the properties of a specific cylinder. -/
theorem specific_cylinder_properties :
  let c := Cylinder.mk 8 (130 * Real.pi)
  baseRadius c = 5 ∧ volume c = 200 * Real.pi :=
sorry

end NUMINAMATH_CALUDE_specific_cylinder_properties_l3066_306698


namespace NUMINAMATH_CALUDE_dealership_sales_expectation_l3066_306645

/-- The number of trucks expected to be sold -/
def expected_trucks : ℕ := 30

/-- The number of vans expected to be sold -/
def expected_vans : ℕ := 15

/-- The ratio of trucks to SUVs -/
def truck_suv_ratio : ℚ := 3 / 5

/-- The ratio of SUVs to vans -/
def suv_van_ratio : ℚ := 2 / 1

/-- The number of SUVs the dealership should expect to sell -/
def expected_suvs : ℕ := 30

theorem dealership_sales_expectation :
  (expected_trucks : ℚ) / truck_suv_ratio ≥ expected_suvs ∧
  suv_van_ratio * expected_vans = expected_suvs :=
sorry

end NUMINAMATH_CALUDE_dealership_sales_expectation_l3066_306645


namespace NUMINAMATH_CALUDE_school_seats_cost_l3066_306629

/-- Calculate the total cost of seats with a group discount -/
def totalCostWithDiscount (rows : ℕ) (seatsPerRow : ℕ) (costPerSeat : ℕ) (discountPercent : ℕ) : ℕ :=
  let totalSeats := rows * seatsPerRow
  let fullGroupsOf10 := totalSeats / 10
  let costPer10Seats := 10 * costPerSeat
  let discountPer10Seats := costPer10Seats * discountPercent / 100
  let costPer10SeatsAfterDiscount := costPer10Seats - discountPer10Seats
  fullGroupsOf10 * costPer10SeatsAfterDiscount

theorem school_seats_cost :
  totalCostWithDiscount 5 8 30 10 = 1080 := by
  sorry

end NUMINAMATH_CALUDE_school_seats_cost_l3066_306629


namespace NUMINAMATH_CALUDE_power_comparison_l3066_306617

theorem power_comparison : (2 : ℕ)^16 / (16 : ℕ)^2 = 256 := by sorry

end NUMINAMATH_CALUDE_power_comparison_l3066_306617


namespace NUMINAMATH_CALUDE_cone_base_radius_l3066_306635

/-- A cone with surface area 3π and lateral surface that unfolds into a semicircle has a base radius of 1 -/
theorem cone_base_radius (r : ℝ) (l : ℝ) : 
  r > 0 → l > 0 → 
  l = 2 * r →  -- Lateral surface unfolds into a semicircle
  3 * π * r^2 = 3 * π →  -- Surface area is 3π
  r = 1 := by sorry

end NUMINAMATH_CALUDE_cone_base_radius_l3066_306635


namespace NUMINAMATH_CALUDE_max_product_sum_max_product_sum_achieved_l3066_306606

theorem max_product_sum (A M C : ℕ) (h : A + M + C = 15) :
  (A * M * C + A * M + M * C + C * A) ≤ 200 :=
by sorry

theorem max_product_sum_achieved :
  ∃ A M C : ℕ, A + M + C = 15 ∧ A * M * C + A * M + M * C + C * A = 200 :=
by sorry

end NUMINAMATH_CALUDE_max_product_sum_max_product_sum_achieved_l3066_306606


namespace NUMINAMATH_CALUDE_work_completion_time_l3066_306671

/-- The time it takes for worker c to complete the work alone -/
def time_for_c : ℚ := 24

/-- The combined work rate of workers a and b -/
def rate_ab : ℚ := 1/3

/-- The combined work rate of workers b and c -/
def rate_bc : ℚ := 1/4

/-- The combined work rate of workers c and a -/
def rate_ca : ℚ := 1/6

theorem work_completion_time :
  rate_ab + rate_bc + rate_ca = 3/8 →
  (1 : ℚ) / time_for_c = 3/8 - rate_ab :=
by sorry

end NUMINAMATH_CALUDE_work_completion_time_l3066_306671


namespace NUMINAMATH_CALUDE_equation_solution_l3066_306672

theorem equation_solution : ∃ x : ℝ, (2 / (x + 5) = 1 / (3 * x)) ∧ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3066_306672


namespace NUMINAMATH_CALUDE_buratino_problem_l3066_306614

/-- The sum of a geometric sequence with first term 1 and common ratio 2 -/
def geometricSum (n : ℕ) : ℕ := 2^n - 1

/-- The total payment in kopeks -/
def totalPayment : ℕ := 65535

theorem buratino_problem :
  ∃ n : ℕ, geometricSum n = totalPayment ∧ n = 16 := by
  sorry

end NUMINAMATH_CALUDE_buratino_problem_l3066_306614


namespace NUMINAMATH_CALUDE_tax_rate_is_ten_percent_l3066_306637

/-- The tax rate for properties in Township K -/
def tax_rate : ℝ := sorry

/-- The initial assessed value of the property -/
def initial_value : ℝ := 20000

/-- The final assessed value of the property -/
def final_value : ℝ := 28000

/-- The increase in property tax -/
def tax_increase : ℝ := 800

/-- Theorem stating that the tax rate is 10% of the assessed value -/
theorem tax_rate_is_ten_percent :
  tax_rate = 0.1 :=
by
  sorry

#check tax_rate_is_ten_percent

end NUMINAMATH_CALUDE_tax_rate_is_ten_percent_l3066_306637


namespace NUMINAMATH_CALUDE_parallel_line_equation_line_K_equation_l3066_306680

/-- Given a line with equation y = mx + b, this function returns the y-intercept of a parallel line
    that is d units away from the original line. -/
def parallelLineYIntercept (m : ℝ) (b : ℝ) (d : ℝ) : Set ℝ :=
  {y | ∃ (sign : ℝ), sign = 1 ∨ sign = -1 ∧ y = b + sign * d * Real.sqrt (m^2 + 1)}

theorem parallel_line_equation (m b d : ℝ) :
  parallelLineYIntercept m b d = {b + d * Real.sqrt (m^2 + 1), b - d * Real.sqrt (m^2 + 1)} := by
  sorry

/-- The equation of line K, which is parallel to y = 1/2x + 3 and 5 units away from it. -/
theorem line_K_equation :
  parallelLineYIntercept (1/2) 3 5 = {3 + 5 * Real.sqrt 5 / 2, 3 - 5 * Real.sqrt 5 / 2} := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_equation_line_K_equation_l3066_306680


namespace NUMINAMATH_CALUDE_correct_flight_distance_l3066_306603

/-- The total distance Peter needs to fly from Germany to Russia and then back to Spain -/
def total_flight_distance (spain_russia_distance spain_germany_distance : ℕ) : ℕ :=
  (spain_russia_distance - spain_germany_distance) + 2 * spain_germany_distance

/-- Theorem stating the correct total flight distance given the problem conditions -/
theorem correct_flight_distance :
  total_flight_distance 7019 1615 = 8634 := by
  sorry

end NUMINAMATH_CALUDE_correct_flight_distance_l3066_306603


namespace NUMINAMATH_CALUDE_circle_triangle_area_difference_l3066_306636

/-- Given an equilateral triangle with side length 12 units and its circumscribed circle,
    the difference between the area of the circle and the area of the triangle
    is 144π - 36√3 square units. -/
theorem circle_triangle_area_difference : 
  let s : ℝ := 12 -- side length of the equilateral triangle
  let r : ℝ := s -- radius of the circumscribed circle (equal to side length)
  let circle_area : ℝ := π * r^2
  let triangle_height : ℝ := s * (Real.sqrt 3) / 2
  let triangle_area : ℝ := (1/2) * s * triangle_height
  circle_area - triangle_area = 144 * π - 36 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_circle_triangle_area_difference_l3066_306636


namespace NUMINAMATH_CALUDE_jungkook_smallest_l3066_306608

def yoongi_collection : ℕ := 4
def jungkook_collection : ℚ := 6 / 3
def yuna_collection : ℕ := 5

theorem jungkook_smallest :
  jungkook_collection < yoongi_collection ∧ jungkook_collection < yuna_collection :=
sorry

end NUMINAMATH_CALUDE_jungkook_smallest_l3066_306608


namespace NUMINAMATH_CALUDE_unique_value_of_a_l3066_306630

theorem unique_value_of_a (a b x : ℝ) 
  (h1 : a ≠ b) 
  (h2 : a^3 - b^3 = 27*x^3) 
  (h3 : a - b = 3*x) : 
  a = 3*x :=
sorry

end NUMINAMATH_CALUDE_unique_value_of_a_l3066_306630


namespace NUMINAMATH_CALUDE_ways_to_buy_three_items_eq_646_l3066_306646

/-- Represents the inventory of a store --/
structure Inventory where
  headphones : Nat
  mice : Nat
  keyboards : Nat
  keyboard_mouse_sets : Nat
  headphone_mouse_sets : Nat

/-- Calculates the number of ways to buy three items (headphones, keyboard, mouse) --/
def ways_to_buy_three_items (inv : Inventory) : Nat :=
  inv.keyboard_mouse_sets * inv.headphones +
  inv.headphone_mouse_sets * inv.keyboards +
  inv.headphones * inv.mice * inv.keyboards

/-- The theorem stating the number of ways to buy three items --/
theorem ways_to_buy_three_items_eq_646 (inv : Inventory) 
  (h1 : inv.headphones = 9)
  (h2 : inv.mice = 13)
  (h3 : inv.keyboards = 5)
  (h4 : inv.keyboard_mouse_sets = 4)
  (h5 : inv.headphone_mouse_sets = 5) :
  ways_to_buy_three_items inv = 646 := by
  sorry


end NUMINAMATH_CALUDE_ways_to_buy_three_items_eq_646_l3066_306646


namespace NUMINAMATH_CALUDE_quadratic_inequality_l3066_306605

theorem quadratic_inequality (a b c : ℝ) :
  (∀ x : ℝ, |x| ≤ 1 → |a * x^2 + b * x + c| ≤ 1) →
  (∀ x : ℝ, |x| ≤ 1 → |c * x^2 + b * x + a| ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l3066_306605


namespace NUMINAMATH_CALUDE_lcm_gcd_equality_l3066_306694

theorem lcm_gcd_equality (a b c : ℕ+) :
  (Nat.lcm a (Nat.lcm b c))^2 / (Nat.lcm a b * Nat.lcm b c * Nat.lcm c a) =
  (Nat.gcd a (Nat.gcd b c))^2 / (Nat.gcd a b * Nat.gcd b c * Nat.gcd c a) :=
sorry

end NUMINAMATH_CALUDE_lcm_gcd_equality_l3066_306694


namespace NUMINAMATH_CALUDE_half_plus_five_equals_thirteen_l3066_306679

theorem half_plus_five_equals_thirteen (n : ℕ) (h : n = 16) : n / 2 + 5 = 13 := by
  sorry

end NUMINAMATH_CALUDE_half_plus_five_equals_thirteen_l3066_306679


namespace NUMINAMATH_CALUDE_french_students_count_l3066_306696

theorem french_students_count (total : ℕ) (german : ℕ) (both : ℕ) (neither : ℕ) 
  (h_total : total = 87)
  (h_german : german = 22)
  (h_both : both = 9)
  (h_neither : neither = 33) :
  ∃ french : ℕ, french = total - german + both - neither :=
by
  sorry

end NUMINAMATH_CALUDE_french_students_count_l3066_306696


namespace NUMINAMATH_CALUDE_cube_surface_area_l3066_306686

theorem cube_surface_area (a : ℝ) (h : a > 0) : 
  ∃ (s : ℝ), s > 0 ∧ s * Real.sqrt 3 = a ∧ 6 * s^2 = 2 * a^2 :=
sorry

end NUMINAMATH_CALUDE_cube_surface_area_l3066_306686


namespace NUMINAMATH_CALUDE_sequence_general_term_l3066_306627

/-- Given a sequence a_n with sum S_n satisfying S_n = 3 - 2a_n,
    prove that the general term of a_n is (2/3)^(n-1) -/
theorem sequence_general_term (a : ℕ → ℚ) (S : ℕ → ℚ)
    (h : ∀ n, S n = 3 - 2 * a n) :
  ∀ n, a n = (2/3)^(n-1) := by
sorry

end NUMINAMATH_CALUDE_sequence_general_term_l3066_306627


namespace NUMINAMATH_CALUDE_middle_integer_of_consecutive_sum_l3066_306612

theorem middle_integer_of_consecutive_sum (n : ℤ) : 
  (n - 1) + n + (n + 1) = 180 → n = 60 := by
  sorry

end NUMINAMATH_CALUDE_middle_integer_of_consecutive_sum_l3066_306612


namespace NUMINAMATH_CALUDE_complementary_implies_mutually_exclusive_l3066_306656

/-- Two events are complementary if one event occurs if and only if the other does not occur -/
def complementary_events (Ω : Type*) (A B : Set Ω) : Prop :=
  A = (Bᶜ)

/-- Two events are mutually exclusive if they cannot occur simultaneously -/
def mutually_exclusive (Ω : Type*) (A B : Set Ω) : Prop :=
  A ∩ B = ∅

/-- The probability of an event is a number between 0 and 1 inclusive -/
axiom probability_range (Ω : Type*) (A : Set Ω) :
  ∃ (P : Set Ω → ℝ), 0 ≤ P A ∧ P A ≤ 1

theorem complementary_implies_mutually_exclusive (Ω : Type*) (A B : Set Ω) :
  complementary_events Ω A B → mutually_exclusive Ω A B :=
sorry

end NUMINAMATH_CALUDE_complementary_implies_mutually_exclusive_l3066_306656


namespace NUMINAMATH_CALUDE_food_drive_cans_l3066_306658

theorem food_drive_cans (mark jaydon rachel : ℕ) : 
  mark = 100 ∧ 
  mark = 4 * jaydon ∧ 
  jaydon > 2 * rachel ∧ 
  mark + jaydon + rachel = 135 → 
  jaydon = 2 * rachel + 5 :=
by sorry

end NUMINAMATH_CALUDE_food_drive_cans_l3066_306658


namespace NUMINAMATH_CALUDE_unique_solution_trigonometric_equation_l3066_306644

theorem unique_solution_trigonometric_equation :
  ∃! x : ℝ, 2 * Real.sin (π * x / 2) - 2 * Real.cos (π * x / 2) = x^5 + 10*x - 54 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_trigonometric_equation_l3066_306644


namespace NUMINAMATH_CALUDE_equal_digits_probability_l3066_306651

/-- The number of sides on each die -/
def num_sides : ℕ := 20

/-- The number of dice rolled -/
def num_dice : ℕ := 6

/-- The number of one-digit outcomes on a die -/
def one_digit_outcomes : ℕ := 9

/-- The number of two-digit outcomes on a die -/
def two_digit_outcomes : ℕ := 11

/-- The probability of rolling an equal number of one-digit and two-digit numbers on 6 20-sided dice -/
def equal_digits_prob : ℚ := 4851495 / 16000000

theorem equal_digits_probability : 
  let p_one_digit := one_digit_outcomes / num_sides
  let p_two_digit := two_digit_outcomes / num_sides
  let combinations := Nat.choose num_dice (num_dice / 2)
  combinations * (p_one_digit ^ (num_dice / 2)) * (p_two_digit ^ (num_dice / 2)) = equal_digits_prob := by
  sorry

end NUMINAMATH_CALUDE_equal_digits_probability_l3066_306651


namespace NUMINAMATH_CALUDE_arithmetic_sequence_difference_l3066_306668

-- Define an arithmetic sequence
def isArithmeticSequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

-- Theorem statement
theorem arithmetic_sequence_difference
  (a : ℕ → ℝ) (d : ℝ) (h : isArithmeticSequence a d) (h_d : d = 2) :
  a 5 - a 2 = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_difference_l3066_306668


namespace NUMINAMATH_CALUDE_range_of_z_l3066_306693

theorem range_of_z (x y z : ℝ) (hx : -1 ≤ x ∧ x ≤ 2) (hy : 0 ≤ y ∧ y ≤ 1) (hz : z = 2*x - y) :
  ∃ (a b : ℝ), a = -3 ∧ b = 4 ∧ (∀ w, (∃ x y, (-1 ≤ x ∧ x ≤ 2) ∧ (0 ≤ y ∧ y ≤ 1) ∧ w = 2*x - y) ↔ a ≤ w ∧ w ≤ b) :=
sorry

end NUMINAMATH_CALUDE_range_of_z_l3066_306693


namespace NUMINAMATH_CALUDE_sum_of_fifth_and_sixth_term_l3066_306699

theorem sum_of_fifth_and_sixth_term (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h : ∀ n, S n = n^3) : a 5 + a 6 = 152 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fifth_and_sixth_term_l3066_306699


namespace NUMINAMATH_CALUDE_decimal_88_to_base5_base5_323_to_decimal_decimal_88_equals_base5_323_l3066_306673

/-- Converts a natural number to its base-5 representation -/
def toBase5 (n : ℕ) : List ℕ :=
  if n < 5 then [n]
  else (n % 5) :: toBase5 (n / 5)

/-- Converts a list of digits in base-5 to its decimal representation -/
def fromBase5 (digits : List ℕ) : ℕ :=
  digits.foldr (fun d acc => d + 5 * acc) 0

theorem decimal_88_to_base5 :
  toBase5 88 = [3, 2, 3] :=
sorry

theorem base5_323_to_decimal :
  fromBase5 [3, 2, 3] = 88 :=
sorry

/-- The base-5 representation of 88 is 323 -/
theorem decimal_88_equals_base5_323 :
  toBase5 88 = [3, 2, 3] ∧ fromBase5 [3, 2, 3] = 88 :=
sorry

end NUMINAMATH_CALUDE_decimal_88_to_base5_base5_323_to_decimal_decimal_88_equals_base5_323_l3066_306673


namespace NUMINAMATH_CALUDE_book_count_proof_l3066_306607

/-- Proves that given a total of 144 books and a ratio of 7:5 for storybooks to science books,
    the number of storybooks is 84 and the number of science books is 60. -/
theorem book_count_proof (total : ℕ) (storybook_ratio : ℕ) (science_ratio : ℕ)
    (h_total : total = 144)
    (h_ratio : (storybook_ratio : ℚ) / (science_ratio : ℚ) = 7 / 5) :
    ∃ (storybooks science_books : ℕ),
      storybooks = 84 ∧
      science_books = 60 ∧
      storybooks + science_books = total ∧
      (storybooks : ℚ) / (science_books : ℚ) = storybook_ratio / science_ratio :=
by
  sorry

end NUMINAMATH_CALUDE_book_count_proof_l3066_306607


namespace NUMINAMATH_CALUDE_rain_probabilities_l3066_306633

theorem rain_probabilities (p_monday p_tuesday : ℝ) 
  (h_monday : p_monday = 0.4)
  (h_tuesday : p_tuesday = 0.3)
  (h_independent : True)  -- This represents the independence assumption
  : (p_monday * p_tuesday = 0.12) ∧ 
    ((1 - p_monday) * (1 - p_tuesday) = 0.42) := by
  sorry

end NUMINAMATH_CALUDE_rain_probabilities_l3066_306633


namespace NUMINAMATH_CALUDE_equation_solution_l3066_306677

theorem equation_solution :
  let f (x : ℝ) := 4 / (Real.sqrt (x + 5) - 7) + 3 / (Real.sqrt (x + 5) - 2) +
                   6 / (Real.sqrt (x + 5) + 2) + 9 / (Real.sqrt (x + 5) + 7)
  {x : ℝ | f x = 0} = {-796/169, 383/22} := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l3066_306677


namespace NUMINAMATH_CALUDE_bus_time_is_ten_l3066_306674

/-- Represents the travel times and conditions for Xiaoming's journey --/
structure TravelTimes where
  total : ℕ  -- Total travel time
  transfer : ℕ  -- Transfer time
  subway_only : ℕ  -- Time if only taking subway
  bus_only : ℕ  -- Time if only taking bus

/-- Calculates the time spent on the bus given the travel times --/
def time_on_bus (t : TravelTimes) : ℕ :=
  let actual_travel_time := t.total - t.transfer
  let extra_time := actual_travel_time - t.subway_only
  let time_unit := extra_time / (t.bus_only / 10 - t.subway_only / 10)
  (t.bus_only / 10) * time_unit

/-- Theorem stating that given the specific travel times, the time spent on the bus is 10 minutes --/
theorem bus_time_is_ten : 
  let t : TravelTimes := { 
    total := 40, 
    transfer := 6, 
    subway_only := 30, 
    bus_only := 50 
  }
  time_on_bus t = 10 := by
  sorry


end NUMINAMATH_CALUDE_bus_time_is_ten_l3066_306674


namespace NUMINAMATH_CALUDE_arcade_change_machine_l3066_306634

theorem arcade_change_machine (total_bills : ℕ) (one_dollar_bills : ℕ) : 
  total_bills = 200 → one_dollar_bills = 175 → 
  (total_bills - one_dollar_bills) * 5 + one_dollar_bills = 300 := by
  sorry

end NUMINAMATH_CALUDE_arcade_change_machine_l3066_306634


namespace NUMINAMATH_CALUDE_f_symmetric_about_origin_l3066_306640

def f (x : ℝ) : ℝ := x^3 + x

theorem f_symmetric_about_origin : ∀ x : ℝ, f (-x) = -f x := by sorry

end NUMINAMATH_CALUDE_f_symmetric_about_origin_l3066_306640


namespace NUMINAMATH_CALUDE_closest_integer_to_k_l3066_306665

theorem closest_integer_to_k : ∃ (k : ℝ), 
  k = Real.sqrt 2 * ((Real.sqrt 5 + Real.sqrt 3) * (Real.sqrt 5 - Real.sqrt 3)) ∧
  ∀ (n : ℤ), |k - 3| ≤ |k - n| :=
by sorry

end NUMINAMATH_CALUDE_closest_integer_to_k_l3066_306665


namespace NUMINAMATH_CALUDE_broken_seashells_l3066_306691

theorem broken_seashells (total : ℕ) (unbroken : ℕ) (h1 : total = 7) (h2 : unbroken = 3) :
  total - unbroken = 4 := by
  sorry

end NUMINAMATH_CALUDE_broken_seashells_l3066_306691


namespace NUMINAMATH_CALUDE_no_infinite_sqrt_sequence_l3066_306692

theorem no_infinite_sqrt_sequence :
  ¬ (∃ (a : ℕ → ℕ+), ∀ (n : ℕ), n ≥ 1 → (a (n + 2)).val = Int.sqrt ((a (n + 1)).val) + (a n).val) :=
by sorry

end NUMINAMATH_CALUDE_no_infinite_sqrt_sequence_l3066_306692


namespace NUMINAMATH_CALUDE_fraction_equality_proof_l3066_306687

theorem fraction_equality_proof (a b z : ℕ+) (h : a * b = z^2 + 1) :
  ∃ (x y : ℕ+), (a : ℚ) / b = ((x^2 : ℚ) + 1) / ((y^2 : ℚ) + 1) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_proof_l3066_306687


namespace NUMINAMATH_CALUDE_spaceship_journey_theorem_l3066_306628

/-- Represents the travel schedule of a spaceship --/
structure SpaceshipJourney where
  totalJourneyTime : ℕ
  firstDayTravelTime1 : ℕ
  firstDayBreakTime1 : ℕ
  firstDayTravelTime2 : ℕ
  firstDayBreakTime2 : ℕ
  routineTravelTime : ℕ
  routineBreakTime : ℕ

/-- Calculates the total time the spaceship was not moving during its journey --/
def totalNotMovingTime (journey : SpaceshipJourney) : ℕ :=
  let firstDayBreakTime := journey.firstDayBreakTime1 + journey.firstDayBreakTime2
  let firstDayTotalTime := journey.firstDayTravelTime1 + journey.firstDayTravelTime2 + firstDayBreakTime
  let remainingTime := journey.totalJourneyTime - firstDayTotalTime
  let routineBlockTime := journey.routineTravelTime + journey.routineBreakTime
  let routineBlocks := remainingTime / routineBlockTime
  firstDayBreakTime + routineBlocks * journey.routineBreakTime

theorem spaceship_journey_theorem (journey : SpaceshipJourney) 
  (h1 : journey.totalJourneyTime = 72)
  (h2 : journey.firstDayTravelTime1 = 10)
  (h3 : journey.firstDayBreakTime1 = 3)
  (h4 : journey.firstDayTravelTime2 = 10)
  (h5 : journey.firstDayBreakTime2 = 1)
  (h6 : journey.routineTravelTime = 11)
  (h7 : journey.routineBreakTime = 1) :
  totalNotMovingTime journey = 8 := by
  sorry

end NUMINAMATH_CALUDE_spaceship_journey_theorem_l3066_306628


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l3066_306625

theorem fractional_equation_solution : 
  ∃ (x : ℝ), (x ≠ 0 ∧ x ≠ 2) ∧ (5 / (x - 2) = 3 / x) ∧ (x = -3) := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l3066_306625


namespace NUMINAMATH_CALUDE_max_third_term_in_arithmetic_sequence_l3066_306639

def is_arithmetic_sequence (a b c d : ℕ) : Prop :=
  b - a = c - b ∧ c - b = d - c

theorem max_third_term_in_arithmetic_sequence :
  ∀ a b c d : ℕ,
  a > 0 → b > 0 → c > 0 → d > 0 →
  is_arithmetic_sequence a b c d →
  a + b + c + d = 50 →
  c ≤ 16 :=
by sorry

end NUMINAMATH_CALUDE_max_third_term_in_arithmetic_sequence_l3066_306639


namespace NUMINAMATH_CALUDE_initial_bees_count_l3066_306609

/-- Given a hive where 8 bees fly in and the total becomes 24, prove that there were initially 16 bees. -/
theorem initial_bees_count (initial_bees : ℕ) : initial_bees + 8 = 24 → initial_bees = 16 := by
  sorry

end NUMINAMATH_CALUDE_initial_bees_count_l3066_306609


namespace NUMINAMATH_CALUDE_positive_numbers_inequality_l3066_306659

theorem positive_numbers_inequality (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum_squares : a^2 + b^2 + 4*c^2 = 3) : 
  (a + b + 2*c ≤ 3) ∧ (b = 2*c → 1/a + 1/c ≥ 3) := by
  sorry

end NUMINAMATH_CALUDE_positive_numbers_inequality_l3066_306659


namespace NUMINAMATH_CALUDE_two_numbers_difference_l3066_306683

theorem two_numbers_difference (x y : ℝ) : 
  x + y = 40 → 
  3 * y - 2 * x = 10 → 
  |x - y| = 4 := by
sorry

end NUMINAMATH_CALUDE_two_numbers_difference_l3066_306683


namespace NUMINAMATH_CALUDE_max_goals_scored_l3066_306676

/-- Represents the number of goals scored by Marlon in a soccer game --/
def goals_scored (penalty_shots free_kicks : ℕ) : ℝ :=
  0.4 * penalty_shots + 0.5 * free_kicks

/-- Proves that the maximum number of goals Marlon could have scored is 20 --/
theorem max_goals_scored : 
  ∀ penalty_shots free_kicks : ℕ, 
  penalty_shots + free_kicks = 40 →
  goals_scored penalty_shots free_kicks ≤ 20 :=
by
  sorry

#check max_goals_scored

end NUMINAMATH_CALUDE_max_goals_scored_l3066_306676


namespace NUMINAMATH_CALUDE_museum_trip_ratio_l3066_306654

theorem museum_trip_ratio : 
  ∀ (p1 p2 p3 p4 : ℕ),
  p1 = 12 →
  p3 = p2 - 6 →
  p4 = p1 + 9 →
  p1 + p2 + p3 + p4 = 75 →
  p2 / p1 = 2 := by
sorry

end NUMINAMATH_CALUDE_museum_trip_ratio_l3066_306654


namespace NUMINAMATH_CALUDE_simple_interest_rate_equivalence_l3066_306631

theorem simple_interest_rate_equivalence (P : ℝ) (P_pos : P > 0) :
  let initial_rate : ℝ := 5 / 100
  let initial_time : ℝ := 8
  let new_time : ℝ := 5
  let new_rate : ℝ := 8 / 100
  (P * initial_rate * initial_time) = (P * new_rate * new_time) := by
sorry

end NUMINAMATH_CALUDE_simple_interest_rate_equivalence_l3066_306631


namespace NUMINAMATH_CALUDE_magic_8_ball_probability_l3066_306649

/-- The probability of getting exactly k successes in n independent trials,
    where each trial has a success probability of p -/
def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  Nat.choose n k * p^k * (1 - p)^(n - k)

/-- The probability of getting exactly 3 positive answers out of 8 questions
    with the Magic 8 Ball, where each question has a 2/5 chance of a positive answer -/
theorem magic_8_ball_probability : 
  binomial_probability 8 3 (2/5) = 108864/390625 := by
  sorry

end NUMINAMATH_CALUDE_magic_8_ball_probability_l3066_306649


namespace NUMINAMATH_CALUDE_root_in_interval_l3066_306613

def f (x : ℝ) := 2*x + 3*x - 7

theorem root_in_interval :
  ∃ r ∈ Set.Ioo 1 2, f r = 0 :=
sorry

end NUMINAMATH_CALUDE_root_in_interval_l3066_306613


namespace NUMINAMATH_CALUDE_omega_sum_equality_l3066_306643

theorem omega_sum_equality (ω : ℂ) (h1 : ω^5 = 1) (h2 : ω ≠ 1) :
  ω^15 + ω^18 + ω^21 + ω^24 + ω^27 + ω^30 + ω^33 + ω^36 + ω^39 + ω^42 + ω^45 = 1 := by
sorry

end NUMINAMATH_CALUDE_omega_sum_equality_l3066_306643


namespace NUMINAMATH_CALUDE_college_application_fee_cost_l3066_306620

/-- Proves that the cost of each college application fee is $25.00 -/
theorem college_application_fee_cost 
  (hourly_rate : ℝ) 
  (num_colleges : ℕ) 
  (hours_worked : ℕ) 
  (h1 : hourly_rate = 10)
  (h2 : num_colleges = 6)
  (h3 : hours_worked = 15) :
  (hourly_rate * hours_worked) / num_colleges = 25 := by
sorry

end NUMINAMATH_CALUDE_college_application_fee_cost_l3066_306620


namespace NUMINAMATH_CALUDE_lcm_5_6_8_9_l3066_306661

theorem lcm_5_6_8_9 : Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 8 9)) = 360 := by
  sorry

end NUMINAMATH_CALUDE_lcm_5_6_8_9_l3066_306661


namespace NUMINAMATH_CALUDE_jerry_age_l3066_306663

/-- Given that Mickey's age is 16 and Mickey's age is 6 years less than 200% of Jerry's age,
    prove that Jerry's age is 11. -/
theorem jerry_age (mickey_age jerry_age : ℕ) 
  (h1 : mickey_age = 16) 
  (h2 : mickey_age = 2 * jerry_age - 6) : 
  jerry_age = 11 := by
  sorry

end NUMINAMATH_CALUDE_jerry_age_l3066_306663


namespace NUMINAMATH_CALUDE_haley_concert_spending_l3066_306622

def ticket_price : ℕ := 4
def tickets_for_self_and_friends : ℕ := 3
def extra_tickets : ℕ := 5

theorem haley_concert_spending :
  (tickets_for_self_and_friends + extra_tickets) * ticket_price = 32 := by
  sorry

end NUMINAMATH_CALUDE_haley_concert_spending_l3066_306622


namespace NUMINAMATH_CALUDE_packets_to_fill_gunny_bag_l3066_306689

/-- Represents the weight of a packet in ounces -/
def packet_weight : ℕ := 16 * 16 + 4

/-- Represents the capacity of the gunny bag in tons -/
def gunny_bag_capacity : ℕ := 13

/-- Conversion rate from tons to pounds -/
def tons_to_pounds : ℕ := 2500

/-- Conversion rate from pounds to ounces -/
def pounds_to_ounces : ℕ := 16

/-- Theorem stating the number of packets needed to fill the gunny bag -/
theorem packets_to_fill_gunny_bag : 
  (gunny_bag_capacity * tons_to_pounds * pounds_to_ounces) / packet_weight = 2000 := by
  sorry

#eval (gunny_bag_capacity * tons_to_pounds * pounds_to_ounces) / packet_weight

end NUMINAMATH_CALUDE_packets_to_fill_gunny_bag_l3066_306689


namespace NUMINAMATH_CALUDE_mary_zoom_time_l3066_306611

def total_time (mac_download : ℕ) (windows_download_factor : ℕ) 
               (audio_glitch_duration : ℕ) (audio_glitch_count : ℕ)
               (video_glitch_duration : ℕ) : ℕ :=
  let windows_download := mac_download * windows_download_factor
  let total_download := mac_download + windows_download
  let audio_glitch_time := audio_glitch_duration * audio_glitch_count
  let total_glitch_time := audio_glitch_time + video_glitch_duration
  let glitch_free_time := 2 * total_glitch_time
  total_download + total_glitch_time + glitch_free_time

theorem mary_zoom_time : 
  total_time 10 3 4 2 6 = 82 := by
  sorry

end NUMINAMATH_CALUDE_mary_zoom_time_l3066_306611


namespace NUMINAMATH_CALUDE_inverse_contrapositive_relation_l3066_306650

theorem inverse_contrapositive_relation (p q r : Prop) :
  (¬p ↔ q) →  -- inverse of p is q
  ((¬p ↔ r) ↔ p) →  -- contrapositive of p is r
  (q ↔ ¬r) :=  -- q and r are negations of each other
by sorry

end NUMINAMATH_CALUDE_inverse_contrapositive_relation_l3066_306650


namespace NUMINAMATH_CALUDE_main_theorem_l3066_306642

-- Define the type for multiplicative functions
def MultFun := ℕ → Fin 2

-- Define the property of being multiplicative
def is_multiplicative (f : MultFun) : Prop :=
  ∀ a b : ℕ, f (a * b) = f a * f b

theorem main_theorem (a b c d : ℕ) (f g : MultFun)
  (h1 : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0)
  (h2 : a * b * c * d ≠ 1)
  (h3 : Nat.gcd a b = 1 ∧ Nat.gcd a c = 1 ∧ Nat.gcd a d = 1 ∧
        Nat.gcd b c = 1 ∧ Nat.gcd b d = 1 ∧ Nat.gcd c d = 1)
  (h4 : is_multiplicative f ∧ is_multiplicative g)
  (h5 : ∀ n : ℕ, f (a * n + b) = g (c * n + d)) :
  (∀ n : ℕ, f (a * n + b) = 0 ∧ g (c * n + d) = 0) ∨
  (∃ k : ℕ, k > 0 ∧ ∀ n : ℕ, Nat.gcd n k = 1 → f n = 1 ∧ g n = 1) :=
by sorry

end NUMINAMATH_CALUDE_main_theorem_l3066_306642


namespace NUMINAMATH_CALUDE_ladies_walk_l3066_306681

/-- The combined distance walked by two ladies in Central Park -/
theorem ladies_walk (distance_lady2 : ℝ) (h1 : distance_lady2 = 4) :
  let distance_lady1 : ℝ := 2 * distance_lady2
  distance_lady1 + distance_lady2 = 12 := by
sorry

end NUMINAMATH_CALUDE_ladies_walk_l3066_306681


namespace NUMINAMATH_CALUDE_fraction_simplification_l3066_306684

theorem fraction_simplification (x : ℝ) (h : x = 7) : 
  (x^6 - 36*x^3 + 324) / (x^3 - 18) = 325 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3066_306684


namespace NUMINAMATH_CALUDE_vector_sum_magnitude_l3066_306619

def a : ℝ × ℝ := (2, 0)

theorem vector_sum_magnitude (b : ℝ × ℝ) 
  (h1 : Real.cos (Real.pi / 3) = (a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))
  (h2 : b.1^2 + b.2^2 = 1) : 
  Real.sqrt ((a.1 + 2*b.1)^2 + (a.2 + 2*b.2)^2) = 2 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_vector_sum_magnitude_l3066_306619


namespace NUMINAMATH_CALUDE_calculate_expression_l3066_306678

theorem calculate_expression : 20.17 * 69 + 201.7 * 1.3 - 8.2 * 1.7 = 1640 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l3066_306678


namespace NUMINAMATH_CALUDE_amount_with_r_l3066_306623

/-- Given three people (p, q, r) with a total amount of 6000 among them,
    where r has two-thirds of the total amount that p and q have together,
    prove that the amount with r is 2400. -/
theorem amount_with_r (total : ℕ) (amount_r : ℕ) : 
  total = 6000 →
  amount_r = (2 / 3 : ℚ) * (total - amount_r) →
  amount_r = 2400 := by
sorry

end NUMINAMATH_CALUDE_amount_with_r_l3066_306623


namespace NUMINAMATH_CALUDE_amanda_ticket_sales_l3066_306602

/-- The number of tickets Amanda needs to sell in total -/
def total_tickets : ℕ := 150

/-- The number of friends Amanda sells tickets to on the first day -/
def friends : ℕ := 8

/-- The number of tickets each friend buys on the first day -/
def tickets_per_friend : ℕ := 4

/-- The number of tickets Amanda sells on the second day -/
def second_day_tickets : ℕ := 45

/-- The number of tickets Amanda sells on the third day -/
def third_day_tickets : ℕ := 25

/-- The number of tickets Amanda needs to sell on the fourth and fifth day combined -/
def remaining_tickets : ℕ := total_tickets - (friends * tickets_per_friend + second_day_tickets + third_day_tickets)

theorem amanda_ticket_sales : remaining_tickets = 48 := by
  sorry

end NUMINAMATH_CALUDE_amanda_ticket_sales_l3066_306602


namespace NUMINAMATH_CALUDE_eight_stairs_climbs_l3066_306638

def climbStairs (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 1
  | 2 => 2
  | n + 3 => climbStairs (n + 2) + climbStairs (n + 1) + climbStairs n

theorem eight_stairs_climbs : climbStairs 8 = 81 := by
  sorry

end NUMINAMATH_CALUDE_eight_stairs_climbs_l3066_306638


namespace NUMINAMATH_CALUDE_intersection_points_inequality_l3066_306615

theorem intersection_points_inequality (a b x₁ x₂ : ℝ) 
  (h₁ : x₁ ≠ x₂) 
  (h₂ : Real.log x₁ / x₁ = a / 2 * x₁ + b) 
  (h₃ : Real.log x₂ / x₂ = a / 2 * x₂ + b) : 
  (x₁ + x₂) * (a / 2 * (x₁ + x₂) + b) > 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_points_inequality_l3066_306615


namespace NUMINAMATH_CALUDE_apple_bag_cost_proof_l3066_306690

/-- The cost of a bag of dozen apples -/
def apple_bag_cost : ℝ := 14

theorem apple_bag_cost_proof :
  let kiwi_cost : ℝ := 10
  let banana_cost : ℝ := 5
  let initial_money : ℝ := 50
  let subway_fare : ℝ := 3.5
  let max_apples : ℕ := 24
  apple_bag_cost = (initial_money - (kiwi_cost + banana_cost) - 2 * subway_fare) / (max_apples / 12) :=
by
  sorry

#check apple_bag_cost_proof

end NUMINAMATH_CALUDE_apple_bag_cost_proof_l3066_306690


namespace NUMINAMATH_CALUDE_smallest_factorizable_b_is_correct_l3066_306685

/-- The smallest positive integer b for which x^2 + bx + 1760 factors into (x + p)(x + q) with integer p and q -/
def smallest_factorizable_b : ℕ := 84

/-- A polynomial of the form x^2 + bx + 1760 -/
def polynomial (b : ℕ) (x : ℤ) : ℤ := x^2 + b * x + 1760

/-- Checks if a polynomial can be factored into (x + p)(x + q) with integer p and q -/
def is_factorizable (b : ℕ) : Prop :=
  ∃ (p q : ℤ), ∀ x, polynomial b x = (x + p) * (x + q)

theorem smallest_factorizable_b_is_correct :
  (is_factorizable smallest_factorizable_b) ∧
  (∀ b : ℕ, b < smallest_factorizable_b → ¬(is_factorizable b)) :=
sorry

end NUMINAMATH_CALUDE_smallest_factorizable_b_is_correct_l3066_306685


namespace NUMINAMATH_CALUDE_limit_to_infinity_l3066_306618

theorem limit_to_infinity (M : ℝ) (h : M > 0) :
  ∃ N : ℕ, ∀ n : ℕ, n > N → (2 * n^2 - 3 * n + 2) / (n + 2) > M := by
  sorry

end NUMINAMATH_CALUDE_limit_to_infinity_l3066_306618


namespace NUMINAMATH_CALUDE_triangle_kp_r3_bound_l3066_306626

/-- For any triangle with circumradius R, perimeter P, and area K, KP/R³ ≤ 27/4 -/
theorem triangle_kp_r3_bound (R P K : ℝ) (hR : R > 0) (hP : P > 0) (hK : K > 0) :
  K * P / R^3 ≤ 27 / 4 := by
  sorry


end NUMINAMATH_CALUDE_triangle_kp_r3_bound_l3066_306626


namespace NUMINAMATH_CALUDE_smallest_prime_2018_factorial_l3066_306616

def is_divisible (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

def factorial (n : ℕ) : ℕ := 
  if n = 0 then 1 else n * factorial (n - 1)

theorem smallest_prime_2018_factorial :
  ∃ p : ℕ, 
    Prime p ∧ 
    p = 509 ∧ 
    is_divisible (factorial 2018) (p^3) ∧ 
    ¬is_divisible (factorial 2018) (p^4) ∧
    ∀ q : ℕ, Prime q → q < p → 
      ¬(is_divisible (factorial 2018) (q^3) ∧ ¬is_divisible (factorial 2018) (q^4)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_2018_factorial_l3066_306616


namespace NUMINAMATH_CALUDE_no_right_triangle_with_sides_13_17_k_l3066_306621

theorem no_right_triangle_with_sides_13_17_k : 
  ¬ ∃ (k : ℕ), k > 0 ∧ 
  ((13 * 13 + 17 * 17 = k * k) ∨ 
   (13 * 13 + k * k = 17 * 17) ∨ 
   (17 * 17 + k * k = 13 * 13)) := by
sorry

end NUMINAMATH_CALUDE_no_right_triangle_with_sides_13_17_k_l3066_306621
