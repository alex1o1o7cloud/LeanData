import Mathlib

namespace NUMINAMATH_CALUDE_prob_A_and_B_l2641_264179

/-- The probability of event A occurring -/
def prob_A : ℝ := 0.55

/-- The probability of event B occurring -/
def prob_B : ℝ := 0.60

/-- The theorem stating that the probability of both A and B occurring simultaneously
    is equal to the product of their individual probabilities -/
theorem prob_A_and_B : prob_A * prob_B = 0.33 := by
  sorry

end NUMINAMATH_CALUDE_prob_A_and_B_l2641_264179


namespace NUMINAMATH_CALUDE_systematic_sampling_interval_l2641_264136

theorem systematic_sampling_interval 
  (total_numbers : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_numbers = 2014) 
  (h2 : sample_size = 100) :
  (total_numbers - total_numbers % sample_size) / sample_size = 20 :=
by sorry

end NUMINAMATH_CALUDE_systematic_sampling_interval_l2641_264136


namespace NUMINAMATH_CALUDE_find_divisor_l2641_264186

theorem find_divisor (x n : ℕ) (h1 : ∃ k : ℕ, x = k * n + 27)
                     (h2 : ∃ m : ℕ, x = 8 * m + 3)
                     (h3 : n > 27) :
  n = 32 := by
sorry

end NUMINAMATH_CALUDE_find_divisor_l2641_264186


namespace NUMINAMATH_CALUDE_decimal_sum_equals_fraction_l2641_264139

theorem decimal_sum_equals_fraction : 
  0.2 + 0.03 + 0.004 + 0.0005 + 0.00006 = 733 / 3125 := by
  sorry

end NUMINAMATH_CALUDE_decimal_sum_equals_fraction_l2641_264139


namespace NUMINAMATH_CALUDE_tank_full_after_45_minutes_l2641_264165

/-- Represents the state of a water tank system with three pipes. -/
structure TankSystem where
  capacity : ℕ
  pipeA_rate : ℕ
  pipeB_rate : ℕ
  pipeC_rate : ℕ

/-- Calculates the net water gain in one cycle. -/
def net_gain_per_cycle (system : TankSystem) : ℕ :=
  system.pipeA_rate + system.pipeB_rate - system.pipeC_rate

/-- Calculates the number of cycles needed to fill the tank. -/
def cycles_to_fill (system : TankSystem) : ℕ :=
  system.capacity / net_gain_per_cycle system

/-- Calculates the time in minutes to fill the tank. -/
def time_to_fill (system : TankSystem) : ℕ :=
  cycles_to_fill system * 3

/-- Theorem stating that the given tank system will be full after 45 minutes. -/
theorem tank_full_after_45_minutes (system : TankSystem)
  (h_capacity : system.capacity = 750)
  (h_pipeA : system.pipeA_rate = 40)
  (h_pipeB : system.pipeB_rate = 30)
  (h_pipeC : system.pipeC_rate = 20) :
  time_to_fill system = 45 := by
  sorry

end NUMINAMATH_CALUDE_tank_full_after_45_minutes_l2641_264165


namespace NUMINAMATH_CALUDE_prop_1_prop_3_l2641_264115

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)

-- Define non-coinciding lines and planes
variable (a b : Line)
variable (α β : Plane)
variable (h_lines_distinct : a ≠ b)
variable (h_planes_distinct : α ≠ β)

-- Proposition 1
theorem prop_1 : 
  parallel a b → perpendicular_line_plane a α → perpendicular_line_plane b α :=
sorry

-- Proposition 3
theorem prop_3 : 
  perpendicular_line_plane a α → perpendicular_line_plane a β → parallel_planes α β :=
sorry

end NUMINAMATH_CALUDE_prop_1_prop_3_l2641_264115


namespace NUMINAMATH_CALUDE_slope_of_l3_l2641_264177

/-- Line passing through two points -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Point in 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Triangle defined by three points -/
structure Triangle where
  a : Point
  b : Point
  c : Point

/-- Calculate the area of a triangle -/
def triangleArea (t : Triangle) : ℝ := sorry

/-- Check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop := sorry

/-- Find the intersection point of two lines -/
def lineIntersection (l1 l2 : Line) : Point := sorry

/-- Calculate the slope between two points -/
def slopeBetweenPoints (p1 p2 : Point) : ℝ := sorry

theorem slope_of_l3 (l1 l2 l3 : Line) (A B C : Point) :
  l1.slope = 4/3 ∧ l1.yIntercept = 2/3 ∧
  pointOnLine A l1 ∧ A.x = -2 ∧ A.y = -3 ∧
  l2.slope = 0 ∧ l2.yIntercept = 2 ∧
  B = lineIntersection l1 l2 ∧
  pointOnLine A l3 ∧ pointOnLine C l3 ∧
  pointOnLine C l2 ∧
  l3.slope > 0 ∧
  triangleArea ⟨A, B, C⟩ = 5 →
  l3.slope = 5/6 := by sorry

end NUMINAMATH_CALUDE_slope_of_l3_l2641_264177


namespace NUMINAMATH_CALUDE_hotel_moves_2_8_l2641_264158

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
| 0 => 1
| 1 => 1
| (n + 2) => fib (n + 1) + fib n

/-- Number of ways guests can move in a 2 × n grid hotel -/
def hotelMoves (n : ℕ) : ℕ := (fib (n + 1)) ^ 2

/-- Theorem: The number of ways guests can move in a 2 × 8 grid hotel is 3025 -/
theorem hotel_moves_2_8 : hotelMoves 8 = 3025 := by
  sorry

end NUMINAMATH_CALUDE_hotel_moves_2_8_l2641_264158


namespace NUMINAMATH_CALUDE_fraction_zero_at_zero_l2641_264185

theorem fraction_zero_at_zero (x : ℝ) : 
  (2 * x) / (x + 3) = 0 ↔ x = 0 :=
by sorry

end NUMINAMATH_CALUDE_fraction_zero_at_zero_l2641_264185


namespace NUMINAMATH_CALUDE_inverse_proportion_points_order_l2641_264171

theorem inverse_proportion_points_order (x₁ x₂ x₃ : ℝ) : 
  10 / x₁ = -5 → 10 / x₂ = 2 → 10 / x₃ = 5 → x₁ < x₃ ∧ x₃ < x₂ := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_points_order_l2641_264171


namespace NUMINAMATH_CALUDE_table_sum_theorem_l2641_264105

-- Define a 3x3 table as a function from (Fin 3 × Fin 3) to ℕ
def Table := Fin 3 → Fin 3 → ℕ

-- Define the property that the table contains numbers from 1 to 9
def containsOneToNine (t : Table) : Prop :=
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 9 → ∃ i j : Fin 3, t i j = n

-- Define the sum of a diagonal
def diagonalSum (t : Table) (d : Bool) : ℕ :=
  if d then t 0 0 + t 1 1 + t 2 2
  else t 0 2 + t 1 1 + t 2 0

-- Define the sum of five specific cells
def fiveCellSum (t : Table) : ℕ :=
  t 0 1 + t 1 0 + t 1 1 + t 1 2 + t 2 1

theorem table_sum_theorem (t : Table) 
  (h1 : containsOneToNine t)
  (h2 : diagonalSum t true = 7)
  (h3 : diagonalSum t false = 21) :
  fiveCellSum t = 25 := by
  sorry


end NUMINAMATH_CALUDE_table_sum_theorem_l2641_264105


namespace NUMINAMATH_CALUDE_lagrange_interpolation_polynomial_l2641_264118

def P (x : ℝ) : ℝ := x^3 - 2*x^2 + 3*x - 5

theorem lagrange_interpolation_polynomial :
  P (-1) = -11 ∧ P 1 = -3 ∧ P 2 = 1 ∧ P 3 = 13 :=
by sorry

end NUMINAMATH_CALUDE_lagrange_interpolation_polynomial_l2641_264118


namespace NUMINAMATH_CALUDE_expression_satisfies_equation_l2641_264194

theorem expression_satisfies_equation (x : ℝ) (E : ℝ → ℝ) : 
  x = 4 → (7 * E x = 21) → E = fun y ↦ y - 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_satisfies_equation_l2641_264194


namespace NUMINAMATH_CALUDE_product_equals_square_l2641_264170

theorem product_equals_square : 
  200 * 39.96 * 3.996 * 500 = (3996 : ℝ)^2 := by sorry

end NUMINAMATH_CALUDE_product_equals_square_l2641_264170


namespace NUMINAMATH_CALUDE_arithmetic_proof_l2641_264173

theorem arithmetic_proof : (3 + 2) - (2 + 1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_proof_l2641_264173


namespace NUMINAMATH_CALUDE_quadratic_equation_coefficients_l2641_264102

theorem quadratic_equation_coefficients :
  ∀ (a b c : ℝ), 
    (∀ x, 3 * x^2 = 2 * x - 3 ↔ a * x^2 + b * x + c = 0) →
    a = 3 ∧ b = -2 ∧ c = 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_coefficients_l2641_264102


namespace NUMINAMATH_CALUDE_triangle_rotation_reflection_l2641_264149

/-- Rotation of 90 degrees clockwise about the origin -/
def rotate90Clockwise (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, -p.1)

/-- Reflection over the y-axis -/
def reflectOverYAxis (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

/-- Given triangle ABC with vertices A(-3, 2), B(0, 5), and C(0, 2),
    prove that after rotating 90 degrees clockwise about the origin
    and then reflecting over the y-axis, point A ends up at (-2, 3) -/
theorem triangle_rotation_reflection :
  let A : ℝ × ℝ := (-3, 2)
  let B : ℝ × ℝ := (0, 5)
  let C : ℝ × ℝ := (0, 2)
  reflectOverYAxis (rotate90Clockwise A) = (-2, 3) := by
sorry


end NUMINAMATH_CALUDE_triangle_rotation_reflection_l2641_264149


namespace NUMINAMATH_CALUDE_floor_ratio_property_l2641_264168

theorem floor_ratio_property (x y : ℝ) (hx : x ≥ 1) (hy : y ≥ 1) 
  (h : ∀ n : ℕ, Int.floor (x / y) = Int.floor (n * x) / Int.floor (n * y)) :
  x = y ∨ (∃ (a b : ℕ), x = a ∧ y = b ∧ (a ∣ b ∨ b ∣ a)) :=
sorry

end NUMINAMATH_CALUDE_floor_ratio_property_l2641_264168


namespace NUMINAMATH_CALUDE_marbles_remaining_l2641_264106

def initial_marbles : ℕ := 47
def shared_marbles : ℕ := 42

theorem marbles_remaining : initial_marbles - shared_marbles = 5 := by
  sorry

end NUMINAMATH_CALUDE_marbles_remaining_l2641_264106


namespace NUMINAMATH_CALUDE_problem_solution_l2641_264195

theorem problem_solution (x y z : ℚ) : 
  x / y = 7 / 3 → y = 21 → z = 3 * y → x = 49 ∧ z = 63 := by
  sorry


end NUMINAMATH_CALUDE_problem_solution_l2641_264195


namespace NUMINAMATH_CALUDE_log_25_between_1_and_2_l2641_264152

theorem log_25_between_1_and_2 :
  ∃ (a b : ℤ), a + 1 = b ∧ (a : ℝ) < Real.log 25 / Real.log 10 ∧ Real.log 25 / Real.log 10 < b :=
sorry

end NUMINAMATH_CALUDE_log_25_between_1_and_2_l2641_264152


namespace NUMINAMATH_CALUDE_complex_magnitude_squared_l2641_264138

theorem complex_magnitude_squared (z : ℂ) (h : 2 * z + Complex.abs z = -3 + 12 * Complex.I) : Complex.normSq z = 61 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_squared_l2641_264138


namespace NUMINAMATH_CALUDE_employee_payment_l2641_264108

theorem employee_payment (total : ℝ) (a_multiplier : ℝ) (b_payment : ℝ) :
  total = 580 →
  a_multiplier = 1.5 →
  total = b_payment + a_multiplier * b_payment →
  b_payment = 232 := by
sorry

end NUMINAMATH_CALUDE_employee_payment_l2641_264108


namespace NUMINAMATH_CALUDE_volume_tetrahedron_C₁LMN_l2641_264135

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cuboid in 3D space -/
structure Cuboid where
  a : Point3D
  b : Point3D
  c : Point3D
  d : Point3D
  a₁ : Point3D
  b₁ : Point3D
  c₁ : Point3D
  d₁ : Point3D

/-- Represents a plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Calculates the volume of a tetrahedron given its four vertices -/
def tetrahedronVolume (p₁ p₂ p₃ p₄ : Point3D) : ℝ := sorry

/-- Finds the intersection of a line and a plane -/
def lineIntersectPlane (p₁ p₂ : Point3D) (plane : Plane) : Point3D := sorry

/-- Theorem: Volume of tetrahedron C₁LMN in the given cuboid -/
theorem volume_tetrahedron_C₁LMN (cuboid : Cuboid) 
  (h₁ : cuboid.a₁.z - cuboid.a.z = 2)
  (h₂ : cuboid.d.y - cuboid.a.y = 3)
  (h₃ : cuboid.b.x - cuboid.a.x = 251) :
  ∃ (volume : ℝ),
    let plane_A₁BD : Plane := sorry
    let L : Point3D := lineIntersectPlane cuboid.c cuboid.c₁ plane_A₁BD
    let M : Point3D := lineIntersectPlane cuboid.c₁ cuboid.b₁ plane_A₁BD
    let N : Point3D := lineIntersectPlane cuboid.c₁ cuboid.d₁ plane_A₁BD
    volume = tetrahedronVolume cuboid.c₁ L M N := by sorry

end NUMINAMATH_CALUDE_volume_tetrahedron_C₁LMN_l2641_264135


namespace NUMINAMATH_CALUDE_existence_of_prime_divisor_greater_than_ten_l2641_264148

/-- A function that returns the smallest prime divisor of a natural number -/
def smallest_prime_divisor (n : ℕ) : ℕ := sorry

/-- A predicate that checks if a natural number is a four-digit number -/
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

theorem existence_of_prime_divisor_greater_than_ten (start : ℕ) 
  (h_start : is_four_digit start) :
  ∃ k : ℕ, k < 10 ∧ smallest_prime_divisor (start + k) > 10 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_prime_divisor_greater_than_ten_l2641_264148


namespace NUMINAMATH_CALUDE_transfer_ratio_l2641_264182

def initial_balance : ℕ := 190
def mom_transfer : ℕ := 60
def final_balance : ℕ := 100

def sister_transfer : ℕ := initial_balance - mom_transfer - final_balance

theorem transfer_ratio : 
  sister_transfer * 2 = mom_transfer := by sorry

end NUMINAMATH_CALUDE_transfer_ratio_l2641_264182


namespace NUMINAMATH_CALUDE_line_perp_plane_properties_l2641_264112

-- Define a structure for a 3D space
structure Space3D where
  Point : Type
  Line : Type
  Plane : Type
  perpendicular : Line → Plane → Prop
  line_in_plane : Line → Plane → Prop
  line_perp_line : Line → Line → Prop

-- Define the theorem
theorem line_perp_plane_properties {S : Space3D} (a : S.Line) (M : S.Plane) :
  (S.perpendicular a M → ∀ (l : S.Line), S.line_in_plane l M → S.line_perp_line a l) ∧
  (∃ (b : S.Line) (N : S.Plane), (∀ (l : S.Line), S.line_in_plane l N → S.line_perp_line b l) ∧ ¬S.perpendicular b N) :=
sorry

end NUMINAMATH_CALUDE_line_perp_plane_properties_l2641_264112


namespace NUMINAMATH_CALUDE_box_volume_formula_l2641_264162

/-- The volume of a box formed by cutting squares from corners of a sheet -/
def boxVolume (x : ℝ) : ℝ := (16 - 2*x) * (12 - 2*x) * x

/-- The constraint on the side length of the cut squares -/
def sideConstraint (x : ℝ) : Prop := x ≤ 12/5

theorem box_volume_formula (x : ℝ) (h : sideConstraint x) :
  boxVolume x = 192*x - 56*x^2 + 4*x^3 := by
  sorry

end NUMINAMATH_CALUDE_box_volume_formula_l2641_264162


namespace NUMINAMATH_CALUDE_difference_of_squares_form_l2641_264180

theorem difference_of_squares_form (x y : ℝ) :
  ∃ a b : ℝ, (-x + y) * (x + y) = -(a + b) * (a - b) := by sorry

end NUMINAMATH_CALUDE_difference_of_squares_form_l2641_264180


namespace NUMINAMATH_CALUDE_equation_solution_l2641_264141

theorem equation_solution (x : ℝ) : 3 - 1 / (2 - x) = 2 * (1 / (2 - x)) → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2641_264141


namespace NUMINAMATH_CALUDE_distance_from_circle_center_to_line_l2641_264147

/-- The distance from the center of the circle x^2 + y^2 - 2x = 0 to the line 2x + y - 1 = 0 is √5/5 -/
theorem distance_from_circle_center_to_line :
  let circle_eq : ℝ → ℝ → Prop := λ x y ↦ x^2 + y^2 - 2*x = 0
  let line_eq : ℝ → ℝ → Prop := λ x y ↦ 2*x + y - 1 = 0
  ∃ (center_x center_y : ℝ), 
    (∀ x y, circle_eq x y ↔ (x - center_x)^2 + (y - center_y)^2 = 1) ∧
    (abs (2*center_x + center_y - 1) / Real.sqrt 5 = 1/5) :=
by sorry

end NUMINAMATH_CALUDE_distance_from_circle_center_to_line_l2641_264147


namespace NUMINAMATH_CALUDE_max_d_plus_r_l2641_264128

theorem max_d_plus_r : ∃ (d r : ℕ), 
  d > 0 ∧
  468 % d = r ∧
  636 % d = r ∧
  867 % d = r ∧
  ∀ (d' r' : ℕ), (d' > 0 ∧ 
                  468 % d' = r' ∧ 
                  636 % d' = r' ∧ 
                  867 % d' = r') → 
                  d + r ≥ d' + r' ∧
  d + r = 27 :=
by sorry

end NUMINAMATH_CALUDE_max_d_plus_r_l2641_264128


namespace NUMINAMATH_CALUDE_g_of_5_l2641_264155

def g (x : ℝ) : ℝ := x^2 - 2*x

theorem g_of_5 : g 5 = 15 := by sorry

end NUMINAMATH_CALUDE_g_of_5_l2641_264155


namespace NUMINAMATH_CALUDE_remainder_3249_div_82_l2641_264119

theorem remainder_3249_div_82 : 3249 % 82 = 51 := by sorry

end NUMINAMATH_CALUDE_remainder_3249_div_82_l2641_264119


namespace NUMINAMATH_CALUDE_system_solution_l2641_264117

theorem system_solution (x y : ℝ) : 
  x^2 + y^2 ≤ 2 ∧ 
  81 * x^4 - 18 * x^2 * y^2 + y^4 - 360 * x^2 - 40 * y^2 + 400 = 0 ↔ 
  ((x = -3 / Real.sqrt 5 ∧ y = 1 / Real.sqrt 5) ∨
   (x = -3 / Real.sqrt 5 ∧ y = -1 / Real.sqrt 5) ∨
   (x = 3 / Real.sqrt 5 ∧ y = -1 / Real.sqrt 5) ∨
   (x = 3 / Real.sqrt 5 ∧ y = 1 / Real.sqrt 5)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l2641_264117


namespace NUMINAMATH_CALUDE_right_triangle_pythagorean_l2641_264124

theorem right_triangle_pythagorean (a b c : ℝ) : 
  a = 1 ∧ b = Real.sqrt 3 ∧ c = 2 → a^2 + b^2 = c^2 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_pythagorean_l2641_264124


namespace NUMINAMATH_CALUDE_initial_bacteria_population_l2641_264175

/-- The number of seconds in 5 minutes -/
def totalTime : ℕ := 300

/-- The doubling time of the bacteria population in seconds -/
def doublingTime : ℕ := 30

/-- The number of bacteria after 5 minutes -/
def finalPopulation : ℕ := 1310720

/-- The number of doublings that occur in 5 minutes -/
def numberOfDoublings : ℕ := totalTime / doublingTime

theorem initial_bacteria_population :
  ∃ (initialPopulation : ℕ),
    initialPopulation * (2 ^ numberOfDoublings) = finalPopulation ∧
    initialPopulation = 1280 :=
by sorry

end NUMINAMATH_CALUDE_initial_bacteria_population_l2641_264175


namespace NUMINAMATH_CALUDE_sqrt_of_sqrt_four_equals_sqrt_two_l2641_264153

theorem sqrt_of_sqrt_four_equals_sqrt_two : Real.sqrt (Real.sqrt 4) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_of_sqrt_four_equals_sqrt_two_l2641_264153


namespace NUMINAMATH_CALUDE_complex_modulus_squared_l2641_264166

theorem complex_modulus_squared (z : ℂ) (h : z^2 + Complex.abs z^2 = 6 - 9*I) : 
  Complex.abs z^2 = 39/4 := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_squared_l2641_264166


namespace NUMINAMATH_CALUDE_square_cutting_solution_l2641_264156

/-- Represents the cutting of an 8x8 square into smaller pieces -/
structure SquareCutting where
  /-- The number of 2x2 squares -/
  num_squares : ℕ
  /-- The number of 1x4 rectangles -/
  num_rectangles : ℕ
  /-- The total length of cuts -/
  total_cut_length : ℕ

/-- Theorem stating the solution to the square cutting problem -/
theorem square_cutting_solution :
  ∃ (cut : SquareCutting),
    cut.num_squares = 10 ∧
    cut.num_rectangles = 6 ∧
    cut.total_cut_length = 54 ∧
    cut.num_squares + cut.num_rectangles = 64 / 4 ∧
    8 * cut.num_squares + 10 * cut.num_rectangles = 32 + 2 * cut.total_cut_length :=
by sorry

end NUMINAMATH_CALUDE_square_cutting_solution_l2641_264156


namespace NUMINAMATH_CALUDE_min_value_sum_l2641_264181

theorem min_value_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2*x + 8*y - x*y = 0) :
  x + y ≥ 18 := by
sorry

end NUMINAMATH_CALUDE_min_value_sum_l2641_264181


namespace NUMINAMATH_CALUDE_log_f_geq_one_f_geq_a_iff_a_leq_one_l2641_264133

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 2| + |x - a|

-- Part 1: Prove that log f(x) ≥ 1 when a = -8
theorem log_f_geq_one (x : ℝ) : Real.log (f (-8) x) ≥ 1 := by
  sorry

-- Part 2: Prove that f(x) ≥ a for all x ∈ ℝ if and only if a ≤ 1
theorem f_geq_a_iff_a_leq_one :
  (∀ x : ℝ, f a x ≥ a) ↔ a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_log_f_geq_one_f_geq_a_iff_a_leq_one_l2641_264133


namespace NUMINAMATH_CALUDE_zero_integer_not_positive_negative_l2641_264137

theorem zero_integer_not_positive_negative :
  (0 : ℤ) ∈ Set.univ ∧ (0 : ℤ) ∉ {x : ℤ | x > 0} ∧ (0 : ℤ) ∉ {x : ℤ | x < 0} := by
  sorry

end NUMINAMATH_CALUDE_zero_integer_not_positive_negative_l2641_264137


namespace NUMINAMATH_CALUDE_partner_a_share_l2641_264167

/-- Calculates a partner's share of the annual gain in a partnership --/
def calculate_share (x : ℚ) (annual_gain : ℚ) : ℚ :=
  let a_share := 12 * x
  let b_share := 12 * x
  let c_share := 12 * x
  let d_share := 36 * x
  let e_share := 35 * x
  let f_share := 30 * x
  let total_investment := a_share + b_share + c_share + d_share + e_share + f_share
  (a_share / total_investment) * annual_gain

/-- The problem statement --/
theorem partner_a_share :
  ∃ (x : ℚ), calculate_share x 38400 = 3360 := by
  sorry

end NUMINAMATH_CALUDE_partner_a_share_l2641_264167


namespace NUMINAMATH_CALUDE_golden_ratio_less_than_one_l2641_264189

theorem golden_ratio_less_than_one : (Real.sqrt 5 - 1) / 2 < 1 := by
  sorry

end NUMINAMATH_CALUDE_golden_ratio_less_than_one_l2641_264189


namespace NUMINAMATH_CALUDE_exam_score_theorem_l2641_264110

/-- Given an exam with 50 questions, where each correct answer scores 4 marks
    and each wrong answer loses 1 mark, if the total score is 130 marks,
    then the number of correctly answered questions is 36. -/
theorem exam_score_theorem (total_questions : ℕ) (correct_score : ℤ) (wrong_score : ℤ)
    (total_score : ℤ) (correct_answers : ℕ) :
  total_questions = 50 →
  correct_score = 4 →
  wrong_score = -1 →
  total_score = 130 →
  (correct_answers : ℤ) * correct_score +
    (total_questions - correct_answers : ℤ) * wrong_score = total_score →
  correct_answers = 36 := by
  sorry

end NUMINAMATH_CALUDE_exam_score_theorem_l2641_264110


namespace NUMINAMATH_CALUDE_fanfan_distance_is_120_l2641_264187

/-- Represents the cost and distance information for a shared car journey -/
structure JourneyInfo where
  ningning_cost : ℝ
  leilei_cost : ℝ
  fanfan_cost : ℝ
  ningning_distance : ℝ

/-- Calculates the distance to Fanfan's home given the journey information -/
def calculate_fanfan_distance (info : JourneyInfo) : ℝ :=
  sorry

/-- Theorem stating that given the journey information, Fanfan's home is 120 km from school -/
theorem fanfan_distance_is_120 (info : JourneyInfo) 
  (h1 : info.ningning_cost = 10)
  (h2 : info.leilei_cost = 25)
  (h3 : info.fanfan_cost = 85)
  (h4 : info.ningning_distance = 12) :
  calculate_fanfan_distance info = 120 := by
  sorry

end NUMINAMATH_CALUDE_fanfan_distance_is_120_l2641_264187


namespace NUMINAMATH_CALUDE_student_selection_l2641_264126

theorem student_selection (n : ℕ) (h : n = 30) : 
  (Nat.choose n 2 = 435) ∧ (Nat.choose n 3 = 4060) := by
  sorry

#check student_selection

end NUMINAMATH_CALUDE_student_selection_l2641_264126


namespace NUMINAMATH_CALUDE_existence_of_common_root_l2641_264163

-- Define the structure of a quadratic polynomial
structure QuadraticPolynomial (R : Type*) [Ring R] where
  a : R
  b : R
  c : R

-- Define the evaluation of a quadratic polynomial
def evaluate {R : Type*} [Ring R] (p : QuadraticPolynomial R) (x : R) : R :=
  p.a * x * x + p.b * x + p.c

-- Theorem statement
theorem existence_of_common_root 
  {R : Type*} [Field R] 
  (f g h : QuadraticPolynomial R)
  (no_roots : ∀ x, evaluate f x ≠ 0 ∧ evaluate g x ≠ 0 ∧ evaluate h x ≠ 0)
  (same_leading_coeff : f.a = g.a ∧ f.a = h.a)
  (diff_x_coeff : f.b ≠ g.b ∧ f.b ≠ h.b ∧ g.b ≠ h.b) :
  ∃ c x, evaluate f x + c * evaluate g x = 0 ∧ evaluate f x + c * evaluate h x = 0 :=
sorry

end NUMINAMATH_CALUDE_existence_of_common_root_l2641_264163


namespace NUMINAMATH_CALUDE_rectangle_area_l2641_264100

/-- Calculates the area of a rectangle given its perimeter and width. -/
theorem rectangle_area (perimeter : ℝ) (width : ℝ) (h1 : perimeter = 42) (h2 : width = 8) :
  width * (perimeter / 2 - width) = 104 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l2641_264100


namespace NUMINAMATH_CALUDE_geometric_seq_property_P_iff_q_range_l2641_264144

/-- Property P for a finite sequence -/
def has_property_P (a : ℕ → ℝ) (n : ℕ) : Prop :=
  ∀ i j, 1 < i ∧ i < j ∧ j ≤ n → |a 1 - a i| ≤ |a 1 - a j|

/-- Geometric sequence with first term 1 and common ratio q -/
def geometric_seq (q : ℝ) (n : ℕ) : ℝ := q^(n-1)

theorem geometric_seq_property_P_iff_q_range :
  ∀ q : ℝ, has_property_P (geometric_seq q) 10 ↔ q ∈ Set.Iic (-2) ∪ Set.Ioi 0 := by
  sorry

end NUMINAMATH_CALUDE_geometric_seq_property_P_iff_q_range_l2641_264144


namespace NUMINAMATH_CALUDE_negation_equivalence_l2641_264176

theorem negation_equivalence :
  (¬ ∀ x : ℝ, x > 0 → x / (x - 1) > 0) ↔ (∃ x : ℝ, x > 0 ∧ 0 ≤ x ∧ x < 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2641_264176


namespace NUMINAMATH_CALUDE_q_value_l2641_264178

theorem q_value (p q : ℝ) (h1 : 1 < p) (h2 : p < q) (h3 : 1/p + 1/q = 1) (h4 : p * q = 12) : q = 6 + 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_q_value_l2641_264178


namespace NUMINAMATH_CALUDE_star_equation_solution_l2641_264121

/-- Custom binary operation ⭐ -/
def star (a b : ℝ) : ℝ := a * b + 3 * b - a

/-- Theorem stating that if 5 ⭐ x = 40, then x = 45/8 -/
theorem star_equation_solution :
  star 5 x = 40 → x = 45 / 8 := by
  sorry

end NUMINAMATH_CALUDE_star_equation_solution_l2641_264121


namespace NUMINAMATH_CALUDE_dan_gave_fourteen_marbles_l2641_264114

/-- The number of marbles Dan gave to Mary -/
def marbles_given (initial : ℕ) (remaining : ℕ) : ℕ := initial - remaining

/-- Proof that Dan gave 14 marbles to Mary -/
theorem dan_gave_fourteen_marbles :
  let initial := 64
  let remaining := 50
  marbles_given initial remaining = 14 := by
sorry

end NUMINAMATH_CALUDE_dan_gave_fourteen_marbles_l2641_264114


namespace NUMINAMATH_CALUDE_largest_quantity_l2641_264140

theorem largest_quantity (A B C : ℚ) : 
  A = 2006/2005 + 2006/2007 →
  B = 2006/2007 + 2008/2007 →
  C = 2007/2006 + 2007/2008 →
  A > B ∧ A > C := by
  sorry

end NUMINAMATH_CALUDE_largest_quantity_l2641_264140


namespace NUMINAMATH_CALUDE_line_equation_proof_l2641_264198

/-- Given a line in the form ax + by + c = 0, prove it has slope -3 and x-intercept 2 -/
theorem line_equation_proof (a b c : ℝ) (h1 : a = 3) (h2 : b = 1) (h3 : c = -6) : 
  (∀ x y : ℝ, a*x + b*y + c = 0 ↔ y = -3*(x - 2)) := by sorry

end NUMINAMATH_CALUDE_line_equation_proof_l2641_264198


namespace NUMINAMATH_CALUDE_dandelion_survival_l2641_264157

/-- The number of seeds produced by each dandelion -/
def seeds_per_dandelion : ℕ := 300

/-- The fraction of seeds that land in water and die -/
def water_death_fraction : ℚ := 1/3

/-- The fraction of starting seeds eaten by insects -/
def insect_eaten_fraction : ℚ := 1/6

/-- The fraction of remaining seeds that sprout and are immediately eaten -/
def sprout_eaten_fraction : ℚ := 1/2

/-- The number of dandelions that survive long enough to flower -/
def surviving_dandelions : ℕ := 75

theorem dandelion_survival :
  (seeds_per_dandelion : ℚ) * (1 - water_death_fraction) * (1 - insect_eaten_fraction) * (1 - sprout_eaten_fraction) = surviving_dandelions := by
  sorry

end NUMINAMATH_CALUDE_dandelion_survival_l2641_264157


namespace NUMINAMATH_CALUDE_binomial_coefficient_divisibility_l2641_264192

def infinitely_many_n (k : ℤ) : Prop :=
  ∀ m : ℕ, ∃ n : ℕ, n > m ∧ ¬((n : ℤ) + k ∣ Nat.choose (2*n) n)

theorem binomial_coefficient_divisibility :
  infinitely_many_n (-1) :=
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_divisibility_l2641_264192


namespace NUMINAMATH_CALUDE_equation_solution_l2641_264183

theorem equation_solution (x y : ℝ) :
  (Real.sqrt (8 * x) / Real.sqrt (4 * (y - 2)) = 3) →
  (x = (9 * y - 18) / 2) := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l2641_264183


namespace NUMINAMATH_CALUDE_total_distance_traveled_l2641_264161

/-- Given the conditions of the problem, prove that the total distance traveled is 5 miles. -/
theorem total_distance_traveled (total_time : ℝ) (walking_time : ℝ) (walking_rate : ℝ) 
  (break_time : ℝ) (running_time : ℝ) (running_rate : ℝ) :
  total_time = 75 / 60 → 
  walking_time = 1 →
  walking_rate = 3 →
  break_time = 5 / 60 →
  running_time = 1 / 6 →
  running_rate = 12 →
  walking_time * walking_rate + running_time * running_rate = 5 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_traveled_l2641_264161


namespace NUMINAMATH_CALUDE_hare_jumps_to_12th_cell_l2641_264107

/-- The number of ways a hare can reach the nth cell -/
def hare_jumps : ℕ → ℕ
| 0 => 0  -- No ways to reach the 0th cell (not part of the strip)
| 1 => 1  -- One way to be at the 1st cell (starting position)
| 2 => 1  -- One way to reach the 2nd cell (single jump from 1st)
| (n + 3) => hare_jumps (n + 2) + hare_jumps (n + 1)

/-- The number of ways a hare can jump from the 1st cell to the 12th cell is 144 -/
theorem hare_jumps_to_12th_cell : hare_jumps 12 = 144 := by
  sorry

end NUMINAMATH_CALUDE_hare_jumps_to_12th_cell_l2641_264107


namespace NUMINAMATH_CALUDE_probability_three_red_cards_l2641_264123

/-- The probability of drawing three red cards in succession from a shuffled standard deck --/
theorem probability_three_red_cards (total_cards : ℕ) (red_cards : ℕ) 
  (h1 : total_cards = 52)
  (h2 : red_cards = 26) : 
  (red_cards * (red_cards - 1) * (red_cards - 2)) / 
  (total_cards * (total_cards - 1) * (total_cards - 2)) = 4 / 17 := by
sorry

end NUMINAMATH_CALUDE_probability_three_red_cards_l2641_264123


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_first_ten_terms_sum_l2641_264197

theorem arithmetic_sequence_sum : ℤ → ℤ → ℕ → ℤ
  | a, l, n => n * (a + l) / 2

theorem first_ten_terms_sum (a l : ℤ) (n : ℕ) (h1 : a = -5) (h2 : l = 40) (h3 : n = 10) :
  arithmetic_sequence_sum a l n = 175 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_first_ten_terms_sum_l2641_264197


namespace NUMINAMATH_CALUDE_inverse_proportion_ordering_l2641_264145

theorem inverse_proportion_ordering (y₁ y₂ y₃ : ℝ) :
  y₁ = 7 / (-3) ∧ y₂ = 7 / (-1) ∧ y₃ = 7 / 2 →
  y₂ < y₁ ∧ y₁ < y₃ := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_ordering_l2641_264145


namespace NUMINAMATH_CALUDE_largest_divisor_of_polynomial_l2641_264104

theorem largest_divisor_of_polynomial (n : ℤ) : 
  ∃ (k : ℕ), k = 120 ∧ (k : ℤ) ∣ (n^5 - 5*n^3 + 4*n) ∧ 
  ∀ (m : ℕ), m > k → ¬((m : ℤ) ∣ (n^5 - 5*n^3 + 4*n)) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_polynomial_l2641_264104


namespace NUMINAMATH_CALUDE_trout_catfish_ratio_is_three_to_one_l2641_264160

/-- Represents the fishing challenge scenario -/
structure FishingChallenge where
  will_catfish : ℕ
  will_eels : ℕ
  total_fish : ℕ

/-- Calculates the ratio of trout to catfish Henry challenged himself to catch -/
def trout_catfish_ratio (challenge : FishingChallenge) : ℚ :=
  let will_total := challenge.will_catfish + challenge.will_eels
  let henry_fish := challenge.total_fish - will_total
  (henry_fish : ℚ) / (challenge.will_catfish : ℚ) / 2

/-- Theorem stating the ratio of trout to catfish Henry challenged himself to catch -/
theorem trout_catfish_ratio_is_three_to_one (challenge : FishingChallenge)
  (h1 : challenge.will_catfish = 16)
  (h2 : challenge.will_eels = 10)
  (h3 : challenge.total_fish = 50) :
  trout_catfish_ratio challenge = 3 := by
  sorry

end NUMINAMATH_CALUDE_trout_catfish_ratio_is_three_to_one_l2641_264160


namespace NUMINAMATH_CALUDE_hexagon_wire_remainder_l2641_264154

/-- Calculates the remaining wire length after creating a regular hexagon -/
def remaining_wire_length (total_wire : ℝ) (hexagon_side : ℝ) : ℝ :=
  total_wire - 6 * hexagon_side

/-- Theorem: Given a 50 cm wire and a regular hexagon with 8 cm sides, 2 cm of wire remains -/
theorem hexagon_wire_remainder :
  remaining_wire_length 50 8 = 2 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_wire_remainder_l2641_264154


namespace NUMINAMATH_CALUDE_abs_sum_inequality_l2641_264146

theorem abs_sum_inequality (x : ℝ) : 
  |x - 2| + |x + 3| < 8 ↔ -13/2 < x ∧ x < 7/2 := by sorry

end NUMINAMATH_CALUDE_abs_sum_inequality_l2641_264146


namespace NUMINAMATH_CALUDE_equation_solution_l2641_264159

theorem equation_solution :
  let f : ℝ → ℝ := λ x => x^2 - |x| - 1
  ∀ x : ℝ, f x = 0 ↔ x = (-1 + Real.sqrt 5) / 2 ∨ x = (1 - Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2641_264159


namespace NUMINAMATH_CALUDE_circumcircle_equation_l2641_264151

/-- Given a triangle ABC with vertices A(0,4), B(0,0), and C(3,0),
    prove that (x-3/2)^2 + (y-2)^2 = 25/4 is the equation of its circumcircle. -/
theorem circumcircle_equation (x y : ℝ) : 
  let A : ℝ × ℝ := (0, 4)
  let B : ℝ × ℝ := (0, 0)
  let C : ℝ × ℝ := (3, 0)
  (x - 3/2)^2 + (y - 2)^2 = 25/4 ↔ 
    ∃ (center : ℝ × ℝ) (radius : ℝ), 
      (center.1 - A.1)^2 + (center.2 - A.2)^2 = radius^2 ∧
      (center.1 - B.1)^2 + (center.2 - B.2)^2 = radius^2 ∧
      (center.1 - C.1)^2 + (center.2 - C.2)^2 = radius^2 :=
by sorry


end NUMINAMATH_CALUDE_circumcircle_equation_l2641_264151


namespace NUMINAMATH_CALUDE_total_stickers_is_110_l2641_264127

/-- Represents a folder with its color, number of sheets, and stickers per sheet -/
structure Folder :=
  (color : String)
  (sheets : Nat)
  (stickers_per_sheet : Nat)

/-- Calculates the total number of stickers in a folder -/
def stickers_in_folder (f : Folder) : Nat :=
  f.sheets * f.stickers_per_sheet

/-- Peggy's folders -/
def peggy_folders : List Folder := [
  ⟨"red", 10, 3⟩,
  ⟨"green", 8, 5⟩,
  ⟨"blue", 6, 2⟩,
  ⟨"yellow", 4, 4⟩,
  ⟨"purple", 2, 6⟩
]

/-- Theorem: The total number of stickers Peggy uses is 110 -/
theorem total_stickers_is_110 : 
  (peggy_folders.map stickers_in_folder).sum = 110 := by
  sorry


end NUMINAMATH_CALUDE_total_stickers_is_110_l2641_264127


namespace NUMINAMATH_CALUDE_last_four_matches_average_l2641_264116

/-- Represents a cricket scoring scenario -/
structure CricketScoring where
  totalMatches : Nat
  firstMatchesCount : Nat
  totalAverage : ℚ
  firstMatchesAverage : ℚ

/-- Calculates the average score of the remaining matches -/
def remainingMatchesAverage (cs : CricketScoring) : ℚ :=
  let totalRuns := cs.totalAverage * cs.totalMatches
  let firstMatchesRuns := cs.firstMatchesAverage * cs.firstMatchesCount
  let remainingMatchesCount := cs.totalMatches - cs.firstMatchesCount
  (totalRuns - firstMatchesRuns) / remainingMatchesCount

/-- Theorem stating that under the given conditions, the average of the last 4 matches is 34.25 -/
theorem last_four_matches_average (cs : CricketScoring) 
  (h1 : cs.totalMatches = 10)
  (h2 : cs.firstMatchesCount = 6)
  (h3 : cs.totalAverage = 389/10)
  (h4 : cs.firstMatchesAverage = 42) :
  remainingMatchesAverage cs = 137/4 := by
  sorry

end NUMINAMATH_CALUDE_last_four_matches_average_l2641_264116


namespace NUMINAMATH_CALUDE_least_clock_equivalent_l2641_264169

def clock_equivalent (n : ℕ) : Prop :=
  24 ∣ (n^2 - n)

theorem least_clock_equivalent : 
  ∀ k : ℕ, k > 5 → clock_equivalent k → k ≥ 9 :=
by sorry

end NUMINAMATH_CALUDE_least_clock_equivalent_l2641_264169


namespace NUMINAMATH_CALUDE_mityas_age_l2641_264193

/-- Represents the ages of Mitya and Shura -/
structure Ages where
  mitya : ℕ
  shura : ℕ

/-- The conditions of the problem -/
def problem_conditions (ages : Ages) : Prop :=
  (ages.mitya = ages.shura + 11) ∧
  (ages.shura = 2 * (ages.shura - (ages.mitya - ages.shura)))

/-- The theorem stating Mitya's age -/
theorem mityas_age :
  ∃ (ages : Ages), problem_conditions ages ∧ ages.mitya = 33 :=
sorry

end NUMINAMATH_CALUDE_mityas_age_l2641_264193


namespace NUMINAMATH_CALUDE_negative_abs_negative_three_l2641_264188

theorem negative_abs_negative_three : -|-3| = -3 := by sorry

end NUMINAMATH_CALUDE_negative_abs_negative_three_l2641_264188


namespace NUMINAMATH_CALUDE_equivalent_operations_l2641_264196

theorem equivalent_operations (x : ℝ) : x * (4/5) / (2/7) = x * (7/5) := by
  sorry

end NUMINAMATH_CALUDE_equivalent_operations_l2641_264196


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2641_264134

/-- Given a geometric sequence {a_n} with a₁ = 2 and a₁ + a₃ + a₅ = 14,
    prove that 1/a₁ + 1/a₃ + 1/a₅ = 7/8 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (h_geom : ∀ n, a (n + 1) / a n = a 2 / a 1) 
    (h_a1 : a 1 = 2) (h_sum : a 1 + a 3 + a 5 = 14) :
  1 / a 1 + 1 / a 3 + 1 / a 5 = 7 / 8 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2641_264134


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l2641_264111

theorem arithmetic_calculation : (2004 - (2011 - 196)) + (2011 - (196 - 2004)) = 4008 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l2641_264111


namespace NUMINAMATH_CALUDE_solution_satisfies_system_and_initial_conditions_l2641_264113

noncomputable def y₁ (x : ℝ) : ℝ := Real.exp (2 * x)
noncomputable def y₂ (x : ℝ) : ℝ := Real.exp (-x) + Real.exp (2 * x)
noncomputable def y₃ (x : ℝ) : ℝ := -Real.exp (-x) + Real.exp (2 * x)

theorem solution_satisfies_system_and_initial_conditions :
  (∀ x, (deriv y₁) x = y₂ x + y₃ x) ∧
  (∀ x, (deriv y₂) x = y₁ x + y₃ x) ∧
  (∀ x, (deriv y₃) x = y₁ x + y₂ x) ∧
  y₁ 0 = 1 ∧
  y₂ 0 = 2 ∧
  y₃ 0 = 0 := by
  sorry

end NUMINAMATH_CALUDE_solution_satisfies_system_and_initial_conditions_l2641_264113


namespace NUMINAMATH_CALUDE_x_squared_mod_25_l2641_264164

theorem x_squared_mod_25 (x : ℤ) 
  (h1 : 5 * x ≡ 15 [ZMOD 25])
  (h2 : 2 * x ≡ 10 [ZMOD 25]) : 
  x^2 ≡ 0 [ZMOD 25] := by
sorry

end NUMINAMATH_CALUDE_x_squared_mod_25_l2641_264164


namespace NUMINAMATH_CALUDE_line_not_in_second_quadrant_l2641_264184

-- Define the line l with equation x - y - a² = 0
def line_equation (x y a : ℝ) : Prop := x - y - a^2 = 0

-- Define the second quadrant
def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

-- Theorem statement
theorem line_not_in_second_quadrant (a : ℝ) (h : a ≠ 0) :
  ∀ x y : ℝ, line_equation x y a → ¬ second_quadrant x y :=
sorry

end NUMINAMATH_CALUDE_line_not_in_second_quadrant_l2641_264184


namespace NUMINAMATH_CALUDE_function_value_at_2012_l2641_264122

theorem function_value_at_2012 (f : ℝ → ℝ) 
  (h1 : f 0 = 2012)
  (h2 : ∀ x : ℝ, f (x + 2) - f x ≤ 3 * 2^x)
  (h3 : ∀ x : ℝ, f (x + 6) - f x ≥ 63 * 2^x) :
  f 2012 = 2^2012 + 2011 := by
sorry

end NUMINAMATH_CALUDE_function_value_at_2012_l2641_264122


namespace NUMINAMATH_CALUDE_circle_radius_from_longest_chord_l2641_264125

theorem circle_radius_from_longest_chord (c : ℝ) (h : c > 0) :
  (∃ r : ℝ, r > 0 ∧ c = 2 * r) → c / 2 = 5 :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_from_longest_chord_l2641_264125


namespace NUMINAMATH_CALUDE_basil_leaves_count_l2641_264129

theorem basil_leaves_count (basil_pots rosemary_pots thyme_pots : ℕ)
  (rosemary_leaves_per_pot thyme_leaves_per_pot : ℕ)
  (total_leaves : ℕ) :
  basil_pots = 3 →
  rosemary_pots = 9 →
  thyme_pots = 6 →
  rosemary_leaves_per_pot = 18 →
  thyme_leaves_per_pot = 30 →
  total_leaves = 354 →
  ∃ (basil_leaves_per_pot : ℕ),
    basil_leaves_per_pot * basil_pots +
    rosemary_leaves_per_pot * rosemary_pots +
    thyme_leaves_per_pot * thyme_pots = total_leaves ∧
    basil_leaves_per_pot = 4 :=
by sorry

end NUMINAMATH_CALUDE_basil_leaves_count_l2641_264129


namespace NUMINAMATH_CALUDE_village_population_l2641_264103

theorem village_population (initial_population : ℝ) : 
  (initial_population * 1.2 * 0.8 = 9600) → initial_population = 10000 := by
  sorry

end NUMINAMATH_CALUDE_village_population_l2641_264103


namespace NUMINAMATH_CALUDE_jenna_photo_groups_l2641_264199

theorem jenna_photo_groups (n : ℕ) (k : ℕ) : n = 7 ∧ k = 3 → Nat.choose n k = 35 := by
  sorry

end NUMINAMATH_CALUDE_jenna_photo_groups_l2641_264199


namespace NUMINAMATH_CALUDE_base3_to_base10_conversion_l2641_264101

/-- Converts a base 3 number represented as a list of digits to its base 10 equivalent -/
def base3ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ i)) 0

/-- The base 3 representation of the number 102012₃ -/
def base3Number : List Nat := [2, 1, 0, 2, 0, 1]

theorem base3_to_base10_conversion :
  base3ToBase10 base3Number = 302 := by
  sorry

end NUMINAMATH_CALUDE_base3_to_base10_conversion_l2641_264101


namespace NUMINAMATH_CALUDE_triangle_area_bound_l2641_264174

/-- For any triangle with area S and semiperimeter p, S ≤ p^2 / (3√3) -/
theorem triangle_area_bound (S p : ℝ) (h_S : S > 0) (h_p : p > 0) 
  (h_triangle : ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    a + b > c ∧ b + c > a ∧ c + a > b ∧
    S = Real.sqrt (p * (p - a) * (p - b) * (p - c)) ∧
    p = (a + b + c) / 2) :
  S ≤ p^2 / (3 * Real.sqrt 3) := by
sorry

end NUMINAMATH_CALUDE_triangle_area_bound_l2641_264174


namespace NUMINAMATH_CALUDE_cube_root_inequality_l2641_264132

theorem cube_root_inequality (x : ℤ) : 
  (2 : ℝ) < (2 * (x : ℝ)^2)^(1/3) ∧ (2 * (x : ℝ)^2)^(1/3) < 3 ↔ x = 3 ∨ x = -3 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_inequality_l2641_264132


namespace NUMINAMATH_CALUDE_jewelry_store_profit_l2641_264150

/-- Calculates the gross profit for a pair of earrings -/
def earrings_gross_profit (purchase_price : ℚ) (markup_percentage : ℚ) (price_decrease_percentage : ℚ) : ℚ :=
  let initial_selling_price := purchase_price / (1 - markup_percentage)
  let price_decrease := initial_selling_price * price_decrease_percentage
  let final_selling_price := initial_selling_price - price_decrease
  final_selling_price - purchase_price

/-- Theorem stating the gross profit for the given scenario -/
theorem jewelry_store_profit :
  earrings_gross_profit 240 (25/100) (20/100) = 16 := by
  sorry

end NUMINAMATH_CALUDE_jewelry_store_profit_l2641_264150


namespace NUMINAMATH_CALUDE_quadratic_form_ratio_l2641_264131

theorem quadratic_form_ratio (k : ℝ) : 
  ∃ (d r s : ℝ), 8 * k^2 - 6 * k + 16 = d * (k + r)^2 + s ∧ s / r = -118 / 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_form_ratio_l2641_264131


namespace NUMINAMATH_CALUDE_digit_sum_2017_power_l2641_264120

/-- Sum of digits function -/
def S (n : ℕ) : ℕ := sorry

/-- The theorem to prove -/
theorem digit_sum_2017_power : S (S (S (S (2017^2017)))) = 1 := by sorry

end NUMINAMATH_CALUDE_digit_sum_2017_power_l2641_264120


namespace NUMINAMATH_CALUDE_even_function_properties_l2641_264172

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f x = -f (-x)

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem even_function_properties (f : ℝ → ℝ) 
  (h_even : is_even f) 
  (h_sum_zero : ∀ x, f x + f (2 - x) = 0) :
  is_periodic f 4 ∧ is_odd (fun x ↦ f (x - 1)) := by
  sorry

end NUMINAMATH_CALUDE_even_function_properties_l2641_264172


namespace NUMINAMATH_CALUDE_complex_number_equality_l2641_264142

theorem complex_number_equality : (1 + Complex.I)^2 * (1 - Complex.I) = 2 - 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_number_equality_l2641_264142


namespace NUMINAMATH_CALUDE_quadratic_condition_l2641_264191

/-- A quadratic equation in one variable is of the form ax^2 + bx + c = 0, where a ≠ 0 -/
def is_quadratic_equation (a b c : ℝ) : Prop :=
  a ≠ 0

/-- The equation ax^2 - x + 2 = 0 -/
def equation (a : ℝ) (x : ℝ) : Prop :=
  a * x^2 - x + 2 = 0

theorem quadratic_condition (a : ℝ) :
  (∃ x, equation a x) → is_quadratic_equation a (-1) 2 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_condition_l2641_264191


namespace NUMINAMATH_CALUDE_work_completion_theorem_l2641_264190

theorem work_completion_theorem (total_work : ℝ) :
  (34 : ℝ) * 18 * total_work = 17 * 36 * total_work := by
  sorry

#check work_completion_theorem

end NUMINAMATH_CALUDE_work_completion_theorem_l2641_264190


namespace NUMINAMATH_CALUDE_morning_faces_l2641_264143

/-- Represents a cuboid room -/
structure CuboidRoom where
  totalFaces : Nat
  eveningFaces : Nat

/-- Theorem: The number of faces Samuel painted in the morning is 3 -/
theorem morning_faces (room : CuboidRoom) 
  (h1 : room.totalFaces = 6)
  (h2 : room.eveningFaces = 3) : 
  room.totalFaces - room.eveningFaces = 3 := by
  sorry

#check morning_faces

end NUMINAMATH_CALUDE_morning_faces_l2641_264143


namespace NUMINAMATH_CALUDE_stock_worth_calculation_l2641_264130

/-- Proves that the total worth of the stock is 20000 given the specified conditions --/
theorem stock_worth_calculation (stock_worth : ℝ) : 
  (0.2 * stock_worth * 1.1 + 0.8 * stock_worth * 0.95 = stock_worth - 400) → 
  stock_worth = 20000 := by
  sorry

end NUMINAMATH_CALUDE_stock_worth_calculation_l2641_264130


namespace NUMINAMATH_CALUDE_cost_of_flour_for_cakes_claire_cake_flour_cost_l2641_264109

/-- The cost of flour for making cakes -/
theorem cost_of_flour_for_cakes (num_cakes : ℕ) (packages_per_cake : ℕ) (cost_per_package : ℕ) : 
  num_cakes * packages_per_cake * cost_per_package = num_cakes * (packages_per_cake * cost_per_package) :=
by sorry

/-- Claire's cake flour cost calculation -/
theorem claire_cake_flour_cost : 2 * (2 * 3) = 12 :=
by sorry

end NUMINAMATH_CALUDE_cost_of_flour_for_cakes_claire_cake_flour_cost_l2641_264109
