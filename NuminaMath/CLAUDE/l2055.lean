import Mathlib

namespace NUMINAMATH_CALUDE_point_on_x_axis_distance_to_origin_l2055_205532

/-- If a point P with coordinates (m-2, m+1) is on the x-axis, then the distance from P to the origin is 3. -/
theorem point_on_x_axis_distance_to_origin :
  ∀ m : ℝ, (m + 1 = 0) → Real.sqrt ((m - 2)^2 + 0^2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_point_on_x_axis_distance_to_origin_l2055_205532


namespace NUMINAMATH_CALUDE_fencing_requirement_l2055_205537

theorem fencing_requirement (area : ℝ) (uncovered_side : ℝ) : 
  area = 680 → uncovered_side = 20 → 
  ∃ (width : ℝ), 
    area = uncovered_side * width ∧ 
    uncovered_side + 2 * width = 88 := by
  sorry

end NUMINAMATH_CALUDE_fencing_requirement_l2055_205537


namespace NUMINAMATH_CALUDE_inscribed_cube_volume_in_specific_pyramid_l2055_205571

/-- A pyramid with a regular hexagonal base and isosceles triangular lateral faces -/
structure HexagonalPyramid where
  base_side_length : ℝ
  lateral_face_height : ℝ

/-- A cube inscribed in a hexagonal pyramid -/
structure InscribedCube where
  pyramid : HexagonalPyramid
  -- Each vertex of the cube is either on the base or touches a point on the lateral faces

/-- The volume of an inscribed cube in a hexagonal pyramid -/
def inscribed_cube_volume (cube : InscribedCube) : ℝ :=
  sorry

theorem inscribed_cube_volume_in_specific_pyramid :
  ∀ (cube : InscribedCube),
    cube.pyramid.base_side_length = 2 →
    cube.pyramid.lateral_face_height = 3 →
    inscribed_cube_volume cube = 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_cube_volume_in_specific_pyramid_l2055_205571


namespace NUMINAMATH_CALUDE_units_digit_of_7_power_2023_l2055_205586

theorem units_digit_of_7_power_2023 : 7^2023 % 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_7_power_2023_l2055_205586


namespace NUMINAMATH_CALUDE_sum_of_y_values_l2055_205531

theorem sum_of_y_values (x y z : ℝ) : 
  x + y = 7 → 
  x * z = -180 → 
  (x + y + z)^2 = 4 → 
  ∃ y₁ y₂ : ℝ, 
    (x + y₁ = 7 ∧ x * z = -180 ∧ (x + y₁ + z)^2 = 4) ∧
    (x + y₂ = 7 ∧ x * z = -180 ∧ (x + y₂ + z)^2 = 4) ∧
    y₁ ≠ y₂ ∧
    -(y₁ + y₂) = 42 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_y_values_l2055_205531


namespace NUMINAMATH_CALUDE_fraction_integrality_l2055_205514

theorem fraction_integrality (a b c : ℤ) 
  (h : ∃ (n : ℤ), (a * b / c + a * c / b + b * c / a) = n) : 
  (∃ (n1 : ℤ), a * b / c = n1) ∧ 
  (∃ (n2 : ℤ), a * c / b = n2) ∧ 
  (∃ (n3 : ℤ), b * c / a = n3) := by
sorry

end NUMINAMATH_CALUDE_fraction_integrality_l2055_205514


namespace NUMINAMATH_CALUDE_arithmetic_square_root_of_36_l2055_205543

theorem arithmetic_square_root_of_36 : 
  ∃ (x : ℝ), x ≥ 0 ∧ x^2 = 36 ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_of_36_l2055_205543


namespace NUMINAMATH_CALUDE_total_profit_is_35000_l2055_205556

/-- Represents the business subscription and profit distribution problem --/
structure BusinessProblem where
  total_subscription : ℕ
  a_more_than_b : ℕ
  b_more_than_c : ℕ
  c_profit : ℕ

/-- Calculates the total profit based on the given business problem --/
def calculate_total_profit (problem : BusinessProblem) : ℕ :=
  sorry

/-- Theorem stating that for the given conditions, the total profit is 35000 --/
theorem total_profit_is_35000 : 
  let problem := BusinessProblem.mk 50000 4000 5000 8400
  calculate_total_profit problem = 35000 := by
  sorry

end NUMINAMATH_CALUDE_total_profit_is_35000_l2055_205556


namespace NUMINAMATH_CALUDE_tan_sum_reciprocal_implies_sin_double_angle_l2055_205529

theorem tan_sum_reciprocal_implies_sin_double_angle (θ : Real) 
  (h : Real.tan θ + (Real.tan θ)⁻¹ = 4) : 
  Real.sin (2 * θ) = 1/2 := by sorry

end NUMINAMATH_CALUDE_tan_sum_reciprocal_implies_sin_double_angle_l2055_205529


namespace NUMINAMATH_CALUDE_nabla_problem_l2055_205567

-- Define the operation
def nabla (a b : ℕ) : ℕ := 3 + b^a

-- State the theorem
theorem nabla_problem : nabla (nabla 2 3) 4 = 16777219 := by
  sorry

end NUMINAMATH_CALUDE_nabla_problem_l2055_205567


namespace NUMINAMATH_CALUDE_sqrt_calculations_l2055_205584

theorem sqrt_calculations :
  (∀ (x : ℝ), x ≥ 0 → Real.sqrt (x ^ 2) = x) ∧
  (Real.sqrt 21 * Real.sqrt 3 / Real.sqrt 7 = 3) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_calculations_l2055_205584


namespace NUMINAMATH_CALUDE_positive_number_equality_l2055_205558

theorem positive_number_equality (x : ℝ) (h1 : x > 0) : 
  (2 / 3) * x = (144 / 216) * (1 / x) → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_positive_number_equality_l2055_205558


namespace NUMINAMATH_CALUDE_positive_numbers_inequality_l2055_205500

theorem positive_numbers_inequality (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum_squares : a^2 + b^2 + 4*c^2 = 3) :
  (a + b + 2*c ≤ 3) ∧ 
  (b = 2*c → 1/a + 1/c ≥ 3) := by
  sorry

end NUMINAMATH_CALUDE_positive_numbers_inequality_l2055_205500


namespace NUMINAMATH_CALUDE_prob_catch_carp_l2055_205509

/-- The probability of catching a carp in a pond with given conditions -/
theorem prob_catch_carp (num_carp num_tilapia : ℕ) (prob_grass_carp : ℚ) : 
  num_carp = 1600 →
  num_tilapia = 800 →
  prob_grass_carp = 1/2 →
  (num_carp : ℚ) / (num_carp + num_tilapia + (prob_grass_carp⁻¹ - 1) * (num_carp + num_tilapia)) = 1/3 :=
by sorry

end NUMINAMATH_CALUDE_prob_catch_carp_l2055_205509


namespace NUMINAMATH_CALUDE_prob_same_school_adjacent_l2055_205594

/-- The number of students from the first school -/
def students_school1 : ℕ := 2

/-- The number of students from the second school -/
def students_school2 : ℕ := 3

/-- The total number of students -/
def total_students : ℕ := students_school1 + students_school2

/-- The probability that students from the same school will be standing next to each other -/
def probability_same_school_adjacent : ℚ := 4/5

/-- Theorem stating that the probability of students from the same school standing next to each other is 4/5 -/
theorem prob_same_school_adjacent :
  probability_same_school_adjacent = 4/5 := by sorry

end NUMINAMATH_CALUDE_prob_same_school_adjacent_l2055_205594


namespace NUMINAMATH_CALUDE_max_prime_area_rectangle_l2055_205587

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def rectangleArea (l w : ℕ) : ℕ := l * w

def rectanglePerimeter (l w : ℕ) : ℕ := 2 * (l + w)

theorem max_prime_area_rectangle (l w : ℕ) :
  rectanglePerimeter l w = 40 →
  isPrime (rectangleArea l w) →
  rectangleArea l w ≤ 19 ∧
  (rectangleArea l w = 19 → (l = 1 ∧ w = 19) ∨ (l = 19 ∧ w = 1)) :=
sorry

end NUMINAMATH_CALUDE_max_prime_area_rectangle_l2055_205587


namespace NUMINAMATH_CALUDE_original_mango_price_l2055_205540

/-- Represents the price increase rate -/
def price_increase_rate : ℝ := 0.15

/-- Represents the original price of an orange -/
def original_orange_price : ℝ := 40

/-- Represents the total cost of 10 oranges and 10 mangoes after price increase -/
def total_cost : ℝ := 1035

/-- Represents the quantity of each fruit -/
def quantity : ℕ := 10

/-- Calculates the new price after applying the price increase -/
def new_price (original_price : ℝ) : ℝ :=
  original_price * (1 + price_increase_rate)

/-- Theorem stating that the original price of a mango was $50 -/
theorem original_mango_price :
  ∃ (original_mango_price : ℝ),
    original_mango_price = 50 ∧
    (quantity : ℝ) * new_price original_orange_price +
    (quantity : ℝ) * new_price original_mango_price = total_cost := by
  sorry

end NUMINAMATH_CALUDE_original_mango_price_l2055_205540


namespace NUMINAMATH_CALUDE_find_number_in_ten_questions_l2055_205526

theorem find_number_in_ten_questions (n : ℕ) (h : n ≤ 1000) :
  ∃ (questions : List (ℕ → Bool)) (answers : List Bool),
    questions.length ≤ 10 ∧
    answers.length = questions.length ∧
    (∀ m : ℕ, m ≤ 1000 → m ≠ n →
      ∃ i : ℕ, i < questions.length ∧
        (questions.get! i) m ≠ (answers.get! i)) :=
by sorry

end NUMINAMATH_CALUDE_find_number_in_ten_questions_l2055_205526


namespace NUMINAMATH_CALUDE_rectangular_field_width_l2055_205576

/-- The width of a rectangular field, given its length and a relationship between length and width. -/
def field_width (length : ℝ) (length_width_relation : ℝ → ℝ → Prop) : ℝ :=
  13.5

/-- Theorem stating that the width of a rectangular field is 13.5 meters, given specific conditions. -/
theorem rectangular_field_width :
  let length := 24
  let length_width_relation := λ l w => l = 2 * w - 3
  field_width length length_width_relation = 13.5 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_width_l2055_205576


namespace NUMINAMATH_CALUDE_max_length_AB_l2055_205524

/-- The function representing the length of AB -/
def f (t : ℝ) : ℝ := -2 * t^2 + 3 * t + 9

/-- The theorem stating the maximum value of f(t) for t in [0, 3] -/
theorem max_length_AB : 
  ∃ (t : ℝ), t ∈ Set.Icc 0 3 ∧ f t = 81/8 ∧ ∀ x ∈ Set.Icc 0 3, f x ≤ 81/8 :=
sorry

end NUMINAMATH_CALUDE_max_length_AB_l2055_205524


namespace NUMINAMATH_CALUDE_yard_area_l2055_205596

/-- The area of a rectangular yard with square cutouts -/
theorem yard_area (length width cutout_side : ℕ) (num_cutouts : ℕ) : 
  length = 20 → 
  width = 18 → 
  cutout_side = 4 → 
  num_cutouts = 2 → 
  length * width - num_cutouts * cutout_side * cutout_side = 328 := by
  sorry

end NUMINAMATH_CALUDE_yard_area_l2055_205596


namespace NUMINAMATH_CALUDE_max_value_cos_sin_l2055_205517

theorem max_value_cos_sin (x : ℝ) : 3 * Real.cos x + 4 * Real.sin x ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_max_value_cos_sin_l2055_205517


namespace NUMINAMATH_CALUDE_parabola_tangent_secant_relation_l2055_205592

/-- A parabola with its axis parallel to the y-axis -/
structure Parabola where
  a : ℝ
  f : ℝ → ℝ
  f_eq : f = fun x ↦ a * x^2

/-- A point on the parabola -/
structure PointOnParabola (p : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : y = p.f x

/-- Tangent of the angle of inclination of the tangent at a point -/
def tangentSlope (p : Parabola) (point : PointOnParabola p) : ℝ :=
  2 * p.a * point.x

/-- Tangent of the angle of inclination of the secant line between two points -/
def secantSlope (p : Parabola) (p1 p2 : PointOnParabola p) : ℝ :=
  p.a * (p1.x + p2.x)

/-- The main theorem -/
theorem parabola_tangent_secant_relation (p : Parabola) 
    (A1 A2 A3 : PointOnParabola p) : 
    tangentSlope p A1 = secantSlope p A1 A2 + secantSlope p A1 A3 - secantSlope p A2 A3 := by
  sorry

end NUMINAMATH_CALUDE_parabola_tangent_secant_relation_l2055_205592


namespace NUMINAMATH_CALUDE_diamond_op_four_three_l2055_205523

def diamond_op (m n : ℕ) : ℕ := n ^ 2 - m

theorem diamond_op_four_three : diamond_op 4 3 = 5 := by sorry

end NUMINAMATH_CALUDE_diamond_op_four_three_l2055_205523


namespace NUMINAMATH_CALUDE_sphere_to_cone_radius_l2055_205507

/-- The radius of a sphere that transforms into a cone with equal volume --/
theorem sphere_to_cone_radius (r : ℝ) (h : r = 3 * Real.rpow 2 (1/3)) :
  ∃ R : ℝ, 
    (4/3) * Real.pi * R^3 = 2 * Real.pi * r^3 ∧ 
    R = 3 * Real.rpow 3 (1/3) :=
sorry

end NUMINAMATH_CALUDE_sphere_to_cone_radius_l2055_205507


namespace NUMINAMATH_CALUDE_green_hats_count_l2055_205581

/-- Proves that the number of green hats is 28 given the problem conditions --/
theorem green_hats_count : ∀ (blue green red : ℕ),
  blue + green + red = 85 →
  6 * blue + 7 * green + 8 * red = 600 →
  blue = 3 * green ∧ green = 2 * red →
  green = 28 := by
  sorry

end NUMINAMATH_CALUDE_green_hats_count_l2055_205581


namespace NUMINAMATH_CALUDE_steven_fruit_difference_l2055_205580

/-- The number of apples Steven has -/
def steven_apples : ℕ := 19

/-- The number of peaches Steven has -/
def steven_peaches : ℕ := 15

/-- The difference between Steven's apples and peaches -/
def apple_peach_difference : ℕ := steven_apples - steven_peaches

theorem steven_fruit_difference : apple_peach_difference = 4 := by
  sorry

end NUMINAMATH_CALUDE_steven_fruit_difference_l2055_205580


namespace NUMINAMATH_CALUDE_solve_for_y_l2055_205597

theorem solve_for_y (x y : ℝ) (h1 : x^2 + x + 4 = y - 4) (h2 : x = 3) : y = 20 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l2055_205597


namespace NUMINAMATH_CALUDE_acceleration_charged_spheres_l2055_205589

/-- Acceleration of a small charged sphere near a uniformly charged larger sphere with removed material -/
theorem acceleration_charged_spheres 
  (k : ℝ) -- Coulomb's constant
  (q Q : ℝ) -- Charges of small and large spheres
  (r R : ℝ) -- Radii of removed portion and large sphere
  (m : ℝ) -- Mass of small sphere
  (L S : ℝ) -- Distances
  (g : ℝ) -- Acceleration due to gravity
  (h_r_small : r < R) -- r is smaller than R
  (h_r_pos : r > 0)
  (h_R_pos : R > 0)
  (h_m_pos : m > 0)
  (h_L_pos : L > 0)
  (h_S_pos : S > 0)
  (h_k_pos : k > 0)
  (h_g_pos : g > 0) :
  ∃ a : ℝ, a = (k * q * Q * r^3) / (m * R^3 * (L + 2*R - S)^2) :=
by sorry

end NUMINAMATH_CALUDE_acceleration_charged_spheres_l2055_205589


namespace NUMINAMATH_CALUDE_ratio_is_pure_imaginary_l2055_205516

theorem ratio_is_pure_imaginary (z₁ z₂ : ℂ) (hz₁ : z₁ ≠ 0) (hz₂ : z₂ ≠ 0) 
  (h : Complex.abs (z₁ + z₂) = Complex.abs (z₁ - z₂)) : 
  ∃ (y : ℝ), z₁ / z₂ = Complex.I * y := by
  sorry

end NUMINAMATH_CALUDE_ratio_is_pure_imaginary_l2055_205516


namespace NUMINAMATH_CALUDE_triangle_solutions_l2055_205546

/-- Function to determine the number of triangle solutions given two sides and an angle --/
def triangleSolutionsCount (a b : ℝ) (angleA : Real) : Nat :=
  sorry

theorem triangle_solutions :
  (triangleSolutionsCount 5 4 (120 * π / 180) = 1) ∧
  (triangleSolutionsCount 7 14 (150 * π / 180) = 0) ∧
  (triangleSolutionsCount 9 10 (60 * π / 180) = 2) := by
  sorry


end NUMINAMATH_CALUDE_triangle_solutions_l2055_205546


namespace NUMINAMATH_CALUDE_fermat_point_distance_sum_l2055_205563

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (12, 0)
def C : ℝ × ℝ := (3, 5)
def P : ℝ × ℝ := (5, 3)

theorem fermat_point_distance_sum :
  let AP := distance A.1 A.2 P.1 P.2
  let BP := distance B.1 B.2 P.1 P.2
  let CP := distance C.1 C.2 P.1 P.2
  AP + BP + CP = Real.sqrt 34 + Real.sqrt 58 + 2 * Real.sqrt 2 ∧
  (1 : ℕ) + (1 : ℕ) + (2 : ℕ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_fermat_point_distance_sum_l2055_205563


namespace NUMINAMATH_CALUDE_slope_of_line_from_equation_l2055_205538

-- Define the equation
def satisfies_equation (x y : ℝ) : Prop := 3 / x + 4 / y = 0

-- Theorem statement
theorem slope_of_line_from_equation :
  ∀ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁ ≠ x₂ →
    satisfies_equation x₁ y₁ →
    satisfies_equation x₂ y₂ →
    (y₂ - y₁) / (x₂ - x₁) = -4/3 :=
by sorry

end NUMINAMATH_CALUDE_slope_of_line_from_equation_l2055_205538


namespace NUMINAMATH_CALUDE_triangle_inequality_bounds_l2055_205505

theorem triangle_inequality_bounds (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a ≤ b) (hbc : b ≤ c) (htri : a + b > c) :
  4 ≤ (a + b + c)^2 / (b * c) ∧ (a + b + c)^2 / (b * c) ≤ 9 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_bounds_l2055_205505


namespace NUMINAMATH_CALUDE_set_operations_and_range_l2055_205562

def A : Set ℝ := {x | 4 ≤ x ∧ x < 8}
def B : Set ℝ := {x | 2 < x ∧ x < 10}
def C (a : ℝ) : Set ℝ := {x | x < a}

theorem set_operations_and_range :
  (A ∪ B = {x | 2 < x ∧ x < 10}) ∧
  ((Set.univ \ A) ∩ B = {x | (8 ≤ x ∧ x < 10) ∨ (2 < x ∧ x < 4)}) ∧
  (∀ a : ℝ, (A ∩ C a).Nonempty → 4 ≤ a) :=
by sorry

end NUMINAMATH_CALUDE_set_operations_and_range_l2055_205562


namespace NUMINAMATH_CALUDE_unique_solution_cube_difference_square_l2055_205530

/-- A predicate that checks if a number is prime -/
def IsPrime (p : ℕ) : Prop := Nat.Prime p

/-- A predicate that checks if a number is not divisible by 3 or by another number -/
def NotDivisibleBy3OrY (z y : ℕ) : Prop := ¬(z % 3 = 0) ∧ ¬(z % y = 0)

theorem unique_solution_cube_difference_square :
  ∀ x y z : ℕ,
    x > 0 → y > 0 → z > 0 →
    IsPrime y →
    NotDivisibleBy3OrY z y →
    x^3 - y^3 = z^2 →
    x = 8 ∧ y = 7 ∧ z = 13 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_cube_difference_square_l2055_205530


namespace NUMINAMATH_CALUDE_f_negative_a_l2055_205552

noncomputable def f (x : ℝ) : ℝ := x^2 * Real.log (-x + Real.sqrt (x^2 + 1)) + 1

theorem f_negative_a (a : ℝ) (h : f a = 11) : f (-a) = -9 := by
  sorry

end NUMINAMATH_CALUDE_f_negative_a_l2055_205552


namespace NUMINAMATH_CALUDE_highest_common_factor_l2055_205512

/- Define the polynomials f and g -/
def f (n : ℕ) (x : ℝ) : ℝ := n * x^(n+1) - (n+1) * x^n + 1

def g (n : ℕ) (x : ℝ) : ℝ := x^n - n*x + n - 1

/- State the theorem -/
theorem highest_common_factor (n : ℕ) (h : n ≥ 2) :
  ∃ (p q : ℝ → ℝ), 
    (∀ x, f n x = (x - 1)^2 * p x) ∧ 
    (∀ x, g n x = (x - 1) * q x) ∧
    (∀ r : ℝ → ℝ, (∀ x, f n x = r x * (p x)) → (∀ x, g n x = r x * (q x)) → 
      ∃ (s : ℝ → ℝ), ∀ x, r x = (x - 1)^2 * s x) :=
sorry

end NUMINAMATH_CALUDE_highest_common_factor_l2055_205512


namespace NUMINAMATH_CALUDE_triangle_perimeter_l2055_205545

theorem triangle_perimeter : ∀ (a b c : ℕ), 
  a = 2 → b = 5 → Odd c → a + b + c = 12 → 
  (a < b + c ∧ b < a + c ∧ c < a + b) → 
  a + b + c = 12 := by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l2055_205545


namespace NUMINAMATH_CALUDE_triangle_properties_l2055_205570

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_properties (t : Triangle) (h1 : (t.a + t.c) * Real.sin t.A = Real.sin t.A + Real.sin t.C)
    (h2 : t.c^2 + t.c = t.b^2 - 1) (h3 : t.a = 1) (h4 : t.c = 2) :
    t.B = 2 * Real.pi / 3 ∧ (1/2 * t.a * t.c * Real.sin t.B = Real.sqrt 3 / 2) := by
  sorry

#check triangle_properties

end NUMINAMATH_CALUDE_triangle_properties_l2055_205570


namespace NUMINAMATH_CALUDE_parallel_iff_slope_eq_l2055_205515

/-- Two lines in the plane -/
structure Line where
  k : ℝ
  b : ℝ

/-- Define when two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.k = l2.k

/-- The main theorem: k1 = k2 iff l1 ∥ l2 -/
theorem parallel_iff_slope_eq (l1 l2 : Line) :
  l1.k = l2.k ↔ parallel l1 l2 :=
by sorry

end NUMINAMATH_CALUDE_parallel_iff_slope_eq_l2055_205515


namespace NUMINAMATH_CALUDE_race_time_proof_l2055_205506

/-- A runner completes a race -/
structure Runner where
  distance : ℝ  -- distance covered
  time : ℝ      -- time taken

/-- A race between two runners -/
structure Race where
  length : ℝ           -- race length
  runner_a : Runner    -- runner A
  runner_b : Runner    -- runner B

/-- Given a race satisfying the problem conditions, prove that runner A's time is 7 seconds -/
theorem race_time_proof (race : Race) 
  (h1 : race.length = 200)
  (h2 : race.runner_a.distance - race.runner_b.distance = 35)
  (h3 : race.runner_a.distance = race.length)
  (h4 : race.runner_a.time = 7) :
  race.runner_a.time = 7 := by sorry

end NUMINAMATH_CALUDE_race_time_proof_l2055_205506


namespace NUMINAMATH_CALUDE_gcd_4557_5115_l2055_205519

theorem gcd_4557_5115 : Nat.gcd 4557 5115 = 93 := by
  sorry

end NUMINAMATH_CALUDE_gcd_4557_5115_l2055_205519


namespace NUMINAMATH_CALUDE_initial_rows_count_l2055_205599

theorem initial_rows_count (chairs_per_row : ℕ) (extra_chairs : ℕ) (total_chairs : ℕ) 
  (h1 : chairs_per_row = 12)
  (h2 : extra_chairs = 11)
  (h3 : total_chairs = 95) :
  ∃ (initial_rows : ℕ), initial_rows * chairs_per_row + extra_chairs = total_chairs ∧ initial_rows = 7 := by
  sorry

end NUMINAMATH_CALUDE_initial_rows_count_l2055_205599


namespace NUMINAMATH_CALUDE_special_line_equation_l2055_205585

/-- A line passing through (1,2) with its y-intercept twice its x-intercept -/
structure SpecialLine where
  /-- The slope of the line -/
  m : ℝ
  /-- The y-intercept of the line -/
  b : ℝ
  /-- The line passes through (1,2) -/
  passes_through : m + b = 2
  /-- The y-intercept is twice the x-intercept -/
  intercept_condition : b = 2 * (-b / m)

/-- The equation of the special line is either y = 2x or 2x + y - 4 = 0 -/
theorem special_line_equation (l : SpecialLine) :
  (l.m = 2 ∧ l.b = 0) ∨ (l.m = -2 ∧ l.b = 4) := by
  sorry

end NUMINAMATH_CALUDE_special_line_equation_l2055_205585


namespace NUMINAMATH_CALUDE_johns_number_is_eight_l2055_205551

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

theorem johns_number_is_eight :
  ∃! x : ℕ, is_two_digit x ∧
    81 ≤ reverse_digits (5 * x + 18) ∧
    reverse_digits (5 * x + 18) ≤ 85 ∧
    x = 8 := by
  sorry

end NUMINAMATH_CALUDE_johns_number_is_eight_l2055_205551


namespace NUMINAMATH_CALUDE_three_digit_45_arithmetic_sequence_l2055_205549

def is_arithmetic_sequence (a b c : ℕ) : Prop :=
  b = (a + c) / 2

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def digits_sum (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

theorem three_digit_45_arithmetic_sequence :
  ∀ n : ℕ, is_three_digit n →
            n % 45 = 0 →
            is_arithmetic_sequence (n / 100) ((n / 10) % 10) (n % 10) →
            (n = 135 ∨ n = 630 ∨ n = 765) :=
sorry

end NUMINAMATH_CALUDE_three_digit_45_arithmetic_sequence_l2055_205549


namespace NUMINAMATH_CALUDE_rational_equation_solution_l2055_205575

theorem rational_equation_solution :
  ∃ x : ℝ, x ≠ 2 ∧ x ≠ (4/5 : ℝ) ∧
  (x^2 - 11*x + 24)/(x - 2) + (5*x^2 + 20*x - 40)/(5*x - 4) = -5 ∧
  x = -3 := by
  sorry

end NUMINAMATH_CALUDE_rational_equation_solution_l2055_205575


namespace NUMINAMATH_CALUDE_bank_withdrawal_total_l2055_205579

theorem bank_withdrawal_total (x y : ℕ) : 
  x / 20 + y / 20 = 30 → x + y = 600 := by
  sorry

end NUMINAMATH_CALUDE_bank_withdrawal_total_l2055_205579


namespace NUMINAMATH_CALUDE_canoe_trip_time_rita_canoe_trip_time_l2055_205534

/-- Calculates the total time for a round trip given upstream and downstream speeds and distance -/
theorem canoe_trip_time (upstream_speed downstream_speed distance : ℝ) :
  upstream_speed > 0 →
  downstream_speed > 0 →
  distance > 0 →
  (distance / upstream_speed) + (distance / downstream_speed) =
    (upstream_speed + downstream_speed) * distance / (upstream_speed * downstream_speed) := by
  sorry

/-- Proves that Rita's canoe trip takes 8 hours -/
theorem rita_canoe_trip_time :
  let upstream_speed : ℝ := 3
  let downstream_speed : ℝ := 9
  let distance : ℝ := 18
  (distance / upstream_speed) + (distance / downstream_speed) = 8 := by
  sorry

end NUMINAMATH_CALUDE_canoe_trip_time_rita_canoe_trip_time_l2055_205534


namespace NUMINAMATH_CALUDE_ball_distribution_proof_l2055_205527

/-- The number of ways to distribute n distinct balls into k distinct boxes -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute n distinct balls into k distinct boxes with no empty boxes -/
def distributeNoEmpty (n k : ℕ) : ℕ := sorry

theorem ball_distribution_proof :
  distributeNoEmpty 4 3 = 36 := by sorry

end NUMINAMATH_CALUDE_ball_distribution_proof_l2055_205527


namespace NUMINAMATH_CALUDE_unique_six_digit_number_l2055_205503

def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 1000000 ∧
  (1000 * (n % 1000) + n / 1000 = 6 * n)

theorem unique_six_digit_number :
  ∃! n : ℕ, is_valid_number n ∧ n = 142857 :=
by sorry

end NUMINAMATH_CALUDE_unique_six_digit_number_l2055_205503


namespace NUMINAMATH_CALUDE_hockey_players_count_l2055_205590

theorem hockey_players_count (total_players cricket_players football_players softball_players : ℕ) 
  (h_total : total_players = 55)
  (h_cricket : cricket_players = 15)
  (h_football : football_players = 13)
  (h_softball : softball_players = 15) :
  total_players - (cricket_players + football_players + softball_players) = 12 := by
  sorry

#check hockey_players_count

end NUMINAMATH_CALUDE_hockey_players_count_l2055_205590


namespace NUMINAMATH_CALUDE_fraction_equality_l2055_205595

theorem fraction_equality (a b : ℝ) (h : b ≠ 0) : a / b = (2 * a) / (2 * b) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2055_205595


namespace NUMINAMATH_CALUDE_x_plus_3_over_x_is_fraction_l2055_205588

/-- A fraction is an expression with a variable in the denominator. -/
def is_fraction (f : ℚ → ℚ) : Prop :=
  ∃ (n d : ℚ → ℚ), ∀ x, f x = (n x) / (d x) ∧ d x ≠ 0

/-- The expression (x + 3) / x is a fraction. -/
theorem x_plus_3_over_x_is_fraction :
  is_fraction (λ x => (x + 3) / x) :=
sorry

end NUMINAMATH_CALUDE_x_plus_3_over_x_is_fraction_l2055_205588


namespace NUMINAMATH_CALUDE_log_equation_equals_zero_l2055_205568

-- Define the logarithm function with base 5
noncomputable def log5 (x : ℝ) : ℝ := Real.log x / Real.log 5

-- State the theorem
theorem log_equation_equals_zero : 2 * log5 10 + log5 0.25 = 0 := by sorry

end NUMINAMATH_CALUDE_log_equation_equals_zero_l2055_205568


namespace NUMINAMATH_CALUDE_abc_inequality_abc_inequality_tight_l2055_205573

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a * b * c * (a + b + c)) / ((a + b)^3 * (b + c)^3) ≤ 1/8 :=
sorry

theorem abc_inequality_tight :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
  (a * b * c * (a + b + c)) / ((a + b)^3 * (b + c)^3) = 1/8 :=
sorry

end NUMINAMATH_CALUDE_abc_inequality_abc_inequality_tight_l2055_205573


namespace NUMINAMATH_CALUDE_track_length_is_24_l2055_205511

/-- Represents a circular ski track -/
structure SkiTrack where
  length : ℝ
  uphill_speed : ℝ
  downhill_speed : ℝ

/-- The conditions of the problem -/
def problem_conditions (track : SkiTrack) : Prop :=
  track.downhill_speed = 4 * track.uphill_speed ∧
  track.length > 0 ∧
  ∃ (min_distance max_distance : ℝ),
    min_distance = 4 ∧
    max_distance = 13 ∧
    max_distance - min_distance = 9

/-- The theorem to be proved -/
theorem track_length_is_24 (track : SkiTrack) :
  problem_conditions track → track.length = 24 := by
  sorry

end NUMINAMATH_CALUDE_track_length_is_24_l2055_205511


namespace NUMINAMATH_CALUDE_medical_team_selection_l2055_205533

theorem medical_team_selection (male_doctors : ℕ) (female_doctors : ℕ) : 
  male_doctors = 6 → female_doctors = 5 → 
  (male_doctors.choose 2) * (female_doctors.choose 1) = 75 := by
sorry

end NUMINAMATH_CALUDE_medical_team_selection_l2055_205533


namespace NUMINAMATH_CALUDE_intersection_point_in_interval_l2055_205539

open Real

theorem intersection_point_in_interval (f g : ℝ → ℝ) (x₀ : ℝ) :
  (∀ x, f x = x^3) →
  (∀ x, g x = 2^x + 1) →
  f x₀ = g x₀ →
  1 < x₀ ∧ x₀ < 2 := by
sorry

end NUMINAMATH_CALUDE_intersection_point_in_interval_l2055_205539


namespace NUMINAMATH_CALUDE_no_integer_solution_l2055_205572

theorem no_integer_solution : ∀ x y : ℤ, (x + 7) * (x + 6) ≠ 8 * y + 3 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l2055_205572


namespace NUMINAMATH_CALUDE_constant_revenue_increase_l2055_205541

def revenue : Fin 14 → ℕ
  | 0  => 150000  -- January (year 1)
  | 1  => 180000  -- February (year 1)
  | 2  => 210000  -- March (year 1)
  | 3  => 240000  -- April (year 1)
  | 4  => 270000  -- May (year 1)
  | 5  => 300000  -- June (year 1)
  | 6  => 330000  -- July (year 1)
  | 7  => 300000  -- August (year 1)
  | 8  => 270000  -- September (year 1)
  | 9  => 300000  -- October (year 1)
  | 10 => 330000  -- November (year 1)
  | 11 => 360000  -- December (year 1)
  | 12 => 390000  -- January (year 2)
  | 13 => 420000  -- February (year 2)

theorem constant_revenue_increase :
  ∀ i : Fin 13, i.val ≠ 6 ∧ i.val ≠ 7 →
    revenue (i + 1) - revenue i = 30000 :=
by sorry

end NUMINAMATH_CALUDE_constant_revenue_increase_l2055_205541


namespace NUMINAMATH_CALUDE_problem_polygon_area_l2055_205528

/-- A polygon in 2D space defined by a list of points --/
def Polygon : Type := List (ℝ × ℝ)

/-- The polygon described in the problem --/
def problemPolygon : Polygon :=
  [(0,0), (5,0), (5,5), (0,5), (0,3), (3,3), (3,0), (0,0)]

/-- Calculates the area of a polygon --/
def polygonArea (p : Polygon) : ℝ :=
  sorry  -- The actual calculation would go here

/-- Theorem: The area of the problem polygon is 19 square units --/
theorem problem_polygon_area :
  polygonArea problemPolygon = 19 := by
  sorry

end NUMINAMATH_CALUDE_problem_polygon_area_l2055_205528


namespace NUMINAMATH_CALUDE_mitzi_amusement_park_money_l2055_205560

def ticket_cost : ℕ := 30
def food_cost : ℕ := 13
def tshirt_cost : ℕ := 23
def remaining_money : ℕ := 9

theorem mitzi_amusement_park_money :
  ticket_cost + food_cost + tshirt_cost + remaining_money = 75 := by
  sorry

end NUMINAMATH_CALUDE_mitzi_amusement_park_money_l2055_205560


namespace NUMINAMATH_CALUDE_x_power_four_plus_reciprocal_l2055_205578

theorem x_power_four_plus_reciprocal (x : ℝ) (h : x ≠ 0) :
  x^2 + (1/x^2) = 2 → x^4 + (1/x^4) = 2 := by
sorry

end NUMINAMATH_CALUDE_x_power_four_plus_reciprocal_l2055_205578


namespace NUMINAMATH_CALUDE_student_count_l2055_205591

theorem student_count (right_rank left_rank : ℕ) 
  (h1 : right_rank = 13) 
  (h2 : left_rank = 8) : 
  right_rank + left_rank - 1 = 20 := by
  sorry

end NUMINAMATH_CALUDE_student_count_l2055_205591


namespace NUMINAMATH_CALUDE_square_area_with_inscribed_triangle_l2055_205583

theorem square_area_with_inscribed_triangle (d : ℝ) (h : d = 16) : 
  let s := d / Real.sqrt 2
  let square_area := s^2
  square_area = 128 := by
  sorry

end NUMINAMATH_CALUDE_square_area_with_inscribed_triangle_l2055_205583


namespace NUMINAMATH_CALUDE_cubic_root_complex_coefficients_l2055_205536

theorem cubic_root_complex_coefficients :
  ∀ (a b : ℝ),
  (∃ (x : ℂ), x^3 + a*x^2 + 2*x + b = 0 ∧ x = Complex.mk 2 (-3)) →
  a = -5/4 ∧ b = 143/4 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_complex_coefficients_l2055_205536


namespace NUMINAMATH_CALUDE_sqrt_inequality_l2055_205521

theorem sqrt_inequality (a b : ℝ) (ha : a > 0) (hb : 1/b - 1/a > 1) :
  Real.sqrt (1 + a) > 1 / Real.sqrt (1 - b) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l2055_205521


namespace NUMINAMATH_CALUDE_sequence_problem_l2055_205550

theorem sequence_problem (n : ℕ) (a_n : ℕ → ℕ) : 
  (∀ k, a_n k = 3 * k + 4) → a_n n = 13 → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_sequence_problem_l2055_205550


namespace NUMINAMATH_CALUDE_total_amount_l2055_205569

/-- Represents the distribution of money among three parties -/
structure MoneyDistribution where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The problem statement -/
def money_problem (d : MoneyDistribution) : Prop :=
  d.y = 45 ∧                    -- y's share is 45 rupees
  d.y = 0.45 * d.x ∧            -- y gets 0.45 rupees for each rupee x gets
  d.z = 0.50 * d.x              -- z gets 0.50 rupees for each rupee x gets

/-- The theorem to prove -/
theorem total_amount (d : MoneyDistribution) :
  money_problem d → d.x + d.y + d.z = 195 :=
by
  sorry


end NUMINAMATH_CALUDE_total_amount_l2055_205569


namespace NUMINAMATH_CALUDE_solution_of_equation_l2055_205548

theorem solution_of_equation :
  let f (x : ℝ) := 
    8 / (Real.sqrt (x - 10) - 10) + 
    2 / (Real.sqrt (x - 10) - 5) + 
    9 / (Real.sqrt (x - 10) + 5) + 
    16 / (Real.sqrt (x - 10) + 10)
  ∀ x : ℝ, f x = 0 ↔ x = 1841 / 121 ∨ x = 190 / 9 := by
  sorry

end NUMINAMATH_CALUDE_solution_of_equation_l2055_205548


namespace NUMINAMATH_CALUDE_spherical_cap_height_theorem_l2055_205565

/-- The height of a spherical cap -/
def spherical_cap_height (R : ℝ) (c : ℝ) : Set ℝ :=
  {h | h = 2*R*(c-1)/c ∨ h = 2*R*(c-2)/(c-1)}

/-- Theorem: The height of a spherical cap with radius R, whose surface area is c times 
    the area of its circular base (c > 1), is either 2R(c-1)/c or 2R(c-2)/(c-1) -/
theorem spherical_cap_height_theorem (R c : ℝ) (hR : R > 0) (hc : c > 1) :
  ∃ h ∈ spherical_cap_height R c,
    (∃ S_cap S_base : ℝ, 
      S_cap = c * S_base ∧
      ((S_cap = 2 * π * R * h ∧ S_base = π * (2*R*h - h^2)) ∨
       (S_cap = 2 * π * R * h + π * (2*R*h - h^2) ∧ S_base = π * (2*R*h - h^2)))) :=
by
  sorry

end NUMINAMATH_CALUDE_spherical_cap_height_theorem_l2055_205565


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2055_205501

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Line → Prop)
variable (perpToPlane : Line → Plane → Prop)
variable (para : Line → Plane → Prop)

-- Define the given objects
variable (l m : Line) (α : Plane)

-- Define the condition that l is perpendicular to α
variable (h : perpToPlane l α)

-- State the theorem
theorem sufficient_not_necessary :
  (∀ m, para m α → perp m l) ∧
  (∃ m, perp m l ∧ ¬para m α) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2055_205501


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l2055_205542

theorem cubic_equation_solution (x : ℝ) : x^3 + 64 = 0 → x = -4 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l2055_205542


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_l2055_205554

theorem root_sum_reciprocal (p : ℝ) (x₁ x₂ : ℝ) 
  (h1 : x₁^2 - 6*p*x₁ + p^2 = 0)
  (h2 : x₂^2 - 6*p*x₂ + p^2 = 0)
  (h3 : x₁ ≠ x₂)
  (h4 : p ≠ 0) :
  1 / (x₁ + p) + 1 / (x₂ + p) = 1 / p :=
sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_l2055_205554


namespace NUMINAMATH_CALUDE_problem_solution_l2055_205520

theorem problem_solution (A B C D : ℕ+) : 
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D →
  A * B = 72 →
  C * D = 72 →
  A - B = C * D →
  A = 3 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2055_205520


namespace NUMINAMATH_CALUDE_mail_delivery_l2055_205553

theorem mail_delivery (total : ℕ) (johann : ℕ) (friends : ℕ) : 
  total = 180 → 
  johann = 98 → 
  friends = 2 → 
  (total - johann) % friends = 0 → 
  (total - johann) / friends = 41 :=
by sorry

end NUMINAMATH_CALUDE_mail_delivery_l2055_205553


namespace NUMINAMATH_CALUDE_solution_characterization_l2055_205525

/-- A function satisfying the given functional equation and smoothness condition. -/
structure SolutionFunction where
  f : ℝ → ℝ
  smooth : ContDiff ℝ 2 f
  eq : ∀ x, f (7 * x + 1) = 49 * f x

/-- The theorem stating the form of all functions satisfying the conditions. -/
theorem solution_characterization (sf : SolutionFunction) :
  ∃ b : ℝ, ∀ x, sf.f x = b * (x + 1/6)^2 := by
  sorry

end NUMINAMATH_CALUDE_solution_characterization_l2055_205525


namespace NUMINAMATH_CALUDE_complex_modulus_three_fourths_minus_two_fifths_i_l2055_205535

theorem complex_modulus_three_fourths_minus_two_fifths_i :
  Complex.abs (3/4 - (2/5)*Complex.I) = 17/20 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_three_fourths_minus_two_fifths_i_l2055_205535


namespace NUMINAMATH_CALUDE_hyperbola_parabola_ratio_l2055_205513

/-- Given a hyperbola and a parabola with specific properties, prove that the ratio of the hyperbola's semi-major and semi-minor axes is equal to √3/3. -/
theorem hyperbola_parabola_ratio (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →  -- Hyperbola equation
  (∃ c : ℝ, c^2 = a^2 + b^2) →  -- Relationship between a, b, and c in a hyperbola
  (c / a = 2) →  -- Eccentricity is 2
  (c = 1) →  -- Right focus coincides with the focus of y^2 = 4x
  a / b = Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_parabola_ratio_l2055_205513


namespace NUMINAMATH_CALUDE_parabola_intersection_theorem_l2055_205518

/-- Parabola with equation y^2 = 2x -/
structure Parabola where
  equation : ℝ → ℝ → Prop
  focus : ℝ × ℝ

/-- Line passing through a point -/
structure Line where
  point : ℝ × ℝ
  slope : ℝ

/-- Intersection points of a line and a parabola -/
def Intersection (p : Parabola) (l : Line) : Set (ℝ × ℝ) :=
  {pt | p.equation pt.1 pt.2 ∧ pt.2 = l.slope * (pt.1 - l.point.1) + l.point.2}

theorem parabola_intersection_theorem (p : Parabola) (l : Line) 
    (A B : ℝ × ℝ) (hA : A ∈ Intersection p l) (hB : B ∈ Intersection p l) :
  p.equation 0.5 0 →  -- Focus is on the parabola
  l.point = p.focus →  -- Line passes through the focus
  ‖A - B‖ = 25/12 →  -- Distance between A and B
  ‖A - p.focus‖ < ‖B - p.focus‖ →  -- AF < BF
  ‖A - p.focus‖ = 5/6 :=  -- |AF| = 5/6
by sorry

end NUMINAMATH_CALUDE_parabola_intersection_theorem_l2055_205518


namespace NUMINAMATH_CALUDE_bells_lcm_l2055_205504

theorem bells_lcm (a b c d e f : ℕ) 
  (ha : a = 3) (hb : b = 5) (hc : c = 8) (hd : d = 11) (he : e = 15) (hf : f = 20) : 
  Nat.lcm a (Nat.lcm b (Nat.lcm c (Nat.lcm d (Nat.lcm e f)))) = 1320 := by
  sorry

end NUMINAMATH_CALUDE_bells_lcm_l2055_205504


namespace NUMINAMATH_CALUDE_sequence_characterization_l2055_205593

/-- An infinite sequence of positive integers -/
def Sequence := ℕ → ℕ

/-- The property that the sequence is strictly increasing -/
def StrictlyIncreasing (a : Sequence) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

/-- The property that no three terms in the sequence sum to another term -/
def NoThreeSum (a : Sequence) : Prop :=
  ∀ i j k : ℕ, a i + a j ≠ a k

/-- The property that infinitely many terms of the sequence are of the form 2k - 1 -/
def InfinitelyManyOdd (a : Sequence) : Prop :=
  ∀ N : ℕ, ∃ k : ℕ, k > N ∧ a k = 2 * k - 1

/-- The main theorem: any sequence satisfying the given properties must be aₙ = 2n - 1 -/
theorem sequence_characterization (a : Sequence)
  (h1 : StrictlyIncreasing a)
  (h2 : NoThreeSum a)
  (h3 : InfinitelyManyOdd a) :
  ∀ n : ℕ, a n = 2 * n - 1 :=
by sorry

end NUMINAMATH_CALUDE_sequence_characterization_l2055_205593


namespace NUMINAMATH_CALUDE_one_thirteenth_150th_digit_l2055_205555

def decimal_representation (n : ℕ) : ℕ := 
  match n % 6 with
  | 1 => 0
  | 2 => 7
  | 3 => 6
  | 4 => 9
  | 5 => 2
  | 0 => 3
  | _ => 0  -- This case should never occur, but Lean requires it for exhaustiveness

theorem one_thirteenth_150th_digit : 
  decimal_representation 150 = 3 := by
sorry


end NUMINAMATH_CALUDE_one_thirteenth_150th_digit_l2055_205555


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fourth_term_l2055_205561

/-- Given an arithmetic sequence {a_n} where a₃ + a₅ = 10, prove that a₄ = 5 -/
theorem arithmetic_sequence_fourth_term 
  (a : ℕ → ℝ) 
  (h_arith : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) 
  (h_sum : a 3 + a 5 = 10) : 
  a 4 = 5 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fourth_term_l2055_205561


namespace NUMINAMATH_CALUDE_newberg_airport_passengers_l2055_205566

theorem newberg_airport_passengers (on_time late : ℕ) 
  (h1 : on_time = 14507) 
  (h2 : late = 213) : 
  on_time + late = 14620 := by
sorry

end NUMINAMATH_CALUDE_newberg_airport_passengers_l2055_205566


namespace NUMINAMATH_CALUDE_craftsman_jars_l2055_205564

theorem craftsman_jars (jars clay_pots : ℕ) (h1 : jars = 2 * clay_pots)
  (h2 : 5 * jars + 3 * 5 * clay_pots = 200) : jars = 16 := by
  sorry

end NUMINAMATH_CALUDE_craftsman_jars_l2055_205564


namespace NUMINAMATH_CALUDE_constant_digit_sum_characterization_l2055_205510

/-- Sum of digits function -/
def S (n : ℕ) : ℕ := sorry

/-- Characterization of numbers with constant digit sum property -/
theorem constant_digit_sum_characterization (M : ℕ) :
  (M > 0 ∧ ∀ k : ℕ, 1 ≤ k ∧ k ≤ M → S (M * k) = S M) ↔
  (M = 1 ∨ ∃ n : ℕ, M = 10^n - 1) :=
sorry

end NUMINAMATH_CALUDE_constant_digit_sum_characterization_l2055_205510


namespace NUMINAMATH_CALUDE_father_son_ages_l2055_205598

/-- Proves that given the conditions about the ages of a father and son, their present ages are 36 and 12 years respectively. -/
theorem father_son_ages (father_age son_age : ℕ) : 
  father_age = 3 * son_age ∧ 
  father_age + 12 = 2 * (son_age + 12) →
  father_age = 36 ∧ son_age = 12 := by
  sorry

end NUMINAMATH_CALUDE_father_son_ages_l2055_205598


namespace NUMINAMATH_CALUDE_inverse_tan_product_range_l2055_205559

/-- Given an acute-angled triangle ABC where b^2 - a^2 = ac, 
    prove that 1 / (tan A * tan B) is in the open interval (0, 1) -/
theorem inverse_tan_product_range (A B C : ℝ) (a b c : ℝ) 
  (h_acute : A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = π)
  (h_triangle : a > 0 ∧ b > 0 ∧ c > 0)
  (h_sides : b^2 - a^2 = a*c) :
  0 < (1 : ℝ) / (Real.tan A * Real.tan B) ∧ (1 : ℝ) / (Real.tan A * Real.tan B) < 1 :=
sorry

end NUMINAMATH_CALUDE_inverse_tan_product_range_l2055_205559


namespace NUMINAMATH_CALUDE_xiao_dong_jump_record_l2055_205544

/-- Represents the recording of a long jump result -/
def record_jump (standard : ℝ) (jump : ℝ) : ℝ :=
  jump - standard

/-- The standard for the long jump -/
def long_jump_standard : ℝ := 4.00

/-- Xiao Dong's jump distance -/
def xiao_dong_jump : ℝ := 3.85

/-- Theorem stating how Xiao Dong's jump should be recorded -/
theorem xiao_dong_jump_record :
  record_jump long_jump_standard xiao_dong_jump = -0.15 := by
  sorry

end NUMINAMATH_CALUDE_xiao_dong_jump_record_l2055_205544


namespace NUMINAMATH_CALUDE_ceiling_floor_calculation_l2055_205508

theorem ceiling_floor_calculation : 
  ⌈(15 : ℝ) / 8 * (-45 : ℝ) / 4⌉ - ⌊(15 : ℝ) / 8 * ⌊(-45 : ℝ) / 4⌋⌋ = 2 := by sorry

end NUMINAMATH_CALUDE_ceiling_floor_calculation_l2055_205508


namespace NUMINAMATH_CALUDE_medal_award_count_l2055_205582

/-- The number of sprinters in the event -/
def total_sprinters : ℕ := 12

/-- The number of American sprinters -/
def american_sprinters : ℕ := 5

/-- The number of medals to be awarded -/
def medals : ℕ := 3

/-- The maximum number of Americans that can receive medals -/
def max_american_medalists : ℕ := 2

/-- The function that calculates the number of ways to award medals -/
def award_medals : ℕ := sorry

theorem medal_award_count : award_medals = 1260 := by sorry

end NUMINAMATH_CALUDE_medal_award_count_l2055_205582


namespace NUMINAMATH_CALUDE_odd_power_sum_divisible_l2055_205574

/-- A number is odd if it can be expressed as 2k + 1 for some integer k -/
def IsOdd (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k + 1

/-- A number is positive if it's greater than zero -/
def IsPositive (n : ℕ) : Prop := n > 0

theorem odd_power_sum_divisible (x y : ℤ) :
  ∀ n : ℕ, IsPositive n → IsOdd n →
  ∃ k : ℤ, x^n + y^n = (x + y) * k :=
sorry

end NUMINAMATH_CALUDE_odd_power_sum_divisible_l2055_205574


namespace NUMINAMATH_CALUDE_average_of_20_and_22_l2055_205557

theorem average_of_20_and_22 : (20 + 22) / 2 = 21 := by
  sorry

end NUMINAMATH_CALUDE_average_of_20_and_22_l2055_205557


namespace NUMINAMATH_CALUDE_johns_toy_store_spending_l2055_205522

/-- Proves that the fraction of John's remaining allowance spent at the toy store is 1/3 -/
theorem johns_toy_store_spending (
  total_allowance : ℚ)
  (arcade_fraction : ℚ)
  (candy_store_amount : ℚ)
  (h1 : total_allowance = 33/10)
  (h2 : arcade_fraction = 3/5)
  (h3 : candy_store_amount = 88/100) :
  let remaining_after_arcade := total_allowance - arcade_fraction * total_allowance
  let toy_store_amount := remaining_after_arcade - candy_store_amount
  toy_store_amount / remaining_after_arcade = 1/3 := by
sorry


end NUMINAMATH_CALUDE_johns_toy_store_spending_l2055_205522


namespace NUMINAMATH_CALUDE_expression_simplification_evaluation_at_one_l2055_205502

theorem expression_simplification (x : ℝ) 
  (h1 : x ≠ 3) (h2 : x ≠ -2) (h3 : x ≠ -1) :
  ((x^2 - 4) / (x^2 - x - 6) + (x + 2) / (x - 3)) / ((x + 1) / (x - 3)) = 2 * x / (x + 1) :=
by sorry

-- Evaluation at x = 1
theorem evaluation_at_one :
  ((1^2 - 4) / (1^2 - 1 - 6) + (1 + 2) / (1 - 3)) / ((1 + 1) / (1 - 3)) = 1 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_evaluation_at_one_l2055_205502


namespace NUMINAMATH_CALUDE_sin_225_degrees_l2055_205577

theorem sin_225_degrees : Real.sin (225 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_225_degrees_l2055_205577


namespace NUMINAMATH_CALUDE_smallest_undefined_fraction_value_l2055_205547

theorem smallest_undefined_fraction_value : ∃ x : ℚ, x = 2/9 ∧ 
  (∀ y : ℚ, y < x → 9*y^2 - 74*y + 8 ≠ 0) ∧ 9*x^2 - 74*x + 8 = 0 := by
  sorry

end NUMINAMATH_CALUDE_smallest_undefined_fraction_value_l2055_205547
