import Mathlib

namespace NUMINAMATH_CALUDE_count_solutions_l38_3808

/-- Sum of digits function -/
def S (n : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem count_solutions : 
  (Finset.filter (fun n => n + S n + S (S n) = 2007) (Finset.range 2008)).card = 4 := by sorry

end NUMINAMATH_CALUDE_count_solutions_l38_3808


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l38_3864

-- Define the sets A and B
def A : Set ℝ := {x | x ≥ 0}
def B : Set ℝ := {x | x^2 - 1 < 0}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 0 ≤ x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l38_3864


namespace NUMINAMATH_CALUDE_average_age_increase_l38_3843

theorem average_age_increase (num_students : ℕ) (student_avg_age : ℚ) (teacher_age : ℕ) : 
  num_students = 25 →
  student_avg_age = 26 →
  teacher_age = 52 →
  (student_avg_age * num_students + teacher_age) / (num_students + 1) - student_avg_age = 1 := by
  sorry

end NUMINAMATH_CALUDE_average_age_increase_l38_3843


namespace NUMINAMATH_CALUDE_tan_beta_value_l38_3887

theorem tan_beta_value (α β : Real) 
  (h1 : Real.tan α = 2) 
  (h2 : Real.tan (α + β) = -1) : 
  Real.tan β = 3 := by
sorry

end NUMINAMATH_CALUDE_tan_beta_value_l38_3887


namespace NUMINAMATH_CALUDE_complex_equation_solution_l38_3892

theorem complex_equation_solution (z : ℂ) : (Complex.I * z = 4 + 3 * Complex.I) → z = 3 - 4 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l38_3892


namespace NUMINAMATH_CALUDE_peppers_weight_l38_3882

theorem peppers_weight (total_weight green_weight : Float) 
  (h1 : total_weight = 0.6666666666666666)
  (h2 : green_weight = 0.3333333333333333) :
  total_weight - green_weight = 0.3333333333333333 := by
  sorry

end NUMINAMATH_CALUDE_peppers_weight_l38_3882


namespace NUMINAMATH_CALUDE_valid_triples_eq_solution_set_l38_3817

def is_valid_triple (a m n : ℕ) : Prop :=
  a ≥ 2 ∧ m ≥ 2 ∧ (a^n + 203) % (a^(m*n) + 1) = 0

def solution_set : Set (ℕ × ℕ × ℕ) :=
  {(a, m, n) | 
    (∃ k, (a = 2 ∧ m = 2 ∧ n = 4*k + 1) ∨
          (a = 2 ∧ m = 3 ∧ n = 6*k + 2) ∨
          (a = 2 ∧ m = 4 ∧ n = 8*k + 8) ∨
          (a = 2 ∧ m = 6 ∧ n = 12*k + 9) ∨
          (a = 3 ∧ m = 2 ∧ n = 4*k + 3) ∨
          (a = 4 ∧ m = 2 ∧ n = 4*k + 4) ∨
          (a = 5 ∧ m = 2 ∧ n = 4*k + 1) ∨
          (a = 8 ∧ m = 2 ∧ n = 4*k + 3) ∨
          (a = 10 ∧ m = 2 ∧ n = 4*k + 2)) ∨
    (a = 203 ∧ m ≥ 2 ∧ ∃ k, n = (2*k + 1)*m + 1)}

theorem valid_triples_eq_solution_set :
  {(a, m, n) : ℕ × ℕ × ℕ | is_valid_triple a m n} = solution_set :=
sorry

end NUMINAMATH_CALUDE_valid_triples_eq_solution_set_l38_3817


namespace NUMINAMATH_CALUDE_farmer_radishes_per_row_l38_3859

/-- Represents the farmer's planting scenario -/
structure FarmerPlanting where
  bean_seedlings : ℕ
  bean_per_row : ℕ
  pumpkin_seeds : ℕ
  pumpkin_per_row : ℕ
  total_radishes : ℕ
  rows_per_bed : ℕ
  total_beds : ℕ

/-- Calculates the number of radishes per row -/
def radishes_per_row (fp : FarmerPlanting) : ℕ :=
  let bean_rows := fp.bean_seedlings / fp.bean_per_row
  let pumpkin_rows := fp.pumpkin_seeds / fp.pumpkin_per_row
  let total_rows := fp.rows_per_bed * fp.total_beds
  let radish_rows := total_rows - (bean_rows + pumpkin_rows)
  fp.total_radishes / radish_rows

/-- Theorem stating that given the farmer's planting conditions, 
    the number of radishes per row is 6 -/
theorem farmer_radishes_per_row :
  let fp : FarmerPlanting := {
    bean_seedlings := 64,
    bean_per_row := 8,
    pumpkin_seeds := 84,
    pumpkin_per_row := 7,
    total_radishes := 48,
    rows_per_bed := 2,
    total_beds := 14
  }
  radishes_per_row fp = 6 := by
  sorry

end NUMINAMATH_CALUDE_farmer_radishes_per_row_l38_3859


namespace NUMINAMATH_CALUDE_parallelogram_division_slope_l38_3807

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parallelogram defined by four vertices -/
structure Parallelogram where
  v1 : Point
  v2 : Point
  v3 : Point
  v4 : Point

/-- Checks if a line with given slope passes through the origin and divides the parallelogram into two congruent polygons -/
def dividesParallelogramEqually (p : Parallelogram) (slope : ℚ) : Prop :=
  ∃ (a : ℝ),
    (p.v1.y + a) / p.v1.x = slope ∧
    (p.v3.y - a) / p.v3.x = slope ∧
    0 < a ∧ a < p.v2.y - p.v1.y

/-- The main theorem stating the slope of the line dividing the parallelogram equally -/
theorem parallelogram_division_slope :
  let p : Parallelogram := {
    v1 := { x := 12, y := 60 },
    v2 := { x := 12, y := 152 },
    v3 := { x := 32, y := 204 },
    v4 := { x := 32, y := 112 }
  }
  dividesParallelogramEqually p 16 := by sorry

end NUMINAMATH_CALUDE_parallelogram_division_slope_l38_3807


namespace NUMINAMATH_CALUDE_eating_contest_l38_3895

/-- Eating contest problem -/
theorem eating_contest (hot_dog_weight burger_weight pie_weight pizza_weight sandwich_weight : ℕ)
  (jacob_pies noah_burgers jacob_pizzas jacob_sandwiches mason_hotdogs mason_sandwiches : ℕ) :
  hot_dog_weight = 2 →
  burger_weight = 5 →
  pie_weight = 10 →
  pizza_weight = 15 →
  sandwich_weight = 3 →
  jacob_pies = noah_burgers - 3 →
  jacob_pizzas = jacob_sandwiches / 2 →
  mason_hotdogs = 3 * jacob_pies →
  mason_hotdogs = (3 : ℚ) / 2 * mason_sandwiches →
  noah_burgers = 8 →
  mason_hotdogs * hot_dog_weight = 30 := by sorry

end NUMINAMATH_CALUDE_eating_contest_l38_3895


namespace NUMINAMATH_CALUDE_whale_plankton_theorem_l38_3811

/-- Calculates the total amount of plankton consumed by a whale during a 5-hour feeding frenzy -/
def whale_plankton_consumption (x : ℕ) : ℕ :=
  let hour1 := x
  let hour2 := x + 3
  let hour3 := x + 6
  let hour4 := x + 9
  let hour5 := x + 12
  hour1 + hour2 + hour3 + hour4 + hour5

/-- Theorem stating the total plankton consumption given the problem conditions -/
theorem whale_plankton_theorem : 
  ∀ x : ℕ, (x + 6 = 93) → whale_plankton_consumption x = 465 :=
by
  sorry

#eval whale_plankton_consumption 87

end NUMINAMATH_CALUDE_whale_plankton_theorem_l38_3811


namespace NUMINAMATH_CALUDE_theater_tickets_sold_l38_3826

/-- Theorem: Total number of tickets sold in a theater -/
theorem theater_tickets_sold 
  (orchestra_price : ℕ) 
  (balcony_price : ℕ) 
  (total_cost : ℕ) 
  (extra_balcony : ℕ) : 
  orchestra_price = 12 →
  balcony_price = 8 →
  total_cost = 3320 →
  extra_balcony = 190 →
  ∃ (orchestra : ℕ) (balcony : ℕ),
    orchestra_price * orchestra + balcony_price * balcony = total_cost ∧
    balcony = orchestra + extra_balcony ∧
    orchestra + balcony = 370 := by
  sorry

end NUMINAMATH_CALUDE_theater_tickets_sold_l38_3826


namespace NUMINAMATH_CALUDE_range_of_m_l38_3836

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, x > 1 → 2*x + m + 2/(x-1) > 0) → m > -6 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l38_3836


namespace NUMINAMATH_CALUDE_geometric_sequence_a5_l38_3828

/-- A geometric sequence {a_n} where a_3 * a_7 = 64 -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_a5 (a : ℕ → ℝ) (h : geometric_sequence a) (h1 : a 3 * a 7 = 64) :
  a 5 = 8 ∨ a 5 = -8 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a5_l38_3828


namespace NUMINAMATH_CALUDE_cube_sum_is_42_l38_3858

/-- Represents a cube with numbers on its faces -/
structure NumberedCube where
  /-- The smallest number on the cube's faces -/
  smallest : ℕ
  /-- Proof that the smallest number is even -/
  smallest_even : Even smallest

/-- The sum of numbers on opposite faces of the cube -/
def opposite_face_sum (cube : NumberedCube) : ℕ :=
  2 * cube.smallest + 10

/-- The sum of all numbers on the cube's faces -/
def total_sum (cube : NumberedCube) : ℕ :=
  6 * cube.smallest + 30

/-- Theorem stating that the sum of numbers on a cube with the given properties is 42 -/
theorem cube_sum_is_42 (cube : NumberedCube) 
  (h : ∀ (i : Fin 3), opposite_face_sum cube = 2 * cube.smallest + 2 * i + 10) :
  total_sum cube = 42 := by
  sorry


end NUMINAMATH_CALUDE_cube_sum_is_42_l38_3858


namespace NUMINAMATH_CALUDE_fish_pond_flowers_l38_3877

/-- Calculates the number of flowers planted around a circular pond -/
def flowers_around_pond (perimeter : ℕ) (tree_spacing : ℕ) (flowers_between : ℕ) : ℕ :=
  (perimeter / tree_spacing) * flowers_between

/-- Theorem: The number of flowers planted around the fish pond is 39 -/
theorem fish_pond_flowers :
  flowers_around_pond 52 4 3 = 39 := by
  sorry

end NUMINAMATH_CALUDE_fish_pond_flowers_l38_3877


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l38_3872

theorem diophantine_equation_solutions :
  ∀ m n : ℤ, m^2 - 2*m*n - 3*n^2 = 5 ↔ 
    ((m = 4 ∧ n = 1) ∨ (m = 2 ∧ n = -1) ∨ (m = -4 ∧ n = -1) ∨ (m = -2 ∧ n = 1)) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l38_3872


namespace NUMINAMATH_CALUDE_garage_sale_pricing_l38_3876

theorem garage_sale_pricing (total_items : ℕ) (n : ℕ) 
  (h1 : total_items = 42)
  (h2 : n < total_items)
  (h3 : n = total_items - 24) : n = 19 := by
  sorry

end NUMINAMATH_CALUDE_garage_sale_pricing_l38_3876


namespace NUMINAMATH_CALUDE_cuboids_painted_l38_3885

theorem cuboids_painted (faces_per_cuboid : ℕ) (total_faces : ℕ) (h1 : faces_per_cuboid = 6) (h2 : total_faces = 36) :
  total_faces / faces_per_cuboid = 6 :=
by sorry

end NUMINAMATH_CALUDE_cuboids_painted_l38_3885


namespace NUMINAMATH_CALUDE_chrysler_building_floors_l38_3880

theorem chrysler_building_floors :
  ∀ (chrysler leeward : ℕ),
    chrysler = leeward + 11 →
    chrysler + leeward = 35 →
    chrysler = 23 :=
by
  sorry

end NUMINAMATH_CALUDE_chrysler_building_floors_l38_3880


namespace NUMINAMATH_CALUDE_triangle_area_is_40_5_l38_3812

/-- A line passing through two points -/
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

/-- A right triangle formed by a line intersecting the x and y axes -/
structure RightTriangle where
  line : Line

/-- Calculate the area of the right triangle -/
def area (triangle : RightTriangle) : ℝ :=
  sorry

/-- The theorem to be proved -/
theorem triangle_area_is_40_5 :
  let l : Line := { point1 := (-3, 6), point2 := (-6, 3) }
  let t : RightTriangle := { line := l }
  area t = 40.5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_is_40_5_l38_3812


namespace NUMINAMATH_CALUDE_inequality_theorem_l38_3841

theorem inequality_theorem (a b : ℝ) : 
  (∀ x : ℝ, x ≥ 0 → x^2 + 1 ≥ a*x + b ∧ a*x + b ≥ (3/2)*x^(2/3)) →
  ((2 - Real.sqrt 2) / 4 ≤ b ∧ b ≤ (2 + Real.sqrt 2) / 4) ∧
  (1 / Real.sqrt (2*b) ≤ a ∧ a ≤ 2 * Real.sqrt (1 - b)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_theorem_l38_3841


namespace NUMINAMATH_CALUDE_hynek_problem_bounds_l38_3847

/-- Represents a digit assignment for Hynek's problem -/
structure DigitAssignment where
  a : Fin 5
  b : Fin 5
  c : Fin 5
  d : Fin 5
  e : Fin 5
  distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e

/-- Calculates the sum for a given digit assignment -/
def calculateSum (assignment : DigitAssignment) : ℕ :=
  (assignment.a + 1) +
  11 * (assignment.b + 1) +
  111 * (assignment.c + 1) +
  1111 * (assignment.d + 1) +
  11111 * (assignment.e + 1)

/-- Checks if a number is divisible by 11 -/
def isDivisibleBy11 (n : ℕ) : Prop :=
  n % 11 = 0

/-- The main theorem stating the smallest and largest possible sums -/
theorem hynek_problem_bounds :
  (∃ (assignment : DigitAssignment),
    isDivisibleBy11 (calculateSum assignment) ∧
    (∀ (other : DigitAssignment),
      isDivisibleBy11 (calculateSum other) →
      calculateSum assignment ≤ calculateSum other)) ∧
  (∃ (assignment : DigitAssignment),
    isDivisibleBy11 (calculateSum assignment) ∧
    (∀ (other : DigitAssignment),
      isDivisibleBy11 (calculateSum other) →
      calculateSum other ≤ calculateSum assignment)) ∧
  (∀ (assignment : DigitAssignment),
    isDivisibleBy11 (calculateSum assignment) →
    23815 ≤ calculateSum assignment ∧ calculateSum assignment ≤ 60589) :=
sorry

end NUMINAMATH_CALUDE_hynek_problem_bounds_l38_3847


namespace NUMINAMATH_CALUDE_school_gender_ratio_l38_3838

theorem school_gender_ratio (boys girls : ℕ) : 
  boys * 13 = girls * 5 →  -- ratio of boys to girls is 5:13
  girls = boys + 80 →      -- there are 80 more girls than boys
  boys = 50 :=             -- prove that the number of boys is 50
by sorry

end NUMINAMATH_CALUDE_school_gender_ratio_l38_3838


namespace NUMINAMATH_CALUDE_largest_root_of_g_l38_3886

-- Define the function g(x)
def g (x : ℝ) : ℝ := 12 * x^4 - 17 * x^2 + 5

-- State the theorem
theorem largest_root_of_g :
  ∃ (r : ℝ), r = Real.sqrt 5 / 2 ∧
  g r = 0 ∧
  ∀ (x : ℝ), g x = 0 → x ≤ r :=
sorry

end NUMINAMATH_CALUDE_largest_root_of_g_l38_3886


namespace NUMINAMATH_CALUDE_exponent_base_problem_l38_3848

theorem exponent_base_problem (x : ℝ) (y : ℝ) :
  4^(2*x + 2) = y^(3*x - 1) → x = 1 → y = 16 := by
  sorry

end NUMINAMATH_CALUDE_exponent_base_problem_l38_3848


namespace NUMINAMATH_CALUDE_lagrange_four_square_theorem_l38_3851

-- Define the property of being expressible as the sum of four squares
def SumOfFourSquares (n : ℕ) : Prop :=
  ∃ a b c d : ℤ, n = a^2 + b^2 + c^2 + d^2

-- State Lagrange's Four Square Theorem
theorem lagrange_four_square_theorem :
  ∀ n : ℕ, SumOfFourSquares n :=
by
  sorry

-- State the given conditions
axiom odd_prime_four_squares :
  ∀ p : ℕ, Nat.Prime p → p % 2 = 1 → SumOfFourSquares p

axiom two_four_squares : SumOfFourSquares 2

axiom product_four_squares :
  ∀ a b : ℕ, SumOfFourSquares a → SumOfFourSquares b → SumOfFourSquares (a * b)

end NUMINAMATH_CALUDE_lagrange_four_square_theorem_l38_3851


namespace NUMINAMATH_CALUDE_triangle_isosceles_if_2cosB_sinA_eq_sinC_l38_3896

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  sum_angles : A + B + C = π

-- Define the property of being isosceles
def isIsosceles (t : Triangle) : Prop :=
  t.A = t.B ∨ t.B = t.C ∨ t.C = t.A

-- State the theorem
theorem triangle_isosceles_if_2cosB_sinA_eq_sinC (t : Triangle) :
  2 * Real.cos t.B * Real.sin t.A = Real.sin t.C → isIsosceles t :=
by
  sorry


end NUMINAMATH_CALUDE_triangle_isosceles_if_2cosB_sinA_eq_sinC_l38_3896


namespace NUMINAMATH_CALUDE_complex_number_point_l38_3829

theorem complex_number_point (z : ℂ) : z = Complex.I * (2 + Complex.I) → z.re = -1 ∧ z.im = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_point_l38_3829


namespace NUMINAMATH_CALUDE_parallel_line_m_value_l38_3894

/-- Given a line passing through points A(-2, m) and B(m, 4) that is parallel to the line 2x + y - 1 = 0, prove that m = -8 -/
theorem parallel_line_m_value :
  ∀ m : ℝ,
  let A : ℝ × ℝ := (-2, m)
  let B : ℝ × ℝ := (m, 4)
  let slope_AB := (B.2 - A.2) / (B.1 - A.1)
  let slope_given := -2  -- Slope of 2x + y - 1 = 0
  slope_AB = slope_given → m = -8 :=
by
  sorry

end NUMINAMATH_CALUDE_parallel_line_m_value_l38_3894


namespace NUMINAMATH_CALUDE_sqrt_product_plus_one_equals_1720_l38_3806

theorem sqrt_product_plus_one_equals_1720 : 
  Real.sqrt ((43 : ℝ) * 42 * 41 * 40 + 1) = 1720 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_plus_one_equals_1720_l38_3806


namespace NUMINAMATH_CALUDE_treadmill_time_difference_l38_3888

theorem treadmill_time_difference : 
  let total_distance : ℝ := 8
  let constant_speed : ℝ := 3
  let day1_speed : ℝ := 6
  let day2_speed : ℝ := 3
  let day3_speed : ℝ := 4
  let day4_speed : ℝ := 3
  let daily_distance : ℝ := 2
  let constant_time := total_distance / constant_speed
  let varied_time := daily_distance / day1_speed + daily_distance / day2_speed + 
                     daily_distance / day3_speed + daily_distance / day4_speed
  (constant_time - varied_time) * 60 = 80 := by sorry

end NUMINAMATH_CALUDE_treadmill_time_difference_l38_3888


namespace NUMINAMATH_CALUDE_reciprocal_sum_of_roots_l38_3883

theorem reciprocal_sum_of_roots (x₁ x₂ : ℝ) : 
  x₁^2 - 3*x₁ + 2 = 0 → 
  x₂^2 - 3*x₂ + 2 = 0 → 
  x₁ ≠ 0 → 
  x₂ ≠ 0 → 
  1/x₁ + 1/x₂ = 3/2 := by
sorry

end NUMINAMATH_CALUDE_reciprocal_sum_of_roots_l38_3883


namespace NUMINAMATH_CALUDE_specific_pyramid_sphere_radius_l38_3862

/-- Pyramid with equilateral triangular base -/
structure Pyramid :=
  (base_side : ℝ)
  (height : ℝ)

/-- The radius of the circumscribed sphere around the pyramid -/
def circumscribed_sphere_radius (p : Pyramid) : ℝ :=
  sorry

/-- Theorem: The radius of the circumscribed sphere for a specific pyramid -/
theorem specific_pyramid_sphere_radius :
  let p : Pyramid := { base_side := 6, height := 4 }
  circumscribed_sphere_radius p = 4 := by
  sorry

end NUMINAMATH_CALUDE_specific_pyramid_sphere_radius_l38_3862


namespace NUMINAMATH_CALUDE_andys_school_distance_l38_3825

/-- The distance between Andy's house and school, given the total distance walked and the distance to the market. -/
theorem andys_school_distance (total_distance : ℕ) (market_distance : ℕ) (h1 : total_distance = 140) (h2 : market_distance = 40) : 
  let school_distance := (total_distance - market_distance) / 2
  school_distance = 50 := by sorry

end NUMINAMATH_CALUDE_andys_school_distance_l38_3825


namespace NUMINAMATH_CALUDE_greatest_multiple_of_eight_remainder_l38_3889

/-- A function that checks if a natural number uses only unique digits from 1 to 9 -/
def uniqueDigits (n : ℕ) : Prop := sorry

/-- The greatest integer multiple of 8 using unique digits from 1 to 9 -/
noncomputable def M : ℕ := sorry

theorem greatest_multiple_of_eight_remainder :
  M % 1000 = 976 ∧ M % 8 = 0 ∧ uniqueDigits M ∧ ∀ k : ℕ, k > M → k % 8 = 0 → ¬(uniqueDigits k) := by
  sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_eight_remainder_l38_3889


namespace NUMINAMATH_CALUDE_floor_negative_seven_fourths_l38_3893

theorem floor_negative_seven_fourths : ⌊(-7 : ℚ) / 4⌋ = -2 := by
  sorry

end NUMINAMATH_CALUDE_floor_negative_seven_fourths_l38_3893


namespace NUMINAMATH_CALUDE_decimal_equivalences_l38_3803

-- Define the decimal number
def decimal_number : ℚ := 209 / 100

-- Theorem to prove the equivalence
theorem decimal_equivalences :
  -- Percentage equivalence
  (decimal_number * 100 : ℚ) = 209 ∧
  -- Simplified fraction equivalence
  decimal_number = 209 / 100 ∧
  -- Mixed number equivalence
  ∃ (whole : ℕ) (numerator : ℕ) (denominator : ℕ),
    whole = 2 ∧
    numerator = 9 ∧
    denominator = 100 ∧
    decimal_number = whole + (numerator : ℚ) / denominator :=
by
  sorry

end NUMINAMATH_CALUDE_decimal_equivalences_l38_3803


namespace NUMINAMATH_CALUDE_students_interested_in_both_l38_3868

theorem students_interested_in_both (total : ℕ) (music : ℕ) (sports : ℕ) (neither : ℕ) :
  total = 55 →
  music = 35 →
  sports = 45 →
  neither = 4 →
  ∃ both : ℕ, both = 29 ∧ total = music + sports - both + neither :=
by sorry

end NUMINAMATH_CALUDE_students_interested_in_both_l38_3868


namespace NUMINAMATH_CALUDE_nancy_crayon_packs_l38_3830

/-- The number of crayons Nancy bought -/
def total_crayons : ℕ := 615

/-- The number of crayons in each pack -/
def crayons_per_pack : ℕ := 15

/-- The number of packs Nancy bought -/
def number_of_packs : ℕ := total_crayons / crayons_per_pack

theorem nancy_crayon_packs : number_of_packs = 41 := by
  sorry

end NUMINAMATH_CALUDE_nancy_crayon_packs_l38_3830


namespace NUMINAMATH_CALUDE_even_increasing_function_inequality_l38_3850

-- Define an even function on ℝ
def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- Define an increasing function on [0, +∞)
def increasing_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → 0 ≤ y → x < y → f x < f y

-- Main theorem
theorem even_increasing_function_inequality
  (f : ℝ → ℝ)
  (h_even : even_function f)
  (h_incr : increasing_on_nonneg f) :
  ∀ k, f k > f 2 ↔ k > 2 ∨ k < -2 :=
sorry

end NUMINAMATH_CALUDE_even_increasing_function_inequality_l38_3850


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l38_3863

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n : ℕ, a n > 0) →
  a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25 →
  a 3 + a 5 = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l38_3863


namespace NUMINAMATH_CALUDE_planted_fraction_specific_case_l38_3832

/-- Represents a right triangle field with an unplanted area -/
structure FieldWithUnplantedArea where
  /-- Length of the first leg of the right triangle field -/
  leg1 : ℝ
  /-- Length of the second leg of the right triangle field -/
  leg2 : ℝ
  /-- Shortest distance from the base of the unplanted triangle to the hypotenuse -/
  unplanted_distance : ℝ

/-- Calculates the fraction of the planted area in the field -/
def planted_fraction (field : FieldWithUnplantedArea) : ℝ :=
  -- Implementation details omitted
  sorry

theorem planted_fraction_specific_case :
  let field := FieldWithUnplantedArea.mk 5 12 3
  planted_fraction field = 2665 / 2890 := by
  sorry

end NUMINAMATH_CALUDE_planted_fraction_specific_case_l38_3832


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l38_3852

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (seq.a 1 + seq.a n) / 2

theorem arithmetic_sequence_ratio (a b : ArithmeticSequence) :
  (∀ n : ℕ, sum_n a n / sum_n b n = (7 * n + 1 : ℚ) / (4 * n + 27)) →
  a.a 6 / b.a 6 = 78 / 71 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l38_3852


namespace NUMINAMATH_CALUDE_circle_symmetry_l38_3833

/-- Given a circle symmetrical to circle C with respect to the line x-y+1=0,
    prove that the equation of circle C is x^2 + (y-2)^2 = 1 -/
theorem circle_symmetry (x y : ℝ) :
  ((x - 1)^2 + (y - 1)^2 = 1) →  -- Equation of the symmetrical circle
  (x - y + 1 = 0 →               -- Equation of the line of symmetry
   (∃ (a b : ℝ), (x - a)^2 + (y - b)^2 = 1 ∧  -- Existence of circle C
    (a^2 + (b - 2)^2 = 1)))      -- Equation of circle C
:= by sorry

end NUMINAMATH_CALUDE_circle_symmetry_l38_3833


namespace NUMINAMATH_CALUDE_largest_int_less_100_rem_5_div_8_l38_3802

theorem largest_int_less_100_rem_5_div_8 : 
  ∃ (n : ℕ), n < 100 ∧ n % 8 = 5 ∧ ∀ (m : ℕ), m < 100 ∧ m % 8 = 5 → m ≤ n :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_largest_int_less_100_rem_5_div_8_l38_3802


namespace NUMINAMATH_CALUDE_triangle_problem_l38_3815

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  -- Condition: sin²A + sin A sin C + sin²C + cos²B = 1
  (Real.sin A)^2 + (Real.sin A) * (Real.sin C) + (Real.sin C)^2 + (Real.cos B)^2 = 1 →
  -- Condition: a = 5
  a = 5 →
  -- Condition: b = 7
  b = 7 →
  -- Prove: B = 2π/3
  B = 2 * Real.pi / 3 ∧
  -- Prove: sin C = 3√3/14
  Real.sin C = 3 * Real.sqrt 3 / 14 := by
  sorry


end NUMINAMATH_CALUDE_triangle_problem_l38_3815


namespace NUMINAMATH_CALUDE_triangle_arithmetic_sequence_properties_l38_3814

/-- Given a triangle with sides a > b > c forming an arithmetic sequence with difference d,
    and inscribed circle radius r, prove the following properties. -/
theorem triangle_arithmetic_sequence_properties
  (a b c d r : ℝ)
  (α γ : ℝ)
  (h1 : a > b)
  (h2 : b > c)
  (h3 : a - b = d)
  (h4 : b - c = d)
  (h5 : r > 0)
  (h6 : α > 0)
  (h7 : γ > 0) :
  (Real.tan (α / 2) * Real.tan (γ / 2) = 1 / 3) ∧
  (r = 2 * d / (3 * (Real.tan (α / 2) - Real.tan (γ / 2)))) :=
by sorry

end NUMINAMATH_CALUDE_triangle_arithmetic_sequence_properties_l38_3814


namespace NUMINAMATH_CALUDE_bus_speed_proof_l38_3804

/-- The speed of Bus A in miles per hour -/
def speed_A : ℝ := 45

/-- The speed of Bus B in miles per hour -/
def speed_B : ℝ := speed_A - 15

/-- The initial distance between Bus A and Bus B in miles -/
def initial_distance : ℝ := 150

/-- The time it takes for Bus A to overtake Bus B when driving in the same direction, in hours -/
def overtake_time : ℝ := 10

/-- The time it would take for the buses to meet if driving towards each other, in hours -/
def meet_time : ℝ := 2

theorem bus_speed_proof :
  (speed_A - speed_B) * overtake_time = initial_distance ∧
  (speed_A + speed_B) * meet_time = initial_distance ∧
  speed_A = 45 := by
  sorry

end NUMINAMATH_CALUDE_bus_speed_proof_l38_3804


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l38_3867

theorem pure_imaginary_condition (x : ℝ) : 
  (x^2 - x : ℂ) + (x - 1 : ℂ) * Complex.I = Complex.I * (y : ℝ) → x = 0 :=
by sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l38_3867


namespace NUMINAMATH_CALUDE_commission_increase_l38_3809

theorem commission_increase (total_sales : ℕ) (big_sale_commission : ℝ) (new_average : ℝ) :
  total_sales = 6 ∧ big_sale_commission = 1000 ∧ new_average = 250 →
  (new_average * total_sales - big_sale_commission) / (total_sales - 1) = 100 ∧
  new_average - (new_average * total_sales - big_sale_commission) / (total_sales - 1) = 150 := by
sorry

end NUMINAMATH_CALUDE_commission_increase_l38_3809


namespace NUMINAMATH_CALUDE_symmetric_circle_equation_l38_3827

/-- The equation of a circle symmetric to (x+2)^2 + y^2 = 5 with respect to y = x -/
theorem symmetric_circle_equation :
  let original_circle := (fun (x y : ℝ) => (x + 2)^2 + y^2 = 5)
  let symmetry_line := (fun (x y : ℝ) => y = x)
  let symmetric_circle := (fun (x y : ℝ) => x^2 + (y + 2)^2 = 5)
  ∀ x y : ℝ, symmetric_circle x y ↔ 
    ∃ x' y' : ℝ, original_circle x' y' ∧ 
    ((x + y = x' + y') ∧ (y - x = x' - y')) :=
by sorry


end NUMINAMATH_CALUDE_symmetric_circle_equation_l38_3827


namespace NUMINAMATH_CALUDE_clothing_sales_properties_l38_3849

/-- Represents the sales pattern of a new clothing item in July -/
structure ClothingSales where
  /-- The day number in July when maximum sales occurred -/
  max_day : ℕ
  /-- The maximum number of pieces sold in a day -/
  max_sales : ℕ
  /-- The number of days the clothing was popular -/
  popular_days : ℕ

/-- Calculates the sales for a given day in July -/
def daily_sales (day : ℕ) : ℕ :=
  if day ≤ 13 then 3 * day else 65 - 2 * day

/-- Calculates the cumulative sales up to a given day in July -/
def cumulative_sales (day : ℕ) : ℕ :=
  if day ≤ 13 
  then (3 + 3 * day) * day / 2
  else 273 + (51 - day) * (day - 13)

/-- Theorem stating the properties of the clothing sales in July -/
theorem clothing_sales_properties : ∃ (s : ClothingSales),
  s.max_day = 13 ∧ 
  s.max_sales = 39 ∧ 
  s.popular_days = 11 ∧
  daily_sales 1 = 3 ∧
  daily_sales 31 = 3 ∧
  (∀ d : ℕ, d < s.max_day → daily_sales (d + 1) = daily_sales d + 3) ∧
  (∀ d : ℕ, s.max_day < d ∧ d ≤ 31 → daily_sales d = daily_sales (d - 1) - 2) ∧
  (∃ d : ℕ, d ≥ 12 ∧ cumulative_sales d ≥ 200 ∧ cumulative_sales (d - 1) < 200) ∧
  (∃ d : ℕ, d ≤ 22 ∧ daily_sales d ≥ 20 ∧ daily_sales (d + 1) < 20) := by
  sorry

end NUMINAMATH_CALUDE_clothing_sales_properties_l38_3849


namespace NUMINAMATH_CALUDE_shooting_probability_l38_3820

/-- The probability of person A hitting the target -/
def prob_A : ℚ := 3/4

/-- The probability of person B hitting the target -/
def prob_B : ℚ := 4/5

/-- The probability of the event where A has shot twice when they stop -/
def prob_A_shoots_twice : ℚ := 19/400

theorem shooting_probability :
  let prob_A_miss := 1 - prob_A
  let prob_B_miss := 1 - prob_B
  prob_A_shoots_twice = 
    (prob_A_miss * prob_B_miss * prob_A) + 
    (prob_A_miss * prob_B_miss * prob_A_miss * prob_B) :=
by sorry

end NUMINAMATH_CALUDE_shooting_probability_l38_3820


namespace NUMINAMATH_CALUDE_expansion_properties_l38_3823

/-- Given an expression (3x - 1/(2*3x))^n where the ratio of the binomial coefficient 
    of the fifth term to that of the third term is 14:3, this theorem proves various 
    properties about the expansion. -/
theorem expansion_properties (n : ℕ) 
  (h : (Nat.choose n 4 : ℚ) / (Nat.choose n 2 : ℚ) = 14 / 3) :
  n = 10 ∧ 
  (let coeff_x2 := (Nat.choose 10 2 : ℚ) * (-1/2)^2 * 3^2;
   coeff_x2 = 45/4) ∧
  (let rational_terms := [
     (Nat.choose 10 2 : ℚ) * (-1/2)^2,
     (Nat.choose 10 5 : ℚ) * (-1/2)^5,
     (Nat.choose 10 8 : ℚ) * (-1/2)^8
   ];
   rational_terms.length = 3) :=
by sorry


end NUMINAMATH_CALUDE_expansion_properties_l38_3823


namespace NUMINAMATH_CALUDE_circle_equation_l38_3891

/-- A circle in the 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in the 2D plane represented by ax + by + c = 0 --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a point lies on a line --/
def Line.contains (l : Line) (p : ℝ × ℝ) : Prop :=
  l.a * p.1 + l.b * p.2 + l.c = 0

/-- Checks if a circle is tangent to a line at a given point --/
def Circle.tangentTo (c : Circle) (l : Line) (p : ℝ × ℝ) : Prop :=
  l.contains p ∧
  (c.center.1 - p.1) ^ 2 + (c.center.2 - p.2) ^ 2 = c.radius ^ 2 ∧
  (c.center.1 - p.1) * l.a + (c.center.2 - p.2) * l.b = 0

/-- The main theorem --/
theorem circle_equation (c : Circle) :
  (c.center.2 = -4 * c.center.1) →  -- Center lies on y = -4x
  (c.tangentTo (Line.mk 1 1 (-1)) (3, -2)) →  -- Tangent to x + y - 1 = 0 at (3, -2)
  (∀ x y : ℝ, (x - 1)^2 + (y + 4)^2 = 8 ↔ (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l38_3891


namespace NUMINAMATH_CALUDE_cos_pi_4_plus_alpha_l38_3898

theorem cos_pi_4_plus_alpha (α : ℝ) (h : Real.sin (π / 4 - α) = -2 / 5) :
  Real.cos (π / 4 + α) = -2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_cos_pi_4_plus_alpha_l38_3898


namespace NUMINAMATH_CALUDE_equation_solution_l38_3821

theorem equation_solution : 
  ∃ (x : ℚ), (x + 2) / 4 - 1 = (2 * x + 1) / 3 ∧ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l38_3821


namespace NUMINAMATH_CALUDE_workshop_assignment_l38_3860

theorem workshop_assignment (total_workers : ℕ) 
  (type_a_rate type_b_rate : ℕ) (ratio_a ratio_b : ℕ) 
  (type_a_workers : ℕ) (type_b_workers : ℕ) : 
  total_workers = 90 →
  type_a_rate = 15 →
  type_b_rate = 8 →
  ratio_a = 3 →
  ratio_b = 2 →
  type_a_workers = 40 →
  type_b_workers = 50 →
  total_workers = type_a_workers + type_b_workers →
  ratio_a * (type_b_rate * type_b_workers) = ratio_b * (type_a_rate * type_a_workers) := by
  sorry

#check workshop_assignment

end NUMINAMATH_CALUDE_workshop_assignment_l38_3860


namespace NUMINAMATH_CALUDE_subtraction_problem_l38_3837

theorem subtraction_problem : 240 - (35 * 4 + 6 * 3) = 82 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_problem_l38_3837


namespace NUMINAMATH_CALUDE_initial_men_correct_l38_3875

/-- Represents the initial number of men working on the project -/
def initial_men : ℕ := 27

/-- Represents the number of days to complete the project with the initial group -/
def initial_days : ℕ := 40

/-- Represents the number of days worked before some men leave -/
def days_before_leaving : ℕ := 18

/-- Represents the number of men who leave the project -/
def men_leaving : ℕ := 12

/-- Represents the number of days to complete the project after some men leave -/
def remaining_days : ℕ := 40

/-- Theorem stating that the initial number of men is correct given the conditions -/
theorem initial_men_correct :
  (initial_men : ℚ) * (days_before_leaving : ℚ) / initial_days +
  (initial_men - men_leaving : ℚ) * remaining_days / initial_days = 1 :=
sorry

end NUMINAMATH_CALUDE_initial_men_correct_l38_3875


namespace NUMINAMATH_CALUDE_x_range_theorem_l38_3890

-- Define the propositions p and q
def p (x : ℝ) : Prop := Real.log (x^2 - 2*x - 2) ≥ 0
def q (x : ℝ) : Prop := 0 < x ∧ x < 4

-- Define the range of x
def range_of_x (x : ℝ) : Prop := x ≤ -1 ∨ (0 < x ∧ x < 3) ∨ x ≥ 4

-- Theorem statement
theorem x_range_theorem (x : ℝ) : 
  (¬(p x) ∧ ¬(q x)) ∧ (p x ∨ q x) → range_of_x x :=
by sorry

end NUMINAMATH_CALUDE_x_range_theorem_l38_3890


namespace NUMINAMATH_CALUDE_goldfish_price_theorem_l38_3856

/-- Represents the selling price of a goldfish -/
def selling_price : ℝ := sorry

/-- Represents the cost price of a goldfish -/
def cost_price : ℝ := 0.25

/-- Represents the price of the new tank -/
def tank_price : ℝ := 100

/-- Represents the number of goldfish sold -/
def goldfish_sold : ℕ := 110

/-- Represents the percentage short of the tank price -/
def percentage_short : ℝ := 0.45

theorem goldfish_price_theorem :
  selling_price = 0.75 :=
by
  sorry

end NUMINAMATH_CALUDE_goldfish_price_theorem_l38_3856


namespace NUMINAMATH_CALUDE_sequence_sixth_term_l38_3834

theorem sequence_sixth_term (a : ℕ → ℝ) 
  (h1 : a 1 = 1)
  (h2 : a 2 = 2)
  (h3 : ∀ n ≥ 2, 2 * (a n)^2 = (a (n+1))^2 + (a (n-1))^2)
  (h4 : ∀ n, a n > 0) :
  a 6 = 4 := by
sorry

end NUMINAMATH_CALUDE_sequence_sixth_term_l38_3834


namespace NUMINAMATH_CALUDE_sum_of_e_and_f_l38_3835

theorem sum_of_e_and_f (a b c d e f : ℝ) 
  (h1 : (a + b) / 2 = 5.2)
  (h2 : (c + d) / 2 = 5.8)
  (h3 : (a + b + c + d + e + f) / 6 = 5.4) :
  e + f = 10.4 := by
sorry

end NUMINAMATH_CALUDE_sum_of_e_and_f_l38_3835


namespace NUMINAMATH_CALUDE_unripe_oranges_per_day_is_24_l38_3874

/-- The number of sacks of unripe oranges harvested per day -/
def unripe_oranges_per_day : ℕ := 24

/-- The total number of sacks of unripe oranges after the harvest period -/
def total_unripe_oranges : ℕ := 1080

/-- The number of days in the harvest period -/
def harvest_days : ℕ := 45

/-- Theorem stating that the number of sacks of unripe oranges harvested per day is 24 -/
theorem unripe_oranges_per_day_is_24 : 
  unripe_oranges_per_day = total_unripe_oranges / harvest_days :=
by sorry

end NUMINAMATH_CALUDE_unripe_oranges_per_day_is_24_l38_3874


namespace NUMINAMATH_CALUDE_restaurant_glasses_count_l38_3822

/-- Represents the number of glasses in a restaurant with two box sizes --/
def total_glasses (small_box_count : ℕ) (large_box_count : ℕ) : ℕ :=
  small_box_count * 12 + large_box_count * 16

/-- Represents the average number of glasses per box --/
def average_glasses_per_box (small_box_count : ℕ) (large_box_count : ℕ) : ℚ :=
  (total_glasses small_box_count large_box_count : ℚ) / (small_box_count + large_box_count : ℚ)

theorem restaurant_glasses_count :
  ∃ (small_box_count : ℕ) (large_box_count : ℕ),
    small_box_count > 0 ∧
    large_box_count = small_box_count + 16 ∧
    average_glasses_per_box small_box_count large_box_count = 15 ∧
    total_glasses small_box_count large_box_count = 480 := by
  sorry


end NUMINAMATH_CALUDE_restaurant_glasses_count_l38_3822


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_quadratic_inequality_empty_solution_l38_3819

def quadratic_inequality (k : ℝ) (x : ℝ) : Prop :=
  k * x^2 - 2 * x + 3 * k < 0

def solution_set_case1 (x : ℝ) : Prop :=
  x < -3 ∨ x > -1

def solution_set_case2 : Set ℝ :=
  ∅

theorem quadratic_inequality_solution (k : ℝ) :
  (∀ x, quadratic_inequality k x ↔ solution_set_case1 x) → k = -1/2 :=
sorry

theorem quadratic_inequality_empty_solution (k : ℝ) :
  (∀ x, ¬ quadratic_inequality k x) → 0 < k ∧ k ≤ Real.sqrt 3 / 3 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_quadratic_inequality_empty_solution_l38_3819


namespace NUMINAMATH_CALUDE_smallest_factor_l38_3800

theorem smallest_factor (n : ℕ) : n = 900 ↔ 
  (∀ m : ℕ, m > 0 → m < n → ¬(2^5 ∣ (936 * m) ∧ 3^3 ∣ (936 * m) ∧ 10^2 ∣ (936 * m))) ∧
  (2^5 ∣ (936 * n) ∧ 3^3 ∣ (936 * n) ∧ 10^2 ∣ (936 * n)) ∧
  (n > 0) :=
by sorry

end NUMINAMATH_CALUDE_smallest_factor_l38_3800


namespace NUMINAMATH_CALUDE_max_sum_with_constraint_l38_3845

theorem max_sum_with_constraint (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x^2 + y = 20) :
  x + y ≤ 81/4 :=
sorry

end NUMINAMATH_CALUDE_max_sum_with_constraint_l38_3845


namespace NUMINAMATH_CALUDE_box_maker_is_bellini_l38_3857

-- Define the possible makers of the box
inductive Maker
  | Bellini
  | Cellini
  | BelliniSon
  | CelliniSon

-- Define the inscription on the box
def inscription (maker : Maker) : Prop :=
  maker ≠ Maker.BelliniSon

-- Define the condition that the box was made by Bellini, Cellini, or one of their sons
def possibleMakers (maker : Maker) : Prop :=
  maker = Maker.Bellini ∨ maker = Maker.Cellini ∨ maker = Maker.BelliniSon ∨ maker = Maker.CelliniSon

-- Theorem: The maker of the box is Bellini
theorem box_maker_is_bellini :
  ∃ (maker : Maker), possibleMakers maker ∧ inscription maker → maker = Maker.Bellini :=
sorry

end NUMINAMATH_CALUDE_box_maker_is_bellini_l38_3857


namespace NUMINAMATH_CALUDE_car_speed_difference_l38_3805

theorem car_speed_difference (distance : ℝ) (speed_R : ℝ) : 
  distance = 750 ∧ 
  speed_R = 56.44102863722254 → 
  ∃ (speed_P : ℝ), 
    distance / speed_P = distance / speed_R - 2 ∧ 
    speed_P > speed_R ∧ 
    speed_P - speed_R = 10 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_difference_l38_3805


namespace NUMINAMATH_CALUDE_square_difference_inapplicable_l38_3842

/-- The square difference formula cannot be applied to (2x+3y)(-3y-2x) -/
theorem square_difference_inapplicable (x y : ℝ) :
  ¬ ∃ (a b : ℝ), (∃ (c₁ c₂ c₃ c₄ : ℝ), a = c₁ * x + c₂ * y ∧ b = c₃ * x + c₄ * y) ∧
    (2 * x + 3 * y) * (-3 * y - 2 * x) = (a + b) * (a - b) := by
  sorry

end NUMINAMATH_CALUDE_square_difference_inapplicable_l38_3842


namespace NUMINAMATH_CALUDE_physics_marks_calculation_l38_3854

def english_marks : ℕ := 76
def math_marks : ℕ := 65
def chemistry_marks : ℕ := 67
def biology_marks : ℕ := 85
def average_marks : ℕ := 75
def total_subjects : ℕ := 5

theorem physics_marks_calculation :
  let known_subjects_total : ℕ := english_marks + math_marks + chemistry_marks + biology_marks
  let total_marks : ℕ := average_marks * total_subjects
  total_marks - known_subjects_total = 82 := by sorry

end NUMINAMATH_CALUDE_physics_marks_calculation_l38_3854


namespace NUMINAMATH_CALUDE_soap_cost_theorem_l38_3871

-- Define the given conditions
def months_per_bar : ℕ := 2
def cost_per_bar : ℚ := 8
def discount_rate : ℚ := 0.1
def discount_threshold : ℕ := 6
def months_in_year : ℕ := 12

-- Define the function to calculate the cost of soap for a year
def soap_cost_for_year : ℚ :=
  let bars_needed := months_in_year / months_per_bar
  let total_cost := bars_needed * cost_per_bar
  let discount := if bars_needed ≥ discount_threshold then discount_rate * total_cost else 0
  total_cost - discount

-- Theorem statement
theorem soap_cost_theorem : soap_cost_for_year = 43.2 := by
  sorry


end NUMINAMATH_CALUDE_soap_cost_theorem_l38_3871


namespace NUMINAMATH_CALUDE_polynomial_equality_sum_l38_3855

theorem polynomial_equality_sum (a b c d : ℤ) : 
  (∀ x : ℝ, (x^2 + a*x + b) * (x^2 + c*x + d) = x^4 + 2*x^3 - x^2 + 18*x + 24) →
  a + b + c + d = 12 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_sum_l38_3855


namespace NUMINAMATH_CALUDE_characterization_of_n_l38_3878

/-- A bijection from {1, ..., n} to itself -/
def Bijection (n : ℕ) := { f : Fin n → Fin n // Function.Bijective f }

/-- The main theorem -/
theorem characterization_of_n (m : ℕ) (h_m : Even m) (h_m_pos : 0 < m) :
  ∀ n : ℕ, (∃ f : Bijection n,
    ∀ x y : Fin n, (m * x.val - y.val) % n = 0 →
      (n + 1) ∣ (f.val x).val^m - (f.val y).val) ↔
  Nat.Prime (n + 1) :=
sorry

end NUMINAMATH_CALUDE_characterization_of_n_l38_3878


namespace NUMINAMATH_CALUDE_circle_op_example_l38_3866

/-- Custom binary operation on real numbers -/
def circle_op (a b : ℝ) : ℝ := (a + b) * (a - b)

/-- The main theorem to prove -/
theorem circle_op_example : circle_op 9 (circle_op 4 3) = 32 := by
  sorry

end NUMINAMATH_CALUDE_circle_op_example_l38_3866


namespace NUMINAMATH_CALUDE_equal_even_odd_probability_l38_3899

/-- The number of dice being rolled -/
def num_dice : ℕ := 8

/-- The probability of a single die showing an even number -/
def prob_even : ℚ := 1/2

/-- The probability of a single die showing an odd number -/
def prob_odd : ℚ := 1/2

/-- The number of dice that need to show even (and odd) for the event to occur -/
def target_even : ℕ := num_dice / 2

-- The theorem statement
theorem equal_even_odd_probability : 
  (Nat.choose num_dice target_even : ℚ) * prob_even ^ num_dice = 35/128 := by
  sorry

end NUMINAMATH_CALUDE_equal_even_odd_probability_l38_3899


namespace NUMINAMATH_CALUDE_profit_growth_equation_l38_3881

theorem profit_growth_equation (x : ℝ) : 
  (250000 : ℝ) * (1 + x)^2 = 360000 → 25 * (1 + x)^2 = 36 := by
sorry

end NUMINAMATH_CALUDE_profit_growth_equation_l38_3881


namespace NUMINAMATH_CALUDE_stock_profit_is_447_5_l38_3801

/-- Calculate the profit from a stock transaction with given parameters -/
def calculate_profit (num_shares : ℕ) (buy_price sell_price : ℚ) 
  (stamp_duty_rate transfer_fee_rate commission_rate : ℚ) 
  (min_commission : ℚ) : ℚ :=
  let total_cost := num_shares * buy_price
  let total_income := num_shares * sell_price
  let total_transaction := total_cost + total_income
  let stamp_duty := total_transaction * stamp_duty_rate
  let transfer_fee := total_transaction * transfer_fee_rate
  let commission := max (total_transaction * commission_rate) min_commission
  total_income - total_cost - stamp_duty - transfer_fee - commission

/-- The profit from the given stock transaction is 447.5 yuan -/
theorem stock_profit_is_447_5 : 
  calculate_profit 1000 5 (11/2) (1/1000) (1/1000) (3/1000) 5 = 447.5 := by
  sorry

end NUMINAMATH_CALUDE_stock_profit_is_447_5_l38_3801


namespace NUMINAMATH_CALUDE_all_propositions_false_l38_3897

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel_lines : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (line_in_plane : Line → Plane → Prop)

-- Notation
local infix:50 " ∥ " => parallel_lines
local infix:50 " ∥ " => parallel_line_plane
local infix:50 " ⊂ " => line_in_plane

-- Theorem statement
theorem all_propositions_false :
  (∀ (a b : Line) (α : Plane), (a ∥ b) → (b ⊂ α) → (a ∥ α)) = False ∧
  (∀ (a b : Line) (α : Plane), (a ∥ α) → (b ∥ α) → (a ∥ b)) = False ∧
  (∀ (a b : Line) (α : Plane), (a ∥ b) → (b ∥ α) → (a ∥ α)) = False ∧
  (∀ (a b : Line) (α : Plane), (a ∥ α) → (b ⊂ α) → (a ∥ b)) = False :=
by sorry

end NUMINAMATH_CALUDE_all_propositions_false_l38_3897


namespace NUMINAMATH_CALUDE_probability_intersection_is_zero_l38_3879

def f (x : Nat) : Int :=
  6 * x - 4

def g (x : Nat) : Int :=
  2 * x - 1

def domain : Finset Nat :=
  {1, 2, 3, 4, 5, 6}

def A : Finset Int :=
  Finset.image f domain

def B : Finset Int :=
  Finset.image g domain

theorem probability_intersection_is_zero :
  (A ∩ B).card / (A ∪ B).card = 0 :=
sorry

end NUMINAMATH_CALUDE_probability_intersection_is_zero_l38_3879


namespace NUMINAMATH_CALUDE_bug_on_square_probability_l38_3884

/-- Probability of returning to the starting vertex after n moves -/
def P (n : ℕ) : ℚ :=
  match n with
  | 0 => 1
  | n + 1 => 2/3 - (1/3) * P n

/-- The problem statement -/
theorem bug_on_square_probability : P 8 = 3248/6561 := by
  sorry

end NUMINAMATH_CALUDE_bug_on_square_probability_l38_3884


namespace NUMINAMATH_CALUDE_solve_equation_l38_3824

theorem solve_equation (B : ℝ) : 
  80 - (5 - (6 + 2 * (B - 8 - 5))) = 89 ↔ B = 17 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l38_3824


namespace NUMINAMATH_CALUDE_three_digit_numbers_with_repeats_eq_252_l38_3865

/-- The number of three-digit numbers with repeated digits using digits 0 to 9 -/
def three_digit_numbers_with_repeats : ℕ :=
  let total_three_digit_numbers := 9 * 10 * 10  -- First digit can't be 0
  let three_digit_numbers_without_repeats := 9 * 9 * 8
  total_three_digit_numbers - three_digit_numbers_without_repeats

/-- Theorem stating that the number of three-digit numbers with repeated digits is 252 -/
theorem three_digit_numbers_with_repeats_eq_252 : 
  three_digit_numbers_with_repeats = 252 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_numbers_with_repeats_eq_252_l38_3865


namespace NUMINAMATH_CALUDE_largest_non_sum_of_composites_l38_3846

def IsComposite (n : ℕ) : Prop :=
  ∃ m : ℕ, m ≠ 1 ∧ m ≠ n ∧ n % m = 0

def IsSumOfTwoComposites (n : ℕ) : Prop :=
  ∃ a b : ℕ, IsComposite a ∧ IsComposite b ∧ n = a + b

theorem largest_non_sum_of_composites :
  (∀ n : ℕ, n > 11 → IsSumOfTwoComposites n) ∧
  ¬IsSumOfTwoComposites 11 :=
sorry

end NUMINAMATH_CALUDE_largest_non_sum_of_composites_l38_3846


namespace NUMINAMATH_CALUDE_cube_monotonically_increasing_l38_3844

/-- A function f: ℝ → ℝ is monotonically increasing if for all x₁ < x₂, f(x₁) ≤ f(x₂) -/
def MonotonicallyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ ≤ f x₂

/-- The cube function -/
def cube (x : ℝ) : ℝ := x^3

/-- The cube function is monotonically increasing on ℝ -/
theorem cube_monotonically_increasing : MonotonicallyIncreasing cube := by
  sorry


end NUMINAMATH_CALUDE_cube_monotonically_increasing_l38_3844


namespace NUMINAMATH_CALUDE_circle_equation_for_given_points_l38_3861

/-- Given two points P and Q in a 2D plane, this function returns the standard equation
    of the circle with diameter PQ as a function from ℝ × ℝ → Prop -/
def circle_equation (P Q : ℝ × ℝ) : (ℝ × ℝ → Prop) :=
  let (x₁, y₁) := P
  let (x₂, y₂) := Q
  let h := (x₁ + x₂) / 2
  let k := (y₁ + y₂) / 2
  let r := Real.sqrt (((x₂ - x₁)^2 + (y₂ - y₁)^2) / 4)
  fun (x, y) ↦ (x - h)^2 + (y - k)^2 = r^2

/-- Theorem stating that the standard equation of the circle with diameter PQ,
    where P(3,4) and Q(-5,6), is (x + 1)^2 + (y - 5)^2 = 17 -/
theorem circle_equation_for_given_points :
  circle_equation (3, 4) (-5, 6) = fun (x, y) ↦ (x + 1)^2 + (y - 5)^2 = 17 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_for_given_points_l38_3861


namespace NUMINAMATH_CALUDE_square_difference_l38_3816

theorem square_difference (x y : ℚ) 
  (h1 : x + y = 7/13) 
  (h2 : x - y = 1/91) : 
  x^2 - y^2 = 1/169 := by
sorry

end NUMINAMATH_CALUDE_square_difference_l38_3816


namespace NUMINAMATH_CALUDE_two_true_statements_l38_3853

theorem two_true_statements : 
  let original := ∀ a : ℝ, a > -5 → a > -8
  let converse := ∀ a : ℝ, a > -8 → a > -5
  let inverse := ∀ a : ℝ, a ≤ -5 → a ≤ -8
  let contrapositive := ∀ a : ℝ, a ≤ -8 → a ≤ -5
  (original ∧ ¬converse ∧ ¬inverse ∧ contrapositive) :=
by
  sorry

end NUMINAMATH_CALUDE_two_true_statements_l38_3853


namespace NUMINAMATH_CALUDE_count_l_shapes_l38_3818

/-- The number of ways to select an L-shaped piece from an m × n chessboard -/
def lShapeCount (m n : ℕ) : ℕ :=
  4 * (m - 1) * (n - 1)

/-- Theorem stating that the number of ways to select an L-shaped piece
    from an m × n chessboard is equal to 4(m-1)(n-1) -/
theorem count_l_shapes (m n : ℕ) (hm : m > 0) (hn : n > 0) :
  lShapeCount m n = 4 * (m - 1) * (n - 1) := by
  sorry

#check count_l_shapes

end NUMINAMATH_CALUDE_count_l_shapes_l38_3818


namespace NUMINAMATH_CALUDE_vectors_collinear_l38_3869

-- Define the vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define vectors a, b, m, and n
variable (a b m n : V)

-- State the theorem
theorem vectors_collinear (h1 : m = a + b) (h2 : n = 2 • a + 2 • b) (h3 : ¬ Collinear ℝ ({0, a, b} : Set V)) :
  Collinear ℝ ({0, m, n} : Set V) := by
  sorry

end NUMINAMATH_CALUDE_vectors_collinear_l38_3869


namespace NUMINAMATH_CALUDE_divide_by_fraction_twelve_divided_by_one_sixth_l38_3810

theorem divide_by_fraction (a b : ℚ) (hb : b ≠ 0) :
  a / b = a * (1 / b) := by sorry

theorem twelve_divided_by_one_sixth :
  12 / (1 / 6 : ℚ) = 72 := by sorry

end NUMINAMATH_CALUDE_divide_by_fraction_twelve_divided_by_one_sixth_l38_3810


namespace NUMINAMATH_CALUDE_division_property_l38_3813

theorem division_property (a b : ℕ+) :
  (∃ (q r : ℕ), a.val^2 + b.val^2 = q * (a.val + b.val) + r ∧
                0 ≤ r ∧ r < a.val + b.val ∧
                q^2 + r = 1977) →
  ((a.val = 50 ∧ b.val = 37) ∨
   (a.val = 50 ∧ b.val = 7) ∨
   (a.val = 37 ∧ b.val = 50) ∨
   (a.val = 7 ∧ b.val = 50)) :=
by sorry

end NUMINAMATH_CALUDE_division_property_l38_3813


namespace NUMINAMATH_CALUDE_min_distance_curve_to_line_l38_3873

/-- The minimum distance from a point on the curve y = x^2 - ln x to the line y = x - 2 is √2 --/
theorem min_distance_curve_to_line :
  let f (x : ℝ) := x^2 - Real.log x
  let g (x : ℝ) := x - 2
  ∀ x > 0, ∃ y : ℝ, y = f x ∧
    (∀ x' > 0, ∃ y' : ℝ, y' = f x' →
      Real.sqrt 2 ≤ |y' - g x'|) ∧
    |y - g x| = Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_distance_curve_to_line_l38_3873


namespace NUMINAMATH_CALUDE_equation_solution_l38_3839

theorem equation_solution : ∃ (x y z : ℝ), 
  2 * Real.sqrt (x - 4) + 3 * Real.sqrt (y - 9) + 4 * Real.sqrt (z - 16) = (1/2) * (x + y + z) ∧
  x = 8 ∧ y = 18 ∧ z = 32 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l38_3839


namespace NUMINAMATH_CALUDE_line_slope_point_sum_l38_3840

/-- Given a line with slope 5 passing through (5, 3), prove m + b^2 = 489 --/
theorem line_slope_point_sum (m b : ℝ) : 
  m = 5 →                   -- The slope is 5
  3 = 5 * 5 + b →           -- The line passes through (5, 3)
  m + b^2 = 489 :=          -- Prove that m + b^2 = 489
by sorry

end NUMINAMATH_CALUDE_line_slope_point_sum_l38_3840


namespace NUMINAMATH_CALUDE_average_difference_l38_3870

def average (a b c : ℕ) : ℚ :=
  (a + b + c : ℚ) / 3

theorem average_difference : average 20 40 60 - average 10 70 28 = 4 := by
  sorry

end NUMINAMATH_CALUDE_average_difference_l38_3870


namespace NUMINAMATH_CALUDE_fraction_equality_l38_3831

theorem fraction_equality : (18 : ℚ) / (5 * 107 + 3) = 18 / 538 := by sorry

end NUMINAMATH_CALUDE_fraction_equality_l38_3831
