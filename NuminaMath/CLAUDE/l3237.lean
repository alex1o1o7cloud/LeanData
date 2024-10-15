import Mathlib

namespace NUMINAMATH_CALUDE_unique_g_50_18_l3237_323735

def divisor_count (n : ℕ) : ℕ := (Nat.divisors n).card

def g₁ (n : ℕ) : ℕ := 3 * divisor_count n

def g (j n : ℕ) : ℕ :=
  match j with
  | 0 => n
  | j+1 => g₁ (g j n)

theorem unique_g_50_18 :
  ∃! (n : ℕ), n ≤ 25 ∧ g 50 n = 18 := by sorry

end NUMINAMATH_CALUDE_unique_g_50_18_l3237_323735


namespace NUMINAMATH_CALUDE_fraction_equality_solution_l3237_323736

theorem fraction_equality_solution : ∃! x : ℝ, (4 + x) / (6 + x) = (1 + x) / (2 + x) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_solution_l3237_323736


namespace NUMINAMATH_CALUDE_infinite_solutions_imply_a_equals_two_l3237_323723

/-- 
Given a system of equations:
  ax + y - 1 = 0
  4x + ay - 2 = 0
If there are infinitely many solutions, then a = 2.
-/
theorem infinite_solutions_imply_a_equals_two (a : ℝ) :
  (∀ x y : ℝ, a * x + y - 1 = 0 ∧ 4 * x + a * y - 2 = 0) →
  (∃ x₁ y₁ x₂ y₂ : ℝ, (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧
    a * x₁ + y₁ - 1 = 0 ∧ 4 * x₁ + a * y₁ - 2 = 0 ∧
    a * x₂ + y₂ - 1 = 0 ∧ 4 * x₂ + a * y₂ - 2 = 0) →
  a = 2 :=
by sorry


end NUMINAMATH_CALUDE_infinite_solutions_imply_a_equals_two_l3237_323723


namespace NUMINAMATH_CALUDE_willey_farm_problem_l3237_323786

/-- The Willey Farm Collective Problem -/
theorem willey_farm_problem (total_land : ℝ) (corn_cost : ℝ) (wheat_cost : ℝ) (available_capital : ℝ)
  (h1 : total_land = 4500)
  (h2 : corn_cost = 42)
  (h3 : wheat_cost = 35)
  (h4 : available_capital = 165200) :
  ∃ (wheat_acres : ℝ), wheat_acres = 3400 ∧
    wheat_acres ≥ 0 ∧
    wheat_acres ≤ total_land ∧
    ∃ (corn_acres : ℝ), corn_acres ≥ 0 ∧
      corn_acres + wheat_acres = total_land ∧
      corn_cost * corn_acres + wheat_cost * wheat_acres = available_capital :=
by sorry

end NUMINAMATH_CALUDE_willey_farm_problem_l3237_323786


namespace NUMINAMATH_CALUDE_grid_arithmetic_progression_l3237_323788

def is_arithmetic_progression (a b c : ℚ) : Prop :=
  b - a = c - b

theorem grid_arithmetic_progression :
  ∀ x : ℚ,
  let pos_3_4 := 2*x - 103
  let pos_1_4 := 251 - 2*x
  let pos_1_3 := 2/3*(51 - 2*x)
  let pos_3_3 := x
  (is_arithmetic_progression pos_1_3 pos_3_3 pos_3_4 ∧
   is_arithmetic_progression pos_1_4 pos_3_3 pos_3_4) →
  x = 60 := by
sorry

end NUMINAMATH_CALUDE_grid_arithmetic_progression_l3237_323788


namespace NUMINAMATH_CALUDE_area_of_parallelogram_EFGH_l3237_323763

/-- Represents a 2D vector -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Calculates the magnitude of the cross product of two 2D vectors -/
def crossProductMagnitude (v1 v2 : Vector2D) : ℝ :=
  |v1.x * v2.y - v1.y * v2.x|

/-- Theorem: Area of parallelogram EFGH -/
theorem area_of_parallelogram_EFGH : 
  let EF : Vector2D := ⟨3, 1⟩
  let EG : Vector2D := ⟨1, 5⟩
  crossProductMagnitude EF EG = 14 := by
  sorry

#check area_of_parallelogram_EFGH

end NUMINAMATH_CALUDE_area_of_parallelogram_EFGH_l3237_323763


namespace NUMINAMATH_CALUDE_ratio_of_segments_l3237_323756

/-- Given four points A, B, C, and D on a line in that order, with AB = 2, BC = 5, and AD = 14,
    prove that the ratio of AC to BD is 7/12. -/
theorem ratio_of_segments (A B C D : ℝ) : 
  (A < B) → (B < C) → (C < D) → 
  (B - A = 2) → (C - B = 5) → (D - A = 14) →
  (C - A) / (D - B) = 7 / 12 := by
sorry

end NUMINAMATH_CALUDE_ratio_of_segments_l3237_323756


namespace NUMINAMATH_CALUDE_cost_of_apple_l3237_323773

/-- The cost of fruit problem -/
theorem cost_of_apple (banana_cost orange_cost : ℚ)
  (apple_count banana_count orange_count : ℕ)
  (average_cost : ℚ)
  (h1 : banana_cost = 1)
  (h2 : orange_cost = 3)
  (h3 : apple_count = 12)
  (h4 : banana_count = 4)
  (h5 : orange_count = 4)
  (h6 : average_cost = 2)
  (h7 : average_cost * (apple_count + banana_count + orange_count : ℚ) =
        apple_cost * apple_count + banana_cost * banana_count + orange_cost * orange_count) :
  apple_cost = 2 :=
sorry

end NUMINAMATH_CALUDE_cost_of_apple_l3237_323773


namespace NUMINAMATH_CALUDE_inequality_proof_l3237_323707

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + 2) * (b^2 + 2) * (c^2 + 2) ≥ 9 * (a*b + b*c + c*a) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3237_323707


namespace NUMINAMATH_CALUDE_subset_relation_and_complement_l3237_323771

open Set

theorem subset_relation_and_complement (S A B : Set α) :
  (∀ x, x ∈ (S \ A) → x ∈ B) →
  (A ⊇ (S \ B) ∧ A ≠ (S \ B)) :=
by sorry

end NUMINAMATH_CALUDE_subset_relation_and_complement_l3237_323771


namespace NUMINAMATH_CALUDE_bowling_average_proof_l3237_323708

-- Define the initial bowling average
def initial_average : ℝ := 12

-- Define the number of new wickets taken
def new_wickets : ℕ := 5

-- Define the runs scored for the new wickets
def new_runs : ℕ := 26

-- Define the decrease in average
def average_decrease : ℝ := 0.4

-- Define the total number of wickets after taking the new wickets
def total_wickets : ℕ := 85

-- Theorem statement
theorem bowling_average_proof :
  (initial_average * (total_wickets - new_wickets) + new_runs) / total_wickets = initial_average - average_decrease :=
by sorry

end NUMINAMATH_CALUDE_bowling_average_proof_l3237_323708


namespace NUMINAMATH_CALUDE_power_function_symmetry_l3237_323759

/-- A function f is a power function if it can be written as f(x) = ax^n for some constant a and real number n. -/
def is_power_function (f : ℝ → ℝ) : Prop :=
  ∃ (a n : ℝ), ∀ x, f x = a * x^n

/-- A function f is symmetric about the y-axis if f(x) = f(-x) for all x. -/
def symmetric_about_y_axis (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

/-- Given that f(x) = (t^2 - t + 1)x^((t+3)/5) is a power function and 
    symmetric about the y-axis, prove that t = 1. -/
theorem power_function_symmetry (t : ℝ) : 
  let f := fun (x : ℝ) ↦ (t^2 - t + 1) * x^((t+3)/5)
  is_power_function f ∧ symmetric_about_y_axis f → t = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_function_symmetry_l3237_323759


namespace NUMINAMATH_CALUDE_trigonometric_properties_l3237_323716

open Real

-- Define the concept of terminal side of an angle
def sameSide (α β : ℝ) : Prop := sorry

-- Define the set of angles with terminal side on x-axis
def xAxisAngles : Set ℝ := { α | ∃ k : ℤ, α = k * π }

-- Define the quadrants
def inFirstOrSecondQuadrant (α : ℝ) : Prop := 0 < α ∧ α < π

theorem trigonometric_properties :
  (∀ α β : ℝ, sameSide α β → sin α = sin β ∧ cos α = cos β) ∧
  (xAxisAngles ≠ { α | ∃ k : ℤ, α = 2 * k * π }) ∧
  (∃ α : ℝ, sin α > 0 ∧ ¬inFirstOrSecondQuadrant α) ∧
  (∃ α β : ℝ, sin α = sin β ∧ ¬(∃ k : ℤ, α = 2 * k * π + β)) := by sorry

end NUMINAMATH_CALUDE_trigonometric_properties_l3237_323716


namespace NUMINAMATH_CALUDE_solve_transactions_problem_l3237_323741

def transactions_problem (mabel_monday : ℕ) : Prop :=
  let mabel_tuesday : ℕ := mabel_monday + mabel_monday / 10
  let anthony_tuesday : ℕ := 2 * mabel_tuesday
  let cal_tuesday : ℕ := (2 * anthony_tuesday + 2) / 3  -- Rounded up
  let jade_tuesday : ℕ := cal_tuesday + 17
  let isla_wednesday : ℕ := mabel_tuesday + cal_tuesday - 12
  let tim_thursday : ℕ := jade_tuesday + isla_wednesday + (jade_tuesday + isla_wednesday) / 2 + 1  -- Rounded up
  (mabel_monday = 100) → (tim_thursday = 614)

theorem solve_transactions_problem :
  transactions_problem 100 := by sorry

end NUMINAMATH_CALUDE_solve_transactions_problem_l3237_323741


namespace NUMINAMATH_CALUDE_real_part_of_z_l3237_323718

theorem real_part_of_z (z : ℂ) (h : z * (1 - Complex.I) = Complex.abs (1 - Complex.I * Real.sqrt 3) + Complex.I) :
  z.re = 1/2 := by sorry

end NUMINAMATH_CALUDE_real_part_of_z_l3237_323718


namespace NUMINAMATH_CALUDE_cube_planes_parallel_l3237_323774

/-- Represents a cube in 3D space -/
structure Cube where
  vertices : Fin 8 → ℝ × ℝ × ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  normal : ℝ × ℝ × ℝ
  point : ℝ × ℝ × ℝ

/-- Given two planes in a cube, determines if they are parallel -/
def are_planes_parallel (cube : Cube) (plane1 plane2 : Plane) : Prop :=
  -- The definition of parallel planes
  ∃ (k : ℝ), k ≠ 0 ∧ plane1.normal = k • plane2.normal

/-- Constructs the plane AB1D1 in the cube -/
def plane_AB1D1 (cube : Cube) : Plane :=
  -- Definition of plane AB1D1
  sorry

/-- Constructs the plane BC1D in the cube -/
def plane_BC1D (cube : Cube) : Plane :=
  -- Definition of plane BC1D
  sorry

/-- Theorem stating that in a cube, plane AB1D1 is parallel to plane BC1D -/
theorem cube_planes_parallel (cube : Cube) : 
  are_planes_parallel cube (plane_AB1D1 cube) (plane_BC1D cube) := by
  sorry

end NUMINAMATH_CALUDE_cube_planes_parallel_l3237_323774


namespace NUMINAMATH_CALUDE_solve_equation_l3237_323780

theorem solve_equation (x : ℚ) : 
  3 - 1 / (3 - 2 * x) = 2 / 3 * (1 / (3 - 2 * x)) → x = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3237_323780


namespace NUMINAMATH_CALUDE_average_after_removal_l3237_323755

theorem average_after_removal (numbers : Finset ℕ) (sum : ℕ) :
  Finset.card numbers = 15 →
  sum / 15 = 100 →
  sum = Finset.sum numbers id →
  80 ∈ numbers →
  90 ∈ numbers →
  95 ∈ numbers →
  (sum - 80 - 90 - 95) / (Finset.card numbers - 3) = 1235 / 12 :=
by sorry

end NUMINAMATH_CALUDE_average_after_removal_l3237_323755


namespace NUMINAMATH_CALUDE_company_male_employees_l3237_323789

theorem company_male_employees (m f : ℕ) : 
  m / f = 7 / 8 →
  (m + 3) / f = 8 / 9 →
  m = 189 := by
sorry

end NUMINAMATH_CALUDE_company_male_employees_l3237_323789


namespace NUMINAMATH_CALUDE_father_son_age_difference_l3237_323750

/-- Represents the ages of a father and son pair -/
structure FatherSonAges where
  father : ℕ
  son : ℕ

/-- The conditions of the problem -/
def satisfiesConditions (ages : FatherSonAges) : Prop :=
  ages.father = 44 ∧
  (ages.father + 4 = 2 * (ages.son + 4) + 20)

/-- The theorem to be proved -/
theorem father_son_age_difference (ages : FatherSonAges) 
  (h : satisfiesConditions ages) : 
  ages.father - 4 * ages.son = 4 := by
  sorry

end NUMINAMATH_CALUDE_father_son_age_difference_l3237_323750


namespace NUMINAMATH_CALUDE_axis_symmetry_implies_equal_coefficients_l3237_323710

theorem axis_symmetry_implies_equal_coefficients 
  (a b : ℝ) (h : a * b ≠ 0) :
  let f : ℝ → ℝ := λ x => a * Real.sin (2 * x) + b * Real.cos (2 * x)
  (∀ x, f (π/8 + x) = f (π/8 - x)) → a = b := by
  sorry

end NUMINAMATH_CALUDE_axis_symmetry_implies_equal_coefficients_l3237_323710


namespace NUMINAMATH_CALUDE_dihedral_angle_distance_l3237_323785

/-- Given a dihedral angle φ and a point A on one of its faces with distance a from the edge,
    the distance from A to the plane of the other face is a * sin(φ). -/
theorem dihedral_angle_distance (φ : ℝ) (a : ℝ) :
  let distance_to_edge := a
  let distance_to_other_face := a * Real.sin φ
  distance_to_other_face = distance_to_edge * Real.sin φ := by
  sorry

end NUMINAMATH_CALUDE_dihedral_angle_distance_l3237_323785


namespace NUMINAMATH_CALUDE_fourth_term_binomial_expansion_l3237_323761

theorem fourth_term_binomial_expansion 
  (a x : ℝ) (hx : x ≠ 0) :
  let binomial := (a / x^2 + x^2 / a)
  let fourth_term := Nat.choose 7 3 * (a / x^2)^(7 - 3) * (x^2 / a)^3
  fourth_term = 35 * a / x^2 := by
  sorry

end NUMINAMATH_CALUDE_fourth_term_binomial_expansion_l3237_323761


namespace NUMINAMATH_CALUDE_profit_maximization_and_threshold_l3237_323724

def price (x : ℕ) : ℕ :=
  if 1 ≤ x ∧ x < 40 then x + 45
  else if 40 ≤ x ∧ x ≤ 70 then 85
  else 0

def dailySales (x : ℕ) : ℕ := 150 - 2 * x

def costPrice : ℕ := 30

def dailyProfit (x : ℕ) : ℤ :=
  if 1 ≤ x ∧ x < 40 then -2 * x^2 + 120 * x + 2250
  else if 40 ≤ x ∧ x ≤ 70 then -110 * x + 8250
  else 0

theorem profit_maximization_and_threshold (x : ℕ) :
  (∀ x, 1 ≤ x ∧ x ≤ 70 → dailyProfit x ≤ dailyProfit 30) ∧
  dailyProfit 30 = 4050 ∧
  (Finset.filter (fun x => dailyProfit x ≥ 3250) (Finset.range 70)).card = 36 := by
  sorry


end NUMINAMATH_CALUDE_profit_maximization_and_threshold_l3237_323724


namespace NUMINAMATH_CALUDE_log_sum_equals_three_l3237_323746

-- Define the logarithm base 10 function
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_sum_equals_three : log10 5 + log10 2 + 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equals_three_l3237_323746


namespace NUMINAMATH_CALUDE_function_properties_l3237_323799

-- Define the function f
def f (x : ℝ) : ℝ := |x - 10| - |x - 25|

-- Define the theorem
theorem function_properties (a : ℝ) 
  (h : ∀ x, f x < 10 * a + 10) : 
  a > 1/2 ∧ ∃ (min_value : ℝ), min_value = 9 ∧ 
  ∀ a, a > 1/2 → 2 * a + 27 / (a^2) ≥ min_value :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l3237_323799


namespace NUMINAMATH_CALUDE_calculation_proof_l3237_323740

theorem calculation_proof : (0.0048 * 3.5) / (0.05 * 0.1 * 0.004) = 840 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l3237_323740


namespace NUMINAMATH_CALUDE_sales_tax_difference_l3237_323784

theorem sales_tax_difference (price : ℝ) (high_rate low_rate : ℝ) :
  price = 50 →
  high_rate = 0.0725 →
  low_rate = 0.0675 →
  price * high_rate - price * low_rate = 0.25 :=
by
  sorry

end NUMINAMATH_CALUDE_sales_tax_difference_l3237_323784


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l3237_323762

theorem decimal_to_fraction :
  (3.75 : ℚ) = 15 / 4 := by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l3237_323762


namespace NUMINAMATH_CALUDE_quadratic_equation_coefficients_l3237_323797

/-- Given a quadratic equation 3x² = -2x + 5, prove that it can be rewritten
    in the general form ax² + bx + c = 0 with specific coefficients. -/
theorem quadratic_equation_coefficients :
  ∃ (a b c : ℝ), 
    (∀ x, 3 * x^2 = -2 * x + 5) →
    (∀ x, a * x^2 + b * x + c = 0) ∧
    a = 3 ∧ b = 2 ∧ c = -5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_coefficients_l3237_323797


namespace NUMINAMATH_CALUDE_perimeter_of_square_C_l3237_323709

/-- Given squares A, B, and C with specified properties, prove that the perimeter of square C is 64 units -/
theorem perimeter_of_square_C (a b c : ℝ) : 
  (4 * a = 16) →  -- Perimeter of square A is 16 units
  (4 * b = 48) →  -- Perimeter of square B is 48 units
  (c = a + b) →   -- Side length of C is sum of side lengths of A and B
  (4 * c = 64) :=  -- Perimeter of square C is 64 units
by
  sorry


end NUMINAMATH_CALUDE_perimeter_of_square_C_l3237_323709


namespace NUMINAMATH_CALUDE_hexagon_division_existence_l3237_323795

/-- A hexagon is a polygon with six sides -/
structure Hexagon where
  vertices : Fin 6 → ℝ × ℝ

/-- A line is represented by two points -/
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

/-- A triangle is represented by three points -/
structure Triangle where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ
  point3 : ℝ × ℝ

/-- Predicate to check if two triangles are congruent -/
def areCongruentTriangles (t1 t2 : Triangle) : Prop := sorry

/-- Predicate to check if a line divides a hexagon into four congruent triangles -/
def dividesIntoFourCongruentTriangles (h : Hexagon) (l : Line) : Prop :=
  ∃ t1 t2 t3 t4 : Triangle,
    areCongruentTriangles t1 t2 ∧
    areCongruentTriangles t1 t3 ∧
    areCongruentTriangles t1 t4

/-- Theorem stating that there exists a hexagon that can be divided by a single line into four congruent triangles -/
theorem hexagon_division_existence :
  ∃ (h : Hexagon) (l : Line), dividesIntoFourCongruentTriangles h l := by sorry

end NUMINAMATH_CALUDE_hexagon_division_existence_l3237_323795


namespace NUMINAMATH_CALUDE_function_bounds_l3237_323743

-- Define the functions F and G
def F (a b c x : ℝ) : ℝ := a * x^2 + b * x + c
def G (a b c x : ℝ) : ℝ := c * x^2 + b * x + a

-- State the theorem
theorem function_bounds (a b c : ℝ) 
  (h1 : |F a b c 0| ≤ 1)
  (h2 : |F a b c 1| ≤ 1)
  (h3 : |F a b c (-1)| ≤ 1) :
  (∀ x : ℝ, |x| ≤ 1 → |F a b c x| ≤ 5/4) ∧
  (∀ x : ℝ, |x| ≤ 1 → |G a b c x| ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_function_bounds_l3237_323743


namespace NUMINAMATH_CALUDE_greatest_two_digit_with_digit_product_12_l3237_323714

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_product (n : ℕ) : ℕ :=
  (n / 10) * (n % 10)

theorem greatest_two_digit_with_digit_product_12 :
  ∀ n : ℕ, is_two_digit n → digit_product n = 12 → n ≤ 43 :=
sorry

end NUMINAMATH_CALUDE_greatest_two_digit_with_digit_product_12_l3237_323714


namespace NUMINAMATH_CALUDE_work_completion_time_l3237_323758

/-- The number of days it takes worker A to complete the work -/
def days_A : ℚ := 10

/-- The efficiency ratio of worker B compared to worker A -/
def efficiency_ratio : ℚ := 1.75

/-- The number of days it takes worker B to complete the work -/
def days_B : ℚ := 40 / 7

theorem work_completion_time :
  days_A * efficiency_ratio = days_B :=
sorry

end NUMINAMATH_CALUDE_work_completion_time_l3237_323758


namespace NUMINAMATH_CALUDE_inequality_theorem_l3237_323781

theorem inequality_theorem (a b : ℝ) (ha : a < 0) (hb : b < 0) :
  (b^2 / a + a^2 / b) ≤ (a + b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_theorem_l3237_323781


namespace NUMINAMATH_CALUDE_trigonometric_identity_l3237_323745

theorem trigonometric_identity (a b : ℝ) (θ : ℝ) (h : a > 0) (k : b > 0) 
  (eq : (Real.sin θ)^6 / a + (Real.cos θ)^6 / b = 1 / (a + b)) :
  (Real.sin θ)^12 / a^2 + (Real.cos θ)^12 / b^2 = (a^4 + b^4) / (a + b)^6 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l3237_323745


namespace NUMINAMATH_CALUDE_borrowed_amount_is_2500_l3237_323702

/-- Proves that the borrowed amount is 2500 given the problem conditions --/
theorem borrowed_amount_is_2500 
  (borrowed_rate : ℚ) 
  (lent_rate : ℚ) 
  (time : ℚ) 
  (yearly_gain : ℚ) 
  (h1 : borrowed_rate = 4 / 100)
  (h2 : lent_rate = 6 / 100)
  (h3 : time = 2)
  (h4 : yearly_gain = 100) : 
  ∃ (P : ℚ), P = 2500 ∧ 
    (lent_rate * P * time) - (borrowed_rate * P * time) = yearly_gain * time :=
by sorry

end NUMINAMATH_CALUDE_borrowed_amount_is_2500_l3237_323702


namespace NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_q_l3237_323787

-- Define the propositions p and q
def p (x : ℝ) : Prop := x^2 = 3*x + 4
def q (x : ℝ) : Prop := x = Real.sqrt (3*x + 4)

-- Theorem statement
theorem not_p_sufficient_not_necessary_for_not_q :
  (∀ x, ¬(p x) → ¬(q x)) ∧ 
  ¬(∀ x, ¬(q x) → ¬(p x)) :=
sorry

end NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_q_l3237_323787


namespace NUMINAMATH_CALUDE_car_speed_second_half_l3237_323733

/-- Calculates the speed of a car during the second half of a journey given the total distance,
    speed for the first half, and average speed for the entire journey. -/
theorem car_speed_second_half
  (total_distance : ℝ)
  (first_half_distance : ℝ)
  (first_half_speed : ℝ)
  (average_speed : ℝ)
  (h1 : total_distance = 320)
  (h2 : first_half_distance = 160)
  (h3 : first_half_speed = 90)
  (h4 : average_speed = 84.70588235294117)
  (h5 : first_half_distance * 2 = total_distance) :
  let second_half_speed := (total_distance / average_speed - first_half_distance / first_half_speed)⁻¹ * first_half_distance
  second_half_speed = 80 := by
sorry


end NUMINAMATH_CALUDE_car_speed_second_half_l3237_323733


namespace NUMINAMATH_CALUDE_bridget_apples_l3237_323776

theorem bridget_apples : ∃ (x : ℕ), 
  x > 0 ∧ 
  (2 * x) % 3 = 0 ∧ 
  (2 * x) / 3 - 5 = 2 ∧ 
  x = 11 := by
  sorry

end NUMINAMATH_CALUDE_bridget_apples_l3237_323776


namespace NUMINAMATH_CALUDE_johns_allowance_l3237_323726

theorem johns_allowance (A : ℚ) : 
  (A > 0) →
  ((4 / 15 : ℚ) * A = 92 / 100) →
  A = 345 / 100 := by
sorry

end NUMINAMATH_CALUDE_johns_allowance_l3237_323726


namespace NUMINAMATH_CALUDE_hangar_length_proof_l3237_323775

/-- The length of an airplane hangar given the number of planes it can fit and the length of each plane. -/
def hangar_length (num_planes : ℕ) (plane_length : ℕ) : ℕ :=
  num_planes * plane_length

/-- Theorem stating that a hangar fitting 7 planes of 40 feet each is 280 feet long. -/
theorem hangar_length_proof :
  hangar_length 7 40 = 280 := by
  sorry

end NUMINAMATH_CALUDE_hangar_length_proof_l3237_323775


namespace NUMINAMATH_CALUDE_inequality_condition_on_a_l3237_323778

theorem inequality_condition_on_a :
  ∀ a : ℝ, (∀ x : ℝ, (a - 3) * x^2 + 2 * (a - 3) * x - 4 < 0) ↔ a ∈ Set.Ioc (-1) 3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_condition_on_a_l3237_323778


namespace NUMINAMATH_CALUDE_floor_sqrt_23_squared_l3237_323738

theorem floor_sqrt_23_squared : ⌊Real.sqrt 23⌋^2 = 16 := by sorry

end NUMINAMATH_CALUDE_floor_sqrt_23_squared_l3237_323738


namespace NUMINAMATH_CALUDE_trapezoid_area_l3237_323747

-- Define the rectangle ABCD
def Rectangle (A B C D : Point) : Prop := sorry

-- Define the trapezoid EFBA
def Trapezoid (E F B A : Point) : Prop := sorry

-- Define the area function
def area (shape : Set Point) : ℝ := sorry

-- Define the points
variable (A B C D E F : Point)

-- State the theorem
theorem trapezoid_area 
  (h1 : Rectangle A B C D) 
  (h2 : area {A, B, C, D} = 20) 
  (h3 : Trapezoid E F B A) : 
  area {E, F, B, A} = 14 := by sorry

end NUMINAMATH_CALUDE_trapezoid_area_l3237_323747


namespace NUMINAMATH_CALUDE_g_of_neg_one_eq_neg_seven_l3237_323794

/-- Given a function g(x) = 5x - 2, prove that g(-1) = -7 -/
theorem g_of_neg_one_eq_neg_seven :
  let g : ℝ → ℝ := fun x ↦ 5 * x - 2
  g (-1) = -7 := by sorry

end NUMINAMATH_CALUDE_g_of_neg_one_eq_neg_seven_l3237_323794


namespace NUMINAMATH_CALUDE_two_true_propositions_l3237_323712

theorem two_true_propositions :
  let P1 := ∀ a b c : ℝ, a > b → a*c^2 > b*c^2
  let P2 := ∀ a b c : ℝ, a*c^2 > b*c^2 → a > b
  let P3 := ∀ a b c : ℝ, a ≤ b → a*c^2 ≤ b*c^2
  let P4 := ∀ a b c : ℝ, a*c^2 ≤ b*c^2 → a ≤ b
  (¬P1 ∧ P2 ∧ P3 ∧ ¬P4) ∨
  (¬P1 ∧ P2 ∧ ¬P3 ∧ P4) ∨
  (P1 ∧ ¬P2 ∧ P3 ∧ ¬P4) ∨
  (P1 ∧ ¬P2 ∧ ¬P3 ∧ P4) :=
by
  sorry

end NUMINAMATH_CALUDE_two_true_propositions_l3237_323712


namespace NUMINAMATH_CALUDE_b_equals_three_l3237_323704

/-- The function f(x) -/
def f (b : ℝ) (x : ℝ) : ℝ := x^3 - b*x^2 + 1

/-- f(x) is monotonically increasing in the interval (1, 2) -/
def monotone_increasing_in_interval (f : ℝ → ℝ) : Prop :=
  ∀ x y, 1 < x ∧ x < y ∧ y < 2 → f x < f y

/-- f(x) is monotonically decreasing in the interval (2, 3) -/
def monotone_decreasing_in_interval (f : ℝ → ℝ) : Prop :=
  ∀ x y, 2 < x ∧ x < y ∧ y < 3 → f x > f y

/-- Main theorem: b equals 3 -/
theorem b_equals_three :
  ∃ b : ℝ, 
    (monotone_increasing_in_interval (f b)) ∧ 
    (monotone_decreasing_in_interval (f b)) → 
    b = 3 := by sorry

end NUMINAMATH_CALUDE_b_equals_three_l3237_323704


namespace NUMINAMATH_CALUDE_range_of_a_l3237_323752

def proposition_p (a : ℝ) : Prop :=
  ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0

def proposition_q (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

theorem range_of_a : 
  ∀ a : ℝ, proposition_p a ∧ proposition_q a → a ≤ -2 ∨ a = 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3237_323752


namespace NUMINAMATH_CALUDE_definite_integral_proofs_l3237_323798

theorem definite_integral_proofs :
  (∫ x in (0:ℝ)..1, x^2 - x) = -1/6 ∧
  (∫ x in (1:ℝ)..3, |x - 2|) = 2 ∧
  (∫ x in (0:ℝ)..1, Real.sqrt (1 - x^2)) = π/4 := by
  sorry

end NUMINAMATH_CALUDE_definite_integral_proofs_l3237_323798


namespace NUMINAMATH_CALUDE_cheese_warehouse_problem_l3237_323728

theorem cheese_warehouse_problem (total_rats : ℕ) (cheese_first_night : ℕ) (rats_second_night : ℕ) :
  total_rats > rats_second_night →
  cheese_first_night = 10 →
  rats_second_night = 7 →
  (rats_second_night : ℚ) * (cheese_first_night : ℚ) / (2 * total_rats : ℚ) = 1 →
  cheese_first_night + 1 = 11 := by
  sorry

#check cheese_warehouse_problem

end NUMINAMATH_CALUDE_cheese_warehouse_problem_l3237_323728


namespace NUMINAMATH_CALUDE_circle_center_l3237_323729

/-- The center of a circle with equation x^2 + 4x + y^2 - 6y = 12 is (-2, 3) -/
theorem circle_center (x y : ℝ) : 
  (x^2 + 4*x + y^2 - 6*y = 12) → (x + 2)^2 + (y - 3)^2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_l3237_323729


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l3237_323706

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 222) 
  (h2 : a*b + b*c + a*c = 131) : 
  a + b + c = 22 := by
sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l3237_323706


namespace NUMINAMATH_CALUDE_jury_duty_days_is_25_l3237_323770

/-- Calculates the total number of days spent on jury duty -/
def totalJuryDutyDays (
  jurySelectionDays : ℕ)
  (trialMultiplier : ℕ)
  (trialDailyHours : ℕ)
  (deliberationDays : List ℕ)
  (deliberationDailyHours : ℕ) : ℕ :=
  let trialDays := jurySelectionDays * trialMultiplier
  let totalDeliberationHours := deliberationDays.sum * (deliberationDailyHours - 2)
  let totalDeliberationDays := (totalDeliberationHours + deliberationDailyHours - 1) / deliberationDailyHours
  jurySelectionDays + trialDays + totalDeliberationDays

/-- Theorem stating that the total jury duty days is 25 -/
theorem jury_duty_days_is_25 :
  totalJuryDutyDays 2 4 9 [6, 4, 5] 14 = 25 := by
  sorry

#eval totalJuryDutyDays 2 4 9 [6, 4, 5] 14

end NUMINAMATH_CALUDE_jury_duty_days_is_25_l3237_323770


namespace NUMINAMATH_CALUDE_prob_three_odd_dice_l3237_323731

/-- The number of sides on each die -/
def numSides : ℕ := 8

/-- The number of dice rolled -/
def numDice : ℕ := 5

/-- The number of odd numbers on each die -/
def numOddSides : ℕ := 4

/-- The number of dice that should show an odd number -/
def targetOddDice : ℕ := 3

/-- The probability of rolling exactly three odd numbers when five 8-sided dice are rolled -/
theorem prob_three_odd_dice : 
  (numOddSides / numSides) ^ targetOddDice * 
  ((numSides - numOddSides) / numSides) ^ (numDice - targetOddDice) * 
  (Nat.choose numDice targetOddDice) = 5 / 16 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_odd_dice_l3237_323731


namespace NUMINAMATH_CALUDE_candidate_vote_percentage_l3237_323713

theorem candidate_vote_percentage
  (total_votes : ℕ)
  (loss_margin : ℕ)
  (h_total : total_votes = 10000)
  (h_margin : loss_margin = 4000) :
  (total_votes - loss_margin) * 2 * 100 / total_votes = 30 := by
  sorry

end NUMINAMATH_CALUDE_candidate_vote_percentage_l3237_323713


namespace NUMINAMATH_CALUDE_jackpot_probability_6_45_100_l3237_323772

/-- Represents the lottery "6 out of 45" -/
structure Lottery :=
  (total_numbers : Nat)
  (numbers_to_choose : Nat)

/-- Represents a player's bet in the lottery -/
structure Bet :=
  (number_of_tickets : Nat)

/-- Calculate the number of combinations for choosing k items from n items -/
def choose (n k : Nat) : Nat :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

/-- Calculate the probability of hitting the jackpot -/
def jackpot_probability (l : Lottery) (b : Bet) : ℚ :=
  b.number_of_tickets / (choose l.total_numbers l.numbers_to_choose)

/-- Theorem: The probability of hitting the jackpot in a "6 out of 45" lottery with 100 unique tickets -/
theorem jackpot_probability_6_45_100 :
  let l : Lottery := ⟨45, 6⟩
  let b : Bet := ⟨100⟩
  jackpot_probability l b = 100 / 8145060 := by sorry

end NUMINAMATH_CALUDE_jackpot_probability_6_45_100_l3237_323772


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3237_323796

theorem complex_equation_solution (z : ℂ) : z * Complex.I = 3 - Complex.I → z = -1 - 3 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3237_323796


namespace NUMINAMATH_CALUDE_trajectory_and_minimum_distance_l3237_323730

-- Define the points M and N
def M : ℝ × ℝ := (4, 0)
def N : ℝ × ℝ := (1, 0)

-- Define the line l
def l (x y : ℝ) : ℝ := x + 2*y - 12

-- Define the condition for point P
def P_condition (x y : ℝ) : Prop :=
  let MP := (x - M.1, y - M.2)
  let MN := (N.1 - M.1, N.2 - M.2)
  let NP := (x - N.1, y - N.2)
  MN.1 * MP.1 + MN.2 * MP.2 = 6 * Real.sqrt (NP.1^2 + NP.2^2)

-- State the theorem
theorem trajectory_and_minimum_distance :
  ∃ (Q : ℝ × ℝ),
    (∀ (x y : ℝ), P_condition x y ↔ x^2/4 + y^2/3 = 1) ∧
    Q = (1, 3/2) ∧
    (∀ (P : ℝ × ℝ), P_condition P.1 P.2 →
      |l P.1 P.2| / Real.sqrt 5 ≥ 8/5) ∧
    |l Q.1 Q.2| / Real.sqrt 5 = 8/5 :=
  sorry

end NUMINAMATH_CALUDE_trajectory_and_minimum_distance_l3237_323730


namespace NUMINAMATH_CALUDE_two_times_greater_l3237_323732

theorem two_times_greater (a b : ℚ) (h : a > b) : 2 * a > 2 * b := by
  sorry

end NUMINAMATH_CALUDE_two_times_greater_l3237_323732


namespace NUMINAMATH_CALUDE_shoe_box_problem_l3237_323760

theorem shoe_box_problem (n : ℕ) (pairs : ℕ) (prob : ℚ) : 
  pairs = 7 →
  prob = 1 / 13 →
  prob = (pairs : ℚ) / (n.choose 2) →
  n = 14 :=
by sorry

end NUMINAMATH_CALUDE_shoe_box_problem_l3237_323760


namespace NUMINAMATH_CALUDE_sarah_brother_books_l3237_323783

/-- The number of books Sarah's brother bought -/
def brothers_books (sarah_paperbacks sarah_hardbacks : ℕ) : ℕ :=
  (sarah_paperbacks / 3) + (sarah_hardbacks * 2)

/-- Theorem: Sarah's brother bought 10 books in total -/
theorem sarah_brother_books :
  brothers_books 6 4 = 10 := by
  sorry

end NUMINAMATH_CALUDE_sarah_brother_books_l3237_323783


namespace NUMINAMATH_CALUDE_AB_vector_l3237_323793

def OA : ℝ × ℝ := (1, -2)
def OB : ℝ × ℝ := (-3, 1)

theorem AB_vector : (OB.1 - OA.1, OB.2 - OA.2) = (-4, 3) := by
  sorry

end NUMINAMATH_CALUDE_AB_vector_l3237_323793


namespace NUMINAMATH_CALUDE_exists_parallel_planes_nonparallel_lines_perpendicular_line_implies_perpendicular_planes_l3237_323705

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (subset : Line → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)

-- Statement 1
theorem exists_parallel_planes_nonparallel_lines :
  ∃ (α β : Plane) (l m : Line),
    subset l α ∧ subset m β ∧ parallel_planes α β ∧ ¬parallel_lines l m :=
sorry

-- Statement 2
theorem perpendicular_line_implies_perpendicular_planes
  (α β : Plane) (l : Line) :
  subset l α → perpendicular_line_plane l β → perpendicular_planes α β :=
sorry

end NUMINAMATH_CALUDE_exists_parallel_planes_nonparallel_lines_perpendicular_line_implies_perpendicular_planes_l3237_323705


namespace NUMINAMATH_CALUDE_inverse_proportion_l3237_323753

/-- Given that x is inversely proportional to y, prove that if x = 5 when y = 15, 
    then x = 5/3 when y = 45 -/
theorem inverse_proportion (x y : ℝ) (k : ℝ) (h1 : x * y = k) 
    (h2 : 5 * 15 = k) : 
  5 / 3 * 45 = k := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_l3237_323753


namespace NUMINAMATH_CALUDE_savings_theorem_l3237_323725

def savings_problem (monday_savings : ℕ) (tuesday_savings : ℕ) (wednesday_savings : ℕ) : ℕ :=
  let total_savings := monday_savings + tuesday_savings + wednesday_savings
  total_savings / 2

theorem savings_theorem (monday_savings tuesday_savings wednesday_savings : ℕ) :
  monday_savings = 15 →
  tuesday_savings = 28 →
  wednesday_savings = 13 →
  savings_problem monday_savings tuesday_savings wednesday_savings = 28 := by
  sorry

end NUMINAMATH_CALUDE_savings_theorem_l3237_323725


namespace NUMINAMATH_CALUDE_equation_solution_l3237_323792

theorem equation_solution : ∃! y : ℚ, 7 * (4 * y + 5) - 4 = -3 * (2 - 9 * y) := by
  use (-37 : ℚ)
  constructor
  · -- Prove that y = -37 satisfies the equation
    sorry
  · -- Prove uniqueness
    sorry

#check equation_solution

end NUMINAMATH_CALUDE_equation_solution_l3237_323792


namespace NUMINAMATH_CALUDE_rth_term_of_arithmetic_progression_l3237_323757

def sum_of_n_terms (n : ℕ) : ℕ := 2*n + 3*n^2 + n^3

theorem rth_term_of_arithmetic_progression (r : ℕ) :
  sum_of_n_terms r - sum_of_n_terms (r - 1) = 3*r^2 + 5*r - 2 :=
by sorry

end NUMINAMATH_CALUDE_rth_term_of_arithmetic_progression_l3237_323757


namespace NUMINAMATH_CALUDE_crayon_eraser_difference_l3237_323719

def prove_crayon_eraser_difference 
  (initial_erasers : ℕ) 
  (initial_crayons : ℕ) 
  (remaining_crayons : ℕ) : Prop :=
  initial_erasers = 457 ∧ 
  initial_crayons = 617 ∧ 
  remaining_crayons = 523 → 
  remaining_crayons - initial_erasers = 66

theorem crayon_eraser_difference : 
  prove_crayon_eraser_difference 457 617 523 :=
by sorry

end NUMINAMATH_CALUDE_crayon_eraser_difference_l3237_323719


namespace NUMINAMATH_CALUDE_positive_multiple_of_seven_find_x_l3237_323764

theorem positive_multiple_of_seven (x : ℕ) : Prop :=
  ∃ k : ℕ, x = 7 * k ∧ k > 0

theorem find_x : ∃ x : ℕ, 
  positive_multiple_of_seven x ∧ 
  x^2 > 50 ∧ 
  x < 30 ∧
  (x = 14 ∨ x = 21 ∨ x = 28) :=
by sorry

end NUMINAMATH_CALUDE_positive_multiple_of_seven_find_x_l3237_323764


namespace NUMINAMATH_CALUDE_rachel_total_problems_l3237_323700

/-- The number of math problems Rachel solved in total -/
def total_problems (problems_per_minute : ℕ) (minutes_solved : ℕ) (problems_next_day : ℕ) : ℕ :=
  problems_per_minute * minutes_solved + problems_next_day

/-- Proof that Rachel solved 76 math problems in total -/
theorem rachel_total_problems :
  total_problems 5 12 16 = 76 := by
  sorry

end NUMINAMATH_CALUDE_rachel_total_problems_l3237_323700


namespace NUMINAMATH_CALUDE_rancher_lasso_probability_l3237_323766

/-- The probability of a rancher placing a lasso around a cow's neck in a single throw. -/
def single_throw_probability : ℚ := 1 / 2

/-- The number of attempts the rancher makes. -/
def number_of_attempts : ℕ := 3

/-- The probability of the rancher placing a lasso around a cow's neck at least once in the given number of attempts. -/
def success_probability : ℚ := 7 / 8

theorem rancher_lasso_probability :
  (1 : ℚ) - (1 - single_throw_probability) ^ number_of_attempts = success_probability :=
sorry

end NUMINAMATH_CALUDE_rancher_lasso_probability_l3237_323766


namespace NUMINAMATH_CALUDE_unique_solution_l3237_323703

-- Define the system of equations
def system (x y z w : ℝ) : Prop :=
  (x + 1 = z + w + z*w*x) ∧
  (y - 1 = w + x + w*x*y) ∧
  (z + 2 = x + y + x*y*z) ∧
  (w - 2 = y + z + y*z*w)

-- Theorem statement
theorem unique_solution : ∃! (x y z w : ℝ), system x y z w :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l3237_323703


namespace NUMINAMATH_CALUDE_math_class_size_l3237_323721

/-- Proves that the number of students in the mathematics class is 170/3 given the conditions of the problem. -/
theorem math_class_size (total : ℕ) (both : ℕ) (math_twice_physics : Prop) :
  total = 75 →
  both = 10 →
  math_twice_physics →
  (∃ (math physics : ℕ),
    math = (170 : ℚ) / 3 ∧
    physics = (total - both) - (math - both) ∧
    math = 2 * physics) :=
by sorry

end NUMINAMATH_CALUDE_math_class_size_l3237_323721


namespace NUMINAMATH_CALUDE_small_circle_radius_l3237_323715

/-- Given a large circle with radius 10 meters and four congruent smaller circles
    touching at its center, prove that the radius of each smaller circle is 5 meters. -/
theorem small_circle_radius (R : ℝ) (r : ℝ) : R = 10 → 2 * r = R → r = 5 := by sorry

end NUMINAMATH_CALUDE_small_circle_radius_l3237_323715


namespace NUMINAMATH_CALUDE_pedro_squares_difference_l3237_323765

theorem pedro_squares_difference (jesus_squares linden_squares pedro_squares : ℕ) 
  (h1 : jesus_squares = 60)
  (h2 : linden_squares = 75)
  (h3 : pedro_squares = 200) :
  pedro_squares - (jesus_squares + linden_squares) = 65 := by
  sorry

end NUMINAMATH_CALUDE_pedro_squares_difference_l3237_323765


namespace NUMINAMATH_CALUDE_janelle_has_72_marbles_l3237_323720

/-- The number of marbles Janelle has after buying and gifting some marbles -/
def janelles_marbles : ℕ :=
  let initial_green := 26
  let blue_bags := 6
  let marbles_per_bag := 10
  let gifted_green := 6
  let gifted_blue := 8
  
  let total_blue := blue_bags * marbles_per_bag
  let total_before_gift := initial_green + total_blue
  let total_gifted := gifted_green + gifted_blue
  
  total_before_gift - total_gifted

/-- Theorem stating that Janelle has 72 marbles after the transactions -/
theorem janelle_has_72_marbles : janelles_marbles = 72 := by
  sorry

end NUMINAMATH_CALUDE_janelle_has_72_marbles_l3237_323720


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l3237_323748

theorem complex_modulus_problem (a : ℝ) (h1 : a > 0) (h2 : Complex.abs (a + Complex.I) = 2) :
  a = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l3237_323748


namespace NUMINAMATH_CALUDE_three_consecutive_heads_probability_l3237_323734

theorem three_consecutive_heads_probability (p : ℝ) :
  p = (1 : ℝ) / 2 →  -- probability of heads on a single flip
  p * p * p = (1 : ℝ) / 8 :=  -- probability of three consecutive heads
by
  sorry

end NUMINAMATH_CALUDE_three_consecutive_heads_probability_l3237_323734


namespace NUMINAMATH_CALUDE_rectangle_equal_diagonals_converse_is_false_contrapositive_is_true_l3237_323790

-- Define a quadrilateral
structure Quadrilateral :=
  (vertices : Fin 4 → ℝ × ℝ)

-- Define a rectangle
def is_rectangle (q : Quadrilateral) : Prop := sorry

-- Define equal diagonals
def has_equal_diagonals (q : Quadrilateral) : Prop := sorry

-- Theorem for the original proposition
theorem rectangle_equal_diagonals (q : Quadrilateral) :
  is_rectangle q → has_equal_diagonals q := sorry

-- Theorem for the converse (which is false)
theorem converse_is_false : ¬(∀ q : Quadrilateral, has_equal_diagonals q → is_rectangle q) := sorry

-- Theorem for the contrapositive (which is true)
theorem contrapositive_is_true :
  ∀ q : Quadrilateral, ¬has_equal_diagonals q → ¬is_rectangle q := sorry

end NUMINAMATH_CALUDE_rectangle_equal_diagonals_converse_is_false_contrapositive_is_true_l3237_323790


namespace NUMINAMATH_CALUDE_ways_to_write_1800_as_sum_of_twos_and_threes_l3237_323722

/-- The number of ways to write a positive integer as a sum of 2s and 3s -/
def num_ways_as_sum_of_twos_and_threes (n : ℕ) : ℕ :=
  (n / 6 + 1) - (n % 6 / 2)

/-- Theorem stating that there are 301 ways to write 1800 as a sum of 2s and 3s -/
theorem ways_to_write_1800_as_sum_of_twos_and_threes :
  num_ways_as_sum_of_twos_and_threes 1800 = 301 := by
  sorry

#eval num_ways_as_sum_of_twos_and_threes 1800

end NUMINAMATH_CALUDE_ways_to_write_1800_as_sum_of_twos_and_threes_l3237_323722


namespace NUMINAMATH_CALUDE_john_reaches_floor_pushups_in_12_weeks_l3237_323739

/-- Represents the number of days John trains per week -/
def training_days_per_week : ℕ := 5

/-- Represents the number of push-up variations John needs to progress through -/
def num_variations : ℕ := 4

/-- Represents the number of reps John needs to reach before progressing to the next variation -/
def reps_to_progress : ℕ := 20

/-- Calculates the total number of days it takes John to progress through all variations -/
def total_training_days : ℕ := (num_variations - 1) * reps_to_progress

/-- Calculates the number of weeks it takes John to reach floor push-ups -/
def weeks_to_floor_pushups : ℕ := total_training_days / training_days_per_week

/-- Theorem stating that it takes John 12 weeks to reach floor push-ups -/
theorem john_reaches_floor_pushups_in_12_weeks : weeks_to_floor_pushups = 12 := by
  sorry

end NUMINAMATH_CALUDE_john_reaches_floor_pushups_in_12_weeks_l3237_323739


namespace NUMINAMATH_CALUDE_solve_cookies_problem_l3237_323779

def cookies_problem (total_baked : ℕ) (kristy_ate : ℕ) (friend1_took : ℕ) (friend2_took : ℕ) (friend3_took : ℕ) (cookies_left : ℕ) : Prop :=
  let cookies_taken := kristy_ate + friend1_took + friend2_took + friend3_took
  let cookies_given_away := total_baked - cookies_left
  let brother_cookies := cookies_given_away - cookies_taken
  brother_cookies = 1

theorem solve_cookies_problem :
  cookies_problem 22 2 3 5 5 6 := by
  sorry

end NUMINAMATH_CALUDE_solve_cookies_problem_l3237_323779


namespace NUMINAMATH_CALUDE_john_drinks_42_quarts_per_week_l3237_323749

/-- The number of quarts John drinks in a week -/
def quarts_per_week (gallons_per_day : ℚ) (days_per_week : ℕ) (quarts_per_gallon : ℕ) : ℚ :=
  gallons_per_day * days_per_week * quarts_per_gallon

/-- Proof that John drinks 42 quarts of water in a week -/
theorem john_drinks_42_quarts_per_week :
  quarts_per_week (3/2) 7 4 = 42 := by
  sorry

end NUMINAMATH_CALUDE_john_drinks_42_quarts_per_week_l3237_323749


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_as_coefficients_l3237_323791

theorem quadratic_equation_roots_as_coefficients :
  ∀ (A B : ℝ),
  (∀ x : ℝ, x^2 + A*x + B = 0 ↔ x = A ∨ x = B) →
  ((A = 0 ∧ B = 0) ∨ (A = 1 ∧ B = -2)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_as_coefficients_l3237_323791


namespace NUMINAMATH_CALUDE_parallel_vectors_magnitude_l3237_323777

/-- Given two planar vectors a and b, where a is parallel to b,
    prove that the magnitude of 3a + b is √5 -/
theorem parallel_vectors_magnitude (y : ℝ) : 
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![-2, y]
  (a 0 * b 1 = a 1 * b 0) →  -- Parallel condition
  Real.sqrt ((3 * a 0 + b 0)^2 + (3 * a 1 + b 1)^2) = Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_magnitude_l3237_323777


namespace NUMINAMATH_CALUDE_smith_laundry_loads_l3237_323717

/-- The number of bath towels Kylie uses in one month -/
def kylie_towels : ℕ := 3

/-- The number of bath towels Kylie's daughters use in one month -/
def daughters_towels : ℕ := 6

/-- The number of bath towels Kylie's husband uses in one month -/
def husband_towels : ℕ := 3

/-- The number of bath towels that fit in one load of laundry -/
def towels_per_load : ℕ := 4

/-- The total number of bath towels used by the Smith family in one month -/
def total_towels : ℕ := kylie_towels + daughters_towels + husband_towels

/-- The number of laundry loads required to clean all used towels -/
def required_loads : ℕ := (total_towels + towels_per_load - 1) / towels_per_load

theorem smith_laundry_loads : required_loads = 3 := by
  sorry

end NUMINAMATH_CALUDE_smith_laundry_loads_l3237_323717


namespace NUMINAMATH_CALUDE_triangle_properties_l3237_323744

-- Define the triangle ABC
def Triangle (A B C : ℝ) := A + B + C = Real.pi

-- Define the conditions
def ConditionOne (A B C : ℝ) := A + B = 3 * C
def ConditionTwo (A B C : ℝ) := 2 * Real.sin (A - C) = Real.sin B
def ConditionThree := 5

-- Define the height function
def Height (A B C : ℝ) : ℝ := sorry

-- Theorem statement
theorem triangle_properties (A B C : ℝ) 
  (h1 : Triangle A B C) 
  (h2 : ConditionOne A B C) 
  (h3 : ConditionTwo A B C) :
  Real.sin A = 3 * Real.sqrt 10 / 10 ∧ 
  Height A B C = 6 := by sorry

end NUMINAMATH_CALUDE_triangle_properties_l3237_323744


namespace NUMINAMATH_CALUDE_tank_b_one_third_full_time_l3237_323727

/-- Represents a rectangular tank with given dimensions -/
structure RectangularTank where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a rectangular tank -/
def RectangularTank.volume (tank : RectangularTank) : ℝ :=
  tank.length * tank.width * tank.height

/-- Represents the filling process of a tank -/
structure TankFilling where
  tank : RectangularTank
  fillRate : ℝ  -- in cm³/s

/-- Theorem: Tank B will be 1/3 full after 30 seconds -/
theorem tank_b_one_third_full_time (tank_b : TankFilling) 
    (h1 : tank_b.tank.length = 5)
    (h2 : tank_b.tank.width = 9)
    (h3 : tank_b.tank.height = 8)
    (h4 : tank_b.fillRate = 4) : 
    tank_b.fillRate * 30 = (1/3) * tank_b.tank.volume := by
  sorry

#check tank_b_one_third_full_time

end NUMINAMATH_CALUDE_tank_b_one_third_full_time_l3237_323727


namespace NUMINAMATH_CALUDE_smallest_z_magnitude_l3237_323737

theorem smallest_z_magnitude (z : ℂ) (h : Complex.abs (z - 9) + Complex.abs (z - 4 * Complex.I) = 15) :
  ∃ (w : ℂ), Complex.abs w ≤ Complex.abs z ∧ Complex.abs w = 36 / Real.sqrt 97 :=
sorry

end NUMINAMATH_CALUDE_smallest_z_magnitude_l3237_323737


namespace NUMINAMATH_CALUDE_traffic_to_driving_ratio_l3237_323768

theorem traffic_to_driving_ratio (total_time driving_time : ℝ) 
  (h1 : total_time = 15)
  (h2 : driving_time = 5) :
  (total_time - driving_time) / driving_time = 2 := by
  sorry

end NUMINAMATH_CALUDE_traffic_to_driving_ratio_l3237_323768


namespace NUMINAMATH_CALUDE_f_symmetry_f_max_min_on_interval_l3237_323769

def f (x : ℝ) : ℝ := x^3 - 27*x

theorem f_symmetry (x : ℝ) : f (-x) = -f x := by sorry

theorem f_max_min_on_interval :
  let a : ℝ := -4
  let b : ℝ := 5
  (∃ x ∈ Set.Icc a b, ∀ y ∈ Set.Icc a b, f y ≤ f x) ∧
  (∃ x ∈ Set.Icc a b, ∀ y ∈ Set.Icc a b, f x ≤ f y) ∧
  (∃ x ∈ Set.Icc a b, f x = 54) ∧
  (∃ x ∈ Set.Icc a b, f x = -54) := by sorry

end NUMINAMATH_CALUDE_f_symmetry_f_max_min_on_interval_l3237_323769


namespace NUMINAMATH_CALUDE_opposite_of_negative_nine_l3237_323782

theorem opposite_of_negative_nine : 
  (-(- 9 : ℤ)) = (9 : ℤ) := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_nine_l3237_323782


namespace NUMINAMATH_CALUDE_two_true_propositions_l3237_323701

-- Define the original proposition
def original_prop (a b c : ℝ) : Prop :=
  a > b → a * c^2 > b * c^2

-- Define the converse of the original proposition
def converse_prop (a b c : ℝ) : Prop :=
  a * c^2 > b * c^2 → a > b

-- Define the negation of the original proposition
def negation_prop (a b c : ℝ) : Prop :=
  ¬(a > b → a * c^2 > b * c^2)

-- Theorem statement
theorem two_true_propositions :
  ∃ (p q : Prop) (r : Prop),
    (p = ∀ a b c : ℝ, original_prop a b c) ∧
    (q = ∀ a b c : ℝ, converse_prop a b c) ∧
    (r = ∀ a b c : ℝ, negation_prop a b c) ∧
    ((¬p ∧ q ∧ r) ∨ (p ∧ ¬q ∧ r) ∨ (p ∧ q ∧ ¬r)) :=
sorry

end NUMINAMATH_CALUDE_two_true_propositions_l3237_323701


namespace NUMINAMATH_CALUDE_sum_b_plus_c_l3237_323754

theorem sum_b_plus_c (a b c d : ℝ) 
  (h1 : a + b = 12)
  (h2 : c + d = 3)
  (h3 : a + d = 6) :
  b + c = 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_b_plus_c_l3237_323754


namespace NUMINAMATH_CALUDE_tangent_line_exists_tangent_line_equation_l3237_323711

/-- The function f(x) = x³ - 3x -/
def f (x : ℝ) : ℝ := x^3 - 3*x

/-- The derivative of f(x) -/
def f_deriv (x : ℝ) : ℝ := 3*x^2 - 3

/-- Theorem: There exists a tangent line to y = f(x) passing through (2, -6) -/
theorem tangent_line_exists : 
  ∃ (x₀ : ℝ), 
    (f x₀ + f_deriv x₀ * (2 - x₀) = -6) ∧ 
    ((f_deriv x₀ = -3) ∨ (f_deriv x₀ = 24)) :=
sorry

/-- Theorem: The tangent line equation is y = -3x or y = 24x - 54 -/
theorem tangent_line_equation (x₀ : ℝ) 
  (h : (f x₀ + f_deriv x₀ * (2 - x₀) = -6) ∧ 
       ((f_deriv x₀ = -3) ∨ (f_deriv x₀ = 24))) : 
  (∀ x y, y = -3*x) ∨ (∀ x y, y = 24*x - 54) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_exists_tangent_line_equation_l3237_323711


namespace NUMINAMATH_CALUDE_quadratic_roots_theorem_l3237_323767

theorem quadratic_roots_theorem (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    (∀ x : ℝ, x^2 + 2*k*x + k^2 = x + 1 ↔ x = x₁ ∨ x = x₂) ∧
    (3*x₁ - x₂)*(x₁ - 3*x₂) = 19) →
  k = 0 ∨ k = -3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_theorem_l3237_323767


namespace NUMINAMATH_CALUDE_baker_weekday_hours_l3237_323742

/-- Represents the baker's baking schedule and output --/
structure BakingSchedule where
  loavesPerHourPerOven : ℕ
  numOvens : ℕ
  weekendHoursPerDay : ℕ
  totalLoavesIn3Weeks : ℕ

/-- Calculates the number of hours the baker bakes from Monday to Friday each week --/
def weekdayHoursPerWeek (schedule : BakingSchedule) : ℕ :=
  let loavesPerHour := schedule.loavesPerHourPerOven * schedule.numOvens
  let weekendHours := schedule.weekendHoursPerDay * 2  -- 2 weekend days
  let weekendLoavesPerWeek := loavesPerHour * weekendHours
  let weekdayLoavesIn3Weeks := schedule.totalLoavesIn3Weeks - (weekendLoavesPerWeek * 3)
  weekdayLoavesIn3Weeks / (loavesPerHour * 3)

/-- Theorem stating that given the baker's schedule, they bake for 25 hours on weekdays --/
theorem baker_weekday_hours (schedule : BakingSchedule)
  (h1 : schedule.loavesPerHourPerOven = 5)
  (h2 : schedule.numOvens = 4)
  (h3 : schedule.weekendHoursPerDay = 2)
  (h4 : schedule.totalLoavesIn3Weeks = 1740) :
  weekdayHoursPerWeek schedule = 25 := by
  sorry


end NUMINAMATH_CALUDE_baker_weekday_hours_l3237_323742


namespace NUMINAMATH_CALUDE_parabola_midpoint_distance_squared_l3237_323751

/-- Given a parabola y = 3x^2 + 6x - 2 and two points C and D on it with the origin as their midpoint,
    the square of the distance between C and D is 740/3. -/
theorem parabola_midpoint_distance_squared :
  ∀ (C D : ℝ × ℝ),
  (∃ (x y : ℝ), C = (x, y) ∧ y = 3 * x^2 + 6 * x - 2) →
  (∃ (x y : ℝ), D = (x, y) ∧ y = 3 * x^2 + 6 * x - 2) →
  (0 : ℝ × ℝ) = ((C.1 + D.1) / 2, (C.2 + D.2) / 2) →
  (C.1 - D.1)^2 + (C.2 - D.2)^2 = 740 / 3 :=
by sorry

end NUMINAMATH_CALUDE_parabola_midpoint_distance_squared_l3237_323751
