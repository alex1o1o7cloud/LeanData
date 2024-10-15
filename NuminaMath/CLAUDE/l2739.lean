import Mathlib

namespace NUMINAMATH_CALUDE_cubic_equation_value_l2739_273977

theorem cubic_equation_value (a b : ℝ) :
  (a * (-2)^3 + b * (-2) - 7 = 9) →
  (a * 2^3 + b * 2 - 7 = -23) :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_value_l2739_273977


namespace NUMINAMATH_CALUDE_classroom_students_l2739_273911

theorem classroom_students (boys girls : ℕ) : 
  boys * 5 = girls * 3 →  -- ratio of boys to girls is 3:5
  girls = boys + 4 →      -- there are 4 more girls than boys
  boys + girls = 16       -- total number of students is 16
:= by sorry

end NUMINAMATH_CALUDE_classroom_students_l2739_273911


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l2739_273939

theorem trigonometric_equation_solution (x : ℝ) : 
  (4 * (Real.tan (8 * x))^4 + 4 * Real.sin (2 * x) * Real.sin (6 * x) - 
   Real.cos (4 * x) - Real.cos (12 * x) + 2) / Real.sqrt (Real.cos x - Real.sin x) = 0 ∧
  Real.cos x - Real.sin x > 0 ↔
  (∃ n : ℤ, x = -π/2 + 2*n*π ∨ x = -π/4 + 2*n*π ∨ x = 2*n*π) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l2739_273939


namespace NUMINAMATH_CALUDE_bags_used_by_kid4_l2739_273961

def hours : ℕ := 5
def ears_per_row : ℕ := 85
def seeds_per_bag : ℕ := 48
def seeds_per_ear : ℕ := 2
def rows_per_hour_kid4 : ℕ := 5

def bags_used_kid4 : ℕ :=
  let rows := hours * rows_per_hour_kid4
  let ears := rows * ears_per_row
  let seeds := ears * seeds_per_ear
  (seeds + seeds_per_bag - 1) / seeds_per_bag

theorem bags_used_by_kid4 : bags_used_kid4 = 89 := by sorry

end NUMINAMATH_CALUDE_bags_used_by_kid4_l2739_273961


namespace NUMINAMATH_CALUDE_kareem_son_age_ratio_l2739_273975

/-- Proves that the ratio of Kareem's age to his son's age is 3:1 --/
theorem kareem_son_age_ratio :
  let kareem_age : ℕ := 42
  let son_age : ℕ := 14
  let future_sum : ℕ := 76
  let future_years : ℕ := 10
  (kareem_age + future_years) + (son_age + future_years) = future_sum →
  (kareem_age : ℚ) / son_age = 3 / 1 := by
sorry

end NUMINAMATH_CALUDE_kareem_son_age_ratio_l2739_273975


namespace NUMINAMATH_CALUDE_common_external_tangent_y_intercept_l2739_273908

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a line in the form y = mx + b -/
structure Line where
  m : ℝ
  b : ℝ

/-- Checks if a line is tangent to a circle -/
def isTangent (l : Line) (c : Circle) : Prop :=
  let (x₀, y₀) := c.center
  (l.m * x₀ - y₀ + l.b)^2 = (c.radius^2 * (l.m^2 + 1))

theorem common_external_tangent_y_intercept :
  let c₁ : Circle := ⟨(1, 3), 3⟩
  let c₂ : Circle := ⟨(15, 10), 8⟩
  ∃ (l : Line), l.m > 0 ∧ isTangent l c₁ ∧ isTangent l c₂ ∧ l.b = 5/3 :=
sorry

end NUMINAMATH_CALUDE_common_external_tangent_y_intercept_l2739_273908


namespace NUMINAMATH_CALUDE_sum_of_coordinates_zero_l2739_273997

/-- For all points (x, y) in the real plane where x + y = 0, prove that y = -x -/
theorem sum_of_coordinates_zero (x y : ℝ) (h : x + y = 0) : y = -x := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coordinates_zero_l2739_273997


namespace NUMINAMATH_CALUDE_mistaken_calculation_l2739_273927

theorem mistaken_calculation (x : ℕ) : 423 - x = 421 → (423 * x) + (423 - x) = 1267 := by
  sorry

end NUMINAMATH_CALUDE_mistaken_calculation_l2739_273927


namespace NUMINAMATH_CALUDE_pool_volume_l2739_273981

/-- Represents a pool with given parameters -/
structure Pool where
  diameter : ℝ
  fill_time : ℝ
  hose_rates : List ℝ

/-- Calculates the volume of water delivered by hoses over a given time -/
def water_volume (p : Pool) : ℝ :=
  (p.hose_rates.sum * p.fill_time * 60)

/-- The theorem states that a pool with given parameters has a volume of 15000 gallons -/
theorem pool_volume (p : Pool) 
  (h1 : p.diameter = 24)
  (h2 : p.fill_time = 25)
  (h3 : p.hose_rates = [2, 2, 3, 3]) :
  water_volume p = 15000 := by
  sorry

#check pool_volume

end NUMINAMATH_CALUDE_pool_volume_l2739_273981


namespace NUMINAMATH_CALUDE_intersection_of_three_lines_l2739_273984

/-- 
Given three lines that intersect at the same point:
1. y = 2x + 7
2. y = -3x - 6
3. y = 4x + m
Prove that m = 61/5
-/
theorem intersection_of_three_lines (x y m : ℝ) : 
  (y = 2*x + 7) ∧ 
  (y = -3*x - 6) ∧ 
  (y = 4*x + m) → 
  m = 61/5 := by
sorry

end NUMINAMATH_CALUDE_intersection_of_three_lines_l2739_273984


namespace NUMINAMATH_CALUDE_complement_intersection_A_B_l2739_273976

def A : Set ℝ := {x | |x - 2| ≤ 3}
def B : Set ℝ := {x | ∃ y, y = Real.log (x - 1)}

theorem complement_intersection_A_B :
  (A ∩ B)ᶜ = {x : ℝ | x ≤ 1 ∨ x > 5} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_A_B_l2739_273976


namespace NUMINAMATH_CALUDE_sine_phase_shift_l2739_273972

theorem sine_phase_shift (x : ℝ) : 
  Real.sin (2 * x - π / 3) = Real.sin (2 * (x - π / 6)) := by sorry

end NUMINAMATH_CALUDE_sine_phase_shift_l2739_273972


namespace NUMINAMATH_CALUDE_x_range_l2739_273901

def p (x : ℝ) : Prop := x^2 + 2*x - 3 > 0

def q (x : ℝ) : Prop := 1 / (3 - x) > 1

theorem x_range (x : ℝ) (h1 : p x) (h2 : ¬(q x)) : 
  x ≥ 3 ∨ (1 < x ∧ x ≤ 2) ∨ x < -3 := by
  sorry

end NUMINAMATH_CALUDE_x_range_l2739_273901


namespace NUMINAMATH_CALUDE_system_solution_l2739_273900

theorem system_solution (a : ℝ) (x y z : ℝ) :
  (x + y + z = a) →
  (x^2 + y^2 + z^2 = a^2) →
  (x^3 + y^3 + z^3 = a^3) →
  ((x = 0 ∧ y = 0 ∧ z = a) ∨ (x = 0 ∧ y = a ∧ z = 0) ∨ (x = a ∧ y = 0 ∧ z = 0)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l2739_273900


namespace NUMINAMATH_CALUDE_jason_has_more_blue_marbles_l2739_273941

/-- The number of blue marbles Jason has -/
def jason_blue : ℕ := 44

/-- The number of blue marbles Tom has -/
def tom_blue : ℕ := 24

/-- The difference in blue marbles between Jason and Tom -/
def blue_marble_difference : ℕ := jason_blue - tom_blue

/-- Theorem stating that Jason has 20 more blue marbles than Tom -/
theorem jason_has_more_blue_marbles : blue_marble_difference = 20 := by
  sorry

end NUMINAMATH_CALUDE_jason_has_more_blue_marbles_l2739_273941


namespace NUMINAMATH_CALUDE_uruguay_goals_conceded_l2739_273926

theorem uruguay_goals_conceded : 
  ∀ (x : ℕ), 
  (5 + 5 + 4 + 0 = 2 + 4 + x + 3) → 
  x = 5 := by
sorry

end NUMINAMATH_CALUDE_uruguay_goals_conceded_l2739_273926


namespace NUMINAMATH_CALUDE_smallest_number_with_remainders_l2739_273965

theorem smallest_number_with_remainders : ∃ (n : ℕ), n > 0 ∧
  n % 3 = 1 ∧
  n % 5 = 3 ∧
  n % 6 = 4 ∧
  ∀ (m : ℕ), m > 0 → m % 3 = 1 → m % 5 = 3 → m % 6 = 4 → n ≤ m :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_number_with_remainders_l2739_273965


namespace NUMINAMATH_CALUDE_equation_is_quadratic_l2739_273921

/-- Represents a quadratic equation in one variable x -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  h : a ≠ 0

/-- Checks if an equation is quadratic in one variable x -/
def is_quadratic_in_x (f : ℝ → ℝ) : Prop :=
  ∃ (q : QuadraticEquation), ∀ x, f x = q.a * x^2 + q.b * x + q.c

/-- The equation x² = 1 -/
def equation (x : ℝ) : ℝ := x^2 - 1

theorem equation_is_quadratic : is_quadratic_in_x equation := by sorry

end NUMINAMATH_CALUDE_equation_is_quadratic_l2739_273921


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l2739_273968

theorem unique_solution_quadratic (k : ℚ) : 
  (∃! x : ℝ, (x + 5) * (x + 2) = k + 3 * x) ↔ k = 6 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l2739_273968


namespace NUMINAMATH_CALUDE_max_y_coordinate_difference_l2739_273979

-- Define the two functions
def f (x : ℝ) : ℝ := 3 - x^2 + x^3
def g (x : ℝ) : ℝ := 1 + x^2 + x^3

-- Define the set of intersection points
def intersection_points : Set ℝ := {x : ℝ | f x = g x}

-- Theorem statement
theorem max_y_coordinate_difference :
  ∃ (a b : ℝ), a ∈ intersection_points ∧ b ∈ intersection_points ∧
  ∀ (x y : ℝ), x ∈ intersection_points → y ∈ intersection_points →
  |f x - f y| ≤ |f a - f b| ∧ |f a - f b| = 2 :=
sorry

end NUMINAMATH_CALUDE_max_y_coordinate_difference_l2739_273979


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_1729_l2739_273991

theorem largest_prime_factor_of_1729 :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 1729 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 1729 → q ≤ p :=
by
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_1729_l2739_273991


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_1729_l2739_273947

theorem largest_prime_factor_of_1729 : ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 1729 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 1729 → q ≤ p :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_1729_l2739_273947


namespace NUMINAMATH_CALUDE_m_range_l2739_273920

def A : Set ℝ := {x : ℝ | -1 < x ∧ x < 3}

def B (m : ℝ) : Set ℝ := {x : ℝ | -1 < x ∧ x < m + 1}

theorem m_range (m : ℝ) : B m ⊆ A ∧ B m ≠ A → -2 < m ∧ m < 2 := by
  sorry

end NUMINAMATH_CALUDE_m_range_l2739_273920


namespace NUMINAMATH_CALUDE_pentagon_segment_parallel_and_length_l2739_273999

/-- Given a pentagon ABCDE with points P, Q, R, S on its sides and points M, N on PR and QS respectively,
    satisfying specific ratios, prove that MN is parallel to AE and its length is AE / ((k₁ + 1)(k₂ + 1)). -/
theorem pentagon_segment_parallel_and_length 
  (A B C D E P Q R S M N : ℝ × ℝ) (k₁ k₂ : ℝ) :
  -- Pentagon ABCDE
  -- Points P, Q, R, S on sides AB, BC, CD, DE respectively
  (P.1 - A.1) / (B.1 - P.1) = k₁ ∧ 
  (P.2 - A.2) / (B.2 - P.2) = k₁ ∧
  (Q.1 - B.1) / (C.1 - Q.1) = k₂ ∧
  (Q.2 - B.2) / (C.2 - Q.2) = k₂ ∧
  (R.1 - D.1) / (C.1 - R.1) = k₁ ∧
  (R.2 - D.2) / (C.2 - R.2) = k₁ ∧
  (S.1 - E.1) / (D.1 - S.1) = k₂ ∧
  (S.2 - E.2) / (D.2 - S.2) = k₂ ∧
  -- Points M and N on PR and QS respectively
  (M.1 - P.1) / (R.1 - M.1) = k₂ ∧
  (M.2 - P.2) / (R.2 - M.2) = k₂ ∧
  (N.1 - S.1) / (Q.1 - N.1) = k₁ ∧
  (N.2 - S.2) / (Q.2 - N.2) = k₁ →
  -- MN is parallel to AE
  (N.2 - M.2) / (N.1 - M.1) = (E.2 - A.2) / (E.1 - A.1) ∧
  -- Length of MN
  Real.sqrt ((N.1 - M.1)^2 + (N.2 - M.2)^2) = 
    Real.sqrt ((E.1 - A.1)^2 + (E.2 - A.2)^2) / ((k₁ + 1) * (k₂ + 1)) := by
  sorry

end NUMINAMATH_CALUDE_pentagon_segment_parallel_and_length_l2739_273999


namespace NUMINAMATH_CALUDE_simplify_expression_l2739_273902

theorem simplify_expression : 
  2 - (1 / (2 + Real.sqrt 5)) + (1 / (2 - Real.sqrt 5)) = 2 - 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2739_273902


namespace NUMINAMATH_CALUDE_average_hamburgers_is_nine_l2739_273909

/-- The number of hamburgers sold in a week -/
def hamburgers_sold : ℕ := 63

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The average number of hamburgers sold per day -/
def average_hamburgers_per_day : ℚ := hamburgers_sold / days_in_week

/-- Theorem stating that the average number of hamburgers sold per day is 9 -/
theorem average_hamburgers_is_nine :
  average_hamburgers_per_day = 9 := by
  sorry

end NUMINAMATH_CALUDE_average_hamburgers_is_nine_l2739_273909


namespace NUMINAMATH_CALUDE_shooter_probability_l2739_273969

theorem shooter_probability (p_10 p_9 p_8 : ℝ) 
  (h1 : p_10 = 0.24)
  (h2 : p_9 = 0.28)
  (h3 : p_8 = 0.19) :
  1 - p_10 - p_9 = 0.48 := by
  sorry

end NUMINAMATH_CALUDE_shooter_probability_l2739_273969


namespace NUMINAMATH_CALUDE_polygon_sides_l2739_273951

theorem polygon_sides (sum_interior_angles : ℝ) : sum_interior_angles = 540 → ∃ n : ℕ, n = 5 ∧ (n - 2) * 180 = sum_interior_angles :=
by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l2739_273951


namespace NUMINAMATH_CALUDE_abc_value_l2739_273933

theorem abc_value (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a * (b + c) = 165) (h2 : b * (c + a) = 195) (h3 : c * (a + b) = 180) :
  a * b * c = 15 * Real.sqrt 210 := by
sorry

end NUMINAMATH_CALUDE_abc_value_l2739_273933


namespace NUMINAMATH_CALUDE_cube_root_simplification_l2739_273942

theorem cube_root_simplification : (2^6 * 3^3 * 7^3 * 13^3 : ℝ)^(1/3) = 1092 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_simplification_l2739_273942


namespace NUMINAMATH_CALUDE_square_roots_sum_zero_l2739_273940

theorem square_roots_sum_zero (x : ℝ) (n : ℝ) (h1 : n > 0) 
  (h2 : (x - 4) + 3 = 0) : x = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_roots_sum_zero_l2739_273940


namespace NUMINAMATH_CALUDE_height_on_side_BC_l2739_273985

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    where a = √3, b = √2, and 1 + 2cos(B+C) = 0, 
    prove that the height h on side BC is equal to (√3 + 1) / 2. -/
theorem height_on_side_BC (A B C : ℝ) (a b c : ℝ) (h : ℝ) : 
  a = Real.sqrt 3 → 
  b = Real.sqrt 2 → 
  1 + 2 * Real.cos (B + C) = 0 → 
  h = (Real.sqrt 3 + 1) / 2 := by
sorry

end NUMINAMATH_CALUDE_height_on_side_BC_l2739_273985


namespace NUMINAMATH_CALUDE_distance_between_vertices_l2739_273945

-- Define the equation
def equation (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + y^2) + abs (y - 2) = 4

-- Define the parabolas
def parabola1 (x y : ℝ) : Prop := y = 3 - (1/12) * x^2
def parabola2 (x y : ℝ) : Prop := y = (1/4) * x^2 - 1

-- Define the vertices
def vertex1 : ℝ × ℝ := (0, 3)
def vertex2 : ℝ × ℝ := (0, -1)

-- Theorem statement
theorem distance_between_vertices :
  ∀ x y : ℝ, equation x y →
  (∃ x1 y1 x2 y2 : ℝ, 
    parabola1 x1 y1 ∧ parabola2 x2 y2 ∧
    (x1, y1) = vertex1 ∧ (x2, y2) = vertex2 ∧
    abs (y1 - y2) = 4) :=
by sorry

end NUMINAMATH_CALUDE_distance_between_vertices_l2739_273945


namespace NUMINAMATH_CALUDE_least_three_digit_with_digit_product_18_l2739_273936

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_product (n : ℕ) : ℕ :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  hundreds * tens * ones

theorem least_three_digit_with_digit_product_18 :
  ∃ (n : ℕ), is_three_digit n ∧ digit_product n = 18 ∧
  ∀ (m : ℕ), is_three_digit m → digit_product m = 18 → n ≤ m :=
by
  sorry

end NUMINAMATH_CALUDE_least_three_digit_with_digit_product_18_l2739_273936


namespace NUMINAMATH_CALUDE_job_completion_solution_l2739_273952

/-- Represents the time taken by machines to complete a job -/
def job_completion_time (x : ℝ) : Prop :=
  let p_alone := x + 5
  let q_alone := x + 2
  let r_alone := 2 * x
  let pq_together := x + 3
  (1 / p_alone + 1 / q_alone + 1 / r_alone = 1 / x) ∧
  (1 / p_alone + 1 / q_alone = 1 / pq_together)

/-- Theorem stating that x = 2 satisfies the job completion time conditions -/
theorem job_completion_solution : job_completion_time 2 := by
  sorry

end NUMINAMATH_CALUDE_job_completion_solution_l2739_273952


namespace NUMINAMATH_CALUDE_total_population_two_villages_l2739_273978

/-- The total population of two villages given partial information about each village's population -/
theorem total_population_two_villages
  (village1_90_percent : ℝ)
  (village2_80_percent : ℝ)
  (h1 : village1_90_percent = 45000)
  (h2 : village2_80_percent = 64000) :
  (village1_90_percent / 0.9 + village2_80_percent / 0.8) = 130000 :=
by sorry

end NUMINAMATH_CALUDE_total_population_two_villages_l2739_273978


namespace NUMINAMATH_CALUDE_fraction_problem_l2739_273923

theorem fraction_problem (x : ℝ) (f : ℝ) (h1 : x > 0) (h2 : x = 1/3) 
  (h3 : f * x = (16/216) * (1/x)) : f = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l2739_273923


namespace NUMINAMATH_CALUDE_max_pairs_sum_l2739_273917

theorem max_pairs_sum (k : ℕ) 
  (a b : Fin k → ℕ) 
  (h1 : ∀ i : Fin k, a i < b i)
  (h2 : ∀ i : Fin k, a i ≤ 1500 ∧ b i ≤ 1500)
  (h3 : ∀ i j : Fin k, i ≠ j → a i ≠ a j ∧ a i ≠ b j ∧ b i ≠ a j ∧ b i ≠ b j)
  (h4 : ∀ i : Fin k, a i + b i ≤ 1500)
  (h5 : ∀ i j : Fin k, i ≠ j → a i + b i ≠ a j + b j) :
  k ≤ 599 :=
sorry

end NUMINAMATH_CALUDE_max_pairs_sum_l2739_273917


namespace NUMINAMATH_CALUDE_cauchy_inequality_and_minimum_value_l2739_273993

theorem cauchy_inequality_and_minimum_value (a b x y : ℝ) :
  (a^2 + b^2) * (x^2 + y^2) ≥ (a*x + b*y)^2 ∧
  (x^2 + y^2 = 2 ∧ |x| ≠ |y| → ∃ (min : ℝ), min = 50/9 ∧ ∀ z, z = 1/(9*x^2) + 9/y^2 → z ≥ min) :=
by sorry

end NUMINAMATH_CALUDE_cauchy_inequality_and_minimum_value_l2739_273993


namespace NUMINAMATH_CALUDE_count_increasing_digit_numbers_eq_502_l2739_273994

def is_increasing_digits (n : ℕ) : Prop :=
  ∀ i j, i < j → (n.digits 10).get i < (n.digits 10).get j

def count_increasing_digit_numbers : ℕ :=
  (Finset.range 8).sum (λ k => Nat.choose 9 (k + 2))

theorem count_increasing_digit_numbers_eq_502 :
  count_increasing_digit_numbers = 502 :=
sorry

end NUMINAMATH_CALUDE_count_increasing_digit_numbers_eq_502_l2739_273994


namespace NUMINAMATH_CALUDE_pen_probabilities_l2739_273988

/-- Represents the total number of pens in the box -/
def total_pens : ℕ := 6

/-- Represents the number of first-class quality pens -/
def first_class_pens : ℕ := 4

/-- Represents the number of second-class quality pens -/
def second_class_pens : ℕ := 2

/-- Represents the number of pens drawn -/
def pens_drawn : ℕ := 2

/-- Calculates the probability of drawing exactly one first-class quality pen -/
def prob_one_first_class : ℚ :=
  (Nat.choose first_class_pens 1 * Nat.choose second_class_pens 1) / Nat.choose total_pens pens_drawn

/-- Calculates the probability of drawing at least one second-class quality pen -/
def prob_at_least_one_second_class : ℚ :=
  1 - (Nat.choose first_class_pens pens_drawn) / Nat.choose total_pens pens_drawn

theorem pen_probabilities :
  prob_one_first_class = 8/15 ∧
  prob_at_least_one_second_class = 3/5 :=
sorry

end NUMINAMATH_CALUDE_pen_probabilities_l2739_273988


namespace NUMINAMATH_CALUDE_line_through_intersection_and_perpendicular_l2739_273922

-- Define the lines l₁, l₂, and l₃
def l₁ (x y : ℝ) : Prop := x - 2*y + 3 = 0
def l₂ (x y : ℝ) : Prop := 2*x + 3*y - 8 = 0
def l₃ (x y : ℝ) : Prop := 3*x - y + 1 = 0

-- Define the line l (the answer)
def l (x y : ℝ) : Prop := x + 3*y - 7 = 0

-- Define the intersection point M
def M : ℝ × ℝ := (1, 2)

-- Theorem statement
theorem line_through_intersection_and_perpendicular :
  (l₁ M.1 M.2 ∧ l₂ M.1 M.2) ∧  -- M is the intersection of l₁ and l₂
  (∀ x y : ℝ, l x y → l₃ x y → (x - M.1) * 3 + (y - M.2) * (-1) = 0) ∧  -- l is perpendicular to l₃
  l M.1 M.2  -- l passes through M
  := by sorry

end NUMINAMATH_CALUDE_line_through_intersection_and_perpendicular_l2739_273922


namespace NUMINAMATH_CALUDE_smallest_equal_prob_sum_l2739_273928

/-- The number of faces on a standard die -/
def faces : ℕ := 6

/-- The target sum we're comparing to -/
def target_sum : ℕ := 2001

/-- The smallest number of dice needed to potentially reach the target sum -/
def min_dice : ℕ := (target_sum + faces - 1) / faces

/-- The function that transforms a die roll -/
def transform (x : ℕ) : ℕ := faces + 1 - x

/-- The smallest value S with equal probability to the target sum -/
def smallest_S : ℕ := (faces + 1) * min_dice - target_sum

theorem smallest_equal_prob_sum :
  smallest_S = 337 :=
sorry

end NUMINAMATH_CALUDE_smallest_equal_prob_sum_l2739_273928


namespace NUMINAMATH_CALUDE_polynomial_equality_l2739_273914

theorem polynomial_equality (a a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (2*x - 1)^5 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a - a₁ + a₂ - a₃ + a₄ - a₅ = -243 := by
sorry

end NUMINAMATH_CALUDE_polynomial_equality_l2739_273914


namespace NUMINAMATH_CALUDE_total_sandcastles_and_towers_l2739_273944

/-- The number of sandcastles on Mark's beach -/
def marks_castles : ℕ := 20

/-- The number of towers per sandcastle on Mark's beach -/
def marks_towers_per_castle : ℕ := 10

/-- The ratio of Jeff's sandcastles to Mark's sandcastles -/
def jeff_castle_ratio : ℕ := 3

/-- The number of towers per sandcastle on Jeff's beach -/
def jeffs_towers_per_castle : ℕ := 5

/-- Theorem stating the combined total number of sandcastles and towers on both beaches -/
theorem total_sandcastles_and_towers :
  marks_castles * marks_towers_per_castle +
  marks_castles +
  (jeff_castle_ratio * marks_castles) * jeffs_towers_per_castle +
  (jeff_castle_ratio * marks_castles) = 580 := by
  sorry

end NUMINAMATH_CALUDE_total_sandcastles_and_towers_l2739_273944


namespace NUMINAMATH_CALUDE_hawks_score_l2739_273930

theorem hawks_score (total_points margin : ℕ) (h1 : total_points = 48) (h2 : margin = 16) :
  ∃ (eagles_score hawks_score : ℕ),
    eagles_score + hawks_score = total_points ∧
    eagles_score - hawks_score = margin ∧
    hawks_score = 16 := by
  sorry

end NUMINAMATH_CALUDE_hawks_score_l2739_273930


namespace NUMINAMATH_CALUDE_min_value_x_over_y_l2739_273998

theorem min_value_x_over_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + y = 2) :
  ∀ x' y' : ℝ, x' > 0 → y' > 0 → 1/x' + y' = 2 → x/y ≤ x'/y' ∧ ∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ 1/x₀ + y₀ = 2 ∧ x₀/y₀ = 1 :=
by sorry

end NUMINAMATH_CALUDE_min_value_x_over_y_l2739_273998


namespace NUMINAMATH_CALUDE_remainder_23_pow_2047_mod_17_l2739_273918

theorem remainder_23_pow_2047_mod_17 : 23^2047 % 17 = 11 := by
  sorry

end NUMINAMATH_CALUDE_remainder_23_pow_2047_mod_17_l2739_273918


namespace NUMINAMATH_CALUDE_fraction_nonzero_digits_l2739_273907

/-- The number of non-zero digits to the right of the decimal point in the decimal representation of a rational number -/
def nonZeroDigitsAfterDecimal (q : ℚ) : ℕ :=
  sorry

/-- The fraction we're considering -/
def fraction : ℚ := 120 / (2^4 * 5^9)

theorem fraction_nonzero_digits :
  nonZeroDigitsAfterDecimal fraction = 3 :=
sorry

end NUMINAMATH_CALUDE_fraction_nonzero_digits_l2739_273907


namespace NUMINAMATH_CALUDE_quilt_gray_percentage_l2739_273970

/-- Represents a square quilt with white and gray parts -/
structure Quilt where
  size : ℕ
  gray_half_squares : ℕ
  gray_quarter_squares : ℕ
  gray_full_squares : ℕ

/-- Calculates the percentage of gray area in the quilt -/
def gray_percentage (q : Quilt) : ℚ :=
  let total_squares := q.size * q.size
  let gray_squares := q.gray_half_squares / 2 + q.gray_quarter_squares / 4 + q.gray_full_squares
  (gray_squares * 100) / total_squares

/-- Theorem stating that the specific quilt configuration has 40% gray area -/
theorem quilt_gray_percentage :
  let q := Quilt.mk 5 8 8 4
  gray_percentage q = 40 := by
  sorry

end NUMINAMATH_CALUDE_quilt_gray_percentage_l2739_273970


namespace NUMINAMATH_CALUDE_binomial_square_expansion_l2739_273983

theorem binomial_square_expansion : 121 + 2*(11*9) + 81 = 400 := by sorry

end NUMINAMATH_CALUDE_binomial_square_expansion_l2739_273983


namespace NUMINAMATH_CALUDE_not_necessary_not_sufficient_neither_necessary_nor_sufficient_l2739_273990

/-- Two lines are parallel if they have the same slope and don't intersect. -/
def are_parallel (m : ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ 
  ∀ (x y : ℝ), (m * x + 4 * y - 6 = 0) ↔ (k * (x + m * y - 3) = 0)

/-- m = 2 is not necessary for the lines to be parallel. -/
theorem not_necessary (m : ℝ) : 
  ∃ m', m' ≠ 2 ∧ are_parallel m' :=
sorry

/-- m = 2 is not sufficient for the lines to be parallel. -/
theorem not_sufficient : ¬(are_parallel 2) :=
sorry

/-- m = 2 is neither necessary nor sufficient for the lines to be parallel. -/
theorem neither_necessary_nor_sufficient : 
  (∃ m', m' ≠ 2 ∧ are_parallel m') ∧ ¬(are_parallel 2) :=
sorry

end NUMINAMATH_CALUDE_not_necessary_not_sufficient_neither_necessary_nor_sufficient_l2739_273990


namespace NUMINAMATH_CALUDE_green_tea_cost_july_l2739_273913

/-- The cost of green tea in July given initial prices and price changes -/
theorem green_tea_cost_july (initial_cost : ℝ) 
  (h1 : initial_cost > 0) 
  (h2 : 3 * (0.1 * initial_cost + 2 * initial_cost) / 2 = 3.15) : 
  0.1 * initial_cost = 0.1 := by sorry

end NUMINAMATH_CALUDE_green_tea_cost_july_l2739_273913


namespace NUMINAMATH_CALUDE_sheet_area_difference_l2739_273953

/-- The combined area (front and back) of a rectangular sheet of paper -/
def combinedArea (length width : ℝ) : ℝ := 2 * length * width

/-- The difference in combined area between two rectangular sheets of paper -/
def areaDifference (l1 w1 l2 w2 : ℝ) : ℝ :=
  combinedArea l1 w1 - combinedArea l2 w2

theorem sheet_area_difference :
  areaDifference 11 17 8.5 11 = 187 := by
  sorry

end NUMINAMATH_CALUDE_sheet_area_difference_l2739_273953


namespace NUMINAMATH_CALUDE_cubic_sum_theorem_l2739_273931

theorem cubic_sum_theorem (a b c : ℝ) 
  (eq1 : a^2 + 3*b = 2)
  (eq2 : b^2 + 5*c = 3)
  (eq3 : c^2 + 7*a = 6) :
  a^3 + b^3 + c^3 = -0.875 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_theorem_l2739_273931


namespace NUMINAMATH_CALUDE_pizza_theorem_l2739_273903

/-- Calculates the total number of pizza slices brought by friends -/
def totalPizzaSlices (numFriends : ℕ) (slicesPerFriend : ℕ) : ℕ :=
  numFriends * slicesPerFriend

/-- Theorem: Given 4 friends, each bringing 4 slices of pizza, the total number of pizza slices is 16 -/
theorem pizza_theorem : totalPizzaSlices 4 4 = 16 := by
  sorry

end NUMINAMATH_CALUDE_pizza_theorem_l2739_273903


namespace NUMINAMATH_CALUDE_book_selection_count_l2739_273915

/-- Represents the number of books in each genre -/
def genre_books : Fin 4 → ℕ
  | 0 => 4  -- Mystery novels
  | 1 => 3  -- Fantasy novels
  | 2 => 3  -- Biographies
  | 3 => 3  -- Science fiction novels

/-- The number of ways to choose three books from three different genres -/
def book_combinations : ℕ := 4 * 3 * 3 * 3

theorem book_selection_count :
  book_combinations = 108 :=
sorry

end NUMINAMATH_CALUDE_book_selection_count_l2739_273915


namespace NUMINAMATH_CALUDE_rectangle_ratio_l2739_273910

theorem rectangle_ratio (w l a : ℝ) : 
  w = 4 → 
  a = 48 → 
  a = l * w → 
  l / w = 3 := by
sorry

end NUMINAMATH_CALUDE_rectangle_ratio_l2739_273910


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2739_273958

theorem sufficient_not_necessary (a : ℝ) : 
  (∀ a, a > 2 → a ≥ 1) ∧ (∃ a, a ≥ 1 ∧ a ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2739_273958


namespace NUMINAMATH_CALUDE_quadratic_function_ratio_bound_l2739_273925

/-- A quadratic function f(x) = ax² + bx + c with specific properties -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  derivative_at_zero_positive : b > 0
  nonnegative : ∀ x : ℝ, a * x^2 + b * x + c ≥ 0

/-- The ratio of f(1) to f'(0) for a QuadraticFunction is always at least 2 -/
theorem quadratic_function_ratio_bound (f : QuadraticFunction) :
  (f.a + f.b + f.c) / f.b ≥ 2 := by sorry

end NUMINAMATH_CALUDE_quadratic_function_ratio_bound_l2739_273925


namespace NUMINAMATH_CALUDE_hanna_money_spent_l2739_273950

theorem hanna_money_spent (rose_price : ℚ) (jenna_fraction : ℚ) (imma_fraction : ℚ) (total_given : ℕ) : 
  rose_price = 2 →
  jenna_fraction = 1/3 →
  imma_fraction = 1/2 →
  total_given = 125 →
  (jenna_fraction + imma_fraction) * (total_given / (jenna_fraction + imma_fraction)) * rose_price = 300 := by
sorry

end NUMINAMATH_CALUDE_hanna_money_spent_l2739_273950


namespace NUMINAMATH_CALUDE_absolute_value_and_square_root_l2739_273989

theorem absolute_value_and_square_root (x : ℝ) (h : 1 < x ∧ x ≤ 2) :
  |x - 3| + Real.sqrt ((x - 2)^2) = 5 - 2*x := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_and_square_root_l2739_273989


namespace NUMINAMATH_CALUDE_shaniqua_styles_l2739_273943

def haircut_price : ℕ := 12
def style_price : ℕ := 25
def total_earned : ℕ := 221
def haircuts_given : ℕ := 8

theorem shaniqua_styles (styles : ℕ) : styles = 5 := by
  sorry

end NUMINAMATH_CALUDE_shaniqua_styles_l2739_273943


namespace NUMINAMATH_CALUDE_solve_equation_l2739_273937

-- Define y as a constant real number
variable (y : ℝ)

-- Define the theorem
theorem solve_equation (x : ℝ) (h : Real.sqrt (x + y - 3) = 10) : x = 103 - y := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2739_273937


namespace NUMINAMATH_CALUDE_range_of_abc_l2739_273946

theorem range_of_abc (a b c : ℝ) (h1 : -3 < b) (h2 : b < a) (h3 : a < -1) (h4 : -2 < c) (h5 : c < -1) :
  0 < (a - b) * c^2 ∧ (a - b) * c^2 < 8 := by
  sorry

end NUMINAMATH_CALUDE_range_of_abc_l2739_273946


namespace NUMINAMATH_CALUDE_pyramid_surface_area_l2739_273960

/-- Given a pyramid with its base coinciding with a face of a cube and its apex at the center
    of the opposite face, the surface area of the pyramid can be expressed in terms of the
    cube's edge length. -/
theorem pyramid_surface_area (a : ℝ) (h : a > 0) :
  ∃ (S : ℝ), S = (a * (3 * Real.sqrt (4 * a^2 - a^2) + a * Real.sqrt 3)) / 36 :=
sorry

end NUMINAMATH_CALUDE_pyramid_surface_area_l2739_273960


namespace NUMINAMATH_CALUDE_pens_left_after_sale_l2739_273932

def initial_pens : ℕ := 42
def sold_pens : ℕ := 23

theorem pens_left_after_sale : initial_pens - sold_pens = 19 := by
  sorry

end NUMINAMATH_CALUDE_pens_left_after_sale_l2739_273932


namespace NUMINAMATH_CALUDE_quadratic_completion_l2739_273982

theorem quadratic_completion (x : ℝ) : ∃ (a b c : ℤ), 
  a > 0 ∧ 
  (a * x + b : ℝ)^2 = 64 * x^2 + 96 * x + c ∧
  a + b + c = 131 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_completion_l2739_273982


namespace NUMINAMATH_CALUDE_similar_triangle_sum_l2739_273948

theorem similar_triangle_sum (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : a / 3 = b / 5) (h5 : b / 5 = c / 7) (h6 : c = 21) : a + b = 24 := by
  sorry

end NUMINAMATH_CALUDE_similar_triangle_sum_l2739_273948


namespace NUMINAMATH_CALUDE_circle_diameter_ratio_l2739_273956

theorem circle_diameter_ratio (R S : Real) (harea : R^2 = 0.64 * S^2) : 
  R = 0.8 * S := by
sorry

end NUMINAMATH_CALUDE_circle_diameter_ratio_l2739_273956


namespace NUMINAMATH_CALUDE_total_squares_5x6_grid_l2739_273971

/-- The number of squares of a given size in a grid --/
def count_squares (grid_width : ℕ) (grid_height : ℕ) (square_size : ℕ) : ℕ :=
  (grid_width - square_size + 1) * (grid_height - square_size + 1)

/-- The total number of squares in a 5x6 grid --/
theorem total_squares_5x6_grid :
  let grid_width := 5
  let grid_height := 6
  (count_squares grid_width grid_height 1) +
  (count_squares grid_width grid_height 2) +
  (count_squares grid_width grid_height 3) +
  (count_squares grid_width grid_height 4) = 68 := by
  sorry

end NUMINAMATH_CALUDE_total_squares_5x6_grid_l2739_273971


namespace NUMINAMATH_CALUDE_arithmetic_sequence_m_value_l2739_273934

/-- Given an arithmetic sequence {a_n} with sum of first n terms S_n -/
def arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = n * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2

/-- The theorem stating that if S_{m-1} = -3, S_m = 0, and S_{m+1} = 5, then m = 4 -/
theorem arithmetic_sequence_m_value
  (a : ℕ → ℝ) (S : ℕ → ℝ) (m : ℕ)
  (h_arithmetic : arithmetic_sequence a S)
  (h_m_minus_1 : S (m - 1) = -3)
  (h_m : S m = 0)
  (h_m_plus_1 : S (m + 1) = 5) :
  m = 4 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_m_value_l2739_273934


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l2739_273973

theorem polynomial_divisibility (n : ℕ) :
  ∃ q : Polynomial ℤ, (X + 1 : Polynomial ℤ)^(2*n + 1) + X^(n + 2) = (X^2 + X + 1) * q := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l2739_273973


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l2739_273924

theorem fixed_point_of_exponential_function (a : ℝ) (h : a > 1) :
  let f : ℝ → ℝ := fun x ↦ a^(x - 2) + 1
  f 2 = 2 := by
sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l2739_273924


namespace NUMINAMATH_CALUDE_inscribed_circle_distance_l2739_273935

theorem inscribed_circle_distance (a b : ℝ) (ha : a = 36) (hb : b = 48) :
  let c := Real.sqrt (a^2 + b^2)
  let r := (a + b - c) / 2
  let h := a * b / c
  let d := Real.sqrt ((r * Real.sqrt 2)^2 - ((h - r) * (h - r)))
  d = 12 / 5 := by sorry

end NUMINAMATH_CALUDE_inscribed_circle_distance_l2739_273935


namespace NUMINAMATH_CALUDE_hospital_opening_date_l2739_273987

theorem hospital_opening_date :
  ∃! (x y h : ℕ+),
    (x.val : ℤ) - (y.val : ℤ) = h.val ∨ (y.val : ℤ) - (x.val : ℤ) = h.val ∧
    x * (y * h - 1) = 1539 ∧
    h = 2 :=
by sorry

end NUMINAMATH_CALUDE_hospital_opening_date_l2739_273987


namespace NUMINAMATH_CALUDE_solution_set_inequality_l2739_273992

theorem solution_set_inequality (x : ℝ) :
  {x : ℝ | |x + 1| - |x - 5| < 4} = Set.Iio 4 :=
sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l2739_273992


namespace NUMINAMATH_CALUDE_remaining_problems_to_grade_l2739_273974

theorem remaining_problems_to_grade
  (total_worksheets : ℕ)
  (graded_worksheets : ℕ)
  (problems_per_worksheet : ℕ)
  (h1 : total_worksheets = 17)
  (h2 : graded_worksheets = 8)
  (h3 : problems_per_worksheet = 7)
  : (total_worksheets - graded_worksheets) * problems_per_worksheet = 63 := by
  sorry

end NUMINAMATH_CALUDE_remaining_problems_to_grade_l2739_273974


namespace NUMINAMATH_CALUDE_negation_of_existence_is_universal_l2739_273966

theorem negation_of_existence_is_universal : 
  (¬ ∃ x : ℝ, x^2 + 2*x + 2 ≤ 0) ↔ (∀ x : ℝ, x^2 + 2*x + 2 > 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_is_universal_l2739_273966


namespace NUMINAMATH_CALUDE_tangent_line_equation_l2739_273955

/-- The curve function f(x) = x³ + 1 -/
def f (x : ℝ) : ℝ := x^3 + 1

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3 * x^2

theorem tangent_line_equation :
  let x₀ : ℝ := -1
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  ∀ x y : ℝ, (y - y₀ = m * (x - x₀)) ↔ (3 * x - y + 3 = 0) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l2739_273955


namespace NUMINAMATH_CALUDE_age_difference_l2739_273995

/-- The difference in total ages of (A, B) and (B, C) given C is 20 years younger than A -/
theorem age_difference (A B C : ℕ) (h : C = A - 20) : 
  (A + B) - (B + C) = 20 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l2739_273995


namespace NUMINAMATH_CALUDE_train_boarding_probability_l2739_273959

theorem train_boarding_probability 
  (cycle_time : ℝ) 
  (favorable_window : ℝ) 
  (h1 : cycle_time = 5) 
  (h2 : favorable_window = 0.5) 
  (h3 : 0 < favorable_window) 
  (h4 : favorable_window < cycle_time) :
  (favorable_window / cycle_time) = (1 / 10) := by
sorry

end NUMINAMATH_CALUDE_train_boarding_probability_l2739_273959


namespace NUMINAMATH_CALUDE_remaining_pages_to_read_l2739_273954

/-- Given a book where 83 pages represent 1/3 of the total, 
    the number of remaining pages to read is 166. -/
theorem remaining_pages_to_read (total_pages : ℕ) 
  (h1 : 83 = total_pages / 3) : total_pages - 83 = 166 := by
  sorry

end NUMINAMATH_CALUDE_remaining_pages_to_read_l2739_273954


namespace NUMINAMATH_CALUDE_sum_of_complex_numbers_l2739_273962

-- Define the complex numbers
def z1 (a b : ℂ) : ℂ := 2 * a + b * Complex.I
def z2 (c d : ℂ) : ℂ := c + 3 * d * Complex.I
def z3 (e f : ℂ) : ℂ := e + f * Complex.I

-- State the theorem
theorem sum_of_complex_numbers (a b c d e f : ℂ) :
  b = 4 →
  e = -2 * a - c →
  z1 a b + z2 c d + z3 e f = 6 * Complex.I →
  d + f = 2 := by
  sorry


end NUMINAMATH_CALUDE_sum_of_complex_numbers_l2739_273962


namespace NUMINAMATH_CALUDE_no_bounded_function_satisfying_conditions_l2739_273980

theorem no_bounded_function_satisfying_conditions :
  ¬ ∃ (f : ℝ → ℝ), 
    (∃ (M : ℝ), ∀ x, |f x| ≤ M) ∧ 
    (f 1 > 0) ∧ 
    (∀ x y, (f (x + y))^2 ≥ (f x)^2 + 2 * f (x * y) + (f y)^2) :=
by sorry

end NUMINAMATH_CALUDE_no_bounded_function_satisfying_conditions_l2739_273980


namespace NUMINAMATH_CALUDE_shaded_area_calculation_l2739_273957

/-- Represents a rectangle with its diagonal divided into 12 equal segments -/
structure DividedRectangle where
  totalSegments : ℕ
  nonShadedArea : ℝ

/-- Calculates the area of shaded parts in a divided rectangle -/
def shadedArea (rect : DividedRectangle) : ℝ :=
  sorry

/-- Theorem stating the relationship between non-shaded and shaded areas -/
theorem shaded_area_calculation (rect : DividedRectangle) 
  (h1 : rect.totalSegments = 12)
  (h2 : rect.nonShadedArea = 10) :
  shadedArea rect = 14 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_calculation_l2739_273957


namespace NUMINAMATH_CALUDE_arcsin_of_one_l2739_273919

theorem arcsin_of_one : Real.arcsin 1 = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_arcsin_of_one_l2739_273919


namespace NUMINAMATH_CALUDE_polar_to_rectangular_l2739_273986

/-- Given a point with polar coordinates (3, π/4), prove that its rectangular coordinates are (3√2/2, 3√2/2) -/
theorem polar_to_rectangular :
  let r : ℝ := 3
  let θ : ℝ := π / 4
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  x = 3 * Real.sqrt 2 / 2 ∧ y = 3 * Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_l2739_273986


namespace NUMINAMATH_CALUDE_people_left_of_kolya_l2739_273996

/-- Given a line of people with the following conditions:
  * There are 12 people to the right of Kolya
  * There are 20 people to the left of Sasha
  * There are 8 people to the right of Sasha
  Then there are 16 people to the left of Kolya -/
theorem people_left_of_kolya 
  (total : ℕ) 
  (kolya_right : ℕ) 
  (sasha_left : ℕ) 
  (sasha_right : ℕ) 
  (h1 : kolya_right = 12)
  (h2 : sasha_left = 20)
  (h3 : sasha_right = 8)
  (h4 : total = sasha_left + sasha_right + 1) : 
  total - kolya_right - 1 = 16 := by
sorry

end NUMINAMATH_CALUDE_people_left_of_kolya_l2739_273996


namespace NUMINAMATH_CALUDE_alberts_brother_age_difference_l2739_273964

/-- Proves that Albert's brother is 2 years younger than Albert given the problem conditions -/
theorem alberts_brother_age_difference : ℕ → Prop :=
  fun albert_age : ℕ =>
    ∀ (father_age mother_age brother_age : ℕ),
      father_age = albert_age + 48 →
      mother_age = brother_age + 46 →
      father_age = mother_age + 4 →
      brother_age < albert_age →
      albert_age - brother_age = 2

/-- Proof of the theorem -/
lemma prove_alberts_brother_age_difference :
  ∀ albert_age : ℕ, alberts_brother_age_difference albert_age :=
by
  sorry

#check prove_alberts_brother_age_difference

end NUMINAMATH_CALUDE_alberts_brother_age_difference_l2739_273964


namespace NUMINAMATH_CALUDE_square_area_ratio_l2739_273905

theorem square_area_ratio (y : ℝ) (y_pos : y > 0) : 
  (3 * y)^2 / (12 * y)^2 = 1 / 16 :=
by sorry

end NUMINAMATH_CALUDE_square_area_ratio_l2739_273905


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l2739_273906

/-- A point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the second quadrant -/
def second_quadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- The point (-1,2) -/
def point : Point :=
  { x := -1, y := 2 }

theorem point_in_second_quadrant : second_quadrant point := by
  sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_l2739_273906


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l2739_273912

theorem rectangle_perimeter (a b : ℝ) (h1 : a + b = 7) (h2 : 2 * a + b = 9.5) :
  2 * (a + b) = 14 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l2739_273912


namespace NUMINAMATH_CALUDE_f_extrema_l2739_273967

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 + 3 * x^2 - 12 * x + 3

theorem f_extrema :
  let a : ℝ := -3
  let b : ℝ := 3
  (∀ x ∈ Set.Icc a b, f x ≤ 48) ∧
  (∃ x ∈ Set.Icc a b, f x = 48) ∧
  (∀ x ∈ Set.Icc a b, f x ≥ -4) ∧
  (∃ x ∈ Set.Icc a b, f x = -4) :=
by sorry

end NUMINAMATH_CALUDE_f_extrema_l2739_273967


namespace NUMINAMATH_CALUDE_simplify_expression_l2739_273904

theorem simplify_expression (x : ℝ) : 2 * x + 1 - (x + 1) = x := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2739_273904


namespace NUMINAMATH_CALUDE_trioball_playing_time_l2739_273916

theorem trioball_playing_time (total_children : ℕ) (playing_children : ℕ) (total_time : ℕ) 
  (h1 : total_children = 6)
  (h2 : playing_children = 3)
  (h3 : total_time = 180) :
  (total_time * playing_children) / total_children = 90 := by
  sorry

end NUMINAMATH_CALUDE_trioball_playing_time_l2739_273916


namespace NUMINAMATH_CALUDE_launderette_machine_count_l2739_273949

/-- Represents a laundry machine with quarters and dimes -/
structure LaundryMachine where
  quarters : ℕ
  dimes : ℕ

/-- Calculates the value of a laundry machine in cents -/
def machine_value (m : LaundryMachine) : ℕ :=
  m.quarters * 25 + m.dimes * 10

/-- Represents the launderette -/
structure Launderette where
  machine : LaundryMachine
  total_value : ℕ
  machine_count : ℕ

/-- Theorem: The number of machines in the launderette is 3 -/
theorem launderette_machine_count (l : Launderette) 
  (h1 : l.machine.quarters = 80)
  (h2 : l.machine.dimes = 100)
  (h3 : l.total_value = 9000) -- $90 in cents
  (h4 : l.machine_count * machine_value l.machine = l.total_value) :
  l.machine_count = 3 := by
  sorry

end NUMINAMATH_CALUDE_launderette_machine_count_l2739_273949


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2739_273963

theorem complex_equation_solution (Z : ℂ) :
  (1 + 2*Complex.I)^3 * Z = 1 + 2*Complex.I →
  Z = -3/25 + 24/125*Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2739_273963


namespace NUMINAMATH_CALUDE_election_result_l2739_273938

theorem election_result (total_votes : ℕ) (second_candidate_votes : ℕ) :
  total_votes = 1200 →
  second_candidate_votes = 480 →
  (total_votes - second_candidate_votes) / total_votes = 3 / 5 := by
sorry

end NUMINAMATH_CALUDE_election_result_l2739_273938


namespace NUMINAMATH_CALUDE_max_min_difference_c_l2739_273929

theorem max_min_difference_c (a b c : ℝ) 
  (sum_eq : a + b + c = 5)
  (sum_squares_eq : a^2 + b^2 + c^2 = 27) :
  ∃ (c_max c_min : ℝ),
    (∀ c' : ℝ, (∃ a' b' : ℝ, a' + b' + c' = 5 ∧ a'^2 + b'^2 + c'^2 = 27) → c' ≤ c_max) ∧
    (∀ c' : ℝ, (∃ a' b' : ℝ, a' + b' + c' = 5 ∧ a'^2 + b'^2 + c'^2 = 27) → c_min ≤ c') ∧
    c_max - c_min = 22/3 :=
sorry

end NUMINAMATH_CALUDE_max_min_difference_c_l2739_273929
