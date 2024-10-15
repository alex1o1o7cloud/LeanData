import Mathlib

namespace NUMINAMATH_CALUDE_infinite_grid_graph_chromatic_number_infinite_grid_graph_chromatic_number_lower_bound_infinite_grid_graph_chromatic_number_exact_l1086_108666

/-- An infinite grid graph -/
def InfiniteGridGraph : Type := ℤ × ℤ

/-- A coloring function for the infinite grid graph -/
def Coloring (G : Type) := G → Fin 2

/-- A valid coloring of the infinite grid graph -/
def IsValidColoring (c : Coloring InfiniteGridGraph) : Prop :=
  ∀ (x y : ℤ), (x + y) % 2 = c (x, y)

/-- The chromatic number of the infinite grid graph is at most 2 -/
theorem infinite_grid_graph_chromatic_number :
  ∃ (c : Coloring InfiniteGridGraph), IsValidColoring c :=
sorry

/-- The chromatic number of the infinite grid graph is at least 2 -/
theorem infinite_grid_graph_chromatic_number_lower_bound :
  ¬∃ (c : InfiniteGridGraph → Fin 1), 
    ∀ (x y : ℤ), c (x, y) ≠ c (x + 1, y) ∨ c (x, y) ≠ c (x, y + 1) :=
sorry

/-- The chromatic number of the infinite grid graph is exactly 2 -/
theorem infinite_grid_graph_chromatic_number_exact : 
  (∃ (c : Coloring InfiniteGridGraph), IsValidColoring c) ∧
  (¬∃ (c : InfiniteGridGraph → Fin 1), 
    ∀ (x y : ℤ), c (x, y) ≠ c (x + 1, y) ∨ c (x, y) ≠ c (x, y + 1)) :=
sorry

end NUMINAMATH_CALUDE_infinite_grid_graph_chromatic_number_infinite_grid_graph_chromatic_number_lower_bound_infinite_grid_graph_chromatic_number_exact_l1086_108666


namespace NUMINAMATH_CALUDE_triangle_area_implies_q_value_l1086_108680

/-- Given a triangle ABC with vertices A(3, 15), B(15, 0), and C(0, q), 
    prove that if the area of the triangle is 50, then q = 125/12 -/
theorem triangle_area_implies_q_value (q : ℝ) : 
  let A : ℝ × ℝ := (3, 15)
  let B : ℝ × ℝ := (15, 0)
  let C : ℝ × ℝ := (0, q)
  let triangle_area := (abs ((A.1 - C.1) * (B.2 - C.2) - (B.1 - C.1) * (A.2 - C.2))) / 2
  triangle_area = 50 → q = 125 / 12 := by
  sorry

#check triangle_area_implies_q_value

end NUMINAMATH_CALUDE_triangle_area_implies_q_value_l1086_108680


namespace NUMINAMATH_CALUDE_midpoint_coordinate_sum_l1086_108624

/-- The sum of the coordinates of the midpoint of a segment with endpoints (10, 7) and (4, -3) is 9 -/
theorem midpoint_coordinate_sum : 
  let p1 : ℝ × ℝ := (10, 7)
  let p2 : ℝ × ℝ := (4, -3)
  let midpoint : ℝ × ℝ := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  midpoint.1 + midpoint.2 = 9 := by sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_sum_l1086_108624


namespace NUMINAMATH_CALUDE_smallest_tangent_circle_slope_l1086_108681

/-- Circle ω₁ -/
def ω₁ (x y : ℝ) : Prop := x^2 + y^2 + 12*x - 20*y - 100 = 0

/-- Circle ω₂ -/
def ω₂ (x y : ℝ) : Prop := x^2 + y^2 - 12*x - 20*y + 196 = 0

/-- A circle is externally tangent to ω₂ -/
def externally_tangent_ω₂ (x y r : ℝ) : Prop :=
  r + 8 = Real.sqrt ((x - 6)^2 + (y - 10)^2)

/-- A circle is internally tangent to ω₁ -/
def internally_tangent_ω₁ (x y r : ℝ) : Prop :=
  16 - r = Real.sqrt ((x + 6)^2 + (y - 10)^2)

/-- The main theorem -/
theorem smallest_tangent_circle_slope :
  ∃ (m : ℝ), m > 0 ∧ m^2 = 160/99 ∧
  (∀ (a : ℝ), a > 0 → a < m →
    ¬∃ (x y r : ℝ), y = a*x ∧
      externally_tangent_ω₂ x y r ∧
      internally_tangent_ω₁ x y r) ∧
  (∃ (x y r : ℝ), y = m*x ∧
    externally_tangent_ω₂ x y r ∧
    internally_tangent_ω₁ x y r) :=
sorry

end NUMINAMATH_CALUDE_smallest_tangent_circle_slope_l1086_108681


namespace NUMINAMATH_CALUDE_expected_regions_100_l1086_108698

/-- The number of points on the circle -/
def n : ℕ := 100

/-- The probability that two randomly chosen chords intersect inside the circle -/
def p_intersect : ℚ := 1/3

/-- The expected number of regions bounded by straight lines when n points are picked 
    independently and uniformly at random on a circle, and connected by line segments -/
def expected_regions (n : ℕ) : ℚ :=
  1 + p_intersect * (n.choose 2 - 3 * n)

theorem expected_regions_100 : 
  expected_regions n = 1651 := by sorry

end NUMINAMATH_CALUDE_expected_regions_100_l1086_108698


namespace NUMINAMATH_CALUDE_abc_divides_sum_power_seven_l1086_108614

theorem abc_divides_sum_power_seven
  (a b c : ℕ+)
  (h1 : a ∣ b^2)
  (h2 : b ∣ c^2)
  (h3 : c ∣ a^2) :
  (a * b * c) ∣ (a + b + c)^7 :=
by sorry

end NUMINAMATH_CALUDE_abc_divides_sum_power_seven_l1086_108614


namespace NUMINAMATH_CALUDE_average_age_of_nine_students_l1086_108657

theorem average_age_of_nine_students (total_students : ℕ) (total_average : ℝ) 
  (five_students : ℕ) (five_average : ℝ) (seventeenth_age : ℝ) 
  (nine_students : ℕ) (h1 : total_students = 17) 
  (h2 : total_average = 17) (h3 : five_students = 5) 
  (h4 : five_average = 14) (h5 : seventeenth_age = 75) 
  (h6 : nine_students = total_students - five_students - 1) :
  (total_students * total_average - five_students * five_average - seventeenth_age) / nine_students = 16 := by
  sorry

#check average_age_of_nine_students

end NUMINAMATH_CALUDE_average_age_of_nine_students_l1086_108657


namespace NUMINAMATH_CALUDE_f_properties_l1086_108622

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then x - Real.sqrt x + 1
  else if x = 0 then 0
  else x + Real.sqrt (-x) - 1

-- State the properties of f
theorem f_properties :
  (∀ x, f (-x) = -f x) ∧  -- f is odd
  (∀ x > 0, f x = x - Real.sqrt x + 1) ∧  -- given definition for x > 0
  (Set.range f = {y | y ≥ 3/4 ∨ y ≤ -3/4 ∨ y = 0}) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l1086_108622


namespace NUMINAMATH_CALUDE_day_crew_loading_fraction_l1086_108647

/-- The fraction of boxes loaded by the day crew given the relative capacities of night and day crews -/
theorem day_crew_loading_fraction 
  (D : ℝ) -- number of boxes loaded by each day crew worker
  (W : ℝ) -- number of workers in the day crew
  (h1 : D > 0) -- assume positive number of boxes
  (h2 : W > 0) -- assume positive number of workers
  : (D * W) / ((D * W) + ((3/4 * D) * (5/6 * W))) = 8/13 := by
  sorry

end NUMINAMATH_CALUDE_day_crew_loading_fraction_l1086_108647


namespace NUMINAMATH_CALUDE_walnut_problem_l1086_108687

theorem walnut_problem (a b c : ℕ) : 
  28 * a + 30 * b + 31 * c = 365 → a + b + c = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_walnut_problem_l1086_108687


namespace NUMINAMATH_CALUDE_percentage_difference_l1086_108696

theorem percentage_difference (total : ℝ) (z_share : ℝ) (x_premium : ℝ) : 
  total = 555 → z_share = 150 → x_premium = 0.25 →
  ∃ y_share : ℝ, 
    y_share = (total - z_share) / (2 + x_premium) ∧
    (y_share - z_share) / z_share = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l1086_108696


namespace NUMINAMATH_CALUDE_race_total_time_l1086_108602

theorem race_total_time (total_runners : ℕ) (first_group : ℕ) (first_time : ℕ) (extra_time : ℕ) 
  (h1 : total_runners = 8)
  (h2 : first_group = 5)
  (h3 : first_time = 8)
  (h4 : extra_time = 2) :
  first_group * first_time + (total_runners - first_group) * (first_time + extra_time) = 70 := by
  sorry

end NUMINAMATH_CALUDE_race_total_time_l1086_108602


namespace NUMINAMATH_CALUDE_lava_lamp_probability_l1086_108693

/-- The number of red lava lamps -/
def num_red : ℕ := 4

/-- The number of blue lava lamps -/
def num_blue : ℕ := 4

/-- The number of green lava lamps -/
def num_green : ℕ := 4

/-- The total number of lava lamps -/
def total_lamps : ℕ := num_red + num_blue + num_green

/-- The number of lamps that are turned on -/
def num_on : ℕ := 6

/-- The probability of the leftmost lamp being green and off, and the rightmost lamp being blue and on -/
def prob_specific_arrangement : ℚ := 80 / 1313

theorem lava_lamp_probability :
  prob_specific_arrangement = (Nat.choose (total_lamps - 2) num_red * Nat.choose (total_lamps - 2 - num_red) (num_blue - 1) * Nat.choose (total_lamps - 1) (num_on - 1)) /
  (Nat.choose total_lamps num_red * Nat.choose (total_lamps - num_red) num_blue * Nat.choose total_lamps num_on) :=
sorry

end NUMINAMATH_CALUDE_lava_lamp_probability_l1086_108693


namespace NUMINAMATH_CALUDE_roots_greater_than_three_l1086_108626

/-- For a quadratic equation x^2 - 6ax + (2 - 2a + 9a^2) = 0, both roots are greater than 3 
    if and only if a > 11/9 -/
theorem roots_greater_than_three (a : ℝ) : 
  (∀ x : ℝ, x^2 - 6*a*x + (2 - 2*a + 9*a^2) = 0 → x > 3) ↔ a > 11/9 := by
  sorry

end NUMINAMATH_CALUDE_roots_greater_than_three_l1086_108626


namespace NUMINAMATH_CALUDE_common_roots_cubic_polynomials_l1086_108619

theorem common_roots_cubic_polynomials (a b : ℝ) :
  (∃ r s : ℝ, r ≠ s ∧
    r^3 + a*r^2 + 13*r + 12 = 0 ∧
    r^3 + b*r^2 + 17*r + 15 = 0 ∧
    s^3 + a*s^2 + 13*s + 12 = 0 ∧
    s^3 + b*s^2 + 17*s + 15 = 0) →
  a = 0 ∧ b = -1 := by
sorry

end NUMINAMATH_CALUDE_common_roots_cubic_polynomials_l1086_108619


namespace NUMINAMATH_CALUDE_sqrt_floor_equality_l1086_108668

theorem sqrt_floor_equality (n : ℕ+) :
  ⌊Real.sqrt n + Real.sqrt (n + 1)⌋ = ⌊Real.sqrt (4 * n + 1)⌋ ∧
  ⌊Real.sqrt (4 * n + 1)⌋ = ⌊Real.sqrt (4 * n + 2)⌋ ∧
  ⌊Real.sqrt (4 * n + 2)⌋ = ⌊Real.sqrt (4 * n + 3)⌋ := by
  sorry

end NUMINAMATH_CALUDE_sqrt_floor_equality_l1086_108668


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1086_108621

theorem quadratic_equation_solution : ∃ (x₁ x₂ : ℝ), 
  x₁^2 - 6*x₁ + 8 = 0 ∧ 
  x₂^2 - 6*x₂ + 8 = 0 ∧ 
  x₁ = 2 ∧ 
  x₂ = 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1086_108621


namespace NUMINAMATH_CALUDE_max_value_abc_l1086_108613

theorem max_value_abc (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = 1) :
  a + a * b + a * b * c ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_max_value_abc_l1086_108613


namespace NUMINAMATH_CALUDE_distance_to_school_is_two_prove_distance_to_school_l1086_108634

/-- The distance to school in miles -/
def distance_to_school : ℝ := 2

/-- Jerry's one-way trip time in minutes -/
def jerry_one_way_time : ℝ := 15

/-- Carson's speed in miles per hour -/
def carson_speed : ℝ := 8

/-- Theorem stating that the distance to school is 2 miles -/
theorem distance_to_school_is_two :
  distance_to_school = 2 :=
by
  sorry

/-- Lemma: Jerry's round trip time equals Carson's one-way trip time -/
lemma jerry_round_trip_equals_carson_one_way :
  2 * jerry_one_way_time = distance_to_school / (carson_speed / 60) :=
by
  sorry

/-- Main theorem proving the distance to school -/
theorem prove_distance_to_school :
  distance_to_school = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_distance_to_school_is_two_prove_distance_to_school_l1086_108634


namespace NUMINAMATH_CALUDE_inequality_system_integer_solutions_l1086_108650

theorem inequality_system_integer_solutions :
  {x : ℤ | 2 * x + 1 > 0 ∧ 2 * x ≤ 4} = {0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_integer_solutions_l1086_108650


namespace NUMINAMATH_CALUDE_y_expression_equivalence_l1086_108620

theorem y_expression_equivalence (x : ℝ) : 
  Real.sqrt ((x - 2)^2) + Real.sqrt (x^2 + 4*x + 5) = 
  |x - 2| + Real.sqrt ((x + 2)^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_y_expression_equivalence_l1086_108620


namespace NUMINAMATH_CALUDE_vector_at_negative_one_l1086_108653

/-- A line parameterized by t in 3D space -/
structure ParametricLine where
  -- The vector on the line at t = 0
  origin : Fin 3 → ℝ
  -- The vector on the line at t = 1
  point_at_one : Fin 3 → ℝ

/-- The vector on the line at a given t -/
def vector_at_t (line : ParametricLine) (t : ℝ) : Fin 3 → ℝ :=
  λ i => line.origin i + t * (line.point_at_one i - line.origin i)

/-- The theorem stating the vector at t = -1 for the given line -/
theorem vector_at_negative_one (line : ParametricLine) 
  (h0 : line.origin = λ i => [2, 4, 9].get i)
  (h1 : line.point_at_one = λ i => [3, 1, 5].get i) :
  vector_at_t line (-1) = λ i => [1, 7, 13].get i := by
  sorry

end NUMINAMATH_CALUDE_vector_at_negative_one_l1086_108653


namespace NUMINAMATH_CALUDE_cookie_distribution_l1086_108609

theorem cookie_distribution (num_people : ℕ) (cookies_per_person : ℕ) 
  (h1 : num_people = 5)
  (h2 : cookies_per_person = 7) : 
  num_people * cookies_per_person = 35 := by
sorry

end NUMINAMATH_CALUDE_cookie_distribution_l1086_108609


namespace NUMINAMATH_CALUDE_line_intersection_l1086_108664

theorem line_intersection :
  ∃! (x y : ℚ), (8 * x - 5 * y = 40) ∧ (6 * x - y = -5) ∧ (x = 15/38) ∧ (y = 140/19) := by
  sorry

end NUMINAMATH_CALUDE_line_intersection_l1086_108664


namespace NUMINAMATH_CALUDE_number_in_different_bases_l1086_108651

theorem number_in_different_bases : ∃ (n : ℕ), 
  (∃ (a : ℕ), a < 7 ∧ n = a * 7 + 0) ∧ 
  (∃ (a b : ℕ), a < 9 ∧ b < 9 ∧ a ≠ b ∧ n = a * 9 + b) ∧ 
  (n = 3 * 8 + 5) := by
  sorry

end NUMINAMATH_CALUDE_number_in_different_bases_l1086_108651


namespace NUMINAMATH_CALUDE_apartment_number_theorem_l1086_108608

/-- The number of apartments on each floor (actual) -/
def apartments_per_floor : ℕ := 7

/-- The number of apartments Anna initially thought were on each floor -/
def assumed_apartments_per_floor : ℕ := 6

/-- The floor number where Anna's apartment is located -/
def target_floor : ℕ := 4

/-- The set of possible apartment numbers on the target floor when there are 6 apartments per floor -/
def apartment_numbers_6 : Set ℕ := Set.Icc ((target_floor - 1) * assumed_apartments_per_floor + 1) (target_floor * assumed_apartments_per_floor)

/-- The set of possible apartment numbers on the target floor when there are 7 apartments per floor -/
def apartment_numbers_7 : Set ℕ := Set.Icc ((target_floor - 1) * apartments_per_floor + 1) (target_floor * apartments_per_floor)

/-- The set of apartment numbers that exist in both scenarios -/
def possible_apartment_numbers : Set ℕ := apartment_numbers_6 ∩ apartment_numbers_7

theorem apartment_number_theorem : possible_apartment_numbers = {22, 23, 24} := by
  sorry

end NUMINAMATH_CALUDE_apartment_number_theorem_l1086_108608


namespace NUMINAMATH_CALUDE_certain_number_proof_l1086_108627

theorem certain_number_proof : ∃ n : ℕ, (73 * n) % 8 = 7 ∧ n = 7 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l1086_108627


namespace NUMINAMATH_CALUDE_equation_solutions_l1086_108688

theorem equation_solutions :
  (∃ x : ℚ, 8 * x = -2 * (x + 5) ∧ x = -1) ∧
  (∃ x : ℚ, (x - 1) / 4 = (5 * x - 7) / 6 + 1 ∧ x = -1 / 7) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l1086_108688


namespace NUMINAMATH_CALUDE_sofia_shopping_cost_l1086_108679

/-- The cost of all items Sofia buys at the department store -/
theorem sofia_shopping_cost :
  let shirt_cost : ℕ := 7
  let shoes_cost : ℕ := shirt_cost + 3
  let two_shirts_and_shoes_cost : ℕ := 2 * shirt_cost + shoes_cost
  let bag_cost : ℕ := two_shirts_and_shoes_cost / 2
  let total_cost : ℕ := 2 * shirt_cost + shoes_cost + bag_cost
  total_cost = 36 := by
  sorry

end NUMINAMATH_CALUDE_sofia_shopping_cost_l1086_108679


namespace NUMINAMATH_CALUDE_f_of_f_has_four_roots_l1086_108607

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 3*x + 2

-- State the theorem
theorem f_of_f_has_four_roots :
  ∃ (a b c d : ℝ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  (∀ x : ℝ, f (f x) = 0 ↔ x = a ∨ x = b ∨ x = c ∨ x = d) :=
sorry

end NUMINAMATH_CALUDE_f_of_f_has_four_roots_l1086_108607


namespace NUMINAMATH_CALUDE_linda_cookie_distribution_l1086_108699

/-- Calculates the number of cookies per student given the problem conditions -/
def cookies_per_student (classmates : ℕ) (cookies_per_batch : ℕ) 
  (choc_chip_batches : ℕ) (oatmeal_batches : ℕ) (additional_batches : ℕ) : ℕ :=
  let total_cookies := (choc_chip_batches + oatmeal_batches + additional_batches) * cookies_per_batch
  total_cookies / classmates

/-- Proves that given the problem conditions, each student receives 10 cookies -/
theorem linda_cookie_distribution : 
  cookies_per_student 24 (4 * 12) 2 1 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_linda_cookie_distribution_l1086_108699


namespace NUMINAMATH_CALUDE_log_equation_solution_l1086_108631

-- Define the logarithm function for base 2
noncomputable def log2 (x : ℝ) := Real.log x / Real.log 2

-- State the theorem
theorem log_equation_solution :
  ∃ x : ℝ, log2 (x + 3) + 2 * log2 5 = 4 ∧ x = -59 / 25 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l1086_108631


namespace NUMINAMATH_CALUDE_investment_sum_l1086_108667

/-- Given a sum of money invested for 2 years, if increasing the interest rate by 3% 
    results in 300 more rupees of interest, then the original sum invested must be 5000 rupees. -/
theorem investment_sum (P : ℝ) (R : ℝ) : 
  (P * (R + 3) * 2) / 100 = (P * R * 2) / 100 + 300 → P = 5000 := by
sorry

end NUMINAMATH_CALUDE_investment_sum_l1086_108667


namespace NUMINAMATH_CALUDE_triangle_angles_theorem_l1086_108683

noncomputable def triangle_angles (a b c : ℝ) : ℝ × ℝ × ℝ := sorry

theorem triangle_angles_theorem :
  let side1 := 3
  let side2 := 3
  let side3 := Real.sqrt 8 - Real.sqrt 3
  let (angle_A, angle_B, angle_C) := triangle_angles side1 side2 side3
  angle_C = Real.arccos ((7 / 18) + (2 * Real.sqrt 6 / 9)) ∧
  angle_A = (π - angle_C) / 2 ∧
  angle_B = (π - angle_C) / 2 :=
sorry

end NUMINAMATH_CALUDE_triangle_angles_theorem_l1086_108683


namespace NUMINAMATH_CALUDE_prism_intersection_area_l1086_108662

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

/-- Represents a rectangular prism -/
structure RectangularPrism where
  width : ℝ
  length : ℝ
  height : ℝ

/-- Calculate the area of intersection between a rectangular prism and a plane -/
def intersectionArea (prism : RectangularPrism) (plane : Plane) : ℝ :=
  sorry

theorem prism_intersection_area :
  let prism : RectangularPrism := ⟨8, 12, 0⟩  -- height is arbitrary, set to 0
  let plane : Plane := ⟨3, -5, 6, 30⟩
  intersectionArea prism plane = 64.92 := by sorry

end NUMINAMATH_CALUDE_prism_intersection_area_l1086_108662


namespace NUMINAMATH_CALUDE_sam_age_two_years_ago_l1086_108673

def john_age (sam_age : ℕ) : ℕ := 3 * sam_age

theorem sam_age_two_years_ago (sam_current_age : ℕ) : 
  john_age sam_current_age = 3 * sam_current_age ∧ 
  john_age sam_current_age + 9 = 2 * (sam_current_age + 9) →
  sam_current_age - 2 = 7 := by
sorry

end NUMINAMATH_CALUDE_sam_age_two_years_ago_l1086_108673


namespace NUMINAMATH_CALUDE_smallest_number_with_same_prime_factors_l1086_108692

def is_prime_factor (p n : ℕ) : Prop :=
  Nat.Prime p ∧ n % p = 0

def has_all_prime_factors (m n : ℕ) : Prop :=
  ∀ p, is_prime_factor p n → is_prime_factor p m

theorem smallest_number_with_same_prime_factors (n : ℕ) (hn : n = 36) :
  ∃ m : ℕ, m = 6 ∧
    has_all_prime_factors m n ∧
    ∀ k : ℕ, k < m → ¬(has_all_prime_factors k n) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_same_prime_factors_l1086_108692


namespace NUMINAMATH_CALUDE_chess_tournament_square_players_l1086_108601

/-- Represents a chess tournament with men and women players -/
structure ChessTournament where
  k : ℕ  -- number of men
  m : ℕ  -- number of women

/-- The total number of players in the tournament -/
def ChessTournament.totalPlayers (t : ChessTournament) : ℕ := t.k + t.m

/-- The condition that total points scored by men against women equals total points scored by women against men -/
def ChessTournament.equalCrossScores (t : ChessTournament) : Prop :=
  (t.k * (t.k - 1)) / 2 + (t.m * (t.m - 1)) / 2 = t.k * t.m

theorem chess_tournament_square_players (t : ChessTournament) 
  (h : t.equalCrossScores) : 
  ∃ n : ℕ, t.totalPlayers = n^2 := by
  sorry


end NUMINAMATH_CALUDE_chess_tournament_square_players_l1086_108601


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l1086_108616

theorem fractional_equation_solution :
  ∃ x : ℝ, (x / (x + 2) + 4 / (x^2 - 4) = 1) ∧ (x = 4) := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l1086_108616


namespace NUMINAMATH_CALUDE_f_derivative_at_1_l1086_108691

-- Define the function f
def f (x : ℝ) : ℝ := (2023 - 2022 * x) ^ 3

-- State the theorem
theorem f_derivative_at_1 : 
  (deriv f) 1 = -6066 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_1_l1086_108691


namespace NUMINAMATH_CALUDE_bottle_t_cost_l1086_108697

/-- The cost of Bottle T given the conditions of the problem -/
theorem bottle_t_cost :
  let bottle_r_capsules : ℕ := 250
  let bottle_r_cost : ℚ := 625 / 100  -- $6.25 represented as a rational number
  let bottle_t_capsules : ℕ := 100
  let cost_per_capsule_diff : ℚ := 5 / 1000  -- $0.005 represented as a rational number
  let bottle_r_cost_per_capsule : ℚ := bottle_r_cost / bottle_r_capsules
  let bottle_t_cost_per_capsule : ℚ := bottle_r_cost_per_capsule - cost_per_capsule_diff
  bottle_t_cost_per_capsule * bottle_t_capsules = 2 := by
sorry

end NUMINAMATH_CALUDE_bottle_t_cost_l1086_108697


namespace NUMINAMATH_CALUDE_bella_ella_meeting_l1086_108636

/-- The distance between Bella's and Ella's houses in feet -/
def distance : ℕ := 15840

/-- The length of Bella's step in feet -/
def step_length : ℕ := 3

/-- The ratio of Ella's speed to Bella's speed -/
def speed_ratio : ℕ := 5

/-- The number of steps Bella takes before meeting Ella -/
def steps_taken : ℕ := 880

theorem bella_ella_meeting :
  distance = 15840 ∧
  step_length = 3 ∧
  speed_ratio = 5 →
  steps_taken = 880 :=
by sorry

end NUMINAMATH_CALUDE_bella_ella_meeting_l1086_108636


namespace NUMINAMATH_CALUDE_hexagon_area_ratio_l1086_108695

/-- A regular hexagon -/
structure RegularHexagon where
  vertices : Fin 6 → ℝ × ℝ

/-- A point on a side of the hexagon -/
def SidePoint (h : RegularHexagon) (i : Fin 6) := ℝ × ℝ

/-- The ratio of areas of two polygons -/
def AreaRatio (p1 p2 : Set (ℝ × ℝ)) : ℚ := sorry

theorem hexagon_area_ratio 
  (ABCDEF : RegularHexagon)
  (P : SidePoint ABCDEF 0) (Q : SidePoint ABCDEF 1) (R : SidePoint ABCDEF 2)
  (S : SidePoint ABCDEF 3) (T : SidePoint ABCDEF 4) (U : SidePoint ABCDEF 5)
  (h_P : P = (2/3 : ℝ) • ABCDEF.vertices 0 + (1/3 : ℝ) • ABCDEF.vertices 1)
  (h_Q : Q = (2/3 : ℝ) • ABCDEF.vertices 1 + (1/3 : ℝ) • ABCDEF.vertices 2)
  (h_R : R = (2/3 : ℝ) • ABCDEF.vertices 2 + (1/3 : ℝ) • ABCDEF.vertices 3)
  (h_S : S = (2/3 : ℝ) • ABCDEF.vertices 3 + (1/3 : ℝ) • ABCDEF.vertices 4)
  (h_T : T = (2/3 : ℝ) • ABCDEF.vertices 4 + (1/3 : ℝ) • ABCDEF.vertices 5)
  (h_U : U = (2/3 : ℝ) • ABCDEF.vertices 5 + (1/3 : ℝ) • ABCDEF.vertices 0) :
  let inner_hexagon := {ABCDEF.vertices 0, R, ABCDEF.vertices 2, T, ABCDEF.vertices 4, P}
  let outer_hexagon := {ABCDEF.vertices i | i : Fin 6}
  AreaRatio inner_hexagon outer_hexagon = 4/9 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_area_ratio_l1086_108695


namespace NUMINAMATH_CALUDE_sum_always_four_digits_l1086_108652

theorem sum_always_four_digits :
  ∀ (A B : ℕ), 1 ≤ A ∧ A ≤ 9 → 1 ≤ B ∧ B ≤ 9 →
  ∃ (n : ℕ), 1000 ≤ n ∧ n < 10000 ∧ n = 7654 + (900 + 10 * A + 7) + (10 + B) :=
by sorry

end NUMINAMATH_CALUDE_sum_always_four_digits_l1086_108652


namespace NUMINAMATH_CALUDE_problem_statement_l1086_108656

theorem problem_statement (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a^2 + b^2 = 1) : 
  (abs a + abs b ≤ Real.sqrt 2) ∧ (abs (a^3 / b) + abs (b^3 / a) ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1086_108656


namespace NUMINAMATH_CALUDE_condition1_implies_bijective_condition2_implies_bijective_condition3_implies_not_injective_not_surjective_condition4_not_necessarily_injective_or_surjective_l1086_108663

-- Define the function type
def RealFunction := ℝ → ℝ

-- Define the properties
def Injective (f : RealFunction) : Prop :=
  ∀ x y, f x = f y → x = y

def Surjective (f : RealFunction) : Prop :=
  ∀ y, ∃ x, f x = y

def Bijective (f : RealFunction) : Prop :=
  Injective f ∧ Surjective f

-- Theorem statements
theorem condition1_implies_bijective (f : RealFunction) 
  (h : ∀ x, f (f x - 1) = x + 1) : Bijective f := by sorry

theorem condition2_implies_bijective (f : RealFunction) 
  (h : ∀ x y, f (x + f y) = f x + y^5) : Bijective f := by sorry

theorem condition3_implies_not_injective_not_surjective (f : RealFunction) 
  (h : ∀ x, f (f x) = Real.sin x) : ¬(Injective f) ∧ ¬(Surjective f) := by sorry

theorem condition4_not_necessarily_injective_or_surjective : 
  ∃ f : RealFunction, (∀ x y, f (x + y^2) = f x * f y + x * f y - y^3 * f x) ∧ 
  ¬(Injective f) ∧ ¬(Surjective f) := by sorry

end NUMINAMATH_CALUDE_condition1_implies_bijective_condition2_implies_bijective_condition3_implies_not_injective_not_surjective_condition4_not_necessarily_injective_or_surjective_l1086_108663


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1086_108632

/-- Given a hyperbola and a circle with specific properties, prove the equation of the hyperbola -/
theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (∀ x y : ℝ, x^2/a^2 - y^2/b^2 = 1 → 
    (∃ t : ℝ, (b*x + a*y = 0 ∨ b*x - a*y = 0) ∧ 
      ((x - 3)^2 + y^2 = 4 ↔ t = 0))) → 
  (∃ c : ℝ, c > 0 ∧ c^2 = a^2 + b^2 ∧ c = 3) →
  (a^2 = 5 ∧ b^2 = 4) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1086_108632


namespace NUMINAMATH_CALUDE_cricket_players_count_l1086_108671

theorem cricket_players_count (total_players hockey_players football_players softball_players : ℕ) 
  (h1 : total_players = 77)
  (h2 : hockey_players = 15)
  (h3 : football_players = 21)
  (h4 : softball_players = 19) :
  total_players - (hockey_players + football_players + softball_players) = 22 := by
sorry

end NUMINAMATH_CALUDE_cricket_players_count_l1086_108671


namespace NUMINAMATH_CALUDE_marta_tips_l1086_108661

/-- Calculates the amount of tips Marta received given her total earnings, hourly rate, and hours worked -/
def tips_received (total_earnings hourly_rate hours_worked : ℕ) : ℕ :=
  total_earnings - hourly_rate * hours_worked

/-- Proves that Marta received $50 in tips -/
theorem marta_tips : tips_received 240 10 19 = 50 := by
  sorry

end NUMINAMATH_CALUDE_marta_tips_l1086_108661


namespace NUMINAMATH_CALUDE_terry_lunch_options_l1086_108600

/-- The number of lunch combination options for Terry's salad bar lunch. -/
def lunch_combinations (lettuce_types : ℕ) (tomato_types : ℕ) (olive_types : ℕ) 
                       (bread_types : ℕ) (fruit_types : ℕ) (soup_types : ℕ) : ℕ :=
  lettuce_types * tomato_types * olive_types * bread_types * fruit_types * soup_types

/-- Theorem stating that Terry's lunch combinations equal 4320. -/
theorem terry_lunch_options :
  lunch_combinations 4 5 6 3 4 3 = 4320 := by
  sorry

end NUMINAMATH_CALUDE_terry_lunch_options_l1086_108600


namespace NUMINAMATH_CALUDE_paige_pencils_l1086_108630

theorem paige_pencils (initial_pencils : ℕ) (used_pencils : ℕ) : 
  initial_pencils = 94 → used_pencils = 3 → initial_pencils - used_pencils = 91 := by
  sorry

end NUMINAMATH_CALUDE_paige_pencils_l1086_108630


namespace NUMINAMATH_CALUDE_simplify_expression_l1086_108648

theorem simplify_expression : (-5) + (-6) - (-5) + 4 = -5 - 6 + 5 + 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1086_108648


namespace NUMINAMATH_CALUDE_no_solution_for_x_equals_one_l1086_108640

theorem no_solution_for_x_equals_one :
  ¬∃ (y : ℝ), (1 : ℝ) / (1 + 1) + y = (1 : ℝ) / (1 - 1) :=
by sorry

end NUMINAMATH_CALUDE_no_solution_for_x_equals_one_l1086_108640


namespace NUMINAMATH_CALUDE_problem_solution_l1086_108669

theorem problem_solution : 
  (5 * Real.sqrt 2 - (Real.sqrt 18 + Real.sqrt (1/2)) = (3/2) * Real.sqrt 2) ∧
  ((2 * Real.sqrt 3 - 1)^2 + Real.sqrt 24 / Real.sqrt 2 = 13 - 2 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1086_108669


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1086_108612

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ) 
  (h1 : ∀ n, S n = (a 1) * (1 - (a 2 / a 1)^n) / (1 - (a 2 / a 1)))
  (h2 : ∀ n, a (n + 1) = a n * (a 2 / a 1))
  (h3 : S 3 = 15)
  (h4 : a 3 = 5) :
  (a 2 / a 1 = -1/2) ∨ (a 2 / a 1 = 1) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1086_108612


namespace NUMINAMATH_CALUDE_box_problem_l1086_108637

theorem box_problem (total_boxes : ℕ) (small_box_units : ℕ) (large_box_units : ℕ)
  (h_total : total_boxes = 62)
  (h_small : small_box_units = 5)
  (h_large : large_box_units = 3)
  (h_load_large_first : ∃ (x : ℕ), x * (1 / large_box_units) + 15 * (1 / small_box_units) = (total_boxes - x) * (1 / small_box_units) + 15 * (1 / large_box_units))
  : ∃ (large_boxes : ℕ), large_boxes = 27 ∧ total_boxes = large_boxes + (total_boxes - large_boxes) :=
by
  sorry

end NUMINAMATH_CALUDE_box_problem_l1086_108637


namespace NUMINAMATH_CALUDE_unique_four_digit_square_with_property_l1086_108674

theorem unique_four_digit_square_with_property : ∃! n : ℕ,
  (1000 ≤ n ∧ n ≤ 9999) ∧  -- four-digit number
  (∃ m : ℕ, n = m^2) ∧     -- perfect square
  (n / 100 = 3 * (n % 100) + 1) ∧  -- satisfies the equation
  n = 2809 := by
sorry

end NUMINAMATH_CALUDE_unique_four_digit_square_with_property_l1086_108674


namespace NUMINAMATH_CALUDE_parabola_parameter_l1086_108646

/-- For a parabola with equation y^2 = 4ax and directrix x = -2, the value of a is 2. -/
theorem parabola_parameter (y x a : ℝ) : 
  (∀ y x, y^2 = 4*a*x) →  -- Equation of the parabola
  (∀ x, x = -2 → x = x) →  -- Equation of the directrix (x = -2 represented as a predicate)
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_parabola_parameter_l1086_108646


namespace NUMINAMATH_CALUDE_min_red_chips_l1086_108617

theorem min_red_chips (r w b : ℕ) : 
  b ≥ w / 3 →
  b ≤ r / 4 →
  w + b ≥ 70 →
  r ≥ 72 ∧ ∀ r' : ℕ, (∃ w' b' : ℕ, b' ≥ w' / 3 ∧ b' ≤ r' / 4 ∧ w' + b' ≥ 70) → r' ≥ r :=
by sorry

end NUMINAMATH_CALUDE_min_red_chips_l1086_108617


namespace NUMINAMATH_CALUDE_power_of_two_equation_solution_l1086_108645

theorem power_of_two_equation_solution :
  ∀ (a n : ℕ), a ≥ n → n ≥ 2 → (∃ x : ℕ, (a + 1)^n + a - 1 = 2^x) →
  a = 4 ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_equation_solution_l1086_108645


namespace NUMINAMATH_CALUDE_geometric_sequence_general_term_l1086_108618

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_general_term
  (a : ℕ → ℝ)
  (h_geo : geometric_sequence a)
  (h_first : a 1 = 2)
  (h_second : a 2 = 4)
  (h_ineq : ∀ x : ℝ, -x^2 + 6*x - 8 > 0 ↔ 2 < x ∧ x < 4) :
  ∀ n : ℕ, a n = 2^n :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_general_term_l1086_108618


namespace NUMINAMATH_CALUDE_g_composition_fixed_points_l1086_108686

def g (x : ℝ) : ℝ := x^2 - 4*x

theorem g_composition_fixed_points :
  ∀ x : ℝ, g (g x) = g x ↔ x = -1 ∨ x = 0 ∨ x = 4 ∨ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_g_composition_fixed_points_l1086_108686


namespace NUMINAMATH_CALUDE_det_B_squared_minus_3B_l1086_108678

def B : Matrix (Fin 2) (Fin 2) ℝ := !![2, 4; 3, 2]

theorem det_B_squared_minus_3B : 
  Matrix.det ((B ^ 2) - 3 • B) = -704 := by sorry

end NUMINAMATH_CALUDE_det_B_squared_minus_3B_l1086_108678


namespace NUMINAMATH_CALUDE_ellipse_circle_intersection_l1086_108658

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive_a : 0 < a
  h_positive_b : 0 < b
  h_a_ge_b : a ≥ b

/-- Represents a circle with center (h, k) and radius r -/
structure Circle where
  h : ℝ
  k : ℝ
  r : ℝ
  h_positive_r : 0 < r

/-- The statement of the problem -/
theorem ellipse_circle_intersection (e : Ellipse) (c : Circle) :
  e.a = 3 ∧ e.b = 2 ∧
  (∃ (x y : ℝ), x^2 / 9 + y^2 / 4 = 1 ∧ (x - c.h)^2 + (y - c.k)^2 = c.r^2) ∧
  (∃ (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ),
    (x₁^2 / 9 + y₁^2 / 4 = 1 ∧ (x₁ - c.h)^2 + (y₁ - c.k)^2 = c.r^2) ∧
    (x₂^2 / 9 + y₂^2 / 4 = 1 ∧ (x₂ - c.h)^2 + (y₂ - c.k)^2 = c.r^2) ∧
    (x₃^2 / 9 + y₃^2 / 4 = 1 ∧ (x₃ - c.h)^2 + (y₃ - c.k)^2 = c.r^2) ∧
    (x₄^2 / 9 + y₄^2 / 4 = 1 ∧ (x₄ - c.h)^2 + (y₄ - c.k)^2 = c.r^2) ∧
    (x₁, y₁) ≠ (x₂, y₂) ∧ (x₁, y₁) ≠ (x₃, y₃) ∧ (x₁, y₁) ≠ (x₄, y₄) ∧
    (x₂, y₂) ≠ (x₃, y₃) ∧ (x₂, y₂) ≠ (x₄, y₄) ∧ (x₃, y₃) ≠ (x₄, y₄)) →
  c.r ≥ Real.sqrt 5 ∧ c.r < 9 :=
sorry

end NUMINAMATH_CALUDE_ellipse_circle_intersection_l1086_108658


namespace NUMINAMATH_CALUDE_parabola_tangent_hyperbola_l1086_108684

/-- The value of m for which the parabola y = 2x^2 + 3 is tangent to the hyperbola 4y^2 - mx^2 = 9 -/
def tangent_value : ℝ := 48

/-- The equation of the parabola -/
def parabola (x y : ℝ) : Prop := y = 2 * x^2 + 3

/-- The equation of the hyperbola -/
def hyperbola (m x y : ℝ) : Prop := 4 * y^2 - m * x^2 = 9

/-- The parabola is tangent to the hyperbola when m equals the tangent_value -/
theorem parabola_tangent_hyperbola :
  ∃ (x y : ℝ), parabola x y ∧ hyperbola tangent_value x y ∧
  ∀ (x' y' : ℝ), parabola x' y' ∧ hyperbola tangent_value x' y' → (x', y') = (x, y) :=
sorry

end NUMINAMATH_CALUDE_parabola_tangent_hyperbola_l1086_108684


namespace NUMINAMATH_CALUDE_salary_decrease_percentage_l1086_108605

/-- Proves that given an original salary of 5000, an initial increase of 10%,
    and a final salary of 5225, the percentage decrease after the initial increase is 5%. -/
theorem salary_decrease_percentage
  (original_salary : ℝ)
  (initial_increase_percentage : ℝ)
  (final_salary : ℝ)
  (h1 : original_salary = 5000)
  (h2 : initial_increase_percentage = 10)
  (h3 : final_salary = 5225)
  : ∃ (decrease_percentage : ℝ),
    decrease_percentage = 5 ∧
    final_salary = original_salary * (1 + initial_increase_percentage / 100) * (1 - decrease_percentage / 100) :=
by sorry

end NUMINAMATH_CALUDE_salary_decrease_percentage_l1086_108605


namespace NUMINAMATH_CALUDE_repeating_decimal_three_three_six_l1086_108677

/-- Represents a repeating decimal where the decimal part repeats infinitely -/
def RepeatingDecimal (whole : ℤ) (repeating : ℕ) : ℚ :=
  whole + (repeating : ℚ) / (99 : ℚ)

/-- The statement that 3.363636... equals 37/11 -/
theorem repeating_decimal_three_three_six : RepeatingDecimal 3 36 = 37 / 11 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_three_three_six_l1086_108677


namespace NUMINAMATH_CALUDE_ellipse_tangent_property_l1086_108682

/-- Ellipse passing through a point with specific tangent properties -/
theorem ellipse_tangent_property (m : ℝ) (r : ℝ) (h_m : m > 0) (h_r : r > 0) :
  (∃ (E F : ℝ × ℝ),
    -- E and F are on the ellipse
    (E.1^2 / 4 + E.2^2 / m = 1) ∧
    (F.1^2 / 4 + F.2^2 / m = 1) ∧
    -- A is on the ellipse
    (1^2 / 4 + (3/2)^2 / m = 1) ∧
    -- Slopes form arithmetic sequence
    (∃ (k : ℝ),
      (F.2 - 3/2) / (F.1 - 1) = k ∧
      (E.2 - 3/2) / (E.1 - 1) = -k ∧
      (F.2 - E.2) / (F.1 - E.1) = 3*k) ∧
    -- AE and AF are tangent to the circle
    ((1 - 2)^2 + (3/2 - 3/2)^2 = r^2)) →
  r = Real.sqrt 37 / 37 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_tangent_property_l1086_108682


namespace NUMINAMATH_CALUDE_tire_purchase_cost_total_cost_proof_l1086_108672

/-- Calculates the total cost of purchasing tires with given prices and tax rate -/
theorem tire_purchase_cost (num_tires : ℕ) (price1 : ℚ) (price2 : ℚ) (tax_rate : ℚ) : ℚ :=
  let first_group_cost := min num_tires 4 * price1
  let second_group_cost := max (num_tires - 4) 0 * price2
  let subtotal := first_group_cost + second_group_cost
  let tax := subtotal * tax_rate
  subtotal + tax

/-- Proves that the total cost of purchasing 8 tires with given prices and tax rate is 3.78 -/
theorem total_cost_proof :
  tire_purchase_cost 8 (1/2) (2/5) (1/20) = 189/50 :=
by sorry

end NUMINAMATH_CALUDE_tire_purchase_cost_total_cost_proof_l1086_108672


namespace NUMINAMATH_CALUDE_english_marks_calculation_l1086_108639

def average_marks : ℝ := 70
def num_subjects : ℕ := 5
def math_marks : ℕ := 65
def physics_marks : ℕ := 82
def chemistry_marks : ℕ := 67
def biology_marks : ℕ := 85

theorem english_marks_calculation :
  ∃ (english_marks : ℕ),
    (english_marks + math_marks + physics_marks + chemistry_marks + biology_marks : ℝ) / num_subjects = average_marks ∧
    english_marks = 51 := by
  sorry

end NUMINAMATH_CALUDE_english_marks_calculation_l1086_108639


namespace NUMINAMATH_CALUDE_backpack_traverse_time_l1086_108641

/-- Theorem: Time taken to carry backpack through obstacle course --/
theorem backpack_traverse_time (total_time door_time second_traverse_minutes second_traverse_seconds : ℕ) :
  let second_traverse_time := second_traverse_minutes * 60 + second_traverse_seconds
  let remaining_time := total_time - (door_time + second_traverse_time)
  total_time = 874 ∧ door_time = 73 ∧ second_traverse_minutes = 5 ∧ second_traverse_seconds = 58 →
  remaining_time = 443 := by
  sorry

end NUMINAMATH_CALUDE_backpack_traverse_time_l1086_108641


namespace NUMINAMATH_CALUDE_first_discount_percentage_l1086_108629

theorem first_discount_percentage (original_price : ℝ) (final_price : ℝ) (second_discount : ℝ) :
  original_price = 340 →
  final_price = 231.2 →
  second_discount = 0.15 →
  ∃ (first_discount : ℝ),
    first_discount = 0.2 ∧
    final_price = original_price * (1 - first_discount) * (1 - second_discount) :=
by sorry

end NUMINAMATH_CALUDE_first_discount_percentage_l1086_108629


namespace NUMINAMATH_CALUDE_divisibility_of_sum_l1086_108643

theorem divisibility_of_sum : 
  let x : ℕ := 50 + 100 + 140 + 180 + 320 + 400 + 5000
  (x % 5 = 0 ∧ x % 10 = 0) ∧ (x % 20 ≠ 0 ∧ x % 40 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_of_sum_l1086_108643


namespace NUMINAMATH_CALUDE_mentoring_program_fraction_l1086_108606

theorem mentoring_program_fraction (total : ℕ) (s : ℕ) (n : ℕ) : 
  total = s + n →
  s > 0 →
  n > 0 →
  (n : ℚ) / 4 = (s : ℚ) / 3 →
  ((n : ℚ) / 4 + (s : ℚ) / 3) / total = 2 / 7 :=
by sorry

end NUMINAMATH_CALUDE_mentoring_program_fraction_l1086_108606


namespace NUMINAMATH_CALUDE_jansen_family_has_three_children_l1086_108623

/-- Represents the Jansen family structure -/
structure JansenFamily where
  mother_age : ℝ
  father_age : ℝ
  grandfather_age : ℝ
  num_children : ℕ
  children_total_age : ℝ

/-- The Jansen family satisfies the given conditions -/
def is_valid_jansen_family (family : JansenFamily) : Prop :=
  family.father_age = 50 ∧
  family.grandfather_age = 70 ∧
  (family.mother_age + family.father_age + family.grandfather_age + family.children_total_age) / 
    (3 + family.num_children : ℝ) = 25 ∧
  (family.mother_age + family.grandfather_age + family.children_total_age) / 
    (2 + family.num_children : ℝ) = 20

/-- The number of children in a valid Jansen family is 3 -/
theorem jansen_family_has_three_children (family : JansenFamily) 
    (h : is_valid_jansen_family family) : family.num_children = 3 := by
  sorry

#check jansen_family_has_three_children

end NUMINAMATH_CALUDE_jansen_family_has_three_children_l1086_108623


namespace NUMINAMATH_CALUDE_intersection_point_expression_l1086_108694

theorem intersection_point_expression (m n : ℝ) : 
  n = m - 2022 → 
  n = -2022 / m → 
  (2022 / m) + ((m^2 - 2022*m) / n) = 2022 := by
sorry

end NUMINAMATH_CALUDE_intersection_point_expression_l1086_108694


namespace NUMINAMATH_CALUDE_A_sufficient_not_necessary_for_D_l1086_108610

-- Define the propositions
variable (A B C D : Prop)

-- Define the relationships between the propositions
variable (h1 : A → B ∧ ¬(B → A))
variable (h2 : (B → C) ∧ (C → B))
variable (h3 : (C → D) ∧ ¬(D → C))

-- Theorem to prove
theorem A_sufficient_not_necessary_for_D : 
  (A → D) ∧ ¬(D → A) :=
sorry

end NUMINAMATH_CALUDE_A_sufficient_not_necessary_for_D_l1086_108610


namespace NUMINAMATH_CALUDE_johns_number_l1086_108628

theorem johns_number : ∃! n : ℕ, 1000 < n ∧ n < 3000 ∧ 64 ∣ n ∧ 45 ∣ n ∧ n = 2880 := by
  sorry

end NUMINAMATH_CALUDE_johns_number_l1086_108628


namespace NUMINAMATH_CALUDE_wills_calories_burned_per_minute_l1086_108642

/-- Calories burned per minute while jogging -/
def calories_burned_per_minute (initial_calories net_calories jogging_duration_minutes : ℕ) : ℚ :=
  (initial_calories - net_calories : ℚ) / jogging_duration_minutes

/-- Theorem stating the calories burned per minute for Will's specific case -/
theorem wills_calories_burned_per_minute :
  calories_burned_per_minute 900 600 30 = 10 := by
  sorry

end NUMINAMATH_CALUDE_wills_calories_burned_per_minute_l1086_108642


namespace NUMINAMATH_CALUDE_set_operations_l1086_108604

theorem set_operations (A B : Set ℕ) (hA : A = {3, 5, 6, 8}) (hB : B = {4, 5, 7, 8}) :
  (A ∩ B = {5, 8}) ∧ (A ∪ B = {3, 4, 5, 6, 7, 8}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l1086_108604


namespace NUMINAMATH_CALUDE_angle_c_90_sufficient_not_necessary_l1086_108676

/-- Triangle ABC with angles A, B, and C -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  angle_sum : A + B + C = π

/-- Theorem stating that in a triangle ABC, angle C = 90° is a sufficient 
    but not necessary condition for cos A + sin A = cos B + sin B -/
theorem angle_c_90_sufficient_not_necessary (t : Triangle) :
  (t.C = π / 2 → Real.cos t.A + Real.sin t.A = Real.cos t.B + Real.sin t.B) ∧
  ∃ t' : Triangle, Real.cos t'.A + Real.sin t'.A = Real.cos t'.B + Real.sin t'.B ∧ t'.C ≠ π / 2 :=
sorry

end NUMINAMATH_CALUDE_angle_c_90_sufficient_not_necessary_l1086_108676


namespace NUMINAMATH_CALUDE_set_separation_iff_disjoint_l1086_108690

universe u

theorem set_separation_iff_disjoint {U : Type u} (A B : Set U) :
  (∃ C : Set U, A ⊆ C ∧ B ⊆ Cᶜ) ↔ A ∩ B = ∅ := by
  sorry

end NUMINAMATH_CALUDE_set_separation_iff_disjoint_l1086_108690


namespace NUMINAMATH_CALUDE_shorter_side_is_eight_l1086_108633

/-- A rectangle with given area and perimeter -/
structure Rectangle where
  length : ℝ
  width : ℝ
  area_eq : length * width = 104
  perimeter_eq : 2 * (length + width) = 42

/-- The shorter side of the rectangle is 8 feet -/
theorem shorter_side_is_eight (r : Rectangle) : min r.length r.width = 8 := by
  sorry

end NUMINAMATH_CALUDE_shorter_side_is_eight_l1086_108633


namespace NUMINAMATH_CALUDE_linear_congruence_solution_l1086_108654

theorem linear_congruence_solution (x : ℤ) : 
  (9 * x + 2) % 15 = 7 → x % 5 = 0 := by
sorry

end NUMINAMATH_CALUDE_linear_congruence_solution_l1086_108654


namespace NUMINAMATH_CALUDE_pure_imaginary_power_l1086_108689

theorem pure_imaginary_power (a : ℝ) (z : ℂ) : 
  z = a + (a + 1) * Complex.I → (z.im ≠ 0 ∧ z.re = 0) → z^2010 = -1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_power_l1086_108689


namespace NUMINAMATH_CALUDE_cosine_sine_fraction_equals_negative_tangent_l1086_108685

theorem cosine_sine_fraction_equals_negative_tangent (α : ℝ) :
  (Real.cos α - Real.cos (3 * α) + Real.cos (5 * α) - Real.cos (7 * α)) / 
  (Real.sin α + Real.sin (3 * α) + Real.sin (5 * α) + Real.sin (7 * α)) = 
  -Real.tan α := by
  sorry

end NUMINAMATH_CALUDE_cosine_sine_fraction_equals_negative_tangent_l1086_108685


namespace NUMINAMATH_CALUDE_car_speed_second_hour_l1086_108644

/-- Proves that given a car's speed in the first hour is 120 km/h and its average speed over two hours is 95 km/h, the speed of the car in the second hour is 70 km/h. -/
theorem car_speed_second_hour 
  (speed_first_hour : ℝ) 
  (average_speed : ℝ) 
  (h1 : speed_first_hour = 120) 
  (h2 : average_speed = 95) : 
  (2 * average_speed - speed_first_hour = 70) := by
  sorry

#check car_speed_second_hour

end NUMINAMATH_CALUDE_car_speed_second_hour_l1086_108644


namespace NUMINAMATH_CALUDE_four_fours_theorem_l1086_108635

def is_valid_expression (e : ℕ → ℕ) : Prop :=
  ∃ (a b c d f : ℕ), 
    (a = 4 ∧ b = 4 ∧ c = 4 ∧ d = 4 ∧ f = 4) ∧
    (∀ n : ℕ, 1 ≤ n ∧ n ≤ 22 → e n = n)

theorem four_fours_theorem :
  ∃ e : ℕ → ℕ, is_valid_expression e :=
sorry

end NUMINAMATH_CALUDE_four_fours_theorem_l1086_108635


namespace NUMINAMATH_CALUDE_no_integer_solutions_l1086_108649

theorem no_integer_solutions : ¬∃ (a b : ℤ), (1 : ℚ) / a + (1 : ℚ) / b = -(1 : ℚ) / (a + b) :=
sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l1086_108649


namespace NUMINAMATH_CALUDE_expand_expression_l1086_108638

theorem expand_expression (y : ℝ) : (7 * y + 12) * (3 * y) = 21 * y^2 + 36 * y := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l1086_108638


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l1086_108615

theorem necessary_but_not_sufficient_condition (a : ℝ) : 
  (∃ x ∈ Set.Icc 1 3, x^2 - a*x + 4 < 0) → 
  (a > 3 ∧ ∃ b > 3, ¬(∃ x ∈ Set.Icc 1 3, x^2 - b*x + 4 < 0)) := by
  sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l1086_108615


namespace NUMINAMATH_CALUDE_probability_x_plus_y_less_than_4_l1086_108659

/-- A square with vertices at (0, 0), (0, 3), (3, 3), and (3, 0) -/
def Square : Set (ℝ × ℝ) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ 3 ∧ 0 ≤ p.2 ∧ p.2 ≤ 3}

/-- The region where x + y < 4 within the square -/
def RegionXPlusYLessThan4 : Set (ℝ × ℝ) :=
  {p ∈ Square | p.1 + p.2 < 4}

/-- The area of the square -/
def squareArea : ℝ := 9

/-- The area of the region where x + y < 4 within the square -/
def regionArea : ℝ := 7

theorem probability_x_plus_y_less_than_4 :
  (regionArea / squareArea : ℝ) = 7 / 9 := by sorry

end NUMINAMATH_CALUDE_probability_x_plus_y_less_than_4_l1086_108659


namespace NUMINAMATH_CALUDE_equal_area_line_slope_l1086_108603

/-- Represents a circle in 2D space -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Calculates the distance from a point to a line -/
def distancePointToLine (p : ℝ × ℝ) (l : Line) : ℝ :=
  sorry

/-- Determines if a line divides a circle into equal areas -/
def divideCircleEqually (c : Circle) (l : Line) : Prop :=
  sorry

theorem equal_area_line_slope :
  let c1 : Circle := ⟨(10, 40), 5⟩
  let c2 : Circle := ⟨(15, 30), 5⟩
  let p : ℝ × ℝ := (12, 20)
  ∃ (l : Line),
    l.slope = (5 - Real.sqrt 73) / 2 ∨ l.slope = (5 + Real.sqrt 73) / 2 ∧
    (p.1 * l.slope + l.yIntercept = p.2) ∧
    divideCircleEqually c1 l ∧
    divideCircleEqually c2 l :=
  sorry

end NUMINAMATH_CALUDE_equal_area_line_slope_l1086_108603


namespace NUMINAMATH_CALUDE_male_democrat_ratio_l1086_108675

/-- Proves the ratio of male democrats to total male participants in a meeting --/
theorem male_democrat_ratio (total_participants : ℕ) 
  (female_democrats : ℕ) (h1 : total_participants = 660) 
  (h2 : female_democrats = 110) 
  (h3 : female_democrats * 2 = total_participants / 3) : 
  (total_participants / 3 - female_democrats) * 4 = 
  (total_participants - female_democrats * 2) :=
sorry

end NUMINAMATH_CALUDE_male_democrat_ratio_l1086_108675


namespace NUMINAMATH_CALUDE_x_plus_q_in_terms_of_q_l1086_108670

theorem x_plus_q_in_terms_of_q (x q : ℝ) (h1 : |x + 3| = q) (h2 : x > -3) :
  x + q = 2*q - 3 := by
sorry

end NUMINAMATH_CALUDE_x_plus_q_in_terms_of_q_l1086_108670


namespace NUMINAMATH_CALUDE_absolute_value_square_l1086_108655

theorem absolute_value_square (a b : ℝ) : a > |b| → a^2 > b^2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_square_l1086_108655


namespace NUMINAMATH_CALUDE_rectangle_ellipse_perimeter_l1086_108611

/-- Given a rectangle and an ellipse with specific properties, prove that the perimeter of the rectangle is 450. -/
theorem rectangle_ellipse_perimeter :
  ∀ (x y : ℝ) (a b : ℝ),
  -- Rectangle conditions
  x * y = 2500 ∧
  x / y = 5 / 4 ∧
  -- Ellipse conditions
  π * a * b = 2500 * π ∧
  x + y = 2 * a ∧
  (x^2 + y^2 : ℝ) = 4 * (a^2 - b^2) →
  -- Conclusion
  2 * (x + y) = 450 := by
sorry

end NUMINAMATH_CALUDE_rectangle_ellipse_perimeter_l1086_108611


namespace NUMINAMATH_CALUDE_intersection_point_satisfies_equations_l1086_108625

/-- Two lines in a 2D plane -/
structure TwoLines where
  line1 : ℝ → ℝ → ℝ
  line2 : ℝ → ℝ → ℝ

/-- The intersection point of two lines -/
def intersection_point : ℝ × ℝ := (-4, 3)

/-- The given two lines -/
def given_lines : TwoLines where
  line1 := fun x y => 3 * x + 2 * y + 6
  line2 := fun x y => 2 * x + 5 * y - 7

theorem intersection_point_satisfies_equations : 
  let (x, y) := intersection_point
  given_lines.line1 x y = 0 ∧ given_lines.line2 x y = 0 := by
  sorry

#check intersection_point_satisfies_equations

end NUMINAMATH_CALUDE_intersection_point_satisfies_equations_l1086_108625


namespace NUMINAMATH_CALUDE_unique_solution_l1086_108665

def A (a b : ℝ) := {x : ℝ | x^2 + a*x + b = 0}
def B (c : ℝ) := {x : ℝ | x^2 + c*x + 15 = 0}

theorem unique_solution :
  ∃! (a b c : ℝ),
    (A a b ∪ B c = {3, 5}) ∧
    (A a b ∩ B c = {3}) ∧
    a = -6 ∧ b = 9 ∧ c = -8 := by sorry

end NUMINAMATH_CALUDE_unique_solution_l1086_108665


namespace NUMINAMATH_CALUDE_bus_profit_maximization_l1086_108660

/-- The profit function for a bus operating for x years -/
def profit (x : ℕ+) : ℚ := -x^2 + 18*x - 36

/-- The average profit function for a bus operating for x years -/
def avgProfit (x : ℕ+) : ℚ := (profit x) / x

theorem bus_profit_maximization :
  (∃ (x : ℕ+), ∀ (y : ℕ+), profit x ≥ profit y) ∧
  (∃ (x : ℕ+), profit x = 45) ∧
  (∃ (x : ℕ+), ∀ (y : ℕ+), avgProfit x ≥ avgProfit y) ∧
  (∃ (x : ℕ+), avgProfit x = 6) :=
sorry

end NUMINAMATH_CALUDE_bus_profit_maximization_l1086_108660
