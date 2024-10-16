import Mathlib

namespace NUMINAMATH_CALUDE_expression_simplification_l3978_397850

theorem expression_simplification (x y z : ℝ) :
  ((x + y) - (z - y)) - ((x + z) - (y + z)) = 3 * y - z := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3978_397850


namespace NUMINAMATH_CALUDE_candy_distribution_l3978_397835

theorem candy_distribution (total : Nat) (friends : Nat) (h1 : total = 17) (h2 : friends = 5) :
  total % friends = 2 := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l3978_397835


namespace NUMINAMATH_CALUDE_factorize_difference_of_squares_factorize_polynomial_l3978_397807

-- Problem 1
theorem factorize_difference_of_squares (x y : ℝ) :
  4 * x^2 - 25 * y^2 = (2*x + 5*y) * (2*x - 5*y) := by
  sorry

-- Problem 2
theorem factorize_polynomial (x y : ℝ) :
  -3 * x * y^3 + 27 * x^3 * y = -3 * x * y * (y + 3*x) * (y - 3*x) := by
  sorry

end NUMINAMATH_CALUDE_factorize_difference_of_squares_factorize_polynomial_l3978_397807


namespace NUMINAMATH_CALUDE_inequality_proof_l3978_397857

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (2 * x^2) / (y + z) + (2 * y^2) / (z + x) + (2 * z^2) / (x + y) ≥ x + y + z := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3978_397857


namespace NUMINAMATH_CALUDE_vertical_angles_are_equal_equal_angles_are_vertical_converse_of_vertical_angles_are_equal_l3978_397830

/-- Definition of vertical angles -/
def VerticalAngles (α β : Angle) : Prop := sorry

/-- The original proposition -/
theorem vertical_angles_are_equal (α β : Angle) : 
  VerticalAngles α β → α = β := sorry

/-- The converse proposition -/
theorem equal_angles_are_vertical (α β : Angle) : 
  α = β → VerticalAngles α β := sorry

/-- Theorem stating that the converse of "Vertical angles are equal" 
    is "Angles that are equal are vertical angles" -/
theorem converse_of_vertical_angles_are_equal :
  (∀ α β : Angle, VerticalAngles α β → α = β) ↔ 
  (∀ α β : Angle, α = β → VerticalAngles α β) :=
sorry

end NUMINAMATH_CALUDE_vertical_angles_are_equal_equal_angles_are_vertical_converse_of_vertical_angles_are_equal_l3978_397830


namespace NUMINAMATH_CALUDE_C_work_duration_l3978_397818

-- Define the work rates and durations
def work_rate_A : ℚ := 1 / 30
def work_rate_B : ℚ := 1 / 30
def days_A_worked : ℕ := 10
def days_B_worked : ℕ := 10
def days_C_worked : ℕ := 10

-- Define the total work as 1 (representing 100%)
def total_work : ℚ := 1

-- Theorem to prove
theorem C_work_duration :
  let work_done_A : ℚ := work_rate_A * days_A_worked
  let work_done_B : ℚ := work_rate_B * days_B_worked
  let work_done_C : ℚ := total_work - (work_done_A + work_done_B)
  let work_rate_C : ℚ := work_done_C / days_C_worked
  (total_work / work_rate_C : ℚ) = 30 := by
  sorry

end NUMINAMATH_CALUDE_C_work_duration_l3978_397818


namespace NUMINAMATH_CALUDE_system_of_equations_l3978_397848

theorem system_of_equations (x y : ℝ) 
  (eq1 : 2 * x + 4 * y = 5)
  (eq2 : x - y = 10) : 
  x + y = 5 := by sorry

end NUMINAMATH_CALUDE_system_of_equations_l3978_397848


namespace NUMINAMATH_CALUDE_range_of_a_l3978_397890

theorem range_of_a (a : ℝ) : 
  (∃ b : ℝ, b ∈ Set.Icc 1 2 ∧ 2^b * (b + a) ≥ 4) ↔ a ∈ Set.Ici (-1) := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l3978_397890


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3978_397869

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (x - 5) = 10 → x = 105 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3978_397869


namespace NUMINAMATH_CALUDE_unique_function_property_l3978_397867

def FunctionProperty (f : ℤ → ℤ) : Prop :=
  ∀ n : ℤ, n^2 + 4 * f n = f (f (f n))

theorem unique_function_property :
  ∃! f : ℤ → ℤ, FunctionProperty f ∧ ∀ n : ℤ, f n = n + 1 :=
sorry

end NUMINAMATH_CALUDE_unique_function_property_l3978_397867


namespace NUMINAMATH_CALUDE_new_drive_free_space_calculation_l3978_397841

/-- Calculates the free space on a new external drive after file operations -/
def new_drive_free_space (initial_free : ℝ) (initial_used : ℝ) (deleted1 : ℝ) (deleted2 : ℝ) (added1 : ℝ) (added2 : ℝ) (new_drive_size : ℝ) : ℝ :=
  new_drive_size - (initial_used - (deleted1 + deleted2) + (added1 + added2))

/-- Theorem stating that the free space on the new drive is 313.5 GB -/
theorem new_drive_free_space_calculation :
  new_drive_free_space 75.8 210.3 34.5 29.7 13 27.4 500 = 313.5 := by
  sorry

#eval new_drive_free_space 75.8 210.3 34.5 29.7 13 27.4 500

end NUMINAMATH_CALUDE_new_drive_free_space_calculation_l3978_397841


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l3978_397805

theorem isosceles_triangle_base_length 
  (congruent_side : ℝ) 
  (perimeter : ℝ) 
  (h1 : congruent_side = 6) 
  (h2 : perimeter = 20) :
  perimeter - 2 * congruent_side = 8 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l3978_397805


namespace NUMINAMATH_CALUDE_power_multiplication_l3978_397897

theorem power_multiplication (b : ℕ) (h : b = 2) : b^3 * b^4 = 128 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l3978_397897


namespace NUMINAMATH_CALUDE_lcm_1260_980_l3978_397891

theorem lcm_1260_980 : Nat.lcm 1260 980 = 8820 := by
  sorry

end NUMINAMATH_CALUDE_lcm_1260_980_l3978_397891


namespace NUMINAMATH_CALUDE_orthocenter_preservation_l3978_397844

-- Define the types for points and triangles
def Point : Type := ℝ × ℝ
def Triangle : Type := Point × Point × Point

-- Define the orthocenter of a triangle
def orthocenter (t : Triangle) : Point := sorry

-- Define the function to check if a point is inside a triangle
def is_inside (p : Point) (t : Triangle) : Prop := sorry

-- Define the function to check if a point is on a line segment
def on_segment (p : Point) (a b : Point) : Prop := sorry

-- Define the function to find the intersection of two line segments
def intersection (a b c d : Point) : Point := sorry

-- Main theorem
theorem orthocenter_preservation 
  (A B C H A₁ B₁ C₁ A₂ B₂ C₂ : Point) 
  (ABC : Triangle) :
  -- Given conditions
  (orthocenter ABC = H) →
  (is_inside A₁ (B, C, H)) →
  (is_inside B₁ (C, A, H)) →
  (is_inside C₁ (A, B, H)) →
  (orthocenter (A₁, B₁, C₁) = H) →
  (A₂ = intersection A H B₁ C₁) →
  (B₂ = intersection B H C₁ A₁) →
  (C₂ = intersection C H A₁ B₁) →
  -- Conclusion
  (orthocenter (A₂, B₂, C₂) = H) := by
  sorry

end NUMINAMATH_CALUDE_orthocenter_preservation_l3978_397844


namespace NUMINAMATH_CALUDE_length_of_BC_l3978_397815

-- Define the parabola
def parabola (x : ℝ) : ℝ := 4 * x^2

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the conditions of the problem
def triangle_conditions (t : Triangle) (b c : ℝ) : Prop :=
  t.A = (0, 0) ∧
  t.B = (-b, parabola (-b)) ∧
  t.C = (c, parabola c) ∧
  b > 0 ∧
  c > 0 ∧
  t.B.2 = t.C.2 ∧  -- BC is parallel to x-axis
  (1/2 * (c + b) * (parabola (-b))) = 96  -- Area of the triangle is 96

-- Theorem to prove
theorem length_of_BC (t : Triangle) (b c : ℝ) 
  (h : triangle_conditions t b c) : 
  (t.C.1 - t.B.1) = 59/9 := by sorry

end NUMINAMATH_CALUDE_length_of_BC_l3978_397815


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3978_397872

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    if a line through one vertex on the imaginary axis and perpendicular to the y-axis
    forms an equilateral triangle with the other vertex on the imaginary axis and
    the two points where it intersects the hyperbola, then the eccentricity is √10/2 -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let hyperbola := fun (x y : ℝ) ↦ x^2 / a^2 - y^2 / b^2 = 1
  let B₁ := (0, b)
  let B₂ := (0, -b)
  let line := fun (x : ℝ) ↦ b
  let P := (-Real.sqrt 2 * a, b)
  let Q := (Real.sqrt 2 * a, b)
  hyperbola P.1 P.2 ∧ hyperbola Q.1 Q.2 ∧
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = (P.1 - B₂.1)^2 + (P.2 - B₂.2)^2 ∧
  (P.1 - B₂.1)^2 + (P.2 - B₂.2)^2 = (Q.1 - B₂.1)^2 + (Q.2 - B₂.2)^2 →
  Real.sqrt (1 + b^2 / a^2) = Real.sqrt 10 / 2 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3978_397872


namespace NUMINAMATH_CALUDE_max_log_sum_l3978_397827

theorem max_log_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 5) :
  ∃ (max_val : ℝ), max_val = 6 ∧ ∀ (a b : ℝ), a > 0 → b > 0 → a + b = 5 → Real.log a + Real.log b ≤ max_val :=
by
  sorry

end NUMINAMATH_CALUDE_max_log_sum_l3978_397827


namespace NUMINAMATH_CALUDE_ratio_problem_l3978_397813

theorem ratio_problem (x y : ℝ) (h1 : 5 * x = 3 * y) (h2 : x * y ≠ 0) :
  (1 / 5 * x) / (1 / 6 * y) = 0.72 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l3978_397813


namespace NUMINAMATH_CALUDE_square_side_length_l3978_397878

theorem square_side_length (area : ℝ) (side : ℝ) : 
  area = 1 / 9 → side * side = area → side = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l3978_397878


namespace NUMINAMATH_CALUDE_length_of_AB_is_two_l3978_397836

-- Define the points A and B
def A (a : ℝ) : ℝ × ℝ := (3, a + 3)
def B (a : ℝ) : ℝ × ℝ := (a, 4)

-- Define the condition that AB is parallel to the x-axis
def parallel_to_x_axis (a : ℝ) : Prop :=
  (A a).2 = (B a).2

-- Define the length of segment AB
def length_AB (a : ℝ) : ℝ :=
  |((A a).1 - (B a).1)|

-- Theorem statement
theorem length_of_AB_is_two (a : ℝ) :
  parallel_to_x_axis a → length_AB a = 2 := by
  sorry

end NUMINAMATH_CALUDE_length_of_AB_is_two_l3978_397836


namespace NUMINAMATH_CALUDE_A_equals_B_l3978_397864

/-- The number of ways to pair r girls with r boys in town A -/
def A (n r : ℕ) : ℕ := (n.choose r)^2 * r.factorial

/-- The number of ways to pair r girls with r boys in town B -/
def B : ℕ → ℕ → ℕ
| 0, _ => 0
| _, 0 => 1
| n+1, r+1 => (2*n+1 - r) * B n r + B n (r+1)

/-- The theorem stating that A(n,r) equals B(n,r) for all valid n and r -/
theorem A_equals_B (n r : ℕ) (h : r ≤ n) : A n r = B n r := by
  sorry

end NUMINAMATH_CALUDE_A_equals_B_l3978_397864


namespace NUMINAMATH_CALUDE_jenna_stamps_problem_l3978_397808

theorem jenna_stamps_problem (a b c : ℕ) 
  (ha : a = 924) (hb : b = 1260) (hc : c = 1386) : 
  Nat.gcd a (Nat.gcd b c) = 42 := by
  sorry

end NUMINAMATH_CALUDE_jenna_stamps_problem_l3978_397808


namespace NUMINAMATH_CALUDE_infinitely_many_solutions_l3978_397843

theorem infinitely_many_solutions (k : ℝ) : 
  (∀ x : ℝ, 5 * (3 * x - k) = 3 * (5 * x + 15)) ↔ k = -9 := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_solutions_l3978_397843


namespace NUMINAMATH_CALUDE_florist_initial_roses_l3978_397880

/-- Represents the number of roses picked in the first round -/
def first_pick : ℝ := 16.0

/-- Represents the number of roses picked in the second round -/
def second_pick : ℝ := 19.0

/-- Represents the total number of roses after all picking -/
def total_roses : ℕ := 72

/-- Calculates the initial number of roses the florist had -/
def initial_roses : ℝ := total_roses - (first_pick + second_pick)

/-- Theorem stating that the initial number of roses was 37 -/
theorem florist_initial_roses : initial_roses = 37 := by sorry

end NUMINAMATH_CALUDE_florist_initial_roses_l3978_397880


namespace NUMINAMATH_CALUDE_penguin_fish_theorem_l3978_397822

theorem penguin_fish_theorem (fish_counts : List ℕ) : 
  fish_counts.length = 10 ∧ 
  fish_counts.sum = 50 ∧ 
  (∀ x ∈ fish_counts, x > 0) →
  ∃ i j, i ≠ j ∧ i < fish_counts.length ∧ j < fish_counts.length ∧ fish_counts[i]! = fish_counts[j]! := by
  sorry

end NUMINAMATH_CALUDE_penguin_fish_theorem_l3978_397822


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3978_397849

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, Real.log x > 1) ↔ (∃ x : ℝ, Real.log x ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3978_397849


namespace NUMINAMATH_CALUDE_find_number_l3978_397884

theorem find_number (x : ℚ) : (x + 113 / 78) * 78 = 4403 → x = 55 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l3978_397884


namespace NUMINAMATH_CALUDE_cos_225_degrees_l3978_397889

theorem cos_225_degrees : Real.cos (225 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_225_degrees_l3978_397889


namespace NUMINAMATH_CALUDE_orange_packing_problem_l3978_397885

/-- Given a total number of oranges and the capacity of each box, 
    calculate the number of boxes needed to pack all oranges. -/
def boxes_needed (total_oranges : ℕ) (oranges_per_box : ℕ) : ℕ :=
  total_oranges / oranges_per_box

/-- Theorem stating that 265 boxes are needed to pack 2650 oranges
    when each box holds 10 oranges. -/
theorem orange_packing_problem :
  boxes_needed 2650 10 = 265 := by
  sorry

end NUMINAMATH_CALUDE_orange_packing_problem_l3978_397885


namespace NUMINAMATH_CALUDE_polygon_interior_exterior_equal_quadrilateral_l3978_397862

-- Define a polygon with n sides
def Polygon (n : ℕ) : Prop :=
  n ≥ 3

-- Define the sum of interior angles of a polygon
def SumInteriorAngles (n : ℕ) : ℝ :=
  (n - 2) * 180

-- Define the sum of exterior angles of a polygon (always 360°)
def SumExteriorAngles : ℝ :=
  360

-- Theorem: If the sum of interior angles equals the sum of exterior angles,
-- then the polygon has 4 sides
theorem polygon_interior_exterior_equal_quadrilateral (n : ℕ) 
  (h1 : Polygon n) 
  (h2 : SumInteriorAngles n = SumExteriorAngles) : 
  n = 4 := by
  sorry

#check polygon_interior_exterior_equal_quadrilateral

end NUMINAMATH_CALUDE_polygon_interior_exterior_equal_quadrilateral_l3978_397862


namespace NUMINAMATH_CALUDE_hyperbola_vertices_distance_l3978_397858

/-- The distance between the vertices of a hyperbola given by the equation
    16x^2 + 64x - 4y^2 + 8y + 36 = 0 is 1. -/
theorem hyperbola_vertices_distance :
  let f : ℝ → ℝ → ℝ := fun x y => 16 * x^2 + 64 * x - 4 * y^2 + 8 * y + 36
  ∃ x₁ x₂ y₁ y₂ : ℝ,
    (∀ x y, f x y = 0 ↔ 4 * (x + 2)^2 - (y - 1)^2 = 1) ∧
    (x₁, y₁) ∈ {p : ℝ × ℝ | f p.1 p.2 = 0} ∧
    (x₂, y₂) ∈ {p : ℝ × ℝ | f p.1 p.2 = 0} ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 1 ∧
    ∀ x y, f x y = 0 → (x - x₁)^2 + (y - y₁)^2 ≤ (x₁ - x₂)^2 + (y₁ - y₂)^2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_vertices_distance_l3978_397858


namespace NUMINAMATH_CALUDE_initial_potatoes_count_l3978_397881

/-- The number of potatoes Dan initially had in the garden --/
def initial_potatoes : ℕ := sorry

/-- The number of potatoes eaten by rabbits --/
def eaten_potatoes : ℕ := 4

/-- The number of potatoes Dan has now --/
def remaining_potatoes : ℕ := 3

/-- Theorem stating the initial number of potatoes --/
theorem initial_potatoes_count : initial_potatoes = 7 := by
  sorry

end NUMINAMATH_CALUDE_initial_potatoes_count_l3978_397881


namespace NUMINAMATH_CALUDE_max_class_size_is_17_l3978_397817

/-- Represents a school with students and buses -/
structure School where
  total_students : ℕ
  num_buses : ℕ
  seats_per_bus : ℕ

/-- Checks if it's possible to seat all students with the given max class size -/
def can_seat_all (s : School) (max_class_size : ℕ) : Prop :=
  ∀ (class_sizes : List ℕ),
    (class_sizes.sum = s.total_students) →
    (∀ size ∈ class_sizes, size ≤ max_class_size) →
    ∃ (allocation : List (List ℕ)),
      (allocation.length ≤ s.num_buses) ∧
      (∀ bus ∈ allocation, bus.sum ≤ s.seats_per_bus) ∧
      (allocation.join.sum = s.total_students)

/-- The theorem to be proved -/
theorem max_class_size_is_17 (s : School) 
    (h1 : s.total_students = 920)
    (h2 : s.num_buses = 16)
    (h3 : s.seats_per_bus = 71) :
  (can_seat_all s 17 ∧ ¬can_seat_all s 18) := by
  sorry

end NUMINAMATH_CALUDE_max_class_size_is_17_l3978_397817


namespace NUMINAMATH_CALUDE_monotonically_decreasing_implies_a_leq_neg_three_l3978_397833

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 3 * x^2 - x + 1

-- State the theorem
theorem monotonically_decreasing_implies_a_leq_neg_three :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x > f a y) → a ≤ -3 := by
  sorry

end NUMINAMATH_CALUDE_monotonically_decreasing_implies_a_leq_neg_three_l3978_397833


namespace NUMINAMATH_CALUDE_harvest_season_duration_l3978_397877

/-- Calculates the number of weeks in a harvest season based on weekly earnings, rent, and total savings. -/
def harvest_season_weeks (weekly_earnings : ℕ) (weekly_rent : ℕ) (total_savings : ℕ) : ℕ :=
  total_savings / (weekly_earnings - weekly_rent)

/-- Proves that the number of weeks in the harvest season is 1181 given the specified conditions. -/
theorem harvest_season_duration :
  harvest_season_weeks 491 216 324775 = 1181 := by
  sorry

end NUMINAMATH_CALUDE_harvest_season_duration_l3978_397877


namespace NUMINAMATH_CALUDE_vector_operation_result_l3978_397882

theorem vector_operation_result :
  let v1 : Fin 2 → ℝ := ![5, -3]
  let v2 : Fin 2 → ℝ := ![0, 4]
  let v3 : Fin 2 → ℝ := ![-2, 1]
  let result : Fin 2 → ℝ := ![3, -14]
  v1 - 3 • v2 + v3 = result :=
by
  sorry

end NUMINAMATH_CALUDE_vector_operation_result_l3978_397882


namespace NUMINAMATH_CALUDE_arithmetic_sequence_inequality_l3978_397855

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_inequality (a : ℕ → ℝ) (h_arith : is_arithmetic_sequence a) :
  0 < a 1 → a 1 < a 2 → a 2 > Real.sqrt (a 1 * a 3) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_inequality_l3978_397855


namespace NUMINAMATH_CALUDE_students_in_chemistry_or_physics_not_both_l3978_397837

theorem students_in_chemistry_or_physics_not_both (total_chemistry : ℕ) (both : ℕ) (only_physics : ℕ)
  (h1 : both = 15)
  (h2 : total_chemistry = 30)
  (h3 : only_physics = 12) :
  total_chemistry - both + only_physics = 27 :=
by sorry

end NUMINAMATH_CALUDE_students_in_chemistry_or_physics_not_both_l3978_397837


namespace NUMINAMATH_CALUDE_chocolate_bars_count_l3978_397820

def total_candies : ℕ := 50
def chewing_gums : ℕ := 15
def assorted_candies : ℕ := 15

theorem chocolate_bars_count :
  total_candies - chewing_gums - assorted_candies = 20 :=
by sorry

end NUMINAMATH_CALUDE_chocolate_bars_count_l3978_397820


namespace NUMINAMATH_CALUDE_right_isosceles_triangle_exists_l3978_397838

-- Define the set of points
def Points : Set (ℤ × ℤ) :=
  {(0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)}

-- Define the color type
inductive Color
| red
| blue

-- Define what it means for three points to form a right isosceles triangle
def isRightIsosceles (p1 p2 p3 : ℤ × ℤ) : Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  ((x2 - x1)^2 + (y2 - y1)^2 = (x3 - x1)^2 + (y3 - y1)^2) ∧
  ((x3 - x1) * (x2 - x1) + (y3 - y1) * (y2 - y1) = 0)

-- State the theorem
theorem right_isosceles_triangle_exists (f : ℤ × ℤ → Color) :
  ∃ (p1 p2 p3 : ℤ × ℤ), p1 ∈ Points ∧ p2 ∈ Points ∧ p3 ∈ Points ∧
  p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧
  f p1 = f p2 ∧ f p2 = f p3 ∧
  isRightIsosceles p1 p2 p3 := by
  sorry


end NUMINAMATH_CALUDE_right_isosceles_triangle_exists_l3978_397838


namespace NUMINAMATH_CALUDE_teammates_score_l3978_397874

def volleyball_scores (total_team_score : ℕ) : Prop :=
  ∃ (lizzie nathalie aimee julia ellen other : ℕ),
    lizzie = 4 ∧
    nathalie = 2 * lizzie + 3 ∧
    aimee = 2 * (lizzie + nathalie) + 1 ∧
    julia = nathalie / 2 - 2 ∧
    ellen = Int.sqrt aimee * 3 ∧
    lizzie + nathalie + aimee + julia + ellen + other = total_team_score

theorem teammates_score :
  volleyball_scores 100 → ∃ other : ℕ, other = 36 :=
by sorry

end NUMINAMATH_CALUDE_teammates_score_l3978_397874


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l3978_397860

theorem smallest_n_congruence (n : ℕ) : 
  (n > 0 ∧ 6^n ≡ n^6 [MOD 3]) → n ≥ 3 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l3978_397860


namespace NUMINAMATH_CALUDE_special_function_inequality_l3978_397806

/-- A function satisfying the given conditions -/
structure SpecialFunction where
  f : ℝ → ℝ
  differentiable : Differentiable ℝ f
  greater_than_derivative : ∀ x, f x > deriv f x
  initial_value : f 0 = 1

/-- The main theorem -/
theorem special_function_inequality (F : SpecialFunction) :
  ∀ x, (F.f x / Real.exp x < 1) ↔ x > 0 := by
  sorry

end NUMINAMATH_CALUDE_special_function_inequality_l3978_397806


namespace NUMINAMATH_CALUDE_store_inventory_difference_l3978_397876

theorem store_inventory_difference (regular_soda diet_soda apples : ℕ) 
  (h1 : regular_soda = 72) 
  (h2 : diet_soda = 32) 
  (h3 : apples = 78) : 
  (regular_soda + diet_soda) - apples = 26 := by
  sorry

end NUMINAMATH_CALUDE_store_inventory_difference_l3978_397876


namespace NUMINAMATH_CALUDE_special_set_property_l3978_397840

/-- A set of points in ℝ³ that intersects every plane but has finite intersection with each plane -/
def SpecialSet : Set (ℝ × ℝ × ℝ) :=
  {(x, y, z) | ∃ t : ℝ, x = t^5 ∧ y = t^3 ∧ z = t}

/-- Definition of a plane in ℝ³ -/
def Plane (a b c d : ℝ) : Set (ℝ × ℝ × ℝ) :=
  {(x, y, z) | a * x + b * y + c * z + d = 0}

theorem special_set_property :
  ∃ S : Set (ℝ × ℝ × ℝ),
    (∀ a b c d : ℝ, (Plane a b c d ∩ S).Nonempty) ∧
    (∀ a b c d : ℝ, (Plane a b c d ∩ S).Finite) :=
by
  use SpecialSet
  sorry

end NUMINAMATH_CALUDE_special_set_property_l3978_397840


namespace NUMINAMATH_CALUDE_washing_time_calculation_l3978_397829

def clothes_time : ℕ := 30

def towels_time (clothes_time : ℕ) : ℕ := 2 * clothes_time

def sheets_time (towels_time : ℕ) : ℕ := towels_time - 15

def total_washing_time (clothes_time towels_time sheets_time : ℕ) : ℕ :=
  clothes_time + towels_time + sheets_time

theorem washing_time_calculation :
  total_washing_time clothes_time (towels_time clothes_time) (sheets_time (towels_time clothes_time)) = 135 := by
  sorry

end NUMINAMATH_CALUDE_washing_time_calculation_l3978_397829


namespace NUMINAMATH_CALUDE_equation_solution_system_of_equations_solution_l3978_397851

-- Problem 1
theorem equation_solution : 
  let x : ℚ := -1
  (2*x + 1) / 6 - (5*x - 1) / 8 = 7 / 12 := by sorry

-- Problem 2
theorem system_of_equations_solution :
  let x : ℚ := 4
  let y : ℚ := 3
  3*x - 2*y = 6 ∧ 2*x + 3*y = 17 := by sorry

end NUMINAMATH_CALUDE_equation_solution_system_of_equations_solution_l3978_397851


namespace NUMINAMATH_CALUDE_inequality_not_always_true_l3978_397812

theorem inequality_not_always_true (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  ¬ (∀ a b, a > b ∧ b > 0 → a + 1/a < b + 1/b) :=
by sorry

end NUMINAMATH_CALUDE_inequality_not_always_true_l3978_397812


namespace NUMINAMATH_CALUDE_unit_price_ratio_l3978_397845

theorem unit_price_ratio (quantity_B price_B : ℝ) (quantity_B_pos : quantity_B > 0) (price_B_pos : price_B > 0) :
  let quantity_A := 1.3 * quantity_B
  let price_A := 0.85 * price_B
  (price_A / quantity_A) / (price_B / quantity_B) = 17 / 26 := by
sorry

end NUMINAMATH_CALUDE_unit_price_ratio_l3978_397845


namespace NUMINAMATH_CALUDE_sequence_existence_and_bound_l3978_397842

theorem sequence_existence_and_bound (a : ℝ) (n : ℕ) :
  ∃! x : ℕ → ℝ, 
    (x 1 - x (n - 1) = 0) ∧ 
    (∀ i ∈ Finset.range n, (x (i - 1) + x i) / 2 = x i + (x i)^3 - a^3) ∧
    (∀ i ∈ Finset.range (n + 2), |x i| ≤ |a|) := by
  sorry

end NUMINAMATH_CALUDE_sequence_existence_and_bound_l3978_397842


namespace NUMINAMATH_CALUDE_lcm_18_24_l3978_397824

theorem lcm_18_24 : Nat.lcm 18 24 = 72 := by
  sorry

end NUMINAMATH_CALUDE_lcm_18_24_l3978_397824


namespace NUMINAMATH_CALUDE_sum_of_fractions_l3978_397861

theorem sum_of_fractions (p q r : ℝ) 
  (h1 : p + q + r = 5) 
  (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) : 
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := by
sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l3978_397861


namespace NUMINAMATH_CALUDE_jills_study_hours_l3978_397875

/-- Represents Jill's study schedule over three days -/
structure StudySchedule where
  day1 : ℝ  -- Hours studied on day 1
  day2 : ℝ  -- Hours studied on day 2
  day3 : ℝ  -- Hours studied on day 3

/-- The theorem representing Jill's study problem -/
theorem jills_study_hours (schedule : StudySchedule) :
  schedule.day2 = 2 * schedule.day1 ∧
  schedule.day3 = 2 * schedule.day1 - 1 ∧
  schedule.day1 + schedule.day2 + schedule.day3 = 9 →
  schedule.day1 = 2 :=
by sorry

end NUMINAMATH_CALUDE_jills_study_hours_l3978_397875


namespace NUMINAMATH_CALUDE_first_square_with_two_twos_l3978_397834

def starts_with_two_twos (n : ℕ) : Prop :=
  (n / 1000 = 2) ∧ ((n / 100) % 10 = 2)

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem first_square_with_two_twos : 
  ∃! n : ℕ, 
    (∀ m : ℕ, m < n → ¬(starts_with_two_twos (m^2))) ∧ 
    (starts_with_two_twos (n^2)) ∧
    (∃ k : ℕ, k > n ∧ starts_with_two_twos (k^2) ∧ sum_of_digits (k^2) = 13) ∧
    n = 47 := by sorry

end NUMINAMATH_CALUDE_first_square_with_two_twos_l3978_397834


namespace NUMINAMATH_CALUDE_volume_of_extended_box_l3978_397803

/-- Represents a rectangular parallelepiped (box) -/
structure Box where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of the set of points inside or within one unit of a box -/
def volumeWithinOneUnit (b : Box) : ℝ := sorry

/-- Checks if two integers are relatively prime -/
def isRelativelyPrime (a b : ℕ) : Prop := sorry

theorem volume_of_extended_box (m n p : ℕ) :
  (∃ b : Box, b.length = 2 ∧ b.width = 3 ∧ b.height = 6) →
  (∃ v : ℝ, v = volumeWithinOneUnit b) →
  v = (m + n * Real.pi) / p →
  m > 0 ∧ n > 0 ∧ p > 0 →
  isRelativelyPrime n p →
  m + n + p = 364 := by
  sorry

end NUMINAMATH_CALUDE_volume_of_extended_box_l3978_397803


namespace NUMINAMATH_CALUDE_range_of_k_l3978_397895

-- Define the complex number z
variable (z : ℂ)

-- Define sets A and B
def A (m k : ℝ) : Set ℂ :=
  {z | z = (2*m - Real.log (k+1)/k / Real.log (Real.sqrt 2)) + (m + Real.log (k+1)/k / Real.log (Real.sqrt 2)) * Complex.I}

def B (m : ℝ) : Set ℂ :=
  {z | Complex.abs z ≤ 2*m - 1}

-- Define the theorem
theorem range_of_k (m : ℝ) :
  (∀ k : ℝ, (A m k) ∩ (B m) = ∅) ↔ 
  ((4 * Real.sqrt 2 + 1) / 31 < k ∧ k < Real.sqrt 2 + 1) :=
sorry

end NUMINAMATH_CALUDE_range_of_k_l3978_397895


namespace NUMINAMATH_CALUDE_conditional_probability_rhinitis_cold_l3978_397825

theorem conditional_probability_rhinitis_cold 
  (P_rhinitis : ℝ) 
  (P_rhinitis_and_cold : ℝ) 
  (h1 : P_rhinitis = 0.8) 
  (h2 : P_rhinitis_and_cold = 0.6) : 
  P_rhinitis_and_cold / P_rhinitis = 0.75 := by
  sorry

end NUMINAMATH_CALUDE_conditional_probability_rhinitis_cold_l3978_397825


namespace NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l3978_397831

/-- For a parabola given by the equation y^2 = 10x, the distance from its focus to its directrix is 5. -/
theorem parabola_focus_directrix_distance :
  ∀ (y x : ℝ), y^2 = 10*x → (∃ (focus_x focus_y directrix_x : ℝ),
    (∀ (point_x point_y : ℝ), point_y^2 = 10*point_x ↔ 
      (point_x - focus_x)^2 + (point_y - focus_y)^2 = (point_x - directrix_x)^2) ∧
    |focus_x - directrix_x| = 5) :=
by sorry

end NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l3978_397831


namespace NUMINAMATH_CALUDE_art_fair_customers_one_painting_l3978_397856

/-- The number of customers who bought one painting each at Tracy's art fair booth -/
def customers_one_painting (total_customers : ℕ) (two_painting_customers : ℕ) (four_painting_customers : ℕ) (total_paintings_sold : ℕ) : ℕ :=
  total_paintings_sold - (2 * two_painting_customers + 4 * four_painting_customers)

/-- Theorem stating that the number of customers who bought one painting each is 12 -/
theorem art_fair_customers_one_painting :
  customers_one_painting 20 4 4 36 = 12 := by
  sorry

#eval customers_one_painting 20 4 4 36

end NUMINAMATH_CALUDE_art_fair_customers_one_painting_l3978_397856


namespace NUMINAMATH_CALUDE_second_train_length_second_train_length_solution_l3978_397865

/-- Calculates the length of the second train given the speeds of two trains, 
    the time they take to cross each other, and the length of the first train. -/
theorem second_train_length 
  (speed1 : ℝ) 
  (speed2 : ℝ) 
  (crossing_time : ℝ) 
  (length1 : ℝ) : ℝ :=
  let relative_speed := speed1 + speed2
  let relative_speed_mps := relative_speed * (5/18)
  let length2 := relative_speed_mps * crossing_time - length1
  length2

/-- The length of the second train is approximately 159.97 meters. -/
theorem second_train_length_solution :
  ∃ ε > 0, |second_train_length 60 40 11.879049676025918 170 - 159.97| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_second_train_length_second_train_length_solution_l3978_397865


namespace NUMINAMATH_CALUDE_sam_study_time_l3978_397853

theorem sam_study_time (total_hours : ℕ) (science_minutes : ℕ) (literature_minutes : ℕ) 
  (h1 : total_hours = 3)
  (h2 : science_minutes = 60)
  (h3 : literature_minutes = 40) :
  total_hours * 60 - (science_minutes + literature_minutes) = 80 :=
by sorry

end NUMINAMATH_CALUDE_sam_study_time_l3978_397853


namespace NUMINAMATH_CALUDE_teacher_score_calculation_l3978_397893

def teacher_total_score (written_score interview_score : ℝ) (written_weight interview_weight : ℝ) : ℝ :=
  written_score * written_weight + interview_score * interview_weight

theorem teacher_score_calculation :
  let written_score : ℝ := 80
  let interview_score : ℝ := 60
  let written_weight : ℝ := 0.6
  let interview_weight : ℝ := 0.4
  teacher_total_score written_score interview_score written_weight interview_weight = 72 := by
  sorry

end NUMINAMATH_CALUDE_teacher_score_calculation_l3978_397893


namespace NUMINAMATH_CALUDE_two_hats_on_first_maximizes_sum_optimal_distribution_l3978_397866

/-- The number of hats in the hat box -/
def total_hats : ℕ := 21

/-- The number of caps in the hat box -/
def total_caps : ℕ := 18

/-- The capacity of the first shelf -/
def first_shelf_capacity : ℕ := 20

/-- The capacity of the second shelf -/
def second_shelf_capacity : ℕ := 19

/-- The percentage of hats on a shelf given the number of hats and total items -/
def hat_percentage (hats : ℕ) (total : ℕ) : ℚ :=
  (hats : ℚ) / (total : ℚ) * 100

/-- The sum of hat percentages for a given distribution -/
def sum_of_percentages (hats_on_first : ℕ) : ℚ :=
  hat_percentage hats_on_first first_shelf_capacity +
  hat_percentage (total_hats - hats_on_first) second_shelf_capacity

/-- Theorem stating that 2 hats on the first shelf maximizes the sum of percentages -/
theorem two_hats_on_first_maximizes_sum :
  ∀ x : ℕ, x ≤ total_hats → sum_of_percentages 2 ≥ sum_of_percentages x :=
sorry

/-- Corollary stating the optimal distribution of hats -/
theorem optimal_distribution :
  sum_of_percentages 2 = hat_percentage 2 first_shelf_capacity +
                         hat_percentage 19 second_shelf_capacity :=
sorry

end NUMINAMATH_CALUDE_two_hats_on_first_maximizes_sum_optimal_distribution_l3978_397866


namespace NUMINAMATH_CALUDE_min_blocks_removed_for_cube_l3978_397899

/-- Given 59 cubic blocks, the minimum number of blocks that need to be taken away
    to construct a solid cube with none left over is 32. -/
theorem min_blocks_removed_for_cube (total_blocks : ℕ) (h : total_blocks = 59) :
  ∃ (n : ℕ), n^3 ≤ total_blocks ∧
             ∀ (m : ℕ), m^3 ≤ total_blocks → m ≤ n ∧
             total_blocks - n^3 = 32 :=
by sorry

end NUMINAMATH_CALUDE_min_blocks_removed_for_cube_l3978_397899


namespace NUMINAMATH_CALUDE_functional_equation_solution_l3978_397804

-- Define the function type
def RealFunction := ℝ → ℝ

-- State the theorem
theorem functional_equation_solution (f : RealFunction) :
  (∀ x y t : ℝ, f (x + t + f y) = f (f x) + f t + y) →
  (∀ x : ℝ, f x = x) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l3978_397804


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l3978_397892

theorem partial_fraction_decomposition :
  let f (x : ℚ) := (7 * x - 4) / (x^2 - 9*x - 18)
  let g (x : ℚ) := 59 / (11 * (x - 9)) + 18 / (11 * (x + 2))
  ∀ x, x ≠ 9 ∧ x ≠ -2 → f x = g x :=
by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l3978_397892


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l3978_397854

theorem trigonometric_equation_solution (x : ℝ) :
  (Real.sin x)^3 + 6 * (Real.cos x)^3 + (1 / Real.sqrt 2) * Real.sin (2 * x) * Real.sin (x + π / 4) = 0 →
  ∃ n : ℤ, x = -Real.arctan 2 + n * π :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l3978_397854


namespace NUMINAMATH_CALUDE_lcm_gcf_ratio_l3978_397847

theorem lcm_gcf_ratio : 
  (Nat.lcm 252 630) / (Nat.gcd 252 630) = 10 := by sorry

end NUMINAMATH_CALUDE_lcm_gcf_ratio_l3978_397847


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l3978_397887

theorem arithmetic_expression_equality : 8 / 4 - 3 * 2 + 9 - 3^2 = -4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l3978_397887


namespace NUMINAMATH_CALUDE_next_monday_birthday_l3978_397873

/-- Represents the day of the week -/
inductive DayOfWeek
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday
| Sunday

/-- Determines if a year is a leap year -/
def isLeapYear (year : Nat) : Bool :=
  year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)

/-- Calculates the day of the week for March 15 in a given year, 
    assuming March 15, 2012 was a Friday -/
def marchFifteenDayOfWeek (year : Nat) : DayOfWeek :=
  sorry

/-- Theorem: The next year after 2012 when March 15 falls on a Monday is 2025 -/
theorem next_monday_birthday (startYear : Nat) (startDay : DayOfWeek) :
  startYear = 2012 →
  startDay = DayOfWeek.Friday →
  (∀ y, startYear < y → y < 2025 → marchFifteenDayOfWeek y ≠ DayOfWeek.Monday) →
  marchFifteenDayOfWeek 2025 = DayOfWeek.Monday :=
by sorry

end NUMINAMATH_CALUDE_next_monday_birthday_l3978_397873


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3978_397846

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 > 0) ↔ (∃ x : ℝ, x^2 ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3978_397846


namespace NUMINAMATH_CALUDE_little_red_riding_hood_waffles_l3978_397886

theorem little_red_riding_hood_waffles (initial_waffles : ℕ) : 
  (∃ (x : ℕ), 
    initial_waffles = 14 * x ∧ 
    (initial_waffles / 2 - x) / 2 - x = x ∧
    x > 0) →
  initial_waffles % 7 = 0 :=
sorry

end NUMINAMATH_CALUDE_little_red_riding_hood_waffles_l3978_397886


namespace NUMINAMATH_CALUDE_meaningful_expression_range_l3978_397811

theorem meaningful_expression_range (x : ℝ) :
  (∃ y : ℝ, y = Real.sqrt (1 - x) ∧ x + 2 ≠ 0) ↔ x ≤ 1 ∧ x ≠ -2 := by sorry

end NUMINAMATH_CALUDE_meaningful_expression_range_l3978_397811


namespace NUMINAMATH_CALUDE_minimum_phrases_to_study_l3978_397821

/-- 
Given a total of 800 French phrases and a required quiz score of 90%,
prove that the minimum number of phrases to study is 720.
-/
theorem minimum_phrases_to_study (total_phrases : ℕ) (required_score : ℚ) : 
  total_phrases = 800 → required_score = 90 / 100 → 
  ⌈(required_score * total_phrases : ℚ)⌉ = 720 := by
sorry

end NUMINAMATH_CALUDE_minimum_phrases_to_study_l3978_397821


namespace NUMINAMATH_CALUDE_min_transportation_cost_l3978_397868

-- Define the problem parameters
def total_items : ℕ := 320
def water_excess : ℕ := 80
def type_a_water_capacity : ℕ := 40
def type_a_veg_capacity : ℕ := 10
def type_b_capacity : ℕ := 20
def total_trucks : ℕ := 8
def type_a_cost : ℕ := 400
def type_b_cost : ℕ := 360

-- Define the transportation cost function
def transportation_cost (num_type_a : ℕ) : ℕ :=
  type_a_cost * num_type_a + type_b_cost * (total_trucks - num_type_a)

-- Theorem statement
theorem min_transportation_cost :
  ∃ (num_water num_veg : ℕ),
    num_water + num_veg = total_items ∧
    num_water - num_veg = water_excess ∧
    (∀ (num_type_a : ℕ),
      2 ≤ num_type_a ∧ num_type_a ≤ 4 →
      type_a_water_capacity * num_type_a + type_b_capacity * (total_trucks - num_type_a) ≥ num_water ∧
      type_a_veg_capacity * num_type_a + type_b_capacity * (total_trucks - num_type_a) ≥ num_veg) ∧
    (∀ (num_type_a : ℕ),
      2 ≤ num_type_a ∧ num_type_a ≤ 4 →
      transportation_cost 2 ≤ transportation_cost num_type_a) ∧
    transportation_cost 2 = 2960 := by
  sorry

end NUMINAMATH_CALUDE_min_transportation_cost_l3978_397868


namespace NUMINAMATH_CALUDE_product_abcd_equals_162_over_185_l3978_397800

theorem product_abcd_equals_162_over_185 
  (a b c d : ℚ) 
  (eq1 : 3*a + 4*b + 6*c + 9*d = 45)
  (eq2 : 4*(d+c) = b + 1)
  (eq3 : 4*b + 2*c = a)
  (eq4 : 2*c - 2 = d) :
  a * b * c * d = 162 / 185 := by
sorry

end NUMINAMATH_CALUDE_product_abcd_equals_162_over_185_l3978_397800


namespace NUMINAMATH_CALUDE_tan_22_5_degrees_l3978_397810

theorem tan_22_5_degrees :
  Real.tan (22.5 * π / 180) = Real.sqrt 8 - Real.sqrt 0 - 2 := by sorry

end NUMINAMATH_CALUDE_tan_22_5_degrees_l3978_397810


namespace NUMINAMATH_CALUDE_rectangle_width_l3978_397898

theorem rectangle_width (w : ℝ) (h1 : w > 0) : 
  (2 * w * w = 3 * 2 * (2 * w + w)) → w = 9 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_width_l3978_397898


namespace NUMINAMATH_CALUDE_proposition_correctness_l3978_397871

theorem proposition_correctness : ∃ (p1 p2 p3 p4 : Prop),
  -- Proposition 1
  (∀ x : ℝ, x = 1 → x^2 - 3*x + 2 = 0) ∧
  (∃ x : ℝ, x^2 - 3*x + 2 = 0 ∧ x ≠ 1) ∧
  
  -- Proposition 2
  (∀ x : ℝ, x ≠ 1 → x^2 - 3*x + 2 ≠ 0) ↔ (∀ x : ℝ, x^2 - 3*x + 2 = 0 → x = 1) ∧
  
  -- Proposition 3
  (¬(∃ x : ℝ, x > 0 ∧ x^2 + x + 1 < 0) ↔ (∀ x : ℝ, x > 0 → x^2 + x + 1 ≥ 0)) ∧
  
  -- Proposition 4
  (∀ p q : Prop, ¬(p ∨ q) → (¬p ∧ ¬q)) ∧
  
  -- Exactly 3 out of 4 propositions are correct
  (p1 ∧ p2 ∧ ¬p3 ∧ p4) :=
by
  sorry

#check proposition_correctness

end NUMINAMATH_CALUDE_proposition_correctness_l3978_397871


namespace NUMINAMATH_CALUDE_cost_for_36_people_l3978_397863

/-- The cost to feed a group of people with chicken combos -/
def cost_to_feed (people : ℕ) (combo_cost : ℚ) (people_per_combo : ℕ) : ℚ :=
  (people / people_per_combo : ℚ) * combo_cost

/-- Theorem: The cost to feed 36 people is $72.00 -/
theorem cost_for_36_people :
  cost_to_feed 36 12 6 = 72 := by
  sorry

end NUMINAMATH_CALUDE_cost_for_36_people_l3978_397863


namespace NUMINAMATH_CALUDE_average_of_first_five_multiples_of_five_l3978_397801

theorem average_of_first_five_multiples_of_five :
  let multiples : List ℕ := [5, 10, 15, 20, 25]
  (multiples.sum / multiples.length : ℚ) = 15 := by
  sorry

end NUMINAMATH_CALUDE_average_of_first_five_multiples_of_five_l3978_397801


namespace NUMINAMATH_CALUDE_only_negative_one_point_one_less_than_negative_one_l3978_397802

theorem only_negative_one_point_one_less_than_negative_one :
  let numbers : List ℝ := [0, 1, -0.9, -1.1]
  ∀ x ∈ numbers, x < -1 ↔ x = -1.1 :=
by sorry

end NUMINAMATH_CALUDE_only_negative_one_point_one_less_than_negative_one_l3978_397802


namespace NUMINAMATH_CALUDE_unique_consecutive_sum_30_l3978_397870

/-- A set of consecutive positive integers -/
structure ConsecutiveSet :=
  (start : ℕ)
  (length : ℕ)
  (h_length : length ≥ 2)

/-- The sum of a set of consecutive positive integers -/
def sum_consecutive (s : ConsecutiveSet) : ℕ :=
  (s.length * (2 * s.start + s.length - 1)) / 2

/-- Theorem: There is exactly one set of consecutive positive integers whose sum is 30 -/
theorem unique_consecutive_sum_30 :
  ∃! s : ConsecutiveSet, sum_consecutive s = 30 :=
sorry

end NUMINAMATH_CALUDE_unique_consecutive_sum_30_l3978_397870


namespace NUMINAMATH_CALUDE_x_expression_l3978_397826

theorem x_expression (m n x : ℝ) (h1 : m ≠ n) (h2 : m ≠ 0) (h3 : n ≠ 0) 
  (h4 : (x + 2*m)^2 - 2*(x + n)^2 = 2*(m - n)^2) : 
  x = 2*m - 2*n := by
sorry

end NUMINAMATH_CALUDE_x_expression_l3978_397826


namespace NUMINAMATH_CALUDE_triangular_number_formula_l3978_397828

/-- The triangular number sequence -/
def triangular_number : ℕ → ℕ
| 0 => 0
| (n + 1) => triangular_number n + n + 1

/-- Theorem: The nth triangular number is equal to n(n+1)/2 -/
theorem triangular_number_formula (n : ℕ) :
  triangular_number n = n * (n + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangular_number_formula_l3978_397828


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_2012_l3978_397809

/-- An arithmetic sequence with specified properties -/
structure ArithmeticSequence where
  a : ℕ → ℤ
  first_term : a 1 = -2012
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  (n : ℤ) * seq.a 1 + (n * (n - 1) : ℤ) * (seq.a 2 - seq.a 1) / 2

theorem arithmetic_sequence_sum_2012 (seq : ArithmeticSequence) 
    (h : (sum_n seq 12 / 12 : ℚ) - (sum_n seq 10 / 10 : ℚ) = 2) :
    sum_n seq 2012 = -2012 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_2012_l3978_397809


namespace NUMINAMATH_CALUDE_business_value_l3978_397814

/-- Proves the value of a business given partial ownership and sale information -/
theorem business_value (
  total_shares : ℚ)
  (owner_share : ℚ)
  (sold_fraction : ℚ)
  (sale_price : ℚ)
  (h1 : owner_share = 1 / 3)
  (h2 : sold_fraction = 3 / 5)
  (h3 : sale_price = 15000) :
  total_shares = 75000 := by
  sorry

end NUMINAMATH_CALUDE_business_value_l3978_397814


namespace NUMINAMATH_CALUDE_octavia_photos_count_l3978_397816

/-- Represents the number of photographs in a photography exhibition --/
structure PhotoExhibition where
  total : ℕ
  octavia_photos : ℕ
  jack_framed : ℕ
  jack_framed_octavia : ℕ
  jack_framed_others : ℕ

/-- The photography exhibition satisfies the given conditions --/
def exhibition_conditions (e : PhotoExhibition) : Prop :=
  e.jack_framed_octavia = 24 ∧
  e.jack_framed_others = 12 ∧
  e.jack_framed = e.jack_framed_octavia + e.jack_framed_others ∧
  e.total = 48 ∧
  e.total = e.octavia_photos + e.jack_framed - e.jack_framed_octavia

/-- Theorem stating that under the given conditions, Octavia took 36 photographs --/
theorem octavia_photos_count (e : PhotoExhibition) 
  (h : exhibition_conditions e) : e.octavia_photos = 36 := by
  sorry


end NUMINAMATH_CALUDE_octavia_photos_count_l3978_397816


namespace NUMINAMATH_CALUDE_center_trajectory_of_circle_family_l3978_397879

-- Define the family of circles
def circle_family (t x y : ℝ) : Prop :=
  x^2 + y^2 - 4*t*x - 2*t*y + 3*t^2 - 4 = 0

-- Define the trajectory of centers
def center_trajectory (t x y : ℝ) : Prop :=
  x = 2*t ∧ y = t

-- Theorem statement
theorem center_trajectory_of_circle_family :
  ∀ t : ℝ, ∃ x y : ℝ,
    circle_family t x y ↔ center_trajectory t x y :=
sorry

end NUMINAMATH_CALUDE_center_trajectory_of_circle_family_l3978_397879


namespace NUMINAMATH_CALUDE_intersection_theorem_l3978_397832

-- Define set A
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 3}

-- Define set B
def B : Set ℝ := {y | ∃ x, y = x^2 + 2}

-- Define the intersection of A and B
def A_intersect_B : Set ℝ := A ∩ B

-- Theorem statement
theorem intersection_theorem : A_intersect_B = {x | 2 ≤ x ∧ x ≤ 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_theorem_l3978_397832


namespace NUMINAMATH_CALUDE_ruffy_orlie_age_difference_l3978_397888

/-- Proves that given Ruffy's current age is 9 and Ruffy is three-fourths as old as Orlie,
    the difference between Ruffy's age and half of Orlie's age four years ago is 1 year. -/
theorem ruffy_orlie_age_difference : ∀ (ruffy_age orlie_age : ℕ),
  ruffy_age = 9 →
  ruffy_age = (3 * orlie_age) / 4 →
  (ruffy_age - 4) - ((orlie_age - 4) / 2) = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ruffy_orlie_age_difference_l3978_397888


namespace NUMINAMATH_CALUDE_common_rational_root_l3978_397819

theorem common_rational_root (a b c d e f g : ℚ) : 
  ∃ (p : ℚ), p = -1/2 ∧ 
  48 * p^4 + a * p^3 + b * p^2 + c * p + 16 = 0 ∧
  16 * p^5 + d * p^4 + e * p^3 + f * p^2 + g * p + 48 = 0 ∧
  ¬(∃ (n : ℤ), p = n) ∧ 
  p < 0 := by
sorry


end NUMINAMATH_CALUDE_common_rational_root_l3978_397819


namespace NUMINAMATH_CALUDE_toilet_paper_packs_is_14_l3978_397896

/-- The number of packs of toilet paper Stella needs to buy after 4 weeks -/
def toilet_paper_packs : ℕ :=
  let bathrooms : ℕ := 6
  let days_per_week : ℕ := 7
  let rolls_per_pack : ℕ := 12
  let weeks : ℕ := 4
  let rolls_per_day : ℕ := bathrooms
  let rolls_per_week : ℕ := rolls_per_day * days_per_week
  let total_rolls : ℕ := rolls_per_week * weeks
  total_rolls / rolls_per_pack

theorem toilet_paper_packs_is_14 : toilet_paper_packs = 14 := by
  sorry

end NUMINAMATH_CALUDE_toilet_paper_packs_is_14_l3978_397896


namespace NUMINAMATH_CALUDE_difference_of_squares_l3978_397839

theorem difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l3978_397839


namespace NUMINAMATH_CALUDE_diagonals_25_sided_polygon_l3978_397823

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A convex polygon with 25 sides has 275 diagonals -/
theorem diagonals_25_sided_polygon : num_diagonals 25 = 275 := by
  sorry

end NUMINAMATH_CALUDE_diagonals_25_sided_polygon_l3978_397823


namespace NUMINAMATH_CALUDE_max_third_side_length_l3978_397883

theorem max_third_side_length (a b : ℝ) (ha : a = 6) (hb : b = 10) :
  ∃ (s : ℕ), s ≤ 15 ∧ 
  (∀ (t : ℕ), (t : ℝ) < a + b ∧ a < (t : ℝ) + b ∧ b < a + (t : ℝ) → t ≤ s) ∧
  ((15 : ℝ) < a + b ∧ a < 15 + b ∧ b < a + 15) :=
by sorry

end NUMINAMATH_CALUDE_max_third_side_length_l3978_397883


namespace NUMINAMATH_CALUDE_probability_standard_deck_l3978_397894

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : Nat)
  (red_cards : Nat)
  (black_cards : Nat)

/-- Probability of drawing two red cards followed by two black cards -/
def probability_two_red_two_black (d : Deck) : Rat :=
  if d.total_cards ≥ 4 ∧ d.red_cards ≥ 2 ∧ d.black_cards ≥ 2 then
    (d.red_cards * (d.red_cards - 1) * d.black_cards * (d.black_cards - 1)) /
    (d.total_cards * (d.total_cards - 1) * (d.total_cards - 2) * (d.total_cards - 3))
  else
    0

theorem probability_standard_deck :
  probability_two_red_two_black ⟨52, 26, 26⟩ = 325 / 4998 := by
  sorry

end NUMINAMATH_CALUDE_probability_standard_deck_l3978_397894


namespace NUMINAMATH_CALUDE_chess_match_duration_l3978_397859

-- Define the given conditions
def polly_time_per_move : ℕ := 28
def peter_time_per_move : ℕ := 40
def total_moves : ℕ := 30

-- Define the theorem
theorem chess_match_duration :
  (total_moves / 2 * polly_time_per_move + total_moves / 2 * peter_time_per_move) / 60 = 17 := by
  sorry

end NUMINAMATH_CALUDE_chess_match_duration_l3978_397859


namespace NUMINAMATH_CALUDE_B_is_largest_l3978_397852

def A : ℚ := 2010 / 2009 + 2010 / 2011
def B : ℚ := 2010 / 2011 + 2012 / 2011
def C : ℚ := 2011 / 2010 + 2011 / 2012

theorem B_is_largest : B > A ∧ B > C := by
  sorry

end NUMINAMATH_CALUDE_B_is_largest_l3978_397852
