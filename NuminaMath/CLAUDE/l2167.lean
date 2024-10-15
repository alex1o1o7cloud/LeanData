import Mathlib

namespace NUMINAMATH_CALUDE_locus_of_fourth_vertex_l2167_216782

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a circle with center and radius -/
structure Circle :=
  (center : Point) (radius : ℝ)

/-- Checks if a point lies on a circle -/
def lies_on_circle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

/-- Represents a rectangle by its vertices -/
structure Rectangle :=
  (A B C D : Point)

theorem locus_of_fourth_vertex 
  (O : Point) (r R : ℝ) (hr : 0 < r) (hR : r < R)
  (c1 : Circle) (c2 : Circle) (rect : Rectangle)
  (hc1 : c1 = ⟨O, r⟩) (hc2 : c2 = ⟨O, R⟩)
  (hA : lies_on_circle rect.A c2 ∨ lies_on_circle rect.A c1)
  (hB : lies_on_circle rect.B c2 ∨ lies_on_circle rect.B c1)
  (hD : lies_on_circle rect.D c2 ∨ lies_on_circle rect.D c1) :
  lies_on_circle rect.C c1 ∨ lies_on_circle rect.C c2 ∨
  (lies_on_circle rect.C c1 ∧ 
   (rect.C.x - O.x)^2 + (rect.C.y - O.y)^2 + 
   (rect.B.x - O.x)^2 + (rect.B.y - O.y)^2 = 2 * R^2) :=
sorry

end NUMINAMATH_CALUDE_locus_of_fourth_vertex_l2167_216782


namespace NUMINAMATH_CALUDE_p_2017_equals_14_l2167_216743

/-- Function that calculates the sum of digits of a positive integer -/
def sumOfDigits (n : ℕ+) : ℕ := sorry

/-- Function that calculates the number of digits of a positive integer -/
def numberOfDigits (n : ℕ+) : ℕ := sorry

/-- Function P(n) as defined in the problem -/
def P (n : ℕ+) : ℕ := sumOfDigits n + numberOfDigits n

/-- Theorem stating that P(2017) = 14 -/
theorem p_2017_equals_14 : P 2017 = 14 := by sorry

end NUMINAMATH_CALUDE_p_2017_equals_14_l2167_216743


namespace NUMINAMATH_CALUDE_max_distance_between_circles_l2167_216730

/-- The maximum distance between a point on circle1 and a point on circle2 -/
theorem max_distance_between_circles (M N : ℝ × ℝ) : 
  (∃ x y, M = (x, y) ∧ (x - 3/2)^2 + y^2 = 23/4) →
  (∃ x y, N = (x, y) ∧ (x + 5)^2 + y^2 = 1) →
  (∀ M' N', 
    (∃ x y, M' = (x, y) ∧ (x - 3/2)^2 + y^2 = 23/4) →
    (∃ x y, N' = (x, y) ∧ (x + 5)^2 + y^2 = 1) →
    Real.sqrt ((M'.1 - N'.1)^2 + (M'.2 - N'.2)^2) ≤ Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2)) →
  Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2) = (15 + Real.sqrt 23) / 2 := by
sorry

end NUMINAMATH_CALUDE_max_distance_between_circles_l2167_216730


namespace NUMINAMATH_CALUDE_min_value_expression_l2167_216773

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 1) (hab : a + b = 2) :
  (∀ x y : ℝ, x > 0 ∧ y > 1 ∧ x + y = 2 → 4/x + 1/(y-1) ≥ 4/a + 1/(b-1)) →
  4/a + 1/(b-1) = 9 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2167_216773


namespace NUMINAMATH_CALUDE_rectangular_floor_length_l2167_216725

theorem rectangular_floor_length (floor_width : ℝ) (square_size : ℝ) (num_squares : ℕ) :
  floor_width = 6 →
  square_size = 2 →
  num_squares = 15 →
  floor_width * (num_squares * square_size^2 / floor_width) = 10 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_floor_length_l2167_216725


namespace NUMINAMATH_CALUDE_strawberry_calculation_l2167_216742

theorem strawberry_calculation (initial : ℝ) (sold : ℝ) (given_away : ℝ) (eaten : ℝ) 
  (h1 : initial = 120.5)
  (h2 : sold = 8.25)
  (h3 : given_away = 33.5)
  (h4 : eaten = 4.3) :
  initial - sold - given_away - eaten = 74.45 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_calculation_l2167_216742


namespace NUMINAMATH_CALUDE_city_distance_proof_l2167_216759

/-- Given a map distance between two cities and a map scale, calculates the actual distance between the cities. -/
def actualDistance (mapDistance : ℝ) (mapScale : ℝ) : ℝ :=
  mapDistance * mapScale

/-- Theorem stating that for a map distance of 120 cm and a scale of 1 cm : 20 km, the actual distance is 2400 km. -/
theorem city_distance_proof :
  let mapDistance : ℝ := 120
  let mapScale : ℝ := 20
  actualDistance mapDistance mapScale = 2400 := by
  sorry

#eval actualDistance 120 20

end NUMINAMATH_CALUDE_city_distance_proof_l2167_216759


namespace NUMINAMATH_CALUDE_diamond_commutative_l2167_216780

-- Define the diamond operation
def diamond (a b : ℝ) : ℝ := a^2 * b^2 - a^2 - b^2

-- Theorem statement
theorem diamond_commutative : ∀ x y : ℝ, diamond x y = diamond y x := by
  sorry

end NUMINAMATH_CALUDE_diamond_commutative_l2167_216780


namespace NUMINAMATH_CALUDE_invalid_votes_percentage_l2167_216760

theorem invalid_votes_percentage (total_votes : ℕ) (winning_percentage : ℚ) (losing_votes : ℕ) :
  total_votes = 7500 →
  winning_percentage = 55 / 100 →
  losing_votes = 2700 →
  (total_votes - (losing_votes / (1 - winning_percentage))) / total_votes = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_invalid_votes_percentage_l2167_216760


namespace NUMINAMATH_CALUDE_smallest_of_seven_consecutive_evens_l2167_216798

/-- Given seven consecutive even integers whose sum is 448, 
    the smallest of these numbers is 58. -/
theorem smallest_of_seven_consecutive_evens (a : ℤ) : 
  (∃ n : ℤ, a = 2*n ∧ 
   (a + (a+2) + (a+4) + (a+6) + (a+8) + (a+10) + (a+12) = 448)) → 
  a = 58 := by
sorry

end NUMINAMATH_CALUDE_smallest_of_seven_consecutive_evens_l2167_216798


namespace NUMINAMATH_CALUDE_vegetable_seedling_price_l2167_216770

theorem vegetable_seedling_price (base_price : ℚ) : 
  (300 / base_price - 300 / (5/4 * base_price) = 3) → base_price = 20 := by
  sorry

end NUMINAMATH_CALUDE_vegetable_seedling_price_l2167_216770


namespace NUMINAMATH_CALUDE_fair_walking_distance_l2167_216722

theorem fair_walking_distance (total_distance : ℝ) (short_segment : ℝ) 
  (h1 : total_distance = 0.75)
  (h2 : short_segment = 0.08)
  (h3 : ∃ x : ℝ, total_distance = 2 * x + short_segment) :
  ∃ x : ℝ, x = 0.335 ∧ total_distance = 2 * x + short_segment :=
sorry

end NUMINAMATH_CALUDE_fair_walking_distance_l2167_216722


namespace NUMINAMATH_CALUDE_two_numbers_with_special_properties_l2167_216705

theorem two_numbers_with_special_properties : ∃ (a b : ℕ), 
  a ≠ b ∧
  a > 9 ∧ b > 9 ∧
  (a + b) / 2 ≥ 10 ∧ (a + b) / 2 ≤ 99 ∧
  Nat.sqrt (a * b) ≥ 10 ∧ Nat.sqrt (a * b) ≤ 99 ∧
  (a = 98 ∧ b = 32 ∨ a = 32 ∧ b = 98) :=
by sorry

end NUMINAMATH_CALUDE_two_numbers_with_special_properties_l2167_216705


namespace NUMINAMATH_CALUDE_total_teachers_is_182_l2167_216707

/-- Represents the number of teachers in different categories and survey selections -/
structure SchoolTeachers where
  senior : ℕ
  intermediate : ℕ
  survey_total : ℕ
  survey_other : ℕ

/-- Calculates the total number of teachers in the school -/
def total_teachers (s : SchoolTeachers) : ℕ :=
  s.senior + s.intermediate + (s.survey_total - (s.survey_other + s.senior + s.intermediate))

/-- Theorem stating that given the specific numbers, the total teachers is 182 -/
theorem total_teachers_is_182 (s : SchoolTeachers) 
  (h1 : s.senior = 26)
  (h2 : s.intermediate = 104)
  (h3 : s.survey_total = 56)
  (h4 : s.survey_other = 16) :
  total_teachers s = 182 := by
  sorry

#eval total_teachers ⟨26, 104, 56, 16⟩

end NUMINAMATH_CALUDE_total_teachers_is_182_l2167_216707


namespace NUMINAMATH_CALUDE_polynomial_derivative_sum_l2167_216767

theorem polynomial_derivative_sum (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (2*x - 3)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₁ + 2*a₂ + 3*a₃ + 4*a₄ + 5*a₅ = 20 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_derivative_sum_l2167_216767


namespace NUMINAMATH_CALUDE_composition_of_transformations_l2167_216775

-- Define the transformations
def f (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)
def g (p : ℝ × ℝ) : ℝ × ℝ := (p.1, p.1 - p.2)

-- State the theorem
theorem composition_of_transformations :
  f (g (-1, 2)) = (1, -3) := by sorry

end NUMINAMATH_CALUDE_composition_of_transformations_l2167_216775


namespace NUMINAMATH_CALUDE_max_value_of_f_l2167_216751

def f (x : ℝ) : ℝ := 3 * x - 4 * x^3

theorem max_value_of_f :
  ∃ (c : ℝ), c ∈ Set.Icc 0 1 ∧ 
  (∀ x, x ∈ Set.Icc 0 1 → f x ≤ f c) ∧
  f c = 1 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_f_l2167_216751


namespace NUMINAMATH_CALUDE_infinite_matrices_squared_zero_l2167_216745

/-- The set of 2x2 real matrices B satisfying B^2 = 0 is infinite -/
theorem infinite_matrices_squared_zero :
  Set.Infinite {B : Matrix (Fin 2) (Fin 2) ℝ | B * B = 0} := by
  sorry

end NUMINAMATH_CALUDE_infinite_matrices_squared_zero_l2167_216745


namespace NUMINAMATH_CALUDE_staircase_region_perimeter_l2167_216704

/-- Represents the staircase-shaped region with an adjoined right triangle -/
structure StaircaseRegion where
  staircase_side_length : ℝ
  staircase_side_count : ℕ
  triangle_leg1 : ℝ
  triangle_leg2 : ℝ
  total_area : ℝ

/-- Calculates the perimeter of the StaircaseRegion -/
def calculate_perimeter (region : StaircaseRegion) : ℝ :=
  sorry

/-- Theorem stating the perimeter of the specific StaircaseRegion -/
theorem staircase_region_perimeter :
  let region : StaircaseRegion := {
    staircase_side_length := 2,
    staircase_side_count := 10,
    triangle_leg1 := 3,
    triangle_leg2 := 4,
    total_area := 150
  }
  calculate_perimeter region = 81.77 := by
  sorry

end NUMINAMATH_CALUDE_staircase_region_perimeter_l2167_216704


namespace NUMINAMATH_CALUDE_coin_problem_l2167_216720

theorem coin_problem (x y : ℕ) : 
  x + y = 12 →
  5 * x + 10 * y = 90 →
  x = 6 ∧ y = 6 :=
by sorry

end NUMINAMATH_CALUDE_coin_problem_l2167_216720


namespace NUMINAMATH_CALUDE_range_of_M_l2167_216721

theorem range_of_M (a θ : ℝ) (ha : a ≠ 0) :
  let M := (a^2 - a * Real.sin θ + 1) / (a^2 - a * Real.cos θ + 1)
  (4 - Real.sqrt 7) / 3 ≤ M ∧ M ≤ (4 + Real.sqrt 7) / 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_M_l2167_216721


namespace NUMINAMATH_CALUDE_count_eights_theorem_l2167_216741

/-- Count of digit 8 appearances in integers from 1 to 800 -/
def count_eights : ℕ := 160

/-- The upper bound of the integer range -/
def upper_bound : ℕ := 800

/-- Counts the occurrences of a specific digit in a given range of integers -/
def count_digit_occurrences (digit : ℕ) (lower : ℕ) (upper : ℕ) : ℕ :=
  sorry

theorem count_eights_theorem :
  count_digit_occurrences 8 1 upper_bound = count_eights :=
sorry

end NUMINAMATH_CALUDE_count_eights_theorem_l2167_216741


namespace NUMINAMATH_CALUDE_sqrt_real_implies_x_geq_8_l2167_216796

theorem sqrt_real_implies_x_geq_8 (x : ℝ) : (∃ y : ℝ, y ^ 2 = x - 8) → x ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_real_implies_x_geq_8_l2167_216796


namespace NUMINAMATH_CALUDE_min_cuts_for_polygons_l2167_216747

/-- Represents the number of sides in the target polygons -/
def target_sides : Nat := 20

/-- Represents the number of target polygons to be created -/
def num_polygons : Nat := 3

/-- Represents the initial number of vertices in the rectangular sheet -/
def initial_vertices : Nat := 4

/-- Represents the maximum increase in vertices per cut -/
def max_vertex_increase : Nat := 4

/-- Theorem stating the minimum number of cuts required -/
theorem min_cuts_for_polygons : 
  ∃ (n : Nat), n = 50 ∧ 
  (∀ m : Nat, m < n → 
    (m + 1) * initial_vertices + m * max_vertex_increase < 
    num_polygons * target_sides + 3 * (m + 1 - num_polygons)) ∧
  ((n + 1) * initial_vertices + n * max_vertex_increase ≥ 
    num_polygons * target_sides + 3 * (n + 1 - num_polygons)) := by
  sorry

end NUMINAMATH_CALUDE_min_cuts_for_polygons_l2167_216747


namespace NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_ten_satisfies_inequality_ten_is_smallest_satisfying_integer_l2167_216794

theorem smallest_integer_satisfying_inequality :
  ∀ n : ℤ, n^2 - 14*n + 45 > 0 → n ≥ 10 :=
by sorry

theorem ten_satisfies_inequality :
  10^2 - 14*10 + 45 > 0 :=
by sorry

theorem ten_is_smallest_satisfying_integer :
  ∀ n : ℤ, n < 10 → n^2 - 14*n + 45 ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_ten_satisfies_inequality_ten_is_smallest_satisfying_integer_l2167_216794


namespace NUMINAMATH_CALUDE_binomial_coefficient_divisible_by_prime_l2167_216750

theorem binomial_coefficient_divisible_by_prime (p k : ℕ) :
  Nat.Prime p → 1 ≤ k → k ≤ p - 1 → p ∣ Nat.choose p k := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_divisible_by_prime_l2167_216750


namespace NUMINAMATH_CALUDE_total_candy_is_54_l2167_216789

/-- The number of students in the group -/
def num_students : ℕ := 9

/-- The number of chocolate pieces given to each student -/
def chocolate_per_student : ℕ := 2

/-- The number of hard candy pieces given to each student -/
def hard_candy_per_student : ℕ := 3

/-- The number of gummy candy pieces given to each student -/
def gummy_per_student : ℕ := 1

/-- The total number of candy pieces given away -/
def total_candy : ℕ := num_students * (chocolate_per_student + hard_candy_per_student + gummy_per_student)

theorem total_candy_is_54 : total_candy = 54 := by
  sorry

end NUMINAMATH_CALUDE_total_candy_is_54_l2167_216789


namespace NUMINAMATH_CALUDE_ding_xiaole_jogging_distances_l2167_216766

/-- Represents the jogging distances for 4 days -/
structure JoggingData :=
  (days : Nat)
  (max_daily : ℝ)
  (min_daily : ℝ)

/-- Calculates the maximum total distance for the given jogging data -/
def max_total_distance (data : JoggingData) : ℝ :=
  data.max_daily * (data.days - 1) + data.min_daily

/-- Calculates the minimum total distance for the given jogging data -/
def min_total_distance (data : JoggingData) : ℝ :=
  data.min_daily * (data.days - 1) + data.max_daily

/-- Theorem stating the maximum and minimum total distances for Ding Xiaole's jogging -/
theorem ding_xiaole_jogging_distances :
  let data : JoggingData := ⟨4, 3.3, 2.4⟩
  max_total_distance data = 12.3 ∧ min_total_distance data = 10.5 := by
  sorry

end NUMINAMATH_CALUDE_ding_xiaole_jogging_distances_l2167_216766


namespace NUMINAMATH_CALUDE_ten_people_handshakes_l2167_216793

/-- Represents the number of handshakes in a group where each person shakes hands only with those taller than themselves. -/
def handshakes (n : ℕ) : ℕ :=
  (n * (n - 1)) / 2

/-- Theorem stating that in a group of 10 people where each person shakes hands only with those taller than themselves, the total number of handshakes is 45. -/
theorem ten_people_handshakes :
  handshakes 10 = 45 := by
  sorry

#eval handshakes 10  -- Should output 45

end NUMINAMATH_CALUDE_ten_people_handshakes_l2167_216793


namespace NUMINAMATH_CALUDE_sixty_one_invalid_l2167_216732

/-- Represents the seat numbers of selected students -/
def selected_seats : List Nat := [5, 16, 27, 38, 49]

/-- The number of selected students -/
def num_selected : Nat := 5

/-- Checks if the given number can be the total number of students in the class -/
def is_valid_class_size (x : Nat) : Prop :=
  ∃ k, x = k * (num_selected - 1) + selected_seats.head!

/-- Theorem stating that 61 cannot be the number of students in the class -/
theorem sixty_one_invalid : ¬ is_valid_class_size 61 := by
  sorry


end NUMINAMATH_CALUDE_sixty_one_invalid_l2167_216732


namespace NUMINAMATH_CALUDE_student_grade_problem_l2167_216791

theorem student_grade_problem (grade2 grade3 overall : ℚ) :
  grade2 = 80 →
  grade3 = 75 →
  overall = 75 →
  ∃ grade1 : ℚ, (grade1 + grade2 + grade3) / 3 = overall ∧ grade1 = 70 :=
by sorry

end NUMINAMATH_CALUDE_student_grade_problem_l2167_216791


namespace NUMINAMATH_CALUDE_circular_arrangement_students_l2167_216718

/-- Given a circular arrangement of students, if the 10th and 45th positions
    are opposite each other, then the total number of students is 70. -/
theorem circular_arrangement_students (n : ℕ) : n = 70 :=
  by
  -- Assume the 10th and 45th positions are opposite each other
  have h1 : 45 - 10 = n / 2 := by sorry
  
  -- The total number of students is twice the difference between opposite positions
  have h2 : n = 2 * (45 - 10) := by sorry
  
  -- Prove that n equals 70
  sorry

end NUMINAMATH_CALUDE_circular_arrangement_students_l2167_216718


namespace NUMINAMATH_CALUDE_triangle_side_length_l2167_216736

theorem triangle_side_length (a b c : Real) (angle_A angle_B : Real) :
  angle_A = 30 * Real.pi / 180 →
  angle_B = 45 * Real.pi / 180 →
  c = 8 →
  b = 4 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2167_216736


namespace NUMINAMATH_CALUDE_inner_segments_sum_l2167_216731

theorem inner_segments_sum (perimeter_quadrilaterals perimeter_triangles perimeter_ABC : ℝ) 
  (h1 : perimeter_quadrilaterals = 25)
  (h2 : perimeter_triangles = 20)
  (h3 : perimeter_ABC = 19) :
  let total_perimeter := perimeter_quadrilaterals + perimeter_triangles
  let inner_segments := total_perimeter - perimeter_ABC
  inner_segments / 2 = 13 := by sorry

end NUMINAMATH_CALUDE_inner_segments_sum_l2167_216731


namespace NUMINAMATH_CALUDE_lines_are_parallel_l2167_216756

/-- Two lines in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Definition of parallel lines -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

theorem lines_are_parallel : 
  let line1 : Line := ⟨3, 1, 1⟩
  let line2 : Line := ⟨6, 2, 1⟩
  parallel line1 line2 := by
  sorry

end NUMINAMATH_CALUDE_lines_are_parallel_l2167_216756


namespace NUMINAMATH_CALUDE_tangent_line_slope_l2167_216700

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the tangent line
def tangent_line (a : ℝ) (x : ℝ) : ℝ := a*x + 16

theorem tangent_line_slope (a : ℝ) :
  (∃ x₀ : ℝ, f x₀ = tangent_line a x₀ ∧
    ∀ x : ℝ, x ≠ x₀ → f x ≠ tangent_line a x) →
  a = 9 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_slope_l2167_216700


namespace NUMINAMATH_CALUDE_cake_and_icing_sum_l2167_216746

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents the cake piece -/
structure CakePiece where
  top : List Point3D
  height : ℝ

/-- Calculates the volume of the cake piece -/
def cakeVolume (piece : CakePiece) : ℝ :=
  sorry

/-- Calculates the area of icing on the cake piece -/
def icingArea (piece : CakePiece) : ℝ :=
  sorry

/-- The main theorem -/
theorem cake_and_icing_sum (R P N : Point3D) (piece : CakePiece) :
  R.x = 0 ∧ R.y = 0 ∧ R.z = 3 ∧
  P.x = 3 ∧ P.y = 0 ∧ P.z = 3 ∧
  N.x = 2 ∧ N.y = 0 ∧ N.z = 3 ∧
  piece.top = [R, N, P] ∧
  piece.height = 3 →
  cakeVolume piece + icingArea piece = 13 := by
  sorry

end NUMINAMATH_CALUDE_cake_and_icing_sum_l2167_216746


namespace NUMINAMATH_CALUDE_least_n_satisfying_inequality_l2167_216710

theorem least_n_satisfying_inequality :
  ∀ n : ℕ+, n < 4 → (1 : ℚ) / n - (1 : ℚ) / (n + 1) ≥ (1 : ℚ) / 15 ∧
  (1 : ℚ) / 4 - (1 : ℚ) / 5 < (1 : ℚ) / 15 :=
by sorry

end NUMINAMATH_CALUDE_least_n_satisfying_inequality_l2167_216710


namespace NUMINAMATH_CALUDE_concyclic_projections_l2167_216729

-- Define a circle and a point on a plane
variable (Circle : Type) (Point : Type)

-- Define a function to check if points are concyclic
variable (are_concyclic : Circle → List Point → Prop)

-- Define a function for orthogonal projection
variable (orthogonal_projection : Point → Point → Point → Point)

-- Theorem statement
theorem concyclic_projections
  (A B C D A' B' C' D' : Point) (circle : Circle) :
  are_concyclic circle [A, B, C, D] →
  A' = orthogonal_projection A B D →
  C' = orthogonal_projection C B D →
  B' = orthogonal_projection B A C →
  D' = orthogonal_projection D A C →
  ∃ (circle' : Circle), are_concyclic circle' [A', B', C', D'] :=
by sorry

end NUMINAMATH_CALUDE_concyclic_projections_l2167_216729


namespace NUMINAMATH_CALUDE_twelve_factorial_base_nine_zeroes_l2167_216795

/-- The number of trailing zeroes in n! when written in base b -/
def trailingZeroes (n : ℕ) (b : ℕ) : ℕ := sorry

/-- 12! ends with 2 zeroes when written in base 9 -/
theorem twelve_factorial_base_nine_zeroes : trailingZeroes 12 9 = 2 := by sorry

end NUMINAMATH_CALUDE_twelve_factorial_base_nine_zeroes_l2167_216795


namespace NUMINAMATH_CALUDE_product_of_repeating_decimal_and_nine_l2167_216792

theorem product_of_repeating_decimal_and_nine (q : ℚ) : 
  (∃ (n : ℕ), q * (100 : ℚ) - q = (45 : ℚ) + n * (100 : ℚ)) → q * 9 = 45 / 11 := by
  sorry

end NUMINAMATH_CALUDE_product_of_repeating_decimal_and_nine_l2167_216792


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2167_216734

theorem sqrt_equation_solution :
  ∃! z : ℝ, Real.sqrt (7 + 3 * z) = 15 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2167_216734


namespace NUMINAMATH_CALUDE_motorboat_speed_calculation_l2167_216709

/-- The flood flow speed in kilometers per hour -/
def flood_speed : ℝ := 10

/-- The downstream distance in kilometers -/
def downstream_distance : ℝ := 2

/-- The upstream distance in kilometers -/
def upstream_distance : ℝ := 1.2

/-- The maximum speed of the motorboat in still water in kilometers per hour -/
def motorboat_speed : ℝ := 40

theorem motorboat_speed_calculation :
  (downstream_distance / (motorboat_speed + flood_speed) = 
   upstream_distance / (motorboat_speed - flood_speed)) ∧
  motorboat_speed = 40 := by sorry

end NUMINAMATH_CALUDE_motorboat_speed_calculation_l2167_216709


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2167_216715

/-- A hyperbola with given asymptotes and passing through a specific point -/
structure Hyperbola where
  /-- The slope of the asymptotes -/
  asymptote_slope : ℝ
  /-- The x-coordinate of the point the hyperbola passes through -/
  point_x : ℝ
  /-- The y-coordinate of the point the hyperbola passes through -/
  point_y : ℝ

/-- The equation of the hyperbola satisfies the given conditions -/
theorem hyperbola_equation (h : Hyperbola) 
  (h_asymptote : h.asymptote_slope = 1/2)
  (h_point : h.point_x = 4 ∧ h.point_y = Real.sqrt 2) :
  ∃ (f : ℝ → ℝ → Prop), 
    (∀ x y, f x y ↔ x^2/8 - y^2/2 = 1) ∧ 
    (f h.point_x h.point_y) ∧
    (∀ x, f x (h.asymptote_slope * x) ∨ f x (-h.asymptote_slope * x)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2167_216715


namespace NUMINAMATH_CALUDE_hugh_initial_candy_l2167_216724

/-- The amount of candy Hugh had initially -/
def hugh_candy : ℕ := sorry

/-- The amount of candy Tommy had initially -/
def tommy_candy : ℕ := 6

/-- The amount of candy Melany had initially -/
def melany_candy : ℕ := 7

/-- The amount of candy each person had after sharing equally -/
def shared_candy : ℕ := 7

/-- The number of people sharing the candy -/
def num_people : ℕ := 3

theorem hugh_initial_candy :
  hugh_candy = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_hugh_initial_candy_l2167_216724


namespace NUMINAMATH_CALUDE_number_of_terms_S_9891_1989_l2167_216783

/-- Elementary symmetric expression -/
def S (k : ℕ) (n : ℕ) : ℕ := Nat.choose k n

/-- The number of terms in S_{9891}(1989) -/
theorem number_of_terms_S_9891_1989 : S 9891 1989 = Nat.choose 9891 1989 := by
  sorry

end NUMINAMATH_CALUDE_number_of_terms_S_9891_1989_l2167_216783


namespace NUMINAMATH_CALUDE_orthocenter_tangents_collinear_l2167_216714

/-- Representation of a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Representation of a triangle -/
structure Triangle :=
  (A : Point) (B : Point) (C : Point)

/-- Definition of an acute-angled triangle -/
def isAcuteAngled (t : Triangle) : Prop := sorry

/-- Definition of the orthocenter of a triangle -/
def orthocenter (t : Triangle) : Point := sorry

/-- Definition of a circle -/
structure Circle :=
  (center : Point) (radius : ℝ)

/-- Definition of a tangent line to a circle -/
def isTangent (p : Point) (c : Circle) : Prop := sorry

/-- Definition of collinearity -/
def areCollinear (p1 p2 p3 : Point) : Prop := sorry

/-- Main theorem -/
theorem orthocenter_tangents_collinear 
  (t : Triangle) 
  (h_acute : isAcuteAngled t) 
  (H : Point) 
  (h_ortho : H = orthocenter t) 
  (c : Circle) 
  (h_circle : c.center = Point.mk ((t.B.x + t.C.x) / 2) ((t.B.y + t.C.y) / 2) ∧ 
              c.radius = (((t.B.x - t.C.x)^2 + (t.B.y - t.C.y)^2)^(1/2)) / 2) 
  (P Q : Point) 
  (h_tangent_P : isTangent P c ∧ (∃ k : ℝ, P = Point.mk (t.A.x + k * (P.x - t.A.x)) (t.A.y + k * (P.y - t.A.y))))
  (h_tangent_Q : isTangent Q c ∧ (∃ k : ℝ, Q = Point.mk (t.A.x + k * (Q.x - t.A.x)) (t.A.y + k * (Q.y - t.A.y))))
  : areCollinear P H Q := 
sorry

end NUMINAMATH_CALUDE_orthocenter_tangents_collinear_l2167_216714


namespace NUMINAMATH_CALUDE_minimum_peanuts_min_peanuts_is_25_l2167_216737

theorem minimum_peanuts : ℕ → Prop :=
  fun n => (n % 3 = 1) ∧ 
           ((n - 1) / 3 - 1) % 3 = 0 ∧ 
           (((n - 1) / 3 - 1 - 1) / 3 - 1) % 3 = 0

theorem min_peanuts_is_25 : minimum_peanuts 25 ∧ ∀ m < 25, ¬minimum_peanuts m := by
  sorry

end NUMINAMATH_CALUDE_minimum_peanuts_min_peanuts_is_25_l2167_216737


namespace NUMINAMATH_CALUDE_total_population_l2167_216733

def population_problem (springfield_population greenville_population : ℕ) : Prop :=
  springfield_population = 482653 ∧
  greenville_population = springfield_population - 119666 ∧
  springfield_population + greenville_population = 845640

theorem total_population :
  ∃ (springfield_population greenville_population : ℕ),
    population_problem springfield_population greenville_population :=
by
  sorry

end NUMINAMATH_CALUDE_total_population_l2167_216733


namespace NUMINAMATH_CALUDE_chastity_lollipop_cost_l2167_216727

def lollipop_cost (initial_money : ℚ) (remaining_money : ℚ) (num_lollipops : ℕ) (num_gummy_packs : ℕ) (gummy_pack_cost : ℚ) : ℚ :=
  ((initial_money - remaining_money) - (num_gummy_packs * gummy_pack_cost)) / num_lollipops

theorem chastity_lollipop_cost :
  lollipop_cost 15 5 4 2 2 = (3/2) :=
sorry

end NUMINAMATH_CALUDE_chastity_lollipop_cost_l2167_216727


namespace NUMINAMATH_CALUDE_alternating_ball_probability_l2167_216778

-- Define the number of balls of each color
def white_balls : ℕ := 5
def black_balls : ℕ := 5
def red_balls : ℕ := 2

-- Define the total number of balls
def total_balls : ℕ := white_balls + black_balls + red_balls

-- Define the function to calculate binomial coefficient
def binomial (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

-- Define the function to calculate the number of successful arrangements
def successful_arrangements : ℕ := 
  (binomial (total_balls) red_balls) * (binomial (white_balls + black_balls) white_balls)

-- Define the function to calculate the total number of arrangements
def total_arrangements : ℕ := 
  Nat.factorial total_balls / (Nat.factorial white_balls * Nat.factorial black_balls * Nat.factorial red_balls)

-- State the theorem
theorem alternating_ball_probability : 
  (successful_arrangements : ℚ) / total_arrangements = 123 / 205 := by sorry

end NUMINAMATH_CALUDE_alternating_ball_probability_l2167_216778


namespace NUMINAMATH_CALUDE_symmetry_of_product_l2167_216748

def IsOddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x ∈ Set.Icc (-2 : ℝ) 2, f (-x) = -f x

def IsEvenFunction (g : ℝ → ℝ) : Prop :=
  ∀ x ∈ Set.Icc (-2 : ℝ) 2, g (-x) = g x

theorem symmetry_of_product (f g : ℝ → ℝ) 
    (hf : IsOddFunction f) (hg : IsEvenFunction g) : 
    IsOddFunction (fun x ↦ f x * g x) := by
  sorry

end NUMINAMATH_CALUDE_symmetry_of_product_l2167_216748


namespace NUMINAMATH_CALUDE_distance_theorem_l2167_216749

/-- The distance between Maxwell's and Brad's homes --/
def distance_between_homes (maxwell_speed : ℝ) (brad_speed : ℝ) (brad_delay : ℝ) (total_time : ℝ) : ℝ :=
  maxwell_speed * total_time + brad_speed * (total_time - brad_delay)

/-- Theorem stating the distance between Maxwell's and Brad's homes --/
theorem distance_theorem (maxwell_speed : ℝ) (brad_speed : ℝ) (brad_delay : ℝ) (total_time : ℝ)
  (h1 : maxwell_speed = 4)
  (h2 : brad_speed = 6)
  (h3 : brad_delay = 1)
  (h4 : total_time = 2) :
  distance_between_homes maxwell_speed brad_speed brad_delay total_time = 14 := by
  sorry

#check distance_theorem

end NUMINAMATH_CALUDE_distance_theorem_l2167_216749


namespace NUMINAMATH_CALUDE_cos_angle_through_point_l2167_216739

/-- Given an angle α whose initial side is the positive x-axis and whose terminal side
    passes through the point (4, -3), prove that cos(α) = 4/5 -/
theorem cos_angle_through_point :
  ∀ α : ℝ,
  (∃ (x y : ℝ), x = 4 ∧ y = -3 ∧ 
    (Real.cos α * x - Real.sin α * y = x) ∧
    (Real.sin α * x + Real.cos α * y = y)) →
  Real.cos α = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_cos_angle_through_point_l2167_216739


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l2167_216740

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}
def B : Set ℝ := {x : ℝ | 1 < x ∧ x < 3}

-- State the theorem
theorem union_of_A_and_B : 
  A ∪ B = {x : ℝ | -1 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l2167_216740


namespace NUMINAMATH_CALUDE_limit_implies_range_l2167_216708

theorem limit_implies_range (a : ℝ) : 
  (∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |3^n / (3^(n+1) + (a+1)^n) - 1/3| < ε) → 
  a ∈ Set.Ioo (-4 : ℝ) 2 := by
sorry

end NUMINAMATH_CALUDE_limit_implies_range_l2167_216708


namespace NUMINAMATH_CALUDE_magazine_cost_l2167_216797

theorem magazine_cost (total_books : ℕ) (book_cost : ℕ) (total_magazines : ℕ) (total_spent : ℕ) :
  total_books = 10 →
  book_cost = 15 →
  total_magazines = 10 →
  total_spent = 170 →
  ∃ (magazine_cost : ℕ), magazine_cost = 2 ∧ total_spent = total_books * book_cost + total_magazines * magazine_cost :=
by
  sorry

end NUMINAMATH_CALUDE_magazine_cost_l2167_216797


namespace NUMINAMATH_CALUDE_chess_and_go_pricing_and_max_purchase_l2167_216785

/-- The unit price of a Chinese chess set -/
def chinese_chess_price : ℝ := 25

/-- The unit price of a Go set -/
def go_price : ℝ := 30

/-- The total number of sets to be purchased -/
def total_sets : ℕ := 120

/-- The maximum total cost -/
def max_total_cost : ℝ := 3500

theorem chess_and_go_pricing_and_max_purchase :
  (2 * chinese_chess_price + go_price = 80) ∧
  (4 * chinese_chess_price + 3 * go_price = 190) ∧
  (∀ m : ℕ, m ≤ total_sets → 
    chinese_chess_price * (total_sets - m) + go_price * m ≤ max_total_cost →
    m ≤ 100) ∧
  (∃ m : ℕ, m = 100 ∧ 
    chinese_chess_price * (total_sets - m) + go_price * m ≤ max_total_cost) :=
by sorry

end NUMINAMATH_CALUDE_chess_and_go_pricing_and_max_purchase_l2167_216785


namespace NUMINAMATH_CALUDE_complement_determines_set_l2167_216772

def U : Set ℕ := {0, 1, 2, 3}

theorem complement_determines_set (M : Set ℕ) (h : Set.compl M = {2}) : M = {0, 1, 3} := by
  sorry

end NUMINAMATH_CALUDE_complement_determines_set_l2167_216772


namespace NUMINAMATH_CALUDE_women_average_age_l2167_216723

theorem women_average_age 
  (n : ℕ) 
  (A : ℝ) 
  (age1 age2 : ℕ) 
  (h1 : n = 8) 
  (h2 : age1 = 20) 
  (h3 : age2 = 22) 
  (h4 : (n * A - (age1 + age2 : ℝ) + (W1 + W2)) / n = A + 2) :
  (W1 + W2) / 2 = 29 :=
by sorry

end NUMINAMATH_CALUDE_women_average_age_l2167_216723


namespace NUMINAMATH_CALUDE_product_from_lcm_hcf_l2167_216777

theorem product_from_lcm_hcf (a b : ℕ+) 
  (h_lcm : Nat.lcm a b = 750) 
  (h_hcf : Nat.gcd a b = 25) : 
  a * b = 18750 := by
  sorry

end NUMINAMATH_CALUDE_product_from_lcm_hcf_l2167_216777


namespace NUMINAMATH_CALUDE_unique_square_divisible_by_three_in_range_l2167_216703

theorem unique_square_divisible_by_three_in_range : 
  ∃! x : ℕ, 
    (∃ n : ℕ, x = n * n) ∧ 
    (∃ k : ℕ, x = 3 * k) ∧ 
    90 < x ∧ x < 150 := by
  sorry

end NUMINAMATH_CALUDE_unique_square_divisible_by_three_in_range_l2167_216703


namespace NUMINAMATH_CALUDE_science_fair_students_l2167_216752

theorem science_fair_students (know_it_all : ℕ) (karen : ℕ) (novel_corona : ℕ) (total : ℕ) :
  know_it_all = 50 →
  karen = 3 * know_it_all / 5 →
  total = 240 →
  total = know_it_all + karen + novel_corona →
  novel_corona = 160 := by
sorry

end NUMINAMATH_CALUDE_science_fair_students_l2167_216752


namespace NUMINAMATH_CALUDE_nine_times_eleven_and_two_fifths_l2167_216768

theorem nine_times_eleven_and_two_fifths (x : ℝ) : 
  9 * (11 + 2/5) = 102 + 3/5 := by
  sorry

end NUMINAMATH_CALUDE_nine_times_eleven_and_two_fifths_l2167_216768


namespace NUMINAMATH_CALUDE_sequence_problem_l2167_216735

theorem sequence_problem (a : Fin 100 → ℚ) :
  (∀ i : Fin 98, a (Fin.succ i) = a i * a (Fin.succ (Fin.succ i))) →
  (∀ i : Fin 100, a i ≠ 0) →
  a 0 = 2018 →
  a 99 = 1 / 2018 := by
sorry

end NUMINAMATH_CALUDE_sequence_problem_l2167_216735


namespace NUMINAMATH_CALUDE_black_men_tshirt_cost_l2167_216738

/-- Represents the cost of t-shirts and number of employees --/
structure TShirtData where
  white_men_cost : ℝ
  black_men_cost : ℝ
  total_employees : ℕ
  total_spent : ℝ

/-- Theorem stating the cost of black men's t-shirts --/
theorem black_men_tshirt_cost (data : TShirtData) 
  (h1 : data.white_men_cost = 20)
  (h2 : data.total_employees = 40)
  (h3 : data.total_spent = 660)
  (h4 : ∃ (n : ℕ), n * 4 = data.total_employees) :
  data.black_men_cost = 18 := by
  sorry

end NUMINAMATH_CALUDE_black_men_tshirt_cost_l2167_216738


namespace NUMINAMATH_CALUDE_triangle_area_approx_036_l2167_216753

-- Define the slopes and intersection point
def slope1 : ℚ := 3/4
def slope2 : ℚ := 1/3
def intersection : ℚ × ℚ := (3, 3)

-- Define the lines
def line1 (x : ℚ) : ℚ := slope1 * (x - intersection.1) + intersection.2
def line2 (x : ℚ) : ℚ := slope2 * (x - intersection.1) + intersection.2
def line3 (x y : ℚ) : Prop := x + y = 8

-- Define the function to calculate the area of a triangle given three points
def triangleArea (p1 p2 p3 : ℚ × ℚ) : ℚ :=
  (1/2) * abs (p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2))

-- Theorem statement
theorem triangle_area_approx_036 :
  ∃ (p1 p2 p3 : ℚ × ℚ),
    p1 = intersection ∧
    line1 p2.1 = p2.2 ∧
    line2 p3.1 = p3.2 ∧
    line3 p2.1 p2.2 ∧
    line3 p3.1 p3.2 ∧
    abs (triangleArea p1 p2 p3 - 0.36) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_approx_036_l2167_216753


namespace NUMINAMATH_CALUDE_angle_measure_in_acute_triangle_l2167_216717

theorem angle_measure_in_acute_triangle (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ A < π/2 ∧
  0 < B ∧ B < π/2 ∧
  0 < C ∧ C < π/2 ∧
  A + B + C = π ∧
  b = 2 * a * Real.sin B →
  A = π/6 := by
sorry

end NUMINAMATH_CALUDE_angle_measure_in_acute_triangle_l2167_216717


namespace NUMINAMATH_CALUDE_martha_problems_l2167_216787

theorem martha_problems (total : ℕ) (angela_unique : ℕ) : total = 20 → angela_unique = 9 → ∃ martha : ℕ,
  martha + (4 * martha - 2) + ((4 * martha - 2) / 2) + angela_unique = total ∧ martha = 2 := by
  sorry

end NUMINAMATH_CALUDE_martha_problems_l2167_216787


namespace NUMINAMATH_CALUDE_system_solution_l2167_216744

theorem system_solution (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  ∃ (x y z : ℝ),
    (x + a*y + a^2*z + a^3 = 0) ∧
    (x + b*y + b^2*z + b^3 = 0) ∧
    (x + c*y + c^2*z + c^3 = 0) ∧
    (x = -a*b*c) ∧
    (y = a*b + b*c + c*a) ∧
    (z = -(a + b + c)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l2167_216744


namespace NUMINAMATH_CALUDE_rectangle_perimeter_from_squares_l2167_216788

theorem rectangle_perimeter_from_squares (side_length : ℝ) : 
  side_length = 3 → 
  ∃ (perimeter₁ perimeter₂ : ℝ), 
    (perimeter₁ = 24 ∧ perimeter₂ = 30) ∧ 
    (∀ (p : ℝ), p ≠ perimeter₁ ∧ p ≠ perimeter₂ → 
      ¬∃ (length width : ℝ), 
        (length * width = 4 * side_length^2) ∧ 
        (2 * (length + width) = p)) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_from_squares_l2167_216788


namespace NUMINAMATH_CALUDE_range_of_a_l2167_216765

theorem range_of_a (x : ℝ) (h1 : x > 0) (h2 : 2^x * (x - a) < 1) : a > -1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2167_216765


namespace NUMINAMATH_CALUDE_truck_travel_distance_l2167_216771

/-- Given a truck that travels 300 kilometers on 5 liters of diesel,
    prove that it can travel 420 kilometers on 7 liters of diesel. -/
theorem truck_travel_distance (initial_distance : ℝ) (initial_fuel : ℝ) (new_fuel : ℝ) :
  initial_distance = 300 ∧ initial_fuel = 5 ∧ new_fuel = 7 →
  (initial_distance / initial_fuel) * new_fuel = 420 := by
  sorry

end NUMINAMATH_CALUDE_truck_travel_distance_l2167_216771


namespace NUMINAMATH_CALUDE_part_one_part_two_l2167_216786

-- Define propositions p and q
def p (x a : ℝ) : Prop := (x - a) * (x - 3 * a) < 0

def q (x : ℝ) : Prop := 2 ≤ x ∧ x ≤ 3

-- Part 1
theorem part_one (x : ℝ) :
  (p x 1) ∧ (q x) → 2 ≤ x ∧ x < 3 := by sorry

-- Part 2
theorem part_two :
  (∀ x a : ℝ, (¬(p x a) → ¬(q x)) ∧ ¬(q x → ¬(p x a))) →
  ∃ a : ℝ, 1 < a ∧ a < 2 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2167_216786


namespace NUMINAMATH_CALUDE_find_number_l2167_216757

theorem find_number : ∃ x : ℝ, 11 * x + 1 = 45 ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l2167_216757


namespace NUMINAMATH_CALUDE_lindas_savings_l2167_216758

theorem lindas_savings (savings : ℚ) : 
  (7/13 : ℚ) * savings + (3/13 : ℚ) * savings + 180 = savings ∧ 
  (3/13 : ℚ) * savings = 2 * 180 → 
  savings = 1560 := by sorry

end NUMINAMATH_CALUDE_lindas_savings_l2167_216758


namespace NUMINAMATH_CALUDE_train_speed_l2167_216781

/-- The speed of a train given its length and time to cross a fixed point -/
theorem train_speed (length time : ℝ) (h1 : length = 200) (h2 : time = 20) :
  length / time = 10 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l2167_216781


namespace NUMINAMATH_CALUDE_fraction_of_books_sold_l2167_216762

theorem fraction_of_books_sold (total_revenue : ℕ) (remaining_books : ℕ) (price_per_book : ℕ) :
  total_revenue = 288 →
  remaining_books = 36 →
  price_per_book = 4 →
  (total_revenue / price_per_book : ℚ) / ((total_revenue / price_per_book) + remaining_books) = 2/3 :=
by sorry

end NUMINAMATH_CALUDE_fraction_of_books_sold_l2167_216762


namespace NUMINAMATH_CALUDE_max_value_complex_expression_l2167_216754

theorem max_value_complex_expression (z : ℂ) (h : Complex.abs z = 1) :
  ∃ (max_val : ℝ), max_val = 3 * Real.sqrt 3 ∧
  ∀ (w : ℂ), Complex.abs w = 1 → Complex.abs (w^3 - 3*w - 2) ≤ max_val :=
sorry

end NUMINAMATH_CALUDE_max_value_complex_expression_l2167_216754


namespace NUMINAMATH_CALUDE_cos_squared_pi_twelfth_plus_one_l2167_216701

theorem cos_squared_pi_twelfth_plus_one :
  2 * (Real.cos (π / 12))^2 + 1 = 2 + Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_squared_pi_twelfth_plus_one_l2167_216701


namespace NUMINAMATH_CALUDE_john_half_decks_l2167_216763

/-- The number of cards in a full deck -/
def full_deck : ℕ := 52

/-- The number of full decks John has -/
def num_full_decks : ℕ := 3

/-- The number of cards John threw away -/
def discarded_cards : ℕ := 34

/-- The number of cards John has after discarding -/
def remaining_cards : ℕ := 200

/-- Calculates the number of half-full decks John found -/
def num_half_decks : ℕ :=
  (remaining_cards + discarded_cards - num_full_decks * full_deck) / (full_deck / 2)

theorem john_half_decks :
  num_half_decks = 3 := by sorry

end NUMINAMATH_CALUDE_john_half_decks_l2167_216763


namespace NUMINAMATH_CALUDE_printing_presses_l2167_216761

theorem printing_presses (papers : ℕ) (initial_time hours : ℝ) (known_presses : ℕ) :
  papers > 0 →
  initial_time > 0 →
  hours > 0 →
  known_presses > 0 →
  (papers : ℝ) / (initial_time * (papers / (hours * known_presses : ℝ))) = 40 :=
by
  sorry

#check printing_presses 500000 9 12 30

end NUMINAMATH_CALUDE_printing_presses_l2167_216761


namespace NUMINAMATH_CALUDE_polynomial_coefficients_l2167_216728

-- Define the polynomial
def p (x a₄ a₃ a₂ a₁ a₀ : ℝ) : ℝ := (x + 2)^5 - (x + 1)^5 - (a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀)

-- State the theorem
theorem polynomial_coefficients :
  ∃ (a₄ a₃ a₂ : ℝ), ∀ x, p x a₄ a₃ a₂ 75 31 = 0 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_coefficients_l2167_216728


namespace NUMINAMATH_CALUDE_nelly_outbid_multiple_l2167_216716

def joes_bid : ℕ := 160000
def nellys_bid : ℕ := 482000
def additional_amount : ℕ := 2000

theorem nelly_outbid_multiple : 
  (nellys_bid - joes_bid - additional_amount) / joes_bid = 2 := by
  sorry

end NUMINAMATH_CALUDE_nelly_outbid_multiple_l2167_216716


namespace NUMINAMATH_CALUDE_consecutive_non_prime_non_prime_power_l2167_216713

/-- For any positive integer n, there exists a positive integer k such that 
    for all i in {1, ..., n}, k + i is neither prime nor a prime power. -/
theorem consecutive_non_prime_non_prime_power (n : ℕ) (hn : 0 < n) :
  ∃ k : ℕ, ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → ¬(Nat.Prime (k + i)) ∧ ¬(∃ p m : ℕ, Nat.Prime p ∧ 1 < m ∧ k + i = p^m) :=
sorry

end NUMINAMATH_CALUDE_consecutive_non_prime_non_prime_power_l2167_216713


namespace NUMINAMATH_CALUDE_mrs_hilt_walking_distance_l2167_216774

/-- The total distance walked to and from a water fountain -/
def total_distance (distance_to_fountain : ℕ) (num_trips : ℕ) : ℕ :=
  2 * distance_to_fountain * num_trips

/-- Theorem: Mrs. Hilt walks 240 feet given the problem conditions -/
theorem mrs_hilt_walking_distance :
  total_distance 30 4 = 240 := by
  sorry

end NUMINAMATH_CALUDE_mrs_hilt_walking_distance_l2167_216774


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_l2167_216711

theorem sum_of_a_and_b (a b : ℝ) : 
  |a - 1/2| + |b + 5| = 0 → a + b = -9/2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_l2167_216711


namespace NUMINAMATH_CALUDE_tan_sum_half_angles_l2167_216726

theorem tan_sum_half_angles (p q : ℝ) 
  (h1 : Real.cos p + Real.cos q = 1/3)
  (h2 : Real.sin p + Real.sin q = 5/13) : 
  Real.tan ((p + q)/2) = 15/13 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_half_angles_l2167_216726


namespace NUMINAMATH_CALUDE_sector_area_l2167_216719

/-- The area of a sector with radius 10 cm and central angle 120° is (100π/3) cm² -/
theorem sector_area (r : ℝ) (θ : ℝ) : 
  r = 10 → θ = 2 * π / 3 → (1/2) * r^2 * θ = (100 * π) / 3 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l2167_216719


namespace NUMINAMATH_CALUDE_expression_simplification_l2167_216769

theorem expression_simplification (x y : ℝ) :
  (1/2) * x - 2 * (x - (1/3) * y^2) + (-3/2 * x + (1/3) * y^2) = -3 * x + y^2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2167_216769


namespace NUMINAMATH_CALUDE_naClOConcentrationDecreases_l2167_216790

-- Define the disinfectant solution
structure DisinfectantSolution :=
  (volume : ℝ)
  (naClOConcentration : ℝ)
  (density : ℝ)

-- Define the properties of the initial solution
def initialSolution : DisinfectantSolution :=
  { volume := 480,
    naClOConcentration := 0.25,
    density := 1.19 }

-- Define the property that NaClO absorbs H₂O and CO₂ from air and degrades
axiom naClODegrades : ∀ (t : ℝ), t > 0 → ∃ (δ : ℝ), δ > 0 ∧ δ < initialSolution.naClOConcentration

-- Theorem stating that NaClO concentration decreases over time
theorem naClOConcentrationDecreases :
  ∀ (t : ℝ), t > 0 →
  ∃ (s : DisinfectantSolution),
    s.volume = initialSolution.volume ∧
    s.density = initialSolution.density ∧
    s.naClOConcentration < initialSolution.naClOConcentration :=
sorry

end NUMINAMATH_CALUDE_naClOConcentrationDecreases_l2167_216790


namespace NUMINAMATH_CALUDE_zeros_in_quotient_l2167_216784

/-- S_k represents the k-length sequence of twos in its decimal presentation -/
def S (k : ℕ) : ℕ := (2 * (10^k - 1)) / 9

/-- The quotient of S_30 divided by S_5 -/
def Q : ℕ := S 30 / S 5

/-- The number of zeros in the decimal representation of Q -/
def num_zeros (n : ℕ) : ℕ := sorry

theorem zeros_in_quotient : num_zeros Q = 20 := by sorry

end NUMINAMATH_CALUDE_zeros_in_quotient_l2167_216784


namespace NUMINAMATH_CALUDE_f_range_l2167_216764

-- Define the closest multiple function
def closestMultiple (k : ℤ) (n : ℕ) : ℤ :=
  let m := (2 * n + 1 : ℤ)
  m * ((k + m / 2) / m)

-- Define the function f
def f (k : ℤ) : ℤ :=
  closestMultiple k 1 + closestMultiple (2 * k) 2 + closestMultiple (3 * k) 3 - 6 * k

-- State the theorem
theorem f_range :
  ∀ k : ℤ, -6 ≤ f k ∧ f k ≤ 6 :=
sorry

end NUMINAMATH_CALUDE_f_range_l2167_216764


namespace NUMINAMATH_CALUDE_product_of_non_shared_sides_squared_l2167_216776

/-- Represents a right triangle with given area and side lengths -/
structure RightTriangle where
  area : ℝ
  side1 : ℝ
  side2 : ℝ
  hypotenuse : ℝ
  area_eq : area = (side1 * side2) / 2
  pythagoras : side1^2 + side2^2 = hypotenuse^2

/-- Theorem about the product of non-shared sides of two specific right triangles -/
theorem product_of_non_shared_sides_squared
  (T₁ T₂ : RightTriangle)
  (h₁ : T₁.area = 3)
  (h₂ : T₂.area = 4)
  (h₃ : T₁.side1 = T₂.side1)  -- Shared side
  (h₄ : T₁.side2 = T₂.side2)  -- Shared side
  (h₅ : T₁.side1 = T₁.side2)  -- 45°-45°-90° triangle condition
  : (T₁.hypotenuse * T₂.hypotenuse)^2 = 64 := by
  sorry

end NUMINAMATH_CALUDE_product_of_non_shared_sides_squared_l2167_216776


namespace NUMINAMATH_CALUDE_product_sum_theorem_l2167_216779

theorem product_sum_theorem (a b c : ℕ+) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a * b * c = 5^4 →
  (a : ℕ) + (b : ℕ) + (c : ℕ) = 131 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_theorem_l2167_216779


namespace NUMINAMATH_CALUDE_job_selection_probability_l2167_216712

theorem job_selection_probability (jamie_prob tom_prob : ℚ) 
  (h1 : jamie_prob = 2 / 3)
  (h2 : tom_prob = 5 / 7) :
  jamie_prob * tom_prob = 10 / 21 := by
sorry

end NUMINAMATH_CALUDE_job_selection_probability_l2167_216712


namespace NUMINAMATH_CALUDE_lcm_hcf_problem_l2167_216799

theorem lcm_hcf_problem (A B : ℕ+) : 
  Nat.lcm A B = 192 →
  Nat.gcd A B = 16 →
  A = 48 →
  B = 64 := by sorry

end NUMINAMATH_CALUDE_lcm_hcf_problem_l2167_216799


namespace NUMINAMATH_CALUDE_f_2_value_l2167_216755

/-- An odd function -/
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- An even function -/
def even_function (g : ℝ → ℝ) : Prop :=
  ∀ x, g (-x) = g x

/-- The main theorem -/
theorem f_2_value (f g : ℝ → ℝ) (a : ℝ) :
  odd_function f →
  even_function g →
  (∀ x, f x + g x = a^x - a^(-x) + 2) →
  a > 0 →
  a ≠ 1 →
  g 2 = a →
  f 2 = 15/4 := by
  sorry


end NUMINAMATH_CALUDE_f_2_value_l2167_216755


namespace NUMINAMATH_CALUDE_circle_ratio_l2167_216706

theorem circle_ratio (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h1 : c^2 - a^2 = 4 * a^2) 
  (h2 : b^2 = (a^2 + c^2) / 2) : 
  a / c = 1 / Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_circle_ratio_l2167_216706


namespace NUMINAMATH_CALUDE_quadratic_polynomial_form_l2167_216702

/-- A quadratic polynomial with specific properties -/
structure QuadraticPolynomial where
  p : ℝ → ℝ
  is_quadratic : ∃ a b c : ℝ, ∀ x, p x = a * x^2 + b * x + c
  value_at_neg_two : p (-2) = 8
  asymptotes : Set ℝ
  asymptotes_def : asymptotes = {-2, 2}
  is_asymptote : ∀ x ∈ asymptotes, ∀ ε > 0, ∃ δ > 0, ∀ y, |y - x| < δ → |1 / (p y)| > 1 / ε

/-- The theorem stating the specific form of the quadratic polynomial -/
theorem quadratic_polynomial_form (f : QuadraticPolynomial) : f.p = λ x => -2 * x^2 + 8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_form_l2167_216702
