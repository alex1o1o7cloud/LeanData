import Mathlib

namespace NUMINAMATH_CALUDE_johns_earnings_l1008_100828

/-- John's earnings over two weeks --/
theorem johns_earnings (hours_week1 hours_week2 : ℕ) (extra_earnings : ℚ) :
  hours_week1 = 20 →
  hours_week2 = 30 →
  extra_earnings = 102.75 →
  let hourly_wage := extra_earnings / (hours_week2 - hours_week1)
  let total_earnings := (hours_week1 + hours_week2) * hourly_wage
  total_earnings = 513.75 := by
  sorry

end NUMINAMATH_CALUDE_johns_earnings_l1008_100828


namespace NUMINAMATH_CALUDE_height_estimate_theorem_l1008_100834

/-- Represents the regression line for estimating height from foot length -/
structure RegressionLine where
  slope : ℝ
  intercept : ℝ

/-- Represents the sample statistics -/
structure SampleStats where
  mean_x : ℝ
  mean_y : ℝ

/-- Calculates the estimated height given a foot length and regression line -/
def estimate_height (x : ℝ) (line : RegressionLine) : ℝ :=
  line.slope * x + line.intercept

/-- Theorem stating that given the sample statistics and slope, 
    the estimated height for a foot length of 24 cm is 166 cm -/
theorem height_estimate_theorem 
  (stats : SampleStats) 
  (given_slope : ℝ) 
  (h_mean_x : stats.mean_x = 22.5) 
  (h_mean_y : stats.mean_y = 160) 
  (h_slope : given_slope = 4) :
  let line := RegressionLine.mk given_slope (stats.mean_y - given_slope * stats.mean_x)
  estimate_height 24 line = 166 := by
  sorry

#check height_estimate_theorem

end NUMINAMATH_CALUDE_height_estimate_theorem_l1008_100834


namespace NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l1008_100833

theorem complex_number_in_second_quadrant :
  let z : ℂ := -1 + Complex.I
  (z.re < 0) ∧ (z.im > 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l1008_100833


namespace NUMINAMATH_CALUDE_solution_distribution_l1008_100850

def test_tube_volumes : List ℝ := [7, 4, 5, 4, 6, 8, 7, 3, 9, 6]
def num_beakers : ℕ := 5

theorem solution_distribution (volumes : List ℝ) (num_beakers : ℕ) 
  (h1 : volumes = test_tube_volumes) 
  (h2 : num_beakers = 5) : 
  (volumes.sum / num_beakers : ℝ) = 11.8 := by
  sorry

#check solution_distribution

end NUMINAMATH_CALUDE_solution_distribution_l1008_100850


namespace NUMINAMATH_CALUDE_grade_assignment_count_l1008_100823

/-- The number of possible grades a professor can assign to each student. -/
def num_grades : ℕ := 4

/-- The number of students in the class. -/
def num_students : ℕ := 12

/-- The number of ways to assign grades to all students. -/
def num_ways : ℕ := num_grades ^ num_students

/-- Theorem stating that the number of ways to assign grades is 16,777,216. -/
theorem grade_assignment_count : num_ways = 16777216 := by
  sorry

end NUMINAMATH_CALUDE_grade_assignment_count_l1008_100823


namespace NUMINAMATH_CALUDE_games_required_equals_participants_minus_one_l1008_100862

/-- Represents a single-elimination tournament -/
structure SingleEliminationTournament where
  participants : ℕ
  games_required : ℕ

/-- The number of games required in a single-elimination tournament is one less than the number of participants -/
theorem games_required_equals_participants_minus_one 
  (tournament : SingleEliminationTournament) 
  (h : tournament.participants = 512) : 
  tournament.games_required = 511 := by
  sorry

end NUMINAMATH_CALUDE_games_required_equals_participants_minus_one_l1008_100862


namespace NUMINAMATH_CALUDE_cube_isosceles_right_probability_l1008_100808

/-- A cube with 8 vertices -/
structure Cube :=
  (vertices : Fin 8)

/-- A triangle formed by 3 vertices of a cube -/
structure CubeTriangle :=
  (v1 v2 v3 : Fin 8)
  (distinct : v1 ≠ v2 ∧ v1 ≠ v3 ∧ v2 ≠ v3)

/-- An isosceles right triangle on a cube face -/
def IsIsoscelesRight (t : CubeTriangle) : Prop :=
  sorry

/-- The number of isosceles right triangles that can be formed on a cube -/
def numIsoscelesRight : ℕ := 24

/-- The total number of ways to select 3 vertices from 8 -/
def totalTriangles : ℕ := 56

/-- The probability of forming an isosceles right triangle -/
def probabilityIsoscelesRight : ℚ := 3/7

theorem cube_isosceles_right_probability :
  (numIsoscelesRight : ℚ) / totalTriangles = probabilityIsoscelesRight :=
sorry

end NUMINAMATH_CALUDE_cube_isosceles_right_probability_l1008_100808


namespace NUMINAMATH_CALUDE_paintings_per_room_l1008_100818

theorem paintings_per_room (total_paintings : ℕ) (num_rooms : ℕ) 
  (h1 : total_paintings = 32) 
  (h2 : num_rooms = 4) 
  (h3 : total_paintings % num_rooms = 0) : 
  total_paintings / num_rooms = 8 := by
sorry

end NUMINAMATH_CALUDE_paintings_per_room_l1008_100818


namespace NUMINAMATH_CALUDE_uphill_distance_l1008_100859

/-- Proves that given specific conditions, the uphill distance traveled by a car is 100 km. -/
theorem uphill_distance (uphill_speed downhill_speed downhill_distance average_speed : ℝ) 
  (h1 : uphill_speed = 30)
  (h2 : downhill_speed = 60)
  (h3 : downhill_distance = 50)
  (h4 : average_speed = 36) : 
  ∃ uphill_distance : ℝ, 
    uphill_distance = 100 ∧ 
    average_speed = (uphill_distance + downhill_distance) / (uphill_distance / uphill_speed + downhill_distance / downhill_speed) := by
  sorry

end NUMINAMATH_CALUDE_uphill_distance_l1008_100859


namespace NUMINAMATH_CALUDE_surface_area_order_l1008_100827

/-- Represents the types of geometric solids -/
inductive Solid
  | Tetrahedron
  | Cube
  | Octahedron
  | Sphere
  | Cylinder
  | Cone

/-- Computes the surface area of a solid given its volume -/
noncomputable def surfaceArea (s : Solid) (v : ℝ) : ℝ :=
  match s with
  | Solid.Tetrahedron => (216 * Real.sqrt 3) ^ (1/3) * v ^ (2/3)
  | Solid.Cube => 6 * v ^ (2/3)
  | Solid.Octahedron => (108 * Real.sqrt 3) ^ (1/3) * v ^ (2/3)
  | Solid.Sphere => (36 * Real.pi) ^ (1/3) * v ^ (2/3)
  | Solid.Cylinder => (54 * Real.pi) ^ (1/3) * v ^ (2/3)
  | Solid.Cone => (81 * Real.pi) ^ (1/3) * v ^ (2/3)

/-- Theorem stating the order of surface areas for equal volume solids -/
theorem surface_area_order (v : ℝ) (h : v > 0) :
  surfaceArea Solid.Sphere v < surfaceArea Solid.Cylinder v ∧
  surfaceArea Solid.Cylinder v < surfaceArea Solid.Octahedron v ∧
  surfaceArea Solid.Octahedron v < surfaceArea Solid.Cube v ∧
  surfaceArea Solid.Cube v < surfaceArea Solid.Cone v ∧
  surfaceArea Solid.Cone v < surfaceArea Solid.Tetrahedron v :=
by
  sorry

end NUMINAMATH_CALUDE_surface_area_order_l1008_100827


namespace NUMINAMATH_CALUDE_distance_to_axis_of_symmetry_l1008_100802

-- Define the parabola C
def parabola_C (x y : ℝ) : Prop := y^2 = 12 * x

-- Define the line l
def line_l (x y : ℝ) : Prop := y = x - 2

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  parabola_C A.1 A.2 ∧ parabola_C B.1 B.2 ∧
  line_l A.1 A.2 ∧ line_l B.1 B.2

-- Define the axis of symmetry
def axis_of_symmetry : ℝ := -3

-- Theorem statement
theorem distance_to_axis_of_symmetry (A B : ℝ × ℝ) :
  intersection_points A B →
  let midpoint_x := (A.1 + B.1) / 2
  |midpoint_x - axis_of_symmetry| = 11 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_axis_of_symmetry_l1008_100802


namespace NUMINAMATH_CALUDE_probability_in_tournament_of_26_l1008_100842

/-- The probability of two specific participants playing against each other in a tournament. -/
def probability_of_match (n : ℕ) : ℚ :=
  (n - 1) / (n * (n - 1) / 2)

/-- Theorem: In a tournament with 26 participants, the probability of two specific participants
    playing against each other is 1/13. -/
theorem probability_in_tournament_of_26 :
  probability_of_match 26 = 1 / 13 := by
  sorry

#eval probability_of_match 26  -- To check the result

end NUMINAMATH_CALUDE_probability_in_tournament_of_26_l1008_100842


namespace NUMINAMATH_CALUDE_min_value_of_a_l1008_100812

def matrixOp (a b c d : ℝ) : ℝ := a * d - b * c

theorem min_value_of_a (a : ℝ) :
  (∀ x : ℝ, matrixOp (x - 1) (a - 2) (a + 1) x ≥ 1) →
  a ≥ -1/2 ∧ ∀ b, b < -1/2 → ∃ x, matrixOp (x - 1) (b - 2) (b + 1) x < 1 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_a_l1008_100812


namespace NUMINAMATH_CALUDE_min_benches_for_equal_occupancy_l1008_100876

/-- Represents the capacity of a bench for adults and children -/
structure BenchCapacity where
  adults : Nat
  children : Nat

/-- Finds the minimum number of benches required for equal and full occupancy -/
def minBenchesRequired (capacity : BenchCapacity) : Nat :=
  Nat.lcm capacity.adults capacity.children / capacity.adults

/-- Theorem stating the minimum number of benches required -/
theorem min_benches_for_equal_occupancy (capacity : BenchCapacity) 
  (h1 : capacity.adults = 8) 
  (h2 : capacity.children = 12) : 
  minBenchesRequired capacity = 3 := by
  sorry

#eval minBenchesRequired ⟨8, 12⟩

end NUMINAMATH_CALUDE_min_benches_for_equal_occupancy_l1008_100876


namespace NUMINAMATH_CALUDE_tangent_line_at_zero_l1008_100843

noncomputable def f (x : ℝ) : ℝ := Real.exp (-x) + 1

theorem tangent_line_at_zero : 
  let p : ℝ × ℝ := (0, f 0)
  let m : ℝ := -((deriv f) 0)
  ∀ x y : ℝ, (y - p.2 = m * (x - p.1)) ↔ (x + y - 2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_zero_l1008_100843


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_expression_evaluation_at_3_l1008_100869

theorem expression_simplification_and_evaluation (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 2) :
  ((x + 1) / (x - 2) + 1) / ((x^2 - 2*x) / (x^2 - 4*x + 4)) = (2*x - 1) / x :=
by sorry

theorem expression_evaluation_at_3 :
  let x : ℝ := 3
  ((x + 1) / (x - 2) + 1) / ((x^2 - 2*x) / (x^2 - 4*x + 4)) = 5/3 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_expression_evaluation_at_3_l1008_100869


namespace NUMINAMATH_CALUDE_student_number_problem_l1008_100877

theorem student_number_problem (x : ℝ) : 6 * x - 138 = 102 → x = 40 := by
  sorry

end NUMINAMATH_CALUDE_student_number_problem_l1008_100877


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l1008_100888

theorem sum_of_roots_quadratic (x₁ x₂ : ℝ) : 
  (∀ x, x^2 - 3*x - 4 = 0 ↔ x = x₁ ∨ x = x₂) → x₁ + x₂ = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l1008_100888


namespace NUMINAMATH_CALUDE_school_ball_dance_l1008_100890

theorem school_ball_dance (b g : ℕ) : 
  (∀ n : ℕ, n ≤ b → n + 2 ≤ g) →  -- Each boy dances with at least 3 girls
  (b + 2 = g) →                   -- The last boy dances with all girls
  b = g - 2 := by
sorry

end NUMINAMATH_CALUDE_school_ball_dance_l1008_100890


namespace NUMINAMATH_CALUDE_great_great_grandmother_age_calculation_l1008_100881

-- Define the ages of family members
def darcie_age : ℚ := 4
def mother_age : ℚ := darcie_age * 6
def grandmother_age : ℚ := mother_age * (5/4)
def great_grandfather_age : ℚ := grandmother_age * (4/3)
def great_great_grandmother_age : ℚ := great_grandfather_age * (10/7)

-- Theorem statement
theorem great_great_grandmother_age_calculation :
  great_great_grandmother_age = 400/7 := by
  sorry

end NUMINAMATH_CALUDE_great_great_grandmother_age_calculation_l1008_100881


namespace NUMINAMATH_CALUDE_cards_lost_l1008_100865

theorem cards_lost (initial_cards remaining_cards : ℕ) : 
  initial_cards = 88 → remaining_cards = 18 → initial_cards - remaining_cards = 70 := by
  sorry

end NUMINAMATH_CALUDE_cards_lost_l1008_100865


namespace NUMINAMATH_CALUDE_movies_watched_undetermined_l1008_100816

/-- Represents the "Crazy Silly School" series -/
structure CrazySillySchool where
  total_movies : ℕ
  total_books : ℕ
  books_read : ℕ
  movie_book_difference : ℕ

/-- The conditions of the problem -/
def series : CrazySillySchool :=
  { total_movies := 17
  , total_books := 11
  , books_read := 13
  , movie_book_difference := 6 }

/-- Predicate to check if the number of movies watched can be determined -/
def can_determine_movies_watched (s : CrazySillySchool) : Prop :=
  ∃! n : ℕ, n ≤ s.total_movies

/-- Theorem stating that it's impossible to determine the number of movies watched -/
theorem movies_watched_undetermined (s : CrazySillySchool) 
  (h1 : s.total_movies = s.total_books + s.movie_book_difference)
  (h2 : s.books_read ≤ s.total_books) :
  ¬(can_determine_movies_watched s) :=
sorry

end NUMINAMATH_CALUDE_movies_watched_undetermined_l1008_100816


namespace NUMINAMATH_CALUDE_largest_number_of_three_l1008_100825

theorem largest_number_of_three (p q r : ℝ) 
  (sum_eq : p + q + r = 3)
  (sum_prod_eq : p * q + p * r + q * r = -8)
  (prod_eq : p * q * r = -20) :
  max p (max q r) = (-1 + Real.sqrt 21) / 2 := by
  sorry

end NUMINAMATH_CALUDE_largest_number_of_three_l1008_100825


namespace NUMINAMATH_CALUDE_probability_both_bins_contain_items_l1008_100857

theorem probability_both_bins_contain_items (p : ℝ) (h1 : 0.5 < p) (h2 : p ≤ 1) :
  let prob_both := 1 - 2 * p^5 + p^10
  prob_both = (1 - p^5)^2 + p^10 := by
  sorry

end NUMINAMATH_CALUDE_probability_both_bins_contain_items_l1008_100857


namespace NUMINAMATH_CALUDE_smallest_c_inequality_l1008_100861

theorem smallest_c_inequality (c : ℝ) : 
  (∀ x y z : ℝ, x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 → 
    (x * y * z) ^ (1/3 : ℝ) + c * |x - y + z| ≥ (x + y + z) / 3) ↔ 
  c ≥ 1/3 :=
sorry

end NUMINAMATH_CALUDE_smallest_c_inequality_l1008_100861


namespace NUMINAMATH_CALUDE_prob_even_sum_is_one_third_l1008_100847

/-- Probability of an even outcome for the first wheel -/
def p_even_1 : ℚ := 1/2

/-- Probability of an even outcome for the second wheel -/
def p_even_2 : ℚ := 1/3

/-- Probability of an even outcome for the third wheel -/
def p_even_3 : ℚ := 3/4

/-- The probability of getting an even sum from three independent events -/
def prob_even_sum (p1 p2 p3 : ℚ) : ℚ :=
  p1 * p2 * p3 +
  (1 - p1) * p2 * p3 +
  p1 * (1 - p2) * p3 +
  p1 * p2 * (1 - p3)

/-- Theorem stating that the probability of an even sum is 1/3 given the specific probabilities -/
theorem prob_even_sum_is_one_third :
  prob_even_sum p_even_1 p_even_2 p_even_3 = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_prob_even_sum_is_one_third_l1008_100847


namespace NUMINAMATH_CALUDE_range_of_a_l1008_100839

noncomputable def f (x : ℝ) : ℝ := 2 * x + (Real.exp x)⁻¹ - Real.exp x

theorem range_of_a (a : ℝ) (h : f (a - 1) + f (2 * a^2) ≤ 0) :
  a ≤ -1 ∨ a ≥ 1/2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1008_100839


namespace NUMINAMATH_CALUDE_candy_remaining_l1008_100863

theorem candy_remaining (initial_candy : ℕ) (people : ℕ) (eaten_per_person : ℕ) 
  (h1 : initial_candy = 68) 
  (h2 : people = 2) 
  (h3 : eaten_per_person = 4) : 
  initial_candy - (people * eaten_per_person) = 60 := by
  sorry

end NUMINAMATH_CALUDE_candy_remaining_l1008_100863


namespace NUMINAMATH_CALUDE_infinite_solutions_cube_equation_l1008_100841

theorem infinite_solutions_cube_equation :
  ∀ n : ℕ, ∃ x y z : ℤ, 
    x^2 + y^2 + z^2 = x^3 + y^3 + z^3 ∧
    (∀ m : ℕ, m < n → 
      ∃ x' y' z' : ℤ, 
        x'^2 + y'^2 + z'^2 = x'^3 + y'^3 + z'^3 ∧
        (x', y', z') ≠ (x, y, z)) :=
by
  sorry

end NUMINAMATH_CALUDE_infinite_solutions_cube_equation_l1008_100841


namespace NUMINAMATH_CALUDE_avg_speed_BC_l1008_100886

/-- Represents the journey of a motorcyclist --/
structure Journey where
  distanceAB : ℝ
  distanceBC : ℝ
  timeAB : ℝ
  timeBC : ℝ
  avgSpeedTotal : ℝ

/-- Theorem stating the average speed from B to C given the journey conditions --/
theorem avg_speed_BC (j : Journey)
  (h1 : j.distanceAB = 120)
  (h2 : j.distanceBC = j.distanceAB / 2)
  (h3 : j.timeAB = 3 * j.timeBC)
  (h4 : j.avgSpeedTotal = 20)
  (h5 : j.avgSpeedTotal = (j.distanceAB + j.distanceBC) / (j.timeAB + j.timeBC)) :
  j.distanceBC / j.timeBC = 80 / 3 := by
  sorry

end NUMINAMATH_CALUDE_avg_speed_BC_l1008_100886


namespace NUMINAMATH_CALUDE_area_outside_smaller_squares_l1008_100895

theorem area_outside_smaller_squares (larger_side : ℝ) (smaller_side : ℝ) : 
  larger_side = 10 → 
  smaller_side = 4 → 
  larger_side^2 - 2 * smaller_side^2 = 68 := by
sorry

end NUMINAMATH_CALUDE_area_outside_smaller_squares_l1008_100895


namespace NUMINAMATH_CALUDE_cube_layer_removal_l1008_100855

/-- Calculates the number of smaller cubes remaining inside a cube after removing layers to form a hollow cuboid --/
def remaining_cubes (original_size : Nat) (hollow_size : Nat) : Nat :=
  hollow_size^3 - (hollow_size - 2)^3

/-- Theorem stating that for a 12x12x12 cube with a 10x10x10 hollow cuboid, 488 smaller cubes remain --/
theorem cube_layer_removal :
  remaining_cubes 12 10 = 488 := by
  sorry

end NUMINAMATH_CALUDE_cube_layer_removal_l1008_100855


namespace NUMINAMATH_CALUDE_total_wattage_calculation_l1008_100830

def light_A_initial : ℝ := 60
def light_B_initial : ℝ := 40
def light_C_initial : ℝ := 50

def light_A_increase : ℝ := 0.12
def light_B_increase : ℝ := 0.20
def light_C_increase : ℝ := 0.15

def total_new_wattage : ℝ :=
  light_A_initial * (1 + light_A_increase) +
  light_B_initial * (1 + light_B_increase) +
  light_C_initial * (1 + light_C_increase)

theorem total_wattage_calculation :
  total_new_wattage = 172.7 := by sorry

end NUMINAMATH_CALUDE_total_wattage_calculation_l1008_100830


namespace NUMINAMATH_CALUDE_sum_of_digits_divisible_by_11_l1008_100897

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: Among any 39 consecutive natural numbers, there is always one whose sum of digits is divisible by 11 -/
theorem sum_of_digits_divisible_by_11 (N : ℕ) : 
  ∃ k : ℕ, k ≤ 38 ∧ (sum_of_digits (N + k)) % 11 = 0 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_divisible_by_11_l1008_100897


namespace NUMINAMATH_CALUDE_base7_perfect_square_last_digit_l1008_100822

/-- Checks if a number is a perfect square -/
def isPerfectSquare (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

/-- Represents a number in base 7 as ab2c -/
structure Base7Rep where
  a : ℕ
  b : ℕ
  c : ℕ
  a_nonzero : a ≠ 0

/-- Converts a Base7Rep to its decimal equivalent -/
def toDecimal (rep : Base7Rep) : ℕ :=
  rep.a * 7^3 + rep.b * 7^2 + 2 * 7 + rep.c

theorem base7_perfect_square_last_digit (n : ℕ) (rep : Base7Rep) :
  isPerfectSquare n ∧ n = toDecimal rep → rep.c = 2 ∨ rep.c = 3 ∨ rep.c = 6 := by
  sorry

end NUMINAMATH_CALUDE_base7_perfect_square_last_digit_l1008_100822


namespace NUMINAMATH_CALUDE_regular_polygon_interior_angle_l1008_100811

theorem regular_polygon_interior_angle (n : ℕ) (n_ge_3 : n ≥ 3) :
  (((n - 2) * 180) / n = 150) → n = 12 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_interior_angle_l1008_100811


namespace NUMINAMATH_CALUDE_complement_union_theorem_l1008_100879

def U : Set ℤ := {-1, 1, 2, 3, 4}
def A : Set ℤ := {1, 2, 3}
def B : Set ℤ := {2, 4}

theorem complement_union_theorem :
  (Set.compl A ∩ U) ∪ B = {-1, 2, 4} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l1008_100879


namespace NUMINAMATH_CALUDE_triangle_side_simplification_l1008_100810

theorem triangle_side_simplification (k : ℝ) (h1 : 3 < k) (h2 : k < 5) :
  |2*k - 5| - Real.sqrt (k^2 - 12*k + 36) = 3*k - 11 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_simplification_l1008_100810


namespace NUMINAMATH_CALUDE_local_monotonicity_not_implies_global_l1008_100804

/-- A function that satisfies the local monotonicity condition but is not globally monotonic -/
def exists_locally_monotonic_not_globally : Prop :=
  ∃ (f : ℝ → ℝ), 
    (∀ a : ℝ, ∃ b : ℝ, b > a ∧ (∀ x y : ℝ, a ≤ x ∧ x < y ∧ y ≤ b → f x ≤ f y ∨ f x ≥ f y)) ∧
    ¬(∀ x y : ℝ, x < y → f x ≤ f y ∨ f x ≥ f y)

theorem local_monotonicity_not_implies_global : exists_locally_monotonic_not_globally :=
sorry

end NUMINAMATH_CALUDE_local_monotonicity_not_implies_global_l1008_100804


namespace NUMINAMATH_CALUDE_minimum_point_of_translated_absolute_value_function_l1008_100892

def f (x : ℝ) := |x + 2| - 6

theorem minimum_point_of_translated_absolute_value_function :
  ∃ (x₀ : ℝ), ∀ (x : ℝ), f x₀ ≤ f x ∧ x₀ = -2 ∧ f x₀ = -6 :=
sorry

end NUMINAMATH_CALUDE_minimum_point_of_translated_absolute_value_function_l1008_100892


namespace NUMINAMATH_CALUDE_selection_ways_eq_six_l1008_100875

/-- The number of types of pencils -/
def num_pencil_types : ℕ := 3

/-- The number of types of erasers -/
def num_eraser_types : ℕ := 2

/-- The number of ways to select one pencil and one eraser -/
def num_selection_ways : ℕ := num_pencil_types * num_eraser_types

/-- Theorem stating that the number of ways to select one pencil and one eraser is 6 -/
theorem selection_ways_eq_six : num_selection_ways = 6 := by
  sorry

end NUMINAMATH_CALUDE_selection_ways_eq_six_l1008_100875


namespace NUMINAMATH_CALUDE_cookie_count_l1008_100835

theorem cookie_count (paul_cookies : ℕ) (paula_difference : ℕ) : 
  paul_cookies = 45 →
  paula_difference = 3 →
  paul_cookies + (paul_cookies - paula_difference) = 87 :=
by
  sorry

end NUMINAMATH_CALUDE_cookie_count_l1008_100835


namespace NUMINAMATH_CALUDE_min_additional_coins_for_alex_l1008_100800

/-- The minimum number of additional coins needed -/
def min_additional_coins (friends : ℕ) (initial_coins : ℕ) : ℕ :=
  let required_coins := friends * (friends + 1) / 2
  if required_coins > initial_coins then
    required_coins - initial_coins
  else
    0

/-- Theorem stating the minimum number of additional coins needed -/
theorem min_additional_coins_for_alex : 
  min_additional_coins 15 90 = 30 := by
  sorry

end NUMINAMATH_CALUDE_min_additional_coins_for_alex_l1008_100800


namespace NUMINAMATH_CALUDE_abc_inequality_l1008_100899

theorem abc_inequality (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (ha2 : a ≤ 2) (hb2 : b ≤ 2) (hc2 : c ≤ 2) :
  (a * b * c) / (a + b + c) ≤ 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l1008_100899


namespace NUMINAMATH_CALUDE_language_course_enrollment_l1008_100817

theorem language_course_enrollment (total : ℕ) (french : ℕ) (german : ℕ) (spanish : ℕ)
  (french_german : ℕ) (french_spanish : ℕ) (german_spanish : ℕ) (all_three : ℕ) :
  total = 150 →
  french = 58 →
  german = 40 →
  spanish = 35 →
  french_german = 20 →
  french_spanish = 15 →
  german_spanish = 10 →
  all_three = 5 →
  total - (french + german + spanish - french_german - french_spanish - german_spanish + all_three) = 62 :=
by sorry

end NUMINAMATH_CALUDE_language_course_enrollment_l1008_100817


namespace NUMINAMATH_CALUDE_completing_square_quadratic_l1008_100807

theorem completing_square_quadratic (x : ℝ) : 
  (x^2 - 4*x - 11 = 0) ↔ ((x - 2)^2 = 15) :=
by sorry

end NUMINAMATH_CALUDE_completing_square_quadratic_l1008_100807


namespace NUMINAMATH_CALUDE_domain_relationship_l1008_100853

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f(2x+1)
def domain_f_2x_plus_1 : Set ℝ := Set.Icc (-3) 3

-- Theorem stating the relationship between the domains
theorem domain_relationship :
  (∀ y ∈ domain_f_2x_plus_1, ∃ x, y = 2*x + 1) →
  {x : ℝ | f x ≠ 0} = Set.Icc (-5) 7 :=
sorry

end NUMINAMATH_CALUDE_domain_relationship_l1008_100853


namespace NUMINAMATH_CALUDE_quadratic_equation_with_given_roots_l1008_100832

theorem quadratic_equation_with_given_roots :
  ∀ (a b c : ℝ), a ≠ 0 →
  (∀ x : ℝ, a * x^2 + b * x + c = 0 ↔ x = -2 ∨ x = 3) →
  a * x^2 + b * x + c = x^2 - x - 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_with_given_roots_l1008_100832


namespace NUMINAMATH_CALUDE_chord_length_l1008_100871

/-- The length of the chord formed by the intersection of a circle and a line --/
theorem chord_length (x y : ℝ) : 
  let circle := (x - 1)^2 + y^2 = 4
  let line := x + y + 1 = 0
  let chord_length := Real.sqrt (8 : ℝ)
  (∃ A B : ℝ × ℝ, A ≠ B ∧ 
    ((A.1 - 1)^2 + A.2^2 = 4) ∧ (A.1 + A.2 + 1 = 0) ∧
    ((B.1 - 1)^2 + B.2^2 = 4) ∧ (B.1 + B.2 + 1 = 0)) →
  ∃ A B : ℝ × ℝ, 
    ((A.1 - 1)^2 + A.2^2 = 4) ∧ (A.1 + A.2 + 1 = 0) ∧
    ((B.1 - 1)^2 + B.2^2 = 4) ∧ (B.1 + B.2 + 1 = 0) ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = chord_length :=
by sorry

end NUMINAMATH_CALUDE_chord_length_l1008_100871


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_two_l1008_100883

theorem reciprocal_of_negative_two :
  ∃ x : ℚ, x * (-2) = 1 ∧ x = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_two_l1008_100883


namespace NUMINAMATH_CALUDE_jose_ducks_count_l1008_100805

/-- Given that Jose has 28 chickens and 46 fowls in total, prove that he has 18 ducks. -/
theorem jose_ducks_count (chickens : ℕ) (total_fowls : ℕ) (ducks : ℕ) 
    (h1 : chickens = 28) 
    (h2 : total_fowls = 46) 
    (h3 : total_fowls = chickens + ducks) : 
  ducks = 18 := by
  sorry

end NUMINAMATH_CALUDE_jose_ducks_count_l1008_100805


namespace NUMINAMATH_CALUDE_square_units_digit_nine_l1008_100873

theorem square_units_digit_nine (n : ℕ) : n ≤ 9 → (n^2 % 10 = 9 ↔ n = 3 ∨ n = 7) := by
  sorry

end NUMINAMATH_CALUDE_square_units_digit_nine_l1008_100873


namespace NUMINAMATH_CALUDE_divisors_not_div_by_3_eq_6_l1008_100803

/-- The number of positive divisors of 180 that are not divisible by 3 -/
def divisors_not_div_by_3 : ℕ :=
  (Finset.filter (fun d => d ∣ 180 ∧ ¬(3 ∣ d)) (Finset.range 181)).card

/-- Theorem stating that the number of positive divisors of 180 not divisible by 3 is 6 -/
theorem divisors_not_div_by_3_eq_6 : divisors_not_div_by_3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_divisors_not_div_by_3_eq_6_l1008_100803


namespace NUMINAMATH_CALUDE_simplify_expression_l1008_100801

theorem simplify_expression (b : ℝ) :
  3 * b * (3 * b^2 - 2 * b + 1) + 2 * b^2 = 9 * b^3 - 4 * b^2 + 3 * b := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1008_100801


namespace NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l1008_100840

/-- Given a geometric sequence {a_n} where a_1 + a_2 = 3 and a_2 + a_3 = 6, prove that a_7 = 64 -/
theorem geometric_sequence_seventh_term
  (a : ℕ → ℝ)
  (h1 : a 1 + a 2 = 3)
  (h2 : a 2 + a 3 = 6)
  (h_geom : ∀ n : ℕ, n ≥ 1 → a (n + 1) / a n = a 2 / a 1) :
  a 7 = 64 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l1008_100840


namespace NUMINAMATH_CALUDE_richard_twice_scott_age_l1008_100845

/-- Represents the ages of the three brothers -/
structure BrothersAges where
  david : ℕ
  richard : ℕ
  scott : ℕ

/-- The current ages of the brothers -/
def currentAges : BrothersAges :=
  { david := 14
    richard := 20
    scott := 6 }

/-- The conditions given in the problem -/
axiom age_difference_richard_david : currentAges.richard = currentAges.david + 6
axiom age_difference_david_scott : currentAges.david = currentAges.scott + 8
axiom david_age_three_years_ago : currentAges.david = 11 + 3

/-- The theorem to be proved -/
theorem richard_twice_scott_age (x : ℕ) :
  x = 8 ↔ currentAges.richard + x = 2 * (currentAges.scott + x) :=
sorry

end NUMINAMATH_CALUDE_richard_twice_scott_age_l1008_100845


namespace NUMINAMATH_CALUDE_expected_carrot_yield_l1008_100885

def garden_length_steps : ℕ := 25
def garden_width_steps : ℕ := 35
def step_length_feet : ℕ := 3
def yield_per_sqft : ℚ := 3/4

theorem expected_carrot_yield :
  let garden_length_feet : ℕ := garden_length_steps * step_length_feet
  let garden_width_feet : ℕ := garden_width_steps * step_length_feet
  let garden_area_sqft : ℕ := garden_length_feet * garden_width_feet
  garden_area_sqft * yield_per_sqft = 5906.25 := by
  sorry

end NUMINAMATH_CALUDE_expected_carrot_yield_l1008_100885


namespace NUMINAMATH_CALUDE_tea_bags_in_box_l1008_100820

theorem tea_bags_in_box (cups_per_bag_min cups_per_bag_max : ℕ) 
                        (natasha_cups inna_cups : ℕ) : 
  cups_per_bag_min = 2 →
  cups_per_bag_max = 3 →
  natasha_cups = 41 →
  inna_cups = 58 →
  ∃ n : ℕ, 
    n * cups_per_bag_min ≤ natasha_cups ∧ 
    natasha_cups ≤ n * cups_per_bag_max ∧
    n * cups_per_bag_min ≤ inna_cups ∧ 
    inna_cups ≤ n * cups_per_bag_max ∧
    n = 20 := by
  sorry

end NUMINAMATH_CALUDE_tea_bags_in_box_l1008_100820


namespace NUMINAMATH_CALUDE_function_value_comparison_l1008_100858

def f (x : ℝ) : ℝ := 3 * (x - 2)^2 + 5

theorem function_value_comparison (x₁ x₂ : ℝ) 
  (h : |x₁ - 2| > |x₂ - 2|) : f x₁ > f x₂ := by
  sorry

end NUMINAMATH_CALUDE_function_value_comparison_l1008_100858


namespace NUMINAMATH_CALUDE_median_and_midpoint_lengths_l1008_100896

/-- A right triangle with specific side lengths and a median -/
structure RightTriangleWithMedian where
  -- The length of side XY
  xy : ℝ
  -- The length of side YZ
  yz : ℝ
  -- The point W on side YZ
  w : ℝ
  -- Condition: XY = 6
  xy_eq : xy = 6
  -- Condition: YZ = 8
  yz_eq : yz = 8
  -- Condition: W is the midpoint of YZ
  w_midpoint : w = yz / 2

/-- The length of XW is 5 and the length of WZ is 4 in the given right triangle -/
theorem median_and_midpoint_lengths (t : RightTriangleWithMedian) : 
  Real.sqrt (t.xy^2 + (t.yz - t.w)^2) = 5 ∧ t.w = 4 := by
  sorry


end NUMINAMATH_CALUDE_median_and_midpoint_lengths_l1008_100896


namespace NUMINAMATH_CALUDE_sum_of_powers_of_i_equals_zero_l1008_100836

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem
theorem sum_of_powers_of_i_equals_zero :
  i^1234 + i^1235 + i^1236 + i^1237 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_of_i_equals_zero_l1008_100836


namespace NUMINAMATH_CALUDE_bookshelf_discount_percentage_l1008_100851

theorem bookshelf_discount_percentage (discount : ℝ) (final_price : ℝ) (tax_rate : ℝ) : 
  discount = 4.50 →
  final_price = 49.50 →
  tax_rate = 0.10 →
  (discount / (final_price / (1 + tax_rate) + discount)) * 100 = 9 := by
sorry

end NUMINAMATH_CALUDE_bookshelf_discount_percentage_l1008_100851


namespace NUMINAMATH_CALUDE_remainder_sum_mod_five_l1008_100809

theorem remainder_sum_mod_five (f y : ℤ) 
  (hf : f % 5 = 3) 
  (hy : y % 5 = 4) : 
  (f + y) % 5 = 2 :=
by sorry

end NUMINAMATH_CALUDE_remainder_sum_mod_five_l1008_100809


namespace NUMINAMATH_CALUDE_bank_coin_count_l1008_100878

/-- The total number of coins turned in by a customer at a bank -/
def total_coins (dimes nickels quarters : ℕ) : ℕ :=
  dimes + nickels + quarters

/-- Theorem stating that the total number of coins is 11 given the specific quantities -/
theorem bank_coin_count : total_coins 2 2 7 = 11 := by
  sorry

end NUMINAMATH_CALUDE_bank_coin_count_l1008_100878


namespace NUMINAMATH_CALUDE_puppies_per_dog_l1008_100894

theorem puppies_per_dog (num_dogs : ℕ) (total_puppies : ℕ) : 
  num_dogs = 15 → total_puppies = 75 → total_puppies / num_dogs = 5 := by
  sorry

end NUMINAMATH_CALUDE_puppies_per_dog_l1008_100894


namespace NUMINAMATH_CALUDE_power_of_seven_mod_ten_l1008_100837

theorem power_of_seven_mod_ten : 7^150 % 10 = 9 := by
  sorry

end NUMINAMATH_CALUDE_power_of_seven_mod_ten_l1008_100837


namespace NUMINAMATH_CALUDE_camping_bowls_l1008_100884

theorem camping_bowls (total_bowls : ℕ) (rice_per_person : ℚ) (dish_per_person : ℚ) (soup_per_person : ℚ) :
  total_bowls = 55 ∧ 
  rice_per_person = 1 ∧ 
  dish_per_person = 1/2 ∧ 
  soup_per_person = 1/3 →
  (total_bowls : ℚ) / (rice_per_person + dish_per_person + soup_per_person) = 30 := by
sorry

end NUMINAMATH_CALUDE_camping_bowls_l1008_100884


namespace NUMINAMATH_CALUDE_binomial_square_coefficient_l1008_100813

theorem binomial_square_coefficient (a : ℚ) : 
  (∃ r s : ℚ, ∀ x, a * x^2 + 18 * x + 16 = (r * x + s)^2) → a = 81/16 := by
  sorry

end NUMINAMATH_CALUDE_binomial_square_coefficient_l1008_100813


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_maximum_l1008_100815

theorem arithmetic_sequence_sum_maximum (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, a (n + 2) - a (n + 1) = a (n + 1) - a n) →  -- arithmetic sequence
  (a 11 / a 10 < -1) →  -- given condition
  (∃ k, ∀ n, S n ≤ S k) →  -- sum has a maximum value
  (∀ n > 19, S n ≤ 0) ∧ (S 19 > 0) :=
by sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_sum_maximum_l1008_100815


namespace NUMINAMATH_CALUDE_solve_equation_l1008_100826

theorem solve_equation (n : ℚ) : (1 / (2 * n)) + (1 / (4 * n)) = 3 / 12 → n = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1008_100826


namespace NUMINAMATH_CALUDE_division_problem_l1008_100829

theorem division_problem (dividend : Nat) (divisor : Nat) (quotient : Nat) (remainder : Nat) :
  dividend = 144 ∧ divisor = 11 ∧ remainder = 1 →
  dividend = divisor * quotient + remainder →
  quotient = 13 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l1008_100829


namespace NUMINAMATH_CALUDE_computer_peripherals_cost_fraction_l1008_100848

theorem computer_peripherals_cost_fraction :
  let computer_cost : ℚ := 1500
  let base_video_card_cost : ℚ := 300
  let upgraded_video_card_cost : ℚ := 2 * base_video_card_cost
  let total_spent : ℚ := 2100
  let computer_with_upgrade_cost : ℚ := computer_cost + upgraded_video_card_cost - base_video_card_cost
  let peripherals_cost : ℚ := total_spent - computer_with_upgrade_cost
  peripherals_cost / computer_cost = 1 / 5 := by
sorry

end NUMINAMATH_CALUDE_computer_peripherals_cost_fraction_l1008_100848


namespace NUMINAMATH_CALUDE_remainder_3_100_mod_7_l1008_100856

theorem remainder_3_100_mod_7 : 3^100 % 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_3_100_mod_7_l1008_100856


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l1008_100819

theorem polynomial_division_remainder :
  ∃ Q : Polynomial ℝ, (X : Polynomial ℝ)^5 - 3 * X^3 + 4 * X + 5 = 
  (X - 3)^2 * Q + (261 * X - 643) := by sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l1008_100819


namespace NUMINAMATH_CALUDE_quadratic_polynomials_sum_l1008_100882

/-- Two distinct quadratic polynomials with the given properties have a + c = -600 -/
theorem quadratic_polynomials_sum (a b c d : ℝ) : 
  let f (x : ℝ) := x^2 + a*x + b
  let g (x : ℝ) := x^2 + c*x + d
  ∀ x y : ℝ, 
  (f ≠ g) →  -- f and g are distinct
  (g (-a/2) = 0) →  -- x-coordinate of vertex of f is a root of g
  (f (-c/2) = 0) →  -- x-coordinate of vertex of g is a root of f
  (∃ (m : ℝ), ∀ (x : ℝ), f x ≥ m ∧ g x ≥ m) →  -- f and g yield the same minimum value
  (f 150 = -200 ∧ g 150 = -200) →  -- f and g intersect at (150, -200)
  a + c = -600 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomials_sum_l1008_100882


namespace NUMINAMATH_CALUDE_min_value_expression_l1008_100887

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x^2 + 5*x + 2) * (y^2 + 5*y + 2) * (z^2 + 5*z + 2) / (x*y*z) ≥ 512 ∧
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    (a^2 + 5*a + 2) * (b^2 + 5*b + 2) * (c^2 + 5*c + 2) / (a*b*c) = 512 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l1008_100887


namespace NUMINAMATH_CALUDE_polygon_sides_l1008_100846

theorem polygon_sides (n : ℕ) : 
  (n ≥ 3) →
  ((n - 2) * 180 = 3 * 360 + 180) →
  n = 9 := by
sorry

end NUMINAMATH_CALUDE_polygon_sides_l1008_100846


namespace NUMINAMATH_CALUDE_dave_chocolate_boxes_l1008_100854

theorem dave_chocolate_boxes (total_boxes : ℕ) (pieces_per_box : ℕ) (pieces_left : ℕ) : 
  total_boxes = 12 → pieces_per_box = 3 → pieces_left = 21 →
  (total_boxes * pieces_per_box - pieces_left) / pieces_per_box = 5 := by
sorry

end NUMINAMATH_CALUDE_dave_chocolate_boxes_l1008_100854


namespace NUMINAMATH_CALUDE_man_son_age_difference_l1008_100852

/-- Represents the age difference between a man and his son -/
def AgeDifference (sonAge manAge : ℕ) : ℕ := manAge - sonAge

theorem man_son_age_difference :
  ∀ (sonAge manAge : ℕ),
  sonAge = 22 →
  manAge + 2 = 2 * (sonAge + 2) →
  AgeDifference sonAge manAge = 24 := by
  sorry

end NUMINAMATH_CALUDE_man_son_age_difference_l1008_100852


namespace NUMINAMATH_CALUDE_positive_root_iff_p_in_set_l1008_100864

-- Define the polynomial equation
def f (p x : ℝ) : ℝ := x^4 + 4*p*x^3 + x^2 + 4*p*x + 4

-- Define the set of p values
def P : Set ℝ := {p | p < -Real.sqrt 2 / 2 ∨ p > Real.sqrt 2 / 2}

-- Theorem statement
theorem positive_root_iff_p_in_set (p : ℝ) :
  (∃ x : ℝ, x > 0 ∧ f p x = 0) ↔ p ∈ P :=
sorry

end NUMINAMATH_CALUDE_positive_root_iff_p_in_set_l1008_100864


namespace NUMINAMATH_CALUDE_perfect_cube_in_range_l1008_100831

theorem perfect_cube_in_range : 
  ∃! (K : ℤ), 
    K > 1 ∧ 
    ∃ (Z : ℤ), 3000 < Z ∧ Z < 4000 ∧ Z = K^4 ∧ 
    ∃ (n : ℤ), Z = n^3 ∧
    K = 7 := by
  sorry

end NUMINAMATH_CALUDE_perfect_cube_in_range_l1008_100831


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_l1008_100891

open Real

theorem isosceles_right_triangle (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  log a - log c = log (sin B) →
  log (sin B) = -log (sqrt 2) →
  B < π / 2 →
  a = b ∧ C = π / 2 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_l1008_100891


namespace NUMINAMATH_CALUDE_x_value_l1008_100866

theorem x_value : ∃ x : ℝ, (49 / 49 = x ^ 4) → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l1008_100866


namespace NUMINAMATH_CALUDE_father_picked_22_8_pounds_l1008_100844

/-- Represents the amount of strawberries picked by each person in pounds -/
structure StrawberryPicking where
  marco : ℝ
  sister : ℝ
  father : ℝ

/-- Converts kilograms to pounds -/
def kg_to_pounds (kg : ℝ) : ℝ := kg * 2.2

/-- Calculates the amount of strawberries picked by each person -/
def strawberry_picking : StrawberryPicking :=
  let marco_pounds := 1 + kg_to_pounds 3
  let sister_pounds := 1.5 * marco_pounds
  let father_pounds := 2 * sister_pounds
  { marco := marco_pounds,
    sister := sister_pounds,
    father := father_pounds }

/-- Theorem stating that the father picked 22.8 pounds of strawberries -/
theorem father_picked_22_8_pounds :
  strawberry_picking.father = 22.8 := by
  sorry

end NUMINAMATH_CALUDE_father_picked_22_8_pounds_l1008_100844


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1008_100838

theorem sufficient_not_necessary (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∀ a b, a > 0 → b > 0 → a * b < 3 → 1 / a + 4 / b > 2) ∧ 
  (∃ a b, a > 0 ∧ b > 0 ∧ 1 / a + 4 / b > 2 ∧ a * b ≥ 3) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1008_100838


namespace NUMINAMATH_CALUDE_absolute_value_inequality_find_a_value_l1008_100880

-- Part 1
theorem absolute_value_inequality (x : ℝ) :
  (|x - 1| + |x + 2| ≥ 5) ↔ (x ≤ -3 ∨ x ≥ 2) := by sorry

-- Part 2
theorem find_a_value (a : ℝ) :
  (∀ x, |a*x - 2| < 3 ↔ -5/3 < x ∧ x < 1/3) → a = -3 := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_find_a_value_l1008_100880


namespace NUMINAMATH_CALUDE_temperature_peak_l1008_100893

theorem temperature_peak (t : ℝ) : 
  (∀ s : ℝ, -s^2 + 10*s + 60 = 80 → s ≤ 5 + Real.sqrt 5) ∧ 
  (-((5 + Real.sqrt 5)^2) + 10*(5 + Real.sqrt 5) + 60 = 80) := by
sorry

end NUMINAMATH_CALUDE_temperature_peak_l1008_100893


namespace NUMINAMATH_CALUDE_eldoras_purchase_cost_is_55_40_l1008_100898

/-- The cost of Eldora's purchase of paper clips and index cards -/
def eldoras_purchase_cost (index_card_price : ℝ) : ℝ :=
  15 * 1.85 + 7 * index_card_price

/-- The cost of Finn's purchase of paper clips and index cards -/
def finns_purchase_cost (index_card_price : ℝ) : ℝ :=
  12 * 1.85 + 10 * index_card_price

/-- Theorem stating the cost of Eldora's purchase -/
theorem eldoras_purchase_cost_is_55_40 :
  ∃ (index_card_price : ℝ),
    finns_purchase_cost index_card_price = 61.70 ∧
    eldoras_purchase_cost index_card_price = 55.40 := by
  sorry

end NUMINAMATH_CALUDE_eldoras_purchase_cost_is_55_40_l1008_100898


namespace NUMINAMATH_CALUDE_junior_score_theorem_l1008_100870

theorem junior_score_theorem (n : ℝ) (h : n > 0) :
  let junior_ratio : ℝ := 0.2
  let senior_ratio : ℝ := 0.8
  let total_average : ℝ := 80
  let senior_average : ℝ := 78
  let junior_count : ℝ := junior_ratio * n
  let senior_count : ℝ := senior_ratio * n
  let total_score : ℝ := total_average * n
  let senior_total_score : ℝ := senior_average * senior_count
  let junior_total_score : ℝ := total_score - senior_total_score
  junior_total_score / junior_count = 88 :=
by sorry

end NUMINAMATH_CALUDE_junior_score_theorem_l1008_100870


namespace NUMINAMATH_CALUDE_building_entry_exit_ways_l1008_100868

/-- The number of ways to enter and exit a building with 4 doors, entering and exiting through different doors -/
def number_of_ways (num_doors : ℕ) : ℕ :=
  num_doors * (num_doors - 1)

/-- Theorem stating that for a building with 4 doors, there are 12 ways to enter and exit through different doors -/
theorem building_entry_exit_ways :
  number_of_ways 4 = 12 := by
  sorry

end NUMINAMATH_CALUDE_building_entry_exit_ways_l1008_100868


namespace NUMINAMATH_CALUDE_matthias_basketballs_l1008_100867

/-- Given information about Matthias' balls, prove the total number of basketballs --/
theorem matthias_basketballs 
  (total_soccer : ℕ)
  (soccer_with_holes : ℕ)
  (basketball_with_holes : ℕ)
  (total_without_holes : ℕ)
  (h1 : total_soccer = 40)
  (h2 : soccer_with_holes = 30)
  (h3 : basketball_with_holes = 7)
  (h4 : total_without_holes = 18)
  (h5 : total_without_holes = total_soccer - soccer_with_holes + (total_basketballs - basketball_with_holes)) :
  total_basketballs = 15 :=
by
  sorry


end NUMINAMATH_CALUDE_matthias_basketballs_l1008_100867


namespace NUMINAMATH_CALUDE_g_of_3_eq_6_l1008_100824

/-- The function g(x) = x^3 - 3x^2 + 2x -/
def g (x : ℝ) : ℝ := x^3 - 3*x^2 + 2*x

/-- Theorem: g(3) = 6 -/
theorem g_of_3_eq_6 : g 3 = 6 := by sorry

end NUMINAMATH_CALUDE_g_of_3_eq_6_l1008_100824


namespace NUMINAMATH_CALUDE_tangent_line_parabola_l1008_100821

/-- The equation of the tangent line to the parabola y = x^2 at the point (-1, 1) -/
theorem tangent_line_parabola :
  let f (x : ℝ) := x^2
  let p : ℝ × ℝ := (-1, 1)
  let tangent_line (x y : ℝ) := 2*x + y + 1 = 0
  (∀ x, (f x, x) ∈ Set.range (λ t => (t, f t))) →
  (p.1, p.2) ∈ Set.range (λ t => (t, f t)) →
  ∃ m b, (∀ x, tangent_line x (m*x + b)) ∧
         (tangent_line p.1 p.2) ∧
         (∀ ε > 0, ∃ δ > 0, ∀ x, |x - p.1| < δ → |f x - (m*x + b)| < ε * |x - p.1|) := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_parabola_l1008_100821


namespace NUMINAMATH_CALUDE_election_percentage_l1008_100874

theorem election_percentage (total_votes : ℝ) (candidate_votes : ℝ) 
  (h1 : candidate_votes > 0)
  (h2 : total_votes > candidate_votes)
  (h3 : candidate_votes + (1/3) * (total_votes - candidate_votes) = (1/2) * total_votes) :
  candidate_votes / total_votes = 1/4 := by
sorry

end NUMINAMATH_CALUDE_election_percentage_l1008_100874


namespace NUMINAMATH_CALUDE_hyperbola_x_axis_m_range_l1008_100872

/-- Represents the equation of a conic section -/
structure ConicSection where
  m : ℝ
  equation : ℝ → ℝ → Prop := λ x y => x^2 / m + y^2 / (m - 4) = 1

/-- Represents a hyperbola with foci on the x-axis -/
class HyperbolaXAxis extends ConicSection

/-- The range of m for a hyperbola with foci on the x-axis -/
def is_valid_m (m : ℝ) : Prop := 0 < m ∧ m < 4

/-- Theorem stating the condition for m to represent a hyperbola with foci on the x-axis -/
theorem hyperbola_x_axis_m_range (h : HyperbolaXAxis) :
  is_valid_m h.m :=
sorry

end NUMINAMATH_CALUDE_hyperbola_x_axis_m_range_l1008_100872


namespace NUMINAMATH_CALUDE_exists_x_for_all_m_greater_than_one_l1008_100814

-- Define the function f
def f (x : ℝ) : ℝ := |x + 3| + |x - 2|

-- State the theorem
theorem exists_x_for_all_m_greater_than_one :
  ∀ m : ℝ, m > 1 → ∃ x : ℝ, f x = 4 / (m - 1) + m :=
by sorry

end NUMINAMATH_CALUDE_exists_x_for_all_m_greater_than_one_l1008_100814


namespace NUMINAMATH_CALUDE_ellipse_chord_slope_l1008_100806

/-- Given an ellipse with equation x²/16 + y²/9 = 1, 
    the slope of any chord with midpoint (1,2) is -9/32 -/
theorem ellipse_chord_slope :
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
  (x₁^2 / 16 + y₁^2 / 9 = 1) →
  (x₂^2 / 16 + y₂^2 / 9 = 1) →
  ((x₁ + x₂) / 2 = 1) →
  ((y₁ + y₂) / 2 = 2) →
  (y₂ - y₁) / (x₂ - x₁) = -9/32 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_chord_slope_l1008_100806


namespace NUMINAMATH_CALUDE_no_triangle_with_special_angles_l1008_100860

theorem no_triangle_with_special_angles : 
  ¬ ∃ (α β γ : Real), 
    α + β + γ = Real.pi ∧ 
    ((3 * Real.cos α - 2) * (14 * Real.sin α ^ 2 + Real.sin (2 * α) - 12) = 0) ∧
    ((3 * Real.cos β - 2) * (14 * Real.sin β ^ 2 + Real.sin (2 * β) - 12) = 0) ∧
    ((3 * Real.cos γ - 2) * (14 * Real.sin γ ^ 2 + Real.sin (2 * γ) - 12) = 0) :=
by sorry

end NUMINAMATH_CALUDE_no_triangle_with_special_angles_l1008_100860


namespace NUMINAMATH_CALUDE_symmetric_circle_equation_l1008_100889

/-- The standard equation of a circle symmetric to x^2 + y^2 = 1 with respect to x + y = 1 -/
theorem symmetric_circle_equation :
  let C : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 1}
  let l : Set (ℝ × ℝ) := {p | p.1 + p.2 = 1}
  let symmetric_circle : Set (ℝ × ℝ) := {p | (p.1 - 1)^2 + (p.2 - 1)^2 = 1}
  symmetric_circle = {p | ∃q ∈ C, p.1 + q.1 = 1 ∧ p.2 + q.2 = 1} := by
  sorry

end NUMINAMATH_CALUDE_symmetric_circle_equation_l1008_100889


namespace NUMINAMATH_CALUDE_bouncy_balls_per_package_l1008_100849

/-- The number of bouncy balls in each package -/
def balls_per_package : ℝ := 10

/-- The number of packs of yellow bouncy balls Maggie bought -/
def yellow_packs : ℝ := 8.0

/-- The number of packs of green bouncy balls Maggie gave away -/
def green_packs_given : ℝ := 4.0

/-- The number of packs of green bouncy balls Maggie bought -/
def green_packs_bought : ℝ := 4.0

/-- The total number of bouncy balls Maggie kept -/
def total_balls : ℕ := 80

theorem bouncy_balls_per_package :
  yellow_packs * balls_per_package = total_balls := by sorry

end NUMINAMATH_CALUDE_bouncy_balls_per_package_l1008_100849
