import Mathlib

namespace NUMINAMATH_CALUDE_total_rain_time_l3191_319179

def rain_duration_day1 : ℕ := 10

def rain_duration_day2 (d1 : ℕ) : ℕ := d1 + 2

def rain_duration_day3 (d2 : ℕ) : ℕ := 2 * d2

def total_rain_duration (d1 d2 d3 : ℕ) : ℕ := d1 + d2 + d3

theorem total_rain_time :
  total_rain_duration rain_duration_day1 
    (rain_duration_day2 rain_duration_day1) 
    (rain_duration_day3 (rain_duration_day2 rain_duration_day1)) = 46 := by
  sorry

end NUMINAMATH_CALUDE_total_rain_time_l3191_319179


namespace NUMINAMATH_CALUDE_expand_product_l3191_319143

theorem expand_product (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l3191_319143


namespace NUMINAMATH_CALUDE_triangle_inequality_constant_l3191_319178

theorem triangle_inequality_constant (a b c : ℝ) (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  (a^2 + b^2) / c^2 > 1/2 ∧ ∀ N : ℝ, (∀ a' b' c' : ℝ, a' > 0 → b' > 0 → c' > 0 → a' + b' > c' → b' + c' > a' → c' + a' > b' → (a'^2 + b'^2) / c'^2 > N) → N ≤ 1/2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_constant_l3191_319178


namespace NUMINAMATH_CALUDE_otimes_two_one_l3191_319154

-- Define the new operation ⊗
def otimes (a b : ℝ) : ℝ := a^2 - b

-- Theorem statement
theorem otimes_two_one : otimes 2 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_otimes_two_one_l3191_319154


namespace NUMINAMATH_CALUDE_tourist_arrangement_count_l3191_319156

/-- The number of tourists --/
def num_tourists : ℕ := 5

/-- The number of scenic spots --/
def num_spots : ℕ := 4

/-- The function to calculate the number of valid arrangements --/
def valid_arrangements (n : ℕ) (k : ℕ) : ℕ :=
  k^n - Nat.choose k 1 * (k-1)^n + Nat.choose k 2 * (k-2)^n - Nat.choose k 3 * (k-3)^n

/-- The main theorem to prove --/
theorem tourist_arrangement_count :
  (valid_arrangements num_tourists num_spots) * (num_spots - 1) * (num_spots - 1) / num_spots = 216 :=
sorry

end NUMINAMATH_CALUDE_tourist_arrangement_count_l3191_319156


namespace NUMINAMATH_CALUDE_dance_class_girls_l3191_319195

theorem dance_class_girls (total : ℕ) (g b : ℚ) : 
  total = 28 →
  g / b = 3 / 4 →
  g + b = total →
  g = 12 := by sorry

end NUMINAMATH_CALUDE_dance_class_girls_l3191_319195


namespace NUMINAMATH_CALUDE_expand_expression_l3191_319101

theorem expand_expression (x : ℝ) : 3 * (x - 7) * (x + 10) + 5 * x = 3 * x^2 + 14 * x - 210 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l3191_319101


namespace NUMINAMATH_CALUDE_polyline_distance_bound_l3191_319176

/-- Polyline distance between two points -/
def polyline_distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := |x₁ - x₂| + |y₁ - y₂|

/-- Theorem: For any point C(x, y) with polyline distance 1 from O(0, 0), √(x² + y²) ≥ √2/2 -/
theorem polyline_distance_bound (x y : ℝ) 
  (h : polyline_distance 0 0 x y = 1) : 
  Real.sqrt (x^2 + y^2) ≥ Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_polyline_distance_bound_l3191_319176


namespace NUMINAMATH_CALUDE_ellipse_and_dot_product_range_l3191_319188

-- Define the ellipse C
def ellipse (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := y^2 / 2 - x^2 = 1

-- Define the line l
def line (m : ℝ) (x y : ℝ) : Prop := x = m * y + 4

-- Define the dot product of OA and OB
def dot_product (xa ya xb yb : ℝ) : ℝ := xa * xb + ya * yb

theorem ellipse_and_dot_product_range :
  ∀ (a b : ℝ),
  a > b ∧ b > 0 →
  (∀ x y, ellipse a b x y → x^2 / a^2 + y^2 / b^2 = 1) →
  a^2 / b^2 - 1 = 1/4 →
  (∃ x, hyperbola 0 x) →
  (∀ m : ℝ, m ≠ 0 → ∃ xa ya xb yb,
    line m xa ya ∧ line m xb yb ∧
    ellipse a b xa ya ∧ ellipse a b xb yb) →
  (∀ x y, ellipse a b x y ↔ x^2 / 4 + y^2 / 3 = 1) ∧
  (∀ xa ya xb yb,
    ellipse a b xa ya ∧ ellipse a b xb yb →
    -4 ≤ dot_product xa ya xb yb ∧ dot_product xa ya xb yb < 13/4) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_and_dot_product_range_l3191_319188


namespace NUMINAMATH_CALUDE_unique_prime_triple_l3191_319174

theorem unique_prime_triple : ∃! (p q r : ℕ), 
  Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧
  Nat.Prime (4 * q - 1) ∧
  (p + q : ℚ) / (p + r) = r - p ∧
  p = 2 ∧ q = 3 ∧ r = 3 := by
sorry

end NUMINAMATH_CALUDE_unique_prime_triple_l3191_319174


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3191_319128

theorem inequality_solution_set (x : ℝ) :
  (Set.Ioo (-1 : ℝ) 3) = {x | (3 - x) * (1 + x) > 0} := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3191_319128


namespace NUMINAMATH_CALUDE_smallest_five_digit_palindrome_divisible_by_three_l3191_319149

/-- A function that checks if a number is a five-digit palindrome -/
def is_five_digit_palindrome (n : ℕ) : Prop :=
  10000 ≤ n ∧ n ≤ 99999 ∧ 
  (n / 10000 = n % 10) ∧ 
  ((n / 1000) % 10 = (n / 10) % 10)

/-- The smallest five-digit palindrome divisible by 3 -/
def smallest_palindrome : ℕ := 10001

theorem smallest_five_digit_palindrome_divisible_by_three :
  is_five_digit_palindrome smallest_palindrome ∧ 
  smallest_palindrome % 3 = 0 ∧
  ∀ n : ℕ, is_five_digit_palindrome n → n % 3 = 0 → n ≥ smallest_palindrome := by
  sorry

#eval smallest_palindrome

end NUMINAMATH_CALUDE_smallest_five_digit_palindrome_divisible_by_three_l3191_319149


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_inequality_l3191_319127

theorem unique_solution_quadratic_inequality (b : ℝ) : 
  (∃! x : ℝ, |x^2 + 2*b*x + 2*b| ≤ 1) ↔ b = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_inequality_l3191_319127


namespace NUMINAMATH_CALUDE_newspaper_distribution_l3191_319147

theorem newspaper_distribution (F : ℚ) : 
  200 * F + 0.6 * (200 - 200 * F) = 200 - 48 → F = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_newspaper_distribution_l3191_319147


namespace NUMINAMATH_CALUDE_alex_walking_distance_l3191_319157

/-- Calculates the distance traveled given speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Represents Alex's bike journey -/
structure BikeJourney where
  totalDistance : ℝ
  flatSpeed : ℝ
  flatTime : ℝ
  uphillSpeed : ℝ
  uphillTime : ℝ
  downhillSpeed : ℝ
  downhillTime : ℝ

/-- Calculates the distance Alex had to walk -/
def distanceToWalk (journey : BikeJourney) : ℝ :=
  journey.totalDistance - (distance journey.flatSpeed journey.flatTime +
                           distance journey.uphillSpeed journey.uphillTime +
                           distance journey.downhillSpeed journey.downhillTime)

theorem alex_walking_distance :
  let journey : BikeJourney := {
    totalDistance := 164,
    flatSpeed := 20,
    flatTime := 4.5,
    uphillSpeed := 12,
    uphillTime := 2.5,
    downhillSpeed := 24,
    downhillTime := 1.5
  }
  distanceToWalk journey = 8 := by
  sorry

end NUMINAMATH_CALUDE_alex_walking_distance_l3191_319157


namespace NUMINAMATH_CALUDE_solution_set_inequalities_l3191_319144

theorem solution_set_inequalities (x : ℝ) :
  (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) := by sorry

end NUMINAMATH_CALUDE_solution_set_inequalities_l3191_319144


namespace NUMINAMATH_CALUDE_problem_surface_area_l3191_319105

/-- Represents a solid block formed by unit cubes -/
structure SolidBlock where
  base_width : ℕ
  base_length : ℕ
  base_height : ℕ
  top_cubes : ℕ

/-- Calculates the surface area of a SolidBlock -/
def surface_area (block : SolidBlock) : ℕ :=
  sorry

/-- The specific solid block described in the problem -/
def problem_block : SolidBlock :=
  { base_width := 3
  , base_length := 2
  , base_height := 2
  , top_cubes := 2 }

theorem problem_surface_area : surface_area problem_block = 42 := by
  sorry

end NUMINAMATH_CALUDE_problem_surface_area_l3191_319105


namespace NUMINAMATH_CALUDE_regular_triangular_pyramid_volume_l3191_319180

/-- The volume of a regular triangular pyramid -/
theorem regular_triangular_pyramid_volume 
  (a b γ : ℝ) 
  (h_a : a > 0) 
  (h_b : b > 0) 
  (h_γ : 0 < γ ∧ γ < π) : 
  ∃ V : ℝ, V = (1/3) * (a^2 * Real.sqrt 3 / 4) * 
    Real.sqrt (b^2 - (a * Real.sqrt 3 / (2 * Real.cos (γ/2)))^2) := by
  sorry

end NUMINAMATH_CALUDE_regular_triangular_pyramid_volume_l3191_319180


namespace NUMINAMATH_CALUDE_parallel_planes_from_parallel_intersecting_lines_parallel_planes_transitive_l3191_319104

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relationships between planes and lines
variable (parallel : Plane → Plane → Prop)
variable (intersect : Line → Line → Prop)
variable (in_plane : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)

-- Theorem 1
theorem parallel_planes_from_parallel_intersecting_lines 
  (α β : Plane) (l1 l2 m1 m2 : Line) :
  in_plane l1 α → in_plane l2 α → intersect l1 l2 →
  in_plane m1 β → in_plane m2 β → intersect m1 m2 →
  parallel_lines l1 m1 → parallel_lines l2 m2 →
  parallel α β :=
sorry

-- Theorem 2
theorem parallel_planes_transitive (α β γ : Plane) :
  parallel α β → parallel β γ → parallel α γ :=
sorry

end NUMINAMATH_CALUDE_parallel_planes_from_parallel_intersecting_lines_parallel_planes_transitive_l3191_319104


namespace NUMINAMATH_CALUDE_ellipse_area_theorem_l3191_319150

/-- The ellipse equation -/
def ellipse_equation (x y : ℝ) : Prop := x^2 / 25 + y^2 / 9 = 1

/-- The foci of the ellipse -/
def F1 : ℝ × ℝ := sorry
def F2 : ℝ × ℝ := sorry

/-- A point on the ellipse -/
def P : ℝ × ℝ := sorry

/-- PF1 is perpendicular to PF2 -/
def PF1_perp_PF2 : Prop := sorry

/-- The area of triangle F1PF2 -/
def area_F1PF2 : ℝ := sorry

theorem ellipse_area_theorem :
  ellipse_equation P.1 P.2 →
  PF1_perp_PF2 →
  area_F1PF2 = 9 := by sorry

end NUMINAMATH_CALUDE_ellipse_area_theorem_l3191_319150


namespace NUMINAMATH_CALUDE_tangent_perpendicular_points_l3191_319181

noncomputable def f (x : ℝ) : ℝ := x^3 + x - 2

theorem tangent_perpendicular_points :
  ∀ x y : ℝ, f x = y →
    (3 * x^2 + 1 = 4 ∨ 3 * x^2 + 1 = -1/4) ↔ (x = 1 ∧ y = 0) ∨ (x = -1 ∧ y = -4) :=
by sorry

end NUMINAMATH_CALUDE_tangent_perpendicular_points_l3191_319181


namespace NUMINAMATH_CALUDE_queens_free_subgrid_l3191_319113

/-- Represents a chessboard with queens -/
structure Chessboard :=
  (size : Nat)
  (queens : Nat)

/-- Theorem: On an 8x8 chessboard with 12 queens, there always exist four rows and four columns
    such that none of the 16 cells at their intersections contain a queen -/
theorem queens_free_subgrid (board : Chessboard) 
  (h1 : board.size = 8) 
  (h2 : board.queens = 12) : 
  ∃ (rows columns : Finset Nat), 
    rows.card = 4 ∧ 
    columns.card = 4 ∧ 
    (∀ r ∈ rows, ∀ c ∈ columns, 
      ¬∃ (queen : Nat × Nat), queen.1 = r ∧ queen.2 = c) :=
sorry

end NUMINAMATH_CALUDE_queens_free_subgrid_l3191_319113


namespace NUMINAMATH_CALUDE_max_salary_for_given_constraints_l3191_319165

/-- Represents a baseball team with salary constraints -/
structure BaseballTeam where
  num_players : ℕ
  min_salary : ℕ
  max_total_salary : ℕ

/-- Calculates the maximum possible salary for a single player -/
def max_single_player_salary (team : BaseballTeam) : ℕ :=
  team.max_total_salary - (team.num_players - 1) * team.min_salary

/-- Theorem stating the maximum possible salary for a single player
    in a team with given constraints -/
theorem max_salary_for_given_constraints :
  let team : BaseballTeam := {
    num_players := 25,
    min_salary := 20000,
    max_total_salary := 800000
  }
  max_single_player_salary team = 320000 := by
  sorry

#eval max_single_player_salary {
  num_players := 25,
  min_salary := 20000,
  max_total_salary := 800000
}

end NUMINAMATH_CALUDE_max_salary_for_given_constraints_l3191_319165


namespace NUMINAMATH_CALUDE_smallest_n_99n_all_threes_l3191_319118

def all_threes (n : ℕ) : Prop :=
  ∀ d, d ∈ (n.digits 10) → d = 3

theorem smallest_n_99n_all_threes :
  ∃ (N : ℕ), (N = 3367 ∧ all_threes (99 * N) ∧ ∀ n < N, ¬ all_threes (99 * n)) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_99n_all_threes_l3191_319118


namespace NUMINAMATH_CALUDE_magic_square_y_value_l3191_319199

/-- Represents a 3x3 modified magic square -/
structure ModifiedMagicSquare where
  entries : Matrix (Fin 3) (Fin 3) ℕ
  is_magic : ∀ (i j : Fin 3), 
    (entries i 0 + entries i 1 + entries i 2 = 
     entries 0 j + entries 1 j + entries 2 j) ∧
    (entries 0 0 + entries 1 1 + entries 2 2 = 
     entries 0 2 + entries 1 1 + entries 2 0)

/-- The theorem stating that y must be 245 in the given modified magic square -/
theorem magic_square_y_value (square : ModifiedMagicSquare) 
  (h1 : square.entries 0 1 = 25)
  (h2 : square.entries 0 2 = 120)
  (h3 : square.entries 1 0 = 5) :
  square.entries 0 0 = 245 := by
  sorry

end NUMINAMATH_CALUDE_magic_square_y_value_l3191_319199


namespace NUMINAMATH_CALUDE_fractional_equation_positive_root_l3191_319103

theorem fractional_equation_positive_root (a : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ (1 - x) / (x - 2) = a / (2 - x) - 2) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_positive_root_l3191_319103


namespace NUMINAMATH_CALUDE_triangle_theorem_l3191_319175

noncomputable def triangle_proof (a b c : ℝ) (A B C : ℝ) : Prop :=
  -- Triangle ABC is acute
  0 < A ∧ A < Real.pi/2 ∧
  0 < B ∧ B < Real.pi/2 ∧
  0 < C ∧ C < Real.pi/2 ∧
  -- Sum of angles is π
  A + B + C = Real.pi ∧
  -- Given conditions
  a = 2*b * Real.sin A ∧
  a = 3 * Real.sqrt 3 ∧
  c = 5 →
  -- Conclusions
  B = Real.pi/6 ∧  -- 30° in radians
  (1/2 * a * c * Real.sin B = 15 * Real.sqrt 3 / 4) ∧
  b = Real.sqrt 7

theorem triangle_theorem :
  ∀ a b c A B C, triangle_proof a b c A B C :=
sorry

end NUMINAMATH_CALUDE_triangle_theorem_l3191_319175


namespace NUMINAMATH_CALUDE_least_n_satisfying_inequality_l3191_319123

theorem least_n_satisfying_inequality : 
  (∃ n : ℕ+, (1 : ℚ) / n - (1 : ℚ) / (n + 2) < (1 : ℚ) / 15) ∧ 
  (∀ m : ℕ+, (1 : ℚ) / m - (1 : ℚ) / (m + 2) < (1 : ℚ) / 15 → m ≥ 6) :=
by sorry

end NUMINAMATH_CALUDE_least_n_satisfying_inequality_l3191_319123


namespace NUMINAMATH_CALUDE_correct_operation_is_multiplication_by_three_l3191_319197

theorem correct_operation_is_multiplication_by_three (x : ℝ) : 
  (((3 * x - x / 5) / (3 * x)) * 100 = 93.33333333333333) → 
  (∃ (y : ℝ), y = 3 ∧ x * y = 3 * x) :=
by
  sorry

end NUMINAMATH_CALUDE_correct_operation_is_multiplication_by_three_l3191_319197


namespace NUMINAMATH_CALUDE_pablo_blocks_l3191_319126

theorem pablo_blocks (stack1 stack2 stack3 stack4 : ℕ) : 
  stack1 = 5 →
  stack3 = stack2 - 5 →
  stack4 = stack3 + 5 →
  stack1 + stack2 + stack3 + stack4 = 21 →
  stack2 - stack1 = 2 :=
by sorry

end NUMINAMATH_CALUDE_pablo_blocks_l3191_319126


namespace NUMINAMATH_CALUDE_volume_relationship_l3191_319189

/-- Given a right circular cone, cylinder, and sphere with specific properties, 
    prove the relationship between their volumes. -/
theorem volume_relationship (h r : ℝ) (A M C : ℝ) : 
  h > 0 → r > 0 →
  A = (1/3) * π * r^2 * h →
  M = π * r^2 * (2*h) →
  C = (4/3) * π * h^3 →
  A + M - C = π * h^3 := by
  sorry


end NUMINAMATH_CALUDE_volume_relationship_l3191_319189


namespace NUMINAMATH_CALUDE_winning_lines_8_cube_l3191_319171

/-- The number of straight lines containing 8 points in a 3D cubic grid --/
def winning_lines (n : ℕ) : ℕ :=
  ((n + 2)^3 - n^3) / 2

/-- Theorem: In an 8×8×8 cubic grid, the number of straight lines containing 8 points is 244 --/
theorem winning_lines_8_cube : winning_lines 8 = 244 := by
  sorry

end NUMINAMATH_CALUDE_winning_lines_8_cube_l3191_319171


namespace NUMINAMATH_CALUDE_consecutive_numbers_percentage_l3191_319116

theorem consecutive_numbers_percentage (a b c d e f g : ℤ) : 
  (a + b + c + d + e + f + g = 7 * 9) →
  (b = a + 1) →
  (c = b + 1) →
  (d = c + 1) →
  (e = d + 1) →
  (f = e + 1) →
  (g = f + 1) →
  (a : ℚ) / g * 100 = 50 := by
sorry

end NUMINAMATH_CALUDE_consecutive_numbers_percentage_l3191_319116


namespace NUMINAMATH_CALUDE_equal_projections_imply_equal_areas_l3191_319111

/-- Represents a parabola -/
structure Parabola where
  -- Add necessary fields to define a parabola

/-- Represents a chord of a parabola -/
structure Chord (p : Parabola) where
  -- Add necessary fields to define a chord

/-- Represents the projection of a chord on the directrix -/
def projection (p : Parabola) (c : Chord p) : ℝ :=
  sorry

/-- Represents the area of the segment cut off by a chord -/
def segmentArea (p : Parabola) (c : Chord p) : ℝ :=
  sorry

/-- Theorem: If two chords of a parabola have equal projections on the directrix,
    then the areas of the segments they cut off are equal -/
theorem equal_projections_imply_equal_areas (p : Parabola) (c1 c2 : Chord p) :
  projection p c1 = projection p c2 → segmentArea p c1 = segmentArea p c2 :=
by sorry

end NUMINAMATH_CALUDE_equal_projections_imply_equal_areas_l3191_319111


namespace NUMINAMATH_CALUDE_sqrt_five_power_l3191_319153

theorem sqrt_five_power : (Real.sqrt ((Real.sqrt 5) ^ 5)) ^ 6 = 5 ^ (15 / 2) := by sorry

end NUMINAMATH_CALUDE_sqrt_five_power_l3191_319153


namespace NUMINAMATH_CALUDE_students_taking_one_subject_l3191_319117

theorem students_taking_one_subject (both : ℕ) (science : ℕ) (only_history : ℕ) 
  (h1 : both = 15)
  (h2 : science = 30)
  (h3 : only_history = 18) :
  science - both + only_history = 33 := by
sorry

end NUMINAMATH_CALUDE_students_taking_one_subject_l3191_319117


namespace NUMINAMATH_CALUDE_opposite_of_negative_three_l3191_319108

theorem opposite_of_negative_three :
  ∃ y : ℤ, ((-3 : ℤ) + y = 0) ∧ y = 3 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_three_l3191_319108


namespace NUMINAMATH_CALUDE_f_2019_value_l3191_319163

/-- A function satisfying the given conditions -/
def special_function (f : ℝ → ℝ) : Prop :=
  f 1 = 1/4 ∧ ∀ x y : ℝ, 4 * f x * f y = f (x + y) + f (x - y)

/-- The main theorem stating that f(2019) = -1/2 for any function satisfying the conditions -/
theorem f_2019_value (f : ℝ → ℝ) (hf : special_function f) : f 2019 = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_f_2019_value_l3191_319163


namespace NUMINAMATH_CALUDE_triangle_angle_A_l3191_319166

theorem triangle_angle_A (a b : ℝ) (A B : ℝ) :
  a = Real.sqrt 2 →
  b = 2 →
  Real.sin B + Real.cos B = Real.sqrt 2 →
  0 < A →
  A < π / 4 →
  0 < B →
  B < π →
  Real.sin A = a / b * Real.sin B →
  A = π / 6 :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_A_l3191_319166


namespace NUMINAMATH_CALUDE_pierced_square_theorem_l3191_319172

/-- Represents a square pierced at n points and cut into triangles --/
structure PiercedSquare where
  n : ℕ  -- number of pierced points
  no_collinear_triples : True  -- represents the condition that no three points are collinear
  no_internal_piercings : True  -- represents the condition that there are no piercings inside triangles

/-- Calculates the number of triangles formed in a pierced square --/
def num_triangles (ps : PiercedSquare) : ℕ :=
  2 * (ps.n + 1)

/-- Calculates the number of cuts made in a pierced square --/
def num_cuts (ps : PiercedSquare) : ℕ :=
  (3 * num_triangles ps - 4) / 2

/-- Theorem stating the relationship between pierced points, triangles, and cuts --/
theorem pierced_square_theorem (ps : PiercedSquare) :
  (num_triangles ps = 2 * (ps.n + 1)) ∧
  (num_cuts ps = (3 * num_triangles ps - 4) / 2) := by
  sorry

end NUMINAMATH_CALUDE_pierced_square_theorem_l3191_319172


namespace NUMINAMATH_CALUDE_may_blue_yarns_l3191_319102

/-- The number of scarves May can knit using one yarn -/
def scarves_per_yarn : ℕ := 3

/-- The number of red yarns May bought -/
def red_yarns : ℕ := 2

/-- The number of yellow yarns May bought -/
def yellow_yarns : ℕ := 4

/-- The total number of scarves May will be able to make -/
def total_scarves : ℕ := 36

/-- The number of blue yarns May bought -/
def blue_yarns : ℕ := 6

theorem may_blue_yarns : 
  scarves_per_yarn * (red_yarns + yellow_yarns + blue_yarns) = total_scarves :=
by sorry

end NUMINAMATH_CALUDE_may_blue_yarns_l3191_319102


namespace NUMINAMATH_CALUDE_zero_discriminant_implies_ratio_l3191_319146

/-- Given a quadratic equation 3ax^2 + 6bx + 2c = 0 with zero discriminant,
    prove that b^2 = (2/3)ac -/
theorem zero_discriminant_implies_ratio (a b c : ℝ) :
  (6 * b)^2 - 4 * (3 * a) * (2 * c) = 0 →
  b^2 = (2/3) * a * c := by
  sorry

end NUMINAMATH_CALUDE_zero_discriminant_implies_ratio_l3191_319146


namespace NUMINAMATH_CALUDE_g_minus_g_is_zero_l3191_319139

def f : ℕ → ℕ
| 0 => 0
| (n + 1) => if n % 2 = 0 then 2 * f (n / 2) + 1 else 2 * f n

def g (n : ℕ) : ℕ := f (f n)

theorem g_minus_g_is_zero (n : ℕ) : g (n - g n) = 0 := by
  sorry

end NUMINAMATH_CALUDE_g_minus_g_is_zero_l3191_319139


namespace NUMINAMATH_CALUDE_sin_cos_range_l3191_319133

theorem sin_cos_range (x y : ℝ) (h : 2 * (Real.sin x)^2 + (Real.cos y)^2 = 1) :
  ∃ (z : ℝ), (Real.sin x)^2 + (Real.cos y)^2 = z ∧ 1/2 ≤ z ∧ z ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_sin_cos_range_l3191_319133


namespace NUMINAMATH_CALUDE_alexis_shoe_cost_l3191_319130

/-- Given Alexis' shopping scenario, prove the cost of shoes --/
theorem alexis_shoe_cost (budget : ℕ) (shirt_cost pants_cost coat_cost socks_cost belt_cost money_left : ℕ) 
  (h1 : budget = 200)
  (h2 : shirt_cost = 30)
  (h3 : pants_cost = 46)
  (h4 : coat_cost = 38)
  (h5 : socks_cost = 11)
  (h6 : belt_cost = 18)
  (h7 : money_left = 16) :
  budget - (shirt_cost + pants_cost + coat_cost + socks_cost + belt_cost + money_left) = 41 := by
  sorry

end NUMINAMATH_CALUDE_alexis_shoe_cost_l3191_319130


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3191_319173

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of specific terms in an arithmetic sequence -/
theorem arithmetic_sequence_sum 
  (a : ℕ → ℝ) 
  (h : arithmetic_sequence a) 
  (h_sum : a 3 + a 7 = 37) : 
  a 2 + a 4 + a 6 + a 8 = 74 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3191_319173


namespace NUMINAMATH_CALUDE_return_flight_time_l3191_319137

/-- Represents the flight scenario between two cities -/
structure FlightScenario where
  d : ℝ  -- distance between cities
  p : ℝ  -- plane's speed in still air
  w : ℝ  -- wind speed
  against_wind_time : ℝ -- time for flight against wind
  still_air_time : ℝ  -- time for flight in still air

/-- The conditions of the flight scenario -/
def flight_conditions (scenario : FlightScenario) : Prop :=
  scenario.against_wind_time = 120 ∧
  scenario.d = scenario.against_wind_time * (scenario.p - scenario.w) ∧
  scenario.d / (scenario.p + scenario.w) = scenario.still_air_time - 10

/-- The theorem stating that under the given conditions, the return flight time is 110 minutes -/
theorem return_flight_time (scenario : FlightScenario) 
  (h : flight_conditions scenario) : 
  scenario.d / (scenario.p + scenario.w) = 110 := by
  sorry


end NUMINAMATH_CALUDE_return_flight_time_l3191_319137


namespace NUMINAMATH_CALUDE_max_y_over_x_for_complex_number_l3191_319160

theorem max_y_over_x_for_complex_number (x y : ℝ) :
  let z : ℂ := (x - 2) + y * I
  (Complex.abs z)^2 = 3 →
  ∃ (k : ℝ), k^2 = 3 ∧ ∀ (t : ℝ), (y / x)^2 ≤ k^2 :=
by sorry

end NUMINAMATH_CALUDE_max_y_over_x_for_complex_number_l3191_319160


namespace NUMINAMATH_CALUDE_table_rotation_l3191_319112

theorem table_rotation (table_width : ℝ) (table_length : ℝ) : 
  table_width = 8 ∧ table_length = 12 →
  ∃ (S : ℕ), (S : ℝ) ≥ (table_width^2 + table_length^2).sqrt ∧
  ∀ (T : ℕ), (T : ℝ) ≥ (table_width^2 + table_length^2).sqrt → S ≤ T →
  S = 15 :=
by sorry

end NUMINAMATH_CALUDE_table_rotation_l3191_319112


namespace NUMINAMATH_CALUDE_quadratic_root_sum_power_l3191_319183

/-- Given a quadratic equation x^2 + mx + 3 = 0 with roots 1 and n, prove (m + n)^2023 = -1 -/
theorem quadratic_root_sum_power (m n : ℝ) : 
  (1 : ℝ) ^ 2 + m * 1 + 3 = 0 → 
  n ^ 2 + m * n + 3 = 0 → 
  (m + n) ^ 2023 = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_sum_power_l3191_319183


namespace NUMINAMATH_CALUDE_grocery_store_soda_l3191_319110

theorem grocery_store_soda (regular_soda : ℕ) (apples : ℕ) (total_bottles : ℕ) 
  (h1 : regular_soda = 72)
  (h2 : apples = 78)
  (h3 : total_bottles = apples + 26) :
  total_bottles - regular_soda = 32 := by
  sorry

end NUMINAMATH_CALUDE_grocery_store_soda_l3191_319110


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_neg_one_and_five_l3191_319120

theorem arithmetic_mean_of_neg_one_and_five (x y : ℝ) : 
  x = -1 → y = 5 → (x + y) / 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_neg_one_and_five_l3191_319120


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l3191_319190

theorem smallest_n_congruence : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), m > 0 → (531 * m) % 24 = (1067 * m) % 24 → m ≥ n) ∧
  (531 * n) % 24 = (1067 * n) % 24 ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l3191_319190


namespace NUMINAMATH_CALUDE_installation_problem_l3191_319167

theorem installation_problem (x₁ x₂ x₃ k : ℕ) :
  x₁ + x₂ + x₃ ≤ 200 ∧
  x₂ = 4 * x₁ ∧
  x₃ = k * x₁ ∧
  5 * x₃ = x₂ + 99 →
  x₁ = 9 ∧ x₂ = 36 ∧ x₃ = 27 := by
sorry

end NUMINAMATH_CALUDE_installation_problem_l3191_319167


namespace NUMINAMATH_CALUDE_odd_function_property_l3191_319135

-- Define an odd function f: ℝ → ℝ
def isOddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Theorem statement
theorem odd_function_property (f : ℝ → ℝ) (h_odd : isOddFunction f) (h_f_neg_three : f (-3) = 2) :
  f 3 + f 0 = -2 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_property_l3191_319135


namespace NUMINAMATH_CALUDE_slope_intercept_sum_l3191_319114

/-- Given two points C and D on a Cartesian plane, this theorem proves that
    the sum of the slope and y-intercept of the line passing through these points is 1. -/
theorem slope_intercept_sum (C D : ℝ × ℝ) : C = (2, 3) → D = (5, 9) →
  let m := (D.2 - C.2) / (D.1 - C.1)
  let b := C.2 - m * C.1
  m + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_slope_intercept_sum_l3191_319114


namespace NUMINAMATH_CALUDE_log_equation_solution_l3191_319138

theorem log_equation_solution (x : ℝ) (h : x > 0) :
  Real.log x / Real.log 8 - 3 * (Real.log x / Real.log 2) = 6 →
  x = (1 : ℝ) / 2^(9/4) :=
by sorry

end NUMINAMATH_CALUDE_log_equation_solution_l3191_319138


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3191_319145

theorem quadratic_inequality_solution (x : ℝ) : x^2 - 3*x + 2 < 0 ↔ 1 < x ∧ x < 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3191_319145


namespace NUMINAMATH_CALUDE_part_one_part_two_l3191_319161

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 3*a|

-- Part I
theorem part_one : 
  {x : ℝ | f 1 x > 5 - |2*x - 1|} = {x : ℝ | x < -1/3 ∨ x > 3} := by sorry

-- Part II
theorem part_two (a : ℝ) : 
  (∃ x₀ : ℝ, f a x₀ + x₀ < 6) → a < 2 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3191_319161


namespace NUMINAMATH_CALUDE_sector_angle_when_length_equals_area_l3191_319185

/-- Theorem: For a circular sector with arc length and area both equal to 6,
    the central angle in radians is 3. -/
theorem sector_angle_when_length_equals_area (r : ℝ) (θ : ℝ) : 
  r * θ = 6 → -- arc length = r * θ = 6
  (1/2) * r^2 * θ = 6 → -- area = (1/2) * r^2 * θ = 6
  θ = 3 := by sorry

end NUMINAMATH_CALUDE_sector_angle_when_length_equals_area_l3191_319185


namespace NUMINAMATH_CALUDE_pyramid_base_theorem_l3191_319129

def isPyramidBase (a b c d e : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e

def pyramidTop (a b c d e : ℕ) : ℕ :=
  a * b^4 * c^6 * d^4 * e

theorem pyramid_base_theorem (a b c d e : ℕ) :
  isPyramidBase a b c d e ∧ pyramidTop a b c d e = 140026320 →
  ((a = 1 ∧ b = 2 ∧ c = 3 ∧ d = 7 ∧ e = 5) ∨
   (a = 1 ∧ b = 7 ∧ c = 3 ∧ d = 2 ∧ e = 5) ∨
   (a = 5 ∧ b = 2 ∧ c = 3 ∧ d = 7 ∧ e = 1) ∨
   (a = 5 ∧ b = 7 ∧ c = 3 ∧ d = 2 ∧ e = 1)) :=
by sorry

end NUMINAMATH_CALUDE_pyramid_base_theorem_l3191_319129


namespace NUMINAMATH_CALUDE_cookie_distribution_l3191_319100

/-- Represents the number of cookie boxes in Sonny's distribution problem -/
structure CookieBoxes where
  total : ℕ
  tobrother : ℕ
  tocousin : ℕ
  kept : ℕ
  tosister : ℕ

/-- Theorem stating the relationship between the number of cookie boxes -/
theorem cookie_distribution (c : CookieBoxes) 
  (h1 : c.total = 45)
  (h2 : c.tobrother = 12)
  (h3 : c.tocousin = 7)
  (h4 : c.kept = 17) :
  c.tosister = c.total - (c.tobrother + c.tocousin + c.kept) :=
by sorry

end NUMINAMATH_CALUDE_cookie_distribution_l3191_319100


namespace NUMINAMATH_CALUDE_asterisk_replacement_l3191_319184

theorem asterisk_replacement : ∃ x : ℝ, (x / 18) * (x / 72) = 1 ∧ x = 36 := by
  sorry

end NUMINAMATH_CALUDE_asterisk_replacement_l3191_319184


namespace NUMINAMATH_CALUDE_students_without_A_l3191_319109

theorem students_without_A (total : ℕ) (history_A : ℕ) (math_A : ℕ) (both_A : ℕ) :
  total = 30 →
  history_A = 7 →
  math_A = 13 →
  both_A = 4 →
  total - ((history_A + math_A) - both_A) = 14 :=
by sorry

end NUMINAMATH_CALUDE_students_without_A_l3191_319109


namespace NUMINAMATH_CALUDE_floor_sqrt_80_l3191_319106

theorem floor_sqrt_80 : ⌊Real.sqrt 80⌋ = 8 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_80_l3191_319106


namespace NUMINAMATH_CALUDE_line_equation_through_point_with_slope_angle_l3191_319155

/-- The equation of a line passing through (2, 3) with a slope angle of 135° -/
theorem line_equation_through_point_with_slope_angle (x y : ℝ) :
  (x + y - 5 = 0) ↔ 
  (∃ (m : ℝ), m = Real.tan (135 * π / 180) ∧ y - 3 = m * (x - 2)) := by
  sorry

end NUMINAMATH_CALUDE_line_equation_through_point_with_slope_angle_l3191_319155


namespace NUMINAMATH_CALUDE_estimate_fish_population_l3191_319158

/-- Estimate the number of fish in a pond using the mark and recapture method. -/
theorem estimate_fish_population (initial_marked : ℕ) (recapture_total : ℕ) (recapture_marked : ℕ) 
  (h1 : initial_marked = 20)
  (h2 : recapture_total = 40)
  (h3 : recapture_marked = 2) :
  (initial_marked * recapture_total) / recapture_marked = 400 := by
  sorry

#check estimate_fish_population

end NUMINAMATH_CALUDE_estimate_fish_population_l3191_319158


namespace NUMINAMATH_CALUDE_banana_mango_equivalence_l3191_319124

/-- Represents the cost relationship between fruits -/
structure FruitCost where
  banana : ℝ
  pear : ℝ
  mango : ℝ

/-- The given cost relationships -/
def cost_relation (c : FruitCost) : Prop :=
  4 * c.banana = 3 * c.pear ∧ 8 * c.pear = 5 * c.mango

/-- The theorem to prove -/
theorem banana_mango_equivalence (c : FruitCost) (h : cost_relation c) :
  20 * c.banana = 9.375 * c.mango :=
sorry

end NUMINAMATH_CALUDE_banana_mango_equivalence_l3191_319124


namespace NUMINAMATH_CALUDE_new_supervisor_salary_l3191_319141

theorem new_supervisor_salary 
  (num_workers : ℕ) 
  (num_supervisors : ℕ) 
  (initial_avg_salary : ℚ) 
  (retiring_supervisor_salary : ℚ) 
  (new_avg_salary : ℚ) 
  (h1 : num_workers = 12) 
  (h2 : num_supervisors = 3) 
  (h3 : initial_avg_salary = 650) 
  (h4 : retiring_supervisor_salary = 1200) 
  (h5 : new_avg_salary = 675) : 
  (num_workers + num_supervisors) * new_avg_salary - 
  ((num_workers + num_supervisors) * initial_avg_salary - retiring_supervisor_salary) = 1575 :=
by sorry

end NUMINAMATH_CALUDE_new_supervisor_salary_l3191_319141


namespace NUMINAMATH_CALUDE_baso4_percentage_yield_is_90_percent_l3191_319140

-- Define the molar quantities
def NaOH_moles : ℚ := 3
def H2SO4_moles : ℚ := 2
def BaCl2_moles : ℚ := 1
def BaSO4_actual_yield : ℚ := 9/10

-- Define the reaction stoichiometry
def NaOH_to_Na2SO4_ratio : ℚ := 2
def H2SO4_to_Na2SO4_ratio : ℚ := 1
def Na2SO4_to_BaSO4_ratio : ℚ := 1
def BaCl2_to_BaSO4_ratio : ℚ := 1

-- Define the theoretical yield calculation
def theoretical_yield (limiting_reactant_moles ratio : ℚ) : ℚ :=
  limiting_reactant_moles / ratio

-- Define the percentage yield calculation
def percentage_yield (actual_yield theoretical_yield : ℚ) : ℚ :=
  actual_yield / theoretical_yield * 100

-- Theorem to prove
theorem baso4_percentage_yield_is_90_percent :
  let na2so4_yield_from_naoh := theoretical_yield NaOH_moles NaOH_to_Na2SO4_ratio
  let na2so4_yield_from_h2so4 := theoretical_yield H2SO4_moles H2SO4_to_Na2SO4_ratio
  let na2so4_actual_yield := min na2so4_yield_from_naoh na2so4_yield_from_h2so4
  let baso4_theoretical_yield := min na2so4_actual_yield BaCl2_moles
  percentage_yield BaSO4_actual_yield baso4_theoretical_yield = 90 :=
by sorry

end NUMINAMATH_CALUDE_baso4_percentage_yield_is_90_percent_l3191_319140


namespace NUMINAMATH_CALUDE_tim_has_twelve_nickels_l3191_319115

/-- Represents the number of coins Tim has -/
structure TimsCoins where
  quarters : ℕ
  nickels : ℕ
  pennies : ℕ

/-- Calculates the total number of nickels Tim has after receiving coins from his dad -/
def total_nickels (initial : TimsCoins) (from_dad : TimsCoins) : ℕ :=
  initial.nickels + from_dad.nickels

/-- Theorem stating that Tim has 12 nickels after receiving coins from his dad -/
theorem tim_has_twelve_nickels :
  let initial := TimsCoins.mk 7 9 0
  let from_dad := TimsCoins.mk 0 3 5
  total_nickels initial from_dad = 12 := by
  sorry


end NUMINAMATH_CALUDE_tim_has_twelve_nickels_l3191_319115


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l3191_319191

theorem partial_fraction_decomposition :
  ∃ (C D : ℚ), C = 32/9 ∧ D = 13/9 ∧
  ∀ x : ℚ, x ≠ 7 ∧ x ≠ -2 →
    (5*x - 3) / (x^2 - 5*x - 14) = C / (x - 7) + D / (x + 2) :=
by sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l3191_319191


namespace NUMINAMATH_CALUDE_not_necessarily_right_triangle_l3191_319122

/-- A triangle with side lengths proportional to 3:4:6 is not necessarily a right triangle -/
theorem not_necessarily_right_triangle (a b c : ℝ) (h : a / b = 3 / 4 ∧ b / c = 4 / 6) :
  ¬ (a^2 + b^2 = c^2) :=
sorry

end NUMINAMATH_CALUDE_not_necessarily_right_triangle_l3191_319122


namespace NUMINAMATH_CALUDE_total_candle_weight_l3191_319148

/-- Represents the composition of a candle in ounces -/
structure CandleComposition where
  beeswax : ℝ
  coconut_oil : ℝ
  essential_oils : ℝ

/-- Calculates the total weight of a candle given its composition -/
def candle_weight (c : CandleComposition) : ℝ :=
  c.beeswax + c.coconut_oil + c.essential_oils

/-- Defines the composition of a small candle -/
def small_candle : CandleComposition :=
  { beeswax := 4, coconut_oil := 2, essential_oils := 0.5 }

/-- Defines the composition of a medium candle -/
def medium_candle : CandleComposition :=
  { beeswax := 8, coconut_oil := 1, essential_oils := 1 }

/-- Defines the composition of a large candle -/
def large_candle : CandleComposition :=
  { beeswax := 16, coconut_oil := 3, essential_oils := 2 }

/-- The number of small candles made -/
def num_small_candles : ℕ := 4

/-- The number of medium candles made -/
def num_medium_candles : ℕ := 3

/-- The number of large candles made -/
def num_large_candles : ℕ := 2

/-- Theorem stating that the total weight of all candles is 98 ounces -/
theorem total_candle_weight :
  (num_small_candles : ℝ) * candle_weight small_candle +
  (num_medium_candles : ℝ) * candle_weight medium_candle +
  (num_large_candles : ℝ) * candle_weight large_candle = 98 := by
  sorry

end NUMINAMATH_CALUDE_total_candle_weight_l3191_319148


namespace NUMINAMATH_CALUDE_fraction_equivalence_l3191_319169

theorem fraction_equivalence : 8 / (4 * 25) = 0.8 / (0.4 * 25) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equivalence_l3191_319169


namespace NUMINAMATH_CALUDE_expression_simplification_l3191_319121

theorem expression_simplification (a b : ℝ) 
  (h1 : a + b = 4) 
  (h2 : a - b = 2) : 
  (a^2 - 6*a*b + 9*b^2) / (a^2 - 2*a*b) / 
  ((5*b^2) / (a - 2*b) - a - 2*b) - 1/a = -1/3 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l3191_319121


namespace NUMINAMATH_CALUDE_fraction_subtraction_simplification_l3191_319131

theorem fraction_subtraction_simplification (d : ℝ) :
  (5 + 4 * d) / 9 - 3 = (4 * d - 22) / 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_simplification_l3191_319131


namespace NUMINAMATH_CALUDE_unusual_bicycle_spokes_l3191_319151

/-- A bicycle with an unusual spoke configuration. -/
structure Bicycle where
  front_spokes : ℕ
  back_spokes : ℕ

/-- The total number of spokes on a bicycle. -/
def total_spokes (b : Bicycle) : ℕ := b.front_spokes + b.back_spokes

/-- Theorem: The total number of spokes on the unusual bicycle is 60. -/
theorem unusual_bicycle_spokes :
  ∃ (b : Bicycle), b.front_spokes = 20 ∧ b.back_spokes = 2 * b.front_spokes ∧ total_spokes b = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_unusual_bicycle_spokes_l3191_319151


namespace NUMINAMATH_CALUDE_flea_treatment_l3191_319134

theorem flea_treatment (initial_fleas : ℕ) : 
  (initial_fleas / 2 / 2 / 2 / 2 = 14) → (initial_fleas - 14 = 210) := by
  sorry

end NUMINAMATH_CALUDE_flea_treatment_l3191_319134


namespace NUMINAMATH_CALUDE_slope_plus_intercept_equals_two_thirds_l3191_319159

-- Define a line passing through two points
def line_through_points (x₁ y₁ x₂ y₂ : ℚ) : ℚ × ℚ :=
  let m := (y₂ - y₁) / (x₂ - x₁)
  let b := y₁ - m * x₁
  (m, b)

-- Theorem statement
theorem slope_plus_intercept_equals_two_thirds :
  let (m, b) := line_through_points 2 (-1) (-1) 4
  m + b = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_slope_plus_intercept_equals_two_thirds_l3191_319159


namespace NUMINAMATH_CALUDE_unique_solution_sqrt_two_equation_l3191_319196

theorem unique_solution_sqrt_two_equation (m n : ℤ) :
  (5 + 3 * Real.sqrt 2) ^ m = (3 + 5 * Real.sqrt 2) ^ n ↔ m = 0 ∧ n = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_sqrt_two_equation_l3191_319196


namespace NUMINAMATH_CALUDE_no_adjacent_standing_probability_l3191_319194

/-- Represents a person's standing state -/
inductive State
  | Standing
  | Seated

/-- Represents the circular arrangement of people -/
def Arrangement := Vector State 10

/-- Checks if two adjacent people are standing -/
def hasAdjacentStanding (arr : Arrangement) : Bool :=
  sorry

/-- Checks if an arrangement is valid according to the problem rules -/
def isValidArrangement (arr : Arrangement) : Bool :=
  sorry

/-- The total number of possible arrangements -/
def totalArrangements : Nat :=
  2^8

/-- The number of valid arrangements where no two adjacent people stand -/
def validArrangements : Nat :=
  sorry

theorem no_adjacent_standing_probability :
  (validArrangements : ℚ) / totalArrangements = 1 / 64 :=
sorry

end NUMINAMATH_CALUDE_no_adjacent_standing_probability_l3191_319194


namespace NUMINAMATH_CALUDE_cone_base_circumference_l3191_319168

/-- Theorem: For a right circular cone with volume 24π cubic centimeters and height 6 cm, 
    the circumference of its base is 4√3π cm. -/
theorem cone_base_circumference (V : ℝ) (h : ℝ) (r : ℝ) :
  V = 24 * Real.pi ∧ h = 6 ∧ V = (1/3) * Real.pi * r^2 * h → 
  2 * Real.pi * r = 4 * Real.sqrt 3 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_cone_base_circumference_l3191_319168


namespace NUMINAMATH_CALUDE_pushup_problem_l3191_319198

theorem pushup_problem (x : ℕ) (h : x = 51) : 
  let zachary := x
  let melanie := 2 * zachary - 7
  let david := zachary + 22
  let karen := (zachary + melanie + david) / 3 - 5
  let john := david - 4
  john + melanie + karen = 232 := by
sorry

end NUMINAMATH_CALUDE_pushup_problem_l3191_319198


namespace NUMINAMATH_CALUDE_intersection_distance_l3191_319152

/-- The distance between the intersection points of two curves in polar coordinates -/
theorem intersection_distance (θ : Real) : 
  ∃ (A B : ℝ × ℝ), 
    (∀ (ρ : ℝ), ρ * Real.sin (θ + π/4) = 1 → (ρ * Real.cos θ, ρ * Real.sin θ) = A ∨ (ρ * Real.cos θ, ρ * Real.sin θ) = B) ∧
    (∀ (ρ : ℝ), ρ = Real.sqrt 2 → (ρ * Real.cos θ, ρ * Real.sin θ) = A ∨ (ρ * Real.cos θ, ρ * Real.sin θ) = B) ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 :=
sorry

end NUMINAMATH_CALUDE_intersection_distance_l3191_319152


namespace NUMINAMATH_CALUDE_product_of_22nd_and_23rd_multiples_l3191_319162

/-- The sequence of multiples of 3 greater than 0 and less than 100 -/
def multiples_of_3 : List Nat :=
  (List.range 33).map (fun n => (n + 1) * 3)

/-- The 22nd element in the sequence -/
def element_22 : Nat := multiples_of_3[21]

/-- The 23rd element in the sequence -/
def element_23 : Nat := multiples_of_3[22]

theorem product_of_22nd_and_23rd_multiples :
  element_22 * element_23 = 4554 :=
by sorry

end NUMINAMATH_CALUDE_product_of_22nd_and_23rd_multiples_l3191_319162


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3191_319142

theorem arithmetic_sequence_problem (a b c d e : ℕ) :
  a < 10 ∧
  b = 12 ∧
  e = 33 ∧
  a < b ∧ b < c ∧ c < d ∧ d < e ∧
  e < 100 ∧
  b - a = c - b ∧
  c - b = d - c ∧
  d - c = e - d →
  a = 5 ∧ b = 12 ∧ c = 19 ∧ d = 26 ∧ e = 33 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3191_319142


namespace NUMINAMATH_CALUDE_binomial_sum_l3191_319182

theorem binomial_sum : Nat.choose 12 4 + Nat.choose 10 3 = 615 := by
  sorry

end NUMINAMATH_CALUDE_binomial_sum_l3191_319182


namespace NUMINAMATH_CALUDE_fixed_point_exponential_l3191_319193

/-- The function f(x) = a^(x-1) + 4 always passes through the point (1, 5) for any a > 0 and a ≠ 1 -/
theorem fixed_point_exponential (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x-1) + 4
  f 1 = 5 := by sorry

end NUMINAMATH_CALUDE_fixed_point_exponential_l3191_319193


namespace NUMINAMATH_CALUDE_point_relationship_l3191_319136

/-- A line in the 2D plane represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def Point.liesOn (p : Point) (l : Line) : Prop :=
  p.y = l.slope * p.x + l.intercept

theorem point_relationship (m n : ℝ) : 
  let l : Line := { slope := -2, intercept := 1 }
  let A : Point := { x := -1, y := m }
  let B : Point := { x := 3, y := n }
  A.liesOn l ∧ B.liesOn l → m > n := by
  sorry

end NUMINAMATH_CALUDE_point_relationship_l3191_319136


namespace NUMINAMATH_CALUDE_even_function_inequality_l3191_319177

/-- A function is even if f(x) = f(-x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

/-- A function is monotonically increasing on a set if f(x) ≤ f(y) whenever x ≤ y in that set -/
def MonoIncOn (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ x y, x ∈ s → y ∈ s → x ≤ y → f x ≤ f y

/-- The theorem statement -/
theorem even_function_inequality (f : ℝ → ℝ) (a : ℝ) 
    (h_even : IsEven f)
    (h_mono : MonoIncOn f (Set.Ici 0))
    (h_ineq : ∀ x ∈ Set.Icc (1/2) 1, f (a*x + 1) - f (x - 2) ≤ 0) :
  a ∈ Set.Icc (-2) 0 := by
  sorry

end NUMINAMATH_CALUDE_even_function_inequality_l3191_319177


namespace NUMINAMATH_CALUDE_principal_is_900_l3191_319187

/-- Proves that given the conditions of the problem, the principal must be $900 -/
theorem principal_is_900 (P R : ℝ) : 
  (P * (R + 3) * 3) / 100 = (P * R * 3) / 100 + 81 → P = 900 := by
  sorry

end NUMINAMATH_CALUDE_principal_is_900_l3191_319187


namespace NUMINAMATH_CALUDE_isosceles_not_equilateral_l3191_319170

-- Define an isosceles triangle
def IsIsosceles (a b c : ℝ) : Prop :=
  (a = b ∨ b = c ∨ a = c) ∧ (a > 0 ∧ b > 0 ∧ c > 0)

-- Define an equilateral triangle
def IsEquilateral (a b c : ℝ) : Prop :=
  a = b ∧ b = c ∧ a > 0

-- Theorem: There exists an isosceles triangle that is not equilateral
theorem isosceles_not_equilateral : ∃ a b c : ℝ, IsIsosceles a b c ∧ ¬IsEquilateral a b c := by
  sorry


end NUMINAMATH_CALUDE_isosceles_not_equilateral_l3191_319170


namespace NUMINAMATH_CALUDE_evaluate_expression_l3191_319107

theorem evaluate_expression : -(18 / 3 * 8 - 72 + 4^2 * 3) = -24 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3191_319107


namespace NUMINAMATH_CALUDE_collinearity_proof_collinear_vectors_l3191_319186

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

noncomputable section

def are_collinear (a b c : V) : Prop := ∃ (t : ℝ), c - a = t • (b - a)

theorem collinearity_proof 
  (e₁ e₂ A B C D : V) 
  (h₁ : e₁ ≠ 0) 
  (h₂ : e₂ ≠ 0) 
  (h₃ : ¬ ∃ (t : ℝ), e₁ = t • e₂) 
  (h₄ : B - A = e₁ + e₂) 
  (h₅ : C - B = 2 • e₁ + 8 • e₂) 
  (h₆ : D - C = 3 • (e₁ - e₂)) : 
  are_collinear A B D :=
sorry

theorem collinear_vectors 
  (e₁ e₂ : V) 
  (h₁ : e₁ ≠ 0) 
  (h₂ : e₂ ≠ 0) 
  (h₃ : ¬ ∃ (t : ℝ), e₁ = t • e₂) :
  ∀ (k : ℝ), (∃ (t : ℝ), k • e₁ + e₂ = t • (e₁ + k • e₂)) ↔ (k = 1 ∨ k = -1) :=
sorry

end

end NUMINAMATH_CALUDE_collinearity_proof_collinear_vectors_l3191_319186


namespace NUMINAMATH_CALUDE_housewife_money_l3191_319192

theorem housewife_money (initial_money : ℚ) : 
  (1 - 2/3) * initial_money = 50 → initial_money = 150 := by
  sorry

end NUMINAMATH_CALUDE_housewife_money_l3191_319192


namespace NUMINAMATH_CALUDE_factorization_problem1_l3191_319125

theorem factorization_problem1 (a m : ℝ) : 2 * a * m^2 - 8 * a = 2 * a * (m + 2) * (m - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_problem1_l3191_319125


namespace NUMINAMATH_CALUDE_base_neg_two_2019_has_six_nonzero_digits_l3191_319132

/-- Represents a number in base -2 as a list of binary digits -/
def BaseNegTwo := List Bool

/-- Converts a natural number to its base -2 representation -/
def toBaseNegTwo (n : ℕ) : BaseNegTwo :=
  sorry

/-- Counts the number of non-zero digits in a base -2 representation -/
def countNonZeroDigits (b : BaseNegTwo) : ℕ :=
  sorry

/-- Theorem: 2019 in base -2 has exactly 6 non-zero digits -/
theorem base_neg_two_2019_has_six_nonzero_digits :
  countNonZeroDigits (toBaseNegTwo 2019) = 6 := by
  sorry

end NUMINAMATH_CALUDE_base_neg_two_2019_has_six_nonzero_digits_l3191_319132


namespace NUMINAMATH_CALUDE_smallest_valid_n_l3191_319164

def is_valid_pair (m n x : ℕ+) : Prop :=
  m = 60 ∧ 
  Nat.gcd m n = x + 5 ∧ 
  Nat.lcm m n = x * (x + 5)

theorem smallest_valid_n : 
  ∃ (x : ℕ+), is_valid_pair 60 100 x ∧ 
  ∀ (y : ℕ+) (n : ℕ+), y < x → ¬ is_valid_pair 60 n y :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_n_l3191_319164


namespace NUMINAMATH_CALUDE_inscribed_circle_theorem_l3191_319119

theorem inscribed_circle_theorem (PQ PR : ℝ) (h_PQ : PQ = 6) (h_PR : PR = 8) :
  let QR := Real.sqrt (PQ^2 + PR^2)
  let s := (PQ + PR + QR) / 2
  let A := PQ * PR / 2
  let r := A / s
  let x := QR - 2*r
  x = 6 := by sorry

end NUMINAMATH_CALUDE_inscribed_circle_theorem_l3191_319119
