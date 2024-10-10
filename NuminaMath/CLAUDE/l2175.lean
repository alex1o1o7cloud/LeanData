import Mathlib

namespace power_inequality_l2175_217561

theorem power_inequality (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : b < 2) : a^b < b^a := by
  sorry

end power_inequality_l2175_217561


namespace kennedy_house_size_l2175_217592

theorem kennedy_house_size (benedict_house_size : ℕ) (kennedy_house_size : ℕ) : 
  benedict_house_size = 2350 →
  kennedy_house_size = 4 * benedict_house_size + 600 →
  kennedy_house_size = 10000 := by
sorry

end kennedy_house_size_l2175_217592


namespace board_coverage_five_by_five_uncoverable_four_by_four_removed_uncoverable_four_by_five_coverable_six_by_three_coverable_l2175_217589

/-- Represents a checkerboard -/
structure Checkerboard where
  rows : Nat
  cols : Nat
  removed_squares : Nat

/-- Checks if a board can be completely covered by non-overlapping dominoes -/
def can_be_covered (board : Checkerboard) : Prop :=
  (board.rows * board.cols - board.removed_squares) % 2 = 0

/-- Theorem: A board can be covered iff the number of squares is even -/
theorem board_coverage (board : Checkerboard) :
  can_be_covered board ↔ (board.rows * board.cols - board.removed_squares) % 2 = 0 := by
  sorry

/-- 5x5 board cannot be covered -/
theorem five_by_five_uncoverable :
  ¬(can_be_covered { rows := 5, cols := 5, removed_squares := 0 }) := by
  sorry

/-- 4x4 board with one square removed cannot be covered -/
theorem four_by_four_removed_uncoverable :
  ¬(can_be_covered { rows := 4, cols := 4, removed_squares := 1 }) := by
  sorry

/-- 4x5 board can be covered -/
theorem four_by_five_coverable :
  can_be_covered { rows := 4, cols := 5, removed_squares := 0 } := by
  sorry

/-- 6x3 board can be covered -/
theorem six_by_three_coverable :
  can_be_covered { rows := 6, cols := 3, removed_squares := 0 } := by
  sorry

end board_coverage_five_by_five_uncoverable_four_by_four_removed_uncoverable_four_by_five_coverable_six_by_three_coverable_l2175_217589


namespace square_difference_l2175_217558

theorem square_difference (m : ℕ) : (m + 1)^2 - m^2 = 2*m + 1 := by
  sorry

end square_difference_l2175_217558


namespace min_distance_point_l2175_217504

def A : ℝ × ℝ := (1, -1)
def B : ℝ × ℝ := (2, 2)

def distance_squared (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

def sum_of_distances (p : ℝ × ℝ) : ℝ :=
  distance_squared p A + distance_squared p B

def is_on_line (p : ℝ × ℝ) : Prop :=
  p.1 = p.2

theorem min_distance_point :
  ∃ (p : ℝ × ℝ), is_on_line p ∧
    ∀ (q : ℝ × ℝ), is_on_line q → sum_of_distances p ≤ sum_of_distances q :=
by sorry

end min_distance_point_l2175_217504


namespace job_pay_difference_l2175_217507

/-- Proves that the difference between two job pays is $375 given the total pay and the pay of the first job. -/
theorem job_pay_difference (total_pay first_job_pay : ℕ) 
  (h1 : total_pay = 3875)
  (h2 : first_job_pay = 2125) :
  first_job_pay - (total_pay - first_job_pay) = 375 := by
  sorry

end job_pay_difference_l2175_217507


namespace cricket_matches_played_l2175_217500

/-- Represents a cricket player's statistics -/
structure CricketPlayer where
  matches_played : ℕ
  total_runs : ℕ

/-- Calculate the batting average of a player -/
def batting_average (player : CricketPlayer) : ℚ :=
  player.total_runs / player.matches_played

theorem cricket_matches_played 
  (rahul ankit : CricketPlayer)
  (h1 : batting_average rahul = 46)
  (h2 : batting_average ankit = 52)
  (h3 : batting_average {matches_played := rahul.matches_played + 1, 
                         total_runs := rahul.total_runs + 78} = 54)
  (h4 : ∃ x : ℕ, 
        batting_average {matches_played := rahul.matches_played + 1, 
                         total_runs := rahul.total_runs + 78} = 54 ∧
        batting_average {matches_played := ankit.matches_played + 1, 
                         total_runs := ankit.total_runs + x} = 54) :
  rahul.matches_played = 3 ∧ ankit.matches_played = 3 := by
sorry

end cricket_matches_played_l2175_217500


namespace divisibility_implies_sum_divisibility_l2175_217550

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n
  else (n % 10) + sum_of_digits (n / 10)

theorem divisibility_implies_sum_divisibility (n : ℕ) 
  (h1 : n < 10000) (h2 : n % 99 = 0) : 
  (sum_of_digits n) % 18 = 0 := by
sorry

end divisibility_implies_sum_divisibility_l2175_217550


namespace grid_sequence_problem_l2175_217515

theorem grid_sequence_problem (row : List ℤ) (d_col : ℤ) (last_col : ℤ) (M : ℤ) :
  row = [15, 11, 7] →
  d_col = -5 →
  last_col = -4 →
  M = last_col - 4 * d_col →
  M = 6 := by
  sorry

end grid_sequence_problem_l2175_217515


namespace cersei_cousin_fraction_l2175_217541

def initial_candies : ℕ := 50
def given_to_siblings : ℕ := 5 + 5
def eaten_by_cersei : ℕ := 12
def left_after_eating : ℕ := 18

theorem cersei_cousin_fraction :
  let remaining_after_siblings := initial_candies - given_to_siblings
  let given_to_cousin := remaining_after_siblings - (left_after_eating + eaten_by_cersei)
  (given_to_cousin : ℚ) / remaining_after_siblings = 1 / 4 := by sorry

end cersei_cousin_fraction_l2175_217541


namespace ceiling_floor_difference_one_implies_fractional_part_l2175_217564

theorem ceiling_floor_difference_one_implies_fractional_part (x : ℝ) :
  ⌈x⌉ - ⌊x⌋ = 1 → 0 < x - ⌊x⌋ ∧ x - ⌊x⌋ < 1 := by sorry

end ceiling_floor_difference_one_implies_fractional_part_l2175_217564


namespace tangerines_left_l2175_217540

def total_tangerines : ℕ := 27
def eaten_tangerines : ℕ := 18

theorem tangerines_left : total_tangerines - eaten_tangerines = 9 := by
  sorry

end tangerines_left_l2175_217540


namespace hyperbola_focal_length_l2175_217572

/-- The focal length of a hyperbola with equation x²/2 - y²/2 = 1 is 2√2 -/
theorem hyperbola_focal_length : 
  ∃ (f : ℝ), f = 2 * Real.sqrt 2 ∧ 
  ∀ (x y : ℝ), x^2/2 - y^2/2 = 1 → 
  f = 2 * Real.sqrt ((x^2/2) + (y^2/2)) := by
  sorry

end hyperbola_focal_length_l2175_217572


namespace min_points_in_segment_seven_is_minimum_l2175_217568

-- Define the type for points on the number line
def Point := ℝ

-- Define the segments
def Segment := Set Point

-- Define the three segments
def leftSegment : Segment := {x : ℝ | x < -2}
def middleSegment : Segment := {x : ℝ | -2 ≤ x ∧ x ≤ 2}
def rightSegment : Segment := {x : ℝ | x > 2}

-- Define a property for a set of points
def hasThreePointsInOneSegment (points : Set Point) : Prop :=
  (points ∩ leftSegment).ncard ≥ 3 ∨
  (points ∩ middleSegment).ncard ≥ 3 ∨
  (points ∩ rightSegment).ncard ≥ 3

-- The main theorem
theorem min_points_in_segment :
  ∀ n : ℕ, n ≥ 7 →
    ∀ points : Set Point, points.ncard = n →
      hasThreePointsInOneSegment points :=
sorry

theorem seven_is_minimum :
  ∃ points : Set Point, points.ncard = 6 ∧
    ¬hasThreePointsInOneSegment points :=
sorry

end min_points_in_segment_seven_is_minimum_l2175_217568


namespace sphere_volume_of_hexagonal_prism_l2175_217523

/-- A hexagonal prism with specific properties -/
structure HexagonalPrism where
  -- The base is a regular hexagon
  base_is_regular : Bool
  -- Side edges are perpendicular to the base
  edges_perpendicular : Bool
  -- All vertices lie on the same spherical surface
  vertices_on_sphere : Bool
  -- Volume of the prism
  volume : ℝ
  -- Perimeter of the base
  base_perimeter : ℝ

/-- Theorem stating the volume of the sphere containing the hexagonal prism -/
theorem sphere_volume_of_hexagonal_prism (prism : HexagonalPrism)
    (h1 : prism.base_is_regular = true)
    (h2 : prism.edges_perpendicular = true)
    (h3 : prism.vertices_on_sphere = true)
    (h4 : prism.volume = 9/8)
    (h5 : prism.base_perimeter = 3) :
    ∃ (sphere_volume : ℝ), sphere_volume = 4/3 * Real.pi := by
  sorry

end sphere_volume_of_hexagonal_prism_l2175_217523


namespace max_ab_empty_solution_set_l2175_217542

theorem max_ab_empty_solution_set (a b : ℝ) : 
  (∀ x > 0, x - a * Real.log x + a - b ≥ 0) → 
  ab ≤ (1/2 : ℝ) * Real.exp 3 := by
  sorry

end max_ab_empty_solution_set_l2175_217542


namespace polynomial_property_l2175_217520

/-- A polynomial of the form x^2 + bx + c -/
def P (b c : ℝ) (x : ℝ) : ℝ := x^2 + b * x + c

/-- Theorem stating that if P(P(1)) = 0, P(P(-2)) = 0, and P(1) ≠ P(-2), then P(0) = -5/2 -/
theorem polynomial_property (b c : ℝ) :
  (P b c (P b c 1) = 0) →
  (P b c (P b c (-2)) = 0) →
  (P b c 1 ≠ P b c (-2)) →
  P b c 0 = -5/2 := by
  sorry

end polynomial_property_l2175_217520


namespace base8_addition_subtraction_l2175_217598

/-- Converts a base-8 number represented as a list of digits to its decimal (base-10) equivalent -/
def base8ToDecimal (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => acc * 8 + d) 0

/-- Converts a decimal (base-10) number to its base-8 representation as a list of digits -/
def decimalToBase8 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else aux (m / 8) ((m % 8) :: acc)
    aux n []

/-- The main theorem to prove -/
theorem base8_addition_subtraction :
  decimalToBase8 ((base8ToDecimal [1, 7, 6] + base8ToDecimal [4, 5]) - base8ToDecimal [6, 3]) = [1, 5, 1] := by
  sorry

end base8_addition_subtraction_l2175_217598


namespace root_range_l2175_217594

theorem root_range (n : ℕ) (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
   2*n - 1 < x₁ ∧ x₁ ≤ 2*n + 1 ∧
   2*n - 1 < x₂ ∧ x₂ ≤ 2*n + 1 ∧
   |x₁ - 2*n| = k * Real.sqrt x₁ ∧
   |x₂ - 2*n| = k * Real.sqrt x₂) →
  0 < k ∧ k ≤ 1 / Real.sqrt (2*n + 1) := by
sorry

end root_range_l2175_217594


namespace nathan_ate_twenty_packages_l2175_217514

/-- The number of gumballs in each package -/
def gumballs_per_package : ℕ := 5

/-- The total number of gumballs Nathan ate -/
def total_gumballs_eaten : ℕ := 100

/-- The number of packages Nathan ate -/
def packages_eaten : ℕ := total_gumballs_eaten / gumballs_per_package

theorem nathan_ate_twenty_packages : packages_eaten = 20 := by
  sorry

end nathan_ate_twenty_packages_l2175_217514


namespace max_gcd_of_sum_1071_l2175_217538

theorem max_gcd_of_sum_1071 :
  ∃ (m : ℕ), m > 0 ∧ 
  (∀ (x y : ℕ), x > 0 → y > 0 → x + y = 1071 → Nat.gcd x y ≤ m) ∧
  (∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ x + y = 1071 ∧ Nat.gcd x y = m) ∧
  m = 357 :=
sorry

end max_gcd_of_sum_1071_l2175_217538


namespace boys_average_age_l2175_217506

theorem boys_average_age (a b c : ℕ) (h1 : a = 15) (h2 : b = 3 * a) (h3 : c = 4 * a) :
  (a + b + c) / 3 = 40 := by
  sorry

end boys_average_age_l2175_217506


namespace concentric_circles_ratio_l2175_217580

/-- Given two concentric circles x and y, if the probability of a randomly selected point 
    inside circle x being outside circle y is 0.9722222222222222, then the ratio of the 
    radius of circle x to the radius of circle y is 6. -/
theorem concentric_circles_ratio (x y : Real) (h : x > y) 
    (prob : (x^2 - y^2) / x^2 = 0.9722222222222222) : x / y = 6 := by
  sorry


end concentric_circles_ratio_l2175_217580


namespace length_in_cube4_is_4root3_l2175_217552

/-- The length of the portion of the line segment from (0,0,0) to (5,5,11) 
    contained in the cube with edge length 4, which extends from (0,0,5) to (4,4,9) -/
def lengthInCube4 : ℝ := sorry

/-- The coordinates of the entry point of the line segment into the cube with edge length 4 -/
def entryPoint : Fin 3 → ℝ
| 0 => 0
| 1 => 0
| 2 => 5

/-- The coordinates of the exit point of the line segment from the cube with edge length 4 -/
def exitPoint : Fin 3 → ℝ
| 0 => 4
| 1 => 4
| 2 => 9

theorem length_in_cube4_is_4root3 : lengthInCube4 = 4 * Real.sqrt 3 := by sorry

end length_in_cube4_is_4root3_l2175_217552


namespace right_triangle_has_multiple_altitudes_l2175_217535

/-- A right triangle is a triangle with one right angle. -/
structure RightTriangle where
  vertices : Fin 3 → ℝ × ℝ
  is_right_angle : sorry

/-- An altitude of a triangle is a line segment from a vertex perpendicular to the opposite side. -/
def altitude (t : RightTriangle) (v : Fin 3) : ℝ × ℝ := sorry

/-- The number of altitudes in a right triangle -/
def num_altitudes (t : RightTriangle) : ℕ := sorry

theorem right_triangle_has_multiple_altitudes (t : RightTriangle) : num_altitudes t > 1 := by
  sorry

end right_triangle_has_multiple_altitudes_l2175_217535


namespace operation_result_l2175_217536

-- Define the operations
def op1 (m n : ℤ) : ℤ := n^2 - m
def op2 (m k : ℚ) : ℚ := (k + 2*m) / 3

-- Theorem statement
theorem operation_result : (op2 (op1 3 3) (op1 2 5)) = 35/3 := by
  sorry

end operation_result_l2175_217536


namespace sin_n_eq_cos_630_l2175_217573

theorem sin_n_eq_cos_630 (n : ℤ) :
  -180 ≤ n ∧ n ≤ 180 →
  (Real.sin (n * π / 180) = Real.cos (630 * π / 180) ↔ n = 0 ∨ n = -180 ∨ n = 180) := by
sorry

end sin_n_eq_cos_630_l2175_217573


namespace check_mistake_problem_l2175_217562

theorem check_mistake_problem :
  ∃ (x y : ℕ), 
    10 ≤ x ∧ x < 100 ∧
    10 ≤ y ∧ y < 100 ∧
    100 * y + x - (100 * x + y) = 2556 ∧
    (x + y) % 11 = 0 ∧
    x = 9 := by
  sorry

end check_mistake_problem_l2175_217562


namespace root_sum_theorem_l2175_217546

-- Define the quadratic equation
def quadratic_eq (k x : ℝ) : ℝ := k * (x^2 - x) + x + 5

-- Define the condition for k1 and k2
def k_condition (k : ℝ) : Prop :=
  ∃ a b : ℝ, quadratic_eq k a = 0 ∧ quadratic_eq k b = 0 ∧ a / b + b / a = 4 / 5

-- Theorem statement
theorem root_sum_theorem (k1 k2 : ℝ) :
  k_condition k1 ∧ k_condition k2 → k1 / k2 + k2 / k1 = 254 := by
  sorry

end root_sum_theorem_l2175_217546


namespace green_blue_difference_l2175_217565

/-- Represents the number of beads of each color in Sue's necklace -/
structure BeadCount where
  purple : Nat
  blue : Nat
  green : Nat

/-- The conditions of Sue's necklace -/
def sueNecklace : BeadCount where
  purple := 7
  blue := 2 * 7
  green := 46 - (7 + 2 * 7)

theorem green_blue_difference :
  sueNecklace.green - sueNecklace.blue = 11 := by
  sorry

end green_blue_difference_l2175_217565


namespace circle_equation_is_correct_l2175_217525

/-- The circle with center M(2, -1) that is tangent to the line x - 2y + 1 = 0 -/
def tangent_circle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 + 1)^2 = 5}

/-- The line x - 2y + 1 = 0 -/
def tangent_line : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 - 2*p.2 + 1 = 0}

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (2, -1)

theorem circle_equation_is_correct :
  ∃! (r : ℝ), r > 0 ∧
  (∀ p ∈ tangent_circle, dist p circle_center = r) ∧
  (∃ q ∈ tangent_line, dist q circle_center = r) ∧
  (∀ q ∈ tangent_line, dist q circle_center ≥ r) :=
sorry


end circle_equation_is_correct_l2175_217525


namespace correct_calculation_l2175_217509

theorem correct_calculation (a b : ℝ) : -7 * a * b^2 + 4 * a * b^2 = -3 * a * b^2 := by
  sorry

end correct_calculation_l2175_217509


namespace arithmetic_simplification_l2175_217581

theorem arithmetic_simplification : 180 * (180 - 12) - (180 * 180 - 12) = -2148 := by
  sorry

end arithmetic_simplification_l2175_217581


namespace parabola_symmetry_line_l2175_217518

/-- The parabola function -/
def parabola (x : ℝ) : ℝ := 2 * x^2

/-- The line of symmetry -/
def symmetry_line (x m : ℝ) : ℝ := x + m

/-- Theorem: For a parabola y = 2x² with two points symmetric about y = x + m, 
    and their x-coordinates multiply to -1/2, m equals 3/2 -/
theorem parabola_symmetry_line (x₁ x₂ y₁ y₂ m : ℝ) : 
  y₁ = parabola x₁ →
  y₂ = parabola x₂ →
  (y₁ + y₂) / 2 = symmetry_line ((x₁ + x₂) / 2) m →
  x₁ * x₂ = -1/2 →
  m = 3/2 := by sorry

end parabola_symmetry_line_l2175_217518


namespace rectangle_max_area_l2175_217570

theorem rectangle_max_area (x y : ℝ) (h : x > 0 ∧ y > 0) :
  2 * x + 2 * y = 60 → x * y ≤ 225 := by
  sorry

end rectangle_max_area_l2175_217570


namespace total_campers_is_150_l2175_217528

/-- The total number of campers recorded for the past three weeks -/
def total_campers (three_weeks_ago two_weeks_ago last_week : ℕ) : ℕ :=
  three_weeks_ago + two_weeks_ago + last_week

/-- Proof that the total number of campers is 150 -/
theorem total_campers_is_150 :
  ∃ (three_weeks_ago two_weeks_ago last_week : ℕ),
    two_weeks_ago = 40 ∧
    two_weeks_ago = three_weeks_ago + 10 ∧
    last_week = 80 ∧
    total_campers three_weeks_ago two_weeks_ago last_week = 150 :=
by
  sorry

end total_campers_is_150_l2175_217528


namespace third_plus_fifth_sum_l2175_217526

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  q : ℝ
  first_third_sum : a 1 + a 3 = 5
  common_ratio : q = 2
  is_geometric : ∀ n : ℕ, a (n + 1) = a n * q

/-- The theorem stating that a_3 + a_5 = 20 for the given geometric sequence -/
theorem third_plus_fifth_sum (seq : GeometricSequence) : seq.a 3 + seq.a 5 = 20 := by
  sorry

end third_plus_fifth_sum_l2175_217526


namespace tims_movie_marathon_duration_l2175_217579

/-- The duration of Tim's movie marathon --/
def movie_marathon_duration (first_movie : ℝ) (second_movie_factor : ℝ) (third_movie_offset : ℝ) : ℝ :=
  let second_movie := first_movie * (1 + second_movie_factor)
  let third_movie := first_movie + second_movie - third_movie_offset
  first_movie + second_movie + third_movie

/-- Theorem stating the duration of Tim's specific movie marathon --/
theorem tims_movie_marathon_duration :
  movie_marathon_duration 2 0.5 1 = 9 := by
  sorry

end tims_movie_marathon_duration_l2175_217579


namespace sum_of_like_monomials_l2175_217595

/-- The sum of like monomials -/
theorem sum_of_like_monomials (m n : ℕ) :
  (2 * n : ℤ) * X^(m + 2) * Y^7 + (-4 * m : ℤ) * X^4 * Y^(3 * n - 2) = -2 * X^4 * Y^7 :=
by
  sorry

#check sum_of_like_monomials

end sum_of_like_monomials_l2175_217595


namespace coin_toss_probability_l2175_217590

-- Define the probability of landing heads
def p : ℚ := 3/5

-- Define the number of tosses
def n : ℕ := 4

-- Define the number of desired heads
def k : ℕ := 2

-- Define the binomial coefficient function
def binomial_coeff (n k : ℕ) : ℚ := (Nat.choose n k : ℚ)

-- Define the probability of getting exactly k heads in n tosses
def probability (p : ℚ) (n k : ℕ) : ℚ :=
  binomial_coeff n k * p^k * (1 - p)^(n - k)

-- State the theorem
theorem coin_toss_probability : probability p n k = 216/625 := by sorry

end coin_toss_probability_l2175_217590


namespace transform_result_l2175_217555

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Rotates a point 90 degrees counterclockwise around (1, 5) -/
def rotate90 (p : Point) : Point :=
  Point.mk (-(p.y - 5) + 1) ((p.x - 1) + 5)

/-- Reflects a point about the line y = -x -/
def reflectAboutNegativeX (p : Point) : Point :=
  Point.mk (-p.y) (-p.x)

/-- The final transformation applied to the initial point -/
def transform (p : Point) : Point :=
  reflectAboutNegativeX (rotate90 p)

theorem transform_result (a b : ℝ) : 
  transform (Point.mk a b) = Point.mk (-6) 3 → b - a = 7 := by
  sorry

end transform_result_l2175_217555


namespace parabola_points_ordering_l2175_217502

/-- The parabola function -/
def f (x : ℝ) : ℝ := -x^2 - 2*x + 2

/-- Point A on the parabola -/
def A : ℝ × ℝ := (-2, f (-2))

/-- Point B on the parabola -/
def B : ℝ × ℝ := (1, f 1)

/-- Point C on the parabola -/
def C : ℝ × ℝ := (2, f 2)

/-- y₁ is the y-coordinate of point A -/
def y₁ : ℝ := A.2

/-- y₂ is the y-coordinate of point B -/
def y₂ : ℝ := B.2

/-- y₃ is the y-coordinate of point C -/
def y₃ : ℝ := C.2

theorem parabola_points_ordering : y₁ > y₂ ∧ y₂ > y₃ := by
  sorry

end parabola_points_ordering_l2175_217502


namespace exponential_inequality_l2175_217576

theorem exponential_inequality (a b : ℝ) (h : a > b) : (2 : ℝ) ^ a > (2 : ℝ) ^ b := by
  sorry

end exponential_inequality_l2175_217576


namespace max_m_is_zero_l2175_217539

/-- The condition function as described in the problem -/
def condition (m : ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₁ < m ∧ x₂ < m → (x₂ * Real.exp x₁ - x₁ * Real.exp x₂) / (Real.exp x₂ - Real.exp x₁) > 1

/-- The theorem stating that the maximum value of m for which the condition holds is 0 -/
theorem max_m_is_zero :
  ∀ m : ℝ, (∀ m' > m, ¬ condition m') → m ≤ 0 :=
sorry

end max_m_is_zero_l2175_217539


namespace smallest_value_in_range_l2175_217571

theorem smallest_value_in_range (x : ℝ) (h : 1 ≤ x ∧ x ≤ 2) :
  (1 / x ≤ x) ∧ (1 / x ≤ x^2) ∧ (1 / x ≤ 2*x) ∧ (1 / x ≤ Real.sqrt x) := by
  sorry

end smallest_value_in_range_l2175_217571


namespace arithmetic_sequence_third_term_l2175_217503

theorem arithmetic_sequence_third_term
  (a : ℤ) (d : ℤ) -- First term and common difference
  (h1 : a + 14 * d = 14) -- 15th term is 14
  (h2 : a + 15 * d = 17) -- 16th term is 17
  : a + 2 * d = -22 := -- 3rd term is -22
by
  sorry

end arithmetic_sequence_third_term_l2175_217503


namespace total_fish_count_l2175_217544

theorem total_fish_count (jerk_tuna : ℕ) (tall_tuna : ℕ) (swell_tuna : ℕ) : 
  jerk_tuna = 144 →
  tall_tuna = 2 * jerk_tuna →
  swell_tuna = tall_tuna + (tall_tuna / 2) →
  jerk_tuna + tall_tuna + swell_tuna = 864 :=
by
  sorry

end total_fish_count_l2175_217544


namespace student_a_score_l2175_217597

/-- Calculates the score for a test based on the given grading method -/
def calculate_score (total_questions : ℕ) (correct_answers : ℕ) : ℕ :=
  let incorrect_answers := total_questions - correct_answers
  correct_answers - 2 * incorrect_answers

/-- Theorem stating that the score for the given conditions is 61 -/
theorem student_a_score :
  let total_questions : ℕ := 100
  let correct_answers : ℕ := 87
  calculate_score total_questions correct_answers = 61 := by
  sorry

end student_a_score_l2175_217597


namespace base_conversion_sum_l2175_217553

/-- Converts a number from base b to base 10 -/
def to_base_10 (digits : List Nat) (b : Nat) : Nat :=
  digits.reverse.enum.foldl (fun acc (i, d) => acc + d * b^i) 0

/-- The main theorem -/
theorem base_conversion_sum :
  let x₁ := to_base_10 [3, 5, 2] 8
  let y₁ := to_base_10 [3, 1] 4
  let x₂ := to_base_10 [2, 3, 1] 5
  let y₂ := to_base_10 [3, 2] 3
  (x₁ : ℚ) / y₁ + (x₂ : ℚ) / y₂ = 28.67 := by
  sorry

end base_conversion_sum_l2175_217553


namespace solve_system_for_q_l2175_217527

theorem solve_system_for_q :
  ∀ p q : ℚ,
  5 * p + 3 * q = 7 →
  3 * p + 5 * q = 8 →
  q = 19 / 16 :=
by
  sorry

end solve_system_for_q_l2175_217527


namespace greatest_q_minus_r_l2175_217578

theorem greatest_q_minus_r : ∃ (q r : ℕ), 
  q > 0 ∧ r > 0 ∧ 
  1013 = 23 * q + r ∧
  ∀ (q' r' : ℕ), q' > 0 → r' > 0 → 1013 = 23 * q' + r' → q' - r' ≤ q - r ∧
  q - r = 39 := by
sorry

end greatest_q_minus_r_l2175_217578


namespace trajectory_is_ray_l2175_217512

/-- The set of complex numbers z satisfying |z+1| - |z-1| = 2 -/
def S : Set ℂ :=
  {z : ℂ | Complex.abs (z + 1) - Complex.abs (z - 1) = 2}

/-- A ray starting from (1, 0) and extending to the right -/
def R : Set ℂ :=
  {z : ℂ | ∃ (t : ℝ), t ≥ 0 ∧ z = 1 + t}

/-- Theorem stating that S equals R -/
theorem trajectory_is_ray : S = R := by
  sorry

end trajectory_is_ray_l2175_217512


namespace probability_four_threes_eight_dice_l2175_217547

def num_dice : ℕ := 8
def num_sides : ℕ := 8
def target_value : ℕ := 3
def num_target : ℕ := 4

def probability_exact_dice : ℚ :=
  (num_dice.choose num_target) *
  (1 / num_sides) ^ num_target *
  ((num_sides - 1) / num_sides) ^ (num_dice - num_target)

theorem probability_four_threes_eight_dice :
  probability_exact_dice = 168070 / 16777216 := by
  sorry

end probability_four_threes_eight_dice_l2175_217547


namespace james_weight_vest_cost_l2175_217587

/-- The cost of James's weight vest -/
def weight_vest_cost : ℝ := 250

/-- The cost of weight plates per pound -/
def weight_plate_cost_per_pound : ℝ := 1.2

/-- The weight of the plates in pounds -/
def weight_plate_pounds : ℝ := 200

/-- The original cost of a 200-pound weight vest -/
def original_vest_cost : ℝ := 700

/-- The discount on the 200-pound weight vest -/
def vest_discount : ℝ := 100

/-- The amount James saves with his vest -/
def james_savings : ℝ := 110

/-- Theorem: The cost of James's weight vest is $250 -/
theorem james_weight_vest_cost : 
  weight_vest_cost = 
    (original_vest_cost - vest_discount) - 
    (weight_plate_cost_per_pound * weight_plate_pounds) - 
    james_savings := by
  sorry

end james_weight_vest_cost_l2175_217587


namespace cubic_roots_squared_l2175_217560

def f (x : ℝ) : ℝ := x^3 + 2*x^2 + 3*x + 4

def g (b c d : ℝ) (x : ℝ) : ℝ := x^3 + b*x^2 + c*x + d

theorem cubic_roots_squared (b c d : ℝ) :
  (∃ r₁ r₂ r₃ : ℝ, r₁ ≠ r₂ ∧ r₂ ≠ r₃ ∧ r₁ ≠ r₃ ∧
    f r₁ = 0 ∧ f r₂ = 0 ∧ f r₃ = 0) →
  (∀ x : ℝ, f x = 0 → g b c d (x^2) = 0) →
  b = 4 ∧ c = -15 ∧ d = -32 := by
sorry

end cubic_roots_squared_l2175_217560


namespace two_digit_penultimate_five_l2175_217559

/-- A function that returns the penultimate digit of a natural number -/
def penultimateDigit (n : ℕ) : ℕ :=
  (n / 10) % 10

/-- A predicate that checks if a number is a two-digit number -/
def isTwoDigit (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

theorem two_digit_penultimate_five :
  ∀ x : ℕ, isTwoDigit x →
    (∃ k : ℤ, penultimateDigit (x * k.natAbs) = 5) ↔
    (x = 25 ∨ x = 50 ∨ x = 75) :=
by sorry

end two_digit_penultimate_five_l2175_217559


namespace piggy_bank_problem_l2175_217583

theorem piggy_bank_problem (total_cents : ℕ) (nickel_quarter_diff : ℕ) 
  (h1 : total_cents = 625)
  (h2 : nickel_quarter_diff = 9) : 
  ∃ (nickels quarters : ℕ),
    nickels = quarters + nickel_quarter_diff ∧
    5 * nickels + 25 * quarters = total_cents ∧
    nickels = 28 := by
  sorry

end piggy_bank_problem_l2175_217583


namespace deck_width_l2175_217530

/-- Proves that for a rectangular pool of 20 feet by 22 feet, surrounded by a deck of uniform width,
    if the total area of the pool and deck is 728 square feet, then the width of the deck is 3 feet. -/
theorem deck_width (w : ℝ) : 
  (20 + 2*w) * (22 + 2*w) = 728 → w = 3 := by sorry

end deck_width_l2175_217530


namespace function_equation_implies_identity_l2175_217599

/-- A function satisfying the given functional equation is the identity function. -/
theorem function_equation_implies_identity (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x^2 + f y) = y + (f x)^2) : 
  ∀ x : ℝ, f x = x := by
  sorry

end function_equation_implies_identity_l2175_217599


namespace survey_size_l2175_217554

-- Define the problem parameters
def percent_independent : ℚ := 752 / 1000
def percent_no_companionship : ℚ := 621 / 1000
def misinformed_students : ℕ := 41

-- Define the theorem
theorem survey_size :
  ∃ (total_students : ℕ),
    (total_students > 0) ∧
    (↑misinformed_students : ℚ) / (percent_independent * percent_no_companionship * ↑total_students) = 1 ∧
    total_students = 90 := by
  sorry

end survey_size_l2175_217554


namespace root_product_equals_eight_l2175_217549

theorem root_product_equals_eight : 
  (64 : ℝ) ^ (1/6) * (8 : ℝ) ^ (1/3) * (16 : ℝ) ^ (1/2) = 8 := by
  sorry

end root_product_equals_eight_l2175_217549


namespace mrs_brown_shoe_price_l2175_217577

/-- Calculates the final price for a Mother's Day purchase with additional discount for multiple children -/
def mothersDayPrice (originalPrice : ℝ) (numChildren : ℕ) : ℝ :=
  let mothersDayDiscount := 0.1
  let additionalDiscount := 0.04
  let discountedPrice := originalPrice * (1 - mothersDayDiscount)
  if numChildren ≥ 3 then
    discountedPrice * (1 - additionalDiscount)
  else
    discountedPrice

theorem mrs_brown_shoe_price :
  mothersDayPrice 125 4 = 108 := by
  sorry

end mrs_brown_shoe_price_l2175_217577


namespace problem_solution_l2175_217511

theorem problem_solution (p q : ℤ) 
  (h1 : p > 1) 
  (h2 : q > 1) 
  (h3 : ∃ k : ℤ, (2 * p - 1) = k * q) 
  (h4 : ∃ m : ℤ, (2 * q - 1) = m * p) : 
  p + q = 8 := by
sorry

end problem_solution_l2175_217511


namespace range_of_fraction_l2175_217524

theorem range_of_fraction (a b : ℝ) (h1 : 0 < a) (h2 : a ≤ 2) (h3 : b ≥ 1) (h4 : b ≤ a^2) :
  ∃ (t : ℝ), t = b / a ∧ 1/2 ≤ t ∧ t ≤ 2 :=
by sorry

end range_of_fraction_l2175_217524


namespace equation_solution_l2175_217575

theorem equation_solution (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 3) 
  (h : 3 / a + 6 / b = 2 / 3) : a = 9 * b / (2 * b - 18) := by
  sorry

end equation_solution_l2175_217575


namespace intersection_point_existence_l2175_217532

theorem intersection_point_existence : ∃! x₀ : ℝ, x₀ ∈ Set.Ioo 1 2 ∧ x₀^3 = (1/2)^(x₀ - 2) := by
  sorry

end intersection_point_existence_l2175_217532


namespace smallest_multiple_of_5_711_1033_l2175_217596

theorem smallest_multiple_of_5_711_1033 :
  ∃ (n : ℕ), n > 0 ∧ 
  5 ∣ n ∧ 711 ∣ n ∧ 1033 ∣ n ∧ 
  (∀ m : ℕ, m > 0 → 5 ∣ m → 711 ∣ m → 1033 ∣ m → n ≤ m) ∧
  n = 3683445 := by
  sorry

end smallest_multiple_of_5_711_1033_l2175_217596


namespace western_village_conscription_l2175_217556

theorem western_village_conscription 
  (north_pop : ℕ) 
  (west_pop : ℕ) 
  (south_pop : ℕ) 
  (total_conscripts : ℕ) 
  (h1 : north_pop = 8758) 
  (h2 : west_pop = 7236) 
  (h3 : south_pop = 8356) 
  (h4 : total_conscripts = 378) : 
  (west_pop : ℚ) / (north_pop + west_pop + south_pop : ℚ) * total_conscripts = 112 := by
sorry

end western_village_conscription_l2175_217556


namespace min_people_in_group_l2175_217588

/-- Represents the number of people who like a specific fruit or combination of fruits. -/
structure FruitPreferences where
  apples : Nat
  blueberries : Nat
  cantaloupe : Nat
  dates : Nat
  blueberriesAndApples : Nat
  blueberriesAndCantaloupe : Nat
  cantaloupeAndDates : Nat

/-- The conditions given in the problem. -/
def problemConditions : FruitPreferences where
  apples := 13
  blueberries := 9
  cantaloupe := 15
  dates := 6
  blueberriesAndApples := 0  -- Derived from the solution
  blueberriesAndCantaloupe := 9  -- Derived from the solution
  cantaloupeAndDates := 6  -- Derived from the solution

/-- Theorem stating the minimum number of people in the group. -/
theorem min_people_in_group (prefs : FruitPreferences) 
  (h1 : prefs.blueberries = prefs.blueberriesAndApples + prefs.blueberriesAndCantaloupe)
  (h2 : prefs.cantaloupe = prefs.blueberriesAndCantaloupe + prefs.cantaloupeAndDates)
  (h3 : prefs = problemConditions) :
  prefs.apples + prefs.blueberriesAndCantaloupe + prefs.cantaloupeAndDates = 22 := by
  sorry

end min_people_in_group_l2175_217588


namespace absolute_value_sum_l2175_217551

theorem absolute_value_sum (a b : ℝ) : a^2 + b^2 > 1 → |a| + |b| > 1 := by
  sorry

end absolute_value_sum_l2175_217551


namespace bobs_total_bushels_l2175_217591

/-- Calculates the number of bushels from a row of corn, rounding down to the nearest whole bushel -/
def bushelsFromRow (stalks : ℕ) (stalksPerBushel : ℕ) : ℕ :=
  stalks / stalksPerBushel

/-- Represents Bob's corn harvest -/
structure CornHarvest where
  row1 : (ℕ × ℕ)
  row2 : (ℕ × ℕ)
  row3 : (ℕ × ℕ)
  row4 : (ℕ × ℕ)
  row5 : (ℕ × ℕ)
  row6 : (ℕ × ℕ)
  row7 : (ℕ × ℕ)

/-- Calculates the total bushels of corn from Bob's harvest -/
def totalBushels (harvest : CornHarvest) : ℕ :=
  bushelsFromRow harvest.row1.1 harvest.row1.2 +
  bushelsFromRow harvest.row2.1 harvest.row2.2 +
  bushelsFromRow harvest.row3.1 harvest.row3.2 +
  bushelsFromRow harvest.row4.1 harvest.row4.2 +
  bushelsFromRow harvest.row5.1 harvest.row5.2 +
  bushelsFromRow harvest.row6.1 harvest.row6.2 +
  bushelsFromRow harvest.row7.1 harvest.row7.2

/-- Bob's actual corn harvest -/
def bobsHarvest : CornHarvest :=
  { row1 := (82, 8)
    row2 := (94, 9)
    row3 := (78, 7)
    row4 := (96, 12)
    row5 := (85, 10)
    row6 := (91, 13)
    row7 := (88, 11) }

theorem bobs_total_bushels :
  totalBushels bobsHarvest = 62 := by
  sorry

end bobs_total_bushels_l2175_217591


namespace line_circle_intersection_l2175_217519

theorem line_circle_intersection (a b : ℝ) (h : ∃ (x y : ℝ), x^2 + y^2 = 1 ∧ x/a + y/b = 1) :
  1/a^2 + 1/b^2 ≥ 1 := by
sorry

end line_circle_intersection_l2175_217519


namespace beverage_selection_probabilities_l2175_217534

def total_cups : ℕ := 5
def type_a_cups : ℕ := 3
def type_b_cups : ℕ := 2
def cups_to_select : ℕ := 3

def probability_all_correct : ℚ := 1 / 10
def probability_at_least_two_correct : ℚ := 7 / 10

theorem beverage_selection_probabilities :
  (total_cups = type_a_cups + type_b_cups) →
  (cups_to_select = type_a_cups) →
  (probability_all_correct = 1 / (Nat.choose total_cups cups_to_select)) ∧
  (probability_at_least_two_correct = 
    (Nat.choose type_a_cups cups_to_select + 
     Nat.choose type_a_cups (cups_to_select - 1) * Nat.choose type_b_cups 1) / 
    (Nat.choose total_cups cups_to_select)) := by
  sorry

end beverage_selection_probabilities_l2175_217534


namespace dice_sum_product_l2175_217586

theorem dice_sum_product (a b c d : ℕ) : 
  1 ≤ a ∧ a ≤ 6 ∧
  1 ≤ b ∧ b ≤ 6 ∧
  1 ≤ c ∧ c ≤ 6 ∧
  1 ≤ d ∧ d ≤ 6 ∧
  a * b * c * d = 144 →
  a + b + c + d ≠ 18 := by
  sorry

end dice_sum_product_l2175_217586


namespace missing_number_proof_l2175_217521

def known_numbers : List ℝ := [13, 8, 13, 21, 23]

theorem missing_number_proof (mean : ℝ) (h_mean : mean = 14.2) :
  ∃ x : ℝ, (known_numbers.sum + x) / 6 = mean ∧ x = 7.2 := by
  sorry

end missing_number_proof_l2175_217521


namespace optimal_point_distribution_l2175_217582

/-- A configuration of points in a space -/
structure PointConfiguration where
  total_points : ℕ
  num_groups : ℕ
  group_sizes : List ℕ
  no_collinear_triple : Prop
  distinct_group_sizes : Prop
  sum_of_sizes_equals_total : group_sizes.sum = total_points

/-- The number of triangles formed by choosing one point from each of any three different groups -/
def num_triangles (config : PointConfiguration) : ℕ :=
  sorry

/-- The optimal configuration maximizes the number of triangles -/
def is_optimal (config : PointConfiguration) : Prop :=
  ∀ other : PointConfiguration, num_triangles config ≥ num_triangles other

/-- The theorem stating the optimal configuration -/
theorem optimal_point_distribution :
  ∃ (optimal_config : PointConfiguration),
    optimal_config.total_points = 1989 ∧
    optimal_config.num_groups = 30 ∧
    optimal_config.group_sizes = [51, 52, 53, 54, 55, 56, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81] ∧
    is_optimal optimal_config :=
  sorry

end optimal_point_distribution_l2175_217582


namespace min_consecutive_sum_36_proof_l2175_217505

/-- The sum of N consecutive integers starting from a -/
def sum_consecutive (a : ℤ) (N : ℕ) : ℤ := N * (2 * a + N - 1) / 2

/-- Predicate to check if a sequence of N consecutive integers starting from a sums to 36 -/
def is_valid_sequence (a : ℤ) (N : ℕ) : Prop := sum_consecutive a N = 36

/-- The minimum number of consecutive integers that sum to 36 -/
def min_consecutive_sum_36 : ℕ := 3

theorem min_consecutive_sum_36_proof :
  (∃ a : ℤ, is_valid_sequence a min_consecutive_sum_36) ∧
  (∀ N : ℕ, N < min_consecutive_sum_36 → ∀ a : ℤ, ¬is_valid_sequence a N) :=
sorry

end min_consecutive_sum_36_proof_l2175_217505


namespace min_blue_chips_correct_l2175_217513

/-- Represents the number of chips of each color in the box -/
structure ChipCounts where
  white : ℕ
  blue : ℕ
  red : ℕ

/-- Checks if the chip counts satisfy the given conditions -/
def satisfiesConditions (counts : ChipCounts) : Prop :=
  counts.blue ≥ counts.white / 3 ∧
  counts.blue ≤ counts.red / 4 ∧
  counts.white + counts.blue ≥ 75

/-- The minimum number of blue chips that satisfies the conditions -/
def minBlueChips : ℕ := 19

theorem min_blue_chips_correct :
  (∀ counts : ChipCounts, satisfiesConditions counts → counts.blue ≥ minBlueChips) ∧
  (∃ counts : ChipCounts, satisfiesConditions counts ∧ counts.blue = minBlueChips) := by
  sorry

end min_blue_chips_correct_l2175_217513


namespace ellipse_theorem_parabola_theorem_l2175_217566

-- Define the ellipses
def ellipse1 (x y : ℝ) := x^2/9 + y^2/4 = 1
def ellipse2 (x y : ℝ) := x^2/12 + y^2/7 = 1

-- Define the parabolas
def parabola1 (x y : ℝ) := x^2 = -2 * Real.sqrt 2 * y
def parabola2 (x y : ℝ) := y^2 = -8 * x

-- Theorem for the ellipse
theorem ellipse_theorem :
  (ellipse2 (-3) 2) ∧
  (∀ (x y : ℝ), ellipse1 x y ↔ ellipse2 x y) := by sorry

-- Theorem for the parabolas
theorem parabola_theorem :
  (parabola1 (-4) (-4 * Real.sqrt 2)) ∧
  (parabola2 (-4) (-4 * Real.sqrt 2)) ∧
  (∀ (x y : ℝ), parabola1 x y → x = 0 ∨ y = 0) ∧
  (∀ (x y : ℝ), parabola2 x y → x = 0 ∨ y = 0) := by sorry

end ellipse_theorem_parabola_theorem_l2175_217566


namespace event_committee_count_l2175_217529

/-- The number of teams in the league -/
def num_teams : ℕ := 5

/-- The number of members in each team -/
def team_size : ℕ := 8

/-- The number of members selected from the host team -/
def host_selection : ℕ := 4

/-- The number of members selected from each non-host team -/
def non_host_selection : ℕ := 3

/-- The total number of possible event committees -/
def total_committees : ℕ := 3442073600

/-- Theorem stating the number of possible event committees -/
theorem event_committee_count :
  (num_teams : ℕ) *
  (Nat.choose team_size host_selection) *
  (Nat.choose team_size non_host_selection)^(num_teams - 1) =
  total_committees := by sorry

end event_committee_count_l2175_217529


namespace two_year_increase_l2175_217574

/-- Calculates the final amount after a given number of years with a fixed annual increase rate -/
def finalAmount (initialAmount : ℝ) (annualRate : ℝ) (years : ℕ) : ℝ :=
  initialAmount * (1 + annualRate) ^ years

/-- Theorem: An initial amount of 57600, increasing by 1/8 annually, becomes 72900 after 2 years -/
theorem two_year_increase : 
  finalAmount 57600 (1/8) 2 = 72900 := by sorry

end two_year_increase_l2175_217574


namespace chess_grandmaster_time_calculation_l2175_217593

theorem chess_grandmaster_time_calculation : 
  let time_learn_rules : ℕ := 2
  let time_get_proficient : ℕ := 49 * time_learn_rules
  let time_become_master : ℕ := 100 * (time_learn_rules + time_get_proficient)
  let total_time : ℕ := time_learn_rules + time_get_proficient + time_become_master
  total_time = 10100 := by
sorry

end chess_grandmaster_time_calculation_l2175_217593


namespace john_popcorn_profit_l2175_217537

/-- Calculates the profit John makes from selling popcorn bags -/
theorem john_popcorn_profit :
  let regular_price : ℚ := 4
  let discount_rate : ℚ := 0.1
  let adult_price : ℚ := 8
  let child_price : ℚ := 6
  let adult_bags : ℕ := 20
  let child_bags : ℕ := 10
  let total_bags : ℕ := adult_bags + child_bags
  let discounted_price : ℚ := regular_price * (1 - discount_rate)
  let total_cost : ℚ := (total_bags : ℚ) * discounted_price
  let total_revenue : ℚ := (adult_bags : ℚ) * adult_price + (child_bags : ℚ) * child_price
  let profit : ℚ := total_revenue - total_cost
  profit = 112 :=
by
  sorry


end john_popcorn_profit_l2175_217537


namespace squares_property_l2175_217585

theorem squares_property (a b c : ℕ) 
  (h : a^2 + b^2 + c^2 = (a - b)^2 + (b - c)^2 + (c - a)^2) :
  ∃ (w x y z : ℕ), 
    a * b = w^2 ∧ 
    b * c = x^2 ∧ 
    c * a = y^2 ∧ 
    a * b + b * c + c * a = z^2 := by
  sorry

end squares_property_l2175_217585


namespace lowest_possible_score_l2175_217567

def test_count : Nat := 6
def max_score : Nat := 100
def target_average : Nat := 85
def min_score : Nat := 75

def first_four_scores : List Nat := [79, 88, 94, 91]

theorem lowest_possible_score :
  ∀ (score1 score2 : Nat),
  (score1 ≥ min_score) →
  (score2 ≥ min_score) →
  (List.sum first_four_scores + score1 + score2) / test_count = target_average →
  (∀ (s : Nat), s ≥ min_score ∧ s < score1 →
    (List.sum first_four_scores + s + score2) / test_count < target_average) →
  score1 = min_score ∨ score2 = min_score :=
by sorry

end lowest_possible_score_l2175_217567


namespace inequality_proof_l2175_217517

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : Real.sqrt a + Real.sqrt b + Real.sqrt c = 3) :
  (a + b) / (2 + a + b) + (b + c) / (2 + b + c) + (c + a) / (2 + c + a) ≥ 3 / 2 ∧
  ((a + b) / (2 + a + b) + (b + c) / (2 + b + c) + (c + a) / (2 + c + a) = 3 / 2 ↔
   a = 1 ∧ b = 1 ∧ c = 1) :=
by sorry

end inequality_proof_l2175_217517


namespace sqrt_three_multiplication_l2175_217584

theorem sqrt_three_multiplication : Real.sqrt 3 * (2 * Real.sqrt 3 - 2) = 6 - 2 * Real.sqrt 3 := by
  sorry

end sqrt_three_multiplication_l2175_217584


namespace orange_price_calculation_l2175_217569

-- Define the price function for oranges
def orange_price (mass : ℝ) : ℝ := sorry

-- State the theorem
theorem orange_price_calculation 
  (proportional : ∀ m₁ m₂ : ℝ, orange_price m₁ / m₁ = orange_price m₂ / m₂)
  (given_price : orange_price 12 = 36) :
  orange_price 2 = 6 := by sorry

end orange_price_calculation_l2175_217569


namespace smallest_number_of_pens_l2175_217543

theorem smallest_number_of_pens (pen_package_size : Nat) (pencil_package_size : Nat)
  (h1 : pen_package_size = 12)
  (h2 : pencil_package_size = 15) :
  Nat.lcm pen_package_size pencil_package_size = 60 := by
  sorry

end smallest_number_of_pens_l2175_217543


namespace celine_change_l2175_217522

def laptop_base_price : ℚ := 600
def smartphone_base_price : ℚ := 400
def tablet_base_price : ℚ := 250
def headphone_base_price : ℚ := 100

def laptop_discount : ℚ := 0.15
def smartphone_increase : ℚ := 0.10
def tablet_discount : ℚ := 0.20

def sales_tax : ℚ := 0.06

def laptop_quantity : ℕ := 2
def smartphone_quantity : ℕ := 3
def tablet_quantity : ℕ := 4
def headphone_quantity : ℕ := 6

def celine_budget : ℚ := 6000

theorem celine_change : 
  let laptop_price := laptop_base_price * (1 - laptop_discount)
  let smartphone_price := smartphone_base_price * (1 + smartphone_increase)
  let tablet_price := tablet_base_price * (1 - tablet_discount)
  let headphone_price := headphone_base_price

  let total_before_tax := 
    laptop_price * laptop_quantity +
    smartphone_price * smartphone_quantity +
    tablet_price * tablet_quantity +
    headphone_price * headphone_quantity

  let total_with_tax := total_before_tax * (1 + sales_tax)

  celine_budget - total_with_tax = 2035.60 := by sorry

end celine_change_l2175_217522


namespace equation_solutions_l2175_217531

theorem equation_solutions :
  (∃ x : ℚ, (17/2 : ℚ) * x = (17/2 : ℚ) + x ∧ x = (17/15 : ℚ)) ∧
  (∃ y : ℚ, y / (2/3 : ℚ) = y + (2/3 : ℚ) ∧ y = (4/3 : ℚ)) :=
by sorry

end equation_solutions_l2175_217531


namespace solution_set_when_m_eq_2_range_of_m_when_f_leq_5_l2175_217545

-- Define the function f
def f (x m : ℝ) : ℝ := |x - 1| - |x - m|

-- Theorem for part I
theorem solution_set_when_m_eq_2 :
  {x : ℝ | f x 2 ≥ 1} = {x : ℝ | x ≥ 2} := by sorry

-- Theorem for part II
theorem range_of_m_when_f_leq_5 :
  {m : ℝ | ∀ x, f x m ≤ 5} = {m : ℝ | -4 ≤ m ∧ m ≤ 6} := by sorry

end solution_set_when_m_eq_2_range_of_m_when_f_leq_5_l2175_217545


namespace correct_geometry_problems_l2175_217548

theorem correct_geometry_problems (total_problems : ℕ) (total_algebra : ℕ) 
  (algebra_correct_ratio : ℚ) (algebra_incorrect_ratio : ℚ)
  (geometry_correct_ratio : ℚ) (geometry_incorrect_ratio : ℚ) :
  total_problems = 60 →
  total_algebra = 25 →
  algebra_correct_ratio = 3 →
  algebra_incorrect_ratio = 2 →
  geometry_correct_ratio = 4 →
  geometry_incorrect_ratio = 1 →
  ∃ (correct_geometry : ℕ), correct_geometry = 28 ∧
    correct_geometry * (geometry_correct_ratio + geometry_incorrect_ratio) = 
    (total_problems - total_algebra) * geometry_correct_ratio :=
by sorry

end correct_geometry_problems_l2175_217548


namespace circle_tangent_sum_radii_l2175_217508

/-- A circle with center C(r,r) is tangent to the positive x-axis and y-axis,
    and externally tangent to a circle centered at (4,0) with radius 2.
    The sum of all possible radii of the circle with center C is 12. -/
theorem circle_tangent_sum_radii :
  ∀ r : ℝ,
  (r > 0) →
  ((r - 4)^2 + r^2 = (r + 2)^2) →
  (∃ r₁ r₂ : ℝ, (r₁ > 0 ∧ r₂ > 0) ∧ 
    ((r₁ - 4)^2 + r₁^2 = (r₁ + 2)^2) ∧
    ((r₂ - 4)^2 + r₂^2 = (r₂ + 2)^2) ∧
    r₁ + r₂ = 12) :=
by
  sorry

#check circle_tangent_sum_radii

end circle_tangent_sum_radii_l2175_217508


namespace addition_preserves_inequality_l2175_217557

theorem addition_preserves_inequality (a b c d : ℝ) :
  a > b → c > d → a + c > b + d := by sorry

end addition_preserves_inequality_l2175_217557


namespace inverse_of_P_l2175_217563

-- Define the original proposition P
def P : Prop → Prop → Prop := λ odd prime => odd → prime

-- Define the inverse proposition
def inverse_prop (p : Prop → Prop → Prop) : Prop → Prop → Prop :=
  λ a b => p b a

-- Theorem stating that the inverse of P is as described
theorem inverse_of_P :
  inverse_prop P = (λ prime odd => prime → odd) :=
by sorry

end inverse_of_P_l2175_217563


namespace comic_books_left_l2175_217510

theorem comic_books_left (initial_total : ℕ) (sold : ℕ) (left : ℕ) : 
  initial_total = 90 → sold = 65 → left = initial_total - sold → left = 25 := by
  sorry

end comic_books_left_l2175_217510


namespace set_A_at_most_one_element_l2175_217516

theorem set_A_at_most_one_element (a : ℝ) : 
  (∃! x : ℝ, a * x^2 - 3 * x + 2 = 0) ↔ (a ≥ 9/8 ∨ a = 0) :=
sorry

end set_A_at_most_one_element_l2175_217516


namespace expression_evaluation_l2175_217501

theorem expression_evaluation :
  let x : ℚ := -3
  let numerator : ℚ := 5 + x * (5 + x) - 5^2
  let denominator : ℚ := x - 5 + x^2
  numerator / denominator = -26 := by sorry

end expression_evaluation_l2175_217501


namespace marie_erasers_l2175_217533

/-- Given that Marie starts with 95 erasers and loses 42, prove that she ends with 53 erasers. -/
theorem marie_erasers : 
  let initial_erasers : ℕ := 95
  let lost_erasers : ℕ := 42
  initial_erasers - lost_erasers = 53 := by sorry

end marie_erasers_l2175_217533
