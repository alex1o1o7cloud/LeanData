import Mathlib

namespace NUMINAMATH_CALUDE_circus_kids_l3807_380775

theorem circus_kids (total_cost : ℕ) (kid_ticket_cost : ℕ) (num_adults : ℕ) : 
  total_cost = 50 →
  kid_ticket_cost = 5 →
  num_adults = 2 →
  ∃ (num_kids : ℕ), 
    (num_kids * kid_ticket_cost + num_adults * (2 * kid_ticket_cost) = total_cost) ∧
    num_kids = 2 := by
  sorry

end NUMINAMATH_CALUDE_circus_kids_l3807_380775


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l3807_380729

theorem partial_fraction_decomposition (C D : ℚ) :
  (∀ x : ℚ, x ≠ 11 ∧ x ≠ -5 →
    (7 * x - 4) / (x^2 - 6 * x - 55) = C / (x - 11) + D / (x + 5)) →
  C = 73 / 16 ∧ D = 39 / 16 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l3807_380729


namespace NUMINAMATH_CALUDE_ellipse_intersection_theorem_l3807_380782

/-- Ellipse C with equation (x^2 / 4) + (y^2 / m) = 1 -/
def ellipse_C (m : ℝ) (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / m = 1

/-- Point P on the x-axis -/
def point_P : ℝ × ℝ := (-1, 0)

/-- Line l passing through point P -/
def line_l (k : ℝ) (x y : ℝ) : Prop :=
  y = k * (x + 1)

/-- Condition for circle with AB as diameter passing through origin -/
def circle_condition (A B : ℝ × ℝ) : Prop :=
  A.1 * B.1 + A.2 * B.2 = 0

/-- Main theorem: If there exists a line l intersecting ellipse C at points A and B
    such that the circle with AB as diameter passes through the origin,
    then m is in the range (0, 4/3] -/
theorem ellipse_intersection_theorem (m : ℝ) :
  (m > 0) →
  (∃ (k : ℝ) (A B : ℝ × ℝ),
    ellipse_C m A.1 A.2 ∧
    ellipse_C m B.1 B.2 ∧
    line_l k A.1 A.2 ∧
    line_l k B.1 B.2 ∧
    circle_condition A B) →
  0 < m ∧ m ≤ 4/3 :=
sorry

end NUMINAMATH_CALUDE_ellipse_intersection_theorem_l3807_380782


namespace NUMINAMATH_CALUDE_exponent_division_l3807_380701

theorem exponent_division (a : ℝ) (h : a ≠ 0) : a^7 / a^3 = a^4 := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l3807_380701


namespace NUMINAMATH_CALUDE_distance_satisfies_conditions_l3807_380739

/-- The distance from the village to the post-office in kilometers. -/
def D : ℝ := 20

/-- The speed of the man traveling to the post-office in km/h. -/
def speed_to_postoffice : ℝ := 25

/-- The speed of the man walking back to the village in km/h. -/
def speed_to_village : ℝ := 4

/-- The total time for the round trip in hours. -/
def total_time : ℝ := 5.8

/-- Theorem stating that the distance D satisfies the given conditions. -/
theorem distance_satisfies_conditions : 
  D / speed_to_postoffice + D / speed_to_village = total_time :=
sorry

end NUMINAMATH_CALUDE_distance_satisfies_conditions_l3807_380739


namespace NUMINAMATH_CALUDE_study_seminar_selection_l3807_380742

theorem study_seminar_selection (n m k : ℕ) (h1 : n = 10) (h2 : m = 6) (h3 : k = 2) :
  (n.choose m) - ((n - k).choose (m - k)) = 140 := by
  sorry

end NUMINAMATH_CALUDE_study_seminar_selection_l3807_380742


namespace NUMINAMATH_CALUDE_intersection_nonempty_implies_a_equals_four_l3807_380745

theorem intersection_nonempty_implies_a_equals_four :
  ∀ (a : ℝ), 
  let A : Set ℝ := {3, 4, 2*a - 3}
  let B : Set ℝ := {a}
  (A ∩ B).Nonempty → a = 4 := by
sorry

end NUMINAMATH_CALUDE_intersection_nonempty_implies_a_equals_four_l3807_380745


namespace NUMINAMATH_CALUDE_union_equals_A_l3807_380708

def A : Set ℤ := {-1, 0, 1}
def B (a : ℤ) : Set ℤ := {a, a^2}

theorem union_equals_A (a : ℤ) : A ∪ B a = A ↔ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_union_equals_A_l3807_380708


namespace NUMINAMATH_CALUDE_first_courier_speed_l3807_380790

/-- The speed of the first courier in km/h -/
def v : ℝ := 30

/-- The distance between cities A and B in km -/
def distance : ℝ := 120

/-- The speed of the second courier in km/h -/
def speed_second : ℝ := 50

/-- The time delay of the second courier in hours -/
def delay : ℝ := 1

theorem first_courier_speed :
  (distance / v = (3 * speed_second) / (v - speed_second)) ∧
  (v > 0) ∧ (v < speed_second) := by
  sorry

#check first_courier_speed

end NUMINAMATH_CALUDE_first_courier_speed_l3807_380790


namespace NUMINAMATH_CALUDE_small_cuboid_width_is_four_l3807_380755

/-- Represents the dimensions of a cuboid -/
structure CuboidDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a cuboid given its dimensions -/
def cuboidVolume (d : CuboidDimensions) : ℝ :=
  d.length * d.width * d.height

theorem small_cuboid_width_is_four
  (large : CuboidDimensions)
  (small_length : ℝ)
  (small_height : ℝ)
  (num_small_cuboids : ℕ)
  (h1 : large.length = 16)
  (h2 : large.width = 10)
  (h3 : large.height = 12)
  (h4 : small_length = 5)
  (h5 : small_height = 3)
  (h6 : num_small_cuboids = 32)
  (h7 : ∃ (small_width : ℝ),
    cuboidVolume large = num_small_cuboids * cuboidVolume
      { length := small_length
        width := small_width
        height := small_height }) :
  ∃ (small_width : ℝ), small_width = 4 := by
  sorry

end NUMINAMATH_CALUDE_small_cuboid_width_is_four_l3807_380755


namespace NUMINAMATH_CALUDE_cryptarithm_solution_l3807_380777

theorem cryptarithm_solution :
  ∃! (A B : ℕ), 
    A < 10 ∧ B < 10 ∧ A ≠ B ∧
    9 * (10 * A + B) = 100 * A + 10 * A + B ∧
    A = 2 ∧ B = 5 :=
by sorry

end NUMINAMATH_CALUDE_cryptarithm_solution_l3807_380777


namespace NUMINAMATH_CALUDE_fraction_inequality_function_minimum_l3807_380725

-- Problem 1
theorem fraction_inequality (c a b : ℝ) (h1 : c > a) (h2 : a > b) (h3 : b > 0) :
  a / (c - a) > b / (c - b) := by sorry

-- Problem 2
theorem function_minimum (x : ℝ) (h : x > 2) :
  x + 16 / (x - 2) ≥ 10 := by sorry

end NUMINAMATH_CALUDE_fraction_inequality_function_minimum_l3807_380725


namespace NUMINAMATH_CALUDE_black_balls_count_l3807_380706

theorem black_balls_count (total_balls : ℕ) (red_balls : ℕ) (prob_red : ℚ) : 
  red_balls = 10 →
  prob_red = 2/7 →
  (red_balls : ℚ) / total_balls = prob_red →
  total_balls - red_balls = 25 := by
sorry

end NUMINAMATH_CALUDE_black_balls_count_l3807_380706


namespace NUMINAMATH_CALUDE_sum_set_cardinality_l3807_380718

/-- A function that generates an arithmetic sequence with a given first term, common difference, and length. -/
def arithmeticSequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : Fin n → ℝ :=
  fun i => a₁ + d * (i : ℝ)

/-- The set of all sums of pairs of elements from the arithmetic sequence. -/
def sumSet (a₁ : ℝ) (d : ℝ) (n : ℕ) : Set ℝ :=
  {x | ∃ (i j : Fin n), i ≤ j ∧ x = arithmeticSequence a₁ d n i + arithmeticSequence a₁ d n j}

/-- The theorem stating that the number of elements in the sum set is 2n - 3. -/
theorem sum_set_cardinality (a₁ : ℝ) (d : ℝ) (n : ℕ) (h₁ : n ≥ 3) (h₂ : d > 0) :
  Nat.card (sumSet a₁ d n) = 2 * n - 3 :=
sorry

end NUMINAMATH_CALUDE_sum_set_cardinality_l3807_380718


namespace NUMINAMATH_CALUDE_system_solution_l3807_380737

theorem system_solution (k : ℚ) (x y z : ℚ) : 
  x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 →
  x + k*y + 4*z = 0 →
  4*x + k*y - 3*z = 0 →
  3*x + 5*y - 4*z = 0 →
  x*z / (y^2) = 147/28 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l3807_380737


namespace NUMINAMATH_CALUDE_bicycling_time_l3807_380746

-- Define the distance in kilometers
def distance : ℝ := 96

-- Define the speed in kilometers per hour
def speed : ℝ := 6

-- Theorem: The time taken is 16 hours
theorem bicycling_time : distance / speed = 16 := by
  sorry

end NUMINAMATH_CALUDE_bicycling_time_l3807_380746


namespace NUMINAMATH_CALUDE_roof_metal_bars_l3807_380772

/-- The number of sets of metal bars needed for the roof -/
def num_sets : ℕ := 2

/-- The number of metal bars in each set -/
def bars_per_set : ℕ := 7

/-- The total number of metal bars needed for the roof -/
def total_bars : ℕ := num_sets * bars_per_set

theorem roof_metal_bars : total_bars = 14 := by
  sorry

end NUMINAMATH_CALUDE_roof_metal_bars_l3807_380772


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_l3807_380740

theorem sum_of_x_and_y (x y : ℚ) 
  (h1 : 2 / x + 3 / y = 4) 
  (h2 : 2 / x - 3 / y = -2) : 
  x + y = 3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_l3807_380740


namespace NUMINAMATH_CALUDE_remainder_problem_l3807_380731

theorem remainder_problem (n : ℕ) : n % 44 = 0 ∧ n / 44 = 432 → n % 35 = 28 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l3807_380731


namespace NUMINAMATH_CALUDE_calculate_speed_l3807_380732

/-- Given two people moving in opposite directions, calculate the speed of one person given the speed of the other and their total distance after a certain time. -/
theorem calculate_speed (riya_speed priya_speed : ℝ) (time : ℝ) (total_distance : ℝ) : 
  riya_speed = 24 →
  time = 0.75 →
  total_distance = 44.25 →
  priya_speed * time + riya_speed * time = total_distance →
  priya_speed = 35 := by
sorry

end NUMINAMATH_CALUDE_calculate_speed_l3807_380732


namespace NUMINAMATH_CALUDE_cube_piercing_theorem_l3807_380779

/-- Represents a brick with dimensions 2 × 2 × 1 -/
structure Brick :=
  (x : ℕ) (y : ℕ) (z : ℕ)

/-- Represents a cube constructed from bricks -/
structure Cube :=
  (size : ℕ)
  (bricks : List Brick)

/-- Represents a line perpendicular to a face of the cube -/
structure PerpLine :=
  (x : ℕ) (y : ℕ) (face : Nat)

/-- Function to check if a line intersects a brick -/
def intersects (l : PerpLine) (b : Brick) : Prop := sorry

/-- Theorem stating that there exists a line not intersecting any brick -/
theorem cube_piercing_theorem (c : Cube) 
  (h1 : c.size = 20) 
  (h2 : c.bricks.length = 2000) 
  (h3 : ∀ b ∈ c.bricks, b.x = 2 ∧ b.y = 2 ∧ b.z = 1) :
  ∃ l : PerpLine, ∀ b ∈ c.bricks, ¬(intersects l b) := by sorry

end NUMINAMATH_CALUDE_cube_piercing_theorem_l3807_380779


namespace NUMINAMATH_CALUDE_eight_teams_satisfy_conditions_l3807_380764

/-- The number of days in the tournament -/
def tournament_days : ℕ := 7

/-- The number of games scheduled per day -/
def games_per_day : ℕ := 4

/-- The total number of games in the tournament -/
def total_games : ℕ := tournament_days * games_per_day

/-- Function to calculate the number of games for a given number of teams -/
def games_for_teams (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem stating that 8 teams satisfy the tournament conditions -/
theorem eight_teams_satisfy_conditions : 
  ∃ (n : ℕ), n > 0 ∧ games_for_teams n = total_games ∧ n = 8 :=
sorry

end NUMINAMATH_CALUDE_eight_teams_satisfy_conditions_l3807_380764


namespace NUMINAMATH_CALUDE_intersection_of_sets_l3807_380709

theorem intersection_of_sets :
  let A : Set ℤ := {-1, 2, 4}
  let B : Set ℤ := {0, 2, 6}
  A ∩ B = {2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l3807_380709


namespace NUMINAMATH_CALUDE_concert_problem_l3807_380769

/-- Represents the number of songs sung by each friend -/
structure SongCount where
  lucy : ℕ
  gina : ℕ
  zoe : ℕ
  sara : ℕ

/-- Calculates the total number of songs performed by the trios -/
def totalSongs (sc : SongCount) : ℚ :=
  (sc.lucy + sc.gina + sc.zoe + sc.sara) / 3

/-- Represents the conditions of the problem -/
def validSongCount (sc : SongCount) : Prop :=
  sc.sara = 9 ∧
  sc.lucy = 3 ∧
  sc.zoe = sc.sara ∧
  sc.gina > sc.lucy ∧
  sc.gina ≤ sc.sara ∧
  (sc.lucy + sc.gina) % 4 = 0

theorem concert_problem (sc : SongCount) (h : validSongCount sc) :
  totalSongs sc = 9 ∨ totalSongs sc = 10 := by
  sorry


end NUMINAMATH_CALUDE_concert_problem_l3807_380769


namespace NUMINAMATH_CALUDE_triangle_angle_calculation_l3807_380715

theorem triangle_angle_calculation (A B C : Real) (a b c : Real) :
  -- Triangle ABC
  -- b = √2
  b = Real.sqrt 2 →
  -- c = 1
  c = 1 →
  -- B = 45°
  B = 45 * π / 180 →
  -- Then C = 30°
  C = 30 * π / 180 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_calculation_l3807_380715


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l3807_380762

/-- The lateral surface area of a cone with base radius 3 and slant height 4 is 12π. -/
theorem cone_lateral_surface_area :
  ∀ (r l : ℝ), r = 3 → l = 4 → π * r * l = 12 * π :=
by
  sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l3807_380762


namespace NUMINAMATH_CALUDE_connie_initial_marbles_l3807_380765

/-- The number of marbles Connie gave to Juan -/
def marbles_given : ℕ := 70

/-- The number of marbles Connie has left -/
def marbles_left : ℕ := 3

/-- The initial number of marbles Connie had -/
def initial_marbles : ℕ := marbles_given + marbles_left

theorem connie_initial_marbles : initial_marbles = 73 := by
  sorry

end NUMINAMATH_CALUDE_connie_initial_marbles_l3807_380765


namespace NUMINAMATH_CALUDE_product_pure_imaginary_l3807_380780

theorem product_pure_imaginary (x : ℝ) :
  (∃ b : ℝ, (x + 1 + Complex.I) * ((x + 2) + Complex.I) * ((x + 3) + Complex.I) = b * Complex.I) ↔ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_product_pure_imaginary_l3807_380780


namespace NUMINAMATH_CALUDE_probability_red_ball_is_two_fifths_l3807_380721

/-- The probability of drawing a red ball from a bag -/
def probability_red_ball (total_balls : ℕ) (red_balls : ℕ) : ℚ :=
  red_balls / total_balls

/-- The total number of balls in the bag -/
def total_balls : ℕ := 5

/-- The number of red balls in the bag -/
def red_balls : ℕ := 2

/-- The number of white balls in the bag -/
def white_balls : ℕ := 3

theorem probability_red_ball_is_two_fifths :
  probability_red_ball total_balls red_balls = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_red_ball_is_two_fifths_l3807_380721


namespace NUMINAMATH_CALUDE_sum_of_integers_l3807_380705

theorem sum_of_integers (a b c d e : ℤ) 
  (eq1 : a - b + c - e = 7)
  (eq2 : b - c + d + e = 8)
  (eq3 : c - d + a - e = 4)
  (eq4 : d - a + b + e = 3) :
  a + b + c + d + e = 22 := by
sorry

end NUMINAMATH_CALUDE_sum_of_integers_l3807_380705


namespace NUMINAMATH_CALUDE_line_parabola_intersection_count_l3807_380751

theorem line_parabola_intersection_count : 
  ∃! (s : Finset ℝ), 
    (∀ a ∈ s, ∃ x y : ℝ, 
      y = 2*x + a + 1 ∧ 
      y = x^2 + (a+1)^2 ∧ 
      ∀ x' : ℝ, x'^2 + (a+1)^2 ≥ x^2 + (a+1)^2) ∧
    s.card = 2 := by
  sorry

end NUMINAMATH_CALUDE_line_parabola_intersection_count_l3807_380751


namespace NUMINAMATH_CALUDE_max_difference_correct_l3807_380736

/-- Represents a convex N-gon divided into triangles by non-intersecting diagonals --/
structure ConvexNgon (N : ℕ) where
  triangles : ℕ
  diagonals : ℕ
  is_valid : triangles = N - 2 ∧ diagonals = N - 3

/-- Represents a coloring of the triangles in the N-gon --/
structure Coloring (N : ℕ) where
  ngon : ConvexNgon N
  white_triangles : ℕ
  black_triangles : ℕ
  is_valid : white_triangles + black_triangles = ngon.triangles
  adjacent_different : True  -- Represents the condition that adjacent triangles have different colors

/-- The maximum difference between white and black triangles for a given N --/
def max_difference (N : ℕ) : ℕ :=
  if N % 3 = 1 then
    N / 3 - 1
  else
    N / 3

/-- The theorem stating the maximum difference between white and black triangles --/
theorem max_difference_correct (N : ℕ) (c : Coloring N) :
  (c.white_triangles : ℤ) - (c.black_triangles : ℤ) ≤ max_difference N := by
  sorry

end NUMINAMATH_CALUDE_max_difference_correct_l3807_380736


namespace NUMINAMATH_CALUDE_probability_less_equal_nine_l3807_380771

def card_set : Finset ℕ := {1, 3, 4, 6, 7, 9}

theorem probability_less_equal_nine : 
  (card_set.filter (λ x => x ≤ 9)).card / card_set.card = 1 := by
  sorry

end NUMINAMATH_CALUDE_probability_less_equal_nine_l3807_380771


namespace NUMINAMATH_CALUDE_division_of_monomials_l3807_380768

theorem division_of_monomials (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  10 * a^3 * b^2 / (-5 * a^2 * b) = -2 * a * b :=
by sorry

end NUMINAMATH_CALUDE_division_of_monomials_l3807_380768


namespace NUMINAMATH_CALUDE_cubic_function_properties_l3807_380799

/-- A cubic function with specific properties -/
def f (a b c : ℝ) (x : ℝ) : ℝ := x^3 + 3*a*x^2 + 3*b*x + c

/-- The derivative of f -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 6*a*x + 3*b

theorem cubic_function_properties (a b c : ℝ) :
  (∃ (y : ℝ), f' a b 2 = 0) ∧ 
  (∃ (m : ℝ), f' a b 1 = m ∧ m = -3) →
  (a = -1 ∧ b = 0) ∧ 
  (∃ (x_max x_min : ℝ), f (-1) 0 c x_max - f (-1) 0 c x_min = 4) :=
by sorry

end NUMINAMATH_CALUDE_cubic_function_properties_l3807_380799


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l3807_380747

theorem arithmetic_expression_equality : 9 - 8 + 7 * 6 + 5 - 4 * 3 + 2 - 1 = 37 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l3807_380747


namespace NUMINAMATH_CALUDE_fruit_shop_apples_l3807_380700

/-- Given the ratio of mangoes : oranges : apples and the number of mangoes,
    calculate the number of apples -/
theorem fruit_shop_apples (ratio_mangoes ratio_oranges ratio_apples num_mangoes : ℕ) 
    (h_ratio : ratio_mangoes = 10 ∧ ratio_oranges = 2 ∧ ratio_apples = 3)
    (h_mangoes : num_mangoes = 120) :
    (num_mangoes / ratio_mangoes) * ratio_apples = 36 := by
  sorry

#check fruit_shop_apples

end NUMINAMATH_CALUDE_fruit_shop_apples_l3807_380700


namespace NUMINAMATH_CALUDE_sqrt_81_equals_3_to_m_l3807_380766

theorem sqrt_81_equals_3_to_m (m : ℝ) : (81 : ℝ)^(1/2) = 3^m → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_81_equals_3_to_m_l3807_380766


namespace NUMINAMATH_CALUDE_total_days_on_island_l3807_380754

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- The duration of the first expedition in weeks -/
def first_expedition : ℕ := 3

/-- Calculates the duration of the second expedition in weeks -/
def second_expedition : ℕ := first_expedition + 2

/-- Calculates the duration of the last expedition in weeks -/
def last_expedition : ℕ := 2 * second_expedition

/-- Calculates the total number of weeks spent on all expeditions -/
def total_weeks : ℕ := first_expedition + second_expedition + last_expedition

/-- Theorem stating the total number of days spent on the island -/
theorem total_days_on_island : total_weeks * days_per_week = 126 := by
  sorry

end NUMINAMATH_CALUDE_total_days_on_island_l3807_380754


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l3807_380744

def U : Set ℕ := {1,2,3,4,5,6}
def A : Set ℕ := {2,4,6}

theorem complement_of_A_in_U :
  (U \ A) = {1,3,5} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l3807_380744


namespace NUMINAMATH_CALUDE_first_five_terms_sequence_1_l3807_380726

def a (n : ℕ+) : ℚ := 1 / (4 * n - 1)

theorem first_five_terms_sequence_1 :
  [a 1, a 2, a 3, a 4, a 5] = [1/3, 1/7, 1/11, 1/15, 1/19] := by sorry

end NUMINAMATH_CALUDE_first_five_terms_sequence_1_l3807_380726


namespace NUMINAMATH_CALUDE_tank_fill_time_l3807_380724

/-- Represents the state of the tank and pipes -/
structure TankSystem where
  pipeA : ℝ  -- Rate at which Pipe A fills the tank (fraction of tank per minute)
  pipeB : ℝ  -- Rate at which Pipe B empties the tank (fraction of tank per minute)
  closeBTime : ℝ  -- Time at which Pipe B is closed (in minutes)

/-- Calculates the time taken to fill the tank given the tank system parameters -/
def timeTakenToFill (system : TankSystem) : ℝ :=
  sorry

/-- Theorem stating that for the given conditions, the tank will be filled in 70 minutes -/
theorem tank_fill_time (system : TankSystem) 
  (hA : system.pipeA = 1 / 8)
  (hB : system.pipeB = 1 / 24)
  (hClose : system.closeBTime = 66) :
  timeTakenToFill system = 70 :=
sorry

end NUMINAMATH_CALUDE_tank_fill_time_l3807_380724


namespace NUMINAMATH_CALUDE_stratified_sampling_middle_aged_l3807_380794

theorem stratified_sampling_middle_aged (total_teachers : ℕ) (middle_aged : ℕ) (sample_size : ℕ)
  (h1 : total_teachers = 480)
  (h2 : middle_aged = 160)
  (h3 : sample_size = 60) :
  (middle_aged : ℚ) / total_teachers * sample_size = 20 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_middle_aged_l3807_380794


namespace NUMINAMATH_CALUDE_parallelogram_count_in_triangular_grid_l3807_380798

/-- Represents a triangular grid formed by subdividing an equilateral triangle -/
structure TriangularGrid where
  sideLength : ℕ
  smallTriangles : ℕ
  smallSideLength : ℕ

/-- Counts the number of parallelograms in a triangular grid -/
def countParallelograms (grid : TriangularGrid) : ℕ :=
  3 * Nat.choose (grid.sideLength + 2) 4

/-- Theorem stating the correct count of parallelograms in a triangular grid -/
theorem parallelogram_count_in_triangular_grid (grid : TriangularGrid) 
  (h1 : grid.smallTriangles = grid.sideLength ^ 2)
  (h2 : grid.smallSideLength = 1) :
  countParallelograms grid = 3 * Nat.choose (grid.sideLength + 2) 4 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_count_in_triangular_grid_l3807_380798


namespace NUMINAMATH_CALUDE_bridge_length_l3807_380711

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 110 →
  train_speed_kmh = 45 →
  crossing_time = 30 →
  ∃ (bridge_length : ℝ),
    bridge_length = 265 ∧
    bridge_length + train_length = (train_speed_kmh * 1000 / 3600) * crossing_time :=
by
  sorry

end NUMINAMATH_CALUDE_bridge_length_l3807_380711


namespace NUMINAMATH_CALUDE_perfect_square_expression_l3807_380727

theorem perfect_square_expression (x : ℝ) : ∃ y : ℝ, x^2 - x + (1/4 : ℝ) = y^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_expression_l3807_380727


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l3807_380759

theorem pure_imaginary_complex_number (m : ℝ) : 
  let z : ℂ := (1 + m * Complex.I) * (2 - Complex.I)
  (∃ (y : ℝ), z = Complex.I * y) → m = -2 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l3807_380759


namespace NUMINAMATH_CALUDE_min_sugar_amount_l3807_380796

theorem min_sugar_amount (f s : ℕ) : 
  (f ≥ 9 + s / 2) → 
  (f ≤ 3 * s) → 
  (∃ (f : ℕ), f ≥ 9 + s / 2 ∧ f ≤ 3 * s) → 
  s ≥ 4 := by
sorry

end NUMINAMATH_CALUDE_min_sugar_amount_l3807_380796


namespace NUMINAMATH_CALUDE_xiaoying_final_score_l3807_380778

/-- Calculates the weighted sum of scores given the scores and weights -/
def weightedSum (scores : List ℝ) (weights : List ℝ) : ℝ :=
  List.sum (List.zipWith (· * ·) scores weights)

/-- Xiaoying's speech competition scores -/
def speechScores : List ℝ := [86, 90, 80]

/-- Weights for each category in the speech competition -/
def categoryWeights : List ℝ := [0.5, 0.4, 0.1]

/-- Theorem stating that Xiaoying's final score is 87 -/
theorem xiaoying_final_score :
  weightedSum speechScores categoryWeights = 87 := by
  sorry

end NUMINAMATH_CALUDE_xiaoying_final_score_l3807_380778


namespace NUMINAMATH_CALUDE_indexCardsPerStudentIs10_l3807_380763

/-- Calculates the number of index cards each student receives given the following conditions:
  * Carl teaches 6 periods a day
  * Each class has 30 students
  * A 50 pack of index cards costs $3
  * Carl spent $108 on index cards
-/
def indexCardsPerStudent (periods : Nat) (studentsPerClass : Nat) (cardsPerPack : Nat) 
  (costPerPack : Nat) (totalSpent : Nat) : Nat :=
  let totalPacks := totalSpent / costPerPack
  let totalCards := totalPacks * cardsPerPack
  let totalStudents := periods * studentsPerClass
  totalCards / totalStudents

theorem indexCardsPerStudentIs10 : 
  indexCardsPerStudent 6 30 50 3 108 = 10 := by
  sorry

end NUMINAMATH_CALUDE_indexCardsPerStudentIs10_l3807_380763


namespace NUMINAMATH_CALUDE_roberts_score_l3807_380749

/-- Proves that Robert's score is 94 given the conditions of the problem -/
theorem roberts_score (total_students : ℕ) (first_19_avg : ℚ) (new_avg : ℚ) : 
  total_students = 20 → 
  first_19_avg = 74 → 
  new_avg = 75 → 
  (total_students - 1) * first_19_avg + 94 = total_students * new_avg :=
by sorry

end NUMINAMATH_CALUDE_roberts_score_l3807_380749


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l3807_380738

theorem arithmetic_calculation : 8 / 4 + 5 * 2^2 - (3 + 7) = 12 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l3807_380738


namespace NUMINAMATH_CALUDE_output_after_five_years_l3807_380773

/-- The output value after n years of growth at a given rate -/
def output_after_n_years (initial_value : ℝ) (growth_rate : ℝ) (n : ℕ) : ℝ :=
  initial_value * (1 + growth_rate) ^ n

/-- Theorem: The output value after 5 years with 10% annual growth -/
theorem output_after_five_years (a : ℝ) :
  output_after_n_years a 0.1 5 = 1.1^5 * a := by
  sorry

end NUMINAMATH_CALUDE_output_after_five_years_l3807_380773


namespace NUMINAMATH_CALUDE_complex_square_equation_l3807_380714

theorem complex_square_equation : 
  ∀ z : ℂ, z^2 = -57 - 48*I ↔ z = 3 - 8*I ∨ z = -3 + 8*I :=
by sorry

end NUMINAMATH_CALUDE_complex_square_equation_l3807_380714


namespace NUMINAMATH_CALUDE_matrix_product_is_zero_l3807_380795

def A (k a c d : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, k * d, -k * c],
    ![-k * d, 0, k * a],
    ![k * c, -k * a, 0]]

def B (d e f : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![d^2, d * e, d * f],
    ![d * e, e^2, e * f],
    ![d * f, e * f, f^2]]

theorem matrix_product_is_zero (k a c d e f : ℝ) :
  A k a c d * B d e f = 0 := by
  sorry

end NUMINAMATH_CALUDE_matrix_product_is_zero_l3807_380795


namespace NUMINAMATH_CALUDE_bianca_drawing_time_l3807_380722

/-- The total time Bianca spent drawing is equal to 41 minutes, given that she spent 22 minutes drawing at school and 19 minutes drawing at home. -/
theorem bianca_drawing_time (time_at_school time_at_home : ℕ) 
  (h1 : time_at_school = 22)
  (h2 : time_at_home = 19) :
  time_at_school + time_at_home = 41 := by
  sorry

end NUMINAMATH_CALUDE_bianca_drawing_time_l3807_380722


namespace NUMINAMATH_CALUDE_expression_equals_three_l3807_380791

-- Define the expression
def expression : ℚ := -25 + 7 * ((8 / 4) ^ 2)

-- Theorem statement
theorem expression_equals_three : expression = 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_three_l3807_380791


namespace NUMINAMATH_CALUDE_current_trees_count_l3807_380761

/-- The number of dogwood trees to be planted today -/
def trees_planted_today : ℕ := 41

/-- The number of dogwood trees to be planted tomorrow -/
def trees_planted_tomorrow : ℕ := 20

/-- The total number of dogwood trees after planting -/
def total_trees_after_planting : ℕ := 100

/-- The number of dogwood trees currently in the park -/
def current_trees : ℕ := total_trees_after_planting - trees_planted_today - trees_planted_tomorrow

theorem current_trees_count : current_trees = 39 := by
  sorry

end NUMINAMATH_CALUDE_current_trees_count_l3807_380761


namespace NUMINAMATH_CALUDE_stratified_sampling_group_b_l3807_380756

theorem stratified_sampling_group_b (total_cities : ℕ) (group_b_cities : ℕ) (sample_size : ℕ) :
  total_cities = 24 →
  group_b_cities = 12 →
  sample_size = 6 →
  (group_b_cities * sample_size) / total_cities = 3 :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_group_b_l3807_380756


namespace NUMINAMATH_CALUDE_tangent_circle_existence_and_radius_l3807_380712

/-- Given three circles with radii r₁, r₂, r₃, where r₁ > r₂ and r₁ > r₃,
    there exists a circle touching the four tangents drawn as described,
    with radius (r₁ * r₂ * r₃) / (r₁ * (r₂ + r₃) - r₂ * r₃) -/
theorem tangent_circle_existence_and_radius 
  (r₁ r₂ r₃ : ℝ) 
  (h₁ : r₁ > r₂) 
  (h₂ : r₁ > r₃) 
  (h₃ : r₁ > 0) 
  (h₄ : r₂ > 0) 
  (h₅ : r₃ > 0) :
  ∃ (r : ℝ), r = (r₁ * r₂ * r₃) / (r₁ * (r₂ + r₃) - r₂ * r₃) ∧ 
  r > 0 :=
sorry

end NUMINAMATH_CALUDE_tangent_circle_existence_and_radius_l3807_380712


namespace NUMINAMATH_CALUDE_abs_sum_lower_bound_l3807_380788

theorem abs_sum_lower_bound :
  (∀ x : ℝ, |x - 4| + |x - 6| ≥ 2) ∧
  (∀ ε > 0, ∃ x : ℝ, |x - 4| + |x - 6| < 2 + ε) := by
  sorry

end NUMINAMATH_CALUDE_abs_sum_lower_bound_l3807_380788


namespace NUMINAMATH_CALUDE_range_of_x_for_sqrt_4_minus_x_l3807_380728

theorem range_of_x_for_sqrt_4_minus_x : 
  ∀ x : ℝ, (∃ y : ℝ, y^2 = 4 - x) ↔ x ≤ 4 := by sorry

end NUMINAMATH_CALUDE_range_of_x_for_sqrt_4_minus_x_l3807_380728


namespace NUMINAMATH_CALUDE_definite_integral_cos_zero_l3807_380716

theorem definite_integral_cos_zero : 
  ∫ x in (π/4)..(9*π/4), Real.sqrt 2 * Real.cos (2*x + π/4) = 0 := by sorry

end NUMINAMATH_CALUDE_definite_integral_cos_zero_l3807_380716


namespace NUMINAMATH_CALUDE_election_winner_percentage_l3807_380789

theorem election_winner_percentage (total_votes : ℕ) (winning_margin : ℕ) :
  total_votes = 900 →
  winning_margin = 360 →
  ∃ (winning_percentage : ℚ),
    winning_percentage = 70 / 100 ∧
    (winning_percentage * total_votes : ℚ) - ((1 - winning_percentage) * total_votes : ℚ) = winning_margin :=
by sorry

end NUMINAMATH_CALUDE_election_winner_percentage_l3807_380789


namespace NUMINAMATH_CALUDE_bakery_doughnuts_given_away_l3807_380757

/-- Given a bakery scenario, prove the number of doughnuts given away -/
theorem bakery_doughnuts_given_away
  (total_doughnuts : ℕ)
  (doughnuts_per_box : ℕ)
  (boxes_sold : ℕ)
  (h1 : total_doughnuts = 300)
  (h2 : doughnuts_per_box = 10)
  (h3 : boxes_sold = 27)
  : (total_doughnuts - boxes_sold * doughnuts_per_box) = 30 :=
by sorry

end NUMINAMATH_CALUDE_bakery_doughnuts_given_away_l3807_380757


namespace NUMINAMATH_CALUDE_factor_quadratic_l3807_380792

theorem factor_quadratic (a : ℝ) : a^2 - 2*a - 15 = (a + 3)*(a - 5) := by
  sorry

end NUMINAMATH_CALUDE_factor_quadratic_l3807_380792


namespace NUMINAMATH_CALUDE_impossibility_proof_l3807_380774

def Square := Fin 4 → ℕ

def initial_state : Square := fun i => if i = 0 then 1 else 0

def S (state : Square) : ℤ :=
  state 0 - state 1 + state 2 - state 3

def is_valid_move (before after : Square) : Prop :=
  ∃ (i : Fin 4) (k : ℕ), 
    after i + k = before i ∧
    after ((i + 1) % 4) = before ((i + 1) % 4) + k ∧
    after ((i + 3) % 4) = before ((i + 3) % 4) + k ∧
    (∀ j, j ≠ i ∧ j ≠ (i + 1) % 4 ∧ j ≠ (i + 3) % 4 → after j = before j)

def reachable (start goal : Square) : Prop :=
  ∃ (n : ℕ) (path : Fin (n + 1) → Square),
    path 0 = start ∧
    path n = goal ∧
    ∀ i : Fin n, is_valid_move (path i) (path (i + 1))

def target_state : Square := fun i => 
  if i = 0 then 1
  else if i = 1 then 9
  else if i = 2 then 8
  else 9

theorem impossibility_proof :
  ¬(reachable initial_state target_state) :=
sorry

end NUMINAMATH_CALUDE_impossibility_proof_l3807_380774


namespace NUMINAMATH_CALUDE_sarah_apple_ratio_l3807_380717

theorem sarah_apple_ratio : 
  let sarah_apples : ℝ := 45.0
  let brother_apples : ℝ := 9.0
  sarah_apples / brother_apples = 5 := by
sorry

end NUMINAMATH_CALUDE_sarah_apple_ratio_l3807_380717


namespace NUMINAMATH_CALUDE_polynomial_root_relation_l3807_380793

/-- Two monic cubic polynomials with specified roots and a relation between them -/
theorem polynomial_root_relation (s : ℝ) (h j : ℝ → ℝ) : 
  (∀ x, h x = (x - (s + 2)) * (x - (s + 6)) * (x - c)) →
  (∀ x, j x = (x - (s + 4)) * (x - (s + 8)) * (x - d)) →
  (∀ x, 2 * (h x - j x) = s) →
  s = 64 := by sorry

end NUMINAMATH_CALUDE_polynomial_root_relation_l3807_380793


namespace NUMINAMATH_CALUDE_bbq_guests_count_l3807_380730

/-- Represents the BBQ scenario with given parameters -/
structure BBQ where
  cook_time_per_side : ℕ  -- cooking time for one side of a burger in minutes
  grill_capacity : ℕ      -- number of burgers that can be cooked simultaneously
  total_cook_time : ℕ     -- total time spent cooking all burgers in minutes

/-- Calculates the number of guests at the BBQ -/
def number_of_guests (bbq : BBQ) : ℕ :=
  let total_burgers := (bbq.total_cook_time / (2 * bbq.cook_time_per_side)) * bbq.grill_capacity
  (2 * total_burgers) / 3

/-- Theorem stating that the number of guests at the BBQ is 30 -/
theorem bbq_guests_count (bbq : BBQ) 
  (h1 : bbq.cook_time_per_side = 4)
  (h2 : bbq.grill_capacity = 5)
  (h3 : bbq.total_cook_time = 72) :
  number_of_guests bbq = 30 := by
  sorry

#eval number_of_guests ⟨4, 5, 72⟩

end NUMINAMATH_CALUDE_bbq_guests_count_l3807_380730


namespace NUMINAMATH_CALUDE_late_car_speed_l3807_380734

/-- Proves that given a journey of 70 km, if a car arrives on time with an average speed
    of 40 km/hr and arrives 15 minutes late with a slower speed, then the slower speed
    is 35 km/hr. -/
theorem late_car_speed (distance : ℝ) (on_time_speed : ℝ) (late_time : ℝ) :
  distance = 70 →
  on_time_speed = 40 →
  late_time = 0.25 →
  let on_time_duration := distance / on_time_speed
  let late_duration := on_time_duration + late_time
  let late_speed := distance / late_duration
  late_speed = 35 := by
  sorry

end NUMINAMATH_CALUDE_late_car_speed_l3807_380734


namespace NUMINAMATH_CALUDE_no_real_solution_l3807_380719

theorem no_real_solution :
  ¬ ∃ (x : ℝ), x^2 - 4*x + 4 < 0 :=
by sorry

end NUMINAMATH_CALUDE_no_real_solution_l3807_380719


namespace NUMINAMATH_CALUDE_a_share_of_profit_l3807_380770

/-- Calculate A's share of the profit in a partnership business -/
theorem a_share_of_profit (a_investment b_investment c_investment total_profit : ℕ) :
  a_investment = 6300 →
  b_investment = 4200 →
  c_investment = 10500 →
  total_profit = 12700 →
  (a_investment * total_profit) / (a_investment + b_investment + c_investment) = 3810 :=
by sorry

end NUMINAMATH_CALUDE_a_share_of_profit_l3807_380770


namespace NUMINAMATH_CALUDE_units_digit_of_2_pow_2012_l3807_380702

theorem units_digit_of_2_pow_2012 : ∃ n : ℕ, 2^2012 ≡ 6 [ZMOD 10] := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_2_pow_2012_l3807_380702


namespace NUMINAMATH_CALUDE_term_2005_is_334th_l3807_380733

-- Define the arithmetic sequence
def arithmeticSequence (n : ℕ) : ℕ := 7 + 6 * (n - 1)

-- State the theorem
theorem term_2005_is_334th :
  arithmeticSequence 334 = 2005 := by
  sorry

end NUMINAMATH_CALUDE_term_2005_is_334th_l3807_380733


namespace NUMINAMATH_CALUDE_intersection_points_l3807_380797

/-- A periodic function with period 2 that equals x^2 on [-1, 1] -/
noncomputable def f : ℝ → ℝ := sorry

/-- The number of intersection points between f and |log₅(x)| -/
def num_intersections : ℕ := sorry

theorem intersection_points :
  (∀ x, f (x + 2) = f x) ∧ 
  (∀ x ∈ Set.Icc (-1) 1, f x = x^2) →
  num_intersections = 5 := by sorry

end NUMINAMATH_CALUDE_intersection_points_l3807_380797


namespace NUMINAMATH_CALUDE_number_count_l3807_380710

theorem number_count (average : ℝ) (sum_of_three : ℝ) (average_of_two : ℝ) (n : ℕ) : 
  average = 20 →
  sum_of_three = 48 →
  average_of_two = 26 →
  (average * n : ℝ) = sum_of_three + 2 * average_of_two →
  n = 5 := by
sorry

end NUMINAMATH_CALUDE_number_count_l3807_380710


namespace NUMINAMATH_CALUDE_regression_increase_l3807_380784

/-- Linear regression equation for annual food expenditure with respect to annual income -/
def regression_equation (x : ℝ) : ℝ := 0.254 * x + 0.321

/-- Theorem stating that the increase in the regression equation's output for a 1 unit increase in input is 0.254 -/
theorem regression_increase : ∀ x : ℝ, regression_equation (x + 1) - regression_equation x = 0.254 := by
  sorry

end NUMINAMATH_CALUDE_regression_increase_l3807_380784


namespace NUMINAMATH_CALUDE_degree_of_x2y3_is_five_l3807_380760

/-- The degree of a monomial is the sum of the exponents of its variables -/
def degree_of_monomial (x_exp y_exp : ℕ) : ℕ := x_exp + y_exp

/-- Theorem: The degree of the monomial x^2 * y^3 is 5 -/
theorem degree_of_x2y3_is_five : degree_of_monomial 2 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_degree_of_x2y3_is_five_l3807_380760


namespace NUMINAMATH_CALUDE_sufficient_condition_quadratic_inequality_l3807_380783

theorem sufficient_condition_quadratic_inequality (m : ℝ) :
  (m ≥ 2) →
  (∀ x : ℝ, x^2 - 2*x + m ≥ 0) ∧
  ¬(∀ m : ℝ, (∀ x : ℝ, x^2 - 2*x + m ≥ 0) → m ≥ 2) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_condition_quadratic_inequality_l3807_380783


namespace NUMINAMATH_CALUDE_original_price_from_discounted_l3807_380787

/-- 
Given a product with an original price, this theorem proves that 
if the price after successive discounts of 15% and 25% is 306, 
then the original price was 480.
-/
theorem original_price_from_discounted (original_price : ℝ) : 
  (1 - 0.25) * (1 - 0.15) * original_price = 306 → original_price = 480 := by
  sorry

end NUMINAMATH_CALUDE_original_price_from_discounted_l3807_380787


namespace NUMINAMATH_CALUDE_factorization_equality_l3807_380703

theorem factorization_equality (a b : ℝ) : a * b^3 - 4 * a * b = a * b * (b + 2) * (b - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3807_380703


namespace NUMINAMATH_CALUDE_parabola_equation_l3807_380707

/-- Represents a parabola in standard form -/
structure Parabola where
  p : ℝ
  axis : Bool  -- True for vertical axis, False for horizontal axis

/-- The hyperbola from the problem statement -/
def hyperbola : Set (ℝ × ℝ) :=
  {(x, y) | 16 * x^2 - 9 * y^2 = 144}

/-- The theorem statement -/
theorem parabola_equation (P : Parabola) :
  P.axis = true ∧  -- Vertical axis of symmetry
  (∀ (x y : ℝ), (x, y) ∈ hyperbola → x^2 = 9 ∧ y^2 = 16) ∧  -- Hyperbola properties
  (0, 0) ∈ hyperbola ∧  -- Vertex at origin
  (-3, 0) ∈ hyperbola ∧  -- Left vertex of hyperbola
  P.p = 6  -- Distance from vertex to directrix is 3
  →
  ∀ (x y : ℝ), y^2 = 2 * P.p * x ↔ y^2 = 12 * x := by
  sorry

end NUMINAMATH_CALUDE_parabola_equation_l3807_380707


namespace NUMINAMATH_CALUDE_disprove_seventh_power_conjecture_l3807_380748

theorem disprove_seventh_power_conjecture :
  144^7 + 110^7 + 84^7 + 27^7 = 206^7 := by
  sorry

end NUMINAMATH_CALUDE_disprove_seventh_power_conjecture_l3807_380748


namespace NUMINAMATH_CALUDE_conic_is_ellipse_with_major_axis_8_l3807_380753

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A conic section passing through five points -/
structure ConicSection where
  p1 : Point
  p2 : Point
  p3 : Point
  p4 : Point
  p5 : Point

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

/-- The conic section defined by the given five points -/
def givenConic : ConicSection :=
  { p1 := { x := -2, y := 0 }
  , p2 := { x := 0,  y := 1 }
  , p3 := { x := 0,  y := 3 }
  , p4 := { x := 4,  y := 1 }
  , p5 := { x := 4,  y := 3 }
  }

/-- Definition of an ellipse -/
def isEllipse (c : ConicSection) : Prop :=
  ∃ (center : Point) (a b : ℝ),
    a > 0 ∧ b > 0 ∧
    ∀ (p : Point),
      ((p.x - center.x)^2 / a^2 + (p.y - center.y)^2 / b^2 = 1) ↔
      (p = c.p1 ∨ p = c.p2 ∨ p = c.p3 ∨ p = c.p4 ∨ p = c.p5)

/-- The main theorem -/
theorem conic_is_ellipse_with_major_axis_8 :
  (¬ collinear givenConic.p1 givenConic.p2 givenConic.p3) ∧
  (¬ collinear givenConic.p1 givenConic.p2 givenConic.p4) ∧
  (¬ collinear givenConic.p1 givenConic.p2 givenConic.p5) ∧
  (¬ collinear givenConic.p1 givenConic.p3 givenConic.p4) ∧
  (¬ collinear givenConic.p1 givenConic.p3 givenConic.p5) ∧
  (¬ collinear givenConic.p1 givenConic.p4 givenConic.p5) ∧
  (¬ collinear givenConic.p2 givenConic.p3 givenConic.p4) ∧
  (¬ collinear givenConic.p2 givenConic.p3 givenConic.p5) ∧
  (¬ collinear givenConic.p2 givenConic.p4 givenConic.p5) ∧
  (¬ collinear givenConic.p3 givenConic.p4 givenConic.p5) →
  isEllipse givenConic ∧ ∃ (a : ℝ), a = 8 :=
by sorry

end NUMINAMATH_CALUDE_conic_is_ellipse_with_major_axis_8_l3807_380753


namespace NUMINAMATH_CALUDE_starting_lineup_count_l3807_380735

def team_size : ℕ := 12
def center_capable : ℕ := 4

def starting_lineup_combinations : ℕ :=
  center_capable * (team_size - 1) * (team_size - 2) * (team_size - 3)

theorem starting_lineup_count :
  starting_lineup_combinations = 3960 := by
  sorry

end NUMINAMATH_CALUDE_starting_lineup_count_l3807_380735


namespace NUMINAMATH_CALUDE_furniture_cost_l3807_380767

/-- Prove that the cost of the furniture is $400, given the conditions of Emma's spending. -/
theorem furniture_cost (initial_amount : ℝ) (remaining_amount : ℝ) 
  (h1 : initial_amount = 2000)
  (h2 : remaining_amount = 400)
  (h3 : ∃ (furniture_cost : ℝ), remaining_amount = (1/4) * (initial_amount - furniture_cost)) :
  ∃ (furniture_cost : ℝ), furniture_cost = 400 := by
  sorry

end NUMINAMATH_CALUDE_furniture_cost_l3807_380767


namespace NUMINAMATH_CALUDE_contradiction_proof_l3807_380704

theorem contradiction_proof (a b c d : ℝ) 
  (h1 : a + b = 1) 
  (h2 : c + d = 1) 
  (h3 : a * c + b * d > 1) : 
  ¬(0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d) := by
  sorry

end NUMINAMATH_CALUDE_contradiction_proof_l3807_380704


namespace NUMINAMATH_CALUDE_oil_measurement_l3807_380752

/-- The total amount of oil in a measuring cup after adding more -/
theorem oil_measurement (initial : ℚ) (additional : ℚ) : 
  initial = 0.16666666666666666 →
  additional = 0.6666666666666666 →
  initial + additional = 0.8333333333333333 := by
  sorry

end NUMINAMATH_CALUDE_oil_measurement_l3807_380752


namespace NUMINAMATH_CALUDE_siblings_water_consumption_l3807_380713

def cups_per_week (daily_cups : ℕ) : ℕ := daily_cups * 7

theorem siblings_water_consumption :
  let theo_daily := 8
  let mason_daily := 7
  let roxy_daily := 9
  let zara_daily := 10
  let lily_daily := 6
  cups_per_week theo_daily +
  cups_per_week mason_daily +
  cups_per_week roxy_daily +
  cups_per_week zara_daily +
  cups_per_week lily_daily = 280 :=
by
  sorry

end NUMINAMATH_CALUDE_siblings_water_consumption_l3807_380713


namespace NUMINAMATH_CALUDE_oil_bottles_volume_l3807_380741

theorem oil_bottles_volume :
  let total_bottles : ℕ := 60
  let bottles_250ml : ℕ := 20
  let bottles_300ml : ℕ := 25
  let bottles_350ml : ℕ := total_bottles - bottles_250ml - bottles_300ml
  let volume_250ml : ℕ := 250
  let volume_300ml : ℕ := 300
  let volume_350ml : ℕ := 350
  let total_volume_ml : ℕ := bottles_250ml * volume_250ml + bottles_300ml * volume_300ml + bottles_350ml * volume_350ml
  let ml_per_liter : ℕ := 1000
  total_volume_ml / ml_per_liter = (17750 : ℚ) / 1000 := by
  sorry

end NUMINAMATH_CALUDE_oil_bottles_volume_l3807_380741


namespace NUMINAMATH_CALUDE_allowance_spent_on_books_l3807_380781

theorem allowance_spent_on_books (total : ℚ) (games snacks toys books : ℚ) : 
  total = 45 → 
  games = 2/9 * total → 
  snacks = 1/3 * total → 
  toys = 1/5 * total → 
  books = total - (games + snacks + toys) → 
  books = 11 := by
sorry

end NUMINAMATH_CALUDE_allowance_spent_on_books_l3807_380781


namespace NUMINAMATH_CALUDE_tablet_price_after_discounts_l3807_380776

theorem tablet_price_after_discounts (original_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) :
  original_price = 250 ∧ discount1 = 0.30 ∧ discount2 = 0.25 →
  original_price * (1 - discount1) * (1 - discount2) = 131.25 := by
  sorry

end NUMINAMATH_CALUDE_tablet_price_after_discounts_l3807_380776


namespace NUMINAMATH_CALUDE_bird_cost_problem_l3807_380723

/-- The cost of birds in a pet store -/
theorem bird_cost_problem (small_bird_cost big_bird_cost : ℚ) : 
  big_bird_cost = 2 * small_bird_cost →
  5 * big_bird_cost + 3 * small_bird_cost = 5 * small_bird_cost + 3 * big_bird_cost + 20 →
  small_bird_cost = 10 ∧ big_bird_cost = 20 := by
  sorry

end NUMINAMATH_CALUDE_bird_cost_problem_l3807_380723


namespace NUMINAMATH_CALUDE_rooms_per_hall_first_wing_is_32_l3807_380786

/-- Represents a hotel with two wings -/
structure Hotel where
  total_rooms : ℕ
  first_wing_floors : ℕ
  first_wing_halls_per_floor : ℕ
  second_wing_floors : ℕ
  second_wing_halls_per_floor : ℕ
  second_wing_rooms_per_hall : ℕ

/-- Calculates the number of rooms in each hall of the first wing -/
def rooms_per_hall_first_wing (h : Hotel) : ℕ :=
  let second_wing_rooms := h.second_wing_floors * h.second_wing_halls_per_floor * h.second_wing_rooms_per_hall
  let first_wing_rooms := h.total_rooms - second_wing_rooms
  let total_halls_first_wing := h.first_wing_floors * h.first_wing_halls_per_floor
  first_wing_rooms / total_halls_first_wing

/-- Theorem stating that for the given hotel configuration, 
    each hall in the first wing has 32 rooms -/
theorem rooms_per_hall_first_wing_is_32 :
  rooms_per_hall_first_wing {
    total_rooms := 4248,
    first_wing_floors := 9,
    first_wing_halls_per_floor := 6,
    second_wing_floors := 7,
    second_wing_halls_per_floor := 9,
    second_wing_rooms_per_hall := 40
  } = 32 := by
  sorry

end NUMINAMATH_CALUDE_rooms_per_hall_first_wing_is_32_l3807_380786


namespace NUMINAMATH_CALUDE_part_one_part_two_l3807_380750

-- Define the function f
def f (x a b : ℝ) : ℝ := x^2 - (a + 1) * x + b

-- Part 1
theorem part_one (a : ℝ) :
  (∃ x ∈ Set.Icc 2 3, f x a (-1) = 0) →
  (1/2 : ℝ) ≤ a ∧ a ≤ 5/3 := by sorry

-- Part 2
theorem part_two (x : ℝ) :
  (∀ a ∈ Set.Icc 2 3, f x a a < 0) →
  1 < x ∧ x < 2 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3807_380750


namespace NUMINAMATH_CALUDE_reading_6005_is_not_six_thousand_zero_zero_five_l3807_380720

/-- Rules for reading integers within ten thousand -/
structure ReadingRules where
  thousandths : ℕ → String
  hundreds : ℕ → String
  tens : ℕ → String
  ones : ℕ → String
  skip_trailing_zeros : Bool
  combine_consecutive_zeros : Bool

/-- Function to read a number according to the given rules -/
def read_number (n : ℕ) (rules : ReadingRules) : String :=
  sorry

/-- The correct reading of 6005 -/
def correct_reading : String :=
  "six thousand five"

theorem reading_6005_is_not_six_thousand_zero_zero_five (rules : ReadingRules) : 
  read_number 6005 rules ≠ "six thousand zero zero five" :=
sorry

end NUMINAMATH_CALUDE_reading_6005_is_not_six_thousand_zero_zero_five_l3807_380720


namespace NUMINAMATH_CALUDE_circle_area_ratio_l3807_380758

-- Define the circles and angles
def circle_small : Real → Real → Real := sorry
def circle_large : Real → Real → Real := sorry
def circle_sum : Real → Real → Real := sorry

def angle_small : Real := 60
def angle_large : Real := 48
def angle_sum : Real := 108

-- Define the radii
def radius_small : Real := sorry
def radius_large : Real := sorry
def radius_sum : Real := radius_small + radius_large

-- Define arc lengths
def arc_length (circle : Real → Real → Real) (angle : Real) : Real := sorry

-- State the theorem
theorem circle_area_ratio :
  let arc_small := arc_length circle_small angle_small
  let arc_large := arc_length circle_large angle_large
  let arc_sum := arc_length circle_sum angle_sum
  arc_small = arc_large ∧
  arc_sum = arc_small + arc_large →
  (circle_small radius_small 0) / (circle_large radius_large 0) = 16 / 25 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l3807_380758


namespace NUMINAMATH_CALUDE_president_vp_advisory_board_selection_l3807_380785

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem president_vp_advisory_board_selection (total_people : ℕ) (h : total_people = 10) :
  (total_people) * (total_people - 1) * (choose (total_people - 2) 2) = 2520 :=
sorry

end NUMINAMATH_CALUDE_president_vp_advisory_board_selection_l3807_380785


namespace NUMINAMATH_CALUDE_high_school_total_students_l3807_380743

/-- Represents a high school with three grades and stratified sampling -/
structure HighSchool where
  total_students : ℕ
  freshmen : ℕ
  sample_size : ℕ
  sampled_sophomores : ℕ
  sampled_seniors : ℕ

/-- The conditions of the problem -/
def problem_conditions (hs : HighSchool) : Prop :=
  hs.freshmen = 600 ∧
  hs.sample_size = 45 ∧
  hs.sampled_sophomores = 20 ∧
  hs.sampled_seniors = 10

/-- The theorem to prove -/
theorem high_school_total_students (hs : HighSchool) 
  (h : problem_conditions hs) : 
  hs.total_students = 1800 :=
sorry

end NUMINAMATH_CALUDE_high_school_total_students_l3807_380743
