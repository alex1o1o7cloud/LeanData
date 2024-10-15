import Mathlib

namespace NUMINAMATH_CALUDE_maxwell_brad_meeting_time_l1565_156516

/-- The time it takes for Maxwell and Brad to meet, given their speeds and the distance between their homes. -/
theorem maxwell_brad_meeting_time 
  (distance : ℝ) 
  (maxwell_speed : ℝ) 
  (brad_speed : ℝ) 
  (head_start : ℝ) 
  (h1 : distance = 54) 
  (h2 : maxwell_speed = 4) 
  (h3 : brad_speed = 6) 
  (h4 : head_start = 1) :
  ∃ (t : ℝ), t + head_start = 6 ∧ 
  maxwell_speed * (t + head_start) + brad_speed * t = distance :=
sorry

end NUMINAMATH_CALUDE_maxwell_brad_meeting_time_l1565_156516


namespace NUMINAMATH_CALUDE_rotation_transformation_l1565_156523

-- Define the triangles
def triangle_DEF : List (ℝ × ℝ) := [(0, 0), (0, 10), (14, 0)]
def triangle_DEF_prime : List (ℝ × ℝ) := [(28, 14), (40, 14), (28, 4)]

-- Define the rotation function
def rotate (center : ℝ × ℝ) (angle : ℝ) (point : ℝ × ℝ) : ℝ × ℝ := sorry

theorem rotation_transformation (n p q : ℝ) :
  0 < n → n < 180 →
  (∀ (point : ℝ × ℝ), point ∈ triangle_DEF →
    rotate (p, q) n point ∈ triangle_DEF_prime) →
  n + p + q = 104 := by sorry

end NUMINAMATH_CALUDE_rotation_transformation_l1565_156523


namespace NUMINAMATH_CALUDE_fruit_salad_cherries_l1565_156565

theorem fruit_salad_cherries (b r g c : ℕ) : 
  b + r + g + c = 390 →
  r = 3 * b →
  g = 2 * c →
  c = 5 * r →
  c = 119 := by
sorry

end NUMINAMATH_CALUDE_fruit_salad_cherries_l1565_156565


namespace NUMINAMATH_CALUDE_max_y_over_x_l1565_156560

theorem max_y_over_x (x y : ℝ) (h : (x - 2)^2 + y^2 = 1) :
  ∃ (M : ℝ), M = Real.sqrt 3 / 3 ∧ ∀ (x' y' : ℝ), (x' - 2)^2 + y'^2 = 1 → |y' / x'| ≤ M := by
  sorry

end NUMINAMATH_CALUDE_max_y_over_x_l1565_156560


namespace NUMINAMATH_CALUDE_distance_A_proof_l1565_156519

/-- The distance that runner A can run, given the conditions of the problem -/
def distance_A : ℝ := 224

theorem distance_A_proof (time_A time_B beat_distance : ℝ) 
  (h1 : time_A = 28)
  (h2 : time_B = 32)
  (h3 : beat_distance = 32)
  (h4 : distance_A / time_A * time_B = distance_A + beat_distance) : 
  distance_A = 224 := by sorry

end NUMINAMATH_CALUDE_distance_A_proof_l1565_156519


namespace NUMINAMATH_CALUDE_opposite_face_of_one_is_three_l1565_156594

/-- Represents a face of a cube --/
inductive CubeFace
| One
| Two
| Three
| Four
| Five
| Six

/-- Represents a net of a cube --/
structure CubeNet where
  faces : Finset CubeFace
  valid : faces.card = 6

/-- Represents a folded cube --/
structure FoldedCube where
  net : CubeNet
  topFace : CubeFace
  bottomFace : CubeFace
  oppositeFaces : CubeFace → CubeFace

/-- Theorem stating that in a cube formed by folding a net with faces numbered 1 to 6,
    where face 1 becomes the top face, the face opposite to face 1 is face 3 --/
theorem opposite_face_of_one_is_three (c : FoldedCube) 
    (h1 : c.topFace = CubeFace.One) :
  c.oppositeFaces CubeFace.One = CubeFace.Three :=
sorry

end NUMINAMATH_CALUDE_opposite_face_of_one_is_three_l1565_156594


namespace NUMINAMATH_CALUDE_find_c_l1565_156535

theorem find_c (p q : ℝ → ℝ) (c : ℝ) 
  (hp : ∀ x, p x = 3 * x - 8)
  (hq : ∀ x, q x = 4 * x - c)
  (h_pq3 : p (q 3) = 14) :
  c = 14 / 3 := by
sorry

end NUMINAMATH_CALUDE_find_c_l1565_156535


namespace NUMINAMATH_CALUDE_octahedron_cube_volume_ratio_l1565_156583

/-- The ratio of the volume of a regular octahedron formed by joining the centers of adjoining faces
    of a cube to the volume of the cube, when the cube has a side length of 2 units. -/
theorem octahedron_cube_volume_ratio : 
  let cube_side : ℝ := 2
  let cube_volume : ℝ := cube_side ^ 3
  let octahedron_side : ℝ := Real.sqrt 2
  let octahedron_volume : ℝ := (octahedron_side ^ 3 * Real.sqrt 2) / 3
  octahedron_volume / cube_volume = 1 / 6 := by
sorry


end NUMINAMATH_CALUDE_octahedron_cube_volume_ratio_l1565_156583


namespace NUMINAMATH_CALUDE_symmetric_point_origin_specific_symmetric_point_l1565_156522

def symmetric_point (x y : ℝ) : ℝ × ℝ := (-x, -y)

theorem symmetric_point_origin (x y : ℝ) : 
  symmetric_point x y = (-x, -y) := by sorry

theorem specific_symmetric_point : 
  symmetric_point (-2) 5 = (2, -5) := by sorry

end NUMINAMATH_CALUDE_symmetric_point_origin_specific_symmetric_point_l1565_156522


namespace NUMINAMATH_CALUDE_divisible_by_twelve_l1565_156511

/-- The function that constructs the number 534n given n -/
def number (n : ℕ) : ℕ := 5340 + n

/-- Predicate to check if a number is four-digit -/
def is_four_digit (x : ℕ) : Prop := 1000 ≤ x ∧ x < 10000

theorem divisible_by_twelve (n : ℕ) : 
  (is_four_digit (number n)) → 
  (n < 10) → 
  ((number n) % 12 = 0 ↔ n = 0) := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_twelve_l1565_156511


namespace NUMINAMATH_CALUDE_min_value_of_product_l1565_156588

theorem min_value_of_product (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 = 1) :
  (a - b) * (b - c) * (c - d) * (d - a) ≥ -1/8 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_product_l1565_156588


namespace NUMINAMATH_CALUDE_cloth_sale_proof_l1565_156581

/-- Given a trader selling cloth with a profit of 55 per meter and a total profit of 2200,
    prove that the number of meters sold is 40. -/
theorem cloth_sale_proof (profit_per_meter : ℕ) (total_profit : ℕ) 
    (h1 : profit_per_meter = 55) (h2 : total_profit = 2200) : 
    total_profit / profit_per_meter = 40 := by
  sorry

end NUMINAMATH_CALUDE_cloth_sale_proof_l1565_156581


namespace NUMINAMATH_CALUDE_cow_count_l1565_156539

theorem cow_count (D C : ℕ) : 
  2 * D + 4 * C = 2 * (D + C) + 30 → C = 15 := by
  sorry

end NUMINAMATH_CALUDE_cow_count_l1565_156539


namespace NUMINAMATH_CALUDE_jenny_calculation_l1565_156545

theorem jenny_calculation (x : ℚ) : (x - 14) / 5 = 11 → (x - 5) / 7 = 64/7 := by
  sorry

end NUMINAMATH_CALUDE_jenny_calculation_l1565_156545


namespace NUMINAMATH_CALUDE_airport_passenger_ratio_l1565_156570

/-- Proves that the ratio of passengers using Miami Airport to those using Logan Airport is 4:1 -/
theorem airport_passenger_ratio :
  let total_passengers : ℝ := 38.3 * 1000000
  let kennedy_passengers : ℝ := total_passengers / 3
  let miami_passengers : ℝ := kennedy_passengers / 2
  let logan_passengers : ℝ := 1.5958333333333332 * 1000000
  miami_passengers / logan_passengers = 4 := by
  sorry

end NUMINAMATH_CALUDE_airport_passenger_ratio_l1565_156570


namespace NUMINAMATH_CALUDE_complex_square_imaginary_part_l1565_156585

theorem complex_square_imaginary_part : 
  ∃ (a b : ℝ), (1 + Complex.I)^2 = (a : ℂ) + (b : ℂ) * Complex.I → b = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_square_imaginary_part_l1565_156585


namespace NUMINAMATH_CALUDE_tangent_perpendicular_implies_a_equals_one_l1565_156548

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + x + 1

-- Define the derivative of f(x)
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 1

-- Theorem statement
theorem tangent_perpendicular_implies_a_equals_one (a : ℝ) :
  (f_deriv a 1 * (-1/4) = -1) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_perpendicular_implies_a_equals_one_l1565_156548


namespace NUMINAMATH_CALUDE_pentagon_angle_measure_l1565_156525

theorem pentagon_angle_measure (a b c d e : ℝ) : 
  -- Pentagon angles sum to 540 degrees
  a + b + c + d + e = 540 ∧
  -- Four angles are congruent
  a = b ∧ b = c ∧ c = d ∧
  -- The fifth angle is 50 degrees more than each of the other angles
  e = a + 50 →
  -- The measure of the fifth angle is 148 degrees
  e = 148 :=
by sorry

end NUMINAMATH_CALUDE_pentagon_angle_measure_l1565_156525


namespace NUMINAMATH_CALUDE_direct_variation_problem_l1565_156567

/-- A function representing direct variation --/
def direct_variation (k : ℝ) (x : ℝ) : ℝ := k * x

theorem direct_variation_problem (k : ℝ) :
  (direct_variation k 2.5 = 10) →
  (direct_variation k (-5) = -20) := by
  sorry

#check direct_variation_problem

end NUMINAMATH_CALUDE_direct_variation_problem_l1565_156567


namespace NUMINAMATH_CALUDE_diane_honey_harvest_l1565_156527

/-- Diane's honey harvest problem -/
theorem diane_honey_harvest 
  (last_year_harvest : ℕ) 
  (harvest_increase : ℕ) 
  (h1 : last_year_harvest = 2479)
  (h2 : harvest_increase = 6085) :
  last_year_harvest + harvest_increase = 8564 :=
by sorry

end NUMINAMATH_CALUDE_diane_honey_harvest_l1565_156527


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1565_156530

theorem polynomial_factorization (x y z : ℝ) : 
  x * (y - z)^4 + y * (z - x)^4 + z * (x - y)^4 = 
  (x - y) * (y - z) * (z - x) * (-(x - y)^2 - (y - z)^2 - (z - x)^2) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1565_156530


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l1565_156502

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  X^6 + X^5 + 2*X^3 - X^2 + 3 = (X + 2) * (X - 1) * q + (-X + 5) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l1565_156502


namespace NUMINAMATH_CALUDE_age_ratio_problem_l1565_156521

/-- Aaron's current age -/
def aaron_age : ℕ := sorry

/-- Beth's current age -/
def beth_age : ℕ := sorry

/-- The number of years until their age ratio is 3:2 -/
def years_until_ratio : ℕ := sorry

/-- Theorem stating the conditions and the result to be proved -/
theorem age_ratio_problem :
  (aaron_age - 4 = 2 * (beth_age - 4)) ∧
  (aaron_age - 6 = 3 * (beth_age - 6)) →
  years_until_ratio = 24 ∧
  (aaron_age + years_until_ratio) * 2 = 3 * (beth_age + years_until_ratio) :=
by sorry

end NUMINAMATH_CALUDE_age_ratio_problem_l1565_156521


namespace NUMINAMATH_CALUDE_final_racers_count_l1565_156536

def race_elimination (initial_racers : ℕ) : ℕ :=
  let after_first := initial_racers - 10
  let after_second := after_first - (after_first / 3)
  let after_third := after_second - (after_second / 2)
  after_third

theorem final_racers_count :
  race_elimination 100 = 30 := by sorry

end NUMINAMATH_CALUDE_final_racers_count_l1565_156536


namespace NUMINAMATH_CALUDE_normal_dist_probability_l1565_156547

-- Define the normal distribution
def normal_dist (μ σ : ℝ) (hσ : σ > 0) : Type := Unit

-- Define the probability function
def P (X : normal_dist 1 σ hσ) (a b : ℝ) : ℝ := sorry

-- Define our theorem
theorem normal_dist_probability 
  (σ : ℝ) (hσ : σ > 0) (X : normal_dist 1 σ hσ) 
  (h : P X 0 1 = 0.4) : P X 0 2 = 0.8 := by sorry

end NUMINAMATH_CALUDE_normal_dist_probability_l1565_156547


namespace NUMINAMATH_CALUDE_age_ratio_in_two_years_l1565_156513

def maya_age : ℕ := 15
def drew_age : ℕ := maya_age + 5
def peter_age : ℕ := drew_age + 4
def john_age : ℕ := 30
def jacob_age : ℕ := 11

theorem age_ratio_in_two_years :
  (jacob_age + 2) / (peter_age + 2) = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_age_ratio_in_two_years_l1565_156513


namespace NUMINAMATH_CALUDE_complex_triple_solution_l1565_156550

theorem complex_triple_solution (x y z : ℂ) :
  (x + y)^3 + (y + z)^3 + (z + x)^3 - 3*(x + y)*(y + z)*(z + x) = 0 →
  x^2*(y + z) + y^2*(z + x) + z^2*(x + y) = 0 →
  x + y + z = 0 ∧ x*y*z = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_triple_solution_l1565_156550


namespace NUMINAMATH_CALUDE_max_polygon_size_no_parallel_sides_l1565_156571

/-- A type representing a point on a circle -/
structure CirclePoint where
  angle : ℝ
  -- Assuming angle is in radians and normalized to [0, 2π)

/-- The number of points marked on the circle -/
def num_points : ℕ := 2012

/-- The set of all points on the circle -/
def circle_points : Finset CirclePoint :=
  sorry

/-- Predicate to check if two line segments are parallel -/
def are_parallel (p1 p2 p3 p4 : CirclePoint) : Prop :=
  sorry

/-- Predicate to check if a set of points forms a convex polygon -/
def is_convex_polygon (points : Finset CirclePoint) : Prop :=
  sorry

/-- The main theorem -/
theorem max_polygon_size_no_parallel_sides :
  ∃ (points : Finset CirclePoint),
    points.card = 1509 ∧
    is_convex_polygon points ∧
    (∀ (p1 p2 p3 p4 : CirclePoint),
      p1 ∈ points → p2 ∈ points → p3 ∈ points → p4 ∈ points →
      p1 ≠ p2 → p3 ≠ p4 → ¬(are_parallel p1 p2 p3 p4)) ∧
    (∀ (larger_set : Finset CirclePoint),
      larger_set.card > 1509 →
      is_convex_polygon larger_set →
      (∃ (q1 q2 q3 q4 : CirclePoint),
        q1 ∈ larger_set ∧ q2 ∈ larger_set ∧ q3 ∈ larger_set ∧ q4 ∈ larger_set ∧
        q1 ≠ q2 ∧ q3 ≠ q4 ∧ are_parallel q1 q2 q3 q4)) :=
by sorry


end NUMINAMATH_CALUDE_max_polygon_size_no_parallel_sides_l1565_156571


namespace NUMINAMATH_CALUDE_sin_cos_bound_l1565_156533

theorem sin_cos_bound (t : ℝ) : -5 ≤ 4 * Real.sin t + Real.cos (2 * t) ∧ 4 * Real.sin t + Real.cos (2 * t) ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_bound_l1565_156533


namespace NUMINAMATH_CALUDE_white_most_likely_probabilities_game_is_fair_l1565_156558

/-- Represents the colors of ping-pong balls in the box -/
inductive Color
  | White
  | Yellow
  | Red

/-- The total number of balls in the box -/
def totalBalls : ℕ := 6

/-- The number of balls of each color -/
def numBalls (c : Color) : ℕ :=
  match c with
  | Color.White => 3
  | Color.Yellow => 2
  | Color.Red => 1

/-- The probability of picking a ball of a given color -/
def prob (c : Color) : ℚ :=
  (numBalls c : ℚ) / totalBalls

/-- Theorem stating that white is the most likely color to be picked -/
theorem white_most_likely :
  ∀ c : Color, c ≠ Color.White → prob Color.White > prob c := by sorry

/-- Theorem stating the probabilities for each color -/
theorem probabilities :
  prob Color.White = 1/2 ∧ prob Color.Yellow = 1/3 ∧ prob Color.Red = 1/6 := by sorry

/-- Theorem stating that the game is fair -/
theorem game_is_fair :
  prob Color.White = 1 - prob Color.White := by sorry

end NUMINAMATH_CALUDE_white_most_likely_probabilities_game_is_fair_l1565_156558


namespace NUMINAMATH_CALUDE_last_two_nonzero_digits_80_factorial_l1565_156584

/-- The last two nonzero digits of n! -/
def lastTwoNonzeroDigits (n : ℕ) : ℕ := sorry

/-- The number of factors of 10 in n! -/
def factorsOfTen (n : ℕ) : ℕ := sorry

theorem last_two_nonzero_digits_80_factorial :
  lastTwoNonzeroDigits 80 = 52 := by sorry

end NUMINAMATH_CALUDE_last_two_nonzero_digits_80_factorial_l1565_156584


namespace NUMINAMATH_CALUDE_call_center_problem_l1565_156500

theorem call_center_problem (team_a_agents : ℚ) (team_b_agents : ℚ) 
  (team_a_calls_per_agent : ℚ) (team_b_calls_per_agent : ℚ) :
  team_a_agents = (5 / 8) * team_b_agents →
  team_a_calls_per_agent = (2 / 5) * team_b_calls_per_agent →
  let total_calls := team_a_agents * team_a_calls_per_agent + team_b_agents * team_b_calls_per_agent
  (team_b_agents * team_b_calls_per_agent) / total_calls = 8 / 9 := by
  sorry

end NUMINAMATH_CALUDE_call_center_problem_l1565_156500


namespace NUMINAMATH_CALUDE_choose_four_from_ten_l1565_156543

theorem choose_four_from_ten (n : ℕ) (k : ℕ) : n = 10 ∧ k = 4 → Nat.choose n k = 210 := by
  sorry

end NUMINAMATH_CALUDE_choose_four_from_ten_l1565_156543


namespace NUMINAMATH_CALUDE_value_of_x_l1565_156572

theorem value_of_x (x y z : ℚ) : x = y / 3 → y = z / 4 → z = 48 → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_value_of_x_l1565_156572


namespace NUMINAMATH_CALUDE_uncles_gift_amount_l1565_156596

def jerseys_cost : ℕ := 5 * 2
def basketball_cost : ℕ := 18
def shorts_cost : ℕ := 8
def money_left : ℕ := 14

theorem uncles_gift_amount : 
  jerseys_cost + basketball_cost + shorts_cost + money_left = 50 := by
  sorry

end NUMINAMATH_CALUDE_uncles_gift_amount_l1565_156596


namespace NUMINAMATH_CALUDE_complex_power_problem_l1565_156538

theorem complex_power_problem (z : ℂ) (i : ℂ) (h1 : i^2 = -1) (h2 : z * (1 - i) = 1 + i) : z^2016 = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_problem_l1565_156538


namespace NUMINAMATH_CALUDE_marble_average_l1565_156590

/-- Given the conditions about the average numbers of marbles of different colors,
    prove that the average number of all three colors is 30. -/
theorem marble_average (R Y B : ℕ) : 
  (R + Y : ℚ) / 2 = 26.5 →
  (B + Y : ℚ) / 2 = 34.5 →
  (R + B : ℚ) / 2 = 29 →
  (R + Y + B : ℚ) / 3 = 30 := by
  sorry

end NUMINAMATH_CALUDE_marble_average_l1565_156590


namespace NUMINAMATH_CALUDE_cereal_difference_theorem_l1565_156504

/-- Represents the probability of eating unsweetened cereal -/
def p_unsweetened : ℚ := 3/5

/-- Represents the probability of eating sweetened cereal -/
def p_sweetened : ℚ := 2/5

/-- Number of days in a non-leap year -/
def days_in_year : ℕ := 365

/-- Expected difference between days of eating unsweetened and sweetened cereal -/
def expected_difference : ℚ := days_in_year * (p_unsweetened - p_sweetened)

theorem cereal_difference_theorem : 
  expected_difference = 73 := by sorry

end NUMINAMATH_CALUDE_cereal_difference_theorem_l1565_156504


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l1565_156568

theorem smallest_n_congruence (n : ℕ+) : 
  (∀ m : ℕ+, m < n → ¬(13 * m.val) % 8 = 567 % 8) ∧ 
  (13 * n.val) % 8 = 567 % 8 → 
  n = 3 := by sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l1565_156568


namespace NUMINAMATH_CALUDE_percent_of_double_is_eighteen_l1565_156592

theorem percent_of_double_is_eighteen (y : ℝ) (h1 : y > 0) (h2 : (y / 100) * (2 * y) = 18) : y = 30 := by
  sorry

end NUMINAMATH_CALUDE_percent_of_double_is_eighteen_l1565_156592


namespace NUMINAMATH_CALUDE_chicken_egg_production_l1565_156593

theorem chicken_egg_production 
  (num_chickens : ℕ) 
  (price_per_dozen : ℚ) 
  (total_revenue : ℚ) 
  (num_weeks : ℕ) :
  num_chickens = 8 →
  price_per_dozen = 5 →
  total_revenue = 280 →
  num_weeks = 4 →
  (total_revenue / price_per_dozen * 12) / (num_weeks * 7) / num_chickens = 3 :=
by sorry

end NUMINAMATH_CALUDE_chicken_egg_production_l1565_156593


namespace NUMINAMATH_CALUDE_inserted_eights_composite_l1565_156512

def insert_eights (n : ℕ) : ℕ :=
  2000 * 10^n + 8 * ((10^n - 1) / 9) + 21

theorem inserted_eights_composite (n : ℕ) :
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ insert_eights n = a * b :=
sorry

end NUMINAMATH_CALUDE_inserted_eights_composite_l1565_156512


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_expression_l1565_156508

theorem imaginary_part_of_complex_expression :
  Complex.im (((1 : ℂ) + Complex.I) / ((1 : ℂ) - Complex.I) + (1 - Complex.I)^2) = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_expression_l1565_156508


namespace NUMINAMATH_CALUDE_min_total_cost_l1565_156514

/-- Represents the number of book corners of each size -/
structure BookCorners where
  medium : ℕ
  small : ℕ

/-- Calculates the total cost for a given configuration of book corners -/
def total_cost (corners : BookCorners) : ℕ :=
  860 * corners.medium + 570 * corners.small

/-- Checks if a configuration of book corners is valid according to the given constraints -/
def is_valid_configuration (corners : BookCorners) : Prop :=
  corners.medium + corners.small = 30 ∧
  80 * corners.medium + 30 * corners.small ≤ 1900 ∧
  50 * corners.medium + 60 * corners.small ≤ 1620

/-- Theorem stating that the minimum total cost is 22320 yuan -/
theorem min_total_cost :
  ∃ (corners : BookCorners),
    is_valid_configuration corners ∧
    total_cost corners = 22320 ∧
    ∀ (other : BookCorners), is_valid_configuration other → total_cost other ≥ 22320 := by
  sorry

end NUMINAMATH_CALUDE_min_total_cost_l1565_156514


namespace NUMINAMATH_CALUDE_no_solutions_cubic_equation_l1565_156573

theorem no_solutions_cubic_equation :
  (∀ x y : ℕ, x ≠ y → x^3 + 5*y ≠ y^3 + 5*x) ∧
  (∀ x y : ℤ, x ≠ y → x^3 + 5*y ≠ y^3 + 5*x) :=
by sorry

end NUMINAMATH_CALUDE_no_solutions_cubic_equation_l1565_156573


namespace NUMINAMATH_CALUDE_spider_dressing_theorem_l1565_156517

def spider_dressing_orders (n : ℕ) : ℚ :=
  (Nat.factorial (3 * n) * (4 ^ n)) / (6 ^ n)

theorem spider_dressing_theorem (n : ℕ) (hn : n = 8) :
  spider_dressing_orders n = (Nat.factorial (3 * n) * (4 ^ n)) / (6 ^ n) :=
by sorry

end NUMINAMATH_CALUDE_spider_dressing_theorem_l1565_156517


namespace NUMINAMATH_CALUDE_monomial_is_algebraic_expression_l1565_156586

-- Define what an algebraic expression is
def AlgebraicExpression (α : Type*) := α → ℝ

-- Define what a monomial is
def Monomial (α : Type*) := AlgebraicExpression α

-- Theorem: Every monomial is an algebraic expression
theorem monomial_is_algebraic_expression {α : Type*} :
  ∀ (m : Monomial α), ∃ (a : AlgebraicExpression α), m = a :=
sorry

end NUMINAMATH_CALUDE_monomial_is_algebraic_expression_l1565_156586


namespace NUMINAMATH_CALUDE_roof_dimension_difference_l1565_156506

theorem roof_dimension_difference (width : ℝ) (length : ℝ) : 
  width > 0 →
  length = 4 * width →
  width * length = 676 →
  length - width = 39 := by
sorry

end NUMINAMATH_CALUDE_roof_dimension_difference_l1565_156506


namespace NUMINAMATH_CALUDE_expression_values_l1565_156559

theorem expression_values (a b : ℝ) (h : (2 * a) / (a + b) + b / (a - b) = 2) :
  (3 * a - b) / (a + 5 * b) = 1 ∨ (3 * a - b) / (a + 5 * b) = 3 :=
by sorry

end NUMINAMATH_CALUDE_expression_values_l1565_156559


namespace NUMINAMATH_CALUDE_ticket_sales_l1565_156575

theorem ticket_sales (adult_price children_price total_amount adult_tickets : ℕ) 
  (h1 : adult_price = 5)
  (h2 : children_price = 2)
  (h3 : total_amount = 275)
  (h4 : adult_tickets = 35) :
  ∃ children_tickets : ℕ, adult_tickets + children_tickets = 85 ∧ 
    adult_price * adult_tickets + children_price * children_tickets = total_amount :=
by sorry

end NUMINAMATH_CALUDE_ticket_sales_l1565_156575


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_three_numbers_l1565_156503

theorem arithmetic_mean_of_three_numbers (a b c : ℕ) (h : a = 25 ∧ b = 41 ∧ c = 50) : 
  (a + b + c) / 3 = 116 / 3 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_three_numbers_l1565_156503


namespace NUMINAMATH_CALUDE_linear_system_no_solution_l1565_156529

/-- A system of two linear equations in two variables -/
structure LinearSystem (a : ℝ) :=
  (eq1 : ℝ → ℝ → ℝ)
  (eq2 : ℝ → ℝ → ℝ)
  (h1 : ∀ x y, eq1 x y = a * x + 2 * y - 3)
  (h2 : ∀ x y, eq2 x y = 2 * x + a * y - 2)

/-- The system has no solution -/
def NoSolution (s : LinearSystem a) : Prop :=
  ∀ x y, ¬(s.eq1 x y = 0 ∧ s.eq2 x y = 0)

theorem linear_system_no_solution (a : ℝ) :
  (∃ s : LinearSystem a, NoSolution s) → a = 2 ∨ a = -2 := by
  sorry

end NUMINAMATH_CALUDE_linear_system_no_solution_l1565_156529


namespace NUMINAMATH_CALUDE_even_function_sum_of_angles_l1565_156546

/-- A function f is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem even_function_sum_of_angles (θ φ : ℝ) :
  IsEven (fun x ↦ Real.cos (x + θ) + Real.sqrt 2 * Real.sin (x + φ)) →
  0 < θ ∧ θ < π / 2 →
  0 < φ ∧ φ < π / 2 →
  Real.cos θ = Real.sqrt 6 / 3 * Real.sin φ →
  θ + φ = 7 * π / 12 := by
  sorry

end NUMINAMATH_CALUDE_even_function_sum_of_angles_l1565_156546


namespace NUMINAMATH_CALUDE_weight_loss_program_l1565_156510

def initial_weight : ℕ := 250
def weeks_phase1 : ℕ := 4
def loss_per_week_phase1 : ℕ := 3
def weeks_phase2 : ℕ := 8
def loss_per_week_phase2 : ℕ := 2

theorem weight_loss_program (w : ℕ) :
  w = initial_weight - (weeks_phase1 * loss_per_week_phase1 + weeks_phase2 * loss_per_week_phase2) →
  w = 222 :=
by sorry

end NUMINAMATH_CALUDE_weight_loss_program_l1565_156510


namespace NUMINAMATH_CALUDE_min_weighings_required_l1565_156555

/-- Represents a 4x4 grid of coins -/
def CoinGrid := Fin 4 → Fin 4 → ℕ

/-- Predicate to check if two positions are adjacent in the grid -/
def adjacent (p q : Fin 4 × Fin 4) : Prop :=
  (p.1 = q.1 ∧ (p.2.val + 1 = q.2.val ∨ q.2.val + 1 = p.2.val)) ∨
  (p.2 = q.2 ∧ (p.1.val + 1 = q.1.val ∨ q.1.val + 1 = p.1.val))

/-- A valid coin grid satisfying the problem conditions -/
def valid_coin_grid (g : CoinGrid) : Prop :=
  ∃ p q : Fin 4 × Fin 4,
    adjacent p q ∧
    g p.1 p.2 = 9 ∧ g q.1 q.2 = 9 ∧
    ∀ r : Fin 4 × Fin 4, (r ≠ p ∧ r ≠ q) → g r.1 r.2 = 10

/-- A weighing selects a subset of coins and returns their total weight -/
def Weighing := Set (Fin 4 × Fin 4) → ℕ

/-- The theorem stating the minimum number of weighings required -/
theorem min_weighings_required (g : CoinGrid) (h : valid_coin_grid g) :
  ∃ (w₁ w₂ w₃ : Weighing),
    (∀ g₁ g₂ : CoinGrid, valid_coin_grid g₁ → valid_coin_grid g₂ →
      (∀ S : Set (Fin 4 × Fin 4), w₁ S = w₁ S → w₂ S = w₂ S → w₃ S = w₃ S) →
      g₁ = g₂) ∧
    (∀ w₁' w₂' : Weighing,
      ¬∀ g₁ g₂ : CoinGrid, valid_coin_grid g₁ → valid_coin_grid g₂ →
        (∀ S : Set (Fin 4 × Fin 4), w₁' S = w₁' S → w₂' S = w₂' S) →
        g₁ = g₂) :=
by
  sorry

end NUMINAMATH_CALUDE_min_weighings_required_l1565_156555


namespace NUMINAMATH_CALUDE_shooting_game_cost_l1565_156574

theorem shooting_game_cost (jen_plays : ℕ) (russel_rides : ℕ) (carousel_cost : ℕ) (total_tickets : ℕ) :
  jen_plays = 2 →
  russel_rides = 3 →
  carousel_cost = 3 →
  total_tickets = 19 →
  ∃ (shooting_cost : ℕ), jen_plays * shooting_cost + russel_rides * carousel_cost = total_tickets ∧ shooting_cost = 5 := by
  sorry

end NUMINAMATH_CALUDE_shooting_game_cost_l1565_156574


namespace NUMINAMATH_CALUDE_sin_2alpha_value_l1565_156595

theorem sin_2alpha_value (α : Real) 
  (h1 : 2 * Real.cos (2 * α) = Real.sin (π / 4 - α))
  (h2 : π / 2 < α ∧ α < π) : 
  Real.sin (2 * α) = -7/8 := by
sorry

end NUMINAMATH_CALUDE_sin_2alpha_value_l1565_156595


namespace NUMINAMATH_CALUDE_indigo_restaurant_reviews_l1565_156576

theorem indigo_restaurant_reviews :
  let five_star : ℕ := 6
  let four_star : ℕ := 7
  let three_star : ℕ := 4
  let two_star : ℕ := 1
  let average_rating : ℚ := 4
  let total_reviews := five_star + four_star + three_star + two_star
  let total_stars := 5 * five_star + 4 * four_star + 3 * three_star + 2 * two_star
  (total_stars : ℚ) / total_reviews = average_rating →
  total_reviews = 18 := by
sorry

end NUMINAMATH_CALUDE_indigo_restaurant_reviews_l1565_156576


namespace NUMINAMATH_CALUDE_rachel_age_2009_l1565_156537

/-- Rachel's age at the end of 2004 -/
def rachel_age_2004 : ℝ := 47.5

/-- Rachel's uncle's age at the end of 2004 -/
def uncle_age_2004 : ℝ := 3 * rachel_age_2004

/-- The sum of Rachel's and her uncle's birth years -/
def birth_years_sum : ℕ := 3818

/-- The year for which we're calculating Rachel's age -/
def target_year : ℕ := 2009

/-- The base year from which we're calculating -/
def base_year : ℕ := 2004

theorem rachel_age_2009 :
  rachel_age_2004 + (target_year - base_year) = 52.5 ∧
  rachel_age_2004 = uncle_age_2004 / 3 ∧
  (base_year - rachel_age_2004) + (base_year - uncle_age_2004) = birth_years_sum :=
by sorry

end NUMINAMATH_CALUDE_rachel_age_2009_l1565_156537


namespace NUMINAMATH_CALUDE_pink_highlighters_l1565_156526

theorem pink_highlighters (total : ℕ) (yellow : ℕ) (blue : ℕ) (h1 : yellow = 2) (h2 : blue = 4) (h3 : total = 12) :
  total - yellow - blue = 6 := by
  sorry

end NUMINAMATH_CALUDE_pink_highlighters_l1565_156526


namespace NUMINAMATH_CALUDE_stratified_sample_theorem_l1565_156554

/-- Calculates the number of employees to be drawn from a department in a stratified sampling method. -/
def stratified_sample_size (total_employees : ℕ) (sample_size : ℕ) (department_size : ℕ) : ℕ :=
  (department_size * sample_size) / total_employees

/-- Theorem stating that for a company with 240 employees and a sample size of 20,
    the number of employees to be drawn from a department with 60 employees is 5. -/
theorem stratified_sample_theorem :
  stratified_sample_size 240 20 60 = 5 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_theorem_l1565_156554


namespace NUMINAMATH_CALUDE_octagon_arc_length_l1565_156532

/-- The length of an arc intercepted by one side of a regular octagon inscribed in a circle -/
theorem octagon_arc_length (side_length : ℝ) (h : side_length = 5) :
  let circumference := 2 * Real.pi * side_length
  let arc_length := circumference / 8
  arc_length = 1.25 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_octagon_arc_length_l1565_156532


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l1565_156505

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

theorem geometric_sequence_ratio 
  (a : ℕ → ℝ) (q : ℝ) 
  (h_geometric : geometric_sequence a q)
  (h_product : a 1 * a 2 * a 3 = 27)
  (h_sum : a 2 + a 4 = 30) :
  q = 3 ∨ q = -3 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l1565_156505


namespace NUMINAMATH_CALUDE_farmers_wheat_estimate_l1565_156509

/-- The farmer's wheat harvest problem -/
theorem farmers_wheat_estimate (total_harvest : ℕ) (extra_bushels : ℕ) 
  (h1 : total_harvest = 48781)
  (h2 : extra_bushels = 684) :
  total_harvest - extra_bushels = 48097 := by
  sorry

end NUMINAMATH_CALUDE_farmers_wheat_estimate_l1565_156509


namespace NUMINAMATH_CALUDE_factorization_equality_l1565_156528

theorem factorization_equality (x : ℝ) : (x^2 + 9)^2 - 36*x^2 = (x + 3)^2 * (x - 3)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1565_156528


namespace NUMINAMATH_CALUDE_path_area_and_cost_l1565_156501

/-- Represents the dimensions of a rectangular field with a path around it -/
structure FieldWithPath where
  fieldLength : ℝ
  fieldWidth : ℝ
  pathWidth : ℝ

/-- Calculates the area of the path around a rectangular field -/
def areaOfPath (f : FieldWithPath) : ℝ :=
  (f.fieldLength + 2 * f.pathWidth) * (f.fieldWidth + 2 * f.pathWidth) - f.fieldLength * f.fieldWidth

/-- Calculates the cost of constructing the path given the cost per square meter -/
def costOfPath (f : FieldWithPath) (costPerSqm : ℝ) : ℝ :=
  areaOfPath f * costPerSqm

/-- Theorem stating the area of the path and its construction cost for the given field dimensions -/
theorem path_area_and_cost (f : FieldWithPath) (h1 : f.fieldLength = 65) (h2 : f.fieldWidth = 55) 
    (h3 : f.pathWidth = 2.5) (h4 : costPerSqm = 2) : 
    areaOfPath f = 625 ∧ costOfPath f costPerSqm = 1250 := by
  sorry

end NUMINAMATH_CALUDE_path_area_and_cost_l1565_156501


namespace NUMINAMATH_CALUDE_opposites_sum_to_zero_l1565_156578

theorem opposites_sum_to_zero (a b : ℝ) (h : a = -b) : a + b = 0 := by
  sorry

end NUMINAMATH_CALUDE_opposites_sum_to_zero_l1565_156578


namespace NUMINAMATH_CALUDE_answer_key_combinations_l1565_156566

/-- Represents the number of possible answers for a true-false question -/
def true_false_options : ℕ := 2

/-- Represents the number of possible answers for a multiple-choice question -/
def multiple_choice_options : ℕ := 4

/-- Represents the number of true-false questions in the quiz -/
def num_true_false : ℕ := 3

/-- Represents the number of multiple-choice questions in the quiz -/
def num_multiple_choice : ℕ := 3

/-- Calculates the number of ways to arrange true-false answers where all answers cannot be the same -/
def true_false_combinations : ℕ := true_false_options ^ num_true_false - 2

/-- Calculates the number of ways to arrange multiple-choice answers -/
def multiple_choice_combinations : ℕ := multiple_choice_options ^ num_multiple_choice

/-- Theorem stating that the total number of ways to create an answer key is 384 -/
theorem answer_key_combinations : 
  true_false_combinations * multiple_choice_combinations = 384 := by
  sorry

end NUMINAMATH_CALUDE_answer_key_combinations_l1565_156566


namespace NUMINAMATH_CALUDE_solution_in_quadrant_I_l1565_156540

/-- A point (x, y) lies in Quadrant I if both x and y are positive -/
def in_quadrant_I (x y : ℝ) : Prop := x > 0 ∧ y > 0

/-- The system of equations -/
def system_equations (k x y : ℝ) : Prop :=
  2 * x - y = 5 ∧ k * x^2 + y = 4

theorem solution_in_quadrant_I (k : ℝ) :
  (∃ x y : ℝ, system_equations k x y ∧ in_quadrant_I x y) ↔ k > 0 :=
sorry

end NUMINAMATH_CALUDE_solution_in_quadrant_I_l1565_156540


namespace NUMINAMATH_CALUDE_decimal_sum_to_fraction_l1565_156562

theorem decimal_sum_to_fraction :
  (0.2 : ℚ) + 0.03 + 0.004 + 0.0005 + 0.00006 = 1466 / 6250 := by
  sorry

end NUMINAMATH_CALUDE_decimal_sum_to_fraction_l1565_156562


namespace NUMINAMATH_CALUDE_pattern_result_l1565_156569

-- Define the pattern function
def pattern (a b : ℕ) : ℕ := sorry

-- Define the given operations
axiom op1 : pattern 3 7 = 27
axiom op2 : pattern 4 5 = 32
axiom op3 : pattern 5 8 = 60
axiom op4 : pattern 6 7 = 72
axiom op5 : pattern 7 8 = 98

-- Theorem to prove
theorem pattern_result : pattern 2 3 = 26 := by sorry

end NUMINAMATH_CALUDE_pattern_result_l1565_156569


namespace NUMINAMATH_CALUDE_al_sandwich_options_l1565_156579

/-- Represents the number of different types of bread available. -/
def num_breads : ℕ := 5

/-- Represents the number of different types of meat available. -/
def num_meats : ℕ := 7

/-- Represents the number of different types of cheese available. -/
def num_cheeses : ℕ := 6

/-- Represents whether turkey is available. -/
def turkey_available : Prop := True

/-- Represents whether roast beef is available. -/
def roast_beef_available : Prop := True

/-- Represents whether Swiss cheese is available. -/
def swiss_cheese_available : Prop := True

/-- Represents whether rye bread is available. -/
def rye_bread_available : Prop := True

/-- Represents the restriction that Al never orders a sandwich with a turkey/Swiss cheese combination. -/
def no_turkey_swiss : Prop := True

/-- Represents the restriction that Al never orders a sandwich with a rye bread/roast beef combination. -/
def no_rye_roast_beef : Prop := True

/-- The number of different sandwiches Al could order. -/
def num_al_sandwiches : ℕ := num_breads * num_meats * num_cheeses - 5 - 6

theorem al_sandwich_options :
  num_breads = 5 →
  num_meats = 7 →
  num_cheeses = 6 →
  turkey_available →
  roast_beef_available →
  swiss_cheese_available →
  rye_bread_available →
  no_turkey_swiss →
  no_rye_roast_beef →
  num_al_sandwiches = 199 := by
  sorry

#eval num_al_sandwiches -- This should output 199

end NUMINAMATH_CALUDE_al_sandwich_options_l1565_156579


namespace NUMINAMATH_CALUDE_unique_line_existence_l1565_156589

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def is_even (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

def line_passes_through (a b : ℚ) (x y : ℚ) : Prop :=
  x / a + y / b = 1

theorem unique_line_existence :
  ∃! (a b : ℚ), 
    (∃ n : ℕ, a = n ∧ is_prime n ∧ n < 10) ∧ 
    (∃ m : ℕ, b = m ∧ is_even m) ∧ 
    line_passes_through a b 5 4 :=
sorry

end NUMINAMATH_CALUDE_unique_line_existence_l1565_156589


namespace NUMINAMATH_CALUDE_complex_modulus_product_l1565_156542

theorem complex_modulus_product : 
  Complex.abs ((5 * Real.sqrt 3 - 5 * Complex.I) * (2 * Real.sqrt 2 + 4 * Complex.I)) = 20 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_product_l1565_156542


namespace NUMINAMATH_CALUDE_number_difference_l1565_156557

theorem number_difference (a b : ℕ) : 
  a + b = 34800 → 
  b % 25 = 0 → 
  b = 25 * a → 
  b - a = 32112 := by
sorry

end NUMINAMATH_CALUDE_number_difference_l1565_156557


namespace NUMINAMATH_CALUDE_min_distance_curve_to_line_l1565_156599

/-- Given a > 0 and b = -1/2 * a^2 + 3 * ln(a), and a point Q(m, n) on the line y = 2x + 1/2,
    the minimum value of (a-m)^2 + (b-n)^2 is 9/5 -/
theorem min_distance_curve_to_line (a b m n : ℝ) (ha : a > 0) 
  (hb : b = -1/2 * a^2 + 3 * Real.log a) (hq : n = 2 * m + 1/2) :
  ∃ (min_val : ℝ), min_val = 9/5 ∧ 
  ∀ (x y : ℝ), (y = -1/2 * x^2 + 3 * Real.log x) → 
  (a - m)^2 + (b - n)^2 ≤ (x - m)^2 + (y - n)^2 :=
sorry

end NUMINAMATH_CALUDE_min_distance_curve_to_line_l1565_156599


namespace NUMINAMATH_CALUDE_tetrahedron_relationships_l1565_156551

/-- Properties of a tetrahedron with inscribed and face-touching spheres -/
structure Tetrahedron where
  ρ : ℝ  -- radius of inscribed sphere
  ρ₁ : ℝ  -- radius of sphere touching face opposite to A
  ρ₂ : ℝ  -- radius of sphere touching face opposite to B
  ρ₃ : ℝ  -- radius of sphere touching face opposite to C
  ρ₄ : ℝ  -- radius of sphere touching face opposite to D
  m₁ : ℝ  -- length of altitude from A to opposite face
  m₂ : ℝ  -- length of altitude from B to opposite face
  m₃ : ℝ  -- length of altitude from C to opposite face
  m₄ : ℝ  -- length of altitude from D to opposite face
  ρ_pos : 0 < ρ
  ρ₁_pos : 0 < ρ₁
  ρ₂_pos : 0 < ρ₂
  ρ₃_pos : 0 < ρ₃
  ρ₄_pos : 0 < ρ₄
  m₁_pos : 0 < m₁
  m₂_pos : 0 < m₂
  m₃_pos : 0 < m₃
  m₄_pos : 0 < m₄

/-- Theorem about relationships in a tetrahedron -/
theorem tetrahedron_relationships (t : Tetrahedron) :
  (2 / t.ρ = 1 / t.ρ₁ + 1 / t.ρ₂ + 1 / t.ρ₃ + 1 / t.ρ₄) ∧
  (1 / t.ρ = 1 / t.m₁ + 1 / t.m₂ + 1 / t.m₃ + 1 / t.m₄) ∧
  (1 / t.ρ₁ = -1 / t.m₁ + 1 / t.m₂ + 1 / t.m₃ + 1 / t.m₄) := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_relationships_l1565_156551


namespace NUMINAMATH_CALUDE_sad_children_count_l1565_156553

theorem sad_children_count (total : ℕ) (happy : ℕ) (neither : ℕ) :
  total = 60 →
  happy = 30 →
  neither = 20 →
  total - (happy + neither) = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_sad_children_count_l1565_156553


namespace NUMINAMATH_CALUDE_complex_numbers_satisfying_conditions_l1565_156561

theorem complex_numbers_satisfying_conditions :
  ∀ z : ℂ,
    (∃ t : ℝ, z + 10 / z = t ∧ 1 < t ∧ t ≤ 6) ∧
    (∃ a b : ℤ, z = ↑a + ↑b * I) →
    z = 1 + 3 * I ∨ z = 1 - 3 * I ∨ z = 3 + I ∨ z = 3 - I :=
by sorry

end NUMINAMATH_CALUDE_complex_numbers_satisfying_conditions_l1565_156561


namespace NUMINAMATH_CALUDE_patrick_caught_eight_l1565_156597

/-- The number of fish caught by each person -/
structure FishCaught where
  patrick : ℕ
  angus : ℕ
  ollie : ℕ

/-- The conditions of the fishing problem -/
def fishing_conditions (fc : FishCaught) : Prop :=
  fc.angus = fc.patrick + 4 ∧
  fc.ollie = fc.angus - 7 ∧
  fc.ollie = 5

/-- Theorem: Given the fishing conditions, Patrick caught 8 fish -/
theorem patrick_caught_eight (fc : FishCaught) 
  (h : fishing_conditions fc) : fc.patrick = 8 := by
  sorry

end NUMINAMATH_CALUDE_patrick_caught_eight_l1565_156597


namespace NUMINAMATH_CALUDE_area_triangle_BXD_l1565_156515

/-- Represents a trapezoid with given properties -/
structure Trapezoid where
  base_AB : ℝ
  base_CD : ℝ
  area : ℝ

/-- Theorem about the area of triangle BXD in a trapezoid -/
theorem area_triangle_BXD (ABCD : Trapezoid) (h1 : ABCD.base_AB = 24)
    (h2 : ABCD.base_CD = 36) (h3 : ABCD.area = 360) : ℝ := by
  -- The area of triangle BXD is 57.6 square units
  sorry

#check area_triangle_BXD

end NUMINAMATH_CALUDE_area_triangle_BXD_l1565_156515


namespace NUMINAMATH_CALUDE_gcd_765432_654321_l1565_156524

theorem gcd_765432_654321 : Nat.gcd 765432 654321 = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_765432_654321_l1565_156524


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1565_156564

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 - 2*x - 3 < 0} = {x : ℝ | -1 < x ∧ x < 3} := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1565_156564


namespace NUMINAMATH_CALUDE_a_10_value_l1565_156598

/-- An arithmetic sequence with given conditions -/
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  a 3 = 5 ∧ a 7 = -7 ∧ ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

/-- Theorem: In the given arithmetic sequence, a_10 = -16 -/
theorem a_10_value (a : ℕ → ℤ) (h : arithmetic_sequence a) : a 10 = -16 := by
  sorry

end NUMINAMATH_CALUDE_a_10_value_l1565_156598


namespace NUMINAMATH_CALUDE_fox_initial_money_l1565_156577

/-- The number of times Fox crosses the bridge -/
def num_crossings : ℕ := 4

/-- The toll paid after each crossing -/
def toll : ℕ := 50

/-- The initial toll paid before the first crossing -/
def initial_toll : ℕ := 10

/-- The function that calculates Fox's money after each crossing -/
def money_after_crossing (initial_money : ℕ) (crossing : ℕ) : ℤ :=
  (2^crossing) * (initial_money - initial_toll) - 
  (2^crossing - 1) * toll - 
  initial_toll

/-- The theorem stating that Fox started with 56 coins -/
theorem fox_initial_money : 
  ∃ (initial_money : ℕ), 
    initial_money = 56 ∧ 
    money_after_crossing initial_money num_crossings = 0 :=
  sorry

end NUMINAMATH_CALUDE_fox_initial_money_l1565_156577


namespace NUMINAMATH_CALUDE_cos_C_value_angle_C_measure_l1565_156507

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ

-- Part 1
theorem cos_C_value (abc : Triangle) 
  (h1 : Real.sin abc.A = 5/13) 
  (h2 : Real.cos abc.B = 3/5) : 
  Real.cos abc.C = -16/65 := by sorry

-- Part 2
theorem angle_C_measure (abc : Triangle) 
  (h : ∃ p : ℝ, (Real.tan abc.A)^2 + p * (Real.tan abc.A + 1) + 1 = 0 ∧ 
                (Real.tan abc.B)^2 + p * (Real.tan abc.B + 1) + 1 = 0) : 
  abc.C = 3 * Real.pi / 4 := by sorry

end NUMINAMATH_CALUDE_cos_C_value_angle_C_measure_l1565_156507


namespace NUMINAMATH_CALUDE_g_range_g_range_achieves_bounds_l1565_156587

noncomputable def g (x : ℝ) : ℝ := 
  (Real.arcsin (x/3))^2 - 2*Real.pi * Real.arccos (x/3) + (Real.arccos (x/3))^2 + 
  (Real.pi^2/4) * (x^2 - 9*x + 27)

theorem g_range : 
  ∀ y ∈ Set.range g, -3*(Real.pi^2/4) ≤ y ∧ y ≤ 33*(Real.pi^2/4) :=
by sorry

theorem g_range_achieves_bounds : 
  ∃ x₁ x₂ : ℝ, x₁ ∈ Set.Icc (-3) 3 ∧ x₂ ∈ Set.Icc (-3) 3 ∧ 
  g x₁ = -3*(Real.pi^2/4) ∧ g x₂ = 33*(Real.pi^2/4) :=
by sorry

end NUMINAMATH_CALUDE_g_range_g_range_achieves_bounds_l1565_156587


namespace NUMINAMATH_CALUDE_complex_number_in_third_quadrant_l1565_156549

/-- The complex number z = i² + i³ corresponds to a point in the third quadrant of the complex plane -/
theorem complex_number_in_third_quadrant :
  let z : ℂ := Complex.I^2 + Complex.I^3
  (z.re < 0) ∧ (z.im < 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_third_quadrant_l1565_156549


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1565_156531

/-- A positive geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q ∧ a n > 0

theorem geometric_sequence_property
  (a : ℕ → ℝ)
  (q : ℝ)
  (h_geo : GeometricSequence a q)
  (h_cond : 2 * a 5 = a 3 - a 4)
  (n m : ℕ)
  (h_terms : a 1 = 4 * Real.sqrt (a n * a m)) :
  n + m = 6 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1565_156531


namespace NUMINAMATH_CALUDE_dilution_correct_l1565_156556

/-- The amount of pure alcohol needed to dilute iodine tincture -/
def alcohol_amount : ℝ := 2275

/-- The initial amount of iodine tincture in grams -/
def initial_tincture : ℝ := 350

/-- The initial iodine content as a percentage -/
def initial_content : ℝ := 15

/-- The desired iodine content as a percentage -/
def desired_content : ℝ := 2

/-- Theorem stating that adding the calculated amount of alcohol results in the desired iodine content -/
theorem dilution_correct : 
  (initial_tincture * initial_content) / (initial_tincture + alcohol_amount) = desired_content := by
  sorry

end NUMINAMATH_CALUDE_dilution_correct_l1565_156556


namespace NUMINAMATH_CALUDE_card_shuffle_bound_l1565_156582

theorem card_shuffle_bound (n : ℕ) (hn : n > 0) : 
  Nat.totient (2 * n - 1) ≤ 2 * n - 2 := by
  sorry

end NUMINAMATH_CALUDE_card_shuffle_bound_l1565_156582


namespace NUMINAMATH_CALUDE_bryden_receive_amount_l1565_156563

/-- The face value of a quarter in dollars -/
def quarter_value : ℚ := 0.25

/-- The number of quarters Bryden has -/
def bryden_quarters : ℕ := 7

/-- The percentage the collector offers for state quarters -/
def collector_offer_percent : ℕ := 2500

/-- Calculate the amount Bryden will receive from the collector -/
def bryden_receive : ℚ :=
  (quarter_value * bryden_quarters) * (collector_offer_percent / 100)

/-- Theorem stating that Bryden will receive $43.75 from the collector -/
theorem bryden_receive_amount :
  bryden_receive = 43.75 := by sorry

end NUMINAMATH_CALUDE_bryden_receive_amount_l1565_156563


namespace NUMINAMATH_CALUDE_sin_cos_identity_l1565_156580

theorem sin_cos_identity :
  Real.sin (20 * π / 180) ^ 2 + Real.cos (50 * π / 180) ^ 2 + 
  Real.sin (20 * π / 180) * Real.cos (50 * π / 180) = 1 :=
by sorry

end NUMINAMATH_CALUDE_sin_cos_identity_l1565_156580


namespace NUMINAMATH_CALUDE_not_sufficient_for_congruence_l1565_156520

/-- Two triangles are congruent -/
def triangles_congruent (A B C D E F : Point) : Prop := sorry

/-- The measure of an angle -/
def angle_measure (A B C : Point) : ℝ := sorry

/-- The length of a line segment -/
def segment_length (A B : Point) : ℝ := sorry

/-- Theorem: Given ∠A = ∠F, ∠B = ∠E, and AC = DE, it's not sufficient to determine 
    the congruence of triangles ABC and DEF -/
theorem not_sufficient_for_congruence 
  (A B C D E F : Point) 
  (h1 : angle_measure A B C = angle_measure F E D)
  (h2 : angle_measure B A C = angle_measure E F D)
  (h3 : segment_length A C = segment_length D E) :
  ¬ (triangles_congruent A B C D E F) := by sorry

end NUMINAMATH_CALUDE_not_sufficient_for_congruence_l1565_156520


namespace NUMINAMATH_CALUDE_largest_number_l1565_156541

def toDecimal (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * base ^ i) 0

def A : Nat := toDecimal [5, 8] 9
def B : Nat := toDecimal [0, 1, 2] 6
def C : Nat := toDecimal [0, 0, 0, 1] 4
def D : Nat := toDecimal [1, 1, 1, 1, 1] 2

theorem largest_number : 
  B > A ∧ B > C ∧ B > D := by sorry

end NUMINAMATH_CALUDE_largest_number_l1565_156541


namespace NUMINAMATH_CALUDE_linear_function_property_l1565_156591

/-- A linear function is a function of the form f(x) = mx + b where m and b are constants. -/
def LinearFunction (f : ℝ → ℝ) : Prop :=
  ∃ m b : ℝ, ∀ x, f x = m * x + b

theorem linear_function_property (g : ℝ → ℝ) 
  (hlinear : LinearFunction g) 
  (hg : g 8 - g 4 = 16) : 
  g 16 - g 4 = 48 := by
sorry

end NUMINAMATH_CALUDE_linear_function_property_l1565_156591


namespace NUMINAMATH_CALUDE_or_true_iff_not_and_not_false_l1565_156518

theorem or_true_iff_not_and_not_false (p q : Prop) :
  (p ∨ q) ↔ ¬(¬p ∧ ¬q) :=
sorry

end NUMINAMATH_CALUDE_or_true_iff_not_and_not_false_l1565_156518


namespace NUMINAMATH_CALUDE_reciprocal_opposite_sum_l1565_156534

theorem reciprocal_opposite_sum (a b c d : ℝ) 
  (h1 : a * b = 1)  -- a and b are reciprocals
  (h2 : c + d = 0)  -- c and d are opposites
  : 2*c + 2*d - 3*a*b = -3 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_opposite_sum_l1565_156534


namespace NUMINAMATH_CALUDE_sin_18_cos_12_plus_cos_18_sin_12_l1565_156552

theorem sin_18_cos_12_plus_cos_18_sin_12 :
  Real.sin (18 * π / 180) * Real.cos (12 * π / 180) + 
  Real.cos (18 * π / 180) * Real.sin (12 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_18_cos_12_plus_cos_18_sin_12_l1565_156552


namespace NUMINAMATH_CALUDE_bonus_distribution_l1565_156544

theorem bonus_distribution (total_bonus : ℕ) (difference : ℕ) (junior_share : ℕ) : 
  total_bonus = 5000 →
  difference = 1200 →
  junior_share + (junior_share + difference) = total_bonus →
  junior_share = 1900 := by
sorry

end NUMINAMATH_CALUDE_bonus_distribution_l1565_156544
