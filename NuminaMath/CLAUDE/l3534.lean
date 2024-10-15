import Mathlib

namespace NUMINAMATH_CALUDE_unique_solution_l3534_353408

theorem unique_solution :
  ∃! (A B C D : ℕ),
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
    A ≤ 9 ∧ B ≤ 9 ∧ C ≤ 9 ∧ D ≤ 9 ∧
    1000 * A + 100 * A + 10 * B + C - (1000 * B + 100 * A + 10 * C + B) = 1000 * A + 100 * B + 10 * C + D ∧
    A = 9 ∧ B = 6 ∧ C = 8 ∧ D = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l3534_353408


namespace NUMINAMATH_CALUDE_total_goals_theorem_l3534_353492

/-- Represents the number of goals scored by Louie in his last match -/
def louie_last_match_goals : ℕ := 4

/-- Represents the number of goals scored by Louie in previous matches -/
def louie_previous_goals : ℕ := 40

/-- Represents the number of seasons Donnie has played -/
def donnie_seasons : ℕ := 3

/-- Represents the number of games in each season -/
def games_per_season : ℕ := 50

/-- Represents the initial number of goals scored by Annie in her first game -/
def annie_initial_goals : ℕ := 2

/-- Represents the increase in Annie's goals per game -/
def annie_goal_increase : ℕ := 2

/-- Represents the number of seasons Annie has played -/
def annie_seasons : ℕ := 2

/-- Theorem stating that the total number of goals scored by all siblings is 11,344 -/
theorem total_goals_theorem :
  let louie_total := louie_last_match_goals + louie_previous_goals
  let donnie_total := 2 * louie_last_match_goals * donnie_seasons * games_per_season
  let annie_games := annie_seasons * games_per_season
  let annie_total := annie_games * (annie_initial_goals + annie_initial_goals + (annie_games - 1) * annie_goal_increase) / 2
  louie_total + donnie_total + annie_total = 11344 := by
  sorry

end NUMINAMATH_CALUDE_total_goals_theorem_l3534_353492


namespace NUMINAMATH_CALUDE_janet_hourly_earnings_l3534_353463

/-- Calculates Janet's hourly earnings for moderating social media posts -/
theorem janet_hourly_earnings (cents_per_post : ℚ) (seconds_per_post : ℕ) : 
  cents_per_post = 25 → seconds_per_post = 10 → 
  (3600 / seconds_per_post) * cents_per_post = 9000 := by
  sorry

#check janet_hourly_earnings

end NUMINAMATH_CALUDE_janet_hourly_earnings_l3534_353463


namespace NUMINAMATH_CALUDE_no_perfect_cubes_l3534_353473

theorem no_perfect_cubes (a b : ℤ) : ¬(∃ x y : ℤ, a^5*b + 3 = x^3 ∧ a*b^5 + 3 = y^3) := by
  sorry

end NUMINAMATH_CALUDE_no_perfect_cubes_l3534_353473


namespace NUMINAMATH_CALUDE_fence_perimeter_is_106_l3534_353491

/-- Given a square field enclosed by posts, calculates the outer perimeter of the fence. -/
def fence_perimeter (num_posts : ℕ) (post_width : ℝ) (gap : ℝ) : ℝ :=
  let posts_per_side : ℕ := (num_posts - 4) / 4 + 2
  let gaps_per_side : ℕ := posts_per_side - 1
  let side_length : ℝ := gaps_per_side * gap + posts_per_side * post_width
  4 * side_length

/-- Theorem stating that the fence with given specifications has a perimeter of 106 feet. -/
theorem fence_perimeter_is_106 :
  fence_perimeter 16 0.5 6 = 106 := by
  sorry

#eval fence_perimeter 16 0.5 6

end NUMINAMATH_CALUDE_fence_perimeter_is_106_l3534_353491


namespace NUMINAMATH_CALUDE_new_job_bonus_calculation_l3534_353483

/-- Represents Maisy's job options and earnings -/
structure JobOption where
  hours_per_week : ℕ
  hourly_wage : ℕ
  bonus : ℕ

/-- Calculates the weekly earnings for a job option -/
def weekly_earnings (job : JobOption) : ℕ :=
  job.hours_per_week * job.hourly_wage + job.bonus

theorem new_job_bonus_calculation (current_job new_job : JobOption) 
  (h1 : current_job.hours_per_week = 8)
  (h2 : current_job.hourly_wage = 10)
  (h3 : current_job.bonus = 0)
  (h4 : new_job.hours_per_week = 4)
  (h5 : new_job.hourly_wage = 15)
  (h6 : weekly_earnings new_job = weekly_earnings current_job + 15) :
  new_job.bonus = 15 := by
  sorry

#check new_job_bonus_calculation

end NUMINAMATH_CALUDE_new_job_bonus_calculation_l3534_353483


namespace NUMINAMATH_CALUDE_only_setC_is_right_triangle_l3534_353466

-- Define a function to check if three numbers satisfy the Pythagorean theorem
def isPythagoreanTriple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a

-- Define the sets of line segments
def setA : List ℕ := [1, 2, 3]
def setB : List ℕ := [5, 11, 12]
def setC : List ℕ := [5, 12, 13]
def setD : List ℕ := [6, 8, 9]

-- Theorem stating that only setC forms a right triangle
theorem only_setC_is_right_triangle :
  (¬ isPythagoreanTriple setA[0]! setA[1]! setA[2]!) ∧
  (¬ isPythagoreanTriple setB[0]! setB[1]! setB[2]!) ∧
  (isPythagoreanTriple setC[0]! setC[1]! setC[2]!) ∧
  (¬ isPythagoreanTriple setD[0]! setD[1]! setD[2]!) :=
by sorry

end NUMINAMATH_CALUDE_only_setC_is_right_triangle_l3534_353466


namespace NUMINAMATH_CALUDE_rice_mixture_cost_l3534_353449

/-- The cost of a mixture of two rice varieties -/
def mixture_cost (c1 c2 r : ℚ) : ℚ :=
  (c1 * r + c2 * 1) / (r + 1)

theorem rice_mixture_cost :
  let c1 : ℚ := 5.5
  let c2 : ℚ := 8.75
  let r : ℚ := 5/8
  mixture_cost c1 c2 r = 7.5 := by
sorry

end NUMINAMATH_CALUDE_rice_mixture_cost_l3534_353449


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3534_353493

-- Define the function f
def f (m n : ℕ) : ℕ := Nat.choose 6 m * Nat.choose 4 n

-- State the theorem
theorem sum_of_coefficients : f 3 0 + f 2 1 + f 1 2 + f 0 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3534_353493


namespace NUMINAMATH_CALUDE_probability_at_least_one_woman_l3534_353421

def num_men : ℕ := 10
def num_women : ℕ := 5
def num_chosen : ℕ := 4

theorem probability_at_least_one_woman :
  let total := num_men + num_women
  (1 - (Nat.choose num_men num_chosen : ℚ) / (Nat.choose total num_chosen : ℚ)) = 77 / 91 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_woman_l3534_353421


namespace NUMINAMATH_CALUDE_negative_one_squared_plus_cubed_equals_zero_l3534_353427

theorem negative_one_squared_plus_cubed_equals_zero :
  (-1 : ℤ)^2 + (-1 : ℤ)^3 = 0 := by sorry

end NUMINAMATH_CALUDE_negative_one_squared_plus_cubed_equals_zero_l3534_353427


namespace NUMINAMATH_CALUDE_roots_sequence_sum_l3534_353478

theorem roots_sequence_sum (p q a b : ℝ) : 
  p > 0 → 
  q > 0 → 
  a ≠ b →
  a^2 - p*a + q = 0 →
  b^2 - p*b + q = 0 →
  (∃ d : ℝ, (a = -4 + d ∧ b = -4 + 2*d) ∨ (b = -4 + d ∧ a = -4 + 2*d)) →
  (∃ r : ℝ, (a = -4*r ∧ b = -4*r^2) ∨ (b = -4*r ∧ a = -4*r^2)) →
  p + q = 26 := by
sorry

end NUMINAMATH_CALUDE_roots_sequence_sum_l3534_353478


namespace NUMINAMATH_CALUDE_triangle_angle_expression_range_l3534_353416

theorem triangle_angle_expression_range (A B C : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  -25/16 < 3 * Real.cos A + 2 * Real.cos (2 * B) + Real.cos (3 * C) ∧ 
  3 * Real.cos A + 2 * Real.cos (2 * B) + Real.cos (3 * C) < 6 :=
sorry

end NUMINAMATH_CALUDE_triangle_angle_expression_range_l3534_353416


namespace NUMINAMATH_CALUDE_frog_paths_count_l3534_353496

/-- Represents a triangular grid -/
structure TriangularGrid :=
  (top_row_squares : ℕ)
  (total_squares : ℕ)

/-- Represents the possible moves of the frog -/
inductive Move
  | down
  | down_left

/-- Calculates the number of distinct paths in a triangular grid -/
def count_distinct_paths (grid : TriangularGrid) : ℕ :=
  sorry

/-- Theorem stating the number of distinct paths in the specific grid -/
theorem frog_paths_count (grid : TriangularGrid) 
  (h1 : grid.top_row_squares = 5)
  (h2 : grid.total_squares = 29) :
  count_distinct_paths grid = 256 :=
sorry

end NUMINAMATH_CALUDE_frog_paths_count_l3534_353496


namespace NUMINAMATH_CALUDE_garden_length_is_32_l3534_353460

/-- Calculates the length of a garden with mango trees -/
def garden_length (num_columns : ℕ) (tree_distance : ℝ) (boundary : ℝ) : ℝ :=
  (num_columns - 1 : ℝ) * tree_distance + 2 * boundary

/-- Theorem: The length of the garden is 32 meters -/
theorem garden_length_is_32 :
  garden_length 12 2 5 = 32 := by
  sorry

end NUMINAMATH_CALUDE_garden_length_is_32_l3534_353460


namespace NUMINAMATH_CALUDE_colored_pencils_erasers_difference_l3534_353451

/-- Proves that the difference between colored pencils and erasers left is 22 --/
theorem colored_pencils_erasers_difference :
  let initial_crayons : ℕ := 531
  let initial_erasers : ℕ := 38
  let initial_colored_pencils : ℕ := 67
  let final_crayons : ℕ := 391
  let final_erasers : ℕ := 28
  let final_colored_pencils : ℕ := 50
  final_colored_pencils - final_erasers = 22 := by
  sorry

end NUMINAMATH_CALUDE_colored_pencils_erasers_difference_l3534_353451


namespace NUMINAMATH_CALUDE_cos_squared_minus_sin_squared_pi_over_twelve_l3534_353462

theorem cos_squared_minus_sin_squared_pi_over_twelve (π : Real) :
  (Real.cos (π / 12))^2 - (Real.sin (π / 12))^2 = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_squared_minus_sin_squared_pi_over_twelve_l3534_353462


namespace NUMINAMATH_CALUDE_problem_solution_l3534_353422

theorem problem_solution (a b c d m : ℝ) 
  (h1 : a = -b)  -- a and b are opposite numbers
  (h2 : c * d = 1)  -- c and d are reciprocals
  (h3 : |m| = 3)  -- absolute value of m is 3
  : (a + b) / 2023 - 4 * c * d + m^2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3534_353422


namespace NUMINAMATH_CALUDE_equation_solution_l3534_353471

theorem equation_solution : 
  let f (x : ℝ) := 1 / ((x - 1) * (x - 2)) + 1 / ((x - 2) * (x - 3)) + 
                   1 / ((x - 3) * (x - 4)) + 1 / ((x - 4) * (x - 5))
  ∀ x : ℝ, f x = 1 / 10 ↔ x = 10 ∨ x = -3.5 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3534_353471


namespace NUMINAMATH_CALUDE_soccer_stars_league_teams_l3534_353488

theorem soccer_stars_league_teams (n : ℕ) : n > 1 → (n * (n - 1)) / 2 = 28 → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_soccer_stars_league_teams_l3534_353488


namespace NUMINAMATH_CALUDE_lanas_boxes_l3534_353497

/-- Given that each box contains 7 pieces of clothing and the total number of pieces is 21,
    prove that the number of boxes is 3. -/
theorem lanas_boxes (pieces_per_box : ℕ) (total_pieces : ℕ) (h1 : pieces_per_box = 7) (h2 : total_pieces = 21) :
  total_pieces / pieces_per_box = 3 := by
  sorry

end NUMINAMATH_CALUDE_lanas_boxes_l3534_353497


namespace NUMINAMATH_CALUDE_four_good_points_l3534_353468

/-- A point (x, y) is a "good point" if x is an integer, y is a perfect square,
    and y = (x - 90)^2 - 4907 -/
def is_good_point (x y : ℤ) : Prop :=
  ∃ (m : ℤ), y = m^2 ∧ y = (x - 90)^2 - 4907

/-- The set of all "good points" -/
def good_points : Set (ℤ × ℤ) :=
  {p | is_good_point p.1 p.2}

/-- The theorem stating that there are exactly four "good points" -/
theorem four_good_points :
  good_points = {(444, 120409), (-264, 120409), (2544, 6017209), (-2364, 6017209)} := by
  sorry

#check four_good_points

end NUMINAMATH_CALUDE_four_good_points_l3534_353468


namespace NUMINAMATH_CALUDE_trees_planted_l3534_353457

/-- Given the initial number of short trees and the final number after planting,
    prove that the number of trees planted is the difference between these two values. -/
theorem trees_planted (initial_short_trees final_short_trees : ℕ) :
  final_short_trees ≥ initial_short_trees →
  final_short_trees - initial_short_trees = final_short_trees - initial_short_trees :=
by
  sorry

/-- Solve the specific problem instance -/
def solve_tree_planting_problem : ℕ :=
  98 - 41

#eval solve_tree_planting_problem

end NUMINAMATH_CALUDE_trees_planted_l3534_353457


namespace NUMINAMATH_CALUDE_intersection_point_is_unique_l3534_353477

/-- The intersection point of two lines -/
def intersection_point : ℚ × ℚ := (-18/17, 46/17)

/-- First line equation: 3y = -2x + 6 -/
def line1 (x y : ℚ) : Prop := 3 * y = -2 * x + 6

/-- Second line equation: 2y = -7x - 2 -/
def line2 (x y : ℚ) : Prop := 2 * y = -7 * x - 2

theorem intersection_point_is_unique :
  ∃! p : ℚ × ℚ, line1 p.1 p.2 ∧ line2 p.1 p.2 ∧ p = intersection_point :=
sorry

end NUMINAMATH_CALUDE_intersection_point_is_unique_l3534_353477


namespace NUMINAMATH_CALUDE_endpoint_coordinate_sum_l3534_353433

/-- Given a line segment with one endpoint (6, -2) and midpoint (3, 5),
    the sum of coordinates of the other endpoint is 12. -/
theorem endpoint_coordinate_sum : 
  ∀ (x y : ℝ), 
  (6 + x) / 2 = 3 ∧ (-2 + y) / 2 = 5 → x + y = 12 := by
  sorry

end NUMINAMATH_CALUDE_endpoint_coordinate_sum_l3534_353433


namespace NUMINAMATH_CALUDE_parallelogram_area_theorem_l3534_353434

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a parallelogram -/
structure Parallelogram :=
  (A B C D : Point)

/-- Represents a triangle -/
structure Triangle :=
  (A B C : Point)

/-- Represents a quadrilateral -/
structure Quadrilateral :=
  (A B C D : Point)

/-- Calculates the area of a triangle -/
def areaTriangle (t : Triangle) : ℝ := sorry

/-- Calculates the area of a quadrilateral -/
def areaQuadrilateral (q : Quadrilateral) : ℝ := sorry

/-- Checks if a point is the midpoint of a line segment -/
def isMidpoint (M A B : Point) : Prop := sorry

/-- Checks if three points are collinear -/
def collinear (A B C : Point) : Prop := sorry

theorem parallelogram_area_theorem (ABCD : Parallelogram) (E F : Point) :
  collinear C E F →
  isMidpoint F ABCD.A ABCD.B →
  areaTriangle ⟨ABCD.B, E, C⟩ = 100 →
  areaQuadrilateral ⟨ABCD.A, F, E, ABCD.D⟩ = 250 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_theorem_l3534_353434


namespace NUMINAMATH_CALUDE_total_vowels_written_l3534_353441

/-- The number of vowels in the English alphabet -/
def num_vowels : ℕ := 5

/-- The number of times each vowel is written -/
def times_written : ℕ := 4

/-- Theorem: The total number of vowels written on the board is 20 -/
theorem total_vowels_written : num_vowels * times_written = 20 := by
  sorry

end NUMINAMATH_CALUDE_total_vowels_written_l3534_353441


namespace NUMINAMATH_CALUDE_opposites_sum_to_zero_l3534_353485

theorem opposites_sum_to_zero (a b : ℚ) (h : a = -b) : a + b = 0 := by
  sorry

end NUMINAMATH_CALUDE_opposites_sum_to_zero_l3534_353485


namespace NUMINAMATH_CALUDE_tan_alpha_value_l3534_353424

theorem tan_alpha_value (α β : ℝ) 
  (h1 : Real.tan (3 * α - 2 * β) = 1 / 2)
  (h2 : Real.tan (5 * α - 4 * β) = 1 / 4) : 
  Real.tan α = 13 / 16 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l3534_353424


namespace NUMINAMATH_CALUDE_product_calculation_l3534_353418

theorem product_calculation : 1500 * 2023 * 0.5023 * 50 = 306903675 := by
  sorry

end NUMINAMATH_CALUDE_product_calculation_l3534_353418


namespace NUMINAMATH_CALUDE_inequality_proof_l3534_353456

theorem inequality_proof (a b : ℝ) (h : a > b) : a^2 - a*b > b*a - b^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3534_353456


namespace NUMINAMATH_CALUDE_sum_of_xyz_l3534_353455

theorem sum_of_xyz (x y z : ℝ) 
  (eq1 : x = y + z + 2)
  (eq2 : y = z + x + 1)
  (eq3 : z = x + y + 4) :
  x + y + z = -7 := by sorry

end NUMINAMATH_CALUDE_sum_of_xyz_l3534_353455


namespace NUMINAMATH_CALUDE_shekhar_shobha_age_ratio_l3534_353406

/-- The ratio of Shekhar's age to Shobha's age -/
def age_ratio (shekhar_age shobha_age : ℕ) : ℚ :=
  shekhar_age / shobha_age

/-- Theorem stating the ratio of Shekhar's age to Shobha's age -/
theorem shekhar_shobha_age_ratio :
  ∃ (shekhar_age : ℕ),
    shekhar_age + 6 = 26 ∧
    age_ratio shekhar_age 15 = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_shekhar_shobha_age_ratio_l3534_353406


namespace NUMINAMATH_CALUDE_seating_arrangements_count_l3534_353479

/-- Represents a seating arrangement -/
def SeatingArrangement := Fin 12 → Fin 12

/-- Represents a married couple -/
structure Couple := (husband : Fin 6) (wife : Fin 6)

/-- Represents a profession -/
def Profession := Fin 3

/-- Check if two positions are adjacent or opposite on a 12-seat round table -/
def isAdjacentOrOpposite (a b : Fin 12) : Prop := 
  (a + 1 = b) ∨ (b + 1 = a) ∨ (a + 6 = b) ∨ (b + 6 = a)

/-- Check if a seating arrangement is valid -/
def isValidArrangement (s : SeatingArrangement) (couples : Fin 6 → Couple) (professions : Fin 12 → Profession) : Prop :=
  ∀ i j : Fin 12, 
    -- Men and women alternate
    (i.val % 2 = 0 ↔ j.val % 2 = 1) →
    -- No one sits next to or across from their spouse
    (∃ c : Fin 6, (couples c).husband = s i ∧ (couples c).wife = s j) →
    ¬ isAdjacentOrOpposite i j ∧
    -- No one sits next to someone of the same profession
    (isAdjacentOrOpposite i j → professions (s i) ≠ professions (s j))

/-- The main theorem stating the number of valid seating arrangements -/
theorem seating_arrangements_count :
  ∃ (arrangements : Finset SeatingArrangement) (couples : Fin 6 → Couple) (professions : Fin 12 → Profession),
    arrangements.card = 2880 ∧
    ∀ s ∈ arrangements, isValidArrangement s couples professions :=
sorry

end NUMINAMATH_CALUDE_seating_arrangements_count_l3534_353479


namespace NUMINAMATH_CALUDE_mass_of_cao_l3534_353445

/-- Calculates the mass of a given number of moles of a compound -/
def calculate_mass (moles : ℝ) (atomic_mass_ca : ℝ) (atomic_mass_o : ℝ) : ℝ :=
  moles * (atomic_mass_ca + atomic_mass_o)

/-- Theorem: The mass of 8 moles of CaO containing only 42Ca is 464 grams -/
theorem mass_of_cao : calculate_mass 8 42 16 = 464 := by
  sorry

end NUMINAMATH_CALUDE_mass_of_cao_l3534_353445


namespace NUMINAMATH_CALUDE_largest_power_dividing_factorial_l3534_353410

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem largest_power_dividing_factorial :
  ∃ (n : ℕ), n = 7 ∧ 
  (∀ m : ℕ, m > n → ¬(factorial 30 % (18^m) = 0)) ∧
  (factorial 30 % (18^n) = 0) :=
by sorry

end NUMINAMATH_CALUDE_largest_power_dividing_factorial_l3534_353410


namespace NUMINAMATH_CALUDE_polynomial_remainder_l3534_353452

def polynomial (x : ℝ) : ℝ := 6*x^8 - 2*x^7 - 10*x^6 + 3*x^4 + 5*x^3 - 15

def divisor (x : ℝ) : ℝ := 3*x - 6

theorem polynomial_remainder :
  ∃ (q : ℝ → ℝ), ∀ x, polynomial x = q x * divisor x + 713 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l3534_353452


namespace NUMINAMATH_CALUDE_line_parallel_perpendicular_characterization_l3534_353403

/-- Two lines in 3D space -/
structure Line3D where
  m : ℝ
  n : ℝ
  p : ℝ

/-- Condition for two lines to be parallel -/
def parallel (l₁ l₂ : Line3D) : Prop :=
  l₁.m / l₂.m = l₁.n / l₂.n ∧ l₁.n / l₂.n = l₁.p / l₂.p

/-- Condition for two lines to be perpendicular -/
def perpendicular (l₁ l₂ : Line3D) : Prop :=
  l₁.m * l₂.m + l₁.n * l₂.n + l₁.p * l₂.p = 0

/-- Theorem: Characterization of parallel and perpendicular lines in 3D space -/
theorem line_parallel_perpendicular_characterization (l₁ l₂ : Line3D) :
  (parallel l₁ l₂ ↔ l₁.m / l₂.m = l₁.n / l₂.n ∧ l₁.n / l₂.n = l₁.p / l₂.p) ∧
  (perpendicular l₁ l₂ ↔ l₁.m * l₂.m + l₁.n * l₂.n + l₁.p * l₂.p = 0) := by
  sorry

end NUMINAMATH_CALUDE_line_parallel_perpendicular_characterization_l3534_353403


namespace NUMINAMATH_CALUDE_isosceles_triangle_obtuse_iff_quadratic_roots_l3534_353431

theorem isosceles_triangle_obtuse_iff_quadratic_roots 
  (A B C : Real) 
  (triangle_sum : A + B + C = π) 
  (isosceles : A = C) : 
  (B > π / 2) ↔ 
  ∃ (x₁ x₂ : Real), x₁ ≠ x₂ ∧ A * x₁^2 + B * x₁ + C = 0 ∧ A * x₂^2 + B * x₂ + C = 0 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_obtuse_iff_quadratic_roots_l3534_353431


namespace NUMINAMATH_CALUDE_root_product_fourth_power_l3534_353454

theorem root_product_fourth_power (r s t : ℂ) : 
  (r^3 + 5*r + 4 = 0) → 
  (s^3 + 5*s + 4 = 0) → 
  (t^3 + 5*t + 4 = 0) → 
  (r+s)^4 * (s+t)^4 * (t+r)^4 = 256 := by
sorry

end NUMINAMATH_CALUDE_root_product_fourth_power_l3534_353454


namespace NUMINAMATH_CALUDE_periodic_decimal_is_rational_l3534_353415

/-- Represents a periodic decimal fraction -/
structure PeriodicDecimal where
  nonRepeatingPart : List Nat
  repeatingPart : List Nat
  nonEmpty : repeatingPart.length > 0

/-- Converts a PeriodicDecimal to a real number -/
noncomputable def toReal (x : PeriodicDecimal) : Real := sorry

/-- Theorem: Every periodic decimal fraction is a rational number -/
theorem periodic_decimal_is_rational (x : PeriodicDecimal) :
  ∃ (p q : Int), q ≠ 0 ∧ toReal x = p / q := by sorry

end NUMINAMATH_CALUDE_periodic_decimal_is_rational_l3534_353415


namespace NUMINAMATH_CALUDE_intersection_implies_k_geq_two_l3534_353486

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 2}
def N (k : ℝ) : Set ℝ := {x : ℝ | x - k ≤ 0}

-- State the theorem
theorem intersection_implies_k_geq_two (k : ℝ) : M ∩ N k = M → k ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_k_geq_two_l3534_353486


namespace NUMINAMATH_CALUDE_x_cubed_coefficient_l3534_353417

theorem x_cubed_coefficient (p q : Polynomial ℤ) : 
  p = 3 * X^3 + 2 * X^2 + 5 * X + 6 →
  q = 4 * X^3 + 7 * X^2 + 9 * X + 8 →
  (p * q).coeff 3 = 83 := by
  sorry

end NUMINAMATH_CALUDE_x_cubed_coefficient_l3534_353417


namespace NUMINAMATH_CALUDE_sample_size_is_sampled_l3534_353459

/-- A survey about middle school students riding electric bikes to school -/
structure Survey where
  population : ℕ
  sampled : ℕ
  negative_attitude : ℕ

/-- The sample size of a survey is equal to the number of people sampled -/
theorem sample_size_is_sampled (s : Survey) (h : s.population = 823 ∧ s.sampled = 150 ∧ s.negative_attitude = 136) : 
  s.sampled = 150 := by
  sorry

end NUMINAMATH_CALUDE_sample_size_is_sampled_l3534_353459


namespace NUMINAMATH_CALUDE_exists_m_not_greater_l3534_353402

theorem exists_m_not_greater (a b : ℝ) (h : a < b) : ∃ m : ℝ, m * a ≤ m * b := by
  sorry

end NUMINAMATH_CALUDE_exists_m_not_greater_l3534_353402


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3534_353490

-- Define set A
def A : Set ℝ := {x | x^2 - 4*x < 0}

-- Define set B
def B : Set ℝ := {x | 1/3 ≤ x ∧ x ≤ 5}

-- Theorem statement
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 1/3 ≤ x ∧ x < 4} :=
by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3534_353490


namespace NUMINAMATH_CALUDE_meters_in_one_kilometer_l3534_353480

/-- Conversion factor from kilometers to hectometers -/
def km_to_hm : ℝ := 5

/-- Conversion factor from hectometers to dekameters -/
def hm_to_dam : ℝ := 10

/-- Conversion factor from dekameters to meters -/
def dam_to_m : ℝ := 15

/-- The number of meters in one kilometer -/
def meters_in_km : ℝ := km_to_hm * hm_to_dam * dam_to_m

theorem meters_in_one_kilometer :
  meters_in_km = 750 := by sorry

end NUMINAMATH_CALUDE_meters_in_one_kilometer_l3534_353480


namespace NUMINAMATH_CALUDE_product_pure_imaginary_l3534_353409

theorem product_pure_imaginary (a : ℝ) : 
  let z₁ : ℂ := a + 2*Complex.I
  let z₂ : ℂ := 2 + Complex.I
  (∃ b : ℝ, z₁ * z₂ = b * Complex.I) → a = 1 := by
sorry

end NUMINAMATH_CALUDE_product_pure_imaginary_l3534_353409


namespace NUMINAMATH_CALUDE_least_multiple_of_17_greater_than_450_l3534_353484

theorem least_multiple_of_17_greater_than_450 :
  ∀ n : ℕ, n > 0 ∧ 17 ∣ n ∧ n > 450 → n ≥ 459 :=
by sorry

end NUMINAMATH_CALUDE_least_multiple_of_17_greater_than_450_l3534_353484


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_6_l3534_353444

/-- A geometric sequence with its partial sums -/
structure GeometricSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Partial sums
  is_geometric : ∀ n : ℕ, n > 0 → a (n + 1) / a n = a 2 / a 1
  sum_formula : ∀ n : ℕ, n > 0 → S n = (a 1) * (1 - (a 2 / a 1)^n) / (1 - (a 2 / a 1))

/-- The main theorem -/
theorem geometric_sequence_sum_6 (seq : GeometricSequence) 
    (h2 : seq.S 2 = 3) (h4 : seq.S 4 = 15) : seq.S 6 = 63 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_6_l3534_353444


namespace NUMINAMATH_CALUDE_converse_square_sum_zero_contrapositive_subset_intersection_l3534_353436

-- Define the propositions
def P (x y : ℝ) : Prop := x^2 + y^2 = 0
def Q (x y : ℝ) : Prop := x = 0 ∧ y = 0

def R (A B : Set α) : Prop := A ∩ B = A
def S (A B : Set α) : Prop := A ⊆ B

-- Theorem for the converse of statement ①
theorem converse_square_sum_zero :
  ∀ x y : ℝ, Q x y → P x y :=
sorry

-- Theorem for the contrapositive of statement ③
theorem contrapositive_subset_intersection :
  ∀ A B : Set α, ¬(S A B) → ¬(R A B) :=
sorry

end NUMINAMATH_CALUDE_converse_square_sum_zero_contrapositive_subset_intersection_l3534_353436


namespace NUMINAMATH_CALUDE_family_ages_l3534_353440

/-- Given the ages and relationships of family members, prove the ages of the younger siblings after 30 years -/
theorem family_ages (elder_son_age : ℕ) (declan_age_diff : ℕ) (younger_son_age_diff : ℕ) (third_sibling_age_diff : ℕ) (years_later : ℕ)
  (h1 : elder_son_age = 40)
  (h2 : declan_age_diff = 25)
  (h3 : younger_son_age_diff = 10)
  (h4 : third_sibling_age_diff = 5)
  (h5 : years_later = 30) :
  let younger_son_age := elder_son_age - younger_son_age_diff
  let third_sibling_age := younger_son_age - third_sibling_age_diff
  (younger_son_age + years_later = 60) ∧ (third_sibling_age + years_later = 55) :=
by sorry

end NUMINAMATH_CALUDE_family_ages_l3534_353440


namespace NUMINAMATH_CALUDE_expected_red_lights_l3534_353430

-- Define the number of intersections
def num_intersections : ℕ := 3

-- Define the probability of encountering a red light at each intersection
def red_light_prob : ℝ := 0.3

-- State the theorem
theorem expected_red_lights :
  let num_intersections : ℕ := 3
  let red_light_prob : ℝ := 0.3
  (num_intersections : ℝ) * red_light_prob = 0.9 := by
  sorry

end NUMINAMATH_CALUDE_expected_red_lights_l3534_353430


namespace NUMINAMATH_CALUDE_initial_principal_is_8000_l3534_353425

/-- The compound interest formula for annual compounding -/
def compound_interest (P : ℝ) (r : ℝ) (t : ℕ) : ℝ :=
  P * (1 + r) ^ t

/-- Theorem: Given the conditions of the problem, the initial principal is 8000 -/
theorem initial_principal_is_8000 :
  ∃ P : ℝ,
    compound_interest P 0.05 2 = 8820 ∧
    P = 8000 := by
  sorry

end NUMINAMATH_CALUDE_initial_principal_is_8000_l3534_353425


namespace NUMINAMATH_CALUDE_triangle_problem_l3534_353405

theorem triangle_problem (A B C : Real) (a b c : Real) :
  -- Triangle ABC is acute
  0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2 →
  -- a, b, c are sides opposite to angles A, B, C
  a > 0 ∧ b > 0 ∧ c > 0 →
  -- Law of Sines holds
  a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C →
  -- Given condition
  a / Real.sin A = 2 * c / Real.sqrt 3 →
  -- c = √7
  c = Real.sqrt 7 →
  -- Area of triangle ABC is 3√3/2
  1/2 * a * b * Real.sin C = 3 * Real.sqrt 3 / 2 →
  -- Prove:
  C = π/3 ∧ a^2 + b^2 = 13 := by
sorry

end NUMINAMATH_CALUDE_triangle_problem_l3534_353405


namespace NUMINAMATH_CALUDE_last_digit_sum_powers_l3534_353423

theorem last_digit_sum_powers : (1023^3923 + 3081^3921) % 10 = 8 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_sum_powers_l3534_353423


namespace NUMINAMATH_CALUDE_ball_ratio_proof_l3534_353432

/-- Given that Robert initially had 25 balls, Tim initially had 40 balls,
    and Robert ended up with 45 balls after Tim gave him some balls,
    prove that the ratio of the number of balls Tim gave to Robert
    to the number of balls Tim had initially is 1:2. -/
theorem ball_ratio_proof (robert_initial : ℕ) (tim_initial : ℕ) (robert_final : ℕ)
    (h1 : robert_initial = 25)
    (h2 : tim_initial = 40)
    (h3 : robert_final = 45) :
    (robert_final - robert_initial) * 2 = tim_initial := by
  sorry

end NUMINAMATH_CALUDE_ball_ratio_proof_l3534_353432


namespace NUMINAMATH_CALUDE_josh_new_marbles_l3534_353426

/-- The number of marbles Josh lost -/
def marbles_lost : ℕ := 8

/-- The additional marbles Josh found compared to those he lost -/
def additional_marbles : ℕ := 2

/-- The number of new marbles Josh found -/
def new_marbles : ℕ := marbles_lost + additional_marbles

theorem josh_new_marbles : new_marbles = 10 := by sorry

end NUMINAMATH_CALUDE_josh_new_marbles_l3534_353426


namespace NUMINAMATH_CALUDE_triangle_properties_l3534_353489

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the conditions
def triangle_condition (t : Triangle) : Prop :=
  t.a^2 + t.b^2 - t.c^2 = Real.sqrt 3 * t.a * t.b

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h : triangle_condition t) 
  (h_angle : 0 < t.A ∧ t.A ≤ 2 * Real.pi / 3) : 
  t.C = Real.pi / 6 ∧ 
  ∀ m : ℝ, m = 2 * (Real.cos (t.A / 2))^2 - Real.sin t.B - 1 → 
  -1 ≤ m ∧ m < 1/2 := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l3534_353489


namespace NUMINAMATH_CALUDE_line_plane_perpendicularity_l3534_353474

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (plane_parallel : Plane → Plane → Prop)
variable (line_perpendicular : Line → Line → Prop)

-- State the theorem
theorem line_plane_perpendicularity 
  (a b : Line) (α β : Plane) 
  (h1 : a ≠ b) 
  (h2 : α ≠ β) 
  (h3 : parallel a α) 
  (h4 : perpendicular b β) 
  (h5 : plane_parallel α β) : 
  line_perpendicular a b :=
sorry

end NUMINAMATH_CALUDE_line_plane_perpendicularity_l3534_353474


namespace NUMINAMATH_CALUDE_common_chord_equation_l3534_353450

/-- Two circles in a 2D plane -/
structure TwoCircles where
  a : ℝ
  b : ℝ

/-- The equation of a line in 2D -/
structure LineEquation where
  A : ℝ
  B : ℝ
  C : ℝ

/-- Given two circles with a common chord of length 1, 
    prove that the equation of the common chord is 2ax + 2by - 3 = 0 -/
theorem common_chord_equation (circles : TwoCircles) : 
  ∃ (line : LineEquation), 
    line.A = 2 * circles.a ∧ 
    line.B = 2 * circles.b ∧ 
    line.C = -3 := by
  sorry

end NUMINAMATH_CALUDE_common_chord_equation_l3534_353450


namespace NUMINAMATH_CALUDE_ant_meeting_probability_l3534_353414

/-- Represents a cube with 8 vertices -/
structure Cube :=
  (vertices : Fin 8)

/-- Represents an ant on a vertex of the cube -/
structure Ant :=
  (position : Fin 8)

/-- Represents a movement of an ant along an edge -/
def AntMovement := Fin 8 → Fin 8

/-- The total number of possible movement combinations for 8 ants -/
def totalMovements : ℕ := 3^8

/-- The number of non-colliding movement configurations -/
def nonCollidingMovements : ℕ := 24

/-- The probability of ants meeting -/
def probabilityOfMeeting : ℚ := 1 - (nonCollidingMovements : ℚ) / totalMovements

theorem ant_meeting_probability (c : Cube) (ants : Fin 8 → Ant) 
  (movements : Fin 8 → AntMovement) : 
  probabilityOfMeeting = 2381/2387 :=
sorry

end NUMINAMATH_CALUDE_ant_meeting_probability_l3534_353414


namespace NUMINAMATH_CALUDE_candy_fundraiser_profit_l3534_353420

def candy_fundraiser (boxes_total : ℕ) (boxes_discounted : ℕ) (bars_per_box : ℕ) 
  (selling_price : ℚ) (regular_price : ℚ) (discounted_price : ℚ) : ℚ :=
  let boxes_regular := boxes_total - boxes_discounted
  let total_revenue := boxes_total * bars_per_box * selling_price
  let cost_regular := boxes_regular * bars_per_box * regular_price
  let cost_discounted := boxes_discounted * bars_per_box * discounted_price
  let total_cost := cost_regular + cost_discounted
  total_revenue - total_cost

theorem candy_fundraiser_profit :
  candy_fundraiser 5 3 10 (3/2) 1 (4/5) = 31 := by
  sorry

end NUMINAMATH_CALUDE_candy_fundraiser_profit_l3534_353420


namespace NUMINAMATH_CALUDE_expression_value_for_x_3_l3534_353428

theorem expression_value_for_x_3 :
  let x : ℕ := 3
  x + x * (x ^ (x + 1)) = 246 :=
by sorry

end NUMINAMATH_CALUDE_expression_value_for_x_3_l3534_353428


namespace NUMINAMATH_CALUDE_mod_nine_power_difference_l3534_353458

theorem mod_nine_power_difference : 54^2023 - 27^2023 ≡ 0 [ZMOD 9] := by sorry

end NUMINAMATH_CALUDE_mod_nine_power_difference_l3534_353458


namespace NUMINAMATH_CALUDE_right_triangle_arithmetic_progression_l3534_353400

theorem right_triangle_arithmetic_progression :
  ∃ (a d c : ℕ), 
    a > 0 ∧ d > 0 ∧ c > 0 ∧
    a * a + (a + d) * (a + d) = c * c ∧
    c = a + 2 * d ∧
    (a = 120 ∨ a + d = 120 ∨ c = 120) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_arithmetic_progression_l3534_353400


namespace NUMINAMATH_CALUDE_total_laundry_count_l3534_353461

/-- Represents the number of items for each person and shared items -/
structure LaundryItems where
  cally : Nat
  danny : Nat
  emily : Nat
  cally_danny : Nat
  emily_danny : Nat
  cally_emily : Nat

/-- Calculates the total number of laundry items -/
def total_laundry (items : LaundryItems) : Nat :=
  items.cally + items.danny + items.emily + items.cally_danny + items.emily_danny + items.cally_emily

/-- Theorem: The total number of clothes and accessories washed is 141 -/
theorem total_laundry_count :
  ∃ (items : LaundryItems),
    items.cally = 40 ∧
    items.danny = 39 ∧
    items.emily = 39 ∧
    items.cally_danny = 8 ∧
    items.emily_danny = 6 ∧
    items.cally_emily = 9 ∧
    total_laundry items = 141 := by
  sorry

end NUMINAMATH_CALUDE_total_laundry_count_l3534_353461


namespace NUMINAMATH_CALUDE_solve_system_l3534_353499

theorem solve_system (x y : ℝ) (eq1 : 3 * x - 2 * y = 18) (eq2 : x + 2 * y = 10) : y = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l3534_353499


namespace NUMINAMATH_CALUDE_jumping_game_l3534_353412

theorem jumping_game (n : ℕ) 
  (h_odd : Odd n)
  (h_mod3 : n % 3 = 2)
  (h_mod5 : n % 5 = 2) : 
  n = 47 := by
  sorry

end NUMINAMATH_CALUDE_jumping_game_l3534_353412


namespace NUMINAMATH_CALUDE_additional_wax_needed_l3534_353494

theorem additional_wax_needed (total_wax : ℕ) (available_wax : ℕ) (h1 : total_wax = 353) (h2 : available_wax = 331) :
  total_wax - available_wax = 22 := by
  sorry

end NUMINAMATH_CALUDE_additional_wax_needed_l3534_353494


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l3534_353443

theorem polynomial_evaluation : 
  let x : ℝ := 2
  2 * x^2 - 3 * x + 4 = 6 := by sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l3534_353443


namespace NUMINAMATH_CALUDE_average_speed_distance_expression_time_range_l3534_353446

-- Define the boat's movement
structure BoatMovement where
  distance : ℕ → ℝ
  time : ℕ → ℝ

-- Define the given data
def givenData : BoatMovement := {
  distance := λ n => match n with
    | 0 => 200
    | 1 => 150
    | 2 => 100
    | 3 => 50
    | _ => 0
  time := λ n => 2 * n
}

-- Theorem for the average speed
theorem average_speed (b : BoatMovement) : 
  (b.distance 0 - b.distance 3) / (b.time 3 - b.time 0) = 25 := by
  sorry

-- Theorem for the analytical expression
theorem distance_expression (b : BoatMovement) (x : ℝ) : 
  ∃ y : ℝ, y = 200 - 25 * x := by
  sorry

-- Theorem for the range of x
theorem time_range (b : BoatMovement) (x : ℝ) : 
  0 ≤ x ∧ x ≤ 8 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_distance_expression_time_range_l3534_353446


namespace NUMINAMATH_CALUDE_marble_count_l3534_353481

theorem marble_count (g y p : ℕ) : 
  y + p = 7 →  -- all but 7 are green
  g + p = 10 → -- all but 10 are yellow
  g + y = 5 →  -- all but 5 are purple
  g + y + p = 11 := by
sorry

end NUMINAMATH_CALUDE_marble_count_l3534_353481


namespace NUMINAMATH_CALUDE_inequality_proof_l3534_353401

theorem inequality_proof (a b c d x y u v : ℝ) (h : a * b * c * d > 0) :
  (a * x + b * u) * (a * v + b * y) * (c * x + d * v) * (c * u + d * y) ≥ 
  (a * c * u * v * x + b * c * u * x * y + a * d * v * x * y + b * d * u * v * y) * 
  (a * c * x + b * c * u + a * d * v + b * d * y) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3534_353401


namespace NUMINAMATH_CALUDE_gcd_problem_l3534_353447

theorem gcd_problem (a b : ℕ+) (h : Nat.gcd a b = 12) :
  (∃ (x y : ℕ+), Nat.gcd x y = 12 ∧ Nat.gcd (12 * x) (18 * y) = 72) ∧
  (∀ (c d : ℕ+), Nat.gcd c d = 12 → Nat.gcd (12 * c) (18 * d) ≥ 72) :=
sorry

end NUMINAMATH_CALUDE_gcd_problem_l3534_353447


namespace NUMINAMATH_CALUDE_jerry_shelf_capacity_l3534_353413

/-- Given the total number of books, the number of books taken by the librarian,
    and the number of shelves needed, calculate the number of books that can fit on each shelf. -/
def books_per_shelf (total_books : ℕ) (books_taken : ℕ) (shelves_needed : ℕ) : ℕ :=
  (total_books - books_taken) / shelves_needed

/-- Prove that Jerry can fit 3 books on each shelf. -/
theorem jerry_shelf_capacity : books_per_shelf 34 7 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_jerry_shelf_capacity_l3534_353413


namespace NUMINAMATH_CALUDE_algebraic_expression_simplification_expression_value_at_negative_quarter_l3534_353487

theorem algebraic_expression_simplification (a : ℝ) :
  (a - 2)^2 + (a + 1) * (a - 1) - 2 * a * (a - 3) = 2 * a + 3 :=
by sorry

theorem expression_value_at_negative_quarter :
  let a : ℝ := -1/4
  (a - 2)^2 + (a + 1) * (a - 1) - 2 * a * (a - 3) = 5/2 :=
by sorry

end NUMINAMATH_CALUDE_algebraic_expression_simplification_expression_value_at_negative_quarter_l3534_353487


namespace NUMINAMATH_CALUDE_power_of_ten_thousand_zeros_after_one_l3534_353469

theorem power_of_ten_thousand (n : ℕ) : (10000 : ℕ) ^ n = (10 : ℕ) ^ (4 * n) := by sorry

theorem zeros_after_one : (10000 : ℕ) ^ 50 = (10 : ℕ) ^ 200 := by sorry

end NUMINAMATH_CALUDE_power_of_ten_thousand_zeros_after_one_l3534_353469


namespace NUMINAMATH_CALUDE_well_depth_rope_length_l3534_353472

/-- 
Given a well of unknown depth and a rope of unknown length, prove that if:
1) Folding the rope three times and lowering it into the well leaves 4 feet outside
2) Folding the rope four times and lowering it into the well leaves 1 foot outside
Then the depth of the well (x) and the length of the rope (h) satisfy the system of equations:
{h/3 = x + 4, h/4 = x + 1}
-/
theorem well_depth_rope_length (x h : ℝ) 
  (h_positive : h > 0) 
  (fold_three : h / 3 = x + 4) 
  (fold_four : h / 4 = x + 1) : 
  h / 3 = x + 4 ∧ h / 4 = x + 1 := by
sorry


end NUMINAMATH_CALUDE_well_depth_rope_length_l3534_353472


namespace NUMINAMATH_CALUDE_max_sum_with_constraints_l3534_353439

theorem max_sum_with_constraints (x y : ℝ) 
  (h1 : 3 * x + 2 * y ≤ 7) 
  (h2 : 2 * x + 4 * y ≤ 8) : 
  x + y ≤ 11 / 4 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_with_constraints_l3534_353439


namespace NUMINAMATH_CALUDE_elevator_is_translation_l3534_353476

/-- A structure representing a movement in space -/
structure Movement where
  is_straight_line : Bool

/-- Definition of translation in mathematics -/
def is_translation (m : Movement) : Prop :=
  m.is_straight_line = true

/-- Representation of an elevator's movement -/
def elevator_movement : Movement where
  is_straight_line := true

/-- Theorem stating that an elevator's movement is a translation -/
theorem elevator_is_translation : is_translation elevator_movement := by
  sorry

end NUMINAMATH_CALUDE_elevator_is_translation_l3534_353476


namespace NUMINAMATH_CALUDE_graces_initial_fruits_l3534_353464

/-- The number of Graces --/
def num_graces : ℕ := 3

/-- The number of Muses --/
def num_muses : ℕ := 9

/-- Represents the distribution of fruits --/
structure FruitDistribution where
  initial_grace : ℕ  -- Initial number of fruits each Grace had
  given_to_muse : ℕ  -- Number of fruits each Grace gave to each Muse

/-- Theorem stating the conditions and the result to be proved --/
theorem graces_initial_fruits (fd : FruitDistribution) : 
  -- Each Grace gives fruits to each Muse
  (fd.initial_grace ≥ num_muses * fd.given_to_muse) →
  -- After exchange, Graces and Muses have the same number of fruits
  (fd.initial_grace - num_muses * fd.given_to_muse = num_graces * fd.given_to_muse) →
  -- Initial number of fruits each Grace had is 12
  fd.initial_grace = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_graces_initial_fruits_l3534_353464


namespace NUMINAMATH_CALUDE_fourth_student_is_18_l3534_353411

/-- Represents a systematic sampling of students -/
structure SystematicSample where
  total_students : ℕ
  sample_size : ℕ
  first_student : ℕ
  h_total_positive : 0 < total_students
  h_sample_positive : 0 < sample_size
  h_sample_size : sample_size ≤ total_students
  h_first_valid : first_student ≤ total_students

/-- The sampling interval for a systematic sample -/
def sampling_interval (s : SystematicSample) : ℕ :=
  s.total_students / s.sample_size

/-- The nth student in the sample -/
def nth_student (s : SystematicSample) (n : ℕ) : ℕ :=
  s.first_student + (n - 1) * sampling_interval s

/-- Theorem: In a systematic sample of 4 from 52, if 5, 31, and 44 are sampled, then 18 is the fourth -/
theorem fourth_student_is_18 (s : SystematicSample) 
    (h_total : s.total_students = 52)
    (h_sample : s.sample_size = 4)
    (h_first : s.first_student = 5)
    (h_third : nth_student s 3 = 31)
    (h_fourth : nth_student s 4 = 44) :
    nth_student s 2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_fourth_student_is_18_l3534_353411


namespace NUMINAMATH_CALUDE_f_minimum_value_l3534_353470

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| + |x - 2|

-- State the theorem
theorem f_minimum_value :
  (∀ x : ℝ, f x ≥ 3) ∧ (∃ x : ℝ, f x = 3) := by
  sorry

end NUMINAMATH_CALUDE_f_minimum_value_l3534_353470


namespace NUMINAMATH_CALUDE_problem_solution_l3534_353495

noncomputable section

variable (g : ℝ → ℝ)

-- g is invertible
variable (h : Function.Bijective g)

-- Define the values of g given in the table
axiom g_1 : g 1 = 4
axiom g_2 : g 2 = 6
axiom g_3 : g 3 = 9
axiom g_4 : g 4 = 10
axiom g_5 : g 5 = 12

-- The theorem to prove
theorem problem_solution :
  g (g 2) + g (Function.invFun g 12) + Function.invFun g (Function.invFun g 10) = 25 := by
  sorry

end

end NUMINAMATH_CALUDE_problem_solution_l3534_353495


namespace NUMINAMATH_CALUDE_set_intersection_complement_l3534_353437

def U : Set Int := Set.univ
def M : Set Int := {1, 2}
def P : Set Int := {-2, -1, 0, 1, 2}

theorem set_intersection_complement : P ∩ (U \ M) = {-2, -1, 0} := by
  sorry

end NUMINAMATH_CALUDE_set_intersection_complement_l3534_353437


namespace NUMINAMATH_CALUDE_cube_fourth_root_inverse_prop_l3534_353453

-- Define the inverse proportionality between a^3 and b^(1/4)
def inverse_prop (a b : ℝ) : Prop := ∃ k : ℝ, a^3 * b^(1/4) = k

-- Define the initial condition
def initial_condition (a b : ℝ) : Prop := a = 3 ∧ b = 16

-- Define the final condition
def final_condition (a b : ℝ) : Prop := a^2 * b = 54

theorem cube_fourth_root_inverse_prop 
  (a b : ℝ) 
  (h_inv_prop : inverse_prop a b) 
  (h_init : initial_condition a b) 
  (h_final : final_condition a b) : 
  b = 54^(2/5) := by
  sorry

end NUMINAMATH_CALUDE_cube_fourth_root_inverse_prop_l3534_353453


namespace NUMINAMATH_CALUDE_beth_crayons_left_l3534_353429

/-- The number of crayons Beth has left after giving some away -/
def crayons_left (initial : ℕ) (given_away : ℕ) : ℕ :=
  initial - given_away

/-- Proof that Beth has 52 crayons left -/
theorem beth_crayons_left :
  let initial_crayons : ℕ := 106
  let crayons_given_away : ℕ := 54
  crayons_left initial_crayons crayons_given_away = 52 := by
sorry

end NUMINAMATH_CALUDE_beth_crayons_left_l3534_353429


namespace NUMINAMATH_CALUDE_quadratic_solution_difference_l3534_353438

theorem quadratic_solution_difference : 
  let f : ℝ → ℝ := λ x => x^2 - 5*x + 7 - (x + 35)
  let s₁ := (6 + Real.sqrt 148) / 2
  let s₂ := (6 - Real.sqrt 148) / 2
  f s₁ = 0 ∧ f s₂ = 0 ∧ s₁ - s₂ = 2 * Real.sqrt 37 := by sorry

end NUMINAMATH_CALUDE_quadratic_solution_difference_l3534_353438


namespace NUMINAMATH_CALUDE_distance_range_m_l3534_353404

-- Define the distance function
def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := |x₁ - x₂| + 2 * |y₁ - y₂|

-- Define the theorem
theorem distance_range_m :
  ∀ m : ℝ,
  (distance 2 1 (-1) m ≤ 5) ↔ (0 ≤ m ∧ m ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_distance_range_m_l3534_353404


namespace NUMINAMATH_CALUDE_student_age_fraction_l3534_353442

theorem student_age_fraction (total_students : ℕ) (below_8_percent : ℚ) (age_8_students : ℕ) : 
  total_students = 50 →
  below_8_percent = 1/5 →
  age_8_students = 24 →
  (total_students - (total_students * below_8_percent).num - age_8_students : ℚ) / age_8_students = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_student_age_fraction_l3534_353442


namespace NUMINAMATH_CALUDE_first_last_gender_l3534_353475

/-- Represents the gender of a person in line -/
inductive Gender
  | Man
  | Woman

/-- Represents the state of bottle passing -/
structure BottlePassing where
  total_people : Nat
  woman_to_woman : Nat
  woman_to_man : Nat
  man_to_man : Nat

/-- Theorem stating the first and last person's gender based on bottle passing information -/
theorem first_last_gender (bp : BottlePassing) 
  (h1 : bp.total_people = 16)
  (h2 : bp.woman_to_woman = 4)
  (h3 : bp.woman_to_man = 3)
  (h4 : bp.man_to_man = 6) :
  (Gender.Woman, Gender.Man) = 
    (match bp.total_people with
      | 0 => (Gender.Woman, Gender.Man)  -- Arbitrary choice for empty line
      | n + 1 => 
        let first := if bp.woman_to_woman + bp.woman_to_man > bp.man_to_man + (n - (bp.woman_to_woman + bp.woman_to_man + bp.man_to_man)) 
                     then Gender.Woman else Gender.Man
        let last := if bp.man_to_man + (n - (bp.woman_to_woman + bp.woman_to_man + bp.man_to_man)) > bp.woman_to_woman + bp.woman_to_man 
                    then Gender.Man else Gender.Woman
        (first, last)
    ) :=
by
  sorry


end NUMINAMATH_CALUDE_first_last_gender_l3534_353475


namespace NUMINAMATH_CALUDE_cubic_root_sum_product_l3534_353419

theorem cubic_root_sum_product (p q r : ℝ) : 
  (6 * p^3 - 4 * p^2 + 15 * p - 10 = 0) ∧ 
  (6 * q^3 - 4 * q^2 + 15 * q - 10 = 0) ∧ 
  (6 * r^3 - 4 * r^2 + 15 * r - 10 = 0) →
  p * q + q * r + r * p = 5/2 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_product_l3534_353419


namespace NUMINAMATH_CALUDE_no_simultaneous_squares_l3534_353448

theorem no_simultaneous_squares : ¬∃ (x y : ℕ), ∃ (a b : ℕ), x^2 + y = a^2 ∧ x + y^2 = b^2 := by
  sorry

end NUMINAMATH_CALUDE_no_simultaneous_squares_l3534_353448


namespace NUMINAMATH_CALUDE_calculate_F_of_5_f_6_l3534_353482

-- Define the functions f and F
def f (a : ℝ) : ℝ := a + 3
def F (a b : ℝ) : ℝ := b^3 - 2*a

-- State the theorem
theorem calculate_F_of_5_f_6 : F 5 (f 6) = 719 := by
  sorry

end NUMINAMATH_CALUDE_calculate_F_of_5_f_6_l3534_353482


namespace NUMINAMATH_CALUDE_ordering_abc_l3534_353467

theorem ordering_abc (a b c : ℝ) : 
  a = 6 - Real.log 2 - Real.log 3 →
  b = Real.exp 1 - Real.log 3 →
  c = Real.exp 2 - 2 →
  c > a ∧ a > b := by sorry

end NUMINAMATH_CALUDE_ordering_abc_l3534_353467


namespace NUMINAMATH_CALUDE_ripe_oranges_count_l3534_353498

/-- The number of sacks of unripe oranges harvested per day -/
def unripe_oranges : ℕ := 25

/-- The difference between the number of sacks of ripe and unripe oranges -/
def difference : ℕ := 19

/-- The number of sacks of ripe oranges harvested per day -/
def ripe_oranges : ℕ := unripe_oranges + difference

theorem ripe_oranges_count : ripe_oranges = 44 := by
  sorry

end NUMINAMATH_CALUDE_ripe_oranges_count_l3534_353498


namespace NUMINAMATH_CALUDE_inverse_proportion_y_comparison_l3534_353407

/-- Given two points on the inverse proportion function y = -5/x,
    where the x-coordinate of the first point is positive and
    the x-coordinate of the second point is negative,
    prove that the y-coordinate of the first point is less than
    the y-coordinate of the second point. -/
theorem inverse_proportion_y_comparison
  (x₁ x₂ y₁ y₂ : ℝ)
  (h1 : y₁ = -5 / x₁)
  (h2 : y₂ = -5 / x₂)
  (h3 : x₁ > 0)
  (h4 : x₂ < 0) :
  y₁ < y₂ :=
sorry

end NUMINAMATH_CALUDE_inverse_proportion_y_comparison_l3534_353407


namespace NUMINAMATH_CALUDE_quarters_per_machine_l3534_353435

/-- Represents the number of machines in the launderette -/
def num_machines : ℕ := 3

/-- Represents the number of dimes in each machine -/
def dimes_per_machine : ℕ := 100

/-- Represents the total amount of money from all machines in cents -/
def total_money : ℕ := 9000  -- $90 in cents

/-- Represents the value of a dime in cents -/
def dime_value : ℕ := 10

/-- Represents the value of a quarter in cents -/
def quarter_value : ℕ := 25

theorem quarters_per_machine :
  ∃ (q : ℕ), 
    q * quarter_value * num_machines + 
    dimes_per_machine * dime_value * num_machines = 
    total_money ∧ 
    q = 80 := by
  sorry

end NUMINAMATH_CALUDE_quarters_per_machine_l3534_353435


namespace NUMINAMATH_CALUDE_total_beneficial_insects_l3534_353465

theorem total_beneficial_insects (ladybugs_with_spots : Nat) (ladybugs_without_spots : Nat) (green_lacewings : Nat) (trichogramma_wasps : Nat)
  (h1 : ladybugs_with_spots = 12170)
  (h2 : ladybugs_without_spots = 54912)
  (h3 : green_lacewings = 67923)
  (h4 : trichogramma_wasps = 45872) :
  ladybugs_with_spots + ladybugs_without_spots + green_lacewings + trichogramma_wasps = 180877 := by
  sorry

end NUMINAMATH_CALUDE_total_beneficial_insects_l3534_353465
