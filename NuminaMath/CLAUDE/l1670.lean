import Mathlib

namespace range_of_3a_plus_2b_l1670_167014

theorem range_of_3a_plus_2b (a b : ℝ) (h : a^2 + b^2 = 4) :
  ∃ (x : ℝ), x ∈ Set.Icc (-2 * Real.sqrt 13) (2 * Real.sqrt 13) ∧ x = 3*a + 2*b :=
by sorry

end range_of_3a_plus_2b_l1670_167014


namespace probability_integer_exponent_x_l1670_167044

theorem probability_integer_exponent_x (a : ℝ) (x : ℝ) :
  let expansion := (x - a / Real.sqrt x) ^ 5
  let total_terms := 6
  let integer_exponent_terms := 3
  (integer_exponent_terms : ℚ) / total_terms = 1 / 2 := by
sorry

end probability_integer_exponent_x_l1670_167044


namespace triangle_shape_l1670_167000

/-- A triangle with side lengths a, b, and c is either isosceles or right-angled if a^4 - b^4 + (b^2c^2 - a^2c^2) = 0 -/
theorem triangle_shape (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (eq : a^4 - b^4 + (b^2 * c^2 - a^2 * c^2) = 0) : 
  (a = b) ∨ (a^2 + b^2 = c^2) := by
  sorry

end triangle_shape_l1670_167000


namespace hannahs_speed_l1670_167005

/-- Proves that Hannah's speed is 15 km/h given the problem conditions --/
theorem hannahs_speed (glen_speed : ℝ) (distance : ℝ) (time : ℝ) 
  (h_glen_speed : glen_speed = 37)
  (h_distance : distance = 130)
  (h_time : time = 5) :
  ∃ hannah_speed : ℝ, hannah_speed = 15 ∧ 
  2 * distance = (glen_speed + hannah_speed) * time :=
by sorry

end hannahs_speed_l1670_167005


namespace equilateral_triangle_reflection_theorem_l1670_167009

/-- Represents a ray path in an equilateral triangle -/
structure RayPath where
  n : ℕ  -- number of reflections
  returns_to_start : Bool  -- whether the ray returns to the starting point
  passes_through_vertices : Bool  -- whether the ray passes through other vertices

/-- Checks if a number is a valid reflection count -/
def is_valid_reflection_count (n : ℕ) : Prop :=
  (n % 6 = 1 ∨ n % 6 = 5) ∧ n ≠ 5 ∧ n ≠ 17

/-- Main theorem: Characterizes valid reflection counts in an equilateral triangle -/
theorem equilateral_triangle_reflection_theorem :
  ∀ (path : RayPath),
    path.returns_to_start ∧ ¬path.passes_through_vertices ↔
    is_valid_reflection_count path.n :=
sorry

end equilateral_triangle_reflection_theorem_l1670_167009


namespace cleanup_drive_total_l1670_167020

/-- The total amount of garbage collected by three groups in a cleanup drive -/
theorem cleanup_drive_total (group1_pounds group2_pounds group3_ounces : ℕ) 
  (h1 : group1_pounds = 387)
  (h2 : group2_pounds = group1_pounds - 39)
  (h3 : group3_ounces = 560)
  (h4 : ∀ (x : ℕ), x * 16 = x * 1 * 16) :
  group1_pounds + group2_pounds + (group3_ounces / 16) = 770 := by
  sorry

end cleanup_drive_total_l1670_167020


namespace quadratic_inequality_solution_range_l1670_167067

theorem quadratic_inequality_solution_range (c : ℝ) :
  (c > 0) →
  (∃ x : ℝ, x^2 - 6*x + c < 0) ↔ (c > 0 ∧ c < 9) :=
sorry

end quadratic_inequality_solution_range_l1670_167067


namespace probability_divisible_by_15_l1670_167054

/-- The set of integers between 1 and 2^30 with exactly two 1s in their binary expansions -/
def T : Set ℕ := {n | 1 ≤ n ∧ n ≤ 2^30 ∧ (∃ j k : ℕ, j < k ∧ k < 30 ∧ n = 2^j + 2^k)}

/-- The number of elements in T -/
def T_count : ℕ := 435

/-- The number of elements in T divisible by 15 -/
def T_div15_count : ℕ := 28

theorem probability_divisible_by_15 :
  (T_div15_count : ℚ) / T_count = 28 / 435 ∧ 28 + 435 = 463 := by sorry

end probability_divisible_by_15_l1670_167054


namespace sum_of_roots_quadratic_sum_of_roots_specific_equation_l1670_167003

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c
  (∃ x y : ℝ, f x = 0 ∧ f y = 0 ∧ x ≠ y) →
  (∃ s : ℝ, s = -(b / a) ∧ s = x + y) :=
by sorry

theorem sum_of_roots_specific_equation :
  let f : ℝ → ℝ := λ x ↦ x^2 + 2023 * x - 2024
  (∃ x y : ℝ, f x = 0 ∧ f y = 0 ∧ x ≠ y) →
  (∃ s : ℝ, s = -2023 ∧ s = x + y) :=
by sorry

end sum_of_roots_quadratic_sum_of_roots_specific_equation_l1670_167003


namespace concert_attendance_l1670_167095

theorem concert_attendance (adults : ℕ) (children : ℕ) : 
  children = 3 * adults →
  7 * adults + 3 * children = 6000 →
  adults + children = 1500 := by
sorry

end concert_attendance_l1670_167095


namespace exam_probability_theorem_l1670_167078

def exam_probability (P : ℝ) : Prop :=
  let correct_prob : List ℝ := [1, P, 1/2, 1/4]
  let perfect_score_prob := List.prod correct_prob
  let prob_15 := P * (1/2) * (3/4) + (1-P) * (1/2) * (3/4)
  let prob_10 := P * (1/2) * (3/4) * (1/4) + (1-P) * (1/2) * (1/4) + (1-P) * (1/2) * (3/4) * (1/4)
  let prob_5 := P * (1/2) * (3/4) + (1-P) * (1/2) * (1/4)
  let prob_0 := 1 - perfect_score_prob - prob_15 - prob_10 - prob_5
  (P = 2/3 → perfect_score_prob = 1/12) ∧
  (prob_15 = 1/8 ∧ prob_10 = 1/8) ∧
  (prob_0 = 1/6)

theorem exam_probability_theorem : exam_probability (2/3) :=
sorry

end exam_probability_theorem_l1670_167078


namespace sticker_difference_l1670_167011

def total_stickers : ℕ := 58
def first_box_stickers : ℕ := 23

theorem sticker_difference : 
  total_stickers - first_box_stickers - first_box_stickers = 12 := by
  sorry

end sticker_difference_l1670_167011


namespace g_evaluation_l1670_167086

def g (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 10

theorem g_evaluation : 3 * g 2 + 2 * g (-2) = 98 := by
  sorry

end g_evaluation_l1670_167086


namespace modified_cube_edge_count_l1670_167025

/-- Represents a cube with smaller cubes removed from its corners -/
structure ModifiedCube where
  originalSideLength : ℕ
  removedCubeSideLength : ℕ

/-- Calculates the number of edges in the modified cube -/
def edgeCount (c : ModifiedCube) : ℕ :=
  -- The actual calculation would go here
  24

/-- Theorem stating that a cube of side length 4 with unit cubes removed from corners has 24 edges -/
theorem modified_cube_edge_count :
  ∀ (c : ModifiedCube), 
    c.originalSideLength = 4 ∧ 
    c.removedCubeSideLength = 1 → 
    edgeCount c = 24 := by
  sorry

end modified_cube_edge_count_l1670_167025


namespace actual_distance_towns_distance_proof_l1670_167077

/-- Calculates the actual distance between two towns given the map distance and scale. -/
theorem actual_distance (map_distance : ℝ) (scale_distance : ℝ) (scale_miles : ℝ) : ℝ :=
  let miles_per_inch := scale_miles / scale_distance
  map_distance * miles_per_inch

/-- Proves that the actual distance between two towns is 400 miles given the specified conditions. -/
theorem towns_distance_proof :
  actual_distance 20 0.5 10 = 400 := by
  sorry

end actual_distance_towns_distance_proof_l1670_167077


namespace inequality_and_equality_condition_l1670_167055

theorem inequality_and_equality_condition (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (x + 1/x)^2 + (y + 1/y)^2 ≥ 25/2 ∧
  ((x + 1/x)^2 + (y + 1/y)^2 = 25/2 ↔ x = 1/2 ∧ y = 1/2) :=
by sorry

end inequality_and_equality_condition_l1670_167055


namespace quadratic_minimum_l1670_167073

theorem quadratic_minimum (x : ℝ) : x^2 + 10*x ≥ -25 ∧ ∃ y : ℝ, y^2 + 10*y = -25 := by
  sorry

end quadratic_minimum_l1670_167073


namespace point_distance_3d_l1670_167018

/-- Given two points A(m, 2, 3) and B(1, -1, 1) in 3D space with distance √13 between them, m = 1 -/
theorem point_distance_3d (m : ℝ) : 
  let A : ℝ × ℝ × ℝ := (m, 2, 3)
  let B : ℝ × ℝ × ℝ := (1, -1, 1)
  (m - 1)^2 + 3^2 + 2^2 = 13 → m = 1 := by
  sorry

end point_distance_3d_l1670_167018


namespace cucumber_water_percentage_l1670_167082

theorem cucumber_water_percentage (initial_weight initial_water_percentage new_weight : ℝ) :
  initial_weight = 100 →
  initial_water_percentage = 99 →
  new_weight = 50 →
  let initial_water := initial_weight * (initial_water_percentage / 100)
  let initial_solid := initial_weight - initial_water
  let new_water := new_weight - initial_solid
  new_water / new_weight * 100 = 98 := by
sorry

end cucumber_water_percentage_l1670_167082


namespace percentage_equality_l1670_167026

theorem percentage_equality (x y : ℝ) (h1 : 1.5 * x = 0.75 * y) (h2 : x = 20) : y = 40 := by
  sorry

end percentage_equality_l1670_167026


namespace different_tens_digit_probability_l1670_167097

theorem different_tens_digit_probability : 
  let total_integers : ℕ := 70
  let chosen_integers : ℕ := 7
  let tens_digits : ℕ := 7
  let integers_per_tens : ℕ := 10

  let favorable_outcomes : ℕ := integers_per_tens ^ chosen_integers
  let total_outcomes : ℕ := Nat.choose total_integers chosen_integers

  (favorable_outcomes : ℚ) / total_outcomes = 20000 / 83342961 := by
  sorry

end different_tens_digit_probability_l1670_167097


namespace parallel_to_line_if_equal_perpendicular_distances_l1670_167075

structure Geometry2D where
  Point : Type
  Line : Type
  perpendicular_distance : Point → Line → ℝ
  on_line : Point → Line → Prop
  parallel : Line → Line → Prop

variable {G : Geometry2D}

theorem parallel_to_line_if_equal_perpendicular_distances
  (A B : G.Point) (l : G.Line) :
  G.perpendicular_distance A l = G.perpendicular_distance B l →
  ∃ (AB : G.Line), G.on_line A AB ∧ G.on_line B AB ∧ G.parallel AB l :=
sorry

end parallel_to_line_if_equal_perpendicular_distances_l1670_167075


namespace f_two_thirds_eq_three_halves_l1670_167048

noncomputable def g (x : ℝ) : ℝ := 2 - 3 * x^2

noncomputable def f (y : ℝ) : ℝ := y / ((2 - y) / 3)

theorem f_two_thirds_eq_three_halves :
  f (2/3) = 3/2 :=
by sorry

end f_two_thirds_eq_three_halves_l1670_167048


namespace quadratic_inequality_solution_l1670_167066

theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x : ℝ, ax^2 + b*x + 2 > 0 ↔ -1/2 < x ∧ x < 1/3) →
  a + b = -14 :=
sorry

end quadratic_inequality_solution_l1670_167066


namespace marbles_difference_l1670_167068

/-- Given information about Josh's marble collection -/
structure MarbleCollection where
  initial : ℕ
  found : ℕ
  lost : ℕ

/-- Theorem stating the difference between lost and found marbles -/
theorem marbles_difference (josh : MarbleCollection)
  (h1 : josh.initial = 15)
  (h2 : josh.found = 9)
  (h3 : josh.lost = 23) :
  josh.lost - josh.found = 14 := by
  sorry

end marbles_difference_l1670_167068


namespace largest_possible_BD_l1670_167002

/-- A cyclic quadrilateral with side lengths that are distinct primes less than 20 -/
structure CyclicQuadrilateral where
  AB : ℕ
  BC : ℕ
  CD : ℕ
  DA : ℕ
  cyclic : Bool
  distinct_primes : AB ≠ BC ∧ AB ≠ CD ∧ AB ≠ DA ∧ BC ≠ CD ∧ BC ≠ DA ∧ CD ≠ DA
  all_prime : Nat.Prime AB ∧ Nat.Prime BC ∧ Nat.Prime CD ∧ Nat.Prime DA
  all_less_than_20 : AB < 20 ∧ BC < 20 ∧ CD < 20 ∧ DA < 20
  AB_is_11 : AB = 11
  product_condition : BC * CD = AB * DA

/-- The diagonal BD of the cyclic quadrilateral -/
def diagonal_BD (q : CyclicQuadrilateral) : ℝ := sorry

theorem largest_possible_BD (q : CyclicQuadrilateral) :
  ∃ (max_bd : ℝ), diagonal_BD q ≤ max_bd ∧ max_bd = Real.sqrt 290 := by
  sorry

end largest_possible_BD_l1670_167002


namespace candy_count_l1670_167090

/-- The number of candies initially in the pile -/
def initial_candies : ℕ := 6

/-- The number of candies added to the pile -/
def added_candies : ℕ := 4

/-- The total number of candies after adding -/
def total_candies : ℕ := initial_candies + added_candies

theorem candy_count : total_candies = 10 := by
  sorry

end candy_count_l1670_167090


namespace collinear_points_sum_l1670_167027

/-- Given three collinear points in 3D space, prove that the sum of x and y coordinates of two points is -1/2 --/
theorem collinear_points_sum (x y : ℝ) : 
  let A : ℝ × ℝ × ℝ := (1, 2, 0)
  let B : ℝ × ℝ × ℝ := (x, 3, -1)
  let C : ℝ × ℝ × ℝ := (4, y, 2)
  (∃ (t : ℝ), B - A = t • (C - A)) → x + y = -1/2 := by
sorry


end collinear_points_sum_l1670_167027


namespace complement_of_40_30_l1670_167088

/-- The complement of an angle is the difference between 90 degrees and the angle. -/
def complementOfAngle (angle : ℚ) : ℚ := 90 - angle

/-- Represents 40 degrees and 30 minutes in decimal degrees. -/
def angleA : ℚ := 40 + 30 / 60

theorem complement_of_40_30 :
  complementOfAngle angleA = 49 + 30 / 60 := by sorry

end complement_of_40_30_l1670_167088


namespace system_solution_implies_2a_minus_3b_equals_6_l1670_167059

theorem system_solution_implies_2a_minus_3b_equals_6
  (a b : ℝ)
  (eq1 : a * 2 - b * 1 = 4)
  (eq2 : a * 2 + b * 1 = 2) :
  2 * a - 3 * b = 6 := by
  sorry

end system_solution_implies_2a_minus_3b_equals_6_l1670_167059


namespace log_equation_solution_l1670_167033

-- Define the logarithm function
noncomputable def log_base (b : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log b

-- Define the property of being a non-square
def is_non_square (x : ℚ) : Prop := ∀ n : ℕ, x ≠ (n : ℚ)^2

-- Define the property of being a non-cube
def is_non_cube (x : ℚ) : Prop := ∀ n : ℕ, x ≠ (n : ℚ)^3

-- Define the property of being non-integral
def is_non_integral (x : ℚ) : Prop := ∀ n : ℤ, x ≠ n

-- Main theorem
theorem log_equation_solution :
  ∃ x : ℝ, 
    log_base (3 * x) 343 = x ∧ 
    x = 4 / 3 ∧
    (∃ q : ℚ, x = q) ∧
    is_non_square (4 / 3) ∧
    is_non_cube (4 / 3) ∧
    is_non_integral (4 / 3) :=
by sorry

end log_equation_solution_l1670_167033


namespace interior_angle_measure_l1670_167060

/-- Given a triangle with an interior angle, if the measures of the three triangle angles are known,
    then the measure of the interior angle can be determined. -/
theorem interior_angle_measure (m1 m2 m3 m4 : ℝ) : 
  m1 = 62 → m2 = 36 → m3 = 24 → 
  m1 + m2 + m3 + m4 < 360 →
  m4 = 122 := by
  sorry

#check interior_angle_measure

end interior_angle_measure_l1670_167060


namespace sqrt_x_minus_8_meaningful_l1670_167021

theorem sqrt_x_minus_8_meaningful (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 8) ↔ x ≥ 8 := by
  sorry

end sqrt_x_minus_8_meaningful_l1670_167021


namespace woodworker_chairs_l1670_167017

/-- Calculates the number of chairs built given the total number of furniture legs,
    number of tables, legs per table, and legs per chair. -/
def chairs_built (total_legs : ℕ) (num_tables : ℕ) (legs_per_table : ℕ) (legs_per_chair : ℕ) : ℕ :=
  (total_legs - num_tables * legs_per_table) / legs_per_chair

/-- Proves that given 40 total furniture legs, 4 tables, 4 legs per table,
    and 4 legs per chair, the number of chairs built is 6. -/
theorem woodworker_chairs : chairs_built 40 4 4 4 = 6 := by
  sorry

end woodworker_chairs_l1670_167017


namespace problem_solution_l1670_167052

theorem problem_solution (x y : ℝ) (h1 : x + y = 3) (h2 : x * y = 1) :
  x + (x^4 / y^3) + (y^4 / x^3) + y = 849 := by
  sorry

end problem_solution_l1670_167052


namespace water_collection_impossible_l1670_167035

def total_water (n : ℕ) : ℕ := n * (n + 1) / 2

def is_odd (n : ℕ) : Prop := ∃ k, n = 2 * k + 1

theorem water_collection_impossible (n : ℕ) (h : n = 2018) :
  is_odd (total_water n) ∧ 
  (∀ m : ℕ, m ≤ n → ∃ k, m = 2 ^ k) → False :=
by sorry

end water_collection_impossible_l1670_167035


namespace ball_problem_l1670_167098

/-- The number of red balls is one more than the number of yellow balls -/
def num_red (a : ℕ) : ℕ := a + 1

/-- The number of yellow balls -/
def num_yellow (a : ℕ) : ℕ := a

/-- The number of blue balls is always 1 -/
def num_blue : ℕ := 1

/-- The total number of balls in the bag -/
def total_balls (a : ℕ) : ℕ := num_red a + num_yellow a + num_blue

/-- The score for drawing a red ball -/
def score_red : ℕ := 1

/-- The score for drawing a yellow ball -/
def score_yellow : ℕ := 2

/-- The score for drawing a blue ball -/
def score_blue : ℕ := 3

/-- The expected value of drawing a ball -/
def expected_value (a : ℕ) : ℚ :=
  (score_red * num_red a + score_yellow * num_yellow a + score_blue * num_blue) / total_balls a

/-- The theorem to be proved -/
theorem ball_problem (a : ℕ) (h1 : a > 0) (h2 : expected_value a = 5/3) :
  a = 2 ∧ (3 : ℚ)/10 = (Nat.choose (num_red 2) 1 * Nat.choose (num_yellow 2) 2 + 
                         Nat.choose (num_red 2) 2 * Nat.choose num_blue 1) / Nat.choose (total_balls 2) 3 :=
by sorry

end ball_problem_l1670_167098


namespace bread_slices_per_loaf_l1670_167079

theorem bread_slices_per_loaf :
  ∀ (num_loaves : ℕ) (payment : ℕ) (change : ℕ) (slice_cost : ℚ),
    num_loaves = 3 →
    payment = 40 →
    change = 16 →
    slice_cost = 2/5 →
    (((payment - change : ℚ) / slice_cost) / num_loaves : ℚ) = 20 := by
  sorry

end bread_slices_per_loaf_l1670_167079


namespace city_H_highest_increase_l1670_167024

structure City where
  name : String
  pop1990 : ℕ
  pop2000 : ℕ
  event_factor : ℚ

def effective_increase (c : City) : ℚ :=
  (c.pop2000 * c.event_factor - c.pop1990) / c.pop1990

def cities : List City := [
  ⟨"F", 90000, 120000, 11/10⟩,
  ⟨"G", 80000, 110000, 19/20⟩,
  ⟨"H", 70000, 115000, 11/10⟩,
  ⟨"I", 65000, 100000, 49/50⟩,
  ⟨"J", 95000, 145000, 1⟩
]

theorem city_H_highest_increase :
  ∃ c ∈ cities, c.name = "H" ∧
    ∀ c' ∈ cities, effective_increase c ≥ effective_increase c' := by
  sorry

end city_H_highest_increase_l1670_167024


namespace inequality_proof_l1670_167015

theorem inequality_proof (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) :
  (a^2 * b + b^2 * c + c^2 * a) * (a * b^2 + b * c^2 + c * a^2) ≥ 9 * (a * b * c)^2 :=
by sorry

end inequality_proof_l1670_167015


namespace vector_simplification_l1670_167019

variable {V : Type*} [AddCommGroup V]

theorem vector_simplification (P M N : V) : 
  (P - M) - (P - N) + (M - N) = (0 : V) := by sorry

end vector_simplification_l1670_167019


namespace minimum_distance_point_to_curve_l1670_167008

open Real

theorem minimum_distance_point_to_curve (t m : ℝ) : 
  (∃ (P : ℝ × ℝ), 
    P.2 = exp P.1 ∧ 
    (∀ (Q : ℝ × ℝ), Q.2 = exp Q.1 → (t - P.1)^2 + P.2^2 ≤ (t - Q.1)^2 + Q.2^2) ∧
    (t - P.1)^2 + P.2^2 = 12) →
  t = 3 + log 3 / 2 := by
sorry


end minimum_distance_point_to_curve_l1670_167008


namespace intersecting_circles_B_coords_l1670_167012

/-- Two circles with centers on the line y = 1 - x, intersecting at points A and B -/
structure IntersectingCircles where
  O₁ : ℝ × ℝ
  O₂ : ℝ × ℝ
  A : ℝ × ℝ
  B : ℝ × ℝ
  centers_on_line : ∀ c, c = O₁ ∨ c = O₂ → c.2 = 1 - c.1
  A_coords : A = (-7, 9)

/-- The theorem stating that point B has coordinates (-8, 8) -/
theorem intersecting_circles_B_coords (circles : IntersectingCircles) : 
  circles.B = (-8, 8) := by
  sorry

end intersecting_circles_B_coords_l1670_167012


namespace special_quadrilateral_not_necessarily_square_l1670_167093

/-- A quadrilateral with perpendicular diagonals, an inscribed circle, and a circumscribed circle -/
structure SpecialQuadrilateral where
  /-- The quadrilateral has perpendicular diagonals -/
  perpendicular_diagonals : Bool
  /-- The quadrilateral has an inscribed circle -/
  has_inscribed_circle : Bool
  /-- The quadrilateral has a circumscribed circle -/
  has_circumscribed_circle : Bool

/-- Definition of a square -/
def is_square (q : SpecialQuadrilateral) : Bool := sorry

/-- Theorem: A quadrilateral with perpendicular diagonals, an inscribed circle, and a circumscribed circle is not necessarily a square -/
theorem special_quadrilateral_not_necessarily_square :
  ∃ q : SpecialQuadrilateral,
    q.perpendicular_diagonals ∧
    q.has_inscribed_circle ∧
    q.has_circumscribed_circle ∧
    ¬(is_square q) :=
  sorry

end special_quadrilateral_not_necessarily_square_l1670_167093


namespace three_numbers_ratio_l1670_167081

theorem three_numbers_ratio (a b c : ℕ+) : 
  (Nat.lcm a (Nat.lcm b c) = 2400) → 
  (Nat.gcd a (Nat.gcd b c) = 40) → 
  (∃ (k : ℕ+), a = 3 * k ∧ b = 4 * k ∧ c = 5 * k) := by
sorry

end three_numbers_ratio_l1670_167081


namespace intersection_point_equality_l1670_167041

theorem intersection_point_equality (a b c d : ℝ) : 
  (1 = 1^2 + a * 1 + b) → 
  (1 = 1^2 + c * 1 + d) → 
  a^5 + d^6 = c^6 - b^5 := by
  sorry

end intersection_point_equality_l1670_167041


namespace smallest_n_with_square_sums_l1670_167040

theorem smallest_n_with_square_sums : ∃ (a b c : ℕ), 
  (a < b ∧ b < c) ∧ 
  (∃ (x y z : ℕ), a + b = x^2 ∧ a + c = y^2 ∧ b + c = z^2) ∧
  a + b + c = 55 ∧
  (∀ (n : ℕ), n < 55 → 
    ¬∃ (a' b' c' : ℕ), (a' < b' ∧ b' < c') ∧ 
    (∃ (x' y' z' : ℕ), a' + b' = x'^2 ∧ a' + c' = y'^2 ∧ b' + c' = z'^2) ∧
    a' + b' + c' = n) :=
by sorry

end smallest_n_with_square_sums_l1670_167040


namespace tangent_point_exists_l1670_167007

-- Define the curve
def f (x : ℝ) : ℝ := x^3 + x - 2

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

theorem tangent_point_exists : ∃ (x₀ y₀ : ℝ), 
  f x₀ = y₀ ∧ 
  f' x₀ = 4 ∧ 
  x₀ = -1 ∧ 
  y₀ = -4 :=
sorry

end tangent_point_exists_l1670_167007


namespace equation_solution_implies_a_range_l1670_167057

theorem equation_solution_implies_a_range (a : ℝ) :
  (∃ x : ℝ, Real.sqrt 3 * Real.sin x + Real.cos x = 2 * a - 1) →
  -1/2 ≤ a ∧ a ≤ 3/2 := by
  sorry

end equation_solution_implies_a_range_l1670_167057


namespace triangle_properties_l1670_167094

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c ∧
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧
  b * Real.cos A + a / 2 = c ∧
  c = 2 * a ∧
  b = 3 * Real.sqrt 3 →
  B = π / 3 ∧ (a * c * Real.sin B) / 2 = (9 * Real.sqrt 3) / 2 := by
  sorry

#check triangle_properties

end triangle_properties_l1670_167094


namespace pie_slices_theorem_l1670_167037

/-- Given the total number of pie slices sold and the number sold yesterday,
    calculate the number of slices served today. -/
def slices_served_today (total : ℕ) (yesterday : ℕ) : ℕ :=
  total - yesterday

theorem pie_slices_theorem :
  slices_served_today 7 5 = 2 :=
by sorry

end pie_slices_theorem_l1670_167037


namespace direct_proportion_relationship_l1670_167031

theorem direct_proportion_relationship (x y : ℝ) :
  (∃ k : ℝ, ∀ x, y - 2 = k * x) →  -- y-2 is directly proportional to x
  (1 = 1 ∧ y = -6) →              -- when x=1, y=-6
  y = -8 * x + 2 :=                -- relationship between y and x
by
  sorry

end direct_proportion_relationship_l1670_167031


namespace c_share_is_64_l1670_167022

/-- Given a total sum of money divided among three parties a, b, and c,
    where b's share is 65% of a's and c's share is 40% of a's,
    prove that c's share is 64 when the total sum is 328. -/
theorem c_share_is_64 (total : ℝ) (a b c : ℝ) :
  total = 328 →
  b = 0.65 * a →
  c = 0.40 * a →
  total = a + b + c →
  c = 64 := by
  sorry

end c_share_is_64_l1670_167022


namespace cos_sum_of_complex_exponentials_l1670_167042

theorem cos_sum_of_complex_exponentials (γ δ : ℝ) :
  Complex.exp (γ * I) = 4/5 + 3/5 * I →
  Complex.exp (δ * I) = -5/13 - 12/13 * I →
  Real.cos (γ + δ) = 16/65 := by
  sorry

end cos_sum_of_complex_exponentials_l1670_167042


namespace fernandez_family_has_nine_children_l1670_167046

/-- Represents the Fernandez family structure and ages -/
structure FernandezFamily where
  num_children : ℕ
  mother_age : ℕ
  children_ages : ℕ → ℕ
  average_family_age : ℕ
  father_age : ℕ
  grandmother_age : ℕ
  average_mother_children_age : ℕ

/-- The Fernandez family satisfies the given conditions -/
def is_valid_fernandez_family (f : FernandezFamily) : Prop :=
  f.average_family_age = 25 ∧
  f.father_age = 50 ∧
  f.grandmother_age = 70 ∧
  f.average_mother_children_age = 18

/-- The theorem stating that the Fernandez family has 9 children -/
theorem fernandez_family_has_nine_children (f : FernandezFamily) 
  (h : is_valid_fernandez_family f) : f.num_children = 9 := by
  sorry

#check fernandez_family_has_nine_children

end fernandez_family_has_nine_children_l1670_167046


namespace remainder_mod_six_l1670_167045

theorem remainder_mod_six (a : ℕ) (h1 : a % 2 = 1) (h2 : a % 3 = 2) : a % 6 = 5 := by
  sorry

end remainder_mod_six_l1670_167045


namespace gcd_n_squared_plus_four_n_plus_three_l1670_167062

theorem gcd_n_squared_plus_four_n_plus_three (n : ℕ) (h : n > 4) :
  Nat.gcd (n^2 + 4) (n + 3) = if (n + 3) % 13 = 0 then 13 else 1 := by
  sorry

end gcd_n_squared_plus_four_n_plus_three_l1670_167062


namespace smallest_n_for_unique_zero_solution_l1670_167049

theorem smallest_n_for_unique_zero_solution :
  ∃ (n : ℕ), n ≥ 1 ∧
  (∀ (a b c d : ℤ), a^2 + b^2 + c^2 - n * d^2 = 0 → a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 0) ∧
  (∀ (m : ℕ), m < n →
    ∃ (a b c d : ℤ), (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0 ∨ d ≠ 0) ∧ a^2 + b^2 + c^2 - m * d^2 = 0) ∧
  n = 4 :=
sorry

end smallest_n_for_unique_zero_solution_l1670_167049


namespace lunchroom_students_l1670_167058

/-- The number of students sitting at each table -/
def students_per_table : ℕ := 6

/-- The number of tables in the lunchroom -/
def number_of_tables : ℕ := 34

/-- The total number of students in the lunchroom -/
def total_students : ℕ := students_per_table * number_of_tables

theorem lunchroom_students : total_students = 204 := by
  sorry

end lunchroom_students_l1670_167058


namespace sqrt_200_simplified_l1670_167087

theorem sqrt_200_simplified : Real.sqrt 200 = 10 * Real.sqrt 2 := by
  sorry

end sqrt_200_simplified_l1670_167087


namespace circle_center_sum_l1670_167050

/-- Given a circle with equation x^2 + y^2 = 6x + 8y + 9, 
    the sum of the x and y coordinates of its center is 7 -/
theorem circle_center_sum (x y : ℝ) : 
  x^2 + y^2 = 6*x + 8*y + 9 → ∃ h k : ℝ, (h, k) = (3, 4) ∧ h + k = 7 := by
  sorry

end circle_center_sum_l1670_167050


namespace volunteers_next_meeting_l1670_167099

def alison_schedule := 5
def ben_schedule := 3
def carla_schedule := 9
def dave_schedule := 8

theorem volunteers_next_meeting :
  Nat.lcm alison_schedule (Nat.lcm ben_schedule (Nat.lcm carla_schedule dave_schedule)) = 360 := by
  sorry

end volunteers_next_meeting_l1670_167099


namespace cricket_average_l1670_167084

theorem cricket_average (innings : Nat) (next_runs : Nat) (increase : Nat) (current_average : Nat) : 
  innings = 10 →
  next_runs = 84 →
  increase = 4 →
  (innings * current_average + next_runs) / (innings + 1) = current_average + increase →
  current_average = 40 := by
  sorry

end cricket_average_l1670_167084


namespace angle_between_points_l1670_167071

/-- The angle between two points on a spherical Earth given their coordinates -/
def angleOnSphere (latA longA latB longB : Real) : Real :=
  360 - longA - longB

/-- Point A's coordinates -/
def pointA : (Real × Real) := (0, 100)

/-- Point B's coordinates -/
def pointB : (Real × Real) := (45, -115)

theorem angle_between_points :
  angleOnSphere pointA.1 pointA.2 pointB.1 pointB.2 = 145 := by
  sorry

end angle_between_points_l1670_167071


namespace stickers_per_page_l1670_167051

theorem stickers_per_page (total_stickers : ℕ) (total_pages : ℕ) (h1 : total_stickers = 220) (h2 : total_pages = 22) :
  total_stickers / total_pages = 10 := by
  sorry

end stickers_per_page_l1670_167051


namespace ap_sum_70_l1670_167004

def arithmetic_progression (a d : ℚ) (n : ℕ) : ℚ :=
  n / 2 * (2 * a + (n - 1) * d)

theorem ap_sum_70 (a d : ℚ) :
  arithmetic_progression a d 20 = 150 →
  arithmetic_progression a d 50 = 20 →
  arithmetic_progression a d 70 = -910/3 := by
  sorry

end ap_sum_70_l1670_167004


namespace book_discount_l1670_167065

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : Nat
  ones : Nat
  h_tens : tens ≥ 1 ∧ tens ≤ 9
  h_ones : ones ≥ 0 ∧ ones ≤ 9

/-- The price reduction percentage -/
def reduction_percentage : Rat := 62.5

/-- Calculates the value of a two-digit number -/
def value (n : TwoDigitNumber) : Nat := 10 * n.tens + n.ones

/-- Checks if two TwoDigitNumbers have the same digits in different order -/
def same_digits (n1 n2 : TwoDigitNumber) : Prop :=
  (n1.tens = n2.ones) ∧ (n1.ones = n2.tens)

theorem book_discount (original reduced : TwoDigitNumber)
  (h_reduction : value reduced = (100 - reduction_percentage) / 100 * value original)
  (h_same_digits : same_digits original reduced) :
  value original - value reduced = 45 := by
  sorry

end book_discount_l1670_167065


namespace salary_may_value_l1670_167038

def salary_problem (jan feb mar apr may : ℕ) : Prop :=
  (jan + feb + mar + apr) / 4 = 8000 ∧
  (feb + mar + apr + may) / 4 = 8300 ∧
  jan = 5300

theorem salary_may_value :
  ∀ (jan feb mar apr may : ℕ),
    salary_problem jan feb mar apr may →
    may = 6500 :=
by
  sorry

end salary_may_value_l1670_167038


namespace canoe_kayak_ratio_l1670_167072

theorem canoe_kayak_ratio : 
  ∀ (C K : ℕ),
  (9 * C + 12 * K = 432) →  -- Total revenue
  (C = K + 6) →             -- 6 more canoes than kayaks
  (∃ (n : ℕ), C = 3 * n * K) →  -- Canoes are a multiple of 3 times kayaks
  (C : ℚ) / K = 4 / 3 :=    -- Ratio of canoes to kayaks is 4:3
by
  sorry

#check canoe_kayak_ratio

end canoe_kayak_ratio_l1670_167072


namespace horner_method_operations_l1670_167089

/-- Horner's method for polynomial evaluation -/
def horner_eval (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- Count operations in Horner's method -/
def horner_count (coeffs : List ℝ) : Nat × Nat :=
  (coeffs.length - 1, coeffs.length - 1)

/-- The polynomial f(x) = 6x^6 + 4x^5 - 2x^4 + 5x^3 - 7x^2 - 2x + 5 -/
def f : List ℝ := [6, 4, -2, 5, -7, -2, 5]

theorem horner_method_operations :
  horner_count f = (6, 3) ∧
  horner_eval f 2 = f.foldl (fun acc a => acc * 2 + a) 0 :=
sorry

end horner_method_operations_l1670_167089


namespace brad_green_balloons_l1670_167056

/-- Calculates the number of green balloons Brad has -/
def green_balloons (total : ℕ) (initial_red : ℕ) (popped_red : ℕ) (blue : ℕ) : ℕ :=
  let remaining_red := initial_red - popped_red
  let non_red := total - remaining_red
  let green_and_yellow := non_red - blue
  (2 * green_and_yellow) / 5

/-- Theorem stating that Brad has 12 green balloons -/
theorem brad_green_balloons :
  green_balloons 50 15 3 7 = 12 := by
  sorry

end brad_green_balloons_l1670_167056


namespace sin_cos_theorem_l1670_167006

theorem sin_cos_theorem (θ : ℝ) (z : ℂ) : 
  z = (Real.sin θ - 2 * Real.cos θ) + (Real.sin θ + 2 * Real.cos θ) * Complex.I →
  z.re = 0 →
  z.im ≠ 0 →
  Real.sin θ * Real.cos θ = 2/5 :=
by sorry

end sin_cos_theorem_l1670_167006


namespace only_cube_has_congruent_views_l1670_167029

-- Define the possible solids
inductive Solid
  | Cone
  | Cylinder
  | Cube
  | SquarePyramid

-- Define a function to check if a solid has congruent views
def hasCongruentViews (s : Solid) : Prop :=
  match s with
  | Solid.Cone => False
  | Solid.Cylinder => False
  | Solid.Cube => True
  | Solid.SquarePyramid => False

-- Theorem stating that only a cube has congruent views
theorem only_cube_has_congruent_views :
  ∀ s : Solid, hasCongruentViews s ↔ s = Solid.Cube :=
by sorry

end only_cube_has_congruent_views_l1670_167029


namespace driver_net_rate_of_pay_l1670_167096

/-- Calculates the net rate of pay for a driver given travel conditions and expenses. -/
theorem driver_net_rate_of_pay
  (travel_time : ℝ)
  (speed : ℝ)
  (fuel_efficiency : ℝ)
  (pay_per_mile : ℝ)
  (gas_price : ℝ)
  (h1 : travel_time = 3)
  (h2 : speed = 50)
  (h3 : fuel_efficiency = 25)
  (h4 : pay_per_mile = 0.60)
  (h5 : gas_price = 2.50)
  : (pay_per_mile * speed * travel_time - (speed * travel_time / fuel_efficiency) * gas_price) / travel_time = 25 := by
  sorry

#check driver_net_rate_of_pay

end driver_net_rate_of_pay_l1670_167096


namespace sum_to_k_is_triangular_square_k_values_l1670_167016

def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

def sum_to_k (k : ℕ) : ℕ := k * (k + 1) / 2

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m^2

theorem sum_to_k_is_triangular_square (k : ℕ) : Prop :=
  ∃ n : ℕ, sum_to_k k = n^2 ∧ n < 150 ∧ is_perfect_square (triangular_number n)

theorem k_values : {k : ℕ | sum_to_k_is_triangular_square k} = {1, 8, 39, 92, 168} := by
  sorry

end sum_to_k_is_triangular_square_k_values_l1670_167016


namespace least_positive_integer_with_remainders_l1670_167061

theorem least_positive_integer_with_remainders (M : ℕ) : 
  (M % 11 = 10 ∧ M % 12 = 11 ∧ M % 13 = 12 ∧ M % 14 = 13) → 
  (∀ n : ℕ, n > 0 ∧ n % 11 = 10 ∧ n % 12 = 11 ∧ n % 13 = 12 ∧ n % 14 = 13 → M ≤ n) → 
  M = 30029 := by
sorry

end least_positive_integer_with_remainders_l1670_167061


namespace item_sale_ratio_l1670_167080

theorem item_sale_ratio (x y c : ℝ) (hx : x = 0.9 * c) (hy : y = 1.2 * c) :
  y / x = 4 / 3 := by
sorry

end item_sale_ratio_l1670_167080


namespace cupcake_package_size_l1670_167069

theorem cupcake_package_size :
  ∀ (small_package_size : ℕ) (total_cupcakes : ℕ) (small_packages : ℕ) (larger_package_size : ℕ),
    small_package_size = 10 →
    total_cupcakes = 100 →
    small_packages = 4 →
    total_cupcakes = small_package_size * small_packages + larger_package_size →
    larger_package_size = 60 := by
  sorry

end cupcake_package_size_l1670_167069


namespace units_digit_product_l1670_167043

theorem units_digit_product (a b c : ℕ) : 
  (3^1004 * 7^1003 * 17^1002) % 10 = 7 := by
  sorry

end units_digit_product_l1670_167043


namespace positive_real_inequality_l1670_167092

theorem positive_real_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a^4 * b^b * c^c ≥ min a (min b c) * min b (min a c) * min c (min a b) := by
  sorry

end positive_real_inequality_l1670_167092


namespace no_perfect_squares_l1670_167064

def digit_repeat (d : ℕ) (n : ℕ) : ℕ :=
  d * (10^100 - 1) / 9

def two_digit_repeat (d₁ d₂ : ℕ) (n : ℕ) : ℕ :=
  (10 * d₁ + d₂) * (10^99 - 1) / 99 + d₁ * 10^99

def N₁ : ℕ := digit_repeat 3 100
def N₂ : ℕ := digit_repeat 6 100
def N₃ : ℕ := two_digit_repeat 1 5 100
def N₄ : ℕ := two_digit_repeat 2 1 100
def N₅ : ℕ := two_digit_repeat 2 7 100

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * k

theorem no_perfect_squares :
  ¬(is_perfect_square N₁ ∨ is_perfect_square N₂ ∨ is_perfect_square N₃ ∨ is_perfect_square N₄ ∨ is_perfect_square N₅) :=
by sorry

end no_perfect_squares_l1670_167064


namespace arithmetic_sequence_problem_l1670_167023

theorem arithmetic_sequence_problem :
  ∃ (a b c : ℝ), 
    (a > b ∧ b > c) ∧  -- Monotonically decreasing
    (b - a = c - b) ∧  -- Arithmetic sequence
    (a + b + c = 12) ∧ -- Sum is 12
    (a * b * c = 48) ∧ -- Product is 48
    (a = 6 ∧ b = 4 ∧ c = 2) := by
  sorry

end arithmetic_sequence_problem_l1670_167023


namespace geometric_series_common_ratio_l1670_167036

theorem geometric_series_common_ratio : 
  let a₁ : ℚ := 2/7
  let a₂ : ℚ := 10/49
  let a₃ : ℚ := 50/343
  let r : ℚ := a₂ / a₁
  (∀ n : ℕ, n ≥ 1 → a₁ * r^(n-1) = (2/7) * (5/7)^(n-1)) →
  r = 5/7 := by sorry

end geometric_series_common_ratio_l1670_167036


namespace box_side_length_l1670_167030

/-- The length of one side of a cubic box given total volume, cost per box, and total cost -/
theorem box_side_length 
  (total_volume : ℝ) 
  (cost_per_box : ℝ) 
  (total_cost : ℝ) 
  (h1 : total_volume = 2.16e6)  -- 2.16 million cubic inches
  (h2 : cost_per_box = 0.5)     -- $0.50 per box
  (h3 : total_cost = 225)       -- $225 total cost
  : ∃ (side_length : ℝ), abs (side_length - 16.89) < 0.01 := by
  sorry


end box_side_length_l1670_167030


namespace annual_compound_interest_rate_exists_l1670_167091

-- Define the initial principal
def initial_principal : ℝ := 780

-- Define the final amount
def final_amount : ℝ := 1300

-- Define the time period in years
def time_period : ℕ := 4

-- Define the compound interest equation
def compound_interest_equation (r : ℝ) : Prop :=
  final_amount = initial_principal * (1 + r) ^ time_period

-- Theorem statement
theorem annual_compound_interest_rate_exists :
  ∃ r : ℝ, compound_interest_equation r ∧ r > 0 ∧ r < 1 :=
sorry

end annual_compound_interest_rate_exists_l1670_167091


namespace dog_ratio_proof_l1670_167063

/-- Proves that for 12 dogs with 36 paws on the ground, split equally between those on back legs and all fours, the ratio of dogs on back legs to all fours is 1:1 -/
theorem dog_ratio_proof (total_dogs : ℕ) (total_paws : ℕ) 
  (h1 : total_dogs = 12) 
  (h2 : total_paws = 36) 
  (h3 : ∃ x y : ℕ, x + y = total_dogs ∧ x = y ∧ 2*x + 4*y = total_paws) : 
  ∃ x y : ℕ, x + y = total_dogs ∧ x = y ∧ x / y = 1 := by
  sorry

#check dog_ratio_proof

end dog_ratio_proof_l1670_167063


namespace megatek_rd_percentage_l1670_167053

theorem megatek_rd_percentage :
  ∀ (manufacturing_angle hr_angle sales_angle rd_angle : ℝ),
  manufacturing_angle = 54 →
  hr_angle = 2 * manufacturing_angle →
  sales_angle = (1/2) * hr_angle →
  rd_angle = 360 - (manufacturing_angle + hr_angle + sales_angle) →
  (rd_angle / 360) * 100 = 40 :=
by
  sorry

end megatek_rd_percentage_l1670_167053


namespace lowest_price_theorem_l1670_167039

/-- Calculates the lowest price per component to break even --/
def lowest_price_per_component (production_cost shipping_cost : ℚ) 
  (fixed_costs : ℚ) (num_components : ℕ) : ℚ :=
  (production_cost + shipping_cost + fixed_costs / num_components)

theorem lowest_price_theorem (production_cost shipping_cost : ℚ) 
  (fixed_costs : ℚ) (num_components : ℕ) :
  lowest_price_per_component production_cost shipping_cost fixed_costs num_components =
  (production_cost * num_components + shipping_cost * num_components + fixed_costs) / num_components :=
by sorry

#eval lowest_price_per_component 80 2 16200 150

end lowest_price_theorem_l1670_167039


namespace chord_length_l1670_167034

/-- Circle C with equation x^2 + y^2 - 4x - 4y + 4 = 0 -/
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 4*y + 4 = 0

/-- Line l passing through points (4,0) and (0,2) -/
def line_l (x y : ℝ) : Prop := x + 2*y = 4

/-- The length of the chord cut by line l on circle C is 8√5/5 -/
theorem chord_length : ∃ (x₁ y₁ x₂ y₂ : ℝ),
  circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧ 
  line_l x₁ y₁ ∧ line_l x₂ y₂ ∧
  ((x₁ - x₂)^2 + (y₁ - y₂)^2) = (8*Real.sqrt 5/5)^2 :=
sorry

end chord_length_l1670_167034


namespace pastry_distribution_l1670_167074

/-- The number of pastries the Hatter initially had -/
def total_pastries : ℕ := 32

/-- The fraction of pastries March Hare ate -/
def march_hare_fraction : ℚ := 5/16

/-- The fraction of remaining pastries Dormouse ate -/
def dormouse_fraction : ℚ := 7/11

/-- The number of pastries left for the Hatter -/
def hatter_leftover : ℕ := 8

/-- The number of pastries March Hare ate -/
def march_hare_eaten : ℕ := 10

/-- The number of pastries Dormouse ate -/
def dormouse_eaten : ℕ := 14

theorem pastry_distribution :
  (march_hare_eaten = (total_pastries : ℚ) * march_hare_fraction) ∧
  (dormouse_eaten = ((total_pastries - march_hare_eaten) : ℚ) * dormouse_fraction) ∧
  (hatter_leftover = total_pastries - march_hare_eaten - dormouse_eaten) :=
by sorry

end pastry_distribution_l1670_167074


namespace odd_function_composition_periodic_function_composition_exists_non_decreasing_composition_inverse_function_zero_l1670_167028

-- Define the function f: ℝ → ℝ
variable (f : ℝ → ℝ)

-- Statement 1
theorem odd_function_composition (h : ∀ x, f (-x) = -f x) : ∀ x, (f ∘ f) (-x) = -(f ∘ f) x :=
sorry

-- Statement 2
theorem periodic_function_composition (h : ∃ T, ∀ x, f (x + T) = f x) : 
  ∃ T, ∀ x, (f ∘ f) (x + T) = (f ∘ f) x :=
sorry

-- Statement 3
theorem exists_non_decreasing_composition :
  ∃ f : ℝ → ℝ, (∀ x y, x < y → f x > f y) ∧ ¬(∀ x y, x < y → (f ∘ f) x > (f ∘ f) y) :=
sorry

-- Statement 4
theorem inverse_function_zero (h₁ : Function.Bijective f) 
  (h₂ : ∃ x, f x = Function.invFun f x) : ∃ x, f x = x :=
sorry

end odd_function_composition_periodic_function_composition_exists_non_decreasing_composition_inverse_function_zero_l1670_167028


namespace phillip_test_results_l1670_167083

/-- Represents the number of questions Phillip gets right on a test -/
def correct_answers (total : ℕ) (percentage : ℚ) : ℚ :=
  (total : ℚ) * percentage

/-- Represents the total number of correct answers across all tests -/
def total_correct_answers (x : ℕ) : ℚ :=
  correct_answers 40 (75/100) + correct_answers 50 (98/100) + (x : ℚ) * ((100 - x : ℚ)/100)

theorem phillip_test_results (x : ℕ) (h : 1 ≤ x ∧ x ≤ 100) :
  total_correct_answers x = 79 + (x : ℚ) * ((100 - x : ℚ)/100) :=
by sorry

end phillip_test_results_l1670_167083


namespace mul_exp_analogy_l1670_167010

-- Define multiplication recursively
def mul_rec (k : ℕ) : ℕ → ℕ
| 0     => 0                   -- Base case
| n + 1 => k + mul_rec k n     -- Recursive step

-- Define exponentiation recursively
def exp_rec (k : ℕ) : ℕ → ℕ
| 0     => 1                   -- Base case
| n + 1 => k * exp_rec k n     -- Recursive step

-- Theorem stating the analogy between multiplication and exponentiation
theorem mul_exp_analogy :
  (∀ k n : ℕ, mul_rec k (n + 1) = k + mul_rec k n) ↔
  (∀ k n : ℕ, exp_rec k (n + 1) = k * exp_rec k n) :=
sorry

end mul_exp_analogy_l1670_167010


namespace distance_XY_is_24_l1670_167001

/-- The distance between points X and Y in miles. -/
def distance_XY : ℝ := 24

/-- Yolanda's walking rate in miles per hour. -/
def yolanda_rate : ℝ := 3

/-- Bob's walking rate in miles per hour. -/
def bob_rate : ℝ := 4

/-- The distance Bob has walked when they meet, in miles. -/
def bob_distance : ℝ := 12

/-- The time difference between Yolanda and Bob's start, in hours. -/
def time_difference : ℝ := 1

theorem distance_XY_is_24 : 
  distance_XY = yolanda_rate * (bob_distance / bob_rate + time_difference) + bob_distance :=
sorry

end distance_XY_is_24_l1670_167001


namespace thermos_capacity_is_16_l1670_167076

/-- The capacity of a coffee thermos -/
def thermos_capacity (fills_per_day : ℕ) (days_per_week : ℕ) (current_consumption : ℚ) (normal_consumption_ratio : ℚ) : ℚ :=
  (current_consumption / normal_consumption_ratio) / (fills_per_day * days_per_week)

/-- Proof that the thermos capacity is 16 ounces -/
theorem thermos_capacity_is_16 :
  thermos_capacity 2 5 40 (1/4) = 16 := by
  sorry

end thermos_capacity_is_16_l1670_167076


namespace hyperbola_equation_l1670_167013

/-- The equation of a hyperbola passing through (6, √3) with asymptotes y = ±x/3 -/
theorem hyperbola_equation (x y : ℝ) :
  (∀ k : ℝ, k * x = 3 * y → k = 1 ∨ k = -1) →  -- asymptotes condition
  6^2 / 9 - (Real.sqrt 3)^2 = 1 →               -- point condition
  x^2 / 9 - y^2 = 1 :=
by sorry

end hyperbola_equation_l1670_167013


namespace sallys_peaches_l1670_167047

/-- The total number of peaches Sally has after picking more from the orchard -/
def total_peaches (initial : ℕ) (picked : ℕ) : ℕ :=
  initial + picked

/-- Theorem stating that given Sally's initial 13 peaches and her additional 55 picked peaches, 
    the total number of peaches is 68 -/
theorem sallys_peaches : total_peaches 13 55 = 68 := by
  sorry

end sallys_peaches_l1670_167047


namespace sphere_surface_area_ratio_l1670_167070

theorem sphere_surface_area_ratio (r₁ r₂ : ℝ) (h : (4/3 * π * r₁^3) / (4/3 * π * r₂^3) = 8/27) :
  (4 * π * r₁^2) / (4 * π * r₂^2) = 4/9 := by
sorry

end sphere_surface_area_ratio_l1670_167070


namespace tangent_line_circle_product_range_l1670_167085

theorem tangent_line_circle_product_range (a b : ℝ) :
  a > 0 →
  b > 0 →
  (∃ x y : ℝ, x + y = 1 ∧ (x - a)^2 + (y - b)^2 = 2) →
  (∀ x y : ℝ, x + y = 1 → (x - a)^2 + (y - b)^2 ≥ 2) →
  0 < a * b ∧ a * b ≤ 9/4 := by
  sorry

end tangent_line_circle_product_range_l1670_167085


namespace quadratic_inequalities_l1670_167032

-- Define the quadratic function
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_inequalities :
  ∀ a b c : ℝ,
  -- Part 1
  (∀ x : ℝ, -3 < x → x < 4 → f a b c x > 0) →
  (∀ x : ℝ, -3 < x → x < 5 → b * x^2 + 2 * a * x - (c + 3 * b) < 0) ∧
  -- Part 2
  (b = 2 → a > c → (∀ x : ℝ, f a b c x ≥ 0) → (∃ x₀ : ℝ, f a b c x₀ = 0) →
    ∃ min : ℝ, min = 2 * Real.sqrt 2 ∧ ∀ x : ℝ, (a^2 + c^2) / (a - c) ≥ min) ∧
  -- Part 3
  (a < b → (∀ x : ℝ, f a b c x ≥ 0) →
    ∃ min : ℝ, min = 8 ∧ ∀ x : ℝ, (a + 2 * b + 4 * c) / (b - a) ≥ min) :=
by sorry

end quadratic_inequalities_l1670_167032
