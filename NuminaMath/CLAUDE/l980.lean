import Mathlib

namespace NUMINAMATH_CALUDE_grading_ratio_l980_98064

/-- A grading method for a test with 100 questions. -/
structure GradingMethod where
  total_questions : Nat
  score : Nat
  correct_answers : Nat

/-- Theorem stating the ratio of points subtracted per incorrect answer
    to points given per correct answer is 2:1 -/
theorem grading_ratio (g : GradingMethod)
  (h1 : g.total_questions = 100)
  (h2 : g.score = 73)
  (h3 : g.correct_answers = 91) :
  (g.correct_answers - g.score) / (g.total_questions - g.correct_answers) = 2 := by
  sorry


end NUMINAMATH_CALUDE_grading_ratio_l980_98064


namespace NUMINAMATH_CALUDE_negation_of_forall_positive_l980_98002

theorem negation_of_forall_positive (f : ℝ → ℝ) :
  (¬ ∀ x : ℝ, f x > 0) ↔ (∃ x : ℝ, f x ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_forall_positive_l980_98002


namespace NUMINAMATH_CALUDE_stream_current_is_six_l980_98053

/-- Represents the man's rowing scenario -/
structure RowingScenario where
  r : ℝ  -- man's usual rowing speed in still water (miles per hour)
  w : ℝ  -- speed of the stream's current (miles per hour)

/-- The conditions of the rowing problem -/
def rowing_conditions (s : RowingScenario) : Prop :=
  -- Downstream time is 6 hours less than upstream time
  18 / (s.r + s.w) + 6 = 18 / (s.r - s.w) ∧
  -- When rowing speed is tripled, downstream time is 2 hours less than upstream time
  18 / (3 * s.r + s.w) + 2 = 18 / (3 * s.r - s.w)

/-- The theorem stating that the stream's current is 6 miles per hour -/
theorem stream_current_is_six (s : RowingScenario) :
  rowing_conditions s → s.w = 6 := by
  sorry

end NUMINAMATH_CALUDE_stream_current_is_six_l980_98053


namespace NUMINAMATH_CALUDE_angle_measure_from_point_l980_98032

/-- If a point P(sin 40°, 1 + cos 40°) is on the terminal side of an acute angle α, then α = 70°. -/
theorem angle_measure_from_point (α : Real) : 
  α > 0 ∧ α < 90 ∧ 
  ∃ (P : ℝ × ℝ), P.1 = Real.sin (40 * π / 180) ∧ P.2 = 1 + Real.cos (40 * π / 180) ∧
  P.2 / P.1 = Real.tan α → 
  α = 70 * π / 180 := by
sorry

end NUMINAMATH_CALUDE_angle_measure_from_point_l980_98032


namespace NUMINAMATH_CALUDE_tree_planting_speeds_l980_98088

-- Define the given constants
def distance : ℝ := 10
def time_difference : ℝ := 1.5
def speed_ratio : ℝ := 2.5

-- Define the walking speed and cycling speed
def walking_speed : ℝ := 4
def cycling_speed : ℝ := 10

-- Define the increased cycling speed
def increased_cycling_speed : ℝ := 12

-- Theorem statement
theorem tree_planting_speeds :
  (distance / walking_speed - distance / cycling_speed = time_difference) ∧
  (cycling_speed = speed_ratio * walking_speed) ∧
  (distance / increased_cycling_speed = distance / cycling_speed - 1/6) :=
sorry

end NUMINAMATH_CALUDE_tree_planting_speeds_l980_98088


namespace NUMINAMATH_CALUDE_inequality_multiplication_l980_98092

theorem inequality_multiplication (a b : ℝ) (h : a > b) : 3 * a > 3 * b := by
  sorry

end NUMINAMATH_CALUDE_inequality_multiplication_l980_98092


namespace NUMINAMATH_CALUDE_optimal_garden_dimensions_l980_98079

/-- Represents a rectangular garden -/
structure RectangularGarden where
  width : ℝ
  length : ℝ

/-- Calculates the area of a rectangular garden -/
def area (g : RectangularGarden) : ℝ := g.width * g.length

/-- Calculates the perimeter of a rectangular garden -/
def perimeter (g : RectangularGarden) : ℝ := 2 * (g.width + g.length)

/-- Theorem: Optimal dimensions for a 600 sq ft garden with length twice the width -/
theorem optimal_garden_dimensions :
  ∃ (g : RectangularGarden),
    area g = 600 ∧
    g.length = 2 * g.width ∧
    g.width = 10 * Real.sqrt 3 ∧
    g.length = 20 * Real.sqrt 3 ∧
    ∀ (h : RectangularGarden),
      area h = 600 → h.length = 2 * h.width → perimeter h ≥ perimeter g :=
by sorry

end NUMINAMATH_CALUDE_optimal_garden_dimensions_l980_98079


namespace NUMINAMATH_CALUDE_total_red_balloons_l980_98012

/-- The total number of red balloons given the number of balloons each person has -/
def total_balloons (fred_balloons sam_balloons dan_balloons : ℕ) : ℕ :=
  fred_balloons + sam_balloons + dan_balloons

/-- Theorem stating that the total number of red balloons is 72 -/
theorem total_red_balloons : 
  total_balloons 10 46 16 = 72 := by
  sorry

end NUMINAMATH_CALUDE_total_red_balloons_l980_98012


namespace NUMINAMATH_CALUDE_man_son_age_ratio_l980_98020

/-- Proves that the ratio of a man's age to his son's age in two years is 2:1,
    given the initial conditions. -/
theorem man_son_age_ratio :
  ∀ (son_age man_age : ℕ),
    son_age = 18 →
    man_age = son_age + 20 →
    ∃ (k : ℕ), (man_age + 2) = k * (son_age + 2) →
    (man_age + 2) / (son_age + 2) = 2 := by
  sorry

#check man_son_age_ratio

end NUMINAMATH_CALUDE_man_son_age_ratio_l980_98020


namespace NUMINAMATH_CALUDE_divisibility_of_sum_of_cubes_l980_98003

theorem divisibility_of_sum_of_cubes (n m : ℕ+) 
  (h : n^3 + (n+1)^3 + (n+2)^3 = m^3) : 
  4 ∣ (n+1) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_sum_of_cubes_l980_98003


namespace NUMINAMATH_CALUDE_candy_necklace_blocks_l980_98030

/-- The number of friends receiving candy necklaces -/
def num_friends : ℕ := 8

/-- The number of candy pieces per necklace -/
def pieces_per_necklace : ℕ := 10

/-- The number of candy pieces produced by one block -/
def pieces_per_block : ℕ := 30

/-- The minimum number of whole blocks needed to make necklaces for all friends -/
def min_blocks_needed : ℕ := 3

theorem candy_necklace_blocks :
  (num_friends * pieces_per_necklace + pieces_per_block - 1) / pieces_per_block = min_blocks_needed :=
sorry

end NUMINAMATH_CALUDE_candy_necklace_blocks_l980_98030


namespace NUMINAMATH_CALUDE_insecticide_potency_range_specific_insecticide_potency_range_l980_98078

/-- Given two insecticide powders, find the range of potency for the second powder
    to achieve a specific mixture potency. -/
theorem insecticide_potency_range 
  (weight1 : ℝ) (potency1 : ℝ) (weight2 : ℝ) 
  (lower_bound : ℝ) (upper_bound : ℝ) :
  weight1 > 0 ∧ weight2 > 0 ∧
  0 < potency1 ∧ potency1 < 1 ∧
  0 < lower_bound ∧ lower_bound < upper_bound ∧ upper_bound < 1 →
  ∃ (lower_x upper_x : ℝ),
    lower_x > potency1 ∧
    ∀ x, lower_x < x ∧ x < upper_x →
      lower_bound < (weight1 * potency1 + weight2 * x) / (weight1 + weight2) ∧
      (weight1 * potency1 + weight2 * x) / (weight1 + weight2) < upper_bound :=
by sorry

/-- The specific insecticide potency range problem. -/
theorem specific_insecticide_potency_range :
  ∃ (lower_x upper_x : ℝ),
    lower_x = 0.33 ∧ upper_x = 0.42 ∧
    ∀ x, 0.33 < x ∧ x < 0.42 →
      0.25 < (40 * 0.15 + 50 * x) / (40 + 50) ∧
      (40 * 0.15 + 50 * x) / (40 + 50) < 0.30 :=
by sorry

end NUMINAMATH_CALUDE_insecticide_potency_range_specific_insecticide_potency_range_l980_98078


namespace NUMINAMATH_CALUDE_line_through_points_l980_98080

/-- Given a line with equation x = 3y + 5 passing through points (m, n) and (m + 2, n + p),
    prove that p = 2/3 -/
theorem line_through_points (m n p : ℝ) : 
  (m = 3 * n + 5) ∧ (m + 2 = 3 * (n + p) + 5) → p = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_line_through_points_l980_98080


namespace NUMINAMATH_CALUDE_student_distribution_theorem_l980_98076

/-- The number of ways to distribute n students among k groups, with each student choosing exactly one group -/
def distribute_students (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of ways to choose m items from a set of n items -/
def choose (n : ℕ) (m : ℕ) : ℕ := sorry

theorem student_distribution_theorem :
  let total_students : ℕ := 4
  let total_groups : ℕ := 4
  let groups_to_fill : ℕ := 3
  distribute_students total_students groups_to_fill = 36 :=
by sorry

end NUMINAMATH_CALUDE_student_distribution_theorem_l980_98076


namespace NUMINAMATH_CALUDE_rainfall_difference_l980_98049

/-- Rainfall data for Tropical Storm Sally -/
structure RainfallData where
  day1 : ℕ
  day2 : ℕ
  day3 : ℕ

/-- Conditions for Tropical Storm Sally's rainfall -/
def sallysRainfall : RainfallData where
  day1 := 4
  day2 := 5 * 4
  day3 := 18

/-- Theorem: The difference between the sum of the first two days' rainfall and the third day's rainfall is 6 inches -/
theorem rainfall_difference (data : RainfallData := sallysRainfall) :
  (data.day1 + data.day2) - data.day3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_rainfall_difference_l980_98049


namespace NUMINAMATH_CALUDE_no_natural_number_with_three_prime_divisors_l980_98019

theorem no_natural_number_with_three_prime_divisors :
  ¬ ∃ (m p q r : ℕ),
    (Prime p ∧ Prime q ∧ Prime r) ∧
    (∃ (a b c : ℕ), m = p^a * q^b * r^c) ∧
    (p - 1 ∣ m) ∧
    (q * r - 1 ∣ m) ∧
    ¬(q - 1 ∣ m) ∧
    ¬(r - 1 ∣ m) ∧
    ¬(3 ∣ q + r) := by
  sorry

end NUMINAMATH_CALUDE_no_natural_number_with_three_prime_divisors_l980_98019


namespace NUMINAMATH_CALUDE_same_solution_l980_98010

theorem same_solution (x y : ℝ) : 
  (4 * x - 8 * y - 5 = 0) ↔ (8 * x - 16 * y - 10 = 0) := by
  sorry

end NUMINAMATH_CALUDE_same_solution_l980_98010


namespace NUMINAMATH_CALUDE_circle_satisfies_conditions_l980_98077

-- Define the given circle
def given_circle (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 6*y + 5 = 0

-- Define the sought circle
def sought_circle (x y : ℝ) : Prop :=
  (x - 5)^2 + (y + 1)^2 = 5

-- Define the tangent condition
def is_tangent (c1 c2 : (ℝ → ℝ → Prop)) (x y : ℝ) : Prop :=
  c1 x y ∧ c2 x y ∧ ∃ (m : ℝ), ∀ (dx dy : ℝ),
    (c1 (x + dx) (y + dy) → m * dx = dy) ∧
    (c2 (x + dx) (y + dy) → m * dx = dy)

theorem circle_satisfies_conditions :
  sought_circle 3 (-2) ∧
  is_tangent given_circle sought_circle 0 1 :=
sorry

end NUMINAMATH_CALUDE_circle_satisfies_conditions_l980_98077


namespace NUMINAMATH_CALUDE_sin_graph_shift_symmetry_l980_98006

open Real

theorem sin_graph_shift_symmetry (φ : ℝ) :
  (∀ x, ∃ y, y = sin (2*x + φ)) →
  (abs φ < π) →
  (∀ x, ∃ y, y = sin (2*(x + π/6) + φ)) →
  (∀ x, sin (2*(x + π/6) + φ) = -sin (2*(-x + π/6) + φ)) →
  (φ = -π/3 ∨ φ = 2*π/3) := by
sorry

end NUMINAMATH_CALUDE_sin_graph_shift_symmetry_l980_98006


namespace NUMINAMATH_CALUDE_entrance_exam_score_l980_98074

theorem entrance_exam_score (total_questions : ℕ) 
  (correct_score incorrect_score unattempted_score : ℤ) 
  (total_score : ℤ) :
  total_questions = 70 ∧ 
  correct_score = 3 ∧ 
  incorrect_score = -1 ∧ 
  unattempted_score = -2 ∧
  total_score = 38 →
  ∃ (correct incorrect : ℕ),
    correct + incorrect = total_questions ∧
    correct_score * correct + incorrect_score * incorrect = total_score ∧
    correct = 27 ∧
    incorrect = 43 := by
  sorry

end NUMINAMATH_CALUDE_entrance_exam_score_l980_98074


namespace NUMINAMATH_CALUDE_second_expression_value_l980_98042

theorem second_expression_value (a x : ℝ) (h1 : ((2 * a + 16) + x) / 2 = 69) (h2 : a = 26) : x = 70 := by
  sorry

end NUMINAMATH_CALUDE_second_expression_value_l980_98042


namespace NUMINAMATH_CALUDE_square_root_equation_l980_98066

theorem square_root_equation (x : ℝ) : (x + 1)^2 = 9 → x = 2 ∨ x = -4 := by
  sorry

end NUMINAMATH_CALUDE_square_root_equation_l980_98066


namespace NUMINAMATH_CALUDE_solve_for_a_l980_98018

theorem solve_for_a : ∃ a : ℝ, 
  (∀ x y : ℝ, x = 1 ∧ y = 2 → 3 * x - a * y = 1) → 
  a = 1 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l980_98018


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l980_98021

/-- A geometric sequence of positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_property (a : ℕ → ℝ) 
  (h_geo : GeometricSequence a) 
  (h_positive : ∀ n, a n > 0)
  (h_sum : a 2 * a 8 + a 3 * a 7 = 32) : 
  a 5 = 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l980_98021


namespace NUMINAMATH_CALUDE_all_cells_equal_l980_98054

/-- Represents an infinite grid of natural numbers -/
def Grid := ℤ → ℤ → ℕ

/-- The condition that each cell's value is greater than or equal to the arithmetic mean of its four neighboring cells -/
def ValidGrid (g : Grid) : Prop :=
  ∀ i j : ℤ, g i j ≥ (g (i-1) j + g (i+1) j + g i (j-1) + g i (j+1)) / 4

/-- The theorem stating that all cells in a valid grid must contain the same number -/
theorem all_cells_equal (g : Grid) (h : ValidGrid g) : 
  ∀ i j k l : ℤ, g i j = g k l :=
sorry

end NUMINAMATH_CALUDE_all_cells_equal_l980_98054


namespace NUMINAMATH_CALUDE_quadratic_inequalities_l980_98071

/-- Given a quadratic inequality and its solution set, prove the value of the coefficient and the solution set of a related inequality -/
theorem quadratic_inequalities (a : ℝ) :
  (∀ x : ℝ, (a * x^2 + 3 * x - 1 > 0) ↔ (1/2 < x ∧ x < 1)) →
  (a = -2 ∧ 
   ∀ x : ℝ, (a * x^2 - 3 * x + a^2 + 1 > 0) ↔ (-5/2 < x ∧ x < 1)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequalities_l980_98071


namespace NUMINAMATH_CALUDE_smallest_n_for_inequality_l980_98046

theorem smallest_n_for_inequality : ∃ (n : ℕ), n = 4 ∧ 
  (∀ (x y z w : ℝ), (x^2 + y^2 + z^2 + w^2)^2 ≤ n * (x^4 + y^4 + z^4 + w^4)) ∧
  (∀ (m : ℕ), m < n → ∃ (x y z w : ℝ), (x^2 + y^2 + z^2 + w^2)^2 > m * (x^4 + y^4 + z^4 + w^4)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_inequality_l980_98046


namespace NUMINAMATH_CALUDE_f_properties_l980_98016

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.log (1 + x)

theorem f_properties :
  (∀ x : ℝ, f x > 0 ↔ x > 0) ∧
  (∀ s t : ℝ, s > 0 → t > 0 → f (s + t) > f s + f t) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l980_98016


namespace NUMINAMATH_CALUDE_boys_to_girls_ratio_l980_98031

/-- Given a class with 100 students where there are 20 more boys than girls,
    prove that the ratio of boys to girls is 3:2. -/
theorem boys_to_girls_ratio (total : ℕ) (difference : ℕ) : 
  total = 100 → difference = 20 → 
  ∃ (boys girls : ℕ), 
    boys + girls = total ∧ 
    boys = girls + difference ∧
    boys / girls = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_boys_to_girls_ratio_l980_98031


namespace NUMINAMATH_CALUDE_no_real_solutions_to_equation_l980_98098

theorem no_real_solutions_to_equation : 
  ¬ ∃ (x : ℝ), x > 0 ∧ x^(Real.log x / Real.log 10) = x^3 / 1000 :=
sorry

end NUMINAMATH_CALUDE_no_real_solutions_to_equation_l980_98098


namespace NUMINAMATH_CALUDE_polygon_sides_count_polygon_has_2023_sides_l980_98094

/-- A polygon with the property that at most 2021 triangles can be formed
    when a diagonal is drawn from a vertex has 2023 sides. -/
theorem polygon_sides_count : ℕ :=
  2023

/-- The maximum number of triangles formed when drawing a diagonal from a vertex
    of a polygon with n sides is n - 2. -/
def max_triangles (n : ℕ) : ℕ := n - 2

/-- The condition that at most 2021 triangles can be formed. -/
axiom triangle_condition : max_triangles polygon_sides_count ≤ 2021

/-- Theorem stating that the polygon has 2023 sides. -/
theorem polygon_has_2023_sides : polygon_sides_count = 2023 := by
  sorry

#check polygon_has_2023_sides

end NUMINAMATH_CALUDE_polygon_sides_count_polygon_has_2023_sides_l980_98094


namespace NUMINAMATH_CALUDE_problem_statement_l980_98060

theorem problem_statement (x y : ℝ) (h : -x + 2*y = 5) : 
  5*(x - 2*y)^2 - 3*(x - 2*y) - 60 = 80 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l980_98060


namespace NUMINAMATH_CALUDE_tom_catches_jerry_l980_98008

/-- The time it takes for Tom to catch Jerry in the given scenario --/
def catch_time : ℝ → Prop := λ t =>
  let rectangle_width : ℝ := 15
  let rectangle_length : ℝ := 30
  let tom_speed : ℝ := 5
  let jerry_speed : ℝ := 3
  16 * t^2 - 45 * Real.sqrt 2 * t - 225 = 0

theorem tom_catches_jerry : ∃ t : ℝ, catch_time t := by sorry

end NUMINAMATH_CALUDE_tom_catches_jerry_l980_98008


namespace NUMINAMATH_CALUDE_correlation_coefficient_measures_linear_relationship_l980_98084

/-- The correlation coefficient is a statistical measure. -/
def correlation_coefficient : Type := sorry

/-- A measure of the strength of a linear relationship between two variables. -/
def linear_relationship_strength : Type := sorry

/-- The correlation coefficient measures the strength of the linear relationship between two variables. -/
theorem correlation_coefficient_measures_linear_relationship :
  correlation_coefficient → linear_relationship_strength :=
sorry

end NUMINAMATH_CALUDE_correlation_coefficient_measures_linear_relationship_l980_98084


namespace NUMINAMATH_CALUDE_vehicle_distance_after_three_minutes_l980_98013

/-- The distance between two vehicles after a given time, given their speeds -/
def distance_between (speed1 speed2 : ℝ) (time : ℝ) : ℝ :=
  (speed2 - speed1) * time

theorem vehicle_distance_after_three_minutes :
  let truck_speed : ℝ := 65
  let car_speed : ℝ := 85
  let time_in_hours : ℝ := 3 / 60
  distance_between truck_speed car_speed time_in_hours = 1 := by
  sorry

end NUMINAMATH_CALUDE_vehicle_distance_after_three_minutes_l980_98013


namespace NUMINAMATH_CALUDE_least_of_four_consecutive_integers_with_sum_two_l980_98058

theorem least_of_four_consecutive_integers_with_sum_two :
  ∀ n : ℤ, (n + (n + 1) + (n + 2) + (n + 3) = 2) → n = -1 :=
by
  sorry

end NUMINAMATH_CALUDE_least_of_four_consecutive_integers_with_sum_two_l980_98058


namespace NUMINAMATH_CALUDE_max_area_is_one_l980_98000

/-- A right triangle with legs 3 and 4, and hypotenuse 5 -/
structure RightTriangle where
  leg1 : ℝ
  leg2 : ℝ
  hypotenuse : ℝ
  is_right_triangle : leg1^2 + leg2^2 = hypotenuse^2
  leg1_is_3 : leg1 = 3
  leg2_is_4 : leg2 = 4
  hypotenuse_is_5 : hypotenuse = 5

/-- A rectangle inscribed in the right triangle with one side along the hypotenuse -/
structure InscribedRectangle (t : RightTriangle) where
  base : ℝ  -- Length of the rectangle's side along the hypotenuse
  height : ℝ -- Height of the rectangle
  is_inscribed : height ≤ t.leg2 * (1 - base / t.hypotenuse)
  on_hypotenuse : base ≤ t.hypotenuse

/-- The area of an inscribed rectangle -/
def area (t : RightTriangle) (r : InscribedRectangle t) : ℝ :=
  r.base * r.height

/-- The maximum area of an inscribed rectangle is 1 -/
theorem max_area_is_one (t : RightTriangle) : 
  ∃ (r : InscribedRectangle t), ∀ (r' : InscribedRectangle t), area t r ≥ area t r' ∧ area t r = 1 :=
sorry

end NUMINAMATH_CALUDE_max_area_is_one_l980_98000


namespace NUMINAMATH_CALUDE_a_range_l980_98081

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * log x - (1/2) * x^2

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a x + 4*x

theorem a_range (a : ℝ) :
  (∃ x_0 : ℝ, x_0 > 0 ∧ IsLocalMin (g a) x_0) →
  (∃ x_0 : ℝ, x_0 > 0 ∧ IsLocalMin (g a) x_0 ∧ g a x_0 - (1/2) * x_0^2 + 2*a > 0) →
  a ∈ Set.Ioo (-4/ℯ + 1/ℯ^2) 0 :=
by sorry

end NUMINAMATH_CALUDE_a_range_l980_98081


namespace NUMINAMATH_CALUDE_a_value_equation_solution_l980_98073

-- Define the positive number whose square root is both a+6 and 2a-9
def positive_number (a : ℝ) : Prop := ∃ n : ℝ, n > 0 ∧ (a + 6 = Real.sqrt n) ∧ (2*a - 9 = Real.sqrt n)

-- Theorem 1: Prove that a = 15
theorem a_value (a : ℝ) (h : positive_number a) : a = 15 := by sorry

-- Theorem 2: Prove that the solution to ax³-64=0 is x = 4 when a = 15
theorem equation_solution (x : ℝ) : 15 * x^3 - 64 = 0 ↔ x = 4 := by sorry

end NUMINAMATH_CALUDE_a_value_equation_solution_l980_98073


namespace NUMINAMATH_CALUDE_yellow_curlers_count_l980_98004

/-- Given the total number of curlers and the proportions of different types,
    prove that the number of extra-large yellow curlers is 18. -/
theorem yellow_curlers_count (total : ℕ) (pink : ℕ) (blue : ℕ) (green : ℕ) (yellow : ℕ) : 
  total = 120 →
  pink = total / 5 →
  blue = 2 * pink →
  green = total / 4 →
  yellow = total - pink - blue - green →
  yellow = 18 := by
sorry

end NUMINAMATH_CALUDE_yellow_curlers_count_l980_98004


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_attained_l980_98028

theorem min_value_expression (x : ℝ) (h1 : 0 < x) (h2 : x < 1) :
  (1 / x + 2 * x / (1 - x)) ≥ 1 + 2 * Real.sqrt 2 := by
  sorry

theorem min_value_attained (x : ℝ) (h1 : 0 < x) (h2 : x < 1) :
  ∃ x₀, 0 < x₀ ∧ x₀ < 1 ∧ (1 / x₀ + 2 * x₀ / (1 - x₀)) = 1 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_attained_l980_98028


namespace NUMINAMATH_CALUDE_cat_dog_food_difference_l980_98047

theorem cat_dog_food_difference :
  let cat_packages : ℕ := 6
  let dog_packages : ℕ := 2
  let cans_per_cat_package : ℕ := 9
  let cans_per_dog_package : ℕ := 3
  let total_cat_cans := cat_packages * cans_per_cat_package
  let total_dog_cans := dog_packages * cans_per_dog_package
  total_cat_cans - total_dog_cans = 48 :=
by sorry

end NUMINAMATH_CALUDE_cat_dog_food_difference_l980_98047


namespace NUMINAMATH_CALUDE_gala_handshakes_l980_98086

/-- Number of married couples at the gala -/
def num_couples : ℕ := 15

/-- Total number of people at the gala -/
def total_people : ℕ := 2 * num_couples

/-- Number of handshakes between men -/
def handshakes_men : ℕ := num_couples.choose 2

/-- Number of handshakes between men and women -/
def handshakes_men_women : ℕ := num_couples * num_couples

/-- Total number of handshakes at the gala -/
def total_handshakes : ℕ := handshakes_men + handshakes_men_women

theorem gala_handshakes : total_handshakes = 330 := by
  sorry

end NUMINAMATH_CALUDE_gala_handshakes_l980_98086


namespace NUMINAMATH_CALUDE_bread_distribution_l980_98036

theorem bread_distribution (a d : ℚ) : 
  d > 0 ∧ 
  (a - 2*d) + (a - d) + a + (a + d) + (a + 2*d) = 100 ∧ 
  (a + (a + d) + (a + 2*d)) = (1/7) * ((a - 2*d) + (a - d)) →
  a - 2*d = 5/3 := by
sorry

end NUMINAMATH_CALUDE_bread_distribution_l980_98036


namespace NUMINAMATH_CALUDE_product_of_difference_and_sum_of_squares_l980_98065

theorem product_of_difference_and_sum_of_squares (a b : ℝ) 
  (h1 : a - b = 3) 
  (h2 : a^2 + b^2 = 39) : 
  a * b = 15 := by
  sorry

end NUMINAMATH_CALUDE_product_of_difference_and_sum_of_squares_l980_98065


namespace NUMINAMATH_CALUDE_two_reciprocal_sets_l980_98090

-- Define a reciprocal set
def ReciprocalSet (A : Set ℝ) : Prop :=
  A.Nonempty ∧ (0 ∉ A) ∧ ∀ x ∈ A, (1 / x) ∈ A

-- Define the three sets
def Set1 (a : ℝ) : Set ℝ := {x : ℝ | x^2 + a*x + 1 = 0}

def Set2 : Set ℝ := {x : ℝ | x^2 - 4*x + 1 < 0}

def Set3 : Set ℝ := {y : ℝ | ∃ x : ℝ, 
  (0 ≤ x ∧ x < 1 ∧ y = 2*x + 2/5) ∨ 
  (1 ≤ x ∧ x ≤ 2 ∧ y = x + 1/x)}

-- Theorem to prove
theorem two_reciprocal_sets : 
  ∃ (a : ℝ), (ReciprocalSet (Set2) ∧ ReciprocalSet (Set3) ∧ ¬ReciprocalSet (Set1 a)) ∨
             (ReciprocalSet (Set1 a) ∧ ReciprocalSet (Set2) ∧ ¬ReciprocalSet (Set3)) ∨
             (ReciprocalSet (Set1 a) ∧ ReciprocalSet (Set3) ∧ ¬ReciprocalSet (Set2)) :=
sorry

end NUMINAMATH_CALUDE_two_reciprocal_sets_l980_98090


namespace NUMINAMATH_CALUDE_honey_harvest_increase_l980_98062

/-- Proves that the increase in honey harvest is 6085 pounds -/
theorem honey_harvest_increase 
  (last_year_harvest : ℕ) 
  (this_year_harvest : ℕ) 
  (h1 : last_year_harvest = 2479)
  (h2 : this_year_harvest = 8564) : 
  this_year_harvest - last_year_harvest = 6085 := by
  sorry

end NUMINAMATH_CALUDE_honey_harvest_increase_l980_98062


namespace NUMINAMATH_CALUDE_infinite_geometric_series_ratio_l980_98061

/-- For an infinite geometric series with first term a and sum S,
    the common ratio r can be calculated. -/
theorem infinite_geometric_series_ratio 
  (a : ℝ) (S : ℝ) (h1 : a = 512) (h2 : S = 3072) :
  ∃ r : ℝ, r = 5 / 6 ∧ S = a / (1 - r) := by
sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_ratio_l980_98061


namespace NUMINAMATH_CALUDE_smallest_number_divisibility_l980_98027

theorem smallest_number_divisibility (n : ℕ) : 
  (∀ m : ℕ, m < n → ¬(((m + 1) % 12 = 0) ∧ ((m + 1) % 18 = 0) ∧ ((m + 1) % 24 = 0) ∧ ((m + 1) % 32 = 0) ∧ ((m + 1) % 40 = 0))) ∧
  ((n + 1) % 12 = 0) ∧ ((n + 1) % 18 = 0) ∧ ((n + 1) % 24 = 0) ∧ ((n + 1) % 32 = 0) ∧ ((n + 1) % 40 = 0) →
  n = 2879 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_divisibility_l980_98027


namespace NUMINAMATH_CALUDE_nancy_savings_l980_98059

-- Define the value of a dozen
def dozen : ℕ := 12

-- Define the value of a quarter in cents
def quarter_value : ℕ := 25

-- Define the number of cents in a dollar
def cents_per_dollar : ℕ := 100

-- Theorem statement
theorem nancy_savings (quarters : ℕ) : 
  quarters = dozen → (quarters * quarter_value) / cents_per_dollar = 3 := by
  sorry

end NUMINAMATH_CALUDE_nancy_savings_l980_98059


namespace NUMINAMATH_CALUDE_least_sum_of_bases_l980_98089

/-- Represent a number in a given base --/
def baseRepresentation (n : ℕ) (base : ℕ) : ℕ := 
  (n / base) * base + (n % base)

/-- The problem statement --/
theorem least_sum_of_bases : 
  ∃ (c d : ℕ), c > 0 ∧ d > 0 ∧ 
  baseRepresentation 58 c = baseRepresentation 85 d ∧
  (∀ (c' d' : ℕ), c' > 0 → d' > 0 → 
    baseRepresentation 58 c' = baseRepresentation 85 d' → 
    c + d ≤ c' + d') ∧
  c + d = 15 :=
sorry

end NUMINAMATH_CALUDE_least_sum_of_bases_l980_98089


namespace NUMINAMATH_CALUDE_x_plus_y_equals_fifteen_l980_98051

theorem x_plus_y_equals_fifteen (x y : ℝ) 
  (h1 : (3 : ℝ)^x = 27^(y + 1)) 
  (h2 : (16 : ℝ)^y = 4^(x - 6)) : 
  x + y = 15 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_y_equals_fifteen_l980_98051


namespace NUMINAMATH_CALUDE_greatest_integer_solution_greatest_integer_value_l980_98007

theorem greatest_integer_solution (x : ℤ) : (5 - 4*x > 17) ↔ (x < -3) :=
  sorry

theorem greatest_integer_value : ∃ (x : ℤ), (∀ (y : ℤ), (5 - 4*y > 17) → y ≤ x) ∧ (5 - 4*x > 17) ∧ x = -4 :=
  sorry

end NUMINAMATH_CALUDE_greatest_integer_solution_greatest_integer_value_l980_98007


namespace NUMINAMATH_CALUDE_catering_weight_calculation_mason_catering_weight_l980_98067

/-- Calculates the total weight of silverware and plates for a catering event. -/
theorem catering_weight_calculation (silverware_weight plate_weight : ℕ)
  (silverware_per_setting plates_per_setting : ℕ)
  (tables settings_per_table backup_settings : ℕ) : ℕ :=
  let total_settings := tables * settings_per_table + backup_settings
  let weight_per_setting := silverware_per_setting * silverware_weight + plates_per_setting * plate_weight
  total_settings * weight_per_setting

/-- Proves that the total weight of all settings for Mason's catering event is 5040 ounces. -/
theorem mason_catering_weight :
  catering_weight_calculation 4 12 3 2 15 8 20 = 5040 := by
  sorry

end NUMINAMATH_CALUDE_catering_weight_calculation_mason_catering_weight_l980_98067


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l980_98048

theorem min_reciprocal_sum (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 2*y = 3) :
  (1/x + 1/y) ≥ 1 + (2*Real.sqrt 2)/3 ∧ 
  ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 2*y₀ = 3 ∧ 1/x₀ + 1/y₀ = 1 + (2*Real.sqrt 2)/3 :=
sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l980_98048


namespace NUMINAMATH_CALUDE_intersection_M_N_l980_98087

def M : Set ℝ := {-1, 0, 1, 2, 3}
def N : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 3}

theorem intersection_M_N : M ∩ N = {-1, 0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l980_98087


namespace NUMINAMATH_CALUDE_det_B_squared_minus_3B_l980_98022

def B : Matrix (Fin 2) (Fin 2) ℝ := !![2, 4; 3, 2]

theorem det_B_squared_minus_3B : Matrix.det (B ^ 2 - 3 • B) = 88 := by
  sorry

end NUMINAMATH_CALUDE_det_B_squared_minus_3B_l980_98022


namespace NUMINAMATH_CALUDE_function_is_even_l980_98093

/-- A function satisfying certain properties is even -/
theorem function_is_even (f : ℝ → ℝ) 
  (h1 : ∀ x, f (x + 2) = f (2 - x))
  (h2 : ∀ x, f (1 + x) = -f x)
  (h3 : ¬ ∀ x y, f x = f y) : 
  ∀ x, f x = f (-x) := by
  sorry

end NUMINAMATH_CALUDE_function_is_even_l980_98093


namespace NUMINAMATH_CALUDE_most_advantageous_order_l980_98009

-- Define the probabilities
variable (p₁ p₂ p₃ : ℝ)

-- Define the conditions
variable (h₁ : 0 ≤ p₁ ∧ p₁ ≤ 1)
variable (h₂ : 0 ≤ p₂ ∧ p₂ ≤ 1)
variable (h₃ : 0 ≤ p₃ ∧ p₃ ≤ 1)
variable (h₄ : p₃ < p₁)
variable (h₅ : p₁ < p₂)

-- Define the probability of winning two games in a row with p₂ as the second opponent
def prob_p₂_second := p₂ * (p₁ + p₃ - p₁ * p₃)

-- Define the probability of winning two games in a row with p₁ as the second opponent
def prob_p₁_second := p₁ * (p₂ + p₃ - p₂ * p₃)

-- The theorem to prove
theorem most_advantageous_order :
  prob_p₂_second p₁ p₂ p₃ > prob_p₁_second p₁ p₂ p₃ :=
sorry

end NUMINAMATH_CALUDE_most_advantageous_order_l980_98009


namespace NUMINAMATH_CALUDE_painted_cubes_count_l980_98096

/-- Represents a cube with side length n --/
structure Cube (n : ℕ) where
  side_length : ℕ := n

/-- Represents a painted cube --/
structure PaintedCube (n : ℕ) extends Cube n where
  painted_faces : ℕ := 6

/-- Represents a cube that has been cut into smaller cubes --/
structure CutCube (n m : ℕ) extends PaintedCube n where
  cut_size : ℕ := m

/-- The number of smaller cubes with at least two painted faces in a cut painted cube --/
def cubes_with_two_plus_painted_faces (c : CutCube 4 1) : ℕ := 32

/-- Theorem stating that a 4x4x4 painted cube cut into 1x1x1 cubes has 32 smaller cubes with at least two painted faces --/
theorem painted_cubes_count (c : CutCube 4 1) : 
  cubes_with_two_plus_painted_faces c = 32 := by sorry

end NUMINAMATH_CALUDE_painted_cubes_count_l980_98096


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l980_98050

-- Define repeating decimals
def repeating_234 : ℚ := 234 / 999
def repeating_567 : ℚ := 567 / 999
def repeating_891 : ℚ := 891 / 999

-- State the theorem
theorem repeating_decimal_sum : 
  repeating_234 - repeating_567 + repeating_891 = 186 / 333 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l980_98050


namespace NUMINAMATH_CALUDE_m_range_l980_98095

/-- The statement "The equation x^2 + 2x + m = 0 has no real roots" -/
def p (m : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*x + m ≠ 0

/-- The statement "The equation x^2/(m-1) + y^2 = 1 is an ellipse with foci on the x-axis" -/
def q (m : ℝ) : Prop := m > 2 ∧ ∀ x y : ℝ, x^2/(m-1) + y^2 = 1 → ∃ c : ℝ, c^2 = m - 1

theorem m_range (m : ℝ) : (¬(¬(p m)) ∧ ¬(p m ∧ q m)) → (1 < m ∧ m ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_m_range_l980_98095


namespace NUMINAMATH_CALUDE_imaginary_part_product_l980_98052

theorem imaginary_part_product : Complex.im ((1 + Complex.I) * (3 - Complex.I)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_product_l980_98052


namespace NUMINAMATH_CALUDE_b_share_is_seven_fifteenths_l980_98001

/-- A partnership with four partners A, B, C, and D -/
structure Partnership where
  total_capital : ℝ
  a_share : ℝ
  b_share : ℝ
  c_share : ℝ
  d_share : ℝ
  total_profit : ℝ
  a_profit : ℝ

/-- The conditions of the partnership -/
def partnership_conditions (p : Partnership) : Prop :=
  p.a_share = (1/3) * p.total_capital ∧
  p.c_share = (1/5) * p.total_capital ∧
  p.d_share = p.total_capital - (p.a_share + p.b_share + p.c_share) ∧
  p.total_profit = 2430 ∧
  p.a_profit = 810

/-- Theorem stating B's share of the capital -/
theorem b_share_is_seven_fifteenths (p : Partnership) 
  (h : partnership_conditions p) : 
  p.b_share = (7/15) * p.total_capital := by
  sorry

end NUMINAMATH_CALUDE_b_share_is_seven_fifteenths_l980_98001


namespace NUMINAMATH_CALUDE_orchard_sections_l980_98085

/-- Given the daily harvest from each orchard section and the total daily harvest,
    calculate the number of orchard sections. -/
theorem orchard_sections 
  (sacks_per_section : ℕ) 
  (total_sacks : ℕ) 
  (h1 : sacks_per_section = 45)
  (h2 : total_sacks = 360) :
  total_sacks / sacks_per_section = 8 := by
  sorry

end NUMINAMATH_CALUDE_orchard_sections_l980_98085


namespace NUMINAMATH_CALUDE_overlap_difference_l980_98034

/-- Represents the student population --/
def StudentPopulation : Set ℕ := {n : ℕ | 1000 ≤ n ∧ n ≤ 1200}

/-- Represents the number of students studying German --/
def GermanStudents (n : ℕ) : Set ℕ := {g : ℕ | (70 * n + 99) / 100 ≤ g ∧ g ≤ (75 * n) / 100}

/-- Represents the number of students studying Russian --/
def RussianStudents (n : ℕ) : Set ℕ := {r : ℕ | (35 * n + 99) / 100 ≤ r ∧ r ≤ (45 * n) / 100}

/-- The minimum number of students studying both languages --/
def m (n : ℕ) (g : ℕ) (r : ℕ) : ℕ := g + r - n

/-- The maximum number of students studying both languages --/
def M (n : ℕ) (g : ℕ) (r : ℕ) : ℕ := min g r

/-- Main theorem --/
theorem overlap_difference (n : StudentPopulation) 
  (g : GermanStudents n) (r : RussianStudents n) : 
  ∃ (m_val : ℕ) (M_val : ℕ), 
    m_val = m n g r ∧ 
    M_val = M n g r ∧ 
    M_val - m_val = 190 := by
  sorry

end NUMINAMATH_CALUDE_overlap_difference_l980_98034


namespace NUMINAMATH_CALUDE_plumber_salary_percentage_l980_98015

-- Define the daily salaries and total labor cost
def construction_worker_salary : ℝ := 100
def electrician_salary : ℝ := 2 * construction_worker_salary
def total_labor_cost : ℝ := 650

-- Define the number of workers
def num_construction_workers : ℕ := 2
def num_electricians : ℕ := 1
def num_plumbers : ℕ := 1

-- Calculate the plumber's salary
def plumber_salary : ℝ :=
  total_labor_cost - (num_construction_workers * construction_worker_salary + num_electricians * electrician_salary)

-- Define the theorem
theorem plumber_salary_percentage :
  plumber_salary / construction_worker_salary * 100 = 250 := by
  sorry


end NUMINAMATH_CALUDE_plumber_salary_percentage_l980_98015


namespace NUMINAMATH_CALUDE_nested_fraction_evaluation_l980_98055

theorem nested_fraction_evaluation :
  1 + 3 / (4 + 5 / (6 + 7/8)) = 85/52 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_evaluation_l980_98055


namespace NUMINAMATH_CALUDE_infinitely_many_rationals_between_one_sixth_and_five_sixths_l980_98017

theorem infinitely_many_rationals_between_one_sixth_and_five_sixths :
  ∃ (S : Set ℚ), Set.Infinite S ∧ ∀ q ∈ S, 1/6 < q ∧ q < 5/6 :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_rationals_between_one_sixth_and_five_sixths_l980_98017


namespace NUMINAMATH_CALUDE_zoo_viewing_time_is_75_minutes_l980_98038

/-- Calculates the total viewing time for a zoo visit -/
def total_zoo_viewing_time (original_times new_times : List ℕ) (break_time : ℕ) : ℕ :=
  let total_viewing_time := original_times.sum + new_times.sum
  let total_break_time := break_time * (original_times.length + new_times.length - 1)
  total_viewing_time + total_break_time

/-- Theorem: The total time required to see all 9 animal types is 75 minutes -/
theorem zoo_viewing_time_is_75_minutes :
  total_zoo_viewing_time [4, 6, 7, 5, 9] [3, 7, 8, 10] 2 = 75 := by
  sorry

end NUMINAMATH_CALUDE_zoo_viewing_time_is_75_minutes_l980_98038


namespace NUMINAMATH_CALUDE_difference_of_squares_l980_98040

theorem difference_of_squares (x y : ℝ) : (y + x) * (y - x) = y^2 - x^2 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l980_98040


namespace NUMINAMATH_CALUDE_problem_solving_probability_l980_98099

/-- The probability that Alex, Kyle, and Catherine solve a problem, but not Bella and David -/
theorem problem_solving_probability 
  (p_alex : ℚ) (p_bella : ℚ) (p_kyle : ℚ) (p_david : ℚ) (p_catherine : ℚ)
  (h_alex : p_alex = 1/4)
  (h_bella : p_bella = 3/5)
  (h_kyle : p_kyle = 1/3)
  (h_david : p_david = 2/7)
  (h_catherine : p_catherine = 5/9) :
  p_alex * p_kyle * p_catherine * (1 - p_bella) * (1 - p_david) = 25/378 := by
sorry

end NUMINAMATH_CALUDE_problem_solving_probability_l980_98099


namespace NUMINAMATH_CALUDE_arrangements_with_separation_l980_98072

/-- The number of ways to arrange 5 people in a line. -/
def total_arrangements : ℕ := 120

/-- The number of ways to arrange 5 people in a line with A and B adjacent. -/
def adjacent_arrangements : ℕ := 48

/-- The number of people in the line. -/
def num_people : ℕ := 5

/-- Theorem: The number of ways to arrange 5 people in a line with at least one person between A and B is 72. -/
theorem arrangements_with_separation :
  total_arrangements - adjacent_arrangements = 72 :=
sorry

end NUMINAMATH_CALUDE_arrangements_with_separation_l980_98072


namespace NUMINAMATH_CALUDE_prime_expressions_solution_l980_98005

def f (n : ℤ) : ℤ := |n^3 - 4*n^2 + 3*n - 35|
def g (n : ℤ) : ℤ := |n^2 + 4*n + 8|

theorem prime_expressions_solution :
  {n : ℤ | Nat.Prime (f n).natAbs ∧ Nat.Prime (g n).natAbs} = {-3, -1, 5} := by
sorry

end NUMINAMATH_CALUDE_prime_expressions_solution_l980_98005


namespace NUMINAMATH_CALUDE_angle_B_measure_l980_98097

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : Point)

-- Define the measure of an angle in a quadrilateral
def angle_measure (q : Quadrilateral) (v : Fin 4) : ℝ := sorry

-- Theorem statement
theorem angle_B_measure (q : Quadrilateral) :
  angle_measure q 0 + angle_measure q 2 = 100 →
  angle_measure q 1 = 130 := by
  sorry

end NUMINAMATH_CALUDE_angle_B_measure_l980_98097


namespace NUMINAMATH_CALUDE_system_solution_l980_98057

-- Define the two equations
def equation1 (x y : ℝ) : Prop :=
  8 * x^2 - 26 * x * y + 15 * y^2 + 116 * x - 150 * y + 360 = 0

def equation2 (x y : ℝ) : Prop :=
  8 * x^2 + 18 * x * y - 18 * y^2 + 60 * x + 45 * y + 108 = 0

-- Define the solution set
def solutions : Set (ℝ × ℝ) :=
  {(0, 4), (-7.5, 1), (-4.5, 0)}

-- Theorem statement
theorem system_solution :
  ∀ x y : ℝ, (equation1 x y ∧ equation2 x y) ↔ (x, y) ∈ solutions := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l980_98057


namespace NUMINAMATH_CALUDE_problem_statement_l980_98014

theorem problem_statement (x : ℝ) (h : x^2 + 8 * (x / (x - 3))^2 = 53) :
  ((x - 3)^3 * (x + 4)) / (2 * x - 5) = 17000 / 21 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l980_98014


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l980_98035

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

-- Define the conditions of the problem
def problem_conditions (a : ℕ → ℝ) (n : ℕ) : Prop :=
  geometric_sequence a ∧
  a 1 * a 2 * a 3 = 4 ∧
  a 4 * a 5 * a 6 = 12 ∧
  a (n - 1) * a n * a (n + 1) = 324

-- Theorem statement
theorem geometric_sequence_problem (a : ℕ → ℝ) (n : ℕ) :
  problem_conditions a n → n = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l980_98035


namespace NUMINAMATH_CALUDE_circle_equation_with_diameter_mn_l980_98033

/-- Given points M (1, -1) and N (-1, 1), prove that the equation of the circle with diameter MN is x² + y² = 2 -/
theorem circle_equation_with_diameter_mn (x y : ℝ) : 
  let m : ℝ × ℝ := (1, -1)
  let n : ℝ × ℝ := (-1, 1)
  let center : ℝ × ℝ := ((m.1 + n.1) / 2, (m.2 + n.2) / 2)
  let radius : ℝ := Real.sqrt ((m.1 - n.1)^2 + (m.2 - n.2)^2) / 2
  (x - center.1)^2 + (y - center.2)^2 = radius^2 ↔ x^2 + y^2 = 2 :=
by sorry


end NUMINAMATH_CALUDE_circle_equation_with_diameter_mn_l980_98033


namespace NUMINAMATH_CALUDE_binomial_expansion_example_l980_98041

theorem binomial_expansion_example : 102^4 - 4 * 102^3 + 6 * 102^2 - 4 * 102 + 1 = 104060401 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_example_l980_98041


namespace NUMINAMATH_CALUDE_store_posters_l980_98043

theorem store_posters (P : ℕ) : 
  (2 : ℚ) / 5 * P + (1 : ℚ) / 2 * P + 5 = P → P = 50 := by
  sorry

end NUMINAMATH_CALUDE_store_posters_l980_98043


namespace NUMINAMATH_CALUDE_special_collection_loans_l980_98037

theorem special_collection_loans (initial_count : ℕ) (return_rate : ℚ) (final_count : ℕ) 
  (h1 : initial_count = 75)
  (h2 : return_rate = 70 / 100)
  (h3 : final_count = 60) :
  ∃ (loaned_out : ℕ), loaned_out = 50 ∧ 
    initial_count - (1 - return_rate) * loaned_out = final_count :=
by sorry

end NUMINAMATH_CALUDE_special_collection_loans_l980_98037


namespace NUMINAMATH_CALUDE_complex_pure_imaginary_m_l980_98025

/-- A complex number is pure imaginary if its real part is zero and its imaginary part is nonzero -/
def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem complex_pure_imaginary_m (m : ℝ) : 
  is_pure_imaginary ((m^2 - 5*m + 6 : ℝ) + (m^2 - 3*m : ℝ) * I) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_pure_imaginary_m_l980_98025


namespace NUMINAMATH_CALUDE_present_cost_l980_98039

/-- Proves that the total amount paid for a present by 4 friends is $60, given specific conditions. -/
theorem present_cost (initial_contribution : ℝ) : 
  (4 : ℝ) > 0 → 
  0 < initial_contribution → 
  0.75 * (4 * initial_contribution) = 4 * (initial_contribution - 5) → 
  0.75 * (4 * initial_contribution) = 60 := by
  sorry

end NUMINAMATH_CALUDE_present_cost_l980_98039


namespace NUMINAMATH_CALUDE_hash_four_six_l980_98044

-- Define the operation #
def hash (x y : ℝ) : ℝ := 4 * x - 2 * y

-- Theorem statement
theorem hash_four_six : hash 4 6 = 4 := by
  sorry

end NUMINAMATH_CALUDE_hash_four_six_l980_98044


namespace NUMINAMATH_CALUDE_walking_problem_solution_l980_98045

/-- Two people walking in opposite directions --/
structure WalkingProblem where
  time : ℝ
  distance : ℝ
  speed1 : ℝ
  speed2 : ℝ

/-- The conditions of our specific problem --/
def problem : WalkingProblem where
  time := 5
  distance := 75
  speed1 := 10
  speed2 := 5 -- This is what we want to prove

theorem walking_problem_solution (p : WalkingProblem) 
  (h1 : p.time * (p.speed1 + p.speed2) = p.distance)
  (h2 : p.time = 5)
  (h3 : p.distance = 75)
  (h4 : p.speed1 = 10) :
  p.speed2 = 5 := by
  sorry

#check walking_problem_solution problem

end NUMINAMATH_CALUDE_walking_problem_solution_l980_98045


namespace NUMINAMATH_CALUDE_number_composition_l980_98029

def place_value (digit : ℕ) (place : ℕ) : ℕ :=
  digit * (10 ^ place)

theorem number_composition :
  let tens_of_millions := place_value 4 7
  let hundreds_of_thousands := place_value 6 5
  let hundreds := place_value 5 2
  tens_of_millions + hundreds_of_thousands + hundreds = 46000500 := by
  sorry

end NUMINAMATH_CALUDE_number_composition_l980_98029


namespace NUMINAMATH_CALUDE_power_of_ten_square_l980_98069

theorem power_of_ten_square (k : ℕ) (N : ℕ) : 
  (10^(k-1) ≤ N) ∧ (N < 10^k) ∧ 
  (∃ m : ℕ, N^2 = N * 10^k + m ∧ m < N * 10^k) → 
  N = 10^(k-1) :=
by sorry

end NUMINAMATH_CALUDE_power_of_ten_square_l980_98069


namespace NUMINAMATH_CALUDE_doughnut_costs_9_l980_98024

/-- The price of a cake in Kč -/
def cake_price : ℕ := sorry

/-- The price of a doughnut in Kč -/
def doughnut_price : ℕ := sorry

/-- The amount of pocket money Honzík has in Kč -/
def pocket_money : ℕ := sorry

/-- Theorem stating the price of one doughnut is 9 Kč -/
theorem doughnut_costs_9 
  (h1 : pocket_money - 4 * cake_price = 5)
  (h2 : 5 * cake_price - pocket_money = 6)
  (h3 : 2 * cake_price + 3 * doughnut_price = pocket_money) :
  doughnut_price = 9 := by sorry

end NUMINAMATH_CALUDE_doughnut_costs_9_l980_98024


namespace NUMINAMATH_CALUDE_largest_reciprocal_l980_98023

theorem largest_reciprocal (a b c d e : ℚ) : 
  a = -1/4 → b = 2/7 → c = -2 → d = 3 → e = -3/2 → 
  (1/b > 1/a ∧ 1/b > 1/c ∧ 1/b > 1/d ∧ 1/b > 1/e) := by
  sorry

end NUMINAMATH_CALUDE_largest_reciprocal_l980_98023


namespace NUMINAMATH_CALUDE_largest_k_exists_l980_98063

def X : ℕ → ℤ
  | 0 => 1
  | 1 => 1
  | (n + 2) => X (n + 1) + 2 * X n

def Y : ℕ → ℤ
  | 0 => 1
  | 1 => 1
  | (n + 2) => 3 * Y (n + 1) + 4 * Y n

theorem largest_k_exists : ∃! k : ℕ, k < 10^2007 ∧
  (∃ i : ℕ+, |X i - k| ≤ 2007) ∧
  (∃ j : ℕ+, |Y j - k| ≤ 2007) ∧
  ∀ m : ℕ, m > k → ¬(
    (∃ i : ℕ+, |X i - m| ≤ 2007) ∧
    (∃ j : ℕ+, |Y j - m| ≤ 2007) ∧
    m < 10^2007
  ) :=
by sorry

end NUMINAMATH_CALUDE_largest_k_exists_l980_98063


namespace NUMINAMATH_CALUDE_daughters_and_granddaughters_without_children_l980_98056

/-- Represents Bertha's family structure -/
structure BerthaFamily where
  daughters : ℕ
  daughtersWithChildren : ℕ
  totalDescendants : ℕ

/-- The actual Bertha family configuration -/
def berthaActual : BerthaFamily :=
  { daughters := 8,
    daughtersWithChildren := 7,  -- This is derived, not given directly
    totalDescendants := 36 }

/-- Theorem stating the number of daughters and granddaughters without children -/
theorem daughters_and_granddaughters_without_children
  (b : BerthaFamily)
  (h1 : b.daughters = berthaActual.daughters)
  (h2 : b.totalDescendants = berthaActual.totalDescendants)
  (h3 : ∀ d, d ≤ b.daughters → (d = b.daughtersWithChildren ∨ d = b.daughters - b.daughtersWithChildren))
  (h4 : b.totalDescendants = b.daughters + 4 * b.daughtersWithChildren) :
  b.daughters - b.daughtersWithChildren + (b.totalDescendants - b.daughters) = 29 := by
  sorry

end NUMINAMATH_CALUDE_daughters_and_granddaughters_without_children_l980_98056


namespace NUMINAMATH_CALUDE_tarts_distribution_l980_98091

/-- Represents the number of tarts eaten by a child in a 10-minute interval -/
structure EatingRate :=
  (tarts : ℕ)

/-- Represents the total eating time in minutes -/
def total_time : ℕ := 90

/-- Represents the total number of tarts eaten -/
def total_tarts : ℕ := 35

/-- Zhenya's eating rate -/
def zhenya_rate : EatingRate := ⟨5⟩

/-- Sasha's eating rate -/
def sasha_rate : EatingRate := ⟨3⟩

/-- Calculates the number of tarts eaten by a child given their eating rate and number of 10-minute intervals -/
def tarts_eaten (rate : EatingRate) (intervals : ℕ) : ℕ := rate.tarts * intervals

/-- The main theorem to prove -/
theorem tarts_distribution :
  ∃ (zhenya_intervals sasha_intervals : ℕ),
    zhenya_intervals + sasha_intervals = total_time / 10 ∧
    tarts_eaten zhenya_rate zhenya_intervals + tarts_eaten sasha_rate sasha_intervals = total_tarts ∧
    tarts_eaten zhenya_rate zhenya_intervals = 20 ∧
    tarts_eaten sasha_rate sasha_intervals = 15 :=
sorry

end NUMINAMATH_CALUDE_tarts_distribution_l980_98091


namespace NUMINAMATH_CALUDE_quadratic_root_implies_k_l980_98068

theorem quadratic_root_implies_k (k : ℝ) : 
  ((k - 3) * (-1)^2 + 6 * (-1) + k^2 - k = 0) → k = -3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_k_l980_98068


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l980_98082

/-- A geometric sequence with common ratio 2 and sum of first 3 terms equal to 21 -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a (n + 1) = 2 * a n) ∧ (a 1 + a 2 + a 3 = 21)

/-- The sum of the 3rd, 4th, and 5th terms of the geometric sequence is 84 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (h : GeometricSequence a) :
  a 3 + a 4 + a 5 = 84 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l980_98082


namespace NUMINAMATH_CALUDE_cubic_sum_problem_l980_98011

theorem cubic_sum_problem (a b : ℝ) (h1 : a + b = 5) (h2 : a * b = -14) :
  a^3 + a^2*b + a*b^2 + b^3 = 265 := by sorry

end NUMINAMATH_CALUDE_cubic_sum_problem_l980_98011


namespace NUMINAMATH_CALUDE_infinitely_many_good_pairs_l980_98083

/-- A natural number is 'good' if every prime factor in its prime factorization appears with at least the power of 2. -/
def is_good (n : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → (∃ k : ℕ, k ≥ 2 ∧ p^k ∣ n)

/-- Definition of the sequence a_n -/
def a : ℕ → ℕ
  | 0 => 8
  | n + 1 => 4 * a n * (a n + 1)

/-- The main theorem stating that there are infinitely many pairs of consecutive 'good' numbers -/
theorem infinitely_many_good_pairs :
  ∀ n : ℕ, is_good (a n) ∧ is_good (a n + 1) :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_good_pairs_l980_98083


namespace NUMINAMATH_CALUDE_p_oplus_q_equals_result_l980_98026

def P : Set ℤ := {4, 5}
def Q : Set ℤ := {1, 2, 3}

def setDifference (P Q : Set ℤ) : Set ℤ :=
  {x | ∃ p ∈ P, ∃ q ∈ Q, x = p - q}

theorem p_oplus_q_equals_result : setDifference P Q = {1, 2, 3, 4} := by
  sorry

end NUMINAMATH_CALUDE_p_oplus_q_equals_result_l980_98026


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l980_98075

/-- A quadratic function of the form y = x^2 + mx + m^2 - 3 -/
def quadratic_function (m : ℝ) (x : ℝ) : ℝ :=
  x^2 + m*x + m^2 - 3

theorem quadratic_function_properties :
  ∀ m : ℝ, m > 0 →
  quadratic_function m 2 = 4 →
  (m = 1 ∧ ∃ x y : ℝ, x ≠ y ∧ quadratic_function m x = 0 ∧ quadratic_function m y = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l980_98075


namespace NUMINAMATH_CALUDE_rectangular_solid_surface_area_l980_98070

-- Define a structure for a rectangular solid
structure RectangularSolid where
  length : ℕ
  width : ℕ
  height : ℕ

-- Define the properties of the rectangular solid
def isPrime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem rectangular_solid_surface_area 
  (solid : RectangularSolid) 
  (prime_edges : isPrime solid.length ∧ isPrime solid.width ∧ isPrime solid.height) 
  (volume_constraint : solid.length * solid.width * solid.height = 105) :
  2 * (solid.length * solid.width + solid.width * solid.height + solid.height * solid.length) = 142 := by
  sorry


end NUMINAMATH_CALUDE_rectangular_solid_surface_area_l980_98070
