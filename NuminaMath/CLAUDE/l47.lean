import Mathlib

namespace NUMINAMATH_CALUDE_hcf_of_specific_numbers_l47_4714

/-- Given two positive integers with a product of 363 and the greater number being 33,
    prove that their highest common factor (HCF) is 11. -/
theorem hcf_of_specific_numbers :
  ∀ A B : ℕ+,
  A * B = 363 →
  A = 33 →
  A > B →
  Nat.gcd A.val B.val = 11 := by
sorry

end NUMINAMATH_CALUDE_hcf_of_specific_numbers_l47_4714


namespace NUMINAMATH_CALUDE_prob_red_or_black_prob_not_green_l47_4774

/-- Represents the colors of balls in the box -/
inductive Color
  | Red
  | Black
  | White
  | Green

/-- The total number of balls in the box -/
def totalBalls : ℕ := 12

/-- The number of balls of each color -/
def ballCount (c : Color) : ℕ :=
  match c with
  | Color.Red => 5
  | Color.Black => 4
  | Color.White => 2
  | Color.Green => 1

/-- The probability of drawing a ball of a given color -/
def probability (c : Color) : ℚ :=
  ballCount c / totalBalls

/-- Theorem: The probability of drawing either a red or black ball is 3/4 -/
theorem prob_red_or_black :
  probability Color.Red + probability Color.Black = 3/4 := by sorry

/-- Theorem: The probability of drawing a ball that is not green is 11/12 -/
theorem prob_not_green :
  1 - probability Color.Green = 11/12 := by sorry

end NUMINAMATH_CALUDE_prob_red_or_black_prob_not_green_l47_4774


namespace NUMINAMATH_CALUDE_invertible_elements_and_inverses_l47_4702

-- Define the invertible elements and their inverses for modulo 8
def invertible_mod_8 : Set ℤ := {1, 3, 5, 7}
def inverse_mod_8 : ℤ → ℤ
  | 1 => 1
  | 3 => 3
  | 5 => 5
  | 7 => 7
  | _ => 0  -- Default case for non-invertible elements

-- Define the invertible elements and their inverses for modulo 9
def invertible_mod_9 : Set ℤ := {1, 2, 4, 5, 7, 8}
def inverse_mod_9 : ℤ → ℤ
  | 1 => 1
  | 2 => 5
  | 4 => 7
  | 5 => 2
  | 7 => 4
  | 8 => 8
  | _ => 0  -- Default case for non-invertible elements

theorem invertible_elements_and_inverses :
  (∀ x ∈ invertible_mod_8, (x * inverse_mod_8 x) % 8 = 1) ∧
  (∀ x ∈ invertible_mod_9, (x * inverse_mod_9 x) % 9 = 1) :=
by sorry

end NUMINAMATH_CALUDE_invertible_elements_and_inverses_l47_4702


namespace NUMINAMATH_CALUDE_cube_sum_is_90_l47_4755

-- Define the type for cube face numbers
def CubeFaces := Fin 6 → ℝ

-- Define the property of consecutive numbers
def IsConsecutive (faces : CubeFaces) : Prop :=
  ∀ i j : Fin 6, i.val < j.val → faces j - faces i = j.val - i.val

-- Define the property of opposite faces summing to 30
def OppositeFacesSum30 (faces : CubeFaces) : Prop :=
  faces 0 + faces 5 = 30 ∧ faces 1 + faces 4 = 30 ∧ faces 2 + faces 3 = 30

-- Theorem statement
theorem cube_sum_is_90 (faces : CubeFaces) 
  (h1 : IsConsecutive faces) (h2 : OppositeFacesSum30 faces) : 
  (Finset.univ.sum faces) = 90 := by sorry

end NUMINAMATH_CALUDE_cube_sum_is_90_l47_4755


namespace NUMINAMATH_CALUDE_correct_number_of_selection_plans_l47_4757

def number_of_people : ℕ := 6
def number_of_cities : ℕ := 4
def number_of_restricted_people : ℕ := 2

def selection_plans : ℕ := 240

theorem correct_number_of_selection_plans :
  (number_of_people.factorial / (number_of_people - number_of_cities).factorial) -
  (number_of_restricted_people * ((number_of_people - 1).factorial / (number_of_people - number_of_cities).factorial)) =
  selection_plans := by
  sorry

end NUMINAMATH_CALUDE_correct_number_of_selection_plans_l47_4757


namespace NUMINAMATH_CALUDE_larger_number_proof_l47_4722

theorem larger_number_proof (L S : ℕ) (h1 : L - S = 1395) (h2 : L = 6 * S + 15) : L = 1671 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l47_4722


namespace NUMINAMATH_CALUDE_circle_intersection_theorem_l47_4739

open Set
open Real

-- Define a type for points in the coordinate plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a function to check if a point has integer coordinates
def isIntegerPoint (p : Point) : Prop :=
  ∃ (m n : ℤ), p.x = m ∧ p.y = n

-- Define a circle with center and radius
def Circle (center : Point) (radius : ℝ) : Set Point :=
  {p : Point | (p.x - center.x)^2 + (p.y - center.y)^2 ≤ radius^2}

-- Define the intersection of two circles
def circlesIntersect (c1 c2 : Set Point) : Prop :=
  ∃ (p : Point), p ∈ c1 ∧ p ∈ c2

-- State the theorem
theorem circle_intersection_theorem :
  ∀ (O : Point),
    ∃ (I : Point),
      isIntegerPoint I ∧
      circlesIntersect (Circle O 100) (Circle I (1/14)) :=
sorry

end NUMINAMATH_CALUDE_circle_intersection_theorem_l47_4739


namespace NUMINAMATH_CALUDE_rhombus_longer_diagonal_l47_4725

/-- A rhombus with side length 65 and shorter diagonal 60 has a longer diagonal of 110 -/
theorem rhombus_longer_diagonal (side_length : ℝ) (shorter_diagonal : ℝ) (longer_diagonal : ℝ)
  (h1 : side_length = 65)
  (h2 : shorter_diagonal = 60)
  (h3 : longer_diagonal * longer_diagonal / 4 + shorter_diagonal * shorter_diagonal / 4 = side_length * side_length) :
  longer_diagonal = 110 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_longer_diagonal_l47_4725


namespace NUMINAMATH_CALUDE_equation_four_real_solutions_l47_4765

theorem equation_four_real_solutions :
  ∃! (s : Finset ℝ), (∀ x ∈ s, (x^2 - 3*x - 4)^2 = 9) ∧ s.card = 4 := by
sorry

end NUMINAMATH_CALUDE_equation_four_real_solutions_l47_4765


namespace NUMINAMATH_CALUDE_tile_draw_probability_l47_4705

/-- The number of tiles in box A -/
def box_a_size : ℕ := 25

/-- The number of tiles in box B -/
def box_b_size : ℕ := 30

/-- The lowest number on a tile in box A -/
def box_a_min : ℕ := 1

/-- The highest number on a tile in box A -/
def box_a_max : ℕ := 25

/-- The lowest number on a tile in box B -/
def box_b_min : ℕ := 10

/-- The highest number on a tile in box B -/
def box_b_max : ℕ := 39

/-- The threshold for "less than" condition in box A -/
def box_a_threshold : ℕ := 18

/-- The threshold for "greater than" condition in box B -/
def box_b_threshold : ℕ := 30

theorem tile_draw_probability : 
  (((box_a_threshold - box_a_min : ℚ) / box_a_size) * 
   ((box_b_size - (box_b_threshold - box_b_min + 1) / 2 + (box_b_max - box_b_threshold)) / box_b_size)) = 323 / 750 := by
  sorry


end NUMINAMATH_CALUDE_tile_draw_probability_l47_4705


namespace NUMINAMATH_CALUDE_pullup_median_is_5_point_5_l47_4742

def pullup_counts : List ℕ := [4, 4, 5, 5, 5, 6, 6, 7, 7, 8]

def median (l : List ℝ) : ℝ := sorry

theorem pullup_median_is_5_point_5 :
  median (pullup_counts.map (λ x => (x : ℝ))) = 5.5 := by sorry

end NUMINAMATH_CALUDE_pullup_median_is_5_point_5_l47_4742


namespace NUMINAMATH_CALUDE_base_seven_digits_of_1234_l47_4753

theorem base_seven_digits_of_1234 : ∃ n : ℕ, (7^n ≤ 1234 ∧ ∀ m : ℕ, 7^m ≤ 1234 → m ≤ n) ∧ n + 1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_base_seven_digits_of_1234_l47_4753


namespace NUMINAMATH_CALUDE_system_solution_l47_4708

theorem system_solution (x y : ℝ) 
  (eq1 : 2019 * x + 2020 * y = 2018)
  (eq2 : 2020 * x + 2019 * y = 2021) :
  x + y = 1 ∧ x - y = 3 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l47_4708


namespace NUMINAMATH_CALUDE_apple_count_bottle_apple_relation_l47_4790

/-- The number of bottles of regular soda -/
def regular_soda : ℕ := 72

/-- The number of bottles of diet soda -/
def diet_soda : ℕ := 32

/-- The number of apples -/
def apples : ℕ := 78

/-- The difference between the number of bottles and apples -/
def bottle_apple_difference : ℕ := 26

/-- Theorem stating that the number of apples is 78 -/
theorem apple_count : apples = 78 := by
  sorry

/-- Theorem proving the relationship between bottles and apples -/
theorem bottle_apple_relation : 
  regular_soda + diet_soda = apples + bottle_apple_difference := by
  sorry

end NUMINAMATH_CALUDE_apple_count_bottle_apple_relation_l47_4790


namespace NUMINAMATH_CALUDE_agatha_bike_purchase_l47_4746

/-- Given Agatha's bike purchase scenario, prove the remaining amount for seat and handlebar tape. -/
theorem agatha_bike_purchase (total_budget : ℕ) (frame_cost : ℕ) (front_wheel_cost : ℕ) :
  total_budget = 60 →
  frame_cost = 15 →
  front_wheel_cost = 25 →
  total_budget - (frame_cost + front_wheel_cost) = 20 :=
by sorry

end NUMINAMATH_CALUDE_agatha_bike_purchase_l47_4746


namespace NUMINAMATH_CALUDE_sunday_letters_zero_l47_4732

/-- Represents the number of letters written on each day of the week -/
structure WeeklyLetters where
  sunday : ℕ
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ
  saturday : ℕ

/-- The average number of letters written per day -/
def averageLettersPerDay : ℕ := 9

/-- The total number of days in a week -/
def daysInWeek : ℕ := 7

/-- Calculates the total number of letters written in a week -/
def totalLetters (w : WeeklyLetters) : ℕ :=
  w.sunday + w.monday + w.tuesday + w.wednesday + w.thursday + w.friday + w.saturday

/-- States that the total number of letters written in a week equals the average per day times the number of days -/
axiom total_letters_axiom (w : WeeklyLetters) :
  totalLetters w = averageLettersPerDay * daysInWeek

/-- Defines the known number of letters written on specific days -/
def knownLetters (w : WeeklyLetters) : Prop :=
  w.wednesday ≥ 13 ∧ w.thursday ≥ 12 ∧ w.friday ≥ 9 ∧ w.saturday ≥ 7

/-- Theorem stating that given the conditions, the number of letters written on Sunday must be zero -/
theorem sunday_letters_zero (w : WeeklyLetters) 
  (h : knownLetters w) : w.sunday = 0 := by
  sorry


end NUMINAMATH_CALUDE_sunday_letters_zero_l47_4732


namespace NUMINAMATH_CALUDE_sine_product_ratio_l47_4724

theorem sine_product_ratio (c : ℝ) (h : c = 2 * Real.pi / 13) :
  (Real.sin (4 * c) * Real.sin (8 * c) * Real.sin (12 * c) * Real.sin (16 * c) * Real.sin (20 * c)) /
  (Real.sin c * Real.sin (3 * c) * Real.sin (5 * c) * Real.sin (7 * c) * Real.sin (9 * c)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sine_product_ratio_l47_4724


namespace NUMINAMATH_CALUDE_largest_integer_with_remainder_l47_4764

theorem largest_integer_with_remainder : ∃ n : ℕ, n < 100 ∧ n % 7 = 4 ∧ ∀ m : ℕ, m < 100 ∧ m % 7 = 4 → m ≤ n :=
  by sorry

end NUMINAMATH_CALUDE_largest_integer_with_remainder_l47_4764


namespace NUMINAMATH_CALUDE_equal_area_if_equal_midpoints_l47_4707

/-- A polygon with an even number of sides -/
structure EvenPolygon where
  vertices : List (ℝ × ℝ)
  even_sides : Even vertices.length

/-- The midpoints of the sides of a polygon -/
def midpoints (p : EvenPolygon) : List (ℝ × ℝ) :=
  sorry

/-- The area of a polygon -/
def area (p : EvenPolygon) : ℝ :=
  sorry

/-- Theorem: If two even-sided polygons have the same midpoints, their areas are equal -/
theorem equal_area_if_equal_midpoints (p q : EvenPolygon) 
  (h : midpoints p = midpoints q) : area p = area q :=
  sorry

end NUMINAMATH_CALUDE_equal_area_if_equal_midpoints_l47_4707


namespace NUMINAMATH_CALUDE_power_equation_solution_l47_4729

theorem power_equation_solution (n : ℕ) : 3^n = 3 * 9^5 * 81^3 → n = 23 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l47_4729


namespace NUMINAMATH_CALUDE_angle_of_inclination_at_max_area_l47_4703

-- Define the line equation
def line_equation (k x y : ℝ) : Prop := y = (k - 1) * x + 2

-- Define the circle equation
def circle_equation (k x y : ℝ) : Prop := x^2 + y^2 + k*x + 2*y + k^2 = 0

-- Define the condition for maximum area of the circle
def max_area_condition (k : ℝ) : Prop := k = 0

-- Theorem statement
theorem angle_of_inclination_at_max_area (k : ℝ) :
  max_area_condition k →
  ∃ (x y : ℝ), line_equation k x y ∧ circle_equation k x y →
  Real.arctan (-1) = 3 * Real.pi / 4 :=
sorry

end NUMINAMATH_CALUDE_angle_of_inclination_at_max_area_l47_4703


namespace NUMINAMATH_CALUDE_rotation_equivalence_l47_4779

/-- 
Given a point P rotated about a center Q:
1. 510 degrees clockwise rotation reaches point R
2. y degrees counterclockwise rotation also reaches point R
3. y < 360

Prove that y = 210
-/
theorem rotation_equivalence (y : ℝ) 
  (h1 : y < 360)
  (h2 : (510 % 360 : ℝ) = (360 - y) % 360) : y = 210 := by
  sorry

end NUMINAMATH_CALUDE_rotation_equivalence_l47_4779


namespace NUMINAMATH_CALUDE_binomial_coefficient_9_5_l47_4737

theorem binomial_coefficient_9_5 : Nat.choose 9 5 = 126 := by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_9_5_l47_4737


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l47_4796

theorem simplify_and_rationalize (x : ℝ) : 
  1 / (2 + 1 / (Real.sqrt 5 + 2)) = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l47_4796


namespace NUMINAMATH_CALUDE_evaporation_problem_l47_4715

/-- Represents the composition of a solution --/
structure Solution where
  total : ℝ
  liquid_x_percent : ℝ
  water_percent : ℝ

/-- The problem statement --/
theorem evaporation_problem (y : Solution) 
  (h1 : y.liquid_x_percent = 0.3)
  (h2 : y.water_percent = 0.7)
  (h3 : y.total = 8)
  (evaporated_water : ℝ)
  (h4 : evaporated_water = 4)
  (added_y : ℝ)
  (h5 : added_y = 4)
  (new_liquid_x_percent : ℝ)
  (h6 : new_liquid_x_percent = 0.45) :
  y.total * y.liquid_x_percent + (y.total * y.water_percent - evaporated_water) = 4 := by
  sorry

#check evaporation_problem

end NUMINAMATH_CALUDE_evaporation_problem_l47_4715


namespace NUMINAMATH_CALUDE_M_dense_in_itself_l47_4771

/-- The set M of real numbers of the form (m+n)/√(m²+n²), where m and n are positive integers. -/
def M : Set ℝ :=
  {x : ℝ | ∃ (m n : ℕ), m > 0 ∧ n > 0 ∧ x = (m + n : ℝ) / Real.sqrt ((m^2 + n^2 : ℕ))}

/-- M is dense in itself -/
theorem M_dense_in_itself : ∀ (x y : ℝ), x ∈ M → y ∈ M → x < y → ∃ (z : ℝ), z ∈ M ∧ x < z ∧ z < y := by
  sorry

end NUMINAMATH_CALUDE_M_dense_in_itself_l47_4771


namespace NUMINAMATH_CALUDE_train_length_train_length_approx_200m_l47_4743

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmph : ℝ) (crossing_time_sec : ℝ) : ℝ :=
  let speed_mps := speed_kmph * 1000 / 3600
  speed_mps * crossing_time_sec

/-- Proof that a train traveling at 120 kmph crossing a pole in 6 seconds is approximately 200 meters long -/
theorem train_length_approx_200m :
  ∃ ε > 0, |train_length 120 6 - 200| < ε :=
sorry

end NUMINAMATH_CALUDE_train_length_train_length_approx_200m_l47_4743


namespace NUMINAMATH_CALUDE_edge_probability_in_cube_l47_4706

/-- A regular cube -/
structure RegularCube where
  vertices : Nat
  edges_per_vertex : Nat

/-- The probability of selecting two vertices that form an edge in a regular cube -/
def edge_probability (cube : RegularCube) : ℚ :=
  (cube.vertices * cube.edges_per_vertex / 2) / (cube.vertices.choose 2)

/-- Theorem stating the probability of selecting two vertices that form an edge in a regular cube -/
theorem edge_probability_in_cube :
  ∃ (cube : RegularCube), cube.vertices = 8 ∧ cube.edges_per_vertex = 3 ∧ edge_probability cube = 3/7 :=
sorry

end NUMINAMATH_CALUDE_edge_probability_in_cube_l47_4706


namespace NUMINAMATH_CALUDE_triangle_count_l47_4700

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- The number of points on the circle -/
def num_points : ℕ := 10

/-- The number of vertices needed to form a triangle -/
def vertices_per_triangle : ℕ := 3

theorem triangle_count :
  choose num_points vertices_per_triangle = 120 := by
  sorry

end NUMINAMATH_CALUDE_triangle_count_l47_4700


namespace NUMINAMATH_CALUDE_discount_percentage_l47_4751

theorem discount_percentage (d : ℝ) (h : d > 0) : 
  ∃ x : ℝ, x ≥ 0 ∧ x ≤ 100 ∧ 
  (1 - x / 100) * 0.9 * d = 0.765 * d ∧ 
  x = 15 := by
sorry

end NUMINAMATH_CALUDE_discount_percentage_l47_4751


namespace NUMINAMATH_CALUDE_log_relationship_depends_on_base_l47_4798

noncomputable def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

theorem log_relationship_depends_on_base (a : ℝ) 
  (h1 : a > 0) (h2 : a ≠ 1) :
  ∃ (a1 a2 : ℝ), 
    (a1 > 0 ∧ a1 ≠ 1 ∧ log a1 2 + log a1 10 > 2 * log a1 6) ∧
    (a2 > 0 ∧ a2 ≠ 1 ∧ log a2 2 + log a2 10 < 2 * log a2 6) :=
by sorry

end NUMINAMATH_CALUDE_log_relationship_depends_on_base_l47_4798


namespace NUMINAMATH_CALUDE_water_intersection_points_l47_4710

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube -/
structure Cube where
  edgeLength : ℝ

/-- Represents the water level in the cube -/
def waterLevel (c : Cube) (vol : ℝ) : ℝ :=
  vol * c.edgeLength

theorem water_intersection_points (c : Cube) (waterVol : ℝ) :
  c.edgeLength = 1 →
  waterVol = 5/6 →
  ∃ (x : ℝ), 
    0.26 < x ∧ x < 0.28 ∧ 
    0.72 < (1 - x) ∧ (1 - x) < 0.74 ∧
    (waterLevel c waterVol = x ∨ waterLevel c waterVol = 1 - x) := by
  sorry

#check water_intersection_points

end NUMINAMATH_CALUDE_water_intersection_points_l47_4710


namespace NUMINAMATH_CALUDE_models_after_price_increase_l47_4799

-- Define the original price, price increase percentage, and initial number of models
def original_price : ℚ := 45/100
def price_increase_percent : ℚ := 15/100
def initial_models : ℕ := 30

-- Calculate the new price after the increase
def new_price : ℚ := original_price * (1 + price_increase_percent)

-- Calculate the total savings
def total_savings : ℚ := original_price * initial_models

-- Define the theorem
theorem models_after_price_increase :
  ⌊total_savings / new_price⌋ = 26 := by
  sorry

#eval ⌊total_savings / new_price⌋

end NUMINAMATH_CALUDE_models_after_price_increase_l47_4799


namespace NUMINAMATH_CALUDE_f_max_value_l47_4728

/-- The function f(x) defined as |tx-2| - |tx+1| where t is a real number -/
def f (t : ℝ) (x : ℝ) : ℝ := |t*x - 2| - |t*x + 1|

/-- The maximum value of f(x) is 3 -/
theorem f_max_value (t : ℝ) : 
  ∃ (M : ℝ), M = 3 ∧ ∀ x, f t x ≤ M :=
sorry

end NUMINAMATH_CALUDE_f_max_value_l47_4728


namespace NUMINAMATH_CALUDE_exists_real_less_than_one_exists_natural_in_real_exists_real_between_two_and_three_forall_int_exists_real_outside_interval_l47_4738

-- 1. Prove that there exists a real number less than 1
theorem exists_real_less_than_one : ∃ x : ℝ, x < 1 := by sorry

-- 2. Prove that there exists a natural number that is also a real number
theorem exists_natural_in_real : ∃ x : ℕ, ∃ y : ℝ, x = y := by sorry

-- 3. Prove that there exists a real number greater than 2 and less than 3
theorem exists_real_between_two_and_three : ∃ x : ℝ, x > 2 ∧ x < 3 := by sorry

-- 4. Prove that for all integers n, there exists a real number x that is either less than n or greater than or equal to n + 1
theorem forall_int_exists_real_outside_interval : ∀ n : ℤ, ∃ x : ℝ, x < n ∨ x ≥ n + 1 := by sorry

end NUMINAMATH_CALUDE_exists_real_less_than_one_exists_natural_in_real_exists_real_between_two_and_three_forall_int_exists_real_outside_interval_l47_4738


namespace NUMINAMATH_CALUDE_roots_of_equation_l47_4741

theorem roots_of_equation (x : ℝ) : 
  (x - 1) * (x - 2) = 0 ↔ x = 1 ∨ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_roots_of_equation_l47_4741


namespace NUMINAMATH_CALUDE_smallest_multiple_with_digit_sum_l47_4766

def N : ℕ := 5 * 10^223 - 10^220 - 10^49 - 1

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem smallest_multiple_with_digit_sum :
  (N % 2009 = 0) ∧
  (sum_of_digits N = 2009) ∧
  (∀ m : ℕ, m < N → (m % 2009 = 0 ∧ sum_of_digits m = 2009) → False) :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiple_with_digit_sum_l47_4766


namespace NUMINAMATH_CALUDE_quadratic_inequality_roots_l47_4747

theorem quadratic_inequality_roots (b : ℝ) : 
  (∀ x, -x^2 + b*x - 7 < 0 ↔ x < 2 ∨ x > 6) → b = 8 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_roots_l47_4747


namespace NUMINAMATH_CALUDE_mixture_solution_l47_4784

/-- Represents the mixture composition and constraints -/
structure Mixture where
  d : ℝ  -- diesel amount
  p : ℝ  -- petrol amount
  w : ℝ  -- water amount
  e : ℝ  -- ethanol amount
  total_volume : ℝ  -- total volume of the mixture

/-- The mixture satisfies the given constraints -/
def satisfies_constraints (m : Mixture) : Prop :=
  m.d = 4 ∧ 
  m.p = 4 ∧ 
  m.d / m.total_volume = 0.2 ∧
  m.p / m.total_volume = 0.15 ∧
  m.e / m.total_volume = 0.25 ∧
  m.w / m.total_volume = 0.4 ∧
  m.total_volume ≤ 30 ∧
  m.total_volume = m.d + m.p + m.w + m.e

/-- The theorem to be proved -/
theorem mixture_solution :
  ∃ (m : Mixture), satisfies_constraints m ∧ m.w = 8 ∧ m.e = 5 ∧ m.total_volume = 20 := by
  sorry


end NUMINAMATH_CALUDE_mixture_solution_l47_4784


namespace NUMINAMATH_CALUDE_first_robber_guarantee_l47_4713

/-- Represents the coin division game between two robbers --/
structure CoinGame where
  totalCoins : ℕ
  maxBags : ℕ

/-- Represents the outcome of the game for the first robber --/
def firstRobberOutcome (game : CoinGame) (coinsPerBag : ℕ) : ℕ :=
  min (game.totalCoins - game.maxBags * coinsPerBag) (game.maxBags * coinsPerBag)

/-- The theorem stating the maximum guaranteed coins for the first robber --/
theorem first_robber_guarantee (game : CoinGame) : 
  game.totalCoins = 300 → game.maxBags = 11 → 
  ∃ (coinsPerBag : ℕ), firstRobberOutcome game coinsPerBag ≥ 146 := by
  sorry

#eval firstRobberOutcome { totalCoins := 300, maxBags := 11 } 14

end NUMINAMATH_CALUDE_first_robber_guarantee_l47_4713


namespace NUMINAMATH_CALUDE_parallelogram_xy_product_l47_4748

/-- A parallelogram with side lengths given in terms of x and y -/
structure Parallelogram (x y : ℝ) :=
  (ef : ℝ)
  (fg : ℝ)
  (gh : ℝ)
  (he : ℝ)
  (ef_eq : ef = 42)
  (fg_eq : fg = 4 * y^3)
  (gh_eq : gh = 2 * x + 10)
  (he_eq : he = 32)
  (opposite_sides_equal : ef = gh ∧ fg = he)

/-- The product of x and y in the given parallelogram is 32 -/
theorem parallelogram_xy_product (x y : ℝ) (p : Parallelogram x y) :
  x * y = 32 :=
sorry

end NUMINAMATH_CALUDE_parallelogram_xy_product_l47_4748


namespace NUMINAMATH_CALUDE_simple_interest_example_l47_4792

/-- Calculate simple interest given principal, rate, and time -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

/-- Proof that the simple interest on $10000 at 8% per annum for 12 months is $800 -/
theorem simple_interest_example : 
  simple_interest 10000 0.08 1 = 800 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_example_l47_4792


namespace NUMINAMATH_CALUDE_hyperbola_derivative_l47_4786

variable (a b x y : ℝ)
variable (h : x^2 / a^2 - y^2 / b^2 = 1)

theorem hyperbola_derivative :
  ∃ (dy_dx : ℝ), dy_dx = (b^2 * x) / (a^2 * y) := by sorry

end NUMINAMATH_CALUDE_hyperbola_derivative_l47_4786


namespace NUMINAMATH_CALUDE_union_P_complement_Q_l47_4778

open Set

-- Define the sets P and Q
def P : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}
def Q : Set ℝ := {x | x^2 ≥ 4}

-- Define the complement of Q in ℝ
def C_R_Q : Set ℝ := {x | ¬(x ∈ Q)}

-- State the theorem
theorem union_P_complement_Q : P ∪ C_R_Q = Ioc (-2) 3 := by sorry

end NUMINAMATH_CALUDE_union_P_complement_Q_l47_4778


namespace NUMINAMATH_CALUDE_geometric_sequence_150th_term_l47_4734

def geometricSequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a₁ * r ^ (n - 1)

theorem geometric_sequence_150th_term :
  let a₁ := 8
  let a₂ := -16
  let r := a₂ / a₁
  geometricSequence a₁ r 150 = 8 * (-2)^149 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_150th_term_l47_4734


namespace NUMINAMATH_CALUDE_parallelepiped_coverage_l47_4760

/-- A parallelepiped with dimensions a, b, and c can have three faces sharing a common vertex
    covered by five-cell strips without overlaps or gaps if and only if at least two of a, b,
    and c are divisible by 5. -/
theorem parallelepiped_coverage (a b c : ℕ) :
  (∃ (faces : Fin 3 → ℕ × ℕ), 
    (faces 0 = (a, b) ∨ faces 0 = (b, c) ∨ faces 0 = (c, a)) ∧
    (faces 1 = (a, b) ∨ faces 1 = (b, c) ∨ faces 1 = (c, a)) ∧
    (faces 2 = (a, b) ∨ faces 2 = (b, c) ∨ faces 2 = (c, a)) ∧
    faces 0 ≠ faces 1 ∧ faces 1 ≠ faces 2 ∧ faces 0 ≠ faces 2 ∧
    ∀ i : Fin 3, ∃ k : ℕ, (faces i).1 * (faces i).2 = 5 * k) ↔
  (a % 5 = 0 ∧ b % 5 = 0) ∨ (b % 5 = 0 ∧ c % 5 = 0) ∨ (c % 5 = 0 ∧ a % 5 = 0) :=
by sorry

end NUMINAMATH_CALUDE_parallelepiped_coverage_l47_4760


namespace NUMINAMATH_CALUDE_shekars_math_marks_l47_4788

def science_marks : ℕ := 65
def social_studies_marks : ℕ := 82
def english_marks : ℕ := 67
def biology_marks : ℕ := 85
def average_marks : ℕ := 75
def total_subjects : ℕ := 5

theorem shekars_math_marks :
  ∃ math_marks : ℕ,
    math_marks = average_marks * total_subjects - (science_marks + social_studies_marks + english_marks + biology_marks) :=
by
  sorry

end NUMINAMATH_CALUDE_shekars_math_marks_l47_4788


namespace NUMINAMATH_CALUDE_stating_max_principals_l47_4717

/-- Represents the duration of the period in years -/
def period_duration : ℕ := 10

/-- Represents the duration of a principal's term in years -/
def term_duration : ℕ := 4

/-- 
Theorem stating that the maximum number of principals 
that can serve during the given period is 3
-/
theorem max_principals :
  ∀ (principal_count : ℕ),
  (∀ (year : ℕ), year ≤ period_duration → 
    ∃ (principal : ℕ), principal ≤ principal_count ∧ 
    ∃ (start_year : ℕ), start_year ≤ period_duration ∧ 
    year ∈ Set.Icc start_year (start_year + term_duration - 1)) →
  principal_count ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_stating_max_principals_l47_4717


namespace NUMINAMATH_CALUDE_f_min_value_a_value_l47_4701

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x + 2| + 2 * |x - 1|

-- Theorem for the minimum value of f(x)
theorem f_min_value : ∃ (x : ℝ), ∀ (y : ℝ), f x ≤ f y ∧ f x = 3 := by sorry

-- Define the solution set condition
def solution_set_condition (a : ℝ) (m n : ℝ) : Prop :=
  (∀ x, m < x ∧ x < n ↔ f x + x - a < 0) ∧ n - m = 6

-- Theorem for the value of a
theorem a_value : ∀ (m n : ℝ), solution_set_condition 8 m n := by sorry

end NUMINAMATH_CALUDE_f_min_value_a_value_l47_4701


namespace NUMINAMATH_CALUDE_segment_inequalities_l47_4789

/-- Given a line segment AD with points B and C, prove inequalities about their lengths -/
theorem segment_inequalities 
  (a b c : ℝ) 
  (h1 : 0 < a) (h2 : a < b) (h3 : b < c) :
  a < c/2 ∧ b < a + c/2 := by
  sorry

end NUMINAMATH_CALUDE_segment_inequalities_l47_4789


namespace NUMINAMATH_CALUDE_train_wheel_rows_l47_4787

/-- Proves that the number of rows of wheels per carriage is 3, given the conditions of the train station. -/
theorem train_wheel_rows (num_trains : ℕ) (carriages_per_train : ℕ) (wheels_per_row : ℕ) (total_wheels : ℕ) :
  num_trains = 4 →
  carriages_per_train = 4 →
  wheels_per_row = 5 →
  total_wheels = 240 →
  (num_trains * carriages_per_train * wheels_per_row * (total_wheels / (num_trains * carriages_per_train * wheels_per_row))) = total_wheels →
  (total_wheels / (num_trains * carriages_per_train * wheels_per_row)) = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_train_wheel_rows_l47_4787


namespace NUMINAMATH_CALUDE_sum_of_coefficients_excluding_constant_l47_4744

/-- The sum of the coefficients of the terms, excluding the constant term, 
    in the expansion of (x^2 - 2/x)^6 is -239 -/
theorem sum_of_coefficients_excluding_constant (x : ℝ) : 
  let f := (x^2 - 2/x)^6
  let all_coeff_sum := (1 - 2)^6
  let constant_term := 240
  all_coeff_sum - constant_term = -239 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_excluding_constant_l47_4744


namespace NUMINAMATH_CALUDE_line_plane_perpendicularity_l47_4727

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (notParallel : Line → Line → Prop)
variable (notParallelToPlane : Line → Plane → Prop)
variable (perpendicularPlanes : Plane → Plane → Prop)

-- State the theorem
theorem line_plane_perpendicularity 
  (m n : Line) (α β : Plane) :
  perpendicular m α → 
  notParallel m n → 
  notParallelToPlane n β → 
  perpendicularPlanes α β :=
sorry

end NUMINAMATH_CALUDE_line_plane_perpendicularity_l47_4727


namespace NUMINAMATH_CALUDE_quadratic_form_sum_l47_4762

theorem quadratic_form_sum (a h k : ℝ) : 
  (∀ x, 5 * x^2 - 10 * x - 7 = a * (x - h)^2 + k) → 
  a + h + k = -6 := by
sorry

end NUMINAMATH_CALUDE_quadratic_form_sum_l47_4762


namespace NUMINAMATH_CALUDE_smallest_fd_minus_de_is_eight_l47_4721

/-- Represents a triangle with integer side lengths -/
structure Triangle where
  de : ℕ
  ef : ℕ
  fd : ℕ

/-- Checks if the given triangle satisfies the triangle inequality -/
def satisfies_triangle_inequality (t : Triangle) : Prop :=
  t.de + t.ef > t.fd ∧ t.ef + t.fd > t.de ∧ t.fd + t.de > t.ef

/-- The main theorem stating the smallest difference between FD and DE -/
theorem smallest_fd_minus_de_is_eight :
  ∀ t : Triangle,
    t.de + t.ef + t.fd = 3009 →
    t.de < t.ef →
    t.ef ≤ t.fd →
    satisfies_triangle_inequality t →
    (∀ t' : Triangle,
      t'.de + t'.ef + t'.fd = 3009 →
      t'.de < t'.ef →
      t'.ef ≤ t'.fd →
      satisfies_triangle_inequality t' →
      t'.fd - t'.de ≥ t.fd - t.de) →
    t.fd - t.de = 8 := by
  sorry

#check smallest_fd_minus_de_is_eight

end NUMINAMATH_CALUDE_smallest_fd_minus_de_is_eight_l47_4721


namespace NUMINAMATH_CALUDE_garden_dimensions_l47_4736

theorem garden_dimensions :
  ∃! n : ℕ, n = (Finset.filter 
    (fun p : ℕ × ℕ => 
      p.2 > p.1 ∧ 
      (p.1 - 6) * (p.2 - 6) = 12 ∧ 
      p.1 ≥ 7 ∧ p.2 ≥ 7)
    (Finset.product (Finset.range 100) (Finset.range 100))).card ∧ 
  n = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_garden_dimensions_l47_4736


namespace NUMINAMATH_CALUDE_max_stores_visited_is_four_l47_4723

/-- Represents the shopping scenario in the town -/
structure ShoppingScenario where
  num_stores : ℕ
  total_visits : ℕ
  num_shoppers : ℕ
  double_visitors : ℕ
  max_stores_visited : ℕ

/-- The specific shopping scenario described in the problem -/
def town_scenario : ShoppingScenario :=
  { num_stores := 7
  , total_visits := 21
  , num_shoppers := 11
  , double_visitors := 7
  , max_stores_visited := 4 }

/-- Theorem stating that the maximum number of stores visited by any single person is 4 -/
theorem max_stores_visited_is_four (s : ShoppingScenario) 
  (h1 : s.num_stores = town_scenario.num_stores)
  (h2 : s.total_visits = town_scenario.total_visits)
  (h3 : s.num_shoppers = town_scenario.num_shoppers)
  (h4 : s.double_visitors = town_scenario.double_visitors)
  (h5 : s.double_visitors * 2 + (s.num_shoppers - s.double_visitors) ≤ s.total_visits) :
  s.max_stores_visited = town_scenario.max_stores_visited :=
by sorry


end NUMINAMATH_CALUDE_max_stores_visited_is_four_l47_4723


namespace NUMINAMATH_CALUDE_students_in_section_B_l47_4761

/-- Proves the number of students in section B given the class information -/
theorem students_in_section_B 
  (students_A : ℕ) 
  (avg_weight_A : ℝ) 
  (avg_weight_B : ℝ) 
  (avg_weight_total : ℝ) 
  (h1 : students_A = 24)
  (h2 : avg_weight_A = 40)
  (h3 : avg_weight_B = 35)
  (h4 : avg_weight_total = 38) : 
  ∃ (students_B : ℕ), 
    (students_A * avg_weight_A + students_B * avg_weight_B) / (students_A + students_B) = avg_weight_total ∧ 
    students_B = 16 := by
  sorry

end NUMINAMATH_CALUDE_students_in_section_B_l47_4761


namespace NUMINAMATH_CALUDE_treasure_chest_value_l47_4731

def base7_to_base10 (n : List Nat) : Nat :=
  n.enum.foldr (fun (i, d) acc => acc + d * (7 ^ i)) 0

theorem treasure_chest_value : 
  let coins := [6, 4, 3, 5]
  let gems := [1, 2, 5, 6]
  let maps := [0, 2, 3]
  base7_to_base10 coins + base7_to_base10 gems + base7_to_base10 maps = 4305 := by
sorry

#eval base7_to_base10 [6, 4, 3, 5] + base7_to_base10 [1, 2, 5, 6] + base7_to_base10 [0, 2, 3]

end NUMINAMATH_CALUDE_treasure_chest_value_l47_4731


namespace NUMINAMATH_CALUDE_a_55_divisible_by_55_l47_4719

/-- Concatenation of integers from 1 to n -/
def a (n : ℕ) : ℕ :=
  -- Definition of a_n goes here
  sorry

/-- Theorem: a_55 is divisible by 55 -/
theorem a_55_divisible_by_55 : 55 ∣ a 55 := by
  sorry

end NUMINAMATH_CALUDE_a_55_divisible_by_55_l47_4719


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l47_4726

theorem algebraic_expression_value (a : ℝ) : 
  (2023 - a)^2 + (a - 2022)^2 = 7 → (2023 - a) * (a - 2022) = -3 := by
sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l47_4726


namespace NUMINAMATH_CALUDE_high_school_enrollment_l47_4745

/-- The number of students in a high school with given enrollment in music and art classes -/
def total_students (music : ℕ) (art : ℕ) (both : ℕ) (neither : ℕ) : ℕ :=
  (music - both) + (art - both) + both + neither

/-- Theorem stating that the total number of students is 500 given the specific enrollment numbers -/
theorem high_school_enrollment : total_students 30 20 10 460 = 500 := by
  sorry

end NUMINAMATH_CALUDE_high_school_enrollment_l47_4745


namespace NUMINAMATH_CALUDE_equal_variance_square_arithmetic_neg_one_power_equal_variance_equal_variance_subsequence_l47_4770

/-- A sequence is an equal variance sequence if the difference of squares of consecutive terms is constant. -/
def EqualVarianceSequence (a : ℕ+ → ℝ) :=
  ∃ p : ℝ, ∀ n : ℕ+, a n ^ 2 - a (n + 1) ^ 2 = p

/-- The square of an equal variance sequence is an arithmetic sequence. -/
theorem equal_variance_square_arithmetic (a : ℕ+ → ℝ) (h : EqualVarianceSequence a) :
  ∃ d : ℝ, ∀ n : ℕ+, (a (n + 1))^2 - (a n)^2 = d := by sorry

/-- The sequence (-1)^n is an equal variance sequence. -/
theorem neg_one_power_equal_variance :
  EqualVarianceSequence (fun n => (-1 : ℝ) ^ (n : ℕ)) := by sorry

/-- If a_n is an equal variance sequence, then a_{kn} is also an equal variance sequence for any positive integer k. -/
theorem equal_variance_subsequence (a : ℕ+ → ℝ) (h : EqualVarianceSequence a) (k : ℕ+) :
  EqualVarianceSequence (fun n => a (k * n)) := by sorry

end NUMINAMATH_CALUDE_equal_variance_square_arithmetic_neg_one_power_equal_variance_equal_variance_subsequence_l47_4770


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l47_4791

/-- The perimeter of a rhombus with given diagonals -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 16) (h2 : d2 = 30) : 
  4 * Real.sqrt ((d1/2)^2 + (d2/2)^2) = 68 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l47_4791


namespace NUMINAMATH_CALUDE_a_share_calculation_l47_4756

/-- Calculates the share of profit for a partner in a business partnership. -/
def calculateShare (investment totalInvestment totalProfit : ℚ) : ℚ :=
  (investment / totalInvestment) * totalProfit

theorem a_share_calculation (investmentA investmentB investmentC shareB : ℚ) 
  (h1 : investmentA = 15000)
  (h2 : investmentB = 21000)
  (h3 : investmentC = 27000)
  (h4 : shareB = 1540) : 
  calculateShare investmentA (investmentA + investmentB + investmentC) 
    ((investmentA + investmentB + investmentC) * shareB / investmentB) = 1100 := by
  sorry

end NUMINAMATH_CALUDE_a_share_calculation_l47_4756


namespace NUMINAMATH_CALUDE_athletes_total_yards_l47_4793

/-- Calculates the total yards run by three athletes over a given number of games -/
def total_yards (yards_per_game_1 yards_per_game_2 yards_per_game_3 : ℕ) (num_games : ℕ) : ℕ :=
  (yards_per_game_1 + yards_per_game_2 + yards_per_game_3) * num_games

/-- Proves that the total yards run by three athletes over 4 games is 204 yards -/
theorem athletes_total_yards :
  total_yards 18 22 11 4 = 204 := by
  sorry

#eval total_yards 18 22 11 4

end NUMINAMATH_CALUDE_athletes_total_yards_l47_4793


namespace NUMINAMATH_CALUDE_circle_parabola_intersection_l47_4794

/-- Circle with center (0,1) and radius 1 -/
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + (p.2 - 1)^2 = 1}

/-- Parabola defined by y = ax² -/
def P (a : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = a * p.1^2}

/-- Theorem stating the condition for C and P to intersect at points other than (0,0) -/
theorem circle_parabola_intersection (a : ℝ) :
  (∃ p : ℝ × ℝ, p ∈ C ∩ P a ∧ p ≠ (0, 0)) ↔ a > 1/2 := by sorry

end NUMINAMATH_CALUDE_circle_parabola_intersection_l47_4794


namespace NUMINAMATH_CALUDE_triangle_side_length_l47_4777

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem stating the relationship between side lengths and angles in the given triangle -/
theorem triangle_side_length (t : Triangle) (h1 : t.a = 2) (h2 : t.B = 135 * π / 180)
    (h3 : (1/2) * t.a * t.c * Real.sin t.B = 4) : t.b = 2 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l47_4777


namespace NUMINAMATH_CALUDE_b_range_l47_4767

/-- A cubic function with a parameter b -/
def f (b : ℝ) (x : ℝ) : ℝ := -x^3 + b*x

/-- Theorem stating the range of b given the conditions -/
theorem b_range :
  ∀ b : ℝ,
  (∀ x : ℝ, f b x = 0 → x ∈ Set.Icc (-2) 2) →
  (∀ x y : ℝ, x ∈ Set.Ioo 0 1 → y ∈ Set.Ioo 0 1 → x < y → f b x < f b y) →
  b ∈ Set.Icc 3 4 := by
sorry

end NUMINAMATH_CALUDE_b_range_l47_4767


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l47_4735

-- Define the sets A and B
def A : Set ℝ := {x | x ≥ 3 ∨ x ≤ 1}
def B : Set ℝ := {x | x^2 - 6*x + 8 < 0}

-- State the theorem
theorem complement_A_intersect_B :
  (Set.univ \ A) ∩ B = Set.Ioo 2 3 := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l47_4735


namespace NUMINAMATH_CALUDE_first_three_digits_after_decimal_l47_4775

theorem first_three_digits_after_decimal (x : ℝ) : 
  x = (10^2003 + 1)^(12/11) → 
  ∃ (n : ℕ), (x - n) * 1000 ≥ 909 ∧ (x - n) * 1000 < 910 := by
  sorry

end NUMINAMATH_CALUDE_first_three_digits_after_decimal_l47_4775


namespace NUMINAMATH_CALUDE_square_sum_over_28_squared_equals_8_l47_4773

theorem square_sum_over_28_squared_equals_8 :
  ∃ x : ℝ, (x^2 + x^2) / 28^2 = 8 ∧ x = 56 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_over_28_squared_equals_8_l47_4773


namespace NUMINAMATH_CALUDE_consecutive_squares_sum_l47_4704

theorem consecutive_squares_sum (x : ℕ) : 
  (x - 1)^2 + x^2 + (x + 1)^2 = 8 * ((x - 1) + x + (x + 1)) + 2 →
  ∃ n : ℕ, (n - 1)^2 + n^2 + (n + 1)^2 = 194 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_squares_sum_l47_4704


namespace NUMINAMATH_CALUDE_sixth_term_is_46_l47_4780

/-- The sequence of small circles in each figure -/
def circleSequence (n : ℕ) : ℕ := n * (n + 1) + 4

/-- The theorem stating that the 6th term of the sequence is 46 -/
theorem sixth_term_is_46 : circleSequence 6 = 46 := by
  sorry

end NUMINAMATH_CALUDE_sixth_term_is_46_l47_4780


namespace NUMINAMATH_CALUDE_roses_equation_initial_roses_count_l47_4768

/-- The number of roses initially in the vase -/
def initial_roses : ℕ := sorry

/-- The number of roses added to the vase -/
def added_roses : ℕ := 16

/-- The final number of roses in the vase -/
def final_roses : ℕ := 22

/-- Theorem stating that the initial number of roses plus the added roses equals the final number of roses -/
theorem roses_equation : initial_roses + added_roses = final_roses := by sorry

/-- Theorem proving that the initial number of roses is 6 -/
theorem initial_roses_count : initial_roses = 6 := by sorry

end NUMINAMATH_CALUDE_roses_equation_initial_roses_count_l47_4768


namespace NUMINAMATH_CALUDE_sampled_bag_number_61st_group_l47_4797

/-- Given a total number of bags, sample size, first sampled bag number, and group number,
    calculate the bag number for that group. -/
def sampledBagNumber (totalBags : ℕ) (sampleSize : ℕ) (firstSampledBag : ℕ) (groupNumber : ℕ) : ℕ :=
  firstSampledBag + (groupNumber - 1) * (totalBags / sampleSize)

/-- Theorem stating that for the given conditions, the 61st group's sampled bag number is 1211. -/
theorem sampled_bag_number_61st_group :
  sampledBagNumber 3000 150 11 61 = 1211 := by
  sorry


end NUMINAMATH_CALUDE_sampled_bag_number_61st_group_l47_4797


namespace NUMINAMATH_CALUDE_prob_odd_after_removal_is_11_21_l47_4752

/-- A standard die with faces numbered 1 to 6 -/
def standardDie : Finset ℕ := {1, 2, 3, 4, 5, 6}

/-- Total number of dots on a standard die -/
def totalDots : ℕ := standardDie.sum id

/-- Probability of removing a dot from a specific face -/
def probRemoveDot (face : ℕ) : ℚ := face / totalDots

/-- Probability of rolling an odd number after removing a dot -/
def probOddAfterRemoval : ℚ :=
  (1 / 6 * (probRemoveDot 2 + probRemoveDot 4 + probRemoveDot 6)) +
  (1 / 3 * (probRemoveDot 1 + probRemoveDot 3 + probRemoveDot 5))

theorem prob_odd_after_removal_is_11_21 : probOddAfterRemoval = 11 / 21 := by
  sorry

end NUMINAMATH_CALUDE_prob_odd_after_removal_is_11_21_l47_4752


namespace NUMINAMATH_CALUDE_ines_peaches_bought_l47_4749

def peaches_bought (initial_amount : ℕ) (remaining_amount : ℕ) (price_per_pound : ℕ) : ℕ :=
  (initial_amount - remaining_amount) / price_per_pound

theorem ines_peaches_bought :
  peaches_bought 20 14 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ines_peaches_bought_l47_4749


namespace NUMINAMATH_CALUDE_subtraction_difference_l47_4754

theorem subtraction_difference (x : ℝ) : (x - 2152) - (x - 1264) = 888 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_difference_l47_4754


namespace NUMINAMATH_CALUDE_birthday_stickers_l47_4785

theorem birthday_stickers (initial : ℕ) (given_away : ℕ) (final : ℕ) : 
  initial = 269 → given_away = 48 → final = 423 → 
  final - given_away - initial = 202 :=
by sorry

end NUMINAMATH_CALUDE_birthday_stickers_l47_4785


namespace NUMINAMATH_CALUDE_new_person_weight_l47_4776

theorem new_person_weight (initial_count : ℕ) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 10 →
  weight_increase = 3.5 →
  replaced_weight = 65 →
  (initial_count : ℝ) * weight_increase + replaced_weight = 100 :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l47_4776


namespace NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_q_l47_4730

-- Define the propositions p and q
def p (x : ℝ) : Prop := |x - 2| > 3
def q (x : ℝ) : Prop := x > 5

-- Statement to prove
theorem not_p_sufficient_not_necessary_for_not_q :
  (∀ x, ¬(p x) → ¬(q x)) ∧ 
  (∃ x, ¬(q x) ∧ p x) :=
sorry

end NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_q_l47_4730


namespace NUMINAMATH_CALUDE_equal_distribution_l47_4718

/-- Represents the weight of a mouse's cheese slice -/
structure CheeseSlice where
  weight : ℝ

/-- Represents the total cheese and its distribution -/
structure Cheese where
  total_weight : ℝ
  white : CheeseSlice
  gray : CheeseSlice
  fat : CheeseSlice
  thin : CheeseSlice

/-- The conditions of the cheese distribution problem -/
def cheese_distribution (c : Cheese) : Prop :=
  c.thin.weight = c.fat.weight - 20 ∧
  c.white.weight = c.gray.weight - 8 ∧
  c.white.weight = c.total_weight / 4 ∧
  c.total_weight = c.white.weight + c.gray.weight + c.fat.weight + c.thin.weight

/-- The theorem stating the equal distribution of surplus cheese -/
theorem equal_distribution (c : Cheese) (h : cheese_distribution c) :
  ∃ (new_c : Cheese),
    cheese_distribution new_c ∧
    new_c.white.weight = new_c.gray.weight ∧
    new_c.fat.weight = new_c.thin.weight ∧
    new_c.fat.weight = c.fat.weight - 6 ∧
    new_c.thin.weight = c.thin.weight + 14 :=
  sorry

end NUMINAMATH_CALUDE_equal_distribution_l47_4718


namespace NUMINAMATH_CALUDE_runners_photo_probability_l47_4781

/-- Represents a runner on a circular track -/
structure Runner where
  lap_time : ℝ
  direction : Bool  -- true for counterclockwise, false for clockwise

/-- Represents the track and photo setup -/
structure TrackSetup where
  photo_fraction : ℝ
  photo_time : ℝ

/-- Calculates the probability of both runners being in the photo -/
def probability_both_in_photo (ellie sam : Runner) (setup : TrackSetup) : ℝ :=
  sorry

/-- The main theorem statement -/
theorem runners_photo_probability :
  let ellie : Runner := { lap_time := 120, direction := true }
  let sam : Runner := { lap_time := 75, direction := false }
  let setup : TrackSetup := { photo_fraction := 1/3, photo_time := 600 }
  probability_both_in_photo ellie sam setup = 5/12 := by
  sorry

end NUMINAMATH_CALUDE_runners_photo_probability_l47_4781


namespace NUMINAMATH_CALUDE_valid_numbers_l47_4711

def isValid (n : ℕ) : Prop :=
  ∃ (d1 d2 d3 : ℕ), d1 > d2 ∧ d2 > d3 ∧
    d1 ∣ n ∧ d2 ∣ n ∧ d3 ∣ n ∧
    d1 + d2 + d3 = 1457 ∧
    ∀ (d : ℕ), d ∣ n → d ≤ d1

theorem valid_numbers : ∀ (n : ℕ), isValid n ↔ n = 987 ∨ n = 1023 ∨ n = 1085 ∨ n = 1175 := by
  sorry

end NUMINAMATH_CALUDE_valid_numbers_l47_4711


namespace NUMINAMATH_CALUDE_initial_speed_is_40_l47_4716

/-- Represents a journey with increasing speed -/
structure Journey where
  totalDistance : ℝ
  totalTime : ℝ
  speedIncrease : ℝ
  intervalTime : ℝ

/-- Calculates the initial speed for a given journey -/
def calculateInitialSpeed (j : Journey) : ℝ :=
  sorry

/-- Theorem stating that for the given journey parameters, the initial speed is 40 km/h -/
theorem initial_speed_is_40 :
  let j : Journey := {
    totalDistance := 56,
    totalTime := 48 / 60, -- converting minutes to hours
    speedIncrease := 20,
    intervalTime := 12 / 60 -- converting minutes to hours
  }
  calculateInitialSpeed j = 40 := by
  sorry

end NUMINAMATH_CALUDE_initial_speed_is_40_l47_4716


namespace NUMINAMATH_CALUDE_football_throw_percentage_l47_4769

theorem football_throw_percentage (parker_throw grant_throw kyle_throw : ℝ) :
  parker_throw = 16 →
  kyle_throw = 2 * grant_throw →
  kyle_throw = parker_throw + 24 →
  (grant_throw - parker_throw) / parker_throw = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_football_throw_percentage_l47_4769


namespace NUMINAMATH_CALUDE_function_inequality_l47_4795

def is_periodic (f : ℝ → ℝ) (period : ℝ) : Prop :=
  ∀ x, f x = f (x + period)

def monotone_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f y < f x

def symmetric_about (f : ℝ → ℝ) (c : ℝ) : Prop :=
  ∀ x, f (c + x) = f (c - x)

theorem function_inequality (f : ℝ → ℝ) 
  (h1 : is_periodic f 6)
  (h2 : monotone_decreasing_on f 0 3)
  (h3 : symmetric_about f 3) :
  f 3.5 < f 1.5 ∧ f 1.5 < f 6.5 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l47_4795


namespace NUMINAMATH_CALUDE_sqrt_ratio_equals_sqrt_five_l47_4750

theorem sqrt_ratio_equals_sqrt_five : 
  Real.sqrt (3^2 + 4^2) / Real.sqrt (4 + 1) = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_sqrt_ratio_equals_sqrt_five_l47_4750


namespace NUMINAMATH_CALUDE_shower_tiles_count_l47_4772

/-- Represents the layout of a shower wall -/
structure WallLayout where
  rectangularTiles : ℕ
  triangularTiles : ℕ
  hexagonalTiles : ℕ
  squareTiles : ℕ

/-- Calculates the total number of tiles in the shower -/
def totalTiles (wall1 wall2 wall3 : WallLayout) : ℕ :=
  wall1.rectangularTiles + wall1.triangularTiles +
  wall2.rectangularTiles + wall2.triangularTiles + wall2.hexagonalTiles +
  wall3.squareTiles + wall3.triangularTiles

/-- Theorem stating the total number of tiles in the shower -/
theorem shower_tiles_count :
  let wall1 : WallLayout := ⟨12 * 30, 150, 0, 0⟩
  let wall2 : WallLayout := ⟨14, 0, 5 * 6, 0⟩
  let wall3 : WallLayout := ⟨0, 150, 0, 40⟩
  totalTiles wall1 wall2 wall3 = 744 := by
  sorry

end NUMINAMATH_CALUDE_shower_tiles_count_l47_4772


namespace NUMINAMATH_CALUDE_point_not_on_line_l47_4740

theorem point_not_on_line (p q : ℝ) (h : p * q > 0) :
  ¬(∃ (x y : ℝ), x = 2023 ∧ y = 0 ∧ y = p * x + q) :=
by sorry

end NUMINAMATH_CALUDE_point_not_on_line_l47_4740


namespace NUMINAMATH_CALUDE_graph_number_example_intersection_condition_l47_4720

-- Define the "graph number" type
def GraphNumber := ℝ × ℝ × ℝ

-- Define a function to get the graph number of a quadratic function
def getGraphNumber (a b c : ℝ) : GraphNumber :=
  (a, b, c)

-- Define a function to check if a quadratic function intersects x-axis at one point
def intersectsAtOnePoint (a b c : ℝ) : Prop :=
  b^2 - 4*a*c = 0

-- Theorem 1: The graph number of y = (1/3)x^2 - x - 1
theorem graph_number_example : getGraphNumber (1/3) (-1) (-1) = (1/3, -1, -1) := by
  sorry

-- Theorem 2: For [m, m+1, m+1] intersecting x-axis at one point, m = -1 or m = 1/3
theorem intersection_condition (m : ℝ) :
  intersectsAtOnePoint m (m+1) (m+1) → m = -1 ∨ m = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_graph_number_example_intersection_condition_l47_4720


namespace NUMINAMATH_CALUDE_soccer_balls_added_l47_4759

theorem soccer_balls_added (initial_balls final_balls : ℕ) (h1 : initial_balls = 6) (h2 : final_balls = 24) :
  final_balls - initial_balls = 18 := by
  sorry

end NUMINAMATH_CALUDE_soccer_balls_added_l47_4759


namespace NUMINAMATH_CALUDE_max_circumference_circle_in_parabola_l47_4758

/-- A circle located inside the parabola x^2 = 4y and passing through its vertex -/
structure CircleInParabola where
  center : ℝ × ℝ
  radius : ℝ
  inside_parabola : ∀ (x y : ℝ), (x - center.1)^2 + (y - center.2)^2 ≤ radius^2 → x^2 ≤ 4*y
  passes_through_vertex : (0 - center.1)^2 + (0 - center.2)^2 = radius^2

/-- The maximum circumference of a circle located inside the parabola x^2 = 4y 
    and passing through its vertex is 4π -/
theorem max_circumference_circle_in_parabola :
  ∃ (C : CircleInParabola), ∀ (D : CircleInParabola), 2 * Real.pi * C.radius ≥ 2 * Real.pi * D.radius ∧
  2 * Real.pi * C.radius = 4 * Real.pi :=
sorry

end NUMINAMATH_CALUDE_max_circumference_circle_in_parabola_l47_4758


namespace NUMINAMATH_CALUDE_count_arithmetic_mean_subsets_l47_4782

/-- The number of three-element subsets of {1, 2, ..., n} where one element
    is the arithmetic mean of the other two. -/
def arithmeticMeanSubsets (n : ℕ) : ℕ :=
  (n / 2) * ((n - 1) / 2)

/-- Theorem stating that for any natural number n ≥ 3, the number of three-element
    subsets of {1, 2, ..., n} where one element is the arithmetic mean of the
    other two is equal to ⌊n/2⌋ * ⌊(n-1)/2⌋. -/
theorem count_arithmetic_mean_subsets (n : ℕ) (h : n ≥ 3) :
  arithmeticMeanSubsets n = (n / 2) * ((n - 1) / 2) := by
  sorry

#check count_arithmetic_mean_subsets

end NUMINAMATH_CALUDE_count_arithmetic_mean_subsets_l47_4782


namespace NUMINAMATH_CALUDE_dave_total_wage_l47_4709

/-- Represents the daily wage information --/
structure DailyWage where
  hourly_rate : ℕ
  hours_worked : ℕ

/-- Calculates the total wage for a given day --/
def daily_total (dw : DailyWage) : ℕ :=
  dw.hourly_rate * dw.hours_worked

/-- Dave's wage information for Monday to Thursday --/
def dave_wages : List DailyWage := [
  ⟨6, 6⟩,  -- Monday
  ⟨7, 2⟩,  -- Tuesday
  ⟨9, 3⟩,  -- Wednesday
  ⟨8, 5⟩   -- Thursday
]

theorem dave_total_wage :
  (dave_wages.map daily_total).sum = 117 := by
  sorry

#eval (dave_wages.map daily_total).sum

end NUMINAMATH_CALUDE_dave_total_wage_l47_4709


namespace NUMINAMATH_CALUDE_restaurant_group_l47_4733

/-- Proves the number of kids in a group given the total number of people, 
    adult meal cost, and total cost. -/
theorem restaurant_group (total_people : ℕ) (adult_meal_cost : ℕ) (total_cost : ℕ) 
  (h1 : total_people = 9)
  (h2 : adult_meal_cost = 2)
  (h3 : total_cost = 14) :
  ∃ (num_kids : ℕ), 
    num_kids = total_people - (total_cost / adult_meal_cost) ∧ 
    num_kids = 2 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_group_l47_4733


namespace NUMINAMATH_CALUDE_total_cost_after_discount_l47_4763

def child_ticket_price : ℚ := 4.25
def adult_ticket_price : ℚ := child_ticket_price + 3.5
def senior_ticket_price : ℚ := adult_ticket_price - 1.75
def discount_per_5_tickets : ℚ := 3
def num_adult_tickets : ℕ := 2
def num_child_tickets : ℕ := 4
def num_senior_tickets : ℕ := 1

def total_ticket_cost : ℚ :=
  num_adult_tickets * adult_ticket_price +
  num_child_tickets * child_ticket_price +
  num_senior_tickets * senior_ticket_price

def total_tickets : ℕ := num_adult_tickets + num_child_tickets + num_senior_tickets

def discount_amount : ℚ := (total_tickets / 5 : ℚ) * discount_per_5_tickets

theorem total_cost_after_discount :
  total_ticket_cost - discount_amount = 35.5 := by sorry

end NUMINAMATH_CALUDE_total_cost_after_discount_l47_4763


namespace NUMINAMATH_CALUDE_hiking_route_length_l47_4783

/-- The total length of the hiking route in kilometers. -/
def total_length : ℝ := 150

/-- The initial distance walked on foot in kilometers. -/
def initial_walk : ℝ := 30

/-- The fraction of the remaining route traveled by raft. -/
def raft_fraction : ℝ := 0.2

/-- The multiplier for the second walking distance compared to the raft distance. -/
def second_walk_multiplier : ℝ := 1.5

/-- The speed of the truck in km/h. -/
def truck_speed : ℝ := 40

/-- The time spent on the truck in hours. -/
def truck_time : ℝ := 1.5

theorem hiking_route_length :
  initial_walk +
  raft_fraction * (total_length - initial_walk) +
  second_walk_multiplier * (raft_fraction * (total_length - initial_walk)) +
  truck_speed * truck_time = total_length := by sorry

end NUMINAMATH_CALUDE_hiking_route_length_l47_4783


namespace NUMINAMATH_CALUDE_equilateral_triangle_product_l47_4712

/-- Given that (0,0), (a,11), and (b,37) form an equilateral triangle, prove that ab = 315 -/
theorem equilateral_triangle_product (a b : ℝ) : 
  (Complex.I ^ 2 = -1) →
  ((a + 11 * Complex.I) * (Complex.exp (Complex.I * Real.pi / 3)) = b + 37 * Complex.I) →
  a * b = 315 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_product_l47_4712
