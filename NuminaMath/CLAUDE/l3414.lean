import Mathlib

namespace NUMINAMATH_CALUDE_common_factor_proof_l3414_341484

theorem common_factor_proof (x y a b : ℝ) :
  ∃ (k : ℝ), 3*x*(a - b) - 9*y*(b - a) = 3*(a - b) * k :=
by sorry

end NUMINAMATH_CALUDE_common_factor_proof_l3414_341484


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_l3414_341401

/-- An isosceles triangle with perimeter 11 and one side length 3 -/
structure IsoscelesTriangle where
  /-- The length of two equal sides -/
  side : ℝ
  /-- The length of the base -/
  base : ℝ
  /-- The triangle is isosceles -/
  isIsosceles : side ≥ 0 ∧ base ≥ 0
  /-- The perimeter is 11 -/
  perimeterIs11 : 2 * side + base = 11
  /-- One side length is 3 -/
  oneSideIs3 : side = 3 ∨ base = 3

/-- The base of an isosceles triangle with perimeter 11 and one side length 3 can only be 3 or 5 -/
theorem isosceles_triangle_base (t : IsoscelesTriangle) : t.base = 3 ∨ t.base = 5 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_l3414_341401


namespace NUMINAMATH_CALUDE_arithmetic_sequence_before_four_l3414_341431

/-- An arithmetic sequence with first term a and common difference d -/
def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ := a + (n - 1) * d

theorem arithmetic_sequence_before_four :
  ∀ n : ℕ, n ≤ 30 → arithmetic_sequence 92 (-3) n > 4 ∧
  arithmetic_sequence 92 (-3) 31 ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_before_four_l3414_341431


namespace NUMINAMATH_CALUDE_initial_paint_amount_l3414_341421

/-- The amount of paint Jimin used for his house -/
def paint_for_house : ℝ := 4.3

/-- The amount of paint Jimin used for his friend's house -/
def paint_for_friend : ℝ := 4.3

/-- The amount of paint remaining after painting both houses -/
def paint_remaining : ℝ := 8.8

/-- The initial amount of paint Jimin had -/
def initial_paint : ℝ := paint_for_house + paint_for_friend + paint_remaining

theorem initial_paint_amount : initial_paint = 17.4 := by sorry

end NUMINAMATH_CALUDE_initial_paint_amount_l3414_341421


namespace NUMINAMATH_CALUDE_probability_sum_seven_is_one_sixth_l3414_341481

/-- The number of faces on each cubic die -/
def dice_faces : ℕ := 6

/-- The number of ways to obtain a sum of 7 -/
def favorable_outcomes : ℕ := 6

/-- The probability of obtaining a sum of 7 when throwing two cubic dice -/
def probability_sum_seven : ℚ := favorable_outcomes / (dice_faces * dice_faces)

theorem probability_sum_seven_is_one_sixth : 
  probability_sum_seven = 1 / 6 := by
sorry

end NUMINAMATH_CALUDE_probability_sum_seven_is_one_sixth_l3414_341481


namespace NUMINAMATH_CALUDE_only_99th_statement_true_l3414_341488

/-- Represents a statement in the notebook -/
def Statement (n : ℕ) := "There are exactly n false statements in this notebook"

/-- The total number of statements in the notebook -/
def totalStatements : ℕ := 100

/-- A function that determines if a statement is true -/
def isTrue (n : ℕ) : Prop := 
  n ≤ totalStatements ∧ (totalStatements - n) = 1

theorem only_99th_statement_true : 
  ∃! n : ℕ, n ≤ totalStatements ∧ isTrue n ∧ n = 99 := by
  sorry

#check only_99th_statement_true

end NUMINAMATH_CALUDE_only_99th_statement_true_l3414_341488


namespace NUMINAMATH_CALUDE_exact_pairing_l3414_341420

/-- The number of workers processing large gears to match pairs exactly -/
def workers_large_gears : ℕ := 18

/-- The total number of workers in the workshop -/
def total_workers : ℕ := 34

/-- The number of large gears processed by one worker per day -/
def large_gears_per_worker : ℕ := 20

/-- The number of small gears processed by one worker per day -/
def small_gears_per_worker : ℕ := 15

/-- The number of large gears in a pair -/
def large_gears_per_pair : ℕ := 3

/-- The number of small gears in a pair -/
def small_gears_per_pair : ℕ := 2

theorem exact_pairing :
  workers_large_gears * large_gears_per_worker * small_gears_per_pair =
  (total_workers - workers_large_gears) * small_gears_per_worker * large_gears_per_pair :=
by sorry

end NUMINAMATH_CALUDE_exact_pairing_l3414_341420


namespace NUMINAMATH_CALUDE_trajectory_is_ellipse_l3414_341422

/-- Circle represented by its center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Definition of circle M -/
def circle_M : Circle :=
  { center := (-1, 0), radius := 1 }

/-- Definition of circle N -/
def circle_N : Circle :=
  { center := (1, 0), radius := 5 }

/-- Definition of external tangency -/
def is_externally_tangent (c1 c2 : Circle) : Prop :=
  (c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2 = (c1.radius + c2.radius)^2

/-- Definition of internal tangency -/
def is_internally_tangent (c1 c2 : Circle) : Prop :=
  (c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2 = (c1.radius - c2.radius)^2

/-- Theorem: The trajectory of the center of circle P is an ellipse -/
theorem trajectory_is_ellipse (P : ℝ × ℝ) :
  is_externally_tangent { center := P, radius := 0 } circle_M →
  is_internally_tangent { center := P, radius := 0 } circle_N →
  P.1^2 / 9 + P.2^2 / 8 = 1 := by sorry

end NUMINAMATH_CALUDE_trajectory_is_ellipse_l3414_341422


namespace NUMINAMATH_CALUDE_girls_count_in_classroom_l3414_341439

theorem girls_count_in_classroom (ratio_girls : ℕ) (ratio_boys : ℕ) 
  (total_count : ℕ) (h1 : ratio_girls = 4) (h2 : ratio_boys = 3) 
  (h3 : total_count = 43) :
  (ratio_girls * total_count - ratio_girls) / (ratio_girls + ratio_boys) = 24 := by
  sorry

end NUMINAMATH_CALUDE_girls_count_in_classroom_l3414_341439


namespace NUMINAMATH_CALUDE_inequality_proof_l3414_341498

theorem inequality_proof (a b c : ℝ) (h1 : a > b) (h2 : b > c) : a - c > b - c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3414_341498


namespace NUMINAMATH_CALUDE_chloe_win_prob_is_25_91_l3414_341480

/-- Represents the probability of rolling a specific number on a six-sided die -/
def roll_probability : ℚ := 1 / 6

/-- Represents the probability of not rolling a '6' on a six-sided die -/
def not_six_probability : ℚ := 5 / 6

/-- Calculates the probability of Chloe winning on her nth turn -/
def chloe_win_nth_turn (n : ℕ) : ℚ :=
  (not_six_probability ^ (3 * n - 1)) * roll_probability

/-- Calculates the sum of the geometric series representing Chloe's win probability -/
def chloe_win_probability : ℚ :=
  (chloe_win_nth_turn 1) / (1 - (not_six_probability ^ 3))

/-- Theorem stating that the probability of Chloe winning is 25/91 -/
theorem chloe_win_prob_is_25_91 : chloe_win_probability = 25 / 91 := by
  sorry

end NUMINAMATH_CALUDE_chloe_win_prob_is_25_91_l3414_341480


namespace NUMINAMATH_CALUDE_perfect_square_fraction_l3414_341405

theorem perfect_square_fraction (m n : ℕ+) : 
  ∃ k : ℕ, (m + n : ℝ)^2 / (4 * (m : ℝ) * (m - n : ℝ)^2 + 4) = (k : ℝ)^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_fraction_l3414_341405


namespace NUMINAMATH_CALUDE_sum_of_parts_l3414_341408

theorem sum_of_parts (x y : ℝ) (h1 : x + y = 56) (h2 : y = 37.66666666666667) :
  10 * x + 22 * y = 1012 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_parts_l3414_341408


namespace NUMINAMATH_CALUDE_initial_players_correct_l3414_341456

/-- The initial number of players in a video game -/
def initial_players : ℕ := 8

/-- The number of players who quit the game -/
def players_quit : ℕ := 3

/-- The number of lives each remaining player has -/
def lives_per_player : ℕ := 3

/-- The total number of lives after some players quit -/
def total_lives : ℕ := 15

/-- Theorem stating that the initial number of players is correct -/
theorem initial_players_correct : 
  lives_per_player * (initial_players - players_quit) = total_lives :=
by sorry

end NUMINAMATH_CALUDE_initial_players_correct_l3414_341456


namespace NUMINAMATH_CALUDE_rectangular_distance_problem_l3414_341464

-- Define the rectangular distance function
def rectangular_distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  |x₁ - x₂| + |y₁ - y₂|

-- Define points A, O, and B
def A : ℝ × ℝ := (-1, 3)
def O : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (1, 0)

-- Define the line equation
def on_line (x y : ℝ) : Prop :=
  x - y + 2 = 0

theorem rectangular_distance_problem :
  (rectangular_distance A.1 A.2 O.1 O.2 = 4) ∧
  (∃ min_dist : ℝ, min_dist = 3 ∧
    ∀ x y : ℝ, on_line x y →
      rectangular_distance B.1 B.2 x y ≥ min_dist) :=
by sorry

end NUMINAMATH_CALUDE_rectangular_distance_problem_l3414_341464


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3414_341444

-- Define set A
def A : Set ℝ := {y | ∃ x, y = x^2 - 1}

-- Define set B
def B : Set ℝ := {x | |x^2 - 1| ≤ 3}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = Set.Icc (-1) 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3414_341444


namespace NUMINAMATH_CALUDE_regular_hexagon_side_length_l3414_341419

/-- The length of a side in a regular hexagon given the distance between opposite sides -/
theorem regular_hexagon_side_length (d : ℝ) (h : d = 20) : 
  let s := d * 2 / Real.sqrt 3
  s = 40 * Real.sqrt 3 / 3 := by sorry

end NUMINAMATH_CALUDE_regular_hexagon_side_length_l3414_341419


namespace NUMINAMATH_CALUDE_hemisphere_diameter_l3414_341438

-- Define the cube
def cube_side_length : ℝ := 2

-- Define the hemisphere properties
structure Hemisphere where
  center : ℝ × ℝ × ℝ
  radius : ℝ

-- Define the cube with hemispheres
structure CubeWithHemispheres where
  side_length : ℝ
  hemispheres : List Hemisphere
  hemispheres_touch : Bool

-- Theorem statement
theorem hemisphere_diameter (cube : CubeWithHemispheres) 
  (h1 : cube.side_length = cube_side_length)
  (h2 : cube.hemispheres.length = 6)
  (h3 : cube.hemispheres_touch = true) :
  ∀ h ∈ cube.hemispheres, 2 * h.radius = Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_hemisphere_diameter_l3414_341438


namespace NUMINAMATH_CALUDE_fraction_meaningful_condition_l3414_341477

theorem fraction_meaningful_condition (x : ℝ) : 
  (∃ y : ℝ, y = x / (x + 1)) ↔ x ≠ -1 := by sorry

end NUMINAMATH_CALUDE_fraction_meaningful_condition_l3414_341477


namespace NUMINAMATH_CALUDE_expression_evaluation_l3414_341448

theorem expression_evaluation (x : ℝ) (h1 : x^6 ≠ -1) (h2 : x^6 ≠ 1) :
  ((((x^2 + 1)^2 * (x^4 - x^2 + 1)^2) / (x^6 + 1)^2)^2 *
   (((x^2 - 1)^2 * (x^4 + x^2 + 1)^2) / (x^6 - 1)^2)^2) = 1 := by
  sorry


end NUMINAMATH_CALUDE_expression_evaluation_l3414_341448


namespace NUMINAMATH_CALUDE_sum_of_m_values_is_correct_l3414_341414

/-- The sum of all possible values of m for which the polynomials x^2 - 6x + 8 and x^2 - 7x + m have a root in common -/
def sum_of_m_values : ℝ := 22

/-- First polynomial: x^2 - 6x + 8 -/
def p1 (x : ℝ) : ℝ := x^2 - 6*x + 8

/-- Second polynomial: x^2 - 7x + m -/
def p2 (x m : ℝ) : ℝ := x^2 - 7*x + m

/-- Theorem stating that the sum of all possible values of m for which p1 and p2 have a common root is equal to sum_of_m_values -/
theorem sum_of_m_values_is_correct : 
  (∃ m1 m2 : ℝ, m1 ≠ m2 ∧ 
    (∃ x1 : ℝ, p1 x1 = 0 ∧ p2 x1 m1 = 0) ∧
    (∃ x2 : ℝ, p1 x2 = 0 ∧ p2 x2 m2 = 0) ∧
    m1 + m2 = sum_of_m_values) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_m_values_is_correct_l3414_341414


namespace NUMINAMATH_CALUDE_smallest_number_l3414_341447

theorem smallest_number (s : Set ℚ) (h : s = {-1, 0, -3, -2}) : 
  ∃ x ∈ s, ∀ y ∈ s, x ≤ y ∧ x = -3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_l3414_341447


namespace NUMINAMATH_CALUDE_opinion_change_difference_l3414_341494

theorem opinion_change_difference (initial_like initial_dislike final_like final_dislike : ℝ) :
  initial_like = 40 →
  initial_dislike = 60 →
  final_like = 80 →
  final_dislike = 20 →
  initial_like + initial_dislike = 100 →
  final_like + final_dislike = 100 →
  let min_change := |final_like - initial_like|
  let max_change := min initial_like initial_dislike + min final_like final_dislike
  max_change - min_change = 60 := by
sorry

end NUMINAMATH_CALUDE_opinion_change_difference_l3414_341494


namespace NUMINAMATH_CALUDE_first_number_is_55_l3414_341452

def number_list : List ℕ := [55, 57, 58, 59, 62, 62, 63, 65, 65]

theorem first_number_is_55 (average_is_60 : (number_list.sum / number_list.length : ℚ) = 60) :
  number_list.head? = some 55 := by
  sorry

end NUMINAMATH_CALUDE_first_number_is_55_l3414_341452


namespace NUMINAMATH_CALUDE_nice_polynomial_characterization_l3414_341430

def is_nice (f : ℝ → ℝ) (A B : Finset ℝ) : Prop :=
  A.card = B.card ∧ B = A.image f

def can_produce_nice (S : ℝ → ℝ) : Prop :=
  ∀ A B : Finset ℝ, A.card = B.card → ∃ f : ℝ → ℝ, is_nice f A B

def is_polynomial (f : ℝ → ℝ) : Prop := sorry

def degree (f : ℝ → ℝ) : ℕ := sorry

def leading_coefficient (f : ℝ → ℝ) : ℝ := sorry

theorem nice_polynomial_characterization (S : ℝ → ℝ) :
  (is_polynomial S ∧ can_produce_nice S) ↔
  (is_polynomial S ∧ degree S ≥ 2 ∧
   (Even (degree S) ∨ (Odd (degree S) ∧ leading_coefficient S < 0))) :=
sorry

end NUMINAMATH_CALUDE_nice_polynomial_characterization_l3414_341430


namespace NUMINAMATH_CALUDE_properties_of_negative_2010_l3414_341428

theorem properties_of_negative_2010 :
  let n : ℤ := -2010
  (1 / n = 1 / -2010) ∧
  (-n = 2010) ∧
  (abs n = 2010) ∧
  (-(1 / n) = 1 / 2010) := by
sorry

end NUMINAMATH_CALUDE_properties_of_negative_2010_l3414_341428


namespace NUMINAMATH_CALUDE_well_diameter_l3414_341412

theorem well_diameter (depth : ℝ) (volume : ℝ) (diameter : ℝ) : 
  depth = 14 →
  volume = 43.982297150257104 →
  volume = Real.pi * (diameter / 2)^2 * depth →
  diameter = 2 := by
sorry

end NUMINAMATH_CALUDE_well_diameter_l3414_341412


namespace NUMINAMATH_CALUDE_parallel_unit_vectors_l3414_341493

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

def is_unit_vector (v : E) : Prop := ‖v‖ = 1

def are_parallel (v w : E) : Prop := ∃ (k : ℝ), v = k • w

theorem parallel_unit_vectors (a b : E) 
  (ha : is_unit_vector a) (hb : is_unit_vector b) (hpar : are_parallel a b) : 
  a = b ∨ a = -b := by
  sorry

end NUMINAMATH_CALUDE_parallel_unit_vectors_l3414_341493


namespace NUMINAMATH_CALUDE_wine_purchase_problem_l3414_341457

theorem wine_purchase_problem :
  ∃ (x y n m : ℕ), 
    5 * x + 8 * y = n ^ 2 ∧
    n ^ 2 + 60 = m ^ 2 ∧
    x + y = m :=
by sorry

end NUMINAMATH_CALUDE_wine_purchase_problem_l3414_341457


namespace NUMINAMATH_CALUDE_shaded_area_sum_l3414_341479

/-- Represents the shaded area in each level of the square division pattern -/
def shadedAreaSeries : ℕ → ℚ
  | 0 => 1/4
  | n+1 => (1/4) * shadedAreaSeries n

/-- The sum of the infinite geometric series representing the total shaded area -/
def totalShadedArea : ℚ := 1/3

/-- Theorem stating that the sum of the infinite geometric series is 1/3 -/
theorem shaded_area_sum : 
  (∑' n, shadedAreaSeries n) = totalShadedArea := by
  sorry

#check shaded_area_sum

end NUMINAMATH_CALUDE_shaded_area_sum_l3414_341479


namespace NUMINAMATH_CALUDE_cube_root_of_sum_l3414_341443

theorem cube_root_of_sum (a b : ℝ) : 
  Real.sqrt (a - 1) + Real.sqrt ((9 + b)^2) = 0 → (a + b)^(1/3 : ℝ) = -2 :=
by sorry

end NUMINAMATH_CALUDE_cube_root_of_sum_l3414_341443


namespace NUMINAMATH_CALUDE_number_times_five_equals_hundred_l3414_341460

theorem number_times_five_equals_hundred (x : ℝ) : x * 5 = 100 → x = 20 := by
  sorry

end NUMINAMATH_CALUDE_number_times_five_equals_hundred_l3414_341460


namespace NUMINAMATH_CALUDE_rhombus_area_fraction_l3414_341441

theorem rhombus_area_fraction (grid_size : ℕ) (rhombus_side : ℝ) :
  grid_size = 7 →
  rhombus_side = Real.sqrt 2 →
  (4 * (1/2 * rhombus_side * rhombus_side)) / ((grid_size - 1)^2) = 1/18 :=
by sorry

end NUMINAMATH_CALUDE_rhombus_area_fraction_l3414_341441


namespace NUMINAMATH_CALUDE_second_number_value_l3414_341467

theorem second_number_value (x : ℚ) 
  (sum_condition : 2*x + x + (2/3)*x + (1/2)*x = 330) : x = 46 := by
  sorry

end NUMINAMATH_CALUDE_second_number_value_l3414_341467


namespace NUMINAMATH_CALUDE_function_properties_l3414_341495

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def symmetric_about (f : ℝ → ℝ) (a : ℝ) : Prop := ∀ x, f (a - x) = f (a + x)

def has_period (f : ℝ → ℝ) (T : ℝ) : Prop := ∀ x, f (x + T) = f x

theorem function_properties (f : ℝ → ℝ) :
  (is_even f ∧ symmetric_about f 1 → has_period f 2) ∧
  (symmetric_about f 1 ∧ has_period f 2 → is_even f) ∧
  (is_even f ∧ has_period f 2 → symmetric_about f 1) →
  is_even f ∧ symmetric_about f 1 ∧ has_period f 2 :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l3414_341495


namespace NUMINAMATH_CALUDE_circle_center_sum_l3414_341415

/-- Given a circle with equation x^2 + y^2 = 4x - 6y + 9, 
    the sum of the coordinates of its center is -1. -/
theorem circle_center_sum (x y : ℝ) : 
  (x^2 + y^2 = 4*x - 6*y + 9) → (∃ h k : ℝ, (x - h)^2 + (y - k)^2 = (h^2 + k^2 - 9) ∧ h + k = -1) :=
by sorry

end NUMINAMATH_CALUDE_circle_center_sum_l3414_341415


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3414_341423

def A : Set ℝ := {x | x + 2 > 0}
def B : Set ℝ := {-3, -2, -1, 0}

theorem intersection_of_A_and_B : A ∩ B = {-1, 0} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3414_341423


namespace NUMINAMATH_CALUDE_parallel_angles_theorem_l3414_341470

theorem parallel_angles_theorem (α β : Real) :
  (∃ k : ℤ, α + β = k * 180) →  -- Parallel sides condition
  (α = 3 * β - 36) →            -- Relationship between α and β
  (α = 18 ∨ α = 126) :=         -- Conclusion
by sorry

end NUMINAMATH_CALUDE_parallel_angles_theorem_l3414_341470


namespace NUMINAMATH_CALUDE_count_convex_cyclic_quads_l3414_341434

/-- A convex cyclic quadrilateral with integer sides --/
structure ConvexCyclicQuad where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  sum_eq_40 : a + b + c + d = 40
  convex : a < b + c + d ∧ b < a + c + d ∧ c < a + b + d ∧ d < a + b + c
  ordered : a ≥ b ∧ b ≥ c ∧ c ≥ d
  has_odd_side : Odd a ∨ Odd b ∨ Odd c ∨ Odd d

/-- The count of valid quadrilaterals --/
def count_valid_quads : ℕ := sorry

theorem count_convex_cyclic_quads : count_valid_quads = 760 := by
  sorry

end NUMINAMATH_CALUDE_count_convex_cyclic_quads_l3414_341434


namespace NUMINAMATH_CALUDE_train_crossing_bridge_time_train_crossing_bridge_time_is_35_seconds_l3414_341491

/-- The time required for a train to cross a bridge -/
theorem train_crossing_bridge_time 
  (train_length : ℝ) 
  (bridge_length : ℝ) 
  (train_speed_kmph : ℝ) 
  (h1 : train_length = 200) 
  (h2 : bridge_length = 150) 
  (h3 : train_speed_kmph = 36) : ℝ :=
let train_speed_mps := train_speed_kmph * (5/18)
let total_distance := train_length + bridge_length
let time := total_distance / train_speed_mps
35

theorem train_crossing_bridge_time_is_35_seconds 
  (train_length : ℝ) 
  (bridge_length : ℝ) 
  (train_speed_kmph : ℝ) 
  (h1 : train_length = 200) 
  (h2 : bridge_length = 150) 
  (h3 : train_speed_kmph = 36) : 
  train_crossing_bridge_time train_length bridge_length train_speed_kmph h1 h2 h3 = 35 := by
sorry

end NUMINAMATH_CALUDE_train_crossing_bridge_time_train_crossing_bridge_time_is_35_seconds_l3414_341491


namespace NUMINAMATH_CALUDE_discount_percentage_l3414_341489

theorem discount_percentage (initial_amount : ℝ) 
  (h1 : initial_amount = 500)
  (h2 : ∃ (needed_before_discount : ℝ), needed_before_discount = initial_amount + 2/5 * initial_amount)
  (h3 : ∃ (amount_still_needed : ℝ), amount_still_needed = 95) : 
  ∃ (discount_percentage : ℝ), discount_percentage = 15 := by
sorry

end NUMINAMATH_CALUDE_discount_percentage_l3414_341489


namespace NUMINAMATH_CALUDE_cube_surface_area_increase_l3414_341455

/-- Represents a cube with a given side length -/
structure Cube where
  side_length : ℝ

/-- Calculates the surface area of a cube -/
def surface_area (c : Cube) : ℝ :=
  6 * c.side_length * c.side_length

/-- Calculates the increase in surface area after making cuts -/
def surface_area_increase (c : Cube) (num_cuts : ℕ) : ℝ :=
  2 * c.side_length * c.side_length * num_cuts

/-- Theorem: The increase in surface area of a 10 cm cube after three cuts is 600 cm² -/
theorem cube_surface_area_increase :
  let c := Cube.mk 10
  surface_area_increase c 3 = 600 := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_increase_l3414_341455


namespace NUMINAMATH_CALUDE_regular_decagon_diagonal_side_difference_l3414_341458

/-- In a regular decagon inscribed in a circle, the difference between the length of the diagonal 
    connecting vertices 3 apart and the side length is equal to the radius of the circumcircle. -/
theorem regular_decagon_diagonal_side_difference (R : ℝ) : 
  let side_length := 2 * R * Real.sin (π / 10)
  let diagonal_length := 2 * R * Real.sin (3 * π / 10)
  diagonal_length - side_length = R := by sorry

end NUMINAMATH_CALUDE_regular_decagon_diagonal_side_difference_l3414_341458


namespace NUMINAMATH_CALUDE_profit_distribution_l3414_341445

/-- The profit distribution problem -/
theorem profit_distribution (total_profit : ℝ) (ratio : List ℕ) : 
  total_profit = 34000 →
  ratio = [2, 3, 4, 1, 5] →
  (ratio.sum : ℝ) * (ratio.maximum.get!) * total_profit / (ratio.sum ^ 2) = 11333.35 := by
  sorry

end NUMINAMATH_CALUDE_profit_distribution_l3414_341445


namespace NUMINAMATH_CALUDE_milk_delivery_theorem_l3414_341483

/-- Calculates the number of jars of milk good for sale given the delivery conditions --/
def goodJarsForSale (
  normalDelivery : ℕ
  ) (jarsPerCarton : ℕ
  ) (cartonShortage : ℕ
  ) (damagedJarsPerCarton : ℕ
  ) (cartonsWithDamagedJars : ℕ
  ) (totallyDamagedCartons : ℕ
  ) : ℕ :=
  let deliveredCartons := normalDelivery - cartonShortage
  let totalJars := deliveredCartons * jarsPerCarton
  let damagedJars := damagedJarsPerCarton * cartonsWithDamagedJars + totallyDamagedCartons * jarsPerCarton
  totalJars - damagedJars

/-- Theorem stating that under the given conditions, there are 565 jars of milk good for sale --/
theorem milk_delivery_theorem :
  goodJarsForSale 50 20 20 3 5 1 = 565 := by
  sorry

end NUMINAMATH_CALUDE_milk_delivery_theorem_l3414_341483


namespace NUMINAMATH_CALUDE_children_fed_theorem_l3414_341454

/-- Represents the number of people a meal can feed -/
structure MealCapacity where
  adults : ℕ
  children : ℕ

/-- Calculates the number of children that can be fed with the remaining food -/
def remainingChildrenFed (totalAdults totalChildren consumedAdultMeals : ℕ) (capacity : MealCapacity) : ℕ :=
  let remainingAdultMeals := capacity.adults - consumedAdultMeals
  let childrenPerAdultMeal := capacity.children / capacity.adults
  remainingAdultMeals * childrenPerAdultMeal

/-- Theorem stating that given the conditions, 63 children can be fed with the remaining food -/
theorem children_fed_theorem (totalAdults totalChildren consumedAdultMeals : ℕ) (capacity : MealCapacity) :
  totalAdults = 55 →
  totalChildren = 70 →
  capacity.adults = 70 →
  capacity.children = 90 →
  consumedAdultMeals = 21 →
  remainingChildrenFed totalAdults totalChildren consumedAdultMeals capacity = 63 := by
  sorry

end NUMINAMATH_CALUDE_children_fed_theorem_l3414_341454


namespace NUMINAMATH_CALUDE_sphere_speeds_solution_l3414_341451

/-- Represents the speeds of two spheres moving towards the vertex of a right angle --/
structure SphereSpeeds where
  small : ℝ
  large : ℝ

/-- The problem setup and conditions --/
def sphereProblem (s : SphereSpeeds) : Prop :=
  let r₁ := 2 -- radius of smaller sphere
  let r₂ := 3 -- radius of larger sphere
  let d₁ := 6 -- initial distance of smaller sphere from vertex
  let d₂ := 16 -- initial distance of larger sphere from vertex
  let t₁ := 1 -- time after which distance between centers is measured
  let t₂ := 3 -- time at which spheres collide
  -- Initial positions
  (d₁ - s.small * t₁) ^ 2 + (d₂ - s.large * t₁) ^ 2 = 13 ^ 2 ∧
  -- Collision positions
  (d₁ - s.small * t₂) ^ 2 + (d₂ - s.large * t₂) ^ 2 = (r₁ + r₂) ^ 2

/-- The theorem stating the solution to the sphere problem --/
theorem sphere_speeds_solution :
  ∃ s : SphereSpeeds, sphereProblem s ∧ s.small = 1 ∧ s.large = 4 := by
  sorry

end NUMINAMATH_CALUDE_sphere_speeds_solution_l3414_341451


namespace NUMINAMATH_CALUDE_novel_sales_ratio_l3414_341426

theorem novel_sales_ratio : 
  let total_copies : ℕ := 440000
  let paperback_copies : ℕ := 363600
  let initial_hardback : ℕ := 36000
  let hardback_copies := total_copies - paperback_copies
  let later_hardback := hardback_copies - initial_hardback
  let later_paperback := paperback_copies
  (later_paperback : ℚ) / (later_hardback : ℚ) = 9 / 1 :=
by sorry

end NUMINAMATH_CALUDE_novel_sales_ratio_l3414_341426


namespace NUMINAMATH_CALUDE_unique_solution_xy_l3414_341459

theorem unique_solution_xy (x y : ℕ) :
  x * (x + 1) = 4 * y * (y + 1) → (x = 0 ∧ y = 0) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_xy_l3414_341459


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3414_341406

theorem complex_fraction_simplification :
  let z : ℂ := (5 - 3*I) / (2 - 3*I)
  z = -19/5 - 9/5*I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3414_341406


namespace NUMINAMATH_CALUDE_min_sheets_theorem_l3414_341440

/-- The minimum number of sheets in a pad of paper -/
def min_sheets_in_pad : ℕ := 36

/-- The number of weekdays -/
def weekdays : ℕ := 5

/-- The number of days Evelyn takes off per week -/
def days_off : ℕ := 2

/-- The number of sheets Evelyn uses per working day -/
def sheets_per_day : ℕ := 12

/-- Theorem stating that the minimum number of sheets in a pad of paper is 36 -/
theorem min_sheets_theorem : 
  min_sheets_in_pad = (weekdays - days_off) * sheets_per_day :=
by sorry

end NUMINAMATH_CALUDE_min_sheets_theorem_l3414_341440


namespace NUMINAMATH_CALUDE_tan_simplification_l3414_341471

theorem tan_simplification (α : Real) (h : Real.tan α = 2) :
  (2 * Real.sin α - Real.cos α) / (2 * Real.cos α + 3 * Real.sin α) = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_tan_simplification_l3414_341471


namespace NUMINAMATH_CALUDE_perfect_square_condition_l3414_341410

/-- A trinomial ax^2 + bx + c is a perfect square if there exist real numbers p and q
    such that ax^2 + bx + c = (px + q)^2 for all real x -/
def is_perfect_square_trinomial (a b c : ℝ) : Prop :=
  ∃ p q : ℝ, ∀ x : ℝ, a * x^2 + b * x + c = (p * x + q)^2

/-- If 4x^2 + kx + 9 is a perfect square trinomial, then k = 12 or k = -12 -/
theorem perfect_square_condition (k : ℝ) :
  is_perfect_square_trinomial 4 k 9 → k = 12 ∨ k = -12 :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l3414_341410


namespace NUMINAMATH_CALUDE_trapezoid_area_l3414_341402

-- Define the lengths of the line segments
def a : ℝ := 1
def b : ℝ := 4
def c : ℝ := 4
def d : ℝ := 5

-- Define the possible areas
def area1 : ℝ := 6
def area2 : ℝ := 10

-- Statement of the theorem
theorem trapezoid_area :
  ∃ (S : ℝ), (S = area1 ∨ S = area2) ∧
  (∃ (h1 h2 base1 base2 : ℝ),
    (h1 = b ∧ h2 = c ∧ base1 = a ∧ base2 = d) ∨
    (h1 = b ∧ h2 = d ∧ base1 = a ∧ base2 = c) ∨
    (h1 = c ∧ h2 = d ∧ base1 = a ∧ base2 = b)) ∧
  S = (base1 + base2) * (h1 + h2) / 4 :=
sorry

end NUMINAMATH_CALUDE_trapezoid_area_l3414_341402


namespace NUMINAMATH_CALUDE_product_of_binary_and_ternary_l3414_341487

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.foldr (fun b n => 2 * n + if b then 1 else 0) 0

/-- Converts a ternary number represented as a list of trits to its decimal equivalent -/
def ternary_to_decimal (trits : List ℕ) : ℕ :=
  trits.foldr (fun t n => 3 * n + t) 0

theorem product_of_binary_and_ternary :
  let binary_num := [true, false, true, true]  -- 1011 in binary
  let ternary_num := [1, 1, 1]  -- 111 in ternary
  (binary_to_decimal binary_num) * (ternary_to_decimal ternary_num) = 143 := by
  sorry

end NUMINAMATH_CALUDE_product_of_binary_and_ternary_l3414_341487


namespace NUMINAMATH_CALUDE_prob_five_odd_in_six_rolls_l3414_341492

/-- The probability of rolling an odd number on a fair 6-sided die -/
def p_odd : ℚ := 1/2

/-- The number of rolls -/
def n : ℕ := 6

/-- The number of successful outcomes (rolls with odd numbers) -/
def k : ℕ := 5

/-- The probability of getting exactly k odd numbers in n rolls of a fair 6-sided die -/
def prob_k_odd_in_n_rolls (p : ℚ) (n k : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) * p^k * (1 - p)^(n - k)

theorem prob_five_odd_in_six_rolls :
  prob_k_odd_in_n_rolls p_odd n k = 3/32 := by
  sorry

end NUMINAMATH_CALUDE_prob_five_odd_in_six_rolls_l3414_341492


namespace NUMINAMATH_CALUDE_second_store_cars_l3414_341450

-- Define the number of stores
def num_stores : ℕ := 5

-- Define the car counts for known stores
def first_store : ℕ := 30
def third_store : ℕ := 14
def fourth_store : ℕ := 21
def fifth_store : ℕ := 25

-- Define the mean
def mean : ℚ := 20.8

-- Define the theorem
theorem second_store_cars :
  ∃ (second_store : ℕ),
    (first_store + second_store + third_store + fourth_store + fifth_store) / num_stores = mean ∧
    second_store = 14 := by
  sorry

end NUMINAMATH_CALUDE_second_store_cars_l3414_341450


namespace NUMINAMATH_CALUDE_drama_club_neither_math_nor_physics_l3414_341435

theorem drama_club_neither_math_nor_physics 
  (total : ℕ) 
  (math : ℕ) 
  (physics : ℕ) 
  (both : ℕ) 
  (h1 : total = 80) 
  (h2 : math = 50) 
  (h3 : physics = 40) 
  (h4 : both = 25) : 
  total - (math + physics - both) = 15 := by
  sorry

end NUMINAMATH_CALUDE_drama_club_neither_math_nor_physics_l3414_341435


namespace NUMINAMATH_CALUDE_tangent_relation_l3414_341425

theorem tangent_relation (α β : Real) 
  (h : Real.tan (α - β) = Real.sin (2 * β) / (5 - Real.cos (2 * β))) :
  2 * Real.tan α = 3 * Real.tan β := by
  sorry

end NUMINAMATH_CALUDE_tangent_relation_l3414_341425


namespace NUMINAMATH_CALUDE_square_difference_divided_by_nine_l3414_341474

theorem square_difference_divided_by_nine : (121^2 - 112^2) / 9 = 233 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_divided_by_nine_l3414_341474


namespace NUMINAMATH_CALUDE_line_intersects_circle_l3414_341475

/-- The line (x-1)a + y = 1 always intersects the circle x^2 + y^2 = 3 for any real value of a -/
theorem line_intersects_circle (a : ℝ) : ∃ (x y : ℝ), 
  ((x - 1) * a + y = 1) ∧ (x^2 + y^2 = 3) := by sorry

end NUMINAMATH_CALUDE_line_intersects_circle_l3414_341475


namespace NUMINAMATH_CALUDE_quadratic_intersection_l3414_341404

def quadratic (a b : ℝ) (x : ℝ) : ℝ := x^2 + a*x + b

theorem quadratic_intersection (a b c d : ℝ) :
  (∃ x₁ x₂ : ℝ, quadratic a b x₁ = quadratic c d x₁ ∧ 
                quadratic a b x₂ = quadratic c d x₂ ∧ 
                x₁ ≠ x₂) →
  (∀ x : ℝ, quadratic a b (-a/2) ≤ quadratic a b x) →
  (∀ x : ℝ, quadratic c d (-c/2) ≤ quadratic c d x) →
  quadratic a b (-a/2) = -200 →
  quadratic c d (-c/2) = -200 →
  (∃ x : ℝ, quadratic c d x = 0 ∧ (-a/2)^2 = x) →
  (∃ x : ℝ, quadratic a b x = 0 ∧ (-c/2)^2 = x) →
  quadratic a b 150 = -200 →
  quadratic c d 150 = -200 →
  a + c = 300 - 4 * Real.sqrt 350 ∨ a + c = 300 + 4 * Real.sqrt 350 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_intersection_l3414_341404


namespace NUMINAMATH_CALUDE_sum_of_coefficients_in_factorization_l3414_341407

theorem sum_of_coefficients_in_factorization (x y : ℝ) : 
  ∃ (a b c d e f : ℤ), 
    (8 * x^8 - 243 * y^8 = (a * x^2 + b * y^2) * (c * x^2 + d * y^2) * (e * x^4 + f * y^4)) ∧
    (a + b + c + d + e + f = 17) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_in_factorization_l3414_341407


namespace NUMINAMATH_CALUDE_tournament_has_25_players_l3414_341463

/-- Represents a tournament with the given conditions -/
structure Tournament where
  n : ℕ  -- number of players not in the lowest 5
  total_players : ℕ := n + 5
  total_games : ℕ := (total_players * (total_players - 1)) / 2
  points_top_n : ℕ := (n * (n - 1)) / 2
  points_bottom_5 : ℕ := 10

/-- The theorem stating that a tournament satisfying the given conditions must have 25 players -/
theorem tournament_has_25_players (t : Tournament) : t.total_players = 25 := by
  sorry

#check tournament_has_25_players

end NUMINAMATH_CALUDE_tournament_has_25_players_l3414_341463


namespace NUMINAMATH_CALUDE_min_point_is_correct_l3414_341400

/-- The equation of the transformed graph -/
def f (x : ℝ) : ℝ := 2 * |x - 4| - 1

/-- The minimum point of the transformed graph -/
def min_point : ℝ × ℝ := (-4, -1)

/-- Theorem: The minimum point of the transformed graph is (-4, -1) -/
theorem min_point_is_correct :
  ∀ x : ℝ, f x ≥ f (min_point.1) ∧ f (min_point.1) = min_point.2 :=
by sorry

end NUMINAMATH_CALUDE_min_point_is_correct_l3414_341400


namespace NUMINAMATH_CALUDE_apple_distribution_equation_l3414_341427

def represents_apple_distribution (x : ℕ) : Prop :=
  (x - 1) % 3 = 0 ∧ (x + 2) % 4 = 0

theorem apple_distribution_equation :
  ∀ x : ℕ, represents_apple_distribution x ↔ (x - 1) / 3 = (x + 2) / 4 :=
by sorry

end NUMINAMATH_CALUDE_apple_distribution_equation_l3414_341427


namespace NUMINAMATH_CALUDE_product_equals_square_l3414_341437

theorem product_equals_square : 100 * 19.98 * 1.998 * 1000 = (1998 : ℝ)^2 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_square_l3414_341437


namespace NUMINAMATH_CALUDE_percentage_calculation_l3414_341468

theorem percentage_calculation (x y : ℝ) : 
  x = 0.8 * 350 → y = 0.6 * x → 1.2 * y = 201.6 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l3414_341468


namespace NUMINAMATH_CALUDE_part1_part2_l3414_341453

-- Define the function f
def f (a x : ℝ) : ℝ := x^2 - (a-1)*x + a-2

-- Part 1
theorem part1 (a : ℝ) : 
  (∀ x : ℝ, f a x ≥ -2) ↔ (3 - 2*Real.sqrt 2 ≤ a ∧ a ≤ 3 + 2*Real.sqrt 2) :=
sorry

-- Part 2
theorem part2 (a x : ℝ) :
  (a < 3 → (f a x < 0 ↔ a-2 < x ∧ x < 1)) ∧
  (a = 3 → ¬∃ x, f a x < 0) ∧
  (a > 3 → (f a x < 0 ↔ 1 < x ∧ x < a-2)) :=
sorry

end NUMINAMATH_CALUDE_part1_part2_l3414_341453


namespace NUMINAMATH_CALUDE_opposites_sum_l3414_341485

theorem opposites_sum (a b : ℝ) : 
  (|a - 2| = -(b + 5)^2) → (a + b = -3) := by
  sorry

end NUMINAMATH_CALUDE_opposites_sum_l3414_341485


namespace NUMINAMATH_CALUDE_sector_central_angle_l3414_341446

/-- Given a circular sector with perimeter 8 and area 4, 
    prove that its central angle has an absolute value of 2 radians. -/
theorem sector_central_angle (r l θ : ℝ) 
  (h_perimeter : 2 * r + l = 8)
  (h_area : 1/2 * l * r = 4)
  (h_angle : θ = l / r)
  (h_positive : r > 0) :
  |θ| = 2 := by
  sorry

#check sector_central_angle

end NUMINAMATH_CALUDE_sector_central_angle_l3414_341446


namespace NUMINAMATH_CALUDE_that_and_this_percentage_l3414_341478

/-- Proves that "that and this" plus half of "that and this" is 200% of three-quarters of "that and this" -/
theorem that_and_this_percentage : 
  ∀ x : ℝ, x > 0 → (x + 0.5 * x) / (0.75 * x) = 2 := by
  sorry

end NUMINAMATH_CALUDE_that_and_this_percentage_l3414_341478


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_l3414_341472

theorem quadratic_roots_sum (a b : ℝ) : 
  a^2 - 4*a - 1 = 0 → b^2 - 4*b - 1 = 0 → 2*a^2 + 3/b + 5*b = 22 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_l3414_341472


namespace NUMINAMATH_CALUDE_birds_in_tree_l3414_341432

theorem birds_in_tree (initial_birds final_birds : ℕ) : 
  initial_birds = 231 → final_birds = 312 → 
  final_birds - initial_birds = 81 := by sorry

end NUMINAMATH_CALUDE_birds_in_tree_l3414_341432


namespace NUMINAMATH_CALUDE_quadratic_rewrite_l3414_341461

theorem quadratic_rewrite (b : ℕ) (n : ℝ) : 
  (∀ x, x^2 + b*x + 68 = (x + n)^2 + 32) →
  b % 2 = 0 →
  b > 0 →
  b = 12 := by
sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_l3414_341461


namespace NUMINAMATH_CALUDE_vending_machine_probability_l3414_341496

-- Define the number of toys
def num_toys : ℕ := 10

-- Define the cost range of toys
def min_cost : ℚ := 1/2
def max_cost : ℚ := 5

-- Define the cost increment
def cost_increment : ℚ := 1/2

-- Define Sam's initial quarters
def initial_quarters : ℕ := 10

-- Define the cost of Sam's favorite toy
def favorite_toy_cost : ℚ := 3

-- Define the function to calculate toy prices
def toy_price (n : ℕ) : ℚ := min_cost + (n - 1) * cost_increment

-- Define the probability of needing to break the twenty-dollar bill
def prob_break_bill : ℚ := 14/15

-- Theorem statement
theorem vending_machine_probability :
  (∀ n ∈ Finset.range num_toys, toy_price n ≤ max_cost) →
  (∀ n ∈ Finset.range num_toys, toy_price n ≥ min_cost) →
  (∀ n ∈ Finset.range (num_toys - 1), toy_price (n + 1) = toy_price n + cost_increment) →
  (favorite_toy_cost ∈ Finset.image toy_price (Finset.range num_toys)) →
  (initial_quarters * (1/4 : ℚ) < favorite_toy_cost) →
  (prob_break_bill = 14/15) :=
sorry

end NUMINAMATH_CALUDE_vending_machine_probability_l3414_341496


namespace NUMINAMATH_CALUDE_x_equation_solution_l3414_341424

theorem x_equation_solution (x : ℝ) (h : x + 1/x = Real.sqrt 5) :
  x^12 - 7*x^8 + x^4 = 343 := by
  sorry

end NUMINAMATH_CALUDE_x_equation_solution_l3414_341424


namespace NUMINAMATH_CALUDE_exponent_division_l3414_341469

theorem exponent_division (a : ℝ) (h : a ≠ 0) : a^5 / a = a^4 := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l3414_341469


namespace NUMINAMATH_CALUDE_element_14_is_si_l3414_341416

/-- Represents chemical elements -/
inductive Element : Type
| helium : Element
| lithium : Element
| silicon : Element
| argon : Element

/-- Returns the atomic number of an element -/
def atomic_number (e : Element) : ℕ :=
  match e with
  | Element.helium => 2
  | Element.lithium => 3
  | Element.silicon => 14
  | Element.argon => 18

/-- Returns the symbol of an element -/
def symbol (e : Element) : String :=
  match e with
  | Element.helium => "He"
  | Element.lithium => "Li"
  | Element.silicon => "Si"
  | Element.argon => "Ar"

/-- Theorem: The symbol for the element with atomic number 14 is Si -/
theorem element_14_is_si :
  ∃ (e : Element), atomic_number e = 14 ∧ symbol e = "Si" :=
by
  sorry

end NUMINAMATH_CALUDE_element_14_is_si_l3414_341416


namespace NUMINAMATH_CALUDE_binary_11010_is_26_l3414_341486

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (λ acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_11010_is_26 :
  binary_to_decimal [false, true, false, true, true] = 26 := by
  sorry

end NUMINAMATH_CALUDE_binary_11010_is_26_l3414_341486


namespace NUMINAMATH_CALUDE_no_positive_integer_solution_l3414_341476

theorem no_positive_integer_solution : 
  ¬ ∃ (n k : ℕ+), (5 + 3 * Real.sqrt 2) ^ n.val = (3 + 5 * Real.sqrt 2) ^ k.val := by
sorry

end NUMINAMATH_CALUDE_no_positive_integer_solution_l3414_341476


namespace NUMINAMATH_CALUDE_a_has_winning_strategy_l3414_341403

/-- Represents the state of the game board -/
structure GameState where
  primes : List Nat
  product_mod_4 : Nat

/-- Represents a move in the game -/
inductive Move
  | erase_and_write (n : Nat) (erased : List Nat) (written : List Nat)

/-- The game between players A and B -/
def Game :=
  List Move

/-- Checks if a number is an odd prime -/
def is_odd_prime (n : Nat) : Prop :=
  Nat.Prime n ∧ n % 2 = 1

/-- The initial setup of the game -/
def initial_setup (primes : List Nat) : Prop :=
  primes.length = 1000 ∧ ∀ p ∈ primes, is_odd_prime p

/-- B's selection of primes -/
def b_selection (all_primes : List Nat) (selected : List Nat) : Prop :=
  selected.length = 500 ∧ ∀ p ∈ selected, p ∈ all_primes

/-- Applies a move to the game state -/
def apply_move (state : GameState) (move : Move) : GameState :=
  sorry

/-- Checks if a move is valid -/
def is_valid_move (state : GameState) (move : Move) : Prop :=
  sorry

/-- Checks if the game is over (board is empty) -/
def is_game_over (state : GameState) : Prop :=
  state.primes.isEmpty

/-- Player A's winning strategy -/
def a_winning_strategy (game : Game) : Prop :=
  sorry

/-- The main theorem stating that player A has a winning strategy -/
theorem a_has_winning_strategy 
  (initial_primes : List Nat)
  (h_initial : initial_setup initial_primes)
  (b_primes : List Nat)
  (h_b_selection : b_selection initial_primes b_primes) :
  ∃ (strategy : Game), a_winning_strategy strategy :=
sorry

end NUMINAMATH_CALUDE_a_has_winning_strategy_l3414_341403


namespace NUMINAMATH_CALUDE_hyperbola_properties_l3414_341418

/-- Given a hyperbola C with equation 9y^2 - 16x^2 = 144 -/
def hyperbola_C (x y : ℝ) : Prop := 9 * y^2 - 16 * x^2 = 144

/-- Point P -/
def point_P : ℝ × ℝ := (6, 4)

/-- Theorem stating properties of hyperbola C and a related hyperbola -/
theorem hyperbola_properties :
  ∃ (a b c : ℝ),
    /- Transverse axis length -/
    2 * a = 8 ∧
    /- Conjugate axis length -/
    2 * b = 6 ∧
    /- Foci coordinates -/
    (∀ (x y : ℝ), hyperbola_C x y → (x = 0 ∧ (y = c ∨ y = -c))) ∧
    /- Eccentricity -/
    c / a = 5 / 4 ∧
    /- New hyperbola equation -/
    (∀ (x y : ℝ), x^2 / 27 - y^2 / 48 = 1 →
      /- Same asymptotes as C -/
      (∃ (k : ℝ), k ≠ 0 ∧ 9 * y^2 - 16 * x^2 = 144 * k) ∧
      /- Passes through point P -/
      (let (px, py) := point_P; x = px ∧ y = py)) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l3414_341418


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l3414_341433

theorem polynomial_division_theorem (x : ℝ) : 
  (x - 3) * (x^4 + 3*x^3 - 7*x^2 - 10*x - 39) + (-47) = 
  x^5 - 16*x^3 + 11*x^2 - 9*x + 10 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l3414_341433


namespace NUMINAMATH_CALUDE_total_is_41X_l3414_341465

/-- Represents the number of people in different categories of a community -/
structure Community where
  children : ℕ
  teenagers : ℕ
  women : ℕ
  men : ℕ

/-- Defines a community with the given relationships between categories -/
def specialCommunity (X : ℕ) : Community where
  children := X
  teenagers := 4 * X
  women := 3 * (4 * X)
  men := 2 * (3 * (4 * X))

/-- Calculates the total number of people in a community -/
def totalPeople (c : Community) : ℕ :=
  c.children + c.teenagers + c.women + c.men

/-- Theorem stating that the total number of people in the special community is 41X -/
theorem total_is_41X (X : ℕ) :
  totalPeople (specialCommunity X) = 41 * X := by
  sorry

end NUMINAMATH_CALUDE_total_is_41X_l3414_341465


namespace NUMINAMATH_CALUDE_root_sum_ratio_l3414_341413

theorem root_sum_ratio (a b c d : ℝ) (h1 : a ≠ 0) (h2 : d = 0)
  (h3 : a * (4 : ℝ)^3 + b * (4 : ℝ)^2 + c * (4 : ℝ) + d = 0)
  (h4 : a * (-3 : ℝ)^3 + b * (-3 : ℝ)^2 + c * (-3 : ℝ) + d = 0) :
  (b + c) / a = -13 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_ratio_l3414_341413


namespace NUMINAMATH_CALUDE_geometric_mean_a4_a8_l3414_341473

def geometric_sequence (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ := a₁ * q^(n - 1)

theorem geometric_mean_a4_a8 :
  let a := geometric_sequence (1/8) 2
  (a 4 * a 8)^(1/2) = 4 := by sorry

end NUMINAMATH_CALUDE_geometric_mean_a4_a8_l3414_341473


namespace NUMINAMATH_CALUDE_relationship_between_exponents_l3414_341482

theorem relationship_between_exponents 
  (a b c d : ℝ) 
  (x y q z : ℝ) 
  (h1 : a^(2*x) = c^(2*q)) 
  (h2 : a^(2*x) = b^2) 
  (h3 : c^(3*y) = a^(3*z)) 
  (h4 : c^(3*y) = d^2) 
  (h5 : a ≠ 0) 
  (h6 : b ≠ 0) 
  (h7 : c ≠ 0) 
  (h8 : d ≠ 0) : 
  x * y = q * z := by
sorry

end NUMINAMATH_CALUDE_relationship_between_exponents_l3414_341482


namespace NUMINAMATH_CALUDE_weight_of_a_l3414_341462

-- Define the people
structure Person where
  weight : ℝ
  height : ℝ
  age : ℝ

-- Define the group of A, B, C
def group_abc (a b c : Person) : Prop :=
  (a.weight + b.weight + c.weight) / 3 = 84 ∧
  (a.height + b.height + c.height) / 3 = 170 ∧
  (a.age + b.age + c.age) / 3 = 30

-- Define the group with D added
def group_abcd (a b c d : Person) : Prop :=
  (a.weight + b.weight + c.weight + d.weight) / 4 = 80 ∧
  (a.height + b.height + c.height + d.height) / 4 = 172 ∧
  (a.age + b.age + c.age + d.age) / 4 = 28

-- Define the group with E replacing A
def group_bcde (b c d e : Person) : Prop :=
  (b.weight + c.weight + d.weight + e.weight) / 4 = 79 ∧
  (b.height + c.height + d.height + e.height) / 4 = 173 ∧
  (b.age + c.age + d.age + e.age) / 4 = 27

-- Define the relationship between D and E
def d_e_relation (d e a : Person) : Prop :=
  e.weight = d.weight + 7 ∧
  e.age = a.age - 3

-- Theorem statement
theorem weight_of_a 
  (a b c d e : Person)
  (h1 : group_abc a b c)
  (h2 : group_abcd a b c d)
  (h3 : group_bcde b c d e)
  (h4 : d_e_relation d e a) :
  a.weight = 79 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_a_l3414_341462


namespace NUMINAMATH_CALUDE_student_count_l3414_341497

/-- The number of students in Elementary and Middle School -/
def total_students (elementary : ℕ) (middle : ℕ) : ℕ :=
  elementary + middle

/-- Theorem stating the total number of students given the conditions -/
theorem student_count : ∃ (elementary : ℕ) (middle : ℕ),
  middle = 50 ∧ 
  elementary = 4 * middle - 3 ∧
  total_students elementary middle = 247 := by
  sorry

end NUMINAMATH_CALUDE_student_count_l3414_341497


namespace NUMINAMATH_CALUDE_silver_coin_percentage_is_31_5_percent_l3414_341490

/-- Represents the composition of objects in an urn --/
structure UrnComposition where
  bead_percentage : ℝ
  gold_coin_percentage : ℝ

/-- Calculates the percentage of silver coins in the urn --/
def silver_coin_percentage (urn : UrnComposition) : ℝ :=
  (1 - urn.bead_percentage) * (1 - urn.gold_coin_percentage)

/-- Theorem stating that the percentage of silver coins is 31.5% --/
theorem silver_coin_percentage_is_31_5_percent :
  let urn : UrnComposition := ⟨0.3, 0.55⟩
  silver_coin_percentage urn = 0.315 := by sorry

end NUMINAMATH_CALUDE_silver_coin_percentage_is_31_5_percent_l3414_341490


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l3414_341442

theorem fixed_point_on_line (m : ℝ) : (m - 1) * 9 + (2 * m - 1) * (-4) = m - 5 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l3414_341442


namespace NUMINAMATH_CALUDE_orthogonal_vectors_solution_l3414_341411

theorem orthogonal_vectors_solution :
  ∃! y : ℝ, (2 : ℝ) * (-1 : ℝ) + (-1 : ℝ) * y + (3 : ℝ) * (0 : ℝ) + (1 : ℝ) * (-4 : ℝ) = 0 :=
by sorry

end NUMINAMATH_CALUDE_orthogonal_vectors_solution_l3414_341411


namespace NUMINAMATH_CALUDE_sin_sum_inverse_sin_tan_l3414_341449

theorem sin_sum_inverse_sin_tan (x y : ℝ) 
  (hx : x = 3 / 5) (hy : y = 1 / 2) : 
  Real.sin (Real.arcsin x + Real.arctan y) = 2 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_inverse_sin_tan_l3414_341449


namespace NUMINAMATH_CALUDE_smallest_four_digit_mod_9_l3414_341499

theorem smallest_four_digit_mod_9 : 
  ∀ n : ℕ, n ≥ 1000 ∧ n ≡ 8 [MOD 9] → n ≥ 1007 :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_mod_9_l3414_341499


namespace NUMINAMATH_CALUDE_largest_band_members_l3414_341466

theorem largest_band_members : ∃ (m r x : ℕ),
  m < 100 ∧
  r * x + 3 = m ∧
  (r - 3) * (x + 1) = m ∧
  ∀ (m' r' x' : ℕ),
    m' < 100 →
    r' * x' + 3 = m' →
    (r' - 3) * (x' + 1) = m' →
    m' ≤ m ∧
  m = 87 := by
sorry

end NUMINAMATH_CALUDE_largest_band_members_l3414_341466


namespace NUMINAMATH_CALUDE_gcd_360_504_l3414_341417

theorem gcd_360_504 : Nat.gcd 360 504 = 72 := by
  sorry

end NUMINAMATH_CALUDE_gcd_360_504_l3414_341417


namespace NUMINAMATH_CALUDE_stadium_length_feet_l3414_341409

/-- Conversion factor from yards to feet -/
def yards_to_feet : ℕ → ℕ := λ x => 3 * x

/-- Length of the stadium in yards -/
def stadium_length_yards : ℕ := 80

/-- Theorem stating that the stadium length in feet is 240 -/
theorem stadium_length_feet : yards_to_feet stadium_length_yards = 240 := by
  sorry

end NUMINAMATH_CALUDE_stadium_length_feet_l3414_341409


namespace NUMINAMATH_CALUDE_linear_function_problem_l3414_341436

/-- A linear function passing through (1, 3) -/
def f (k : ℝ) (x : ℝ) : ℝ := k * x + 1

/-- The linear function shifted up by 2 units -/
def g (k : ℝ) (x : ℝ) : ℝ := f k x + 2

theorem linear_function_problem (k : ℝ) (h : k ≠ 0) (h1 : f k 1 = 3) :
  k = 2 ∧ ∀ x, g k x = 2 * x + 3 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_problem_l3414_341436


namespace NUMINAMATH_CALUDE_count_minimally_intersecting_mod_1000_l3414_341429

def Universe : Finset Nat := {1,2,3,4,5,6,7,8}

def MinimallyIntersecting (D E F : Finset Nat) : Prop :=
  (D ∩ E).card = 1 ∧ (E ∩ F).card = 1 ∧ (F ∩ D).card = 1 ∧ (D ∩ E ∩ F).card = 0

def CountMinimallyIntersecting : Nat :=
  (Finset.powerset Universe).card.choose 3

theorem count_minimally_intersecting_mod_1000 :
  CountMinimallyIntersecting % 1000 = 64 := by sorry

end NUMINAMATH_CALUDE_count_minimally_intersecting_mod_1000_l3414_341429
