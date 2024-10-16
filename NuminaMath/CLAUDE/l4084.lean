import Mathlib

namespace NUMINAMATH_CALUDE_angle_relationship_indeterminate_l4084_408416

-- Define a plane
def Plane : Type := ℝ × ℝ → ℝ

-- Define a point in 3D space
def Point : Type := ℝ × ℝ × ℝ

-- Define a ray in 3D space
def Ray : Type := Point × Point

-- Function to calculate angle between two rays
def angle_between_rays : Ray → Ray → ℝ := sorry

-- Function to project a ray onto a plane
def project_ray : Ray → Plane → Ray := sorry

-- Function to check if a point is outside a plane
def is_outside_plane : Point → Plane → Prop := sorry

-- Theorem statement
theorem angle_relationship_indeterminate 
  (M : Plane) (P : Point) (r1 r2 : Ray) 
  (h_outside : is_outside_plane P M)
  (h_alpha : 0 < angle_between_rays r1 r2 ∧ angle_between_rays r1 r2 < π)
  (h_beta : 0 < angle_between_rays (project_ray r1 M) (project_ray r2 M) ∧ 
            angle_between_rays (project_ray r1 M) (project_ray r2 M) < π) :
  ¬ ∃ (R : ℝ → ℝ → Prop), 
    ∀ (α β : ℝ), 
      α = angle_between_rays r1 r2 → 
      β = angle_between_rays (project_ray r1 M) (project_ray r2 M) → 
      R α β :=
sorry

end NUMINAMATH_CALUDE_angle_relationship_indeterminate_l4084_408416


namespace NUMINAMATH_CALUDE_cubic_symmetry_about_origin_l4084_408495

def f (x : ℝ) : ℝ := x^3

theorem cubic_symmetry_about_origin :
  ∀ x : ℝ, f (-x) = -f x :=
by sorry

end NUMINAMATH_CALUDE_cubic_symmetry_about_origin_l4084_408495


namespace NUMINAMATH_CALUDE_green_and_yellow_peaches_count_l4084_408403

/-- Given a basket of peaches, prove that the total number of green and yellow peaches is 20. -/
theorem green_and_yellow_peaches_count (yellow_peaches green_peaches : ℕ) 
  (h1 : yellow_peaches = 14)
  (h2 : green_peaches = 6) : 
  yellow_peaches + green_peaches = 20 := by
  sorry

end NUMINAMATH_CALUDE_green_and_yellow_peaches_count_l4084_408403


namespace NUMINAMATH_CALUDE_binomial_max_fifth_term_l4084_408436

/-- 
If in the binomial expansion of (√x + 2/x)^n, only the fifth term has the maximum 
binomial coefficient, then n = 8.
-/
theorem binomial_max_fifth_term (n : ℕ) : 
  (∀ k : ℕ, k ≠ 4 → Nat.choose n k ≤ Nat.choose n 4) ∧ 
  (∀ k : ℕ, k ≠ 4 → Nat.choose n k < Nat.choose n 4) → 
  n = 8 := by sorry

end NUMINAMATH_CALUDE_binomial_max_fifth_term_l4084_408436


namespace NUMINAMATH_CALUDE_smallest_b_probability_l4084_408412

/-- The number of cards in the deck -/
def deckSize : ℕ := 40

/-- The probability that Carly and Fiona are on the same team when Carly picks card number b and Fiona picks card number b+7 -/
def q (b : ℕ) : ℚ :=
  let totalCombinations := (deckSize - 2).choose 2
  let lowerTeamCombinations := (deckSize - b - 7).choose 2
  let higherTeamCombinations := (b - 1).choose 2
  (lowerTeamCombinations + higherTeamCombinations : ℚ) / totalCombinations

/-- The smallest value of b for which q(b) ≥ 1/2 -/
def smallestB : ℕ := 18

theorem smallest_b_probability (b : ℕ) :
  b < smallestB → q b < 1/2 ∧
  q smallestB = 318/703 :=
sorry

end NUMINAMATH_CALUDE_smallest_b_probability_l4084_408412


namespace NUMINAMATH_CALUDE_bread_distribution_l4084_408438

theorem bread_distribution (a d : ℚ) (h1 : d > 0) 
  (h2 : (a - 2*d) + (a - d) + a + (a + d) + (a + 2*d) = 100)
  (h3 : (1/7) * (a + (a + d) + (a + 2*d)) = (a - 2*d) + (a - d)) :
  a - 2*d = 5/3 := by
sorry

end NUMINAMATH_CALUDE_bread_distribution_l4084_408438


namespace NUMINAMATH_CALUDE_perpendicular_lines_parallel_l4084_408488

-- Define the concept of a line in a plane
def Line : Type := Unit

-- Define the concept of a plane
def Plane : Type := Unit

-- Define the perpendicular relation between two lines in a plane
def perpendicular (p : Plane) (l1 l2 : Line) : Prop := sorry

-- Define the parallel relation between two lines in a plane
def parallel (p : Plane) (l1 l2 : Line) : Prop := sorry

-- State the theorem
theorem perpendicular_lines_parallel (p : Plane) (a b c : Line) :
  perpendicular p a c → perpendicular p b c → parallel p a b := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_parallel_l4084_408488


namespace NUMINAMATH_CALUDE_remainder_x_plus_one_2025_mod_x_squared_plus_one_l4084_408455

theorem remainder_x_plus_one_2025_mod_x_squared_plus_one (x : ℤ) :
  (x + 1) ^ 2025 % (x^2 + 1) = 0 := by
sorry

end NUMINAMATH_CALUDE_remainder_x_plus_one_2025_mod_x_squared_plus_one_l4084_408455


namespace NUMINAMATH_CALUDE_prime_value_theorem_l4084_408404

theorem prime_value_theorem (n : ℕ+) (h : Nat.Prime (n^4 - 16*n^2 + 100)) : n = 3 := by
  sorry

end NUMINAMATH_CALUDE_prime_value_theorem_l4084_408404


namespace NUMINAMATH_CALUDE_non_square_seq_2003_l4084_408461

/-- The sequence of positive integers with perfect squares removed -/
def non_square_seq : ℕ → ℕ := sorry

/-- The 2003rd term of the sequence of positive integers with perfect squares removed is 2048 -/
theorem non_square_seq_2003 : non_square_seq 2003 = 2048 := by sorry

end NUMINAMATH_CALUDE_non_square_seq_2003_l4084_408461


namespace NUMINAMATH_CALUDE_two_x_minus_y_value_l4084_408434

theorem two_x_minus_y_value (x y : ℝ) 
  (hx : |x| = 2) 
  (hy : |y| = 3) 
  (hxy : x / y < 0) : 
  2 * x - y = 7 ∨ 2 * x - y = -7 := by
sorry

end NUMINAMATH_CALUDE_two_x_minus_y_value_l4084_408434


namespace NUMINAMATH_CALUDE_trig_expression_equals_one_l4084_408421

theorem trig_expression_equals_one :
  let α : Real := 37 * π / 180
  let β : Real := 53 * π / 180
  (1 - 1 / Real.cos α) * (1 + 1 / Real.sin β) * (1 - 1 / Real.sin α) * (1 + 1 / Real.cos β) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equals_one_l4084_408421


namespace NUMINAMATH_CALUDE_decimal_5_17_repetend_l4084_408458

/-- The repetend of the decimal representation of 5/17 -/
def repetend_5_17 : ℕ := 294117

/-- The decimal representation of 5/17 -/
def decimal_5_17 : ℚ := 5 / 17

theorem decimal_5_17_repetend :
  ∃ (k : ℕ), decimal_5_17 = (k : ℚ) / 999999 + (repetend_5_17 : ℚ) / 999999 :=
sorry

end NUMINAMATH_CALUDE_decimal_5_17_repetend_l4084_408458


namespace NUMINAMATH_CALUDE_fish_weight_theorem_l4084_408459

/-- The total weight of fish caught by Ali, Peter, and Joey -/
def total_fish_weight (ali_weight peter_weight joey_weight : ℝ) : ℝ :=
  ali_weight + peter_weight + joey_weight

/-- Theorem: Given the conditions, the total weight of fish caught is 25 kg -/
theorem fish_weight_theorem (peter_weight : ℝ) 
  (h1 : peter_weight > 0)
  (h2 : 2 * peter_weight = 12)
  (h3 : joey_weight = peter_weight + 1) :
  total_fish_weight 12 peter_weight joey_weight = 25 := by
  sorry

#check fish_weight_theorem

end NUMINAMATH_CALUDE_fish_weight_theorem_l4084_408459


namespace NUMINAMATH_CALUDE_matrix_not_invertible_l4084_408400

theorem matrix_not_invertible : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := ![![2 * (29/13 : ℝ) - 1, 5], ![4 + (29/13 : ℝ), 9]]
  ¬(IsUnit (Matrix.det A)) := by sorry

end NUMINAMATH_CALUDE_matrix_not_invertible_l4084_408400


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l4084_408454

theorem sufficient_not_necessary (x y : ℝ) :
  (∀ x y : ℝ, x > 2 ∧ y > 2 → x + y > 4) ∧
  (∃ x y : ℝ, x + y > 4 ∧ ¬(x > 2 ∧ y > 2)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l4084_408454


namespace NUMINAMATH_CALUDE_fraction_ratio_equality_l4084_408447

theorem fraction_ratio_equality : 
  let certain_fraction : ℚ := 84 / 25
  let given_fraction : ℚ := 6 / 5
  let comparison_fraction : ℚ := 2 / 5
  let answer : ℚ := 1 / 7  -- 0.14285714285714288 is approximately 1/7
  (certain_fraction / given_fraction) = (comparison_fraction / answer) :=
by sorry

end NUMINAMATH_CALUDE_fraction_ratio_equality_l4084_408447


namespace NUMINAMATH_CALUDE_tony_weightlifting_ratio_l4084_408443

/-- Given Tony's weightlifting capabilities, prove the ratio of squat to military press weight -/
theorem tony_weightlifting_ratio :
  let curl_weight : ℝ := 90
  let military_press_weight : ℝ := 2 * curl_weight
  let squat_weight : ℝ := 900
  squat_weight / military_press_weight = 5 := by sorry

end NUMINAMATH_CALUDE_tony_weightlifting_ratio_l4084_408443


namespace NUMINAMATH_CALUDE_absolute_value_nonnegative_l4084_408437

theorem absolute_value_nonnegative (x : ℝ) : ¬(|x| < 0) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_nonnegative_l4084_408437


namespace NUMINAMATH_CALUDE_inductive_reasoning_methods_l4084_408490

-- Define the type for reasoning methods
inductive ReasoningMethod
  | InferBallFromCircle
  | InferTriangleAngles
  | DeductBrokenChairs
  | InferPolygonAngles

-- Define a predicate for inductive reasoning
def isInductiveReasoning : ReasoningMethod → Prop
  | ReasoningMethod.InferBallFromCircle => False
  | ReasoningMethod.InferTriangleAngles => True
  | ReasoningMethod.DeductBrokenChairs => False
  | ReasoningMethod.InferPolygonAngles => True

-- Theorem stating which methods are inductive reasoning
theorem inductive_reasoning_methods :
  (isInductiveReasoning ReasoningMethod.InferTriangleAngles) ∧
  (isInductiveReasoning ReasoningMethod.InferPolygonAngles) ∧
  (¬ isInductiveReasoning ReasoningMethod.InferBallFromCircle) ∧
  (¬ isInductiveReasoning ReasoningMethod.DeductBrokenChairs) :=
by sorry


end NUMINAMATH_CALUDE_inductive_reasoning_methods_l4084_408490


namespace NUMINAMATH_CALUDE_new_cost_percentage_l4084_408475

/-- The cost function -/
def cost (t c a x : ℝ) (n : ℕ) : ℝ := t * c * (a * x) ^ n

/-- Theorem stating the relationship between the original and new cost -/
theorem new_cost_percentage (t c a x : ℝ) (n : ℕ) :
  let O := cost t c a x n
  let E := cost t (2*c) (2*a) x (n+2)
  E = 2^(n+1) * x^2 * O :=
by sorry

end NUMINAMATH_CALUDE_new_cost_percentage_l4084_408475


namespace NUMINAMATH_CALUDE_product_in_fourth_quadrant_l4084_408431

def complex_multiply (a b c d : ℝ) : ℂ :=
  Complex.mk (a * c - b * d) (a * d + b * c)

def fourth_quadrant (z : ℂ) : Prop :=
  z.re > 0 ∧ z.im < 0

theorem product_in_fourth_quadrant :
  let z₁ : ℂ := Complex.mk 3 1
  let z₂ : ℂ := Complex.mk 1 (-1)
  let z : ℂ := complex_multiply z₁.re z₁.im z₂.re z₂.im
  fourth_quadrant z := by sorry

end NUMINAMATH_CALUDE_product_in_fourth_quadrant_l4084_408431


namespace NUMINAMATH_CALUDE_sales_tax_difference_l4084_408427

/-- The price of the item before tax -/
def item_price : ℝ := 50

/-- The first sales tax rate -/
def tax_rate_1 : ℝ := 0.075

/-- The second sales tax rate -/
def tax_rate_2 : ℝ := 0.0625

/-- Theorem: The difference between the sales taxes is $0.625 -/
theorem sales_tax_difference : 
  item_price * tax_rate_1 - item_price * tax_rate_2 = 0.625 := by
  sorry

end NUMINAMATH_CALUDE_sales_tax_difference_l4084_408427


namespace NUMINAMATH_CALUDE_five_integers_sum_20_product_420_l4084_408460

theorem five_integers_sum_20_product_420 :
  ∃ (a b c d e : ℕ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧
    a + b + c + d + e = 20 ∧
    a * b * c * d * e = 420 :=
by
  sorry

end NUMINAMATH_CALUDE_five_integers_sum_20_product_420_l4084_408460


namespace NUMINAMATH_CALUDE_red_tile_probability_l4084_408451

theorem red_tile_probability (n : ℕ) (h : n = 77) : 
  let red_tiles := (Finset.range n).filter (λ x => (x + 1) % 7 = 3)
  Finset.card red_tiles = 10 ∧ 
  (Finset.card red_tiles : ℚ) / n = 10 / 77 := by
  sorry

end NUMINAMATH_CALUDE_red_tile_probability_l4084_408451


namespace NUMINAMATH_CALUDE_parallelepiped_length_l4084_408429

theorem parallelepiped_length : ∃ (n : ℕ), 
  (n > 0) ∧ 
  (n - 2 > 0) ∧ 
  (n - 4 > 0) ∧
  ((n - 2) * (n - 4) * (n - 6) = (2 * n * (n - 2) * (n - 4)) / 3) ∧
  (n = 18) := by
sorry

end NUMINAMATH_CALUDE_parallelepiped_length_l4084_408429


namespace NUMINAMATH_CALUDE_average_mark_is_35_l4084_408449

/-- The average mark obtained by candidates in an examination. -/
def average_mark (total_marks : ℕ) (num_candidates : ℕ) : ℚ :=
  total_marks / num_candidates

/-- Theorem stating that the average mark is 35 given the conditions. -/
theorem average_mark_is_35 :
  average_mark 4200 120 = 35 := by
  sorry

end NUMINAMATH_CALUDE_average_mark_is_35_l4084_408449


namespace NUMINAMATH_CALUDE_apples_in_box_l4084_408418

theorem apples_in_box (initial_apples : ℕ) : 
  (initial_apples / 2 - 25 = 6) → initial_apples = 62 := by
  sorry

end NUMINAMATH_CALUDE_apples_in_box_l4084_408418


namespace NUMINAMATH_CALUDE_ten_by_ten_not_tileable_l4084_408423

/-- Represents a checkerboard -/
structure Checkerboard :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents a tile -/
structure Tile :=
  (width : ℕ)
  (height : ℕ)

/-- Defines a function to check if a checkerboard can be tiled with given tiles -/
def can_tile (board : Checkerboard) (tile : Tile) : Prop :=
  ∃ (n : ℕ), board.rows * board.cols = n * tile.width * tile.height

/-- Theorem stating that a 10x10 checkerboard cannot be tiled with 1x4 tiles -/
theorem ten_by_ten_not_tileable :
  ¬ can_tile (Checkerboard.mk 10 10) (Tile.mk 1 4) :=
sorry

end NUMINAMATH_CALUDE_ten_by_ten_not_tileable_l4084_408423


namespace NUMINAMATH_CALUDE_expression_simplification_l4084_408446

theorem expression_simplification (x : ℝ) :
  2 * x * (4 * x^2 - 3) - 6 * (x^2 - 3 * x + 8) = 8 * x^3 - 6 * x^2 + 12 * x - 48 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l4084_408446


namespace NUMINAMATH_CALUDE_remainder_of_282_l4084_408457

theorem remainder_of_282 : ∃ r : ℕ, r < 9 ∧ r < 31 ∧ 282 % 31 = r ∧ 282 % 9 = r :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_remainder_of_282_l4084_408457


namespace NUMINAMATH_CALUDE_tank_filling_ratio_l4084_408468

/-- Proves that the ratio of initial water to total capacity is 1/2 given specific tank conditions -/
theorem tank_filling_ratio 
  (capacity : ℝ) 
  (inflow_rate : ℝ) 
  (outflow_rate1 : ℝ) 
  (outflow_rate2 : ℝ) 
  (fill_time : ℝ) 
  (h1 : capacity = 10) 
  (h2 : inflow_rate = 0.5) 
  (h3 : outflow_rate1 = 0.25) 
  (h4 : outflow_rate2 = 1/6) 
  (h5 : fill_time = 60) : 
  (capacity - (inflow_rate - outflow_rate1 - outflow_rate2) * fill_time) / capacity = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_tank_filling_ratio_l4084_408468


namespace NUMINAMATH_CALUDE_football_game_attendance_l4084_408497

/-- Prove the total attendance at a football game -/
theorem football_game_attendance
  (adult_price : ℚ)
  (child_price : ℚ)
  (total_collected : ℚ)
  (num_adults : ℕ)
  (h1 : adult_price = 60 / 100)
  (h2 : child_price = 25 / 100)
  (h3 : total_collected = 140)
  (h4 : num_adults = 200) :
  num_adults + ((total_collected - (↑num_adults * adult_price)) / child_price) = 280 :=
by sorry

end NUMINAMATH_CALUDE_football_game_attendance_l4084_408497


namespace NUMINAMATH_CALUDE_intersection_with_complement_l4084_408478

open Set

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 3}

theorem intersection_with_complement : A ∩ (U \ B) = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l4084_408478


namespace NUMINAMATH_CALUDE_periodic_trig_function_l4084_408482

/-- Given a function f(x) = a*sin(π*x + α) + b*cos(π*x + β) + 4,
    where a, b, α, and β are constants, 
    if f(2009) = 5, then f(2010) = 3 -/
theorem periodic_trig_function 
  (a b α β : ℝ) 
  (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = a * Real.sin (π * x + α) + b * Real.cos (π * x + β) + 4) 
  (h2 : f 2009 = 5) : 
  f 2010 = 3 := by
sorry

end NUMINAMATH_CALUDE_periodic_trig_function_l4084_408482


namespace NUMINAMATH_CALUDE_solution_relationship_l4084_408442

theorem solution_relationship (c c' d d' : ℝ) 
  (hc : c ≠ 0) (hc' : c' ≠ 0)
  (h : -d / (2 * c) = 2 * (-d' / (3 * c'))) :
  d / (2 * c) = 2 * d' / (3 * c') := by
  sorry

end NUMINAMATH_CALUDE_solution_relationship_l4084_408442


namespace NUMINAMATH_CALUDE_sin_arccos_twelve_thirteenths_l4084_408493

theorem sin_arccos_twelve_thirteenths : 
  Real.sin (Real.arccos (12 / 13)) = 5 / 13 := by
  sorry

end NUMINAMATH_CALUDE_sin_arccos_twelve_thirteenths_l4084_408493


namespace NUMINAMATH_CALUDE_sin_double_angle_l4084_408489

theorem sin_double_angle (θ : Real) (h : Real.sin θ = 3/5) : Real.sin (2 * θ) = 24/25 := by
  sorry

end NUMINAMATH_CALUDE_sin_double_angle_l4084_408489


namespace NUMINAMATH_CALUDE_line_points_relation_l4084_408402

/-- 
Given a line with equation x = 6y + 5, 
if two points (m, n) and (m + Q, n + p) lie on this line, 
and p = 1/3, then Q = 2.
-/
theorem line_points_relation (m n Q p : ℝ) : 
  (m = 6 * n + 5) →
  (m + Q = 6 * (n + p) + 5) →
  (p = 1/3) →
  Q = 2 := by
  sorry

end NUMINAMATH_CALUDE_line_points_relation_l4084_408402


namespace NUMINAMATH_CALUDE_general_equation_l4084_408462

theorem general_equation (n : ℤ) : 
  (n / (n - 4)) + ((8 - n) / ((8 - n) - 4)) = 2 :=
by sorry

end NUMINAMATH_CALUDE_general_equation_l4084_408462


namespace NUMINAMATH_CALUDE_calculate_difference_l4084_408474

theorem calculate_difference (x y z : ℝ) (hx : x = 40) (hy : y = 20) (hz : z = 5) :
  0.8 * (3 * (2 * x))^2 - 0.8 * Real.sqrt ((y / 4)^3 * z^3) = 45980 := by
  sorry

end NUMINAMATH_CALUDE_calculate_difference_l4084_408474


namespace NUMINAMATH_CALUDE_chord_sum_l4084_408477

/-- Definition of the circle --/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 4*y - 20 = 0

/-- The point (1, -1) lies on the circle --/
axiom point_on_circle : circle_equation 1 (-1)

/-- Definition of the longest chord length --/
def longest_chord_length : ℝ := sorry

/-- Definition of the shortest chord length --/
def shortest_chord_length : ℝ := sorry

/-- Theorem: The sum of the longest and shortest chord lengths is 18 --/
theorem chord_sum :
  longest_chord_length + shortest_chord_length = 18 := by sorry

end NUMINAMATH_CALUDE_chord_sum_l4084_408477


namespace NUMINAMATH_CALUDE_cube_side_ratio_l4084_408406

theorem cube_side_ratio (a b : ℝ) (h : a > 0 ∧ b > 0) :
  (6 * a^2) / (6 * b^2) = 25 → a / b = 5 := by
  sorry

end NUMINAMATH_CALUDE_cube_side_ratio_l4084_408406


namespace NUMINAMATH_CALUDE_square_difference_given_sum_and_weighted_sum_l4084_408472

theorem square_difference_given_sum_and_weighted_sum (x y : ℝ) 
  (h1 : x + y = 15) (h2 : 3 * x + y = 20) : x^2 - y^2 = -150 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_given_sum_and_weighted_sum_l4084_408472


namespace NUMINAMATH_CALUDE_yura_catches_lena_l4084_408456

/-- The time it takes for Yura to catch up with Lena -/
def catchUpTime : ℝ := 5

/-- Lena's walking speed -/
def lenaSpeed : ℝ := 1

/-- The time difference between Lena and Yura's start -/
def timeDifference : ℝ := 5

theorem yura_catches_lena :
  ∀ (t : ℝ),
  t = catchUpTime →
  (lenaSpeed * (t + timeDifference)) = (2 * lenaSpeed * t) :=
by sorry

end NUMINAMATH_CALUDE_yura_catches_lena_l4084_408456


namespace NUMINAMATH_CALUDE_two_white_balls_probability_l4084_408496

/-- The probability of drawing two white balls from a box with white and black balls -/
theorem two_white_balls_probability
  (total_balls : ℕ)
  (white_balls : ℕ)
  (black_balls : ℕ)
  (h1 : total_balls = white_balls + black_balls)
  (h2 : white_balls = 7)
  (h3 : black_balls = 8) :
  (white_balls.choose 2 : ℚ) / (total_balls.choose 2) = 1 / 5 :=
by sorry

end NUMINAMATH_CALUDE_two_white_balls_probability_l4084_408496


namespace NUMINAMATH_CALUDE_derivative_of_y_l4084_408408

noncomputable def y (x : ℝ) : ℝ := Real.sin x - 2^x

theorem derivative_of_y (x : ℝ) :
  deriv y x = Real.cos x - 2^x * Real.log 2 := by sorry

end NUMINAMATH_CALUDE_derivative_of_y_l4084_408408


namespace NUMINAMATH_CALUDE_largest_constant_K_l4084_408445

theorem largest_constant_K : ∃ (K : ℝ), K > 0 ∧
  (∀ (k : ℝ) (a b c : ℝ), 0 ≤ k ∧ k ≤ K ∧ 
    a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ 
    a^2 + b^2 + c^2 + k*a*b*c = k + 3 →
    a + b + c ≤ 3) ∧
  (∀ (K' : ℝ), K' > K →
    ∃ (k : ℝ) (a b c : ℝ), 0 ≤ k ∧ k ≤ K' ∧ 
      a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ 
      a^2 + b^2 + c^2 + k*a*b*c = k + 3 ∧
      a + b + c > 3) ∧
  K = 1 := by
sorry

end NUMINAMATH_CALUDE_largest_constant_K_l4084_408445


namespace NUMINAMATH_CALUDE_min_total_weight_proof_l4084_408499

/-- The maximum number of crates the trailer can carry on a single trip -/
def max_crates : ℕ := 6

/-- The minimum weight of each crate in kilograms -/
def min_crate_weight : ℕ := 120

/-- The minimum total weight of crates on a single trip when carrying the maximum number of crates -/
def min_total_weight : ℕ := max_crates * min_crate_weight

theorem min_total_weight_proof :
  min_total_weight = 720 := by sorry

end NUMINAMATH_CALUDE_min_total_weight_proof_l4084_408499


namespace NUMINAMATH_CALUDE_jar_water_problem_l4084_408492

theorem jar_water_problem (S L : ℝ) (hS : S > 0) (hL : L > 0) (h_capacities : S ≠ L) : 
  let water := (1/5) * S
  (water = (1/4) * L) → ((2 * water) / L = 1/2) := by sorry

end NUMINAMATH_CALUDE_jar_water_problem_l4084_408492


namespace NUMINAMATH_CALUDE_largest_inscribed_square_side_length_l4084_408463

theorem largest_inscribed_square_side_length 
  (square_side : ℝ) 
  (h_square_side : square_side = 12) :
  ∃ (triangle_side inscribed_square_side : ℝ),
    -- Equilateral triangle side length
    triangle_side = 4 * Real.sqrt 3 ∧
    -- Inscribed square side length
    inscribed_square_side = 12 - 4 * Real.sqrt 3 ∧
    -- Relation between square side, triangle side, and inscribed square side
    2 * triangle_side + inscribed_square_side * Real.sqrt 2 = square_side * Real.sqrt 2 ∧
    -- Triangle height equals half of square side
    (Real.sqrt 3 / 2) * triangle_side = square_side / 2 :=
by sorry

end NUMINAMATH_CALUDE_largest_inscribed_square_side_length_l4084_408463


namespace NUMINAMATH_CALUDE_at_least_one_irrational_l4084_408409

theorem at_least_one_irrational (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0)
  (h3 : a^2 * b^2 * (a^2 * b^2 + 4) = 2 * (a^6 + b^6)) :
  ¬(∃ (q r : ℚ), (↑q : ℝ) = a ∧ (↑r : ℝ) = b) :=
sorry

end NUMINAMATH_CALUDE_at_least_one_irrational_l4084_408409


namespace NUMINAMATH_CALUDE_kennedy_drive_home_l4084_408428

/-- Calculates the remaining miles that can be driven given the car's efficiency,
    initial gas amount, and distance already driven. -/
def remaining_miles (efficiency : ℝ) (initial_gas : ℝ) (driven_miles : ℝ) : ℝ :=
  efficiency * initial_gas - driven_miles

theorem kennedy_drive_home 
  (efficiency : ℝ) 
  (initial_gas : ℝ) 
  (school_miles : ℝ) 
  (softball_miles : ℝ) 
  (restaurant_miles : ℝ) 
  (friend_miles : ℝ) 
  (h1 : efficiency = 19)
  (h2 : initial_gas = 2)
  (h3 : school_miles = 15)
  (h4 : softball_miles = 6)
  (h5 : restaurant_miles = 2)
  (h6 : friend_miles = 4) :
  remaining_miles efficiency initial_gas (school_miles + softball_miles + restaurant_miles + friend_miles) = 11 := by
  sorry

end NUMINAMATH_CALUDE_kennedy_drive_home_l4084_408428


namespace NUMINAMATH_CALUDE_angle_expression_equality_l4084_408419

theorem angle_expression_equality (θ : Real) 
  (h1 : ∃ (x y : Real), x < 0 ∧ y < 0 ∧ Real.cos θ = x ∧ Real.sin θ = y) 
  (h2 : Real.tan θ ^ 2 = -2 * Real.sqrt 2) : 
  Real.sin θ ^ 2 - Real.sin (3 * Real.pi + θ) * Real.cos (Real.pi + θ) - Real.sqrt 2 * Real.cos θ ^ 2 = (2 - 2 * Real.sqrt 2) / 3 := by
  sorry


end NUMINAMATH_CALUDE_angle_expression_equality_l4084_408419


namespace NUMINAMATH_CALUDE_town_square_length_l4084_408444

/-- The length of the town square in miles -/
def square_length : ℝ := 5.25

/-- The number of times runners go around the square -/
def laps : ℕ := 7

/-- The time (in minutes) it took the winner to finish the race this year -/
def winner_time : ℝ := 42

/-- The time (in minutes) it took last year's winner to finish the race -/
def last_year_time : ℝ := 47.25

/-- The time difference (in minutes) for running one mile between this year and last year -/
def speed_improvement : ℝ := 1

theorem town_square_length :
  square_length = (last_year_time - winner_time) / speed_improvement :=
by sorry

end NUMINAMATH_CALUDE_town_square_length_l4084_408444


namespace NUMINAMATH_CALUDE_N_is_positive_l4084_408430

theorem N_is_positive (a b : ℝ) : 
  let N := 4*a^2 - 12*a*b + 13*b^2 - 6*a + 4*b + 13
  0 < N := by sorry

end NUMINAMATH_CALUDE_N_is_positive_l4084_408430


namespace NUMINAMATH_CALUDE_isosceles_base_length_l4084_408415

/-- Represents a triangle with sides a, b, and c. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : a > 0
  pos_b : b > 0
  pos_c : c > 0

/-- An equilateral triangle is a triangle with all sides equal. -/
def IsEquilateral (t : Triangle) : Prop :=
  t.a = t.b ∧ t.b = t.c

/-- An isosceles triangle is a triangle with at least two sides equal. -/
def IsIsosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.a = t.c

/-- The perimeter of a triangle is the sum of its sides. -/
def Perimeter (t : Triangle) : ℝ :=
  t.a + t.b + t.c

theorem isosceles_base_length
  (equi : Triangle)
  (iso : Triangle)
  (h_equi_equilateral : IsEquilateral equi)
  (h_iso_isosceles : IsIsosceles iso)
  (h_equi_perimeter : Perimeter equi = 60)
  (h_iso_perimeter : Perimeter iso = 70)
  (h_shared_side : equi.a = iso.a) :
  iso.c = 30 :=
sorry

end NUMINAMATH_CALUDE_isosceles_base_length_l4084_408415


namespace NUMINAMATH_CALUDE_missing_number_proof_l4084_408484

theorem missing_number_proof (some_number : ℤ) : 
  (|4 - some_number * (3 - 12)| - |5 - 11| = 70) → some_number = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_missing_number_proof_l4084_408484


namespace NUMINAMATH_CALUDE_normal_distribution_probability_theorem_l4084_408414

/-- A random variable following a normal distribution -/
structure NormalRandomVariable where
  μ : ℝ
  σ : ℝ
  hσ_pos : σ > 0

/-- The probability that a normal random variable is less than a given value -/
noncomputable def prob_less (ξ : NormalRandomVariable) (x : ℝ) : ℝ := sorry

/-- The probability that a normal random variable is greater than a given value -/
noncomputable def prob_greater (ξ : NormalRandomVariable) (x : ℝ) : ℝ := sorry

/-- The probability that a normal random variable is between two given values -/
noncomputable def prob_between (ξ : NormalRandomVariable) (a b : ℝ) : ℝ := sorry

theorem normal_distribution_probability_theorem (ξ : NormalRandomVariable) 
  (h1 : prob_less ξ (-3) = 0.2)
  (h2 : prob_greater ξ 1 = 0.2) :
  prob_between ξ (-1) 1 = 0.3 := by sorry

end NUMINAMATH_CALUDE_normal_distribution_probability_theorem_l4084_408414


namespace NUMINAMATH_CALUDE_outside_bookshop_discount_l4084_408469

/-- The price of a math textbook in the school bookshop -/
def school_price : ℝ := 45

/-- The amount Peter saves by buying 3 math textbooks from outside bookshops -/
def savings : ℝ := 27

/-- The number of textbooks Peter buys -/
def num_textbooks : ℕ := 3

/-- The percentage discount offered by outside bookshops -/
def discount_percentage : ℝ := 20

theorem outside_bookshop_discount :
  let outside_price := school_price - (savings / num_textbooks)
  discount_percentage = (school_price - outside_price) / school_price * 100 :=
by sorry

end NUMINAMATH_CALUDE_outside_bookshop_discount_l4084_408469


namespace NUMINAMATH_CALUDE_evil_vile_live_l4084_408450

theorem evil_vile_live (E V I L : Nat) : 
  E ≠ 0 → V ≠ 0 → I ≠ 0 → L ≠ 0 →
  E < 10 → V < 10 → I < 10 → L < 10 →
  (1000 * E + 100 * V + 10 * I + L) % 73 = 0 →
  (1000 * V + 100 * I + 10 * L + E) % 74 = 0 →
  1000 * L + 100 * I + 10 * V + E = 5499 := by
sorry

end NUMINAMATH_CALUDE_evil_vile_live_l4084_408450


namespace NUMINAMATH_CALUDE_snowfall_difference_l4084_408439

/-- Snowfall difference calculation -/
theorem snowfall_difference (bald_mountain : ℝ) (billy_mountain : ℝ) (mount_pilot : ℝ) :
  bald_mountain = 1.5 →
  billy_mountain = 3.5 →
  mount_pilot = 1.26 →
  (billy_mountain + mount_pilot - bald_mountain) * 100 = 326 := by
  sorry

end NUMINAMATH_CALUDE_snowfall_difference_l4084_408439


namespace NUMINAMATH_CALUDE_day_150_of_previous_year_is_wednesday_l4084_408440

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a year -/
structure Year where
  value : ℕ

/-- Function to determine the day of the week for a given day number in a year -/
def dayOfWeek (year : Year) (dayNumber : ℕ) : DayOfWeek :=
  sorry

/-- Function to determine if a year is a leap year -/
def isLeapYear (year : Year) : Bool :=
  sorry

theorem day_150_of_previous_year_is_wednesday 
  (P : Year)
  (h1 : dayOfWeek P 250 = DayOfWeek.Friday)
  (h2 : dayOfWeek (Year.mk (P.value + 1)) 300 = DayOfWeek.Friday)
  : dayOfWeek (Year.mk (P.value - 1)) 150 = DayOfWeek.Wednesday :=
sorry

end NUMINAMATH_CALUDE_day_150_of_previous_year_is_wednesday_l4084_408440


namespace NUMINAMATH_CALUDE_morning_routine_ratio_l4084_408426

def total_routine_time : ℕ := 45
def coffee_bagel_time : ℕ := 15

theorem morning_routine_ratio :
  (total_routine_time - coffee_bagel_time) / coffee_bagel_time = 2 := by
  sorry

end NUMINAMATH_CALUDE_morning_routine_ratio_l4084_408426


namespace NUMINAMATH_CALUDE_sequence_a_odd_l4084_408407

def sequence_a : ℕ → ℤ
  | 0 => 2
  | 1 => 7
  | (n + 2) => sequence_a (n + 1)

axiom sequence_a_positive (n : ℕ) : 0 < sequence_a n

axiom sequence_a_inequality (n : ℕ) (h : n ≥ 2) :
  -1/2 < (sequence_a n - (sequence_a (n-1))^2 / sequence_a (n-2)) ∧
  (sequence_a n - (sequence_a (n-1))^2 / sequence_a (n-2)) ≤ 1/2

theorem sequence_a_odd (n : ℕ) (h : n > 1) : Odd (sequence_a n) := by
  sorry

end NUMINAMATH_CALUDE_sequence_a_odd_l4084_408407


namespace NUMINAMATH_CALUDE_eight_bead_bracelet_arrangements_l4084_408422

/-- The number of distinct arrangements of n beads on a bracelet, 
    considering rotational symmetry but not reflection -/
def bracelet_arrangements (n : ℕ) : ℕ := Nat.factorial n / n

/-- Theorem: The number of distinct arrangements of 8 beads on a bracelet, 
    considering rotational symmetry but not reflection, is 5040 -/
theorem eight_bead_bracelet_arrangements :
  bracelet_arrangements 8 = 5040 := by
  sorry

end NUMINAMATH_CALUDE_eight_bead_bracelet_arrangements_l4084_408422


namespace NUMINAMATH_CALUDE_remaining_drivable_distance_l4084_408466

/-- Proves the remaining drivable distance after a trip --/
theorem remaining_drivable_distance
  (fuel_efficiency : ℝ)
  (tank_capacity : ℝ)
  (trip_distance : ℝ)
  (h1 : fuel_efficiency = 20)
  (h2 : tank_capacity = 16)
  (h3 : trip_distance = 220) :
  fuel_efficiency * tank_capacity - trip_distance = 100 :=
by
  sorry

end NUMINAMATH_CALUDE_remaining_drivable_distance_l4084_408466


namespace NUMINAMATH_CALUDE_mary_garbage_bill_calculation_l4084_408420

/-- Calculates Mary's garbage bill given the specified conditions -/
def maryGarbageBill (weeksInMonth : ℕ) (trashBinCost recyclingBinCost greenWasteBinCost : ℚ)
  (trashBinCount recyclingBinCount greenWasteBinCount : ℕ) (serviceFee : ℚ)
  (discountRate : ℚ) (inappropriateItemsFine lateFee : ℚ) : ℚ :=
  let weeklyBinCost := trashBinCost * trashBinCount + recyclingBinCost * recyclingBinCount +
                       greenWasteBinCost * greenWasteBinCount
  let monthlyBinCost := weeklyBinCost * weeksInMonth
  let totalBeforeDiscount := monthlyBinCost + serviceFee
  let discountAmount := totalBeforeDiscount * discountRate
  let totalAfterDiscount := totalBeforeDiscount - discountAmount
  totalAfterDiscount + inappropriateItemsFine + lateFee

/-- Theorem stating that Mary's garbage bill is $134.14 under the given conditions -/
theorem mary_garbage_bill_calculation :
  maryGarbageBill 4 10 5 3 2 1 1 15 (18/100) 20 10 = 134.14 := by
  sorry

end NUMINAMATH_CALUDE_mary_garbage_bill_calculation_l4084_408420


namespace NUMINAMATH_CALUDE_cricket_average_score_l4084_408401

theorem cricket_average_score 
  (total_matches : ℕ) 
  (matches_set1 : ℕ) 
  (matches_set2 : ℕ) 
  (avg_score_set1 : ℝ) 
  (avg_score_set2 : ℝ) 
  (h1 : total_matches = matches_set1 + matches_set2)
  (h2 : matches_set1 = 2)
  (h3 : matches_set2 = 3)
  (h4 : avg_score_set1 = 20)
  (h5 : avg_score_set2 = 30) :
  (matches_set1 * avg_score_set1 + matches_set2 * avg_score_set2) / total_matches = 26 := by
  sorry

end NUMINAMATH_CALUDE_cricket_average_score_l4084_408401


namespace NUMINAMATH_CALUDE_baby_grab_outcomes_l4084_408413

theorem baby_grab_outcomes (educational_items living_items entertainment_items : ℕ) 
  (h1 : educational_items = 4)
  (h2 : living_items = 3)
  (h3 : entertainment_items = 4) :
  educational_items + living_items + entertainment_items = 11 := by
  sorry

end NUMINAMATH_CALUDE_baby_grab_outcomes_l4084_408413


namespace NUMINAMATH_CALUDE_complex_number_coordinates_i_times_one_minus_i_l4084_408410

theorem complex_number_coordinates : Complex → Complex → Prop :=
  fun z w => z = w

theorem i_times_one_minus_i (i : Complex) (h : i * i = -1) :
  complex_number_coordinates (i * (1 - i)) (1 + i) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_coordinates_i_times_one_minus_i_l4084_408410


namespace NUMINAMATH_CALUDE_triangle_angle_bounds_l4084_408441

theorem triangle_angle_bounds (y : ℝ) : 
  y > 0 → 
  y + 10 > y + 5 → 
  y + 10 > 4 * y →
  y + 5 + 4 * y > y + 10 →
  y + 5 + y + 10 > 4 * y →
  4 * y + y + 10 > y + 5 →
  (∃ (p q : ℝ), p < y ∧ y < q ∧ 
    (∀ (p' q' : ℝ), p' < y ∧ y < q' → q' - p' ≥ q - p) ∧
    q - p = 25 / 12) :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_bounds_l4084_408441


namespace NUMINAMATH_CALUDE_z_in_third_quadrant_l4084_408435

def is_in_third_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im < 0

theorem z_in_third_quadrant :
  let z : ℂ := -1 - 2*I
  is_in_third_quadrant z :=
by sorry

end NUMINAMATH_CALUDE_z_in_third_quadrant_l4084_408435


namespace NUMINAMATH_CALUDE_exactly_three_combinations_l4084_408498

/-- Represents a combination of banknotes -/
structure BanknoteCombination :=
  (n_2000 : Nat)
  (n_1000 : Nat)
  (n_500  : Nat)
  (n_200  : Nat)

/-- Checks if a combination is valid according to the problem conditions -/
def isValidCombination (c : BanknoteCombination) : Prop :=
  c.n_2000 + c.n_1000 + c.n_500 + c.n_200 = 10 ∧
  2000 * c.n_2000 + 1000 * c.n_1000 + 500 * c.n_500 + 200 * c.n_200 = 5000

/-- The set of all valid combinations -/
def validCombinations : Set BanknoteCombination :=
  { c | isValidCombination c }

/-- The three specific combinations mentioned in the solution -/
def solution1 : BanknoteCombination := ⟨0, 0, 10, 0⟩
def solution2 : BanknoteCombination := ⟨1, 0, 4, 5⟩
def solution3 : BanknoteCombination := ⟨0, 3, 2, 5⟩

/-- Theorem stating that there are exactly three valid combinations -/
theorem exactly_three_combinations :
  validCombinations = {solution1, solution2, solution3} := by sorry

#check exactly_three_combinations

end NUMINAMATH_CALUDE_exactly_three_combinations_l4084_408498


namespace NUMINAMATH_CALUDE_button_probability_l4084_408480

/-- Represents a jar containing buttons -/
structure Jar :=
  (red : ℕ)
  (blue : ℕ)

/-- Calculate the probability of choosing a blue button from a jar -/
def blueProbability (jar : Jar) : ℚ :=
  jar.blue / (jar.red + jar.blue)

theorem button_probability (jarA jarB : Jar) : 
  jarA.red = 6 ∧ 
  jarA.blue = 10 ∧ 
  jarB.red = 3 ∧ 
  jarB.blue = 5 ∧ 
  (jarA.red + jarA.blue : ℚ) = 2/3 * (6 + 10) →
  blueProbability jarA * blueProbability jarB = 25/64 := by
  sorry

#check button_probability

end NUMINAMATH_CALUDE_button_probability_l4084_408480


namespace NUMINAMATH_CALUDE_expected_adjacent_red_pairs_l4084_408405

/-- The number of cards in a standard deck -/
def deckSize : ℕ := 52

/-- The number of red cards in a standard deck -/
def redCards : ℕ := 26

/-- The probability that a card adjacent to a red card is also red -/
def probAdjacentRed : ℚ := 25 / 51

theorem expected_adjacent_red_pairs (deckSize : ℕ) (redCards : ℕ) (probAdjacentRed : ℚ) :
  deckSize = 52 → redCards = 26 → probAdjacentRed = 25 / 51 →
  (redCards : ℚ) * probAdjacentRed = 650 / 51 := by
  sorry

end NUMINAMATH_CALUDE_expected_adjacent_red_pairs_l4084_408405


namespace NUMINAMATH_CALUDE_solution_value_l4084_408487

/-- Given that (a, b) is a solution to the linear equation 2x-7y=8,
    prove that the value of the algebraic expression 17-4a+14b is 1 -/
theorem solution_value (a b : ℝ) (h : 2*a - 7*b = 8) : 17 - 4*a + 14*b = 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_value_l4084_408487


namespace NUMINAMATH_CALUDE_nina_raisins_l4084_408464

/-- The number of raisins Nina and Max received satisfies the given conditions -/
def raisin_distribution (nina max : ℕ) : Prop :=
  nina = max + 8 ∧ 
  max = nina / 3 ∧ 
  nina + max = 16

theorem nina_raisins :
  ∀ nina max : ℕ, raisin_distribution nina max → nina = 12 := by
  sorry

end NUMINAMATH_CALUDE_nina_raisins_l4084_408464


namespace NUMINAMATH_CALUDE_triangle_problem_l4084_408424

/-- Triangle sum for the nth row -/
def triangle_sum (n : ℕ) (a d : ℕ) : ℕ := 2^n * a + (2^n - 2) * d

/-- The problem statement -/
theorem triangle_problem (a d : ℕ) (ha : a > 0) (hd : d > 0) :
  (∃ n : ℕ, triangle_sum n a d = 1988) →
  (∃ n : ℕ, n = 6 ∧ a = 2 ∧ d = 30 ∧ 
    (∀ m : ℕ, triangle_sum m a d = 1988 → m ≤ n)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l4084_408424


namespace NUMINAMATH_CALUDE_swap_values_l4084_408433

theorem swap_values (a b : ℕ) (ha : a = 8) (hb : b = 17) :
  ∃ c : ℕ, (c = b) ∧ (b = a) ∧ (a = c) ∧ (a = 17 ∧ b = 8) := by
  sorry

end NUMINAMATH_CALUDE_swap_values_l4084_408433


namespace NUMINAMATH_CALUDE_simplify_expression_l4084_408453

theorem simplify_expression : (1 / ((-5^4)^2)) * (-5)^9 = -5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l4084_408453


namespace NUMINAMATH_CALUDE_bike_riders_count_l4084_408483

theorem bike_riders_count (total : ℕ) (hikers : ℕ) (bikers : ℕ) :
  total = hikers + bikers →
  hikers = bikers + 178 →
  total = 676 →
  bikers = 249 := by
sorry

end NUMINAMATH_CALUDE_bike_riders_count_l4084_408483


namespace NUMINAMATH_CALUDE_sports_participation_l4084_408465

theorem sports_participation (B C S Ba : ℕ)
  (BC BS BBa CS CBa SBa BCSL : ℕ)
  (h1 : B = 12)
  (h2 : C = 10)
  (h3 : S = 9)
  (h4 : Ba = 6)
  (h5 : BC = 5)
  (h6 : BS = 4)
  (h7 : BBa = 3)
  (h8 : CS = 2)
  (h9 : CBa = 3)
  (h10 : SBa = 2)
  (h11 : BCSL = 1) :
  B + C + S + Ba - BC - BS - BBa - CS - CBa - SBa + BCSL = 19 := by
  sorry

end NUMINAMATH_CALUDE_sports_participation_l4084_408465


namespace NUMINAMATH_CALUDE_power_sum_product_simplification_l4084_408425

theorem power_sum_product_simplification :
  (7^5 + 2^7) * (2^3 - (-2)^3)^8 = 72778137514496 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_product_simplification_l4084_408425


namespace NUMINAMATH_CALUDE_largest_integer_for_negative_quadratic_l4084_408479

theorem largest_integer_for_negative_quadratic : 
  ∃ n : ℤ, n^2 - 11*n + 28 < 0 ∧ 
  ∀ m : ℤ, m^2 - 11*m + 28 < 0 → m ≤ n ∧ 
  n = 6 := by sorry

end NUMINAMATH_CALUDE_largest_integer_for_negative_quadratic_l4084_408479


namespace NUMINAMATH_CALUDE_triangle_incenter_inequality_l4084_408476

-- Define a triangle
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the incenter of a triangle
def incenter (t : Triangle) : ℝ × ℝ := sorry

-- Define the intersection points of angle bisectors with opposite sides
def angleBisectorIntersection (t : Triangle) : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) := sorry

-- Define the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem triangle_incenter_inequality (t : Triangle) :
  let I := incenter t
  let (A', B', C') := angleBisectorIntersection t
  let ratio := (distance I t.A * distance I t.B * distance I t.C) /
               (distance A' t.A * distance B' t.B * distance C' t.C)
  (1 / 4 : ℝ) < ratio ∧ ratio ≤ (8 / 27 : ℝ) := by sorry

end NUMINAMATH_CALUDE_triangle_incenter_inequality_l4084_408476


namespace NUMINAMATH_CALUDE_envelope_ratio_l4084_408448

theorem envelope_ratio (blue_envelopes : ℕ) (yellow_diff : ℕ) (total_envelopes : ℕ)
  (h1 : blue_envelopes = 14)
  (h2 : yellow_diff = 6)
  (h3 : total_envelopes = 46) :
  ∃ (green_envelopes yellow_envelopes : ℕ),
    yellow_envelopes = blue_envelopes - yellow_diff ∧
    green_envelopes = 3 * yellow_envelopes ∧
    blue_envelopes + yellow_envelopes + green_envelopes = total_envelopes ∧
    green_envelopes / yellow_envelopes = 3 := by
  sorry

end NUMINAMATH_CALUDE_envelope_ratio_l4084_408448


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l4084_408494

theorem quadratic_equation_roots (k : ℝ) : 
  let eq := fun x : ℝ => x^2 + (2*k - 1)*x + k^2 - 1
  ∃ x₁ x₂ : ℝ, 
    (eq x₁ = 0 ∧ eq x₂ = 0) ∧ 
    (x₁ ≠ x₂) ∧
    (x₁^2 + x₂^2 = 16 + x₁*x₂) →
  k = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l4084_408494


namespace NUMINAMATH_CALUDE_defective_shipped_percentage_l4084_408467

/-- The percentage of units that are defective -/
def defective_percentage : ℝ := 6

/-- The percentage of defective units that are shipped for sale -/
def shipped_percentage : ℝ := 4

/-- The result we want to prove -/
def result : ℝ := 0.24

theorem defective_shipped_percentage :
  (defective_percentage / 100) * (shipped_percentage / 100) * 100 = result := by
  sorry

end NUMINAMATH_CALUDE_defective_shipped_percentage_l4084_408467


namespace NUMINAMATH_CALUDE_set_operations_and_range_l4084_408485

def A : Set ℝ := {x | 3 ≤ x ∧ x ≤ 9}
def B : Set ℝ := {x | 2 < x ∧ x < 5}
def C (a : ℝ) : Set ℝ := {x | x > a}

theorem set_operations_and_range (a : ℝ) : 
  (A ∪ B = {x | 2 < x ∧ x ≤ 9}) ∧ 
  (B ∩ C a = ∅ → a ≥ 5) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_and_range_l4084_408485


namespace NUMINAMATH_CALUDE_heart_shape_area_l4084_408452

/-- The area of a heart shape composed of specific geometric elements -/
theorem heart_shape_area : 
  let π : ℝ := 3.14
  let semicircle_diameter : ℝ := 10
  let sector_radius : ℝ := 10
  let sector_angle : ℝ := 45
  let square_side : ℝ := 10
  let semicircle_area : ℝ := 2 * (1/2 * π * (semicircle_diameter/2)^2)
  let sector_area : ℝ := 2 * ((sector_angle/360) * π * sector_radius^2)
  let square_area : ℝ := square_side^2
  semicircle_area + sector_area + square_area = 257 := by
  sorry

end NUMINAMATH_CALUDE_heart_shape_area_l4084_408452


namespace NUMINAMATH_CALUDE_expression_value_l4084_408411

theorem expression_value (b c a : ℤ) (h1 : b = 10) (h2 : c = 3) (h3 : a = 2 * b) :
  (a - (b - c)) - ((a - b) - c) = 6 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l4084_408411


namespace NUMINAMATH_CALUDE_river_current_speed_l4084_408473

/-- The speed of a river's current given swimmer's performance -/
theorem river_current_speed 
  (distance : ℝ) 
  (time : ℝ) 
  (still_water_speed : ℝ) 
  (h1 : distance = 7) 
  (h2 : time = 3.684210526315789) 
  (h3 : still_water_speed = 4.4) : 
  ∃ current_speed : ℝ, 
    current_speed = 2.5 ∧ 
    distance / time = still_water_speed - current_speed := by
  sorry

#check river_current_speed

end NUMINAMATH_CALUDE_river_current_speed_l4084_408473


namespace NUMINAMATH_CALUDE_inverse_i_minus_inverse_l4084_408471

/-- Given a complex number i where i^2 = -1, prove that (i - i⁻¹)⁻¹ = -i/2 -/
theorem inverse_i_minus_inverse (i : ℂ) (h : i^2 = -1) : (i - i⁻¹)⁻¹ = -i/2 := by
  sorry

end NUMINAMATH_CALUDE_inverse_i_minus_inverse_l4084_408471


namespace NUMINAMATH_CALUDE_line_intersects_curve_l4084_408432

/-- Given real numbers a and b where ab ≠ 0, the line ax - y + b = 0 intersects
    the curve bx² + ay² = ab. -/
theorem line_intersects_curve (a b : ℝ) (h : a * b ≠ 0) :
  ∃ (x y : ℝ), (a * x - y + b = 0) ∧ (b * x^2 + a * y^2 = a * b) := by
  sorry

end NUMINAMATH_CALUDE_line_intersects_curve_l4084_408432


namespace NUMINAMATH_CALUDE_number_ratio_l4084_408481

theorem number_ratio (x : ℚ) (h : 3 * (2 * x + 15) = 75) : x / (2 * x) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_number_ratio_l4084_408481


namespace NUMINAMATH_CALUDE_ellipse_from_hyperbola_l4084_408470

/-- The hyperbola from which we derive the ellipse parameters -/
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 12 = -1

/-- The vertex of the hyperbola becomes the focus of the ellipse -/
def hyperbola_vertex_to_ellipse_focus (x y : ℝ) : Prop :=
  hyperbola x y → (x = 0 ∧ (y = 2 ∨ y = -2))

/-- The focus of the hyperbola becomes the vertex of the ellipse -/
def hyperbola_focus_to_ellipse_vertex (x y : ℝ) : Prop :=
  hyperbola x y → (x = 0 ∧ (y = 4 ∨ y = -4))

/-- The resulting ellipse equation -/
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 16 = 1

/-- Theorem stating that the ellipse with the given properties has the specified equation -/
theorem ellipse_from_hyperbola :
  (∀ x y, hyperbola_vertex_to_ellipse_focus x y) →
  (∀ x y, hyperbola_focus_to_ellipse_vertex x y) →
  (∀ x y, ellipse x y) :=
sorry

end NUMINAMATH_CALUDE_ellipse_from_hyperbola_l4084_408470


namespace NUMINAMATH_CALUDE_smallest_x_for_cube_l4084_408491

theorem smallest_x_for_cube (x : ℕ+) (N : ℤ) : 
  (∀ y : ℕ+, y < x → ¬∃ M : ℤ, 1890 * y = M^3) ∧ 
  1890 * x = N^3 ↔ 
  x = 4900 := by
sorry

end NUMINAMATH_CALUDE_smallest_x_for_cube_l4084_408491


namespace NUMINAMATH_CALUDE_geometric_sequence_a3_l4084_408417

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_a3 (a : ℕ → ℝ) :
  geometric_sequence a →
  a 4 - a 2 = 6 →
  a 5 - a 1 = 15 →
  a 3 = 4 ∨ a 3 = -4 :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a3_l4084_408417


namespace NUMINAMATH_CALUDE_function_extrema_l4084_408486

open Real

theorem function_extrema (a : ℝ) (h : a > 0) :
  let f : ℝ → ℝ := fun x => (sin x + a) / sin x
  ∃ (m : ℝ), (∀ x, 0 < x → x < π → f x ≥ m) ∧
  (∀ M : ℝ, ∃ x, 0 < x ∧ x < π ∧ f x > M) := by
  sorry

end NUMINAMATH_CALUDE_function_extrema_l4084_408486
