import Mathlib

namespace sqrt_equation_solution_l2815_281563

theorem sqrt_equation_solution :
  ∀ x : ℝ, Real.sqrt (5 * x + 9) = 12 → x = 27 :=
by
  sorry

end sqrt_equation_solution_l2815_281563


namespace duck_cow_leg_count_l2815_281566

theorem duck_cow_leg_count :
  ∀ (num_ducks : ℕ),
  let num_cows : ℕ := 12
  let total_heads : ℕ := num_ducks + num_cows
  let total_legs : ℕ := 2 * num_ducks + 4 * num_cows
  total_legs - 2 * total_heads = 24 :=
by
  sorry

end duck_cow_leg_count_l2815_281566


namespace smallest_cube_for_pyramid_l2815_281534

/-- Represents a pyramid with a square base -/
structure Pyramid where
  height : ℝ
  baseLength : ℝ

/-- Represents a cube -/
structure Cube where
  sideLength : ℝ

/-- Calculates the volume of a cube -/
def cubeVolume (c : Cube) : ℝ := c.sideLength ^ 3

/-- Determines if a cube can contain a pyramid standing upright -/
def canContainPyramid (c : Cube) (p : Pyramid) : Prop :=
  c.sideLength ≥ p.height ∧ c.sideLength ≥ p.baseLength

theorem smallest_cube_for_pyramid (p : Pyramid) (h1 : p.height = 12) (h2 : p.baseLength = 10) :
  ∃ (c : Cube), canContainPyramid c p ∧
    cubeVolume c = 1728 ∧
    ∀ (c' : Cube), canContainPyramid c' p → cubeVolume c' ≥ cubeVolume c :=
by sorry

end smallest_cube_for_pyramid_l2815_281534


namespace two_in_A_implies_a_is_one_or_two_l2815_281585

-- Define the set A
def A (a : ℝ) : Set ℝ := {-2, 2*a, a^2 - a}

-- Theorem statement
theorem two_in_A_implies_a_is_one_or_two :
  ∀ a : ℝ, 2 ∈ A a → a = 1 ∨ a = 2 :=
by sorry

end two_in_A_implies_a_is_one_or_two_l2815_281585


namespace total_flowers_l2815_281524

theorem total_flowers (roses tulips lilies : ℕ) : 
  roses = 58 ∧ 
  tulips = roses - 15 ∧ 
  lilies = roses + 25 → 
  roses + tulips + lilies = 184 := by
sorry

end total_flowers_l2815_281524


namespace trigonometric_expression_equals_one_l2815_281555

theorem trigonometric_expression_equals_one : 
  (Real.sin (15 * π / 180) * Real.cos (15 * π / 180) + 
   Real.cos (165 * π / 180) * Real.cos (105 * π / 180)) / 
  (Real.sin (19 * π / 180) * Real.cos (11 * π / 180) + 
   Real.cos (161 * π / 180) * Real.cos (101 * π / 180)) = 1 := by
  sorry

end trigonometric_expression_equals_one_l2815_281555


namespace sixth_row_third_number_l2815_281518

/-- Represents the sequence of positive odd numbers -/
def oddSequence (n : ℕ) : ℕ := 2 * n - 1

/-- Represents the number of elements in the nth row of the table -/
def rowSize (n : ℕ) : ℕ := 2^(n - 1)

/-- Represents the total number of elements up to and including the nth row -/
def totalElements (n : ℕ) : ℕ := 2^n - 1

theorem sixth_row_third_number : 
  let rowNumber := 6
  let positionInRow := 3
  oddSequence (totalElements (rowNumber - 1) + positionInRow) = 67 := by
  sorry

end sixth_row_third_number_l2815_281518


namespace david_min_score_l2815_281527

def david_scores : List Int := [88, 92, 75, 83, 90]

def current_average : Rat :=
  (david_scores.sum : Rat) / david_scores.length

def target_average : Rat := current_average + 4

def min_score : Int :=
  Int.ceil ((target_average * (david_scores.length + 1) : Rat) - david_scores.sum)

theorem david_min_score :
  min_score = 110 := by sorry

end david_min_score_l2815_281527


namespace intersection_of_A_and_B_l2815_281589

def A : Set ℤ := {-1, 1}
def B : Set ℤ := {-1, 3}

theorem intersection_of_A_and_B : A ∩ B = {-1} := by sorry

end intersection_of_A_and_B_l2815_281589


namespace dish_price_proof_l2815_281548

/-- The original price of a dish satisfying the given conditions -/
def original_price : ℝ := 34

/-- The discount rate applied to the original price -/
def discount_rate : ℝ := 0.1

/-- The tip rate applied to either the original or discounted price -/
def tip_rate : ℝ := 0.15

/-- The difference in total payments between the two people -/
def payment_difference : ℝ := 0.51

theorem dish_price_proof :
  let discounted_price := original_price * (1 - discount_rate)
  let payment1 := discounted_price + original_price * tip_rate
  let payment2 := discounted_price + discounted_price * tip_rate
  payment1 - payment2 = payment_difference := by sorry

end dish_price_proof_l2815_281548


namespace no_valid_coloring_200_points_l2815_281557

/-- Represents a coloring of points and segments -/
structure Coloring (n : ℕ) (k : ℕ) where
  pointColor : Fin n → Fin k
  segmentColor : Fin n → Fin n → Fin k

/-- Predicate for a valid coloring -/
def isValidColoring (n : ℕ) (k : ℕ) (c : Coloring n k) : Prop :=
  ∀ i j : Fin n, i ≠ j →
    c.pointColor i ≠ c.pointColor j ∧
    c.pointColor i ≠ c.segmentColor i j ∧
    c.pointColor j ≠ c.segmentColor i j

/-- Theorem stating the impossibility of valid coloring for 200 points with 7 or 10 colors -/
theorem no_valid_coloring_200_points :
  ¬ (∃ c : Coloring 200 7, isValidColoring 200 7 c) ∧
  ¬ (∃ c : Coloring 200 10, isValidColoring 200 10 c) := by
  sorry


end no_valid_coloring_200_points_l2815_281557


namespace eliminate_x_y_l2815_281588

-- Define the tangent and cotangent functions
noncomputable def tg (x : ℝ) : ℝ := Real.tan x
noncomputable def ctg (x : ℝ) : ℝ := 1 / Real.tan x

-- State the theorem
theorem eliminate_x_y (x y a b c : ℝ) 
  (h1 : tg x + tg y = a)
  (h2 : ctg x + ctg y = b)
  (h3 : x + y = c) :
  ctg c = 1 / a - 1 / b :=
by sorry

end eliminate_x_y_l2815_281588


namespace f_sum_two_three_l2815_281569

/-- An odd function satisfying the given conditions -/
def f (x : ℝ) : ℝ := sorry

/-- f is an odd function -/
axiom f_odd (x : ℝ) : f (-x) = -f x

/-- f satisfies the symmetry condition -/
axiom f_sym (x : ℝ) : f (3/2 + x) = -f (3/2 - x)

/-- f(1) = 2 -/
axiom f_one : f 1 = 2

/-- Theorem: f(2) + f(3) = -2 -/
theorem f_sum_two_three : f 2 + f 3 = -2 := by sorry

end f_sum_two_three_l2815_281569


namespace least_number_remainder_l2815_281581

theorem least_number_remainder (n : ℕ) (h1 : n % 20 = 14) (h2 : n % 2535 = 1929) (h3 : n = 1394) : n % 40 = 34 := by
  sorry

end least_number_remainder_l2815_281581


namespace coin_distribution_six_boxes_l2815_281556

def coinDistribution (n : ℕ) : ℕ :=
  match n with
  | 0 => 0
  | 1 => 1
  | m + 1 => 2 * coinDistribution m

theorem coin_distribution_six_boxes :
  coinDistribution 6 = 32 := by
  sorry

end coin_distribution_six_boxes_l2815_281556


namespace arithmetic_sequence_2005_unique_position_2005_l2815_281540

/-- An arithmetic sequence with first term 1 and common difference 3 -/
def arithmetic_sequence (n : ℕ) : ℤ := 1 + 3 * (n - 1)

/-- The theorem stating that the 669th term of the sequence is 2005 -/
theorem arithmetic_sequence_2005 : arithmetic_sequence 669 = 2005 := by
  sorry

/-- The theorem stating that 669 is the unique position where the sequence equals 2005 -/
theorem unique_position_2005 : ∀ n : ℕ, arithmetic_sequence n = 2005 ↔ n = 669 := by
  sorry

end arithmetic_sequence_2005_unique_position_2005_l2815_281540


namespace intern_teacher_distribution_l2815_281519

/-- The number of ways to distribute n teachers among k classes with at least one teacher per class -/
def distribution_schemes (n k : ℕ) : ℕ :=
  if n < k then 0
  else (n - k + 1).choose k * (k - 1).choose (n - k)

/-- Theorem: There are 60 ways to distribute 5 intern teachers among 3 freshman classes with at least 1 teacher in each class -/
theorem intern_teacher_distribution : distribution_schemes 5 3 = 60 := by
  sorry


end intern_teacher_distribution_l2815_281519


namespace two_ab_value_l2815_281567

theorem two_ab_value (a b : ℝ) 
  (h1 : a^4 + a^2*b^2 + b^4 = 900) 
  (h2 : a^2 + a*b + b^2 = 45) : 
  2*a*b = 25 := by
sorry

end two_ab_value_l2815_281567


namespace negation_equivalence_l2815_281521

theorem negation_equivalence :
  (¬ ∃ x₀ : ℝ, x₀^2 + 2*x₀ + 5 = 0) ↔ (∀ x : ℝ, x^2 + 2*x + 5 ≠ 0) :=
by sorry

end negation_equivalence_l2815_281521


namespace square_areas_tiles_l2815_281572

theorem square_areas_tiles (x : ℝ) : 
  x > 0 ∧ 
  x^2 + (x + 12)^2 = 2120 → 
  x = 26 ∧ x + 12 = 38 := by
sorry

end square_areas_tiles_l2815_281572


namespace range_of_a_l2815_281551

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then (1/2)^x - 7 else Real.sqrt x

-- State the theorem
theorem range_of_a (a : ℝ) (h : f a < 1) : -3 < a ∧ a < 1 := by
  sorry

end range_of_a_l2815_281551


namespace last_three_average_l2815_281525

theorem last_three_average (numbers : List ℝ) : 
  numbers.length = 5 →
  numbers.sum / numbers.length = 54 →
  (numbers.take 2).sum / 2 = 48 →
  (numbers.drop 2).sum / 3 = 58 := by
sorry

end last_three_average_l2815_281525


namespace halloween_candy_proof_l2815_281544

/-- Represents the number of candy pieces Debby's sister had -/
def sisters_candy : ℕ := 42

theorem halloween_candy_proof :
  let debbys_candy : ℕ := 32
  let eaten_candy : ℕ := 35
  let remaining_candy : ℕ := 39
  debbys_candy + sisters_candy - eaten_candy = remaining_candy :=
by
  sorry

#check halloween_candy_proof

end halloween_candy_proof_l2815_281544


namespace M_on_angle_bisector_coordinates_M_distance_to_x_axis_coordinates_l2815_281508

def M (m : ℚ) : ℚ × ℚ := (m - 1, 2 * m + 3)

def on_angle_bisector (p : ℚ × ℚ) : Prop :=
  p.1 = p.2 ∨ p.1 = -p.2

def distance_to_x_axis (p : ℚ × ℚ) : ℚ := |p.2|

theorem M_on_angle_bisector_coordinates (m : ℚ) :
  on_angle_bisector (M m) → M m = (-5/3, 5/3) ∨ M m = (-5, -5) := by sorry

theorem M_distance_to_x_axis_coordinates (m : ℚ) :
  distance_to_x_axis (M m) = 1 → M m = (-2, 1) ∨ M m = (-3, -1) := by sorry

end M_on_angle_bisector_coordinates_M_distance_to_x_axis_coordinates_l2815_281508


namespace last_digit_of_power_tower_plus_one_l2815_281559

theorem last_digit_of_power_tower_plus_one :
  (2^(2^1989) + 1) % 10 = 7 := by
  sorry

end last_digit_of_power_tower_plus_one_l2815_281559


namespace prob_at_most_two_heads_prove_prob_at_most_two_heads_l2815_281538

/-- The probability of getting at most 2 heads when tossing three unbiased coins -/
theorem prob_at_most_two_heads : ℚ :=
  7 / 8

/-- Prove that the probability of getting at most 2 heads when tossing three unbiased coins is 7/8 -/
theorem prove_prob_at_most_two_heads :
  prob_at_most_two_heads = 7 / 8 := by
  sorry

end prob_at_most_two_heads_prove_prob_at_most_two_heads_l2815_281538


namespace monochromatic_right_triangle_exists_l2815_281536

/-- A point on the contour of a square -/
structure ContourPoint where
  x : ℝ
  y : ℝ

/-- Color of a point -/
inductive Color
  | Blue
  | Red

/-- A coloring of the contour of a square -/
def Coloring := ContourPoint → Color

/-- Predicate to check if three points form a right triangle -/
def is_right_triangle (p1 p2 p3 : ContourPoint) : Prop :=
  sorry

/-- Theorem: For any coloring of the contour of a square, there exists a right triangle
    with vertices of the same color -/
theorem monochromatic_right_triangle_exists (coloring : Coloring) :
  ∃ (p1 p2 p3 : ContourPoint),
    is_right_triangle p1 p2 p3 ∧
    coloring p1 = coloring p2 ∧
    coloring p2 = coloring p3 :=
  sorry

end monochromatic_right_triangle_exists_l2815_281536


namespace shirt_pricing_l2815_281501

theorem shirt_pricing (total_shirts : Nat) (price_shirt1 price_shirt2 : ℝ) (min_avg_price_remaining : ℝ) :
  total_shirts = 5 →
  price_shirt1 = 30 →
  price_shirt2 = 20 →
  min_avg_price_remaining = 33.333333333333336 →
  (price_shirt1 + price_shirt2 + (total_shirts - 2) * min_avg_price_remaining) / total_shirts = 30 := by
  sorry

end shirt_pricing_l2815_281501


namespace matrix_multiplication_result_l2815_281510

theorem matrix_multiplication_result : 
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![3, 1; 4, -2]
  let B : Matrix (Fin 2) (Fin 2) ℤ := !![7, -3; 2, 2]
  A * B = !![23, -7; 24, -16] := by
  sorry

end matrix_multiplication_result_l2815_281510


namespace division_remainder_composition_l2815_281511

theorem division_remainder_composition (P D Q R D' Q' R' : ℕ) 
  (h1 : P = Q * D + R) 
  (h2 : Q = D' * Q' + R') : 
  ∃ k : ℕ, P = (D * D') * Q' + (R + R' * D) + k * (D * D') := by
  sorry

end division_remainder_composition_l2815_281511


namespace shekars_mathematics_marks_l2815_281529

/-- Represents the marks scored by Shekar in different subjects -/
structure Marks where
  mathematics : ℕ
  science : ℕ
  social_studies : ℕ
  english : ℕ
  biology : ℕ

/-- Calculates the average of marks -/
def average (m : Marks) : ℚ :=
  (m.mathematics + m.science + m.social_studies + m.english + m.biology) / 5

/-- Theorem stating that Shekar's marks in mathematics are 76 -/
theorem shekars_mathematics_marks :
  ∃ m : Marks,
    m.science = 65 ∧
    m.social_studies = 82 ∧
    m.english = 67 ∧
    m.biology = 75 ∧
    average m = 73 ∧
    m.mathematics = 76 := by
  sorry


end shekars_mathematics_marks_l2815_281529


namespace jean_spots_on_sides_l2815_281591

/-- Represents the number of spots on different parts of Jean the jaguar. -/
structure JeanSpots where
  total : ℕ
  upperTorso : ℕ
  backAndHindquarters : ℕ
  sides : ℕ

/-- Theorem stating the number of spots on Jean's sides given the distribution of spots. -/
theorem jean_spots_on_sides (j : JeanSpots) 
  (h1 : j.upperTorso = j.total / 2)
  (h2 : j.backAndHindquarters = j.total / 3)
  (h3 : j.sides = j.total - j.upperTorso - j.backAndHindquarters)
  (h4 : j.upperTorso = 30) :
  j.sides = 10 := by
  sorry

end jean_spots_on_sides_l2815_281591


namespace xyz_value_l2815_281578

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 40)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 10) :
  x * y * z = 10 := by
sorry

end xyz_value_l2815_281578


namespace minimum_sum_geometric_sequence_l2815_281547

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, ∃ r : ℝ, r > 0 ∧ a (n + 1) = r * a n

theorem minimum_sum_geometric_sequence (a : ℕ → ℝ) 
    (h_geo : GeometricSequence a)
    (h_positive : ∀ n, a n > 0)
    (h_product : a 3 * a 5 = 64) :
    ∃ (m : ℝ), m = 16 ∧ ∀ x y, x > 0 → y > 0 → x * y = 64 → x + y ≥ m :=
  sorry

end minimum_sum_geometric_sequence_l2815_281547


namespace oliver_unwashed_shirts_l2815_281586

theorem oliver_unwashed_shirts 
  (short_sleeve : ℕ) 
  (long_sleeve : ℕ) 
  (washed : ℕ) 
  (h1 : short_sleeve = 39)
  (h2 : long_sleeve = 47)
  (h3 : washed = 20) :
  short_sleeve + long_sleeve - washed = 66 := by
sorry

end oliver_unwashed_shirts_l2815_281586


namespace base_10_to_base_7_l2815_281577

theorem base_10_to_base_7 :
  ∃ (a b c d : ℕ),
    804 = a * 7^3 + b * 7^2 + c * 7^1 + d * 7^0 ∧
    a < 7 ∧ b < 7 ∧ c < 7 ∧ d < 7 ∧
    a = 2 ∧ b = 2 ∧ c = 2 ∧ d = 6 :=
by sorry

end base_10_to_base_7_l2815_281577


namespace complex_modulus_example_l2815_281506

theorem complex_modulus_example : Complex.abs (-3 - (5/4)*Complex.I) = 13/4 := by
  sorry

end complex_modulus_example_l2815_281506


namespace same_color_probability_l2815_281598

/-- The probability of drawing three marbles of the same color from a bag containing
    3 red marbles, 7 white marbles, and 5 blue marbles, without replacement. -/
theorem same_color_probability (red : ℕ) (white : ℕ) (blue : ℕ) 
    (h_red : red = 3) (h_white : white = 7) (h_blue : blue = 5) :
    let total := red + white + blue
    let p_all_red := (red / total) * ((red - 1) / (total - 1)) * ((red - 2) / (total - 2))
    let p_all_white := (white / total) * ((white - 1) / (total - 1)) * ((white - 2) / (total - 2))
    let p_all_blue := (blue / total) * ((blue - 1) / (total - 1)) * ((blue - 2) / (total - 2))
    p_all_red + p_all_white + p_all_blue = 23 / 455 := by
  sorry

end same_color_probability_l2815_281598


namespace min_break_even_quantity_l2815_281565

/-- The cost function for a product -/
def cost (x : ℕ) : ℝ := 3000 + 20 * x - 0.1 * x^2

/-- The revenue function for a product -/
def revenue (x : ℕ) : ℝ := 25 * x

/-- The break-even condition -/
def breaks_even (x : ℕ) : Prop := revenue x ≥ cost x

theorem min_break_even_quantity :
  ∃ (x : ℕ), x > 0 ∧ x < 240 ∧ breaks_even x ∧
  ∀ (y : ℕ), y > 0 ∧ y < 240 ∧ breaks_even y → y ≥ 150 :=
sorry

end min_break_even_quantity_l2815_281565


namespace parabola_max_area_l2815_281545

-- Define the parabola C
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := y = x + 1

-- Define the condition for p
def p_condition (p : ℝ) : Prop := p > 0

-- Define the points A and B on the parabola
def point_on_parabola (p x y : ℝ) : Prop := parabola p x y

-- Define the condition for x₁ and x₂
def x_condition (x₁ x₂ : ℝ) : Prop := x₁ ≠ x₂ ∧ x₁ + x₂ = 4

-- Define the theorem
theorem parabola_max_area (p : ℝ) (x₁ y₁ x₂ y₂ : ℝ) :
  p_condition p →
  parabola p x₁ y₁ →
  parabola p x₂ y₂ →
  x_condition x₁ x₂ →
  (∃ (x y : ℝ), parabola p x y ∧ tangent_line x y) →
  (∃ (area : ℝ), area ≤ 8 ∧ 
    (∀ (other_area : ℝ), other_area ≤ area)) :=
by sorry

end parabola_max_area_l2815_281545


namespace triangle_sine_law_l2815_281541

theorem triangle_sine_law (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  0 < a ∧ 0 < b ∧ 0 < c →
  A = π / 3 →
  a = Real.sqrt 3 →
  c / Real.sin C = 2 := by
sorry

end triangle_sine_law_l2815_281541


namespace digit_puzzle_solution_l2815_281553

theorem digit_puzzle_solution (c o u n t s : ℕ) 
  (h1 : c + o = u)
  (h2 : u + n = t + 1)
  (h3 : t + c = s)
  (h4 : o + n + s = 15)
  (h5 : c ≠ 0 ∧ o ≠ 0 ∧ u ≠ 0 ∧ n ≠ 0 ∧ t ≠ 0 ∧ s ≠ 0)
  (h6 : c < 10 ∧ o < 10 ∧ u < 10 ∧ n < 10 ∧ t < 10 ∧ s < 10) :
  t = 7 := by sorry

end digit_puzzle_solution_l2815_281553


namespace expression_evaluation_l2815_281526

theorem expression_evaluation :
  let x : ℝ := Real.sin (30 * π / 180)
  (1 - 2 / (x - 1)) / ((x - 3) / (x^2 - 1)) = 3/2 := by
  sorry

end expression_evaluation_l2815_281526


namespace tom_books_theorem_l2815_281576

def books_problem (initial_books sold_books bought_books : ℕ) : Prop :=
  let remaining_books := initial_books - sold_books
  let final_books := remaining_books + bought_books
  final_books = 39

theorem tom_books_theorem :
  books_problem 5 4 38 := by sorry

end tom_books_theorem_l2815_281576


namespace downstream_distance_l2815_281554

/-- The distance swum downstream by a woman given certain conditions -/
theorem downstream_distance (t : ℝ) (d_up : ℝ) (v_still : ℝ) : 
  t > 0 ∧ d_up > 0 ∧ v_still > 0 →
  t = 6 ∧ d_up = 6 ∧ v_still = 5 →
  ∃ d_down : ℝ, d_down = 54 ∧ 
    d_down / (v_still + (d_up / t - v_still)) = t ∧
    d_up / (v_still - (d_up / t - v_still)) = t :=
by sorry


end downstream_distance_l2815_281554


namespace inequality_proof_l2815_281505

theorem inequality_proof (a b c d e f : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0) 
  (h_sum : d * e * f + d * e + e * f + f * d = 4) : 
  ((a + b) * d * e + (b + c) * e * f + (c + a) * f * d)^2 ≥ 
  12 * (a * b * d * e + b * c * e * f + c * a * f * d) := by
sorry

end inequality_proof_l2815_281505


namespace vowel_word_count_l2815_281550

def vowel_count : ℕ := 5
def word_length : ℕ := 5
def max_vowel_occurrence : ℕ := 3

def total_distributions : ℕ := Nat.choose (word_length + vowel_count - 1) (vowel_count - 1)

def invalid_distributions : ℕ := vowel_count * (vowel_count - 1)

theorem vowel_word_count :
  total_distributions - invalid_distributions = 106 :=
sorry

end vowel_word_count_l2815_281550


namespace max_difference_PA_PB_l2815_281513

/-- Curve C₂ -/
def C₂ (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

/-- Point A on the negative x-axis -/
def A : ℝ × ℝ := (-2, 0)

/-- Given point B -/
def B : ℝ × ℝ := (1, 1)

/-- Distance squared between two points -/
def dist_squared (p₁ p₂ : ℝ × ℝ) : ℝ :=
  (p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2

theorem max_difference_PA_PB :
  ∃ (max : ℝ), max = 2 + 2 * Real.sqrt 39 ∧
  ∀ (P : ℝ × ℝ), C₂ P.1 P.2 →
  dist_squared P A - dist_squared P B ≤ max :=
sorry

end max_difference_PA_PB_l2815_281513


namespace sin_squared_minus_cos_squared_range_l2815_281599

/-- 
Given an angle θ in standard position with terminal side passing through (x, y),
prove that sin²θ - cos²θ is between -1 and 1, inclusive.
-/
theorem sin_squared_minus_cos_squared_range (θ x y r : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 = r^2) →  -- r is the distance from origin to (x, y)
  r > 0 →  -- r is positive (implicitly given in the problem)
  Real.sin θ = y / r → 
  Real.cos θ = x / r → 
  -1 ≤ Real.sin θ^2 - Real.cos θ^2 ∧ Real.sin θ^2 - Real.cos θ^2 ≤ 1 := by
  sorry

end sin_squared_minus_cos_squared_range_l2815_281599


namespace greatest_x_value_l2815_281537

theorem greatest_x_value : ∃ (x_max : ℚ), 
  (∀ x : ℚ, ((4*x - 16) / (3*x - 4))^2 + ((4*x - 16) / (3*x - 4)) = 6 → x ≤ x_max) ∧
  ((4*x_max - 16) / (3*x_max - 4))^2 + ((4*x_max - 16) / (3*x_max - 4)) = 6 ∧
  x_max = 28/13 := by
  sorry

end greatest_x_value_l2815_281537


namespace distance_is_sqrt_51_l2815_281560

def point : ℝ × ℝ × ℝ := (3, 5, -1)
def line_point : ℝ × ℝ × ℝ := (2, 4, 6)
def line_direction : ℝ × ℝ × ℝ := (4, 3, -1)

def distance_to_line (p : ℝ × ℝ × ℝ) (l_point : ℝ × ℝ × ℝ) (l_dir : ℝ × ℝ × ℝ) : ℝ :=
  sorry

theorem distance_is_sqrt_51 :
  distance_to_line point line_point line_direction = Real.sqrt 51 :=
sorry

end distance_is_sqrt_51_l2815_281560


namespace election_winner_percentage_l2815_281571

theorem election_winner_percentage 
  (total_votes : ℕ) 
  (winning_margin : ℕ) 
  (h1 : total_votes = 6900)
  (h2 : winning_margin = 1380) :
  (winning_margin : ℚ) / total_votes + 1/2 = 7/10 := by
  sorry

end election_winner_percentage_l2815_281571


namespace complex_magnitude_one_l2815_281579

theorem complex_magnitude_one (r : ℝ) (z : ℂ) (h1 : |r| < 4) (h2 : z + 1/z + 2 = r) : 
  Complex.abs z = 1 := by sorry

end complex_magnitude_one_l2815_281579


namespace apple_sellers_average_prices_l2815_281516

/-- Represents the sales data for a fruit seller --/
structure FruitSeller where
  morning_price : ℚ
  afternoon_price : ℚ
  morning_quantity : ℚ
  afternoon_quantity : ℚ

/-- Calculates the average price per apple for a fruit seller --/
def average_price (seller : FruitSeller) : ℚ :=
  (seller.morning_price * seller.morning_quantity + seller.afternoon_price * seller.afternoon_quantity) /
  (seller.morning_quantity + seller.afternoon_quantity)

theorem apple_sellers_average_prices
  (john bill george : FruitSeller)
  (h_morning_price : john.morning_price = bill.morning_price ∧ bill.morning_price = george.morning_price ∧ george.morning_price = 5/2)
  (h_afternoon_price : john.afternoon_price = bill.afternoon_price ∧ bill.afternoon_price = george.afternoon_price ∧ george.afternoon_price = 5/3)
  (h_john_quantities : john.morning_quantity = john.afternoon_quantity)
  (h_bill_revenue : bill.morning_price * bill.morning_quantity = bill.afternoon_price * bill.afternoon_quantity)
  (h_george_ratio : george.morning_quantity / george.afternoon_quantity = (5/3) / (5/2)) :
  average_price john = 25/12 ∧ average_price bill = 2 ∧ average_price george = 2 := by
  sorry


end apple_sellers_average_prices_l2815_281516


namespace exponent_multiplication_l2815_281561

theorem exponent_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end exponent_multiplication_l2815_281561


namespace computer_B_most_popular_l2815_281512

/-- Represents the sales data for a computer over three years -/
structure ComputerSales where
  year2018 : Nat
  year2019 : Nat
  year2020 : Nat

/-- Checks if the sales are consistently increasing -/
def isConsistentlyIncreasing (sales : ComputerSales) : Prop :=
  sales.year2018 < sales.year2019 ∧ sales.year2019 < sales.year2020

/-- Defines the sales data for computers A, B, and C -/
def computerA : ComputerSales := { year2018 := 600, year2019 := 610, year2020 := 590 }
def computerB : ComputerSales := { year2018 := 590, year2019 := 650, year2020 := 700 }
def computerC : ComputerSales := { year2018 := 650, year2019 := 670, year2020 := 660 }

/-- Theorem: Computer B is the most popular choice -/
theorem computer_B_most_popular :
  isConsistentlyIncreasing computerB ∧
  ¬isConsistentlyIncreasing computerA ∧
  ¬isConsistentlyIncreasing computerC :=
sorry

end computer_B_most_popular_l2815_281512


namespace drone_production_equations_correct_l2815_281539

/-- Represents the number of drones of type A and B produced by a company -/
structure DroneProduction where
  x : ℝ  -- number of type A drones
  y : ℝ  -- number of type B drones

/-- The system of equations representing the drone production conditions -/
def satisfiesConditions (p : DroneProduction) : Prop :=
  p.x = (1/2) * (p.x + p.y) + 11 ∧ p.y = (1/3) * (p.x + p.y) - 2

/-- Theorem stating that the system of equations correctly represents the given conditions -/
theorem drone_production_equations_correct (p : DroneProduction) :
  satisfiesConditions p ↔
    (p.x = (1/2) * (p.x + p.y) + 11 ∧   -- Type A drones condition
     p.y = (1/3) * (p.x + p.y) - 2) :=  -- Type B drones condition
by sorry

end drone_production_equations_correct_l2815_281539


namespace sum_of_squares_of_roots_l2815_281592

theorem sum_of_squares_of_roots (a b c : ℝ) : 
  (3 * a^3 - 2 * a^2 + 5 * a + 15 = 0) →
  (3 * b^3 - 2 * b^2 + 5 * b + 15 = 0) →
  (3 * c^3 - 2 * c^2 + 5 * c + 15 = 0) →
  a^2 + b^2 + c^2 = -26/9 := by
sorry

end sum_of_squares_of_roots_l2815_281592


namespace farmland_cleanup_theorem_l2815_281542

/-- Calculates the remaining area to be cleaned given the total area and cleaned areas -/
def remaining_area (total : Float) (lizzie : Float) (hilltown : Float) (green_valley : Float) : Float :=
  total - (lizzie + hilltown + green_valley)

/-- Theorem stating that the remaining area to be cleaned is 2442.38 square feet -/
theorem farmland_cleanup_theorem :
  remaining_area 9500.0 2534.1 2675.95 1847.57 = 2442.38 := by
  sorry

end farmland_cleanup_theorem_l2815_281542


namespace five_power_sum_of_squares_l2815_281532

theorem five_power_sum_of_squares (n : ℕ) : ∃ (a b : ℕ), 5^n = a^2 + b^2 := by
  sorry

end five_power_sum_of_squares_l2815_281532


namespace sufficient_not_necessary_l2815_281535

theorem sufficient_not_necessary : 
  (∀ x : ℝ, |x| < 2 → x^2 - x - 6 < 0) ∧ 
  (∃ x : ℝ, x^2 - x - 6 < 0 ∧ ¬(|x| < 2)) := by
  sorry

end sufficient_not_necessary_l2815_281535


namespace alfred_incurred_loss_no_gain_percent_l2815_281522

/-- Represents the financial transaction of buying and selling a scooter --/
structure ScooterTransaction where
  purchase_price : ℝ
  repair_cost : ℝ
  taxes_and_fees : ℝ
  accessories_cost : ℝ
  selling_price : ℝ

/-- Calculates the total cost of the scooter transaction --/
def total_cost (t : ScooterTransaction) : ℝ :=
  t.purchase_price + t.repair_cost + t.taxes_and_fees + t.accessories_cost

/-- Theorem stating that Alfred incurred a loss on the scooter transaction --/
theorem alfred_incurred_loss (t : ScooterTransaction) 
  (h1 : t.purchase_price = 4700)
  (h2 : t.repair_cost = 800)
  (h3 : t.taxes_and_fees = 300)
  (h4 : t.accessories_cost = 250)
  (h5 : t.selling_price = 6000) :
  total_cost t > t.selling_price := by
  sorry

/-- Corollary stating that there is no gain percent as Alfred incurred a loss --/
theorem no_gain_percent (t : ScooterTransaction) 
  (h1 : t.purchase_price = 4700)
  (h2 : t.repair_cost = 800)
  (h3 : t.taxes_and_fees = 300)
  (h4 : t.accessories_cost = 250)
  (h5 : t.selling_price = 6000) :
  ¬∃ (gain_percent : ℝ), gain_percent > 0 ∧ t.selling_price = total_cost t * (1 + gain_percent / 100) := by
  sorry

end alfred_incurred_loss_no_gain_percent_l2815_281522


namespace aunt_age_l2815_281562

/-- Proves that given Cori is 3 years old today, and in 5 years she will be one-third the age of her aunt, her aunt's current age is 19 years. -/
theorem aunt_age (cori_age : ℕ) (aunt_age : ℕ) : 
  cori_age = 3 → 
  (cori_age + 5 : ℕ) = (aunt_age + 5) / 3 → 
  aunt_age = 19 :=
by
  sorry

end aunt_age_l2815_281562


namespace lucy_money_ratio_l2815_281520

/-- Proves that the ratio of money lost to initial amount is 1:3 given the conditions of Lucy's spending --/
theorem lucy_money_ratio (initial_amount : ℝ) (lost_amount : ℝ) (remainder : ℝ) (final_amount : ℝ) :
  initial_amount = 30 →
  remainder = initial_amount - lost_amount →
  final_amount = remainder - (1/4) * remainder →
  final_amount = 15 →
  lost_amount / initial_amount = 1/3 := by
sorry

end lucy_money_ratio_l2815_281520


namespace divisibility_problem_l2815_281568

theorem divisibility_problem (a b : Nat) (n : Nat) : 
  a ≤ 9 → b ≤ 9 → a * b ≤ 15 → (110 * a + b) % n = 0 → n = 5 := by
  sorry

end divisibility_problem_l2815_281568


namespace faster_train_speed_l2815_281533

/-- Proves that given two trains of specified lengths running in opposite directions,
    with a given crossing time and speed of the slower train, the speed of the faster train
    is as calculated. -/
theorem faster_train_speed
  (length_train1 : ℝ)
  (length_train2 : ℝ)
  (crossing_time : ℝ)
  (slower_train_speed : ℝ)
  (h1 : length_train1 = 180)
  (h2 : length_train2 = 360)
  (h3 : crossing_time = 21.598272138228943)
  (h4 : slower_train_speed = 30) :
  ∃ (faster_train_speed : ℝ),
    faster_train_speed = 60 ∧
    (length_train1 + length_train2) / crossing_time * 3.6 = slower_train_speed + faster_train_speed :=
by sorry

end faster_train_speed_l2815_281533


namespace cube_root_of_sum_l2815_281531

theorem cube_root_of_sum (a b : ℝ) : 
  (2*a + 1) + (2*a - 5) = 0 → 
  b^(1/3 : ℝ) = 2 → 
  (a + b)^(1/3 : ℝ) = 9^(1/3 : ℝ) := by
sorry

end cube_root_of_sum_l2815_281531


namespace tetrahedron_inequality_l2815_281574

theorem tetrahedron_inequality 
  (h₁ h₂ h₃ h₄ x₁ x₂ x₃ x₄ : ℝ) 
  (h_nonneg : h₁ ≥ 0 ∧ h₂ ≥ 0 ∧ h₃ ≥ 0 ∧ h₄ ≥ 0)
  (x_nonneg : x₁ ≥ 0 ∧ x₂ ≥ 0 ∧ x₃ ≥ 0 ∧ x₄ ≥ 0)
  (h_tetrahedron : ∃ (S₁ S₂ S₃ S₄ : ℝ), 
    S₁ > 0 ∧ S₂ > 0 ∧ S₃ > 0 ∧ S₄ > 0 ∧
    S₁ * h₁ = S₁ * x₁ ∧ 
    S₂ * h₂ = S₂ * x₂ ∧ 
    S₃ * h₃ = S₃ * x₃ ∧ 
    S₄ * h₄ = S₄ * x₄) :
  Real.sqrt (h₁ + h₂ + h₃ + h₄) ≥ Real.sqrt x₁ + Real.sqrt x₂ + Real.sqrt x₃ + Real.sqrt x₄ := by
  sorry

end tetrahedron_inequality_l2815_281574


namespace system_solution_l2815_281584

theorem system_solution (a b c x y z : ℝ) 
  (eq1 : x - a * y + a^2 * z = a^3)
  (eq2 : x - b * y + b^2 * z = b^3)
  (eq3 : x - c * y + c^2 * z = c^3)
  (hx : x = a * b * c)
  (hy : y = a * b + a * c + b * c)
  (hz : z = a + b + c)
  (ha : a ≠ b)
  (hb : a ≠ c)
  (hc : b ≠ c) :
  x - a * y + a^2 * z = a^3 ∧
  x - b * y + b^2 * z = b^3 ∧
  x - c * y + c^2 * z = c^3 :=
by sorry

end system_solution_l2815_281584


namespace f_satisfies_properties_l2815_281580

def f (x : ℝ) : ℝ := (x - 2)^2

theorem f_satisfies_properties :
  (∀ x, f (x + 2) = f (-x + 2)) ∧
  (∀ x y, x < y → x < 2 → y < 2 → f x > f y) ∧
  (∀ x y, x < y → x > 2 → y > 2 → f x < f y) := by
sorry

end f_satisfies_properties_l2815_281580


namespace equation_one_solutions_l2815_281552

theorem equation_one_solutions (x : ℝ) : x^2 - 9 = 0 ↔ x = 3 ∨ x = -3 := by
  sorry

end equation_one_solutions_l2815_281552


namespace unique_solution_l2815_281573

/-- Represents a pair of digits in base r -/
structure DigitPair (r : ℕ) where
  first : ℕ
  second : ℕ
  h_first : first < r
  h_second : second < r

/-- Constructs a number from repeating a digit pair n times in base r -/
def construct_number (r : ℕ) (pair : DigitPair r) (n : ℕ) : ℕ :=
  pair.first * r + pair.second

/-- Checks if a number consists of only ones in base r -/
def all_ones (r : ℕ) (x : ℕ) : Prop :=
  ∀ k, (x / r^k) % r = 1 ∨ (x / r^k) = 0

theorem unique_solution :
  ∀ (r : ℕ) (x : ℕ) (n : ℕ) (pair : DigitPair r),
    2 ≤ r →
    r ≤ 70 →
    x = construct_number r pair n →
    all_ones r (x^2) →
    (r = 7 ∧ x = 26) := by sorry

end unique_solution_l2815_281573


namespace absolute_value_inequality_l2815_281597

theorem absolute_value_inequality (x : ℝ) :
  abs (x - 2) + abs (x + 3) < 8 ↔ x ∈ Set.Ioo (-6.5) 3.5 :=
by sorry

end absolute_value_inequality_l2815_281597


namespace farmer_water_capacity_l2815_281575

/-- Calculates the total water capacity of a single truck -/
def truckCapacity (tankCapacities : List ℕ) : ℕ :=
  tankCapacities.sum

/-- Calculates the amount of water in a truck given its capacity and fill percentage -/
def waterInTruck (capacity : ℕ) (fillPercentage : ℕ) : ℕ :=
  capacity * fillPercentage / 100

/-- Represents the problem of calculating total water capacity across multiple trucks -/
def waterCapacityProblem (tankCapacities : List ℕ) (fillPercentages : List ℕ) : Prop :=
  let capacity := truckCapacity tankCapacities
  let waterAmounts := fillPercentages.map (waterInTruck capacity)
  waterAmounts.sum = 2750

/-- The main theorem stating the solution to the water capacity problem -/
theorem farmer_water_capacity :
  waterCapacityProblem [200, 250, 300, 350] [100, 75, 50, 25, 0] := by
  sorry

#check farmer_water_capacity

end farmer_water_capacity_l2815_281575


namespace no_parallel_m_l2815_281509

/-- Two lines are parallel if and only if their slopes are equal -/
def parallel_lines (m : ℝ) : Prop :=
  (2 : ℝ) / (m + 1) = m / 3

/-- There is no real number m that makes the lines parallel -/
theorem no_parallel_m : ¬ ∃ m : ℝ, parallel_lines m := by
  sorry

end no_parallel_m_l2815_281509


namespace p_on_x_axis_equal_distance_to_axes_l2815_281523

-- Define the point P as a function of m
def P (m : ℝ) : ℝ × ℝ := (8 - 2*m, m - 1)

-- Part 1: P lies on the x-axis implies m = 1
theorem p_on_x_axis (m : ℝ) : (P m).2 = 0 → m = 1 := by sorry

-- Part 2: Equal distance to both axes implies P(2,2) or P(-6,6)
theorem equal_distance_to_axes (m : ℝ) : 
  |8 - 2*m| = |m - 1| → (P m = (2, 2) ∨ P m = (-6, 6)) := by sorry

end p_on_x_axis_equal_distance_to_axes_l2815_281523


namespace isosceles_triangle_base_length_l2815_281514

/-- An isosceles triangle with two sides of length 7 cm and perimeter 23 cm has a base of length 9 cm. -/
theorem isosceles_triangle_base_length :
  ∀ (base : ℝ),
  base > 0 →
  7 + 7 + base = 23 →
  base = 9 :=
by
  sorry

end isosceles_triangle_base_length_l2815_281514


namespace x_coordinate_of_Q_l2815_281596

/-- A line through the origin equidistant from two points -/
structure EquidistantLine where
  slope : ℝ
  P : ℝ × ℝ
  Q : ℝ × ℝ
  is_equidistant : ∀ (x y : ℝ), y = slope * x → 
    (x - P.1)^2 + (y - P.2)^2 = (x - Q.1)^2 + (y - Q.2)^2

/-- Theorem: Given the conditions, the x-coordinate of Q is 2.5 -/
theorem x_coordinate_of_Q (L : EquidistantLine) 
  (h_slope : L.slope = 0.8)
  (h_Q_y : L.Q.2 = 2) :
  L.Q.1 = 2.5 := by
  sorry

end x_coordinate_of_Q_l2815_281596


namespace infinite_segment_sum_l2815_281582

/-- Given a triangle ABC with sides a, b, c where b > c, and an infinite sequence
    of line segments constructed as follows:
    - BB1 is antiparallel to BC, intersecting AC at B1
    - B1C1 is parallel to BC, intersecting AB at C1
    - This process continues infinitely
    Then the sum of the lengths of these segments (BC + BB1 + B1C1 + ...) is ab / (b - c) -/
theorem infinite_segment_sum (a b c : ℝ) (h : b > c) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) :
  ∃ (sequence : ℕ → ℝ),
    (sequence 0 = a) ∧
    (∀ n, sequence (n + 1) = sequence n * (c / b)) ∧
    (∑' n, sequence n) = a * b / (b - c) := by
  sorry

end infinite_segment_sum_l2815_281582


namespace white_marbles_count_l2815_281546

theorem white_marbles_count (total : ℕ) (blue : ℕ) (red : ℕ) (prob_red_or_white : ℚ) :
  total = 50 →
  blue = 5 →
  red = 9 →
  prob_red_or_white = 9/10 →
  (total - blue - red : ℚ) / total = prob_red_or_white - (red : ℚ) / total →
  total - blue - red = 36 :=
by
  sorry

#check white_marbles_count

end white_marbles_count_l2815_281546


namespace last_three_digits_of_7_to_50_l2815_281528

theorem last_three_digits_of_7_to_50 : 7^50 % 1000 = 991 := by sorry

end last_three_digits_of_7_to_50_l2815_281528


namespace extreme_value_implies_a_equals_5_l2815_281500

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + 3*x + 5

-- Define the derivative of f(x)
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + 3

-- Theorem statement
theorem extreme_value_implies_a_equals_5 (a : ℝ) :
  (∃ (ε : ℝ), ∀ (x : ℝ), |x + 3| < ε → f a (-3) ≥ f a x) →
  f' a (-3) = 0 →
  a = 5 :=
sorry

end extreme_value_implies_a_equals_5_l2815_281500


namespace complex_magnitude_of_i_times_one_minus_i_l2815_281583

theorem complex_magnitude_of_i_times_one_minus_i : 
  let z : ℂ := Complex.I * (1 - Complex.I)
  Complex.abs z = Real.sqrt 2 := by sorry

end complex_magnitude_of_i_times_one_minus_i_l2815_281583


namespace pencil_pen_cost_l2815_281530

/-- Given the cost of pencils and pens, calculate the cost of a different combination -/
theorem pencil_pen_cost (p q : ℝ) 
  (h1 : 3 * p + 2 * q = 4.30)
  (h2 : 2 * p + 3 * q = 4.05) :
  4 * p + 3 * q = 5.97 := by
  sorry

end pencil_pen_cost_l2815_281530


namespace concurrent_or_parallel_iff_concyclic_l2815_281570

/-- A point in the Euclidean plane -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A line in the Euclidean plane -/
structure Line :=
  (a : ℝ) (b : ℝ) (c : ℝ)

/-- Definition of a triangle -/
structure Triangle :=
  (A : Point) (B : Point) (C : Point)

/-- Definition of a circumcenter -/
def circumcenter (t : Triangle) : Point :=
  sorry

/-- Definition of concurrency for three lines -/
def are_concurrent (l1 l2 l3 : Line) : Prop :=
  sorry

/-- Definition of parallel lines -/
def are_parallel (l1 l2 : Line) : Prop :=
  sorry

/-- Definition of pairwise parallel lines -/
def are_pairwise_parallel (l1 l2 l3 : Line) : Prop :=
  sorry

/-- Definition of concyclic points -/
def are_concyclic (A B C D : Point) : Prop :=
  sorry

/-- The main theorem -/
theorem concurrent_or_parallel_iff_concyclic 
  (A B C D E F : Point) 
  (G : Point := circumcenter ⟨B, C, E⟩) 
  (H : Point := circumcenter ⟨A, D, F⟩) 
  (AB CD GH : Line) :
  (are_concurrent AB CD GH ∨ are_pairwise_parallel AB CD GH) ↔ 
  are_concyclic A B E F :=
sorry

end concurrent_or_parallel_iff_concyclic_l2815_281570


namespace right_triangle_hypotenuse_l2815_281543

theorem right_triangle_hypotenuse : 
  ∀ (a b c : ℝ),
  a > 0 ∧ b > 0 ∧ c > 0 →
  a^2 + b^2 = c^2 →
  b = 6 →
  c = a + 2 →
  c = 10 :=
by
  sorry

end right_triangle_hypotenuse_l2815_281543


namespace F_is_second_from_left_l2815_281502

-- Define a structure for rectangles
structure Rectangle where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ

-- Define the four rectangles
def F : Rectangle := ⟨7, 2, 5, 9⟩
def G : Rectangle := ⟨6, 9, 1, 3⟩
def H : Rectangle := ⟨2, 5, 7, 10⟩
def J : Rectangle := ⟨3, 1, 6, 8⟩

-- Define a function to check if two rectangles can connect
def canConnect (r1 r2 : Rectangle) : Prop :=
  (r1.a = r2.a) ∨ (r1.a = r2.b) ∨ (r1.a = r2.c) ∨ (r1.a = r2.d) ∨
  (r1.b = r2.a) ∨ (r1.b = r2.b) ∨ (r1.b = r2.c) ∨ (r1.b = r2.d) ∨
  (r1.c = r2.a) ∨ (r1.c = r2.b) ∨ (r1.c = r2.c) ∨ (r1.c = r2.d) ∨
  (r1.d = r2.a) ∨ (r1.d = r2.b) ∨ (r1.d = r2.c) ∨ (r1.d = r2.d)

-- Theorem stating that F is second from the left
theorem F_is_second_from_left :
  ∃ (left right : Rectangle), left ≠ F ∧ right ≠ F ∧
  canConnect left F ∧ canConnect F right ∧
  (∀ r : Rectangle, r ≠ F → r ≠ left → r ≠ right → ¬(canConnect left r ∧ canConnect r right)) :=
by
  sorry

end F_is_second_from_left_l2815_281502


namespace multiplication_associative_l2815_281503

theorem multiplication_associative (x y z : ℝ) : (x * y) * z = x * (y * z) := by
  sorry

end multiplication_associative_l2815_281503


namespace three_number_problem_l2815_281564

theorem three_number_problem (a b c : ℚ) 
  (sum_eq : a + b + c = 30)
  (first_eq : a = 6 * (b + c))
  (second_eq : b = 9 * c) :
  a - c = 177 / 7 := by
sorry

end three_number_problem_l2815_281564


namespace right_triangle_side_length_l2815_281504

theorem right_triangle_side_length 
  (P Q R : ℝ × ℝ) -- Points in 2D plane
  (is_right_triangle : (Q.1 - R.1) * (P.1 - R.1) + (Q.2 - R.2) * (P.2 - R.2) = 0) -- Right angle condition
  (cos_R : ((Q.1 - R.1) * (P.1 - R.1) + (Q.2 - R.2) * (P.2 - R.2)) / 
           (Real.sqrt ((P.1 - R.1)^2 + (P.2 - R.2)^2) * Real.sqrt ((Q.1 - R.1)^2 + (Q.2 - R.2)^2)) = 3/5) -- cos R = 3/5
  (RP_length : Real.sqrt ((P.1 - R.1)^2 + (P.2 - R.2)^2) = 10) -- RP = 10
  : Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = 8 := by -- PQ = 8
sorry

end right_triangle_side_length_l2815_281504


namespace not_divisible_by_nine_l2815_281515

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem not_divisible_by_nine : ¬(∃ k : ℕ, 48767621 = 9 * k) :=
  by
  have h1 : ∀ n : ℕ, (∃ k : ℕ, n = 9 * k) ↔ (∃ m : ℕ, sum_of_digits n = 9 * m) := by sorry
  have h2 : sum_of_digits 48767621 = 41 := by sorry
  have h3 : ¬(∃ m : ℕ, 41 = 9 * m) := by sorry
  sorry

end not_divisible_by_nine_l2815_281515


namespace symmetric_parabola_l2815_281517

/-- 
Given a parabola with equation y^2 = 2x and a point (-1, 0),
prove that the equation y^2 = -2(x + 2) represents the parabola 
symmetric to the original parabola with respect to the given point.
-/
theorem symmetric_parabola (x y : ℝ) : 
  (∀ x y, y^2 = 2*x → 
   ∃ x' y', x' = -x - 2 ∧ y' = -y ∧ y'^2 = -2*(x' + 2)) := by
  sorry

end symmetric_parabola_l2815_281517


namespace abs_equation_solution_l2815_281507

theorem abs_equation_solution :
  ∃! x : ℝ, |x - 3| = 5 - x :=
by
  sorry

end abs_equation_solution_l2815_281507


namespace matrix_property_l2815_281593

/-- A 4x4 complex matrix with the given structure -/
def M (a b c d : ℂ) : Matrix (Fin 4) (Fin 4) ℂ :=
  ![![a, b, c, d],
    ![b, c, d, a],
    ![c, d, a, b],
    ![d, a, b, c]]

theorem matrix_property (a b c d : ℂ) :
  M a b c d ^ 2 = 1 → a * b * c * d = 1 → a^4 + b^4 + c^4 + d^4 = 1 := by
  sorry

end matrix_property_l2815_281593


namespace modulus_of_complex_number_l2815_281590

theorem modulus_of_complex_number (z : ℂ) (h : z = 3 - 2*Complex.I) : Complex.abs z = Real.sqrt 13 := by
  sorry

end modulus_of_complex_number_l2815_281590


namespace product_2000_sum_bounds_l2815_281587

theorem product_2000_sum_bounds (a b c d e : ℕ) 
  (ha : a > 1) (hb : b > 1) (hc : c > 1) (hd : d > 1) (he : e > 1)
  (h_product : a * b * c * d * e = 2000) :
  (∃ (x y z w v : ℕ), x > 1 ∧ y > 1 ∧ z > 1 ∧ w > 1 ∧ v > 1 ∧ 
    x * y * z * w * v = 2000 ∧ x + y + z + w + v = 133) ∧
  (∃ (x y z w v : ℕ), x > 1 ∧ y > 1 ∧ z > 1 ∧ w > 1 ∧ v > 1 ∧ 
    x * y * z * w * v = 2000 ∧ x + y + z + w + v = 23) ∧
  (∀ (x y z w v : ℕ), x > 1 → y > 1 → z > 1 → w > 1 → v > 1 → 
    x * y * z * w * v = 2000 → 23 ≤ x + y + z + w + v ∧ x + y + z + w + v ≤ 133) :=
by sorry

end product_2000_sum_bounds_l2815_281587


namespace range_of_a_l2815_281595

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 - 4*a*x + 3*a^2 ≤ 0 → x^2 + 5*x + 4 < 0) →
  a < 0 →
  -4/3 ≤ a ∧ a ≤ -1 :=
by sorry

end range_of_a_l2815_281595


namespace min_moves_for_n_triangles_l2815_281594

/-- Represents a robot on a vertex of a polygon -/
structure Robot where
  vertex : ℕ
  target : ℕ

/-- Represents the state of the polygon -/
structure PolygonState where
  n : ℕ
  robots : List Robot

/-- A move rotates a robot to point at a new target -/
def move (state : PolygonState) (robot_index : ℕ) : PolygonState :=
  sorry

/-- Checks if three robots form a triangle -/
def is_triangle (r1 r2 r3 : Robot) : Bool :=
  sorry

/-- Counts the number of triangles in the current state -/
def count_triangles (state : PolygonState) : ℕ :=
  sorry

/-- The main theorem stating the minimum number of moves required -/
theorem min_moves_for_n_triangles (n : ℕ) :
  ∃ (initial_state : PolygonState),
    initial_state.n = n ∧
    initial_state.robots.length = 3 * n ∧
    ∀ (final_state : PolygonState),
      (count_triangles final_state = n) →
      (∃ (move_sequence : List ℕ),
        final_state = (move_sequence.foldl move initial_state) ∧
        move_sequence.length ≥ (9 * n^2 - 7 * n) / 2) :=
sorry

end min_moves_for_n_triangles_l2815_281594


namespace length_of_PQ_l2815_281558

-- Define the points
variable (P Q R S : ℝ × ℝ)

-- Define the distances between points
def distance (A B : ℝ × ℝ) : ℝ := sorry

-- Define isosceles triangle
def isIsosceles (A B C : ℝ × ℝ) : Prop :=
  distance A B = distance A C

-- Define perimeter of a triangle
def perimeter (A B C : ℝ × ℝ) : ℝ :=
  distance A B + distance B C + distance C A

-- State the theorem
theorem length_of_PQ (P Q R S : ℝ × ℝ) :
  isIsosceles P Q R →
  isIsosceles Q R S →
  perimeter Q R S = 24 →
  perimeter P Q R = 23 →
  distance Q R = 10 →
  distance P Q = 6.5 := by sorry

end length_of_PQ_l2815_281558


namespace rabbit_log_cutting_l2815_281549

theorem rabbit_log_cutting (cuts pieces : ℕ) (h1 : cuts = 10) (h2 : pieces = 16) :
  ∃ logs : ℕ, logs + cuts = pieces ∧ logs = 6 := by
  sorry

end rabbit_log_cutting_l2815_281549
