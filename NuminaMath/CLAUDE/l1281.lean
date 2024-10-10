import Mathlib

namespace solution_set_inequality_proof_l1281_128130

-- Define the function f(x) = |x-2|
def f (x : ℝ) : ℝ := |x - 2|

-- Part 1: Prove that the solution set of f(x) + f(x+1) ≤ 2 is [0.5, 2.5]
theorem solution_set (x : ℝ) : 
  (f x + f (x + 1) ≤ 2) ↔ (0.5 ≤ x ∧ x ≤ 2.5) := by sorry

-- Part 2: Prove that for all a < 0 and all x, f(ax) - af(x) ≥ f(2a)
theorem inequality_proof (a x : ℝ) (h : a < 0) : 
  f (a * x) - a * f x ≥ f (2 * a) := by sorry

end solution_set_inequality_proof_l1281_128130


namespace magnitude_of_one_plus_two_i_to_eighth_l1281_128190

theorem magnitude_of_one_plus_two_i_to_eighth : Complex.abs ((1 + 2*Complex.I)^8) = 625 := by
  sorry

end magnitude_of_one_plus_two_i_to_eighth_l1281_128190


namespace veronica_yellow_balls_l1281_128100

theorem veronica_yellow_balls :
  let total_balls : ℕ := 60
  let yellow_balls : ℕ := 27
  let brown_balls : ℕ := 33
  (yellow_balls : ℚ) / total_balls = 45 / 100 ∧
  brown_balls + yellow_balls = total_balls →
  yellow_balls = 27 :=
by sorry

end veronica_yellow_balls_l1281_128100


namespace sara_received_four_onions_l1281_128197

/-- The number of onions given to Sara -/
def onions_given_to_sara (sally_onions fred_onions remaining_onions : ℕ) : ℕ :=
  sally_onions + fred_onions - remaining_onions

/-- Theorem stating that Sara received 4 onions -/
theorem sara_received_four_onions :
  onions_given_to_sara 5 9 10 = 4 := by
  sorry

end sara_received_four_onions_l1281_128197


namespace triangle_perimeter_sum_specific_triangle_perimeter_sum_l1281_128192

/-- The sum of perimeters of an infinite series of equilateral triangles -/
theorem triangle_perimeter_sum (initial_perimeter : ℝ) :
  initial_perimeter > 0 →
  (∑' n, initial_perimeter * (1/2)^n) = 2 * initial_perimeter :=
by sorry

/-- The specific case where the initial triangle has a perimeter of 90 cm -/
theorem specific_triangle_perimeter_sum :
  (∑' n, 90 * (1/2)^n) = 180 :=
by sorry

end triangle_perimeter_sum_specific_triangle_perimeter_sum_l1281_128192


namespace problem_statement_l1281_128124

theorem problem_statement (a b : ℝ) (h : Real.exp a + Real.exp b = 4) :
  a + b ≤ 2 * Real.log 2 ∧ Real.exp a + b ≤ 3 ∧ Real.exp (2 * a) + Real.exp (2 * b) ≥ 8 := by
  sorry

end problem_statement_l1281_128124


namespace opposite_of_three_minus_one_l1281_128180

theorem opposite_of_three_minus_one :
  -(3 - 1) = -2 := by
  sorry

end opposite_of_three_minus_one_l1281_128180


namespace incorrect_arrangements_count_l1281_128147

/-- The number of unique arrangements of the letters "e", "o", "h", "l", "l" -/
def total_arrangements : ℕ := 60

/-- The number of correct arrangements (spelling "hello") -/
def correct_arrangements : ℕ := 1

/-- Theorem stating the number of incorrect arrangements -/
theorem incorrect_arrangements_count :
  total_arrangements - correct_arrangements = 59 := by
  sorry

end incorrect_arrangements_count_l1281_128147


namespace parallelogram_area_l1281_128123

-- Define the lines
def L1 (x y : ℝ) : Prop := y = 2
def L2 (x y : ℝ) : Prop := y = -2
def L3 (x y : ℝ) : Prop := 4 * x + 7 * y - 10 = 0
def L4 (x y : ℝ) : Prop := 4 * x + 7 * y + 20 = 0

-- Define the vertices of the parallelogram
def A : ℝ × ℝ := (-1.5, -2)
def B : ℝ × ℝ := (6, -2)
def C : ℝ × ℝ := (-1, 2)
def D : ℝ × ℝ := (-8.5, 2)

-- State the theorem
theorem parallelogram_area : 
  (A.1 = -1.5 ∧ A.2 = -2) →
  (B.1 = 6 ∧ B.2 = -2) →
  (C.1 = -1 ∧ C.2 = 2) →
  (D.1 = -8.5 ∧ D.2 = 2) →
  L1 C.1 C.2 →
  L1 D.1 D.2 →
  L2 A.1 A.2 →
  L2 B.1 B.2 →
  L3 A.1 A.2 →
  L3 C.1 C.2 →
  L4 B.1 B.2 →
  L4 D.1 D.2 →
  (B.1 - A.1) * (C.2 - A.2) = 30 :=
by sorry

end parallelogram_area_l1281_128123


namespace water_per_day_per_man_l1281_128158

/-- Calculates the amount of water needed per day per man on a sea voyage --/
theorem water_per_day_per_man 
  (total_men : ℕ) 
  (miles_per_day : ℕ) 
  (total_miles : ℕ) 
  (total_water : ℕ) : 
  total_men = 25 → 
  miles_per_day = 200 → 
  total_miles = 4000 → 
  total_water = 250 → 
  (total_water : ℚ) / ((total_miles : ℚ) / (miles_per_day : ℚ)) / (total_men : ℚ) = 1/2 := by
  sorry

end water_per_day_per_man_l1281_128158


namespace divisibility_of_linear_combination_l1281_128121

theorem divisibility_of_linear_combination (a b c : ℕ+) : 
  ∃ (r s : ℕ+), (Nat.gcd r s = 1) ∧ (∃ k : ℤ, (a : ℤ) * (r : ℤ) + (b : ℤ) * (s : ℤ) = k * (c : ℤ)) := by
  sorry

end divisibility_of_linear_combination_l1281_128121


namespace donut_distribution_theorem_l1281_128126

/-- A structure representing the donut distribution problem -/
structure DonutDistribution where
  num_boxes : ℕ
  total_donuts : ℕ
  num_flavors : ℕ

/-- The result of the donut distribution -/
structure DistributionResult where
  extra_donuts : ℕ
  donuts_per_flavor_per_box : ℕ

/-- Function to calculate the distribution result -/
def calculate_distribution (d : DonutDistribution) : DistributionResult :=
  { extra_donuts := d.total_donuts % d.num_boxes,
    donuts_per_flavor_per_box := (d.total_donuts / d.num_flavors) / d.num_boxes }

/-- Theorem stating the correct distribution for the given problem -/
theorem donut_distribution_theorem (d : DonutDistribution) 
  (h1 : d.num_boxes = 12)
  (h2 : d.total_donuts = 125)
  (h3 : d.num_flavors = 5) :
  let result := calculate_distribution d
  result.extra_donuts = 5 ∧ result.donuts_per_flavor_per_box = 2 := by
  sorry

#check donut_distribution_theorem

end donut_distribution_theorem_l1281_128126


namespace negation_of_universal_proposition_l1281_128145

theorem negation_of_universal_proposition :
  ¬(∀ x : ℝ, x > 1 → x^2 > 1) ↔ (∃ x : ℝ, x > 1 ∧ x^2 ≤ 1) :=
by sorry

end negation_of_universal_proposition_l1281_128145


namespace average_age_of_eight_students_l1281_128194

theorem average_age_of_eight_students 
  (total_students : Nat)
  (average_age_all : ℝ)
  (num_group1 : Nat)
  (num_group2 : Nat)
  (average_age_group2 : ℝ)
  (age_last_student : ℝ)
  (h1 : total_students = 15)
  (h2 : average_age_all = 15)
  (h3 : num_group1 = 8)
  (h4 : num_group2 = 6)
  (h5 : average_age_group2 = 16)
  (h6 : age_last_student = 17)
  (h7 : total_students = num_group1 + num_group2 + 1) :
  (total_students : ℝ) * average_age_all - 
  (num_group2 : ℝ) * average_age_group2 - 
  age_last_student = (num_group1 : ℝ) * 14 := by
    sorry

#check average_age_of_eight_students

end average_age_of_eight_students_l1281_128194


namespace shuai_shuai_memorization_l1281_128133

/-- The number of words memorized by Shuai Shuai over 7 days -/
def total_words : ℕ := 198

/-- The number of words memorized in the first 3 days -/
def first_three_days : ℕ := 44

/-- The number of words memorized on the fourth day -/
def fourth_day : ℕ := 10

/-- The number of words memorized in the last 3 days -/
def last_three_days : ℕ := 45

/-- Theorem stating the conditions and the result -/
theorem shuai_shuai_memorization :
  (first_three_days + fourth_day + last_three_days = total_words) ∧
  (first_three_days = (4 : ℚ) / 5 * (fourth_day + last_three_days)) ∧
  (first_three_days + fourth_day = (6 : ℚ) / 5 * last_three_days) ∧
  (total_words > 100) ∧
  (total_words < 200) :=
by sorry

end shuai_shuai_memorization_l1281_128133


namespace right_triangle_area_l1281_128141

theorem right_triangle_area (a b c : ℝ) (h1 : a^2 + b^2 = c^2) (h2 : c = 13) (h3 : a = 5) : 
  (1/2) * a * b = 30 := by
sorry

end right_triangle_area_l1281_128141


namespace arrangement_theorems_l1281_128156

/-- The number of boys in the group -/
def num_boys : ℕ := 3

/-- The number of girls in the group -/
def num_girls : ℕ := 4

/-- The total number of people in the group -/
def total_people : ℕ := num_boys + num_girls

/-- The number of arrangements with boys together -/
def arrangements_boys_together : ℕ := 720

/-- The number of arrangements with alternating genders -/
def arrangements_alternating : ℕ := 144

/-- The number of arrangements with person A left of person B -/
def arrangements_A_left_of_B : ℕ := 2520

theorem arrangement_theorems :
  (arrangements_boys_together = 720) ∧
  (arrangements_alternating = 144) ∧
  (arrangements_A_left_of_B = 2520) := by sorry

end arrangement_theorems_l1281_128156


namespace sufficient_not_necessary_l1281_128199

theorem sufficient_not_necessary : 
  (∀ x : ℝ, |x| < 2 → x^2 - x - 6 < 0) ∧ 
  (∃ x : ℝ, x^2 - x - 6 < 0 ∧ |x| ≥ 2) := by
  sorry

end sufficient_not_necessary_l1281_128199


namespace red_candy_count_l1281_128135

theorem red_candy_count (total : ℕ) (blue : ℕ) (h1 : total = 3409) (h2 : blue = 3264) :
  total - blue = 145 := by
  sorry

end red_candy_count_l1281_128135


namespace pascal_triangle_12th_row_4th_number_l1281_128195

theorem pascal_triangle_12th_row_4th_number : Nat.choose 12 3 = 220 := by
  sorry

end pascal_triangle_12th_row_4th_number_l1281_128195


namespace exists_non_square_product_l1281_128113

theorem exists_non_square_product (a b : ℤ) (ha : a > 1) (hb : b > 1) (hab : a ≠ b) :
  ∃ n : ℕ+, ¬∃ m : ℤ, (a^n.val - 1) * (b^n.val - 1) = m^2 := by
  sorry

end exists_non_square_product_l1281_128113


namespace circle_radius_l1281_128117

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define the axis of symmetry
def axis_of_symmetry (x : ℝ) : Prop := x = -1

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := x + Real.sqrt 3 * y - 3 = 0

-- Define the circle C
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- State the theorem
theorem circle_radius :
  ∀ C : Circle,
  (C.center.1 = -1) →  -- Center is on the axis of symmetry
  (C.center.2 ≠ 0) →  -- Center is not on the x-axis
  (C.center.1 + C.radius)^2 + C.center.2^2 = C.radius^2 →  -- Circle passes through the focus
  (∃ (x y : ℝ), tangent_line x y ∧ 
    ((x - C.center.1)^2 + (y - C.center.2)^2 = C.radius^2)) →  -- Circle is tangent to the line
  C.radius = 14 :=
sorry

end circle_radius_l1281_128117


namespace jordan_oreos_l1281_128132

theorem jordan_oreos (jordan : ℕ) (james : ℕ) : 
  james = 2 * jordan + 3 →
  jordan + james = 36 →
  jordan = 11 := by
sorry

end jordan_oreos_l1281_128132


namespace jeff_cabinets_l1281_128134

/-- The total number of cabinets after Jeff's installation --/
def total_cabinets (initial : ℕ) (counters : ℕ) (extra : ℕ) : ℕ :=
  initial + counters * (2 * initial) + extra

/-- Proof that Jeff has 26 cabinets after the installation --/
theorem jeff_cabinets : total_cabinets 3 3 5 = 26 := by
  sorry

end jeff_cabinets_l1281_128134


namespace exist_decreasing_gcd_sequence_l1281_128136

theorem exist_decreasing_gcd_sequence :
  ∃ (a : Fin 100 → ℕ), 
    (∀ i j : Fin 100, i < j → a i < a j) ∧ 
    (∀ i : Fin 99, Nat.gcd (a i) (a (i + 1)) > Nat.gcd (a (i + 1)) (a (i + 2))) :=
by sorry

end exist_decreasing_gcd_sequence_l1281_128136


namespace average_equals_expression_l1281_128105

theorem average_equals_expression (x : ℝ) : 
  (1/3) * ((3*x + 8) + (7*x + 3) + (4*x + 9)) = 5*x - 10 → x = 50 := by
  sorry

end average_equals_expression_l1281_128105


namespace inequality_proof_equality_conditions_l1281_128106

theorem inequality_proof (x y : ℝ) (h1 : x ≥ y) (h2 : y ≥ 1) :
  (x / Real.sqrt (x + y)) + (y / Real.sqrt (y + 1)) + (1 / Real.sqrt (x + 1)) ≥
  (y / Real.sqrt (x + y)) + (x / Real.sqrt (x + 1)) + (1 / Real.sqrt (y + 1)) :=
sorry

theorem equality_conditions (x y : ℝ) (h1 : x ≥ y) (h2 : y ≥ 1) :
  (x / Real.sqrt (x + y)) + (y / Real.sqrt (y + 1)) + (1 / Real.sqrt (x + 1)) =
  (y / Real.sqrt (x + y)) + (x / Real.sqrt (x + 1)) + (1 / Real.sqrt (y + 1)) ↔
  (x = y ∨ x = 1 ∨ y = 1) :=
sorry

end inequality_proof_equality_conditions_l1281_128106


namespace order_relation_l1281_128149

theorem order_relation (a b c : ℝ) : 
  a = 1 / 2023 ∧ 
  b = Real.tan (Real.exp (1 / 2023) / 2023) ∧ 
  c = Real.sin (Real.exp (1 / 2024) / 2024) →
  c < a ∧ a < b :=
by sorry

end order_relation_l1281_128149


namespace classroom_ratio_l1281_128167

theorem classroom_ratio :
  ∀ (x y : ℕ),
    x + y = 15 →
    30 * x + 25 * y = 400 →
    x / 15 = 1 / 3 :=
by
  sorry

end classroom_ratio_l1281_128167


namespace marble_selection_probability_l1281_128104

def total_marbles : ℕ := 3 + 2 + 2
def red_marbles : ℕ := 3
def blue_marbles : ℕ := 2
def green_marbles : ℕ := 2
def selected_marbles : ℕ := 4

theorem marble_selection_probability :
  (Nat.choose red_marbles 2 * Nat.choose blue_marbles 1 * Nat.choose green_marbles 1) /
  (Nat.choose total_marbles selected_marbles) = 12 / 35 := by
  sorry

end marble_selection_probability_l1281_128104


namespace three_digit_number_difference_l1281_128160

theorem three_digit_number_difference (X Y : ℕ) : 
  X > Y → 
  X + Y = 999 → 
  X ≥ 100 → 
  X ≤ 999 → 
  Y ≥ 100 → 
  Y ≤ 999 → 
  1000 * X + Y = 6 * (1000 * Y + X) → 
  X - Y = 715 := by
sorry

end three_digit_number_difference_l1281_128160


namespace function_equation_solution_l1281_128182

theorem function_equation_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, (x - y) * (f x - f y) = f (x - f y) * f (f x - y)) : 
  (∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = x) :=
sorry

end function_equation_solution_l1281_128182


namespace rotational_symmetry_180_l1281_128114

/-- Represents a 2D shape -/
structure Shape :=
  (points : Set (ℝ × ℝ))

/-- Represents a rotation of a shape -/
def rotate (s : Shape) (angle : ℝ) : Shape :=
  sorry

/-- Defines rotational symmetry for a shape -/
def is_rotationally_symmetric (s : Shape) (angle : ℝ) : Prop :=
  rotate s angle = s

/-- The original L-like shape -/
def original_shape : Shape :=
  sorry

/-- Theorem: The shape rotated 180 degrees is rotationally symmetric to the original shape -/
theorem rotational_symmetry_180 :
  is_rotationally_symmetric (rotate original_shape π) π :=
sorry

end rotational_symmetry_180_l1281_128114


namespace rope_around_cylinders_l1281_128125

theorem rope_around_cylinders (rope_length : ℝ) (r1 r2 : ℝ) (rounds1 : ℕ) :
  r1 = 14 →
  r2 = 20 →
  rounds1 = 70 →
  rope_length = 2 * π * r1 * (rounds1 : ℝ) →
  ∃ (rounds2 : ℕ), rounds2 = 49 ∧ rope_length = 2 * π * r2 * (rounds2 : ℝ) :=
by sorry

end rope_around_cylinders_l1281_128125


namespace florist_bouquets_l1281_128171

/-- Calculates the number of bouquets that can be made given the initial number of seeds,
    the number of flowers killed by fungus, and the number of flowers per bouquet. -/
def calculateBouquets (seedsPerColor : ℕ) (redKilled yellowKilled orangeKilled purpleKilled : ℕ) (flowersPerBouquet : ℕ) : ℕ :=
  let redLeft := seedsPerColor - redKilled
  let yellowLeft := seedsPerColor - yellowKilled
  let orangeLeft := seedsPerColor - orangeKilled
  let purpleLeft := seedsPerColor - purpleKilled
  let totalFlowersLeft := redLeft + yellowLeft + orangeLeft + purpleLeft
  totalFlowersLeft / flowersPerBouquet

/-- Theorem stating that given the specific conditions of the problem,
    the florist can make 36 bouquets. -/
theorem florist_bouquets :
  calculateBouquets 125 45 61 30 40 9 = 36 := by
  sorry

end florist_bouquets_l1281_128171


namespace inequality_proof_l1281_128170

theorem inequality_proof (a : ℝ) (h : -1 < a ∧ a < 0) : -a > a^2 ∧ a^2 > -a^3 := by
  sorry

end inequality_proof_l1281_128170


namespace christmas_tree_perimeter_l1281_128110

/-- A Christmas tree is a geometric shape with the following properties:
  1. It is symmetric about the y-axis
  2. It has a height of 1
  3. Its branches form a 45° angle with the vertical
  4. It consists of isosceles right triangles
-/
structure ChristmasTree where
  height : ℝ
  branchAngle : ℝ
  isSymmetric : Bool

/-- The perimeter of a Christmas tree is the sum of all its branch lengths -/
def perimeter (tree : ChristmasTree) : ℝ :=
  sorry

/-- The main theorem stating that the perimeter of a Christmas tree
    with the given properties is 2(1 + √2) -/
theorem christmas_tree_perimeter :
  ∀ (tree : ChristmasTree),
  tree.height = 1 ∧ tree.branchAngle = π/4 ∧ tree.isSymmetric = true →
  perimeter tree = 2 * (1 + Real.sqrt 2) :=
by sorry

end christmas_tree_perimeter_l1281_128110


namespace cube_root_of_one_sixty_fourth_l1281_128139

theorem cube_root_of_one_sixty_fourth (x : ℝ) : x^3 = 1/64 → x = 1/4 := by
  sorry

end cube_root_of_one_sixty_fourth_l1281_128139


namespace units_digit_of_expression_l1281_128102

def units_digit (n : ℤ) : ℕ := n.natAbs % 10

theorem units_digit_of_expression : units_digit (7 * 17 * 1977 - 7^3) = 0 := by
  sorry

end units_digit_of_expression_l1281_128102


namespace hyperbola_m_range_l1281_128137

/-- If the equation x²/(4-m) - y²/(2+m) = 1 represents a hyperbola, 
    then the range of m is (-2, 4) -/
theorem hyperbola_m_range (m : ℝ) : 
  (∃ x y : ℝ, x^2 / (4 - m) - y^2 / (2 + m) = 1) → 
  -2 < m ∧ m < 4 :=
sorry

end hyperbola_m_range_l1281_128137


namespace square_root_problem_l1281_128152

-- Define the variables
variable (a b : ℝ)

-- State the theorem
theorem square_root_problem (h1 : a = 9) (h2 : b = 4/9) :
  (∃ (x : ℝ), x^2 = a ∧ (x = 3 ∨ x = -3)) ∧
  (Real.sqrt (a * b) = 2) →
  (a = 9 ∧ b = 4/9) ∧
  (∃ (y : ℝ), y^2 = a + 2*b ∧ (y = Real.sqrt 89 / 3 ∨ y = -Real.sqrt 89 / 3)) :=
by sorry

end square_root_problem_l1281_128152


namespace reciprocal_expression_l1281_128138

theorem reciprocal_expression (m n : ℝ) (h : m * n = 1) : m * n^2 - (n - 3) = 3 := by
  sorry

end reciprocal_expression_l1281_128138


namespace tan_inequality_l1281_128184

theorem tan_inequality (x : ℝ) (h : 0 ≤ x ∧ x < 1) :
  (2 / Real.pi) * (x / (1 - x)) ≤ Real.tan (Real.pi * x / 2) ∧
  Real.tan (Real.pi * x / 2) ≤ (Real.pi / 2) * (x / (1 - x)) := by
  sorry

end tan_inequality_l1281_128184


namespace coin_flip_probability_l1281_128189

/-- Represents the outcome of a single coin flip -/
inductive CoinFlip
| Heads
| Tails

/-- Represents a sequence of 8 coin flips -/
def CoinSequence := Vector CoinFlip 8

/-- Checks if a given sequence has exactly one pair of consecutive heads and one pair of consecutive tails -/
def hasExactlyOnePairEach (seq : CoinSequence) : Bool :=
  sorry

/-- The total number of possible 8-flip sequences -/
def totalSequences : Nat := 256

/-- The number of favorable sequences (with exactly one pair each of heads and tails) -/
def favorableSequences : Nat := 18

/-- The probability of getting exactly one pair each of heads and tails in 8 flips -/
def probability : Rat := favorableSequences / totalSequences

theorem coin_flip_probability :
  probability = 9 / 128 := by sorry

end coin_flip_probability_l1281_128189


namespace angle_sum_at_point_l1281_128146

theorem angle_sum_at_point (x : ℝ) : 
  (120 : ℝ) + x + x + 2*x = 360 → x = 60 := by
  sorry

end angle_sum_at_point_l1281_128146


namespace union_condition_intersection_condition_l1281_128159

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | -2 < x ∧ x < 3}
def N (m : ℝ) : Set ℝ := {x : ℝ | x - m ≥ 0}

-- Theorem for the first part
theorem union_condition (m : ℝ) : M ∪ N m = N m ↔ m ≤ -2 := by sorry

-- Theorem for the second part
theorem intersection_condition (m : ℝ) : M ∩ N m = ∅ ↔ m ≥ 3 := by sorry

end union_condition_intersection_condition_l1281_128159


namespace probability_identical_after_rotation_l1281_128157

/-- Represents the colors available for painting the cube faces -/
inductive Color
  | Black
  | White
  | Red

/-- Represents a cube with painted faces -/
structure Cube :=
  (faces : Fin 6 → Color)

/-- Checks if a cube satisfies the adjacent face color constraint -/
def validCube (c : Cube) : Prop := sorry

/-- Counts the number of valid cube colorings -/
def validColoringsCount : Nat := sorry

/-- Counts the number of ways cubes can be identical after rotation -/
def identicalAfterRotationCount : Nat := sorry

/-- Theorem stating the probability of three cubes being identical after rotation -/
theorem probability_identical_after_rotation :
  (identicalAfterRotationCount : ℚ) / (validColoringsCount ^ 3 : ℚ) = 1 / 45 := by sorry

end probability_identical_after_rotation_l1281_128157


namespace triangle_angles_sum_l1281_128193

theorem triangle_angles_sum (x y : ℕ) : 
  x > 0 ∧ y > 0 ∧ 8 * x + 13 * y = 130 → x + y = 1289 := by
  sorry

end triangle_angles_sum_l1281_128193


namespace plywood_cut_perimeter_difference_l1281_128101

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.width + r.height)

/-- Represents the possible cuts of the plywood -/
inductive PlywoodCut
  | Vertical
  | Horizontal
  | Mixed

theorem plywood_cut_perimeter_difference :
  let plywood : Rectangle := { width := 6, height := 9 }
  let possible_cuts : List PlywoodCut := [PlywoodCut.Vertical, PlywoodCut.Horizontal, PlywoodCut.Mixed]
  let cut_rectangles : PlywoodCut → Rectangle
    | PlywoodCut.Vertical => { width := 1, height := 9 }
    | PlywoodCut.Horizontal => { width := 1, height := 6 }
    | PlywoodCut.Mixed => { width := 2, height := 3 }
  let perimeters : List ℝ := possible_cuts.map (fun cut => perimeter (cut_rectangles cut))
  (∃ (max_perimeter min_perimeter : ℝ),
    max_perimeter ∈ perimeters ∧
    min_perimeter ∈ perimeters ∧
    max_perimeter = perimeters.maximum ∧
    min_perimeter = perimeters.minimum ∧
    max_perimeter - min_perimeter = 10) := by
  sorry

end plywood_cut_perimeter_difference_l1281_128101


namespace greater_than_negative_two_by_one_l1281_128173

theorem greater_than_negative_two_by_one : 
  ∃ x : ℝ, x = -2 + 1 ∧ x = -1 := by sorry

end greater_than_negative_two_by_one_l1281_128173


namespace common_root_of_three_equations_l1281_128169

/-- Given nonzero real numbers a, b, c, and the fact that any two of the equations
    ax^11 + bx^4 + c = 0, bx^11 + cx^4 + a = 0, cx^11 + ax^4 + b = 0 have a common root,
    prove that all three equations have a common root. -/
theorem common_root_of_three_equations (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h_common_12 : ∃ x : ℝ, a * x^11 + b * x^4 + c = 0 ∧ b * x^11 + c * x^4 + a = 0)
  (h_common_23 : ∃ x : ℝ, b * x^11 + c * x^4 + a = 0 ∧ c * x^11 + a * x^4 + b = 0)
  (h_common_13 : ∃ x : ℝ, a * x^11 + b * x^4 + c = 0 ∧ c * x^11 + a * x^4 + b = 0) :
  ∃ x : ℝ, a * x^11 + b * x^4 + c = 0 ∧ b * x^11 + c * x^4 + a = 0 ∧ c * x^11 + a * x^4 + b = 0 :=
sorry

end common_root_of_three_equations_l1281_128169


namespace equation_solution_l1281_128187

theorem equation_solution : ∃ x : ℚ, (1 / 3 + 1 / x = 2 / 3) ∧ (x = 3) := by
  sorry

end equation_solution_l1281_128187


namespace least_difference_consecutive_primes_l1281_128111

theorem least_difference_consecutive_primes (x y z : ℕ) : 
  Prime x ∧ Prime y ∧ Prime z ∧  -- x, y, and z are prime numbers
  x < y ∧ y < z ∧                -- x < y < z
  y - x > 3 ∧                    -- y - x > 3
  Even x ∧                       -- x is an even integer
  Odd y ∧ Odd z →                -- y and z are odd integers
  ∀ w, (Prime w ∧ Prime (w + 1) ∧ Prime (w + 2) ∧ 
        w < w + 1 ∧ w + 1 < w + 2 ∧
        (w + 1) - w > 3 ∧
        Even w ∧ Odd (w + 1) ∧ Odd (w + 2)) →
    (w + 2) - w ≥ 9 :=
by sorry

end least_difference_consecutive_primes_l1281_128111


namespace chef_cooked_ten_wings_l1281_128174

/-- The number of additional chicken wings cooked by the chef for a group of friends -/
def additional_wings (num_friends : ℕ) (pre_cooked_wings : ℕ) (wings_per_person : ℕ) : ℕ :=
  num_friends * wings_per_person - pre_cooked_wings

/-- Theorem: Given 3 friends, 8 pre-cooked wings, and 6 wings per person, 
    the number of additional wings cooked is 10 -/
theorem chef_cooked_ten_wings : additional_wings 3 8 6 = 10 := by
  sorry

end chef_cooked_ten_wings_l1281_128174


namespace ratio_of_sum_to_difference_l1281_128112

theorem ratio_of_sum_to_difference (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x > y) 
  (h : x + y = 7 * (x - y)) : x / y = 4 / 3 := by
  sorry

end ratio_of_sum_to_difference_l1281_128112


namespace drinks_calculation_l1281_128198

/-- Given a number of pitchers and the number of glasses each pitcher can fill,
    calculate the total number of glasses that can be filled. -/
def total_glasses (num_pitchers : ℕ) (glasses_per_pitcher : ℕ) : ℕ :=
  num_pitchers * glasses_per_pitcher

/-- Theorem: With 9 pitchers and 6 glasses per pitcher, the total number of glasses is 54. -/
theorem drinks_calculation :
  total_glasses 9 6 = 54 := by
  sorry

end drinks_calculation_l1281_128198


namespace tangent_line_range_l1281_128122

/-- Given k > 0, if a line can always be drawn through the point (3, 1) to be tangent
    to the circle (x-2k)^2 + (y-k)^2 = k, then k ∈ (0, 1) ∪ (2, +∞) -/
theorem tangent_line_range (k : ℝ) (h_pos : k > 0) 
  (h_tangent : ∀ (x y : ℝ), (x - 2*k)^2 + (y - k)^2 = k → 
    ∃ (m b : ℝ), y = m*x + b ∧ (3 - 2*k)^2 + (1 - k)^2 ≥ k) :
  k ∈ Set.Ioo 0 1 ∪ Set.Ioi 2 :=
sorry

end tangent_line_range_l1281_128122


namespace tangent_line_and_extrema_l1281_128168

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.cos x - x

theorem tangent_line_and_extrema :
  ∃ (tangent_line : ℝ → ℝ) (max_value min_value : ℝ),
    (∀ x, tangent_line x = 1) ∧
    (f 0 = max_value) ∧
    (f (Real.pi / 2) = min_value) ∧
    (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≤ max_value) ∧
    (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≥ min_value) ∧
    (max_value = 1) ∧
    (min_value = -Real.pi / 2) :=
by sorry

end tangent_line_and_extrema_l1281_128168


namespace right_triangle_angle_calculation_l1281_128175

theorem right_triangle_angle_calculation (x : ℝ) : 
  (3 * x > 3 * x - 40) →  -- Smallest angle condition
  (3 * x + (3 * x - 40) + 90 = 180) →  -- Sum of angles in a triangle
  x = 65 / 3 := by
sorry

end right_triangle_angle_calculation_l1281_128175


namespace P_evaluation_l1281_128109

/-- The polynomial P(x) = x^6 - 3x^3 - x^2 - x - 2 -/
def P (x : ℤ) : ℤ := x^6 - 3*x^3 - x^2 - x - 2

/-- P is irreducible over the integers -/
axiom P_irreducible : Irreducible P

theorem P_evaluation : P 3 = 634 := by
  sorry

end P_evaluation_l1281_128109


namespace lucy_groceries_l1281_128120

/-- The number of packs of cookies Lucy bought -/
def cookies : ℕ := 12

/-- The number of packs of noodles Lucy bought -/
def noodles : ℕ := 16

/-- The total number of packs of groceries Lucy bought -/
def total_groceries : ℕ := cookies + noodles

theorem lucy_groceries : total_groceries = 28 := by
  sorry

end lucy_groceries_l1281_128120


namespace abs_sum_diff_less_than_two_l1281_128186

theorem abs_sum_diff_less_than_two (a b : ℝ) (ha : |a| < 1) (hb : |b| < 1) :
  |a + b| + |a - b| < 2 := by
  sorry

end abs_sum_diff_less_than_two_l1281_128186


namespace chinese_characters_equation_l1281_128177

theorem chinese_characters_equation (x : ℝ) 
  (h1 : x > 100) -- Ensure x - 100 is positive
  (h2 : x ≠ 0) -- Ensure division by x is valid
  : (8000 / x = 6000 / (x - 100)) ↔ 
    (∃ (days : ℝ), 
      days > 0 ∧ 
      days * x = 8000 ∧ 
      days * (x - 100) = 6000) := by
sorry

end chinese_characters_equation_l1281_128177


namespace literary_club_probability_l1281_128164

theorem literary_club_probability : 
  let num_clubs : ℕ := 2
  let num_students : ℕ := 3
  let total_outcomes : ℕ := num_clubs ^ num_students
  let same_club_outcomes : ℕ := num_clubs
  let diff_club_probability : ℚ := 1 - (same_club_outcomes : ℚ) / total_outcomes
  diff_club_probability = 3/4 := by sorry

end literary_club_probability_l1281_128164


namespace students_liking_both_sports_l1281_128116

/-- Given a class of students with information about their sports preferences,
    prove the number of students who like both basketball and table tennis. -/
theorem students_liking_both_sports
  (total : ℕ)
  (basketball : ℕ)
  (table_tennis : ℕ)
  (neither : ℕ)
  (h1 : total = 40)
  (h2 : basketball = 20)
  (h3 : table_tennis = 15)
  (h4 : neither = 8)
  : ∃ x : ℕ, x = 3 ∧ basketball + table_tennis - x + neither = total :=
by sorry

end students_liking_both_sports_l1281_128116


namespace ellipse_and_line_theorem_l1281_128155

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop :=
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧ x^2 / a^2 + y^2 / b^2 = 1

-- Define the focal length
def focal_length (c : ℝ) : Prop := c = 2

-- Define that C passes through P(2, 5/3)
def passes_through_P (C : ℝ → ℝ → Prop) : Prop :=
  C 2 (5/3)

-- Define line l
def line_l (k : ℝ) (x y : ℝ) : Prop :=
  y = k * x + 1

-- Define that l passes through M(0, 1)
def passes_through_M (l : ℝ → ℝ → Prop) : Prop :=
  l 0 1

-- Define the condition for A and B
def vector_condition (A B : ℝ × ℝ) : Prop :=
  (A.1 - 0, A.2 - 1) = (-2/3 * (B.1 - 0), -2/3 * (B.2 - 1))

-- Main theorem
theorem ellipse_and_line_theorem :
  ∀ C : ℝ → ℝ → Prop,
  (∀ x y, C x y ↔ x^2 / 9 + y^2 / 5 = 1) →
  focal_length 2 →
  passes_through_P C →
  ∃ k : ℝ, k = 1/3 ∨ k = -1/3 ∧
    ∀ x y, line_l k x y →
    passes_through_M (line_l k) ∧
    ∃ A B : ℝ × ℝ,
      C A.1 A.2 ∧ C B.1 B.2 ∧
      line_l k A.1 A.2 ∧ line_l k B.1 B.2 ∧
      vector_condition A B :=
sorry

end ellipse_and_line_theorem_l1281_128155


namespace min_value_expression_l1281_128196

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2*a + b = 1) :
  1/a + 1/(2*b) ≥ 9/2 ∧ (1/a + 1/(2*b) = 9/2 ↔ a = 1/3 ∧ b = 1/3) :=
sorry

end min_value_expression_l1281_128196


namespace marcos_strawberries_weight_l1281_128107

theorem marcos_strawberries_weight (total_weight dad_weight : ℝ) 
  (h1 : total_weight = 20)
  (h2 : dad_weight = 17) :
  total_weight - dad_weight = 3 := by
sorry

end marcos_strawberries_weight_l1281_128107


namespace parabola_adjoint_tangent_locus_l1281_128181

/-- Given a parabola y = 2px, prove that the locus of points (x, y) where the tangents 
    to the parabola are its own adjoint lines is described by the equation y² = -p/2 * x -/
theorem parabola_adjoint_tangent_locus (p : ℝ) (x y x₁ y₁ : ℝ) 
  (h1 : y₁ = 2 * p * x₁)  -- Original parabola equation
  (h2 : x = -x₁)          -- Relation between x and x₁
  (h3 : y = y₁ / 2)       -- Relation between y and y₁
  : y^2 = -p/2 * x := by sorry

end parabola_adjoint_tangent_locus_l1281_128181


namespace repeating_37_equals_fraction_l1281_128165

/-- The repeating decimal 0.373737... -/
def repeating_37 : ℚ := 37 / 99

theorem repeating_37_equals_fraction : 
  repeating_37 = 37 / 99 := by sorry

end repeating_37_equals_fraction_l1281_128165


namespace lemons_for_ten_gallons_l1281_128179

/-- The number of lemons required to make a certain amount of lemonade -/
structure LemonadeRecipe where
  lemons : ℕ
  gallons : ℕ

/-- Calculates the number of lemons needed for a given number of gallons,
    based on a known recipe. The result is rounded up to the nearest integer. -/
def calculate_lemons (recipe : LemonadeRecipe) (target_gallons : ℕ) : ℕ :=
  ((recipe.lemons : ℚ) * target_gallons / recipe.gallons).ceil.toNat

/-- The known recipe for lemonade -/
def known_recipe : LemonadeRecipe := ⟨48, 64⟩

/-- The target amount of lemonade to make -/
def target_gallons : ℕ := 10

/-- Theorem stating that 8 lemons are needed to make 10 gallons of lemonade -/
theorem lemons_for_ten_gallons :
  calculate_lemons known_recipe target_gallons = 8 := by sorry

end lemons_for_ten_gallons_l1281_128179


namespace fixed_point_of_exponential_function_l1281_128176

/-- Given a > 0 and a ≠ 1, prove that f(x) = a^(x-1) + 3 passes through (1, 4) -/
theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) (ha' : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x-1) + 3
  f 1 = 4 := by
  sorry

end fixed_point_of_exponential_function_l1281_128176


namespace video_voting_result_l1281_128142

/-- Represents the voting system for a video --/
structure VideoVoting where
  totalVotes : ℕ
  likePercentage : ℚ
  finalScore : ℤ

/-- Theorem stating the conditions and the result to be proved --/
theorem video_voting_result (v : VideoVoting) 
  (h1 : v.likePercentage = 3/4)
  (h2 : v.finalScore = 140) :
  v.totalVotes = 280 := by
  sorry

end video_voting_result_l1281_128142


namespace rectangle_width_l1281_128118

theorem rectangle_width (square_perimeter : ℝ) (rectangle_length : ℝ) (rectangle_width : ℝ) : 
  square_perimeter = 160 →
  rectangle_length = 32 →
  (square_perimeter / 4) ^ 2 = 5 * (rectangle_length * rectangle_width) →
  rectangle_width = 10 := by
sorry

end rectangle_width_l1281_128118


namespace fraction_problem_l1281_128115

theorem fraction_problem (a b c d e f : ℚ) :
  (∃ (k : ℚ), a = k * 1 ∧ b = k * 2 ∧ c = k * 5) →
  (∃ (m : ℚ), d = m * 1 ∧ e = m * 3 ∧ f = m * 7) →
  (a / d + b / e + c / f) / 3 = 200 / 441 →
  a / d = 4 / 7 ∧ b / e = 8 / 21 ∧ c / f = 20 / 49 := by
sorry

end fraction_problem_l1281_128115


namespace product_equation_solution_l1281_128143

theorem product_equation_solution :
  ∀ B : ℕ,
  B < 10 →
  (10 * B + 4) * (10 * 8 + B) = 7008 →
  B = 7 := by
sorry

end product_equation_solution_l1281_128143


namespace sum_and_count_theorem_l1281_128129

def sum_of_range (a b : ℕ) : ℕ := 
  ((b - a + 1) * (a + b)) / 2

def count_even_in_range (a b : ℕ) : ℕ :=
  (b - a) / 2 + 1

theorem sum_and_count_theorem : 
  sum_of_range 50 60 + count_even_in_range 50 60 = 611 := by
  sorry

end sum_and_count_theorem_l1281_128129


namespace selling_price_theorem_l1281_128128

/-- The selling price of an article that results in a loss, given the cost price and a selling price that results in a profit. -/
def selling_price_with_loss (cost_price profit_price : ℕ) : ℕ :=
  2 * cost_price - profit_price

theorem selling_price_theorem (cost_price profit_price : ℕ) 
  (h1 : cost_price = 64)
  (h2 : profit_price = 86)
  (h3 : profit_price > cost_price) :
  selling_price_with_loss cost_price profit_price = 42 := by
  sorry

#eval selling_price_with_loss 64 86  -- Should output 42

end selling_price_theorem_l1281_128128


namespace xyz_equals_one_l1281_128108

theorem xyz_equals_one (x y z : ℝ) 
  (eq1 : x + 1/y = 4)
  (eq2 : y + 1/z = 1)
  (eq3 : z + 1/x = 7/3) :
  x * y * z = 1 := by
sorry

end xyz_equals_one_l1281_128108


namespace vacation_cost_distribution_l1281_128150

/-- Represents the vacation cost distribution problem -/
theorem vacation_cost_distribution 
  (anna_paid ben_paid carol_paid dan_paid : ℚ)
  (a b c : ℚ)
  (h1 : anna_paid = 130)
  (h2 : ben_paid = 150)
  (h3 : carol_paid = 110)
  (h4 : dan_paid = 190)
  (h5 : (anna_paid + ben_paid + carol_paid + dan_paid) / 4 = 145)
  (h6 : a = 5)
  (h7 : b = 5)
  (h8 : c = 35)
  : a - b + c = 35 := by
  sorry

end vacation_cost_distribution_l1281_128150


namespace parabola_c_value_l1281_128153

/-- A parabola with equation x = ay^2 + by + c, vertex at (-3, -1), and passing through (-1, 1) -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  vertex_x : ℝ := -3
  vertex_y : ℝ := -1
  point_x : ℝ := -1
  point_y : ℝ := 1
  eq_vertex : -3 = a * (-1)^2 + b * (-1) + c
  eq_point : -1 = a * 1^2 + b * 1 + c

/-- The value of c for the given parabola is -2.5 -/
theorem parabola_c_value (p : Parabola) : p.c = -2.5 := by
  sorry

end parabola_c_value_l1281_128153


namespace inequality_equivalence_l1281_128103

theorem inequality_equivalence (x : ℝ) : 3 * x^2 - 5 * x > 9 ↔ x < -1 ∨ x > 3 := by
  sorry

end inequality_equivalence_l1281_128103


namespace P_n_has_n_distinct_real_roots_P_2018_has_2018_distinct_real_roots_l1281_128119

-- Define the sequence of polynomials
def P : ℕ → (ℝ → ℝ)
  | 0 => λ _ => 1
  | 1 => λ x => x
  | (n + 2) => λ x => x * P (n + 1) x - P n x

-- Define a function to count distinct real roots
noncomputable def count_distinct_real_roots (f : ℝ → ℝ) : ℕ := sorry

-- State the theorem
theorem P_n_has_n_distinct_real_roots (n : ℕ) :
  count_distinct_real_roots (P n) = n := by sorry

-- The specific case for P₂₀₁₈
theorem P_2018_has_2018_distinct_real_roots :
  count_distinct_real_roots (P 2018) = 2018 := by sorry

end P_n_has_n_distinct_real_roots_P_2018_has_2018_distinct_real_roots_l1281_128119


namespace original_decimal_proof_l1281_128154

theorem original_decimal_proof (x : ℝ) : x * 12 = 84.6 ↔ x = 7.05 := by
  sorry

end original_decimal_proof_l1281_128154


namespace bridge_length_calculation_l1281_128144

/-- The length of a bridge given train parameters and crossing time -/
theorem bridge_length_calculation (train_length : ℝ) (train_speed_kmph : ℝ) (crossing_time : ℝ) :
  train_length = 110 →
  train_speed_kmph = 72 →
  crossing_time = 13.998880089592832 →
  ∃ (bridge_length : ℝ), 
    (169.97 < bridge_length) ∧ 
    (bridge_length < 169.99) ∧
    (bridge_length = train_speed_kmph * (1000 / 3600) * crossing_time - train_length) :=
by sorry

end bridge_length_calculation_l1281_128144


namespace custom_op_solution_l1281_128151

/-- Custom operation for integers -/
def customOp (a b : ℤ) : ℤ := (a - 1) * (b - 1)

/-- Theorem: Given the custom operation and x9 = 160, x must equal 21 -/
theorem custom_op_solution : ∃ x : ℤ, customOp x 9 = 160 ∧ x = 21 := by
  sorry

end custom_op_solution_l1281_128151


namespace geometric_sequence_sum_l1281_128140

-- Define a geometric sequence
def isGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- State the theorem
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  isGeometricSequence a →
  a 4 + a 7 = 2 →
  a 5 * a 6 = -8 →
  a 1 + a 10 = -7 := by
  sorry


end geometric_sequence_sum_l1281_128140


namespace congruence_solution_l1281_128185

theorem congruence_solution : ∃ x : ℤ, x ≡ 1 [ZMOD 7] ∧ x ≡ 2 [ZMOD 11] :=
by
  use 57
  sorry

end congruence_solution_l1281_128185


namespace intersection_of_A_and_B_l1281_128178

def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 4}

theorem intersection_of_A_and_B : A ∩ B = {2} := by sorry

end intersection_of_A_and_B_l1281_128178


namespace payment_calculation_l1281_128183

/-- Represents a store's pricing and promotion options for suits and ties. -/
structure StorePricing where
  suit_price : ℕ
  tie_price : ℕ
  option1 : ℕ → ℕ  -- Function representing the cost for Option 1
  option2 : ℕ → ℕ  -- Function representing the cost for Option 2

/-- Calculates the payment for a customer buying suits and ties under different options. -/
def calculate_payment (pricing : StorePricing) (suits : ℕ) (ties : ℕ) : ℕ × ℕ :=
  (pricing.option1 ties, pricing.option2 ties)

/-- Theorem stating the correct calculation of payments for the given problem. -/
theorem payment_calculation (x : ℕ) (h : x > 20) :
  let pricing := StorePricing.mk 1000 200
    (fun ties => 20000 + 200 * (ties - 20))
    (fun ties => (20 * 1000 + ties * 200) * 9 / 10)
  (calculate_payment pricing 20 x).1 = 200 * x + 16000 ∧
  (calculate_payment pricing 20 x).2 = 180 * x + 18000 := by
  sorry

end payment_calculation_l1281_128183


namespace veranda_area_l1281_128163

/-- The area of a veranda surrounding a rectangular room -/
theorem veranda_area (room_length room_width veranda_width : ℝ) 
  (h1 : room_length = 19)
  (h2 : room_width = 12)
  (h3 : veranda_width = 2) : 
  (room_length + 2 * veranda_width) * (room_width + 2 * veranda_width) - room_length * room_width = 140 :=
by sorry

end veranda_area_l1281_128163


namespace cubic_polynomial_root_l1281_128166

-- Define the polynomial Q(x) = x³ - 5x² + 6x - 2
def Q (x : ℝ) : ℝ := x^3 - 5*x^2 + 6*x - 2

-- Theorem statement
theorem cubic_polynomial_root :
  -- Q is a monic cubic polynomial with integer coefficients
  (∀ x, Q x = x^3 - 5*x^2 + 6*x - 2) ∧
  -- The leading coefficient is 1 (monic)
  (∃ a b c, ∀ x, Q x = x^3 + a*x^2 + b*x + c) ∧
  -- All coefficients are integers
  (∃ a b c : ℤ, ∀ x, Q x = x^3 + a*x^2 + b*x + c) ∧
  -- √2 + 2 is a root of Q
  Q (Real.sqrt 2 + 2) = 0 :=
sorry

end cubic_polynomial_root_l1281_128166


namespace increasing_sequence_bound_l1281_128148

theorem increasing_sequence_bound (a : ℝ) :
  (∀ n : ℕ+, (n.val - a)^2 < ((n + 1).val - a)^2) →
  a < 3/2 := by
sorry

end increasing_sequence_bound_l1281_128148


namespace amount_paid_is_fifty_l1281_128162

/-- Represents the purchase and change scenario --/
structure Purchase where
  book_cost : ℕ
  pen_cost : ℕ
  ruler_cost : ℕ
  change_received : ℕ

/-- Calculates the total cost of items --/
def total_cost (p : Purchase) : ℕ :=
  p.book_cost + p.pen_cost + p.ruler_cost

/-- Calculates the amount paid --/
def amount_paid (p : Purchase) : ℕ :=
  total_cost p + p.change_received

/-- Theorem stating that the amount paid is $50 --/
theorem amount_paid_is_fifty (p : Purchase) 
  (h1 : p.book_cost = 25)
  (h2 : p.pen_cost = 4)
  (h3 : p.ruler_cost = 1)
  (h4 : p.change_received = 20) :
  amount_paid p = 50 := by
  sorry

end amount_paid_is_fifty_l1281_128162


namespace circle_M_properties_l1281_128188

-- Define the circle M
def circle_M (x y : ℝ) : Prop := (x - 1)^2 + (y + 2)^2 = 2

-- Define the center of the circle
def center_M : ℝ × ℝ := (1, -2)

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := x + y - 1 = 0

-- Define point P
def point_P : ℝ × ℝ := (2, -1)

-- Define point A
def point_A : ℝ × ℝ := (3, 0)

-- Theorem statement
theorem circle_M_properties :
  (center_M.2 = -2 * center_M.1) ∧ 
  tangent_line point_P.1 point_P.2 ∧
  (∀ x y, tangent_line x y → ¬ circle_M x y) ∧
  circle_M point_P.1 point_P.2 →
  (∀ x y, circle_M x y → 
    Real.sqrt ((x - point_A.1)^2 + (y - point_A.2)^2) ≥ Real.sqrt 2) ∧
  (∃ x y, circle_M x y ∧ 
    Real.sqrt ((x - point_A.1)^2 + (y - point_A.2)^2) = Real.sqrt 2) :=
by sorry

end circle_M_properties_l1281_128188


namespace article_price_calculation_l1281_128127

theorem article_price_calculation (P : ℝ) : 
  P * 0.75 * 0.85 * 1.10 * 1.05 = 1226.25 → P = 1843.75 := by
  sorry

end article_price_calculation_l1281_128127


namespace equality_of_fractions_l1281_128161

theorem equality_of_fractions (M N : ℚ) : 
  (4 : ℚ) / 7 = M / 63 ∧ (4 : ℚ) / 7 = 84 / N → M + N = 183 := by
  sorry

end equality_of_fractions_l1281_128161


namespace max_subtract_add_result_l1281_128172

def S : Set Int := {-20, -10, 0, 5, 15, 25}

theorem max_subtract_add_result (a b c : Int) (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) :
  (a - b + c) ≤ 70 ∧ ∃ (x y z : Int), x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ x - y + z = 70 := by
  sorry

end max_subtract_add_result_l1281_128172


namespace intersection_sum_zero_l1281_128191

theorem intersection_sum_zero (x₁ x₂ : ℝ) (h₁ : x₁^2 + 6^2 = 144) (h₂ : x₂^2 + 6^2 = 144) :
  x₁ + x₂ = 0 := by
  sorry

end intersection_sum_zero_l1281_128191


namespace min_value_of_z_l1281_128131

theorem min_value_of_z (x y : ℝ) : 
  ∃ (m : ℝ), m = 8 ∧ ∀ (x y : ℝ), 3*x^2 + 4*y^2 + 12*x - 8*y + 3*x*y + 30 ≥ m := by
  sorry

end min_value_of_z_l1281_128131
