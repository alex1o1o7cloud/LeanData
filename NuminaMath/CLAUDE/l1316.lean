import Mathlib

namespace NUMINAMATH_CALUDE_log_base_4_properties_l1316_131647

noncomputable def y (x : ℝ) : ℝ := Real.log x / Real.log 4

theorem log_base_4_properties :
  (∀ x : ℝ, x = 1 → y x = 0) ∧
  (∀ x : ℝ, x = 4 → y x = 1) ∧
  (∀ x : ℝ, x = -4 → ¬ ∃ (r : ℝ), y x = r) ∧
  (∀ x : ℝ, 0 < x → x < 1 → y x < 0 ∧ 
    ∀ ε > 0, ∃ δ > 0, ∀ x', 0 < x' → x' < δ → y x' < -ε) :=
by sorry

end NUMINAMATH_CALUDE_log_base_4_properties_l1316_131647


namespace NUMINAMATH_CALUDE_complex_magnitude_l1316_131616

theorem complex_magnitude (w : ℂ) (h : w^2 = -75 + 100*I) : Complex.abs w = 5 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l1316_131616


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l1316_131650

/-- An arithmetic sequence defined by the given recurrence relation. -/
def ArithmeticSequence (x : ℕ → ℚ) : Prop :=
  ∀ n ≥ 3, x (n - 1) = (x n + x (n - 1) + x (n - 2)) / 3

/-- The main theorem stating the ratio of differences in the sequence. -/
theorem arithmetic_sequence_ratio 
  (x : ℕ → ℚ) 
  (h : ArithmeticSequence x) : 
  (x 300 - x 33) / (x 333 - x 3) = 89 / 110 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l1316_131650


namespace NUMINAMATH_CALUDE_sum_of_squares_l1316_131643

theorem sum_of_squares (x y : ℝ) (h1 : x * y = 120) (h2 : x + y = 23) : x^2 + y^2 = 289 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l1316_131643


namespace NUMINAMATH_CALUDE_jason_gave_four_cards_l1316_131658

/-- The number of Pokemon cards Jason gave to his friends -/
def cards_given (initial_cards current_cards : ℕ) : ℕ :=
  initial_cards - current_cards

theorem jason_gave_four_cards :
  let initial_cards : ℕ := 9
  let current_cards : ℕ := 5
  cards_given initial_cards current_cards = 4 := by
  sorry

end NUMINAMATH_CALUDE_jason_gave_four_cards_l1316_131658


namespace NUMINAMATH_CALUDE_is_quadratic_equation_l1316_131603

theorem is_quadratic_equation (x : ℝ) : ∃ (a b c : ℝ), a ≠ 0 ∧ 3*(x-1)^2 = 2*(x-1) ↔ a*x^2 + b*x + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_is_quadratic_equation_l1316_131603


namespace NUMINAMATH_CALUDE_fourth_selected_is_34_l1316_131683

/-- Represents a systematic sampling of students -/
structure SystematicSampling where
  total_students : Nat
  num_groups : Nat
  first_selected : Nat
  second_selected : Nat

/-- Calculates the number of the selected student for a given group -/
def selected_student (s : SystematicSampling) (group : Nat) : Nat :=
  s.first_selected + (s.total_students / s.num_groups) * group

/-- Theorem stating that the fourth selected student will be number 34 -/
theorem fourth_selected_is_34 (s : SystematicSampling) 
  (h1 : s.total_students = 50)
  (h2 : s.num_groups = 5)
  (h3 : s.first_selected = 4)
  (h4 : s.second_selected = 14) :
  selected_student s 3 = 34 := by
  sorry

end NUMINAMATH_CALUDE_fourth_selected_is_34_l1316_131683


namespace NUMINAMATH_CALUDE_factorial_difference_l1316_131659

theorem factorial_difference (n : ℕ) (h : n.factorial = 362880) : 
  (n + 1).factorial - n.factorial = 3265920 := by
  sorry

end NUMINAMATH_CALUDE_factorial_difference_l1316_131659


namespace NUMINAMATH_CALUDE_min_value_of_z3_l1316_131678

open Complex

theorem min_value_of_z3 (z₁ z₂ z₃ : ℂ) 
  (h1 : ∃ (a : ℝ), z₁ / z₂ = Complex.I * a)
  (h2 : abs z₁ = 1)
  (h3 : abs z₂ = 1)
  (h4 : abs (z₁ + z₂ + z₃) = 1) :
  abs z₃ ≥ Real.sqrt 2 - 1 := by
sorry

end NUMINAMATH_CALUDE_min_value_of_z3_l1316_131678


namespace NUMINAMATH_CALUDE_teacher_selection_arrangements_l1316_131667

theorem teacher_selection_arrangements (n_male : ℕ) (n_female : ℕ) (n_select : ℕ) : 
  n_male = 5 → n_female = 4 → n_select = 3 →
  (Nat.choose (n_male + n_female) n_select - Nat.choose n_male n_select - Nat.choose n_female n_select) = 70 := by
  sorry

end NUMINAMATH_CALUDE_teacher_selection_arrangements_l1316_131667


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1316_131609

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x * (x - 2) < 0}
def B : Set ℝ := {x : ℝ | Real.log x > 0}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1316_131609


namespace NUMINAMATH_CALUDE_jimmy_stair_climbing_l1316_131682

/-- The sum of the first n terms of an arithmetic sequence -/
def arithmetic_sum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

/-- Jimmy's stair climbing problem -/
theorem jimmy_stair_climbing : arithmetic_sum 25 7 6 = 255 := by
  sorry

end NUMINAMATH_CALUDE_jimmy_stair_climbing_l1316_131682


namespace NUMINAMATH_CALUDE_equation_solution_l1316_131626

theorem equation_solution (y : ℝ) (h : (1 : ℝ) / 3 + 1 / y = 7 / 9) : y = 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1316_131626


namespace NUMINAMATH_CALUDE_magnitude_of_complex_fraction_l1316_131695

theorem magnitude_of_complex_fraction :
  let z : ℂ := (1 + Complex.I) / (2 - 2 * Complex.I)
  Complex.abs z = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_magnitude_of_complex_fraction_l1316_131695


namespace NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l1316_131661

theorem complex_number_in_second_quadrant (z : ℂ) (a : ℝ) :
  z = a + Complex.I * Real.sqrt 3 →
  (Complex.re z < 0 ∧ Complex.im z > 0) →
  Complex.abs z = 2 →
  z = -1 + Complex.I * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l1316_131661


namespace NUMINAMATH_CALUDE_pictures_in_first_album_l1316_131665

theorem pictures_in_first_album (total_pictures : ℕ) (albums : ℕ) (pictures_per_album : ℕ) :
  total_pictures = 35 →
  albums = 3 →
  pictures_per_album = 7 →
  total_pictures - (albums * pictures_per_album) = 14 := by
  sorry

end NUMINAMATH_CALUDE_pictures_in_first_album_l1316_131665


namespace NUMINAMATH_CALUDE_six_digit_number_divisible_by_7_8_9_l1316_131637

theorem six_digit_number_divisible_by_7_8_9 : ∃ (n₁ n₂ : ℕ),
  n₁ ≠ n₂ ∧
  523000 ≤ n₁ ∧ n₁ < 524000 ∧
  523000 ≤ n₂ ∧ n₂ < 524000 ∧
  n₁ % 7 = 0 ∧ n₁ % 8 = 0 ∧ n₁ % 9 = 0 ∧
  n₂ % 7 = 0 ∧ n₂ % 8 = 0 ∧ n₂ % 9 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_six_digit_number_divisible_by_7_8_9_l1316_131637


namespace NUMINAMATH_CALUDE_line_inclination_angle_l1316_131694

/-- Given a line l with equation represented by the determinant |1 0 2; x 2 3; y -1 2| = 0,
    prove that its inclination angle is π - arctan(1/2) -/
theorem line_inclination_angle (x y : ℝ) : 
  let l : Set (ℝ × ℝ) := {(x, y) | Matrix.det !![1, 0, 2; x, 2, 3; y, -1, 2] = 0}
  ∃ θ : ℝ, θ = π - Real.arctan (1/2) ∧ 
    ∀ (x₁ y₁ x₂ y₂ : ℝ), (x₁, y₁) ∈ l → (x₂, y₂) ∈ l → 
      x₁ ≠ x₂ → θ = Real.arctan ((y₂ - y₁) / (x₂ - x₁)) := by
  sorry

end NUMINAMATH_CALUDE_line_inclination_angle_l1316_131694


namespace NUMINAMATH_CALUDE_polynomial_value_theorem_l1316_131692

-- Define the function f
def f (a b c d e : ℝ) (x : ℝ) : ℝ := a * x^4 + b * x^3 + c * x^2 + d * x + e

-- State the theorem
theorem polynomial_value_theorem (a b c d e : ℝ) :
  f a b c d e (-1) = 2 → 16 * a - 8 * b + 4 * c - 2 * d + e = 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_theorem_l1316_131692


namespace NUMINAMATH_CALUDE_y1_gt_y2_iff_x_gt_neg_one_fifth_l1316_131601

/-- Given y₁ = a^(2x+1), y₂ = a^(-3x), a > 0, and a > 1, y₁ > y₂ if and only if x > -1/5 -/
theorem y1_gt_y2_iff_x_gt_neg_one_fifth (a x : ℝ) (h1 : a > 0) (h2 : a > 1) :
  a^(2*x + 1) > a^(-3*x) ↔ x > -1/5 := by
  sorry

end NUMINAMATH_CALUDE_y1_gt_y2_iff_x_gt_neg_one_fifth_l1316_131601


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1316_131638

theorem inequality_solution_set (a x : ℝ) :
  (a^2 - 4) * x^2 + 4 * x - 1 > 0 ↔
  (a = 2 ∨ a = -2 → x > 1/4) ∧
  (a > 2 → x > 1/(a + 2) ∨ x < 1/(2 - a)) ∧
  (a < -2 → x < 1/(a + 2) ∨ x > 1/(2 - a)) ∧
  (-2 < a ∧ a < 2 → 1/(a + 2) < x ∧ x < 1/(2 - a)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1316_131638


namespace NUMINAMATH_CALUDE_integral_inequality_l1316_131697

theorem integral_inequality (m : ℕ+) : 
  0 ≤ ∫ x in (0:ℝ)..1, (x + 1 - Real.sqrt (x^2 + 2*x * Real.cos (2*Real.pi / (2*(m:ℝ) + 1)) + 1)) ∧
  ∫ x in (0:ℝ)..1, (x + 1 - Real.sqrt (x^2 + 2*x * Real.cos (2*Real.pi / (2*(m:ℝ) + 1)) + 1)) ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_integral_inequality_l1316_131697


namespace NUMINAMATH_CALUDE_f_max_value_l1316_131644

/-- The quadratic function f(x) = -2x^2 - 8x + 16 -/
def f (x : ℝ) : ℝ := -2 * x^2 - 8 * x + 16

/-- The maximum value of f(x) -/
def max_value : ℝ := 24

/-- The x-coordinate where f(x) achieves its maximum value -/
def max_point : ℝ := -2

theorem f_max_value :
  (∀ x : ℝ, f x ≤ max_value) ∧ f max_point = max_value := by sorry

end NUMINAMATH_CALUDE_f_max_value_l1316_131644


namespace NUMINAMATH_CALUDE_sum_difference_equals_210_l1316_131620

theorem sum_difference_equals_210 : 152 + 29 + 25 + 14 - 10 = 210 := by
  sorry

end NUMINAMATH_CALUDE_sum_difference_equals_210_l1316_131620


namespace NUMINAMATH_CALUDE_trivia_team_score_l1316_131686

/-- Represents a trivia team with their scores -/
structure TriviaTeam where
  totalMembers : Nat
  absentMembers : Nat
  scores : List Nat

/-- Calculates the total score of a trivia team -/
def totalScore (team : TriviaTeam) : Nat :=
  team.scores.sum

/-- Theorem: The trivia team's total score is 26 points -/
theorem trivia_team_score : 
  ∀ (team : TriviaTeam), 
    team.totalMembers = 8 → 
    team.absentMembers = 3 → 
    team.scores = [4, 6, 8, 8] → 
    totalScore team = 26 := by
  sorry

end NUMINAMATH_CALUDE_trivia_team_score_l1316_131686


namespace NUMINAMATH_CALUDE_solution_characterization_l1316_131628

/-- The set of integers satisfying the given conditions -/
def SolutionSet : Set ℕ := {16, 72, 520}

/-- The predicate defining the conditions of the problem -/
def SatisfiesConditions (n : ℕ) : Prop :=
  n ≥ 2 ∧
  ∃ (a d : ℕ), 
    (a.Prime ∧ ∀ p, p.Prime → p ∣ n → a ≤ p) ∧  -- a is the smallest prime divisor of n
    (d > 0 ∧ d ∣ n) ∧  -- d is a positive divisor of n
    n = a^3 + d^3  -- n = a^3 + d^3

/-- The main theorem stating that SolutionSet contains exactly the numbers satisfying the conditions -/
theorem solution_characterization :
  ∀ n : ℕ, n ∈ SolutionSet ↔ SatisfiesConditions n :=
by sorry

end NUMINAMATH_CALUDE_solution_characterization_l1316_131628


namespace NUMINAMATH_CALUDE_probability_one_class_no_spot_l1316_131680

/-- The number of spots for top students -/
def num_spots : ℕ := 6

/-- The number of classes -/
def num_classes : ℕ := 3

/-- The number of ways to distribute spots such that exactly one class doesn't receive a spot -/
def favorable_outcomes : ℕ := (num_classes.choose 2) * ((num_spots - 1).choose 1)

/-- The total number of ways to distribute spots among classes -/
def total_outcomes : ℕ := 
  (num_classes.choose 1) + 
  (num_classes.choose 2) * ((num_spots - 1).choose 1) + 
  (num_classes.choose 3) * ((num_spots - 1).choose 2)

/-- The probability that exactly one class does not receive a spot -/
theorem probability_one_class_no_spot : 
  (favorable_outcomes : ℚ) / total_outcomes = 15 / 28 := by
  sorry

end NUMINAMATH_CALUDE_probability_one_class_no_spot_l1316_131680


namespace NUMINAMATH_CALUDE_volume_ratio_of_cubes_l1316_131693

/-- The ratio of volumes of two cubes -/
theorem volume_ratio_of_cubes (inches_per_foot : ℚ) (edge_length_small : ℚ) (edge_length_large : ℚ) :
  inches_per_foot = 12 →
  edge_length_small = 4 →
  edge_length_large = 2 * inches_per_foot →
  (edge_length_small ^ 3) / (edge_length_large ^ 3) = 1 / 216 := by
  sorry

end NUMINAMATH_CALUDE_volume_ratio_of_cubes_l1316_131693


namespace NUMINAMATH_CALUDE_semicircle_perimeter_equilateral_triangle_l1316_131664

/-- The perimeter of a region formed by three semicircular arcs,
    each constructed on a side of an equilateral triangle with side length 1,
    is equal to 3π/2. -/
theorem semicircle_perimeter_equilateral_triangle :
  let triangle_side_length : ℝ := 1
  let semicircle_radius : ℝ := triangle_side_length / 2
  let num_sides : ℕ := 3
  let perimeter : ℝ := num_sides * (π * semicircle_radius)
  perimeter = 3 * π / 2 := by sorry

end NUMINAMATH_CALUDE_semicircle_perimeter_equilateral_triangle_l1316_131664


namespace NUMINAMATH_CALUDE_most_frequent_digit_l1316_131633

/-- The digital root of a natural number -/
def digitalRoot (n : ℕ) : ℕ :=
  if n = 0 then 0 else (n - 1) % 9 + 1

/-- The count of occurrences of each digit (1-9) in the digital roots of numbers from 1 to 1,000,000 -/
def digitCounts : Fin 9 → ℕ
| ⟨i, _⟩ => if i = 0 then 111112 else 111111

theorem most_frequent_digit :
  ∃ (d : Fin 9), ∀ (d' : Fin 9), digitCounts d ≥ digitCounts d' ∧
  (d = ⟨0, by norm_num⟩ ∨ digitCounts d > digitCounts d') :=
sorry

end NUMINAMATH_CALUDE_most_frequent_digit_l1316_131633


namespace NUMINAMATH_CALUDE_or_implies_at_least_one_true_l1316_131668

theorem or_implies_at_least_one_true (p q : Prop) : 
  (p ∨ q) → (p ∨ q) := by sorry

end NUMINAMATH_CALUDE_or_implies_at_least_one_true_l1316_131668


namespace NUMINAMATH_CALUDE_quadratic_equation_unique_solution_positive_n_value_l1316_131662

theorem quadratic_equation_unique_solution (n : ℝ) : 
  (∃! x : ℝ, 5 * x^2 + n * x + 45 = 0) → n = 30 ∨ n = -30 :=
by sorry

theorem positive_n_value (n : ℝ) : 
  (∃! x : ℝ, 5 * x^2 + n * x + 45 = 0) → n > 0 → n = 30 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_unique_solution_positive_n_value_l1316_131662


namespace NUMINAMATH_CALUDE_average_towel_price_l1316_131666

def towel_price_problem (price1 price2 price3 : ℕ) (quantity1 quantity2 quantity3 : ℕ) : Prop :=
  let total_cost := price1 * quantity1 + price2 * quantity2 + price3 * quantity3
  let total_quantity := quantity1 + quantity2 + quantity3
  (total_cost : ℚ) / total_quantity = 205

theorem average_towel_price :
  towel_price_problem 100 150 500 3 5 2 := by
  sorry

end NUMINAMATH_CALUDE_average_towel_price_l1316_131666


namespace NUMINAMATH_CALUDE_product_remainder_l1316_131698

theorem product_remainder (x : ℕ) :
  (1274 * x * 1277 * 1285) % 12 = 6 → x % 12 = 1 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_l1316_131698


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l1316_131648

theorem polynomial_divisibility : 
  ∃ (q : ℝ → ℝ), ∀ x : ℝ, 5 * x^2 - 6 * x - 95 = (x - 5) * q x := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l1316_131648


namespace NUMINAMATH_CALUDE_steve_jellybeans_l1316_131655

/-- Given the following conditions:
    - Matilda has half as many jellybeans as Matt
    - Matt has ten times as many jellybeans as Steve
    - Matilda has 420 jellybeans
    Prove that Steve has 84 jellybeans -/
theorem steve_jellybeans (steve matt matilda : ℕ) 
  (h1 : matilda = matt / 2)
  (h2 : matt = 10 * steve)
  (h3 : matilda = 420) :
  steve = 84 := by
  sorry

end NUMINAMATH_CALUDE_steve_jellybeans_l1316_131655


namespace NUMINAMATH_CALUDE_complement_A_in_U_l1316_131657

def U : Set ℕ := {x : ℕ | x ≥ 2}
def A : Set ℕ := {x : ℕ | x^2 ≥ 5}

theorem complement_A_in_U : (U \ A) = {2} := by sorry

end NUMINAMATH_CALUDE_complement_A_in_U_l1316_131657


namespace NUMINAMATH_CALUDE_tangent_circle_exists_l1316_131634

-- Define the types for points and circles
def Point := ℝ × ℝ
def Circle := Point × ℝ  -- Center and radius

-- Define a function to check if a point is on a circle
def is_on_circle (p : Point) (c : Circle) : Prop :=
  let (center, radius) := c
  (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2

-- Define a function to check if two circles are tangent
def are_circles_tangent (c1 c2 : Circle) : Prop :=
  let (center1, radius1) := c1
  let (center2, radius2) := c2
  (center1.1 - center2.1)^2 + (center1.2 - center2.2)^2 = (radius1 + radius2)^2

-- Theorem statement
theorem tangent_circle_exists (c1 c2 : Circle) (T : Point) 
  (h1 : is_on_circle T c1) : 
  ∃ (c : Circle), are_circles_tangent c c1 ∧ are_circles_tangent c c2 ∧ is_on_circle T c :=
sorry

end NUMINAMATH_CALUDE_tangent_circle_exists_l1316_131634


namespace NUMINAMATH_CALUDE_cos_theta_range_l1316_131687

-- Define the circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6*x - 8*y + 21 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the center of circle2
def O : ℝ × ℝ := (0, 0)

-- Define a point P on circle1
def P : ℝ × ℝ := sorry

-- Define points A and B where tangents from P touch circle2
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry

-- Define the angle θ between vectors PA and PB
def θ : ℝ := sorry

-- State the theorem
theorem cos_theta_range :
  circle1 P.1 P.2 →
  circle2 A.1 A.2 →
  circle2 B.1 B.2 →
  (1 : ℝ) / 9 ≤ Real.cos θ ∧ Real.cos θ ≤ 41 / 49 :=
sorry

end NUMINAMATH_CALUDE_cos_theta_range_l1316_131687


namespace NUMINAMATH_CALUDE_unused_card_is_one_l1316_131646

def cards : Finset Nat := {1, 3, 4}

def largest_two_digit (a b : Nat) : Nat := 10 * max a b + min a b

def is_largest_two_digit (n : Nat) : Prop :=
  ∃ (a b : Nat), a ∈ cards ∧ b ∈ cards ∧ a ≠ b ∧
  n = largest_two_digit a b ∧
  ∀ (x y : Nat), x ∈ cards → y ∈ cards → x ≠ y →
  largest_two_digit x y ≤ n

theorem unused_card_is_one :
  ∃ (n : Nat), is_largest_two_digit n ∧ (cards \ {n.div 10, n.mod 10}).toList = [1] := by
  sorry

end NUMINAMATH_CALUDE_unused_card_is_one_l1316_131646


namespace NUMINAMATH_CALUDE_coefficient_x3y5_in_expansion_of_x_plus_y_to_8_l1316_131672

theorem coefficient_x3y5_in_expansion_of_x_plus_y_to_8 :
  (Finset.range 9).sum (fun k => (Nat.choose 8 k : ℕ) * (if k = 3 then 1 else 0)) = 56 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x3y5_in_expansion_of_x_plus_y_to_8_l1316_131672


namespace NUMINAMATH_CALUDE_nineteenth_row_red_squares_l1316_131681

/-- Represents the number of squares in the nth row of a stair-step figure -/
def num_squares (n : ℕ) : ℕ := 3 * n - 1

/-- Represents the number of red squares in the nth row of a stair-step figure -/
def num_red_squares (n : ℕ) : ℕ := (num_squares n) / 2

theorem nineteenth_row_red_squares :
  num_red_squares 19 = 28 := by sorry

end NUMINAMATH_CALUDE_nineteenth_row_red_squares_l1316_131681


namespace NUMINAMATH_CALUDE_stool_height_is_80_l1316_131600

/-- The height of the stool Alice needs to reach the light bulb -/
def stool_height : ℝ :=
  let ceiling_height : ℝ := 300
  let light_bulb_below_ceiling : ℝ := 15
  let alice_height : ℝ := 150
  let alice_reach : ℝ := 50
  let decoration_below_bulb : ℝ := 5
  let light_bulb_height : ℝ := ceiling_height - light_bulb_below_ceiling
  let effective_reach_height : ℝ := light_bulb_height - decoration_below_bulb
  effective_reach_height - (alice_height + alice_reach)

theorem stool_height_is_80 :
  stool_height = 80 := by
  sorry

end NUMINAMATH_CALUDE_stool_height_is_80_l1316_131600


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1316_131684

theorem quadratic_equation_solution (c d : ℝ) (hc : c ≠ 0) (hd : d ≠ 0) :
  (∀ x : ℝ, x^2 + c*x + d = 0 ↔ x = c ∨ x = 2*d) →
  c = 1/2 ∧ d = -1/2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1316_131684


namespace NUMINAMATH_CALUDE_mildred_weight_l1316_131608

/-- Mildred's weight problem -/
theorem mildred_weight (carol_weight : ℕ) (weight_difference : ℕ) 
  (h1 : carol_weight = 9)
  (h2 : weight_difference = 50) :
  carol_weight + weight_difference = 59 := by
  sorry

end NUMINAMATH_CALUDE_mildred_weight_l1316_131608


namespace NUMINAMATH_CALUDE_triangle_inequality_l1316_131685

-- Define a triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the area of a triangle
def area (t : Triangle) : ℝ := sorry

-- Define the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem triangle_inequality (ABC : Triangle) (P : ℝ × ℝ) :
  let S := area ABC
  distance P ABC.A + distance P ABC.B + distance P ABC.C ≥ 2 * (3 ^ (1/4)) * Real.sqrt S := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1316_131685


namespace NUMINAMATH_CALUDE_unique_zero_of_exp_plus_linear_l1316_131618

/-- The function f(x) = e^x + 3x has exactly one zero. -/
theorem unique_zero_of_exp_plus_linear : ∃! x : ℝ, Real.exp x + 3 * x = 0 := by sorry

end NUMINAMATH_CALUDE_unique_zero_of_exp_plus_linear_l1316_131618


namespace NUMINAMATH_CALUDE_greatest_power_of_two_factor_l1316_131641

theorem greatest_power_of_two_factor (n : ℕ) : 
  ∃ (k : ℕ), 2^k ∣ (10^1000 - 4^500) ∧ 
  ∀ (m : ℕ), 2^m ∣ (10^1000 - 4^500) → m ≤ k := by
  sorry

end NUMINAMATH_CALUDE_greatest_power_of_two_factor_l1316_131641


namespace NUMINAMATH_CALUDE_minimal_sum_roots_and_qtilde_value_l1316_131679

/-- Represents a quadratic polynomial q(x) = x^2 - (a+b)x + ab -/
def QuadPoly (a b : ℝ) (x : ℝ) : ℝ :=
  x^2 - (a + b) * x + a * b

/-- The condition that q(q(x)) = 0 has exactly three real solutions -/
def HasThreeSolutions (a b : ℝ) : Prop :=
  ∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    QuadPoly a b (QuadPoly a b x) = 0 ∧
    QuadPoly a b (QuadPoly a b y) = 0 ∧
    QuadPoly a b (QuadPoly a b z) = 0 ∧
    ∀ w : ℝ, QuadPoly a b (QuadPoly a b w) = 0 → w = x ∨ w = y ∨ w = z

/-- The sum of roots of q(x) = 0 -/
def SumOfRoots (a b : ℝ) : ℝ := a + b

/-- The polynomial ̃q(x) = x^2 + 2x + 1 -/
def QTilde (x : ℝ) : ℝ := x^2 + 2 * x + 1

theorem minimal_sum_roots_and_qtilde_value :
  ∀ a b : ℝ,
  HasThreeSolutions a b →
  (∀ c d : ℝ, HasThreeSolutions c d → SumOfRoots a b ≤ SumOfRoots c d) →
  (QuadPoly a b = QTilde) ∧ QTilde 2 = 9 := by sorry

end NUMINAMATH_CALUDE_minimal_sum_roots_and_qtilde_value_l1316_131679


namespace NUMINAMATH_CALUDE_cos_double_angle_fourth_quadrant_l1316_131607

/-- Prove that for an angle in the fourth quadrant, if the sum of coordinates of its terminal point on the unit circle is -1/3, then cos 2θ = -√17/9 -/
theorem cos_double_angle_fourth_quadrant (θ : ℝ) (x₀ y₀ : ℝ) :
  (π < θ ∧ θ < 2*π) →  -- θ is in the fourth quadrant
  x₀^2 + y₀^2 = 1 →    -- point (x₀, y₀) is on the unit circle
  x₀ = Real.cos θ →    -- x₀ is the cosine of θ
  y₀ = Real.sin θ →    -- y₀ is the sine of θ
  x₀ + y₀ = -1/3 →     -- sum of coordinates is -1/3
  Real.cos (2*θ) = -Real.sqrt 17 / 9 := by
sorry

end NUMINAMATH_CALUDE_cos_double_angle_fourth_quadrant_l1316_131607


namespace NUMINAMATH_CALUDE_birthday_gift_contribution_l1316_131605

theorem birthday_gift_contribution (total_cost boss_contribution num_remaining_employees : ℕ) 
  (h1 : total_cost = 100)
  (h2 : boss_contribution = 15)
  (h3 : num_remaining_employees = 5) :
  let todd_contribution := 2 * boss_contribution
  let remaining_cost := total_cost - todd_contribution - boss_contribution
  remaining_cost / num_remaining_employees = 11 := by
sorry

end NUMINAMATH_CALUDE_birthday_gift_contribution_l1316_131605


namespace NUMINAMATH_CALUDE_smallest_five_digit_multiple_l1316_131688

theorem smallest_five_digit_multiple : ∃ n : ℕ, 
  (n ≥ 10000 ∧ n < 100000) ∧  -- five-digit number
  n % 9 = 0 ∧                 -- multiple of 9
  n % 6 = 0 ∧                 -- multiple of 6
  n % 2 = 0 ∧                 -- multiple of 2
  (∀ m : ℕ, 
    (m ≥ 10000 ∧ m < 100000) ∧ 
    m % 9 = 0 ∧ 
    m % 6 = 0 ∧ 
    m % 2 = 0 → 
    n ≤ m) ∧
  n = 10008 := by
sorry

end NUMINAMATH_CALUDE_smallest_five_digit_multiple_l1316_131688


namespace NUMINAMATH_CALUDE_female_kittens_count_l1316_131640

theorem female_kittens_count (initial_cats : ℕ) (total_cats : ℕ) (male_kittens : ℕ) : 
  initial_cats = 2 → total_cats = 7 → male_kittens = 2 → 
  total_cats - initial_cats - male_kittens = 3 := by
sorry

end NUMINAMATH_CALUDE_female_kittens_count_l1316_131640


namespace NUMINAMATH_CALUDE_arithmetic_sequence_eighth_term_l1316_131610

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_eighth_term
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_first : a 1 = 1)
  (h_sum : a 3 + a 4 + a 5 + a 6 = 20) :
  a 8 = 9 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_eighth_term_l1316_131610


namespace NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l1316_131660

theorem infinite_geometric_series_first_term
  (r : ℝ) (S : ℝ) (h_r : r = 1/4) (h_S : S = 24) :
  let a := S * (1 - r)
  a = 18 := by
sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l1316_131660


namespace NUMINAMATH_CALUDE_min_value_sum_of_fractions_l1316_131619

theorem min_value_sum_of_fractions (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / b + b / c + c / a + a / c ≥ 4 ∧
  (a / b + b / c + c / a + a / c = 4 ↔ a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_of_fractions_l1316_131619


namespace NUMINAMATH_CALUDE_total_prank_combinations_l1316_131636

/-- The number of different combinations of people Tim could involve in the prank --/
def prank_combinations (day1 day2 day3 day4 day5 : ℕ) : ℕ :=
  day1 * day2 * day3 * day4 * day5

/-- Theorem stating the total number of different combinations for Tim's prank --/
theorem total_prank_combinations :
  prank_combinations 1 2 5 4 1 = 40 := by
  sorry

end NUMINAMATH_CALUDE_total_prank_combinations_l1316_131636


namespace NUMINAMATH_CALUDE_percentage_unsold_books_l1316_131606

def initial_stock : ℕ := 900
def monday_sales : ℕ := 75
def tuesday_sales : ℕ := 50
def wednesday_sales : ℕ := 64
def thursday_sales : ℕ := 78
def friday_sales : ℕ := 135

theorem percentage_unsold_books :
  let total_sales := monday_sales + tuesday_sales + wednesday_sales + thursday_sales + friday_sales
  let unsold_books := initial_stock - total_sales
  (unsold_books : ℚ) / initial_stock * 100 = 55.33 := by
  sorry

end NUMINAMATH_CALUDE_percentage_unsold_books_l1316_131606


namespace NUMINAMATH_CALUDE_point_on_line_value_l1316_131627

/-- A point lies on a line if it satisfies the line's equation -/
def PointOnLine (x₁ y₁ x₂ y₂ x y : ℝ) : Prop :=
  (y - y₁) * (x₂ - x₁) = (y₂ - y₁) * (x - x₁)

theorem point_on_line_value :
  ∀ x : ℝ, PointOnLine 1 4 4 1 x 8 → x = -3 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_value_l1316_131627


namespace NUMINAMATH_CALUDE_arithmetic_sequence_with_prime_factors_l1316_131632

theorem arithmetic_sequence_with_prime_factors
  (n d : ℕ) :
  ∃ (a : ℕ → ℕ),
    (∀ i ∈ Finset.range n, a i > 0) ∧
    (∀ i ∈ Finset.range (n - 1), a (i + 1) = a i + d) ∧
    (∀ i ∈ Finset.range n, ∃ p : ℕ, Prime p ∧ p ≥ i + 1 ∧ p ∣ a i) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_with_prime_factors_l1316_131632


namespace NUMINAMATH_CALUDE_billys_restaurant_bill_l1316_131689

/-- The total bill for a group at Billy's Restaurant -/
def total_bill (adults children meal_cost : ℕ) : ℕ :=
  (adults + children) * meal_cost

/-- Theorem: The total bill for a group of 2 adults and 5 children, 
    with each meal costing 3 dollars, is 21 dollars -/
theorem billys_restaurant_bill : total_bill 2 5 3 = 21 := by
  sorry

end NUMINAMATH_CALUDE_billys_restaurant_bill_l1316_131689


namespace NUMINAMATH_CALUDE_sin_ratio_minus_sqrt3_over_sin_l1316_131612

theorem sin_ratio_minus_sqrt3_over_sin : 
  (Real.sin (80 * π / 180)) / (Real.sin (20 * π / 180)) - 
  (Real.sqrt 3) / (2 * Real.sin (80 * π / 180)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_ratio_minus_sqrt3_over_sin_l1316_131612


namespace NUMINAMATH_CALUDE_lizard_to_gecko_ratio_l1316_131670

/-- Represents the number of bugs eaten by each animal -/
structure BugsEaten where
  gecko : ℕ
  lizard : ℕ
  frog : ℕ
  toad : ℕ

/-- Conditions of the bug-eating scenario -/
def bugEatingScenario (b : BugsEaten) : Prop :=
  b.gecko = 12 ∧
  b.frog = 3 * b.lizard ∧
  b.toad = (3 * b.lizard) + (3 * b.lizard) / 2 ∧
  b.gecko + b.lizard + b.frog + b.toad = 63

/-- The ratio of bugs eaten by the lizard to bugs eaten by the gecko is 1:2 -/
theorem lizard_to_gecko_ratio (b : BugsEaten) 
  (h : bugEatingScenario b) : b.lizard * 2 = b.gecko := by
  sorry

#check lizard_to_gecko_ratio

end NUMINAMATH_CALUDE_lizard_to_gecko_ratio_l1316_131670


namespace NUMINAMATH_CALUDE_parallelogram_on_circle_l1316_131669

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Check if a point is inside a circle -/
def isInside (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 < c.radius^2

/-- Check if a point is on a circle -/
def isOn (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

/-- Check if four points form a parallelogram -/
def isParallelogram (a b c d : ℝ × ℝ) : Prop :=
  (b.1 - a.1 = d.1 - c.1) ∧ (b.2 - a.2 = d.2 - c.2) ∧
  (c.1 - b.1 = a.1 - d.1) ∧ (c.2 - b.2 = a.2 - d.2)

theorem parallelogram_on_circle (ω : Circle) (A B : ℝ × ℝ) 
  (h_A : isInside ω A) (h_B : isOn ω B) :
  ∃ (C D : ℝ × ℝ), isOn ω C ∧ isOn ω D ∧ isParallelogram A B C D :=
sorry

end NUMINAMATH_CALUDE_parallelogram_on_circle_l1316_131669


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l1316_131653

theorem solve_exponential_equation :
  ∃ y : ℝ, (3 : ℝ) ^ (y + 3) = 81 ^ y ∧ y = 1 := by sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l1316_131653


namespace NUMINAMATH_CALUDE_k_range_l1316_131675

theorem k_range (x y k : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : ∀ x y, x > 0 → y > 0 → Real.sqrt x + 3 * Real.sqrt y < k * Real.sqrt (x + y)) : 
  k > Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_k_range_l1316_131675


namespace NUMINAMATH_CALUDE_exists_winning_strategy_l1316_131663

/-- Represents the color of a hat -/
inductive HatColor
| Black
| White

/-- Represents an agent's guess about their own hat color -/
def Guess := HatColor

/-- Represents a strategy function that takes the observed hat color and returns a guess -/
def Strategy := HatColor → Guess

/-- Represents the outcome of applying strategies to a pair of hat colors -/
def Outcome (c1 c2 : HatColor) (s1 s2 : Strategy) : Prop :=
  (s1 c2 = c1) ∨ (s2 c1 = c2)

/-- Theorem stating that there exists a pair of strategies that guarantees
    at least one correct guess for any combination of hat colors -/
theorem exists_winning_strategy :
  ∃ (s1 s2 : Strategy), ∀ (c1 c2 : HatColor), Outcome c1 c2 s1 s2 := by
  sorry

end NUMINAMATH_CALUDE_exists_winning_strategy_l1316_131663


namespace NUMINAMATH_CALUDE_smallest_factor_for_perfect_square_l1316_131691

theorem smallest_factor_for_perfect_square (n : ℕ) (h : n = 2 * 3^2 * 5^2 * 7) :
  ∃ (a : ℕ), a > 0 ∧ 
  (∀ (b : ℕ), b > 0 → ∃ (k : ℕ), n * b = k^2 → a ≤ b) ∧
  (∃ (k : ℕ), n * a = k^2) ∧
  a = 14 := by
sorry

end NUMINAMATH_CALUDE_smallest_factor_for_perfect_square_l1316_131691


namespace NUMINAMATH_CALUDE_angle_between_vectors_l1316_131651

def a : ℝ × ℝ := (2, -1)

theorem angle_between_vectors (b : ℝ × ℝ) (θ : ℝ) 
  (h1 : ‖b‖ = 2 * Real.sqrt 5)
  (h2 : (a.1 + b.1) * a.1 + (a.2 + b.2) * a.2 = 10)
  (h3 : θ = Real.arccos ((a.1 * b.1 + a.2 * b.2) / (‖a‖ * ‖b‖)))
  : θ = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_angle_between_vectors_l1316_131651


namespace NUMINAMATH_CALUDE_cube_volume_surface_area_l1316_131615

theorem cube_volume_surface_area (y : ℝ) : 
  (∃ (s : ℝ), s > 0 ∧ s^3 = 3*y ∧ 6*s^2 = 3*y^2/100) → y = 600 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_surface_area_l1316_131615


namespace NUMINAMATH_CALUDE_derivative_implies_power_l1316_131690

/-- Given a function f(x) = m * x^(m-n) where its derivative f'(x) = 8 * x^3,
    prove that m^n = 1/4 -/
theorem derivative_implies_power (m n : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = m * x^(m-n))
  (h2 : ∀ x, deriv f x = 8 * x^3) :
  m^n = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_derivative_implies_power_l1316_131690


namespace NUMINAMATH_CALUDE_distance_between_trees_l1316_131674

/-- Given a yard with trees planted at equal distances, calculate the distance between consecutive trees. -/
theorem distance_between_trees (yard_length : ℝ) (num_trees : ℕ) : 
  yard_length = 255 ∧ num_trees = 18 → 
  (yard_length / (num_trees - 1 : ℝ)) = 15 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_trees_l1316_131674


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l1316_131677

theorem expression_simplification_and_evaluation :
  let x : ℝ := Real.sqrt 3 - 2
  ((x - 1) / (x - 2) + (2 * x - 8) / (x^2 - 4)) / (x + 5) = Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l1316_131677


namespace NUMINAMATH_CALUDE_students_pets_difference_fourth_grade_classrooms_difference_l1316_131630

theorem students_pets_difference : ℕ → ℕ → ℕ → ℕ → ℕ
  | num_classrooms, students_per_class, rabbits_per_class, hamsters_per_class =>
    let total_students := num_classrooms * students_per_class
    let total_rabbits := num_classrooms * rabbits_per_class
    let total_hamsters := num_classrooms * hamsters_per_class
    let total_pets := total_rabbits + total_hamsters
    total_students - total_pets

theorem fourth_grade_classrooms_difference :
  students_pets_difference 5 20 2 1 = 85 := by
  sorry

end NUMINAMATH_CALUDE_students_pets_difference_fourth_grade_classrooms_difference_l1316_131630


namespace NUMINAMATH_CALUDE_rectangle_perimeter_equals_26_l1316_131613

/-- Given a triangle with sides 5, 12, and 13 units, and a rectangle with width 3 units
    and area equal to the triangle's area, the perimeter of the rectangle is 26 units. -/
theorem rectangle_perimeter_equals_26 (triangle_side1 triangle_side2 triangle_side3 : ℝ)
  (rectangle_width : ℝ) (h1 : triangle_side1 = 5)
  (h2 : triangle_side2 = 12) (h3 : triangle_side3 = 13) (h4 : rectangle_width = 3)
  (h5 : (1/2) * triangle_side1 * triangle_side2 = rectangle_width * (((1/2) * triangle_side1 * triangle_side2) / rectangle_width)) :
  2 * (rectangle_width + (((1/2) * triangle_side1 * triangle_side2) / rectangle_width)) = 26 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_equals_26_l1316_131613


namespace NUMINAMATH_CALUDE_arthurs_hamburgers_l1316_131639

/-- Given the prices and quantities of hamburgers and hot dogs purchased over two days,
    prove that Arthur bought 2 hamburgers on the second day. -/
theorem arthurs_hamburgers (H D x : ℚ) : 
  3 * H + 4 * D = 10 →  -- Day 1 purchase
  x * H + 3 * D = 7 →   -- Day 2 purchase
  D = 1 →               -- Price of a hot dog
  x = 2 := by            
  sorry

end NUMINAMATH_CALUDE_arthurs_hamburgers_l1316_131639


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1316_131673

theorem quadratic_inequality_solution_set (a : ℝ) (h : a > 0) :
  let solution_set := {x : ℝ | a * x^2 - (a + 2) * x + 2 ≥ 0}
  (a = 2 → solution_set = Set.univ) ∧
  (0 < a ∧ a < 2 → solution_set = Set.Iic 1 ∪ Set.Ici (2 / a)) ∧
  (a > 2 → solution_set = Set.Iic (2 / a) ∪ Set.Ici 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1316_131673


namespace NUMINAMATH_CALUDE_right_triangle_case1_right_triangle_case2_right_triangle_case3_l1316_131635

-- Define a right triangle
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angleA : ℝ
  angleB : ℝ
  angleC : ℝ
  right_angle : angleC = 90
  angle_sum : angleA + angleB + angleC = 180

-- Case 1
theorem right_triangle_case1 (t : RightTriangle) (h1 : t.angleB = 60) (h2 : t.a = 4) :
  t.b = 4 * Real.sqrt 3 ∧ t.c = 8 := by
  sorry

-- Case 2
theorem right_triangle_case2 (t : RightTriangle) (h1 : t.a = Real.sqrt 3 - 1) (h2 : t.b = 3 - Real.sqrt 3) :
  t.angleB = 60 ∧ t.angleA = 30 ∧ t.c = 2 * Real.sqrt 3 - 2 := by
  sorry

-- Case 3
theorem right_triangle_case3 (t : RightTriangle) (h1 : t.angleA = 60) (h2 : t.c = 2 + Real.sqrt 3) :
  t.angleB = 30 ∧ t.a = Real.sqrt 3 + 3/2 ∧ t.b = (2 + Real.sqrt 3)/2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_case1_right_triangle_case2_right_triangle_case3_l1316_131635


namespace NUMINAMATH_CALUDE_solution_set_for_specific_values_minimum_value_for_general_case_l1316_131604

-- Define the function f
def f (x a b : ℝ) : ℝ := |x - a| + |x + b|

-- Theorem for part (I)
theorem solution_set_for_specific_values (x : ℝ) :
  let a := 1
  let b := 2
  (f x a b ≤ 5) ↔ (x ∈ Set.Icc (-3) 2) :=
sorry

-- Theorem for part (II)
theorem minimum_value_for_general_case (x a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (hab : a + 4*b = 2*a*b) :
  f x a b ≥ 9/2 :=
sorry

end NUMINAMATH_CALUDE_solution_set_for_specific_values_minimum_value_for_general_case_l1316_131604


namespace NUMINAMATH_CALUDE_matrix_power_100_l1316_131629

def A : Matrix (Fin 2) (Fin 2) ℝ := !![1, 0; 2, 1]

theorem matrix_power_100 :
  A ^ 100 = !![1, 0; 200, 1] := by sorry

end NUMINAMATH_CALUDE_matrix_power_100_l1316_131629


namespace NUMINAMATH_CALUDE_water_heater_problem_l1316_131656

/-- Represents the capacity and current water level of a water heater -/
structure WaterHeater where
  capacity : ℚ
  fillRatio : ℚ

/-- Calculates the total amount of water in all water heaters -/
def totalWater (wallace catherine albert belinda : WaterHeater) : ℚ :=
  wallace.capacity * wallace.fillRatio +
  catherine.capacity * catherine.fillRatio +
  albert.capacity * albert.fillRatio - 5 +
  belinda.capacity * belinda.fillRatio

theorem water_heater_problem 
  (wallace catherine albert belinda : WaterHeater)
  (h1 : wallace.capacity = 2 * catherine.capacity)
  (h2 : albert.capacity = 3/2 * wallace.capacity)
  (h3 : wallace.capacity = 40)
  (h4 : wallace.fillRatio = 3/4)
  (h5 : albert.fillRatio = 2/3)
  (h6 : belinda.capacity = 1/2 * catherine.capacity)
  (h7 : belinda.fillRatio = 5/8)
  (h8 : catherine.fillRatio = 7/8) :
  totalWater wallace catherine albert belinda = 89 := by
  sorry


end NUMINAMATH_CALUDE_water_heater_problem_l1316_131656


namespace NUMINAMATH_CALUDE_animal_sightings_sum_l1316_131614

/-- The number of animal sightings in January -/
def january_sightings : ℕ := 26

/-- The number of animal sightings in February -/
def february_sightings : ℕ := 3 * january_sightings

/-- The number of animal sightings in March -/
def march_sightings : ℕ := february_sightings / 2

/-- The number of animal sightings in April -/
def april_sightings : ℕ := 2 * march_sightings

/-- The total number of animal sightings over the four months -/
def total_sightings : ℕ := january_sightings + february_sightings + march_sightings + april_sightings

theorem animal_sightings_sum : total_sightings = 221 := by
  sorry

end NUMINAMATH_CALUDE_animal_sightings_sum_l1316_131614


namespace NUMINAMATH_CALUDE_arithmetic_sequence_terms_l1316_131623

/-- An arithmetic sequence with first term 10, last term 140, and common difference 5 has 27 terms. -/
theorem arithmetic_sequence_terms : ∀ (a : ℕ → ℕ),
  (∀ n, a (n + 1) = a n + 5) →  -- arithmetic sequence with common difference 5
  a 1 = 10 →                    -- first term is 10
  (∃ m, a m = 140) →            -- last term is 140
  (∃ m, a m = 140 ∧ ∀ k, k > m → a k > 140) →  -- 140 is the last term not exceeding 140
  (∃ m, m = 27 ∧ a m = 140) :=  -- the sequence has exactly 27 terms
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_terms_l1316_131623


namespace NUMINAMATH_CALUDE_elena_garden_petals_l1316_131624

/-- The number of lilies in Elena's garden -/
def num_lilies : ℕ := 8

/-- The number of tulips in Elena's garden -/
def num_tulips : ℕ := 5

/-- The number of petals each lily has -/
def petals_per_lily : ℕ := 6

/-- The number of petals each tulip has -/
def petals_per_tulip : ℕ := 3

/-- The total number of flower petals in Elena's garden -/
def total_petals : ℕ := num_lilies * petals_per_lily + num_tulips * petals_per_tulip

theorem elena_garden_petals : total_petals = 63 := by
  sorry

end NUMINAMATH_CALUDE_elena_garden_petals_l1316_131624


namespace NUMINAMATH_CALUDE_log_ratio_squared_l1316_131649

theorem log_ratio_squared (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (hx_neq : x ≠ 1) (hy_neq : y ≠ 1)
  (h_log : Real.log x / Real.log 2 = Real.log 16 / Real.log y)
  (h_prod : x * y = 64) : 
  ((Real.log x - Real.log y) / Real.log 2)^2 = 20 := by
sorry

end NUMINAMATH_CALUDE_log_ratio_squared_l1316_131649


namespace NUMINAMATH_CALUDE_section_plane_angle_cosine_l1316_131617

/-- Regular hexagonal pyramid with given properties -/
structure HexagonalPyramid where
  -- Base side length
  a : ℝ
  -- Distance from apex to section plane
  d : ℝ
  -- Base is a regular hexagon
  is_regular_hexagon : a > 0
  -- Section plane properties
  section_plane_properties : True
  -- Given distance
  distance_constraint : d = 1
  -- Given base side length
  base_side_length : a = 2

/-- The angle between the section plane and the base plane -/
def section_angle (pyramid : HexagonalPyramid) : ℝ := sorry

/-- Theorem stating the cosine of the angle between the section plane and base plane -/
theorem section_plane_angle_cosine (pyramid : HexagonalPyramid) : 
  Real.cos (section_angle pyramid) = 3/4 := by sorry

end NUMINAMATH_CALUDE_section_plane_angle_cosine_l1316_131617


namespace NUMINAMATH_CALUDE_back_section_total_revenue_l1316_131654

/-- Calculates the total revenue from the back section of a concert arena --/
def back_section_revenue (capacity : ℕ) (regular_price : ℚ) (half_price : ℚ) : ℚ :=
  let regular_revenue := regular_price * capacity
  let half_price_tickets := capacity / 6
  let half_price_revenue := half_price * half_price_tickets
  regular_revenue + half_price_revenue

/-- Theorem stating the total revenue from the back section --/
theorem back_section_total_revenue :
  back_section_revenue 25000 55 27.5 = 1489565 := by
  sorry

#eval back_section_revenue 25000 55 27.5

end NUMINAMATH_CALUDE_back_section_total_revenue_l1316_131654


namespace NUMINAMATH_CALUDE_A_intersect_B_l1316_131676

def A : Set ℕ := {0, 1, 2}

def B : Set ℕ := {x | ∃ a ∈ A, x = 2 * a}

theorem A_intersect_B : A ∩ B = {0, 2} := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_l1316_131676


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1316_131645

theorem sufficient_not_necessary_condition (a : ℝ) : 
  (∀ a, a = -2 → |a| = 2) ∧ 
  (∃ a, |a| = 2 ∧ a ≠ -2) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1316_131645


namespace NUMINAMATH_CALUDE_shooting_probability_l1316_131622

theorem shooting_probability (p : ℝ) : 
  (p ≥ 0 ∧ p ≤ 1) →  -- Ensure p is a valid probability
  (1 - (1 - 1/2) * (1 - 2/3) * (1 - p) = 7/8) → 
  p = 1/4 := by
sorry

end NUMINAMATH_CALUDE_shooting_probability_l1316_131622


namespace NUMINAMATH_CALUDE_intersection_point_x_coordinate_l1316_131611

theorem intersection_point_x_coordinate (x y : ℝ) : 
  y = 3 * x + 4 ∧ 5 * x - y = 41 → x = 22.5 := by sorry

end NUMINAMATH_CALUDE_intersection_point_x_coordinate_l1316_131611


namespace NUMINAMATH_CALUDE_sqrt_2_pow_12_l1316_131652

theorem sqrt_2_pow_12 : Real.sqrt (2^12) = 64 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_2_pow_12_l1316_131652


namespace NUMINAMATH_CALUDE_refrigerator_loss_percentage_l1316_131625

/-- Represents the problem of calculating the loss percentage on a refrigerator. -/
def RefrigeratorLossProblem (refrigerator_cp mobile_cp : ℕ) (mobile_profit overall_profit : ℕ) : Prop :=
  let refrigerator_sp := refrigerator_cp + mobile_cp + overall_profit - (mobile_cp + mobile_cp * mobile_profit / 100)
  let loss_percentage := (refrigerator_cp - refrigerator_sp) * 100 / refrigerator_cp
  loss_percentage = 5

/-- The main theorem stating that given the problem conditions, the loss percentage on the refrigerator is 5%. -/
theorem refrigerator_loss_percentage :
  RefrigeratorLossProblem 15000 8000 10 50 := by
  sorry

end NUMINAMATH_CALUDE_refrigerator_loss_percentage_l1316_131625


namespace NUMINAMATH_CALUDE_marks_of_a_l1316_131621

theorem marks_of_a (a b c d e : ℝ) : 
  (a + b + c) / 3 = 48 →
  (a + b + c + d) / 4 = 47 →
  e = d + 3 →
  (b + c + d + e) / 4 = 48 →
  a = 43 := by
sorry

end NUMINAMATH_CALUDE_marks_of_a_l1316_131621


namespace NUMINAMATH_CALUDE_cubic_equation_solutions_mean_l1316_131696

theorem cubic_equation_solutions_mean (x : ℝ) : 
  x^3 + 2*x^2 - 8*x - 4 = 0 → 
  ∃ (s : Finset ℝ), (∀ y ∈ s, y^3 + 2*y^2 - 8*y - 4 = 0) ∧ 
                    (s.card = 3) ∧ 
                    ((s.sum id) / s.card = -2/3) :=
sorry

end NUMINAMATH_CALUDE_cubic_equation_solutions_mean_l1316_131696


namespace NUMINAMATH_CALUDE_x_range_l1316_131631

theorem x_range (x : ℝ) (h1 : 1 / x < 3) (h2 : 1 / x > -4) : x > 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_x_range_l1316_131631


namespace NUMINAMATH_CALUDE_number_of_basic_events_l1316_131699

/-- The number of ways to choose 2 items from a set of 3 items -/
def choose_two_from_three : ℕ := 3

/-- The set of interest groups -/
def interest_groups : Finset String := {"Mathematics", "Computer Science", "Model Aviation"}

/-- Xiao Ming must join exactly two groups -/
def join_two_groups (groups : Finset String) : Finset (Finset String) :=
  groups.powerset.filter (fun s => s.card = 2)

theorem number_of_basic_events :
  (join_two_groups interest_groups).card = choose_two_from_three := by sorry

end NUMINAMATH_CALUDE_number_of_basic_events_l1316_131699


namespace NUMINAMATH_CALUDE_stating_four_of_a_kind_hands_l1316_131602

/-- Represents the number of distinct values in a standard deck of cards. -/
def num_values : ℕ := 13

/-- Represents the number of distinct suits in a standard deck of cards. -/
def num_suits : ℕ := 4

/-- Represents the total number of cards in a standard deck. -/
def total_cards : ℕ := num_values * num_suits

/-- Represents the number of cards in a hand. -/
def hand_size : ℕ := 5

/-- 
Theorem stating that the number of 5-card hands containing four cards of the same value 
in a standard 52-card deck is equal to 624.
-/
theorem four_of_a_kind_hands : 
  (num_values : ℕ) * (total_cards - num_suits : ℕ) = 624 := by
  sorry


end NUMINAMATH_CALUDE_stating_four_of_a_kind_hands_l1316_131602


namespace NUMINAMATH_CALUDE_monotonic_quadratic_range_l1316_131642

def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 3

theorem monotonic_quadratic_range (a : ℝ) :
  (∀ x ∈ Set.Icc 2 3, Monotone (fun x => f a x)) →
  a ∈ Set.Iic 2 ∪ Set.Ici 3 :=
by sorry

end NUMINAMATH_CALUDE_monotonic_quadratic_range_l1316_131642


namespace NUMINAMATH_CALUDE_system_solution_ratio_l1316_131671

theorem system_solution_ratio :
  ∃ (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0),
    x + 10*y + 5*z = 0 ∧
    2*x + 5*y + 4*z = 0 ∧
    3*x + 6*y + 5*z = 0 ∧
    y*z / (x^2) = -3/49 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_ratio_l1316_131671
