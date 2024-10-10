import Mathlib

namespace intersection_collinearity_l48_4821

/-- A point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A line in 2D space -/
structure Line :=
  (a : ℝ) (b : ℝ) (c : ℝ)

/-- Quadrilateral ABCD -/
structure Quadrilateral :=
  (A B C D : Point)

/-- Check if three points are collinear -/
def collinear (P Q R : Point) : Prop :=
  (Q.y - P.y) * (R.x - P.x) = (R.y - P.y) * (Q.x - P.x)

/-- The main theorem -/
theorem intersection_collinearity 
  (ABCD : Quadrilateral) 
  (P Q : Point) 
  (l : Line) 
  (E F : Point) 
  (R S T : Point) :
  (∃ (l1 : Line), l1.a * ABCD.A.x + l1.b * ABCD.A.y + l1.c = 0 ∧ 
                  l1.a * ABCD.B.x + l1.b * ABCD.B.y + l1.c = 0 ∧ 
                  l1.a * P.x + l1.b * P.y + l1.c = 0) →  -- AB extended through P
  (∃ (l2 : Line), l2.a * ABCD.C.x + l2.b * ABCD.C.y + l2.c = 0 ∧ 
                  l2.a * ABCD.D.x + l2.b * ABCD.D.y + l2.c = 0 ∧ 
                  l2.a * P.x + l2.b * P.y + l2.c = 0) →  -- CD extended through P
  (∃ (l3 : Line), l3.a * ABCD.B.x + l3.b * ABCD.B.y + l3.c = 0 ∧ 
                  l3.a * ABCD.C.x + l3.b * ABCD.C.y + l3.c = 0 ∧ 
                  l3.a * Q.x + l3.b * Q.y + l3.c = 0) →  -- BC extended through Q
  (∃ (l4 : Line), l4.a * ABCD.A.x + l4.b * ABCD.A.y + l4.c = 0 ∧ 
                  l4.a * ABCD.D.x + l4.b * ABCD.D.y + l4.c = 0 ∧ 
                  l4.a * Q.x + l4.b * Q.y + l4.c = 0) →  -- AD extended through Q
  (l.a * P.x + l.b * P.y + l.c = 0) →  -- P is on line l
  (l.a * E.x + l.b * E.y + l.c = 0) →  -- E is on line l
  (l.a * F.x + l.b * F.y + l.c = 0) →  -- F is on line l
  (∃ (l5 l6 : Line), l5.a * ABCD.A.x + l5.b * ABCD.A.y + l5.c = 0 ∧ 
                     l5.a * ABCD.C.x + l5.b * ABCD.C.y + l5.c = 0 ∧ 
                     l6.a * ABCD.B.x + l6.b * ABCD.B.y + l6.c = 0 ∧ 
                     l6.a * ABCD.D.x + l6.b * ABCD.D.y + l6.c = 0 ∧ 
                     l5.a * R.x + l5.b * R.y + l5.c = 0 ∧ 
                     l6.a * R.x + l6.b * R.y + l6.c = 0) →  -- R is intersection of AC and BD
  (∃ (l7 l8 : Line), l7.a * ABCD.A.x + l7.b * ABCD.A.y + l7.c = 0 ∧ 
                     l7.a * E.x + l7.b * E.y + l7.c = 0 ∧ 
                     l8.a * ABCD.B.x + l8.b * ABCD.B.y + l8.c = 0 ∧ 
                     l8.a * F.x + l8.b * F.y + l8.c = 0 ∧ 
                     l7.a * S.x + l7.b * S.y + l7.c = 0 ∧ 
                     l8.a * S.x + l8.b * S.y + l8.c = 0) →  -- S is intersection of AE and BF
  (∃ (l9 l10 : Line), l9.a * ABCD.C.x + l9.b * ABCD.C.y + l9.c = 0 ∧ 
                      l9.a * F.x + l9.b * F.y + l9.c = 0 ∧ 
                      l10.a * ABCD.D.x + l10.b * ABCD.D.y + l10.c = 0 ∧ 
                      l10.a * E.x + l10.b * E.y + l10.c = 0 ∧ 
                      l9.a * T.x + l9.b * T.y + l9.c = 0 ∧ 
                      l10.a * T.x + l10.b * T.y + l10.c = 0) →  -- T is intersection of CF and DE
  collinear R S T ∧ collinear R S Q :=
by sorry

end intersection_collinearity_l48_4821


namespace real_numbers_greater_than_eight_is_set_real_numbers_greater_than_eight_definite_membership_real_numbers_greater_than_eight_fixed_standards_l48_4895

-- Define the property of being greater than 8
def GreaterThanEight (x : ℝ) : Prop := x > 8

-- Define the set of real numbers greater than 8
def RealNumbersGreaterThanEight : Set ℝ := {x : ℝ | GreaterThanEight x}

-- Theorem stating that RealNumbersGreaterThanEight is a well-defined set
theorem real_numbers_greater_than_eight_is_set :
  ∀ (x : ℝ), x ∈ RealNumbersGreaterThanEight ↔ GreaterThanEight x :=
by
  sorry

-- Theorem stating that RealNumbersGreaterThanEight has definite membership criteria
theorem real_numbers_greater_than_eight_definite_membership :
  ∀ (x : ℝ), Decidable (x ∈ RealNumbersGreaterThanEight) :=
by
  sorry

-- Theorem stating that RealNumbersGreaterThanEight has fixed standards for inclusion
theorem real_numbers_greater_than_eight_fixed_standards :
  ∀ (x y : ℝ), x > 8 ∧ y > 8 → (x ∈ RealNumbersGreaterThanEight ∧ y ∈ RealNumbersGreaterThanEight) :=
by
  sorry

end real_numbers_greater_than_eight_is_set_real_numbers_greater_than_eight_definite_membership_real_numbers_greater_than_eight_fixed_standards_l48_4895


namespace color_tv_price_l48_4864

theorem color_tv_price (x : ℝ) : 
  (1 + 0.4) * x * 0.8 - x = 144 → x = 1200 := by
  sorry

end color_tv_price_l48_4864


namespace negation_existential_square_plus_one_less_than_zero_l48_4805

theorem negation_existential_square_plus_one_less_than_zero :
  (¬ ∃ x : ℝ, x^2 + 1 < 0) ↔ (∀ x : ℝ, x^2 + 1 ≥ 0) := by sorry

end negation_existential_square_plus_one_less_than_zero_l48_4805


namespace lemonade_stand_profit_l48_4876

/-- Represents the profit calculation for a lemonade stand -/
theorem lemonade_stand_profit :
  ∀ (lemon_cost sugar_cost cup_cost : ℕ) 
    (price_per_cup cups_sold : ℕ),
  lemon_cost = 10 →
  sugar_cost = 5 →
  cup_cost = 3 →
  price_per_cup = 4 →
  cups_sold = 21 →
  cups_sold * price_per_cup - (lemon_cost + sugar_cost + cup_cost) = 66 := by
sorry

end lemonade_stand_profit_l48_4876


namespace largest_even_five_digit_number_with_square_and_cube_l48_4892

/-- A function that checks if a number is a perfect square --/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

/-- A function that checks if a number is a perfect cube --/
def is_perfect_cube (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m * m = n

/-- A function that returns the first three digits of a 5-digit number --/
def first_three_digits (n : ℕ) : ℕ :=
  n / 100

/-- A function that returns the last three digits of a 5-digit number --/
def last_three_digits (n : ℕ) : ℕ :=
  n % 1000

/-- Main theorem --/
theorem largest_even_five_digit_number_with_square_and_cube : 
  ∀ n : ℕ, 
    10000 ≤ n ∧ n < 100000 ∧  -- 5-digit number
    Even n ∧  -- even number
    is_perfect_square (first_three_digits n) ∧  -- first three digits form a perfect square
    is_perfect_cube (last_three_digits n)  -- last three digits form a perfect cube
    → n ≤ 62512 :=
by sorry

end largest_even_five_digit_number_with_square_and_cube_l48_4892


namespace partial_fraction_decomposition_l48_4815

theorem partial_fraction_decomposition (N₁ N₂ : ℚ) :
  (∀ x : ℚ, x ≠ 1 → x ≠ 3 →
    (56 * x - 14) / (x^2 - 4*x + 3) = N₁ / (x - 1) + N₂ / (x - 3)) →
  N₁ * N₂ = -1617 := by
sorry

end partial_fraction_decomposition_l48_4815


namespace obstacle_course_total_time_l48_4820

def first_part_minutes : ℕ := 7
def first_part_seconds : ℕ := 23
def second_part_seconds : ℕ := 73
def third_part_minutes : ℕ := 5
def third_part_seconds : ℕ := 58

def seconds_per_minute : ℕ := 60

theorem obstacle_course_total_time :
  (first_part_minutes * seconds_per_minute + first_part_seconds) +
  second_part_seconds +
  (third_part_minutes * seconds_per_minute + third_part_seconds) = 874 := by
  sorry

end obstacle_course_total_time_l48_4820


namespace wall_building_time_l48_4811

/-- The number of days required for a group of workers to build a wall, given:
  * The number of workers in the reference group
  * The length of the wall built by the reference group
  * The number of days taken by the reference group
  * The number of workers in the new group
  * The length of the wall to be built by the new group
-/
def days_required (
  ref_workers : ℕ
  ) (ref_length : ℕ
  ) (ref_days : ℕ
  ) (new_workers : ℕ
  ) (new_length : ℕ
  ) : ℚ :=
  (ref_workers * ref_days * new_length : ℚ) / (new_workers * ref_length)

/-- Theorem stating that 30 workers will take 18 days to build a 100m wall,
    given that 18 workers can build a 140m wall in 42 days -/
theorem wall_building_time :
  days_required 18 140 42 30 100 = 18 := by
  sorry

end wall_building_time_l48_4811


namespace arithmetic_sequence_shared_prime_factor_l48_4893

theorem arithmetic_sequence_shared_prime_factor (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  ∃ (p : ℕ) (hp : Prime p), ∀ n : ℕ, ∃ k ≥ n, p ∣ (a * k + b) :=
sorry

end arithmetic_sequence_shared_prime_factor_l48_4893


namespace min_moves_is_22_l48_4801

/-- A move consists of transferring one coin to an adjacent box. -/
def Move := ℕ

/-- The configuration of coins in the boxes. -/
def Configuration := Fin 7 → ℕ

/-- The initial configuration of coins in the boxes. -/
def initial_config : Configuration :=
  fun i => [5, 8, 11, 17, 20, 15, 10].get i

/-- A configuration is balanced if all boxes have the same number of coins. -/
def is_balanced (c : Configuration) : Prop :=
  ∀ i j : Fin 7, c i = c j

/-- The number of moves required to transform one configuration into another. -/
def moves_required (start finish : Configuration) : ℕ := sorry

/-- The minimum number of moves required to balance the configuration. -/
def min_moves_to_balance (c : Configuration) : ℕ := sorry

/-- The theorem stating that the minimum number of moves required to balance
    the initial configuration is 22. -/
theorem min_moves_is_22 :
  min_moves_to_balance initial_config = 22 := by sorry

end min_moves_is_22_l48_4801


namespace proportional_enlargement_l48_4813

/-- Given a rectangle that is enlarged proportionally, this theorem proves
    that the new height can be calculated from the original dimensions
    and the new width. -/
theorem proportional_enlargement
  (original_width original_height new_width : ℝ)
  (h_positive : original_width > 0 ∧ original_height > 0 ∧ new_width > 0)
  (h_original_width : original_width = 2)
  (h_original_height : original_height = 1.5)
  (h_new_width : new_width = 8) :
  let new_height := original_height * (new_width / original_width)
  new_height = 6 := by
sorry

end proportional_enlargement_l48_4813


namespace product_of_four_integers_l48_4869

theorem product_of_four_integers (P Q R S : ℕ+) : 
  P + Q + R + S = 100 →
  (P : ℚ) + 5 = (Q : ℚ) - 5 →
  (P : ℚ) + 5 = (R : ℚ) * 2 →
  (P : ℚ) + 5 = (S : ℚ) / 2 →
  (P : ℚ) * (Q : ℚ) * (R : ℚ) * (S : ℚ) = 1509400000 / 6561 := by
  sorry

end product_of_four_integers_l48_4869


namespace least_exponent_sum_for_2023_l48_4881

def is_sum_of_distinct_powers_of_two (n : ℕ) (exponents : List ℕ) : Prop :=
  n = (exponents.map (fun e => 2^e)).sum ∧ exponents.Nodup

theorem least_exponent_sum_for_2023 :
  ∃ (exponents : List ℕ),
    is_sum_of_distinct_powers_of_two 2023 exponents ∧
    exponents.sum = 48 ∧
    ∀ (other_exponents : List ℕ),
      is_sum_of_distinct_powers_of_two 2023 other_exponents →
      other_exponents.sum ≥ 48 :=
sorry

end least_exponent_sum_for_2023_l48_4881


namespace total_stones_l48_4899

/-- Represents the number of stones in each pile -/
structure StonePiles where
  pile1 : ℕ
  pile2 : ℕ
  pile3 : ℕ
  pile4 : ℕ
  pile5 : ℕ

/-- Defines the conditions for the stone piles -/
def ValidStonePiles (p : StonePiles) : Prop :=
  p.pile5 = 6 * p.pile3 ∧
  p.pile2 = 2 * (p.pile3 + p.pile5) ∧
  p.pile1 = p.pile5 / 3 ∧
  p.pile1 = p.pile4 - 10 ∧
  p.pile4 = 2 * p.pile2

/-- The theorem to be proved -/
theorem total_stones (p : StonePiles) (h : ValidStonePiles p) : 
  p.pile1 + p.pile2 + p.pile3 + p.pile4 + p.pile5 = 60 := by
  sorry

end total_stones_l48_4899


namespace line_bisecting_segment_l48_4803

/-- The equation of a line passing through a point and bisecting a segment between two other lines -/
theorem line_bisecting_segment (M : ℝ × ℝ) (l₁ l₂ : ℝ → ℝ → ℝ) :
  M = (3/2, -1/2) →
  (∀ x y, l₁ x y = 2*x - 5*y + 10) →
  (∀ x y, l₂ x y = 3*x + 8*y + 15) →
  ∃ P₁ P₂ : ℝ × ℝ,
    l₁ P₁.1 P₁.2 = 0 ∧
    l₂ P₂.1 P₂.2 = 0 ∧
    M = ((P₁.1 + P₂.1) / 2, (P₁.2 + P₂.2) / 2) →
  ∃ A B C : ℝ,
    A = 5 ∧ B = 3 ∧ C = -6 ∧
    ∀ x y, A*x + B*y + C = 0 ↔ (y - M.2) / (x - M.1) = -A / B :=
by sorry

end line_bisecting_segment_l48_4803


namespace ellipse_line_intersection_l48_4858

/-- An ellipse with given properties and a line intersecting it -/
structure EllipseWithLine where
  a : ℝ
  b : ℝ
  k : ℝ
  h_a_pos : 0 < a
  h_b_pos : 0 < b
  h_a_gt_b : b < a
  h_right_focus : (1 : ℝ) = a * (a^2 - b^2).sqrt / a
  h_eccentricity : (a^2 - b^2).sqrt / a = 1/2
  h_ellipse_eq : ∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 ↔ (x, y) ∈ {p : ℝ × ℝ | p.1^2/a^2 + p.2^2/b^2 = 1}
  h_line_eq : ∀ x : ℝ, (x, k*x + 1) ∈ {p : ℝ × ℝ | p.1^2/a^2 + p.2^2/b^2 = 1}
  h_intersect : ∃ A B : ℝ × ℝ, A ∈ {p : ℝ × ℝ | p.1^2/a^2 + p.2^2/b^2 = 1} ∧ 
                                B ∈ {p : ℝ × ℝ | p.1^2/a^2 + p.2^2/b^2 = 1} ∧
                                A.2 = k * A.1 + 1 ∧ B.2 = k * B.1 + 1 ∧ A ≠ B
  h_midpoints : ∀ A B : ℝ × ℝ, A ∈ {p : ℝ × ℝ | p.1^2/a^2 + p.2^2/b^2 = 1} ∧ 
                                B ∈ {p : ℝ × ℝ | p.1^2/a^2 + p.2^2/b^2 = 1} ∧
                                A.2 = k * A.1 + 1 ∧ B.2 = k * B.1 + 1 ∧ A ≠ B →
                ∃ M N : ℝ × ℝ, M = ((A.1 + 1)/2, A.2/2) ∧ N = ((B.1 + 1)/2, B.2/2)
  h_origin_on_circle : ∀ M N : ℝ × ℝ, M.1 * N.1 + M.2 * N.2 = 0

/-- The main theorem: given the ellipse and line with specified properties, k = -1/2 -/
theorem ellipse_line_intersection (e : EllipseWithLine) : e.k = -1/2 :=
sorry

end ellipse_line_intersection_l48_4858


namespace limit_x_minus_pi_half_times_tan_x_approaches_pi_half_l48_4832

/-- The limit of (x - π/2) * tan(x) as x approaches π/2 is -1. -/
theorem limit_x_minus_pi_half_times_tan_x_approaches_pi_half :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x - π/2| ∧ |x - π/2| < δ →
    |(x - π/2) * Real.tan x + 1| < ε :=
sorry

end limit_x_minus_pi_half_times_tan_x_approaches_pi_half_l48_4832


namespace circle_center_trajectory_l48_4808

/-- A moving circle with center (x, y) passes through (1, 0) and is tangent to x = -1 -/
def MovingCircle (x y : ℝ) : Prop :=
  (x - 1)^2 + y^2 = (x + 1)^2

/-- The trajectory of the circle's center satisfies y^2 = 4x -/
theorem circle_center_trajectory (x y : ℝ) :
  MovingCircle x y → y^2 = 4*x := by
  sorry

end circle_center_trajectory_l48_4808


namespace quadratic_inequality_empty_solution_l48_4861

theorem quadratic_inequality_empty_solution (a : ℝ) :
  (∀ x : ℝ, x^2 + a*x + 4 ≥ 0) → a ∈ Set.Icc (-4 : ℝ) 4 := by
  sorry

end quadratic_inequality_empty_solution_l48_4861


namespace noah_jelly_beans_l48_4840

-- Define the total number of jelly beans
def total_jelly_beans : ℝ := 600

-- Define the percentages for Thomas and Sarah
def thomas_percentage : ℝ := 0.06
def sarah_percentage : ℝ := 0.10

-- Define the ratio for Barry, Emmanuel, and Miguel
def barry_ratio : ℝ := 4
def emmanuel_ratio : ℝ := 5
def miguel_ratio : ℝ := 6

-- Define the percentages for Chloe and Noah
def chloe_percentage : ℝ := 0.40
def noah_percentage : ℝ := 0.30

-- Theorem to prove
theorem noah_jelly_beans :
  let thomas_share := total_jelly_beans * thomas_percentage
  let sarah_share := total_jelly_beans * sarah_percentage
  let remaining_jelly_beans := total_jelly_beans - (thomas_share + sarah_share)
  let total_ratio := barry_ratio + emmanuel_ratio + miguel_ratio
  let emmanuel_share := (emmanuel_ratio / total_ratio) * remaining_jelly_beans
  let noah_share := emmanuel_share * noah_percentage
  noah_share = 50.4 := by
  sorry

end noah_jelly_beans_l48_4840


namespace size_relationship_l48_4844

theorem size_relationship : 
  let a : ℝ := 1 + Real.sqrt 7
  let b : ℝ := Real.sqrt 3 + Real.sqrt 5
  let c : ℝ := 4
  a < b ∧ b < c := by sorry

end size_relationship_l48_4844


namespace correct_propositions_l48_4878

-- Define the type for propositions
inductive Proposition
  | one
  | two
  | three
  | four
  | five
  | six
  | seven

-- Define a function to check if a proposition is correct
def is_correct (p : Proposition) : Prop :=
  match p with
  | .two => True
  | .six => True
  | .seven => True
  | _ => False

-- Define the theorem
theorem correct_propositions :
  ∀ p : Proposition, is_correct p ↔ (p = .two ∨ p = .six ∨ p = .seven) :=
by sorry

end correct_propositions_l48_4878


namespace square_diagonal_side_area_l48_4865

/-- Given a square with diagonal length 4, prove its side length and area. -/
theorem square_diagonal_side_area :
  ∃ (side_length area : ℝ),
    4^2 = 2 * side_length^2 ∧
    side_length = 2 * Real.sqrt 2 ∧
    area = 8 := by
  sorry

end square_diagonal_side_area_l48_4865


namespace imaginary_part_of_complex_fraction_l48_4852

theorem imaginary_part_of_complex_fraction : 
  let z : ℂ := 2 * Complex.I / (3 - 2 * Complex.I)
  Complex.im z = 6 / 13 := by
sorry

end imaginary_part_of_complex_fraction_l48_4852


namespace conditional_probability_B_given_A_l48_4884

/-- The total number of zongzi -/
def total_zongzi : ℕ := 5

/-- The number of zongzi with pork filling -/
def pork_zongzi : ℕ := 2

/-- The number of zongzi with red bean paste filling -/
def red_bean_zongzi : ℕ := 3

/-- Event A: the two picked zongzi have the same filling -/
def event_A : Set (Fin total_zongzi × Fin total_zongzi) := sorry

/-- Event B: the two picked zongzi both have red bean paste filling -/
def event_B : Set (Fin total_zongzi × Fin total_zongzi) := sorry

/-- The probability measure on the sample space -/
def P : Set (Fin total_zongzi × Fin total_zongzi) → ℝ := sorry

theorem conditional_probability_B_given_A :
  P event_B / P event_A = 3/4 := by sorry

end conditional_probability_B_given_A_l48_4884


namespace book_arrangement_theorem_l48_4806

/-- The number of ways to arrange books on a shelf -/
def arrange_books (math_books : ℕ) (english_books : ℕ) (science_books : ℕ) : ℕ :=
  (Nat.factorial 3) * (Nat.factorial math_books) * (Nat.factorial english_books) * (Nat.factorial science_books)

/-- Theorem: The number of ways to arrange 4 math books, 6 English books, and 2 science books
    on a shelf, where all books of the same subject must stay together and the books within
    each subject are different, is equal to 207360. -/
theorem book_arrangement_theorem :
  arrange_books 4 6 2 = 207360 := by
  sorry

end book_arrangement_theorem_l48_4806


namespace stratified_sampling_male_athletes_l48_4837

theorem stratified_sampling_male_athletes 
  (total_male : ℕ) 
  (total_female : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_male = 48) 
  (h2 : total_female = 36) 
  (h3 : sample_size = 21) : 
  ℕ :=
  12

#check stratified_sampling_male_athletes

end stratified_sampling_male_athletes_l48_4837


namespace sector_angle_l48_4866

/-- Given a circular sector with arc length 4 and area 4, 
    prove that the absolute value of its central angle in radians is 2. -/
theorem sector_angle (r : ℝ) (θ : ℝ) (h1 : r * θ = 4) (h2 : (1/2) * r^2 * θ = 4) : 
  |θ| = 2 := by
sorry

end sector_angle_l48_4866


namespace exchange_process_duration_l48_4845

/-- Represents the number of children of each gender -/
def n : ℕ := 10

/-- Calculates the sum of the first n even numbers -/
def sum_even (n : ℕ) : ℕ := n * (n + 1)

/-- Calculates the sum of the first n natural numbers -/
def sum_natural (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Represents the total number of swaps required to move boys from even positions to the first n positions -/
def total_swaps (n : ℕ) : ℕ := sum_even n - sum_natural n

theorem exchange_process_duration :
  total_swaps n = 55 ∧ total_swaps n < 60 := by sorry

end exchange_process_duration_l48_4845


namespace quadratic_inequality_l48_4831

theorem quadratic_inequality (x : ℝ) : -x^2 - 2*x + 3 ≥ 0 ↔ -3 ≤ x ∧ x ≤ 1 := by
  sorry

end quadratic_inequality_l48_4831


namespace sufficient_but_not_necessary_l48_4819

theorem sufficient_but_not_necessary (x : ℝ) : 
  (∀ x, x > 0 → x^2 + 4*x + 3 > 0) ∧ 
  (∃ x, x^2 + 4*x + 3 > 0 ∧ ¬(x > 0)) := by
sorry

end sufficient_but_not_necessary_l48_4819


namespace students_per_group_l48_4838

theorem students_per_group 
  (total_students : ℕ) 
  (num_groups : ℕ) 
  (h1 : total_students = 30) 
  (h2 : num_groups = 6) 
  (h3 : total_students % num_groups = 0) :
  total_students / num_groups = 5 := by
sorry

end students_per_group_l48_4838


namespace arithmetic_sequence_sum_l48_4870

-- Define arithmetic sequences a_n and b_n
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

-- Define the problem statement
theorem arithmetic_sequence_sum (a b : ℕ → ℝ) :
  arithmetic_sequence a →
  arithmetic_sequence b →
  a 1 = 25 →
  b 1 = 75 →
  a 2 + b 2 = 100 →
  a 37 + b 37 = 100 := by
sorry

end arithmetic_sequence_sum_l48_4870


namespace min_red_chips_is_72_l48_4854

/-- Represents the number of chips of each color in the box -/
structure ChipCounts where
  white : ℕ
  blue : ℕ
  red : ℕ

/-- Checks if the chip counts satisfy the given conditions -/
def valid_counts (c : ChipCounts) : Prop :=
  c.blue ≥ c.white / 3 ∧
  c.blue ≤ c.red / 4 ∧
  c.white + c.blue ≥ 72

/-- The minimum number of red chips required -/
def min_red_chips : ℕ := 72

/-- Theorem stating that the minimum number of red chips is 72 -/
theorem min_red_chips_is_72 :
  ∀ c : ChipCounts, valid_counts c → c.red ≥ min_red_chips :=
by sorry

end min_red_chips_is_72_l48_4854


namespace famous_artists_not_set_l48_4816

/-- A structure representing a collection of objects -/
structure Collection where
  elements : Set α
  is_definite : Bool
  is_distinct : Bool
  is_unordered : Bool

/-- Definition of a set -/
def is_set (c : Collection) : Prop :=
  c.is_definite ∧ c.is_distinct ∧ c.is_unordered

/-- Famous artists collection -/
def famous_artists : Collection := sorry

/-- Theorem stating that famous artists cannot form a set -/
theorem famous_artists_not_set : ¬(is_set famous_artists) := by
  sorry

end famous_artists_not_set_l48_4816


namespace apple_selling_price_l48_4830

/-- Calculates the selling price of an apple given its cost price and loss fraction. -/
def selling_price (cost_price : ℝ) (loss_fraction : ℝ) : ℝ :=
  cost_price * (1 - loss_fraction)

/-- Theorem stating the selling price of an apple given specific conditions. -/
theorem apple_selling_price :
  let cost_price : ℝ := 19
  let loss_fraction : ℝ := 1/6
  let calculated_price := selling_price cost_price loss_fraction
  ∃ ε > 0, |calculated_price - 15.83| < ε :=
by
  sorry

end apple_selling_price_l48_4830


namespace min_sum_m_n_l48_4842

theorem min_sum_m_n : ∃ (m n : ℕ+), 
  108 * (m : ℕ) = (n : ℕ)^3 ∧ 
  (∀ (m' n' : ℕ+), 108 * (m' : ℕ) = (n' : ℕ)^3 → (m : ℕ) + (n : ℕ) ≤ (m' : ℕ) + (n' : ℕ)) ∧
  (m : ℕ) + (n : ℕ) = 8 := by
  sorry

end min_sum_m_n_l48_4842


namespace tiles_in_row_l48_4804

/-- Given a rectangular room with area 144 sq ft and length twice the width,
    prove that 25 tiles of size 4 inches by 4 inches fit in a row along the width. -/
theorem tiles_in_row (room_area : ℝ) (tile_size : ℝ) : 
  room_area = 144 →
  tile_size = 4 →
  ⌊(12 * (144 / 2).sqrt) / tile_size⌋ = 25 := by sorry

end tiles_in_row_l48_4804


namespace basketball_games_total_l48_4877

theorem basketball_games_total (games_won games_lost : ℕ) : 
  games_won - games_lost = 28 → games_won = 45 → games_lost = 17 → 
  games_won + games_lost = 62 := by
  sorry

end basketball_games_total_l48_4877


namespace sugar_recipe_reduction_l48_4860

theorem sugar_recipe_reduction : 
  let full_recipe : ℚ := 5 + 3/4
  let reduced_recipe : ℚ := full_recipe / 3
  reduced_recipe = 1 + 11/12 := by sorry

end sugar_recipe_reduction_l48_4860


namespace sevenPeopleRoundTable_l48_4809

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def seatingArrangements (totalPeople : ℕ) (adjacentPair : ℕ) : ℕ :=
  if totalPeople ≤ 1 then 0
  else
    let effectiveUnits := totalPeople - adjacentPair + 1
    (factorial effectiveUnits * adjacentPair) / totalPeople

theorem sevenPeopleRoundTable :
  seatingArrangements 7 2 = 240 := by
  sorry

end sevenPeopleRoundTable_l48_4809


namespace imaginary_part_of_i_over_one_minus_i_l48_4868

/-- The imaginary part of the complex number i / (1 - i) is 1/2 -/
theorem imaginary_part_of_i_over_one_minus_i : Complex.im (Complex.I / (1 - Complex.I)) = 1 / 2 := by
  sorry

end imaginary_part_of_i_over_one_minus_i_l48_4868


namespace crazy_silly_school_unwatched_movies_l48_4874

/-- Given a total number of movies and the number of watched movies,
    calculate the number of unwatched movies -/
def unwatched_movies (total : ℕ) (watched : ℕ) : ℕ :=
  total - watched

/-- Theorem: In the 'crazy silly school' series, with 8 total movies
    and 4 watched movies, there are 4 unwatched movies -/
theorem crazy_silly_school_unwatched_movies :
  unwatched_movies 8 4 = 4 := by
  sorry

end crazy_silly_school_unwatched_movies_l48_4874


namespace tshirt_cost_l48_4887

def amusement_park_problem (initial_amount ticket_cost food_cost remaining_amount : ℕ) : Prop :=
  let total_spent := ticket_cost + food_cost + (initial_amount - ticket_cost - food_cost - remaining_amount)
  total_spent = initial_amount - remaining_amount

theorem tshirt_cost (initial_amount ticket_cost food_cost remaining_amount : ℕ) 
  (h1 : initial_amount = 75)
  (h2 : ticket_cost = 30)
  (h3 : food_cost = 13)
  (h4 : remaining_amount = 9)
  (h5 : amusement_park_problem initial_amount ticket_cost food_cost remaining_amount) :
  initial_amount - ticket_cost - food_cost - remaining_amount = 23 := by
  sorry

end tshirt_cost_l48_4887


namespace fraction_equality_l48_4818

theorem fraction_equality (x y : ℝ) (h : x / y = 2 / 5) : 
  ((x + 3 * y) / (2 * y) ≠ 13 / 10) ∧ 
  ((2 * x) / (y - x) = 4 / 3) ∧ 
  ((x + 5 * y) / (2 * x) = 27 / 4) ∧ 
  ((2 * y - x) / (3 * y) ≠ 7 / 15) ∧ 
  (y / (3 * x) = 5 / 6) := by
  sorry

end fraction_equality_l48_4818


namespace function_minimum_implies_inequality_l48_4882

/-- Given a function f(x) = ax^2 + bx - ln(x) where a, b ∈ ℝ,
    if a > 0 and for any x > 0, f(x) ≥ f(1), then ln(a) < -2b -/
theorem function_minimum_implies_inequality (a b : ℝ) :
  a > 0 →
  (∀ x > 0, a * x^2 + b * x - Real.log x ≥ a + b) →
  Real.log a < -2 * b :=
by sorry

end function_minimum_implies_inequality_l48_4882


namespace min_value_theorem_l48_4859

theorem min_value_theorem (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hsum : x + y + z = 9) :
  (x^2 + y^2)/(x + y) + (x^2 + z^2)/(x + z) + (y^2 + z^2)/(y + z) ≥ 9 ∧
  ((x^2 + y^2)/(x + y) + (x^2 + z^2)/(x + z) + (y^2 + z^2)/(y + z) = 9 ↔ x = 3 ∧ y = 3 ∧ z = 3) :=
by sorry

end min_value_theorem_l48_4859


namespace polynomial_division_proof_l48_4800

theorem polynomial_division_proof (z : ℝ) : 
  ((4/3 : ℝ) * z^4 - (17/9 : ℝ) * z^3 + (56/27 : ℝ) * z^2 - (167/81 : ℝ) * z + 500/243) * (3 * z + 1) = 
  4 * z^5 - 5 * z^4 + 7 * z^3 - 15 * z^2 + 9 * z - 3 :=
by sorry

end polynomial_division_proof_l48_4800


namespace eg_length_l48_4839

-- Define the quadrilateral EFGH
structure Quadrilateral :=
  (E F G H : ℝ × ℝ)

-- Define the properties of the quadrilateral
def is_valid_quadrilateral (q : Quadrilateral) : Prop :=
  let (ex, ey) := q.E
  let (fx, fy) := q.F
  let (gx, gy) := q.G
  let (hx, hy) := q.H
  -- EF = 6
  (ex - fx)^2 + (ey - fy)^2 = 36 ∧
  -- FG = 18
  (fx - gx)^2 + (fy - gy)^2 = 324 ∧
  -- GH = 6
  (gx - hx)^2 + (gy - hy)^2 = 36 ∧
  -- HE = 10
  (hx - ex)^2 + (hy - ey)^2 = 100 ∧
  -- Angle EFG is a right angle
  (ex - fx) * (gx - fx) + (ey - fy) * (gy - fy) = 0

-- Theorem statement
theorem eg_length (q : Quadrilateral) (h : is_valid_quadrilateral q) :
  let (ex, ey) := q.E
  let (gx, gy) := q.G
  (ex - gx)^2 + (ey - gy)^2 = 360 := by
  sorry

end eg_length_l48_4839


namespace option2_higher_expectation_l48_4875

/-- Represents the number of red and white balls in the box -/
structure BallCount where
  red : ℕ
  white : ℕ

/-- Represents the two lottery options -/
inductive LotteryOption
  | Option1
  | Option2

/-- Calculates the expected value for Option 1 -/
def expectedValueOption1 (initial : BallCount) : ℚ :=
  sorry

/-- Calculates the expected value for Option 2 -/
def expectedValueOption2 (initial : BallCount) : ℚ :=
  sorry

/-- Theorem stating that Option 2 has a higher expected value -/
theorem option2_higher_expectation (initial : BallCount) :
  initial.red = 3 ∧ initial.white = 3 →
  expectedValueOption2 initial > expectedValueOption1 initial :=
sorry

end option2_higher_expectation_l48_4875


namespace polynomial_remainder_l48_4829

/-- Given a polynomial q(x) = Ax^6 + Bx^4 + Cx^2 + 10, if the remainder when
    q(x) is divided by x - 2 is 20, then the remainder when q(x) is divided
    by x + 2 is also 20. -/
theorem polynomial_remainder (A B C : ℝ) : 
  let q : ℝ → ℝ := λ x ↦ A * x^6 + B * x^4 + C * x^2 + 10
  (q 2 = 20) → (q (-2) = 20) := by
  sorry

end polynomial_remainder_l48_4829


namespace pascal_high_school_students_l48_4863

/-- The number of students at Pascal High School -/
def total_students : ℕ := sorry

/-- The number of students who went on the first trip -/
def first_trip : ℕ := sorry

/-- The number of students who went on the second trip -/
def second_trip : ℕ := sorry

/-- The number of students who went on the third trip -/
def third_trip : ℕ := sorry

/-- The number of students who went on all three trips -/
def all_three_trips : ℕ := 160

theorem pascal_high_school_students :
  (first_trip = total_students / 2) ∧
  (second_trip = (total_students * 4) / 5) ∧
  (third_trip = (total_students * 9) / 10) ∧
  (all_three_trips = 160) ∧
  (∀ s, s ∈ Finset.range total_students →
    (s ∈ Finset.range first_trip ∧ s ∈ Finset.range second_trip) ∨
    (s ∈ Finset.range first_trip ∧ s ∈ Finset.range third_trip) ∨
    (s ∈ Finset.range second_trip ∧ s ∈ Finset.range third_trip) ∨
    (s ∈ Finset.range all_three_trips)) →
  total_students = 800 := by sorry

end pascal_high_school_students_l48_4863


namespace smallest_difference_for_8_factorial_l48_4828

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem smallest_difference_for_8_factorial :
  ∀ a b c : ℕ+,
  a * b * c = factorial 8 →
  a < b →
  b < c →
  ∀ a' b' c' : ℕ+,
  a' * b' * c' = factorial 8 →
  a' < b' →
  b' < c' →
  c - a ≤ c' - a' :=
sorry

end smallest_difference_for_8_factorial_l48_4828


namespace empty_solution_set_implies_positive_a_nonpositive_discriminant_l48_4873

/-- The discriminant of a quadratic equation ax^2 + bx + c = 0 -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- The solution set of a quadratic inequality ax^2 + bx + c < 0 -/
def solutionSet (a b c : ℝ) : Set ℝ := {x : ℝ | a*x^2 + b*x + c < 0}

theorem empty_solution_set_implies_positive_a_nonpositive_discriminant
  (a b c : ℝ) (h_a_nonzero : a ≠ 0) :
  IsEmpty (solutionSet a b c) → a > 0 ∧ discriminant a b c ≤ 0 :=
by sorry

end empty_solution_set_implies_positive_a_nonpositive_discriminant_l48_4873


namespace equation_with_positive_root_l48_4898

theorem equation_with_positive_root (x m : ℝ) : 
  ((x - 2) / (x + 1) = m / (x + 1) ∧ x > 0) → m = -3 :=
by sorry

end equation_with_positive_root_l48_4898


namespace kathys_candy_collection_l48_4890

theorem kathys_candy_collection (num_groups : ℕ) (candies_per_group : ℕ) 
  (h1 : num_groups = 10) (h2 : candies_per_group = 3) : 
  num_groups * candies_per_group = 30 := by
  sorry

end kathys_candy_collection_l48_4890


namespace complex_expression_eighth_root_of_unity_l48_4888

theorem complex_expression_eighth_root_of_unity :
  let z := (Complex.tan (Real.pi / 4) + Complex.I) / (Complex.tan (Real.pi / 4) - Complex.I)
  z = Complex.I ∧
  z^8 = 1 ∧
  ∃ n : ℕ, n = 2 ∧ z = Complex.exp (Complex.I * (2 * ↑n * Real.pi / 8)) := by
  sorry

end complex_expression_eighth_root_of_unity_l48_4888


namespace ellipse_and_line_properties_l48_4841

/-- An ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0
  eccentricity : ℝ
  h_eccentricity : eccentricity = Real.sqrt 6 / 3
  triangle_area : ℝ
  h_triangle_area : triangle_area = 5 * Real.sqrt 2 / 3

/-- The equation of the ellipse -/
def ellipse_equation (e : Ellipse) : ℝ × ℝ → Prop :=
  fun (x, y) ↦ x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- The moving line that intersects the ellipse -/
def moving_line (k : ℝ) : ℝ → ℝ :=
  fun x ↦ k * (x + 1)

/-- Theorem stating the properties of the ellipse and the intersecting line -/
theorem ellipse_and_line_properties (e : Ellipse) :
  (∀ x y, ellipse_equation e (x, y) ↔ x^2 / 5 + y^2 / (5/3) = 1) ∧
  (∃ k : ℝ, k = Real.sqrt 3 / 3 ∨ k = -Real.sqrt 3 / 3 ∧
    ∃ x₁ x₂ : ℝ, 
      ellipse_equation e (x₁, moving_line k x₁) ∧
      ellipse_equation e (x₂, moving_line k x₂) ∧
      (x₁ + x₂) / 2 = -1/2) :=
sorry

end ellipse_and_line_properties_l48_4841


namespace continuous_function_integrable_l48_4891

theorem continuous_function_integrable 
  {a b : ℝ} (f : ℝ → ℝ) (h : ContinuousOn f (Set.Icc a b)) : 
  IntervalIntegrable f volume a b :=
sorry

end continuous_function_integrable_l48_4891


namespace container_volume_scaling_l48_4822

theorem container_volume_scaling (original_volume : ℝ) :
  let scale_factor : ℝ := 2
  let new_volume : ℝ := original_volume * scale_factor^3
  new_volume = 8 * original_volume := by sorry

end container_volume_scaling_l48_4822


namespace otimes_inequality_solutions_l48_4849

-- Define the ⊗ operation
def otimes (a b : ℝ) : ℝ := a * (a - b) + 1

-- Define the set of non-negative integers satisfying the inequality
def solution_set : Set ℕ := {x | otimes 2 ↑x ≥ 3}

-- Theorem statement
theorem otimes_inequality_solutions :
  solution_set = {0, 1} := by sorry

end otimes_inequality_solutions_l48_4849


namespace inscribed_circles_radii_sum_l48_4810

/-- Given a triangle with an inscribed circle of radius r and three smaller triangles
    formed by tangent lines parallel to the sides of the original triangle, each with
    their own inscribed circles of radii r₁, r₂, and r₃, the sum of the radii of the
    smaller inscribed circles equals the radius of the original inscribed circle. -/
theorem inscribed_circles_radii_sum (r r₁ r₂ r₃ : ℝ) 
  (h : r > 0 ∧ r₁ > 0 ∧ r₂ > 0 ∧ r₃ > 0) : r₁ + r₂ + r₃ = r := by
  sorry

end inscribed_circles_radii_sum_l48_4810


namespace binomial_12_9_l48_4885

theorem binomial_12_9 : Nat.choose 12 9 = 220 := by sorry

end binomial_12_9_l48_4885


namespace alpine_school_math_players_l48_4855

/-- The number of players taking mathematics in Alpine School -/
def mathematics_players (total_players physics_players both_players : ℕ) : ℕ :=
  total_players - (physics_players - both_players)

/-- Theorem: Given the conditions, prove that 10 players are taking mathematics -/
theorem alpine_school_math_players :
  mathematics_players 15 9 4 = 10 := by
  sorry

end alpine_school_math_players_l48_4855


namespace kopeck_enough_for_kvass_l48_4814

/-- Represents the price of bread before any increase -/
def x : ℝ := sorry

/-- Represents the price of kvass before any increase -/
def y : ℝ := sorry

/-- The value of one kopeck -/
def kopeck : ℝ := 1

/-- Initial condition: total spending equals one kopeck -/
axiom initial_condition : x + y = kopeck

/-- Condition after first price increase -/
axiom first_increase : 0.6 * x + 1.2 * y = kopeck

/-- Theorem stating that one kopeck is enough for kvass after two 20% price increases -/
theorem kopeck_enough_for_kvass : kopeck > 1.44 * y := by sorry

end kopeck_enough_for_kvass_l48_4814


namespace absolute_value_and_roots_calculation_l48_4894

theorem absolute_value_and_roots_calculation : 
  |(-3)| + (1/2)^0 - Real.sqrt 8 * Real.sqrt 2 = 0 := by
  sorry

end absolute_value_and_roots_calculation_l48_4894


namespace cone_base_area_l48_4896

/-- Given a cone whose unfolded lateral surface is a semicircle with area 2π,
    prove that the area of its base is π. -/
theorem cone_base_area (r : ℝ) (h : r > 0) : 
  (2 * π = π * r^2) → (π * r^2 / 2 = π) :=
by sorry

end cone_base_area_l48_4896


namespace homework_problem_l48_4836

theorem homework_problem (a b c d : ℤ) 
  (h1 : a = -1) 
  (h2 : b = -c) 
  (h3 : d = -2) : 
  4*a + (b + c) - |3*d| = -10 := by
  sorry

end homework_problem_l48_4836


namespace centroid_altitude_distance_l48_4807

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the lengths of the sides
def side_lengths (t : Triangle) : ℝ × ℝ × ℝ :=
  (7, 15, 20)

-- Define the centroid G
def centroid (t : Triangle) : ℝ × ℝ := sorry

-- Define the foot of the altitude P
def altitude_foot (t : Triangle) (G : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem centroid_altitude_distance (t : Triangle) :
  let G := centroid t
  let P := altitude_foot t G
  distance G P = 1.4 := by sorry

end centroid_altitude_distance_l48_4807


namespace rectangle_100_101_diagonal_segments_l48_4897

/-- The number of segments a diagonal is divided into by grid lines in a rectangle -/
def diagonal_segments (width : ℕ) (height : ℕ) : ℕ :=
  width + height - Nat.gcd width height

/-- Theorem: In a 100 × 101 rectangle, the diagonal is divided into 200 segments by grid lines -/
theorem rectangle_100_101_diagonal_segments :
  diagonal_segments 100 101 = 200 := by
  sorry

end rectangle_100_101_diagonal_segments_l48_4897


namespace complex_arithmetic_equality_l48_4853

theorem complex_arithmetic_equality : 
  2004 - (2003 - 2004 * (2003 - 2002 * (2003 - 2004)^2004)) = 2005 := by
  sorry

end complex_arithmetic_equality_l48_4853


namespace nine_sided_polygon_diagonals_l48_4880

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A nine-sided polygon has 27 diagonals -/
theorem nine_sided_polygon_diagonals :
  num_diagonals 9 = 27 := by sorry

end nine_sided_polygon_diagonals_l48_4880


namespace ricciana_long_jump_l48_4848

/-- Ricciana's long jump problem -/
theorem ricciana_long_jump (R : ℕ) : R = 20 :=
  let ricciana_jump := 4
  let margarita_run := 18
  let margarita_jump := 2 * ricciana_jump - 1
  let ricciana_total := R + ricciana_jump
  let margarita_total := margarita_run + margarita_jump
  have h1 : margarita_total = ricciana_total + 1 := by sorry
  sorry

#check ricciana_long_jump

end ricciana_long_jump_l48_4848


namespace side_bc_equation_proof_l48_4802

/-- A triangle with two known altitudes and one known vertex -/
structure Triangle where
  -- First altitude equation: 2x - 3y + 1 = 0
  altitude1 : ℝ → ℝ → Prop
  altitude1_eq : ∀ x y, altitude1 x y ↔ 2 * x - 3 * y + 1 = 0

  -- Second altitude equation: x + y = 0
  altitude2 : ℝ → ℝ → Prop
  altitude2_eq : ∀ x y, altitude2 x y ↔ x + y = 0

  -- Vertex A coordinates
  vertex_a : ℝ × ℝ
  vertex_a_def : vertex_a = (1, 2)

/-- The equation of the line on which side BC lies -/
def side_bc_equation (t : Triangle) (x y : ℝ) : Prop :=
  2 * x + 3 * y + 7 = 0

/-- Theorem stating that the equation of side BC is 2x + 3y + 7 = 0 -/
theorem side_bc_equation_proof (t : Triangle) :
  ∀ x y, side_bc_equation t x y ↔ 2 * x + 3 * y + 7 = 0 :=
sorry

end side_bc_equation_proof_l48_4802


namespace inequality_proof_l48_4825

theorem inequality_proof (a b : ℝ) (ha : a > 1) (hb : b > 1) :
  a^2 / (b - 1) + b^2 / (a - 1) ≥ 8 := by
  sorry

end inequality_proof_l48_4825


namespace fruit_weights_correct_l48_4850

structure Fruit where
  name : String
  weight : Nat

def banana : Fruit := ⟨"banana", 170⟩
def orange : Fruit := ⟨"orange", 180⟩
def watermelon : Fruit := ⟨"watermelon", 1400⟩
def kiwi : Fruit := ⟨"kiwi", 200⟩
def apple : Fruit := ⟨"apple", 210⟩

def fruits : List Fruit := [banana, orange, watermelon, kiwi, apple]

theorem fruit_weights_correct : 
  (∀ f ∈ fruits, f.weight ∈ [170, 180, 200, 210, 1400]) ∧ 
  (watermelon.weight > banana.weight + orange.weight + kiwi.weight + apple.weight) ∧
  (orange.weight + kiwi.weight = banana.weight + apple.weight) ∧
  (orange.weight > banana.weight) ∧
  (orange.weight < kiwi.weight) := by
  sorry

end fruit_weights_correct_l48_4850


namespace banknote_replacement_theorem_l48_4856

/-- Represents the state of the banknote replacement process -/
structure BanknoteState where
  total_banknotes : ℕ
  remaining_banknotes : ℕ
  budget : ℕ
  days : ℕ

/-- Calculates the number of banknotes that can be replaced on a given day -/
def replace_banknotes (state : BanknoteState) (day : ℕ) : ℕ :=
  min state.remaining_banknotes (state.remaining_banknotes / (day + 1))

/-- Updates the state after a day of replacement -/
def update_state (state : BanknoteState) (day : ℕ) : BanknoteState :=
  let replaced := replace_banknotes state day
  { state with
    remaining_banknotes := state.remaining_banknotes - replaced
    budget := state.budget - 90000
    days := state.days + 1 }

/-- Checks if the budget is exceeded -/
def budget_exceeded (state : BanknoteState) : Prop :=
  state.budget < 0

/-- Checks if 80% of banknotes have been replaced -/
def eighty_percent_replaced (state : BanknoteState) : Prop :=
  state.remaining_banknotes ≤ state.total_banknotes / 5

/-- Main theorem statement -/
theorem banknote_replacement_theorem (initial_state : BanknoteState)
    (h_total : initial_state.total_banknotes = 3628800)
    (h_budget : initial_state.budget = 1000000) :
    ∃ (final_state : BanknoteState),
      final_state.days ≥ 4 ∧
      eighty_percent_replaced final_state ∧
      ¬∃ (complete_state : BanknoteState),
        complete_state.remaining_banknotes = 0 ∧
        ¬budget_exceeded complete_state :=
  sorry


end banknote_replacement_theorem_l48_4856


namespace circle_area_theorem_l48_4871

theorem circle_area_theorem (r : ℝ) (A : ℝ) (h : r > 0) :
  8 * (1 / A) = r^2 → A = 2 * Real.sqrt (2 * Real.pi) :=
by sorry

end circle_area_theorem_l48_4871


namespace part1_solution_part2_solution_l48_4823

-- Define A_n (falling factorial)
def A (n : ℕ) : ℕ := n * (n - 1) * (n - 2) * (n - 3)

-- Define C_n (binomial coefficient)
def C (n : ℕ) : ℕ := n * (n - 1) * (n - 2) * (n - 3) * (n - 4) * (n - 5) / 720

-- Part 1: Prove that the only positive integer solution to A_{2n+1}^4 = 140A_n^3 is n = 3
theorem part1_solution : {n : ℕ | n > 0 ∧ A (2*n + 1)^4 = 140 * A n^3} = {3} := by sorry

-- Part 2: Prove that the positive integer solutions to A_N^4 ≥ 24C_n^6 where n ≥ 6 are n = 6, 7, 8, 9, 10
theorem part2_solution : {n : ℕ | n ≥ 6 ∧ A n^4 ≥ 24 * C n^6} = {6, 7, 8, 9, 10} := by sorry

end part1_solution_part2_solution_l48_4823


namespace max_area_rectangle_l48_4817

/-- Represents a rectangle with width and length -/
structure Rectangle where
  width : ℝ
  length : ℝ

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.width + r.length)

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.width * r.length

/-- Theorem: The maximum area of a rectangle with perimeter 60 and length 5 more than width -/
theorem max_area_rectangle :
  ∃ (r : Rectangle),
    perimeter r = 60 ∧
    r.length = r.width + 5 ∧
    area r = 218.75 ∧
    ∀ (r' : Rectangle),
      perimeter r' = 60 →
      r'.length = r'.width + 5 →
      area r' ≤ area r := by
  sorry

end max_area_rectangle_l48_4817


namespace cbd_represents_115_l48_4833

/-- Represents the encoding of a base 5 digit --/
inductive Encoding
| A
| B
| C
| D
| E

/-- Represents a coded number as a list of Encodings --/
def CodedNumber := List Encoding

/-- Converts a CodedNumber to its base 10 representation --/
def to_base_10 (code : CodedNumber) : ℕ := sorry

/-- Checks if two CodedNumbers are consecutive --/
def are_consecutive (a b : CodedNumber) : Prop := sorry

theorem cbd_represents_115 
  (h1 : are_consecutive [Encoding.A, Encoding.B, Encoding.C] [Encoding.A, Encoding.B, Encoding.D])
  (h2 : are_consecutive [Encoding.A, Encoding.B, Encoding.D] [Encoding.A, Encoding.C, Encoding.E])
  (h3 : are_consecutive [Encoding.A, Encoding.C, Encoding.E] [Encoding.A, Encoding.D, Encoding.A]) :
  to_base_10 [Encoding.C, Encoding.B, Encoding.D] = 115 := by sorry

end cbd_represents_115_l48_4833


namespace tabletop_qualification_l48_4827

theorem tabletop_qualification (length width diagonal : ℝ) 
  (h_length : length = 60)
  (h_width : width = 32)
  (h_diagonal : diagonal = 68) : 
  length ^ 2 + width ^ 2 = diagonal ^ 2 := by
  sorry

end tabletop_qualification_l48_4827


namespace sapphire_percentage_l48_4834

def total_gems : ℕ := 12000
def diamonds : ℕ := 1800
def rubies : ℕ := 4000
def emeralds : ℕ := 3500

def sapphires : ℕ := total_gems - (diamonds + rubies + emeralds)

theorem sapphire_percentage :
  (sapphires : ℚ) / total_gems * 100 = 22.5 := by sorry

end sapphire_percentage_l48_4834


namespace inequality_proof_l48_4857

theorem inequality_proof (a b c d : ℝ) (ha : a ≥ -1) (hb : b ≥ -1) (hc : c ≥ -1) (hd : d ≥ -1) :
  ∃! k : ℝ, ∀ (a b c d : ℝ), a ≥ -1 → b ≥ -1 → c ≥ -1 → d ≥ -1 →
    a^3 + b^3 + c^3 + d^3 + 1 ≥ k * (a + b + c + d) ∧ k = 3/4 :=
by sorry

end inequality_proof_l48_4857


namespace smallest_square_sum_12_consecutive_l48_4847

/-- The sum of 12 consecutive integers starting from n -/
def sum_12_consecutive (n : ℕ) : ℕ := 6 * (2 * n + 11)

/-- Predicate to check if a number is a perfect square -/
def is_perfect_square (m : ℕ) : Prop := ∃ k : ℕ, m = k * k

theorem smallest_square_sum_12_consecutive :
  (∀ n : ℕ, n > 0 → sum_12_consecutive n < 150 → ¬ is_perfect_square (sum_12_consecutive n)) ∧
  is_perfect_square 150 ∧
  (∃ n : ℕ, n > 0 ∧ sum_12_consecutive n = 150) :=
sorry

end smallest_square_sum_12_consecutive_l48_4847


namespace counterexamples_exist_l48_4889

def is_counterexample (n : ℕ) : Prop :=
  ¬(Nat.Prime n) ∧ ¬(Nat.Prime (n - 3))

theorem counterexamples_exist : 
  is_counterexample 18 ∧ is_counterexample 24 :=
by sorry

end counterexamples_exist_l48_4889


namespace probability_at_least_four_same_l48_4872

/-- The number of sides on each die -/
def num_sides : ℕ := 6

/-- The number of dice rolled -/
def num_dice : ℕ := 5

/-- The probability of rolling a specific number on a fair die -/
def prob_single : ℚ := 1 / num_sides

/-- The probability that at least four out of five fair six-sided dice show the same value -/
def prob_at_least_four_same : ℚ := 13 / 648

/-- Theorem stating that the probability of at least four out of five fair six-sided dice 
    showing the same value is 13/648 -/
theorem probability_at_least_four_same : 
  prob_at_least_four_same = (1 / num_sides^4) + (5 * (1 / num_sides^3) * (5 / 6)) :=
by sorry

end probability_at_least_four_same_l48_4872


namespace clothes_washing_time_l48_4867

/-- Represents the time in minutes for washing different types of laundry -/
structure LaundryTime where
  clothes : ℕ
  towels : ℕ
  sheets : ℕ

/-- Defines the conditions for the laundry washing problem -/
def valid_laundry_time (t : LaundryTime) : Prop :=
  t.towels = 2 * t.clothes ∧
  t.sheets = t.towels - 15 ∧
  t.clothes + t.towels + t.sheets = 135

/-- Theorem stating that the time to wash clothes is 30 minutes -/
theorem clothes_washing_time (t : LaundryTime) :
  valid_laundry_time t → t.clothes = 30 := by
  sorry

end clothes_washing_time_l48_4867


namespace remainder_theorem_remainder_is_16_l48_4851

/-- The polynomial f(x) = x^4 - 6x^3 + 11x^2 + 12x - 20 -/
def f (x : ℝ) : ℝ := x^4 - 6*x^3 + 11*x^2 + 12*x - 20

/-- The remainder when f(x) is divided by (x - 2) is equal to f(2) -/
theorem remainder_theorem (x : ℝ) : 
  ∃ (q : ℝ → ℝ), f x = (x - 2) * q x + f 2 := by sorry

/-- The remainder when x^4 - 6x^3 + 11x^2 + 12x - 20 is divided by x - 2 is 16 -/
theorem remainder_is_16 : f 2 = 16 := by sorry

end remainder_theorem_remainder_is_16_l48_4851


namespace first_day_of_month_l48_4835

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

def next_day (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

def day_after_n_days (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => next_day (day_after_n_days d n)

theorem first_day_of_month (d : DayOfWeek) :
  day_after_n_days d 27 = DayOfWeek.Tuesday → d = DayOfWeek.Wednesday :=
by sorry

end first_day_of_month_l48_4835


namespace original_equals_scientific_l48_4883

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- The number to be converted -/
def original_number : ℕ := 28000

/-- The scientific notation representation of the original number -/
def scientific_representation : ScientificNotation :=
  { coefficient := 2.8
    exponent := 4
    is_valid := by sorry }

/-- Theorem stating that the original number is equal to its scientific notation representation -/
theorem original_equals_scientific :
  (original_number : ℝ) = scientific_representation.coefficient * (10 : ℝ) ^ scientific_representation.exponent :=
by sorry

end original_equals_scientific_l48_4883


namespace laser_path_distance_correct_l48_4824

/-- The total distance traveled by a laser beam with specified bounces -/
def laser_path_distance : ℝ := 12

/-- Starting point of the laser -/
def start_point : ℝ × ℝ := (4, 6)

/-- Final point of the laser -/
def end_point : ℝ × ℝ := (8, 6)

/-- Theorem stating that the laser path distance is correct -/
theorem laser_path_distance_correct :
  let path := laser_path_distance
  let start := start_point
  let end_ := end_point
  (path = ‖(start.1 + end_.1, start.2 - end_.2)‖) ∧
  (path > 0) ∧
  (start.1 > 0) ∧
  (start.2 > 0) ∧
  (end_.1 > 0) ∧
  (end_.2 > 0) :=
by sorry

end laser_path_distance_correct_l48_4824


namespace equilateral_triangle_inscribed_circle_radius_l48_4862

/-- Given an equilateral triangle inscribed in a circle with area 81 cm²,
    prove that the radius of the circle is 6 * (3^(1/4)) cm. -/
theorem equilateral_triangle_inscribed_circle_radius 
  (S : ℝ) (r : ℝ) (h1 : S = 81) :
  r = 6 * (3 : ℝ)^(1/4) :=
by
  sorry


end equilateral_triangle_inscribed_circle_radius_l48_4862


namespace cricket_team_age_difference_l48_4812

theorem cricket_team_age_difference (team_size : ℕ) (captain_age : ℕ) (team_avg_age : ℕ) (remaining_avg_age : ℕ) : 
  team_size = 11 → 
  captain_age = 24 → 
  team_avg_age = 23 → 
  remaining_avg_age = team_avg_age - 1 → 
  ∃ (wicket_keeper_age : ℕ), wicket_keeper_age = captain_age + 7 :=
by sorry

end cricket_team_age_difference_l48_4812


namespace earrings_to_necklace_ratio_l48_4846

theorem earrings_to_necklace_ratio 
  (total_cost : ℝ) 
  (num_necklaces : ℕ) 
  (single_necklace_cost : ℝ) 
  (h1 : total_cost = 240000)
  (h2 : num_necklaces = 3)
  (h3 : single_necklace_cost = 40000) :
  (total_cost - num_necklaces * single_necklace_cost) / single_necklace_cost = 3 := by
  sorry

end earrings_to_necklace_ratio_l48_4846


namespace perpendicular_lines_from_perpendicular_planes_l48_4886

-- Define the space
variable (Space : Type)

-- Define lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (skew : Line → Line → Prop)
variable (parallel : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_plane : Plane → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (perpendicular_plane : Plane → Plane → Prop)
variable (perpendicular_line : Line → Line → Prop)

-- Define the lines and planes
variable (l m n : Line)
variable (α β : Plane)

-- State that l, m, n are different lines
variable (h_diff_lm : l ≠ m)
variable (h_diff_ln : l ≠ n)
variable (h_diff_mn : m ≠ n)

-- State that α and β are non-coincident planes
variable (h_non_coincident : α ≠ β)

-- State the theorem to be proved
theorem perpendicular_lines_from_perpendicular_planes :
  (perpendicular_plane α β ∧ perpendicular_line_plane l α ∧ perpendicular_line_plane m β) →
  perpendicular_line l m :=
sorry

end perpendicular_lines_from_perpendicular_planes_l48_4886


namespace desk_chair_cost_l48_4879

theorem desk_chair_cost (cost_A cost_B : ℝ) : 
  (cost_B = cost_A + 40) →
  (4 * cost_A + 5 * cost_B = 1820) →
  (cost_A = 180 ∧ cost_B = 220) := by
sorry

end desk_chair_cost_l48_4879


namespace absolute_value_of_complex_fraction_l48_4826

theorem absolute_value_of_complex_fraction : 
  Complex.abs (2 / (1 + Complex.I)) = Real.sqrt 2 := by sorry

end absolute_value_of_complex_fraction_l48_4826


namespace complex_real_condition_l48_4843

theorem complex_real_condition (i : ℂ) (m : ℝ) : 
  i * i = -1 →
  (1 / (2 + i) + m * i).im = 0 →
  m = 1 / 5 := by
  sorry

end complex_real_condition_l48_4843
