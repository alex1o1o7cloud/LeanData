import Mathlib

namespace NUMINAMATH_CALUDE_two_numbers_problem_l3207_320798

theorem two_numbers_problem (a b : ℝ) (h1 : a + b = 60) (h2 : a - b = 10) : 
  a * b = 875 ∧ a^2 + b^2 = 1850 := by
sorry

end NUMINAMATH_CALUDE_two_numbers_problem_l3207_320798


namespace NUMINAMATH_CALUDE_paint_usage_fraction_l3207_320793

theorem paint_usage_fraction (initial_paint : ℚ) (first_week_fraction : ℚ) (total_used : ℚ) :
  initial_paint = 360 →
  first_week_fraction = 1/4 →
  total_used = 180 →
  let remaining_after_first_week := initial_paint - first_week_fraction * initial_paint
  let used_second_week := total_used - first_week_fraction * initial_paint
  used_second_week / remaining_after_first_week = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_paint_usage_fraction_l3207_320793


namespace NUMINAMATH_CALUDE_four_digit_numbers_with_5_or_7_l3207_320702

def four_digit_numbers : ℕ := 9000

def digits_without_5_or_7 : ℕ := 8

def first_digit_options : ℕ := 7

def numbers_without_5_or_7 : ℕ := first_digit_options * (digits_without_5_or_7 ^ 3)

theorem four_digit_numbers_with_5_or_7 :
  four_digit_numbers - numbers_without_5_or_7 = 5416 :=
by sorry

end NUMINAMATH_CALUDE_four_digit_numbers_with_5_or_7_l3207_320702


namespace NUMINAMATH_CALUDE_basketball_game_proof_l3207_320725

def basketball_game (basket_points : ℕ) (matthew_points : ℕ) (shawn_points : ℕ) : Prop :=
  ∃ (matthew_baskets shawn_baskets : ℕ),
    basket_points * matthew_baskets = matthew_points ∧
    basket_points * shawn_baskets = shawn_points ∧
    matthew_baskets + shawn_baskets = 5

theorem basketball_game_proof :
  basketball_game 3 9 6 := by
  sorry

end NUMINAMATH_CALUDE_basketball_game_proof_l3207_320725


namespace NUMINAMATH_CALUDE_same_grade_probability_l3207_320781

/-- Represents the grades in the school -/
inductive Grade
| A
| B
| C

/-- Represents a student volunteer -/
structure Student where
  grade : Grade

/-- The total number of student volunteers -/
def total_students : Nat := 560

/-- The number of students in each grade -/
def students_per_grade (g : Grade) : Nat :=
  match g with
  | Grade.A => 240
  | Grade.B => 160
  | Grade.C => 160

/-- The number of students selected from each grade for the charity event -/
def selected_per_grade (g : Grade) : Nat :=
  match g with
  | Grade.A => 3
  | Grade.B => 2
  | Grade.C => 2

/-- The total number of students selected for the charity event -/
def total_selected : Nat := 7

/-- The number of students to be selected for sanitation work -/
def sanitation_workers : Nat := 2

/-- Theorem: The probability of selecting 2 students from the same grade for sanitation work is 5/21 -/
theorem same_grade_probability :
  (Nat.choose total_selected sanitation_workers) = 21 ∧
  (Nat.choose (selected_per_grade Grade.A) sanitation_workers +
   Nat.choose (selected_per_grade Grade.B) sanitation_workers +
   Nat.choose (selected_per_grade Grade.C) sanitation_workers) = 5 :=
by sorry


end NUMINAMATH_CALUDE_same_grade_probability_l3207_320781


namespace NUMINAMATH_CALUDE_division_addition_problem_l3207_320739

theorem division_addition_problem : (-144) / (-36) + 10 = 14 := by
  sorry

end NUMINAMATH_CALUDE_division_addition_problem_l3207_320739


namespace NUMINAMATH_CALUDE_jay_change_is_twenty_l3207_320722

-- Define the prices of items and the payment amount
def book_price : ℕ := 25
def pen_price : ℕ := 4
def ruler_price : ℕ := 1
def payment : ℕ := 50

-- Define the change received
def change : ℕ := payment - (book_price + pen_price + ruler_price)

-- Theorem statement
theorem jay_change_is_twenty : change = 20 := by
  sorry

end NUMINAMATH_CALUDE_jay_change_is_twenty_l3207_320722


namespace NUMINAMATH_CALUDE_average_marks_math_biology_l3207_320797

theorem average_marks_math_biology 
  (P C M B : ℕ) -- Marks in Physics, Chemistry, Mathematics, and Biology
  (h : P + C + M + B = P + C + 200) -- Total marks condition
  : (M + B) / 2 = 100 := by
sorry

end NUMINAMATH_CALUDE_average_marks_math_biology_l3207_320797


namespace NUMINAMATH_CALUDE_tangent_line_to_parabola_l3207_320769

theorem tangent_line_to_parabola (d : ℝ) : 
  (∃ (x y : ℝ), y = 3*x + d ∧ y^2 = 12*x ∧ 
   ∀ (x' y' : ℝ), y' = 3*x' + d → y'^2 ≥ 12*x') → d = 1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_to_parabola_l3207_320769


namespace NUMINAMATH_CALUDE_batsman_average_increase_l3207_320762

/-- Represents a batsman's performance -/
structure Batsman where
  innings : Nat
  totalScore : Nat
  lastInningsScore : Nat

/-- Calculates the average score of a batsman -/
def average (b : Batsman) : ℚ :=
  b.totalScore / b.innings

/-- The increase in average for a batsman after their last innings -/
def averageIncrease (b : Batsman) : ℚ :=
  average b - average { b with
    innings := b.innings - 1
    totalScore := b.totalScore - b.lastInningsScore
  }

theorem batsman_average_increase :
  ∀ b : Batsman,
    b.innings = 12 ∧
    b.lastInningsScore = 70 ∧
    average b = 37 →
    averageIncrease b = 3 := by
  sorry

end NUMINAMATH_CALUDE_batsman_average_increase_l3207_320762


namespace NUMINAMATH_CALUDE_f_difference_l3207_320728

/-- k(n) is the largest odd divisor of n -/
def k (n : ℕ+) : ℕ+ := sorry

/-- f(n) is the sum of k(i) from i=1 to n -/
def f (n : ℕ+) : ℕ := sorry

/-- Theorem: f(2n) - f(n) = n^2 for any positive integer n -/
theorem f_difference (n : ℕ+) : f (2 * n) - f n = n^2 := by sorry

end NUMINAMATH_CALUDE_f_difference_l3207_320728


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3207_320753

theorem min_value_reciprocal_sum (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : 2 * m + n = 1) :
  (1 / m + 1 / n) ≥ 3 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3207_320753


namespace NUMINAMATH_CALUDE_root_equation_consequence_l3207_320771

theorem root_equation_consequence (m : ℝ) : 
  m^2 - 2*m - 7 = 0 → m^2 - 2*m + 1 = 8 := by
  sorry

end NUMINAMATH_CALUDE_root_equation_consequence_l3207_320771


namespace NUMINAMATH_CALUDE_function_properties_l3207_320754

noncomputable def f (a b x : ℝ) : ℝ := 2 * x^3 + a * x^2 + b * x + 1

def f_derivative_symmetric (a b : ℝ) : Prop :=
  ∀ x : ℝ, (6 * x^2 + 2 * a * x + b) = (6 * (-x - 1)^2 + 2 * a * (-x - 1) + b)

theorem function_properties (a b : ℝ) 
  (h1 : f_derivative_symmetric a b)
  (h2 : 6 + 2 * a + b = 0) :
  (a = 3 ∧ b = -12) ∧
  (∀ x : ℝ, f a b x ≤ f a b (-2)) ∧
  (∀ x : ℝ, f a b x ≥ f a b 1) ∧
  (f a b (-2) = 21) ∧
  (f a b 1 = -6) := by sorry

end NUMINAMATH_CALUDE_function_properties_l3207_320754


namespace NUMINAMATH_CALUDE_complement_union_equals_singleton_l3207_320788

def M : Set (ℝ × ℝ) := {p | p.2 ≠ p.1 + 2}

def N : Set (ℝ × ℝ) := {p | p.2 ≠ -p.1}

def I : Set (ℝ × ℝ) := Set.univ

theorem complement_union_equals_singleton : 
  (I \ (M ∪ N)) = {(-1, 1)} := by sorry

end NUMINAMATH_CALUDE_complement_union_equals_singleton_l3207_320788


namespace NUMINAMATH_CALUDE_notched_circle_distance_l3207_320764

-- Define the circle and points
def Circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 72}

def B : ℝ × ℝ := (1, -4)
def A : ℝ × ℝ := (1, 4)
def C : ℝ × ℝ := (7, -4)

-- State the theorem
theorem notched_circle_distance :
  B ∈ Circle ∧
  A ∈ Circle ∧
  C ∈ Circle ∧
  A.1 = B.1 ∧
  A.2 - B.2 = 8 ∧
  C.1 - B.1 = 6 ∧
  C.2 = B.2 ∧
  (A.1 - B.1) * (C.1 - B.1) + (A.2 - B.2) * (C.2 - B.2) = 0 →
  B.1^2 + B.2^2 = 17 := by
  sorry

end NUMINAMATH_CALUDE_notched_circle_distance_l3207_320764


namespace NUMINAMATH_CALUDE_jeff_tennis_matches_l3207_320731

/-- Calculates the number of matches won in a tennis game -/
def matches_won (total_time : ℕ) (point_interval : ℕ) (points_per_match : ℕ) : ℕ :=
  (total_time * 60 / point_interval) / points_per_match

/-- Theorem stating that under given conditions, 3 matches are won -/
theorem jeff_tennis_matches : 
  matches_won 2 5 8 = 3 := by
  sorry

end NUMINAMATH_CALUDE_jeff_tennis_matches_l3207_320731


namespace NUMINAMATH_CALUDE_absolute_value_and_opposite_l3207_320765

theorem absolute_value_and_opposite :
  (|-2/5| = 2/5) ∧ (-(2023 : ℤ) = -2023) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_and_opposite_l3207_320765


namespace NUMINAMATH_CALUDE_solution_set_inequality_l3207_320758

theorem solution_set_inequality (x : ℝ) : (3*x - 1) / (2 - x) ≥ 1 ↔ 3/4 ≤ x ∧ x < 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l3207_320758


namespace NUMINAMATH_CALUDE_mary_potatoes_l3207_320730

def total_potatoes (initial : ℕ) (remaining_new : ℕ) : ℕ :=
  initial + remaining_new

theorem mary_potatoes : total_potatoes 8 3 = 11 := by
  sorry

end NUMINAMATH_CALUDE_mary_potatoes_l3207_320730


namespace NUMINAMATH_CALUDE_distinct_roots_equal_integer_roots_l3207_320723

-- Define the quadratic equation
def quadratic (m : ℝ) (x : ℝ) : ℝ := x^2 + (m + 3) * x + 2 * m

-- Part 1: Prove the equation always has two distinct real roots
theorem distinct_roots (m : ℝ) : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic m x₁ = 0 ∧ quadratic m x₂ = 0 :=
sorry

-- Part 2: Prove the specific case has two equal integer roots
theorem equal_integer_roots : 
  ∃ x : ℤ, quadratic 2 (x : ℝ) = 0 ∧ x = -2 :=
sorry

end NUMINAMATH_CALUDE_distinct_roots_equal_integer_roots_l3207_320723


namespace NUMINAMATH_CALUDE_range_of_a_l3207_320703

theorem range_of_a (a : ℝ) : (∃ x : ℝ, (Real.exp x - a)^2 + x^2 - 2*a*x + a^2 ≤ 1/2) → a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3207_320703


namespace NUMINAMATH_CALUDE_square_difference_minus_sum_of_squares_specific_case_l3207_320738

theorem square_difference_minus_sum_of_squares (a b : ℤ) :
  (a - b)^2 - (b^2 + a^2 - 2*a*b) = 0 :=
by sorry

-- Specific case for a = 36 and b = 15
theorem specific_case : (36 - 15)^2 - (15^2 + 36^2 - 2*15*36) = 0 :=
by sorry

end NUMINAMATH_CALUDE_square_difference_minus_sum_of_squares_specific_case_l3207_320738


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l3207_320784

theorem polynomial_divisibility (C D : ℝ) :
  (∀ x : ℂ, x^2 + x + 1 = 0 → x^104 + C*x + D = 0) →
  C + D = 2 := by
sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l3207_320784


namespace NUMINAMATH_CALUDE_dalton_needs_four_more_l3207_320704

/-- The amount of additional money Dalton needs to buy his desired items -/
def additional_money_needed (jump_rope_cost board_game_cost ball_cost saved_allowance uncle_gift : ℕ) : ℕ :=
  let total_cost := jump_rope_cost + board_game_cost + ball_cost
  let available_money := saved_allowance + uncle_gift
  if total_cost > available_money then
    total_cost - available_money
  else
    0

/-- Theorem stating that Dalton needs $4 more to buy his desired items -/
theorem dalton_needs_four_more :
  additional_money_needed 7 12 4 6 13 = 4 := by
  sorry

end NUMINAMATH_CALUDE_dalton_needs_four_more_l3207_320704


namespace NUMINAMATH_CALUDE_odd_power_divisibility_l3207_320782

theorem odd_power_divisibility (a b : ℕ) (ha : Odd a) (hb : Odd b) (ha_pos : 0 < a) (hb_pos : 0 < b) :
  ∀ n : ℕ, 0 < n → ∃ m : ℕ, (2^n ∣ a^m * b^2 - 1) ∨ (2^n ∣ b^m * a^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_odd_power_divisibility_l3207_320782


namespace NUMINAMATH_CALUDE_pair_probability_l3207_320750

/-- Represents a deck of cards -/
structure Deck :=
  (total : Nat)
  (numbers : Nat)
  (cards_per_number : Nat)

/-- The probability of selecting a pair from a deck -/
def probability_of_pair (d : Deck) : Rat :=
  sorry

/-- The original deck -/
def original_deck : Deck :=
  { total := 52, numbers := 13, cards_per_number := 4 }

/-- The deck after removing a matching pair -/
def reduced_deck : Deck :=
  { total := 48, numbers := 12, cards_per_number := 4 }

theorem pair_probability :
  probability_of_pair reduced_deck = 3 / 47 := by
  sorry

end NUMINAMATH_CALUDE_pair_probability_l3207_320750


namespace NUMINAMATH_CALUDE_rectangle_longest_side_l3207_320759

theorem rectangle_longest_side (l w : ℝ) : 
  l > 0 ∧ w > 0 ∧                 -- positive dimensions
  2 * (l + w) = 180 ∧             -- perimeter is 180 feet
  l * w = 8 * 180 →               -- area is 8 times perimeter
  max l w = 72 := by sorry

end NUMINAMATH_CALUDE_rectangle_longest_side_l3207_320759


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l3207_320740

theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ 2 + a^(x - 2)
  f 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l3207_320740


namespace NUMINAMATH_CALUDE_no_real_roots_quadratic_l3207_320743

theorem no_real_roots_quadratic (m : ℝ) : 
  (∀ x : ℝ, (m - 1) * x^2 - 2 * x + 1 ≠ 0) → m > 2 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_quadratic_l3207_320743


namespace NUMINAMATH_CALUDE_log_difference_times_sqrt10_l3207_320785

-- Define the logarithm function (base 10)
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_difference_times_sqrt10 :
  (log10 (1/4) - log10 25) * (10 ^ (1/2 : ℝ)) = -10 := by
  sorry

end NUMINAMATH_CALUDE_log_difference_times_sqrt10_l3207_320785


namespace NUMINAMATH_CALUDE_course_selection_schemes_l3207_320734

/-- The number of elective courses in physical education -/
def pe_courses : ℕ := 4

/-- The number of elective courses in art -/
def art_courses : ℕ := 4

/-- The minimum number of courses a student can choose -/
def min_courses : ℕ := 2

/-- The maximum number of courses a student can choose -/
def max_courses : ℕ := 3

/-- The minimum number of courses a student must choose from each category -/
def min_per_category : ℕ := 1

/-- The total number of different course selection schemes -/
def total_schemes : ℕ := 64

/-- Theorem stating that the total number of different course selection schemes is 64 -/
theorem course_selection_schemes :
  (pe_courses = 4) →
  (art_courses = 4) →
  (min_courses = 2) →
  (max_courses = 3) →
  (min_per_category = 1) →
  (total_schemes = 64) :=
by sorry

end NUMINAMATH_CALUDE_course_selection_schemes_l3207_320734


namespace NUMINAMATH_CALUDE_orange_distribution_l3207_320735

theorem orange_distribution (total_oranges : ℕ) (bad_oranges : ℕ) (difference : ℕ) :
  total_oranges = 108 →
  bad_oranges = 36 →
  difference = 3 →
  (total_oranges : ℚ) / (total_oranges / difference - bad_oranges / difference) - 
  ((total_oranges - bad_oranges) : ℚ) / (total_oranges / difference - bad_oranges / difference) = difference →
  total_oranges / difference - bad_oranges / difference = 12 :=
by sorry

end NUMINAMATH_CALUDE_orange_distribution_l3207_320735


namespace NUMINAMATH_CALUDE_doughnut_boxes_l3207_320741

theorem doughnut_boxes (total_doughnuts : ℕ) (doughnuts_per_box : ℕ) (h1 : total_doughnuts = 48) (h2 : doughnuts_per_box = 12) :
  total_doughnuts / doughnuts_per_box = 4 := by
  sorry

end NUMINAMATH_CALUDE_doughnut_boxes_l3207_320741


namespace NUMINAMATH_CALUDE_new_person_weight_l3207_320777

/-- Given a group of 8 people, where one person weighing 70 kg is replaced by a new person,
    causing the average weight to increase by 2.5 kg, 
    prove that the weight of the new person is 90 kg. -/
theorem new_person_weight (initial_count : ℕ) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 8 →
  weight_increase = 2.5 →
  replaced_weight = 70 →
  (initial_count : ℝ) * weight_increase + replaced_weight = 90 := by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_l3207_320777


namespace NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_q_l3207_320710

-- Define propositions p and q
def p (x : ℝ) : Prop := |x + 1| ≤ 4
def q (x : ℝ) : Prop := x^2 < 5*x - 6

-- Define the negations of p and q
def not_p (x : ℝ) : Prop := ¬(p x)
def not_q (x : ℝ) : Prop := ¬(q x)

-- Theorem stating that ¬p is a sufficient but not necessary condition for ¬q
theorem not_p_sufficient_not_necessary_for_not_q :
  (∀ x : ℝ, not_p x → not_q x) ∧ 
  (∃ x : ℝ, not_q x ∧ ¬(not_p x)) :=
sorry

end NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_q_l3207_320710


namespace NUMINAMATH_CALUDE_min_value_of_sum_of_ratios_l3207_320772

theorem min_value_of_sum_of_ratios (a₁ a₂ a₃ a₄ a₅ a₆ : ℕ) 
  (h1 : 1 ≤ a₁) (h2 : a₁ ≤ a₂) (h3 : a₂ ≤ a₃) (h4 : a₃ ≤ a₄) 
  (h5 : a₄ ≤ a₅) (h6 : a₅ ≤ a₆) (h7 : a₆ ≤ 64) :
  (a₁ : ℚ) / a₂ + (a₃ : ℚ) / a₄ + (a₅ : ℚ) / a₆ ≥ 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_sum_of_ratios_l3207_320772


namespace NUMINAMATH_CALUDE_no_integer_solutions_l3207_320786

theorem no_integer_solutions : ¬ ∃ (x y : ℤ), 14 * x^2 + 15 * y^2 = 7^2000 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l3207_320786


namespace NUMINAMATH_CALUDE_complementary_point_on_line_complementary_point_on_general_line_unique_complementary_point_quadratic_l3207_320791

-- Define complementary point
def is_complementary_point (x y : ℝ) : Prop := y = -x

-- Part 1
theorem complementary_point_on_line :
  ∃ (x y : ℝ), is_complementary_point x y ∧ y = 2 * x - 3 ∧ x = 1 ∧ y = -1 := by sorry

-- Part 2
theorem complementary_point_on_general_line (k : ℝ) (h : k ≠ 0) :
  (k ≠ -1) →
  ∃ (x y : ℝ), is_complementary_point x y ∧ y = k * x + 2 ∧ x = -2 / (k + 1) ∧ y = 2 / (k + 1) := by sorry

-- Part 3
theorem unique_complementary_point_quadratic (n m : ℝ) (h : -1 ≤ n ∧ n ≤ 2) :
  (∃! (x y : ℝ), is_complementary_point x y ∧ y = 1/4 * x^2 + (n - k - 1) * x + m + k - 2) ∧
  (∀ m', m' ≥ m → m' ≥ k) →
  (k = 1 ∨ k = 3 + Real.sqrt 3) := by sorry

end NUMINAMATH_CALUDE_complementary_point_on_line_complementary_point_on_general_line_unique_complementary_point_quadratic_l3207_320791


namespace NUMINAMATH_CALUDE_parallelogram_fourth_vertex_l3207_320778

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of a parallelogram -/
structure Parallelogram where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Check if four points form a parallelogram -/
def is_parallelogram (p : Parallelogram) : Prop :=
  (p.A.x + p.C.x = p.B.x + p.D.x) ∧ (p.A.y + p.C.y = p.B.y + p.D.y)

theorem parallelogram_fourth_vertex :
  ∀ (p : Parallelogram),
  p.A = Point.mk (-2) 1 →
  p.B = Point.mk (-1) 3 →
  p.C = Point.mk 3 4 →
  is_parallelogram p →
  (p.D = Point.mk 2 2 ∨ p.D = Point.mk (-6) 0 ∨ p.D = Point.mk 4 6) :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_fourth_vertex_l3207_320778


namespace NUMINAMATH_CALUDE_train_crossing_time_l3207_320757

/-- Proves that a train 400 meters long, traveling at 36 km/h, takes 40 seconds to cross an electric pole. -/
theorem train_crossing_time (train_length : Real) (train_speed_kmh : Real) (crossing_time : Real) :
  train_length = 400 →
  train_speed_kmh = 36 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) →
  crossing_time = 40 := by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l3207_320757


namespace NUMINAMATH_CALUDE_inequality_system_solution_range_l3207_320711

theorem inequality_system_solution_range (a : ℝ) : 
  (∃! (s : Finset ℤ), s.card = 3 ∧ 
    (∀ x : ℤ, x ∈ s ↔ (x > 2*a - 3 ∧ 2*x ≥ 3*(x-2) + 5))) →
  (1/2 : ℝ) ≤ a ∧ a < 1 :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_range_l3207_320711


namespace NUMINAMATH_CALUDE_sequence_term_proof_l3207_320751

/-- Given a sequence where the sum of the first n terms is 5n + 2n^2,
    this function represents the rth term of the sequence. -/
def sequence_term (r : ℕ) : ℕ := 4 * r + 3

/-- The sum of the first n terms of the sequence. -/
def sequence_sum (n : ℕ) : ℕ := 5 * n + 2 * n^2

/-- Theorem stating that the rth term of the sequence is 4r + 3,
    given that the sum of the first n terms is 5n + 2n^2 for all n. -/
theorem sequence_term_proof (r : ℕ) : 
  sequence_term r = sequence_sum r - sequence_sum (r - 1) :=
sorry

end NUMINAMATH_CALUDE_sequence_term_proof_l3207_320751


namespace NUMINAMATH_CALUDE_total_sugar_needed_l3207_320779

def sugar_for_frosting : ℝ := 0.6
def sugar_for_cake : ℝ := 0.2

theorem total_sugar_needed : sugar_for_frosting + sugar_for_cake = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_total_sugar_needed_l3207_320779


namespace NUMINAMATH_CALUDE_carbonated_water_percentage_carbonated_water_percentage_proof_l3207_320763

theorem carbonated_water_percentage : ℝ → Prop :=
  λ x =>
    let first_solution_carbonated := 0.80
    let mixture_carbonated := 0.65
    let first_solution_ratio := 0.40
    let second_solution_ratio := 0.60
    first_solution_carbonated * first_solution_ratio + x * second_solution_ratio = mixture_carbonated →
    x = 0.55

-- The proof is omitted
theorem carbonated_water_percentage_proof : carbonated_water_percentage 0.55 := by sorry

end NUMINAMATH_CALUDE_carbonated_water_percentage_carbonated_water_percentage_proof_l3207_320763


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l3207_320794

/-- The equation of a line perpendicular to another line and passing through a given point. -/
theorem perpendicular_line_equation (m : ℚ) (b : ℚ) (x₀ : ℚ) (y₀ : ℚ) :
  let l₁ : ℚ → ℚ := λ x => m * x + b
  let m₂ : ℚ := -1 / m
  let l₂ : ℚ → ℚ := λ x => m₂ * (x - x₀) + y₀
  (∀ x, x - 2 * l₁ x + 3 = 0) →
  (∀ x, 2 * x + l₂ x - 1 = 0) := by
sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l3207_320794


namespace NUMINAMATH_CALUDE_zero_success_probability_l3207_320775

/-- The probability of 0 successes in 7 Bernoulli trials with success probability 2/7 -/
def prob_zero_success (n : ℕ) (p : ℚ) : ℚ :=
  (1 - p) ^ n

/-- The number of trials -/
def num_trials : ℕ := 7

/-- The probability of success in a single trial -/
def success_prob : ℚ := 2/7

theorem zero_success_probability :
  prob_zero_success num_trials success_prob = (5/7) ^ 7 := by
  sorry

end NUMINAMATH_CALUDE_zero_success_probability_l3207_320775


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3207_320780

/-- An arithmetic sequence with first term 2 and 10th term 20 has common difference 2 -/
theorem arithmetic_sequence_common_difference : ∀ (a : ℕ → ℝ), 
  (∀ n, a (n + 1) - a n = a 2 - a 1) →  -- arithmetic sequence condition
  a 1 = 2 →                            -- first term is 2
  a 10 = 20 →                          -- 10th term is 20
  a 2 - a 1 = 2 :=                     -- common difference is 2
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3207_320780


namespace NUMINAMATH_CALUDE_cube_sum_divisibility_l3207_320708

theorem cube_sum_divisibility (a b c : ℤ) 
  (h1 : 6 ∣ (a^2 + b^2 + c^2))
  (h2 : 3 ∣ (a*b + b*c + c*a)) :
  6 ∣ (a^3 + b^3 + c^3) := by
sorry

end NUMINAMATH_CALUDE_cube_sum_divisibility_l3207_320708


namespace NUMINAMATH_CALUDE_cubic_fraction_zero_l3207_320700

theorem cubic_fraction_zero (a b c : ℝ) (h : a + b + c ≠ 0) :
  ((a^3 - b^3)^2 + (b^3 - c^3)^2 + (c^3 - a^3)^2) / (a^3 + b^3 + c^3 - 3*a*b*c) = 0 :=
by sorry

end NUMINAMATH_CALUDE_cubic_fraction_zero_l3207_320700


namespace NUMINAMATH_CALUDE_least_common_multiple_first_ten_l3207_320719

theorem least_common_multiple_first_ten : ∃ n : ℕ, 
  (∀ k : ℕ, k ≤ 10 → k > 0 → n % k = 0) ∧ 
  (∀ m : ℕ, m < n → ∃ k : ℕ, k ≤ 10 ∧ k > 0 ∧ m % k ≠ 0) ∧
  n = 2520 := by
sorry

end NUMINAMATH_CALUDE_least_common_multiple_first_ten_l3207_320719


namespace NUMINAMATH_CALUDE_inscribed_squares_ratio_l3207_320706

theorem inscribed_squares_ratio : ∀ x y : ℝ,
  (5 : ℝ) ^ 2 + 12 ^ 2 = 13 ^ 2 →
  (12 - x) / 12 = x / 5 →
  y + 2 * (5 * y / 13) = 13 →
  x / y = 1380 / 2873 := by
sorry

end NUMINAMATH_CALUDE_inscribed_squares_ratio_l3207_320706


namespace NUMINAMATH_CALUDE_new_students_l3207_320789

theorem new_students (initial : ℕ) (left : ℕ) (final : ℕ) : 
  initial = 33 → left = 18 → final = 29 → final - (initial - left) = 14 := by
  sorry

end NUMINAMATH_CALUDE_new_students_l3207_320789


namespace NUMINAMATH_CALUDE_nested_square_root_value_l3207_320705

theorem nested_square_root_value :
  ∃ y : ℝ, y = Real.sqrt (2 + y) → y = 2 := by
  sorry

end NUMINAMATH_CALUDE_nested_square_root_value_l3207_320705


namespace NUMINAMATH_CALUDE_unique_intersection_l3207_320712

/-- A function f(x) that represents a quadratic or linear equation depending on the value of a -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + (a - 3) * x + 1

/-- Theorem stating that f(x) intersects the x-axis at only one point iff a = 0, 1, or 9 -/
theorem unique_intersection (a : ℝ) :
  (∃! x, f a x = 0) ↔ (a = 0 ∨ a = 1 ∨ a = 9) := by
  sorry

end NUMINAMATH_CALUDE_unique_intersection_l3207_320712


namespace NUMINAMATH_CALUDE_count_numbers_eq_243_l3207_320767

/-- The count of three-digit numbers less than 500 that do not contain the digit 1 -/
def count_numbers : Nat :=
  let hundreds := {2, 3, 4}
  let other_digits := {0, 2, 3, 4, 5, 6, 7, 8, 9}
  (Finset.card hundreds) * (Finset.card other_digits) * (Finset.card other_digits)

/-- Theorem stating that the count of three-digit numbers less than 500 
    that do not contain the digit 1 is equal to 243 -/
theorem count_numbers_eq_243 : count_numbers = 243 := by
  sorry

end NUMINAMATH_CALUDE_count_numbers_eq_243_l3207_320767


namespace NUMINAMATH_CALUDE_total_production_8_minutes_l3207_320774

/-- Represents a machine type in the factory -/
inductive MachineType
| A
| B
| C

/-- Represents the state of the factory at a given time -/
structure FactoryState where
  machineCount : MachineType → ℕ
  productionRate : MachineType → ℕ

/-- Calculates the total production for a given time interval -/
def totalProduction (state : FactoryState) (minutes : ℕ) : ℕ :=
  (state.machineCount MachineType.A * state.productionRate MachineType.A +
   state.machineCount MachineType.B * state.productionRate MachineType.B +
   state.machineCount MachineType.C * state.productionRate MachineType.C) * minutes

/-- The initial state of the factory -/
def initialState : FactoryState := {
  machineCount := λ t => match t with
    | MachineType.A => 6
    | MachineType.B => 4
    | MachineType.C => 3
  productionRate := λ t => match t with
    | MachineType.A => 270
    | MachineType.B => 200
    | MachineType.C => 150
}

/-- The state of the factory after 2 minutes -/
def stateAfter2Min : FactoryState := {
  machineCount := λ t => match t with
    | MachineType.A => 4
    | MachineType.B => 7
    | MachineType.C => 3
  productionRate := λ t => match t with
    | MachineType.A => 270
    | MachineType.B => 200
    | MachineType.C => 150
}

/-- The state of the factory after 4 minutes -/
def stateAfter4Min : FactoryState := {
  machineCount := λ t => match t with
    | MachineType.A => 6
    | MachineType.B => 9
    | MachineType.C => 3
  productionRate := λ t => match t with
    | MachineType.A => 300
    | MachineType.B => 180
    | MachineType.C => 170
}

/-- Theorem stating the total production over 8 minutes -/
theorem total_production_8_minutes :
  totalProduction initialState 2 +
  totalProduction stateAfter2Min 2 +
  totalProduction stateAfter4Min 4 = 27080 := by
  sorry


end NUMINAMATH_CALUDE_total_production_8_minutes_l3207_320774


namespace NUMINAMATH_CALUDE_harry_earnings_theorem_l3207_320761

/-- Harry's weekly dog-walking earnings -/
def harry_weekly_earnings : ℕ :=
  let monday_wednesday_friday_dogs := 7
  let tuesday_dogs := 12
  let thursday_dogs := 9
  let pay_per_dog := 5
  let days_with_7_dogs := 3
  
  (monday_wednesday_friday_dogs * days_with_7_dogs + tuesday_dogs + thursday_dogs) * pay_per_dog

/-- Theorem stating Harry's weekly earnings -/
theorem harry_earnings_theorem : harry_weekly_earnings = 210 := by
  sorry

end NUMINAMATH_CALUDE_harry_earnings_theorem_l3207_320761


namespace NUMINAMATH_CALUDE_least_positive_integer_to_multiple_of_four_l3207_320790

theorem least_positive_integer_to_multiple_of_four (n : ℕ) : 
  (∀ m : ℕ, m < n → ¬((530 + m) % 4 = 0)) ∧ ((530 + n) % 4 = 0) → n = 2 := by
  sorry

end NUMINAMATH_CALUDE_least_positive_integer_to_multiple_of_four_l3207_320790


namespace NUMINAMATH_CALUDE_total_boys_in_assembly_l3207_320756

/-- Represents the assembly of boys in two rows --/
structure Assembly where
  first_row : ℕ
  second_row : ℕ

/-- The position of a boy in a row --/
structure Position where
  from_left : ℕ
  from_right : ℕ

/-- Represents the assembly with given conditions --/
def school_assembly : Assembly where
  first_row := 24
  second_row := 24

/-- Rajan's position in the first row --/
def rajan_position : Position where
  from_left := 6
  from_right := school_assembly.first_row - 5

/-- Vinay's position in the first row --/
def vinay_position : Position where
  from_left := school_assembly.first_row - 9
  from_right := 10

/-- Number of boys between Rajan and Vinay --/
def boys_between : ℕ := 8

/-- Suresh's position in the second row --/
def suresh_position : Position where
  from_left := 5
  from_right := school_assembly.second_row - 4

theorem total_boys_in_assembly :
  school_assembly.first_row + school_assembly.second_row = 48 ∧
  school_assembly.first_row = school_assembly.second_row ∧
  rajan_position.from_left = 6 ∧
  vinay_position.from_right = 10 ∧
  vinay_position.from_left - rajan_position.from_left - 1 = boys_between ∧
  suresh_position.from_left = 5 :=
by sorry

end NUMINAMATH_CALUDE_total_boys_in_assembly_l3207_320756


namespace NUMINAMATH_CALUDE_bumper_car_queue_l3207_320737

/-- Calculates the final number of people in a queue after a given time period,
    given an initial number of people and a constant rate of change. -/
def final_queue_size (initial : ℕ) (net_change : ℕ) (interval : ℕ) (total_time : ℕ) : ℕ :=
  initial + (total_time / interval) * net_change

/-- Proves that given an initial queue of 12 people, with a net increase of 1 person
    every 5 minutes over the course of 1 hour, the final number of people in the queue will be 24. -/
theorem bumper_car_queue : final_queue_size 12 1 5 60 = 24 := by
  sorry

end NUMINAMATH_CALUDE_bumper_car_queue_l3207_320737


namespace NUMINAMATH_CALUDE_complement_of_A_l3207_320745

theorem complement_of_A (U A : Set ℕ) : 
  U = {1, 2, 3, 4, 5, 6} → 
  A = {3, 4, 5} → 
  Aᶜ = {1, 2, 6} := by
sorry

end NUMINAMATH_CALUDE_complement_of_A_l3207_320745


namespace NUMINAMATH_CALUDE_homework_ratio_proof_l3207_320720

/-- Given a total of 15 problems and 6 problems finished, 
    prove that the simplified ratio of problems still to complete 
    to problems already finished is 3:2. -/
theorem homework_ratio_proof (total : ℕ) (finished : ℕ) 
    (h1 : total = 15) (h2 : finished = 6) : 
    (total - finished) / Nat.gcd (total - finished) finished = 3 ∧ 
    finished / Nat.gcd (total - finished) finished = 2 := by
  sorry

end NUMINAMATH_CALUDE_homework_ratio_proof_l3207_320720


namespace NUMINAMATH_CALUDE_period_2_gym_class_size_l3207_320721

theorem period_2_gym_class_size :
  ∀ (period_2_size : ℕ),
  (2 * period_2_size - 5 = 11) →
  period_2_size = 8 := by
sorry

end NUMINAMATH_CALUDE_period_2_gym_class_size_l3207_320721


namespace NUMINAMATH_CALUDE_thirtieth_term_of_sequence_l3207_320736

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

theorem thirtieth_term_of_sequence (a₁ a₂ a₃ : ℝ) (h1 : a₁ = 3) (h2 : a₂ = 13) (h3 : a₃ = 23) :
  arithmetic_sequence a₁ (a₂ - a₁) 30 = 293 := by
sorry

end NUMINAMATH_CALUDE_thirtieth_term_of_sequence_l3207_320736


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3207_320746

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (∀ a b : ℝ, a > b ∧ b > 0 → abs a > abs b) ∧
  (∃ a b : ℝ, abs a > abs b ∧ ¬(a > b ∧ b > 0)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3207_320746


namespace NUMINAMATH_CALUDE_bus_dispatch_theorem_l3207_320724

/-- Represents the bus dispatch problem -/
structure BusDispatchProblem where
  initial_buses : ℕ := 15
  dispatch_interval : ℕ := 6
  entry_interval : ℕ := 8
  entry_delay : ℕ := 3
  total_time : ℕ := 840

/-- Calculates the time when the parking lot is first empty -/
def first_empty_time (problem : BusDispatchProblem) : ℕ :=
  sorry

/-- Calculates the time when buses can no longer be dispatched on time -/
def dispatch_failure_time (problem : BusDispatchProblem) : ℕ :=
  sorry

/-- Calculates the delay for the first bus that can't be dispatched on time -/
def first_delay_time (problem : BusDispatchProblem) : ℕ :=
  sorry

/-- Calculates the minimum interval for continuous dispatching -/
def min_continuous_interval (problem : BusDispatchProblem) : ℕ :=
  sorry

/-- Calculates the minimum number of additional buses needed for 6-minute interval dispatching -/
def min_additional_buses (problem : BusDispatchProblem) : ℕ :=
  sorry

theorem bus_dispatch_theorem (problem : BusDispatchProblem) :
  first_empty_time problem = 330 ∧
  dispatch_failure_time problem = 354 ∧
  first_delay_time problem = 1 ∧
  min_continuous_interval problem = 8 ∧
  min_additional_buses problem = 22 := by
  sorry

end NUMINAMATH_CALUDE_bus_dispatch_theorem_l3207_320724


namespace NUMINAMATH_CALUDE_all_terms_perfect_squares_l3207_320755

/-- A sequence of integers satisfying specific conditions -/
def SpecialSequence (a : ℕ → ℤ) : Prop :=
  (∀ n ≥ 2, a (n + 1) = 3 * a n - 3 * a (n - 1) + a (n - 2)) ∧
  (2 * a 1 = a 0 + a 2 - 2) ∧
  (∀ m : ℕ, ∃ k : ℕ, ∀ i < m, ∃ j : ℤ, a (k + i) = j ^ 2)

/-- All terms in the special sequence are perfect squares -/
theorem all_terms_perfect_squares (a : ℕ → ℤ) (h : SpecialSequence a) :
  ∀ n : ℕ, ∃ k : ℤ, a n = k ^ 2 :=
by sorry

end NUMINAMATH_CALUDE_all_terms_perfect_squares_l3207_320755


namespace NUMINAMATH_CALUDE_five_leaders_three_cities_l3207_320770

/-- The number of ways to allocate n leaders to k cities, with each city having at least one leader -/
def allocationSchemes (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem stating that allocating 5 leaders to 3 cities results in 240 schemes -/
theorem five_leaders_three_cities : allocationSchemes 5 3 = 240 := by sorry

end NUMINAMATH_CALUDE_five_leaders_three_cities_l3207_320770


namespace NUMINAMATH_CALUDE_bill_total_is_95_l3207_320701

/-- Represents a person's order at the restaurant -/
structure Order where
  appetizer_share : ℚ
  drinks_cost : ℚ
  dessert_cost : ℚ

/-- Calculates the total cost of an order -/
def total_cost (order : Order) : ℚ :=
  order.appetizer_share + order.drinks_cost + order.dessert_cost

/-- Represents the restaurant bill -/
def restaurant_bill (mary nancy fred steve : Order) : Prop :=
  let appetizer_total : ℚ := 28
  let appetizer_share : ℚ := appetizer_total / 4
  mary.appetizer_share = appetizer_share ∧
  nancy.appetizer_share = appetizer_share ∧
  fred.appetizer_share = appetizer_share ∧
  steve.appetizer_share = appetizer_share ∧
  mary.drinks_cost = 14 ∧
  nancy.drinks_cost = 11 ∧
  fred.drinks_cost = 12 ∧
  steve.drinks_cost = 6 ∧
  mary.dessert_cost = 8 ∧
  nancy.dessert_cost = 0 ∧
  fred.dessert_cost = 10 ∧
  steve.dessert_cost = 6

theorem bill_total_is_95 (mary nancy fred steve : Order) 
  (h : restaurant_bill mary nancy fred steve) : 
  total_cost mary + total_cost nancy + total_cost fred + total_cost steve = 95 := by
  sorry

end NUMINAMATH_CALUDE_bill_total_is_95_l3207_320701


namespace NUMINAMATH_CALUDE_parabola_property_l3207_320733

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the focus F
def focus : ℝ × ℝ := (2, 0)

-- Define the directrix l
def directrix : ℝ → Prop := λ x => x = -2

-- Define a point on the parabola
def point_on_parabola (P : ℝ × ℝ) : Prop :=
  parabola P.1 P.2

-- Define the perpendicular condition
def perpendicular_to_directrix (P A : ℝ × ℝ) : Prop :=
  directrix A.1 ∧ (P.2 = A.2)

-- Define the slope condition for AF
def slope_AF_is_neg_one (A : ℝ × ℝ) : Prop :=
  (A.2 - focus.2) / (A.1 - focus.1) = -1

theorem parabola_property :
  ∀ P : ℝ × ℝ,
  point_on_parabola P →
  ∃ A : ℝ × ℝ,
  perpendicular_to_directrix P A ∧
  slope_AF_is_neg_one A →
  Real.sqrt ((P.1 - focus.1)^2 + (P.2 - focus.2)^2) = 4 :=
sorry

end NUMINAMATH_CALUDE_parabola_property_l3207_320733


namespace NUMINAMATH_CALUDE_equation_solution_l3207_320792

theorem equation_solution (x : ℝ) : 
  x ≠ 2 → x ≠ -2 → 
  ((3 - x^2) / (x + 2) + (2*x^2 - 8) / (x^2 - 4) = 3) ↔ 
  (x = (-1 + Real.sqrt 5) / 2 ∨ x = (-1 - Real.sqrt 5) / 2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3207_320792


namespace NUMINAMATH_CALUDE_logarithm_expression_equality_l3207_320717

-- Define the logarithm base 10 function
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem logarithm_expression_equality :
  lg (4 * Real.sqrt 2 / 7) - lg (2 / 3) + lg (7 * Real.sqrt 5) = lg 6 + 1/2 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_expression_equality_l3207_320717


namespace NUMINAMATH_CALUDE_turtleneck_sweater_profit_profit_percentage_l3207_320773

theorem turtleneck_sweater_profit (C : ℝ) : 
  let first_markup := C * 1.20
  let second_markup := first_markup * 1.25
  let final_price := second_markup * 0.88
  final_price = C * 1.32 := by sorry

theorem profit_percentage (C : ℝ) : 
  let first_markup := C * 1.20
  let second_markup := first_markup * 1.25
  let final_price := second_markup * 0.88
  (final_price - C) / C = 0.32 := by sorry

end NUMINAMATH_CALUDE_turtleneck_sweater_profit_profit_percentage_l3207_320773


namespace NUMINAMATH_CALUDE_vector_orthogonality_l3207_320795

theorem vector_orthogonality (a b : ℝ × ℝ) (ha : a ≠ (0, 0)) (hb : b ≠ (0, 0)) :
  ‖a + b‖ = ‖a - b‖ → a • b = 0 :=
by sorry

end NUMINAMATH_CALUDE_vector_orthogonality_l3207_320795


namespace NUMINAMATH_CALUDE_dog_age_difference_l3207_320749

/-- Proves that the 5th fastest dog is 20 years older than the 4th fastest dog --/
theorem dog_age_difference :
  let dog1_age : ℕ := 10
  let dog2_age : ℕ := dog1_age - 2
  let dog3_age : ℕ := dog2_age + 4
  let dog4_age : ℕ := dog3_age / 2
  let dog5_age : ℕ := dog4_age + 20
  (dog1_age + dog5_age) / 2 = 18 →
  dog5_age - dog4_age = 20 := by
sorry

end NUMINAMATH_CALUDE_dog_age_difference_l3207_320749


namespace NUMINAMATH_CALUDE_diamond_example_l3207_320707

/-- Diamond operation for real numbers -/
def diamond (a b : ℝ) : ℝ := (a + b) * (a - b) + a

/-- Theorem stating that 2 ◊ (3 ◊ 4) = -10 -/
theorem diamond_example : diamond 2 (diamond 3 4) = -10 := by
  sorry

end NUMINAMATH_CALUDE_diamond_example_l3207_320707


namespace NUMINAMATH_CALUDE_average_age_of_contestants_l3207_320783

/-- Represents an age in years and months -/
structure Age where
  years : ℕ
  months : ℕ
  valid : months < 12

/-- Converts an age to total months -/
def ageToMonths (a : Age) : ℕ := a.years * 12 + a.months

/-- Converts total months to an age -/
def monthsToAge (m : ℕ) : Age :=
  { years := m / 12
  , months := m % 12
  , valid := by exact Nat.mod_lt m (by norm_num) }

/-- Calculates the average age of three contestants -/
def averageAge (a1 a2 a3 : Age) : Age :=
  monthsToAge ((ageToMonths a1 + ageToMonths a2 + ageToMonths a3) / 3)

theorem average_age_of_contestants :
  let age1 : Age := { years := 15, months := 9, valid := by norm_num }
  let age2 : Age := { years := 16, months := 1, valid := by norm_num }
  let age3 : Age := { years := 15, months := 8, valid := by norm_num }
  let avgAge := averageAge age1 age2 age3
  avgAge.years = 15 ∧ avgAge.months = 10 := by
  sorry

end NUMINAMATH_CALUDE_average_age_of_contestants_l3207_320783


namespace NUMINAMATH_CALUDE_expand_expression_l3207_320799

theorem expand_expression (x : ℝ) : 24 * (3 * x + 4 - 2) = 72 * x + 48 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l3207_320799


namespace NUMINAMATH_CALUDE_diamond_720_1001_cubed_l3207_320766

/-- The diamond operation on positive integers -/
def diamond (a b : ℕ+) : ℕ := sorry

/-- The number of distinct prime factors of a positive integer -/
def num_distinct_prime_factors (n : ℕ+) : ℕ := sorry

theorem diamond_720_1001_cubed : 
  (diamond 720 1001)^3 = 216 := by sorry

end NUMINAMATH_CALUDE_diamond_720_1001_cubed_l3207_320766


namespace NUMINAMATH_CALUDE_total_time_remaining_wallpaper_l3207_320713

/-- Represents the time in hours to remove wallpaper from one wall -/
def time_per_wall : ℕ := 2

/-- Represents the number of walls in the dining room -/
def dining_room_walls : ℕ := 4

/-- Represents the number of walls in the living room -/
def living_room_walls : ℕ := 4

/-- Represents the number of walls already completed in the dining room -/
def completed_walls : ℕ := 1

/-- Theorem stating the total time to remove wallpaper from the remaining walls -/
theorem total_time_remaining_wallpaper :
  time_per_wall * (dining_room_walls + living_room_walls - completed_walls) = 14 := by
  sorry

end NUMINAMATH_CALUDE_total_time_remaining_wallpaper_l3207_320713


namespace NUMINAMATH_CALUDE_pauls_cousin_score_l3207_320716

/-- Given Paul's score and the total score of Paul and his cousin, 
    calculate Paul's cousin's score. -/
theorem pauls_cousin_score (paul_score total_score : ℕ) 
  (h1 : paul_score = 3103)
  (h2 : total_score = 5816) :
  total_score - paul_score = 2713 := by
  sorry

end NUMINAMATH_CALUDE_pauls_cousin_score_l3207_320716


namespace NUMINAMATH_CALUDE_minimum_value_of_reciprocal_sum_l3207_320742

theorem minimum_value_of_reciprocal_sum (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  let a : ℝ × ℝ := (m, 1 - n)
  let b : ℝ × ℝ := (1, 2)
  (∃ (k : ℝ), a = k • b) →
  (∀ (x y : ℝ), x > 0 → y > 0 → 1/x + 1/y ≥ 1/m + 1/n) →
  1/m + 1/n = 3 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_minimum_value_of_reciprocal_sum_l3207_320742


namespace NUMINAMATH_CALUDE_jovanas_shells_l3207_320748

/-- The total amount of shells Jovana has after her friends add to her collection -/
def total_shells (initial : ℕ) (friend1 : ℕ) (friend2 : ℕ) : ℕ :=
  initial + friend1 + friend2

/-- Theorem stating that Jovana's total shells equal 37 pounds -/
theorem jovanas_shells :
  total_shells 5 15 17 = 37 := by
  sorry

end NUMINAMATH_CALUDE_jovanas_shells_l3207_320748


namespace NUMINAMATH_CALUDE_max_a_for_increasing_f_l3207_320787

def f (x : ℝ) : ℝ := -x^2 + 2*x - 2

theorem max_a_for_increasing_f :
  (∃ a : ℝ, ∀ x₁ x₂ : ℝ, x₁ ≤ x₂ ∧ x₂ ≤ a → f x₁ ≤ f x₂) ∧
  (∀ b : ℝ, b > 1 → ∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₂ ≤ b ∧ f x₁ > f x₂) :=
by sorry

end NUMINAMATH_CALUDE_max_a_for_increasing_f_l3207_320787


namespace NUMINAMATH_CALUDE_problem_solution_l3207_320752

theorem problem_solution (a b : ℚ) 
  (h1 : a + b = 8/15) 
  (h2 : a - b = 2/15) : 
  a^2 - b^2 = 16/225 ∧ a * b = 1/25 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3207_320752


namespace NUMINAMATH_CALUDE_regular_octagon_diagonal_ratio_l3207_320718

/-- The ratio of the shortest diagonal to the longest diagonal in a regular octagon -/
theorem regular_octagon_diagonal_ratio : 
  ∃ (shortest_diagonal longest_diagonal : ℝ), 
    shortest_diagonal > 0 ∧ 
    longest_diagonal > 0 ∧
    shortest_diagonal / longest_diagonal = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_regular_octagon_diagonal_ratio_l3207_320718


namespace NUMINAMATH_CALUDE_prob_odd_top_face_l3207_320727

/-- The number of sides on the die -/
def num_sides : ℕ := 12

/-- The total number of dots on the die initially -/
def total_dots : ℕ := (num_sides * (num_sides + 1)) / 2

/-- The number of ways to choose 2 dots from the total -/
def ways_to_choose_two_dots : ℕ := total_dots.choose 2

/-- The probability of rolling a specific face -/
def prob_single_face : ℚ := 1 / num_sides

/-- The sum of even numbers from 2 to 12 -/
def sum_even_faces : ℕ := 2 + 4 + 6 + 8 + 10 + 12

/-- Theorem: The probability of rolling an odd number of dots on the top face
    of a 12-sided die, after randomly removing two dots, is 7/3003 -/
theorem prob_odd_top_face : 
  (prob_single_face * (2 * sum_even_faces : ℚ)) / ways_to_choose_two_dots = 7 / 3003 :=
sorry

end NUMINAMATH_CALUDE_prob_odd_top_face_l3207_320727


namespace NUMINAMATH_CALUDE_water_beaker_problem_l3207_320796

theorem water_beaker_problem (s h : ℚ) :
  s - 7/3 = h + 7/3 + 3/2 →
  s - h = 37/6 := by
  sorry

end NUMINAMATH_CALUDE_water_beaker_problem_l3207_320796


namespace NUMINAMATH_CALUDE_only_C_is_perfect_square_l3207_320747

-- Define the expressions
def expr_A : ℕ := 3^3 * 4^5 * 7^7
def expr_B : ℕ := 3^4 * 4^4 * 7^5
def expr_C : ℕ := 3^6 * 4^3 * 7^6
def expr_D : ℕ := 3^5 * 4^6 * 7^4
def expr_E : ℕ := 3^4 * 4^6 * 7^6

-- Define a perfect square
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m^2

-- Theorem statement
theorem only_C_is_perfect_square :
  is_perfect_square expr_C ∧
  ¬is_perfect_square expr_A ∧
  ¬is_perfect_square expr_B ∧
  ¬is_perfect_square expr_D ∧
  ¬is_perfect_square expr_E :=
sorry

end NUMINAMATH_CALUDE_only_C_is_perfect_square_l3207_320747


namespace NUMINAMATH_CALUDE_geometric_series_problem_l3207_320732

theorem geometric_series_problem (q : ℝ) (b₁ : ℝ) (h₁ : |q| < 1) 
  (h₂ : b₁ / (1 - q) = 16) (h₃ : b₁^2 / (1 - q^2) = 153.6) :
  b₁ * q^3 = 3/16 ∧ q = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_problem_l3207_320732


namespace NUMINAMATH_CALUDE_log_base_three_squared_l3207_320776

theorem log_base_three_squared (m : ℝ) (b : ℝ) (h : 3^m = b) :
  Real.log b / Real.log (3^2) = m / 2 := by
  sorry

end NUMINAMATH_CALUDE_log_base_three_squared_l3207_320776


namespace NUMINAMATH_CALUDE_log_sum_sqrt_equality_l3207_320729

theorem log_sum_sqrt_equality :
  Real.sqrt (Real.log 12 / Real.log 3 + Real.log 12 / Real.log 4) =
  Real.sqrt (Real.log 3 / Real.log 4) + Real.sqrt (Real.log 4 / Real.log 3) := by
  sorry

end NUMINAMATH_CALUDE_log_sum_sqrt_equality_l3207_320729


namespace NUMINAMATH_CALUDE_city_fuel_efficiency_l3207_320715

/-- Represents the fuel efficiency of a car in miles per gallon -/
structure FuelEfficiency where
  highway : ℝ
  city : ℝ

/-- Represents the distance a car can travel on a full tank in miles -/
structure TankDistance where
  highway : ℝ
  city : ℝ

theorem city_fuel_efficiency 
  (fe : FuelEfficiency) 
  (td : TankDistance) 
  (h1 : fe.city = fe.highway - 9)
  (h2 : td.highway = 462)
  (h3 : td.city = 336)
  (h4 : fe.highway * (td.city / fe.city) = td.highway) :
  fe.city = 24 := by
  sorry

end NUMINAMATH_CALUDE_city_fuel_efficiency_l3207_320715


namespace NUMINAMATH_CALUDE_sum_of_powers_equals_product_l3207_320744

theorem sum_of_powers_equals_product (n : ℕ) : 
  5^n + 5^n + 5^n + 5^n = 4 * 5^n := by sorry

end NUMINAMATH_CALUDE_sum_of_powers_equals_product_l3207_320744


namespace NUMINAMATH_CALUDE_m_range_l3207_320714

/-- Given conditions p and q, prove that the range of real numbers m is [-2, -1). -/
theorem m_range (p : ∀ x : ℝ, 2 * x > m * (x^2 + 1))
                (q : ∃ x₀ : ℝ, x₀^2 + 2 * x₀ - m - 1 = 0) :
  m ≥ -2 ∧ m < -1 :=
by sorry

end NUMINAMATH_CALUDE_m_range_l3207_320714


namespace NUMINAMATH_CALUDE_eleven_hash_five_l3207_320709

/-- The # operation on real numbers -/
noncomputable def hash (r s : ℝ) : ℝ :=
  sorry

/-- Properties of the # operation -/
axiom hash_zero (r : ℝ) : hash r 0 = r
axiom hash_comm (r s : ℝ) : hash r s = hash s r
axiom hash_succ (r s : ℝ) : hash (r + 1) s = hash r s + s + 1

/-- The main theorem to prove -/
theorem eleven_hash_five : hash 11 5 = 71 := by
  sorry

end NUMINAMATH_CALUDE_eleven_hash_five_l3207_320709


namespace NUMINAMATH_CALUDE_area_of_intersection_l3207_320760

/-- Given two overlapping rectangles ABNF and CMKD, prove the area of their intersection MNFK --/
theorem area_of_intersection (BN KD : ℝ) (area_ABMK area_CDFN : ℝ) :
  BN = 8 →
  KD = 9 →
  area_ABMK = 25 →
  area_CDFN = 32 →
  ∃ (AB CD : ℝ),
    AB * BN - area_ABMK = CD * KD - area_CDFN ∧
    AB * BN - area_ABMK = 31 :=
by sorry

end NUMINAMATH_CALUDE_area_of_intersection_l3207_320760


namespace NUMINAMATH_CALUDE_successive_integers_product_l3207_320768

theorem successive_integers_product (n : ℕ) : 
  n * (n + 1) = 7832 → n = 88 := by sorry

end NUMINAMATH_CALUDE_successive_integers_product_l3207_320768


namespace NUMINAMATH_CALUDE_order_relation_l3207_320726

theorem order_relation (a b c : ℝ) (ha : a = Real.log (1 + Real.exp 1))
    (hb : b = Real.sqrt (Real.exp 1)) (hc : c = (2 * Real.exp 1) / 3) :
  a < b ∧ b < c := by sorry

end NUMINAMATH_CALUDE_order_relation_l3207_320726
