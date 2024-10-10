import Mathlib

namespace mathopolis_intersections_l1028_102890

/-- A city with a grid-like street layout. -/
structure City where
  ns_streets : ℕ  -- Number of north-south streets
  ew_streets : ℕ  -- Number of east-west streets

/-- The number of intersections in a city with a grid-like street layout. -/
def num_intersections (c : City) : ℕ := c.ns_streets * c.ew_streets

/-- Mathopolis with its specific street layout. -/
def mathopolis : City := { ns_streets := 10, ew_streets := 10 }

/-- Theorem stating that Mathopolis has 100 intersections. -/
theorem mathopolis_intersections : num_intersections mathopolis = 100 := by
  sorry

#eval num_intersections mathopolis

end mathopolis_intersections_l1028_102890


namespace pascals_triangle_25th_number_l1028_102849

theorem pascals_triangle_25th_number (n : ℕ) (k : ℕ) : 
  n = 27 ∧ k = 24 → Nat.choose n k = 2925 :=
by
  sorry

end pascals_triangle_25th_number_l1028_102849


namespace arithmetic_sequence_formula_l1028_102861

def arithmeticSequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_formula
  (a : ℕ → ℝ)
  (d : ℝ)
  (h_arith : arithmeticSequence a d)
  (h_d_neg : d < 0)
  (h_prod : a 2 * a 4 = 12)
  (h_sum : a 2 + a 4 = 8) :
  ∀ n : ℕ+, a n = -2 * n + 10 := by
sorry

end arithmetic_sequence_formula_l1028_102861


namespace unique_c_complex_magnitude_l1028_102838

theorem unique_c_complex_magnitude : ∃! c : ℝ, Complex.abs (1 - (c + 1) * Complex.I) = 1 := by
  sorry

end unique_c_complex_magnitude_l1028_102838


namespace six_lines_six_intersections_l1028_102893

/-- A point in the plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- A line in the plane -/
structure Line :=
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)

/-- Check if a point lies on a line -/
def Point.onLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines intersect -/
def Line.intersect (l1 l2 : Line) : Prop :=
  l1.a * l2.b ≠ l1.b * l2.a

/-- A configuration of six lines -/
structure SixLineConfig :=
  (lines : Fin 6 → Line)

/-- Count the number of intersection points in a configuration -/
def SixLineConfig.intersectionCount (config : SixLineConfig) : ℕ :=
  sorry

/-- Theorem: There exists a configuration of six lines with exactly six intersection points -/
theorem six_lines_six_intersections :
  ∃ (config : SixLineConfig), config.intersectionCount = 6 :=
sorry

end six_lines_six_intersections_l1028_102893


namespace sin_alpha_eq_neg_half_l1028_102842

theorem sin_alpha_eq_neg_half (α : Real) 
  (h : Real.sin (α/2 - Real.pi/4) * Real.cos (α/2 + Real.pi/4) = -3/4) : 
  Real.sin α = -1/2 := by
  sorry

end sin_alpha_eq_neg_half_l1028_102842


namespace share_calculation_l1028_102872

/-- Represents the share of each party in rupees -/
structure Share where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The problem statement -/
theorem share_calculation (s : Share) : 
  s.x + s.y + s.z = 175 →  -- Total sum is 175
  s.z = 0.3 * s.x →        -- z gets 0.3 for each rupee x gets
  s.x > 0 →                -- Ensure x's share is positive
  s.y = 173.7 :=           -- y's share is 173.7
by sorry

end share_calculation_l1028_102872


namespace hcf_problem_l1028_102896

theorem hcf_problem (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  a + b = 55 →
  Nat.lcm a b = 120 →
  (1 : ℚ) / a + (1 : ℚ) / b = 11 / 120 →
  Nat.gcd a b = 5 := by
sorry

end hcf_problem_l1028_102896


namespace min_values_theorem_l1028_102848

theorem min_values_theorem (a b c : ℕ+) (h : a^2 + b^2 - c^2 = 2018) :
  (∀ x y z : ℕ+, x^2 + y^2 - z^2 = 2018 → a + b - c ≤ x + y - z) ∧
  (∀ x y z : ℕ+, x^2 + y^2 - z^2 = 2018 → a + b + c ≤ x + y + z) ∧
  a + b - c = 2 ∧ a + b + c = 52 :=
sorry

end min_values_theorem_l1028_102848


namespace smallest_class_size_l1028_102895

theorem smallest_class_size (n : ℕ) : 
  n > 0 ∧ 
  (6 * 120 + (n - 6) * 70 : ℝ) ≤ (n * 85 : ℝ) ∧ 
  (∀ m : ℕ, m > 0 → m < n → (6 * 120 + (m - 6) * 70 : ℝ) > (m * 85 : ℝ)) → 
  n = 20 := by
sorry

end smallest_class_size_l1028_102895


namespace coefficient_x2y2_l1028_102897

-- Define the binomial coefficient function
def binomial (n k : ℕ) : ℕ := sorry

-- Define the expansion of (1+x)^7(1+y)^4
def expansion : ℕ → ℕ → ℕ := sorry

-- Theorem statement
theorem coefficient_x2y2 : expansion 2 2 = 126 := by sorry

end coefficient_x2y2_l1028_102897


namespace max_value_of_f_l1028_102882

def f (x : ℝ) := -x^2 + 6*x - 10

theorem max_value_of_f :
  ∃ (m : ℝ), m = -1 ∧
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 4 → f x ≤ m) ∧
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ 4 ∧ f x = m) :=
by sorry

end max_value_of_f_l1028_102882


namespace diamond_op_five_three_l1028_102877

def diamond_op (x y : ℝ) : ℝ := 4 * x + 6 * y

theorem diamond_op_five_three : diamond_op 5 3 = 38 := by
  sorry

end diamond_op_five_three_l1028_102877


namespace rectangular_parallelepiped_surface_area_l1028_102878

theorem rectangular_parallelepiped_surface_area 
  (x y z : ℝ) 
  (h1 : (x + 1) * (y + 1) * (z + 1) - x * y * z = 18)
  (h2 : 2 * ((x + 1) * (y + 1) + (y + 1) * (z + 1) + (z + 1) * (x + 1)) - 2 * (x * y + x * z + y * z) = 30) :
  2 * (x * y + x * z + y * z) = 22 := by
sorry

end rectangular_parallelepiped_surface_area_l1028_102878


namespace cd_cost_fraction_l1028_102885

theorem cd_cost_fraction (m : ℝ) (n : ℕ) (h : n > 0) : 
  let total_cd_cost : ℝ := 2 * (1/3 * m)
  let cd_cost : ℝ := total_cd_cost / n
  let savings : ℝ := m - total_cd_cost
  (1/3 * m = (1/2 * n) * (cd_cost)) ∧ 
  (savings ≥ 1/4 * m) →
  cd_cost = 1/3 * m := by
sorry

end cd_cost_fraction_l1028_102885


namespace vector_parallel_condition_l1028_102858

/-- Given vectors in R², prove that if they satisfy certain conditions, then x = 1/2 -/
theorem vector_parallel_condition (x : ℝ) : 
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![x, 1]
  let u : Fin 2 → ℝ := a + 2 • b
  let v : Fin 2 → ℝ := 2 • a - b
  (∃ (k : ℝ), k ≠ 0 ∧ u = k • v) → x = 1/2 := by
  sorry

end vector_parallel_condition_l1028_102858


namespace total_soccer_balls_l1028_102806

/-- Represents a school with elementary and middle school classes --/
structure School where
  elementary_classes : ℕ
  middle_classes : ℕ
  elementary_students : List ℕ
  middle_students : List ℕ

/-- Calculates the number of soccer balls for a given number of students in an elementary class --/
def elementary_balls (students : ℕ) : ℕ :=
  if students ≤ 30 then 4 else 5

/-- Calculates the number of soccer balls for a given number of students in a middle school class --/
def middle_balls (students : ℕ) : ℕ :=
  if students ≤ 24 then 6 else 7

/-- Calculates the total number of soccer balls for a school --/
def school_balls (school : School) : ℕ :=
  (school.elementary_students.map elementary_balls).sum +
  (school.middle_students.map middle_balls).sum

/-- The three schools as described in the problem --/
def school_A : School :=
  { elementary_classes := 4
  , middle_classes := 5
  , elementary_students := List.replicate 4 28
  , middle_students := List.replicate 5 25 }

def school_B : School :=
  { elementary_classes := 5
  , middle_classes := 3
  , elementary_students := [32, 32, 32, 30, 30]
  , middle_students := [22, 22, 26] }

def school_C : School :=
  { elementary_classes := 6
  , middle_classes := 4
  , elementary_students := [30, 30, 30, 30, 31, 31]
  , middle_students := List.replicate 4 24 }

/-- The main theorem stating that the total number of soccer balls donated is 143 --/
theorem total_soccer_balls :
  school_balls school_A + school_balls school_B + school_balls school_C = 143 := by
  sorry


end total_soccer_balls_l1028_102806


namespace connors_garage_wheels_l1028_102813

/-- The number of wheels in Connor's garage -/
def total_wheels (bicycles cars motorcycles : ℕ) : ℕ :=
  2 * bicycles + 4 * cars + 2 * motorcycles

/-- Theorem: The total number of wheels in Connor's garage is 90 -/
theorem connors_garage_wheels :
  total_wheels 20 10 5 = 90 := by
  sorry

end connors_garage_wheels_l1028_102813


namespace max_correct_is_23_l1028_102871

/-- Represents the scoring system and Amy's exam results -/
structure ExamResults where
  total_questions : ℕ
  correct_score : ℤ
  incorrect_score : ℤ
  total_score : ℤ

/-- Calculates the maximum number of correctly answered questions -/
def max_correct_answers (exam : ExamResults) : ℕ :=
  sorry

/-- Theorem stating that given the exam conditions, the maximum number of correct answers is 23 -/
theorem max_correct_is_23 (exam : ExamResults) 
  (h1 : exam.total_questions = 30)
  (h2 : exam.correct_score = 4)
  (h3 : exam.incorrect_score = -1)
  (h4 : exam.total_score = 85) :
  max_correct_answers exam = 23 :=
sorry

end max_correct_is_23_l1028_102871


namespace symmetric_parabolas_product_l1028_102851

/-- Given two parabolas that are symmetric with respect to a line, 
    prove that the product of their parameters is -3 -/
theorem symmetric_parabolas_product (a p m : ℝ) : 
  a ≠ 0 → p > 0 → 
  (∀ x y : ℝ, y = a * x^2 - 3 * x + 3 ↔ 
    ∃ x' y', y' = x + m ∧ x = y' - m ∧ y'^2 = 2 * p * x') →
  a * p * m = -3 := by
  sorry

end symmetric_parabolas_product_l1028_102851


namespace complex_equality_l1028_102867

theorem complex_equality (z : ℂ) : z = -1 + I ↔ Complex.abs (z - 2) = Complex.abs (z + 4) ∧ Complex.abs (z - 2) = Complex.abs (z - 2*I) := by
  sorry

end complex_equality_l1028_102867


namespace part_one_part_two_l1028_102841

-- Define the new operation *
def star (a b : ℚ) : ℚ := 4 * a * b

-- Theorem for part (1)
theorem part_one : star 3 (-4) = -48 := by sorry

-- Theorem for part (2)
theorem part_two : star (-2) (star 6 3) = -576 := by sorry

end part_one_part_two_l1028_102841


namespace investment_ratio_proof_l1028_102828

/-- Represents the investment and return for an investor -/
structure Investor where
  investment : ℝ
  returnRate : ℝ

/-- Proves that the ratio of investments is 6:5:4 given the problem conditions -/
theorem investment_ratio_proof 
  (a b c : Investor)
  (return_ratio : a.returnRate / b.returnRate = 6/5 ∧ b.returnRate / c.returnRate = 5/4)
  (b_earns_more : b.investment * b.returnRate = a.investment * a.returnRate + 100)
  (total_earnings : a.investment * a.returnRate + b.investment * b.returnRate + c.investment * c.returnRate = 2900)
  : a.investment / b.investment = 6/5 ∧ b.investment / c.investment = 5/4 := by
  sorry


end investment_ratio_proof_l1028_102828


namespace work_completion_proof_l1028_102822

/-- The original number of men working on a task -/
def original_men : ℕ := 20

/-- The number of days it takes the original group to complete the work -/
def original_days : ℕ := 10

/-- The number of men removed from the original group -/
def removed_men : ℕ := 10

/-- The number of additional days it takes to complete the work with fewer men -/
def additional_days : ℕ := 10

theorem work_completion_proof :
  (original_men * original_days = (original_men - removed_men) * (original_days + additional_days)) →
  original_men = 20 := by
  sorry

end work_completion_proof_l1028_102822


namespace sin_B_value_l1028_102864

-- Define a right triangle ABC
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  right_angle : c^2 = a^2 + b^2

-- Define the given triangle
def given_triangle : RightTriangle where
  a := 3
  b := 4
  c := 5
  right_angle := by norm_num

-- Theorem to prove
theorem sin_B_value (triangle : RightTriangle) (h1 : triangle.a = 3) (h2 : triangle.b = 4) :
  Real.sin (Real.arcsin (triangle.b / triangle.c)) = 4/5 := by
  sorry

#check sin_B_value given_triangle rfl rfl

end sin_B_value_l1028_102864


namespace number_divided_by_three_l1028_102865

theorem number_divided_by_three : ∃ x : ℝ, x / 3 = 3 ∧ x = 9 := by
  sorry

end number_divided_by_three_l1028_102865


namespace simplify_expression_l1028_102808

theorem simplify_expression (x : ℝ) : (2*x - 5)*(x + 6) - (x + 4)*(2*x - 1) = -26 := by
  sorry

end simplify_expression_l1028_102808


namespace base7UnitsDigitIs6_l1028_102816

/-- The units digit of the base-7 representation of the product of 328 and 57 -/
def base7UnitsDigit : ℕ :=
  (328 * 57) % 7

/-- Theorem stating that the units digit of the base-7 representation of the product of 328 and 57 is 6 -/
theorem base7UnitsDigitIs6 : base7UnitsDigit = 6 := by
  sorry

end base7UnitsDigitIs6_l1028_102816


namespace wide_right_field_goals_l1028_102898

theorem wide_right_field_goals 
  (total_attempts : ℕ) 
  (missed_fraction : ℚ) 
  (wide_right_percentage : ℚ) : ℕ :=
by
  have h1 : total_attempts = 60 := by sorry
  have h2 : missed_fraction = 1 / 4 := by sorry
  have h3 : wide_right_percentage = 1 / 5 := by sorry
  
  let missed_goals := total_attempts * missed_fraction
  let wide_right_goals := missed_goals * wide_right_percentage
  
  exact 3
  
#check wide_right_field_goals

end wide_right_field_goals_l1028_102898


namespace rectangle_construction_solutions_l1028_102875

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle -/
structure Rectangle where
  topLeft : Point
  topRight : Point
  bottomRight : Point
  bottomLeft : Point

/-- Check if a point lies on any side of the rectangle -/
def pointOnRectangle (p : Point) (r : Rectangle) : Prop :=
  (p.x = r.topLeft.x ∧ p.y ≥ r.bottomLeft.y ∧ p.y ≤ r.topLeft.y) ∨
  (p.x = r.topRight.x ∧ p.y ≥ r.bottomRight.y ∧ p.y ≤ r.topRight.y) ∨
  (p.y = r.topLeft.y ∧ p.x ≥ r.topLeft.x ∧ p.x ≤ r.topRight.x) ∨
  (p.y = r.bottomLeft.y ∧ p.x ≥ r.bottomLeft.x ∧ p.x ≤ r.bottomRight.x)

/-- Check if the rectangle has a side of length 'a' -/
def hasLengthA (r : Rectangle) (a : ℝ) : Prop :=
  (r.topRight.x - r.topLeft.x = a) ∨
  (r.topRight.y - r.bottomRight.y = a)

/-- The main theorem -/
theorem rectangle_construction_solutions 
  (A B C D : Point) (a : ℝ) (h : a > 0) :
  ∃ (solutions : Finset Rectangle), 
    solutions.card = 12 ∧
    ∀ r ∈ solutions, 
      pointOnRectangle A r ∧
      pointOnRectangle B r ∧
      pointOnRectangle C r ∧
      pointOnRectangle D r ∧
      hasLengthA r a :=
sorry

end rectangle_construction_solutions_l1028_102875


namespace average_of_six_numbers_l1028_102826

theorem average_of_six_numbers (a b c d e f : ℝ) 
  (h1 : (a + b) / 2 = 2.4)
  (h2 : (c + d) / 2 = 2.3)
  (h3 : (e + f) / 2 = 3.7) :
  (a + b + c + d + e + f) / 6 = 2.8 := by
  sorry

end average_of_six_numbers_l1028_102826


namespace four_dice_same_number_probability_l1028_102870

/-- The probability of a single die showing a specific number -/
def single_die_prob : ℚ := 1 / 6

/-- The number of dice being tossed -/
def num_dice : ℕ := 4

/-- The probability of all dice showing the same number -/
def all_same_prob : ℚ := single_die_prob ^ (num_dice - 1)

theorem four_dice_same_number_probability :
  all_same_prob = 1 / 216 := by
  sorry

end four_dice_same_number_probability_l1028_102870


namespace quartic_sum_l1028_102846

/-- A quartic polynomial Q with specific values at 0, 1, and -1 -/
def QuarticPolynomial (m : ℝ) : ℝ → ℝ := sorry

/-- Properties of the QuarticPolynomial -/
axiom quartic_prop_0 (m : ℝ) : QuarticPolynomial m 0 = m
axiom quartic_prop_1 (m : ℝ) : QuarticPolynomial m 1 = 3 * m
axiom quartic_prop_neg1 (m : ℝ) : QuarticPolynomial m (-1) = 2 * m

/-- Theorem: For a quartic polynomial Q with Q(0) = m, Q(1) = 3m, and Q(-1) = 2m, Q(3) + Q(-3) = 56m -/
theorem quartic_sum (m : ℝ) : 
  QuarticPolynomial m 3 + QuarticPolynomial m (-3) = 56 * m := by sorry

end quartic_sum_l1028_102846


namespace m_range_l1028_102819

-- Define the condition function
def condition (m : ℝ) : Set ℝ := {x | 1 - m < x ∧ x < 1 + m}

-- Define the inequality function
def inequality : Set ℝ := {x | (x - 1)^2 < 1}

-- Theorem statement
theorem m_range :
  ∀ m : ℝ, (condition m ⊆ inequality ∧ condition m ≠ inequality) → m ∈ Set.Ioo 0 1 :=
by sorry

end m_range_l1028_102819


namespace annie_children_fruits_l1028_102888

/-- The number of fruits Annie's children received -/
def total_fruits (mike_oranges matt_apples mark_bananas : ℕ) : ℕ :=
  mike_oranges + matt_apples + mark_bananas

theorem annie_children_fruits :
  ∃ (mike_oranges matt_apples mark_bananas : ℕ),
    mike_oranges = 3 ∧
    matt_apples = 2 * mike_oranges ∧
    mark_bananas = mike_oranges + matt_apples ∧
    total_fruits mike_oranges matt_apples mark_bananas = 18 := by
  sorry

end annie_children_fruits_l1028_102888


namespace solve_for_N_l1028_102876

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  first : ℤ
  diff : ℤ

/-- Represents the grid of numbers -/
structure NumberGrid where
  row : ArithmeticSequence
  col1 : ArithmeticSequence
  col2 : ArithmeticSequence

/-- The problem setup -/
def problem_setup : NumberGrid where
  row := { first := 21, diff := -5 }
  col1 := { first := 6, diff := 4 }
  col2 := { first := -7, diff := -2 }

/-- The theorem to prove -/
theorem solve_for_N (grid : NumberGrid) : 
  grid.row.first = 21 ∧ 
  (grid.col1.first + 3 * grid.col1.diff = 14) ∧
  (grid.col1.first + 4 * grid.col1.diff = 18) ∧
  (grid.col2.first + 4 * grid.col2.diff = -17) →
  grid.col2.first = -7 := by
  sorry

#eval problem_setup.col2.first

end solve_for_N_l1028_102876


namespace andy_location_after_10_turns_l1028_102884

/-- Represents a direction on the coordinate plane -/
inductive Direction
  | North
  | East
  | South
  | West

/-- Represents Andy's position and facing direction -/
structure State where
  x : Int
  y : Int
  dir : Direction
  moveCount : Nat

/-- Turns the current direction 90 degrees right -/
def turnRight (d : Direction) : Direction :=
  match d with
  | Direction.North => Direction.East
  | Direction.East => Direction.South
  | Direction.South => Direction.West
  | Direction.West => Direction.North

/-- Moves Andy according to his current state -/
def move (s : State) : State :=
  let newMoveCount := s.moveCount + 1
  match s.dir with
  | Direction.North => { s with y := s.y + newMoveCount, dir := turnRight s.dir, moveCount := newMoveCount }
  | Direction.East => { s with x := s.x + newMoveCount, dir := turnRight s.dir, moveCount := newMoveCount }
  | Direction.South => { s with y := s.y - newMoveCount, dir := turnRight s.dir, moveCount := newMoveCount }
  | Direction.West => { s with x := s.x - newMoveCount, dir := turnRight s.dir, moveCount := newMoveCount }

/-- Applies the move function n times to the initial state -/
def applyMoves (n : Nat) : State :=
  match n with
  | 0 => { x := 0, y := 0, dir := Direction.North, moveCount := 0 }
  | n + 1 => move (applyMoves n)

theorem andy_location_after_10_turns :
  let finalState := applyMoves 10
  finalState.x = 6 ∧ finalState.y = 5 :=
sorry

end andy_location_after_10_turns_l1028_102884


namespace rate_categories_fractions_l1028_102839

/-- Represents the three rate categories for electricity usage --/
inductive RateCategory
  | A
  | B
  | C

/-- Total hours in a week --/
def hoursInWeek : ℕ := 7 * 24

/-- Hours that Category A applies in a week --/
def categoryAHours : ℕ := 12 * 5

/-- Hours that Category B applies in a week --/
def categoryBHours : ℕ := 10 * 2

/-- Hours that Category C applies in a week --/
def categoryCHours : ℕ := hoursInWeek - (categoryAHours + categoryBHours)

/-- Function to get the fraction of the week a category applies to --/
def categoryFraction (c : RateCategory) : ℚ :=
  match c with
  | RateCategory.A => categoryAHours / hoursInWeek
  | RateCategory.B => categoryBHours / hoursInWeek
  | RateCategory.C => categoryCHours / hoursInWeek

theorem rate_categories_fractions :
  categoryFraction RateCategory.A = 5 / 14 ∧
  categoryFraction RateCategory.B = 5 / 42 ∧
  categoryFraction RateCategory.C = 11 / 21 ∧
  categoryFraction RateCategory.A + categoryFraction RateCategory.B + categoryFraction RateCategory.C = 1 := by
  sorry


end rate_categories_fractions_l1028_102839


namespace arithmetic_mean_log_implies_geometric_mean_but_not_conversely_l1028_102857

open Real

theorem arithmetic_mean_log_implies_geometric_mean_but_not_conversely 
  (x y z : ℝ) : 
  (2 * log y = log x + log z → y ^ 2 = x * z) ∧
  ¬(y ^ 2 = x * z → 2 * log y = log x + log z) :=
by sorry

end arithmetic_mean_log_implies_geometric_mean_but_not_conversely_l1028_102857


namespace problem_1_l1028_102844

theorem problem_1 : -3⁻¹ * Real.sqrt 27 + |1 - Real.sqrt 3| + (-1)^2023 = -2 := by
  sorry

end problem_1_l1028_102844


namespace sin_B_value_l1028_102889

-- Define a right triangle ABC
structure RightTriangle :=
  (A B C : Real)
  (right_angle : C = 90)
  (bc_half_ac : B = 1/2 * A)

-- Theorem statement
theorem sin_B_value (t : RightTriangle) : Real.sin (t.B) = 2 * Real.sqrt 5 / 5 := by
  sorry

end sin_B_value_l1028_102889


namespace line_segment_endpoint_l1028_102899

theorem line_segment_endpoint (x : ℝ) : 
  (∃ (y : ℝ), (x = 3 - Real.sqrt 69 ∨ x = 3 + Real.sqrt 69) ∧ 
   ((3 - x)^2 + (8 - (-2))^2 = 13^2)) ↔ 
  (∃ (y : ℝ), ((3 - x)^2 + (y - (-2))^2 = 13^2) ∧ y = 8) :=
by sorry

end line_segment_endpoint_l1028_102899


namespace distribute_four_teachers_three_schools_l1028_102836

/-- Number of ways to distribute n distinct teachers among k distinct schools,
    with each school receiving at least one teacher -/
def distribute_teachers (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 4 distinct teachers among 3 distinct schools,
    with each school receiving at least one teacher, is 36 -/
theorem distribute_four_teachers_three_schools :
  distribute_teachers 4 3 = 36 := by sorry

end distribute_four_teachers_three_schools_l1028_102836


namespace matrix_commutation_fraction_l1028_102853

/-- Given two matrices A and B, where A is fixed and B has variable entries,
    if AB = BA and 4b ≠ c, then (a - d) / (c - 4b) = 3/8 -/
theorem matrix_commutation_fraction (a b c d : ℝ) : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![2, 3; 4, 5]
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![a, b; c, d]
  (A * B = B * A) → (4 * b ≠ c) → ((a - d) / (c - 4 * b) = 3 / 8) := by
  sorry

end matrix_commutation_fraction_l1028_102853


namespace shirts_not_washed_l1028_102829

theorem shirts_not_washed 
  (short_sleeve : ℕ) 
  (long_sleeve : ℕ) 
  (washed : ℕ) 
  (h1 : short_sleeve = 40)
  (h2 : long_sleeve = 23)
  (h3 : washed = 29) :
  short_sleeve + long_sleeve - washed = 34 := by
sorry

end shirts_not_washed_l1028_102829


namespace inscribed_circle_radius_is_five_halves_l1028_102866

/-- A trapezoid with an inscribed circle -/
structure InscribedCircleTrapezoid where
  /-- The length of the larger base -/
  a : ℕ
  /-- The length of the smaller base -/
  b : ℕ
  /-- The height of the trapezoid -/
  h : ℕ
  /-- The radius of the inscribed circle -/
  r : ℚ
  /-- The area of the upper part divided by the median -/
  upper_area : ℕ
  /-- The area of the lower part divided by the median -/
  lower_area : ℕ
  /-- Ensure the bases are different (it's a trapezoid) -/
  base_diff : a > b
  /-- The total area of the trapezoid -/
  total_area : (a + b) * h / 2 = upper_area + lower_area
  /-- The median divides the trapezoid into two parts -/
  median_division : upper_area = 15 ∧ lower_area = 30
  /-- The radius is half the height (property of inscribed circle in trapezoid) -/
  radius_height_relation : r = h / 2

/-- Theorem stating that the radius of the inscribed circle is 5/2 -/
theorem inscribed_circle_radius_is_five_halves (t : InscribedCircleTrapezoid) : t.r = 5 / 2 := by
  sorry


end inscribed_circle_radius_is_five_halves_l1028_102866


namespace quadratic_equation_equivalence_quadratic_form_components_l1028_102809

theorem quadratic_equation_equivalence :
  ∀ x : ℝ, (x + 1)^2 + (x - 2) * (x + 2) = 1 ↔ 2 * x^2 + 2 * x - 4 = 0 :=
by sorry

-- Definitions for the components of the quadratic equation
def quadratic_term (x : ℝ) : ℝ := 2 * x^2
def quadratic_coefficient : ℝ := 2
def linear_term (x : ℝ) : ℝ := 2 * x
def linear_coefficient : ℝ := 2
def constant_term : ℝ := -4

-- Theorem stating that the transformed equation is in the general form of a quadratic equation
theorem quadratic_form_components (x : ℝ) :
  2 * x^2 + 2 * x - 4 = quadratic_term x + linear_term x + constant_term :=
by sorry

end quadratic_equation_equivalence_quadratic_form_components_l1028_102809


namespace compound_molecular_weight_l1028_102817

/-- The atomic weight of hydrogen in atomic mass units (amu) -/
def hydrogen_weight : ℝ := 1.008

/-- The atomic weight of chlorine in atomic mass units (amu) -/
def chlorine_weight : ℝ := 35.45

/-- The atomic weight of oxygen in atomic mass units (amu) -/
def oxygen_weight : ℝ := 15.999

/-- The number of hydrogen atoms in the compound -/
def hydrogen_count : ℕ := 1

/-- The number of chlorine atoms in the compound -/
def chlorine_count : ℕ := 1

/-- The number of oxygen atoms in the compound -/
def oxygen_count : ℕ := 2

/-- The molecular weight of the compound in atomic mass units (amu) -/
def molecular_weight : ℝ :=
  hydrogen_count * hydrogen_weight +
  chlorine_count * chlorine_weight +
  oxygen_count * oxygen_weight

theorem compound_molecular_weight :
  molecular_weight = 68.456 := by sorry

end compound_molecular_weight_l1028_102817


namespace closest_integer_to_cube_root_150_l1028_102850

theorem closest_integer_to_cube_root_150 : 
  ∀ n : ℤ, |n - (150 : ℝ)^(1/3)| ≥ |6 - (150 : ℝ)^(1/3)| := by
  sorry

end closest_integer_to_cube_root_150_l1028_102850


namespace elena_garden_petals_l1028_102820

/-- The number of lilies in Elena's garden -/
def num_lilies : ℕ := 8

/-- The number of tulips in Elena's garden -/
def num_tulips : ℕ := 5

/-- The number of petals on each lily -/
def petals_per_lily : ℕ := 6

/-- The number of petals on each tulip -/
def petals_per_tulip : ℕ := 3

/-- The total number of flower petals in Elena's garden -/
def total_petals : ℕ := num_lilies * petals_per_lily + num_tulips * petals_per_tulip

theorem elena_garden_petals : total_petals = 63 := by
  sorry

end elena_garden_petals_l1028_102820


namespace beef_not_used_in_soup_l1028_102852

-- Define the variables
def total_beef : ℝ := 4
def vegetables_used : ℝ := 6

-- Define the theorem
theorem beef_not_used_in_soup :
  ∃ (beef_used beef_not_used : ℝ),
    beef_used = vegetables_used / 2 ∧
    beef_not_used = total_beef - beef_used ∧
    beef_not_used = 1 := by
  sorry

end beef_not_used_in_soup_l1028_102852


namespace mika_stickers_problem_l1028_102874

/-- The number of stickers Mika's mother gave her -/
def mothers_stickers (initial : Float) (bought : Float) (birthday : Float) (sister : Float) (final_total : Float) : Float :=
  final_total - (initial + bought + birthday + sister)

theorem mika_stickers_problem (initial : Float) (bought : Float) (birthday : Float) (sister : Float) (final_total : Float)
  (h1 : initial = 20.0)
  (h2 : bought = 26.0)
  (h3 : birthday = 20.0)
  (h4 : sister = 6.0)
  (h5 : final_total = 130.0) :
  mothers_stickers initial bought birthday sister final_total = 58.0 := by
  sorry

end mika_stickers_problem_l1028_102874


namespace soccer_handshakes_l1028_102854

theorem soccer_handshakes (team_size : Nat) (referee_count : Nat) (coach_count : Nat) :
  team_size = 7 ∧ referee_count = 3 ∧ coach_count = 2 →
  let player_count := 2 * team_size
  let player_player_handshakes := team_size * team_size
  let player_referee_handshakes := player_count * referee_count
  let coach_handshakes := coach_count * (player_count + referee_count)
  player_player_handshakes + player_referee_handshakes + coach_handshakes = 125 :=
by sorry


end soccer_handshakes_l1028_102854


namespace root_conditions_imply_m_range_l1028_102894

/-- A quadratic function f(x) with parameter m -/
def f (m : ℝ) (x : ℝ) : ℝ := (m - 2) * x^2 + m * x + (2 * m + 1)

/-- The theorem stating the range of m given the root conditions -/
theorem root_conditions_imply_m_range :
  ∀ m : ℝ,
  (∃ r₁ r₂ : ℝ, r₁ ≠ r₂ ∧ f m r₁ = 0 ∧ f m r₂ = 0) →
  (∃ r₁ : ℝ, -1 < r₁ ∧ r₁ < 0 ∧ f m r₁ = 0) →
  (∃ r₂ : ℝ, 1 < r₂ ∧ r₂ < 2 ∧ f m r₂ = 0) →
  1/4 < m ∧ m < 1/2 :=
sorry

end root_conditions_imply_m_range_l1028_102894


namespace polynomial_value_l1028_102835

theorem polynomial_value (x : ℝ) : 
  let a : ℝ := 2002 * x + 2003
  let b : ℝ := 2002 * x + 2004
  let c : ℝ := 2002 * x + 2005
  a^2 + b^2 + c^2 - a*b - b*c - c*a = 3 := by
sorry

end polynomial_value_l1028_102835


namespace red_to_colored_lipstick_ratio_l1028_102891

/-- Represents the number of students who attended school -/
def total_students : ℕ := 200

/-- Represents the number of students who wore blue lipstick -/
def blue_lipstick_students : ℕ := 5

/-- Represents the number of students who wore colored lipstick -/
def colored_lipstick_students : ℕ := total_students / 2

/-- Represents the number of students who wore red lipstick -/
def red_lipstick_students : ℕ := blue_lipstick_students * 5

/-- Theorem stating the ratio of students who wore red lipstick to those who wore colored lipstick -/
theorem red_to_colored_lipstick_ratio :
  (red_lipstick_students : ℚ) / colored_lipstick_students = 1 / 4 := by
  sorry

end red_to_colored_lipstick_ratio_l1028_102891


namespace little_red_journey_l1028_102807

-- Define the parameters
def total_distance : ℝ := 1500  -- in meters
def total_time : ℝ := 18  -- in minutes
def uphill_speed : ℝ := 2  -- in km/h
def downhill_speed : ℝ := 3  -- in km/h

-- Define variables for uphill and downhill time
variable (x y : ℝ)

-- Theorem statement
theorem little_red_journey :
  (x + y = total_time) ∧
  ((uphill_speed / 60) * x + (downhill_speed / 60) * y = total_distance / 1000) :=
by sorry

end little_red_journey_l1028_102807


namespace shadow_length_theorem_l1028_102824

theorem shadow_length_theorem (α β : Real) (h : Real) 
  (shadow_length : Real → Real → Real)
  (h_shadow : ∀ θ, shadow_length h θ = h * Real.tan θ)
  (h_first_measurement : Real.tan α = 3)
  (h_angle_diff : Real.tan (α - β) = 1/3) :
  Real.tan β = 4/3 := by
  sorry

end shadow_length_theorem_l1028_102824


namespace sum_of_two_numbers_l1028_102873

theorem sum_of_two_numbers (x y : ℝ) : 
  0.5 * x + 0.3333 * y = 11 → 
  max x y = 15 → 
  x + y = 27 := by
sorry

end sum_of_two_numbers_l1028_102873


namespace rationalize_denominator_l1028_102818

theorem rationalize_denominator : 7 / Real.sqrt 98 = Real.sqrt 2 / 2 := by
  sorry

end rationalize_denominator_l1028_102818


namespace cone_height_ratio_l1028_102840

theorem cone_height_ratio (base_circumference : ℝ) (original_height : ℝ) (shorter_volume : ℝ) :
  base_circumference = 20 * Real.pi →
  original_height = 24 →
  shorter_volume = 500 * Real.pi →
  ∃ (shorter_height : ℝ),
    shorter_volume = (1 / 3) * Real.pi * (base_circumference / (2 * Real.pi))^2 * shorter_height ∧
    shorter_height / original_height = 5 / 8 := by
  sorry

end cone_height_ratio_l1028_102840


namespace h_is_smallest_l1028_102827

/-- Definition of the partition property for h(n) -/
def has_partition_property (h n : ℕ) : Prop :=
  ∀ (A : Fin n → Set ℕ), 
    (∀ i j, i ≠ j → A i ∩ A j = ∅) → 
    (⋃ i, A i) = Finset.range h →
    ∃ (a x y : ℕ), 
      1 ≤ x ∧ x ≤ y ∧ y ≤ h ∧
      ∃ i, {a + x, a + y, a + x + y} ⊆ A i

/-- The function h(n) -/
def h (n : ℕ) : ℕ := Nat.choose n (n / 2)

/-- Main theorem: h(n) is the smallest positive integer satisfying the partition property -/
theorem h_is_smallest (n : ℕ) (hn : 0 < n) : 
  has_partition_property (h n) n ∧ 
  ∀ m, 0 < m ∧ m < h n → ¬has_partition_property m n :=
sorry

end h_is_smallest_l1028_102827


namespace equation_solutions_l1028_102805

theorem equation_solutions : 
  ∀ x : ℝ, (x - 2)^2 = 9*x^2 ↔ x = -1 ∨ x = 1/2 := by sorry

end equation_solutions_l1028_102805


namespace james_writing_speed_l1028_102892

/-- Represents the writing schedule and book information --/
structure WritingInfo where
  hours_per_day : ℕ
  weeks : ℕ
  total_pages : ℕ

/-- Calculates the number of pages written per hour --/
def pages_per_hour (info : WritingInfo) : ℚ :=
  info.total_pages / (info.hours_per_day * 7 * info.weeks)

/-- Theorem stating that given the specific writing schedule and book length,
    the number of pages written per hour is 5 --/
theorem james_writing_speed :
  let info : WritingInfo := ⟨3, 7, 735⟩
  pages_per_hour info = 5 := by
  sorry

end james_writing_speed_l1028_102892


namespace first_day_is_sunday_l1028_102883

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Function to get the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

/-- Function to get the day of the week after n days -/
def afterDays (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => nextDay (afterDays d n)

/-- Theorem: If the 18th day of a month is a Wednesday, then the 1st day of that month is a Sunday -/
theorem first_day_is_sunday (d : DayOfWeek) (h : afterDays d 17 = DayOfWeek.Wednesday) :
  d = DayOfWeek.Sunday := by
  sorry

end first_day_is_sunday_l1028_102883


namespace line_equation_problem_l1028_102804

-- Define a line by its slope and y-intercept
structure Line where
  slope : ℝ
  y_intercept : ℝ

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define the problem
theorem line_equation_problem (l : Line) (P : Point) :
  (P.x = 2 ∧ P.y = 3) →
  (
    (l.slope = -Real.sqrt 3) ∨
    (l.slope = -2) ∨
    (l.slope = 3/2 ∧ l.y_intercept = 0) ∨
    (l.slope = 1 ∧ l.y_intercept = -1)
  ) →
  (
    (Real.sqrt 3 * P.x + P.y - 3 - 2 * Real.sqrt 3 = 0) ∨
    (2 * P.x + P.y - 7 = 0) ∨
    (3 * P.x - 2 * P.y = 0) ∨
    (P.x - P.y + 1 = 0)
  ) :=
by sorry

end line_equation_problem_l1028_102804


namespace library_books_taken_out_l1028_102825

theorem library_books_taken_out (initial_books : ℕ) (books_returned : ℕ) (books_taken_out : ℕ) (final_books : ℕ) :
  initial_books = 235 →
  books_returned = 56 →
  books_taken_out = 35 →
  final_books = 29 →
  ∃ (tuesday_books : ℕ), tuesday_books = 227 ∧ 
    initial_books - tuesday_books + books_returned - books_taken_out = final_books :=
by
  sorry


end library_books_taken_out_l1028_102825


namespace complex_equation_solution_l1028_102812

theorem complex_equation_solution (z : ℂ) : z = Complex.I * (2 + z) → z = -1 + Complex.I := by
  sorry

end complex_equation_solution_l1028_102812


namespace intersection_point_is_unique_l1028_102803

/-- The intersection point of two lines -/
def intersection_point : ℚ × ℚ := (-1/4, -3/4)

/-- First line equation -/
def line1 (x y : ℚ) : Prop := y = 3 * x

/-- Second line equation -/
def line2 (x y : ℚ) : Prop := y + 3 = -9 * x

theorem intersection_point_is_unique :
  ∀ x y : ℚ, line1 x y ∧ line2 x y ↔ (x, y) = intersection_point := by sorry

end intersection_point_is_unique_l1028_102803


namespace count_odd_increasing_integers_l1028_102845

/-- A three-digit integer with odd digits in strictly increasing order -/
structure OddIncreasingInteger where
  hundreds : Nat
  tens : Nat
  ones : Nat
  is_odd_hundreds : Odd hundreds
  is_odd_tens : Odd tens
  is_odd_ones : Odd ones
  is_increasing : hundreds < tens ∧ tens < ones
  is_three_digit : 100 ≤ hundreds * 100 + tens * 10 + ones ∧ hundreds * 100 + tens * 10 + ones < 1000

/-- The count of three-digit integers with odd digits in strictly increasing order -/
def countOddIncreasingIntegers : Nat := sorry

/-- Theorem stating that there are exactly 10 three-digit integers with odd digits in strictly increasing order -/
theorem count_odd_increasing_integers :
  countOddIncreasingIntegers = 10 := by sorry

end count_odd_increasing_integers_l1028_102845


namespace fair_division_of_walls_l1028_102821

/-- The number of people in Amanda's family -/
def family_size : ℕ := 5

/-- The number of rooms with 4 walls -/
def rooms_with_4_walls : ℕ := 5

/-- The number of rooms with 5 walls -/
def rooms_with_5_walls : ℕ := 4

/-- The total number of walls in the house -/
def total_walls : ℕ := rooms_with_4_walls * 4 + rooms_with_5_walls * 5

/-- The number of walls each person should paint for fair division -/
def walls_per_person : ℕ := total_walls / family_size

theorem fair_division_of_walls :
  walls_per_person = 8 := by sorry

end fair_division_of_walls_l1028_102821


namespace monotonicity_condition_max_k_value_l1028_102863

noncomputable section

def f (x : ℝ) : ℝ := (x^2 - 3*x + 3) * Real.exp x

def is_monotonic (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y ∨ (∀ z, a ≤ z ∧ z ≤ b → f z = f x)

theorem monotonicity_condition (t : ℝ) :
  (is_monotonic f (-2) t) ↔ -2 < t ∧ t ≤ 0 := by sorry

theorem max_k_value :
  ∃ k : ℕ, k = 6 ∧ 
  (∀ x : ℝ, x > 0 → (f x / Real.exp x) + 7*x - 2 > k * (x * Real.log x - 1)) ∧
  (∀ m : ℕ, m > k → ∃ x : ℝ, x > 0 ∧ (f x / Real.exp x) + 7*x - 2 ≤ m * (x * Real.log x - 1)) := by sorry

end

end monotonicity_condition_max_k_value_l1028_102863


namespace snake_revenue_theorem_l1028_102886

/-- Calculates the total revenue from selling Jake's baby snakes --/
def calculate_snake_revenue (num_snakes : ℕ) (eggs_per_snake : ℕ) (regular_price : ℕ) (rare_multiplier : ℕ) : ℕ :=
  let total_babies := num_snakes * eggs_per_snake
  let regular_babies := total_babies - 1
  let regular_revenue := regular_babies * regular_price
  let rare_revenue := regular_price * rare_multiplier
  regular_revenue + rare_revenue

/-- Proves that the total revenue from selling Jake's baby snakes is $2250 --/
theorem snake_revenue_theorem :
  calculate_snake_revenue 3 2 250 4 = 2250 := by
  sorry

end snake_revenue_theorem_l1028_102886


namespace meadowood_58_impossible_l1028_102833

/-- Represents the village of Meadowood with its animal and people relationships -/
structure Meadowood where
  sheep : ℕ
  horses : ℕ
  ducks : ℕ := 5 * sheep
  cows : ℕ := 2 * horses
  people : ℕ := 4 * ducks

/-- The total population in Meadowood -/
def Meadowood.total (m : Meadowood) : ℕ :=
  m.people + m.horses + m.sheep + m.cows + m.ducks

/-- Theorem stating that 58 cannot be the total population in Meadowood -/
theorem meadowood_58_impossible : ¬∃ m : Meadowood, m.total = 58 := by
  sorry

end meadowood_58_impossible_l1028_102833


namespace september_first_was_wednesday_l1028_102837

/-- Represents the days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Calculates the number of lessons Vasya skips on a given day -/
def lessonsSkipped (day : DayOfWeek) : Nat :=
  match day with
  | DayOfWeek.Monday => 1
  | DayOfWeek.Tuesday => 2
  | DayOfWeek.Wednesday => 3
  | DayOfWeek.Thursday => 4
  | DayOfWeek.Friday => 5
  | _ => 0

/-- Calculates the day of the week for a given date in September -/
def dayOfWeekForDate (date : Nat) (sept1 : DayOfWeek) : DayOfWeek :=
  sorry

/-- Calculates the total number of lessons Vasya skipped in September -/
def totalLessonsSkipped (sept1 : DayOfWeek) : Nat :=
  sorry

theorem september_first_was_wednesday :
  totalLessonsSkipped DayOfWeek.Wednesday = 64 :=
by sorry

end september_first_was_wednesday_l1028_102837


namespace project_hours_l1028_102859

theorem project_hours (x y z : ℕ) (h1 : y = (5 * x) / 3) (h2 : z = 2 * x) (h3 : z = x + 30) :
  x + y + z = 140 :=
by sorry

end project_hours_l1028_102859


namespace mean_temperature_l1028_102802

def temperatures : List ℝ := [80, 79, 81, 85, 87, 89, 87, 90, 89, 88]

theorem mean_temperature :
  (temperatures.sum / temperatures.length : ℝ) = 85.5 := by
  sorry

end mean_temperature_l1028_102802


namespace range_of_y_l1028_102880

-- Define the function f
def f (x : ℝ) : ℝ := |x^2 - x| - 4*x

-- State the theorem
theorem range_of_y (y : ℝ) :
  (∃ x : ℝ, 0 < x ∧ x < 2 ∧ f x = 0) ↔ -4 < y ∧ y < 12 :=
sorry

end range_of_y_l1028_102880


namespace prime_divides_square_implies_divides_l1028_102881

theorem prime_divides_square_implies_divides (p n : ℕ) : 
  Prime p → (p ∣ n^2) → (p ∣ n) := by
  sorry

end prime_divides_square_implies_divides_l1028_102881


namespace binomial_coefficient_x4_in_expansion_l1028_102868

/-- The binomial coefficient of the term containing x^4 in the expansion of (x^2 + 1/x)^5 is 10 -/
theorem binomial_coefficient_x4_in_expansion : 
  ∃ k : ℕ, (Nat.choose 5 k) * (4 : ℤ) = (10 : ℤ) ∧ 
    10 - 3 * k = 4 := by sorry

end binomial_coefficient_x4_in_expansion_l1028_102868


namespace averageIs295_l1028_102832

/-- Calculates the average number of visitors per day in a 30-day month starting on a Sunday -/
def averageVisitorsPerDay (sundayVisitors : ℕ) (otherDayVisitors : ℕ) : ℚ :=
  let totalSundays : ℕ := 5
  let totalOtherDays : ℕ := 25
  let totalVisitors : ℕ := sundayVisitors * totalSundays + otherDayVisitors * totalOtherDays
  totalVisitors / 30

/-- Theorem stating that the average number of visitors per day is 295 -/
theorem averageIs295 (sundayVisitors : ℕ) (otherDayVisitors : ℕ) 
    (h1 : sundayVisitors = 570) (h2 : otherDayVisitors = 240) : 
    averageVisitorsPerDay sundayVisitors otherDayVisitors = 295 := by
  sorry

end averageIs295_l1028_102832


namespace triangle_properties_l1028_102830

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles
  (a b c : ℝ)  -- Sides

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.b^2 = t.a * t.c ∧ Real.cos (t.A - t.C) = Real.cos t.B + 1/2

-- Theorem statement
theorem triangle_properties (t : Triangle) (h : triangle_conditions t) :
  t.B = π/3 ∧ t.A = π/3 ∧
  ∀ (CD : ℝ), CD = 6 → 
    (∃ (max_perimeter : ℝ), max_perimeter = 4 * Real.sqrt 3 + 6 ∧
      ∀ (perimeter : ℝ), perimeter ≤ max_perimeter) :=
by sorry

end triangle_properties_l1028_102830


namespace inequality_system_solution_l1028_102860

theorem inequality_system_solution (x : ℝ) :
  (2 * x - 1 ≥ x + 2) ∧ (x + 5 < 4 * x - 1) → x ≥ 3 := by
  sorry

end inequality_system_solution_l1028_102860


namespace count_valid_permutations_l1028_102856

/-- The set of digits in the number 2033 -/
def digits : Finset ℕ := {2, 0, 3, 3}

/-- A function that checks if a number is a 4-digit number -/
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

/-- A function that calculates the sum of digits of a number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- The set of all 4-digit permutations of the digits in 2033 -/
def valid_permutations : Finset ℕ := sorry

theorem count_valid_permutations : Finset.card valid_permutations = 15 := by sorry

end count_valid_permutations_l1028_102856


namespace sixth_term_of_arithmetic_sequence_l1028_102869

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sixth_term_of_arithmetic_sequence 
  (a : ℕ → ℚ) 
  (h_arith : arithmetic_sequence a) 
  (h_sum : a 1 + a 2 + a 3 + a 4 + a 5 = 25) 
  (h_a2 : a 2 = 3) :
  a 6 = 11 := by
  sorry

end sixth_term_of_arithmetic_sequence_l1028_102869


namespace betty_herb_garden_l1028_102855

theorem betty_herb_garden (basil thyme oregano : ℕ) : 
  basil = 5 →
  thyme = 4 →
  oregano = 2 * basil + 2 →
  basil = 3 * thyme - 3 →
  basil + oregano + thyme = 21 := by
  sorry

end betty_herb_garden_l1028_102855


namespace negation_of_universal_proposition_l1028_102810

-- Define the set of all functions
variable (F : Type)

-- Define the property of being a logarithmic function
variable (isLogarithmic : F → Prop)

-- Define the property of being a monotonic function
variable (isMonotonic : F → Prop)

-- The theorem to prove
theorem negation_of_universal_proposition :
  (¬ ∀ f : F, isLogarithmic f → isMonotonic f) ↔ 
  (∃ f : F, isLogarithmic f ∧ ¬isMonotonic f) :=
by sorry

end negation_of_universal_proposition_l1028_102810


namespace solution_set_inequality_l1028_102823

theorem solution_set_inequality (x : ℝ) :
  (x - 3) * (x - 1) > 0 ↔ x < 1 ∨ x > 3 := by
  sorry

end solution_set_inequality_l1028_102823


namespace pencil_sorting_l1028_102831

theorem pencil_sorting (box2 box3 box4 box5 : ℕ) : 
  box2 = 87 →
  box3 = box2 + 9 →
  box4 = box3 + 9 →
  box5 = box4 + 9 →
  box5 = 114 →
  box2 - 9 = 78 := by
sorry

end pencil_sorting_l1028_102831


namespace trapezoid_triangle_area_ratio_l1028_102862

/-- An inscribed acute-angled isosceles triangle in a circle -/
structure IsoscelesTriangle (α : ℝ) :=
  (angle_base : 0 < α ∧ α < π/2)

/-- An inscribed trapezoid in a circle -/
structure Trapezoid (α : ℝ) :=
  (base_is_diameter : True)
  (sides_parallel_to_triangle : True)

/-- The theorem stating that the area of the trapezoid equals the area of the triangle -/
theorem trapezoid_triangle_area_ratio 
  (α : ℝ) 
  (triangle : IsoscelesTriangle α) 
  (trapezoid : Trapezoid α) : 
  ∃ (area_trapezoid area_triangle : ℝ), 
    area_trapezoid / area_triangle = 1 := by
  sorry

end trapezoid_triangle_area_ratio_l1028_102862


namespace set_C_elements_l1028_102834

def A : Set ℕ := {1, 3, 5, 7}
def B : Set ℕ := {2, 4, 6}
def C : Set (ℕ × ℕ) := {p | p.1 ∈ A ∧ p.2 ∈ B}

theorem set_C_elements : C = {(1,2), (1,4), (1,6), (3,2), (3,4), (3,6), (5,2), (5,4), (5,6), (7,2), (7,4), (7,6)} := by
  sorry

end set_C_elements_l1028_102834


namespace log_101600_value_l1028_102879

-- Define the logarithm function
noncomputable def log : ℝ → ℝ := Real.log

-- State the theorem
theorem log_101600_value (h : log 102 = 0.3010) : log 101600 = 3.3010 := by
  sorry

end log_101600_value_l1028_102879


namespace number_equation_solution_l1028_102800

theorem number_equation_solution : ∃ x : ℝ, 2 * x - 3 = 7 ∧ x = 5 := by
  sorry

end number_equation_solution_l1028_102800


namespace age_problem_l1028_102811

/-- The age problem involving Sebastian, his siblings, and their father. -/
theorem age_problem (sebastian_age : ℕ) (sister_age_diff : ℕ) (brother_age_diff : ℕ) : 
  sebastian_age = 40 →
  sister_age_diff = 10 →
  brother_age_diff = 7 →
  (sebastian_age - 5 + (sebastian_age - sister_age_diff - 5) + 
   (sebastian_age - sister_age_diff - brother_age_diff - 5) : ℚ) = 
   (3 / 4 : ℚ) * ((109 : ℕ) - 5) →
  109 = sebastian_age + 69 := by
  sorry

#check age_problem

end age_problem_l1028_102811


namespace base7_divisible_by_19_l1028_102843

/-- Given a digit y, returns the decimal representation of 52y3 in base 7 -/
def base7ToDecimal (y : ℕ) : ℕ := 5 * 7^3 + 2 * 7^2 + y * 7 + 3

/-- Theorem stating that when 52y3 in base 7 is divisible by 19, y must be 8 -/
theorem base7_divisible_by_19 :
  ∃ y : ℕ, y < 7 ∧ (base7ToDecimal y) % 19 = 0 → y = 8 :=
by sorry

end base7_divisible_by_19_l1028_102843


namespace distance_between_specific_lines_l1028_102887

/-- Line represented by a parametric equation -/
structure ParametricLine where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- Line represented by a slope-intercept equation -/
structure SlopeInterceptLine where
  slope : ℝ
  intercept : ℝ

/-- The distance between two lines -/
def distance_between_lines (l₁ : ParametricLine) (l₂ : SlopeInterceptLine) : ℝ :=
  sorry

/-- The given problem statement -/
theorem distance_between_specific_lines :
  let l₁ : ParametricLine := {
    x := λ t => 1 + t,
    y := λ t => 1 + 3*t
  }
  let l₂ : SlopeInterceptLine := {
    slope := 3,
    intercept := 4
  }
  distance_between_lines l₁ l₂ = 3 * Real.sqrt 10 / 5 :=
sorry

end distance_between_specific_lines_l1028_102887


namespace shaded_area_semicircles_l1028_102814

/-- The area of the shaded region formed by semicircles -/
theorem shaded_area_semicircles (UV VW WX XY YZ : ℝ) 
  (h_UV : UV = 3) 
  (h_VW : VW = 5) 
  (h_WX : WX = 4) 
  (h_XY : XY = 6) 
  (h_YZ : YZ = 7) : 
  let UZ := UV + VW + WX + XY + YZ
  let area_large := (π / 8) * UZ^2
  let area_small := (π / 8) * (UV^2 + VW^2 + WX^2 + XY^2 + YZ^2)
  area_large - area_small = (247 / 4) * π := by
  sorry

end shaded_area_semicircles_l1028_102814


namespace expression_simplification_l1028_102815

theorem expression_simplification (x : ℝ) : 3*x + 6*x + 9*x + 12*x + 15*x + 18 + 6 = 45*x + 24 := by
  sorry

end expression_simplification_l1028_102815


namespace octal_subtraction_example_l1028_102801

/-- Represents a number in base 8 --/
def OctalNumber := ℕ

/-- Converts a natural number to its octal representation --/
def toOctal (n : ℕ) : OctalNumber :=
  sorry

/-- Performs subtraction in base 8 --/
def octalSubtract (a b : OctalNumber) : OctalNumber :=
  sorry

theorem octal_subtraction_example :
  octalSubtract (toOctal 641) (toOctal 324) = toOctal 317 := by
  sorry

end octal_subtraction_example_l1028_102801


namespace x_younger_than_w_l1028_102847

-- Define the ages of the individuals
variable (w_years x_years y_years z_years : ℤ)

-- Define the conditions
axiom sum_condition : w_years + x_years = y_years + z_years + 15
axiom difference_condition : |w_years - x_years| = 2 * |y_years - z_years|
axiom w_z_relation : w_years = z_years + 30

-- Theorem to prove
theorem x_younger_than_w : x_years = w_years - 45 := by
  sorry

end x_younger_than_w_l1028_102847
