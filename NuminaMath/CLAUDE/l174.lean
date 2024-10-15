import Mathlib

namespace NUMINAMATH_CALUDE_kittens_sold_l174_17474

theorem kittens_sold (initial_puppies initial_kittens puppies_sold remaining_pets : ℕ) : 
  initial_puppies = 7 →
  initial_kittens = 6 →
  puppies_sold = 2 →
  remaining_pets = 8 →
  initial_puppies + initial_kittens - puppies_sold - remaining_pets = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_kittens_sold_l174_17474


namespace NUMINAMATH_CALUDE_wednesday_sales_l174_17416

def initial_stock : ℕ := 700
def monday_sales : ℕ := 50
def tuesday_sales : ℕ := 82
def thursday_sales : ℕ := 48
def friday_sales : ℕ := 40
def unsold_percentage : ℚ := 60 / 100

theorem wednesday_sales :
  let total_sales := initial_stock - (initial_stock * unsold_percentage).floor
  let other_days_sales := monday_sales + tuesday_sales + thursday_sales + friday_sales
  total_sales - other_days_sales = 60 := by
sorry

end NUMINAMATH_CALUDE_wednesday_sales_l174_17416


namespace NUMINAMATH_CALUDE_expression_value_l174_17419

theorem expression_value (x y : ℤ) (hx : x = -6) (hy : y = -3) :
  4 * (x - y)^2 - x * y = 18 := by sorry

end NUMINAMATH_CALUDE_expression_value_l174_17419


namespace NUMINAMATH_CALUDE_smallest_positive_multiple_of_45_l174_17472

theorem smallest_positive_multiple_of_45 :
  ∀ n : ℕ, n > 0 → (∃ k : ℕ, k > 0 ∧ n = 45 * k) → n ≥ 45 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_multiple_of_45_l174_17472


namespace NUMINAMATH_CALUDE_broker_commission_rate_l174_17473

theorem broker_commission_rate 
  (initial_rate : ℝ) 
  (slump_percentage : ℝ) 
  (new_rate : ℝ) :
  initial_rate = 0.04 →
  slump_percentage = 0.20000000000000007 →
  new_rate = initial_rate / (1 - slump_percentage) →
  new_rate = 0.05 := by
sorry

end NUMINAMATH_CALUDE_broker_commission_rate_l174_17473


namespace NUMINAMATH_CALUDE_fifa_world_cup_2010_matches_l174_17453

/-- Calculates the number of matches in a round-robin tournament -/
def roundRobinMatches (n : Nat) : Nat :=
  n * (n - 1) / 2

/-- Calculates the number of matches in a knockout tournament -/
def knockoutMatches (n : Nat) : Nat :=
  n - 1

theorem fifa_world_cup_2010_matches : 
  let totalTeams : Nat := 24
  let groups : Nat := 6
  let teamsPerGroup : Nat := 4
  let knockoutTeams : Nat := 16
  let firstRoundMatches := groups * roundRobinMatches teamsPerGroup
  let knockoutStageMatches := knockoutMatches knockoutTeams
  firstRoundMatches + knockoutStageMatches = 51 := by
  sorry

end NUMINAMATH_CALUDE_fifa_world_cup_2010_matches_l174_17453


namespace NUMINAMATH_CALUDE_wire_cutting_l174_17496

theorem wire_cutting (total_length : ℝ) (ratio : ℝ) (shorter_piece : ℝ) : 
  total_length = 90 →
  ratio = 2 / 7 →
  shorter_piece + (shorter_piece / ratio) = total_length →
  shorter_piece = 20 :=
by sorry

end NUMINAMATH_CALUDE_wire_cutting_l174_17496


namespace NUMINAMATH_CALUDE_max_product_difference_l174_17449

theorem max_product_difference (a₁ a₂ a₃ a₄ a₅ : ℝ) 
  (h₁ : 0 ≤ a₁ ∧ a₁ ≤ 1) (h₂ : 0 ≤ a₂ ∧ a₂ ≤ 1) (h₃ : 0 ≤ a₃ ∧ a₃ ≤ 1) 
  (h₄ : 0 ≤ a₄ ∧ a₄ ≤ 1) (h₅ : 0 ≤ a₅ ∧ a₅ ≤ 1) : 
  |a₁ - a₂| * |a₁ - a₃| * |a₁ - a₄| * |a₁ - a₅| * 
  |a₂ - a₃| * |a₂ - a₄| * |a₂ - a₅| * 
  |a₃ - a₄| * |a₃ - a₅| * 
  |a₄ - a₅| ≤ 3 * Real.sqrt 21 / 38416 := by
  sorry

end NUMINAMATH_CALUDE_max_product_difference_l174_17449


namespace NUMINAMATH_CALUDE_complex_equation_solution_l174_17452

theorem complex_equation_solution (z : ℂ) (h : Complex.I * (z + 2 * Complex.I) = 1) : z = -3 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l174_17452


namespace NUMINAMATH_CALUDE_sum_of_number_and_square_l174_17409

theorem sum_of_number_and_square : 
  let n : ℕ := 15
  n + n^2 = 240 := by sorry

end NUMINAMATH_CALUDE_sum_of_number_and_square_l174_17409


namespace NUMINAMATH_CALUDE_bethany_current_age_l174_17483

def bethany_age_problem (bethany_age : ℕ) (sister_age : ℕ) : Prop :=
  (bethany_age - 3 = 2 * (sister_age - 3)) ∧ (sister_age + 5 = 16)

theorem bethany_current_age :
  ∃ (bethany_age : ℕ) (sister_age : ℕ), bethany_age_problem bethany_age sister_age ∧ bethany_age = 19 :=
by
  sorry

end NUMINAMATH_CALUDE_bethany_current_age_l174_17483


namespace NUMINAMATH_CALUDE_boys_in_class_l174_17406

theorem boys_in_class (total : ℕ) (diff : ℕ) (boys : ℕ) : 
  total = 345 → diff = 69 → boys + (boys + diff) = total → boys = 138 := by
  sorry

end NUMINAMATH_CALUDE_boys_in_class_l174_17406


namespace NUMINAMATH_CALUDE_evaluate_expression_l174_17480

theorem evaluate_expression : (4^4 - 4*(4-1)^4)^4 = 21381376 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l174_17480


namespace NUMINAMATH_CALUDE_area_of_specific_triangle_l174_17404

/-- The line equation ax + by = c --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A triangle bounded by the x-axis, y-axis, and a line --/
structure AxisAlignedTriangle where
  boundingLine : Line

/-- The area of an axis-aligned triangle --/
def areaOfAxisAlignedTriangle (t : AxisAlignedTriangle) : ℝ :=
  sorry

theorem area_of_specific_triangle : 
  let t : AxisAlignedTriangle := { boundingLine := { a := 3, b := 4, c := 12 } }
  areaOfAxisAlignedTriangle t = 6 := by sorry

end NUMINAMATH_CALUDE_area_of_specific_triangle_l174_17404


namespace NUMINAMATH_CALUDE_complex_power_20_l174_17413

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_power_20 : (1 + i) ^ 20 = -1024 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_20_l174_17413


namespace NUMINAMATH_CALUDE_exactly_two_consecutive_sets_sum_18_l174_17428

/-- A set of consecutive positive integers -/
structure ConsecutiveSet :=
  (start : ℕ)
  (length : ℕ)
  (h_length : length ≥ 2)

/-- The sum of a set of consecutive positive integers -/
def sum_consecutive (s : ConsecutiveSet) : ℕ :=
  s.length * (2 * s.start + s.length - 1) / 2

/-- Predicate for a ConsecutiveSet that sums to 18 -/
def sums_to_18 (s : ConsecutiveSet) : Prop :=
  sum_consecutive s = 18

theorem exactly_two_consecutive_sets_sum_18 :
  ∃! (sets : Finset ConsecutiveSet), (∀ s ∈ sets, sums_to_18 s) ∧ sets.card = 2 :=
sorry

end NUMINAMATH_CALUDE_exactly_two_consecutive_sets_sum_18_l174_17428


namespace NUMINAMATH_CALUDE_constant_term_expansion_l174_17442

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℚ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

-- Define the function for the general term of the expansion
def generalTerm (r : ℕ) : ℚ := (-1/2)^r * binomial 6 r

-- Define the constant term as the term where the power of x is zero
def constantTerm : ℚ := generalTerm 4

-- Theorem statement
theorem constant_term_expansion :
  constantTerm = 15/16 := by sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l174_17442


namespace NUMINAMATH_CALUDE_sum_of_21st_set_l174_17463

/-- The sum of elements in the n-th set of a sequence where:
    1. Each set contains consecutive integers
    2. Each set contains one more element than the previous set
    3. The first element of each set is one greater than the last element of the previous set
-/
def S (n : ℕ) : ℚ :=
  n * (n^2 - n + 2) / 2

theorem sum_of_21st_set :
  S 21 = 4641 :=
sorry

end NUMINAMATH_CALUDE_sum_of_21st_set_l174_17463


namespace NUMINAMATH_CALUDE_bisecting_line_theorem_l174_17411

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Calculates the area of a quadrilateral given its vertices -/
def quadrilateralArea (a b c d : Point) : ℝ := sorry

/-- Checks if two lines are parallel -/
def isParallel (l1 l2 : Line) : Prop := sorry

/-- Calculates the area of the part of the quadrilateral below a given line -/
def areaBelow (a b c d : Point) (l : Line) : ℝ := sorry

/-- The main theorem to be proved -/
theorem bisecting_line_theorem (a b c d : Point) (l : Line) : 
  a = Point.mk 0 0 →
  b = Point.mk 16 0 →
  c = Point.mk 8 8 →
  d = Point.mk 0 8 →
  l = Line.mk 1 (-4) →
  isParallel l (Line.mk 1 0) ∧ 
  areaBelow a b c d l = (quadrilateralArea a b c d) / 2 := by
  sorry

end NUMINAMATH_CALUDE_bisecting_line_theorem_l174_17411


namespace NUMINAMATH_CALUDE_fraction_reduction_l174_17451

theorem fraction_reduction (a b : ℕ) (h : a = 4128 ∧ b = 4386) :
  ∃ (c d : ℕ), c = 295 ∧ d = 313 ∧ a / b = c / d ∧ Nat.gcd c d = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_reduction_l174_17451


namespace NUMINAMATH_CALUDE_beverage_selection_probabilities_l174_17441

/-- The number of cups of Beverage A -/
def num_a : ℕ := 3

/-- The number of cups of Beverage B -/
def num_b : ℕ := 2

/-- The total number of cups -/
def total_cups : ℕ := num_a + num_b

/-- The number of cups to be selected -/
def select_cups : ℕ := 3

/-- The probability of selecting all cups of Beverage A -/
def prob_excellent : ℚ := 1 / 10

/-- The probability of selecting at least 2 cups of Beverage A -/
def prob_good_or_above : ℚ := 7 / 10

theorem beverage_selection_probabilities :
  (Nat.choose total_cups select_cups : ℚ) * prob_excellent = Nat.choose num_a select_cups ∧
  (Nat.choose total_cups select_cups : ℚ) * prob_good_or_above = 
    Nat.choose num_a select_cups + Nat.choose num_a 2 * Nat.choose num_b 1 := by
  sorry

end NUMINAMATH_CALUDE_beverage_selection_probabilities_l174_17441


namespace NUMINAMATH_CALUDE_new_years_day_theorem_l174_17470

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a specific date in a year -/
structure Date where
  month : Nat
  day : Nat

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

/-- Function to advance a day by n days -/
def advanceDay (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => advanceDay (nextDay d) n

theorem new_years_day_theorem 
  (february_has_29_days : Nat)
  (february_has_four_mondays : Nat)
  (february_has_five_sundays : Nat)
  (february_13_is_friday : DayOfWeek)
  : (february_has_29_days = 29) →
    (february_has_four_mondays = 4) →
    (february_has_five_sundays = 5) →
    (february_13_is_friday = DayOfWeek.Friday) →
    (∃ (new_years_day : DayOfWeek), 
      new_years_day = DayOfWeek.Thursday ∧ 
      advanceDay new_years_day 366 = DayOfWeek.Saturday) :=
by sorry


end NUMINAMATH_CALUDE_new_years_day_theorem_l174_17470


namespace NUMINAMATH_CALUDE_fraction_problem_l174_17479

theorem fraction_problem (n : ℚ) (f : ℚ) (h1 : n = 120) (h2 : (1/2) * f * n = 36) : f = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l174_17479


namespace NUMINAMATH_CALUDE_inequality_one_l174_17415

theorem inequality_one (x : ℝ) : 
  (x + 2) / (x - 4) ≤ 0 ↔ -2 ≤ x ∧ x < 4 :=
sorry

end NUMINAMATH_CALUDE_inequality_one_l174_17415


namespace NUMINAMATH_CALUDE_allocation_schemes_count_l174_17460

/-- The number of ways to divide 6 volunteers into 4 groups and assign them to venues -/
def allocationSchemes : ℕ :=
  let n := 6  -- number of volunteers
  let k := 4  -- number of groups/venues
  let g₂ := 2  -- number of groups with 2 people
  let g₁ := 2  -- number of groups with 1 person
  540

/-- Theorem stating that the number of allocation schemes is 540 -/
theorem allocation_schemes_count : allocationSchemes = 540 := by
  sorry

end NUMINAMATH_CALUDE_allocation_schemes_count_l174_17460


namespace NUMINAMATH_CALUDE_at_least_one_inequality_holds_l174_17447

theorem at_least_one_inequality_holds (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y > 2) :
  (1 + x) / y < 2 ∨ (1 + y) / x < 2 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_inequality_holds_l174_17447


namespace NUMINAMATH_CALUDE_natural_numbers_less_than_two_l174_17458

theorem natural_numbers_less_than_two :
  {n : ℕ | n < 2} = {0, 1} := by sorry

end NUMINAMATH_CALUDE_natural_numbers_less_than_two_l174_17458


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_equal_intercepts_line_equation_l174_17489

-- Define the types for points and lines
def Point := ℝ × ℝ
def Line := ℝ → ℝ → Prop

-- Define the intersection point of two lines
def intersection (l1 l2 : Line) : Point :=
  sorry

-- Define perpendicularity of two lines
def perpendicular (l1 l2 : Line) : Prop :=
  sorry

-- Define a line passing through a point
def passes_through (l : Line) (p : Point) : Prop :=
  sorry

-- Define a line having equal intercepts on coordinate axes
def equal_intercepts (l : Line) : Prop :=
  sorry

-- Define the lines given in the problem
def line1 : Line := λ x y ↦ 2*x + 3*y - 9 = 0
def line2 : Line := λ x y ↦ 3*x - y - 8 = 0
def line3 : Line := λ x y ↦ 3*x + 4*y - 1 = 0

-- Part 1
theorem perpendicular_line_equation :
  ∀ l : Line,
  passes_through l (intersection line1 line2) →
  perpendicular l line3 →
  l = λ x y ↦ y = (4/3)*x - 3 :=
sorry

-- Part 2
theorem equal_intercepts_line_equation :
  ∀ l : Line,
  passes_through l (intersection line1 line2) →
  equal_intercepts l →
  (l = λ x y ↦ y = -x + 4) ∨ (l = λ x y ↦ y = (1/3)*x) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_equal_intercepts_line_equation_l174_17489


namespace NUMINAMATH_CALUDE_percentage_of_B_grades_l174_17448

def grading_scale : List (String × (Int × Int)) :=
  [("A", (94, 100)), ("B", (87, 93)), ("C", (78, 86)), ("D", (70, 77)), ("F", (0, 69))]

def scores : List Int := [93, 65, 88, 100, 72, 95, 82, 68, 79, 56, 87, 81, 74, 85, 91]

def is_grade (score : Int) (grade : String × (Int × Int)) : Bool :=
  let (_, (low, high)) := grade
  low ≤ score ∧ score ≤ high

def count_grade (scores : List Int) (grade : String × (Int × Int)) : Nat :=
  (scores.filter (fun score => is_grade score grade)).length

theorem percentage_of_B_grades :
  let b_grade := ("B", (87, 93))
  let total_students := scores.length
  let b_students := count_grade scores b_grade
  (b_students : Rat) / total_students * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_B_grades_l174_17448


namespace NUMINAMATH_CALUDE_counterexample_exists_l174_17436

theorem counterexample_exists : ∃ (a b : ℝ), a < b ∧ a^2 ≥ b^2 := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l174_17436


namespace NUMINAMATH_CALUDE_max_squared_ratio_is_one_l174_17465

/-- The maximum squared ratio of a to b satisfying the given conditions -/
def max_squared_ratio (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ a ≥ b ∧
  ∃ ρ : ℝ, ρ > 0 ∧
    ∀ x y : ℝ,
      0 ≤ x ∧ x < a ∧
      0 ≤ y ∧ y < b ∧
      a^2 + y^2 = b^2 + x^2 ∧
      b^2 + x^2 = (a - x)^2 + (b - y)^2 ∧
      (a - x) * (b - y) = 0 →
      (a / b)^2 ≤ ρ^2 ∧
      ρ^2 = 1

theorem max_squared_ratio_is_one (a b : ℝ) (h : max_squared_ratio a b) :
  ∃ ρ : ℝ, ρ > 0 ∧ ρ^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_max_squared_ratio_is_one_l174_17465


namespace NUMINAMATH_CALUDE_fruit_cost_solution_l174_17417

/-- Given a system of linear equations representing the cost of fruits,
    prove that the solution satisfies the given equations. -/
theorem fruit_cost_solution (x y z : ℝ) : 
  x + 2 * y = 8.9 ∧ 
  2 * z + 3 * y = 23 ∧ 
  3 * z + 4 * x = 30.1 →
  x = 2.5 ∧ y = 3.2 ∧ z = 6.7 := by
sorry

end NUMINAMATH_CALUDE_fruit_cost_solution_l174_17417


namespace NUMINAMATH_CALUDE_segment_length_l174_17433

/-- Given a line segment AB with points P and Q on it, prove that AB has length 120 -/
theorem segment_length (A B P Q : Real) : 
  (∃ x y u v : Real,
    -- P divides AB in ratio 3:5
    5 * x = 3 * y ∧ 
    -- Q divides AB in ratio 2:3
    3 * u = 2 * v ∧ 
    -- P is closer to A than Q
    u = x + 3 ∧ 
    v = y - 3 ∧ 
    -- AB is the sum of its parts
    A + B = x + y) → 
  A + B = 120 := by
sorry

end NUMINAMATH_CALUDE_segment_length_l174_17433


namespace NUMINAMATH_CALUDE_average_hamburgers_per_day_l174_17431

theorem average_hamburgers_per_day :
  let total_hamburgers : ℕ := 49
  let days_in_week : ℕ := 7
  let average := total_hamburgers / days_in_week
  average = 7 := by sorry

end NUMINAMATH_CALUDE_average_hamburgers_per_day_l174_17431


namespace NUMINAMATH_CALUDE_match_end_probability_l174_17497

/-- The probability of player A winning a single game -/
def prob_A_win : ℝ := 0.6

/-- The probability of player B winning a single game -/
def prob_B_win : ℝ := 0.4

/-- The probability that the match ends after two more games -/
def prob_match_ends : ℝ := prob_A_win * prob_A_win + prob_B_win * prob_B_win

/-- Theorem stating that the probability of the match ending after two more games is 0.52 -/
theorem match_end_probability : prob_match_ends = 0.52 := by
  sorry

end NUMINAMATH_CALUDE_match_end_probability_l174_17497


namespace NUMINAMATH_CALUDE_grandpa_xiaoqiang_age_relation_l174_17492

theorem grandpa_xiaoqiang_age_relation (x : ℕ) : 
  66 - x = 7 * (12 - x) ↔ 
  (∃ (grandpa_age xiaoqiang_age : ℕ), 
    grandpa_age = 66 ∧ 
    xiaoqiang_age = 12 ∧ 
    grandpa_age - x = 7 * (xiaoqiang_age - x)) :=
by sorry

end NUMINAMATH_CALUDE_grandpa_xiaoqiang_age_relation_l174_17492


namespace NUMINAMATH_CALUDE_problem_solution_l174_17401

theorem problem_solution (p q : ℝ) 
  (h1 : 1 < p) (h2 : p < q) 
  (h3 : 1 / p + 1 / q = 1) 
  (h4 : p * q = 8) : 
  q = 4 + 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l174_17401


namespace NUMINAMATH_CALUDE_willie_cream_total_l174_17444

/-- The amount of whipped cream Willie needs in total -/
def total_cream (farm_cream : ℕ) (bought_cream : ℕ) : ℕ :=
  farm_cream + bought_cream

/-- Theorem stating that Willie needs 300 lbs. of whipped cream in total -/
theorem willie_cream_total :
  total_cream 149 151 = 300 := by
  sorry

end NUMINAMATH_CALUDE_willie_cream_total_l174_17444


namespace NUMINAMATH_CALUDE_sum_of_k_values_for_distinct_integer_solutions_l174_17418

theorem sum_of_k_values_for_distinct_integer_solutions : ∃ (S : Finset ℤ), 
  (∀ k ∈ S, ∃ x y : ℤ, x ≠ y ∧ 3 * x^2 - k * x + 12 = 0 ∧ 3 * y^2 - k * y + 12 = 0) ∧ 
  (∀ k : ℤ, (∃ x y : ℤ, x ≠ y ∧ 3 * x^2 - k * x + 12 = 0 ∧ 3 * y^2 - k * y + 12 = 0) → k ∈ S) ∧
  (Finset.sum S id = 0) := by
sorry

end NUMINAMATH_CALUDE_sum_of_k_values_for_distinct_integer_solutions_l174_17418


namespace NUMINAMATH_CALUDE_shaded_fraction_is_37_72_l174_17434

/-- Represents a digit drawn on the grid -/
inductive Digit
  | one
  | nine
  | eight

/-- Represents the grid with drawn digits -/
structure Grid :=
  (rows : Nat)
  (cols : Nat)
  (digits : List Digit)

/-- Calculates the number of small squares occupied by a digit -/
def squaresOccupied (d : Digit) : Nat :=
  match d with
  | Digit.one => 8
  | Digit.nine => 15
  | Digit.eight => 16

/-- Calculates the total number of squares in the grid -/
def totalSquares (g : Grid) : Nat :=
  g.rows * g.cols

/-- Calculates the number of squares occupied by all digits -/
def occupiedSquares (g : Grid) : Nat :=
  g.digits.foldl (fun acc d => acc + squaresOccupied d) 0

/-- Represents the fraction of shaded area -/
def shadedFraction (g : Grid) : Rat :=
  occupiedSquares g / totalSquares g

theorem shaded_fraction_is_37_72 (g : Grid) 
  (h1 : g.rows = 18)
  (h2 : g.cols = 8)
  (h3 : g.digits = [Digit.one, Digit.nine, Digit.nine, Digit.eight]) :
  shadedFraction g = 37 / 72 := by
  sorry

#eval shadedFraction { rows := 18, cols := 8, digits := [Digit.one, Digit.nine, Digit.nine, Digit.eight] }

end NUMINAMATH_CALUDE_shaded_fraction_is_37_72_l174_17434


namespace NUMINAMATH_CALUDE_victoria_worked_five_weeks_l174_17412

/-- Calculates the number of weeks worked given the total hours and daily hours. -/
def weeksWorked (totalHours : ℕ) (dailyHours : ℕ) : ℚ :=
  (totalHours : ℚ) / (dailyHours * 7 : ℚ)

/-- Theorem: Victoria worked for 5 weeks -/
theorem victoria_worked_five_weeks :
  weeksWorked 315 9 = 5 := by sorry

end NUMINAMATH_CALUDE_victoria_worked_five_weeks_l174_17412


namespace NUMINAMATH_CALUDE_f_max_value_l174_17464

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt 3 / 2) * Real.sin (x + Real.pi / 2) + Real.cos (Real.pi / 6 - x)

theorem f_max_value : ∀ x : ℝ, f x ≤ Real.sqrt 13 / 2 ∧ ∃ y : ℝ, f y = Real.sqrt 13 / 2 := by sorry

end NUMINAMATH_CALUDE_f_max_value_l174_17464


namespace NUMINAMATH_CALUDE_jims_weight_l174_17429

theorem jims_weight (jim steve stan : ℕ) 
  (h1 : stan = steve + 5)
  (h2 : steve = jim - 8)
  (h3 : jim + steve + stan = 319) :
  jim = 110 := by
sorry

end NUMINAMATH_CALUDE_jims_weight_l174_17429


namespace NUMINAMATH_CALUDE_monotone_increasing_iff_a_in_range_l174_17435

/-- A quadratic function f(x) = ax^2 + 2x - 3 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 * x - 3

/-- The statement that f is monotonically increasing on (-∞, 4) iff a ∈ [-1/4, 0] -/
theorem monotone_increasing_iff_a_in_range (a : ℝ) :
  (∀ x y, x < y → x < 4 → f a x < f a y) ↔ -1/4 ≤ a ∧ a ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_monotone_increasing_iff_a_in_range_l174_17435


namespace NUMINAMATH_CALUDE_largest_vertex_sum_l174_17427

/-- Represents a parabola passing through specific points -/
structure Parabola where
  a : ℤ
  T : ℤ
  h : T ≠ 0

/-- The sum of coordinates of the vertex of the parabola -/
def vertexSum (p : Parabola) : ℤ := p.T - p.a * p.T^2

/-- The parabola passes through the point (2T+1, 28) -/
def passesThroughC (p : Parabola) : Prop :=
  p.a * (2 * p.T + 1) = 28

theorem largest_vertex_sum :
  ∀ p : Parabola, passesThroughC p → vertexSum p ≤ 60 :=
sorry

end NUMINAMATH_CALUDE_largest_vertex_sum_l174_17427


namespace NUMINAMATH_CALUDE_total_ninja_stars_l174_17403

-- Define the number of ninja throwing stars for each person
def eric_stars : ℕ := 4
def chad_stars_initial : ℕ := 2 * eric_stars
def jeff_bought : ℕ := 2
def jeff_stars_final : ℕ := 6

-- Define Chad's final number of stars
def chad_stars_final : ℕ := chad_stars_initial - jeff_bought

-- Theorem to prove
theorem total_ninja_stars :
  eric_stars + chad_stars_final + jeff_stars_final = 16 :=
by sorry

end NUMINAMATH_CALUDE_total_ninja_stars_l174_17403


namespace NUMINAMATH_CALUDE_child_ticket_cost_l174_17405

theorem child_ticket_cost (num_children num_adults : ℕ) (adult_ticket_cost total_cost : ℚ) :
  num_children = 6 →
  num_adults = 10 →
  adult_ticket_cost = 16 →
  total_cost = 220 →
  (total_cost - num_adults * adult_ticket_cost) / num_children = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_child_ticket_cost_l174_17405


namespace NUMINAMATH_CALUDE_abc_inequality_l174_17486

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_sum : a + b + c + 2 = a * b * c) :
  (a + 1) * (b + 1) * (c + 1) ≥ 27 ∧
  ((a + 1) * (b + 1) * (c + 1) = 27 ↔ a = 2 ∧ b = 2 ∧ c = 2) :=
by sorry

end NUMINAMATH_CALUDE_abc_inequality_l174_17486


namespace NUMINAMATH_CALUDE_smallest_staircase_steps_l174_17400

/-- The number of steps Cozy takes to climb the stairs -/
def cozy_jumps (n : ℕ) : ℕ := (n + 2) / 3

/-- The number of steps Dash takes to climb the stairs -/
def dash_jumps (n : ℕ) : ℕ := (n + 6) / 7

/-- Theorem stating the smallest number of steps in the staircase -/
theorem smallest_staircase_steps : 
  ∃ (n : ℕ), 
    n % 11 = 0 ∧ 
    cozy_jumps n - dash_jumps n = 13 ∧ 
    ∀ (m : ℕ), m < n → (m % 11 ≠ 0 ∨ cozy_jumps m - dash_jumps m ≠ 13) :=
by sorry

end NUMINAMATH_CALUDE_smallest_staircase_steps_l174_17400


namespace NUMINAMATH_CALUDE_quadratic_properties_l174_17432

/-- Quadratic function -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_properties
  (a b c : ℝ)
  (h_a : a ≠ 0)
  (h_down : a < 0)
  (h_b : b < 0)
  (h_c : c > 0)
  (h_sym : ∀ x, f a b c (x - 1) = f a b c (-x - 1)) :
  abc > 0 ∧
  (∀ x, -3 < x ∧ x < 1 → f a b c x > 0) ∧
  f a b c (-4) = -10/3 ∧
  f a b c 2 = -10/3 ∧
  f a b c 1 = 0 ∧
  f a b c (-3/2) = 5/2 ∧
  f a b c (-1/2) = 5/2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_properties_l174_17432


namespace NUMINAMATH_CALUDE_cats_and_dogs_sum_l174_17468

/-- Represents the number of individuals of each type on the ship --/
structure ShipPopulation where
  cats : ℕ
  parrots : ℕ
  dogs : ℕ
  sailors : ℕ
  cook : ℕ := 1
  captain : ℕ := 1

/-- The total number of heads on the ship --/
def totalHeads (p : ShipPopulation) : ℕ :=
  p.cats + p.parrots + p.dogs + p.sailors + p.cook + p.captain

/-- The total number of legs on the ship --/
def totalLegs (p : ShipPopulation) : ℕ :=
  4 * p.cats + 2 * p.parrots + 4 * p.dogs + 2 * p.sailors + 2 * p.cook + 1 * p.captain

/-- Theorem stating that the total number of cats and dogs is 14 --/
theorem cats_and_dogs_sum (p : ShipPopulation) 
    (h1 : totalHeads p = 38) 
    (h2 : totalLegs p = 103) : 
  p.cats + p.dogs = 14 := by
  sorry

end NUMINAMATH_CALUDE_cats_and_dogs_sum_l174_17468


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l174_17461

/-- Two vectors are parallel if and only if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b.1 = k * a.1 ∧ b.2 = k * a.2

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (4, 8)
  let b : ℝ × ℝ := (x, 4)
  parallel a b → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l174_17461


namespace NUMINAMATH_CALUDE_proposition_contrapositive_equivalence_l174_17495

theorem proposition_contrapositive_equivalence (P Q : Prop) :
  (P → Q) ↔ (¬Q → ¬P) := by sorry

end NUMINAMATH_CALUDE_proposition_contrapositive_equivalence_l174_17495


namespace NUMINAMATH_CALUDE_second_number_value_l174_17498

theorem second_number_value (x y z : ℝ) : 
  x + y + z = 660 ∧ 
  x = 2 * y ∧ 
  z = (1 / 3) * x → 
  y = 180 := by
  sorry

end NUMINAMATH_CALUDE_second_number_value_l174_17498


namespace NUMINAMATH_CALUDE_andrew_sandwiches_l174_17426

/-- The number of friends Andrew has coming over -/
def num_friends : ℕ := 4

/-- The number of sandwiches Andrew makes for each friend -/
def sandwiches_per_friend : ℕ := 3

/-- The total number of sandwiches Andrew made -/
def total_sandwiches : ℕ := num_friends * sandwiches_per_friend

theorem andrew_sandwiches : total_sandwiches = 12 := by
  sorry

end NUMINAMATH_CALUDE_andrew_sandwiches_l174_17426


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l174_17402

theorem fraction_to_decimal : (5 : ℚ) / 16 = 0.3125 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l174_17402


namespace NUMINAMATH_CALUDE_giraffe_difference_l174_17445

/-- In a zoo with giraffes and other animals, where the number of giraffes
    is 3 times the number of all other animals, prove that there are 200
    more giraffes than other animals. -/
theorem giraffe_difference (total_giraffes : ℕ) (other_animals : ℕ) : 
  total_giraffes = 300 →
  total_giraffes = 3 * other_animals →
  total_giraffes - other_animals = 200 :=
by sorry

end NUMINAMATH_CALUDE_giraffe_difference_l174_17445


namespace NUMINAMATH_CALUDE_square_difference_l174_17477

theorem square_difference (x y : ℝ) 
  (h1 : (x + y)^2 = 81) 
  (h2 : x * y = 18) : 
  (x - y)^2 = 9 := by
sorry

end NUMINAMATH_CALUDE_square_difference_l174_17477


namespace NUMINAMATH_CALUDE_tigers_games_count_l174_17467

theorem tigers_games_count :
  ∀ (initial_games : ℕ) (initial_wins : ℕ),
    initial_wins = (60 * initial_games) / 100 →
    ∀ (final_games : ℕ),
      final_games = initial_games + 11 →
      (initial_wins + 8) = (65 * final_games) / 100 →
      final_games = 28 := by
sorry

end NUMINAMATH_CALUDE_tigers_games_count_l174_17467


namespace NUMINAMATH_CALUDE_two_segment_journey_average_speed_l174_17478

/-- Calculates the average speed of a two-segment journey -/
theorem two_segment_journey_average_speed 
  (distance1 : ℝ) (speed1 : ℝ) (distance2 : ℝ) (speed2 : ℝ) 
  (h1 : distance1 = 20) (h2 : speed1 = 10) (h3 : distance2 = 30) (h4 : speed2 = 20) :
  (distance1 + distance2) / ((distance1 / speed1) + (distance2 / speed2)) = 50 / 3.5 := by
sorry

#eval (20 + 30) / ((20 / 10) + (30 / 20)) -- To verify the result

end NUMINAMATH_CALUDE_two_segment_journey_average_speed_l174_17478


namespace NUMINAMATH_CALUDE_complement_of_M_in_U_l174_17438

open Set

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2, 4}

theorem complement_of_M_in_U : 
  (U \ M) = {3, 5} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_in_U_l174_17438


namespace NUMINAMATH_CALUDE_hyperbola_locus_l174_17408

/-- The locus of points P satisfying |PM| - |PN| = 4, where M(-3, 0) and N(3, 0) are fixed points -/
def rightBranchHyperbola : Set (ℝ × ℝ) :=
  {P | ‖P - (-3, 0)‖ - ‖P - (3, 0)‖ = 4 ∧ P.1 > 3}

/-- Theorem stating that the locus of points P satisfying |PM| - |PN| = 4 
    is the right branch of a hyperbola with foci M(-3, 0) and N(3, 0) -/
theorem hyperbola_locus :
  ∀ P : ℝ × ℝ, P ∈ rightBranchHyperbola ↔ 
    (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 
      (P.1 / a)^2 - (P.2 / b)^2 = 1 ∧
      a^2 - b^2 = 9 ∧
      P.1 > 3) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_locus_l174_17408


namespace NUMINAMATH_CALUDE_f_odd_and_decreasing_l174_17476

-- Define the function f(x) = -x|x|
def f (x : ℝ) : ℝ := -x * abs x

-- Theorem statement
theorem f_odd_and_decreasing :
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x y : ℝ, x < y → f x > f y) :=
by sorry

end NUMINAMATH_CALUDE_f_odd_and_decreasing_l174_17476


namespace NUMINAMATH_CALUDE_sum_a_b_value_l174_17422

theorem sum_a_b_value (a b : ℚ) (h1 : 2 * a + 5 * b = 43) (h2 : 8 * a + 2 * b = 50) :
  a + b = 34 / 3 := by
sorry

end NUMINAMATH_CALUDE_sum_a_b_value_l174_17422


namespace NUMINAMATH_CALUDE_unique_six_digit_number_l174_17440

/-- A function that checks if a number is a 6-digit number beginning and ending with 2 -/
def is_valid_number (n : ℕ) : Prop :=
  100000 ≤ n ∧ n < 1000000 ∧ n % 10 = 2 ∧ n / 100000 = 2

/-- A function that checks if a number is the product of three consecutive even integers -/
def is_product_of_three_consecutive_even (n : ℕ) : Prop :=
  ∃ k : ℕ, n = (2*k) * (2*k + 2) * (2*k + 4)

theorem unique_six_digit_number : 
  ∀ n : ℕ, is_valid_number n ∧ is_product_of_three_consecutive_even n ↔ n = 287232 :=
sorry

end NUMINAMATH_CALUDE_unique_six_digit_number_l174_17440


namespace NUMINAMATH_CALUDE_square_sum_geq_product_l174_17484

theorem square_sum_geq_product {a b c d : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h : 2 * (a + b + c + d) ≥ a * b * c * d) :
  a^2 + b^2 + c^2 + d^2 ≥ a * b * c * d := by
  sorry

end NUMINAMATH_CALUDE_square_sum_geq_product_l174_17484


namespace NUMINAMATH_CALUDE_g_is_even_l174_17407

noncomputable def g (x : ℝ) : ℝ := Real.log (x^2 + Real.sqrt (1 + x^4))

theorem g_is_even : ∀ x : ℝ, g (-x) = g x := by sorry

end NUMINAMATH_CALUDE_g_is_even_l174_17407


namespace NUMINAMATH_CALUDE_square_number_ones_digit_l174_17420

/-- A number is a square number if it's the square of an integer -/
def IsSquareNumber (a : ℕ) : Prop := ∃ x : ℕ, a = x^2

/-- Get the tens digit of a natural number -/
def TensDigit (n : ℕ) : ℕ := (n / 10) % 10

/-- Get the ones digit of a natural number -/
def OnesDigit (n : ℕ) : ℕ := n % 10

/-- A number is odd if it's not divisible by 2 -/
def IsOdd (n : ℕ) : Prop := n % 2 = 1

theorem square_number_ones_digit
  (a : ℕ)
  (h1 : IsSquareNumber a)
  (h2 : IsOdd (TensDigit a)) :
  OnesDigit a = 6 := by
  sorry

end NUMINAMATH_CALUDE_square_number_ones_digit_l174_17420


namespace NUMINAMATH_CALUDE_conditional_probability_l174_17454

-- Define the probability measure z
variable (z : Set α → ℝ)

-- Define events x and y
variable (x y : Set α)

-- State the theorem
theorem conditional_probability
  (hx : z x = 0.02)
  (hy : z y = 0.10)
  (hxy : z (x ∩ y) = 0.10)
  (h_prob : ∀ A, 0 ≤ z A ∧ z A ≤ 1)
  (h_add : ∀ A B, z (A ∪ B) = z A + z B - z (A ∩ B))
  : z x / z y = 1 :=
sorry

end NUMINAMATH_CALUDE_conditional_probability_l174_17454


namespace NUMINAMATH_CALUDE_octadecagon_diagonals_l174_17471

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- An octadecagon has 18 sides -/
def octadecagon_sides : ℕ := 18

theorem octadecagon_diagonals :
  num_diagonals octadecagon_sides = 135 := by
  sorry

end NUMINAMATH_CALUDE_octadecagon_diagonals_l174_17471


namespace NUMINAMATH_CALUDE_line_m_equation_l174_17423

/-- Two distinct lines in the xy-plane -/
structure TwoLines where
  ℓ : Set (ℝ × ℝ)
  m : Set (ℝ × ℝ)
  distinct : ℓ ≠ m
  intersect_origin : (0, 0) ∈ ℓ ∩ m

/-- Equation of a line in the form ax + by = 0 -/
def LineEquation (a b : ℝ) : Set (ℝ × ℝ) :=
  {(x, y) | a * x + b * y = 0}

/-- Reflection of a point about a line -/
def reflect (p : ℝ × ℝ) (line : Set (ℝ × ℝ)) : ℝ × ℝ := sorry

theorem line_m_equation (lines : TwoLines) 
  (h_ℓ : lines.ℓ = LineEquation 2 1)
  (h_Q : reflect (3, -2) lines.ℓ = reflect (reflect (3, -2) lines.ℓ) lines.m)
  (h_Q'' : reflect (reflect (3, -2) lines.ℓ) lines.m = (-1, 5)) :
  lines.m = LineEquation 3 1 := by sorry

end NUMINAMATH_CALUDE_line_m_equation_l174_17423


namespace NUMINAMATH_CALUDE_point_C_coordinates_l174_17414

def A : ℝ × ℝ := (-2, 1)
def B : ℝ × ℝ := (4, 9)

theorem point_C_coordinates :
  ∀ C : ℝ × ℝ,
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ C = (1 - t) • A + t • B) →  -- C lies on segment AB
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = 16 * ((B.1 - C.1)^2 + (B.2 - C.2)^2) →  -- AC = 4CB
  C = (8/5, 14/5) :=
by sorry

end NUMINAMATH_CALUDE_point_C_coordinates_l174_17414


namespace NUMINAMATH_CALUDE_polynomial_root_product_l174_17459

theorem polynomial_root_product (d e f : ℝ) : 
  let Q : ℝ → ℝ := λ x ↦ x^3 + d*x^2 + e*x + f
  (Q (Real.cos (π/5)) = 0) ∧ 
  (Q (Real.cos (3*π/5)) = 0) ∧ 
  (Q (Real.cos (4*π/5)) = 0) →
  d * e * f = 0 := by
sorry

end NUMINAMATH_CALUDE_polynomial_root_product_l174_17459


namespace NUMINAMATH_CALUDE_f_value_at_one_l174_17485

/-- The polynomial g(x) -/
def g (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + x + 10

/-- The polynomial f(x) -/
def f (b c : ℝ) (x : ℝ) : ℝ := x^4 + x^3 + b*x^2 + 100*x + c

theorem f_value_at_one (a b c : ℝ) : 
  (∃ r₁ r₂ r₃ : ℝ, r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₂ ≠ r₃ ∧ 
    g a r₁ = 0 ∧ g a r₂ = 0 ∧ g a r₃ = 0) →
  (∀ x : ℝ, g a x = 0 → f b c x = 0) →
  f b c 1 = -7007 := by
  sorry

end NUMINAMATH_CALUDE_f_value_at_one_l174_17485


namespace NUMINAMATH_CALUDE_abs_sum_zero_implies_sum_l174_17421

theorem abs_sum_zero_implies_sum (x y : ℝ) :
  |x - 1| + |y + 3| = 0 → x + y = -2 := by
sorry

end NUMINAMATH_CALUDE_abs_sum_zero_implies_sum_l174_17421


namespace NUMINAMATH_CALUDE_algebra_test_correct_percentage_l174_17490

theorem algebra_test_correct_percentage (x : ℕ) (h : x > 0) :
  let total_problems := 5 * x
  let missed_problems := x
  let correct_problems := total_problems - missed_problems
  (correct_problems : ℚ) / (total_problems : ℚ) * 100 = 80 := by
  sorry

end NUMINAMATH_CALUDE_algebra_test_correct_percentage_l174_17490


namespace NUMINAMATH_CALUDE_square_area_ratio_l174_17491

theorem square_area_ratio (x : ℝ) (hx : x > 0) : 
  (x^2) / ((3*x)^2 + (6*x)^2) = 1/45 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_l174_17491


namespace NUMINAMATH_CALUDE_smallest_cube_ending_392_l174_17488

theorem smallest_cube_ending_392 : 
  ∃ (n : ℕ), n > 0 ∧ n^3 % 1000 = 392 ∧ ∀ (m : ℕ), m > 0 ∧ m^3 % 1000 = 392 → n ≤ m :=
by
  use 22
  sorry

end NUMINAMATH_CALUDE_smallest_cube_ending_392_l174_17488


namespace NUMINAMATH_CALUDE_marble_probability_l174_17443

/-- The probability of drawing 1 blue marble and 2 black marbles from a basket -/
theorem marble_probability (blue yellow black : ℕ) 
  (h_blue : blue = 4)
  (h_yellow : yellow = 6)
  (h_black : black = 7) :
  let total := blue + yellow + black
  (blue : ℚ) / total * (black * (black - 1) : ℚ) / ((total - 1) * (total - 2)) = 7 / 170 := by
  sorry

end NUMINAMATH_CALUDE_marble_probability_l174_17443


namespace NUMINAMATH_CALUDE_grandma_last_birthday_age_l174_17487

/-- Represents Grandma's age in various units -/
structure GrandmaAge where
  years : Nat
  months : Nat
  weeks : Nat
  days : Nat

/-- Calculates Grandma's age on her last birthday given her current age -/
def lastBirthdayAge (age : GrandmaAge) : Nat :=
  age.years + (age.months / 12) + 1

/-- Theorem stating that Grandma's age on her last birthday was 65 years -/
theorem grandma_last_birthday_age :
  let currentAge : GrandmaAge := { years := 60, months := 50, weeks := 40, days := 30 }
  lastBirthdayAge currentAge = 65 := by
  sorry

#eval lastBirthdayAge { years := 60, months := 50, weeks := 40, days := 30 }

end NUMINAMATH_CALUDE_grandma_last_birthday_age_l174_17487


namespace NUMINAMATH_CALUDE_sum_nine_is_negative_fiftyfour_l174_17499

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℤ
  first_term : a 1 = 2
  fifth_term : a 5 = 3 * a 3
  is_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1

/-- Sum of first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  n * (2 * seq.a 1 + (n - 1) * (seq.a 2 - seq.a 1)) / 2

/-- Theorem: The sum of the first 9 terms of the given arithmetic sequence is -54 -/
theorem sum_nine_is_negative_fiftyfour (seq : ArithmeticSequence) : sum_n seq 9 = -54 := by
  sorry

end NUMINAMATH_CALUDE_sum_nine_is_negative_fiftyfour_l174_17499


namespace NUMINAMATH_CALUDE_birthday_height_calculation_l174_17493

/-- Given an initial height and a growth rate, calculates the new height -/
def new_height (initial_height : ℝ) (growth_rate : ℝ) : ℝ :=
  initial_height * (1 + growth_rate)

/-- Proves that given an initial height of 119.7 cm and a growth rate of 5%,
    the new height is 125.685 cm -/
theorem birthday_height_calculation :
  new_height 119.7 0.05 = 125.685 := by
  sorry

end NUMINAMATH_CALUDE_birthday_height_calculation_l174_17493


namespace NUMINAMATH_CALUDE_work_rate_problem_l174_17425

theorem work_rate_problem (A B C D : ℚ) :
  A = 1 / 4 →
  A + C = 1 / 2 →
  B + C = 1 / 3 →
  D = 1 / 5 →
  A + B + C + D = 1 →
  B = 13 / 60 :=
by sorry

end NUMINAMATH_CALUDE_work_rate_problem_l174_17425


namespace NUMINAMATH_CALUDE_smallest_price_with_tax_l174_17437

theorem smallest_price_with_tax (n : ℕ) (x : ℕ) : n = 21 ↔ 
  n > 0 ∧ 
  (∀ m : ℕ, m > 0 → m < n → ¬∃ y : ℕ, y > 0 ∧ 105 * y = 100 * m * 100) ∧
  x > 0 ∧ 
  105 * x = 100 * n * 100 :=
sorry

end NUMINAMATH_CALUDE_smallest_price_with_tax_l174_17437


namespace NUMINAMATH_CALUDE_factorization_equality_l174_17462

theorem factorization_equality (x y : ℝ) : 
  (x + y)^2 - 14*(x + y) + 49 = (x + y - 7)^2 := by sorry

end NUMINAMATH_CALUDE_factorization_equality_l174_17462


namespace NUMINAMATH_CALUDE_leon_order_total_l174_17469

/-- Calculates the total amount Leon paid for his order, including discounts and delivery fee. -/
def total_paid (toy_organizer_price : ℚ) (toy_organizer_count : ℕ) 
                (gaming_chair_price : ℚ) (gaming_chair_count : ℕ)
                (desk_price : ℚ) (bookshelf_price : ℚ)
                (toy_organizer_discount : ℚ) (gaming_chair_discount : ℚ)
                (delivery_fee_rate : ℚ → ℚ) : ℚ :=
  let toy_organizer_total := toy_organizer_price * toy_organizer_count * (1 - toy_organizer_discount)
  let gaming_chair_total := gaming_chair_price * gaming_chair_count * (1 - gaming_chair_discount)
  let subtotal := toy_organizer_total + gaming_chair_total + desk_price + bookshelf_price
  let total_items := toy_organizer_count + gaming_chair_count + 2
  let delivery_fee := subtotal * delivery_fee_rate total_items
  subtotal + delivery_fee

/-- The statement to be proved -/
theorem leon_order_total :
  let toy_organizer_price : ℚ := 78
  let toy_organizer_count : ℕ := 3
  let gaming_chair_price : ℚ := 83
  let gaming_chair_count : ℕ := 2
  let desk_price : ℚ := 120
  let bookshelf_price : ℚ := 95
  let toy_organizer_discount : ℚ := 0.1
  let gaming_chair_discount : ℚ := 0.05
  let delivery_fee_rate (items : ℚ) : ℚ :=
    if items ≤ 3 then 0.04
    else if items ≤ 5 then 0.06
    else 0.08
  total_paid toy_organizer_price toy_organizer_count 
             gaming_chair_price gaming_chair_count
             desk_price bookshelf_price
             toy_organizer_discount gaming_chair_discount
             delivery_fee_rate = 629.96 := by
  sorry

end NUMINAMATH_CALUDE_leon_order_total_l174_17469


namespace NUMINAMATH_CALUDE_john_needs_60_bags_l174_17481

/-- Calculates the number of half-ton bags of horse food needed for a given number of horses, 
    feedings per day, pounds per feeding, and number of days. --/
def bags_needed (num_horses : ℕ) (feedings_per_day : ℕ) (pounds_per_feeding : ℕ) (days : ℕ) : ℕ :=
  let daily_food_per_horse := feedings_per_day * pounds_per_feeding
  let total_daily_food := daily_food_per_horse * num_horses
  let total_food := total_daily_food * days
  let bag_weight := 1000  -- half-ton in pounds
  total_food / bag_weight

/-- Theorem stating that John needs 60 bags of food for his horses over 60 days. --/
theorem john_needs_60_bags : 
  bags_needed 25 2 20 60 = 60 := by
  sorry

end NUMINAMATH_CALUDE_john_needs_60_bags_l174_17481


namespace NUMINAMATH_CALUDE_arithmetic_progression_sum_l174_17439

/-- Given an arithmetic progression where the sum of the 4th and 12th terms is 10,
    prove that the sum of the first 15 terms is 75. -/
theorem arithmetic_progression_sum (a d : ℝ) : 
  (a + 3*d) + (a + 11*d) = 10 → 
  (15 : ℝ) / 2 * (2*a + 14*d) = 75 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_sum_l174_17439


namespace NUMINAMATH_CALUDE_quadratic_root_sum_l174_17424

/-- Given that 3i - 2 is a root of the quadratic equation 2x^2 + px + q = 0,
    prove that p + q = 34. -/
theorem quadratic_root_sum (p q : ℝ) : 
  (2 * (Complex.I * 3 - 2)^2 + p * (Complex.I * 3 - 2) + q = 0) →
  p + q = 34 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_sum_l174_17424


namespace NUMINAMATH_CALUDE_investment_calculation_l174_17475

/-- Represents the total investment amount in dollars -/
def total_investment : ℝ := 22000

/-- Represents the amount invested at 8% interest rate in dollars -/
def investment_at_8_percent : ℝ := 17000

/-- Represents the total interest earned in dollars -/
def total_interest : ℝ := 1710

/-- Represents the interest rate for the 8% investment -/
def rate_8_percent : ℝ := 0.08

/-- Represents the interest rate for the 7% investment -/
def rate_7_percent : ℝ := 0.07

theorem investment_calculation :
  rate_8_percent * investment_at_8_percent +
  rate_7_percent * (total_investment - investment_at_8_percent) =
  total_interest :=
sorry

end NUMINAMATH_CALUDE_investment_calculation_l174_17475


namespace NUMINAMATH_CALUDE_mikes_land_profit_l174_17466

/-- Calculates the profit from a land development project -/
def calculate_profit (total_acres : ℕ) (purchase_price_per_acre : ℕ) (sell_price_per_acre : ℕ) : ℕ :=
  let total_cost := total_acres * purchase_price_per_acre
  let acres_sold := total_acres / 2
  let total_revenue := acres_sold * sell_price_per_acre
  total_revenue - total_cost

/-- Proves that the profit from Mike's land development project is $6,000 -/
theorem mikes_land_profit :
  calculate_profit 200 70 200 = 6000 := by
  sorry

#eval calculate_profit 200 70 200

end NUMINAMATH_CALUDE_mikes_land_profit_l174_17466


namespace NUMINAMATH_CALUDE_janet_paper_clips_used_l174_17494

/-- Calculates the number of paper clips Janet used during the day -/
def paperClipsUsed (initial : ℝ) (found : ℝ) (givenPerFriend : ℝ) (numFriends : ℕ) (final : ℝ) : ℝ :=
  initial + found - givenPerFriend * (numFriends : ℝ) - final

/-- Theorem stating that Janet used 62.5 paper clips during the day -/
theorem janet_paper_clips_used :
  paperClipsUsed 85 17.5 3.5 4 26 = 62.5 := by
  sorry

#eval paperClipsUsed 85 17.5 3.5 4 26

end NUMINAMATH_CALUDE_janet_paper_clips_used_l174_17494


namespace NUMINAMATH_CALUDE_chair_capacity_l174_17430

theorem chair_capacity (total_chairs : ℕ) (attended : ℕ) : 
  total_chairs = 40 →
  (2 : ℚ) / 5 * total_chairs = total_chairs - (3 : ℚ) / 5 * total_chairs →
  2 * ((3 : ℚ) / 5 * total_chairs) = attended →
  attended = 48 →
  ∃ (capacity : ℕ), capacity = 48 ∧ capacity * total_chairs = capacity * attended :=
by
  sorry

end NUMINAMATH_CALUDE_chair_capacity_l174_17430


namespace NUMINAMATH_CALUDE_cymbal_strike_interval_l174_17456

def beats_between_triangle_strikes : ℕ := 2
def lcm_cymbal_triangle_strikes : ℕ := 14

theorem cymbal_strike_interval :
  ∃ (c : ℕ), c > 0 ∧ Nat.lcm c beats_between_triangle_strikes = lcm_cymbal_triangle_strikes ∧ c = 14 := by
  sorry

end NUMINAMATH_CALUDE_cymbal_strike_interval_l174_17456


namespace NUMINAMATH_CALUDE_M_intersect_N_eq_singleton_one_l174_17410

def M : Set ℕ := {0, 1, 2}

def N : Set ℕ := {x | ∃ a : ℕ+, x = 2 * a - 1}

theorem M_intersect_N_eq_singleton_one : M ∩ N = {1} := by
  sorry

end NUMINAMATH_CALUDE_M_intersect_N_eq_singleton_one_l174_17410


namespace NUMINAMATH_CALUDE_perpendicular_line_through_P_l174_17457

-- Define the given line
def given_line (x y : ℝ) : Prop := x + 2*y - 1 = 0

-- Define the point P
def point_P : ℝ × ℝ := (-1, 2)

-- Define the perpendicular line
def perpendicular_line (x y : ℝ) : Prop := 2*x - y + 4 = 0

theorem perpendicular_line_through_P : 
  (∀ x y : ℝ, given_line x y → (∃ m : ℝ, m * (2*x - y) = -1)) ∧ 
  perpendicular_line point_P.1 point_P.2 :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_P_l174_17457


namespace NUMINAMATH_CALUDE_greatest_integer_radius_l174_17450

theorem greatest_integer_radius (A : ℝ) (h : A < 200 * Real.pi) :
  ∃ (r : ℕ), r * r * Real.pi ≤ A ∧ ∀ (s : ℕ), s * s * Real.pi ≤ A → s ≤ r :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_radius_l174_17450


namespace NUMINAMATH_CALUDE_rice_division_l174_17482

/-- Proves that dividing 25/4 pounds of rice equally among 4 containers results in 25 ounces per container. -/
theorem rice_division (total_weight : ℚ) (num_containers : ℕ) (pound_to_ounce : ℕ) :
  total_weight = 25 / 4 →
  num_containers = 4 →
  pound_to_ounce = 16 →
  (total_weight / num_containers) * pound_to_ounce = 25 := by
  sorry

end NUMINAMATH_CALUDE_rice_division_l174_17482


namespace NUMINAMATH_CALUDE_function_value_at_log_third_l174_17446

/-- Given a function f and a real number a, proves that f(ln(1/3)) = -1 -/
theorem function_value_at_log_third (a : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ 2^x / (2^x + 1) + a * x
  f (Real.log 3) = 2 → f (Real.log (1/3)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_log_third_l174_17446


namespace NUMINAMATH_CALUDE_hen_egg_production_l174_17455

/-- Given the following conditions:
    - There are 10 hens
    - Eggs are sold for $3 per dozen
    - In 4 weeks, $120 worth of eggs were sold
    Prove that each hen lays 12 eggs per week. -/
theorem hen_egg_production 
  (num_hens : ℕ) 
  (price_per_dozen : ℚ) 
  (weeks : ℕ) 
  (total_sales : ℚ) 
  (h1 : num_hens = 10)
  (h2 : price_per_dozen = 3)
  (h3 : weeks = 4)
  (h4 : total_sales = 120) :
  (total_sales / price_per_dozen * 12 / weeks / num_hens : ℚ) = 12 := by
sorry


end NUMINAMATH_CALUDE_hen_egg_production_l174_17455
