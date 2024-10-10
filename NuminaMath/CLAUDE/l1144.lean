import Mathlib

namespace zero_variance_median_equals_mean_l1144_114447

-- Define a sample as a finite multiset of real numbers
def Sample := Multiset ℝ

-- Define the variance of a sample
def variance (s : Sample) : ℝ := sorry

-- Define the median of a sample
def median (s : Sample) : ℝ := sorry

-- Define the mean of a sample
def mean (s : Sample) : ℝ := sorry

-- Theorem statement
theorem zero_variance_median_equals_mean (s : Sample) (a : ℝ) :
  variance s = 0 ∧ median s = a → mean s = a := by sorry

end zero_variance_median_equals_mean_l1144_114447


namespace hexagon_perimeter_l1144_114498

/-- The perimeter of a regular hexagon inscribed in a circle -/
theorem hexagon_perimeter (r : ℝ) (h : r = 10) : 
  6 * (2 * r * Real.sin (π / 6)) = 60 := by
  sorry

#check hexagon_perimeter

end hexagon_perimeter_l1144_114498


namespace binary_to_octal_conversion_l1144_114490

/-- Converts a base 2 number to base 10 --/
def binary_to_decimal (b : List Bool) : Nat :=
  b.foldr (fun bit acc => 2 * acc + if bit then 1 else 0) 0

/-- Converts a base 10 number to base 8 --/
def decimal_to_octal (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc
      else aux (m / 8) ((m % 8) :: acc)
    aux n []

theorem binary_to_octal_conversion :
  decimal_to_octal (binary_to_decimal [true, true, false, true, true, false, true, true, false]) = [6, 6, 6] := by
  sorry

end binary_to_octal_conversion_l1144_114490


namespace pump_fill_time_l1144_114449

/-- The time it takes for the pump to fill the tank without the leak -/
def pump_time : ℝ := 2

/-- The time it takes for the pump and leak together to fill the tank -/
def combined_time : ℝ := 2.8

/-- The time it takes for the leak to empty the full tank -/
def leak_time : ℝ := 7

theorem pump_fill_time :
  (1 / pump_time) - (1 / leak_time) = (1 / combined_time) :=
by sorry

end pump_fill_time_l1144_114449


namespace kate_emily_hair_ratio_l1144_114428

/-- The ratio of hair lengths -/
def hair_length_ratio (kate_length emily_length : ℕ) : ℚ :=
  kate_length / emily_length

/-- Theorem stating the ratio of Kate's hair length to Emily's hair length -/
theorem kate_emily_hair_ratio :
  let logan_length : ℕ := 20
  let emily_length : ℕ := logan_length + 6
  let kate_length : ℕ := 7
  hair_length_ratio kate_length emily_length = 7 / 26 := by
sorry

end kate_emily_hair_ratio_l1144_114428


namespace opposite_sides_line_range_l1144_114483

/-- Given that points (-3,-1) and (4,-6) are on opposite sides of the line 3x-2y-a=0,
    the range of values for a is (-7, 24). -/
theorem opposite_sides_line_range (a : ℝ) : 
  (∃ (x₁ y₁ x₂ y₂ : ℝ), x₁ = -3 ∧ y₁ = -1 ∧ x₂ = 4 ∧ y₂ = -6 ∧ 
    (3*x₁ - 2*y₁ - a) * (3*x₂ - 2*y₂ - a) < 0) ↔ 
  -7 < a ∧ a < 24 :=
sorry

end opposite_sides_line_range_l1144_114483


namespace system_solution_l1144_114436

theorem system_solution (x y : ℝ) (eq1 : 2 * x + y = 6) (eq2 : x + 2 * y = 3) : x - y = 3 := by
  sorry

end system_solution_l1144_114436


namespace manuscript_revision_problem_l1144_114476

/-- The number of pages revised twice in a manuscript -/
def pages_revised_twice (total_pages : ℕ) (pages_revised_once : ℕ) (total_cost : ℕ) : ℕ :=
  (total_cost - (5 * total_pages + 3 * pages_revised_once)) / 6

/-- Theorem stating the number of pages revised twice -/
theorem manuscript_revision_problem (total_pages : ℕ) (pages_revised_once : ℕ) (total_cost : ℕ) 
  (h1 : total_pages = 200)
  (h2 : pages_revised_once = 80)
  (h3 : total_cost = 1360) :
  pages_revised_twice total_pages pages_revised_once total_cost = 20 := by
sorry

#eval pages_revised_twice 200 80 1360

end manuscript_revision_problem_l1144_114476


namespace decimal_25_to_binary_binary_to_decimal_25_l1144_114429

/-- Represents a binary digit (0 or 1) -/
inductive BinaryDigit
| zero : BinaryDigit
| one : BinaryDigit

/-- Represents a binary number as a list of binary digits -/
def BinaryNumber := List BinaryDigit

/-- Converts a decimal number to its binary representation -/
def decimalToBinary (n : ℕ) : BinaryNumber :=
  sorry

/-- Converts a binary number to its decimal representation -/
def binaryToDecimal (b : BinaryNumber) : ℕ :=
  sorry

theorem decimal_25_to_binary :
  decimalToBinary 25 = [BinaryDigit.one, BinaryDigit.one, BinaryDigit.zero, BinaryDigit.zero, BinaryDigit.one] :=
by sorry

theorem binary_to_decimal_25 :
  binaryToDecimal [BinaryDigit.one, BinaryDigit.one, BinaryDigit.zero, BinaryDigit.zero, BinaryDigit.one] = 25 :=
by sorry

end decimal_25_to_binary_binary_to_decimal_25_l1144_114429


namespace min_value_theorem_l1144_114492

theorem min_value_theorem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 27) :
  ∃ (min : ℝ), min = 30 ∧ ∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 → a * b * c = 27 → 
    a^2 + 3*b + 6*c ≥ min := by
  sorry

#check min_value_theorem

end min_value_theorem_l1144_114492


namespace perfect_square_properties_l1144_114463

theorem perfect_square_properties (a : ℕ) :
  (∀ n : ℕ, n > 0 → a ∈ ({1, 2, 4} : Set ℕ) → ¬∃ m : ℕ, n * (a + n) = m^2) ∧
  ((∃ k : ℕ, k ≥ 3 ∧ a = 2^k) → ∃ n : ℕ, n > 0 ∧ ∃ m : ℕ, n * (a + n) = m^2) ∧
  (a ∉ ({1, 2, 4} : Set ℕ) → ∃ n : ℕ, n > 0 ∧ ∃ m : ℕ, n * (a + n) = m^2) :=
by sorry

end perfect_square_properties_l1144_114463


namespace spider_reachable_points_l1144_114409

/-- A cube with edge length 1 -/
structure Cube where
  edge_length : ℝ
  edge_length_pos : edge_length = 1

/-- A point on the surface of a cube -/
structure CubePoint (c : Cube) where
  x : ℝ
  y : ℝ
  z : ℝ
  on_surface : (x = 0 ∨ x = c.edge_length ∨ y = 0 ∨ y = c.edge_length ∨ z = 0 ∨ z = c.edge_length) ∧
               0 ≤ x ∧ x ≤ c.edge_length ∧
               0 ≤ y ∧ y ≤ c.edge_length ∧
               0 ≤ z ∧ z ≤ c.edge_length

/-- The distance between two points on the surface of a cube -/
def surface_distance (c : Cube) (p1 p2 : CubePoint c) : ℝ :=
  sorry -- Definition of surface distance calculation

/-- The set of points reachable by the spider in 2 seconds -/
def reachable_points (c : Cube) (start : CubePoint c) : Set (CubePoint c) :=
  {p : CubePoint c | surface_distance c start p ≤ 2}

/-- Theorem: The set of points reachable by the spider in 2 seconds
    is equivalent to the set of points within 2 cm of the starting vertex -/
theorem spider_reachable_points (c : Cube) (start : CubePoint c) :
  reachable_points c start = {p : CubePoint c | surface_distance c start p ≤ 2} :=
by sorry

end spider_reachable_points_l1144_114409


namespace greatest_x_value_l1144_114475

theorem greatest_x_value (x : ℤ) (h : (3.134 : ℝ) * (10 : ℝ) ^ (x : ℝ) < 31000) :
  x ≤ 3 ∧ ∃ y : ℤ, y > 3 → (3.134 : ℝ) * (10 : ℝ) ^ (y : ℝ) ≥ 31000 :=
by sorry

end greatest_x_value_l1144_114475


namespace exists_problem_solved_by_all_l1144_114441

/-- Represents a problem on the exam -/
def Problem : Type := ℕ

/-- Represents a student in the class -/
def Student : Type := ℕ

/-- Given n students and 2^(n-1) problems, if for each pair of distinct problems
    there is at least one student who has solved both and at least one student
    who has solved one but not the other, then there exists a problem solved by
    all n students. -/
theorem exists_problem_solved_by_all
  (n : ℕ)
  (problems : Finset Problem)
  (students : Finset Student)
  (solved : Problem → Student → Prop)
  (h_num_students : students.card = n)
  (h_num_problems : problems.card = 2^(n-1))
  (h_solved_both : ∀ p q : Problem, p ≠ q →
    ∃ s : Student, solved p s ∧ solved q s)
  (h_solved_one_not_other : ∀ p q : Problem, p ≠ q →
    ∃ s : Student, (solved p s ∧ ¬solved q s) ∨ (solved q s ∧ ¬solved p s)) :
  ∃ p : Problem, ∀ s : Student, solved p s :=
sorry

end exists_problem_solved_by_all_l1144_114441


namespace chessboard_tiling_impossible_l1144_114438

/-- Represents a chessboard tile -/
inductive Tile
| L
| T

/-- Represents the color distribution of a tile placement -/
structure ColorDistribution :=
  (black : ℕ)
  (white : ℕ)

/-- The color distribution of an L-tile -/
def l_tile_distribution : ColorDistribution :=
  ⟨2, 2⟩

/-- The possible color distributions of a T-tile -/
def t_tile_distributions : List ColorDistribution :=
  [⟨3, 1⟩, ⟨1, 3⟩]

/-- The number of squares on the chessboard -/
def chessboard_squares : ℕ := 64

/-- The number of black squares on the chessboard -/
def chessboard_black_squares : ℕ := 32

/-- The number of white squares on the chessboard -/
def chessboard_white_squares : ℕ := 32

/-- The number of L-tiles -/
def num_l_tiles : ℕ := 15

/-- The number of T-tiles -/
def num_t_tiles : ℕ := 1

theorem chessboard_tiling_impossible :
  ∀ (t_dist : ColorDistribution),
    t_dist ∈ t_tile_distributions →
    (num_l_tiles * l_tile_distribution.black + t_dist.black ≠ chessboard_black_squares ∨
     num_l_tiles * l_tile_distribution.white + t_dist.white ≠ chessboard_white_squares) :=
by sorry

end chessboard_tiling_impossible_l1144_114438


namespace min_value_on_circle_l1144_114453

theorem min_value_on_circle (x y : ℝ) (h : (x + 5)^2 + (y - 12)^2 = 14^2) :
  ∃ (m : ℝ), (∀ (a b : ℝ), (a + 5)^2 + (b - 12)^2 = 14^2 → x^2 + y^2 ≤ a^2 + b^2) ∧ m = 1 :=
by sorry

end min_value_on_circle_l1144_114453


namespace cubic_factorization_l1144_114482

theorem cubic_factorization (x : ℝ) : x^3 - 16*x = x*(x+4)*(x-4) := by
  sorry

end cubic_factorization_l1144_114482


namespace largest_common_divisor_414_345_l1144_114451

theorem largest_common_divisor_414_345 : Nat.gcd 414 345 = 69 := by
  sorry

end largest_common_divisor_414_345_l1144_114451


namespace selfie_count_l1144_114489

theorem selfie_count (last_year this_year : ℕ) : 
  (this_year : ℚ) / last_year = 17 / 10 →
  this_year - last_year = 630 →
  last_year + this_year = 2430 :=
by
  sorry

end selfie_count_l1144_114489


namespace smallest_five_digit_multiple_of_18_l1144_114478

theorem smallest_five_digit_multiple_of_18 : 
  ∀ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ 18 ∣ n → 10008 ≤ n :=
by sorry

end smallest_five_digit_multiple_of_18_l1144_114478


namespace lune_area_minus_square_l1144_114479

theorem lune_area_minus_square (r1 r2 s : ℝ) : r1 = 2 → r2 = 1 → s = 1 →
  (π * r1^2 / 2 - π * r2^2 / 2) - s^2 = 3 * π / 2 - 1 := by
  sorry

end lune_area_minus_square_l1144_114479


namespace angstadt_student_count_l1144_114472

/-- Given that:
  1. Half of Mr. Angstadt's students are enrolled in Statistics.
  2. 90% of the students in Statistics are seniors.
  3. There are 54 seniors enrolled in Statistics.
  Prove that Mr. Angstadt has 120 students throughout the school day. -/
theorem angstadt_student_count :
  ∀ (total_students stats_students seniors : ℕ),
  stats_students = total_students / 2 →
  seniors = (90 * stats_students) / 100 →
  seniors = 54 →
  total_students = 120 :=
by sorry

end angstadt_student_count_l1144_114472


namespace proposition_p_q_range_l1144_114488

theorem proposition_p_q_range (x a : ℝ) : 
  (∀ x, x^2 ≤ 5*x - 4 → x^2 - (a + 2)*x + 2*a ≤ 0) ∧ 
  (∃ x, x^2 ≤ 5*x - 4 ∧ x^2 - (a + 2)*x + 2*a > 0) →
  1 ≤ a ∧ a ≤ 4 := by
sorry

end proposition_p_q_range_l1144_114488


namespace teal_color_survey_l1144_114469

theorem teal_color_survey (total : ℕ) (green : ℕ) (both : ℕ) (neither : ℕ) :
  total = 150 →
  green = 90 →
  both = 40 →
  neither = 20 →
  ∃ blue : ℕ, blue = 80 ∧ blue + green - both + neither = total :=
by sorry

end teal_color_survey_l1144_114469


namespace sum_of_squares_is_384_l1144_114496

/-- Represents the rates of cycling, jogging, and swimming -/
structure Rates where
  cycling : ℕ
  jogging : ℕ
  swimming : ℕ

/-- The conditions of the problem -/
def satisfies_conditions (r : Rates) : Prop :=
  -- Rates are even
  r.cycling % 2 = 0 ∧ r.jogging % 2 = 0 ∧ r.swimming % 2 = 0 ∧
  -- Ed's distance equation
  3 * r.cycling + 4 * r.jogging + 2 * r.swimming = 88 ∧
  -- Sue's distance equation
  2 * r.cycling + 3 * r.jogging + 4 * r.swimming = 104

/-- The theorem to be proved -/
theorem sum_of_squares_is_384 :
  ∃ r : Rates, satisfies_conditions r ∧ 
    r.cycling^2 + r.jogging^2 + r.swimming^2 = 384 := by
  sorry

end sum_of_squares_is_384_l1144_114496


namespace balance_forces_l1144_114433

/-- A force is represented by a pair of real numbers -/
def Force : Type := ℝ × ℝ

/-- Addition of forces -/
def add_forces (f1 f2 : Force) : Force :=
  (f1.1 + f2.1, f1.2 + f2.2)

/-- The zero force -/
def zero_force : Force := (0, 0)

/-- Two forces are balanced by a third force if their sum is the zero force -/
def balances (f1 f2 f3 : Force) : Prop :=
  add_forces (add_forces f1 f2) f3 = zero_force

theorem balance_forces :
  let f1 : Force := (1, 1)
  let f2 : Force := (2, 3)
  let f3 : Force := (-3, -4)
  balances f1 f2 f3 := by sorry

end balance_forces_l1144_114433


namespace equation_seven_solutions_l1144_114457

-- Define the equation
def equation (a x : ℝ) : Prop :=
  Real.sin (Real.sqrt (a^2 - x^2 - 2*x - 1)) = 0.5

-- Define the number of distinct solutions
def has_seven_distinct_solutions (a : ℝ) : Prop :=
  ∃ (s : Finset ℝ), s.card = 7 ∧ (∀ x ∈ s, equation a x) ∧
    (∀ x : ℝ, equation a x → x ∈ s)

-- State the theorem
theorem equation_seven_solutions :
  ∀ a : ℝ, has_seven_distinct_solutions a ↔ a = 17 * Real.pi / 6 := by
  sorry

end equation_seven_solutions_l1144_114457


namespace savings_ratio_first_year_l1144_114494

/-- Represents the financial situation of a person over two years -/
structure FinancialSituation where
  firstYearIncome : ℝ
  firstYearSavingsRatio : ℝ
  incomeIncrease : ℝ
  savingsIncrease : ℝ

/-- The theorem stating the savings ratio in the first year -/
theorem savings_ratio_first_year 
  (fs : FinancialSituation)
  (h1 : fs.incomeIncrease = 0.3)
  (h2 : fs.savingsIncrease = 1.0)
  (h3 : fs.firstYearIncome > 0)
  (h4 : 0 ≤ fs.firstYearSavingsRatio ∧ fs.firstYearSavingsRatio ≤ 1) :
  let firstYearExpenditure := fs.firstYearIncome * (1 - fs.firstYearSavingsRatio)
  let secondYearIncome := fs.firstYearIncome * (1 + fs.incomeIncrease)
  let secondYearSavings := fs.firstYearIncome * fs.firstYearSavingsRatio * (1 + fs.savingsIncrease)
  let secondYearExpenditure := secondYearIncome - secondYearSavings
  firstYearExpenditure + secondYearExpenditure = 2 * firstYearExpenditure →
  fs.firstYearSavingsRatio = 0.3 := by
sorry

end savings_ratio_first_year_l1144_114494


namespace part_one_part_two_l1144_114499

noncomputable section

open Real

-- Define the function f
def f (a b x : ℝ) : ℝ := a * log x - b * x^2

-- Define the tangent line equation
def tangent_line (x : ℝ) : ℝ := -3 * x + 2 * log 2 + 2

-- Part 1: Prove that a = 2 and b = 1
theorem part_one : 
  ∀ a b : ℝ, (∀ x : ℝ, f a b x = f a b 2 + (x - 2) * (-3)) → a = 2 ∧ b = 1 := 
by sorry

-- Define the function h for part 2
def h (x m : ℝ) : ℝ := 2 * log x - x^2 + m

-- Part 2: Prove the range of m
theorem part_two : 
  ∀ m : ℝ, (∃ x y : ℝ, 1/exp 1 ≤ x ∧ x < y ∧ y ≤ exp 1 ∧ h x m = 0 ∧ h y m = 0) 
  → 1 < m ∧ m ≤ 1/(exp 1)^2 + 2 := 
by sorry

end

end part_one_part_two_l1144_114499


namespace fraction_inequality_l1144_114412

theorem fraction_inequality (x : ℝ) : (x + 6) / (x^2 + 2*x + 7) ≥ 0 ↔ x ≥ -6 := by
  sorry

end fraction_inequality_l1144_114412


namespace f_fixed_points_l1144_114424

def f (x : ℝ) : ℝ := x^2 - 5*x + 6

theorem f_fixed_points : {x : ℝ | f (f x) = f x} = {0, 2, 3, 5} := by
  sorry

end f_fixed_points_l1144_114424


namespace simplify_expression_1_simplify_expression_2_l1144_114468

-- Problem 1
theorem simplify_expression_1 (a : ℝ) : 2 * a^2 - 3*a - 5*a^2 + 6*a = -3*a^2 + 3*a := by
  sorry

-- Problem 2
theorem simplify_expression_2 (a : ℝ) : 2*(a-1) - (2*a-3) + 3 = 4 := by
  sorry

end simplify_expression_1_simplify_expression_2_l1144_114468


namespace factor_t_squared_minus_81_l1144_114467

theorem factor_t_squared_minus_81 (t : ℝ) : t^2 - 81 = (t - 9) * (t + 9) := by
  sorry

end factor_t_squared_minus_81_l1144_114467


namespace dice_roll_probability_l1144_114485

/-- The probability of rolling a specific number on a standard six-sided die -/
def prob_single_roll : ℚ := 1 / 6

/-- The probability of not rolling a 1 on a single die -/
def prob_not_one : ℚ := 5 / 6

/-- The total number of possible outcomes when rolling two dice -/
def total_outcomes : ℕ := 36

/-- The number of favorable outcomes (numbers between 10 and 20 inclusive) -/
def favorable_outcomes : ℕ := 11

theorem dice_roll_probability : 
  (1 : ℚ) - prob_not_one * prob_not_one = favorable_outcomes / total_outcomes := by
  sorry

end dice_roll_probability_l1144_114485


namespace solve_system_l1144_114448

theorem solve_system (p q : ℚ) 
  (eq1 : 5 * p + 3 * q = 10) 
  (eq2 : 3 * p + 5 * q = 20) : 
  q = 35 / 8 := by
  sorry

end solve_system_l1144_114448


namespace crates_needed_is_fifteen_l1144_114426

/-- Calculates the number of crates needed to load items in a warehouse --/
def calculate_crates (crate_capacity : ℕ) (nail_bags : ℕ) (nail_weight : ℕ) 
  (hammer_bags : ℕ) (hammer_weight : ℕ) (plank_bags : ℕ) (plank_weight : ℕ) 
  (left_out_weight : ℕ) : ℕ :=
  let total_weight := nail_bags * nail_weight + hammer_bags * hammer_weight + plank_bags * plank_weight
  let loadable_weight := total_weight - left_out_weight
  (loadable_weight + crate_capacity - 1) / crate_capacity

/-- Theorem stating that given the problem conditions, 15 crates are needed --/
theorem crates_needed_is_fifteen :
  calculate_crates 20 4 5 12 5 10 30 80 = 15 := by
  sorry

end crates_needed_is_fifteen_l1144_114426


namespace draw_three_with_red_standard_deck_l1144_114410

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : Nat)
  (suits : Nat)
  (cards_per_suit : Nat)
  (black_suits : Nat)
  (red_suits : Nat)

/-- Calculate the number of ways to draw three cards with at least one red card -/
def draw_three_with_red (d : Deck) : Nat :=
  d.total_cards * (d.total_cards - 1) * (d.total_cards - 2) - 
  (d.black_suits * d.cards_per_suit) * (d.black_suits * d.cards_per_suit - 1) * (d.black_suits * d.cards_per_suit - 2)

/-- Theorem: The number of ways to draw three cards with at least one red from a standard deck is 117000 -/
theorem draw_three_with_red_standard_deck :
  let standard_deck : Deck := {
    total_cards := 52,
    suits := 4,
    cards_per_suit := 13,
    black_suits := 2,
    red_suits := 2
  }
  draw_three_with_red standard_deck = 117000 := by
  sorry

end draw_three_with_red_standard_deck_l1144_114410


namespace prove_equation_l1144_114425

theorem prove_equation (c d : ℝ) 
  (h1 : 5 + c = 6 - d) 
  (h2 : 3 + d = 8 + c) : 
  5 - c = 7 := by
sorry

end prove_equation_l1144_114425


namespace triangle_inequality_l1144_114465

theorem triangle_inequality (a b c : ℝ) (h_triangle : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  a^3 / c^3 + b^3 / c^3 + 3 * a * b / c^2 > 1 := by
  sorry

end triangle_inequality_l1144_114465


namespace incorrect_operator_is_second_l1144_114486

def original_expression : List Int := [3, 5, -7, 9, -11, 13, -15, 17]

def calculate (expr : List Int) : Int :=
  expr.foldl (· + ·) 0

def flip_operator (expr : List Int) (index : Nat) : List Int :=
  expr.mapIdx (fun i x => if i == index then -x else x)

theorem incorrect_operator_is_second :
  ∃ (i : Nat), i < original_expression.length ∧
    calculate (flip_operator original_expression i) = -4 ∧
    i = 1 ∧
    ∀ (j : Nat), j < original_expression.length → j ≠ i →
      calculate (flip_operator original_expression j) ≠ -4 := by
  sorry

end incorrect_operator_is_second_l1144_114486


namespace arithmetic_sequence_1005th_term_l1144_114422

/-- An arithmetic sequence with specific conditions -/
def ArithmeticSequence (p r : ℚ) : ℕ → ℚ
  | 0 => p
  | 1 => 10
  | 2 => 4 * p - r
  | 3 => 4 * p + r
  | n + 4 => ArithmeticSequence p r 3 + (n + 1) * (ArithmeticSequence p r 3 - ArithmeticSequence p r 2)

/-- The 1005th term of the arithmetic sequence is 5480 -/
theorem arithmetic_sequence_1005th_term (p r : ℚ) :
  ArithmeticSequence p r 1004 = 5480 := by
  sorry

end arithmetic_sequence_1005th_term_l1144_114422


namespace recurring_decimal_equals_fraction_l1144_114417

/-- The decimal representation of 3.127̄ as a rational number -/
def recurring_decimal : ℚ := 3 + 127 / 999

/-- The fraction 3124/999 -/
def target_fraction : ℚ := 3124 / 999

/-- Theorem stating that the recurring decimal 3.127̄ is equal to the fraction 3124/999 -/
theorem recurring_decimal_equals_fraction : recurring_decimal = target_fraction := by
  sorry

end recurring_decimal_equals_fraction_l1144_114417


namespace quadratic_equation_solution_l1144_114493

theorem quadratic_equation_solution (y : ℝ) : 
  y^2 + 6*y + 8 = -(y + 4)*(y + 6) ↔ y = -4 := by
  sorry

end quadratic_equation_solution_l1144_114493


namespace alex_coin_distribution_l1144_114434

/-- The number of friends Alex has -/
def num_friends : ℕ := 15

/-- The initial number of coins Alex has -/
def initial_coins : ℕ := 100

/-- The minimum number of coins needed to distribute to friends -/
def min_coins_needed : ℕ := (num_friends * (num_friends + 1)) / 2

/-- The number of additional coins needed -/
def additional_coins_needed : ℕ := min_coins_needed - initial_coins

theorem alex_coin_distribution :
  additional_coins_needed = 20 := by sorry

end alex_coin_distribution_l1144_114434


namespace sufficient_not_necessary_l1144_114420

/-- A sequence of real numbers -/
def Sequence := ℕ → ℝ

/-- A sequence is arithmetic if the difference between consecutive terms is constant -/
def is_arithmetic (a : Sequence) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- Points lie on the line y = x + 1 -/
def points_on_line (a : Sequence) : Prop :=
  ∀ n : ℕ+, a n = n + 1

theorem sufficient_not_necessary :
  (∀ a : Sequence, points_on_line a → is_arithmetic a) ∧
  (∃ a : Sequence, is_arithmetic a ∧ ¬points_on_line a) :=
sorry

end sufficient_not_necessary_l1144_114420


namespace quadratic_equation_properties_l1144_114401

theorem quadratic_equation_properties (k : ℝ) :
  let f : ℝ → ℝ := λ x => x^2 - (k + 1) * x - 6
  (∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0) ∧
  (f 2 = 0 → k = -2 ∧ f (-3) = 0) := by
  sorry

end quadratic_equation_properties_l1144_114401


namespace arcsin_cos_arcsin_plus_arccos_sin_arccos_l1144_114408

theorem arcsin_cos_arcsin_plus_arccos_sin_arccos (x : ℝ) : 
  Real.arcsin (Real.cos (Real.arcsin x)) + Real.arccos (Real.sin (Real.arccos x)) = π / 2 := by
  sorry

end arcsin_cos_arcsin_plus_arccos_sin_arccos_l1144_114408


namespace product_sum_of_digits_77_sevens_77_threes_l1144_114454

/-- Represents a string of digits repeated n times -/
def RepeatedDigitString (digit : Nat) (n : Nat) : Nat :=
  -- Definition omitted for brevity
  sorry

/-- Calculates the sum of digits in a natural number -/
def sumOfDigits (n : Nat) : Nat :=
  -- Definition omitted for brevity
  sorry

/-- The main theorem to prove -/
theorem product_sum_of_digits_77_sevens_77_threes :
  sumOfDigits (RepeatedDigitString 7 77 * RepeatedDigitString 3 77) = 231 := by
  sorry

end product_sum_of_digits_77_sevens_77_threes_l1144_114454


namespace cubic_root_sum_squares_l1144_114481

theorem cubic_root_sum_squares (p q r : ℝ) (x y z : ℝ) : 
  (x^3 - p*x^2 + q*x - r = 0) → 
  (y^3 - p*y^2 + q*y - r = 0) → 
  (z^3 - p*z^2 + q*z - r = 0) → 
  (x + y + z = p) →
  (x*y + x*z + y*z = q) →
  x^2 + y^2 + z^2 = p^2 - 2*q := by
sorry

end cubic_root_sum_squares_l1144_114481


namespace money_split_l1144_114430

theorem money_split (total : ℝ) (ratio_small : ℕ) (ratio_large : ℕ) (smaller_share : ℝ) :
  total = 125 →
  ratio_small = 2 →
  ratio_large = 3 →
  smaller_share = (ratio_small : ℝ) / ((ratio_small : ℝ) + (ratio_large : ℝ)) * total →
  smaller_share = 50 := by
sorry

end money_split_l1144_114430


namespace odd_digits_345_base5_l1144_114497

/-- Counts the number of odd digits in a base-5 number -/
def countOddDigitsBase5 (n : ℕ) : ℕ := sorry

/-- Converts a base-10 number to base-5 -/
def toBase5 (n : ℕ) : ℕ := sorry

theorem odd_digits_345_base5 :
  countOddDigitsBase5 (toBase5 345) = 2 := by sorry

end odd_digits_345_base5_l1144_114497


namespace inequality_proof_l1144_114464

theorem inequality_proof (p : ℝ) (h1 : 18 * p < 10) (h2 : p > 0.5) : 0.5 < p ∧ p < 5/9 := by
  sorry

end inequality_proof_l1144_114464


namespace a_squared_gt_one_sufficient_not_necessary_l1144_114404

-- Define the equation
def is_ellipse (a : ℝ) : Prop :=
  ∃ x y : ℝ, x^2 / a^2 + y^2 = 1 ∧ (x ≠ 0 ∨ y ≠ 0)

-- Theorem statement
theorem a_squared_gt_one_sufficient_not_necessary :
  (∀ a : ℝ, a^2 > 1 → is_ellipse a) ∧
  (∃ a : ℝ, is_ellipse a ∧ a^2 ≤ 1) :=
sorry

end a_squared_gt_one_sufficient_not_necessary_l1144_114404


namespace alexander_pencil_count_l1144_114495

/-- The number of pencils Alexander uses for all exhibitions -/
def total_pencils (initial_pictures : ℕ) (new_galleries : ℕ) (pictures_per_new_gallery : ℕ) 
  (pencils_per_picture : ℕ) (pencils_for_signing : ℕ) : ℕ :=
  let total_pictures := initial_pictures + new_galleries * pictures_per_new_gallery
  let pencils_for_drawing := total_pictures * pencils_per_picture
  let total_exhibitions := 1 + new_galleries
  let pencils_for_all_signings := total_exhibitions * pencils_for_signing
  pencils_for_drawing + pencils_for_all_signings

/-- Theorem stating that Alexander uses 88 pencils in total -/
theorem alexander_pencil_count : 
  total_pencils 9 5 2 4 2 = 88 := by
  sorry


end alexander_pencil_count_l1144_114495


namespace integer_root_theorem_l1144_114484

def polynomial (x b : ℤ) : ℤ := x^3 + 4*x^2 + b*x + 12

def has_integer_root (b : ℤ) : Prop :=
  ∃ x : ℤ, polynomial x b = 0

def valid_b_values : Set ℤ := {-177, -62, -35, -25, -18, -17, 9, 16, 27, 48, 144, 1296}

theorem integer_root_theorem :
  ∀ b : ℤ, has_integer_root b ↔ b ∈ valid_b_values :=
by sorry

end integer_root_theorem_l1144_114484


namespace sequence_properties_l1144_114474

def S (n : ℕ+) : ℤ := 3 * n - 2 * n^2

def a (n : ℕ+) : ℤ := -4 * n + 5

theorem sequence_properties :
  ∀ n : ℕ+,
  (∀ k : ℕ+, k ≤ n → S k - S (k-1) = a k) ∧
  S n ≥ n * a n :=
sorry

end sequence_properties_l1144_114474


namespace island_population_theorem_l1144_114415

/-- Represents the number of turtles and rabbits on an island -/
structure IslandPopulation where
  turtles : ℕ
  rabbits : ℕ

/-- Represents the populations of the four islands -/
structure IslandSystem where
  happy : IslandPopulation
  lonely : IslandPopulation
  serene : IslandPopulation
  tranquil : IslandPopulation

/-- Theorem stating the conditions and the result to be proven -/
theorem island_population_theorem (islands : IslandSystem) : 
  (islands.happy.turtles = 120) →
  (islands.happy.rabbits = 80) →
  (islands.lonely.turtles = islands.happy.turtles / 3) →
  (islands.lonely.rabbits = islands.lonely.turtles) →
  (islands.serene.rabbits = 2 * islands.lonely.rabbits) →
  (islands.serene.turtles = 3 * islands.lonely.rabbits / 4) →
  (islands.tranquil.turtles = islands.tranquil.rabbits) →
  (islands.tranquil.turtles = 
    (islands.happy.turtles - islands.serene.turtles) + 5) →
  (islands.happy.turtles + islands.lonely.turtles + 
   islands.serene.turtles + islands.tranquil.turtles = 285) ∧
  (islands.happy.rabbits + islands.lonely.rabbits + 
   islands.serene.rabbits + islands.tranquil.rabbits = 295) := by
  sorry

end island_population_theorem_l1144_114415


namespace total_movie_time_is_172_l1144_114456

/-- Represents a segment of movie watching, including the time spent watching and rewinding --/
structure MovieSegment where
  watchTime : ℕ
  rewindTime : ℕ

/-- Calculates the total time for a movie segment --/
def segmentTime (segment : MovieSegment) : ℕ :=
  segment.watchTime + segment.rewindTime

/-- The sequence of movie segments as described in the problem --/
def movieSegments : List MovieSegment := [
  ⟨30, 5⟩,
  ⟨20, 7⟩,
  ⟨10, 12⟩,
  ⟨15, 8⟩,
  ⟨25, 15⟩,
  ⟨15, 10⟩
]

/-- Theorem stating that the total time to watch the movie is 172 minutes --/
theorem total_movie_time_is_172 :
  (movieSegments.map segmentTime).sum = 172 := by
  sorry

end total_movie_time_is_172_l1144_114456


namespace remaining_bottles_l1144_114480

theorem remaining_bottles (small_initial big_initial : ℕ) 
  (small_percent big_percent : ℚ) : 
  small_initial = 6000 →
  big_initial = 10000 →
  small_percent = 12 / 100 →
  big_percent = 15 / 100 →
  (small_initial - small_initial * small_percent).floor +
  (big_initial - big_initial * big_percent).floor = 13780 := by
  sorry

end remaining_bottles_l1144_114480


namespace f_monotone_increasing_min_m_value_l1144_114471

open Real

noncomputable def f (x : ℝ) : ℝ := (x + 1/x) * log x

theorem f_monotone_increasing :
  StrictMono f := by sorry

theorem min_m_value (m : ℝ) :
  (∀ x > 0, (2 * f x - m) / (exp (m * x)) ≤ m) ↔ m ≥ 2/exp 1 := by sorry

end f_monotone_increasing_min_m_value_l1144_114471


namespace smallest_x_value_l1144_114431

theorem smallest_x_value : 
  let f (x : ℚ) := 7 * (4 * x^2 + 4 * x + 5) - x * (4 * x - 35)
  ∃ (x : ℚ), f x = 0 ∧ ∀ (y : ℚ), f y = 0 → x ≤ y ∧ x = -5/3 :=
by sorry

end smallest_x_value_l1144_114431


namespace solution_set_f_range_of_m_l1144_114461

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1|

-- Part I
theorem solution_set_f (x : ℝ) : 
  (|f x - 3| ≤ 4) ↔ (-6 ≤ x ∧ x ≤ 8) := by sorry

-- Part II
theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, f x + f (x + 3) ≥ m^2 - 2*m) ↔ (-1 ≤ m ∧ m ≤ 3) := by sorry

end solution_set_f_range_of_m_l1144_114461


namespace drops_used_proof_l1144_114400

/-- Represents the number of drops used to test a single beaker -/
def drops_per_beaker : ℕ := 3

/-- Represents the total number of beakers -/
def total_beakers : ℕ := 22

/-- Represents the number of beakers with copper ions -/
def copper_beakers : ℕ := 8

/-- Represents the number of beakers without copper ions that were tested -/
def tested_non_copper : ℕ := 7

theorem drops_used_proof :
  drops_per_beaker * (copper_beakers + tested_non_copper) = 45 := by
  sorry

end drops_used_proof_l1144_114400


namespace duck_profit_l1144_114446

/-- Calculates the profit from buying and selling ducks -/
theorem duck_profit
  (num_ducks : ℕ)
  (cost_per_duck : ℝ)
  (weight_per_duck : ℝ)
  (sell_price_per_pound : ℝ)
  (h1 : num_ducks = 30)
  (h2 : cost_per_duck = 10)
  (h3 : weight_per_duck = 4)
  (h4 : sell_price_per_pound = 5) :
  let total_cost := num_ducks * cost_per_duck
  let total_weight := num_ducks * weight_per_duck
  let total_revenue := total_weight * sell_price_per_pound
  total_revenue - total_cost = 300 :=
by sorry

end duck_profit_l1144_114446


namespace repeating_decimal_to_fraction_l1144_114443

theorem repeating_decimal_to_fraction :
  ∀ (x : ℚ), (∃ (n : ℕ), x = 2 + (35 / 100) * (1 / (1 - 1/100)^n)) →
  x = 233 / 99 := by
  sorry

end repeating_decimal_to_fraction_l1144_114443


namespace negation_of_implication_l1144_114403

theorem negation_of_implication (A B : Set α) :
  ¬(A ∪ B = A → A ∩ B = B) ↔ (A ∪ B = A ∧ A ∩ B ≠ B) :=
by sorry

end negation_of_implication_l1144_114403


namespace return_speed_calculation_l1144_114473

theorem return_speed_calculation (total_distance : ℝ) (outbound_speed : ℝ) (average_speed : ℝ) 
  (h1 : total_distance = 300) 
  (h2 : outbound_speed = 75) 
  (h3 : average_speed = 50) :
  ∃ inbound_speed : ℝ, 
    inbound_speed = 37.5 ∧ 
    average_speed = total_distance / (total_distance / (2 * outbound_speed) + total_distance / (2 * inbound_speed)) := by
  sorry

end return_speed_calculation_l1144_114473


namespace max_safe_sages_is_82_l1144_114437

/-- Represents a train with a given number of wagons. -/
structure Train :=
  (num_wagons : ℕ)

/-- Represents the journey details. -/
structure Journey :=
  (start_station : ℕ)
  (end_station : ℕ)
  (controller_start : ℕ)
  (controller_move_interval : ℕ)

/-- Represents the movement capabilities of sages. -/
structure SageMovement :=
  (max_move : ℕ)

/-- Represents the visibility range of sages. -/
structure SageVisibility :=
  (range : ℕ)

/-- Represents the maximum number of sages that can avoid controllers. -/
def max_safe_sages (t : Train) (j : Journey) (sm : SageMovement) (sv : SageVisibility) : ℕ :=
  82

/-- Theorem stating that 82 is the maximum number of sages that can avoid controllers. -/
theorem max_safe_sages_is_82 
  (t : Train) 
  (j : Journey) 
  (sm : SageMovement) 
  (sv : SageVisibility) : 
  max_safe_sages t j sm sv = 82 :=
by sorry

end max_safe_sages_is_82_l1144_114437


namespace train_length_l1144_114439

/-- The length of a train given its speed, platform length, and crossing time --/
theorem train_length (train_speed : ℝ) (platform_length : ℝ) (crossing_time : ℝ) :
  train_speed = 72 * (5 / 18) →
  platform_length = 230 →
  crossing_time = 26 →
  train_speed * crossing_time - platform_length = 290 := by
sorry

end train_length_l1144_114439


namespace sum_of_squares_l1144_114440

theorem sum_of_squares (a b c : ℝ) 
  (eq1 : a^2 + 2*b = 7)
  (eq2 : b^2 + 4*c = -7)
  (eq3 : c^2 + 6*a = -14) :
  a^2 + b^2 + c^2 = 14 := by sorry

end sum_of_squares_l1144_114440


namespace m_squared_divisors_l1144_114455

/-- A number with exactly 4 divisors -/
def HasFourDivisors (m : ℕ) : Prop :=
  (Finset.filter (· ∣ m) (Finset.range (m + 1))).card = 4

/-- The number of divisors of a natural number -/
def NumberOfDivisors (n : ℕ) : ℕ :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

theorem m_squared_divisors (m : ℕ) (h : HasFourDivisors m) : 
  NumberOfDivisors (m^2) = 7 := by
  sorry

end m_squared_divisors_l1144_114455


namespace tan_405_degrees_l1144_114421

theorem tan_405_degrees : Real.tan (405 * π / 180) = 1 := by
  sorry

end tan_405_degrees_l1144_114421


namespace circle_equation_l1144_114416

/-- The line to which the circle is tangent -/
def tangent_line (x y : ℝ) : Prop := x + y - 2 = 0

/-- A circle centered at the origin -/
def circle_at_origin (x y r : ℝ) : Prop := x^2 + y^2 = r^2

/-- The circle is tangent to the line -/
def is_tangent (r : ℝ) : Prop := ∃ x y : ℝ, tangent_line x y ∧ circle_at_origin x y r

theorem circle_equation : 
  ∃ r : ℝ, is_tangent r → circle_at_origin x y 2 :=
sorry

end circle_equation_l1144_114416


namespace arithmetic_sequence_30th_term_l1144_114419

/-- A sequence {a_n} with sum S_n satisfying the given conditions -/
def ArithmeticSequence (a : ℕ → ℚ) (S : ℕ → ℚ) : Prop :=
  (∀ n : ℕ, 3 * S n / n + n = 3 * a n + 1) ∧ (a 1 = -1/3)

/-- Theorem stating that the 30th term of the sequence is 19 -/
theorem arithmetic_sequence_30th_term
  (a : ℕ → ℚ) (S : ℕ → ℚ) (h : ArithmeticSequence a S) :
  a 30 = 19 := by
  sorry

end arithmetic_sequence_30th_term_l1144_114419


namespace cuts_through_examples_l1144_114450

/-- A line cuts through a curve at a point if it's tangent to the curve at that point
    and the curve lies on both sides of the line near that point. -/
def cuts_through (l : ℝ → ℝ) (c : ℝ → ℝ) (p : ℝ × ℝ) : Prop :=
  (∀ x, l x = c x → x = p.1) ∧  -- l is tangent to c at p
  (∃ ε > 0, ∀ x, |x - p.1| < ε → 
    ((c x > l x ∧ x < p.1) ∨ (c x < l x ∧ x > p.1) ∨
     (c x < l x ∧ x < p.1) ∨ (c x > l x ∧ x > p.1)))

theorem cuts_through_examples :
  (cuts_through (λ _ ↦ 0) (λ x ↦ x^3) (0, 0)) ∧
  (cuts_through (λ x ↦ x) Real.sin (0, 0)) ∧
  (cuts_through (λ x ↦ x) Real.tan (0, 0)) :=
sorry

end cuts_through_examples_l1144_114450


namespace charge_account_interest_l1144_114423

/-- Calculates the total amount owed after one year with simple interest -/
def total_amount_owed (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal + principal * rate * time

/-- Proves that the total amount owed after one year on a $75 charge with 7% simple annual interest is $80.25 -/
theorem charge_account_interest : total_amount_owed 75 0.07 1 = 80.25 := by
  sorry

end charge_account_interest_l1144_114423


namespace eleven_points_form_120_triangles_l1144_114414

/-- The number of triangles formed by 11 points on two segments -/
def numTriangles (n m : ℕ) : ℕ :=
  n * m * (m - 1) / 2 + m * n * (n - 1) / 2 + (n * (n - 1) * (n - 2)) / 6

/-- Theorem stating that 11 points on two segments (7 on one, 4 on another) form 120 triangles -/
theorem eleven_points_form_120_triangles :
  numTriangles 7 4 = 120 := by
  sorry

end eleven_points_form_120_triangles_l1144_114414


namespace cube_volume_doubling_l1144_114491

theorem cube_volume_doubling (a : ℝ) (h : a > 0) :
  (2 * a) ^ 3 = 8 * a ^ 3 :=
by sorry

end cube_volume_doubling_l1144_114491


namespace smallest_other_integer_l1144_114487

theorem smallest_other_integer (a b x : ℕ) : 
  a > 0 → b > 0 → x > 0 → a = 72 → 
  Nat.gcd a b = x + 6 → 
  Nat.lcm a b = 2 * x * (x + 6) → 
  b ≥ 24 ∧ (∃ (y : ℕ), y > 0 ∧ y + 6 ∣ 72 ∧ 
    Nat.gcd 72 24 = y + 6 ∧ 
    Nat.lcm 72 24 = 2 * y * (y + 6)) :=
by sorry

end smallest_other_integer_l1144_114487


namespace line_passes_through_circle_center_l1144_114402

-- Define the line equation
def line_equation (x y m : ℝ) : Prop := x - y + m = 0

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 2*y = 0

-- Define the center of the circle
def circle_center (x y : ℝ) : Prop := circle_equation x y ∧ ∀ x' y', circle_equation x' y' → (x - x')^2 + (y - y')^2 ≤ (x' - x)^2 + (y' - y)^2

-- Theorem statement
theorem line_passes_through_circle_center :
  ∃ x y : ℝ, circle_center x y ∧ line_equation x y (-3) :=
sorry

end line_passes_through_circle_center_l1144_114402


namespace problem_solution_l1144_114444

noncomputable def f (a : ℝ) (x : ℝ) := a * Real.log x - x + 1

theorem problem_solution :
  (∃! a : ℝ, ∀ x > 0, f a x ≤ 0) ∧
  (∀ x ∈ Set.Ioo 0 (Real.pi / 2), Real.exp x * Real.sin x - x > f 1 x) := by
  sorry

end problem_solution_l1144_114444


namespace sum_divisible_by_143_l1144_114442

theorem sum_divisible_by_143 : ∃ k : ℕ, (1000 * 1001) / 2 = 143 * k := by
  sorry

end sum_divisible_by_143_l1144_114442


namespace closest_integer_to_seven_times_three_fourths_l1144_114460

theorem closest_integer_to_seven_times_three_fourths : 
  ∃ (n : ℤ), ∀ (m : ℤ), |n - (7 * 3 / 4)| ≤ |m - (7 * 3 / 4)| ∧ n = 5 :=
by sorry

end closest_integer_to_seven_times_three_fourths_l1144_114460


namespace matrix_commutation_l1144_114462

open Matrix

theorem matrix_commutation (A B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h1 : A + B = A * B) 
  (h2 : A * B = ![![5, 1], ![-2, 2]]) : 
  B * A = A * B := by sorry

end matrix_commutation_l1144_114462


namespace point_A_coordinates_l1144_114452

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the translation operation
def translate (p : Point2D) (dx dy : ℝ) : Point2D :=
  { x := p.x + dx, y := p.y + dy }

theorem point_A_coordinates : 
  ∀ A : Point2D, 
  let B := translate (translate A 0 (-3)) 2 0
  B = Point2D.mk (-1) 5 → A = Point2D.mk (-3) 8 := by
sorry

end point_A_coordinates_l1144_114452


namespace greatest_common_divisor_and_sum_of_digits_l1144_114470

def numbers : List Nat := [23115, 34365, 83197, 153589]

def differences (nums : List Nat) : List Nat :=
  List.map (λ (pair : Nat × Nat) => pair.2 - pair.1) (List.zip nums (List.tail nums))

def sumOfDigits (n : Nat) : Nat :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

theorem greatest_common_divisor_and_sum_of_digits :
  let diffs := differences numbers
  let n := diffs.foldl Nat.gcd (diffs.head!)
  n = 1582 ∧ sumOfDigits n = 16 := by
  sorry

end greatest_common_divisor_and_sum_of_digits_l1144_114470


namespace kai_born_in_1995_l1144_114406

/-- Kai's birth year, given his 25th birthday is in March 2020 -/
def kais_birth_year : ℕ := sorry

/-- The year of Kai's 25th birthday -/
def birthday_year : ℕ := 2020

/-- Kai's age at his birthday in 2020 -/
def kais_age : ℕ := 25

theorem kai_born_in_1995 : kais_birth_year = 1995 := by
  sorry

end kai_born_in_1995_l1144_114406


namespace sum_of_digits_of_product_of_repeated_digits_l1144_114413

/-- The integer consisting of n repetitions of digit d in base 10 -/
def repeatedDigit (d : ℕ) (n : ℕ) : ℕ :=
  d * (10^n - 1) / 9

/-- The sum of digits of a natural number in base 10 -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

theorem sum_of_digits_of_product_of_repeated_digits :
  let a := repeatedDigit 6 1000
  let b := repeatedDigit 7 1000
  sumOfDigits (9 * a * b) = 19986 := by
sorry

end sum_of_digits_of_product_of_repeated_digits_l1144_114413


namespace sum_of_solutions_quadratic_l1144_114407

theorem sum_of_solutions_quadratic (x : ℝ) : 
  (x^2 - 6*x + 5 = 2*x - 8) → 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - 6*x₁ + 5 = 2*x₁ - 8 ∧ x₂^2 - 6*x₂ + 5 = 2*x₂ - 8 ∧ x₁ + x₂ = 8) :=
by sorry

end sum_of_solutions_quadratic_l1144_114407


namespace average_age_decrease_l1144_114418

/-- Proves that the average age of a class decreases by 4 years when new students join --/
theorem average_age_decrease (original_strength original_average new_students new_average : ℕ) :
  original_strength = 12 →
  original_average = 40 →
  new_students = 12 →
  new_average = 32 →
  let total_age_before := original_strength * original_average
  let total_age_new := new_students * new_average
  let total_age_after := total_age_before + total_age_new
  let new_strength := original_strength + new_students
  let new_average_age := total_age_after / new_strength
  original_average - new_average_age = 4 := by
  sorry

end average_age_decrease_l1144_114418


namespace admission_price_is_two_l1144_114411

/-- Calculates the admission price for adults given the total number of people,
    admission price for children, total admission receipts, and number of adults. -/
def admission_price_for_adults (total_people : ℕ) (child_price : ℚ) 
                               (total_receipts : ℚ) (num_adults : ℕ) : ℚ :=
  (total_receipts - (total_people - num_adults : ℚ) * child_price) / num_adults

/-- Proves that the admission price for adults is $2 given the specific conditions. -/
theorem admission_price_is_two :
  admission_price_for_adults 610 1 960 350 = 2 := by sorry

end admission_price_is_two_l1144_114411


namespace complex_equation_solution_l1144_114458

theorem complex_equation_solution (z : ℂ) : 4 + 2 * Complex.I * z = 2 - 6 * Complex.I * z ↔ z = Complex.I / 4 := by
  sorry

end complex_equation_solution_l1144_114458


namespace largest_reciprocal_l1144_114435

theorem largest_reciprocal (a b c d e : ℝ) (ha : a = 1/4) (hb : b = 3/8) (hc : c = 1/2) (hd : d = 4) (he : e = 1000) :
  (1 / a > 1 / b) ∧ (1 / a > 1 / c) ∧ (1 / a > 1 / d) ∧ (1 / a > 1 / e) := by
  sorry

#check largest_reciprocal

end largest_reciprocal_l1144_114435


namespace max_min_product_l1144_114445

theorem max_min_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hsum : x + y + z = 15) (hprod : x*y + y*z + z*x = 45) :
  ∃ (m : ℝ), m = min (x*y) (min (y*z) (z*x)) ∧ m ≤ 17.5 ∧
  ∀ (m' : ℝ), m' = min (x*y) (min (y*z) (z*x)) → m' ≤ 17.5 := by
sorry

end max_min_product_l1144_114445


namespace equal_interest_l1144_114477

def interest_rate_1_1 : ℝ := 0.07
def interest_rate_1_2 : ℝ := 0.10
def interest_rate_2_1 : ℝ := 0.05
def interest_rate_2_2 : ℝ := 0.12

def principal_1 : ℝ := 600
def principal_2 : ℝ := 800

def time_1_1 : ℕ := 3
def time_2_1 : ℕ := 2
def time_2_2 : ℕ := 3

def total_interest_2 : ℝ := principal_2 * interest_rate_2_1 * time_2_1 + principal_2 * interest_rate_2_2 * time_2_2

theorem equal_interest (n : ℕ) : 
  principal_1 * interest_rate_1_1 * time_1_1 + principal_1 * interest_rate_1_2 * n = total_interest_2 → n = 5 := by
  sorry

end equal_interest_l1144_114477


namespace shaded_area_rectangle_circles_l1144_114427

/-- The area of the shaded region formed by the intersection of a rectangle and two circles -/
theorem shaded_area_rectangle_circles (rectangle_width : ℝ) (rectangle_height : ℝ) (circle_radius : ℝ) : 
  rectangle_width = 12 →
  rectangle_height = 10 →
  circle_radius = 3 →
  let rectangle_area := rectangle_width * rectangle_height
  let circle_area := π * circle_radius^2
  let shaded_area := rectangle_area - 2 * circle_area
  shaded_area = 120 - 18 * π := by
  sorry

end shaded_area_rectangle_circles_l1144_114427


namespace alcohol_mixture_proof_l1144_114459

/-- Proves that mixing 100 mL of 10% alcohol solution with 300 mL of 30% alcohol solution
    results in a 25% alcohol solution -/
theorem alcohol_mixture_proof :
  let x_volume : ℝ := 100
  let x_concentration : ℝ := 0.1
  let y_volume : ℝ := 300
  let y_concentration : ℝ := 0.3
  let target_concentration : ℝ := 0.25
  let total_volume := x_volume + y_volume
  let total_alcohol := x_volume * x_concentration + y_volume * y_concentration
  total_alcohol / total_volume = target_concentration := by
  sorry

#check alcohol_mixture_proof

end alcohol_mixture_proof_l1144_114459


namespace first_negative_term_is_14th_l1144_114405

/-- The index of the first negative term in the arithmetic sequence -/
def first_negative_term_index : ℕ := 14

/-- The first term of the arithmetic sequence -/
def a₁ : ℤ := 51

/-- The common difference of the arithmetic sequence -/
def d : ℤ := -4

/-- The general term of the arithmetic sequence -/
def aₙ (n : ℕ) : ℤ := a₁ + (n - 1 : ℤ) * d

theorem first_negative_term_is_14th :
  (∀ k < first_negative_term_index, aₙ k ≥ 0) ∧
  aₙ first_negative_term_index < 0 := by
  sorry

#eval aₙ first_negative_term_index

end first_negative_term_is_14th_l1144_114405


namespace range_of_m_l1144_114466

-- Define the propositions p and q
def p (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 10

def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

-- Define the condition that ¬p is necessary but not sufficient for ¬q
def not_p_necessary_not_sufficient_for_not_q (m : ℝ) : Prop :=
  (∀ x, ¬(q x m) → ¬(p x)) ∧ (∃ x, ¬(p x) ∧ q x m)

-- State the theorem
theorem range_of_m :
  ∀ m : ℝ, (m > 0 ∧ not_p_necessary_not_sufficient_for_not_q m) ↔ m ≥ 9 :=
sorry

end range_of_m_l1144_114466


namespace reflection_across_y_axis_l1144_114432

/-- Reflects a point across the y-axis -/
def reflect_y_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (-(p.1), p.2)

/-- The original point P -/
def P : ℝ × ℝ := (3, -5)

theorem reflection_across_y_axis :
  reflect_y_axis P = (-3, -5) := by sorry

end reflection_across_y_axis_l1144_114432
