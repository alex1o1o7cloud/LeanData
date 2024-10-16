import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_coefficient_relation_quadratic_coefficient_relation_alt_l1827_182716

/-- Given two quadratic equations, prove the relationship between their coefficients -/
theorem quadratic_coefficient_relation (a b c d r s : ℝ) : 
  (r + s = -a ∧ r * s = b) →  -- roots of first equation
  (r^2 + s^2 = -c ∧ r^2 * s^2 = d) →  -- roots of second equation
  r * s = 2 * b →  -- additional condition
  c = -a^2 + 2*b ∧ d = b^2 := by
sorry

/-- Alternative formulation using polynomial roots -/
theorem quadratic_coefficient_relation_alt (a b c d : ℝ) :
  (∃ r s : ℝ, (r + s = -a ∧ r * s = b) ∧ 
              (r^2 + s^2 = -c ∧ r^2 * s^2 = d) ∧
              r * s = 2 * b) →
  c = -a^2 + 2*b ∧ d = b^2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_relation_quadratic_coefficient_relation_alt_l1827_182716


namespace NUMINAMATH_CALUDE_share_calculation_l1827_182787

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

end NUMINAMATH_CALUDE_share_calculation_l1827_182787


namespace NUMINAMATH_CALUDE_larger_number_proof_l1827_182717

theorem larger_number_proof (L S : ℕ) (h1 : L > S) (h2 : L - S = 1335) (h3 : L = 6 * S + 15) :
  L = 1599 := by
sorry

end NUMINAMATH_CALUDE_larger_number_proof_l1827_182717


namespace NUMINAMATH_CALUDE_min_voters_for_tall_to_win_l1827_182730

/-- Represents the voting structure and rules of the giraffe beauty contest -/
structure GiraffeContest where
  total_voters : Nat
  num_districts : Nat
  precincts_per_district : Nat
  voters_per_precinct : Nat
  (total_voters_eq : total_voters = num_districts * precincts_per_district * voters_per_precinct)

/-- Calculates the minimum number of voters required to win the contest -/
def min_voters_to_win (contest : GiraffeContest) : Nat :=
  let min_districts_to_win := contest.num_districts / 2 + 1
  let min_precincts_to_win := contest.precincts_per_district / 2 + 1
  let min_votes_per_precinct := contest.voters_per_precinct / 2 + 1
  min_districts_to_win * min_precincts_to_win * min_votes_per_precinct

/-- The theorem stating the minimum number of voters required for Tall to win -/
theorem min_voters_for_tall_to_win (contest : GiraffeContest)
  (h1 : contest.total_voters = 135)
  (h2 : contest.num_districts = 5)
  (h3 : contest.precincts_per_district = 9)
  (h4 : contest.voters_per_precinct = 3) :
  min_voters_to_win contest = 30 := by
  sorry

#eval min_voters_to_win { total_voters := 135, num_districts := 5, precincts_per_district := 9, voters_per_precinct := 3, total_voters_eq := rfl }

end NUMINAMATH_CALUDE_min_voters_for_tall_to_win_l1827_182730


namespace NUMINAMATH_CALUDE_extreme_points_theorem_l1827_182713

open Real

/-- The function f(x) = x ln x - ax^2 -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * log x - a * x^2

/-- Predicate indicating that f has two extreme points -/
def has_two_extreme_points (a : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ 
    (∀ x : ℝ, x > 0 → (deriv (f a)) x = 0 ↔ (x = x₁ ∨ x = x₂))

/-- The main theorem -/
theorem extreme_points_theorem :
  (∀ a : ℝ, has_two_extreme_points a → 0 < a ∧ a < 1/2) ∧
  (∃ a : ℝ, has_two_extreme_points a ∧
    ∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ 
      (∀ x : ℝ, x > 0 → (deriv (f a)) x = 0 ↔ (x = x₁ ∨ x = x₂)) ∧
      x₁ + x₂ = x₂ / x₁) :=
sorry

end NUMINAMATH_CALUDE_extreme_points_theorem_l1827_182713


namespace NUMINAMATH_CALUDE_house_transaction_loss_l1827_182703

def initial_value : ℝ := 12000
def loss_percentage : ℝ := 0.15
def gain_percentage : ℝ := 0.20

def first_transaction (value : ℝ) (loss : ℝ) : ℝ :=
  value * (1 - loss)

def second_transaction (value : ℝ) (gain : ℝ) : ℝ :=
  value * (1 + gain)

theorem house_transaction_loss :
  second_transaction (first_transaction initial_value loss_percentage) gain_percentage - initial_value = 240 := by
  sorry

end NUMINAMATH_CALUDE_house_transaction_loss_l1827_182703


namespace NUMINAMATH_CALUDE_cube_less_than_triple_l1827_182729

theorem cube_less_than_triple (x : ℤ) : x^3 < 3*x ↔ x = 1 ∨ x = -2 := by sorry

end NUMINAMATH_CALUDE_cube_less_than_triple_l1827_182729


namespace NUMINAMATH_CALUDE_complex_sequence_sum_l1827_182752

theorem complex_sequence_sum (a b : ℕ → ℝ) :
  (∀ n : ℕ, (Complex.I + 2) ^ n = Complex.mk (a n) (b n)) →
  (∑' n, (a n * b n) / (7 : ℝ) ^ n) = 7 / 16 := by
sorry

end NUMINAMATH_CALUDE_complex_sequence_sum_l1827_182752


namespace NUMINAMATH_CALUDE_slightly_used_crayons_l1827_182783

theorem slightly_used_crayons (total : ℕ) (new_percent : ℚ) (broken_fraction : ℚ) : 
  total = 250 →
  new_percent = 40 / 100 →
  broken_fraction = 1 / 5 →
  (total : ℚ) * new_percent + (total : ℚ) * broken_fraction + (total : ℚ) * (1 - new_percent - broken_fraction) = (total : ℚ) →
  (total : ℚ) * (1 - new_percent - broken_fraction) = 100 := by
sorry

end NUMINAMATH_CALUDE_slightly_used_crayons_l1827_182783


namespace NUMINAMATH_CALUDE_complex_exp_13pi_over_2_l1827_182779

theorem complex_exp_13pi_over_2 : Complex.exp (13 * Real.pi * Complex.I / 2) = Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_exp_13pi_over_2_l1827_182779


namespace NUMINAMATH_CALUDE_parabola_shift_theorem_l1827_182718

/-- Represents a parabola of the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a parabola horizontally -/
def shift_horizontal (p : Parabola) (h : ℝ) : Parabola :=
  { a := p.a,
    b := p.b - 2 * p.a * h,
    c := p.c + p.a * h^2 - p.b * h }

/-- Shifts a parabola vertically -/
def shift_vertical (p : Parabola) (v : ℝ) : Parabola :=
  { a := p.a,
    b := p.b,
    c := p.c + v }

theorem parabola_shift_theorem (x : ℝ) :
  let initial_parabola := Parabola.mk (-2) 0 0
  let shifted_left := shift_horizontal initial_parabola 1
  let final_parabola := shift_vertical shifted_left 3
  final_parabola.a * x^2 + final_parabola.b * x + final_parabola.c = -2 * (x + 1)^2 + 3 := by
  sorry

end NUMINAMATH_CALUDE_parabola_shift_theorem_l1827_182718


namespace NUMINAMATH_CALUDE_opposite_of_negative_two_l1827_182745

theorem opposite_of_negative_two : 
  (∀ x : ℤ, x + (-x) = 0) → (-2 + 2 = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_two_l1827_182745


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l1827_182741

-- Define the polynomial
def p (x : ℝ) : ℝ := 10 * x^3 + 502 * x + 3010

-- Define the roots
theorem cubic_roots_sum (a b c : ℝ) (ha : p a = 0) (hb : p b = 0) (hc : p c = 0) :
  (a + b)^3 + (b + c)^3 + (c + a)^3 = 903 := by
  sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l1827_182741


namespace NUMINAMATH_CALUDE_least_possible_smallest_integer_l1827_182702

theorem least_possible_smallest_integer 
  (a b c d e f : ℤ) -- Six different integers
  (h_diff : a < b ∧ b < c ∧ c < d ∧ d < e ∧ e < f) -- Integers are different and in ascending order
  (h_median : (c + d) / 2 = 75) -- Median is 75
  (h_largest : f = 120) -- Largest is 120
  (h_smallest_neg : a < 0) -- Smallest is negative
  : a ≥ -1 := by
sorry

end NUMINAMATH_CALUDE_least_possible_smallest_integer_l1827_182702


namespace NUMINAMATH_CALUDE_march14_is_tuesday_l1827_182737

/-- 
Represents days of the week.
-/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- 
Represents a specific date in February or March.
-/
structure Date where
  month : Nat
  day : Nat

/-- 
Returns the number of days between two dates, assuming they are in the same year
and the year is not a leap year.
-/
def daysBetween (d1 d2 : Date) : Nat :=
  sorry

/-- 
Returns the day of the week that occurs 'n' days after a given day of the week.
-/
def dayAfter (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  sorry

/-- 
Theorem: If February 14th is on a Tuesday, then March 14th is also on a Tuesday.
-/
theorem march14_is_tuesday (h : dayAfter DayOfWeek.Tuesday 
  (daysBetween ⟨2, 14⟩ ⟨3, 14⟩) = DayOfWeek.Tuesday) :
  dayAfter DayOfWeek.Tuesday (daysBetween ⟨2, 14⟩ ⟨3, 14⟩) = DayOfWeek.Tuesday := by
  sorry

end NUMINAMATH_CALUDE_march14_is_tuesday_l1827_182737


namespace NUMINAMATH_CALUDE_max_correct_is_23_l1827_182786

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

end NUMINAMATH_CALUDE_max_correct_is_23_l1827_182786


namespace NUMINAMATH_CALUDE_complex_fraction_equals_two_l1827_182725

theorem complex_fraction_equals_two (a b : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : a^2 + a*b + b^2 = 0) : 
  (a^12 + b^12) / (a + b)^12 = 2 := by
sorry

end NUMINAMATH_CALUDE_complex_fraction_equals_two_l1827_182725


namespace NUMINAMATH_CALUDE_short_trees_after_planting_verify_total_short_trees_l1827_182722

/-- The number of short trees in the park after planting -/
def total_short_trees (current_short_trees new_short_trees : ℕ) : ℕ :=
  current_short_trees + new_short_trees

/-- Theorem: The total number of short trees after planting is the sum of current and new short trees -/
theorem short_trees_after_planting 
  (current_short_trees : ℕ) (new_short_trees : ℕ) :
  total_short_trees current_short_trees new_short_trees = current_short_trees + new_short_trees :=
by sorry

/-- The correct number of short trees after planting, given the problem conditions -/
def correct_total : ℕ := 98

/-- Theorem: The total number of short trees after planting, given the problem conditions, is 98 -/
theorem verify_total_short_trees :
  total_short_trees 41 57 = correct_total :=
by sorry

end NUMINAMATH_CALUDE_short_trees_after_planting_verify_total_short_trees_l1827_182722


namespace NUMINAMATH_CALUDE_paulas_shopping_problem_l1827_182774

/-- Paula's shopping problem -/
theorem paulas_shopping_problem (initial_amount : ℕ) (shirt_cost : ℕ) (pants_cost : ℕ) (remaining_amount : ℕ) :
  initial_amount = 109 →
  shirt_cost = 11 →
  pants_cost = 13 →
  remaining_amount = 74 →
  ∃ (num_shirts : ℕ), num_shirts * shirt_cost + pants_cost = initial_amount - remaining_amount ∧ num_shirts = 2 :=
by sorry

end NUMINAMATH_CALUDE_paulas_shopping_problem_l1827_182774


namespace NUMINAMATH_CALUDE_permutations_of_47722_l1827_182777

def digits : List ℕ := [4, 7, 7, 2, 2]

theorem permutations_of_47722 : Nat.factorial 5 / (Nat.factorial 2 * Nat.factorial 2) = 30 := by
  sorry

end NUMINAMATH_CALUDE_permutations_of_47722_l1827_182777


namespace NUMINAMATH_CALUDE_barbaras_candy_purchase_l1827_182781

/-- Theorem: Barbara's Candy Purchase
Given:
- initial_candies: The number of candies Barbara had initially
- final_candies: The number of candies Barbara has after buying more
- bought_candies: The number of candies Barbara bought

Prove that bought_candies = 18, given initial_candies = 9 and final_candies = 27
-/
theorem barbaras_candy_purchase 
  (initial_candies : ℕ) 
  (final_candies : ℕ) 
  (bought_candies : ℕ) 
  (h1 : initial_candies = 9)
  (h2 : final_candies = 27)
  (h3 : final_candies = initial_candies + bought_candies) :
  bought_candies = 18 := by
  sorry

end NUMINAMATH_CALUDE_barbaras_candy_purchase_l1827_182781


namespace NUMINAMATH_CALUDE_dove_population_growth_l1827_182714

theorem dove_population_growth (initial_doves : ℕ) (eggs_per_dove : ℕ) (hatch_rate : ℚ) : 
  initial_doves = 20 →
  eggs_per_dove = 3 →
  hatch_rate = 3/4 →
  initial_doves + (initial_doves * eggs_per_dove * hatch_rate).floor = 65 :=
by sorry

end NUMINAMATH_CALUDE_dove_population_growth_l1827_182714


namespace NUMINAMATH_CALUDE_sum_of_square_perimeters_l1827_182769

/-- The sum of the perimeters of an infinite sequence of squares, where each subsequent square
    is formed by connecting the midpoints of the sides of the previous square, given that the
    initial square has a side length of s. -/
theorem sum_of_square_perimeters (s : ℝ) (h : s > 0) :
  (∑' n, 4 * s / (2 ^ n)) = 8 * s := by
  sorry

end NUMINAMATH_CALUDE_sum_of_square_perimeters_l1827_182769


namespace NUMINAMATH_CALUDE_car_part_cost_l1827_182762

/-- Calculates the cost of a car part given the total repair cost, labor time, and hourly rate. -/
theorem car_part_cost (total_cost labor_time hourly_rate : ℝ) : 
  total_cost = 300 ∧ labor_time = 2 ∧ hourly_rate = 75 → 
  total_cost - (labor_time * hourly_rate) = 150 := by
sorry

end NUMINAMATH_CALUDE_car_part_cost_l1827_182762


namespace NUMINAMATH_CALUDE_four_dice_same_number_probability_l1827_182785

/-- The probability of a single die showing a specific number -/
def single_die_prob : ℚ := 1 / 6

/-- The number of dice being tossed -/
def num_dice : ℕ := 4

/-- The probability of all dice showing the same number -/
def all_same_prob : ℚ := single_die_prob ^ (num_dice - 1)

theorem four_dice_same_number_probability :
  all_same_prob = 1 / 216 := by
  sorry

end NUMINAMATH_CALUDE_four_dice_same_number_probability_l1827_182785


namespace NUMINAMATH_CALUDE_fraction_division_simplification_l1827_182750

theorem fraction_division_simplification :
  (3 / 4) / (5 / 8) = 6 / 5 := by sorry

end NUMINAMATH_CALUDE_fraction_division_simplification_l1827_182750


namespace NUMINAMATH_CALUDE_odd_function_domain_symmetry_l1827_182700

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_domain_symmetry
  (f : ℝ → ℝ) (t : ℝ)
  (h_odd : is_odd_function f)
  (h_domain : Set.Ioo t (2*t + 3) = {x | f x ≠ 0}) :
  t = -1 := by
sorry

end NUMINAMATH_CALUDE_odd_function_domain_symmetry_l1827_182700


namespace NUMINAMATH_CALUDE_coin_collection_value_l1827_182719

/-- Represents the value of a coin in cents -/
def coin_value (coin : String) : ℕ :=
  match coin with
  | "nickel" => 5
  | "dime" => 10
  | _ => 0

/-- Calculates the total value of coins given the number of nickels and dimes -/
def total_value (nickels dimes : ℕ) : ℕ :=
  nickels * coin_value "nickel" + dimes * coin_value "dime"

theorem coin_collection_value :
  ∀ (total_coins nickels : ℕ),
    total_coins = 8 →
    nickels = 2 →
    total_value nickels (total_coins - nickels) = 70 := by
  sorry

end NUMINAMATH_CALUDE_coin_collection_value_l1827_182719


namespace NUMINAMATH_CALUDE_catch_up_distance_l1827_182746

/-- Proves that B catches up with A 100 km from the start given the specified conditions -/
theorem catch_up_distance (speed_a speed_b : ℝ) (delay : ℝ) (catch_up_dist : ℝ) : 
  speed_a = 10 →
  speed_b = 20 →
  delay = 5 →
  catch_up_dist = speed_b * (catch_up_dist / (speed_b - speed_a)) →
  catch_up_dist = speed_a * (delay + catch_up_dist / (speed_b - speed_a)) →
  catch_up_dist = 100 := by
  sorry

#check catch_up_distance

end NUMINAMATH_CALUDE_catch_up_distance_l1827_182746


namespace NUMINAMATH_CALUDE_circumcircle_equation_l1827_182771

-- Define the points O, A, and B
def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (2, 4)
def B : ℝ × ℝ := (6, 2)

-- Define the equation of the circle
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 6*x - 2*y = 0

-- State the theorem
theorem circumcircle_equation :
  circle_equation O.1 O.2 ∧
  circle_equation A.1 A.2 ∧
  circle_equation B.1 B.2 ∧
  (∀ (x y : ℝ), circle_equation x y → (x - 3)^2 + (y - 1)^2 = 10) :=
sorry

end NUMINAMATH_CALUDE_circumcircle_equation_l1827_182771


namespace NUMINAMATH_CALUDE_total_work_hours_l1827_182733

theorem total_work_hours (hours_per_day : ℕ) (days_worked : ℕ) (total_hours : ℕ) : 
  hours_per_day = 3 → days_worked = 6 → total_hours = hours_per_day * days_worked → total_hours = 18 := by
  sorry

end NUMINAMATH_CALUDE_total_work_hours_l1827_182733


namespace NUMINAMATH_CALUDE_marias_new_quarters_l1827_182784

def dime_value : ℚ := 0.1
def quarter_value : ℚ := 0.25
def nickel_value : ℚ := 0.05

def initial_dimes : ℕ := 4
def initial_quarters : ℕ := 4
def initial_nickels : ℕ := 7

def total_after : ℚ := 3

theorem marias_new_quarters :
  ∃ (new_quarters : ℕ),
    (initial_dimes : ℚ) * dime_value +
    (initial_quarters : ℚ) * quarter_value +
    (initial_nickels : ℚ) * nickel_value +
    (new_quarters : ℚ) * quarter_value = total_after ∧
    new_quarters = 5 := by
  sorry

end NUMINAMATH_CALUDE_marias_new_quarters_l1827_182784


namespace NUMINAMATH_CALUDE_remaining_segments_length_is_23_l1827_182797

/-- Represents a polygon with perpendicular adjacent sides -/
structure Polygon where
  vertical_height : ℕ
  top_horizontal : ℕ
  first_descent : ℕ
  middle_horizontal : ℕ
  final_descent : ℕ

/-- Calculates the length of segments in the new figure after removing four sides -/
def remaining_segments_length (p : Polygon) : ℕ :=
  p.vertical_height + (p.top_horizontal + p.middle_horizontal) + 
  (p.first_descent + p.final_descent) + p.middle_horizontal

/-- The original polygon described in the problem -/
def original_polygon : Polygon :=
  { vertical_height := 7
  , top_horizontal := 3
  , first_descent := 2
  , middle_horizontal := 4
  , final_descent := 3 }

theorem remaining_segments_length_is_23 :
  remaining_segments_length original_polygon = 23 := by
  sorry

#eval remaining_segments_length original_polygon

end NUMINAMATH_CALUDE_remaining_segments_length_is_23_l1827_182797


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l1827_182723

theorem imaginary_part_of_complex_fraction (i : ℂ) :
  i * i = -1 →
  Complex.im ((1 + 2*i) / (i - 1)) = -3/2 := by
sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l1827_182723


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l1827_182705

theorem system_of_equations_solution :
  ∃! (x y : ℝ), (2 * x - y = -3) ∧ (4 * x - 5 * y = -21) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l1827_182705


namespace NUMINAMATH_CALUDE_combined_males_below_50_l1827_182796

/-- Represents an office branch with employee information -/
structure Branch where
  total_employees : ℕ
  male_percentage : ℚ
  male_over_50_percentage : ℚ

/-- Calculates the number of males below 50 in a branch -/
def males_below_50 (b : Branch) : ℚ :=
  b.total_employees * b.male_percentage * (1 - b.male_over_50_percentage)

/-- The given information about the three branches -/
def branch_A : Branch :=
  { total_employees := 4500
  , male_percentage := 60 / 100
  , male_over_50_percentage := 40 / 100 }

def branch_B : Branch :=
  { total_employees := 3500
  , male_percentage := 50 / 100
  , male_over_50_percentage := 55 / 100 }

def branch_C : Branch :=
  { total_employees := 2200
  , male_percentage := 35 / 100
  , male_over_50_percentage := 70 / 100 }

/-- The main theorem stating the combined number of males below 50 -/
theorem combined_males_below_50 :
  ⌊males_below_50 branch_A + males_below_50 branch_B + males_below_50 branch_C⌋ = 2638 := by
  sorry

end NUMINAMATH_CALUDE_combined_males_below_50_l1827_182796


namespace NUMINAMATH_CALUDE_side_bc_equation_proof_l1827_182756

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

end NUMINAMATH_CALUDE_side_bc_equation_proof_l1827_182756


namespace NUMINAMATH_CALUDE_remainder_problem_l1827_182799

theorem remainder_problem (x : ℤ) : 
  x % 62 = 7 → (x + 11) % 31 = 18 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l1827_182799


namespace NUMINAMATH_CALUDE_first_race_length_l1827_182766

/-- Represents the length of the first race in meters -/
def L : ℝ := sorry

/-- Theorem stating that the length of the first race is 100 meters -/
theorem first_race_length : L = 100 := by
  -- Define the relationships between runners based on the given conditions
  let A_finish := L
  let B_finish := L - 10
  let C_finish := L - 13
  
  -- Define the relationship in the second race
  let B_second_race := 180
  let C_second_race := 174  -- 180 - 6
  
  -- The ratio of B's performance to C's performance should be consistent across races
  have ratio_equality : (B_finish / C_finish) = (B_second_race / C_second_race) := by sorry
  
  -- Use the ratio equality to solve for L
  sorry

end NUMINAMATH_CALUDE_first_race_length_l1827_182766


namespace NUMINAMATH_CALUDE_set_C_elements_l1827_182789

def A : Set ℕ := {1, 3, 5, 7}
def B : Set ℕ := {2, 4, 6}
def C : Set (ℕ × ℕ) := {p | p.1 ∈ A ∧ p.2 ∈ B}

theorem set_C_elements : C = {(1,2), (1,4), (1,6), (3,2), (3,4), (3,6), (5,2), (5,4), (5,6), (7,2), (7,4), (7,6)} := by
  sorry

end NUMINAMATH_CALUDE_set_C_elements_l1827_182789


namespace NUMINAMATH_CALUDE_expected_value_is_one_l1827_182740

/-- Represents the outcome of a coin flip -/
inductive CoinFlip
| Heads
| Tails

/-- The probability of getting heads or tails -/
def flip_probability : CoinFlip → ℚ
| CoinFlip.Heads => 1/2
| CoinFlip.Tails => 1/2

/-- The payoff for each outcome -/
def payoff : CoinFlip → ℤ
| CoinFlip.Heads => 5
| CoinFlip.Tails => -3

/-- The expected value of a single coin flip -/
def expected_value : ℚ :=
  (flip_probability CoinFlip.Heads * payoff CoinFlip.Heads) +
  (flip_probability CoinFlip.Tails * payoff CoinFlip.Tails)

theorem expected_value_is_one :
  expected_value = 1 := by sorry

end NUMINAMATH_CALUDE_expected_value_is_one_l1827_182740


namespace NUMINAMATH_CALUDE_triangle_problem_l1827_182749

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  a > 0 ∧ b > 0 ∧ c > 0 →
  Real.sqrt 3 * c * Real.cos A + a * Real.sin C = Real.sqrt 3 * c →
  b + c = 5 →
  (1/2) * b * c * Real.sin A = Real.sqrt 3 →
  A = π/3 ∧ a = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l1827_182749


namespace NUMINAMATH_CALUDE_gcd_of_squares_sum_l1827_182712

theorem gcd_of_squares_sum : Nat.gcd (168^2 + 301^2 + 502^2) (169^2 + 300^2 + 501^2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_squares_sum_l1827_182712


namespace NUMINAMATH_CALUDE_solve_dimes_problem_l1827_182761

def dimes_problem (initial_dimes : ℕ) (given_to_mother : ℕ) (final_dimes : ℕ) : Prop :=
  ∃ (dimes_from_dad : ℕ),
    initial_dimes - given_to_mother + dimes_from_dad = final_dimes

theorem solve_dimes_problem :
  dimes_problem 7 4 11 → ∃ (dimes_from_dad : ℕ), dimes_from_dad = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_solve_dimes_problem_l1827_182761


namespace NUMINAMATH_CALUDE_integer_solutions_of_equation_l1827_182778

theorem integer_solutions_of_equation : 
  {(x, y) : ℤ × ℤ | x^2 + x = y^4 + y^3 + y^2 + y} = 
  {(0, -1), (-1, -1), (0, 0), (-1, 0), (5, 2), (-6, 2)} := by
sorry

end NUMINAMATH_CALUDE_integer_solutions_of_equation_l1827_182778


namespace NUMINAMATH_CALUDE_miguel_book_pages_l1827_182707

/-- The number of pages Miguel read in his book over two weeks --/
def total_pages : ℕ :=
  let first_four_days := 4 * 48
  let next_five_days := 5 * 35
  let subsequent_four_days := 4 * 28
  let last_day := 19
  first_four_days + next_five_days + subsequent_four_days + last_day

/-- Theorem stating that the total number of pages in Miguel's book is 498 --/
theorem miguel_book_pages : total_pages = 498 := by
  sorry

end NUMINAMATH_CALUDE_miguel_book_pages_l1827_182707


namespace NUMINAMATH_CALUDE_friends_in_all_activities_l1827_182743

theorem friends_in_all_activities (movie : ℕ) (picnic : ℕ) (games : ℕ) 
  (movie_and_picnic : ℕ) (movie_and_games : ℕ) (picnic_and_games : ℕ) 
  (total : ℕ) : 
  movie = 10 → 
  picnic = 20 → 
  games = 5 → 
  movie_and_picnic = 4 → 
  movie_and_games = 2 → 
  picnic_and_games = 0 → 
  total = 31 → 
  ∃ (all_three : ℕ), 
    all_three = 2 ∧ 
    total = movie + picnic + games - movie_and_picnic - movie_and_games - picnic_and_games + all_three :=
by sorry

end NUMINAMATH_CALUDE_friends_in_all_activities_l1827_182743


namespace NUMINAMATH_CALUDE_simplify_expression_l1827_182724

theorem simplify_expression (x y : ℝ) (hx : x = 4) (hy : y = 2) :
  (16 * x^2 * y^3) / (8 * x * y^2) = 16 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1827_182724


namespace NUMINAMATH_CALUDE_parallelogram_roots_l1827_182765

/-- The polynomial equation with parameter b -/
def P (b : ℝ) (z : ℂ) : ℂ :=
  z^4 - 8*z^3 + 15*b*z^2 - 5*(3*b^2 + 4*b - 4)*z + 9

/-- Condition for roots to form a parallelogram -/
def forms_parallelogram (b : ℝ) : Prop :=
  ∃ (w₁ w₂ : ℂ), (P b w₁ = 0) ∧ (P b (-w₁) = 0) ∧ (P b w₂ = 0) ∧ (P b (-w₂) = 0)

/-- The main theorem stating the values of b for which the roots form a parallelogram -/
theorem parallelogram_roots :
  ∀ b : ℝ, forms_parallelogram b ↔ (b = 2/3 ∨ b = -2) :=
sorry

end NUMINAMATH_CALUDE_parallelogram_roots_l1827_182765


namespace NUMINAMATH_CALUDE_fraction_subtraction_decreases_l1827_182776

theorem fraction_subtraction_decreases (a b n : ℕ) 
  (h1 : a < b) (h2 : 0 < n) (h3 : n < a) : 
  (a - n : ℚ) / (b - n) < (a : ℚ) / b :=
by sorry

end NUMINAMATH_CALUDE_fraction_subtraction_decreases_l1827_182776


namespace NUMINAMATH_CALUDE_min_value_sqrt_sum_l1827_182747

theorem min_value_sqrt_sum (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 4) :
  ∃ (min_val : ℝ), 
    (∀ a b c : ℝ, a ≥ 0 → b ≥ 0 → c ≥ 0 → a + b + c = 4 →
      Real.sqrt (2 * a + 1) + Real.sqrt (3 * b + 1) + Real.sqrt (4 * c + 1) ≥ min_val) ∧
    Real.sqrt (2 * (17 / 27) + 1) + Real.sqrt (3 * (49 / 36) + 1) + Real.sqrt (4 * (217 / 108) + 1) = min_val ∧
    min_val = Real.sqrt (61 / 27) + Real.sqrt (183 / 36) + Real.sqrt (976 / 108) := by
  sorry

end NUMINAMATH_CALUDE_min_value_sqrt_sum_l1827_182747


namespace NUMINAMATH_CALUDE_polynomial_value_l1827_182790

theorem polynomial_value (x : ℝ) : 
  let a : ℝ := 2002 * x + 2003
  let b : ℝ := 2002 * x + 2004
  let c : ℝ := 2002 * x + 2005
  a^2 + b^2 + c^2 - a*b - b*c - c*a = 3 := by
sorry

end NUMINAMATH_CALUDE_polynomial_value_l1827_182790


namespace NUMINAMATH_CALUDE_solution_set_f_range_of_a_l1827_182721

-- Define the function f
def f (x : ℝ) : ℝ := |1 - 2*x| - |1 + x|

-- Theorem for part 1
theorem solution_set_f (x : ℝ) : f x ≥ 4 ↔ x ≤ -2 ∨ x ≥ 6 := by sorry

-- Theorem for part 2
theorem range_of_a (a : ℝ) : 
  (∀ x, a^2 + 2*a + |1 + x| > f x) ↔ a < -3 ∨ a > 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_range_of_a_l1827_182721


namespace NUMINAMATH_CALUDE_equation_solution_l1827_182767

theorem equation_solution : 
  ∃ x : ℝ, (45 * x) + (625 / 25) - (300 * 4) = 2950 + 1500 / (75 * 2) ∧ x = 4135 / 45 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1827_182767


namespace NUMINAMATH_CALUDE_linda_savings_l1827_182775

theorem linda_savings : ∃ S : ℚ,
  (5/8 : ℚ) * S + (1/4 : ℚ) * S + (1/8 : ℚ) * S = S ∧
  (1/4 : ℚ) * S = 400 ∧
  (1/8 : ℚ) * S = 600 := by
  sorry

end NUMINAMATH_CALUDE_linda_savings_l1827_182775


namespace NUMINAMATH_CALUDE_rectangular_box_surface_area_l1827_182701

/-- Theorem: Surface Area of a Rectangular Box
Given a rectangular box with dimensions a, b, and c, if the sum of the lengths of its twelve edges
is 180 and the distance from one corner to the farthest corner is 25, then its total surface area
is 1400. -/
theorem rectangular_box_surface_area
  (a b c : ℝ)
  (edge_sum : 4 * a + 4 * b + 4 * c = 180)
  (diagonal : Real.sqrt (a^2 + b^2 + c^2) = 25) :
  2 * (a * b + b * c + a * c) = 1400 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_box_surface_area_l1827_182701


namespace NUMINAMATH_CALUDE_polynomial_remainder_l1827_182753

/-- The polynomial p(x) = x^4 - x^3 - 4x + 7 -/
def p (x : ℝ) : ℝ := x^4 - x^3 - 4*x + 7

/-- The remainder when p(x) is divided by (x - 3) -/
def remainder : ℝ := p 3

theorem polynomial_remainder : remainder = 49 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l1827_182753


namespace NUMINAMATH_CALUDE_not_repeating_decimal_l1827_182763

/-- Definition of the number we're considering -/
def x : ℚ := 3.66666

/-- Definition of a repeating decimal -/
def is_repeating_decimal (q : ℚ) : Prop :=
  ∃ (a b : ℕ) (c : ℤ), q = (c : ℚ) + (a : ℚ) / (10^b - 1)

/-- Theorem stating that 3.66666 is not a repeating decimal -/
theorem not_repeating_decimal : ¬ is_repeating_decimal x := by
  sorry

end NUMINAMATH_CALUDE_not_repeating_decimal_l1827_182763


namespace NUMINAMATH_CALUDE_medicine_supply_duration_l1827_182706

-- Define the given conditions
def pills_per_supply : ℕ := 90
def pill_fraction : ℚ := 3/4
def days_between_doses : ℕ := 3
def days_per_month : ℕ := 30

-- Define the theorem
theorem medicine_supply_duration :
  (pills_per_supply * days_between_doses / pill_fraction) / days_per_month = 12 := by
  sorry

end NUMINAMATH_CALUDE_medicine_supply_duration_l1827_182706


namespace NUMINAMATH_CALUDE_decimal_difference_l1827_182720

-- Define the repeating decimal 0.72̄
def repeating_decimal : ℚ := 72 / 99

-- Define the terminating decimal 0.72
def terminating_decimal : ℚ := 72 / 100

-- Theorem statement
theorem decimal_difference : 
  repeating_decimal - terminating_decimal = 2 / 275 := by
  sorry

end NUMINAMATH_CALUDE_decimal_difference_l1827_182720


namespace NUMINAMATH_CALUDE_toms_remaining_balloons_l1827_182758

/-- Theorem: Tom's remaining violet balloons -/
theorem toms_remaining_balloons (initial_balloons : ℕ) (given_balloons : ℕ) 
  (h1 : initial_balloons = 30)
  (h2 : given_balloons = 16) :
  initial_balloons - given_balloons = 14 := by
  sorry

end NUMINAMATH_CALUDE_toms_remaining_balloons_l1827_182758


namespace NUMINAMATH_CALUDE_inequality_proof_l1827_182710

theorem inequality_proof (x y z : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (sum_one : x + y + z = 1) :
  x^2 + y^2 + z^2 + 2 * Real.sqrt (3 * x * y * z) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1827_182710


namespace NUMINAMATH_CALUDE_symmetry_correctness_l1827_182727

-- Define a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define symmetry operations
def symmetryXAxis (p : Point3D) : Point3D :=
  { x := p.x, y := -p.y, z := p.z }

def symmetryYOzPlane (p : Point3D) : Point3D :=
  { x := -p.x, y := p.y, z := p.z }

def symmetryYAxis (p : Point3D) : Point3D :=
  { x := -p.x, y := p.y, z := -p.z }

def symmetryOrigin (p : Point3D) : Point3D :=
  { x := -p.x, y := -p.y, z := -p.z }

-- Theorem statement
theorem symmetry_correctness (p : Point3D) :
  (symmetryXAxis p ≠ p) ∧
  (symmetryYOzPlane p ≠ p) ∧
  (symmetryYAxis p ≠ p) ∧
  (symmetryOrigin p = { x := -p.x, y := -p.y, z := -p.z }) :=
by sorry

end NUMINAMATH_CALUDE_symmetry_correctness_l1827_182727


namespace NUMINAMATH_CALUDE_molecular_weight_calculation_l1827_182788

/-- Given a compound where 3 moles weigh 528 grams, prove its molecular weight is 176 grams/mole. -/
theorem molecular_weight_calculation (total_weight : ℝ) (num_moles : ℝ) 
  (h1 : total_weight = 528)
  (h2 : num_moles = 3) :
  total_weight / num_moles = 176 := by
  sorry

end NUMINAMATH_CALUDE_molecular_weight_calculation_l1827_182788


namespace NUMINAMATH_CALUDE_cubic_integer_root_l1827_182792

/-- A cubic polynomial with integer coefficients -/
structure CubicPolynomial where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ
  a_nonzero : a ≠ 0

/-- Evaluation of a cubic polynomial at a point -/
def CubicPolynomial.eval (P : CubicPolynomial) (x : ℤ) : ℤ :=
  P.a * x^3 + P.b * x^2 + P.c * x + P.d

/-- The property that xP(x) = yP(y) for infinitely many integer pairs (x,y) with x ≠ y -/
def InfinitelyManySolutions (P : CubicPolynomial) : Prop :=
  ∀ n : ℕ, ∃ (x y : ℤ), x ≠ y ∧ n < |x| ∧ n < |y| ∧ x * P.eval x = y * P.eval y

theorem cubic_integer_root (P : CubicPolynomial) 
    (h : InfinitelyManySolutions P) : 
    ∃ k : ℤ, P.eval k = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_integer_root_l1827_182792


namespace NUMINAMATH_CALUDE_polynomial_sum_l1827_182728

theorem polynomial_sum (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (2 - x)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  a₀ - a₁ + a₂ - a₃ + a₄ = 81 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_sum_l1827_182728


namespace NUMINAMATH_CALUDE_furniture_fraction_l1827_182751

def original_savings : ℚ := 960
def tv_cost : ℚ := 240

theorem furniture_fraction : 
  (original_savings - tv_cost) / original_savings = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_furniture_fraction_l1827_182751


namespace NUMINAMATH_CALUDE_approximate_profit_percent_l1827_182704

-- Define the selling price and cost price
def selling_price : Float := 2552.36
def cost_price : Float := 2400.0

-- Define the profit amount
def profit_amount : Float := selling_price - cost_price

-- Define the profit percent
def profit_percent : Float := (profit_amount / cost_price) * 100

-- Theorem to prove the approximate profit percent
theorem approximate_profit_percent :
  (Float.round (profit_percent * 100) / 100) = 6.35 := by
  sorry

end NUMINAMATH_CALUDE_approximate_profit_percent_l1827_182704


namespace NUMINAMATH_CALUDE_angle_conversion_l1827_182760

theorem angle_conversion :
  ∃ (k : ℤ) (α : ℝ), -1485 = k * 360 + α ∧ 0 ≤ α ∧ α < 360 :=
by
  use -5
  use 315
  sorry

end NUMINAMATH_CALUDE_angle_conversion_l1827_182760


namespace NUMINAMATH_CALUDE_space_divided_by_five_spheres_l1827_182793

/-- Maximum number of regions a sphere can be divided by n circles -/
def a : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | n + 2 => a (n + 1) + 2 * (n + 1)

/-- Maximum number of regions space can be divided by n spheres -/
def b : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | n + 2 => b (n + 1) + a (n + 1)

theorem space_divided_by_five_spheres :
  b 5 = 22 := by sorry

end NUMINAMATH_CALUDE_space_divided_by_five_spheres_l1827_182793


namespace NUMINAMATH_CALUDE_ratio_to_percentage_difference_l1827_182795

theorem ratio_to_percentage_difference (A B : ℝ) (hA : A > 0) (hB : B > 0) (h_ratio : A / B = 1/6 / (1/5)) :
  (B - A) / A * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_ratio_to_percentage_difference_l1827_182795


namespace NUMINAMATH_CALUDE_spherical_coordinate_transformation_l1827_182782

/-- Given a point with rectangular coordinates (-3, -4, 5) and spherical coordinates (ρ, θ, φ),
    prove that the point with spherical coordinates (ρ, -θ, φ) has rectangular coordinates (-3, 4, 5) -/
theorem spherical_coordinate_transformation (ρ θ φ : Real) 
  (h1 : -3 = ρ * Real.sin φ * Real.cos θ)
  (h2 : -4 = ρ * Real.sin φ * Real.sin θ)
  (h3 : 5 = ρ * Real.cos φ) :
  (-3 = ρ * Real.sin φ * Real.cos (-θ)) ∧ 
  (4 = ρ * Real.sin φ * Real.sin (-θ)) ∧ 
  (5 = ρ * Real.cos φ) :=
by sorry

end NUMINAMATH_CALUDE_spherical_coordinate_transformation_l1827_182782


namespace NUMINAMATH_CALUDE_target_hit_probability_l1827_182735

theorem target_hit_probability (p_a p_b : ℝ) : 
  p_a = 0.4 →
  p_a + p_b - p_a * p_b = 0.7 →
  p_b = 0.5 := by
sorry

end NUMINAMATH_CALUDE_target_hit_probability_l1827_182735


namespace NUMINAMATH_CALUDE_fraction_multiplication_l1827_182764

theorem fraction_multiplication : ((1 / 4 : ℚ) * (1 / 8 : ℚ)) * 4 = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_l1827_182764


namespace NUMINAMATH_CALUDE_max_movies_watched_l1827_182794

def movie_duration : ℕ := 90
def tuesday_watch_time : ℕ := 270
def wednesday_movie_multiplier : ℕ := 2

theorem max_movies_watched (movie_duration : ℕ) (tuesday_watch_time : ℕ) (wednesday_movie_multiplier : ℕ) :
  movie_duration = 90 →
  tuesday_watch_time = 270 →
  wednesday_movie_multiplier = 2 →
  (tuesday_watch_time / movie_duration + wednesday_movie_multiplier * (tuesday_watch_time / movie_duration)) = 9 :=
by sorry

end NUMINAMATH_CALUDE_max_movies_watched_l1827_182794


namespace NUMINAMATH_CALUDE_fourth_place_votes_l1827_182711

theorem fourth_place_votes (total_votes : ℕ) (winner_margin1 winner_margin2 winner_margin3 : ℕ) :
  total_votes = 979 →
  winner_margin1 = 53 →
  winner_margin2 = 79 →
  winner_margin3 = 105 →
  ∃ (winner_votes fourth_place_votes : ℕ),
    winner_votes - winner_margin1 + winner_votes - winner_margin2 + winner_votes - winner_margin3 + fourth_place_votes = total_votes ∧
    fourth_place_votes = 199 :=
by sorry

end NUMINAMATH_CALUDE_fourth_place_votes_l1827_182711


namespace NUMINAMATH_CALUDE_min_simultaneous_return_time_l1827_182726

def first_seven_primes : List Nat := [2, 3, 5, 7, 11, 13, 17]

def is_simultaneous_return (t : Nat) (horse_times : List Nat) : Bool :=
  (horse_times.filter (fun time => t % time = 0)).length ≥ 4

theorem min_simultaneous_return_time :
  let horse_times := first_seven_primes
  (∃ (t : Nat), t > 0 ∧ is_simultaneous_return t horse_times) ∧
  (∀ (t : Nat), 0 < t ∧ t < 210 → ¬is_simultaneous_return t horse_times) ∧
  is_simultaneous_return 210 horse_times :=
by sorry

end NUMINAMATH_CALUDE_min_simultaneous_return_time_l1827_182726


namespace NUMINAMATH_CALUDE_rectangle_area_preservation_l1827_182731

theorem rectangle_area_preservation (original_length original_width : ℝ) 
  (h_length : original_length = 280)
  (h_width : original_width = 80)
  (length_increase_percent : ℝ) 
  (h_increase : length_increase_percent = 60) : 
  let new_length := original_length * (1 + length_increase_percent / 100)
  let new_width := (original_length * original_width) / new_length
  let width_decrease_percent := (original_width - new_width) / original_width * 100
  width_decrease_percent = 37.5 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_preservation_l1827_182731


namespace NUMINAMATH_CALUDE_max_profit_theorem_l1827_182748

/-- Represents the profit function for a product given its price increase -/
def profit_function (x : ℕ) : ℝ := -10 * x^2 + 170 * x + 2100

/-- Represents the constraint on the price increase -/
def price_increase_constraint (x : ℕ) : Prop := 0 < x ∧ x ≤ 15

theorem max_profit_theorem :
  ∃ (x : ℕ), price_increase_constraint x ∧
    (∀ (y : ℕ), price_increase_constraint y → profit_function x ≥ profit_function y) ∧
    profit_function x = 2400 ∧
    (x = 5 ∨ x = 6) := by sorry

end NUMINAMATH_CALUDE_max_profit_theorem_l1827_182748


namespace NUMINAMATH_CALUDE_not_prime_base_n_2022_l1827_182772

-- Define the base-n representation of 2022
def base_n_2022 (n : ℕ) : ℕ := 2 * n^3 + 2 * n + 2

-- Theorem statement
theorem not_prime_base_n_2022 (n : ℕ) (h : n ≥ 2) : ¬ Nat.Prime (base_n_2022 n) := by
  sorry

end NUMINAMATH_CALUDE_not_prime_base_n_2022_l1827_182772


namespace NUMINAMATH_CALUDE_base_8_to_base_10_l1827_182732

theorem base_8_to_base_10 : 
  (3 * 8^3 + 5 * 8^2 + 2 * 8^1 + 6 * 8^0 : ℕ) = 1878 := by
  sorry

end NUMINAMATH_CALUDE_base_8_to_base_10_l1827_182732


namespace NUMINAMATH_CALUDE_sum_of_digits_of_seven_to_eleven_l1827_182734

theorem sum_of_digits_of_seven_to_eleven (n : ℕ) : 
  (3 + 4)^11 % 100 = 43 → 
  (((3 + 4)^11 / 10) % 10 + (3 + 4)^11 % 10) = 7 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_seven_to_eleven_l1827_182734


namespace NUMINAMATH_CALUDE_luke_laundry_problem_l1827_182738

/-- Given a total number of clothing pieces, the number of pieces in the first load,
    and the number of remaining loads, calculate the number of pieces in each small load. -/
def pieces_per_small_load (total : ℕ) (first_load : ℕ) (num_small_loads : ℕ) : ℕ :=
  (total - first_load) / num_small_loads

/-- Theorem stating that given the specific conditions of the problem,
    the number of pieces in each small load is 10. -/
theorem luke_laundry_problem :
  pieces_per_small_load 105 34 7 = 10 := by
  sorry

end NUMINAMATH_CALUDE_luke_laundry_problem_l1827_182738


namespace NUMINAMATH_CALUDE_basketball_free_throws_l1827_182715

theorem basketball_free_throws :
  ∀ (two_points three_points free_throws : ℕ),
    2 * (2 * two_points) = 3 * three_points →
    free_throws = 2 * two_points →
    2 * two_points + 3 * three_points + free_throws = 74 →
    free_throws = 30 := by
  sorry

end NUMINAMATH_CALUDE_basketball_free_throws_l1827_182715


namespace NUMINAMATH_CALUDE_investment_growth_l1827_182739

/-- The compound interest formula -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- The problem statement -/
theorem investment_growth :
  let principal : ℝ := 2500
  let rate : ℝ := 0.06
  let time : ℕ := 21
  let final_amount := compound_interest principal rate time
  ∃ ε > 0, |final_amount - 8280.91| < ε :=
by sorry

end NUMINAMATH_CALUDE_investment_growth_l1827_182739


namespace NUMINAMATH_CALUDE_line_bisecting_segment_l1827_182757

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

end NUMINAMATH_CALUDE_line_bisecting_segment_l1827_182757


namespace NUMINAMATH_CALUDE_equation_solution_l1827_182770

theorem equation_solution : ∃ x : ℚ, (x - 3) / 2 - (2 * x) / 3 = 1 ∧ x = -15 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1827_182770


namespace NUMINAMATH_CALUDE_exists_isosceles_right_triangle_same_color_l1827_182759

/-- A color type with three possible values -/
inductive Color
  | Red
  | Green
  | Blue

/-- A point in the 2D grid -/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- A coloring function that assigns a color to each point in the grid -/
def ColoringFunction := GridPoint → Color

/-- An isosceles right triangle in the grid -/
structure IsoscelesRightTriangle where
  a : GridPoint
  b : GridPoint
  c : GridPoint
  is_isosceles : (a.x - b.x)^2 + (a.y - b.y)^2 = (a.x - c.x)^2 + (a.y - c.y)^2
  is_right : (b.x - c.x) * (a.x - c.x) + (b.y - c.y) * (a.y - c.y) = 0

/-- The main theorem: There exists an isosceles right triangle with vertices of the same color -/
theorem exists_isosceles_right_triangle_same_color (coloring : ColoringFunction) :
  ∃ (t : IsoscelesRightTriangle), coloring t.a = coloring t.b ∧ coloring t.b = coloring t.c :=
sorry

end NUMINAMATH_CALUDE_exists_isosceles_right_triangle_same_color_l1827_182759


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1827_182755

theorem quadratic_equation_solution :
  ∃ (x₁ x₂ : ℝ), (∀ x : ℝ, x^2 = 2*x ↔ x = x₁ ∨ x = x₂) ∧ x₁ = 2 ∧ x₂ = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1827_182755


namespace NUMINAMATH_CALUDE_resulting_polygon_sides_l1827_182709

/-- Represents a regular polygon with a given number of sides -/
structure RegularPolygon where
  sides : ℕ

/-- Represents the arrangement of regular polygons -/
structure PolygonArrangement where
  polygons : List RegularPolygon

/-- Calculates the number of exposed sides in the resulting polygon -/
def exposedSides (arrangement : PolygonArrangement) : ℕ :=
  sorry

/-- The specific arrangement of polygons in our problem -/
def ourArrangement : PolygonArrangement :=
  { polygons := [
      { sides := 5 },  -- pentagon
      { sides := 4 },  -- square
      { sides := 6 },  -- hexagon
      { sides := 7 },  -- heptagon
      { sides := 9 }   -- nonagon
    ] }

/-- Theorem stating that the resulting polygon has 23 sides -/
theorem resulting_polygon_sides : exposedSides ourArrangement = 23 :=
  sorry

end NUMINAMATH_CALUDE_resulting_polygon_sides_l1827_182709


namespace NUMINAMATH_CALUDE_inequality_holds_l1827_182744

theorem inequality_holds (a b c : ℝ) (h : a > b) : (a - b) * c^2 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_l1827_182744


namespace NUMINAMATH_CALUDE_sqrt_one_third_same_type_as_2sqrt3_l1827_182768

-- Define a function to check if a number is of the same type as 2√3
def isSameTypeAs2Sqrt3 (x : ℝ) : Prop :=
  ∃ (a : ℝ), x = a * Real.sqrt 3

-- Theorem statement
theorem sqrt_one_third_same_type_as_2sqrt3 :
  isSameTypeAs2Sqrt3 (Real.sqrt (1/3)) ∧
  ¬isSameTypeAs2Sqrt3 (Real.sqrt 8) ∧
  ¬isSameTypeAs2Sqrt3 (Real.sqrt 18) ∧
  ¬isSameTypeAs2Sqrt3 (Real.sqrt 9) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_one_third_same_type_as_2sqrt3_l1827_182768


namespace NUMINAMATH_CALUDE_turning_point_sum_turning_point_is_correct_l1827_182736

/-- The point where the dog starts moving away from the cat -/
def turning_point : ℚ × ℚ :=
  (27/17, 135/68)

/-- The theorem stating the sum of coordinates of the turning point -/
theorem turning_point_sum :
  let (c, d) := turning_point
  c + d = 243/68 := by sorry

/-- The cat's position -/
def cat_position : ℚ × ℚ := (15, 12)

/-- The dog's path -/
def dog_path (x : ℚ) : ℚ := -4*x + 15

/-- The theorem proving the turning point is where the dog starts moving away from the cat -/
theorem turning_point_is_correct :
  let (c, d) := turning_point
  let (cat_x, cat_y) := cat_position
  -- The line perpendicular to the dog's path passing through the cat's position
  -- intersects the dog's path at the turning point
  (d - cat_y) / (c - cat_x) = 1 / 4 ∧
  d = dog_path c := by sorry

end NUMINAMATH_CALUDE_turning_point_sum_turning_point_is_correct_l1827_182736


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1827_182708

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence condition
  (a 3)^2 - 6*(a 3) + 10 = 0 →                      -- a₃ is a root of x² - 6x + 10 = 0
  (a 15)^2 - 6*(a 15) + 10 = 0 →                    -- a₁₅ is a root of x² - 6x + 10 = 0
  (∀ n, S n = (n/2) * (2*(a 1) + (n - 1)*(a 2 - a 1))) →  -- sum formula
  S 17 = 51 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1827_182708


namespace NUMINAMATH_CALUDE_logarithm_sum_approximation_l1827_182798

theorem logarithm_sum_approximation : 
  let expr := (1 / (Real.log 3 / Real.log 8 + 1)) + 
              (1 / (Real.log 2 / Real.log 12 + 1)) + 
              (1 / (Real.log 4 / Real.log 9 + 1))
  ∃ ε > 0, |expr - 3| < ε := by
  sorry

end NUMINAMATH_CALUDE_logarithm_sum_approximation_l1827_182798


namespace NUMINAMATH_CALUDE_problem_solution_l1827_182773

theorem problem_solution (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : x - y = 4 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1827_182773


namespace NUMINAMATH_CALUDE_formula_holds_for_given_pairs_l1827_182780

def formula (x : ℕ) : ℕ := x^2 + 4*x + 3

theorem formula_holds_for_given_pairs : 
  (formula 1 = 3) ∧ 
  (formula 2 = 8) ∧ 
  (formula 3 = 15) ∧ 
  (formula 4 = 24) ∧ 
  (formula 5 = 35) := by
  sorry

end NUMINAMATH_CALUDE_formula_holds_for_given_pairs_l1827_182780


namespace NUMINAMATH_CALUDE_ladybugs_with_spots_count_l1827_182742

/-- Given the total number of ladybugs and the number of ladybugs without spots,
    calculate the number of ladybugs with spots. -/
def ladybugsWithSpots (total : ℕ) (withoutSpots : ℕ) : ℕ :=
  total - withoutSpots

/-- Theorem stating that given 67,082 total ladybugs and 54,912 ladybugs without spots,
    there are 12,170 ladybugs with spots. -/
theorem ladybugs_with_spots_count :
  ladybugsWithSpots 67082 54912 = 12170 := by
  sorry

end NUMINAMATH_CALUDE_ladybugs_with_spots_count_l1827_182742


namespace NUMINAMATH_CALUDE_function_increasing_iff_a_in_range_range_of_a_for_increasing_function_l1827_182754

/-- The function f(x) = ax² - (a-1)x - 3 is increasing on [-1, +∞) if and only if a ∈ [0, 1/3] -/
theorem function_increasing_iff_a_in_range (a : ℝ) :
  (∀ x ≥ -1, ∀ y ≥ x, a * x^2 - (a - 1) * x - 3 ≤ a * y^2 - (a - 1) * y - 3) ↔
  0 ≤ a ∧ a ≤ 1/3 := by
  sorry

/-- The range of a for which f(x) = ax² - (a-1)x - 3 is increasing on [-1, +∞) is [0, 1/3] -/
theorem range_of_a_for_increasing_function :
  {a : ℝ | ∀ x ≥ -1, ∀ y ≥ x, a * x^2 - (a - 1) * x - 3 ≤ a * y^2 - (a - 1) * y - 3} =
  {a : ℝ | 0 ≤ a ∧ a ≤ 1/3} := by
  sorry

end NUMINAMATH_CALUDE_function_increasing_iff_a_in_range_range_of_a_for_increasing_function_l1827_182754


namespace NUMINAMATH_CALUDE_expected_black_pairs_in_deck_l1827_182791

/-- The expected number of pairs of adjacent black cards in a circular arrangement -/
def expected_black_pairs (total_cards : ℕ) (black_cards : ℕ) : ℚ :=
  (black_cards : ℚ) * ((black_cards - 1 : ℚ) / (total_cards - 1 : ℚ))

theorem expected_black_pairs_in_deck : 
  expected_black_pairs 52 30 = 870 / 51 := by
  sorry

end NUMINAMATH_CALUDE_expected_black_pairs_in_deck_l1827_182791
