import Mathlib

namespace NUMINAMATH_CALUDE_cute_isosceles_angle_l13_1381

-- Define a cute triangle
def is_cute_triangle (A B C : Real) : Prop :=
  B = 2 * C

-- Define an isosceles triangle
def is_isosceles_triangle (a b c : Real) : Prop :=
  a = b ∨ b = c ∨ a = c

-- Theorem statement
theorem cute_isosceles_angle (A B C : Real) :
  is_cute_triangle A B C →
  is_isosceles_triangle A B C →
  A = 45 ∨ A = 72 :=
by sorry

end NUMINAMATH_CALUDE_cute_isosceles_angle_l13_1381


namespace NUMINAMATH_CALUDE_roots_product_l13_1349

def Q (d e f : ℝ) (x : ℝ) : ℝ := x^3 + d*x^2 + e*x + f

theorem roots_product (d e f : ℝ) :
  (∀ x, Q d e f x = 0 ↔ x = Real.cos (2*π/9) ∨ x = Real.cos (4*π/9) ∨ x = Real.cos (8*π/9)) →
  d * e * f = 1 / 27 :=
sorry

end NUMINAMATH_CALUDE_roots_product_l13_1349


namespace NUMINAMATH_CALUDE_expression_value_l13_1365

theorem expression_value (x y : ℝ) (h : 2*x + y = 1) : 
  (y + 1)^2 - (y^2 - 4*x + 4) = -1 := by sorry

end NUMINAMATH_CALUDE_expression_value_l13_1365


namespace NUMINAMATH_CALUDE_alien_trees_conversion_l13_1340

/-- Converts a base-7 number to base-10 --/
def base7ToBase10 (hundreds tens units : Nat) : Nat :=
  hundreds * 7^2 + tens * 7^1 + units * 7^0

/-- The problem statement --/
theorem alien_trees_conversion :
  base7ToBase10 2 5 3 = 136 := by
  sorry

end NUMINAMATH_CALUDE_alien_trees_conversion_l13_1340


namespace NUMINAMATH_CALUDE_first_group_size_l13_1303

/-- Given a work that takes 25 days for some men to complete and 21 days for 50 men to complete,
    prove that the number of men in the first group is 42. -/
theorem first_group_size (days_first : ℕ) (days_second : ℕ) (men_second : ℕ) :
  days_first = 25 →
  days_second = 21 →
  men_second = 50 →
  (men_second * days_second : ℕ) = days_first * (42 : ℕ) :=
by sorry

end NUMINAMATH_CALUDE_first_group_size_l13_1303


namespace NUMINAMATH_CALUDE_negative_two_hash_negative_seven_l13_1328

/-- The # operation for rational numbers -/
def hash (a b : ℚ) : ℚ := a * b + 1

/-- Theorem stating that (-2) # (-7) = 15 -/
theorem negative_two_hash_negative_seven :
  hash (-2) (-7) = 15 := by
  sorry

end NUMINAMATH_CALUDE_negative_two_hash_negative_seven_l13_1328


namespace NUMINAMATH_CALUDE_proportion_solution_l13_1318

theorem proportion_solution (x : ℝ) : (0.60 / x = 6 / 4) → x = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_proportion_solution_l13_1318


namespace NUMINAMATH_CALUDE_ice_cream_bill_l13_1338

/-- The final bill for four ice cream sundaes with a 20% tip -/
theorem ice_cream_bill (alicia_cost brant_cost josh_cost yvette_cost : ℚ) 
  (h1 : alicia_cost = 7.5)
  (h2 : brant_cost = 10)
  (h3 : josh_cost = 8.5)
  (h4 : yvette_cost = 9)
  (tip_rate : ℚ)
  (h5 : tip_rate = 0.2) :
  alicia_cost + brant_cost + josh_cost + yvette_cost + 
  (alicia_cost + brant_cost + josh_cost + yvette_cost) * tip_rate = 42 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_bill_l13_1338


namespace NUMINAMATH_CALUDE_rotten_oranges_count_l13_1348

/-- The number of rotten oranges on a truck --/
def rotten_oranges : ℕ :=
  let total_oranges : ℕ := 10 * 30
  let oranges_for_juice : ℕ := 30
  let oranges_sold : ℕ := 220
  total_oranges - oranges_for_juice - oranges_sold

/-- Theorem stating that the number of rotten oranges is 50 --/
theorem rotten_oranges_count : rotten_oranges = 50 := by
  sorry

end NUMINAMATH_CALUDE_rotten_oranges_count_l13_1348


namespace NUMINAMATH_CALUDE_gcd_digits_bound_l13_1315

theorem gcd_digits_bound (a b : ℕ) (ha : 10^6 ≤ a ∧ a < 10^7) (hb : 10^6 ≤ b ∧ b < 10^7)
  (hlcm : 10^11 ≤ Nat.lcm a b ∧ Nat.lcm a b < 10^12) :
  Nat.gcd a b < 10^3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_digits_bound_l13_1315


namespace NUMINAMATH_CALUDE_complex_fraction_equals_i_l13_1364

theorem complex_fraction_equals_i : (1 + 5*Complex.I) / (5 - Complex.I) = Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equals_i_l13_1364


namespace NUMINAMATH_CALUDE_largest_T_for_inequality_l13_1382

theorem largest_T_for_inequality (a b c d e : ℝ) 
  (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d ∧ 0 ≤ e) 
  (h_sum : a + b = c + d + e) : 
  ∃ T : ℝ, T = (5 * Real.sqrt 30 - 2 * Real.sqrt 5) / 6 ∧
  (∀ S : ℝ, (Real.sqrt (a^2 + b^2 + c^2 + d^2 + e^2) ≥ 
    S * (Real.sqrt a + Real.sqrt b + Real.sqrt c + Real.sqrt d + Real.sqrt e)^2) → 
    S ≤ T) :=
by sorry

end NUMINAMATH_CALUDE_largest_T_for_inequality_l13_1382


namespace NUMINAMATH_CALUDE_product_equality_l13_1377

theorem product_equality : 1500 * 2987 * 0.2987 * 15 = 2989502.987 := by
  sorry

end NUMINAMATH_CALUDE_product_equality_l13_1377


namespace NUMINAMATH_CALUDE_no_six_correct_distribution_l13_1366

/-- Represents a distribution of letters to people -/
def LetterDistribution := Fin 7 → Fin 7

/-- A distribution where exactly 6 people get the correct letter -/
def SixCorrect (d : LetterDistribution) : Prop :=
  ∃ (i : Fin 7), (∀ j : Fin 7, j ≠ i → d j = j) ∧ d i ≠ i

/-- Theorem: It's impossible for exactly 6 out of 7 people to get the correct letter -/
theorem no_six_correct_distribution : ¬∃ (d : LetterDistribution), SixCorrect d := by
  sorry


end NUMINAMATH_CALUDE_no_six_correct_distribution_l13_1366


namespace NUMINAMATH_CALUDE_krista_hens_count_l13_1367

def egg_price_per_dozen : ℚ := 3
def total_sales : ℚ := 120
def weeks : ℕ := 4
def eggs_per_hen_per_week : ℕ := 12

def num_hens : ℕ := 10

theorem krista_hens_count :
  (egg_price_per_dozen * (total_sales / egg_price_per_dozen) = 
   ↑num_hens * ↑eggs_per_hen_per_week * ↑weeks) := by sorry

end NUMINAMATH_CALUDE_krista_hens_count_l13_1367


namespace NUMINAMATH_CALUDE_johns_remaining_money_l13_1378

/-- Calculates the remaining money after John's expenses -/
def remaining_money (initial : ℚ) (sweets : ℚ) (friend_gift : ℚ) (num_friends : ℕ) : ℚ :=
  initial - sweets - (friend_gift * num_friends)

/-- Theorem stating that John will be left with $2.45 -/
theorem johns_remaining_money :
  remaining_money 10.10 3.25 2.20 2 = 2.45 := by
  sorry

end NUMINAMATH_CALUDE_johns_remaining_money_l13_1378


namespace NUMINAMATH_CALUDE_equation_solution_l13_1309

theorem equation_solution : 
  ∃ y : ℚ, 3 * (4 * y - 5) + 1 = -3 * (2 - 5 * y) ∧ y = -8/3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l13_1309


namespace NUMINAMATH_CALUDE_h_not_prime_l13_1305

/-- The function h(n) as defined in the problem -/
def h (n : ℕ+) : ℤ := n.val^4 - 500 * n.val^2 + 625

/-- Theorem stating that h(n) is not prime for any positive integer n -/
theorem h_not_prime : ∀ n : ℕ+, ¬ Nat.Prime (Int.natAbs (h n)) := by sorry

end NUMINAMATH_CALUDE_h_not_prime_l13_1305


namespace NUMINAMATH_CALUDE_joan_balloons_l13_1357

theorem joan_balloons (total : ℕ) (melanie : ℕ) (joan : ℕ) : 
  total = 81 → melanie = 41 → total = joan + melanie → joan = 40 := by
  sorry

end NUMINAMATH_CALUDE_joan_balloons_l13_1357


namespace NUMINAMATH_CALUDE_percentage_problem_l13_1347

theorem percentage_problem (p : ℝ) (x : ℝ) : 
  (p / 100) * x = 100 → 
  (120 / 100) * x = 600 → 
  p = 20 := by
sorry

end NUMINAMATH_CALUDE_percentage_problem_l13_1347


namespace NUMINAMATH_CALUDE_partition_exists_five_equal_parts_exist_l13_1320

/-- Represents a geometric shape composed of squares and triangles -/
structure GeometricFigure where
  squares : ℕ
  triangles : ℕ

/-- Represents a partition of a geometric figure -/
structure Partition where
  parts : ℕ
  part_composition : GeometricFigure

/-- Predicate to check if a partition is valid for a given figure -/
def is_valid_partition (figure : GeometricFigure) (partition : Partition) : Prop :=
  figure.squares = partition.parts * partition.part_composition.squares ∧
  figure.triangles = partition.parts * partition.part_composition.triangles

/-- The specific figure from the problem -/
def problem_figure : GeometricFigure :=
  { squares := 10, triangles := 5 }

/-- The desired partition -/
def desired_partition : Partition :=
  { parts := 5, part_composition := { squares := 2, triangles := 1 } }

/-- Theorem stating that the desired partition is valid for the problem figure -/
theorem partition_exists : is_valid_partition problem_figure desired_partition := by
  sorry

/-- Main theorem proving the existence of the required partition -/
theorem five_equal_parts_exist : ∃ (p : Partition), 
  p.parts = 5 ∧ 
  p.part_composition.squares = 2 ∧ 
  p.part_composition.triangles = 1 ∧
  is_valid_partition problem_figure p := by
  sorry

end NUMINAMATH_CALUDE_partition_exists_five_equal_parts_exist_l13_1320


namespace NUMINAMATH_CALUDE_min_value_sum_of_reciprocals_l13_1373

theorem min_value_sum_of_reciprocals (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq_6 : a + b + c = 6) : 
  (9 / a) + (16 / b) + (25 / c) ≥ 24 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_of_reciprocals_l13_1373


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l13_1354

theorem complex_fraction_simplification :
  (2 + 4 * Complex.I) / ((1 + Complex.I)^2) = 2 - Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l13_1354


namespace NUMINAMATH_CALUDE_employee_count_l13_1368

theorem employee_count (avg_salary : ℝ) (salary_increase : ℝ) (manager_salary : ℝ)
  (h1 : avg_salary = 1500)
  (h2 : salary_increase = 600)
  (h3 : manager_salary = 14100) :
  ∃ n : ℕ, 
    (n : ℝ) * avg_salary + manager_salary = ((n : ℝ) + 1) * (avg_salary + salary_increase) ∧
    n = 20 :=
by sorry

end NUMINAMATH_CALUDE_employee_count_l13_1368


namespace NUMINAMATH_CALUDE_arrangement_count_is_24_l13_1355

/-- The number of ways to arrange 8 balls in a row, with 5 red balls and 3 white balls,
    such that exactly three red balls are consecutive. -/
def arrangement_count : ℕ := 24

/-- The total number of balls -/
def total_balls : ℕ := 8

/-- The number of red balls -/
def red_balls : ℕ := 5

/-- The number of white balls -/
def white_balls : ℕ := 3

/-- The number of consecutive red balls required -/
def consecutive_red : ℕ := 3

theorem arrangement_count_is_24 :
  arrangement_count = 24 ∧
  total_balls = 8 ∧
  red_balls = 5 ∧
  white_balls = 3 ∧
  consecutive_red = 3 :=
by sorry

end NUMINAMATH_CALUDE_arrangement_count_is_24_l13_1355


namespace NUMINAMATH_CALUDE_prism_volume_theorem_l13_1345

def prism_volume (AC_1 PQ phi : ℝ) (sin_phi cos_phi : ℝ) : Prop :=
  AC_1 = 3 ∧ 
  PQ = Real.sqrt 3 ∧ 
  phi = 30 * Real.pi / 180 ∧ 
  sin_phi = 1 / 2 ∧ 
  cos_phi = Real.sqrt 3 / 2 ∧ 
  ∃ (DL PK OK CL AL AC CC_1 : ℝ),
    DL = PK ∧
    DL = 1 / 2 * PQ * sin_phi ∧
    OK = 1 / 2 * PQ * cos_phi ∧
    CL / AL = (AC_1 / 2 - OK) / (AC_1 / 2 + OK) ∧
    AC = CL + AL ∧
    DL ^ 2 = CL * AL ∧
    CC_1 ^ 2 = AC_1 ^ 2 - AC ^ 2 ∧
    AC * DL * CC_1 = Real.sqrt 6 / 2

theorem prism_volume_theorem (AC_1 PQ phi sin_phi cos_phi : ℝ) :
  prism_volume AC_1 PQ phi sin_phi cos_phi → 
  ∃ (V : ℝ), V = Real.sqrt 6 / 2 := by
  sorry

end NUMINAMATH_CALUDE_prism_volume_theorem_l13_1345


namespace NUMINAMATH_CALUDE_soccer_teams_count_l13_1311

/-- The number of different teams formed when 12 students play soccer such that
    each group of 5 students plays together on the same team exactly once. -/
theorem soccer_teams_count : ℕ := by
  -- Define the number of students
  let n : ℕ := 12

  -- Define the team size
  let k : ℕ := 6

  -- Define the subset size that plays together exactly once
  let s : ℕ := 5

  -- The number of teams is equal to the number of ways to choose 5 students from 12,
  -- divided by the number of ways to choose 5 students from a team of 6
  have h : (n.choose s) / (k.choose s) = 132 := by sorry

  -- The result is the right-hand side of the equation
  exact 132


end NUMINAMATH_CALUDE_soccer_teams_count_l13_1311


namespace NUMINAMATH_CALUDE_grandfather_grandson_ages_l13_1371

def isComposite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

theorem grandfather_grandson_ages :
  ∀ (grandfather grandson : ℕ),
    isComposite grandfather →
    isComposite grandson →
    (grandfather + 1) * (grandson + 1) = 1610 →
    grandfather = 69 ∧ grandson = 22 := by
  sorry

end NUMINAMATH_CALUDE_grandfather_grandson_ages_l13_1371


namespace NUMINAMATH_CALUDE_power_of_two_with_nines_l13_1343

theorem power_of_two_with_nines (k : ℕ) (h : k > 1) :
  ∃ n : ℕ, ∃ m : ℕ,
    (2^n % 10^k = m) ∧ 
    (∃ count : ℕ, count ≥ k/2 ∧ 
      (∀ i : ℕ, i < k → 
        ((m / 10^i) % 10 = 9 → count > 0) ∧
        ((m / 10^i) % 10 ≠ 9 → count = count))) :=
sorry

end NUMINAMATH_CALUDE_power_of_two_with_nines_l13_1343


namespace NUMINAMATH_CALUDE_card_area_reduction_l13_1300

theorem card_area_reduction (initial_width initial_height : ℝ) 
  (h1 : initial_width = 10 ∧ initial_height = 8)
  (h2 : ∃ (reduced_side : ℝ), reduced_side = initial_width - 2 ∨ reduced_side = initial_height - 2)
  (h3 : ∃ (unreduced_side : ℝ), (reduced_side = initial_width - 2 → unreduced_side = initial_height) ∧
                                (reduced_side = initial_height - 2 → unreduced_side = initial_width))
  (h4 : reduced_side * unreduced_side = 64) :
  (initial_width - 2) * initial_height = 60 ∨ initial_width * (initial_height - 2) = 60 :=
sorry

end NUMINAMATH_CALUDE_card_area_reduction_l13_1300


namespace NUMINAMATH_CALUDE_junior_high_ten_total_games_l13_1330

/-- Represents a basketball conference -/
structure BasketballConference where
  num_teams : ℕ
  intra_conference_games : ℕ
  non_conference_games : ℕ

/-- Calculates the total number of games in a season for a given basketball conference -/
def total_games (conf : BasketballConference) : ℕ :=
  (conf.num_teams.choose 2 * conf.intra_conference_games) + (conf.num_teams * conf.non_conference_games)

/-- The Junior High Ten conference -/
def junior_high_ten : BasketballConference :=
  { num_teams := 10
  , intra_conference_games := 3
  , non_conference_games := 5 }

theorem junior_high_ten_total_games :
  total_games junior_high_ten = 185 := by
  sorry


end NUMINAMATH_CALUDE_junior_high_ten_total_games_l13_1330


namespace NUMINAMATH_CALUDE_five_cubed_sum_equals_five_to_fourth_l13_1352

theorem five_cubed_sum_equals_five_to_fourth : 5^3 + 5^3 + 5^3 + 5^3 + 5^3 = 5^4 := by
  sorry

end NUMINAMATH_CALUDE_five_cubed_sum_equals_five_to_fourth_l13_1352


namespace NUMINAMATH_CALUDE_limit_exponential_function_l13_1336

theorem limit_exponential_function :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 
    0 < |x - 1| ∧ |x - 1| < δ → 
    |((2 * Real.exp (x - 1) - 1) ^ ((3 * x - 1) / (x - 1))) - Real.exp 4| < ε :=
by sorry

end NUMINAMATH_CALUDE_limit_exponential_function_l13_1336


namespace NUMINAMATH_CALUDE_valid_last_score_l13_1350

def scores : List Nat := [65, 72, 75, 79, 82, 86, 90, 98]

def isIntegerAverage (sublist : List Nat) : Prop :=
  (sublist.sum * 100) % sublist.length = 0

def isValidLastScore (last : Nat) : Prop :=
  ∀ n : Nat, n ≥ 1 → n < 8 → 
    isIntegerAverage (scores.take n ++ [last])

theorem valid_last_score : 
  isValidLastScore 79 := by sorry

end NUMINAMATH_CALUDE_valid_last_score_l13_1350


namespace NUMINAMATH_CALUDE_rug_area_proof_l13_1335

/-- Given three rugs covering a floor area, prove their combined area -/
theorem rug_area_proof (total_covered_area single_layer_area double_layer_area triple_layer_area : ℝ) 
  (h1 : total_covered_area = 140)
  (h2 : double_layer_area = 24)
  (h3 : triple_layer_area = 20)
  (h4 : single_layer_area = total_covered_area - double_layer_area - triple_layer_area) :
  single_layer_area + 2 * double_layer_area + 3 * triple_layer_area = 204 := by
  sorry

end NUMINAMATH_CALUDE_rug_area_proof_l13_1335


namespace NUMINAMATH_CALUDE_white_or_black_ball_probability_l13_1313

/-- The number of balls in the bag -/
def total_balls : ℕ := 5

/-- The number of balls to be drawn -/
def drawn_balls : ℕ := 3

/-- The number of favorable outcomes (combinations including at least one white or black ball) -/
def favorable_outcomes : ℕ := 9

/-- The total number of possible outcomes when drawing 3 balls from 5 -/
def total_outcomes : ℕ := Nat.choose total_balls drawn_balls

/-- The probability of drawing at least one white or black ball -/
def probability : ℚ := favorable_outcomes / total_outcomes

theorem white_or_black_ball_probability :
  probability = 9 / 10 := by
  sorry

end NUMINAMATH_CALUDE_white_or_black_ball_probability_l13_1313


namespace NUMINAMATH_CALUDE_japanese_study_fraction_l13_1395

theorem japanese_study_fraction (J S : ℕ) (x : ℚ) : 
  S = 2 * J →
  (3 / 8 : ℚ) * S + x * J = (1 / 3 : ℚ) * (J + S) →
  x = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_japanese_study_fraction_l13_1395


namespace NUMINAMATH_CALUDE_infinite_solutions_sum_l13_1326

/-- If the equation ax - 4 = 14x + b has infinitely many solutions, then a + b = 10 -/
theorem infinite_solutions_sum (a b : ℝ) : 
  (∀ x, a * x - 4 = 14 * x + b) → a + b = 10 := by
  sorry

end NUMINAMATH_CALUDE_infinite_solutions_sum_l13_1326


namespace NUMINAMATH_CALUDE_bank_teller_coins_l13_1306

theorem bank_teller_coins (num_5c num_10c : ℕ) (total_value : ℚ) : 
  num_5c = 16 →
  num_10c = 16 →
  total_value = (5 * num_5c + 10 * num_10c) / 100 →
  total_value = 21/5 →
  num_5c + num_10c = 32 := by
sorry

end NUMINAMATH_CALUDE_bank_teller_coins_l13_1306


namespace NUMINAMATH_CALUDE_max_median_length_l13_1385

theorem max_median_length (a b c m : ℝ) (hA : Real.cos A = 15/17) (ha : a = 2) :
  m ≤ 4 ∧ ∃ (b c : ℝ), m = 4 := by
  sorry

end NUMINAMATH_CALUDE_max_median_length_l13_1385


namespace NUMINAMATH_CALUDE_tan_five_pi_quarters_l13_1396

theorem tan_five_pi_quarters : Real.tan (5 * π / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_five_pi_quarters_l13_1396


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l13_1392

theorem inverse_variation_problem (x y : ℝ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : ∃ k : ℝ, k > 0 ∧ ∀ x y, x^3 * y = k) 
  (h4 : 2^3 * 8 = 64) : 
  (x^3 * 64 = 64) → x = 1 := by
sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l13_1392


namespace NUMINAMATH_CALUDE_bottles_drank_l13_1389

def initial_bottles : ℕ := 17
def remaining_bottles : ℕ := 14

theorem bottles_drank : initial_bottles - remaining_bottles = 3 := by
  sorry

end NUMINAMATH_CALUDE_bottles_drank_l13_1389


namespace NUMINAMATH_CALUDE_resident_price_proof_l13_1394

/-- Calculates the ticket price for residents given the total attendees, number of residents,
    price for non-residents, and total revenue. -/
def resident_price (total_attendees : ℕ) (num_residents : ℕ) (non_resident_price : ℚ) (total_revenue : ℚ) : ℚ :=
  (total_revenue - (total_attendees - num_residents : ℚ) * non_resident_price) / num_residents

/-- Proves that the resident price is approximately $12.95 given the problem conditions. -/
theorem resident_price_proof :
  let total_attendees : ℕ := 586
  let num_residents : ℕ := 219
  let non_resident_price : ℚ := 17.95
  let total_revenue : ℚ := 9423.70
  abs (resident_price total_attendees num_residents non_resident_price total_revenue - 12.95) < 0.01 := by
  sorry

#eval resident_price 586 219 (17.95 : ℚ) (9423.70 : ℚ)

end NUMINAMATH_CALUDE_resident_price_proof_l13_1394


namespace NUMINAMATH_CALUDE_twenty_percent_of_three_and_three_quarters_l13_1384

theorem twenty_percent_of_three_and_three_quarters :
  (20 : ℚ) / 100 * (15 : ℚ) / 4 = (3 : ℚ) / 4 := by sorry

end NUMINAMATH_CALUDE_twenty_percent_of_three_and_three_quarters_l13_1384


namespace NUMINAMATH_CALUDE_problem_l13_1361

noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem problem (a b : ℝ) (h : f (a * b) = 1) : f (a^2) + f (b^2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_l13_1361


namespace NUMINAMATH_CALUDE_soda_cost_theorem_l13_1342

/-- The cost of a hamburger in dollars -/
def hamburger_cost : ℝ := 2

/-- The cost of a sandwich in dollars -/
def sandwich_cost : ℝ := 3

/-- The cost of Bob's fruit drink in dollars -/
def fruit_drink_cost : ℝ := 2

/-- The cost of Andy's soda in dollars -/
def soda_cost : ℝ := 4

/-- Andy's total spending in dollars -/
def andy_spending : ℝ := 2 * hamburger_cost + soda_cost

/-- Bob's total spending in dollars -/
def bob_spending : ℝ := 2 * sandwich_cost + fruit_drink_cost

theorem soda_cost_theorem : 
  andy_spending = bob_spending → soda_cost = 4 := by sorry

end NUMINAMATH_CALUDE_soda_cost_theorem_l13_1342


namespace NUMINAMATH_CALUDE_min_value_quadratic_l13_1301

theorem min_value_quadratic (x y : ℝ) : 
  x^2 + y^2 - 8*x + 6*y + 25 ≥ 0 ∧ 
  ∃ (a b : ℝ), a^2 + b^2 - 8*a + 6*b + 25 = 0 := by
sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l13_1301


namespace NUMINAMATH_CALUDE_P_plus_Q_equals_46_l13_1379

theorem P_plus_Q_equals_46 (P Q : ℝ) :
  (∀ x : ℝ, x ≠ 3 → P / (x - 3) + Q * (x - 2) = (-5 * x^2 + 20 * x + 36) / (x - 3)) →
  P + Q = 46 := by
sorry

end NUMINAMATH_CALUDE_P_plus_Q_equals_46_l13_1379


namespace NUMINAMATH_CALUDE_highest_throw_is_37_l13_1387

def highest_throw (christine_first : ℕ) (janice_first_diff : ℕ) 
                  (christine_second_diff : ℕ) (christine_third_diff : ℕ) 
                  (janice_third_diff : ℕ) : ℕ :=
  let christine_first := christine_first
  let janice_first := christine_first - janice_first_diff
  let christine_second := christine_first + christine_second_diff
  let janice_second := janice_first * 2
  let christine_third := christine_second + christine_third_diff
  let janice_third := christine_first + janice_third_diff
  max christine_first (max christine_second (max christine_third 
    (max janice_first (max janice_second janice_third))))

theorem highest_throw_is_37 : 
  highest_throw 20 4 10 4 17 = 37 := by
  sorry

end NUMINAMATH_CALUDE_highest_throw_is_37_l13_1387


namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l13_1383

theorem unique_solution_for_equation (x y z : ℕ) : 
  x > 1 → y > 1 → z > 1 → (x + 1)^y - x^z = 1 → x = 2 ∧ y = 2 ∧ z = 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_equation_l13_1383


namespace NUMINAMATH_CALUDE_complex_number_range_l13_1344

theorem complex_number_range (z₁ z₂ : ℂ) (a : ℝ) : 
  z₁ = ((-1 + 3*I) * (1 - I) - (1 + 3*I)) / I →
  z₂ = z₁ + a * I →
  Complex.abs z₂ ≤ 2 →
  a ∈ Set.Icc (1 - Real.sqrt 3) (1 + Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_range_l13_1344


namespace NUMINAMATH_CALUDE_farm_feet_count_l13_1398

/-- Represents a farm with hens and cows -/
structure Farm where
  total_heads : ℕ
  num_hens : ℕ
  hen_feet : ℕ
  cow_feet : ℕ

/-- Calculates the total number of feet on the farm -/
def total_feet (f : Farm) : ℕ :=
  f.num_hens * f.hen_feet + (f.total_heads - f.num_hens) * f.cow_feet

/-- Theorem: Given a farm with 48 total animals, 24 hens, 2 feet per hen, and 4 feet per cow, 
    the total number of feet is 144 -/
theorem farm_feet_count : 
  ∀ (f : Farm), f.total_heads = 48 → f.num_hens = 24 → f.hen_feet = 2 → f.cow_feet = 4 
  → total_feet f = 144 := by
  sorry


end NUMINAMATH_CALUDE_farm_feet_count_l13_1398


namespace NUMINAMATH_CALUDE_equation_solution_l13_1321

theorem equation_solution : 
  ∃ x : ℝ, (5 + 3.5 * x = 2 * x - 25 + x) ∧ (x = -60) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l13_1321


namespace NUMINAMATH_CALUDE_complex_equation_solution_l13_1370

theorem complex_equation_solution (z : ℂ) (h : (1 - Complex.I) * z = 2 * Complex.I) : 
  z = -1 + Complex.I := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l13_1370


namespace NUMINAMATH_CALUDE_sphere_volume_in_specific_cone_l13_1397

/-- A right circular cone with a sphere inscribed inside it. -/
structure ConeWithSphere where
  /-- The diameter of the cone's base in inches. -/
  base_diameter : ℝ
  /-- The vertex angle of the cross-section triangle perpendicular to the base. -/
  vertex_angle : ℝ

/-- Calculate the volume of the inscribed sphere in cubic inches. -/
def sphere_volume (cone : ConeWithSphere) : ℝ :=
  sorry

/-- Theorem stating the volume of the inscribed sphere in the specific cone. -/
theorem sphere_volume_in_specific_cone :
  let cone : ConeWithSphere := { base_diameter := 24, vertex_angle := 90 }
  sphere_volume cone = 288 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_in_specific_cone_l13_1397


namespace NUMINAMATH_CALUDE_robert_can_finish_both_books_l13_1391

/-- Represents the number of pages Robert can read per hour -/
def reading_speed : ℕ := 120

/-- Represents the number of pages in the first book -/
def book1_pages : ℕ := 360

/-- Represents the number of pages in the second book -/
def book2_pages : ℕ := 180

/-- Represents the number of hours Robert has available for reading -/
def available_time : ℕ := 7

/-- Theorem stating that Robert can finish both books within the available time -/
theorem robert_can_finish_both_books :
  (book1_pages / reading_speed + book2_pages / reading_speed : ℚ) ≤ available_time :=
sorry

end NUMINAMATH_CALUDE_robert_can_finish_both_books_l13_1391


namespace NUMINAMATH_CALUDE_round_23_36_to_nearest_tenth_l13_1372

/-- Represents a repeating decimal with an integer part and a repeating fractional part. -/
structure RepeatingDecimal where
  integerPart : ℤ
  repeatingPart : ℕ

/-- Rounds a RepeatingDecimal to the nearest tenth. -/
def roundToNearestTenth (x : RepeatingDecimal) : ℚ :=
  sorry

/-- The given repeating decimal 23.363636... -/
def givenNumber : RepeatingDecimal :=
  { integerPart := 23, repeatingPart := 36 }

theorem round_23_36_to_nearest_tenth :
  roundToNearestTenth givenNumber = 23.4 := by
  sorry

end NUMINAMATH_CALUDE_round_23_36_to_nearest_tenth_l13_1372


namespace NUMINAMATH_CALUDE_edward_money_left_l13_1359

def initial_money : ℕ := 41
def books_cost : ℕ := 6
def pens_cost : ℕ := 16

theorem edward_money_left : initial_money - (books_cost + pens_cost) = 19 := by
  sorry

end NUMINAMATH_CALUDE_edward_money_left_l13_1359


namespace NUMINAMATH_CALUDE_baseball_groups_l13_1316

/-- The number of groups formed from baseball players -/
def number_of_groups (new_players returning_players players_per_group : ℕ) : ℕ :=
  (new_players + returning_players) / players_per_group

/-- Theorem: The number of groups formed is 9 -/
theorem baseball_groups :
  number_of_groups 48 6 6 = 9 := by
  sorry

end NUMINAMATH_CALUDE_baseball_groups_l13_1316


namespace NUMINAMATH_CALUDE_georges_trivia_score_l13_1388

/-- George's trivia game score calculation -/
theorem georges_trivia_score :
  ∀ (first_half_correct second_half_correct points_per_question : ℕ),
    first_half_correct = 6 →
    second_half_correct = 4 →
    points_per_question = 3 →
    (first_half_correct + second_half_correct) * points_per_question = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_georges_trivia_score_l13_1388


namespace NUMINAMATH_CALUDE_f_has_root_iff_f_ln_b_gt_inv_b_l13_1358

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a / x

-- Theorem 1: f has a root iff 0 < a ≤ 1/e
theorem f_has_root_iff (a : ℝ) (h : a > 0) :
  (∃ x > 0, f a x = 0) ↔ a ≤ (Real.exp 1)⁻¹ :=
sorry

-- Theorem 2: When a ≥ 2/e and b > 1, f(ln b) > 1/b
theorem f_ln_b_gt_inv_b (a b : ℝ) (ha : a ≥ 2 / Real.exp 1) (hb : b > 1) :
  f a (Real.log b) > b⁻¹ :=
sorry

end NUMINAMATH_CALUDE_f_has_root_iff_f_ln_b_gt_inv_b_l13_1358


namespace NUMINAMATH_CALUDE_product_and_difference_equation_l13_1322

theorem product_and_difference_equation (n v : ℝ) : 
  n = -4.5 → 10 * n = v - 2 * n → v = -9 := by sorry

end NUMINAMATH_CALUDE_product_and_difference_equation_l13_1322


namespace NUMINAMATH_CALUDE_find_positive_integer_l13_1363

def first_seven_multiples_of_six : List ℕ := [6, 12, 18, 24, 30, 36, 42]

def a : ℚ := (first_seven_multiples_of_six.sum : ℚ) / 7

def b (n : ℕ) : ℚ := 2 * n

theorem find_positive_integer (n : ℕ) (h : n > 0) :
  a ^ 2 - (b n) ^ 2 = 0 → n = 12 := by
  sorry

end NUMINAMATH_CALUDE_find_positive_integer_l13_1363


namespace NUMINAMATH_CALUDE_min_value_of_sum_l13_1346

theorem min_value_of_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_sum : a + b + c = 5) :
  (9 / a + 16 / b + 25 / c) ≥ 30 ∧ 
  ∃ (a₀ b₀ c₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧ a₀ + b₀ + c₀ = 5 ∧ 9 / a₀ + 16 / b₀ + 25 / c₀ = 30 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_sum_l13_1346


namespace NUMINAMATH_CALUDE_area_of_triangle_ABC_l13_1302

/-- Calculate the area of a triangle given its vertices' coordinates -/
def triangleArea (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  0.5 * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

/-- The coordinates of points X, Y, and Z -/
def X : ℝ × ℝ := (6, 0)
def Y : ℝ × ℝ := (8, 4)
def Z : ℝ × ℝ := (10, 0)

/-- The ratio of areas between triangles XYZ and ABC -/
def areaRatio : ℝ := 0.1111111111111111

theorem area_of_triangle_ABC : 
  ∃ (A B C : ℝ × ℝ), 
    triangleArea X.1 X.2 Y.1 Y.2 Z.1 Z.2 / triangleArea A.1 A.2 B.1 B.2 C.1 C.2 = areaRatio ∧ 
    triangleArea A.1 A.2 B.1 B.2 C.1 C.2 = 72 := by
  sorry

end NUMINAMATH_CALUDE_area_of_triangle_ABC_l13_1302


namespace NUMINAMATH_CALUDE_probability_of_winning_l13_1360

def total_balls : ℕ := 10
def red_balls : ℕ := 5
def white_balls : ℕ := 5
def drawn_balls : ℕ := 5

def winning_outcomes : ℕ := Nat.choose red_balls 4 * Nat.choose white_balls 1 + Nat.choose red_balls 5

def total_outcomes : ℕ := Nat.choose total_balls drawn_balls

theorem probability_of_winning :
  (winning_outcomes : ℚ) / total_outcomes = 26 / 252 :=
sorry

end NUMINAMATH_CALUDE_probability_of_winning_l13_1360


namespace NUMINAMATH_CALUDE_decimal_20_equals_base4_110_l13_1329

/-- Converts a decimal number to its base-4 representation -/
def toBase4 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 4) ((m % 4) :: acc)
    aux n []

/-- Theorem: The decimal number 20 is equivalent to 110 in base 4 -/
theorem decimal_20_equals_base4_110 : toBase4 20 = [1, 1, 0] := by
  sorry

end NUMINAMATH_CALUDE_decimal_20_equals_base4_110_l13_1329


namespace NUMINAMATH_CALUDE_factory_production_equation_l13_1334

/-- Given the production data of an agricultural machinery factory,
    this theorem states the equation that the average monthly growth rate satisfies. -/
theorem factory_production_equation (x : ℝ) : 
  (500000 : ℝ) = 500000 ∧ 
  (1820000 : ℝ) = 1820000 → 
  50 + 50*(1+x) + 50*(1+x)^2 = 182 :=
by sorry

end NUMINAMATH_CALUDE_factory_production_equation_l13_1334


namespace NUMINAMATH_CALUDE_pauls_money_duration_l13_1304

/-- Given Paul's earnings and spending, prove how long the money will last. -/
theorem pauls_money_duration (lawn_money weed_money weekly_spending : ℕ) 
  (h1 : lawn_money = 44)
  (h2 : weed_money = 28)
  (h3 : weekly_spending = 9) :
  (lawn_money + weed_money) / weekly_spending = 8 := by
  sorry

end NUMINAMATH_CALUDE_pauls_money_duration_l13_1304


namespace NUMINAMATH_CALUDE_notebook_cost_per_page_l13_1332

/-- Calculates the cost per page in cents given the number of notebooks, pages per notebook, and total cost in dollars. -/
def cost_per_page (notebooks : ℕ) (pages_per_notebook : ℕ) (total_cost_dollars : ℕ) : ℚ :=
  (total_cost_dollars * 100) / (notebooks * pages_per_notebook)

/-- Proves that for 2 notebooks with 50 pages each, purchased for $5, the cost per page is 5 cents. -/
theorem notebook_cost_per_page :
  cost_per_page 2 50 5 = 5 := by
  sorry

end NUMINAMATH_CALUDE_notebook_cost_per_page_l13_1332


namespace NUMINAMATH_CALUDE_sequence_formulas_l13_1337

-- Sequence of all positive even numbers
def evenSequence (n : ℕ+) : ℕ := 2 * n

-- Sequence of all positive odd numbers
def oddSequence (n : ℕ+) : ℕ := 2 * n - 1

-- Sequence 1, 4, 9, 16, ...
def squareSequence (n : ℕ+) : ℕ := n ^ 2

-- Sequence -4, -1, 2, 5, ..., 23
def arithmeticSequence (n : ℕ+) : ℤ := 3 * n - 7

theorem sequence_formulas :
  (∀ n : ℕ+, evenSequence n = 2 * n) ∧
  (∀ n : ℕ+, oddSequence n = 2 * n - 1) ∧
  (∀ n : ℕ+, squareSequence n = n ^ 2) ∧
  (∀ n : ℕ+, arithmeticSequence n = 3 * n - 7) := by
  sorry

end NUMINAMATH_CALUDE_sequence_formulas_l13_1337


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_l13_1362

/-- Vector in R^2 -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Check if two vectors are parallel -/
def areParallel (v1 v2 : Vector2D) : Prop :=
  v1.x * v2.y = v1.y * v2.x

/-- Proposition p: vectors a and b are parallel -/
def p (m : ℝ) : Prop :=
  areParallel ⟨m, -2⟩ ⟨4, -2*m⟩

/-- Proposition q: m = 2 -/
def q (m : ℝ) : Prop :=
  m = 2

/-- p is necessary but not sufficient for q -/
theorem p_necessary_not_sufficient :
  (∀ m, q m → p m) ∧ (∃ m, p m ∧ ¬q m) :=
sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_l13_1362


namespace NUMINAMATH_CALUDE_pure_imaginary_m_value_l13_1324

/-- A complex number is pure imaginary if its real part is zero and its imaginary part is non-zero -/
def IsPureImaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

/-- The complex number z defined in terms of a real number m -/
def z (m : ℝ) : ℂ := Complex.mk (m^2 + 2*m - 3) (m - 1)

theorem pure_imaginary_m_value :
  ∀ m : ℝ, IsPureImaginary (z m) → m = -3 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_m_value_l13_1324


namespace NUMINAMATH_CALUDE_prank_week_combinations_l13_1351

/-- The number of available helpers for each day of the week -/
def helpers_per_day : List Nat := [1, 2, 3, 4, 1]

/-- The total number of possible combinations of helpers throughout the week -/
def total_combinations : Nat := List.prod helpers_per_day

/-- Theorem stating that the total number of combinations is 24 -/
theorem prank_week_combinations :
  total_combinations = 24 := by
  sorry

end NUMINAMATH_CALUDE_prank_week_combinations_l13_1351


namespace NUMINAMATH_CALUDE_square_side_length_l13_1319

theorem square_side_length (perimeter : ℝ) (h1 : perimeter = 16) : 
  perimeter / 4 = 4 := by
  sorry

#check square_side_length

end NUMINAMATH_CALUDE_square_side_length_l13_1319


namespace NUMINAMATH_CALUDE_hypercoplanar_iff_b_eq_plusminus_one_over_sqrt_two_l13_1317

/-- A point in 4D space -/
def Point4D := Fin 4 → ℝ

/-- The determinant of a 4x4 matrix -/
def det4 (m : Fin 4 → Fin 4 → ℝ) : ℝ := sorry

/-- Check if five points in 4D space are hypercoplanar -/
def are_hypercoplanar (p1 p2 p3 p4 p5 : Point4D) : Prop :=
  det4 (λ i j => match i, j with
    | 0, _ => p2 j - p1 j
    | 1, _ => p3 j - p1 j
    | 2, _ => p4 j - p1 j
    | 3, _ => p5 j - p1 j) = 0

/-- The given points in 4D space -/
def p1 : Point4D := λ _ => 0
def p2 (b : ℝ) : Point4D := λ i => match i with | 0 => 1 | 1 => b | _ => 0
def p3 (b : ℝ) : Point4D := λ i => match i with | 1 => 1 | 2 => b | _ => 0
def p4 (b : ℝ) : Point4D := λ i => match i with | 0 => b | 2 => 1 | _ => 0
def p5 (b : ℝ) : Point4D := λ i => match i with | 1 => b | 3 => 1 | _ => 0

theorem hypercoplanar_iff_b_eq_plusminus_one_over_sqrt_two :
  ∀ b : ℝ, are_hypercoplanar (p1) (p2 b) (p3 b) (p4 b) (p5 b) ↔ b = 1 / Real.sqrt 2 ∨ b = -1 / Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_hypercoplanar_iff_b_eq_plusminus_one_over_sqrt_two_l13_1317


namespace NUMINAMATH_CALUDE_midpoint_slope_l13_1376

/-- The slope of the line containing the midpoints of two specific line segments is 1.5 -/
theorem midpoint_slope : 
  let midpoint1 := ((0 + 8) / 2, (0 + 6) / 2)
  let midpoint2 := ((5 + 5) / 2, (0 + 9) / 2)
  let slope := (midpoint2.2 - midpoint1.2) / (midpoint2.1 - midpoint1.1)
  slope = 1.5 := by sorry

end NUMINAMATH_CALUDE_midpoint_slope_l13_1376


namespace NUMINAMATH_CALUDE_bus_calculation_l13_1390

theorem bus_calculation (total_students : ℕ) (capacity_40 capacity_30 : ℕ) : 
  total_students = 186 → capacity_40 = 40 → capacity_30 = 30 →
  (Nat.ceil (total_students / capacity_40) = 5 ∧
   Nat.ceil (total_students / capacity_30) = 7) := by
  sorry

#check bus_calculation

end NUMINAMATH_CALUDE_bus_calculation_l13_1390


namespace NUMINAMATH_CALUDE_cookie_cost_l13_1353

theorem cookie_cost (initial_amount : ℚ) (hat_cost : ℚ) (pencil_cost : ℚ) (num_cookies : ℕ) (remaining_amount : ℚ)
  (h1 : initial_amount = 20)
  (h2 : hat_cost = 10)
  (h3 : pencil_cost = 2)
  (h4 : num_cookies = 4)
  (h5 : remaining_amount = 3)
  : (initial_amount - hat_cost - pencil_cost - remaining_amount) / num_cookies = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_cookie_cost_l13_1353


namespace NUMINAMATH_CALUDE_arccos_one_half_l13_1314

theorem arccos_one_half : Real.arccos (1/2) = π/3 := by
  sorry

end NUMINAMATH_CALUDE_arccos_one_half_l13_1314


namespace NUMINAMATH_CALUDE_rotate_180_proof_l13_1399

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Rotates a line 180 degrees around the origin -/
def rotate180 (l : Line) : Line :=
  { a := l.a, b := l.b, c := -l.c }

theorem rotate_180_proof (l : Line) (h : l = { a := 1, b := -1, c := 4 }) :
  rotate180 l = { a := 1, b := -1, c := -4 } := by
  sorry

end NUMINAMATH_CALUDE_rotate_180_proof_l13_1399


namespace NUMINAMATH_CALUDE_rhombus_diagonal_l13_1380

/-- Given a rhombus with one diagonal of 80 meters and an area of 2480 square meters,
    prove that the length of the other diagonal is 62 meters. -/
theorem rhombus_diagonal (d₂ : ℝ) (area : ℝ) (h1 : d₂ = 80) (h2 : area = 2480) :
  ∃ d₁ : ℝ, d₁ = 62 ∧ area = (d₁ * d₂) / 2 :=
by sorry

end NUMINAMATH_CALUDE_rhombus_diagonal_l13_1380


namespace NUMINAMATH_CALUDE_x_squared_is_quadratic_l13_1375

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function representing x² = 0 -/
def f (x : ℝ) : ℝ := x^2

/-- Theorem: x² = 0 is a quadratic equation -/
theorem x_squared_is_quadratic : is_quadratic_equation f := by
  sorry


end NUMINAMATH_CALUDE_x_squared_is_quadratic_l13_1375


namespace NUMINAMATH_CALUDE_distance_to_nearest_city_l13_1308

-- Define the distance to the nearest city
variable (d : ℝ)

-- Define the conditions based on the false statements
def alice_condition : Prop := d < 8
def bob_condition : Prop := d > 7
def charlie_condition : Prop := d > 5
def david_condition : Prop := d ≠ 3

-- Theorem statement
theorem distance_to_nearest_city :
  alice_condition d ∧ bob_condition d ∧ charlie_condition d ∧ david_condition d ↔ d ∈ Set.Ioo 7 8 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_nearest_city_l13_1308


namespace NUMINAMATH_CALUDE_aras_height_is_55_l13_1374

/-- Calculates Ara's current height given the conditions of the problem -/
def aras_current_height (original_height : ℝ) (sheas_growth_rate : ℝ) (sheas_current_height : ℝ) (aras_growth_fraction : ℝ) : ℝ :=
  let sheas_growth := sheas_current_height - original_height
  let aras_growth := aras_growth_fraction * sheas_growth
  original_height + aras_growth

/-- The theorem stating that Ara's current height is 55 inches -/
theorem aras_height_is_55 :
  let original_height := 50
  let sheas_growth_rate := 0.3
  let sheas_current_height := 65
  let aras_growth_fraction := 1/3
  aras_current_height original_height sheas_growth_rate sheas_current_height aras_growth_fraction = 55 := by
  sorry


end NUMINAMATH_CALUDE_aras_height_is_55_l13_1374


namespace NUMINAMATH_CALUDE_right_triangle_line_equation_l13_1323

/-- Given a right triangle in the first quadrant with vertices at (0, 0), (a, 0), and (0, b),
    where the area of the triangle is T, prove that the equation of the line passing through
    (0, b) and (a, 0) in its standard form is 2Tx - a²y + 2Ta = 0. -/
theorem right_triangle_line_equation (a b T : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : T = (1/2) * a * b) :
  ∃ (A B C : ℝ), A * a + B * b + C = 0 ∧ 
                 (∀ x y : ℝ, A * x + B * y + C = 0 ↔ 2 * T * x - a^2 * y + 2 * T * a = 0) :=
sorry

end NUMINAMATH_CALUDE_right_triangle_line_equation_l13_1323


namespace NUMINAMATH_CALUDE_brads_running_speed_l13_1386

/-- Proof of Brad's running speed given the conditions of the problem -/
theorem brads_running_speed 
  (total_distance : ℝ) 
  (maxwells_speed : ℝ) 
  (time_until_meeting : ℝ) 
  (brad_delay : ℝ) 
  (h1 : total_distance = 24) 
  (h2 : maxwells_speed = 4) 
  (h3 : time_until_meeting = 3) 
  (h4 : brad_delay = 1) : 
  (total_distance - maxwells_speed * time_until_meeting) / (time_until_meeting - brad_delay) = 6 := by
  sorry

#check brads_running_speed

end NUMINAMATH_CALUDE_brads_running_speed_l13_1386


namespace NUMINAMATH_CALUDE_integral_even_function_l13_1325

-- Define an even function
def EvenFunction (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- State the theorem
theorem integral_even_function 
  (f : ℝ → ℝ) 
  (h1 : EvenFunction f) 
  (h2 : ∫ x in (0:ℝ)..6, f x = 8) : 
  ∫ x in (-6:ℝ)..6, f x = 16 := by
  sorry

end NUMINAMATH_CALUDE_integral_even_function_l13_1325


namespace NUMINAMATH_CALUDE_division_result_l13_1356

theorem division_result : (5 / 2) / 7 = 5 / 14 := by sorry

end NUMINAMATH_CALUDE_division_result_l13_1356


namespace NUMINAMATH_CALUDE_equation_real_solution_l13_1327

theorem equation_real_solution (x : ℝ) :
  (∀ y : ℝ, ∃ z : ℝ, x^2 + y^2 + z^2 + 2*x*y*z = 1) ↔ (x = 1 ∨ x = -1) := by
  sorry

end NUMINAMATH_CALUDE_equation_real_solution_l13_1327


namespace NUMINAMATH_CALUDE_functions_with_inverses_l13_1393

-- Define the four functions
def function_A : ℝ → ℝ := sorry
def function_B : ℝ → ℝ := sorry
def function_C : ℝ → ℝ := sorry
def function_D : ℝ → ℝ := sorry

-- Define the property of being a straight line through the origin
def is_straight_line_through_origin (f : ℝ → ℝ) : Prop := sorry

-- Define the property of being a downward-opening parabola with vertex at (0, 1)
def is_downward_parabola_vertex_0_1 (f : ℝ → ℝ) : Prop := sorry

-- Define the property of being an upper semicircle with radius 3 centered at origin
def is_upper_semicircle_radius_3 (f : ℝ → ℝ) : Prop := sorry

-- Define the property of being a piecewise linear function as described
def is_piecewise_linear_as_described (f : ℝ → ℝ) : Prop := sorry

-- Define the property of having an inverse
def has_inverse (f : ℝ → ℝ) : Prop := sorry

theorem functions_with_inverses :
  is_straight_line_through_origin function_A ∧
  is_downward_parabola_vertex_0_1 function_B ∧
  is_upper_semicircle_radius_3 function_C ∧
  is_piecewise_linear_as_described function_D →
  has_inverse function_A ∧
  ¬ has_inverse function_B ∧
  ¬ has_inverse function_C ∧
  has_inverse function_D := by sorry

end NUMINAMATH_CALUDE_functions_with_inverses_l13_1393


namespace NUMINAMATH_CALUDE_fraction_inequality_solution_set_l13_1331

theorem fraction_inequality_solution_set (x : ℝ) (h : x ≠ 1) :
  x / (x - 1) < 0 ↔ 0 < x ∧ x < 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_solution_set_l13_1331


namespace NUMINAMATH_CALUDE_ratio_proof_l13_1341

theorem ratio_proof (x y : ℝ) (h : (3 * x^2 - 2 * y^2) / (x^2 + 4 * y^2) = 5 / 7) :
  x / y = Real.sqrt (17 / 8) :=
by sorry

end NUMINAMATH_CALUDE_ratio_proof_l13_1341


namespace NUMINAMATH_CALUDE_total_minutes_played_l13_1333

/-- The number of days in 2 weeks -/
def days_in_two_weeks : ℕ := 14

/-- The number of gigs Mark does in 2 weeks -/
def gigs_in_two_weeks : ℕ := days_in_two_weeks / 2

/-- The number of songs Mark plays in each gig -/
def songs_per_gig : ℕ := 3

/-- The duration of the first two songs in minutes -/
def short_song_duration : ℕ := 5

/-- The duration of the last song in minutes -/
def long_song_duration : ℕ := 2 * short_song_duration

/-- The total duration of all songs in one gig in minutes -/
def duration_per_gig : ℕ := 2 * short_song_duration + long_song_duration

/-- The theorem stating the total number of minutes Mark played in 2 weeks -/
theorem total_minutes_played : gigs_in_two_weeks * duration_per_gig = 140 := by
  sorry

end NUMINAMATH_CALUDE_total_minutes_played_l13_1333


namespace NUMINAMATH_CALUDE_units_digit_problem_l13_1310

theorem units_digit_problem : ∃ n : ℕ, n % 10 = 7 ∧ n = (2 + 1) * (2^2 + 1) * (2^4 + 1) * (2^8 + 1) * (2^16 + 1) * (2^32 + 1) + 2 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_problem_l13_1310


namespace NUMINAMATH_CALUDE_root_product_theorem_l13_1307

-- Define the polynomial f(x)
def f (x : ℝ) : ℝ := x^4 - x^3 + 2*x - 3

-- Define the function g(x)
def g (x : ℝ) : ℝ := x^2 - 3

-- State the theorem
theorem root_product_theorem (x₁ x₂ x₃ x₄ : ℝ) : 
  f x₁ = 0 → f x₂ = 0 → f x₃ = 0 → f x₄ = 0 → 
  g x₁ * g x₂ * g x₃ * g x₄ = 33 := by
sorry

end NUMINAMATH_CALUDE_root_product_theorem_l13_1307


namespace NUMINAMATH_CALUDE_arctan_sum_equals_pi_over_four_l13_1369

theorem arctan_sum_equals_pi_over_four :
  ∃ (n : ℕ), n > 0 ∧
  Real.arctan (1 / 6) + Real.arctan (1 / 7) + Real.arctan (1 / 5) + Real.arctan (1 / n) = π / 4 ∧
  n = 311 := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_equals_pi_over_four_l13_1369


namespace NUMINAMATH_CALUDE_henry_twice_jills_age_l13_1312

/-- 
Given that:
- The sum of Henry and Jill's present ages is 33
- Henry's present age is 20
- Jill's present age is 13

This theorem proves that 6 years ago, Henry was twice the age of Jill.
-/
theorem henry_twice_jills_age : 
  ∀ (henry_age jill_age : ℕ),
  henry_age + jill_age = 33 →
  henry_age = 20 →
  jill_age = 13 →
  ∃ (years_ago : ℕ), 
    years_ago = 6 ∧ 
    henry_age - years_ago = 2 * (jill_age - years_ago) :=
by
  sorry

end NUMINAMATH_CALUDE_henry_twice_jills_age_l13_1312


namespace NUMINAMATH_CALUDE_average_rate_of_change_x_squared_l13_1339

-- Define the function
def f (x : ℝ) : ℝ := x^2

-- Define the interval
def a : ℝ := 1
def b : ℝ := 2

-- State the theorem
theorem average_rate_of_change_x_squared :
  (f b - f a) / (b - a) = 3 := by
  sorry

end NUMINAMATH_CALUDE_average_rate_of_change_x_squared_l13_1339
