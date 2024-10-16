import Mathlib

namespace NUMINAMATH_CALUDE_remaining_chess_pieces_l1625_162542

def standard_chess_pieces : ℕ := 32
def initial_player_pieces : ℕ := 16
def arianna_lost_pieces : ℕ := 3
def samantha_lost_pieces : ℕ := 9

theorem remaining_chess_pieces :
  standard_chess_pieces - (arianna_lost_pieces + samantha_lost_pieces) = 20 :=
by sorry

end NUMINAMATH_CALUDE_remaining_chess_pieces_l1625_162542


namespace NUMINAMATH_CALUDE_largest_x_sqrt_3x_eq_6x_l1625_162537

theorem largest_x_sqrt_3x_eq_6x :
  ∃ (x_max : ℚ), x_max = 1/12 ∧
  (∀ x : ℚ, x ≥ 0 → Real.sqrt (3 * x) = 6 * x → x ≤ x_max) ∧
  Real.sqrt (3 * x_max) = 6 * x_max :=
sorry

end NUMINAMATH_CALUDE_largest_x_sqrt_3x_eq_6x_l1625_162537


namespace NUMINAMATH_CALUDE_equation_four_solutions_l1625_162506

theorem equation_four_solutions 
  (a b c d e : ℝ) 
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e) :
  ∃! (s : Finset ℝ), s.card = 4 ∧ ∀ x ∈ s, 
    (x - a) * (x - b) * (x - c) * (x - d) +
    (x - a) * (x - b) * (x - c) * (x - e) +
    (x - a) * (x - b) * (x - d) * (x - e) +
    (x - a) * (x - c) * (x - d) * (x - e) +
    (x - b) * (x - c) * (x - d) * (x - e) = 0 :=
by sorry

end NUMINAMATH_CALUDE_equation_four_solutions_l1625_162506


namespace NUMINAMATH_CALUDE_circles_intersect_l1625_162529

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 16*y - 48 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 8*y - 44 = 0

-- Theorem stating that the circles intersect
theorem circles_intersect : ∃ (x y : ℝ), circle1 x y ∧ circle2 x y :=
sorry

end NUMINAMATH_CALUDE_circles_intersect_l1625_162529


namespace NUMINAMATH_CALUDE_f_equiv_g_l1625_162595

/-- Function f defined as f(x) = x^2 - 2x - 1 -/
def f (x : ℝ) : ℝ := x^2 - 2*x - 1

/-- Function g defined as g(t) = t^2 - 2t + 1 -/
def g (t : ℝ) : ℝ := t^2 - 2*t + 1

/-- Theorem stating that f and g are equivalent functions -/
theorem f_equiv_g : ∀ x : ℝ, f x = g x := by sorry

end NUMINAMATH_CALUDE_f_equiv_g_l1625_162595


namespace NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l1625_162570

/-- An arithmetic sequence is increasing if its common difference is positive -/
def IsIncreasingArithmeticSequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  (∀ n, a (n + 1) = a n + d) ∧ d > 0

/-- The sum of the first three terms of an arithmetic sequence -/
def SumFirstThree (a : ℕ → ℝ) : ℝ :=
  a 1 + a 2 + a 3

/-- The product of the first three terms of an arithmetic sequence -/
def ProductFirstThree (a : ℕ → ℝ) : ℝ :=
  a 1 * a 2 * a 3

theorem arithmetic_sequence_first_term
  (a : ℕ → ℝ) (d : ℝ)
  (h_increasing : IsIncreasingArithmeticSequence a d)
  (h_sum : SumFirstThree a = 12)
  (h_product : ProductFirstThree a = 48) :
  a 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l1625_162570


namespace NUMINAMATH_CALUDE_min_value_trig_expression_l1625_162556

theorem min_value_trig_expression (α β : ℝ) :
  (3 * Real.cos α + 4 * Real.sin β - 7)^2 + (3 * Real.sin α + 4 * Real.cos β - 12)^2 ≥ 36 ∧
  ∃ α₀ β₀ : ℝ, (3 * Real.cos α₀ + 4 * Real.sin β₀ - 7)^2 + (3 * Real.sin α₀ + 4 * Real.cos β₀ - 12)^2 = 36 :=
sorry

end NUMINAMATH_CALUDE_min_value_trig_expression_l1625_162556


namespace NUMINAMATH_CALUDE_amount_distributed_l1625_162544

theorem amount_distributed (A : ℝ) : 
  (A / 20 = A / 25 + 100) → A = 10000 := by
  sorry

end NUMINAMATH_CALUDE_amount_distributed_l1625_162544


namespace NUMINAMATH_CALUDE_first_place_points_l1625_162572

/-- Represents a tournament with the given conditions -/
structure Tournament :=
  (num_teams : Nat)
  (games_per_pair : Nat)
  (total_points : Nat)
  (last_place_points : Nat)

/-- Calculates the number of games played in the tournament -/
def num_games (t : Tournament) : Nat :=
  (t.num_teams.choose 2) * t.games_per_pair

/-- Theorem stating the first-place team's points in the given tournament conditions -/
theorem first_place_points (t : Tournament)
  (h1 : t.num_teams = 4)
  (h2 : t.games_per_pair = 2)
  (h3 : t.total_points = num_games t * 2)
  (h4 : t.last_place_points = 5) :
  ∃ (first_place_points : Nat),
    first_place_points = 7 ∧
    first_place_points + t.last_place_points ≤ t.total_points :=
by
  sorry


end NUMINAMATH_CALUDE_first_place_points_l1625_162572


namespace NUMINAMATH_CALUDE_last_term_of_ap_l1625_162550

def arithmeticProgression (a : ℕ) (d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d

theorem last_term_of_ap : 
  let a := 2  -- first term
  let d := 2  -- common difference
  let n := 31 -- number of terms
  arithmeticProgression a d n = 62 := by
  sorry

end NUMINAMATH_CALUDE_last_term_of_ap_l1625_162550


namespace NUMINAMATH_CALUDE_triangle_theorem_l1625_162567

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem triangle_theorem (t : Triangle) 
  (h1 : t.A + t.B + t.C = Real.pi)
  (h2 : Real.sin t.C * Real.sin (t.A - t.B) = Real.sin t.B * Real.sin (t.C - t.A)) :
  (t.A = 2 * t.B → t.C = 5 * Real.pi / 8) ∧ 
  (2 * t.a^2 = t.b^2 + t.c^2) := by
  sorry


end NUMINAMATH_CALUDE_triangle_theorem_l1625_162567


namespace NUMINAMATH_CALUDE_real_part_of_i_times_one_plus_i_l1625_162582

theorem real_part_of_i_times_one_plus_i : Complex.re (Complex.I * (1 + Complex.I)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_i_times_one_plus_i_l1625_162582


namespace NUMINAMATH_CALUDE_opposite_pairs_l1625_162554

theorem opposite_pairs (a b : ℝ) : 
  (∀ x, (a + b) + (-a - b) = x ↔ x = 0) ∧ 
  (∀ x, (-a + b) + (a - b) = x ↔ x = 0) ∧ 
  ¬(∀ x, (a - b) + (-a - b) = x ↔ x = 0) ∧ 
  ¬(∀ x, (a + 1) + (1 - a) = x ↔ x = 0) :=
by sorry

end NUMINAMATH_CALUDE_opposite_pairs_l1625_162554


namespace NUMINAMATH_CALUDE_millions_to_scientific_l1625_162557

-- Define the number in millions
def number_in_millions : ℝ := 3.111

-- Define the number in standard form
def number_standard : ℝ := 3111000

-- Define the number in scientific notation
def number_scientific : ℝ := 3.111 * (10 ^ 6)

-- Theorem to prove
theorem millions_to_scientific : number_standard = number_scientific := by
  sorry

end NUMINAMATH_CALUDE_millions_to_scientific_l1625_162557


namespace NUMINAMATH_CALUDE_question_mark_value_l1625_162508

theorem question_mark_value : ∃ x : ℚ, (786 * x) / 30 = 1938.8 ∧ x = 74 := by sorry

end NUMINAMATH_CALUDE_question_mark_value_l1625_162508


namespace NUMINAMATH_CALUDE_snowdrift_depth_change_l1625_162569

/-- Given a snowdrift with certain depth changes over four days, 
    calculate the amount of snow added on the fourth day. -/
theorem snowdrift_depth_change (initial_depth final_depth third_day_addition : ℕ) : 
  initial_depth = 20 →
  final_depth = 34 →
  third_day_addition = 6 →
  final_depth - (initial_depth / 2 + third_day_addition) = 18 := by
  sorry

#check snowdrift_depth_change

end NUMINAMATH_CALUDE_snowdrift_depth_change_l1625_162569


namespace NUMINAMATH_CALUDE_arrangement_theorem_l1625_162540

/-- The number of ways to arrange 2 teachers and 5 students in a row,
    with the teachers adjacent but not at the ends. -/
def arrangement_count : ℕ := 960

/-- The number of ways to arrange n distinct objects. -/
def permutations (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to choose k objects from n distinct objects,
    where order matters. -/
def permutations_of_k (n k : ℕ) : ℕ := 
  if k ≤ n then Nat.factorial n / Nat.factorial (n - k) else 0

theorem arrangement_theorem :
  arrangement_count = 
    2 * permutations_of_k 5 2 * permutations 4 :=
by sorry

end NUMINAMATH_CALUDE_arrangement_theorem_l1625_162540


namespace NUMINAMATH_CALUDE_leftover_coins_value_l1625_162507

/-- The number of nickels in a complete roll -/
def nickels_per_roll : ℕ := 40

/-- The number of pennies in a complete roll -/
def pennies_per_roll : ℕ := 50

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The value of a penny in cents -/
def penny_value : ℕ := 1

/-- Michael's nickels -/
def michael_nickels : ℕ := 183

/-- Michael's pennies -/
def michael_pennies : ℕ := 259

/-- Sarah's nickels -/
def sarah_nickels : ℕ := 167

/-- Sarah's pennies -/
def sarah_pennies : ℕ := 342

/-- The value of leftover coins in cents -/
def leftover_value : ℕ :=
  ((michael_nickels + sarah_nickels) % nickels_per_roll) * nickel_value +
  ((michael_pennies + sarah_pennies) % pennies_per_roll) * penny_value

theorem leftover_coins_value : leftover_value = 151 := by
  sorry

end NUMINAMATH_CALUDE_leftover_coins_value_l1625_162507


namespace NUMINAMATH_CALUDE_divisibility_of_expression_l1625_162581

theorem divisibility_of_expression (m : ℕ) 
  (h1 : m > 0) 
  (h2 : Odd m) 
  (h3 : ¬(3 ∣ m)) : 
  112 ∣ (Int.floor (4^m - (2 + Real.sqrt 2)^m)) := by
sorry

end NUMINAMATH_CALUDE_divisibility_of_expression_l1625_162581


namespace NUMINAMATH_CALUDE_valid_arrangements_count_l1625_162565

/-- The number of ways to arrange n people in a row -/
def totalArrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange n people in a row where k specific people sit together -/
def arrangementsWithGrouped (n k : ℕ) : ℕ :=
  Nat.factorial (n - k + 1) * Nat.factorial k

/-- The number of valid arrangements for 8 people where 3 specific people cannot sit together -/
def validArrangements : ℕ :=
  totalArrangements 8 - arrangementsWithGrouped 8 3

theorem valid_arrangements_count :
  validArrangements = 36000 := by sorry

end NUMINAMATH_CALUDE_valid_arrangements_count_l1625_162565


namespace NUMINAMATH_CALUDE_tetromino_properties_l1625_162527

-- Define a tetromino as a shape formed from 4 squares
structure Tetromino :=
  (squares : Finset (ℤ × ℤ))
  (size : squares.card = 4)

-- Define rotation equivalence
def rotationEquivalent (t1 t2 : Tetromino) : Prop := sorry

-- Define the set of distinct tetrominos
def distinctTetrominos : Finset Tetromino := sorry

-- Define a tiling of a rectangle
def tiling (w h : ℕ) (pieces : Finset Tetromino) : Prop := sorry

theorem tetromino_properties :
  -- There are exactly 7 distinct tetrominos
  distinctTetrominos.card = 7 ∧
  -- It is impossible to tile a 4 × 7 rectangle with one of each distinct tetromino
  ¬ tiling 4 7 distinctTetrominos := by sorry

end NUMINAMATH_CALUDE_tetromino_properties_l1625_162527


namespace NUMINAMATH_CALUDE_resistor_value_l1625_162553

/-- Given two identical resistors R₀ connected in series, with a voltmeter reading U across one resistor
    and an ammeter reading I when replacing the voltmeter, prove that R₀ = 9 Ω. -/
theorem resistor_value (R₀ : ℝ) (U I : ℝ) : 
  U = 9 → I = 2 → R₀ = 9 := by
  sorry

end NUMINAMATH_CALUDE_resistor_value_l1625_162553


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1625_162593

theorem polynomial_factorization (x : ℝ) : 
  x^15 + x^10 + x^5 + 1 = (x^2 + x + 1) * (x^13 - x^12 + x^10 - x^9 + x^7 - x^6 + x^4 - x^3 + x - 1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1625_162593


namespace NUMINAMATH_CALUDE_total_amount_paid_l1625_162599

def grape_quantity : ℕ := 7
def grape_rate : ℕ := 70
def mango_quantity : ℕ := 9
def mango_rate : ℕ := 55

theorem total_amount_paid :
  grape_quantity * grape_rate + mango_quantity * mango_rate = 985 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_paid_l1625_162599


namespace NUMINAMATH_CALUDE_shopping_trip_percentages_l1625_162524

/-- Represents the percentage of total amount spent on each category -/
structure SpendingPercentages where
  clothing : ℝ
  food : ℝ
  other : ℝ

/-- Represents the tax rates for each category -/
structure TaxRates where
  clothing : ℝ
  food : ℝ
  other : ℝ

/-- The problem statement -/
theorem shopping_trip_percentages 
  (s : SpendingPercentages)
  (t : TaxRates)
  (h1 : s.clothing = 0.40)
  (h2 : s.other = 0.30)
  (h3 : s.clothing + s.food + s.other = 1)
  (h4 : t.clothing = 0.04)
  (h5 : t.food = 0)
  (h6 : t.other = 0.08)
  (h7 : t.clothing * s.clothing + t.food * s.food + t.other * s.other = 0.04) :
  s.food = 0.30 := by
  sorry


end NUMINAMATH_CALUDE_shopping_trip_percentages_l1625_162524


namespace NUMINAMATH_CALUDE_mathilda_debt_l1625_162518

theorem mathilda_debt (initial_payment : ℝ) (remaining_percentage : ℝ) (original_debt : ℝ) : 
  initial_payment = 125 →
  remaining_percentage = 75 →
  initial_payment = (100 - remaining_percentage) / 100 * original_debt →
  original_debt = 500 := by
sorry

end NUMINAMATH_CALUDE_mathilda_debt_l1625_162518


namespace NUMINAMATH_CALUDE_garage_wheels_eq_22_l1625_162588

/-- The number of wheels in Timmy's parents' garage -/
def garage_wheels : ℕ :=
  let num_cars : ℕ := 2
  let num_lawnmowers : ℕ := 1
  let num_bicycles : ℕ := 3
  let num_tricycles : ℕ := 1
  let num_unicycles : ℕ := 1
  let wheels_per_car : ℕ := 4
  let wheels_per_lawnmower : ℕ := 4
  let wheels_per_bicycle : ℕ := 2
  let wheels_per_tricycle : ℕ := 3
  let wheels_per_unicycle : ℕ := 1
  num_cars * wheels_per_car +
  num_lawnmowers * wheels_per_lawnmower +
  num_bicycles * wheels_per_bicycle +
  num_tricycles * wheels_per_tricycle +
  num_unicycles * wheels_per_unicycle

theorem garage_wheels_eq_22 : garage_wheels = 22 := by
  sorry

end NUMINAMATH_CALUDE_garage_wheels_eq_22_l1625_162588


namespace NUMINAMATH_CALUDE_quadratic_inequality_equivalence_l1625_162568

theorem quadratic_inequality_equivalence (a : ℝ) : 
  (∀ x : ℝ, x^2 - x - 2 < 0 ↔ -2 < x ∧ x < a) ↔ a ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_equivalence_l1625_162568


namespace NUMINAMATH_CALUDE_maria_hardcover_volumes_l1625_162533

/-- Proof that Maria bought 9 hardcover volumes -/
theorem maria_hardcover_volumes :
  ∀ (h p : ℕ), -- h: number of hardcover volumes, p: number of paperback volumes
  h + p = 15 → -- total number of volumes
  10 * p + 30 * h = 330 → -- total cost equation
  h = 9 := by
sorry

end NUMINAMATH_CALUDE_maria_hardcover_volumes_l1625_162533


namespace NUMINAMATH_CALUDE_fair_coin_five_tosses_l1625_162502

/-- The probability of a fair coin landing on the same side for all tosses -/
def same_side_probability (n : ℕ) : ℚ :=
  (1 / 2) ^ n

/-- Theorem: The probability of a fair coin landing on the same side for 5 tosses is 1/32 -/
theorem fair_coin_five_tosses :
  same_side_probability 5 = 1 / 32 := by
  sorry


end NUMINAMATH_CALUDE_fair_coin_five_tosses_l1625_162502


namespace NUMINAMATH_CALUDE_chris_sick_one_week_l1625_162580

/-- Calculates the number of weeks Chris got sick based on Cathy's work hours -/
def weeks_chris_sick (hours_per_week : ℕ) (total_weeks : ℕ) (cathy_total_hours : ℕ) : ℕ :=
  (cathy_total_hours - (hours_per_week * total_weeks)) / hours_per_week

/-- Proves that Chris got sick for 1 week given the conditions in the problem -/
theorem chris_sick_one_week :
  let hours_per_week : ℕ := 20
  let months : ℕ := 2
  let weeks_per_month : ℕ := 4
  let total_weeks : ℕ := months * weeks_per_month
  let cathy_total_hours : ℕ := 180
  weeks_chris_sick hours_per_week total_weeks cathy_total_hours = 1 := by
  sorry

#eval weeks_chris_sick 20 8 180

end NUMINAMATH_CALUDE_chris_sick_one_week_l1625_162580


namespace NUMINAMATH_CALUDE_nested_expression_simplification_l1625_162551

theorem nested_expression_simplification (x : ℝ) : 1 - (1 - (1 - (1 + (1 - (1 - x))))) = 2 - x := by
  sorry

end NUMINAMATH_CALUDE_nested_expression_simplification_l1625_162551


namespace NUMINAMATH_CALUDE_tomato_price_per_pound_l1625_162503

/-- Calculates the price per pound of a tomato based on grocery shopping details. -/
theorem tomato_price_per_pound
  (meat_weight : Real)
  (meat_price_per_pound : Real)
  (buns_price : Real)
  (lettuce_price : Real)
  (tomato_weight : Real)
  (pickles_price : Real)
  (pickles_coupon : Real)
  (paid_amount : Real)
  (change_received : Real)
  (h1 : meat_weight = 2)
  (h2 : meat_price_per_pound = 3.5)
  (h3 : buns_price = 1.5)
  (h4 : lettuce_price = 1)
  (h5 : tomato_weight = 1.5)
  (h6 : pickles_price = 2.5)
  (h7 : pickles_coupon = 1)
  (h8 : paid_amount = 20)
  (h9 : change_received = 6) :
  (paid_amount - change_received - (meat_weight * meat_price_per_pound + buns_price + lettuce_price + (pickles_price - pickles_coupon))) / tomato_weight = 2 := by
  sorry


end NUMINAMATH_CALUDE_tomato_price_per_pound_l1625_162503


namespace NUMINAMATH_CALUDE_total_pencil_length_l1625_162559

/-- The length of Isha's first pencil in cubes -/
def first_pencil_cubes : ℕ := 12

/-- The length of each cube in Isha's first pencil in centimeters -/
def first_pencil_cube_length : ℚ := 3/2

/-- The length of the second pencil in cubes -/
def second_pencil_cubes : ℕ := 13

/-- The length of each cube in the second pencil in centimeters -/
def second_pencil_cube_length : ℚ := 17/10

/-- The total length of both pencils in centimeters -/
def total_length : ℚ := first_pencil_cubes * first_pencil_cube_length + 
                        second_pencil_cubes * second_pencil_cube_length

theorem total_pencil_length : total_length = 401/10 := by
  sorry

end NUMINAMATH_CALUDE_total_pencil_length_l1625_162559


namespace NUMINAMATH_CALUDE_max_min_on_interval_l1625_162501

def f (x : ℝ) : ℝ := 3 * x^4 + 4 * x^3 + 34

theorem max_min_on_interval :
  ∃ (max min : ℝ),
    (∀ x ∈ Set.Icc (-2 : ℝ) 1, f x ≤ max) ∧
    (∃ x ∈ Set.Icc (-2 : ℝ) 1, f x = max) ∧
    (∀ x ∈ Set.Icc (-2 : ℝ) 1, min ≤ f x) ∧
    (∃ x ∈ Set.Icc (-2 : ℝ) 1, f x = min) ∧
    max = 50 ∧ min = 33 := by
  sorry

end NUMINAMATH_CALUDE_max_min_on_interval_l1625_162501


namespace NUMINAMATH_CALUDE_total_distance_equals_expected_l1625_162560

/-- The initial travel distance per year in kilometers -/
def initial_distance : ℝ := 983400000000

/-- The factor by which the speed increases every 50 years -/
def speed_increase_factor : ℝ := 2

/-- The number of years for each speed increase -/
def years_per_increase : ℕ := 50

/-- The total number of years of travel -/
def total_years : ℕ := 150

/-- The function to calculate the total distance traveled -/
def total_distance : ℝ := 
  initial_distance * years_per_increase * (1 + speed_increase_factor + speed_increase_factor^2)

theorem total_distance_equals_expected : 
  total_distance = 3.4718e14 := by sorry

end NUMINAMATH_CALUDE_total_distance_equals_expected_l1625_162560


namespace NUMINAMATH_CALUDE_percentage_problem_l1625_162564

theorem percentage_problem (p : ℝ) (x : ℝ) 
  (h1 : (p / 100) * x = 300)
  (h2 : (120 / 100) * x = 1800) : p = 20 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l1625_162564


namespace NUMINAMATH_CALUDE_partner_C_investment_l1625_162530

/-- Represents the investment and profit structure of a business partnership --/
structure BusinessPartnership where
  investment_A : ℕ
  investment_B : ℕ
  profit_share_B : ℕ
  profit_diff_AC : ℕ

/-- Calculates the investment of partner C given the business partnership details --/
def calculate_investment_C (bp : BusinessPartnership) : ℕ :=
  -- The actual calculation would go here
  sorry

/-- Theorem stating that given the specific business partnership details, 
    partner C's investment is 120000 --/
theorem partner_C_investment 
  (bp : BusinessPartnership) 
  (h1 : bp.investment_A = 8000)
  (h2 : bp.investment_B = 10000)
  (h3 : bp.profit_share_B = 1400)
  (h4 : bp.profit_diff_AC = 560) : 
  calculate_investment_C bp = 120000 := by
  sorry

end NUMINAMATH_CALUDE_partner_C_investment_l1625_162530


namespace NUMINAMATH_CALUDE_pond_amphibians_l1625_162555

/-- Calculates the total number of amphibians observed in a pond -/
def total_amphibians (green_frogs : ℕ) (observed_tree_frogs : ℕ) (bullfrogs : ℕ) 
  (exotic_tree_frogs : ℕ) (salamanders : ℕ) (first_tadpole_group : ℕ) (baby_frogs : ℕ) 
  (newts : ℕ) (toads : ℕ) (caecilians : ℕ) : ℕ :=
  let total_tree_frogs := observed_tree_frogs * 3
  let second_tadpole_group := first_tadpole_group - (first_tadpole_group / 5)
  green_frogs + total_tree_frogs + bullfrogs + exotic_tree_frogs + salamanders + 
  first_tadpole_group + second_tadpole_group + baby_frogs + newts + toads + caecilians

/-- Theorem stating the total number of amphibians observed in the pond -/
theorem pond_amphibians : 
  total_amphibians 6 5 2 8 3 50 10 1 2 1 = 138 := by
  sorry

end NUMINAMATH_CALUDE_pond_amphibians_l1625_162555


namespace NUMINAMATH_CALUDE_trigonometric_identity_l1625_162531

theorem trigonometric_identity (α β γ : ℝ) :
  3.400 * Real.cos (α + β) * Real.cos γ + Real.cos α + Real.cos β + Real.cos γ - Real.sin (α + β) * Real.sin γ =
  4 * Real.cos ((α + β) / 2) * Real.cos ((α + γ) / 2) * Real.cos ((β + γ) / 2) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l1625_162531


namespace NUMINAMATH_CALUDE_quadratic_min_max_l1625_162592

/-- The quadratic function f(x) = 2x^2 - 8x + 3 -/
def f (x : ℝ) : ℝ := 2 * x^2 - 8 * x + 3

theorem quadratic_min_max :
  (∀ x : ℝ, f x ≥ -5) ∧
  (f 2 = -5) ∧
  (∀ M : ℝ, ∃ x : ℝ, f x > M) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_min_max_l1625_162592


namespace NUMINAMATH_CALUDE_a_plus_b_equals_one_l1625_162528

-- Define the universe U as the real numbers
def U : Type := ℝ

-- Define set A
def A (a b : ℝ) : Set ℝ := {x | (x^2 + a*x + b)*(x - 1) = 0}

-- Define the theorem
theorem a_plus_b_equals_one (a b : ℝ) (B : Set ℝ) :
  (∃ (B : Set ℝ), (A a b ∩ B = {1, 2}) ∧ (A a b ∩ (Set.univ \ B) = {3})) →
  a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_a_plus_b_equals_one_l1625_162528


namespace NUMINAMATH_CALUDE_parallelograms_count_formula_l1625_162549

/-- The number of parallelograms formed by the intersection of two sets of parallel lines -/
def parallelograms_count (m n : ℕ) : ℕ :=
  Nat.choose m 2 * Nat.choose n 2

/-- Theorem stating that the number of parallelograms formed by the intersection
    of two sets of parallel lines is equal to C_m^2 * C_n^2 -/
theorem parallelograms_count_formula (m n : ℕ) :
  parallelograms_count m n = Nat.choose m 2 * Nat.choose n 2 := by
  sorry

end NUMINAMATH_CALUDE_parallelograms_count_formula_l1625_162549


namespace NUMINAMATH_CALUDE_recyclable_containers_l1625_162526

theorem recyclable_containers (total_guests : ℕ) (soda_cans : ℕ) (water_bottles : ℕ) (juice_bottles : ℕ)
  (h_guests : total_guests = 90)
  (h_soda : soda_cans = 50)
  (h_water : water_bottles = 50)
  (h_juice : juice_bottles = 50)
  (h_soda_drinkers : total_guests / 2 = 45)
  (h_water_drinkers : total_guests / 3 = 30)
  (h_juice_consumed : juice_bottles * 4 / 5 = 40) :
  45 + 30 + 40 = 115 := by
  sorry

#check recyclable_containers

end NUMINAMATH_CALUDE_recyclable_containers_l1625_162526


namespace NUMINAMATH_CALUDE_solve_fraction_equation_l1625_162552

theorem solve_fraction_equation :
  ∀ x : ℚ, (2 / 3 : ℚ) - (1 / 4 : ℚ) = 1 / x → x = 12 / 5 := by
  sorry

end NUMINAMATH_CALUDE_solve_fraction_equation_l1625_162552


namespace NUMINAMATH_CALUDE_rectangle_forms_same_solid_l1625_162516

-- Define the shapes
inductive Shape
  | RightTriangle
  | Rectangle
  | RightTrapezoid
  | IsoscelesRightTriangle

-- Define a function that determines if a shape forms the same solid when rotated around any edge
def forms_same_solid (s : Shape) : Prop :=
  match s with
  | Shape.Rectangle => true
  | _ => false

-- Theorem statement
theorem rectangle_forms_same_solid :
  ∀ s : Shape, forms_same_solid s ↔ s = Shape.Rectangle :=
by sorry

end NUMINAMATH_CALUDE_rectangle_forms_same_solid_l1625_162516


namespace NUMINAMATH_CALUDE_evaluate_expression_l1625_162545

-- Define the $ operation
def dollar (a b : ℝ) : ℝ := (a - b)^2

-- State the theorem
theorem evaluate_expression (x y : ℝ) : 
  dollar ((x + y)^2) ((y - x)^2) = 16 * x^2 * y^2 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1625_162545


namespace NUMINAMATH_CALUDE_correct_sampling_pairing_l1625_162585

-- Define the sampling methods
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified

-- Define the sampling scenarios
structure SamplingScenario where
  description : String
  populationSize : Nat
  sampleSize : Nat
  hasStrata : Bool

-- Define the correct pairing function
def correctPairing (scenario : SamplingScenario) : SamplingMethod :=
  if scenario.hasStrata then
    SamplingMethod.Stratified
  else if scenario.populationSize ≤ 100 then
    SamplingMethod.SimpleRandom
  else
    SamplingMethod.Systematic

-- Define the three scenarios
def universityScenario : SamplingScenario :=
  { description := "University student sampling"
  , populationSize := 300
  , sampleSize := 100
  , hasStrata := true }

def productScenario : SamplingScenario :=
  { description := "Product quality inspection"
  , populationSize := 20
  , sampleSize := 7
  , hasStrata := false }

def habitScenario : SamplingScenario :=
  { description := "Daily habits sampling"
  , populationSize := 2000
  , sampleSize := 10
  , hasStrata := false }

-- Theorem statement
theorem correct_sampling_pairing :
  (correctPairing universityScenario = SamplingMethod.Stratified) ∧
  (correctPairing productScenario = SamplingMethod.SimpleRandom) ∧
  (correctPairing habitScenario = SamplingMethod.Systematic) :=
sorry

end NUMINAMATH_CALUDE_correct_sampling_pairing_l1625_162585


namespace NUMINAMATH_CALUDE_arithmetic_matrix_middle_value_l1625_162583

/-- Represents a 5x5 matrix where each row and column forms an arithmetic sequence -/
def ArithmeticMatrix := Matrix (Fin 5) (Fin 5) ℝ

/-- Checks if a given row or column of the matrix forms an arithmetic sequence -/
def isArithmeticSequence (seq : Fin 5 → ℝ) : Prop :=
  ∃ d : ℝ, ∀ i : Fin 5, i.val < 4 → seq (i + 1) = seq i + d

/-- The property that all rows and columns of the matrix form arithmetic sequences -/
def allArithmeticSequences (M : ArithmeticMatrix) : Prop :=
  (∀ i : Fin 5, isArithmeticSequence (λ j => M i j)) ∧
  (∀ j : Fin 5, isArithmeticSequence (λ i => M i j))

theorem arithmetic_matrix_middle_value
  (M : ArithmeticMatrix)
  (all_arithmetic : allArithmeticSequences M)
  (first_row_start : M 0 0 = 3)
  (first_row_end : M 0 4 = 15)
  (last_row_start : M 4 0 = 25)
  (last_row_end : M 4 4 = 65) :
  M 2 2 = 27 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_matrix_middle_value_l1625_162583


namespace NUMINAMATH_CALUDE_early_bird_dinner_bill_l1625_162548

def early_bird_dinner (curtis_steak rob_steak curtis_side rob_side curtis_drink rob_drink : ℝ)
  (discount_rate tax_rate tip_rate : ℝ) : ℝ :=
  let discounted_curtis_steak := curtis_steak * discount_rate
  let discounted_rob_steak := rob_steak * discount_rate
  let curtis_total := discounted_curtis_steak + curtis_side + curtis_drink
  let rob_total := discounted_rob_steak + rob_side + rob_drink
  let combined_total := curtis_total + rob_total
  let tax := combined_total * tax_rate
  let tip := combined_total * tip_rate
  combined_total + tax + tip

theorem early_bird_dinner_bill : 
  early_bird_dinner 16 18 6 7 3 3.5 0.5 0.07 0.2 = 46.36 := by
  sorry

end NUMINAMATH_CALUDE_early_bird_dinner_bill_l1625_162548


namespace NUMINAMATH_CALUDE_vector_c_satisfies_conditions_l1625_162522

/-- Given vectors a and b in ℝ², prove that vector c satisfies the required conditions -/
theorem vector_c_satisfies_conditions (a b c : ℝ × ℝ) : 
  a = (1, 2) → b = (2, -3) → c = (7/2, -7/4) → 
  (c.1 * a.1 + c.2 * a.2 = 0) ∧ 
  (∃ k : ℝ, b.1 = k * (a.1 - c.1) ∧ b.2 = k * (a.2 - c.2)) := by
sorry

end NUMINAMATH_CALUDE_vector_c_satisfies_conditions_l1625_162522


namespace NUMINAMATH_CALUDE_domain_of_f_composed_with_exp2_l1625_162512

/-- Given a function f with domain (1, 2), this theorem states that the domain of f(2^x) is (0, 1) -/
theorem domain_of_f_composed_with_exp2 (f : ℝ → ℝ) :
  (∀ x, f x ≠ 0 → 1 < x ∧ x < 2) →
  (∀ x, f (2^x) ≠ 0 → 0 < x ∧ x < 1) :=
sorry


end NUMINAMATH_CALUDE_domain_of_f_composed_with_exp2_l1625_162512


namespace NUMINAMATH_CALUDE_complement_union_theorem_l1625_162517

open Set

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 4}
def N : Set ℕ := {1, 3, 5}

theorem complement_union_theorem : 
  (U \ M) ∪ (U \ N) = {2, 3, 4, 5} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l1625_162517


namespace NUMINAMATH_CALUDE_cos_sin_sum_equals_half_l1625_162521

theorem cos_sin_sum_equals_half : 
  Real.cos (π / 4) * Real.cos (π / 12) - Real.sin (π / 4) * Real.sin (π / 12) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_sin_sum_equals_half_l1625_162521


namespace NUMINAMATH_CALUDE_second_group_average_age_l1625_162505

theorem second_group_average_age 
  (n₁ : ℕ) (n₂ : ℕ) (m₁ : ℝ) (m_combined : ℝ) :
  n₁ = 11 →
  n₂ = 7 →
  m₁ = 25 →
  m_combined = 32 →
  (n₁ * m₁ + n₂ * ((n₁ + n₂) * m_combined - n₁ * m₁) / n₂) / (n₁ + n₂) = m_combined →
  ((n₁ + n₂) * m_combined - n₁ * m₁) / n₂ = 43 := by
sorry

end NUMINAMATH_CALUDE_second_group_average_age_l1625_162505


namespace NUMINAMATH_CALUDE_basement_water_pump_time_l1625_162578

/-- Proves that it takes 225 minutes to pump out water from a flooded basement --/
theorem basement_water_pump_time : 
  let basement_length : ℝ := 30
  let basement_width : ℝ := 40
  let water_depth_inches : ℝ := 12
  let num_pumps : ℕ := 4
  let pump_rate : ℝ := 10  -- gallons per minute
  let gallons_per_cubic_foot : ℝ := 7.5
  let inches_per_foot : ℝ := 12

  let water_depth_feet : ℝ := water_depth_inches / inches_per_foot
  let water_volume_cubic_feet : ℝ := basement_length * basement_width * water_depth_feet
  let water_volume_gallons : ℝ := water_volume_cubic_feet * gallons_per_cubic_foot
  let total_pump_rate : ℝ := num_pumps * pump_rate
  let pump_time_minutes : ℝ := water_volume_gallons / total_pump_rate

  pump_time_minutes = 225 := by
  sorry

end NUMINAMATH_CALUDE_basement_water_pump_time_l1625_162578


namespace NUMINAMATH_CALUDE_smallest_divisor_after_323_l1625_162538

theorem smallest_divisor_after_323 (n : ℕ) (h1 : 1000 ≤ n ∧ n < 10000) 
  (h2 : Even n) (h3 : n % 323 = 0) :
  (∃ k : ℕ, k > 323 ∧ n % k = 0 ∧ ∀ m : ℕ, m > 323 ∧ n % m = 0 → k ≤ m) ∧
  (∀ k : ℕ, k > 323 ∧ n % k = 0 ∧ (∀ m : ℕ, m > 323 ∧ n % m = 0 → k ≤ m) → k = 340) :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisor_after_323_l1625_162538


namespace NUMINAMATH_CALUDE_extra_day_percentage_increase_l1625_162558

/-- Calculates the percentage increase in daily rate for an extra workday --/
theorem extra_day_percentage_increase
  (regular_daily_rate : ℚ)
  (regular_work_days_per_week : ℕ)
  (weeks_per_month : ℕ)
  (total_monthly_earnings_with_extra_day : ℚ)
  (h1 : regular_daily_rate = 8)
  (h2 : regular_work_days_per_week = 5)
  (h3 : weeks_per_month = 4)
  (h4 : total_monthly_earnings_with_extra_day = 208) :
  let regular_monthly_earnings := regular_daily_rate * regular_work_days_per_week * weeks_per_month
  let extra_day_earnings := total_monthly_earnings_with_extra_day - regular_monthly_earnings
  let extra_day_rate := extra_day_earnings / weeks_per_month
  let percentage_increase := (extra_day_rate - regular_daily_rate) / regular_daily_rate * 100
  percentage_increase = 50 := by
sorry

end NUMINAMATH_CALUDE_extra_day_percentage_increase_l1625_162558


namespace NUMINAMATH_CALUDE_modular_inverse_13_mod_2000_l1625_162547

theorem modular_inverse_13_mod_2000 : ∃ x : ℤ, 0 ≤ x ∧ x < 2000 ∧ (13 * x) % 2000 = 1 :=
by
  use 1077
  sorry

end NUMINAMATH_CALUDE_modular_inverse_13_mod_2000_l1625_162547


namespace NUMINAMATH_CALUDE_marcella_shoes_l1625_162510

theorem marcella_shoes (initial_pairs : ℕ) : 
  (initial_pairs * 2 - 9 ≥ 21 * 2) ∧ 
  (∀ n : ℕ, n > initial_pairs → n * 2 - 9 < 21 * 2) → 
  initial_pairs = 25 := by
sorry

end NUMINAMATH_CALUDE_marcella_shoes_l1625_162510


namespace NUMINAMATH_CALUDE_hexagon_diagonal_small_triangle_l1625_162511

/-- A convex hexagon in the plane -/
structure ConvexHexagon where
  -- We don't need to define the specific properties of a convex hexagon for this statement
  area : ℝ
  area_pos : area > 0

/-- A diagonal of a hexagon -/
structure Diagonal (h : ConvexHexagon) where
  -- We don't need to define the specific properties of a diagonal for this statement

/-- The area of the triangle cut off by a diagonal -/
noncomputable def triangle_area (h : ConvexHexagon) (d : Diagonal h) : ℝ :=
  sorry -- Definition not provided, as it's not part of the original conditions

theorem hexagon_diagonal_small_triangle (h : ConvexHexagon) :
  ∃ (d : Diagonal h), triangle_area h d ≤ h.area / 6 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_diagonal_small_triangle_l1625_162511


namespace NUMINAMATH_CALUDE_goldbach_conjecture_negation_l1625_162563

-- Define the Goldbach Conjecture
def goldbach_conjecture : Prop :=
  ∀ n : ℕ, n > 2 → Even n → ∃ p q : ℕ, Prime p ∧ Prime q ∧ n = p + q

-- State the theorem
theorem goldbach_conjecture_negation :
  ¬goldbach_conjecture ↔ ∃ n : ℕ, n > 2 ∧ Even n ∧ ¬∃ p q : ℕ, Prime p ∧ Prime q ∧ n = p + q :=
by
  sorry

end NUMINAMATH_CALUDE_goldbach_conjecture_negation_l1625_162563


namespace NUMINAMATH_CALUDE_circle_area_quadrupled_l1625_162520

theorem circle_area_quadrupled (r n : ℝ) : 
  (r > 0) → (n > 0) → (π * (r + n)^2 = 4 * π * r^2) → r = n / 3 :=
by sorry

end NUMINAMATH_CALUDE_circle_area_quadrupled_l1625_162520


namespace NUMINAMATH_CALUDE_trajectory_intersection_slope_ratio_l1625_162586

-- Define the curve E: y² = 2x
def E : Set (ℝ × ℝ) := {p | p.2^2 = 2 * p.1}

-- Define points S and Q
def S : ℝ × ℝ := (2, 0)
def Q : ℝ × ℝ := (1, 0)

-- Define the theorem
theorem trajectory_intersection_slope_ratio 
  (k₁ : ℝ) 
  (A B C D : ℝ × ℝ) 
  (hA : A ∈ E) 
  (hB : B ∈ E) 
  (hC : C ∈ E) 
  (hD : D ∈ E) 
  (hAB : (B.2 - A.2) = k₁ * (B.1 - A.1)) 
  (hABS : (A.2 - S.2) = k₁ * (A.1 - S.1)) 
  (hAC : (C.2 - A.2) * (Q.1 - A.1) = (Q.2 - A.2) * (C.1 - A.1)) 
  (hBD : (D.2 - B.2) * (Q.1 - B.1) = (Q.2 - B.2) * (D.1 - B.1)) :
  ∃ (k₂ : ℝ), (D.2 - C.2) = k₂ * (D.1 - C.1) ∧ k₂ / k₁ = 2 := by
sorry

end NUMINAMATH_CALUDE_trajectory_intersection_slope_ratio_l1625_162586


namespace NUMINAMATH_CALUDE_quadratic_function_positive_l1625_162574

/-- The quadratic function y = ax² - 2ax + 3 -/
def f (a x : ℝ) : ℝ := a * x^2 - 2 * a * x + 3

/-- The set of x values we're interested in -/
def X : Set ℝ := {x | 0 < x ∧ x < 3}

/-- The set of a values that satisfy the condition -/
def A : Set ℝ := {a | -1 ≤ a ∧ a < 0} ∪ {a | 0 < a ∧ a < 3}

theorem quadratic_function_positive (a : ℝ) :
  (∀ x ∈ X, f a x > 0) ↔ a ∈ A :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_positive_l1625_162574


namespace NUMINAMATH_CALUDE_fibonacci_fifth_divisible_by_five_l1625_162541

def fibonacci : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fibonacci (n + 1) + fibonacci n

theorem fibonacci_fifth_divisible_by_five (k : ℕ) :
  5 ∣ fibonacci (5 * k) := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_fifth_divisible_by_five_l1625_162541


namespace NUMINAMATH_CALUDE_sum_of_four_consecutive_integers_l1625_162589

theorem sum_of_four_consecutive_integers (n : ℤ) :
  n + (n + 1) + (n + 2) + (n + 3) = 34 → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_consecutive_integers_l1625_162589


namespace NUMINAMATH_CALUDE_remainder_problem_l1625_162539

theorem remainder_problem (N : ℕ) (k : ℕ) (h : N = 35 * k + 25) :
  N % 15 = 10 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l1625_162539


namespace NUMINAMATH_CALUDE_min_beta_delta_sum_l1625_162515

open Complex

/-- A complex-valued function with specific properties -/
def f (β δ : ℂ) (z : ℂ) : ℂ := (3 + 2*I)*z^2 + β*z + δ

/-- The theorem stating the minimum value of |β| + |δ| -/
theorem min_beta_delta_sum :
  ∃ (β δ : ℂ), 
    (∀ (β' δ' : ℂ), (f β' δ' (1 + I)).im = 0 ∧ (f β' δ' (-I)).im = 0 → 
      Complex.abs β + Complex.abs δ ≤ Complex.abs β' + Complex.abs δ') ∧
    Complex.abs β + Complex.abs δ = Real.sqrt 5 + 3 := by
  sorry

end NUMINAMATH_CALUDE_min_beta_delta_sum_l1625_162515


namespace NUMINAMATH_CALUDE_cartesian_product_A_B_l1625_162577

def A : Set ℕ := {1, 2}
def B : Set ℕ := {1, 2, 3}

theorem cartesian_product_A_B :
  A ×ˢ B = {(1,1), (1,2), (1,3), (2,1), (2,2), (2,3)} := by
  sorry

end NUMINAMATH_CALUDE_cartesian_product_A_B_l1625_162577


namespace NUMINAMATH_CALUDE_circle_ratio_l1625_162561

theorem circle_ratio (r R a b : ℝ) (hr : r > 0) (hR : R > r) (hab : a > b) (hb : b > 0) 
  (h : R^2 = (a/b) * (R^2 - r^2)) : 
  R/r = Real.sqrt (a/(a-b)) := by
  sorry

end NUMINAMATH_CALUDE_circle_ratio_l1625_162561


namespace NUMINAMATH_CALUDE_triangle_sine_theorem_l1625_162598

theorem triangle_sine_theorem (D E F : ℝ) (area : ℝ) (geo_mean : ℝ) :
  area = 81 →
  geo_mean = 15 →
  geo_mean^2 = D * F →
  area = 1/2 * D * F * Real.sin E →
  Real.sin E = 18/25 := by
  sorry

end NUMINAMATH_CALUDE_triangle_sine_theorem_l1625_162598


namespace NUMINAMATH_CALUDE_norris_remaining_money_l1625_162587

/-- Calculates the remaining money for Norris after savings and spending --/
theorem norris_remaining_money 
  (september_savings : ℕ) 
  (october_savings : ℕ) 
  (november_savings : ℕ) 
  (game_cost : ℕ) : 
  september_savings = 29 →
  october_savings = 25 →
  november_savings = 31 →
  game_cost = 75 →
  (september_savings + october_savings + november_savings) - game_cost = 10 := by
  sorry

end NUMINAMATH_CALUDE_norris_remaining_money_l1625_162587


namespace NUMINAMATH_CALUDE_remaining_balloons_l1625_162514

-- Define the type for balloon labels
inductive BalloonLabel
| A | B | C | D | E | F | G | H | I | J | K | L

-- Define the function to get the next balloon to pop
def nextBalloon (current : BalloonLabel) : BalloonLabel :=
  match current with
  | BalloonLabel.A => BalloonLabel.D
  | BalloonLabel.B => BalloonLabel.E
  | BalloonLabel.C => BalloonLabel.F
  | BalloonLabel.D => BalloonLabel.G
  | BalloonLabel.E => BalloonLabel.H
  | BalloonLabel.F => BalloonLabel.I
  | BalloonLabel.G => BalloonLabel.J
  | BalloonLabel.H => BalloonLabel.K
  | BalloonLabel.I => BalloonLabel.L
  | BalloonLabel.J => BalloonLabel.A
  | BalloonLabel.K => BalloonLabel.B
  | BalloonLabel.L => BalloonLabel.C

-- Define the function to pop balloons
def popBalloons (start : BalloonLabel) (n : Nat) : List BalloonLabel :=
  if n = 0 then []
  else start :: popBalloons (nextBalloon (nextBalloon start)) (n - 1)

-- Theorem statement
theorem remaining_balloons :
  popBalloons BalloonLabel.C 10 = [BalloonLabel.C, BalloonLabel.F, BalloonLabel.I, BalloonLabel.L, BalloonLabel.D, BalloonLabel.H, BalloonLabel.A, BalloonLabel.G, BalloonLabel.B, BalloonLabel.K] ∧
  (∀ b : BalloonLabel, b ∉ popBalloons BalloonLabel.C 10 → b = BalloonLabel.E ∨ b = BalloonLabel.J) :=
by sorry


end NUMINAMATH_CALUDE_remaining_balloons_l1625_162514


namespace NUMINAMATH_CALUDE_chess_club_boys_l1625_162532

theorem chess_club_boys (total_members : ℕ) (total_attendees : ℕ) :
  total_members = 30 →
  total_attendees = 18 →
  ∃ (boys girls : ℕ),
    boys + girls = total_members ∧
    boys + (2 * girls / 3) = total_attendees ∧
    boys = 6 := by
  sorry

end NUMINAMATH_CALUDE_chess_club_boys_l1625_162532


namespace NUMINAMATH_CALUDE_complex_arithmetic_equality_l1625_162594

theorem complex_arithmetic_equality : 
  |(-3) - (-5)| + ((-1/2 : ℚ)^3) / (1/4 : ℚ) * 2 - 6 * ((1/3 : ℚ) - (1/2 : ℚ)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_equality_l1625_162594


namespace NUMINAMATH_CALUDE_supplement_of_supplement_35_l1625_162525

/-- The supplement of an angle is the angle that, when added to the original angle, forms a straight angle (180 degrees). -/
def supplement (angle : ℝ) : ℝ := 180 - angle

/-- Theorem: The supplement of the supplement of a 35-degree angle is 35 degrees. -/
theorem supplement_of_supplement_35 :
  supplement (supplement 35) = 35 := by
  sorry

end NUMINAMATH_CALUDE_supplement_of_supplement_35_l1625_162525


namespace NUMINAMATH_CALUDE_three_letter_initials_count_l1625_162584

theorem three_letter_initials_count (n : ℕ) (h : n = 7) : n^3 = 343 := by
  sorry

end NUMINAMATH_CALUDE_three_letter_initials_count_l1625_162584


namespace NUMINAMATH_CALUDE_solve_equation_l1625_162504

theorem solve_equation (x : ℝ) (n : ℝ) (expr : ℝ → ℝ) : 
  x = 1 → 
  n = 4 * x → 
  2 * x * expr x = 10 → 
  n = 4 := by sorry

end NUMINAMATH_CALUDE_solve_equation_l1625_162504


namespace NUMINAMATH_CALUDE_cost_of_3200_pencils_l1625_162562

/-- The cost of a given number of pencils based on a known price for a box of pencils -/
def pencil_cost (box_size : ℕ) (box_cost : ℚ) (num_pencils : ℕ) : ℚ :=
  (box_cost * num_pencils) / box_size

/-- Theorem: Given a box of 160 personalized pencils costs $48, the cost of 3200 pencils is $960 -/
theorem cost_of_3200_pencils :
  pencil_cost 160 48 3200 = 960 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_3200_pencils_l1625_162562


namespace NUMINAMATH_CALUDE_square_field_area_l1625_162591

/-- The area of a square field with side length 17 meters is 289 square meters. -/
theorem square_field_area :
  ∀ (side_length area : ℝ),
  side_length = 17 →
  area = side_length * side_length →
  area = 289 :=
by sorry

end NUMINAMATH_CALUDE_square_field_area_l1625_162591


namespace NUMINAMATH_CALUDE_solution_implies_a_value_l1625_162500

theorem solution_implies_a_value (a : ℝ) :
  (5 * a - 8 = 10 + 4 * a) → a = 18 := by
  sorry

end NUMINAMATH_CALUDE_solution_implies_a_value_l1625_162500


namespace NUMINAMATH_CALUDE_a_union_b_iff_c_l1625_162519

-- Define the sets A, B, and C
def A : Set ℝ := {x | x > 2}
def B : Set ℝ := {x | x < 0}
def C : Set ℝ := {x | x * (x - 2) > 0}

-- State the theorem
theorem a_union_b_iff_c : ∀ x : ℝ, x ∈ (A ∪ B) ↔ x ∈ C := by
  sorry

end NUMINAMATH_CALUDE_a_union_b_iff_c_l1625_162519


namespace NUMINAMATH_CALUDE_unique_solutions_l1625_162543

-- Define the coprime relation
def coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

-- Define the equation
def satisfies_equation (x y : ℕ) : Prop := x^2 - x + 1 = y^3

-- Main theorem
theorem unique_solutions :
  ∀ x y : ℕ, x > 0 ∧ y > 0 →
  coprime x (y-1) →
  satisfies_equation x y →
  (x = 1 ∧ y = 1) ∨ (x = 19 ∧ y = 7) :=
sorry

end NUMINAMATH_CALUDE_unique_solutions_l1625_162543


namespace NUMINAMATH_CALUDE_min_value_problem_l1625_162575

theorem min_value_problem (a b c d e f g h : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (pos_e : 0 < e) (pos_f : 0 < f) (pos_g : 0 < g) (pos_h : 0 < h)
  (h1 : a * b * c * d = 8)
  (h2 : e * f * g * h = 16)
  (h3 : a + b + c + d = e * f * g) :
  64 ≤ (a*e)^2 + (b*f)^2 + (c*g)^2 + (d*h)^2 ∧ 
  ∃ (a' b' c' d' e' f' g' h' : ℝ), 
    0 < a' ∧ 0 < b' ∧ 0 < c' ∧ 0 < d' ∧ 
    0 < e' ∧ 0 < f' ∧ 0 < g' ∧ 0 < h' ∧
    a' * b' * c' * d' = 8 ∧
    e' * f' * g' * h' = 16 ∧
    a' + b' + c' + d' = e' * f' * g' ∧
    (a'*e')^2 + (b'*f')^2 + (c'*g')^2 + (d'*h')^2 = 64 :=
by sorry

end NUMINAMATH_CALUDE_min_value_problem_l1625_162575


namespace NUMINAMATH_CALUDE_tshirt_sale_problem_l1625_162573

theorem tshirt_sale_problem (sale_duration : ℕ) (black_price white_price : ℚ) 
  (revenue_per_minute : ℚ) (h1 : sale_duration = 25) 
  (h2 : black_price = 30) (h3 : white_price = 25) (h4 : revenue_per_minute = 220) :
  ∃ (total_shirts : ℕ), 
    (total_shirts : ℚ) / 2 * black_price + (total_shirts : ℚ) / 2 * white_price = 
      sale_duration * revenue_per_minute ∧ total_shirts = 200 := by
  sorry

end NUMINAMATH_CALUDE_tshirt_sale_problem_l1625_162573


namespace NUMINAMATH_CALUDE_netGainDifference_l1625_162576

/-- Represents an applicant for a job position -/
structure Applicant where
  salary : ℕ
  revenue : ℕ
  trainingMonths : ℕ
  trainingCostPerMonth : ℕ
  hiringBonusPercent : ℕ

/-- Calculates the net gain for the company from an applicant -/
def netGain (a : Applicant) : ℕ :=
  a.revenue - a.salary - (a.trainingMonths * a.trainingCostPerMonth) - (a.salary * a.hiringBonusPercent / 100)

/-- The first applicant's details -/
def applicant1 : Applicant :=
  { salary := 42000
    revenue := 93000
    trainingMonths := 3
    trainingCostPerMonth := 1200
    hiringBonusPercent := 0 }

/-- The second applicant's details -/
def applicant2 : Applicant :=
  { salary := 45000
    revenue := 92000
    trainingMonths := 0
    trainingCostPerMonth := 0
    hiringBonusPercent := 1 }

/-- Theorem stating the difference in net gain between the two applicants -/
theorem netGainDifference : netGain applicant1 - netGain applicant2 = 850 := by
  sorry

end NUMINAMATH_CALUDE_netGainDifference_l1625_162576


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1625_162534

/-- Given a geometric sequence {a_n} where a_1 = 3 and a_4 = 24, 
    prove that a_3 + a_4 + a_5 = 84 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  (∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q) →  -- geometric sequence condition
  a 1 = 3 →                                  -- a_1 = 3
  a 4 = 24 →                                 -- a_4 = 24
  a 3 + a 4 + a 5 = 84 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1625_162534


namespace NUMINAMATH_CALUDE_remainder_problem_l1625_162579

theorem remainder_problem (k : ℕ) 
  (h1 : k > 0) 
  (h2 : k % 5 = 2) 
  (h3 : k % 6 = 5) 
  (h4 : k < 42) : 
  k % 7 = 3 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l1625_162579


namespace NUMINAMATH_CALUDE_ribbon_length_difference_equals_side_length_l1625_162566

/-- Represents the dimensions of a box in centimeters -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the ribbon length for the first wrapping method -/
def ribbonLength1 (box : BoxDimensions) (bowLength : ℝ) : ℝ :=
  2 * box.length + 2 * box.width + 4 * box.height + bowLength

/-- Calculates the ribbon length for the second wrapping method -/
def ribbonLength2 (box : BoxDimensions) (bowLength : ℝ) : ℝ :=
  2 * box.length + 4 * box.width + 2 * box.height + bowLength

/-- Theorem stating that the difference in ribbon lengths equals one side of the box -/
theorem ribbon_length_difference_equals_side_length
  (box : BoxDimensions)
  (bowLength : ℝ)
  (h1 : box.length = 22)
  (h2 : box.width = 22)
  (h3 : box.height = 11)
  (h4 : bowLength = 24) :
  ribbonLength2 box bowLength - ribbonLength1 box bowLength = box.length := by
  sorry

end NUMINAMATH_CALUDE_ribbon_length_difference_equals_side_length_l1625_162566


namespace NUMINAMATH_CALUDE_rectangle_with_hole_to_square_l1625_162590

/-- Represents a rectangle with a hole -/
structure RectangleWithHole where
  width : ℝ
  height : ℝ
  hole_width : ℝ
  hole_height : ℝ

/-- Calculates the usable area of a rectangle with a hole -/
def usable_area (r : RectangleWithHole) : ℝ :=
  r.width * r.height - r.hole_width * r.hole_height

/-- Theorem: A 9x12 rectangle with a 1x8 hole can be cut into two equal parts that form a 10x10 square -/
theorem rectangle_with_hole_to_square :
  ∃ (r : RectangleWithHole),
    r.width = 9 ∧
    r.height = 12 ∧
    r.hole_width = 1 ∧
    r.hole_height = 8 ∧
    usable_area r = 100 ∧
    ∃ (side_length : ℝ),
      side_length * side_length = usable_area r ∧
      side_length = 10 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_with_hole_to_square_l1625_162590


namespace NUMINAMATH_CALUDE_mikes_payment_l1625_162597

/-- Calculates Mike's out-of-pocket payment for medical procedures -/
theorem mikes_payment (xray_cost : ℝ) (mri_multiplier : ℝ) (insurance_coverage_percent : ℝ) : 
  xray_cost = 250 →
  mri_multiplier = 3 →
  insurance_coverage_percent = 80 →
  let total_cost := xray_cost + mri_multiplier * xray_cost
  let insurance_coverage := (insurance_coverage_percent / 100) * total_cost
  total_cost - insurance_coverage = 200 := by
sorry


end NUMINAMATH_CALUDE_mikes_payment_l1625_162597


namespace NUMINAMATH_CALUDE_x_squared_minus_y_squared_l1625_162571

theorem x_squared_minus_y_squared (x y : ℝ) 
  (sum_eq : x + y = 20) 
  (diff_eq : x - y = 4) : 
  x^2 - y^2 = 80 := by
sorry

end NUMINAMATH_CALUDE_x_squared_minus_y_squared_l1625_162571


namespace NUMINAMATH_CALUDE_stratified_sampling_second_year_count_l1625_162523

theorem stratified_sampling_second_year_count 
  (total_students : ℕ) 
  (first_year_students : ℕ) 
  (second_year_students : ℕ) 
  (first_year_sampled : ℕ) 
  (h1 : total_students = first_year_students + second_year_students)
  (h2 : total_students = 70)
  (h3 : first_year_students = 30)
  (h4 : second_year_students = 40)
  (h5 : first_year_sampled = 9) :
  (first_year_sampled : ℚ) / first_year_students * second_year_students = 12 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_second_year_count_l1625_162523


namespace NUMINAMATH_CALUDE_landscaping_equation_l1625_162513

-- Define the variables and constants
def total_area : ℝ := 180
def original_workers : ℕ := 6
def additional_workers : ℕ := 2
def time_saved : ℝ := 3

-- Define the theorem
theorem landscaping_equation (x : ℝ) :
  (total_area / (original_workers * x)) - (total_area / ((original_workers + additional_workers) * x)) = time_saved :=
by sorry

end NUMINAMATH_CALUDE_landscaping_equation_l1625_162513


namespace NUMINAMATH_CALUDE_closest_integer_to_sqrt17_minus_1_l1625_162509

theorem closest_integer_to_sqrt17_minus_1 :
  ∃ (n : ℤ), (4 < Real.sqrt 17 ∧ Real.sqrt 17 < 4.5) →
  (∀ (m : ℤ), |Real.sqrt 17 - 1 - n| ≤ |Real.sqrt 17 - 1 - m|) ∧ n = 3 :=
sorry

end NUMINAMATH_CALUDE_closest_integer_to_sqrt17_minus_1_l1625_162509


namespace NUMINAMATH_CALUDE_banana_distribution_l1625_162596

theorem banana_distribution (total : Nat) (friends : Nat) (bananas_per_friend : Nat) :
  total = 36 → friends = 5 → bananas_per_friend = 7 →
  total / friends = bananas_per_friend :=
by sorry

end NUMINAMATH_CALUDE_banana_distribution_l1625_162596


namespace NUMINAMATH_CALUDE_race_speed_ratio_l1625_162535

/-- Given two racers a and b, where a's speed is some multiple of b's speed,
    and a gives b a 0.2 part of the race length as a head start resulting in a dead heat,
    prove that the ratio of a's speed to b's speed is 5:4 -/
theorem race_speed_ratio (L : ℝ) (v_a v_b : ℝ) (h1 : v_a > 0) (h2 : v_b > 0) 
    (h3 : ∃ k : ℝ, v_a = k * v_b) 
    (h4 : L / v_a = (0.8 * L) / v_b) : 
  v_a / v_b = 5 / 4 := by
sorry

end NUMINAMATH_CALUDE_race_speed_ratio_l1625_162535


namespace NUMINAMATH_CALUDE_expand_and_simplify_l1625_162546

theorem expand_and_simplify (y : ℝ) : -3 * (y - 4) * (y + 9) = -3 * y^2 - 15 * y + 108 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l1625_162546


namespace NUMINAMATH_CALUDE_initial_blocks_l1625_162536

theorem initial_blocks (initial : ℕ) (added : ℕ) (total : ℕ) : 
  added = 9 → total = 95 → initial + added = total → initial = 86 := by
  sorry

end NUMINAMATH_CALUDE_initial_blocks_l1625_162536
