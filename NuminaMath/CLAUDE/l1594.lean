import Mathlib

namespace NUMINAMATH_CALUDE_simplify_expression_l1594_159488

theorem simplify_expression (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h1 : a^4 + b^4 = a + b) (h2 : a^2 + b^2 = 2) :
  a^2 / b^2 + b^2 / a^2 - 1 / (a^2 * b^2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1594_159488


namespace NUMINAMATH_CALUDE_tangent_circle_equation_l1594_159478

-- Define the given circle C
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + (y + 1)^2 = 4

-- Define the point (4, -1) on circle C
def point_on_C : Prop := circle_C 4 (-1)

-- Define the new circle with center (a, b) and radius 1
def new_circle (a b x y : ℝ) : Prop := (x - a)^2 + (y - b)^2 = 1

-- Define the tangency condition
def is_tangent (a b : ℝ) : Prop := 
  ∃ (x y : ℝ), circle_C x y ∧ new_circle a b x y

-- The theorem to prove
theorem tangent_circle_equation :
  point_on_C →
  is_tangent 5 (-1) ∨ is_tangent 3 (-1) :=
sorry

end NUMINAMATH_CALUDE_tangent_circle_equation_l1594_159478


namespace NUMINAMATH_CALUDE_salesman_pear_sales_l1594_159427

/-- Represents the amount of pears sold by a salesman -/
structure PearSales where
  morning : ℕ
  afternoon : ℕ

/-- The total amount of pears sold in a day -/
def total_sales (sales : PearSales) : ℕ :=
  sales.morning + sales.afternoon

/-- Theorem stating the total sales of pears given the conditions -/
theorem salesman_pear_sales :
  ∃ (sales : PearSales),
    sales.afternoon = 340 ∧
    sales.afternoon = 2 * sales.morning ∧
    total_sales sales = 510 :=
by
  sorry

end NUMINAMATH_CALUDE_salesman_pear_sales_l1594_159427


namespace NUMINAMATH_CALUDE_december_november_difference_l1594_159405

def october_visitors : ℕ := 100

def november_visitors : ℕ := (october_visitors * 115) / 100

def total_visitors : ℕ := 345

theorem december_november_difference :
  ∃ (december_visitors : ℕ),
    december_visitors > november_visitors ∧
    october_visitors + november_visitors + december_visitors = total_visitors ∧
    december_visitors - november_visitors = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_december_november_difference_l1594_159405


namespace NUMINAMATH_CALUDE_comic_books_total_l1594_159450

theorem comic_books_total (jake_books : ℕ) (brother_difference : ℕ) : 
  jake_books = 36 → brother_difference = 15 → 
  jake_books + (jake_books + brother_difference) = 87 := by
  sorry

end NUMINAMATH_CALUDE_comic_books_total_l1594_159450


namespace NUMINAMATH_CALUDE_find_b_value_l1594_159446

/-- The cube of a and the fourth root of b vary inversely -/
def inverse_relation (a b : ℝ) : Prop := ∃ k : ℝ, a^3 * b^(1/4) = k

theorem find_b_value (a b : ℝ) :
  inverse_relation a b →
  (3: ℝ)^3 * (256 : ℝ)^(1/4) = a^3 * b^(1/4) →
  a * b = 81 →
  b = 16 := by
sorry

end NUMINAMATH_CALUDE_find_b_value_l1594_159446


namespace NUMINAMATH_CALUDE_bathroom_kitchen_bulbs_l1594_159474

theorem bathroom_kitchen_bulbs 
  (total_packs : ℕ) 
  (bulbs_per_pack : ℕ) 
  (bedroom_bulbs : ℕ) 
  (basement_bulbs : ℕ) 
  (h1 : total_packs = 6) 
  (h2 : bulbs_per_pack = 2) 
  (h3 : bedroom_bulbs = 2) 
  (h4 : basement_bulbs = 4) :
  total_packs * bulbs_per_pack - (bedroom_bulbs + basement_bulbs + basement_bulbs / 2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_bathroom_kitchen_bulbs_l1594_159474


namespace NUMINAMATH_CALUDE_valid_representation_characterization_l1594_159493

def is_valid_representation (n : ℕ) : Prop :=
  ∃ (a b : ℕ), 
    a > 1 ∧ 
    a ∣ n ∧ 
    (∀ d : ℕ, d > 1 → d ∣ n → d ≥ a) ∧
    b ∣ n ∧
    n = a^2 + b^2

theorem valid_representation_characterization :
  ∀ n : ℕ, is_valid_representation n ↔ (n = 8 ∨ n = 20) :=
sorry

end NUMINAMATH_CALUDE_valid_representation_characterization_l1594_159493


namespace NUMINAMATH_CALUDE_range_of_a_l1594_159402

theorem range_of_a (a : ℝ) : 
  (∀ x y : ℝ, x > 0 → y > 0 → x + y + 4 = 2*x*y → 
    x^2 + 2*x*y + y^2 - a*x - a*y + 1 ≥ 0) → 
  a ≤ 17/4 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l1594_159402


namespace NUMINAMATH_CALUDE_repeating_digits_divisible_by_11_l1594_159485

/-- A function that generates a 9-digit number by repeating the first three digits three times -/
def repeatingDigits (a b c : ℕ) : ℕ :=
  100000000 * a + 10000000 * b + 1000000 * c +
  100000 * a + 10000 * b + 1000 * c +
  100 * a + 10 * b + c

/-- Theorem stating that any 9-digit number formed by repeating the first three digits three times is divisible by 11 -/
theorem repeating_digits_divisible_by_11 (a b c : ℕ) (h : 0 < a ∧ a < 10 ∧ b < 10 ∧ c < 10) :
  11 ∣ repeatingDigits a b c := by
  sorry


end NUMINAMATH_CALUDE_repeating_digits_divisible_by_11_l1594_159485


namespace NUMINAMATH_CALUDE_equal_utility_at_two_l1594_159416

/-- Utility function -/
def utility (swimming : ℝ) (coding : ℝ) : ℝ := 2 * swimming * coding + 1

/-- Saturday's utility -/
def saturday_utility (t : ℝ) : ℝ := utility t (10 - 2*t)

/-- Sunday's utility -/
def sunday_utility (t : ℝ) : ℝ := utility (4 - t) (2*t + 2)

/-- Theorem: The value of t that results in equal utility for both days is 2 -/
theorem equal_utility_at_two :
  ∃ t : ℝ, saturday_utility t = sunday_utility t ∧ t = 2 := by
sorry

end NUMINAMATH_CALUDE_equal_utility_at_two_l1594_159416


namespace NUMINAMATH_CALUDE_no_solution_to_double_inequality_l1594_159400

theorem no_solution_to_double_inequality :
  ¬∃ (x : ℝ), (3 * x + 2 < (x + 2)^2) ∧ ((x + 2)^2 < 5 * x + 1) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_to_double_inequality_l1594_159400


namespace NUMINAMATH_CALUDE_custom_op_two_three_l1594_159457

-- Define the custom operation
def customOp (x y : ℕ) : ℕ := x + y^2

-- Theorem statement
theorem custom_op_two_three : customOp 2 3 = 11 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_two_three_l1594_159457


namespace NUMINAMATH_CALUDE_lions_mortality_rate_l1594_159438

/-- The number of lions that die each month in Londolozi -/
def lions_die_per_month : ℕ := 1

/-- The initial number of lions in Londolozi -/
def initial_lions : ℕ := 100

/-- The number of lion cubs born per month in Londolozi -/
def cubs_born_per_month : ℕ := 5

/-- The number of months in a year -/
def months_in_year : ℕ := 12

/-- The number of lions after one year in Londolozi -/
def lions_after_year : ℕ := 148

theorem lions_mortality_rate :
  lions_after_year = initial_lions + (cubs_born_per_month - lions_die_per_month) * months_in_year :=
sorry

end NUMINAMATH_CALUDE_lions_mortality_rate_l1594_159438


namespace NUMINAMATH_CALUDE_investment_rate_proof_l1594_159496

/-- Proves that given the described investment scenario, the initial interest rate is approximately 0.2 -/
theorem investment_rate_proof (initial_investment : ℝ) (years : ℕ) (final_amount : ℝ) : 
  initial_investment = 10000 →
  years = 3 →
  final_amount = 59616 →
  ∃ (r : ℝ), 
    (r ≥ 0) ∧ 
    (r ≤ 1) ∧
    (abs (r - 0.2) < 0.001) ∧
    (final_amount = 3 * initial_investment * (1 + r)^years * 1.15) :=
by sorry

end NUMINAMATH_CALUDE_investment_rate_proof_l1594_159496


namespace NUMINAMATH_CALUDE_student_age_problem_l1594_159410

theorem student_age_problem (total_students : ℕ) (total_avg_age : ℝ)
  (group1_size group2_size group3_size : ℕ) (group1_avg group2_avg group3_avg : ℝ) :
  total_students = 24 →
  total_avg_age = 18 →
  group1_size = 6 →
  group2_size = 10 →
  group3_size = 7 →
  group1_avg = 16 →
  group2_avg = 20 →
  group3_avg = 17 →
  ∃ (last_student_age : ℝ),
    (group1_size * group1_avg + group2_size * group2_avg + group3_size * group3_avg + last_student_age) / total_students = total_avg_age ∧
    last_student_age = 15 :=
by sorry

end NUMINAMATH_CALUDE_student_age_problem_l1594_159410


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l1594_159415

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  h : ∀ n, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (seq.a 1 + seq.a n) / 2

theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (seq.a 4 + seq.a 14 = 2 → S seq 17 = 17) ∧
  (seq.a 11 = 10 → S seq 21 = 210) ∧
  (S seq 11 = 55 → seq.a 6 = 5) ∧
  (S seq 8 = 100 ∧ S seq 16 = 392 → S seq 24 = 876) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l1594_159415


namespace NUMINAMATH_CALUDE_jolene_earnings_180_l1594_159458

/-- Represents Jolene's earnings from various jobs -/
structure JoleneEarnings where
  babysitting_families : ℕ
  babysitting_rate : ℕ
  car_washing_jobs : ℕ
  car_washing_rate : ℕ

/-- Calculates Jolene's total earnings -/
def total_earnings (e : JoleneEarnings) : ℕ :=
  e.babysitting_families * e.babysitting_rate + e.car_washing_jobs * e.car_washing_rate

/-- Theorem stating that Jolene's total earnings are $180 -/
theorem jolene_earnings_180 :
  ∃ (e : JoleneEarnings),
    e.babysitting_families = 4 ∧
    e.babysitting_rate = 30 ∧
    e.car_washing_jobs = 5 ∧
    e.car_washing_rate = 12 ∧
    total_earnings e = 180 := by
  sorry

end NUMINAMATH_CALUDE_jolene_earnings_180_l1594_159458


namespace NUMINAMATH_CALUDE_pandas_minus_lions_l1594_159409

/-- The number of animals in John's zoo --/
structure ZooAnimals where
  snakes : ℕ
  monkeys : ℕ
  lions : ℕ
  pandas : ℕ
  dogs : ℕ

/-- The conditions of John's zoo --/
def validZoo (zoo : ZooAnimals) : Prop :=
  zoo.snakes = 15 ∧
  zoo.monkeys = 2 * zoo.snakes ∧
  zoo.lions = zoo.monkeys - 5 ∧
  zoo.dogs = zoo.pandas / 3 ∧
  zoo.snakes + zoo.monkeys + zoo.lions + zoo.pandas + zoo.dogs = 114

/-- The theorem to prove --/
theorem pandas_minus_lions (zoo : ZooAnimals) (h : validZoo zoo) : 
  zoo.pandas - zoo.lions = 8 := by
  sorry


end NUMINAMATH_CALUDE_pandas_minus_lions_l1594_159409


namespace NUMINAMATH_CALUDE_optimal_height_minimizes_surface_area_l1594_159419

/-- Represents a rectangular box with a lid -/
structure Box where
  x : ℝ  -- Length of one side of the base
  y : ℝ  -- Height of the box

/-- Calculates the volume of the box -/
def volume (b : Box) : ℝ := 2 * b.x^2 * b.y

/-- Calculates the surface area of the box -/
def surfaceArea (b : Box) : ℝ := 4 * b.x^2 + 6 * b.x * b.y

/-- States that the volume of the box is 72 -/
def volumeConstraint (b : Box) : Prop := volume b = 72

/-- Finds the height that minimizes the surface area -/
def optimalHeight : ℝ := 4

theorem optimal_height_minimizes_surface_area :
  ∃ (b : Box), volumeConstraint b ∧
    ∀ (b' : Box), volumeConstraint b' → surfaceArea b ≤ surfaceArea b' ∧
    b.y = optimalHeight := by sorry

end NUMINAMATH_CALUDE_optimal_height_minimizes_surface_area_l1594_159419


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_is_three_l1594_159445

theorem sum_of_a_and_b_is_three (a b : ℝ) (i : ℂ) (h1 : i * i = -1) 
  (h2 : (a + 2 * i) / i = b - i * a) : a + b = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_is_three_l1594_159445


namespace NUMINAMATH_CALUDE_limit_sum_geometric_sequence_l1594_159432

def geometricSequence (n : ℕ) : ℚ := (1/2) * (1/2)^(n-1)

def sumGeometricSequence (n : ℕ) : ℚ := 
  (1/2) * (1 - (1/2)^n) / (1 - 1/2)

theorem limit_sum_geometric_sequence :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |sumGeometricSequence n - 1| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_sum_geometric_sequence_l1594_159432


namespace NUMINAMATH_CALUDE_order_of_logarithmic_fractions_l1594_159460

theorem order_of_logarithmic_fractions :
  let a : ℝ := (Real.exp 1)⁻¹
  let b : ℝ := (Real.log 2) / 2
  let c : ℝ := (Real.log 3) / 3
  b < c ∧ c < a := by sorry

end NUMINAMATH_CALUDE_order_of_logarithmic_fractions_l1594_159460


namespace NUMINAMATH_CALUDE_middle_digit_zero_l1594_159408

theorem middle_digit_zero (N : ℕ) (a b c : ℕ) : 
  (N = 49*a + 7*b + c) →  -- N in base 7
  (N = 81*c + 9*b + a) →  -- N in base 9
  (0 ≤ a ∧ a < 7) →       -- a is a valid digit in base 7
  (0 ≤ b ∧ b < 7) →       -- b is a valid digit in base 7
  (0 ≤ c ∧ c < 7) →       -- c is a valid digit in base 7
  (b = 0) :=              -- middle digit is 0
by sorry

end NUMINAMATH_CALUDE_middle_digit_zero_l1594_159408


namespace NUMINAMATH_CALUDE_hcf_problem_l1594_159433

theorem hcf_problem (a b : ℕ+) (h1 : a * b = 82500) (h2 : Nat.lcm a b = 1500) :
  Nat.gcd a b = 55 := by
  sorry

end NUMINAMATH_CALUDE_hcf_problem_l1594_159433


namespace NUMINAMATH_CALUDE_complex_power_four_equals_negative_four_l1594_159412

theorem complex_power_four_equals_negative_four : 
  (1 + (1 / Complex.I)) ^ 4 = (-4 : ℂ) := by sorry

end NUMINAMATH_CALUDE_complex_power_four_equals_negative_four_l1594_159412


namespace NUMINAMATH_CALUDE_inequality_equivalence_l1594_159480

theorem inequality_equivalence (x : ℝ) : 
  3 * x - 6 > 12 - 2 * x + x^2 ↔ -1 < x ∧ x < 6 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l1594_159480


namespace NUMINAMATH_CALUDE_jungkook_final_ball_count_l1594_159469

-- Define the initial state
def jungkook_red_balls : ℕ := 3
def yoongi_blue_balls : ℕ := 2

-- Define the transfer
def transferred_balls : ℕ := 1

-- Theorem to prove
theorem jungkook_final_ball_count :
  jungkook_red_balls + transferred_balls = 4 := by
  sorry

end NUMINAMATH_CALUDE_jungkook_final_ball_count_l1594_159469


namespace NUMINAMATH_CALUDE_a_plus_b_and_abs_a_minus_b_l1594_159475

theorem a_plus_b_and_abs_a_minus_b (a b : ℝ) 
  (h1 : |a| = 2) 
  (h2 : b^2 = 25) 
  (h3 : a * b < 0) : 
  ((a + b = 3) ∨ (a + b = -3)) ∧ |a - b| = 7 := by
sorry

end NUMINAMATH_CALUDE_a_plus_b_and_abs_a_minus_b_l1594_159475


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainders_l1594_159462

theorem smallest_integer_with_remainders : ∃ N : ℕ, 
  N > 0 ∧
  N % 5 = 2 ∧
  N % 6 = 3 ∧
  N % 7 = 4 ∧
  N % 11 = 9 ∧
  (∀ M : ℕ, M > 0 ∧ M % 5 = 2 ∧ M % 6 = 3 ∧ M % 7 = 4 ∧ M % 11 = 9 → N ≤ M) ∧
  N = 207 := by
sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainders_l1594_159462


namespace NUMINAMATH_CALUDE_total_games_in_our_league_l1594_159476

/-- Represents a sports league with sub-leagues and playoffs -/
structure SportsLeague where
  total_teams : Nat
  num_sub_leagues : Nat
  teams_per_sub_league : Nat
  games_against_each_team : Nat
  teams_advancing : Nat

/-- Calculates the total number of games in the entire season -/
def total_games (league : SportsLeague) : Nat :=
  let sub_league_games := league.num_sub_leagues * (league.teams_per_sub_league * (league.teams_per_sub_league - 1) / 2 * league.games_against_each_team)
  let playoff_teams := league.num_sub_leagues * league.teams_advancing
  let playoff_games := playoff_teams * (playoff_teams - 1) / 2
  sub_league_games + playoff_games

/-- The specific league configuration -/
def our_league : SportsLeague :=
  { total_teams := 100
  , num_sub_leagues := 5
  , teams_per_sub_league := 20
  , games_against_each_team := 6
  , teams_advancing := 4 }

/-- Theorem stating that the total number of games in our league is 5890 -/
theorem total_games_in_our_league : total_games our_league = 5890 := by
  sorry

end NUMINAMATH_CALUDE_total_games_in_our_league_l1594_159476


namespace NUMINAMATH_CALUDE_absolute_value_equation_l1594_159473

theorem absolute_value_equation (a : ℝ) : |a - 1| = 2 → a = 3 ∨ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_l1594_159473


namespace NUMINAMATH_CALUDE_hilton_lost_marbles_l1594_159442

/-- Proves the number of marbles Hilton lost given the initial and final conditions -/
theorem hilton_lost_marbles (initial : ℕ) (found : ℕ) (final : ℕ) : 
  initial = 26 → found = 6 → final = 42 → 
  ∃ (lost : ℕ), lost = 10 ∧ final = initial + found - lost + 2 * lost := by
  sorry

end NUMINAMATH_CALUDE_hilton_lost_marbles_l1594_159442


namespace NUMINAMATH_CALUDE_max_value_part1_l1594_159481

theorem max_value_part1 (x : ℝ) (h1 : 0 < x) (h2 : x < 1/2) :
  (1/2) * x * (1 - 2*x) ≤ 1/16 := by sorry

end NUMINAMATH_CALUDE_max_value_part1_l1594_159481


namespace NUMINAMATH_CALUDE_factor_implies_c_value_l1594_159463

theorem factor_implies_c_value (c : ℚ) : 
  (∀ x : ℚ, (x - 3) ∣ (c * x^3 - 6 * x^2 - c * x + 10)) → c = 11/6 := by
  sorry

end NUMINAMATH_CALUDE_factor_implies_c_value_l1594_159463


namespace NUMINAMATH_CALUDE_inequality_always_true_iff_a_in_range_l1594_159498

theorem inequality_always_true_iff_a_in_range (a : ℝ) :
  (∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0) ↔ a ∈ Set.Ioc (-2) 2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_always_true_iff_a_in_range_l1594_159498


namespace NUMINAMATH_CALUDE_absolute_value_greater_than_one_l1594_159453

theorem absolute_value_greater_than_one (a b : ℝ) 
  (h1 : b * (a + b + 1) < 0) 
  (h2 : b * (a + b - 1) < 0) : 
  |a| > 1 := by
sorry

end NUMINAMATH_CALUDE_absolute_value_greater_than_one_l1594_159453


namespace NUMINAMATH_CALUDE_football_team_throwers_l1594_159430

/-- Represents the number of throwers on a football team given specific conditions -/
def number_of_throwers (total_players : ℕ) (right_handed : ℕ) : ℕ :=
  total_players - (3 * (total_players - (right_handed - (total_players - right_handed))) / 2)

/-- Theorem stating that under given conditions, there are 28 throwers on the team -/
theorem football_team_throwers :
  let total_players : ℕ := 70
  let right_handed : ℕ := 56
  number_of_throwers total_players right_handed = 28 := by
  sorry

end NUMINAMATH_CALUDE_football_team_throwers_l1594_159430


namespace NUMINAMATH_CALUDE_inverse_32_mod_97_l1594_159459

theorem inverse_32_mod_97 (h : (2⁻¹ : ZMod 97) = 49) : (32⁻¹ : ZMod 97) = 49 := by
  sorry

end NUMINAMATH_CALUDE_inverse_32_mod_97_l1594_159459


namespace NUMINAMATH_CALUDE_dog_fruit_problem_l1594_159477

/-- The number of bonnies eaten by the third dog -/
def B : ℕ := sorry

/-- The number of blueberries eaten by the second dog -/
def blueberries : ℕ := sorry

/-- The number of apples eaten by the first dog -/
def apples : ℕ := sorry

/-- The total number of fruits eaten by all three dogs -/
def total_fruits : ℕ := 240

theorem dog_fruit_problem :
  (blueberries = (3 * B) / 4) →
  (apples = 3 * blueberries) →
  (B + blueberries + apples = total_fruits) →
  B = 60 := by sorry

end NUMINAMATH_CALUDE_dog_fruit_problem_l1594_159477


namespace NUMINAMATH_CALUDE_regular_polygon_interior_angle_l1594_159406

theorem regular_polygon_interior_angle (n : ℕ) : 
  (n ≥ 3) → (((n - 2) * 180 : ℝ) / n = 144) → n = 10 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_interior_angle_l1594_159406


namespace NUMINAMATH_CALUDE_cookie_problem_l1594_159447

theorem cookie_problem (total_cookies : ℕ) (total_nuts : ℕ) (nuts_per_cookie : ℕ) 
  (h_total_cookies : total_cookies = 60)
  (h_total_nuts : total_nuts = 72)
  (h_nuts_per_cookie : nuts_per_cookie = 2)
  (h_quarter_nuts : (total_cookies / 4 : ℚ) = (total_cookies - (total_nuts / nuts_per_cookie) : ℕ)) :
  (((total_cookies - (total_cookies / 4) - (total_nuts / nuts_per_cookie - total_cookies / 4)) / total_cookies : ℚ) * 100 = 40) := by
  sorry

end NUMINAMATH_CALUDE_cookie_problem_l1594_159447


namespace NUMINAMATH_CALUDE_hatcher_students_l1594_159431

/-- Calculates the total number of students Ms. Hatcher taught -/
def total_students (third_graders : ℕ) : ℕ :=
  let fourth_graders := 2 * third_graders
  let fifth_graders := third_graders / 2
  third_graders + fourth_graders + fifth_graders

/-- Theorem stating that Ms. Hatcher taught 70 students -/
theorem hatcher_students : total_students 20 = 70 := by
  sorry

end NUMINAMATH_CALUDE_hatcher_students_l1594_159431


namespace NUMINAMATH_CALUDE_inequality_proof_l1594_159425

theorem inequality_proof (a b c d e p q : ℝ) 
  (hp_pos : 0 < p) 
  (hq_pos : 0 < q) 
  (ha : p ≤ a ∧ a ≤ q) 
  (hb : p ≤ b ∧ b ≤ q) 
  (hc : p ≤ c ∧ c ≤ q) 
  (hd : p ≤ d ∧ d ≤ q) 
  (he : p ≤ e ∧ e ≤ q) : 
  (a + b + c + d + e) * (1/a + 1/b + 1/c + 1/d + 1/e) ≤ 25 + 6 * (Real.sqrt (p/q) - Real.sqrt (q/p))^2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1594_159425


namespace NUMINAMATH_CALUDE_max_value_constraint_l1594_159441

theorem max_value_constraint (x y z : ℝ) (h : 9 * x^2 + 4 * y^2 + 25 * z^2 = 1) :
  ∃ (M : ℝ), M = Real.sqrt 298 ∧ (8 * x + 3 * y + 15 * z ≤ M) ∧
  ∃ (x₀ y₀ z₀ : ℝ), 9 * x₀^2 + 4 * y₀^2 + 25 * z₀^2 = 1 ∧ 8 * x₀ + 3 * y₀ + 15 * z₀ = M :=
by sorry

end NUMINAMATH_CALUDE_max_value_constraint_l1594_159441


namespace NUMINAMATH_CALUDE_weekly_earnings_l1594_159417

def phone_repair_cost : ℕ := 11
def laptop_repair_cost : ℕ := 15
def computer_repair_cost : ℕ := 18

def phone_repairs : ℕ := 5
def laptop_repairs : ℕ := 2
def computer_repairs : ℕ := 2

def total_earnings : ℕ := phone_repair_cost * phone_repairs + 
                          laptop_repair_cost * laptop_repairs + 
                          computer_repair_cost * computer_repairs

theorem weekly_earnings : total_earnings = 121 := by
  sorry

end NUMINAMATH_CALUDE_weekly_earnings_l1594_159417


namespace NUMINAMATH_CALUDE_collinear_points_q_value_l1594_159490

/-- 
If the points (7, q), (5, 3), and (1, -1) are collinear, then q = 5.
-/
theorem collinear_points_q_value (q : ℝ) : 
  (∃ (t : ℝ), (7 - 1) = t * (5 - 1) ∧ (q + 1) = t * (3 + 1)) → q = 5 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_q_value_l1594_159490


namespace NUMINAMATH_CALUDE_pond_length_l1594_159448

/-- Given a rectangular pond with width 10 meters, depth 8 meters, and volume 1600 cubic meters,
    prove that the length of the pond is 20 meters. -/
theorem pond_length (width : ℝ) (depth : ℝ) (volume : ℝ) (length : ℝ) :
  width = 10 →
  depth = 8 →
  volume = 1600 →
  volume = length * width * depth →
  length = 20 := by
sorry

end NUMINAMATH_CALUDE_pond_length_l1594_159448


namespace NUMINAMATH_CALUDE_multiplication_equalities_l1594_159452

theorem multiplication_equalities : 
  (50 * 6 = 300) ∧ (5 * 60 = 300) ∧ (4 * 300 = 1200) := by
  sorry

end NUMINAMATH_CALUDE_multiplication_equalities_l1594_159452


namespace NUMINAMATH_CALUDE_trout_catch_total_l1594_159479

theorem trout_catch_total (people : ℕ) (individual_share : ℕ) (h1 : people = 2) (h2 : individual_share = 9) :
  people * individual_share = 18 := by
  sorry

end NUMINAMATH_CALUDE_trout_catch_total_l1594_159479


namespace NUMINAMATH_CALUDE_smallest_positive_integer_solution_l1594_159494

theorem smallest_positive_integer_solution : ∃ (x : ℕ), 
  (x > 0) ∧ 
  (|5 * (x : ℤ) - 8| = 47) ∧ 
  (∀ (y : ℕ), y > 0 ∧ |5 * (y : ℤ) - 8| = 47 → x ≤ y) ∧
  (x = 11) := by
sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_solution_l1594_159494


namespace NUMINAMATH_CALUDE_slope_of_line_slope_to_angle_l1594_159403

/-- The slope of the line x + √3 * y - 1 = 0 is -√3/3 -/
theorem slope_of_line (x y : ℝ) : 
  (x + Real.sqrt 3 * y - 1 = 0) → (y = -(1 / Real.sqrt 3) * x + 1 / Real.sqrt 3) :=
by sorry

/-- The slope -√3/3 corresponds to an angle of 150° -/
theorem slope_to_angle (θ : ℝ) :
  Real.tan θ = -(Real.sqrt 3 / 3) → θ = 150 * (Real.pi / 180) :=
by sorry

end NUMINAMATH_CALUDE_slope_of_line_slope_to_angle_l1594_159403


namespace NUMINAMATH_CALUDE_lose_sector_area_l1594_159486

theorem lose_sector_area (radius : ℝ) (win_probability : ℝ) 
  (h1 : radius = 12)
  (h2 : win_probability = 1/3) : 
  (1 - win_probability) * π * radius^2 = 96 * π := by
  sorry

end NUMINAMATH_CALUDE_lose_sector_area_l1594_159486


namespace NUMINAMATH_CALUDE_total_candies_l1594_159423

theorem total_candies (linda_candies chloe_candies olivia_candies : ℕ)
  (h1 : linda_candies = 34)
  (h2 : chloe_candies = 28)
  (h3 : olivia_candies = 43) :
  linda_candies + chloe_candies + olivia_candies = 105 := by
  sorry

end NUMINAMATH_CALUDE_total_candies_l1594_159423


namespace NUMINAMATH_CALUDE_smallest_base_perfect_square_l1594_159404

theorem smallest_base_perfect_square : 
  ∀ b : ℕ, b > 3 → (∃ n : ℕ, 2 * b + 3 = n^2) → b ≥ 11 :=
sorry

end NUMINAMATH_CALUDE_smallest_base_perfect_square_l1594_159404


namespace NUMINAMATH_CALUDE_rectangular_solid_depth_l1594_159443

/-- The surface area of a rectangular solid given its length, width, and height. -/
def surfaceArea (l w h : ℝ) : ℝ := 2 * l * w + 2 * l * h + 2 * w * h

/-- Theorem: A rectangular solid with length 10, width 9, and surface area 408 has depth 6. -/
theorem rectangular_solid_depth :
  ∃ (h : ℝ), h > 0 ∧ surfaceArea 10 9 h = 408 → h = 6 := by
sorry

end NUMINAMATH_CALUDE_rectangular_solid_depth_l1594_159443


namespace NUMINAMATH_CALUDE_difference_of_squares_l1594_159465

theorem difference_of_squares (x y : ℝ) : x^2 - y^2 = (x + y) * (x - y) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l1594_159465


namespace NUMINAMATH_CALUDE_lcm_problem_l1594_159424

theorem lcm_problem (a b : ℕ+) (h1 : Nat.gcd a b = 84) (h2 : b = 4 * a) (h3 : b = 84) :
  Nat.lcm a b = 21 := by
  sorry

end NUMINAMATH_CALUDE_lcm_problem_l1594_159424


namespace NUMINAMATH_CALUDE_even_odd_sum_difference_l1594_159428

theorem even_odd_sum_difference : 
  (Finset.sum (Finset.range 100) (fun i => 2 * (i + 1))) - 
  (Finset.sum (Finset.range 100) (fun i => 2 * i + 1)) = 100 := by
  sorry

end NUMINAMATH_CALUDE_even_odd_sum_difference_l1594_159428


namespace NUMINAMATH_CALUDE_min_value_expression_equality_condition_l1594_159461

theorem min_value_expression (x : ℝ) : 
  Real.sqrt (x^2 + (2 - x)^2) + Real.sqrt ((2 - x)^2 + (2 + x)^2) ≥ 2 * Real.sqrt 5 :=
by sorry

theorem equality_condition : 
  ∃ x : ℝ, Real.sqrt (x^2 + (2 - x)^2) + Real.sqrt ((2 - x)^2 + (2 + x)^2) = 2 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_equality_condition_l1594_159461


namespace NUMINAMATH_CALUDE_square_sum_identity_l1594_159444

theorem square_sum_identity (x : ℝ) : (x + 2)^2 + 2*(x + 2)*(4 - x) + (4 - x)^2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_identity_l1594_159444


namespace NUMINAMATH_CALUDE_factorial_ratio_twelve_eleven_l1594_159437

theorem factorial_ratio_twelve_eleven : Nat.factorial 12 / Nat.factorial 11 = 12 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_twelve_eleven_l1594_159437


namespace NUMINAMATH_CALUDE_hawking_implications_l1594_159489

-- Define the philosophical implications
def unity_of_world_materiality : Prop := true
def thought_existence_identical : Prop := true

-- Define Hawking's statement
def hawking_statement : Prop := true

-- Theorem to prove
theorem hawking_implications :
  hawking_statement → unity_of_world_materiality ∧ thought_existence_identical :=
by
  sorry

end NUMINAMATH_CALUDE_hawking_implications_l1594_159489


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l1594_159482

open Set Real

-- Define set A
def A : Set ℝ := {x | ∃ y, y = Real.sqrt (x - 1)}

-- Define set B
def B : Set ℝ := {x | 2 * x - x^2 > 0}

-- State the theorem
theorem complement_A_intersect_B : (𝒰 \ A) ∩ B = Ioo 0 1 := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l1594_159482


namespace NUMINAMATH_CALUDE_cubic_sum_from_elementary_symmetric_polynomials_l1594_159487

theorem cubic_sum_from_elementary_symmetric_polynomials (p q r : ℝ) 
  (h1 : p + q + r = 7)
  (h2 : p * q + p * r + q * r = 8)
  (h3 : p * q * r = -15) :
  p^3 + q^3 + r^3 = 151 := by
sorry

end NUMINAMATH_CALUDE_cubic_sum_from_elementary_symmetric_polynomials_l1594_159487


namespace NUMINAMATH_CALUDE_red_markers_count_l1594_159491

theorem red_markers_count (total_markers blue_markers : ℕ) 
  (h1 : total_markers = 105)
  (h2 : blue_markers = 64) :
  total_markers - blue_markers = 41 := by
sorry

end NUMINAMATH_CALUDE_red_markers_count_l1594_159491


namespace NUMINAMATH_CALUDE_javier_first_throw_l1594_159422

/-- Represents the distance of Javier's second throw before adjustments -/
def second_throw : ℝ := sorry

/-- Calculates the adjusted distance of a throw -/
def adjusted_distance (base : ℝ) (wind_reduction : ℝ) (incline : ℝ) : ℝ :=
  base * (1 - wind_reduction) - incline

theorem javier_first_throw :
  let first_throw := 2 * second_throw
  let third_throw := 2 * first_throw
  adjusted_distance first_throw 0.05 2 +
  adjusted_distance second_throw 0.08 4 +
  adjusted_distance third_throw 0 1 = 1050 →
  first_throw = 310 := by sorry

end NUMINAMATH_CALUDE_javier_first_throw_l1594_159422


namespace NUMINAMATH_CALUDE_factorial_sum_equality_l1594_159434

theorem factorial_sum_equality : 6 * Nat.factorial 6 + 5 * Nat.factorial 5 + Nat.factorial 5 = 5040 := by
  sorry

end NUMINAMATH_CALUDE_factorial_sum_equality_l1594_159434


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l1594_159413

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5}

-- Define set A
def A : Set Nat := {3, 4, 5}

-- Theorem statement
theorem complement_of_A_in_U :
  {x ∈ U | x ∉ A} = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l1594_159413


namespace NUMINAMATH_CALUDE_line_slope_problem_l1594_159407

theorem line_slope_problem (k : ℚ) : 
  (∃ line : ℝ → ℝ, 
    (line (-1) = -4) ∧ 
    (line 3 = k) ∧ 
    (∀ x y : ℝ, x ≠ -1 → (line y - line x) / (y - x) = k)) → 
  k = 4/3 := by
sorry

end NUMINAMATH_CALUDE_line_slope_problem_l1594_159407


namespace NUMINAMATH_CALUDE_juice_box_days_l1594_159499

theorem juice_box_days (num_children : ℕ) (school_weeks : ℕ) (total_juice_boxes : ℕ) :
  num_children = 3 →
  school_weeks = 25 →
  total_juice_boxes = 375 →
  (total_juice_boxes / (num_children * school_weeks) : ℚ) = 5 := by
  sorry

end NUMINAMATH_CALUDE_juice_box_days_l1594_159499


namespace NUMINAMATH_CALUDE_min_translation_for_symmetry_l1594_159484

theorem min_translation_for_symmetry (f : ℝ → ℝ) (h : ∀ x, f x = Real.sin x + Real.cos x) :
  ∃ φ : ℝ, φ > 0 ∧
    (∀ x, f (x - φ) = -f (-x + φ)) ∧
    (∀ ψ, ψ > 0 ∧ (∀ x, f (x - ψ) = -f (-x + ψ)) → φ ≤ ψ) ∧
    φ = Real.pi / 4 :=
by sorry

end NUMINAMATH_CALUDE_min_translation_for_symmetry_l1594_159484


namespace NUMINAMATH_CALUDE_total_share_calculation_l1594_159497

/-- Given three shares x, y, and z, where x is 25% more than y, y is 20% more than z,
    and z is 100, prove that the total amount shared is 370. -/
theorem total_share_calculation (x y z : ℚ) : 
  x = 1.25 * y ∧ y = 1.2 * z ∧ z = 100 → x + y + z = 370 := by
  sorry

end NUMINAMATH_CALUDE_total_share_calculation_l1594_159497


namespace NUMINAMATH_CALUDE_first_number_proof_l1594_159492

theorem first_number_proof (y : ℝ) (h1 : y = 48) (h2 : ∃ x : ℝ, x + (1/4) * y = 27) : 
  ∃ x : ℝ, x + (1/4) * y = 27 ∧ x = 15 := by
  sorry

end NUMINAMATH_CALUDE_first_number_proof_l1594_159492


namespace NUMINAMATH_CALUDE_combined_standard_deviation_l1594_159466

/-- Given two groups of numbers with known means and variances, 
    calculate the standard deviation of the combined set. -/
theorem combined_standard_deviation 
  (n₁ n₂ : ℕ) 
  (mean₁ mean₂ : ℝ) 
  (var₁ var₂ : ℝ) :
  n₁ = 10 →
  n₂ = 10 →
  mean₁ = 50 →
  mean₂ = 40 →
  var₁ = 33 →
  var₂ = 45 →
  let n_total := n₁ + n₂
  let var_total := (n₁ * var₁ + n₂ * var₂) / n_total + 
                   (n₁ * n₂ : ℝ) / (n_total ^ 2 : ℝ) * (mean₁ - mean₂) ^ 2
  Real.sqrt var_total = 8 := by
  sorry

#check combined_standard_deviation

end NUMINAMATH_CALUDE_combined_standard_deviation_l1594_159466


namespace NUMINAMATH_CALUDE_solve_for_x_l1594_159429

def star_op (x y : ℤ) : ℤ := x * y - 2 * (x + y)

theorem solve_for_x : ∃ x : ℤ, (∀ y : ℤ, star_op x y = x * y - 2 * (x + y)) ∧ star_op x (-3) = 1 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_x_l1594_159429


namespace NUMINAMATH_CALUDE_horner_method_v4_l1594_159464

def f (x : ℝ) : ℝ := 3*x^6 + 5*x^5 + 6*x^4 + 20*x^3 - 8*x^2 + 35*x + 12

def horner_v4 (a₆ a₅ a₄ a₃ a₂ a₁ a₀ x : ℝ) : ℝ :=
  let v₀ := a₆
  let v₁ := v₀ * x + a₅
  let v₂ := v₁ * x + a₄
  let v₃ := v₂ * x + a₃
  v₃ * x + a₂

theorem horner_method_v4 :
  horner_v4 3 5 6 20 (-8) 35 12 (-2) = -16 :=
by sorry

end NUMINAMATH_CALUDE_horner_method_v4_l1594_159464


namespace NUMINAMATH_CALUDE_hyperbola_m_range_l1594_159411

def is_hyperbola (m : ℝ) : Prop :=
  ∀ x y : ℝ, x^2 / m - y^2 / (m + 1) = 1 → 
    (m > 0 ∧ m + 1 > 0) ∨ (m < 0 ∧ m + 1 < 0)

theorem hyperbola_m_range :
  {m : ℝ | is_hyperbola m} = {m | m < -1 ∨ m > 0} := by sorry

end NUMINAMATH_CALUDE_hyperbola_m_range_l1594_159411


namespace NUMINAMATH_CALUDE_f_is_odd_l1594_159414

def f (p : ℝ) (x : ℝ) : ℝ := x * |x| + p * x

theorem f_is_odd (p : ℝ) : 
  ∀ x : ℝ, f p (-x) = -(f p x) := by
sorry

end NUMINAMATH_CALUDE_f_is_odd_l1594_159414


namespace NUMINAMATH_CALUDE_magician_numbers_l1594_159451

theorem magician_numbers : ∃! (a b : ℕ), 
  a * b = 2280 ∧ 
  a + b < 100 ∧ 
  a + b > 9 ∧ 
  Odd (a + b) ∧ 
  a = 40 ∧ 
  b = 57 := by sorry

end NUMINAMATH_CALUDE_magician_numbers_l1594_159451


namespace NUMINAMATH_CALUDE_tv_price_change_l1594_159418

theorem tv_price_change (P : ℝ) (x : ℝ) : 
  P * (1 - x / 100) * 1.3 = P * 1.17 → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_tv_price_change_l1594_159418


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l1594_159456

theorem algebraic_expression_value (m n : ℝ) 
  (h1 : m - n = -2) 
  (h2 : m * n = 3) : 
  -m^3*n + 2*m^2*n^2 - m*n^3 = -12 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l1594_159456


namespace NUMINAMATH_CALUDE_trailing_zeros_100_factorial_l1594_159426

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25)

/-- Theorem: The number of trailing zeros in 100! is 24 -/
theorem trailing_zeros_100_factorial :
  trailingZeros 100 = 24 := by
  sorry

end NUMINAMATH_CALUDE_trailing_zeros_100_factorial_l1594_159426


namespace NUMINAMATH_CALUDE_number_puzzle_l1594_159454

theorem number_puzzle : ∃ x : ℝ, 35 + 3 * x = 50 ∧ x - 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l1594_159454


namespace NUMINAMATH_CALUDE_smallest_sum_abc_l1594_159468

theorem smallest_sum_abc (a b c : ℕ+) : 
  (∃ x : ℝ, (Real.sin x)^2 + (Real.sin (3*x))^2 + (Real.sin (5*x))^2 + (Real.sin (7*x))^2 = 2.5 ∧
             Real.cos (a.val * x) * Real.cos (b.val * x) * Real.cos (c.val * x) = 0) →
  (∀ a' b' c' : ℕ+, 
    (∃ x : ℝ, (Real.sin x)^2 + (Real.sin (3*x))^2 + (Real.sin (5*x))^2 + (Real.sin (7*x))^2 = 2.5 ∧
               Real.cos (a'.val * x) * Real.cos (b'.val * x) * Real.cos (c'.val * x) = 0) →
    a'.val + b'.val + c'.val ≥ a.val + b.val + c.val) →
  a.val + b.val + c.val = 14 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_abc_l1594_159468


namespace NUMINAMATH_CALUDE_derivative_symmetry_l1594_159455

-- Define the function f
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^4 + b * x^2 + c

-- Define the derivative of f
def f' (a b : ℝ) (x : ℝ) : ℝ := 4 * a * x^3 + 2 * b * x

-- Theorem statement
theorem derivative_symmetry (a b c : ℝ) :
  f' a b 1 = 2 → f' a b (-1) = -2 := by
  sorry

end NUMINAMATH_CALUDE_derivative_symmetry_l1594_159455


namespace NUMINAMATH_CALUDE_orange_juice_serving_volume_l1594_159470

/-- Proves that the volume of each serving of orange juice is 6 ounces given the specified conditions. -/
theorem orange_juice_serving_volume
  (concentrate_cans : ℕ)
  (concentrate_oz_per_can : ℕ)
  (water_cans_per_concentrate : ℕ)
  (total_servings : ℕ)
  (h1 : concentrate_cans = 60)
  (h2 : concentrate_oz_per_can = 5)
  (h3 : water_cans_per_concentrate = 3)
  (h4 : total_servings = 200) :
  (concentrate_cans * concentrate_oz_per_can * (water_cans_per_concentrate + 1)) / total_servings = 6 :=
by sorry

end NUMINAMATH_CALUDE_orange_juice_serving_volume_l1594_159470


namespace NUMINAMATH_CALUDE_floor_times_self_eq_90_l1594_159449

theorem floor_times_self_eq_90 :
  ∃ (x : ℝ), x > 0 ∧ (⌊x⌋ : ℝ) * x = 90 ∧ x = 10 := by
  sorry

end NUMINAMATH_CALUDE_floor_times_self_eq_90_l1594_159449


namespace NUMINAMATH_CALUDE_car_profit_percent_l1594_159467

/-- Calculate the profit percent from buying, repairing, and selling a car -/
theorem car_profit_percent (purchase_price repair_cost selling_price : ℚ) : 
  purchase_price = 48000 →
  repair_cost = 14000 →
  selling_price = 72900 →
  (selling_price - (purchase_price + repair_cost)) / (purchase_price + repair_cost) * 100 = 1758/100 := by
sorry

end NUMINAMATH_CALUDE_car_profit_percent_l1594_159467


namespace NUMINAMATH_CALUDE_hyperbola_focal_length_l1594_159483

/-- The focal length of a hyperbola with equation x² - y² = 1 is 2√2 -/
theorem hyperbola_focal_length :
  let h : ℝ → ℝ → Prop := λ x y ↦ x^2 - y^2 = 1
  ∃ (f : ℝ), (f = 2 * Real.sqrt 2 ∧ 
    ∀ (c : ℝ), (c^2 = 2 → f = 2*c) ∧
    ∃ (a b : ℝ), (a^2 = 1 ∧ b^2 = 1 ∧ c^2 = a^2 + b^2)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_focal_length_l1594_159483


namespace NUMINAMATH_CALUDE_multiply_times_theorem_l1594_159439

theorem multiply_times_theorem (n : ℝ) (x : ℝ) (h1 : n = 1) :
  x * n - 1 = 2 * n → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_multiply_times_theorem_l1594_159439


namespace NUMINAMATH_CALUDE_point_in_third_quadrant_implies_a_less_than_one_l1594_159495

/-- A point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Predicate to check if a point is in the third quadrant -/
def in_third_quadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y < 0

/-- Theorem: If M(-1, a-1) is in the third quadrant, then a < 1 -/
theorem point_in_third_quadrant_implies_a_less_than_one (a : ℝ) :
  in_third_quadrant (Point.mk (-1) (a - 1)) → a < 1 := by
  sorry

end NUMINAMATH_CALUDE_point_in_third_quadrant_implies_a_less_than_one_l1594_159495


namespace NUMINAMATH_CALUDE_min_sum_of_prime_factors_l1594_159401

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

/-- The sum of n consecutive integers starting from x -/
def consecutiveSum (x n : ℕ) : ℕ :=
  n * (2 * x + n - 1) / 2

theorem min_sum_of_prime_factors (a b c d : ℕ) :
  isPrime a → isPrime b → isPrime c → isPrime d →
  (∃ x : ℕ, a * b * c * d = consecutiveSum x 35) →
  22 ≤ a + b + c + d :=
sorry

end NUMINAMATH_CALUDE_min_sum_of_prime_factors_l1594_159401


namespace NUMINAMATH_CALUDE_bracelet_ratio_l1594_159420

theorem bracelet_ratio : 
  ∀ (x : ℕ), 
  (5 + x : ℚ) - (1/3) * (5 + x) = 6 → 
  (x : ℚ) / 16 = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_bracelet_ratio_l1594_159420


namespace NUMINAMATH_CALUDE_kim_hard_round_correct_l1594_159435

/-- A math contest with three rounds of questions --/
structure MathContest where
  easy_points : ℕ
  average_points : ℕ
  hard_points : ℕ
  easy_correct : ℕ
  average_correct : ℕ
  total_points : ℕ

/-- Kim's performance in the math contest --/
def kim_contest : MathContest :=
  { easy_points := 2
  , average_points := 3
  , hard_points := 5
  , easy_correct := 6
  , average_correct := 2
  , total_points := 38 }

/-- The number of correct answers in the hard round --/
def hard_round_correct (contest : MathContest) : ℕ :=
  (contest.total_points - (contest.easy_points * contest.easy_correct + contest.average_points * contest.average_correct)) / contest.hard_points

theorem kim_hard_round_correct :
  hard_round_correct kim_contest = 4 := by
  sorry

end NUMINAMATH_CALUDE_kim_hard_round_correct_l1594_159435


namespace NUMINAMATH_CALUDE_inequality_proof_l1594_159472

theorem inequality_proof (x : ℝ) : (2*x - 1)/3 + 1 ≤ 0 → x ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1594_159472


namespace NUMINAMATH_CALUDE_equivalence_point_cost_effectiveness_l1594_159471

-- Define the full ticket price
def full_price : ℝ := 240

-- Define the charge functions for Travel Agency A and B
def charge_A (x : ℝ) : ℝ := 120 * x + 240
def charge_B (x : ℝ) : ℝ := 144 * x + 144

-- Theorem for the equivalence point
theorem equivalence_point :
  ∃ x : ℝ, charge_A x = charge_B x ∧ x = 4 := by sorry

-- Theorem for cost-effectiveness comparison
theorem cost_effectiveness (x : ℝ) :
  (x < 4 → charge_B x < charge_A x) ∧
  (x > 4 → charge_A x < charge_B x) := by sorry

end NUMINAMATH_CALUDE_equivalence_point_cost_effectiveness_l1594_159471


namespace NUMINAMATH_CALUDE_lcm_gcd_product_l1594_159440

theorem lcm_gcd_product (a b : ℕ) (ha : a = 12) (hb : b = 9) :
  Nat.lcm a b * Nat.gcd a b = 108 := by
sorry

end NUMINAMATH_CALUDE_lcm_gcd_product_l1594_159440


namespace NUMINAMATH_CALUDE_book_pages_calculation_l1594_159421

/-- Given a book where:
  1. The initial ratio of pages read to pages not read is 3:4
  2. After reading 33 more pages, the ratio becomes 5:3
  This theorem states that the total number of pages in the book
  is equal to 33 divided by the difference between 5/8 and 3/7. -/
theorem book_pages_calculation (initial_read : ℚ) (initial_unread : ℚ) 
  (final_read : ℚ) (final_unread : ℚ) :
  initial_read / initial_unread = 3 / 4 →
  (initial_read + 33) / initial_unread = 5 / 3 →
  (initial_read + initial_unread) = 33 / (5/8 - 3/7) := by
  sorry

end NUMINAMATH_CALUDE_book_pages_calculation_l1594_159421


namespace NUMINAMATH_CALUDE_centroid_property_l1594_159436

/-- The centroid of a triangle divides each median in the ratio 2:1 -/
def is_centroid (P Q R S : ℝ × ℝ) : Prop :=
  S.1 = (P.1 + Q.1 + R.1) / 3 ∧ S.2 = (P.2 + Q.2 + R.2) / 3

theorem centroid_property :
  let P : ℝ × ℝ := (2, 5)
  let Q : ℝ × ℝ := (9, 3)
  let R : ℝ × ℝ := (4, -4)
  let S : ℝ × ℝ := (x, y)
  is_centroid P Q R S → 9 * x + 4 * y = 151 / 3 := by
  sorry

end NUMINAMATH_CALUDE_centroid_property_l1594_159436
