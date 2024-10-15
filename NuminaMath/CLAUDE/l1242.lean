import Mathlib

namespace NUMINAMATH_CALUDE_lemonade_stand_cost_l1242_124247

theorem lemonade_stand_cost (net_profit babysitting_income : ℝ)
  (gross_revenue num_lemonades : ℕ)
  (lemon_cost sugar_cost ice_cost : ℝ)
  (bulk_discount sales_tax sunhat_cost : ℝ) :
  net_profit = 44 →
  babysitting_income = 31 →
  gross_revenue = 47 →
  num_lemonades = 50 →
  lemon_cost = 0.20 →
  sugar_cost = 0.15 →
  ice_cost = 0.05 →
  bulk_discount = 0.10 →
  sales_tax = 0.05 →
  sunhat_cost = 10 →
  ∃ (total_cost : ℝ),
    total_cost = (num_lemonades * (lemon_cost + sugar_cost + ice_cost) -
      num_lemonades * (lemon_cost + sugar_cost) * bulk_discount +
      gross_revenue * sales_tax + sunhat_cost) ∧
    total_cost = 30.60 :=
by sorry

end NUMINAMATH_CALUDE_lemonade_stand_cost_l1242_124247


namespace NUMINAMATH_CALUDE_exists_digit_sum_div_11_l1242_124241

def digit_sum (n : ℕ) : ℕ := sorry

theorem exists_digit_sum_div_11 (n : ℕ) : ∃ k, n ≤ k ∧ k < n + 39 ∧ (digit_sum k) % 11 = 0 := by
  sorry

end NUMINAMATH_CALUDE_exists_digit_sum_div_11_l1242_124241


namespace NUMINAMATH_CALUDE_sequence_property_l1242_124282

theorem sequence_property (m : ℤ) (a : ℕ → ℤ) (r s : ℕ) : 
  (|m| ≥ 2) →
  (∃ k, a k ≠ 0) →
  (∀ n : ℕ, a (n + 2) = a (n + 1) - m * a n) →
  (r > s) →
  (s ≥ 2) →
  (a r = a s) →
  (a r = a 1) →
  (r - s ≥ |m|) := by
sorry

end NUMINAMATH_CALUDE_sequence_property_l1242_124282


namespace NUMINAMATH_CALUDE_prob_two_non_defective_pens_l1242_124265

/-- Given a box of 16 pens with 3 defective pens, prove that the probability
    of selecting 2 non-defective pens at random is 13/20. -/
theorem prob_two_non_defective_pens :
  let total_pens : ℕ := 16
  let defective_pens : ℕ := 3
  let non_defective_pens : ℕ := total_pens - defective_pens
  let prob_first_non_defective : ℚ := non_defective_pens / total_pens
  let prob_second_non_defective : ℚ := (non_defective_pens - 1) / (total_pens - 1)
  prob_first_non_defective * prob_second_non_defective = 13 / 20 := by
  sorry


end NUMINAMATH_CALUDE_prob_two_non_defective_pens_l1242_124265


namespace NUMINAMATH_CALUDE_variable_value_l1242_124263

theorem variable_value (x y : ℤ) (h1 : 2 * x - y = 11) (h2 : 4 * x + y ≠ 17) : y = -9 := by
  sorry

end NUMINAMATH_CALUDE_variable_value_l1242_124263


namespace NUMINAMATH_CALUDE_complex_pure_imaginary_l1242_124203

theorem complex_pure_imaginary (m : ℝ) : 
  (m + (10 : ℂ) / (3 + Complex.I)).im ≠ 0 ∧ (m + (10 : ℂ) / (3 + Complex.I)).re = 0 → m = -3 := by
  sorry

end NUMINAMATH_CALUDE_complex_pure_imaginary_l1242_124203


namespace NUMINAMATH_CALUDE_exists_integer_divisible_by_24_with_cube_root_between_9_and_9_5_l1242_124280

theorem exists_integer_divisible_by_24_with_cube_root_between_9_and_9_5 :
  ∃ n : ℕ+, 24 ∣ n ∧ 9 < (n : ℝ) ^ (1/3) ∧ (n : ℝ) ^ (1/3) < 9.5 := by
  sorry

end NUMINAMATH_CALUDE_exists_integer_divisible_by_24_with_cube_root_between_9_and_9_5_l1242_124280


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1242_124292

/-- An arithmetic sequence with a_2 = 1 and a_5 = 7 has common difference 2 -/
theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)  -- The arithmetic sequence
  (h1 : a 2 = 1)  -- Given: a_2 = 1
  (h2 : a 5 = 7)  -- Given: a_5 = 7
  (h3 : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1))  -- Definition of arithmetic sequence
  : a 3 - a 2 = 2 :=  -- Conclusion: The common difference is 2
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1242_124292


namespace NUMINAMATH_CALUDE_johns_naps_per_week_l1242_124216

/-- Given that John takes 60 hours of naps in 70 days, and each nap is 2 hours long,
    prove that he takes 3 naps per week. -/
theorem johns_naps_per_week 
  (nap_duration : ℝ) 
  (total_days : ℝ) 
  (total_nap_hours : ℝ) 
  (h1 : nap_duration = 2)
  (h2 : total_days = 70)
  (h3 : total_nap_hours = 60) :
  (total_nap_hours / (total_days / 7)) / nap_duration = 3 :=
by sorry

end NUMINAMATH_CALUDE_johns_naps_per_week_l1242_124216


namespace NUMINAMATH_CALUDE_range_of_m_l1242_124296

theorem range_of_m (x y m : ℝ) 
  (hx : 1 < x ∧ x < 3) 
  (hy : -3 < y ∧ y < 1) 
  (hm : m = x - 3*y) : 
  -2 < m ∧ m < 12 := by
sorry

end NUMINAMATH_CALUDE_range_of_m_l1242_124296


namespace NUMINAMATH_CALUDE_same_solution_systems_l1242_124224

theorem same_solution_systems (m n : ℝ) : 
  (∃ x y : ℝ, 5*x - 2*y = 3 ∧ m*x + 5*y = 4 ∧ x - 4*y = -3 ∧ 5*x + n*y = 1) →
  m = -1 ∧ n = -4 := by
  sorry

end NUMINAMATH_CALUDE_same_solution_systems_l1242_124224


namespace NUMINAMATH_CALUDE_inequality_preservation_l1242_124237

theorem inequality_preservation (a b c : ℝ) (h : a > b) : a - c > b - c := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l1242_124237


namespace NUMINAMATH_CALUDE_sum_of_distinct_prime_factors_882_l1242_124211

def sum_of_distinct_prime_factors (n : ℕ) : ℕ := sorry

theorem sum_of_distinct_prime_factors_882 :
  sum_of_distinct_prime_factors 882 = 12 := by sorry

end NUMINAMATH_CALUDE_sum_of_distinct_prime_factors_882_l1242_124211


namespace NUMINAMATH_CALUDE_prob_level_b_part1_prob_not_qualifying_part2_l1242_124235

-- Define the probability of success for a single attempt
def p_success : ℚ := 1/2

-- Define the number of attempts for part 1
def attempts_part1 : ℕ := 4

-- Define the number of successes required for level B
def level_b_successes : ℕ := 3

-- Define the maximum number of attempts for part 2
def max_attempts_part2 : ℕ := 5

-- Part 1: Probability of exactly 3 successes in 4 attempts
theorem prob_level_b_part1 :
  (Nat.choose attempts_part1 level_b_successes : ℚ) * p_success^level_b_successes * (1 - p_success)^(attempts_part1 - level_b_successes) = 3/16 := by
  sorry

-- Part 2: Probability of not qualifying as level B or A player
theorem prob_not_qualifying_part2 :
  let seq := List.cons p_success (List.cons p_success (List.cons (1 - p_success) (List.cons (1 - p_success) [])))
  let p_exactly_3 := (Nat.choose 4 2 : ℚ) * p_success^3 * (1 - p_success)^2
  let p_exactly_2 := p_success^2 * (1 - p_success)^2 + 3 * p_success^2 * (1 - p_success)^3
  let p_exactly_1 := p_success * (1 - p_success)^2 + p_success * (1 - p_success)^3
  let p_exactly_0 := (1 - p_success)^2
  p_exactly_3 + p_exactly_2 + p_exactly_1 + p_exactly_0 = 25/32 := by
  sorry

end NUMINAMATH_CALUDE_prob_level_b_part1_prob_not_qualifying_part2_l1242_124235


namespace NUMINAMATH_CALUDE_diagonals_divisible_by_3_count_l1242_124251

/-- A convex polygon with 30 sides -/
structure ConvexPolygon30 where
  sides : ℕ
  convex : Bool
  sides_eq_30 : sides = 30

/-- The number of diagonals in a polygon that are divisible by 3 -/
def diagonals_divisible_by_3 (p : ConvexPolygon30) : ℕ := 17

/-- Theorem stating that the number of diagonals divisible by 3 in a convex 30-sided polygon is 17 -/
theorem diagonals_divisible_by_3_count (p : ConvexPolygon30) : 
  diagonals_divisible_by_3 p = 17 := by sorry

end NUMINAMATH_CALUDE_diagonals_divisible_by_3_count_l1242_124251


namespace NUMINAMATH_CALUDE_eleventhDrawnNumber_l1242_124248

/-- Systematic sampling function -/
def systematicSample (totalParticipants : ℕ) (sampleSize : ℕ) (firstDrawn : ℕ) (n : ℕ) : ℕ :=
  firstDrawn + (n - 1) * (totalParticipants / sampleSize)

/-- Theorem: 11th number drawn in the systematic sampling -/
theorem eleventhDrawnNumber (totalParticipants : ℕ) (sampleSize : ℕ) (firstDrawn : ℕ) :
  totalParticipants = 1000 →
  sampleSize = 50 →
  firstDrawn = 15 →
  systematicSample totalParticipants sampleSize firstDrawn 11 = 215 := by
  sorry

#check eleventhDrawnNumber

end NUMINAMATH_CALUDE_eleventhDrawnNumber_l1242_124248


namespace NUMINAMATH_CALUDE_games_needed_for_512_players_l1242_124254

/-- Represents a single-elimination tournament -/
structure SingleEliminationTournament where
  initial_players : ℕ
  games_played : ℕ

/-- Calculates the number of games needed to declare a champion -/
def games_needed (tournament : SingleEliminationTournament) : ℕ :=
  tournament.initial_players - 1

/-- Theorem: In a single-elimination tournament with 512 initial players,
    511 games are needed to declare a champion -/
theorem games_needed_for_512_players :
  ∀ (tournament : SingleEliminationTournament),
    tournament.initial_players = 512 →
    games_needed tournament = 511 := by
  sorry

#check games_needed_for_512_players

end NUMINAMATH_CALUDE_games_needed_for_512_players_l1242_124254


namespace NUMINAMATH_CALUDE_floor_ceil_sum_l1242_124213

theorem floor_ceil_sum : ⌊(1.999 : ℝ)⌋ + ⌈(3.001 : ℝ)⌉ + ⌈(0.001 : ℝ)⌉ = 6 := by
  sorry

end NUMINAMATH_CALUDE_floor_ceil_sum_l1242_124213


namespace NUMINAMATH_CALUDE_age_difference_proof_l1242_124201

theorem age_difference_proof (total_age : ℕ) (ratio_a ratio_b ratio_c ratio_d : ℕ) :
  total_age = 190 ∧ ratio_a = 4 ∧ ratio_b = 3 ∧ ratio_c = 7 ∧ ratio_d = 5 →
  ∃ (x : ℚ), x * (ratio_a + ratio_b + ratio_c + ratio_d) = total_age ∧
             x * ratio_a - x * ratio_b = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_age_difference_proof_l1242_124201


namespace NUMINAMATH_CALUDE_max_value_polynomial_l1242_124221

theorem max_value_polynomial (x y : ℝ) (h : x + y = 4) :
  x^4*y + x^3*y + x^2*y + x*y + x*y^2 + x*y^3 + x*y^4 ≤ 7225/28 := by
  sorry

end NUMINAMATH_CALUDE_max_value_polynomial_l1242_124221


namespace NUMINAMATH_CALUDE_total_selections_l1242_124219

/-- Represents a hexagonal arrangement of circles -/
structure HexCircleArrangement :=
  (total_circles : ℕ)
  (side_length : ℕ)

/-- Calculates the number of ways to select three consecutive circles in one direction -/
def consecutive_selections (n : ℕ) : ℕ :=
  if n < 3 then 0 else n - 2

/-- Calculates the number of ways to select three consecutive circles in a diagonal direction -/
def diagonal_selections (n : ℕ) : ℕ :=
  if n < 3 then 0 else (n - 2) * (n - 1) / 2

/-- The main theorem stating the total number of ways to select three consecutive circles -/
theorem total_selections (h : HexCircleArrangement) 
  (h_total : h.total_circles = 33) 
  (h_side : h.side_length = 7) : 
  consecutive_selections h.side_length + 2 * diagonal_selections h.side_length = 57 := by
  sorry


end NUMINAMATH_CALUDE_total_selections_l1242_124219


namespace NUMINAMATH_CALUDE_tan_four_thirds_pi_l1242_124295

theorem tan_four_thirds_pi : Real.tan (4 * π / 3) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_four_thirds_pi_l1242_124295


namespace NUMINAMATH_CALUDE_equal_digit_probability_l1242_124268

/-- The number of sides on each die -/
def num_sides : ℕ := 20

/-- The number of dice rolled -/
def num_dice : ℕ := 6

/-- The probability of rolling a one-digit number on a single die -/
def prob_one_digit : ℚ := 9 / 20

/-- The probability of rolling a two-digit number on a single die -/
def prob_two_digit : ℚ := 11 / 20

/-- The number of ways to choose half the dice -/
def num_combinations : ℕ := (num_dice.choose (num_dice / 2))

/-- The probability of getting an equal number of one-digit and two-digit numbers when rolling 6 20-sided dice -/
theorem equal_digit_probability : 
  (num_combinations : ℚ) * (prob_one_digit ^ (num_dice / 2)) * (prob_two_digit ^ (num_dice / 2)) = 485264 / 1600000 := by
  sorry

end NUMINAMATH_CALUDE_equal_digit_probability_l1242_124268


namespace NUMINAMATH_CALUDE_wood_cutting_problem_l1242_124264

theorem wood_cutting_problem (original_length : ℚ) (first_cut : ℚ) (second_cut : ℚ) :
  original_length = 35/8 ∧ first_cut = 5/3 ∧ second_cut = 9/4 →
  (original_length - first_cut - second_cut) / 3 = 11/72 := by
  sorry

end NUMINAMATH_CALUDE_wood_cutting_problem_l1242_124264


namespace NUMINAMATH_CALUDE_conference_arrangement_count_l1242_124271

/-- Represents the number of teachers from each school -/
structure SchoolTeachers :=
  (A : ℕ)
  (B : ℕ)
  (C : ℕ)

/-- Calculates the number of ways to arrange teachers from different schools -/
def arrangementCount (teachers : SchoolTeachers) : ℕ :=
  sorry

/-- The specific arrangement of teachers from the problem -/
def conferenceTeachers : SchoolTeachers :=
  { A := 2, B := 2, C := 1 }

/-- Theorem stating that the number of valid arrangements is 48 -/
theorem conference_arrangement_count :
  arrangementCount conferenceTeachers = 48 :=
sorry

end NUMINAMATH_CALUDE_conference_arrangement_count_l1242_124271


namespace NUMINAMATH_CALUDE_train_speed_l1242_124244

/-- Calculates the speed of a train in km/hr given its length and time to pass a tree -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 275) (h2 : time = 11) :
  (length / time) * 3.6 = 90 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l1242_124244


namespace NUMINAMATH_CALUDE_perfect_square_base_9_l1242_124220

/-- Represents a number in base 9 of the form ab5c where a ≠ 0 -/
structure Base9Number where
  a : ℕ
  b : ℕ
  c : ℕ
  a_nonzero : a ≠ 0
  b_less_than_9 : b < 9
  c_less_than_9 : c < 9

/-- Converts a Base9Number to its decimal representation -/
def toDecimal (n : Base9Number) : ℕ :=
  729 * n.a + 81 * n.b + 45 + n.c

theorem perfect_square_base_9 (n : Base9Number) :
  ∃ (k : ℕ), toDecimal n = k^2 → n.c = 0 ∨ n.c = 7 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_base_9_l1242_124220


namespace NUMINAMATH_CALUDE_passengers_taken_at_first_station_l1242_124249

/-- Represents the number of passengers on the train at various points --/
structure TrainPassengers where
  initial : ℕ
  afterFirstDrop : ℕ
  afterFirstPickup : ℕ
  afterSecondDrop : ℕ
  afterSecondPickup : ℕ
  final : ℕ

/-- Represents the passenger flow on the train's journey --/
def trainJourney (x : ℕ) : TrainPassengers :=
  { initial := 270,
    afterFirstDrop := 270 - (270 / 3),
    afterFirstPickup := 270 - (270 / 3) + x,
    afterSecondDrop := (270 - (270 / 3) + x) - ((270 - (270 / 3) + x) / 2),
    afterSecondPickup := (270 - (270 / 3) + x) - ((270 - (270 / 3) + x) / 2) + 12,
    final := 242 }

/-- Theorem stating that 280 passengers were taken at the first station --/
theorem passengers_taken_at_first_station :
  ∃ (x : ℕ), trainJourney x = trainJourney 280 ∧ 
  (trainJourney x).afterSecondPickup = (trainJourney x).final :=
sorry


end NUMINAMATH_CALUDE_passengers_taken_at_first_station_l1242_124249


namespace NUMINAMATH_CALUDE_fourth_root_equation_solution_l1242_124290

theorem fourth_root_equation_solution : 
  ∃ (p q r : ℕ+), 
    4 * (7^(1/4) - 6^(1/4))^(1/4) = p^(1/4) + q^(1/4) - r^(1/4) ∧ 
    p + q + r = 99 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_equation_solution_l1242_124290


namespace NUMINAMATH_CALUDE_conference_handshakes_l1242_124245

/-- The number of unique handshakes in a conference -/
def unique_handshakes (n : ℕ) (k : ℕ) : ℕ :=
  (n * k) / 2

/-- Theorem: In a conference of 12 people where each person shakes hands with 6 others,
    there are 36 unique handshakes -/
theorem conference_handshakes :
  unique_handshakes 12 6 = 36 := by
  sorry

end NUMINAMATH_CALUDE_conference_handshakes_l1242_124245


namespace NUMINAMATH_CALUDE_part1_part2_l1242_124208

-- Define the complex number z as a function of m
def z (m : ℝ) : ℂ := Complex.mk (m^2 + m - 6) (m^2 + m - 2)

-- Part 1: Prove that if z - 2m is purely imaginary, then m = 3
theorem part1 (m : ℝ) : (z m - 2 * m).re = 0 → m = 3 := by sorry

-- Part 2: Prove that if z is in the second quadrant, then m is in (-3, -2) ∪ (1, 2)
theorem part2 (m : ℝ) : (z m).re < 0 ∧ (z m).im > 0 → m ∈ Set.Ioo (-3) (-2) ∪ Set.Ioo 1 2 := by sorry

end NUMINAMATH_CALUDE_part1_part2_l1242_124208


namespace NUMINAMATH_CALUDE_no_solution_when_n_negative_one_l1242_124277

-- Define the system of equations
def system (n x y z : ℝ) : Prop :=
  n * x^2 + y = 2 ∧ n * y^2 + z = 2 ∧ n * z^2 + x = 2

-- Theorem stating that the system has no solution when n = -1
theorem no_solution_when_n_negative_one :
  ¬ ∃ (x y z : ℝ), system (-1) x y z :=
sorry

end NUMINAMATH_CALUDE_no_solution_when_n_negative_one_l1242_124277


namespace NUMINAMATH_CALUDE_product_of_real_parts_of_complex_solutions_l1242_124236

theorem product_of_real_parts_of_complex_solutions : ∃ (z₁ z₂ : ℂ),
  (z₁^2 + 2*z₁ = Complex.I) ∧ 
  (z₂^2 + 2*z₂ = Complex.I) ∧
  (z₁ ≠ z₂) ∧
  (Complex.re z₁ * Complex.re z₂ = (1 - Real.sqrt 2) / 2) := by
  sorry

end NUMINAMATH_CALUDE_product_of_real_parts_of_complex_solutions_l1242_124236


namespace NUMINAMATH_CALUDE_jenna_round_trip_pay_l1242_124299

/-- Calculates the pay for a round trip given the pay rate per mile and one-way distance -/
def round_trip_pay (rate : ℚ) (one_way_distance : ℚ) : ℚ :=
  2 * rate * one_way_distance

/-- Proves that the round trip pay for a rate of $0.40 per mile and 400 miles one-way is $320 -/
theorem jenna_round_trip_pay :
  round_trip_pay (40 / 100) 400 = 320 := by
  sorry

#eval round_trip_pay (40 / 100) 400

end NUMINAMATH_CALUDE_jenna_round_trip_pay_l1242_124299


namespace NUMINAMATH_CALUDE_problem_solution_l1242_124217

theorem problem_solution (x y z : ℚ) 
  (sum_condition : x + y + z = 120)
  (equal_condition : x + 10 = y - 5 ∧ y - 5 = 4*z) : 
  y = 545/9 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1242_124217


namespace NUMINAMATH_CALUDE_birthday_crayons_count_l1242_124273

/-- The number of crayons Paul got for his birthday -/
def birthday_crayons : ℕ := sorry

/-- The number of crayons Paul got at the end of the school year -/
def school_year_crayons : ℕ := 134

/-- The total number of crayons Paul has now -/
def total_crayons : ℕ := 613

/-- Theorem stating that the number of crayons Paul got for his birthday is 479 -/
theorem birthday_crayons_count : birthday_crayons = 479 := by
  sorry

end NUMINAMATH_CALUDE_birthday_crayons_count_l1242_124273


namespace NUMINAMATH_CALUDE_x_axis_ellipse_iff_condition_l1242_124285

/-- An ellipse with foci on the x-axis -/
structure XAxisEllipse where
  k : ℝ
  eq : ∀ (x y : ℝ), x^2 / 2 + y^2 / k = 1

/-- The condition for an ellipse with foci on the x-axis -/
def is_x_axis_ellipse_condition (k : ℝ) : Prop :=
  0 < k ∧ k < 2

/-- The theorem stating that 0 < k < 2 is a necessary and sufficient condition 
    for the equation x^2/2 + y^2/k = 1 to represent an ellipse with foci on the x-axis -/
theorem x_axis_ellipse_iff_condition (e : XAxisEllipse) :
  is_x_axis_ellipse_condition e.k ↔ True :=
sorry

end NUMINAMATH_CALUDE_x_axis_ellipse_iff_condition_l1242_124285


namespace NUMINAMATH_CALUDE_pairwise_product_signs_l1242_124239

theorem pairwise_product_signs (a b c : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  let products := [a * b, b * c, c * a]
  (products.filter (· > 0)).length = 1 ∨ (products.filter (· > 0)).length = 3 :=
sorry

end NUMINAMATH_CALUDE_pairwise_product_signs_l1242_124239


namespace NUMINAMATH_CALUDE_f_domain_l1242_124279

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x + 1) + 1 / (2 - x)

def domain (f : ℝ → ℝ) : Set ℝ := {x | ∃ y, f x = y}

theorem f_domain : domain f = {x : ℝ | x ≥ -1 ∧ x ≠ 2} := by
  sorry

end NUMINAMATH_CALUDE_f_domain_l1242_124279


namespace NUMINAMATH_CALUDE_arccos_negative_one_equals_pi_l1242_124202

theorem arccos_negative_one_equals_pi : Real.arccos (-1) = π := by
  sorry

end NUMINAMATH_CALUDE_arccos_negative_one_equals_pi_l1242_124202


namespace NUMINAMATH_CALUDE_rectangle_area_l1242_124233

theorem rectangle_area (r : ℝ) (ratio : ℝ) : 
  r = 3 → ratio = 3 → 2 * r * (ratio + 1) = 108 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l1242_124233


namespace NUMINAMATH_CALUDE_consecutive_product_sum_l1242_124238

theorem consecutive_product_sum : ∃ (a b x y z : ℕ), 
  (a + 1 = b) ∧ 
  (x + 1 = y) ∧ 
  (y + 1 = z) ∧
  (a * b = 1320) ∧ 
  (x * y * z = 1320) ∧ 
  (a + b + x + y + z = 106) := by
sorry

end NUMINAMATH_CALUDE_consecutive_product_sum_l1242_124238


namespace NUMINAMATH_CALUDE_product_36_sum_0_l1242_124283

theorem product_36_sum_0 (a b c d e f : ℤ) : 
  a * b * c * d * e * f = 36 ∧ 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
  d ≠ e ∧ d ≠ f ∧
  e ≠ f →
  a + b + c + d + e + f = 0 :=
sorry

end NUMINAMATH_CALUDE_product_36_sum_0_l1242_124283


namespace NUMINAMATH_CALUDE_binomial_coefficient_equality_l1242_124294

theorem binomial_coefficient_equality (n : ℕ+) : 
  (Nat.choose n.val 2 = Nat.choose n.val 3) → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equality_l1242_124294


namespace NUMINAMATH_CALUDE_symmetric_points_range_l1242_124250

theorem symmetric_points_range (a : ℝ) : 
  (∃ x ∈ Set.Icc 1 2, a - x^2 = -(2*x + 1)) → a ∈ Set.Icc (-2) (-1) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_range_l1242_124250


namespace NUMINAMATH_CALUDE_parallelograms_in_triangle_l1242_124252

/-- The number of parallelograms formed inside a triangle -/
def num_parallelograms (n : ℕ) : ℕ := 3 * Nat.choose (n + 2) 4

/-- 
Theorem: The number of parallelograms formed inside a triangle 
whose sides are divided into n equal parts with parallel lines 
drawn through these points is equal to 3 * (n+2 choose 4).
-/
theorem parallelograms_in_triangle (n : ℕ) : 
  num_parallelograms n = 3 * Nat.choose (n + 2) 4 := by
  sorry

end NUMINAMATH_CALUDE_parallelograms_in_triangle_l1242_124252


namespace NUMINAMATH_CALUDE_oil_floats_on_water_l1242_124214

-- Define the density of a substance
def density (substance : Type) : ℝ := sorry

-- Define what it means for a substance to float on another
def floats_on (a b : Type) : Prop := 
  density a < density b

-- Define oil and water as types
def oil : Type := sorry
def water : Type := sorry

-- State the theorem
theorem oil_floats_on_water : 
  (density oil < density water) → floats_on oil water := by sorry

end NUMINAMATH_CALUDE_oil_floats_on_water_l1242_124214


namespace NUMINAMATH_CALUDE_gcd_16016_20020_l1242_124298

theorem gcd_16016_20020 : Nat.gcd 16016 20020 = 4004 := by
  sorry

end NUMINAMATH_CALUDE_gcd_16016_20020_l1242_124298


namespace NUMINAMATH_CALUDE_carolyn_embroiders_50_flowers_l1242_124215

/-- Represents the embroidery problem with given conditions -/
structure EmbroideryProblem where
  stitches_per_minute : ℕ
  stitches_per_flower : ℕ
  stitches_per_unicorn : ℕ
  stitches_for_godzilla : ℕ
  num_unicorns : ℕ
  total_minutes : ℕ

/-- Calculates the number of flowers Carolyn wants to embroider -/
def flowers_to_embroider (p : EmbroideryProblem) : ℕ :=
  let total_stitches := p.stitches_per_minute * p.total_minutes
  let stitches_for_creatures := p.stitches_for_godzilla + p.num_unicorns * p.stitches_per_unicorn
  let remaining_stitches := total_stitches - stitches_for_creatures
  remaining_stitches / p.stitches_per_flower

/-- Theorem stating that given the problem conditions, Carolyn wants to embroider 50 flowers -/
theorem carolyn_embroiders_50_flowers :
  let p := EmbroideryProblem.mk 4 60 180 800 3 1085
  flowers_to_embroider p = 50 := by
  sorry


end NUMINAMATH_CALUDE_carolyn_embroiders_50_flowers_l1242_124215


namespace NUMINAMATH_CALUDE_henrys_money_l1242_124212

theorem henrys_money (x : ℤ) : 
  (x + 18 - 10 = 19) → (x = 11) := by
  sorry

end NUMINAMATH_CALUDE_henrys_money_l1242_124212


namespace NUMINAMATH_CALUDE_isosceles_triangle_third_side_l1242_124276

/-- An isosceles triangle with side lengths 4 and 8 has its third side equal to 8 -/
theorem isosceles_triangle_third_side : ∀ (a b c : ℝ),
  a = 4 ∧ b = 8 ∧ (a = b ∨ b = c ∨ a = c) →  -- isosceles condition
  (a + b > c ∧ b + c > a ∧ a + c > b) →      -- triangle inequality
  c = 8 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_third_side_l1242_124276


namespace NUMINAMATH_CALUDE_solution_set_inequality_l1242_124259

/-- Given that the solution set of x^2 + ax + b > 0 is (-∞, -2) ∪ (-1/2, +∞),
    prove that the solution set of bx^2 + ax + 1 < 0 is (-2, -1/2) -/
theorem solution_set_inequality (a b : ℝ) : 
  (∀ x, x^2 + a*x + b > 0 ↔ x < -2 ∨ x > -1/2) →
  (∀ x, b*x^2 + a*x + 1 < 0 ↔ -2 < x ∧ x < -1/2) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l1242_124259


namespace NUMINAMATH_CALUDE_softball_team_ratio_l1242_124225

/-- Represents a co-ed softball team -/
structure SoftballTeam where
  men : ℕ
  women : ℕ

/-- The ratio of two natural numbers -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

/-- Theorem stating the ratio of men to women on the softball team -/
theorem softball_team_ratio (team : SoftballTeam) : 
  team.women = team.men + 4 → 
  team.men + team.women = 14 → 
  ∃ (r : Ratio), r.numerator = team.men ∧ r.denominator = team.women ∧ r.numerator = 5 ∧ r.denominator = 9 :=
by sorry

end NUMINAMATH_CALUDE_softball_team_ratio_l1242_124225


namespace NUMINAMATH_CALUDE_max_blue_points_l1242_124210

theorem max_blue_points (total_spheres : ℕ) (h : total_spheres = 2016) :
  ∃ (red_spheres : ℕ), 
    red_spheres ≤ total_spheres ∧
    red_spheres * (total_spheres - red_spheres) = 1016064 ∧
    ∀ (x : ℕ), x ≤ total_spheres → 
      x * (total_spheres - x) ≤ 1016064 := by
  sorry

end NUMINAMATH_CALUDE_max_blue_points_l1242_124210


namespace NUMINAMATH_CALUDE_items_per_charge_l1242_124262

def total_items : ℕ := 20
def num_cards : ℕ := 4

theorem items_per_charge :
  total_items / num_cards = 5 := by
  sorry

end NUMINAMATH_CALUDE_items_per_charge_l1242_124262


namespace NUMINAMATH_CALUDE_tan_80_in_terms_of_cos_100_l1242_124275

theorem tan_80_in_terms_of_cos_100 (m : ℝ) (h : Real.cos (100 * π / 180) = m) :
  Real.tan (80 * π / 180) = Real.sqrt (1 - m^2) / (-m) := by
  sorry

end NUMINAMATH_CALUDE_tan_80_in_terms_of_cos_100_l1242_124275


namespace NUMINAMATH_CALUDE_intersection_points_on_hyperbola_l1242_124230

/-- The intersection points of the lines 2tx - 3y - 5t = 0 and x - 3ty + 5 = 0,
    where t is a real number, lie on a hyperbola. -/
theorem intersection_points_on_hyperbola :
  ∀ (t x y : ℝ),
    (2 * t * x - 3 * y - 5 * t = 0) →
    (x - 3 * t * y + 5 = 0) →
    ∃ (a b : ℝ), x^2 / a^2 - y^2 / b^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_intersection_points_on_hyperbola_l1242_124230


namespace NUMINAMATH_CALUDE_max_value_of_roots_squared_sum_l1242_124218

theorem max_value_of_roots_squared_sum (k : ℝ) (x₁ x₂ : ℝ) : 
  x₁^2 - (k-2)*x₁ + (k^2 + 3*k + 5) = 0 →
  x₂^2 - (k-2)*x₂ + (k^2 + 3*k + 5) = 0 →
  x₁ ≠ x₂ →
  ∃ (max : ℝ), max = 18 ∧ x₁^2 + x₂^2 ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_roots_squared_sum_l1242_124218


namespace NUMINAMATH_CALUDE_disney_banquet_revenue_l1242_124243

/-- Calculates the total revenue from ticket sales for a Disney banquet --/
theorem disney_banquet_revenue :
  let total_attendees : ℕ := 586
  let resident_price : ℚ := 12.95
  let non_resident_price : ℚ := 17.95
  let num_residents : ℕ := 219
  let num_non_residents : ℕ := total_attendees - num_residents
  let resident_revenue : ℚ := num_residents * resident_price
  let non_resident_revenue : ℚ := num_non_residents * non_resident_price
  let total_revenue : ℚ := resident_revenue + non_resident_revenue
  total_revenue = 9423.70 := by
  sorry

end NUMINAMATH_CALUDE_disney_banquet_revenue_l1242_124243


namespace NUMINAMATH_CALUDE_sphere_volume_from_inscribed_cube_l1242_124231

theorem sphere_volume_from_inscribed_cube (s : Real) (r : Real) : 
  (6 * s^2 = 32) →  -- surface area of cube is 32
  (r = s * Real.sqrt 3 / 2) →  -- radius of sphere in terms of cube side length
  (4 / 3 * Real.pi * r^3 = 32 * Real.pi / 3) :=  -- volume of sphere
by sorry

end NUMINAMATH_CALUDE_sphere_volume_from_inscribed_cube_l1242_124231


namespace NUMINAMATH_CALUDE_function_inequality_condition_l1242_124293

theorem function_inequality_condition (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (∀ x : ℝ, |x + 0.4| < b → |5 * x - 3 + 1| < a) ↔ b ≤ a / 5 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_condition_l1242_124293


namespace NUMINAMATH_CALUDE_total_tires_changed_l1242_124281

/-- The number of tires on a motorcycle -/
def motorcycle_tires : ℕ := 2

/-- The number of tires on a car -/
def car_tires : ℕ := 4

/-- The number of motorcycles Mike changed tires on -/
def num_motorcycles : ℕ := 12

/-- The number of cars Mike changed tires on -/
def num_cars : ℕ := 10

/-- Theorem: The total number of tires Mike changed is 64 -/
theorem total_tires_changed : 
  num_motorcycles * motorcycle_tires + num_cars * car_tires = 64 := by
  sorry

end NUMINAMATH_CALUDE_total_tires_changed_l1242_124281


namespace NUMINAMATH_CALUDE_tangent_line_to_cubic_curve_l1242_124228

theorem tangent_line_to_cubic_curve (k : ℝ) : 
  (∃ x y : ℝ, y = x^3 ∧ y = k*x + 2 ∧ (3 * x^2 = k)) → k = 3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_to_cubic_curve_l1242_124228


namespace NUMINAMATH_CALUDE_problem_statement_l1242_124278

theorem problem_statement (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b)
  (h_condition : (a / (1 + a)) + (b / (1 + b)) = 1) :
  (a / (1 + b^2)) - (b / (1 + a^2)) = a - b := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1242_124278


namespace NUMINAMATH_CALUDE_rye_flour_amount_l1242_124253

/-- The amount of rye flour Sarah bought -/
def rye_flour : ℝ := sorry

/-- The amount of whole-wheat bread flour Sarah bought -/
def whole_wheat_bread : ℝ := 10

/-- The amount of chickpea flour Sarah bought -/
def chickpea : ℝ := 3

/-- The amount of whole-wheat pastry flour Sarah had at home -/
def whole_wheat_pastry : ℝ := 2

/-- The total amount of flour Sarah has now -/
def total_flour : ℝ := 20

/-- Theorem stating that the amount of rye flour Sarah bought is 5 pounds -/
theorem rye_flour_amount : rye_flour = 5 := by
  sorry

end NUMINAMATH_CALUDE_rye_flour_amount_l1242_124253


namespace NUMINAMATH_CALUDE_triangle_shortest_side_l1242_124234

theorem triangle_shortest_side (a b c : ℕ) (h : ℕ) : 
  a = 24 →                                  -- One side is 24
  a + b + c = 66 →                          -- Perimeter is 66
  b ≤ c →                                   -- b is the shortest side
  ∃ (A : ℕ), A * A = 297 * (33 - b) * (b - 9) →  -- Area is an integer (using Heron's formula)
  24 * h = 2 * A →                          -- Integer altitude condition
  b = 15 := by sorry

end NUMINAMATH_CALUDE_triangle_shortest_side_l1242_124234


namespace NUMINAMATH_CALUDE_newspaper_probability_l1242_124200

-- Define the time intervals
def delivery_start : ℝ := 6.5
def delivery_end : ℝ := 7.5
def departure_start : ℝ := 7.0
def departure_end : ℝ := 8.0

-- Define the probability function
def probability_of_getting_newspaper : ℝ := sorry

-- Theorem statement
theorem newspaper_probability :
  probability_of_getting_newspaper = 7 / 8 := by sorry

end NUMINAMATH_CALUDE_newspaper_probability_l1242_124200


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l1242_124272

theorem simplify_and_evaluate (a : ℕ) (ha : a = 2030) :
  (a + 1 : ℚ) / a - a / (a + 1) = (2 * a + 1 : ℚ) / (a * (a + 1)) ∧
  2 * a + 1 = 4061 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l1242_124272


namespace NUMINAMATH_CALUDE_proportion_equality_l1242_124255

theorem proportion_equality : (5 / 34) / (7 / 48) = (120 / 1547) / (1 / 13) := by
  sorry

end NUMINAMATH_CALUDE_proportion_equality_l1242_124255


namespace NUMINAMATH_CALUDE_tan_945_degrees_l1242_124240

theorem tan_945_degrees (x : ℝ) : 
  (∀ x, Real.tan (x + 2 * Real.pi) = Real.tan x) → 
  Real.tan (945 * Real.pi / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_945_degrees_l1242_124240


namespace NUMINAMATH_CALUDE_max_player_salary_l1242_124291

theorem max_player_salary (n : ℕ) (min_salary : ℕ) (total_cap : ℕ) :
  n = 18 →
  min_salary = 20000 →
  total_cap = 600000 →
  ∃ (max_salary : ℕ),
    max_salary = 260000 ∧
    max_salary = total_cap - (n - 1) * min_salary ∧
    max_salary ≥ min_salary ∧
    (n - 1) * min_salary + max_salary ≤ total_cap :=
by sorry

end NUMINAMATH_CALUDE_max_player_salary_l1242_124291


namespace NUMINAMATH_CALUDE_smallest_product_l1242_124207

def digits : List ℕ := [6, 7, 8, 9]

def is_valid_placement (a b c d : ℕ) : Prop :=
  a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

def product (a b c d : ℕ) : ℕ := (10 * a + b) * (10 * c + d)

theorem smallest_product :
  ∀ a b c d : ℕ, is_valid_placement a b c d →
  product a b c d ≥ 5372 :=
sorry

end NUMINAMATH_CALUDE_smallest_product_l1242_124207


namespace NUMINAMATH_CALUDE_largest_number_l1242_124266

theorem largest_number : 
  let a := 0.938
  let b := 0.9389
  let c := 0.93809
  let d := 0.839
  let e := 0.8909
  b > a ∧ b > c ∧ b > d ∧ b > e := by
  sorry

end NUMINAMATH_CALUDE_largest_number_l1242_124266


namespace NUMINAMATH_CALUDE_cooking_and_yoga_count_l1242_124257

/-- Represents the number of people in different curriculum groups -/
structure CurriculumGroups where
  yoga : ℕ
  cooking : ℕ
  weaving : ℕ
  cookingOnly : ℕ
  allCurriculums : ℕ
  cookingAndWeaving : ℕ

/-- Theorem stating the number of people who study both cooking and yoga -/
theorem cooking_and_yoga_count (g : CurriculumGroups) 
  (h1 : g.yoga = 25)
  (h2 : g.cooking = 15)
  (h3 : g.weaving = 8)
  (h4 : g.cookingOnly = 2)
  (h5 : g.allCurriculums = 3)
  (h6 : g.cookingAndWeaving = 3) :
  g.cooking - g.cookingOnly - g.cookingAndWeaving + g.allCurriculums = 10 := by
  sorry


end NUMINAMATH_CALUDE_cooking_and_yoga_count_l1242_124257


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_1_l1242_124209

theorem simplify_and_evaluate_1 (m : ℤ) :
  m = -2023 → 7 * m^2 + 4 - 2 * m^2 - 3 * m - 5 * m^2 - 5 + 4 * m = -2024 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_1_l1242_124209


namespace NUMINAMATH_CALUDE_one_meeting_before_first_lap_l1242_124226

/-- Represents a runner on a circular track -/
structure Runner where
  speed : ℝ
  direction : Bool  -- True for clockwise, False for counterclockwise

/-- Calculates the number of meetings between two runners on a circular track -/
def meetings (track_length : ℝ) (runner1 runner2 : Runner) : ℕ :=
  sorry

theorem one_meeting_before_first_lap (track_length : ℝ) (runner1 runner2 : Runner) :
  track_length = 190 →
  runner1.speed = 7 →
  runner2.speed = 12 →
  runner1.direction ≠ runner2.direction →
  meetings track_length runner1 runner2 = 1 :=
sorry

end NUMINAMATH_CALUDE_one_meeting_before_first_lap_l1242_124226


namespace NUMINAMATH_CALUDE_undefined_expression_expression_undefined_at_nine_l1242_124289

theorem undefined_expression (x : ℝ) : 
  (x^2 - 18*x + 81 = 0) ↔ (x = 9) := by sorry

theorem expression_undefined_at_nine : 
  ∃! x : ℝ, x^2 - 18*x + 81 = 0 := by sorry

end NUMINAMATH_CALUDE_undefined_expression_expression_undefined_at_nine_l1242_124289


namespace NUMINAMATH_CALUDE_quadratic_maximum_value_l1242_124242

theorem quadratic_maximum_value :
  ∃ (max : ℝ), max = 111 / 4 ∧ ∀ (x : ℝ), -3 * x^2 + 15 * x + 9 ≤ max :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_maximum_value_l1242_124242


namespace NUMINAMATH_CALUDE_halloween_goodie_bags_cost_l1242_124286

/-- Represents the minimum cost to purchase Halloween goodie bags --/
def minimum_cost (total_students : ℕ) (vampire_students : ℕ) (pumpkin_students : ℕ) 
  (package_size : ℕ) (package_cost : ℚ) (individual_cost : ℚ) : ℚ :=
  let vampire_packages := (vampire_students + package_size - 1) / package_size
  let pumpkin_packages := pumpkin_students / package_size
  let pumpkin_individual := pumpkin_students % package_size
  let base_cost := (vampire_packages + pumpkin_packages) * package_cost + 
                   pumpkin_individual * individual_cost
  let discounted_cost := if base_cost > 10 then base_cost * (1 - 0.1) else base_cost
  ⌈discounted_cost * 100⌉ / 100

/-- Theorem stating the minimum cost for Halloween goodie bags --/
theorem halloween_goodie_bags_cost :
  minimum_cost 25 11 14 5 3 1 = 14.4 := by
  sorry

end NUMINAMATH_CALUDE_halloween_goodie_bags_cost_l1242_124286


namespace NUMINAMATH_CALUDE_tangent_and_normal_equations_l1242_124229

-- Define the function f(x) = x^3
def f (x : ℝ) : ℝ := x^3

-- Define the point M₀
def M₀ : ℝ × ℝ := (2, 8)

-- Theorem statement
theorem tangent_and_normal_equations :
  let (x₀, y₀) := M₀
  let f' := λ x => 3 * x^2  -- Derivative of f
  let m_tangent := f' x₀    -- Slope of tangent line
  let m_normal := -1 / m_tangent  -- Slope of normal line
  -- Equation of tangent line
  (∀ x y, 12 * x - y - 16 = 0 ↔ y - y₀ = m_tangent * (x - x₀)) ∧
  -- Equation of normal line
  (∀ x y, x + 12 * y - 98 = 0 ↔ y - y₀ = m_normal * (x - x₀)) :=
by sorry

end NUMINAMATH_CALUDE_tangent_and_normal_equations_l1242_124229


namespace NUMINAMATH_CALUDE_inequality_proof_l1242_124205

theorem inequality_proof (x y : ℝ) (h : x^8 + y^8 ≤ 1) :
  x^12 - y^12 + 2*x^6*y^6 ≤ π/2 := by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1242_124205


namespace NUMINAMATH_CALUDE_rowers_who_voted_l1242_124246

theorem rowers_who_voted (num_coaches : ℕ) (votes_per_rower : ℕ) (votes_per_coach : ℕ) : 
  num_coaches = 36 → votes_per_rower = 3 → votes_per_coach = 5 → 
  (num_coaches * votes_per_coach) / votes_per_rower = 60 := by
  sorry

end NUMINAMATH_CALUDE_rowers_who_voted_l1242_124246


namespace NUMINAMATH_CALUDE_kerrys_age_l1242_124287

/-- Given Kerry's birthday celebration setup, prove his age. -/
theorem kerrys_age :
  ∀ (num_cakes : ℕ) 
    (candles_per_box : ℕ) 
    (cost_per_box : ℚ) 
    (total_cost : ℚ),
  num_cakes = 3 →
  candles_per_box = 12 →
  cost_per_box = 5/2 →
  total_cost = 5 →
  ∃ (age : ℕ),
    age * num_cakes = (total_cost / cost_per_box) * candles_per_box ∧
    age = 8 := by
sorry

end NUMINAMATH_CALUDE_kerrys_age_l1242_124287


namespace NUMINAMATH_CALUDE_negation_equivalence_l1242_124204

theorem negation_equivalence :
  (¬ ∃ x : ℝ, (2 : ℝ) ^ x < x ^ 2) ↔ (∀ x : ℝ, (2 : ℝ) ^ x ≥ x ^ 2) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1242_124204


namespace NUMINAMATH_CALUDE_max_value_xyz_l1242_124222

theorem max_value_xyz (x y z : ℝ) 
  (sum_zero : x + y + z = 0) 
  (sum_squares_six : x^2 + y^2 + z^2 = 6) : 
  x^2*y + y^2*z + z^2*x ≤ 6 := by
  sorry

end NUMINAMATH_CALUDE_max_value_xyz_l1242_124222


namespace NUMINAMATH_CALUDE_max_EH_value_l1242_124258

/-- A cyclic quadrilateral with integer side lengths --/
structure CyclicQuadrilateral where
  EF : ℕ
  FG : ℕ
  GH : ℕ
  EH : ℕ
  distinct : EF ≠ FG ∧ EF ≠ GH ∧ EF ≠ EH ∧ FG ≠ GH ∧ FG ≠ EH ∧ GH ≠ EH
  less_than_20 : EF < 20 ∧ FG < 20 ∧ GH < 20 ∧ EH < 20
  cyclic_property : EF * GH = FG * EH

/-- The maximum possible value of EH in a cyclic quadrilateral with given constraints --/
theorem max_EH_value (q : CyclicQuadrilateral) :
  (∀ q' : CyclicQuadrilateral, q'.EH ≤ q.EH) → q.EH^2 = 394 :=
sorry

end NUMINAMATH_CALUDE_max_EH_value_l1242_124258


namespace NUMINAMATH_CALUDE_fraction_inequality_l1242_124261

theorem fraction_inequality (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > 0) :
  (b + c) / (a + c) > b / a :=
by sorry

end NUMINAMATH_CALUDE_fraction_inequality_l1242_124261


namespace NUMINAMATH_CALUDE_power_function_quadrants_l1242_124227

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, ∀ x : ℝ, f x = x ^ a

-- Define the condition f(1/3) = 9
def satisfiesCondition (f : ℝ → ℝ) : Prop :=
  f (1/3) = 9

-- Define the property of being in first and second quadrants
def isInFirstAndSecondQuadrants (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x > 0 → f x > 0

-- Theorem statement
theorem power_function_quadrants (f : ℝ → ℝ) 
  (h1 : isPowerFunction f) 
  (h2 : satisfiesCondition f) : 
  isInFirstAndSecondQuadrants f :=
sorry

end NUMINAMATH_CALUDE_power_function_quadrants_l1242_124227


namespace NUMINAMATH_CALUDE_product_of_numbers_l1242_124260

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 60) (h2 : x - y = 10) : x * y = 875 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l1242_124260


namespace NUMINAMATH_CALUDE_tic_tac_toe_tournament_contradiction_l1242_124223

/-- Represents a single-elimination tournament -/
structure Tournament :=
  (participants : ℕ)

/-- Calculates the total number of matches in a single-elimination tournament -/
def total_matches (t : Tournament) : ℕ := t.participants - 1

/-- Represents the claims of some participants -/
structure Claims :=
  (num_claimants : ℕ)
  (matches_per_claimant : ℕ)

/-- Calculates the total number of matches implied by the claims -/
def implied_matches (c : Claims) : ℕ := c.num_claimants * c.matches_per_claimant / 2

theorem tic_tac_toe_tournament_contradiction (t : Tournament) (c : Claims) 
  (h1 : t.participants = 18)
  (h2 : c.num_claimants = 6)
  (h3 : c.matches_per_claimant = 4) :
  implied_matches c ≠ total_matches t :=
sorry

end NUMINAMATH_CALUDE_tic_tac_toe_tournament_contradiction_l1242_124223


namespace NUMINAMATH_CALUDE_sophomore_sample_size_l1242_124270

/-- Calculates the number of sophomores in a stratified sample -/
def sophomores_in_sample (total_students : ℕ) (total_sophomores : ℕ) (sample_size : ℕ) : ℕ :=
  (total_sophomores * sample_size) / total_students

theorem sophomore_sample_size :
  let total_students : ℕ := 4500
  let total_sophomores : ℕ := 1500
  let sample_size : ℕ := 600
  sophomores_in_sample total_students total_sophomores sample_size = 200 := by
  sorry

end NUMINAMATH_CALUDE_sophomore_sample_size_l1242_124270


namespace NUMINAMATH_CALUDE_square_perimeter_from_p_shape_l1242_124297

/-- Represents a rectangle with width and length -/
structure Rectangle where
  width : ℝ
  length : ℝ

/-- Represents a square -/
structure Square where
  side : ℝ

/-- Represents the P shape formed by rectangles -/
structure PShape where
  rectangles : Fin 4 → Rectangle

theorem square_perimeter_from_p_shape 
  (s : Square) 
  (p : PShape) 
  (h1 : ∀ i, p.rectangles i = ⟨s.side / 5, 4 * s.side / 5⟩) 
  (h2 : (6 * (4 * s.side / 5) + 4 * (s.side / 5) : ℝ) = 56) :
  4 * s.side = 40 := by
sorry

end NUMINAMATH_CALUDE_square_perimeter_from_p_shape_l1242_124297


namespace NUMINAMATH_CALUDE_q_gt_one_neither_sufficient_nor_necessary_l1242_124206

/-- A geometric sequence with common ratio q -/
def GeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = q * a n

/-- An increasing sequence -/
def IncreasingSequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

/-- Theorem: "q > 1" is neither sufficient nor necessary for a geometric sequence to be increasing -/
theorem q_gt_one_neither_sufficient_nor_necessary :
  ¬(∀ a : ℕ → ℝ, ∀ q : ℝ, GeometricSequence a q → (q > 1 → IncreasingSequence a)) ∧
  ¬(∀ a : ℕ → ℝ, ∀ q : ℝ, GeometricSequence a q → (IncreasingSequence a → q > 1)) :=
sorry

end NUMINAMATH_CALUDE_q_gt_one_neither_sufficient_nor_necessary_l1242_124206


namespace NUMINAMATH_CALUDE_gcd_10010_15015_l1242_124284

theorem gcd_10010_15015 : Nat.gcd 10010 15015 = 5005 := by
  sorry

end NUMINAMATH_CALUDE_gcd_10010_15015_l1242_124284


namespace NUMINAMATH_CALUDE_garden_area_l1242_124269

theorem garden_area (total_posts : ℕ) (post_distance : ℕ) (longer_side_posts : ℕ) (shorter_side_posts : ℕ) :
  total_posts = 24 →
  post_distance = 4 →
  longer_side_posts = 2 * shorter_side_posts →
  longer_side_posts + shorter_side_posts = total_posts + 4 →
  (shorter_side_posts - 1) * post_distance * (longer_side_posts - 1) * post_distance = 576 :=
by sorry

end NUMINAMATH_CALUDE_garden_area_l1242_124269


namespace NUMINAMATH_CALUDE_sum_of_coefficients_equals_value_at_one_sum_of_coefficients_is_eight_l1242_124267

/-- The polynomial in question -/
def p (x : ℝ) : ℝ := 2 * (4 * x^6 + 9 * x^3 - 5) + 8 * (x^4 - 8 * x + 6)

/-- The sum of coefficients of a polynomial is equal to its value at x = 1 -/
theorem sum_of_coefficients_equals_value_at_one :
  (p 1) = 8 := by sorry

/-- The sum of coefficients of the given polynomial is 8 -/
theorem sum_of_coefficients_is_eight :
  ∃ (f : ℝ → ℝ), (∀ x, f x = p x) ∧ (f 1 = 8) := by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_equals_value_at_one_sum_of_coefficients_is_eight_l1242_124267


namespace NUMINAMATH_CALUDE_cds_per_rack_l1242_124232

/-- Given a shelf that can hold 4 racks and 32 CDs, prove that each rack can hold 8 CDs. -/
theorem cds_per_rack (racks_per_shelf : ℕ) (cds_per_shelf : ℕ) (h1 : racks_per_shelf = 4) (h2 : cds_per_shelf = 32) :
  cds_per_shelf / racks_per_shelf = 8 := by
  sorry


end NUMINAMATH_CALUDE_cds_per_rack_l1242_124232


namespace NUMINAMATH_CALUDE_solve_equation_l1242_124288

-- Define the @ operation
def at_op (a b : ℝ) : ℝ := a * (b ^ (1/2))

-- Theorem statement
theorem solve_equation (x : ℝ) (h : at_op 4 x = 12) : x = 9 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1242_124288


namespace NUMINAMATH_CALUDE_mountain_trail_length_l1242_124274

/-- Represents the hike on the Mountain Trail -/
structure MountainTrail where
  -- Daily distances hiked
  day1 : ℝ
  day2 : ℝ
  day3 : ℝ
  day4 : ℝ
  day5 : ℝ
  -- Conditions
  first_two_days : day1 + day2 = 30
  second_third_avg : (day2 + day3) / 2 = 16
  last_three_days : day3 + day4 + day5 = 45
  first_fourth_days : day1 + day4 = 32

/-- The theorem stating the total length of the Mountain Trail -/
theorem mountain_trail_length (hike : MountainTrail) : 
  hike.day1 + hike.day2 + hike.day3 + hike.day4 + hike.day5 = 107 := by
  sorry


end NUMINAMATH_CALUDE_mountain_trail_length_l1242_124274


namespace NUMINAMATH_CALUDE_tangent_length_specific_circle_l1242_124256

/-- A circle passing through three points -/
structure Circle where
  p1 : ℝ × ℝ
  p2 : ℝ × ℝ
  p3 : ℝ × ℝ

/-- The length of the tangent from a point to a circle -/
def tangentLength (p : ℝ × ℝ) (c : Circle) : ℝ :=
  sorry  -- Definition omitted as it's not given in the problem conditions

/-- The theorem stating the length of the tangent from the origin to the specific circle -/
theorem tangent_length_specific_circle :
  let origin : ℝ × ℝ := (0, 0)
  let c : Circle := { p1 := (3, 4), p2 := (6, 8), p3 := (5, 13) }
  tangentLength origin c = 5 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_tangent_length_specific_circle_l1242_124256
