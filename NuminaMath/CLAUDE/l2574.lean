import Mathlib

namespace archery_competition_theorem_l2574_257441

/-- Represents the point system for the archery competition -/
def PointSystem : Fin 4 → ℕ
  | 0 => 11  -- 1st place
  | 1 => 7   -- 2nd place
  | 2 => 5   -- 3rd place
  | 3 => 2   -- 4th place

/-- Represents the participation counts for each place -/
structure Participation where
  first : ℕ
  second : ℕ
  third : ℕ
  fourth : ℕ

/-- Calculates the product of points based on participation -/
def pointProduct (p : Participation) : ℕ :=
  (PointSystem 0) ^ p.first * 
  (PointSystem 1) ^ p.second * 
  (PointSystem 2) ^ p.third * 
  (PointSystem 3) ^ p.fourth

/-- Calculates the total number of participations -/
def totalParticipations (p : Participation) : ℕ :=
  p.first + p.second + p.third + p.fourth

/-- Theorem: If the product of points is 38500, then the total participations is 7 -/
theorem archery_competition_theorem (p : Participation) :
  pointProduct p = 38500 → totalParticipations p = 7 := by
  sorry


end archery_competition_theorem_l2574_257441


namespace piece_in_313th_row_l2574_257439

/-- Represents a chessboard with pieces -/
structure Chessboard :=
  (size : ℕ)
  (pieces : ℕ)
  (symmetrical : Bool)

/-- Checks if a row contains a piece -/
def has_piece_in_row (board : Chessboard) (row : ℕ) : Prop :=
  sorry

theorem piece_in_313th_row (board : Chessboard) 
  (h1 : board.size = 625)
  (h2 : board.pieces = 1977)
  (h3 : board.symmetrical = true) :
  has_piece_in_row board 313 :=
sorry

end piece_in_313th_row_l2574_257439


namespace square_difference_equals_six_l2574_257417

theorem square_difference_equals_six (a b : ℝ) 
  (sum_eq : a + b = 2) 
  (diff_eq : a - b = 3) : 
  a^2 - b^2 = 6 := by
sorry

end square_difference_equals_six_l2574_257417


namespace elections_with_at_least_two_past_officers_l2574_257481

def total_candidates : ℕ := 20
def past_officers : ℕ := 10
def positions : ℕ := 6

def total_elections : ℕ := Nat.choose total_candidates positions

def elections_no_past_officers : ℕ := Nat.choose (total_candidates - past_officers) positions

def elections_one_past_officer : ℕ := 
  Nat.choose past_officers 1 * Nat.choose (total_candidates - past_officers) (positions - 1)

theorem elections_with_at_least_two_past_officers : 
  total_elections - elections_no_past_officers - elections_one_past_officer = 36030 := by
  sorry

end elections_with_at_least_two_past_officers_l2574_257481


namespace sum_five_consecutive_integers_l2574_257459

/-- Given a sequence of five consecutive integers with middle number m,
    prove that their sum is equal to 5m. -/
theorem sum_five_consecutive_integers (m : ℤ) : 
  (m - 2) + (m - 1) + m + (m + 1) + (m + 2) = 5 * m := by
  sorry

end sum_five_consecutive_integers_l2574_257459


namespace binomial_expansion_problem_l2574_257473

theorem binomial_expansion_problem (x y : ℝ) (n : ℕ) 
  (h1 : n * x^(n-1) * y = 240)
  (h2 : n * (n-1) / 2 * x^(n-2) * y^2 = 720)
  (h3 : n * (n-1) * (n-2) / 6 * x^(n-3) * y^3 = 1080) :
  x = 2 ∧ y = 3 ∧ n = 5 := by
sorry

end binomial_expansion_problem_l2574_257473


namespace complex_expression_equality_l2574_257460

theorem complex_expression_equality : 
  let a : ℂ := 3 + 2*I
  let b : ℂ := 2 - I
  3*a + 4*b = 17 + 2*I :=
by sorry

end complex_expression_equality_l2574_257460


namespace robin_oatmeal_cookies_l2574_257479

/-- Calculates the number of oatmeal cookies Robin had -/
def oatmeal_cookies (cookies_per_bag : ℕ) (chocolate_chip_cookies : ℕ) (baggies : ℕ) : ℕ :=
  cookies_per_bag * baggies - chocolate_chip_cookies

/-- Proves that Robin had 25 oatmeal cookies -/
theorem robin_oatmeal_cookies :
  oatmeal_cookies 6 23 8 = 25 := by
  sorry

end robin_oatmeal_cookies_l2574_257479


namespace factors_of_48_l2574_257413

def number_of_factors (n : Nat) : Nat :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

theorem factors_of_48 : number_of_factors 48 = 10 := by
  sorry

end factors_of_48_l2574_257413


namespace glasses_displayed_is_70_l2574_257401

/-- Represents the cupboard system with given capacities and a broken shelf --/
structure CupboardSystem where
  tall_capacity : ℕ
  wide_capacity : ℕ
  narrow_capacity : ℕ
  narrow_shelves : ℕ
  broken_shelves : ℕ

/-- Calculates the total number of glasses displayed in the cupboard system --/
def total_glasses_displayed (cs : CupboardSystem) : ℕ :=
  cs.tall_capacity + cs.wide_capacity + 
  (cs.narrow_capacity / cs.narrow_shelves) * (cs.narrow_shelves - cs.broken_shelves)

/-- Theorem stating that the total number of glasses displayed is 70 --/
theorem glasses_displayed_is_70 : ∃ (cs : CupboardSystem), 
  cs.tall_capacity = 20 ∧
  cs.wide_capacity = 2 * cs.tall_capacity ∧
  cs.narrow_capacity = 15 ∧
  cs.narrow_shelves = 3 ∧
  cs.broken_shelves = 1 ∧
  total_glasses_displayed cs = 70 := by
  sorry

end glasses_displayed_is_70_l2574_257401


namespace vector_coordinates_l2574_257471

/-- Given a vector a with magnitude √5 that is parallel to vector b=(1,2),
    prove that the coordinates of a are either (1,2) or (-1,-2) -/
theorem vector_coordinates (a b : ℝ × ℝ) : 
  (‖a‖ = Real.sqrt 5) → 
  (b = (1, 2)) → 
  (∃ (k : ℝ), a = k • b) → 
  (a = (1, 2) ∨ a = (-1, -2)) := by
  sorry

#check vector_coordinates

end vector_coordinates_l2574_257471


namespace hyperbola_sum_l2574_257469

/-- Given a hyperbola with center (2, 0), one focus at (2, 8), and one vertex at (2, 5),
    prove that h + k + a + b = 7 + √39, where (h, k) is the center, a is the distance
    from the center to a vertex, and b is derived from b^2 = c^2 - a^2, with c being
    the distance from the center to a focus. -/
theorem hyperbola_sum (h k a b c : ℝ) : 
  h = 2 ∧ k = 0 ∧ a = 5 ∧ c = 8 ∧ b^2 = c^2 - a^2 →
  h + k + a + b = 7 + Real.sqrt 39 := by
  sorry

end hyperbola_sum_l2574_257469


namespace stephanie_distance_l2574_257435

/-- Calculates the distance traveled given time and speed -/
def distance (time : ℝ) (speed : ℝ) : ℝ := time * speed

/-- Proves that running for 3 hours at 5 miles per hour results in a distance of 15 miles -/
theorem stephanie_distance :
  let time : ℝ := 3
  let speed : ℝ := 5
  distance time speed = 15 := by
  sorry

end stephanie_distance_l2574_257435


namespace black_ball_count_l2574_257468

theorem black_ball_count (total : ℕ) (red : ℕ) (white : ℕ) (black : ℕ)
  (h_total : total = red + white + black)
  (h_red_prob : (red : ℚ) / total = 42 / 100)
  (h_white_prob : (white : ℚ) / total = 28 / 100)
  (h_red_count : red = 21) :
  black = 15 := by
  sorry

end black_ball_count_l2574_257468


namespace matches_played_calculation_l2574_257440

/-- A football competition with a specific scoring system and number of matches --/
structure FootballCompetition where
  totalMatches : ℕ
  pointsForWin : ℕ
  pointsForDraw : ℕ
  pointsForLoss : ℕ

/-- A team's current state in the competition --/
structure TeamState where
  pointsScored : ℕ
  matchesPlayed : ℕ

/-- Theorem stating the number of matches played by the team --/
theorem matches_played_calculation (comp : FootballCompetition)
    (state : TeamState) (minWinsNeeded : ℕ) (targetPoints : ℕ) :
    comp.totalMatches = 20 ∧
    comp.pointsForWin = 3 ∧
    comp.pointsForDraw = 1 ∧
    comp.pointsForLoss = 0 ∧
    state.pointsScored = 14 ∧
    minWinsNeeded = 6 ∧
    targetPoints = 40 →
    state.matchesPlayed = 14 := by
  sorry

end matches_played_calculation_l2574_257440


namespace fourth_power_inequality_l2574_257456

theorem fourth_power_inequality (a b c : ℝ) :
  a^4 + b^4 + c^4 ≥ a*b*c*(a + b + c) := by
  sorry

end fourth_power_inequality_l2574_257456


namespace no_triple_perfect_squares_l2574_257485

theorem no_triple_perfect_squares (n : ℕ+) : 
  ¬(∃ a b c : ℕ, (2 * n.val^2 + 1 = a^2) ∧ (3 * n.val^2 + 1 = b^2) ∧ (6 * n.val^2 + 1 = c^2)) :=
by sorry

end no_triple_perfect_squares_l2574_257485


namespace digit_222_of_55_div_777_l2574_257487

/-- The decimal representation of a rational number -/
def decimal_representation (n d : ℕ) : ℕ → ℕ :=
  sorry

/-- The length of the repeating block in the decimal representation of a rational number -/
def repeating_block_length (n d : ℕ) : ℕ :=
  sorry

theorem digit_222_of_55_div_777 :
  decimal_representation 55 777 222 = 7 :=
sorry

end digit_222_of_55_div_777_l2574_257487


namespace johns_initial_money_l2574_257466

theorem johns_initial_money (M : ℝ) : 
  (M > 0) →
  ((1 - 1/5) * M * (1 - 3/4) = 4) →
  (M = 20) := by
sorry

end johns_initial_money_l2574_257466


namespace compute_expression_l2574_257415

theorem compute_expression : 9 * (2 / 7 : ℚ)^4 = 144 / 2401 := by
  sorry

end compute_expression_l2574_257415


namespace average_salary_increase_proof_l2574_257449

def average_salary_increase 
  (initial_employees : ℕ) 
  (initial_average_salary : ℚ) 
  (manager_salary : ℚ) : ℚ :=
  let total_initial_salary := initial_employees * initial_average_salary
  let new_total_salary := total_initial_salary + manager_salary
  let new_average_salary := new_total_salary / (initial_employees + 1)
  new_average_salary - initial_average_salary

theorem average_salary_increase_proof :
  average_salary_increase 24 1500 11500 = 400 := by
  sorry

end average_salary_increase_proof_l2574_257449


namespace percent_of_percent_l2574_257445

theorem percent_of_percent (y : ℝ) (h : y ≠ 0) :
  (0.6 * (0.3 * y)) / y * 100 = 18 := by
  sorry

end percent_of_percent_l2574_257445


namespace gary_money_after_sale_l2574_257425

theorem gary_money_after_sale (initial_amount selling_price : ℝ) 
  (h1 : initial_amount = 73.0) 
  (h2 : selling_price = 55.0) : 
  initial_amount + selling_price = 128.0 :=
by sorry

end gary_money_after_sale_l2574_257425


namespace absent_children_count_l2574_257442

/-- Proves that the number of absent children is 32 given the conditions of the sweet distribution problem --/
theorem absent_children_count (total_children : ℕ) (original_sweets_per_child : ℕ) (extra_sweets : ℕ) : 
  total_children = 112 →
  original_sweets_per_child = 15 →
  extra_sweets = 6 →
  (total_children - (total_children - 32)) * (original_sweets_per_child + extra_sweets) = total_children * original_sweets_per_child :=
by sorry

end absent_children_count_l2574_257442


namespace ten_dollar_bill_count_l2574_257499

/-- Represents the number of bills of a certain denomination in a wallet. -/
structure BillCount where
  fives : Nat
  tens : Nat
  twenties : Nat

/-- Calculates the total amount in the wallet given the bill counts. -/
def totalAmount (bills : BillCount) : Nat :=
  5 * bills.fives + 10 * bills.tens + 20 * bills.twenties

/-- Theorem stating that given the conditions, there are 2 $10 bills in the wallet. -/
theorem ten_dollar_bill_count : ∃ (bills : BillCount), 
  bills.fives = 4 ∧ 
  bills.twenties = 3 ∧ 
  totalAmount bills = 100 ∧ 
  bills.tens = 2 := by
  sorry

end ten_dollar_bill_count_l2574_257499


namespace min_value_of_expression_l2574_257486

theorem min_value_of_expression (a b : ℝ) 
  (h1 : a > 0) 
  (h2 : b < 0) 
  (h3 : a - b = 5) : 
  ∃ (m : ℝ), m = 1/2 ∧ ∀ x, x = 1/(a+1) + 1/(2-b) → x ≥ m :=
sorry

end min_value_of_expression_l2574_257486


namespace third_term_base_l2574_257429

theorem third_term_base (h a b c : ℕ+) (base : ℕ+) : 
  (225 ∣ h) → 
  (216 ∣ h) → 
  h = 2^(a.val) * 3^(b.val) * base^(c.val) →
  a.val + b.val + c.val = 8 →
  base = 5 := by sorry

end third_term_base_l2574_257429


namespace cricket_team_handedness_l2574_257491

theorem cricket_team_handedness (total_players : ℕ) (throwers : ℕ) (right_handed : ℕ) 
  (h1 : total_players = 67)
  (h2 : throwers = 37)
  (h3 : right_handed = 57)
  (h4 : throwers ≤ right_handed) :
  (total_players - throwers - (right_handed - throwers)) / (total_players - throwers) = 1 / 3 := by
sorry

end cricket_team_handedness_l2574_257491


namespace nut_distribution_l2574_257418

def distribute_nuts (total : ℕ) : ℕ × ℕ × ℕ × ℕ × ℕ := sorry

theorem nut_distribution (total : ℕ) :
  let (tamas, erzsi, bela, juliska, remaining) := distribute_nuts total
  (tamas + bela) - (erzsi + juliska) = 100 →
  total = 1021 ∧ remaining = 321 := by sorry

end nut_distribution_l2574_257418


namespace cafeteria_apples_l2574_257488

/-- The number of apples initially in the cafeteria -/
def initial_apples : ℕ := 50

/-- The number of oranges initially in the cafeteria -/
def initial_oranges : ℕ := 40

/-- The cost of an apple in dollars -/
def apple_cost : ℚ := 4/5

/-- The cost of an orange in dollars -/
def orange_cost : ℚ := 1/2

/-- The number of apples left after selling -/
def remaining_apples : ℕ := 10

/-- The number of oranges left after selling -/
def remaining_oranges : ℕ := 6

/-- The total earnings from selling apples and oranges in dollars -/
def total_earnings : ℚ := 49

theorem cafeteria_apples :
  apple_cost * (initial_apples - remaining_apples : ℚ) +
  orange_cost * (initial_oranges - remaining_oranges : ℚ) = total_earnings :=
by sorry

end cafeteria_apples_l2574_257488


namespace survey_mn_value_l2574_257404

/-- Proves that mn = 2.5 given the survey conditions --/
theorem survey_mn_value (total : ℕ) (table_tennis basketball soccer : ℕ) 
  (h1 : total = 100)
  (h2 : table_tennis = 40)
  (h3 : (table_tennis : ℚ) / total = 2/5)
  (h4 : (basketball : ℚ) / total = 1/4)
  (h5 : soccer = total - (table_tennis + basketball))
  (h6 : (soccer : ℚ) / total = (soccer : ℚ) / 100) :
  (basketball : ℚ) * ((soccer : ℚ) / 100) = 5/2 := by
  sorry


end survey_mn_value_l2574_257404


namespace sqrt_expressions_l2574_257409

-- Define the theorem
theorem sqrt_expressions :
  -- Part 1
  (∀ (a b m n : ℤ), a + b * Real.sqrt 3 = (m + n * Real.sqrt 3)^2 → 
    a = m^2 + 3*n^2 ∧ b = 2*m*n) ∧
  -- Part 2
  (∀ (a m n : ℕ+), a + 4 * Real.sqrt 3 = (m + n * Real.sqrt 3)^2 → 
    a = 13 ∨ a = 7) ∧
  -- Part 3
  Real.sqrt (6 + 2 * Real.sqrt 5) = 1 + Real.sqrt 5 := by
sorry


end sqrt_expressions_l2574_257409


namespace perfect_square_trinomial_l2574_257482

theorem perfect_square_trinomial 
  (a b c : ℤ) 
  (h : ∀ x : ℤ, ∃ y : ℤ, a * x^2 + b * x + c = y^2) :
  ∃ d e : ℤ, ∀ x : ℤ, a * x^2 + b * x + c = (d * x + e)^2 :=
sorry

end perfect_square_trinomial_l2574_257482


namespace dividend_rate_calculation_l2574_257477

/-- Dividend calculation problem -/
theorem dividend_rate_calculation
  (preferred_shares : ℕ)
  (common_shares : ℕ)
  (par_value : ℚ)
  (common_dividend_rate : ℚ)
  (total_annual_dividend : ℚ)
  (h1 : preferred_shares = 1200)
  (h2 : common_shares = 3000)
  (h3 : par_value = 50)
  (h4 : common_dividend_rate = 7/200)  -- 3.5% converted to a fraction
  (h5 : total_annual_dividend = 16500) :
  let preferred_dividend_rate := (total_annual_dividend - 2 * common_shares * par_value * common_dividend_rate) / (preferred_shares * par_value)
  preferred_dividend_rate = 1/10 := by sorry

end dividend_rate_calculation_l2574_257477


namespace product_sum_multiple_l2574_257498

theorem product_sum_multiple (a b m : ℤ) : 
  b = 7 → 
  b - a = 2 → 
  a * b = m * (a + b) + 11 → 
  m = 2 := by
sorry

end product_sum_multiple_l2574_257498


namespace derivative_at_pi_over_four_l2574_257433

open Real

theorem derivative_at_pi_over_four :
  let f (x : ℝ) := cos x * (sin x - cos x)
  let f' := deriv f
  f' (π / 4) = 1 := by
  sorry

end derivative_at_pi_over_four_l2574_257433


namespace sarahs_bowling_score_l2574_257483

theorem sarahs_bowling_score (greg_score sarah_score : ℕ) : 
  sarah_score = greg_score + 60 →
  (sarah_score + greg_score) / 2 = 108 →
  sarah_score = 138 := by
sorry

end sarahs_bowling_score_l2574_257483


namespace prime_consecutive_property_l2574_257438

theorem prime_consecutive_property (p : ℕ) (hp : Prime p) (hp2 : Prime (p + 2)) :
  p = 3 ∨ 6 ∣ (p + 1) :=
sorry

end prime_consecutive_property_l2574_257438


namespace prime_power_divisibility_l2574_257431

theorem prime_power_divisibility (p : ℕ) (x : ℕ) (h_prime : Nat.Prime p) :
  1 ≤ x ∧ x ≤ 2 * p ∧ (x ^ (p - 1) ∣ (p - 1) ^ x + 1) →
  (p = 2 ∧ (x = 1 ∨ x = 2)) ∨ (p = 3 ∧ (x = 1 ∨ x = 3)) ∨ x = 1 :=
by sorry

end prime_power_divisibility_l2574_257431


namespace son_score_calculation_l2574_257467

def father_score : ℕ := 48
def son_score_difference : ℕ := 8

theorem son_score_calculation (father_score : ℕ) (son_score_difference : ℕ) :
  father_score = 48 →
  son_score_difference = 8 →
  father_score / 2 - son_score_difference = 16 :=
by sorry

end son_score_calculation_l2574_257467


namespace inverse_sum_equals_target_l2574_257422

noncomputable def g (x : ℝ) : ℝ :=
  if x ≤ 2 then x + 3 else x^2 - 4*x + 5

noncomputable def g_inverse (y : ℝ) : ℝ :=
  if y ≤ 5 then y - 3 else 2 + Real.sqrt (y - 1)

theorem inverse_sum_equals_target : g_inverse 1 + g_inverse 6 + g_inverse 11 = 2 + Real.sqrt 5 + Real.sqrt 10 := by
  sorry

end inverse_sum_equals_target_l2574_257422


namespace age_difference_l2574_257494

/-- Given that the sum of X and Y is 12 years greater than the sum of Y and Z,
    prove that Z is 12 years younger than X. -/
theorem age_difference (X Y Z : ℕ) (h : X + Y = Y + Z + 12) : X - Z = 12 := by
  sorry

end age_difference_l2574_257494


namespace multiply_subtract_equal_compute_expression_l2574_257495

theorem multiply_subtract_equal (a b c : ℤ) : a * c - b * c = (a - b) * c := by sorry

theorem compute_expression : 45 * 1313 - 10 * 1313 = 45955 := by sorry

end multiply_subtract_equal_compute_expression_l2574_257495


namespace part1_part2_l2574_257474

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given condition c - b = 2b cos A -/
def satisfiesCondition (t : Triangle) : Prop :=
  t.c - t.b = 2 * t.b * Real.cos t.A

theorem part1 (t : Triangle) (h : satisfiesCondition t) 
    (ha : t.a = 2 * Real.sqrt 6) (hb : t.b = 3) : 
  t.c = 5 := by
  sorry

theorem part2 (t : Triangle) (h : satisfiesCondition t) 
    (hc : t.C = Real.pi / 2) : 
  t.B = Real.pi / 6 := by
  sorry

end part1_part2_l2574_257474


namespace trigonometric_ratio_equals_one_l2574_257452

theorem trigonometric_ratio_equals_one :
  (Real.cos (70 * π / 180) * Real.cos (10 * π / 180) + Real.cos (80 * π / 180) * Real.cos (20 * π / 180)) /
  (Real.cos (69 * π / 180) * Real.cos (9 * π / 180) + Real.cos (81 * π / 180) * Real.cos (21 * π / 180)) = 1 := by
  sorry

end trigonometric_ratio_equals_one_l2574_257452


namespace points_collinear_implies_a_equals_4_l2574_257411

-- Define the points
def A : ℝ × ℝ := (4, 3)
def B (a : ℝ) : ℝ × ℝ := (5, a)
def C : ℝ × ℝ := (6, 5)

-- Define collinearity
def collinear (p q r : ℝ × ℝ) : Prop :=
  (r.2 - q.2) * (q.1 - p.1) = (q.2 - p.2) * (r.1 - q.1)

-- Theorem statement
theorem points_collinear_implies_a_equals_4 (a : ℝ) :
  collinear A (B a) C → a = 4 := by
  sorry

end points_collinear_implies_a_equals_4_l2574_257411


namespace environmental_protection_contest_l2574_257470

theorem environmental_protection_contest (A B C : ℝ) 
  (hA : A = 3/4)
  (hAC : (1 - A) * (1 - C) = 1/12)
  (hBC : B * C = 1/4)
  (hIndep : ∀ X Y : ℝ, X * Y = X * Y) : 
  A * B * C + (1 - A) * B * C + A * (1 - B) * C + A * B * (1 - C) = 21/32 := by
  sorry

end environmental_protection_contest_l2574_257470


namespace calum_disco_ball_spending_l2574_257465

/-- Represents the problem of calculating the maximum amount Calum can spend on each disco ball. -/
theorem calum_disco_ball_spending (
  disco_ball_count : ℕ)
  (food_box_count : ℕ)
  (decoration_set_count : ℕ)
  (food_box_cost : ℚ)
  (decoration_set_cost : ℚ)
  (total_budget : ℚ)
  (disco_ball_budget_percentage : ℚ)
  (h1 : disco_ball_count = 4)
  (h2 : food_box_count = 10)
  (h3 : decoration_set_count = 20)
  (h4 : food_box_cost = 25)
  (h5 : decoration_set_cost = 10)
  (h6 : total_budget = 600)
  (h7 : disco_ball_budget_percentage = 0.3)
  : (total_budget * disco_ball_budget_percentage) / disco_ball_count = 45 := by
  sorry

end calum_disco_ball_spending_l2574_257465


namespace line_slope_theorem_l2574_257412

/-- Given a line with equation x = 5y + 5 passing through points (m, n) and (m + a, n + p),
    where p = 0.4, prove that a = 2. -/
theorem line_slope_theorem (m n a p : ℝ) : 
  p = 0.4 →
  m = 5 * n + 5 →
  (m + a) = 5 * (n + p) + 5 →
  a = 2 := by
  sorry

end line_slope_theorem_l2574_257412


namespace periodic_sine_condition_l2574_257424

/-- Given a function f(x) = 2sin(ωx - π/3), prove that
    "∀x∈ℝ, f(x+π)=f(x)" is a necessary but not sufficient condition for ω = 2 -/
theorem periodic_sine_condition (ω : ℝ) :
  let f : ℝ → ℝ := λ x ↦ 2 * Real.sin (ω * x - π / 3)
  (∀ x, f (x + π) = f x) → ω = 2 ∧
  ∃ ω', ω' ≠ 2 ∧ (∀ x, 2 * Real.sin (ω' * x - π / 3) = 2 * Real.sin (ω' * (x + π) - π / 3)) :=
by sorry

end periodic_sine_condition_l2574_257424


namespace expression_evaluation_l2574_257453

theorem expression_evaluation :
  let x : ℝ := 2
  let expr := (x^2 + x) / (x^2 - 2*x + 1) / ((2 / (x - 1)) - (1 / x))
  expr = 4 := by
sorry

end expression_evaluation_l2574_257453


namespace inequality_solution_l2574_257430

theorem inequality_solution (x : ℝ) :
  (1 / (x * (x + 1)) - 1 / ((x + 1) * (x + 3)) < 1 / 4) ↔
  (x < -3 ∨ (-1 < x ∧ x < 0) ∨ 1 < x) :=
sorry

end inequality_solution_l2574_257430


namespace complex_equation_solution_l2574_257414

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the equation that z satisfies
def satisfies_equation (z : ℂ) : Prop := (2 * i) / z = 1 - i

-- Theorem statement
theorem complex_equation_solution :
  ∀ z : ℂ, satisfies_equation z → z = -1 + i :=
by sorry

end complex_equation_solution_l2574_257414


namespace snow_probability_l2574_257423

theorem snow_probability (p : ℝ) (h : p = 3/4) : 
  1 - (1 - p)^4 = 255/256 := by
  sorry

end snow_probability_l2574_257423


namespace ratio_NBQ_ABQ_l2574_257484

-- Define the points
variable (A B C P Q N : Point)

-- Define the angles
def angle (P Q R : Point) : ℝ := sorry

-- BP and BQ divide ∠ABC into three equal parts
axiom divide_three_equal : angle A B P = angle P B Q ∧ angle P B Q = angle Q B C

-- BN bisects ∠QBP
axiom bisect_QBP : angle Q B N = angle N B P

-- Theorem to prove
theorem ratio_NBQ_ABQ : 
  (angle N B Q) / (angle A B Q) = 3 / 4 :=
sorry

end ratio_NBQ_ABQ_l2574_257484


namespace eiffel_tower_height_is_324_l2574_257463

/-- The height of the Burj Khalifa in meters -/
def burj_khalifa_height : ℝ := 830

/-- The difference in height between the Burj Khalifa and the Eiffel Tower in meters -/
def height_difference : ℝ := 506

/-- The height of the Eiffel Tower in meters -/
def eiffel_tower_height : ℝ := burj_khalifa_height - height_difference

/-- Proves that the height of the Eiffel Tower is 324 meters -/
theorem eiffel_tower_height_is_324 : eiffel_tower_height = 324 := by
  sorry

end eiffel_tower_height_is_324_l2574_257463


namespace right_triangle_hypotenuse_from_medians_l2574_257497

/-- A right triangle with specific median lengths has a hypotenuse of 3√51 -/
theorem right_triangle_hypotenuse_from_medians 
  (a b c : ℝ) 
  (right_triangle : a^2 + b^2 = c^2) 
  (median1 : (b^2 + (a/2)^2) = 7^2) 
  (median2 : (a^2 + (b/2)^2) = (3*Real.sqrt 13)^2) : 
  c = 3 * Real.sqrt 51 := by
  sorry


end right_triangle_hypotenuse_from_medians_l2574_257497


namespace greatest_integer_jo_l2574_257419

theorem greatest_integer_jo (n : ℕ) : n < 150 → 
  (∃ k : ℤ, n = 9 * k - 1) → 
  (∃ l : ℤ, n = 6 * l - 5) → 
  n ≤ 125 :=
by sorry

end greatest_integer_jo_l2574_257419


namespace frieda_corner_probability_l2574_257492

/-- Represents the different types of squares on the 4x4 grid -/
inductive GridSquare
| Corner
| Edge
| Center

/-- Represents the possible directions of movement -/
inductive Direction
| Up
| Down
| Left
| Right

/-- Represents the state of Frieda on the grid -/
structure FriedaState :=
(position : GridSquare)
(hops : Nat)

/-- The probability of reaching a corner square within n hops -/
def probability_reach_corner (n : Nat) (start : GridSquare) : Rat :=
sorry

/-- The main theorem stating the probability of reaching a corner within 5 hops -/
theorem frieda_corner_probability :
  probability_reach_corner 5 GridSquare.Edge = 299 / 1024 :=
sorry

end frieda_corner_probability_l2574_257492


namespace jackson_running_program_l2574_257462

/-- Calculates the final running distance after a given number of days,
    given an initial distance and daily increase. -/
def finalRunningDistance (initialDistance : ℝ) (dailyIncrease : ℝ) (days : ℕ) : ℝ :=
  initialDistance + dailyIncrease * (days - 1)

/-- Theorem stating that given the initial conditions of Jackson's running program,
    the final running distance on the last day is 16.5 miles. -/
theorem jackson_running_program :
  let initialDistance : ℝ := 3
  let dailyIncrease : ℝ := 0.5
  let programDays : ℕ := 28
  finalRunningDistance initialDistance dailyIncrease programDays = 16.5 := by
  sorry

end jackson_running_program_l2574_257462


namespace circle_diameter_from_area_l2574_257464

/-- The diameter of a circle with area 64π cm² is 16 cm. -/
theorem circle_diameter_from_area :
  ∀ (r : ℝ), r > 0 → π * r^2 = 64 * π → 2 * r = 16 := by
  sorry

end circle_diameter_from_area_l2574_257464


namespace tigrasha_first_snezhok_last_l2574_257410

-- Define the kittens
inductive Kitten : Type
| Chernysh : Kitten
| Tigrasha : Kitten
| Snezhok : Kitten
| Pushok : Kitten

-- Define the eating speed for each kitten
def eating_speed (k : Kitten) : ℕ :=
  match k with
  | Kitten.Chernysh => 2
  | Kitten.Tigrasha => 5
  | Kitten.Snezhok => 3
  | Kitten.Pushok => 4

-- Define the initial number of sausages (same for all kittens)
def initial_sausages : ℕ := 7

-- Define the time to finish eating for each kitten
def time_to_finish (k : Kitten) : ℚ :=
  (initial_sausages : ℚ) / (eating_speed k : ℚ)

-- Theorem statement
theorem tigrasha_first_snezhok_last :
  (∀ k : Kitten, k ≠ Kitten.Tigrasha → time_to_finish Kitten.Tigrasha ≤ time_to_finish k) ∧
  (∀ k : Kitten, k ≠ Kitten.Snezhok → time_to_finish k ≤ time_to_finish Kitten.Snezhok) :=
sorry

end tigrasha_first_snezhok_last_l2574_257410


namespace cardboard_box_square_cutout_l2574_257427

theorem cardboard_box_square_cutout (length width area : ℝ) 
  (h1 : length = 80)
  (h2 : width = 60)
  (h3 : area = 1500) :
  ∃ (x : ℝ), x > 0 ∧ x < 30 ∧ (length - 2*x) * (width - 2*x) = area ∧ x = 15 :=
sorry

end cardboard_box_square_cutout_l2574_257427


namespace middle_part_of_proportional_division_l2574_257400

theorem middle_part_of_proportional_division (total : ℚ) (a b c : ℚ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  total = 104 ∧ a = 2 ∧ b = 3/2 ∧ c = 1/2 →
  (b * total) / (a + b + c) = 39 := by
sorry

end middle_part_of_proportional_division_l2574_257400


namespace log_sequence_a_is_geometric_l2574_257444

def sequence_a : ℕ → ℝ
  | 0 => 2
  | n + 1 => (sequence_a n) ^ 2

theorem log_sequence_a_is_geometric :
  ∃ r : ℝ, ∀ n : ℕ, n > 0 → Real.log (sequence_a (n + 1)) = r * Real.log (sequence_a n) := by
  sorry

end log_sequence_a_is_geometric_l2574_257444


namespace square_root_sum_of_squares_l2574_257434

theorem square_root_sum_of_squares (x y : ℝ) : 
  (∃ (s : ℝ), s^2 = x - 2 ∧ (s = 2 ∨ s = -2)) →
  (2*x + y + 7)^(1/3) = 3 →
  ∃ (t : ℝ), t^2 = x^2 + y^2 ∧ (t = 10 ∨ t = -10) := by
  sorry

end square_root_sum_of_squares_l2574_257434


namespace base6_arithmetic_equality_l2574_257490

/-- Converts a base 6 number to base 10 --/
def base6ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a base 10 number to base 6 --/
def base10ToBase6 (n : ℕ) : ℕ := sorry

theorem base6_arithmetic_equality :
  base10ToBase6 ((base6ToBase10 45321 - base6ToBase10 23454) + base6ToBase10 14553) = 45550 := by
  sorry

end base6_arithmetic_equality_l2574_257490


namespace only_set_A_forms_triangle_l2574_257458

def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ triangle_inequality a b c

theorem only_set_A_forms_triangle :
  can_form_triangle 3 4 5 ∧
  ¬can_form_triangle 4 4 8 ∧
  ¬can_form_triangle 3 10 4 ∧
  ¬can_form_triangle 4 5 10 :=
sorry

end only_set_A_forms_triangle_l2574_257458


namespace existence_of_uv_l2574_257461

theorem existence_of_uv (m n X : ℕ) (hm : X ≥ m) (hn : X ≥ n) :
  ∃ u v : ℤ,
    (|u| + |v| > 0) ∧
    (|u| ≤ Real.sqrt X) ∧
    (|v| ≤ Real.sqrt X) ∧
    (0 ≤ m * u + n * v) ∧
    (m * u + n * v ≤ 2 * Real.sqrt X) := by
  sorry

end existence_of_uv_l2574_257461


namespace imaginary_part_of_z_l2574_257436

/-- The imaginary part of the complex number z = (1-i)/(2i) is equal to -1/2 -/
theorem imaginary_part_of_z (i : ℂ) (h : i^2 = -1) : 
  Complex.im ((1 - i) / (2 * i)) = -1/2 := by
  sorry

end imaginary_part_of_z_l2574_257436


namespace number_pairing_l2574_257402

theorem number_pairing (numbers : List ℕ) (h1 : numbers = [41, 35, 19, 9, 26, 45, 13, 28]) :
  let total_sum := numbers.sum
  let pair_sum := total_sum / 4
  ∃ (pairs : List (ℕ × ℕ)), 
    (∀ p ∈ pairs, p.1 + p.2 = pair_sum) ∧ 
    (∀ n ∈ numbers, ∃ p ∈ pairs, n = p.1 ∨ n = p.2) ∧
    (∃ p ∈ pairs, p = (13, 41) ∨ p = (41, 13)) :=
by sorry

end number_pairing_l2574_257402


namespace arithmetic_sequence_sum_l2574_257428

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem statement -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 3)^2 - 3*(a 3) + 1 = 0 →
  (a 7)^2 - 3*(a 7) + 1 = 0 →
  a 4 + a 6 = 3 := by
  sorry

end arithmetic_sequence_sum_l2574_257428


namespace quadratic_function_zeros_l2574_257437

theorem quadratic_function_zeros (a : ℝ) :
  (∃ x y : ℝ, x > 2 ∧ y < -1 ∧
   -x^2 + a*x + 4 = 0 ∧
   -y^2 + a*y + 4 = 0) →
  0 < a ∧ a < 3 :=
by sorry


end quadratic_function_zeros_l2574_257437


namespace honey_jars_needed_l2574_257450

theorem honey_jars_needed (num_hives : ℕ) (honey_per_hive : ℝ) (jar_capacity : ℝ) 
  (h1 : num_hives = 5)
  (h2 : honey_per_hive = 20)
  (h3 : jar_capacity = 0.5)
  (h4 : jar_capacity > 0) :
  ⌈(↑num_hives * honey_per_hive / 2) / jar_capacity⌉ = 100 := by
  sorry

end honey_jars_needed_l2574_257450


namespace computers_produced_per_month_l2574_257476

/-- Represents the number of computers produced per 30-minute interval -/
def computers_per_interval : ℕ := 4

/-- Represents the number of days in a month -/
def days_per_month : ℕ := 28

/-- Represents the number of 30-minute intervals in a day -/
def intervals_per_day : ℕ := 48

/-- Calculates the total number of computers produced in a month -/
def computers_per_month : ℕ :=
  computers_per_interval * days_per_month * intervals_per_day

/-- Theorem stating that the number of computers produced per month is 5376 -/
theorem computers_produced_per_month :
  computers_per_month = 5376 := by sorry

end computers_produced_per_month_l2574_257476


namespace f_4_1981_l2574_257447

/-- Definition of the function f satisfying the given conditions -/
def f : ℕ → ℕ → ℕ
| 0, y => y + 1
| x + 1, 0 => f x 1
| x + 1, y + 1 => f x (f (x + 1) y)

/-- Theorem stating that f(4, 1981) equals 2^1984 - 3 -/
theorem f_4_1981 : f 4 1981 = 2^1984 - 3 := by
  sorry

end f_4_1981_l2574_257447


namespace delta_problem_l2574_257446

-- Define the delta operation
def delta (a b : ℕ) : ℕ := a^2 - b

-- State the theorem
theorem delta_problem : delta (5^(delta 7 2)) (4^(delta 3 8)) = 5^94 - 4 := by
  sorry

end delta_problem_l2574_257446


namespace larger_integer_value_l2574_257408

theorem larger_integer_value (a b : ℕ+) 
  (h_quotient : (a : ℚ) / (b : ℚ) = 7 / 3)
  (h_product : (a : ℕ) * b = 189) : 
  max a b = 21 := by
  sorry

end larger_integer_value_l2574_257408


namespace quadratic_inequality_range_l2574_257403

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, x^2 + 2*a*x + a > 0) → (0 < a ∧ a < 1) := by sorry

end quadratic_inequality_range_l2574_257403


namespace total_checks_is_30_l2574_257421

/-- The number of $50 checks -/
def F : ℕ := sorry

/-- The number of $100 checks -/
def H : ℕ := sorry

/-- The total worth of all checks is $1800 -/
axiom total_worth : 50 * F + 100 * H = 1800

/-- The average of remaining checks after removing 18 $50 checks is $75 -/
axiom remaining_average : (1800 - 18 * 50) / (F + H - 18) = 75

/-- The total number of travelers checks -/
def total_checks : ℕ := F + H

/-- Theorem: The total number of travelers checks is 30 -/
theorem total_checks_is_30 : total_checks = 30 := by sorry

end total_checks_is_30_l2574_257421


namespace group_element_identity_l2574_257416

theorem group_element_identity (G : Type) [Group G] (a b : G) 
  (h1 : a * b^2 = b^3 * a) (h2 : b * a^2 = a^3 * b) : a = 1 ∧ b = 1 := by
  sorry

end group_element_identity_l2574_257416


namespace triangle_area_theorem_l2574_257455

def triangle_area (r R : ℝ) (cosA cosB cosC : ℝ) (a b c : ℝ) : Prop :=
  r = 7 ∧
  R = 20 ∧
  3 * cosB = 2 * cosA + cosC ∧
  cosA + cosB + cosC = 1 + r / R ∧
  b = 2 * R * Real.sqrt (1 - cosB^2) ∧
  a^2 + c^2 - a * c * cosB = b^2 ∧
  cosA = (b^2 + c^2 - a^2) / (2 * b * c) ∧
  cosC = (a^2 + b^2 - c^2) / (2 * a * b) ∧
  (7 * (a + c + 2 * Real.sqrt 319)) / 2 = 7 * ((a + b + c) / 2)

theorem triangle_area_theorem :
  ∀ (r R : ℝ) (cosA cosB cosC : ℝ) (a b c : ℝ),
    triangle_area r R cosA cosB cosC a b c →
    (7 * (a + c + 2 * Real.sqrt 319)) / 2 = 7 * ((a + b + c) / 2) :=
by sorry

end triangle_area_theorem_l2574_257455


namespace decryption_works_l2574_257475

-- Define the Russian alphabet (excluding 'ё')
def russian_alphabet : List Char := ['а', 'б', 'в', 'г', 'д', 'е', 'ж', 'з', 'и', 'й', 'к', 'л', 'м', 'н', 'о', 'п', 'р', 'с', 'т', 'у', 'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ы', 'ь', 'э', 'ю', 'я']

-- Define the encryption steps
def swap_adjacent (s : String) : String := sorry

def shift_right (s : String) (n : Nat) : String := sorry

def reverse_string (s : String) : String := sorry

-- Define the decryption steps
def shift_left (s : String) (n : Nat) : String := sorry

-- Define the full encryption and decryption processes
def encrypt (s : String) : String :=
  reverse_string (shift_right (swap_adjacent s) 2)

def decrypt (s : String) : String :=
  swap_adjacent (shift_left (reverse_string s) 2)

-- Theorem to prove
theorem decryption_works (encrypted : String) (decrypted : String) :
  encrypted = "врпвл терпраиэ вйзгцфпз" ∧ 
  decrypted = "нефте базы южного района" →
  decrypt encrypted = decrypted := by sorry

end decryption_works_l2574_257475


namespace stock_price_calculation_l2574_257420

theorem stock_price_calculation (initial_price : ℝ) (first_year_increase : ℝ) (second_year_decrease : ℝ) : 
  initial_price = 100 ∧ 
  first_year_increase = 1.5 ∧ 
  second_year_decrease = 0.4 → 
  initial_price * (1 + first_year_increase) * (1 - second_year_decrease) = 150 := by
sorry

end stock_price_calculation_l2574_257420


namespace sara_movie_expenses_l2574_257432

/-- The total amount Sara spent on movies -/
def total_spent (ticket_price : ℚ) (num_tickets : ℕ) (rental_price : ℚ) (purchase_price : ℚ) : ℚ :=
  ticket_price * num_tickets + rental_price + purchase_price

/-- Theorem stating the total amount Sara spent on movies -/
theorem sara_movie_expenses :
  let ticket_price : ℚ := 10.62
  let num_tickets : ℕ := 2
  let rental_price : ℚ := 1.59
  let purchase_price : ℚ := 13.95
  total_spent ticket_price num_tickets rental_price purchase_price = 36.78 := by
  sorry

end sara_movie_expenses_l2574_257432


namespace only_first_proposition_correct_l2574_257454

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the necessary relations
variable (perpendicular : Line → Line → Prop)
variable (parallel : Line → Line → Prop)
variable (perpendicular_plane_line : Plane → Line → Prop)
variable (parallel_plane_line : Plane → Line → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)
variable (line_in_plane : Line → Plane → Prop)

-- State the theorem
theorem only_first_proposition_correct 
  (m l : Line) (α β : Plane) 
  (h_diff_lines : m ≠ l) 
  (h_diff_planes : α ≠ β) :
  ((perpendicular_plane_line α l ∧ parallel_plane_line α m → perpendicular l m) ∧
   ¬(parallel m l ∧ line_in_plane m α → parallel_plane_line α l) ∧
   ¬(perpendicular_planes α β ∧ line_in_plane m α ∧ line_in_plane l β → perpendicular m l) ∧
   ¬(perpendicular m l ∧ line_in_plane m α ∧ line_in_plane l β → perpendicular_planes α β)) :=
by sorry

end only_first_proposition_correct_l2574_257454


namespace unique_two_digit_number_l2574_257480

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : Nat
  ones : Nat
  h_tens : tens ≥ 1 ∧ tens ≤ 9
  h_ones : ones ≥ 0 ∧ ones ≤ 9

/-- Converts a TwoDigitNumber to its decimal representation -/
def TwoDigitNumber.toNat (n : TwoDigitNumber) : Nat :=
  10 * n.tens + n.ones

/-- The sum of digits of a TwoDigitNumber -/
def TwoDigitNumber.digitSum (n : TwoDigitNumber) : Nat :=
  n.tens + n.ones

/-- The product of digits of a TwoDigitNumber -/
def TwoDigitNumber.digitProduct (n : TwoDigitNumber) : Nat :=
  n.tens * n.ones

theorem unique_two_digit_number :
  ∃! (n : TwoDigitNumber),
    (n.toNat / n.digitSum = 4 ∧ n.toNat % n.digitSum = 3) ∧
    (n.toNat / n.digitProduct = 3 ∧ n.toNat % n.digitProduct = 5) ∧
    n.toNat = 23 := by
  sorry

end unique_two_digit_number_l2574_257480


namespace log_xy_value_l2574_257478

-- Define a real-valued logarithm function
noncomputable def log : ℝ → ℝ := sorry

-- State the theorem
theorem log_xy_value (x y : ℝ) (h1 : log (x^2 * y^3) = 2) (h2 : log (x^3 * y^2) = 2) :
  log (x * y) = 4/5 := by sorry

end log_xy_value_l2574_257478


namespace exactly_one_defective_two_genuine_mutually_exclusive_not_contradictory_l2574_257406

/-- Represents the outcome of selecting two products -/
inductive SelectionOutcome
  | TwoGenuine
  | OneGenuineOneDefective
  | TwoDefective

/-- Represents the total number of products -/
def totalProducts : Nat := 5

/-- Represents the number of genuine products -/
def genuineProducts : Nat := 3

/-- Represents the number of defective products -/
def defectiveProducts : Nat := 2

/-- Checks if two events are mutually exclusive -/
def mutuallyExclusive (e1 e2 : Set SelectionOutcome) : Prop :=
  e1 ∩ e2 = ∅

/-- Checks if two events are not contradictory -/
def notContradictory (e1 e2 : Set SelectionOutcome) : Prop :=
  e1 ∪ e2 ≠ Set.univ

/-- The event of selecting exactly one defective product -/
def exactlyOneDefective : Set SelectionOutcome :=
  {SelectionOutcome.OneGenuineOneDefective}

/-- The event of selecting exactly two genuine products -/
def exactlyTwoGenuine : Set SelectionOutcome :=
  {SelectionOutcome.TwoGenuine}

/-- Theorem stating that exactly one defective and exactly two genuine are mutually exclusive but not contradictory -/
theorem exactly_one_defective_two_genuine_mutually_exclusive_not_contradictory :
  mutuallyExclusive exactlyOneDefective exactlyTwoGenuine ∧
  notContradictory exactlyOneDefective exactlyTwoGenuine :=
sorry

end exactly_one_defective_two_genuine_mutually_exclusive_not_contradictory_l2574_257406


namespace inequality_proof_l2574_257493

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : a + b + c = 1) : 
  a / (b + c^2) + b / (c + a^2) + c / (a + b^2) ≥ 9/4 := by
sorry

end inequality_proof_l2574_257493


namespace girls_count_l2574_257448

/-- Represents the number of students in a college -/
structure College where
  boys : ℕ
  girls : ℕ

/-- Theorem stating that given the conditions, the number of girls in the college is 160 -/
theorem girls_count (c : College) 
  (ratio : c.boys * 5 = c.girls * 8) 
  (total : c.boys + c.girls = 416) : 
  c.girls = 160 := by
  sorry

end girls_count_l2574_257448


namespace arrangement_counts_l2574_257426

def num_boys : ℕ := 4
def num_girls : ℕ := 3
def total_students : ℕ := num_boys + num_girls

def girls_not_adjacent : ℕ := sorry

def boys_adjacent : ℕ := sorry

def girl_A_not_left_B_not_right : ℕ := sorry

def girls_ABC_height_order : ℕ := sorry

theorem arrangement_counts :
  girls_not_adjacent = 1440 ∧
  boys_adjacent = 576 ∧
  girl_A_not_left_B_not_right = 3720 ∧
  girls_ABC_height_order = 840 := by sorry

end arrangement_counts_l2574_257426


namespace tenth_term_is_24_l2574_257407

/-- The sum of the first n terms of an arithmetic sequence -/
def sequence_sum (n : ℕ) : ℕ := n^2 + 5*n

/-- The nth term of the arithmetic sequence -/
def nth_term (n : ℕ) : ℕ := sequence_sum n - sequence_sum (n-1)

theorem tenth_term_is_24 : nth_term 10 = 24 := by
  sorry

end tenth_term_is_24_l2574_257407


namespace proposition_relationship_l2574_257472

theorem proposition_relationship :
  ∀ (p q : Prop),
  (p → q) →                        -- Proposition A: p is sufficient for q
  (p ↔ q) →                        -- Proposition B: p is necessary and sufficient for q
  ((p ↔ q) → (p → q)) ∧            -- Proposition A is necessary for Proposition B
  ¬((p → q) → (p ↔ q)) :=          -- Proposition A is not sufficient for Proposition B
by
  sorry

end proposition_relationship_l2574_257472


namespace max_log_sum_l2574_257496

theorem max_log_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 4*y = 40) :
  ∃ (max : ℝ), max = 2 ∧ ∀ (a b : ℝ), a > 0 → b > 0 → a + 4*b = 40 → Real.log a + Real.log b ≤ max :=
sorry

end max_log_sum_l2574_257496


namespace x_varies_as_four_thirds_power_of_z_l2574_257443

/-- If x varies as the fourth power of y, and y varies as the cube root of z,
    then x varies as the (4/3)th power of z. -/
theorem x_varies_as_four_thirds_power_of_z 
  (x y z : ℝ) 
  (hxy : ∃ (a : ℝ), x = a * y^4) 
  (hyz : ∃ (b : ℝ), y = b * z^(1/3)) :
  ∃ (c : ℝ), x = c * z^(4/3) := by
sorry

end x_varies_as_four_thirds_power_of_z_l2574_257443


namespace xy_sum_l2574_257405

theorem xy_sum (x y : ℕ+) (h : (2 * x - 5) * (2 * y - 5) = 25) :
  x + y = 10 ∨ x + y = 18 := by
  sorry

end xy_sum_l2574_257405


namespace hundred_million_composition_l2574_257489

-- Define the decimal counting system progression rate
def decimal_progression_rate : ℕ := 10

-- Define the units
def one_million : ℕ := 1000000
def ten_million : ℕ := 10000000
def hundred_million : ℕ := 100000000

-- Theorem statement
theorem hundred_million_composition :
  hundred_million = decimal_progression_rate * ten_million ∧
  hundred_million = (decimal_progression_rate * decimal_progression_rate) * one_million :=
by sorry

end hundred_million_composition_l2574_257489


namespace distinct_bracelets_count_l2574_257457

/-- Represents a bead color -/
inductive BeadColor
| Red
| Blue
| Purple

/-- Represents a bracelet as a circular arrangement of beads -/
def Bracelet := List BeadColor

/-- Checks if two bracelets are equivalent under rotation and reflection -/
def are_equivalent (b1 b2 : Bracelet) : Bool :=
  sorry

/-- Counts the number of beads of each color in a bracelet -/
def count_beads (b : Bracelet) : Nat × Nat × Nat :=
  sorry

/-- Generates all possible bracelets with 2 red, 2 blue, and 2 purple beads -/
def generate_bracelets : List Bracelet :=
  sorry

/-- Counts the number of distinct bracelets -/
def count_distinct_bracelets : Nat :=
  sorry

/-- Theorem: The number of distinct bracelets with 2 red, 2 blue, and 2 purple beads is 11 -/
theorem distinct_bracelets_count :
  count_distinct_bracelets = 11 := by
  sorry

end distinct_bracelets_count_l2574_257457


namespace paint_cost_exceeds_budget_l2574_257451

/-- Represents the paint requirements for a mansion --/
structure MansionPaint where
  bedroom_count : Nat
  bathroom_count : Nat
  kitchen_count : Nat
  living_room_count : Nat
  dining_room_count : Nat
  study_room_count : Nat
  bedroom_paint : Nat
  bathroom_paint : Nat
  kitchen_paint : Nat
  living_room_paint : Nat
  dining_room_paint : Nat
  study_room_paint : Nat
  colored_paint_price : Nat
  white_paint_can_size : Nat
  white_paint_can_price : Nat
  budget : Nat

/-- Calculates the total cost of paint for the mansion --/
def total_paint_cost (m : MansionPaint) : Nat :=
  let colored_paint_gallons := 
    m.bedroom_count * m.bedroom_paint +
    m.kitchen_count * m.kitchen_paint +
    m.living_room_count * m.living_room_paint +
    m.dining_room_count * m.dining_room_paint +
    m.study_room_count * m.study_room_paint
  let white_paint_gallons := m.bathroom_count * m.bathroom_paint
  let white_paint_cans := (white_paint_gallons + m.white_paint_can_size - 1) / m.white_paint_can_size
  colored_paint_gallons * m.colored_paint_price + white_paint_cans * m.white_paint_can_price

/-- Theorem stating that the total paint cost exceeds the budget --/
theorem paint_cost_exceeds_budget (m : MansionPaint) 
  (h : m = { bedroom_count := 5, bathroom_count := 10, kitchen_count := 1, 
             living_room_count := 2, dining_room_count := 1, study_room_count := 1,
             bedroom_paint := 3, bathroom_paint := 2, kitchen_paint := 4,
             living_room_paint := 6, dining_room_paint := 4, study_room_paint := 3,
             colored_paint_price := 18, white_paint_can_size := 3, 
             white_paint_can_price := 40, budget := 500 }) : 
  total_paint_cost m > m.budget := by
  sorry


end paint_cost_exceeds_budget_l2574_257451
