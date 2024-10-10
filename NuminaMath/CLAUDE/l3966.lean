import Mathlib

namespace sum_first_six_primes_mod_seventh_prime_l3966_396666

def first_seven_primes : List Nat := [2, 3, 5, 7, 11, 13, 17]

theorem sum_first_six_primes_mod_seventh_prime : 
  (List.sum (List.take 6 first_seven_primes)) % (List.get! first_seven_primes 6) = 7 := by
  sorry

end sum_first_six_primes_mod_seventh_prime_l3966_396666


namespace modulus_of_z_l3966_396696

theorem modulus_of_z (z : ℂ) (h : z * Complex.I = 3 + 4 * Complex.I) : Complex.abs z = 5 := by
  sorry

end modulus_of_z_l3966_396696


namespace family_celebration_attendees_l3966_396643

theorem family_celebration_attendees :
  ∀ (n : ℕ) (s : ℕ),
    s / n = n →
    (s - 29) / (n - 1) = n - 1 →
    n = 15 := by
  sorry

end family_celebration_attendees_l3966_396643


namespace winProbA_le_two_p_squared_l3966_396646

/-- A tennis game between players A and B where A wins a point with probability p ≤ 1/2 -/
structure TennisGame where
  /-- The probability of player A winning a point -/
  p : ℝ
  /-- The condition that p is at most 1/2 -/
  h_p_le_half : p ≤ 1/2

/-- The probability of player A winning the entire game -/
def winProbA (game : TennisGame) : ℝ :=
  sorry

/-- Theorem stating that the probability of A winning is at most 2p² -/
theorem winProbA_le_two_p_squared (game : TennisGame) :
  winProbA game ≤ 2 * game.p^2 := by
  sorry

end winProbA_le_two_p_squared_l3966_396646


namespace f_minimum_l3966_396664

noncomputable def f (a x : ℝ) : ℝ := x^2 + |x - a| - 1

theorem f_minimum (a : ℝ) : 
  (∀ x, f a x ≥ -a - 5/4 ∧ ∃ x, f a x = -a - 5/4) ∨
  (∀ x, f a x ≥ a^2 - 1 ∧ ∃ x, f a x = a^2 - 1) ∨
  (∀ x, f a x ≥ a - 5/4 ∧ ∃ x, f a x = a - 5/4) :=
by sorry

end f_minimum_l3966_396664


namespace real_part_of_z_l3966_396604

theorem real_part_of_z (z : ℂ) (h : (1 + 2*I)*z = 3 - 2*I) : 
  z.re = -1/5 := by
  sorry

end real_part_of_z_l3966_396604


namespace total_money_l3966_396673

theorem total_money (john emma lucas : ℚ) 
  (h1 : john = 4 / 5)
  (h2 : emma = 2 / 5)
  (h3 : lucas = 1 / 2) :
  john + emma + lucas = 17 / 10 := by
  sorry

end total_money_l3966_396673


namespace dance_attendance_l3966_396632

/-- The number of men at the dance -/
def num_men : ℕ := 15

/-- The number of women each man dances with -/
def dances_per_man : ℕ := 4

/-- The number of men each woman dances with -/
def dances_per_woman : ℕ := 3

/-- The number of women at the dance -/
def num_women : ℕ := num_men * dances_per_man / dances_per_woman

theorem dance_attendance : num_women = 20 := by
  sorry

end dance_attendance_l3966_396632


namespace quadratic_inequality_solution_l3966_396681

theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x : ℝ, ax^2 + bx + 2 > 0 ↔ -1/2 < x ∧ x < 1/3) → 
  a + b = -14 := by
sorry

end quadratic_inequality_solution_l3966_396681


namespace image_of_negative_one_two_l3966_396622

-- Define the set of real pairs
def RealPair := ℝ × ℝ

-- Define the mapping f
def f (p : RealPair) : RealPair :=
  (p.1 - p.2, p.1 + p.2)

-- Theorem statement
theorem image_of_negative_one_two :
  f (-1, 2) = (-3, 1) := by
  sorry

end image_of_negative_one_two_l3966_396622


namespace sun_op_example_l3966_396644

-- Define the ☼ operation
def sunOp (a b : ℚ) : ℚ := a^3 - 2*a*b + 4

-- Theorem statement
theorem sun_op_example : sunOp 4 (-9) = 140 := by sorry

end sun_op_example_l3966_396644


namespace cubic_factorization_l3966_396642

theorem cubic_factorization (x : ℝ) : x^3 - 5*x^2 + 4*x = x*(x-1)*(x-4) := by
  sorry

end cubic_factorization_l3966_396642


namespace marie_erasers_l3966_396699

theorem marie_erasers (initial : ℕ) (lost : ℕ) (final : ℕ) : 
  initial = 95 → lost = 42 → final = initial - lost → final = 53 := by
sorry

end marie_erasers_l3966_396699


namespace outfit_choices_l3966_396605

/-- The number of shirts, pants, and hats available -/
def num_items : ℕ := 8

/-- The number of colors available for each item -/
def num_colors : ℕ := 8

/-- The total number of possible outfits -/
def total_outfits : ℕ := num_items * num_items * num_items

/-- The number of outfits where all items are the same color -/
def mono_color_outfits : ℕ := num_colors

/-- The number of acceptable outfit choices -/
def acceptable_outfits : ℕ := total_outfits - mono_color_outfits

theorem outfit_choices : acceptable_outfits = 504 := by
  sorry

end outfit_choices_l3966_396605


namespace p_sufficient_not_necessary_l3966_396647

-- Define the conditions
def p (x : ℝ) : Prop := |x - 1| < 2
def q (x : ℝ) : Prop := x^2 - 5*x - 6 < 0

-- Theorem statement
theorem p_sufficient_not_necessary : 
  (∀ x : ℝ, p x → q x) ∧ 
  (∃ x : ℝ, q x ∧ ¬(p x)) := by sorry

end p_sufficient_not_necessary_l3966_396647


namespace no_solution_for_equation_l3966_396619

theorem no_solution_for_equation (x y z t : ℕ) : 3 * x^4 + 5 * y^4 + 7 * z^4 ≠ 11 * t^4 := by
  sorry

end no_solution_for_equation_l3966_396619


namespace combined_total_value_l3966_396600

/-- Represents the coin counts for a person -/
structure CoinCounts where
  pennies : ℕ
  nickels : ℕ
  dimes : ℕ
  quarters : ℕ
  halfDollars : ℕ
  dollarCoins : ℕ

/-- Calculates the total value of coins for a person -/
def totalValue (coins : CoinCounts) : ℕ :=
  coins.pennies * 1 +
  coins.nickels * 5 +
  coins.dimes * 10 +
  coins.quarters * 25 +
  coins.halfDollars * 50 +
  coins.dollarCoins * 100

/-- The coin counts for Kate -/
def kate : CoinCounts := {
  pennies := 223
  nickels := 156
  dimes := 87
  quarters := 25
  halfDollars := 7
  dollarCoins := 4
}

/-- The coin counts for John -/
def john : CoinCounts := {
  pennies := 388
  nickels := 94
  dimes := 105
  quarters := 45
  halfDollars := 15
  dollarCoins := 6
}

/-- The coin counts for Marie -/
def marie : CoinCounts := {
  pennies := 517
  nickels := 64
  dimes := 78
  quarters := 63
  halfDollars := 12
  dollarCoins := 9
}

/-- The coin counts for George -/
def george : CoinCounts := {
  pennies := 289
  nickels := 72
  dimes := 132
  quarters := 50
  halfDollars := 4
  dollarCoins := 3
}

/-- Theorem stating that the combined total value of all coins is 16042 cents -/
theorem combined_total_value :
  totalValue kate + totalValue john + totalValue marie + totalValue george = 16042 := by
  sorry

end combined_total_value_l3966_396600


namespace total_coins_is_188_l3966_396655

/-- The number of US pennies turned in -/
def us_pennies : ℕ := 38

/-- The number of US nickels turned in -/
def us_nickels : ℕ := 27

/-- The number of US dimes turned in -/
def us_dimes : ℕ := 19

/-- The number of US quarters turned in -/
def us_quarters : ℕ := 24

/-- The number of US half-dollars turned in -/
def us_half_dollars : ℕ := 13

/-- The number of US one-dollar coins turned in -/
def us_one_dollar_coins : ℕ := 17

/-- The number of US two-dollar coins turned in -/
def us_two_dollar_coins : ℕ := 5

/-- The number of Australian fifty-cent coins turned in -/
def australian_fifty_cent_coins : ℕ := 4

/-- The number of Mexican one-Peso coins turned in -/
def mexican_one_peso_coins : ℕ := 12

/-- The number of Canadian loonies turned in -/
def canadian_loonies : ℕ := 3

/-- The number of British 20 pence coins turned in -/
def british_20_pence_coins : ℕ := 7

/-- The number of pre-1965 US dimes turned in -/
def pre_1965_us_dimes : ℕ := 6

/-- The number of post-2005 Euro two-euro coins turned in -/
def euro_two_euro_coins : ℕ := 5

/-- The number of Swiss 5 franc coins turned in -/
def swiss_5_franc_coins : ℕ := 8

/-- Theorem: The total number of coins turned in is 188 -/
theorem total_coins_is_188 :
  us_pennies + us_nickels + us_dimes + us_quarters + us_half_dollars +
  us_one_dollar_coins + us_two_dollar_coins + australian_fifty_cent_coins +
  mexican_one_peso_coins + canadian_loonies + british_20_pence_coins +
  pre_1965_us_dimes + euro_two_euro_coins + swiss_5_franc_coins = 188 := by
  sorry

end total_coins_is_188_l3966_396655


namespace infinite_nested_radical_twenty_l3966_396682

theorem infinite_nested_radical_twenty : ∃! (x : ℝ), x > 0 ∧ x = Real.sqrt (20 + x) ∧ x = 5 := by sorry

end infinite_nested_radical_twenty_l3966_396682


namespace tangent_circles_area_ratio_l3966_396663

/-- Regular hexagon with side length 2 -/
structure RegularHexagon :=
  (side_length : ℝ)
  (is_regular : side_length = 2)

/-- Circle tangent to two lines of the hexagon -/
structure TangentCircle (h : RegularHexagon) :=
  (radius : ℝ)
  (tangent_to_side : Bool)
  (tangent_to_ef : Bool)

/-- The ratio of areas of two tangent circles is 1 -/
theorem tangent_circles_area_ratio 
  (h : RegularHexagon) 
  (c1 c2 : TangentCircle h) 
  (h1 : c1.tangent_to_side = true) 
  (h2 : c2.tangent_to_side = true) 
  (h3 : c1.tangent_to_ef = true) 
  (h4 : c2.tangent_to_ef = true) : 
  (c2.radius^2) / (c1.radius^2) = 1 := by
  sorry

end tangent_circles_area_ratio_l3966_396663


namespace output_is_76_l3966_396669

def function_machine (input : ℕ) : ℕ :=
  let step1 := input * 3
  let step2 := if step1 ≤ 40 then step1 + 10 else step1 - 7
  step2 * 2

theorem output_is_76 : function_machine 15 = 76 := by
  sorry

end output_is_76_l3966_396669


namespace prob_different_colors_specific_l3966_396690

/-- The probability of drawing two chips of different colors with replacement -/
def prob_different_colors (blue red yellow : ℕ) : ℚ :=
  let total := blue + red + yellow
  let prob_blue := blue / total
  let prob_red := red / total
  let prob_yellow := yellow / total
  let prob_not_blue := (red + yellow) / total
  let prob_not_red := (blue + yellow) / total
  let prob_not_yellow := (blue + red) / total
  prob_blue * prob_not_blue + prob_red * prob_not_red + prob_yellow * prob_not_yellow

/-- Theorem stating the probability of drawing two chips of different colors -/
theorem prob_different_colors_specific : 
  prob_different_colors 7 5 4 = 83 / 128 := by
  sorry

end prob_different_colors_specific_l3966_396690


namespace car_wheels_count_l3966_396628

theorem car_wheels_count (num_cars : ℕ) (wheels_per_car : ℕ) (h1 : num_cars = 12) (h2 : wheels_per_car = 4) :
  num_cars * wheels_per_car = 48 := by
  sorry

end car_wheels_count_l3966_396628


namespace single_working_day_between_holidays_l3966_396680

def is_holiday (n : ℕ) : Prop := n % 6 = 0 ∨ Nat.Prime n

def working_day_between_holidays (n : ℕ) : Prop :=
  n > 1 ∧ n < 40 ∧ is_holiday (n - 1) ∧ ¬is_holiday n ∧ is_holiday (n + 1)

theorem single_working_day_between_holidays :
  ∃! n, working_day_between_holidays n :=
sorry

end single_working_day_between_holidays_l3966_396680


namespace school_classes_l3966_396631

theorem school_classes (s : ℕ) (h1 : s > 0) : 
  ∃ c : ℕ, c * s * (7 * 12) = 84 ∧ c = 1 :=
by
  sorry

end school_classes_l3966_396631


namespace sqrt_five_irrational_l3966_396686

theorem sqrt_five_irrational :
  ∀ (x : ℝ), x ^ 2 = 5 → ¬ (∃ (a b : ℤ), b ≠ 0 ∧ x = a / b) :=
by sorry

def zero_rational : ℚ := 0

def three_point_fourteen_rational : ℚ := 314 / 100

def negative_eight_sevenths_rational : ℚ := -8 / 7

#check sqrt_five_irrational
#check zero_rational
#check three_point_fourteen_rational
#check negative_eight_sevenths_rational

end sqrt_five_irrational_l3966_396686


namespace dissimilar_terms_eq_choose_l3966_396640

/-- The number of dissimilar terms in the expansion of (x + y + z)^8 -/
def dissimilar_terms : ℕ :=
  Nat.choose 10 2

/-- Theorem stating that the number of dissimilar terms in (x + y + z)^8 is equal to (10 choose 2) -/
theorem dissimilar_terms_eq_choose : dissimilar_terms = 45 := by
  sorry

end dissimilar_terms_eq_choose_l3966_396640


namespace adjacent_sum_divisible_by_four_l3966_396613

/-- Represents a 2006 × 2006 table filled with numbers from 1 to 2006² -/
def Table := Fin 2006 → Fin 2006 → Fin (2006^2)

/-- Checks if two positions in the table are adjacent -/
def adjacent (p q : Fin 2006 × Fin 2006) : Prop :=
  (p.1 = q.1 ∧ (p.2 = q.2 + 1 ∨ q.2 = p.2 + 1)) ∨
  (p.2 = q.2 ∧ (p.1 = q.1 + 1 ∨ q.1 = p.1 + 1)) ∨
  (p.1 = q.1 + 1 ∧ p.2 = q.2 + 1) ∨
  (q.1 = p.1 + 1 ∧ q.2 = p.2 + 1) ∨
  (p.1 = q.1 + 1 ∧ q.2 = p.2 + 1) ∨
  (q.1 = p.1 + 1 ∧ p.2 = q.2 + 1)

/-- The main theorem to be proved -/
theorem adjacent_sum_divisible_by_four (t : Table) :
  ∃ (p q : Fin 2006 × Fin 2006),
    adjacent p q ∧ (((t p.1 p.2).val + (t q.1 q.2).val + 2) % 4 = 0) := by
  sorry


end adjacent_sum_divisible_by_four_l3966_396613


namespace total_fishes_in_aquatic_reserve_l3966_396650

theorem total_fishes_in_aquatic_reserve (bodies_of_water : ℕ) (fishes_per_body : ℕ) 
  (h1 : bodies_of_water = 6) 
  (h2 : fishes_per_body = 175) : 
  bodies_of_water * fishes_per_body = 1050 := by
  sorry

end total_fishes_in_aquatic_reserve_l3966_396650


namespace max_value_expression_tightness_of_bound_l3966_396660

theorem max_value_expression (x y : ℝ) :
  (x + 3*y + 2) / Real.sqrt (2*x^2 + y^2 + 1) ≤ Real.sqrt 14 :=
sorry

theorem tightness_of_bound : 
  ∀ ε > 0, ∃ x y : ℝ, Real.sqrt 14 - (x + 3*y + 2) / Real.sqrt (2*x^2 + y^2 + 1) < ε :=
sorry

end max_value_expression_tightness_of_bound_l3966_396660


namespace tan_sum_thirteen_thirtytwo_l3966_396679

theorem tan_sum_thirteen_thirtytwo : 
  let tan13 := Real.tan (13 * π / 180)
  let tan32 := Real.tan (32 * π / 180)
  tan13 + tan32 + tan13 * tan32 = 1 := by
  sorry

end tan_sum_thirteen_thirtytwo_l3966_396679


namespace point_not_in_fourth_quadrant_l3966_396651

theorem point_not_in_fourth_quadrant (m : ℝ) :
  ¬(m > 0 ∧ m + 1 < 0) :=
by sorry

end point_not_in_fourth_quadrant_l3966_396651


namespace max_mondays_in_45_days_l3966_396674

/-- The maximum number of Mondays in 45 consecutive days -/
def max_mondays : ℕ := 7

/-- A function that returns the day number of the nth Monday in a sequence, 
    assuming the first day is a Monday -/
def monday_sequence (n : ℕ) : ℕ := 1 + 7 * n

theorem max_mondays_in_45_days : 
  (∃ (start : ℕ), ∀ (i : ℕ), i < max_mondays → 
    start + monday_sequence i ≤ 45) ∧ 
  (∀ (start : ℕ), ∃ (i : ℕ), i = max_mondays → 
    45 < start + monday_sequence i) :=
sorry

end max_mondays_in_45_days_l3966_396674


namespace not_all_squares_congruent_all_squares_equiangular_all_squares_rectangles_all_squares_regular_polygons_all_squares_similar_l3966_396694

/-- A square is a quadrilateral with four equal sides and four right angles. -/
structure Square where
  side_length : ℝ
  side_length_pos : side_length > 0

/-- Two squares are congruent if they have the same side length. -/
def congruent (s1 s2 : Square) : Prop :=
  s1.side_length = s2.side_length

/-- Theorem: Not all squares are congruent to each other. -/
theorem not_all_squares_congruent : ¬ ∀ (s1 s2 : Square), congruent s1 s2 := by
  sorry

/-- All squares are equiangular. -/
theorem all_squares_equiangular : True := by
  sorry

/-- All squares are rectangles. -/
theorem all_squares_rectangles : True := by
  sorry

/-- All squares are regular polygons. -/
theorem all_squares_regular_polygons : True := by
  sorry

/-- All squares are similar to each other. -/
theorem all_squares_similar : True := by
  sorry

end not_all_squares_congruent_all_squares_equiangular_all_squares_rectangles_all_squares_regular_polygons_all_squares_similar_l3966_396694


namespace reciprocals_not_arithmetic_sequence_l3966_396657

theorem reciprocals_not_arithmetic_sequence (a b c : ℝ) :
  a > 0 → b > 0 → c > 0 →
  (b - a ≠ 0) →
  (c - b = b - a) →
  ¬(1/b - 1/a = 1/c - 1/b) := by
sorry

end reciprocals_not_arithmetic_sequence_l3966_396657


namespace bean_feast_spending_l3966_396627

/-- The bean-feast spending problem -/
theorem bean_feast_spending
  (cobblers tailors hatters glovers : ℕ)
  (total_spent : ℕ)
  (h_cobblers : cobblers = 25)
  (h_tailors : tailors = 20)
  (h_hatters : hatters = 18)
  (h_glovers : glovers = 12)
  (h_total : total_spent = 133)  -- 133 shillings = £6 13s
  (h_cobbler_tailor : 5 * (cobblers : ℚ) = 4 * (tailors : ℚ))
  (h_tailor_hatter : 12 * (tailors : ℚ) = 9 * (hatters : ℚ))
  (h_hatter_glover : 6 * (hatters : ℚ) = 8 * (glovers : ℚ)) :
  ∃ (g h t c : ℚ),
    g = 21 ∧ h = 42 ∧ t = 35 ∧ c = 35 ∧
    g * glovers + h * hatters + t * tailors + c * cobblers = total_spent :=
by sorry


end bean_feast_spending_l3966_396627


namespace total_spider_legs_l3966_396671

/-- The number of legs a single spider has -/
def spider_legs : ℕ := 8

/-- The number of spiders in the group -/
def group_size : ℕ := spider_legs / 2 + 10

/-- The total number of spider legs in the group -/
def total_legs : ℕ := group_size * spider_legs

/-- Theorem stating that the total number of spider legs in the group is 112 -/
theorem total_spider_legs : total_legs = 112 := by
  sorry

end total_spider_legs_l3966_396671


namespace det_C_equals_2142_l3966_396684

theorem det_C_equals_2142 (A B C : Matrix (Fin 3) (Fin 3) ℝ) : 
  A = ![![3, 2, 5], ![0, 2, 8], ![4, 1, 7]] →
  B = ![![-2, 3, 4], ![-1, -3, 5], ![0, 4, 3]] →
  C = A * B →
  Matrix.det C = 2142 := by
  sorry

end det_C_equals_2142_l3966_396684


namespace principal_amount_proof_l3966_396638

/-- Proves that given the conditions of the problem, the principal amount is 1500 --/
theorem principal_amount_proof (P : ℝ) : 
  (P * 0.04 * 4 = P - 1260) → P = 1500 := by
  sorry

end principal_amount_proof_l3966_396638


namespace least_x_for_even_prime_l3966_396614

theorem least_x_for_even_prime (x : ℕ+) (p : ℕ) : 
  Nat.Prime p → (x.val : ℚ) / (11 * p) = 2 → x.val ≥ 44 :=
sorry

end least_x_for_even_prime_l3966_396614


namespace total_crayons_is_116_l3966_396688

/-- The total number of crayons Wanda, Dina, and Jacob have -/
def total_crayons (wanda_crayons dina_crayons : ℕ) : ℕ :=
  wanda_crayons + dina_crayons + (dina_crayons - 2)

/-- Theorem stating that the total number of crayons is 116 -/
theorem total_crayons_is_116 :
  total_crayons 62 28 = 116 := by
  sorry

end total_crayons_is_116_l3966_396688


namespace sqrt_2_simplest_l3966_396620

def is_simplest_quadratic_radical (x : ℝ) : Prop :=
  ∃ (y : ℝ), x = Real.sqrt y ∧ (∀ (z : ℝ), z ^ 2 = y → z = x)

theorem sqrt_2_simplest :
  is_simplest_quadratic_radical (Real.sqrt 2) ∧
  ¬is_simplest_quadratic_radical (3 ^ (1/3 : ℝ)) ∧
  ¬is_simplest_quadratic_radical (Real.sqrt (1/2)) ∧
  ¬is_simplest_quadratic_radical (Real.sqrt 16) :=
by sorry

end sqrt_2_simplest_l3966_396620


namespace bridge_length_calculation_l3966_396601

/-- Calculates the length of a bridge given train and crossing parameters. -/
theorem bridge_length_calculation (train_length : ℝ) (train_speed : ℝ) (crossing_time : ℝ) :
  train_length = 280 →
  train_speed = 18 →
  crossing_time = 20 →
  train_speed * crossing_time - train_length = 80 := by
  sorry

#check bridge_length_calculation

end bridge_length_calculation_l3966_396601


namespace line_through_points_slope_one_l3966_396641

/-- Given a line passing through points M(-2, m) and N(m, 4) with a slope of 1, prove that m = 1 -/
theorem line_through_points_slope_one (m : ℝ) : 
  (4 - m) / (m - (-2)) = 1 → m = 1 := by
  sorry

end line_through_points_slope_one_l3966_396641


namespace shirt_markup_proof_l3966_396689

/-- Proves that for a shirt with an initial price of $45 after an 80% markup from wholesale,
    increasing the price by $5 results in a 100% markup from the wholesale price. -/
theorem shirt_markup_proof (initial_price : ℝ) (initial_markup : ℝ) (price_increase : ℝ) :
  initial_price = 45 ∧
  initial_markup = 0.8 ∧
  price_increase = 5 →
  let wholesale_price := initial_price / (1 + initial_markup)
  let new_price := initial_price + price_increase
  (new_price - wholesale_price) / wholesale_price = 1 :=
by sorry

end shirt_markup_proof_l3966_396689


namespace temperature_conversion_l3966_396672

theorem temperature_conversion (t k : ℝ) : 
  t = 5 / 9 * (k - 32) → t = 20 → k = 68 := by
  sorry

end temperature_conversion_l3966_396672


namespace charity_ticket_revenue_l3966_396636

theorem charity_ticket_revenue 
  (total_tickets : ℕ) 
  (total_revenue : ℕ) 
  (full_price : ℕ) 
  (full_price_tickets : ℕ) 
  (half_price_tickets : ℕ) :
  total_tickets = 140 →
  total_revenue = 2001 →
  total_tickets = full_price_tickets + half_price_tickets →
  total_revenue = full_price * full_price_tickets + (full_price / 2) * half_price_tickets →
  full_price > 0 →
  full_price_tickets * full_price = 782 :=
by sorry

end charity_ticket_revenue_l3966_396636


namespace smallest_a_value_l3966_396623

/-- Given two quadratic equations with integer coefficients and integer roots less than -1,
    this theorem states that the smallest possible value for the constant term 'a' is 15. -/
theorem smallest_a_value (a b c : ℤ) : 
  (∃ x y : ℤ, x < -1 ∧ y < -1 ∧ x^2 + b*x + a = 0 ∧ y^2 + b*y + a = 0) →
  (∃ z w : ℤ, z < -1 ∧ w < -1 ∧ z^2 + c*z + a = 1 ∧ w^2 + c*w + a = 1) →
  (∀ a' : ℤ, (∃ b' c' : ℤ, 
    (∃ x y : ℤ, x < -1 ∧ y < -1 ∧ x^2 + b'*x + a' = 0 ∧ y^2 + b'*y + a' = 0) ∧
    (∃ z w : ℤ, z < -1 ∧ w < -1 ∧ z^2 + c'*z + a' = 1 ∧ w^2 + c'*w + a' = 1)) →
    a' ≥ 15) →
  a = 15 :=
by sorry

end smallest_a_value_l3966_396623


namespace high_jump_probabilities_l3966_396662

/-- Probability of success for athlete A -/
def pA : ℝ := 0.7

/-- Probability of success for athlete B -/
def pB : ℝ := 0.6

/-- Probability that athlete A succeeds on the third attempt for the first time -/
def prob_A_third : ℝ := (1 - pA) * (1 - pA) * pA

/-- Probability that at least one athlete succeeds in their first attempt -/
def prob_at_least_one : ℝ := 1 - (1 - pA) * (1 - pB)

/-- Probability that after two attempts each, A has exactly one more successful attempt than B -/
def prob_A_one_more : ℝ := 
  2 * pA * (1 - pA) * (1 - pB) * (1 - pB) + 
  pA * pA * 2 * pB * (1 - pB)

theorem high_jump_probabilities :
  prob_A_third = 0.063 ∧ 
  prob_at_least_one = 0.88 ∧ 
  prob_A_one_more = 0.3024 := by
  sorry

end high_jump_probabilities_l3966_396662


namespace lines_parallel_iff_x_eq_9_l3966_396625

/-- Two 2D vectors are parallel if one is a scalar multiple of the other -/
def are_parallel (v1 v2 : ℝ × ℝ) : Prop :=
  ∃ (c : ℝ), v1 = (c * v2.1, c * v2.2)

/-- Definition of the first line -/
def line1 (u : ℝ) : ℝ × ℝ := (1 + 6*u, 3 - 2*u)

/-- Definition of the second line -/
def line2 (x v : ℝ) : ℝ × ℝ := (-4 + x*v, 5 - 3*v)

/-- The theorem stating that the lines are parallel iff x = 9 -/
theorem lines_parallel_iff_x_eq_9 :
  ∀ x : ℝ, (∀ u v : ℝ, line1 u ≠ line2 x v) ↔ x = 9 :=
sorry

end lines_parallel_iff_x_eq_9_l3966_396625


namespace xiaochun_current_age_l3966_396697

-- Define Xiaochun's current age
def xiaochun_age : ℕ := sorry

-- Define Xiaochun's brother's current age
def brother_age : ℕ := sorry

-- Condition 1: Xiaochun's age is 18 years less than his brother's age
axiom age_difference : xiaochun_age = brother_age - 18

-- Condition 2: In 3 years, Xiaochun's age will be half of his brother's age
axiom future_age_relation : xiaochun_age + 3 = (brother_age + 3) / 2

-- Theorem to prove
theorem xiaochun_current_age : xiaochun_age = 15 := by sorry

end xiaochun_current_age_l3966_396697


namespace game_cost_l3966_396609

def initial_money : ℕ := 63
def toy_price : ℕ := 3
def toys_affordable : ℕ := 5

def remaining_money : ℕ := toy_price * toys_affordable

theorem game_cost : initial_money - remaining_money = 48 := by
  sorry

end game_cost_l3966_396609


namespace quadratic_function_properties_l3966_396607

/-- A quadratic function f(x) = 3ax^2 + 2bx + c satisfying certain conditions -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  sum_zero : a + b + c = 0
  f_zero_pos : c > 0
  f_one_pos : 3*a + 2*b + c > 0

/-- The main theorem about the properties of the quadratic function -/
theorem quadratic_function_properties (f : QuadraticFunction) :
  f.a > 0 ∧ -2 < f.b / f.a ∧ f.b / f.a < -1 ∧
  (∃ x y : ℝ, 0 < x ∧ x < y ∧ y < 1 ∧
    3*f.a*x^2 + 2*f.b*x + f.c = 0 ∧
    3*f.a*y^2 + 2*f.b*y + f.c = 0) :=
by sorry

end quadratic_function_properties_l3966_396607


namespace work_completion_time_l3966_396626

/-- Represents the time taken to complete a work -/
structure WorkTime where
  days : ℝ
  hours_per_day : ℝ := 24
  total_hours : ℝ := days * hours_per_day

/-- Represents the rate of work -/
def WorkRate := ℝ

/-- The problem setup -/
structure WorkProblem where
  total_work : ℝ
  a_alone_time : WorkTime
  ab_initial_time : WorkTime
  a_final_time : WorkTime
  ab_together_time : WorkTime

/-- The main theorem to prove -/
theorem work_completion_time 
  (w : WorkProblem)
  (h1 : w.a_alone_time.days = 20)
  (h2 : w.ab_initial_time.days = 10)
  (h3 : w.a_final_time.days = 15)
  (h4 : w.total_work = (w.ab_initial_time.days / w.ab_together_time.days + 
                        w.a_final_time.days / w.a_alone_time.days) * w.total_work) :
  w.ab_together_time.days = 40 := by
  sorry

end work_completion_time_l3966_396626


namespace goods_train_length_l3966_396692

/-- The length of a goods train given its speed, platform length, and time to cross the platform. -/
theorem goods_train_length (speed : ℝ) (platform_length : ℝ) (crossing_time : ℝ) :
  speed = 96 →
  platform_length = 360 →
  crossing_time = 32 →
  let speed_mps := speed * (5 / 18)
  let total_distance := speed_mps * crossing_time
  let train_length := total_distance - platform_length
  train_length = 493.44 := by
  sorry

end goods_train_length_l3966_396692


namespace square_of_binomial_l3966_396648

theorem square_of_binomial (b : ℚ) : 
  (∃ (c : ℚ), ∀ (x : ℚ), 9*x^2 + 21*x + b = (3*x + c)^2) → b = 49/4 := by
  sorry

end square_of_binomial_l3966_396648


namespace two_inequalities_always_true_l3966_396687

theorem two_inequalities_always_true 
  (x y a b : ℝ) 
  (hx : x ≠ 0) 
  (hy : y ≠ 0) 
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (hxa : x < a) 
  (hyb : y < b) 
  (hx_neg : x < 0) 
  (hy_neg : y < 0) 
  (ha_pos : a > 0) 
  (hb_pos : b > 0) :
  ∃! n : ℕ, n = (Bool.toNat (x + y < a + b)) + 
               (Bool.toNat (x - y < a - b)) + 
               (Bool.toNat (x * y < a * b)) + 
               (Bool.toNat (x / y < a / b)) ∧ 
               n = 2 := by
sorry

end two_inequalities_always_true_l3966_396687


namespace remainder_problem_l3966_396633

theorem remainder_problem (n a b c d : ℕ) : 
  n = 102 * a + b ∧ 
  0 ≤ b ∧ b < 102 ∧
  n = 103 * c + d ∧ 
  0 ≤ d ∧ d < 103 ∧
  a + d = 20 
  → b = 20 := by
  sorry

end remainder_problem_l3966_396633


namespace lawrence_county_kids_count_lawrence_county_total_kids_l3966_396629

theorem lawrence_county_kids_count : ℕ → ℕ → ℕ
  | kids_at_home, kids_at_camp =>
    kids_at_home + kids_at_camp

theorem lawrence_county_total_kids :
  lawrence_county_kids_count 907611 455682 = 1363293 := by
  sorry

end lawrence_county_kids_count_lawrence_county_total_kids_l3966_396629


namespace equal_std_dev_and_range_l3966_396676

variable (n : ℕ) (c : ℝ)
variable (x y : Fin n → ℝ)

-- Define the relationship between x and y
def y_def : Prop := ∀ i : Fin n, y i = x i + c

-- Define sample standard deviation
def sample_std_dev (z : Fin n → ℝ) : ℝ := sorry

-- Define sample range
def sample_range (z : Fin n → ℝ) : ℝ := sorry

-- Theorem statement
theorem equal_std_dev_and_range (hc : c ≠ 0) (h_y_def : y_def n c x y) :
  (sample_std_dev n x = sample_std_dev n y) ∧
  (sample_range n x = sample_range n y) := by sorry

end equal_std_dev_and_range_l3966_396676


namespace least_bench_sections_l3966_396683

/- A single bench section can hold 5 adults or 13 children -/
def adults_per_bench : ℕ := 5
def children_per_bench : ℕ := 13

/- M bench sections are connected end to end -/
def bench_sections (M : ℕ) : ℕ := M

/- An equal number of adults and children are to occupy all benches completely -/
def equal_occupancy (M : ℕ) : Prop :=
  ∃ x : ℕ, x > 0 ∧ adults_per_bench * bench_sections M = x ∧ children_per_bench * bench_sections M = x

/- The least possible positive integer value of M -/
theorem least_bench_sections : 
  ∃ M : ℕ, M > 0 ∧ equal_occupancy M ∧ ∀ N : ℕ, N > 0 → equal_occupancy N → M ≤ N :=
by sorry

end least_bench_sections_l3966_396683


namespace cone_base_radius_l3966_396675

/-- The radius of the base of a cone, given its surface area and net shape. -/
theorem cone_base_radius (S : ℝ) (r : ℝ) : 
  S = 9 * Real.pi  -- Surface area condition
  → S = 3 * Real.pi * r^2  -- Surface area formula for a cone
  → r = Real.sqrt 3 :=  -- Conclusion: radius is √3
by
  sorry

end cone_base_radius_l3966_396675


namespace man_mass_l3966_396618

-- Define the boat's dimensions
def boat_length : Real := 3
def boat_breadth : Real := 2
def sinking_depth : Real := 0.01  -- 1 cm in meters

-- Define water density
def water_density : Real := 1000  -- kg/m³

-- Define the theorem
theorem man_mass (volume : Real) (h1 : volume = boat_length * boat_breadth * sinking_depth)
  (mass : Real) (h2 : mass = water_density * volume) : mass = 60 := by
  sorry

end man_mass_l3966_396618


namespace line_direction_vector_l3966_396630

/-- Given a line with direction vector (a, -2) passing through points (-3, 7) and (2, -1),
    prove that a = 5/4 -/
theorem line_direction_vector (a : ℝ) : 
  (∃ t : ℝ, (2 : ℝ) = -3 + t * a ∧ (-1 : ℝ) = 7 + t * (-2)) → a = 5/4 := by
  sorry

end line_direction_vector_l3966_396630


namespace cara_speed_l3966_396659

/-- 
Proves that given a distance of 120 miles between two cities, 
if a person (Dan) leaving 60 minutes after another person (Cara) 
must exceed 40 mph to arrive first, then the first person's (Cara's) 
constant speed is 30 mph.
-/
theorem cara_speed (distance : ℝ) (dan_delay : ℝ) (dan_min_speed : ℝ) : 
  distance = 120 → 
  dan_delay = 1 → 
  dan_min_speed = 40 → 
  ∃ (cara_speed : ℝ), 
    cara_speed * (distance / dan_min_speed + dan_delay) = distance ∧ 
    cara_speed = 30 := by
  sorry


end cara_speed_l3966_396659


namespace chess_competition_games_l3966_396653

theorem chess_competition_games (W M : ℕ) (h1 : W = 12) (h2 : M = 24) : W * M = 288 := by
  sorry

end chess_competition_games_l3966_396653


namespace decimal_to_fraction_sum_l3966_396667

theorem decimal_to_fraction_sum (m n : ℕ+) : 
  (m : ℚ) / (n : ℚ) = 1824 / 10000 → 
  ∀ (a b : ℕ+), (a : ℚ) / (b : ℚ) = 1824 / 10000 → 
  (a : ℕ) ≤ (m : ℕ) ∧ (b : ℕ) ≤ (n : ℕ) →
  m + n = 739 := by
sorry


end decimal_to_fraction_sum_l3966_396667


namespace like_terms_exponent_difference_l3966_396654

/-- Given that 2a^m * b^2 and -a^5 * b^n are like terms, prove that n-m = -3 -/
theorem like_terms_exponent_difference (a b : ℝ) (m n : ℤ) 
  (h : ∃ (k : ℝ), 2 * a^m * b^2 = k * (-a^5 * b^n)) : n - m = -3 := by
  sorry

end like_terms_exponent_difference_l3966_396654


namespace ellipse_condition_l3966_396617

theorem ellipse_condition (k a : ℝ) : 
  (∀ x y : ℝ, 3*x^2 + 9*y^2 - 12*x + 27*y = k → 
    ∃ h₁ h₂ c : ℝ, h₁ > 0 ∧ h₂ > 0 ∧ 
    (x - c)^2 / h₁^2 + (y - c)^2 / h₂^2 = 1) ↔ 
  k > a := by
sorry

end ellipse_condition_l3966_396617


namespace sin_plus_cos_value_l3966_396635

theorem sin_plus_cos_value (x : Real) 
  (h1 : 0 < x ∧ x < Real.pi / 2) 
  (h2 : Real.tan (x - Real.pi / 4) = -1 / 7) : 
  Real.sin x + Real.cos x = 7 / 5 := by
  sorry

end sin_plus_cos_value_l3966_396635


namespace sculpture_cost_in_cny_l3966_396624

/-- Exchange rate from US dollars to Namibian dollars -/
def usd_to_nad : ℝ := 8

/-- Exchange rate from US dollars to Chinese yuan -/
def usd_to_cny : ℝ := 5

/-- Cost of the sculpture in Namibian dollars -/
def sculpture_cost_nad : ℝ := 160

/-- Theorem stating the cost of the sculpture in Chinese yuan -/
theorem sculpture_cost_in_cny :
  (sculpture_cost_nad / usd_to_nad) * usd_to_cny = 100 := by
  sorry

end sculpture_cost_in_cny_l3966_396624


namespace normal_distribution_probabilities_l3966_396677

-- Define a random variable following a normal distribution
def normal_distribution (μ : ℝ) (σ : ℝ) : Type := ℝ

-- Define the cumulative distribution function (CDF) for a normal distribution
noncomputable def normal_cdf (μ σ : ℝ) (x : ℝ) : ℝ := sorry

-- State the theorem
theorem normal_distribution_probabilities 
  (ξ : normal_distribution 1.5 σ) 
  (h : normal_cdf 1.5 σ 2.5 = 0.78) : 
  normal_cdf 1.5 σ 0.5 = 0.22 := by sorry

end normal_distribution_probabilities_l3966_396677


namespace unique_n_with_no_constant_term_l3966_396658

/-- The expansion of (1+x+x²)(x+1/x³)ⁿ has no constant term -/
def has_no_constant_term (n : ℕ) : Prop :=
  ∀ (x : ℝ), x ≠ 0 → (1 + x + x^2) * (x + 1/x^3)^n ≠ 1

theorem unique_n_with_no_constant_term :
  ∃! (n : ℕ), 2 ≤ n ∧ n ≤ 8 ∧ has_no_constant_term n ∧ n = 5 := by
  sorry

end unique_n_with_no_constant_term_l3966_396658


namespace painted_cubes_ratio_l3966_396637

/-- Represents a rectangular prism with integer dimensions -/
structure RectangularPrism where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Counts the number of cubes with exactly two painted faces in a rectangular prism -/
def count_two_faces (prism : RectangularPrism) : ℕ :=
  4 * ((prism.length - 2) + (prism.width - 2) + (prism.height - 2))

/-- Counts the number of cubes with exactly three painted faces in a rectangular prism -/
def count_three_faces (prism : RectangularPrism) : ℕ := 8

/-- The main theorem statement -/
theorem painted_cubes_ratio (prism : RectangularPrism)
    (h_length : prism.length = 4)
    (h_width : prism.width = 5)
    (h_height : prism.height = 6) :
    (count_two_faces prism) / (count_three_faces prism) = 9 / 2 := by
  sorry


end painted_cubes_ratio_l3966_396637


namespace largest_radius_special_circle_l3966_396621

/-- A circle with the given properties -/
structure SpecialCircle where
  center : ℝ × ℝ
  radius : ℝ
  contains_points : center.1^2 + (center.2 - 11)^2 = radius^2 ∧
                    center.1^2 + (center.2 + 11)^2 = radius^2
  contains_unit_disk : ∀ (x y : ℝ), x^2 + y^2 < 1 →
                       (x - center.1)^2 + (y - center.2)^2 < radius^2

/-- The theorem stating the largest possible radius -/
theorem largest_radius_special_circle :
  ∃ (c : SpecialCircle), ∀ (c' : SpecialCircle), c'.radius ≤ c.radius ∧ c.radius = Real.sqrt 122 :=
sorry

end largest_radius_special_circle_l3966_396621


namespace milk_pumping_time_l3966_396608

theorem milk_pumping_time (initial_milk : ℝ) (pump_rate : ℝ) (add_rate : ℝ) (add_time : ℝ) (final_milk : ℝ) :
  initial_milk = 30000 ∧
  pump_rate = 2880 ∧
  add_rate = 1500 ∧
  add_time = 7 ∧
  final_milk = 28980 →
  ∃ (h : ℝ), h = 4 ∧ initial_milk - pump_rate * h + add_rate * add_time = final_milk :=
by sorry

end milk_pumping_time_l3966_396608


namespace bus_speed_excluding_stoppages_l3966_396606

/-- Given a bus that stops for 10 minutes per hour and travels at 45 kmph including stoppages,
    prove that its speed excluding stoppages is 54 kmph. -/
theorem bus_speed_excluding_stoppages :
  let stop_time : ℚ := 10 / 60  -- 10 minutes per hour
  let speed_with_stops : ℚ := 45  -- 45 kmph including stoppages
  let actual_travel_time : ℚ := 1 - stop_time  -- fraction of hour bus is moving
  speed_with_stops / actual_travel_time = 54 := by
  sorry

end bus_speed_excluding_stoppages_l3966_396606


namespace parabola_vertex_l3966_396685

/-- The parabola equation -/
def parabola_equation (x y : ℝ) : Prop := y = -3 * (x - 1)^2 - 2

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (1, -2)

/-- Theorem: The vertex of the parabola y = -3(x-1)^2 - 2 is at the point (1, -2) -/
theorem parabola_vertex :
  ∀ x y : ℝ, parabola_equation x y → (x, y) = vertex :=
sorry

end parabola_vertex_l3966_396685


namespace divisibility_by_7_implies_37_l3966_396612

/-- Given a natural number n, returns the number consisting of n repeated digits of 1 -/
def repeatedOnes (n : ℕ) : ℕ := 
  (10^n - 1) / 9

/-- Theorem: If a number consisting of n repeated digits of 1 is divisible by 7, 
    then it is also divisible by 37 -/
theorem divisibility_by_7_implies_37 (n : ℕ) :
  (repeatedOnes n) % 7 = 0 → (repeatedOnes n) % 37 = 0 := by
  sorry

end divisibility_by_7_implies_37_l3966_396612


namespace problem_solution_l3966_396616

theorem problem_solution (a b m : ℝ) 
  (h1 : 2^a = m) 
  (h2 : 5^b = m) 
  (h3 : 1/a + 1/b = 2) : 
  m = Real.sqrt 10 := by
  sorry

end problem_solution_l3966_396616


namespace mat_cost_per_square_meter_l3966_396602

/-- Given a rectangular hall with specified dimensions and total expenditure for floor covering,
    calculate the cost per square meter of the mat. -/
theorem mat_cost_per_square_meter
  (length width height : ℝ)
  (total_expenditure : ℝ)
  (h_length : length = 20)
  (h_width : width = 15)
  (h_height : height = 5)
  (h_expenditure : total_expenditure = 57000) :
  total_expenditure / (length * width) = 190 := by
  sorry

end mat_cost_per_square_meter_l3966_396602


namespace absolute_value_equation_l3966_396611

theorem absolute_value_equation (x y : ℝ) :
  |x - Real.log y| = x + Real.log y → x * (y - 1) = 0 := by
  sorry

end absolute_value_equation_l3966_396611


namespace binomial_coeff_divisible_by_two_primes_l3966_396639

theorem binomial_coeff_divisible_by_two_primes (n k : ℕ) 
  (h1 : k > 1) (h2 : k < n - 1) : 
  ∃ (p q : ℕ), Prime p ∧ Prime q ∧ p ≠ q ∧ p ∣ Nat.choose n k ∧ q ∣ Nat.choose n k :=
by sorry

end binomial_coeff_divisible_by_two_primes_l3966_396639


namespace rectangle_length_l3966_396603

/-- Proves that the length of a rectangle is 16 centimeters, given specific conditions. -/
theorem rectangle_length (square_side : ℝ) (rect_width : ℝ) : 
  square_side = 8 →
  rect_width = 4 →
  square_side * square_side = rect_width * (16 : ℝ) :=
by
  sorry

#check rectangle_length

end rectangle_length_l3966_396603


namespace hyperbola_equation_l3966_396649

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 6*x - 4*y + 8 = 0

-- Define the hyperbola with focus at (2, 0) and vertex at (4, 0)
def hyperbola_focus : ℝ × ℝ := (2, 0)
def hyperbola_vertex : ℝ × ℝ := (4, 0)

-- Theorem statement
theorem hyperbola_equation : 
  ∀ x y : ℝ, 
  circle_C x y →
  (hyperbola_focus = (2, 0) ∧ hyperbola_vertex = (4, 0)) →
  x^2/4 - y^2/12 = 1 :=
sorry

end hyperbola_equation_l3966_396649


namespace division_negative_ten_by_five_l3966_396656

theorem division_negative_ten_by_five : -10 / 5 = -2 := by sorry

end division_negative_ten_by_five_l3966_396656


namespace cubic_factorization_l3966_396615

theorem cubic_factorization (a : ℝ) : a^3 - 2*a^2 + a = a*(a-1)^2 := by
  sorry

end cubic_factorization_l3966_396615


namespace unique_divisible_by_72_l3966_396698

theorem unique_divisible_by_72 : ∃! n : ℕ,
  (n ≥ 1000000000 ∧ n < 10000000000) ∧
  (∃ a b : ℕ, a < 10 ∧ b < 10 ∧ n = a * 1000000000 + 20222023 * 10 + b) ∧
  n % 72 = 0 :=
by
  sorry

end unique_divisible_by_72_l3966_396698


namespace floor_equation_solution_l3966_396645

theorem floor_equation_solution (x : ℝ) : x - Int.floor (x / 2016) = 2016 ↔ x = 2017 := by
  sorry

end floor_equation_solution_l3966_396645


namespace no_five_integers_solution_l3966_396695

theorem no_five_integers_solution :
  ¬ ∃ (a b c d e : ℕ),
    (0 < a) ∧ (a < b) ∧ (b < c) ∧ (c < d) ∧ (d < e) ∧
    ({a + b, a + c, a + d, a + e, b + c, b + d, b + e, c + d, c + e, d + e} = 
     {15, 16, 17, 18, 19, 20, 21, 23} ∪ {x | x < 15} ∪ {y | y > 23}) :=
by
  sorry

#check no_five_integers_solution

end no_five_integers_solution_l3966_396695


namespace sum_of_reciprocals_l3966_396665

theorem sum_of_reciprocals (a b : ℝ) (ha : a^2 + a = 4) (hb : b^2 + b = 4) (hab : a ≠ b) :
  b / a + a / b = -9/4 := by
  sorry

end sum_of_reciprocals_l3966_396665


namespace absolute_value_inequality_solution_set_l3966_396668

theorem absolute_value_inequality_solution_set (x : ℝ) :
  (1 < |2*x - 1| ∧ |2*x - 1| < 3) ↔ ((-1 < x ∧ x < 0) ∨ (1 < x ∧ x < 2)) := by
  sorry

end absolute_value_inequality_solution_set_l3966_396668


namespace hundredth_decimal_is_9_l3966_396678

/-- The decimal expansion of 10/11 -/
def decimal_expansion_10_11 : ℕ → ℕ := 
  fun n => if n % 2 = 0 then 0 else 9

/-- The 100th decimal digit in the expansion of 10/11 -/
def hundredth_decimal : ℕ := decimal_expansion_10_11 100

theorem hundredth_decimal_is_9 : hundredth_decimal = 9 := by sorry

end hundredth_decimal_is_9_l3966_396678


namespace stream_current_rate_l3966_396610

/-- Proves that the rate of a stream's current is 4 kmph given the conditions of a boat's travel --/
theorem stream_current_rate (distance_one_way : ℝ) (total_time : ℝ) (still_water_speed : ℝ) 
  (h1 : distance_one_way = 6)
  (h2 : total_time = 2)
  (h3 : still_water_speed = 8) :
  ∃ c : ℝ, c = 4 ∧ 
    (distance_one_way / (still_water_speed - c) + distance_one_way / (still_water_speed + c) = total_time) :=
by sorry

end stream_current_rate_l3966_396610


namespace arithmetic_sequence_problem_l3966_396691

/-- An arithmetic sequence with given conditions -/
structure ArithmeticSequence where
  a : ℕ → ℤ  -- The sequence
  d : ℤ      -- Common difference
  h1 : a 2 = 12
  h2 : d = -2
  h3 : ∀ n : ℕ, a (n + 1) = a n + d  -- Definition of arithmetic sequence

/-- The theorem to prove -/
theorem arithmetic_sequence_problem (seq : ArithmeticSequence) :
  ∃ n : ℕ, seq.a n = -20 ∧ n = 18 := by
  sorry


end arithmetic_sequence_problem_l3966_396691


namespace magazine_fraction_l3966_396661

theorem magazine_fraction (initial_amount : ℚ) (grocery_fraction : ℚ) (remaining_amount : ℚ) :
  initial_amount = 600 →
  grocery_fraction = 1/5 →
  remaining_amount = 360 →
  let amount_after_groceries := initial_amount - grocery_fraction * initial_amount
  (amount_after_groceries - remaining_amount) / amount_after_groceries = 1/4 := by
  sorry

end magazine_fraction_l3966_396661


namespace unique_solution_quadratic_positive_n_value_l3966_396693

theorem unique_solution_quadratic (n : ℝ) : 
  (∃! x : ℝ, 9 * x^2 + n * x + 36 = 0) → n = 36 ∨ n = -36 :=
by sorry

theorem positive_n_value (n : ℝ) :
  (∃! x : ℝ, 9 * x^2 + n * x + 36 = 0) ∧ n > 0 → n = 36 :=
by sorry

end unique_solution_quadratic_positive_n_value_l3966_396693


namespace fourth_power_of_one_minus_i_l3966_396634

theorem fourth_power_of_one_minus_i :
  (1 - Complex.I) ^ 4 = -4 := by sorry

end fourth_power_of_one_minus_i_l3966_396634


namespace equation_real_root_l3966_396652

theorem equation_real_root (k : ℝ) : ∃ x : ℝ, x = k^2 * (x - 1) * (x - 2) := by
  sorry

end equation_real_root_l3966_396652


namespace mean_proportional_problem_l3966_396670

theorem mean_proportional_problem (x : ℝ) :
  (156 : ℝ)^2 = 234 * x → x = 104 := by
  sorry

end mean_proportional_problem_l3966_396670
