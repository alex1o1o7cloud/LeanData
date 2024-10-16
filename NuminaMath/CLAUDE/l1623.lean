import Mathlib

namespace NUMINAMATH_CALUDE_sandwich_problem_l1623_162361

theorem sandwich_problem (sandwich_cost soda_cost total_cost : ℚ) 
                         (num_sodas : ℕ) :
  sandwich_cost = 245/100 →
  soda_cost = 87/100 →
  num_sodas = 4 →
  total_cost = 838/100 →
  ∃ (num_sandwiches : ℕ), 
    num_sandwiches * sandwich_cost + num_sodas * soda_cost = total_cost ∧
    num_sandwiches = 2 :=
by sorry

end NUMINAMATH_CALUDE_sandwich_problem_l1623_162361


namespace NUMINAMATH_CALUDE_complex_root_quadratic_equation_l1623_162373

theorem complex_root_quadratic_equation (a b : ℝ) :
  (∃ (x : ℂ), x = 1 + Complex.I * Real.sqrt 3 ∧ a * x^2 + b * x + 1 = 0) →
  a = (1 : ℝ) / 4 ∧ b = -(1 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_root_quadratic_equation_l1623_162373


namespace NUMINAMATH_CALUDE_water_moles_theorem_l1623_162321

/-- Represents a chemical equation with reactants and products -/
structure ChemicalEquation where
  naoh_reactant : ℕ
  h2so4_reactant : ℕ
  na2so4_product : ℕ
  h2o_product : ℕ

/-- The balanced equation for the reaction -/
def balanced_equation : ChemicalEquation :=
  { naoh_reactant := 2
  , h2so4_reactant := 1
  , na2so4_product := 1
  , h2o_product := 2 }

/-- The number of moles of NaOH reacting -/
def naoh_moles : ℕ := 4

/-- The number of moles of H₂SO₄ reacting -/
def h2so4_moles : ℕ := 2

/-- Calculates the number of moles of water produced -/
def water_moles_produced (eq : ChemicalEquation) (naoh : ℕ) : ℕ :=
  (naoh * eq.h2o_product) / eq.naoh_reactant

/-- Theorem stating that 4 moles of water are produced -/
theorem water_moles_theorem :
  water_moles_produced balanced_equation naoh_moles = 4 :=
sorry

end NUMINAMATH_CALUDE_water_moles_theorem_l1623_162321


namespace NUMINAMATH_CALUDE_fraction_ordering_l1623_162344

theorem fraction_ordering : (8 : ℚ) / 25 < 6 / 17 ∧ 6 / 17 < 10 / 27 := by
  sorry

end NUMINAMATH_CALUDE_fraction_ordering_l1623_162344


namespace NUMINAMATH_CALUDE_dice_configuration_dots_l1623_162314

/-- Represents a single die face -/
inductive DieFace
  | one
  | two
  | three
  | four
  | five
  | six

/-- Represents the configuration of four dice -/
structure DiceConfiguration :=
  (faceA : DieFace)
  (faceB : DieFace)
  (faceC : DieFace)
  (faceD : DieFace)

/-- Counts the number of dots on a die face -/
def dotCount (face : DieFace) : Nat :=
  match face with
  | DieFace.one => 1
  | DieFace.two => 2
  | DieFace.three => 3
  | DieFace.four => 4
  | DieFace.five => 5
  | DieFace.six => 6

/-- The specific configuration of dice in the problem -/
def problemConfiguration : DiceConfiguration :=
  { faceA := DieFace.three
  , faceB := DieFace.five
  , faceC := DieFace.six
  , faceD := DieFace.five }

theorem dice_configuration_dots :
  dotCount problemConfiguration.faceA = 3 ∧
  dotCount problemConfiguration.faceB = 5 ∧
  dotCount problemConfiguration.faceC = 6 ∧
  dotCount problemConfiguration.faceD = 5 := by
  sorry

end NUMINAMATH_CALUDE_dice_configuration_dots_l1623_162314


namespace NUMINAMATH_CALUDE_circle_center_radius_sum_l1623_162348

/-- Given a circle C with equation x^2 + 6x + 36 = -y^2 - 8y + 45,
    prove that its center coordinates (a, b) and radius r satisfy a + b + r = -7 + √34 -/
theorem circle_center_radius_sum (x y a b r : ℝ) : 
  (∀ x y, x^2 + 6*x + 36 = -y^2 - 8*y + 45) →
  (∀ x y, (x + 3)^2 + (y + 4)^2 = 34) →
  (a = -3 ∧ b = -4) →
  r = Real.sqrt 34 →
  a + b + r = -7 + Real.sqrt 34 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_radius_sum_l1623_162348


namespace NUMINAMATH_CALUDE_divisibility_theorem_l1623_162359

theorem divisibility_theorem (m n : ℕ+) (h : 5 ∣ (2^n.val + 3^m.val)) :
  5 ∣ (2^m.val + 3^n.val) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_theorem_l1623_162359


namespace NUMINAMATH_CALUDE_standard_time_proof_l1623_162322

/-- The standard time to complete one workpiece -/
def standard_time : ℝ := 15

/-- The time taken by the first worker after innovation -/
def worker1_time (x : ℝ) : ℝ := x - 5

/-- The time taken by the second worker after innovation -/
def worker2_time (x : ℝ) : ℝ := x - 3

/-- The performance improvement factor -/
def improvement_factor : ℝ := 1.375

theorem standard_time_proof :
  ∃ (x : ℝ),
    x > 0 ∧
    worker1_time x > 0 ∧
    worker2_time x > 0 ∧
    (1 / worker1_time x + 1 / worker2_time x) = (2 / x) * improvement_factor ∧
    x = standard_time :=
by sorry

end NUMINAMATH_CALUDE_standard_time_proof_l1623_162322


namespace NUMINAMATH_CALUDE_lottery_theorem_l1623_162327

/-- The number of integers from 1 to 90 that can be expressed as the sum of two square numbers -/
def p : ℕ := 40

/-- The total number of integers in the lottery pool -/
def n : ℕ := 90

/-- The number of integers drawn in each lottery -/
def k : ℕ := 5

/-- The probability of drawing 5 numbers that can all be expressed as the sum of two square numbers -/
def lottery_probability : ℚ := (Nat.choose p k : ℚ) / (Nat.choose n k : ℚ)

theorem lottery_theorem : 
  ∃ (ε : ℚ), abs (lottery_probability - 0.015) < ε ∧ ε > 0 :=
sorry

end NUMINAMATH_CALUDE_lottery_theorem_l1623_162327


namespace NUMINAMATH_CALUDE_probability_sum_nine_l1623_162301

/-- The number of sides on a standard die -/
def sides : ℕ := 6

/-- The number of dice rolled -/
def numDice : ℕ := 3

/-- The target sum we're looking for -/
def targetSum : ℕ := 9

/-- The total number of possible outcomes when rolling three dice -/
def totalOutcomes : ℕ := sides ^ numDice

/-- The number of favorable outcomes (ways to get a sum of 9) -/
def favorableOutcomes : ℕ := 19

/-- The probability of rolling a sum of 9 with three fair, standard six-sided dice -/
theorem probability_sum_nine :
  (favorableOutcomes : ℚ) / totalOutcomes = 19 / 216 := by
  sorry

end NUMINAMATH_CALUDE_probability_sum_nine_l1623_162301


namespace NUMINAMATH_CALUDE_max_piles_l1623_162354

/-- The total number of stones --/
def total_stones : ℕ := 660

/-- A function to check if two pile sizes are similar (differ by strictly less than 2 times) --/
def are_similar (a b : ℕ) : Prop := 
  a < 2 * b ∧ b < 2 * a

/-- A type representing a valid distribution of stones into piles --/
structure StoneDistribution where
  piles : List ℕ
  sum_to_total : piles.sum = total_stones
  all_similar : ∀ (a b : ℕ), a ∈ piles → b ∈ piles → are_similar a b

/-- The theorem stating the maximum number of piles --/
theorem max_piles : 
  (∀ d : StoneDistribution, d.piles.length ≤ 30) ∧ 
  (∃ d : StoneDistribution, d.piles.length = 30) := by
  sorry

end NUMINAMATH_CALUDE_max_piles_l1623_162354


namespace NUMINAMATH_CALUDE_stating_holiday_lodge_assignments_l1623_162337

/-- Represents the number of rooms in the holiday lodge -/
def num_rooms : ℕ := 4

/-- Represents the number of friends staying at the lodge -/
def num_friends : ℕ := 6

/-- Represents the maximum number of friends allowed per room -/
def max_friends_per_room : ℕ := 2

/-- Represents the minimum number of empty rooms required -/
def min_empty_rooms : ℕ := 1

/-- 
Calculates the number of ways to assign friends to rooms 
given the constraints
-/
def num_assignments (n_rooms : ℕ) (n_friends : ℕ) 
  (max_per_room : ℕ) (min_empty : ℕ) : ℕ := 
  sorry

/-- 
Theorem stating that the number of assignments for the given problem is 1080
-/
theorem holiday_lodge_assignments : 
  num_assignments num_rooms num_friends max_friends_per_room min_empty_rooms = 1080 := by
  sorry

end NUMINAMATH_CALUDE_stating_holiday_lodge_assignments_l1623_162337


namespace NUMINAMATH_CALUDE_coin_value_proof_l1623_162317

/-- The value of a penny in dollars -/
def penny_value : ℚ := 1 / 100

/-- The value of a nickel in dollars -/
def nickel_value : ℚ := 5 / 100

/-- The value of a dime in dollars -/
def dime_value : ℚ := 10 / 100

/-- The number of pennies -/
def num_pennies : ℕ := 9

/-- The number of nickels -/
def num_nickels : ℕ := 4

/-- The number of dimes -/
def num_dimes : ℕ := 3

/-- The total value of the coins in dollars -/
def total_value : ℚ := num_pennies * penny_value + num_nickels * nickel_value + num_dimes * dime_value

theorem coin_value_proof : total_value = 59 / 100 := by
  sorry

end NUMINAMATH_CALUDE_coin_value_proof_l1623_162317


namespace NUMINAMATH_CALUDE_race_outcome_l1623_162336

/-- Represents the distance traveled by an animal at a given time --/
structure DistanceTime where
  distance : ℝ
  time : ℝ

/-- Represents the race between a tortoise and a hare --/
structure Race where
  tortoise : List DistanceTime
  hare : List DistanceTime

/-- Checks if a list of DistanceTime points represents a steady pace --/
def isSteadyPace (points : List DistanceTime) : Prop := sorry

/-- Checks if a list of DistanceTime points has exactly two stops --/
def hasTwoStops (points : List DistanceTime) : Prop := sorry

/-- Checks if the first point in a list finishes before the first point in another list --/
def finishesFirst (winner loser : List DistanceTime) : Prop := sorry

/-- Theorem representing the race conditions and outcome --/
theorem race_outcome (race : Race) : 
  isSteadyPace race.tortoise ∧ 
  hasTwoStops race.hare ∧ 
  finishesFirst race.tortoise race.hare := by
  sorry

#check race_outcome

end NUMINAMATH_CALUDE_race_outcome_l1623_162336


namespace NUMINAMATH_CALUDE_lesser_number_proof_l1623_162364

theorem lesser_number_proof (a b : ℝ) (h1 : a + b = 60) (h2 : a - b = 10) : min a b = 25 := by
  sorry

end NUMINAMATH_CALUDE_lesser_number_proof_l1623_162364


namespace NUMINAMATH_CALUDE_pirate_costume_group_size_l1623_162360

theorem pirate_costume_group_size 
  (costume_cost : ℕ) 
  (total_spent : ℕ) 
  (h1 : costume_cost = 5)
  (h2 : total_spent = 40) :
  total_spent / costume_cost = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_pirate_costume_group_size_l1623_162360


namespace NUMINAMATH_CALUDE_value_of_a_l1623_162362

theorem value_of_a (a : ℝ) : 4 ∈ ({a^2 - 3*a, a} : Set ℝ) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l1623_162362


namespace NUMINAMATH_CALUDE_area_of_special_parallelogram_l1623_162302

/-- Represents a parallelogram with base and altitude. -/
structure Parallelogram where
  base : ℝ
  altitude : ℝ

/-- The area of a parallelogram. -/
def area (p : Parallelogram) : ℝ := p.base * p.altitude

/-- A parallelogram with altitude twice the base and base length 12. -/
def special_parallelogram : Parallelogram where
  base := 12
  altitude := 2 * 12

theorem area_of_special_parallelogram :
  area special_parallelogram = 288 := by
  sorry

end NUMINAMATH_CALUDE_area_of_special_parallelogram_l1623_162302


namespace NUMINAMATH_CALUDE_total_value_calculation_l1623_162390

-- Define coin quantities
def us_quarters : ℕ := 25
def us_dimes : ℕ := 15
def us_nickels : ℕ := 12
def us_half_dollars : ℕ := 7
def us_dollar_coins : ℕ := 3
def us_pennies : ℕ := 375
def canadian_quarters : ℕ := 10
def canadian_dimes : ℕ := 5
def canadian_nickels : ℕ := 4

-- Define coin values in their respective currencies
def us_quarter_value : ℚ := 0.25
def us_dime_value : ℚ := 0.10
def us_nickel_value : ℚ := 0.05
def us_half_dollar_value : ℚ := 0.50
def us_dollar_coin_value : ℚ := 1.00
def us_penny_value : ℚ := 0.01
def canadian_quarter_value : ℚ := 0.25
def canadian_dime_value : ℚ := 0.10
def canadian_nickel_value : ℚ := 0.05

-- Define exchange rate
def cad_to_usd_rate : ℚ := 0.80

-- Theorem to prove
theorem total_value_calculation :
  (us_quarters * us_quarter_value +
   us_dimes * us_dime_value +
   us_nickels * us_nickel_value +
   us_half_dollars * us_half_dollar_value +
   us_dollar_coins * us_dollar_coin_value +
   us_pennies * us_penny_value) +
  ((canadian_quarters * canadian_quarter_value +
    canadian_dimes * canadian_dime_value +
    canadian_nickels * canadian_nickel_value) * cad_to_usd_rate) = 21.16 := by
  sorry

end NUMINAMATH_CALUDE_total_value_calculation_l1623_162390


namespace NUMINAMATH_CALUDE_betty_cookie_consumption_l1623_162394

/-- The number of cookies Betty eats per day -/
def cookies_per_day : ℕ := 7

/-- The number of brownies Betty eats per day -/
def brownies_per_day : ℕ := 1

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The difference between cookies and brownies after a week -/
def cookie_brownie_difference : ℕ := 36

theorem betty_cookie_consumption :
  cookies_per_day * days_in_week - brownies_per_day * days_in_week = cookie_brownie_difference :=
by sorry

end NUMINAMATH_CALUDE_betty_cookie_consumption_l1623_162394


namespace NUMINAMATH_CALUDE_decimal_118_to_base6_l1623_162375

def decimal_to_base6 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 6) ((m % 6) :: acc)
    aux n []

theorem decimal_118_to_base6 :
  decimal_to_base6 118 = [3, 1, 4] :=
sorry

end NUMINAMATH_CALUDE_decimal_118_to_base6_l1623_162375


namespace NUMINAMATH_CALUDE_five_digit_divisible_by_165_l1623_162307

/-- Represents a 5-digit number in the form XX4XY -/
structure FiveDigitNumber where
  x : ℕ
  y : ℕ
  is_valid : x < 10 ∧ y < 10

/-- The 5-digit number as an integer -/
def FiveDigitNumber.to_int (n : FiveDigitNumber) : ℤ :=
  ↑(n.x * 10000 + n.x * 1000 + 400 + n.x * 10 + n.y)

theorem five_digit_divisible_by_165 (n : FiveDigitNumber) :
  n.to_int % 165 = 0 → n.x + n.y = 14 := by
  sorry


end NUMINAMATH_CALUDE_five_digit_divisible_by_165_l1623_162307


namespace NUMINAMATH_CALUDE_factorization_xy_minus_8y_l1623_162388

theorem factorization_xy_minus_8y (x y : ℝ) : x * y - 8 * y = y * (x - 8) := by
  sorry

end NUMINAMATH_CALUDE_factorization_xy_minus_8y_l1623_162388


namespace NUMINAMATH_CALUDE_decreasing_function_inequality_l1623_162396

theorem decreasing_function_inequality (f : ℝ → ℝ) (a : ℝ) 
  (h_decreasing : ∀ x y : ℝ, x < y → f x > f y) 
  (h_inequality : f a ≥ f (-2)) : 
  a ≤ -2 := by
  sorry

end NUMINAMATH_CALUDE_decreasing_function_inequality_l1623_162396


namespace NUMINAMATH_CALUDE_stock_price_change_l1623_162325

theorem stock_price_change (initial_price : ℝ) (h : initial_price > 0) :
  let day1_price := initial_price * (1 - 0.15)
  let day2_price := day1_price * (1 + 0.25)
  (day2_price - initial_price) / initial_price = 0.0625 := by
sorry

end NUMINAMATH_CALUDE_stock_price_change_l1623_162325


namespace NUMINAMATH_CALUDE_negative_fraction_comparison_l1623_162399

theorem negative_fraction_comparison : -5/4 < -4/5 := by
  sorry

end NUMINAMATH_CALUDE_negative_fraction_comparison_l1623_162399


namespace NUMINAMATH_CALUDE_greatest_integer_value_l1623_162387

theorem greatest_integer_value (x : ℤ) : 
  (∀ y : ℤ, y > x → ¬(∃ z : ℤ, (y^2 + 5*y + 6) / (y - 2) = z)) →
  (∃ z : ℤ, (x^2 + 5*x + 6) / (x - 2) = z) →
  x = 22 :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_value_l1623_162387


namespace NUMINAMATH_CALUDE_max_min_sum_xy_xz_yz_l1623_162350

theorem max_min_sum_xy_xz_yz (x y z : ℝ) (h : 3 * (x + y + z) = x^2 + y^2 + z^2) : 
  ∃ (M m : ℝ), (∀ t : ℝ, t = x*y + x*z + y*z → t ≤ M) ∧ 
                (∀ t : ℝ, t = x*y + x*z + y*z → m ≤ t) ∧ 
                M + 10*m = 27 := by
  sorry

end NUMINAMATH_CALUDE_max_min_sum_xy_xz_yz_l1623_162350


namespace NUMINAMATH_CALUDE_pinwheel_shaded_area_l1623_162391

/-- Represents the pinwheel toy configuration -/
structure PinwheelToy where
  square_side : Real
  triangle_leg : Real
  π : Real

/-- Calculates the area of the shaded region in the pinwheel toy -/
def shaded_area (toy : PinwheelToy) : Real :=
  -- The actual calculation would go here
  286

/-- Theorem stating that the shaded area of the specific pinwheel toy is 286 square cm -/
theorem pinwheel_shaded_area :
  ∃ (toy : PinwheelToy),
    toy.square_side = 20 ∧
    toy.triangle_leg = 10 ∧
    toy.π = 3.14 ∧
    shaded_area toy = 286 := by
  sorry


end NUMINAMATH_CALUDE_pinwheel_shaded_area_l1623_162391


namespace NUMINAMATH_CALUDE_consecutive_composites_l1623_162334

theorem consecutive_composites (n : ℕ) (hn : n > 0) :
  ∃ k : ℕ, (∀ i : ℕ, i < n → ¬ Nat.Prime (k + i + 2)) ∧
           (k + n + 1 < 4^(n + 1)) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_composites_l1623_162334


namespace NUMINAMATH_CALUDE_money_ratio_proof_l1623_162311

def josh_money (doug_money : ℚ) : ℚ := (3 / 4) * doug_money

theorem money_ratio_proof (doug_money : ℚ) 
  (h1 : josh_money doug_money + doug_money + 12 = 68) : 
  josh_money doug_money / 12 = 2 := by
  sorry

end NUMINAMATH_CALUDE_money_ratio_proof_l1623_162311


namespace NUMINAMATH_CALUDE_mojave_population_increase_l1623_162333

/-- Calculates the percentage increase between two populations -/
def percentageIncrease (initial : ℕ) (final : ℕ) : ℚ :=
  (final - initial : ℚ) / initial * 100

theorem mojave_population_increase : 
  let initialPopulation : ℕ := 4000
  let currentPopulation : ℕ := initialPopulation * 3
  let futurePopulation : ℕ := 16800
  percentageIncrease currentPopulation futurePopulation = 40 := by
sorry

end NUMINAMATH_CALUDE_mojave_population_increase_l1623_162333


namespace NUMINAMATH_CALUDE_sqrt_mixed_number_simplification_l1623_162310

theorem sqrt_mixed_number_simplification :
  Real.sqrt (12 + 1/9) = Real.sqrt 109 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_mixed_number_simplification_l1623_162310


namespace NUMINAMATH_CALUDE_prime_divisor_problem_l1623_162386

theorem prime_divisor_problem (n : ℕ) : 
  (∃ p : ℕ, Nat.Prime p ∧ p = Nat.sqrt n ∧ ∀ q : ℕ, Nat.Prime q ∧ q ∣ n → q ≤ p) ∧
  (∃ p : ℕ, Nat.Prime p ∧ p = Nat.sqrt (n + 72) ∧ ∀ q : ℕ, Nat.Prime q ∧ q ∣ (n + 72) → q ≤ p) →
  n = 49 ∨ n = 289 :=
by sorry

end NUMINAMATH_CALUDE_prime_divisor_problem_l1623_162386


namespace NUMINAMATH_CALUDE_no_convex_function_exists_l1623_162395

theorem no_convex_function_exists : 
  ¬∃ f : ℝ → ℝ, ∀ x y : ℝ, (f x + f y) / 2 ≥ f ((x + y) / 2) + |x - y| :=
by sorry

end NUMINAMATH_CALUDE_no_convex_function_exists_l1623_162395


namespace NUMINAMATH_CALUDE_expenditure_ratio_proof_l1623_162380

/-- Given the income ratio and savings of Uma and Bala, prove their expenditure ratio -/
theorem expenditure_ratio_proof (uma_income bala_income uma_expenditure bala_expenditure : ℚ) 
  (h1 : uma_income = (8 : ℚ) / 7 * bala_income)
  (h2 : uma_income = 16000)
  (h3 : uma_income - uma_expenditure = 2000)
  (h4 : bala_income - bala_expenditure = 2000) :
  uma_expenditure / bala_expenditure = 7 / 6 := by
  sorry

end NUMINAMATH_CALUDE_expenditure_ratio_proof_l1623_162380


namespace NUMINAMATH_CALUDE_max_d_minus_r_l1623_162355

theorem max_d_minus_r : ∃ (d r : ℕ), 
  (2017 % d = r ∧ 1029 % d = r ∧ 725 % d = r) ∧
  (∀ (d' r' : ℕ), (2017 % d' = r' ∧ 1029 % d' = r' ∧ 725 % d' = r') → d' - r' ≤ d - r) ∧
  d - r = 35 := by
sorry

end NUMINAMATH_CALUDE_max_d_minus_r_l1623_162355


namespace NUMINAMATH_CALUDE_five_items_three_bags_l1623_162372

/-- The number of ways to distribute n distinct items into k identical bags --/
def distributionWays (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 5 distinct items into 3 identical bags --/
theorem five_items_three_bags : distributionWays 5 3 = 36 := by sorry

end NUMINAMATH_CALUDE_five_items_three_bags_l1623_162372


namespace NUMINAMATH_CALUDE_four_circles_minus_large_circle_area_l1623_162335

/-- Four circles with radius r > 0 centered at (0, r), (r, 0), (0, -r), and (-r, 0) -/
def four_circles (r : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), (x - 0)^2 + (y - r)^2 ≤ r^2 ∨
                    (x - r)^2 + (y - 0)^2 ≤ r^2 ∨
                    (x - 0)^2 + (y + r)^2 ≤ r^2 ∨
                    (x + r)^2 + (y - 0)^2 ≤ r^2}

/-- Circle with radius 2r centered at (0, 0) -/
def large_circle (r : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), x^2 + y^2 ≤ (2*r)^2}

/-- Area of a set in ℝ² -/
noncomputable def area (s : Set (ℝ × ℝ)) : ℝ := sorry

/-- The theorem to be proved -/
theorem four_circles_minus_large_circle_area (r : ℝ) (hr : r > 0) :
  area (four_circles r) - area (large_circle r \ four_circles r) = 8 * r^2 := by sorry

end NUMINAMATH_CALUDE_four_circles_minus_large_circle_area_l1623_162335


namespace NUMINAMATH_CALUDE_product_in_geometric_sequence_l1623_162330

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem product_in_geometric_sequence (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 2 + a 18 = -15) →
  (a 2 * a 18 = 16) →
  a 3 * a 10 * a 17 = -64 := by
  sorry

end NUMINAMATH_CALUDE_product_in_geometric_sequence_l1623_162330


namespace NUMINAMATH_CALUDE_ball_travel_distance_l1623_162338

/-- The total distance traveled by a bouncing ball -/
def total_distance (initial_height : ℝ) (rebound_ratio : ℝ) : ℝ :=
  let first_rebound := initial_height * rebound_ratio
  let second_rebound := first_rebound * rebound_ratio
  initial_height + first_rebound + first_rebound + second_rebound + second_rebound

/-- Theorem: The ball travels 260 cm when it touches the floor for the third time -/
theorem ball_travel_distance :
  total_distance 104 0.5 = 260 := by
  sorry

end NUMINAMATH_CALUDE_ball_travel_distance_l1623_162338


namespace NUMINAMATH_CALUDE_negation_of_implication_l1623_162329

theorem negation_of_implication (x : ℝ) :
  ¬(x > 1 → x > 0) ↔ (x ≤ 1 → x ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l1623_162329


namespace NUMINAMATH_CALUDE_function_equality_l1623_162313

theorem function_equality (f : ℝ → ℝ) 
  (h1 : ∀ x y : ℝ, f (x + f y) ≥ f (f x + y)) 
  (h2 : f 0 = 0) : 
  ∀ x : ℝ, f x = x := by
sorry

end NUMINAMATH_CALUDE_function_equality_l1623_162313


namespace NUMINAMATH_CALUDE_crayon_count_l1623_162316

theorem crayon_count (initial_crayons added_crayons : ℕ) 
  (h1 : initial_crayons = 9)
  (h2 : added_crayons = 3) : 
  initial_crayons + added_crayons = 12 := by
  sorry

end NUMINAMATH_CALUDE_crayon_count_l1623_162316


namespace NUMINAMATH_CALUDE_work_completion_proof_l1623_162351

/-- The number of days it takes the first group to complete the work -/
def days_group1 : ℕ := 35

/-- The number of days it takes the second group to complete the work -/
def days_group2 : ℕ := 50

/-- The number of men in the second group -/
def men_group2 : ℕ := 7

/-- The number of men in the first group -/
def men_group1 : ℕ := men_group2 * days_group2 / days_group1

theorem work_completion_proof : men_group1 = 10 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_proof_l1623_162351


namespace NUMINAMATH_CALUDE_sequence_with_geometric_differences_l1623_162342

def geometric_difference_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → a n - a (n - 1) = 2 * (a (n - 1) - a (n - 2))

theorem sequence_with_geometric_differences 
  (a : ℕ → ℝ) 
  (h1 : a 1 = 1) 
  (h2 : geometric_difference_sequence a) :
  ∀ n : ℕ, n ≥ 1 → a n = 2^n - 1 := by
sorry

end NUMINAMATH_CALUDE_sequence_with_geometric_differences_l1623_162342


namespace NUMINAMATH_CALUDE_other_number_proof_l1623_162340

/-- Given two positive integers with specific HCF, LCM, and one known value, prove the other value -/
theorem other_number_proof (a b : ℕ+) (h1 : Nat.gcd a b = 12) (h2 : Nat.lcm a b = 396) (h3 : a = 36) : b = 132 := by
  sorry

end NUMINAMATH_CALUDE_other_number_proof_l1623_162340


namespace NUMINAMATH_CALUDE_cistern_width_is_six_l1623_162339

/-- Represents the dimensions and properties of a rectangular cistern --/
structure Cistern where
  length : ℝ
  width : ℝ
  waterDepth : ℝ
  wetSurfaceArea : ℝ

/-- Calculates the total wet surface area of a cistern --/
def totalWetSurfaceArea (c : Cistern) : ℝ :=
  c.length * c.width + 2 * c.length * c.waterDepth + 2 * c.width * c.waterDepth

/-- Theorem stating that a cistern with given dimensions has a width of 6 meters --/
theorem cistern_width_is_six (c : Cistern) 
  (h1 : c.length = 9)
  (h2 : c.waterDepth = 2.25)
  (h3 : c.wetSurfaceArea = 121.5)
  (h4 : totalWetSurfaceArea c = c.wetSurfaceArea) : 
  c.width = 6 := by
  sorry

end NUMINAMATH_CALUDE_cistern_width_is_six_l1623_162339


namespace NUMINAMATH_CALUDE_range_of_function_l1623_162312

theorem range_of_function (x : ℝ) : -13 ≤ 5 * Real.sin x - 12 * Real.cos x ∧ 
                                     5 * Real.sin x - 12 * Real.cos x ≤ 13 := by
  sorry

end NUMINAMATH_CALUDE_range_of_function_l1623_162312


namespace NUMINAMATH_CALUDE_sine_three_fourths_pi_minus_alpha_l1623_162353

theorem sine_three_fourths_pi_minus_alpha (α : ℝ) 
  (h : Real.sin (π / 4 + α) = 3 / 5) : 
  Real.sin (3 * π / 4 - α) = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sine_three_fourths_pi_minus_alpha_l1623_162353


namespace NUMINAMATH_CALUDE_problem_solution_l1623_162303

theorem problem_solution (x : ℝ) : (20 / 100 * 30 = 25 / 100 * x + 2) → x = 16 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1623_162303


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l1623_162305

-- Define the complex polynomial
def p (z : ℂ) : ℂ := (z - 1) * (z^2 + 2*z + 4) * (z^2 + 4*z + 6)

-- Define the set of solutions
def S : Set ℂ := {z : ℂ | p z = 0}

-- Define the ellipse passing through the solutions
def E : Set (ℝ × ℝ) :=
  {xy : ℝ × ℝ | ∃ z ∈ S, (xy.1 = z.re ∧ xy.2 = z.im)}

-- State the theorem
theorem ellipse_eccentricity :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  E = {xy : ℝ × ℝ | (xy.1 + 9/10)^2 / (361/100) + xy.2^2 / (361/120) = 1} ∧
  (a^2 - b^2) / a^2 = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l1623_162305


namespace NUMINAMATH_CALUDE_inverse_exp_range_l1623_162324

noncomputable def f : ℝ → ℝ := Real.log

theorem inverse_exp_range (a b : ℝ) :
  (∀ x, f x = Real.log x) →
  (|f a| = |f b|) →
  (a ≠ b) →
  (∀ x > 2, ∃ a b : ℝ, a + b = x ∧ |f a| = |f b| ∧ a ≠ b) ∧
  (|f a| = |f b| ∧ a ≠ b → a + b > 2) :=
sorry

end NUMINAMATH_CALUDE_inverse_exp_range_l1623_162324


namespace NUMINAMATH_CALUDE_vector_calculation_l1623_162347

def a : ℝ × ℝ × ℝ := (1, 0, 1)
def b : ℝ × ℝ × ℝ := (-1, 2, 3)
def c : ℝ × ℝ × ℝ := (0, 1, 1)

theorem vector_calculation :
  a - b + 2 • c = (2, 0, 0) := by sorry

end NUMINAMATH_CALUDE_vector_calculation_l1623_162347


namespace NUMINAMATH_CALUDE_isosceles_triangle_leg_length_l1623_162367

/-- Given an isosceles triangle with perimeter 24 and base 10, prove the leg length is 7 -/
theorem isosceles_triangle_leg_length 
  (perimeter : ℝ) 
  (base : ℝ) 
  (leg : ℝ) 
  (h1 : perimeter = 24) 
  (h2 : base = 10) 
  (h3 : perimeter = base + 2 * leg) : 
  leg = 7 :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_leg_length_l1623_162367


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l1623_162397

theorem quadratic_inequality_condition (a : ℝ) :
  (∀ x : ℝ, (a^2 - 1) * x^2 + (a + 1) * x + 1/2 > 0) ↔ 
  (a ≤ -1 ∨ a > 3) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l1623_162397


namespace NUMINAMATH_CALUDE_min_max_tan_sum_l1623_162365

/-- The system of equations -/
def system (x y z : ℝ) : Prop :=
  (Real.tan x)^3 + (Real.tan y)^3 + (Real.tan z)^3 = 36 ∧
  (Real.tan x)^2 + (Real.tan y)^2 + (Real.tan z)^2 = 14 ∧
  ((Real.tan x)^2 + Real.tan y) * (Real.tan x + Real.tan z) * (Real.tan y + Real.tan z) = 60

/-- The theorem to prove -/
theorem min_max_tan_sum (x y z : ℝ) :
  system x y z →
  ∃ (min_tan max_tan : ℝ),
    (∀ w, system x w z → Real.tan x ≤ max_tan ∧ min_tan ≤ Real.tan x) ∧
    min_tan + max_tan = 4 :=
sorry

end NUMINAMATH_CALUDE_min_max_tan_sum_l1623_162365


namespace NUMINAMATH_CALUDE_complex_arithmetic_expression_l1623_162384

theorem complex_arithmetic_expression : (2019 - (2000 - (10 - 9))) - (2000 - (10 - (9 - 2019))) = 40 := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_expression_l1623_162384


namespace NUMINAMATH_CALUDE_largest_interior_angle_of_triangle_l1623_162376

theorem largest_interior_angle_of_triangle (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  (a : ℝ) / 2 = b / 3 → b / 3 = c / 4 →
  a + b + c = 360 →
  180 - min a (min b c) = 100 := by
  sorry

end NUMINAMATH_CALUDE_largest_interior_angle_of_triangle_l1623_162376


namespace NUMINAMATH_CALUDE_grass_field_width_l1623_162393

theorem grass_field_width (length width path_width cost_per_sqm total_cost : ℝ) :
  length = 95 →
  path_width = 2.5 →
  cost_per_sqm = 2 →
  total_cost = 1550 →
  (length + 2 * path_width) * (width + 2 * path_width) - length * width = total_cost / cost_per_sqm →
  width = 55 := by
sorry

end NUMINAMATH_CALUDE_grass_field_width_l1623_162393


namespace NUMINAMATH_CALUDE_cone_sphere_radius_theorem_l1623_162370

/-- Represents a right cone with a sphere inscribed within it. -/
structure ConeWithSphere where
  base_radius : ℝ
  height : ℝ
  sphere_radius : ℝ

/-- Checks if the sphere radius can be expressed in the form b√d - b. -/
def has_valid_sphere_radius (cone : ConeWithSphere) (b d : ℕ) : Prop :=
  cone.sphere_radius = b * Real.sqrt d - b

/-- The main theorem stating the relationship between b and d for the given cone. -/
theorem cone_sphere_radius_theorem (cone : ConeWithSphere) (b d : ℕ) :
  cone.base_radius = 15 →
  cone.height = 20 →
  has_valid_sphere_radius cone b d →
  b + d = 17 := by
  sorry

#check cone_sphere_radius_theorem

end NUMINAMATH_CALUDE_cone_sphere_radius_theorem_l1623_162370


namespace NUMINAMATH_CALUDE_trapezoid_area_l1623_162363

/-- The area of a trapezoid bounded by y=x, y=-x, x=10, and y=10 is 150 square units. -/
theorem trapezoid_area : Real := by
  -- Define the lines bounding the trapezoid
  let line1 : Real → Real := λ x => x
  let line2 : Real → Real := λ x => -x
  let line3 : Real → Real := λ _ => 10
  let vertical_line : Real := 10

  -- Define the trapezoid
  let trapezoid := {(x, y) : Real × Real | 
    (y = line1 x ∨ y = line2 x ∨ y = line3 x) ∧ 
    x ≤ vertical_line ∧ 
    y ≤ line3 x}

  -- Calculate the area of the trapezoid
  let area : Real := 150

  sorry -- Proof goes here

#check trapezoid_area

end NUMINAMATH_CALUDE_trapezoid_area_l1623_162363


namespace NUMINAMATH_CALUDE_sum_reciprocals_l1623_162358

theorem sum_reciprocals (a b c d : ℝ) (ω : ℂ) 
  (ha : a ≠ -2) (hb : b ≠ -2) (hc : c ≠ -2) (hd : d ≠ -2)
  (hω1 : ω^4 = 1) (hω2 : ω ≠ 1)
  (h : (1 / (a + ω)) + (1 / (b + ω)) + (1 / (c + ω)) + (1 / (d + ω)) = 3 / ω) :
  (1 / (a + 2)) + (1 / (b + 2)) + (1 / (c + 2)) + (1 / (d + 2)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocals_l1623_162358


namespace NUMINAMATH_CALUDE_quadratic_one_solution_l1623_162345

theorem quadratic_one_solution (a : ℝ) : 
  (∃! x : ℝ, x^2 + a*x + 1 = 0) ↔ (a = 2 ∨ a = -2) := by
sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_l1623_162345


namespace NUMINAMATH_CALUDE_cassidy_grounding_period_l1623_162389

/-- Calculates the total grounding period based on initial days, extra days per grade below B, and number of grades below B. -/
def total_grounding_period (initial_days : ℕ) (extra_days_per_grade : ℕ) (grades_below_b : ℕ) : ℕ :=
  initial_days + extra_days_per_grade * grades_below_b

/-- Proves that given the specified conditions, the total grounding period is 26 days. -/
theorem cassidy_grounding_period :
  total_grounding_period 14 3 4 = 26 := by
  sorry

end NUMINAMATH_CALUDE_cassidy_grounding_period_l1623_162389


namespace NUMINAMATH_CALUDE_mary_breeding_balls_l1623_162357

theorem mary_breeding_balls (snakes_per_ball : ℕ) (additional_pairs : ℕ) (total_snakes : ℕ) 
  (h1 : snakes_per_ball = 8)
  (h2 : additional_pairs = 6)
  (h3 : total_snakes = 36) :
  ∃ (num_balls : ℕ), 
    num_balls * snakes_per_ball + additional_pairs * 2 = total_snakes ∧ 
    num_balls = 3 := by
  sorry

end NUMINAMATH_CALUDE_mary_breeding_balls_l1623_162357


namespace NUMINAMATH_CALUDE_square_difference_l1623_162382

theorem square_difference (x y : ℚ) 
  (h1 : x + y = 8/15) 
  (h2 : x - y = 1/35) : 
  x^2 - y^2 = 8/525 := by
sorry

end NUMINAMATH_CALUDE_square_difference_l1623_162382


namespace NUMINAMATH_CALUDE_two_solutions_for_second_trace_l1623_162369

/-- Represents a trace of a plane -/
structure Trace where
  -- Add necessary fields

/-- Represents an inclination angle -/
structure InclinationAngle where
  -- Add necessary fields

/-- Represents a plane -/
structure Plane where
  firstTrace : Trace
  firstInclinationAngle : InclinationAngle
  axisPointOutside : Bool

/-- Represents a solution for the second trace -/
structure SecondTraceSolution where
  -- Add necessary fields

/-- 
Given a plane's first trace, first inclination angle, and the condition that the axis point 
is outside the drawing frame, there exist exactly two possible solutions for the second trace.
-/
theorem two_solutions_for_second_trace (p : Plane) : 
  p.axisPointOutside → ∃! (s : Finset SecondTraceSolution), s.card = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_solutions_for_second_trace_l1623_162369


namespace NUMINAMATH_CALUDE_charging_station_profit_l1623_162377

/-- Represents the total profit function for electric car charging stations -/
def profit_function (a b c : ℝ) (x : ℕ+) : ℝ := a * (x : ℝ)^2 + b * (x : ℝ) + c

theorem charging_station_profit 
  (a b c : ℝ) 
  (h1 : profit_function a b c 3 = 2) 
  (h2 : profit_function a b c 6 = 11) 
  (h3 : ∀ x : ℕ+, profit_function a b c x ≤ 11) :
  (∀ x : ℕ+, profit_function a b c x = -10 * (x : ℝ)^2 + 120 * (x : ℝ) - 250) ∧ 
  (∀ x : ℕ+, (profit_function a b c x) / (x : ℝ) ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_charging_station_profit_l1623_162377


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l1623_162331

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_fourth_term
  (a : ℕ → ℝ)
  (h_pos : ∀ n, a n > 0)
  (h_sum : a 1 + 2 * a 2 = 3)
  (h_prod : a 3 ^ 2 = 4 * a 2 * a 6)
  (h_geo : GeometricSequence a) :
  a 4 = 3 / 16 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l1623_162331


namespace NUMINAMATH_CALUDE_acute_triangle_trig_ranges_l1623_162378

variable (B C : Real)

theorem acute_triangle_trig_ranges 
  (acute : 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2)
  (angle_sum : B + C = π/3) :
  let A : Real := π/3
  (((3 + Real.sqrt 3) / 2 < Real.sin A + Real.sin B + Real.sin C) ∧ 
   (Real.sin A + Real.sin B + Real.sin C ≤ (6 + Real.sqrt 3) / 2)) ∧
  ((0 < Real.sin A * Real.sin B * Real.sin C) ∧ 
   (Real.sin A * Real.sin B * Real.sin C ≤ 3 * Real.sqrt 3 / 8)) := by
  sorry

end NUMINAMATH_CALUDE_acute_triangle_trig_ranges_l1623_162378


namespace NUMINAMATH_CALUDE_teacher_age_l1623_162368

theorem teacher_age (num_students : ℕ) (student_avg : ℝ) (new_avg : ℝ) : 
  num_students = 50 → 
  student_avg = 14 → 
  new_avg = 15 → 
  (num_students : ℝ) * student_avg + (new_avg * (num_students + 1) - num_students * student_avg) = 65 := by
  sorry

end NUMINAMATH_CALUDE_teacher_age_l1623_162368


namespace NUMINAMATH_CALUDE_final_values_after_assignments_l1623_162349

/-- This theorem proves that after a series of assignments, 
    the final values of a and b are both 4. -/
theorem final_values_after_assignments :
  let a₀ : ℕ := 3
  let b₀ : ℕ := 4
  let a₁ : ℕ := b₀
  let b₁ : ℕ := a₁
  (a₁ = 4 ∧ b₁ = 4) :=
by sorry

#check final_values_after_assignments

end NUMINAMATH_CALUDE_final_values_after_assignments_l1623_162349


namespace NUMINAMATH_CALUDE_stream_speed_l1623_162374

/-- Given a canoe that rows upstream at 4 km/hr and downstream at 12 km/hr, 
    the speed of the stream is 4 km/hr. -/
theorem stream_speed (upstream_speed downstream_speed : ℝ) 
  (h1 : upstream_speed = 4)
  (h2 : downstream_speed = 12) : 
  ∃ (canoe_speed stream_speed : ℝ),
    canoe_speed - stream_speed = upstream_speed ∧
    canoe_speed + stream_speed = downstream_speed ∧
    stream_speed = 4 := by
  sorry

end NUMINAMATH_CALUDE_stream_speed_l1623_162374


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l1623_162346

-- Define the ellipse type
structure Ellipse where
  endpoints : List (ℝ × ℝ)
  axes_perpendicular : Bool

-- Define the function to calculate the distance between foci
noncomputable def distance_between_foci (e : Ellipse) : ℝ :=
  sorry

-- Theorem statement
theorem ellipse_foci_distance 
  (e : Ellipse) 
  (h1 : e.endpoints = [(1, 3), (7, -5), (1, -5)])
  (h2 : e.axes_perpendicular = true) : 
  distance_between_foci e = 12 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l1623_162346


namespace NUMINAMATH_CALUDE_fraction_inequality_l1623_162381

theorem fraction_inequality (x : ℝ) : 
  -1 ≤ x ∧ x ≤ 3 → 
  (4 * x + 3 > 9 - 3 * x ↔ 6 / 7 < x ∧ x ≤ 3) :=
by sorry

end NUMINAMATH_CALUDE_fraction_inequality_l1623_162381


namespace NUMINAMATH_CALUDE_room_length_is_four_meters_l1623_162383

/-- Given the conditions of a room, prove its length is 4 meters -/
theorem room_length_is_four_meters 
  (breadth : ℝ) 
  (bricks_per_sqm : ℝ) 
  (total_bricks : ℝ) 
  (h_breadth : breadth = 5) 
  (h_bricks_per_sqm : bricks_per_sqm = 17) 
  (h_total_bricks : total_bricks = 340) : 
  total_bricks / bricks_per_sqm / breadth = 4 := by
  sorry


end NUMINAMATH_CALUDE_room_length_is_four_meters_l1623_162383


namespace NUMINAMATH_CALUDE_symmetric_function_properties_l1623_162398

/-- A function satisfying certain symmetry properties -/
structure SymmetricFunction where
  f : ℝ → ℝ
  sym_2 : ∀ x, f (2 - x) = f (2 + x)
  sym_7 : ∀ x, f (7 - x) = f (7 + x)
  zero_at_origin : f 0 = 0

/-- The number of zeros of a function in an interval -/
def num_zeros (f : ℝ → ℝ) (a b : ℝ) : ℕ := sorry

/-- A function is periodic with period p -/
def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

/-- Main theorem about SymmetricFunction -/
theorem symmetric_function_properties (sf : SymmetricFunction) :
  num_zeros sf.f (-30) 30 ≥ 13 ∧ is_periodic sf.f 10 := by sorry

end NUMINAMATH_CALUDE_symmetric_function_properties_l1623_162398


namespace NUMINAMATH_CALUDE_fraction_sum_l1623_162323

theorem fraction_sum (a b : ℚ) (h : a / b = 1 / 3) : (a + b) / b = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_l1623_162323


namespace NUMINAMATH_CALUDE_smaller_number_problem_l1623_162308

theorem smaller_number_problem (x y : ℕ) : 
  x * y = 56 → x + y = 15 → x ≤ y → x ∣ 28 → x = 7 := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_problem_l1623_162308


namespace NUMINAMATH_CALUDE_estimate_fish_population_verify_fish_estimate_l1623_162371

/-- Estimates the total number of fish in a pond using the mark-recapture method. -/
theorem estimate_fish_population (initial_catch : ℕ) (recapture : ℕ) (marked_recaught : ℕ) : ℕ :=
  let estimated_population := initial_catch * recapture / marked_recaught
  -- Proof that estimated_population = 750 given the conditions
  sorry

/-- Verifies the estimated fish population for the given problem. -/
theorem verify_fish_estimate : estimate_fish_population 30 50 2 = 750 := by
  -- Proof that the estimate is correct for the given values
  sorry

end NUMINAMATH_CALUDE_estimate_fish_population_verify_fish_estimate_l1623_162371


namespace NUMINAMATH_CALUDE_age_ratio_proof_l1623_162379

/-- Given that B's current age is 42 years and A is 12 years older than B,
    prove that the ratio of A's age in 10 years to B's age 10 years ago is 2:1 -/
theorem age_ratio_proof (B_current : ℕ) (A_current : ℕ) : 
  B_current = 42 →
  A_current = B_current + 12 →
  (A_current + 10) / (B_current - 10) = 2 := by
sorry

end NUMINAMATH_CALUDE_age_ratio_proof_l1623_162379


namespace NUMINAMATH_CALUDE_win_sector_area_l1623_162315

theorem win_sector_area (r : ℝ) (p : ℝ) (h1 : r = 8) (h2 : p = 3/8) :
  p * π * r^2 = 24 * π := by
  sorry

end NUMINAMATH_CALUDE_win_sector_area_l1623_162315


namespace NUMINAMATH_CALUDE_unique_prime_product_l1623_162304

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def distinct (a b c : ℕ) : Prop := a ≠ b ∧ b ≠ c ∧ a ≠ c

theorem unique_prime_product :
  ∀ n : ℕ,
  n ≠ 2103 →
  (∃ p q r : ℕ, is_prime p ∧ is_prime q ∧ is_prime r ∧ distinct p q r ∧ n = p * q * r) →
  ¬(∃ p1 p2 p3 : ℕ, is_prime p1 ∧ is_prime p2 ∧ is_prime p3 ∧ distinct p1 p2 p3 ∧ p1 + p2 + p3 = 59) :=
by sorry

end NUMINAMATH_CALUDE_unique_prime_product_l1623_162304


namespace NUMINAMATH_CALUDE_quality_difference_significant_frequency_machine_A_frequency_machine_B_l1623_162385

-- Define the contingency table
def machine_A_first_class : ℕ := 150
def machine_A_second_class : ℕ := 50
def machine_B_first_class : ℕ := 120
def machine_B_second_class : ℕ := 80
def total_products : ℕ := 400

-- Define the K^2 formula
def K_squared (a b c d n : ℕ) : ℚ :=
  (n * (a * d - b * c)^2 : ℚ) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Define the critical value for 99% confidence
def critical_value_99_percent : ℚ := 6635 / 1000

-- Theorem statement
theorem quality_difference_significant :
  K_squared machine_A_first_class machine_A_second_class
            machine_B_first_class machine_B_second_class
            total_products > critical_value_99_percent := by
  sorry

-- Frequencies of first-class products
theorem frequency_machine_A : (machine_A_first_class : ℚ) / (machine_A_first_class + machine_A_second_class) = 3 / 4 := by
  sorry

theorem frequency_machine_B : (machine_B_first_class : ℚ) / (machine_B_first_class + machine_B_second_class) = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_quality_difference_significant_frequency_machine_A_frequency_machine_B_l1623_162385


namespace NUMINAMATH_CALUDE_bob_has_62_pennies_l1623_162306

/-- The number of pennies Alex currently has -/
def a : ℕ := sorry

/-- The number of pennies Bob currently has -/
def b : ℕ := sorry

/-- Condition 1: If Alex gives Bob two pennies, Bob will have four times as many pennies as Alex has left -/
axiom condition1 : b + 2 = 4 * (a - 2)

/-- Condition 2: If Bob gives Alex two pennies, Bob will have three times as many pennies as Alex has -/
axiom condition2 : b - 2 = 3 * (a + 2)

/-- Theorem: Bob currently has 62 pennies -/
theorem bob_has_62_pennies : b = 62 := by sorry

end NUMINAMATH_CALUDE_bob_has_62_pennies_l1623_162306


namespace NUMINAMATH_CALUDE_smallest_a_for_equation_l1623_162319

theorem smallest_a_for_equation : 
  (∀ a : ℝ, a < -8 → ¬∃ b : ℝ, a^4 + 2*a^2*b + 2*a*b + b^2 = 960) ∧ 
  (∃ b : ℝ, (-8)^4 + 2*(-8)^2*b + 2*(-8)*b + b^2 = 960) := by
  sorry

end NUMINAMATH_CALUDE_smallest_a_for_equation_l1623_162319


namespace NUMINAMATH_CALUDE_tims_age_l1623_162300

/-- Given that Tom's age is 6 years more than 200% of Tim's age, 
    and Tom is 22 years old, Tim's age is 8 years. -/
theorem tims_age (tom_age tim_age : ℕ) 
  (h1 : tom_age = 2 * tim_age + 6)  -- Tom's age relation to Tim's
  (h2 : tom_age = 22)               -- Tom's actual age
  : tim_age = 8 := by
  sorry

#check tims_age

end NUMINAMATH_CALUDE_tims_age_l1623_162300


namespace NUMINAMATH_CALUDE_factorization_difference_l1623_162356

theorem factorization_difference (a b : ℤ) : 
  (∀ y : ℝ, 4 * y^2 - 3 * y - 28 = (4 * y + a) * (y + b)) → 
  a - b = 11 := by
sorry

end NUMINAMATH_CALUDE_factorization_difference_l1623_162356


namespace NUMINAMATH_CALUDE_stating_saucepan_capacity_l1623_162366

/-- Represents a cylindrical saucepan with a volume scale in cups. -/
structure Saucepan where
  capacity : ℝ
  partialFill : ℝ
  partialVolume : ℝ

/-- 
Theorem stating that a saucepan's capacity is 125 cups when 28% of it contains 35 cups.
-/
theorem saucepan_capacity (s : Saucepan) 
  (h1 : s.partialFill = 0.28)
  (h2 : s.partialVolume = 35) :
  s.capacity = 125 := by
  sorry

#check saucepan_capacity

end NUMINAMATH_CALUDE_stating_saucepan_capacity_l1623_162366


namespace NUMINAMATH_CALUDE_omega_value_l1623_162326

theorem omega_value (f : ℝ → ℝ) (ω : ℝ) : 
  (∀ x, f x = 2 * Real.cos (ω * x)) →
  ω > 0 →
  (∀ x₁ x₂, 0 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 2 * Real.pi / 3 → f x₁ > f x₂) →
  (∀ x, 0 ≤ x ∧ x ≤ 2 * Real.pi / 3 → f x ≥ 1) →
  f (2 * Real.pi / 3) = 1 →
  ω = 1 / 2 := by
sorry


end NUMINAMATH_CALUDE_omega_value_l1623_162326


namespace NUMINAMATH_CALUDE_polygon_sides_from_diagonals_l1623_162332

theorem polygon_sides_from_diagonals (D : ℕ) (n : ℕ) : D = n * (n - 3) / 2 → D = 44 → n = 11 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_from_diagonals_l1623_162332


namespace NUMINAMATH_CALUDE_tangent_relations_l1623_162392

theorem tangent_relations (α : Real) (h : Real.tan (α / 2) = 2) :
  Real.tan α = -4/3 ∧
  Real.tan (α + π/4) = -1/7 ∧
  (6 * Real.sin α + Real.cos α) / (3 * Real.sin α - 2 * Real.cos α) = 7/6 := by
  sorry

end NUMINAMATH_CALUDE_tangent_relations_l1623_162392


namespace NUMINAMATH_CALUDE_speed_calculation_l1623_162341

/-- 
Given a speed v, if increasing the speed by 21 miles per hour reduces the time by 1/3, 
then v must be 42 miles per hour.
-/
theorem speed_calculation (v : ℝ) : 
  (v * 1 = (v + 21) * (2/3)) → v = 42 := by
  sorry

end NUMINAMATH_CALUDE_speed_calculation_l1623_162341


namespace NUMINAMATH_CALUDE_inequality_proof_l1623_162318

theorem inequality_proof (a b : ℝ) (h : a < b) : 1 - a > 1 - b := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1623_162318


namespace NUMINAMATH_CALUDE_revenue_change_l1623_162352

/-- Proves that when the price increases by 50% and the quantity sold decreases by 20%, the revenue increases by 20% -/
theorem revenue_change 
  (P Q : ℝ) 
  (P' : ℝ) (hP' : P' = 1.5 * P) 
  (Q' : ℝ) (hQ' : Q' = 0.8 * Q) : 
  P' * Q' = 1.2 * (P * Q) := by
  sorry

#check revenue_change

end NUMINAMATH_CALUDE_revenue_change_l1623_162352


namespace NUMINAMATH_CALUDE_unique_three_digit_number_l1623_162320

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def has_nonzero_digits (n : ℕ) : Prop :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  hundreds ≠ 0 ∧ tens ≠ 0 ∧ ones ≠ 0

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

theorem unique_three_digit_number :
  ∃! n : ℕ, is_three_digit n ∧ has_nonzero_digits n ∧ 222 * (sum_of_digits n) - n = 1990 :=
sorry

end NUMINAMATH_CALUDE_unique_three_digit_number_l1623_162320


namespace NUMINAMATH_CALUDE_triangle_problem_l1623_162309

/-- Given a triangle ABC with side lengths a, b, c and angles A, B, C opposite to them respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement for the given triangle problem -/
theorem triangle_problem (t : Triangle) 
  (h1 : t.a = 3)
  (h2 : t.b - t.c = 2)
  (h3 : Real.cos t.B = -1/2) :
  t.b = 7 ∧ t.c = 5 ∧ Real.sin (t.B - t.C) = (4 * Real.sqrt 3) / 7 := by
  sorry


end NUMINAMATH_CALUDE_triangle_problem_l1623_162309


namespace NUMINAMATH_CALUDE_two_thirds_of_number_is_fifty_l1623_162343

theorem two_thirds_of_number_is_fifty (y : ℝ) : (2 / 3 : ℝ) * y = 50 → y = 75 := by
  sorry

end NUMINAMATH_CALUDE_two_thirds_of_number_is_fifty_l1623_162343


namespace NUMINAMATH_CALUDE_extremum_at_one_min_value_one_l1623_162328

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (a * x + 1) + (1 - x) / (1 + x)

-- Theorem 1: If f attains an extremum at x=1, then a = 1
theorem extremum_at_one (a : ℝ) (h : a > 0) :
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f a x ≤ f a 1 ∨ f a x ≥ f a 1) →
  a = 1 :=
sorry

-- Theorem 2: If the minimum value of f is 1, then a ≥ 2
theorem min_value_one (a : ℝ) (h : a > 0) :
  (∀ x : ℝ, f a x ≥ 1) →
  a ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_extremum_at_one_min_value_one_l1623_162328
