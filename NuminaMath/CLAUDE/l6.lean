import Mathlib

namespace NUMINAMATH_CALUDE_insurance_payment_count_l6_684

/-- Calculates the number of insurance payments per year -/
def insurance_payments_per_year (quarterly_payment : ℕ) (annual_total : ℕ) : ℕ :=
  annual_total / quarterly_payment

/-- Proves that the number of insurance payments per year is 4 -/
theorem insurance_payment_count :
  insurance_payments_per_year 378 1512 = 4 := by
  sorry

end NUMINAMATH_CALUDE_insurance_payment_count_l6_684


namespace NUMINAMATH_CALUDE_functional_equation_characterization_l6_661

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℕ → ℕ) : Prop :=
  ∀ m n : ℕ, f (m + f n) = f (f m) + f n

/-- The characterization of functions satisfying the functional equation -/
theorem functional_equation_characterization (f : ℕ → ℕ) 
  (h : FunctionalEquation f) : 
  ∃ d : ℕ, d > 0 ∧ ∀ m : ℕ, ∃ k : ℕ, f m = k * d :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_characterization_l6_661


namespace NUMINAMATH_CALUDE_p_adic_valuation_factorial_formula_l6_647

/-- The sum of digits of n in base p -/
def sum_of_digits (n : ℕ) (p : ℕ) : ℕ :=
  sorry

/-- The p-adic valuation of n! -/
def p_adic_valuation_factorial (p : ℕ) (n : ℕ) : ℕ :=
  sorry

/-- Theorem: The p-adic valuation of n! equals (n - s(n)) / (p - 1) -/
theorem p_adic_valuation_factorial_formula (p : ℕ) (n : ℕ) (hp : Prime p) (hn : n > 0) :
  p_adic_valuation_factorial p n = (n - sum_of_digits n p) / (p - 1) :=
sorry

end NUMINAMATH_CALUDE_p_adic_valuation_factorial_formula_l6_647


namespace NUMINAMATH_CALUDE_negative_integer_problem_l6_631

theorem negative_integer_problem (n : ℤ) : n < 0 → n * (-3) + 2 = 65 → n = -21 := by
  sorry

end NUMINAMATH_CALUDE_negative_integer_problem_l6_631


namespace NUMINAMATH_CALUDE_best_and_most_stable_values_l6_699

/-- Represents a student's performance data -/
structure StudentPerformance where
  average : ℝ
  variance : ℝ

/-- The given data for students B, C, and D -/
def studentB : StudentPerformance := ⟨90, 12.5⟩
def studentC : StudentPerformance := ⟨91, 14.5⟩
def studentD : StudentPerformance := ⟨88, 11⟩

/-- Conditions for Student A to be the best-performing and most stable -/
def isBestAndMostStable (m n : ℝ) : Prop :=
  m > studentB.average ∧
  m > studentC.average ∧
  m > studentD.average ∧
  n < studentB.variance ∧
  n < studentC.variance ∧
  n < studentD.variance

/-- Theorem stating that m = 92 and n = 8.5 are the only values satisfying the conditions -/
theorem best_and_most_stable_values :
  ∀ m n : ℝ, isBestAndMostStable m n ↔ m = 92 ∧ n = 8.5 := by
  sorry

end NUMINAMATH_CALUDE_best_and_most_stable_values_l6_699


namespace NUMINAMATH_CALUDE_f_decreasing_interval_a_upper_bound_l6_648

-- Define the function f(x) = x ln x
noncomputable def f (x : ℝ) : ℝ := x * Real.log x

-- Theorem for the monotonically decreasing interval
theorem f_decreasing_interval :
  ∀ x ∈ Set.Ioo (0 : ℝ) (Real.exp (-1)),
  StrictMonoOn f (Set.Ioo 0 (Real.exp (-1))) :=
sorry

-- Theorem for the range of a
theorem a_upper_bound
  (h : ∀ x > 0, f x ≥ -x^2 + a*x - 6) :
  a ≤ 5 + Real.log 2 :=
sorry

end NUMINAMATH_CALUDE_f_decreasing_interval_a_upper_bound_l6_648


namespace NUMINAMATH_CALUDE_circle_center_l6_625

/-- A circle is tangent to two parallel lines and its center lies on a third line. -/
theorem circle_center (x y : ℝ) :
  (3 * x + 4 * y = 24 ∨ 3 * x + 4 * y = -6) →  -- Circle is tangent to these lines
  (3 * x - y = 0) →                           -- Center lies on this line
  (x = 3/5 ∧ y = 9/5) →                       -- Proposed center coordinates
  ∃ (r : ℝ), r > 0 ∧                          -- There exists a positive radius
    (∀ (x' y' : ℝ), (x' - x)^2 + (y' - y)^2 = r^2 →  -- Points on the circle
      (3 * x' + 4 * y' = 24 ∨ 3 * x' + 4 * y' = -6))  -- Touch the given lines
  := by sorry


end NUMINAMATH_CALUDE_circle_center_l6_625


namespace NUMINAMATH_CALUDE_dryer_price_difference_dryer_costs_less_l6_637

/-- Given a washing machine price of $100 and a dryer with an unknown price,
    if there's a 10% discount on the total and the final price is $153,
    then the dryer costs $30 less than the washing machine. -/
theorem dryer_price_difference (dryer_price : ℝ) : 
  (100 + dryer_price) * 0.9 = 153 → dryer_price = 70 :=
by
  sorry

/-- The difference in price between the washing machine and the dryer -/
def price_difference : ℝ := 100 - 70

theorem dryer_costs_less : price_difference = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_dryer_price_difference_dryer_costs_less_l6_637


namespace NUMINAMATH_CALUDE_west_7m_is_negative_7m_l6_607

/-- Represents the direction of movement on an east-west road -/
inductive Direction
  | East
  | West

/-- Represents a movement on the road with a direction and distance -/
structure Movement where
  direction : Direction
  distance : ℝ

/-- Converts a movement to its signed representation -/
def Movement.toSigned (m : Movement) : ℝ :=
  match m.direction with
  | Direction.East => m.distance
  | Direction.West => -m.distance

/-- The theorem stating that moving west by 7m should be denoted as -7m -/
theorem west_7m_is_negative_7m
  (h : Movement.toSigned { direction := Direction.East, distance := 3 } = 3) :
  Movement.toSigned { direction := Direction.West, distance := 7 } = -7 := by
  sorry

end NUMINAMATH_CALUDE_west_7m_is_negative_7m_l6_607


namespace NUMINAMATH_CALUDE_p_geq_q_l6_608

theorem p_geq_q (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : a^a * b^b ≥ a^b * b^a := by
  sorry

end NUMINAMATH_CALUDE_p_geq_q_l6_608


namespace NUMINAMATH_CALUDE_sine_cosine_inequality_l6_689

theorem sine_cosine_inequality (x : ℝ) (n m : ℕ) 
  (h1 : 0 < x) (h2 : x < π / 2) (h3 : n > m) : 
  2 * |Real.sin x ^ n - Real.cos x ^ n| ≤ 3 * |Real.sin x ^ m - Real.cos x ^ m| := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_inequality_l6_689


namespace NUMINAMATH_CALUDE_pond_width_calculation_l6_688

/-- Represents a rectangular pond -/
structure RectangularPond where
  length : ℝ
  width : ℝ
  depth : ℝ
  volume : ℝ

/-- Theorem: Given a rectangular pond with length 20 meters, depth 5 meters, 
    and volume 1200 cubic meters, its width is 12 meters -/
theorem pond_width_calculation (pond : RectangularPond) 
  (h1 : pond.length = 20)
  (h2 : pond.depth = 5)
  (h3 : pond.volume = 1200)
  (h4 : pond.volume = pond.length * pond.width * pond.depth) :
  pond.width = 12 := by
  sorry

end NUMINAMATH_CALUDE_pond_width_calculation_l6_688


namespace NUMINAMATH_CALUDE_trillion_equals_ten_to_sixteen_l6_601

-- Define the relationships between numbers
def ten_thousand : ℕ := 10^4
def million : ℕ := 10^6
def billion : ℕ := 10^8
def trillion : ℕ := ten_thousand * million * billion

-- Theorem statement
theorem trillion_equals_ten_to_sixteen : trillion = 10^16 := by
  sorry

end NUMINAMATH_CALUDE_trillion_equals_ten_to_sixteen_l6_601


namespace NUMINAMATH_CALUDE_refrigerator_price_calculation_l6_602

/-- The purchase price of the refrigerator in rupees -/
def refrigerator_price : ℝ := 15000

/-- The purchase price of the mobile phone in rupees -/
def mobile_price : ℝ := 8000

/-- The loss percentage on the refrigerator -/
def refrigerator_loss_percent : ℝ := 0.03

/-- The profit percentage on the mobile phone -/
def mobile_profit_percent : ℝ := 0.10

/-- The overall profit in rupees -/
def overall_profit : ℝ := 350

theorem refrigerator_price_calculation :
  refrigerator_price * (1 - refrigerator_loss_percent) +
  mobile_price * (1 + mobile_profit_percent) =
  refrigerator_price + mobile_price + overall_profit := by
  sorry

end NUMINAMATH_CALUDE_refrigerator_price_calculation_l6_602


namespace NUMINAMATH_CALUDE_lisa_marbles_problem_l6_618

/-- The minimum number of additional marbles needed -/
def min_additional_marbles (num_friends : ℕ) (initial_marbles : ℕ) : ℕ :=
  (num_friends * (num_friends + 1)) / 2 - initial_marbles

/-- Theorem stating the minimum number of additional marbles needed for Lisa's problem -/
theorem lisa_marbles_problem :
  min_additional_marbles 12 50 = 28 := by
  sorry

end NUMINAMATH_CALUDE_lisa_marbles_problem_l6_618


namespace NUMINAMATH_CALUDE_principal_amount_proof_l6_639

/-- Compound interest calculation --/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- Problem statement --/
theorem principal_amount_proof :
  let final_amount : ℝ := 8820
  let rate : ℝ := 0.05
  let time : ℕ := 2
  ∃ (principal : ℝ), principal = 8000 ∧ compound_interest principal rate time = final_amount := by
sorry

end NUMINAMATH_CALUDE_principal_amount_proof_l6_639


namespace NUMINAMATH_CALUDE_sequence_matches_formula_l6_694

-- Define the sequence
def a (n : ℕ) : ℚ := (-1)^(n+1) * (2*n + 1) / 2^n

-- State the theorem
theorem sequence_matches_formula : 
  a 1 = 3/2 ∧ a 2 = -5/4 ∧ a 3 = 7/8 ∧ a 4 = -9/16 := by
  sorry

end NUMINAMATH_CALUDE_sequence_matches_formula_l6_694


namespace NUMINAMATH_CALUDE_divisibility_of_sixth_power_difference_l6_671

theorem divisibility_of_sixth_power_difference (a b : ℤ) 
  (ha : ¬ 3 ∣ a) (hb : ¬ 3 ∣ b) : 
  9 ∣ (a^6 - b^6) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_of_sixth_power_difference_l6_671


namespace NUMINAMATH_CALUDE_correct_calculation_l6_678

theorem correct_calculation (x y : ℝ) : 3 * x^2 * y + 2 * y * x^2 = 5 * x^2 * y := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l6_678


namespace NUMINAMATH_CALUDE_rationalize_denominator_sum_l6_610

theorem rationalize_denominator_sum :
  ∃ (A B C D E F : ℤ),
    (F > 0) ∧
    (∀ (x : ℝ), x > 0 → (1 / (Real.sqrt 5 + Real.sqrt 2 + Real.sqrt 11) = 
      (A * Real.sqrt 2 + B * Real.sqrt 5 + C * Real.sqrt 11 + D * Real.sqrt E) / F)) ∧
    (A + B + C + D + E + F = 136) :=
by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_sum_l6_610


namespace NUMINAMATH_CALUDE_frog_jump_probability_l6_643

/-- Represents a jump as a vector in 3D space -/
structure Jump where
  x : ℝ
  y : ℝ
  z : ℝ
  magnitude_is_one : x^2 + y^2 + z^2 = 1

/-- Represents the frog's position after a series of jumps -/
def FinalPosition (jumps : List Jump) : ℝ × ℝ × ℝ :=
  let sum := jumps.foldl (fun (ax, ay, az) j => (ax + j.x, ay + j.y, az + j.z)) (0, 0, 0)
  sum

/-- The probability of the frog's final position being exactly 1 meter from the start -/
noncomputable def probability_one_meter_away (num_jumps : ℕ) : ℝ :=
  sorry

/-- Theorem stating the probability for 4 jumps is 1/8 -/
theorem frog_jump_probability :
  probability_one_meter_away 4 = 1/8 := by sorry

end NUMINAMATH_CALUDE_frog_jump_probability_l6_643


namespace NUMINAMATH_CALUDE_hannah_remaining_money_l6_654

def county_fair_expenses (initial_amount : ℚ) (ride_percentage : ℚ) (game_percentage : ℚ)
  (dessert_cost : ℚ) (cotton_candy_cost : ℚ) (hotdog_cost : ℚ)
  (keychain_cost : ℚ) (poster_cost : ℚ) (attraction_cost : ℚ) : ℚ :=
  let ride_expense := initial_amount * ride_percentage
  let game_expense := initial_amount * game_percentage
  let food_souvenir_expense := dessert_cost + cotton_candy_cost + hotdog_cost + keychain_cost + poster_cost + attraction_cost
  initial_amount - (ride_expense + game_expense + food_souvenir_expense)

theorem hannah_remaining_money :
  county_fair_expenses 120 0.4 0.15 8 5 6 7 10 15 = 3 := by
  sorry

end NUMINAMATH_CALUDE_hannah_remaining_money_l6_654


namespace NUMINAMATH_CALUDE_two_digit_perfect_squares_divisible_by_four_l6_663

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem two_digit_perfect_squares_divisible_by_four :
  ∃! (s : Finset ℕ), 
    (∀ n ∈ s, is_two_digit n ∧ is_perfect_square n ∧ n % 4 = 0) ∧
    s.card = 3 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_perfect_squares_divisible_by_four_l6_663


namespace NUMINAMATH_CALUDE_downstream_distance_l6_655

/-- Prove that given the conditions of a boat rowing upstream and downstream,
    the distance rowed downstream is 200 km. -/
theorem downstream_distance
  (boat_speed : ℝ)
  (upstream_distance : ℝ)
  (upstream_time : ℝ)
  (downstream_time : ℝ)
  (h1 : boat_speed = 14)
  (h2 : upstream_distance = 96)
  (h3 : upstream_time = 12)
  (h4 : downstream_time = 10)
  (h5 : upstream_distance / upstream_time = boat_speed - (boat_speed - upstream_distance / upstream_time)) :
  (boat_speed + (boat_speed - upstream_distance / upstream_time)) * downstream_time = 200 := by
sorry

end NUMINAMATH_CALUDE_downstream_distance_l6_655


namespace NUMINAMATH_CALUDE_other_solution_of_quadratic_l6_615

theorem other_solution_of_quadratic (x : ℚ) : 
  (65 * (6/5)^2 + 18 = 104 * (6/5) - 13) →
  (65 * x^2 + 18 = 104 * x - 13) →
  (x ≠ 6/5) →
  x = 5/13 := by
sorry

end NUMINAMATH_CALUDE_other_solution_of_quadratic_l6_615


namespace NUMINAMATH_CALUDE_gumball_ratio_l6_635

/-- Represents the gumball problem scenario -/
structure GumballScenario where
  alicia_gumballs : ℕ
  pedro_multiplier : ℚ
  remaining_gumballs : ℕ

/-- The specific scenario given in the problem -/
def problem_scenario : GumballScenario :=
  { alicia_gumballs := 20
  , pedro_multiplier := 3
  , remaining_gumballs := 60 }

/-- Calculates the total number of gumballs initially in the bowl -/
def total_gumballs (s : GumballScenario) : ℚ :=
  s.alicia_gumballs * (2 + s.pedro_multiplier)

/-- Calculates Pedro's additional gumballs -/
def pedro_additional_gumballs (s : GumballScenario) : ℚ :=
  s.alicia_gumballs * s.pedro_multiplier

/-- The main theorem to prove -/
theorem gumball_ratio (s : GumballScenario) :
  s.alicia_gumballs = 20 →
  s.remaining_gumballs = 60 →
  (total_gumballs s * (3/5) : ℚ) = s.remaining_gumballs →
  (pedro_additional_gumballs s) / s.alicia_gumballs = 3 :=
by sorry

#check gumball_ratio problem_scenario

end NUMINAMATH_CALUDE_gumball_ratio_l6_635


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l6_677

theorem complex_fraction_simplification :
  (3 / 7 - 2 / 5) / (5 / 12 + 1 / 4) = 3 / 70 := by
sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l6_677


namespace NUMINAMATH_CALUDE_triangle_area_l6_674

/-- Given a triangle with perimeter 28 cm, inradius 2.5 cm, one angle of 75 degrees,
    and side lengths in the ratio 3:4:5, prove that its area is 35 cm². -/
theorem triangle_area (p : ℝ) (r : ℝ) (angle : ℝ) (a b c : ℝ)
  (h_perimeter : p = 28)
  (h_inradius : r = 2.5)
  (h_angle : angle = 75)
  (h_ratio : ∃ k : ℝ, a = 3 * k ∧ b = 4 * k ∧ c = 5 * k)
  (h_sides : a + b + c = p) :
  r * p / 2 = 35 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l6_674


namespace NUMINAMATH_CALUDE_parking_lot_wheels_l6_658

/-- The number of wheels on a car -/
def wheels_per_car : ℕ := 4

/-- The number of wheels on a bike -/
def wheels_per_bike : ℕ := 2

/-- The number of cars in the parking lot -/
def num_cars : ℕ := 10

/-- The number of bikes in the parking lot -/
def num_bikes : ℕ := 2

/-- The total number of wheels in the parking lot -/
def total_wheels : ℕ := num_cars * wheels_per_car + num_bikes * wheels_per_bike

theorem parking_lot_wheels : total_wheels = 44 := by
  sorry

end NUMINAMATH_CALUDE_parking_lot_wheels_l6_658


namespace NUMINAMATH_CALUDE_primes_between_50_and_60_l6_685

def count_primes_in_range (a b : ℕ) : ℕ :=
  (Finset.range (b - a + 1)).filter (fun i => Nat.Prime (i + a)) |>.card

theorem primes_between_50_and_60 : count_primes_in_range 50 60 = 2 := by
  sorry

end NUMINAMATH_CALUDE_primes_between_50_and_60_l6_685


namespace NUMINAMATH_CALUDE_coin_game_probability_l6_633

def num_players : ℕ := 4
def initial_coins : ℕ := 5
def num_rounds : ℕ := 5
def num_balls : ℕ := 5
def num_green : ℕ := 2
def num_red : ℕ := 1
def num_white : ℕ := 2

def coin_transfer : ℕ := 2

def game_round_probability : ℚ := 1 / 5

theorem coin_game_probability :
  (game_round_probability ^ num_rounds : ℚ) = 1 / 3125 := by
  sorry

end NUMINAMATH_CALUDE_coin_game_probability_l6_633


namespace NUMINAMATH_CALUDE_lowest_common_multiple_10_to_30_l6_653

theorem lowest_common_multiple_10_to_30 :
  ∃ (n : ℕ), n > 0 ∧
  (∀ k : ℕ, 10 ≤ k ∧ k ≤ 30 → k ∣ n) ∧
  (∀ m : ℕ, m > 0 ∧ (∀ k : ℕ, 10 ≤ k ∧ k ≤ 30 → k ∣ m) → n ≤ m) ∧
  n = 232792560 :=
by sorry

end NUMINAMATH_CALUDE_lowest_common_multiple_10_to_30_l6_653


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l6_675

/-- Given a geometric sequence {a_n} where a_2 + a_3 = 1 and a_3 + a_4 = -2,
    prove that a_5 + a_6 + a_7 = 24 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  (∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q) →  -- {a_n} is a geometric sequence
  a 2 + a 3 = 1 →                           -- a_2 + a_3 = 1
  a 3 + a 4 = -2 →                          -- a_3 + a_4 = -2
  a 5 + a 6 + a 7 = 24 :=                   -- Conclusion to prove
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l6_675


namespace NUMINAMATH_CALUDE_inequality_proof_l6_600

theorem inequality_proof (x y z : ℝ) 
  (sum_zero : x + y + z = 0) 
  (abs_sum_le_one : |x| + |y| + |z| ≤ 1) : 
  x + y/2 + z/3 ≤ 1/3 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l6_600


namespace NUMINAMATH_CALUDE_triangle_area_l6_646

/-- The area of a triangle with sides 5, 4, and 4 units is (5√39)/4 square units. -/
theorem triangle_area (a b c : ℝ) (h1 : a = 5) (h2 : b = 4) (h3 : c = 4) :
  (1/2 : ℝ) * a * (((b^2 - (a/2)^2).sqrt : ℝ)) = (5 * Real.sqrt 39) / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l6_646


namespace NUMINAMATH_CALUDE_digit_value_difference_l6_604

/-- The numeral we are working with -/
def numeral : ℕ := 657903

/-- The digit we are focusing on -/
def digit : ℕ := 7

/-- The position of the digit in the numeral (counting from right, starting at 0) -/
def position : ℕ := 4

/-- The local value of a digit in a given position -/
def local_value (d : ℕ) (pos : ℕ) : ℕ := d * (10 ^ pos)

/-- The face value of a digit -/
def face_value (d : ℕ) : ℕ := d

/-- The difference between local value and face value -/
def value_difference (d : ℕ) (pos : ℕ) : ℕ := local_value d pos - face_value d

theorem digit_value_difference :
  value_difference digit position = 69993 := by sorry

end NUMINAMATH_CALUDE_digit_value_difference_l6_604


namespace NUMINAMATH_CALUDE_ellipse_equation_correct_l6_622

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ

/-- Check if a point lies on an ellipse -/
def pointOnEllipse (p : Point) (e : Ellipse) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

/-- Check if two points are foci of an ellipse -/
def areFoci (f1 f2 : Point) (e : Ellipse) : Prop :=
  (f2.x - f1.x)^2 / 4 = e.a^2 - e.b^2

theorem ellipse_equation_correct (P A B : Point) (E : Ellipse) :
  P.x = 5/2 ∧ P.y = -3/2 ∧
  A.x = -2 ∧ A.y = 0 ∧
  B.x = 2 ∧ B.y = 0 ∧
  E.a^2 = 10 ∧ E.b^2 = 6 →
  pointOnEllipse P E ∧ areFoci A B E := by
  sorry

#check ellipse_equation_correct

end NUMINAMATH_CALUDE_ellipse_equation_correct_l6_622


namespace NUMINAMATH_CALUDE_burgers_spent_l6_641

def total_allowance : ℚ := 50

def movies_fraction : ℚ := 1/4
def music_fraction : ℚ := 3/10
def ice_cream_fraction : ℚ := 2/5

def burgers_amount : ℚ := total_allowance - (movies_fraction * total_allowance + music_fraction * total_allowance + ice_cream_fraction * total_allowance)

theorem burgers_spent :
  burgers_amount = 5/2 := by sorry

end NUMINAMATH_CALUDE_burgers_spent_l6_641


namespace NUMINAMATH_CALUDE_find_c_l6_626

theorem find_c (p q : ℝ → ℝ) (c : ℝ) : 
  (∀ x, p x = 4 * x - 3) →
  (∀ x, q x = 5 * x - c) →
  p (q 3) = 53 →
  c = 1 := by
sorry

end NUMINAMATH_CALUDE_find_c_l6_626


namespace NUMINAMATH_CALUDE_paper_length_calculation_l6_680

/-- Calculates the length of paper wrapped around a tube -/
theorem paper_length_calculation 
  (paper_width : ℝ) 
  (initial_diameter : ℝ) 
  (final_diameter : ℝ) 
  (num_layers : ℕ) 
  (h1 : paper_width = 4)
  (h2 : initial_diameter = 4)
  (h3 : final_diameter = 16)
  (h4 : num_layers = 500) :
  (π * num_layers * (initial_diameter + final_diameter) / 2) / 100 = 50 * π := by
sorry

end NUMINAMATH_CALUDE_paper_length_calculation_l6_680


namespace NUMINAMATH_CALUDE_abs_sum_iff_positive_l6_650

theorem abs_sum_iff_positive (x y : ℝ) : x + y > |x - y| ↔ x > 0 ∧ y > 0 := by
  sorry

end NUMINAMATH_CALUDE_abs_sum_iff_positive_l6_650


namespace NUMINAMATH_CALUDE_coefficient_x_sqrt_x_in_expansion_l6_651

theorem coefficient_x_sqrt_x_in_expansion :
  let expansion := (λ x : ℝ => (Real.sqrt x - 1)^5)
  ∃ c : ℝ, ∀ x : ℝ, x > 0 →
    expansion x = c * x * Real.sqrt x + (λ y => y - c * x * Real.sqrt x) (expansion x) ∧
    c = 10 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_sqrt_x_in_expansion_l6_651


namespace NUMINAMATH_CALUDE_product_inequality_l6_638

theorem product_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + 1) * (b + 1) * (a + c) * (b + c) ≥ 16 * a * b * c := by
  sorry

end NUMINAMATH_CALUDE_product_inequality_l6_638


namespace NUMINAMATH_CALUDE_smallest_number_with_unique_digits_divisible_by_990_l6_682

theorem smallest_number_with_unique_digits_divisible_by_990 : ∃ (n : ℕ), 
  (n = 1234758690) ∧ 
  (∀ m : ℕ, m < n → ¬(∀ d : Fin 10, (m.digits 10).count d = 1)) ∧
  (∀ d : Fin 10, (n.digits 10).count d = 1) ∧
  (n % 990 = 0) := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_with_unique_digits_divisible_by_990_l6_682


namespace NUMINAMATH_CALUDE_simple_interest_rate_calculation_l6_657

theorem simple_interest_rate_calculation 
  (initial_sum : ℝ) 
  (final_amount : ℝ) 
  (time : ℝ) 
  (h1 : initial_sum = 12500)
  (h2 : final_amount = 15500)
  (h3 : time = 4)
  (h4 : final_amount = initial_sum * (1 + time * (rate / 100))) :
  rate = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_simple_interest_rate_calculation_l6_657


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l6_681

theorem quadratic_equation_solution :
  ∃! y : ℝ, y^2 + 6*y + 8 = -(y + 4)*(y + 6) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l6_681


namespace NUMINAMATH_CALUDE_negation_of_forall_exp_positive_l6_632

theorem negation_of_forall_exp_positive :
  (¬ ∀ x : ℝ, Real.exp x > 0) ↔ (∃ x : ℝ, Real.exp x ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_forall_exp_positive_l6_632


namespace NUMINAMATH_CALUDE_oil_price_reduction_l6_687

/-- Proves that given a 20% reduction in the price of oil, if a housewife can obtain 10 kgs more for Rs. 1500 after the reduction, then the reduced price per kg is Rs. 30. -/
theorem oil_price_reduction (original_price : ℝ) : 
  (1500 / (0.8 * original_price) - 1500 / original_price = 10) → 
  (0.8 * original_price = 30) := by
sorry

end NUMINAMATH_CALUDE_oil_price_reduction_l6_687


namespace NUMINAMATH_CALUDE_janeles_cats_average_weight_l6_676

/-- The average weight of Janele's cats -/
theorem janeles_cats_average_weight :
  let num_cats : ℕ := 4
  let weight_cat1 : ℚ := 12
  let weight_cat2 : ℚ := 12
  let weight_cat3 : ℚ := 147/10
  let weight_cat4 : ℚ := 93/10
  let total_weight : ℚ := weight_cat1 + weight_cat2 + weight_cat3 + weight_cat4
  let average_weight : ℚ := total_weight / num_cats
  average_weight = 12 := by
  sorry

end NUMINAMATH_CALUDE_janeles_cats_average_weight_l6_676


namespace NUMINAMATH_CALUDE_divisible_by_six_ratio_l6_629

theorem divisible_by_six_ratio (n : ℕ) : n = 120 →
  (Finset.filter (fun x => x % 6 = 0) (Finset.range (n + 1))).card / (n + 1 : ℚ) = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_six_ratio_l6_629


namespace NUMINAMATH_CALUDE_gcd_of_98_and_63_l6_691

theorem gcd_of_98_and_63 : Nat.gcd 98 63 = 7 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_98_and_63_l6_691


namespace NUMINAMATH_CALUDE_max_students_distribution_l6_664

theorem max_students_distribution (pens pencils : ℕ) (h1 : pens = 2730) (h2 : pencils = 1890) :
  Nat.gcd pens pencils = 210 := by
  sorry

end NUMINAMATH_CALUDE_max_students_distribution_l6_664


namespace NUMINAMATH_CALUDE_min_mutual_greetings_school_l6_673

/-- Represents a school with students and their greetings. -/
structure School :=
  (num_students : Nat)
  (greetings_per_student : Nat)
  (h_students : num_students = 400)
  (h_greetings : greetings_per_student = 200)

/-- The minimum number of pairs of students who have mutually greeted each other. -/
def min_mutual_greetings (s : School) : Nat :=
  s.greetings_per_student * s.num_students - Nat.choose s.num_students 2

/-- Theorem stating the minimum number of mutual greetings in the given school. -/
theorem min_mutual_greetings_school :
    ∀ s : School, min_mutual_greetings s = 200 :=
  sorry

end NUMINAMATH_CALUDE_min_mutual_greetings_school_l6_673


namespace NUMINAMATH_CALUDE_tangent_line_implies_a_minus_b_l6_686

noncomputable def f (a b x : ℝ) : ℝ := x + a / x + b

theorem tangent_line_implies_a_minus_b (a b : ℝ) :
  (∀ x ≠ 0, HasDerivAt (f a b) (1 - a / (x^2)) x) →
  (f a b 1 = 1 + a + b) →
  (HasDerivAt (f a b) (-2) 1) →
  (∃ c, ∀ x, f a b x = -2 * x + c) →
  a - b = 4 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_implies_a_minus_b_l6_686


namespace NUMINAMATH_CALUDE_remainder_sum_l6_695

theorem remainder_sum (n : ℤ) (h : n % 20 = 9) : (n % 4 + n % 5 = 5) := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_l6_695


namespace NUMINAMATH_CALUDE_minimum_value_problems_l6_628

theorem minimum_value_problems :
  (∀ x > 0, x + 1/x ≥ 2 ∧ (x + 1/x = 2 ↔ x = 1)) ∧
  (∀ m > 0, (m^2 + 5*m + 12) / m ≥ 4 * Real.sqrt 3 + 5) := by
  sorry

end NUMINAMATH_CALUDE_minimum_value_problems_l6_628


namespace NUMINAMATH_CALUDE_intersection_x_coordinate_l6_672

-- Define the equations of the two lines
def line1 (x y : ℝ) : Prop := y = 3 * x - 17
def line2 (x y : ℝ) : Prop := 3 * x + y = 103

-- Theorem stating that the x-coordinate of the intersection is 20
theorem intersection_x_coordinate :
  ∃ (x y : ℝ), line1 x y ∧ line2 x y ∧ x = 20 := by
  sorry

end NUMINAMATH_CALUDE_intersection_x_coordinate_l6_672


namespace NUMINAMATH_CALUDE_chord_squares_difference_l6_652

/-- Given a circle with a chord at distance h from the center, and two squares inscribed in the 
segments subtended by the chord (with two adjacent vertices on the arc and two on the chord or 
its extension), the difference in the side lengths of these squares is 8h/5. -/
theorem chord_squares_difference (h : ℝ) (h_pos : h > 0) : ℝ := by
  sorry

end NUMINAMATH_CALUDE_chord_squares_difference_l6_652


namespace NUMINAMATH_CALUDE_inverse_as_linear_combination_l6_640

def N : Matrix (Fin 2) (Fin 2) ℚ := !![3, -1; 2, -4]

theorem inverse_as_linear_combination :
  ∃ (c d : ℚ), N⁻¹ = c • N + d • (1 : Matrix (Fin 2) (Fin 2) ℚ) ∧ c = 1/10 ∧ d = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_inverse_as_linear_combination_l6_640


namespace NUMINAMATH_CALUDE_max_notebooks_purchase_l6_667

theorem max_notebooks_purchase (total_items : ℕ) (notebook_cost pencil_case_cost max_cost : ℚ) :
  total_items = 10 →
  notebook_cost = 12 →
  pencil_case_cost = 7 →
  max_cost = 100 →
  (∀ x : ℕ, x ≤ total_items →
    x * notebook_cost + (total_items - x) * pencil_case_cost ≤ max_cost →
    x ≤ 6) ∧
  ∃ x : ℕ, x = 6 ∧ x ≤ total_items ∧
    x * notebook_cost + (total_items - x) * pencil_case_cost ≤ max_cost :=
by sorry

end NUMINAMATH_CALUDE_max_notebooks_purchase_l6_667


namespace NUMINAMATH_CALUDE_staircase_extension_l6_668

/-- Calculates the number of toothpicks needed for a staircase of n steps -/
def toothpicks (n : ℕ) : ℕ := 
  if n = 0 then 0
  else if n = 1 then 4
  else 4 + (n - 1) * 3 + ((n - 1) * (n - 2)) / 2

/-- The number of additional toothpicks needed to extend an n-step staircase to an m-step staircase -/
def additional_toothpicks (n m : ℕ) : ℕ := toothpicks m - toothpicks n

theorem staircase_extension :
  additional_toothpicks 3 6 = 36 :=
sorry

end NUMINAMATH_CALUDE_staircase_extension_l6_668


namespace NUMINAMATH_CALUDE_integer_points_count_l6_690

/-- Represents a line segment on a number line -/
structure LineSegment where
  start : ℝ
  length : ℝ

/-- Counts the number of integer points covered by a line segment -/
def count_integer_points (segment : LineSegment) : ℕ :=
  sorry

/-- Theorem stating that a line segment of length 2020 covers either 2020 or 2021 integer points -/
theorem integer_points_count (segment : LineSegment) :
  segment.length = 2020 → count_integer_points segment = 2020 ∨ count_integer_points segment = 2021 :=
sorry

end NUMINAMATH_CALUDE_integer_points_count_l6_690


namespace NUMINAMATH_CALUDE_multiply_658217_and_99999_l6_662

theorem multiply_658217_and_99999 : 658217 * 99999 = 65821034183 := by
  sorry

end NUMINAMATH_CALUDE_multiply_658217_and_99999_l6_662


namespace NUMINAMATH_CALUDE_equation_solution_l6_669

theorem equation_solution :
  ∃ x : ℚ, (5 * x + 6 * x = 360 - 10 * (x - 4)) ∧ x = 400 / 21 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l6_669


namespace NUMINAMATH_CALUDE_x_gt_y_necessary_not_sufficient_l6_623

theorem x_gt_y_necessary_not_sufficient (x y : ℝ) (hx : x > 0) :
  (∀ y, x > |y| → x > y) ∧ 
  (∃ y, x > y ∧ ¬(x > |y|)) := by
  sorry

end NUMINAMATH_CALUDE_x_gt_y_necessary_not_sufficient_l6_623


namespace NUMINAMATH_CALUDE_smallest_zero_floor_is_three_l6_693

noncomputable def g (x : ℝ) : ℝ := Real.cos x - Real.sin x + 4 * Real.tan x

theorem smallest_zero_floor_is_three :
  ∃ (s : ℝ), s > 0 ∧ g s = 0 ∧ (∀ x, x > 0 ∧ g x = 0 → x ≥ s) ∧ ⌊s⌋ = 3 :=
sorry

end NUMINAMATH_CALUDE_smallest_zero_floor_is_three_l6_693


namespace NUMINAMATH_CALUDE_car_journey_time_l6_614

/-- Proves that given a car traveling 210 km in 7 hours for the forward journey,
    and increasing its speed by 12 km/hr for the return journey,
    the time taken for the return journey is 5 hours. -/
theorem car_journey_time (distance : ℝ) (forward_time : ℝ) (speed_increase : ℝ) :
  distance = 210 →
  forward_time = 7 →
  speed_increase = 12 →
  (distance / (distance / forward_time + speed_increase)) = 5 := by
  sorry

end NUMINAMATH_CALUDE_car_journey_time_l6_614


namespace NUMINAMATH_CALUDE_compound_ratio_example_l6_612

def ratio (a b : ℤ) := (a, b)

def compound_ratio (r1 r2 r3 : ℤ × ℤ) : ℤ × ℤ :=
  let (a1, b1) := r1
  let (a2, b2) := r2
  let (a3, b3) := r3
  (a1 * a2 * a3, b1 * b2 * b3)

def simplify_ratio (r : ℤ × ℤ) : ℤ × ℤ :=
  let (a, b) := r
  let gcd := Int.gcd a b
  (a / gcd, b / gcd)

theorem compound_ratio_example : 
  simplify_ratio (compound_ratio (ratio 2 3) (ratio 6 11) (ratio 11 2)) = (2, 1) := by
  sorry

end NUMINAMATH_CALUDE_compound_ratio_example_l6_612


namespace NUMINAMATH_CALUDE_equation_solution_l6_660

theorem equation_solution :
  ∀ x : ℚ, (Real.sqrt (4 * x + 9) / Real.sqrt (8 * x + 9) = Real.sqrt 3 / 2) → x = 9/8 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l6_660


namespace NUMINAMATH_CALUDE_circle_area_through_points_l6_630

/-- The area of a circle with center P(-5, 3) passing through Q(7, -2) is 169π -/
theorem circle_area_through_points :
  let P : ℝ × ℝ := (-5, 3)
  let Q : ℝ × ℝ := (7, -2)
  let r := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)
  π * r^2 = 169 * π := by
  sorry

end NUMINAMATH_CALUDE_circle_area_through_points_l6_630


namespace NUMINAMATH_CALUDE_algebraic_identities_l6_656

theorem algebraic_identities (a b x : ℝ) : 
  ((3 * a * b^3)^2 = 9 * a^2 * b^6) ∧ 
  (x * x^3 + x^2 * x^2 = 2 * x^4) ∧ 
  ((12 * x^4 - 6 * x^3) / (3 * x^2) = 4 * x^2 - 2 * x) := by
  sorry

end NUMINAMATH_CALUDE_algebraic_identities_l6_656


namespace NUMINAMATH_CALUDE_simple_interest_principal_l6_616

/-- Simple interest calculation -/
theorem simple_interest_principal 
  (rate : ℝ) (interest : ℝ) (time : ℝ) (principal : ℝ) :
  rate = 12.5 →
  interest = 100 →
  time = 2 →
  principal = (interest * 100) / (rate * time) →
  principal = 400 :=
by sorry

end NUMINAMATH_CALUDE_simple_interest_principal_l6_616


namespace NUMINAMATH_CALUDE_symmetric_point_and_line_l6_670

-- Define the line l₁
def l₁ (x y : ℝ) : Prop := 2 * x - 3 * y + 1 = 0

-- Define point A
def A : ℝ × ℝ := (-1, -2)

-- Define line m
def m (x y : ℝ) : Prop := 3 * x - 2 * y - 6 = 0

-- Define the symmetric point of a given point with respect to l₁
def symmetric_point (p : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define the symmetric line of a given line with respect to l₁
def symmetric_line (l : (ℝ → ℝ → Prop)) : (ℝ → ℝ → Prop) := sorry

theorem symmetric_point_and_line :
  (symmetric_point A = (-33/13, 4/13)) ∧
  (∀ x y, symmetric_line m x y ↔ 3 * x - 11 * y + 34 = 0) :=
sorry

end NUMINAMATH_CALUDE_symmetric_point_and_line_l6_670


namespace NUMINAMATH_CALUDE_art_gallery_pieces_l6_698

theorem art_gallery_pieces (total : ℕ) 
  (displayed : ℕ) (sculptures_displayed : ℕ) 
  (paintings_not_displayed : ℕ) (sculptures_not_displayed : ℕ) :
  displayed = total / 3 →
  sculptures_displayed = displayed / 6 →
  paintings_not_displayed = (total - displayed) / 3 →
  sculptures_not_displayed = 1400 →
  total = 3150 :=
by
  sorry

end NUMINAMATH_CALUDE_art_gallery_pieces_l6_698


namespace NUMINAMATH_CALUDE_a_3_equals_1_l6_649

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → a (n + 1) = a n - 3

theorem a_3_equals_1 (a : ℕ → ℤ) 
  (h1 : arithmetic_sequence a) 
  (h2 : a 1 = 7) : 
  a 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_a_3_equals_1_l6_649


namespace NUMINAMATH_CALUDE_expand_difference_of_squares_l6_692

theorem expand_difference_of_squares (a : ℝ) : (a + 1) * (a - 1) = a^2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_expand_difference_of_squares_l6_692


namespace NUMINAMATH_CALUDE_injective_implies_different_outputs_injective_implies_at_most_one_preimage_l6_609

-- Define the function f from set A to set B
variable {A B : Type*} (f : A → B)

-- Define injectivity
def Injective (f : A → B) : Prop :=
  ∀ x₁ x₂ : A, f x₁ = f x₂ → x₁ = x₂

-- Theorem 1: If f is injective and x₁ ≠ x₂, then f(x₁) ≠ f(x₂)
theorem injective_implies_different_outputs
  (hf : Injective f) :
  ∀ x₁ x₂ : A, x₁ ≠ x₂ → f x₁ ≠ f x₂ := by
sorry

-- Theorem 2: If f is injective, then for any b ∈ B, there is at most one pre-image in A
theorem injective_implies_at_most_one_preimage
  (hf : Injective f) :
  ∀ b : B, ∃! x : A, f x = b := by
sorry

end NUMINAMATH_CALUDE_injective_implies_different_outputs_injective_implies_at_most_one_preimage_l6_609


namespace NUMINAMATH_CALUDE_soccer_team_lineup_combinations_l6_603

def choose (n k : ℕ) : ℕ := Nat.choose n k

def total_players : ℕ := 18
def twins : ℕ := 2
def lineup_size : ℕ := 8
def defenders : ℕ := 5

theorem soccer_team_lineup_combinations : 
  (choose 2 1 * choose 5 3 * choose 11 4) +
  (choose 2 2 * choose 5 3 * choose 11 3) +
  (choose 2 1 * choose 5 4 * choose 11 3) +
  (choose 2 2 * choose 5 4 * choose 11 2) +
  (choose 2 1 * choose 5 5 * choose 11 2) +
  (choose 2 2 * choose 5 5 * choose 11 1) = 3602 := by
  sorry

end NUMINAMATH_CALUDE_soccer_team_lineup_combinations_l6_603


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l6_642

/-- Given an arithmetic sequence {a_n} where S_n is the sum of the first n terms,
    if (S_2016 / 2016) - (S_2015 / 2015) = 3, then a_2016 - a_2014 = 12. -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n : ℕ, S n = (n : ℝ) * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2) →
  (S 2016 / 2016 - S 2015 / 2015 = 3) →
  a 2016 - a 2014 = 12 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l6_642


namespace NUMINAMATH_CALUDE_pen_cost_l6_665

theorem pen_cost (pen pencil : ℚ) 
  (h1 : 3 * pen + 4 * pencil = 264/100)
  (h2 : 4 * pen + 2 * pencil = 230/100) : 
  pen = 392/1000 := by
sorry

end NUMINAMATH_CALUDE_pen_cost_l6_665


namespace NUMINAMATH_CALUDE_meal_serving_ways_correct_l6_605

/-- The number of ways to serve 10 meals (5 pasta and 5 salad) such that exactly 2 people receive the type of meal they ordered. -/
def mealServingWays : ℕ := 945

/-- The number of people. -/
def numPeople : ℕ := 10

/-- The number of people who ordered pasta. -/
def numPasta : ℕ := 5

/-- The number of people who ordered salad. -/
def numSalad : ℕ := 5

/-- The number of people who receive the correct meal. -/
def numCorrect : ℕ := 2

theorem meal_serving_ways_correct :
  mealServingWays = numPeople.choose numCorrect * 21 :=
sorry

end NUMINAMATH_CALUDE_meal_serving_ways_correct_l6_605


namespace NUMINAMATH_CALUDE_games_sale_value_l6_636

def initial_cost : ℝ := 200
def value_multiplier : ℝ := 3
def sold_percentage : ℝ := 0.4

theorem games_sale_value :
  let new_value := initial_cost * value_multiplier
  let sold_value := new_value * sold_percentage
  sold_value = 240 := by
  sorry

end NUMINAMATH_CALUDE_games_sale_value_l6_636


namespace NUMINAMATH_CALUDE_smallest_base_for_perfect_fourth_power_l6_696

theorem smallest_base_for_perfect_fourth_power : 
  (∃ (b : ℕ), b > 0 ∧ ∃ (x : ℕ), 7 * b^2 + 7 * b + 7 = x^4) ∧ 
  (∀ (b : ℕ), b > 0 → (∃ (x : ℕ), 7 * b^2 + 7 * b + 7 = x^4) → b ≥ 18) :=
by sorry

end NUMINAMATH_CALUDE_smallest_base_for_perfect_fourth_power_l6_696


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l6_627

theorem fraction_sum_equality (a b c : ℝ) 
  (h : a / (25 - a) + b / (65 - b) + c / (60 - c) = 7) :
  5 / (25 - a) + 13 / (65 - b) + 12 / (60 - c) = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l6_627


namespace NUMINAMATH_CALUDE_sphere_volume_is_10936_l6_683

/-- The volume of a small hemisphere container in liters -/
def small_hemisphere_volume : ℝ := 4

/-- The number of small hemisphere containers required -/
def num_hemispheres : ℕ := 2734

/-- The total volume of water in the sphere container in liters -/
def sphere_volume : ℝ := small_hemisphere_volume * num_hemispheres

/-- Theorem stating that the total volume of water in the sphere container is 10936 liters -/
theorem sphere_volume_is_10936 : sphere_volume = 10936 := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_is_10936_l6_683


namespace NUMINAMATH_CALUDE_prob_diff_tens_digits_l6_634

/-- The probability of selecting 6 different integers from 10 to 59 with different tens digits -/
theorem prob_diff_tens_digits : ℝ := by
  -- Define the range of integers
  let range : Set ℕ := {n : ℕ | 10 ≤ n ∧ n ≤ 59}

  -- Define the number of integers to be selected
  let k : ℕ := 6

  -- Define the function that returns the tens digit of a number
  let tens_digit (n : ℕ) : ℕ := n / 10

  -- Define the probability
  let prob : ℝ := (5 * 10 * 9 * 10^4 : ℝ) / (Nat.choose 50 6 : ℝ)

  -- State that the probability is equal to 1500000/5296900
  have h : prob = 1500000 / 5296900 := by sorry

  -- Return the probability
  exact prob

end NUMINAMATH_CALUDE_prob_diff_tens_digits_l6_634


namespace NUMINAMATH_CALUDE_pythagorean_theorem_isosceles_right_l6_624

/-- An isosceles right triangle with legs of unit length -/
structure IsoscelesRightTriangle where
  /-- The length of each leg is 1 -/
  leg : ℝ
  leg_eq_one : leg = 1

/-- The Pythagorean theorem for an isosceles right triangle -/
theorem pythagorean_theorem_isosceles_right (t : IsoscelesRightTriangle) :
  t.leg ^ 2 + t.leg ^ 2 = (Real.sqrt 2) ^ 2 := by
  sorry

#check pythagorean_theorem_isosceles_right

end NUMINAMATH_CALUDE_pythagorean_theorem_isosceles_right_l6_624


namespace NUMINAMATH_CALUDE_base_85_problem_l6_619

/-- Represents a number in base 85 --/
def BaseEightyFive : Type := List Nat

/-- Converts a number in base 85 to its decimal representation --/
def to_decimal (n : BaseEightyFive) : Nat :=
  sorry

/-- The specific number 3568432 in base 85 --/
def number : BaseEightyFive :=
  [3, 5, 6, 8, 4, 3, 2]

theorem base_85_problem (b : Int) 
  (h1 : 0 ≤ b) (h2 : b ≤ 19) 
  (h3 : (to_decimal number - b) % 17 = 0) : 
  b = 3 := by
  sorry

end NUMINAMATH_CALUDE_base_85_problem_l6_619


namespace NUMINAMATH_CALUDE_unique_positive_number_l6_611

theorem unique_positive_number : ∃! x : ℝ, x > 0 ∧ x^2 = (Real.sqrt 16)^3 := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_number_l6_611


namespace NUMINAMATH_CALUDE_tim_interest_rate_l6_621

/-- Tim's investment amount -/
def tim_investment : ℝ := 500

/-- Lana's investment amount -/
def lana_investment : ℝ := 1000

/-- Lana's annual interest rate -/
def lana_rate : ℝ := 0.05

/-- Number of years -/
def years : ℕ := 2

/-- Interest difference between Tim and Lana after 2 years -/
def interest_difference : ℝ := 2.5

/-- Calculate the compound interest -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time - principal

/-- Tim's annual interest rate -/
def tim_rate : ℝ := 0.1

theorem tim_interest_rate :
  compound_interest tim_investment tim_rate years =
  compound_interest lana_investment lana_rate years + interest_difference := by
  sorry

#check tim_interest_rate

end NUMINAMATH_CALUDE_tim_interest_rate_l6_621


namespace NUMINAMATH_CALUDE_diamond_equation_solution_l6_644

-- Define the diamond operation
def diamond (a b : ℚ) : ℚ := a * b + 3 * b - 2 * a

-- State the theorem
theorem diamond_equation_solution :
  ∀ y : ℚ, diamond 4 y = 50 → y = 58 / 7 := by
  sorry

end NUMINAMATH_CALUDE_diamond_equation_solution_l6_644


namespace NUMINAMATH_CALUDE_morning_evening_email_difference_l6_666

/-- The number of emails Jack received in the morning -/
def morning_emails : ℕ := 9

/-- The number of emails Jack received in the afternoon -/
def afternoon_emails : ℕ := 10

/-- The number of emails Jack received in the evening -/
def evening_emails : ℕ := 7

/-- Theorem stating the difference between morning and evening emails -/
theorem morning_evening_email_difference : 
  morning_emails - evening_emails = 2 := by sorry

end NUMINAMATH_CALUDE_morning_evening_email_difference_l6_666


namespace NUMINAMATH_CALUDE_vector_properties_l6_606

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

theorem vector_properties (a b c : E) :
  (a = b → ‖a‖ = ‖b‖) ∧
  (a = b → b = c → a = c) :=
by sorry

end NUMINAMATH_CALUDE_vector_properties_l6_606


namespace NUMINAMATH_CALUDE_f_g_product_positive_l6_659

-- Define the real-valued functions f and g
variable (f g : ℝ → ℝ)

-- Define the properties of f and g
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def is_even (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x
def monotone_decreasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ s → y ∈ s → x ≤ y → f y ≤ f x
def monotone_increasing_on (g : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ s → y ∈ s → x ≤ y → g x ≤ g y

-- State the theorem
theorem f_g_product_positive
  (h_f_odd : is_odd f)
  (h_f_decr : monotone_decreasing_on f {x | x < 0})
  (h_g_even : is_even g)
  (h_g_incr : monotone_increasing_on g {x | x ≤ 0})
  (h_f_1 : f 1 = 0)
  (h_g_1 : g 1 = 0) :
  {x : ℝ | f x * g x > 0} = {x : ℝ | 0 < x ∧ x < 1} ∪ {x : ℝ | x > 1} :=
sorry

end NUMINAMATH_CALUDE_f_g_product_positive_l6_659


namespace NUMINAMATH_CALUDE_P_n_formula_S_3_formula_geometric_sequence_condition_l6_613

-- Define the sequence and expansion operation
def Sequence := List ℝ

def expand_by_sum (s : Sequence) : Sequence :=
  match s with
  | [] => []
  | [x] => [x]
  | x::y::rest => x :: (x+y) :: expand_by_sum (y::rest)

-- Define P_n and S_n
def P (n : ℕ) (a b c : ℝ) : ℕ := 
  (expand_by_sum^[n] [a, b, c]).length

def S (n : ℕ) (a b c : ℝ) : ℝ := 
  (expand_by_sum^[n] [a, b, c]).sum

-- Theorem statements
theorem P_n_formula (n : ℕ) (a b c : ℝ) : 
  P n a b c = 2^(n+1) + 1 := by sorry

theorem S_3_formula (a b c : ℝ) :
  S 3 a b c = 14*a + 27*b + 14*c := by sorry

theorem geometric_sequence_condition (a b c : ℝ) :
  (∃ r : ℝ, ∀ n : ℕ, S (n+1) a b c = r * S n a b c) ↔ 
  ((a + c = 0 ∧ b ≠ 0) ∨ (2*b + a + c = 0 ∧ b ≠ 0)) := by sorry

end NUMINAMATH_CALUDE_P_n_formula_S_3_formula_geometric_sequence_condition_l6_613


namespace NUMINAMATH_CALUDE_coordinates_of_A_min_length_AB_l6_697

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define a point on the parabola
structure PointOnParabola where
  x : ℝ
  y : ℝ
  on_parabola : parabola x y

-- Define a line passing through the focus
structure LineThruFocus where
  slope : ℝ ⊕ PUnit  -- ℝ for finite slopes, PUnit for vertical line
  passes_thru_focus : True

-- Define the intersection points
def intersectionPoints (l : LineThruFocus) : PointOnParabola × PointOnParabola := sorry

-- Statement for part (1)
theorem coordinates_of_A (l : LineThruFocus) (A B : PointOnParabola) 
  (h : intersectionPoints l = (A, B)) (dist_AF : Real.sqrt ((A.x - 1)^2 + A.y^2) = 4) :
  (A.x = 3 ∧ A.y = 2 * Real.sqrt 3) ∨ (A.x = 3 ∧ A.y = -2 * Real.sqrt 3) := sorry

-- Statement for part (2)
theorem min_length_AB : 
  ∃ (min_length : ℝ), ∀ (l : LineThruFocus) (A B : PointOnParabola),
    intersectionPoints l = (A, B) → 
    Real.sqrt ((A.x - B.x)^2 + (A.y - B.y)^2) ≥ min_length ∧
    min_length = 4 := sorry

end NUMINAMATH_CALUDE_coordinates_of_A_min_length_AB_l6_697


namespace NUMINAMATH_CALUDE_coin_stack_solution_l6_620

/-- Represents the different types of coins --/
inductive CoinType
  | A
  | B
  | C
  | D

/-- Returns the thickness of a given coin type in millimeters --/
def coinThickness (t : CoinType) : ℚ :=
  match t with
  | CoinType.A => 21/10
  | CoinType.B => 18/10
  | CoinType.C => 12/10
  | CoinType.D => 2

/-- Represents a stack of coins --/
structure CoinStack :=
  (a b c d : ℕ)

/-- Calculates the height of a coin stack in millimeters --/
def stackHeight (s : CoinStack) : ℚ :=
  s.a * coinThickness CoinType.A +
  s.b * coinThickness CoinType.B +
  s.c * coinThickness CoinType.C +
  s.d * coinThickness CoinType.D

/-- The target height of the stack in millimeters --/
def targetHeight : ℚ := 18

theorem coin_stack_solution :
  ∃ (s : CoinStack), stackHeight s = targetHeight ∧
  s.a = 0 ∧ s.b = 0 ∧ s.c = 0 ∧ s.d = 9 :=
sorry

end NUMINAMATH_CALUDE_coin_stack_solution_l6_620


namespace NUMINAMATH_CALUDE_carl_weekly_earnings_l6_645

/-- Represents Carl's earnings and candy bar purchases over 4 weeks -/
structure CarlEarnings where
  weeks : ℕ
  candyBars : ℕ
  candyBarPrice : ℚ
  weeklyEarnings : ℚ

/-- Theorem stating that Carl's weekly earnings are $0.75 given the conditions -/
theorem carl_weekly_earnings (e : CarlEarnings) 
  (h_weeks : e.weeks = 4)
  (h_candyBars : e.candyBars = 6)
  (h_candyBarPrice : e.candyBarPrice = 1/2) :
  e.weeklyEarnings = 3/4 := by
sorry

end NUMINAMATH_CALUDE_carl_weekly_earnings_l6_645


namespace NUMINAMATH_CALUDE_student_count_last_year_l6_679

theorem student_count_last_year 
  (increase_rate : Real) 
  (current_count : Nat) 
  (h1 : increase_rate = 0.2) 
  (h2 : current_count = 960) : 
  ∃ (last_year_count : Nat), 
    (last_year_count : Real) * (1 + increase_rate) = current_count ∧ 
    last_year_count = 800 := by
  sorry

end NUMINAMATH_CALUDE_student_count_last_year_l6_679


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l6_617

/-- Given an ellipse where the length of the major axis is twice the length of the minor axis,
    prove that its eccentricity is √3/2. -/
theorem ellipse_eccentricity (a b : ℝ) (h : a = 2 * b) (h_pos : a > 0) :
  let c := Real.sqrt (a^2 - b^2)
  c / a = Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l6_617
