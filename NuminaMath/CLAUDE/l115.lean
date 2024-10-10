import Mathlib

namespace max_value_reciprocal_l115_11578

theorem max_value_reciprocal (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + y = 1 → 1 / (x + 2*y - 3*x*y) ≤ 3/2) ∧
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 1 ∧ 1 / (x + 2*y - 3*x*y) = 3/2) :=
by sorry

end max_value_reciprocal_l115_11578


namespace piggy_bank_pennies_l115_11515

theorem piggy_bank_pennies (compartments initial_per_compartment final_total : ℕ) 
  (h1 : compartments = 12)
  (h2 : initial_per_compartment = 2)
  (h3 : final_total = 96)
  : (final_total - compartments * initial_per_compartment) / compartments = 6 := by
  sorry

end piggy_bank_pennies_l115_11515


namespace range_of_f_l115_11597

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 2*x

-- Define the domain
def domain : Set ℝ := {x | -2 ≤ x ∧ x ≤ 1}

-- Theorem statement
theorem range_of_f :
  {y | ∃ x ∈ domain, f x = y} = {y | -1 ≤ y ∧ y ≤ 3} := by sorry

end range_of_f_l115_11597


namespace intersection_of_A_and_complement_of_B_l115_11522

open Set

def A : Set ℝ := {x | 1 < x ∧ x < 4}
def B : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

theorem intersection_of_A_and_complement_of_B :
  A ∩ (Bᶜ) = {x | 3 < x ∧ x < 4} := by sorry

end intersection_of_A_and_complement_of_B_l115_11522


namespace complex_1_2i_in_first_quadrant_l115_11581

/-- A complex number is in the first quadrant if its real part is positive and its imaginary part is positive -/
def in_first_quadrant (z : ℂ) : Prop := 0 < z.re ∧ 0 < z.im

/-- The theorem states that the complex number 1+2i is in the first quadrant -/
theorem complex_1_2i_in_first_quadrant : in_first_quadrant (1 + 2*I) := by
  sorry

end complex_1_2i_in_first_quadrant_l115_11581


namespace sufficient_not_necessary_condition_l115_11519

theorem sufficient_not_necessary_condition :
  (∀ x : ℝ, x ≥ 3 → x > 2) ∧
  ¬(∀ x : ℝ, x > 2 → x ≥ 3) :=
by sorry

end sufficient_not_necessary_condition_l115_11519


namespace ones_digit_of_complex_expression_l115_11591

-- Define a function to get the ones digit of a natural number
def ones_digit (n : ℕ) : ℕ := n % 10

-- Define the expression
def complex_expression : ℕ := 
  ones_digit ((73^567 % 10) * (47^123 % 10) + (86^784 % 10) - (32^259 % 10))

-- Theorem statement
theorem ones_digit_of_complex_expression :
  complex_expression = 9 := by sorry

end ones_digit_of_complex_expression_l115_11591


namespace marksmen_hit_probability_l115_11583

theorem marksmen_hit_probability (p1 p2 p3 : ℝ) 
  (h1 : p1 = 0.6) (h2 : p2 = 0.7) (h3 : p3 = 0.75) :
  1 - (1 - p1) * (1 - p2) * (1 - p3) = 0.97 := by
  sorry

end marksmen_hit_probability_l115_11583


namespace circle_circumference_area_equal_diameter_l115_11517

/-- When the circumference and area of a circle are numerically equal, the diameter is 4. -/
theorem circle_circumference_area_equal_diameter (r : ℝ) :
  2 * Real.pi * r = Real.pi * r^2 → 2 * r = 4 := by sorry

end circle_circumference_area_equal_diameter_l115_11517


namespace factorization_1_factorization_2_triangle_shape_l115_11545

/-- Factorization of 2a^2 - 8a + 8 --/
theorem factorization_1 (a : ℝ) : 2*a^2 - 8*a + 8 = 2*(a-2)^2 := by sorry

/-- Factorization of x^2 - y^2 + 3x - 3y --/
theorem factorization_2 (x y : ℝ) : x^2 - y^2 + 3*x - 3*y = (x-y)*(x+y+3) := by sorry

/-- Shape of triangle ABC given a^2 - ab - ac + bc = 0 --/
theorem triangle_shape (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) (eq : a^2 - a*b - a*c + b*c = 0) :
  (a = b ∨ a = c ∨ b = c) := by sorry

end factorization_1_factorization_2_triangle_shape_l115_11545


namespace factorial_prime_factorization_l115_11595

theorem factorial_prime_factorization (x a k m p : ℕ) : 
  x = Nat.factorial 8 →
  x = 2^a * 3^k * 5^m * 7^p →
  a > 0 ∧ k > 0 ∧ m > 0 ∧ p > 0 →
  a + k + m + p = 11 →
  a = 7 := by
sorry

end factorial_prime_factorization_l115_11595


namespace smallest_divisible_by_one_to_ten_l115_11533

/-- The smallest positive integer divisible by all integers from 1 to 10 -/
def smallestDivisibleByOneToTen : ℕ := 2520

/-- Proposition: smallestDivisibleByOneToTen is the smallest positive integer 
    divisible by all integers from 1 to 10 -/
theorem smallest_divisible_by_one_to_ten :
  ∀ n : ℕ, n > 0 → (∀ k : ℕ, 1 ≤ k ∧ k ≤ 10 → k ∣ n) → smallestDivisibleByOneToTen ≤ n :=
by sorry

end smallest_divisible_by_one_to_ten_l115_11533


namespace buckingham_palace_visitors_l115_11598

theorem buckingham_palace_visitors (previous_day_visitors : ℕ) (additional_visitors : ℕ) 
  (h1 : previous_day_visitors = 295) 
  (h2 : additional_visitors = 22) : 
  previous_day_visitors + additional_visitors = 317 := by
  sorry

end buckingham_palace_visitors_l115_11598


namespace num_algebraic_expressions_is_five_l115_11543

/-- An expression is algebraic if it consists of numbers, variables, and arithmetic operations, without equality or inequality symbols. -/
def is_algebraic_expression (e : String) : Bool :=
  match e with
  | "2x^2" => true
  | "1-2x=0" => false
  | "ab" => true
  | "a>0" => false
  | "0" => true
  | "1/a" => true
  | "π" => true
  | _ => false

/-- The list of expressions to be checked -/
def expressions : List String :=
  ["2x^2", "1-2x=0", "ab", "a>0", "0", "1/a", "π"]

/-- The number of algebraic expressions in the list -/
def num_algebraic_expressions : Nat :=
  (expressions.filter is_algebraic_expression).length

theorem num_algebraic_expressions_is_five :
  num_algebraic_expressions = 5 := by
  sorry

end num_algebraic_expressions_is_five_l115_11543


namespace james_total_score_l115_11550

theorem james_total_score (field_goals : ℕ) (two_point_shots : ℕ) : field_goals = 13 → two_point_shots = 20 → field_goals * 3 + two_point_shots * 2 = 79 := by
  sorry

end james_total_score_l115_11550


namespace max_perfect_squares_pairwise_products_l115_11534

/-- Given two distinct natural numbers, the maximum number of perfect squares
    among the pairwise products of these numbers and their +2 counterparts is 2. -/
theorem max_perfect_squares_pairwise_products (a b : ℕ) (h : a ≠ b) :
  let products := {a * (a + 2), a * b, a * (b + 2), (a + 2) * b, (a + 2) * (b + 2), b * (b + 2)}
  (∃ (s : Finset ℕ), s ⊆ products ∧ (∀ x ∈ s, ∃ y, x = y^2) ∧ s.card = 2) ∧
  (∀ (s : Finset ℕ), s ⊆ products → (∀ x ∈ s, ∃ y, x = y^2) → s.card ≤ 2) :=
sorry

end max_perfect_squares_pairwise_products_l115_11534


namespace tournament_games_theorem_l115_11569

/-- Represents a single-elimination tournament -/
structure Tournament :=
  (num_teams : ℕ)
  (num_players_per_team : ℕ)

/-- Calculates the number of games needed to determine the champion -/
def games_to_champion (t : Tournament) : ℕ :=
  t.num_teams - 1

/-- The theorem stating that a tournament with 128 teams requires 127 games to determine the champion -/
theorem tournament_games_theorem :
  ∀ (t : Tournament), t.num_teams = 128 → t.num_players_per_team = 4 → games_to_champion t = 127 := by
  sorry

end tournament_games_theorem_l115_11569


namespace count_satisfying_numbers_l115_11523

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

def sum_of_digits (n : ℕ) : ℕ :=
  (n % 10) + (n / 10)

def satisfies_conditions (n : ℕ) : Prop :=
  is_two_digit n ∧
  n + reverse_digits n = 110 ∧
  sum_of_digits n % 3 = 0

theorem count_satisfying_numbers :
  ∃! (s : Finset ℕ), (∀ n ∈ s, satisfies_conditions n) ∧ s.card = 3 :=
sorry

end count_satisfying_numbers_l115_11523


namespace sqrt_equation_solution_l115_11585

theorem sqrt_equation_solution (a b : ℕ+) (h : a < b) :
  Real.sqrt (1 + Real.sqrt (25 + 14 * Real.sqrt 3)) = Real.sqrt a + Real.sqrt b ↔ 
  a = 1 ∧ b = 3 := by
sorry

end sqrt_equation_solution_l115_11585


namespace roots_exist_in_intervals_l115_11547

-- Define the polynomial function
def f (x : ℝ) : ℝ := x^4 - (2*10^10 + 1)*x^2 - x + 10^20 + 10^10 - 1

-- State the theorem
theorem roots_exist_in_intervals : 
  ∃ (x₁ x₂ : ℝ), 
    f x₁ = 0 ∧ f x₂ = 0 ∧ 
    99999.9996 ≤ x₁ ∧ x₁ ≤ 99999.9998 ∧
    100000.0002 ≤ x₂ ∧ x₂ ≤ 100000.0004 :=
sorry

end roots_exist_in_intervals_l115_11547


namespace michael_truck_meet_once_l115_11576

-- Define the constants
def michael_speed : ℝ := 6
def truck_speed : ℝ := 12
def bench_distance : ℝ := 180
def truck_stop_time : ℝ := 40

-- Define the positions of Michael and the truck as functions of time
def michael_position (t : ℝ) : ℝ := michael_speed * t

-- The truck's position is more complex due to stops, so we'll define it as a noncomputable function
noncomputable def truck_position (t : ℝ) : ℝ := 
  let cycle_time := bench_distance / truck_speed + truck_stop_time
  let full_cycles := ⌊t / cycle_time⌋
  let remaining_time := t - full_cycles * cycle_time
  bench_distance * (full_cycles + 1) + 
    if remaining_time ≤ bench_distance / truck_speed 
    then truck_speed * remaining_time
    else bench_distance

-- Define the theorem
theorem michael_truck_meet_once :
  ∃! t : ℝ, t > 0 ∧ michael_position t = truck_position t :=
sorry


end michael_truck_meet_once_l115_11576


namespace circus_dog_paws_l115_11504

theorem circus_dog_paws (total_dogs : ℕ) (back_leg_fraction : ℚ) : total_dogs = 24 → back_leg_fraction = 2/3 → (total_dogs : ℚ) * back_leg_fraction * 2 + (total_dogs : ℚ) * (1 - back_leg_fraction) * 4 = 64 := by
  sorry

end circus_dog_paws_l115_11504


namespace water_left_over_l115_11565

/-- Calculates the amount of water left over after distributing to players and accounting for spillage -/
theorem water_left_over
  (total_players : ℕ)
  (initial_water_liters : ℕ)
  (water_per_player_ml : ℕ)
  (spilled_water_ml : ℕ)
  (h1 : total_players = 30)
  (h2 : initial_water_liters = 8)
  (h3 : water_per_player_ml = 200)
  (h4 : spilled_water_ml = 250) :
  initial_water_liters * 1000 - (total_players * water_per_player_ml + spilled_water_ml) = 1750 :=
by sorry

end water_left_over_l115_11565


namespace twice_one_fifth_of_ten_times_fifteen_l115_11593

theorem twice_one_fifth_of_ten_times_fifteen : 2 * ((1 / 5 : ℚ) * (10 * 15)) = 60 := by
  sorry

end twice_one_fifth_of_ten_times_fifteen_l115_11593


namespace birdhouse_flew_1200_feet_l115_11513

/-- The distance the car was transported, in feet -/
def car_distance : ℕ := 200

/-- The distance the lawn chair was blown, in feet -/
def lawn_chair_distance : ℕ := 2 * car_distance

/-- The distance the birdhouse flew, in feet -/
def birdhouse_distance : ℕ := 3 * lawn_chair_distance

/-- Theorem stating that the birdhouse flew 1200 feet -/
theorem birdhouse_flew_1200_feet : birdhouse_distance = 1200 := by
  sorry

end birdhouse_flew_1200_feet_l115_11513


namespace average_of_w_x_z_l115_11531

theorem average_of_w_x_z (w x y z a : ℝ) 
  (h1 : 2/w + 2/x + 2/z = 2/y)
  (h2 : w*x*z = y)
  (h3 : w + x + z = a) :
  (w + x + z) / 3 = a / 3 := by
  sorry

end average_of_w_x_z_l115_11531


namespace percentage_difference_l115_11592

theorem percentage_difference (original : ℝ) (result : ℝ) (h : result < original) :
  (original - result) / original * 100 = 50 :=
by
  -- Assuming original = 60 and result = 30
  have h1 : original = 60 := by sorry
  have h2 : result = 30 := by sorry
  
  -- The proof goes here
  sorry

end percentage_difference_l115_11592


namespace tree_cutting_theorem_l115_11552

/-- The number of trees James cuts per day -/
def james_trees_per_day : ℕ := 20

/-- The number of days James works alone -/
def solo_days : ℕ := 2

/-- The number of days James works with his brothers -/
def team_days : ℕ := 3

/-- The number of brothers helping James -/
def num_brothers : ℕ := 2

/-- The percentage of trees each brother cuts compared to James -/
def brother_efficiency : ℚ := 4/5

/-- The total number of trees cut down -/
def total_trees : ℕ := 196

theorem tree_cutting_theorem :
  james_trees_per_day * solo_days + 
  (james_trees_per_day + (james_trees_per_day * brother_efficiency).floor * num_brothers) * team_days = 
  total_trees :=
sorry

end tree_cutting_theorem_l115_11552


namespace elephant_to_big_cat_ratio_l115_11542

/-- Represents the population of animals in a park -/
structure ParkPopulation where
  lions : ℕ
  leopards : ℕ
  elephants : ℕ

/-- The ratio of two natural numbers -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

/-- Theorem about the ratio of elephants to lions and leopards in a park -/
theorem elephant_to_big_cat_ratio 
  (park : ParkPopulation) 
  (h1 : park.lions = 2 * park.leopards) 
  (h2 : park.lions = 200) 
  (h3 : park.lions + park.leopards + park.elephants = 450) : 
  Ratio.mk park.elephants (park.lions + park.leopards) = Ratio.mk 1 2 := by
  sorry

end elephant_to_big_cat_ratio_l115_11542


namespace certain_number_problem_l115_11509

theorem certain_number_problem (h : 2994 / 14.5 = 179) : 
  ∃ x : ℝ, x / 1.45 = 17.9 ∧ x = 25.955 := by sorry

end certain_number_problem_l115_11509


namespace certain_number_is_eight_l115_11584

theorem certain_number_is_eight (x n : ℚ) : x = 6 ∧ 9 - n / x = 7 + 8 / x → n = 8 := by
  sorry

end certain_number_is_eight_l115_11584


namespace intersection_of_A_and_B_l115_11518

-- Define set A
def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 2}

-- Define set B
def B : Set ℝ := {-1, 0, 1, 2}

-- Theorem statement
theorem intersection_of_A_and_B :
  A ∩ B = {-1, 0, 1} := by sorry

end intersection_of_A_and_B_l115_11518


namespace positive_intervals_l115_11512

def f (x : ℝ) := (x + 2) * (x - 2) * (x + 1)

theorem positive_intervals (x : ℝ) : 
  f x > 0 ↔ (x > -2 ∧ x < -1) ∨ x > 2 :=
sorry

end positive_intervals_l115_11512


namespace cylinder_cut_area_l115_11559

/-- The area of the newly exposed circular segment face when cutting a cylinder -/
theorem cylinder_cut_area (r h : ℝ) (h_r : r = 8) (h_h : h = 10) :
  let base_area := π * r^2
  let sector_area := (1/4) * base_area
  sector_area = 16 * π := by sorry

end cylinder_cut_area_l115_11559


namespace smallest_two_digit_with_digit_product_12_l115_11524

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_product (n : ℕ) : ℕ := (n / 10) * (n % 10)

theorem smallest_two_digit_with_digit_product_12 :
  ∀ n : ℕ, is_two_digit n → digit_product n = 12 → 26 ≤ n :=
sorry

end smallest_two_digit_with_digit_product_12_l115_11524


namespace repeating_decimal_equals_fraction_l115_11586

/-- The repeating decimal 0.4444... expressed as a real number -/
def repeating_decimal : ℚ := 0.4444444444

/-- The theorem states that the repeating decimal 0.4444... is equal to 4/9 -/
theorem repeating_decimal_equals_fraction : repeating_decimal = 4/9 := by
  sorry

end repeating_decimal_equals_fraction_l115_11586


namespace josh_marbles_remaining_l115_11548

def initial_marbles : ℕ := 19
def lost_marbles : ℕ := 11

theorem josh_marbles_remaining : initial_marbles - lost_marbles = 8 := by
  sorry

end josh_marbles_remaining_l115_11548


namespace fraction_zero_implies_x_equals_two_l115_11589

theorem fraction_zero_implies_x_equals_two (x : ℝ) : 
  (x^2 - 4)/(x + 2) = 0 → x = 2 :=
by sorry

end fraction_zero_implies_x_equals_two_l115_11589


namespace test_probabilities_l115_11541

/-- Probability of an event occurring -/
def Prob (event : Prop) : ℝ := sorry

/-- The probability that individual A passes the test -/
def probA : ℝ := 0.8

/-- The probability that individual B passes the test -/
def probB : ℝ := 0.6

/-- The probability that individual C passes the test -/
def probC : ℝ := 0.5

/-- A passes the test -/
def A : Prop := sorry

/-- B passes the test -/
def B : Prop := sorry

/-- C passes the test -/
def C : Prop := sorry

theorem test_probabilities :
  (Prob A = probA) ∧
  (Prob B = probB) ∧
  (Prob C = probC) ∧
  (Prob (A ∧ B ∧ C) = 0.24) ∧
  (Prob (A ∨ B ∨ C) = 0.96) := by sorry

end test_probabilities_l115_11541


namespace student_transportation_l115_11530

theorem student_transportation (total : ℚ) 
  (bus car scooter skateboard : ℚ) 
  (h1 : total = 1)
  (h2 : bus = 1/3)
  (h3 : car = 1/5)
  (h4 : scooter = 1/6)
  (h5 : skateboard = 1/8) :
  total - (bus + car + scooter + skateboard) = 7/40 := by
  sorry

end student_transportation_l115_11530


namespace plane_equation_proof_l115_11573

/-- A plane in 3D space represented by its equation coefficients -/
structure Plane where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ

/-- A point in 3D space -/
structure Point where
  x : ℤ
  y : ℤ
  z : ℤ

/-- Check if a point lies on a plane -/
def pointOnPlane (plane : Plane) (point : Point) : Prop :=
  plane.a * point.x + plane.b * point.y + plane.c * point.z + plane.d = 0

/-- Check if two planes are parallel -/
def planesParallel (plane1 : Plane) (plane2 : Plane) : Prop :=
  ∃ (k : ℚ), k ≠ 0 ∧ plane1.a = k * plane2.a ∧ plane1.b = k * plane2.b ∧ plane1.c = k * plane2.c

/-- The greatest common divisor of four integers is 1 -/
def gcdOne (a b c d : ℤ) : Prop :=
  Nat.gcd (Nat.gcd (Nat.gcd a.natAbs b.natAbs) c.natAbs) d.natAbs = 1

theorem plane_equation_proof (givenPlane : Plane) (point : Point) :
  givenPlane.a = 3 ∧ givenPlane.b = -2 ∧ givenPlane.c = 4 ∧ givenPlane.d = 5 →
  point.x = 2 ∧ point.y = -3 ∧ point.z = 1 →
  ∃ (soughtPlane : Plane),
    soughtPlane.a = 3 ∧
    soughtPlane.b = -2 ∧
    soughtPlane.c = 4 ∧
    soughtPlane.d = -16 ∧
    soughtPlane.a > 0 ∧
    pointOnPlane soughtPlane point ∧
    planesParallel soughtPlane givenPlane ∧
    gcdOne soughtPlane.a soughtPlane.b soughtPlane.c soughtPlane.d :=
by sorry

end plane_equation_proof_l115_11573


namespace least_positive_tangent_inverse_l115_11521

theorem least_positive_tangent_inverse (y p q : ℝ) (h1 : Real.tan y = p / q) (h2 : Real.tan (3 * y) = q / (p + q)) :
  ∃ m : ℝ, m > 0 ∧ y = Real.arctan m ∧ ∀ m' : ℝ, m' > 0 → y = Real.arctan m' → m ≤ m' ∧ m = 1 :=
sorry

end least_positive_tangent_inverse_l115_11521


namespace right_triangle_area_l115_11574

theorem right_triangle_area (hypotenuse : ℝ) (angle : ℝ) :
  hypotenuse = 8 * Real.sqrt 2 →
  angle = 45 * π / 180 →
  let area := (hypotenuse^2 / 4 : ℝ)
  area = 32 := by sorry

end right_triangle_area_l115_11574


namespace intersection_equals_M_l115_11596

def M : Set ℝ := {y | ∃ x, y = 3^x}
def N : Set ℝ := {y | ∃ x, y = x^2 - 1}

theorem intersection_equals_M : M ∩ N = M := by sorry

end intersection_equals_M_l115_11596


namespace product_of_repeating_decimals_l115_11562

/-- Represents a repeating decimal with a single repeating digit -/
def repeatingDecimal (wholePart : ℚ) (repeatingDigit : ℕ) : ℚ :=
  wholePart + (repeatingDigit : ℚ) / 99

theorem product_of_repeating_decimals :
  (repeatingDecimal 0 3) * (repeatingDecimal 0 81) = 9 / 363 := by
  sorry

end product_of_repeating_decimals_l115_11562


namespace discount_equation_l115_11560

/-- Represents the discount rate as a real number between 0 and 1 -/
def discount_rate : ℝ := sorry

/-- The original price in yuan -/
def original_price : ℝ := 200

/-- The final selling price in yuan -/
def final_price : ℝ := 148

/-- Theorem stating the relationship between original price, discount rate, and final price -/
theorem discount_equation : 
  original_price * (1 - discount_rate)^2 = final_price := by sorry

end discount_equation_l115_11560


namespace sum_of_fractions_equals_seven_l115_11551

theorem sum_of_fractions_equals_seven : 
  let S := 1 / (4 - Real.sqrt 15) - 1 / (Real.sqrt 15 - Real.sqrt 14) + 
           1 / (Real.sqrt 14 - Real.sqrt 13) - 1 / (Real.sqrt 13 - Real.sqrt 12) + 
           1 / (Real.sqrt 12 - 3)
  S = 7 := by
  sorry

end sum_of_fractions_equals_seven_l115_11551


namespace binary_11_equals_3_l115_11516

/-- Converts a binary number represented as a list of bits (least significant bit first) to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : Nat :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 3 -/
def binary_three : List Bool := [true, true]

/-- Theorem stating that the binary number 11 (base 2) is equal to 3 (base 10) -/
theorem binary_11_equals_3 : binary_to_decimal binary_three = 3 := by
  sorry

end binary_11_equals_3_l115_11516


namespace pants_cost_l115_11564

def initial_amount : ℕ := 109
def shirt_cost : ℕ := 11
def num_shirts : ℕ := 2
def remaining_amount : ℕ := 74

theorem pants_cost : 
  initial_amount - (shirt_cost * num_shirts) - remaining_amount = 13 := by
  sorry

end pants_cost_l115_11564


namespace line_perpendicular_plane_parallel_l115_11580

structure Space where
  Line : Type
  Plane : Type
  perpendicular : Line → Plane → Prop
  parallel : Line → Line → Prop

variable (S : Space)

theorem line_perpendicular_plane_parallel
  (l m : S.Line) (α : S.Plane)
  (h1 : l ≠ m)
  (h2 : S.perpendicular l α)
  (h3 : S.parallel l m) :
  S.perpendicular m α :=
sorry

end line_perpendicular_plane_parallel_l115_11580


namespace apple_sharing_l115_11558

theorem apple_sharing (total_apples : ℕ) (num_friends : ℕ) (apples_per_friend : ℕ) :
  total_apples = 9 →
  num_friends = 3 →
  total_apples = num_friends * apples_per_friend →
  apples_per_friend = 3 := by
  sorry

end apple_sharing_l115_11558


namespace base_conversion_subtraction_l115_11528

/-- Converts a number from base b to base 10 -/
def to_base_10 (digits : List Nat) (b : Nat) : Nat :=
  digits.reverse.enum.foldr (fun (i, d) acc => acc + d * b^i) 0

/-- The problem statement -/
theorem base_conversion_subtraction :
  let base_7_num := to_base_10 [0, 3, 4, 2, 5] 7
  let base_8_num := to_base_10 [0, 2, 3, 4] 8
  base_7_num - base_8_num = 10652 := by
  sorry

end base_conversion_subtraction_l115_11528


namespace davids_math_marks_l115_11537

def englishMarks : ℝ := 70
def physicsMarks : ℝ := 78
def chemistryMarks : ℝ := 60
def biologyMarks : ℝ := 65
def averageMarks : ℝ := 66.6
def totalSubjects : ℕ := 5

theorem davids_math_marks :
  let totalMarks := averageMarks * totalSubjects
  let knownSubjectsMarks := englishMarks + physicsMarks + chemistryMarks + biologyMarks
  let mathMarks := totalMarks - knownSubjectsMarks
  mathMarks = 60 := by sorry

end davids_math_marks_l115_11537


namespace no_valid_class_composition_l115_11582

theorem no_valid_class_composition : ¬ ∃ (n b g : ℕ+), 
  32 < n ∧ n < 40 ∧ 
  n = b + g ∧
  3 * b = 5 * g :=
by sorry

end no_valid_class_composition_l115_11582


namespace pure_imaginary_complex_number_l115_11587

theorem pure_imaginary_complex_number (m : ℝ) :
  let z : ℂ := (m^2 - 1) + (m + 1) * Complex.I
  (z.re = 0 ∧ z ≠ 0) → m = 1 := by sorry

end pure_imaginary_complex_number_l115_11587


namespace final_price_theorem_l115_11505

def mothers_day_discount : ℝ := 0.10
def additional_children_discount : ℝ := 0.04
def vip_discount : ℝ := 0.05
def shoes_cost : ℝ := 125
def handbag_cost : ℝ := 75
def min_purchase : ℝ := 150

def total_cost : ℝ := shoes_cost + handbag_cost

def discounted_price (price : ℝ) : ℝ :=
  let price_after_mothers_day := price * (1 - mothers_day_discount)
  let price_after_children := price_after_mothers_day * (1 - additional_children_discount)
  price_after_children * (1 - vip_discount)

theorem final_price_theorem :
  total_cost ≥ min_purchase →
  discounted_price total_cost = 164.16 :=
by sorry

end final_price_theorem_l115_11505


namespace smallest_measurement_count_l115_11507

theorem smallest_measurement_count : ∃ N : ℕ+, 
  (∀ m : ℕ+, m < N → 
    (¬(20 * m.val % 100 = 0) ∨ 
     ¬(375 * m.val % 1000 = 0) ∨ 
     ¬(25 * m.val % 100 = 0) ∨ 
     ¬(125 * m.val % 1000 = 0) ∨ 
     ¬(5 * m.val % 100 = 0))) ∧
  (20 * N.val % 100 = 0) ∧ 
  (375 * N.val % 1000 = 0) ∧ 
  (25 * N.val % 100 = 0) ∧ 
  (125 * N.val % 1000 = 0) ∧ 
  (5 * N.val % 100 = 0) ∧
  N.val = 40 := by
sorry

end smallest_measurement_count_l115_11507


namespace mixture_replacement_l115_11502

/-- Given a mixture of liquids A and B with an initial ratio of 4:1 and a final ratio of 2:3 after
    replacing some mixture with pure B, prove that 60 liters of mixture were replaced when the
    initial amount of liquid A was 48 liters. -/
theorem mixture_replacement (initial_A : ℝ) (initial_B : ℝ) (replaced : ℝ) :
  initial_A = 48 →
  initial_A / initial_B = 4 / 1 →
  initial_A / (initial_B + replaced) = 2 / 3 →
  replaced = 60 :=
by sorry

end mixture_replacement_l115_11502


namespace four_is_square_root_of_sixteen_l115_11568

-- Definition of square root
def is_square_root (x y : ℝ) : Prop := y * y = x

-- Theorem to prove
theorem four_is_square_root_of_sixteen : is_square_root 16 4 := by
  sorry

end four_is_square_root_of_sixteen_l115_11568


namespace georges_socks_l115_11577

theorem georges_socks (initial_socks bought_socks dad_socks : ℝ) 
  (h1 : initial_socks = 28.0)
  (h2 : bought_socks = 36.0)
  (h3 : dad_socks = 4.0) :
  initial_socks + bought_socks + dad_socks = 68.0 :=
by sorry

end georges_socks_l115_11577


namespace circle_passes_through_points_l115_11539

/-- Defines a circle equation passing through three points -/
def CircleEquation (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 6*y = 0

/-- Theorem stating that the circle equation passes through the given points -/
theorem circle_passes_through_points :
  CircleEquation 0 0 ∧ CircleEquation 4 0 ∧ CircleEquation (-1) 1 := by
  sorry

end circle_passes_through_points_l115_11539


namespace first_patient_sessions_l115_11520

/-- Given a group of patients and their session requirements, prove the number of sessions for the first patient. -/
theorem first_patient_sessions
  (total_patients : ℕ)
  (total_sessions : ℕ)
  (patient2_sessions : ℕ → ℕ)
  (remaining_patients_sessions : ℕ)
  (h1 : total_patients = 4)
  (h2 : total_sessions = 25)
  (h3 : patient2_sessions x = x + 5)
  (h4 : remaining_patients_sessions = 8 + 8)
  (h5 : x + patient2_sessions x + remaining_patients_sessions = total_sessions) :
  x = 2 :=
by sorry

end first_patient_sessions_l115_11520


namespace sine_graph_shift_l115_11594

theorem sine_graph_shift (x : ℝ) :
  3 * Real.sin (2 * x - π / 6) = 3 * Real.sin (2 * (x - π / 12)) :=
by sorry

#check sine_graph_shift

end sine_graph_shift_l115_11594


namespace harkamal_payment_l115_11500

def grapes_qty : ℝ := 8
def grapes_price : ℝ := 80
def mangoes_qty : ℝ := 9
def mangoes_price : ℝ := 55
def apples_qty : ℝ := 6
def apples_price : ℝ := 120
def oranges_qty : ℝ := 4
def oranges_price : ℝ := 75
def apple_discount : ℝ := 0.1
def sales_tax : ℝ := 0.05

def total_cost : ℝ :=
  grapes_qty * grapes_price +
  mangoes_qty * mangoes_price +
  apples_qty * apples_price * (1 - apple_discount) +
  oranges_qty * oranges_price

def final_cost : ℝ := total_cost * (1 + sales_tax)

theorem harkamal_payment : final_cost = 2187.15 := by
  sorry

end harkamal_payment_l115_11500


namespace a_minus_c_equals_three_l115_11572

theorem a_minus_c_equals_three
  (e f a b c d : ℝ)
  (h1 : e = a^2 + b^2)
  (h2 : f = c^2 + d^2)
  (h3 : a - b = c + d + 9)
  (h4 : a + b = c - d - 3)
  (h5 : f - e = 5*a + 2*b + 3*c + 4*d) :
  a - c = 3 := by
sorry

end a_minus_c_equals_three_l115_11572


namespace equal_angles_45_degrees_l115_11563

theorem equal_angles_45_degrees (α₁ α₂ α₃ : Real) : 
  0 < α₁ ∧ α₁ < π / 2 →
  0 < α₂ ∧ α₂ < π / 2 →
  0 < α₃ ∧ α₃ < π / 2 →
  Real.sin α₁ = Real.cos α₂ →
  Real.sin α₂ = Real.cos α₃ →
  Real.sin α₃ = Real.cos α₁ →
  α₁ = π / 4 ∧ α₂ = π / 4 ∧ α₃ = π / 4 := by
  sorry

end equal_angles_45_degrees_l115_11563


namespace vector_projection_l115_11575

/-- Given two vectors a and e in a real inner product space, 
    where |a| = 4, e is a unit vector, and the angle between a and e is 2π/3,
    prove that the projection of a + e on a - e is 5√21 / 7 -/
theorem vector_projection (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] 
  (a e : V) (h1 : ‖a‖ = 4) (h2 : ‖e‖ = 1) 
  (h3 : Real.cos (Real.arccos (inner a e / (‖a‖ * ‖e‖))) = Real.cos (2 * Real.pi / 3)) :
  ‖a + e‖ * (inner (a + e) (a - e) / (‖a + e‖ * ‖a - e‖)) = 5 * Real.sqrt 21 / 7 := by
  sorry

end vector_projection_l115_11575


namespace cos_sum_thirteen_l115_11526

theorem cos_sum_thirteen : 
  Real.cos (2 * Real.pi / 13) + Real.cos (6 * Real.pi / 13) + Real.cos (8 * Real.pi / 13) = (Real.sqrt 13 - 1) / 4 := by
  sorry

end cos_sum_thirteen_l115_11526


namespace purely_imaginary_fraction_l115_11556

theorem purely_imaginary_fraction (a : ℝ) : 
  (∃ b : ℝ, (Complex.I * b : ℂ) = (2 * Complex.I - 1) / (1 + a * Complex.I)) → a = 1/2 := by
  sorry

end purely_imaginary_fraction_l115_11556


namespace third_quadrant_condition_l115_11503

-- Define the complex number z as a function of m
def z (m : ℝ) : ℂ := Complex.mk (m + 3) (m - 1)

-- Define what it means for a complex number to be in the third quadrant
def in_third_quadrant (w : ℂ) : Prop := w.re < 0 ∧ w.im < 0

-- The theorem statement
theorem third_quadrant_condition (m : ℝ) :
  in_third_quadrant (z m) ↔ m < -3 := by
  sorry

end third_quadrant_condition_l115_11503


namespace parabola_shift_l115_11510

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := -2 * x^2

-- Define the shifted parabola
def shifted_parabola (x : ℝ) : ℝ := -2 * (x - 1)^2 + 3

-- Theorem stating that the shifted parabola is the result of the described transformations
theorem parabola_shift :
  ∀ x : ℝ, shifted_parabola x = original_parabola (x - 1) + 3 :=
by sorry

end parabola_shift_l115_11510


namespace zoo_count_difference_l115_11529

/-- Proves that the difference between the number of monkeys and giraffes is 22 -/
theorem zoo_count_difference : 
  let zebras : ℕ := 12
  let camels : ℕ := zebras / 2
  let monkeys : ℕ := 4 * camels
  let giraffes : ℕ := 2
  monkeys - giraffes = 22 := by sorry

end zoo_count_difference_l115_11529


namespace tangent_intersection_y_coord_l115_11561

/-- Given two points on the parabola y = x^2 + 1 with perpendicular tangents,
    the y-coordinate of their intersection is 3/4 -/
theorem tangent_intersection_y_coord (a b : ℝ) : 
  (2 * a) * (2 * b) = -1 →  -- Perpendicular tangents condition
  (∃ (x : ℝ), (2 * a) * (x - a) + a^2 + 1 = (2 * b) * (x - b) + b^2 + 1) →  -- Intersection exists
  (2 * a) * ((a + b) / 2 - a) + a^2 + 1 = 3 / 4 :=
by sorry

end tangent_intersection_y_coord_l115_11561


namespace typing_service_problem_l115_11538

/-- Typing service problem -/
theorem typing_service_problem
  (total_pages : ℕ)
  (first_time_cost : ℕ)
  (revision_cost : ℕ)
  (pages_revised_once : ℕ)
  (total_cost : ℕ)
  (h1 : total_pages = 100)
  (h2 : first_time_cost = 5)
  (h3 : revision_cost = 3)
  (h4 : pages_revised_once = 30)
  (h5 : total_cost = 710)
  : ∃ (pages_revised_twice : ℕ),
    pages_revised_twice = 20 ∧
    total_cost = total_pages * first_time_cost +
                 pages_revised_once * revision_cost +
                 pages_revised_twice * revision_cost * 2 :=
by sorry

end typing_service_problem_l115_11538


namespace average_students_count_l115_11506

theorem average_students_count (total : ℕ) (top_yes : ℕ) (avg_yes : ℕ) (under_yes : ℕ) :
  total = 30 →
  top_yes = 19 →
  avg_yes = 12 →
  under_yes = 9 →
  ∃ (top avg under : ℕ),
    top + avg + under = total ∧
    top = top_yes ∧
    avg = avg_yes ∧
    under = under_yes :=
by
  sorry

end average_students_count_l115_11506


namespace abs_neg_five_l115_11579

theorem abs_neg_five : |(-5 : ℤ)| = 5 := by
  sorry

end abs_neg_five_l115_11579


namespace bridge_length_l115_11546

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length (train_length : Real) (train_speed_kmh : Real) (crossing_time_s : Real) :
  train_length = 100 →
  train_speed_kmh = 45 →
  crossing_time_s = 30 →
  (train_speed_kmh * 1000 / 3600 * crossing_time_s) - train_length = 275 := by
sorry

end bridge_length_l115_11546


namespace night_shift_nine_hours_l115_11553

/-- Represents the number of hours worked by each guard -/
structure GuardShift :=
  (first : ℕ)
  (middle1 : ℕ)
  (middle2 : ℕ)
  (last : ℕ)

/-- Calculates the total length of the night shift -/
def nightShiftLength (shift : GuardShift) : ℕ :=
  shift.first + shift.middle1 + shift.middle2 + shift.last

/-- Theorem stating that the night shift length is 9 hours -/
theorem night_shift_nine_hours :
  ∃ (shift : GuardShift),
    shift.first = 3 ∧
    shift.middle1 = 2 ∧
    shift.middle2 = 2 ∧
    shift.last = 2 ∧
    nightShiftLength shift = 9 :=
by
  sorry

end night_shift_nine_hours_l115_11553


namespace angle_measure_problem_l115_11514

theorem angle_measure_problem (C D E F G : Real) : 
  C = 120 →
  C + D = 180 →
  E = 50 →
  F = D →
  E + F + G = 180 →
  G = 70 := by sorry

end angle_measure_problem_l115_11514


namespace exponential_function_properties_l115_11501

theorem exponential_function_properties (a : ℝ) (h : a > 1) :
  (∀ x : ℝ, (x = 0 → a^x = 1) ∧
            (x = 1 → a^x = a) ∧
            (x = -1 → a^x = 1/a) ∧
            (x < 0 → a^x > 0 ∧ ∀ ε > 0, ∃ N : ℝ, ∀ y < N, 0 < a^y ∧ a^y < ε)) :=
by sorry

end exponential_function_properties_l115_11501


namespace multiple_solutions_exist_l115_11527

-- Define the system of equations
def system (x y z w : ℝ) : Prop :=
  x = z + w - z*w ∧
  y = w + x - w*x ∧
  z = x + y - x*y ∧
  w = y + z - y*z

-- Theorem statement
theorem multiple_solutions_exist :
  ∃ (x₁ y₁ z₁ w₁ x₂ y₂ z₂ w₂ : ℝ),
    system x₁ y₁ z₁ w₁ ∧
    system x₂ y₂ z₂ w₂ ∧
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂ ∨ z₁ ≠ z₂ ∨ w₁ ≠ w₂) :=
by
  sorry


end multiple_solutions_exist_l115_11527


namespace triangle_angle_measure_l115_11555

theorem triangle_angle_measure (P Q R : ℝ) : 
  P = 90 → 
  Q = 4 * R - 10 → 
  P + Q + R = 180 → 
  R = 20 := by sorry

end triangle_angle_measure_l115_11555


namespace inverse_proportion_intersection_l115_11590

theorem inverse_proportion_intersection (b : ℝ) :
  ∃ k : ℝ, 1 < k ∧ k < 2 ∧
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
    (6 - 3 * k) / x₁ = -7 * x₁ + b ∧
    (6 - 3 * k) / x₂ = -7 * x₂ + b ∧
    x₁ * x₂ > 0) :=
by sorry

end inverse_proportion_intersection_l115_11590


namespace high_school_twelve_games_l115_11566

/-- The number of teams in the "High School Twelve" soccer conference -/
def num_teams : ℕ := 12

/-- The number of times each team plays every other conference team -/
def games_per_pair : ℕ := 3

/-- The number of non-conference games each team plays -/
def non_conference_games : ℕ := 5

/-- The total number of games in a season involving the "High School Twelve" teams -/
def total_games : ℕ := (num_teams.choose 2 * games_per_pair) + (num_teams * non_conference_games)

theorem high_school_twelve_games :
  total_games = 258 :=
by sorry

end high_school_twelve_games_l115_11566


namespace sequence_increasing_l115_11557

/-- Given positive real numbers a, b, c, and a natural number n,
    prove that a_n < a_{n+1} where a_n = (a*n)/(b*n + c) -/
theorem sequence_increasing (a b c : ℝ) (n : ℕ) 
    (ha : a > 0) (hb : b > 0) (hc : c > 0) :
    let a_n := (a * n) / (b * n + c)
    let a_n_plus_1 := (a * (n + 1)) / (b * (n + 1) + c)
    a_n < a_n_plus_1 := by
  sorry

end sequence_increasing_l115_11557


namespace power_calculation_l115_11567

theorem power_calculation : (16^4 * 8^6) / 4^12 = 1024 := by
  sorry

end power_calculation_l115_11567


namespace balls_in_boxes_l115_11532

/-- The number of ways to place n different balls into m different boxes, with at most one ball per box -/
def place_balls (n m : ℕ) : ℕ :=
  Nat.descFactorial m n

theorem balls_in_boxes : place_balls 3 5 = 60 := by
  sorry

end balls_in_boxes_l115_11532


namespace commute_time_difference_l115_11544

theorem commute_time_difference (distance : ℝ) (speed_actual : ℝ) (speed_suggested : ℝ) :
  distance = 10 ∧ speed_actual = 30 ∧ speed_suggested = 25 →
  (distance / speed_suggested - distance / speed_actual) * 60 = 4 := by
  sorry

end commute_time_difference_l115_11544


namespace solve_equation_l115_11511

theorem solve_equation (x : ℝ) (h : x + 1 = 3) : x = 2 := by
  sorry

end solve_equation_l115_11511


namespace arithmetic_sequence_fifth_term_l115_11549

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_fifth_term
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum : a 3 + a 8 = 22)
  (h_sixth : a 6 = 7) :
  a 5 = 15 := by
sorry

end arithmetic_sequence_fifth_term_l115_11549


namespace vanessa_score_l115_11571

/-- Vanessa's basketball score record problem -/
theorem vanessa_score (total_score : ℕ) (num_players : ℕ) (other_players_avg : ℚ) :
  total_score = 68 →
  num_players = 9 →
  other_players_avg = 4.5 →
  ∃ vanessa_score : ℕ,
    vanessa_score = 32 ∧
    vanessa_score = total_score - (num_players - 1) * (other_players_avg.num / other_players_avg.den) :=
by sorry

end vanessa_score_l115_11571


namespace polygon_sides_l115_11540

theorem polygon_sides (sum_interior_angles : ℝ) : sum_interior_angles = 720 → ∃ n : ℕ, n = 6 ∧ (n - 2) * 180 = sum_interior_angles := by
  sorry

end polygon_sides_l115_11540


namespace arithmetic_equality_l115_11536

theorem arithmetic_equality : 12.05 * 5.4 + 0.6 = 65.67 := by
  sorry

end arithmetic_equality_l115_11536


namespace profit_percentage_calculation_l115_11588

/-- Calculate the profit percentage given the selling price and profit -/
theorem profit_percentage_calculation (selling_price profit : ℝ) :
  selling_price = 850 ∧ profit = 205 →
  abs ((profit / (selling_price - profit)) * 100 - 31.78) < 0.01 := by
  sorry

end profit_percentage_calculation_l115_11588


namespace change_percentage_difference_l115_11554

/- Given conditions -/
def initial_yes : ℚ := 60 / 100
def initial_no : ℚ := 40 / 100
def final_yes : ℚ := 80 / 100
def final_no : ℚ := 20 / 100
def new_students : ℚ := 10 / 100

/- Theorem statement -/
theorem change_percentage_difference :
  let min_change := (final_yes - new_students) - initial_yes
  let max_change := min initial_no (final_yes - new_students) + min initial_yes final_no
  max_change - min_change = 40 / 100 := by
  sorry


end change_percentage_difference_l115_11554


namespace decimal_to_fraction_l115_11525

theorem decimal_to_fraction :
  (3.75 : ℚ) = 15 / 4 := by sorry

end decimal_to_fraction_l115_11525


namespace mens_haircut_time_is_correct_l115_11535

/-- The time it takes to cut a man's hair -/
def mens_haircut_time : ℕ := 15

/-- The time it takes to cut a woman's hair -/
def womens_haircut_time : ℕ := 50

/-- The time it takes to cut a kid's hair -/
def kids_haircut_time : ℕ := 25

/-- The number of women's haircuts Joe performed -/
def num_womens_haircuts : ℕ := 3

/-- The number of men's haircuts Joe performed -/
def num_mens_haircuts : ℕ := 2

/-- The number of kids' haircuts Joe performed -/
def num_kids_haircuts : ℕ := 3

/-- The total time Joe spent cutting hair -/
def total_time : ℕ := 255

theorem mens_haircut_time_is_correct :
  num_womens_haircuts * womens_haircut_time +
  num_mens_haircuts * mens_haircut_time +
  num_kids_haircuts * kids_haircut_time = total_time := by
sorry

end mens_haircut_time_is_correct_l115_11535


namespace nonzero_sum_zero_power_equality_l115_11508

theorem nonzero_sum_zero_power_equality (a b c : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (sum_zero : a + b + c = 0)
  (power_equality : a^4 + b^4 + c^4 = a^6 + b^6 + c^6) :
  a^2 + b^2 + c^2 = 3/2 := by
sorry

end nonzero_sum_zero_power_equality_l115_11508


namespace first_divisor_problem_l115_11570

theorem first_divisor_problem (m d : ℕ) : 
  (∃ q : ℕ, m = d * q + 47) →
  (∃ p : ℕ, m = 24 * p + 23) →
  (∀ x < d, ¬(∃ q : ℕ, m = x * q + 47)) →
  d = 72 := by
sorry

end first_divisor_problem_l115_11570


namespace arthur_walked_six_miles_l115_11599

/-- Calculates the total distance walked in miles given the number of blocks walked east and north, 
    and the length of each block in miles. -/
def total_distance (blocks_east : ℕ) (blocks_north : ℕ) (miles_per_block : ℚ) : ℚ :=
  (blocks_east + blocks_north : ℚ) * miles_per_block

/-- Theorem stating that Arthur walked 6 miles given the problem conditions. -/
theorem arthur_walked_six_miles :
  let blocks_east : ℕ := 6
  let blocks_north : ℕ := 12
  let miles_per_block : ℚ := 1/3
  total_distance blocks_east blocks_north miles_per_block = 6 := by
  sorry

end arthur_walked_six_miles_l115_11599
