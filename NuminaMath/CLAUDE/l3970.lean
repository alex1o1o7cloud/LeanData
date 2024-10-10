import Mathlib

namespace total_gas_used_l3970_397015

def gas_usage : List Float := [0.02, 0.015, 0.01, 0.03, 0.005, 0.025, 0.008, 0.018, 0.012, 0.005, 0.014, 0.01]

theorem total_gas_used (gas_usage : List Float) :
  gas_usage.sum = 0.172 := by
  sorry

#eval gas_usage.sum

end total_gas_used_l3970_397015


namespace ceiling_product_equation_l3970_397063

theorem ceiling_product_equation : ∃ x : ℝ, ⌈x⌉ * x = 220 ∧ x = 220 / 15 := by sorry

end ceiling_product_equation_l3970_397063


namespace sandy_correct_sums_l3970_397033

theorem sandy_correct_sums (total_sums : ℕ) (correct_marks : ℕ) (incorrect_marks : ℕ) (total_marks : ℤ) :
  total_sums = 30 →
  correct_marks = 3 →
  incorrect_marks = 2 →
  total_marks = 45 →
  ∃ (correct_sums : ℕ),
    correct_sums * correct_marks - (total_sums - correct_sums) * incorrect_marks = total_marks ∧
    correct_sums = 21 :=
by sorry

end sandy_correct_sums_l3970_397033


namespace T_divisibility_l3970_397078

-- Define the set T
def T : Set ℕ := {x | ∃ n : ℕ, x = (2*n - 2)^2 + (2*n)^2 + (2*n + 2)^2}

-- Theorem statement
theorem T_divisibility :
  (∀ x ∈ T, 4 ∣ x) ∧ (∃ x ∈ T, 5 ∣ x) := by
  sorry

end T_divisibility_l3970_397078


namespace hazel_caught_24_salmons_l3970_397084

/-- Represents the number of salmons caught by Hazel and her father -/
structure FishingTrip where
  total : ℕ
  father : ℕ

/-- Calculates the number of salmons Hazel caught -/
def hazel_catch (trip : FishingTrip) : ℕ :=
  trip.total - trip.father

/-- Theorem: Given the conditions of the fishing trip, prove that Hazel caught 24 salmons -/
theorem hazel_caught_24_salmons (trip : FishingTrip)
  (h1 : trip.total = 51)
  (h2 : trip.father = 27) :
  hazel_catch trip = 24 := by
sorry

end hazel_caught_24_salmons_l3970_397084


namespace pick_two_different_colors_custom_deck_l3970_397095

/-- A custom deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (num_suits : ℕ)
  (red_suits : ℕ)
  (black_suits : ℕ)
  (cards_per_suit : ℕ)

/-- The number of ways to pick two different cards of different colors -/
def pick_two_different_colors (d : Deck) : ℕ :=
  d.total_cards * (d.cards_per_suit * d.red_suits)

/-- Theorem stating the number of ways to pick two different cards of different colors -/
theorem pick_two_different_colors_custom_deck :
  ∃ (d : Deck), 
    d.total_cards = 60 ∧
    d.num_suits = 4 ∧
    d.red_suits = 2 ∧
    d.black_suits = 2 ∧
    d.cards_per_suit = 15 ∧
    pick_two_different_colors d = 1800 := by
  sorry

end pick_two_different_colors_custom_deck_l3970_397095


namespace parabola_tangent_to_line_l3970_397043

/-- Given a parabola y = ax^2 + 4 that is tangent to the line y = 3x + 1, prove that a = 3/4 -/
theorem parabola_tangent_to_line (a : ℝ) :
  (∃ x : ℝ, a * x^2 + 4 = 3 * x + 1 ∧ 
   ∀ y : ℝ, y ≠ x → a * y^2 + 4 ≠ 3 * y + 1) →
  a = 3/4 := by
sorry

end parabola_tangent_to_line_l3970_397043


namespace abc_positive_l3970_397086

/-- A quadratic function y = ax^2 + bx + c with specific properties -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  opens_upwards : a > 0
  has_two_real_roots : b^2 > 4*a*c
  right_root_larger : ∃ (r₁ r₂ : ℝ), r₁ < 0 ∧ r₂ > 0 ∧ |r₂| > |r₁| ∧
    a*r₁^2 + b*r₁ + c = 0 ∧ a*r₂^2 + b*r₂ + c = 0

/-- Theorem: For a quadratic function with the given properties, abc > 0 -/
theorem abc_positive (f : QuadraticFunction) : f.a * f.b * f.c > 0 := by
  sorry

end abc_positive_l3970_397086


namespace marcus_savings_l3970_397077

-- Define the given values
def max_budget : ℚ := 250
def shoe_price : ℚ := 120
def shoe_discount : ℚ := 0.3
def shoe_cashback : ℚ := 10
def shoe_tax : ℚ := 0.08
def sock_price : ℚ := 25
def sock_tax : ℚ := 0.06
def shirt_price : ℚ := 55
def shirt_discount : ℚ := 0.1
def shirt_tax : ℚ := 0.07

-- Define the calculation functions
def calculate_shoe_cost : ℚ := 
  (shoe_price * (1 - shoe_discount) - shoe_cashback) * (1 + shoe_tax)

def calculate_sock_cost : ℚ := 
  sock_price * (1 + sock_tax) / 2

def calculate_shirt_cost : ℚ := 
  shirt_price * (1 - shirt_discount) * (1 + shirt_tax)

def total_cost : ℚ := 
  calculate_shoe_cost + calculate_sock_cost + calculate_shirt_cost

-- Theorem statement
theorem marcus_savings : 
  max_budget - total_cost = 103.86 := by sorry

end marcus_savings_l3970_397077


namespace sibling_pair_implies_a_gt_one_l3970_397048

/-- A point pair (x₁, y₁) and (x₂, y₂) is a "sibling point pair" for a function f
    if they both lie on the graph of f and are symmetric about the origin. -/
def is_sibling_point_pair (f : ℝ → ℝ) (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  f x₁ = y₁ ∧ f x₂ = y₂ ∧ x₁ = -x₂ ∧ y₁ = -y₂

/-- The function f(x) = a^x - x - a has only one sibling point pair. -/
def has_unique_sibling_pair (a : ℝ) : Prop :=
  ∃! (x₁ y₁ x₂ y₂ : ℝ), is_sibling_point_pair (fun x => a^x - x - a) x₁ y₁ x₂ y₂

theorem sibling_pair_implies_a_gt_one (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) 
    (h₃ : has_unique_sibling_pair a) : a > 1 := by
  sorry

end sibling_pair_implies_a_gt_one_l3970_397048


namespace points_earned_is_75_l3970_397026

/-- Represents the point system and enemy counts in the video game level -/
structure GameLevel where
  goblin_points : ℕ
  orc_points : ℕ
  dragon_points : ℕ
  goblins_defeated : ℕ
  orcs_defeated : ℕ
  dragons_defeated : ℕ

/-- Calculates the total points earned in a game level -/
def total_points (level : GameLevel) : ℕ :=
  level.goblin_points * level.goblins_defeated +
  level.orc_points * level.orcs_defeated +
  level.dragon_points * level.dragons_defeated

/-- Theorem stating that the total points earned in the given scenario is 75 -/
theorem points_earned_is_75 (level : GameLevel) 
  (h1 : level.goblin_points = 3)
  (h2 : level.orc_points = 5)
  (h3 : level.dragon_points = 10)
  (h4 : level.goblins_defeated = 10)
  (h5 : level.orcs_defeated = 7)
  (h6 : level.dragons_defeated = 1) :
  total_points level = 75 := by
  sorry


end points_earned_is_75_l3970_397026


namespace school_club_members_l3970_397052

theorem school_club_members :
  ∃! n : ℕ, 0 < n ∧ n < 50 ∧ n % 6 = 4 ∧ n % 5 = 2 ∧ n = 28 := by
  sorry

end school_club_members_l3970_397052


namespace dodge_truck_count_l3970_397094

/-- The number of vehicles in the Taco Castle parking lot -/
structure VehicleCount where
  dodge : ℕ
  ford : ℕ
  toyota : ℕ
  volkswagen : ℕ
  honda : ℕ
  chevrolet : ℕ

/-- The relationships between different vehicle types in the parking lot -/
def valid_count (v : VehicleCount) : Prop :=
  v.ford = v.dodge / 3 ∧
  v.ford = 2 * v.toyota ∧
  v.volkswagen = v.toyota / 2 ∧
  v.honda = (3 * v.ford) / 4 ∧
  v.chevrolet = (2 * v.honda) / 3 ∧
  v.volkswagen = 5

theorem dodge_truck_count (v : VehicleCount) (h : valid_count v) : v.dodge = 60 := by
  sorry

end dodge_truck_count_l3970_397094


namespace parabola_point_relationship_l3970_397057

/-- A parabola defined by y = 3x² - 6x + c -/
def parabola (x y c : ℝ) : Prop := y = 3 * x^2 - 6 * x + c

/-- Three points on the parabola -/
def point_A (y₁ c : ℝ) : Prop := parabola (-3) y₁ c
def point_B (y₂ c : ℝ) : Prop := parabola (-1) y₂ c
def point_C (y₃ c : ℝ) : Prop := parabola 5 y₃ c

/-- Theorem stating the relationship between y₁, y₂, and y₃ -/
theorem parabola_point_relationship (y₁ y₂ y₃ c : ℝ) 
  (hA : point_A y₁ c) (hB : point_B y₂ c) (hC : point_C y₃ c) :
  y₁ = y₃ ∧ y₁ > y₂ := by sorry

end parabola_point_relationship_l3970_397057


namespace price_reduction_percentage_l3970_397050

theorem price_reduction_percentage (initial_price final_price : ℝ) 
  (h1 : initial_price = 100)
  (h2 : final_price = 81)
  (h3 : final_price = initial_price * (1 - x)^2)
  (h4 : 0 < x ∧ x < 1) :
  x = 0.1 := by sorry

end price_reduction_percentage_l3970_397050


namespace inequality_proof_l3970_397031

theorem inequality_proof (x y z : ℝ) 
  (h1 : x + 2*y + 4*z ≥ 3) 
  (h2 : y - 3*x + 2*z ≥ 5) : 
  y - x + 2*z ≥ 3 := by
  sorry

end inequality_proof_l3970_397031


namespace beach_trip_time_l3970_397005

/-- Calculates the total trip time given the one-way drive time and the ratio of destination time to total drive time -/
def totalTripTime (oneWayDriveTime : ℝ) (destinationTimeRatio : ℝ) : ℝ :=
  let totalDriveTime := 2 * oneWayDriveTime
  let destinationTime := destinationTimeRatio * totalDriveTime
  totalDriveTime + destinationTime

/-- Proves that for a trip with 2 hours one-way drive time and 2.5 ratio of destination time to total drive time, the total trip time is 14 hours -/
theorem beach_trip_time : totalTripTime 2 2.5 = 14 := by
  sorry

end beach_trip_time_l3970_397005


namespace gcd_of_sides_gt_one_l3970_397097

/-- A triangle with integer sides -/
structure IntegerTriangle where
  a : ℕ  -- side BC
  b : ℕ  -- side CA
  c : ℕ  -- side AB
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- The theorem to be proved -/
theorem gcd_of_sides_gt_one
  (t : IntegerTriangle)
  (side_order : t.c < t.b)
  (tangent_intersect : ℕ)  -- AD, the intersection of tangent at A with BC
  : Nat.gcd t.b t.c > 1 := by
  sorry

end gcd_of_sides_gt_one_l3970_397097


namespace count_paths_l3970_397082

/-- The number of paths on a 6x5 grid from A to B with specific conditions -/
def num_paths : ℕ := 252

/-- The width of the grid -/
def grid_width : ℕ := 6

/-- The height of the grid -/
def grid_height : ℕ := 5

/-- The total number of moves required -/
def total_moves : ℕ := 11

/-- Theorem stating the number of paths under given conditions -/
theorem count_paths :
  num_paths = Nat.choose (total_moves - 1) grid_height :=
sorry

end count_paths_l3970_397082


namespace tenth_fibonacci_is_55_l3970_397030

def fibonacci : ℕ → ℕ
| 0 => 1
| 1 => 1
| (n + 2) => fibonacci n + fibonacci (n + 1)

theorem tenth_fibonacci_is_55 : fibonacci 9 = 55 := by
  sorry

end tenth_fibonacci_is_55_l3970_397030


namespace andre_flowers_l3970_397038

/-- The number of flowers Andre gave to Rosa -/
def flowers_given : ℕ := 90 - 67

/-- Rosa's initial number of flowers -/
def initial_flowers : ℕ := 67

/-- Rosa's final number of flowers -/
def final_flowers : ℕ := 90

theorem andre_flowers : flowers_given = final_flowers - initial_flowers := by
  sorry

end andre_flowers_l3970_397038


namespace friends_receiving_balls_l3970_397099

/-- The number of ping pong balls Eunji has -/
def total_balls : ℕ := 44

/-- The number of ping pong balls given to each friend -/
def balls_per_friend : ℕ := 4

/-- The number of friends who will receive ping pong balls -/
def num_friends : ℕ := total_balls / balls_per_friend

theorem friends_receiving_balls : num_friends = 11 := by
  sorry

end friends_receiving_balls_l3970_397099


namespace complex_fraction_sum_l3970_397049

theorem complex_fraction_sum : 
  (Complex.I : ℂ) ^ 2 = -1 → 
  (7 + 3 * Complex.I) / (7 - 3 * Complex.I) + (7 - 3 * Complex.I) / (7 + 3 * Complex.I) = 2 := by
  sorry

end complex_fraction_sum_l3970_397049


namespace cubic_sum_theorem_l3970_397070

theorem cubic_sum_theorem (p q r : ℝ) (h_distinct : p ≠ q ∧ q ≠ r ∧ p ≠ r) 
  (h_eq : (p^3 - 12)/p = (q^3 - 12)/q ∧ (q^3 - 12)/q = (r^3 - 12)/r) : 
  p^3 + q^3 + r^3 = -36 := by
sorry

end cubic_sum_theorem_l3970_397070


namespace arithmetic_sequence_formula_l3970_397096

/-- An arithmetic sequence with given first three terms -/
def arithmetic_sequence (x : ℝ) (n : ℕ) : ℝ :=
  let a₁ := x - 1
  let a₂ := x + 1
  let a₃ := 2 * x + 3
  let d := a₂ - a₁  -- common difference
  a₁ + (n - 1) * d

/-- Theorem stating the general formula for the given arithmetic sequence -/
theorem arithmetic_sequence_formula (x : ℝ) (n : ℕ) :
  arithmetic_sequence x n = 2 * n - 3 := by
  sorry

end arithmetic_sequence_formula_l3970_397096


namespace find_divisor_l3970_397051

theorem find_divisor (divisor : ℕ) : divisor = 2 := by
  have h1 : 2 = (433126 : ℕ) - 433124 := by sorry
  have h2 : (433126 : ℕ) % divisor = 0 := by sorry
  have h3 : ∀ n : ℕ, n < 2 → (433124 + n) % divisor ≠ 0 := by sorry
  sorry

end find_divisor_l3970_397051


namespace popped_kernels_in_final_bag_l3970_397046

theorem popped_kernels_in_final_bag 
  (bag1_popped bag1_total bag2_popped bag2_total bag3_total : ℕ)
  (average_percent : ℚ)
  (h1 : bag1_popped = 60)
  (h2 : bag1_total = 75)
  (h3 : bag2_popped = 42)
  (h4 : bag2_total = 50)
  (h5 : bag3_total = 100)
  (h6 : average_percent = 82/100)
  (h7 : (bag1_popped : ℚ) / bag1_total + (bag2_popped : ℚ) / bag2_total + 
        (bag3_popped : ℚ) / bag3_total = 3 * average_percent) :
  bag3_popped = 82 := by
  sorry

#check popped_kernels_in_final_bag

end popped_kernels_in_final_bag_l3970_397046


namespace expression_simplification_l3970_397067

theorem expression_simplification (x y : ℚ) 
  (hx : x = -1/2) (hy : y = 2) : 
  6 * (x^2 - (1/3) * x * y) - 3 * (x^2 - x * y) - 2 * x^2 = -3/4 := by
  sorry

end expression_simplification_l3970_397067


namespace boat_speed_in_still_water_l3970_397087

/-- Given a boat traveling downstream with a current of 5 km/hr,
    if it covers a distance of 7.5 km in 18 minutes,
    then its speed in still water is 20 km/hr. -/
theorem boat_speed_in_still_water 
  (current_speed : ℝ) 
  (distance_downstream : ℝ) 
  (time_minutes : ℝ) 
  (boat_speed : ℝ) :
  current_speed = 5 →
  distance_downstream = 7.5 →
  time_minutes = 18 →
  distance_downstream = (boat_speed + current_speed) * (time_minutes / 60) →
  boat_speed = 20 :=
by sorry

end boat_speed_in_still_water_l3970_397087


namespace tangent_point_value_l3970_397091

/-- The value of 'a' for which the line y = x + 1 is tangent to the curve y = ln(x + a) --/
def tangent_point (a : ℝ) : Prop :=
  ∃ x : ℝ, 
    -- The y-coordinate of the line and curve are equal at the point of tangency
    x + 1 = Real.log (x + a) ∧ 
    -- The slope of the line (which is 1) equals the derivative of ln(x + a) at the point of tangency
    1 = 1 / (x + a)

/-- Theorem stating that 'a' must equal 2 for the tangency condition to be satisfied --/
theorem tangent_point_value : 
  ∃ a : ℝ, tangent_point a ∧ a = 2 :=
sorry

end tangent_point_value_l3970_397091


namespace circle_radius_in_square_l3970_397011

theorem circle_radius_in_square (side_length : ℝ) (l_shape_ratio : ℝ) : 
  side_length = 144 →
  l_shape_ratio = 5/18 →
  let total_area := side_length^2
  let l_shape_area := l_shape_ratio * total_area
  let center_square_area := total_area - 4 * l_shape_area
  let center_square_side := Real.sqrt center_square_area
  let radius := center_square_side / 2
  radius = 61.2 := by sorry

end circle_radius_in_square_l3970_397011


namespace circuit_board_count_l3970_397016

/-- The number of circuit boards that fail verification -/
def failed_boards : ℕ := 64

/-- The fraction of boards that pass verification but are faulty -/
def faulty_fraction : ℚ := 1 / 8

/-- The total number of faulty boards -/
def total_faulty : ℕ := 456

/-- The total number of circuit boards in the group -/
def total_boards : ℕ := 3200

theorem circuit_board_count :
  (failed_boards : ℚ) + faulty_fraction * (total_boards - failed_boards : ℚ) = total_faulty ∧
  total_boards = failed_boards + (total_faulty - failed_boards) / faulty_fraction := by
  sorry

end circuit_board_count_l3970_397016


namespace fraction_calculation_l3970_397017

theorem fraction_calculation : (17/5) + (-23/8) - (-28/5) - (1/8) = 6 := by
  sorry

end fraction_calculation_l3970_397017


namespace problem_statement_l3970_397032

theorem problem_statement (h1 : x = y → z ≠ w) (h2 : z = w → p ≠ q) : x ≠ y → p ≠ q := by
  sorry

end problem_statement_l3970_397032


namespace greatest_prime_factor_of_154_l3970_397068

theorem greatest_prime_factor_of_154 : ∃ p : ℕ, Nat.Prime p ∧ p ∣ 154 ∧ ∀ q : ℕ, Nat.Prime q → q ∣ 154 → q ≤ p ∧ p = 11 := by
  sorry

end greatest_prime_factor_of_154_l3970_397068


namespace algorithm_characteristic_is_determinacy_l3970_397075

-- Define the concept of an algorithm step
structure AlgorithmStep where
  definite : Bool
  executable : Bool
  yieldsDefiniteResult : Bool

-- Define the characteristic of determinacy
def isDeterminacy (step : AlgorithmStep) : Prop :=
  step.definite ∧ step.executable ∧ step.yieldsDefiniteResult

-- Theorem statement
theorem algorithm_characteristic_is_determinacy (step : AlgorithmStep) :
  step.definite ∧ step.executable ∧ step.yieldsDefiniteResult → isDeterminacy step :=
by
  sorry

#check algorithm_characteristic_is_determinacy

end algorithm_characteristic_is_determinacy_l3970_397075


namespace least_subtraction_for_divisibility_l3970_397080

theorem least_subtraction_for_divisibility : ∃ (n : ℕ), n = 5 ∧ 
  (∀ (m : ℕ), m < n → ¬(31 ∣ (42739 - m))) ∧ (31 ∣ (42739 - n)) := by
  sorry

end least_subtraction_for_divisibility_l3970_397080


namespace teagan_savings_proof_l3970_397014

def nickel_value : ℚ := 0.05
def dime_value : ℚ := 0.10
def penny_value : ℚ := 0.01

def rex_nickels : ℕ := 100
def toni_dimes : ℕ := 330
def total_savings : ℚ := 40

def teagan_pennies : ℕ := 200

theorem teagan_savings_proof :
  (rex_nickels : ℚ) * nickel_value + (toni_dimes : ℚ) * dime_value + (teagan_pennies : ℚ) * penny_value = total_savings :=
by sorry

end teagan_savings_proof_l3970_397014


namespace sum_of_triangles_l3970_397064

/-- The triangle operation defined as a × b - c -/
def triangle (a b c : ℝ) : ℝ := a * b - c

/-- Theorem stating that the sum of two specific triangle operations equals -2 -/
theorem sum_of_triangles : triangle 2 3 5 + triangle 1 4 7 = -2 := by sorry

end sum_of_triangles_l3970_397064


namespace square_sum_equals_three_l3970_397003

theorem square_sum_equals_three (a b : ℝ) 
  (h : a^4 + b^4 = a^2 - 2*a^2*b^2 + b^2 + 6) : 
  a^2 + b^2 = 3 := by
  sorry

end square_sum_equals_three_l3970_397003


namespace two_divisors_of_ten_billion_sum_to_157_l3970_397036

theorem two_divisors_of_ten_billion_sum_to_157 :
  ∃ (a b : ℕ), 
    a ≠ b ∧
    a > 0 ∧
    b > 0 ∧
    (10^10 % a = 0) ∧
    (10^10 % b = 0) ∧
    a + b = 157 ∧
    a = 32 ∧
    b = 125 := by
  sorry

end two_divisors_of_ten_billion_sum_to_157_l3970_397036


namespace number_with_specific_remainders_l3970_397044

theorem number_with_specific_remainders : ∃ (N : ℕ), N % 13 = 11 ∧ N % 17 = 9 := by
  sorry

end number_with_specific_remainders_l3970_397044


namespace reciprocal_of_negative_one_third_l3970_397024

theorem reciprocal_of_negative_one_third :
  let x : ℚ := -1/3
  let y : ℚ := -3
  x * y = 1 ∧ ∀ z : ℚ, x * z = 1 → z = y :=
by sorry

end reciprocal_of_negative_one_third_l3970_397024


namespace pencil_cost_l3970_397041

theorem pencil_cost (total_money : ℕ) (num_pencils : ℕ) (cost_per_pencil : ℕ) : 
  total_money = 50 → 
  num_pencils = 10 → 
  total_money = num_pencils * cost_per_pencil → 
  cost_per_pencil = 5 :=
by
  sorry

end pencil_cost_l3970_397041


namespace fraction_equals_875_l3970_397009

theorem fraction_equals_875 (a : ℕ+) (h : (a : ℚ) / ((a : ℚ) + 35) = 875 / 1000) : 
  a = 245 := by sorry

end fraction_equals_875_l3970_397009


namespace trapezoid_perimeter_l3970_397090

/-- Represents a trapezoid EFGH with specific properties -/
structure Trapezoid where
  -- Length of side EF
  ef : ℝ
  -- Length of side GH
  gh : ℝ
  -- Height of the trapezoid
  height : ℝ
  -- Area of the trapezoid
  area : ℝ
  -- EF is half the length of GH
  ef_half_gh : ef = gh / 2
  -- Height is 6 units
  height_is_6 : height = 6
  -- Area is 90 square units
  area_is_90 : area = 90

/-- Calculate the perimeter of the trapezoid -/
def perimeter (t : Trapezoid) : ℝ := sorry

/-- Theorem stating that the perimeter of the trapezoid is 30 + 2√61 -/
theorem trapezoid_perimeter (t : Trapezoid) : perimeter t = 30 + 2 * Real.sqrt 61 := by
  sorry

end trapezoid_perimeter_l3970_397090


namespace quadratic_real_roots_l3970_397037

theorem quadratic_real_roots (k : ℝ) :
  (∃ x : ℝ, x^2 + 2*k*x + 3*k^2 + 2*k = 0) ↔ -1 ≤ k ∧ k ≤ 0 := by
  sorry

end quadratic_real_roots_l3970_397037


namespace third_number_is_58_l3970_397020

def number_list : List ℕ := [54, 55, 58, 59, 62, 62, 63, 65, 65]

theorem third_number_is_58 : 
  number_list[2] = 58 := by sorry

end third_number_is_58_l3970_397020


namespace smallest_n_for_square_root_96n_l3970_397069

theorem smallest_n_for_square_root_96n (n : ℕ) : 
  (∃ k : ℕ, k * k = 96 * n) → n ≥ 6 :=
by sorry

end smallest_n_for_square_root_96n_l3970_397069


namespace solve_proportion_l3970_397019

theorem solve_proportion (x : ℚ) (h : x / 6 = 15 / 10) : x = 9 := by
  sorry

end solve_proportion_l3970_397019


namespace cement_total_l3970_397004

theorem cement_total (bought : ℕ) (brought : ℕ) (total : ℕ) : 
  bought = 215 → brought = 137 → total = bought + brought → total = 352 := by
  sorry

end cement_total_l3970_397004


namespace transaction_outcome_l3970_397061

theorem transaction_outcome : 
  let house_sell := 15000
  let store_sell := 14000
  let vehicle_sell := 18000
  let house_loss_percent := 25
  let store_gain_percent := 16.67
  let vehicle_gain_percent := 12.5
  
  let house_cost := house_sell / (1 - house_loss_percent / 100)
  let store_cost := store_sell / (1 + store_gain_percent / 100)
  let vehicle_cost := vehicle_sell / (1 + vehicle_gain_percent / 100)
  
  let total_cost := house_cost + store_cost + vehicle_cost
  let total_sell := house_sell + store_sell + vehicle_sell
  
  total_cost - total_sell = 1000 := by sorry

end transaction_outcome_l3970_397061


namespace symmetric_point_coordinates_l3970_397093

def Point := ℝ × ℝ

def symmetric_origin (p1 p2 : Point) : Prop :=
  p1.1 = -p2.1 ∧ p1.2 = -p2.2

def symmetric_y_axis (p1 p2 : Point) : Prop :=
  p1.1 = -p2.1 ∧ p1.2 = p2.2

theorem symmetric_point_coordinates :
  ∀ (P P1 P2 : Point),
    symmetric_origin P1 P →
    P1 = (-2, 3) →
    symmetric_y_axis P2 P →
    P2 = (-2, -3) := by
  sorry

end symmetric_point_coordinates_l3970_397093


namespace investment_of_c_is_120000_l3970_397059

/-- Represents the investment and profit share of a business partner -/
structure Partner where
  investment : ℕ
  profitShare : ℕ

/-- Calculates the investment of partner C given the investments and profit shares of A and B -/
def calculateInvestmentC (a : Partner) (b : Partner) (profitShareDiffAC : ℕ) : ℕ :=
  let profitShareA := a.investment * b.profitShare / b.investment
  let profitShareC := profitShareA + profitShareDiffAC
  profitShareC * b.investment / b.profitShare

/-- Theorem stating that given the problem conditions, C's investment is 120000 -/
theorem investment_of_c_is_120000 : 
  let a : Partner := ⟨8000, 0⟩
  let b : Partner := ⟨10000, 1700⟩
  let profitShareDiffAC := 680
  calculateInvestmentC a b profitShareDiffAC = 120000 := by
  sorry

#eval calculateInvestmentC ⟨8000, 0⟩ ⟨10000, 1700⟩ 680

end investment_of_c_is_120000_l3970_397059


namespace sum_of_coefficients_l3970_397079

theorem sum_of_coefficients : 
  let p (x : ℝ) := (3*x^8 - 2*x^7 + 4*x^6 - x^4 + 6*x^2 - 7) - 
                   5*(x^5 - 2*x^3 + 2*x - 8) + 
                   6*(x^6 + x^4 - 3)
  p 1 = 32 := by sorry

end sum_of_coefficients_l3970_397079


namespace two_thousand_seventeenth_number_l3970_397065

def is_divisible_by_2_or_3 (n : ℕ) : Prop := 2 ∣ n ∨ 3 ∣ n

def sequence_2_or_3 : ℕ → ℕ := sorry

theorem two_thousand_seventeenth_number :
  sequence_2_or_3 2017 = 3026 := by sorry

end two_thousand_seventeenth_number_l3970_397065


namespace speed_calculation_l3970_397034

/-- Given a speed v and time t, if increasing the speed by 12 miles per hour
    reduces the time by 1/4, then v = 36 miles per hour. -/
theorem speed_calculation (v t : ℝ) (h : v * t = (v + 12) * (3/4 * t)) : v = 36 :=
sorry

end speed_calculation_l3970_397034


namespace ab_is_zero_l3970_397010

theorem ab_is_zero (a b : ℝ) (h1 : a + b = 5) (h2 : a^3 + b^3 = 125) : a * b = 0 := by
  sorry

end ab_is_zero_l3970_397010


namespace series_relationship_l3970_397001

-- Define the sequence of exponents
def a : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | n + 2 => a n + a (n + 1)

-- Define the series
def series (n : ℕ) : ℕ := 2^(a n)

-- Theorem statement
theorem series_relationship (n : ℕ) :
  series n * series (n + 1) = series (n + 2) := by
  sorry


end series_relationship_l3970_397001


namespace doug_money_l3970_397088

theorem doug_money (j d b s : ℚ) : 
  j + d + b + s = 150 →
  j = 2 * b →
  j = (3/4) * d →
  s = (1/2) * (j + d + b) →
  d = (4/3) * (150 * 12/41) := by
sorry

end doug_money_l3970_397088


namespace jerry_shelf_theorem_l3970_397055

/-- The number of action figures and books on Jerry's shelf -/
def shelf_contents : ℕ × ℕ := (5, 9)

/-- The number of action figures added later -/
def added_figures : ℕ := 7

/-- The final difference between action figures and books -/
def figure_book_difference : ℤ :=
  (shelf_contents.1 + added_figures : ℤ) - shelf_contents.2

theorem jerry_shelf_theorem :
  figure_book_difference = 3 := by sorry

end jerry_shelf_theorem_l3970_397055


namespace locus_and_slope_theorem_l3970_397007

noncomputable def A : ℝ × ℝ := (0, 4/3)
noncomputable def B : ℝ × ℝ := (-1, 0)
noncomputable def C : ℝ × ℝ := (1, 0)

def distance_to_line (P : ℝ × ℝ) (l : ℝ × ℝ → Prop) : ℝ := sorry

def line_BC : ℝ × ℝ → Prop := sorry
def line_AB : ℝ × ℝ → Prop := sorry
def line_AC : ℝ × ℝ → Prop := sorry

def locus_equation_1 (P : ℝ × ℝ) : Prop :=
  (P.1^2 + P.2^2 + 3/2 * P.2 - 1 = 0)

def locus_equation_2 (P : ℝ × ℝ) : Prop :=
  (8 * P.1^2 - 17 * P.2^2 + 12 * P.2 - 8 = 0)

def incenter (triangle : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)) : ℝ × ℝ := sorry

def line_intersects_locus_at_3_points (l : ℝ → ℝ) : Prop := sorry

def slope_set : Set ℝ := {0, 1/2, -1/2, 2 * Real.sqrt 34 / 17, -2 * Real.sqrt 34 / 17, Real.sqrt 2 / 2, -Real.sqrt 2 / 2}

theorem locus_and_slope_theorem :
  ∀ P : ℝ × ℝ,
  (distance_to_line P line_BC)^2 = (distance_to_line P line_AB) * (distance_to_line P line_AC) →
  (locus_equation_1 P ∨ locus_equation_2 P) ∧
  ∀ l : ℝ → ℝ,
  (∃ x : ℝ, l x = (incenter (A, B, C)).2) →
  line_intersects_locus_at_3_points l →
  ∃ k : ℝ, k ∈ slope_set ∧ ∀ x : ℝ, l x = k * x + (incenter (A, B, C)).2 :=
sorry

end locus_and_slope_theorem_l3970_397007


namespace bag_price_with_discount_l3970_397025

theorem bag_price_with_discount (selling_price : ℝ) (discount_percentage : ℝ) 
  (h1 : selling_price = 120)
  (h2 : discount_percentage = 4) : 
  selling_price / (1 - discount_percentage / 100) = 125 := by
  sorry

end bag_price_with_discount_l3970_397025


namespace expression_value_l3970_397081

theorem expression_value (a b : ℤ) (ha : a = -4) (hb : b = 3) :
  -a - b^2 + a*b = -17 := by sorry

end expression_value_l3970_397081


namespace stable_performance_comparison_l3970_397012

/-- Represents a student's performance in standing long jumps --/
structure StudentPerformance where
  average_score : ℝ
  variance : ℝ

/-- Determines if a student's performance is more stable --/
def more_stable (a b : StudentPerformance) : Prop :=
  a.variance < b.variance

/-- Theorem: Given two students with the same average score, 
    the one with lower variance has more stable performance --/
theorem stable_performance_comparison 
  (student_a student_b : StudentPerformance)
  (h_same_average : student_a.average_score = student_b.average_score)
  (h_a_variance : student_a.variance = 0.48)
  (h_b_variance : student_b.variance = 0.53) :
  more_stable student_a student_b :=
by
  sorry

end stable_performance_comparison_l3970_397012


namespace smallest_n_with_unique_k_l3970_397035

theorem smallest_n_with_unique_k : ∃ (k : ℤ),
  (7 : ℚ) / 16 < (63 : ℚ) / (63 + k) ∧ (63 : ℚ) / (63 + k) < 9 / 20 ∧
  (∀ (k' : ℤ), k' ≠ k →
    ((7 : ℚ) / 16 ≥ (63 : ℚ) / (63 + k') ∨ (63 : ℚ) / (63 + k') ≥ 9 / 20)) ∧
  (∀ (n : ℕ), 0 < n → n < 63 →
    ¬(∃! (k : ℤ), (7 : ℚ) / 16 < (n : ℚ) / (n + k) ∧ (n : ℚ) / (n + k) < 9 / 20)) :=
by sorry

end smallest_n_with_unique_k_l3970_397035


namespace odd_function_implies_m_zero_l3970_397013

/-- A function f is odd if f(-x) = -f(x) for all x in its domain -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The function f(x) = x^3 + 3mx^2 + nx + m^2 -/
def f (m n : ℝ) (x : ℝ) : ℝ :=
  x^3 + 3*m*x^2 + n*x + m^2

theorem odd_function_implies_m_zero (m n : ℝ) :
  IsOdd (f m n) → m = 0 := by
  sorry

end odd_function_implies_m_zero_l3970_397013


namespace fred_balloons_l3970_397006

theorem fred_balloons (total sam mary : ℕ) (h1 : total = 18) (h2 : sam = 6) (h3 : mary = 7) :
  total - (sam + mary) = 5 := by
  sorry

end fred_balloons_l3970_397006


namespace card_difference_l3970_397021

/-- The number of cards each person has -/
structure CardCounts where
  ann : ℕ
  anton : ℕ
  heike : ℕ

/-- The conditions of the problem -/
def card_problem (c : CardCounts) : Prop :=
  c.ann = 60 ∧
  c.ann = 6 * c.heike ∧
  c.anton = c.heike

/-- The theorem to prove -/
theorem card_difference (c : CardCounts) (h : card_problem c) : 
  c.ann - c.anton = 50 := by
  sorry

end card_difference_l3970_397021


namespace lagrange_interpolation_uniqueness_existence_l3970_397047

theorem lagrange_interpolation_uniqueness_existence
  (n : ℕ) 
  (x : Fin (n + 1) → ℝ) 
  (a : Fin (n + 1) → ℝ) 
  (h_distinct : ∀ (i j : Fin (n + 1)), i ≠ j → x i ≠ x j) :
  ∃! P : Polynomial ℝ, 
    (Polynomial.degree P ≤ n) ∧ 
    (∀ i : Fin (n + 1), P.eval (x i) = a i) :=
sorry

end lagrange_interpolation_uniqueness_existence_l3970_397047


namespace trisha_chicken_expense_l3970_397054

/-- Given Trisha's shopping expenses and initial amount, prove that she spent $22 on chicken -/
theorem trisha_chicken_expense (meat_cost veggies_cost eggs_cost dog_food_cost initial_amount remaining_amount : ℕ) 
  (h1 : meat_cost = 17)
  (h2 : veggies_cost = 43)
  (h3 : eggs_cost = 5)
  (h4 : dog_food_cost = 45)
  (h5 : initial_amount = 167)
  (h6 : remaining_amount = 35) :
  initial_amount - remaining_amount - (meat_cost + veggies_cost + eggs_cost + dog_food_cost) = 22 := by
  sorry

end trisha_chicken_expense_l3970_397054


namespace count_congruent_is_77_l3970_397089

/-- The number of positive integers less than 1000 that are congruent to 7 (mod 13) -/
def count_congruent : ℕ :=
  (Finset.filter (fun n => n > 0 ∧ n < 1000 ∧ n % 13 = 7) (Finset.range 1000)).card

/-- Theorem stating that the count of such integers is 77 -/
theorem count_congruent_is_77 : count_congruent = 77 := by
  sorry

end count_congruent_is_77_l3970_397089


namespace complex_exponential_to_rectangular_l3970_397062

theorem complex_exponential_to_rectangular : 
  Complex.exp (Complex.I * (13 * Real.pi / 6)) * (Real.sqrt 3 : ℂ) = (3 / 2 : ℂ) + Complex.I * ((Real.sqrt 3) / 2 : ℂ) := by
  sorry

end complex_exponential_to_rectangular_l3970_397062


namespace arithmetic_sequence_2013_l3970_397056

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  a₁_eq_1 : a 1 = 1
  d : ℝ
  d_neq_0 : d ≠ 0
  is_arithmetic : ∀ n : ℕ, a (n + 1) = a n + d
  is_geometric : (a 2)^2 = a 1 * a 5

/-- The 2013th term of the arithmetic sequence is 4025 -/
theorem arithmetic_sequence_2013 (seq : ArithmeticSequence) : seq.a 2013 = 4025 := by
  sorry

end arithmetic_sequence_2013_l3970_397056


namespace not_parabola_l3970_397083

/-- The equation x² + y²cos(θ) = 1, where θ is any real number, cannot represent a parabola -/
theorem not_parabola (θ : ℝ) : 
  ¬ (∃ (a b c d e : ℝ), ∀ (x y : ℝ), 
    (x^2 + y^2 * Real.cos θ = 1) ↔ (a*x^2 + b*x*y + c*y^2 + d*x + e*y = 1 ∧ b^2 = 4*a*c)) :=
by sorry

end not_parabola_l3970_397083


namespace tire_company_cost_per_batch_l3970_397073

/-- A tire company's production and sales model -/
structure TireCompany where
  cost_per_batch : ℝ
  cost_per_tire : ℝ
  selling_price : ℝ
  batch_size : ℕ
  profit_per_tire : ℝ

/-- The cost per batch for the tire company -/
def cost_per_batch (company : TireCompany) : ℝ :=
  company.cost_per_batch

/-- Theorem stating the cost per batch for the given scenario -/
theorem tire_company_cost_per_batch :
  ∀ (company : TireCompany),
    company.cost_per_tire = 8 →
    company.selling_price = 20 →
    company.batch_size = 15000 →
    company.profit_per_tire = 10.5 →
    cost_per_batch company = 22500 := by
  sorry

end tire_company_cost_per_batch_l3970_397073


namespace arrangement_count_is_7200_l3970_397058

/-- The number of consonants in the word "ИНТЕГРАЛ" -/
def num_consonants : ℕ := 5

/-- The number of vowels in the word "ИНТЕГРАЛ" -/
def num_vowels : ℕ := 3

/-- The total number of letters in the word "ИНТЕГРАЛ" -/
def total_letters : ℕ := num_consonants + num_vowels

/-- The number of positions that must be occupied by consonants -/
def required_consonant_positions : ℕ := 3

/-- The number of remaining positions after placing consonants in required positions -/
def remaining_positions : ℕ := total_letters - required_consonant_positions

/-- The number of ways to arrange the letters in "ИНТЕГРАЛ" with consonants in specific positions -/
def arrangement_count : ℕ := 
  (num_consonants.factorial / (num_consonants - required_consonant_positions).factorial) * 
  remaining_positions.factorial

theorem arrangement_count_is_7200 : arrangement_count = 7200 := by
  sorry

end arrangement_count_is_7200_l3970_397058


namespace distance_between_points_l3970_397092

/-- The distance between two points (3, 0) and (7, 7) on a Cartesian coordinate plane is √65. -/
theorem distance_between_points : Real.sqrt 65 = Real.sqrt ((7 - 3)^2 + (7 - 0)^2) := by
  sorry

end distance_between_points_l3970_397092


namespace decimal_437_equals_fraction_l3970_397072

/-- The decimal representation of 0.4̄37 as a rational number -/
def decimal_437 : ℚ := 437/990 - 4/990

/-- The fraction 43693/99900 -/
def fraction_43693_99900 : ℚ := 43693/99900

theorem decimal_437_equals_fraction : 
  decimal_437 = fraction_43693_99900 ∧ 
  (∀ n d : ℕ, n ≠ 0 ∧ d ≠ 0 → fraction_43693_99900 = n / d → n = 43693 ∧ d = 99900) := by
  sorry

#check decimal_437_equals_fraction

end decimal_437_equals_fraction_l3970_397072


namespace inequality_equivalence_l3970_397060

theorem inequality_equivalence (x : ℝ) : (x - 3) / (x^2 + 2*x + 7) ≥ 0 ↔ x ≥ 3 := by
  sorry

end inequality_equivalence_l3970_397060


namespace point_on_linear_function_l3970_397029

theorem point_on_linear_function (m : ℝ) : 
  (3 : ℝ) = 2 * m + 1 → m = 1 := by
  sorry

end point_on_linear_function_l3970_397029


namespace lcm_factor_proof_l3970_397098

theorem lcm_factor_proof (A B : ℕ+) (h_hcf : Nat.gcd A B = 25) 
  (h_lcm : ∃ X : ℕ+, Nat.lcm A B = 25 * X * 14) (h_A : A = 350) (h_order : A > B) : 
  ∃ X : ℕ+, Nat.lcm A B = 25 * X * 14 ∧ X = 1 := by
  sorry

end lcm_factor_proof_l3970_397098


namespace sin_2alpha_value_l3970_397066

theorem sin_2alpha_value (α : Real) 
  (h1 : α > 0 ∧ α < Real.pi) 
  (h2 : 3 * Real.cos (2 * α) - 4 * Real.cos α + 1 = 0) : 
  Real.sin (2 * α) = -4 * Real.sqrt 2 / 9 := by
  sorry

end sin_2alpha_value_l3970_397066


namespace existence_of_rationals_l3970_397000

theorem existence_of_rationals (a b c d m n : ℤ) (ε : ℝ) 
  (h : a * d - b * c ≠ 0) (hε : ε > 0) :
  ∃ x y : ℚ, 0 < |a * x + b * y - m| ∧ |a * x + b * y - m| < ε ∧
           0 < |c * x + d * y - n| ∧ |c * x + d * y - n| < ε :=
by sorry


end existence_of_rationals_l3970_397000


namespace circle_reduction_l3970_397045

/-- Represents a letter in the circle -/
inductive Letter
| A
| B

/-- Represents the circle of letters -/
def Circle := List Letter

/-- Represents a transformation rule -/
inductive Transform
| ABA_to_B
| B_to_ABA
| VAV_to_A
| A_to_VAV

/-- Applies a single transformation to the circle -/
def applyTransform (c : Circle) (t : Transform) : Circle :=
  sorry

/-- Checks if the circle contains exactly one letter -/
def isSingleLetter (c : Circle) : Bool :=
  sorry

/-- The main theorem to prove -/
theorem circle_reduction (initial : Circle) :
  initial.length = 41 →
  ∃ (final : Circle), (∃ (transforms : List Transform),
    (List.foldl applyTransform initial transforms = final) ∧
    isSingleLetter final) :=
  sorry

end circle_reduction_l3970_397045


namespace average_of_pqrs_l3970_397042

theorem average_of_pqrs (p q r s : ℝ) (h : (5 / 4) * (p + q + r + s) = 20) :
  (p + q + r + s) / 4 = 4 := by
  sorry

end average_of_pqrs_l3970_397042


namespace line_through_points_l3970_397023

/-- Given a line x = 6y + 5 passing through points (m, n) and (m + 2, n + p), prove p = 1/3 -/
theorem line_through_points (m n p : ℝ) : 
  (m = 6 * n + 5) ∧ (m + 2 = 6 * (n + p) + 5) → p = 1/3 := by
  sorry

end line_through_points_l3970_397023


namespace trail_mix_packs_needed_l3970_397074

def total_people : ℕ := 18
def pouches_per_pack : ℕ := 6

theorem trail_mix_packs_needed :
  ∃ (packs : ℕ), packs * pouches_per_pack ≥ total_people ∧
  ∀ (x : ℕ), x * pouches_per_pack ≥ total_people → x ≥ packs :=
by sorry

end trail_mix_packs_needed_l3970_397074


namespace special_square_area_l3970_397022

/-- A square with special points and segments -/
structure SpecialSquare where
  -- The side length of the square
  side : ℝ
  -- The distance BS
  bs : ℝ
  -- The distance PS
  ps : ℝ
  -- Assumption that BS = 8
  bs_eq : bs = 8
  -- Assumption that PS = 9
  ps_eq : ps = 9
  -- Assumption that BP and DQ intersect perpendicularly
  perpendicular : True

/-- The area of a SpecialSquare is 136 -/
theorem special_square_area (sq : SpecialSquare) : sq.side ^ 2 = 136 := by
  sorry

#check special_square_area

end special_square_area_l3970_397022


namespace last_remaining_100_l3970_397053

def last_remaining (n : ℕ) : ℕ :=
  if n ≤ 1 then n else
  let m := n / 2
  2 * (if m % 2 = 0 then last_remaining m else m + 1 - last_remaining m)

theorem last_remaining_100 : last_remaining 100 = 64 := by
  sorry

end last_remaining_100_l3970_397053


namespace rob_pennies_l3970_397008

/-- The number of pennies Rob has -/
def num_pennies : ℕ := 12

/-- The number of quarters Rob has -/
def num_quarters : ℕ := 7

/-- The number of dimes Rob has -/
def num_dimes : ℕ := 3

/-- The number of nickels Rob has -/
def num_nickels : ℕ := 5

/-- The total amount Rob has in cents -/
def total_amount : ℕ := 242

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The value of a penny in cents -/
def penny_value : ℕ := 1

theorem rob_pennies :
  num_quarters * quarter_value + num_dimes * dime_value + num_nickels * nickel_value + num_pennies * penny_value = total_amount :=
by sorry

end rob_pennies_l3970_397008


namespace remainder_104_pow_2006_mod_29_l3970_397076

theorem remainder_104_pow_2006_mod_29 : 104^2006 % 29 = 28 := by
  sorry

end remainder_104_pow_2006_mod_29_l3970_397076


namespace work_completion_theorem_l3970_397027

/-- Calculates the number of men needed to complete a job in a given number of days,
    given the initial number of men and days required. -/
def men_needed (initial_men : ℕ) (initial_days : ℕ) (new_days : ℕ) : ℕ :=
  (initial_men * initial_days) / new_days

theorem work_completion_theorem (initial_men : ℕ) (initial_days : ℕ) (new_days : ℕ) :
  initial_men = 25 → initial_days = 96 → new_days = 60 →
  men_needed initial_men initial_days new_days = 40 := by
  sorry

#eval men_needed 25 96 60

end work_completion_theorem_l3970_397027


namespace cubic_equation_product_l3970_397040

theorem cubic_equation_product (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) 
  (h₁ : x₁^3 - 3*x₁*y₁^2 = 2006) (h₂ : y₁^3 - 3*x₁^2*y₁ = 2007)
  (h₃ : x₂^3 - 3*x₂*y₂^2 = 2006) (h₄ : y₂^3 - 3*x₂^2*y₂ = 2007)
  (h₅ : x₃^3 - 3*x₃*y₃^2 = 2006) (h₆ : y₃^3 - 3*x₃^2*y₃ = 2007) :
  (1 - x₁/y₁) * (1 - x₂/y₂) * (1 - x₃/y₃) = 1/1003 := by
sorry

end cubic_equation_product_l3970_397040


namespace tom_payment_tom_paid_1908_l3970_397039

/-- Calculates the total amount Tom paid to the shopkeeper after discount -/
theorem tom_payment (apple_kg : ℕ) (apple_rate : ℕ) (mango_kg : ℕ) (mango_rate : ℕ) 
                    (grape_kg : ℕ) (grape_rate : ℕ) (discount_percent : ℕ) : ℕ :=
  let total_cost := apple_kg * apple_rate + mango_kg * mango_rate + grape_kg * grape_rate
  let discount := total_cost * discount_percent / 100
  total_cost - discount

/-- Proves that Tom paid 1908 to the shopkeeper -/
theorem tom_paid_1908 : 
  tom_payment 8 70 9 90 5 150 10 = 1908 := by
  sorry

end tom_payment_tom_paid_1908_l3970_397039


namespace x_range_for_quadratic_inequality_l3970_397002

theorem x_range_for_quadratic_inequality (x : ℝ) :
  (∀ a : ℝ, a ∈ Set.Icc (-1) 1 → x^2 + (a - 4)*x + 4 - 2*a > 0) →
  x ∈ Set.Iio 1 ∪ Set.Ioi 3 :=
by sorry

end x_range_for_quadratic_inequality_l3970_397002


namespace quadratic_equation_root_l3970_397085

theorem quadratic_equation_root (m : ℝ) : 
  ((-1 : ℝ)^2 + m * (-1) - 4 = 0) → 
  ∃ (x : ℝ), x ≠ -1 ∧ x^2 + m*x - 4 = 0 ∧ x = 4 := by
sorry

end quadratic_equation_root_l3970_397085


namespace jimin_candies_l3970_397028

/-- The number of candies Jimin gave to Yuna -/
def candies_given : ℕ := 25

/-- The number of candies left over -/
def candies_left : ℕ := 13

/-- The total number of candies Jimin had at the start -/
def total_candies : ℕ := candies_given + candies_left

theorem jimin_candies : total_candies = 38 := by
  sorry

end jimin_candies_l3970_397028


namespace gcd_360_210_l3970_397071

theorem gcd_360_210 : Nat.gcd 360 210 = 30 := by
  sorry

end gcd_360_210_l3970_397071


namespace tyler_sanctuary_species_l3970_397018

/-- The number of pairs of birds per species in Tyler's sanctuary -/
def pairs_per_species : ℕ := 7

/-- The total number of pairs of birds in Tyler's sanctuary -/
def total_pairs : ℕ := 203

/-- The number of endangered bird species in Tyler's sanctuary -/
def num_species : ℕ := total_pairs / pairs_per_species

theorem tyler_sanctuary_species :
  num_species = 29 :=
sorry

end tyler_sanctuary_species_l3970_397018
