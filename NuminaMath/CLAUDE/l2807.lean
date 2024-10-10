import Mathlib

namespace balloon_tank_capacity_l2807_280708

theorem balloon_tank_capacity 
  (num_balloons : ℕ) 
  (air_per_balloon : ℕ) 
  (num_tanks : ℕ) 
  (h1 : num_balloons = 1000)
  (h2 : air_per_balloon = 10)
  (h3 : num_tanks = 20) :
  (num_balloons * air_per_balloon) / num_tanks = 500 := by
  sorry

end balloon_tank_capacity_l2807_280708


namespace number_satisfies_equation_l2807_280757

theorem number_satisfies_equation : ∃ x : ℝ, (45 - 3 * x^2 = 12) ∧ (x = Real.sqrt 11 ∨ x = -Real.sqrt 11) := by
  sorry

end number_satisfies_equation_l2807_280757


namespace product_sum_equality_l2807_280795

theorem product_sum_equality (x y z : ℕ) 
  (h1 : 2014 + y = 2015 + x)
  (h2 : 2015 + x = 2016 + z)
  (h3 : y * x * z = 504) :
  y * x + x * z = 128 := by
sorry

end product_sum_equality_l2807_280795


namespace fraction_simplification_l2807_280744

theorem fraction_simplification (m : ℝ) (h : m ≠ 3 ∧ m ≠ -3) :
  3 / (m^2 - 9) + m / (9 - m^2) = -1 / (m + 3) :=
by sorry

end fraction_simplification_l2807_280744


namespace smallest_valid_amount_l2807_280764

/-- Represents the number of bags -/
def num_bags : List Nat := [8, 7, 6]

/-- Represents the types of currency -/
inductive Currency
| Dollar
| HalfDollar
| QuarterDollar

/-- Checks if a given amount can be equally distributed into the specified number of bags for all currency types -/
def is_valid_distribution (amount : Nat) (bags : Nat) : Prop :=
  ∀ c : Currency, ∃ n : Nat, n * bags = amount

/-- The main theorem stating the smallest valid amount -/
theorem smallest_valid_amount :
  (∀ bags ∈ num_bags, is_valid_distribution 294 bags) ∧
  (∀ amount < 294, ¬(∀ bags ∈ num_bags, is_valid_distribution amount bags)) :=
sorry

end smallest_valid_amount_l2807_280764


namespace intersection_implies_m_value_l2807_280787

def A (m : ℝ) : Set ℝ := {m - 1, -3}
def B (m : ℝ) : Set ℝ := {2*m - 1, m - 3}

theorem intersection_implies_m_value :
  ∀ m : ℝ, A m ∩ B m = {-3} → m = -1 := by
  sorry

end intersection_implies_m_value_l2807_280787


namespace student_subtraction_problem_l2807_280724

theorem student_subtraction_problem (x y : ℤ) : 
  x = 40 → 7 * x - y = 130 → y = 150 := by sorry

end student_subtraction_problem_l2807_280724


namespace canoe_upstream_speed_l2807_280794

/-- The speed of a canoe rowing upstream, given its downstream speed and the stream speed -/
theorem canoe_upstream_speed (downstream_speed stream_speed : ℝ) :
  downstream_speed = 12 →
  stream_speed = 4 →
  downstream_speed - 2 * stream_speed = 4 := by
  sorry

end canoe_upstream_speed_l2807_280794


namespace trapezoid_perimeter_l2807_280728

/-- A trapezoid with given dimensions -/
structure Trapezoid where
  base1 : ℝ
  base2 : ℝ
  side1 : ℝ
  side2 : ℝ

/-- The perimeter of a trapezoid -/
def perimeter (t : Trapezoid) : ℝ :=
  t.base1 + t.base2 + t.side1 + t.side2

/-- Theorem: The perimeter of the specific trapezoid is 42 -/
theorem trapezoid_perimeter : 
  let t : Trapezoid := { base1 := 10, base2 := 14, side1 := 9, side2 := 9 }
  perimeter t = 42 := by sorry

end trapezoid_perimeter_l2807_280728


namespace james_chores_total_time_l2807_280736

/-- Given James' chore schedule, prove that the total time spent is 16.5 hours -/
theorem james_chores_total_time :
  let vacuuming_time : ℝ := 3
  let cleaning_time : ℝ := 3 * vacuuming_time
  let laundry_time : ℝ := cleaning_time / 2
  vacuuming_time + cleaning_time + laundry_time = 16.5 := by
  sorry

end james_chores_total_time_l2807_280736


namespace nickel_probability_l2807_280776

/-- Represents the types of coins in the jar -/
inductive Coin
| Dime
| Nickel
| Penny

/-- The value of each coin type in cents -/
def coin_value : Coin → ℕ
| Coin.Dime => 10
| Coin.Nickel => 5
| Coin.Penny => 1

/-- The total value of each coin type in cents -/
def total_value : Coin → ℕ
| Coin.Dime => 500
| Coin.Nickel => 300
| Coin.Penny => 200

/-- The number of coins of each type -/
def coin_count (c : Coin) : ℕ := total_value c / coin_value c

/-- The total number of coins in the jar -/
def total_coins : ℕ := coin_count Coin.Dime + coin_count Coin.Nickel + coin_count Coin.Penny

/-- The probability of randomly choosing a nickel from the jar -/
theorem nickel_probability : 
  (coin_count Coin.Nickel : ℚ) / total_coins = 6 / 31 := by sorry

end nickel_probability_l2807_280776


namespace quadratic_always_positive_l2807_280760

theorem quadratic_always_positive (k : ℝ) :
  (∀ x : ℝ, x^2 - (k - 2)*x - k + 4 > 0) ↔ k > -2*Real.sqrt 3 ∧ k < 2*Real.sqrt 3 :=
sorry

end quadratic_always_positive_l2807_280760


namespace chess_tournament_orders_l2807_280701

/-- Represents the number of players in the tournament -/
def num_players : ℕ := 4

/-- Represents the number of possible outcomes for each match -/
def outcomes_per_match : ℕ := 2

/-- Represents the number of matches in the tournament -/
def num_matches : ℕ := num_players - 1

/-- Calculates the total number of possible finishing orders -/
def total_possible_orders : ℕ := outcomes_per_match ^ num_matches

/-- Theorem stating that there are exactly 8 different possible finishing orders -/
theorem chess_tournament_orders : total_possible_orders = 8 := by
  sorry

end chess_tournament_orders_l2807_280701


namespace correct_result_l2807_280737

theorem correct_result (x : ℤ) (h : x - 27 + 19 = 84) : x - 19 + 27 = 100 := by
  sorry

end correct_result_l2807_280737


namespace suyeong_run_distance_l2807_280796

/-- The circumference of the playground in meters -/
def playground_circumference : ℝ := 242.7

/-- The number of laps Suyeong ran -/
def laps_run : ℕ := 5

/-- The total distance Suyeong ran in meters -/
def total_distance : ℝ := playground_circumference * (laps_run : ℝ)

theorem suyeong_run_distance : total_distance = 1213.5 := by
  sorry

end suyeong_run_distance_l2807_280796


namespace rotation_90_degrees_l2807_280790

theorem rotation_90_degrees (z : ℂ) : z = -4 - I → z * I = 1 - 4*I := by
  sorry

end rotation_90_degrees_l2807_280790


namespace complement_of_A_l2807_280703

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | x ≤ 0 ∨ x ≥ 1}

theorem complement_of_A : Set.compl A = Set.Ioo 0 1 := by sorry

end complement_of_A_l2807_280703


namespace sum_of_first_100_factorials_mod_100_l2807_280719

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_of_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem sum_of_first_100_factorials_mod_100 :
  sum_of_factorials 100 % 100 = 13 := by sorry

end sum_of_first_100_factorials_mod_100_l2807_280719


namespace lantern_probability_l2807_280721

def total_large_lanterns : ℕ := 360
def total_small_lanterns : ℕ := 1200

def large_with_two_small (x : ℕ) : Prop := 
  x * 2 + (total_large_lanterns - x) * 4 = total_small_lanterns

def large_with_four_small (x : ℕ) : ℕ := total_large_lanterns - x

def total_combinations : ℕ := total_large_lanterns.choose 2

def favorable_outcomes (x : ℕ) : ℕ := 
  (large_with_four_small x).choose 2 + (large_with_four_small x).choose 1 * x.choose 1

theorem lantern_probability (x : ℕ) (h : large_with_two_small x) : 
  (favorable_outcomes x : ℚ) / total_combinations = 958 / 1077 := by sorry

end lantern_probability_l2807_280721


namespace race_speed_ratio_l2807_280775

theorem race_speed_ratio (va vb : ℝ) (h : va > 0 ∧ vb > 0) :
  (1 / va = (1 - 0.09523809523809523) / vb) → (va / vb = 21 / 19) := by
  sorry

end race_speed_ratio_l2807_280775


namespace betty_order_hair_color_cost_l2807_280727

/-- Given Betty's order details, prove the cost of each hair color. -/
theorem betty_order_hair_color_cost 
  (total_items : ℕ) 
  (slipper_count : ℕ) 
  (slipper_cost : ℚ) 
  (lipstick_count : ℕ) 
  (lipstick_cost : ℚ) 
  (hair_color_count : ℕ) 
  (total_paid : ℚ) 
  (h_total_items : total_items = 18) 
  (h_slipper_count : slipper_count = 6) 
  (h_slipper_cost : slipper_cost = 5/2) 
  (h_lipstick_count : lipstick_count = 4) 
  (h_lipstick_cost : lipstick_cost = 5/4) 
  (h_hair_color_count : hair_color_count = 8) 
  (h_total_paid : total_paid = 44) 
  (h_item_sum : total_items = slipper_count + lipstick_count + hair_color_count) : 
  (total_paid - (slipper_count * slipper_cost + lipstick_count * lipstick_cost)) / hair_color_count = 3 := by
  sorry


end betty_order_hair_color_cost_l2807_280727


namespace jerrys_age_l2807_280766

theorem jerrys_age (mickey_age jerry_age : ℕ) : 
  mickey_age = 2 * jerry_age - 4 →
  mickey_age = 22 →
  jerry_age = 13 := by
sorry

end jerrys_age_l2807_280766


namespace prime_power_implies_prime_n_l2807_280732

theorem prime_power_implies_prime_n (n : ℕ) (p : ℕ) (k : ℕ) :
  (∃ (p : ℕ), Prime p ∧ ∃ (k : ℕ), 3^n - 2^n = p^k) →
  Prime n :=
by sorry

end prime_power_implies_prime_n_l2807_280732


namespace seating_arrangements_count_l2807_280771

/-- Represents the number of seats in each row -/
def seats : Fin 2 → Nat
  | 0 => 6  -- front row
  | 1 => 7  -- back row

/-- Calculates the number of ways to arrange 2 people in two rows of seats
    such that they are not sitting next to each other -/
def seating_arrangements : Nat :=
  let different_rows := seats 0 * seats 1 * 2
  let front_row := 2 * 4 + 4 * 3
  let back_row := 2 * 5 + 5 * 4
  different_rows + front_row + back_row

/-- Theorem stating that the number of seating arrangements is 134 -/
theorem seating_arrangements_count : seating_arrangements = 134 := by
  sorry

end seating_arrangements_count_l2807_280771


namespace rectangular_plot_poles_l2807_280738

/-- The number of poles needed to enclose a rectangular plot -/
def num_poles (length width long_spacing short_spacing : ℕ) : ℕ :=
  2 * ((length / long_spacing - 1) + (width / short_spacing - 1))

/-- Theorem stating the number of poles needed for the given rectangular plot -/
theorem rectangular_plot_poles : 
  num_poles 120 80 5 4 = 84 := by
  sorry

end rectangular_plot_poles_l2807_280738


namespace parabola_transformation_l2807_280769

/-- The original parabola function -/
def original_parabola (x : ℝ) : ℝ := x^2 - 1

/-- The transformed parabola function -/
def transformed_parabola (x : ℝ) : ℝ := (x - 1)^2 + 1

/-- Theorem stating that the transformation of the original parabola
    by shifting 2 units up and 1 unit right results in the transformed parabola -/
theorem parabola_transformation :
  ∀ x : ℝ, transformed_parabola x = original_parabola (x - 1) + 2 :=
by sorry

end parabola_transformation_l2807_280769


namespace third_circle_radius_l2807_280782

theorem third_circle_radius (r₁ r₂ r₃ : ℝ) (h₁ : r₁ = 21) (h₂ : r₂ = 35) (h₃ : r₃ = 28) :
  π * r₃^2 = π * r₂^2 - π * r₁^2 := by
  sorry

#check third_circle_radius

end third_circle_radius_l2807_280782


namespace arithmetic_sequence_property_l2807_280750

/-- An arithmetic sequence {a_n} satisfying the given conditions -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ (d : ℝ), ∀ (n : ℕ), a (n + 1) = a n + d

theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h_arith : ArithmeticSequence a) 
  (h_sum : a 4 + a 6 + a 8 + a 10 + a 12 = 60) : 
  a 7 - (1/3) * a 5 = 8 := by
  sorry

end arithmetic_sequence_property_l2807_280750


namespace coin_inverted_after_two_rolls_l2807_280731

/-- Represents the orientation of a coin -/
inductive CoinOrientation
  | Upright
  | Inverted

/-- Represents a single roll of the coin -/
def single_roll_rotation : ℕ := 270

/-- The total rotation after two equal rolls -/
def total_rotation : ℕ := 2 * single_roll_rotation

/-- Function to determine the final orientation after a given rotation -/
def final_orientation (initial : CoinOrientation) (rotation : ℕ) : CoinOrientation :=
  if rotation % 360 = 180 then
    match initial with
    | CoinOrientation.Upright => CoinOrientation.Inverted
    | CoinOrientation.Inverted => CoinOrientation.Upright
  else initial

/-- Theorem stating that after two equal rolls, the coin will be inverted -/
theorem coin_inverted_after_two_rolls (initial : CoinOrientation) :
  final_orientation initial total_rotation = CoinOrientation.Inverted :=
sorry

end coin_inverted_after_two_rolls_l2807_280731


namespace gcd_of_390_455_546_l2807_280718

theorem gcd_of_390_455_546 : Nat.gcd 390 (Nat.gcd 455 546) = 13 := by
  sorry

end gcd_of_390_455_546_l2807_280718


namespace min_triangles_for_100gon_l2807_280709

/-- A convex polygon with 100 sides -/
def ConvexPolygon100 : Type := Unit

/-- The number of triangles needed to represent a convex 100-gon as their intersection -/
def num_triangles (p : ConvexPolygon100) : ℕ := sorry

/-- The smallest number of triangles needed to represent any convex 100-gon as their intersection -/
def min_num_triangles : ℕ := sorry

theorem min_triangles_for_100gon :
  min_num_triangles = 50 := by sorry

end min_triangles_for_100gon_l2807_280709


namespace rice_distribution_l2807_280759

theorem rice_distribution (total_pounds : ℚ) (num_containers : ℕ) (ounces_per_pound : ℕ) :
  total_pounds = 35 / 2 →
  num_containers = 4 →
  ounces_per_pound = 16 →
  (total_pounds * ounces_per_pound) / num_containers = 70 := by
  sorry

end rice_distribution_l2807_280759


namespace quadratic_rewrite_l2807_280793

theorem quadratic_rewrite (g h j : ℤ) :
  (∀ x : ℝ, 4 * x^2 - 16 * x - 21 = (g * x + h)^2 + j) →
  g * h = -8 := by
sorry

end quadratic_rewrite_l2807_280793


namespace nonagon_diagonal_intersection_probability_nonagon_diagonal_intersection_probability_proof_l2807_280789

/-- The probability of two randomly chosen diagonals intersecting inside a regular nonagon -/
theorem nonagon_diagonal_intersection_probability : ℚ :=
  14 / 39

/-- The number of sides in a nonagon -/
def nonagon_sides : ℕ := 9

/-- The number of diagonals in a nonagon -/
def nonagon_diagonals : ℕ := (nonagon_sides.choose 2) - nonagon_sides

/-- The number of ways to choose 2 diagonals from the total number of diagonals -/
def diagonal_pairs : ℕ := nonagon_diagonals.choose 2

/-- The number of ways to choose 4 points from the nonagon vertices -/
def intersecting_diagonal_sets : ℕ := nonagon_sides.choose 4

theorem nonagon_diagonal_intersection_probability_proof :
  (intersecting_diagonal_sets : ℚ) / diagonal_pairs = nonagon_diagonal_intersection_probability :=
sorry

end nonagon_diagonal_intersection_probability_nonagon_diagonal_intersection_probability_proof_l2807_280789


namespace range_of_f_l2807_280751

-- Define the function
def f (x : ℝ) : ℝ := x^2 - 2*x - 3

-- Define the domain
def domain : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}

-- Theorem statement
theorem range_of_f :
  {y | ∃ x ∈ domain, f x = y} = {y | -4 ≤ y ∧ y ≤ 5} := by sorry

end range_of_f_l2807_280751


namespace inequality_proof_l2807_280788

theorem inequality_proof (a b c : ℝ) :
  a = 31/32 →
  b = Real.cos (1/4) →
  c = 4 * Real.sin (1/4) →
  c > b ∧ b > a := by
sorry

end inequality_proof_l2807_280788


namespace million_to_scientific_notation_two_point_684_million_scientific_notation_l2807_280767

theorem million_to_scientific_notation (n : ℝ) : 
  n * 1000000 = n * (10 : ℝ) ^ 6 := by sorry

-- Define 2.684 million
def two_point_684_million : ℝ := 2.684 * 1000000

-- Theorem to prove
theorem two_point_684_million_scientific_notation : 
  two_point_684_million = 2.684 * (10 : ℝ) ^ 6 := by sorry

end million_to_scientific_notation_two_point_684_million_scientific_notation_l2807_280767


namespace jessica_candy_distribution_l2807_280743

/-- The number of candies Jessica must remove to distribute them equally among her friends -/
def candies_to_remove (total : Nat) (friends : Nat) : Nat :=
  total % friends

theorem jessica_candy_distribution :
  candies_to_remove 30 4 = 2 := by
  sorry

end jessica_candy_distribution_l2807_280743


namespace parabola_x_axis_intersection_l2807_280717

/-- The parabola defined by y = x^2 - 2x + 1 -/
def parabola (x : ℝ) : ℝ := x^2 - 2*x + 1

/-- Theorem stating that (1, 0) is the only intersection point of the parabola and the x-axis -/
theorem parabola_x_axis_intersection :
  ∃! x : ℝ, parabola x = 0 ∧ x = 1 := by sorry

end parabola_x_axis_intersection_l2807_280717


namespace souvenir_store_problem_l2807_280783

/-- Souvenir store problem -/
theorem souvenir_store_problem 
  (total_souvenirs : ℕ)
  (cost_40A_30B cost_10A_50B : ℕ)
  (sell_price_A sell_price_B : ℕ)
  (m : ℚ)
  (h_total : total_souvenirs = 300)
  (h_cost1 : cost_40A_30B = 5000)
  (h_cost2 : cost_10A_50B = 3800)
  (h_sell_A : sell_price_A = 120)
  (h_sell_B : sell_price_B = 80)
  (h_m_range : 4 < m ∧ m < 8) :
  ∃ (cost_A cost_B max_profit : ℕ) (reduced_profit : ℚ),
    cost_A = 80 ∧ 
    cost_B = 60 ∧
    max_profit = 7500 ∧
    reduced_profit = 5720 ∧
    ∀ (a : ℕ), 
      a ≤ total_souvenirs →
      (total_souvenirs - a) ≥ 3 * a →
      (sell_price_A - cost_A) * a + (sell_price_B - cost_B) * (total_souvenirs - a) ≥ 7400 →
      (sell_price_A - cost_A) * a + (sell_price_B - cost_B) * (total_souvenirs - a) ≤ max_profit ∧
      ((sell_price_A - 5 * m - cost_A) * 70 + (sell_price_B - cost_B) * (total_souvenirs - 70) : ℚ) = reduced_profit →
      m = 4.8 := by
  sorry

end souvenir_store_problem_l2807_280783


namespace motion_analysis_l2807_280763

-- Define the motion law
def s (t : ℝ) : ℝ := 4 * t + t^3

-- Define velocity as the derivative of s
noncomputable def v (t : ℝ) : ℝ := deriv s t

-- Define acceleration as the derivative of v
noncomputable def a (t : ℝ) : ℝ := deriv v t

-- Theorem statement
theorem motion_analysis :
  (∀ t, v t = 4 + 3 * t^2) ∧
  (∀ t, a t = 6 * t) ∧
  (v 0 = 4 ∧ a 0 = 0) ∧
  (v 1 = 7 ∧ a 1 = 6) ∧
  (v 2 = 16 ∧ a 2 = 12) := by sorry

end motion_analysis_l2807_280763


namespace least_common_multiple_of_denominators_l2807_280730

theorem least_common_multiple_of_denominators : Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 9)))))) = 2520 := by
  sorry

end least_common_multiple_of_denominators_l2807_280730


namespace april_spending_l2807_280785

def initial_savings : ℕ := 11000
def february_percentage : ℚ := 20 / 100
def march_percentage : ℚ := 40 / 100
def remaining_savings : ℕ := 2900

theorem april_spending :
  let february_spending := (february_percentage * initial_savings).floor
  let march_spending := (march_percentage * initial_savings).floor
  let total_spent := initial_savings - remaining_savings
  total_spent - february_spending - march_spending = 1500 := by sorry

end april_spending_l2807_280785


namespace max_value_of_z_l2807_280716

-- Define the variables and the objective function
variables (x y : ℝ)
def z : ℝ → ℝ → ℝ := λ x y => 2 * x + y

-- Define the constraints
def constraint1 (x y : ℝ) : Prop := x + 2 * y ≤ 2
def constraint2 (x y : ℝ) : Prop := x + y ≥ 0
def constraint3 (x : ℝ) : Prop := x ≤ 4

-- Theorem statement
theorem max_value_of_z :
  ∀ x y : ℝ, constraint1 x y → constraint2 x y → constraint3 x →
  z x y ≤ 11 ∧ ∃ x₀ y₀ : ℝ, constraint1 x₀ y₀ ∧ constraint2 x₀ y₀ ∧ constraint3 x₀ ∧ z x₀ y₀ = 11 :=
sorry

end max_value_of_z_l2807_280716


namespace pet_store_cages_l2807_280756

def bird_cages (total_birds : ℕ) (parrots_per_cage : ℕ) (parakeets_per_cage : ℕ) : ℕ :=
  total_birds / (parrots_per_cage + parakeets_per_cage)

theorem pet_store_cages :
  bird_cages 36 2 2 = 9 :=
by sorry

end pet_store_cages_l2807_280756


namespace job_completion_time_l2807_280797

theorem job_completion_time (x : ℝ) : 
  x > 0 →  -- A's completion time is positive
  8 * (1 / x + 1 / 20) = 1 - 0.06666666666666665 →  -- Condition after 8 days of working together
  x = 15 := by
sorry

end job_completion_time_l2807_280797


namespace complex_equation_solutions_l2807_280755

theorem complex_equation_solutions :
  ∃ (S : Set ℂ), S = {z : ℂ | z^6 + 6*I = 0} ∧
  S = {I, -I} ∪ {z : ℂ | ∃ k : ℕ, 0 ≤ k ∧ k < 4 ∧ z = (-6*I)^(1/6) * (Complex.exp (2*π*I*(k:ℝ)/4))} :=
by sorry

end complex_equation_solutions_l2807_280755


namespace valeria_apartment_number_l2807_280792

def is_not_multiple_of_5 (n : ℕ) : Prop := n % 5 ≠ 0

def is_odd (n : ℕ) : Prop := n % 2 = 1

def sum_of_digits_less_than_8 (n : ℕ) : Prop :=
  (n / 10 + n % 10) < 8

def units_digit_is_6 (n : ℕ) : Prop := n % 10 = 6

theorem valeria_apartment_number (n : ℕ) :
  n ≥ 10 ∧ n < 100 →
  (is_not_multiple_of_5 n ∧ is_odd n ∧ units_digit_is_6 n) ∨
  (is_not_multiple_of_5 n ∧ is_odd n ∧ sum_of_digits_less_than_8 n) ∨
  (is_not_multiple_of_5 n ∧ sum_of_digits_less_than_8 n ∧ units_digit_is_6 n) ∨
  (is_odd n ∧ sum_of_digits_less_than_8 n ∧ units_digit_is_6 n) →
  units_digit_is_6 n :=
by sorry

end valeria_apartment_number_l2807_280792


namespace diana_wins_probability_l2807_280746

def standard_die := Finset.range 6

def favorable_outcomes : Finset (ℕ × ℕ) :=
  (standard_die.product standard_die).filter (fun (d, a) => d > a)

theorem diana_wins_probability :
  (favorable_outcomes.card : ℚ) / (standard_die.card * standard_die.card) = 5 / 12 := by
  sorry

end diana_wins_probability_l2807_280746


namespace man_and_son_work_time_l2807_280770

/-- Given a task that takes a man 4 days and his son 12 days to complete individually, 
    prove that they can complete the task together in 3 days. -/
theorem man_and_son_work_time (task : ℝ) (man_rate son_rate combined_rate : ℝ) : 
  task > 0 ∧ 
  man_rate = task / 4 ∧ 
  son_rate = task / 12 ∧ 
  combined_rate = man_rate + son_rate → 
  task / combined_rate = 3 := by
sorry

end man_and_son_work_time_l2807_280770


namespace intersection_implies_a_value_l2807_280781

def A (a : ℝ) : Set ℝ := {2, a}
def B (a : ℝ) : Set ℝ := {-1, a^2 - 2}

theorem intersection_implies_a_value (a : ℝ) :
  (A a ∩ B a).Nonempty → a = -2 := by
  sorry

end intersection_implies_a_value_l2807_280781


namespace binary_multiplication_division_l2807_280749

/-- Represents a binary number as a list of bits (0 or 1) -/
def BinaryNumber := List Nat

/-- Converts a decimal number to its binary representation -/
def toBinary (n : Nat) : BinaryNumber :=
  sorry

/-- Converts a binary number to its decimal representation -/
def toDecimal (b : BinaryNumber) : Nat :=
  sorry

/-- Multiplies two binary numbers -/
def binaryMultiply (a b : BinaryNumber) : BinaryNumber :=
  sorry

/-- Divides a binary number by 2 -/
def binaryDivideByTwo (b : BinaryNumber) : BinaryNumber :=
  sorry

theorem binary_multiplication_division :
  let a := [1, 0, 1, 1, 0]  -- 10110₂
  let b := [1, 0, 1, 0, 0]  -- 10100₂
  let result := [1, 1, 0, 1, 1, 1, 0, 0]  -- 11011100₂
  binaryDivideByTwo (binaryMultiply a b) = result :=
sorry

end binary_multiplication_division_l2807_280749


namespace regular_pay_limit_l2807_280740

theorem regular_pay_limit (regular_rate : ℝ) (overtime_hours : ℝ) (total_pay : ℝ) 
  (h1 : regular_rate = 3)
  (h2 : overtime_hours = 13)
  (h3 : total_pay = 198) :
  ∃ (regular_hours : ℝ),
    regular_hours * regular_rate + overtime_hours * (2 * regular_rate) = total_pay ∧
    regular_hours = 40 := by
  sorry

end regular_pay_limit_l2807_280740


namespace largest_inexpressible_number_l2807_280752

/-- A function that checks if a natural number can be expressed as a non-negative linear combination of 5 and 6 -/
def isExpressible (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = 5 * a + 6 * b

/-- The theorem stating that 19 is the largest natural number not exceeding 50 that cannot be expressed as a non-negative linear combination of 5 and 6 -/
theorem largest_inexpressible_number :
  (∀ (k : ℕ), k > 19 ∧ k ≤ 50 → isExpressible k) ∧
  ¬isExpressible 19 :=
sorry

end largest_inexpressible_number_l2807_280752


namespace megan_candy_from_sister_l2807_280780

/-- Calculates the number of candy pieces Megan received from her older sister. -/
def candy_from_sister (candy_from_neighbors : ℝ) (candy_eaten_per_day : ℝ) (days_lasted : ℝ) : ℝ :=
  candy_eaten_per_day * days_lasted - candy_from_neighbors

/-- Proves that Megan received 5.0 pieces of candy from her older sister. -/
theorem megan_candy_from_sister :
  candy_from_sister 11.0 8.0 2.0 = 5.0 := by
  sorry

end megan_candy_from_sister_l2807_280780


namespace rectangle_circle_union_area_l2807_280702

/-- The area of the union of a rectangle and a circle with specified dimensions -/
theorem rectangle_circle_union_area :
  let rectangle_width : ℝ := 12
  let rectangle_height : ℝ := 15
  let circle_radius : ℝ := 15
  let rectangle_area : ℝ := rectangle_width * rectangle_height
  let circle_area : ℝ := π * circle_radius^2
  let overlap_area : ℝ := (1/4) * circle_area
  let union_area : ℝ := rectangle_area + circle_area - overlap_area
  union_area = 180 + 168.75 * π := by
  sorry

end rectangle_circle_union_area_l2807_280702


namespace union_A_B_when_a_is_one_range_of_a_for_necessary_not_sufficient_l2807_280773

-- Define set A
def A (a : ℝ) : Set ℝ := {x | (x - a) / (x - 3*a) < 0}

-- Define set B
def B : Set ℝ := {x | x^2 - 5*x + 6 < 0}

-- Theorem for part (1)
theorem union_A_B_when_a_is_one : 
  A 1 ∪ B = {x | 1 < x ∧ x < 3} := by sorry

-- Theorem for part (2)
theorem range_of_a_for_necessary_not_sufficient : 
  {a : ℝ | ∀ x, x ∈ B → x ∈ A a ∧ ∃ y, y ∈ A a ∧ y ∉ B} = {a | 1 ≤ a ∧ a ≤ 2} := by sorry

end union_A_B_when_a_is_one_range_of_a_for_necessary_not_sufficient_l2807_280773


namespace f_geq_2_iff_max_m_value_l2807_280748

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + |x - 2|

-- Theorem for the first part of the problem
theorem f_geq_2_iff (x : ℝ) :
  f x ≥ 2 ↔ x ≤ 1/2 ∨ x ≥ 5/2 :=
sorry

-- Theorem for the second part of the problem
theorem max_m_value :
  (∃ m : ℝ, ∀ x : ℝ, f x ≥ -2*x^2 + m) ∧
  (∀ m : ℝ, (∀ x : ℝ, f x ≥ -2*x^2 + m) → m ≤ 5/2) :=
sorry

end f_geq_2_iff_max_m_value_l2807_280748


namespace triangle_inequality_l2807_280774

theorem triangle_inequality (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a + b > c) (hbc : b + c > a) (hca : c + a > b) :
  a^3 + b^3 + c^3 + 4*a*b*c ≤ 9/32 * (a + b + c)^3 := by
sorry

end triangle_inequality_l2807_280774


namespace find_k_l2807_280710

-- Define the circles and points
def larger_circle : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 100}
def smaller_circle : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 25}
def P : ℝ × ℝ := (6, 8)
def S (k : ℝ) : ℝ × ℝ := (0, k)
def QR : ℝ := 5

-- Theorem statement
theorem find_k :
  P ∈ larger_circle ∧
  ∀ k, S k ∈ smaller_circle →
  QR = 5 →
  ∃ k, S k ∈ smaller_circle ∧ k = 5 := by
sorry

end find_k_l2807_280710


namespace rally_ticket_cost_l2807_280779

/-- The cost of tickets bought at the door at a rally --/
def ticket_cost_at_door (total_attendance : ℕ) (pre_rally_ticket_cost : ℚ) 
  (total_receipts : ℚ) (pre_rally_tickets : ℕ) : ℚ :=
  (total_receipts - pre_rally_ticket_cost * pre_rally_tickets) / (total_attendance - pre_rally_tickets)

/-- Theorem stating the cost of tickets bought at the door --/
theorem rally_ticket_cost : 
  ticket_cost_at_door 750 2 (1706.25) 475 = (2.75 : ℚ) := by sorry

end rally_ticket_cost_l2807_280779


namespace inequality_proof_l2807_280722

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : a + b + c = 3) :
  (a * b / Real.sqrt (c^2 + 3)) + 
  (b * c / Real.sqrt (a^2 + 3)) + 
  (c * a / Real.sqrt (b^2 + 3)) ≤ 3/2 := by
  sorry

end inequality_proof_l2807_280722


namespace eggs_broken_count_l2807_280711

-- Define the number of brown eggs
def brown_eggs : ℕ := 10

-- Define the number of white eggs
def white_eggs : ℕ := 3 * brown_eggs

-- Define the total number of eggs before dropping
def total_eggs_before : ℕ := brown_eggs + white_eggs

-- Define the number of eggs left after dropping
def eggs_left_after : ℕ := 20

-- Theorem to prove
theorem eggs_broken_count : total_eggs_before - eggs_left_after = 20 := by
  sorry

end eggs_broken_count_l2807_280711


namespace total_amount_paid_l2807_280720

/-- Calculates the discounted price for a fruit given its weight, price per kg, and discount percentage. -/
def discountedPrice (weight : Float) (pricePerKg : Float) (discountPercent : Float) : Float :=
  weight * pricePerKg * (1 - discountPercent / 100)

/-- Represents the shopping trip and calculates the total amount paid. -/
def shoppingTrip : Float :=
  discountedPrice 8 70 10 +    -- Grapes
  discountedPrice 11 55 0 +    -- Mangoes
  discountedPrice 5 45 20 +    -- Oranges
  discountedPrice 3 90 5 +     -- Apples
  discountedPrice 4.5 120 0    -- Cherries

/-- Theorem stating that the total amount paid is $2085.50 -/
theorem total_amount_paid : shoppingTrip = 2085.50 := by
  sorry

end total_amount_paid_l2807_280720


namespace factorial_base_representation_823_l2807_280704

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def factorial_base_coeff (n k : ℕ) : ℕ :=
  (n / factorial k) % (k + 1)

theorem factorial_base_representation_823 :
  factorial_base_coeff 823 4 = 4 := by
  sorry

end factorial_base_representation_823_l2807_280704


namespace greater_number_problem_l2807_280784

theorem greater_number_problem (x y : ℝ) (h1 : x + y = 30) (h2 : x - y = 10) (h3 : x > y) : x = 20 := by
  sorry

end greater_number_problem_l2807_280784


namespace bernard_white_notebooks_l2807_280799

/-- The number of white notebooks Bernard had -/
def white_notebooks : ℕ := sorry

/-- The number of red notebooks Bernard had -/
def red_notebooks : ℕ := 15

/-- The number of blue notebooks Bernard had -/
def blue_notebooks : ℕ := 17

/-- The number of notebooks Bernard gave to Tom -/
def notebooks_given : ℕ := 46

/-- The number of notebooks Bernard had left -/
def notebooks_left : ℕ := 5

/-- The total number of notebooks Bernard originally had -/
def total_notebooks : ℕ := notebooks_given + notebooks_left

theorem bernard_white_notebooks : 
  white_notebooks = total_notebooks - (red_notebooks + blue_notebooks) ∧ 
  white_notebooks = 19 := by sorry

end bernard_white_notebooks_l2807_280799


namespace shopkeeper_profit_l2807_280765

theorem shopkeeper_profit (discount : ℝ) (profit_with_discount : ℝ) :
  discount = 0.04 →
  profit_with_discount = 0.26 →
  let cost_price := 100
  let selling_price := cost_price * (1 + profit_with_discount)
  let marked_price := selling_price / (1 - discount)
  let profit_without_discount := (marked_price - cost_price) / cost_price
  profit_without_discount = 0.3125 := by sorry

end shopkeeper_profit_l2807_280765


namespace game_lives_per_player_l2807_280798

theorem game_lives_per_player (initial_players : ℕ) (new_players : ℕ) (total_lives : ℕ) :
  initial_players = 7 →
  new_players = 2 →
  total_lives = 63 →
  (total_lives / (initial_players + new_players) : ℚ) = 7 :=
by sorry

end game_lives_per_player_l2807_280798


namespace volunteers_selection_ways_l2807_280725

/-- The number of ways to select volunteers for community service --/
def select_volunteers (n : ℕ) : ℕ :=
  let both_days := n  -- Select 1 person for both days
  let saturday := n - 1  -- Select 1 person for Saturday from remaining n-1
  let sunday := n - 2  -- Select 1 person for Sunday from remaining n-2
  both_days * saturday * sunday

/-- Theorem: The number of ways to select exactly one person to serve for both days
    out of 5 volunteers, with 2 people selected each day, is equal to 60 --/
theorem volunteers_selection_ways :
  select_volunteers 5 = 60 := by
  sorry

end volunteers_selection_ways_l2807_280725


namespace polar_to_rectangular_equivalence_l2807_280768

theorem polar_to_rectangular_equivalence (ρ θ x y : ℝ) :
  (x = ρ * Real.cos θ) →
  (y = ρ * Real.sin θ) →
  (3 * ρ * Real.cos θ + 4 * ρ * Real.sin θ = 2) →
  (3 * x + 4 * y - 2 = 0) :=
by sorry

end polar_to_rectangular_equivalence_l2807_280768


namespace saree_final_price_l2807_280723

/-- Calculates the final sale price of a saree after discounts, tax, and custom fee. -/
def saree_sale_price (original_price : ℝ) (discount1 discount2 discount3 tax : ℝ) (custom_fee : ℝ) : ℝ :=
  let price_after_discounts := original_price * (1 - discount1) * (1 - discount2) * (1 - discount3)
  let price_after_tax := price_after_discounts * (1 + tax)
  price_after_tax + custom_fee

/-- Theorem stating that the final sale price of the saree is 773.2 -/
theorem saree_final_price :
  saree_sale_price 1200 0.25 0.20 0.15 0.10 100 = 773.2 := by
  sorry

#eval saree_sale_price 1200 0.25 0.20 0.15 0.10 100

end saree_final_price_l2807_280723


namespace complex_number_equal_parts_l2807_280734

theorem complex_number_equal_parts (a : ℝ) : 
  let z : ℂ := (a + Complex.I) / Complex.I
  (z.re = z.im) → a = -1 := by
  sorry

end complex_number_equal_parts_l2807_280734


namespace expression_evaluation_l2807_280786

theorem expression_evaluation : 
  let x : ℝ := -3
  (5 + x * (4 + x) - 4^2 + (x^2 - 3*x + 2)) / (x^2 - 4 + x - 1) = 6 := by
  sorry

end expression_evaluation_l2807_280786


namespace hyperbola_equation_l2807_280715

theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (∃ x y : ℝ, y = Real.sqrt 3 * x) →
  (∃ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ∧ x = -6 ∧ y = 0) →
  (∃ x y : ℝ, y^2 = 2*x ∧ x = -6) →
  (∀ x y : ℝ, x^2 / 9 - y^2 / 27 = 1) :=
by sorry

end hyperbola_equation_l2807_280715


namespace probability_consonant_initials_l2807_280758

/-- The probability of selecting a student with consonant initials -/
theorem probability_consonant_initials :
  let total_letters : ℕ := 26
  let vowels : Finset Char := {'A', 'E', 'I', 'O', 'U', 'Y'}
  let consonants : ℕ := total_letters - vowels.card
  consonants / total_letters = 10 / 13 := by
  sorry

end probability_consonant_initials_l2807_280758


namespace share_price_increase_l2807_280761

theorem share_price_increase (P : ℝ) (X : ℝ) : 
  X > 0 →
  (P * (1 + X / 100)) * (1 + 1 / 3) = P * 1.6 →
  X = 20 := by
sorry

end share_price_increase_l2807_280761


namespace triangle_area_and_side_l2807_280791

theorem triangle_area_and_side (a b c : ℝ) (A B C : ℝ) :
  b = 2 →
  c = Real.sqrt 3 →
  A = π / 6 →
  (1/2 * b * c * Real.sin A = Real.sqrt 3 / 2) ∧
  (a^2 = b^2 + c^2 - 2*b*c*Real.cos A) →
  (1/2 * b * c * Real.sin A = Real.sqrt 3 / 2) ∧ (a = 1) := by
  sorry

end triangle_area_and_side_l2807_280791


namespace ratio_problem_l2807_280733

theorem ratio_problem (x y : ℚ) (h : (3 * x + 2 * y) / (2 * x - y) = 5 / 4) : 
  x / y = -13 / 2 := by
sorry

end ratio_problem_l2807_280733


namespace fixed_point_on_quadratic_graph_l2807_280778

/-- The fixed point on the graph of y = 9x^2 + mx + 3m for all real m -/
theorem fixed_point_on_quadratic_graph :
  ∀ (m : ℝ), 9 * (-3)^2 + m * (-3) + 3 * m = 81 := by
  sorry

end fixed_point_on_quadratic_graph_l2807_280778


namespace f_recursive_relation_l2807_280742

def f (n : ℕ) : ℕ := (Finset.range (2 * n + 1)).sum (λ i => i ^ 2)

theorem f_recursive_relation (k : ℕ) : f (k + 1) = f k + (2 * k + 1) ^ 2 + (2 * k + 2) ^ 2 := by
  sorry

end f_recursive_relation_l2807_280742


namespace simplify_nested_expression_l2807_280700

theorem simplify_nested_expression (x : ℝ) : 2 - (3 - (4 - (5 - (2*x - 3)))) = -5 + 2*x := by
  sorry

end simplify_nested_expression_l2807_280700


namespace book_loss_percentage_l2807_280706

/-- Given that the cost price of 30 books equals the selling price of 40 books,
    prove that the loss percentage is 25%. -/
theorem book_loss_percentage (C S : ℝ) (h : C > 0) (h1 : 30 * C = 40 * S) : 
  (C - S) / C * 100 = 25 := by
  sorry

end book_loss_percentage_l2807_280706


namespace expected_twos_l2807_280777

/-- The probability of rolling a 2 on a standard die -/
def prob_two : ℚ := 1 / 6

/-- The number of dice rolled -/
def num_dice : ℕ := 3

/-- The expected number of 2's when rolling three standard dice -/
theorem expected_twos : 
  num_dice * prob_two = 1 / 2 := by sorry

end expected_twos_l2807_280777


namespace cos_alpha_for_point_neg_one_two_l2807_280735

/-- Given an angle α whose terminal side passes through the point (-1, 2), 
    prove that cos α = -√5 / 5 -/
theorem cos_alpha_for_point_neg_one_two (α : Real) : 
  (∃ (t : Real), t > 0 ∧ t * Real.cos α = -1 ∧ t * Real.sin α = 2) → 
  Real.cos α = -Real.sqrt 5 / 5 := by
sorry

end cos_alpha_for_point_neg_one_two_l2807_280735


namespace probability_green_yellow_blue_l2807_280713

def total_balls : ℕ := 500
def green_balls : ℕ := 100
def yellow_balls : ℕ := 70
def blue_balls : ℕ := 50

theorem probability_green_yellow_blue :
  (green_balls + yellow_balls + blue_balls : ℚ) / total_balls = 220 / 500 := by
  sorry

end probability_green_yellow_blue_l2807_280713


namespace right_triangle_sine_l2807_280729

theorem right_triangle_sine (a b c : ℝ) (h1 : a^2 + b^2 = c^2) (h2 : a = 9) (h3 : c = 15) :
  a / c = 3 / 5 := by
  sorry

end right_triangle_sine_l2807_280729


namespace volume_ratio_cylinder_cone_sphere_l2807_280762

noncomputable def sphere_volume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

noncomputable def cylinder_volume (r h : ℝ) : ℝ := Real.pi * r^2 * h

noncomputable def cone_volume (r h : ℝ) : ℝ := (1 / 3) * Real.pi * r^2 * h

theorem volume_ratio_cylinder_cone_sphere (r : ℝ) (h_pos : r > 0) :
  ∃ (k : ℝ), k > 0 ∧ 
    cylinder_volume r (2 * r) = 3 * k ∧
    cone_volume r (2 * r) = k ∧
    sphere_volume r = 2 * k := by
  sorry

end volume_ratio_cylinder_cone_sphere_l2807_280762


namespace cost_for_three_roofs_is_1215_l2807_280714

/-- Calculates the total cost of materials for building roofs with discounts applied --/
def total_cost_with_discounts (
  num_roofs : ℕ
  ) (
  metal_bars_per_roof : ℕ
  ) (
  wooden_beams_per_roof : ℕ
  ) (
  steel_rods_per_roof : ℕ
  ) (
  bars_per_set : ℕ
  ) (
  beams_per_set : ℕ
  ) (
  rods_per_set : ℕ
  ) (
  cost_per_bar : ℕ
  ) (
  cost_per_beam : ℕ
  ) (
  cost_per_rod : ℕ
  ) (
  discount_threshold : ℕ
  ) (
  discount_rate : ℚ
  ) : ℕ :=
  sorry

/-- Theorem stating that the total cost for building 3 roofs with given specifications is $1215 --/
theorem cost_for_three_roofs_is_1215 :
  total_cost_with_discounts 3 2 3 1 7 5 4 10 15 20 10 (1/10) = 1215 :=
  sorry

end cost_for_three_roofs_is_1215_l2807_280714


namespace second_number_value_l2807_280712

theorem second_number_value (a b c : ℝ) 
  (sum_eq : a + b + c = 120)
  (ratio_ab : a / b = 3 / 4)
  (ratio_bc : b / c = 2 / 5)
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) : 
  b = 480 / 17 := by
sorry

end second_number_value_l2807_280712


namespace smallest_number_of_eggs_l2807_280705

theorem smallest_number_of_eggs : ∀ (n : ℕ), 
  (n > 150) → 
  (∃ (c : ℕ), n = 15 * c - 5) → 
  (∀ (m : ℕ), m > 150 ∧ (∃ (d : ℕ), m = 15 * d - 5) → m ≥ n) → 
  n = 160 := by
sorry

end smallest_number_of_eggs_l2807_280705


namespace trajectory_of_M_equation_of_l_area_of_POM_smallest_circle_l2807_280754

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 8*y = 0

-- Define point P
def P : ℝ × ℝ := (2, 2)

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- Define the midpoint M of chord AB
def M (x y : ℝ) : Prop := ∃ (a b : ℝ × ℝ), 
  circle_C a.1 a.2 ∧ circle_C b.1 b.2 ∧ 
  x = (a.1 + b.1) / 2 ∧ y = (a.2 + b.2) / 2

-- Theorem 1: Trajectory of M
theorem trajectory_of_M : 
  ∀ x y : ℝ, M x y → (x - 1)^2 + (y - 3)^2 = 2 :=
sorry

-- Theorem 2a: Equation of line l when |OP| = |OM|
theorem equation_of_l : 
  ∃ x y : ℝ, M x y ∧ (x^2 + y^2 = P.1^2 + P.2^2) → 
  ∀ x y : ℝ, y = -1/3 * x + 8/3 :=
sorry

-- Theorem 2b: Area of triangle POM when |OP| = |OM|
theorem area_of_POM : 
  ∃ x y : ℝ, M x y ∧ (x^2 + y^2 = P.1^2 + P.2^2) → 
  (1/2) * |P.1 * y - P.2 * x| = 16/5 :=
sorry

-- Theorem 3: Equation of smallest circle through intersection of C and l
theorem smallest_circle : 
  ∃ x y : ℝ, circle_C x y ∧ y = -1/3 * x + 8/3 → 
  ∀ x y : ℝ, (x + 2/5)^2 + (y - 14/5)^2 = 72/5 :=
sorry

end trajectory_of_M_equation_of_l_area_of_POM_smallest_circle_l2807_280754


namespace r_daily_earnings_l2807_280726

/-- Represents the daily earnings of individuals p, q, and r -/
structure DailyEarnings where
  p : ℝ
  q : ℝ
  r : ℝ

/-- The conditions given in the problem -/
def problem_conditions (e : DailyEarnings) : Prop :=
  9 * (e.p + e.q + e.r) = 1710 ∧
  5 * (e.p + e.r) = 600 ∧
  7 * (e.q + e.r) = 910

/-- Theorem stating that given the problem conditions, r's daily earnings are 60 -/
theorem r_daily_earnings (e : DailyEarnings) :
  problem_conditions e → e.r = 60 := by
  sorry


end r_daily_earnings_l2807_280726


namespace distance_between_towns_l2807_280741

theorem distance_between_towns (total_distance : ℝ) : total_distance = 50 :=
  let petya_distance := 10 + (1/4) * (total_distance - 10)
  let kolya_distance := 20 + (1/3) * (total_distance - 20)
  have h1 : petya_distance + kolya_distance = total_distance := by sorry
  have h2 : petya_distance = 10 + (1/4) * (total_distance - 10) := by sorry
  have h3 : kolya_distance = 20 + (1/3) * (total_distance - 20) := by sorry
  sorry

end distance_between_towns_l2807_280741


namespace student_age_problem_l2807_280753

theorem student_age_problem (total_students : Nat) 
  (avg_age_all : Nat) (num_group1 : Nat) (avg_age_group1 : Nat) 
  (num_group2 : Nat) (avg_age_group2 : Nat) :
  total_students = 17 →
  avg_age_all = 17 →
  num_group1 = 5 →
  avg_age_group1 = 14 →
  num_group2 = 9 →
  avg_age_group2 = 16 →
  (total_students * avg_age_all) - (num_group1 * avg_age_group1) - (num_group2 * avg_age_group2) = 75 := by
  sorry

end student_age_problem_l2807_280753


namespace fraction_multiplication_l2807_280739

theorem fraction_multiplication : (2 : ℚ) / 3 * 4 / 7 * 9 / 11 = 24 / 77 := by
  sorry

end fraction_multiplication_l2807_280739


namespace tangential_quadrilateral_l2807_280772

/-- A point in the plane -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A circle in the plane -/
structure Circle :=
  (center : Point) (radius : ℝ)

/-- A quadrilateral in the plane -/
structure Quadrilateral :=
  (A B C D : Point)

/-- Predicate to check if a quadrilateral is cyclic -/
def is_cyclic (q : Quadrilateral) : Prop := sorry

/-- Predicate to check if a quadrilateral is tangential -/
def is_tangential (q : Quadrilateral) : Prop := sorry

/-- Get the incenter of a triangle -/
def incenter (A B C : Point) : Point := sorry

/-- Check if four points are concyclic -/
def are_concyclic (A B C D : Point) : Prop := sorry

/-- Main theorem -/
theorem tangential_quadrilateral 
  (q : Quadrilateral) 
  (h1 : is_cyclic q) 
  (h2 : let I := incenter q.A q.B q.C
        let J := incenter q.A q.D q.C
        are_concyclic q.B I J q.D) : 
  is_tangential q :=
sorry

end tangential_quadrilateral_l2807_280772


namespace original_number_l2807_280707

theorem original_number : ∃ x : ℝ, 3 * (2 * x + 6) = 72 ∧ x = 9 := by
  sorry

end original_number_l2807_280707


namespace cyclic_equation_solution_l2807_280745

def cyclic_index (n i : ℕ) : ℕ :=
  (i - 1) % n + 1

theorem cyclic_equation_solution (n : ℕ) (x : ℕ → ℝ) :
  (∀ i, 0 ≤ x i) →
  (∀ k, x k + x (cyclic_index n (k + 1)) = (x (cyclic_index n (k + 2)))^2) →
  (∀ i, x i = 0 ∨ x i = 2) :=
sorry

end cyclic_equation_solution_l2807_280745


namespace frisbee_price_l2807_280747

/-- Given the conditions of frisbee sales, prove the price of non-$4 frisbees -/
theorem frisbee_price (total_frisbees : ℕ) (total_receipts : ℕ) (price_known : ℕ) (min_known_price : ℕ) :
  total_frisbees = 64 →
  total_receipts = 196 →
  price_known = 4 →
  min_known_price = 4 →
  ∃ (price_unknown : ℕ),
    price_unknown * (total_frisbees - min_known_price) + price_known * min_known_price = total_receipts ∧
    price_unknown = 3 := by
  sorry

#check frisbee_price

end frisbee_price_l2807_280747
