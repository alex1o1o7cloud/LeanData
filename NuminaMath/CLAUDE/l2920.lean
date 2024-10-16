import Mathlib

namespace NUMINAMATH_CALUDE_smallest_number_with_conditions_l2920_292021

theorem smallest_number_with_conditions : ∃ n : ℕ, 
  (∀ m : ℕ, m ≥ n → (11 ∣ m ∧ ∀ k : ℕ, 3 ≤ k ∧ k ≤ 8 → m % k = 2) → m ≥ 3362) ∧
  (11 ∣ 3362) ∧
  (∀ k : ℕ, 3 ≤ k ∧ k ≤ 8 → 3362 % k = 2) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_conditions_l2920_292021


namespace NUMINAMATH_CALUDE_unique_solution_system_of_equations_l2920_292024

theorem unique_solution_system_of_equations :
  ∃! (x y : ℝ), x + 2 * y = 2 ∧ 3 * x - 4 * y = -24 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_solution_system_of_equations_l2920_292024


namespace NUMINAMATH_CALUDE_factorial_ratio_simplification_l2920_292004

theorem factorial_ratio_simplification : (Nat.factorial 10 * Nat.factorial 6 * Nat.factorial 3) / (Nat.factorial 9 * Nat.factorial 7) = 60 / 7 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_simplification_l2920_292004


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2920_292045

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 5 * x - 2

-- Define the solution set of the original inequality
def solution_set (a : ℝ) : Set ℝ := {x : ℝ | (1/2 : ℝ) < x ∧ x < 2}

-- State the theorem
theorem quadratic_inequality_solution :
  ∃ (a : ℝ), 
    (∀ x, x ∈ solution_set a ↔ f a x > 0) ∧
    (a = -2) ∧
    (∀ x, a * x^2 - 5 * x + a^2 - 1 > 0 ↔ -3 < x ∧ x < 1/2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2920_292045


namespace NUMINAMATH_CALUDE_at_hash_product_l2920_292044

-- Define the @ operation
def at_op (a b c : ℤ) : ℤ := a * b - b^2 + c

-- Define the # operation
def hash_op (a b c : ℤ) : ℤ := a + b - a * b^2 + c

-- Theorem statement
theorem at_hash_product : 
  let c : ℤ := 3
  (at_op 4 3 c) * (hash_op 4 3 c) = -156 := by
  sorry

end NUMINAMATH_CALUDE_at_hash_product_l2920_292044


namespace NUMINAMATH_CALUDE_sound_distance_at_zero_celsius_sound_distance_in_five_seconds_l2920_292080

/-- Represents the speed of sound in air at different temperatures -/
def speed_of_sound (temp : Int) : Int :=
  match temp with
  | -20 => 318
  | -10 => 324
  | 0 => 330
  | 10 => 336
  | 20 => 342
  | 30 => 348
  | _ => 0  -- Default case for temperatures not in the table

/-- Calculates the distance traveled by sound in a given time at 0°C -/
def distance_traveled (time : Int) : Int :=
  (speed_of_sound 0) * time

theorem sound_distance_at_zero_celsius (time : Int) :
  distance_traveled time = speed_of_sound 0 * time :=
by sorry

theorem sound_distance_in_five_seconds :
  distance_traveled 5 = 1650 :=
by sorry

end NUMINAMATH_CALUDE_sound_distance_at_zero_celsius_sound_distance_in_five_seconds_l2920_292080


namespace NUMINAMATH_CALUDE_greatest_fourth_term_l2920_292095

/-- An arithmetic sequence of five positive integers with sum 60 -/
structure ArithmeticSequence where
  a : ℕ+  -- first term
  d : ℕ+  -- common difference
  sum_eq_60 : a + (a + d) + (a + 2*d) + (a + 3*d) + (a + 4*d) = 60

/-- The fourth term of an arithmetic sequence -/
def fourth_term (seq : ArithmeticSequence) : ℕ := seq.a + 3 * seq.d

/-- The greatest possible fourth term is 34 -/
theorem greatest_fourth_term :
  ∀ seq : ArithmeticSequence, fourth_term seq ≤ 34 ∧ 
  ∃ seq : ArithmeticSequence, fourth_term seq = 34 :=
sorry

end NUMINAMATH_CALUDE_greatest_fourth_term_l2920_292095


namespace NUMINAMATH_CALUDE_complex_product_l2920_292065

/-- Given complex numbers Q, E, and D, prove that their product is -25i. -/
theorem complex_product (Q E D : ℂ) : 
  Q = 3 + 4*I ∧ E = -I ∧ D = 3 - 4*I → Q * E * D = -25 * I :=
by sorry

end NUMINAMATH_CALUDE_complex_product_l2920_292065


namespace NUMINAMATH_CALUDE_remainder_444_power_222_mod_13_l2920_292083

theorem remainder_444_power_222_mod_13 : 444^222 ≡ 1 [ZMOD 13] := by
  sorry

end NUMINAMATH_CALUDE_remainder_444_power_222_mod_13_l2920_292083


namespace NUMINAMATH_CALUDE_installation_problem_l2920_292053

theorem installation_problem (x₁ x₂ x₃ k : ℕ) :
  x₁ + x₂ + x₃ ≤ 200 ∧
  x₂ = 4 * x₁ ∧
  x₃ = k * x₁ ∧
  5 * x₃ = x₂ + 99 →
  x₁ = 9 ∧ x₂ = 36 ∧ x₃ = 27 := by
sorry

end NUMINAMATH_CALUDE_installation_problem_l2920_292053


namespace NUMINAMATH_CALUDE_max_salary_for_given_constraints_l2920_292050

/-- Represents a baseball team with salary constraints -/
structure BaseballTeam where
  num_players : ℕ
  min_salary : ℕ
  max_total_salary : ℕ

/-- Calculates the maximum possible salary for a single player -/
def max_single_player_salary (team : BaseballTeam) : ℕ :=
  team.max_total_salary - (team.num_players - 1) * team.min_salary

/-- Theorem stating the maximum possible salary for a single player
    in a team with given constraints -/
theorem max_salary_for_given_constraints :
  let team : BaseballTeam := {
    num_players := 25,
    min_salary := 20000,
    max_total_salary := 800000
  }
  max_single_player_salary team = 320000 := by
  sorry

#eval max_single_player_salary {
  num_players := 25,
  min_salary := 20000,
  max_total_salary := 800000
}

end NUMINAMATH_CALUDE_max_salary_for_given_constraints_l2920_292050


namespace NUMINAMATH_CALUDE_equation_solution_l2920_292001

theorem equation_solution : 
  {x : ℝ | Real.sqrt ((2 + Real.sqrt 3) ^ x) + Real.sqrt ((2 - Real.sqrt 3) ^ x) = 4} = {2, -2} := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2920_292001


namespace NUMINAMATH_CALUDE_complex_magnitude_one_l2920_292035

theorem complex_magnitude_one (r : ℝ) (z : ℂ) (h1 : |r| < 4) (h2 : z + 1/z + 2 = r) : 
  Complex.abs z = 1 := by sorry

end NUMINAMATH_CALUDE_complex_magnitude_one_l2920_292035


namespace NUMINAMATH_CALUDE_inequality_proof_l2920_292003

theorem inequality_proof (x y z : ℝ) 
  (hpos : x > 0 ∧ y > 0 ∧ z > 0) 
  (hsum : x + y + z = 1) : 
  (1 - x) * (1 - y) * (1 - z) > 8 * x * y * z := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2920_292003


namespace NUMINAMATH_CALUDE_melanie_cats_count_l2920_292034

theorem melanie_cats_count (jacob_cats : ℕ) (annie_cats : ℕ) (melanie_cats : ℕ)
  (h1 : jacob_cats = 90)
  (h2 : annie_cats * 3 = jacob_cats)
  (h3 : melanie_cats = annie_cats * 2) :
  melanie_cats = 60 := by
  sorry

end NUMINAMATH_CALUDE_melanie_cats_count_l2920_292034


namespace NUMINAMATH_CALUDE_unique_number_property_l2920_292094

theorem unique_number_property : ∃! x : ℝ, 3 * x = x + 18 := by sorry

end NUMINAMATH_CALUDE_unique_number_property_l2920_292094


namespace NUMINAMATH_CALUDE_trigonometric_expression_equals_one_l2920_292096

theorem trigonometric_expression_equals_one : 
  (Real.sin (15 * π / 180) * Real.cos (15 * π / 180) + 
   Real.cos (165 * π / 180) * Real.cos (105 * π / 180)) / 
  (Real.sin (19 * π / 180) * Real.cos (11 * π / 180) + 
   Real.cos (161 * π / 180) * Real.cos (101 * π / 180)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_equals_one_l2920_292096


namespace NUMINAMATH_CALUDE_wilson_number_l2920_292030

theorem wilson_number (N : ℚ) : N - (1/3) * N = 16/3 → N = 8 := by
  sorry

end NUMINAMATH_CALUDE_wilson_number_l2920_292030


namespace NUMINAMATH_CALUDE_symmetric_parabola_l2920_292079

/-- 
Given a parabola with equation y^2 = 2x and a point (-1, 0),
prove that the equation y^2 = -2(x + 2) represents the parabola 
symmetric to the original parabola with respect to the given point.
-/
theorem symmetric_parabola (x y : ℝ) : 
  (∀ x y, y^2 = 2*x → 
   ∃ x' y', x' = -x - 2 ∧ y' = -y ∧ y'^2 = -2*(x' + 2)) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_parabola_l2920_292079


namespace NUMINAMATH_CALUDE_pyramid_base_theorem_l2920_292016

def isPyramidBase (a b c d e : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e

def pyramidTop (a b c d e : ℕ) : ℕ :=
  a * b^4 * c^6 * d^4 * e

theorem pyramid_base_theorem (a b c d e : ℕ) :
  isPyramidBase a b c d e ∧ pyramidTop a b c d e = 140026320 →
  ((a = 1 ∧ b = 2 ∧ c = 3 ∧ d = 7 ∧ e = 5) ∨
   (a = 1 ∧ b = 7 ∧ c = 3 ∧ d = 2 ∧ e = 5) ∨
   (a = 5 ∧ b = 2 ∧ c = 3 ∧ d = 7 ∧ e = 1) ∨
   (a = 5 ∧ b = 7 ∧ c = 3 ∧ d = 2 ∧ e = 1)) :=
by sorry

end NUMINAMATH_CALUDE_pyramid_base_theorem_l2920_292016


namespace NUMINAMATH_CALUDE_tourist_arrangement_count_l2920_292088

/-- The number of tourists --/
def num_tourists : ℕ := 5

/-- The number of scenic spots --/
def num_spots : ℕ := 4

/-- The function to calculate the number of valid arrangements --/
def valid_arrangements (n : ℕ) (k : ℕ) : ℕ :=
  k^n - Nat.choose k 1 * (k-1)^n + Nat.choose k 2 * (k-2)^n - Nat.choose k 3 * (k-3)^n

/-- The main theorem to prove --/
theorem tourist_arrangement_count :
  (valid_arrangements num_tourists num_spots) * (num_spots - 1) * (num_spots - 1) / num_spots = 216 :=
sorry

end NUMINAMATH_CALUDE_tourist_arrangement_count_l2920_292088


namespace NUMINAMATH_CALUDE_remainder_5_divisors_2002_l2920_292072

def divides_with_remainder_5 (d : ℕ) : Prop :=
  ∃ q : ℕ, 2007 = d * q + 5

def divisors_of_2002 : Set ℕ :=
  {d : ℕ | d > 0 ∧ 2002 % d = 0}

theorem remainder_5_divisors_2002 :
  {d : ℕ | divides_with_remainder_5 d} = {d ∈ divisors_of_2002 | d > 5} :=
by sorry

end NUMINAMATH_CALUDE_remainder_5_divisors_2002_l2920_292072


namespace NUMINAMATH_CALUDE_messages_total_680_l2920_292058

/-- Calculates the total number of messages sent by Alina and Lucia over three days -/
def total_messages (lucia_day1 : ℕ) (alina_difference : ℕ) : ℕ :=
  let alina_day1 := lucia_day1 - alina_difference
  let day1_total := lucia_day1 + alina_day1
  let lucia_day2 := lucia_day1 / 3
  let alina_day2 := alina_day1 * 2
  let day2_total := lucia_day2 + alina_day2
  day1_total + day2_total + day1_total

theorem messages_total_680 :
  total_messages 120 20 = 680 := by
  sorry

end NUMINAMATH_CALUDE_messages_total_680_l2920_292058


namespace NUMINAMATH_CALUDE_power_of_two_plus_one_l2920_292084

-- Define a relation for numbers with the same prime factors
def same_prime_factors (x y : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → (p ∣ x ↔ p ∣ y)

theorem power_of_two_plus_one (b m n : ℕ) 
  (hb : b > 1) 
  (hm : m > 0) 
  (hn : n > 0) 
  (hmn : m ≠ n) 
  (h_same_factors : same_prime_factors (b^m - 1) (b^n - 1)) : 
  ∃ k : ℕ, b + 1 = 2^k :=
sorry

end NUMINAMATH_CALUDE_power_of_two_plus_one_l2920_292084


namespace NUMINAMATH_CALUDE_line_inclination_angle_l2920_292070

/-- The inclination angle of the line √3x - y + 1 = 0 is π/3 -/
theorem line_inclination_angle :
  let line := {(x, y) : ℝ × ℝ | Real.sqrt 3 * x - y + 1 = 0}
  ∃ α : ℝ, α = π / 3 ∧ ∀ (x y : ℝ), (x, y) ∈ line → Real.tan α = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_line_inclination_angle_l2920_292070


namespace NUMINAMATH_CALUDE_intersection_of_lines_l2920_292027

/-- Given four points in 3D space, this theorem states that the intersection
    of the lines formed by these points is at a specific coordinate. -/
theorem intersection_of_lines (P Q R S : ℝ × ℝ × ℝ) : 
  P = (4, -8, 8) →
  Q = (14, -18, 14) →
  R = (1, 2, -7) →
  S = (3, -6, 9) →
  ∃ t s : ℝ, 
    (4 + 10*t, -8 - 10*t, 8 + 6*t) = (1 + 2*s, 2 - 8*s, -7 + 16*s) ∧
    (4 + 10*t, -8 - 10*t, 8 + 6*t) = (14/3, -22/3, 38/3) :=
by sorry


end NUMINAMATH_CALUDE_intersection_of_lines_l2920_292027


namespace NUMINAMATH_CALUDE_blue_highlighters_count_l2920_292029

def total_highlighters : ℕ := 15
def pink_highlighters : ℕ := 3
def yellow_highlighters : ℕ := 7

theorem blue_highlighters_count :
  total_highlighters - (pink_highlighters + yellow_highlighters) = 5 :=
by sorry

end NUMINAMATH_CALUDE_blue_highlighters_count_l2920_292029


namespace NUMINAMATH_CALUDE_hiking_problem_l2920_292002

/-- Proves that the number of people in each van is 5, given the conditions of the hiking problem --/
theorem hiking_problem (num_cars num_taxis num_vans : ℕ) 
                       (people_per_car people_per_taxi total_people : ℕ) 
                       (h1 : num_cars = 3)
                       (h2 : num_taxis = 6)
                       (h3 : num_vans = 2)
                       (h4 : people_per_car = 4)
                       (h5 : people_per_taxi = 6)
                       (h6 : total_people = 58)
                       (h7 : total_people = num_cars * people_per_car + 
                                            num_taxis * people_per_taxi + 
                                            num_vans * (total_people - num_cars * people_per_car - num_taxis * people_per_taxi) / num_vans) : 
  (total_people - num_cars * people_per_car - num_taxis * people_per_taxi) / num_vans = 5 := by
  sorry

end NUMINAMATH_CALUDE_hiking_problem_l2920_292002


namespace NUMINAMATH_CALUDE_octal_to_binary_conversion_l2920_292057

theorem octal_to_binary_conversion :
  (135 : Nat).digits 8 = [1, 3, 5] →
  (135 : Nat).digits 2 = [1, 0, 1, 1, 1, 0, 1] :=
by
  sorry

end NUMINAMATH_CALUDE_octal_to_binary_conversion_l2920_292057


namespace NUMINAMATH_CALUDE_length_of_PQ_l2920_292031

-- Define the points
variable (P Q R S : ℝ × ℝ)

-- Define the distances between points
def distance (A B : ℝ × ℝ) : ℝ := sorry

-- Define isosceles triangle
def isIsosceles (A B C : ℝ × ℝ) : Prop :=
  distance A B = distance A C

-- Define perimeter of a triangle
def perimeter (A B C : ℝ × ℝ) : ℝ :=
  distance A B + distance B C + distance C A

-- State the theorem
theorem length_of_PQ (P Q R S : ℝ × ℝ) :
  isIsosceles P Q R →
  isIsosceles Q R S →
  perimeter Q R S = 24 →
  perimeter P Q R = 23 →
  distance Q R = 10 →
  distance P Q = 6.5 := by sorry

end NUMINAMATH_CALUDE_length_of_PQ_l2920_292031


namespace NUMINAMATH_CALUDE_planes_parallel_if_common_perpendicular_l2920_292012

-- Define the basic types
variable (Point : Type) (Line : Type) (Plane : Type)

-- Define the relationships
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)

-- State the theorem
theorem planes_parallel_if_common_perpendicular 
  (a b : Plane) (m : Line) : 
  a ≠ b → 
  perpendicular m a → 
  perpendicular m b → 
  parallel a b :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_if_common_perpendicular_l2920_292012


namespace NUMINAMATH_CALUDE_distribute_men_and_women_l2920_292066

/- Define the number of men and women -/
def num_men : ℕ := 4
def num_women : ℕ := 5

/- Define the size of each group -/
def group_size : ℕ := 3

/- Define a function to calculate the number of ways to distribute people -/
def distribute_people (m : ℕ) (w : ℕ) : ℕ :=
  let ways_group1 := (m.choose 1) * (w.choose 2)
  let ways_group2 := ((m - 1).choose 1) * ((w - 2).choose 2)
  ways_group1 * ways_group2 / 2

/- Theorem statement -/
theorem distribute_men_and_women :
  distribute_people num_men num_women = 180 :=
by sorry

end NUMINAMATH_CALUDE_distribute_men_and_women_l2920_292066


namespace NUMINAMATH_CALUDE_alex_walking_distance_l2920_292073

/-- Calculates the distance traveled given speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Represents Alex's bike journey -/
structure BikeJourney where
  totalDistance : ℝ
  flatSpeed : ℝ
  flatTime : ℝ
  uphillSpeed : ℝ
  uphillTime : ℝ
  downhillSpeed : ℝ
  downhillTime : ℝ

/-- Calculates the distance Alex had to walk -/
def distanceToWalk (journey : BikeJourney) : ℝ :=
  journey.totalDistance - (distance journey.flatSpeed journey.flatTime +
                           distance journey.uphillSpeed journey.uphillTime +
                           distance journey.downhillSpeed journey.downhillTime)

theorem alex_walking_distance :
  let journey : BikeJourney := {
    totalDistance := 164,
    flatSpeed := 20,
    flatTime := 4.5,
    uphillSpeed := 12,
    uphillTime := 2.5,
    downhillSpeed := 24,
    downhillTime := 1.5
  }
  distanceToWalk journey = 8 := by
  sorry

end NUMINAMATH_CALUDE_alex_walking_distance_l2920_292073


namespace NUMINAMATH_CALUDE_xyz_value_l2920_292020

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 40)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 10) :
  x * y * z = 10 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l2920_292020


namespace NUMINAMATH_CALUDE_larger_number_proof_l2920_292023

theorem larger_number_proof (a b : ℕ+) : 
  (Nat.gcd a b = 20) →
  (Nat.lcm a b = 3640) →
  (13 ∣ Nat.lcm a b) →
  (14 ∣ Nat.lcm a b) →
  max a b = 280 := by
sorry

end NUMINAMATH_CALUDE_larger_number_proof_l2920_292023


namespace NUMINAMATH_CALUDE_minimum_coins_for_all_amounts_l2920_292068

/-- Represents the different types of coins available --/
inductive Coin
  | Penny
  | Nickel
  | Dime
  | Quarter

/-- The value of each coin in cents --/
def coinValue : Coin → ℕ
  | Coin.Penny => 1
  | Coin.Nickel => 5
  | Coin.Dime => 10
  | Coin.Quarter => 25

/-- A list of coins --/
def CoinList := List Coin

/-- Calculates the total value of a list of coins in cents --/
def totalValue (coins : CoinList) : ℕ :=
  coins.foldl (fun acc coin => acc + coinValue coin) 0

/-- Checks if a given amount can be made with a list of coins --/
def canMakeAmount (coins : CoinList) (amount : ℕ) : Prop :=
  ∃ (subset : CoinList), subset.Subset coins ∧ totalValue subset = amount

/-- The main theorem to prove --/
theorem minimum_coins_for_all_amounts :
  ∃ (coins : CoinList),
    coins.length = 11 ∧
    (∀ (amount : ℕ), amount > 0 ∧ amount < 100 → canMakeAmount coins amount) ∧
    (∀ (otherCoins : CoinList),
      (∀ (amount : ℕ), amount > 0 ∧ amount < 100 → canMakeAmount otherCoins amount) →
      otherCoins.length ≥ 11) :=
sorry

end NUMINAMATH_CALUDE_minimum_coins_for_all_amounts_l2920_292068


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2920_292006

theorem quadratic_equation_roots (m n : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + m*x₁ + n = 0 ∧ x₂^2 + m*x₂ + n = 0) →
  (n = 3 - m ∧ (∀ x : ℝ, x^2 + m*x + n = 0 → x < 0) → 2 ≤ m ∧ m < 3) ∧
  (∃ t : ℝ, ∀ m n : ℝ, (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + m*x₁ + n = 0 ∧ x₂^2 + m*x₂ + n = 0) →
    t ≤ (m-1)^2 + (n-1)^2 + (m-n)^2 ∧
    t = 9/8 ∧
    ∀ t' : ℝ, (∀ m n : ℝ, (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + m*x₁ + n = 0 ∧ x₂^2 + m*x₂ + n = 0) →
      t' ≤ (m-1)^2 + (n-1)^2 + (m-n)^2) → t' ≤ t) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2920_292006


namespace NUMINAMATH_CALUDE_cone_base_circumference_l2920_292054

/-- Theorem: For a right circular cone with volume 24π cubic centimeters and height 6 cm, 
    the circumference of its base is 4√3π cm. -/
theorem cone_base_circumference (V : ℝ) (h : ℝ) (r : ℝ) :
  V = 24 * Real.pi ∧ h = 6 ∧ V = (1/3) * Real.pi * r^2 * h → 
  2 * Real.pi * r = 4 * Real.sqrt 3 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_cone_base_circumference_l2920_292054


namespace NUMINAMATH_CALUDE_hair_length_calculation_l2920_292040

theorem hair_length_calculation (initial_length cut_length growth_length : ℕ) 
  (h1 : initial_length = 16)
  (h2 : cut_length = 11)
  (h3 : growth_length = 12) :
  initial_length - cut_length + growth_length = 17 := by
  sorry

end NUMINAMATH_CALUDE_hair_length_calculation_l2920_292040


namespace NUMINAMATH_CALUDE_fraction_equivalence_l2920_292055

theorem fraction_equivalence : 8 / (4 * 25) = 0.8 / (0.4 * 25) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equivalence_l2920_292055


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_inequality_l2920_292048

theorem unique_solution_quadratic_inequality (b : ℝ) : 
  (∃! x : ℝ, |x^2 + 2*b*x + 2*b| ≤ 1) ↔ b = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_inequality_l2920_292048


namespace NUMINAMATH_CALUDE_fifteenth_row_seats_l2920_292033

/-- Represents the number of seats in a given row of the sports palace. -/
def seats_in_row (n : ℕ) : ℕ := 5 + 2 * (n - 1)

/-- Theorem stating that the 15th row of the sports palace has 33 seats. -/
theorem fifteenth_row_seats : seats_in_row 15 = 33 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_row_seats_l2920_292033


namespace NUMINAMATH_CALUDE_smallest_valid_n_l2920_292049

def is_valid_pair (m n x : ℕ+) : Prop :=
  m = 60 ∧ 
  Nat.gcd m n = x + 5 ∧ 
  Nat.lcm m n = x * (x + 5)

theorem smallest_valid_n : 
  ∃ (x : ℕ+), is_valid_pair 60 100 x ∧ 
  ∀ (y : ℕ+) (n : ℕ+), y < x → ¬ is_valid_pair 60 n y :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_n_l2920_292049


namespace NUMINAMATH_CALUDE_total_visitors_is_440_l2920_292000

/-- Represents the survey results from a modern art museum --/
structure SurveyResults where
  totalVisitors : ℕ
  notEnjoyedNotUnderstood : ℕ
  enjoyedAndUnderstood : ℕ
  visitorsBelowFortyRatio : ℚ
  visitorFortyAndAboveRatio : ℚ
  expertRatio : ℚ
  nonExpertRatio : ℚ
  enjoyedAndUnderstoodRatio : ℚ
  fortyAndAboveEnjoyedRatio : ℚ

/-- Theorem stating the total number of visitors based on survey conditions --/
theorem total_visitors_is_440 (survey : SurveyResults) :
  survey.totalVisitors = 440 ∧
  survey.notEnjoyedNotUnderstood = 110 ∧
  survey.enjoyedAndUnderstood = survey.totalVisitors - survey.notEnjoyedNotUnderstood ∧
  survey.visitorsBelowFortyRatio = 2 * survey.visitorFortyAndAboveRatio ∧
  survey.expertRatio = 3/5 ∧
  survey.nonExpertRatio = 2/5 ∧
  survey.enjoyedAndUnderstoodRatio = 3/4 ∧
  survey.fortyAndAboveEnjoyedRatio = 3/5 :=
by sorry

end NUMINAMATH_CALUDE_total_visitors_is_440_l2920_292000


namespace NUMINAMATH_CALUDE_negative_integer_squared_plus_self_equals_twelve_l2920_292036

theorem negative_integer_squared_plus_self_equals_twelve (N : ℤ) : 
  N < 0 → N^2 + N = 12 → N = -4 := by
sorry

end NUMINAMATH_CALUDE_negative_integer_squared_plus_self_equals_twelve_l2920_292036


namespace NUMINAMATH_CALUDE_windows_preference_l2920_292018

theorem windows_preference (total : ℕ) (mac_pref : ℕ) (no_pref : ℕ) 
  (h1 : total = 210)
  (h2 : mac_pref = 60)
  (h3 : no_pref = 90) :
  total - mac_pref - (mac_pref / 3) - no_pref = 40 := by
  sorry

#check windows_preference

end NUMINAMATH_CALUDE_windows_preference_l2920_292018


namespace NUMINAMATH_CALUDE_smallest_square_containing_circle_l2920_292075

theorem smallest_square_containing_circle (r : ℝ) (h : r = 5) : 
  (2 * r) ^ 2 = 100 := by
  sorry

end NUMINAMATH_CALUDE_smallest_square_containing_circle_l2920_292075


namespace NUMINAMATH_CALUDE_last_digit_of_power_tower_plus_one_l2920_292032

theorem last_digit_of_power_tower_plus_one :
  (2^(2^1989) + 1) % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_of_power_tower_plus_one_l2920_292032


namespace NUMINAMATH_CALUDE_rhombus_perimeter_rhombus_perimeter_is_20_l2920_292093

/-- The perimeter of a rhombus whose diagonals are the roots of x^2 - 14x + 48 = 0 -/
theorem rhombus_perimeter : ℝ → Prop :=
  fun p =>
    ∀ (x₁ x₂ : ℝ),
      x₁^2 - 14*x₁ + 48 = 0 →
      x₂^2 - 14*x₂ + 48 = 0 →
      x₁ ≠ x₂ →
      let s := Real.sqrt ((x₁^2 + x₂^2) / 4)
      p = 4 * s

/-- The perimeter of the rhombus is 20 -/
theorem rhombus_perimeter_is_20 : rhombus_perimeter 20 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_rhombus_perimeter_is_20_l2920_292093


namespace NUMINAMATH_CALUDE_agent_percentage_l2920_292059

def total_copies : ℕ := 1000000
def earnings_per_copy : ℚ := 2
def steve_kept_earnings : ℚ := 1620000

theorem agent_percentage : 
  let total_earnings := total_copies * earnings_per_copy
  let agent_earnings := total_earnings - steve_kept_earnings
  (agent_earnings / total_earnings) * 100 = 19 := by sorry

end NUMINAMATH_CALUDE_agent_percentage_l2920_292059


namespace NUMINAMATH_CALUDE_a_divides_b_l2920_292042

theorem a_divides_b (a b : ℕ) (h1 : a > 1) (h2 : b > 1)
  (r : ℕ → ℕ)
  (h3 : ∀ n : ℕ, n > 0 → r n = b^n % a^n)
  (h4 : ∃ N : ℕ, ∀ n : ℕ, n ≥ N → (r n : ℚ) < 2^n / n) :
  a ∣ b :=
by sorry

end NUMINAMATH_CALUDE_a_divides_b_l2920_292042


namespace NUMINAMATH_CALUDE_not_strictly_monotone_sequence_l2920_292005

/-- d(k) denotes the number of natural divisors of a natural number k -/
def d (k : ℕ) : ℕ := (Finset.filter (· ∣ k) (Finset.range (k + 1))).card

/-- The sequence {d(n^2+1)}_{n=n_0}^∞ is not strictly monotone -/
theorem not_strictly_monotone_sequence (n_0 : ℕ) :
  ∃ m n : ℕ, m > n ∧ n ≥ n_0 ∧ d (m^2 + 1) ≤ d (n^2 + 1) :=
sorry

end NUMINAMATH_CALUDE_not_strictly_monotone_sequence_l2920_292005


namespace NUMINAMATH_CALUDE_collinear_vectors_product_l2920_292009

/-- Given two non-collinear vectors i and j in a vector space V over ℝ,
    if AB = i + m*j, AD = n*i + j, m ≠ 1, and points A, B, and D are collinear,
    then mn = 1 -/
theorem collinear_vectors_product (V : Type*) [AddCommGroup V] [Module ℝ V]
  (i j : V) (m n : ℝ) (A B D : V) :
  LinearIndependent ℝ ![i, j] →
  B - A = i + m • j →
  D - A = n • i + j →
  m ≠ 1 →
  ∃ (k : ℝ), B - A = k • (D - A) →
  m * n = 1 := by
  sorry

end NUMINAMATH_CALUDE_collinear_vectors_product_l2920_292009


namespace NUMINAMATH_CALUDE_prime_sequence_l2920_292089

theorem prime_sequence (A : ℕ) : 
  Nat.Prime A ∧ 
  Nat.Prime (A - 4) ∧ 
  Nat.Prime (A - 6) ∧ 
  Nat.Prime (A - 12) ∧ 
  Nat.Prime (A - 18) → 
  A = 23 := by
sorry

end NUMINAMATH_CALUDE_prime_sequence_l2920_292089


namespace NUMINAMATH_CALUDE_intersection_complement_eq_interval_l2920_292039

open Set

/-- Given sets A and B, prove that their intersection with the complement of B is [1, 3) -/
theorem intersection_complement_eq_interval :
  let A : Set ℝ := {x | x - 1 ≥ 0}
  let B : Set ℝ := {x | 3 / x ≤ 1}
  A ∩ (univ \ B) = Icc 1 3 ∩ Iio 3 := by sorry

end NUMINAMATH_CALUDE_intersection_complement_eq_interval_l2920_292039


namespace NUMINAMATH_CALUDE_complex_modulus_example_l2920_292064

theorem complex_modulus_example : Complex.abs (-3 - (5/4)*Complex.I) = 13/4 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_example_l2920_292064


namespace NUMINAMATH_CALUDE_solve_for_x_l2920_292082

theorem solve_for_x (y : ℝ) (h1 : y = 1) (h2 : 4 * x - 2 * y + 3 = 3 * x + 3 * y) : x = 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_x_l2920_292082


namespace NUMINAMATH_CALUDE_song_game_theorem_l2920_292060

/-- Represents the "Guess the Song Title" game -/
structure SongGame where
  /-- Probability of passing each level -/
  pass_prob : Fin 3 → ℚ
  /-- Probability of continuing to next level -/
  continue_prob : ℚ
  /-- Reward for passing each level -/
  reward : Fin 3 → ℕ

/-- The specific game instance as described in the problem -/
def game : SongGame :=
  { pass_prob := λ i => [3/4, 2/3, 1/2].get i
    continue_prob := 1/2
    reward := λ i => [1000, 2000, 3000].get i }

/-- Probability of passing first level but receiving zero reward -/
def prob_pass_first_zero_reward (g : SongGame) : ℚ :=
  g.pass_prob 0 * g.continue_prob * (1 - g.pass_prob 1) +
  g.pass_prob 0 * g.continue_prob * g.pass_prob 1 * g.continue_prob * (1 - g.pass_prob 2)

/-- Expected value of total reward -/
def expected_reward (g : SongGame) : ℚ :=
  g.pass_prob 0 * (1 - g.continue_prob) * g.reward 0 +
  g.pass_prob 0 * g.continue_prob * g.pass_prob 1 * (1 - g.continue_prob) * (g.reward 0 + g.reward 1) +
  g.pass_prob 0 * g.continue_prob * g.pass_prob 1 * g.continue_prob * g.pass_prob 2 * (g.reward 0 + g.reward 1 + g.reward 2)

theorem song_game_theorem (g : SongGame) :
  prob_pass_first_zero_reward g = 3/16 ∧ expected_reward g = 1125 :=
sorry

end NUMINAMATH_CALUDE_song_game_theorem_l2920_292060


namespace NUMINAMATH_CALUDE_total_games_in_league_l2920_292098

theorem total_games_in_league (n : ℕ) (h : n = 12) : 
  (n * (n - 1)) / 2 = 66 := by
  sorry

end NUMINAMATH_CALUDE_total_games_in_league_l2920_292098


namespace NUMINAMATH_CALUDE_line_equation_through_point_with_slope_angle_l2920_292087

/-- The equation of a line passing through (2, 3) with a slope angle of 135° -/
theorem line_equation_through_point_with_slope_angle (x y : ℝ) :
  (x + y - 5 = 0) ↔ 
  (∃ (m : ℝ), m = Real.tan (135 * π / 180) ∧ y - 3 = m * (x - 2)) := by
  sorry

end NUMINAMATH_CALUDE_line_equation_through_point_with_slope_angle_l2920_292087


namespace NUMINAMATH_CALUDE_quadratic_inequality_l2920_292022

/-- Given a quadratic function f(x) = x^2 + 4x + c, prove that f(1) > c > f(-2) -/
theorem quadratic_inequality (c : ℝ) : let f : ℝ → ℝ := λ x ↦ x^2 + 4*x + c
  f 1 > c ∧ c > f (-2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l2920_292022


namespace NUMINAMATH_CALUDE_tenth_fibonacci_is_89_l2920_292028

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | n + 2 => fibonacci n + fibonacci (n + 1)

theorem tenth_fibonacci_is_89 : fibonacci 9 = 89 := by
  sorry

end NUMINAMATH_CALUDE_tenth_fibonacci_is_89_l2920_292028


namespace NUMINAMATH_CALUDE_coffee_consumption_ratio_l2920_292010

/-- Given that Brayan drinks 4 cups of coffee per hour and they drink a total of 30 cups of coffee
    together in 5 hours, prove that the ratio of the amount of coffee Brayan drinks to the amount
    Ivory drinks is 2:1. -/
theorem coffee_consumption_ratio :
  let brayan_per_hour : ℚ := 4
  let total_cups : ℚ := 30
  let total_hours : ℚ := 5
  let ivory_per_hour : ℚ := total_cups / total_hours - brayan_per_hour
  brayan_per_hour / ivory_per_hour = 2 := by
  sorry

end NUMINAMATH_CALUDE_coffee_consumption_ratio_l2920_292010


namespace NUMINAMATH_CALUDE_lines_parallel_iff_m_eq_neg_two_l2920_292014

/-- Represents a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- The first line: 2x + my - 2m + 4 = 0 -/
def line1 (m : ℝ) : Line :=
  { a := 2, b := m, c := -2*m + 4 }

/-- The second line: mx + 2y - m + 2 = 0 -/
def line2 (m : ℝ) : Line :=
  { a := m, b := 2, c := -m + 2 }

/-- Theorem stating that the lines are parallel if and only if m = -2 -/
theorem lines_parallel_iff_m_eq_neg_two :
  ∀ m : ℝ, parallel (line1 m) (line2 m) ↔ m = -2 := by
  sorry

end NUMINAMATH_CALUDE_lines_parallel_iff_m_eq_neg_two_l2920_292014


namespace NUMINAMATH_CALUDE_factorization_problem1_l2920_292046

theorem factorization_problem1 (a m : ℝ) : 2 * a * m^2 - 8 * a = 2 * a * (m + 2) * (m - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_problem1_l2920_292046


namespace NUMINAMATH_CALUDE_afternoon_eggs_l2920_292051

theorem afternoon_eggs (total_eggs day_eggs morning_eggs : ℕ) 
  (h1 : total_eggs = 1339)
  (h2 : morning_eggs = 816) :
  total_eggs - morning_eggs = 523 := by
  sorry

end NUMINAMATH_CALUDE_afternoon_eggs_l2920_292051


namespace NUMINAMATH_CALUDE_expensive_rock_cost_l2920_292085

/-- Given a mixture of two types of rock, prove the cost of the more expensive rock -/
theorem expensive_rock_cost 
  (total_weight : ℝ) 
  (total_cost : ℝ) 
  (cheap_rock_cost : ℝ) 
  (cheap_rock_weight : ℝ) 
  (expensive_rock_weight : ℝ)
  (h1 : total_weight = 24)
  (h2 : total_cost = 800)
  (h3 : cheap_rock_cost = 30)
  (h4 : cheap_rock_weight = 8)
  (h5 : expensive_rock_weight = 8)
  : (total_cost - cheap_rock_cost * cheap_rock_weight) / (total_weight - cheap_rock_weight) = 35 := by
  sorry

end NUMINAMATH_CALUDE_expensive_rock_cost_l2920_292085


namespace NUMINAMATH_CALUDE_carousel_revolutions_l2920_292091

theorem carousel_revolutions (r₁ r₂ : ℝ) (n₁ : ℕ) :
  r₁ = 30 →
  r₂ = 10 →
  n₁ = 40 →
  r₁ * n₁ = r₂ * (120 : ℕ) :=
by sorry

end NUMINAMATH_CALUDE_carousel_revolutions_l2920_292091


namespace NUMINAMATH_CALUDE_symmetric_line_equation_l2920_292008

/-- Given two lines in a plane, this function returns the line that is symmetric to the first line with respect to the second line. -/
def symmetricLine (l1 l2 : ℝ → ℝ → Prop) : ℝ → ℝ → Prop :=
  sorry

/-- The line y = 2x + 1 -/
def line1 : ℝ → ℝ → Prop :=
  fun x y ↦ y = 2 * x + 1

/-- The line y = x - 2 -/
def line2 : ℝ → ℝ → Prop :=
  fun x y ↦ y = x - 2

/-- The line x - 2y - 7 = 0 -/
def lineL : ℝ → ℝ → Prop :=
  fun x y ↦ x - 2 * y - 7 = 0

theorem symmetric_line_equation :
  symmetricLine line1 line2 = lineL :=
sorry

end NUMINAMATH_CALUDE_symmetric_line_equation_l2920_292008


namespace NUMINAMATH_CALUDE_principal_is_15000_l2920_292062

/-- Calculates the principal amount given simple interest, rate, and time -/
def calculate_principal (interest : ℚ) (rate : ℚ) (time : ℚ) : ℚ :=
  (interest * 100) / (rate * time)

/-- Theorem: Given the specified conditions, the principal sum is 15000 -/
theorem principal_is_15000 :
  let interest : ℚ := 2700
  let rate : ℚ := 6
  let time : ℚ := 3
  calculate_principal interest rate time = 15000 := by
  sorry

end NUMINAMATH_CALUDE_principal_is_15000_l2920_292062


namespace NUMINAMATH_CALUDE_otimes_two_one_l2920_292086

-- Define the new operation ⊗
def otimes (a b : ℝ) : ℝ := a^2 - b

-- Theorem statement
theorem otimes_two_one : otimes 2 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_otimes_two_one_l2920_292086


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2920_292061

-- Define the set M
def M : Set ℝ := {α | ∃ k : ℤ, α = k * 90 - 36}

-- Define the set N
def N : Set ℝ := {α | -180 < α ∧ α < 180}

-- Define the intersection set
def intersection : Set ℝ := {-36, 54, -126, 144}

-- Theorem statement
theorem intersection_of_M_and_N : M ∩ N = intersection := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2920_292061


namespace NUMINAMATH_CALUDE_other_number_given_hcf_lcm_and_one_number_l2920_292041

theorem other_number_given_hcf_lcm_and_one_number 
  (a b : ℕ+) 
  (hcf : Nat.gcd a b = 12)
  (lcm : Nat.lcm a b = 396)
  (a_val : a = 24) :
  b = 198 := by
  sorry

end NUMINAMATH_CALUDE_other_number_given_hcf_lcm_and_one_number_l2920_292041


namespace NUMINAMATH_CALUDE_estimate_fish_population_l2920_292074

/-- Estimate the number of fish in a pond using the mark and recapture method. -/
theorem estimate_fish_population (initial_marked : ℕ) (recapture_total : ℕ) (recapture_marked : ℕ) 
  (h1 : initial_marked = 20)
  (h2 : recapture_total = 40)
  (h3 : recapture_marked = 2) :
  (initial_marked * recapture_total) / recapture_marked = 400 := by
  sorry

#check estimate_fish_population

end NUMINAMATH_CALUDE_estimate_fish_population_l2920_292074


namespace NUMINAMATH_CALUDE_distance_circle_center_to_point_l2920_292099

/-- The distance between the center of the circle with equation x^2 + y^2 = 6x - 8y + 24
    and the point (-3, 4) is 10. -/
theorem distance_circle_center_to_point :
  let circle_eq : ℝ → ℝ → Prop := λ x y => x^2 + y^2 = 6*x - 8*y + 24
  let center : ℝ × ℝ := (3, -4)
  let point : ℝ × ℝ := (-3, 4)
  (∃ (x y : ℝ), circle_eq x y) →
  Real.sqrt ((center.1 - point.1)^2 + (center.2 - point.2)^2) = 10 :=
by sorry

end NUMINAMATH_CALUDE_distance_circle_center_to_point_l2920_292099


namespace NUMINAMATH_CALUDE_counterfeit_coin_determination_l2920_292092

/-- Represents the result of a weighing -/
inductive WeighingResult
  | Equal : WeighingResult
  | LeftHeavier : WeighingResult
  | RightHeavier : WeighingResult

/-- Represents a group of coins -/
structure CoinGroup where
  size : Nat
  containsCounterfeit : Bool

/-- Represents a weighing operation -/
def weighing (left right : CoinGroup) : WeighingResult :=
  sorry

/-- The main theorem stating that it's possible to determine if counterfeit coins are heavier or lighter -/
theorem counterfeit_coin_determination :
  ∀ (coins : List CoinGroup),
    coins.length = 3 →
    (coins.map CoinGroup.size).sum = 103 →
    (coins.filter CoinGroup.containsCounterfeit).length = 2 →
    ∃ (w₁ w₂ w₃ : CoinGroup × CoinGroup),
      (∀ g₁ g₂, weighing g₁ g₂ = WeighingResult.Equal → g₁.containsCounterfeit = g₂.containsCounterfeit) →
      let r₁ := weighing w₁.1 w₁.2
      let r₂ := weighing w₂.1 w₂.2
      let r₃ := weighing w₃.1 w₃.2
      (r₁ ≠ WeighingResult.Equal ∨ r₂ ≠ WeighingResult.Equal ∨ r₃ ≠ WeighingResult.Equal) :=
by
  sorry

end NUMINAMATH_CALUDE_counterfeit_coin_determination_l2920_292092


namespace NUMINAMATH_CALUDE_cricket_team_age_difference_l2920_292076

theorem cricket_team_age_difference (team_size : ℕ) (avg_age : ℝ) (captain_age : ℝ) (keeper_age_diff : ℝ) :
  team_size = 11 →
  avg_age = 25 →
  captain_age = 28 →
  keeper_age_diff = 3 →
  let total_age := avg_age * team_size
  let keeper_age := captain_age + keeper_age_diff
  let remaining_players := team_size - 2
  let remaining_age := total_age - (captain_age + keeper_age)
  let remaining_avg := remaining_age / remaining_players
  avg_age - remaining_avg = 1 := by sorry

end NUMINAMATH_CALUDE_cricket_team_age_difference_l2920_292076


namespace NUMINAMATH_CALUDE_burj_khalifa_height_l2920_292056

theorem burj_khalifa_height (sears_height burj_difference : ℕ) 
  (h1 : sears_height = 527)
  (h2 : burj_difference = 303) : 
  sears_height + burj_difference = 830 := by
sorry

end NUMINAMATH_CALUDE_burj_khalifa_height_l2920_292056


namespace NUMINAMATH_CALUDE_beth_double_age_in_8_years_l2920_292019

/-- The number of years until Beth is twice her sister's age -/
def years_until_double_age (beth_age : ℕ) (sister_age : ℕ) : ℕ :=
  beth_age + sister_age

theorem beth_double_age_in_8_years (beth_age : ℕ) (sister_age : ℕ) 
  (h1 : beth_age = 18) (h2 : sister_age = 5) :
  years_until_double_age beth_age sister_age = 8 :=
sorry

#check beth_double_age_in_8_years

end NUMINAMATH_CALUDE_beth_double_age_in_8_years_l2920_292019


namespace NUMINAMATH_CALUDE_inscribed_circle_theorem_l2920_292017

theorem inscribed_circle_theorem (PQ PR : ℝ) (h_PQ : PQ = 6) (h_PR : PR = 8) :
  let QR := Real.sqrt (PQ^2 + PR^2)
  let s := (PQ + PR + QR) / 2
  let A := PQ * PR / 2
  let r := A / s
  let x := QR - 2*r
  x = 6 := by sorry

end NUMINAMATH_CALUDE_inscribed_circle_theorem_l2920_292017


namespace NUMINAMATH_CALUDE_equation_solutions_l2920_292069

theorem equation_solutions : 
  let f (x : ℝ) := (x - 1)^3 * (x - 2)^3 * (x - 3)^3 * (x - 4)^3 / ((x - 2) * (x - 4) * (x - 2)^2)
  ∀ x : ℝ, f x = 64 ↔ x = 2 + Real.sqrt 5 ∨ x = 2 - Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2920_292069


namespace NUMINAMATH_CALUDE_donovan_percentage_l2920_292067

/-- Calculates the weighted percentage of correct answers for a math test -/
def weighted_percentage (
  mc_total : ℕ) (mc_correct : ℕ) (mc_points : ℕ)
  (sa_total : ℕ) (sa_correct : ℕ) (sa_partial : ℕ) (sa_points : ℕ)
  (essay_total : ℕ) (essay_correct : ℕ) (essay_points : ℕ) : ℚ :=
  let total_possible := mc_total * mc_points + sa_total * sa_points + essay_total * essay_points
  let total_earned := mc_correct * mc_points + sa_correct * sa_points + sa_partial * (sa_points / 2) + essay_correct * essay_points
  (total_earned : ℚ) / total_possible * 100

/-- Theorem stating that Donovan's weighted percentage is 68.75% -/
theorem donovan_percentage :
  weighted_percentage 25 20 2 20 10 5 4 3 2 10 = 68.75 := by
  sorry

end NUMINAMATH_CALUDE_donovan_percentage_l2920_292067


namespace NUMINAMATH_CALUDE_sum_of_first_10_common_elements_l2920_292071

/-- Arithmetic progression with first term 4 and common difference 3 -/
def arithmetic_progression (n : ℕ) : ℕ := 4 + 3 * n

/-- Geometric progression with first term 20 and common ratio 2 -/
def geometric_progression (k : ℕ) : ℕ := 20 * 2^k

/-- Common elements between the arithmetic and geometric progressions -/
def common_elements (n : ℕ) : Prop :=
  ∃ k : ℕ, arithmetic_progression n = geometric_progression k

/-- The sum of the first 10 common elements -/
def sum_of_common_elements : ℕ := 13981000

/-- Theorem stating that the sum of the first 10 common elements is 13981000 -/
theorem sum_of_first_10_common_elements :
  sum_of_common_elements = 13981000 := by sorry

end NUMINAMATH_CALUDE_sum_of_first_10_common_elements_l2920_292071


namespace NUMINAMATH_CALUDE_apple_sellers_average_prices_l2920_292078

/-- Represents the sales data for a fruit seller --/
structure FruitSeller where
  morning_price : ℚ
  afternoon_price : ℚ
  morning_quantity : ℚ
  afternoon_quantity : ℚ

/-- Calculates the average price per apple for a fruit seller --/
def average_price (seller : FruitSeller) : ℚ :=
  (seller.morning_price * seller.morning_quantity + seller.afternoon_price * seller.afternoon_quantity) /
  (seller.morning_quantity + seller.afternoon_quantity)

theorem apple_sellers_average_prices
  (john bill george : FruitSeller)
  (h_morning_price : john.morning_price = bill.morning_price ∧ bill.morning_price = george.morning_price ∧ george.morning_price = 5/2)
  (h_afternoon_price : john.afternoon_price = bill.afternoon_price ∧ bill.afternoon_price = george.afternoon_price ∧ george.afternoon_price = 5/3)
  (h_john_quantities : john.morning_quantity = john.afternoon_quantity)
  (h_bill_revenue : bill.morning_price * bill.morning_quantity = bill.afternoon_price * bill.afternoon_quantity)
  (h_george_ratio : george.morning_quantity / george.afternoon_quantity = (5/3) / (5/2)) :
  average_price john = 25/12 ∧ average_price bill = 2 ∧ average_price george = 2 := by
  sorry


end NUMINAMATH_CALUDE_apple_sellers_average_prices_l2920_292078


namespace NUMINAMATH_CALUDE_shortest_tangent_length_l2920_292063

-- Define the circles
def C₁ (x y : ℝ) : Prop := (x - 12)^2 + y^2 = 64
def C₂ (x y : ℝ) : Prop := (x + 18)^2 + y^2 = 100

-- Define the tangent line segment
def is_tangent_to (P Q : ℝ × ℝ) : Prop :=
  C₁ P.1 P.2 ∧ C₂ Q.1 Q.2 ∧
  ∀ R : ℝ × ℝ, (C₁ R.1 R.2 ∨ C₂ R.1 R.2) → Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) ≤ Real.sqrt ((P.1 - R.1)^2 + (P.2 - R.2)^2) + Real.sqrt ((Q.1 - R.1)^2 + (Q.2 - R.2)^2)

-- Theorem statement
theorem shortest_tangent_length :
  ∃ P Q : ℝ × ℝ, is_tangent_to P Q ∧
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = 190 / 3 ∧
  ∀ P' Q' : ℝ × ℝ, is_tangent_to P' Q' →
    Real.sqrt ((P'.1 - Q'.1)^2 + (P'.2 - Q'.2)^2) ≥ 190 / 3 := by
  sorry

end NUMINAMATH_CALUDE_shortest_tangent_length_l2920_292063


namespace NUMINAMATH_CALUDE_pablo_blocks_l2920_292047

theorem pablo_blocks (stack1 stack2 stack3 stack4 : ℕ) : 
  stack1 = 5 →
  stack3 = stack2 - 5 →
  stack4 = stack3 + 5 →
  stack1 + stack2 + stack3 + stack4 = 21 →
  stack2 - stack1 = 2 :=
by sorry

end NUMINAMATH_CALUDE_pablo_blocks_l2920_292047


namespace NUMINAMATH_CALUDE_baso4_percentage_yield_is_90_percent_l2920_292038

-- Define the molar quantities
def NaOH_moles : ℚ := 3
def H2SO4_moles : ℚ := 2
def BaCl2_moles : ℚ := 1
def BaSO4_actual_yield : ℚ := 9/10

-- Define the reaction stoichiometry
def NaOH_to_Na2SO4_ratio : ℚ := 2
def H2SO4_to_Na2SO4_ratio : ℚ := 1
def Na2SO4_to_BaSO4_ratio : ℚ := 1
def BaCl2_to_BaSO4_ratio : ℚ := 1

-- Define the theoretical yield calculation
def theoretical_yield (limiting_reactant_moles ratio : ℚ) : ℚ :=
  limiting_reactant_moles / ratio

-- Define the percentage yield calculation
def percentage_yield (actual_yield theoretical_yield : ℚ) : ℚ :=
  actual_yield / theoretical_yield * 100

-- Theorem to prove
theorem baso4_percentage_yield_is_90_percent :
  let na2so4_yield_from_naoh := theoretical_yield NaOH_moles NaOH_to_Na2SO4_ratio
  let na2so4_yield_from_h2so4 := theoretical_yield H2SO4_moles H2SO4_to_Na2SO4_ratio
  let na2so4_actual_yield := min na2so4_yield_from_naoh na2so4_yield_from_h2so4
  let baso4_theoretical_yield := min na2so4_actual_yield BaCl2_moles
  percentage_yield BaSO4_actual_yield baso4_theoretical_yield = 90 :=
by sorry

end NUMINAMATH_CALUDE_baso4_percentage_yield_is_90_percent_l2920_292038


namespace NUMINAMATH_CALUDE_range_of_a_for_f_with_two_zeros_l2920_292081

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 3 * a * x + 3 * a - 5

-- State the theorem
theorem range_of_a_for_f_with_two_zeros :
  (∃ a : ℝ, ∀ x : ℝ, (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0)) →
  (∀ a : ℝ, (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0) → 1 ≤ a ∧ a ≤ 5) ∧
  (∀ a : ℝ, 1 ≤ a ∧ a ≤ 5 → ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_for_f_with_two_zeros_l2920_292081


namespace NUMINAMATH_CALUDE_adjacent_angle_measure_l2920_292090

-- Define the angle type
def Angle : Type := ℝ

-- Define parallel lines
def ParallelLines (m n : Line) : Prop := sorry

-- Define a transversal line
def Transversal (p m n : Line) : Prop := sorry

-- Define the measure of an angle
def AngleMeasure (θ : Angle) : ℝ := sorry

-- Define supplementary angles
def Supplementary (θ₁ θ₂ : Angle) : Prop :=
  AngleMeasure θ₁ + AngleMeasure θ₂ = 180

-- Theorem statement
theorem adjacent_angle_measure
  (m n p : Line)
  (θ₁ θ₂ : Angle)
  (h_parallel : ParallelLines m n)
  (h_transversal : Transversal p m n)
  (h_internal : AngleMeasure θ₁ = 70)
  (h_supplementary : Supplementary θ₁ θ₂) :
  AngleMeasure θ₂ = 110 :=
sorry

end NUMINAMATH_CALUDE_adjacent_angle_measure_l2920_292090


namespace NUMINAMATH_CALUDE_total_earnings_before_car_purchase_l2920_292026

def monthly_income : ℕ := 4000
def monthly_savings : ℕ := 500
def car_cost : ℕ := 45000

theorem total_earnings_before_car_purchase :
  (car_cost / monthly_savings) * monthly_income = 360000 := by
  sorry

end NUMINAMATH_CALUDE_total_earnings_before_car_purchase_l2920_292026


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2920_292015

theorem inequality_solution_set (x : ℝ) :
  (Set.Ioo (-1 : ℝ) 3) = {x | (3 - x) * (1 + x) > 0} := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2920_292015


namespace NUMINAMATH_CALUDE_combined_weight_theorem_l2920_292037

/-- The weight of a regular dinosaur in pounds -/
def regular_dino_weight : ℕ := 800

/-- The number of regular dinosaurs -/
def num_regular_dinos : ℕ := 5

/-- The additional weight of Barney compared to the regular dinosaurs in pounds -/
def barney_extra_weight : ℕ := 1500

/-- The combined weight of Barney and the regular dinosaurs in pounds -/
def total_weight : ℕ := regular_dino_weight * num_regular_dinos + barney_extra_weight + regular_dino_weight * num_regular_dinos

theorem combined_weight_theorem : total_weight = 9500 := by
  sorry

end NUMINAMATH_CALUDE_combined_weight_theorem_l2920_292037


namespace NUMINAMATH_CALUDE_max_value_complex_expression_l2920_292043

theorem max_value_complex_expression (x y : ℂ) :
  ∃ (M : ℝ), M = (5 * Real.sqrt 2) / 2 ∧
  Complex.abs (3 * x + 4 * y) / Real.sqrt (Complex.abs x ^ 2 + Complex.abs y ^ 2 + Complex.abs (x ^ 2 + y ^ 2)) ≤ M ∧
  ∃ (x₀ y₀ : ℂ), Complex.abs (3 * x₀ + 4 * y₀) / Real.sqrt (Complex.abs x₀ ^ 2 + Complex.abs y₀ ^ 2 + Complex.abs (x₀ ^ 2 + y₀ ^ 2)) = M :=
by sorry

end NUMINAMATH_CALUDE_max_value_complex_expression_l2920_292043


namespace NUMINAMATH_CALUDE_queens_free_subgrid_l2920_292011

/-- Represents a chessboard with queens -/
structure Chessboard :=
  (size : Nat)
  (queens : Nat)

/-- Theorem: On an 8x8 chessboard with 12 queens, there always exist four rows and four columns
    such that none of the 16 cells at their intersections contain a queen -/
theorem queens_free_subgrid (board : Chessboard) 
  (h1 : board.size = 8) 
  (h2 : board.queens = 12) : 
  ∃ (rows columns : Finset Nat), 
    rows.card = 4 ∧ 
    columns.card = 4 ∧ 
    (∀ r ∈ rows, ∀ c ∈ columns, 
      ¬∃ (queen : Nat × Nat), queen.1 = r ∧ queen.2 = c) :=
sorry

end NUMINAMATH_CALUDE_queens_free_subgrid_l2920_292011


namespace NUMINAMATH_CALUDE_coin_distribution_six_boxes_l2920_292097

def coinDistribution (n : ℕ) : ℕ :=
  match n with
  | 0 => 0
  | 1 => 1
  | m + 1 => 2 * coinDistribution m

theorem coin_distribution_six_boxes :
  coinDistribution 6 = 32 := by
  sorry

end NUMINAMATH_CALUDE_coin_distribution_six_boxes_l2920_292097


namespace NUMINAMATH_CALUDE_ellipse_equation_l2920_292077

/-- The standard equation of an ellipse with given properties -/
theorem ellipse_equation (b c : ℝ) (h1 : b = 3) (h2 : c = 2) :
  ∃ a : ℝ, a^2 = b^2 + c^2 ∧ 
  (∀ x y : ℝ, (x^2 / b^2) + (y^2 / a^2) = 1 ↔ 
    x^2 / 9 + y^2 / 13 = 1) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l2920_292077


namespace NUMINAMATH_CALUDE_problem_solution_l2920_292007

theorem problem_solution (m : ℝ) (h : m + 1/m = 10) :
  m^2 + 1/m^2 + m^3 + 1/m^3 + 4 = 1072 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2920_292007


namespace NUMINAMATH_CALUDE_pyramid_volume_l2920_292013

/-- The volume of a pyramid with a rectangular base and given edge length --/
theorem pyramid_volume (base_length base_width edge_length : ℝ) 
  (h_base_length : base_length = 6)
  (h_base_width : base_width = 8)
  (h_edge_length : edge_length = 13) : 
  (1 / 3 : ℝ) * base_length * base_width * 
    Real.sqrt (edge_length^2 - ((base_length^2 + base_width^2) / 4)) = 192 := by
  sorry

#check pyramid_volume

end NUMINAMATH_CALUDE_pyramid_volume_l2920_292013


namespace NUMINAMATH_CALUDE_final_expression_l2920_292025

theorem final_expression (b : ℚ) : 
  (3 * b + 6 - 5 * b) / 3 = -2/3 * b + 2 := by sorry

end NUMINAMATH_CALUDE_final_expression_l2920_292025


namespace NUMINAMATH_CALUDE_commission_percentage_l2920_292052

/-- Proves that the commission percentage for the first $500 is 20% given the conditions --/
theorem commission_percentage (x : ℝ) : 
  let total_sale := 800
  let commission_over_500 := 0.25
  let total_commission_percentage := 0.21875
  (x / 100 * 500 + commission_over_500 * (total_sale - 500)) / total_sale = total_commission_percentage →
  x = 20 := by
sorry

end NUMINAMATH_CALUDE_commission_percentage_l2920_292052
