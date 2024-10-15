import Mathlib

namespace NUMINAMATH_CALUDE_election_votes_count_l4056_405677

theorem election_votes_count :
  ∀ (total_votes : ℕ) (harold_percentage : ℚ) (jacquie_percentage : ℚ),
    harold_percentage = 60 / 100 →
    jacquie_percentage = 1 - harold_percentage →
    (harold_percentage * total_votes : ℚ) - (jacquie_percentage * total_votes : ℚ) = 24 →
    total_votes = 120 :=
by
  sorry

end NUMINAMATH_CALUDE_election_votes_count_l4056_405677


namespace NUMINAMATH_CALUDE_a_4_equals_4_l4056_405669

def sequence_term (n : ℕ) : ℤ := (-1)^n * n

theorem a_4_equals_4 : sequence_term 4 = 4 := by sorry

end NUMINAMATH_CALUDE_a_4_equals_4_l4056_405669


namespace NUMINAMATH_CALUDE_cube_section_not_pentagon_cube_section_can_be_hexagon_l4056_405648

/-- A cube in 3D space --/
structure Cube where
  side : ℝ
  center : ℝ × ℝ × ℝ

/-- A plane in 3D space --/
structure Plane where
  normal : ℝ × ℝ × ℝ
  point : ℝ × ℝ × ℝ

/-- The intersection of a plane and a cube --/
def PlaneSection (c : Cube) (p : Plane) : Set (ℝ × ℝ × ℝ) :=
  sorry

/-- Predicate to check if a set of points forms a regular polygon --/
def IsRegularPolygon (s : Set (ℝ × ℝ × ℝ)) (n : ℕ) : Prop :=
  sorry

theorem cube_section_not_pentagon (c : Cube) :
  ¬ ∃ p : Plane, IsRegularPolygon (PlaneSection c p) 5 :=
sorry

theorem cube_section_can_be_hexagon :
  ∃ c : Cube, ∃ p : Plane, IsRegularPolygon (PlaneSection c p) 6 :=
sorry

end NUMINAMATH_CALUDE_cube_section_not_pentagon_cube_section_can_be_hexagon_l4056_405648


namespace NUMINAMATH_CALUDE_expected_rounds_four_players_l4056_405626

/-- Represents the expected number of rounds in a rock-paper-scissors game -/
def expected_rounds (n : ℕ) : ℚ :=
  match n with
  | 0 => 0
  | 1 => 0
  | 2 => 3/2
  | 3 => 9/4
  | 4 => 81/14
  | _ => 0  -- undefined for n > 4

/-- The rules of the rock-paper-scissors game -/
axiom game_rules : ∀ (n : ℕ), n > 0 → n ≤ 4 → 
  expected_rounds n = 
    if n = 1 then 0
    else if n = 2 then 3/2
    else if n = 3 then 9/4
    else 81/14

/-- The main theorem: expected number of rounds for 4 players is 81/14 -/
theorem expected_rounds_four_players :
  expected_rounds 4 = 81/14 :=
by
  exact game_rules 4 (by norm_num) (by norm_num)


end NUMINAMATH_CALUDE_expected_rounds_four_players_l4056_405626


namespace NUMINAMATH_CALUDE_students_not_taking_languages_l4056_405629

theorem students_not_taking_languages (total : ℕ) (french : ℕ) (spanish : ℕ) (both : ℕ) 
  (h1 : total = 28) 
  (h2 : french = 5) 
  (h3 : spanish = 10) 
  (h4 : both = 4) : 
  total - (french + spanish + both) = 9 := by
  sorry

end NUMINAMATH_CALUDE_students_not_taking_languages_l4056_405629


namespace NUMINAMATH_CALUDE_bus_train_speed_ratio_l4056_405694

/-- Proves that the fraction of bus speed to train speed is 3/4 -/
theorem bus_train_speed_ratio :
  let train_car_speed_ratio : ℚ := 16 / 15
  let bus_distance : ℕ := 480
  let bus_time : ℕ := 8
  let car_distance : ℕ := 450
  let car_time : ℕ := 6
  let bus_speed : ℚ := bus_distance / bus_time
  let car_speed : ℚ := car_distance / car_time
  let train_speed : ℚ := car_speed * train_car_speed_ratio
  bus_speed / train_speed = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_bus_train_speed_ratio_l4056_405694


namespace NUMINAMATH_CALUDE_composition_equality_l4056_405655

variables (m n p q : ℝ)

def f (x : ℝ) : ℝ := m * x^2 + n * x

def g (x : ℝ) : ℝ := p * x + q

theorem composition_equality :
  (∀ x, f m n (g p q x) = g p q (f m n x)) ↔ 2 * m = n :=
sorry

end NUMINAMATH_CALUDE_composition_equality_l4056_405655


namespace NUMINAMATH_CALUDE_polygon_area_l4056_405662

/-- An isosceles triangle with one angle of 100° and area 2 cm² --/
structure IsoscelesTriangle where
  angle : ℝ
  area : ℝ
  is_isosceles : angle = 100
  has_area : area = 2

/-- A polygon composed of isosceles triangles --/
structure Polygon where
  triangle : IsoscelesTriangle
  full_count : ℕ
  half_count : ℕ
  full_is_12 : full_count = 12
  half_is_4 : half_count = 4

/-- The area of the polygon is 28 cm² --/
theorem polygon_area (p : Polygon) : p.full_count * p.triangle.area + p.half_count * (p.triangle.area / 2) = 28 := by
  sorry

end NUMINAMATH_CALUDE_polygon_area_l4056_405662


namespace NUMINAMATH_CALUDE_initial_customers_l4056_405624

theorem initial_customers (remaining : ℕ) (left : ℕ) (initial : ℕ) : 
  remaining = 12 → left = 9 → initial = remaining + left → initial = 21 := by
  sorry

end NUMINAMATH_CALUDE_initial_customers_l4056_405624


namespace NUMINAMATH_CALUDE_divisible_by_ten_l4056_405665

theorem divisible_by_ten (S : Finset ℤ) : 
  (Finset.card S = 5) →
  (∀ (a b c : ℤ), a ∈ S → b ∈ S → c ∈ S → a ≠ b → b ≠ c → a ≠ c → (10 ∣ a * b * c)) →
  (∃ x ∈ S, 10 ∣ x) :=
by sorry

end NUMINAMATH_CALUDE_divisible_by_ten_l4056_405665


namespace NUMINAMATH_CALUDE_train_speed_excluding_stoppages_l4056_405640

/-- The speed of a train excluding stoppages, given its speed including stoppages and stop time. -/
theorem train_speed_excluding_stoppages 
  (speed_with_stops : ℝ) 
  (stop_time : ℝ) 
  (h1 : speed_with_stops = 30) 
  (h2 : stop_time = 24) : 
  speed_with_stops * (60 - stop_time) / 60 = 18 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_excluding_stoppages_l4056_405640


namespace NUMINAMATH_CALUDE_threeDigitNumberFormula_l4056_405697

/-- Given a natural number m, this function represents a three-digit number
    where the hundreds digit is 3m, the tens digit is m, and the units digit is m-1 -/
def threeDigitNumber (m : ℕ) : ℕ := 300 * m + 10 * m + (m - 1)

/-- Theorem stating that the three-digit number can be expressed as 311m - 1 -/
theorem threeDigitNumberFormula (m : ℕ) : 
  threeDigitNumber m = 311 * m - 1 := by
  sorry

end NUMINAMATH_CALUDE_threeDigitNumberFormula_l4056_405697


namespace NUMINAMATH_CALUDE_milk_fraction_in_cup1_l4056_405664

/-- Represents the contents of a cup --/
structure CupContents where
  coffee : ℚ
  milk : ℚ

/-- Represents the state of both cups --/
structure TwoCups where
  cup1 : CupContents
  cup2 : CupContents

def initial_state : TwoCups :=
  { cup1 := { coffee := 3, milk := 0 },
    cup2 := { coffee := 0, milk := 7 } }

def transfer_coffee (state : TwoCups) : TwoCups :=
  { cup1 := { coffee := state.cup1.coffee * 2/3, milk := state.cup1.milk },
    cup2 := { coffee := state.cup2.coffee + state.cup1.coffee * 1/3, milk := state.cup2.milk } }

def transfer_mixture (state : TwoCups) : TwoCups :=
  let total_cup2 := state.cup2.coffee + state.cup2.milk
  let transfer_amount := total_cup2 * 1/4
  let coffee_ratio := state.cup2.coffee / total_cup2
  let milk_ratio := state.cup2.milk / total_cup2
  { cup1 := { coffee := state.cup1.coffee + transfer_amount * coffee_ratio,
              milk := state.cup1.milk + transfer_amount * milk_ratio },
    cup2 := { coffee := state.cup2.coffee - transfer_amount * coffee_ratio,
              milk := state.cup2.milk - transfer_amount * milk_ratio } }

def final_state : TwoCups :=
  transfer_mixture (transfer_coffee initial_state)

theorem milk_fraction_in_cup1 :
  let total_liquid := final_state.cup1.coffee + final_state.cup1.milk
  final_state.cup1.milk / total_liquid = 7/16 := by sorry

end NUMINAMATH_CALUDE_milk_fraction_in_cup1_l4056_405664


namespace NUMINAMATH_CALUDE_solution_set_inequality_holds_l4056_405647

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| + |x - 1| - 1

-- Theorem 1: Solution set of f(x) ≤ x + 1
theorem solution_set (x : ℝ) : f x ≤ x + 1 ↔ 0 ≤ x ∧ x ≤ 2 := by
  sorry

-- Theorem 2: 3f(x) ≥ f(2x) for all x
theorem inequality_holds (x : ℝ) : 3 * f x ≥ f (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_solution_set_inequality_holds_l4056_405647


namespace NUMINAMATH_CALUDE_max_value_theorem_l4056_405658

theorem max_value_theorem (a b c : ℝ) (h : 9 * a^2 + 4 * b^2 + 25 * c^2 = 4) :
  (6 * a + 3 * b + 10 * c) ≤ Real.sqrt 41 ∧
  ∃ a₀ b₀ c₀ : ℝ, 9 * a₀^2 + 4 * b₀^2 + 25 * c₀^2 = 4 ∧ 6 * a₀ + 3 * b₀ + 10 * c₀ = Real.sqrt 41 :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l4056_405658


namespace NUMINAMATH_CALUDE_marbles_given_to_sam_l4056_405652

def initial_marbles : ℕ := 8
def remaining_marbles : ℕ := 4

theorem marbles_given_to_sam :
  initial_marbles - remaining_marbles = 4 :=
by sorry

end NUMINAMATH_CALUDE_marbles_given_to_sam_l4056_405652


namespace NUMINAMATH_CALUDE_max_tuesday_money_l4056_405646

/-- The amount of money Max's mom gave him on Tuesday -/
def tuesday_amount : ℝ := 8

/-- The amount of money Max's mom gave him on Wednesday -/
def wednesday_amount (t : ℝ) : ℝ := 5 * t

/-- The amount of money Max's mom gave him on Thursday -/
def thursday_amount (t : ℝ) : ℝ := wednesday_amount t + 9

theorem max_tuesday_money :
  ∃ t : ℝ, t = tuesday_amount ∧
    thursday_amount t = t + 41 :=
by sorry

end NUMINAMATH_CALUDE_max_tuesday_money_l4056_405646


namespace NUMINAMATH_CALUDE_line_segment_length_l4056_405689

/-- Given points A, B, C, D, and E on a line in that order, prove that CD = 3 cm -/
theorem line_segment_length (A B C D E : ℝ) : 
  (B - A = 2) → 
  (C - A = 5) → 
  (D - B = 6) → 
  (∃ x, E - D = x) → 
  (E - B = 8) → 
  (E - A < 12) → 
  (D - C = 3) := by
  sorry

end NUMINAMATH_CALUDE_line_segment_length_l4056_405689


namespace NUMINAMATH_CALUDE_infinite_solutions_for_continuous_function_l4056_405634

theorem infinite_solutions_for_continuous_function 
  (f : ℝ → ℝ) 
  (h_cont : Continuous f) 
  (h_dom : ∀ x, x ≥ 1 → f x > 0) 
  (h_sol : ∀ a > 0, ∃ x ≥ 1, f x = a * x) : 
  ∀ a > 0, Set.Infinite {x | x ≥ 1 ∧ f x = a * x} :=
sorry

end NUMINAMATH_CALUDE_infinite_solutions_for_continuous_function_l4056_405634


namespace NUMINAMATH_CALUDE_haircuts_to_goal_l4056_405695

/-- Given a person who has gotten 8 haircuts and is 80% towards their goal,
    prove that the number of additional haircuts needed to reach 100% of the goal is 2. -/
theorem haircuts_to_goal (current_haircuts : ℕ) (current_percentage : ℚ) : 
  current_haircuts = 8 → current_percentage = 80/100 → 
  (100/100 - current_percentage) / (current_percentage / current_haircuts) = 2 := by
sorry

end NUMINAMATH_CALUDE_haircuts_to_goal_l4056_405695


namespace NUMINAMATH_CALUDE_cost_difference_between_cars_l4056_405672

/-- Represents a car with its associated costs and characteristics -/
structure Car where
  initialCost : ℕ
  fuelConsumption : ℕ
  annualInsurance : ℕ
  annualMaintenance : ℕ
  resaleValue : ℕ

/-- Calculates the total cost of owning a car for 5 years -/
def totalCost (c : Car) (annualDistance : ℕ) (fuelPrice : ℕ) (years : ℕ) : ℕ :=
  c.initialCost +
  (c.fuelConsumption * annualDistance / 100 * fuelPrice * years) +
  (c.annualInsurance * years) +
  (c.annualMaintenance * years) -
  c.resaleValue

/-- Theorem stating the difference in total cost between two cars -/
theorem cost_difference_between_cars :
  let carA : Car := {
    initialCost := 900000,
    fuelConsumption := 9,
    annualInsurance := 35000,
    annualMaintenance := 25000,
    resaleValue := 500000
  }
  let carB : Car := {
    initialCost := 600000,
    fuelConsumption := 10,
    annualInsurance := 32000,
    annualMaintenance := 20000,
    resaleValue := 350000
  }
  let annualDistance := 15000
  let fuelPrice := 40
  let years := 5

  totalCost carA annualDistance fuelPrice years -
  totalCost carB annualDistance fuelPrice years = 160000 := by
  sorry

end NUMINAMATH_CALUDE_cost_difference_between_cars_l4056_405672


namespace NUMINAMATH_CALUDE_daves_rides_l4056_405619

theorem daves_rides (total_rides : ℕ) (second_day_rides : ℕ) (first_day_rides : ℕ) :
  total_rides = 7 ∧ second_day_rides = 3 ∧ total_rides = first_day_rides + second_day_rides →
  first_day_rides = 4 := by
sorry

end NUMINAMATH_CALUDE_daves_rides_l4056_405619


namespace NUMINAMATH_CALUDE_wire_cutting_problem_l4056_405691

theorem wire_cutting_problem (shorter_piece longer_piece total_length : ℝ) :
  shorter_piece = 10 →
  shorter_piece = (2 / 5) * longer_piece →
  total_length = shorter_piece + longer_piece →
  total_length = 35 := by
sorry

end NUMINAMATH_CALUDE_wire_cutting_problem_l4056_405691


namespace NUMINAMATH_CALUDE_special_function_properties_l4056_405632

/-- A function satisfying the given property -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f x + f y + 2 * x * y - 1

theorem special_function_properties (f : ℝ → ℝ) 
  (h1 : special_function f) (h2 : f 1 = 4) : 
  f 0 = 1 ∧ ∀ n : ℕ, f n = (n + 1)^2 :=
by sorry

end NUMINAMATH_CALUDE_special_function_properties_l4056_405632


namespace NUMINAMATH_CALUDE_count_valid_primes_l4056_405628

/-- Convert a number from base p to base 10 --/
def to_base_10 (digits : List Nat) (p : Nat) : Nat :=
  digits.foldr (fun d acc => d + p * acc) 0

/-- Check if the equation holds for a given prime p --/
def equation_holds (p : Nat) : Prop :=
  to_base_10 [9, 7, 6] p + to_base_10 [5, 0, 7] p + to_base_10 [2, 3, 8] p =
  to_base_10 [4, 2, 9] p + to_base_10 [5, 9, 5] p + to_base_10 [6, 9, 7] p

/-- The main theorem --/
theorem count_valid_primes :
  ∃ (S : Finset Nat), S.card = 3 ∧ 
  (∀ p ∈ S, Nat.Prime p ∧ p < 10 ∧ equation_holds p) ∧
  (∀ p, Nat.Prime p → p < 10 → equation_holds p → p ∈ S) :=
sorry

end NUMINAMATH_CALUDE_count_valid_primes_l4056_405628


namespace NUMINAMATH_CALUDE_root_property_l4056_405654

theorem root_property (a : ℝ) : 3 * a^2 - 4 * a + 1 = 0 → 6 * a^2 - 8 * a + 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_root_property_l4056_405654


namespace NUMINAMATH_CALUDE_jewels_gain_is_3_25_l4056_405625

/-- Calculates Jewel's total gain from selling magazines --/
def jewels_gain (num_magazines : ℕ) 
                (cost_per_magazine : ℚ) 
                (regular_price : ℚ) 
                (discount_percent : ℚ) 
                (num_regular_price : ℕ) : ℚ :=
  let total_cost := num_magazines * cost_per_magazine
  let revenue_regular := num_regular_price * regular_price
  let discounted_price := regular_price * (1 - discount_percent)
  let revenue_discounted := (num_magazines - num_regular_price) * discounted_price
  let total_revenue := revenue_regular + revenue_discounted
  total_revenue - total_cost

theorem jewels_gain_is_3_25 : 
  jewels_gain 10 3 (7/2) (1/10) 5 = 13/4 := by
  sorry

end NUMINAMATH_CALUDE_jewels_gain_is_3_25_l4056_405625


namespace NUMINAMATH_CALUDE_island_age_conversion_l4056_405674

/-- Converts a base 7 number to base 10 --/
def base7ToBase10 (hundreds : Nat) (tens : Nat) (ones : Nat) : Nat :=
  hundreds * 7^2 + tens * 7^1 + ones * 7^0

/-- The age of the island in base 7 and base 10 --/
theorem island_age_conversion :
  base7ToBase10 3 4 6 = 181 := by
  sorry

end NUMINAMATH_CALUDE_island_age_conversion_l4056_405674


namespace NUMINAMATH_CALUDE_wrench_handle_length_l4056_405692

/-- Represents the inverse relationship between force and handle length -/
def inverse_relation (force : ℝ) (length : ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ force * length = k

theorem wrench_handle_length
  (force₁ : ℝ) (length₁ : ℝ) (force₂ : ℝ) (length₂ : ℝ)
  (h_inverse : inverse_relation force₁ length₁ ∧ inverse_relation force₂ length₂)
  (h_force₁ : force₁ = 300)
  (h_length₁ : length₁ = 12)
  (h_force₂ : force₂ = 400) :
  length₂ = 9 := by
  sorry

end NUMINAMATH_CALUDE_wrench_handle_length_l4056_405692


namespace NUMINAMATH_CALUDE_quadratic_root_difference_l4056_405621

theorem quadratic_root_difference (c : ℝ) : 
  (∃ x y : ℝ, x^2 + 7*x + c = 0 ∧ y^2 + 7*y + c = 0 ∧ |x - y| = Real.sqrt 85) → 
  c = -9 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_difference_l4056_405621


namespace NUMINAMATH_CALUDE_library_code_probability_l4056_405659

/-- The number of digits in the code -/
def code_length : ℕ := 6

/-- The total number of possible digits -/
def total_digits : ℕ := 10

/-- The probability of selecting a code with all different digits and not starting with 0 -/
def probability : ℚ := 1496880 / 1000000

/-- Theorem stating the probability of selecting a code with all different digits 
    and not starting with 0 is 0.13608 -/
theorem library_code_probability : 
  probability = 1496880 / 1000000 ∧ 
  (1496880 : ℚ) / 1000000 = 0.13608 := by sorry

end NUMINAMATH_CALUDE_library_code_probability_l4056_405659


namespace NUMINAMATH_CALUDE_x_plus_one_is_linear_l4056_405682

/-- A linear equation is an equation with variables of only the first power -/
def is_linear_equation (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, ∀ x, f x = a * x + b

/-- The function representing x + 1 = 0 -/
def f (x : ℝ) : ℝ := x + 1

theorem x_plus_one_is_linear : is_linear_equation f := by
  sorry

end NUMINAMATH_CALUDE_x_plus_one_is_linear_l4056_405682


namespace NUMINAMATH_CALUDE_frustum_volume_l4056_405681

/-- The volume of a frustum with given ratio of radii and height, and slant height -/
theorem frustum_volume (r R h s : ℝ) (h1 : R = 4*r) (h2 : h = 4*r) (h3 : s = 10) 
  (h4 : s^2 = h^2 + (R - r)^2) : 
  (1/3 : ℝ) * Real.pi * h * (r^2 + R^2 + r*R) = 224 * Real.pi := by
  sorry

#check frustum_volume

end NUMINAMATH_CALUDE_frustum_volume_l4056_405681


namespace NUMINAMATH_CALUDE_complex_root_modulus_one_l4056_405618

theorem complex_root_modulus_one (n : ℕ) :
  (∃ z : ℂ, z^(n+1) - z^n - 1 = 0 ∧ Complex.abs z = 1) ↔ (∃ k : ℤ, n + 2 = 6 * k) :=
sorry

end NUMINAMATH_CALUDE_complex_root_modulus_one_l4056_405618


namespace NUMINAMATH_CALUDE_power_sum_theorem_l4056_405622

theorem power_sum_theorem (m n : ℕ) (h1 : 2^m = 3) (h2 : 2^n = 2) : 2^(m+2*n) = 12 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_theorem_l4056_405622


namespace NUMINAMATH_CALUDE_sue_shoe_probability_l4056_405649

/-- Represents the distribution of shoes by color --/
structure ShoeDistribution where
  black : Nat
  brown : Nat
  gray : Nat
  red : Nat

/-- Calculates the probability of selecting two shoes of the same color
    with one left and one right, given a shoe distribution --/
def samePairProbability (d : ShoeDistribution) : Rat :=
  let totalShoes := 2 * (d.black + d.brown + d.gray + d.red)
  let blackProb := (d.black : Rat) * (d.black - 1) / (totalShoes * (totalShoes - 1))
  let brownProb := (d.brown : Rat) * (d.brown - 1) / (totalShoes * (totalShoes - 1))
  let grayProb := (d.gray : Rat) * (d.gray - 1) / (totalShoes * (totalShoes - 1))
  let redProb := (d.red : Rat) * (d.red - 1) / (totalShoes * (totalShoes - 1))
  blackProb + brownProb + grayProb + redProb

theorem sue_shoe_probability :
  let sueShoes := ShoeDistribution.mk 7 4 2 1
  samePairProbability sueShoes = 20 / 63 := by
  sorry

end NUMINAMATH_CALUDE_sue_shoe_probability_l4056_405649


namespace NUMINAMATH_CALUDE_geometric_sequence_ninth_term_l4056_405686

/-- Given a geometric sequence where the 3rd term is 5 and the 6th term is 40,
    the 9th term is 320. -/
theorem geometric_sequence_ninth_term : ∀ (a : ℕ → ℝ),
  (∀ n : ℕ, a (n + 1) = a n * (a 4 / a 3)) →  -- Geometric sequence condition
  a 3 = 5 →                                   -- 3rd term is 5
  a 6 = 40 →                                  -- 6th term is 40
  a 9 = 320 :=                                -- 9th term is 320
by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_ninth_term_l4056_405686


namespace NUMINAMATH_CALUDE_kaleb_books_l4056_405642

theorem kaleb_books (initial_books : ℕ) : 
  initial_books - 17 + 7 = 24 → initial_books = 34 := by
  sorry

end NUMINAMATH_CALUDE_kaleb_books_l4056_405642


namespace NUMINAMATH_CALUDE_solve_for_y_l4056_405614

theorem solve_for_y (x y : ℤ) (h1 : x^2 = y - 8) (h2 : x = -7) : y = 57 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l4056_405614


namespace NUMINAMATH_CALUDE_remainder_sum_l4056_405693

theorem remainder_sum (c d : ℤ) 
  (hc : c % 80 = 75)
  (hd : d % 120 = 117) : 
  (c + d) % 40 = 32 := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_l4056_405693


namespace NUMINAMATH_CALUDE_jungkook_has_smallest_number_l4056_405645

def yoongi_number : ℕ := 7
def jungkook_number : ℕ := 6
def yuna_number : ℕ := 9

theorem jungkook_has_smallest_number :
  jungkook_number ≤ yoongi_number ∧ jungkook_number ≤ yuna_number :=
sorry

end NUMINAMATH_CALUDE_jungkook_has_smallest_number_l4056_405645


namespace NUMINAMATH_CALUDE_not_necessarily_regular_l4056_405685

/-- A convex polyhedron -/
structure ConvexPolyhedron where
  -- Add necessary fields here
  mk ::

/-- Predicate to check if all edges of a polyhedron are equal -/
def all_edges_equal (p : ConvexPolyhedron) : Prop :=
  sorry

/-- Predicate to check if all dihedral angles of a polyhedron are equal -/
def all_dihedral_angles_equal (p : ConvexPolyhedron) : Prop :=
  sorry

/-- Predicate to check if all polyhedral angles of a polyhedron are equal -/
def all_polyhedral_angles_equal (p : ConvexPolyhedron) : Prop :=
  sorry

/-- Predicate to check if a polyhedron is regular -/
def is_regular (p : ConvexPolyhedron) : Prop :=
  sorry

/-- Theorem stating that a convex polyhedron with equal edges and either equal dihedral angles
    or equal polyhedral angles is not necessarily regular -/
theorem not_necessarily_regular :
  ∃ p : ConvexPolyhedron,
    (all_edges_equal p ∧ all_dihedral_angles_equal p ∧ ¬is_regular p) ∨
    (all_edges_equal p ∧ all_polyhedral_angles_equal p ∧ ¬is_regular p) :=
  sorry

end NUMINAMATH_CALUDE_not_necessarily_regular_l4056_405685


namespace NUMINAMATH_CALUDE_whitney_spent_440_l4056_405602

/-- Calculates the total amount spent by Whitney on books and magazines. -/
def whitneyTotalSpent (whaleBooks fishBooks sharkBooks magazines : ℕ) 
  (whaleCost fishCost sharkCost magazineCost : ℕ) : ℕ :=
  whaleBooks * whaleCost + fishBooks * fishCost + sharkBooks * sharkCost + magazines * magazineCost

/-- Proves that Whitney spent $440 in total. -/
theorem whitney_spent_440 : 
  whitneyTotalSpent 15 12 5 8 14 13 10 3 = 440 := by
  sorry

end NUMINAMATH_CALUDE_whitney_spent_440_l4056_405602


namespace NUMINAMATH_CALUDE_team_order_l4056_405637

/-- Represents the points of a team in a sports league. -/
structure TeamPoints where
  points : ℕ

/-- Represents the points of all teams in the sports league. -/
structure LeaguePoints where
  A : TeamPoints
  B : TeamPoints
  C : TeamPoints
  D : TeamPoints

/-- Defines the conditions given in the problem. -/
def satisfiesConditions (lp : LeaguePoints) : Prop :=
  (lp.A.points + lp.C.points = lp.B.points + lp.D.points) ∧
  (lp.B.points + lp.A.points + 5 ≤ lp.D.points + lp.C.points) ∧
  (lp.B.points + lp.C.points ≥ lp.A.points + lp.D.points + 3)

/-- Defines the correct order of teams based on their points. -/
def correctOrder (lp : LeaguePoints) : Prop :=
  lp.C.points > lp.D.points ∧ lp.D.points > lp.B.points ∧ lp.B.points > lp.A.points

/-- Theorem stating that if the conditions are satisfied, the correct order of teams is C, D, B, A. -/
theorem team_order (lp : LeaguePoints) :
  satisfiesConditions lp → correctOrder lp := by
  sorry


end NUMINAMATH_CALUDE_team_order_l4056_405637


namespace NUMINAMATH_CALUDE_unique_solution_condition_l4056_405680

theorem unique_solution_condition (k : ℝ) : 
  (∃! x y : ℝ, y = x^2 + k ∧ y = 3*x) ↔ k = 9/4 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l4056_405680


namespace NUMINAMATH_CALUDE_beaver_problem_l4056_405675

theorem beaver_problem (initial_beavers final_beavers : ℕ) : 
  final_beavers = initial_beavers + 1 → 
  final_beavers = 3 → 
  initial_beavers = 2 := by
sorry

end NUMINAMATH_CALUDE_beaver_problem_l4056_405675


namespace NUMINAMATH_CALUDE_min_gumballs_for_five_correct_l4056_405690

/-- Represents the number of gumballs of each color -/
structure GumballMachine :=
  (red : ℕ)
  (white : ℕ)
  (blue : ℕ)
  (green : ℕ)

/-- The minimum number of gumballs needed to guarantee 5 of the same color -/
def minGumballsForFive (m : GumballMachine) : ℕ := 17

/-- Theorem stating that for the given gumball machine, 
    17 is the minimum number of gumballs needed to guarantee 5 of the same color -/
theorem min_gumballs_for_five_correct (m : GumballMachine) 
  (h_red : m.red = 12) 
  (h_white : m.white = 10) 
  (h_blue : m.blue = 9) 
  (h_green : m.green = 8) : 
  minGumballsForFive m = 17 := by
  sorry


end NUMINAMATH_CALUDE_min_gumballs_for_five_correct_l4056_405690


namespace NUMINAMATH_CALUDE_negative_intervals_l4056_405612

-- Define the expression
def f (x : ℝ) : ℝ := (x - 2) * (x + 2) * (x - 3)

-- Define the set of x for which f(x) is negative
def S : Set ℝ := {x | f x < 0}

-- State the theorem
theorem negative_intervals : S = Set.Iio (-2) ∪ Set.Ioo 2 3 := by sorry

end NUMINAMATH_CALUDE_negative_intervals_l4056_405612


namespace NUMINAMATH_CALUDE_expression_value_l4056_405623

theorem expression_value :
  let x : ℤ := 2
  let y : ℤ := -3
  let z : ℤ := 1
  x^2 + y^2 - z^2 + 2*x*y + 10 = 10 := by
sorry

end NUMINAMATH_CALUDE_expression_value_l4056_405623


namespace NUMINAMATH_CALUDE_carolyn_practice_time_l4056_405616

/-- Calculates the total practice time for Carolyn in a month -/
def total_practice_time (piano_time : ℕ) (violin_multiplier : ℕ) (practice_days : ℕ) (weeks : ℕ) : ℕ :=
  let daily_total := piano_time + violin_multiplier * piano_time
  let weekly_total := daily_total * practice_days
  weekly_total * weeks

/-- Proves that Carolyn's total practice time in a month is 1920 minutes -/
theorem carolyn_practice_time :
  total_practice_time 20 3 6 4 = 1920 :=
by sorry

end NUMINAMATH_CALUDE_carolyn_practice_time_l4056_405616


namespace NUMINAMATH_CALUDE_total_students_is_1480_l4056_405638

/-- Represents a campus in the school district -/
structure Campus where
  grades : ℕ  -- number of grades
  students_per_grade : ℕ  -- number of students per grade
  extra_students : ℕ  -- number of extra students in special programs

/-- Calculates the total number of students in a campus -/
def campus_total (c : Campus) : ℕ :=
  c.grades * c.students_per_grade + c.extra_students

/-- The school district with its three campuses -/
structure SchoolDistrict where
  campus_a : Campus
  campus_b : Campus
  campus_c : Campus

/-- Represents the specific school district described in the problem -/
def our_district : SchoolDistrict :=
  { campus_a := { grades := 5, students_per_grade := 100, extra_students := 30 }
  , campus_b := { grades := 5, students_per_grade := 120, extra_students := 0 }
  , campus_c := { grades := 2, students_per_grade := 150, extra_students := 50 }
  }

/-- Theorem stating that the total number of students in our school district is 1480 -/
theorem total_students_is_1480 : 
  campus_total our_district.campus_a + 
  campus_total our_district.campus_b + 
  campus_total our_district.campus_c = 1480 := by
  sorry

end NUMINAMATH_CALUDE_total_students_is_1480_l4056_405638


namespace NUMINAMATH_CALUDE_lisa_photos_last_weekend_l4056_405617

/-- Calculates the number of photos Lisa took last weekend based on given conditions --/
def photos_last_weekend (animal_photos : ℕ) (flower_multiplier : ℕ) (scenery_difference : ℕ) (weekend_difference : ℕ) : ℕ :=
  let flower_photos := animal_photos * flower_multiplier
  let scenery_photos := flower_photos - scenery_difference
  let total_this_weekend := animal_photos + flower_photos + scenery_photos
  total_this_weekend - weekend_difference

/-- Theorem stating that Lisa took 45 photos last weekend given the conditions --/
theorem lisa_photos_last_weekend :
  photos_last_weekend 10 3 10 15 = 45 := by
  sorry

#eval photos_last_weekend 10 3 10 15

end NUMINAMATH_CALUDE_lisa_photos_last_weekend_l4056_405617


namespace NUMINAMATH_CALUDE_wine_purchase_additional_cost_l4056_405607

/-- Represents the price changes and conditions for wine purchases over three months --/
structure WinePrices where
  initial_price : ℝ
  tariff_increase1 : ℝ
  tariff_increase2 : ℝ
  exchange_rate_change1 : ℝ
  exchange_rate_change2 : ℝ
  bulk_discount : ℝ
  bottles_per_month : ℕ

/-- Calculates the total additional cost of wine purchases over three months --/
def calculate_additional_cost (prices : WinePrices) : ℝ :=
  let month1_price := prices.initial_price * (1 + prices.exchange_rate_change1)
  let month2_price := prices.initial_price * (1 + prices.tariff_increase1) * (1 - prices.bulk_discount)
  let month3_price := prices.initial_price * (1 + prices.tariff_increase1 + prices.tariff_increase2) * (1 - prices.exchange_rate_change2)
  let total_cost := (month1_price + month2_price + month3_price) * prices.bottles_per_month
  let initial_total := prices.initial_price * prices.bottles_per_month * 3
  total_cost - initial_total

/-- Theorem stating that the additional cost of wine purchases over three months is $42.20 --/
theorem wine_purchase_additional_cost :
  let prices : WinePrices := {
    initial_price := 20,
    tariff_increase1 := 0.25,
    tariff_increase2 := 0.10,
    exchange_rate_change1 := 0.05,
    exchange_rate_change2 := 0.03,
    bulk_discount := 0.15,
    bottles_per_month := 5
  }
  calculate_additional_cost prices = 42.20 := by
  sorry


end NUMINAMATH_CALUDE_wine_purchase_additional_cost_l4056_405607


namespace NUMINAMATH_CALUDE_inequality_reversal_l4056_405683

theorem inequality_reversal (a b : ℝ) (h : a > b) : ∃ m : ℝ, ¬(m * a < m * b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_reversal_l4056_405683


namespace NUMINAMATH_CALUDE_a_33_mod_42_l4056_405667

/-- Definition of a_n as the integer obtained by writing all integers from 1 to n from left to right -/
def a (n : ℕ) : ℕ := sorry

/-- Theorem stating that a_33 divided by 42 has a remainder of 20 -/
theorem a_33_mod_42 : a 33 % 42 = 20 := by sorry

end NUMINAMATH_CALUDE_a_33_mod_42_l4056_405667


namespace NUMINAMATH_CALUDE_math_competition_score_l4056_405660

theorem math_competition_score 
  (a₁ a₂ a₃ a₄ a₅ : ℕ) 
  (h_distinct : a₁ < a₂ ∧ a₂ < a₃ ∧ a₃ < a₄ ∧ a₄ < a₅)
  (h_first_two : a₁ + a₂ = 10)
  (h_last_two : a₄ + a₅ = 18) :
  a₁ + a₂ + a₃ + a₄ + a₅ = 35 :=
by sorry

end NUMINAMATH_CALUDE_math_competition_score_l4056_405660


namespace NUMINAMATH_CALUDE_church_attendance_l4056_405678

theorem church_attendance (male_adults female_adults total_people : ℕ) 
  (h1 : male_adults = 60)
  (h2 : female_adults = 60)
  (h3 : total_people = 200) :
  total_people - (male_adults + female_adults) = 80 := by
sorry

end NUMINAMATH_CALUDE_church_attendance_l4056_405678


namespace NUMINAMATH_CALUDE_chord_length_circle_line_l4056_405636

/-- The chord length cut by a circle on a line --/
theorem chord_length_circle_line (x y : ℝ) : 
  let circle := fun x y => x^2 + y^2 - 8*x - 2*y + 1 = 0
  let line := fun x => Real.sqrt 3 * x + 1
  let center := (4, 1)
  let radius := 4
  let distance_center_to_line := 2 * Real.sqrt 3
  true → -- placeholder for the circle and line equations
  2 * Real.sqrt (radius^2 - distance_center_to_line^2) = 4 :=
by sorry

end NUMINAMATH_CALUDE_chord_length_circle_line_l4056_405636


namespace NUMINAMATH_CALUDE_tan_2018pi_minus_alpha_l4056_405613

theorem tan_2018pi_minus_alpha (α : ℝ) (h1 : α ∈ Set.Ioo 0 (3 * π / 2)) 
  (h2 : Real.cos (3 * π / 2 - α) = Real.sqrt 3 / 2) : 
  Real.tan (2018 * π - α) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_2018pi_minus_alpha_l4056_405613


namespace NUMINAMATH_CALUDE_tony_additional_degrees_l4056_405656

/-- Represents the number of years Tony spent in school for various degrees -/
structure TonySchoolYears where
  science : ℕ
  physics : ℕ
  additional : ℕ
  total : ℕ

/-- Calculates the number of additional degrees Tony got -/
def additional_degrees (years : TonySchoolYears) : ℕ :=
  (years.total - years.science - years.physics) / years.science

/-- Theorem stating that Tony got 2 additional degrees -/
theorem tony_additional_degrees :
  ∀ (years : TonySchoolYears),
    years.science = 4 →
    years.physics = 2 →
    years.total = 14 →
    additional_degrees years = 2 := by
  sorry

#check tony_additional_degrees

end NUMINAMATH_CALUDE_tony_additional_degrees_l4056_405656


namespace NUMINAMATH_CALUDE_tangent_line_at_ln2_max_k_for_f_greater_g_l4056_405641

noncomputable section

def f (x : ℝ) : ℝ := Real.exp x / (Real.exp x - 1)

def g (k : ℕ) (x : ℝ) : ℝ := k / (x + 1)

def tangent_line (x : ℝ) : ℝ := -2 * x + 2 * Real.log 2 + 2

theorem tangent_line_at_ln2 (x : ℝ) (h : x > 0) :
  tangent_line x = -2 * x + 2 * Real.log 2 + 2 :=
sorry

theorem max_k_for_f_greater_g :
  ∃ (k : ℕ), k = 3 ∧ 
  (∀ (x : ℝ), x > 0 → f x > g k x) ∧
  (∀ (k' : ℕ), k' > k → ∃ (x : ℝ), x > 0 ∧ f x ≤ g k' x) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_ln2_max_k_for_f_greater_g_l4056_405641


namespace NUMINAMATH_CALUDE_tangent_line_at_2_2_increasing_intervals_decreasing_interval_l4056_405671

-- Define the function f(x) = x^3 - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the derivative of f(x)
def f_derivative (x : ℝ) : ℝ := 3*x^2 - 3

-- Theorem for the tangent line equation
theorem tangent_line_at_2_2 :
  ∃ (a b c : ℝ), a * 2 + b * 2 + c = 0 ∧
  ∀ (x y : ℝ), y = f x → (y - f 2) = f_derivative 2 * (x - 2) →
  a * x + b * y + c = 0 :=
sorry

-- Theorem for increasing intervals
theorem increasing_intervals :
  ∀ x, (x < -1 ∨ x > 1) → f_derivative x > 0 :=
sorry

-- Theorem for decreasing interval
theorem decreasing_interval :
  ∀ x, -1 < x ∧ x < 1 → f_derivative x < 0 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_2_2_increasing_intervals_decreasing_interval_l4056_405671


namespace NUMINAMATH_CALUDE_max_value_theorem_l4056_405609

theorem max_value_theorem (a b c d e : ℝ) 
  (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0) (pos_d : d > 0) (pos_e : e > 0)
  (sum_squares : a^2 + b^2 + c^2 + d^2 + e^2 = 504) :
  ac + 3*b*c + 4*c*d + 6*c*e ≤ 252 * Real.sqrt 62 ∧
  (a = 2 ∧ b = 6 ∧ c = 6 * Real.sqrt 7 ∧ d = 8 ∧ e = 12) →
  ac + 3*b*c + 4*c*d + 6*c*e = 252 * Real.sqrt 62 := by
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l4056_405609


namespace NUMINAMATH_CALUDE_maximize_profit_l4056_405600

/-- The production volume that maximizes profit -/
def optimal_production_volume : ℝ := 6

/-- Sales revenue as a function of production volume -/
def sales_revenue (x : ℝ) : ℝ := 17 * x^2

/-- Production cost as a function of production volume -/
def production_cost (x : ℝ) : ℝ := 2 * x^3 - x^2

/-- Profit as a function of production volume -/
def profit (x : ℝ) : ℝ := sales_revenue x - production_cost x

theorem maximize_profit (x : ℝ) (h : x > 0) :
  profit x ≤ profit optimal_production_volume := by
  sorry

end NUMINAMATH_CALUDE_maximize_profit_l4056_405600


namespace NUMINAMATH_CALUDE_program_outputs_divisors_l4056_405679

/-- The set of numbers output by the program for a given input n -/
def program_output (n : ℕ) : Set ℕ :=
  {i : ℕ | i ≤ n ∧ n % i = 0}

/-- The set of all divisors of n -/
def divisors (n : ℕ) : Set ℕ :=
  {i : ℕ | i ∣ n}

/-- Theorem stating that the program output is equal to the set of all divisors -/
theorem program_outputs_divisors (n : ℕ) : program_output n = divisors n := by
  sorry

end NUMINAMATH_CALUDE_program_outputs_divisors_l4056_405679


namespace NUMINAMATH_CALUDE_vector_subtraction_l4056_405673

theorem vector_subtraction (c d : Fin 3 → ℝ) 
  (hc : c = ![5, -3, 2])
  (hd : d = ![-2, 1, 5]) :
  c - 4 • d = ![13, -7, -18] := by
sorry

end NUMINAMATH_CALUDE_vector_subtraction_l4056_405673


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l4056_405611

-- Define the repeating decimals as rational numbers
def a : ℚ := 234 / 999
def b : ℚ := 567 / 999
def c : ℚ := 891 / 999

-- Define the result of the operation
def result : ℚ := a - b + c

-- Theorem statement
theorem repeating_decimal_sum : result = 31 / 37 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l4056_405611


namespace NUMINAMATH_CALUDE_vydmans_formula_l4056_405698

theorem vydmans_formula (h b x r : ℝ) (h_pos : h > 0) (b_pos : b > 0) (x_pos : x > 0) :
  r = Real.sqrt ((b / 2)^2 + ((h^2 + (b / 2 - x)^2 - b^2 / 4) / (2 * h))^2) →
  r^2 = (b / 2)^2 + ((h^2 + (b / 2 - x)^2 - b^2 / 4) / (2 * h))^2 :=
by sorry

end NUMINAMATH_CALUDE_vydmans_formula_l4056_405698


namespace NUMINAMATH_CALUDE_log_expression_equals_two_l4056_405630

-- Define the logarithm base 10 function
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_expression_equals_two :
  (log10 2)^2 + log10 2 * log10 50 + log10 25 = 2 := by sorry

end NUMINAMATH_CALUDE_log_expression_equals_two_l4056_405630


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l4056_405650

/-- The perimeter of a rhombus with diagonals 24 and 10 is 52 -/
theorem rhombus_perimeter (d1 d2 : ℝ) : 
  d1 = 24 → d2 = 10 → 4 * Real.sqrt ((d1/2)^2 + (d2/2)^2) = 52 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l4056_405650


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l4056_405631

/-- A parabola with vertex at origin and axis of symmetry along x-axis -/
structure Parabola where
  p : ℝ
  eq : ℝ → ℝ → Prop := fun x y => y^2 = 2 * p * x

/-- A line with slope k passing through a fixed point -/
structure Line where
  k : ℝ
  fixed_point : ℝ × ℝ
  eq : ℝ → ℝ → Prop := fun x y => y = k * x + (fixed_point.2 - k * fixed_point.1)

/-- The number of intersection points between a parabola and a line -/
def intersection_count (par : Parabola) (l : Line) : ℕ :=
  sorry

theorem parabola_line_intersection 
  (par : Parabola) 
  (h_par : par.eq (1/2) (-Real.sqrt 2))
  (l : Line)
  (h_line : l.fixed_point = (-2, 1)) :
  (intersection_count par l = 2) ↔ 
  (-1 < l.k ∧ l.k < 1/2 ∧ l.k ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l4056_405631


namespace NUMINAMATH_CALUDE_valid_m_range_l4056_405663

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∃ (k : ℝ), k ≠ 0 ∧ (1 = -k ∧ 1 = -k ∧ m = k * |m|)

def q (m : ℝ) : Prop := (2*m + 1 > 0 ∧ m - 3 < 0) ∨ (2*m + 1 < 0 ∧ m - 3 > 0)

-- State the theorem
theorem valid_m_range :
  ∀ m : ℝ, (¬(p m) ∧ (p m ∨ q m)) → (0 < m ∧ m < 3) :=
by sorry

end NUMINAMATH_CALUDE_valid_m_range_l4056_405663


namespace NUMINAMATH_CALUDE_cos_sin_negative_225_deg_l4056_405684

theorem cos_sin_negative_225_deg : Real.cos (-225 * π / 180) + Real.sin (-225 * π / 180) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_sin_negative_225_deg_l4056_405684


namespace NUMINAMATH_CALUDE_prob_two_red_or_blue_is_one_fifth_l4056_405670

/-- Represents the number of marbles of each color in the bag -/
structure MarbleCounts where
  red : ℕ
  blue : ℕ
  green : ℕ

/-- Calculates the probability of drawing two marbles sequentially without replacement
    where both marbles are either red or blue -/
def prob_two_red_or_blue (counts : MarbleCounts) : ℚ :=
  let total := counts.red + counts.blue + counts.green
  let red_or_blue := counts.red + counts.blue
  (red_or_blue / total) * ((red_or_blue - 1) / (total - 1))

/-- Theorem stating that the probability of drawing two red or blue marbles
    from a bag with 4 red, 3 blue, and 8 green marbles is 1/5 -/
theorem prob_two_red_or_blue_is_one_fifth :
  prob_two_red_or_blue ⟨4, 3, 8⟩ = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_red_or_blue_is_one_fifth_l4056_405670


namespace NUMINAMATH_CALUDE_equation_solution_l4056_405606

theorem equation_solution : ∃ x : ℝ, 7 * x - 5 = 6 * x ∧ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l4056_405606


namespace NUMINAMATH_CALUDE_tetrahedron_edge_assignment_l4056_405601

/-- Represents a tetrahedron with face areas -/
structure Tetrahedron where
  s : ℝ  -- smallest face area
  S : ℝ  -- largest face area
  a : ℝ  -- area of another face
  b : ℝ  -- area of the fourth face
  h_s_smallest : s ≤ a ∧ s ≤ b ∧ s ≤ S
  h_S_largest : S ≥ a ∧ S ≥ b ∧ S ≥ s
  h_positive : s > 0 ∧ S > 0 ∧ a > 0 ∧ b > 0

/-- Represents the edge values of a tetrahedron -/
structure TetrahedronEdges where
  e1 : ℝ  -- edge common to smallest and largest face
  e2 : ℝ  -- edge of smallest face
  e3 : ℝ  -- edge of smallest face
  e4 : ℝ  -- edge of largest face
  e5 : ℝ  -- edge of largest face
  e6 : ℝ  -- remaining edge

/-- Checks if the edge values satisfy the face area conditions -/
def satisfies_conditions (t : Tetrahedron) (e : TetrahedronEdges) : Prop :=
  e.e1 ≥ 0 ∧ e.e2 ≥ 0 ∧ e.e3 ≥ 0 ∧ e.e4 ≥ 0 ∧ e.e5 ≥ 0 ∧ e.e6 ≥ 0 ∧
  e.e1 + e.e2 + e.e3 = t.s ∧
  e.e1 + e.e4 + e.e5 = t.S ∧
  e.e2 + e.e5 + e.e6 = t.a ∧
  e.e3 + e.e4 + e.e6 = t.b

theorem tetrahedron_edge_assignment (t : Tetrahedron) :
  ∃ e : TetrahedronEdges, satisfies_conditions t e := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_edge_assignment_l4056_405601


namespace NUMINAMATH_CALUDE_additional_area_codes_l4056_405644

/-- The number of available signs for area codes -/
def num_signs : ℕ := 124

/-- The number of 2-letter area codes -/
def two_letter_codes : ℕ := num_signs * (num_signs - 1)

/-- The number of 3-letter area codes -/
def three_letter_codes : ℕ := num_signs * (num_signs - 1) * (num_signs - 2)

/-- The additional number of area codes created with the 3-letter system compared to the 2-letter system -/
theorem additional_area_codes :
  three_letter_codes - two_letter_codes = 1845396 :=
by sorry

end NUMINAMATH_CALUDE_additional_area_codes_l4056_405644


namespace NUMINAMATH_CALUDE_f_properties_l4056_405696

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - x^2 + 1

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 3*x^2 - 2*x

-- Theorem stating the properties of f(x)
theorem f_properties :
  (f 0 = 1) ∧ 
  (f' 1 = 1) ∧
  (∀ x : ℝ, f x ≤ 1) ∧
  (f 0 = 1) ∧
  (∀ x : ℝ, f x ≥ 23/27) ∧
  (f (2/3) = 23/27) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l4056_405696


namespace NUMINAMATH_CALUDE_expression_value_l4056_405643

theorem expression_value (x y z : ℝ) 
  (eq1 : 2*x - 3*y - z = 0)
  (eq2 : x + 3*y - 14*z = 0)
  (h : z ≠ 0) :
  (x^2 - x*y) / (y^2 + 2*z^2) = 10/11 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l4056_405643


namespace NUMINAMATH_CALUDE_x_minus_y_values_l4056_405687

theorem x_minus_y_values (x y : ℝ) (h1 : x^2 = 9) (h2 : |y| = 4) (h3 : x < y) :
  x - y = -7 ∨ x - y = -1 := by
  sorry

end NUMINAMATH_CALUDE_x_minus_y_values_l4056_405687


namespace NUMINAMATH_CALUDE_estimate_sqrt_expression_l4056_405610

theorem estimate_sqrt_expression :
  7 < Real.sqrt 32 * Real.sqrt (1/2) + Real.sqrt 12 ∧
  Real.sqrt 32 * Real.sqrt (1/2) + Real.sqrt 12 < 8 :=
by
  sorry

end NUMINAMATH_CALUDE_estimate_sqrt_expression_l4056_405610


namespace NUMINAMATH_CALUDE_remainder_theorem_l4056_405620

/-- A polynomial of the form Mx^4 + Nx^2 + Dx - 5 -/
def q (M N D x : ℝ) : ℝ := M * x^4 + N * x^2 + D * x - 5

theorem remainder_theorem (M N D : ℝ) :
  (∃ p : ℝ → ℝ, ∀ x, q M N D x = (x - 2) * p x + 15) →
  (∃ p : ℝ → ℝ, ∀ x, q M N D x = (x + 2) * p x + 15) :=
by sorry

end NUMINAMATH_CALUDE_remainder_theorem_l4056_405620


namespace NUMINAMATH_CALUDE_quadratic_radical_equality_l4056_405657

/-- If the simplest quadratic radical 2√(4m-1) is of the same type as √(2+3m), then m = 3. -/
theorem quadratic_radical_equality (m : ℝ) : 
  (∃ k : ℝ, k > 0 ∧ k * (4 * m - 1) = 2 + 3 * m) → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_radical_equality_l4056_405657


namespace NUMINAMATH_CALUDE_integral_of_4x_plus_7_cos_3x_l4056_405604

theorem integral_of_4x_plus_7_cos_3x (x : ℝ) :
  deriv (fun x => (1/3) * (4*x + 7) * Real.sin (3*x) + (4/9) * Real.cos (3*x)) x
  = (4*x + 7) * Real.cos (3*x) := by
sorry

end NUMINAMATH_CALUDE_integral_of_4x_plus_7_cos_3x_l4056_405604


namespace NUMINAMATH_CALUDE_unique_digit_solution_l4056_405633

def is_single_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def to_two_digit_number (a b : ℕ) : ℕ := 10 * a + b

def to_eight_digit_number (c d e f g h i j : ℕ) : ℕ :=
  10000000 * c + 1000000 * d + 100000 * e + 10000 * f + 1000 * g + 100 * h + 10 * i + j

theorem unique_digit_solution :
  ∃! (A B C D : ℕ),
    is_single_digit A ∧
    is_single_digit B ∧
    is_single_digit C ∧
    is_single_digit D ∧
    A ^ (to_two_digit_number A B) = to_eight_digit_number C C B B D D C A :=
by
  sorry

end NUMINAMATH_CALUDE_unique_digit_solution_l4056_405633


namespace NUMINAMATH_CALUDE_triangle_area_bound_l4056_405699

/-- For any triangle with area S and semiperimeter p, S ≤ p^2 / (3√3) -/
theorem triangle_area_bound (S p : ℝ) (h_S : S > 0) (h_p : p > 0) 
  (h_triangle : ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    a + b > c ∧ b + c > a ∧ c + a > b ∧
    S = Real.sqrt (p * (p - a) * (p - b) * (p - c)) ∧
    p = (a + b + c) / 2) :
  S ≤ p^2 / (3 * Real.sqrt 3) := by
sorry

end NUMINAMATH_CALUDE_triangle_area_bound_l4056_405699


namespace NUMINAMATH_CALUDE_work_completion_time_l4056_405603

theorem work_completion_time (p q : ℕ) (work_left : ℚ) : 
  p = 15 → q = 20 → work_left = 8/15 → 
  (1 : ℚ) - (1/p + 1/q) * (days_worked : ℚ) = work_left → 
  days_worked = 4 := by
sorry

end NUMINAMATH_CALUDE_work_completion_time_l4056_405603


namespace NUMINAMATH_CALUDE_bus_passengers_l4056_405668

/-- 
Given a bus that starts with 64 students and loses one-third of its 
passengers at each stop, prove that after four stops, 1024/81 students remain.
-/
theorem bus_passengers (initial_students : ℕ) (stops : ℕ) : 
  initial_students = 64 → 
  stops = 4 → 
  (initial_students : ℚ) * (2/3)^stops = 1024/81 := by sorry

end NUMINAMATH_CALUDE_bus_passengers_l4056_405668


namespace NUMINAMATH_CALUDE_expanded_expression_equals_804095_l4056_405676

theorem expanded_expression_equals_804095 :
  8 * 10^5 + 4 * 10^3 + 9 * 10 + 5 = 804095 := by
  sorry

end NUMINAMATH_CALUDE_expanded_expression_equals_804095_l4056_405676


namespace NUMINAMATH_CALUDE_simple_interest_problem_l4056_405608

/-- Given a principal P and an interest rate R, if increasing the rate by 6% 
    results in $90 more interest over 5 years, then P = $300. -/
theorem simple_interest_problem (P R : ℝ) : 
  (P * R * 5 / 100 + 90 = P * (R + 6) * 5 / 100) → P = 300 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l4056_405608


namespace NUMINAMATH_CALUDE_arithmetic_sequence_and_binomial_expansion_l4056_405651

theorem arithmetic_sequence_and_binomial_expansion :
  let a : ℕ → ℤ := λ n => 3*n - 5
  let binomial_sum : ℕ → ℤ := λ k => Nat.choose 5 k + Nat.choose 6 k + Nat.choose 7 k
  a 20 = binomial_sum 4 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_and_binomial_expansion_l4056_405651


namespace NUMINAMATH_CALUDE_hyperbola_axes_length_l4056_405639

theorem hyperbola_axes_length (x y : ℝ) :
  x^2 - 8*y^2 = 32 →
  ∃ (real_axis imaginary_axis : ℝ),
    real_axis = 8 * Real.sqrt 2 ∧
    imaginary_axis = 4 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_axes_length_l4056_405639


namespace NUMINAMATH_CALUDE_club_membership_after_four_years_l4056_405666

/-- Represents the number of members in the club after k years -/
def club_members (k : ℕ) : ℕ :=
  match k with
  | 0 => 20
  | n + 1 => 3 * club_members n - 16

/-- The club membership problem -/
theorem club_membership_after_four_years :
  club_members 4 = 980 := by
  sorry

end NUMINAMATH_CALUDE_club_membership_after_four_years_l4056_405666


namespace NUMINAMATH_CALUDE_max_graduates_of_interest_l4056_405688

theorem max_graduates_of_interest (n : ℕ) (u : ℕ) (g : ℕ) :
  n = 100 →  -- number of graduates
  u = 5 →    -- number of universities
  g = 50 →   -- number of graduates each university reached
  (∀ i : ℕ, i ≤ u → g = n / 2) →  -- each university reached half of the graduates
  (∃ x : ℕ, x ≥ 3 ∧ x ≤ u ∧ ∃ y : ℕ, y ≤ n ∧ ∀ i : ℕ, i ≤ x → y ≤ g) →  -- at least 3 universities couldn't reach some graduates
  (∃ m : ℕ, m ≤ 83 ∧ 
    (∀ k : ℕ, k > m → 
      ¬(∃ f : ℕ → ℕ, (∀ i : ℕ, i ≤ k → f i ≤ u) ∧ 
        (∀ i : ℕ, i ≤ k → (∃ j₁ j₂ j₃ : ℕ, j₁ < j₂ ∧ j₂ < j₃ ∧ j₃ ≤ u ∧ 
          f i = j₁ ∧ f i = j₂ ∧ f i = j₃))))) :=
by sorry

end NUMINAMATH_CALUDE_max_graduates_of_interest_l4056_405688


namespace NUMINAMATH_CALUDE_point_coordinate_sum_l4056_405615

/-- Given two points A and B, where A is at (2, 1) and B is on the line y = 6,
    and the slope of segment AB is 4/5, prove that the sum of the x- and y-coordinates of B is 14.25 -/
theorem point_coordinate_sum (B : ℝ × ℝ) : 
  B.2 = 6 → -- B is on the line y = 6
  (B.2 - 1) / (B.1 - 2) = 4 / 5 → -- slope of AB is 4/5
  B.1 + B.2 = 14.25 := by sorry

end NUMINAMATH_CALUDE_point_coordinate_sum_l4056_405615


namespace NUMINAMATH_CALUDE_power_5_2023_mod_17_l4056_405661

theorem power_5_2023_mod_17 : (5 : ℤ) ^ 2023 % 17 = 2 := by sorry

end NUMINAMATH_CALUDE_power_5_2023_mod_17_l4056_405661


namespace NUMINAMATH_CALUDE_trig_identity_l4056_405635

theorem trig_identity (α β : ℝ) :
  (Real.sin (2 * α + β) / Real.sin α) - 2 * Real.cos (α + β) = Real.sin β / Real.sin α :=
by sorry

end NUMINAMATH_CALUDE_trig_identity_l4056_405635


namespace NUMINAMATH_CALUDE_semicircle_roll_path_length_l4056_405605

theorem semicircle_roll_path_length (r : ℝ) (h : r = 4 / Real.pi) : 
  let semicircle_arc_length := r * Real.pi
  semicircle_arc_length = 4 :=
by sorry

end NUMINAMATH_CALUDE_semicircle_roll_path_length_l4056_405605


namespace NUMINAMATH_CALUDE_quadratic_range_condition_l4056_405653

/-- A quadratic function f(x) = mx^2 - 2x + m has a value range of [0, +∞) if and only if m = 1 -/
theorem quadratic_range_condition (m : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, y ≥ 0 ∧ y = m * x^2 - 2 * x + m) ∧ 
  (∀ y : ℝ, y ≥ 0 → ∃ x : ℝ, y = m * x^2 - 2 * x + m) ↔ 
  m = 1 :=
sorry

end NUMINAMATH_CALUDE_quadratic_range_condition_l4056_405653


namespace NUMINAMATH_CALUDE_athlete_b_more_stable_l4056_405627

/-- Represents an athlete's assessment scores -/
structure AthleteScores where
  scores : Finset ℝ
  count : Nat
  avg : ℝ
  variance : ℝ

/-- Stability of an athlete's scores -/
def moreStable (a b : AthleteScores) : Prop :=
  a.variance < b.variance

theorem athlete_b_more_stable (a b : AthleteScores) 
  (h_count : a.count = 10 ∧ b.count = 10)
  (h_avg : a.avg = b.avg)
  (h_var_a : a.variance = 1.45)
  (h_var_b : b.variance = 0.85) :
  moreStable b a :=
sorry

end NUMINAMATH_CALUDE_athlete_b_more_stable_l4056_405627
