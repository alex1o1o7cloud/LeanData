import Mathlib

namespace NUMINAMATH_CALUDE_average_and_relation_implies_values_l170_17049

theorem average_and_relation_implies_values :
  ∀ x y : ℝ,
  (15 + 30 + x + y) / 4 = 25 →
  x = y + 10 →
  x = 32.5 ∧ y = 22.5 := by
sorry

end NUMINAMATH_CALUDE_average_and_relation_implies_values_l170_17049


namespace NUMINAMATH_CALUDE_r_th_term_of_sequence_l170_17016

/-- Given a sequence where the sum of the first n terms is Sn = 3n + 4n^2,
    prove that the r-th term of the sequence is 8r - 1 -/
theorem r_th_term_of_sequence (n r : ℕ) (Sn : ℕ → ℤ) 
  (h : ∀ n, Sn n = 3*n + 4*n^2) :
  Sn r - Sn (r-1) = 8*r - 1 := by
  sorry

end NUMINAMATH_CALUDE_r_th_term_of_sequence_l170_17016


namespace NUMINAMATH_CALUDE_potato_problem_solution_l170_17001

/-- Represents the potato problem with given conditions --/
def potato_problem (total_potatoes wedge_potatoes wedges_per_potato chips_per_potato : ℕ) : Prop :=
  let remaining_potatoes := total_potatoes - wedge_potatoes
  let chip_potatoes := remaining_potatoes / 2
  let total_chips := chip_potatoes * chips_per_potato
  let total_wedges := wedge_potatoes * wedges_per_potato
  total_chips - total_wedges = 436

/-- Theorem stating the solution to the potato problem --/
theorem potato_problem_solution :
  potato_problem 67 13 8 20 := by
  sorry

#check potato_problem_solution

end NUMINAMATH_CALUDE_potato_problem_solution_l170_17001


namespace NUMINAMATH_CALUDE_incorrect_statement_proof_l170_17082

/-- Given non-empty sets A and B where A is not a subset of B, 
    prove that the statement "If x ∉ A, then x ∈ B is an impossible event" is false. -/
theorem incorrect_statement_proof 
  {α : Type*} (A B : Set α) (h_nonempty_A : A.Nonempty) (h_nonempty_B : B.Nonempty) 
  (h_not_subset : ¬(A ⊆ B)) :
  ¬(∀ x, x ∉ A → x ∉ B) :=
sorry

end NUMINAMATH_CALUDE_incorrect_statement_proof_l170_17082


namespace NUMINAMATH_CALUDE_percentage_difference_l170_17064

theorem percentage_difference (A B : ℝ) (h1 : A > 0) (h2 : B > A) :
  let x := 100 * (B - A) / A
  B = A * (1 + x / 100) := by sorry

end NUMINAMATH_CALUDE_percentage_difference_l170_17064


namespace NUMINAMATH_CALUDE_extended_morse_code_symbols_l170_17081

theorem extended_morse_code_symbols : 
  (Finset.range 5).sum (fun n => 2^(n+1)) = 62 := by
  sorry

end NUMINAMATH_CALUDE_extended_morse_code_symbols_l170_17081


namespace NUMINAMATH_CALUDE_exactly_one_absent_l170_17056

-- Define the three guests
variable (B K Z : Prop)

-- B: Baba Yaga comes to the festival
-- K: Koschei comes to the festival
-- Z: Zmey Gorynych comes to the festival

-- Define the conditions
axiom condition1 : ¬B → K
axiom condition2 : ¬K → Z
axiom condition3 : ¬Z → B
axiom at_least_one_absent : ¬B ∨ ¬K ∨ ¬Z

-- Theorem to prove
theorem exactly_one_absent : (¬B ∧ K ∧ Z) ∨ (B ∧ ¬K ∧ Z) ∨ (B ∧ K ∧ ¬Z) :=
sorry

end NUMINAMATH_CALUDE_exactly_one_absent_l170_17056


namespace NUMINAMATH_CALUDE_no_triangle_satisfies_equation_l170_17076

-- Define a structure for a triangle
structure Triangle where
  x : ℝ
  y : ℝ
  z : ℝ
  x_pos : 0 < x
  y_pos : 0 < y
  z_pos : 0 < z
  triangle_ineq1 : x + y > z
  triangle_ineq2 : y + z > x
  triangle_ineq3 : z + x > y

-- Theorem statement
theorem no_triangle_satisfies_equation :
  ¬∃ t : Triangle, t.x^3 + t.y^3 + t.z^3 = (t.x + t.y) * (t.y + t.z) * (t.z + t.x) :=
sorry

end NUMINAMATH_CALUDE_no_triangle_satisfies_equation_l170_17076


namespace NUMINAMATH_CALUDE_polynomial_simplification_l170_17051

theorem polynomial_simplification (x : ℝ) :
  (12 * x^10 + 9 * x^9 + 5 * x^8) + (2 * x^12 + x^10 + 2 * x^9 + 3 * x^8 + 4 * x^4 + 6 * x^2 + 9) =
  2 * x^12 + 13 * x^10 + 11 * x^9 + 8 * x^8 + 4 * x^4 + 6 * x^2 + 9 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l170_17051


namespace NUMINAMATH_CALUDE_students_taking_one_subject_l170_17098

theorem students_taking_one_subject (both : ℕ) (geometry : ℕ) (only_biology : ℕ)
  (h1 : both = 15)
  (h2 : geometry = 40)
  (h3 : only_biology = 20) :
  geometry - both + only_biology = 45 := by
  sorry

end NUMINAMATH_CALUDE_students_taking_one_subject_l170_17098


namespace NUMINAMATH_CALUDE_inequality_sine_square_l170_17062

theorem inequality_sine_square (x : ℝ) (h : x ∈ Set.Ioo 0 (π / 2)) : 
  0 < (1 / Real.sin x ^ 2) - (1 / x ^ 2) ∧ (1 / Real.sin x ^ 2) - (1 / x ^ 2) < 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_sine_square_l170_17062


namespace NUMINAMATH_CALUDE_distance_to_big_rock_l170_17067

/-- The distance to Big Rock given the rower's speed, river current, and round trip time -/
theorem distance_to_big_rock 
  (rower_speed : ℝ) 
  (river_current : ℝ) 
  (round_trip_time : ℝ) 
  (h1 : rower_speed = 7)
  (h2 : river_current = 1)
  (h3 : round_trip_time = 1) : 
  ∃ (distance : ℝ), distance = 24 / 7 ∧ 
    (distance / (rower_speed - river_current) + 
     distance / (rower_speed + river_current) = round_trip_time) :=
by sorry

end NUMINAMATH_CALUDE_distance_to_big_rock_l170_17067


namespace NUMINAMATH_CALUDE_fuel_tank_capacity_l170_17077

theorem fuel_tank_capacity : ∀ (x : ℚ), 
  (5 / 6 : ℚ) * x - (2 / 3 : ℚ) * x = 15 → x = 90 := by
  sorry

end NUMINAMATH_CALUDE_fuel_tank_capacity_l170_17077


namespace NUMINAMATH_CALUDE_largest_common_divisor_408_310_l170_17086

theorem largest_common_divisor_408_310 : Nat.gcd 408 310 = 2 := by
  sorry

end NUMINAMATH_CALUDE_largest_common_divisor_408_310_l170_17086


namespace NUMINAMATH_CALUDE_negation_of_positive_quadratic_inequality_l170_17065

theorem negation_of_positive_quadratic_inequality :
  (¬ ∀ x : ℝ, x > 0 → x^2 + x + 1 > 0) ↔ (∃ x : ℝ, x > 0 ∧ x^2 + x + 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_positive_quadratic_inequality_l170_17065


namespace NUMINAMATH_CALUDE_average_weight_l170_17002

/-- Given three weights a, b, and c, prove that their average is 45 kg
    under the following conditions:
    1. The average of a and b is 40 kg
    2. The average of b and c is 43 kg
    3. The weight of b is 31 kg -/
theorem average_weight (a b c : ℝ) 
  (avg_ab : (a + b) / 2 = 40)
  (avg_bc : (b + c) / 2 = 43)
  (weight_b : b = 31) :
  (a + b + c) / 3 = 45 := by
  sorry


end NUMINAMATH_CALUDE_average_weight_l170_17002


namespace NUMINAMATH_CALUDE_parallelogram_area_l170_17073

-- Define the conversion factor
def inch_to_mm : ℝ := 25.4

-- Define the parallelogram's dimensions
def base_inches : ℝ := 18
def height_mm : ℝ := 25.4

-- Theorem statement
theorem parallelogram_area :
  (base_inches * (height_mm / inch_to_mm)) = 18 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l170_17073


namespace NUMINAMATH_CALUDE_value_calculation_l170_17066

theorem value_calculation : 0.833 * (-72.0) = -59.976 := by
  sorry

end NUMINAMATH_CALUDE_value_calculation_l170_17066


namespace NUMINAMATH_CALUDE_equal_cost_at_48_miles_l170_17030

-- Define the daily rates and per-mile charges
def sunshine_daily_rate : ℝ := 17.99
def sunshine_per_mile : ℝ := 0.18
def city_daily_rate : ℝ := 18.95
def city_per_mile : ℝ := 0.16

-- Define the cost functions for each rental company
def sunshine_cost (miles : ℝ) : ℝ := sunshine_daily_rate + sunshine_per_mile * miles
def city_cost (miles : ℝ) : ℝ := city_daily_rate + city_per_mile * miles

-- Theorem stating that the costs are equal at 48 miles
theorem equal_cost_at_48_miles :
  sunshine_cost 48 = city_cost 48 := by sorry

end NUMINAMATH_CALUDE_equal_cost_at_48_miles_l170_17030


namespace NUMINAMATH_CALUDE_power_division_equals_integer_l170_17015

theorem power_division_equals_integer : 3^18 / 27^3 = 19683 := by sorry

end NUMINAMATH_CALUDE_power_division_equals_integer_l170_17015


namespace NUMINAMATH_CALUDE_school_boys_count_l170_17035

theorem school_boys_count (total_pupils : ℕ) (girls : ℕ) (boys : ℕ) : 
  total_pupils = 485 → girls = 232 → boys = total_pupils - girls → boys = 253 := by
  sorry

end NUMINAMATH_CALUDE_school_boys_count_l170_17035


namespace NUMINAMATH_CALUDE_weight_of_top_l170_17046

/-- Given 9 robots each weighing 0.8 kg and 7 tops with a total weight of 10.98 kg,
    the weight of one top is 0.54 kg. -/
theorem weight_of_top (robot_weight : ℝ) (total_weight : ℝ) (num_robots : ℕ) (num_tops : ℕ) :
  robot_weight = 0.8 →
  num_robots = 9 →
  num_tops = 7 →
  total_weight = 10.98 →
  total_weight = (↑num_robots * robot_weight) + (↑num_tops * 0.54) :=
by sorry

end NUMINAMATH_CALUDE_weight_of_top_l170_17046


namespace NUMINAMATH_CALUDE_z_max_min_difference_l170_17079

theorem z_max_min_difference (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x ≠ y) : 
  let z := fun (a b : ℝ) => |a^2 - b^2| / (|a^2| + |b^2|)
  ∃ (max min : ℝ), 
    (∀ a b, a ≠ 0 → b ≠ 0 → a ≠ b → z a b ≤ max) ∧
    (∃ a b, a ≠ 0 ∧ b ≠ 0 ∧ a ≠ b ∧ z a b = max) ∧
    (∀ a b, a ≠ 0 → b ≠ 0 → a ≠ b → min ≤ z a b) ∧
    (∃ a b, a ≠ 0 ∧ b ≠ 0 ∧ a ≠ b ∧ z a b = min) ∧
    max = 1 ∧ min = 0 ∧ max - min = 1 :=
by sorry

end NUMINAMATH_CALUDE_z_max_min_difference_l170_17079


namespace NUMINAMATH_CALUDE_discount_percentage_l170_17026

theorem discount_percentage (regular_price : ℝ) (num_shirts : ℕ) (total_paid : ℝ) : 
  regular_price = 50 ∧ num_shirts = 2 ∧ total_paid = 60 →
  (regular_price * num_shirts - total_paid) / (regular_price * num_shirts) * 100 = 40 := by
sorry

end NUMINAMATH_CALUDE_discount_percentage_l170_17026


namespace NUMINAMATH_CALUDE_prob_catch_carp_l170_17044

/-- The probability of catching a carp in a pond with given conditions -/
theorem prob_catch_carp (num_carp num_tilapia : ℕ) (prob_grass_carp : ℚ) : 
  num_carp = 1600 →
  num_tilapia = 800 →
  prob_grass_carp = 1/2 →
  (num_carp : ℚ) / (num_carp + num_tilapia + (prob_grass_carp⁻¹ - 1) * (num_carp + num_tilapia)) = 1/3 :=
by sorry

end NUMINAMATH_CALUDE_prob_catch_carp_l170_17044


namespace NUMINAMATH_CALUDE_binomial_theorem_and_sum_l170_17072

def binomial_expansion (m : ℝ) : ℕ → ℝ
| 0 => 1
| 1 => 7 * m
| 2 => 21 * m^2
| 3 => 35 * m^3
| 4 => 35 * m^4
| 5 => 21 * m^5
| 6 => 7 * m^6
| 7 => m^7
| _ => 0

def a (m : ℝ) (i : ℕ) : ℝ := binomial_expansion m i

theorem binomial_theorem_and_sum (m : ℝ) :
  a m 3 = -280 →
  (m = -2 ∧ a m 1 + a m 3 + a m 5 + a m 7 = -1094) := by sorry

end NUMINAMATH_CALUDE_binomial_theorem_and_sum_l170_17072


namespace NUMINAMATH_CALUDE_brothers_ticket_cost_l170_17038

/-- Proves that each brother's ticket costs $10 given the problem conditions -/
theorem brothers_ticket_cost (isabelle_ticket_cost : ℕ) 
  (total_savings : ℕ) (weeks_worked : ℕ) (wage_per_week : ℕ) :
  isabelle_ticket_cost = 20 →
  total_savings = 10 →
  weeks_worked = 10 →
  wage_per_week = 3 →
  let total_money := total_savings + weeks_worked * wage_per_week
  let remaining_money := total_money - isabelle_ticket_cost
  remaining_money / 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_brothers_ticket_cost_l170_17038


namespace NUMINAMATH_CALUDE_units_digit_of_17_power_2007_l170_17071

theorem units_digit_of_17_power_2007 : 17^2007 % 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_17_power_2007_l170_17071


namespace NUMINAMATH_CALUDE_sum_of_powers_positive_l170_17025

theorem sum_of_powers_positive 
  (a b c : ℝ) 
  (h1 : a * b * c > 0) 
  (h2 : a + b + c > 0) : 
  ∀ n : ℕ, a^n + b^n + c^n > 0 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_powers_positive_l170_17025


namespace NUMINAMATH_CALUDE_brownie_pieces_theorem_l170_17017

/-- The number of big square pieces the brownies were cut into -/
def num_pieces : ℕ := sorry

/-- The total amount of money Tamara made from selling brownies -/
def total_amount : ℕ := 32

/-- The cost of each brownie -/
def cost_per_brownie : ℕ := 2

/-- The number of pans of brownies made -/
def num_pans : ℕ := 2

theorem brownie_pieces_theorem :
  num_pieces = total_amount / cost_per_brownie :=
sorry

end NUMINAMATH_CALUDE_brownie_pieces_theorem_l170_17017


namespace NUMINAMATH_CALUDE_cubic_kilometer_to_cubic_meters_l170_17083

/-- Given that one kilometer equals 1000 meters, prove that one cubic kilometer equals 1,000,000,000 cubic meters. -/
theorem cubic_kilometer_to_cubic_meters :
  (1 : ℝ) * (1000 : ℝ)^3 = (1000000000 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_cubic_kilometer_to_cubic_meters_l170_17083


namespace NUMINAMATH_CALUDE_cos_24_cos_36_minus_sin_24_cos_54_l170_17060

theorem cos_24_cos_36_minus_sin_24_cos_54 : 
  Real.cos (24 * π / 180) * Real.cos (36 * π / 180) - 
  Real.sin (24 * π / 180) * Real.cos (54 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_24_cos_36_minus_sin_24_cos_54_l170_17060


namespace NUMINAMATH_CALUDE_set_operations_l170_17048

def M : Set ℝ := {x | 4 * x^2 - 4 * x - 15 > 0}

def N : Set ℝ := {x | (x + 1) / (6 - x) < 0}

theorem set_operations (M N : Set ℝ) :
  (M = {x | 4 * x^2 - 4 * x - 15 > 0}) →
  (N = {x | (x + 1) / (6 - x) < 0}) →
  (M ∪ N = {x | x < -1 ∨ x ≥ 5/2}) ∧
  ((Set.univ \ M) ∩ (Set.univ \ N) = {x | -1 ≤ x ∧ x < 5/2}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l170_17048


namespace NUMINAMATH_CALUDE_arithmetic_progression_squares_l170_17097

theorem arithmetic_progression_squares (a b c : ℝ) 
  (h : (1 / (a + b) + 1 / (b + c)) / 2 = 1 / (a + c)) : 
  a^2 + c^2 = 2 * b^2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_squares_l170_17097


namespace NUMINAMATH_CALUDE_square_diagonal_l170_17054

theorem square_diagonal (area : ℝ) (side : ℝ) (diagonal : ℝ) 
  (h1 : area = 4802) 
  (h2 : area = side ^ 2) 
  (h3 : diagonal ^ 2 = 2 * side ^ 2) : 
  diagonal = Real.sqrt (2 * 4802) := by
sorry

end NUMINAMATH_CALUDE_square_diagonal_l170_17054


namespace NUMINAMATH_CALUDE_susie_fish_count_l170_17084

/-- The number of fish caught by each family member and the total number of filets --/
structure FishingTrip where
  ben_fish : ℕ
  judy_fish : ℕ
  billy_fish : ℕ
  jim_fish : ℕ
  susie_fish : ℕ
  thrown_back : ℕ
  total_filets : ℕ

/-- Theorem stating that Susie caught 3 fish given the conditions of the fishing trip --/
theorem susie_fish_count (trip : FishingTrip) 
  (h1 : trip.ben_fish = 4)
  (h2 : trip.judy_fish = 1)
  (h3 : trip.billy_fish = 3)
  (h4 : trip.jim_fish = 2)
  (h5 : trip.thrown_back = 3)
  (h6 : trip.total_filets = 24)
  (h7 : ∀ (fish : ℕ), fish * 2 = trip.total_filets → 
    fish = trip.ben_fish + trip.judy_fish + trip.billy_fish + trip.jim_fish + trip.susie_fish - trip.thrown_back) :
  trip.susie_fish = 3 := by
  sorry

end NUMINAMATH_CALUDE_susie_fish_count_l170_17084


namespace NUMINAMATH_CALUDE_point_existence_and_uniqueness_l170_17093

theorem point_existence_and_uniqueness :
  ∃! (x y : ℝ), 
    y = 8 ∧ 
    (x - 3)^2 + (y - 9)^2 = 12^2 ∧ 
    x^2 + y^2 = 14^2 ∧ 
    x > 3 := by
  sorry

end NUMINAMATH_CALUDE_point_existence_and_uniqueness_l170_17093


namespace NUMINAMATH_CALUDE_melanie_picked_seven_plums_l170_17012

/-- The number of plums Melanie picked from the orchard -/
def plums_picked : ℕ := sorry

/-- The number of plums Sam gave to Melanie -/
def plums_from_sam : ℕ := 3

/-- The total number of plums Melanie has now -/
def total_plums : ℕ := 10

/-- Theorem stating that Melanie picked 7 plums from the orchard -/
theorem melanie_picked_seven_plums :
  plums_picked = 7 ∧ plums_picked + plums_from_sam = total_plums :=
sorry

end NUMINAMATH_CALUDE_melanie_picked_seven_plums_l170_17012


namespace NUMINAMATH_CALUDE_problem_solution_l170_17058

theorem problem_solution (a b c d : ℝ) 
  (h : a^2 + b^2 + c^2 + 2 = d + (a + b + c - d)^(1/3)) : 
  d = 1/2 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l170_17058


namespace NUMINAMATH_CALUDE_reflection_matrix_values_l170_17010

theorem reflection_matrix_values (a b : ℚ) :
  let R : Matrix (Fin 2) (Fin 2) ℚ := !![a, 9/26; b, 17/26]
  (R * R = 1) → (a = -17/26 ∧ b = 0) := by
  sorry

end NUMINAMATH_CALUDE_reflection_matrix_values_l170_17010


namespace NUMINAMATH_CALUDE_factor_sum_l170_17005

theorem factor_sum (P Q : ℝ) : 
  (∃ b c : ℝ, (X^2 + 3*X + 4) * (X^2 + b*X + c) = X^4 + P*X^2 + Q) → 
  P + Q = 15 := by
sorry

end NUMINAMATH_CALUDE_factor_sum_l170_17005


namespace NUMINAMATH_CALUDE_sum_after_100_operations_l170_17055

/-- The operation that inserts the difference between each pair of neighboring numbers -/
def insertDifferences (s : List Int) : List Int :=
  sorry

/-- Applies the insertDifferences operation n times to a list -/
def applyNTimes (s : List Int) (n : Nat) : List Int :=
  sorry

/-- The sum of a list of integers -/
def listSum (s : List Int) : Int :=
  sorry

theorem sum_after_100_operations :
  let initialSequence : List Int := [1, 9, 8, 8]
  listSum (applyNTimes initialSequence 100) = 726 := by
  sorry

end NUMINAMATH_CALUDE_sum_after_100_operations_l170_17055


namespace NUMINAMATH_CALUDE_inequalities_not_equivalent_l170_17069

theorem inequalities_not_equivalent : 
  ¬(∀ x : ℝ, (x - 3) / (x^2 - 5*x + 6) < 2 ↔ 2*x^2 - 11*x + 15 > 0) :=
by sorry

end NUMINAMATH_CALUDE_inequalities_not_equivalent_l170_17069


namespace NUMINAMATH_CALUDE_marble_distribution_l170_17013

/-- Given the distribution of marbles between two classes, prove the difference between
    the number of marbles each male in Class 2 receives and the total number of marbles
    taken by Class 1. -/
theorem marble_distribution (total_marbles : ℕ) (class1_marbles : ℕ) (class2_marbles : ℕ)
  (boys_marbles : ℕ) (girls_marbles : ℕ) (num_boys : ℕ) :
  total_marbles = 1000 →
  class1_marbles = class2_marbles + 50 →
  class1_marbles + class2_marbles = total_marbles →
  boys_marbles = girls_marbles + 35 →
  boys_marbles + girls_marbles = class2_marbles →
  num_boys = 17 →
  class1_marbles - (boys_marbles / num_boys) = 510 :=
by sorry

end NUMINAMATH_CALUDE_marble_distribution_l170_17013


namespace NUMINAMATH_CALUDE_father_daughter_speed_problem_l170_17041

theorem father_daughter_speed_problem 
  (total_distance : ℝ) 
  (speed_ratio : ℝ) 
  (speed_increase : ℝ) 
  (time_difference : ℝ) :
  total_distance = 60 ∧ 
  speed_ratio = 2 ∧ 
  speed_increase = 2 ∧ 
  time_difference = 1/12 →
  ∃ (father_speed daughter_speed : ℝ),
    father_speed = 14 ∧ 
    daughter_speed = 28 ∧
    daughter_speed = speed_ratio * father_speed ∧
    (total_distance / (2 * father_speed + speed_increase) - 
     (total_distance / 2) / (father_speed + speed_increase)) = time_difference :=
by sorry

end NUMINAMATH_CALUDE_father_daughter_speed_problem_l170_17041


namespace NUMINAMATH_CALUDE_range_of_p_l170_17047

-- Define the set A
def A (p : ℝ) : Set ℝ := {x | x^2 + (p+2)*x + 1 = 0}

-- Theorem statement
theorem range_of_p (p : ℝ) : (A p ∩ Set.Ioi 0 = ∅) → p > -4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_p_l170_17047


namespace NUMINAMATH_CALUDE_function_property_l170_17020

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then Real.sqrt x + 3 else a * x + b

theorem function_property (a b : ℝ) :
  (∀ x₁ : ℝ, x₁ ≠ 0 → ∃! x₂ : ℝ, x₁ ≠ x₂ ∧ f a b x₁ = f a b x₂) →
  f a b (2 * a) = f a b (3 * b) →
  a + b = -Real.sqrt 6 / 2 + 3 := by
  sorry

end NUMINAMATH_CALUDE_function_property_l170_17020


namespace NUMINAMATH_CALUDE_f_greater_than_exp_l170_17078

open Set
open Function
open Real

-- Define the function f
variable (f : ℝ → ℝ)

-- State the conditions
variable (h1 : Differentiable ℝ f)
variable (h2 : ∀ x, deriv f x > f x)
variable (h3 : f 0 = 1)

-- Theorem statement
theorem f_greater_than_exp (x : ℝ) : f x > Real.exp x ↔ x > 0 := by
  sorry

end NUMINAMATH_CALUDE_f_greater_than_exp_l170_17078


namespace NUMINAMATH_CALUDE_unique_function_solution_l170_17003

-- Define the property that f must satisfy
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x - f y) = 1 - x - y

-- State the theorem
theorem unique_function_solution :
  ∀ f : ℝ → ℝ, SatisfiesEquation f → ∀ x : ℝ, f x = 1/2 - x :=
by sorry

end NUMINAMATH_CALUDE_unique_function_solution_l170_17003


namespace NUMINAMATH_CALUDE_prob_same_color_is_89_169_l170_17099

def total_balls : ℕ := 13
def blue_balls : ℕ := 8
def yellow_balls : ℕ := 5

def prob_same_color : ℚ :=
  (blue_balls * blue_balls + yellow_balls * yellow_balls) / (total_balls * total_balls)

theorem prob_same_color_is_89_169 : prob_same_color = 89 / 169 := by
  sorry

end NUMINAMATH_CALUDE_prob_same_color_is_89_169_l170_17099


namespace NUMINAMATH_CALUDE_inequality_implication_l170_17021

theorem inequality_implication (a b : ℝ) (h : a > b) : a + 2 > b + 1 := by
  sorry

#check inequality_implication

end NUMINAMATH_CALUDE_inequality_implication_l170_17021


namespace NUMINAMATH_CALUDE_half_abs_diff_squares_20_15_l170_17019

theorem half_abs_diff_squares_20_15 : (1/2 : ℝ) * |20^2 - 15^2| = 87.5 := by
  sorry

end NUMINAMATH_CALUDE_half_abs_diff_squares_20_15_l170_17019


namespace NUMINAMATH_CALUDE_square_perimeter_sum_l170_17022

theorem square_perimeter_sum (x y : ℝ) (h1 : x^2 + y^2 = 130) (h2 : x^2 - y^2 = 42) :
  4*x + 4*y = 4*Real.sqrt 86 + 8*Real.sqrt 11 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_sum_l170_17022


namespace NUMINAMATH_CALUDE_ellipse_condition_necessary_not_sufficient_l170_17004

/-- The condition for the equation to potentially represent an ellipse -/
def ellipse_condition (m : ℝ) : Prop := 1 < m ∧ m < 3

/-- The equation representing a potential ellipse -/
def ellipse_equation (m x y : ℝ) : Prop :=
  x^2 / (m - 1) + y^2 / (3 - m) = 1

/-- Predicate for whether the equation represents an ellipse -/
def is_ellipse (m : ℝ) : Prop :=
  ∃ x y : ℝ, ellipse_equation m x y ∧ 
  ¬(∃ c : ℝ, ∀ x y : ℝ, ellipse_equation m x y ↔ x^2 + y^2 = c)

theorem ellipse_condition_necessary_not_sufficient :
  (∀ m : ℝ, is_ellipse m → ellipse_condition m) ∧
  ¬(∀ m : ℝ, ellipse_condition m → is_ellipse m) :=
sorry

end NUMINAMATH_CALUDE_ellipse_condition_necessary_not_sufficient_l170_17004


namespace NUMINAMATH_CALUDE_arctan_sum_equals_pi_fourth_l170_17057

theorem arctan_sum_equals_pi_fourth (y : ℝ) : 
  2 * Real.arctan (1/3) + Real.arctan (1/7) + Real.arctan (1/y) = π/4 → y = 2 := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_equals_pi_fourth_l170_17057


namespace NUMINAMATH_CALUDE_gcd_sum_diff_l170_17040

theorem gcd_sum_diff (a b : ℕ) (h1 : a > b) (h2 : Nat.gcd a b = 1) :
  (Nat.gcd (a + b) (a - b) = 1) ∨ (Nat.gcd (a + b) (a - b) = 2) :=
sorry

end NUMINAMATH_CALUDE_gcd_sum_diff_l170_17040


namespace NUMINAMATH_CALUDE_x_coordinate_difference_at_y_20_l170_17028

/-- A line in a 2D coordinate system --/
structure Line where
  slope : ℚ
  intercept : ℚ

/-- Calculate the x-coordinate for a given y-coordinate on a line --/
def xCoordAtY (line : Line) (y : ℚ) : ℚ :=
  (y - line.intercept) / line.slope

/-- Create a line from two points --/
def lineFromPoints (x1 y1 x2 y2 : ℚ) : Line where
  slope := (y2 - y1) / (x2 - x1)
  intercept := y1 - (y2 - y1) / (x2 - x1) * x1

theorem x_coordinate_difference_at_y_20 :
  let l := lineFromPoints 0 5 3 0
  let m := lineFromPoints 0 4 6 0
  let x_l := xCoordAtY l 20
  let x_m := xCoordAtY m 20
  |x_l - x_m| = 15 := by sorry

end NUMINAMATH_CALUDE_x_coordinate_difference_at_y_20_l170_17028


namespace NUMINAMATH_CALUDE_inequality_always_true_l170_17090

theorem inequality_always_true (a b c d : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > d) :
  a + c > b + d := by
  sorry

end NUMINAMATH_CALUDE_inequality_always_true_l170_17090


namespace NUMINAMATH_CALUDE_incorrect_exponent_operation_l170_17053

theorem incorrect_exponent_operation (a : ℝ) : (-a^2)^3 ≠ -a^5 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_exponent_operation_l170_17053


namespace NUMINAMATH_CALUDE_circumcircle_tangent_to_excircle_l170_17061

-- Define the points and circles
variable (A B C D E B₁ C₁ I J S : Point)
variable (Ω : Circle)

-- Define the convex quadrilateral ABCD
def is_convex_quadrilateral (A B C D : Point) : Prop := sorry

-- Define the intersection of diagonals
def diagonals_intersect_at (A B C D E : Point) : Prop := sorry

-- Define the common excircle of triangles
def common_excircle (A B C D E : Point) (Ω : Circle) : Prop := sorry

-- Define tangent points
def tangent_points (A E D B₁ C₁ : Point) (Ω : Circle) : Prop := sorry

-- Define incircle centers
def incircle_centers (A B E C D I J : Point) : Prop := sorry

-- Define intersection of IC₁ and JB₁
def segments_intersect_at (I C₁ J B₁ S : Point) : Prop := sorry

-- Define S lying on Ω
def point_on_circle (S : Point) (Ω : Circle) : Prop := sorry

-- Define the circumcircle of a triangle
def circumcircle (A E D : Point) : Circle := sorry

-- Define tangency of circles
def circles_tangent (c₁ c₂ : Circle) : Prop := sorry

-- Theorem statement
theorem circumcircle_tangent_to_excircle 
  (h₁ : is_convex_quadrilateral A B C D)
  (h₂ : diagonals_intersect_at A B C D E)
  (h₃ : common_excircle A B C D E Ω)
  (h₄ : tangent_points A E D B₁ C₁ Ω)
  (h₅ : incircle_centers A B E C D I J)
  (h₆ : segments_intersect_at I C₁ J B₁ S)
  (h₇ : point_on_circle S Ω) :
  circles_tangent (circumcircle A E D) Ω :=
sorry

end NUMINAMATH_CALUDE_circumcircle_tangent_to_excircle_l170_17061


namespace NUMINAMATH_CALUDE_paris_weekday_study_hours_l170_17007

/-- The number of hours Paris studies each weekday. -/
def weekday_study_hours : ℝ := 3

/-- The number of weeks in the fall semester. -/
def semester_weeks : ℕ := 15

/-- The number of hours Paris studies on Saturday. -/
def saturday_study_hours : ℝ := 4

/-- The number of hours Paris studies on Sunday. -/
def sunday_study_hours : ℝ := 5

/-- The total number of hours Paris studies during the semester. -/
def total_study_hours : ℝ := 360

/-- Theorem stating that the number of hours Paris studies each weekday is 3. -/
theorem paris_weekday_study_hours :
  weekday_study_hours * (5 * semester_weeks) +
  (saturday_study_hours + sunday_study_hours) * semester_weeks =
  total_study_hours :=
sorry

end NUMINAMATH_CALUDE_paris_weekday_study_hours_l170_17007


namespace NUMINAMATH_CALUDE_triangle_side_length_l170_17075

/-- Given a triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively,
    if the area is √3, B = 60°, and a² + c² = 3ac, then b = 2√2. -/
theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) : 
  (1/2 : ℝ) * a * c * Real.sin B = Real.sqrt 3 →
  B = π / 3 →
  a^2 + c^2 = 3 * a * c →
  b = 2 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_triangle_side_length_l170_17075


namespace NUMINAMATH_CALUDE_number_problem_l170_17088

theorem number_problem (N : ℝ) : 
  (1/2 : ℝ) * ((3/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * N) = 45 → 
  (65/100 : ℝ) * N = 585 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l170_17088


namespace NUMINAMATH_CALUDE_average_growth_rate_proof_l170_17037

def initial_sales : ℝ := 50000
def final_sales : ℝ := 72000
def time_period : ℝ := 2

theorem average_growth_rate_proof :
  (final_sales / initial_sales) ^ (1 / time_period) - 1 = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_average_growth_rate_proof_l170_17037


namespace NUMINAMATH_CALUDE_one_intersection_values_l170_17092

/-- The function f(x) parameterized by m -/
def f (m : ℝ) (x : ℝ) : ℝ := (m - 4) * x^2 - 2 * m * x - m - 6

/-- The discriminant of the quadratic function f(x) -/
def discriminant (m : ℝ) : ℝ := 4 * m^2 - 4 * (m - 4) * (-m - 6)

/-- Predicate to check if f(x) has only one intersection with x-axis -/
def has_one_intersection (m : ℝ) : Prop :=
  (m = 4) ∨ (discriminant m = 0)

/-- Theorem stating the values of m for which f(x) has one intersection with x-axis -/
theorem one_intersection_values :
  ∀ m : ℝ, has_one_intersection m ↔ m = -4 ∨ m = 3 ∨ m = 4 :=
sorry

end NUMINAMATH_CALUDE_one_intersection_values_l170_17092


namespace NUMINAMATH_CALUDE_not_tangent_implies_a_less_than_one_third_l170_17063

/-- The function f(x) = x³ - 3ax --/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*a*x

/-- The derivative of f(x) --/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 3*a

/-- Theorem stating that if the line x + y + m = 0 is not a tangent to y = f(x) for any m,
    then a < 1/3 --/
theorem not_tangent_implies_a_less_than_one_third (a : ℝ) :
  (∀ m : ℝ, ¬∃ x : ℝ, f_derivative a x = -1 ∧ f a x = -(x + m)) →
  a < 1/3 :=
sorry

end NUMINAMATH_CALUDE_not_tangent_implies_a_less_than_one_third_l170_17063


namespace NUMINAMATH_CALUDE_one_nonneg_solution_iff_l170_17036

/-- The quadratic equation with parameter a -/
def quadratic (a : ℝ) (x : ℝ) : ℝ := (a - 1) * x^2 - 2 * (a + 1) * x + 2 * (a + 1)

/-- The condition for having exactly one non-negative solution -/
def has_one_nonneg_solution (a : ℝ) : Prop :=
  ∃! x : ℝ, x ≥ 0 ∧ quadratic a x = 0

/-- The theorem stating the condition on parameter a -/
theorem one_nonneg_solution_iff (a : ℝ) :
  has_one_nonneg_solution a ↔ (-1 ≤ a ∧ a ≤ 1) ∨ a = 3 := by sorry

end NUMINAMATH_CALUDE_one_nonneg_solution_iff_l170_17036


namespace NUMINAMATH_CALUDE_trailer_count_proof_l170_17031

theorem trailer_count_proof (initial_count : ℕ) (initial_avg_age : ℝ) (current_avg_age : ℝ) :
  initial_count = 30 ∧ initial_avg_age = 12 ∧ current_avg_age = 10 →
  ∃ (new_count : ℕ), 
    (initial_count * (initial_avg_age + 4) + new_count * 4) / (initial_count + new_count) = current_avg_age ∧
    new_count = 30 := by
  sorry

end NUMINAMATH_CALUDE_trailer_count_proof_l170_17031


namespace NUMINAMATH_CALUDE_probability_neither_red_nor_purple_l170_17068

def total_balls : ℕ := 60
def white_balls : ℕ := 22
def green_balls : ℕ := 18
def yellow_balls : ℕ := 5
def red_balls : ℕ := 6
def purple_balls : ℕ := 9

theorem probability_neither_red_nor_purple :
  (total_balls - (red_balls + purple_balls)) / total_balls = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_probability_neither_red_nor_purple_l170_17068


namespace NUMINAMATH_CALUDE_binomial_10_2_l170_17014

theorem binomial_10_2 : Nat.choose 10 2 = 45 := by
  sorry

end NUMINAMATH_CALUDE_binomial_10_2_l170_17014


namespace NUMINAMATH_CALUDE_roots_sum_product_equal_l170_17059

theorem roots_sum_product_equal (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ 
    x^2 - (m+2)*x + m^2 = 0 ∧ 
    y^2 - (m+2)*y + m^2 = 0 ∧ 
    x + y = x * y) → 
  m = 2 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_product_equal_l170_17059


namespace NUMINAMATH_CALUDE_largest_b_in_box_l170_17050

theorem largest_b_in_box (a b c : ℕ) : 
  (a * b * c = 360) →
  (1 < c) → (c < b) → (b < a) →
  (∀ a' b' c' : ℕ, (a' * b' * c' = 360) → (1 < c') → (c' < b') → (b' < a') → b' ≤ b) →
  b = 12 :=
sorry

end NUMINAMATH_CALUDE_largest_b_in_box_l170_17050


namespace NUMINAMATH_CALUDE_unique_solution_for_2n_plus_1_eq_m2_l170_17096

theorem unique_solution_for_2n_plus_1_eq_m2 :
  ∃! (m n : ℕ), 2^n + 1 = m^2 ∧ m = 3 ∧ n = 3 := by sorry

end NUMINAMATH_CALUDE_unique_solution_for_2n_plus_1_eq_m2_l170_17096


namespace NUMINAMATH_CALUDE_multiple_of_112_implies_multiple_of_8_l170_17074

theorem multiple_of_112_implies_multiple_of_8 (n : ℤ) : 
  (∃ k : ℤ, 14 * n = 112 * k) → (∃ m : ℤ, n = 8 * m) := by
  sorry

end NUMINAMATH_CALUDE_multiple_of_112_implies_multiple_of_8_l170_17074


namespace NUMINAMATH_CALUDE_parkway_elementary_soccer_l170_17094

theorem parkway_elementary_soccer (total_students : ℕ) (boys : ℕ) (soccer_players : ℕ) 
  (boys_soccer_percentage : ℚ) :
  total_students = 420 →
  boys = 296 →
  soccer_players = 250 →
  boys_soccer_percentage = 86 / 100 →
  (total_students - boys) - (soccer_players - (boys_soccer_percentage * soccer_players).floor) = 89 := by
  sorry

end NUMINAMATH_CALUDE_parkway_elementary_soccer_l170_17094


namespace NUMINAMATH_CALUDE_actual_distance_traveled_l170_17034

/-- Proves that the actual distance traveled is 20 km given the conditions -/
theorem actual_distance_traveled (initial_speed time_taken : ℝ) 
  (h1 : initial_speed = 5)
  (h2 : initial_speed * time_taken + 20 = 2 * initial_speed * time_taken) :
  initial_speed * time_taken = 20 := by
  sorry

#check actual_distance_traveled

end NUMINAMATH_CALUDE_actual_distance_traveled_l170_17034


namespace NUMINAMATH_CALUDE_min_rectangle_dimensions_l170_17043

/-- A rectangle with length twice its width and area at least 500 square feet has minimum dimensions of width = 5√10 feet and length = 10√10 feet. -/
theorem min_rectangle_dimensions (w : ℝ) (h : w > 0) :
  (2 * w ^ 2 ≥ 500) → (∀ x > 0, 2 * x ^ 2 ≥ 500 → w ≤ x) → w = 5 * Real.sqrt 10 :=
by sorry

end NUMINAMATH_CALUDE_min_rectangle_dimensions_l170_17043


namespace NUMINAMATH_CALUDE_mandy_jackson_age_difference_l170_17070

/-- Proves that Mandy is 10 years older than Jackson given the conditions of the problem -/
theorem mandy_jackson_age_difference :
  ∀ (mandy_age jackson_age adele_age : ℕ),
    jackson_age = 20 →
    adele_age = (3 * jackson_age) / 4 →
    mandy_age + jackson_age + adele_age + 30 = 95 →
    mandy_age > jackson_age →
    mandy_age - jackson_age = 10 := by
  sorry

end NUMINAMATH_CALUDE_mandy_jackson_age_difference_l170_17070


namespace NUMINAMATH_CALUDE_N_rightmost_ten_l170_17032

/-- A number with 1999 digits where each pair of consecutive digits
    is either a multiple of 17 or 23, and the sum of all digits is 9599 -/
def N : ℕ :=
  sorry

/-- Checks if a two-digit number is a multiple of 17 or 23 -/
def is_valid_pair (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100 ∧ (n % 17 = 0 ∨ n % 23 = 0)

/-- The property that each pair of consecutive digits in N
    is either a multiple of 17 or 23 -/
def valid_pairs (n : ℕ) : Prop :=
  ∀ i, i < 1998 → is_valid_pair ((n / 10^i) % 100)

/-- The sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ :=
  sorry

/-- The rightmost ten digits of a natural number -/
def rightmost_ten (n : ℕ) : ℕ :=
  n % 10^10

theorem N_rightmost_ten :
  N ≥ 10^1998 ∧
  N < 10^1999 ∧
  valid_pairs N ∧
  digit_sum N = 9599 →
  rightmost_ten N = 3469234685 :=
sorry

end NUMINAMATH_CALUDE_N_rightmost_ten_l170_17032


namespace NUMINAMATH_CALUDE_johns_primary_colors_l170_17006

/-- Given that John has 5 liters of paint for each color and 15 liters of paint in total,
    prove that the number of primary colors he is using is 3. -/
theorem johns_primary_colors (paint_per_color : ℝ) (total_paint : ℝ) 
    (h1 : paint_per_color = 5)
    (h2 : total_paint = 15) :
    total_paint / paint_per_color = 3 := by
  sorry

end NUMINAMATH_CALUDE_johns_primary_colors_l170_17006


namespace NUMINAMATH_CALUDE_inequality_solution_range_l170_17009

theorem inequality_solution_range (m : ℚ) : 
  (∃! (s : Finset ℤ), s.card = 3 ∧ 
   (∀ x ∈ s, x < 0 ∧ (x - 1) / 2 + 3 > (x + m) / 3) ∧
   (∀ x : ℤ, x < 0 → (x - 1) / 2 + 3 > (x + m) / 3 → x ∈ s)) ↔ 
  (11 / 2 : ℚ) ≤ m ∧ m < 6 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l170_17009


namespace NUMINAMATH_CALUDE_factorization_equality_l170_17011

theorem factorization_equality (a x y : ℝ) : 
  a^2 * (x - y) + 9 * (y - x) = (x - y) * (a + 3) * (a - 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l170_17011


namespace NUMINAMATH_CALUDE_B_power_200_is_identity_l170_17029

def B : Matrix (Fin 4) (Fin 4) ℝ := !![0,0,0,1; 1,0,0,0; 0,1,0,0; 0,0,1,0]

theorem B_power_200_is_identity :
  B ^ 200 = (1 : Matrix (Fin 4) (Fin 4) ℝ) := by
  sorry

end NUMINAMATH_CALUDE_B_power_200_is_identity_l170_17029


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_l170_17052

theorem sum_of_a_and_b (a b : ℚ) 
  (eq1 : 3 * a + 5 * b = 47) 
  (eq2 : 4 * a + 2 * b = 38) : 
  a + b = 85 / 7 := by
sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_l170_17052


namespace NUMINAMATH_CALUDE_jessica_paper_count_l170_17042

def paper_weight : ℚ := 1/5
def envelope_weight : ℚ := 2/5

def total_weight (num_papers : ℕ) : ℚ :=
  paper_weight * num_papers + envelope_weight

theorem jessica_paper_count :
  ∃ (num_papers : ℕ),
    (1 < total_weight num_papers) ∧
    (total_weight num_papers ≤ 2) ∧
    (num_papers = 8) := by
  sorry

end NUMINAMATH_CALUDE_jessica_paper_count_l170_17042


namespace NUMINAMATH_CALUDE_death_rate_calculation_l170_17091

/-- The average birth rate in people per two seconds -/
def average_birth_rate : ℕ := 10

/-- The population net increase in one day -/
def population_net_increase : ℕ := 345600

/-- The number of seconds in a day -/
def seconds_per_day : ℕ := 24 * 60 * 60

/-- The average death rate in people per two seconds -/
def average_death_rate : ℕ := 2

theorem death_rate_calculation :
  average_birth_rate - average_death_rate = 
    2 * (population_net_increase / seconds_per_day) :=
by sorry

end NUMINAMATH_CALUDE_death_rate_calculation_l170_17091


namespace NUMINAMATH_CALUDE_expression_evaluation_l170_17024

theorem expression_evaluation (d : ℕ) (h : d = 2) : 
  (d^d + d*(d+1)^d)^d = 484 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l170_17024


namespace NUMINAMATH_CALUDE_years_until_double_age_l170_17023

/-- Proves the number of years until a man's age is twice his son's age -/
theorem years_until_double_age (son_age : ℕ) (age_difference : ℕ) (years : ℕ) : 
  son_age = 44 → 
  age_difference = 46 → 
  (son_age + age_difference + years) = 2 * (son_age + years) → 
  years = 2 := by
sorry

end NUMINAMATH_CALUDE_years_until_double_age_l170_17023


namespace NUMINAMATH_CALUDE_exactly_one_even_l170_17000

theorem exactly_one_even (a b c : ℕ) : 
  ¬((a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 1) ∨ 
    (a % 2 = 0 ∧ b % 2 = 0) ∨ 
    (a % 2 = 0 ∧ c % 2 = 0) ∨ 
    (b % 2 = 0 ∧ c % 2 = 0)) → 
  ((a % 2 = 0 ∧ b % 2 = 1 ∧ c % 2 = 1) ∨
   (a % 2 = 1 ∧ b % 2 = 0 ∧ c % 2 = 1) ∨
   (a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 0)) :=
by sorry

end NUMINAMATH_CALUDE_exactly_one_even_l170_17000


namespace NUMINAMATH_CALUDE_sin_alpha_plus_7pi_over_6_l170_17018

theorem sin_alpha_plus_7pi_over_6 (α : ℝ) 
  (h : Real.cos (α - π/6) + Real.sin α = (4/5) * Real.sqrt 3) : 
  Real.sin (α + 7*π/6) = -4/5 := by sorry

end NUMINAMATH_CALUDE_sin_alpha_plus_7pi_over_6_l170_17018


namespace NUMINAMATH_CALUDE_emilys_typing_speed_l170_17085

/-- Emily's typing speed problem -/
theorem emilys_typing_speed : 
  ∀ (words_typed : ℕ) (hours_taken : ℕ),
  words_typed = 10800 ∧ hours_taken = 3 →
  words_typed / (hours_taken * 60) = 60 :=
by sorry

end NUMINAMATH_CALUDE_emilys_typing_speed_l170_17085


namespace NUMINAMATH_CALUDE_constant_digit_sum_characterization_l170_17045

/-- Sum of digits function -/
def S (n : ℕ) : ℕ := sorry

/-- Characterization of numbers with constant digit sum property -/
theorem constant_digit_sum_characterization (M : ℕ) :
  (M > 0 ∧ ∀ k : ℕ, 1 ≤ k ∧ k ≤ M → S (M * k) = S M) ↔
  (M = 1 ∨ ∃ n : ℕ, M = 10^n - 1) :=
sorry

end NUMINAMATH_CALUDE_constant_digit_sum_characterization_l170_17045


namespace NUMINAMATH_CALUDE_juice_bottles_theorem_l170_17089

theorem juice_bottles_theorem (bottle_capacity : ℕ) (required_amount : ℕ) (min_bottles : ℕ) : 
  bottle_capacity = 15 →
  required_amount = 195 →
  min_bottles = 13 →
  (min_bottles * bottle_capacity ≥ required_amount ∧
   ∀ n : ℕ, n * bottle_capacity ≥ required_amount → n ≥ min_bottles) :=
by sorry

end NUMINAMATH_CALUDE_juice_bottles_theorem_l170_17089


namespace NUMINAMATH_CALUDE_flooring_per_box_l170_17027

theorem flooring_per_box 
  (room_length : ℝ) 
  (room_width : ℝ) 
  (flooring_done : ℝ) 
  (boxes_needed : ℕ) 
  (h1 : room_length = 16)
  (h2 : room_width = 20)
  (h3 : flooring_done = 250)
  (h4 : boxes_needed = 7) :
  (room_length * room_width - flooring_done) / boxes_needed = 10 := by
  sorry

end NUMINAMATH_CALUDE_flooring_per_box_l170_17027


namespace NUMINAMATH_CALUDE_absolute_value_identity_l170_17095

theorem absolute_value_identity (x : ℝ) (h : x = 2023) : 
  |‖x‖ - x| - ‖x‖ - x = 0 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_identity_l170_17095


namespace NUMINAMATH_CALUDE_xy_neq_6_sufficient_not_necessary_l170_17039

theorem xy_neq_6_sufficient_not_necessary (x y : ℝ) :
  (∀ x y, x * y ≠ 6 → (x ≠ 2 ∨ y ≠ 3)) ∧
  (∃ x y, (x ≠ 2 ∨ y ≠ 3) ∧ x * y = 6) :=
sorry

end NUMINAMATH_CALUDE_xy_neq_6_sufficient_not_necessary_l170_17039


namespace NUMINAMATH_CALUDE_f_properties_l170_17033

noncomputable section

-- Define the function f
def f (p : ℝ) (x : ℝ) : ℝ := Real.log (1 - x) + Real.log (p + x)

-- State the theorem
theorem f_properties (p : ℝ) (a : ℝ) (h_p : p > -1) (h_a : 0 < a ∧ a < 1) :
  -- Part 1: Domain of f
  (∀ x, f p x ≠ 0 ↔ -p < x ∧ x < 1) ∧
  -- Part 2: Minimum value of f when p = 1
  (∃ min_val, ∀ x, -a < x ∧ x ≤ a → f 1 x ≥ min_val) ∧
  (f 1 a = Real.log (1 - a^2)) :=
sorry

end

end NUMINAMATH_CALUDE_f_properties_l170_17033


namespace NUMINAMATH_CALUDE_trigonometric_expression_equality_l170_17087

theorem trigonometric_expression_equality : 
  (Real.sin (15 * π / 180) * Real.cos (10 * π / 180) + 
   Real.cos (165 * π / 180) * Real.cos (105 * π / 180)) / 
  (Real.sin (25 * π / 180) * Real.cos (5 * π / 180) + 
   Real.cos (155 * π / 180) * Real.cos (95 * π / 180)) = 
  1 / (2 * Real.cos (25 * π / 180)) := by sorry

end NUMINAMATH_CALUDE_trigonometric_expression_equality_l170_17087


namespace NUMINAMATH_CALUDE_exp_gt_m_ln_plus_two_l170_17080

theorem exp_gt_m_ln_plus_two (x m : ℝ) (hx : x > 0) (hm : 0 < m) (hm1 : m ≤ 1) :
  Real.exp x > m * (Real.log x + 2) := by
  sorry

end NUMINAMATH_CALUDE_exp_gt_m_ln_plus_two_l170_17080


namespace NUMINAMATH_CALUDE_f_sum_equals_four_l170_17008

-- Define the function f
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^3 - b * x^5 + c * x^3 + 2

-- State the theorem
theorem f_sum_equals_four (a b c : ℝ) (h : f a b c (-5) = 3) : f a b c 5 + f a b c (-5) = 4 := by
  sorry

end NUMINAMATH_CALUDE_f_sum_equals_four_l170_17008
