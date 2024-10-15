import Mathlib

namespace NUMINAMATH_CALUDE_pencil_distribution_theorem_l1811_181161

/-- The number of ways to distribute pencils among friends -/
def distribute_pencils (total_pencils : ℕ) (num_friends : ℕ) (min_pencils : ℕ) (max_pencils : ℕ) : ℕ := 
  sorry

/-- Theorem stating the number of ways to distribute 10 pencils among 4 friends -/
theorem pencil_distribution_theorem :
  distribute_pencils 10 4 1 5 = 64 := by sorry

end NUMINAMATH_CALUDE_pencil_distribution_theorem_l1811_181161


namespace NUMINAMATH_CALUDE_complex_exp_add_l1811_181190

-- Define the complex exponential function
noncomputable def cexp (z : ℂ) : ℂ := Real.exp z.re * (Complex.cos z.im + Complex.I * Complex.sin z.im)

-- State the theorem
theorem complex_exp_add (z w : ℂ) : cexp z * cexp w = cexp (z + w) := by
  sorry

end NUMINAMATH_CALUDE_complex_exp_add_l1811_181190


namespace NUMINAMATH_CALUDE_total_boxes_in_cases_l1811_181139

/-- The number of cases Jenny needs to deliver -/
def num_cases : ℕ := 3

/-- The number of boxes in each case -/
def boxes_per_case : ℕ := 8

/-- Theorem: The total number of boxes in the cases Jenny needs to deliver is 24 -/
theorem total_boxes_in_cases : num_cases * boxes_per_case = 24 := by
  sorry

end NUMINAMATH_CALUDE_total_boxes_in_cases_l1811_181139


namespace NUMINAMATH_CALUDE_at_least_one_alarm_probability_l1811_181192

theorem at_least_one_alarm_probability (pA pB : ℝ) 
  (hpA : 0 ≤ pA ∧ pA ≤ 1) (hpB : 0 ≤ pB ∧ pB ≤ 1) :
  1 - (1 - pA) * (1 - pB) = pA + pB - pA * pB :=
by sorry

end NUMINAMATH_CALUDE_at_least_one_alarm_probability_l1811_181192


namespace NUMINAMATH_CALUDE_exactly_one_positive_integer_solution_l1811_181142

theorem exactly_one_positive_integer_solution : 
  ∃! (n : ℕ+), 24 - 6 * (n : ℝ) > 12 :=
sorry

end NUMINAMATH_CALUDE_exactly_one_positive_integer_solution_l1811_181142


namespace NUMINAMATH_CALUDE_unique_determination_from_subset_sums_l1811_181166

/-- Given a set of n integers, this function returns all possible subset sums excluding the empty subset -/
def allSubsetSums (s : Finset Int) : Finset Int :=
  sorry

theorem unique_determination_from_subset_sums
  (n : Nat)
  (s : Finset Int)
  (h1 : s.card = n)
  (h2 : 0 ∉ allSubsetSums s)
  (h3 : (allSubsetSums s).card = 2^n - 1) :
  ∀ t : Finset Int, allSubsetSums s = allSubsetSums t → s = t :=
sorry

end NUMINAMATH_CALUDE_unique_determination_from_subset_sums_l1811_181166


namespace NUMINAMATH_CALUDE_beach_conditions_l1811_181116

-- Define the weather conditions
structure WeatherConditions where
  temperature : ℝ
  sunny : Prop
  windSpeed : ℝ

-- Define when the beach is crowded
def isCrowded (w : WeatherConditions) : Prop :=
  w.temperature ≥ 85 ∧ w.sunny ∧ w.windSpeed < 15

-- Theorem statement
theorem beach_conditions (w : WeatherConditions) :
  ¬(isCrowded w) →
  (w.temperature < 85 ∨ ¬w.sunny ∨ w.windSpeed ≥ 15) :=
by
  sorry

end NUMINAMATH_CALUDE_beach_conditions_l1811_181116


namespace NUMINAMATH_CALUDE_spade_calculation_l1811_181101

def spade (x y : ℝ) : ℝ := (x + y) * (x - y)

theorem spade_calculation : spade 2 (spade 6 1) = -1221 := by
  sorry

end NUMINAMATH_CALUDE_spade_calculation_l1811_181101


namespace NUMINAMATH_CALUDE_ant_path_circle_containment_l1811_181110

/-- A closed path in a plane -/
structure ClosedPath where
  path : Set (ℝ × ℝ)
  is_closed : path.Nonempty ∧ ∃ p, p ∈ path ∧ p ∈ frontier path
  length : ℝ

/-- A circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The theorem statement -/
theorem ant_path_circle_containment (γ : ClosedPath) (h : γ.length = 1) :
  ∃ (c : Circle), c.radius = 1/4 ∧ γ.path ⊆ {p : ℝ × ℝ | dist p c.center ≤ c.radius } :=
sorry

end NUMINAMATH_CALUDE_ant_path_circle_containment_l1811_181110


namespace NUMINAMATH_CALUDE_complex_power_twelve_l1811_181149

/-- If z = 2 cos(π/8) * (sin(3π/4) + i*cos(3π/4) + i), then z^12 = -64i. -/
theorem complex_power_twelve (z : ℂ) : 
  z = 2 * Real.cos (π/8) * (Real.sin (3*π/4) + Complex.I * Real.cos (3*π/4) + Complex.I) → 
  z^12 = -64 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_power_twelve_l1811_181149


namespace NUMINAMATH_CALUDE_onions_removed_l1811_181112

/-- Proves that 5 onions were removed from the scale given the problem conditions -/
theorem onions_removed (total_onions : ℕ) (remaining_onions : ℕ) (total_weight : ℚ) 
  (avg_weight_remaining : ℚ) (avg_weight_removed : ℚ) :
  total_onions = 40 →
  remaining_onions = 35 →
  total_weight = 768/100 →
  avg_weight_remaining = 190/1000 →
  avg_weight_removed = 206/1000 →
  total_onions - remaining_onions = 5 := by
  sorry

#check onions_removed

end NUMINAMATH_CALUDE_onions_removed_l1811_181112


namespace NUMINAMATH_CALUDE_distance_between_points_l1811_181140

theorem distance_between_points :
  let x₁ : ℝ := 2
  let y₁ : ℝ := 3
  let x₂ : ℝ := 5
  let y₂ : ℝ := 9
  let distance := Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)
  distance = 3 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_distance_between_points_l1811_181140


namespace NUMINAMATH_CALUDE_combined_weight_l1811_181153

/-- Given weights of John, Mary, and Jamison, prove their combined weight -/
theorem combined_weight 
  (mary_weight : ℝ) 
  (john_weight : ℝ) 
  (jamison_weight : ℝ)
  (h1 : john_weight = mary_weight * (5/4))
  (h2 : mary_weight = jamison_weight - 20)
  (h3 : mary_weight = 160) :
  mary_weight + john_weight + jamison_weight = 540 := by
  sorry

end NUMINAMATH_CALUDE_combined_weight_l1811_181153


namespace NUMINAMATH_CALUDE_climb_eight_steps_climb_ways_eq_fib_l1811_181174

/-- Fibonacci sequence starting with 1, 1 -/
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib n + fib (n + 1)

/-- Number of ways to climb n steps -/
def climbWays : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => climbWays n + climbWays (n + 1)

theorem climb_eight_steps : climbWays 8 = 34 := by
  sorry

theorem climb_ways_eq_fib (n : ℕ) : climbWays n = fib n := by
  sorry

end NUMINAMATH_CALUDE_climb_eight_steps_climb_ways_eq_fib_l1811_181174


namespace NUMINAMATH_CALUDE_perimeter_after_cut_l1811_181175

/-- The perimeter of the figure remaining after cutting a square corner from a larger square -/
def remaining_perimeter (original_side_length cut_side_length : ℝ) : ℝ :=
  2 * original_side_length + 3 * (original_side_length - cut_side_length)

/-- Theorem stating that the perimeter of the remaining figure is 17 -/
theorem perimeter_after_cut :
  remaining_perimeter 4 1 = 17 := by
  sorry

#eval remaining_perimeter 4 1

end NUMINAMATH_CALUDE_perimeter_after_cut_l1811_181175


namespace NUMINAMATH_CALUDE_sport_formulation_water_amount_l1811_181115

/-- Prove that the sport formulation of a flavored drink contains 30 ounces of water -/
theorem sport_formulation_water_amount :
  -- Standard formulation ratio
  let standard_ratio : Fin 3 → ℚ := ![1, 12, 30]
  -- Sport formulation ratios relative to standard
  let sport_flavoring_corn_ratio := 3
  let sport_flavoring_water_ratio := 1 / 2
  -- Amount of corn syrup in sport formulation
  let sport_corn_syrup := 2

  -- The amount of water in the sport formulation
  ∃ (water : ℚ),
    -- Sport formulation flavoring to corn syrup ratio
    sport_flavoring_corn_ratio * standard_ratio 0 / standard_ratio 1 = sport_corn_syrup / 2 ∧
    -- Sport formulation flavoring to water ratio
    sport_flavoring_water_ratio * standard_ratio 0 / standard_ratio 2 = 2 / water ∧
    -- The amount of water is 30 ounces
    water = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_sport_formulation_water_amount_l1811_181115


namespace NUMINAMATH_CALUDE_elias_bananas_l1811_181113

/-- The number of bananas in a dozen -/
def dozen : ℕ := 12

/-- The number of bananas Elias ate -/
def eaten : ℕ := 1

/-- The number of bananas left after Elias ate some -/
def bananas_left (initial : ℕ) (eaten : ℕ) : ℕ := initial - eaten

/-- Theorem: If Elias bought a dozen bananas and ate 1, he has 11 left -/
theorem elias_bananas : bananas_left dozen eaten = 11 := by
  sorry

end NUMINAMATH_CALUDE_elias_bananas_l1811_181113


namespace NUMINAMATH_CALUDE_sergio_total_amount_l1811_181158

/-- Represents the total amount Mr. Sergio got from selling his fruits -/
def total_amount (mango_produce : ℕ) (price_per_kg : ℕ) : ℕ :=
  let apple_produce := 2 * mango_produce
  let orange_produce := mango_produce + 200
  (apple_produce + mango_produce + orange_produce) * price_per_kg

/-- Theorem stating that Mr. Sergio's total amount is $90,000 -/
theorem sergio_total_amount :
  total_amount 400 50 = 90000 := by
  sorry

end NUMINAMATH_CALUDE_sergio_total_amount_l1811_181158


namespace NUMINAMATH_CALUDE_octal_addition_theorem_l1811_181125

/-- Represents a number in base 8 --/
def OctalNumber := List Nat

/-- Converts a natural number to its octal representation --/
def toOctal (n : Nat) : OctalNumber := sorry

/-- Converts an octal number to its decimal representation --/
def fromOctal (o : OctalNumber) : Nat := sorry

/-- Adds two octal numbers --/
def addOctal (a b : OctalNumber) : OctalNumber := sorry

theorem octal_addition_theorem :
  let a := [5, 3, 2, 6]
  let b := [1, 4, 7, 3]
  addOctal a b = [7, 0, 4, 3] := by sorry

end NUMINAMATH_CALUDE_octal_addition_theorem_l1811_181125


namespace NUMINAMATH_CALUDE_resulting_number_divisibility_l1811_181154

theorem resulting_number_divisibility : ∃ k : ℕ, (722425 + 335) = 30 * k := by
  sorry

end NUMINAMATH_CALUDE_resulting_number_divisibility_l1811_181154


namespace NUMINAMATH_CALUDE_solve_equation_l1811_181167

theorem solve_equation (x : ℝ) :
  let y := 1 / (4 * x^2 + 2 * x + 1)
  y = 1 → x = 0 ∨ x = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1811_181167


namespace NUMINAMATH_CALUDE_probability_theorem_l1811_181189

/-- The number of tiles in box A -/
def num_tiles_A : ℕ := 20

/-- The number of tiles in box B -/
def num_tiles_B : ℕ := 30

/-- The lowest number on tiles in box A -/
def min_num_A : ℕ := 1

/-- The highest number on tiles in box A -/
def max_num_A : ℕ := 20

/-- The lowest number on tiles in box B -/
def min_num_B : ℕ := 10

/-- The highest number on tiles in box B -/
def max_num_B : ℕ := 39

/-- The probability of drawing a tile less than 10 from box A -/
def prob_A : ℚ := 9 / 20

/-- The probability of drawing a tile that is either odd or greater than 35 from box B -/
def prob_B : ℚ := 17 / 30

theorem probability_theorem :
  prob_A * prob_B = 51 / 200 := by sorry

end NUMINAMATH_CALUDE_probability_theorem_l1811_181189


namespace NUMINAMATH_CALUDE_circular_arrangement_l1811_181133

/-- 
Given a circular arrangement of n people numbered 1 to n,
if the distance from person 31 to person 7 is equal to 
the distance from person 31 to person 14, then n = 41.
-/
theorem circular_arrangement (n : ℕ) : 
  n ≥ 31 → 
  (min ((7 - 31 + n) % n) ((31 - 7) % n) = min ((14 - 31 + n) % n) ((31 - 14) % n)) → 
  n = 41 := by
  sorry

end NUMINAMATH_CALUDE_circular_arrangement_l1811_181133


namespace NUMINAMATH_CALUDE_congruence_solutions_count_l1811_181111

theorem congruence_solutions_count : 
  ∃! (s : Finset Nat), 
    (∀ x ∈ s, x > 0 ∧ x < 150 ∧ (x + 17) % 45 = 80 % 45) ∧ 
    (∀ x, x > 0 ∧ x < 150 ∧ (x + 17) % 45 = 80 % 45 → x ∈ s) ∧
    s.card = 3 := by
  sorry

end NUMINAMATH_CALUDE_congruence_solutions_count_l1811_181111


namespace NUMINAMATH_CALUDE_distance_covered_l1811_181136

/-- Proves that the total distance covered is 10 km given the specified conditions --/
theorem distance_covered (walking_speed running_speed : ℝ) (total_time : ℝ) 
  (h1 : walking_speed = 4)
  (h2 : running_speed = 8)
  (h3 : total_time = 3.75)
  (h4 : (total_distance / 2) / walking_speed + (total_distance / 2) / running_speed = total_time) :
  total_distance = 10 :=
by sorry

#check distance_covered

end NUMINAMATH_CALUDE_distance_covered_l1811_181136


namespace NUMINAMATH_CALUDE_correct_good_carrots_l1811_181165

/-- The number of good carrots given the number of carrots picked by Haley and her mother, and the number of bad carrots. -/
def goodCarrots (haleyCarrots motherCarrots badCarrots : ℕ) : ℕ :=
  haleyCarrots + motherCarrots - badCarrots

/-- Theorem stating that the number of good carrots is 64 given the specific conditions. -/
theorem correct_good_carrots :
  goodCarrots 39 38 13 = 64 := by
  sorry

end NUMINAMATH_CALUDE_correct_good_carrots_l1811_181165


namespace NUMINAMATH_CALUDE_intersection_M_N_l1811_181137

def U : Finset ℕ := {0, 1, 2, 3, 4}
def M : Finset ℕ := {0, 2, 3}
def complement_N : Finset ℕ := {1, 2, 4}

theorem intersection_M_N :
  (M ∩ (U \ complement_N) : Finset ℕ) = {0, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1811_181137


namespace NUMINAMATH_CALUDE_F_range_l1811_181104

def F (x : ℝ) : ℝ := |x + 2| - |x - 2|

theorem F_range : Set.range F = Set.Ici 4 := by sorry

end NUMINAMATH_CALUDE_F_range_l1811_181104


namespace NUMINAMATH_CALUDE_prob_at_least_two_same_correct_l1811_181100

/-- The number of sides on each die -/
def num_sides : Nat := 8

/-- The number of dice rolled -/
def num_dice : Nat := 7

/-- The probability of rolling 7 fair 8-sided dice and getting at least two dice showing the same number -/
def prob_at_least_two_same : ℚ := 319 / 320

/-- Theorem stating that the probability of at least two dice showing the same number
    when rolling 7 fair 8-sided dice is equal to 319/320 -/
theorem prob_at_least_two_same_correct :
  (1 : ℚ) - (Nat.factorial num_sides / Nat.factorial (num_sides - num_dice)) / (num_sides ^ num_dice) = prob_at_least_two_same := by
  sorry


end NUMINAMATH_CALUDE_prob_at_least_two_same_correct_l1811_181100


namespace NUMINAMATH_CALUDE_divisibility_by_35_l1811_181127

theorem divisibility_by_35 : ∃! n : ℕ, n < 10 ∧ 35 ∣ (80000 + 10000 * n + 975) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_35_l1811_181127


namespace NUMINAMATH_CALUDE_sum_x_y_is_three_sevenths_l1811_181176

theorem sum_x_y_is_three_sevenths (x y : ℚ) 
  (eq1 : 2 * x + y = 3)
  (eq2 : 3 * x - 2 * y = 12) : 
  x + y = 3 / 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_x_y_is_three_sevenths_l1811_181176


namespace NUMINAMATH_CALUDE_product_xy_l1811_181109

theorem product_xy (x y : ℝ) (h1 : x - y = 5) (h2 : x^3 - y^3 = 35) : x * y = 190 / 9 := by
  sorry

end NUMINAMATH_CALUDE_product_xy_l1811_181109


namespace NUMINAMATH_CALUDE_track_team_composition_l1811_181157

/-- The number of children on a track team after changes in composition -/
theorem track_team_composition (initial_girls initial_boys girls_joined boys_quit : ℕ) :
  initial_girls = 18 →
  initial_boys = 15 →
  girls_joined = 7 →
  boys_quit = 4 →
  (initial_girls + girls_joined) + (initial_boys - boys_quit) = 36 := by
  sorry


end NUMINAMATH_CALUDE_track_team_composition_l1811_181157


namespace NUMINAMATH_CALUDE_quadratic_value_l1811_181106

/-- A quadratic function with specific properties -/
def f (a b c : ℝ) : ℝ → ℝ := λ x => a * x^2 + b * x + c

/-- The theorem statement -/
theorem quadratic_value (a b c : ℝ) :
  (∃ (x : ℝ), f a b c x = -6 ∧ ∀ (y : ℝ), f a b c y ≥ -6) ∧  -- Minimum value is -6
  (∀ (x : ℝ), f a b c x ≥ f a b c (-2)) ∧                   -- Minimum occurs at x = -2
  f a b c 0 = 20 →                                          -- Passes through (0, 20)
  f a b c (-3) = 0.5 :=                                     -- Value at x = -3 is 0.5
by sorry

end NUMINAMATH_CALUDE_quadratic_value_l1811_181106


namespace NUMINAMATH_CALUDE_ryosuke_trip_gas_cost_l1811_181178

/-- Calculates the cost of gas for a trip given odometer readings, fuel efficiency, and gas price -/
def gas_cost_for_trip (initial_reading final_reading : ℕ) (fuel_efficiency : ℚ) (gas_price : ℚ) : ℚ :=
  let distance := final_reading - initial_reading
  let gas_used := (distance : ℚ) / fuel_efficiency
  gas_used * gas_price

/-- Theorem: The cost of gas for Ryosuke's trip is approximately $3.47 -/
theorem ryosuke_trip_gas_cost :
  let cost := gas_cost_for_trip 74568 74592 28 (405/100)
  ∃ (ε : ℚ), ε > 0 ∧ ε < (1/100) ∧ |cost - (347/100)| < ε := by
  sorry

#eval gas_cost_for_trip 74568 74592 28 (405/100)

end NUMINAMATH_CALUDE_ryosuke_trip_gas_cost_l1811_181178


namespace NUMINAMATH_CALUDE_lg_root_relationship_l1811_181187

theorem lg_root_relationship : ∃ (M1 M2 M3 : ℝ),
  M1 > 0 ∧ M2 > 0 ∧ M3 > 0 ∧
  Real.log M1 / Real.log 10 < M1 ^ (1/10) ∧
  Real.log M2 / Real.log 10 > M2 ^ (1/10) ∧
  Real.log M3 / Real.log 10 = M3 ^ (1/10) :=
by sorry

end NUMINAMATH_CALUDE_lg_root_relationship_l1811_181187


namespace NUMINAMATH_CALUDE_complex_modulus_of_z_l1811_181145

theorem complex_modulus_of_z (z : ℂ) : z = 1 - (1 / Complex.I) → Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_of_z_l1811_181145


namespace NUMINAMATH_CALUDE_tax_calculation_l1811_181151

/-- Calculate tax given income and tax rate -/
def calculate_tax (income : ℝ) (rate : ℝ) : ℝ :=
  income * rate

/-- Calculate total tax for given gross pay and tax brackets -/
def total_tax (gross_pay : ℝ) : ℝ :=
  let tax1 := calculate_tax 1500 0.10
  let tax2 := calculate_tax 2000 0.15
  let tax3 := calculate_tax (gross_pay - 1500 - 2000) 0.20
  tax1 + tax2 + tax3

/-- Apply standard deduction to total tax -/
def tax_after_deduction (total_tax : ℝ) (deduction : ℝ) : ℝ :=
  total_tax - deduction

theorem tax_calculation (gross_pay : ℝ) (deduction : ℝ) 
  (h1 : gross_pay = 4500)
  (h2 : deduction = 100) :
  tax_after_deduction (total_tax gross_pay) deduction = 550 := by
  sorry

#eval tax_after_deduction (total_tax 4500) 100

end NUMINAMATH_CALUDE_tax_calculation_l1811_181151


namespace NUMINAMATH_CALUDE_parallel_planes_normal_vectors_l1811_181168

/-- Given two planes α and β with normal vectors (1, 2, -2) and (-2, -4, k) respectively,
    if α is parallel to β, then k = 4. -/
theorem parallel_planes_normal_vectors (k : ℝ) :
  let nα : ℝ × ℝ × ℝ := (1, 2, -2)
  let nβ : ℝ × ℝ × ℝ := (-2, -4, k)
  (∃ (t : ℝ), t ≠ 0 ∧ nα = t • nβ) →
  k = 4 := by
sorry

end NUMINAMATH_CALUDE_parallel_planes_normal_vectors_l1811_181168


namespace NUMINAMATH_CALUDE_sum_product_squares_ratio_l1811_181134

theorem sum_product_squares_ratio (x y z a : ℝ) (h1 : x ≠ y ∧ y ≠ z ∧ x ≠ z) (h2 : x + y + z = a) (h3 : a ≠ 0) :
  (x * y + y * z + z * x) / (x^2 + y^2 + z^2) = 1/3 := by
sorry

end NUMINAMATH_CALUDE_sum_product_squares_ratio_l1811_181134


namespace NUMINAMATH_CALUDE_real_part_of_i_squared_times_one_plus_i_l1811_181191

theorem real_part_of_i_squared_times_one_plus_i :
  Complex.re (Complex.I^2 * (1 + Complex.I)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_i_squared_times_one_plus_i_l1811_181191


namespace NUMINAMATH_CALUDE_election_votes_l1811_181180

theorem election_votes (total_votes : ℕ) : 
  (0.7 * (0.85 * total_votes : ℝ) = 333200) → 
  total_votes = 560000 := by
sorry

end NUMINAMATH_CALUDE_election_votes_l1811_181180


namespace NUMINAMATH_CALUDE_equal_points_per_game_l1811_181193

/-- 
Given a player who scores a total of 36 points in 3 games, 
with points equally distributed among the games,
prove that the player scores 12 points in each game.
-/
theorem equal_points_per_game 
  (total_points : ℕ) 
  (num_games : ℕ) 
  (h1 : total_points = 36) 
  (h2 : num_games = 3) 
  (h3 : total_points % num_games = 0) : 
  total_points / num_games = 12 := by
sorry


end NUMINAMATH_CALUDE_equal_points_per_game_l1811_181193


namespace NUMINAMATH_CALUDE_perpendicular_preservation_l1811_181108

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (parallel_planes : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_preservation 
  (m n : Line) (α β : Plane) 
  (h1 : m ≠ n) 
  (h2 : α ≠ β) 
  (h3 : perpendicular m α) 
  (h4 : parallel_lines m n) 
  (h5 : parallel_planes α β) : 
  perpendicular n β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_preservation_l1811_181108


namespace NUMINAMATH_CALUDE_union_of_sets_l1811_181117

def A (a : ℝ) : Set ℝ := {1, 2^a}
def B (a b : ℝ) : Set ℝ := {a, b}

theorem union_of_sets (a b : ℝ) :
  A a ∩ B a b = {1/4} → A a ∪ B a b = {-2, 1, 1/4} := by
  sorry

end NUMINAMATH_CALUDE_union_of_sets_l1811_181117


namespace NUMINAMATH_CALUDE_count_even_perfect_square_factors_l1811_181105

/-- The number of even perfect square factors of 2^6 * 7^3 * 3^4 -/
def evenPerfectSquareFactors : ℕ := 18

/-- The exponent of 2 in the given number -/
def exponent2 : ℕ := 6

/-- The exponent of 7 in the given number -/
def exponent7 : ℕ := 3

/-- The exponent of 3 in the given number -/
def exponent3 : ℕ := 4

theorem count_even_perfect_square_factors :
  evenPerfectSquareFactors = (exponent2 / 2 + 1) * ((exponent7 / 2) + 1) * ((exponent3 / 2) + 1) :=
by sorry

end NUMINAMATH_CALUDE_count_even_perfect_square_factors_l1811_181105


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l1811_181172

/-- The distance between foci of an ellipse with given semi-major and semi-minor axes -/
theorem ellipse_foci_distance (a b : ℝ) (ha : a = 10) (hb : b = 4) :
  2 * Real.sqrt (a^2 - b^2) = 4 * Real.sqrt 21 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l1811_181172


namespace NUMINAMATH_CALUDE_number_problem_l1811_181146

theorem number_problem (x y a : ℝ) :
  x * y = 1 →
  (a^((x + y)^2)) / (a^((x - y)^2)) = 1296 →
  a = 6 := by sorry

end NUMINAMATH_CALUDE_number_problem_l1811_181146


namespace NUMINAMATH_CALUDE_polygon_sides_when_interior_thrice_exterior_l1811_181177

theorem polygon_sides_when_interior_thrice_exterior :
  ∀ n : ℕ,
  (n ≥ 3) →
  (180 * (n - 2) = 3 * 360) →
  n = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_when_interior_thrice_exterior_l1811_181177


namespace NUMINAMATH_CALUDE_initial_bottles_correct_l1811_181150

/-- The number of water bottles initially in Samira's box -/
def initial_bottles : ℕ := 48

/-- The number of players on the field -/
def num_players : ℕ := 11

/-- The number of bottles each player takes in the first break -/
def bottles_first_break : ℕ := 2

/-- The number of bottles each player takes at the end of the game -/
def bottles_end_game : ℕ := 1

/-- The number of bottles remaining after the game -/
def remaining_bottles : ℕ := 15

/-- Theorem stating that the initial number of bottles is correct -/
theorem initial_bottles_correct :
  initial_bottles = num_players * (bottles_first_break + bottles_end_game) + remaining_bottles :=
by sorry

end NUMINAMATH_CALUDE_initial_bottles_correct_l1811_181150


namespace NUMINAMATH_CALUDE_class_average_mark_l1811_181114

theorem class_average_mark (n1 n2 : ℕ) (avg2 avg_total : ℝ) (h1 : n1 = 30) (h2 : n2 = 50) 
    (h3 : avg2 = 90) (h4 : avg_total = 71.25) : 
  (n1 + n2 : ℝ) * avg_total = n1 * ((n1 + n2 : ℝ) * avg_total - n2 * avg2) / n1 + n2 * avg2 := by
  sorry

end NUMINAMATH_CALUDE_class_average_mark_l1811_181114


namespace NUMINAMATH_CALUDE_polynomial_has_three_real_roots_l1811_181138

def P (x : ℝ) : ℝ := x^5 + x^4 - x^3 - x^2 - 2*x - 2

theorem polynomial_has_three_real_roots :
  ∃ (a b c : ℝ), (∀ x : ℝ, P x = 0 ↔ x = a ∨ x = b ∨ x = c) ∧
  (a ≠ b ∧ b ≠ c ∧ a ≠ c) :=
sorry

end NUMINAMATH_CALUDE_polynomial_has_three_real_roots_l1811_181138


namespace NUMINAMATH_CALUDE_divisibility_equivalence_l1811_181169

theorem divisibility_equivalence (m n k : ℕ) (h : m > n) :
  (∃ a : ℤ, 4^m - 4^n = a * 3^(k+1)) ↔ (∃ b : ℤ, m - n = b * 3^k) :=
sorry

end NUMINAMATH_CALUDE_divisibility_equivalence_l1811_181169


namespace NUMINAMATH_CALUDE_inscribed_circle_rectangle_area_l1811_181126

theorem inscribed_circle_rectangle_area :
  ∀ (r : ℝ) (l w : ℝ),
  r = 7 →
  l = 3 * w →
  w = 2 * r →
  l * w = 588 :=
by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_rectangle_area_l1811_181126


namespace NUMINAMATH_CALUDE_intersection_when_a_is_4_subset_condition_l1811_181185

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 8*x + 7 < 0}
def B (a : ℝ) : Set ℝ := {x | x^2 - 2*x - a^2 - 2*a < 0}

-- Theorem 1: When a = 4, A ∩ B = (1, 6)
theorem intersection_when_a_is_4 : A ∩ (B 4) = Set.Ioo 1 6 := by sorry

-- Theorem 2: A ⊆ B if and only if a ∈ (-∞, -7] ∪ [5, +∞)
theorem subset_condition (a : ℝ) : A ⊆ B a ↔ a ≤ -7 ∨ a ≥ 5 := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_is_4_subset_condition_l1811_181185


namespace NUMINAMATH_CALUDE_xy_equals_zero_l1811_181199

theorem xy_equals_zero (x y : ℝ) (h1 : x - y = 6) (h2 : x^3 - y^3 = 54) : x * y = 0 := by
  sorry

end NUMINAMATH_CALUDE_xy_equals_zero_l1811_181199


namespace NUMINAMATH_CALUDE_largest_y_coordinate_l1811_181121

theorem largest_y_coordinate (x y : ℝ) : 
  (x^2 / 49) + ((y - 3)^2 / 25) = 0 → y ≤ 3 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_y_coordinate_l1811_181121


namespace NUMINAMATH_CALUDE_rectangle_area_change_l1811_181164

theorem rectangle_area_change (L B : ℝ) (h1 : L > 0) (h2 : B > 0) :
  let A := L * B
  let L' := L / 2
  let A' := (3 / 2) * A
  let B' := A' / L'
  B' = 3 * B :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_change_l1811_181164


namespace NUMINAMATH_CALUDE_sqrt_less_than_3y_iff_l1811_181171

theorem sqrt_less_than_3y_iff (y : ℝ) (h : y > 0) : 
  Real.sqrt y < 3 * y ↔ y > 1 / 9 := by
sorry

end NUMINAMATH_CALUDE_sqrt_less_than_3y_iff_l1811_181171


namespace NUMINAMATH_CALUDE_complex_fraction_sum_l1811_181118

theorem complex_fraction_sum (a b : ℝ) : 
  (Complex.I : ℂ) = (a + Complex.I) / (b + 2 * Complex.I) → a + b = 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_sum_l1811_181118


namespace NUMINAMATH_CALUDE_smallest_congruent_to_zero_l1811_181170

theorem smallest_congruent_to_zero : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (k : ℕ), k > 0 → k < 10 → n % k = 0) ∧
  (∀ (m : ℕ), m > 0 → (∀ (k : ℕ), k > 0 → k < 10 → m % k = 0) → m ≥ n) :=
by sorry

end NUMINAMATH_CALUDE_smallest_congruent_to_zero_l1811_181170


namespace NUMINAMATH_CALUDE_cube_sum_and_reciprocal_l1811_181182

theorem cube_sum_and_reciprocal (x : ℝ) (h : x + 1/x = -7) : x^3 + 1/x^3 = -322 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_and_reciprocal_l1811_181182


namespace NUMINAMATH_CALUDE_inequality_proof_l1811_181147

theorem inequality_proof (x y : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : x + y ≤ 1) :
  12 * x * y ≤ 4 * x * (1 - y) + 9 * y * (1 - x) := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1811_181147


namespace NUMINAMATH_CALUDE_givenPointInFirstQuadrant_l1811_181128

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the first quadrant -/
def isInFirstQuadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y > 0

/-- The given point -/
def givenPoint : Point :=
  { x := 2, y := 1 }

/-- Theorem stating that the given point is in the first quadrant -/
theorem givenPointInFirstQuadrant : isInFirstQuadrant givenPoint := by
  sorry

end NUMINAMATH_CALUDE_givenPointInFirstQuadrant_l1811_181128


namespace NUMINAMATH_CALUDE_square_area_ratio_sqrt_l1811_181130

theorem square_area_ratio_sqrt (side_C side_D : ℝ) (h1 : side_C = 45) (h2 : side_D = 60) :
  Real.sqrt ((side_C ^ 2) / (side_D ^ 2)) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_sqrt_l1811_181130


namespace NUMINAMATH_CALUDE_equation_solution_l1811_181188

theorem equation_solution :
  ∃ x : ℝ, (x + Real.sqrt (x^2 - x) = 2) ∧ (x = 4/3) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1811_181188


namespace NUMINAMATH_CALUDE_log_sum_equals_three_l1811_181119

theorem log_sum_equals_three : Real.log 50 + Real.log 20 = 3 * Real.log 10 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equals_three_l1811_181119


namespace NUMINAMATH_CALUDE_existence_of_parameters_l1811_181159

theorem existence_of_parameters : ∃ (a b c : ℝ), ∀ (x : ℝ), 
  (x + a)^2 + (2*x + b)^2 + (2*x + c)^2 = (3*x + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_parameters_l1811_181159


namespace NUMINAMATH_CALUDE_selection_theorem_l1811_181198

theorem selection_theorem (n_volunteers : ℕ) (n_bokchoys : ℕ) : 
  n_volunteers = 4 → n_bokchoys = 3 → 
  (Nat.choose (n_volunteers + n_bokchoys) 4 - Nat.choose n_volunteers 4) = 34 := by
  sorry

end NUMINAMATH_CALUDE_selection_theorem_l1811_181198


namespace NUMINAMATH_CALUDE_percentage_calculation_l1811_181107

theorem percentage_calculation (x y : ℝ) (h : x = 875.3 ∧ y = 318.65) : 
  (y / x) * 100 = 36.4 := by sorry

end NUMINAMATH_CALUDE_percentage_calculation_l1811_181107


namespace NUMINAMATH_CALUDE_hockey_pads_cost_l1811_181132

def initial_amount : ℕ := 150
def remaining_amount : ℕ := 25

def cost_of_skates : ℕ := initial_amount / 2

def cost_of_pads : ℕ := initial_amount - cost_of_skates - remaining_amount

theorem hockey_pads_cost : cost_of_pads = 50 := by
  sorry

end NUMINAMATH_CALUDE_hockey_pads_cost_l1811_181132


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1811_181123

/-- Given a geometric sequence {aₙ} satisfying a₂ + a₄ = 20 and a₃ + a₅ = 40, prove a₅ + a₇ = 160 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (h_geom : ∀ n, a (n + 1) / a n = a 2 / a 1) 
  (h_sum1 : a 2 + a 4 = 20) (h_sum2 : a 3 + a 5 = 40) : 
  a 5 + a 7 = 160 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1811_181123


namespace NUMINAMATH_CALUDE_tracing_time_5x5_l1811_181160

/-- Represents a rectangular grid with width and height -/
structure Grid where
  width : ℕ
  height : ℕ

/-- Calculates the total length of lines in a grid -/
def totalLength (g : Grid) : ℕ :=
  (g.width + 1) * g.height + (g.height + 1) * g.width

/-- Time taken to trace a grid given a reference grid and its tracing time -/
def tracingTime (refGrid : Grid) (refTime : ℕ) (targetGrid : Grid) : ℕ :=
  (totalLength targetGrid * refTime) / (totalLength refGrid)

theorem tracing_time_5x5 :
  let refGrid : Grid := { width := 7, height := 3 }
  let targetGrid : Grid := { width := 5, height := 5 }
  tracingTime refGrid 26 targetGrid = 30 := by
  sorry

end NUMINAMATH_CALUDE_tracing_time_5x5_l1811_181160


namespace NUMINAMATH_CALUDE_absolute_difference_of_mn_l1811_181179

theorem absolute_difference_of_mn (m n : ℝ) 
  (h1 : m * n = 6) 
  (h2 : m + n = 7) : 
  |m - n| = 5 := by
  sorry

end NUMINAMATH_CALUDE_absolute_difference_of_mn_l1811_181179


namespace NUMINAMATH_CALUDE_airplane_seats_theorem_l1811_181173

theorem airplane_seats_theorem :
  ∀ (total_seats : ℕ),
  (30 : ℕ) +                            -- First Class seats
  (total_seats * 20 / 100 : ℕ) +         -- Business Class seats (20% of total)
  (15 : ℕ) +                            -- Premium Economy Class seats
  (total_seats - (30 + (total_seats * 20 / 100) + 15) : ℕ) -- Economy Class seats
  = total_seats →
  total_seats = 288 := by
sorry

end NUMINAMATH_CALUDE_airplane_seats_theorem_l1811_181173


namespace NUMINAMATH_CALUDE_beam_equation_l1811_181162

/-- The equation for buying beams problem -/
theorem beam_equation (x : ℕ+) (h : x > 1) : 
  (3 : ℚ) * ((x : ℚ) - 1) = 6210 / (x : ℚ) :=
sorry

/-- The total cost of beams in wen -/
def total_cost : ℕ := 6210

/-- The transportation cost per beam in wen -/
def transport_cost : ℕ := 3

/-- The number of beams that can be bought -/
def num_beams : ℕ+ := sorry

end NUMINAMATH_CALUDE_beam_equation_l1811_181162


namespace NUMINAMATH_CALUDE_building_floors_l1811_181155

/-- Given information about three buildings A, B, and C, prove that Building C has 59 floors. -/
theorem building_floors :
  let floors_A : ℕ := 4
  let floors_B : ℕ := floors_A + 9
  let floors_C : ℕ := 5 * floors_B - 6
  floors_C = 59 := by sorry

end NUMINAMATH_CALUDE_building_floors_l1811_181155


namespace NUMINAMATH_CALUDE_both_questions_correct_l1811_181194

/-- Represents a class of students and their test results. -/
structure ClassTestResults where
  total_students : ℕ
  correct_q1 : ℕ
  correct_q2 : ℕ
  absent : ℕ

/-- Calculates the number of students who answered both questions correctly. -/
def both_correct (c : ClassTestResults) : ℕ :=
  c.correct_q1 + c.correct_q2 - (c.total_students - c.absent)

/-- Theorem stating that given the specific class conditions, 
    22 students answered both questions correctly. -/
theorem both_questions_correct 
  (c : ClassTestResults) 
  (h1 : c.total_students = 30)
  (h2 : c.correct_q1 = 25)
  (h3 : c.correct_q2 = 22)
  (h4 : c.absent = 5) :
  both_correct c = 22 := by
  sorry

#eval both_correct ⟨30, 25, 22, 5⟩

end NUMINAMATH_CALUDE_both_questions_correct_l1811_181194


namespace NUMINAMATH_CALUDE_functional_equation_solution_l1811_181152

theorem functional_equation_solution (f : ℝ → ℝ) 
  (h_not_zero : ∃ x, f x ≠ 0)
  (h_eq : ∀ x y, f x * f y = f (x - y)) :
  (∀ x, f x = 1) ∨ (∀ x, f x = -1) := by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l1811_181152


namespace NUMINAMATH_CALUDE_range_of_a_l1811_181183

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |x| ≥ 4 * a * x) → |a| ≤ (1/4 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1811_181183


namespace NUMINAMATH_CALUDE_woman_lawyer_probability_l1811_181122

/-- Represents a study group with members, women, and lawyers. -/
structure StudyGroup where
  total_members : ℕ
  women_percentage : ℝ
  lawyer_percentage : ℝ
  women_percentage_valid : 0 ≤ women_percentage ∧ women_percentage ≤ 1
  lawyer_percentage_valid : 0 ≤ lawyer_percentage ∧ lawyer_percentage ≤ 1

/-- Calculates the probability of selecting a woman lawyer from the study group. -/
def probability_woman_lawyer (group : StudyGroup) : ℝ :=
  group.women_percentage * group.lawyer_percentage

/-- Theorem stating that the probability of selecting a woman lawyer is 0.08
    given the specified conditions. -/
theorem woman_lawyer_probability (group : StudyGroup) 
  (h1 : group.women_percentage = 0.4)
  (h2 : group.lawyer_percentage = 0.2) : 
  probability_woman_lawyer group = 0.08 := by
  sorry

#check woman_lawyer_probability

end NUMINAMATH_CALUDE_woman_lawyer_probability_l1811_181122


namespace NUMINAMATH_CALUDE_sequence_general_term_l1811_181196

-- Define the sequence a_n and its partial sum S_n
def S (a : ℕ → ℤ) (n : ℕ) : ℤ := 2 * a n + 1

-- State the theorem
theorem sequence_general_term (a : ℕ → ℤ) :
  (∀ n : ℕ, S a n = 2 * a n + 1) →
  (∀ n : ℕ, a n = -2^(n-1)) :=
by sorry

end NUMINAMATH_CALUDE_sequence_general_term_l1811_181196


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l1811_181120

-- Define the sets A and B
def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x | x ≤ 4}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {x : ℝ | x ≤ 4} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l1811_181120


namespace NUMINAMATH_CALUDE_conic_sections_l1811_181143

-- Hyperbola
def hyperbola_equation (e : ℝ) (c : ℝ) : Prop :=
  e = Real.sqrt 3 ∧ c = 5 * Real.sqrt 3 →
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧
    (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ↔ x^2 / 25 - y^2 / 50 = 1)

-- Ellipse
def ellipse_equation (e : ℝ) (d : ℝ) : Prop :=
  e = 1/2 ∧ d = 4 * Real.sqrt 3 →
  ∃ a b : ℝ, a > b ∧ b > 0 ∧
    (∀ x y : ℝ, y^2 / a^2 + x^2 / b^2 = 1 ↔ y^2 / 12 + x^2 / 9 = 1)

-- Parabola
def parabola_equation (p : ℝ) : Prop :=
  p = 4 →
  ∀ x y : ℝ, x^2 = 4 * p * y ↔ x^2 = 8 * y

theorem conic_sections :
  ∀ (e_hyp e_ell c d p : ℝ),
    hyperbola_equation e_hyp c ∧
    ellipse_equation e_ell d ∧
    parabola_equation p :=
by sorry

end NUMINAMATH_CALUDE_conic_sections_l1811_181143


namespace NUMINAMATH_CALUDE_negative_5643_mod_10_l1811_181163

theorem negative_5643_mod_10 :
  ∃! (n : ℤ), 0 ≤ n ∧ n ≤ 9 ∧ n ≡ -5643 [ZMOD 10] := by
  sorry

end NUMINAMATH_CALUDE_negative_5643_mod_10_l1811_181163


namespace NUMINAMATH_CALUDE_intersection_implies_a_zero_l1811_181131

theorem intersection_implies_a_zero (a : ℝ) : 
  let A : Set ℝ := {a^2, a + 1, -1}
  let B : Set ℝ := {2*a - 1, |a - 2|, 3*a^2 + 4}
  A ∩ B = {-1} → a = 0 := by
sorry

end NUMINAMATH_CALUDE_intersection_implies_a_zero_l1811_181131


namespace NUMINAMATH_CALUDE_smallest_c_value_l1811_181103

theorem smallest_c_value (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (∀ x, a * Real.cos (b * x + c) ≤ a * Real.cos (b * (-π/4) + c)) →
  c ≥ π/4 :=
sorry

end NUMINAMATH_CALUDE_smallest_c_value_l1811_181103


namespace NUMINAMATH_CALUDE_meeting_point_l1811_181144

/-- Represents the walking speed of a person -/
structure WalkingSpeed where
  speed : ℝ
  speed_pos : speed > 0

/-- Represents a person walking around the block -/
structure Walker where
  name : String
  speed : WalkingSpeed

/-- The scenario of Jane and Hector walking around the block -/
structure WalkingScenario where
  jane : Walker
  hector : Walker
  block_size : ℝ
  jane_speed_ratio : ℝ
  start_point : ℝ
  jane_speed_twice_hector : jane.speed.speed = 2 * hector.speed.speed
  block_size_positive : block_size > 0
  jane_speed_ratio_half : jane_speed_ratio = 1/2
  start_point_zero : start_point = 0

/-- The theorem stating where Jane and Hector meet -/
theorem meeting_point (scenario : WalkingScenario) : 
  ∃ t : ℝ, t > 0 ∧ 
  (scenario.hector.speed.speed * t + scenario.jane.speed.speed * t = scenario.block_size) ∧
  (scenario.hector.speed.speed * t = 12) := by
  sorry

end NUMINAMATH_CALUDE_meeting_point_l1811_181144


namespace NUMINAMATH_CALUDE_min_distance_to_line_l1811_181195

/-- The curve C in the x-y plane -/
def C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / 4) + p.2^2 = 1}

/-- The distance function from a point on C to the line x - y - 4 = 0 -/
def distance_to_line (p : ℝ × ℝ) : ℝ :=
  |p.1 - p.2 - 4|

/-- Theorem: The minimum distance from C to the line x - y - 4 = 0 is 4 - √5 -/
theorem min_distance_to_line :
  ∃ (min_dist : ℝ), min_dist = 4 - Real.sqrt 5 ∧
  (∀ p ∈ C, distance_to_line p ≥ min_dist) ∧
  (∃ p ∈ C, distance_to_line p = min_dist) := by
  sorry

end NUMINAMATH_CALUDE_min_distance_to_line_l1811_181195


namespace NUMINAMATH_CALUDE_largest_divisor_consecutive_odd_squares_l1811_181184

theorem largest_divisor_consecutive_odd_squares (m n : ℤ) : 
  (∃ k : ℤ, n = 2*k + 1 ∧ m = 2*k + 3) →  -- m and n are consecutive odd integers
  n < m →                                 -- n is less than m
  (∀ d : ℤ, d ∣ (m^2 - n^2) → d ≤ 8) ∧    -- 8 is an upper bound for divisors
  8 ∣ (m^2 - n^2)                         -- 8 divides m^2 - n^2
  := by sorry

end NUMINAMATH_CALUDE_largest_divisor_consecutive_odd_squares_l1811_181184


namespace NUMINAMATH_CALUDE_fractional_equation_solution_range_l1811_181102

theorem fractional_equation_solution_range (m : ℝ) (x : ℝ) : 
  (m / (2 * x - 1) + 2 = 0) → (x > 0) → (m < 2 ∧ m ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_range_l1811_181102


namespace NUMINAMATH_CALUDE_inverse_47_mod_48_l1811_181124

theorem inverse_47_mod_48 : ∃! x : ℕ, x ∈ Finset.range 48 ∧ (47 * x) % 48 = 1 :=
by
  use 47
  sorry

end NUMINAMATH_CALUDE_inverse_47_mod_48_l1811_181124


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1811_181181

theorem imaginary_part_of_z (i : ℂ) (h : i^2 = -1) :
  let z : ℂ := (2 + i) / (1 + i)^2
  (z.im : ℝ) = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1811_181181


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_sum_l1811_181197

theorem partial_fraction_decomposition_sum (x A B C D E : ℝ) : 
  (1 : ℝ) / (x * (x + 1) * (x + 2) * (x + 3) * (x - 4)) = 
    A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x - 4) →
  A + B + C + D + E = 0 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_sum_l1811_181197


namespace NUMINAMATH_CALUDE_weekly_profit_calculation_l1811_181135

def planned_daily_sales : ℕ := 10

def daily_differences : List ℤ := [4, -3, -2, 7, -6, 18, -5]

def selling_price : ℕ := 65

def num_workers : ℕ := 3

def daily_expense_per_worker : ℕ := 80

def packaging_fee : ℕ := 5

def total_days : ℕ := 7

theorem weekly_profit_calculation :
  let total_sales := planned_daily_sales * total_days + daily_differences.sum
  let revenue := total_sales * (selling_price - packaging_fee)
  let expenses := num_workers * daily_expense_per_worker * total_days
  let profit := revenue - expenses
  profit = 3300 := by sorry

end NUMINAMATH_CALUDE_weekly_profit_calculation_l1811_181135


namespace NUMINAMATH_CALUDE_sum_of_fractions_l1811_181129

theorem sum_of_fractions : 
  let fractions : List ℚ := [2/8, 4/8, 6/8, 8/8, 10/8, 12/8, 14/8, 16/8, 18/8, 20/8]
  fractions.sum = 13.75 := by
sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l1811_181129


namespace NUMINAMATH_CALUDE_rhombus_area_l1811_181141

/-- The area of a rhombus with side length 10 and angle 60 degrees between sides is 50√3 -/
theorem rhombus_area (side_length : ℝ) (angle : ℝ) :
  side_length = 10 →
  angle = 60 * π / 180 →
  side_length * side_length * Real.sin angle = 50 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_rhombus_area_l1811_181141


namespace NUMINAMATH_CALUDE_sum_six_consecutive_even_integers_l1811_181156

/-- The sum of six consecutive even integers, starting from m, is equal to 6m + 30 -/
theorem sum_six_consecutive_even_integers (m : ℤ) (h : Even m) :
  m + (m + 2) + (m + 4) + (m + 6) + (m + 8) + (m + 10) = 6 * m + 30 := by
  sorry

end NUMINAMATH_CALUDE_sum_six_consecutive_even_integers_l1811_181156


namespace NUMINAMATH_CALUDE_max_sum_of_squares_l1811_181186

theorem max_sum_of_squares (a b c d : ℝ) : 
  a + b = 20 →
  a * b + c + d = 100 →
  a * d + b * c = 250 →
  c * d = 144 →
  a^2 + b^2 + c^2 + d^2 ≤ 1760 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_squares_l1811_181186


namespace NUMINAMATH_CALUDE_negation_of_existence_l1811_181148

theorem negation_of_existence (a : ℝ) :
  (¬ ∃ x : ℝ, x^2 + a*x + 1 < 0) ↔ (∀ x : ℝ, x^2 + a*x + 1 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_l1811_181148
