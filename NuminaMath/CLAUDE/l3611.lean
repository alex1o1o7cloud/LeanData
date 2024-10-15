import Mathlib

namespace NUMINAMATH_CALUDE_joes_age_l3611_361177

theorem joes_age (B J E : ℕ) : 
  B = 3 * J →                  -- Billy's age is three times Joe's age
  E = (B + J) / 2 →            -- Emily's age is the average of Billy's and Joe's ages
  B + J + E = 90 →             -- The sum of their ages is 90
  J = 15 :=                    -- Joe's age is 15
by sorry

end NUMINAMATH_CALUDE_joes_age_l3611_361177


namespace NUMINAMATH_CALUDE_regression_line_intercept_l3611_361148

-- Define the number of data points
def n : ℕ := 8

-- Define the slope of the regression line
def m : ℚ := 1/3

-- Define the sum of x values
def sum_x : ℚ := 3

-- Define the sum of y values
def sum_y : ℚ := 5

-- Define the mean of x values
def mean_x : ℚ := sum_x / n

-- Define the mean of y values
def mean_y : ℚ := sum_y / n

-- Theorem statement
theorem regression_line_intercept :
  ∃ (a : ℚ), mean_y = m * mean_x + a ∧ a = 1/2 := by sorry

end NUMINAMATH_CALUDE_regression_line_intercept_l3611_361148


namespace NUMINAMATH_CALUDE_negative_two_star_negative_three_l3611_361174

-- Define the new operation
def star (a b : ℤ) : ℤ := b^2 - a

-- State the theorem
theorem negative_two_star_negative_three : star (-2) (-3) = 11 := by
  sorry

end NUMINAMATH_CALUDE_negative_two_star_negative_three_l3611_361174


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l3611_361107

/-- Given that the solution set of ax² + bx + c > 0 is (-1/3, 2),
    prove that the solution set of cx² + bx + a < 0 is (-3, 1/2) -/
theorem quadratic_inequality_solution_sets
  (a b c : ℝ)
  (h : Set.Ioo (-1/3 : ℝ) 2 = {x | a * x^2 + b * x + c > 0}) :
  {x : ℝ | c * x^2 + b * x + a < 0} = Set.Ioo (-3 : ℝ) (1/2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l3611_361107


namespace NUMINAMATH_CALUDE_equation_solutions_l3611_361168

theorem equation_solutions : 
  let f : ℝ → ℝ := λ x => (x + 3)^2 - 4*(x - 1)^2
  (f (-1/3) = 0) ∧ (f 5 = 0) ∧ 
  (∀ x : ℝ, f x = 0 → (x = -1/3 ∨ x = 5)) := by
sorry

end NUMINAMATH_CALUDE_equation_solutions_l3611_361168


namespace NUMINAMATH_CALUDE_polynomial_remainder_l3611_361167

theorem polynomial_remainder (x : ℝ) : 
  (x^3 - 3*x + 5) % (x - 1) = 3 := by sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l3611_361167


namespace NUMINAMATH_CALUDE_average_trees_is_36_l3611_361188

/-- The number of trees planted by class A -/
def trees_A : ℕ := 35

/-- The number of trees planted by class B -/
def trees_B : ℕ := trees_A + 6

/-- The number of trees planted by class C -/
def trees_C : ℕ := trees_A - 3

/-- The average number of trees planted by the three classes -/
def average_trees : ℚ := (trees_A + trees_B + trees_C) / 3

theorem average_trees_is_36 : average_trees = 36 := by
  sorry

end NUMINAMATH_CALUDE_average_trees_is_36_l3611_361188


namespace NUMINAMATH_CALUDE_sector_central_angle_l3611_361141

/-- A circular sector with perimeter 8 and area 4 has a central angle of 2 radians -/
theorem sector_central_angle (r : ℝ) (l : ℝ) (θ : ℝ) : 
  r > 0 → 
  2 * r + l = 8 →  -- perimeter equation
  1 / 2 * l * r = 4 →  -- area equation
  θ = l / r →  -- definition of central angle in radians
  θ = 2 := by
sorry

end NUMINAMATH_CALUDE_sector_central_angle_l3611_361141


namespace NUMINAMATH_CALUDE_at_least_three_positive_and_negative_l3611_361140

theorem at_least_three_positive_and_negative 
  (a : Fin 12 → ℝ) 
  (h : ∀ i ∈ Finset.range 10, a (i + 2) * (a (i + 1) - a (i + 2) + a (i + 3)) < 0) :
  (∃ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ a i > 0 ∧ a j > 0 ∧ a k > 0) ∧
  (∃ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ a i < 0 ∧ a j < 0 ∧ a k < 0) :=
sorry

end NUMINAMATH_CALUDE_at_least_three_positive_and_negative_l3611_361140


namespace NUMINAMATH_CALUDE_x_value_proof_l3611_361130

theorem x_value_proof : ∃ x : ℝ, 
  3.5 * ((3.6 * 0.48 * x) / (0.12 * 0.09 * 0.5)) = 2800.0000000000005 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l3611_361130


namespace NUMINAMATH_CALUDE_total_cement_is_15_point_1_l3611_361178

/-- The amount of cement (in tons) used for Lexi's street -/
def lexis_street_cement : ℝ := 10

/-- The amount of cement (in tons) used for Tess's street -/
def tess_street_cement : ℝ := 5.1

/-- The total amount of cement used by Roadster's Paving Company -/
def total_cement : ℝ := lexis_street_cement + tess_street_cement

/-- Theorem stating that the total cement used is 15.1 tons -/
theorem total_cement_is_15_point_1 : total_cement = 15.1 := by sorry

end NUMINAMATH_CALUDE_total_cement_is_15_point_1_l3611_361178


namespace NUMINAMATH_CALUDE_mall_garage_third_level_spaces_l3611_361171

/-- Represents the number of parking spaces on each level of a four-story parking garage -/
structure ParkingGarage :=
  (level1 : ℕ)
  (level2 : ℕ)
  (level3 : ℕ)
  (level4 : ℕ)

/-- Calculates the total number of parking spaces in the garage -/
def total_spaces (g : ParkingGarage) : ℕ := g.level1 + g.level2 + g.level3 + g.level4

/-- Represents the parking garage described in the problem -/
def mall_garage (x : ℕ) : ParkingGarage :=
  { level1 := 90
  , level2 := 90 + 8
  , level3 := 90 + 8 + x
  , level4 := 90 + 8 + x - 9 }

/-- The theorem to be proved -/
theorem mall_garage_third_level_spaces :
  ∃ x : ℕ, 
    total_spaces (mall_garage x) = 399 ∧ 
    x = 12 := by sorry

end NUMINAMATH_CALUDE_mall_garage_third_level_spaces_l3611_361171


namespace NUMINAMATH_CALUDE_measurement_error_probability_l3611_361164

/-- The standard deviation of the measurement errors -/
def σ : ℝ := 10

/-- The maximum allowed absolute error -/
def δ : ℝ := 15

/-- The cumulative distribution function of the standard normal distribution -/
noncomputable def Φ : ℝ → ℝ := sorry

/-- The probability that the absolute error is less than δ -/
noncomputable def P (δ : ℝ) (σ : ℝ) : ℝ := 2 * Φ (δ / σ)

theorem measurement_error_probability :
  ∃ ε > 0, |P δ σ - 0.8664| < ε :=
sorry

end NUMINAMATH_CALUDE_measurement_error_probability_l3611_361164


namespace NUMINAMATH_CALUDE_maple_leaf_picking_l3611_361173

theorem maple_leaf_picking (elder_points younger_points : ℕ) 
  (h1 : elder_points = 5)
  (h2 : younger_points = 3)
  (h3 : ∃ (x y : ℕ), elder_points * x + younger_points * y = 102 ∧ x = y + 6) :
  ∃ (x y : ℕ), x = 15 ∧ y = 9 ∧ 
    elder_points * x + younger_points * y = 102 ∧ x = y + 6 := by
  sorry

end NUMINAMATH_CALUDE_maple_leaf_picking_l3611_361173


namespace NUMINAMATH_CALUDE_sum_a_b_equals_negative_one_l3611_361150

theorem sum_a_b_equals_negative_one (a b : ℝ) : 
  (|a + 3| + (b - 2)^2 = 0) → (a + b = -1) := by
  sorry

end NUMINAMATH_CALUDE_sum_a_b_equals_negative_one_l3611_361150


namespace NUMINAMATH_CALUDE_weeks_to_save_for_games_l3611_361189

/-- Calculates the minimum number of weeks required to save for a games console and a video game -/
theorem weeks_to_save_for_games (console_cost video_game_cost initial_savings weekly_allowance : ℚ)
  (tax_rate : ℚ) (h_console : console_cost = 282)
  (h_video_game : video_game_cost = 75) (h_tax : tax_rate = 0.1)
  (h_initial : initial_savings = 42) (h_allowance : weekly_allowance = 24) :
  ⌈(console_cost + video_game_cost * (1 + tax_rate) - initial_savings) / weekly_allowance⌉ = 14 := by
sorry

end NUMINAMATH_CALUDE_weeks_to_save_for_games_l3611_361189


namespace NUMINAMATH_CALUDE_perfect_square_between_prime_sums_l3611_361147

def S (n : ℕ) : ℕ := sorry

theorem perfect_square_between_prime_sums (n : ℕ) :
  ∃ k : ℕ, S n < k^2 ∧ k^2 < S (n + 1) :=
sorry

end NUMINAMATH_CALUDE_perfect_square_between_prime_sums_l3611_361147


namespace NUMINAMATH_CALUDE_haley_tree_count_l3611_361104

/-- The number of trees Haley has after a typhoon and replanting -/
def final_tree_count (initial : ℕ) (died : ℕ) (replanted : ℕ) : ℕ :=
  initial - died + replanted

/-- Theorem stating that Haley has 10 trees at the end -/
theorem haley_tree_count : final_tree_count 9 4 5 = 10 := by
  sorry

end NUMINAMATH_CALUDE_haley_tree_count_l3611_361104


namespace NUMINAMATH_CALUDE_equation_solutions_l3611_361103

theorem equation_solutions :
  (∀ x : ℝ, x^2 - 3*x = 0 ↔ x = 0 ∨ x = 3) ∧
  (∀ x : ℝ, 5*x + 2 = 3*x^2 ↔ x = -1/3 ∨ x = 2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l3611_361103


namespace NUMINAMATH_CALUDE_solve_for_y_l3611_361100

theorem solve_for_y (x y : ℝ) (h1 : x - y = 10) (h2 : x + y = 18) : y = 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l3611_361100


namespace NUMINAMATH_CALUDE_collectors_edition_dolls_l3611_361182

/-- Prove that given the conditions, Ivy and Luna have 30 collectors edition dolls combined -/
theorem collectors_edition_dolls (dina ivy luna : ℕ) : 
  dina = 60 →
  dina = 2 * ivy →
  ivy = luna + 10 →
  dina + ivy + luna = 150 →
  (2 * ivy / 3 : ℚ) + (luna / 2 : ℚ) = 30 := by
  sorry

end NUMINAMATH_CALUDE_collectors_edition_dolls_l3611_361182


namespace NUMINAMATH_CALUDE_figure_50_squares_initial_values_correct_l3611_361144

/-- Represents the number of nonoverlapping unit squares in the nth figure -/
def g (n : ℕ) : ℕ := 2 * n^2 + 5 * n + 2

/-- The theorem states that the 50th term of the sequence equals 5252 -/
theorem figure_50_squares : g 50 = 5252 := by
  sorry

/-- Verifies that the function g matches the given initial values -/
theorem initial_values_correct :
  g 0 = 2 ∧ g 1 = 9 ∧ g 2 = 20 ∧ g 3 = 35 := by
  sorry

end NUMINAMATH_CALUDE_figure_50_squares_initial_values_correct_l3611_361144


namespace NUMINAMATH_CALUDE_stratified_sample_female_count_l3611_361160

theorem stratified_sample_female_count (male_count : ℕ) (female_count : ℕ) (sample_size : ℕ) :
  male_count = 48 →
  female_count = 36 →
  sample_size = 35 →
  (female_count : ℚ) / (male_count + female_count) * sample_size = 15 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_female_count_l3611_361160


namespace NUMINAMATH_CALUDE_oil_change_cost_l3611_361126

/-- Calculates the cost of each oil change given the specified conditions. -/
theorem oil_change_cost
  (miles_per_month : ℕ)
  (miles_per_oil_change : ℕ)
  (free_oil_changes_per_year : ℕ)
  (yearly_oil_change_cost : ℕ)
  (h1 : miles_per_month = 1000)
  (h2 : miles_per_oil_change = 3000)
  (h3 : free_oil_changes_per_year = 1)
  (h4 : yearly_oil_change_cost = 150) :
  yearly_oil_change_cost / (miles_per_month * 12 / miles_per_oil_change - free_oil_changes_per_year) = 50 := by
  sorry

end NUMINAMATH_CALUDE_oil_change_cost_l3611_361126


namespace NUMINAMATH_CALUDE_johns_age_l3611_361133

theorem johns_age (john dad : ℕ) : 
  john = dad - 30 →
  john + dad = 80 →
  john = 25 := by sorry

end NUMINAMATH_CALUDE_johns_age_l3611_361133


namespace NUMINAMATH_CALUDE_passengers_landed_late_l3611_361117

theorem passengers_landed_late (on_time passengers : ℕ) (total_passengers : ℕ) 
  (h1 : on_time_passengers = 14507)
  (h2 : total_passengers = 14720) :
  total_passengers - on_time_passengers = 213 := by
  sorry

end NUMINAMATH_CALUDE_passengers_landed_late_l3611_361117


namespace NUMINAMATH_CALUDE_units_digit_of_seven_to_sixth_l3611_361172

theorem units_digit_of_seven_to_sixth (n : ℕ) : n = 7^6 → n % 10 = 9 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_seven_to_sixth_l3611_361172


namespace NUMINAMATH_CALUDE_gcd_840_1764_l3611_361196

theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := by
  sorry

end NUMINAMATH_CALUDE_gcd_840_1764_l3611_361196


namespace NUMINAMATH_CALUDE_xiaoxiang_age_problem_l3611_361132

theorem xiaoxiang_age_problem :
  let xiaoxiang_age : ℕ := 5
  let father_age : ℕ := 48
  let mother_age : ℕ := 42
  let years_passed : ℕ := 15
  (father_age + years_passed) + (mother_age + years_passed) = 6 * (xiaoxiang_age + years_passed) :=
by
  sorry

end NUMINAMATH_CALUDE_xiaoxiang_age_problem_l3611_361132


namespace NUMINAMATH_CALUDE_y_derivative_l3611_361112

noncomputable def y (x : ℝ) : ℝ := Real.sin (2 * x - 1) ^ 2

theorem y_derivative (x : ℝ) : 
  deriv y x = 2 * Real.sin (2 * (2 * x - 1)) :=
by sorry

end NUMINAMATH_CALUDE_y_derivative_l3611_361112


namespace NUMINAMATH_CALUDE_max_transition_BC_l3611_361183

def channel_A_transition : ℕ := 51
def channel_B_transition : ℕ := 63
def channel_C_transition : ℕ := 63

theorem max_transition_BC : 
  max channel_B_transition channel_C_transition = 63 := by
  sorry

end NUMINAMATH_CALUDE_max_transition_BC_l3611_361183


namespace NUMINAMATH_CALUDE_students_playing_both_sports_l3611_361186

theorem students_playing_both_sports (total : ℕ) (football : ℕ) (cricket : ℕ) (neither : ℕ) :
  total = 250 →
  football = 160 →
  cricket = 90 →
  neither = 50 →
  (total - neither) = (football + cricket - (football + cricket - (total - neither))) :=
by sorry

end NUMINAMATH_CALUDE_students_playing_both_sports_l3611_361186


namespace NUMINAMATH_CALUDE_last_digits_of_powers_last_two_digits_of_nine_powers_last_six_digits_of_seven_powers_l3611_361198

theorem last_digits_of_powers (n m : ℕ) : 
  n^(n^n) ≡ n^(n^(n^n)) [MOD 10^m] :=
sorry

theorem last_two_digits_of_nine_powers : 
  9^(9^9) ≡ 9^(9^(9^9)) [MOD 100] ∧ 
  9^(9^9) ≡ 99 [MOD 100] :=
sorry

theorem last_six_digits_of_seven_powers : 
  7^(7^(7^7)) ≡ 7^(7^(7^(7^7))) [MOD 1000000] ∧ 
  7^(7^(7^7)) ≡ 999999 [MOD 1000000] :=
sorry

end NUMINAMATH_CALUDE_last_digits_of_powers_last_two_digits_of_nine_powers_last_six_digits_of_seven_powers_l3611_361198


namespace NUMINAMATH_CALUDE_hot_air_balloon_theorem_l3611_361185

def hot_air_balloon_problem (initial_balloons : ℕ) : ℕ :=
  let after_first_30_min := initial_balloons - initial_balloons / 5
  let after_next_hour := after_first_30_min - (after_first_30_min * 3) / 10
  let durable_balloons := after_next_hour / 10
  let regular_balloons := after_next_hour - durable_balloons
  let blown_up_regular := min regular_balloons (2 * (initial_balloons - after_next_hour))
  durable_balloons

theorem hot_air_balloon_theorem :
  hot_air_balloon_problem 200 = 11 := by
  sorry

end NUMINAMATH_CALUDE_hot_air_balloon_theorem_l3611_361185


namespace NUMINAMATH_CALUDE_power_two_plus_one_div_by_three_l3611_361179

theorem power_two_plus_one_div_by_three (n : ℕ) :
  n > 0 → (3 ∣ 2^n + 1 ↔ n % 2 = 1) := by sorry

end NUMINAMATH_CALUDE_power_two_plus_one_div_by_three_l3611_361179


namespace NUMINAMATH_CALUDE_smallest_number_in_sequence_l3611_361161

theorem smallest_number_in_sequence (a b c d : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 →  -- Four positive integers
  (a + b + c + d) / 4 = 30 →  -- Arithmetic mean is 30
  b = 33 →  -- Second largest is 33
  d = b + 3 →  -- Largest is 3 more than second largest
  a < b ∧ b < c ∧ c < d →  -- Ascending order
  a = 17 :=  -- The smallest number is 17
by sorry

end NUMINAMATH_CALUDE_smallest_number_in_sequence_l3611_361161


namespace NUMINAMATH_CALUDE_luther_pancakes_correct_l3611_361154

/-- The number of people in Luther's family -/
def family_size : ℕ := 8

/-- The number of additional pancakes needed for everyone to have a second pancake -/
def additional_pancakes : ℕ := 4

/-- The number of pancakes Luther made initially -/
def initial_pancakes : ℕ := 12

/-- Theorem stating that the number of pancakes Luther made initially is correct -/
theorem luther_pancakes_correct :
  initial_pancakes = family_size * 2 - additional_pancakes :=
by sorry

end NUMINAMATH_CALUDE_luther_pancakes_correct_l3611_361154


namespace NUMINAMATH_CALUDE_fourth_region_area_l3611_361170

/-- A regular hexagon divided into four regions by three line segments -/
structure DividedHexagon where
  /-- The total area of the hexagon -/
  total_area : ℝ
  /-- The areas of the four regions -/
  region_areas : Fin 4 → ℝ
  /-- The hexagon is regular and divided into four regions -/
  is_regular_divided : total_area = region_areas 0 + region_areas 1 + region_areas 2 + region_areas 3

/-- The theorem stating the area of the fourth region -/
theorem fourth_region_area (h : DividedHexagon) 
  (h1 : h.region_areas 0 = 2)
  (h2 : h.region_areas 1 = 3)
  (h3 : h.region_areas 2 = 4) :
  h.region_areas 3 = 11 := by
  sorry

end NUMINAMATH_CALUDE_fourth_region_area_l3611_361170


namespace NUMINAMATH_CALUDE_smallest_n_for_50000_quadruplets_l3611_361105

def count_quadruplets (n : ℕ) : ℕ :=
  (Finset.filter (fun (q : ℕ × ℕ × ℕ × ℕ) => 
    Nat.gcd q.1 (Nat.gcd q.2.1 (Nat.gcd q.2.2.1 q.2.2.2)) = 50 ∧ 
    Nat.lcm q.1 (Nat.lcm q.2.1 (Nat.lcm q.2.2.1 q.2.2.2)) = n
  ) (Finset.product (Finset.range (n + 1)) (Finset.product (Finset.range (n + 1)) (Finset.product (Finset.range (n + 1)) (Finset.range (n + 1)))))).card

theorem smallest_n_for_50000_quadruplets :
  ∃ n : ℕ, n > 0 ∧ count_quadruplets n = 50000 ∧ 
  ∀ m : ℕ, m > 0 ∧ m < n → count_quadruplets m ≠ 50000 ∧
  n = 48600 := by
sorry

end NUMINAMATH_CALUDE_smallest_n_for_50000_quadruplets_l3611_361105


namespace NUMINAMATH_CALUDE_cylindrical_glass_volume_l3611_361152

/-- The volume of a cylindrical glass with specific straw conditions -/
theorem cylindrical_glass_volume : 
  ∀ (h r : ℝ),
  h > 0 → 
  r > 0 →
  h = 8 →
  r = 6 →
  h^2 + r^2 = 10^2 →
  (π : ℝ) = 3.14 →
  π * r^2 * h = 226.08 :=
by sorry

end NUMINAMATH_CALUDE_cylindrical_glass_volume_l3611_361152


namespace NUMINAMATH_CALUDE_logarithm_sum_property_l3611_361143

theorem logarithm_sum_property (a b : ℝ) (ha : a > 1) (hb : b > 1) 
  (h : Real.log (a + b) = Real.log a + Real.log b) : 
  Real.log (a - 1) + Real.log (b - 1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_sum_property_l3611_361143


namespace NUMINAMATH_CALUDE_fuel_cost_theorem_l3611_361190

theorem fuel_cost_theorem (x : ℝ) : 
  (x / 4 - x / 6 = 8) → x = 96 := by
  sorry

end NUMINAMATH_CALUDE_fuel_cost_theorem_l3611_361190


namespace NUMINAMATH_CALUDE_M_is_range_of_f_l3611_361138

-- Define the set M
def M : Set ℝ := {y | ∃ x, y = x^2}

-- Define the function f
def f : ℝ → ℝ := fun x ↦ x^2

-- Theorem statement
theorem M_is_range_of_f : M = Set.range f := by sorry

end NUMINAMATH_CALUDE_M_is_range_of_f_l3611_361138


namespace NUMINAMATH_CALUDE_school_club_profit_l3611_361197

/-- Calculates the profit for a school club selling cookies -/
def cookie_profit (num_cookies : ℕ) (buy_rate : ℚ) (sell_price : ℚ) (handling_fee : ℚ) : ℚ :=
  let cost := (num_cookies : ℚ) / buy_rate + handling_fee
  let revenue := (num_cookies : ℚ) * sell_price
  revenue - cost

/-- The profit for the school club selling cookies is $190 -/
theorem school_club_profit :
  cookie_profit 1200 3 (1/2) 10 = 190 := by
  sorry

end NUMINAMATH_CALUDE_school_club_profit_l3611_361197


namespace NUMINAMATH_CALUDE_square_plus_one_ge_double_abs_l3611_361110

theorem square_plus_one_ge_double_abs (x : ℝ) : x^2 + 1 ≥ 2 * |x| := by
  sorry

end NUMINAMATH_CALUDE_square_plus_one_ge_double_abs_l3611_361110


namespace NUMINAMATH_CALUDE_triangle_problem_l3611_361175

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2 →
  a = 2 * Real.sin B / Real.sqrt 3 →
  a = 2 →
  (1/2) * b * c * Real.sin A = Real.sqrt 3 →
  A = π/3 ∧ b = 2 ∧ c = 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l3611_361175


namespace NUMINAMATH_CALUDE_prime_squares_sum_theorem_l3611_361139

theorem prime_squares_sum_theorem (p q : ℕ) (hp : Prime p) (hq : Prime q) :
  (∃ (x y z : ℕ), p^(2*x) + q^(2*y) = z^2) ↔ ((p = 2 ∧ q = 3) ∨ (p = 3 ∧ q = 2)) := by
  sorry

end NUMINAMATH_CALUDE_prime_squares_sum_theorem_l3611_361139


namespace NUMINAMATH_CALUDE_product_of_powers_equals_1260_l3611_361156

theorem product_of_powers_equals_1260 (w x y z : ℕ) 
  (h : 2^w * 3^x * 5^y * 7^z = 1260) : 
  3*w + 4*x + 2*y + 2*z = 18 := by
  sorry

end NUMINAMATH_CALUDE_product_of_powers_equals_1260_l3611_361156


namespace NUMINAMATH_CALUDE_acid_mixture_percentage_l3611_361119

theorem acid_mixture_percentage : ∀ (a w : ℝ),
  a + w = 6 →
  a / (a + w + 2) = 15 / 100 →
  (a + 2) / (a + w + 4) = 25 / 100 →
  a / (a + w) = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_acid_mixture_percentage_l3611_361119


namespace NUMINAMATH_CALUDE_solve_for_k_l3611_361109

theorem solve_for_k : ∃ k : ℝ, 
  (∀ x y : ℝ, x = 1 ∧ y = 4 → k * x + y = 3) → 
  k = -1 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_k_l3611_361109


namespace NUMINAMATH_CALUDE_sixteen_solutions_l3611_361124

-- Define the function g
def g (x : ℝ) : ℝ := x^2 - 4*x

-- State the theorem
theorem sixteen_solutions :
  ∃! (s : Finset ℝ), (∀ c ∈ s, g (g (g (g c))) = 2) ∧ Finset.card s = 16 :=
sorry

end NUMINAMATH_CALUDE_sixteen_solutions_l3611_361124


namespace NUMINAMATH_CALUDE_movie_revenue_growth_equation_l3611_361151

theorem movie_revenue_growth_equation 
  (initial_revenue : ℝ) 
  (revenue_after_three_weeks : ℝ) 
  (x : ℝ) 
  (h1 : initial_revenue = 2.5)
  (h2 : revenue_after_three_weeks = 3.6)
  (h3 : ∀ t : ℕ, t < 3 → 
    initial_revenue * (1 + x)^t < initial_revenue * (1 + x)^(t+1)) :
  initial_revenue * (1 + x)^2 = revenue_after_three_weeks :=
sorry

end NUMINAMATH_CALUDE_movie_revenue_growth_equation_l3611_361151


namespace NUMINAMATH_CALUDE_equation_solution_l3611_361169

theorem equation_solution : 
  {x : ℝ | 3 * (x + 2) = x * (x + 2)} = {-2, 3} := by sorry

end NUMINAMATH_CALUDE_equation_solution_l3611_361169


namespace NUMINAMATH_CALUDE_ball_ratio_l3611_361108

theorem ball_ratio (blue : ℕ) (red : ℕ) (green : ℕ) (yellow : ℕ) : 
  blue = 6 → 
  red = 4 → 
  yellow = 2 * red → 
  blue + red + green + yellow = 36 → 
  green / blue = 3 := by
  sorry

end NUMINAMATH_CALUDE_ball_ratio_l3611_361108


namespace NUMINAMATH_CALUDE_girls_average_age_l3611_361106

/-- Proves that the average age of girls is 11 years given the school statistics --/
theorem girls_average_age (total_students : ℕ) (boys_avg_age : ℚ) (school_avg_age : ℚ) (num_girls : ℕ) :
  total_students = 600 →
  boys_avg_age = 12 →
  school_avg_age = 47 / 4 →  -- 11.75 years
  num_girls = 150 →
  let num_boys : ℕ := total_students - num_girls
  let total_age : ℚ := total_students * school_avg_age
  let boys_total_age : ℚ := num_boys * boys_avg_age
  let girls_total_age : ℚ := total_age - boys_total_age
  girls_total_age / num_girls = 11 := by
sorry


end NUMINAMATH_CALUDE_girls_average_age_l3611_361106


namespace NUMINAMATH_CALUDE_probability_sum_three_l3611_361176

/-- The number of sides on each die -/
def numSides : ℕ := 6

/-- The total number of possible outcomes when rolling two dice -/
def totalOutcomes : ℕ := numSides * numSides

/-- The number of ways to roll a sum of 3 with two dice -/
def favorableOutcomes : ℕ := 2

/-- The probability of rolling a sum of 3 with two fair six-sided dice -/
theorem probability_sum_three (numSides : ℕ) (totalOutcomes : ℕ) (favorableOutcomes : ℕ) :
  numSides = 6 →
  totalOutcomes = numSides * numSides →
  favorableOutcomes = 2 →
  (favorableOutcomes : ℚ) / totalOutcomes = 1 / 18 := by
  sorry

end NUMINAMATH_CALUDE_probability_sum_three_l3611_361176


namespace NUMINAMATH_CALUDE_cube_volume_l3611_361142

theorem cube_volume (cube_side : ℝ) (h1 : cube_side > 0) (h2 : cube_side ^ 2 = 36) :
  cube_side ^ 3 = 216 :=
by sorry

end NUMINAMATH_CALUDE_cube_volume_l3611_361142


namespace NUMINAMATH_CALUDE_fayes_age_l3611_361127

/-- Given the ages of four people and their relationships, prove Faye's age -/
theorem fayes_age 
  (C D E F : ℕ) -- Chad's, Diana's, Eduardo's, and Faye's ages
  (h1 : D = E - 2) -- Diana is two years younger than Eduardo
  (h2 : E = C + 5) -- Eduardo is five years older than Chad
  (h3 : F = C + 4) -- Faye is four years older than Chad
  (h4 : D = 15) -- Diana is 15 years old
  : F = 16 := by
  sorry

end NUMINAMATH_CALUDE_fayes_age_l3611_361127


namespace NUMINAMATH_CALUDE_cube_and_fifth_power_existence_l3611_361184

theorem cube_and_fifth_power_existence (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  ∃ n : ℕ, n ≥ 1 ∧ ∃ k l : ℕ, a * n = k^3 ∧ b * n = l^5 :=
sorry

end NUMINAMATH_CALUDE_cube_and_fifth_power_existence_l3611_361184


namespace NUMINAMATH_CALUDE_inverse_f_sum_squares_l3611_361180

-- Define the function f
def f (x : ℝ) : ℝ := x * |x|

-- State the theorem
theorem inverse_f_sum_squares : 
  (∃ y₁ y₂ : ℝ, f y₁ = 9 ∧ f y₂ = -49) → 
  (∃ y₁ y₂ : ℝ, f y₁ = 9 ∧ f y₂ = -49 ∧ y₁^2 + y₂^2 = 58) := by
sorry

end NUMINAMATH_CALUDE_inverse_f_sum_squares_l3611_361180


namespace NUMINAMATH_CALUDE_paper_towel_pricing_l3611_361115

theorem paper_towel_pricing (case_price : ℝ) (savings_percent : ℝ) (rolls_per_case : ℕ) :
  case_price = 9 →
  savings_percent = 25 →
  rolls_per_case = 12 →
  let individual_price := case_price * (1 + savings_percent / 100) / rolls_per_case
  individual_price = 0.9375 := by
  sorry

end NUMINAMATH_CALUDE_paper_towel_pricing_l3611_361115


namespace NUMINAMATH_CALUDE_problem_distribution_l3611_361114

theorem problem_distribution (n m : ℕ) (h1 : n = 7) (h2 : m = 5) :
  (Nat.choose n m) * (m ^ (n - m)) = 525 := by
  sorry

end NUMINAMATH_CALUDE_problem_distribution_l3611_361114


namespace NUMINAMATH_CALUDE_factorial_division_l3611_361192

theorem factorial_division (h : Nat.factorial 10 = 3628800) :
  Nat.factorial 10 / Nat.factorial 5 = 30240 := by
  sorry

end NUMINAMATH_CALUDE_factorial_division_l3611_361192


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3611_361121

theorem complex_equation_solution (z : ℂ) : (1 - Complex.I) * z = 2 + 3 * Complex.I → z = -1/2 + 5/2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3611_361121


namespace NUMINAMATH_CALUDE_property_value_calculation_l3611_361135

/-- Calculate the total value of a property with different types of buildings --/
theorem property_value_calculation (condo_price condo_area barn_price barn_area 
  detached_price detached_area garage_price garage_area : ℕ) : 
  condo_price = 98 → 
  condo_area = 2400 → 
  barn_price = 84 → 
  barn_area = 1200 → 
  detached_price = 102 → 
  detached_area = 3500 → 
  garage_price = 60 → 
  garage_area = 480 → 
  (condo_price * condo_area + barn_price * barn_area + 
   detached_price * detached_area + garage_price * garage_area) = 721800 := by
  sorry

end NUMINAMATH_CALUDE_property_value_calculation_l3611_361135


namespace NUMINAMATH_CALUDE_quiz_score_average_l3611_361187

theorem quiz_score_average (n : ℕ) (initial_avg : ℚ) (dropped_score : ℚ) : 
  n = 16 → 
  initial_avg = 62.5 → 
  dropped_score = 55 → 
  let total_score := n * initial_avg
  let remaining_total := total_score - dropped_score
  let new_avg := remaining_total / (n - 1)
  new_avg = 63 := by sorry

end NUMINAMATH_CALUDE_quiz_score_average_l3611_361187


namespace NUMINAMATH_CALUDE_opposite_of_negative_fraction_l3611_361111

theorem opposite_of_negative_fraction : 
  (-(-(1 : ℚ) / 2023)) = 1 / 2023 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_fraction_l3611_361111


namespace NUMINAMATH_CALUDE_pipe_problem_l3611_361199

theorem pipe_problem (fill_rate_A fill_rate_B empty_rate_C : ℝ) 
  (h_A : fill_rate_A = 1 / 20)
  (h_B : fill_rate_B = 1 / 30)
  (h_C : empty_rate_C > 0)
  (h_fill : 2 * fill_rate_A + 2 * fill_rate_B - 2 * empty_rate_C = 1) :
  empty_rate_C = 1 / 3 :=
by sorry

end NUMINAMATH_CALUDE_pipe_problem_l3611_361199


namespace NUMINAMATH_CALUDE_part1_part2_l3611_361120

/-- Definition of a golden equation -/
def is_golden_equation (a b c : ℝ) : Prop := a ≠ 0 ∧ a - b + c = 0

/-- Part 1: Prove that 2x^2 + 5x + 3 = 0 is a golden equation -/
theorem part1 : is_golden_equation 2 5 3 := by sorry

/-- Part 2: Prove that if 3x^2 - ax + b = 0 is a golden equation and a is a root, then a = -1 or a = 3/2 -/
theorem part2 (a b : ℝ) (h1 : is_golden_equation 3 (-a) b) (h2 : 3 * a^2 - a * a + b = 0) :
  a = -1 ∨ a = 3/2 := by sorry

end NUMINAMATH_CALUDE_part1_part2_l3611_361120


namespace NUMINAMATH_CALUDE_systematic_sampling_first_number_l3611_361125

theorem systematic_sampling_first_number 
  (population_size : ℕ) 
  (sample_size : ℕ) 
  (last_sample : ℕ) 
  (h1 : population_size = 2000)
  (h2 : sample_size = 100)
  (h3 : last_sample = 1994)
  (h4 : last_sample < population_size) :
  let interval := population_size / sample_size
  let first_sample := last_sample - (sample_size - 1) * interval
  first_sample = 14 := by
sorry

end NUMINAMATH_CALUDE_systematic_sampling_first_number_l3611_361125


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l3611_361129

def U : Set Int := Set.univ

def A : Set Int := {-1, 0, 1, 2}

def B : Set Int := {x | x^2 ≠ x}

theorem intersection_A_complement_B : A ∩ (U \ B) = {-1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l3611_361129


namespace NUMINAMATH_CALUDE_carla_marbles_l3611_361131

/-- The number of marbles Carla has after buying more -/
def total_marbles (initial : ℕ) (bought : ℕ) : ℕ :=
  initial + bought

/-- Theorem stating the total number of marbles Carla has -/
theorem carla_marbles :
  total_marbles 2289 489 = 2778 := by
  sorry

end NUMINAMATH_CALUDE_carla_marbles_l3611_361131


namespace NUMINAMATH_CALUDE_append_digits_divisible_by_36_l3611_361102

/-- A function that checks if a number is divisible by 36 -/
def isDivisibleBy36 (n : ℕ) : Prop := n % 36 = 0

/-- A function that appends two digits to 2020 -/
def appendTwoDigits (a b : ℕ) : ℕ := 202000 + 10 * a + b

theorem append_digits_divisible_by_36 :
  ∀ a b : ℕ, a < 10 → b < 10 →
    (isDivisibleBy36 (appendTwoDigits a b) ↔ (a = 3 ∧ b = 2) ∨ (a = 6 ∧ b = 8)) := by
  sorry

#check append_digits_divisible_by_36

end NUMINAMATH_CALUDE_append_digits_divisible_by_36_l3611_361102


namespace NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l3611_361153

/-- An arithmetic sequence with a non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  h1 : d ≠ 0
  h2 : ∀ n : ℕ, a (n + 1) = a n + d

/-- Condition for terms forming a geometric sequence -/
def isGeometric (s : ArithmeticSequence) : Prop :=
  (s.a 3)^2 = (s.a 1) * (s.a 9)

theorem arithmetic_geometric_ratio
  (s : ArithmeticSequence)
  (h : isGeometric s) :
  (s.a 1 + s.a 3 + s.a 9) / (s.a 2 + s.a 4 + s.a 10) = 13 / 16 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l3611_361153


namespace NUMINAMATH_CALUDE_unique_values_theorem_l3611_361159

/-- Definition of the sequence P_n -/
def P (a b : ℝ) : ℕ → ℝ × ℝ
  | 0 => (1, 0)
  | n + 1 => let (x, y) := P a b n; (a * x - b * y, b * x + a * y)

/-- Condition (i): P_0 = P_6 -/
def condition_i (a b : ℝ) : Prop := P a b 0 = P a b 6

/-- Condition (ii): All P_0, P_1, P_2, P_3, P_4, P_5 are distinct -/
def condition_ii (a b : ℝ) : Prop :=
  ∀ i j, 0 ≤ i ∧ i < j ∧ j < 6 → P a b i ≠ P a b j

/-- The main theorem -/
theorem unique_values_theorem :
  {(a, b) : ℝ × ℝ | condition_i a b ∧ condition_ii a b} =
  {(1/2, Real.sqrt 3/2), (1/2, -Real.sqrt 3/2)} :=
sorry

end NUMINAMATH_CALUDE_unique_values_theorem_l3611_361159


namespace NUMINAMATH_CALUDE_product_of_sums_equals_difference_of_powers_l3611_361155

theorem product_of_sums_equals_difference_of_powers : 
  (3 + 4) * (3^2 + 4^2) * (3^4 + 4^4) * (3^8 + 4^8) * 
  (3^16 + 4^16) * (3^32 + 4^32) * (3^64 + 4^64) = 3^128 - 4^128 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sums_equals_difference_of_powers_l3611_361155


namespace NUMINAMATH_CALUDE_magnitude_of_z_l3611_361162

theorem magnitude_of_z (z : ℂ) (h : (z + 1) * (1 + Complex.I) = 1 - Complex.I) :
  Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_z_l3611_361162


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l3611_361165

-- Define the set S with exactly two subsets
def S (a b : ℝ) := {x : ℝ | x^2 + a*x + b = 0}

-- Theorem statement
theorem quadratic_equation_properties
  (a b : ℝ)
  (h_a_pos : a > 0)
  (h_two_subsets : ∃ (x y : ℝ), x ≠ y ∧ S a b = {x, y}) :
  (a^2 - b^2 ≤ 4) ∧
  (a^2 + 1/b ≥ 4) ∧
  (∀ c x₁ x₂ : ℝ, (∀ x : ℝ, x^2 + a*x + b < c ↔ x₁ < x ∧ x < x₂) →
    |x₁ - x₂| = 4 → c = 4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l3611_361165


namespace NUMINAMATH_CALUDE_f_max_min_difference_l3611_361137

noncomputable def f (x : ℝ) : ℝ := |Real.sin x| + max (Real.sin (2 * x)) 0 + |Real.cos x|

theorem f_max_min_difference :
  (⨆ x, f x) - (⨅ x, f x) = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_f_max_min_difference_l3611_361137


namespace NUMINAMATH_CALUDE_horse_food_bags_l3611_361149

/-- Calculates the number of food bags needed for horses over a period of time. -/
theorem horse_food_bags 
  (num_horses : ℕ) 
  (feedings_per_day : ℕ) 
  (food_per_feeding : ℕ) 
  (days : ℕ) 
  (bag_weight_in_pounds : ℕ) : 
  num_horses = 25 → 
  feedings_per_day = 2 → 
  food_per_feeding = 20 → 
  days = 60 → 
  bag_weight_in_pounds = 1000 → 
  (num_horses * feedings_per_day * food_per_feeding * days) / bag_weight_in_pounds = 60 := by
  sorry

#check horse_food_bags

end NUMINAMATH_CALUDE_horse_food_bags_l3611_361149


namespace NUMINAMATH_CALUDE_certain_number_problem_l3611_361193

theorem certain_number_problem (x y : ℝ) 
  (h1 : 0.25 * x = 0.15 * y - 20) 
  (h2 : x = 820) : 
  y = 1500 := by
sorry

end NUMINAMATH_CALUDE_certain_number_problem_l3611_361193


namespace NUMINAMATH_CALUDE_complement_union_theorem_l3611_361145

def U : Finset Nat := {1,2,3,4,5,6,7}
def A : Finset Nat := {2,4,5,7}
def B : Finset Nat := {3,4,5}

theorem complement_union_theorem :
  (U \ A) ∪ (U \ B) = {1,2,3,6,7} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l3611_361145


namespace NUMINAMATH_CALUDE_bike_lock_rotation_l3611_361194

/-- Rotates a single digit by 180 degrees on a 10-digit wheel. -/
def rotate_digit (d : Nat) : Nat :=
  (d + 5) % 10

/-- The original code of the bike lock. -/
def original_code : List Nat := [6, 3, 4, 8]

/-- The correct code after rotation. -/
def correct_code : List Nat := [1, 8, 9, 3]

/-- Theorem stating that rotating each digit of the original code results in the correct code. -/
theorem bike_lock_rotation :
  original_code.map rotate_digit = correct_code := by
  sorry

#eval original_code.map rotate_digit

end NUMINAMATH_CALUDE_bike_lock_rotation_l3611_361194


namespace NUMINAMATH_CALUDE_feet_in_garden_l3611_361118

/-- The number of feet in the garden --/
def total_feet (num_dogs num_ducks num_cats num_birds num_insects : ℕ) : ℕ :=
  num_dogs * 4 + num_ducks * 2 + num_cats * 4 + num_birds * 2 + num_insects * 6

/-- Theorem stating that the total number of feet in the garden is 118 --/
theorem feet_in_garden : total_feet 6 2 4 7 10 = 118 := by
  sorry

end NUMINAMATH_CALUDE_feet_in_garden_l3611_361118


namespace NUMINAMATH_CALUDE_arithmetic_sequence_12th_term_l3611_361134

/-- An arithmetic sequence is a sequence where the difference between
    consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_12th_term
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_sum : a 7 + a 9 = 16)
  (h_4th : a 4 = 1) :
  a 12 = 15 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_12th_term_l3611_361134


namespace NUMINAMATH_CALUDE_complex_modulus_power_four_l3611_361146

theorem complex_modulus_power_four : Complex.abs ((2 + Complex.I) ^ 4) = 25 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_power_four_l3611_361146


namespace NUMINAMATH_CALUDE_fraction_simplification_l3611_361113

theorem fraction_simplification (x : ℝ) (h : x = 5) :
  (x^6 - 2*x^3 + 1) / (x^3 - 1) = 124 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3611_361113


namespace NUMINAMATH_CALUDE_sum_of_numbers_with_lcm_and_ratio_l3611_361158

/-- Given three positive integers a, b, and c in the ratio 2:3:5 with LCM 120, their sum is 40 -/
theorem sum_of_numbers_with_lcm_and_ratio (a b c : ℕ+) : 
  (a : ℕ) + b + c = 40 ∧ 
  Nat.lcm a (Nat.lcm b c) = 120 ∧ 
  3 * a = 2 * b ∧ 
  5 * a = 2 * c := by
sorry


end NUMINAMATH_CALUDE_sum_of_numbers_with_lcm_and_ratio_l3611_361158


namespace NUMINAMATH_CALUDE_vector_perpendicular_l3611_361195

/-- Given vectors a and b in ℝ², prove that a - b is perpendicular to b -/
theorem vector_perpendicular (a b : ℝ × ℝ) (h1 : a = (1, 0)) (h2 : b = (1/2, 1/2)) : 
  (a - b) • b = 0 := by
  sorry

end NUMINAMATH_CALUDE_vector_perpendicular_l3611_361195


namespace NUMINAMATH_CALUDE_distribute_four_students_three_companies_l3611_361116

/-- The number of ways to distribute students among companies -/
def distribute_students (num_students : ℕ) (num_companies : ℕ) : ℕ :=
  3^4 - 3 * 2^4 + 3

/-- Theorem stating the correct number of ways to distribute 4 students among 3 companies -/
theorem distribute_four_students_three_companies :
  distribute_students 4 3 = 36 := by
  sorry

#eval distribute_students 4 3

end NUMINAMATH_CALUDE_distribute_four_students_three_companies_l3611_361116


namespace NUMINAMATH_CALUDE_circle_equation_condition_l3611_361123

/-- The equation x^2 + y^2 - x + y + m = 0 represents a circle if and only if m < 1/2 -/
theorem circle_equation_condition (x y m : ℝ) : 
  (∃ (h k r : ℝ), r > 0 ∧ (x - h)^2 + (y - k)^2 = r^2 ↔ x^2 + y^2 - x + y + m = 0) ↔ 
  m < (1/2 : ℝ) := by sorry

end NUMINAMATH_CALUDE_circle_equation_condition_l3611_361123


namespace NUMINAMATH_CALUDE_hover_solution_l3611_361191

def hover_problem (central_day1 : ℝ) : Prop :=
  let mountain_day1 : ℝ := 3
  let eastern_day1 : ℝ := 2
  let extra_day2 : ℝ := 2
  let total_time : ℝ := 24
  let mountain_day2 : ℝ := mountain_day1 + extra_day2
  let central_day2 : ℝ := central_day1 + extra_day2
  let eastern_day2 : ℝ := eastern_day1 + extra_day2
  mountain_day1 + central_day1 + eastern_day1 + mountain_day2 + central_day2 + eastern_day2 = total_time

theorem hover_solution : hover_problem 4 := by
  sorry

end NUMINAMATH_CALUDE_hover_solution_l3611_361191


namespace NUMINAMATH_CALUDE_contest_probabilities_l3611_361136

/-- Represents the total number of questions -/
def total_questions : ℕ := 8

/-- Represents the number of listening questions -/
def listening_questions : ℕ := 3

/-- Represents the number of written response questions -/
def written_questions : ℕ := 5

/-- Calculates the probability of the first student drawing a listening question
    and the second student drawing a written response question -/
def prob_listening_written : ℚ :=
  (listening_questions * written_questions : ℚ) / (total_questions * (total_questions - 1))

/-- Calculates the probability of at least one student drawing a listening question -/
def prob_at_least_one_listening : ℚ :=
  1 - (written_questions * (written_questions - 1) : ℚ) / (total_questions * (total_questions - 1))

theorem contest_probabilities :
  prob_listening_written = 15 / 56 ∧ prob_at_least_one_listening = 9 / 14 := by
  sorry

end NUMINAMATH_CALUDE_contest_probabilities_l3611_361136


namespace NUMINAMATH_CALUDE_cricket_bat_price_l3611_361157

/-- The final price of a cricket bat after two sales with given profits -/
def final_price (initial_cost : ℝ) (profit1 : ℝ) (profit2 : ℝ) : ℝ :=
  initial_cost * (1 + profit1) * (1 + profit2)

/-- Theorem stating the final price of the cricket bat -/
theorem cricket_bat_price :
  final_price 148 0.20 0.25 = 222 := by
  sorry

end NUMINAMATH_CALUDE_cricket_bat_price_l3611_361157


namespace NUMINAMATH_CALUDE_largest_root_divisibility_l3611_361163

theorem largest_root_divisibility (a : ℝ) : 
  (a^3 - 3*a^2 + 1 = 0) →
  (∀ x : ℝ, x^3 - 3*x^2 + 1 = 0 → x ≤ a) →
  (17 ∣ ⌊a^1788⌋) ∧ (17 ∣ ⌊a^1988⌋) := by
sorry

end NUMINAMATH_CALUDE_largest_root_divisibility_l3611_361163


namespace NUMINAMATH_CALUDE_quadratic_root_problem_l3611_361101

theorem quadratic_root_problem (p d c : ℝ) : 
  c = 1 / 216 →
  (∀ x, p * x^2 + d * x = 1 ↔ x = -2 ∨ x = 216 * c) →
  d = -1/2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_problem_l3611_361101


namespace NUMINAMATH_CALUDE_converse_correct_l3611_361122

-- Define the original proposition
def original_proposition (x : ℝ) : Prop := x > 1 → x > 2

-- Define the converse proposition
def converse_proposition (x : ℝ) : Prop := x > 2 → x > 1

-- Theorem stating that the converse_proposition is indeed the converse of the original_proposition
theorem converse_correct :
  (∀ x : ℝ, original_proposition x) ↔ (∀ x : ℝ, converse_proposition x) :=
sorry

end NUMINAMATH_CALUDE_converse_correct_l3611_361122


namespace NUMINAMATH_CALUDE_division_remainder_theorem_l3611_361166

theorem division_remainder_theorem (a b : ℕ) :
  (∃ (q r : ℕ), a^2 + b^2 = (a + b) * q + r ∧ q^2 + r = 1977) →
  ((a = 37 ∧ b = 50) ∨ (a = 50 ∧ b = 37) ∨ (a = 7 ∧ b = 50) ∨ (a = 50 ∧ b = 7)) :=
by sorry

end NUMINAMATH_CALUDE_division_remainder_theorem_l3611_361166


namespace NUMINAMATH_CALUDE_journey_fraction_is_one_fourth_l3611_361181

/-- Represents the journey from Petya's home to school -/
structure Journey where
  totalTime : ℕ
  timeBeforeBell : ℕ
  timeLateIfReturn : ℕ

/-- Calculates the fraction of the journey completed when Petya remembered the pen -/
def fractionCompleted (j : Journey) : ℚ :=
  let detourTime := j.timeBeforeBell + j.timeLateIfReturn
  let timeToRememberedPoint := detourTime / 2
  timeToRememberedPoint / j.totalTime

/-- Theorem stating that the fraction of the journey completed when Petya remembered the pen is 1/4 -/
theorem journey_fraction_is_one_fourth (j : Journey) 
  (h1 : j.totalTime = 20)
  (h2 : j.timeBeforeBell = 3)
  (h3 : j.timeLateIfReturn = 7) : 
  fractionCompleted j = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_journey_fraction_is_one_fourth_l3611_361181


namespace NUMINAMATH_CALUDE_unique_integral_solution_l3611_361128

/-- Given a system of equations, prove that there is only one integral solution -/
theorem unique_integral_solution :
  ∃! (x y z : ℤ),
    (z : ℝ) ^ (x : ℝ) = (y : ℝ) ^ (2 * x : ℝ) ∧
    (2 : ℝ) ^ (z : ℝ) = 2 * (8 : ℝ) ^ (x : ℝ) ∧
    x + y + z = 18 ∧
    x = 8 ∧ y = 5 ∧ z = 25 := by
  sorry

end NUMINAMATH_CALUDE_unique_integral_solution_l3611_361128
