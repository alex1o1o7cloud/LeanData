import Mathlib

namespace NUMINAMATH_CALUDE_correct_stratified_sample_l811_81134

/-- Represents the number of people in each age group -/
structure Population :=
  (elderly : ℕ)
  (middleAged : ℕ)
  (young : ℕ)

/-- Calculates the total population -/
def totalPopulation (p : Population) : ℕ :=
  p.elderly + p.middleAged + p.young

/-- Calculates the number of people to be sampled from each stratum -/
def stratifiedSample (p : Population) (sampleSize : ℕ) : Population :=
  { elderly := (p.elderly * sampleSize) / totalPopulation p,
    middleAged := (p.middleAged * sampleSize) / totalPopulation p,
    young := (p.young * sampleSize) / totalPopulation p }

/-- The theorem to be proved -/
theorem correct_stratified_sample :
  let p : Population := { elderly := 27, middleAged := 54, young := 81 }
  let sample := stratifiedSample p 36
  sample.elderly = 6 ∧ sample.middleAged = 12 ∧ sample.young = 18 := by
  sorry


end NUMINAMATH_CALUDE_correct_stratified_sample_l811_81134


namespace NUMINAMATH_CALUDE_x_less_than_y_l811_81105

theorem x_less_than_y :
  let x : ℝ := Real.sqrt 7 - Real.sqrt 3
  let y : ℝ := Real.sqrt 6 - Real.sqrt 2
  x < y := by sorry

end NUMINAMATH_CALUDE_x_less_than_y_l811_81105


namespace NUMINAMATH_CALUDE_complex_number_modulus_l811_81101

theorem complex_number_modulus (z : ℂ) : z = -5 + 12 * Complex.I → Complex.abs z = 13 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_modulus_l811_81101


namespace NUMINAMATH_CALUDE_special_property_implies_interval_l811_81174

/-- A positive integer n < 1000 has the property that 1/n is a repeating decimal
    of period 3 and 1/(n+6) is a repeating decimal of period 2 -/
def has_special_property (n : ℕ) : Prop :=
  n > 0 ∧ n < 1000 ∧
  ∃ (a b c : ℕ), (1 : ℚ) / n = (a * 100 + b * 10 + c : ℚ) / 999 ∧
  ∃ (x y : ℕ), (1 : ℚ) / (n + 6) = (x * 10 + y : ℚ) / 99

theorem special_property_implies_interval :
  ∀ n : ℕ, has_special_property n → n ∈ Set.Icc 1 250 :=
by
  sorry

end NUMINAMATH_CALUDE_special_property_implies_interval_l811_81174


namespace NUMINAMATH_CALUDE_train_length_calculation_l811_81183

/-- Proves that given a train and a platform of equal length, if the train crosses the platform
    in one minute at a speed of 144 km/hr, then the length of the train is 1200 meters. -/
theorem train_length_calculation (train_length platform_length : ℝ) 
    (h1 : train_length = platform_length)
    (h2 : train_length + platform_length = 144 * 1000 / 60) : 
    train_length = 1200 := by
  sorry

end NUMINAMATH_CALUDE_train_length_calculation_l811_81183


namespace NUMINAMATH_CALUDE_frank_floor_l811_81190

/-- Given information about the floors where Dennis, Charlie, and Frank live,
    prove that Frank lives on the 16th floor. -/
theorem frank_floor (dennis_floor charlie_floor frank_floor : ℕ) 
  (h1 : dennis_floor = charlie_floor + 2)
  (h2 : charlie_floor = frank_floor / 4)
  (h3 : dennis_floor = 6) :
  frank_floor = 16 := by
  sorry

end NUMINAMATH_CALUDE_frank_floor_l811_81190


namespace NUMINAMATH_CALUDE_fraction_sum_l811_81125

theorem fraction_sum : (3 : ℚ) / 5 + (2 : ℚ) / 15 = (11 : ℚ) / 15 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_l811_81125


namespace NUMINAMATH_CALUDE_ordering_of_a_b_c_l811_81155

theorem ordering_of_a_b_c :
  let a : ℝ := (2 : ℝ) ^ (4/3)
  let b : ℝ := (4 : ℝ) ^ (2/5)
  let c : ℝ := (5 : ℝ) ^ (2/3)
  c > a ∧ a > b := by sorry

end NUMINAMATH_CALUDE_ordering_of_a_b_c_l811_81155


namespace NUMINAMATH_CALUDE_sequence_properties_l811_81121

-- Define a sequence as a function from ℕ to ℝ
def Sequence := ℕ → ℝ

-- Define a property for isolated points in a graph
def HasIsolatedPoints (s : Sequence) : Prop :=
  ∀ n : ℕ, ∃ ε > 0, ∀ m : ℕ, m ≠ n → |s m - s n| ≥ ε

-- Theorem statement
theorem sequence_properties :
  (∃ (s : Sequence), True) ∧
  (∀ (s : Sequence), HasIsolatedPoints s) :=
sorry

end NUMINAMATH_CALUDE_sequence_properties_l811_81121


namespace NUMINAMATH_CALUDE_all_less_than_one_l811_81108

theorem all_less_than_one (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a^2 < b) (hbc : b^2 < c) (hca : c^2 < a) :
  a < 1 ∧ b < 1 ∧ c < 1 := by
  sorry

end NUMINAMATH_CALUDE_all_less_than_one_l811_81108


namespace NUMINAMATH_CALUDE_mehki_age_l811_81191

/-- Given the ages of Zrinka, Jordyn, and Mehki, prove Mehki's age is 22 years. -/
theorem mehki_age (zrinka jordyn mehki : ℕ) 
  (h1 : mehki = jordyn + 10)
  (h2 : jordyn = 2 * zrinka)
  (h3 : zrinka = 6) : 
  mehki = 22 := by
sorry

end NUMINAMATH_CALUDE_mehki_age_l811_81191


namespace NUMINAMATH_CALUDE_expression_value_l811_81173

theorem expression_value (a b : ℤ) (ha : a = -4) (hb : b = 3) :
  -a - b^2 + a*b = -17 := by sorry

end NUMINAMATH_CALUDE_expression_value_l811_81173


namespace NUMINAMATH_CALUDE_fraction_equals_d_minus_one_l811_81145

theorem fraction_equals_d_minus_one (n d : ℕ) (h : d ∣ n) :
  ∃ k : ℕ, k < n ∧ (k : ℚ) / (n - k : ℚ) = d - 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equals_d_minus_one_l811_81145


namespace NUMINAMATH_CALUDE_count_numbers_with_three_ones_l811_81165

/-- Recursive function to count numbers without three consecutive 1's -/
def count_without_three_ones (n : ℕ) : ℕ :=
  if n ≤ 3 then
    match n with
    | 1 => 2
    | 2 => 4
    | 3 => 7
    | _ => 0
  else
    count_without_three_ones (n - 1) + count_without_three_ones (n - 2) + count_without_three_ones (n - 3)

/-- Theorem stating the count of 12-digit numbers with three consecutive 1's -/
theorem count_numbers_with_three_ones : 
  (2^12 : ℕ) - count_without_three_ones 12 = 3592 :=
sorry

end NUMINAMATH_CALUDE_count_numbers_with_three_ones_l811_81165


namespace NUMINAMATH_CALUDE_select_two_with_boy_l811_81161

/-- The number of ways to select 2 people from 4 boys and 2 girls, with at least one boy -/
def select_with_boy (total : ℕ) (boys : ℕ) (girls : ℕ) (to_select : ℕ) : ℕ :=
  Nat.choose total to_select - Nat.choose girls to_select

theorem select_two_with_boy :
  select_with_boy 6 4 2 2 = 14 :=
by sorry

end NUMINAMATH_CALUDE_select_two_with_boy_l811_81161


namespace NUMINAMATH_CALUDE_train_crossing_time_l811_81196

/-- Proves that a train 40 meters long, traveling at 144 km/hr, will take 1 second to cross an electric pole. -/
theorem train_crossing_time (train_length : Real) (train_speed_kmh : Real) :
  train_length = 40 ∧ train_speed_kmh = 144 →
  (train_length / (train_speed_kmh * (5/18))) = 1 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l811_81196


namespace NUMINAMATH_CALUDE_symmetric_complex_numbers_l811_81186

theorem symmetric_complex_numbers (z₁ z₂ : ℂ) :
  (z₁ = 2 - 3*I) →
  (z₁ = -z₂) →
  (z₂ = -2 + 3*I) := by
sorry

end NUMINAMATH_CALUDE_symmetric_complex_numbers_l811_81186


namespace NUMINAMATH_CALUDE_jogging_time_difference_fathers_jogging_time_saved_l811_81136

/-- Calculates the time difference in minutes between jogging at varying speeds and a constant speed -/
theorem jogging_time_difference (distance : ℝ) (constant_speed : ℝ) 
  (speeds : List ℝ) : ℝ :=
  let varying_time := (speeds.map (λ s => distance / s)).sum
  let constant_time := speeds.length * (distance / constant_speed)
  (varying_time - constant_time) * 60

/-- Proves that the time difference for the given scenario is 3 minutes -/
theorem fathers_jogging_time_saved : 
  jogging_time_difference 3 5 [6, 5, 4, 5] = 3 := by
  sorry

end NUMINAMATH_CALUDE_jogging_time_difference_fathers_jogging_time_saved_l811_81136


namespace NUMINAMATH_CALUDE_same_color_marble_probability_l811_81133

/-- The probability of drawing three marbles of the same color from a bag containing
    6 red marbles, 4 white marbles, and 8 blue marbles, without replacement. -/
theorem same_color_marble_probability :
  let red : ℕ := 6
  let white : ℕ := 4
  let blue : ℕ := 8
  let total : ℕ := red + white + blue
  let prob_same_color : ℚ := (Nat.choose red 3 + Nat.choose white 3 + Nat.choose blue 3 : ℚ) / Nat.choose total 3
  prob_same_color = 5 / 51 :=
by sorry

end NUMINAMATH_CALUDE_same_color_marble_probability_l811_81133


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l811_81104

theorem sqrt_equation_solution :
  ∃! x : ℝ, (Real.sqrt x + 2 * Real.sqrt (x^2 + 9*x) + Real.sqrt (x + 9) = 45 - 2*x) ∧ 
             (x = 729/144) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l811_81104


namespace NUMINAMATH_CALUDE_coffee_shop_sales_l811_81146

/-- Represents the number of lattes sold by the coffee shop. -/
def lattes : ℕ := sorry

/-- Represents the number of teas sold by the coffee shop. -/
def teas : ℕ := 6

/-- The relationship between lattes and teas sold. -/
axiom latte_tea_relation : lattes = 4 * teas + 8

theorem coffee_shop_sales : lattes = 32 := by sorry

end NUMINAMATH_CALUDE_coffee_shop_sales_l811_81146


namespace NUMINAMATH_CALUDE_system_solvability_l811_81189

-- Define the system of equations
def system (x y a : ℝ) : Prop :=
  (x - a)^2 = 4*(y - x + a - 1) ∧
  x ≠ 1 ∧ x > 0 ∧
  (Real.sqrt y - 1) / (Real.sqrt x - 1) = 1

-- Define the solution set for a
def solution_set (a : ℝ) : Prop :=
  a > 1 ∧ a ≠ 5

-- Theorem statement
theorem system_solvability (a : ℝ) :
  (∃ x y, system x y a) ↔ solution_set a :=
sorry

end NUMINAMATH_CALUDE_system_solvability_l811_81189


namespace NUMINAMATH_CALUDE_exists_distinct_power_sum_l811_81152

/-- Represents a sum of distinct powers of 3, 4, and 7 -/
structure DistinctPowerSum where
  powers_of_3 : List Nat
  powers_of_4 : List Nat
  powers_of_7 : List Nat
  distinct : powers_of_3.Nodup ∧ powers_of_4.Nodup ∧ powers_of_7.Nodup

/-- Calculates the sum of the powers in a DistinctPowerSum -/
def sumPowers (dps : DistinctPowerSum) : Nat :=
  (dps.powers_of_3.map (fun x => 3^x)).sum +
  (dps.powers_of_4.map (fun x => 4^x)).sum +
  (dps.powers_of_7.map (fun x => 7^x)).sum

/-- Theorem: Every positive integer can be represented as a sum of distinct powers of 3, 4, and 7 -/
theorem exists_distinct_power_sum (n : Nat) (h : n > 0) :
  ∃ (dps : DistinctPowerSum), sumPowers dps = n := by
  sorry

end NUMINAMATH_CALUDE_exists_distinct_power_sum_l811_81152


namespace NUMINAMATH_CALUDE_paint_mixing_l811_81167

/-- Represents the mixing of two paints to achieve a target yellow percentage -/
theorem paint_mixing (light_green_volume : ℝ) (light_green_yellow_percent : ℝ)
  (dark_green_yellow_percent : ℝ) (target_yellow_percent : ℝ) :
  light_green_volume = 5 →
  light_green_yellow_percent = 0.2 →
  dark_green_yellow_percent = 0.4 →
  target_yellow_percent = 0.25 →
  ∃ dark_green_volume : ℝ,
    dark_green_volume = 5 / 3 ∧
    (light_green_volume * light_green_yellow_percent + dark_green_volume * dark_green_yellow_percent) /
      (light_green_volume + dark_green_volume) = target_yellow_percent :=
by sorry

end NUMINAMATH_CALUDE_paint_mixing_l811_81167


namespace NUMINAMATH_CALUDE_final_short_bushes_count_l811_81170

/-- The number of short bushes in the park -/
def initial_short_bushes : ℕ := 37

/-- The number of tall trees in the park -/
def tall_trees : ℕ := 30

/-- The number of short bushes to be planted -/
def new_short_bushes : ℕ := 20

/-- The total number of short bushes after planting -/
def total_short_bushes : ℕ := initial_short_bushes + new_short_bushes

theorem final_short_bushes_count : total_short_bushes = 57 := by
  sorry

end NUMINAMATH_CALUDE_final_short_bushes_count_l811_81170


namespace NUMINAMATH_CALUDE_sum_of_A_and_B_is_31_l811_81149

theorem sum_of_A_and_B_is_31 (A B : ℝ) :
  (∀ x : ℝ, x ≠ 3 → A / (x - 3) + B * (x + 2) = (-4 * x^2 + 11 * x + 35) / (x - 3)) →
  A + B = 31 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_A_and_B_is_31_l811_81149


namespace NUMINAMATH_CALUDE_parabola_intersection_value_l811_81144

theorem parabola_intersection_value (a : ℝ) : 
  a^2 - a - 1 = 0 → a^2 - a + 2014 = 2015 := by
  sorry

end NUMINAMATH_CALUDE_parabola_intersection_value_l811_81144


namespace NUMINAMATH_CALUDE_trig_identity_l811_81102

theorem trig_identity (a : ℝ) (h : Real.sin (π / 6 - a) - Real.cos a = 1 / 3) :
  Real.cos (2 * a + π / 3) = 7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l811_81102


namespace NUMINAMATH_CALUDE_incorrect_observation_value_l811_81103

/-- Theorem: Given 20 observations with an original mean of 36, if one observation
    is corrected from an unknown value to 25, resulting in a new mean of 34.9,
    then the unknown (incorrect) value must have been 47. -/
theorem incorrect_observation_value
  (n : ℕ) -- number of observations
  (original_mean : ℝ) -- original mean
  (correct_value : ℝ) -- correct value of the observation
  (new_mean : ℝ) -- new mean after correction
  (h_n : n = 20)
  (h_original_mean : original_mean = 36)
  (h_correct_value : correct_value = 25)
  (h_new_mean : new_mean = 34.9)
  : ∃ (incorrect_value : ℝ),
    n * original_mean - incorrect_value + correct_value = n * new_mean ∧
    incorrect_value = 47 :=
sorry

end NUMINAMATH_CALUDE_incorrect_observation_value_l811_81103


namespace NUMINAMATH_CALUDE_range_proof_l811_81176

theorem range_proof (a b : ℝ) 
  (h1 : 1 ≤ a + b) (h2 : a + b ≤ 5) 
  (h3 : -1 ≤ a - b) (h4 : a - b ≤ 3) : 
  (0 ≤ a ∧ a ≤ 4) ∧ 
  (-1 ≤ b ∧ b ≤ 3) ∧ 
  (-2 ≤ 3*a - 2*b ∧ 3*a - 2*b ≤ 10) := by
  sorry

end NUMINAMATH_CALUDE_range_proof_l811_81176


namespace NUMINAMATH_CALUDE_parallel_lines_chord_distance_l811_81110

theorem parallel_lines_chord_distance (r : ℝ) (d : ℝ) : 
  r > 0 → d > 0 →
  36 * r^2 = 36 * 324 + (1/4) * d^2 * 36 →
  40 * r^2 = 40 * 400 + 40 * d^2 →
  d = Real.sqrt (304/3) :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_chord_distance_l811_81110


namespace NUMINAMATH_CALUDE_no_solution_for_pair_C_solutions_for_other_pairs_roots_of_original_equation_l811_81156

theorem no_solution_for_pair_C (x y : ℝ) : ¬(y = x ∧ y = x + 1) := by sorry

theorem solutions_for_other_pairs :
  (∃ x y : ℝ, y = x^2 ∧ y = 5*x - 6 ∧ (x = 2 ∨ x = 3)) ∧
  (∃ x : ℝ, x^2 - 5*x + 6 = 0 ∧ (x = 2 ∨ x = 3)) ∧
  (∃ x y : ℝ, y = x^2 - 5*x + 7 ∧ y = 1 ∧ (x = 2 ∨ x = 3)) ∧
  (∃ x y : ℝ, y = x^2 - 1 ∧ y = 5*x - 7 ∧ (x = 2 ∨ x = 3)) := by sorry

theorem roots_of_original_equation (x : ℝ) : x^2 - 5*x + 6 = 0 ↔ (x = 2 ∨ x = 3) := by sorry

end NUMINAMATH_CALUDE_no_solution_for_pair_C_solutions_for_other_pairs_roots_of_original_equation_l811_81156


namespace NUMINAMATH_CALUDE_line_through_points_l811_81130

/-- Given a line x = 6y + 5 passing through points (m, n) and (m + 2, n + p), prove p = 1/3 -/
theorem line_through_points (m n p : ℝ) : 
  (m = 6 * n + 5) ∧ (m + 2 = 6 * (n + p) + 5) → p = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_line_through_points_l811_81130


namespace NUMINAMATH_CALUDE_exists_larger_area_same_perimeter_l811_81106

/-- A shape in a 2D plane -/
structure Shape where
  area : ℝ
  perimeter : ℝ

/-- Theorem stating the existence of a shape with larger area and same perimeter -/
theorem exists_larger_area_same_perimeter (Φ Φ' : Shape) 
  (h1 : Φ'.area ≥ Φ.area) 
  (h2 : Φ'.perimeter < Φ.perimeter) : 
  ∃ Ψ : Shape, Ψ.perimeter = Φ.perimeter ∧ Ψ.area > Φ.area := by
  sorry

end NUMINAMATH_CALUDE_exists_larger_area_same_perimeter_l811_81106


namespace NUMINAMATH_CALUDE_brother_sister_age_diff_l811_81119

/-- The age difference between Mandy's brother and sister -/
def age_difference (mandy_age brother_age_factor sister_mandy_diff : ℕ) : ℕ :=
  brother_age_factor * mandy_age - (mandy_age + sister_mandy_diff)

/-- Theorem stating the age difference between Mandy's brother and sister -/
theorem brother_sister_age_diff :
  ∀ (mandy_age brother_age_factor sister_mandy_diff : ℕ),
    mandy_age = 3 →
    brother_age_factor = 4 →
    sister_mandy_diff = 4 →
    age_difference mandy_age brother_age_factor sister_mandy_diff = 5 :=
by
  sorry


end NUMINAMATH_CALUDE_brother_sister_age_diff_l811_81119


namespace NUMINAMATH_CALUDE_chocolate_distribution_problem_l811_81159

/-- The number of ways to distribute n chocolates among k people, 
    with each person receiving at least m chocolates -/
def distribute_chocolates (n k m : ℕ) : ℕ := 
  Nat.choose (n - k * m + k - 1) (k - 1)

/-- The problem statement -/
theorem chocolate_distribution_problem : 
  distribute_chocolates 30 3 3 = 253 := by sorry

end NUMINAMATH_CALUDE_chocolate_distribution_problem_l811_81159


namespace NUMINAMATH_CALUDE_nested_subtraction_simplification_l811_81151

theorem nested_subtraction_simplification (x : ℝ) : 1 - (2 - (3 - (4 - (5 - x)))) = 3 - x := by
  sorry

end NUMINAMATH_CALUDE_nested_subtraction_simplification_l811_81151


namespace NUMINAMATH_CALUDE_fourth_root_of_46656000_l811_81166

theorem fourth_root_of_46656000 : (46656000 : ℝ) ^ (1/4 : ℝ) = 216 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_of_46656000_l811_81166


namespace NUMINAMATH_CALUDE_bag_price_with_discount_l811_81132

theorem bag_price_with_discount (selling_price : ℝ) (discount_percentage : ℝ) 
  (h1 : selling_price = 120)
  (h2 : discount_percentage = 4) : 
  selling_price / (1 - discount_percentage / 100) = 125 := by
  sorry

end NUMINAMATH_CALUDE_bag_price_with_discount_l811_81132


namespace NUMINAMATH_CALUDE_remainder_104_pow_2006_mod_29_l811_81194

theorem remainder_104_pow_2006_mod_29 : 104^2006 % 29 = 28 := by
  sorry

end NUMINAMATH_CALUDE_remainder_104_pow_2006_mod_29_l811_81194


namespace NUMINAMATH_CALUDE_exists_permutation_1984_divisible_by_7_l811_81175

/-- A permutation of the digits of 1984 -/
def Permutation1984 : Type :=
  { p : Nat // p ∈ ({1498, 1849, 1948, 1984, 1894, 1489, 9148} : Set Nat) }

/-- Theorem: For any positive integer N, there exists a permutation of 1984's digits
    that when added to N, is divisible by 7 -/
theorem exists_permutation_1984_divisible_by_7 (N : Nat) :
  ∃ (p : Permutation1984), 7 ∣ (N + p.val) := by
  sorry

end NUMINAMATH_CALUDE_exists_permutation_1984_divisible_by_7_l811_81175


namespace NUMINAMATH_CALUDE_area_ratio_bound_for_special_triangles_l811_81123

/-- Given two right-angled triangles where the incircle radius of the first equals
    the circumcircle radius of the second, prove that the ratio of their areas
    is at least 3 + 2√2 -/
theorem area_ratio_bound_for_special_triangles (S S' r : ℝ) :
  (∃ (a b c a' b' c' : ℝ),
    -- First triangle is right-angled
    a^2 + b^2 = c^2 ∧
    -- Second triangle is right-angled
    a'^2 + b'^2 = c'^2 ∧
    -- Incircle radius of first triangle equals circumcircle radius of second
    r = c' / 2 ∧
    -- Area formulas
    S = r^2 * (a/r + b/r + c/r - π) / 2 ∧
    S' = a' * b' / 2) →
  S / S' ≥ 3 + 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_area_ratio_bound_for_special_triangles_l811_81123


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l811_81171

theorem sum_of_coefficients : 
  let p (x : ℝ) := (3*x^8 - 2*x^7 + 4*x^6 - x^4 + 6*x^2 - 7) - 
                   5*(x^5 - 2*x^3 + 2*x - 8) + 
                   6*(x^6 + x^4 - 3)
  p 1 = 32 := by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l811_81171


namespace NUMINAMATH_CALUDE_shopkeeper_loss_per_metre_l811_81142

/-- Calculates the loss per metre of cloth sold by a shopkeeper -/
theorem shopkeeper_loss_per_metre 
  (total_metres : ℕ) 
  (total_selling_price : ℕ) 
  (cost_price_per_metre : ℕ) 
  (h1 : total_metres = 600)
  (h2 : total_selling_price = 36000)
  (h3 : cost_price_per_metre = 70) :
  (cost_price_per_metre * total_metres - total_selling_price) / total_metres = 10 := by
  sorry

end NUMINAMATH_CALUDE_shopkeeper_loss_per_metre_l811_81142


namespace NUMINAMATH_CALUDE_only_shanxi_spirit_census_l811_81193

-- Define the survey types
inductive SurveyType
  | Census
  | Sample

-- Define the survey options
inductive SurveyOption
  | ArtilleryShells
  | TVRatings
  | FishSpecies
  | ShanxiSpiritAwareness

-- Function to determine the appropriate survey type for each option
def appropriateSurveyType (option : SurveyOption) : SurveyType :=
  match option with
  | SurveyOption.ArtilleryShells => SurveyType.Sample
  | SurveyOption.TVRatings => SurveyType.Sample
  | SurveyOption.FishSpecies => SurveyType.Sample
  | SurveyOption.ShanxiSpiritAwareness => SurveyType.Census

-- Theorem stating that only ShanxiSpiritAwareness is suitable for a census survey
theorem only_shanxi_spirit_census :
  ∀ (option : SurveyOption),
    appropriateSurveyType option = SurveyType.Census ↔ option = SurveyOption.ShanxiSpiritAwareness :=
by
  sorry


end NUMINAMATH_CALUDE_only_shanxi_spirit_census_l811_81193


namespace NUMINAMATH_CALUDE_find_x_value_l811_81112

theorem find_x_value (x : ℝ) (h1 : x > 0) (h2 : Real.sqrt ((3 * x) / 7) = x) : x = 3 / 7 := by
  sorry

end NUMINAMATH_CALUDE_find_x_value_l811_81112


namespace NUMINAMATH_CALUDE_point_on_x_axis_l811_81127

/-- If a point P(m, m-3) lies on the x-axis, then its coordinates are (3, 0) -/
theorem point_on_x_axis (m : ℝ) :
  (m : ℝ) = m ∧ (m - 3 : ℝ) = 0 → (m : ℝ) = 3 ∧ (m - 3 : ℝ) = 0 :=
by sorry

end NUMINAMATH_CALUDE_point_on_x_axis_l811_81127


namespace NUMINAMATH_CALUDE_plane_division_l811_81199

/-- Represents a line on a plane -/
structure Line

/-- Represents a point on a plane -/
structure Point

/-- λ(P) represents the number of lines passing through a point P -/
def lambda (P : Point) (lines : Finset Line) : ℕ := sorry

/-- The set of all intersection points of the given lines -/
def intersectionPoints (lines : Finset Line) : Finset Point := sorry

/-- Theorem: For n lines on a plane, the total number of regions formed is 1+n+∑(λ(P)-1),
    and the number of unbounded regions is 2n -/
theorem plane_division (n : ℕ) (lines : Finset Line) 
  (h : lines.card = n) :
  (∃ (regions unboundedRegions : ℕ),
    regions = 1 + n + (intersectionPoints lines).sum (λ P => lambda P lines - 1) ∧
    unboundedRegions = 2 * n) :=
  sorry

end NUMINAMATH_CALUDE_plane_division_l811_81199


namespace NUMINAMATH_CALUDE_sin_graph_transformation_l811_81180

theorem sin_graph_transformation :
  ∀ (x y : ℝ),
  (y = Real.sin x) →
  (∃ (x' y' : ℝ),
    x' = 2 * (x + π / 10) ∧
    y' = y ∧
    y' = Real.sin (x' / 2 - π / 10)) :=
by sorry

end NUMINAMATH_CALUDE_sin_graph_transformation_l811_81180


namespace NUMINAMATH_CALUDE_money_split_l811_81157

theorem money_split (total : ℝ) (moses_percent : ℝ) (moses_esther_diff : ℝ) : 
  total = 50 ∧ moses_percent = 0.4 ∧ moses_esther_diff = 5 →
  ∃ (tony esther moses : ℝ),
    moses = total * moses_percent ∧
    tony + esther = total - moses ∧
    moses = esther + moses_esther_diff ∧
    tony = 15 ∧ esther = 15 :=
by sorry

end NUMINAMATH_CALUDE_money_split_l811_81157


namespace NUMINAMATH_CALUDE_min_distance_to_line_l811_81139

theorem min_distance_to_line (x y : ℝ) (h : 6 * x + 8 * y - 1 = 0) :
  ∃ (min_val : ℝ), min_val = 7 / 10 ∧
  ∀ (x' y' : ℝ), 6 * x' + 8 * y' - 1 = 0 →
    Real.sqrt (x'^2 + y'^2 - 2*y' + 1) ≥ min_val :=
by sorry

end NUMINAMATH_CALUDE_min_distance_to_line_l811_81139


namespace NUMINAMATH_CALUDE_parabola_y_axis_intersection_l811_81164

/-- The parabola defined by y = 2x^2 + 3 intersects the y-axis at the point (0, 3) -/
theorem parabola_y_axis_intersection :
  let f : ℝ → ℝ := fun x ↦ 2 * x^2 + 3
  (0, f 0) = (0, 3) := by sorry

end NUMINAMATH_CALUDE_parabola_y_axis_intersection_l811_81164


namespace NUMINAMATH_CALUDE_curve_transformation_l811_81117

theorem curve_transformation (x : ℝ) : 
  Real.sin (2 * x + 2 * Real.pi / 3) = Real.cos (2 * (x - Real.pi / 12)) := by
  sorry

end NUMINAMATH_CALUDE_curve_transformation_l811_81117


namespace NUMINAMATH_CALUDE_ducks_in_marsh_l811_81169

theorem ducks_in_marsh (total_birds geese : ℕ) (h1 : total_birds = 95) (h2 : geese = 58) :
  total_birds - geese = 37 := by
  sorry

end NUMINAMATH_CALUDE_ducks_in_marsh_l811_81169


namespace NUMINAMATH_CALUDE_house_height_calculation_l811_81158

/-- The height of Lily's house in feet -/
def house_height : ℝ := 56.25

/-- The length of the shadow cast by Lily's house in feet -/
def house_shadow : ℝ := 75

/-- The height of the tree in feet -/
def tree_height : ℝ := 15

/-- The length of the shadow cast by the tree in feet -/
def tree_shadow : ℝ := 20

/-- Theorem stating that the calculated house height is correct -/
theorem house_height_calculation :
  house_height = tree_height * (house_shadow / tree_shadow) :=
by sorry

end NUMINAMATH_CALUDE_house_height_calculation_l811_81158


namespace NUMINAMATH_CALUDE_joint_investment_l811_81179

def total_investment : ℝ := 5000

theorem joint_investment (x : ℝ) :
  ∃ (a b : ℝ),
    a + b = total_investment ∧
    a * (1 + x / 100) = 2100 ∧
    b * (1 + (x + 1) / 100) = 3180 ∧
    a = 2000 ∧
    b = 3000 :=
by sorry

end NUMINAMATH_CALUDE_joint_investment_l811_81179


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l811_81192

theorem sqrt_product_equality : Real.sqrt 128 * Real.sqrt 50 * Real.sqrt 18 = 240 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l811_81192


namespace NUMINAMATH_CALUDE_shoe_price_calculation_l811_81162

theorem shoe_price_calculation (initial_price : ℝ) (increase_rate : ℝ) (discount_rate : ℝ) : 
  initial_price = 50 →
  increase_rate = 0.2 →
  discount_rate = 0.15 →
  initial_price * (1 + increase_rate) * (1 - discount_rate) = 51 :=
by sorry

end NUMINAMATH_CALUDE_shoe_price_calculation_l811_81162


namespace NUMINAMATH_CALUDE_linear_function_value_l811_81138

theorem linear_function_value (f : ℝ → ℝ) (a b : ℝ) 
  (h1 : f 1 = 5) 
  (h2 : f 2 = 8) 
  (h3 : f 3 = 11) 
  (h_linear : ∀ x, f x = a * x + b) : 
  f 4 = 14 := by
sorry

end NUMINAMATH_CALUDE_linear_function_value_l811_81138


namespace NUMINAMATH_CALUDE_point_B_coordinates_l811_81185

/-- Given that point A(m+2, m) lies on the y-axis, prove that point B(m+5, m-1) has coordinates (3, -3) -/
theorem point_B_coordinates (m : ℝ) 
  (h_A_on_y_axis : m + 2 = 0) : 
  (m + 5, m - 1) = (3, -3) := by
  sorry

end NUMINAMATH_CALUDE_point_B_coordinates_l811_81185


namespace NUMINAMATH_CALUDE_halfway_between_one_seventh_and_one_ninth_l811_81100

theorem halfway_between_one_seventh_and_one_ninth :
  (1 / 7 + 1 / 9) / 2 = 8 / 63 := by
  sorry

end NUMINAMATH_CALUDE_halfway_between_one_seventh_and_one_ninth_l811_81100


namespace NUMINAMATH_CALUDE_divisible_by_5040_l811_81135

theorem divisible_by_5040 (n : ℤ) (h : n > 3) : 
  ∃ k : ℤ, n^7 - 14*n^5 + 49*n^3 - 36*n = 5040 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_5040_l811_81135


namespace NUMINAMATH_CALUDE_absolute_value_equality_l811_81177

theorem absolute_value_equality (a b c d e f : ℝ) 
  (h1 : a * c * e ≠ 0)
  (h2 : ∀ x : ℝ, |a * x + b| + |c * x + d| = |e * x + f|) : 
  a * d = b * c := by sorry

end NUMINAMATH_CALUDE_absolute_value_equality_l811_81177


namespace NUMINAMATH_CALUDE_venus_speed_conversion_l811_81148

/-- Converts a speed from miles per second to miles per hour -/
def miles_per_second_to_miles_per_hour (speed_mps : ℝ) : ℝ :=
  speed_mps * 3600

/-- The speed of Venus around the sun in miles per second -/
def venus_speed_mps : ℝ := 21.9

theorem venus_speed_conversion :
  miles_per_second_to_miles_per_hour venus_speed_mps = 78840 := by
  sorry

end NUMINAMATH_CALUDE_venus_speed_conversion_l811_81148


namespace NUMINAMATH_CALUDE_largest_mersenne_prime_under_1000_l811_81124

def is_mersenne_prime (p : Nat) : Prop :=
  ∃ n : Nat, Prime n ∧ p = 2^n - 1 ∧ Prime p

theorem largest_mersenne_prime_under_1000 :
  (∀ q : Nat, is_mersenne_prime q ∧ q < 1000 → q ≤ 127) ∧
  is_mersenne_prime 127 ∧
  127 < 1000 :=
sorry

end NUMINAMATH_CALUDE_largest_mersenne_prime_under_1000_l811_81124


namespace NUMINAMATH_CALUDE_square_difference_l811_81182

theorem square_difference (a b : ℝ) (h1 : a + b = 3) (h2 : a - b = 5) : a^2 - b^2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l811_81182


namespace NUMINAMATH_CALUDE_range_of_f_l811_81153

open Real

theorem range_of_f (x : ℝ) (h : x ∈ Set.Icc 0 (π / 2)) :
  let f := λ x : ℝ => 3 * sin (2 * x - π / 6)
  ∃ y, y ∈ Set.Icc (-3/2) 3 ∧ ∃ x, x ∈ Set.Icc 0 (π / 2) ∧ f x = y :=
by sorry

end NUMINAMATH_CALUDE_range_of_f_l811_81153


namespace NUMINAMATH_CALUDE_square_difference_plus_double_l811_81160

theorem square_difference_plus_double (x y : ℝ) (h : x + y = 1) : x^2 - y^2 + 2*y = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_plus_double_l811_81160


namespace NUMINAMATH_CALUDE_cube_diagonal_l811_81120

theorem cube_diagonal (s : ℝ) (h : s > 0) (eq : s^3 + 36*s = 12*s^2) : 
  Real.sqrt (3 * s^2) = 6 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_cube_diagonal_l811_81120


namespace NUMINAMATH_CALUDE_total_protest_days_l811_81141

/-- Given a first protest lasting 4 days and a second protest lasting 25% longer,
    prove that the total number of days spent protesting is 9. -/
theorem total_protest_days : 
  let first_protest_days : ℕ := 4
  let second_protest_days : ℕ := first_protest_days + first_protest_days / 4
  first_protest_days + second_protest_days = 9 := by sorry

end NUMINAMATH_CALUDE_total_protest_days_l811_81141


namespace NUMINAMATH_CALUDE_one_third_of_1206_percent_of_400_l811_81178

theorem one_third_of_1206_percent_of_400 : (1206 / 3) / 400 * 100 = 100.5 := by
  sorry

end NUMINAMATH_CALUDE_one_third_of_1206_percent_of_400_l811_81178


namespace NUMINAMATH_CALUDE_two_pipes_fill_time_l811_81122

/-- Given two pipes filling a tank, where one pipe is 3 times as fast as the other,
    and the slower pipe can fill the tank in 160 minutes,
    prove that both pipes together can fill the tank in 40 minutes. -/
theorem two_pipes_fill_time (slow_pipe_time : ℝ) (fast_pipe_time : ℝ) : 
  slow_pipe_time = 160 →
  fast_pipe_time = slow_pipe_time / 3 →
  (1 / fast_pipe_time + 1 / slow_pipe_time)⁻¹ = 40 :=
by sorry

end NUMINAMATH_CALUDE_two_pipes_fill_time_l811_81122


namespace NUMINAMATH_CALUDE_snail_movement_bound_l811_81129

/-- Represents the movement of a snail over time -/
structure SnailMovement where
  /-- The total observation time in minutes -/
  total_time : ℝ
  /-- The movement function: time → distance -/
  movement : ℝ → ℝ
  /-- Ensures the movement is non-negative -/
  non_negative : ∀ t, 0 ≤ movement t
  /-- Ensures the movement is monotonically increasing -/
  monotone : ∀ t₁ t₂, t₁ ≤ t₂ → movement t₁ ≤ movement t₂

/-- The observation condition: for any 1-minute interval, the snail moves exactly 1 meter -/
def observation_condition (sm : SnailMovement) : Prop :=
  ∀ t, 0 ≤ t ∧ t + 1 ≤ sm.total_time → sm.movement (t + 1) - sm.movement t = 1

/-- The theorem statement -/
theorem snail_movement_bound (sm : SnailMovement) 
    (h_time : sm.total_time = 6)
    (h_obs : observation_condition sm) :
    sm.movement sm.total_time ≤ 10 := by
  sorry

end NUMINAMATH_CALUDE_snail_movement_bound_l811_81129


namespace NUMINAMATH_CALUDE_rectangular_solid_volume_l811_81111

/-- Given a rectangular solid with side areas 3, 5, and 15 sharing a common vertex,
    its volume is 15. -/
theorem rectangular_solid_volume (a b c : ℝ) (h1 : a * b = 3) (h2 : a * c = 5) (h3 : b * c = 15) :
  a * b * c = 15 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_volume_l811_81111


namespace NUMINAMATH_CALUDE_divisor_coloring_game_strategy_l811_81147

/-- A player in the divisor coloring game -/
inductive Player
| A
| B

/-- The result of the divisor coloring game -/
inductive GameResult
| AWins
| BWins

/-- The divisor coloring game for a positive integer n -/
def divisorColoringGame (n : ℕ+) : GameResult := sorry

/-- Check if a number is a perfect square -/
def isPerfectSquare (n : ℕ+) : Prop := ∃ m : ℕ+, n = m * m

/-- Theorem: Player A wins if and only if n is a perfect square or prime -/
theorem divisor_coloring_game_strategy (n : ℕ+) :
  divisorColoringGame n = GameResult.AWins ↔ isPerfectSquare n ∨ Nat.Prime n.val := by sorry

end NUMINAMATH_CALUDE_divisor_coloring_game_strategy_l811_81147


namespace NUMINAMATH_CALUDE_farmer_ploughing_problem_l811_81109

/-- Represents the farmer's ploughing problem -/
def FarmerProblem (initial_productivity : ℝ) (productivity_increase : ℝ) 
  (total_area : ℝ) (days_ahead : ℕ) (initial_days : ℕ) : Prop :=
  let improved_productivity := initial_productivity * (1 + productivity_increase)
  let area_first_two_days := 2 * initial_productivity
  let remaining_area := total_area - area_first_two_days
  let remaining_days := remaining_area / improved_productivity
  initial_days = ⌈remaining_days⌉ + 2 + days_ahead

/-- The theorem statement for the farmer's ploughing problem -/
theorem farmer_ploughing_problem :
  FarmerProblem 120 0.25 1440 2 12 := by
  sorry

end NUMINAMATH_CALUDE_farmer_ploughing_problem_l811_81109


namespace NUMINAMATH_CALUDE_min_value_product_l811_81114

theorem min_value_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_sum : a/b + b/c + c/a + b/a + c/b + a/c = 10) :
  (a/b + b/c + c/a) * (b/a + c/b + a/c) ≥ 47 := by sorry

end NUMINAMATH_CALUDE_min_value_product_l811_81114


namespace NUMINAMATH_CALUDE_equation_solution_l811_81187

theorem equation_solution : 
  ∃ x₁ x₂ : ℝ, x₁ = 4 ∧ x₂ = -2 ∧ 
  (∀ x : ℝ, (x + 1) * (x - 3) = 5 ↔ x = x₁ ∨ x = x₂) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l811_81187


namespace NUMINAMATH_CALUDE_shaded_percentage_is_75_percent_l811_81113

/-- Represents a square grid composed of smaller squares -/
structure Grid where
  side_length : ℕ
  small_squares : ℕ
  shaded_squares : ℕ

/-- Calculates the percentage of shaded squares in the grid -/
def shaded_percentage (g : Grid) : ℚ :=
  (g.shaded_squares : ℚ) / (g.small_squares : ℚ) * 100

/-- Theorem stating that the percentage of shaded squares is 75% -/
theorem shaded_percentage_is_75_percent (g : Grid) 
  (h1 : g.side_length = 8)
  (h2 : g.small_squares = g.side_length * g.side_length)
  (h3 : g.shaded_squares = 48) : 
  shaded_percentage g = 75 := by
  sorry

end NUMINAMATH_CALUDE_shaded_percentage_is_75_percent_l811_81113


namespace NUMINAMATH_CALUDE_marys_sheep_ratio_l811_81115

theorem marys_sheep_ratio (total_sheep : ℕ) (sister_fraction : ℚ) (remaining_sheep : ℕ) : 
  total_sheep = 400 →
  sister_fraction = 1/4 →
  remaining_sheep = 150 →
  let sheep_to_sister := total_sheep * sister_fraction
  let sheep_after_sister := total_sheep - sheep_to_sister
  let sheep_to_brother := sheep_after_sister - remaining_sheep
  (sheep_to_brother : ℚ) / sheep_after_sister = 1/2 := by
    sorry

end NUMINAMATH_CALUDE_marys_sheep_ratio_l811_81115


namespace NUMINAMATH_CALUDE_log_equality_implies_y_value_l811_81184

-- Define the logarithm relationship
def log_relation (m y : ℝ) : Prop :=
  (Real.log y / Real.log m) * (Real.log m / Real.log 7) = 4

-- Theorem statement
theorem log_equality_implies_y_value :
  ∀ m y : ℝ, m > 0 ∧ m ≠ 1 ∧ y > 0 → log_relation m y → y = 2401 :=
by
  sorry

end NUMINAMATH_CALUDE_log_equality_implies_y_value_l811_81184


namespace NUMINAMATH_CALUDE_tunnel_length_specific_tunnel_length_l811_81143

/-- The length of a tunnel given train and time information -/
theorem tunnel_length (train_length : ℝ) (time_diff : ℝ) (train_speed : ℝ) : ℝ :=
  let tunnel_length := train_speed * time_diff / 60
  by
    -- Proof goes here
    sorry

/-- The specific tunnel length for the given problem -/
theorem specific_tunnel_length : 
  tunnel_length 2 4 30 = 2 := by sorry

end NUMINAMATH_CALUDE_tunnel_length_specific_tunnel_length_l811_81143


namespace NUMINAMATH_CALUDE_four_propositions_two_correct_l811_81126

theorem four_propositions_two_correct :
  (∀ A B : Set α, A ∩ B = A → A ⊆ B) ∧
  (∀ a : ℝ, (∃ x y : ℝ, a * x + y + 1 = 0 ∧ x - y + 1 = 0 ∧ (∀ x' y' : ℝ, a * x' + y' + 1 = 0 → x' - y' + 1 = 0 → (x, y) ≠ (x', y'))) → a = 1) ∧
  ¬(∀ p q : Prop, p ∨ q → p ∧ q) ∧
  ¬(∀ a b m : ℝ, a < b → a * m^2 < b * m^2) :=
by sorry

end NUMINAMATH_CALUDE_four_propositions_two_correct_l811_81126


namespace NUMINAMATH_CALUDE_complex_magnitude_theorem_l811_81163

theorem complex_magnitude_theorem (s : ℝ) (w : ℂ) (h1 : |s| < 3) (h2 : w + 3 / w = s) : 
  Complex.abs w = (3 : ℝ) / 2 := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_theorem_l811_81163


namespace NUMINAMATH_CALUDE_complex_number_location_l811_81154

theorem complex_number_location :
  let z : ℂ := 2 / (1 - Complex.I)
  z = 1 + Complex.I ∧ z.re > 0 ∧ z.im > 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_location_l811_81154


namespace NUMINAMATH_CALUDE_property_P_implications_l811_81197

def has_property_P (f : ℕ → ℕ) : Prop :=
  ∀ x : ℕ, f x + f (x + 2) ≤ 2 * f (x + 1)

def d (f : ℕ → ℕ) (x : ℕ) : ℤ :=
  f (x + 1) - f x

theorem property_P_implications (f : ℕ → ℕ) (h : has_property_P f) :
  (∀ x : ℕ, d f x ≥ 0 ∧ d f (x + 1) ≤ d f x) ∧
  (∃ c : ℕ, c ≤ d f 1 ∧ Set.Infinite {n : ℕ | d f n = c}) :=
sorry

end NUMINAMATH_CALUDE_property_P_implications_l811_81197


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l811_81172

theorem least_subtraction_for_divisibility : ∃ (n : ℕ), n = 5 ∧ 
  (∀ (m : ℕ), m < n → ¬(31 ∣ (42739 - m))) ∧ (31 ∣ (42739 - n)) := by
  sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l811_81172


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_one_third_l811_81131

theorem reciprocal_of_negative_one_third :
  let x : ℚ := -1/3
  let y : ℚ := -3
  x * y = 1 ∧ ∀ z : ℚ, x * z = 1 → z = y :=
by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_one_third_l811_81131


namespace NUMINAMATH_CALUDE_cos_36_degrees_l811_81188

theorem cos_36_degrees (h : Real.sin (108 * π / 180) = 3 * Real.sin (36 * π / 180) - 4 * (Real.sin (36 * π / 180))^3) :
  Real.cos (36 * π / 180) = (1 + Real.sqrt 5) / 4 := by
sorry

end NUMINAMATH_CALUDE_cos_36_degrees_l811_81188


namespace NUMINAMATH_CALUDE_problem_solution_l811_81140

theorem problem_solution : ∃ (S L x : ℕ), 
  S = 18 ∧ 
  S + L = 51 ∧ 
  L = 2 * S - x ∧ 
  x > 0 ∧ 
  x = 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l811_81140


namespace NUMINAMATH_CALUDE_problem_solution_l811_81128

-- Statement ①
def statement1 (a b c : ℝ) : Prop :=
  (a > b → c^2 * a > c^2 * b)

-- Statement ②
def statement2 (m : ℝ) : Prop :=
  (m > 0 → ∃ x : ℝ, x^2 + x - m = 0)

-- Statement ③
def statement3 (x y : ℝ) : Prop :=
  (x + y = 5 → x^2 - y^2 - 3*x + 7*y = 10)

theorem problem_solution :
  (¬ ∀ a b c : ℝ, ¬statement1 a b c) ∧
  (∀ m : ℝ, ¬(∃ x : ℝ, x^2 + x - m = 0) → m ≤ 0) ∧
  ((∀ x y : ℝ, x + y = 5 → x^2 - y^2 - 3*x + 7*y = 10) ∧
   ¬(∀ x y : ℝ, x^2 - y^2 - 3*x + 7*y = 10 → x + y = 5)) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l811_81128


namespace NUMINAMATH_CALUDE_fraction_sum_simplest_form_fraction_simplest_form_l811_81118

theorem fraction_sum_simplest_form : (7 : ℚ) / 12 + (8 : ℚ) / 15 = (67 : ℚ) / 60 := by
  sorry

theorem fraction_simplest_form : (67 : ℚ) / 60 = (67 : ℚ) / 60 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_simplest_form_fraction_simplest_form_l811_81118


namespace NUMINAMATH_CALUDE_complex_fraction_power_l811_81107

theorem complex_fraction_power (i : ℂ) : i * i = -1 → (((1 + i) / (1 - i)) ^ 2014 : ℂ) = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_power_l811_81107


namespace NUMINAMATH_CALUDE_sequence_formula_and_sum_bound_l811_81116

def S (n : ℕ) : ℚ := 3/2 * n^2 - 1/2 * n

def a (n : ℕ+) : ℚ := 3 * n - 2

def T (n : ℕ+) : ℚ := 1 - 1 / (3 * n + 1)

theorem sequence_formula_and_sum_bound :
  (∀ n : ℕ+, a n = S n - S (n-1)) ∧
  (∃ m : ℕ+, (∀ n : ℕ+, T n < m / 20) ∧
             (∀ k : ℕ+, k < m → ∃ n : ℕ+, T n ≥ k / 20)) :=
sorry

end NUMINAMATH_CALUDE_sequence_formula_and_sum_bound_l811_81116


namespace NUMINAMATH_CALUDE_ear_muffs_bought_in_december_l811_81150

theorem ear_muffs_bought_in_december (before_december : ℕ) (total : ℕ) 
  (h1 : before_december = 1346)
  (h2 : total = 7790) :
  total - before_december = 6444 :=
by sorry

end NUMINAMATH_CALUDE_ear_muffs_bought_in_december_l811_81150


namespace NUMINAMATH_CALUDE_similar_triangle_lines_count_l811_81137

/-- A triangle in a 2D plane -/
structure Triangle :=
  (A B C : ℝ × ℝ)

/-- A point in a 2D plane -/
def Point := ℝ × ℝ

/-- Predicate to check if a point is inside a triangle -/
def isInside (P : Point) (T : Triangle) : Prop := sorry

/-- A line in a 2D plane -/
structure Line :=
  (point : Point)
  (direction : ℝ × ℝ)

/-- Predicate to check if a line intersects a triangle -/
def intersects (L : Line) (T : Triangle) : Prop := sorry

/-- Predicate to check if two triangles are similar -/
def areSimilar (T1 T2 : Triangle) : Prop := sorry

/-- Function to count the number of lines through a point inside a triangle
    that intersect the triangle and form similar triangles -/
def countSimilarTriangleLines (T : Triangle) (P : Point) : ℕ := sorry

/-- Theorem stating that the number of lines through a point inside a triangle
    that intersect the triangle and form similar triangles is 6 -/
theorem similar_triangle_lines_count (T : Triangle) (P : Point) 
  (h : isInside P T) : countSimilarTriangleLines T P = 6 := by sorry

end NUMINAMATH_CALUDE_similar_triangle_lines_count_l811_81137


namespace NUMINAMATH_CALUDE_only_statements_3_and_4_are_propositions_l811_81195

-- Define a type for our statements
inductive Statement
  | equation : Statement
  | question : Statement
  | arithmeticFalse : Statement
  | universalFalse : Statement

-- Define a function to check if a statement is a proposition
def isProposition (s : Statement) : Prop :=
  match s with
  | Statement.equation => False
  | Statement.question => False
  | Statement.arithmeticFalse => True
  | Statement.universalFalse => True

-- Define our statements
def statement1 : Statement := Statement.equation
def statement2 : Statement := Statement.question
def statement3 : Statement := Statement.arithmeticFalse
def statement4 : Statement := Statement.universalFalse

-- Theorem to prove
theorem only_statements_3_and_4_are_propositions :
  (isProposition statement1 = False) ∧
  (isProposition statement2 = False) ∧
  (isProposition statement3 = True) ∧
  (isProposition statement4 = True) :=
sorry

end NUMINAMATH_CALUDE_only_statements_3_and_4_are_propositions_l811_81195


namespace NUMINAMATH_CALUDE_friday_temperature_l811_81168

theorem friday_temperature
  (temp : Fin 5 → ℝ)
  (avg_mon_to_thu : (temp 0 + temp 1 + temp 2 + temp 3) / 4 = 48)
  (avg_tue_to_fri : (temp 1 + temp 2 + temp 3 + temp 4) / 4 = 46)
  (monday_temp : temp 0 = 42) :
  temp 4 = 34 :=
by
  sorry

end NUMINAMATH_CALUDE_friday_temperature_l811_81168


namespace NUMINAMATH_CALUDE_sin_alpha_value_l811_81181

theorem sin_alpha_value (α : Real) 
  (h1 : α > -π/2 ∧ α < π/2)
  (h2 : Real.tan α = Real.sin (76 * π / 180) * Real.cos (46 * π / 180) - 
                     Real.cos (76 * π / 180) * Real.sin (46 * π / 180)) : 
  Real.sin α = Real.sqrt 5 / 5 := by
sorry

end NUMINAMATH_CALUDE_sin_alpha_value_l811_81181


namespace NUMINAMATH_CALUDE_power_of_three_squared_cubed_squared_l811_81198

theorem power_of_three_squared_cubed_squared :
  ((3^2)^3)^2 = 531441 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_squared_cubed_squared_l811_81198
