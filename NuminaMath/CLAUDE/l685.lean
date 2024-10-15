import Mathlib

namespace NUMINAMATH_CALUDE_smallest_Y_value_l685_68510

/-- A function that checks if a natural number consists only of digits 0 and 1 -/
def only_zero_and_one (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d = 0 ∨ d = 1

/-- The theorem stating the smallest possible value of Y -/
theorem smallest_Y_value (S : ℕ) (hS : S > 0) (h_digits : only_zero_and_one S) (h_div : S % 15 = 0) :
  (S / 15 : ℕ) ≥ 74 :=
sorry

end NUMINAMATH_CALUDE_smallest_Y_value_l685_68510


namespace NUMINAMATH_CALUDE_tyson_race_time_l685_68516

/-- Calculates the total time Tyson spent in races given his swimming speeds and race details. -/
theorem tyson_race_time (lake_speed ocean_speed : ℝ) (total_races : ℕ) (race_distance : ℝ) : 
  lake_speed = 3 →
  ocean_speed = 2.5 →
  total_races = 10 →
  race_distance = 3 →
  (total_races / 2 : ℝ) * race_distance / lake_speed + 
  (total_races / 2 : ℝ) * race_distance / ocean_speed = 11 := by
  sorry


end NUMINAMATH_CALUDE_tyson_race_time_l685_68516


namespace NUMINAMATH_CALUDE_speed_conversion_l685_68591

/-- Conversion factor from m/s to km/h -/
def mps_to_kmph : ℝ := 3.6

/-- Given speed in meters per second -/
def speed_mps : ℝ := 20.0016

/-- Speed in kilometers per hour to be proven -/
def speed_kmph : ℝ := 72.00576

/-- Theorem stating that the given speed in km/h is equivalent to the speed in m/s -/
theorem speed_conversion : speed_kmph = speed_mps * mps_to_kmph := by
  sorry

end NUMINAMATH_CALUDE_speed_conversion_l685_68591


namespace NUMINAMATH_CALUDE_quadratic_root_in_interval_l685_68550

theorem quadratic_root_in_interval (a b c : ℝ) (h : 2*a + 3*b + 6*c = 0) :
  ∃ x : ℝ, 0 < x ∧ x < 1 ∧ a*x^2 + b*x + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_in_interval_l685_68550


namespace NUMINAMATH_CALUDE_polygon_sides_for_900_degrees_l685_68548

/-- The sum of interior angles of a polygon with n sides is (n - 2) * 180° --/
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

/-- For a polygon with n sides and sum of interior angles equal to 900°, n = 7 --/
theorem polygon_sides_for_900_degrees (n : ℕ) :
  sum_interior_angles n = 900 → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_for_900_degrees_l685_68548


namespace NUMINAMATH_CALUDE_statistical_relationships_properties_l685_68520

-- Define the basic concepts
def FunctionalRelationship : Type := Unit
def DeterministicRelationship : Type := Unit
def Correlation : Type := Unit
def NonDeterministicRelationship : Type := Unit
def RegressionAnalysis : Type := Unit
def StatisticalAnalysisMethod : Type := Unit
def TwoVariables : Type := Unit

-- Define the properties
def isDeterministic (r : FunctionalRelationship) : Prop := sorry
def isNonDeterministic (c : Correlation) : Prop := sorry
def isUsedFor (m : StatisticalAnalysisMethod) (v : TwoVariables) (c : Correlation) : Prop := sorry

-- Theorem to prove
theorem statistical_relationships_properties :
  (∀ (r : FunctionalRelationship), isDeterministic r) ∧
  (∀ (c : Correlation), isNonDeterministic c) ∧
  (∃ (m : RegressionAnalysis) (v : TwoVariables) (c : Correlation), 
    isUsedFor m v c) :=
by sorry

end NUMINAMATH_CALUDE_statistical_relationships_properties_l685_68520


namespace NUMINAMATH_CALUDE_cube_root_sum_of_eighth_powers_l685_68529

theorem cube_root_sum_of_eighth_powers (a b c : ℝ) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a^3 - 3*a + 1 = 0 →
  b^3 - 3*b + 1 = 0 →
  c^3 - 3*c + 1 = 0 →
  a^8 + b^8 + c^8 = 186 := by
sorry

end NUMINAMATH_CALUDE_cube_root_sum_of_eighth_powers_l685_68529


namespace NUMINAMATH_CALUDE_water_level_change_notation_l685_68576

/-- Represents the change in water level -/
def WaterLevelChange : ℤ → ℝ
  | 2 => 2
  | -2 => -2
  | _ => 0  -- Default case, not relevant for this problem

/-- The water level rise notation -/
def WaterLevelRiseNotation : ℝ := 2

/-- The water level drop notation -/
def WaterLevelDropNotation : ℝ := -2

theorem water_level_change_notation :
  WaterLevelChange 2 = WaterLevelRiseNotation ∧
  WaterLevelChange (-2) = WaterLevelDropNotation :=
by sorry

end NUMINAMATH_CALUDE_water_level_change_notation_l685_68576


namespace NUMINAMATH_CALUDE_max_value_of_expression_l685_68557

theorem max_value_of_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  let A := (a^4 + b^4 + c^4) / ((a + b + c)^4 - 80*(a*b*c)^(4/3))
  A ≤ 3 ∧ ∃ (x : ℝ), x > 0 ∧ 
    let A' := (x^4 + x^4 + x^4) / ((x + x + x)^4 - 80*(x*x*x)^(4/3))
    A' = 3 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l685_68557


namespace NUMINAMATH_CALUDE_arrangements_count_l685_68562

/-- The number of ways to arrange four people in a row with one person not at the ends -/
def arrangements_with_restriction : ℕ :=
  let total_people : ℕ := 4
  let restricted_person : ℕ := 1
  let unrestricted_people : ℕ := total_people - restricted_person
  let unrestricted_arrangements : ℕ := Nat.factorial unrestricted_people
  let valid_positions : ℕ := unrestricted_people - 1
  unrestricted_arrangements * valid_positions

theorem arrangements_count :
  arrangements_with_restriction = 12 :=
sorry

end NUMINAMATH_CALUDE_arrangements_count_l685_68562


namespace NUMINAMATH_CALUDE_function_properties_l685_68539

open Real

theorem function_properties (e : ℝ) (h_e : e = exp 1) :
  let f (a : ℝ) (x : ℝ) := a * x - log x
  let g (x : ℝ) := (log x) / x
  
  -- Part 1
  (∀ x ∈ Set.Ioo 0 e, |f 1 x| > g x + 1/2) ∧
  (∃ x₀ ∈ Set.Ioo 0 e, ∀ x ∈ Set.Ioo 0 e, f 1 x₀ ≤ f 1 x) ∧
  (∃ x₀ ∈ Set.Ioo 0 e, f 1 x₀ = 1) ∧
  
  -- Part 2
  (∃ a : ℝ, ∃ x₀ ∈ Set.Ioo 0 e, 
    (∀ x ∈ Set.Ioo 0 e, f a x₀ ≤ f a x) ∧
    f a x₀ = 3 ∧
    a = e^2) := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l685_68539


namespace NUMINAMATH_CALUDE_factor_3x_squared_minus_75_l685_68526

theorem factor_3x_squared_minus_75 (x : ℝ) : 3 * x^2 - 75 = 3 * (x + 5) * (x - 5) := by
  sorry

end NUMINAMATH_CALUDE_factor_3x_squared_minus_75_l685_68526


namespace NUMINAMATH_CALUDE_max_value_theorem_l685_68573

theorem max_value_theorem (a b c : ℝ) (h : 9 * a^2 + 4 * b^2 + 25 * c^2 = 1) :
  (∃ x y z : ℝ, 9 * x^2 + 4 * y^2 + 25 * z^2 = 1 ∧ 8 * x + 3 * y + 5 * z > 8 * a + 3 * b + 5 * c) →
  8 * a + 3 * b + 5 * c ≤ Real.sqrt (373 / 36) :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l685_68573


namespace NUMINAMATH_CALUDE_absolute_difference_l685_68592

theorem absolute_difference (a x : ℝ) (h1 : a < 0) (h2 : |a| * x ≤ a) : 
  |x + 1| - |x - 2| = -3 := by
sorry

end NUMINAMATH_CALUDE_absolute_difference_l685_68592


namespace NUMINAMATH_CALUDE_dividend_calculation_l685_68589

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 36)
  (h2 : quotient = 21)
  (h3 : remainder = 4) :
  divisor * quotient + remainder = 760 := by
sorry

end NUMINAMATH_CALUDE_dividend_calculation_l685_68589


namespace NUMINAMATH_CALUDE_car_distance_problem_l685_68515

/-- Proves that Car X travels 105 miles from when Car Y starts until both cars stop -/
theorem car_distance_problem (speed_x speed_y : ℝ) (head_start : ℝ) (distance : ℝ) : 
  speed_x = 35 →
  speed_y = 49 →
  head_start = 1.2 →
  distance = speed_x * (head_start + (distance - speed_x * head_start) / (speed_y - speed_x)) →
  distance - speed_x * head_start = 105 := by
  sorry

#check car_distance_problem

end NUMINAMATH_CALUDE_car_distance_problem_l685_68515


namespace NUMINAMATH_CALUDE_factorial_250_trailing_zeros_l685_68551

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

/-- Theorem: 250! ends with 62 zeros -/
theorem factorial_250_trailing_zeros :
  trailingZeros 250 = 62 := by
  sorry

end NUMINAMATH_CALUDE_factorial_250_trailing_zeros_l685_68551


namespace NUMINAMATH_CALUDE_correct_locus_definition_l685_68566

-- Define a type for points in a geometric space
variable {Point : Type}

-- Define a predicate for the locus condition
variable (locus_condition : Point → Prop)

-- Define the locus as a set of points
def locus (locus_condition : Point → Prop) : Set Point :=
  {p : Point | locus_condition p}

-- State the theorem
theorem correct_locus_definition (p : Point) :
  p ∈ locus locus_condition ↔ locus_condition p :=
sorry

end NUMINAMATH_CALUDE_correct_locus_definition_l685_68566


namespace NUMINAMATH_CALUDE_imaginary_unit_multiplication_l685_68535

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem imaginary_unit_multiplication :
  i * (1 - i) = 1 + i := by sorry

end NUMINAMATH_CALUDE_imaginary_unit_multiplication_l685_68535


namespace NUMINAMATH_CALUDE_equilateral_iff_rhombus_l685_68519

-- Define a parallelogram
structure Parallelogram :=
  (sides : Fin 4 → ℝ)
  (is_parallelogram : sides 0 = sides 2 ∧ sides 1 = sides 3)

-- Define an equilateral parallelogram
def is_equilateral (p : Parallelogram) : Prop :=
  p.sides 0 = p.sides 1 ∧ p.sides 1 = p.sides 2 ∧ p.sides 2 = p.sides 3

-- Define a rhombus
def is_rhombus (p : Parallelogram) : Prop :=
  p.sides 0 = p.sides 1 ∧ p.sides 1 = p.sides 2 ∧ p.sides 2 = p.sides 3

-- Theorem: A parallelogram is equilateral if and only if it is a rhombus
theorem equilateral_iff_rhombus (p : Parallelogram) :
  is_equilateral p ↔ is_rhombus p :=
sorry

end NUMINAMATH_CALUDE_equilateral_iff_rhombus_l685_68519


namespace NUMINAMATH_CALUDE_rectangle_area_with_three_squares_l685_68584

/-- The area of a rectangle containing three non-overlapping squares -/
theorem rectangle_area_with_three_squares 
  (small_square_area : ℝ) 
  (large_square_side_multiplier : ℝ) : 
  small_square_area = 1 →
  large_square_side_multiplier = 3 →
  2 * small_square_area + (large_square_side_multiplier^2 * small_square_area) = 11 :=
by
  sorry

#check rectangle_area_with_three_squares

end NUMINAMATH_CALUDE_rectangle_area_with_three_squares_l685_68584


namespace NUMINAMATH_CALUDE_min_value_expression_l685_68537

theorem min_value_expression (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0) :
  (p + q + r) * (1 / (p + q) + 1 / (p + r) + 1 / (q + r) + 1 / (p + q + r)) ≥ 5 ∧
  (∃ t : ℝ, t > 0 ∧ (t + t + t) * (1 / (t + t) + 1 / (t + t) + 1 / (t + t) + 1 / (t + t + t)) = 5) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l685_68537


namespace NUMINAMATH_CALUDE_andrea_reach_time_l685_68549

/-- The time it takes Andrea to reach Lauren's stop location -/
def time_to_reach (initial_distance : ℝ) (speed_ratio : ℝ) (distance_decrease_rate : ℝ) (lauren_stop_time : ℝ) : ℝ :=
  sorry

/-- Theorem stating the time it takes Andrea to reach Lauren's stop location -/
theorem andrea_reach_time :
  let initial_distance : ℝ := 30
  let speed_ratio : ℝ := 2
  let distance_decrease_rate : ℝ := 90
  let lauren_stop_time : ℝ := 1/6 -- 10 minutes in hours
  time_to_reach initial_distance speed_ratio distance_decrease_rate lauren_stop_time = 25/60 := by
  sorry

end NUMINAMATH_CALUDE_andrea_reach_time_l685_68549


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_l685_68588

theorem binomial_expansion_coefficient (a b : ℝ) :
  (∃ x, (1 + a*x)^5 = 1 + 10*x + b*x^2 + a^3*x^3 + a^4*x^4 + a^5*x^5) →
  b = 40 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_l685_68588


namespace NUMINAMATH_CALUDE_smallest_positive_solution_l685_68547

theorem smallest_positive_solution :
  ∃ (x : ℝ), x > 0 ∧ Real.sqrt (3 * x) = 5 * x - 2 ∧
  ∀ (y : ℝ), y > 0 → Real.sqrt (3 * y) = 5 * y - 2 → x ≤ y ∧
  x = 4 / 25 :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_solution_l685_68547


namespace NUMINAMATH_CALUDE_sqrt_real_iff_nonneg_l685_68553

theorem sqrt_real_iff_nonneg (a : ℝ) : ∃ (x : ℝ), x ^ 2 = a ↔ a ≥ 0 := by sorry

end NUMINAMATH_CALUDE_sqrt_real_iff_nonneg_l685_68553


namespace NUMINAMATH_CALUDE_system_solution_l685_68590

theorem system_solution (x y z : ℝ) : 
  (x * y * z) / (x + y) = 6/5 ∧ 
  (x * y * z) / (y + z) = 2 ∧ 
  (x * y * z) / (z + x) = 3/2 →
  ((x = 3 ∧ y = 2 ∧ z = 1) ∨ (x = -3 ∧ y = -2 ∧ z = -1)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l685_68590


namespace NUMINAMATH_CALUDE_liter_milliliter_comparison_l685_68593

theorem liter_milliliter_comparison : ¬(1000 < 9000 / 1000) := by
  sorry

end NUMINAMATH_CALUDE_liter_milliliter_comparison_l685_68593


namespace NUMINAMATH_CALUDE_min_square_difference_of_roots_l685_68571

theorem min_square_difference_of_roots (α β b : ℝ) : 
  α^2 + 2*b*α + b = 1 → β^2 + 2*b*β + b = 1 → 
  ∀ γ δ c : ℝ, (γ^2 + 2*c*γ + c = 1 ∧ δ^2 + 2*c*δ + c = 1) → 
  (α - β)^2 ≥ 3 ∧ (∃ e : ℝ, (α - β)^2 = 3) :=
by sorry

end NUMINAMATH_CALUDE_min_square_difference_of_roots_l685_68571


namespace NUMINAMATH_CALUDE_integer_roots_of_polynomial_l685_68501

def polynomial (a₂ a₁ : ℤ) (x : ℤ) : ℤ := x^3 + a₂*x^2 + a₁*x - 18

def possible_roots : Set ℤ := {-18, -9, -6, -3, -2, -1, 1, 2, 3, 6, 9, 18}

theorem integer_roots_of_polynomial (a₂ a₁ : ℤ) :
  ∀ r : ℤ, polynomial a₂ a₁ r = 0 → r ∈ possible_roots :=
sorry

end NUMINAMATH_CALUDE_integer_roots_of_polynomial_l685_68501


namespace NUMINAMATH_CALUDE_gcd_2028_2295_l685_68560

theorem gcd_2028_2295 : Nat.gcd 2028 2295 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_2028_2295_l685_68560


namespace NUMINAMATH_CALUDE_smallest_largest_five_digit_reverse_multiple_of_four_l685_68582

/-- Reverses the digits of a natural number -/
def reverseDigits (n : ℕ) : ℕ := sorry

/-- Checks if a number is a five-digit number -/
def isFiveDigit (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

theorem smallest_largest_five_digit_reverse_multiple_of_four :
  ∀ n : ℕ, isFiveDigit n → (reverseDigits n % 4 = 0) →
    21001 ≤ n ∧ n ≤ 88999 ∧
    (∀ m : ℕ, isFiveDigit m → (reverseDigits m % 4 = 0) →
      (m < 21001 ∨ 88999 < m) → False) :=
sorry

end NUMINAMATH_CALUDE_smallest_largest_five_digit_reverse_multiple_of_four_l685_68582


namespace NUMINAMATH_CALUDE_prime_product_660_l685_68536

theorem prime_product_660 (w x y z a b c d : ℕ) : 
  (w.Prime ∧ x.Prime ∧ y.Prime ∧ z.Prime) →
  (w < x ∧ x < y ∧ y < z) →
  ((w^a) * (x^b) * (y^c) * (z^d) = 660) →
  ((a + b) - (c + d) = 1) →
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_prime_product_660_l685_68536


namespace NUMINAMATH_CALUDE_savings_fraction_proof_l685_68518

def total_savings : ℕ := 180000
def ppf_savings : ℕ := 72000

theorem savings_fraction_proof :
  let nsc_savings : ℕ := total_savings - ppf_savings
  let fraction : ℚ := (1/3 : ℚ) * nsc_savings / ppf_savings
  fraction = (1/2 : ℚ) := by sorry

end NUMINAMATH_CALUDE_savings_fraction_proof_l685_68518


namespace NUMINAMATH_CALUDE_meeting_point_27_blocks_l685_68502

/-- Two people walking around a circular loop -/
def CircularWalk (total_blocks : ℕ) (speed_ratio : ℚ) : Prop :=
  ∃ (meeting_point : ℚ),
    meeting_point > 0 ∧
    meeting_point < total_blocks ∧
    meeting_point = total_blocks / (1 + speed_ratio)

/-- Theorem: In a 27-block loop with a 3:1 speed ratio, the meeting point is at 27/4 blocks -/
theorem meeting_point_27_blocks :
  CircularWalk 27 3 → (27 : ℚ) / 4 = 27 / (1 + 3) :=
by
  sorry

#check meeting_point_27_blocks

end NUMINAMATH_CALUDE_meeting_point_27_blocks_l685_68502


namespace NUMINAMATH_CALUDE_milk_needed_for_cookies_l685_68570

/-- The number of cookies in a dozen -/
def dozen : ℕ := 12

/-- The number of cookies that can be baked with 10 half-gallons of milk -/
def cookies_per_ten_halfgallons : ℕ := 40

/-- The number of half-gallons of milk needed for 40 cookies -/
def milk_for_forty_cookies : ℕ := 10

/-- The number of dozens of cookies to be baked -/
def dozens_to_bake : ℕ := 200

theorem milk_needed_for_cookies : 
  (dozens_to_bake * dozen * milk_for_forty_cookies) / cookies_per_ten_halfgallons = 600 := by
  sorry

end NUMINAMATH_CALUDE_milk_needed_for_cookies_l685_68570


namespace NUMINAMATH_CALUDE_arrange_40555_l685_68532

def digit_arrangements (n : ℕ) : ℕ := 
  if n = 40555 then 12 else 0

theorem arrange_40555 :
  digit_arrangements 40555 = 12 ∧
  (∀ x : ℕ, x ≠ 40555 → digit_arrangements x = 0) :=
sorry

end NUMINAMATH_CALUDE_arrange_40555_l685_68532


namespace NUMINAMATH_CALUDE_ceiling_sqrt_twelve_count_l685_68563

theorem ceiling_sqrt_twelve_count : 
  (Finset.filter (fun x : ℕ => ⌈Real.sqrt x⌉ = 12) (Finset.range 1000)).card = 25 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_sqrt_twelve_count_l685_68563


namespace NUMINAMATH_CALUDE_amelia_win_probability_l685_68575

/-- Probability of Amelia's coin landing heads -/
def p_amelia : ℚ := 2/7

/-- Probability of Blaine's coin landing heads -/
def p_blaine : ℚ := 1/3

/-- Probability of Amelia getting two heads in one turn -/
def p_amelia_win_turn : ℚ := p_amelia ^ 2

/-- Probability of Blaine getting two heads in one turn -/
def p_blaine_win_turn : ℚ := p_blaine ^ 2

/-- Probability of neither player winning in one round -/
def p_no_win_round : ℚ := (1 - p_amelia_win_turn) * (1 - p_blaine_win_turn)

/-- The probability that Amelia wins the game -/
theorem amelia_win_probability : 
  (p_amelia_win_turn / (1 - p_no_win_round)) = 4/9 :=
sorry

end NUMINAMATH_CALUDE_amelia_win_probability_l685_68575


namespace NUMINAMATH_CALUDE_sum_of_even_coefficients_l685_68504

theorem sum_of_even_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ a₁₂ : ℝ) :
  (∀ x : ℝ, (x + 1)^4 * (x + 4)^8 = a + a₁*(x + 3) + a₂*(x + 3)^2 + a₃*(x + 3)^3 + 
    a₄*(x + 3)^4 + a₅*(x + 3)^5 + a₆*(x + 3)^6 + a₇*(x + 3)^7 + a₈*(x + 3)^8 + 
    a₉*(x + 3)^9 + a₁₀*(x + 3)^10 + a₁₁*(x + 3)^11 + a₁₂*(x + 3)^12) →
  a₂ + a₄ + a₆ + a₈ + a₁₀ + a₁₂ = 112 := by
sorry

end NUMINAMATH_CALUDE_sum_of_even_coefficients_l685_68504


namespace NUMINAMATH_CALUDE_no_consistent_solution_l685_68545

-- Define the types for teams and match results
inductive Team : Type
| Spartak | Dynamo | Zenit | Lokomotiv

structure MatchResult :=
(winner : Team)
(loser : Team)

-- Define the problem setup
def problem_setup (match1 match2 : MatchResult) (fan_count : Team → ℕ) : Prop :=
  match1.winner ≠ match1.loser ∧ 
  match2.winner ≠ match2.loser ∧
  match1.winner ≠ match2.winner ∧
  (fan_count Team.Spartak + fan_count match1.loser + fan_count match2.loser = 200) ∧
  (fan_count Team.Dynamo + fan_count match1.loser + fan_count match2.loser = 300) ∧
  (fan_count Team.Zenit = 500) ∧
  (fan_count Team.Lokomotiv = 600)

-- Theorem statement
theorem no_consistent_solution :
  ∀ (match1 match2 : MatchResult) (fan_count : Team → ℕ),
  problem_setup match1 match2 fan_count → False :=
sorry

end NUMINAMATH_CALUDE_no_consistent_solution_l685_68545


namespace NUMINAMATH_CALUDE_perpendicular_iff_m_eq_neg_two_thirds_l685_68538

/-- Two lines in the cartesian plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The condition for two lines to be perpendicular -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- The first line in the problem -/
def line1 (m : ℝ) : Line :=
  { a := 1, b := m + 1, c := m - 2 }

/-- The second line in the problem -/
def line2 (m : ℝ) : Line :=
  { a := m, b := 2, c := 8 }

/-- The theorem to be proved -/
theorem perpendicular_iff_m_eq_neg_two_thirds :
  ∀ m : ℝ, perpendicular (line1 m) (line2 m) ↔ m = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_iff_m_eq_neg_two_thirds_l685_68538


namespace NUMINAMATH_CALUDE_price_change_theorem_l685_68599

/-- Proves that a price of $100 after a 10% increase followed by a 10% decrease results in $99 -/
theorem price_change_theorem (initial_price : ℝ) (increase_rate : ℝ) (decrease_rate : ℝ) : 
  initial_price = 100 ∧ 
  increase_rate = 0.1 ∧ 
  decrease_rate = 0.1 → 
  initial_price * (1 + increase_rate) * (1 - decrease_rate) = 99 :=
by
  sorry

#check price_change_theorem

end NUMINAMATH_CALUDE_price_change_theorem_l685_68599


namespace NUMINAMATH_CALUDE_unique_factorial_equation_l685_68585

theorem unique_factorial_equation : ∃! (N : ℕ), N > 0 ∧ ∃ (m : ℕ), m > 0 ∧ (7 : ℕ).factorial * (11 : ℕ).factorial = 20 * m * N.factorial := by
  sorry

end NUMINAMATH_CALUDE_unique_factorial_equation_l685_68585


namespace NUMINAMATH_CALUDE_ball_distribution_theorem_l685_68511

/-- The number of ways to choose 3 boxes out of 4 -/
def choose_boxes : ℕ := 4

/-- The number of ways to distribute the extra white ball -/
def distribute_white : ℕ := 3

/-- The number of ways to distribute the extra black balls -/
def distribute_black : ℕ := 6

/-- The number of ways to distribute the extra red balls -/
def distribute_red : ℕ := 10

/-- The total number of ways to distribute the balls -/
def total_ways : ℕ := choose_boxes * distribute_white * distribute_black * distribute_red

theorem ball_distribution_theorem : total_ways = 720 := by
  sorry

end NUMINAMATH_CALUDE_ball_distribution_theorem_l685_68511


namespace NUMINAMATH_CALUDE_more_girls_than_boys_l685_68506

theorem more_girls_than_boys :
  ∀ (boys girls : ℕ),
  (boys : ℚ) / (girls : ℚ) = 3 / 5 →
  boys + girls = 16 →
  girls - boys = 4 :=
by sorry

end NUMINAMATH_CALUDE_more_girls_than_boys_l685_68506


namespace NUMINAMATH_CALUDE_sqrt_sum_equality_l685_68595

theorem sqrt_sum_equality (a b m n : ℚ) : 
  Real.sqrt a + Real.sqrt b = 1 →
  Real.sqrt a = m + (a - b) / 2 →
  Real.sqrt b = n - (a - b) / 2 →
  m^2 + n^2 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equality_l685_68595


namespace NUMINAMATH_CALUDE_sphere_surface_area_l685_68567

theorem sphere_surface_area (d : ℝ) (h : d = 12) : 
  4 * Real.pi * (d / 2)^2 = 144 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l685_68567


namespace NUMINAMATH_CALUDE_consecutive_integers_divisibility_l685_68546

theorem consecutive_integers_divisibility (a₁ a₂ a₃ : ℕ) :
  a₁ > 0 ∧ a₂ > 0 ∧ a₃ > 0 ∧ a₂ = a₁ + 1 ∧ a₃ = a₂ + 1 →
  a₂^3 ∣ (a₁ * a₂ * a₃ + a₂) :=
by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_divisibility_l685_68546


namespace NUMINAMATH_CALUDE_abs_equation_solution_difference_l685_68534

theorem abs_equation_solution_difference : ∃ x₁ x₂ : ℝ, 
  (|x₁ - 4| = 12 ∧ |x₂ - 4| = 12 ∧ x₁ ≠ x₂) ∧ |x₁ - x₂| = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_abs_equation_solution_difference_l685_68534


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l685_68533

theorem quadratic_inequality_solution_set (a : ℝ) : 
  (∀ x : ℝ, x^2 - a*x < 0 ↔ 0 < x ∧ x < 1) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l685_68533


namespace NUMINAMATH_CALUDE_de_moivre_and_rationality_l685_68583

/-- De Moivre's formula and its implication on rationality of trigonometric functions -/
theorem de_moivre_and_rationality (θ : ℝ) (n : ℕ) :
  (Complex.exp (θ * Complex.I))^n = Complex.exp (n * θ * Complex.I) ∧
  (∀ (a b : ℚ), Complex.exp (θ * Complex.I) = ↑a + ↑b * Complex.I →
    ∃ (c d : ℚ), Complex.exp (n * θ * Complex.I) = ↑c + ↑d * Complex.I) := by
  sorry

end NUMINAMATH_CALUDE_de_moivre_and_rationality_l685_68583


namespace NUMINAMATH_CALUDE_property_of_x_l685_68569

theorem property_of_x (x : ℝ) (h1 : x > 0) :
  (100 - x) / 100 * x = 16 → x = 40 ∨ x = 60 := by
  sorry

end NUMINAMATH_CALUDE_property_of_x_l685_68569


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l685_68530

theorem sufficient_not_necessary_condition (a b c d : ℝ) :
  (∀ a b c d : ℝ, a > b ∧ c > d → a + c > b + d) ∧
  (∃ a b c d : ℝ, a + c > b + d ∧ ¬(a > b ∧ c > d)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l685_68530


namespace NUMINAMATH_CALUDE_sum_squares_products_bound_l685_68503

theorem sum_squares_products_bound (a b c d : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : c ≥ 0) (h4 : d ≥ 0) (h5 : a * b * c * d = 1) :
  a^2 + b^2 + c^2 + d^2 + a*b + a*c + a*d + b*c + b*d + c*d ≥ 10 := by
sorry

end NUMINAMATH_CALUDE_sum_squares_products_bound_l685_68503


namespace NUMINAMATH_CALUDE_initial_girls_count_l685_68554

theorem initial_girls_count (b g : ℕ) : 
  (3 * (g - 18) = b) →
  (4 * (b - 36) = g - 18) →
  g = 31 :=
by sorry

end NUMINAMATH_CALUDE_initial_girls_count_l685_68554


namespace NUMINAMATH_CALUDE_camping_cost_equalization_l685_68525

theorem camping_cost_equalization 
  (X Y Z : ℝ) 
  (h_order : X < Y ∧ Y < Z) :
  let total_cost := X + Y + Z
  let equal_share := total_cost / 3
  (equal_share - X) = (Y + Z - 2 * X) / 3 := by
sorry

end NUMINAMATH_CALUDE_camping_cost_equalization_l685_68525


namespace NUMINAMATH_CALUDE_parallel_postulate_l685_68542

-- Define a line in a 2D Euclidean plane
structure Line where
  -- You might represent a line using two points or a point and a direction
  -- For simplicity, we'll leave the internal representation abstract
  dummy : Unit

-- Define a point in a 2D Euclidean plane
structure Point where
  -- You might represent a point using x and y coordinates
  -- For simplicity, we'll leave the internal representation abstract
  dummy : Unit

-- Define what it means for a point to not be on a line
def Point.notOn (p : Point) (l : Line) : Prop := sorry

-- Define what it means for two lines to be parallel
def Line.parallel (l1 l2 : Line) : Prop := sorry

-- Define what it means for a line to pass through a point
def Line.passesThroughPoint (l : Line) (p : Point) : Prop := sorry

-- The parallel postulate
theorem parallel_postulate (L : Line) (P : Point) (h : P.notOn L) :
  ∃! L' : Line, L'.parallel L ∧ L'.passesThroughPoint P := by sorry

end NUMINAMATH_CALUDE_parallel_postulate_l685_68542


namespace NUMINAMATH_CALUDE_arithmetic_sequence_100th_term_nth_term_is_298_implies_n_is_100_l685_68514

/-- An arithmetic sequence with first term 1 and common difference 3 -/
def arithmetic_sequence (n : ℕ) : ℤ :=
  1 + 3 * (n - 1)

/-- Theorem stating that the 100th term of the sequence is 298 -/
theorem arithmetic_sequence_100th_term :
  arithmetic_sequence 100 = 298 :=
sorry

/-- Theorem proving that when the nth term is 298, n must be 100 -/
theorem nth_term_is_298_implies_n_is_100 (n : ℕ) :
  arithmetic_sequence n = 298 → n = 100 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_100th_term_nth_term_is_298_implies_n_is_100_l685_68514


namespace NUMINAMATH_CALUDE_rationalize_denominator_l685_68513

theorem rationalize_denominator :
  (Real.sqrt 12 + Real.sqrt 5) / (Real.sqrt 3 + Real.sqrt 5) = (Real.sqrt 15 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l685_68513


namespace NUMINAMATH_CALUDE_triangle_angle_B_value_l685_68528

theorem triangle_angle_B_value 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h1 : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h2 : 0 < A ∧ 0 < B ∧ 0 < C) 
  (h3 : A + B + C = π) 
  (h4 : (c - b) / (c - a) = Real.sin A / (Real.sin C + Real.sin B)) : 
  B = π / 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_B_value_l685_68528


namespace NUMINAMATH_CALUDE_adult_ticket_price_l685_68512

theorem adult_ticket_price 
  (total_amount : ℕ)
  (child_price : ℕ)
  (total_tickets : ℕ)
  (child_tickets : ℕ)
  (h1 : total_amount = 104)
  (h2 : child_price = 4)
  (h3 : total_tickets = 21)
  (h4 : child_tickets = 11) :
  ∃ (adult_price : ℕ), 
    adult_price * (total_tickets - child_tickets) + child_price * child_tickets = total_amount ∧ 
    adult_price = 6 :=
by sorry

end NUMINAMATH_CALUDE_adult_ticket_price_l685_68512


namespace NUMINAMATH_CALUDE_greater_number_proof_l685_68541

theorem greater_number_proof (x y : ℝ) (h1 : 4 * y = 5 * x) (h2 : x + y = 26) : 
  y = 130 / 9 := by
  sorry

end NUMINAMATH_CALUDE_greater_number_proof_l685_68541


namespace NUMINAMATH_CALUDE_regular_pentagon_diagonal_inequality_l685_68556

/-- A regular pentagon -/
structure RegularPentagon where
  side_length : ℝ
  diagonal_short : ℝ
  diagonal_long : ℝ
  side_length_pos : 0 < side_length

/-- The longer diagonal is greater than the shorter diagonal in a regular pentagon -/
theorem regular_pentagon_diagonal_inequality (p : RegularPentagon) : 
  p.diagonal_long > p.diagonal_short := by
  sorry

end NUMINAMATH_CALUDE_regular_pentagon_diagonal_inequality_l685_68556


namespace NUMINAMATH_CALUDE_max_distance_line_theorem_l685_68598

/-- The line equation that passes through point A(1, 2) and is at the maximum distance from the origin -/
def max_distance_line : ℝ → ℝ → Prop :=
  fun x y => x + 2 * y - 5 = 0

/-- Point A -/
def point_A : ℝ × ℝ := (1, 2)

/-- The origin -/
def origin : ℝ × ℝ := (0, 0)

theorem max_distance_line_theorem :
  (max_distance_line (point_A.1) (point_A.2)) ∧
  (∀ x y, max_distance_line x y → 
    ∀ a b, (a, b) ≠ origin → 
      (a - origin.1)^2 + (b - origin.2)^2 ≤ (x - origin.1)^2 + (y - origin.2)^2) :=
sorry

end NUMINAMATH_CALUDE_max_distance_line_theorem_l685_68598


namespace NUMINAMATH_CALUDE_distance_between_points_l685_68505

def point1 : ℝ × ℝ := (3, 7)
def point2 : ℝ × ℝ := (3, -4)

theorem distance_between_points : 
  |point1.2 - point2.2| = 11 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l685_68505


namespace NUMINAMATH_CALUDE_ellipse_param_sum_l685_68521

/-- An ellipse with given properties -/
structure Ellipse where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ
  constant_sum : ℝ
  tangent_slope : ℝ

/-- The standard form parameters of an ellipse -/
structure EllipseParams where
  h : ℝ
  k : ℝ
  a : ℝ
  b : ℝ

/-- Theorem stating the sum of h, k, a, and b for the given ellipse -/
theorem ellipse_param_sum (e : Ellipse) (p : EllipseParams) : 
  e.F₁ = (-1, 1) → 
  e.F₂ = (5, 1) → 
  e.constant_sum = 10 → 
  e.tangent_slope = 1 → 
  p.h + p.k + p.a + p.b = 12 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_param_sum_l685_68521


namespace NUMINAMATH_CALUDE_units_digit_of_expression_l685_68517

def units_digit (n : ℤ) : ℕ := n.natAbs % 10

theorem units_digit_of_expression : units_digit (8 * 19 * 1981 + 6^3 - 2^5) = 6 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_expression_l685_68517


namespace NUMINAMATH_CALUDE_spongebob_burger_price_l685_68587

/-- The price of a burger in Spongebob's shop -/
def burger_price : ℝ := 2

/-- The number of burgers sold -/
def burgers_sold : ℕ := 30

/-- The number of large fries sold -/
def fries_sold : ℕ := 12

/-- The price of each large fries -/
def fries_price : ℝ := 1.5

/-- The total earnings for the day -/
def total_earnings : ℝ := 78

theorem spongebob_burger_price :
  burger_price * burgers_sold + fries_price * fries_sold = total_earnings :=
by sorry

end NUMINAMATH_CALUDE_spongebob_burger_price_l685_68587


namespace NUMINAMATH_CALUDE_infinite_series_sum_l685_68544

/-- The sum of the infinite series ∑(k=1 to ∞) [6^k / ((3^k - 2^k)(3^(k+1) - 2^(k+1)))] is equal to 2. -/
theorem infinite_series_sum : 
  (∑' k : ℕ, (6:ℝ)^k / ((3:ℝ)^k - (2:ℝ)^k * ((3:ℝ)^(k+1) - (2:ℝ)^(k+1)))) = 2 := by
  sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l685_68544


namespace NUMINAMATH_CALUDE_equation_solution_l685_68561

theorem equation_solution : ∃ (S : Set ℝ), S = {x : ℝ | (3*x + 6) / (x^2 + 5*x + 6) = (3 - x) / (x - 2) ∧ x ≠ 2 ∧ x ≠ -2} ∧ S = {3, -3} := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l685_68561


namespace NUMINAMATH_CALUDE_qr_length_l685_68509

/-- Right triangle ABC with hypotenuse AB = 13, AC = 12, and BC = 5 -/
structure RightTriangle where
  AB : ℝ
  AC : ℝ
  BC : ℝ
  right_angle : AB^2 = AC^2 + BC^2
  AB_eq : AB = 13
  AC_eq : AC = 12
  BC_eq : BC = 5

/-- Circle P passing through C and tangent to BC -/
structure CircleP (t : RightTriangle) where
  center : ℝ × ℝ
  radius : ℝ
  passes_through_C : True  -- Simplified condition
  tangent_to_BC : True     -- Simplified condition
  smallest : True          -- Simplified condition

/-- Points Q and R as intersections of circle P with AC and AB -/
structure Intersections (t : RightTriangle) (p : CircleP t) where
  Q : ℝ × ℝ
  R : ℝ × ℝ
  Q_on_AC : True           -- Simplified condition
  R_on_AB : True           -- Simplified condition
  Q_on_circle : True       -- Simplified condition
  R_on_circle : True       -- Simplified condition

/-- Main theorem: Length of QR is 5.42 -/
theorem qr_length (t : RightTriangle) (p : CircleP t) (i : Intersections t p) :
  Real.sqrt ((i.Q.1 - i.R.1)^2 + (i.Q.2 - i.R.2)^2) = 5.42 := by
  sorry

end NUMINAMATH_CALUDE_qr_length_l685_68509


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l685_68594

/-- The sum of the repeating decimals 0.4̄ and 0.26̄ is equal to 70/99 -/
theorem sum_of_repeating_decimals : 
  (∃ (x y : ℚ), x = 4/9 ∧ y = 26/99 ∧ x + y = 70/99) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l685_68594


namespace NUMINAMATH_CALUDE_arithmetic_sequence_quadratic_root_l685_68555

/-- Given real numbers x, y, z forming an arithmetic sequence with x ≥ y ≥ z ≥ 0,
    and the quadratic equation yx^2 + zx + y = 0 having exactly one root,
    prove that this root is 4. -/
theorem arithmetic_sequence_quadratic_root :
  ∀ (x y z : ℝ),
  (∃ (d : ℝ), y = x - d ∧ z = x - 2*d) →  -- arithmetic sequence condition
  x ≥ y ∧ y ≥ z ∧ z ≥ 0 →                -- ordering condition
  (∀ r : ℝ, y*r^2 + z*r + y = 0 ↔ r = 4) →  -- unique root condition
  ∀ r : ℝ, y*r^2 + z*r + y = 0 → r = 4 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_quadratic_root_l685_68555


namespace NUMINAMATH_CALUDE_gumball_count_l685_68508

def gumball_machine (red : ℕ) : Prop :=
  ∃ (blue green yellow orange : ℕ),
    blue = red / 2 ∧
    green = 4 * blue ∧
    yellow = (60 * green) / 100 ∧
    orange = (red + blue) / 3 ∧
    red + blue + green + yellow + orange = 124

theorem gumball_count : gumball_machine 24 := by
  sorry

end NUMINAMATH_CALUDE_gumball_count_l685_68508


namespace NUMINAMATH_CALUDE_odd_divides_power_two_minus_one_l685_68574

theorem odd_divides_power_two_minus_one (a : ℕ) (h : Odd a) :
  ∃ b : ℕ, a ∣ (2^b - 1) := by sorry

end NUMINAMATH_CALUDE_odd_divides_power_two_minus_one_l685_68574


namespace NUMINAMATH_CALUDE_parallelogram_area_l685_68500

/-- The area of a parallelogram with vertices at (0, 0), (3, 0), (1, 5), and (4, 5) is 15 square units. -/
theorem parallelogram_area : 
  let v1 : ℝ × ℝ := (0, 0)
  let v2 : ℝ × ℝ := (3, 0)
  let v3 : ℝ × ℝ := (1, 5)
  let v4 : ℝ × ℝ := (4, 5)
  let base : ℝ := v2.1 - v1.1
  let height : ℝ := v3.2 - v1.2
  base * height = 15 := by sorry

end NUMINAMATH_CALUDE_parallelogram_area_l685_68500


namespace NUMINAMATH_CALUDE_compute_fraction_power_l685_68507

theorem compute_fraction_power : 8 * (2 / 3)^4 = 128 / 81 := by
  sorry

end NUMINAMATH_CALUDE_compute_fraction_power_l685_68507


namespace NUMINAMATH_CALUDE_quadratic_inequality_l685_68597

theorem quadratic_inequality (x : ℝ) : x^2 - 42*x + 400 ≤ 10 ↔ 13 ≤ x ∧ x ≤ 30 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l685_68597


namespace NUMINAMATH_CALUDE_market_price_calculation_l685_68543

/-- Given an initial sales tax rate, a reduced sales tax rate, and the difference in tax amount,
    proves that the market price of an article is 6600. -/
theorem market_price_calculation (initial_rate reduced_rate : ℚ) (tax_difference : ℝ) :
  initial_rate = 35 / 1000 →
  reduced_rate = 100 / 3000 →
  tax_difference = 10.999999999999991 →
  ∃ (price : ℕ), price = 6600 ∧ (initial_rate - reduced_rate) * price = tax_difference :=
sorry

end NUMINAMATH_CALUDE_market_price_calculation_l685_68543


namespace NUMINAMATH_CALUDE_system_solution_l685_68524

theorem system_solution (a b c x y z : ℝ) : 
  (a * x + (a - b) * y + (a - c) * z = a^2 + (b - c)^2) ∧
  ((b - a) * x + b * y + (b - c) * z = b^2 + (c - a)^2) ∧
  ((c - a) * x + (c - b) * y + c * z = c^2 + (a - b)^2) →
  (x = b + c - a ∧ y = c + a - b ∧ z = a + b - c) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l685_68524


namespace NUMINAMATH_CALUDE_elder_sister_age_when_sum_was_twenty_l685_68580

/-- 
Given:
- The younger sister is currently 18 years old
- The elder sister is currently 26 years old
- At some point in the past, the sum of their ages was 20 years

Prove that when the sum of their ages was 20 years, the elder sister was 14 years old.
-/
theorem elder_sister_age_when_sum_was_twenty 
  (younger_current : ℕ) 
  (elder_current : ℕ) 
  (years_ago : ℕ) 
  (h1 : younger_current = 18) 
  (h2 : elder_current = 26) 
  (h3 : younger_current - years_ago + elder_current - years_ago = 20) : 
  elder_current - years_ago = 14 :=
sorry

end NUMINAMATH_CALUDE_elder_sister_age_when_sum_was_twenty_l685_68580


namespace NUMINAMATH_CALUDE_consecutive_products_divisibility_l685_68540

theorem consecutive_products_divisibility (a : ℤ) :
  ∃ k : ℤ, a * (a + 1) + (a + 1) * (a + 2) + (a + 2) * (a + 3) + a * (a + 3) + 1 = 12 * k :=
by sorry

end NUMINAMATH_CALUDE_consecutive_products_divisibility_l685_68540


namespace NUMINAMATH_CALUDE_principal_proof_l685_68596

/-- The principal amount that satisfies the given conditions -/
def principal_amount : ℝ := by sorry

theorem principal_proof :
  let R : ℝ := 0.05  -- Interest rate (5% per annum)
  let T : ℝ := 10    -- Time period in years
  let P : ℝ := principal_amount
  let I : ℝ := P * R * T  -- Interest calculation
  (P - I = P - 3100) →  -- Interest is 3100 less than principal
  P = 6200 := by sorry

end NUMINAMATH_CALUDE_principal_proof_l685_68596


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l685_68586

theorem quadratic_expression_value (a : ℝ) (h : 2 * a^2 - a - 3 = 0) :
  (2 * a + 3) * (2 * a - 3) + (2 * a - 1)^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l685_68586


namespace NUMINAMATH_CALUDE_added_number_proof_l685_68531

theorem added_number_proof : 
  let n : ℝ := 90
  let x : ℝ := 3
  (1/2 : ℝ) * (1/3 : ℝ) * (1/5 : ℝ) * n + x = (1/15 : ℝ) * n := by
  sorry

end NUMINAMATH_CALUDE_added_number_proof_l685_68531


namespace NUMINAMATH_CALUDE_tan_sum_product_equals_one_l685_68559

theorem tan_sum_product_equals_one :
  Real.tan (22 * π / 180) + Real.tan (23 * π / 180) + Real.tan (22 * π / 180) * Real.tan (23 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_product_equals_one_l685_68559


namespace NUMINAMATH_CALUDE_surface_area_circumscribed_sphere_l685_68579

/-- The surface area of a sphere circumscribing a cube with edge length 1 is 3π. -/
theorem surface_area_circumscribed_sphere (cube_edge : Real) (sphere_radius : Real) :
  cube_edge = 1 →
  sphere_radius = (Real.sqrt 3) / 2 →
  4 * Real.pi * sphere_radius ^ 2 = 3 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_surface_area_circumscribed_sphere_l685_68579


namespace NUMINAMATH_CALUDE_vector_equality_conditions_l685_68558

theorem vector_equality_conditions (n : ℕ) :
  ∃ (a b : Fin n → ℝ),
    (norm a = norm b ∧ norm (a + b) ≠ norm (a - b)) ∧
    ∃ (c d : Fin n → ℝ),
      (norm (c + d) = norm (c - d) ∧ norm c ≠ norm d) :=
by sorry

end NUMINAMATH_CALUDE_vector_equality_conditions_l685_68558


namespace NUMINAMATH_CALUDE_cabbage_production_increase_cabbage_production_increase_holds_l685_68577

theorem cabbage_production_increase : ℕ → Prop :=
  fun n =>
    (∃ a : ℕ, a * a = 11236) ∧
    (∀ b : ℕ, b * b < 11236 → b * b ≤ (n - 1) * (n - 1)) ∧
    (n * n < 11236) →
    11236 - (n - 1) * (n - 1) = 211

theorem cabbage_production_increase_holds : cabbage_production_increase 106 := by
  sorry

end NUMINAMATH_CALUDE_cabbage_production_increase_cabbage_production_increase_holds_l685_68577


namespace NUMINAMATH_CALUDE_limit_proof_l685_68564

theorem limit_proof : ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ,
  0 < |x - 1/3| ∧ |x - 1/3| < δ →
  |(15*x^2 - 2*x - 1) / (x - 1/3) - 8| < ε := by
  sorry

end NUMINAMATH_CALUDE_limit_proof_l685_68564


namespace NUMINAMATH_CALUDE_polynomial_factorization_l685_68565

theorem polynomial_factorization :
  ∀ x : ℝ, x^12 + x^6 + 1 = (x^4 - x^2 + 1) * (x^8 + x^4 + 1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l685_68565


namespace NUMINAMATH_CALUDE_apple_piles_l685_68581

/-- Given two piles of apples, prove the original number in the second pile -/
theorem apple_piles (a : ℕ) : 
  (∃ b : ℕ, (a - 2) * 2 = b + 2) → 
  (∃ b : ℕ, b = 2 * a - 6) :=
by sorry

end NUMINAMATH_CALUDE_apple_piles_l685_68581


namespace NUMINAMATH_CALUDE_triangle_inequality_l685_68552

/-- Given a triangle ABC with sides a ≤ b ≤ c, angle bisectors l_a, l_b, l_c,
    and corresponding medians m_a, m_b, m_c, prove that
    h_n/m_n + h_n/m_h_n + l_c/m_m_p > 1 -/
theorem triangle_inequality (a b c : ℝ) (h_sides : 0 < a ∧ a ≤ b ∧ b ≤ c)
  (l_a l_b l_c : ℝ) (h_bisectors : l_a > 0 ∧ l_b > 0 ∧ l_c > 0)
  (m_a m_b m_c : ℝ) (h_medians : m_a > 0 ∧ m_b > 0 ∧ m_c > 0)
  (h_n m_n m_h_n m_m_p : ℝ) (h_positive : h_n > 0 ∧ m_n > 0 ∧ m_h_n > 0 ∧ m_m_p > 0) :
  h_n / m_n + h_n / m_h_n + l_c / m_m_p > 1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l685_68552


namespace NUMINAMATH_CALUDE_composite_19_8n_17_l685_68578

theorem composite_19_8n_17 (n : ℕ) (hn : n > 0) : 
  ∃ (k : ℕ), k > 1 ∧ k < 19 * 8^n + 17 ∧ (19 * 8^n + 17) % k = 0 := by
  sorry

end NUMINAMATH_CALUDE_composite_19_8n_17_l685_68578


namespace NUMINAMATH_CALUDE_rectangle_x_value_l685_68568

/-- A rectangle in a 2D plane --/
structure Rectangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- The area of a rectangle --/
def rectangleArea (r : Rectangle) : ℝ :=
  (r.C.1 - r.A.1) * (r.B.2 - r.A.2)

theorem rectangle_x_value 
  (x : ℝ) 
  (h_pos : x > 0) 
  (rect : Rectangle) 
  (h_vertices : rect = { 
    A := (0, 0), 
    B := (0, 4), 
    C := (x, 4), 
    D := (x, 0) 
  }) 
  (h_area : rectangleArea rect = 28) : 
  x = 7 := by
sorry

end NUMINAMATH_CALUDE_rectangle_x_value_l685_68568


namespace NUMINAMATH_CALUDE_candy_chocolate_cost_difference_l685_68572

/-- The cost difference between a candy bar and a chocolate -/
def cost_difference (candy_bar_cost chocolate_cost : ℕ) : ℕ :=
  candy_bar_cost - chocolate_cost

/-- Theorem: The cost difference between a $7 candy bar and a $3 chocolate is $4 -/
theorem candy_chocolate_cost_difference :
  cost_difference 7 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_candy_chocolate_cost_difference_l685_68572


namespace NUMINAMATH_CALUDE_cats_adopted_count_l685_68527

/-- The cost to get a cat ready for adoption -/
def cat_cost : ℕ := 50

/-- The cost to get an adult dog ready for adoption -/
def adult_dog_cost : ℕ := 100

/-- The cost to get a puppy ready for adoption -/
def puppy_cost : ℕ := 150

/-- The number of adult dogs adopted -/
def adult_dogs_adopted : ℕ := 3

/-- The number of puppies adopted -/
def puppies_adopted : ℕ := 2

/-- The total cost for all adopted animals -/
def total_cost : ℕ := 700

/-- Theorem stating that the number of cats adopted is 2 -/
theorem cats_adopted_count : 
  ∃ (c : ℕ), c * cat_cost + adult_dogs_adopted * adult_dog_cost + puppies_adopted * puppy_cost = total_cost ∧ c = 2 :=
by sorry

end NUMINAMATH_CALUDE_cats_adopted_count_l685_68527


namespace NUMINAMATH_CALUDE_vote_change_theorem_l685_68523

/-- Represents the voting results of an assembly --/
structure VotingResults where
  total_members : ℕ
  initial_for : ℕ
  initial_against : ℕ
  revote_for : ℕ
  revote_against : ℕ

/-- Theorem about the change in votes for a resolution --/
theorem vote_change_theorem (v : VotingResults) : 
  v.total_members = 500 →
  v.initial_for + v.initial_against = v.total_members →
  v.revote_for + v.revote_against = v.total_members →
  v.initial_against > v.initial_for →
  v.revote_for > v.revote_against →
  (v.revote_for - v.revote_against) = 3 * (v.initial_against - v.initial_for) →
  v.revote_for = (7 * v.initial_against) / 6 →
  v.revote_for - v.initial_for = 90 := by
  sorry


end NUMINAMATH_CALUDE_vote_change_theorem_l685_68523


namespace NUMINAMATH_CALUDE_polynomial_root_sum_l685_68522

theorem polynomial_root_sum (a b c d e : ℤ) : 
  let g : ℝ → ℝ := λ x => x^5 + a*x^4 + b*x^3 + c*x^2 + d*x + e
  (∀ r : ℝ, g r = 0 → ∃ k : ℤ, r = -k ∧ k > 0) →
  a + b + c + d + e = 3403 →
  e = 9240 := by
sorry

end NUMINAMATH_CALUDE_polynomial_root_sum_l685_68522
