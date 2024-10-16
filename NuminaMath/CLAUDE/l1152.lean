import Mathlib

namespace NUMINAMATH_CALUDE_f_property_l1152_115231

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 1 then Real.sqrt x
  else if x ≥ 1 then 2 * (x - 1)
  else 0  -- We define f as 0 for x ≤ 0 to make it total

-- State the theorem
theorem f_property : ∃ a : ℝ, f a = f (a + 1) → f (1 / a) = 6 := by
  sorry

end NUMINAMATH_CALUDE_f_property_l1152_115231


namespace NUMINAMATH_CALUDE_b_work_time_l1152_115291

-- Define the work rates for A, B, and C
def work_rate_A : ℚ := 1 / 5
def work_rate_BC : ℚ := 1 / 3
def work_rate_AC : ℚ := 1 / 2

-- Theorem to prove
theorem b_work_time (work_rate_B : ℚ) : 
  work_rate_A + (work_rate_BC - work_rate_B) = work_rate_AC → 
  (1 : ℚ) / work_rate_B = 30 := by
  sorry

end NUMINAMATH_CALUDE_b_work_time_l1152_115291


namespace NUMINAMATH_CALUDE_heather_start_time_l1152_115276

/-- Proves that Heather started her journey 24 minutes after Stacy given the problem conditions -/
theorem heather_start_time (total_distance : ℝ) (heather_speed : ℝ) (stacy_speed : ℝ) 
  (heather_distance : ℝ) :
  total_distance = 10 →
  heather_speed = 5 →
  stacy_speed = heather_speed + 1 →
  heather_distance = 3.4545454545454546 →
  (total_distance - heather_distance) / stacy_speed - heather_distance / heather_speed = 0.4 :=
by sorry

end NUMINAMATH_CALUDE_heather_start_time_l1152_115276


namespace NUMINAMATH_CALUDE_square_side_length_l1152_115227

theorem square_side_length 
  (total_width : ℕ) 
  (total_height : ℕ) 
  (r : ℕ) 
  (s : ℕ) :
  total_width = 3330 →
  total_height = 2030 →
  2 * r + s = total_height →
  2 * r + 3 * s = total_width →
  s = 650 :=
by sorry

end NUMINAMATH_CALUDE_square_side_length_l1152_115227


namespace NUMINAMATH_CALUDE_negative_product_implies_positive_fraction_l1152_115260

theorem negative_product_implies_positive_fraction
  (x y z : ℝ) (h : x * y^3 * z^2 < 0) (hy : y ≠ 0) :
  -(x^3 * z^4) / y^5 > 0 :=
by sorry

end NUMINAMATH_CALUDE_negative_product_implies_positive_fraction_l1152_115260


namespace NUMINAMATH_CALUDE_roots_of_quadratic_l1152_115235

theorem roots_of_quadratic (x : ℝ) : x * (x - 1) = 0 ↔ x = 0 ∨ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_roots_of_quadratic_l1152_115235


namespace NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l1152_115282

-- Define the vectors a and b
def a : Fin 2 → ℝ := ![2, -1]
def b : Fin 2 → ℝ := ![1, 3]

-- Define the dot product of two 2D vectors
def dot_product (v w : Fin 2 → ℝ) : ℝ :=
  (v 0) * (w 0) + (v 1) * (w 1)

-- Define vector addition
def vec_add (v w : Fin 2 → ℝ) : Fin 2 → ℝ :=
  ![v 0 + w 0, v 1 + w 1]

-- Define scalar multiplication
def scalar_mult (c : ℝ) (v : Fin 2 → ℝ) : Fin 2 → ℝ :=
  ![c * (v 0), c * (v 1)]

-- Theorem statement
theorem perpendicular_vectors_m_value :
  ∃ m : ℝ, dot_product a (vec_add a (scalar_mult m b)) = 0 ∧ m = 5 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l1152_115282


namespace NUMINAMATH_CALUDE_sum_of_digits_3125_base6_l1152_115223

/-- Converts a natural number to its base 6 representation -/
def toBase6 (n : ℕ) : List ℕ :=
  sorry

/-- Sums the digits in a list -/
def sumDigits (digits : List ℕ) : ℕ :=
  sorry

/-- Theorem: The sum of digits of 3125 in base 6 equals 15 -/
theorem sum_of_digits_3125_base6 : sumDigits (toBase6 3125) = 15 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_3125_base6_l1152_115223


namespace NUMINAMATH_CALUDE_no_integer_square_root_l1152_115245

-- Define the polynomial Q
def Q (x : ℤ) : ℤ := x^4 + 5*x^3 + 10*x^2 + 5*x + 25

-- Theorem statement
theorem no_integer_square_root : ∀ x : ℤ, ¬∃ y : ℤ, Q x = y^2 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_square_root_l1152_115245


namespace NUMINAMATH_CALUDE_vector_projection_l1152_115259

/-- Given vectors m and n, prove that the projection of m onto n is 8√13/13 -/
theorem vector_projection (m n : ℝ × ℝ) : m = (1, 2) → n = (2, 3) → 
  (m.1 * n.1 + m.2 * n.2) / Real.sqrt (n.1^2 + n.2^2) = 8 * Real.sqrt 13 / 13 := by
  sorry

end NUMINAMATH_CALUDE_vector_projection_l1152_115259


namespace NUMINAMATH_CALUDE_equation_transformation_l1152_115246

theorem equation_transformation (x y : ℝ) : x = y → -2 * x = -2 * y := by
  sorry

end NUMINAMATH_CALUDE_equation_transformation_l1152_115246


namespace NUMINAMATH_CALUDE_carter_drum_sticks_l1152_115244

/-- The number of drum stick sets Carter uses per show -/
def sticks_used_per_show : ℕ := 8

/-- The number of drum stick sets Carter tosses to the audience after each show -/
def sticks_tossed_per_show : ℕ := 10

/-- The number of nights Carter performs -/
def number_of_shows : ℕ := 45

/-- The total number of drum stick sets Carter goes through -/
def total_sticks : ℕ := (sticks_used_per_show + sticks_tossed_per_show) * number_of_shows

theorem carter_drum_sticks :
  total_sticks = 810 := by sorry

end NUMINAMATH_CALUDE_carter_drum_sticks_l1152_115244


namespace NUMINAMATH_CALUDE_game_result_l1152_115261

def point_function (n : Nat) : Nat :=
  if n % 3 = 0 then 8
  else if n % 2 = 0 then 4
  else 0

def alex_rolls : List Nat := [6, 4, 3, 2, 1]
def bob_rolls : List Nat := [5, 6, 2, 3, 3]

def calculate_points (rolls : List Nat) : Nat :=
  (rolls.map point_function).sum

theorem game_result : 
  (calculate_points alex_rolls) * (calculate_points bob_rolls) = 672 := by
  sorry

end NUMINAMATH_CALUDE_game_result_l1152_115261


namespace NUMINAMATH_CALUDE_pump_fill_time_solution_l1152_115247

def pump_fill_time (P : ℝ) : Prop :=
  P > 0 ∧ (1 / P - 1 / 14 = 3 / 7)

theorem pump_fill_time_solution :
  ∃ P, pump_fill_time P ∧ P = 2 := by sorry

end NUMINAMATH_CALUDE_pump_fill_time_solution_l1152_115247


namespace NUMINAMATH_CALUDE_five_people_six_chairs_l1152_115204

/-- The number of ways to arrange n people in m chairs in a row -/
def arrangements (n m : ℕ) : ℕ :=
  (m.factorial) / ((m - n).factorial)

/-- Theorem: There are 720 ways to arrange 5 people in a row of 6 chairs -/
theorem five_people_six_chairs : arrangements 5 6 = 720 := by
  sorry

end NUMINAMATH_CALUDE_five_people_six_chairs_l1152_115204


namespace NUMINAMATH_CALUDE_simple_interest_rate_problem_l1152_115269

/-- Given the principal, amount, time, and formulas for simple interest and amount,
    prove that the rate percent is 5%. -/
theorem simple_interest_rate_problem (P A : ℕ) (T : ℕ) (h1 : P = 750) (h2 : A = 900) (h3 : T = 4) :
  ∃ R : ℚ,
    R = 5 ∧
    A = P + P * R * (T : ℚ) / 100 :=
by sorry

end NUMINAMATH_CALUDE_simple_interest_rate_problem_l1152_115269


namespace NUMINAMATH_CALUDE_phillip_and_paula_numbers_l1152_115229

theorem phillip_and_paula_numbers (a b : ℚ) 
  (h1 : a = b + 12)
  (h2 : a^2 + b^2 = 169/2)
  (h3 : a^4 - b^4 = 5070) : 
  a + b = 5 := by
  sorry

end NUMINAMATH_CALUDE_phillip_and_paula_numbers_l1152_115229


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_range_l1152_115234

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_common_ratio_range 
  (a : ℕ → ℝ) (q : ℝ) (h1 : is_geometric_sequence a) 
  (h2 : a 1 * (a 2 + a 3) = 6 * a 1 - 9) :
  (-1 - Real.sqrt 5) / 2 ≤ q ∧ q ≤ (-1 + Real.sqrt 5) / 2 ∧ q ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_range_l1152_115234


namespace NUMINAMATH_CALUDE_power_and_division_equality_l1152_115232

theorem power_and_division_equality : (12 : ℕ)^3 * 6^4 / 432 = 5184 := by sorry

end NUMINAMATH_CALUDE_power_and_division_equality_l1152_115232


namespace NUMINAMATH_CALUDE_percentage_relation_l1152_115253

/-- Given that j is 25% less than p, j is 20% less than t, and t is q% less than p, prove that q = 6.25% -/
theorem percentage_relation (p t j : ℝ) (q : ℝ) 
  (h1 : j = p * (1 - 0.25))
  (h2 : j = t * (1 - 0.20))
  (h3 : t = p * (1 - q / 100)) :
  q = 6.25 := by
sorry

end NUMINAMATH_CALUDE_percentage_relation_l1152_115253


namespace NUMINAMATH_CALUDE_max_stores_visited_l1152_115255

theorem max_stores_visited (total_stores : ℕ) (total_visits : ℕ) (total_shoppers : ℕ) (double_visitors : ℕ) 
  (h1 : total_stores = 8)
  (h2 : total_visits = 23)
  (h3 : total_shoppers = 12)
  (h4 : double_visitors = 8)
  (h5 : double_visitors ≤ total_shoppers)
  (h6 : double_visitors * 2 ≤ total_visits)
  (h7 : ∀ n : ℕ, n ≤ total_shoppers → n > 0) :
  ∃ max_visits : ℕ, max_visits ≤ total_stores ∧ 
    max_visits * 1 + (total_shoppers - 1) * 1 + double_visitors * 1 = total_visits ∧
    ∀ n : ℕ, n ≤ total_shoppers → n * total_stores ≥ total_visits → n ≥ total_shoppers - 1 :=
by sorry

end NUMINAMATH_CALUDE_max_stores_visited_l1152_115255


namespace NUMINAMATH_CALUDE_triangle_property_l1152_115266

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
theorem triangle_property (A B C : Real) (a b c : Real) :
  -- Triangle existence conditions
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  0 < a ∧ 0 < b ∧ 0 < c →
  -- Law of sines
  a / Real.sin A = b / Real.sin B →
  -- Given condition
  Real.cos A / (1 + Real.sin A) = Real.sin B / (1 + Real.cos B) →
  -- Conclusions
  C = π / 2 ∧ 
  1 < (a * b + b * c + c * a) / (c^2) ∧ 
  (a * b + b * c + c * a) / (c^2) ≤ (1 + 2 * Real.sqrt 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_property_l1152_115266


namespace NUMINAMATH_CALUDE_no_right_triangle_with_given_conditions_l1152_115299

theorem no_right_triangle_with_given_conditions :
  ¬ ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  a + b = 8 ∧ c = 5 ∧ a^2 + b^2 = c^2 :=
sorry

end NUMINAMATH_CALUDE_no_right_triangle_with_given_conditions_l1152_115299


namespace NUMINAMATH_CALUDE_dealership_anticipation_l1152_115271

/-- Given a ratio of SUVs to trucks and an expected number of SUVs,
    calculate the anticipated number of trucks -/
def anticipatedTrucks (suvRatio truckRatio expectedSUVs : ℕ) : ℕ :=
  (expectedSUVs * truckRatio) / suvRatio

/-- Theorem: Given the ratio of SUVs to trucks is 3:5,
    if 45 SUVs are expected to be sold,
    then 75 trucks are anticipated to be sold -/
theorem dealership_anticipation :
  anticipatedTrucks 3 5 45 = 75 := by
  sorry

end NUMINAMATH_CALUDE_dealership_anticipation_l1152_115271


namespace NUMINAMATH_CALUDE_actual_time_when_clock_shows_7pm_l1152_115258

/-- Represents time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  hLt24 : hours < 24
  mLt60 : minutes < 60

/-- Converts Time to minutes since midnight -/
def timeToMinutes (t : Time) : ℕ := t.hours * 60 + t.minutes

/-- Represents a clock that may gain or lose time -/
structure Clock where
  rate : ℚ  -- Rate of time gain/loss (1 means accurate, >1 means gaining time)

theorem actual_time_when_clock_shows_7pm 
  (c : Clock) 
  (h1 : c.rate = 7 / 6)  -- Clock gains 5 minutes in 30 minutes
  (h2 : timeToMinutes { hours := 7, minutes := 0, hLt24 := by norm_num, mLt60 := by norm_num } = 
        c.rate * timeToMinutes { hours := 18, minutes := 0, hLt24 := by norm_num, mLt60 := by norm_num }) :
  timeToMinutes { hours := 18, minutes := 0, hLt24 := by norm_num, mLt60 := by norm_num } = 
  timeToMinutes { hours := 18, minutes := 0, hLt24 := by norm_num, mLt60 := by norm_num } := by
  sorry

end NUMINAMATH_CALUDE_actual_time_when_clock_shows_7pm_l1152_115258


namespace NUMINAMATH_CALUDE_equation_solution_l1152_115211

theorem equation_solution : 
  ∃ x : ℝ, (7 + 3.5 * x = 2.1 * x - 25) ∧ (x = -32 / 1.4) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1152_115211


namespace NUMINAMATH_CALUDE_trig_sum_zero_l1152_115297

theorem trig_sum_zero (θ : ℝ) (a : ℝ) (h : Real.cos (π / 6 - θ) = a) :
  Real.cos (5 * π / 6 + θ) + Real.sin (2 * π / 3 - θ) = 0 := by
  sorry

end NUMINAMATH_CALUDE_trig_sum_zero_l1152_115297


namespace NUMINAMATH_CALUDE_binary_to_octal_conversion_l1152_115226

/-- Converts a binary number represented as a list of bits to its decimal representation -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- Converts a decimal number to its octal representation -/
def decimal_to_octal (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) :=
      if m = 0 then acc
      else aux (m / 8) ((m % 8) :: acc)
    aux n []

/-- The binary representation of 1010 101₂ -/
def binary_num : List Bool := [true, false, true, false, true, false, true]

/-- The octal representation of 125₈ -/
def octal_num : List ℕ := [1, 2, 5]

theorem binary_to_octal_conversion :
  decimal_to_octal (binary_to_decimal binary_num) = octal_num := by
  sorry

#eval binary_to_decimal binary_num
#eval decimal_to_octal (binary_to_decimal binary_num)

end NUMINAMATH_CALUDE_binary_to_octal_conversion_l1152_115226


namespace NUMINAMATH_CALUDE_min_value_of_f_l1152_115228

/-- The function f(x) = x^2 + 14x + 10 -/
def f (x : ℝ) : ℝ := x^2 + 14*x + 10

/-- The minimum value of f(x) is -39 -/
theorem min_value_of_f :
  ∃ (m : ℝ), (∀ x, f x ≥ m) ∧ (∃ x₀, f x₀ = m) ∧ m = -39 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l1152_115228


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_expression_evaluation_at_3_l1152_115289

theorem expression_simplification_and_evaluation (x : ℝ) (h1 : x ≠ -2) (h2 : x ≠ 2) :
  (x^2 - 4*x + 4) / (x^2 - 4) / ((x - 2) / (x^2 + 2*x)) + 3 = x + 3 :=
by sorry

theorem expression_evaluation_at_3 :
  let x : ℝ := 3
  (x^2 - 4*x + 4) / (x^2 - 4) / ((x - 2) / (x^2 + 2*x)) + 3 = 6 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_expression_evaluation_at_3_l1152_115289


namespace NUMINAMATH_CALUDE_two_digit_numbers_with_specific_remainders_l1152_115201

theorem two_digit_numbers_with_specific_remainders :
  let S := {n : ℕ | 10 ≤ n ∧ n < 100 ∧ n % 4 = 3 ∧ n % 3 = 2}
  S = {11, 23, 35, 47, 59, 71, 83, 95} := by
  sorry

end NUMINAMATH_CALUDE_two_digit_numbers_with_specific_remainders_l1152_115201


namespace NUMINAMATH_CALUDE_concentric_circles_ratio_l1152_115262

theorem concentric_circles_ratio (r R : ℝ) (h1 : R = 10) 
  (h2 : π * R^2 = 2 * (π * R^2 - π * r^2)) : r = 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_concentric_circles_ratio_l1152_115262


namespace NUMINAMATH_CALUDE_flag_designs_count_l1152_115221

/-- The number of colors available for the flag design. -/
def num_colors : ℕ := 3

/-- The number of stripes on the flag. -/
def num_stripes : ℕ := 3

/-- Calculate the number of possible flag designs. -/
def num_flag_designs : ℕ := num_colors ^ num_stripes

/-- Theorem stating that the number of possible flag designs is 27. -/
theorem flag_designs_count : num_flag_designs = 27 := by
  sorry

end NUMINAMATH_CALUDE_flag_designs_count_l1152_115221


namespace NUMINAMATH_CALUDE_gcd_333_481_l1152_115292

theorem gcd_333_481 : Nat.gcd 333 481 = 37 := by
  sorry

end NUMINAMATH_CALUDE_gcd_333_481_l1152_115292


namespace NUMINAMATH_CALUDE_muffin_cost_l1152_115263

theorem muffin_cost (num_muffins : ℕ) (juice_cost total_cost : ℚ) : 
  num_muffins = 3 → 
  juice_cost = 29/20 → 
  total_cost = 37/10 → 
  (total_cost - juice_cost) / num_muffins = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_muffin_cost_l1152_115263


namespace NUMINAMATH_CALUDE_random_simulation_approximates_actual_probability_l1152_115243

/-- Random simulation method for estimating probabilities -/
def RandomSimulationMethod : Type := Unit

/-- Estimated probability from random simulation -/
def estimated_probability (method : RandomSimulationMethod) : ℝ := sorry

/-- Actual probability of the event -/
def actual_probability : ℝ := sorry

/-- Definition of approximation -/
def is_approximation (x y : ℝ) : Prop := sorry

theorem random_simulation_approximates_actual_probability 
  (method : RandomSimulationMethod) : 
  is_approximation (estimated_probability method) actual_probability := by
  sorry

end NUMINAMATH_CALUDE_random_simulation_approximates_actual_probability_l1152_115243


namespace NUMINAMATH_CALUDE_arrangement_probability_l1152_115280

/-- The probability of arranging n(n + 1)/2 distinct numbers into n rows,
    where the i-th row has i numbers, such that the largest number in each row
    is smaller than the largest number in all rows with more numbers. -/
def probability (n : ℕ) : ℚ :=
  (2 ^ n : ℚ) / (Nat.factorial (n + 1) : ℚ)

/-- Theorem stating that the probability of the described arrangement
    is equal to 2^n / (n+1)! -/
theorem arrangement_probability (n : ℕ) :
  probability n = (2 ^ n : ℚ) / (Nat.factorial (n + 1) : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_arrangement_probability_l1152_115280


namespace NUMINAMATH_CALUDE_quadratic_range_on_unit_interval_l1152_115252

/-- The range of a quadratic function on a closed interval --/
theorem quadratic_range_on_unit_interval
  (a b c : ℝ) (ha : a < 0) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c
  ∃ (min max : ℝ), min = c ∧ max = -b^2 / (4 * a) + c ∧
    Set.Icc min max = Set.Icc 0 1 ∩ Set.range f :=
by sorry

end NUMINAMATH_CALUDE_quadratic_range_on_unit_interval_l1152_115252


namespace NUMINAMATH_CALUDE_smallest_cube_box_volume_l1152_115222

def cone_height : ℝ := 20
def cone_base_diameter : ℝ := 18

theorem smallest_cube_box_volume (h : cone_height ≥ cone_base_diameter) :
  let box_side := max cone_height cone_base_diameter
  box_side ^ 3 = 8000 := by sorry

end NUMINAMATH_CALUDE_smallest_cube_box_volume_l1152_115222


namespace NUMINAMATH_CALUDE_cube_sum_from_sum_and_square_sum_l1152_115254

theorem cube_sum_from_sum_and_square_sum (x y : ℝ) 
  (h1 : x + y = 5) 
  (h2 : x^2 + y^2 = 13) : 
  x^3 + y^3 = 35 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_from_sum_and_square_sum_l1152_115254


namespace NUMINAMATH_CALUDE_pencil_purchase_count_l1152_115278

/-- Represents the number of pencils and pens purchased -/
structure Purchase where
  pencils : ℕ
  pens : ℕ

/-- Represents the cost in won -/
@[reducible] def Won := ℕ

theorem pencil_purchase_count (p : Purchase) 
  (h1 : p.pencils + p.pens = 12)
  (h2 : 1000 * p.pencils + 1300 * p.pens = 15000) :
  p.pencils = 2 := by
  sorry

#check pencil_purchase_count

end NUMINAMATH_CALUDE_pencil_purchase_count_l1152_115278


namespace NUMINAMATH_CALUDE_flag_width_calculation_l1152_115281

theorem flag_width_calculation (height : ℝ) (paint_cost : ℝ) (paint_coverage : ℝ) 
  (total_spent : ℝ) (h1 : height = 4) (h2 : paint_cost = 2) (h3 : paint_coverage = 4) 
  (h4 : total_spent = 20) : ∃ (width : ℝ), width = 5 := by
  sorry

end NUMINAMATH_CALUDE_flag_width_calculation_l1152_115281


namespace NUMINAMATH_CALUDE_express_y_in_terms_of_x_l1152_115240

/-- Given the equation 3x - y = 9, prove that y can be expressed as 3x - 9 -/
theorem express_y_in_terms_of_x (x y : ℝ) (h : 3 * x - y = 9) : y = 3 * x - 9 := by
  sorry

end NUMINAMATH_CALUDE_express_y_in_terms_of_x_l1152_115240


namespace NUMINAMATH_CALUDE_emily_calculation_l1152_115213

def round_to_nearest_ten (x : ℤ) : ℤ :=
  (x + 5) / 10 * 10

theorem emily_calculation : round_to_nearest_ten ((68 + 74 + 59) - 20) = 180 := by
  sorry

end NUMINAMATH_CALUDE_emily_calculation_l1152_115213


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l1152_115218

theorem arithmetic_calculations :
  (12 - (-18) + (-7) - 15 = 8) ∧
  (5 + 1 / 7 : ℚ) * (7 / 8 : ℚ) / (-8 / 9 : ℚ) / 3 = -27 / 16 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l1152_115218


namespace NUMINAMATH_CALUDE_fuchsia_survey_l1152_115251

/-- Given a survey about the color fuchsia with the following parameters:
  * total_surveyed: Total number of people surveyed
  * mostly_pink: Number of people who believe fuchsia is "mostly pink"
  * both: Number of people who believe fuchsia is both "mostly pink" and "mostly purple"
  * neither: Number of people who believe fuchsia is neither "mostly pink" nor "mostly purple"

  This theorem proves that the number of people who believe fuchsia is "mostly purple"
  is equal to total_surveyed - (mostly_pink - both) - neither.
-/
theorem fuchsia_survey (total_surveyed mostly_pink both neither : ℕ)
  (h1 : total_surveyed = 150)
  (h2 : mostly_pink = 80)
  (h3 : both = 40)
  (h4 : neither = 25) :
  total_surveyed - (mostly_pink - both) - neither = 85 := by
  sorry

#check fuchsia_survey

end NUMINAMATH_CALUDE_fuchsia_survey_l1152_115251


namespace NUMINAMATH_CALUDE_length_breadth_difference_l1152_115296

/-- A rectangular plot with specific properties -/
structure RectangularPlot where
  breadth : ℝ
  length : ℝ
  area_is_24_times_breadth : area = 24 * breadth
  breadth_is_14 : breadth = 14
  area_def : area = length * breadth

/-- The difference between length and breadth is 10 meters -/
theorem length_breadth_difference (plot : RectangularPlot) : 
  plot.length - plot.breadth = 10 := by
  sorry

end NUMINAMATH_CALUDE_length_breadth_difference_l1152_115296


namespace NUMINAMATH_CALUDE_trigonometric_problem_l1152_115220

theorem trigonometric_problem (α β : Real) 
  (h1 : 0 < α ∧ α < Real.pi / 2)
  (h2 : -Real.pi / 2 < β ∧ β < 0)
  (h3 : Real.cos (Real.pi / 4 + α) = 1 / 3)
  (h4 : Real.cos (Real.pi / 4 - β / 2) = Real.sqrt 3 / 3) :
  (Real.cos α = (Real.sqrt 2 + 4) / 6) ∧
  (Real.cos (α + β / 2) = 5 * Real.sqrt 3 / 9) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_problem_l1152_115220


namespace NUMINAMATH_CALUDE_joe_trip_theorem_l1152_115298

/-- Represents Joe's trip expenses and calculations -/
def joe_trip (exchange_rate : ℝ) (initial_savings flight hotel food transportation entertainment miscellaneous : ℝ) : Prop :=
  let total_savings_aud := initial_savings * exchange_rate
  let total_expenses_usd := flight + hotel + food + transportation + entertainment + miscellaneous
  let total_expenses_aud := total_expenses_usd * exchange_rate
  let amount_left := total_savings_aud - total_expenses_aud
  (total_expenses_aud = 9045) ∧ (amount_left = -945)

/-- Theorem stating the correctness of Joe's trip calculations -/
theorem joe_trip_theorem : 
  joe_trip 1.35 6000 1200 800 3000 500 850 350 := by
  sorry

end NUMINAMATH_CALUDE_joe_trip_theorem_l1152_115298


namespace NUMINAMATH_CALUDE_common_divisors_product_l1152_115238

theorem common_divisors_product (list : List Int) : 
  list = [48, 64, -18, 162, 144] →
  ∃ (a b c d e : Nat), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧
    (∀ x ∈ list, a ∣ x.natAbs) ∧
    (∀ x ∈ list, b ∣ x.natAbs) ∧
    (∀ x ∈ list, c ∣ x.natAbs) ∧
    (∀ x ∈ list, d ∣ x.natAbs) ∧
    (∀ x ∈ list, e ∣ x.natAbs) ∧
    a * b * c * d * e = 108 :=
by sorry

end NUMINAMATH_CALUDE_common_divisors_product_l1152_115238


namespace NUMINAMATH_CALUDE_jenny_calculation_l1152_115208

theorem jenny_calculation (x : ℤ) (h : x - 26 = -14) : x + 26 = 38 := by
  sorry

end NUMINAMATH_CALUDE_jenny_calculation_l1152_115208


namespace NUMINAMATH_CALUDE_bread_weight_equals_antons_weight_l1152_115277

/-- Prove that the weight of bread eaten by Vladimir equals Anton's weight before his birthday -/
theorem bread_weight_equals_antons_weight 
  (A : ℝ) -- Anton's weight
  (B : ℝ) -- Vladimir's weight before eating bread
  (F : ℝ) -- Fyodor's weight
  (X : ℝ) -- Weight of the bread
  (h1 : X + F = A + B) -- Condition 1: Bread and Fyodor weigh as much as Anton and Vladimir
  (h2 : B + X = A + F) -- Condition 2: Vladimir's weight after eating equals Anton and Fyodor
  : X = A := by
  sorry

end NUMINAMATH_CALUDE_bread_weight_equals_antons_weight_l1152_115277


namespace NUMINAMATH_CALUDE_min_abc_value_l1152_115294

/-- Given prime numbers a, b, c where a^5 divides (b^2 - c) and b + c is a perfect square,
    the minimum value of abc is 1958. -/
theorem min_abc_value (a b c : ℕ) : Prime a → Prime b → Prime c →
  (a^5 ∣ (b^2 - c)) → ∃ (n : ℕ), b + c = n^2 → (∀ x y z : ℕ, Prime x → Prime y → Prime z →
  (x^5 ∣ (y^2 - z)) → ∃ (m : ℕ), y + z = m^2 → x*y*z ≥ a*b*c) → a*b*c = 1958 := by
  sorry

end NUMINAMATH_CALUDE_min_abc_value_l1152_115294


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1152_115206

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

/-- The theorem statement -/
theorem geometric_sequence_property (a : ℕ → ℝ) :
  GeometricSequence a →
  a 3 * a 5 * a 7 * a 9 * a 11 = 243 →
  a 9 ^ 2 / a 11 = 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1152_115206


namespace NUMINAMATH_CALUDE_factorization_problem_1_factorization_problem_2_l1152_115241

-- Problem 1
theorem factorization_problem_1 (m n : ℝ) :
  2 * (m - n)^2 - m * (n - m) = (n - m) * (2 * n - 3 * m) := by
  sorry

-- Problem 2
theorem factorization_problem_2 (x y : ℝ) :
  -4 * x * y^2 + 4 * x^2 * y + y^3 = y * (2 * x - y)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_problem_1_factorization_problem_2_l1152_115241


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1152_115250

theorem quadratic_inequality (x : ℝ) : x^2 + x - 20 < 0 ↔ -5 < x ∧ x < 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1152_115250


namespace NUMINAMATH_CALUDE_even_sum_condition_l1152_115200

-- Define what it means for a number to be even
def IsEven (n : Int) : Prop := ∃ k : Int, n = 2 * k

-- Statement of the theorem
theorem even_sum_condition :
  (∀ a b : Int, IsEven a ∧ IsEven b → IsEven (a + b)) ∧
  (∃ a b : Int, IsEven (a + b) ∧ (¬IsEven a ∨ ¬IsEven b)) := by
  sorry

end NUMINAMATH_CALUDE_even_sum_condition_l1152_115200


namespace NUMINAMATH_CALUDE_complex_sum_problem_l1152_115285

theorem complex_sum_problem (a b c d e f : ℝ) : 
  d = 2 →
  e = -a - 2*c →
  (a + b*Complex.I) + (c + d*Complex.I) + (e + f*Complex.I) = -7*Complex.I →
  b + f = -9 := by
sorry

end NUMINAMATH_CALUDE_complex_sum_problem_l1152_115285


namespace NUMINAMATH_CALUDE_total_games_five_months_l1152_115284

def games_month1 : ℕ := 32
def games_month2 : ℕ := 24
def games_month3 : ℕ := 29
def games_month4 : ℕ := 19
def games_month5 : ℕ := 34

theorem total_games_five_months :
  games_month1 + games_month2 + games_month3 + games_month4 + games_month5 = 138 := by
  sorry

end NUMINAMATH_CALUDE_total_games_five_months_l1152_115284


namespace NUMINAMATH_CALUDE_rectangle_to_square_l1152_115256

/-- Represents a rectangle with given width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents a square with given side length -/
structure Square where
  side : ℝ

/-- Theorem: A 12 × 3 rectangle can be cut into three equal parts that form a 6 × 6 square -/
theorem rectangle_to_square (rect : Rectangle) (sq : Square) : 
  rect.width = 12 ∧ rect.height = 3 ∧ sq.side = 6 →
  ∃ (part_width part_height : ℝ),
    part_width * part_height = rect.width * rect.height / 3 ∧
    3 * part_width = sq.side ∧
    part_height = sq.side :=
by sorry

end NUMINAMATH_CALUDE_rectangle_to_square_l1152_115256


namespace NUMINAMATH_CALUDE_perfect_square_binomial_condition_l1152_115239

/-- A quadratic expression is a perfect square binomial if it can be written as (px + q)^2 for some real p and q -/
def IsPerfectSquareBinomial (f : ℝ → ℝ) : Prop :=
  ∃ p q : ℝ, ∀ x, f x = (p * x + q)^2

/-- Given that 9x^2 - 27x + a is a perfect square binomial, prove that a = 20.25 -/
theorem perfect_square_binomial_condition (a : ℝ) 
  (h : IsPerfectSquareBinomial (fun x ↦ 9*x^2 - 27*x + a)) : 
  a = 20.25 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_binomial_condition_l1152_115239


namespace NUMINAMATH_CALUDE_crickets_to_collect_l1152_115215

theorem crickets_to_collect (collected : ℕ) (target : ℕ) (additional : ℕ) : 
  collected = 7 → target = 11 → additional = target - collected :=
by
  sorry

end NUMINAMATH_CALUDE_crickets_to_collect_l1152_115215


namespace NUMINAMATH_CALUDE_division_of_fractions_l1152_115225

theorem division_of_fractions : (5 : ℚ) / 6 / (7 / 4) = 10 / 21 := by sorry

end NUMINAMATH_CALUDE_division_of_fractions_l1152_115225


namespace NUMINAMATH_CALUDE_slope_intercept_product_l1152_115290

theorem slope_intercept_product (m b : ℚ) 
  (h1 : m = 3/4)
  (h2 : b = -5/3)
  (h3 : m > 0)
  (h4 : b < 0) : 
  m * b < -1 := by
sorry

end NUMINAMATH_CALUDE_slope_intercept_product_l1152_115290


namespace NUMINAMATH_CALUDE_unique_integer_divisibility_l1152_115205

theorem unique_integer_divisibility : ∃! n : ℕ, n > 1 ∧
  ∀ p : ℕ, Prime p → (p ∣ (n^6 - 1) → p ∣ ((n^3 - 1) * (n^2 - 1))) ∧
  n = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_divisibility_l1152_115205


namespace NUMINAMATH_CALUDE_expression_factorization_l1152_115203

theorem expression_factorization (x : ℝ) :
  (4 * x^3 + 64 * x^2 - 8) - (-6 * x^3 + 2 * x^2 - 8) = 2 * x^2 * (5 * x + 31) :=
by sorry

end NUMINAMATH_CALUDE_expression_factorization_l1152_115203


namespace NUMINAMATH_CALUDE_candy_bar_fundraiser_l1152_115257

theorem candy_bar_fundraiser (cost_per_bar : ℝ) (avg_sold_per_member : ℝ) (total_earnings : ℝ)
  (h1 : cost_per_bar = 0.5)
  (h2 : avg_sold_per_member = 8)
  (h3 : total_earnings = 80) :
  (total_earnings / cost_per_bar) / avg_sold_per_member = 20 := by
  sorry

end NUMINAMATH_CALUDE_candy_bar_fundraiser_l1152_115257


namespace NUMINAMATH_CALUDE_inverse_proportionality_l1152_115288

/-- Proves that given α is inversely proportional to β, and α = 4 when β = 12, then α = -16 when β = -3 -/
theorem inverse_proportionality (α β : ℝ → ℝ) (k : ℝ) : 
  (∀ x, α x * β x = k) →  -- α is inversely proportional to β
  (α 12 = 4) →            -- α = 4 when β = 12
  (β 12 = 12) →           -- ensuring β 12 is indeed 12
  (β (-3) = -3) →         -- ensuring β (-3) is indeed -3
  (α (-3) = -16) :=       -- α = -16 when β = -3
by
  sorry


end NUMINAMATH_CALUDE_inverse_proportionality_l1152_115288


namespace NUMINAMATH_CALUDE_commission_percentage_is_21_875_l1152_115295

/-- Calculates the commission percentage for a sale with given rates and total amount -/
def commission_percentage (rate_below_500 : ℚ) (rate_above_500 : ℚ) (total_amount : ℚ) : ℚ :=
  let commission_below_500 := min total_amount 500 * rate_below_500
  let commission_above_500 := max (total_amount - 500) 0 * rate_above_500
  let total_commission := commission_below_500 + commission_above_500
  (total_commission / total_amount) * 100

/-- Theorem stating that the commission percentage for the given problem is 21.875% -/
theorem commission_percentage_is_21_875 :
  commission_percentage (20 / 100) (25 / 100) 800 = 21875 / 1000 := by sorry

end NUMINAMATH_CALUDE_commission_percentage_is_21_875_l1152_115295


namespace NUMINAMATH_CALUDE_subsets_containing_six_l1152_115210

def S : Finset ℕ := {1, 2, 3, 4, 5, 6}

theorem subsets_containing_six (A : Finset ℕ) (h : A ⊆ S) (h6 : 6 ∈ A) :
  (Finset.filter (fun A => 6 ∈ A) (Finset.powerset S)).card = 32 := by
  sorry

end NUMINAMATH_CALUDE_subsets_containing_six_l1152_115210


namespace NUMINAMATH_CALUDE_unique_intersection_point_l1152_115248

-- Define the three lines
def line1 (x y : ℝ) : Prop := 2 * y - 3 * x = 4
def line2 (x y : ℝ) : Prop := x + 4 * y = 1
def line3 (x y : ℝ) : Prop := 6 * x - 4 * y = 8

-- Define a point that lies on at least two of the lines
def intersection_point (p : ℝ × ℝ) : Prop :=
  (line1 p.1 p.2 ∧ line2 p.1 p.2) ∨
  (line1 p.1 p.2 ∧ line3 p.1 p.2) ∨
  (line2 p.1 p.2 ∧ line3 p.1 p.2)

-- Theorem stating that there is exactly one intersection point
theorem unique_intersection_point :
  ∃! p : ℝ × ℝ, intersection_point p :=
sorry

end NUMINAMATH_CALUDE_unique_intersection_point_l1152_115248


namespace NUMINAMATH_CALUDE_direct_proportional_function_inequality_l1152_115212

/-- A direct proportional function satisfying f[f(x)] ≥ x - 3 for all real x
    must be either f(x) = -x or f(x) = x -/
theorem direct_proportional_function_inequality 
  (f : ℝ → ℝ) 
  (h_prop : ∃ (a : ℝ), ∀ x, f x = a * x) 
  (h_ineq : ∀ x, f (f x) ≥ x - 3) :
  (∀ x, f x = -x) ∨ (∀ x, f x = x) := by
sorry

end NUMINAMATH_CALUDE_direct_proportional_function_inequality_l1152_115212


namespace NUMINAMATH_CALUDE_polynomial_simplification_l1152_115216

theorem polynomial_simplification (y : ℝ) : 
  (3*y - 2) * (5*y^11 + 3*y^10 + 5*y^9 + 3*y^8 + 5*y^7) = 
  15*y^12 - y^11 + 9*y^10 - y^9 + 9*y^8 - 10*y^7 := by
sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l1152_115216


namespace NUMINAMATH_CALUDE_max_students_distribution_l1152_115209

theorem max_students_distribution (pens toys books : ℕ) 
  (h_pens : pens = 451) 
  (h_toys : toys = 410) 
  (h_books : books = 325) : 
  (∃ (students : ℕ), students > 0 ∧ 
    pens % students = 0 ∧ 
    toys % students = 0 ∧ 
    books % students = 0) →
  (∀ (n : ℕ), n > 1 → 
    (pens % n ≠ 0 ∨ toys % n ≠ 0 ∨ books % n ≠ 0)) :=
by sorry

end NUMINAMATH_CALUDE_max_students_distribution_l1152_115209


namespace NUMINAMATH_CALUDE_circle_radius_values_l1152_115267

/-- Given a circle and its tangent line, prove the possible values of its radius -/
theorem circle_radius_values (r : ℝ) (k : ℝ) : 
  r > 0 → 
  (∀ x y, (x - 1)^2 + (y - 3 * Real.sqrt 3)^2 = r^2) →
  (∃ x y, y = k * x + Real.sqrt 3) →
  (k = Real.sqrt 3 ∨ k = -Real.sqrt 3) →
  (r = Real.sqrt 3 / 2 ∨ r = 3 * Real.sqrt 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_values_l1152_115267


namespace NUMINAMATH_CALUDE_complement_of_A_l1152_115274

def U : Set Nat := {1, 3, 5, 7, 9}
def A : Set Nat := {1, 5, 7}

theorem complement_of_A :
  (U \ A) = {3, 9} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l1152_115274


namespace NUMINAMATH_CALUDE_cube_volume_surface_area_l1152_115275

theorem cube_volume_surface_area (x : ℝ) : 
  (∃ (s : ℝ), s^3 = 8*x ∧ 6*s^2 = 2*x) → x = 1728 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_surface_area_l1152_115275


namespace NUMINAMATH_CALUDE_prime_divisibility_l1152_115242

theorem prime_divisibility (n : ℕ) (h1 : n ≥ 3) (h2 : Nat.Prime (4 * n + 1)) :
  (4 * n + 1) ∣ (n^(2*n) - 1) := by
  sorry

end NUMINAMATH_CALUDE_prime_divisibility_l1152_115242


namespace NUMINAMATH_CALUDE_race_distance_difference_l1152_115214

/-- In a race scenario where:
  * The race distance is 240 meters
  * Runner A finishes in 23 seconds
  * Runner A beats runner B by 7 seconds
This theorem proves that A beats B by 56 meters -/
theorem race_distance_difference (race_distance : ℝ) (a_time : ℝ) (time_difference : ℝ) :
  race_distance = 240 ∧ 
  a_time = 23 ∧ 
  time_difference = 7 →
  (race_distance - (race_distance / (a_time + time_difference)) * a_time) = 56 :=
by sorry

end NUMINAMATH_CALUDE_race_distance_difference_l1152_115214


namespace NUMINAMATH_CALUDE_invisibility_elixir_combinations_l1152_115219

/-- The number of valid combinations for the invisibility elixir. -/
def valid_combinations (roots : ℕ) (minerals : ℕ) (incompatible : ℕ) : ℕ :=
  roots * minerals - incompatible

/-- Theorem: Given 4 roots, 6 minerals, and 3 incompatible combinations,
    the number of valid combinations for the invisibility elixir is 21. -/
theorem invisibility_elixir_combinations :
  valid_combinations 4 6 3 = 21 := by
  sorry

end NUMINAMATH_CALUDE_invisibility_elixir_combinations_l1152_115219


namespace NUMINAMATH_CALUDE_line_param_values_l1152_115293

/-- The line equation y = (1/3)x + 3 parameterized as (x, y) = (-5, r) + t(m, -6) -/
def line_equation (x y : ℝ) : Prop := y = (1/3) * x + 3

/-- The parameterization of the line -/
def line_param (t r m : ℝ) (x y : ℝ) : Prop :=
  x = -5 + t * m ∧ y = r + t * (-6)

/-- Theorem stating that r = 4/3 and m = 0 for the given line and parameterization -/
theorem line_param_values :
  ∃ (r m : ℝ), (∀ (x y t : ℝ), line_equation x y ↔ line_param t r m x y) ∧ r = 4/3 ∧ m = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_param_values_l1152_115293


namespace NUMINAMATH_CALUDE_smallest_n_satisfying_conditions_l1152_115287

def is_multiple_of_75 (n : ℕ) : Prop := ∃ k : ℕ, n = 75 * k

def count_divisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

def satisfies_conditions (n : ℕ) : Prop :=
  is_multiple_of_75 n ∧ count_divisors n = 75

theorem smallest_n_satisfying_conditions :
  ∃! n : ℕ, satisfies_conditions n ∧ ∀ m : ℕ, satisfies_conditions m → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_satisfying_conditions_l1152_115287


namespace NUMINAMATH_CALUDE_ellipse_equation_l1152_115264

theorem ellipse_equation (a b : ℝ) (ha : a = 6) (hb : b = Real.sqrt 35) :
  (∃ x y : ℝ, x^2 / 36 + y^2 / 35 = 1) ∧ (∃ x y : ℝ, y^2 / 36 + x^2 / 35 = 1) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_equation_l1152_115264


namespace NUMINAMATH_CALUDE_complex_exponential_sum_l1152_115265

theorem complex_exponential_sum (γ δ : ℝ) :
  Complex.exp (Complex.I * γ) + Complex.exp (Complex.I * δ) = (2 / 5 : ℂ) + Complex.I * (1 / 2 : ℂ) →
  Complex.exp (-Complex.I * γ) + Complex.exp (-Complex.I * δ) = (2 / 5 : ℂ) - Complex.I * (1 / 2 : ℂ) :=
by sorry

end NUMINAMATH_CALUDE_complex_exponential_sum_l1152_115265


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1152_115268

def arithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmeticSequence a → a 3 + a 7 = 37 → a 2 + a 4 + a 6 + a 8 = 74 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1152_115268


namespace NUMINAMATH_CALUDE_oak_trees_after_five_days_l1152_115207

/-- Calculates the final number of oak trees in the park after 5 days -/
def final_oak_trees (initial : ℕ) (plant_rate_1 plant_rate_2 remove_rate_1 remove_rate_2 : ℕ) : ℕ :=
  let net_change_1 := (plant_rate_1 - remove_rate_1) * 2
  let net_change_2 := (plant_rate_2 - remove_rate_1)
  let net_change_3 := (plant_rate_2 - remove_rate_2) * 2
  initial + net_change_1 + net_change_2 + net_change_3

/-- Theorem stating that given the initial number of oak trees and planting/removal rates, 
    the final number of oak trees after 5 days will be 15 -/
theorem oak_trees_after_five_days :
  final_oak_trees 5 3 4 2 1 = 15 := by
  sorry

end NUMINAMATH_CALUDE_oak_trees_after_five_days_l1152_115207


namespace NUMINAMATH_CALUDE_equal_face_areas_not_imply_equal_volumes_l1152_115286

/-- A tetrahedron with its volume and face areas -/
structure Tetrahedron where
  volume : ℝ
  face_areas : Fin 4 → ℝ

/-- Two tetrahedrons have equal face areas -/
def equal_face_areas (t1 t2 : Tetrahedron) : Prop :=
  ∀ i : Fin 4, t1.face_areas i = t2.face_areas i

/-- Theorem stating that equal face areas do not imply equal volumes -/
theorem equal_face_areas_not_imply_equal_volumes :
  ∃ (t1 t2 : Tetrahedron), equal_face_areas t1 t2 ∧ t1.volume ≠ t2.volume :=
sorry

end NUMINAMATH_CALUDE_equal_face_areas_not_imply_equal_volumes_l1152_115286


namespace NUMINAMATH_CALUDE_nested_fraction_evaluation_l1152_115217

theorem nested_fraction_evaluation : 
  2 + 1 / (3 + 1 / (2 + 2)) = 30 / 13 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_evaluation_l1152_115217


namespace NUMINAMATH_CALUDE_geometric_sequence_from_formula_l1152_115236

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_from_formula (c q : ℝ) (hcq : c * q ≠ 0) :
  is_geometric_sequence (fun n => c * q ^ n) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_from_formula_l1152_115236


namespace NUMINAMATH_CALUDE_scooter_gain_percent_correct_l1152_115230

def scooter_gain_percent (purchase_price repair1 repair2 repair3 taxes maintenance selling_price : ℚ) : ℚ :=
  let total_cost := purchase_price + repair1 + repair2 + repair3 + taxes + maintenance
  let gain := selling_price - total_cost
  (gain / total_cost) * 100

theorem scooter_gain_percent_correct 
  (purchase_price repair1 repair2 repair3 taxes maintenance selling_price : ℚ) :
  scooter_gain_percent purchase_price repair1 repair2 repair3 taxes maintenance selling_price =
  ((selling_price - (purchase_price + repair1 + repair2 + repair3 + taxes + maintenance)) / 
   (purchase_price + repair1 + repair2 + repair3 + taxes + maintenance)) * 100 :=
by sorry

end NUMINAMATH_CALUDE_scooter_gain_percent_correct_l1152_115230


namespace NUMINAMATH_CALUDE_midpoint_octagon_area_ratio_l1152_115273

/-- A regular octagon -/
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ
  is_regular : Sorry

/-- The octagon formed by joining the midpoints of the sides of a regular octagon -/
def midpoint_octagon (oct : RegularOctagon) : RegularOctagon :=
  sorry

/-- The area of a regular octagon -/
def area (oct : RegularOctagon) : ℝ :=
  sorry

/-- Theorem: The area of the octagon formed by joining the midpoints of the sides
    of a regular octagon is 1/2 of the area of the original octagon -/
theorem midpoint_octagon_area_ratio (oct : RegularOctagon) :
  area (midpoint_octagon oct) = (1/2 : ℝ) * area oct :=
sorry

end NUMINAMATH_CALUDE_midpoint_octagon_area_ratio_l1152_115273


namespace NUMINAMATH_CALUDE_inequality_proof_l1152_115233

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_condition : x * y + y * z + z * x = x + y + z) : 
  1 / (x^2 + y + 1) + 1 / (y^2 + z + 1) + 1 / (z^2 + x + 1) ≤ 1 ∧ 
  (1 / (x^2 + y + 1) + 1 / (y^2 + z + 1) + 1 / (z^2 + x + 1) = 1 ↔ x = 1 ∧ y = 1 ∧ z = 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1152_115233


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1152_115279

theorem imaginary_part_of_z (z : ℂ) (h : (3 + 4*I)*z = Complex.abs (4 - 3*I)) : 
  z.im = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1152_115279


namespace NUMINAMATH_CALUDE_potato_ratio_l1152_115283

theorem potato_ratio (total_potatoes : ℕ) (num_people : ℕ) (potatoes_per_person : ℕ) 
  (h1 : total_potatoes = 24)
  (h2 : num_people = 3)
  (h3 : potatoes_per_person = 8)
  (h4 : total_potatoes = num_people * potatoes_per_person) :
  ∃ (r : ℕ), r > 0 ∧ 
    (potatoes_per_person, potatoes_per_person, potatoes_per_person) = (r, r, r) := by
  sorry

end NUMINAMATH_CALUDE_potato_ratio_l1152_115283


namespace NUMINAMATH_CALUDE_pi_is_infinite_decimal_l1152_115272

-- Define the property of being an infinite decimal
def IsInfiniteDecimal (x : ℝ) : Prop := sorry

-- Define the property of being an irrational number
def IsIrrational (x : ℝ) : Prop := sorry

-- State the theorem
theorem pi_is_infinite_decimal :
  (∀ x : ℝ, IsIrrational x → IsInfiniteDecimal x) →  -- Condition: Irrational numbers are infinite decimals
  IsIrrational Real.pi →                             -- Condition: π is an irrational number
  IsInfiniteDecimal Real.pi :=                       -- Conclusion: π is an infinite decimal
by sorry

end NUMINAMATH_CALUDE_pi_is_infinite_decimal_l1152_115272


namespace NUMINAMATH_CALUDE_cubic_geometric_roots_l1152_115237

theorem cubic_geometric_roots (a b c : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    x^3 + a*x^2 + b*x + c = 0 ∧
    y^3 + a*y^2 + b*y + c = 0 ∧
    z^3 + a*z^2 + b*z + c = 0 ∧
    ∃ q : ℝ, q ≠ 0 ∧ q ≠ 1 ∧ q ≠ -1 ∧ y = x*q ∧ z = x*q^2) ↔
  (b^3 = a^3*c ∧
   c ≠ 0 ∧
   ∃ m : ℝ, m^3 = -c ∧ a < m ∧ m < -a/3) :=
by sorry

end NUMINAMATH_CALUDE_cubic_geometric_roots_l1152_115237


namespace NUMINAMATH_CALUDE_distance_focus_to_asymptote_l1152_115249

-- Define the hyperbola C
def C (x y : ℝ) : Prop := x^2 - y^2 = 2

-- Define the focus of the hyperbola
def focus : ℝ × ℝ := (2, 0)

-- Define the asymptote of the hyperbola
def asymptote (x y : ℝ) : Prop := y + x = 0

-- Theorem stating the distance from focus to asymptote
theorem distance_focus_to_asymptote :
  ∃ (d : ℝ), d = Real.sqrt 2 ∧
  ∀ (x y : ℝ), C x y → asymptote x y →
  Real.sqrt ((x - focus.1)^2 + (y - focus.2)^2) ≥ d :=
sorry

end NUMINAMATH_CALUDE_distance_focus_to_asymptote_l1152_115249


namespace NUMINAMATH_CALUDE_negation_of_implication_l1152_115270

theorem negation_of_implication (x : ℝ) :
  ¬(x > 2015 → x > 0) ↔ (x ≤ 2015 → x ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l1152_115270


namespace NUMINAMATH_CALUDE_product_of_solutions_l1152_115202

theorem product_of_solutions (x : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
   (abs (5 * x₁) + 4 = abs (40 - 5)) ∧ 
   (abs (5 * x₂) + 4 = abs (40 - 5)) ∧
   x₁ * x₂ = -961 / 25) :=
by sorry

end NUMINAMATH_CALUDE_product_of_solutions_l1152_115202


namespace NUMINAMATH_CALUDE_sound_engineer_selection_probability_l1152_115224

theorem sound_engineer_selection_probability :
  let total_candidates : ℕ := 5
  let selected_engineers : ℕ := 3
  let specific_engineers : ℕ := 2

  let total_combinations := Nat.choose total_candidates selected_engineers
  let favorable_outcomes := 
    Nat.choose specific_engineers 1 * Nat.choose (total_candidates - specific_engineers) (selected_engineers - 1) +
    Nat.choose specific_engineers 2 * Nat.choose (total_candidates - specific_engineers) (selected_engineers - 2)

  (favorable_outcomes : ℚ) / total_combinations = 9 / 10 :=
by
  sorry

end NUMINAMATH_CALUDE_sound_engineer_selection_probability_l1152_115224
