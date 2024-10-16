import Mathlib

namespace NUMINAMATH_CALUDE_max_area_equilateral_triangle_in_rectangle_max_area_equilateral_triangle_proof_l1616_161665

/-- The maximum area of an equilateral triangle inscribed in a 10x11 rectangle -/
theorem max_area_equilateral_triangle_in_rectangle : ℝ :=
  let rectangle_width := 10
  let rectangle_height := 11
  let max_area := 221 * Real.sqrt 3 - 330
  max_area

/-- Proof that the maximum area of an equilateral triangle inscribed in a 10x11 rectangle is 221√3 - 330 -/
theorem max_area_equilateral_triangle_proof : 
  max_area_equilateral_triangle_in_rectangle = 221 * Real.sqrt 3 - 330 := by
  sorry

end NUMINAMATH_CALUDE_max_area_equilateral_triangle_in_rectangle_max_area_equilateral_triangle_proof_l1616_161665


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1616_161620

def A : Set ℕ := {1, 2, 9}
def B : Set ℕ := {1, 7}

theorem intersection_of_A_and_B : A ∩ B = {1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1616_161620


namespace NUMINAMATH_CALUDE_elder_person_age_l1616_161603

/-- Proves that given two persons whose ages differ by 16 years, and 6 years ago the elder one was 3 times as old as the younger one, the present age of the elder person is 30 years. -/
theorem elder_person_age (y e : ℕ) : 
  e = y + 16 → 
  e - 6 = 3 * (y - 6) → 
  e = 30 :=
by sorry

end NUMINAMATH_CALUDE_elder_person_age_l1616_161603


namespace NUMINAMATH_CALUDE_no_solution_implies_m_equals_two_l1616_161606

theorem no_solution_implies_m_equals_two :
  (∀ x : ℝ, (2 - m) / (1 - x) ≠ 1) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_implies_m_equals_two_l1616_161606


namespace NUMINAMATH_CALUDE_mod_product_equivalence_l1616_161683

theorem mod_product_equivalence : ∃ m : ℕ, 
  198 * 955 ≡ m [ZMOD 50] ∧ 0 ≤ m ∧ m < 50 ∧ m = 40 := by
  sorry

end NUMINAMATH_CALUDE_mod_product_equivalence_l1616_161683


namespace NUMINAMATH_CALUDE_books_added_by_marta_l1616_161641

def initial_books : ℕ := 38
def final_books : ℕ := 48

theorem books_added_by_marta : 
  final_books - initial_books = 10 := by sorry

end NUMINAMATH_CALUDE_books_added_by_marta_l1616_161641


namespace NUMINAMATH_CALUDE_unique_two_digit_number_l1616_161695

/-- A two-digit number satisfying specific conditions -/
def TwoDigitNumber (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧
  (n % 10 = n / 10 + 2) ∧
  (n * (n / 10 + n % 10) = 144)

/-- Theorem stating that 24 is the only two-digit number satisfying the given conditions -/
theorem unique_two_digit_number : ∃! n : ℕ, TwoDigitNumber n ∧ n = 24 := by
  sorry

end NUMINAMATH_CALUDE_unique_two_digit_number_l1616_161695


namespace NUMINAMATH_CALUDE_enthalpy_relationship_l1616_161633

/-- Represents the enthalpy change of a chemical reaction -/
structure EnthalpyChange where
  value : ℝ
  units : String

/-- Represents a chemical reaction with its enthalpy change -/
structure ChemicalReaction where
  equation : String
  enthalpyChange : EnthalpyChange

/-- Given chemical reactions and their enthalpy changes, prove that 2a = b < 0 -/
theorem enthalpy_relationship (
  reaction1 reaction2 reaction3 reaction4 : ChemicalReaction
) (h1 : reaction1.equation = "H₂(g) + ½O₂(g) → H₂O(g)")
  (h2 : reaction2.equation = "2H₂(g) + O₂(g) → 2H₂O(g)")
  (h3 : reaction3.equation = "H₂(g) + ½O₂(g) → H₂O(l)")
  (h4 : reaction4.equation = "2H₂(g) + O₂(g) → 2H₂O(l)")
  (h5 : reaction1.enthalpyChange.units = "KJ·mol⁻¹")
  (h6 : reaction2.enthalpyChange.units = "KJ·mol⁻¹")
  (h7 : reaction3.enthalpyChange.units = "KJ·mol⁻¹")
  (h8 : reaction4.enthalpyChange.units = "KJ·mol⁻¹")
  (h9 : reaction1.enthalpyChange.value = reaction3.enthalpyChange.value)
  (h10 : reaction2.enthalpyChange.value = reaction4.enthalpyChange.value) :
  2 * reaction1.enthalpyChange.value = reaction2.enthalpyChange.value ∧ 
  reaction2.enthalpyChange.value < 0 := by
  sorry


end NUMINAMATH_CALUDE_enthalpy_relationship_l1616_161633


namespace NUMINAMATH_CALUDE_complex_modulus_example_l1616_161652

theorem complex_modulus_example : 
  let z : ℂ := 1 - 2*I
  Complex.abs z = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_complex_modulus_example_l1616_161652


namespace NUMINAMATH_CALUDE_d_value_when_x_plus_3_is_factor_l1616_161624

/-- The polynomial Q(x) with parameter d -/
def Q (d : ℝ) (x : ℝ) : ℝ := x^3 - 3*x^2 + d*x - 27

/-- Theorem stating that d = -27 when x+3 is a factor of Q(x) -/
theorem d_value_when_x_plus_3_is_factor :
  ∃ d : ℝ, (∀ x : ℝ, Q d x = 0 ↔ x = -3) → d = -27 := by
  sorry

end NUMINAMATH_CALUDE_d_value_when_x_plus_3_is_factor_l1616_161624


namespace NUMINAMATH_CALUDE_yogurt_production_cost_l1616_161691

/-- The cost of producing three batches of yogurt given the following conditions:
  - Milk costs $1.5 per liter
  - Fruit costs $2 per kilogram
  - One batch of yogurt requires 10 liters of milk and 3 kilograms of fruit
-/
theorem yogurt_production_cost :
  let milk_cost_per_liter : ℚ := 3/2
  let fruit_cost_per_kg : ℚ := 2
  let milk_per_batch : ℚ := 10
  let fruit_per_batch : ℚ := 3
  let num_batches : ℕ := 3
  (milk_cost_per_liter * milk_per_batch + fruit_cost_per_kg * fruit_per_batch) * num_batches = 63 := by
  sorry

end NUMINAMATH_CALUDE_yogurt_production_cost_l1616_161691


namespace NUMINAMATH_CALUDE_exam_mean_score_l1616_161659

theorem exam_mean_score (a b : ℕ) (mean_a mean_b : ℝ) :
  a > 0 ∧ b > 0 →
  mean_a = 90 →
  mean_b = 78 →
  a = (5 : ℝ) / 7 * b →
  ∃ (max_score_a : ℝ), max_score_a = 100 ∧ max_score_a ≥ mean_b + 20 →
  (mean_a * a + mean_b * b) / (a + b) = 83 :=
by sorry

end NUMINAMATH_CALUDE_exam_mean_score_l1616_161659


namespace NUMINAMATH_CALUDE_square_39_relation_l1616_161690

theorem square_39_relation : (39 : ℕ)^2 = (40 : ℕ)^2 - 79 := by
  sorry

end NUMINAMATH_CALUDE_square_39_relation_l1616_161690


namespace NUMINAMATH_CALUDE_leahs_coins_value_l1616_161658

/-- Represents the number of coins of each type --/
structure CoinCount where
  pennies : ℕ
  nickels : ℕ
  dimes : ℕ

/-- Calculates the total value of coins in cents --/
def totalValue (coins : CoinCount) : ℕ :=
  coins.pennies + 5 * coins.nickels + 10 * coins.dimes

/-- Theorem stating that Leah's coins are worth 88 cents --/
theorem leahs_coins_value :
  ∃ (coins : CoinCount),
    coins.pennies + coins.nickels + coins.dimes = 20 ∧
    coins.pennies = coins.nickels ∧
    coins.pennies = coins.dimes + 4 ∧
    totalValue coins = 88 := by
  sorry

#check leahs_coins_value

end NUMINAMATH_CALUDE_leahs_coins_value_l1616_161658


namespace NUMINAMATH_CALUDE_permutations_count_l1616_161697

/-- The total number of permutations of the string "HMMTHMMT" -/
def total_permutations : ℕ := 420

/-- The number of permutations containing the substring "HMMT" -/
def permutations_with_substring : ℕ := 60

/-- The number of cases over-counted -/
def over_counted_cases : ℕ := 1

/-- The number of permutations without the consecutive substring "HMMT" -/
def permutations_without_substring : ℕ := total_permutations - permutations_with_substring + over_counted_cases

theorem permutations_count : permutations_without_substring = 361 := by
  sorry

end NUMINAMATH_CALUDE_permutations_count_l1616_161697


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1616_161686

theorem complex_fraction_simplification :
  (3 - 2 * Complex.I) / (1 + 4 * Complex.I) = -5/17 - 14/17 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1616_161686


namespace NUMINAMATH_CALUDE_largest_multiple_with_negation_constraint_l1616_161687

theorem largest_multiple_with_negation_constraint : 
  ∀ n : ℤ, n % 12 = 0 ∧ -n > -150 → n ≤ 144 := by sorry

end NUMINAMATH_CALUDE_largest_multiple_with_negation_constraint_l1616_161687


namespace NUMINAMATH_CALUDE_marcus_percentage_of_team_points_l1616_161635

def marcus_three_pointers : ℕ := 5
def marcus_two_pointers : ℕ := 10
def team_total_points : ℕ := 70

def marcus_points : ℕ := marcus_three_pointers * 3 + marcus_two_pointers * 2

theorem marcus_percentage_of_team_points :
  (marcus_points : ℚ) / team_total_points * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_marcus_percentage_of_team_points_l1616_161635


namespace NUMINAMATH_CALUDE_sin_30_degrees_l1616_161673

/-- Sine of 30 degrees is equal to 1/2 -/
theorem sin_30_degrees : Real.sin (π / 6) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_30_degrees_l1616_161673


namespace NUMINAMATH_CALUDE_opposite_of_negative_2023_l1616_161638

theorem opposite_of_negative_2023 : -((-2023 : ℤ)) = 2023 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_2023_l1616_161638


namespace NUMINAMATH_CALUDE_doctor_appointment_distance_l1616_161694

/-- Represents the distances Tony needs to drive for his errands -/
structure ErrandDistances where
  groceries : ℕ
  haircut : ℕ
  doctor : ℕ

/-- Calculates the total distance for all errands -/
def totalDistance (d : ErrandDistances) : ℕ :=
  d.groceries + d.haircut + d.doctor

theorem doctor_appointment_distance :
  ∀ (d : ErrandDistances),
    d.groceries = 10 →
    d.haircut = 15 →
    totalDistance d / 2 = 15 →
    d.doctor = 5 := by
  sorry

end NUMINAMATH_CALUDE_doctor_appointment_distance_l1616_161694


namespace NUMINAMATH_CALUDE_shelbys_journey_l1616_161616

/-- Shelby's scooter journey with varying weather conditions -/
theorem shelbys_journey 
  (speed_sunny : ℝ) 
  (speed_rainy : ℝ) 
  (total_distance : ℝ) 
  (total_time : ℝ) 
  (break_time : ℝ)
  (h1 : speed_sunny = 40)
  (h2 : speed_rainy = 15)
  (h3 : total_distance = 20)
  (h4 : total_time = 50)
  (h5 : break_time = 5) :
  ∃ (rainy_time : ℝ),
    rainy_time = 24 ∧
    speed_sunny * (total_time - rainy_time - break_time) / 60 + 
    speed_rainy * rainy_time / 60 = total_distance :=
by sorry

end NUMINAMATH_CALUDE_shelbys_journey_l1616_161616


namespace NUMINAMATH_CALUDE_wage_restoration_l1616_161619

theorem wage_restoration (original_wage : ℝ) (h : original_wage > 0) :
  let reduced_wage := 0.7 * original_wage
  let raise_percentage := 100 * (1 / 0.7 - 1)
  reduced_wage * (1 + raise_percentage / 100) = original_wage := by
sorry

end NUMINAMATH_CALUDE_wage_restoration_l1616_161619


namespace NUMINAMATH_CALUDE_floor_painting_rate_l1616_161642

/-- Proves that the painting rate for a rectangular floor is 3 Rs/m² given specific conditions --/
theorem floor_painting_rate (length breadth area cost : ℝ) : 
  length = 3 * breadth →
  length = 15.491933384829668 →
  area = length * breadth →
  cost = 240 →
  cost / area = 3 := by
  sorry

end NUMINAMATH_CALUDE_floor_painting_rate_l1616_161642


namespace NUMINAMATH_CALUDE_expression_factorization_l1616_161692

theorem expression_factorization (b : ℝ) : 
  (4 * b^3 + 126 * b^2 - 9) - (-9 * b^3 + 2 * b^2 - 9) = b^2 * (13 * b + 124) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l1616_161692


namespace NUMINAMATH_CALUDE_triple_sum_power_divisibility_l1616_161668

theorem triple_sum_power_divisibility (a b c : ℤ) (h : a + b + c = 0) :
  ∃ k : ℤ, a^1999 + b^1999 + c^1999 = 6 * k :=
by sorry

end NUMINAMATH_CALUDE_triple_sum_power_divisibility_l1616_161668


namespace NUMINAMATH_CALUDE_square_difference_2019_l1616_161608

theorem square_difference_2019 (a b : ℕ) (h1 : 2019 = a^2 - b^2) (h2 : a < 1000) : a = 338 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_2019_l1616_161608


namespace NUMINAMATH_CALUDE_problem_statement_l1616_161643

theorem problem_statement (s x y : ℝ) 
  (h1 : s > 0) 
  (h2 : x^2 + y^2 ≠ 0) 
  (h3 : x*s^2 < y*s^2) : 
  ¬(-x^2 < -y^2) ∧ ¬(-x^2 < y^2) ∧ ¬(x^2 < -y^2) ∧ ¬(x^2 > y^2) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1616_161643


namespace NUMINAMATH_CALUDE_cos_identity_l1616_161614

theorem cos_identity (α : ℝ) (h : Real.cos (π / 6 - α) = 3 / 5) :
  Real.cos (5 * π / 6 + α) = -(3 / 5) := by
  sorry

end NUMINAMATH_CALUDE_cos_identity_l1616_161614


namespace NUMINAMATH_CALUDE_c_investment_is_half_l1616_161611

/-- Represents the investment of a partner in a partnership --/
structure Investment where
  capital : ℚ  -- Fraction of total capital invested
  time : ℚ     -- Fraction of total time invested

/-- Represents a partnership with three investors --/
structure Partnership where
  a : Investment
  b : Investment
  c : Investment
  total_profit : ℚ
  a_share : ℚ

/-- The theorem stating that given the conditions of the problem, C's investment is 1/2 of the total capital --/
theorem c_investment_is_half (p : Partnership) : 
  p.a = ⟨1/6, 1/6⟩ → 
  p.b = ⟨1/3, 1/3⟩ → 
  p.c.time = 1 →
  p.total_profit = 2300 →
  p.a_share = 100 →
  p.c.capital = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_c_investment_is_half_l1616_161611


namespace NUMINAMATH_CALUDE_bachuan_jiaoqing_extrema_l1616_161689

/-- Definition of a "Bachuan Jiaoqing password number" -/
def is_bachuan_jiaoqing (n : ℕ) : Prop :=
  let a := n / 1000
  let b := (n / 100) % 10
  let c := (n / 10) % 10
  let d := n % 10
  1000 ≤ n ∧ n < 10000 ∧ b ≥ c ∧ a = b + c ∧ d = b - c

/-- Additional divisibility condition -/
def satisfies_divisibility (n : ℕ) : Prop :=
  let a := n / 1000
  let bcd := n % 1000
  (bcd - 7 * a) % 13 = 0

/-- Theorem stating the largest and smallest "Bachuan Jiaoqing password numbers" -/
theorem bachuan_jiaoqing_extrema :
  (∀ n, is_bachuan_jiaoqing n → n ≤ 9909) ∧
  (∃ n, is_bachuan_jiaoqing n ∧ satisfies_divisibility n ∧
    ∀ m, is_bachuan_jiaoqing m ∧ satisfies_divisibility m → n ≤ m) ∧
  (is_bachuan_jiaoqing 9909) ∧
  (is_bachuan_jiaoqing 5321 ∧ satisfies_divisibility 5321) := by
  sorry

end NUMINAMATH_CALUDE_bachuan_jiaoqing_extrema_l1616_161689


namespace NUMINAMATH_CALUDE_solution_eq1_solution_eq2_l1616_161670

-- Define the average method for quadratic equations
def average_method (a b c : ℝ) : Set ℝ :=
  let avg := (a + b) / 2
  let diff := b - avg
  {x | (x + avg)^2 - diff^2 = c}

-- Theorem for the first equation
theorem solution_eq1 : 
  average_method 2 8 40 = {2, -12} := by sorry

-- Theorem for the second equation
theorem solution_eq2 : 
  average_method (-2) 6 4 = {-2 + 2 * Real.sqrt 5, -2 - 2 * Real.sqrt 5} := by sorry

end NUMINAMATH_CALUDE_solution_eq1_solution_eq2_l1616_161670


namespace NUMINAMATH_CALUDE_fixed_point_coordinates_l1616_161649

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The line equation y - 2 = k(x + 1) -/
def lineEquation (k : ℝ) (p : Point) : Prop :=
  p.y - 2 = k * (p.x + 1)

/-- The theorem statement -/
theorem fixed_point_coordinates :
  (∃ M : Point, ∀ k : ℝ, lineEquation k M) →
  ∃ M : Point, M.x = -1 ∧ M.y = 2 :=
by sorry

end NUMINAMATH_CALUDE_fixed_point_coordinates_l1616_161649


namespace NUMINAMATH_CALUDE_intersection_M_N_l1616_161634

def M : Set ℝ := {x | x^2 ≤ 1}
def N : Set ℝ := {-2, 0, 1}

theorem intersection_M_N : M ∩ N = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1616_161634


namespace NUMINAMATH_CALUDE_flower_shop_problem_l1616_161647

/-- The number of flowers brought at dawn -/
def flowers_at_dawn : ℕ := 300

/-- The fraction of flowers sold in the morning -/
def morning_sale_fraction : ℚ := 3/5

/-- The total number of flowers sold in the afternoon -/
def afternoon_sales : ℕ := 180

theorem flower_shop_problem :
  (flowers_at_dawn : ℚ) * morning_sale_fraction = afternoon_sales ∧
  (flowers_at_dawn : ℚ) * morning_sale_fraction = (flowers_at_dawn : ℚ) * (1 - morning_sale_fraction) + (afternoon_sales - (flowers_at_dawn : ℚ) * (1 - morning_sale_fraction)) :=
by sorry

end NUMINAMATH_CALUDE_flower_shop_problem_l1616_161647


namespace NUMINAMATH_CALUDE_smallest_lcm_with_gcd_3_l1616_161672

theorem smallest_lcm_with_gcd_3 (k l : ℕ) : 
  k ≥ 1000 ∧ k ≤ 9999 ∧ l ≥ 1000 ∧ l ≤ 9999 ∧ Nat.gcd k l = 3 →
  Nat.lcm k l ≥ 335670 ∧ ∃ (k₀ l₀ : ℕ), k₀ ≥ 1000 ∧ k₀ ≤ 9999 ∧ l₀ ≥ 1000 ∧ l₀ ≤ 9999 ∧ 
  Nat.gcd k₀ l₀ = 3 ∧ Nat.lcm k₀ l₀ = 335670 :=
by sorry

end NUMINAMATH_CALUDE_smallest_lcm_with_gcd_3_l1616_161672


namespace NUMINAMATH_CALUDE_power_functions_inequality_l1616_161618

theorem power_functions_inequality (x₁ x₂ : ℝ) (h : 0 < x₁ ∧ x₁ < x₂) :
  (((x₁ + x₂) / 2) ^ 2 < (x₁^2 + x₂^2) / 2) ∧
  (2 / (x₁ + x₂) < (1 / x₁ + 1 / x₂) / 2) := by
  sorry

end NUMINAMATH_CALUDE_power_functions_inequality_l1616_161618


namespace NUMINAMATH_CALUDE_not_parabola_l1616_161621

/-- A conic section represented by the equation x^2 + ky^2 = 1 -/
structure ConicSection (k : ℝ) where
  x : ℝ
  y : ℝ
  eq : x^2 + k * y^2 = 1

/-- Definition of a parabola -/
def IsParabola (c : ConicSection k) : Prop :=
  ∃ (a b h : ℝ), h ≠ 0 ∧ (c.x - a)^2 = 4 * h * (c.y - b)

/-- Theorem: For any real k, the equation x^2 + ky^2 = 1 cannot represent a parabola -/
theorem not_parabola (k : ℝ) : ¬∃ (c : ConicSection k), IsParabola c := by
  sorry

end NUMINAMATH_CALUDE_not_parabola_l1616_161621


namespace NUMINAMATH_CALUDE_stock_price_increase_l1616_161679

theorem stock_price_increase (P : ℝ) (X : ℝ) : 
  P * (1 + X / 100) * 0.75 * 1.35 = P * 1.215 → X = 20 := by
  sorry

end NUMINAMATH_CALUDE_stock_price_increase_l1616_161679


namespace NUMINAMATH_CALUDE_factorization_problems_l1616_161612

theorem factorization_problems :
  (∀ m : ℝ, m * (m - 3) + 3 * (3 - m) = (m - 3)^2) ∧
  (∀ x : ℝ, 4 * x^3 - 12 * x^2 + 9 * x = x * (2 * x - 3)^2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_problems_l1616_161612


namespace NUMINAMATH_CALUDE_inequality_proof_l1616_161636

theorem inequality_proof (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) 
  (h_abc : a * b * c = 9 / 4) : 
  a^3 + b^3 + c^3 > a * Real.sqrt (b + c) + b * Real.sqrt (c + a) + c * Real.sqrt (a + b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1616_161636


namespace NUMINAMATH_CALUDE_nina_bought_two_card_packs_l1616_161628

def num_toys : ℕ := 3
def toy_price : ℕ := 10
def num_shirts : ℕ := 5
def shirt_price : ℕ := 6
def card_pack_price : ℕ := 5
def total_spent : ℕ := 70

theorem nina_bought_two_card_packs :
  (num_toys * toy_price + num_shirts * shirt_price + 2 * card_pack_price = total_spent) := by
  sorry

end NUMINAMATH_CALUDE_nina_bought_two_card_packs_l1616_161628


namespace NUMINAMATH_CALUDE_first_day_distance_l1616_161645

/-- Proves the distance covered on the first day of a three-day hike -/
theorem first_day_distance (total_distance : ℝ) (second_day : ℝ) (third_day : ℝ)
  (h1 : total_distance = 50)
  (h2 : second_day = total_distance / 2)
  (h3 : third_day = 15)
  : total_distance - second_day - third_day = 10 := by
  sorry

end NUMINAMATH_CALUDE_first_day_distance_l1616_161645


namespace NUMINAMATH_CALUDE_equal_cost_guests_l1616_161681

def caesars_cost (guests : ℕ) : ℚ := 800 + 30 * guests
def venus_cost (guests : ℕ) : ℚ := 500 + 35 * guests

theorem equal_cost_guests : ∃ (x : ℕ), caesars_cost x = venus_cost x ∧ x = 60 := by
  sorry

end NUMINAMATH_CALUDE_equal_cost_guests_l1616_161681


namespace NUMINAMATH_CALUDE_unique_solution_absolute_value_equation_l1616_161696

theorem unique_solution_absolute_value_equation :
  ∃! x : ℝ, |x - 2| = |x - 1| + |x - 4| :=
by
  -- The unique solution is x = 4
  use 4
  constructor
  · -- Prove that x = 4 satisfies the equation
    sorry
  · -- Prove that any solution must equal 4
    sorry

#check unique_solution_absolute_value_equation

end NUMINAMATH_CALUDE_unique_solution_absolute_value_equation_l1616_161696


namespace NUMINAMATH_CALUDE_delores_remaining_money_l1616_161623

/-- Calculates the remaining money after purchasing a computer and printer -/
def remaining_money (initial_amount computer_cost printer_cost : ℕ) : ℕ :=
  initial_amount - (computer_cost + printer_cost)

/-- Theorem: Given the specific amounts, the remaining money is $10 -/
theorem delores_remaining_money :
  remaining_money 450 400 40 = 10 := by
  sorry

end NUMINAMATH_CALUDE_delores_remaining_money_l1616_161623


namespace NUMINAMATH_CALUDE_angle_in_first_or_third_quadrant_l1616_161651

def is_acute (α : Real) : Prop := 0 < α ∧ α < Real.pi / 2

def in_first_quadrant (θ : Real) : Prop :=
  0 ≤ θ % (2 * Real.pi) ∧ θ % (2 * Real.pi) < Real.pi / 2

def in_third_quadrant (θ : Real) : Prop :=
  Real.pi < θ % (2 * Real.pi) ∧ θ % (2 * Real.pi) < 3 * Real.pi / 2

theorem angle_in_first_or_third_quadrant (α : Real) (k : Int) 
  (h_acute : is_acute α) :
  in_first_quadrant (k * Real.pi + α) ∨ in_third_quadrant (k * Real.pi + α) := by
  sorry

end NUMINAMATH_CALUDE_angle_in_first_or_third_quadrant_l1616_161651


namespace NUMINAMATH_CALUDE_max_temperature_range_l1616_161646

theorem max_temperature_range (temps : Finset ℝ) (avg : ℝ) (min_temp : ℝ) :
  temps.card = 5 →
  Finset.sum temps id / temps.card = avg →
  avg = 60 →
  min_temp = 40 →
  min_temp ∈ temps →
  ∀ t ∈ temps, t ≥ min_temp →
  ∃ max_temp ∈ temps, max_temp - min_temp ≤ 100 ∧
    ∀ t ∈ temps, t - min_temp ≤ max_temp - min_temp :=
by sorry

end NUMINAMATH_CALUDE_max_temperature_range_l1616_161646


namespace NUMINAMATH_CALUDE_min_value_of_reciprocal_sum_l1616_161627

/-- An arithmetic sequence of positive terms -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

theorem min_value_of_reciprocal_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (∀ n, a n > 0) →
  a 1 + a 2015 = 2 →
  (∀ x y : ℝ, x > 0 → y > 0 → 1/x + 1/y ≥ 2) →
  ∃ m : ℝ, m = 1/a 2 + 1/a 2014 ∧ m ≥ 2 ∧ ∀ z, z = 1/a 2 + 1/a 2014 → z ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_of_reciprocal_sum_l1616_161627


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l1616_161688

theorem quadratic_roots_condition (b c : ℝ) :
  (c < 0 → ∃ x : ℂ, x^2 + b*x + c = 0) ∧
  ¬(∃ x : ℂ, x^2 + b*x + c = 0 → c < 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l1616_161688


namespace NUMINAMATH_CALUDE_kalebs_clothing_l1616_161674

def total_clothing (first_load : ℕ) (num_equal_loads : ℕ) (pieces_per_equal_load : ℕ) : ℕ :=
  first_load + num_equal_loads * pieces_per_equal_load

theorem kalebs_clothing :
  total_clothing 19 5 4 = 39 := by
  sorry

end NUMINAMATH_CALUDE_kalebs_clothing_l1616_161674


namespace NUMINAMATH_CALUDE_solution_x_proportion_l1616_161656

/-- Represents a solution with a given percentage of material a -/
structure Solution where
  a_percent : ℚ
  b_percent : ℚ
  sum_to_one : a_percent + b_percent = 1

/-- Represents the mixture of solutions -/
structure Mixture where
  x : ℚ
  y : ℚ
  z : ℚ
  sum_to_one : x + y + z = 1

theorem solution_x_proportion (sol_x sol_y sol_z : Solution) (mix : Mixture) :
  sol_x.a_percent = 1/5 →
  sol_y.a_percent = 3/10 →
  sol_z.a_percent = 2/5 →
  mix.y = (3/5) * (mix.y + mix.z) →
  mix.z = (2/5) * (mix.y + mix.z) →
  sol_x.a_percent * mix.x + sol_y.a_percent * mix.y + sol_z.a_percent * mix.z = 1/4 →
  mix.x / (mix.x + mix.y + mix.z) = 9/14 := by
  sorry

end NUMINAMATH_CALUDE_solution_x_proportion_l1616_161656


namespace NUMINAMATH_CALUDE_eleventh_number_with_digit_sum_13_l1616_161630

/-- A function that returns the sum of digits of a positive integer -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- A function that returns the nth positive integer whose digits sum to 13 -/
def nthNumberWithDigitSum13 (n : ℕ) : ℕ := sorry

/-- Theorem stating that the 11th number with digit sum 13 is 166 -/
theorem eleventh_number_with_digit_sum_13 : nthNumberWithDigitSum13 11 = 166 := by sorry

end NUMINAMATH_CALUDE_eleventh_number_with_digit_sum_13_l1616_161630


namespace NUMINAMATH_CALUDE_triangle_ABC_area_l1616_161660

-- Define the vertices of the triangle
def A : ℝ × ℝ := (-1, 2)
def B : ℝ × ℝ := (8, 2)
def C : ℝ × ℝ := (6, -1)

-- Define the function to calculate the area of a triangle given its vertices
def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem triangle_ABC_area :
  triangleArea A B C = 13.5 := by sorry

end NUMINAMATH_CALUDE_triangle_ABC_area_l1616_161660


namespace NUMINAMATH_CALUDE_no_infinite_prime_sequence_l1616_161657

theorem no_infinite_prime_sequence (p : ℕ → ℕ) :
  (∀ n, Prime (p n)) →
  (∀ n, p n < p (n + 1)) →
  (∀ k, p (k + 1) = 2 * p k - 1 ∨ p (k + 1) = 2 * p k + 1) →
  ∃ N, ∀ n > N, ¬ Prime (p n) :=
sorry

end NUMINAMATH_CALUDE_no_infinite_prime_sequence_l1616_161657


namespace NUMINAMATH_CALUDE_unique_fraction_that_triples_l1616_161648

def is_proper_fraction (a b : ℕ) : Prop := 0 < a ∧ a < b

def triples_when_modified (a b : ℕ) : Prop :=
  (a^3 : ℚ) / (b + 3) = 3 * (a : ℚ) / b

theorem unique_fraction_that_triples :
  ∃! (a b : ℕ), is_proper_fraction a b ∧ triples_when_modified a b ∧ a = 2 ∧ b = 9 :=
sorry

end NUMINAMATH_CALUDE_unique_fraction_that_triples_l1616_161648


namespace NUMINAMATH_CALUDE_f_has_unique_zero_in_interval_l1616_161631

noncomputable def f (x : ℝ) := Real.exp x + 4 * x - 3

theorem f_has_unique_zero_in_interval :
  ∃! x, x ∈ (Set.Ioo 0 (1/2)) ∧ f x = 0 :=
sorry

end NUMINAMATH_CALUDE_f_has_unique_zero_in_interval_l1616_161631


namespace NUMINAMATH_CALUDE_range_of_p_l1616_161698

/-- The set A of real numbers x satisfying the quadratic equation x^2 + (p+2)x + 1 = 0 -/
def A (p : ℝ) : Set ℝ := {x | x^2 + (p+2)*x + 1 = 0}

/-- The theorem stating the range of p given the conditions -/
theorem range_of_p (p : ℝ) (h : A p ∩ Set.Ici (0 : ℝ) = ∅) : p > -4 :=
sorry

end NUMINAMATH_CALUDE_range_of_p_l1616_161698


namespace NUMINAMATH_CALUDE_equality_comparison_l1616_161680

theorem equality_comparison : 
  (2^3 ≠ 6) ∧ 
  (-1^2 ≠ (-1)^2) ∧ 
  (-2^3 = (-2)^3) ∧ 
  (4^2 / 9 ≠ (4/9)^2) :=
by sorry

end NUMINAMATH_CALUDE_equality_comparison_l1616_161680


namespace NUMINAMATH_CALUDE_fraction_equality_l1616_161685

def fraction_pairs : Set (ℤ × ℤ) :=
  {(0, 6), (1, -1), (6, -6), (13, -7), (-2, -22), (-3, -15), (-8, -10), (-15, -9)}

theorem fraction_equality (k l : ℤ) :
  (7 * k - 5) / (5 * k - 3) = (6 * l - 1) / (4 * l - 3) ↔ (k, l) ∈ fraction_pairs := by
  sorry

#check fraction_equality

end NUMINAMATH_CALUDE_fraction_equality_l1616_161685


namespace NUMINAMATH_CALUDE_ones_digit_of_power_l1616_161639

theorem ones_digit_of_power (x : ℕ) : (2^3)^x = 4096 → (3^(x^3)) % 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ones_digit_of_power_l1616_161639


namespace NUMINAMATH_CALUDE_sandwich_bread_packs_l1616_161610

theorem sandwich_bread_packs (total_sandwiches : ℕ) (slices_per_sandwich : ℕ) (packs_bought : ℕ) :
  total_sandwiches = 8 →
  slices_per_sandwich = 2 →
  packs_bought = 4 →
  (total_sandwiches * slices_per_sandwich) / packs_bought = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_sandwich_bread_packs_l1616_161610


namespace NUMINAMATH_CALUDE_solution_exists_for_quadratic_cubic_congruence_l1616_161654

theorem solution_exists_for_quadratic_cubic_congruence (p : ℕ) (hp : Prime p) (a : ℤ) :
  ∃ (x y : ℤ), (x^2 + y^3) % p = a % p := by
  sorry

end NUMINAMATH_CALUDE_solution_exists_for_quadratic_cubic_congruence_l1616_161654


namespace NUMINAMATH_CALUDE_holly_initial_milk_l1616_161671

/-- Represents the amount of chocolate milk Holly has throughout the day -/
structure ChocolateMilk where
  initial : ℕ
  breakfast : ℕ
  lunch_purchased : ℕ
  lunch : ℕ
  dinner : ℕ
  final : ℕ

/-- The conditions of Holly's chocolate milk consumption -/
def holly_milk : ChocolateMilk where
  breakfast := 8
  lunch_purchased := 64
  lunch := 8
  dinner := 8
  final := 56
  initial := 0  -- This will be proven

/-- Theorem stating that Holly's initial amount of chocolate milk was 80 ounces -/
theorem holly_initial_milk :
  holly_milk.initial = 80 :=
by sorry

end NUMINAMATH_CALUDE_holly_initial_milk_l1616_161671


namespace NUMINAMATH_CALUDE_ratio_antecedent_l1616_161615

theorem ratio_antecedent (ratio_a ratio_b consequent : ℚ) : 
  ratio_a / ratio_b = 4 / 6 →
  consequent = 45 →
  ratio_a / ratio_b = ratio_a / consequent →
  ratio_a = 30 := by
  sorry

end NUMINAMATH_CALUDE_ratio_antecedent_l1616_161615


namespace NUMINAMATH_CALUDE_recreation_spending_comparison_l1616_161613

theorem recreation_spending_comparison 
  (last_week_wages : ℝ) 
  (last_week_recreation_percent : ℝ) 
  (this_week_wage_reduction : ℝ) 
  (this_week_recreation_percent : ℝ) 
  (h1 : last_week_recreation_percent = 0.1)
  (h2 : this_week_wage_reduction = 0.1)
  (h3 : this_week_recreation_percent = 0.4) :
  (this_week_recreation_percent * (1 - this_week_wage_reduction) * last_week_wages) / 
  (last_week_recreation_percent * last_week_wages) * 100 = 360 := by
  sorry

end NUMINAMATH_CALUDE_recreation_spending_comparison_l1616_161613


namespace NUMINAMATH_CALUDE_height_comparison_l1616_161609

theorem height_comparison (a b : ℝ) (h : a = 0.6 * b) :
  (b - a) / a * 100 = 200 / 3 :=
sorry

end NUMINAMATH_CALUDE_height_comparison_l1616_161609


namespace NUMINAMATH_CALUDE_base9_813_equals_base3_220110_l1616_161644

/-- Converts a base-9 number to base-3 --/
def base9_to_base3 (n : ℕ) : ℕ :=
  sorry

/-- Theorem: 813 in base 9 is equal to 220110 in base 3 --/
theorem base9_813_equals_base3_220110 : base9_to_base3 813 = 220110 := by
  sorry

end NUMINAMATH_CALUDE_base9_813_equals_base3_220110_l1616_161644


namespace NUMINAMATH_CALUDE_train_travel_time_equation_l1616_161675

/-- Proves that the equation for the difference in travel times between two trains is correct -/
theorem train_travel_time_equation (x : ℝ) (h : x > 0) : 
  700 / x - 700 / (2.8 * x) = 3.6 := by
  sorry

end NUMINAMATH_CALUDE_train_travel_time_equation_l1616_161675


namespace NUMINAMATH_CALUDE_trig_identity_l1616_161601

theorem trig_identity : 
  2 * Real.sin (50 * π / 180) + 
  Real.sin (10 * π / 180) * (1 + Real.sqrt 3 * Real.tan (10 * π / 180)) * 
  Real.sqrt (2 * Real.sin (80 * π / 180) ^ 2) = Real.sqrt 6 := by sorry

end NUMINAMATH_CALUDE_trig_identity_l1616_161601


namespace NUMINAMATH_CALUDE_function_equality_implies_a_equals_two_l1616_161677

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then a^x else 1 - x

theorem function_equality_implies_a_equals_two (a : ℝ) :
  f a 1 = f a (-1) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_function_equality_implies_a_equals_two_l1616_161677


namespace NUMINAMATH_CALUDE_elephant_count_theorem_l1616_161637

/-- The total number of elephants in two parks, given the number in one park
    and a multiplier for the other park. -/
def total_elephants (park1_count : ℕ) (multiplier : ℕ) : ℕ :=
  park1_count + multiplier * park1_count

/-- Theorem stating that the total number of elephants in two parks is 280,
    given that one park has 70 elephants and the other has 3 times as many. -/
theorem elephant_count_theorem :
  total_elephants 70 3 = 280 := by
  sorry

end NUMINAMATH_CALUDE_elephant_count_theorem_l1616_161637


namespace NUMINAMATH_CALUDE_sequence_gcd_property_l1616_161604

/-- Given a sequence of natural numbers satisfying the GCD property, prove that a_i = i for all i. -/
theorem sequence_gcd_property (a : ℕ → ℕ) 
  (h : ∀ (i j : ℕ), i ≠ j → Nat.gcd (a i) (a j) = Nat.gcd i j) :
  ∀ (i : ℕ), a i = i :=
by sorry

end NUMINAMATH_CALUDE_sequence_gcd_property_l1616_161604


namespace NUMINAMATH_CALUDE_complex_subtraction_l1616_161693

theorem complex_subtraction (z₁ z₂ : ℂ) (h₁ : z₁ = -2 - I) (h₂ : z₂ = I) : 
  z₁ - 2 * z₂ = -2 - 3 * I := by
  sorry

end NUMINAMATH_CALUDE_complex_subtraction_l1616_161693


namespace NUMINAMATH_CALUDE_odd_function_decomposition_l1616_161684

/-- An odd function. -/
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- A periodic function with period T. -/
def PeriodicFunction (φ : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, φ (x + T) = φ x

/-- A linear function. -/
def LinearFunction (g : ℝ → ℝ) : Prop :=
  ∃ k h : ℝ, ∀ x, g x = k * x + h

/-- A function with a center of symmetry at (a, b). -/
def HasCenterOfSymmetry (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, f (a + x) + f (a - x) = 2 * b

theorem odd_function_decomposition (f : ℝ → ℝ) :
  OddFunction f →
  (∃ φ g : ℝ → ℝ, ∃ T : ℝ, T ≠ 0 ∧
    PeriodicFunction φ T ∧
    LinearFunction g ∧
    (∀ x, f x = φ x + g x)) ↔
  (∃ a b : ℝ, (a, b) ≠ (0, 0) ∧ HasCenterOfSymmetry f a b ∧ ∃ k : ℝ, b = k * a) :=
sorry

end NUMINAMATH_CALUDE_odd_function_decomposition_l1616_161684


namespace NUMINAMATH_CALUDE_fish_tank_leak_bucket_size_l1616_161663

/-- 
Given a fish tank leaking at a rate of 1.5 ounces per hour and a maximum time away of 12 hours,
prove that a bucket with twice the capacity of the total leakage will hold 36 ounces.
-/
theorem fish_tank_leak_bucket_size 
  (leak_rate : ℝ) 
  (max_time : ℝ) 
  (h1 : leak_rate = 1.5)
  (h2 : max_time = 12) : 
  2 * (leak_rate * max_time) = 36 := by
  sorry

#check fish_tank_leak_bucket_size

end NUMINAMATH_CALUDE_fish_tank_leak_bucket_size_l1616_161663


namespace NUMINAMATH_CALUDE_tan_22_5_deg_decomposition_l1616_161622

theorem tan_22_5_deg_decomposition :
  ∃ (a b c d : ℕ+),
    (Real.tan (22.5 * π / 180) = (a : ℝ).sqrt - b + (c : ℝ).sqrt - (d : ℝ).sqrt) ∧
    (a ≥ b) ∧ (b ≥ c) ∧ (c ≥ d) ∧
    (a + b + c + d = 3) := by
  sorry

end NUMINAMATH_CALUDE_tan_22_5_deg_decomposition_l1616_161622


namespace NUMINAMATH_CALUDE_inequality_solutions_l1616_161617

-- Define the solution sets
def solution_set1 : Set ℝ := {x | -3/2 < x ∧ x < 3}
def solution_set2 : Set ℝ := {x | -5 < x ∧ x ≤ 3/2}
def solution_set3 : Set ℝ := ∅
def solution_set4 : Set ℝ := Set.univ

-- Define the inequalities
def inequality1 (x : ℝ) : Prop := -2*x^2 + 3*x + 9 > 0
def inequality2 (x : ℝ) : Prop := (8 - x) / (5 + x) > 1
def inequality3 (x : ℝ) : Prop := -x^2 + 2*x - 3 > 0
def inequality4 (x : ℝ) : Prop := x^2 - 14*x + 50 > 0

-- Theorem statements
theorem inequality_solutions :
  (∀ x, x ∈ solution_set1 ↔ inequality1 x) ∧
  (∀ x, x ∈ solution_set2 ↔ inequality2 x) ∧
  (∀ x, x ∈ solution_set3 ↔ inequality3 x) ∧
  (∀ x, x ∈ solution_set4 ↔ inequality4 x) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solutions_l1616_161617


namespace NUMINAMATH_CALUDE_existence_of_solution_l1616_161662

theorem existence_of_solution (n : ℕ+) (a b c : ℕ+) 
  (ha : a ≤ 3 * n^2 + 4 * n) 
  (hb : b ≤ 3 * n^2 + 4 * n) 
  (hc : c ≤ 3 * n^2 + 4 * n) : 
  ∃ (x y z : ℤ), 
    (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0) ∧ 
    (abs x ≤ 2 * n) ∧ 
    (abs y ≤ 2 * n) ∧ 
    (abs z ≤ 2 * n) ∧ 
    (a * x + b * y + c * z = 0) :=
sorry

end NUMINAMATH_CALUDE_existence_of_solution_l1616_161662


namespace NUMINAMATH_CALUDE_pages_left_to_read_l1616_161600

/-- Calculates the number of pages left to read in a storybook -/
theorem pages_left_to_read (a b : ℕ) : a - 8 * b = a - 8 * b :=
by
  sorry

#check pages_left_to_read

end NUMINAMATH_CALUDE_pages_left_to_read_l1616_161600


namespace NUMINAMATH_CALUDE_odd_function_property_l1616_161667

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_property (f : ℝ → ℝ) 
  (h_odd : is_odd_function f)
  (h_prop : ∀ x, f (-x + 1) = f (x + 1))
  (h_val : f (-1) = 1) :
  f 2017 = -1 := by
sorry

end NUMINAMATH_CALUDE_odd_function_property_l1616_161667


namespace NUMINAMATH_CALUDE_triangle_dance_nine_people_l1616_161653

def triangle_dance_rounds (n : ℕ) : ℕ :=
  if n % 3 = 0 then
    (Nat.factorial n) / ((Nat.factorial 3)^3 * Nat.factorial (n / 3))
  else
    0

theorem triangle_dance_nine_people :
  triangle_dance_rounds 9 = 280 :=
by sorry

end NUMINAMATH_CALUDE_triangle_dance_nine_people_l1616_161653


namespace NUMINAMATH_CALUDE_remainder_preserving_operation_l1616_161632

theorem remainder_preserving_operation (N : ℤ) (f : ℤ → ℤ) :
  N % 6 = 3 → f N % 6 = 3 →
  ∃ k : ℤ, f N = N + 6 * k :=
sorry

end NUMINAMATH_CALUDE_remainder_preserving_operation_l1616_161632


namespace NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l1616_161605

/-- Given a parabola and a circle with specific properties, prove the distance from the focus to the directrix -/
theorem parabola_focus_directrix_distance 
  (p : ℝ) 
  (h_p_pos : p > 0)
  (parabola : ℝ → ℝ → Prop)
  (circle : ℝ → ℝ → Prop)
  (h_parabola : ∀ x y, parabola x y ↔ y^2 = 2*p*x)
  (h_circle : ∀ x y, circle x y ↔ x^2 + (y-1)^2 = 1)
  (h_intersect : ∃ (x1 y1 x2 y2 : ℝ), 
    parabola x1 y1 ∧ circle x1 y1 ∧
    parabola x2 y2 ∧ circle x2 y2 ∧
    x1 ≠ x2 ∧ y1 ≠ y2)
  (h_distance : ∃ (x1 y1 x2 y2 : ℝ),
    parabola x1 y1 ∧ circle x1 y1 ∧
    parabola x2 y2 ∧ circle x2 y2 ∧
    (x1 - x2)^2 + (y1 - y2)^2 = (2 * Real.sqrt 3 / 3)^2) :
  p = Real.sqrt 2 / 6 := by
sorry

end NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l1616_161605


namespace NUMINAMATH_CALUDE_oil_purchase_increase_l1616_161640

/-- Calculates the additional amount of oil that can be purchased after a price reduction -/
def additional_oil_purchase (price_reduction : ℚ) (budget : ℚ) (reduced_price : ℚ) : ℚ :=
  let original_price := reduced_price / (1 - price_reduction)
  let original_amount := budget / original_price
  let new_amount := budget / reduced_price
  new_amount - original_amount

/-- Proves that given a 30% price reduction, a budget of 700, and a reduced price of 70,
    the additional amount of oil that can be purchased is 3 -/
theorem oil_purchase_increase :
  additional_oil_purchase (30 / 100) 700 70 = 3 := by
  sorry

end NUMINAMATH_CALUDE_oil_purchase_increase_l1616_161640


namespace NUMINAMATH_CALUDE_solution_set_when_a_zero_range_of_a_no_solution_l1616_161669

-- Define the function f(x) = |2x+2| - |x-1|
def f (x : ℝ) : ℝ := |2*x + 2| - |x - 1|

-- Part 1: Solution set when a = 0
theorem solution_set_when_a_zero :
  {x : ℝ | f x > 0} = {x : ℝ | x < -3 ∨ x > -1/3} := by sorry

-- Part 2: Range of a when no solution in [-4, 2]
theorem range_of_a_no_solution :
  ∀ a : ℝ, (∀ x ∈ Set.Icc (-4 : ℝ) 2, f x ≤ a) → a ≤ 3 := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_zero_range_of_a_no_solution_l1616_161669


namespace NUMINAMATH_CALUDE_correct_monk_bun_equations_l1616_161676

/-- Represents the monk and bun distribution problem -/
def monk_bun_problem (x y : ℕ) : Prop :=
  -- Total number of monks is 100
  x + y = 100 ∧
  -- Total number of buns is 100, distributed as 3 per elder monk and 1/3 per younger monk
  3 * x + y / 3 = 100

/-- The correct system of equations for the monk and bun distribution problem -/
theorem correct_monk_bun_equations :
  ∀ x y : ℕ, monk_bun_problem x y ↔ x + y = 100 ∧ 3 * x + y / 3 = 100 := by
  sorry

end NUMINAMATH_CALUDE_correct_monk_bun_equations_l1616_161676


namespace NUMINAMATH_CALUDE_subset_intersection_one_element_l1616_161678

/-- Given n+1 distinct subsets of [n], each with exactly 3 elements,
    there must exist a pair of subsets whose intersection has exactly one element. -/
theorem subset_intersection_one_element
  (n : ℕ)
  (A : Fin (n + 1) → Finset (Fin n))
  (h_distinct : ∀ i j, i ≠ j → A i ≠ A j)
  (h_card : ∀ i, (A i).card = 3) :
  ∃ i j, i ≠ j ∧ (A i ∩ A j).card = 1 := by
  sorry

end NUMINAMATH_CALUDE_subset_intersection_one_element_l1616_161678


namespace NUMINAMATH_CALUDE_parabola_sum_a_c_l1616_161666

/-- A parabola that intersects the x-axis at x = -1 -/
structure Parabola where
  a : ℝ
  c : ℝ
  intersect_at_neg_one : a * (-1)^2 + (-1) + c = 0

/-- The sum of a and c for a parabola intersecting the x-axis at x = -1 is 1 -/
theorem parabola_sum_a_c (p : Parabola) : p.a + p.c = 1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_sum_a_c_l1616_161666


namespace NUMINAMATH_CALUDE_solve_ice_problem_l1616_161661

def ice_problem (ice_in_glass : ℕ) (num_trays : ℕ) : Prop :=
  let ice_in_pitcher : ℕ := 2 * ice_in_glass
  let total_ice : ℕ := ice_in_glass + ice_in_pitcher
  let spaces_per_tray : ℕ := total_ice / num_trays
  (ice_in_glass = 8) ∧ (num_trays = 2) → (spaces_per_tray = 12)

theorem solve_ice_problem : ice_problem 8 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_ice_problem_l1616_161661


namespace NUMINAMATH_CALUDE_square_ratio_proof_l1616_161664

theorem square_ratio_proof : ∃ (a b c : ℕ), 
  (300 : ℚ) / 75 = (a * Real.sqrt b / c)^2 ∧ a + b + c = 4 :=
by sorry

end NUMINAMATH_CALUDE_square_ratio_proof_l1616_161664


namespace NUMINAMATH_CALUDE_sum_of_seven_consecutive_integers_l1616_161682

theorem sum_of_seven_consecutive_integers (n : ℤ) :
  n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) = 7 * n + 21 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_seven_consecutive_integers_l1616_161682


namespace NUMINAMATH_CALUDE_range_of_a_l1616_161626

-- Define the function f(x) = x^2 - 2x + a
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*x + a

-- State the theorem
theorem range_of_a (h : ∀ x ∈ Set.Icc 2 3, f a x > 0) : a > 0 := by
  sorry


end NUMINAMATH_CALUDE_range_of_a_l1616_161626


namespace NUMINAMATH_CALUDE_first_day_charge_l1616_161650

/-- Represents the charge and attendance for a three-day show -/
structure ShowData where
  day1_charge : ℝ
  day2_charge : ℝ
  day3_charge : ℝ
  attendance_ratio : Fin 3 → ℝ
  average_charge : ℝ

/-- Theorem stating the charge on the first day given the show data -/
theorem first_day_charge (s : ShowData)
  (h1 : s.day2_charge = 7.5)
  (h2 : s.day3_charge = 2.5)
  (h3 : s.attendance_ratio 0 = 2)
  (h4 : s.attendance_ratio 1 = 5)
  (h5 : s.attendance_ratio 2 = 13)
  (h6 : s.average_charge = 5)
  (h7 : (s.attendance_ratio 0 * s.day1_charge + 
         s.attendance_ratio 1 * s.day2_charge + 
         s.attendance_ratio 2 * s.day3_charge) / 
        (s.attendance_ratio 0 + s.attendance_ratio 1 + s.attendance_ratio 2) = s.average_charge) :
  s.day1_charge = 15 := by
  sorry

end NUMINAMATH_CALUDE_first_day_charge_l1616_161650


namespace NUMINAMATH_CALUDE_range_of_m_m_value_when_sum_eq_neg_product_l1616_161625

-- Define the quadratic equation
def quadratic (m x : ℝ) : ℝ := x^2 - (2*m - 3)*x + m^2

-- Define the roots of the quadratic equation
def roots (m : ℝ) : Set ℝ := {x | quadratic m x = 0}

-- Theorem for the range of m
theorem range_of_m : ∀ m : ℝ, (∃ x₁ x₂ : ℝ, x₁ ∈ roots m ∧ x₂ ∈ roots m) → m ≤ 3/4 := by sorry

-- Theorem for the value of m when x₁ + x₂ = -x₁x₂
theorem m_value_when_sum_eq_neg_product : 
  ∀ m : ℝ, m ≤ 3/4 → 
  (∃ x₁ x₂ : ℝ, x₁ ∈ roots m ∧ x₂ ∈ roots m ∧ x₁ + x₂ = -(x₁ * x₂)) → 
  m = -3 := by sorry

end NUMINAMATH_CALUDE_range_of_m_m_value_when_sum_eq_neg_product_l1616_161625


namespace NUMINAMATH_CALUDE_pairing_possibility_l1616_161629

/-- Represents a pairing of children -/
structure Pairing :=
  (boys : ℕ)   -- Number of boy-boy pairs
  (girls : ℕ)  -- Number of girl-girl pairs
  (mixed : ℕ)  -- Number of boy-girl pairs

/-- Represents a group of children that can be arranged in different pairings -/
structure ChildrenGroup :=
  (to_museum : Pairing)
  (from_museum : Pairing)
  (total_boys : ℕ)
  (total_girls : ℕ)

/-- The theorem to be proved -/
theorem pairing_possibility (group : ChildrenGroup) 
  (h1 : group.to_museum.boys = 3 * group.to_museum.girls)
  (h2 : group.from_museum.boys = 4 * group.from_museum.girls)
  (h3 : group.total_boys = 2 * group.to_museum.boys + group.to_museum.mixed)
  (h4 : group.total_girls = 2 * group.to_museum.girls + group.to_museum.mixed)
  (h5 : group.total_boys = 2 * group.from_museum.boys + group.from_museum.mixed)
  (h6 : group.total_girls = 2 * group.from_museum.girls + group.from_museum.mixed) :
  ∃ (new_pairing : Pairing), 
    new_pairing.boys = 7 * new_pairing.girls ∧ 
    2 * new_pairing.boys + 2 * new_pairing.girls + new_pairing.mixed = group.total_boys + group.total_girls :=
sorry

end NUMINAMATH_CALUDE_pairing_possibility_l1616_161629


namespace NUMINAMATH_CALUDE_meaningful_expression_l1616_161607

theorem meaningful_expression (x : ℝ) : 
  (∃ y : ℝ, y = (Real.sqrt (x + 1)) / x) ↔ x ≥ -1 ∧ x ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_meaningful_expression_l1616_161607


namespace NUMINAMATH_CALUDE_triangle_side_value_l1616_161602

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that if a = 3, c = 2√3, and bsinA = acos(B + π/6), then b = √3 -/
theorem triangle_side_value (A B C : ℝ) (a b c : ℝ) :
  a = 3 →
  c = 2 * Real.sqrt 3 →
  b * Real.sin A = a * Real.cos (B + π/6) →
  b = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_value_l1616_161602


namespace NUMINAMATH_CALUDE_stratified_sample_size_is_72_l1616_161699

/-- Represents the number of teachers in each category -/
structure TeacherCounts where
  fullProf : Nat
  assocProf : Nat
  lecturers : Nat
  teachingAssistants : Nat

/-- Calculates the total number of teachers -/
def totalTeachers (counts : TeacherCounts) : Nat :=
  counts.fullProf + counts.assocProf + counts.lecturers + counts.teachingAssistants

/-- Calculates the sample size for stratified sampling -/
def stratifiedSampleSize (counts : TeacherCounts) (lecturersDrawn : Nat) : Nat :=
  let samplingRate := lecturersDrawn / counts.lecturers
  (totalTeachers counts) * samplingRate

/-- Theorem: Given the specific teacher counts and 16 lecturers drawn, 
    the stratified sample size is 72 -/
theorem stratified_sample_size_is_72 
  (counts : TeacherCounts) 
  (h1 : counts.fullProf = 120) 
  (h2 : counts.assocProf = 100) 
  (h3 : counts.lecturers = 80) 
  (h4 : counts.teachingAssistants = 60) 
  (h5 : stratifiedSampleSize counts 16 = 72) : 
  stratifiedSampleSize counts 16 = 72 := by
  sorry

#eval stratifiedSampleSize 
  { fullProf := 120, assocProf := 100, lecturers := 80, teachingAssistants := 60 } 16

end NUMINAMATH_CALUDE_stratified_sample_size_is_72_l1616_161699


namespace NUMINAMATH_CALUDE_x_intercept_ratio_l1616_161655

/-- Two lines with the same non-zero y-intercept -/
structure TwoLines where
  b : ℝ
  s : ℝ
  t : ℝ
  b_nonzero : b ≠ 0
  first_line : ∀ x y : ℝ, y = 8 * x + b ↔ (x = s ∧ y = 0) ∨ (x = 0 ∧ y = b)
  second_line : ∀ x y : ℝ, y = 4 * x + b ↔ (x = t ∧ y = 0) ∨ (x = 0 ∧ y = b)

/-- The ratio of x-intercepts is 1/2 -/
theorem x_intercept_ratio (lines : TwoLines) : lines.s / lines.t = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_x_intercept_ratio_l1616_161655
