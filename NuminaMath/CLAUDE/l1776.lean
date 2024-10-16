import Mathlib

namespace NUMINAMATH_CALUDE_cos_10_coeff_sum_l1776_177636

/-- Represents the coefficients in the expansion of cos(10α) -/
structure Cos10Coeffs where
  m : ℤ
  n : ℤ
  p : ℤ

/-- 
Given the equation for cos(10α) in the form:
cos(10α) = m*cos^10(α) - 1280*cos^8(α) + 1120*cos^6(α) + n*cos^4(α) + p*cos^2(α) - 1,
prove that m - n + p = 962
-/
theorem cos_10_coeff_sum (coeffs : Cos10Coeffs) : coeffs.m - coeffs.n + coeffs.p = 962 := by
  sorry

end NUMINAMATH_CALUDE_cos_10_coeff_sum_l1776_177636


namespace NUMINAMATH_CALUDE_quadratic_root_form_l1776_177617

theorem quadratic_root_form (d : ℝ) : 
  (∀ x : ℝ, x^2 - 7*x + d = 0 ↔ x = (7 + Real.sqrt d) / 2 ∨ x = (7 - Real.sqrt d) / 2) → 
  d = 49 / 5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_form_l1776_177617


namespace NUMINAMATH_CALUDE_susan_strawberry_picking_l1776_177699

theorem susan_strawberry_picking (basket_capacity : ℕ) (total_picked : ℕ) (eaten_per_handful : ℕ) :
  basket_capacity = 60 →
  total_picked = 75 →
  eaten_per_handful = 1 →
  ∃ (strawberries_per_handful : ℕ),
    strawberries_per_handful * (total_picked / strawberries_per_handful) = total_picked ∧
    (strawberries_per_handful - eaten_per_handful) * (total_picked / strawberries_per_handful) = basket_capacity ∧
    strawberries_per_handful = 5 :=
by sorry

end NUMINAMATH_CALUDE_susan_strawberry_picking_l1776_177699


namespace NUMINAMATH_CALUDE_product_of_squares_and_fourth_powers_l1776_177684

theorem product_of_squares_and_fourth_powers (a b : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b)
  (h_sum_squares : a^2 + b^2 = 5)
  (h_sum_fourth_powers : a^4 + b^4 = 17) : 
  a * b = 2 := by sorry

end NUMINAMATH_CALUDE_product_of_squares_and_fourth_powers_l1776_177684


namespace NUMINAMATH_CALUDE_smallest_value_l1776_177693

theorem smallest_value (y : ℝ) (h1 : 0 < y) (h2 : y < 1) :
  y^3 < 3*y ∧ y^3 < y^(1/2) ∧ y^3 < 1/y ∧ y^3 < Real.exp y := by
  sorry

#check smallest_value

end NUMINAMATH_CALUDE_smallest_value_l1776_177693


namespace NUMINAMATH_CALUDE_only_third_equation_has_nontrivial_solution_l1776_177690

theorem only_third_equation_has_nontrivial_solution :
  ∃ (a b : ℝ), (a ≠ 0 ∨ b ≠ 0) ∧ (Real.sqrt (a^2 + b^2) = a + 2*b) ∧
  (∀ (a b : ℝ), (a ≠ 0 ∨ b ≠ 0) → Real.sqrt (a^2 + b^2) ≠ a - b) ∧
  (∀ (a b : ℝ), (a ≠ 0 ∨ b ≠ 0) → Real.sqrt (a^2 + b^2) ≠ a^2 - b^2) ∧
  (∀ (a b : ℝ), (a ≠ 0 ∨ b ≠ 0) → Real.sqrt (a^2 + b^2) ≠ a^2*b - a*b^2) :=
by sorry

end NUMINAMATH_CALUDE_only_third_equation_has_nontrivial_solution_l1776_177690


namespace NUMINAMATH_CALUDE_female_student_stats_l1776_177691

/-- Represents the class statistics -/
structure ClassStats where
  total_students : ℕ
  male_students : ℕ
  overall_avg_score : ℚ
  male_algebra_avg : ℚ
  male_geometry_avg : ℚ
  male_calculus_avg : ℚ
  female_algebra_avg : ℚ
  female_geometry_avg : ℚ
  female_calculus_avg : ℚ
  algebra_geometry_attendance : ℚ
  calculus_attendance_increase : ℚ

/-- Theorem stating the proportion and number of female students -/
theorem female_student_stats (stats : ClassStats)
  (h_total : stats.total_students = 30)
  (h_male : stats.male_students = 8)
  (h_overall_avg : stats.overall_avg_score = 90)
  (h_male_algebra : stats.male_algebra_avg = 87)
  (h_male_geometry : stats.male_geometry_avg = 95)
  (h_male_calculus : stats.male_calculus_avg = 89)
  (h_female_algebra : stats.female_algebra_avg = 92)
  (h_female_geometry : stats.female_geometry_avg = 94)
  (h_female_calculus : stats.female_calculus_avg = 91)
  (h_alg_geo_attendance : stats.algebra_geometry_attendance = 85)
  (h_calc_attendance : stats.calculus_attendance_increase = 4) :
  (stats.total_students - stats.male_students : ℚ) / stats.total_students = 11 / 15 ∧
  stats.total_students - stats.male_students = 22 := by
    sorry


end NUMINAMATH_CALUDE_female_student_stats_l1776_177691


namespace NUMINAMATH_CALUDE_min_value_theorem_l1776_177653

theorem min_value_theorem (x : ℝ) (h : x > 0) :
  x^2 + 12*x + 128/x^4 ≥ 256 ∧ ∃ y > 0, y^2 + 12*y + 128/y^4 = 256 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1776_177653


namespace NUMINAMATH_CALUDE_ninth_term_is_15_l1776_177627

/-- An arithmetic sequence with properties S3 = 3 and S6 = 24 -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  S : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n, S n = (n / 2) * (a 1 + a n)
  S3_eq_3 : S 3 = 3
  S6_eq_24 : S 6 = 24

/-- The 9th term of the arithmetic sequence is 15 -/
theorem ninth_term_is_15 (seq : ArithmeticSequence) : seq.a 9 = 15 := by
  sorry

end NUMINAMATH_CALUDE_ninth_term_is_15_l1776_177627


namespace NUMINAMATH_CALUDE_equation_solution_l1776_177622

theorem equation_solution : 
  ∃! x : ℚ, (x + 10) / (x - 4) = (x - 3) / (x + 6) ∧ x = -48 / 23 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1776_177622


namespace NUMINAMATH_CALUDE_beth_dive_tanks_l1776_177676

/-- Calculates the number of supplemental tanks needed for a scuba dive. -/
def supplementalTanksNeeded (totalDiveTime primaryTankDuration supplementalTankDuration : ℕ) : ℕ :=
  (totalDiveTime - primaryTankDuration) / supplementalTankDuration

/-- Proves that for the given dive parameters, 6 supplemental tanks are needed. -/
theorem beth_dive_tanks : 
  supplementalTanksNeeded 8 2 1 = 6 := by
  sorry

#eval supplementalTanksNeeded 8 2 1

end NUMINAMATH_CALUDE_beth_dive_tanks_l1776_177676


namespace NUMINAMATH_CALUDE_five_toppings_from_eight_l1776_177654

theorem five_toppings_from_eight (n m : ℕ) (hn : n = 8) (hm : m = 5) :
  Nat.choose n m = 56 := by
  sorry

end NUMINAMATH_CALUDE_five_toppings_from_eight_l1776_177654


namespace NUMINAMATH_CALUDE_at_least_one_passes_l1776_177650

def exam_pool : ℕ := 10
def A_correct : ℕ := 6
def B_correct : ℕ := 8
def test_questions : ℕ := 3
def passing_threshold : ℕ := 2

def prob_A_pass : ℚ := (Nat.choose A_correct 2 * Nat.choose (exam_pool - A_correct) 1 + Nat.choose A_correct 3) / Nat.choose exam_pool test_questions

def prob_B_pass : ℚ := (Nat.choose B_correct 2 * Nat.choose (exam_pool - B_correct) 1 + Nat.choose B_correct 3) / Nat.choose exam_pool test_questions

theorem at_least_one_passes : 
  1 - (1 - prob_A_pass) * (1 - prob_B_pass) = 44 / 45 := by sorry

end NUMINAMATH_CALUDE_at_least_one_passes_l1776_177650


namespace NUMINAMATH_CALUDE_polka_dot_price_is_67_l1776_177643

def checkered_price : ℝ := 75
def plain_price : ℝ := 45
def striped_price : ℝ := 63
def total_price : ℝ := 250

def checkered_per_yard : ℝ := 7.5
def plain_per_yard : ℝ := 6
def striped_per_yard : ℝ := 9
def polka_dot_per_yard : ℝ := 4.5

def discount_rate : ℝ := 0.1
def discount_threshold : ℝ := 10

def polka_dot_price : ℝ := total_price - (checkered_price + plain_price + striped_price)

theorem polka_dot_price_is_67 : polka_dot_price = 67 := by
  sorry

end NUMINAMATH_CALUDE_polka_dot_price_is_67_l1776_177643


namespace NUMINAMATH_CALUDE_males_chose_malt_l1776_177698

/-- Represents the number of cheerleaders who chose malt or coke -/
structure CheerleaderChoices where
  males : ℕ
  females : ℕ

/-- The properties of the cheerleader group and their choices -/
def CheerleaderProblem (choices : CheerleaderChoices) : Prop :=
  -- Total number of cheerleaders
  choices.males + choices.females = 26 ∧
  -- Number of males
  choices.males = 10 ∧
  -- Number of females
  choices.females = 16 ∧
  -- Number of malt choosers is double the number of coke choosers
  choices.males + choices.females = 3 * (26 - (choices.males + choices.females)) ∧
  -- 8 females chose malt
  choices.females = 8

/-- Theorem stating the number of males who chose malt -/
theorem males_chose_malt (choices : CheerleaderChoices) 
  (h : CheerleaderProblem choices) : choices.males = 9 := by
  sorry


end NUMINAMATH_CALUDE_males_chose_malt_l1776_177698


namespace NUMINAMATH_CALUDE_gcd_18_24_l1776_177616

theorem gcd_18_24 : Nat.gcd 18 24 = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcd_18_24_l1776_177616


namespace NUMINAMATH_CALUDE_power_of_two_equality_l1776_177681

theorem power_of_two_equality (K : ℕ) : 32^2 * 4^5 = 2^K ↔ K = 20 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_equality_l1776_177681


namespace NUMINAMATH_CALUDE_correct_average_l1776_177677

theorem correct_average (n : ℕ) (initial_avg : ℚ) (incorrect_num correct_num : ℚ) :
  n = 10 →
  initial_avg = 16 →
  incorrect_num = 26 →
  correct_num = 46 →
  (n : ℚ) * initial_avg - incorrect_num + correct_num = n * 18 := by
  sorry

end NUMINAMATH_CALUDE_correct_average_l1776_177677


namespace NUMINAMATH_CALUDE_problem_solution_l1776_177612

noncomputable def f (a : ℝ) : ℝ → ℝ := fun x ↦ -1/2 + a / (3^x + 1)

theorem problem_solution (a : ℝ) (h_odd : ∀ x, f a x = -(f a (-x))) :
  (a = 1) ∧
  (∀ x y, x < y → f a x > f a y) ∧
  (∀ m, (∃ t ∈ Set.Ioo 1 2, f a (-2*t^2 + t + 1) + f a (t^2 - 2*m*t) ≤ 0) → m < 1/2) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1776_177612


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l1776_177620

theorem polynomial_divisibility (F : ℤ → ℤ) (A : Finset ℤ) :
  (∀ (n : ℤ), ∃ (a : ℤ), a ∈ A ∧ a ∣ F n) →
  (∃ (a : ℤ), a ∈ A ∧ ∀ (n : ℤ), a ∣ F n) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l1776_177620


namespace NUMINAMATH_CALUDE_sqrt_76_between_8_and_9_l1776_177618

theorem sqrt_76_between_8_and_9 : 8 < Real.sqrt 76 ∧ Real.sqrt 76 < 9 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_76_between_8_and_9_l1776_177618


namespace NUMINAMATH_CALUDE_parallel_implies_n_eq_two_transform_implies_m_n_eq_neg_one_l1776_177679

-- Define points A and B in the Cartesian coordinate system
def A (m : ℝ) : ℝ × ℝ := (3, 2*m - 1)
def B (n : ℝ) : ℝ × ℝ := (n + 1, -1)

-- Define the condition that A and B are not coincident
def not_coincident (m n : ℝ) : Prop := A m ≠ B n

-- Define what it means for AB to be parallel to y-axis
def parallel_to_y_axis (m n : ℝ) : Prop := (A m).1 = (B n).1

-- Define the transformation of A to B
def transform_A_to_B (m n : ℝ) : Prop :=
  (A m).1 - 3 = (B n).1 ∧ (A m).2 + 2 = (B n).2

-- Theorem 1
theorem parallel_implies_n_eq_two (m n : ℝ) 
  (h1 : not_coincident m n) (h2 : parallel_to_y_axis m n) : n = 2 := by sorry

-- Theorem 2
theorem transform_implies_m_n_eq_neg_one (m n : ℝ) 
  (h1 : not_coincident m n) (h2 : transform_A_to_B m n) : m = -1 ∧ n = -1 := by sorry

end NUMINAMATH_CALUDE_parallel_implies_n_eq_two_transform_implies_m_n_eq_neg_one_l1776_177679


namespace NUMINAMATH_CALUDE_coin_problem_l1776_177619

/-- Represents the number of different coin values that can be made -/
def different_values (x y : ℕ) : ℕ := 29 - (3 * x + 2 * y) / 2

/-- The coin problem -/
theorem coin_problem (total : ℕ) (values : ℕ) :
  total = 12 ∧ values = 21 →
  ∃ x y : ℕ, x + y = total ∧ different_values x y = values ∧ y = 7 := by
  sorry

end NUMINAMATH_CALUDE_coin_problem_l1776_177619


namespace NUMINAMATH_CALUDE_complex_magnitude_product_l1776_177646

theorem complex_magnitude_product : 
  Complex.abs ((3 * Real.sqrt 2 - 5 * Complex.I) * (2 * Real.sqrt 3 + 2 * Complex.I)) = 4 * Real.sqrt 43 := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_product_l1776_177646


namespace NUMINAMATH_CALUDE_line_parameterization_l1776_177631

/-- Given a line y = 5x - 7 parameterized by (x, y) = (s, 2) + t(3, m),
    prove that s = 9/5 and m = 3 -/
theorem line_parameterization (s m : ℝ) :
  (∀ t : ℝ, ∀ x y : ℝ, 
    x = s + 3*t ∧ y = 2 + m*t → y = 5*x - 7) →
  s = 9/5 ∧ m = 3 := by
sorry

end NUMINAMATH_CALUDE_line_parameterization_l1776_177631


namespace NUMINAMATH_CALUDE_chromium_54_neutrons_l1776_177668

/-- The number of neutrons in an atom of chromium-54 -/
def neutrons_per_atom : ℕ := 54 - 24

/-- Avogadro's constant (atoms per mole) -/
def avogadro : ℝ := 6.022e23

/-- Amount of substance in moles -/
def amount : ℝ := 0.025

/-- Approximate number of neutrons in the given amount of chromium-54 -/
def total_neutrons : ℝ := amount * avogadro * neutrons_per_atom

theorem chromium_54_neutrons : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1e23 ∧ |total_neutrons - 4.5e23| < ε :=
sorry

end NUMINAMATH_CALUDE_chromium_54_neutrons_l1776_177668


namespace NUMINAMATH_CALUDE_sucrose_concentration_in_mixture_l1776_177660

/-- Concentration of sucrose in a mixture of two solutions --/
theorem sucrose_concentration_in_mixture 
  (conc_A : ℝ) (conc_B : ℝ) (vol_A : ℝ) (vol_B : ℝ) 
  (h1 : conc_A = 15.3) 
  (h2 : conc_B = 27.8) 
  (h3 : vol_A = 45) 
  (h4 : vol_B = 75) : 
  (conc_A * vol_A + conc_B * vol_B) / (vol_A + vol_B) = 
  (15.3 * 45 + 27.8 * 75) / (45 + 75) :=
by sorry

#eval (15.3 * 45 + 27.8 * 75) / (45 + 75)

end NUMINAMATH_CALUDE_sucrose_concentration_in_mixture_l1776_177660


namespace NUMINAMATH_CALUDE_intersection_k_value_l1776_177680

/-- The intersection point of two lines -3x + y = k and 2x + y = 20 when x = -10 -/
def intersection_point : ℝ × ℝ := (-10, 40)

/-- The first line equation: -3x + y = k -/
def line1 (k : ℝ) (p : ℝ × ℝ) : Prop :=
  -3 * p.1 + p.2 = k

/-- The second line equation: 2x + y = 20 -/
def line2 (p : ℝ × ℝ) : Prop :=
  2 * p.1 + p.2 = 20

/-- Theorem: The value of k is 70 given that the lines -3x + y = k and 2x + y = 20 intersect when x = -10 -/
theorem intersection_k_value :
  line2 intersection_point →
  (∃ k, line1 k intersection_point) →
  (∃! k, line1 k intersection_point ∧ k = 70) :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_k_value_l1776_177680


namespace NUMINAMATH_CALUDE_max_sum_absolute_values_l1776_177630

theorem max_sum_absolute_values (x y : ℝ) (h : 4 * x^2 + y^2 = 4) :
  ∃ (M : ℝ), M = 2 ∧ ∀ (a b : ℝ), 4 * a^2 + b^2 = 4 → |a| + |b| ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_sum_absolute_values_l1776_177630


namespace NUMINAMATH_CALUDE_ratio_of_a_to_d_l1776_177604

theorem ratio_of_a_to_d (a b c d : ℚ) 
  (hab : a / b = 5 / 3)
  (hbc : b / c = 1 / 5)
  (hcd : c / d = 3 / 2) :
  a / d = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_a_to_d_l1776_177604


namespace NUMINAMATH_CALUDE_equation_solution_l1776_177659

theorem equation_solution :
  ∀ x : ℚ, (6 * x / (x + 4) - 2 / (x + 4) = 3 / (x + 4)) ↔ (x = 5 / 6) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1776_177659


namespace NUMINAMATH_CALUDE_sum_of_abc_l1776_177613

theorem sum_of_abc (a b c : ℝ) 
  (eq1 : a^2 + 6*b = -17)
  (eq2 : b^2 + 8*c = -23)
  (eq3 : c^2 + 2*a = 14) :
  a + b + c = -8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_abc_l1776_177613


namespace NUMINAMATH_CALUDE_distributor_profit_percentage_l1776_177601

/-- Proves that the distributor's profit percentage is 65% given the specified conditions -/
theorem distributor_profit_percentage
  (commission_rate : Real)
  (producer_price : Real)
  (final_price : Real)
  (h1 : commission_rate = 0.2)
  (h2 : producer_price = 15)
  (h3 : final_price = 19.8) :
  (((final_price / (1 - commission_rate)) - producer_price) / producer_price) * 100 = 65 := by
  sorry

end NUMINAMATH_CALUDE_distributor_profit_percentage_l1776_177601


namespace NUMINAMATH_CALUDE_derivative_implies_antiderivative_l1776_177645

theorem derivative_implies_antiderivative (f : ℝ → ℝ) :
  (∀ x, deriv f x = 6 * x^2 + 5) →
  ∃ c, ∀ x, f x = 2 * x^3 + 5 * x + c :=
sorry

end NUMINAMATH_CALUDE_derivative_implies_antiderivative_l1776_177645


namespace NUMINAMATH_CALUDE_watch_cost_price_l1776_177663

/-- The cost price of a watch given specific selling conditions -/
theorem watch_cost_price (C : ℚ) : 
  (0.9 * C = C - 0.1 * C) →  -- Selling price at 10% loss
  (1.04 * C = C + 0.04 * C) →  -- Selling price at 4% gain
  (1.04 * C - 0.9 * C = 200) →  -- Difference between selling prices
  C = 10000 / 7 := by
sorry

end NUMINAMATH_CALUDE_watch_cost_price_l1776_177663


namespace NUMINAMATH_CALUDE_cubic_equation_solutions_l1776_177683

theorem cubic_equation_solutions :
  {x : ℝ | x^3 + (2 - x)^3 = 8} = {0, 2} := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_solutions_l1776_177683


namespace NUMINAMATH_CALUDE_fraction_to_seventh_power_l1776_177644

theorem fraction_to_seventh_power : (2 / 5 : ℚ) ^ 7 = 128 / 78125 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_seventh_power_l1776_177644


namespace NUMINAMATH_CALUDE_total_weight_proof_l1776_177610

/-- Represents the weight of construction materials in various units -/
structure ConstructionMaterials where
  concrete : Float  -- in pounds
  bricks : Float    -- in kilograms
  stone : Float     -- in pounds
  wood : Float      -- in kilograms
  steel : Float     -- in ounces
  glass : Float     -- in tons
  sand : Float      -- in pounds

/-- Conversion rates between different units of weight -/
structure ConversionRates where
  kg_to_lbs : Float
  ton_to_lbs : Float
  oz_to_lbs : Float

/-- Calculates the total weight of construction materials in pounds -/
def totalWeightInPounds (materials : ConstructionMaterials) (rates : ConversionRates) : Float :=
  materials.concrete +
  materials.bricks * rates.kg_to_lbs +
  materials.stone +
  materials.wood * rates.kg_to_lbs +
  materials.steel * rates.oz_to_lbs +
  materials.glass * rates.ton_to_lbs +
  materials.sand

/-- Theorem stating that the total weight of materials is 60,129.72 pounds -/
theorem total_weight_proof (materials : ConstructionMaterials) (rates : ConversionRates) :
  materials.concrete = 12568.3 →
  materials.bricks = 2108 →
  materials.stone = 7099.5 →
  materials.wood = 3778 →
  materials.steel = 5879 →
  materials.glass = 12.5 →
  materials.sand = 2114.8 →
  rates.kg_to_lbs = 2.20462 →
  rates.ton_to_lbs = 2000 →
  rates.oz_to_lbs = 1/16 →
  totalWeightInPounds materials rates = 60129.72 := by
  sorry


end NUMINAMATH_CALUDE_total_weight_proof_l1776_177610


namespace NUMINAMATH_CALUDE_period_1989_points_count_l1776_177649

-- Define the unit circle
def UnitCircle : Set ℂ := {z : ℂ | Complex.abs z = 1}

-- Define the function f
def f (m : ℕ) (z : ℂ) : ℂ := z ^ m

-- Define the set of period n points
def PeriodPoints (m : ℕ) (n : ℕ) : Set ℂ :=
  {z ∈ UnitCircle | (f m)^[n] z = z ∧ ∀ k < n, (f m)^[k] z ≠ z}

-- Theorem statement
theorem period_1989_points_count (m : ℕ) (h : m > 1) :
  (PeriodPoints m 1989).ncard = m^1989 - m^663 - m^153 - m^117 + m^51 + m^39 + m^9 - m^3 := by
  sorry

end NUMINAMATH_CALUDE_period_1989_points_count_l1776_177649


namespace NUMINAMATH_CALUDE_students_liking_both_desserts_l1776_177662

theorem students_liking_both_desserts 
  (total_students : ℕ) 
  (apple_pie_lovers : ℕ) 
  (chocolate_cake_lovers : ℕ) 
  (neither_dessert_lovers : ℕ) 
  (h1 : total_students = 35)
  (h2 : apple_pie_lovers = 20)
  (h3 : chocolate_cake_lovers = 17)
  (h4 : neither_dessert_lovers = 10) :
  total_students - neither_dessert_lovers + apple_pie_lovers + chocolate_cake_lovers - total_students = 12 := by
sorry

end NUMINAMATH_CALUDE_students_liking_both_desserts_l1776_177662


namespace NUMINAMATH_CALUDE_soccer_camp_ratio_l1776_177652

/-- Soccer camp ratio problem -/
theorem soccer_camp_ratio :
  ∀ (total_kids soccer_kids : ℕ),
  total_kids = 2000 →
  soccer_kids * 3 = 750 * 4 →
  soccer_kids * 2 = total_kids :=
by
  sorry

end NUMINAMATH_CALUDE_soccer_camp_ratio_l1776_177652


namespace NUMINAMATH_CALUDE_xy_sum_over_three_l1776_177640

theorem xy_sum_over_three (x y : ℚ) 
  (eq1 : 2 * x + y = 6) 
  (eq2 : x + 2 * y = 5) : 
  (x + y) / 3 = 11 / 9 := by
  sorry

end NUMINAMATH_CALUDE_xy_sum_over_three_l1776_177640


namespace NUMINAMATH_CALUDE_difference_of_numbers_l1776_177621

theorem difference_of_numbers (x y : ℝ) (h1 : x + y = 30) (h2 : x * y = 200) : 
  |x - y| = 10 := by sorry

end NUMINAMATH_CALUDE_difference_of_numbers_l1776_177621


namespace NUMINAMATH_CALUDE_unique_solution_l1776_177626

def f (d : ℝ) (x : ℝ) : ℝ := 4 * x^3 - d * x

def g (a b c : ℝ) (x : ℝ) : ℝ := 4 * x^3 + a * x^2 + b * x + c

theorem unique_solution :
  ∃! (a b c d : ℝ),
    (∀ x ∈ Set.Icc (-1 : ℝ) 1, |f d x| ≤ 1) ∧
    (∀ x ∈ Set.Icc (-1 : ℝ) 1, |g a b c x| ≤ 1) ∧
    a = 0 ∧ b = -3 ∧ c = 0 ∧ d = 3 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_l1776_177626


namespace NUMINAMATH_CALUDE_percent_y_of_x_l1776_177655

theorem percent_y_of_x (x y : ℝ) (h : 0.2 * (x - y) = 0.14 * (x + y)) :
  y = (300 / 17) / 100 * x := by
  sorry

end NUMINAMATH_CALUDE_percent_y_of_x_l1776_177655


namespace NUMINAMATH_CALUDE_solve_for_y_l1776_177606

-- Define the variables
variable (n x y : ℝ)

-- Define the conditions
def condition1 : Prop := (n + 200 + 300 + x) / 4 = 250
def condition2 : Prop := (300 + 150 + n + x + y) / 5 = 200

-- Theorem statement
theorem solve_for_y (h1 : condition1 n x) (h2 : condition2 n x y) : y = 50 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l1776_177606


namespace NUMINAMATH_CALUDE_age_difference_patrick_nathan_l1776_177697

theorem age_difference_patrick_nathan (patrick michael monica nathan : ℝ) 
  (ratio_patrick_michael : patrick / michael = 3 / 5)
  (ratio_michael_monica : michael / monica = 3 / 4)
  (ratio_monica_nathan : monica / nathan = 5 / 7)
  (sum_ages : patrick + michael + monica + nathan = 228) :
  nathan - patrick = 69.5 := by
sorry

end NUMINAMATH_CALUDE_age_difference_patrick_nathan_l1776_177697


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l1776_177665

theorem sum_of_squares_of_roots (x : ℝ) : 
  x^2 - 10*x + 9 = 0 → ∃ s₁ s₂ : ℝ, s₁^2 + s₂^2 = 82 ∧ (x = s₁ ∨ x = s₂) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l1776_177665


namespace NUMINAMATH_CALUDE_temporary_worker_percentage_is_18_l1776_177628

/-- Represents the composition of workers in a factory -/
structure WorkerComposition where
  total : ℝ
  technician_ratio : ℝ
  non_technician_ratio : ℝ
  permanent_technician_ratio : ℝ
  permanent_non_technician_ratio : ℝ
  (total_positive : total > 0)
  (technician_ratio_valid : technician_ratio ≥ 0 ∧ technician_ratio ≤ 1)
  (non_technician_ratio_valid : non_technician_ratio ≥ 0 ∧ non_technician_ratio ≤ 1)
  (ratios_sum_to_one : technician_ratio + non_technician_ratio = 1)
  (permanent_technician_ratio_valid : permanent_technician_ratio ≥ 0 ∧ permanent_technician_ratio ≤ 1)
  (permanent_non_technician_ratio_valid : permanent_non_technician_ratio ≥ 0 ∧ permanent_non_technician_ratio ≤ 1)

/-- Calculates the percentage of temporary workers in the factory -/
def temporaryWorkerPercentage (w : WorkerComposition) : ℝ :=
  (1 - (w.technician_ratio * w.permanent_technician_ratio + w.non_technician_ratio * w.permanent_non_technician_ratio)) * 100

/-- Theorem stating that given the specific worker composition, the percentage of temporary workers is 18% -/
theorem temporary_worker_percentage_is_18 (w : WorkerComposition)
  (h1 : w.technician_ratio = 0.9)
  (h2 : w.non_technician_ratio = 0.1)
  (h3 : w.permanent_technician_ratio = 0.9)
  (h4 : w.permanent_non_technician_ratio = 0.1) :
  temporaryWorkerPercentage w = 18 := by
  sorry


end NUMINAMATH_CALUDE_temporary_worker_percentage_is_18_l1776_177628


namespace NUMINAMATH_CALUDE_binary_1101_equals_base5_23_l1776_177686

/-- Converts a binary number to decimal --/
def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

/-- Converts a decimal number to base-5 --/
def decimal_to_base5 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc else aux (m / 5) ((m % 5) :: acc)
    aux n []

/-- The binary representation of 1101 --/
def binary_1101 : List Bool := [true, false, true, true]

theorem binary_1101_equals_base5_23 :
  decimal_to_base5 (binary_to_decimal binary_1101) = [2, 3] := by
  sorry

#eval binary_to_decimal binary_1101
#eval decimal_to_base5 (binary_to_decimal binary_1101)

end NUMINAMATH_CALUDE_binary_1101_equals_base5_23_l1776_177686


namespace NUMINAMATH_CALUDE_solution_existence_unique_solution_l1776_177678

noncomputable def has_solution (a : ℝ) : Prop :=
  ∃ x : ℝ, x > 0 ∧ x ≠ 1 ∧ 2*a - x > 0 ∧
    (Real.log x / Real.log a) / (Real.log 2 / Real.log a) +
    (Real.log (2*a - x) / Real.log x) / (Real.log 2 / Real.log x) =
    1 / (Real.log 2 / Real.log (a^2 - 1))

noncomputable def has_unique_solution (a : ℝ) : Prop :=
  ∃! x : ℝ, x > 0 ∧ x ≠ 1 ∧ 2*a - x > 0 ∧
    (Real.log x / Real.log a) / (Real.log 2 / Real.log a) +
    (Real.log (2*a - x) / Real.log x) / (Real.log 2 / Real.log x) =
    1 / (Real.log 2 / Real.log (a^2 - 1))

theorem solution_existence (a : ℝ) :
  has_solution a ↔ (a > 1 ∧ a ≠ Real.sqrt 2) :=
sorry

theorem unique_solution (a : ℝ) :
  has_unique_solution a ↔ a = 2 :=
sorry

end NUMINAMATH_CALUDE_solution_existence_unique_solution_l1776_177678


namespace NUMINAMATH_CALUDE_sticker_ratio_l1776_177689

theorem sticker_ratio : 
  ∀ (dan_stickers tom_stickers bob_stickers : ℕ),
    dan_stickers = tom_stickers →
    tom_stickers = 3 * bob_stickers →
    bob_stickers = 12 →
    dan_stickers = 72 →
    dan_stickers / tom_stickers = 2 := by
  sorry

end NUMINAMATH_CALUDE_sticker_ratio_l1776_177689


namespace NUMINAMATH_CALUDE_expression_value_l1776_177682

theorem expression_value (x y : ℝ) (h : x^2 - 2*y = -1) : 
  3*x^2 - 6*y + 2023 = 2020 := by
sorry

end NUMINAMATH_CALUDE_expression_value_l1776_177682


namespace NUMINAMATH_CALUDE_largest_n_for_inequality_l1776_177633

theorem largest_n_for_inequality : ∃ (n : ℕ), n = 2 ∧ 
  (∀ (a b c d : ℝ), 
    (n + 2) * Real.sqrt (a^2 + b^2) + (n + 1) * Real.sqrt (a^2 + c^2) + (n + 1) * Real.sqrt (a^2 + d^2) ≥ n * (a + b + c + d)) ∧
  (∀ (m : ℕ), m > n → 
    ∃ (a b c d : ℝ), 
      (m + 2) * Real.sqrt (a^2 + b^2) + (m + 1) * Real.sqrt (a^2 + c^2) + (m + 1) * Real.sqrt (a^2 + d^2) < m * (a + b + c + d)) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_for_inequality_l1776_177633


namespace NUMINAMATH_CALUDE_arithmetic_triangle_inradius_l1776_177632

/-- A triangle with sides in arithmetic progression and an inscribed circle -/
structure ArithmeticTriangle where
  -- The three sides of the triangle
  a : ℝ
  b : ℝ
  c : ℝ
  -- The sides form an arithmetic progression
  progression : ∃ d : ℝ, b = a + d ∧ c = a + 2*d
  -- The triangle is valid (sum of any two sides is greater than the third)
  valid : a + b > c ∧ b + c > a ∧ c + a > b
  -- The triangle has positive area
  positive_area : a > 0 ∧ b > 0 ∧ c > 0
  -- The inscribed circle exists
  inradius : ℝ
  -- One of the altitudes
  altitude : ℝ

/-- 
The radius of the inscribed circle of a triangle with sides in arithmetic progression 
is equal to 1/3 of one of its altitudes
-/
theorem arithmetic_triangle_inradius (t : ArithmeticTriangle) : 
  t.inradius = (1/3) * t.altitude := by sorry

end NUMINAMATH_CALUDE_arithmetic_triangle_inradius_l1776_177632


namespace NUMINAMATH_CALUDE_tinas_hourly_wage_l1776_177641

/-- Represents Tina's work schedule and pay structure -/
structure WorkSchedule where
  regularHours : ℕ := 8
  overtimeRate : ℚ := 3/2
  daysWorked : ℕ := 5
  hoursPerDay : ℕ := 10
  totalPay : ℚ := 990

/-- Calculates Tina's hourly wage based on her work schedule -/
def calculateHourlyWage (schedule : WorkSchedule) : ℚ :=
  let regularHoursPerWeek := schedule.regularHours * schedule.daysWorked
  let overtimeHoursPerWeek := (schedule.hoursPerDay - schedule.regularHours) * schedule.daysWorked
  let totalHoursEquivalent := regularHoursPerWeek + overtimeHoursPerWeek * schedule.overtimeRate
  schedule.totalPay / totalHoursEquivalent

/-- Theorem stating that Tina's hourly wage is $18 -/
theorem tinas_hourly_wage (schedule : WorkSchedule) : 
  calculateHourlyWage schedule = 18 := by
  sorry

#eval calculateHourlyWage {} -- Should output 18

end NUMINAMATH_CALUDE_tinas_hourly_wage_l1776_177641


namespace NUMINAMATH_CALUDE_johnny_earnings_l1776_177608

/-- Calculates the total earnings from two jobs with overtime --/
def total_earnings (
  job1_rate : ℚ)
  (job1_hours : ℕ)
  (job1_regular_hours : ℕ)
  (job1_overtime_multiplier : ℚ)
  (job2_rate : ℚ)
  (job2_hours : ℕ) : ℚ :=
  let job1_regular_pay := job1_rate * (job1_regular_hours : ℚ)
  let job1_overtime_hours := job1_hours - job1_regular_hours
  let job1_overtime_rate := job1_rate * job1_overtime_multiplier
  let job1_overtime_pay := job1_overtime_rate * (job1_overtime_hours : ℚ)
  let job1_total_pay := job1_regular_pay + job1_overtime_pay
  let job2_pay := job2_rate * (job2_hours : ℚ)
  job1_total_pay + job2_pay

/-- Johnny's total earnings from two jobs with overtime --/
theorem johnny_earnings : 
  total_earnings 3.25 8 6 1.5 4.5 5 = 58.25 := by
  sorry

end NUMINAMATH_CALUDE_johnny_earnings_l1776_177608


namespace NUMINAMATH_CALUDE_divisibility_problem_l1776_177696

theorem divisibility_problem :
  (∃ (a b : Nat), a < 10 ∧ b < 10 ∧
    (∀ n : Nat, 73 ∣ (10 * a + b) * 10^n + (200 * 10^n + 79) / 9)) ∧
  (¬ ∃ (c d : Nat), c < 10 ∧ d < 10 ∧
    (∀ n : Nat, 79 ∣ (10 * c + d) * 10^n + (200 * 10^n + 79) / 9)) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_problem_l1776_177696


namespace NUMINAMATH_CALUDE_intersection_point_l1776_177688

/-- Two lines in a 2D plane -/
structure TwoLines where
  line1 : ℝ → ℝ → Prop
  line2 : ℝ → ℝ → Prop

/-- The given two lines -/
def givenLines : TwoLines where
  line1 := fun x y => x - y = 0
  line2 := fun x y => 3 * x + 2 * y - 5 = 0

/-- Theorem: The point (1, 1) is the unique intersection of the given lines -/
theorem intersection_point (l : TwoLines := givenLines) :
  (∃! p : ℝ × ℝ, l.line1 p.1 p.2 ∧ l.line2 p.1 p.2) ∧
  (l.line1 1 1 ∧ l.line2 1 1) :=
sorry

end NUMINAMATH_CALUDE_intersection_point_l1776_177688


namespace NUMINAMATH_CALUDE_prime_sum_problem_l1776_177669

theorem prime_sum_problem (p q r : ℕ) : 
  Nat.Prime p → Nat.Prime q → Nat.Prime r →
  p + q = r + 2 →
  1 < p →
  p < q →
  p = 2 := by
sorry

end NUMINAMATH_CALUDE_prime_sum_problem_l1776_177669


namespace NUMINAMATH_CALUDE_min_sum_abc_l1776_177675

theorem min_sum_abc (a b c : ℕ+) (h : a.val * b.val * c.val + b.val * c.val + c.val = 2014) :
  ∃ (a' b' c' : ℕ+), 
    a'.val * b'.val * c'.val + b'.val * c'.val + c'.val = 2014 ∧
    a'.val + b'.val + c'.val = 40 ∧
    ∀ (x y z : ℕ+), x.val * y.val * z.val + y.val * z.val + z.val = 2014 → 
      x.val + y.val + z.val ≥ 40 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_abc_l1776_177675


namespace NUMINAMATH_CALUDE_infinite_series_sum_l1776_177634

theorem infinite_series_sum : 
  let series := fun n : ℕ => (n + 1 : ℝ) / 5^n
  ∑' n, series n = 9/16 := by sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l1776_177634


namespace NUMINAMATH_CALUDE_eleven_motorcycles_in_lot_l1776_177615

/-- Represents the number of motorcycles in a parking lot --/
def motorcycles_in_lot (total_wheels car_count : ℕ) : ℕ :=
  (total_wheels - 5 * car_count) / 2

/-- Theorem: Given the conditions in the problem, there are 11 motorcycles in the parking lot --/
theorem eleven_motorcycles_in_lot :
  motorcycles_in_lot 117 19 = 11 := by
  sorry

#eval motorcycles_in_lot 117 19

end NUMINAMATH_CALUDE_eleven_motorcycles_in_lot_l1776_177615


namespace NUMINAMATH_CALUDE_intersection_point_l1776_177614

/-- The point (3, 2) is the unique solution to the system of equations x + y = 5 and x - y = 1 -/
theorem intersection_point : ∃! p : ℝ × ℝ, p.1 + p.2 = 5 ∧ p.1 - p.2 = 1 ∧ p = (3, 2) := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_l1776_177614


namespace NUMINAMATH_CALUDE_uniform_price_is_250_l1776_177623

/-- Represents the agreement between an employer and a servant --/
structure Agreement where
  full_year_salary : ℕ
  uniform_included : Bool

/-- Represents the actual outcome of the servant's employment --/
structure Outcome where
  months_worked : ℕ
  salary_received : ℕ
  uniform_received : Bool

/-- Calculates the price of the uniform given the agreement and outcome --/
def uniform_price (agreement : Agreement) (outcome : Outcome) : ℕ :=
  agreement.full_year_salary - outcome.salary_received

/-- Theorem stating that under the given conditions, the uniform price is 250 --/
theorem uniform_price_is_250 (agreement : Agreement) (outcome : Outcome) :
  agreement.full_year_salary = 500 ∧
  agreement.uniform_included = true ∧
  outcome.months_worked = 9 ∧
  outcome.salary_received = 250 ∧
  outcome.uniform_received = true →
  uniform_price agreement outcome = 250 := by
  sorry

#eval uniform_price
  { full_year_salary := 500, uniform_included := true }
  { months_worked := 9, salary_received := 250, uniform_received := true }

end NUMINAMATH_CALUDE_uniform_price_is_250_l1776_177623


namespace NUMINAMATH_CALUDE_blue_ball_probability_l1776_177602

/-- The probability of selecting 3 blue balls from a jar containing 6 red and 4 blue balls -/
theorem blue_ball_probability (red_balls blue_balls selected : ℕ) 
  (h1 : red_balls = 6)
  (h2 : blue_balls = 4)
  (h3 : selected = 3) :
  (Nat.choose blue_balls selected) / (Nat.choose (red_balls + blue_balls) selected) = 1 / 30 := by
  sorry

end NUMINAMATH_CALUDE_blue_ball_probability_l1776_177602


namespace NUMINAMATH_CALUDE_runners_in_picture_probability_l1776_177600

/-- Represents a runner on a circular track -/
structure Runner where
  lapTime : ℝ
  direction : Bool  -- True for counterclockwise, False for clockwise

/-- Represents the track and photograph setup -/
structure TrackSetup where
  rachelLapTime : ℝ
  robertLapTime : ℝ
  totalTime : ℝ
  photographerPosition : ℝ
  pictureWidth : ℝ

/-- Calculates the probability of both runners being in the picture -/
def probabilityBothInPicture (setup : TrackSetup) : ℝ :=
  sorry  -- Proof omitted

theorem runners_in_picture_probability (setup : TrackSetup) 
  (h1 : setup.rachelLapTime = 75)
  (h2 : setup.robertLapTime = 100)
  (h3 : setup.totalTime = 12 * 60)
  (h4 : setup.photographerPosition = 1/3)
  (h5 : setup.pictureWidth = 1/5) :
  probabilityBothInPicture setup = 4/15 := by
  sorry

#check runners_in_picture_probability

end NUMINAMATH_CALUDE_runners_in_picture_probability_l1776_177600


namespace NUMINAMATH_CALUDE_cheap_coat_duration_proof_l1776_177637

/-- The duration of the less expensive coat -/
def cheap_coat_duration : ℕ := 5

/-- The cost of the expensive coat -/
def expensive_coat_cost : ℕ := 300

/-- The duration of the expensive coat -/
def expensive_coat_duration : ℕ := 15

/-- The cost of the less expensive coat -/
def cheap_coat_cost : ℕ := 120

/-- The total time period considered -/
def total_time : ℕ := 30

/-- The amount saved by buying the expensive coat over the total time period -/
def amount_saved : ℕ := 120

theorem cheap_coat_duration_proof :
  cheap_coat_duration * cheap_coat_cost * (total_time / cheap_coat_duration) =
  expensive_coat_cost * (total_time / expensive_coat_duration) + amount_saved :=
by sorry

end NUMINAMATH_CALUDE_cheap_coat_duration_proof_l1776_177637


namespace NUMINAMATH_CALUDE_composite_function_evaluation_l1776_177695

def f (x : ℝ) : ℝ := 5 * x + 2

def g (x : ℝ) : ℝ := 3 * x + 4

theorem composite_function_evaluation :
  f (g (f 3)) = 277 :=
by sorry

end NUMINAMATH_CALUDE_composite_function_evaluation_l1776_177695


namespace NUMINAMATH_CALUDE_yellow_marbles_count_l1776_177639

theorem yellow_marbles_count (yellow blue : ℕ) 
  (h1 : blue = yellow - 2)
  (h2 : yellow + blue = 240) : 
  yellow = 121 := by
sorry

end NUMINAMATH_CALUDE_yellow_marbles_count_l1776_177639


namespace NUMINAMATH_CALUDE_floor_plus_x_equals_seventeen_fourths_l1776_177635

theorem floor_plus_x_equals_seventeen_fourths :
  ∃ x : ℚ, (⌊x⌋ : ℚ) + x = 17/4 ∧ x = 9/4 := by sorry

end NUMINAMATH_CALUDE_floor_plus_x_equals_seventeen_fourths_l1776_177635


namespace NUMINAMATH_CALUDE_min_typical_parallelepipeds_is_four_l1776_177667

/-- A typical parallelepiped has all dimensions different -/
structure TypicalParallelepiped where
  length : ℝ
  width : ℝ
  height : ℝ
  all_different : length ≠ width ∧ width ≠ height ∧ length ≠ height

/-- A cube with side length s -/
structure Cube where
  side : ℝ

/-- The minimum number of typical parallelepipeds into which a cube can be cut -/
def min_typical_parallelepipeds_in_cube (c : Cube) : ℕ :=
  4

/-- Theorem stating that the minimum number of typical parallelepipeds 
    into which a cube can be cut is 4 -/
theorem min_typical_parallelepipeds_is_four (c : Cube) :
  min_typical_parallelepipeds_in_cube c = 4 := by
  sorry

end NUMINAMATH_CALUDE_min_typical_parallelepipeds_is_four_l1776_177667


namespace NUMINAMATH_CALUDE_arithmetic_mean_after_removal_l1776_177647

theorem arithmetic_mean_after_removal (s : Finset ℕ) (a : ℕ → ℝ) :
  Finset.card s = 75 →
  (Finset.sum s a) / 75 = 60 →
  72 ∈ s →
  48 ∈ s →
  let s' := s.erase 72 ∩ s.erase 48
  (Finset.sum s' a) / 73 = 60 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_after_removal_l1776_177647


namespace NUMINAMATH_CALUDE_equal_roots_k_value_l1776_177670

/-- The cubic equation with parameter k -/
def cubic_equation (x k : ℝ) : ℝ :=
  3 * x^3 + 9 * x^2 - 162 * x + k

/-- Theorem stating that if the cubic equation has two equal roots and k is positive, then k = 7983/125 -/
theorem equal_roots_k_value (k : ℝ) :
  (∃ a b : ℝ, a ≠ b ∧
    cubic_equation a k = 0 ∧
    cubic_equation b k = 0 ∧
    (∃ x : ℝ, x ≠ a ∧ x ≠ b ∧ cubic_equation x k = 0)) →
  k > 0 →
  k = 7983 / 125 := by
  sorry

end NUMINAMATH_CALUDE_equal_roots_k_value_l1776_177670


namespace NUMINAMATH_CALUDE_course_selection_problem_l1776_177648

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The total number of ways two people can choose 2 courses each from 4 courses -/
def totalWays : ℕ :=
  choose 4 2 * choose 4 2

/-- The number of ways two people can choose 2 courses each from 4 courses with at least one course in common -/
def waysWithCommon : ℕ :=
  totalWays - choose 4 2

theorem course_selection_problem :
  (totalWays = 36) ∧
  (waysWithCommon / totalWays = 5 / 6) := by sorry

end NUMINAMATH_CALUDE_course_selection_problem_l1776_177648


namespace NUMINAMATH_CALUDE_simplify_fraction_l1776_177666

theorem simplify_fraction : (84 : ℚ) / 1764 * 21 = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1776_177666


namespace NUMINAMATH_CALUDE_range_of_a_l1776_177607

-- Define the propositions P and Q as functions of a
def P (a : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*a*x + 4 > 0

def Q (a : ℝ) : Prop := ∀ x y : ℝ, x < y → (5 - 2*a)^x < (5 - 2*a)^y

-- Define the theorem
theorem range_of_a : 
  (∀ a : ℝ, (P a ∨ Q a) ∧ ¬(P a ∧ Q a)) → 
  {a : ℝ | a ≤ -2} = {a : ℝ | ∀ a' : ℝ, (P a' ∨ Q a') ∧ ¬(P a' ∧ Q a') → a ≤ a'} :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1776_177607


namespace NUMINAMATH_CALUDE_value_of_expression_l1776_177656

theorem value_of_expression (x : ℝ) (h : x = 5) : 3 * x + 4 = 19 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l1776_177656


namespace NUMINAMATH_CALUDE_bottle_caps_problem_l1776_177625

theorem bottle_caps_problem (sammy janine billie : ℕ) 
  (h1 : sammy = 8)
  (h2 : sammy = janine + 2)
  (h3 : janine = 3 * billie) :
  billie = 2 := by
  sorry

end NUMINAMATH_CALUDE_bottle_caps_problem_l1776_177625


namespace NUMINAMATH_CALUDE_conclusion_l1776_177638

-- Define the variables
variable (p q r s u v : ℝ)

-- State the given conditions
axiom cond1 : p > q → r > s
axiom cond2 : r = s → u < v
axiom cond3 : p = q → s > r

-- State the theorem to be proved
theorem conclusion : p ≠ q → s ≠ r := by
  sorry

end NUMINAMATH_CALUDE_conclusion_l1776_177638


namespace NUMINAMATH_CALUDE_cos_sin_relation_l1776_177661

theorem cos_sin_relation (α : ℝ) (h : Real.cos (α - π/5) = 5/13) :
  Real.sin (α - 7*π/10) = -5/13 := by sorry

end NUMINAMATH_CALUDE_cos_sin_relation_l1776_177661


namespace NUMINAMATH_CALUDE_poles_inside_base_l1776_177685

/-- A non-convex polygon representing the fence -/
structure Fence where
  isNonConvex : Bool

/-- A power line with poles -/
structure PowerLine where
  totalPoles : Nat

/-- A spy walking around the fence -/
structure Spy where
  totalCount : Nat

/-- The secret base surrounded by the fence -/
structure Base where
  polesInside : Nat

/-- Theorem stating the number of poles inside the base -/
theorem poles_inside_base 
  (fence : Fence) 
  (powerLine : PowerLine)
  (spy : Spy) :
  fence.isNonConvex = true →
  powerLine.totalPoles = 36 →
  spy.totalCount = 2015 →
  ∃ (base : Base), base.polesInside = 1 := by
  sorry

end NUMINAMATH_CALUDE_poles_inside_base_l1776_177685


namespace NUMINAMATH_CALUDE_rationalize_denominator_l1776_177629

theorem rationalize_denominator :
  ∃ (A B C D E : ℤ),
    (B < D) ∧
    (3 : ℚ) / (2 * Real.sqrt 18 + 5 * Real.sqrt 20) =
      (A * Real.sqrt B + C * Real.sqrt D) / E ∧
    A = -18 ∧ B = 2 ∧ C = 30 ∧ D = 5 ∧ E = 428 :=
by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l1776_177629


namespace NUMINAMATH_CALUDE_product_zero_implies_factor_zero_unit_circle_sum_one_implies_diff_sqrt_three_l1776_177651

variables (z₁ z₂ : ℂ)

-- Statement B
theorem product_zero_implies_factor_zero : z₁ * z₂ = 0 → z₁ = 0 ∨ z₂ = 0 := by sorry

-- Statement D
theorem unit_circle_sum_one_implies_diff_sqrt_three : 
  Complex.abs z₁ = 1 → Complex.abs z₂ = 1 → z₁ + z₂ = 1 → Complex.abs (z₁ - z₂) = Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_product_zero_implies_factor_zero_unit_circle_sum_one_implies_diff_sqrt_three_l1776_177651


namespace NUMINAMATH_CALUDE_age_ratio_proof_l1776_177671

/-- Given three people a, b, and c, prove that the ratio of b's age to c's age is 2:1 -/
theorem age_ratio_proof (a b c : ℕ) : 
  a = b + 2 →  -- a is two years older than b
  ∃ k : ℕ, b = k * c →  -- b is some multiple of c's age
  a + b + c = 32 →  -- The total of the ages of a, b, and c is 32
  b = 12 →  -- b is 12 years old
  b = 2 * c  -- The ratio of b's age to c's age is 2:1
:= by sorry

end NUMINAMATH_CALUDE_age_ratio_proof_l1776_177671


namespace NUMINAMATH_CALUDE_half_angle_in_second_quadrant_l1776_177664

open Real

/-- An angle is in the third quadrant if it's between π and 3π/2 -/
def in_third_quadrant (θ : ℝ) : Prop := π < θ ∧ θ < 3*π/2

/-- An angle is in the second quadrant if it's between π/2 and π -/
def in_second_quadrant (θ : ℝ) : Prop := π/2 < θ ∧ θ < π

theorem half_angle_in_second_quadrant (θ : ℝ) 
  (h1 : in_third_quadrant θ) 
  (h2 : |cos θ| = -cos (θ/2)) : 
  in_second_quadrant (θ/2) := by
  sorry

end NUMINAMATH_CALUDE_half_angle_in_second_quadrant_l1776_177664


namespace NUMINAMATH_CALUDE_exponent_equality_l1776_177609

theorem exponent_equality (x : ℝ) (n : ℤ) : x * x^(3*n) = x^(3*n + 1) := by
  sorry

end NUMINAMATH_CALUDE_exponent_equality_l1776_177609


namespace NUMINAMATH_CALUDE_coefficient_x_cubed_in_expansion_l1776_177687

theorem coefficient_x_cubed_in_expansion :
  let n : ℕ := 5
  let a : ℤ := 3
  let r : ℕ := 3
  let coeff : ℤ := (n.choose r) * a^(n-r) * (-1)^r
  coeff = -90 := by sorry

end NUMINAMATH_CALUDE_coefficient_x_cubed_in_expansion_l1776_177687


namespace NUMINAMATH_CALUDE_triangle_side_length_l1776_177692

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  A = 30 * π / 180 →
  B = 45 * π / 180 →
  b = 8 →
  a / Real.sin A = b / Real.sin B →
  a = 4 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1776_177692


namespace NUMINAMATH_CALUDE_vegetable_planting_methods_l1776_177603

/-- The number of vegetable types available --/
def total_vegetables : ℕ := 4

/-- The number of vegetable types to be chosen --/
def chosen_vegetables : ℕ := 3

/-- The number of soil types --/
def soil_types : ℕ := 3

/-- The number of vegetables to be chosen excluding cucumber --/
def vegetables_to_choose : ℕ := chosen_vegetables - 1

/-- The number of remaining vegetables to choose from --/
def remaining_vegetables : ℕ := total_vegetables - 1

theorem vegetable_planting_methods :
  (Nat.choose remaining_vegetables vegetables_to_choose) * (Nat.factorial chosen_vegetables) = 18 :=
sorry

end NUMINAMATH_CALUDE_vegetable_planting_methods_l1776_177603


namespace NUMINAMATH_CALUDE_bottle_caps_count_l1776_177673

theorem bottle_caps_count (initial_caps : ℕ) (added_caps : ℕ) : 
  initial_caps = 7 → added_caps = 7 → initial_caps + added_caps = 14 := by
  sorry

end NUMINAMATH_CALUDE_bottle_caps_count_l1776_177673


namespace NUMINAMATH_CALUDE_banana_arrangements_count_l1776_177658

/-- The number of distinct arrangements of the letters in "BANANA" -/
def banana_arrangements : ℕ := 60

/-- The total number of letters in "BANANA" -/
def total_letters : ℕ := 6

/-- The number of occurrences of 'A' in "BANANA" -/
def count_A : ℕ := 3

/-- The number of occurrences of 'N' in "BANANA" -/
def count_N : ℕ := 2

/-- The number of occurrences of 'B' in "BANANA" -/
def count_B : ℕ := 1

/-- Theorem stating that the number of distinct arrangements of the letters in "BANANA" is 60 -/
theorem banana_arrangements_count :
  banana_arrangements = (Nat.factorial total_letters) / ((Nat.factorial count_A) * (Nat.factorial count_N)) :=
by sorry

end NUMINAMATH_CALUDE_banana_arrangements_count_l1776_177658


namespace NUMINAMATH_CALUDE_residual_plot_vertical_axis_l1776_177694

/-- Represents a residual plot in regression analysis -/
structure ResidualPlot where
  verticalAxis : Set ℝ
  horizontalAxis : Set ℝ

/-- Definition of a residual in regression analysis -/
def Residual : Type := ℝ

/-- Theorem stating that the vertical axis of a residual plot represents residuals -/
theorem residual_plot_vertical_axis (plot : ResidualPlot) : 
  plot.verticalAxis = Set.range (λ r : Residual => r) := by
  sorry

end NUMINAMATH_CALUDE_residual_plot_vertical_axis_l1776_177694


namespace NUMINAMATH_CALUDE_solve_equation_l1776_177674

theorem solve_equation (x : ℝ) : (3 * x + 36 = 48) → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1776_177674


namespace NUMINAMATH_CALUDE_two_zeros_neither_necessary_nor_sufficient_l1776_177642

open Real

-- Define the function f and its derivative f'
variable (f : ℝ → ℝ) (f' : ℝ → ℝ)

-- Define the property of f' having exactly two zeros in (0, 2)
def has_two_zeros (f' : ℝ → ℝ) : Prop :=
  ∃ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 2 ∧ f' x₁ = 0 ∧ f' x₂ = 0 ∧
  ∀ x, 0 < x ∧ x < 2 ∧ f' x = 0 → x = x₁ ∨ x = x₂

-- Define the property of f having exactly two extreme points in (0, 2)
def has_two_extreme_points (f : ℝ → ℝ) (f' : ℝ → ℝ) : Prop :=
  ∃ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 2 ∧ f' x₁ = 0 ∧ f' x₂ = 0 ∧
  (∀ x, 0 < x ∧ x < x₁ → f' x ≠ 0) ∧
  (∀ x, x₁ < x ∧ x < x₂ → f' x ≠ 0) ∧
  (∀ x, x₂ < x ∧ x < 2 → f' x ≠ 0)

-- Theorem stating that has_two_zeros is neither necessary nor sufficient for has_two_extreme_points
theorem two_zeros_neither_necessary_nor_sufficient :
  ¬(∀ f f', has_two_zeros f' → has_two_extreme_points f f') ∧
  ¬(∀ f f', has_two_extreme_points f f' → has_two_zeros f') :=
sorry

end NUMINAMATH_CALUDE_two_zeros_neither_necessary_nor_sufficient_l1776_177642


namespace NUMINAMATH_CALUDE_max_sum_cubes_max_sum_cubes_achieved_l1776_177624

theorem max_sum_cubes (a b c d e : ℝ) (h : a^2 + b^2 + c^2 + d^2 + e^2 = 5) :
  a^3 + b^3 + c^3 + d^3 + e^3 ≤ 5 * Real.sqrt 5 :=
by sorry

theorem max_sum_cubes_achieved (h : ∃ a b c d e : ℝ, a^2 + b^2 + c^2 + d^2 + e^2 = 5 ∧ a^3 + b^3 + c^3 + d^3 + e^3 = 5 * Real.sqrt 5) :
  ∃ a b c d e : ℝ, a^2 + b^2 + c^2 + d^2 + e^2 = 5 ∧ a^3 + b^3 + c^3 + d^3 + e^3 = 5 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_cubes_max_sum_cubes_achieved_l1776_177624


namespace NUMINAMATH_CALUDE_number_difference_l1776_177605

theorem number_difference (N : ℝ) (h : 0.25 * N = 100) : N - (3/4 * N) = 100 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_l1776_177605


namespace NUMINAMATH_CALUDE_correct_operation_l1776_177611

theorem correct_operation (a b : ℝ) : 3*a + (a - 3*b) = 4*a - 3*b := by sorry

end NUMINAMATH_CALUDE_correct_operation_l1776_177611


namespace NUMINAMATH_CALUDE_number_problem_l1776_177657

theorem number_problem (x : ℝ) : (0.40 * x = 0.80 * 5 + 2) → x = 15 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l1776_177657


namespace NUMINAMATH_CALUDE_quadratic_square_plus_constant_l1776_177672

theorem quadratic_square_plus_constant :
  ∃ k : ℤ, ∀ z : ℂ, z^2 - 6*z + 17 = (z - 3)^2 + k := by
  sorry

end NUMINAMATH_CALUDE_quadratic_square_plus_constant_l1776_177672
