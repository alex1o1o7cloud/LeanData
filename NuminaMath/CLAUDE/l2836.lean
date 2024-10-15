import Mathlib

namespace NUMINAMATH_CALUDE_mold_growth_problem_l2836_283672

/-- Calculates the number of mold spores after a given time period -/
def mold_growth (initial_spores : ℕ) (doubling_time : ℕ) (elapsed_time : ℕ) : ℕ :=
  initial_spores * 2^(elapsed_time / doubling_time)

/-- The mold growth problem -/
theorem mold_growth_problem :
  let initial_spores : ℕ := 50
  let doubling_time : ℕ := 10  -- in minutes
  let elapsed_time : ℕ := 70   -- time from 9:00 a.m. to 10:10 a.m. in minutes
  mold_growth initial_spores doubling_time elapsed_time = 6400 :=
by
  sorry

end NUMINAMATH_CALUDE_mold_growth_problem_l2836_283672


namespace NUMINAMATH_CALUDE_consecutive_numbers_sequence_l2836_283636

def is_valid_sequence (a b : ℕ) : Prop :=
  let n := b - a + 1
  let sum := n * (a + b) / 2
  let mean := sum / n
  let sum_without_122_123 := sum - 122 - 123
  let mean_without_122_123 := sum_without_122_123 / (n - 2)
  (mean = 85) ∧
  (mean = (70 + 82 + 103) / 3) ∧
  (mean_without_122_123 + 1 = mean) ∧
  (a = 47) ∧
  (b = 123)

theorem consecutive_numbers_sequence :
  ∃ a b : ℕ, is_valid_sequence a b :=
sorry

end NUMINAMATH_CALUDE_consecutive_numbers_sequence_l2836_283636


namespace NUMINAMATH_CALUDE_nested_fraction_equality_l2836_283640

theorem nested_fraction_equality : 
  1 + 1 / (1 - 1 / (2 + 1 / 3)) = 11 / 4 := by sorry

end NUMINAMATH_CALUDE_nested_fraction_equality_l2836_283640


namespace NUMINAMATH_CALUDE_starting_number_proof_l2836_283630

theorem starting_number_proof (x : ℕ) : 
  (x ≤ 26) → 
  (x % 2 = 0) → 
  ((x + 26) / 2 = 19) → 
  x = 12 := by
sorry

end NUMINAMATH_CALUDE_starting_number_proof_l2836_283630


namespace NUMINAMATH_CALUDE_revenue_comparison_l2836_283653

theorem revenue_comparison (base_revenue : ℝ) (projected_increase : ℝ) (actual_decrease : ℝ) :
  projected_increase = 0.2 →
  actual_decrease = 0.25 →
  (base_revenue * (1 - actual_decrease)) / (base_revenue * (1 + projected_increase)) = 0.625 := by
  sorry

end NUMINAMATH_CALUDE_revenue_comparison_l2836_283653


namespace NUMINAMATH_CALUDE_base6_addition_l2836_283632

/-- Convert a number from base 6 to base 10 --/
def base6ToBase10 (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (6 ^ i)) 0

/-- Convert a number from base 10 to base 6 --/
def base10ToBase6 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc else aux (m / 6) ((m % 6) :: acc)
  aux n []

/-- Definition of the first number in base 6 --/
def num1 : List Nat := [2, 3, 5, 4]

/-- Definition of the second number in base 6 --/
def num2 : List Nat := [3, 5, 2, 4, 2]

/-- Theorem stating that the sum of the two numbers in base 6 equals the result --/
theorem base6_addition :
  base10ToBase6 (base6ToBase10 num1 + base6ToBase10 num2) = [5, 2, 2, 2, 3, 3] := by sorry

end NUMINAMATH_CALUDE_base6_addition_l2836_283632


namespace NUMINAMATH_CALUDE_plywood_cut_perimeter_difference_l2836_283647

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

theorem plywood_cut_perimeter_difference :
  let original : Rectangle := { length := 9, width := 6 }
  let pieces : ℕ := 4
  ∃ (max_piece min_piece : Rectangle),
    (pieces * max_piece.length * max_piece.width = original.length * original.width) ∧
    (pieces * min_piece.length * min_piece.width = original.length * original.width) ∧
    (∀ piece : Rectangle, 
      (pieces * piece.length * piece.width = original.length * original.width) → 
      (perimeter piece ≤ perimeter max_piece ∧ perimeter piece ≥ perimeter min_piece)) ∧
    (perimeter max_piece - perimeter min_piece = 9) := by
  sorry

end NUMINAMATH_CALUDE_plywood_cut_perimeter_difference_l2836_283647


namespace NUMINAMATH_CALUDE_line_slope_l2836_283654

theorem line_slope (x y : ℝ) (h : x + 2 * y - 3 = 0) : 
  ∃ m b : ℝ, m = -1/2 ∧ y = m * x + b :=
sorry

end NUMINAMATH_CALUDE_line_slope_l2836_283654


namespace NUMINAMATH_CALUDE_fourth_grade_students_l2836_283674

theorem fourth_grade_students (initial : ℕ) (left : ℕ) (new : ℕ) (final : ℕ) : 
  initial = 31 → left = 5 → new = 11 → final = initial - left + new → final = 37 := by
  sorry

end NUMINAMATH_CALUDE_fourth_grade_students_l2836_283674


namespace NUMINAMATH_CALUDE_original_number_proof_l2836_283608

theorem original_number_proof (x : ℝ) : 
  268 * x = 19832 ∧ 2.68 * x = 1.9832 → x = 74 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l2836_283608


namespace NUMINAMATH_CALUDE_hyperbola_theorem_l2836_283637

/-- A hyperbola C that shares a common asymptote with x^2 - 2y^2 = 2 and passes through (2, -2) -/
structure Hyperbola where
  -- The equation of the hyperbola in the form y^2/a^2 - x^2/b^2 = 1
  a : ℝ
  b : ℝ
  -- The hyperbola passes through (2, -2)
  point_condition : (2 : ℝ)^2 / b^2 - (-2 : ℝ)^2 / a^2 = 1
  -- The hyperbola shares a common asymptote with x^2 - 2y^2 = 2
  asymptote_condition : a^2 / b^2 = 2

/-- Properties of the hyperbola C -/
def hyperbola_properties (C : Hyperbola) : Prop :=
  -- The equation of C is y^2/2 - x^2/4 = 1
  C.a^2 = 2 ∧ C.b^2 = 4 ∧
  -- The eccentricity of C is √3
  Real.sqrt ((C.a^2 + C.b^2) / C.a^2) = Real.sqrt 3 ∧
  -- The asymptotes of C are y = ±(√2/2)x
  ∀ (x y : ℝ), (y = Real.sqrt 2 / 2 * x ∨ y = -Real.sqrt 2 / 2 * x) ↔ 
    (y^2 / C.a^2 - x^2 / C.b^2 = 0)

/-- Main theorem: The hyperbola C satisfies the required properties -/
theorem hyperbola_theorem (C : Hyperbola) : hyperbola_properties C :=
sorry

end NUMINAMATH_CALUDE_hyperbola_theorem_l2836_283637


namespace NUMINAMATH_CALUDE_anthony_tax_deduction_l2836_283610

/-- Calculates the total tax deduction in cents given an hourly wage and tax rates -/
def totalTaxDeduction (hourlyWage : ℚ) (federalTaxRate : ℚ) (stateTaxRate : ℚ) : ℚ :=
  hourlyWage * 100 * (federalTaxRate + stateTaxRate)

/-- Theorem: Given Anthony's wage and tax rates, the total tax deduction is 62.5 cents -/
theorem anthony_tax_deduction :
  totalTaxDeduction 25 (2/100) (1/200) = 125/2 := by
  sorry

end NUMINAMATH_CALUDE_anthony_tax_deduction_l2836_283610


namespace NUMINAMATH_CALUDE_tom_has_two_yellow_tickets_l2836_283618

/-- Represents the number of tickets Tom has -/
structure TicketHoldings where
  yellow : ℕ
  red : ℕ
  blue : ℕ

/-- Conversion rates between ticket types -/
def yellow_to_red : ℕ := 10
def red_to_blue : ℕ := 10

/-- The number of additional blue tickets Tom needs -/
def additional_blue_needed : ℕ := 163

/-- Tom's current ticket holdings -/
def toms_tickets : TicketHoldings := {
  yellow := 0,  -- We don't know this value yet, so we set it to 0
  red := 3,
  blue := 7
}

/-- Theorem stating that Tom has 2 yellow tickets -/
theorem tom_has_two_yellow_tickets :
  ∃ (y : ℕ), 
    y * (yellow_to_red * red_to_blue) + 
    toms_tickets.red * red_to_blue + 
    toms_tickets.blue + 
    additional_blue_needed = 
    2 * (yellow_to_red * red_to_blue) ∧
    y = 2 := by
  sorry


end NUMINAMATH_CALUDE_tom_has_two_yellow_tickets_l2836_283618


namespace NUMINAMATH_CALUDE_matrix_equation_l2836_283686

def A : Matrix (Fin 2) (Fin 2) ℚ := !![2, -5; 4, -3]
def B : Matrix (Fin 2) (Fin 2) ℚ := !![-20, 10; 8, -4]
def N : Matrix (Fin 2) (Fin 2) ℚ := !![10/7, -40/7; -4/7, 16/7]

theorem matrix_equation : N * A = B := by sorry

end NUMINAMATH_CALUDE_matrix_equation_l2836_283686


namespace NUMINAMATH_CALUDE_floor_painting_dimensions_l2836_283657

theorem floor_painting_dimensions :
  ∀ (a b x : ℕ),
  0 < a → 0 < b →
  b > a →
  a + b = 15 →
  (a - 2*x) * (b - 2*x) = 2 * a * b / 3 →
  (a = 8 ∧ b = 7) ∨ (a = 7 ∧ b = 8) :=
by sorry

end NUMINAMATH_CALUDE_floor_painting_dimensions_l2836_283657


namespace NUMINAMATH_CALUDE_rectangle_diagonal_l2836_283668

theorem rectangle_diagonal (l w : ℝ) (h_area : l * w = 20) (h_perimeter : 2 * l + 2 * w = 18) :
  l^2 + w^2 = 41 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_l2836_283668


namespace NUMINAMATH_CALUDE_minimum_pipes_needed_l2836_283664

/-- Represents the cutting methods for a 6m steel pipe -/
inductive CuttingMethod
  | method2 -- 4 pieces of 0.8m and 1 piece of 2.5m
  | method3 -- 1 piece of 0.8m and 2 pieces of 2.5m

/-- Represents the number of pieces obtained from each cutting method -/
def piecesObtained (m : CuttingMethod) : (ℕ × ℕ) :=
  match m with
  | CuttingMethod.method2 => (4, 1)
  | CuttingMethod.method3 => (1, 2)

theorem minimum_pipes_needed :
  ∃ (x y : ℕ),
    x * (piecesObtained CuttingMethod.method2).1 + y * (piecesObtained CuttingMethod.method3).1 = 100 ∧
    x * (piecesObtained CuttingMethod.method2).2 + y * (piecesObtained CuttingMethod.method3).2 = 32 ∧
    x + y = 28 ∧
    ∀ (a b : ℕ),
      a * (piecesObtained CuttingMethod.method2).1 + b * (piecesObtained CuttingMethod.method3).1 = 100 →
      a * (piecesObtained CuttingMethod.method2).2 + b * (piecesObtained CuttingMethod.method3).2 = 32 →
      a + b ≥ 28 := by
  sorry

end NUMINAMATH_CALUDE_minimum_pipes_needed_l2836_283664


namespace NUMINAMATH_CALUDE_problem_statement_l2836_283639

theorem problem_statement (x y : ℝ) : 
  (x - 1)^2 + |y + 1| = 0 → 2*(x^2 - y^2 + 1) - 2*(x^2 + y^2) + x*y = -3 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l2836_283639


namespace NUMINAMATH_CALUDE_julia_tag_game_l2836_283695

theorem julia_tag_game (tuesday_kids : ℕ) (extra_monday_kids : ℕ) : 
  tuesday_kids = 14 → extra_monday_kids = 8 → tuesday_kids + extra_monday_kids = 22 :=
by
  sorry

end NUMINAMATH_CALUDE_julia_tag_game_l2836_283695


namespace NUMINAMATH_CALUDE_elvins_internet_charge_l2836_283684

/-- Represents Elvin's monthly telephone bill structure -/
structure MonthlyBill where
  callCharge : ℝ
  internetCharge : ℝ
  totalBill : ℝ
  totalBill_eq : totalBill = callCharge + internetCharge

/-- Theorem stating Elvin's fixed monthly internet charge -/
theorem elvins_internet_charge 
  (jan : MonthlyBill) 
  (feb : MonthlyBill) 
  (h1 : jan.totalBill = 40)
  (h2 : feb.totalBill = 76)
  (h3 : feb.callCharge = 2 * jan.callCharge)
  (h4 : jan.internetCharge = feb.internetCharge) : 
  jan.internetCharge = 4 := by
sorry

end NUMINAMATH_CALUDE_elvins_internet_charge_l2836_283684


namespace NUMINAMATH_CALUDE_temperature_at_midnight_l2836_283687

/-- Given temperature changes throughout a day, calculate the temperature at midnight. -/
theorem temperature_at_midnight 
  (morning_temp : Int) 
  (noon_rise : Int) 
  (midnight_drop : Int) 
  (h1 : morning_temp = -2)
  (h2 : noon_rise = 13)
  (h3 : midnight_drop = 8) : 
  morning_temp + noon_rise - midnight_drop = 3 :=
by sorry

end NUMINAMATH_CALUDE_temperature_at_midnight_l2836_283687


namespace NUMINAMATH_CALUDE_circle_intersection_theorem_l2836_283607

-- Define the circle equation
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 4*y + m = 0

-- Define the line equation
def line_equation (x y : ℝ) : Prop :=
  x + 2*y - 4 = 0

-- Define the condition for the equation to represent a circle
def is_circle (m : ℝ) : Prop :=
  m < 5/4

-- Define the intersection condition
def intersects_at_mn (m : ℝ) : Prop :=
  ∃ (M N : ℝ × ℝ),
    circle_equation M.1 M.2 m ∧
    circle_equation N.1 N.2 m ∧
    line_equation M.1 M.2 ∧
    line_equation N.1 N.2 ∧
    (M.1 - N.1)^2 + (M.2 - N.2)^2 = (4/5 * Real.sqrt 5)^2

theorem circle_intersection_theorem :
  ∀ m : ℝ, is_circle m → intersects_at_mn m → m = 3.62 :=
by sorry

end NUMINAMATH_CALUDE_circle_intersection_theorem_l2836_283607


namespace NUMINAMATH_CALUDE_arun_weight_average_l2836_283650

-- Define Arun's weight as a real number
def arun_weight : ℝ := sorry

-- Define the conditions on Arun's weight
def condition1 : Prop := 65 < arun_weight ∧ arun_weight < 72
def condition2 : Prop := 60 < arun_weight ∧ arun_weight < 70
def condition3 : Prop := arun_weight ≤ 68

-- Theorem to prove
theorem arun_weight_average :
  condition1 ∧ condition2 ∧ condition3 →
  (65 + 68) / 2 = 66.5 :=
by sorry

end NUMINAMATH_CALUDE_arun_weight_average_l2836_283650


namespace NUMINAMATH_CALUDE_eight_power_fifteen_divided_by_sixtyfour_power_seven_l2836_283641

theorem eight_power_fifteen_divided_by_sixtyfour_power_seven :
  8^15 / 64^7 = 8 := by sorry

end NUMINAMATH_CALUDE_eight_power_fifteen_divided_by_sixtyfour_power_seven_l2836_283641


namespace NUMINAMATH_CALUDE_definite_integral_abs_quadratic_l2836_283646

theorem definite_integral_abs_quadratic : ∫ x in (-2)..2, |x^2 - 2*x| = 8 := by
  sorry

end NUMINAMATH_CALUDE_definite_integral_abs_quadratic_l2836_283646


namespace NUMINAMATH_CALUDE_intersection_theorem_l2836_283651

-- Define set A
def A : Set ℝ := {x | (x + 2) / (x - 2) ≤ 0}

-- Define set B
def B : Set ℝ := {x | |x - 1| < 2}

-- Define the complement of A with respect to ℝ
def not_A : Set ℝ := {x | x ∉ A}

-- Define the intersection of B and the complement of A
def B_intersect_not_A : Set ℝ := B ∩ not_A

-- Theorem statement
theorem intersection_theorem : B_intersect_not_A = {x | 2 ≤ x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_theorem_l2836_283651


namespace NUMINAMATH_CALUDE_senior_policy_more_profitable_l2836_283603

/-- Represents a customer group with their characteristics -/
structure CustomerGroup where
  repaymentReliability : ℝ
  incomeStability : ℝ
  savingInclination : ℝ
  longTermPreference : ℝ

/-- Represents the bank's policy for a customer group -/
structure BankPolicy where
  depositRate : ℝ
  loanRate : ℝ

/-- Calculates the bank's profit from a customer group under a given policy -/
def bankProfit (group : CustomerGroup) (policy : BankPolicy) : ℝ :=
  group.repaymentReliability * policy.loanRate +
  group.savingInclination * group.longTermPreference * policy.depositRate

/-- Theorem: Under certain conditions, a bank can achieve higher profit 
    by offering better rates to seniors -/
theorem senior_policy_more_profitable 
  (seniors : CustomerGroup) 
  (others : CustomerGroup)
  (seniorPolicy : BankPolicy)
  (otherPolicy : BankPolicy)
  (h1 : seniors.repaymentReliability > others.repaymentReliability)
  (h2 : seniors.incomeStability > others.incomeStability)
  (h3 : seniors.savingInclination > others.savingInclination)
  (h4 : seniors.longTermPreference > others.longTermPreference)
  (h5 : seniorPolicy.depositRate > otherPolicy.depositRate)
  (h6 : seniorPolicy.loanRate < otherPolicy.loanRate) :
  bankProfit seniors seniorPolicy > bankProfit others otherPolicy :=
sorry

end NUMINAMATH_CALUDE_senior_policy_more_profitable_l2836_283603


namespace NUMINAMATH_CALUDE_charity_ticket_sales_l2836_283690

theorem charity_ticket_sales (total_tickets : ℕ) (total_revenue : ℕ) 
  (h_total_tickets : total_tickets = 140)
  (h_total_revenue : total_revenue = 2001) :
  ∃ (full_price : ℕ) (half_price : ℕ) (full_price_tickets : ℕ) (half_price_tickets : ℕ),
    full_price > 0 ∧
    half_price = full_price / 2 ∧
    full_price_tickets + half_price_tickets = total_tickets ∧
    full_price_tickets * full_price + half_price_tickets * half_price = total_revenue ∧
    full_price_tickets * full_price = 782 :=
by sorry

end NUMINAMATH_CALUDE_charity_ticket_sales_l2836_283690


namespace NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l2836_283670

theorem infinite_geometric_series_first_term 
  (r : ℝ) (S : ℝ) (a : ℝ) 
  (h1 : r = -1/3) 
  (h2 : S = 12) 
  (h3 : S = a / (1 - r)) : 
  a = 16 := by
sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l2836_283670


namespace NUMINAMATH_CALUDE_custom_equation_solution_l2836_283665

-- Define the custom operation *
def star (a b : ℝ) : ℝ := 2 * a - b

-- State the theorem
theorem custom_equation_solution :
  ∃! x : ℝ, star 2 (star 6 x) = 2 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_custom_equation_solution_l2836_283665


namespace NUMINAMATH_CALUDE_remainder_plus_fraction_equals_result_l2836_283648

def rem (x y : ℚ) : ℚ := x - y * ⌊x / y⌋

theorem remainder_plus_fraction_equals_result :
  rem (5/7) (-3/4) + 1/14 = 1/28 := by
  sorry

end NUMINAMATH_CALUDE_remainder_plus_fraction_equals_result_l2836_283648


namespace NUMINAMATH_CALUDE_equation_roots_opposite_signs_l2836_283642

theorem equation_roots_opposite_signs (a b c d m : ℝ) (hd : d ≠ 0) :
  (∀ x, (x^2 - (b+1)*x) / ((a-1)*x - (c+d)) = (m-2) / (m+2)) →
  (∃ r : ℝ, r ≠ 0 ∧ (r^2 - (b+1)*r = 0) ∧ (-r^2 - (b+1)*(-r) = 0)) →
  m = 2*(a-b-2) / (a+b) :=
by sorry

end NUMINAMATH_CALUDE_equation_roots_opposite_signs_l2836_283642


namespace NUMINAMATH_CALUDE_multiple_of_two_three_five_l2836_283643

theorem multiple_of_two_three_five : ∃ n : ℕ, 2 ∣ n ∧ 3 ∣ n ∧ 5 ∣ n :=
  by
  use 30
  sorry

end NUMINAMATH_CALUDE_multiple_of_two_three_five_l2836_283643


namespace NUMINAMATH_CALUDE_tan_C_minus_pi_4_max_area_l2836_283604

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfiesCondition (t : Triangle) : Prop :=
  t.a^2 + t.b^2 = t.a * t.b + t.c^2

-- Part I
theorem tan_C_minus_pi_4 (t : Triangle) (h : satisfiesCondition t) :
  Real.tan (t.C - π/4) = 2 - Real.sqrt 3 := by
  sorry

-- Part II
theorem max_area (t : Triangle) (h1 : satisfiesCondition t) (h2 : t.c = Real.sqrt 3) :
  (∀ s : Triangle, satisfiesCondition s → s.c = Real.sqrt 3 →
    t.a * t.b * Real.sin t.C / 2 ≥ s.a * s.b * Real.sin s.C / 2) →
  t.a * t.b * Real.sin t.C / 2 = 3 * Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_tan_C_minus_pi_4_max_area_l2836_283604


namespace NUMINAMATH_CALUDE_max_soap_boxes_in_carton_l2836_283617

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℕ :=
  d.length * d.width * d.height

/-- Theorem: Maximum number of soap boxes in a carton -/
theorem max_soap_boxes_in_carton (carton soap : BoxDimensions)
    (h_carton : carton = ⟨25, 42, 60⟩)
    (h_soap : soap = ⟨7, 6, 10⟩) :
    (boxVolume carton) / (boxVolume soap) = 150 := by
  sorry


end NUMINAMATH_CALUDE_max_soap_boxes_in_carton_l2836_283617


namespace NUMINAMATH_CALUDE_expression_not_equal_one_l2836_283620

theorem expression_not_equal_one (a y : ℝ) (ha : a ≠ 0) (hay : a ≠ y) :
  (a / (a - y) + y / (a + y)) / (y / (a - y) - a / (a + y)) ≠ 1 :=
by sorry

end NUMINAMATH_CALUDE_expression_not_equal_one_l2836_283620


namespace NUMINAMATH_CALUDE_inverse_mod_53_l2836_283611

theorem inverse_mod_53 (h : (19⁻¹ : ZMod 53) = 31) : (44⁻¹ : ZMod 53) = 22 := by
  sorry

end NUMINAMATH_CALUDE_inverse_mod_53_l2836_283611


namespace NUMINAMATH_CALUDE_candy_cost_l2836_283673

/-- 
Given that each piece of bulk candy costs 8 cents and 28 gumdrops can be bought,
prove that the total amount of cents is 224.
-/
theorem candy_cost (cost_per_piece : ℕ) (num_gumdrops : ℕ) (h1 : cost_per_piece = 8) (h2 : num_gumdrops = 28) :
  cost_per_piece * num_gumdrops = 224 := by
  sorry

end NUMINAMATH_CALUDE_candy_cost_l2836_283673


namespace NUMINAMATH_CALUDE_parallel_vectors_t_value_l2836_283631

/-- Two vectors are parallel if their corresponding components are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  (a.1 * b.2 = a.2 * b.1)

/-- Given vectors a and b, prove that if they are parallel, then t = 9 -/
theorem parallel_vectors_t_value (t : ℝ) :
  let a : ℝ × ℝ := (1, 3)
  let b : ℝ × ℝ := (3, t)
  are_parallel a b → t = 9 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_t_value_l2836_283631


namespace NUMINAMATH_CALUDE_solve_for_T_l2836_283623

theorem solve_for_T : ∃ T : ℚ, (3/4) * (1/6) * T = (1/5) * (1/4) * 120 ∧ T = 48 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_T_l2836_283623


namespace NUMINAMATH_CALUDE_probability_one_black_one_white_l2836_283616

def total_balls : ℕ := 5
def black_balls : ℕ := 3
def white_balls : ℕ := 2
def drawn_balls : ℕ := 2

theorem probability_one_black_one_white :
  (black_balls.choose 1 * white_balls.choose 1 : ℚ) / total_balls.choose drawn_balls = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_one_black_one_white_l2836_283616


namespace NUMINAMATH_CALUDE_coefficient_d_nonzero_l2836_283675

-- Define the polynomial Q(x)
def Q (a b c d e : ℂ) (x : ℂ) : ℂ := x^5 + a*x^4 + b*x^3 + c*x^2 + d*x + e

-- State the theorem
theorem coefficient_d_nonzero 
  (a b c d e : ℂ) 
  (h1 : ∃ u v : ℂ, ∀ x : ℂ, Q a b c d e x = x * (x - (2 + 3*I)) * (x - (2 - 3*I)) * (x - u) * (x - v))
  (h2 : Q a b c d e 0 = 0)
  (h3 : Q a b c d e (2 + 3*I) = 0)
  (h4 : ∀ x : ℂ, Q a b c d e x = 0 → x = 0 ∨ x = 2 + 3*I ∨ x = 2 - 3*I ∨ (∃ y : ℂ, y ≠ 0 ∧ y ≠ 2 + 3*I ∧ y ≠ 2 - 3*I ∧ x = y)) :
  d ≠ 0 := by
  sorry


end NUMINAMATH_CALUDE_coefficient_d_nonzero_l2836_283675


namespace NUMINAMATH_CALUDE_trapezoid_area_sum_l2836_283614

/-- Represents a trapezoid with four side lengths -/
structure Trapezoid where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ

/-- Calculates the sum of all possible areas of a trapezoid -/
def sum_of_areas (t : Trapezoid) : ℝ := sorry

/-- Checks if a number is not divisible by the square of any prime -/
def not_divisible_by_square_prime (n : ℕ) : Prop := sorry

/-- Main theorem statement -/
theorem trapezoid_area_sum (t : Trapezoid) :
  t.side1 = 4 ∧ t.side2 = 6 ∧ t.side3 = 8 ∧ t.side4 = 10 →
  ∃ (r₁ r₂ r₃ : ℚ) (n₁ n₂ : ℕ),
    sum_of_areas t = r₁ * Real.sqrt n₁ + r₂ * Real.sqrt n₂ + r₃ ∧
    not_divisible_by_square_prime n₁ ∧
    not_divisible_by_square_prime n₂ ∧
    ⌊r₁ + r₂ + r₃ + n₁ + n₂⌋ = 18 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_sum_l2836_283614


namespace NUMINAMATH_CALUDE_rooster_stamps_count_l2836_283606

theorem rooster_stamps_count (daffodil_stamps : ℕ) (rooster_stamps : ℕ) 
  (h1 : daffodil_stamps = 2) 
  (h2 : rooster_stamps - daffodil_stamps = 0) : 
  rooster_stamps = 2 := by
  sorry

end NUMINAMATH_CALUDE_rooster_stamps_count_l2836_283606


namespace NUMINAMATH_CALUDE_vector_angle_problem_l2836_283624

noncomputable def angle_between_vectors (a b : ℝ × ℝ) : ℝ := 
  Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))

theorem vector_angle_problem (a b : ℝ × ℝ) 
  (h1 : a.1^2 + a.2^2 = 4)
  (h2 : Real.sqrt (b.1^2 + b.2^2) = 2)
  (h3 : (a.1 + b.1) * (3 * a.1 - b.1) + (a.2 + b.2) * (3 * a.2 - b.2) = 4) :
  angle_between_vectors a b = 2 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_vector_angle_problem_l2836_283624


namespace NUMINAMATH_CALUDE_cylinder_volume_unit_dimensions_l2836_283676

/-- The volume of a cylinder with base radius 1 and height 1 is π. -/
theorem cylinder_volume_unit_dimensions : 
  let r : ℝ := 1
  let h : ℝ := 1
  let V := π * r^2 * h
  V = π := by sorry

end NUMINAMATH_CALUDE_cylinder_volume_unit_dimensions_l2836_283676


namespace NUMINAMATH_CALUDE_janes_apple_baskets_l2836_283697

theorem janes_apple_baskets :
  ∀ (total_apples : ℕ) (apples_taken : ℕ) (apples_left : ℕ),
    total_apples = 64 →
    apples_taken = 3 →
    apples_left = 13 →
    ∃ (num_baskets : ℕ),
      num_baskets * (apples_left + apples_taken) = total_apples ∧
      num_baskets = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_janes_apple_baskets_l2836_283697


namespace NUMINAMATH_CALUDE_f_2015_value_l2836_283679

def f (x : ℝ) : ℝ := sorry

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def piecewise_def (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  (∀ x, -2 ≤ x ∧ x < 0 → f x = a * x + b) ∧
  (∀ x, 0 < x ∧ x ≤ 2 → f x = a * x - 1)

theorem f_2015_value (a b : ℝ) :
  is_odd f ∧ has_period f 4 ∧ piecewise_def f a b →
  f 2015 = 3/2 := by sorry

end NUMINAMATH_CALUDE_f_2015_value_l2836_283679


namespace NUMINAMATH_CALUDE_exists_larger_area_same_perimeter_l2836_283625

-- Define a convex figure
structure ConvexFigure where
  perimeter : ℝ
  area : ℝ

-- Define a property for a figure to be a circle
def isCircle (f : ConvexFigure) : Prop := sorry

-- Theorem statement
theorem exists_larger_area_same_perimeter 
  (Φ : ConvexFigure) 
  (h_not_circle : ¬ isCircle Φ) : 
  ∃ (Ψ : ConvexFigure), 
    Ψ.perimeter = Φ.perimeter ∧ 
    Ψ.area > Φ.area := by
  sorry

end NUMINAMATH_CALUDE_exists_larger_area_same_perimeter_l2836_283625


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l2836_283682

theorem quadratic_coefficient (α : ℝ) (p q : ℝ) : 
  (∀ x, x^2 - (α - 2)*x - α - 1 = 0 ↔ x = p ∨ x = q) →
  (∀ a b, a^2 + b^2 ≥ 5 ∧ (a = p ∧ b = q ∨ a = q ∧ b = p) → p^2 + q^2 ≥ 5) →
  p^2 + q^2 = 5 →
  α - 2 = -1 :=
sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l2836_283682


namespace NUMINAMATH_CALUDE_subtraction_result_l2836_283663

/-- The value obtained when 20² is subtracted from the square of 68.70953354520753 is approximately 4321 -/
theorem subtraction_result : 
  let x : ℝ := 68.70953354520753
  ∃ ε > 0, abs ((x^2 - 20^2) - 4321) < ε :=
by sorry

end NUMINAMATH_CALUDE_subtraction_result_l2836_283663


namespace NUMINAMATH_CALUDE_distribute_negative_two_l2836_283615

theorem distribute_negative_two (m n : ℝ) : -2 * (m - n) = -2 * m + 2 * n := by
  sorry

end NUMINAMATH_CALUDE_distribute_negative_two_l2836_283615


namespace NUMINAMATH_CALUDE_geometric_series_first_term_l2836_283649

/-- For an infinite geometric series with common ratio 1/4 and sum 40, the first term is 30 -/
theorem geometric_series_first_term (a : ℝ) : 
  (∀ n : ℕ, ∑' k, a * (1/4)^k = 40) → a = 30 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_first_term_l2836_283649


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2836_283698

theorem geometric_sequence_sum (n : ℕ) (a r : ℚ) (h1 : a = 1/3) (h2 : r = 1/3) :
  (a * (1 - r^n) / (1 - r) = 80/243) → n = 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2836_283698


namespace NUMINAMATH_CALUDE_initial_pens_l2836_283671

theorem initial_pens (initial : ℕ) (mike_gives : ℕ) (cindy_doubles : ℕ → ℕ) (sharon_takes : ℕ) (final : ℕ) : 
  mike_gives = 22 →
  cindy_doubles = (· * 2) →
  sharon_takes = 19 →
  final = 39 →
  cindy_doubles (initial + mike_gives) - sharon_takes = final →
  initial = 7 := by
sorry

end NUMINAMATH_CALUDE_initial_pens_l2836_283671


namespace NUMINAMATH_CALUDE_vector_magnitude_proof_l2836_283694

/-- Given a plane vector a = (2,0), |b| = 2, and a ⋅ b = 2, prove |a - 2b| = 2√3 -/
theorem vector_magnitude_proof (b : ℝ × ℝ) :
  let a : ℝ × ℝ := (2, 0)
  (norm b = 2) →
  (a.1 * b.1 + a.2 * b.2 = 2) →
  norm (a - 2 • b) = 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_vector_magnitude_proof_l2836_283694


namespace NUMINAMATH_CALUDE_distribute_six_balls_three_boxes_l2836_283655

/-- The number of ways to distribute distinguishable balls into distinguishable boxes -/
def distribute_balls (num_balls : ℕ) (num_boxes : ℕ) : ℕ :=
  num_boxes ^ num_balls

/-- Theorem: There are 729 ways to distribute 6 distinguishable balls into 3 distinguishable boxes -/
theorem distribute_six_balls_three_boxes :
  distribute_balls 6 3 = 729 := by
  sorry

end NUMINAMATH_CALUDE_distribute_six_balls_three_boxes_l2836_283655


namespace NUMINAMATH_CALUDE_sample_size_correct_l2836_283612

/-- Represents a population of students -/
structure Population where
  size : ℕ

/-- Represents a sample of students -/
structure Sample where
  size : ℕ

/-- Theorem stating that the sample size is correct -/
theorem sample_size_correct (pop : Population) (samp : Sample) : 
  pop.size = 8000 → samp.size = 400 → samp.size = 400 := by sorry

end NUMINAMATH_CALUDE_sample_size_correct_l2836_283612


namespace NUMINAMATH_CALUDE_sin_product_equals_one_sixteenth_l2836_283600

theorem sin_product_equals_one_sixteenth : 
  Real.sin (12 * π / 180) * Real.sin (48 * π / 180) * 
  Real.sin (54 * π / 180) * Real.sin (78 * π / 180) = 1/16 := by
  sorry

end NUMINAMATH_CALUDE_sin_product_equals_one_sixteenth_l2836_283600


namespace NUMINAMATH_CALUDE_curve_self_intersection_l2836_283678

-- Define the curve
def curve (t : ℝ) : ℝ × ℝ :=
  (t^2 - 3, t^4 - t^2 - 9*t + 6)

-- Define the self-intersection point
def intersection_point : ℝ × ℝ := (6, 6)

-- Theorem statement
theorem curve_self_intersection :
  ∃ (a b : ℝ), a ≠ b ∧ curve a = curve b ∧ curve a = intersection_point :=
sorry

end NUMINAMATH_CALUDE_curve_self_intersection_l2836_283678


namespace NUMINAMATH_CALUDE_sum_of_2001_and_1015_l2836_283660

theorem sum_of_2001_and_1015 : 2001 + 1015 = 3016 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_2001_and_1015_l2836_283660


namespace NUMINAMATH_CALUDE_smallest_multiple_l2836_283688

theorem smallest_multiple (n : ℕ) : n = 255 ↔ 
  (∃ k : ℕ, n = 15 * k) ∧ 
  (∃ m : ℕ, n = 65 * m + 7) ∧ 
  (∃ p : ℕ, n = 5 * p) ∧ 
  (∀ x : ℕ, x < n → 
    (¬(∃ k : ℕ, x = 15 * k) ∨ 
     ¬(∃ m : ℕ, x = 65 * m + 7) ∨ 
     ¬(∃ p : ℕ, x = 5 * p))) :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiple_l2836_283688


namespace NUMINAMATH_CALUDE_fraction_integer_iff_p_in_set_l2836_283667

theorem fraction_integer_iff_p_in_set (p : ℕ) (hp : p > 0) :
  (∃ k : ℕ, k > 0 ∧ (4 * p + 34 : ℚ) / (3 * p - 8 : ℚ) = k) ↔ p ∈ ({3, 4, 5, 12} : Set ℕ) :=
sorry

end NUMINAMATH_CALUDE_fraction_integer_iff_p_in_set_l2836_283667


namespace NUMINAMATH_CALUDE_wage_calculation_l2836_283629

/-- A worker's wage calculation problem -/
theorem wage_calculation 
  (total_days : ℕ) 
  (absent_days : ℕ) 
  (fine_per_day : ℕ) 
  (total_pay : ℕ) 
  (h1 : total_days = 30)
  (h2 : absent_days = 7)
  (h3 : fine_per_day = 2)
  (h4 : total_pay = 216) :
  ∃ (daily_wage : ℕ),
    (total_days - absent_days) * daily_wage - absent_days * fine_per_day = total_pay ∧
    daily_wage = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_wage_calculation_l2836_283629


namespace NUMINAMATH_CALUDE_cubic_difference_l2836_283658

theorem cubic_difference (a b : ℝ) 
  (h1 : a - b = 7) 
  (h2 : a^2 + b^2 = 47) : 
  a^3 - b^3 = 322 := by
  sorry

end NUMINAMATH_CALUDE_cubic_difference_l2836_283658


namespace NUMINAMATH_CALUDE_largest_divisor_of_difference_of_squares_l2836_283619

theorem largest_divisor_of_difference_of_squares (m n : ℤ) : 
  Odd m → Odd n → n < m → 
  (∀ k : ℤ, k ∣ (m^2 - n^2) → k ≤ 8) ∧ 8 ∣ (m^2 - n^2) :=
sorry

end NUMINAMATH_CALUDE_largest_divisor_of_difference_of_squares_l2836_283619


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l2836_283685

theorem polynomial_division_remainder : ∃ q : Polynomial ℚ,
  x^30 + x^24 + x^18 + x^12 + x^6 + 1 = (x^4 + x^3 + x^2 + x + 1) * q + 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l2836_283685


namespace NUMINAMATH_CALUDE_journey_speed_theorem_l2836_283613

/-- Given a journey with the following parameters:
  * total_distance: The total distance traveled in miles
  * total_time: The total time of the journey in minutes
  * speed_first_30: The average speed during the first 30 minutes in mph
  * speed_second_30: The average speed during the second 30 minutes in mph

  This function calculates the average speed during the last 60 minutes of the journey. -/
def average_speed_last_60 (total_distance : ℝ) (total_time : ℝ) (speed_first_30 : ℝ) (speed_second_30 : ℝ) : ℝ :=
  sorry

/-- Theorem stating that for the given journey parameters, 
    the average speed during the last 60 minutes is 77.5 mph -/
theorem journey_speed_theorem :
  average_speed_last_60 150 120 75 70 = 77.5 := by
  sorry

end NUMINAMATH_CALUDE_journey_speed_theorem_l2836_283613


namespace NUMINAMATH_CALUDE_may_fourth_is_sunday_l2836_283633

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a specific month -/
structure Month where
  fridayCount : Nat
  fridayDatesSum : Nat

/-- Returns the day of the week for a given date in the month -/
def dayOfWeek (m : Month) (date : Nat) : DayOfWeek := sorry

theorem may_fourth_is_sunday (m : Month) 
  (h1 : m.fridayCount = 5) 
  (h2 : m.fridayDatesSum = 80) : 
  dayOfWeek m 4 = DayOfWeek.Sunday := by sorry

end NUMINAMATH_CALUDE_may_fourth_is_sunday_l2836_283633


namespace NUMINAMATH_CALUDE_perpendicular_bisector_m_value_l2836_283659

/-- Given points A and B, if the equation of the perpendicular bisector of segment AB is x + 2y - 2 = 0, then m = 3 -/
theorem perpendicular_bisector_m_value (m : ℝ) : 
  let A : ℝ × ℝ := (1, -2)
  let B : ℝ × ℝ := (m, 2)
  let midpoint : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  (midpoint.1 + 2 * midpoint.2 - 2 = 0) → m = 3 := by
sorry


end NUMINAMATH_CALUDE_perpendicular_bisector_m_value_l2836_283659


namespace NUMINAMATH_CALUDE_statement_c_is_false_l2836_283622

theorem statement_c_is_false : ¬(∀ (p q : Prop), ¬(p ∧ q) → (¬p ∧ ¬q)) := by
  sorry

end NUMINAMATH_CALUDE_statement_c_is_false_l2836_283622


namespace NUMINAMATH_CALUDE_min_students_class_5_7_l2836_283634

theorem min_students_class_5_7 (n : ℕ) : 
  (∃ k : ℕ, n = 7 * k + 3) ∧ 
  (∃ m : ℕ, n = 8 * m + 3) → 
  n ≥ 59 :=
sorry

end NUMINAMATH_CALUDE_min_students_class_5_7_l2836_283634


namespace NUMINAMATH_CALUDE_max_volume_special_tetrahedron_l2836_283691

/-- A tetrahedron with two vertices on a sphere of radius √10 and two on a concentric sphere of radius 2 -/
structure SpecialTetrahedron where
  /-- The radius of the larger sphere -/
  R : ℝ
  /-- The radius of the smaller sphere -/
  r : ℝ
  /-- Assertion that R = √10 -/
  h_R : R = Real.sqrt 10
  /-- Assertion that r = 2 -/
  h_r : r = 2

/-- The volume of a SpecialTetrahedron -/
def volume (t : SpecialTetrahedron) : ℝ :=
  sorry

/-- The maximum volume of a SpecialTetrahedron is 6√2 -/
theorem max_volume_special_tetrahedron :
  ∀ t : SpecialTetrahedron, volume t ≤ 6 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_max_volume_special_tetrahedron_l2836_283691


namespace NUMINAMATH_CALUDE_quadratic_function_property_l2836_283644

/-- 
Given a quadratic function f(x) = ax^2 + bx + 5 with a ≠ 0,
if there exist two distinct points (x₁, 2023) and (x₂, 2023) on the graph of f,
then f(x₁ + x₂) = 5
-/
theorem quadratic_function_property (a b x₁ x₂ : ℝ) (ha : a ≠ 0) :
  let f : ℝ → ℝ := λ x => a * x^2 + b * x + 5
  (f x₁ = 2023) → (f x₂ = 2023) → (x₁ ≠ x₂) → f (x₁ + x₂) = 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l2836_283644


namespace NUMINAMATH_CALUDE_min_value_of_2x_plus_y_l2836_283602

theorem min_value_of_2x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 2/y = 1) :
  2*x + y ≥ 8 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_2x_plus_y_l2836_283602


namespace NUMINAMATH_CALUDE_angle_between_clock_hands_at_8_30_angle_between_clock_hands_at_8_30_is_75_l2836_283628

/-- The angle between clock hands at 8:30 --/
theorem angle_between_clock_hands_at_8_30 : ℝ :=
  let hours : ℝ := 8.5
  let minutes : ℝ := 30
  let angle_per_hour : ℝ := 360 / 12
  let hour_hand_angle : ℝ := hours * angle_per_hour
  let minute_hand_angle : ℝ := minutes * (360 / 60)
  let angle_diff : ℝ := |hour_hand_angle - minute_hand_angle|
  75

/-- Proof that the angle between clock hands at 8:30 is 75° --/
theorem angle_between_clock_hands_at_8_30_is_75 :
  angle_between_clock_hands_at_8_30 = 75 := by
  sorry

end NUMINAMATH_CALUDE_angle_between_clock_hands_at_8_30_angle_between_clock_hands_at_8_30_is_75_l2836_283628


namespace NUMINAMATH_CALUDE_f_range_l2836_283666

noncomputable def f (x : ℝ) : ℝ := Real.arctan x + Real.arctan ((2 - x) / (2 + x))

theorem f_range : ∀ x : ℝ, f x = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_f_range_l2836_283666


namespace NUMINAMATH_CALUDE_cube_cutting_l2836_283626

theorem cube_cutting (n : ℕ) : 
  (∃ s : ℕ, n > s ∧ n^3 - s^3 = 152) → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_cube_cutting_l2836_283626


namespace NUMINAMATH_CALUDE_divisibility_condition_l2836_283627

theorem divisibility_condition (a : ℤ) : 
  5 ∣ (a^3 + 3*a + 1) ↔ a % 5 = 1 ∨ a % 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_condition_l2836_283627


namespace NUMINAMATH_CALUDE_range_of_m_l2836_283693

theorem range_of_m (m : ℝ) : 
  (∃ x : ℝ, |x - 3| + |x - 5| < m^2 - m) → m < -1 ∨ m > 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l2836_283693


namespace NUMINAMATH_CALUDE_roots_of_equation_l2836_283662

def equation (x : ℝ) : ℝ := (x^2 - 5*x + 6) * x * (x - 5)

theorem roots_of_equation :
  {x : ℝ | equation x = 0} = {0, 2, 3, 5} := by
  sorry

end NUMINAMATH_CALUDE_roots_of_equation_l2836_283662


namespace NUMINAMATH_CALUDE_two_digit_sum_square_property_l2836_283609

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Check if a number is two-digit -/
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- The set of numbers satisfying the condition -/
def valid_numbers : Set ℕ := {10, 20, 11, 30, 21, 12, 31, 22, 13}

/-- Main theorem -/
theorem two_digit_sum_square_property (A : ℕ) :
  is_two_digit A →
  (((sum_of_digits A) ^ 2 = sum_of_digits (A ^ 2)) ↔ A ∈ valid_numbers) := by
  sorry

end NUMINAMATH_CALUDE_two_digit_sum_square_property_l2836_283609


namespace NUMINAMATH_CALUDE_davids_physics_marks_l2836_283683

/-- Given David's marks in various subjects and his average, prove his marks in Physics --/
theorem davids_physics_marks
  (english_marks : ℕ)
  (math_marks : ℕ)
  (chemistry_marks : ℕ)
  (biology_marks : ℕ)
  (average_marks : ℕ)
  (total_subjects : ℕ)
  (h1 : english_marks = 86)
  (h2 : math_marks = 85)
  (h3 : chemistry_marks = 87)
  (h4 : biology_marks = 85)
  (h5 : average_marks = 85)
  (h6 : total_subjects = 5)
  : ∃ (physics_marks : ℕ),
    physics_marks = average_marks * total_subjects - (english_marks + math_marks + chemistry_marks + biology_marks) ∧
    physics_marks = 82 :=
by sorry

end NUMINAMATH_CALUDE_davids_physics_marks_l2836_283683


namespace NUMINAMATH_CALUDE_total_rocks_in_border_l2836_283605

/-- The number of rocks in Mrs. Hilt's garden border -/
def garden_border (placed : ℝ) (additional : ℝ) : ℝ :=
  placed + additional

/-- Theorem stating the total number of rocks in the completed border -/
theorem total_rocks_in_border :
  garden_border 125.0 64.0 = 189.0 := by
  sorry

end NUMINAMATH_CALUDE_total_rocks_in_border_l2836_283605


namespace NUMINAMATH_CALUDE_only_white_balls_drawn_is_random_variable_l2836_283669

/-- A bag containing white and red balls -/
structure Bag where
  white_balls : ℕ
  red_balls : ℕ

/-- The options for potential random variables -/
inductive DrawOption
  | BallsDrawn
  | WhiteBallsDrawn
  | TotalBallsDrawn
  | TotalBallsInBag

/-- Definition of a random variable in this context -/
def is_random_variable (option : DrawOption) (bag : Bag) (num_drawn : ℕ) : Prop :=
  match option with
  | DrawOption.BallsDrawn => num_drawn ≠ num_drawn
  | DrawOption.WhiteBallsDrawn => true
  | DrawOption.TotalBallsDrawn => num_drawn ≠ num_drawn
  | DrawOption.TotalBallsInBag => bag.white_balls + bag.red_balls ≠ bag.white_balls + bag.red_balls

/-- The main theorem stating that only the number of white balls drawn is a random variable -/
theorem only_white_balls_drawn_is_random_variable (bag : Bag) (num_drawn : ℕ) :
  bag.white_balls = 5 → bag.red_balls = 3 → num_drawn = 3 →
  ∀ (option : DrawOption), is_random_variable option bag num_drawn ↔ option = DrawOption.WhiteBallsDrawn :=
by sorry

end NUMINAMATH_CALUDE_only_white_balls_drawn_is_random_variable_l2836_283669


namespace NUMINAMATH_CALUDE_ball_drawing_theorem_l2836_283696

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  white : Nat
  red : Nat
  black : Nat

/-- The minimum number of balls to draw to ensure at least one of each color -/
def minDrawForAllColors (counts : BallCounts) : Nat :=
  counts.white + counts.red + counts.black - 2

/-- The minimum number of balls to draw to ensure 10 balls of one color -/
def minDrawForTenOfOneColor (counts : BallCounts) : Nat :=
  min counts.white counts.red + min counts.white counts.black + 
  min counts.red counts.black + 10 - 1

/-- Theorem stating the correct answers for the given ball counts -/
theorem ball_drawing_theorem (counts : BallCounts) 
  (h1 : counts.white = 5) (h2 : counts.red = 12) (h3 : counts.black = 20) : 
  minDrawForAllColors counts = 33 ∧ minDrawForTenOfOneColor counts = 24 := by
  sorry

#eval minDrawForAllColors ⟨5, 12, 20⟩
#eval minDrawForTenOfOneColor ⟨5, 12, 20⟩

end NUMINAMATH_CALUDE_ball_drawing_theorem_l2836_283696


namespace NUMINAMATH_CALUDE_block_distribution_l2836_283601

theorem block_distribution (n : ℕ) (h : n > 0) (h_divides : n ∣ 49) :
  ∃ (blocks_per_color : ℕ), blocks_per_color > 0 ∧ blocks_per_color * n = 49 := by
  sorry

end NUMINAMATH_CALUDE_block_distribution_l2836_283601


namespace NUMINAMATH_CALUDE_lowest_score_problem_l2836_283645

theorem lowest_score_problem (scores : Finset ℕ) (highest lowest : ℕ) :
  Finset.card scores = 15 →
  highest ∈ scores →
  lowest ∈ scores →
  highest = 100 →
  (Finset.sum scores id) / 15 = 85 →
  ((Finset.sum scores id) - highest - lowest) / 13 = 86 →
  lowest = 57 := by
  sorry

end NUMINAMATH_CALUDE_lowest_score_problem_l2836_283645


namespace NUMINAMATH_CALUDE_count_valid_numbers_l2836_283638

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def is_even (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

def valid_number (n : ℕ) : Prop :=
  n ≥ 10 ∧ n ≤ 99 ∧
  n > 80 ∧
  is_prime (n / 10) ∧
  is_even (n % 10)

theorem count_valid_numbers :
  ∃ (S : Finset ℕ), (∀ n ∈ S, valid_number n) ∧ S.card = 5 :=
sorry

end NUMINAMATH_CALUDE_count_valid_numbers_l2836_283638


namespace NUMINAMATH_CALUDE_lily_painting_rate_l2836_283689

/-- Represents the number of cups Gina can paint per hour -/
structure PaintingRate where
  roses : ℕ
  lilies : ℕ

/-- Represents an order of cups -/
structure Order where
  roses : ℕ
  lilies : ℕ

theorem lily_painting_rate 
  (gina_rate : PaintingRate)
  (order : Order)
  (total_payment : ℕ)
  (hourly_rate : ℕ)
  (h1 : gina_rate.roses = 6)
  (h2 : order.roses = 6)
  (h3 : order.lilies = 14)
  (h4 : total_payment = 90)
  (h5 : hourly_rate = 30) :
  gina_rate.lilies = 7 := by
  sorry

end NUMINAMATH_CALUDE_lily_painting_rate_l2836_283689


namespace NUMINAMATH_CALUDE_distribute_students_count_l2836_283652

/-- The number of ways to distribute four students into three classes -/
def distribute_students : ℕ :=
  let total_distributions := (4 : ℕ).choose 2 * (3 : ℕ).factorial
  let invalid_distributions := (3 : ℕ).factorial
  total_distributions - invalid_distributions

/-- Theorem stating that the number of valid distributions is 30 -/
theorem distribute_students_count : distribute_students = 30 := by
  sorry

end NUMINAMATH_CALUDE_distribute_students_count_l2836_283652


namespace NUMINAMATH_CALUDE_complex_fraction_equals_i_l2836_283661

theorem complex_fraction_equals_i : (1 + Complex.I) / (1 - Complex.I) = Complex.I := by sorry

end NUMINAMATH_CALUDE_complex_fraction_equals_i_l2836_283661


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_E_subset_B_implies_a_geq_neg_one_l2836_283656

-- Define the sets A, B, and E
def A : Set ℝ := {x | (x + 3) * (x - 6) ≥ 0}
def B : Set ℝ := {x | (x + 2) / (x - 14) < 0}
def E (a : ℝ) : Set ℝ := {x | 2 * a < x ∧ x < a + 1}

-- Theorem for the first part of the problem
theorem intersection_A_complement_B :
  A ∩ (Set.univ \ B) = {x : ℝ | x ≤ -3 ∨ x ≥ 14} :=
sorry

-- Theorem for the second part of the problem
theorem E_subset_B_implies_a_geq_neg_one (a : ℝ) :
  E a ⊆ B → a ≥ -1 :=
sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_E_subset_B_implies_a_geq_neg_one_l2836_283656


namespace NUMINAMATH_CALUDE_A_equals_2B_l2836_283677

-- Define the functions f and g
def f (A B x : ℝ) : ℝ := A * x - 2 * B^2
def g (B x : ℝ) : ℝ := B * x

-- State the theorem
theorem A_equals_2B (A B : ℝ) (h1 : B ≠ 0) (h2 : f A B (g B 1) = 0) : A = 2 * B := by
  sorry

end NUMINAMATH_CALUDE_A_equals_2B_l2836_283677


namespace NUMINAMATH_CALUDE_camel_cost_l2836_283692

-- Define the cost of each animal as a real number
variable (camel horse ox elephant lion bear : ℝ)

-- Define the relationships between animal costs
axiom camel_horse : 10 * camel = 24 * horse
axiom horse_ox : 16 * horse = 4 * ox
axiom ox_elephant : 6 * ox = 4 * elephant
axiom elephant_lion : 3 * elephant = 8 * lion
axiom lion_bear : 2 * lion = 6 * bear
axiom bear_cost : 14 * bear = 204000

-- Theorem to prove
theorem camel_cost : camel = 46542.86 := by sorry

end NUMINAMATH_CALUDE_camel_cost_l2836_283692


namespace NUMINAMATH_CALUDE_star_calculation_l2836_283699

def star (x y : ℝ) : ℝ := (x + y) * (x - y)

theorem star_calculation : star 2 (star 3 (star 1 2)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_star_calculation_l2836_283699


namespace NUMINAMATH_CALUDE_distribution_law_l2836_283681

/-- A discrete random variable with two possible values -/
structure DiscreteRV where
  x₁ : ℝ
  x₂ : ℝ
  p₁ : ℝ
  h_x_lt : x₁ < x₂
  h_p_bound : 0 ≤ p₁ ∧ p₁ ≤ 1

/-- Expectation of a DiscreteRV -/
def expectation (X : DiscreteRV) : ℝ := X.x₁ * X.p₁ + X.x₂ * (1 - X.p₁)

/-- Variance of a DiscreteRV -/
def variance (X : DiscreteRV) : ℝ :=
  X.p₁ * (X.x₁ - expectation X)^2 + (1 - X.p₁) * (X.x₂ - expectation X)^2

/-- Theorem stating the distribution law of the given discrete random variable -/
theorem distribution_law (X : DiscreteRV)
  (h_p₁ : X.p₁ = 0.5)
  (h_expectation : expectation X = 3.5)
  (h_variance : variance X = 0.25) :
  X.x₁ = 3 ∧ X.x₂ = 4 :=
sorry

end NUMINAMATH_CALUDE_distribution_law_l2836_283681


namespace NUMINAMATH_CALUDE_linear_equation_solution_l2836_283621

theorem linear_equation_solution (b : ℝ) : 
  (∀ x y : ℝ, x - 2*y + b = 0 → y = (1/2)*x + b - 1) → b = 2 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l2836_283621


namespace NUMINAMATH_CALUDE_sqrt_product_plus_one_l2836_283680

theorem sqrt_product_plus_one : Real.sqrt ((43 : ℝ) * 42 * 41 * 40 + 1) = 1721 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_plus_one_l2836_283680


namespace NUMINAMATH_CALUDE_opposite_signs_abs_sum_less_abs_diff_l2836_283635

theorem opposite_signs_abs_sum_less_abs_diff
  (a b : ℝ) (h : a * b < 0) : |a + b| < |a - b| := by
  sorry

end NUMINAMATH_CALUDE_opposite_signs_abs_sum_less_abs_diff_l2836_283635
