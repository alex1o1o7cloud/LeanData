import Mathlib

namespace NUMINAMATH_CALUDE_complement_event_probability_formula_l3740_374053

/-- The probability of the complement event Ā occurring k times in n trials, 
    given that the probability of event A is p -/
def complementEventProbability (n k : ℕ) (p : ℝ) : ℝ :=
  Nat.choose n k * (1 - p) ^ k * p ^ (n - k)

/-- Theorem stating that the probability of the complement event Ā occurring k times 
    in n trials is equal to ⁽ᵏⁿ)(1-p)ᵏp⁽ⁿ⁻ᵏ⁾, given that the probability of event A is p -/
theorem complement_event_probability_formula (n k : ℕ) (p : ℝ) 
    (h1 : 0 ≤ p ∧ p ≤ 1) (h2 : k ≤ n) : 
  complementEventProbability n k p = Nat.choose n k * (1 - p) ^ k * p ^ (n - k) := by
  sorry

#check complement_event_probability_formula

end NUMINAMATH_CALUDE_complement_event_probability_formula_l3740_374053


namespace NUMINAMATH_CALUDE_square_root_difference_equality_l3740_374062

theorem square_root_difference_equality : 2 * (Real.sqrt (49 + 81) - Real.sqrt (36 - 25)) = 2 * (Real.sqrt 130 - Real.sqrt 11) := by
  sorry

end NUMINAMATH_CALUDE_square_root_difference_equality_l3740_374062


namespace NUMINAMATH_CALUDE_wheel_probability_l3740_374071

theorem wheel_probability (p_D p_E p_F : ℚ) : 
  p_D = 2/5 → p_E = 1/3 → p_D + p_E + p_F = 1 → p_F = 4/15 := by
  sorry

end NUMINAMATH_CALUDE_wheel_probability_l3740_374071


namespace NUMINAMATH_CALUDE_original_number_proof_l3740_374043

theorem original_number_proof (x : ℝ) : 3 * ((2 * x)^2 + 5) = 129 → x = Real.sqrt 9.5 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l3740_374043


namespace NUMINAMATH_CALUDE_problem_solution_l3740_374015

theorem problem_solution (a b c : ℤ) : 
  a < b → b < c → 
  (a + b + c) / 3 = 4 * b → 
  c / b = 11 → 
  a = 0 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3740_374015


namespace NUMINAMATH_CALUDE_coordinates_of_N_l3740_374076

/-- Given a point M and a line segment MN parallel to the x-axis, 
    this function returns the possible coordinates of point N -/
def possible_coordinates_of_N (M : ℝ × ℝ) (length_MN : ℝ) : Set (ℝ × ℝ) :=
  let (x, y) := M
  { (x - length_MN, y), (x + length_MN, y) }

/-- Theorem stating that given M(2, -4) and MN of length 5 parallel to x-axis,
    N has coordinates either (-3, -4) or (7, -4) -/
theorem coordinates_of_N : 
  possible_coordinates_of_N (2, -4) 5 = {(-3, -4), (7, -4)} := by
  sorry


end NUMINAMATH_CALUDE_coordinates_of_N_l3740_374076


namespace NUMINAMATH_CALUDE_melanie_plum_count_l3740_374005

/-- The number of plums Melanie picked -/
def melanie_picked : ℝ := 7.0

/-- The number of plums Sam gave to Melanie -/
def sam_gave : ℝ := 3.0

/-- The total number of plums Melanie has now -/
def total_plums : ℝ := melanie_picked + sam_gave

theorem melanie_plum_count : total_plums = 10.0 := by
  sorry

end NUMINAMATH_CALUDE_melanie_plum_count_l3740_374005


namespace NUMINAMATH_CALUDE_no_valid_n_for_ap_l3740_374020

theorem no_valid_n_for_ap : ¬∃ (n : ℕ), n > 1 ∧ 
  ∃ (a : ℤ), 136 = (n : ℤ) * (2 * a + (n - 1) * 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_no_valid_n_for_ap_l3740_374020


namespace NUMINAMATH_CALUDE_weight_of_BaBr2_l3740_374098

/-- The molecular weight of BaBr2 in grams per mole -/
def molecular_weight_BaBr2 : ℝ := 137.33 + 2 * 79.90

/-- The number of moles of BaBr2 -/
def moles_BaBr2 : ℝ := 8

/-- Calculates the total weight of a given number of moles of BaBr2 -/
def total_weight (mw : ℝ) (moles : ℝ) : ℝ := mw * moles

/-- Theorem stating that the total weight of 8 moles of BaBr2 is 2377.04 grams -/
theorem weight_of_BaBr2 : 
  total_weight molecular_weight_BaBr2 moles_BaBr2 = 2377.04 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_BaBr2_l3740_374098


namespace NUMINAMATH_CALUDE_perpendicular_vectors_x_coord_l3740_374021

/-- Given two vectors a and b in ℝ³, if a is perpendicular to b, then the x-coordinate of a is 4. -/
theorem perpendicular_vectors_x_coord (a b : ℝ × ℝ × ℝ) :
  a.1 = x ∧ a.2.1 = 2 ∧ a.2.2 = -2 ∧
  b = (3, -4, 2) ∧
  a.1 * b.1 + a.2.1 * b.2.1 + a.2.2 * b.2.2 = 0 →
  x = 4 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_x_coord_l3740_374021


namespace NUMINAMATH_CALUDE_mark_additional_spending_l3740_374013

-- Define Mark's initial amount
def initial_amount : ℚ := 180

-- Define the amount spent in the first store
def first_store_spent (initial : ℚ) : ℚ := (1/2 * initial) + 14

-- Define the amount spent in the second store before the additional spending
def second_store_initial_spent (initial : ℚ) : ℚ := 1/3 * initial

-- Theorem to prove
theorem mark_additional_spending :
  initial_amount - first_store_spent initial_amount - second_store_initial_spent initial_amount = 16 := by
  sorry

end NUMINAMATH_CALUDE_mark_additional_spending_l3740_374013


namespace NUMINAMATH_CALUDE_alice_purchases_cost_l3740_374047

/-- The exchange rate from British Pounds to USD -/
def gbp_to_usd : ℝ := 1.25

/-- The exchange rate from Euros to USD -/
def eur_to_usd : ℝ := 1.10

/-- The cost of the book in British Pounds -/
def book_cost_gbp : ℝ := 15

/-- The cost of the souvenir in Euros -/
def souvenir_cost_eur : ℝ := 20

/-- The total cost of Alice's purchases in USD -/
def total_cost_usd : ℝ := book_cost_gbp * gbp_to_usd + souvenir_cost_eur * eur_to_usd

theorem alice_purchases_cost : total_cost_usd = 40.75 := by
  sorry

end NUMINAMATH_CALUDE_alice_purchases_cost_l3740_374047


namespace NUMINAMATH_CALUDE_blood_type_distribution_l3740_374014

theorem blood_type_distribution (total : ℕ) (type_a : ℕ) (type_b : ℕ) : 
  (2 : ℚ) / 9 * total = type_a →
  (2 : ℚ) / 5 * total = type_b →
  type_a = 10 →
  type_b = 18 := by
sorry

end NUMINAMATH_CALUDE_blood_type_distribution_l3740_374014


namespace NUMINAMATH_CALUDE_mets_fans_count_l3740_374028

/-- Represents the number of fans for each team -/
structure FanCounts where
  yankees : ℕ
  mets : ℕ
  dodgers : ℕ
  red_sox : ℕ
  cubs : ℕ

/-- The conditions of the problem -/
def fan_distribution (fc : FanCounts) : Prop :=
  -- Ratio of Yankees : Mets : Dodgers is 3 : 2 : 1
  3 * fc.dodgers = fc.yankees ∧
  2 * fc.dodgers = fc.mets ∧
  -- Ratio of Mets : Red Sox : Cubs is 4 : 5 : 2
  4 * fc.cubs = 2 * fc.mets ∧
  5 * fc.cubs = fc.red_sox ∧
  -- Total number of fans is 585
  fc.yankees + fc.mets + fc.dodgers + fc.red_sox + fc.cubs = 585

/-- The theorem to be proved -/
theorem mets_fans_count (fc : FanCounts) :
  fan_distribution fc → fc.mets = 120 := by
  sorry


end NUMINAMATH_CALUDE_mets_fans_count_l3740_374028


namespace NUMINAMATH_CALUDE_intersection_P_Q_l3740_374049

def P : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 2}
def Q : Set ℝ := {1, 2, 3, 4}

theorem intersection_P_Q : P ∩ Q = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_P_Q_l3740_374049


namespace NUMINAMATH_CALUDE_sin_210_degrees_l3740_374052

theorem sin_210_degrees : Real.sin (210 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_210_degrees_l3740_374052


namespace NUMINAMATH_CALUDE_solve_for_y_l3740_374050

theorem solve_for_y (x y : ℚ) 
  (h1 : x = 103)
  (h2 : x^3 * y - 2 * x^2 * y + x * y - 100 * y = 1061500) : 
  y = 125 / 126 := by
sorry

end NUMINAMATH_CALUDE_solve_for_y_l3740_374050


namespace NUMINAMATH_CALUDE_book_transaction_theorem_l3740_374084

/-- Represents the book transaction problem --/
structure BookTransaction where
  initial_books : ℕ
  sold_books : ℕ
  new_books : ℕ
  p1 : ℚ
  p2 : ℚ

/-- The main theorem for the book transaction problem --/
theorem book_transaction_theorem (t : BookTransaction) 
  (h1 : t.initial_books = 5)
  (h2 : t.sold_books = 4)
  (h3 : t.new_books = 38)
  (h4 : t.p1 * 4 = 38 * t.p2) : 
  (t.initial_books - t.sold_books + t.new_books = 39) ∧ 
  (t.p1 = 9.5 * t.p2) := by
  sorry


end NUMINAMATH_CALUDE_book_transaction_theorem_l3740_374084


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l3740_374069

theorem quadratic_roots_property (m n : ℝ) : 
  m^2 - 1840*m + 2009 = 0 → 
  n^2 - 1840*n + 2009 = 0 → 
  (m^2 - 1841*m + 2009) * (n^2 - 1841*n + 2009) = 2009 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l3740_374069


namespace NUMINAMATH_CALUDE_next_simultaneous_event_l3740_374060

/-- Represents the number of minutes between events for a clock -/
structure ClockEvents where
  lightup : ℕ  -- Number of minutes between light-ups
  ring : ℕ     -- Number of minutes between rings

/-- Calculates the time until the next simultaneous light-up and ring -/
def timeToNextSimultaneousEvent (c : ClockEvents) : ℕ :=
  Nat.lcm c.lightup c.ring

/-- The theorem stating that for a clock that lights up every 9 minutes
    and rings every 60 minutes, the next simultaneous event occurs after 180 minutes -/
theorem next_simultaneous_event :
  let c := ClockEvents.mk 9 60
  timeToNextSimultaneousEvent c = 180 := by
  sorry

end NUMINAMATH_CALUDE_next_simultaneous_event_l3740_374060


namespace NUMINAMATH_CALUDE_students_just_passed_l3740_374092

theorem students_just_passed (total : ℕ) (first_div_percent : ℚ) (second_div_percent : ℚ) 
  (h_total : total = 300)
  (h_first : first_div_percent = 28 / 100)
  (h_second : second_div_percent = 54 / 100)
  (h_no_fail : first_div_percent + second_div_percent ≤ 1) :
  total - (total * first_div_percent).floor - (total * second_div_percent).floor = 54 :=
by sorry

end NUMINAMATH_CALUDE_students_just_passed_l3740_374092


namespace NUMINAMATH_CALUDE_circle_angle_change_l3740_374097

theorem circle_angle_change (R L α r l β : ℝ) : 
  r = R / 2 → 
  l = 3 * L / 2 → 
  L = R * α → 
  l = r * β → 
  β / α = 3 := by sorry

end NUMINAMATH_CALUDE_circle_angle_change_l3740_374097


namespace NUMINAMATH_CALUDE_no_triangle_with_geometric_angles_l3740_374031

theorem no_triangle_with_geometric_angles : ¬∃ (a r : ℕ), 
  a ≥ 10 ∧ 
  a < a * r ∧ 
  a * r < a * r * r ∧ 
  a + a * r + a * r * r = 180 := by
  sorry

end NUMINAMATH_CALUDE_no_triangle_with_geometric_angles_l3740_374031


namespace NUMINAMATH_CALUDE_max_log_product_l3740_374063

theorem max_log_product (a b : ℝ) (ha : a > 1) (hb : b > 1) (hab : a * b = 100) :
  ∃ (m : ℝ), m = 1 ∧ ∀ (x y : ℝ), x > 1 → y > 1 → x * y = 100 → Real.log x * Real.log y ≤ m :=
sorry

end NUMINAMATH_CALUDE_max_log_product_l3740_374063


namespace NUMINAMATH_CALUDE_school_referendum_l3740_374064

theorem school_referendum (U A B : Finset Nat) (h1 : Finset.card U = 250)
  (h2 : Finset.card A = 190) (h3 : Finset.card B = 150)
  (h4 : Finset.card (U \ (A ∪ B)) = 40) :
  Finset.card (A ∩ B) = 130 := by
  sorry

end NUMINAMATH_CALUDE_school_referendum_l3740_374064


namespace NUMINAMATH_CALUDE_younger_brother_age_l3740_374012

theorem younger_brother_age (x y : ℕ) 
  (h1 : x + y = 46) 
  (h2 : y = x / 3 + 10) : 
  y = 19 := by
  sorry

end NUMINAMATH_CALUDE_younger_brother_age_l3740_374012


namespace NUMINAMATH_CALUDE_coefficient_theorem_l3740_374095

def expression (x : ℝ) : ℝ := 2 * (3 * x - 5) + 5 * (6 - 3 * x^2 + 2 * x) - 9 * (4 * x - 2)

theorem coefficient_theorem :
  ∃ (a b c : ℝ), ∀ x, expression x = a * x^2 + b * x + c ∧ a = -15 ∧ b = -20 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_theorem_l3740_374095


namespace NUMINAMATH_CALUDE_arithmetic_seq_nth_term_l3740_374041

/-- Arithmetic sequence with first term 3 and common difference 2 -/
def arithmeticSeq (n : ℕ) : ℝ := 3 + 2 * (n - 1)

/-- Theorem: If the nth term of the arithmetic sequence is 25, then n is 12 -/
theorem arithmetic_seq_nth_term (n : ℕ) :
  arithmeticSeq n = 25 → n = 12 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_seq_nth_term_l3740_374041


namespace NUMINAMATH_CALUDE_max_n_value_l3740_374061

theorem max_n_value (a b c : ℝ) (n : ℕ) 
  (h1 : a > b) (h2 : b > c) 
  (h3 : ∀ a b c, a > b → b > c → 1 / (a - b) + 1 / (b - c) ≥ n^2 / (a - c)) :
  n ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_max_n_value_l3740_374061


namespace NUMINAMATH_CALUDE_problem_solution_l3740_374011

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 1}
def B : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 2}

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := 2 * x^2 + m * x - 1

-- Define the set C
def C (m : ℝ) : Set ℝ := {x : ℝ | f m x ≤ 0}

-- Define the function g
def g (a : ℝ) (m : ℝ) (x : ℝ) : ℝ := 2 * |x - a| - x^2 - m * x

theorem problem_solution :
  (∀ m : ℝ, C m ⊆ (A ∩ B) ↔ -1 ≤ m ∧ m ≤ 1) ∧
  (∀ x : ℝ, f (-4) (1 - x) = f (-4) (1 + x) →
    Set.range (fun x => f (-4) x) ∩ B = {y : ℝ | -3 ≤ y ∧ y ≤ 15}) ∧
  (∀ a : ℝ, 
    (a ≤ -1 → ∀ x : ℝ, f (-4) x + g a (-4) x ≥ -2*a - 2) ∧
    (-1 < a ∧ a < 1 → ∀ x : ℝ, f (-4) x + g a (-4) x ≥ a^2 - 1) ∧
    (1 ≤ a → ∀ x : ℝ, f (-4) x + g a (-4) x ≥ 2*a - 2)) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3740_374011


namespace NUMINAMATH_CALUDE_jason_additional_manager_months_l3740_374003

/-- Calculates the additional months Jason worked as a manager -/
def additional_manager_months (bartender_years : ℕ) (manager_years : ℕ) (total_months : ℕ) : ℕ :=
  total_months - (bartender_years * 12 + manager_years * 12)

/-- Proves that Jason worked 6 additional months as a manager -/
theorem jason_additional_manager_months :
  additional_manager_months 9 3 150 = 6 := by
  sorry

end NUMINAMATH_CALUDE_jason_additional_manager_months_l3740_374003


namespace NUMINAMATH_CALUDE_scientific_notation_equivalence_l3740_374039

/-- Represents the value in billion yuan -/
def original_value : ℝ := 8450

/-- Represents the coefficient in scientific notation -/
def coefficient : ℝ := 8.45

/-- Represents the exponent in scientific notation -/
def exponent : ℤ := 3

/-- Theorem stating that the original value is equal to its scientific notation representation -/
theorem scientific_notation_equivalence :
  original_value = coefficient * (10 : ℝ) ^ exponent :=
sorry

end NUMINAMATH_CALUDE_scientific_notation_equivalence_l3740_374039


namespace NUMINAMATH_CALUDE_min_value_quadratic_expression_l3740_374057

theorem min_value_quadratic_expression :
  ∀ x y : ℝ, 2 * x^2 + 2 * x * y + y^2 - 2 * x + 2 * y + 4 ≥ -1 ∧
  ∃ x y : ℝ, 2 * x^2 + 2 * x * y + y^2 - 2 * x + 2 * y + 4 = -1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_expression_l3740_374057


namespace NUMINAMATH_CALUDE_absolute_value_equation_unique_solution_l3740_374004

theorem absolute_value_equation_unique_solution :
  ∃! x : ℝ, |x - 5| = |x + 3| :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_unique_solution_l3740_374004


namespace NUMINAMATH_CALUDE_valid_arrangements_count_l3740_374085

/-- Represents a student in the line -/
inductive Student
  | boyA
  | boyB
  | girl1
  | girl2
  | girl3

/-- Represents a row of students -/
def Row := List Student

/-- Checks if exactly two of the three girls are adjacent in the row -/
def exactlyTwoGirlsAdjacent (row : Row) : Bool := sorry

/-- Checks if boy A is not at either end of the row -/
def boyANotAtEnds (row : Row) : Bool := sorry

/-- Generates all valid permutations of the students -/
def validPermutations : List Row := sorry

/-- Counts the number of valid arrangements -/
def countValidArrangements : Nat :=
  validPermutations.filter (λ row => exactlyTwoGirlsAdjacent row && boyANotAtEnds row) |>.length

theorem valid_arrangements_count :
  countValidArrangements = 36 := by sorry

end NUMINAMATH_CALUDE_valid_arrangements_count_l3740_374085


namespace NUMINAMATH_CALUDE_probability_is_one_twelfth_l3740_374065

/-- A rectangle in the 2D plane --/
structure Rectangle where
  x_min : ℝ
  x_max : ℝ
  y_min : ℝ
  y_max : ℝ
  h_x : x_min < x_max
  h_y : y_min < y_max

/-- The probability of a point satisfying certain conditions within a rectangle --/
def probability_in_rectangle (R : Rectangle) (P : ℝ × ℝ → Prop) : ℝ :=
  sorry

/-- The specific rectangle in the problem --/
def problem_rectangle : Rectangle := {
  x_min := 0
  x_max := 6
  y_min := 0
  y_max := 2
  h_x := by norm_num
  h_y := by norm_num
}

/-- The condition that needs to be satisfied --/
def condition (p : ℝ × ℝ) : Prop :=
  p.1 < p.2 ∧ p.1 + p.2 < 2

/-- The main theorem --/
theorem probability_is_one_twelfth :
  probability_in_rectangle problem_rectangle condition = 1/12 := by
  sorry

end NUMINAMATH_CALUDE_probability_is_one_twelfth_l3740_374065


namespace NUMINAMATH_CALUDE_breath_holding_difference_l3740_374090

/-- 
Given that:
- Kelly held her breath for 3 minutes
- Brittany held her breath for 20 seconds less than Kelly
- Buffy held her breath for 120 seconds

Prove that Buffy held her breath for 40 seconds less than Brittany
-/
theorem breath_holding_difference : 
  let kelly_time := 3 * 60 -- Kelly's time in seconds
  let brittany_time := kelly_time - 20 -- Brittany's time in seconds
  let buffy_time := 120 -- Buffy's time in seconds
  brittany_time - buffy_time = 40 := by
  sorry

end NUMINAMATH_CALUDE_breath_holding_difference_l3740_374090


namespace NUMINAMATH_CALUDE_inequality_holds_iff_l3740_374044

theorem inequality_holds_iff (a : ℝ) : 
  (∀ x : ℝ, -3 < (x^2 + a*x - 2) / (x^2 - x + 1) ∧ (x^2 + a*x - 2) / (x^2 - x + 1) < 2) ↔ 
  (-1 < a ∧ a < 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_l3740_374044


namespace NUMINAMATH_CALUDE_find_number_l3740_374036

theorem find_number (x : ℝ) : (2 * x - 8 = -12) → x = -2 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l3740_374036


namespace NUMINAMATH_CALUDE_oil_bill_problem_l3740_374074

/-- The oil bill problem -/
theorem oil_bill_problem (january_bill : ℝ) (february_bill : ℝ) (additional_amount : ℝ) :
  january_bill = 180 →
  february_bill / january_bill = 5 / 4 →
  (february_bill + additional_amount) / january_bill = 3 / 2 →
  additional_amount = 45 := by
  sorry

end NUMINAMATH_CALUDE_oil_bill_problem_l3740_374074


namespace NUMINAMATH_CALUDE_sum_of_largest_and_smallest_prime_factors_of_1260_l3740_374093

theorem sum_of_largest_and_smallest_prime_factors_of_1260 : ∃ (p q : Nat), 
  Nat.Prime p ∧ Nat.Prime q ∧ 
  p ∣ 1260 ∧ q ∣ 1260 ∧
  (∀ r : Nat, Nat.Prime r → r ∣ 1260 → p ≤ r ∧ r ≤ q) ∧
  p + q = 9 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_largest_and_smallest_prime_factors_of_1260_l3740_374093


namespace NUMINAMATH_CALUDE_ellipse_x_intercept_l3740_374006

-- Define the ellipse
def ellipse (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + y^2) + Real.sqrt ((x - 4)^2 + (y - 3)^2) = 7

-- Define the foci
def F₁ : ℝ × ℝ := (0, 3)
def F₂ : ℝ × ℝ := (4, 0)

-- Theorem statement
theorem ellipse_x_intercept :
  ellipse 0 0 → -- The ellipse passes through (0,0)
  (∃ x : ℝ, x ≠ 0 ∧ ellipse x 0) → -- There exists another x-intercept
  (∃ x : ℝ, x = 56/11 ∧ ellipse x 0) -- The other x-intercept is (56/11, 0)
  := by sorry

end NUMINAMATH_CALUDE_ellipse_x_intercept_l3740_374006


namespace NUMINAMATH_CALUDE_shoe_rebate_problem_l3740_374055

/-- Calculates the total rebate and quantity discount for a set of shoe purchases --/
def calculate_rebate_and_discount (prices : List ℝ) (rebate_percentages : List ℝ) 
  (discount_threshold_1 : ℝ) (discount_threshold_2 : ℝ) 
  (discount_rate_1 : ℝ) (discount_rate_2 : ℝ) : ℝ × ℝ :=
  sorry

/-- Theorem stating the correct rebate and discount for the given problem --/
theorem shoe_rebate_problem :
  let prices := [28, 35, 40, 45, 50]
  let rebate_percentages := [10, 12, 15, 18, 20]
  let discount_threshold_1 := 200
  let discount_threshold_2 := 250
  let discount_rate_1 := 5
  let discount_rate_2 := 7
  let (total_rebate, quantity_discount) := 
    calculate_rebate_and_discount prices rebate_percentages 
      discount_threshold_1 discount_threshold_2 
      discount_rate_1 discount_rate_2
  total_rebate = 31.1 ∧ quantity_discount = 0 := by
  sorry

end NUMINAMATH_CALUDE_shoe_rebate_problem_l3740_374055


namespace NUMINAMATH_CALUDE_parabola_and_line_equations_l3740_374017

-- Define the parabola E
structure Parabola where
  equation : ℝ → ℝ → Prop

-- Define a point on the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define the focus of the parabola
def focus : Point := ⟨1, 0⟩

-- Define the midpoint M
def M : Point := ⟨2, 1⟩

-- Define the property of A and B being on the parabola E
def on_parabola (E : Parabola) (p : Point) : Prop :=
  E.equation p.x p.y

-- Define the property of M being the midpoint of AB
def is_midpoint (M A B : Point) : Prop :=
  M.x = (A.x + B.x) / 2 ∧ M.y = (A.y + B.y) / 2

-- Theorem statement
theorem parabola_and_line_equations 
  (E : Parabola) (A B : Point) 
  (h1 : on_parabola E A) 
  (h2 : on_parabola E B) 
  (h3 : A ≠ B) 
  (h4 : is_midpoint M A B) :
  (∀ (x y : ℝ), E.equation x y ↔ y^2 = 4*x) ∧ 
  (∀ (x y : ℝ), (y - M.y = 2*(x - M.x)) ↔ (2*x - y - 3 = 0)) :=
sorry

end NUMINAMATH_CALUDE_parabola_and_line_equations_l3740_374017


namespace NUMINAMATH_CALUDE_negative_squares_inequality_l3740_374088

theorem negative_squares_inequality (x b a : ℝ) 
  (h1 : x < b) (h2 : b < a) (h3 : a < 0) : x^2 > b*x ∧ b*x > b^2 := by
  sorry

end NUMINAMATH_CALUDE_negative_squares_inequality_l3740_374088


namespace NUMINAMATH_CALUDE_inequality_group_solution_set_l3740_374070

theorem inequality_group_solution_set :
  ∀ x : ℝ, (x > -3 ∧ x < 5) ↔ (-3 < x ∧ x < 5) :=
by sorry

end NUMINAMATH_CALUDE_inequality_group_solution_set_l3740_374070


namespace NUMINAMATH_CALUDE_abs_sum_equals_two_l3740_374075

theorem abs_sum_equals_two (a b c : ℤ) 
  (h : |a - b|^19 + |c - a|^2010 = 1) : 
  |a - b| + |b - c| + |c - a| = 2 := by
sorry

end NUMINAMATH_CALUDE_abs_sum_equals_two_l3740_374075


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l3740_374040

def A : Set ℝ := {x | x * (x + 1) ≤ 0}
def B : Set ℝ := {x | -1 < x ∧ x < 1}

theorem union_of_A_and_B : A ∪ B = {x : ℝ | -1 ≤ x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l3740_374040


namespace NUMINAMATH_CALUDE_segment_area_approx_l3740_374054

/-- Represents a circular segment -/
structure CircularSegment where
  arcLength : ℝ
  chordLength : ℝ

/-- Calculates the area of a circular segment -/
noncomputable def segmentArea (segment : CircularSegment) : ℝ :=
  sorry

/-- Theorem stating that the area of the given circular segment is approximately 14.6 -/
theorem segment_area_approx :
  let segment : CircularSegment := { arcLength := 10, chordLength := 8 }
  abs (segmentArea segment - 14.6) < 0.1 := by
  sorry

end NUMINAMATH_CALUDE_segment_area_approx_l3740_374054


namespace NUMINAMATH_CALUDE_points_on_line_or_circle_l3740_374048

/-- A point in a 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A line in a 2D plane -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A circle in a 2D plane -/
structure Circle2D where
  center : Point2D
  radius : ℝ

/-- Function to check if three points are collinear -/
def areCollinear (p1 p2 p3 : Point2D) : Prop :=
  (p2.y - p1.y) * (p3.x - p2.x) = (p3.y - p2.y) * (p2.x - p1.x)

/-- Function to generate points based on the described process -/
def generatePoints (p1 p2 p3 : Point2D) : Set Point2D :=
  sorry

/-- The main theorem -/
theorem points_on_line_or_circle (p1 p2 p3 : Point2D) :
  ∃ (l : Line2D) (c : Circle2D), 
    (areCollinear p1 p2 p3 ∧ generatePoints p1 p2 p3 ⊆ {p | p.x * l.a + p.y * l.b + l.c = 0}) ∨
    (¬areCollinear p1 p2 p3 ∧ generatePoints p1 p2 p3 ⊆ {p | (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2}) :=
  sorry

end NUMINAMATH_CALUDE_points_on_line_or_circle_l3740_374048


namespace NUMINAMATH_CALUDE_centroid_division_weight_theorem_l3740_374079

/-- Represents a triangle with a given total weight -/
structure WeightedTriangle where
  totalWeight : ℝ
  weightProportionalToArea : Bool

/-- Represents a line passing through the centroid of a triangle -/
structure CentroidLine where
  triangle : WeightedTriangle

/-- Represents the two parts of a triangle divided by a centroid line -/
structure DividedTriangle where
  centroidLine : CentroidLine
  part1Weight : ℝ
  part2Weight : ℝ

/-- The theorem to be proved -/
theorem centroid_division_weight_theorem (t : WeightedTriangle) (l : CentroidLine) (d : DividedTriangle) :
  t.totalWeight = 900 ∧ t.weightProportionalToArea = true ∧ l.triangle = t ∧ d.centroidLine = l →
  d.part1Weight ≥ 400 ∧ d.part2Weight ≥ 400 :=
by sorry

end NUMINAMATH_CALUDE_centroid_division_weight_theorem_l3740_374079


namespace NUMINAMATH_CALUDE_special_rhombus_center_distance_l3740_374059

/-- A rhombus with a specific acute angle and projection length. -/
structure SpecialRhombus where
  /-- The acute angle of the rhombus in degrees -/
  acute_angle : ℝ
  /-- The length of the projection of side AB onto side AD -/
  projection_length : ℝ
  /-- The acute angle is 45 degrees -/
  angle_is_45 : acute_angle = 45
  /-- The projection length is 12 -/
  projection_is_12 : projection_length = 12

/-- The distance from the center of the rhombus to any side -/
def center_to_side_distance (r : SpecialRhombus) : ℝ := 6

/-- 
Theorem: In a rhombus where the acute angle is 45° and the projection of one side 
onto an adjacent side is 12, the distance from the center to any side is 6.
-/
theorem special_rhombus_center_distance (r : SpecialRhombus) : 
  center_to_side_distance r = 6 := by
  sorry

end NUMINAMATH_CALUDE_special_rhombus_center_distance_l3740_374059


namespace NUMINAMATH_CALUDE_primes_arithmetic_sequence_ones_digit_l3740_374000

/-- A function that returns the ones digit of a natural number -/
def onesDigit (n : ℕ) : ℕ := n % 10

/-- A function that checks if a natural number is prime -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem primes_arithmetic_sequence_ones_digit 
  (p q r s : ℕ) 
  (hp : isPrime p) 
  (hq : isPrime q) 
  (hr : isPrime r) 
  (hs : isPrime s)
  (hseq : q = p + 8 ∧ r = q + 8 ∧ s = r + 8)
  (hp_gt_5 : p > 5) :
  onesDigit p = 3 := by
sorry

end NUMINAMATH_CALUDE_primes_arithmetic_sequence_ones_digit_l3740_374000


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l3740_374037

open Real

-- Define the proposition
def P (a : ℝ) : Prop := ∀ x ∈ Set.Ioo 1 2, x^2 - a > 0

-- State the theorem
theorem necessary_but_not_sufficient_condition :
  (∃ a : ℝ, a < 2 ∧ ¬(P a)) ∧
  (∀ a : ℝ, P a → a < 2) :=
sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l3740_374037


namespace NUMINAMATH_CALUDE_two_person_subcommittees_from_eight_l3740_374056

theorem two_person_subcommittees_from_eight (n : ℕ) (k : ℕ) : 
  n = 8 → k = 2 → Nat.choose n k = 28 := by
  sorry

end NUMINAMATH_CALUDE_two_person_subcommittees_from_eight_l3740_374056


namespace NUMINAMATH_CALUDE_average_speed_ratio_l3740_374016

/-- Given that Eddy travels 480 km in 3 hours and Freddy travels 300 km in 4 hours,
    prove that the ratio of their average speeds is 32:15. -/
theorem average_speed_ratio (eddy_distance : ℝ) (eddy_time : ℝ) (freddy_distance : ℝ) (freddy_time : ℝ)
    (h1 : eddy_distance = 480)
    (h2 : eddy_time = 3)
    (h3 : freddy_distance = 300)
    (h4 : freddy_time = 4) :
    (eddy_distance / eddy_time) / (freddy_distance / freddy_time) = 32 / 15 := by
  sorry

#check average_speed_ratio

end NUMINAMATH_CALUDE_average_speed_ratio_l3740_374016


namespace NUMINAMATH_CALUDE_marigold_sale_ratio_l3740_374038

/-- Proves that the ratio of marigolds sold on the third day to the second day is 2:1 --/
theorem marigold_sale_ratio :
  ∀ (day3 : ℕ),
  14 + 25 + day3 = 89 →
  (day3 : ℚ) / 25 = 2 := by
  sorry

end NUMINAMATH_CALUDE_marigold_sale_ratio_l3740_374038


namespace NUMINAMATH_CALUDE_composition_is_homomorphism_l3740_374029

variable {G H I : Type*} [Group G] [Group H] [Group I]
variable (φ : G → H) (ψ : H → I)

-- φ is a group homomorphism
variable (hφ : ∀ x y : G, φ (x * y) = φ x * φ y)

-- ψ is a group homomorphism
variable (hψ : ∀ x y : H, ψ (x * y) = ψ x * ψ y)

theorem composition_is_homomorphism :
  ∀ x y : G, (ψ ∘ φ) (x * y) = (ψ ∘ φ) x * (ψ ∘ φ) y :=
by sorry

end NUMINAMATH_CALUDE_composition_is_homomorphism_l3740_374029


namespace NUMINAMATH_CALUDE_lunch_with_tip_l3740_374033

/-- Calculate the total amount spent on lunch including tip -/
theorem lunch_with_tip (lunch_cost : ℝ) (tip_percentage : ℝ) :
  lunch_cost = 50.20 →
  tip_percentage = 20 →
  lunch_cost * (1 + tip_percentage / 100) = 60.24 := by
  sorry

end NUMINAMATH_CALUDE_lunch_with_tip_l3740_374033


namespace NUMINAMATH_CALUDE_min_value_of_expression_l3740_374080

theorem min_value_of_expression :
  (∀ x : ℝ, (x + 1) * (x + 3) * (x + 5) * (x + 7) + 2024 ≥ 2008) ∧
  (∃ x : ℝ, (x + 1) * (x + 3) * (x + 5) * (x + 7) + 2024 = 2008) := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l3740_374080


namespace NUMINAMATH_CALUDE_syllogism_correctness_l3740_374002

theorem syllogism_correctness : 
  (∀ n : ℕ, (n : ℤ) = n) →  -- All natural numbers are integers
  (4 : ℕ) = 4 →             -- 4 is a natural number
  (4 : ℤ) = 4               -- Therefore, 4 is an integer
  := by sorry

end NUMINAMATH_CALUDE_syllogism_correctness_l3740_374002


namespace NUMINAMATH_CALUDE_fraction_base_k_representation_l3740_374083

/-- Represents a repeating decimal in base k -/
def repeating_decimal (k : ℕ) (a b : ℕ) := (a : ℚ) / k + (b : ℚ) / (k^2 - 1)

theorem fraction_base_k_representation (k : ℕ) :
  k > 1 →
  repeating_decimal k 1 4 = 5 / 31 →
  k = 14 := by
sorry

end NUMINAMATH_CALUDE_fraction_base_k_representation_l3740_374083


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3740_374078

/-- A geometric sequence with sum of first n terms S_n -/
def GeometricSequence (S : ℕ → ℝ) : Prop :=
  ∃ (a r : ℝ), ∀ n : ℕ, S n = a * (1 - r^n) / (1 - r)

theorem geometric_sequence_sum (S : ℕ → ℝ) :
  GeometricSequence S →
  S 5 = 10 →
  S 10 = 50 →
  S 15 = 210 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3740_374078


namespace NUMINAMATH_CALUDE_jia_candies_theorem_l3740_374001

/-- Represents a configuration of lines in a plane -/
structure LineConfiguration where
  num_lines : ℕ
  num_parallel_pairs : ℕ

/-- Calculates the number of intersections for a given number of lines -/
def num_intersections (n : ℕ) : ℕ :=
  n * (n - 1) / 2

/-- Calculates the total number of candies for a given line configuration -/
def total_candies (config : LineConfiguration) : ℕ :=
  num_intersections config.num_lines + config.num_parallel_pairs

/-- Theorem: Given 5 lines with one parallel pair, Jia receives 11 candies -/
theorem jia_candies_theorem (config : LineConfiguration) 
  (h1 : config.num_lines = 5)
  (h2 : config.num_parallel_pairs = 1) :
  total_candies config = 11 := by
  sorry

end NUMINAMATH_CALUDE_jia_candies_theorem_l3740_374001


namespace NUMINAMATH_CALUDE_greatest_b_value_l3740_374025

theorem greatest_b_value (b : ℝ) : 
  (∀ x : ℝ, -x^2 + 9*x - 18 ≥ 0 → x ≤ 6) ∧ 
  (-6^2 + 9*6 - 18 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_greatest_b_value_l3740_374025


namespace NUMINAMATH_CALUDE_circle_radius_from_tangents_l3740_374023

/-- Given a circle with diameter AB, tangents AD and BC, and a line through D and C
    intersecting the circle at E, prove that the radius is (c+d)/2 when AD = c and BC = d. -/
theorem circle_radius_from_tangents (c d : ℝ) (h : c ≠ d) :
  let circle : Set (ℝ × ℝ) := {p | ∃ (x y : ℝ), p = (x, y) ∧ (x - 0)^2 + (y - 0)^2 = ((c + d)/2)^2}
  let A : ℝ × ℝ := (-(c + d)/2, 0)
  let B : ℝ × ℝ := ((c + d)/2, 0)
  let D : ℝ × ℝ := (-c, c)
  let C : ℝ × ℝ := (d, d)
  let E : ℝ × ℝ := (0, (c + d)/2)
  (∀ p ∈ circle, (p.1 - A.1)^2 + (p.2 - A.2)^2 = ((c + d)/2)^2) ∧
  (∀ p ∈ circle, (p.1 - B.1)^2 + (p.2 - B.2)^2 = ((c + d)/2)^2) ∧
  (D ∉ circle) ∧ (C ∉ circle) ∧
  ((D.1 - A.1) * (D.2 - A.2) + (D.1 - 0) * (D.2 - 0) = 0) ∧
  ((C.1 - B.1) * (C.2 - B.2) + (C.1 - 0) * (C.2 - 0) = 0) ∧
  (E ∈ circle) ∧
  (D.2 - A.2)/(D.1 - A.1) = (E.2 - D.2)/(E.1 - D.1) ∧
  (C.2 - B.2)/(C.1 - B.1) = (E.2 - C.2)/(E.1 - C.1) →
  (c + d)/2 = (c + d)/2 := by
sorry

end NUMINAMATH_CALUDE_circle_radius_from_tangents_l3740_374023


namespace NUMINAMATH_CALUDE_find_T_l3740_374019

theorem find_T : ∃ T : ℚ, (1/3 : ℚ) * (1/8 : ℚ) * T = (1/4 : ℚ) * (1/6 : ℚ) * 120 ∧ T = 120 := by
  sorry

end NUMINAMATH_CALUDE_find_T_l3740_374019


namespace NUMINAMATH_CALUDE_fraction_simplification_l3740_374072

theorem fraction_simplification :
  (1 / 3 + 1 / 4) / (2 / 5 - 1 / 6) = 5 / 2 := by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3740_374072


namespace NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l3740_374051

theorem tan_alpha_plus_pi_fourth (α : Real) 
  (h1 : π/2 < α ∧ α < π) -- α is in the second quadrant
  (h2 : Real.sin α = 3/5) : 
  Real.tan (α + π/4) = 1/7 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l3740_374051


namespace NUMINAMATH_CALUDE_largest_difference_in_grid_l3740_374035

/-- A type representing a 20x20 grid of integers -/
def Grid := Fin 20 → Fin 20 → Fin 400

/-- The property that a grid contains all integers from 1 to 400 -/
def contains_all_integers (g : Grid) : Prop :=
  ∀ n : Fin 400, ∃ i j : Fin 20, g i j = n

/-- The property that there exist two numbers in the same row or column with a difference of at least N -/
def has_difference_at_least (g : Grid) (N : ℕ) : Prop :=
  ∃ i j k : Fin 20, (g i j).val + N ≤ (g i k).val ∨ (g j i).val + N ≤ (g k i).val

/-- The main theorem: 209 is the largest N satisfying the condition -/
theorem largest_difference_in_grid :
  (∀ g : Grid, contains_all_integers g → has_difference_at_least g 209) ∧
  ¬(∀ g : Grid, contains_all_integers g → has_difference_at_least g 210) :=
sorry

end NUMINAMATH_CALUDE_largest_difference_in_grid_l3740_374035


namespace NUMINAMATH_CALUDE_crane_flock_size_l3740_374027

theorem crane_flock_size (duck_flock_size : ℕ) (min_ducks : ℕ) (crane_flock_size : ℕ) : 
  duck_flock_size = 13 →
  min_ducks = 221 →
  ∃ (num_duck_flocks num_crane_flocks : ℕ), 
    num_duck_flocks * duck_flock_size = num_crane_flocks * crane_flock_size →
    num_duck_flocks * duck_flock_size ≥ min_ducks →
    crane_flock_size = 221 := by
  sorry

end NUMINAMATH_CALUDE_crane_flock_size_l3740_374027


namespace NUMINAMATH_CALUDE_scientific_notation_equality_l3740_374096

theorem scientific_notation_equality : ∃ (a : ℝ) (n : ℤ), 
  0.00000012 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 1.2 ∧ n = -7 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_equality_l3740_374096


namespace NUMINAMATH_CALUDE_tom_distance_before_karen_wins_l3740_374094

/-- Represents the car race scenario between Karen and Tom -/
def CarRace (karen_speed tom_speed : ℝ) (karen_delay : ℝ) (winning_margin : ℝ) : Prop :=
  let race_time := (karen_delay * karen_speed + winning_margin) / (karen_speed - tom_speed)
  tom_speed * race_time = 24

/-- Theorem stating the distance Tom drives before Karen wins -/
theorem tom_distance_before_karen_wins :
  CarRace 60 45 (4/60) 4 :=
by sorry

end NUMINAMATH_CALUDE_tom_distance_before_karen_wins_l3740_374094


namespace NUMINAMATH_CALUDE_square_root_of_1024_l3740_374066

theorem square_root_of_1024 (y : ℝ) (h1 : y > 0) (h2 : y^2 = 1024) : y = 32 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_1024_l3740_374066


namespace NUMINAMATH_CALUDE_division_problem_l3740_374018

theorem division_problem (a b c : ℚ) 
  (h1 : a / b = 3) 
  (h2 : b / c = 2 / 5) : 
  c / a = 5 / 6 := by sorry

end NUMINAMATH_CALUDE_division_problem_l3740_374018


namespace NUMINAMATH_CALUDE_probability_is_one_sixth_l3740_374082

def cards : Finset Int := {0, -1, 2, -3}

def is_in_fourth_quadrant (m n : Int) : Bool :=
  m > 0 ∧ n < 0

def probability_in_fourth_quadrant : ℚ :=
  (Finset.filter (fun (p : Int × Int) => is_in_fourth_quadrant p.1 p.2) 
    (Finset.product cards cards)).card / (cards.card * (cards.card - 1))

theorem probability_is_one_sixth : 
  probability_in_fourth_quadrant = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_probability_is_one_sixth_l3740_374082


namespace NUMINAMATH_CALUDE_netflix_binge_watching_l3740_374042

theorem netflix_binge_watching (episode_length : ℕ) (daily_watch_time : ℕ) (days_to_finish : ℕ) : 
  episode_length = 20 →
  daily_watch_time = 120 →
  days_to_finish = 15 →
  (daily_watch_time * days_to_finish) / episode_length = 90 :=
by
  sorry

end NUMINAMATH_CALUDE_netflix_binge_watching_l3740_374042


namespace NUMINAMATH_CALUDE_mr_a_speed_l3740_374058

/-- Proves that Mr. A's speed is 30 kmph given the problem conditions --/
theorem mr_a_speed (initial_distance : ℝ) (mrs_a_speed : ℝ) (bee_speed : ℝ) (bee_distance : ℝ)
  (h1 : initial_distance = 120)
  (h2 : mrs_a_speed = 10)
  (h3 : bee_speed = 60)
  (h4 : bee_distance = 180) :
  ∃ (mr_a_speed : ℝ), mr_a_speed = 30 ∧ 
    (bee_distance / bee_speed) * (mr_a_speed + mrs_a_speed) = initial_distance :=
by sorry

end NUMINAMATH_CALUDE_mr_a_speed_l3740_374058


namespace NUMINAMATH_CALUDE_range_of_f_l3740_374032

def f (x : ℕ) : ℤ := 2 * x - 3

def domain : Set ℕ := {x | 1 ≤ x ∧ x ≤ 5}

theorem range_of_f :
  {y | ∃ x ∈ domain, f x = y} = {-1, 1, 3, 5, 7} := by sorry

end NUMINAMATH_CALUDE_range_of_f_l3740_374032


namespace NUMINAMATH_CALUDE_complex_number_real_imag_equal_l3740_374010

theorem complex_number_real_imag_equal (a : ℝ) : 
  let x : ℂ := (1 + a * Complex.I) * (2 + Complex.I)
  (x.re = x.im) → a = 1/3 := by sorry

end NUMINAMATH_CALUDE_complex_number_real_imag_equal_l3740_374010


namespace NUMINAMATH_CALUDE_surface_area_increase_l3740_374086

/-- The increase in surface area when a cube of edge length a is cut into 27 congruent smaller cubes -/
theorem surface_area_increase (a : ℝ) (h : a > 0) : 
  let original_surface_area := 6 * a^2
  let small_cube_edge := a / 3
  let small_cube_surface_area := 6 * small_cube_edge^2
  let total_new_surface_area := 27 * small_cube_surface_area
  total_new_surface_area - original_surface_area = 12 * a^2 := by
sorry


end NUMINAMATH_CALUDE_surface_area_increase_l3740_374086


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3740_374068

theorem inequality_solution_set (k : ℝ) :
  (∃ x : ℝ, |x - 2| - |x - 5| > k) → k < 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3740_374068


namespace NUMINAMATH_CALUDE_cosine_value_l3740_374099

theorem cosine_value (α : Real) 
  (h1 : 0 < α) (h2 : α < π) 
  (h3 : 3 * Real.sin (2 * α) = Real.sin α) : 
  Real.cos (α - π) = -1/6 := by
sorry

end NUMINAMATH_CALUDE_cosine_value_l3740_374099


namespace NUMINAMATH_CALUDE_solution_set_f_less_than_2_range_of_a_for_solutions_l3740_374067

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 1| - |x - 1|

-- Theorem for the solution set of f(x) < 2
theorem solution_set_f_less_than_2 :
  {x : ℝ | f x < 2} = Set.Ioo (-4 : ℝ) (2/3) := by sorry

-- Theorem for the range of a
theorem range_of_a_for_solutions (a : ℝ) :
  (∃ x, f x ≤ a - a^2/2) ↔ a ∈ Set.Icc (-1 : ℝ) 3 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_less_than_2_range_of_a_for_solutions_l3740_374067


namespace NUMINAMATH_CALUDE_adoption_fee_is_correct_l3740_374022

/-- The adoption fee for an untrained seeing-eye dog. -/
def adoption_fee : ℝ := 150

/-- The weekly training cost for a seeing-eye dog. -/
def weekly_training_cost : ℝ := 250

/-- The number of weeks of training required. -/
def training_weeks : ℕ := 12

/-- The total cost of certification. -/
def certification_cost : ℝ := 3000

/-- The percentage of certification cost covered by insurance. -/
def insurance_coverage : ℝ := 0.9

/-- The total out-of-pocket cost for John. -/
def total_out_of_pocket : ℝ := 3450

/-- Theorem stating that the adoption fee is correct given the conditions. -/
theorem adoption_fee_is_correct : 
  adoption_fee + (weekly_training_cost * training_weeks) + 
  (certification_cost * (1 - insurance_coverage)) = total_out_of_pocket :=
by sorry

end NUMINAMATH_CALUDE_adoption_fee_is_correct_l3740_374022


namespace NUMINAMATH_CALUDE_trader_profit_percentage_l3740_374024

theorem trader_profit_percentage (original_price : ℝ) (original_price_positive : original_price > 0) : 
  let discount_rate : ℝ := 0.20
  let purchase_price : ℝ := original_price * (1 - discount_rate)
  let markup_rate : ℝ := 0.60
  let selling_price : ℝ := purchase_price * (1 + markup_rate)
  let profit : ℝ := selling_price - original_price
  let profit_percentage : ℝ := (profit / original_price) * 100
  profit_percentage = 28 := by
sorry

end NUMINAMATH_CALUDE_trader_profit_percentage_l3740_374024


namespace NUMINAMATH_CALUDE_power_of_product_l3740_374091

theorem power_of_product (a b : ℝ) : (2 * a * b^2)^3 = 8 * a^3 * b^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_l3740_374091


namespace NUMINAMATH_CALUDE_second_tree_groups_count_l3740_374034

/-- Represents the number of rings in a group -/
def rings_per_group : ℕ := 6

/-- Represents the number of ring groups in the first tree -/
def first_tree_groups : ℕ := 70

/-- Represents the age difference between the first and second tree in years -/
def age_difference : ℕ := 180

/-- Calculates the number of ring groups in the second tree -/
def second_tree_groups : ℕ := 
  (first_tree_groups * rings_per_group - age_difference) / rings_per_group

theorem second_tree_groups_count : second_tree_groups = 40 := by
  sorry

end NUMINAMATH_CALUDE_second_tree_groups_count_l3740_374034


namespace NUMINAMATH_CALUDE_star_example_l3740_374077

-- Define the ⋆ operation
def star (a b c d : ℚ) : ℚ := a * c * (d / (2 * b))

-- Theorem statement
theorem star_example : star 5 6 9 4 = 15 := by
  sorry

end NUMINAMATH_CALUDE_star_example_l3740_374077


namespace NUMINAMATH_CALUDE_greatest_integer_solution_greatest_integer_value_minus_four_is_solution_minus_four_is_greatest_l3740_374008

theorem greatest_integer_solution (x : ℤ) : (5 - 4 * x > 17) ↔ x < -3 :=
  sorry

theorem greatest_integer_value : ∀ x : ℤ, (5 - 4 * x > 17) → x ≤ -4 :=
  sorry

theorem minus_four_is_solution : 5 - 4 * (-4) > 17 :=
  sorry

theorem minus_four_is_greatest : ∀ x : ℤ, x > -4 → ¬(5 - 4 * x > 17) :=
  sorry

end NUMINAMATH_CALUDE_greatest_integer_solution_greatest_integer_value_minus_four_is_solution_minus_four_is_greatest_l3740_374008


namespace NUMINAMATH_CALUDE_power_two_ge_square_l3740_374030

theorem power_two_ge_square (n : ℕ) : 2^n ≥ n^2 ↔ n ≠ 3 :=
sorry

end NUMINAMATH_CALUDE_power_two_ge_square_l3740_374030


namespace NUMINAMATH_CALUDE_line_decreasing_direct_proportion_range_l3740_374089

/-- A line passing through two points -/
structure Line where
  k : ℝ
  b : ℝ
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ
  eq₁ : y₁ = k * x₁ + b
  eq₂ : y₂ = k * x₂ + b

/-- A direct proportion function passing through two points -/
structure DirectProportion where
  m : ℝ
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ
  eq₁ : y₁ = (1 - 2*m) * x₁
  eq₂ : y₂ = (1 - 2*m) * x₂

theorem line_decreasing (l : Line) (h₁ : l.k < 0) (h₂ : l.x₁ < l.x₂) : l.y₁ > l.y₂ := by
  sorry

theorem direct_proportion_range (d : DirectProportion) (h₁ : d.x₁ < d.x₂) (h₂ : d.y₁ > d.y₂) : d.m > 1/2 := by
  sorry

end NUMINAMATH_CALUDE_line_decreasing_direct_proportion_range_l3740_374089


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l3740_374046

theorem fractional_equation_solution :
  ∃ x : ℝ, (1 / x = 2 / (x + 3)) ∧ (x = 3) := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l3740_374046


namespace NUMINAMATH_CALUDE_sum_seven_consecutive_integers_l3740_374073

theorem sum_seven_consecutive_integers (n : ℤ) :
  n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) = 7 * n + 21 := by
  sorry

end NUMINAMATH_CALUDE_sum_seven_consecutive_integers_l3740_374073


namespace NUMINAMATH_CALUDE_simplify_trigonometric_expression_sector_central_angle_l3740_374026

-- Problem 1
theorem simplify_trigonometric_expression (x : ℝ) :
  (1 + Real.sin x) / Real.cos x * Real.sin (2 * x) / (2 * (Real.cos (π / 4 - x / 2))^2) = 2 * Real.sin x :=
sorry

-- Problem 2
theorem sector_central_angle (r α : ℝ) (h1 : 2 * r + α * r = 4) (h2 : 1/2 * α * r^2 = 1) :
  α = 2 :=
sorry

end NUMINAMATH_CALUDE_simplify_trigonometric_expression_sector_central_angle_l3740_374026


namespace NUMINAMATH_CALUDE_assignment_schemes_with_girl_l3740_374081

theorem assignment_schemes_with_girl (num_boys num_girls : ℕ) 
  (h1 : num_boys = 4) 
  (h2 : num_girls = 3) 
  (total_people : ℕ := num_boys + num_girls) 
  (tasks : ℕ := 3) : 
  (total_people * (total_people - 1) * (total_people - 2)) - 
  (num_boys * (num_boys - 1) * (num_boys - 2)) = 186 := by
  sorry

#check assignment_schemes_with_girl

end NUMINAMATH_CALUDE_assignment_schemes_with_girl_l3740_374081


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3740_374045

/-- Given a complex number z satisfying (z - 3i)(2 + i) = 5i, prove that z = 2 + 5i -/
theorem complex_equation_solution (z : ℂ) (h : (z - 3*Complex.I)*(2 + Complex.I) = 5*Complex.I) : 
  z = 2 + 5*Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3740_374045


namespace NUMINAMATH_CALUDE_proposition_relation_necessary_not_sufficient_l3740_374007

theorem proposition_relation (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 2 * a * x + 1 > 0) ↔ (0 ≤ a ∧ a < 1) :=
by sorry

theorem necessary_not_sufficient :
  (∃ a : ℝ, (∀ x : ℝ, a * x^2 + 2 * a * x + 1 > 0) ∧ (a ≤ 0 ∨ a ≥ 1)) :=
by sorry

end NUMINAMATH_CALUDE_proposition_relation_necessary_not_sufficient_l3740_374007


namespace NUMINAMATH_CALUDE_trapezoid_segment_length_l3740_374087

/-- Given a trapezoid ABCD, proves that if the ratio of the areas of triangles ABC and ADC is 5:2,
    and the sum of AB and CD is 280, then AB equals 200. -/
theorem trapezoid_segment_length (A B C D : Point) (h : ℝ) :
  let triangle_ABC := (1/2) * AB * h
  let triangle_ADC := (1/2) * CD * h
  triangle_ABC / triangle_ADC = 5/2 →
  AB + CD = 280 →
  AB = 200 :=
by
  sorry


end NUMINAMATH_CALUDE_trapezoid_segment_length_l3740_374087


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l3740_374009

theorem inverse_variation_problem (y z : ℝ) (k : ℝ) :
  (∀ y z, y^2 * Real.sqrt z = k) →  -- y² varies inversely with √z
  (3^2 * Real.sqrt 16 = k) →        -- y = 3 when z = 16
  (6^2 * Real.sqrt z = k) →         -- condition for y = 6
  z = 1 :=                          -- prove z = 1 when y = 6
by
  sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l3740_374009
