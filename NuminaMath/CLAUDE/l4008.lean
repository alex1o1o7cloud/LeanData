import Mathlib

namespace NUMINAMATH_CALUDE_relationship_abc_l4008_400872

theorem relationship_abc (x : ℝ) (a b c : ℝ) 
  (h1 : x > Real.exp (-1)) 
  (h2 : x < 1) 
  (h3 : a = Real.log x) 
  (h4 : b = (1/2) ^ (Real.log x)) 
  (h5 : c = Real.exp (Real.log x)) : 
  b > c ∧ c > a := by
  sorry

end NUMINAMATH_CALUDE_relationship_abc_l4008_400872


namespace NUMINAMATH_CALUDE_equal_population_after_14_years_second_village_initial_population_is_correct_l4008_400856

/-- The initial population of Village X -/
def village_x_initial : ℕ := 70000

/-- The yearly decrease in population of Village X -/
def village_x_decrease : ℕ := 1200

/-- The yearly increase in population of the second village -/
def village_2_increase : ℕ := 800

/-- The number of years after which the populations will be equal -/
def years_until_equal : ℕ := 14

/-- The initial population of the second village -/
def village_2_initial : ℕ := 42000

theorem equal_population_after_14_years :
  village_x_initial - village_x_decrease * years_until_equal = 
  village_2_initial + village_2_increase * years_until_equal :=
by sorry

/-- The theorem stating that the calculated initial population of the second village is correct -/
theorem second_village_initial_population_is_correct : village_2_initial = 42000 :=
by sorry

end NUMINAMATH_CALUDE_equal_population_after_14_years_second_village_initial_population_is_correct_l4008_400856


namespace NUMINAMATH_CALUDE_age_sum_in_five_years_l4008_400893

/-- Given a person (Mike) who is 30 years younger than his mom, and the sum of their ages is 70 years,
    the sum of their ages in 5 years will be 80 years. -/
theorem age_sum_in_five_years (mike_age mom_age : ℕ) : 
  mike_age = mom_age - 30 → 
  mike_age + mom_age = 70 → 
  (mike_age + 5) + (mom_age + 5) = 80 := by
sorry

end NUMINAMATH_CALUDE_age_sum_in_five_years_l4008_400893


namespace NUMINAMATH_CALUDE_sum_of_coefficients_is_thirteen_thirds_l4008_400865

/-- Given two functions f and g, where f is linear and g(f(x)) = 4x + 2,
    prove that the sum of coefficients of f is 13/3 -/
theorem sum_of_coefficients_is_thirteen_thirds
  (f g : ℝ → ℝ)
  (a b : ℝ)
  (h1 : ∀ x, f x = a * x + b)
  (h2 : ∀ x, g x = 3 * x - 7)
  (h3 : ∀ x, g (f x) = 4 * x + 2) :
  a + b = 13 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_is_thirteen_thirds_l4008_400865


namespace NUMINAMATH_CALUDE_min_value_and_max_value_l4008_400832

theorem min_value_and_max_value :
  (∀ x : ℝ, x > 1 → (x + 1 / (x - 1)) ≥ 3) ∧
  (∃ x : ℝ, x > 1 ∧ (x + 1 / (x - 1)) = 3) ∧
  (∀ x : ℝ, 0 < x ∧ x < 10 → Real.sqrt (x * (10 - x)) ≤ 5) ∧
  (∃ x : ℝ, 0 < x ∧ x < 10 ∧ Real.sqrt (x * (10 - x)) = 5) :=
by sorry

end NUMINAMATH_CALUDE_min_value_and_max_value_l4008_400832


namespace NUMINAMATH_CALUDE_rogers_money_l4008_400809

/-- Roger's money calculation -/
theorem rogers_money (initial_amount spent_amount received_amount : ℕ) :
  initial_amount = 45 →
  spent_amount = 20 →
  received_amount = 46 →
  initial_amount - spent_amount + received_amount = 71 := by
  sorry

end NUMINAMATH_CALUDE_rogers_money_l4008_400809


namespace NUMINAMATH_CALUDE_only_valid_root_l4008_400816

def original_equation (x : ℝ) : Prop :=
  (2 * x^2) / (x - 1) - (2 * x + 7) / 3 + (4 - 6 * x) / (x - 1) + 1 = 0

def transformed_equation (x : ℝ) : Prop :=
  x^2 - 5 * x + 4 = 0

theorem only_valid_root :
  (∀ x : ℝ, transformed_equation x ↔ (x = 4 ∨ x = 1)) →
  (∀ x : ℝ, original_equation x ↔ x = 4) :=
sorry

end NUMINAMATH_CALUDE_only_valid_root_l4008_400816


namespace NUMINAMATH_CALUDE_smallest_common_multiple_of_5_and_8_l4008_400829

theorem smallest_common_multiple_of_5_and_8 : 
  ∃ (n : ℕ), n > 0 ∧ Even n ∧ 5 ∣ n ∧ 8 ∣ n ∧ ∀ (m : ℕ), m > 0 → Even m → 5 ∣ m → 8 ∣ m → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_common_multiple_of_5_and_8_l4008_400829


namespace NUMINAMATH_CALUDE_quadratic_equation_coefficients_l4008_400897

theorem quadratic_equation_coefficients :
  ∀ (a b c d e f : ℝ),
  (∀ x, a * x^2 + b * x + c = d * x^2 + e * x + f) →
  (a - d = 4) →
  (b - e = -2 ∧ c - f = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_coefficients_l4008_400897


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l4008_400853

/-- Arithmetic sequence sum -/
def arithmetic_sum (a : ℕ → ℚ) (n : ℕ) : ℚ := (n : ℚ) * (a 1 + a n) / 2

theorem arithmetic_sequence_ratio (a b : ℕ → ℚ) (S T : ℕ → ℚ) :
  (∀ n, S n = arithmetic_sum a n) →
  (∀ n, T n = arithmetic_sum b n) →
  (∀ n, S n / T n = (7 * n + 1) / (4 * n + 27)) →
  a 11 / b 11 = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l4008_400853


namespace NUMINAMATH_CALUDE_min_value_of_y_l4008_400818

theorem min_value_of_y (x : ℝ) (hx : x > 0) : x + 1/x ≥ 2 ∧ (x + 1/x = 2 ↔ x = 1) := by sorry

end NUMINAMATH_CALUDE_min_value_of_y_l4008_400818


namespace NUMINAMATH_CALUDE_correct_calculation_l4008_400806

theorem correct_calculation : (-4) * (-3) * (-5) = -60 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l4008_400806


namespace NUMINAMATH_CALUDE_calculate_not_less_than_50_l4008_400858

/-- Represents the frequency of teachers in different age groups -/
structure TeacherAgeFrequency where
  less_than_30 : ℝ
  between_30_and_50 : ℝ
  not_less_than_50 : ℝ

/-- The sum of all frequencies in a probability distribution is 1 -/
axiom sum_of_frequencies (f : TeacherAgeFrequency) : 
  f.less_than_30 + f.between_30_and_50 + f.not_less_than_50 = 1

/-- Theorem: Given the frequencies for two age groups, we can calculate the third -/
theorem calculate_not_less_than_50 (f : TeacherAgeFrequency) 
    (h1 : f.less_than_30 = 0.3) 
    (h2 : f.between_30_and_50 = 0.5) : 
  f.not_less_than_50 = 0.2 := by
  sorry


end NUMINAMATH_CALUDE_calculate_not_less_than_50_l4008_400858


namespace NUMINAMATH_CALUDE_imaginary_complex_magnitude_l4008_400883

theorem imaginary_complex_magnitude (a : ℝ) : 
  (∃ (b : ℝ), (Complex.I : ℂ) * b = (a + 2 * Complex.I) / (1 + Complex.I)) → 
  Complex.abs (a + Complex.I) = Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_imaginary_complex_magnitude_l4008_400883


namespace NUMINAMATH_CALUDE_jacks_purchase_cost_l4008_400860

/-- The cost of Jack's purchase of a squat rack and barbell -/
theorem jacks_purchase_cost (squat_rack_cost : ℝ) (barbell_cost : ℝ) : 
  squat_rack_cost = 2500 →
  barbell_cost = squat_rack_cost / 10 →
  squat_rack_cost + barbell_cost = 2750 := by
sorry

end NUMINAMATH_CALUDE_jacks_purchase_cost_l4008_400860


namespace NUMINAMATH_CALUDE_salt_trade_initial_investment_l4008_400826

/-- Represents the merchant's salt trading scenario -/
structure SaltTrade where
  initial_investment : ℕ  -- Initial investment in rubles
  first_profit : ℕ        -- Profit from first sale in rubles
  second_profit : ℕ       -- Profit from second sale in rubles

/-- Theorem stating the initial investment in the salt trade scenario -/
theorem salt_trade_initial_investment (trade : SaltTrade) 
  (h1 : trade.first_profit = 100)
  (h2 : trade.second_profit = 120)
  (h3 : (trade.initial_investment + trade.first_profit + trade.second_profit) = 
        (trade.initial_investment + trade.first_profit) * 
        (trade.initial_investment + trade.first_profit) / trade.initial_investment) :
  trade.initial_investment = 500 := by
  sorry

end NUMINAMATH_CALUDE_salt_trade_initial_investment_l4008_400826


namespace NUMINAMATH_CALUDE_quadratic_coefficients_from_absolute_value_l4008_400852

theorem quadratic_coefficients_from_absolute_value (x : ℝ) :
  (|x - 3| = 4 ↔ x = 7 ∨ x = -1) →
  ∃ d e : ℝ, (∀ x : ℝ, x^2 + d*x + e = 0 ↔ x = 7 ∨ x = -1) ∧ d = -6 ∧ e = -7 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_coefficients_from_absolute_value_l4008_400852


namespace NUMINAMATH_CALUDE_probability_of_double_domino_l4008_400875

/-- Represents a domino tile with two integers -/
structure Domino :=
  (a b : ℕ)

/-- The set of all possible domino tiles -/
def dominoSet : Set Domino :=
  {d : Domino | d.a ≤ 12 ∧ d.b ≤ 12}

/-- A domino is considered a double if both numbers are the same -/
def isDouble (d : Domino) : Prop :=
  d.a = d.b

/-- The number of unique domino tiles in the complete set -/
def totalDominos : ℕ :=
  (13 * 14) / 2

/-- The number of double dominos in the complete set -/
def doubleDominos : ℕ := 13

theorem probability_of_double_domino :
  (doubleDominos : ℚ) / totalDominos = 13 / 91 :=
sorry

end NUMINAMATH_CALUDE_probability_of_double_domino_l4008_400875


namespace NUMINAMATH_CALUDE_park_area_l4008_400821

/-- The area of a rectangular park with a given length-to-breadth ratio and perimeter -/
theorem park_area (length breadth perimeter : ℝ) : 
  length > 0 →
  breadth > 0 →
  length / breadth = 1 / 3 →
  perimeter = 2 * (length + breadth) →
  length * breadth = 30000 := by
  sorry

#check park_area

end NUMINAMATH_CALUDE_park_area_l4008_400821


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l4008_400896

/-- The equation of the circle is x^2 + y^2 - 2x + 6y + 6 = 0 -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 6*y + 6 = 0

/-- The center of the circle -/
def center : ℝ × ℝ := (1, -3)

/-- The radius of the circle -/
def radius : ℝ := 2

theorem circle_center_and_radius :
  ∀ x y : ℝ, circle_equation x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2 :=
sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_l4008_400896


namespace NUMINAMATH_CALUDE_power_division_l4008_400843

theorem power_division (n : ℕ) : n = 3^4053 → n / 3^2 = 3^4051 := by
  sorry

end NUMINAMATH_CALUDE_power_division_l4008_400843


namespace NUMINAMATH_CALUDE_inequality_solution_subset_l4008_400817

theorem inequality_solution_subset (a : ℝ) :
  (∀ x : ℝ, x^2 < |x - 1| + a → -3 < x ∧ x < 3) →
  a ≤ 5 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_subset_l4008_400817


namespace NUMINAMATH_CALUDE_aisha_age_l4008_400888

/-- Given the ages of Ali, Yusaf, and Umar, prove Aisha's age --/
theorem aisha_age (ali_age : ℕ) (yusaf_age : ℕ) (umar_age : ℕ) 
  (h1 : ali_age = 8)
  (h2 : ali_age = yusaf_age + 3)
  (h3 : umar_age = 2 * yusaf_age)
  (h4 : ∃ (aisha_age : ℕ), aisha_age = (ali_age + umar_age) / 2) :
  ∃ (aisha_age : ℕ), aisha_age = 9 := by
  sorry

end NUMINAMATH_CALUDE_aisha_age_l4008_400888


namespace NUMINAMATH_CALUDE_ken_kept_pencils_l4008_400845

def pencil_distribution (total : ℕ) (manny : ℕ) : Prop :=
  let nilo := 2 * manny
  let carlos := nilo / 2
  let tina := carlos + 10
  let rina := tina - 20
  let given_away := manny + nilo + carlos + tina + rina
  total - given_away = 100

theorem ken_kept_pencils :
  pencil_distribution 250 25 := by sorry

end NUMINAMATH_CALUDE_ken_kept_pencils_l4008_400845


namespace NUMINAMATH_CALUDE_fraction_simplification_l4008_400879

theorem fraction_simplification :
  (5 : ℝ) / (Real.sqrt 200 + 3 * Real.sqrt 50 + 5) = (5 * Real.sqrt 2 - 1) / 49 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l4008_400879


namespace NUMINAMATH_CALUDE_steak_cost_solution_l4008_400819

/-- The cost of a steak given the conditions of the problem -/
def steak_cost : ℝ → Prop := λ s =>
  let drink_cost : ℝ := 5
  let tip_paid : ℝ := 8
  let tip_percentage : ℝ := 0.2
  let tip_coverage : ℝ := 0.8
  let total_meal_cost : ℝ := 2 * s + 2 * drink_cost
  tip_paid = tip_coverage * tip_percentage * total_meal_cost ∧ s = 20

theorem steak_cost_solution :
  ∃ s : ℝ, steak_cost s :=
sorry

end NUMINAMATH_CALUDE_steak_cost_solution_l4008_400819


namespace NUMINAMATH_CALUDE_tangent_line_to_circle_l4008_400828

/-- The line x + 2y - 6 = 0 is tangent to the circle (x-1)^2 + y^2 = 5 at the point (2, 2) -/
theorem tangent_line_to_circle : 
  let circle : ℝ × ℝ → Prop := λ (x, y) ↦ (x - 1)^2 + y^2 = 5
  let line : ℝ × ℝ → Prop := λ (x, y) ↦ x + 2*y - 6 = 0
  let P : ℝ × ℝ := (2, 2)
  (circle P) ∧ (line P) ∧ 
  (∀ Q : ℝ × ℝ, Q ≠ P → (circle Q ∧ line Q → False)) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_to_circle_l4008_400828


namespace NUMINAMATH_CALUDE_intersection_M_N_l4008_400807

def M : Set ℝ := {x : ℝ | |x + 1| ≤ 1}
def N : Set ℝ := {-1, 0, 1}

theorem intersection_M_N : M ∩ N = {-1, 0} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l4008_400807


namespace NUMINAMATH_CALUDE_paper_length_calculation_l4008_400841

/-- The length of a rectangular sheet of paper satisfying specific area conditions -/
theorem paper_length_calculation (L : ℝ) : 
  (2 * 11 * L = 2 * 9.5 * 11 + 100) → L = 14 := by
  sorry

end NUMINAMATH_CALUDE_paper_length_calculation_l4008_400841


namespace NUMINAMATH_CALUDE_joan_has_sixteen_seashells_l4008_400803

/-- The number of seashells Joan has after giving some to Mike -/
def joans_remaining_seashells (initial : ℕ) (given : ℕ) : ℕ :=
  initial - given

/-- Theorem: Joan has 16 seashells after giving Mike 63 of her initial 79 seashells -/
theorem joan_has_sixteen_seashells :
  joans_remaining_seashells 79 63 = 16 := by
  sorry

end NUMINAMATH_CALUDE_joan_has_sixteen_seashells_l4008_400803


namespace NUMINAMATH_CALUDE_distance_maximized_at_neg_one_l4008_400802

/-- The point P -/
def P : ℝ × ℝ := (3, 2)

/-- The point Q -/
def Q : ℝ × ℝ := (2, 1)

/-- The line equation: mx - y + 1 - 2m = 0 -/
def line_equation (m : ℝ) (x y : ℝ) : Prop :=
  m * x - y + 1 - 2 * m = 0

/-- The line passes through point Q for all m -/
axiom line_through_Q (m : ℝ) : line_equation m Q.1 Q.2

/-- Distance from a point to a line -/
noncomputable def distance_to_line (p : ℝ × ℝ) (m : ℝ) : ℝ :=
  sorry

/-- The theorem to be proved -/
theorem distance_maximized_at_neg_one :
  ∃ (max_dist : ℝ), ∀ (m : ℝ),
    distance_to_line P m ≤ max_dist ∧
    distance_to_line P (-1) = max_dist :=
  sorry

end NUMINAMATH_CALUDE_distance_maximized_at_neg_one_l4008_400802


namespace NUMINAMATH_CALUDE_triangle_conditions_l4008_400884

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define what it means for a triangle to be right-angled
def isRightTriangle (t : Triangle) : Prop :=
  t.a^2 = t.b^2 + t.c^2 ∨ t.b^2 = t.a^2 + t.c^2 ∨ t.c^2 = t.a^2 + t.b^2

-- Define the conditions
def conditionA (t : Triangle) : Prop := t.A = t.B - t.C
def conditionB (t : Triangle) : Prop := t.A = t.B ∧ t.C = 2 * t.A
def conditionC (t : Triangle) : Prop := t.b^2 = t.a^2 - t.c^2
def conditionD (t : Triangle) : Prop := ∃ (k : ℝ), t.a = 2*k ∧ t.b = 3*k ∧ t.c = 4*k

-- The theorem to prove
theorem triangle_conditions (t : Triangle) :
  (conditionA t → isRightTriangle t) ∧
  (conditionB t → isRightTriangle t) ∧
  (conditionC t → isRightTriangle t) ∧
  (conditionD t → ¬isRightTriangle t) := by
  sorry

end NUMINAMATH_CALUDE_triangle_conditions_l4008_400884


namespace NUMINAMATH_CALUDE_product_of_square_roots_l4008_400815

theorem product_of_square_roots (x y z : ℝ) (hx : x = 75) (hy : y = 48) (hz : z = 3) :
  Real.sqrt x * Real.sqrt y * Real.sqrt z = 60 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_product_of_square_roots_l4008_400815


namespace NUMINAMATH_CALUDE_selection_probabilities_l4008_400892

/-- The number of boys in the group -/
def num_boys : ℕ := 3

/-- The number of girls in the group -/
def num_girls : ℕ := 2

/-- The number of people to be selected -/
def num_selected : ℕ := 3

/-- The total number of ways to select 3 people from 5 people -/
def total_combinations : ℕ := Nat.choose (num_boys + num_girls) num_selected

/-- The probability of selecting all boys -/
def prob_all_boys : ℚ := (Nat.choose num_boys num_selected : ℚ) / total_combinations

/-- The probability of selecting exactly one girl -/
def prob_one_girl : ℚ := (Nat.choose num_boys (num_selected - 1) * Nat.choose num_girls 1 : ℚ) / total_combinations

/-- The probability of selecting at least one girl -/
def prob_at_least_one_girl : ℚ := 1 - prob_all_boys

theorem selection_probabilities :
  prob_all_boys = 1/10 ∧
  prob_one_girl = 6/10 ∧
  prob_at_least_one_girl = 9/10 := by
  sorry

end NUMINAMATH_CALUDE_selection_probabilities_l4008_400892


namespace NUMINAMATH_CALUDE_car_distance_covered_l4008_400838

/-- Prove that a car traveling at 195 km/h for 3 1/5 hours covers a distance of 624 km. -/
theorem car_distance_covered (speed : ℝ) (time : ℝ) (distance : ℝ) : 
  speed = 195 → time = 3 + 1 / 5 → distance = speed * time → distance = 624 := by
sorry

end NUMINAMATH_CALUDE_car_distance_covered_l4008_400838


namespace NUMINAMATH_CALUDE_square_difference_identity_l4008_400898

theorem square_difference_identity : (25 + 15)^2 - (25 - 15)^2 = 1500 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_identity_l4008_400898


namespace NUMINAMATH_CALUDE_twelve_star_x_multiple_of_144_l4008_400890

def star (a b : ℤ) : ℤ := a^2 * b

theorem twelve_star_x_multiple_of_144 (x : ℤ) : ∃ k : ℤ, star 12 x = 144 * k := by
  sorry

end NUMINAMATH_CALUDE_twelve_star_x_multiple_of_144_l4008_400890


namespace NUMINAMATH_CALUDE_morgan_hula_hoop_time_l4008_400851

/-- Given information about hula hooping times for Nancy, Casey, and Morgan,
    prove that Morgan can hula hoop for 21 minutes. -/
theorem morgan_hula_hoop_time :
  ∀ (nancy casey morgan : ℕ),
    nancy = 10 →
    casey = nancy - 3 →
    morgan = 3 * casey →
    morgan = 21 := by
  sorry

end NUMINAMATH_CALUDE_morgan_hula_hoop_time_l4008_400851


namespace NUMINAMATH_CALUDE_square_diff_product_plus_square_l4008_400804

theorem square_diff_product_plus_square (a b : ℝ) 
  (h1 : a - b = 5) 
  (h2 : a * b = 2) : 
  a^2 - a*b + b^2 = 27 := by
  sorry

end NUMINAMATH_CALUDE_square_diff_product_plus_square_l4008_400804


namespace NUMINAMATH_CALUDE_sum_of_y_values_l4008_400808

theorem sum_of_y_values (x y : ℝ) : 
  x^2 + x^2*y^2 + x^2*y^4 = 525 ∧ x + x*y + x*y^2 = 35 →
  ∃ (y1 y2 : ℝ), y = y1 ∨ y = y2 ∧ y1 + y2 = 5/2 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_y_values_l4008_400808


namespace NUMINAMATH_CALUDE_cryptarithmetic_solution_l4008_400867

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- The cryptarithmetic equation ABAC + BAC = KCKDC -/
def CryptarithmeticEquation (A B C D K : Digit) : Prop :=
  1000 * A.val + 100 * B.val + 10 * A.val + C.val +
  100 * B.val + 10 * A.val + C.val =
  10000 * K.val + 1000 * C.val + 100 * K.val + 10 * D.val + C.val

/-- All digits are different -/
def AllDifferent (A B C D K : Digit) : Prop :=
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ K ∧
  B ≠ C ∧ B ≠ D ∧ B ≠ K ∧
  C ≠ D ∧ C ≠ K ∧
  D ≠ K

theorem cryptarithmetic_solution :
  ∃! (A B C D K : Digit),
    CryptarithmeticEquation A B C D K ∧
    AllDifferent A B C D K ∧
    A.val = 9 ∧ B.val = 5 ∧ C.val = 0 ∧ D.val = 8 ∧ K.val = 1 :=
sorry

end NUMINAMATH_CALUDE_cryptarithmetic_solution_l4008_400867


namespace NUMINAMATH_CALUDE_floor_sqrt_150_l4008_400814

theorem floor_sqrt_150 : ⌊Real.sqrt 150⌋ = 12 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_150_l4008_400814


namespace NUMINAMATH_CALUDE_exists_sequence_with_finite_primes_l4008_400848

theorem exists_sequence_with_finite_primes :
  ∃ (a : ℕ → ℕ), 
    (∀ n m : ℕ, n < m → a n < a m) ∧ 
    (∀ k : ℕ, k ≥ 2 → ∃ N : ℕ, ∀ n ≥ N, ¬ Prime (k + a n)) :=
by sorry

end NUMINAMATH_CALUDE_exists_sequence_with_finite_primes_l4008_400848


namespace NUMINAMATH_CALUDE_exponential_inequality_l4008_400827

theorem exponential_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : Real.exp a + 2 * a = Real.exp b + 3 * b) : a > b := by
  sorry

end NUMINAMATH_CALUDE_exponential_inequality_l4008_400827


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l4008_400800

/-- A sequence satisfying the given conditions -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∀ n ≥ 2, 2 * a n = a (n - 1) + a (n + 1)) ∧
  (a 1 + a 3 + a 5 = 9) ∧
  (a 3 + a 5 + a 7 = 15)

/-- The main theorem -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h : ArithmeticSequence a) :
  a 3 + a 4 + a 5 = 12 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l4008_400800


namespace NUMINAMATH_CALUDE_a_less_than_reciprocal_relationship_l4008_400844

theorem a_less_than_reciprocal_relationship (a : ℝ) :
  (a < -1 → a < 1/a) ∧ ¬(a < 1/a → a < -1) :=
by sorry

end NUMINAMATH_CALUDE_a_less_than_reciprocal_relationship_l4008_400844


namespace NUMINAMATH_CALUDE_profit_percentage_before_decrease_l4008_400891

/-- Proves that the profit percentage before the decrease in manufacturing cost was 20% --/
theorem profit_percentage_before_decrease
  (selling_price : ℝ)
  (manufacturing_cost_before : ℝ)
  (manufacturing_cost_after : ℝ)
  (h1 : manufacturing_cost_before = 80)
  (h2 : manufacturing_cost_after = 50)
  (h3 : selling_price - manufacturing_cost_after = 0.5 * selling_price) :
  (selling_price - manufacturing_cost_before) / selling_price = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_before_decrease_l4008_400891


namespace NUMINAMATH_CALUDE_knights_count_l4008_400878

/-- Represents an islander, who can be either a knight or a liar -/
inductive Islander
| Knight
| Liar

/-- The total number of islanders -/
def total_islanders : Nat := 6

/-- Determines if an islander's statement is true based on the actual number of liars -/
def statement_is_true (actual_liars : Nat) : Prop :=
  actual_liars = 4

/-- Determines if an islander's behavior is consistent with their type and statement -/
def is_consistent (islander : Islander) (actual_liars : Nat) : Prop :=
  match islander with
  | Islander.Knight => statement_is_true actual_liars
  | Islander.Liar => ¬statement_is_true actual_liars

/-- The main theorem to prove -/
theorem knights_count :
  ∀ (knights : Nat),
    (knights ≤ total_islanders) →
    (∀ i : Fin total_islanders,
      is_consistent
        (if i.val < knights then Islander.Knight else Islander.Liar)
        (total_islanders - knights - 1)) →
    (knights = 0 ∨ knights = 2) :=
by sorry


end NUMINAMATH_CALUDE_knights_count_l4008_400878


namespace NUMINAMATH_CALUDE_prism_volume_problem_l4008_400885

/-- 
Given a rectangular prism with dimensions 15 cm × 5 cm × 4 cm and a smaller prism
with dimensions y cm × 5 cm × x cm removed, if the remaining volume is 120 cm³,
then x + y = 15, where x and y are integers.
-/
theorem prism_volume_problem (x y : ℤ) : 
  (15 * 5 * 4 - y * 5 * x = 120) → (x + y = 15) := by sorry

end NUMINAMATH_CALUDE_prism_volume_problem_l4008_400885


namespace NUMINAMATH_CALUDE_article_cost_changes_l4008_400834

theorem article_cost_changes (initial_cost : ℝ) : 
  initial_cost = 75 →
  (initial_cost * (1 + 0.2) * (1 - 0.2) * (1 + 0.3) * (1 - 0.25)) = 70.2 := by
  sorry

end NUMINAMATH_CALUDE_article_cost_changes_l4008_400834


namespace NUMINAMATH_CALUDE_concert_tickets_sold_l4008_400876

theorem concert_tickets_sold (T : ℕ) : 
  (3 / 4 : ℚ) * T + (5 / 9 : ℚ) * (1 / 4 : ℚ) * T + 80 + 20 = T → T = 900 := by
  sorry

end NUMINAMATH_CALUDE_concert_tickets_sold_l4008_400876


namespace NUMINAMATH_CALUDE_total_herd_count_l4008_400835

/-- Represents the number of animals in a shepherd's herd -/
structure Herd where
  count : ℕ

/-- Represents a shepherd with their herd -/
structure Shepherd where
  name : String
  herd : Herd

/-- The conditions of the problem -/
def exchange_conditions (jack jim dan : Shepherd) : Prop :=
  (jim.herd.count + 6 = 2 * (jack.herd.count - 1)) ∧
  (jack.herd.count + 14 = 3 * (dan.herd.count - 1)) ∧
  (dan.herd.count + 4 = 6 * (jim.herd.count - 1))

/-- The theorem to be proved -/
theorem total_herd_count (jack jim dan : Shepherd) :
  exchange_conditions jack jim dan →
  jack.herd.count + jim.herd.count + dan.herd.count = 39 := by
  sorry


end NUMINAMATH_CALUDE_total_herd_count_l4008_400835


namespace NUMINAMATH_CALUDE_range_of_a_l4008_400868

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Ioo 0 2, x^2 - 2*a*x + 2 ≥ 0) → 
  a ∈ Set.Iic (Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l4008_400868


namespace NUMINAMATH_CALUDE_ladybugs_with_spots_count_l4008_400836

/-- The total number of ladybugs -/
def total_ladybugs : ℕ := 67082

/-- The number of ladybugs without spots -/
def ladybugs_without_spots : ℕ := 54912

/-- The number of ladybugs with spots -/
def ladybugs_with_spots : ℕ := total_ladybugs - ladybugs_without_spots

theorem ladybugs_with_spots_count : ladybugs_with_spots = 12170 := by
  sorry

end NUMINAMATH_CALUDE_ladybugs_with_spots_count_l4008_400836


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l4008_400869

def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

theorem arithmetic_sequence_problem (a₁ d : ℝ) :
  arithmetic_sequence a₁ d 5 = 0.3 →
  arithmetic_sequence a₁ d 12 = 3.1 →
  a₁ = -1.3 ∧ d = 0.4 ∧
  (arithmetic_sequence a₁ d 18 +
   arithmetic_sequence a₁ d 19 +
   arithmetic_sequence a₁ d 20 +
   arithmetic_sequence a₁ d 21 +
   arithmetic_sequence a₁ d 22) = 31.5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l4008_400869


namespace NUMINAMATH_CALUDE_triangle_side_length_l4008_400831

theorem triangle_side_length (a b c : ℝ) (B : ℝ) :
  a = 3 →
  b = Real.sqrt 6 →
  B = π / 4 →
  c = (3 * Real.sqrt 2 + Real.sqrt 6) / 2 ∨ c = (3 * Real.sqrt 2 - Real.sqrt 6) / 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l4008_400831


namespace NUMINAMATH_CALUDE_base6_210_equals_base4_1032_l4008_400824

-- Define a function to convert a base 6 number to base 10
def base6ToBase10 (n : ℕ) : ℕ :=
  (n / 100) * 36 + ((n / 10) % 10) * 6 + (n % 10)

-- Define a function to convert a base 10 number to base 4
def base10ToBase4 (n : ℕ) : ℕ :=
  (n / 64) * 1000 + ((n / 16) % 4) * 100 + ((n / 4) % 4) * 10 + (n % 4)

-- Theorem statement
theorem base6_210_equals_base4_1032 :
  base10ToBase4 (base6ToBase10 210) = 1032 :=
sorry

end NUMINAMATH_CALUDE_base6_210_equals_base4_1032_l4008_400824


namespace NUMINAMATH_CALUDE_probability_all_sweet_is_one_sixth_l4008_400812

def total_oranges : ℕ := 10
def sweet_oranges : ℕ := 6
def picked_oranges : ℕ := 3

def probability_all_sweet : ℚ :=
  (sweet_oranges.choose picked_oranges) / (total_oranges.choose picked_oranges)

theorem probability_all_sweet_is_one_sixth :
  probability_all_sweet = 1 / 6 := by sorry

end NUMINAMATH_CALUDE_probability_all_sweet_is_one_sixth_l4008_400812


namespace NUMINAMATH_CALUDE_nilpotent_matrix_square_zero_l4008_400857

theorem nilpotent_matrix_square_zero 
  (A : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : A ^ 4 = 0) : 
  A ^ 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_nilpotent_matrix_square_zero_l4008_400857


namespace NUMINAMATH_CALUDE_spring_fills_sixty_barrels_per_day_l4008_400880

/-- A spring fills barrels of water -/
structure Spring where
  fill_time : ℕ  -- Time to fill one barrel in minutes

/-- A day has a certain number of hours and minutes per hour -/
structure Day where
  hours : ℕ
  minutes_per_hour : ℕ

def barrels_filled_per_day (s : Spring) (d : Day) : ℕ :=
  (d.hours * d.minutes_per_hour) / s.fill_time

/-- Theorem: A spring that fills a barrel in 24 minutes will fill 60 barrels in a day -/
theorem spring_fills_sixty_barrels_per_day (s : Spring) (d : Day) :
  s.fill_time = 24 → d.hours = 24 → d.minutes_per_hour = 60 →
  barrels_filled_per_day s d = 60 := by
  sorry

end NUMINAMATH_CALUDE_spring_fills_sixty_barrels_per_day_l4008_400880


namespace NUMINAMATH_CALUDE_square_root_of_x_plus_y_l4008_400866

theorem square_root_of_x_plus_y (x y : ℝ) 
  (h1 : 2*x + 7*y + 1 = 6^2) 
  (h2 : 8*x + 3*y = 5^3) : 
  (x + y).sqrt = 4 ∨ (x + y).sqrt = -4 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_x_plus_y_l4008_400866


namespace NUMINAMATH_CALUDE_sqrt_five_squared_times_seven_sixth_power_l4008_400825

theorem sqrt_five_squared_times_seven_sixth_power : 
  Real.sqrt (5^2 * 7^6) = 1715 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_five_squared_times_seven_sixth_power_l4008_400825


namespace NUMINAMATH_CALUDE_y_minus_3x_equals_7_l4008_400833

theorem y_minus_3x_equals_7 (x y : ℝ) (h1 : x + y = 8) (h2 : y - x = 7.5) : y - 3 * x = 7 := by
  sorry

end NUMINAMATH_CALUDE_y_minus_3x_equals_7_l4008_400833


namespace NUMINAMATH_CALUDE_if_statement_properties_l4008_400886

-- Define the structure of an IF statement
structure IfStatement where
  has_else : Bool
  has_end_if : Bool

-- Define what makes an IF statement valid
def is_valid_if_statement (stmt : IfStatement) : Prop :=
  stmt.has_end_if ∧ (stmt.has_else ∨ ¬stmt.has_else)

-- Theorem statement
theorem if_statement_properties :
  ∀ (stmt : IfStatement),
    is_valid_if_statement stmt →
    (stmt.has_else ∨ ¬stmt.has_else) ∧ stmt.has_end_if :=
by sorry

end NUMINAMATH_CALUDE_if_statement_properties_l4008_400886


namespace NUMINAMATH_CALUDE_f_maximum_l4008_400820

/-- The quadratic function f(x) = -3x^2 + 9x + 5 -/
def f (x : ℝ) : ℝ := -3 * x^2 + 9 * x + 5

/-- The value of x that maximizes f(x) -/
def x_max : ℝ := 1.5

theorem f_maximum :
  ∀ x : ℝ, f x ≤ f x_max :=
sorry

end NUMINAMATH_CALUDE_f_maximum_l4008_400820


namespace NUMINAMATH_CALUDE_equilateral_triangle_count_l4008_400859

/-- Represents a line in the coordinate plane --/
structure Line where
  equation : ℝ → ℝ → Prop

/-- Generates horizontal lines y = k for k ∈ [-15, 15] --/
def horizontal_lines : List Line :=
  sorry

/-- Generates sloped lines y = √2x + 3k and y = -√2x + 3k for k ∈ [-15, 15] --/
def sloped_lines : List Line :=
  sorry

/-- All lines in the problem --/
def all_lines : List Line :=
  horizontal_lines ++ sloped_lines

/-- Predicate for an equilateral triangle with side length √2 --/
def is_unit_triangle (p q r : ℝ × ℝ) : Prop :=
  sorry

/-- Count of equilateral triangles formed by the intersection of lines --/
def triangle_count : ℕ :=
  sorry

/-- Main theorem stating the number of equilateral triangles formed --/
theorem equilateral_triangle_count :
  triangle_count = 12336 :=
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_count_l4008_400859


namespace NUMINAMATH_CALUDE_pure_imaginary_magnitude_l4008_400811

theorem pure_imaginary_magnitude (a : ℝ) : 
  (((a - 2 * Complex.I) / (1 + Complex.I)).re = 0) → 
  Complex.abs (1 + a * Complex.I) = Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_pure_imaginary_magnitude_l4008_400811


namespace NUMINAMATH_CALUDE_bulls_win_probability_l4008_400862

-- Define the probability of Heat winning a single game
def heat_win_prob : ℚ := 3/4

-- Define the probability of Bulls winning a single game
def bulls_win_prob : ℚ := 1 - heat_win_prob

-- Define the number of games needed to win the series
def games_to_win : ℕ := 4

-- Define the total number of games in a full series
def total_games : ℕ := 7

-- Define the function to calculate the probability of Bulls winning in 7 games
def bulls_win_in_seven : ℚ :=
  -- Probability of 3-3 tie after 6 games
  (Nat.choose 6 3 : ℚ) * bulls_win_prob^3 * heat_win_prob^3 *
  -- Probability of Bulls winning the 7th game
  bulls_win_prob

-- Theorem statement
theorem bulls_win_probability :
  bulls_win_in_seven = 540 / 16384 := by sorry

end NUMINAMATH_CALUDE_bulls_win_probability_l4008_400862


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l4008_400840

theorem imaginary_part_of_z (z : ℂ) (h : (1 + 2*I)/z = I) : z.im = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l4008_400840


namespace NUMINAMATH_CALUDE_ellipse_line_and_fixed_circle_l4008_400855

/-- Given an ellipse C and points P, Q, and conditions for line l, prove the equation of l and that point S lies on a fixed circle. -/
theorem ellipse_line_and_fixed_circle 
  (x₀ y₀ : ℝ) 
  (hy₀ : y₀ ≠ 0)
  (hP : x₀^2/4 + y₀^2/3 = 1) 
  (Q : ℝ × ℝ)
  (hQ : Q = (x₀/4, y₀/3))
  (l : Set (ℝ × ℝ))
  (hl : ∀ M ∈ l, (M.1 - x₀) * (x₀/4) + (M.2 - y₀) * (y₀/3) = 0)
  (F : ℝ × ℝ)
  (hF : F.1 > 0 ∧ F.1^2 = 1 + F.2^2/3)  -- Condition for right focus
  (S : ℝ × ℝ)
  (hS : ∃ k, S = (4 + k * (4*y₀)/(3*x₀), k) ∧ 
             S.2 = (y₀/(x₀-1)) * (S.1 - 1)) :
  (∀ x y, (x, y) ∈ l ↔ x₀*x/4 + y₀*y/3 = 1) ∧ 
  ((S.1 - 1)^2 + S.2^2 = 36) := by
  sorry


end NUMINAMATH_CALUDE_ellipse_line_and_fixed_circle_l4008_400855


namespace NUMINAMATH_CALUDE_sugar_profit_percentage_l4008_400847

/-- Proves that given 1000 kg of sugar, with 400 kg sold at 8% profit and 600 kg sold at x% profit,
    if the overall profit is 14%, then x = 18. -/
theorem sugar_profit_percentage 
  (total_sugar : ℝ) 
  (sugar_at_8_percent : ℝ) 
  (sugar_at_x_percent : ℝ) 
  (x : ℝ) :
  total_sugar = 1000 →
  sugar_at_8_percent = 400 →
  sugar_at_x_percent = 600 →
  sugar_at_8_percent * 0.08 + sugar_at_x_percent * (x / 100) = total_sugar * 0.14 →
  x = 18 := by
  sorry

end NUMINAMATH_CALUDE_sugar_profit_percentage_l4008_400847


namespace NUMINAMATH_CALUDE_missing_number_proof_l4008_400895

theorem missing_number_proof (x : ℝ) : 
  let numbers := [1, 22, 23, 24, 25, 26, x, 2]
  (List.sum numbers) / (List.length numbers) = 20 → x = 37 := by
sorry

end NUMINAMATH_CALUDE_missing_number_proof_l4008_400895


namespace NUMINAMATH_CALUDE_intersection_A_B_union_B_complement_A_l4008_400882

-- Define the universal set U
def U : Set ℝ := {x | -5 ≤ x ∧ x ≤ 5}

-- Define set A
def A : Set ℝ := {x ∈ U | 0 < x ∧ x ≤ 3}

-- Define set B
def B : Set ℝ := {x ∈ U | -2 ≤ x ∧ x ≤ 1}

-- Theorem for A ∩ B
theorem intersection_A_B : A ∩ B = {x ∈ U | 0 < x ∧ x ≤ 1} := by sorry

-- Theorem for B ∪ (ᶜA)
theorem union_B_complement_A : B ∪ (U \ A) = {x ∈ U | x ≤ 1 ∨ 3 < x} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_union_B_complement_A_l4008_400882


namespace NUMINAMATH_CALUDE_graph_is_pair_of_straight_lines_l4008_400842

/-- The equation of the graph -/
def equation (x y : ℝ) : Prop := x^2 - 9*y^2 = 0

/-- Definition of a straight line -/
def is_straight_line (f : ℝ → ℝ) : Prop :=
  ∃ m b : ℝ, ∀ x : ℝ, f x = m * x + b

/-- Theorem stating that the graph of x^2 - 9y^2 = 0 is a pair of straight lines -/
theorem graph_is_pair_of_straight_lines :
  ∃ f g : ℝ → ℝ, 
    (is_straight_line f ∧ is_straight_line g) ∧
    (∀ x y : ℝ, equation x y ↔ (y = f x ∨ y = g x)) :=
sorry

end NUMINAMATH_CALUDE_graph_is_pair_of_straight_lines_l4008_400842


namespace NUMINAMATH_CALUDE_expression_value_l4008_400823

theorem expression_value : (100 - (3010 - 301)) + (3010 - (301 - 100)) = 200 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l4008_400823


namespace NUMINAMATH_CALUDE_garden_perimeter_l4008_400871

/-- A rectangular garden with given diagonal and area has a specific perimeter -/
theorem garden_perimeter (x y : ℝ) (h_rectangle : x > 0 ∧ y > 0) 
  (h_diagonal : x^2 + y^2 = 34^2) (h_area : x * y = 240) : 
  2 * (x + y) = 80 := by
  sorry

end NUMINAMATH_CALUDE_garden_perimeter_l4008_400871


namespace NUMINAMATH_CALUDE_binary_111_equals_7_l4008_400877

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Nat) : Nat :=
  bits.enum.foldl (fun acc (i, b) => acc + b * 2^i) 0

/-- The binary representation of the number we want to convert -/
def binary_111 : List Nat := [1, 1, 1]

/-- Theorem stating that the binary number 111 is equal to the decimal number 7 -/
theorem binary_111_equals_7 : binary_to_decimal binary_111 = 7 := by
  sorry

end NUMINAMATH_CALUDE_binary_111_equals_7_l4008_400877


namespace NUMINAMATH_CALUDE_upper_bound_y_l4008_400899

theorem upper_bound_y (x y : ℤ) (h1 : 3 < x) (h2 : x < 6) (h3 : 6 < y) 
  (h4 : ∀ (a b : ℤ), 3 < a → a < 6 → 6 < b → b - a ≤ 6) : y ≤ 11 :=
sorry

end NUMINAMATH_CALUDE_upper_bound_y_l4008_400899


namespace NUMINAMATH_CALUDE_thirty_percent_of_number_l4008_400894

theorem thirty_percent_of_number (x : ℝ) (h : (3 / 7) * x = 0.4 * x + 12) : 0.3 * x = 126 := by
  sorry

end NUMINAMATH_CALUDE_thirty_percent_of_number_l4008_400894


namespace NUMINAMATH_CALUDE_joans_sandwiches_l4008_400849

/-- Given the conditions for Joan's sandwich making, prove the number of grilled cheese sandwiches. -/
theorem joans_sandwiches (total_cheese : ℕ) (ham_sandwiches : ℕ) (cheese_per_ham : ℕ) (cheese_per_grilled : ℕ)
  (h_total : total_cheese = 50)
  (h_ham : ham_sandwiches = 10)
  (h_cheese_ham : cheese_per_ham = 2)
  (h_cheese_grilled : cheese_per_grilled = 3) :
  (total_cheese - ham_sandwiches * cheese_per_ham) / cheese_per_grilled = 10 := by
  sorry

#eval (50 - 10 * 2) / 3  -- Expected output: 10

end NUMINAMATH_CALUDE_joans_sandwiches_l4008_400849


namespace NUMINAMATH_CALUDE_line_through_points_l4008_400850

/-- Given a line passing through points (-3, 1) and (1, 5) with equation y = mx + b, prove that m + b = 5 -/
theorem line_through_points (m b : ℝ) : 
  (1 = m * (-3) + b) → (5 = m * 1 + b) → m + b = 5 := by
  sorry

end NUMINAMATH_CALUDE_line_through_points_l4008_400850


namespace NUMINAMATH_CALUDE_min_value_sum_squares_l4008_400863

theorem min_value_sum_squares (a b c d e f g h : ℝ) 
  (h1 : a * b * c * d = 4) 
  (h2 : e * f * g * h = 9) : 
  (a * e)^2 + (b * f)^2 + (c * g)^2 + (d * h)^2 ≥ 24 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_squares_l4008_400863


namespace NUMINAMATH_CALUDE_inequality_solution_sum_l4008_400837

/-- Given an inequality ax^2 - 3x + 2 > 0 with solution set {x | x < 1 or x > b}, prove a + b = 3 -/
theorem inequality_solution_sum (a b : ℝ) : 
  (∀ x, ax^2 - 3*x + 2 > 0 ↔ (x < 1 ∨ x > b)) → a + b = 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_sum_l4008_400837


namespace NUMINAMATH_CALUDE_total_lunch_cost_l4008_400864

/-- Calculates the total cost of lunch for all students in an elementary school --/
theorem total_lunch_cost (third_grade_classes fourth_grade_classes fifth_grade_classes : ℕ)
  (third_grade_students fourth_grade_students fifth_grade_students : ℕ)
  (hamburger_cost carrot_cost cookie_cost : ℚ) : ℚ :=
  by
  have h1 : third_grade_classes = 5 := by sorry
  have h2 : fourth_grade_classes = 4 := by sorry
  have h3 : fifth_grade_classes = 4 := by sorry
  have h4 : third_grade_students = 30 := by sorry
  have h5 : fourth_grade_students = 28 := by sorry
  have h6 : fifth_grade_students = 27 := by sorry
  have h7 : hamburger_cost = 2.1 := by sorry
  have h8 : carrot_cost = 0.5 := by sorry
  have h9 : cookie_cost = 0.2 := by sorry

  have total_students : ℕ := 
    third_grade_classes * third_grade_students + 
    fourth_grade_classes * fourth_grade_students + 
    fifth_grade_classes * fifth_grade_students

  have lunch_cost_per_student : ℚ := hamburger_cost + carrot_cost + cookie_cost

  have total_cost : ℚ := total_students * lunch_cost_per_student

  exact 1036

end NUMINAMATH_CALUDE_total_lunch_cost_l4008_400864


namespace NUMINAMATH_CALUDE_odd_square_not_sum_of_five_odd_squares_l4008_400874

theorem odd_square_not_sum_of_five_odd_squares :
  ∀ n a b c d e : ℤ,
  Odd n → Odd a → Odd b → Odd c → Odd d → Odd e →
  ¬(n^2 ≡ a^2 + b^2 + c^2 + d^2 + e^2 [ZMOD 8]) :=
by sorry

end NUMINAMATH_CALUDE_odd_square_not_sum_of_five_odd_squares_l4008_400874


namespace NUMINAMATH_CALUDE_units_digit_of_a_l4008_400846

theorem units_digit_of_a (a : ℕ) : a = 2003^2004 - 2004^2003 → a % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_a_l4008_400846


namespace NUMINAMATH_CALUDE_xy_equals_one_l4008_400822

theorem xy_equals_one (x y : ℝ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : x + y = 25) (h4 : x^2 * y^3 + y^2 * x^3 = 25) : x * y = 1 := by
  sorry

end NUMINAMATH_CALUDE_xy_equals_one_l4008_400822


namespace NUMINAMATH_CALUDE_sphere_volume_increase_l4008_400887

theorem sphere_volume_increase (r₁ r₂ : ℝ) (h : r₂ = 2 * r₁) : 
  (4 / 3) * π * r₂^3 = 8 * ((4 / 3) * π * r₁^3) := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_increase_l4008_400887


namespace NUMINAMATH_CALUDE_tom_clothing_count_l4008_400870

/-- The total number of pieces of clothing Tom had -/
def total_clothing : ℕ := 36

/-- The number of pieces in the first load -/
def first_load : ℕ := 18

/-- The number of pieces in each of the two equal loads -/
def equal_load : ℕ := 9

/-- The number of equal loads -/
def num_equal_loads : ℕ := 2

theorem tom_clothing_count :
  total_clothing = first_load + num_equal_loads * equal_load :=
by sorry

end NUMINAMATH_CALUDE_tom_clothing_count_l4008_400870


namespace NUMINAMATH_CALUDE_largest_number_l4008_400839

/-- Represents a repeating decimal number -/
structure RepeatingDecimal where
  integerPart : ℕ
  nonRepeatingPart : List ℕ
  repeatingPart : List ℕ

/-- Convert a RepeatingDecimal to a rational number -/
def toRational (r : RepeatingDecimal) : ℚ :=
  sorry

/-- The number 5.14322 -/
def a : ℚ := 5.14322

/-- The number 5.143̅2 -/
def b : RepeatingDecimal := ⟨5, [1, 4, 3], [2]⟩

/-- The number 5.14̅32 -/
def c : RepeatingDecimal := ⟨5, [1, 4], [3, 2]⟩

/-- The number 5.1̅432 -/
def d : RepeatingDecimal := ⟨5, [1], [4, 3, 2]⟩

/-- The number 5.̅4321 -/
def e : RepeatingDecimal := ⟨5, [], [4, 3, 2, 1]⟩

theorem largest_number : 
  toRational d > a ∧ 
  toRational d > toRational b ∧ 
  toRational d > toRational c ∧ 
  toRational d > toRational e :=
sorry

end NUMINAMATH_CALUDE_largest_number_l4008_400839


namespace NUMINAMATH_CALUDE_smallest_marble_count_l4008_400801

/-- Represents the number of marbles of each color in the urn -/
structure MarbleCount where
  red : ℕ
  white : ℕ
  blue : ℕ
  green : ℕ

/-- Calculates the total number of marbles in the urn -/
def total_marbles (mc : MarbleCount) : ℕ :=
  mc.red + mc.white + mc.blue + mc.green

/-- Calculates the probability of selecting a specific combination of marbles -/
def probability (mc : MarbleCount) (red white blue green : ℕ) : ℚ :=
  (mc.red.choose red * mc.white.choose white * mc.blue.choose blue * mc.green.choose green : ℚ) /
  (total_marbles mc).choose 5

/-- Checks if all specified probabilities are equal -/
def probabilities_equal (mc : MarbleCount) : Prop :=
  probability mc 5 0 0 0 = probability mc 3 2 0 0 ∧
  probability mc 3 2 0 0 = probability mc 1 2 2 0 ∧
  probability mc 1 2 2 0 = probability mc 2 1 1 1

/-- The theorem stating that the smallest number of marbles satisfying the conditions is 24 -/
theorem smallest_marble_count : 
  ∃ (mc : MarbleCount), probabilities_equal mc ∧ total_marbles mc = 24 ∧
  (∀ (mc' : MarbleCount), probabilities_equal mc' → total_marbles mc' ≥ 24) :=
sorry

end NUMINAMATH_CALUDE_smallest_marble_count_l4008_400801


namespace NUMINAMATH_CALUDE_rhombus_count_in_triangle_l4008_400813

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  sideLength : ℝ

/-- Represents a rhombus composed of smaller triangles -/
structure Rhombus where
  smallTrianglesCount : ℕ

/-- Counts the number of rhombuses in a given equilateral triangle -/
def countRhombuses (triangle : EquilateralTriangle) (rhombusSize : ℕ) : ℕ :=
  sorry

/-- The main theorem to prove -/
theorem rhombus_count_in_triangle :
  let largeTriangle := EquilateralTriangle.mk 10
  let rhombusType := Rhombus.mk 8
  countRhombuses largeTriangle rhombusType.smallTrianglesCount = 84 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_count_in_triangle_l4008_400813


namespace NUMINAMATH_CALUDE_white_squares_42nd_row_l4008_400889

/-- Represents the number of squares in a row of the stair-step figure -/
def squares_in_row (n : ℕ) : ℕ := 2 * n - 1

/-- Represents the number of white squares in a row of the stair-step figure -/
def white_squares_in_row (n : ℕ) : ℕ := (squares_in_row n + 1) / 2

/-- Theorem stating the number of white squares in the 42nd row -/
theorem white_squares_42nd_row :
  white_squares_in_row 42 = 42 := by
  sorry

end NUMINAMATH_CALUDE_white_squares_42nd_row_l4008_400889


namespace NUMINAMATH_CALUDE_power_five_mod_hundred_l4008_400830

theorem power_five_mod_hundred : 5^2023 % 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_power_five_mod_hundred_l4008_400830


namespace NUMINAMATH_CALUDE_find_h_l4008_400805

-- Define the two quadratic functions
def f (h j x : ℝ) : ℝ := 4 * (x - h)^2 + j
def g (h k x : ℝ) : ℝ := 3 * (x - h)^2 + k

-- State the theorem
theorem find_h : 
  ∃ (h j k : ℝ),
    (f h j 0 = 2024) ∧ 
    (g h k 0 = 2025) ∧
    (∃ (x₁ x₂ y₁ y₂ : ℤ), x₁ > 0 ∧ x₂ > 0 ∧ y₁ > 0 ∧ y₂ > 0 ∧ 
      f h j (x₁ : ℝ) = 0 ∧ f h j (x₂ : ℝ) = 0 ∧
      g h k (y₁ : ℝ) = 0 ∧ g h k (y₂ : ℝ) = 0) →
    h = 22.5 := by
  sorry


end NUMINAMATH_CALUDE_find_h_l4008_400805


namespace NUMINAMATH_CALUDE_function_symmetry_periodicity_l4008_400854

theorem function_symmetry_periodicity 
  (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = f (-x)) 
  (h2 : ∀ x, f x = -f (2 - x)) : 
  ∀ x, f (x + 4) = f x := by
sorry

end NUMINAMATH_CALUDE_function_symmetry_periodicity_l4008_400854


namespace NUMINAMATH_CALUDE_friend_reading_time_l4008_400881

/-- Given a person who reads at half the speed of their friend and takes 4 hours to read a book,
    prove that their friend will take 120 minutes to read the same book. -/
theorem friend_reading_time (my_speed friend_speed : ℝ) (my_time friend_time : ℝ) :
  my_speed = (1/2) * friend_speed →
  my_time = 4 →
  friend_time = 2 →
  friend_time * 60 = 120 := by
  sorry

end NUMINAMATH_CALUDE_friend_reading_time_l4008_400881


namespace NUMINAMATH_CALUDE_sin_630_degrees_l4008_400873

theorem sin_630_degrees : Real.sin (630 * π / 180) = -1 := by sorry

end NUMINAMATH_CALUDE_sin_630_degrees_l4008_400873


namespace NUMINAMATH_CALUDE_shirt_profit_theorem_l4008_400810

/-- Represents the daily profit function for a shirt department -/
def daily_profit (initial_sales : ℕ) (initial_profit : ℝ) (price_reduction : ℝ) : ℝ :=
  (initial_profit - price_reduction) * (initial_sales + 2 * price_reduction)

theorem shirt_profit_theorem 
  (initial_sales : ℕ) 
  (initial_profit : ℝ) 
  (h_initial_sales : initial_sales = 30)
  (h_initial_profit : initial_profit = 40) :
  (∃ (x : ℝ), daily_profit initial_sales initial_profit x = 1200) ∧
  (∀ (y : ℝ), daily_profit initial_sales initial_profit y ≠ 1600) :=
sorry

#check shirt_profit_theorem

end NUMINAMATH_CALUDE_shirt_profit_theorem_l4008_400810


namespace NUMINAMATH_CALUDE_power_of_two_greater_than_n_and_factorial_greater_than_power_of_two_l4008_400861

theorem power_of_two_greater_than_n_and_factorial_greater_than_power_of_two :
  (∀ n : ℕ, 2^n > n) ∧
  (∀ n : ℕ, n ≥ 4 → n.factorial > 2^n) := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_greater_than_n_and_factorial_greater_than_power_of_two_l4008_400861
