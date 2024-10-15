import Mathlib

namespace NUMINAMATH_CALUDE_at_least_one_le_quarter_l399_39919

theorem at_least_one_le_quarter (a b c : Real) 
  (ha : 0 < a ∧ a < 1) 
  (hb : 0 < b ∧ b < 1) 
  (hc : 0 < c ∧ c < 1) : 
  (a * (1 - b) ≤ 1/4) ∨ (b * (1 - c) ≤ 1/4) ∨ (c * (1 - a) ≤ 1/4) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_le_quarter_l399_39919


namespace NUMINAMATH_CALUDE_parallelepiped_length_l399_39966

theorem parallelepiped_length (n : ℕ) : 
  n > 6 →
  (n - 2) * (n - 4) * (n - 6) = (2 / 3) * n * (n - 2) * (n - 4) →
  n = 18 := by
sorry

end NUMINAMATH_CALUDE_parallelepiped_length_l399_39966


namespace NUMINAMATH_CALUDE_quadratic_inequality_always_positive_l399_39991

theorem quadratic_inequality_always_positive (c : ℝ) :
  (∀ x : ℝ, x^2 + x + c > 0) ↔ c > 1/4 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_always_positive_l399_39991


namespace NUMINAMATH_CALUDE_three_power_plus_one_not_divisible_l399_39934

theorem three_power_plus_one_not_divisible (n : ℕ) 
  (h_odd : Odd n) (h_gt_one : n > 1) : ¬(n ∣ 3^n + 1) := by
  sorry

end NUMINAMATH_CALUDE_three_power_plus_one_not_divisible_l399_39934


namespace NUMINAMATH_CALUDE_nickels_to_dimes_ratio_l399_39988

/-- Represents the number of coins of each type in Tommy's collection -/
structure CoinCollection where
  pennies : ℕ
  nickels : ℕ
  dimes : ℕ
  quarters : ℕ

/-- Defines Tommy's coin collection based on the given conditions -/
def tommys_collection : CoinCollection where
  pennies := 40
  nickels := 100
  dimes := 50
  quarters := 4

/-- Theorem stating the ratio of nickels to dimes in Tommy's collection -/
theorem nickels_to_dimes_ratio (c : CoinCollection) 
  (h1 : c.dimes = c.pennies + 10)
  (h2 : c.quarters = 4)
  (h3 : c.pennies = 10 * c.quarters)
  (h4 : c.nickels = 100) :
  c.nickels / c.dimes = 2 := by
  sorry

#check nickels_to_dimes_ratio tommys_collection

end NUMINAMATH_CALUDE_nickels_to_dimes_ratio_l399_39988


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l399_39911

def M : Set ℕ := {0, 1, 3}

def N : Set ℕ := {x | ∃ a ∈ M, x = 3 * a}

theorem union_of_M_and_N : M ∪ N = {0, 1, 3, 9} := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l399_39911


namespace NUMINAMATH_CALUDE_inverse_of_M_l399_39942

def A : Matrix (Fin 2) (Fin 2) ℚ := !![1, 0; 0, -1]
def B : Matrix (Fin 2) (Fin 2) ℚ := !![4, 1; 2, 3]
def M : Matrix (Fin 2) (Fin 2) ℚ := B * A

theorem inverse_of_M :
  M⁻¹ = !![3/10, -1/10; 1/5, -2/5] := by sorry

end NUMINAMATH_CALUDE_inverse_of_M_l399_39942


namespace NUMINAMATH_CALUDE_gregorian_calendar_properties_l399_39900

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents the Gregorian calendar system -/
structure GregorianCalendar where
  -- Add necessary fields and methods

/-- Counts occurrences of a specific day for January 1st in a 400-year cycle -/
def countJanuary1Occurrences (day : DayOfWeek) (calendar : GregorianCalendar) : Nat :=
  sorry

/-- Counts occurrences of a specific day for the 30th of each month in a 400-year cycle -/
def count30thOccurrences (day : DayOfWeek) (calendar : GregorianCalendar) : Nat :=
  sorry

theorem gregorian_calendar_properties (calendar : GregorianCalendar) :
  (countJanuary1Occurrences DayOfWeek.Sunday calendar > countJanuary1Occurrences DayOfWeek.Saturday calendar) ∧
  (∀ d : DayOfWeek, count30thOccurrences DayOfWeek.Friday calendar ≥ count30thOccurrences d calendar) :=
by sorry

end NUMINAMATH_CALUDE_gregorian_calendar_properties_l399_39900


namespace NUMINAMATH_CALUDE_cubic_polynomial_inequality_l399_39905

/-- 
A cubic polynomial with real coefficients that has three real roots 
satisfies the inequality 6a^3 + 10(a^2 - 2b)^(3/2) - 12ab ≥ 27c, 
with equality if and only if b = 0, c = -4/27 * a^3, and a ≤ 0.
-/
theorem cubic_polynomial_inequality (a b c : ℝ) : 
  (∃ x y z : ℝ, x^3 + a*x^2 + b*x + c = 0 ∧ 
               y^3 + a*y^2 + b*y + c = 0 ∧ 
               z^3 + a*z^2 + b*z + c = 0 ∧ 
               x ≠ y ∧ y ≠ z ∧ x ≠ z) →
  6*a^3 + 10*(a^2 - 2*b)^(3/2) - 12*a*b ≥ 27*c ∧
  (6*a^3 + 10*(a^2 - 2*b)^(3/2) - 12*a*b = 27*c ↔ b = 0 ∧ c = -4/27 * a^3 ∧ a ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_cubic_polynomial_inequality_l399_39905


namespace NUMINAMATH_CALUDE_first_month_sale_l399_39929

def average_sale : ℝ := 5400
def num_months : ℕ := 6
def sale_month2 : ℝ := 5366
def sale_month3 : ℝ := 5808
def sale_month4 : ℝ := 5399
def sale_month5 : ℝ := 6124
def sale_month6 : ℝ := 4579

theorem first_month_sale :
  let total_sales := average_sale * num_months
  let known_sales := sale_month2 + sale_month3 + sale_month4 + sale_month5 + sale_month6
  total_sales - known_sales = 5124 := by
sorry

end NUMINAMATH_CALUDE_first_month_sale_l399_39929


namespace NUMINAMATH_CALUDE_min_value_sum_product_l399_39916

theorem min_value_sum_product (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = 20) :
  a + 2 * b ≥ 4 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_product_l399_39916


namespace NUMINAMATH_CALUDE_average_equation_l399_39910

theorem average_equation (a : ℝ) : 
  ((2 * a + 16) + (3 * a - 8)) / 2 = 89 → a = 34 := by
  sorry

end NUMINAMATH_CALUDE_average_equation_l399_39910


namespace NUMINAMATH_CALUDE_fraction_sum_l399_39945

theorem fraction_sum : (1 : ℚ) / 6 + (5 : ℚ) / 12 = (7 : ℚ) / 12 := by sorry

end NUMINAMATH_CALUDE_fraction_sum_l399_39945


namespace NUMINAMATH_CALUDE_reduced_price_calculation_l399_39961

/-- Represents the price of oil in Rupees per kg -/
structure OilPrice where
  price : ℝ
  price_positive : price > 0

def reduction_percentage : ℝ := 0.30

def total_cost : ℝ := 700

def additional_quantity : ℝ := 3

theorem reduced_price_calculation (original_price : OilPrice) :
  let reduced_price := original_price.price * (1 - reduction_percentage)
  let original_quantity := total_cost / original_price.price
  let new_quantity := total_cost / reduced_price
  new_quantity = original_quantity + additional_quantity →
  reduced_price = 70 := by
  sorry

end NUMINAMATH_CALUDE_reduced_price_calculation_l399_39961


namespace NUMINAMATH_CALUDE_quotient_60_55_is_recurring_l399_39909

/-- Represents a recurring decimal with an integer part and a repeating fractional part -/
structure RecurringDecimal where
  integerPart : ℤ
  repeatingPart : ℕ

/-- The quotient of 60 divided by 55 as a recurring decimal -/
def quotient_60_55 : RecurringDecimal :=
  { integerPart := 1,
    repeatingPart := 9 }

/-- Theorem stating that 60 divided by 55 is equal to the recurring decimal 1.090909... -/
theorem quotient_60_55_is_recurring : (60 : ℚ) / 55 = 1 + (9 : ℚ) / 99 := by sorry

end NUMINAMATH_CALUDE_quotient_60_55_is_recurring_l399_39909


namespace NUMINAMATH_CALUDE_solve_equation_l399_39927

theorem solve_equation (x : ℝ) (h : x + 1 = 5) : x = 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l399_39927


namespace NUMINAMATH_CALUDE_range_of_m_l399_39903

-- Define the proposition
def P (m : ℝ) : Prop := ∀ x : ℝ, 5^x + 3 > m

-- Theorem statement
theorem range_of_m :
  (∃ m : ℝ, P m) → (∀ m : ℝ, P m → m ≤ 3) ∧ (∀ y : ℝ, y < 3 → P y) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l399_39903


namespace NUMINAMATH_CALUDE_aubree_beaver_count_l399_39973

/-- The number of beavers Aubree initially saw -/
def initial_beavers : ℕ := 20

/-- The number of chipmunks Aubree initially saw -/
def initial_chipmunks : ℕ := 40

/-- The total number of animals Aubree saw -/
def total_animals : ℕ := 130

theorem aubree_beaver_count :
  initial_beavers = 20 ∧
  initial_chipmunks = 40 ∧
  total_animals = 130 ∧
  initial_beavers + initial_chipmunks + 2 * initial_beavers + (initial_chipmunks - 10) = total_animals :=
by sorry

end NUMINAMATH_CALUDE_aubree_beaver_count_l399_39973


namespace NUMINAMATH_CALUDE_circle_symmetry_l399_39949

/-- Given a circle C1 with equation (x+1)^2 + (y-1)^2 = 1, 
    prove that the circle C2 with equation (x-2)^2 + (y+2)^2 = 1 
    is symmetric to C1 with respect to the line x - y - 1 = 0 -/
theorem circle_symmetry (x y : ℝ) : 
  (∀ x y, (x + 1)^2 + (y - 1)^2 = 1 → 
    ∃ x' y', x' - y' = -(x - y) ∧ (x' + 1)^2 + (y' - 1)^2 = 1) → 
  (x - 2)^2 + (y + 2)^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_circle_symmetry_l399_39949


namespace NUMINAMATH_CALUDE_systematic_sampling_interval_l399_39975

-- Define the total population size
def N : ℕ := 1200

-- Define the sample size
def n : ℕ := 30

-- Define the systematic sampling interval
def k : ℕ := N / n

-- Theorem to prove
theorem systematic_sampling_interval :
  k = 40 :=
by sorry

end NUMINAMATH_CALUDE_systematic_sampling_interval_l399_39975


namespace NUMINAMATH_CALUDE_linear_function_passes_through_point_l399_39998

/-- The linear function f(x) = -2x - 6 passes through the point (-4, 2) -/
theorem linear_function_passes_through_point :
  let f : ℝ → ℝ := λ x => -2 * x - 6
  f (-4) = 2 := by sorry

end NUMINAMATH_CALUDE_linear_function_passes_through_point_l399_39998


namespace NUMINAMATH_CALUDE_victors_initial_money_l399_39947

/-- Victor's money problem -/
theorem victors_initial_money (initial_amount allowance total : ℕ) : 
  allowance = 8 → total = 18 → initial_amount + allowance = total → initial_amount = 10 := by
  sorry

end NUMINAMATH_CALUDE_victors_initial_money_l399_39947


namespace NUMINAMATH_CALUDE_sin_45_cos_15_plus_cos_45_sin_15_l399_39976

theorem sin_45_cos_15_plus_cos_45_sin_15 :
  Real.sin (45 * π / 180) * Real.cos (15 * π / 180) + 
  Real.cos (45 * π / 180) * Real.sin (15 * π / 180) = 
  Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_sin_45_cos_15_plus_cos_45_sin_15_l399_39976


namespace NUMINAMATH_CALUDE_xyz_value_l399_39951

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 12)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 16) :
  x * y * z = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_xyz_value_l399_39951


namespace NUMINAMATH_CALUDE_common_difference_is_fifteen_exists_valid_prism_l399_39939

/-- Represents a rectangular prism with sides that are consecutive multiples of a certain number -/
structure RectangularPrism where
  base_number : ℕ
  common_difference : ℕ

/-- The base area of the rectangular prism -/
def base_area (prism : RectangularPrism) : ℕ :=
  prism.base_number * (prism.base_number + prism.common_difference)

/-- Theorem stating that for a rectangular prism with base area 450,
    the common difference between consecutive multiples is 15 -/
theorem common_difference_is_fifteen (prism : RectangularPrism) 
    (h : base_area prism = 450) : prism.common_difference = 15 := by
  sorry

/-- Proof of the existence of a rectangular prism satisfying the conditions -/
theorem exists_valid_prism : ∃ (prism : RectangularPrism), base_area prism = 450 ∧ prism.common_difference = 15 := by
  sorry

end NUMINAMATH_CALUDE_common_difference_is_fifteen_exists_valid_prism_l399_39939


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l399_39932

/-- A geometric sequence with a_2 = 2 and a_10 = 8 has a_6 = 4 -/
theorem geometric_sequence_property (a : ℕ → ℝ) :
  (∀ n m : ℕ, a (n + m) = a n * a m) →  -- geometric sequence property
  a 2 = 2 →
  a 10 = 8 →
  a 6 = 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l399_39932


namespace NUMINAMATH_CALUDE_conference_handshakes_l399_39970

/-- Represents a conference with two groups of people -/
structure Conference where
  total_people : ℕ
  group_x : ℕ
  group_y : ℕ
  known_people : ℕ
  h_total : total_people = group_x + group_y
  h_group_x : group_x = 25
  h_group_y : group_y = 15
  h_known : known_people = 5

/-- Calculates the number of handshakes in the conference -/
def handshakes (c : Conference) : ℕ :=
  let between_groups := c.group_x * c.group_y
  let within_x := (c.group_x * (c.group_x - 1 - c.known_people)) / 2
  let within_y := (c.group_y * (c.group_y - 1)) / 2
  between_groups + within_x + within_y

/-- Theorem stating that the number of handshakes in the given conference is 717 -/
theorem conference_handshakes :
    ∃ (c : Conference), handshakes c = 717 :=
  sorry

end NUMINAMATH_CALUDE_conference_handshakes_l399_39970


namespace NUMINAMATH_CALUDE_library_experience_problem_l399_39972

/-- Represents the years of experience of a library employee -/
structure LibraryExperience where
  current : ℕ
  fiveYearsAgo : ℕ

/-- Represents the age and experience of a library employee -/
structure Employee where
  name : String
  age : ℕ
  experience : LibraryExperience

/-- The problem statement -/
theorem library_experience_problem 
  (bill : Employee)
  (joan : Employee)
  (h1 : bill.age = 40)
  (h2 : joan.age = 50)
  (h3 : joan.experience.fiveYearsAgo = 3 * bill.experience.fiveYearsAgo)
  (h4 : joan.experience.current = 2 * bill.experience.current)
  (h5 : bill.experience.current = bill.experience.fiveYearsAgo + 5)
  (h6 : ∃ (total_experience : ℕ), total_experience = bill.experience.current + 5) :
  bill.experience.current = 10 := by
  sorry

end NUMINAMATH_CALUDE_library_experience_problem_l399_39972


namespace NUMINAMATH_CALUDE_number_equal_nine_l399_39930

theorem number_equal_nine : ∃ x : ℝ, x^6 = 3^12 ∧ x = 9 := by
  sorry

end NUMINAMATH_CALUDE_number_equal_nine_l399_39930


namespace NUMINAMATH_CALUDE_range_of_m_l399_39995

theorem range_of_m (x m : ℝ) : 
  (∀ x, (|x - 4| ≤ 6 → x ≤ 1 + m) ∧ 
  ¬(x ≤ 1 + m → |x - 4| ≤ 6)) → 
  m ∈ Set.Ici 9 := by
sorry

end NUMINAMATH_CALUDE_range_of_m_l399_39995


namespace NUMINAMATH_CALUDE_smallest_number_from_digits_l399_39912

def digits : List Nat := [2, 0, 1, 6]

def isValidPermutation (n : Nat) : Bool :=
  let digits_n := n.digits 10
  digits_n.length == 4 && digits_n.head? != some 0 && digits_n.toFinset == digits.toFinset

theorem smallest_number_from_digits :
  ∀ n : Nat, isValidPermutation n → 1026 ≤ n := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_from_digits_l399_39912


namespace NUMINAMATH_CALUDE_range_of_a_l399_39940

theorem range_of_a (a : ℝ) : 
  (∀ x y z : ℝ, x + y + z = 1 → |a - 2| ≤ x^2 + 2*y^2 + 3*z^2) →
  16/11 ≤ a ∧ a ≤ 28/11 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l399_39940


namespace NUMINAMATH_CALUDE_fourth_root_equivalence_l399_39982

theorem fourth_root_equivalence (x : ℝ) (hx : x > 0) : 
  (x / x^(1/2))^(1/4) = x^(1/8) := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_equivalence_l399_39982


namespace NUMINAMATH_CALUDE_spending_percentage_l399_39915

/-- Represents Roger's entertainment budget and spending --/
structure Entertainment where
  budget : ℝ
  movie_cost : ℝ
  soda_cost : ℝ
  popcorn_cost : ℝ
  tax_rate : ℝ

/-- Calculates the total spending including tax --/
def total_spending (e : Entertainment) : ℝ :=
  (e.movie_cost + e.soda_cost + e.popcorn_cost) * (1 + e.tax_rate)

/-- Theorem stating that the total spending is approximately 28% of the budget --/
theorem spending_percentage (e : Entertainment) 
  (h1 : e.movie_cost = 0.25 * (e.budget - e.soda_cost))
  (h2 : e.soda_cost = 0.10 * (e.budget - e.movie_cost))
  (h3 : e.popcorn_cost = 5)
  (h4 : e.tax_rate = 0.10)
  (h5 : e.budget > 0) :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |total_spending e / e.budget - 0.28| < ε :=
sorry

end NUMINAMATH_CALUDE_spending_percentage_l399_39915


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficients_l399_39962

theorem binomial_expansion_coefficients :
  ∀ (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ),
  (∀ x : ℝ, (1 + 2*x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  (a₀ = 1 ∧ a₁ + a₃ + a₅ = 122) :=
by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficients_l399_39962


namespace NUMINAMATH_CALUDE_equation_solutions_l399_39936

theorem equation_solutions :
  (∃ y : ℝ, 6 - 3*y = 15 + 6*y ∧ y = -1) ∧
  (∃ x : ℝ, (1 - 2*x) / 3 = (3*x + 1) / 7 - 2 ∧ x = 2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l399_39936


namespace NUMINAMATH_CALUDE_tangent_slope_angle_l399_39922

open Real

theorem tangent_slope_angle (f : ℝ → ℝ) (x : ℝ) :
  f = (λ x => Real.log (x^2 + 1)) →
  x = 1 →
  let slope := (deriv f) x
  let angle := Real.arctan slope
  angle = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_tangent_slope_angle_l399_39922


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l399_39926

/-- Given a hyperbola with the following properties:
    - Equation: x²/a² - y²/b² = 1, where a > 0 and b > 0
    - O is the origin
    - F₁ and F₂ are the left and right foci
    - P is a point on the left branch
    - M is the midpoint of F₂P
    - |OM| = c/5, where c is the focal distance
    Then the eccentricity e of the hyperbola satisfies 1 < e ≤ 5/3 -/
theorem hyperbola_eccentricity_range 
  (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0)
  (O : ℝ × ℝ) (F₁ F₂ P M : ℝ × ℝ)
  (hyperbola_eq : ∀ (x y : ℝ), x^2/a^2 - y^2/b^2 = 1 → (x, y) ∈ {p : ℝ × ℝ | p.1^2/a^2 - p.2^2/b^2 = 1})
  (O_origin : O = (0, 0))
  (F₁_left : F₁.1 < 0)
  (F₂_right : F₂.1 > 0)
  (P_left_branch : P.1 < 0)
  (M_midpoint : M = ((F₂.1 + P.1)/2, (F₂.2 + P.2)/2))
  (OM_length : Real.sqrt ((M.1 - O.1)^2 + (M.2 - O.2)^2) = c/5)
  (e_def : e = c/a) :
  1 < e ∧ e ≤ 5/3 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l399_39926


namespace NUMINAMATH_CALUDE_marie_magazines_sold_l399_39917

/-- The number of magazines Marie sold -/
def magazines_sold : ℕ := 700 - 275

/-- The total number of reading materials Marie sold -/
def total_reading_materials : ℕ := 700

/-- The number of newspapers Marie sold -/
def newspapers_sold : ℕ := 275

theorem marie_magazines_sold :
  magazines_sold = 425 ∧
  magazines_sold + newspapers_sold = total_reading_materials :=
sorry

end NUMINAMATH_CALUDE_marie_magazines_sold_l399_39917


namespace NUMINAMATH_CALUDE_inequality_proof_l399_39978

theorem inequality_proof (a b x y : ℝ) 
  (ha : a > 0) (hb : b > 0) (hx : x > 0) (hy : y > 0) : 
  a^2 / x + b^2 / y ≥ (a + b)^2 / (x + y) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l399_39978


namespace NUMINAMATH_CALUDE_pushups_percentage_l399_39954

def jumping_jacks : ℕ := 12
def pushups : ℕ := 8
def situps : ℕ := 20

def total_exercises : ℕ := jumping_jacks + pushups + situps

def percentage_pushups : ℚ := (pushups : ℚ) / (total_exercises : ℚ) * 100

theorem pushups_percentage : percentage_pushups = 20 := by
  sorry

end NUMINAMATH_CALUDE_pushups_percentage_l399_39954


namespace NUMINAMATH_CALUDE_inequality_proof_l399_39933

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  (1 / (a^5 + b^5 + c^2)) + (1 / (b^5 + c^5 + a^2)) + (1 / (c^5 + a^5 + b^2)) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l399_39933


namespace NUMINAMATH_CALUDE_richard_remaining_distance_l399_39931

/-- Calculates the remaining distance Richard has to walk to reach New York City. -/
def remaining_distance (total_distance day1_distance day2_fraction day2_reduction day3_distance : ℝ) : ℝ :=
  let day2_distance := day1_distance * day2_fraction - day2_reduction
  let distance_walked := day1_distance + day2_distance + day3_distance
  total_distance - distance_walked

/-- Theorem stating that Richard has 36 miles left to walk to reach New York City. -/
theorem richard_remaining_distance :
  remaining_distance 70 20 (1/2) 6 10 = 36 := by
  sorry

end NUMINAMATH_CALUDE_richard_remaining_distance_l399_39931


namespace NUMINAMATH_CALUDE_investment_growth_l399_39946

-- Define the compound interest function
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

-- Define the problem parameters
def initial_investment : ℝ := 15000
def interest_rate : ℝ := 0.04
def investment_time : ℕ := 10

-- State the theorem
theorem investment_growth :
  round (compound_interest initial_investment interest_rate investment_time) = 22204 := by
  sorry


end NUMINAMATH_CALUDE_investment_growth_l399_39946


namespace NUMINAMATH_CALUDE_smallest_angle_measure_l399_39920

/-- Represents a triangle with angles in degrees -/
structure Triangle where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ
  sum_180 : angle1 + angle2 + angle3 = 180
  all_positive : 0 < angle1 ∧ 0 < angle2 ∧ 0 < angle3

/-- An isosceles, obtuse triangle with one angle 50% larger than a right angle -/
def special_triangle : Triangle :=
  { angle1 := 135
    angle2 := 22.5
    angle3 := 22.5
    sum_180 := by sorry
    all_positive := by sorry }

theorem smallest_angle_measure :
  ∃ (t : Triangle), 
    (t.angle1 = 90 * 1.5) ∧  -- One angle is 50% larger than right angle
    (t.angle2 = t.angle3) ∧  -- Isosceles property
    (t.angle1 > 90) ∧        -- Obtuse triangle
    (t.angle2 = 22.5 ∧ t.angle3 = 22.5) -- The two smallest angles
    := by
  sorry

end NUMINAMATH_CALUDE_smallest_angle_measure_l399_39920


namespace NUMINAMATH_CALUDE_f_is_linear_l399_39906

/-- A function f: ℝ → ℝ is linear if there exist constants m and b such that 
    f(x) = mx + b for all x, where m ≠ 0 -/
def IsLinearFunction (f : ℝ → ℝ) : Prop :=
  ∃ (m b : ℝ), m ≠ 0 ∧ ∀ x, f x = m * x + b

/-- The function f(x) = -8x -/
def f (x : ℝ) : ℝ := -8 * x

/-- Theorem: f(x) = -8x is a linear function -/
theorem f_is_linear : IsLinearFunction f := by
  sorry


end NUMINAMATH_CALUDE_f_is_linear_l399_39906


namespace NUMINAMATH_CALUDE_intersection_A_B_range_of_a_l399_39908

-- Define the sets A, B, and C
def A : Set ℝ := {x | -1 ≤ x ∧ x < 3}
def B : Set ℝ := {x | 2*x - 4 ≥ x - 2}
def C (a : ℝ) : Set ℝ := {x | x ≥ a - 1}

-- Theorem 1: Intersection of A and B
theorem intersection_A_B : A ∩ B = {x : ℝ | 2 ≤ x ∧ x < 3} := by sorry

-- Theorem 2: Range of a given B ∪ C = C
theorem range_of_a (a : ℝ) (h : B ∪ C a = C a) : a ≤ 3 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_range_of_a_l399_39908


namespace NUMINAMATH_CALUDE_eagles_win_probability_l399_39983

/-- The number of games in the series -/
def n : ℕ := 5

/-- The probability of winning a single game -/
def p : ℚ := 1/2

/-- The probability of winning exactly k games out of n -/
def prob_win (k : ℕ) : ℚ :=
  (n.choose k) * p^k * (1-p)^(n-k)

/-- The probability of winning at least 3 games out of 5 -/
def prob_win_at_least_three : ℚ :=
  prob_win 3 + prob_win 4 + prob_win 5

theorem eagles_win_probability : prob_win_at_least_three = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_eagles_win_probability_l399_39983


namespace NUMINAMATH_CALUDE_jake_candy_cost_l399_39963

/-- The cost of a single candy given Jake's feeding allowance and sharing behavior -/
def candy_cost (feeding_allowance : ℚ) (share_fraction : ℚ) (candies_bought : ℕ) : ℚ :=
  (feeding_allowance * share_fraction) / candies_bought

theorem jake_candy_cost :
  candy_cost 4 (1/4) 5 = 1/5 := by sorry

end NUMINAMATH_CALUDE_jake_candy_cost_l399_39963


namespace NUMINAMATH_CALUDE_monthly_interest_advantage_l399_39952

theorem monthly_interest_advantage (p : ℝ) (n : ℕ) (hp : p > 0) (hn : n > 0) :
  (1 + p / (12 * 100)) ^ (6 * n) > (1 + p / (2 * 100)) ^ n :=
sorry

end NUMINAMATH_CALUDE_monthly_interest_advantage_l399_39952


namespace NUMINAMATH_CALUDE_expression_evaluation_l399_39985

theorem expression_evaluation : 
  3 + 2 * Real.sqrt 3 + (3 + 2 * Real.sqrt 3)⁻¹ + (2 * Real.sqrt 3 - 3)⁻¹ = 3 + (10 * Real.sqrt 3) / 3 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l399_39985


namespace NUMINAMATH_CALUDE_tan_equality_implies_negative_thirty_l399_39965

theorem tan_equality_implies_negative_thirty
  (n : ℤ)
  (h1 : -90 < n ∧ n < 90)
  (h2 : Real.tan (n * π / 180) = Real.tan (1230 * π / 180)) :
  n = -30 :=
by sorry

end NUMINAMATH_CALUDE_tan_equality_implies_negative_thirty_l399_39965


namespace NUMINAMATH_CALUDE_production_rates_theorem_l399_39974

-- Define the number of machines
def num_machines : ℕ := 5

-- Define the list of pairwise production numbers
def pairwise_production : List ℕ := [35, 39, 40, 49, 44, 46, 30, 41, 32, 36]

-- Define the function to check if a list of production rates is valid
def is_valid_production (rates : List ℕ) : Prop :=
  rates.length = num_machines ∧
  rates.sum = 98 ∧
  (∀ i j, i < j → i < rates.length → j < rates.length →
    (rates.get ⟨i, by sorry⟩ + rates.get ⟨j, by sorry⟩) ∈ pairwise_production)

-- Theorem statement
theorem production_rates_theorem :
  ∃ (rates : List ℕ), is_valid_production rates ∧ rates = [13, 17, 19, 22, 27] := by
  sorry

end NUMINAMATH_CALUDE_production_rates_theorem_l399_39974


namespace NUMINAMATH_CALUDE_wicket_keeper_age_difference_l399_39923

/-- Represents a cricket team with given properties -/
structure CricketTeam where
  total_members : ℕ
  captain_age : ℕ
  team_average_age : ℕ
  wicket_keeper_age_difference : ℕ

/-- The age difference between the wicket keeper and the captain is correct
    if it satisfies the given conditions -/
def correct_age_difference (team : CricketTeam) : Prop :=
  let remaining_players := team.total_members - 2
  let remaining_average := team.team_average_age - 1
  let total_age := team.team_average_age * team.total_members
  let remaining_age := remaining_average * remaining_players
  total_age = remaining_age + team.captain_age + (team.captain_age + team.wicket_keeper_age_difference)

/-- The theorem stating that the wicket keeper is 3 years older than the captain -/
theorem wicket_keeper_age_difference (team : CricketTeam) 
  (h1 : team.total_members = 11)
  (h2 : team.captain_age = 26)
  (h3 : team.team_average_age = 23)
  : correct_age_difference team → team.wicket_keeper_age_difference = 3 := by
  sorry

end NUMINAMATH_CALUDE_wicket_keeper_age_difference_l399_39923


namespace NUMINAMATH_CALUDE_factorial_divides_power_difference_l399_39950

theorem factorial_divides_power_difference (n : ℕ) : 
  (n.factorial : ℤ) ∣ (2^(2*n.factorial) - 2^n.factorial) :=
sorry

end NUMINAMATH_CALUDE_factorial_divides_power_difference_l399_39950


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l399_39984

/-- An increasing geometric sequence -/
def IsIncreasingGeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  (∀ n, a (n + 1) = a n * q) ∧ (q > 1) ∧ (a 1 > 0)

/-- The theorem stating the common ratio of the geometric sequence -/
theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) (q : ℝ) 
  (h_geom : IsIncreasingGeometricSequence a q) 
  (h_sum : a 1 + a 4 = 9) 
  (h_prod : a 2 * a 3 = 8) : 
  q = 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l399_39984


namespace NUMINAMATH_CALUDE_investment_interest_l399_39956

theorem investment_interest (total_investment : ℝ) (high_rate_investment : ℝ) (high_rate : ℝ) (low_rate : ℝ) :
  total_investment = 22000 →
  high_rate_investment = 7000 →
  high_rate = 0.18 →
  low_rate = 0.14 →
  let low_rate_investment := total_investment - high_rate_investment
  let high_rate_interest := high_rate_investment * high_rate
  let low_rate_interest := low_rate_investment * low_rate
  let total_interest := high_rate_interest + low_rate_interest
  total_interest = 3360 := by
sorry

end NUMINAMATH_CALUDE_investment_interest_l399_39956


namespace NUMINAMATH_CALUDE_kevin_kangaroo_hops_l399_39977

def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem kevin_kangaroo_hops :
  let a : ℚ := 1/4
  let r : ℚ := 7/16
  let n : ℕ := 5
  geometric_sum a r n = 1031769/2359296 := by sorry

end NUMINAMATH_CALUDE_kevin_kangaroo_hops_l399_39977


namespace NUMINAMATH_CALUDE_committee_formation_count_l399_39913

/-- The number of ways to choose a committee from a basketball team -/
def choose_committee (total_players : ℕ) (committee_size : ℕ) (total_guards : ℕ) : ℕ :=
  total_guards * (Nat.choose (total_players - total_guards) (committee_size - 1))

/-- Theorem: The number of ways to form the committee is 112 -/
theorem committee_formation_count :
  choose_committee 12 3 4 = 112 := by
  sorry

end NUMINAMATH_CALUDE_committee_formation_count_l399_39913


namespace NUMINAMATH_CALUDE_equation_rewrite_l399_39938

theorem equation_rewrite (a x y c : ℤ) :
  ∃ (m n p : ℕ), 
    (m = 4 ∧ n = 3 ∧ p = 4) ∧
    (a^8*x*y - a^7*y - a^6*x = a^5*(c^5 - 1)) ↔ 
    ((a^m*x - a^n)*(a^p*y - a^3) = a^5*c^5) := by
  sorry

end NUMINAMATH_CALUDE_equation_rewrite_l399_39938


namespace NUMINAMATH_CALUDE_product_from_lcm_and_gcd_l399_39979

theorem product_from_lcm_and_gcd (a b : ℕ) (h1 : a > 0) (h2 : b > 0) 
  (h_lcm : Nat.lcm a b = 48) (h_gcd : Nat.gcd a b = 8) : a * b = 384 := by
  sorry

end NUMINAMATH_CALUDE_product_from_lcm_and_gcd_l399_39979


namespace NUMINAMATH_CALUDE_range_of_x_when_a_is_neg_one_range_of_a_when_not_p_necessary_not_sufficient_for_not_q_l399_39959

-- Define propositions p and q
def p (x a : ℝ) : Prop := x^2 + 4*a*x + 3*a^2 < 0

def q (x : ℝ) : Prop := x^2 - 6*x - 72 ≤ 0 ∧ x^2 + x - 6 > 0

-- Part 1
theorem range_of_x_when_a_is_neg_one :
  (∀ x : ℝ, p x (-1) → q x) →
  ∀ x : ℝ, (x ∈ Set.Icc (-6) (-3) ∪ Set.Ioc 1 12) ↔ (p x (-1) ∨ q x) :=
sorry

-- Part 2
theorem range_of_a_when_not_p_necessary_not_sufficient_for_not_q :
  (∀ x a : ℝ, ¬(p x a) → ¬(q x)) ∧ 
  (∃ x a : ℝ, ¬(p x a) ∧ q x) →
  ∀ a : ℝ, a ∈ Set.Icc (-4) (-2) ↔ (∃ x : ℝ, p x a ∧ q x) :=
sorry

end NUMINAMATH_CALUDE_range_of_x_when_a_is_neg_one_range_of_a_when_not_p_necessary_not_sufficient_for_not_q_l399_39959


namespace NUMINAMATH_CALUDE_polynomial_roots_sum_l399_39990

theorem polynomial_roots_sum (p q : ℝ) : 
  (∃ a b c d : ℕ+, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    ∀ x : ℝ, x^4 - 10*x^3 + p*x^2 - q*x + 24 = 0 ↔ (x = a ∨ x = b ∨ x = c ∨ x = d)) →
  p + q = 85 := by
sorry

end NUMINAMATH_CALUDE_polynomial_roots_sum_l399_39990


namespace NUMINAMATH_CALUDE_watermelon_slices_l399_39993

/-- The number of slices in a watermelon, given the number of seeds per slice and the total number of seeds. -/
def number_of_slices (black_seeds_per_slice : ℕ) (white_seeds_per_slice : ℕ) (total_seeds : ℕ) : ℕ :=
  total_seeds / (black_seeds_per_slice + white_seeds_per_slice)

/-- Theorem stating that the number of slices is 40 given the conditions of the problem. -/
theorem watermelon_slices :
  number_of_slices 20 20 1600 = 40 := by
  sorry

end NUMINAMATH_CALUDE_watermelon_slices_l399_39993


namespace NUMINAMATH_CALUDE_billy_crayons_left_l399_39960

/-- The number of crayons Billy has left after a hippopotamus eats some. -/
def crayons_left (initial : ℕ) (eaten : ℕ) : ℕ := initial - eaten

/-- Theorem stating that Billy has 163 crayons left after starting with 856 and a hippopotamus eating 693. -/
theorem billy_crayons_left : crayons_left 856 693 = 163 := by
  sorry

end NUMINAMATH_CALUDE_billy_crayons_left_l399_39960


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l399_39999

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem complex_fraction_simplification :
  (2 - 2 * i) / (1 + 4 * i) = -6 / 17 - (10 / 17) * i :=
by
  -- The proof would go here, but we'll skip it
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l399_39999


namespace NUMINAMATH_CALUDE_validSchedules_eq_1296_l399_39948

/-- Represents a chess tournament between two universities -/
structure ChessTournament where
  university1 : Fin 3 → Type
  university2 : Fin 3 → Type
  rounds : Fin 6 → Fin 3 → (Fin 3 × Fin 3)
  no_immediate_repeat : ∀ (r : Fin 5) (i : Fin 3),
    (rounds r i).1 ≠ (rounds (r + 1) i).1 ∨ (rounds r i).2 ≠ (rounds (r + 1) i).2

/-- The number of valid tournament schedules -/
def validSchedules : ℕ := sorry

/-- Theorem stating the number of valid tournament schedules is 1296 -/
theorem validSchedules_eq_1296 : validSchedules = 1296 := by sorry

end NUMINAMATH_CALUDE_validSchedules_eq_1296_l399_39948


namespace NUMINAMATH_CALUDE_anne_bottle_caps_l399_39953

/-- Anne's initial bottle cap count -/
def initial_count : ℕ := 10

/-- Number of bottle caps Anne finds -/
def found_count : ℕ := 5

/-- Anne's final bottle cap count -/
def final_count : ℕ := initial_count + found_count

theorem anne_bottle_caps : final_count = 15 := by
  sorry

end NUMINAMATH_CALUDE_anne_bottle_caps_l399_39953


namespace NUMINAMATH_CALUDE_log_equation_solution_l399_39971

theorem log_equation_solution :
  ∃ y : ℝ, y > 0 ∧ 2 * Real.log y - 4 * Real.log 2 = 2 → y = 160 :=
by sorry

end NUMINAMATH_CALUDE_log_equation_solution_l399_39971


namespace NUMINAMATH_CALUDE_line_segment_point_sum_l399_39987

/-- Given a line y = -2/3x + 6 that crosses the x-axis at P and y-axis at Q,
    and a point T(r, s) on line segment PQ, prove that if the area of triangle POQ
    is four times the area of triangle TOP, then r + s = 8.25. -/
theorem line_segment_point_sum (r s : ℝ) : 
  let line := fun (x : ℝ) ↦ -2/3 * x + 6
  let P := (9, 0)
  let Q := (0, 6)
  let T := (r, s)
  (T.1 ≥ 0 ∧ T.1 ≤ 9) →  -- T is on line segment PQ
  (T.2 = line T.1) →  -- T is on the line
  (1/2 * 9 * 6 = 4 * (1/2 * 9 * s)) →  -- Area condition
  r + s = 8.25 :=
by sorry

end NUMINAMATH_CALUDE_line_segment_point_sum_l399_39987


namespace NUMINAMATH_CALUDE_birthday_age_multiple_l399_39955

theorem birthday_age_multiple :
  let current_age : ℕ := 9
  let years_ago : ℕ := 6
  let age_then : ℕ := current_age - years_ago
  current_age / age_then = 3 :=
by sorry

end NUMINAMATH_CALUDE_birthday_age_multiple_l399_39955


namespace NUMINAMATH_CALUDE_class_weight_problem_l399_39964

theorem class_weight_problem (total_boys : Nat) (group_boys : Nat) (group_avg : Real) (total_avg : Real) :
  total_boys = 34 →
  group_boys = 26 →
  group_avg = 50.25 →
  total_avg = 49.05 →
  let remaining_boys := total_boys - group_boys
  let remaining_avg := (total_boys * total_avg - group_boys * group_avg) / remaining_boys
  remaining_avg = 45.15 := by
  sorry

end NUMINAMATH_CALUDE_class_weight_problem_l399_39964


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l399_39941

theorem quadratic_inequality_solution (x : ℝ) : 3 * x^2 - 2 * x + 1 > 7 ↔ x < -2/3 ∨ x > 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l399_39941


namespace NUMINAMATH_CALUDE_no_integer_solutions_l399_39901

def ω : ℂ := Complex.I

theorem no_integer_solutions : ∀ a b : ℤ, (Complex.abs (a • ω + b) ≠ Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l399_39901


namespace NUMINAMATH_CALUDE_infinite_greater_than_index_l399_39943

/-- A sequence of integers -/
def IntegerSequence := ℕ → ℤ

/-- Property: all elements in the sequence are pairwise distinct -/
def PairwiseDistinct (a : IntegerSequence) : Prop :=
  ∀ i j, i ≠ j → a i ≠ a j

/-- Property: all elements in the sequence are greater than 1 -/
def AllGreaterThanOne (a : IntegerSequence) : Prop :=
  ∀ k, a k > 1

/-- The main theorem -/
theorem infinite_greater_than_index
  (a : IntegerSequence)
  (h_distinct : PairwiseDistinct a)
  (h_greater : AllGreaterThanOne a) :
  ∃ S : Set ℕ, (Set.Infinite S) ∧ (∀ k ∈ S, a k > k) :=
sorry

end NUMINAMATH_CALUDE_infinite_greater_than_index_l399_39943


namespace NUMINAMATH_CALUDE_complex_division_by_i_l399_39980

theorem complex_division_by_i (i : ℂ) (h : i^2 = -1) : 
  (3 + 4*i) / i = 4 - 3*i := by sorry

end NUMINAMATH_CALUDE_complex_division_by_i_l399_39980


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l399_39944

theorem imaginary_part_of_z (i : ℂ) (z : ℂ) : 
  i^2 = -1 → z = i * (i - 1) → Complex.im z = -1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l399_39944


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_35_second_smallest_four_digit_divisible_by_35_l399_39918

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def divisible_by_35 (n : ℕ) : Prop := n % 35 = 0

theorem smallest_four_digit_divisible_by_35 :
  (∀ n : ℕ, is_four_digit n ∧ divisible_by_35 n → 1005 ≤ n) ∧
  is_four_digit 1005 ∧ divisible_by_35 1005 :=
sorry

theorem second_smallest_four_digit_divisible_by_35 :
  (∀ n : ℕ, is_four_digit n ∧ divisible_by_35 n ∧ n ≠ 1005 → 1045 ≤ n) ∧
  is_four_digit 1045 ∧ divisible_by_35 1045 :=
sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_35_second_smallest_four_digit_divisible_by_35_l399_39918


namespace NUMINAMATH_CALUDE_smallest_number_l399_39968

theorem smallest_number (a b c d : ℤ) (ha : a = 2) (hb : b = 1) (hc : c = -1) (hd : d = -2) :
  d ≤ a ∧ d ≤ b ∧ d ≤ c := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_l399_39968


namespace NUMINAMATH_CALUDE_parallelogram_height_eq_two_thirds_rectangle_side_l399_39924

/-- Given a rectangle with side length r and a parallelogram with base b = 1.5r,
    prove that the height of the parallelogram h = 2r/3 when their areas are equal. -/
theorem parallelogram_height_eq_two_thirds_rectangle_side 
  (r : ℝ) (b h : ℝ) (h_positive : r > 0) : 
  b = 1.5 * r → r * r = b * h → h = 2 * r / 3 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_height_eq_two_thirds_rectangle_side_l399_39924


namespace NUMINAMATH_CALUDE_intersection_theorem_l399_39914

-- Define the sets M and N
def M : Set ℝ := {x | x^2 < 4}
def N : Set ℝ := {x | x^2 - 2*x - 3 < 0}

-- Define the intersection set
def intersection_set : Set ℝ := {x | -1 < x ∧ x < 2}

-- Theorem statement
theorem intersection_theorem : M ∩ N = intersection_set := by
  sorry

end NUMINAMATH_CALUDE_intersection_theorem_l399_39914


namespace NUMINAMATH_CALUDE_vertical_pairwise_sets_l399_39928

-- Definition of a vertical pairwise set
def is_vertical_pairwise_set (M : Set (ℝ × ℝ)) : Prop :=
  ∀ (p₁ : ℝ × ℝ), p₁ ∈ M → ∃ (p₂ : ℝ × ℝ), p₂ ∈ M ∧ p₁.1 * p₂.1 + p₁.2 * p₂.2 = 0

-- Define the four sets
def M₁ : Set (ℝ × ℝ) := {p | p.2 = 1 / p.1 ∧ p.1 ≠ 0}
def M₂ : Set (ℝ × ℝ) := {p | p.2 = Real.log p.1 ∧ p.1 > 0}
def M₃ : Set (ℝ × ℝ) := {p | p.2 = Real.exp p.1 - 2}
def M₄ : Set (ℝ × ℝ) := {p | p.2 = Real.sin p.1 + 1}

-- Theorem stating which sets are vertical pairwise sets
theorem vertical_pairwise_sets :
  ¬(is_vertical_pairwise_set M₁) ∧
  ¬(is_vertical_pairwise_set M₂) ∧
  (is_vertical_pairwise_set M₃) ∧
  (is_vertical_pairwise_set M₄) := by
  sorry

end NUMINAMATH_CALUDE_vertical_pairwise_sets_l399_39928


namespace NUMINAMATH_CALUDE_exactly_one_correct_proposition_l399_39989

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation between lines and planes
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the parallel relation between lines
variable (parallel_line_line : Line → Line → Prop)

-- Define the perpendicular relation between lines and planes
variable (perpendicular_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between lines
variable (perpendicular_line_line : Line → Line → Prop)

-- Define the subset relation for a line in a plane
variable (line_in_plane : Line → Plane → Prop)

-- Define the "not subset" relation for a line not in a plane
variable (line_not_in_plane : Line → Plane → Prop)

theorem exactly_one_correct_proposition (a b : Line) (M : Plane) : 
  (∃! i : Fin 4, 
    (i = 0 → (parallel_line_plane a M ∧ parallel_line_plane b M → parallel_line_line a b)) ∧
    (i = 1 → (line_in_plane b M ∧ line_not_in_plane a M ∧ parallel_line_line a b → parallel_line_plane a M)) ∧
    (i = 2 → (perpendicular_line_line a b ∧ line_in_plane b M → perpendicular_line_plane a M)) ∧
    (i = 3 → (perpendicular_line_plane a M ∧ perpendicular_line_line a b → parallel_line_plane b M))) :=
by sorry

end NUMINAMATH_CALUDE_exactly_one_correct_proposition_l399_39989


namespace NUMINAMATH_CALUDE_probability_four_twos_in_five_rolls_l399_39957

theorem probability_four_twos_in_five_rolls (p : ℝ) :
  p = 1 / 8 →  -- probability of rolling a 2 on a fair 8-sided die
  (5 : ℝ) * p^4 * (1 - p) = 35 / 4096 := by
  sorry

end NUMINAMATH_CALUDE_probability_four_twos_in_five_rolls_l399_39957


namespace NUMINAMATH_CALUDE_sally_score_l399_39937

/-- Calculates the total score in a math competition given the number of correct, incorrect, and unanswered questions. -/
def calculate_score (correct : ℕ) (incorrect : ℕ) (unanswered : ℕ) : ℚ :=
  (correct : ℚ) - 0.25 * (incorrect : ℚ)

/-- Theorem: Sally's total score in the math competition is 15 points. -/
theorem sally_score :
  let correct : ℕ := 17
  let incorrect : ℕ := 8
  let unanswered : ℕ := 5
  calculate_score correct incorrect unanswered = 15 := by
  sorry

end NUMINAMATH_CALUDE_sally_score_l399_39937


namespace NUMINAMATH_CALUDE_first_player_wins_l399_39986

/-- Represents the game state with k piles of stones -/
structure GameState where
  k : ℕ+
  n : Fin k → ℕ

/-- Defines the set of winning positions -/
def WinningPositions : Set GameState :=
  sorry

/-- Defines a valid move in the game -/
def ValidMove (s₁ s₂ : GameState) : Prop :=
  sorry

/-- Theorem stating the winning condition for the first player -/
theorem first_player_wins (s : GameState) :
  s ∈ WinningPositions ↔
    ∃ (s' : GameState), ValidMove s s' ∧ 
      ∀ (s'' : GameState), ValidMove s' s'' → s'' ∈ WinningPositions :=
by sorry

end NUMINAMATH_CALUDE_first_player_wins_l399_39986


namespace NUMINAMATH_CALUDE_x_value_from_fraction_equality_l399_39969

theorem x_value_from_fraction_equality (x y : ℝ) :
  x / (x - 1) = (y^2 + 3*y + 2) / (y^2 + 3*y - 1) →
  x = (y^2 + 3*y + 2) / 3 := by
sorry

end NUMINAMATH_CALUDE_x_value_from_fraction_equality_l399_39969


namespace NUMINAMATH_CALUDE_hoseok_english_score_l399_39994

theorem hoseok_english_score 
  (korean_math_avg : ℝ) 
  (all_subjects_avg : ℝ) 
  (h1 : korean_math_avg = 88)
  (h2 : all_subjects_avg = 90) :
  ∃ (korean math english : ℝ),
    (korean + math) / 2 = korean_math_avg ∧
    (korean + math + english) / 3 = all_subjects_avg ∧
    english = 94 := by
  sorry

end NUMINAMATH_CALUDE_hoseok_english_score_l399_39994


namespace NUMINAMATH_CALUDE_journey_time_increase_l399_39997

theorem journey_time_increase (total_distance : ℝ) (first_half_speed : ℝ) (average_speed : ℝ) :
  total_distance = 640 →
  first_half_speed = 80 →
  average_speed = 40 →
  let first_half_time := (total_distance / 2) / first_half_speed
  let total_time := total_distance / average_speed
  let second_half_time := total_time - first_half_time
  ((second_half_time - first_half_time) / first_half_time) * 100 = 200 := by
sorry

end NUMINAMATH_CALUDE_journey_time_increase_l399_39997


namespace NUMINAMATH_CALUDE_digit_placement_l399_39904

theorem digit_placement (n : ℕ) (h : n < 10) :
  100 + 10 * n + 1 = 101 + 10 * n := by
  sorry

end NUMINAMATH_CALUDE_digit_placement_l399_39904


namespace NUMINAMATH_CALUDE_geometric_sequence_general_term_l399_39981

/-- A geometric sequence {a_n} where a_3 = 2 and a_6 = 16 -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  (∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n) ∧ a 3 = 2 ∧ a 6 = 16

/-- The general term of the geometric sequence -/
def general_term (n : ℕ) : ℝ := 2^(n - 2)

theorem geometric_sequence_general_term (a : ℕ → ℝ) (h : geometric_sequence a) :
  ∀ n : ℕ, n ≥ 1 → a n = general_term n :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_general_term_l399_39981


namespace NUMINAMATH_CALUDE_complex_power_one_minus_i_six_l399_39996

theorem complex_power_one_minus_i_six :
  (1 - Complex.I : ℂ)^6 = 8 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_power_one_minus_i_six_l399_39996


namespace NUMINAMATH_CALUDE_impossibility_of_simultaneous_inequalities_l399_39935

theorem impossibility_of_simultaneous_inequalities 
  (a b c : Real) 
  (ha : 0 < a ∧ a < 1) 
  (hb : 0 < b ∧ b < 1) 
  (hc : 0 < c ∧ c < 1) : 
  ¬(a * (1 - b) > 1/4 ∧ b * (1 - c) > 1/4 ∧ c * (1 - a) > 1/4) := by
sorry

end NUMINAMATH_CALUDE_impossibility_of_simultaneous_inequalities_l399_39935


namespace NUMINAMATH_CALUDE_f_monotonicity_and_extrema_l399_39907

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + (2*a - 1) * x - Real.log x

theorem f_monotonicity_and_extrema :
  (∀ x > 0, ∀ a : ℝ,
    (a = 1/2 →
      (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 → f a x₁ > f a x₂) ∧
      (∀ x₁ x₂, 1 < x₁ ∧ x₁ < x₂ → f a x₁ < f a x₂) ∧
      (f a 1 = 1/2) ∧
      (∀ x ≠ 1, f a x > 1/2)) ∧
    (a ≤ 0 →
      (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ → f a x₁ > f a x₂)) ∧
    (a > 0 →
      (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1/(2*a) → f a x₁ > f a x₂) ∧
      (∀ x₁ x₂, 1/(2*a) < x₁ ∧ x₁ < x₂ → f a x₁ < f a x₂))) :=
by sorry

end NUMINAMATH_CALUDE_f_monotonicity_and_extrema_l399_39907


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l399_39902

/-- The speed of a boat in still water given its travel distances with and against a stream. -/
theorem boat_speed_in_still_water (along_stream : ℝ) (against_stream : ℝ) 
  (h_along : along_stream = 11) 
  (h_against : against_stream = 3) : 
  (along_stream + against_stream) / 2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l399_39902


namespace NUMINAMATH_CALUDE_boxes_left_to_sell_l399_39958

/-- Represents the number of cookie boxes sold to each customer --/
structure CustomerSales where
  first : ℕ
  second : ℕ
  third : ℕ
  fourth : ℕ
  fifth : ℕ

/-- Calculates the total number of boxes sold --/
def totalSold (sales : CustomerSales) : ℕ :=
  sales.first + sales.second + sales.third + sales.fourth + sales.fifth

/-- Represents Jill's cookie sales --/
def jillSales : CustomerSales where
  first := 5
  second := 4 * 5
  third := (4 * 5) / 2
  fourth := 3 * ((4 * 5) / 2)
  fifth := 10

/-- Jill's sales goal --/
def salesGoal : ℕ := 150

/-- Theorem stating that Jill has 75 boxes left to sell to reach her goal --/
theorem boxes_left_to_sell : salesGoal - totalSold jillSales = 75 := by
  sorry

end NUMINAMATH_CALUDE_boxes_left_to_sell_l399_39958


namespace NUMINAMATH_CALUDE_inequality_preservation_l399_39925

theorem inequality_preservation (x y : ℝ) : x < y → 3 - x > 3 - y := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l399_39925


namespace NUMINAMATH_CALUDE_cos_arcsin_three_fifths_l399_39967

theorem cos_arcsin_three_fifths : Real.cos (Real.arcsin (3/5)) = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_cos_arcsin_three_fifths_l399_39967


namespace NUMINAMATH_CALUDE_twelve_integers_with_properties_l399_39921

theorem twelve_integers_with_properties : ∃ (S : Finset ℤ),
  (Finset.card S = 12) ∧
  (∃ (P : Finset ℤ), P ⊆ S ∧ Finset.card P = 6 ∧ ∀ p ∈ P, Nat.Prime p.natAbs) ∧
  (∃ (O : Finset ℤ), O ⊆ S ∧ Finset.card O = 9 ∧ ∀ n ∈ O, n % 2 ≠ 0) ∧
  (∃ (NN : Finset ℤ), NN ⊆ S ∧ Finset.card NN = 10 ∧ ∀ n ∈ NN, n ≥ 0) ∧
  (∃ (GT : Finset ℤ), GT ⊆ S ∧ Finset.card GT = 7 ∧ ∀ n ∈ GT, n > 10) :=
by
  sorry


end NUMINAMATH_CALUDE_twelve_integers_with_properties_l399_39921


namespace NUMINAMATH_CALUDE_parallelepiped_diagonal_bounds_l399_39992

/-- Regular tetrahedron with side length and height 1 -/
structure Tetrahedron where
  side_length : ℝ
  height : ℝ
  is_regular : side_length = 1 ∧ height = 1

/-- Rectangular parallelepiped inscribed in the tetrahedron -/
structure Parallelepiped (t : Tetrahedron) where
  base_area : ℝ
  base_in_tetrahedron_base : Prop
  opposite_vertex_on_lateral_surface : Prop
  diagonal : ℝ

/-- Theorem stating the bounds of the parallelepiped's diagonal -/
theorem parallelepiped_diagonal_bounds (t : Tetrahedron) (p : Parallelepiped t) :
  (0 < p.base_area ∧ p.base_area ≤ 1/18 →
    Real.sqrt (2/3 - 2*p.base_area) ≤ p.diagonal ∧ p.diagonal < Real.sqrt (2 - 2*p.base_area)) ∧
  ((7 + 2*Real.sqrt 6)/25 ≤ p.base_area ∧ p.base_area < 1/2 →
    Real.sqrt (1 - 2*Real.sqrt (2*p.base_area) + 4*p.base_area) ≤ p.diagonal ∧
    p.diagonal < Real.sqrt (1 - 2*Real.sqrt p.base_area + 3*p.base_area)) ∧
  (1/2 ≤ p.base_area ∧ p.base_area < 1 →
    Real.sqrt (2*p.base_area) < p.diagonal ∧
    p.diagonal ≤ Real.sqrt (1 - 2*Real.sqrt p.base_area + 3*p.base_area)) := by
  sorry

end NUMINAMATH_CALUDE_parallelepiped_diagonal_bounds_l399_39992
