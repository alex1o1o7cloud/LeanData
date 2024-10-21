import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l352_35281

-- Define an odd, monotonically decreasing function
noncomputable def f : ℝ → ℝ := sorry

-- Axioms for the properties of f
axiom f_odd : ∀ x : ℝ, f (-x) = -f x
axiom f_decreasing : ∀ x y : ℝ, x < y → f x > f y

-- Theorem representing the problem statement
theorem problem_statement (x₁ x₂ x₃ : ℝ) 
  (h1 : x₁ + x₂ > 0) (h2 : x₂ + x₃ > 0) (h3 : x₃ + x₁ > 0) :
  f x₁ + f x₂ + f x₃ < 0 :=
by
  -- Step 1: Prove that x₁ + x₂ + x₃ > 0
  have sum_positive : x₁ + x₂ + x₃ > 0 := by
    linarith [h1, h2, h3]
  
  -- The rest of the proof would go here
  -- We would use the properties of f and the fact that x₁ + x₂ + x₃ > 0
  -- to show that f x₁ + f x₂ + f x₃ < 0
  
  sorry -- We use sorry to skip the detailed proof for now

-- Example usage of the theorem
example (a b c : ℝ) (hab : a + b > 0) (hbc : b + c > 0) (hca : c + a > 0) :
  f a + f b + f c < 0 :=
problem_statement a b c hab hbc hca

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l352_35281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_total_net_earnings_eq_157_5_l352_35229

/-- Calculate the total net earnings for Kem, Shem, and Tiff given their hourly rates, work hours, and deductions. -/
def total_net_earnings (kem_hourly_rate : ℝ) (kem_hours : ℝ) (shem_hours : ℝ) (tiff_hours : ℝ) 
  (kem_tax_rate : ℝ) (shem_tax_rate : ℝ) (shem_health_deduction : ℝ) 
  (tiff_pension_rate : ℝ) (tiff_transport_deduction : ℝ) : ℝ :=
  let shem_hourly_rate := 2.5 * kem_hourly_rate
  let tiff_hourly_rate := kem_hourly_rate + 3

  let kem_gross := kem_hourly_rate * kem_hours
  let shem_gross := shem_hourly_rate * shem_hours
  let tiff_gross := tiff_hourly_rate * tiff_hours

  let kem_net := kem_gross * (1 - kem_tax_rate)
  let shem_net := shem_gross * (1 - shem_tax_rate) - shem_health_deduction
  let tiff_net := tiff_gross * (1 - tiff_pension_rate) - tiff_transport_deduction

  kem_net + shem_net + tiff_net

/-- Theorem stating that the total net earnings for the specific instance equals 157.5 -/
theorem specific_total_net_earnings_eq_157_5 :
  total_net_earnings 4 6 8 10 0.1 0.05 5 0.03 3 = 157.5 := by
  -- Unfold the definition and simplify
  unfold total_net_earnings
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

#eval total_net_earnings 4 6 8 10 0.1 0.05 5 0.03 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_total_net_earnings_eq_157_5_l352_35229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_second_diagonal_length_l352_35260

/-- Represents a rhombus with given area and one diagonal length -/
structure Rhombus where
  area : ℝ
  diagonal1 : ℝ

/-- Calculates the length of the second diagonal of a rhombus -/
noncomputable def second_diagonal (r : Rhombus) : ℝ :=
  (2 * r.area) / r.diagonal1

theorem rhombus_second_diagonal_length :
  let r : Rhombus := { area := 127.5, diagonal1 := 17 }
  second_diagonal r = 15 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_second_diagonal_length_l352_35260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_of_g_x_coordinate_l352_35271

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x + a

-- Define the function g
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a x - 4

-- Theorem statement
theorem zero_of_g_x_coordinate (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : f a 0 = 3) :
  ∃ x : ℝ, g a x = 0 ∧ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_of_g_x_coordinate_l352_35271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_juniors_average_score_l352_35223

theorem juniors_average_score 
  (total_students : ℕ) 
  (junior_ratio : ℚ) 
  (senior_ratio : ℚ) 
  (overall_average : ℚ) 
  (senior_average : ℚ) 
  (h1 : junior_ratio = 1/5)
  (h2 : senior_ratio = 4/5)
  (h3 : junior_ratio + senior_ratio = 1)
  (h4 : overall_average = 86)
  (h5 : senior_average = 85)
  : 
  let junior_count := (junior_ratio * total_students : ℚ).floor
  let senior_count := (senior_ratio * total_students : ℚ).floor
  let total_score := overall_average * total_students
  let senior_total_score := senior_average * senior_count
  let junior_total_score := total_score - senior_total_score
  junior_total_score / junior_count = 90 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_juniors_average_score_l352_35223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_P_value_at_7_l352_35230

def P (a b c d e f : ℝ) (x : ℂ) : ℂ :=
  (3 * x^4 - 39 * x^3 + a * x^2 + b * x + c) *
  (4 * x^4 - 96 * x^3 + d * x^2 + e * x + f)

theorem P_value_at_7 (a b c d e f : ℝ) :
  (∀ z : ℂ, P a b c d e f z = 0 ↔ z ∈ ({1, 2, 3, 4, 6} : Set ℂ) ∨ (z.im = 0 ∧ z.re = 2 ∨ z.re = 3)) →
  P a b c d e f 7 = 86400 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_P_value_at_7_l352_35230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_calculation_l352_35237

/-- Calculates the simple interest rate given initial principal, final amount, and time period. -/
noncomputable def simple_interest_rate (principal : ℝ) (final_amount : ℝ) (time : ℝ) : ℝ :=
  (final_amount - principal) * 100 / (principal * time)

/-- Theorem stating that the simple interest rate for the given problem is correct. -/
theorem interest_rate_calculation (principal : ℝ) (final_amount : ℝ) (time : ℝ) 
    (h1 : principal = 900)
    (h2 : final_amount = 950)
    (h3 : time = 5) :
  simple_interest_rate principal final_amount time = (950 - 900) * 100 / (900 * 5) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_calculation_l352_35237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_decimal_digits_fraction_l352_35240

/-- The minimum number of digits to the right of the decimal point needed to express
    the fraction 987654321 / (2^24 * 5^6) as a decimal is 24. -/
theorem min_decimal_digits_fraction : ℕ := by
  -- Define the fraction
  let fraction : ℚ := 987654321 / (2^24 * 5^6)
  
  -- Define the minimum number of digits
  let min_digits : ℕ := 24
  
  -- State the property that defines the minimum number of digits
  have property : ∀ (n : ℕ), n < min_digits → ∃ (k : ℕ), fraction * 10^n ≠ k := by sorry
  
  -- State the property that shows min_digits is sufficient
  have sufficiency : ∃ (k : ℕ), fraction * 10^min_digits = k := by sorry
  
  -- Return the result
  exact min_digits


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_decimal_digits_fraction_l352_35240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_prime_upper_bound_m_l352_35286

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.exp x + Real.log x

-- Define the derivative of f
noncomputable def f_prime (x : ℝ) : ℝ := Real.exp x + 1 / x

-- Theorem for the minimum value of f'(x)
theorem min_value_f_prime :
  ∀ x : ℝ, x ≥ 1 → f_prime x ≥ Real.exp 1 + 1 :=
by
  sorry

-- Theorem for the upper bound of m
theorem upper_bound_m :
  ∀ m : ℝ, m > Real.exp 1 + 1 →
  ∃ x : ℝ, x ≥ 1 ∧ f x < Real.exp 1 + m * (x - 1) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_prime_upper_bound_m_l352_35286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeat_divide_by_factors_l352_35292

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  hundreds_nonzero : hundreds ≠ 0
  hundreds_lt_ten : hundreds < 10
  tens_lt_ten : tens < 10
  ones_lt_ten : ones < 10

/-- Converts a ThreeDigitNumber to its numerical value -/
def ThreeDigitNumber.toNat (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

/-- Repeats a three-digit number to form a six-digit number -/
def repeatThreeDigits (n : ThreeDigitNumber) : Nat :=
  1000 * n.toNat + n.toNat

/-- Main theorem: Dividing the repeated three-digit number by 7, 11, and 13 
    yields the original three-digit number -/
theorem repeat_divide_by_factors (n : ThreeDigitNumber) : 
  (((repeatThreeDigits n) / 7) / 11) / 13 = n.toNat := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeat_divide_by_factors_l352_35292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_term_value_l352_35219

def my_sequence (a : ℕ → ℚ) : Prop :=
  a 1 = 1 ∧ ∀ n, a (n + 1) = a n / (2 * a n + 1)

theorem sixth_term_value (a : ℕ → ℚ) (h : my_sequence a) : a 6 = 1 / 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_term_value_l352_35219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_2_lt_xi_le_4_eq_3_16_l352_35246

/-- The probability distribution of the random variable ξ -/
noncomputable def P (k : ℕ) : ℝ := 1 / (2 ^ k)

/-- The probability that 2 < ξ ≤ 4 -/
noncomputable def prob_2_lt_xi_le_4 : ℝ := P 3 + P 4

theorem prob_2_lt_xi_le_4_eq_3_16 : prob_2_lt_xi_le_4 = 3/16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_2_lt_xi_le_4_eq_3_16_l352_35246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_B_hits_once_in_two_shots_at_least_one_hits_l352_35250

-- Define the probabilities for A, B, and C hitting the target
noncomputable def prob_A : ℝ := 1/2
noncomputable def prob_B : ℝ := 1/3
noncomputable def prob_C : ℝ := 1/4

-- Theorem for part (I)
theorem B_hits_once_in_two_shots : 
  (prob_B * (1 - prob_B) + (1 - prob_B) * prob_B) = 4/9 := by sorry

-- Theorem for part (II)
theorem at_least_one_hits :
  1 - ((1 - prob_A) * (1 - prob_B) * (1 - prob_C)) = 3/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_B_hits_once_in_two_shots_at_least_one_hits_l352_35250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_l352_35272

theorem line_inclination_angle :
  let l : Set (ℝ × ℝ) := {(x, y) | Real.sqrt 3 * x + y + 3 = 0}
  let m : ℝ := -Real.sqrt 3
  let α : ℝ := 120 * Real.pi / 180
  (∀ (x y : ℝ), (x, y) ∈ l → y = m * x - 3) →
  m = Real.tan (-α) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_l352_35272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_sum_of_coefficients_l352_35295

-- Define a power function
noncomputable def power_function (k a : ℝ) : ℝ → ℝ := λ x ↦ k * (x ^ a)

-- State the theorem
theorem power_function_sum_of_coefficients :
  ∀ k a : ℝ, power_function k a (1/2) = 1/4 → k + a = 3 :=
by
  intros k a h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_sum_of_coefficients_l352_35295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rice_sale_price_per_kg_l352_35212

/-- Represents the price and weight of a sack of rice, and the profit made from selling it. -/
structure RiceSale where
  weight : ℚ  -- Weight in kilograms (using rational numbers)
  cost : ℚ    -- Cost in dollars (using rational numbers)
  profit : ℚ  -- Profit in dollars (using rational numbers)

/-- Calculates the selling price per kilogram for a sack of rice. -/
def sellingPricePerKg (sale : RiceSale) : ℚ :=
  (sale.cost + sale.profit) / sale.weight

/-- Theorem stating that given the conditions of the problem, 
    the selling price per kilogram is $1.20. -/
theorem rice_sale_price_per_kg (sale : RiceSale) 
    (h_weight : sale.weight = 50)
    (h_cost : sale.cost = 50)
    (h_profit : sale.profit = 10) : 
    sellingPricePerKg sale = 6/5 := by
  sorry

#eval sellingPricePerKg { weight := 50, cost := 50, profit := 10 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rice_sale_price_per_kg_l352_35212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cable_length_theorem_l352_35279

/-- The length of the curve defined by the intersection of a plane and a sphere --/
noncomputable def curve_length (a b c d : ℝ) (r : ℝ) : ℝ :=
  2 * Real.pi * Real.sqrt (r^2 - (d^2 / (a^2 + b^2 + c^2)))

theorem cable_length_theorem (x y z : ℝ) :
  x + y + z = 8 →
  x * y + y * z + x * z = -18 →
  curve_length 1 1 1 8 10 = 4 * Real.pi * Real.sqrt (59 / 3) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cable_length_theorem_l352_35279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_from_parabola_intersection_l352_35202

/-- A point in the 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A line in the 2D plane -/
structure Line2D where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept

/-- A parabola in the 2D plane -/
structure Parabola2D where
  p : ℝ  -- parameter, p > 0

/-- A circle in the 2D plane -/
structure Circle2D where
  center : Point2D
  radius : ℝ

/-- Define membership for Point2D in Circle2D -/
def Point2D.mem (p : Point2D) (c : Circle2D) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

instance : Membership Point2D Circle2D where
  mem := Point2D.mem

/-- Given a line y = x + b intersecting a parabola y² = 2px at points A and B,
    and a circle through A and B intersecting the same parabola at C and D,
    prove that AB is perpendicular to CD -/
theorem perpendicular_lines_from_parabola_intersection
  (line : Line2D)
  (parabola : Parabola2D)
  (A B C D : Point2D)
  (h1 : line.m = 1)
  (h2 : parabola.p > 0)
  (h3 : A.y = A.x + line.b)
  (h4 : B.y = B.x + line.b)
  (h5 : A.y^2 = 2 * parabola.p * A.x)
  (h6 : B.y^2 = 2 * parabola.p * B.x)
  (h7 : C.y^2 = 2 * parabola.p * C.x)
  (h8 : D.y^2 = 2 * parabola.p * D.x)
  (h9 : ∃ (circle : Circle2D), A ∈ circle ∧ B ∈ circle ∧ C ∈ circle ∧ D ∈ circle)
  : (B.y - A.y) * (D.y - C.y) = -(B.x - A.x) * (D.x - C.x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_from_parabola_intersection_l352_35202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_sum_thirteen_l352_35274

def sixSidedDie : Finset ℕ := {1, 2, 3, 4, 5, 6}
def sevenSidedDie : Finset ℕ := {1, 2, 3, 4, 5, 6, 7}

def sumThirteen (roll : ℕ × ℕ) : Bool :=
  roll.1 + roll.2 = 13

theorem probability_sum_thirteen :
  (Finset.filter (fun roll => roll.1 ∈ sixSidedDie ∧ roll.2 ∈ sevenSidedDie ∧ sumThirteen roll)
    (Finset.product sixSidedDie sevenSidedDie)).card / 
    (sixSidedDie.card * sevenSidedDie.card : ℕ) = 1 / 42 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_sum_thirteen_l352_35274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tournament_possible_iff_matches_divisible_by_four_main_theorem_l352_35251

/-- A tournament is valid if the number of participants satisfies n = 8k + 1 for some natural number k. -/
def valid_tournament (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 8 * k + 1

/-- The main theorem: A tournament is possible if and only if the number of participants is of the form 8k + 1. -/
theorem tournament_possible_iff (n : ℕ) :
  (∃ k : ℕ, n = 8 * k + 1) ↔ valid_tournament n :=
by
  -- We define the tournament as valid if it satisfies our condition
  unfold valid_tournament
  -- The proof is trivial as we've defined valid_tournament to be exactly our condition
  rfl

/-- Helper lemma: If n = 8k + 1, then n(n-1) is divisible by 8. -/
lemma divisibility_condition (n k : ℕ) (h : n = 8 * k + 1) :
  8 ∣ (n * (n - 1)) :=
sorry  -- The actual proof would go here

/-- The number of matches in a tournament with n participants is n(n-1)/2. -/
def num_matches (n : ℕ) : ℕ := n * (n - 1) / 2

/-- In a valid tournament, the number of matches is divisible by 4. -/
theorem matches_divisible_by_four {n : ℕ} (h : valid_tournament n) :
  4 ∣ num_matches n :=
sorry  -- The actual proof would go here

/-- Main theorem: A tournament is possible if and only if n = 8k + 1 for some natural k. -/
theorem main_theorem (n : ℕ) :
  (∃ tournament : Type, valid_tournament n) ↔ (∃ k : ℕ, n = 8 * k + 1) :=
sorry  -- The actual proof would go here

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tournament_possible_iff_matches_divisible_by_four_main_theorem_l352_35251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_distance_l352_35267

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hyperbola -/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- Check if a point is on the hyperbola -/
def isOnHyperbola (h : Hyperbola) (p : Point) : Prop :=
  p.x^2 / h.a^2 - p.y^2 / h.b^2 = 1

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem about the distances from a point on a hyperbola to its foci -/
theorem hyperbola_foci_distance (h : Hyperbola) (p f1 f2 : Point) :
  h.a = 3 →
  h.b = 4 →
  isOnHyperbola h p →
  f1.x = -5 ∧ f1.y = 0 →
  f2.x = 5 ∧ f2.y = 0 →
  distance p f1 = 7 →
  distance p f2 = 13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_distance_l352_35267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_penny_income_three_months_l352_35299

/-- Calculates the tax on a given income based on the specified tax brackets. -/
noncomputable def calculate_tax (income : ℝ) : ℝ :=
  if income ≤ 800 then income * 0.1
  else if income ≤ 2000 then 80 + (income - 800) * 0.15
  else 260 + (income - 2000) * 0.2

/-- Calculates the net income after taxes and expenses for a given month. -/
noncomputable def net_income_after_taxes_and_expenses (daily_income : ℝ) (days_worked : ℕ) (expenses : ℝ) : ℝ :=
  let gross_income := daily_income * (days_worked : ℝ)
  let taxes := calculate_tax gross_income
  gross_income - taxes - expenses

/-- Theorem: Penny's total income after taxes and expenses over three months is $344.40. -/
theorem penny_income_three_months :
  let initial_daily_income : ℝ := 10
  let monthly_increase_rate : ℝ := 0.2
  let monthly_expenses : ℝ := 100

  let first_month := net_income_after_taxes_and_expenses initial_daily_income 20 monthly_expenses
  let second_month := net_income_after_taxes_and_expenses (initial_daily_income * (1 + monthly_increase_rate)) 25 monthly_expenses
  let third_month := net_income_after_taxes_and_expenses (initial_daily_income * (1 + monthly_increase_rate) * (1 + monthly_increase_rate)) 15 monthly_expenses

  first_month + second_month + third_month = 344.40 := by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_penny_income_three_months_l352_35299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_mixture_theorem_l352_35254

/-- Represents the composition of a milk mixture --/
structure MilkMixture where
  volume : ℝ
  butterfat_percentage : ℝ

/-- Calculates the total amount of butterfat in a milk mixture --/
noncomputable def total_butterfat (m : MilkMixture) : ℝ :=
  m.volume * m.butterfat_percentage / 100

/-- Combines two milk mixtures --/
noncomputable def combine_mixtures (m1 m2 : MilkMixture) : MilkMixture :=
  { volume := m1.volume + m2.volume,
    butterfat_percentage := (total_butterfat m1 + total_butterfat m2) / (m1.volume + m2.volume) * 100 }

theorem milk_mixture_theorem :
  let initial_milk : MilkMixture := { volume := 8, butterfat_percentage := 35 }
  let added_milk : MilkMixture := { volume := 12, butterfat_percentage := 10 }
  let result := combine_mixtures initial_milk added_milk
  result.butterfat_percentage = 20 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_mixture_theorem_l352_35254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_major_axis_length_l352_35224

-- Define the ellipse
structure Ellipse where
  foci1 : ℝ × ℝ
  foci2 : ℝ × ℝ
  tangent_to_x_axis : Bool
  tangent_to_y_axis : Bool

-- Define our specific ellipse
noncomputable def our_ellipse : Ellipse :=
  { foci1 := (-3 + Real.sqrt 5, 2)
  , foci2 := (-3 - Real.sqrt 5, 2)
  , tangent_to_x_axis := true
  , tangent_to_y_axis := true
  }

-- Helper function (not proven)
def is_major_axis_length (e : Ellipse) (length : ℝ) : Prop := sorry

-- Theorem statement
theorem major_axis_length (e : Ellipse) (h1 : e = our_ellipse) :
  ∃ (length : ℝ), length = 6 ∧ is_major_axis_length e length := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_major_axis_length_l352_35224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_magnitude_l352_35293

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem vector_sum_magnitude (a b : V) 
  (ha : ‖a‖ = 1) 
  (hb : ‖b‖ = 1) 
  (hab : inner a b = -(1/2 : ℝ)) : 
  ‖a + (2 : ℝ) • b‖ = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_magnitude_l352_35293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_truth_l352_35277

-- Define the structure for Triangle
structure Triangle where
  -- You can add more fields if needed
  mk :: -- This allows creating a Triangle without specifying fields

-- Define the functions
noncomputable def area : Triangle → ℝ := sorry
def congruent : Triangle → Triangle → Prop := sorry
def equilateral : Triangle → Prop := sorry
noncomputable def interior_angle : Triangle → ℝ → ℝ := sorry

-- Define the propositions
def prop1 : Prop := ∀ T1 T2 : Triangle, area T1 = area T2 → congruent T1 T2
def prop2 : Prop := ∃ a b : ℝ, a * b ≠ 0 ∧ a ≠ 0
def prop3 : Prop := ∀ T : Triangle, ¬(equilateral T) → ∃ θ : ℝ, interior_angle T θ ≠ 60

-- Define the theorem
theorem proposition_truth : ¬prop1 ∧ prop2 ∧ prop3 := by
  sorry -- The proof is skipped for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_truth_l352_35277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l352_35243

theorem triangle_abc_properties (a b c : ℝ) (A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ A < Real.pi ∧ 0 < B ∧ B < Real.pi ∧ 0 < C ∧ C < Real.pi →
  Real.cos C = a / b - c / (2 * b) →
  b = 2 →
  a - c = 1 →
  B = Real.pi / 3 ∧ (1 / 2) * a * b * Real.sin C = (3 * Real.sqrt 3) / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l352_35243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_condition_implies_t_range_l352_35283

open Real

-- Define the function f
noncomputable def f (t : ℝ) (x : ℝ) : ℝ := (log x + (x - t)^2) / x

-- State the theorem
theorem f_derivative_condition_implies_t_range (t : ℝ) :
  (∀ x ∈ Set.Icc 1 2, (deriv (f t) x) * x + f t x > 0) →
  t < 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_condition_implies_t_range_l352_35283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_theorem_l352_35262

/-- The area of a circle given its radius -/
noncomputable def circle_area (r : ℝ) : ℝ := Real.pi * r^2

/-- The radius of a circle given its area -/
noncomputable def circle_radius (a : ℝ) : ℝ := Real.sqrt (a / Real.pi)

/-- The theorem statement -/
theorem shaded_area_theorem (large_circle_area : ℝ) 
  (h1 : large_circle_area = 100 * Real.pi)
  (h2 : ∃ (r : ℝ), r > 0 ∧ circle_area r = large_circle_area / 2)
  (h3 : ∃ (small_r : ℝ), small_r > 0 ∧ 
    2 * small_r = circle_radius large_circle_area ∧ 
    circle_area small_r * 2 < large_circle_area) :
  large_circle_area / 2 + circle_area (circle_radius large_circle_area / 2) = 75 * Real.pi :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_theorem_l352_35262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_worker_c_daily_wage_l352_35275

-- Define the workers and their working days
def days_a : ℕ := 16
def days_b : ℕ := 9
def days_c : ℕ := 4

-- Define the wage ratio
def wage_ratio_a : ℕ := 3
def wage_ratio_b : ℕ := 4
def wage_ratio_c : ℕ := 5

-- Define the total earnings
def total_earnings : ℕ := 1480

-- Define the daily wage of worker c
def daily_wage_c : ℚ := 355/5  -- 71.15 as a rational number

-- Theorem statement
theorem worker_c_daily_wage :
  ∃ (x : ℚ), 
    (wage_ratio_a : ℚ) * days_a * x + 
    (wage_ratio_b : ℚ) * days_b * x + 
    (wage_ratio_c : ℚ) * days_c * x = total_earnings ∧
    (wage_ratio_c : ℚ) * x = daily_wage_c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_worker_c_daily_wage_l352_35275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_tournament_impossibility_l352_35208

/-- Represents a chess tournament. -/
structure ChessTournament where
  num_players : ℕ
  num_games : ℕ
  games_per_player : Fin num_players → Fin 4
  hgames_per_player : ∀ i, 2 ≤ games_per_player i ∧ games_per_player i ≤ 3

/-- Represents a game between two players. -/
def players_in_game (t : ChessTournament) (game : Fin t.num_games) (player1 player2 : Fin t.num_players) : Prop :=
  sorry

/-- The theorem stating the impossibility of the given scenario. -/
theorem chess_tournament_impossibility (t : ChessTournament) 
  (h_players : t.num_players = 50)
  (h_games : t.num_games = 61) :
  ¬(∀ i j, t.games_per_player i = 3 → t.games_per_player j = 3 → 
    (∃ k, t.games_per_player k = 2 ∧ 
      (∃ game1 game2, game1 ≠ game2 ∧ 
        (players_in_game t game1 i k ∧ players_in_game t game2 j k)))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_tournament_impossibility_l352_35208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_specific_l352_35236

/-- The area of a trapezoid with given side lengths -/
noncomputable def trapezoidArea (a b c d : ℝ) : ℝ :=
  let s := (c + d + (b - a)) / 2
  let h := 2 * Real.sqrt (s * (s - c) * (s - (b - a)) * (s - d)) / (b - a)
  (a + b) * h / 2

theorem trapezoid_area_specific : 
  trapezoidArea 16 44 17 25 = 450 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_specific_l352_35236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grasshopper_jumps_l352_35248

/-- Represents the number of points on the circle. -/
def n : ℕ := 2014

/-- Represents the first jump distance. -/
def jump1 : ℕ := 57

/-- Represents the second jump distance. -/
def jump2 : ℕ := 10

/-- Represents the minimum number of jump2 jumps required. -/
def min_jumps : ℕ := 18

theorem grasshopper_jumps :
  ∀ (start : ℕ),
  ∃ (seq : List ℕ),
  (∀ i : ℕ, i < seq.length → 
    seq[i]! = (start + i * jump1) % n ∨ 
    seq[i]! = (start + (i - 1) * jump1 + jump2) % n) ∧
  (∀ p : ℕ, p < n → p ∈ seq) ∧
  (seq.filter (λ x => ∃ i, i < seq.length ∧ seq[i]! = (start + (i - 1) * jump1 + jump2) % n)).length = min_jumps :=
by
  sorry

#check grasshopper_jumps

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grasshopper_jumps_l352_35248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_growth_equation_l352_35249

/-- Represents the annual average growth rate of education funds -/
def x : Real := Real.mk 0  -- Placeholder value, can be adjusted as needed

/-- Represents the initial investment in millions of yuan -/
def initial_investment : Real := 300

/-- Represents the expected investment in millions of yuan after two years -/
def expected_investment : Real := 500

/-- Theorem stating the relationship between initial investment, growth rate, and expected investment -/
theorem investment_growth_equation :
  initial_investment * (1 + x)^2 = expected_investment :=
by
  sorry  -- Proof is omitted for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_growth_equation_l352_35249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_completion_time_l352_35266

noncomputable section

/-- The number of days A needs to complete the work alone -/
def a_days : ℝ := 20

/-- The number of days B needs to complete the work alone -/
def b_days : ℝ := 12

/-- The number of days A and B work together before A leaves -/
def days_together : ℝ := 3

/-- The work rate of A per day -/
noncomputable def a_rate : ℝ := 1 / a_days

/-- The work rate of B per day -/
noncomputable def b_rate : ℝ := 1 / b_days

/-- The combined work rate of A and B per day -/
noncomputable def combined_rate : ℝ := a_rate + b_rate

/-- The amount of work completed by A and B in 3 days -/
noncomputable def work_completed : ℝ := combined_rate * days_together

/-- The amount of work remaining after A leaves -/
noncomputable def work_remaining : ℝ := 1 - work_completed

/-- The number of days B needs to complete the remaining work -/
noncomputable def days_for_b_to_finish : ℝ := work_remaining / b_rate

theorem b_completion_time : days_for_b_to_finish = 7.2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_completion_time_l352_35266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_sqrt_x_minus_two_l352_35284

-- Define the function as noncomputable due to the use of Real.sqrt
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x - 2)

-- State the theorem
theorem domain_of_sqrt_x_minus_two :
  {x | ∃ y, f x = y} = {x : ℝ | x ≥ 2} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_sqrt_x_minus_two_l352_35284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_pole_time_l352_35285

/-- The time taken for a train to cross a pole -/
noncomputable def train_crossing_time (speed_kmph : ℝ) (length_m : ℝ) : ℝ :=
  length_m / (speed_kmph * 1000 / 3600)

/-- Theorem: The time taken for a train to cross a pole is approximately 5.0004 seconds -/
theorem train_crossing_pole_time :
  let speed_kmph := (18 : ℝ)
  let length_m := 25.002
  abs (train_crossing_time speed_kmph length_m - 5.0004) < 0.00001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_pole_time_l352_35285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_3_15_l352_35241

/-- The angle in degrees that the hour hand moves per hour -/
def hourHandAnglePerHour : ℚ := 30

/-- The angle in degrees that the minute hand moves per minute -/
def minuteHandAnglePerMinute : ℚ := 6

/-- The number of hours past 12 o'clock -/
def hours : ℚ := 3

/-- The number of minutes past the hour -/
def minutes : ℚ := 15

/-- The angle of the hour hand at 3:15 -/
noncomputable def hourHandAngle : ℚ := hours * hourHandAnglePerHour + (minutes / 60) * hourHandAnglePerHour

/-- The angle of the minute hand at 3:15 -/
def minuteHandAngle : ℚ := minutes * minuteHandAnglePerMinute

/-- The smaller angle between the hour hand and the minute hand at 3:15 -/
noncomputable def smallerAngle : ℚ := abs (hourHandAngle - minuteHandAngle)

theorem clock_angle_at_3_15 : smallerAngle = 15/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_3_15_l352_35241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_study_game_relationship_l352_35298

/-- Represents the relationship between study time and video games played on a given day -/
structure StudyGameDay where
  study_time : ℝ
  games_played : ℝ

/-- The constant of proportionality between study time and games played -/
def proportionality_constant (day : StudyGameDay) : ℝ :=
  day.study_time * day.games_played

theorem study_game_relationship 
  (day1 day2 day3 : StudyGameDay)
  (h1 : day1.study_time = 4 ∧ day1.games_played = 3)
  (h2 : day2.study_time = 6)
  (h3 : day3.games_played = 6)
  (h_const : proportionality_constant day1 = proportionality_constant day2 ∧ 
             proportionality_constant day1 = proportionality_constant day3) :
  day2.games_played = 2 ∧ day3.study_time = 2 := by
  sorry

#check study_game_relationship

end NUMINAMATH_CALUDE_ERRORFEEDBACK_study_game_relationship_l352_35298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_between_roots_l352_35207

theorem count_integers_between_roots : ∃ (n : ℕ), n = (List.filter (λ k => Real.sqrt 50 < k ∧ k < Real.sqrt 200 + 1) (List.range 201)).length ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_between_roots_l352_35207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_a_2016_eq_zero_l352_35287

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the sequence a_n
def a : ℕ → ℝ := sorry

-- Define S_n (sum of first n terms of a_n)
def S : ℕ → ℝ := sorry

-- Axioms based on the given conditions
axiom f_odd (x : ℝ) : f (-x) = -f x
axiom f_periodic (x : ℝ) : f (x + 1) = f (x - 1)
axiom S_def (n : ℕ) : S n = 2 * a n + 2

-- The theorem to prove
theorem f_a_2016_eq_zero : f (a 2016) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_a_2016_eq_zero_l352_35287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mabel_distance_from_school_l352_35258

/-- Mabel's distance from school in steps -/
def M : ℕ := sorry

/-- Helen's distance from school in steps -/
def H : ℕ := sorry

/-- The total number of steps Mabel walks to visit Helen -/
def total_steps : ℕ := 7875

theorem mabel_distance_from_school :
  H = (3 * M) / 4 →
  total_steps = M + H →
  M = 4500 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mabel_distance_from_school_l352_35258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_position_after_rotation_l352_35244

/-- Represents a circular pathway -/
structure Pathway where
  radius : ℝ

/-- Represents a bicycle wheel -/
structure Wheel where
  radius : ℝ

/-- Represents the position on the circular pathway -/
inductive Position
  | Twelve
  | Six

/-- Calculates the angle traveled on the pathway for one full rotation of the wheel -/
noncomputable def angle_traveled (p : Pathway) (w : Wheel) : ℝ :=
  2 * Real.pi * w.radius / p.radius

theorem wheel_position_after_rotation (p : Pathway) (w : Wheel) 
  (h1 : p.radius = 30)
  (h2 : w.radius = 15)
  (h3 : angle_traveled p w = Real.pi) :
  Position.Six = 
    (if angle_traveled p w ≥ Real.pi then Position.Six else Position.Twelve) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_position_after_rotation_l352_35244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l352_35232

open Real

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x - m * log x - (m - 1) / x

noncomputable def g (x : ℝ) : ℝ := (1/2) * x^2 + exp x - x * exp x

theorem function_properties (m : ℝ) :
  (m ≤ 2) →
  (∀ x ∈ Set.Icc 1 (exp 1), f m x ≥ 2 - m) ∧
  (∀ x ∈ Set.Icc 1 (exp 1), f m x ≥ f m 1) ∧
  (∀ x₁ ∈ Set.Icc (exp 1) (exp 2), ∀ x₂ ∈ Set.Icc (-2) 0,
    f m x₁ ≤ g x₂ ↔ ((exp 2 - exp 1 + 1) / (exp 1 + 1) ≤ m ∧ m ≤ 2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l352_35232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_measure_max_altitude_max_altitude_value_l352_35239

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  h : ℝ -- altitude on edge AC

-- Define the conditions
def triangle_condition (t : Triangle) : Prop :=
  (t.a - t.b) * (Real.sin t.A + Real.sin t.B) = (t.a - t.c) * Real.sin t.C

-- Theorem 1
theorem angle_B_measure (t : Triangle) (h : triangle_condition t) : t.B = π / 3 :=
sorry

-- Theorem 2
theorem max_altitude (t : Triangle) (h : triangle_condition t) (h2 : t.b = 3) :
  t.h ≤ 3 * Real.sqrt 3 / 2 :=
sorry

-- Theorem for the maximum value of h
theorem max_altitude_value (t : Triangle) (h : triangle_condition t) (h2 : t.b = 3) :
  ∃ (t' : Triangle), t'.b = 3 ∧ triangle_condition t' ∧ t'.h = 3 * Real.sqrt 3 / 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_measure_max_altitude_max_altitude_value_l352_35239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_card_sorting_theorem_l352_35268

/-- Represents a permutation of cards in cells -/
def Arrangement (n : ℕ) := Fin (n + 1) → Fin (n + 1)

/-- The allowed move in the card sorting problem -/
def move (n : ℕ) (a : Arrangement n) : Arrangement n := sorry

/-- Predicate to check if an arrangement is sorted -/
def is_sorted (n : ℕ) (a : Arrangement n) : Prop := 
  ∀ i : Fin (n + 1), a i = i

/-- The number of moves required to sort an arrangement -/
def moves_to_sort (n : ℕ) (a : Arrangement n) : ℕ := sorry

/-- The maximum number of moves required for any arrangement -/
noncomputable def max_moves (n : ℕ) : ℕ := 
  Finset.sup (Finset.univ : Finset (Fin (n + 1) → Fin (n + 1))) (moves_to_sort n)

/-- The unique arrangement requiring the maximum number of moves -/
def worst_arrangement (n : ℕ) : Arrangement n := sorry

/-- The main theorem about card sorting -/
theorem card_sorting_theorem (n : ℕ) : 
  max_moves n = 2^n - 1 ∧ 
  moves_to_sort n (worst_arrangement n) = 2^n - 1 ∧
  ∀ a : Arrangement n, moves_to_sort n a = 2^n - 1 → a = worst_arrangement n :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_card_sorting_theorem_l352_35268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_chord_properties_l352_35273

-- Define the circles C₁ and C₂
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 4
def C₂ (x y m n : ℝ) : Prop := (x - m)^2 + (y - n)^2 = 4

-- Define the common chord AB
noncomputable def common_chord_length : ℝ := 2 * Real.sqrt 3

-- Theorem statement
theorem common_chord_properties
  (m n : ℝ)
  (h_common_chord : ∃ (x y : ℝ), C₁ x y ∧ C₂ x y m n)
  (h_chord_length : ∃ (a b c d : ℝ), 
    C₁ a b ∧ C₁ c d ∧ C₂ a b m n ∧ C₂ c d m n ∧
    (a - c)^2 + (b - d)^2 = common_chord_length^2) :
  (m^2 + n^2 = 4) ∧
  (∃ (x y : ℝ), m*x + n*y = 2 ∧ C₁ x y ∧ C₂ x y m n) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_chord_properties_l352_35273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_modulus_of_z_l352_35226

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define z as given in the problem
noncomputable def z : ℂ := i / (1 - i)

-- State the theorem to be proved
theorem inverse_modulus_of_z : 1 / Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_modulus_of_z_l352_35226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_value_range_of_difference_l352_35228

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the conditions
def satisfies_condition (t : Triangle) : Prop :=
  Real.cos (3 * t.A) - Real.cos (2 * t.B) = 2 * Real.cos (Real.pi / 6 - t.A) * Real.cos (Real.pi / 6 + t.A)

def side_condition (t : Triangle) : Prop :=
  t.b = Real.sqrt 3 ∧ t.b ≤ t.a

-- Theorem 1
theorem angle_B_value (t : Triangle) (h : satisfies_condition t) : t.B = Real.pi / 3 :=
sorry

-- Theorem 2
theorem range_of_difference (t : Triangle) (h1 : side_condition t) (h2 : t.B = Real.pi / 3) :
  t.a - (1 / 2) * t.c ∈ Set.Icc (Real.sqrt 3 / 2) (Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_value_range_of_difference_l352_35228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_l352_35257

noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 1 then x * (1 - x)
  else if 1 < x ∧ x ≤ 2 then Real.sin (Real.pi * x)
  else 0  -- This case is arbitrary since we don't know f outside [0, 2]

axiom f_property (x : ℝ) : f (x + 4) = f x

axiom f_odd (x : ℝ) : f (-x) = -f x

theorem f_value : f (29/4) + f (17/6) = -11/16 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_l352_35257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_minimum_distance_l352_35290

-- Define the circle
def my_circle (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 4

-- Define the line that point A is on
def line_A (x y : ℝ) : Prop := x - y - 2 = 0

-- Define point A
def point_A : ℝ × ℝ := (1, -1)

-- Define the line passing through tangency points
def tangent_line (x y : ℝ) : Prop := x - 3*y + 2 = 0

theorem tangent_line_equation :
  ∀ (T1 T2 : ℝ × ℝ),
  my_circle T1.1 T1.2 →
  my_circle T2.1 T2.2 →
  line_A point_A.1 point_A.2 →
  (∃ (t : ℝ), T1 = (1 - t, -1 - t) ∧ T2 = (1 + t, -1 + t)) →
  tangent_line T1.1 T1.2 ∧ tangent_line T2.1 T2.2 :=
by
  sorry

theorem minimum_distance :
  ∀ (A : ℝ × ℝ),
  line_A A.1 A.2 →
  (∃ (T : ℝ × ℝ), my_circle T.1 T.2 ∧
    ∀ (P : ℝ × ℝ), my_circle P.1 P.2 →
      (A.1 - T.1)^2 + (A.2 - T.2)^2 ≤ (A.1 - P.1)^2 + (A.2 - P.2)^2) →
  ∃ (T : ℝ × ℝ), my_circle T.1 T.2 ∧
    (A.1 - T.1)^2 + (A.2 - T.2)^2 = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_minimum_distance_l352_35290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_area_l352_35247

-- Define the lines
noncomputable def line1 (x : ℝ) : ℝ := 2 * x - 5
noncomputable def line2 (x : ℝ) : ℝ := x - 1

-- Define the intersection point P
noncomputable def P : ℝ × ℝ := (4, 3)

-- Define points A and B where the lines intersect the x-axis
noncomputable def A : ℝ × ℝ := (5/2, 0)
noncomputable def B : ℝ × ℝ := (1, 0)

-- Theorem to prove
theorem intersection_and_area :
  (line1 P.1 = P.2 ∧ line2 P.1 = P.2) ∧  -- P is on both lines
  (line1 A.1 = A.2 ∧ A.2 = 0) ∧          -- A is on line1 and x-axis
  (line2 B.1 = B.2 ∧ B.2 = 0) ∧          -- B is on line2 and x-axis
  (1/2 * |A.1 - B.1| * P.2 = 9/4)        -- Area of triangle ABP
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_area_l352_35247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_invertible_sum_mod_16_l352_35225

def is_invertible_mod_16 (n : ℕ) : Prop := 
  ∃ m : ℕ, (n * m) % 16 = 1

theorem invertible_sum_mod_16 (a b c d e : ℕ) 
  (ha : a < 16 ∧ is_invertible_mod_16 a) 
  (hb : b < 16 ∧ is_invertible_mod_16 b)
  (hc : c < 16 ∧ is_invertible_mod_16 c)
  (hd : d < 16 ∧ is_invertible_mod_16 d)
  (he : e < 16 ∧ is_invertible_mod_16 e)
  (hdistinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e) :
  ∃ (a' b' c' d' e' : ℕ), 
    (a * a') % 16 = 1 ∧
    (b * b') % 16 = 1 ∧
    (c * c') % 16 = 1 ∧
    (d * d') % 16 = 1 ∧
    (e * e') % 16 = 1 ∧
    (a' + b' + c' + d' + e') % 16 = 9 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_invertible_sum_mod_16_l352_35225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_correct_answers_l352_35276

theorem percentage_correct_answers (total_questions : ℕ) (correct_answers : ℕ) 
  (h1 : total_questions = 84) (h2 : correct_answers = 58) :
  Int.floor ((correct_answers : ℝ) / (total_questions : ℝ) * 100) = 69 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_correct_answers_l352_35276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_statue_scale_factor_l352_35227

/-- The height of the Statue of Liberty in feet -/
noncomputable def statue_height : ℝ := 305

/-- The height of the scale model in inches -/
noncomputable def model_height : ℝ := 10

/-- The scale factor between the statue and its model -/
noncomputable def scale_factor : ℝ := statue_height / model_height

theorem statue_scale_factor :
  scale_factor = 30.5 := by
  -- Unfold the definitions
  unfold scale_factor statue_height model_height
  -- Perform the division
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_statue_scale_factor_l352_35227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_power_magnitude_l352_35264

theorem complex_power_magnitude (z : ℂ) (n : ℕ) (h : z = Complex.ofReal (1/2) + Complex.I * Complex.ofReal (Real.sqrt 3 / 2)) :
  Complex.abs (z^n) = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_power_magnitude_l352_35264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_salary_is_25000_l352_35200

structure Position where
  title : String
  count : Nat
  salary : Nat

def company_data : List Position := [
  ⟨"President", 1, 130000⟩,
  ⟨"Vice-President", 6, 95000⟩,
  ⟨"Director", 12, 80000⟩,
  ⟨"Associate Director", 10, 55000⟩,
  ⟨"Administrative Specialist", 30, 25000⟩
]

def total_employees : Nat := (company_data.map (·.count)).sum

theorem median_salary_is_25000 :
  let sorted_salaries := (company_data.map (λ p => List.replicate p.count p.salary)).join.toArray.qsort (· ≤ ·)
  sorted_salaries[(total_employees - 1) / 2]! = 25000 := by
  sorry

#eval total_employees

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_salary_is_25000_l352_35200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l352_35216

open Real

noncomputable def area_triangle (A B C : ℝ) : ℝ := 
  1 / 2 * (4 / (2 - Real.sqrt 2)) * (4 / (2 - Real.sqrt 2)) * sin B

theorem triangle_properties (A B C : ℝ) (b : ℝ) 
  (h1 : sin A = sin B * cos C + sin C * cos B)
  (h2 : sin B = cos B)
  (h3 : b = 2) :
  B = π / 4 ∧ area_triangle A B C = Real.sqrt 2 + 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l352_35216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_range_l352_35259

theorem inequality_solution_range (a : ℝ) : 
  (∀ x : ℤ, x ∈ ({1, 2, 3, 4} : Set ℤ) → 0 ≤ a * ↑x + 5 ∧ a * ↑x + 5 ≤ 4) ↔ 
  -5/4 ≤ a ∧ a < -1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_range_l352_35259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extra_postage_count_l352_35297

-- Define the envelope structure
structure Envelope where
  length : ℚ
  height : ℚ

-- Define the condition for extra postage
def needsExtraPostage (e : Envelope) : Bool :=
  e.length / e.height < 1.5 ∨ e.length / e.height > 3

-- Define the list of envelopes
def envelopes : List Envelope := [
  { length := 7, height := 5 },
  { length := 10, height := 2 },
  { length := 5, height := 5 },
  { length := 12, height := 3 }
]

-- Theorem statement
theorem extra_postage_count : 
  (envelopes.filter needsExtraPostage).length = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extra_postage_count_l352_35297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_32_l352_35265

/-- A point in a 2D coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- The area of a triangle given its three vertices -/
noncomputable def triangleArea (a b c : Point) : ℝ :=
  (1/2) * abs ((b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y))

theorem triangle_area_is_32 :
  let p1 : Point := ⟨-2, 6⟩
  let p2 : Point := ⟨-6, 2⟩
  let p3 : Point := ⟨0, 8⟩
  triangleArea p1 p2 p3 = 32 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_32_l352_35265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l352_35235

theorem circle_equation (M : Set (ℝ × ℝ)) (center : ℝ × ℝ) (r : ℝ) :
  (∃ a : ℝ, center = (a, 0) ∧ a < 0) →
  r = Real.sqrt 5 →
  (∀ p ∈ M, (p.1 - center.1)^2 + p.2^2 = r^2) →
  (∀ p ∈ M, p.1 + 2*p.2 ≥ 0) →
  (∃ q ∈ M, q.1 + 2*q.2 = 0) →
  (∀ p ∈ M, (p.1 + 5)^2 + p.2^2 = 5) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l352_35235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_output_range_for_max_cost_min_average_cost_output_lowest_average_cost_l352_35256

-- Define the cost function
noncomputable def cost_function (x : ℝ) : ℝ := (1/10) * x^2 - 30 * x + 4000

-- Define the valid range for x
def valid_range (x : ℝ) : Prop := 150 ≤ x ∧ x ≤ 250

-- Theorem 1: Range of x when cost doesn't exceed 20 million yuan
theorem output_range_for_max_cost :
  {x : ℝ | valid_range x ∧ cost_function x ≤ 2000} = {x : ℝ | 150 ≤ x ∧ x ≤ 200} :=
by sorry

-- Theorem 2: Output level for lowest average cost per ton
theorem min_average_cost_output :
  ∃ (x : ℝ), valid_range x ∧
  (∀ (y : ℝ), valid_range y → cost_function x / x ≤ cost_function y / y) ∧
  x = 200 :=
by sorry

-- Theorem 3: Lowest average cost per ton
theorem lowest_average_cost :
  ∃ (x : ℝ), valid_range x ∧
  (∀ (y : ℝ), valid_range y → cost_function x / x ≤ cost_function y / y) ∧
  cost_function x / x = 10 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_output_range_for_max_cost_min_average_cost_output_lowest_average_cost_l352_35256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_white_area_approx_l352_35291

/-- Represents the dimensions of the rectangular board --/
structure BoardDimensions where
  width : ℕ
  height : ℕ

/-- Represents the dimensions of the letter P --/
structure PDimensions where
  verticalHeight : ℕ
  horizontalWidth : ℕ

/-- Represents the dimensions of the letter O --/
structure ODimensions where
  diameter : ℕ

/-- Represents the dimensions of the letter S --/
structure SDimensions where
  horizontalBarLength : ℕ
  horizontalBarCount : ℕ
  verticalBarLength : ℕ
  verticalBarCount : ℕ

/-- Represents the dimensions of the letter T --/
structure TDimensions where
  horizontalWidth : ℕ
  verticalHeight : ℕ

/-- Calculates the white area on the board after writing "POST" --/
noncomputable def whiteArea (board : BoardDimensions) (p : PDimensions) (o : ODimensions) 
               (s : SDimensions) (t : TDimensions) : ℝ :=
  let totalArea := (board.width : ℝ) * (board.height : ℝ)
  let pArea := (p.verticalHeight : ℝ) + (p.horizontalWidth : ℝ)
  let oArea := Real.pi * ((o.diameter : ℝ) / 2) ^ 2
  let sArea := (s.horizontalBarLength * s.horizontalBarCount : ℝ) + (s.verticalBarLength * s.verticalBarCount : ℝ)
  let tArea := (t.horizontalWidth : ℝ) + (t.verticalHeight : ℝ)
  totalArea - (pArea + oArea + sArea + tArea)

/-- The main theorem stating the white area is approximately 43.365 --/
theorem white_area_approx (board : BoardDimensions) (p : PDimensions) (o : ODimensions) 
                          (s : SDimensions) (t : TDimensions) : 
  board.width = 18 ∧ board.height = 6 ∧
  p.verticalHeight = 6 ∧ p.horizontalWidth = 4 ∧
  o.diameter = 5 ∧
  s.horizontalBarLength = 3 ∧ s.horizontalBarCount = 3 ∧ 
  s.verticalBarLength = 1 ∧ s.verticalBarCount = 2 ∧
  t.horizontalWidth = 18 ∧ t.verticalHeight = 6 →
  ∃ (ε : ℝ), abs (whiteArea board p o s t - 43.365) < ε ∧ ε < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_white_area_approx_l352_35291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_fractal_fourth_term_l352_35221

def h_fractal_sequence : ℕ → ℕ
  | 0 => 1
  | n + 1 => 2 * h_fractal_sequence n + 1

theorem h_fractal_fourth_term : h_fractal_sequence 3 = 15 := by
  rfl

#eval h_fractal_sequence 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_fractal_fourth_term_l352_35221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_axis_of_symmetry_l352_35206

/-- Given a parabola y = ax² + bx + c where a ≠ 0, if a + b + c = 0 and 9a - 3b + c = 0, 
    then its axis of symmetry is x = -1 -/
theorem parabola_axis_of_symmetry (a b c : ℝ) (ha : a ≠ 0) 
  (h1 : a + b + c = 0) (h2 : 9*a - 3*b + c = 0) : 
  ∀ x : ℝ, a * (x + 1)^2 + b * (x + 1) + c = a * ((-x - 1) + 1)^2 + b * ((-x - 1) + 1) + c := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_axis_of_symmetry_l352_35206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sampling_is_systematic_l352_35296

/-- Represents a sampling method -/
inductive SamplingMethod
  | Stratified
  | Lottery
  | Systematic
  | SimpleRandom

/-- Represents a class of students -/
structure StudentClass :=
  (students : Finset Nat)
  (size : students.card = 50)
  (numbering : ∀ n, n ∈ students ↔ 1 ≤ n ∧ n ≤ 50)

/-- Represents a grade consisting of multiple classes -/
structure Grade :=
  (classes : Finset StudentClass)
  (size : classes.card = 20)

/-- Represents the sampling process -/
def sampling (g : Grade) : Finset Nat :=
  g.classes.image (λ _ => 16)

/-- Theorem stating that the sampling method is systematic -/
theorem sampling_is_systematic (g : Grade) : 
  sampling g ≠ ∅ → SamplingMethod.Systematic = 
    (let sample := sampling g
     let method := 
       if sample.card = g.classes.card ∧ 
          (∀ c ∈ g.classes, (16 : Nat) ∈ c.students)
       then SamplingMethod.Systematic
       else SamplingMethod.SimpleRandom
     method) :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sampling_is_systematic_l352_35296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_is_two_l352_35217

/-- Line L: x + 2y + 1 = 0 -/
def L : Set (ℝ × ℝ) := {p | p.1 + 2 * p.2 + 1 = 0}

/-- Circle C: (x+2)² + (y+2)² = 1 -/
def C : Set (ℝ × ℝ) := {p | (p.1 + 2)^2 + (p.2 + 2)^2 = 1}

/-- The center of circle C -/
def center : ℝ × ℝ := (-2, -2)

/-- The radius of circle C -/
def radius : ℝ := 1

/-- Point P is on line L -/
noncomputable def P : {p : ℝ × ℝ // p ∈ L} := sorry

/-- Point T is on circle C -/
noncomputable def T : {t : ℝ × ℝ // t ∈ C} := sorry

/-- PT is tangent to C -/
def is_tangent (P : ℝ × ℝ) (T : ℝ × ℝ) : Prop := sorry

/-- The distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem min_distance_is_two :
  ∀ (P : {p : ℝ × ℝ // p ∈ L}) (T : {t : ℝ × ℝ // t ∈ C}),
    is_tangent P.val T.val →
    ∃ (min_dist : ℝ), min_dist = 2 ∧
      ∀ (P' : {p : ℝ × ℝ // p ∈ L}) (T' : {t : ℝ × ℝ // t ∈ C}),
        is_tangent P'.val T'.val →
        distance P'.val T'.val ≥ min_dist :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_is_two_l352_35217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_digit_for_divisibility_by_6_l352_35255

def is_divisible_by_6 (n : Nat) : Prop := n % 6 = 0

def digit_sum (n : Nat) : Nat :=
  (n.digits 10).sum

theorem unique_digit_for_divisibility_by_6 :
  ∃! d : Nat, d < 10 ∧ is_divisible_by_6 (142850 + d) :=
by
  -- The proof would go here
  sorry

-- Remove the #eval line as it's not necessary for building
-- and may cause issues if the theorem is not fully proven

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_digit_for_divisibility_by_6_l352_35255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_distance_time_theorem_l352_35269

/-- The time (in hours) for two cars to be D miles apart -/
noncomputable def time_to_distance (v1 v2 : ℝ) (D : ℝ) : ℝ :=
  D / (v2 - v1)

theorem car_distance_time_theorem (v1 v2 D : ℝ) 
  (h1 : v1 = 50) 
  (h2 : v2 = 60) 
  (h3 : (v2 - v1) * 3 = 30) : 
  time_to_distance v1 v2 D = D / 10 := by
  sorry

#check car_distance_time_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_distance_time_theorem_l352_35269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_perpendicular_area_l352_35209

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 49 + y^2 / 24 = 1

-- Define the foci
def F1 : ℝ × ℝ := (-5, 0)
def F2 : ℝ × ℝ := (5, 0)

-- Define a point on the ellipse
def point_on_ellipse (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  ellipse x y

-- Define perpendicular lines
def perpendicular_lines (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  (y / (x + 5)) * (y / (x - 5)) = -1

-- Define the area of a triangle
noncomputable def area_triangle (A B C : ℝ × ℝ) : ℝ :=
  let (x1, y1) := A
  let (x2, y2) := B
  let (x3, y3) := C
  (1/2) * abs ((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)))

-- Theorem statement
theorem ellipse_perpendicular_area :
  ∀ P : ℝ × ℝ,
  point_on_ellipse P →
  perpendicular_lines P →
  area_triangle P F1 F2 = 24 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_perpendicular_area_l352_35209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_night_market_revenue_l352_35211

-- Define the price function
noncomputable def P (k : ℝ) (x : ℝ) : ℝ := 10 + k / x

-- Define the sales volume function
noncomputable def Q : ℝ → ℝ := sorry

-- Define the revenue function
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := P k x * Q x

-- Theorem statement
theorem night_market_revenue 
  (k : ℝ) 
  (hk : k > 0)
  (hQ10 : Q 10 = 50)
  (hQ15 : Q 15 = 55)
  (hQ20 : Q 20 = 60)
  (hQ25 : Q 25 = 55)
  (hQ30 : Q 30 = 50)
  (hRevenue10 : Q 10 * P k 10 = 505) :
  k = 1 ∧ 
  (∀ x, x ≥ 10 ∧ x ≤ 30 → Q x = -|x - 20| + 60) ∧
  (∀ x, x ≥ 1 ∧ x ≤ 30 → f k x ≥ 441) ∧
  (∃ x, x ≥ 1 ∧ x ≤ 30 ∧ f k x = 441) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_night_market_revenue_l352_35211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l352_35214

/-- The constant term of the expansion of (2x+1)(1-1/x)^5 is -9 -/
theorem constant_term_expansion : ∃ (f : ℝ → ℝ), 
  (∀ x : ℝ, x ≠ 0 → f x = (2*x+1)*(1-1/x)^5) ∧ 
  (∃ c : ℝ, c = -9 ∧ ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x| ∧ |x| < δ → |f x - c| < ε) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l352_35214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_pyramid_cross_section_area_l352_35238

noncomputable def cross_section_area (pyramid : ℝ → ℝ → ℝ → ℝ) : ℝ := sorry

noncomputable def regular_truncated_quadrangular_pyramid (H α β : ℝ) : ℝ → ℝ → ℝ → ℝ := sorry

theorem truncated_pyramid_cross_section_area 
  (H α β : ℝ) 
  (h_pos : H > 0) 
  (h_α_pos : 0 < α ∧ α < π / 2) 
  (h_β_pos : 0 < β ∧ β < π / 2) :
  let area := H^2 * Real.sin (α + β) * Real.sin (α - β) / 
    (Real.sin α^2 * Real.sin β * Real.sin (2 * β))
  ∃ (S : ℝ), S = area ∧ 
    S = cross_section_area (regular_truncated_quadrangular_pyramid H α β) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_pyramid_cross_section_area_l352_35238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l352_35245

-- Define the two equations
def equation1 (x y : ℝ) : Prop := x^2 + y = 11
def equation2 (x y : ℝ) : Prop := x + y = 11

-- Define the distance function between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- Theorem statement
theorem intersection_distance :
  ∃ x1 y1 x2 y2 : ℝ,
    equation1 x1 y1 ∧ equation2 x1 y1 ∧
    equation1 x2 y2 ∧ equation2 x2 y2 ∧
    x1 ≠ x2 ∧
    distance x1 y1 x2 y2 = Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l352_35245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_min_area_tangent_line_l352_35282

-- Define the circle equation
def circle_eq (x y m : ℝ) : Prop :=
  x^2 - 2*x + y^2 - 2*m*y + 2*m - 1 = 0

-- Define the line equation
def line_eq (x y b : ℝ) : Prop :=
  y = x + b

-- Define the tangency condition
def is_tangent (m b : ℝ) : Prop :=
  ∃ x y, circle_eq x y m ∧ line_eq x y b

-- State the theorem
theorem circle_min_area_tangent_line (m b : ℝ) :
  (∀ m' : ℝ, (∃ x y, circle_eq x y m') → (∃ x y, circle_eq x y m)) →  -- m minimizes the circle's area
  is_tangent m b →
  b = Real.sqrt 2 ∨ b = -Real.sqrt 2 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_min_area_tangent_line_l352_35282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_l352_35280

-- Define the type for a 2D point
def Point := ℝ × ℝ

-- Define the ellipse type
structure Ellipse where
  endpoint1 : Point
  endpoint2 : Point
  endpoint3 : Point

-- Define the distance between two points
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the function to calculate the distance between foci
noncomputable def distanceBetweenFoci (e : Ellipse) : ℝ :=
  -- The actual calculation would go here
  sorry

-- State the theorem
theorem ellipse_foci_distance :
  ∀ e : Ellipse,
  e.endpoint1 = (1, 3) ∧ e.endpoint2 = (6, -1) ∧ e.endpoint3 = (11, 3) →
  distanceBetweenFoci e = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_l352_35280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l352_35210

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := 
  (Real.cos x ^ 3 + 8 * Real.cos x ^ 2 - 5 * Real.cos x + 4 * Real.sin x ^ 2 - 11) / (Real.cos x - 2)

-- Theorem statement
theorem f_range :
  ∀ y ∈ Set.range f, -0.5 ≤ y ∧ y ≤ 11.5 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l352_35210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tens_digit_of_2023_pow_2024_minus_2025_squared_l352_35261

theorem tens_digit_of_2023_pow_2024_minus_2025_squared :
  (2023^2024 - 2025^2) / 10 % 10 = 1 := by
  -- Define the expression
  let expression := 2023^2024 - 2025^2

  -- Define a function to get the tens digit
  let tens_digit (n : ℕ) : ℕ := (n / 10) % 10

  -- The proof
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tens_digit_of_2023_pow_2024_minus_2025_squared_l352_35261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_equation_l352_35242

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 6*x = 0

-- Define the point P
def P : ℝ × ℝ := (1, 1)

-- Define the property that P is the midpoint of chord MN
def is_midpoint (P M N : ℝ × ℝ) : Prop :=
  P.1 = (M.1 + N.1) / 2 ∧ P.2 = (M.2 + N.2) / 2

-- Define the line equation
def line_equation (x y : ℝ) : Prop := 2*x - y - 1 = 0

-- Theorem statement
theorem chord_equation :
  ∀ M N : ℝ × ℝ,
  (circle_eq M.1 M.2) →
  (circle_eq N.1 N.2) →
  (is_midpoint P M N) →
  (∀ x y : ℝ, line_equation x y ↔ ∃ t : ℝ, (x, y) = (1 - t, 1 - 2*t) ∨ (x, y) = (1 + t, 1 + 2*t)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_equation_l352_35242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_properties_l352_35263

noncomputable def f (ω φ x : ℝ) : ℝ := 2 * Real.sin (ω * x + φ)

theorem sine_function_properties (ω φ : ℝ) 
  (h_ω : ω > 0) (h_φ : |φ| < π/2) :
  (f ω φ 0 = 1 → φ = π/6) ∧
  (∃ x : ℝ, f ω φ (x + 2) - f ω φ x = 4 → ω ≥ π/2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_properties_l352_35263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_detergent_concentration_part1_detergent_concentration_part2_l352_35252

-- Define the piecewise function f(x)
noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 5 then 16 / (9 - x) - 1
  else if 5 < x ∧ x ≤ 16 then 11 - 2 / 45 * x^2
  else 0

-- Theorem for part (1)
theorem detergent_concentration_part1 :
  ∃ k : ℝ, 1 ≤ k ∧ k ≤ 4 ∧ k * (f 3) = 4 ∧ k = 12/5 := by
  sorry

-- Theorem for part (2)
theorem detergent_concentration_part2 :
  ∃ t : ℝ, t = 14 ∧ 
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ t → 4 * f x ≥ 4) ∧
  (∀ x : ℝ, x > t → 4 * f x < 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_detergent_concentration_part1_detergent_concentration_part2_l352_35252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_abc_l352_35203

noncomputable def a : ℝ := (5 : ℝ) ^ (1.2 : ℝ)
noncomputable def b : ℝ := Real.log 6 / Real.log 0.2
noncomputable def c : ℝ := (2 : ℝ) ^ (1.2 : ℝ)

theorem order_of_abc : a > c ∧ c > b := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_abc_l352_35203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equality_l352_35253

-- Define the function f
noncomputable def f : ℝ → ℝ := λ x ↦ (1/4) * x^2 - 1/4

-- State the theorem
theorem function_equality : ∀ x : ℝ, f (2*x + 1) = x^2 + x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equality_l352_35253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_balanced_partition_l352_35213

theorem balanced_partition (weights : List ℕ) : 
  weights.length = 10 → 
  weights.sum = 20 → 
  (∀ w ∈ weights, 0 < w ∧ w ≤ 10) →
  ∃ subset : List ℕ, subset.toFinset ⊆ weights.toFinset ∧ subset.sum = 10 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_balanced_partition_l352_35213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_t_mobile_base_cost_l352_35205

/-- The cost of the first two lines at T-Mobile -/
def T : ℕ := sorry

/-- The number of cell phone lines needed by Moore's family -/
def total_lines : ℕ := 5

/-- The cost per additional line at T-Mobile -/
def t_mobile_additional : ℕ := 16

/-- The cost for the first two lines at M-Mobile -/
def m_mobile_base : ℕ := 45

/-- The cost per additional line at M-Mobile -/
def m_mobile_additional : ℕ := 14

/-- The price difference between T-Mobile and M-Mobile plans -/
def price_difference : ℕ := 11

theorem t_mobile_base_cost : 
  T = 50 :=
by
  sorry

#check t_mobile_base_cost

end NUMINAMATH_CALUDE_ERRORFEEDBACK_t_mobile_base_cost_l352_35205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_parallelepiped_properties_l352_35215

/-- A rectangular parallelepiped with edge lengths 2, 3, and √3, whose vertices lie on a sphere. -/
structure RectangularParallelepiped :=
  (edge1 : ℝ) (edge2 : ℝ) (edge3 : ℝ)
  (vertices_on_sphere : Bool)
  (h1 : edge1 = 2)
  (h2 : edge2 = 3)
  (h3 : edge3 = Real.sqrt 3)
  (h4 : vertices_on_sphere = true)

/-- The length of the space diagonal of the rectangular parallelepiped. -/
noncomputable def space_diagonal (r : RectangularParallelepiped) : ℝ :=
  Real.sqrt (r.edge1^2 + r.edge2^2 + r.edge3^2)

/-- The surface area of the sphere that circumscribes the rectangular parallelepiped. -/
noncomputable def sphere_surface_area (r : RectangularParallelepiped) : ℝ :=
  4 * Real.pi * (space_diagonal r / 2)^2

/-- Theorem stating the properties of the rectangular parallelepiped and its circumscribing sphere. -/
theorem rectangular_parallelepiped_properties (r : RectangularParallelepiped) :
  space_diagonal r = 4 ∧ sphere_surface_area r = 16 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_parallelepiped_properties_l352_35215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_chord_length_l352_35288

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2 = 1

-- Define a line with slope 1
def line (x y t : ℝ) : Prop := y = x + t

-- Define the intersection points
def intersection_points (A B : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, ellipse A.1 A.2 ∧ ellipse B.1 B.2 ∧ 
            line A.1 A.2 t ∧ line B.1 B.2 t

-- Define the chord length
noncomputable def chord_length (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- Theorem statement
theorem max_chord_length :
  ∃ (A B : ℝ × ℝ), intersection_points A B ∧
    ∀ (C D : ℝ × ℝ), intersection_points C D →
      chord_length C D ≤ chord_length A B ∧
      chord_length A B = 4 * Real.sqrt 10 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_chord_length_l352_35288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_choir_arrangement_l352_35278

theorem choir_arrangement (n : Nat) (lower upper : Nat) : 
  n = 90 → lower = 6 → upper = 18 → 
  (Finset.filter (λ x => lower ≤ x ∧ x ≤ upper ∧ n % x = 0) (Finset.range (upper + 1))).card = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_choir_arrangement_l352_35278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_sum_lower_bound_l352_35222

/-- Predicate to check if the given side lengths and diagonals form a convex quadrilateral -/
def is_convex_quadrilateral (a b c d x y : ℝ) : Prop :=
  -- Add appropriate conditions for convexity
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ x > 0 ∧ y > 0 ∧
  (a + b + c + d > 2 * (max a b + max c d)) ∧
  (x + y > max (a + c) (b + d))

theorem diagonal_sum_lower_bound (a b c d x y : ℝ) :
  a > 0 →
  b ≥ a →
  c ≥ a →
  d ≥ a →
  x ≥ a →
  y ≥ a →
  is_convex_quadrilateral a b c d x y →
  x + y ≥ (1 + Real.sqrt 3) * a :=
by
  intros ha hb hc hd hx hy hconv
  sorry

#check diagonal_sum_lower_bound

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_sum_lower_bound_l352_35222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_triangle_y_coordinate_l352_35231

/-- Triangle ABC with horizontal symmetry through B -/
structure SymmetricTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  symmetry : B.1 = (A.1 + C.1) / 2

/-- Area of a triangle given base and height -/
noncomputable def triangleArea (base height : ℝ) : ℝ := (1 / 2) * base * height

theorem symmetric_triangle_y_coordinate 
  (triangle : SymmetricTriangle)
  (h_A : triangle.A = (0, 0))
  (h_C : triangle.C = (8, 0))
  (h_area : triangleArea 8 triangle.B.2 = 32) :
  triangle.B.2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_triangle_y_coordinate_l352_35231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_working_hours_in_four_weeks_l352_35201

/-- Calculates the total working hours in 4 weeks given the specified conditions -/
theorem total_working_hours_in_four_weeks : 
  let regular_days_per_week : ℕ := 5
  let regular_hours_per_day : ℕ := 8
  let regular_pay_per_hour : ℚ := 2.4
  let overtime_pay_per_hour : ℚ := 3.2
  let total_earnings_four_weeks : ℚ := 432

  let regular_hours_four_weeks : ℕ := 4 * regular_days_per_week * regular_hours_per_day
  let regular_earnings : ℚ := ↑regular_hours_four_weeks * regular_pay_per_hour
  let overtime_earnings : ℚ := total_earnings_four_weeks - regular_earnings
  let overtime_hours : ℕ := (overtime_earnings / overtime_pay_per_hour).floor.toNat
  let total_hours : ℕ := regular_hours_four_weeks + overtime_hours

  total_hours = 175 := by sorry

#check total_working_hours_in_four_weeks

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_working_hours_in_four_weeks_l352_35201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_group_proportion_l352_35270

theorem fifth_group_proportion (total_students : ℕ) (group_1 group_2 group_3 group_4 : ℕ)
  (h1 : total_students = 40)
  (h2 : group_1 = 14)
  (h3 : group_2 = 10)
  (h4 : group_3 = 8)
  (h5 : group_4 = 4) :
  (total_students - (group_1 + group_2 + group_3 + group_4)) / total_students = (1 : ℚ) / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_group_proportion_l352_35270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_odd_functions_l352_35220

-- Define the four functions
def f₁ : ℝ → ℝ := λ x ↦ x^3
def f₂ : ℝ → ℝ := λ x ↦ 2*x
def f₃ : ℝ → ℝ := λ x ↦ x^2 + 1
noncomputable def f₄ : ℝ → ℝ := λ x ↦ 2*Real.sin x

-- Define what it means for a function to be odd
def isOdd (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

-- State the theorem
theorem two_odd_functions :
  (isOdd f₁ ∧ isOdd f₄) ∧ 
  ¬(isOdd f₂ ∨ isOdd f₃) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_odd_functions_l352_35220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_64m4_l352_35218

/-- Given a positive integer m where 120m^3 has 120 positive integer divisors,
    prove that 64m^4 has 75 positive integer divisors -/
theorem divisors_of_64m4 (m : ℕ+) 
  (h : (Finset.filter (λ x ↦ (120 * m.val^3) % x = 0) (Finset.range (120 * m.val^3 + 1))).card = 120) :
  (Finset.filter (λ x ↦ (64 * m.val^4) % x = 0) (Finset.range (64 * m.val^4 + 1))).card = 75 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_64m4_l352_35218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l352_35289

-- Define the function f as noncomputable
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 9 * x / (a * x^2 + 1)

-- State the theorem
theorem f_inequality (a : ℝ) (h1 : a > 0) (h2 : a ≤ 1) :
  ∀ x > 1, (x^3 + 1) * f a x > 9 + Real.log x :=
by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l352_35289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roundRepeatingDecimal_l352_35233

/-- Rounds a real number to the nearest hundredth -/
noncomputable def roundToHundredth (x : ℝ) : ℝ := 
  ⌊x * 100 + 0.5⌋ / 100

/-- The repeating decimal 123.456456... as a real number -/
noncomputable def repeatingDecimal : ℝ := 123 + 456 / 999

theorem roundRepeatingDecimal : 
  roundToHundredth repeatingDecimal = 123.46 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roundRepeatingDecimal_l352_35233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_bound_l352_35204

noncomputable section

def semiperimeter (a b c : ℝ) : ℝ := (a + b + c) / 2

def area_triangle (a b c : ℝ) : ℝ :=
  let s := semiperimeter a b c
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem inscribed_circle_radius_bound (a b c ρ : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hρ : ρ > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_inscribed : ρ = (area_triangle a b c) / (semiperimeter a b c)) :
  ρ ≤ Real.sqrt (a^2 + b^2 + c^2) / 6 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_bound_l352_35204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_4_equals_8_l352_35294

def sequence_a : ℕ → ℕ
  | 0 => 2  -- Add this case for 0
  | 1 => 2
  | (n + 1) => sequence_a n + n

theorem a_4_equals_8 : sequence_a 4 = 8 := by
  -- Expand the definition of sequence_a
  have h1 : sequence_a 1 = 2 := rfl
  have h2 : sequence_a 2 = 3 := by simp [sequence_a, h1]
  have h3 : sequence_a 3 = 5 := by simp [sequence_a, h2]
  have h4 : sequence_a 4 = 8 := by simp [sequence_a, h3]
  exact h4


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_4_equals_8_l352_35294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l352_35234

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the point P
variable (P : ℝ × ℝ)

-- Define the condition that P is on the parabola
def P_on_parabola (P : ℝ × ℝ) : Prop := parabola P.1 P.2

-- Define d₁ (distance to axis of symmetry)
noncomputable def d₁ (P : ℝ × ℝ) : ℝ := P.1 + 1

-- Define d₂ (distance to line 3x - 4y + 9 = 0)
noncomputable def d₂ (P : ℝ × ℝ) : ℝ := |3*P.1 - 4*P.2 + 9| / Real.sqrt (3^2 + (-4)^2)

-- Theorem statement
theorem min_distance_sum :
  ∃ (P : ℝ × ℝ), P_on_parabola P ∧ 
  ∀ (Q : ℝ × ℝ), P_on_parabola Q → d₁ P + d₂ P ≤ d₁ Q + d₂ Q ∧
  d₁ P + d₂ P = 12/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l352_35234
