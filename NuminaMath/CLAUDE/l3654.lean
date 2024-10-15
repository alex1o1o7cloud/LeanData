import Mathlib

namespace NUMINAMATH_CALUDE_unique_m_value_l3654_365461

theorem unique_m_value (m : ℝ) : 
  let A : Set ℝ := {0, m, m^2 - 3*m + 2}
  2 ∈ A → m = 3 :=
by sorry

end NUMINAMATH_CALUDE_unique_m_value_l3654_365461


namespace NUMINAMATH_CALUDE_water_polo_team_selection_l3654_365418

def team_size : ℕ := 15
def starting_players : ℕ := 7
def coach_count : ℕ := 1

theorem water_polo_team_selection :
  (team_size * (team_size - 1) * (Nat.choose (team_size - 2) (starting_players - 2))) = 270270 := by
  sorry

end NUMINAMATH_CALUDE_water_polo_team_selection_l3654_365418


namespace NUMINAMATH_CALUDE_decimal_to_fraction_sum_l3654_365450

theorem decimal_to_fraction_sum (x : ℚ) (n d : ℕ) (v : ℕ) : 
  x = 2.52 →
  x = n / d →
  (∀ k : ℕ, k > 1 → ¬(k ∣ n ∧ k ∣ d)) →
  n + v = 349 →
  v = 286 := by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_sum_l3654_365450


namespace NUMINAMATH_CALUDE_last_three_digits_of_5_to_9000_l3654_365475

theorem last_three_digits_of_5_to_9000 (h : 5^300 ≡ 1 [MOD 800]) :
  5^9000 ≡ 1 [MOD 800] := by
sorry

end NUMINAMATH_CALUDE_last_three_digits_of_5_to_9000_l3654_365475


namespace NUMINAMATH_CALUDE_function_property_implies_k_values_l3654_365463

-- Define the function type
def FunctionType := ℕ → ℤ

-- Define the property that the function must satisfy
def SatisfiesProperty (f : FunctionType) (k : ℤ) : Prop :=
  f 1995 = 1996 ∧
  ∀ x y : ℕ, f (x * y) = f x + f y + k * f (Nat.gcd x y)

-- Theorem statement
theorem function_property_implies_k_values :
  ∀ f : FunctionType, ∀ k : ℤ,
    SatisfiesProperty f k → (k = -1 ∨ k = 0) :=
sorry

end NUMINAMATH_CALUDE_function_property_implies_k_values_l3654_365463


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_fraction_l3654_365492

theorem purely_imaginary_complex_fraction (a : ℝ) :
  let z : ℂ := (a + Complex.I) / (1 - Complex.I)
  (∃ (b : ℝ), z = Complex.I * b) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_fraction_l3654_365492


namespace NUMINAMATH_CALUDE_min_value_of_f_l3654_365458

-- Define the function
def f (x : ℝ) : ℝ := x^3 + x^2 - x + 1

-- State the theorem
theorem min_value_of_f :
  ∃ (x : ℝ), x ∈ Set.Icc (-2) 1 ∧
  (∀ (y : ℝ), y ∈ Set.Icc (-2) 1 → f x ≤ f y) ∧
  f x = -1 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l3654_365458


namespace NUMINAMATH_CALUDE_ab_equality_l3654_365464

theorem ab_equality (a b : ℚ) (h1 : 10 * a = 20) (h2 : 6 * b = 20) : 120 * a * b = 800 := by
  sorry

end NUMINAMATH_CALUDE_ab_equality_l3654_365464


namespace NUMINAMATH_CALUDE_total_employee_costs_february_l3654_365449

/-- Represents an employee in the car dealership -/
structure Employee where
  name : String
  hoursPerWeek : Nat
  hourlyRate : Nat
  weeksWorked : Nat
  overtime : Nat
  overtimeRate : Nat
  bonus : Int
  deduction : Nat

/-- Calculates the monthly earnings for an employee -/
def monthlyEarnings (e : Employee) : Int :=
  e.hoursPerWeek * e.hourlyRate * e.weeksWorked +
  e.overtime * e.overtimeRate +
  e.bonus -
  e.deduction

/-- Theorem stating the total employee costs for February -/
theorem total_employee_costs_february :
  let fiona : Employee := ⟨"Fiona", 40, 20, 3, 0, 0, 0, 0⟩
  let john : Employee := ⟨"John", 30, 22, 4, 10, 33, 0, 0⟩
  let jeremy : Employee := ⟨"Jeremy", 25, 18, 4, 0, 0, 200, 0⟩
  let katie : Employee := ⟨"Katie", 35, 21, 4, 0, 0, 0, 150⟩
  let matt : Employee := ⟨"Matt", 28, 19, 4, 0, 0, 0, 0⟩
  monthlyEarnings fiona + monthlyEarnings john + monthlyEarnings jeremy +
  monthlyEarnings katie + monthlyEarnings matt = 13278 := by
  sorry


end NUMINAMATH_CALUDE_total_employee_costs_february_l3654_365449


namespace NUMINAMATH_CALUDE_johns_fee_value_l3654_365439

/-- The one-time sitting fee for John's Photo World -/
def johns_fee : ℝ := sorry

/-- The price per sheet at John's Photo World -/
def johns_price_per_sheet : ℝ := 2.75

/-- The price per sheet at Sam's Picture Emporium -/
def sams_price_per_sheet : ℝ := 1.50

/-- The one-time sitting fee for Sam's Picture Emporium -/
def sams_fee : ℝ := 140

/-- The number of sheets being compared -/
def num_sheets : ℝ := 12

theorem johns_fee_value : johns_fee = 125 :=
  by
    have h : johns_price_per_sheet * num_sheets + johns_fee = sams_price_per_sheet * num_sheets + sams_fee :=
      sorry
    sorry

#check johns_fee_value

end NUMINAMATH_CALUDE_johns_fee_value_l3654_365439


namespace NUMINAMATH_CALUDE_sam_bought_one_lollipop_l3654_365436

/-- Calculates the number of lollipops Sam bought -/
def lollipops_bought (initial_dimes : ℕ) (initial_quarters : ℕ) (candy_bars : ℕ) 
  (dimes_per_candy : ℕ) (cents_per_lollipop : ℕ) (cents_left : ℕ) : ℕ :=
  let initial_cents := initial_dimes * 10 + initial_quarters * 25
  let candy_cost := candy_bars * dimes_per_candy * 10
  let cents_for_lollipops := initial_cents - candy_cost - cents_left
  cents_for_lollipops / cents_per_lollipop

theorem sam_bought_one_lollipop :
  lollipops_bought 19 6 4 3 25 195 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sam_bought_one_lollipop_l3654_365436


namespace NUMINAMATH_CALUDE_min_value_x_plus_2y_l3654_365451

theorem min_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + y = x * y) :
  x + 2 * y ≥ 3 + 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_2y_l3654_365451


namespace NUMINAMATH_CALUDE_one_of_each_color_probability_l3654_365410

/-- The probability of selecting one marble of each color from a bag with 3 red, 3 blue, and 3 green marbles -/
theorem one_of_each_color_probability : 
  let total_marbles : ℕ := 3 + 3 + 3
  let marbles_per_color : ℕ := 3
  let selected_marbles : ℕ := 3
  (marbles_per_color ^ selected_marbles : ℚ) / (Nat.choose total_marbles selected_marbles) = 9 / 28 :=
by sorry

end NUMINAMATH_CALUDE_one_of_each_color_probability_l3654_365410


namespace NUMINAMATH_CALUDE_power_difference_l3654_365422

theorem power_difference (a m n : ℝ) (h1 : a^m = 8) (h2 : a^n = 2) : a^(m-3*n) = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_difference_l3654_365422


namespace NUMINAMATH_CALUDE_line_equation_l3654_365417

/-- A line passing through a point and intersecting a circle with a given chord length -/
def intersecting_line (P : ℝ × ℝ) (center : ℝ × ℝ) (radius : ℝ) (chord_length : ℝ) :=
  {l : Set (ℝ × ℝ) | ∃ (A B : ℝ × ℝ),
    A ∈ l ∧ B ∈ l ∧
    P ∈ l ∧
    (A.1 - center.1)^2 + (A.2 - center.2)^2 = radius^2 ∧
    (B.1 - center.1)^2 + (B.2 - center.2)^2 = radius^2 ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = chord_length^2}

theorem line_equation (P : ℝ × ℝ) (center : ℝ × ℝ) (radius chord_length : ℝ)
  (h1 : P = (3, 6))
  (h2 : center = (0, 0))
  (h3 : radius = 5)
  (h4 : chord_length = 8) :
  ∀ l ∈ intersecting_line P center radius chord_length,
  (∀ (x y : ℝ), (x, y) ∈ l ↔ (x - 3 = 0 ∨ 3*x - 4*y + 15 = 0)) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_l3654_365417


namespace NUMINAMATH_CALUDE_three_planes_intersection_count_l3654_365445

structure Plane

/-- Three planes that intersect pairwise -/
structure ThreePlanesIntersectingPairwise where
  plane1 : Plane
  plane2 : Plane
  plane3 : Plane
  intersect12 : plane1 ≠ plane2
  intersect23 : plane2 ≠ plane3
  intersect13 : plane1 ≠ plane3

/-- A line of intersection between two planes -/
def LineOfIntersection (p1 p2 : Plane) : Type := Unit

/-- Count the number of distinct lines of intersection -/
def CountLinesOfIntersection (t : ThreePlanesIntersectingPairwise) : Nat :=
  sorry

theorem three_planes_intersection_count
  (t : ThreePlanesIntersectingPairwise) :
  CountLinesOfIntersection t = 1 ∨ CountLinesOfIntersection t = 3 :=
sorry

end NUMINAMATH_CALUDE_three_planes_intersection_count_l3654_365445


namespace NUMINAMATH_CALUDE_first_three_average_l3654_365429

theorem first_three_average (a b c d : ℝ) : 
  a = 33 →
  d = 18 →
  (b + c + d) / 3 = 15 →
  (a + b + c) / 3 = 20 := by
sorry

end NUMINAMATH_CALUDE_first_three_average_l3654_365429


namespace NUMINAMATH_CALUDE_greatest_q_minus_r_l3654_365478

theorem greatest_q_minus_r (q r : ℕ+) (h : 1001 = 17 * q + r) : 
  ∀ (q' r' : ℕ+), 1001 = 17 * q' + r' → q - r ≥ q' - r' := by
  sorry

end NUMINAMATH_CALUDE_greatest_q_minus_r_l3654_365478


namespace NUMINAMATH_CALUDE_solution_satisfies_equation_l3654_365435

def F (a b c d : ℕ) : ℕ := a^b + c * d

theorem solution_satisfies_equation : F 3 6 5 15 = 490 := by sorry

end NUMINAMATH_CALUDE_solution_satisfies_equation_l3654_365435


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l3654_365415

theorem quadratic_two_distinct_roots (a : ℝ) : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
  x₁^2 + 2*a*x₁ + a^2 - 1 = 0 ∧ 
  x₂^2 + 2*a*x₂ + a^2 - 1 = 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l3654_365415


namespace NUMINAMATH_CALUDE_complex_equality_and_minimum_distance_l3654_365413

open Complex

theorem complex_equality_and_minimum_distance (z : ℂ) :
  (abs z = abs (z + 1 + I)) →
  (∃ (a : ℝ), z = a + a * I) →
  (z = -1 - I) ∧
  (∃ (min_dist : ℝ), min_dist = Real.sqrt 2 ∧
    ∀ (w : ℂ), abs w = abs (w + 1 + I) → abs (w - (2 - I)) ≥ min_dist) :=
by sorry

end NUMINAMATH_CALUDE_complex_equality_and_minimum_distance_l3654_365413


namespace NUMINAMATH_CALUDE_power_zero_eq_one_l3654_365438

theorem power_zero_eq_one (n : ℤ) (h : n ≠ 0) : n^0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_zero_eq_one_l3654_365438


namespace NUMINAMATH_CALUDE_first_sales_amount_l3654_365443

/-- Proves that the amount of the first sales is $10 million -/
theorem first_sales_amount (initial_royalties : ℝ) (subsequent_royalties : ℝ) 
  (subsequent_sales : ℝ) (royalty_rate_ratio : ℝ) :
  initial_royalties = 2 →
  subsequent_royalties = 8 →
  subsequent_sales = 100 →
  royalty_rate_ratio = 0.4 →
  ∃ (initial_sales : ℝ), initial_sales = 10 ∧ 
    (initial_royalties / initial_sales = subsequent_royalties / subsequent_sales / royalty_rate_ratio) :=
by sorry

end NUMINAMATH_CALUDE_first_sales_amount_l3654_365443


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l3654_365457

/-- Simple interest calculation -/
def simple_interest (principal rate time : ℚ) : ℚ :=
  principal * rate * time / 100

/-- Problem statement -/
theorem interest_rate_calculation (principal interest time : ℚ) 
  (h_principal : principal = 800)
  (h_interest : interest = 160)
  (h_time : time = 4)
  (h_simple_interest : simple_interest principal (5 : ℚ) time = interest) :
  simple_interest principal (5 : ℚ) time = interest :=
by sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l3654_365457


namespace NUMINAMATH_CALUDE_replacement_concentration_theorem_l3654_365460

/-- Given an initial solution concentration, a replacement solution concentration,
    and the fraction of solution replaced, calculate the new concentration. -/
def new_concentration (initial_conc : ℝ) (replacement_conc : ℝ) (fraction_replaced : ℝ) : ℝ :=
  (initial_conc * (1 - fraction_replaced) + replacement_conc * fraction_replaced)

/-- Theorem stating that replacing half of a 45% solution with a 25% solution
    results in a 35% solution. -/
theorem replacement_concentration_theorem :
  new_concentration 0.45 0.25 0.5 = 0.35 := by
  sorry

#eval new_concentration 0.45 0.25 0.5

end NUMINAMATH_CALUDE_replacement_concentration_theorem_l3654_365460


namespace NUMINAMATH_CALUDE_derivative_zero_necessary_not_sufficient_l3654_365419

-- Define a differentiable function
variable (f : ℝ → ℝ)
variable (hf : Differentiable ℝ f)

-- Define what it means for a function to have an extremum at a point
def HasExtremumAt (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∃ ε > 0, ∀ y ∈ Set.Ioo (x - ε) (x + ε), f y ≤ f x ∨ f y ≥ f x

-- State the theorem
theorem derivative_zero_necessary_not_sufficient :
  (∀ x : ℝ, HasExtremumAt f x → deriv f x = 0) ∧
  ¬(∀ x : ℝ, deriv f x = 0 → HasExtremumAt f x) :=
sorry

end NUMINAMATH_CALUDE_derivative_zero_necessary_not_sufficient_l3654_365419


namespace NUMINAMATH_CALUDE_increasing_function_derivative_relation_l3654_365468

open Set
open Function
open Topology

theorem increasing_function_derivative_relation 
  {a b : ℝ} (hab : a < b) (f : ℝ → ℝ) (hf : DifferentiableOn ℝ f (Ioo a b)) :
  (∀ x ∈ Ioo a b, (deriv f) x > 0 → StrictMonoOn f (Ioo a b)) ∧
  ¬(StrictMonoOn f (Ioo a b) → ∀ x ∈ Ioo a b, (deriv f) x > 0) :=
sorry

end NUMINAMATH_CALUDE_increasing_function_derivative_relation_l3654_365468


namespace NUMINAMATH_CALUDE_semi_annual_annuity_payment_l3654_365403

/-- Calculates the semi-annual annuity payment given the following conditions:
  * Initial annual payment of 2500 HUF
  * Payment duration of 15 years
  * No collection for first 5 years
  * Convert to semi-annual annuity lasting 20 years, starting at beginning of 6th year
  * Annual interest rate of 4.75%
-/
def calculate_semi_annual_annuity (
  initial_payment : ℝ
  ) (payment_duration : ℕ
  ) (no_collection_years : ℕ
  ) (annuity_duration : ℕ
  ) (annual_interest_rate : ℝ
  ) : ℝ :=
  sorry

/-- The semi-annual annuity payment is approximately 2134.43 HUF -/
theorem semi_annual_annuity_payment :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |calculate_semi_annual_annuity 2500 15 5 20 0.0475 - 2134.43| < ε :=
sorry

end NUMINAMATH_CALUDE_semi_annual_annuity_payment_l3654_365403


namespace NUMINAMATH_CALUDE_fifteenth_recalibration_in_march_l3654_365441

/-- Calculates the month of the nth recalibration given a start month and recalibration interval -/
def recalibrationMonth (startMonth : Nat) (interval : Nat) (n : Nat) : Nat :=
  ((startMonth - 1) + (n - 1) * interval) % 12 + 1

/-- The month of the 15th recalibration is March (month 3) -/
theorem fifteenth_recalibration_in_march :
  recalibrationMonth 1 7 15 = 3 := by
  sorry

#eval recalibrationMonth 1 7 15

end NUMINAMATH_CALUDE_fifteenth_recalibration_in_march_l3654_365441


namespace NUMINAMATH_CALUDE_solution_systems_l3654_365408

-- System a
def system_a (x y : ℝ) : Prop :=
  x + y + x*y = 5 ∧ x*y*(x + y) = 6

-- System b
def system_b (x y : ℝ) : Prop :=
  x^3 + y^3 + 2*x*y = 4 ∧ x^2 - x*y + y^2 = 1

theorem solution_systems :
  (∃ x y : ℝ, system_a x y ∧ ((x = 2 ∧ y = 1) ∨ (x = 1 ∧ y = 2))) ∧
  (∃ x y : ℝ, system_b x y ∧ x = 1 ∧ y = 1) := by
  sorry

end NUMINAMATH_CALUDE_solution_systems_l3654_365408


namespace NUMINAMATH_CALUDE_range_of_t_l3654_365480

-- Define the solution set of (a-1)^x > 1
def solution_set (a : ℝ) : Set ℝ := {x | x < 0}

-- Define the inequality q
def q (a t : ℝ) : Prop := a^2 - 2*t*a + t^2 - 1 < 0

-- Define the negation of p
def not_p (a : ℝ) : Prop := a ≤ 1 ∨ a ≥ 2

-- Define the negation of q
def not_q (a t : ℝ) : Prop := ¬(q a t)

-- Statement of the theorem
theorem range_of_t :
  (∀ a, solution_set a = {x | x < 0}) →
  (∀ a t, not_p a → not_q a t) →
  (∃ a t, not_p a ∧ q a t) →
  ∀ t, (∀ a, not_p a → not_q a t) → 1 ≤ t ∧ t ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_t_l3654_365480


namespace NUMINAMATH_CALUDE_sticks_per_matchbox_l3654_365467

/-- Given the following:
  * num_boxes: The number of boxes ordered
  * matchboxes_per_box: The number of matchboxes in each box
  * total_sticks: The total number of match sticks ordered

  Prove that the number of match sticks in each matchbox is 300.
-/
theorem sticks_per_matchbox
  (num_boxes : ℕ)
  (matchboxes_per_box : ℕ)
  (total_sticks : ℕ)
  (h1 : num_boxes = 4)
  (h2 : matchboxes_per_box = 20)
  (h3 : total_sticks = 24000) :
  total_sticks / (num_boxes * matchboxes_per_box) = 300 := by
  sorry

end NUMINAMATH_CALUDE_sticks_per_matchbox_l3654_365467


namespace NUMINAMATH_CALUDE_total_money_l3654_365440

theorem total_money (a b : ℝ) (h1 : (4/15) * a = (2/5) * b) (h2 : b = 484) :
  a + b = 1210 := by
  sorry

end NUMINAMATH_CALUDE_total_money_l3654_365440


namespace NUMINAMATH_CALUDE_quadratic_form_j_value_l3654_365402

/-- Given a quadratic expression px^2 + qx + r that can be expressed as 5(x - 3)^2 + 15,
    prove that when 4px^2 + 4qx + 4r is expressed as m(x - j)^2 + k, j = 3. -/
theorem quadratic_form_j_value (p q r : ℝ) :
  (∃ m j k : ℝ, ∀ x : ℝ, 
    px^2 + q*x + r = 5*(x - 3)^2 + 15 ∧ 
    4*p*x^2 + 4*q*x + 4*r = m*(x - j)^2 + k) →
  (∃ m k : ℝ, ∀ x : ℝ, 4*p*x^2 + 4*q*x + 4*r = m*(x - 3)^2 + k) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_form_j_value_l3654_365402


namespace NUMINAMATH_CALUDE_min_value_theorem_l3654_365405

-- Define the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n + r

-- Define the conditions of the problem
def problem_conditions (a : ℕ → ℝ) : Prop :=
  arithmetic_sequence a ∧
  (∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n + r) ∧
  a 2018 = a 2017 + 2 * a 2016 ∧
  ∃ m n : ℕ, m > 0 ∧ n > 0 ∧ a m * a n = 16 * (a 1)^2

-- State the theorem
theorem min_value_theorem (a : ℕ → ℝ) :
  problem_conditions a →
  ∃ m n : ℕ, m > 0 ∧ n > 0 ∧ 4/m + 1/n ≥ 5/3 ∧
  (∀ k l : ℕ, k > 0 → l > 0 → 4/k + 1/l ≥ 4/m + 1/n) :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3654_365405


namespace NUMINAMATH_CALUDE_line_equation_l3654_365491

/-- The curve y = 3x^2 - 4x + 2 -/
def f (x : ℝ) : ℝ := 3 * x^2 - 4 * x + 2

/-- The derivative of the curve -/
def f' (x : ℝ) : ℝ := 6 * x - 4

/-- The point P -/
def P : ℝ × ℝ := (-1, 2)

/-- The point M -/
def M : ℝ × ℝ := (1, 1)

/-- The slope of the tangent line at M -/
def m : ℝ := f' M.1

theorem line_equation (x y : ℝ) :
  (2 * x - y + 4 = 0) ↔
  (y - P.2 = m * (x - P.1) ∧ 
   ∃ (t : ℝ), (x, y) = (t, f t) → y - f M.1 = m * (x - M.1)) := by
  sorry

end NUMINAMATH_CALUDE_line_equation_l3654_365491


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_50_l3654_365421

theorem consecutive_integers_sum_50 : 
  ∃ (x : ℕ), x > 0 ∧ x + (x + 1) + (x + 2) + (x + 3) = 50 :=
by sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_50_l3654_365421


namespace NUMINAMATH_CALUDE_complex_fraction_equals_962_l3654_365482

/-- Helper function to represent the factorization of x^4 + 400 --/
def factor (x : ℤ) : ℤ := (x * (x - 10) + 20) * (x * (x + 10) + 20)

/-- The main theorem stating that the given expression equals 962 --/
theorem complex_fraction_equals_962 : 
  (factor 10 * factor 26 * factor 42 * factor 58) / 
  (factor 2 * factor 18 * factor 34 * factor 50) = 962 := by
  sorry


end NUMINAMATH_CALUDE_complex_fraction_equals_962_l3654_365482


namespace NUMINAMATH_CALUDE_square_side_length_with_inscribed_circle_l3654_365455

theorem square_side_length_with_inscribed_circle (s : ℝ) : 
  (4 * s = π * (s / 2)^2) → s = 16 / π := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_with_inscribed_circle_l3654_365455


namespace NUMINAMATH_CALUDE_cos_x_plus_pi_sixth_l3654_365486

theorem cos_x_plus_pi_sixth (x : ℝ) (h : Real.sin (π / 3 - x) = 3 / 5) :
  Real.cos (x + π / 6) = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_cos_x_plus_pi_sixth_l3654_365486


namespace NUMINAMATH_CALUDE_inequality_relationship_l3654_365490

theorem inequality_relationship (a b : ℝ) : 
  (∀ x y : ℝ, x > y → x + 1 > y - 2) ∧ 
  (∃ x y : ℝ, x + 1 > y - 2 ∧ ¬(x > y)) :=
sorry

end NUMINAMATH_CALUDE_inequality_relationship_l3654_365490


namespace NUMINAMATH_CALUDE_max_profit_computer_sales_profit_per_computer_type_l3654_365485

/-- Profit function for computer sales -/
def profit_function (m : ℕ) : ℝ := -50 * m + 15000

/-- Constraint on the number of type B computers -/
def type_b_constraint (m : ℕ) : Prop := 100 - m ≤ 2 * m

/-- Theorem stating the maximum profit and optimal purchase strategy -/
theorem max_profit_computer_sales :
  ∃ (m : ℕ),
    m = 34 ∧
    type_b_constraint m ∧
    profit_function m = 13300 ∧
    ∀ (n : ℕ), type_b_constraint n → profit_function n ≤ profit_function m :=
by
  sorry

/-- Theorem verifying the profit for each computer type -/
theorem profit_per_computer_type :
  ∃ (a b : ℝ),
    a = 100 ∧
    b = 150 ∧
    10 * a + 20 * b = 4000 ∧
    20 * a + 10 * b = 3500 :=
by
  sorry

end NUMINAMATH_CALUDE_max_profit_computer_sales_profit_per_computer_type_l3654_365485


namespace NUMINAMATH_CALUDE_garden_fencing_length_l3654_365472

theorem garden_fencing_length (garden_area : ℝ) (π_approx : ℝ) (extra_length : ℝ) : 
  garden_area = 616 → 
  π_approx = 22 / 7 → 
  extra_length = 5 → 
  2 * π_approx * Real.sqrt (garden_area / π_approx) + extra_length = 93 := by
sorry

end NUMINAMATH_CALUDE_garden_fencing_length_l3654_365472


namespace NUMINAMATH_CALUDE_product_of_sums_equals_difference_of_powers_l3654_365494

theorem product_of_sums_equals_difference_of_powers : 
  (4 + 3) * (4^2 + 3^2) * (4^4 + 3^4) * (4^8 + 3^8) * (4^16 + 3^16) * (4^32 + 3^32) * (4^64 + 3^64) = 3^128 - 4^128 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sums_equals_difference_of_powers_l3654_365494


namespace NUMINAMATH_CALUDE_smallest_value_between_one_and_two_l3654_365433

theorem smallest_value_between_one_and_two (y : ℝ) (h1 : 1 < y) (h2 : y < 2) :
  (1 / y < y) ∧ (1 / y < y^2) ∧ (1 / y < 2*y) ∧ (1 / y < Real.sqrt y) := by
  sorry

end NUMINAMATH_CALUDE_smallest_value_between_one_and_two_l3654_365433


namespace NUMINAMATH_CALUDE_cone_surface_area_l3654_365401

/-- The surface area of a cone, given its lateral surface properties -/
theorem cone_surface_area (r : Real) (arc_length : Real) : 
  r = 4 → arc_length = 4 * Real.pi → 
  (π * (arc_length / (2 * π))^2) + (1/2 * r * arc_length) = 12 * π := by
sorry

end NUMINAMATH_CALUDE_cone_surface_area_l3654_365401


namespace NUMINAMATH_CALUDE_relationship_equation_l3654_365479

theorem relationship_equation (x : ℝ) : 
  (2023 : ℝ) = (1/4 : ℝ) * x + 1 ↔ 
    (∃ A B : ℝ, A = 2023 ∧ B = x ∧ A = (1/4 : ℝ) * B + 1) :=
by sorry

end NUMINAMATH_CALUDE_relationship_equation_l3654_365479


namespace NUMINAMATH_CALUDE_limit_at_neg_three_is_zero_l3654_365465

/-- The limit of (x^2 + 2x - 3)^2 / (x^3 + 4x^2 + 3x) as x approaches -3 is 0 -/
theorem limit_at_neg_three_is_zero :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 
    0 < |x + 3| ∧ |x + 3| < δ → 
    |(x^2 + 2*x - 3)^2 / (x^3 + 4*x^2 + 3*x) - 0| < ε :=
by
  sorry

#check limit_at_neg_three_is_zero

end NUMINAMATH_CALUDE_limit_at_neg_three_is_zero_l3654_365465


namespace NUMINAMATH_CALUDE_johns_notebooks_l3654_365420

theorem johns_notebooks (total_children : Nat) (wife_notebooks_per_child : Nat) (total_notebooks : Nat) :
  total_children = 3 →
  wife_notebooks_per_child = 5 →
  total_notebooks = 21 →
  ∃ (johns_notebooks_per_child : Nat),
    johns_notebooks_per_child * total_children + wife_notebooks_per_child * total_children = total_notebooks ∧
    johns_notebooks_per_child = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_johns_notebooks_l3654_365420


namespace NUMINAMATH_CALUDE_least_xy_value_l3654_365447

theorem least_xy_value (x y : ℕ+) (h : (1 : ℚ) / x + (1 : ℚ) / (3 * y) = (1 : ℚ) / 6) :
  (∀ a b : ℕ+, (1 : ℚ) / a + (1 : ℚ) / (3 * b) = (1 : ℚ) / 6 → x * y ≤ a * b) ∧ x * y = 64 := by
  sorry

end NUMINAMATH_CALUDE_least_xy_value_l3654_365447


namespace NUMINAMATH_CALUDE_french_fries_cost_is_ten_l3654_365442

/-- Represents the cost of a meal at Wendy's -/
structure WendysMeal where
  taco_salad : ℕ
  hamburgers : ℕ
  lemonade : ℕ
  friends : ℕ
  individual_payment : ℕ

/-- Calculates the total cost of french fries in a Wendy's meal -/
def french_fries_cost (meal : WendysMeal) : ℕ :=
  meal.friends * meal.individual_payment -
  (meal.taco_salad + 5 * meal.hamburgers + 5 * meal.lemonade)

/-- Theorem stating that the total cost of french fries is $10 -/
theorem french_fries_cost_is_ten (meal : WendysMeal)
  (h1 : meal.taco_salad = 10)
  (h2 : meal.hamburgers = 5)
  (h3 : meal.lemonade = 2)
  (h4 : meal.friends = 5)
  (h5 : meal.individual_payment = 11) :
  french_fries_cost meal = 10 := by
  sorry

#eval french_fries_cost { taco_salad := 10, hamburgers := 5, lemonade := 2, friends := 5, individual_payment := 11 }

end NUMINAMATH_CALUDE_french_fries_cost_is_ten_l3654_365442


namespace NUMINAMATH_CALUDE_cost_price_calculation_l3654_365487

theorem cost_price_calculation (selling_price : ℝ) (profit_percentage : ℝ) 
  (h1 : selling_price = 240)
  (h2 : profit_percentage = 0.25) : 
  ∃ (cost_price : ℝ), cost_price = 192 ∧ selling_price = cost_price * (1 + profit_percentage) :=
by
  sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l3654_365487


namespace NUMINAMATH_CALUDE_badminton_players_count_l3654_365434

/-- Represents a sports club with members playing badminton and tennis -/
structure SportsClub where
  total_members : ℕ
  badminton_players : ℕ
  tennis_players : ℕ
  both_players : ℕ
  neither_players : ℕ

/-- Theorem stating the number of badminton players in the given conditions -/
theorem badminton_players_count (club : SportsClub)
  (h1 : club.total_members = 30)
  (h2 : club.badminton_players = club.tennis_players)
  (h3 : club.neither_players = 2)
  (h4 : club.both_players = 6)
  (h5 : club.total_members = club.badminton_players + club.tennis_players - club.both_players + club.neither_players) :
  club.badminton_players = 17 := by
  sorry


end NUMINAMATH_CALUDE_badminton_players_count_l3654_365434


namespace NUMINAMATH_CALUDE_opposite_reciprocal_expression_value_l3654_365426

theorem opposite_reciprocal_expression_value (a b c d m : ℝ) :
  a + b = 0 →
  c * d = 1 →
  |m| = 4 →
  (a + b) / (3 * m) + m^2 - 5 * c * d + 6 * m = 35 ∨
  (a + b) / (3 * m) + m^2 - 5 * c * d + 6 * m = -13 :=
by sorry

end NUMINAMATH_CALUDE_opposite_reciprocal_expression_value_l3654_365426


namespace NUMINAMATH_CALUDE_angle_calculation_l3654_365471

/-- Represents an angle in degrees and minutes -/
structure Angle :=
  (degrees : ℤ)
  (minutes : ℤ)

/-- Multiplication of an angle by an integer -/
def Angle.mul (a : Angle) (n : ℤ) : Angle :=
  ⟨a.degrees * n, a.minutes * n⟩

/-- Addition of two angles -/
def Angle.add (a b : Angle) : Angle :=
  ⟨a.degrees + b.degrees, a.minutes + b.minutes⟩

/-- Subtraction of two angles -/
def Angle.sub (a b : Angle) : Angle :=
  ⟨a.degrees - b.degrees, a.minutes - b.minutes⟩

/-- Normalize an angle by converting excess minutes to degrees -/
def Angle.normalize (a : Angle) : Angle :=
  let extraDegrees := a.minutes / 60
  let normalizedMinutes := a.minutes % 60
  ⟨a.degrees + extraDegrees, normalizedMinutes⟩

theorem angle_calculation :
  (Angle.normalize ((Angle.mul ⟨24, 31⟩ 4).sub ⟨62, 10⟩)) = ⟨35, 54⟩ := by
  sorry

end NUMINAMATH_CALUDE_angle_calculation_l3654_365471


namespace NUMINAMATH_CALUDE_percentage_difference_l3654_365476

theorem percentage_difference : 
  let sixty_percent_of_fifty : ℝ := (60 / 100) * 50
  let fifty_percent_of_thirty : ℝ := (50 / 100) * 30
  sixty_percent_of_fifty - fifty_percent_of_thirty = 15 := by
sorry

end NUMINAMATH_CALUDE_percentage_difference_l3654_365476


namespace NUMINAMATH_CALUDE_truncated_pyramid_edges_and_height_l3654_365452

theorem truncated_pyramid_edges_and_height :
  ∃ (x y z u r s t : ℤ),
    x = 4 * r * t ∧
    y = 4 * s * t ∧
    z = (r - s)^2 - 2 * t^2 ∧
    u = (r - s)^2 + 2 * t^2 ∧
    (x - y)^2 + 2 * z^2 = 2 * u^2 :=
by sorry

end NUMINAMATH_CALUDE_truncated_pyramid_edges_and_height_l3654_365452


namespace NUMINAMATH_CALUDE_band_repertoire_size_l3654_365406

def prove_band_repertoire (first_set second_set encore third_and_fourth_avg : ℕ) : Prop :=
  let total_songs := first_set + second_set + encore + 2 * third_and_fourth_avg
  total_songs = 30

theorem band_repertoire_size :
  prove_band_repertoire 5 7 2 8 := by
  sorry

end NUMINAMATH_CALUDE_band_repertoire_size_l3654_365406


namespace NUMINAMATH_CALUDE_cost_minimized_at_35_l3654_365481

/-- Represents the cost function for ordering hand sanitizers -/
def cost_function (x : ℝ) : ℝ := -2 * x^2 + 102 * x + 5000

/-- Represents the constraint on the number of boxes of type A sanitizer -/
def constraint (x : ℝ) : Prop := 15 ≤ x ∧ x ≤ 35

/-- Theorem stating that the cost function is minimized at x = 35 within the given constraints -/
theorem cost_minimized_at_35 :
  ∀ x : ℝ, constraint x → cost_function x ≥ cost_function 35 :=
sorry

end NUMINAMATH_CALUDE_cost_minimized_at_35_l3654_365481


namespace NUMINAMATH_CALUDE_club_officer_selection_l3654_365454

/-- Represents the number of ways to choose officers in a club -/
def choose_officers (total_members boys girls : ℕ) : ℕ :=
  let president_vp_combinations := boys * girls * 2
  let secretary_choices := girls
  president_vp_combinations * secretary_choices

/-- Theorem stating the number of ways to choose officers under given conditions -/
theorem club_officer_selection :
  choose_officers 15 9 6 = 648 :=
by
  sorry


end NUMINAMATH_CALUDE_club_officer_selection_l3654_365454


namespace NUMINAMATH_CALUDE_gcd_lcm_product_48_75_l3654_365499

theorem gcd_lcm_product_48_75 : Nat.gcd 48 75 * Nat.lcm 48 75 = 3600 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_48_75_l3654_365499


namespace NUMINAMATH_CALUDE_f_min_value_inequality_property_l3654_365416

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + |x + 3|

-- Theorem for the minimum value of f
theorem f_min_value :
  (∀ x : ℝ, f x ≥ 4) ∧ (∃ x : ℝ, f x = 4) := by sorry

-- Theorem for the inequality
theorem inequality_property (a b x : ℝ) (ha : |a| < 2) (hb : |b| < 2) :
  |a + b| + |a - b| < f x := by sorry

end NUMINAMATH_CALUDE_f_min_value_inequality_property_l3654_365416


namespace NUMINAMATH_CALUDE_triangle_area_and_length_l3654_365430

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively, and point D as the midpoint of BC -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ × ℝ

/-- The area of a triangle -/
def area (t : Triangle) : ℝ := sorry

/-- The distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

theorem triangle_area_and_length (t : Triangle) :
  (t.c * Real.cos t.B = Real.sqrt 3 * t.b * Real.sin t.C) →
  (t.a^2 * Real.sin t.C = 4 * Real.sqrt 3 * Real.sin t.A) →
  (area t = Real.sqrt 3) ∧
  (t.a = 2 * Real.sqrt 3 → t.b = Real.sqrt 7 → t.c > t.b →
   distance (0, 0) t.D = Real.sqrt 13) := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_and_length_l3654_365430


namespace NUMINAMATH_CALUDE_sum_product_equality_l3654_365495

theorem sum_product_equality (x y z : ℝ) (h : x + y + z = x * y * z) :
  x * (1 - y^2) * (1 - z^2) + y * (1 - z^2) * (1 - x^2) + z * (1 - x^2) * (1 - y^2) = 4 * x * y * z := by
  sorry

end NUMINAMATH_CALUDE_sum_product_equality_l3654_365495


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_equation_l3654_365473

theorem sum_of_roots_quadratic_equation : 
  let f : ℝ → ℝ := λ x => x^2 + x - 2
  ∃ r₁ r₂ : ℝ, f r₁ = 0 ∧ f r₂ = 0 ∧ r₁ + r₂ = -1 :=
sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_equation_l3654_365473


namespace NUMINAMATH_CALUDE_odot_calculation_l3654_365496

-- Define the ⊙ operation
def odot (a b : ℤ) : ℤ := a * b - (a + b)

-- State the theorem
theorem odot_calculation : odot 6 (odot 5 4) = 49 := by
  sorry

end NUMINAMATH_CALUDE_odot_calculation_l3654_365496


namespace NUMINAMATH_CALUDE_equal_angle_point_exists_l3654_365477

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Three non-overlapping circles -/
structure ThreeCircles where
  c₁ : Circle
  c₂ : Circle
  c₃ : Circle
  non_overlapping : c₁.center ≠ c₂.center ∧ c₂.center ≠ c₃.center ∧ c₁.center ≠ c₃.center

/-- Distance between two points in 2D plane -/
def distance (p₁ p₂ : ℝ × ℝ) : ℝ := sorry

/-- The point from which all circles are seen at the same angle -/
def equal_angle_point (circles : ThreeCircles) (R : ℝ × ℝ) : Prop :=
  let O₁ := circles.c₁.center
  let O₂ := circles.c₂.center
  let O₃ := circles.c₃.center
  let r₁ := circles.c₁.radius
  let r₂ := circles.c₂.radius
  let r₃ := circles.c₃.radius
  (distance O₁ R / distance O₂ R = r₁ / r₂) ∧
  (distance O₂ R / distance O₃ R = r₂ / r₃) ∧
  (distance O₁ R / distance O₃ R = r₁ / r₃)

theorem equal_angle_point_exists (circles : ThreeCircles) :
  ∃ R : ℝ × ℝ, equal_angle_point circles R :=
sorry

end NUMINAMATH_CALUDE_equal_angle_point_exists_l3654_365477


namespace NUMINAMATH_CALUDE_average_difference_l3654_365474

theorem average_difference : 
  let m := (12 + 15 + 9 + 14 + 10) / 5
  let n := (24 + 8 + 8 + 12) / 4
  n - m = 1 := by
sorry

end NUMINAMATH_CALUDE_average_difference_l3654_365474


namespace NUMINAMATH_CALUDE_sunday_calorie_intake_l3654_365489

-- Define the calorie content for base meals
def breakfast_calories : ℝ := 500
def lunch_calories : ℝ := breakfast_calories * 1.25
def dinner_calories : ℝ := lunch_calories * 2
def snack_calories : ℝ := lunch_calories * 0.7
def morning_snack_calories : ℝ := breakfast_calories + 200
def afternoon_snack_calories : ℝ := lunch_calories * 0.8
def dessert_calories : ℝ := 350
def energy_drink_calories : ℝ := 220

-- Define the total calories for each day
def monday_calories : ℝ := breakfast_calories + lunch_calories + dinner_calories + snack_calories
def tuesday_calories : ℝ := breakfast_calories + morning_snack_calories + lunch_calories + afternoon_snack_calories + dinner_calories
def wednesday_calories : ℝ := breakfast_calories + lunch_calories + (dinner_calories * 0.85) + dessert_calories
def thursday_calories : ℝ := tuesday_calories
def friday_calories : ℝ := wednesday_calories + (2 * energy_drink_calories)
def weekend_calories : ℝ := tuesday_calories

-- Theorem to prove
theorem sunday_calorie_intake : weekend_calories = 3575 := by
  sorry

end NUMINAMATH_CALUDE_sunday_calorie_intake_l3654_365489


namespace NUMINAMATH_CALUDE_bricklayer_electrician_problem_l3654_365407

theorem bricklayer_electrician_problem :
  ∀ (bricklayer_rate electrician_rate total_pay bricklayer_hours : ℝ),
    bricklayer_rate = 12 →
    electrician_rate = 16 →
    total_pay = 1350 →
    bricklayer_hours = 67.5 →
    ∃ (electrician_hours : ℝ),
      electrician_hours = (total_pay - bricklayer_rate * bricklayer_hours) / electrician_rate ∧
      bricklayer_hours + electrician_hours = 101.25 :=
by sorry

end NUMINAMATH_CALUDE_bricklayer_electrician_problem_l3654_365407


namespace NUMINAMATH_CALUDE_martin_answered_40_l3654_365470

/-- The number of questions Campbell answered correctly -/
def campbell_correct : ℕ := 35

/-- The number of questions Kelsey answered correctly -/
def kelsey_correct : ℕ := campbell_correct + 8

/-- The number of questions Martin answered correctly -/
def martin_correct : ℕ := kelsey_correct - 3

/-- Theorem stating that Martin answered 40 questions correctly -/
theorem martin_answered_40 : martin_correct = 40 := by
  sorry

end NUMINAMATH_CALUDE_martin_answered_40_l3654_365470


namespace NUMINAMATH_CALUDE_probability_diamond_or_ace_l3654_365493

def standard_deck : ℕ := 52

def diamond_count : ℕ := 13

def ace_count : ℕ := 4

def favorable_outcomes : ℕ := diamond_count + ace_count - 1

theorem probability_diamond_or_ace :
  (favorable_outcomes : ℚ) / standard_deck = 4 / 13 :=
by sorry

end NUMINAMATH_CALUDE_probability_diamond_or_ace_l3654_365493


namespace NUMINAMATH_CALUDE_special_sequence_a10_l3654_365424

/-- A sequence with the property that for any p, q ∈ ℕ*, aₚ₊ₖ = aₚ · aₖ -/
def SpecialSequence (a : ℕ → ℕ) : Prop :=
  ∀ p q : ℕ, 0 < p → 0 < q → a (p + q) = a p * a q

theorem special_sequence_a10 (a : ℕ → ℕ) (h : SpecialSequence a) (h2 : a 2 = 4) :
  a 10 = 1024 := by
  sorry

end NUMINAMATH_CALUDE_special_sequence_a10_l3654_365424


namespace NUMINAMATH_CALUDE_range_of_k_for_inequality_l3654_365459

theorem range_of_k_for_inequality (k : ℝ) : 
  (∀ a b : ℝ, (a - b)^2 ≥ k * a * b) ↔ k ∈ Set.Icc (-4) 0 := by
  sorry

end NUMINAMATH_CALUDE_range_of_k_for_inequality_l3654_365459


namespace NUMINAMATH_CALUDE_middle_box_statement_l3654_365446

/-- Represents the two possible statements on a box. -/
inductive BoxStatement
  | NoPrizeHere
  | PrizeInNeighbor

/-- Represents a configuration of boxes with their statements. -/
def BoxConfiguration := Fin 23 → BoxStatement

/-- Checks if the given configuration is valid according to the problem rules. -/
def isValidConfiguration (config : BoxConfiguration) (prizeBox : Fin 23) : Prop :=
  -- Exactly one statement is true
  (∃! i, (config i = BoxStatement.NoPrizeHere ∧ i = prizeBox) ∨
         (config i = BoxStatement.PrizeInNeighbor ∧ (i + 1 = prizeBox ∨ i - 1 = prizeBox))) ∧
  -- The prize box exists
  (∃ i, i = prizeBox)

/-- The middle box index (0-based). -/
def middleBoxIndex : Fin 23 := ⟨11, by norm_num⟩

/-- The main theorem stating that the middle box must be labeled "The prize is in the neighboring box." -/
theorem middle_box_statement (config : BoxConfiguration) (prizeBox : Fin 23) 
    (h : isValidConfiguration config prizeBox) :
    config middleBoxIndex = BoxStatement.PrizeInNeighbor := by
  sorry


end NUMINAMATH_CALUDE_middle_box_statement_l3654_365446


namespace NUMINAMATH_CALUDE_first_stop_students_correct_l3654_365425

/-- The number of students who got on the bus at the first stop -/
def first_stop_students : ℕ := 39

/-- The number of students who got on the bus at the second stop -/
def second_stop_students : ℕ := 29

/-- The total number of students on the bus after the second stop -/
def total_students : ℕ := 68

/-- Theorem stating that the number of students who got on at the first stop is correct -/
theorem first_stop_students_correct :
  first_stop_students + second_stop_students = total_students := by
  sorry

end NUMINAMATH_CALUDE_first_stop_students_correct_l3654_365425


namespace NUMINAMATH_CALUDE_fraction_order_l3654_365409

theorem fraction_order (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (hac : a < c) (hbd : b > d) :
  a / b < (a + c) / (b + d) ∧ (a + c) / (b + d) < c / d ∧
  a / b < (a + c) / (b + d) ∧ (a + c) / (b + d) < (a + c) / (b - d) ∧
  (c - a) / (b + d) < (a + c) / (b + d) ∧ (a + c) / (b + d) < c / d ∧
  (c - a) / (b + d) < (a + c) / (b + d) ∧ (a + c) / (b + d) < (a + c) / (b - d) ∧
  (c - a) / (b + d) < (c - a) / (b - d) ∧ (c - a) / (b - d) < (a + c) / (b - d) :=
by sorry

end NUMINAMATH_CALUDE_fraction_order_l3654_365409


namespace NUMINAMATH_CALUDE_cost_of_melons_l3654_365462

/-- The cost of a single melon in dollars -/
def cost_per_melon : ℕ := 3

/-- The number of melons we want to calculate the cost for -/
def num_melons : ℕ := 6

/-- Theorem stating that the cost of 6 melons is $18 -/
theorem cost_of_melons : cost_per_melon * num_melons = 18 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_melons_l3654_365462


namespace NUMINAMATH_CALUDE_cone_volume_ratio_l3654_365448

/-- Two cones sharing a common base on a sphere -/
structure ConePair where
  R : ℝ  -- Radius of the sphere
  r : ℝ  -- Radius of the base of the cones
  h₁ : ℝ  -- Height of the first cone
  h₂ : ℝ  -- Height of the second cone

/-- The conditions of the problem -/
def ConePairConditions (cp : ConePair) : Prop :=
  cp.r^2 = 3 * cp.R^2 / 4 ∧  -- Area of base is 3/16 of sphere area
  cp.h₁ + cp.h₂ = 2 * cp.R ∧  -- Sum of heights equals diameter
  cp.r^2 + (cp.h₁ / 2)^2 = cp.R^2  -- Pythagorean theorem

/-- The theorem to be proved -/
theorem cone_volume_ratio (cp : ConePair) 
  (hc : ConePairConditions cp) : 
  cp.h₁ * cp.r^2 / (cp.h₂ * cp.r^2) = 1 / 3 :=
sorry

end NUMINAMATH_CALUDE_cone_volume_ratio_l3654_365448


namespace NUMINAMATH_CALUDE_original_number_proof_l3654_365466

theorem original_number_proof (N : ℝ) (x y z : ℝ) : 
  (N * 1.2 = 480) →
  ((480 * 0.85) * x^2 = 5*x^3 + 24*x - 50) →
  ((N / y) * 0.75 = z) →
  (z = x * y) →
  N = 400 := by
sorry

end NUMINAMATH_CALUDE_original_number_proof_l3654_365466


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l3654_365469

-- Define the inverse relationship between x and y
def inverse_relation (x y : ℝ) : Prop := ∃ k : ℝ, x * y^3 = k

-- Theorem statement
theorem inverse_variation_problem (x₁ x₂ y₁ y₂ : ℝ) 
  (h1 : inverse_relation x₁ y₁)
  (h2 : inverse_relation x₂ y₂)
  (h3 : x₁ = 8)
  (h4 : y₁ = 1)
  (h5 : y₂ = 2) :
  x₂ = 1 := by
sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l3654_365469


namespace NUMINAMATH_CALUDE_certain_number_equation_l3654_365497

theorem certain_number_equation : ∃ x : ℝ, 
  (3889 + x - 47.95000000000027 = 3854.002) ∧ 
  (x = 12.95200000000054) := by
  sorry

end NUMINAMATH_CALUDE_certain_number_equation_l3654_365497


namespace NUMINAMATH_CALUDE_ship_passengers_l3654_365498

theorem ship_passengers : ∃ (P : ℕ), 
  P > 0 ∧ 
  (P : ℚ) * (1/3 + 1/8 + 1/5 + 1/6) + 42 = P ∧ 
  P = 240 := by
  sorry

end NUMINAMATH_CALUDE_ship_passengers_l3654_365498


namespace NUMINAMATH_CALUDE_pizza_toppings_l3654_365437

theorem pizza_toppings (total_slices ham_slices pineapple_slices : ℕ) 
  (h_total : total_slices = 15)
  (h_ham : ham_slices = 9)
  (h_pineapple : pineapple_slices = 12)
  (h_at_least_one : ∀ slice, slice ≤ total_slices → (slice ≤ ham_slices ∨ slice ≤ pineapple_slices)) :
  ∃ both_toppings : ℕ, 
    both_toppings = ham_slices + pineapple_slices - total_slices ∧
    both_toppings = 6 := by
  sorry


end NUMINAMATH_CALUDE_pizza_toppings_l3654_365437


namespace NUMINAMATH_CALUDE_division_remainder_proof_l3654_365453

theorem division_remainder_proof (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ)
  (h1 : dividend = 176)
  (h2 : divisor = 14)
  (h3 : quotient = 12)
  (h4 : dividend = divisor * quotient + remainder) :
  remainder = 8 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_proof_l3654_365453


namespace NUMINAMATH_CALUDE_consecutive_four_plus_one_is_square_l3654_365432

theorem consecutive_four_plus_one_is_square (a : ℕ) (h : a ≥ 1) :
  a * (a + 1) * (a + 2) * (a + 3) + 1 = (a^2 + 3*a + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_four_plus_one_is_square_l3654_365432


namespace NUMINAMATH_CALUDE_exterior_angle_regular_hexagon_l3654_365400

theorem exterior_angle_regular_hexagon :
  let n : ℕ := 6  -- Number of sides in a hexagon
  let sum_interior_angles : ℝ := 180 * (n - 2)  -- Sum of interior angles formula
  let interior_angle : ℝ := sum_interior_angles / n  -- Each interior angle in a regular polygon
  let exterior_angle : ℝ := 180 - interior_angle  -- Exterior angle is supplementary to interior angle
  exterior_angle = 60 := by sorry

end NUMINAMATH_CALUDE_exterior_angle_regular_hexagon_l3654_365400


namespace NUMINAMATH_CALUDE_tank_capacity_l3654_365444

theorem tank_capacity : ∀ (x : ℚ), 
  (x / 8 + 90 = x / 2) → x = 240 := by
  sorry

end NUMINAMATH_CALUDE_tank_capacity_l3654_365444


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3654_365428

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), r > 0 ∧ ∀ n, a (n + 1) = r * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n, a n > 0) →
  a 1 + a 2 = 4/9 →
  a 3 + a 4 + a 5 + a 6 = 40 →
  (a 7 + a 8 + a 9) / 9 = 117 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3654_365428


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l3654_365423

/-- Given that x^2 and y vary inversely and are positive integers, 
    with y = 16 when x = 4, and z = x - y with z = 10 when y = 4, 
    prove that x = 1 when y = 256 -/
theorem inverse_variation_problem (x y z : ℕ+) (k : ℝ) : 
  (∀ (x y : ℕ+), (x:ℝ)^2 * y = k) →   -- x^2 and y vary inversely
  (4:ℝ)^2 * 16 = k →                  -- y = 16 when x = 4
  z = x - y →                         -- definition of z
  (∃ (x : ℕ+), z = 10 ∧ y = 4) →      -- z = 10 when y = 4
  (∃ (x : ℕ+), x = 1 ∧ y = 256) :=    -- to prove: x = 1 when y = 256
by sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l3654_365423


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_squared_l3654_365456

theorem arithmetic_geometric_mean_squared (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (a^2 + b^2) / 2 ≥ ((a + b) / 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_squared_l3654_365456


namespace NUMINAMATH_CALUDE_infinite_square_double_numbers_l3654_365483

/-- Definition of a double number -/
def is_double_number (x : ℕ) : Prop :=
  ∃ (d : ℕ), x = d * (10^(Nat.log 10 d + 1) + 1) ∧ d ≠ 0

/-- The main theorem -/
theorem infinite_square_double_numbers :
  ∀ k : ℕ, ∃ N : ℕ,
    let n := 21 * (1 + 14 * k)
    is_double_number (N * (10^n + 1)) ∧
    ∃ m : ℕ, N * (10^n + 1) = m^2 :=
by sorry

end NUMINAMATH_CALUDE_infinite_square_double_numbers_l3654_365483


namespace NUMINAMATH_CALUDE_max_gcd_of_sum_1089_l3654_365488

theorem max_gcd_of_sum_1089 (c d : ℕ+) (h : c + d = 1089) :
  (∃ (x y : ℕ+), x + y = 1089 ∧ Nat.gcd x y = 363) ∧
  (∀ (a b : ℕ+), a + b = 1089 → Nat.gcd a b ≤ 363) := by
  sorry

end NUMINAMATH_CALUDE_max_gcd_of_sum_1089_l3654_365488


namespace NUMINAMATH_CALUDE_max_min_x_plus_y_l3654_365412

theorem max_min_x_plus_y (x y : ℝ) (h : x^2 + y^2 - 4*x + 2*y + 2 = 0) :
  (∃ (a b : ℝ), (∀ (x' y' : ℝ), x'^2 + y'^2 - 4*x' + 2*y' + 2 = 0 → x' + y' ≤ a ∧ b ≤ x' + y') ∧
  a = 1 + Real.sqrt 6 ∧ b = 1 - Real.sqrt 6) :=
by sorry

end NUMINAMATH_CALUDE_max_min_x_plus_y_l3654_365412


namespace NUMINAMATH_CALUDE_ratio_problem_l3654_365427

theorem ratio_problem (N X : ℚ) (h1 : N / 2 = 150 / X) (h2 : N = 300) : X = 1 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l3654_365427


namespace NUMINAMATH_CALUDE_min_brown_eyes_and_lunch_box_l3654_365404

theorem min_brown_eyes_and_lunch_box 
  (total_students : ℕ) 
  (brown_eyes : ℕ) 
  (lunch_box : ℕ) 
  (h1 : total_students = 25) 
  (h2 : brown_eyes = 15) 
  (h3 : lunch_box = 18) :
  (brown_eyes + lunch_box - total_students : ℕ) ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_min_brown_eyes_and_lunch_box_l3654_365404


namespace NUMINAMATH_CALUDE_thirty_sided_polygon_diagonals_l3654_365431

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

/-- Theorem: A convex polygon with 30 sides has 405 diagonals -/
theorem thirty_sided_polygon_diagonals :
  num_diagonals 30 = 405 := by sorry

end NUMINAMATH_CALUDE_thirty_sided_polygon_diagonals_l3654_365431


namespace NUMINAMATH_CALUDE_max_small_boxes_in_large_box_l3654_365411

/-- Represents the dimensions of a box in centimeters -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℕ := d.length * d.width * d.height

/-- Converts meters to centimeters -/
def metersToCentimeters (m : ℕ) : ℕ := m * 100

/-- The dimensions of the large wooden box in meters -/
def largeBoxDimensionsMeters : BoxDimensions := {
  length := 8,
  width := 10,
  height := 6
}

/-- The dimensions of the large wooden box in centimeters -/
def largeBoxDimensionsCm : BoxDimensions := {
  length := metersToCentimeters largeBoxDimensionsMeters.length,
  width := metersToCentimeters largeBoxDimensionsMeters.width,
  height := metersToCentimeters largeBoxDimensionsMeters.height
}

/-- The dimensions of the small rectangular box in centimeters -/
def smallBoxDimensions : BoxDimensions := {
  length := 4,
  width := 5,
  height := 6
}

/-- Theorem: The maximum number of small boxes that can fit in the large box is 4,000,000 -/
theorem max_small_boxes_in_large_box :
  (boxVolume largeBoxDimensionsCm) / (boxVolume smallBoxDimensions) = 4000000 := by
  sorry

end NUMINAMATH_CALUDE_max_small_boxes_in_large_box_l3654_365411


namespace NUMINAMATH_CALUDE_dodecagon_pie_trim_l3654_365414

theorem dodecagon_pie_trim (d : ℝ) (h : d = 8) : ∃ (a b : ℤ),
  (π * (d / 2)^2 - 3 * (d / 2)^2 = a * π - b) ∧ (a + b = 64) := by
  sorry

end NUMINAMATH_CALUDE_dodecagon_pie_trim_l3654_365414


namespace NUMINAMATH_CALUDE_johns_skateboarding_distance_l3654_365484

/-- The total distance John skateboarded, given his journey details -/
def total_skateboarding_distance (initial_skate : ℝ) (walk : ℝ) : ℝ :=
  2 * (initial_skate + walk) - walk

/-- Theorem stating that John's total skateboarding distance is 24 miles -/
theorem johns_skateboarding_distance :
  total_skateboarding_distance 10 4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_johns_skateboarding_distance_l3654_365484
