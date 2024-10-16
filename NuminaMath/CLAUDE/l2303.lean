import Mathlib

namespace NUMINAMATH_CALUDE_prime_odd_sum_2009_l2303_230375

theorem prime_odd_sum_2009 :
  ∃! (a b : ℕ), Prime a ∧ Odd b ∧ a^2 + b = 2009 ∧ (a + b : ℕ) = 45 := by
  sorry

end NUMINAMATH_CALUDE_prime_odd_sum_2009_l2303_230375


namespace NUMINAMATH_CALUDE_shinyoung_read_most_l2303_230302

theorem shinyoung_read_most (shinyoung seokgi woong : ℚ) : 
  shinyoung = 1/3 ∧ seokgi = 1/4 ∧ woong = 1/5 → 
  shinyoung > seokgi ∧ shinyoung > woong := by
  sorry

end NUMINAMATH_CALUDE_shinyoung_read_most_l2303_230302


namespace NUMINAMATH_CALUDE_arithmetic_sum_l2303_230349

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sum (a : ℕ → ℝ) :
  arithmetic_sequence a → a 1 + a 13 = 10 →
  a 3 + a 5 + a 7 + a 9 + a 11 = 25 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sum_l2303_230349


namespace NUMINAMATH_CALUDE_quadratic_function_property_l2303_230348

theorem quadratic_function_property (a b c : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = a * x^2 + b * x + c)
  (h2 : f 0 = f 4)
  (h3 : f 0 > f 1) :
  a > 0 ∧ 4 * a + b = 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l2303_230348


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l2303_230369

-- Define the sets A and B
def A : Set Int := {-1, 1, 2, 4}
def B : Set Int := {-1, 0, 2}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {-1, 0, 1, 2, 4} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l2303_230369


namespace NUMINAMATH_CALUDE_set_A_range_l2303_230357

theorem set_A_range (a : ℝ) : 
  let A := {x : ℝ | a * x^2 - 3 * x - 4 = 0}
  (∀ x y : ℝ, x ∈ A → y ∈ A → x = y) → 
  (a ≤ -9/16 ∨ a = 0) := by
sorry

end NUMINAMATH_CALUDE_set_A_range_l2303_230357


namespace NUMINAMATH_CALUDE_zero_exists_in_interval_l2303_230374

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^3 + 3 * x - 3

-- State the theorem
theorem zero_exists_in_interval :
  ∃ c ∈ Set.Ioo (1/2 : ℝ) 1, f c = 0 :=
sorry

end NUMINAMATH_CALUDE_zero_exists_in_interval_l2303_230374


namespace NUMINAMATH_CALUDE_max_distinct_roots_special_polynomial_l2303_230327

/-- A polynomial with the property that the product of any two distinct roots is also a root -/
def SpecialPolynomial (P : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x ≠ y → P x = 0 → P y = 0 → P (x * y) = 0

/-- The maximum number of distinct real roots for a special polynomial is 4 -/
theorem max_distinct_roots_special_polynomial :
  ∃ (P : ℝ → ℝ), SpecialPolynomial P ∧
    (∃ (roots : Finset ℝ), (∀ x ∈ roots, P x = 0) ∧ roots.card = 4) ∧
    (∀ (Q : ℝ → ℝ), SpecialPolynomial Q →
      ∀ (roots : Finset ℝ), (∀ x ∈ roots, Q x = 0) → roots.card ≤ 4) :=
sorry

end NUMINAMATH_CALUDE_max_distinct_roots_special_polynomial_l2303_230327


namespace NUMINAMATH_CALUDE_perimeter_of_specific_rectangle_l2303_230324

/-- A figure that can be completed to form a rectangle -/
structure CompletableRectangle where
  length : ℝ
  width : ℝ

/-- The perimeter of a CompletableRectangle -/
def perimeter (r : CompletableRectangle) : ℝ :=
  2 * (r.length + r.width)

theorem perimeter_of_specific_rectangle :
  ∃ (r : CompletableRectangle), r.length = 6 ∧ r.width = 5 ∧ perimeter r = 22 :=
by sorry

end NUMINAMATH_CALUDE_perimeter_of_specific_rectangle_l2303_230324


namespace NUMINAMATH_CALUDE_chemical_reaction_results_l2303_230387

/-- Represents the chemical reaction between CaCO3 and HCl -/
structure ChemicalReaction where
  temperature : ℝ
  pressure : ℝ
  hcl_moles : ℝ
  cacl2_moles : ℝ
  co2_moles : ℝ
  h2o_moles : ℝ
  std_enthalpy_change : ℝ

/-- Calculates the amount of CaCO3 required and the change in enthalpy -/
def calculate_reaction_results (reaction : ChemicalReaction) : ℝ × ℝ :=
  sorry

/-- Theorem stating the correct results of the chemical reaction -/
theorem chemical_reaction_results :
  let reaction := ChemicalReaction.mk 25 1 4 2 2 2 (-178)
  let (caco3_grams, enthalpy_change) := calculate_reaction_results reaction
  caco3_grams = 200.18 ∧ enthalpy_change = -356 := by
  sorry

end NUMINAMATH_CALUDE_chemical_reaction_results_l2303_230387


namespace NUMINAMATH_CALUDE_restaurant_bill_calculation_l2303_230301

theorem restaurant_bill_calculation (adults children meal_cost : ℕ) 
  (h1 : adults = 2) 
  (h2 : children = 5) 
  (h3 : meal_cost = 3) : 
  (adults + children) * meal_cost = 21 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_bill_calculation_l2303_230301


namespace NUMINAMATH_CALUDE_spiral_staircase_handrail_length_l2303_230372

/-- The length of a spiral staircase handrail -/
theorem spiral_staircase_handrail_length 
  (turn : Real) -- angle of turn in degrees
  (rise : Real) -- height of staircase in feet
  (radius : Real) -- radius of staircase in feet
  (h1 : turn = 450)
  (h2 : rise = 15)
  (h3 : radius = 4) :
  ∃ (length : Real), 
    abs (length - Real.sqrt (rise^2 + (turn / 360 * 2 * Real.pi * radius)^2)) < 0.1 ∧ 
    abs (length - 17.4) < 0.1 :=
by sorry

end NUMINAMATH_CALUDE_spiral_staircase_handrail_length_l2303_230372


namespace NUMINAMATH_CALUDE_expected_sticky_corn_l2303_230380

theorem expected_sticky_corn (total_corn : ℕ) (sticky_corn : ℕ) (sample_size : ℕ) :
  total_corn = 140 →
  sticky_corn = 56 →
  sample_size = 40 →
  (sample_size * sticky_corn) / total_corn = 16 := by
  sorry

end NUMINAMATH_CALUDE_expected_sticky_corn_l2303_230380


namespace NUMINAMATH_CALUDE_sum_of_integers_50_to_75_l2303_230320

def sum_of_integers (a b : ℕ) : ℕ := (b - a + 1) * (a + b) / 2

theorem sum_of_integers_50_to_75 : sum_of_integers 50 75 = 1625 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_50_to_75_l2303_230320


namespace NUMINAMATH_CALUDE_largest_prime_divisor_of_13_squared_plus_84_squared_l2303_230376

theorem largest_prime_divisor_of_13_squared_plus_84_squared : 
  (Nat.factors (13^2 + 84^2)).maximum = some 17 := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_of_13_squared_plus_84_squared_l2303_230376


namespace NUMINAMATH_CALUDE_belinda_age_difference_l2303_230339

/-- Given the ages of Tony and Belinda, prove that Belinda's age is 8 years more than twice Tony's age. -/
theorem belinda_age_difference (tony_age belinda_age : ℕ) : 
  tony_age = 16 →
  belinda_age = 40 →
  tony_age + belinda_age = 56 →
  belinda_age > 2 * tony_age →
  belinda_age - 2 * tony_age = 8 := by
sorry

end NUMINAMATH_CALUDE_belinda_age_difference_l2303_230339


namespace NUMINAMATH_CALUDE_rectangle_area_preservation_l2303_230352

theorem rectangle_area_preservation (L W : ℝ) (h : L > 0 ∧ W > 0) :
  ∃ x : ℝ, x > 0 ∧ x < 100 ∧
  (L * (1 - x / 100)) * (W * 1.25) = L * W ∧
  x = 20 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_preservation_l2303_230352


namespace NUMINAMATH_CALUDE_polar_to_rectangular_conversion_l2303_230342

theorem polar_to_rectangular_conversion :
  let r : ℝ := 5
  let θ : ℝ := 5 * Real.pi / 4
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  (x = -5 * Real.sqrt 2 / 2) ∧ (y = -5 * Real.sqrt 2 / 2) := by sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_conversion_l2303_230342


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2303_230385

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - 2*x + 4 ≤ 0) ↔ (∃ x : ℝ, x^2 - 2*x + 4 > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2303_230385


namespace NUMINAMATH_CALUDE_polynomial_sum_l2303_230364

/-- Given a polynomial P such that P + (x^2 - y^2) = x^2 + y^2, then P = 2y^2 -/
theorem polynomial_sum (x y : ℝ) (P : ℝ → ℝ) :
  (∀ x, P x + (x^2 - y^2) = x^2 + y^2) → P x = 2 * y^2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_sum_l2303_230364


namespace NUMINAMATH_CALUDE_geometry_propositions_l2303_230341

-- Define the basic types
variable (Point : Type) (Line : Type) (Plane : Type)

-- Define the basic relations
variable (parallel_line : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (perpendicular_plane : Plane → Plane → Prop)

-- Theorem statement
theorem geometry_propositions 
  (m n : Line) (α β : Plane) :
  (∀ m α β, parallel_line_plane m α → perpendicular_line_plane m β → perpendicular_plane α β) ∧
  (∀ m n α, parallel_line m n → perpendicular_line_plane m α → perpendicular_line_plane n α) :=
sorry

end NUMINAMATH_CALUDE_geometry_propositions_l2303_230341


namespace NUMINAMATH_CALUDE_stella_monthly_income_l2303_230345

def months_in_year : ℕ := 12
def unpaid_leave_months : ℕ := 2
def annual_income : ℕ := 49190

def monthly_income : ℕ := annual_income / (months_in_year - unpaid_leave_months)

theorem stella_monthly_income : monthly_income = 4919 := by
  sorry

end NUMINAMATH_CALUDE_stella_monthly_income_l2303_230345


namespace NUMINAMATH_CALUDE_necessary_and_sufficient_l2303_230377

theorem necessary_and_sufficient (p q : Prop) : 
  (p → q) → (q → p) → (p ↔ q) := by
  sorry

end NUMINAMATH_CALUDE_necessary_and_sufficient_l2303_230377


namespace NUMINAMATH_CALUDE_cube_sum_and_reciprocal_l2303_230326

theorem cube_sum_and_reciprocal (x : ℝ) (h : x + 1/x = -3) : x^3 + 1/x^3 = -18 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_and_reciprocal_l2303_230326


namespace NUMINAMATH_CALUDE_lg_equation_l2303_230305

-- Define lg as the base-10 logarithm
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem lg_equation : (lg 5)^2 + lg 2 * lg 50 = 1 := by
  sorry

end NUMINAMATH_CALUDE_lg_equation_l2303_230305


namespace NUMINAMATH_CALUDE_ratio_problem_l2303_230346

theorem ratio_problem (x y : ℚ) (h : (3 * x - 2 * y) / (2 * x + y) = 3 / 4) :
  x / y = 11 / 6 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l2303_230346


namespace NUMINAMATH_CALUDE_odd_integer_m_exists_l2303_230325

def g (n : ℤ) : ℤ :=
  if n % 2 = 1 then n + 5 else n / 3

theorem odd_integer_m_exists (m : ℤ) (h_odd : m % 2 = 1) (h_g : g (g (g m)) = 14) : m = 121 := by
  sorry

end NUMINAMATH_CALUDE_odd_integer_m_exists_l2303_230325


namespace NUMINAMATH_CALUDE_courtyard_paving_l2303_230388

/-- The length of the courtyard in meters -/
def courtyard_length : ℝ := 25

/-- The width of the courtyard in meters -/
def courtyard_width : ℝ := 20

/-- The length of a brick in meters -/
def brick_length : ℝ := 0.15

/-- The width of a brick in meters -/
def brick_width : ℝ := 0.08

/-- The total number of bricks required to cover the courtyard -/
def total_bricks : ℕ := 41667

theorem courtyard_paving :
  ⌈(courtyard_length * courtyard_width) / (brick_length * brick_width)⌉ = total_bricks := by
  sorry

end NUMINAMATH_CALUDE_courtyard_paving_l2303_230388


namespace NUMINAMATH_CALUDE_stating_table_tennis_sequences_count_l2303_230351

/-- Represents the number of possible game sequences in a table tennis match --/
def table_tennis_sequences : ℕ := 20

/-- 
Theorem stating that the number of possible game sequences in a table tennis match,
where the first player to win three games wins the match, is exactly 20.
--/
theorem table_tennis_sequences_count : table_tennis_sequences = 20 := by
  sorry

end NUMINAMATH_CALUDE_stating_table_tennis_sequences_count_l2303_230351


namespace NUMINAMATH_CALUDE_triangle_inequality_violation_l2303_230307

/-- Theorem: A triangle cannot be formed with side lengths 9, 4, and 3. -/
theorem triangle_inequality_violation (a b c : ℝ) 
  (ha : a = 9) (hb : b = 4) (hc : c = 3) : 
  ¬(a + b > c ∧ a + c > b ∧ b + c > a) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_violation_l2303_230307


namespace NUMINAMATH_CALUDE_greatest_four_digit_divisible_l2303_230370

/-- Reverses the digits of a positive integer -/
def reverseDigits (n : ℕ) : ℕ := sorry

/-- Checks if a natural number is a four-digit integer -/
def isFourDigit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

theorem greatest_four_digit_divisible :
  ∃ (p : ℕ),
    isFourDigit p ∧
    isFourDigit (reverseDigits p) ∧
    p % 99 = 0 ∧
    (reverseDigits p) % 99 = 0 ∧
    p % 7 = 0 ∧
    p = 7623 ∧
    ∀ (q : ℕ),
      isFourDigit q ∧
      isFourDigit (reverseDigits q) ∧
      q % 99 = 0 ∧
      (reverseDigits q) % 99 = 0 ∧
      q % 7 = 0 →
      q ≤ 7623 :=
by sorry

end NUMINAMATH_CALUDE_greatest_four_digit_divisible_l2303_230370


namespace NUMINAMATH_CALUDE_max_value_of_trig_function_l2303_230344

theorem max_value_of_trig_function (a b : ℝ) :
  (∀ x : ℝ, a * Real.cos x + b ≤ 1) →
  (∀ x : ℝ, a * Real.cos x + b ≥ -7) →
  (∃ x : ℝ, a * Real.cos x + b = 1) →
  (∃ x : ℝ, a * Real.cos x + b = -7) →
  (∀ x : ℝ, a * Real.cos x + b * Real.sin x ≤ 5) ∧
  (∃ x : ℝ, a * Real.cos x + b * Real.sin x = 5) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_trig_function_l2303_230344


namespace NUMINAMATH_CALUDE_triangle_inequality_l2303_230335

-- Define a triangle ABC in 2D space
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define a point M
def Point := ℝ × ℝ

-- Function to calculate distance between two points
def distance (p q : ℝ × ℝ) : ℝ := sorry

-- Function to check if a point is inside a triangle
def isInside (t : Triangle) (p : Point) : Prop := sorry

-- Function to calculate the perimeter of a triangle
def perimeter (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem triangle_inequality (t : Triangle) (M : Point) 
  (h : isInside t M) : 
  min (distance M t.A) (min (distance M t.B) (distance M t.C)) + 
  distance M t.A + distance M t.B + distance M t.C < 
  perimeter t := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2303_230335


namespace NUMINAMATH_CALUDE_salary_comparison_l2303_230392

theorem salary_comparison (a b : ℝ) (h : a = 0.8 * b) :
  (b - a) / a * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_salary_comparison_l2303_230392


namespace NUMINAMATH_CALUDE_soccer_boys_percentage_l2303_230399

theorem soccer_boys_percentage (total_students boys soccer_players girls_not_playing : ℕ) : 
  total_students = 420 →
  boys = 312 →
  soccer_players = 250 →
  girls_not_playing = 63 →
  (boys - (total_students - boys - girls_not_playing)) / soccer_players * 100 = 82 := by
sorry

end NUMINAMATH_CALUDE_soccer_boys_percentage_l2303_230399


namespace NUMINAMATH_CALUDE_same_row_exists_l2303_230389

/-- Represents a seating arrangement for a class session -/
def SeatingArrangement := Fin 50 → Fin 7

theorem same_row_exists (morning afternoon : SeatingArrangement) : 
  ∃ (s1 s2 : Fin 50), s1 ≠ s2 ∧ morning s1 = morning s2 ∧ afternoon s1 = afternoon s2 := by
  sorry

end NUMINAMATH_CALUDE_same_row_exists_l2303_230389


namespace NUMINAMATH_CALUDE_z_in_second_quadrant_l2303_230303

-- Define the complex number z
def z : ℂ := sorry

-- Define the condition z / (1 + i) = 2i
axiom z_condition : z / (1 + Complex.I) = 2 * Complex.I

-- Define the second quadrant
def second_quadrant (w : ℂ) : Prop :=
  w.re < 0 ∧ w.im > 0

-- Theorem statement
theorem z_in_second_quadrant : second_quadrant z := by
  sorry

end NUMINAMATH_CALUDE_z_in_second_quadrant_l2303_230303


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2303_230362

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x ↦ x^2 - 2*x
  ∃ x₁ x₂ : ℝ, x₁ = 0 ∧ x₂ = 2 ∧ (∀ x : ℝ, f x = 0 ↔ x = x₁ ∨ x = x₂) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2303_230362


namespace NUMINAMATH_CALUDE_optimal_plan_is_most_cost_effective_l2303_230338

/-- Represents a sewage treatment equipment purchase plan -/
structure PurchasePlan where
  modelA : ℕ
  modelB : ℕ

/-- Checks if a purchase plan is valid according to the given conditions -/
def isValidPlan (p : PurchasePlan) : Prop :=
  p.modelA + p.modelB = 10 ∧
  12 * p.modelA + 10 * p.modelB ≤ 105 ∧
  240 * p.modelA + 200 * p.modelB ≥ 2040

/-- Calculates the total cost of a purchase plan -/
def totalCost (p : PurchasePlan) : ℕ :=
  12 * p.modelA + 10 * p.modelB

/-- The optimal purchase plan -/
def optimalPlan : PurchasePlan :=
  { modelA := 1, modelB := 9 }

/-- Theorem stating that the optimal plan is the most cost-effective valid plan -/
theorem optimal_plan_is_most_cost_effective :
  isValidPlan optimalPlan ∧
  ∀ p : PurchasePlan, isValidPlan p → totalCost optimalPlan ≤ totalCost p :=
sorry

end NUMINAMATH_CALUDE_optimal_plan_is_most_cost_effective_l2303_230338


namespace NUMINAMATH_CALUDE_solution_characterization_l2303_230378

/-- The set of solutions to the equation x y z + x y + y z + z x + x + y + z = 1977 --/
def SolutionSet : Set (ℕ × ℕ × ℕ) :=
  {(1, 22, 42), (1, 42, 22), (22, 1, 42), (22, 42, 1), (42, 1, 22), (42, 22, 1)}

/-- The equation x y z + x y + y z + z x + x + y + z = 1977 --/
def SatisfiesEquation (x y z : ℕ) : Prop :=
  x * y * z + x * y + y * z + z * x + x + y + z = 1977

theorem solution_characterization :
  ∀ x y z : ℕ, (x > 0 ∧ y > 0 ∧ z > 0) →
    (SatisfiesEquation x y z ↔ (x, y, z) ∈ SolutionSet) := by
  sorry

end NUMINAMATH_CALUDE_solution_characterization_l2303_230378


namespace NUMINAMATH_CALUDE_symmetric_points_range_l2303_230366

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then 2 * x^2 - 3 * x else a / Real.exp x

theorem symmetric_points_range (a : ℝ) :
  (∃ x y : ℝ, x > 0 ∧ y < 0 ∧ f a x = f a (-y)) →
  -Real.exp (-1/2) ≤ a ∧ a ≤ 9 * Real.exp (-3) :=
sorry

end NUMINAMATH_CALUDE_symmetric_points_range_l2303_230366


namespace NUMINAMATH_CALUDE_article_cost_price_l2303_230329

/-- Given an article with a 15% markup, sold at Rs. 456 after a 26.570048309178745% discount,
    prove that the cost price of the article is Rs. 540. -/
theorem article_cost_price (markup_percentage : ℝ) (selling_price : ℝ) (discount_percentage : ℝ)
    (h1 : markup_percentage = 15)
    (h2 : selling_price = 456)
    (h3 : discount_percentage = 26.570048309178745) :
    ∃ (cost_price : ℝ),
      cost_price * (1 + markup_percentage / 100) * (1 - discount_percentage / 100) = selling_price ∧
      cost_price = 540 := by
  sorry

end NUMINAMATH_CALUDE_article_cost_price_l2303_230329


namespace NUMINAMATH_CALUDE_stating_second_team_completes_in_45_days_l2303_230395

/-- Represents the number of days in the original plan -/
def original_days : ℕ := 30

/-- Represents the number of days worked before the change -/
def days_worked : ℕ := 10

/-- Represents the number of days the project needed to be completed earlier -/
def days_earlier : ℕ := 8

/-- Represents the number of days it would take the second team to complete the project alone -/
def second_team_days : ℕ := 45

/-- Represents the fraction of work completed by the first team before the change -/
def work_completed : ℚ := days_worked / original_days

/-- Represents the remaining work to be completed -/
def remaining_work : ℚ := 1 - work_completed

/-- Represents the remaining days after the change -/
def remaining_days : ℕ := original_days - days_worked - days_earlier

/-- 
Theorem stating that given the conditions, the second team would take 45 days to complete the project alone
-/
theorem second_team_completes_in_45_days :
  (1 / second_team_days + 1 / original_days) * remaining_days = remaining_work :=
sorry

end NUMINAMATH_CALUDE_stating_second_team_completes_in_45_days_l2303_230395


namespace NUMINAMATH_CALUDE_function_fits_data_l2303_230393

/-- The set of data points representing the relationship between x and y -/
def data_points : List (ℚ × ℚ) := [(0, 200), (2, 160), (4, 80), (6, 0), (8, -120)]

/-- The proposed quadratic function -/
def f (x : ℚ) : ℚ := -10 * x^2 + 200

/-- Theorem stating that the proposed function fits all data points -/
theorem function_fits_data : ∀ (point : ℚ × ℚ), point ∈ data_points → f point.1 = point.2 := by
  sorry

end NUMINAMATH_CALUDE_function_fits_data_l2303_230393


namespace NUMINAMATH_CALUDE_fish_catch_theorem_l2303_230306

def mike_rate : ℕ := 30
def jim_rate : ℕ := 2 * mike_rate
def bob_rate : ℕ := (3 * jim_rate) / 2

def total_fish_caught (mike_rate jim_rate bob_rate : ℕ) : ℕ :=
  let fish_40_min := (mike_rate + jim_rate + bob_rate) * 2 / 3
  let fish_20_min := jim_rate * 1 / 3
  fish_40_min + fish_20_min

theorem fish_catch_theorem :
  total_fish_caught mike_rate jim_rate bob_rate = 140 := by
  sorry

end NUMINAMATH_CALUDE_fish_catch_theorem_l2303_230306


namespace NUMINAMATH_CALUDE_stairs_climbed_together_l2303_230310

theorem stairs_climbed_together (jonny_stairs : ℕ) (julia_stairs : ℕ) : 
  jonny_stairs = 1269 →
  julia_stairs = jonny_stairs / 3 - 7 →
  jonny_stairs + julia_stairs = 1685 := by
sorry

end NUMINAMATH_CALUDE_stairs_climbed_together_l2303_230310


namespace NUMINAMATH_CALUDE_polygon_diagonals_l2303_230394

/-- 
For an n-sided polygon, if 6 diagonals can be drawn from a single vertex, then n = 9.
-/
theorem polygon_diagonals (n : ℕ) : (n - 3 = 6) → n = 9 := by
  sorry

end NUMINAMATH_CALUDE_polygon_diagonals_l2303_230394


namespace NUMINAMATH_CALUDE_decorations_count_l2303_230397

/-- The total number of decorations Danai will put up -/
def total_decorations (skulls broomsticks spiderwebs cauldron more_to_buy left_to_put_up : ℕ) : ℕ :=
  skulls + broomsticks + spiderwebs + (2 * spiderwebs) + cauldron + more_to_buy + left_to_put_up

/-- Theorem stating the total number of decorations -/
theorem decorations_count :
  total_decorations 12 4 12 1 20 10 = 83 := by
  sorry


end NUMINAMATH_CALUDE_decorations_count_l2303_230397


namespace NUMINAMATH_CALUDE_solve_linear_equation_l2303_230354

/-- Given an equation mx + ny = 6 with two known solutions, prove that m = 4 and n = 2 -/
theorem solve_linear_equation (m n : ℝ) : 
  (m * 1 + n * 1 = 6) → 
  (m * 2 + n * (-1) = 6) → 
  (m = 4 ∧ n = 2) := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l2303_230354


namespace NUMINAMATH_CALUDE_smallest_triple_consecutive_sum_l2303_230316

def sum_of_consecutive (n : ℕ) (k : ℕ) : ℕ := 
  k * n + k * (k - 1) / 2

def is_sum_of_consecutive (x : ℕ) (k : ℕ) : Prop :=
  ∃ n : ℕ, sum_of_consecutive n k = x

theorem smallest_triple_consecutive_sum : 
  (∀ m : ℕ, m < 105 → ¬(is_sum_of_consecutive m 5 ∧ is_sum_of_consecutive m 6 ∧ is_sum_of_consecutive m 7)) ∧ 
  (is_sum_of_consecutive 105 5 ∧ is_sum_of_consecutive 105 6 ∧ is_sum_of_consecutive 105 7) :=
sorry

end NUMINAMATH_CALUDE_smallest_triple_consecutive_sum_l2303_230316


namespace NUMINAMATH_CALUDE_matrix_det_minus_two_l2303_230328

theorem matrix_det_minus_two (A : Matrix (Fin 2) (Fin 2) ℤ) :
  A = ![![9, 5], ![-3, 4]] →
  Matrix.det A - 2 = 49 := by
  sorry

end NUMINAMATH_CALUDE_matrix_det_minus_two_l2303_230328


namespace NUMINAMATH_CALUDE_unique_solution_for_prime_power_equation_l2303_230386

theorem unique_solution_for_prime_power_equation :
  ∀ m p x : ℕ,
    Prime p →
    2^m * p^2 + 27 = x^3 →
    m = 1 ∧ p = 7 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_for_prime_power_equation_l2303_230386


namespace NUMINAMATH_CALUDE_probability_penny_dime_halfdollar_heads_l2303_230330

/-- Represents the outcome of a coin flip -/
inductive CoinFlip
| Heads
| Tails

/-- Represents the set of coins being flipped -/
structure CoinSet :=
  (penny : CoinFlip)
  (nickel : CoinFlip)
  (dime : CoinFlip)
  (quarter : CoinFlip)
  (half_dollar : CoinFlip)
  (dollar : CoinFlip)

/-- The total number of possible outcomes when flipping six coins -/
def total_outcomes : ℕ := 64

/-- Predicate for the desired outcome (penny, dime, and half-dollar are heads) -/
def desired_outcome (cs : CoinSet) : Prop :=
  cs.penny = CoinFlip.Heads ∧ cs.dime = CoinFlip.Heads ∧ cs.half_dollar = CoinFlip.Heads

/-- The number of outcomes satisfying the desired condition -/
def favorable_outcomes : ℕ := 8

/-- Theorem stating the probability of the desired outcome -/
theorem probability_penny_dime_halfdollar_heads :
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 8 := by
  sorry


end NUMINAMATH_CALUDE_probability_penny_dime_halfdollar_heads_l2303_230330


namespace NUMINAMATH_CALUDE_thirteen_rectangles_l2303_230347

/-- A rectangle with integer side lengths. -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Checks if a rectangle meets the given criteria. -/
def meetsConditions (rect : Rectangle) : Prop :=
  rect.width > 0 ∧ rect.height > 0 ∧
  2 * (rect.width + rect.height) = 80 ∧
  ∃ k : ℕ, rect.width = 3 * k

/-- Two rectangles are considered congruent if they have the same dimensions (ignoring orientation). -/
def areCongruent (rect1 rect2 : Rectangle) : Prop :=
  (rect1.width = rect2.width ∧ rect1.height = rect2.height) ∨
  (rect1.width = rect2.height ∧ rect1.height = rect2.width)

/-- The main theorem stating that there are exactly 13 non-congruent rectangles meeting the conditions. -/
theorem thirteen_rectangles :
  ∃ (rectangles : Finset Rectangle),
    rectangles.card = 13 ∧
    (∀ rect ∈ rectangles, meetsConditions rect) ∧
    (∀ rect, meetsConditions rect → ∃ unique_rect ∈ rectangles, areCongruent rect unique_rect) :=
  sorry

end NUMINAMATH_CALUDE_thirteen_rectangles_l2303_230347


namespace NUMINAMATH_CALUDE_base4_10203_equals_291_l2303_230368

def base4_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (4 ^ i)) 0

theorem base4_10203_equals_291 :
  base4_to_decimal [3, 0, 2, 0, 1] = 291 := by
  sorry

end NUMINAMATH_CALUDE_base4_10203_equals_291_l2303_230368


namespace NUMINAMATH_CALUDE_sequence_bound_l2303_230396

theorem sequence_bound (k : ℕ) (h_k : k > 0) : 
  (∃ (a : ℕ → ℚ), a 0 = 1 / k ∧ 
    (∀ n : ℕ, n > 0 → a n = a (n - 1) + (1 : ℚ) / (n ^ 2 : ℚ) * (a (n - 1)) ^ 2) ∧
    (∀ n : ℕ, n > 0 → a n < 1)) ↔ 
  k ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_sequence_bound_l2303_230396


namespace NUMINAMATH_CALUDE_digit_deletion_divisibility_l2303_230331

theorem digit_deletion_divisibility (d : ℕ) (h : d > 0) : 
  ∃ (n n1 k a b c : ℕ), 
    n = 10^k * (10*a + b) + c ∧
    n1 = 10^k * a + c ∧
    0 < b ∧ b < 10 ∧
    c < 10^k ∧
    d ∣ n ∧
    d ∣ n1 :=
sorry

end NUMINAMATH_CALUDE_digit_deletion_divisibility_l2303_230331


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2303_230350

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 + x - 6 ≤ 0} = {x : ℝ | -3 ≤ x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2303_230350


namespace NUMINAMATH_CALUDE_symmetric_point_coordinates_l2303_230343

/-- A point in a 2D plane. -/
structure Point where
  x : ℝ
  y : ℝ

/-- Given two points, checks if they are symmetric with respect to the origin. -/
def symmetricWrtOrigin (p q : Point) : Prop :=
  q.x = -p.x ∧ q.y = -p.y

/-- The theorem stating that the point (-1, 2) is symmetric to (1, -2) with respect to the origin. -/
theorem symmetric_point_coordinates :
  let p : Point := ⟨1, -2⟩
  let q : Point := ⟨-1, 2⟩
  symmetricWrtOrigin p q := by sorry

end NUMINAMATH_CALUDE_symmetric_point_coordinates_l2303_230343


namespace NUMINAMATH_CALUDE_exam_failure_marks_l2303_230336

theorem exam_failure_marks (T : ℕ) (passing_mark : ℕ) : 
  (60 * T / 100 - 20 = passing_mark) →
  (passing_mark = 160) →
  (passing_mark - 40 * T / 100 = 40) :=
by sorry

end NUMINAMATH_CALUDE_exam_failure_marks_l2303_230336


namespace NUMINAMATH_CALUDE_expand_and_simplify_l2303_230311

theorem expand_and_simplify (x : ℝ) (h : x ≠ 0) :
  (3 / 7) * (7 / x^2 + 15 * x^3 - 4 * x) = 3 / x^2 + 45 * x^3 / 7 - 12 * x / 7 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l2303_230311


namespace NUMINAMATH_CALUDE_floor_coverage_l2303_230384

/-- A type representing a rectangular floor -/
structure RectangularFloor where
  m : ℕ
  n : ℕ
  h_m : m > 3
  h_n : n > 3

/-- A predicate that determines if a floor can be fully covered by 2x4 tiles -/
def canBeCovered (floor : RectangularFloor) : Prop :=
  floor.m % 2 = 0 ∧ floor.n % 2 = 0

/-- Theorem stating that a rectangular floor can be fully covered by 2x4 tiles 
    if and only if both dimensions are even -/
theorem floor_coverage (floor : RectangularFloor) :
  canBeCovered floor ↔ (floor.m % 2 = 0 ∧ floor.n % 2 = 0) := by
  sorry

#check floor_coverage

end NUMINAMATH_CALUDE_floor_coverage_l2303_230384


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l2303_230323

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_product (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n : ℕ, a n > 0) →
  a 4 * a 8 = 4 →
  a 5 * a 6 * a 7 = 8 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l2303_230323


namespace NUMINAMATH_CALUDE_tims_change_l2303_230371

/-- Represents the bread purchase scenario -/
structure BreadPurchase where
  loaves : ℕ
  slices_per_loaf : ℕ
  cost_per_slice : ℚ
  payment : ℚ

/-- Calculates the change received in a bread purchase -/
def calculate_change (purchase : BreadPurchase) : ℚ :=
  purchase.payment - (purchase.loaves * purchase.slices_per_loaf * purchase.cost_per_slice)

/-- Theorem: Tim's change is $16.00 -/
theorem tims_change (purchase : BreadPurchase) 
  (h1 : purchase.loaves = 3)
  (h2 : purchase.slices_per_loaf = 20)
  (h3 : purchase.cost_per_slice = 40/100)
  (h4 : purchase.payment = 40) :
  calculate_change purchase = 16 := by
  sorry

end NUMINAMATH_CALUDE_tims_change_l2303_230371


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l2303_230367

theorem simplify_sqrt_expression :
  Real.sqrt (75 - 30 * Real.sqrt 5) = 5 - 3 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l2303_230367


namespace NUMINAMATH_CALUDE_xyz_ratio_l2303_230321

theorem xyz_ratio (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0)
  (diff_xy : x ≠ y) (diff_xz : x ≠ z) (diff_yz : y ≠ z)
  (eq1 : y / (x - z) = (x + y) / z)
  (eq2 : (x + y) / z = x / y) : 
  x / y = 2 := by sorry

end NUMINAMATH_CALUDE_xyz_ratio_l2303_230321


namespace NUMINAMATH_CALUDE_circle_properties_l2303_230313

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := (x + 1)^2 + (y - 2)^2 = 4

-- Define the center coordinates
def center : ℝ × ℝ := (-1, 2)

-- Define the radius
def radius : ℝ := 2

-- Theorem statement
theorem circle_properties :
  (∀ x y : ℝ, circle_equation x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2) :=
by sorry

end NUMINAMATH_CALUDE_circle_properties_l2303_230313


namespace NUMINAMATH_CALUDE_inner_circle_radius_l2303_230308

/-- The radius of the inner tangent circle in a rectangle with semicircles --/
theorem inner_circle_radius (length width : ℝ) (h_length : length = 4) (h_width : width = 2) :
  let semicircle_radius := length / 8
  let center_to_semicircle := (3 * length / 8)^2 + (width / 2)^2
  (Real.sqrt center_to_semicircle / 2) - semicircle_radius = (Real.sqrt 10 - 1) / 2 :=
sorry

end NUMINAMATH_CALUDE_inner_circle_radius_l2303_230308


namespace NUMINAMATH_CALUDE_special_hyperbola_equation_l2303_230356

/-- A hyperbola with a specific property -/
structure SpecialHyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0
  h_equation : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 → (y = 2*x - 4 → ∃! p : ℝ × ℝ, p.1^2 / a^2 - p.2^2 / b^2 = 1 ∧ p.2 = 2*p.1 - 4)

/-- The theorem about the special hyperbola -/
theorem special_hyperbola_equation (h : SpecialHyperbola) : 
  h.a^2 = 4/5 ∧ h.b^2 = 16/5 :=
sorry

end NUMINAMATH_CALUDE_special_hyperbola_equation_l2303_230356


namespace NUMINAMATH_CALUDE_three_circles_arrangement_l2303_230332

/-- A circle in a plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The intersection points of two circles --/
def intersectionPoints (c1 c2 : Circle) : Set (ℝ × ℝ) := sorry

/-- Three circles have only two common points --/
def haveOnlyTwoCommonPoints (c1 c2 c3 : Circle) : Prop :=
  ∃ (p q : ℝ × ℝ), p ≠ q ∧
    intersectionPoints c1 c2 ∩ intersectionPoints c2 c3 ∩ intersectionPoints c1 c3 = {p, q}

/-- All three circles intersect at the same two points --/
def allIntersectAtSamePoints (c1 c2 c3 : Circle) : Prop :=
  ∃ (p q : ℝ × ℝ), p ≠ q ∧
    intersectionPoints c1 c2 = {p, q} ∧
    intersectionPoints c2 c3 = {p, q} ∧
    intersectionPoints c1 c3 = {p, q}

/-- One circle intersects each of the other two circles at two distinct points --/
def oneIntersectsOthersAtDistinctPoints (c1 c2 c3 : Circle) : Prop :=
  ∃ (p q : ℝ × ℝ), p ≠ q ∧
    ((intersectionPoints c1 c2 = {p, q} ∧ intersectionPoints c1 c3 = {p, q}) ∨
     (intersectionPoints c2 c1 = {p, q} ∧ intersectionPoints c2 c3 = {p, q}) ∨
     (intersectionPoints c3 c1 = {p, q} ∧ intersectionPoints c3 c2 = {p, q}))

/-- The main theorem --/
theorem three_circles_arrangement (c1 c2 c3 : Circle) :
  haveOnlyTwoCommonPoints c1 c2 c3 →
  allIntersectAtSamePoints c1 c2 c3 ∨ oneIntersectsOthersAtDistinctPoints c1 c2 c3 := by
  sorry


end NUMINAMATH_CALUDE_three_circles_arrangement_l2303_230332


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2303_230340

theorem complex_equation_solution (z : ℂ) : 
  z * (1 + Complex.I) = Complex.abs (1 - Complex.I * Real.sqrt 3) →
  z = Real.sqrt 2 * (Complex.cos (Real.pi / 4) - Complex.I * Complex.sin (Real.pi / 4)) :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2303_230340


namespace NUMINAMATH_CALUDE_quadratic_function_bound_l2303_230363

theorem quadratic_function_bound (a b : ℝ) : 
  let f : ℝ → ℝ := fun x ↦ x^2 + a*x + b
  (max (|f 1|) (max (|f 2|) (|f 3|))) ≥ (1/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_bound_l2303_230363


namespace NUMINAMATH_CALUDE_max_queens_101_88_l2303_230382

/-- Represents a chessboard with a red corner -/
structure RedCornerBoard :=
  (size : Nat)
  (red_size : Nat)
  (h_size : size > red_size)

/-- Represents the maximum number of non-attacking queens on a RedCornerBoard -/
def max_queens (board : RedCornerBoard) : Nat :=
  2 * (board.size - board.red_size)

/-- Theorem stating the maximum number of non-attacking queens on a 101x101 board with 88x88 red corner -/
theorem max_queens_101_88 :
  let board : RedCornerBoard := ⟨101, 88, by norm_num⟩
  max_queens board = 26 := by
  sorry

#eval max_queens ⟨101, 88, by norm_num⟩

end NUMINAMATH_CALUDE_max_queens_101_88_l2303_230382


namespace NUMINAMATH_CALUDE_cube_equation_solution_l2303_230314

theorem cube_equation_solution (x y : ℝ) : x^3 - 8*y^3 = 0 ↔ x = 2*y := by
  sorry

end NUMINAMATH_CALUDE_cube_equation_solution_l2303_230314


namespace NUMINAMATH_CALUDE_distance_origin_to_line_l2303_230319

/-- The distance from the origin to the line 4x + 3y - 12 = 0 is 12/5 -/
theorem distance_origin_to_line : 
  let line := {(x, y) : ℝ × ℝ | 4 * x + 3 * y - 12 = 0}
  ∃ d : ℝ, d = 12/5 ∧ ∀ (p : ℝ × ℝ), p ∈ line → Real.sqrt ((p.1)^2 + (p.2)^2) ≥ d := by
  sorry

end NUMINAMATH_CALUDE_distance_origin_to_line_l2303_230319


namespace NUMINAMATH_CALUDE_checkerboard_domino_cover_l2303_230300

/-- A checkerboard is a rectangular grid of squares. -/
structure Checkerboard where
  rows : ℕ
  cols : ℕ

/-- A domino covers exactly two squares. -/
def domino_cover := 2

/-- The total number of squares on a checkerboard. -/
def total_squares (board : Checkerboard) : ℕ :=
  board.rows * board.cols

/-- A checkerboard can be covered by dominoes if its total number of squares is even. -/
theorem checkerboard_domino_cover (board : Checkerboard) :
  ∃ (n : ℕ), total_squares board = n * domino_cover ↔ Even (total_squares board) :=
sorry

end NUMINAMATH_CALUDE_checkerboard_domino_cover_l2303_230300


namespace NUMINAMATH_CALUDE_a_squared_gt_b_squared_necessary_not_sufficient_l2303_230334

theorem a_squared_gt_b_squared_necessary_not_sufficient (a b : ℝ) :
  (∀ a b : ℝ, a^3 > b^3 ∧ b^3 > 0 → a^2 > b^2) ∧
  (∃ a b : ℝ, a^2 > b^2 ∧ ¬(a^3 > b^3 ∧ b^3 > 0)) :=
by sorry

end NUMINAMATH_CALUDE_a_squared_gt_b_squared_necessary_not_sufficient_l2303_230334


namespace NUMINAMATH_CALUDE_average_glasses_per_box_l2303_230309

/-- Proves that the average number of glasses per box is 15 given the specified conditions -/
theorem average_glasses_per_box (small_box_count : ℕ) (large_box_count : ℕ) : 
  large_box_count = small_box_count + 16 →
  12 * small_box_count + 16 * large_box_count = 480 →
  (480 : ℚ) / (small_box_count + large_box_count) = 15 := by
sorry

end NUMINAMATH_CALUDE_average_glasses_per_box_l2303_230309


namespace NUMINAMATH_CALUDE_survey_respondents_l2303_230304

theorem survey_respondents (prefer_x : ℕ) (ratio_x : ℕ) (ratio_y : ℕ) : 
  prefer_x = 150 → ratio_x = 5 → ratio_y = 1 →
  ∃ (total : ℕ), total = prefer_x + (prefer_x / ratio_x * ratio_y) ∧ total = 180 :=
by
  sorry

end NUMINAMATH_CALUDE_survey_respondents_l2303_230304


namespace NUMINAMATH_CALUDE_select_students_theorem_l2303_230337

/-- Represents the number of students in each category for a group -/
structure GroupComposition :=
  (male : ℕ)
  (female : ℕ)

/-- Calculates the number of ways to select students from two groups with exactly one female -/
def selectStudentsWithOneFemale (groupA groupB : GroupComposition) : ℕ :=
  let selectOneFromA := groupA.female * groupA.male * (groupB.male.choose 2)
  let selectOneFromB := groupB.female * groupB.male * (groupA.male.choose 2)
  selectOneFromA + selectOneFromB

/-- The main theorem stating the number of ways to select students -/
theorem select_students_theorem (groupA groupB : GroupComposition) : 
  groupA.male = 5 → groupA.female = 3 → groupB.male = 6 → groupB.female = 2 →
  selectStudentsWithOneFemale groupA groupB = 345 := by
  sorry

#eval selectStudentsWithOneFemale ⟨5, 3⟩ ⟨6, 2⟩

end NUMINAMATH_CALUDE_select_students_theorem_l2303_230337


namespace NUMINAMATH_CALUDE_interference_facts_l2303_230359

/-- A fact about light -/
inductive LightFact
  | SignalTransmission
  | SurfaceFlatness
  | PrismSpectrum
  | OilFilmColors

/-- Predicate to determine if a light fact involves interference -/
def involves_interference (fact : LightFact) : Prop :=
  match fact with
  | LightFact.SurfaceFlatness => true
  | LightFact.OilFilmColors => true
  | _ => false

/-- Theorem stating that only facts 2 and 4 involve interference -/
theorem interference_facts :
  (∀ f : LightFact, involves_interference f ↔ (f = LightFact.SurfaceFlatness ∨ f = LightFact.OilFilmColors)) :=
by sorry

end NUMINAMATH_CALUDE_interference_facts_l2303_230359


namespace NUMINAMATH_CALUDE_library_visitors_l2303_230365

theorem library_visitors (sunday_visitors : ℕ) (total_days : ℕ) (sundays : ℕ) (avg_visitors : ℕ) :
  sunday_visitors = 510 →
  total_days = 30 →
  sundays = 5 →
  avg_visitors = 285 →
  (sunday_visitors * sundays + (total_days - sundays) * 
    ((total_days * avg_visitors - sunday_visitors * sundays) / (total_days - sundays))) / total_days = avg_visitors :=
by sorry

end NUMINAMATH_CALUDE_library_visitors_l2303_230365


namespace NUMINAMATH_CALUDE_floss_leftover_is_five_l2303_230379

/-- The amount of floss left over after distributing to students -/
def floss_leftover (num_students : ℕ) (floss_per_student : ℚ) (floss_per_packet : ℕ) : ℚ :=
  let total_floss_needed : ℚ := num_students * floss_per_student
  let packets_needed : ℕ := (total_floss_needed / floss_per_packet).ceil.toNat
  (packets_needed * floss_per_packet : ℚ) - total_floss_needed

/-- Theorem stating the amount of floss left over in the given scenario -/
theorem floss_leftover_is_five :
  floss_leftover 20 (3/2) 35 = 5 := by
  sorry

end NUMINAMATH_CALUDE_floss_leftover_is_five_l2303_230379


namespace NUMINAMATH_CALUDE_parentheses_placement_l2303_230318

theorem parentheses_placement : 90 - 72 / (6 + 3) = 82 := by sorry

end NUMINAMATH_CALUDE_parentheses_placement_l2303_230318


namespace NUMINAMATH_CALUDE_computers_probability_l2303_230355

def CAMPUS : Finset Char := {'C', 'A', 'M', 'P', 'U', 'S'}
def THREADS : Finset Char := {'T', 'H', 'R', 'E', 'A', 'D', 'S'}
def GLOW : Finset Char := {'G', 'L', 'O', 'W'}
def COMPUTERS : Finset Char := {'C', 'O', 'M', 'P', 'U', 'T', 'E', 'R', 'S'}

def probability_CAMPUS : ℚ := 1 / (CAMPUS.card.choose 3)
def probability_THREADS : ℚ := 1 / (THREADS.card.choose 5)
def probability_GLOW : ℚ := (GLOW.card - 1).choose 1 / (GLOW.card.choose 2)

theorem computers_probability :
  probability_CAMPUS * probability_THREADS * probability_GLOW = 1 / 840 := by
  sorry

end NUMINAMATH_CALUDE_computers_probability_l2303_230355


namespace NUMINAMATH_CALUDE_distinct_color_selections_eq_62_l2303_230360

/-- The number of ways to select 6 objects from 5 red and 5 blue objects, where order matters only for color. -/
def distinct_color_selections : ℕ :=
  let red := 5
  let blue := 5
  let total_select := 6
  (2 * (Nat.choose total_select 1) +  -- 5 of one color, 1 of the other
   2 * (Nat.choose total_select 2) +  -- 4 of one color, 2 of the other
   Nat.choose total_select 3)         -- 3 of each color

/-- Theorem stating that the number of distinct color selections is 62. -/
theorem distinct_color_selections_eq_62 : distinct_color_selections = 62 := by
  sorry

end NUMINAMATH_CALUDE_distinct_color_selections_eq_62_l2303_230360


namespace NUMINAMATH_CALUDE_difference_of_squares_special_case_l2303_230312

theorem difference_of_squares_special_case (m : ℝ) : 
  (2 * m + 1/2) * (2 * m - 1/2) = 4 * m^2 - 1/4 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_special_case_l2303_230312


namespace NUMINAMATH_CALUDE_line_equation_with_slope_and_area_l2303_230322

theorem line_equation_with_slope_and_area (x y : ℝ) :
  ∃ (b : ℝ), (3 * x - 4 * y + 12 * b = 0 ∨ 3 * x - 4 * y - 12 * b = 0) ∧
  (3 / 4 : ℝ) = (y - 0) / (x - 0) ∧
  6 = (1 / 2) * |0 - x| * |0 - y| :=
sorry

end NUMINAMATH_CALUDE_line_equation_with_slope_and_area_l2303_230322


namespace NUMINAMATH_CALUDE_fourth_power_sum_l2303_230358

theorem fourth_power_sum (a : ℝ) (h : a^2 - 3*a + 1 = 0) : a^4 + 1/a^4 = 47 := by
  sorry

end NUMINAMATH_CALUDE_fourth_power_sum_l2303_230358


namespace NUMINAMATH_CALUDE_parabola_kite_sum_l2303_230317

/-- Given two parabolas y = ax^2 + 4 and y = 6 - bx^2 that intersect the coordinate axes
    in exactly four points forming a kite with area 18, prove that a + b = 4/45 -/
theorem parabola_kite_sum (a b : ℝ) : 
  (∃ x₁ x₂ y₁ y₂ : ℝ, 
    -- First parabola intersects x-axis
    (a * x₁^2 + 4 = 0 ∧ a * x₂^2 + 4 = 0 ∧ x₁ ≠ x₂) ∧ 
    -- Second parabola intersects x-axis
    (6 - b * x₁^2 = 0 ∧ 6 - b * x₂^2 = 0 ∧ x₁ ≠ x₂) ∧
    -- First parabola intersects y-axis
    (a * 0^2 + 4 = y₁) ∧
    -- Second parabola intersects y-axis
    (6 - b * 0^2 = y₂) ∧
    -- Area of the kite formed by these points is 18
    (1/2 * (x₂ - x₁) * (y₂ - y₁) = 18)) →
  a + b = 4/45 := by
sorry

end NUMINAMATH_CALUDE_parabola_kite_sum_l2303_230317


namespace NUMINAMATH_CALUDE_radical_sum_equals_eight_sqrt_three_l2303_230398

theorem radical_sum_equals_eight_sqrt_three :
  Real.sqrt (16 - 8 * Real.sqrt 3) + Real.sqrt (16 + 8 * Real.sqrt 3) = 8 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_radical_sum_equals_eight_sqrt_three_l2303_230398


namespace NUMINAMATH_CALUDE_fraction_not_simplifiable_l2303_230333

theorem fraction_not_simplifiable (n : ℕ) : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_not_simplifiable_l2303_230333


namespace NUMINAMATH_CALUDE_renovation_sand_required_l2303_230391

/-- The amount of sand required for a renovation project -/
theorem renovation_sand_required (total_material dirt cement : ℝ) 
  (h_total : total_material = 0.67)
  (h_dirt : dirt = 0.33)
  (h_cement : cement = 0.17) :
  total_material - dirt - cement = 0.17 := by
  sorry

end NUMINAMATH_CALUDE_renovation_sand_required_l2303_230391


namespace NUMINAMATH_CALUDE_inequality_proof_l2303_230390

theorem inequality_proof (x : ℝ) (h1 : 3/2 ≤ x) (h2 : x ≤ 5) :
  2 * Real.sqrt (x + 1) + Real.sqrt (2 * x - 3) + Real.sqrt (15 - 3 * x) < 2 * Real.sqrt 19 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2303_230390


namespace NUMINAMATH_CALUDE_intersection_M_N_l2303_230353

def M : Set ℕ := {1, 3, 5, 7, 9}
def N : Set ℕ := {x : ℕ | 2 * x > 7}

theorem intersection_M_N : M ∩ N = {5, 7, 9} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2303_230353


namespace NUMINAMATH_CALUDE_same_last_six_digits_l2303_230381

/-- Given a positive integer N where N and N^2 both end in the same sequence
    of six digits abcdef in base 10 (with a ≠ 0), prove that the five-digit
    number abcde is equal to 48437. -/
theorem same_last_six_digits (N : ℕ) : 
  (N > 0) →
  (∃ (a b c d e f : ℕ), 
    a ≠ 0 ∧
    a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10 ∧ f < 10 ∧
    N % 1000000 = a * 100000 + b * 10000 + c * 1000 + d * 100 + e * 10 + f ∧
    (N^2) % 1000000 = a * 100000 + b * 10000 + c * 1000 + d * 100 + e * 10 + f) →
  (∃ (a b c d e : ℕ),
    a ≠ 0 ∧
    a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10 ∧
    a * 10000 + b * 1000 + c * 100 + d * 10 + e = 48437) :=
by sorry

end NUMINAMATH_CALUDE_same_last_six_digits_l2303_230381


namespace NUMINAMATH_CALUDE_max_area_equilateral_triangle_in_rectangle_l2303_230373

/-- The maximum area of an equilateral triangle inscribed in a 12 by 5 rectangle -/
theorem max_area_equilateral_triangle_in_rectangle :
  ∃ (A : ℝ),
    A = (15 : ℝ) * Real.sqrt 3 - 10 ∧
    (∀ (s : ℝ),
      s > 0 →
      s ≤ 5 →
      s * Real.sqrt 3 ≤ 12 →
      (Real.sqrt 3 / 4) * s^2 ≤ A) :=
by sorry

end NUMINAMATH_CALUDE_max_area_equilateral_triangle_in_rectangle_l2303_230373


namespace NUMINAMATH_CALUDE_square_difference_81_49_l2303_230361

theorem square_difference_81_49 : 81^2 - 49^2 = 4160 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_81_49_l2303_230361


namespace NUMINAMATH_CALUDE_vitamin_a_weekly_pills_l2303_230383

/-- Calculates the number of pills needed for a week's supply of Vitamin A -/
def weekly_vitamin_pills (daily_recommended : ℕ) (mg_per_pill : ℕ) : ℕ :=
  (daily_recommended / mg_per_pill) * 7

/-- Theorem stating that 28 pills are needed for a week's supply of Vitamin A -/
theorem vitamin_a_weekly_pills :
  weekly_vitamin_pills 200 50 = 28 := by
  sorry

#eval weekly_vitamin_pills 200 50

end NUMINAMATH_CALUDE_vitamin_a_weekly_pills_l2303_230383


namespace NUMINAMATH_CALUDE_jogger_distance_ahead_l2303_230315

/-- Prove that a jogger is 250 meters ahead of a train's engine given the specified conditions -/
theorem jogger_distance_ahead (jogger_speed : ℝ) (train_speed : ℝ) (train_length : ℝ) (passing_time : ℝ) :
  jogger_speed = 9 * (5/18) →
  train_speed = 45 * (5/18) →
  train_length = 120 →
  passing_time = 37 →
  (train_speed - jogger_speed) * passing_time - train_length = 250 := by
  sorry

end NUMINAMATH_CALUDE_jogger_distance_ahead_l2303_230315
