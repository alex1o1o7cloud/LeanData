import Mathlib

namespace NUMINAMATH_CALUDE_log_865_between_consecutive_integers_l2374_237478

theorem log_865_between_consecutive_integers :
  ∃ c d : ℤ, c + 1 = d ∧ (c : ℝ) < Real.log 865 / Real.log 10 ∧ Real.log 865 / Real.log 10 < (d : ℝ) ∧ c + d = 5 := by
  sorry

end NUMINAMATH_CALUDE_log_865_between_consecutive_integers_l2374_237478


namespace NUMINAMATH_CALUDE_no_solution_for_equation_l2374_237446

theorem no_solution_for_equation : ¬∃ (x : ℝ), (1 / (x + 11) + 1 / (x + 8) = 1 / (x + 12) + 1 / (x + 7)) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_equation_l2374_237446


namespace NUMINAMATH_CALUDE_speed_difference_meeting_l2374_237429

/-- The difference in speed between two travelers meeting at a point -/
theorem speed_difference_meeting (distance : ℝ) (time : ℝ) (speed_enrique : ℝ) (speed_jamal : ℝ)
  (h1 : distance = 200)  -- Total distance between Enrique and Jamal
  (h2 : time = 8)        -- Time taken to meet
  (h3 : speed_enrique = 16)  -- Enrique's speed
  (h4 : speed_jamal = 23)    -- Jamal's speed
  (h5 : distance = (speed_enrique + speed_jamal) * time)  -- Distance traveled equals total speed times time
  : speed_jamal - speed_enrique = 7 := by
  sorry

end NUMINAMATH_CALUDE_speed_difference_meeting_l2374_237429


namespace NUMINAMATH_CALUDE_value_after_two_years_theorem_l2374_237433

/-- Calculates the value of an amount after two years, considering annual increases and inflation rates -/
def value_after_two_years (initial_amount : ℝ) (annual_increase_rate : ℝ) (inflation_rate_year1 : ℝ) (inflation_rate_year2 : ℝ) : ℝ :=
  let amount_year1 := initial_amount * (1 + annual_increase_rate)
  let value_year1 := amount_year1 * (1 - inflation_rate_year1)
  let amount_year2 := value_year1 * (1 + annual_increase_rate)
  let value_year2 := amount_year2 * (1 - inflation_rate_year2)
  value_year2

/-- Theorem stating that the value after two years is approximately 3771.36 -/
theorem value_after_two_years_theorem :
  let initial_amount : ℝ := 3200
  let annual_increase_rate : ℝ := 1/8
  let inflation_rate_year1 : ℝ := 3/100
  let inflation_rate_year2 : ℝ := 4/100
  abs (value_after_two_years initial_amount annual_increase_rate inflation_rate_year1 inflation_rate_year2 - 3771.36) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_value_after_two_years_theorem_l2374_237433


namespace NUMINAMATH_CALUDE_max_product_constrained_max_product_constrained_achieved_l2374_237414

theorem max_product_constrained (a b : ℝ) : 
  a > 0 → b > 0 → a + 2*b = 2 → ab ≤ 1/2 := by
  sorry

theorem max_product_constrained_achieved (a b : ℝ) : 
  ∃ a b, a > 0 ∧ b > 0 ∧ a + 2*b = 2 ∧ ab = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_max_product_constrained_max_product_constrained_achieved_l2374_237414


namespace NUMINAMATH_CALUDE_salary_increase_percentage_l2374_237431

theorem salary_increase_percentage (S : ℝ) (h1 : S + 0.16 * S = 348) (h2 : S + x * S = 375) : x = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_salary_increase_percentage_l2374_237431


namespace NUMINAMATH_CALUDE_january_salary_l2374_237468

-- Define variables for each month's salary
variable (jan feb mar apr may : ℕ)

-- Define the conditions
def condition1 : Prop := (jan + feb + mar + apr) / 4 = 8000
def condition2 : Prop := (feb + mar + apr + may) / 4 = 8700
def condition3 : Prop := may = 6500

-- Theorem statement
theorem january_salary 
  (h1 : condition1 jan feb mar apr)
  (h2 : condition2 feb mar apr may)
  (h3 : condition3 may) :
  jan = 3700 := by
  sorry

end NUMINAMATH_CALUDE_january_salary_l2374_237468


namespace NUMINAMATH_CALUDE_parabola_shift_l2374_237403

/-- Represents a parabola in the form y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a parabola horizontally and vertically -/
def shift (p : Parabola) (h : ℝ) (v : ℝ) : Parabola :=
  { a := p.a
    b := -2 * p.a * h + p.b
    c := p.a * h^2 - p.b * h + p.c + v }

theorem parabola_shift :
  let original := Parabola.mk 5 0 0
  let shifted := shift original 2 3
  shifted = Parabola.mk 5 (-20) 23 := by
  sorry

end NUMINAMATH_CALUDE_parabola_shift_l2374_237403


namespace NUMINAMATH_CALUDE_initial_workers_count_l2374_237410

theorem initial_workers_count (W : ℕ) : 
  (2 : ℚ) / 3 * W = W - (W / 3) →  -- Initially, 2/3 of workers are men
  (W / 3 + 10 : ℚ) / (W + 10) = 2 / 5 →  -- After hiring 10 women, 40% of workforce is female
  W = 90 := by
sorry

end NUMINAMATH_CALUDE_initial_workers_count_l2374_237410


namespace NUMINAMATH_CALUDE_triangle_side_length_l2374_237405

/-- Given a triangle XYZ with side lengths and median, prove the length of XZ -/
theorem triangle_side_length (XY YZ XM : ℝ) (h1 : XY = 7) (h2 : YZ = 10) (h3 : XM = 5) :
  ∃ (XZ : ℝ), XZ = Real.sqrt 51 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2374_237405


namespace NUMINAMATH_CALUDE_base2_digit_difference_l2374_237448

-- Function to calculate the number of digits in base-2 representation
def base2Digits (n : ℕ) : ℕ :=
  if n = 0 then 1 else Nat.log2 n + 1

-- Theorem statement
theorem base2_digit_difference : base2Digits 1800 - base2Digits 500 = 2 := by
  sorry

end NUMINAMATH_CALUDE_base2_digit_difference_l2374_237448


namespace NUMINAMATH_CALUDE_regular_price_is_15_l2374_237418

-- Define the variables
def num_shirts : ℕ := 20
def discount_rate : ℚ := 0.2
def tax_rate : ℚ := 0.1
def total_paid : ℚ := 264

-- Define the theorem
theorem regular_price_is_15 :
  ∃ (regular_price : ℚ),
    regular_price * num_shirts * (1 - discount_rate) * (1 + tax_rate) = total_paid ∧
    regular_price = 15 := by
  sorry

end NUMINAMATH_CALUDE_regular_price_is_15_l2374_237418


namespace NUMINAMATH_CALUDE_unique_triple_l2374_237426

theorem unique_triple (a b c : ℕ) : 
  a > 1 → b > 1 → c > 1 → 
  (bc + 1) % a = 0 → 
  (ac + 1) % b = 0 → 
  (ab + 1) % c = 0 → 
  a = 2 ∧ b = 3 ∧ c = 7 := by
sorry

end NUMINAMATH_CALUDE_unique_triple_l2374_237426


namespace NUMINAMATH_CALUDE_shaun_age_l2374_237428

/-- Represents the current ages of Kay, Gordon, and Shaun --/
structure Ages where
  kay : ℕ
  gordon : ℕ
  shaun : ℕ

/-- Checks if the given ages satisfy the conditions of the problem --/
def satisfiesConditions (ages : Ages) : Prop :=
  (ages.kay + 4 = 2 * (ages.gordon + 4)) ∧
  (ages.shaun + 8 = 2 * (ages.kay + 8)) ∧
  (ages.shaun + 12 = 3 * (ages.gordon + 12))

/-- Theorem stating that if the ages satisfy the conditions, then Shaun's current age is 48 --/
theorem shaun_age (ages : Ages) :
  satisfiesConditions ages → ages.shaun = 48 := by sorry

end NUMINAMATH_CALUDE_shaun_age_l2374_237428


namespace NUMINAMATH_CALUDE_sum_of_square_areas_l2374_237477

/-- The sum of areas of an infinite sequence of squares -/
theorem sum_of_square_areas (first_side : ℝ) (h : first_side = 4) : 
  let area_ratio : ℝ := (0.5 * Real.sqrt 2)^2
  let first_area : ℝ := first_side^2
  let sum_areas : ℝ := first_area / (1 - area_ratio)
  sum_areas = 32 := by sorry

end NUMINAMATH_CALUDE_sum_of_square_areas_l2374_237477


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2374_237474

theorem complex_fraction_simplification :
  let z₁ : ℂ := 2 + 4*I
  let z₂ : ℂ := 2 - 4*I
  (z₁ / z₂ - z₂ / z₁) = (4:ℝ)/5 * I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2374_237474


namespace NUMINAMATH_CALUDE_library_tables_l2374_237416

/-- Converts a number from base 6 to base 10 -/
def base6ToBase10 (n : Nat) : Nat :=
  let units := n % 10
  let sixes := (n / 10) % 10
  let thirty_sixes := n / 100
  thirty_sixes * 36 + sixes * 6 + units

/-- Calculates the number of tables needed given the total number of people and people per table -/
def tablesNeeded (totalPeople : Nat) (peoplePerTable : Nat) : Nat :=
  (totalPeople + peoplePerTable - 1) / peoplePerTable

theorem library_tables (seatingCapacity : Nat) (peoplePerTable : Nat) :
  seatingCapacity = 231 ∧ peoplePerTable = 3 →
  tablesNeeded (base6ToBase10 seatingCapacity) peoplePerTable = 31 := by
  sorry

end NUMINAMATH_CALUDE_library_tables_l2374_237416


namespace NUMINAMATH_CALUDE_continuous_piecewise_function_sum_l2374_237432

noncomputable def f (c d : ℝ) (x : ℝ) : ℝ :=
  if x > 3 then c * x + 1
  else if x ≥ -1 then 2 * x - 7
  else 3 * x - d

theorem continuous_piecewise_function_sum (c d : ℝ) :
  Continuous (f c d) → c + d = 16/3 := by
  sorry

end NUMINAMATH_CALUDE_continuous_piecewise_function_sum_l2374_237432


namespace NUMINAMATH_CALUDE_concentric_circles_ratio_l2374_237420

theorem concentric_circles_ratio (r R : ℝ) (h1 : R = 10) 
  (h2 : π * R^2 = 2 * (π * R^2 - π * r^2)) : r = 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_concentric_circles_ratio_l2374_237420


namespace NUMINAMATH_CALUDE_line_inclination_l2374_237498

-- Define the line equation
def line_equation (x y : ℝ) : Prop := x - Real.sqrt 3 * y + 1 = 0

-- Define the angle of inclination
def angle_of_inclination (θ : ℝ) : Prop := Real.tan θ = 1 / Real.sqrt 3

-- Theorem statement
theorem line_inclination :
  ∃ θ, angle_of_inclination θ ∧ θ = 30 * π / 180 :=
sorry

end NUMINAMATH_CALUDE_line_inclination_l2374_237498


namespace NUMINAMATH_CALUDE_coordinates_in_new_basis_l2374_237419

open LinearAlgebra

variable {𝕜 : Type*} [Field 𝕜]
variable {E : Type*} [AddCommGroup E] [Module 𝕜 E]

/-- Given a vector space E over a field 𝕜, and two bases e and e' of E, 
    prove that the coordinates of a vector x in the new basis e' are {0, 1, -1} -/
theorem coordinates_in_new_basis 
  (e : Basis (Fin 3) 𝕜 E) 
  (e' : Basis (Fin 3) 𝕜 E) 
  (x : E) :
  (∀ i : Fin 3, e' i = 
    if i = 0 then e 0 + 2 • (e 2)
    else if i = 1 then e 1 + e 2
    else -(e 0) - (e 1) - 2 • (e 2)) →
  (x = e 0 + 2 • (e 1) + 3 • (e 2)) →
  (∃ a b c : 𝕜, x = a • (e' 0) + b • (e' 1) + c • (e' 2) ∧ a = 0 ∧ b = 1 ∧ c = -1) :=
by sorry

end NUMINAMATH_CALUDE_coordinates_in_new_basis_l2374_237419


namespace NUMINAMATH_CALUDE_max_value_is_60_l2374_237458

-- Define the types of jewels
structure Jewel :=
  (weight : ℕ)
  (value : ℕ)

-- Define the jewel types
def typeA : Jewel := ⟨6, 18⟩
def typeB : Jewel := ⟨3, 9⟩
def typeC : Jewel := ⟨1, 4⟩

-- Define the maximum carrying capacity
def maxCapacity : ℕ := 15

-- Define the function to calculate the maximum value
def maxValue (typeA typeB typeC : Jewel) (maxCapacity : ℕ) : ℕ :=
  sorry

-- Theorem stating the maximum value is 60
theorem max_value_is_60 :
  maxValue typeA typeB typeC maxCapacity = 60 :=
sorry

end NUMINAMATH_CALUDE_max_value_is_60_l2374_237458


namespace NUMINAMATH_CALUDE_complex_number_properties_l2374_237445

theorem complex_number_properties (w : ℂ) (h : w^2 = 16 - 48*I) : 
  Complex.abs w = 4 * (10 : ℝ)^(1/4) ∧ 
  Complex.arg w = (Real.arctan (-3) / 2 + Real.pi / 2) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_properties_l2374_237445


namespace NUMINAMATH_CALUDE_line_equivalence_l2374_237488

/-- Given a line in the form (3, 4) · ((x, y) - (2, 8)) = 0, 
    prove that it's equivalent to y = -3/4 * x + 9.5 -/
theorem line_equivalence :
  ∀ (x y : ℝ), 3 * (x - 2) + 4 * (y - 8) = 0 ↔ y = -3/4 * x + 9.5 := by
sorry

end NUMINAMATH_CALUDE_line_equivalence_l2374_237488


namespace NUMINAMATH_CALUDE_sin_2alpha_value_l2374_237436

theorem sin_2alpha_value (α : ℝ) (h1 : α ∈ Set.Ioo 0 (π / 2)) 
  (h2 : Real.sin (2 * α - π / 3) = Real.sqrt 3 / 3) : 
  Real.sin (2 * α) = (Real.sqrt 3 + 3 * Real.sqrt 2) / 6 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_value_l2374_237436


namespace NUMINAMATH_CALUDE_inverse_81_mod_101_l2374_237469

theorem inverse_81_mod_101 (h : (9⁻¹ : ZMod 101) = 65) : (81⁻¹ : ZMod 101) = 84 := by
  sorry

end NUMINAMATH_CALUDE_inverse_81_mod_101_l2374_237469


namespace NUMINAMATH_CALUDE_inequality_solution_sets_l2374_237481

-- Define the solution set of the first inequality
def solution_set_1 : Set ℝ := {x : ℝ | 2 < x ∧ x < 3}

-- Define the coefficients a and b
def a : ℝ := 5
def b : ℝ := -6

-- Define the solution set of the second inequality
def solution_set_2 : Set ℝ := {x : ℝ | -1/2 < x ∧ x < -1/3}

theorem inequality_solution_sets :
  (∀ x : ℝ, x ∈ solution_set_1 ↔ x^2 - a*x - b < 0) →
  (∀ x : ℝ, x ∈ solution_set_2 ↔ b*x^2 - a*x - 1 > 0) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_sets_l2374_237481


namespace NUMINAMATH_CALUDE_system_equation_result_l2374_237449

theorem system_equation_result (x y : ℝ) 
  (eq1 : 5 * x + y = 19) 
  (eq2 : x + 3 * y = 1) : 
  3 * x + 2 * y = 10 := by
sorry

end NUMINAMATH_CALUDE_system_equation_result_l2374_237449


namespace NUMINAMATH_CALUDE_quadratic_roots_theorem_l2374_237497

/-- Quadratic equation with parameter k -/
def quadratic (k : ℝ) (x : ℝ) : ℝ := x^2 + (2*k + 1)*x + k^2 + 1

/-- Discriminant of the quadratic equation -/
def discriminant (k : ℝ) : ℝ := (2*k + 1)^2 - 4*(k^2 + 1)

/-- Theorem stating the conditions for distinct real roots and the value of k -/
theorem quadratic_roots_theorem (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic k x₁ = 0 ∧ quadratic k x₂ = 0) ↔ k > 3/4 ∧
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic k x₁ = 0 ∧ quadratic k x₂ = 0 ∧
   |x₁| + |x₂| = x₁ * x₂ → k = 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_theorem_l2374_237497


namespace NUMINAMATH_CALUDE_derivative_f_at_pi_l2374_237456

noncomputable def f (x : ℝ) : ℝ := x * Real.sin x

theorem derivative_f_at_pi : 
  deriv f π = -π := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_pi_l2374_237456


namespace NUMINAMATH_CALUDE_square_is_three_l2374_237490

/-- Represents a digit in base 8 -/
def Digit8 := Fin 8

/-- The addition problem in base 8 -/
def addition_problem (x : Digit8) : Prop :=
  ∃ (carry1 carry2 carry3 : Nat),
    (5 * 8^3 + 3 * 8^2 + 2 * 8 + x.val) +
    (x.val * 8^2 + 6 * 8 + 1) +
    (x.val * 8 + 4) =
    6 * 8^3 + 3 * 8^2 + x.val * 8 + 2 +
    carry1 * 8 + carry2 * 8^2 + carry3 * 8^3

/-- The theorem stating that 3 is the unique solution to the addition problem -/
theorem square_is_three :
  ∃! (x : Digit8), addition_problem x ∧ x.val = 3 := by sorry

end NUMINAMATH_CALUDE_square_is_three_l2374_237490


namespace NUMINAMATH_CALUDE_total_payment_is_correct_l2374_237492

-- Define the payment per lawn
def payment_per_lawn : ℚ := 13 / 3

-- Define the number of lawns mowed
def lawns_mowed : ℚ := 8 / 5

-- Define the base fee
def base_fee : ℚ := 5

-- Theorem statement
theorem total_payment_is_correct :
  payment_per_lawn * lawns_mowed + base_fee = 179 / 15 := by
  sorry

end NUMINAMATH_CALUDE_total_payment_is_correct_l2374_237492


namespace NUMINAMATH_CALUDE_price_increase_2008_2009_l2374_237430

/-- Given a 60% increase from 2006 to 2008 and a 20% annual average growth rate
    from 2006 to 2009, the increase from 2008 to 2009 is 8%. -/
theorem price_increase_2008_2009 
  (price_2006 : ℝ) 
  (price_2008 : ℝ) 
  (price_2009 : ℝ) 
  (h1 : price_2008 = price_2006 * (1 + 0.60))
  (h2 : price_2009 = price_2006 * (1 + 0.20)^3) :
  price_2009 = price_2008 * (1 + 0.08) :=
by sorry

end NUMINAMATH_CALUDE_price_increase_2008_2009_l2374_237430


namespace NUMINAMATH_CALUDE_total_area_three_shapes_l2374_237496

theorem total_area_three_shapes 
  (rect_area square_area tri_area : ℝ)
  (rect_square_overlap rect_tri_overlap square_tri_overlap : ℝ)
  (all_overlap : ℝ) :
  let total_area := rect_area + square_area + tri_area - 
                    rect_square_overlap - rect_tri_overlap - square_tri_overlap + 
                    all_overlap
  total_area = 66 :=
by sorry

end NUMINAMATH_CALUDE_total_area_three_shapes_l2374_237496


namespace NUMINAMATH_CALUDE_muffin_cost_l2374_237421

theorem muffin_cost (num_muffins : ℕ) (juice_cost total_cost : ℚ) : 
  num_muffins = 3 → 
  juice_cost = 29/20 → 
  total_cost = 37/10 → 
  (total_cost - juice_cost) / num_muffins = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_muffin_cost_l2374_237421


namespace NUMINAMATH_CALUDE_fraction_equality_l2374_237494

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (4*x - 3*y) / (x + 4*y) = 3) : 
  (x - 4*y) / (4*x + 3*y) = 11/63 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2374_237494


namespace NUMINAMATH_CALUDE_a_fourth_plus_inverse_a_fourth_l2374_237495

theorem a_fourth_plus_inverse_a_fourth (a : ℝ) (h : a - 1/a = -2) : 
  a^4 + 1/a^4 = 34 := by
sorry

end NUMINAMATH_CALUDE_a_fourth_plus_inverse_a_fourth_l2374_237495


namespace NUMINAMATH_CALUDE_right_triangle_sides_l2374_237499

theorem right_triangle_sides : ∀ (a b c : ℝ), 
  (a > 0 ∧ b > 0 ∧ c > 0) →
  (a = 1 ∧ b = 2 ∧ c = Real.sqrt 3) ↔ a * a + b * b = c * c :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_sides_l2374_237499


namespace NUMINAMATH_CALUDE_parabola_max_value_l2374_237482

theorem parabola_max_value :
  let f : ℝ → ℝ := fun x ↦ -2 * x^2 + 4 * x + 3
  ∃ (max : ℝ), ∀ (x : ℝ), f x ≤ max ∧ ∃ (x_max : ℝ), f x_max = max ∧ max = 5 := by
  sorry

end NUMINAMATH_CALUDE_parabola_max_value_l2374_237482


namespace NUMINAMATH_CALUDE_total_earnings_proof_l2374_237475

/-- Represents a work day with various attributes -/
structure WorkDay where
  regular_hours : ℝ
  night_shift_hours : ℝ
  overtime_hours : ℝ
  weekend_hours : ℝ
  sales : ℝ

/-- Calculates total earnings for two weeks given work conditions -/
def calculate_total_earnings (
  last_week_hours : ℝ)
  (last_week_rate : ℝ)
  (regular_rate_increase : ℝ)
  (overtime_multiplier : ℝ)
  (weekend_multiplier : ℝ)
  (night_shift_multiplier : ℝ)
  (commission_rate : ℝ)
  (sales_bonus : ℝ)
  (satisfaction_deduction : ℝ)
  (work_week : List WorkDay)
  (total_sales : ℝ)
  (sales_target_reached : Bool)
  (satisfaction_below_threshold : Bool) : ℝ :=
  sorry

/-- Theorem stating that given the problem conditions, total earnings equal $1208.05 -/
theorem total_earnings_proof :
  let last_week_hours : ℝ := 35
  let last_week_rate : ℝ := 10
  let regular_rate_increase : ℝ := 0.5
  let overtime_multiplier : ℝ := 1.5
  let weekend_multiplier : ℝ := 1.7
  let night_shift_multiplier : ℝ := 1.3
  let commission_rate : ℝ := 0.05
  let sales_bonus : ℝ := 50
  let satisfaction_deduction : ℝ := 20
  let work_week : List WorkDay := [
    ⟨8, 3, 0, 0, 200⟩,
    ⟨10, 4, 2, 0, 400⟩,
    ⟨8, 0, 0, 0, 500⟩,
    ⟨9, 3, 1, 0, 300⟩,
    ⟨5, 0, 0, 0, 200⟩,
    ⟨6, 0, 0, 6, 300⟩,
    ⟨4, 2, 0, 4, 100⟩
  ]
  let total_sales : ℝ := 2000
  let sales_target_reached : Bool := true
  let satisfaction_below_threshold : Bool := true
  
  calculate_total_earnings
    last_week_hours
    last_week_rate
    regular_rate_increase
    overtime_multiplier
    weekend_multiplier
    night_shift_multiplier
    commission_rate
    sales_bonus
    satisfaction_deduction
    work_week
    total_sales
    sales_target_reached
    satisfaction_below_threshold = 1208.05 :=
  by sorry

end NUMINAMATH_CALUDE_total_earnings_proof_l2374_237475


namespace NUMINAMATH_CALUDE_line_AC_passes_through_fixed_point_l2374_237464

-- Define the moving circle M
def M : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (center : ℝ × ℝ), 
    ((p.1 - center.1)^2 + (p.2 - center.2)^2 = (p.2 + 1)^2) ∧
    ((0 - center.1)^2 + (1 - center.2)^2 = (1 + 1)^2)}

-- Define the trajectory of M's center
def trajectory : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 = 4 * p.2}

-- Define the moving line l
def l (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = k * p.1 - 2}

-- Define points A and B as intersections of l and trajectory
def intersectionPoints (k : ℝ) : Set (ℝ × ℝ) :=
  trajectory ∩ l k

-- Define point C as symmetric to B with respect to y-axis
def C (B : ℝ × ℝ) : ℝ × ℝ :=
  (-B.1, B.2)

-- Theorem statement
theorem line_AC_passes_through_fixed_point :
  ∀ (k : ℝ) (A B : ℝ × ℝ),
    A ∈ intersectionPoints k →
    B ∈ intersectionPoints k →
    A ≠ B →
    (0, 2) ∈ {p : ℝ × ℝ | ∃ t : ℝ, p = (1 - t) • A + t • C B} :=
sorry

end NUMINAMATH_CALUDE_line_AC_passes_through_fixed_point_l2374_237464


namespace NUMINAMATH_CALUDE_point_movement_l2374_237424

/-- 
Given a point P on a number line that is moved 4 units to the right and then 7 units to the left,
if its final position is 9, then its original position was 12.
-/
theorem point_movement (P : ℝ) : 
  (P + 4 - 7 = 9) → P = 12 := by
sorry

end NUMINAMATH_CALUDE_point_movement_l2374_237424


namespace NUMINAMATH_CALUDE_sum_of_extremal_x_values_l2374_237470

theorem sum_of_extremal_x_values (x y z : ℝ) 
  (sum_condition : x + y + z = 5)
  (square_sum_condition : x^2 + y^2 + z^2 = 11) :
  ∃ (m M : ℝ), 
    (∀ x', (∃ y' z', x' + y' + z' = 5 ∧ x'^2 + y'^2 + z'^2 = 11) → m ≤ x' ∧ x' ≤ M) ∧
    m + M = 10/3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_extremal_x_values_l2374_237470


namespace NUMINAMATH_CALUDE_expansion_has_four_nonzero_terms_count_equals_nonzero_terms_count_l2374_237442

/-- The number of nonzero terms in the expansion of (x-2)(3x^2-2x+5)+4(x^3+x^2-3x) -/
def nonzero_terms_count : ℕ := 4

/-- The expansion of (x-2)(3x^2-2x+5)+4(x^3+x^2-3x) -/
def expanded_polynomial (x : ℝ) : ℝ := 7*x^3 - 4*x^2 - 3*x - 10

theorem expansion_has_four_nonzero_terms :
  (∃ (a b c d : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧
    ∀ x, expanded_polynomial x = a*x^3 + b*x^2 + c*x + d) ∧
  (∀ (a b c d e : ℝ), (∀ x, expanded_polynomial x = a*x^4 + b*x^3 + c*x^2 + d*x + e) →
    (a = 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0) ∨
    (a = 0 ∧ b = 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0) ∨
    (a = 0 ∧ b ≠ 0 ∧ c = 0 ∧ d ≠ 0 ∧ e ≠ 0) ∨
    (a = 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d = 0 ∧ e ≠ 0) ∨
    (a = 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e = 0)) :=
by sorry

theorem count_equals_nonzero_terms_count :
  nonzero_terms_count = 4 :=
by sorry

end NUMINAMATH_CALUDE_expansion_has_four_nonzero_terms_count_equals_nonzero_terms_count_l2374_237442


namespace NUMINAMATH_CALUDE_blue_segments_count_l2374_237411

/-- Represents the number of rows and columns in the square array -/
def n : ℕ := 10

/-- Represents the total number of red dots -/
def total_red_dots : ℕ := 52

/-- Represents the number of red dots at corners -/
def corner_red_dots : ℕ := 2

/-- Represents the number of red dots on edges (excluding corners) -/
def edge_red_dots : ℕ := 16

/-- Represents the number of green line segments -/
def green_segments : ℕ := 98

/-- Theorem stating that the number of blue line segments is 37 -/
theorem blue_segments_count :
  let total_segments := 2 * n * (n - 1)
  let interior_red_dots := total_red_dots - corner_red_dots - edge_red_dots
  let red_connections := 2 * corner_red_dots + 3 * edge_red_dots + 4 * interior_red_dots
  let red_segments := (red_connections - green_segments) / 2
  let blue_segments := total_segments - red_segments - green_segments
  blue_segments = 37 := by sorry

end NUMINAMATH_CALUDE_blue_segments_count_l2374_237411


namespace NUMINAMATH_CALUDE_puzzle_solvable_l2374_237476

/-- Represents a polygonal piece --/
structure Piece where
  vertices : List (ℝ × ℝ)
  is_valid : List.length vertices ≥ 3

/-- Represents a shape formed by arranging pieces --/
structure Shape where
  pieces : List Piece
  arrangement : List (ℝ × ℝ) -- positions of pieces

/-- The original rectangle --/
def original_rectangle : Piece :=
  { vertices := [(0, 0), (4, 0), (4, 5), (0, 5)],
    is_valid := by sorry }

/-- The set of seven pieces cut from the original rectangle --/
def puzzle_pieces : List Piece :=
  sorry -- Define the seven pieces here

/-- The set of target shapes to be formed --/
def target_shapes : List Shape :=
  sorry -- Define the target shapes here

/-- Checks if a given arrangement of pieces forms a valid shape --/
def is_valid_arrangement (pieces : List Piece) (arrangement : List (ℝ × ℝ)) : Prop :=
  sorry -- Define the conditions for a valid arrangement

/-- The main theorem stating that the puzzle pieces can form the target shapes --/
theorem puzzle_solvable :
  ∀ shape ∈ target_shapes,
  ∃ arrangement : List (ℝ × ℝ),
  is_valid_arrangement puzzle_pieces arrangement ∧
  Shape.pieces shape = puzzle_pieces ∧
  Shape.arrangement shape = arrangement :=
sorry

end NUMINAMATH_CALUDE_puzzle_solvable_l2374_237476


namespace NUMINAMATH_CALUDE_cheryl_material_usage_l2374_237489

def material_a_initial : ℚ := 2/9
def material_b_initial : ℚ := 1/8
def material_c_initial : ℚ := 3/10

def material_a_leftover : ℚ := 4/18
def material_b_leftover : ℚ := 1/12
def material_c_leftover : ℚ := 3/15

def total_used : ℚ := 17/120

theorem cheryl_material_usage :
  (material_a_initial - material_a_leftover) +
  (material_b_initial - material_b_leftover) +
  (material_c_initial - material_c_leftover) = total_used := by
  sorry

end NUMINAMATH_CALUDE_cheryl_material_usage_l2374_237489


namespace NUMINAMATH_CALUDE_wall_washing_problem_l2374_237434

theorem wall_washing_problem (boys_5 boys_7 : ℕ) (wall_5 wall_7 : ℝ) (days : ℕ) :
  boys_5 = 5 →
  boys_7 = 7 →
  wall_5 = 25 →
  days = 4 →
  (boys_5 : ℝ) * wall_5 * (boys_7 : ℝ) = boys_7 * wall_7 * (boys_5 : ℝ) →
  wall_7 = 35 := by
sorry

end NUMINAMATH_CALUDE_wall_washing_problem_l2374_237434


namespace NUMINAMATH_CALUDE_consecutive_integers_permutation_divisibility_l2374_237447

theorem consecutive_integers_permutation_divisibility
  (p : ℕ) (h_prime : Nat.Prime p)
  (m : ℕ → ℕ) (h_consecutive : ∀ i ∈ Finset.range p, m (i + 1) = m i + 1)
  (σ : Fin p → Fin p) (h_perm : Function.Bijective σ) :
  ∃ (k l : Fin p), k ≠ l ∧ p ∣ (m k * m (σ k) - m l * m (σ l)) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_permutation_divisibility_l2374_237447


namespace NUMINAMATH_CALUDE_convex_polygon_sides_l2374_237439

theorem convex_polygon_sides (n : ℕ) : n > 2 → (n - 1) * 180 - 2008 < 180 ∧ 2008 < (n - 1) * 180 → n = 14 := by
  sorry

end NUMINAMATH_CALUDE_convex_polygon_sides_l2374_237439


namespace NUMINAMATH_CALUDE_speed_relationship_l2374_237457

/-- Represents the speed of travel between two towns -/
structure TravelSpeed where
  xy : ℝ  -- Speed from x to y
  yx : ℝ  -- Speed from y to x
  avg : ℝ  -- Average speed for the whole journey

/-- Theorem stating the relationship between speeds -/
theorem speed_relationship (s : TravelSpeed) (h1 : s.xy = 60) (h2 : s.avg = 40) : s.yx = 30 := by
  sorry

end NUMINAMATH_CALUDE_speed_relationship_l2374_237457


namespace NUMINAMATH_CALUDE_buddy_system_fraction_l2374_237479

theorem buddy_system_fraction (f e : ℕ) (h : e = (4 * f) / 3) : 
  (f / 3 + e / 4) / (f + e) = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_buddy_system_fraction_l2374_237479


namespace NUMINAMATH_CALUDE_third_number_is_five_l2374_237409

def hcf (a b c : ℕ) : ℕ := sorry

def lcm (a b c : ℕ) : ℕ := sorry

theorem third_number_is_five (a b c : ℕ) 
  (ha : a = 30)
  (hb : b = 75)
  (hhcf : hcf a b c = 15)
  (hlcm : lcm a b c = 750) :
  c = 5 := by sorry

end NUMINAMATH_CALUDE_third_number_is_five_l2374_237409


namespace NUMINAMATH_CALUDE_quadratic_inequality_equivalence_l2374_237487

theorem quadratic_inequality_equivalence (a b c : ℝ) :
  (a > 0 ∧ b^2 - 4*a*c < 0) ↔ (∀ x : ℝ, a*x^2 + b*x + c > 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_equivalence_l2374_237487


namespace NUMINAMATH_CALUDE_area_of_special_triangle_l2374_237459

-- Define the triangle ABC and point P
structure Triangle :=
  (A B C P : ℝ × ℝ)

-- Define the properties of the triangle
def is_right_triangle (t : Triangle) : Prop :=
  -- Implement the condition for a right triangle
  sorry

def is_scalene (t : Triangle) : Prop :=
  -- Implement the condition for a scalene triangle
  sorry

def on_hypotenuse (t : Triangle) : Prop :=
  -- Implement the condition that P is on the hypotenuse AC
  sorry

def angle_ABP_45 (t : Triangle) : Prop :=
  -- Implement the condition that ∠ABP = 45°
  sorry

def AP_equals_2 (t : Triangle) : Prop :=
  -- Implement the condition that AP = 2
  sorry

def CP_equals_3 (t : Triangle) : Prop :=
  -- Implement the condition that CP = 3
  sorry

-- Define the area of a triangle
def triangle_area (t : Triangle) : ℝ :=
  -- Implement the formula for triangle area
  sorry

-- Theorem statement
theorem area_of_special_triangle (t : Triangle) :
  is_right_triangle t →
  is_scalene t →
  on_hypotenuse t →
  angle_ABP_45 t →
  AP_equals_2 t →
  CP_equals_3 t →
  triangle_area t = 75 / 13 :=
sorry

end NUMINAMATH_CALUDE_area_of_special_triangle_l2374_237459


namespace NUMINAMATH_CALUDE_investment_value_after_six_weeks_l2374_237423

/-- Calculates the final investment value after six weeks of changes and compound interest --/
def calculate_investment (initial_investment : ℝ) (week1_gain : ℝ) (week1_add : ℝ)
  (week2_gain : ℝ) (week2_withdraw : ℝ) (week3_loss : ℝ) (week4_gain : ℝ) (week4_add : ℝ)
  (week5_gain : ℝ) (week6_loss : ℝ) (week6_withdraw : ℝ) (weekly_interest : ℝ) : ℝ :=
  let week1 := (initial_investment * (1 + week1_gain) * (1 + weekly_interest)) + week1_add
  let week2 := (week1 * (1 + week2_gain) * (1 + weekly_interest)) - week2_withdraw
  let week3 := week2 * (1 - week3_loss) * (1 + weekly_interest)
  let week4 := (week3 * (1 + week4_gain) * (1 + weekly_interest)) + week4_add
  let week5 := week4 * (1 + week5_gain) * (1 + weekly_interest)
  let week6 := (week5 * (1 - week6_loss) * (1 + weekly_interest)) - week6_withdraw
  week6

/-- The final investment value after six weeks is approximately $819.74 --/
theorem investment_value_after_six_weeks :
  ∃ ε > 0, |calculate_investment 400 0.25 200 0.50 150 0.10 0.20 100 0.05 0.15 250 0.02 - 819.74| < ε :=
sorry

end NUMINAMATH_CALUDE_investment_value_after_six_weeks_l2374_237423


namespace NUMINAMATH_CALUDE_condition_equivalence_l2374_237413

theorem condition_equivalence (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (a^2 + b^2 ≥ 2*a*b) ↔ (a/b + b/a ≥ 2) :=
by sorry

end NUMINAMATH_CALUDE_condition_equivalence_l2374_237413


namespace NUMINAMATH_CALUDE_unanswered_questions_count_l2374_237450

/-- Represents the test scenario with given conditions -/
structure TestScenario where
  total_questions : ℕ
  first_set_questions : ℕ
  second_set_questions : ℕ
  third_set_questions : ℕ
  first_set_time : ℕ  -- in minutes
  second_set_time : ℕ  -- in seconds
  third_set_time : ℕ  -- in minutes
  total_time : ℕ  -- in hours

/-- Calculates the number of unanswered questions in the given test scenario -/
def unanswered_questions (scenario : TestScenario) : ℕ :=
  scenario.total_questions - (scenario.first_set_questions + scenario.second_set_questions + scenario.third_set_questions)

/-- Theorem stating that for the given test scenario, the number of unanswered questions is 75 -/
theorem unanswered_questions_count (scenario : TestScenario) 
  (h1 : scenario.total_questions = 200)
  (h2 : scenario.first_set_questions = 50)
  (h3 : scenario.second_set_questions = 50)
  (h4 : scenario.third_set_questions = 25)
  (h5 : scenario.first_set_time = 1)
  (h6 : scenario.second_set_time = 90)
  (h7 : scenario.third_set_time = 2)
  (h8 : scenario.total_time = 4) :
  unanswered_questions scenario = 75 := by
  sorry

#eval unanswered_questions {
  total_questions := 200,
  first_set_questions := 50,
  second_set_questions := 50,
  third_set_questions := 25,
  first_set_time := 1,
  second_set_time := 90,
  third_set_time := 2,
  total_time := 4
}

end NUMINAMATH_CALUDE_unanswered_questions_count_l2374_237450


namespace NUMINAMATH_CALUDE_sector_area_l2374_237402

theorem sector_area (r : ℝ) (θ : ℝ) (h1 : r = 6) (h2 : θ = π / 6) :
  (1 / 2) * r^2 * θ = 3 * π := by
  sorry

#check sector_area

end NUMINAMATH_CALUDE_sector_area_l2374_237402


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2374_237466

/-- The eccentricity of a hyperbola with equation x²/a² - y²/b² = 1 and asymptote 3x + y = 0 -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_asymptote : b = 3 * a) : Real.sqrt 10 = 
  Real.sqrt ((a^2 + b^2) / a^2) := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2374_237466


namespace NUMINAMATH_CALUDE_triangle_perimeter_impossibility_l2374_237437

theorem triangle_perimeter_impossibility (a b x : ℝ) : 
  a = 10 → b = 25 → a + b + x = 72 → ¬(a + x > b ∧ b + x > a ∧ a + b > x) :=
by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_impossibility_l2374_237437


namespace NUMINAMATH_CALUDE_expression_simplification_find_k_value_l2374_237415

-- Problem 1: Simplify the expression
theorem expression_simplification (x : ℝ) :
  (2*x + 1)^2 - (2*x + 1)*(2*x - 1) + (x + 1)*(x - 3) = x^2 + 2*x - 1 :=
by sorry

-- Problem 2: Find the value of k
theorem find_k_value (x y k : ℝ) 
  (eq1 : x + y = 1)
  (eq2 : k*x + (k-1)*y = 7)
  (eq3 : 3*x - 2*y = 5) :
  k = 33/5 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_find_k_value_l2374_237415


namespace NUMINAMATH_CALUDE_sine_inequality_in_acute_triangle_l2374_237435

theorem sine_inequality_in_acute_triangle (A B C : Real) 
  (triangle_condition : A ≤ B ∧ B ≤ C ∧ C < Real.pi / 2) : 
  Real.sin (2 * A) ≥ Real.sin (2 * B) ∧ Real.sin (2 * B) ≥ Real.sin (2 * C) := by
  sorry

end NUMINAMATH_CALUDE_sine_inequality_in_acute_triangle_l2374_237435


namespace NUMINAMATH_CALUDE_correct_match_probability_l2374_237472

theorem correct_match_probability (n : ℕ) (h : n = 6) :
  (1 : ℚ) / (Nat.factorial n) = (1 : ℚ) / 720 :=
sorry

end NUMINAMATH_CALUDE_correct_match_probability_l2374_237472


namespace NUMINAMATH_CALUDE_octal_to_decimal_l2374_237471

-- Define the octal number
def octal_number : ℕ := 724

-- Define the decimal number
def decimal_number : ℕ := 468

-- Theorem stating that the octal number 724 is equal to the decimal number 468
theorem octal_to_decimal :
  octal_number.digits 8 = [4, 2, 7] ∧ 
  decimal_number = 4 * 8^0 + 2 * 8^1 + 7 * 8^2 := by
  sorry

#check octal_to_decimal

end NUMINAMATH_CALUDE_octal_to_decimal_l2374_237471


namespace NUMINAMATH_CALUDE_albert_purchase_cost_l2374_237465

/-- The total cost of horses and cows bought by Albert --/
def total_cost (num_horses num_cows : ℕ) (horse_cost cow_cost : ℕ) : ℕ :=
  num_horses * horse_cost + num_cows * cow_cost

/-- The profit from selling an item at a certain percentage --/
def profit_from_sale (cost : ℕ) (profit_percentage : ℚ) : ℚ :=
  (cost : ℚ) * profit_percentage

theorem albert_purchase_cost :
  ∃ (cow_cost : ℕ),
    let num_horses : ℕ := 4
    let num_cows : ℕ := 9
    let horse_cost : ℕ := 2000
    let horse_profit_percentage : ℚ := 1/10
    let cow_profit_percentage : ℚ := 1/5
    let total_profit : ℕ := 1880
    (num_horses : ℚ) * profit_from_sale horse_cost horse_profit_percentage +
    (num_cows : ℚ) * profit_from_sale cow_cost cow_profit_percentage = total_profit ∧
    total_cost num_horses num_cows horse_cost cow_cost = 13400 :=
by sorry


end NUMINAMATH_CALUDE_albert_purchase_cost_l2374_237465


namespace NUMINAMATH_CALUDE_triangle_area_l2374_237463

/-- The area of a triangle with side lengths 7, 8, and 10 -/
theorem triangle_area : ℝ := by
  -- Define the side lengths
  let a : ℝ := 7
  let b : ℝ := 8
  let c : ℝ := 10

  -- Define the semi-perimeter
  let s : ℝ := (a + b + c) / 2

  -- Define the area using Heron's formula
  let area : ℝ := Real.sqrt (s * (s - a) * (s - b) * (s - c))

  -- The actual proof would go here
  sorry

end NUMINAMATH_CALUDE_triangle_area_l2374_237463


namespace NUMINAMATH_CALUDE_allison_june_uploads_l2374_237425

/-- Calculates the total number of video hours uploaded by Allison in June -/
def total_video_hours (initial_rate : ℕ) (days_in_june : ℕ) (initial_period : ℕ) : ℕ :=
  let doubled_rate := 2 * initial_rate
  let remaining_period := days_in_june - initial_period
  initial_rate * initial_period + doubled_rate * remaining_period

/-- Theorem stating that Allison's total uploaded video hours in June is 450 -/
theorem allison_june_uploads :
  total_video_hours 10 30 15 = 450 := by
  sorry

end NUMINAMATH_CALUDE_allison_june_uploads_l2374_237425


namespace NUMINAMATH_CALUDE_counterexample_exists_l2374_237401

theorem counterexample_exists : ∃ (n : ℕ), n ≥ 2 ∧ 
  ∃ (k : ℕ), (2^(2^n) % (2^n - 1) = k) ∧ ¬∃ (m : ℕ), k = 4^m :=
by sorry

end NUMINAMATH_CALUDE_counterexample_exists_l2374_237401


namespace NUMINAMATH_CALUDE_minimum_cost_is_2200_l2374_237406

/-- Represents the transportation problem for washing machines -/
structure TransportationProblem where
  totalWashingMachines : ℕ
  typeATrucks : ℕ
  typeBTrucks : ℕ
  typeACapacity : ℕ
  typeBCapacity : ℕ
  typeACost : ℕ
  typeBCost : ℕ

/-- Calculates the minimum transportation cost for the given problem -/
def minimumTransportationCost (p : TransportationProblem) : ℕ :=
  sorry

/-- The main theorem stating that the minimum transportation cost is 2200 yuan -/
theorem minimum_cost_is_2200 :
  let p : TransportationProblem := {
    totalWashingMachines := 100,
    typeATrucks := 4,
    typeBTrucks := 8,
    typeACapacity := 20,
    typeBCapacity := 10,
    typeACost := 400,
    typeBCost := 300
  }
  minimumTransportationCost p = 2200 := by
  sorry

end NUMINAMATH_CALUDE_minimum_cost_is_2200_l2374_237406


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2374_237444

/-- The equation 7x^2 + 13x + d = 0 has rational solutions for d -/
def has_rational_solution (d : ℕ+) : Prop :=
  ∃ x : ℚ, 7 * x^2 + 13 * x + d.val = 0

/-- The set of positive integers d for which the equation has rational solutions -/
def solution_set : Set ℕ+ :=
  {d | has_rational_solution d}

theorem quadratic_equation_solution :
  ∃ (d₁ d₂ : ℕ+), d₁ ≠ d₂ ∧ 
    solution_set = {d₁, d₂} ∧
    d₁.val * d₂.val = 2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2374_237444


namespace NUMINAMATH_CALUDE_unique_solution_cube_equation_l2374_237440

theorem unique_solution_cube_equation (y : ℝ) (hy : y ≠ 0) :
  (3 * y)^5 = (9 * y)^4 ↔ y = 27 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_cube_equation_l2374_237440


namespace NUMINAMATH_CALUDE_berry_difference_change_l2374_237452

/-- Represents the number of berries in a box -/
structure Berry where
  count : ℕ

/-- Represents a box of berries -/
inductive Box
  | Red : Berry → Box
  | Blue : Berry → Box

/-- The problem setup -/
structure BerryProblem where
  blue_berry_count : ℕ
  red_berry_count : ℕ
  berry_increase : ℕ
  blue_box_count : ℕ
  red_box_count : ℕ

/-- The theorem to prove -/
theorem berry_difference_change (problem : BerryProblem) 
  (h1 : problem.blue_berry_count = 36)
  (h2 : problem.red_berry_count = problem.blue_berry_count + problem.berry_increase)
  (h3 : problem.berry_increase = 15) :
  problem.red_berry_count - problem.blue_berry_count = 15 := by
  sorry

#check berry_difference_change

end NUMINAMATH_CALUDE_berry_difference_change_l2374_237452


namespace NUMINAMATH_CALUDE_apartment_room_sizes_l2374_237417

/-- The apartment shared by Jenny, Martha, and Sam has three rooms with a total area of 800 square feet. Jenny's room is 100 square feet larger than Martha's, and Sam's room is 50 square feet smaller than Martha's. This theorem proves that Jenny's and Sam's rooms combined have an area of 550 square feet. -/
theorem apartment_room_sizes (total_area : ℝ) (martha_size : ℝ) 
  (h1 : total_area = 800)
  (h2 : martha_size + (martha_size + 100) + (martha_size - 50) = total_area) :
  (martha_size + 100) + (martha_size - 50) = 550 := by
  sorry

end NUMINAMATH_CALUDE_apartment_room_sizes_l2374_237417


namespace NUMINAMATH_CALUDE_two_digit_multiple_problem_l2374_237473

theorem two_digit_multiple_problem : ∃ (n : ℕ), 
  10 ≤ n ∧ n < 100 ∧  -- two-digit number
  n % 2 = 0 ∧  -- multiple of 2
  (n + 1) % 3 = 0 ∧  -- adding 1 results in multiple of 3
  (n + 2) % 4 = 0 ∧  -- adding 2 results in multiple of 4
  (n + 3) % 5 = 0 ∧  -- adding 3 results in multiple of 5
  (∀ m : ℕ, 10 ≤ m ∧ m < n → 
    (m % 2 ≠ 0 ∨ (m + 1) % 3 ≠ 0 ∨ (m + 2) % 4 ≠ 0 ∨ (m + 3) % 5 ≠ 0)) ∧
  n = 62 := by
sorry

end NUMINAMATH_CALUDE_two_digit_multiple_problem_l2374_237473


namespace NUMINAMATH_CALUDE_transaction_gain_per_year_l2374_237427

def principal : ℝ := 5000
def duration : ℕ := 2
def borrow_rate_year1 : ℝ := 0.04
def borrow_rate_year2 : ℝ := 0.06
def lend_rate_year1 : ℝ := 0.05
def lend_rate_year2 : ℝ := 0.07

theorem transaction_gain_per_year : 
  let amount_lend_year1 := principal * (1 + lend_rate_year1)
  let amount_lend_year2 := amount_lend_year1 * (1 + lend_rate_year2)
  let interest_earned := amount_lend_year2 - principal
  let amount_borrow_year1 := principal * (1 + borrow_rate_year1)
  let amount_borrow_year2 := amount_borrow_year1 * (1 + borrow_rate_year2)
  let interest_paid := amount_borrow_year2 - principal
  let total_gain := interest_earned - interest_paid
  let gain_per_year := total_gain / duration
  gain_per_year = 52.75 := by sorry

end NUMINAMATH_CALUDE_transaction_gain_per_year_l2374_237427


namespace NUMINAMATH_CALUDE_regular_octagon_exterior_angle_regular_octagon_exterior_angle_is_45_l2374_237407

/-- The measure of an exterior angle in a regular octagon is 45 degrees. -/
theorem regular_octagon_exterior_angle : ℝ :=
  let n : ℕ := 8  -- number of sides in an octagon
  let interior_angle_sum : ℝ := (n - 2) * 180
  let interior_angle : ℝ := interior_angle_sum / n
  let exterior_angle : ℝ := 180 - interior_angle
  exterior_angle

/-- The measure of an exterior angle in a regular octagon is 45 degrees. -/
theorem regular_octagon_exterior_angle_is_45 :
  regular_octagon_exterior_angle = 45 := by
  sorry

end NUMINAMATH_CALUDE_regular_octagon_exterior_angle_regular_octagon_exterior_angle_is_45_l2374_237407


namespace NUMINAMATH_CALUDE_prob_one_white_two_drawn_correct_expectation_white_three_drawn_correct_l2374_237461

/-- The number of black balls in the bag -/
def num_black : ℕ := 2

/-- The number of white balls in the bag -/
def num_white : ℕ := 3

/-- The total number of balls in the bag -/
def total_balls : ℕ := num_black + num_white

/-- The probability of drawing exactly one white ball when two balls are randomly drawn -/
def prob_one_white_two_drawn : ℚ := 3 / 5

/-- The mathematical expectation of the number of white balls when three balls are randomly drawn -/
def expectation_white_three_drawn : ℚ := 18 / 10

/-- Theorem stating the probability of drawing exactly one white ball when two balls are randomly drawn -/
theorem prob_one_white_two_drawn_correct :
  prob_one_white_two_drawn = (num_black * num_white : ℚ) / ((total_balls * (total_balls - 1)) / 2) :=
sorry

/-- Theorem stating the mathematical expectation of the number of white balls when three balls are randomly drawn -/
theorem expectation_white_three_drawn_correct :
  expectation_white_three_drawn = 
    (1 * (num_black * num_black * num_white : ℚ) +
     2 * (num_black * num_white * (num_white - 1)) +
     3 * (num_white * (num_white - 1) * (num_white - 2))) /
    ((total_balls * (total_balls - 1) * (total_balls - 2)) / 6) :=
sorry

end NUMINAMATH_CALUDE_prob_one_white_two_drawn_correct_expectation_white_three_drawn_correct_l2374_237461


namespace NUMINAMATH_CALUDE_angle_equality_l2374_237486

theorem angle_equality (θ : Real) (A B : Set Real) : 
  A = {1, Real.cos θ} → B = {1/2, 1} → A = B → 0 < θ → θ < π/2 → θ = π/3 := by
  sorry

end NUMINAMATH_CALUDE_angle_equality_l2374_237486


namespace NUMINAMATH_CALUDE_sum_product_inequality_l2374_237480

theorem sum_product_inequality (a b c : ℝ) (h : a + b + c = 0) : a * b + b * c + c * a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_product_inequality_l2374_237480


namespace NUMINAMATH_CALUDE_least_multiple_ending_zero_l2374_237412

theorem least_multiple_ending_zero : ∃ n : ℕ, 
  (∀ k : ℕ, k ≤ 10 → k > 0 → n % k = 0) ∧ 
  (n % 10 = 0) ∧
  (∀ m : ℕ, m < n → (∃ k : ℕ, k ≤ 10 ∧ k > 0 ∧ m % k ≠ 0) ∨ m % 10 ≠ 0) ∧
  n = 2520 :=
by sorry

end NUMINAMATH_CALUDE_least_multiple_ending_zero_l2374_237412


namespace NUMINAMATH_CALUDE_sequence_2023rd_term_l2374_237485

theorem sequence_2023rd_term (a : ℕ → ℚ) (h1 : a 1 = 2) 
  (h2 : ∀ n : ℕ, 1 / a n - 1 / a (n + 1) - 1 / (a n * a (n + 1)) = 1) : 
  a 2023 = -1/2 := by
sorry

end NUMINAMATH_CALUDE_sequence_2023rd_term_l2374_237485


namespace NUMINAMATH_CALUDE_largest_two_digit_remainder_two_l2374_237408

theorem largest_two_digit_remainder_two : ∃ n : ℕ, 
  (n ≥ 10 ∧ n ≤ 99) ∧ 
  n % 13 = 2 ∧ 
  (∀ m : ℕ, (m ≥ 10 ∧ m ≤ 99) ∧ m % 13 = 2 → m ≤ n) ∧
  n = 93 := by
sorry

end NUMINAMATH_CALUDE_largest_two_digit_remainder_two_l2374_237408


namespace NUMINAMATH_CALUDE_total_amount_calculation_l2374_237451

/-- Calculate the total amount paid for a suit, shoes, dress shirt, and tie, considering discounts and taxes. -/
theorem total_amount_calculation (suit_price suit_discount suit_tax_rate : ℚ)
                                 (shoes_price shoes_discount shoes_tax_rate : ℚ)
                                 (shirt_price shirt_tax_rate : ℚ)
                                 (tie_price tie_tax_rate : ℚ)
                                 (shirt_tie_discount_rate : ℚ) :
  suit_price = 430 →
  suit_discount = 100 →
  suit_tax_rate = 5/100 →
  shoes_price = 190 →
  shoes_discount = 30 →
  shoes_tax_rate = 7/100 →
  shirt_price = 80 →
  shirt_tax_rate = 6/100 →
  tie_price = 50 →
  tie_tax_rate = 4/100 →
  shirt_tie_discount_rate = 20/100 →
  ∃ total_amount : ℚ,
    total_amount = (suit_price - suit_discount) * (1 + suit_tax_rate) +
                   (shoes_price - shoes_discount) * (1 + shoes_tax_rate) +
                   ((shirt_price + tie_price) * (1 - shirt_tie_discount_rate)) * 
                   ((shirt_price / (shirt_price + tie_price)) * (1 + shirt_tax_rate) +
                    (tie_price / (shirt_price + tie_price)) * (1 + tie_tax_rate)) ∧
    total_amount = 627.14 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_calculation_l2374_237451


namespace NUMINAMATH_CALUDE_unique_fixed_point_for_rotation_invariant_function_l2374_237454

/-- A function is invariant under π rotation around the origin -/
def RotationInvariant (f : ℝ → ℝ) : Prop :=
  ∀ x y, f x = y ↔ f (-x) = -y

/-- The main theorem -/
theorem unique_fixed_point_for_rotation_invariant_function (f : ℝ → ℝ) 
    (h : RotationInvariant f) : 
    ∃! x, f x = x :=
  sorry

end NUMINAMATH_CALUDE_unique_fixed_point_for_rotation_invariant_function_l2374_237454


namespace NUMINAMATH_CALUDE_trapezium_other_side_length_l2374_237484

theorem trapezium_other_side_length (a b h : ℝ) (area : ℝ) : 
  a = 20 → h = 13 → area = 247 → area = (1/2) * (a + b) * h → b = 18 := by
  sorry

end NUMINAMATH_CALUDE_trapezium_other_side_length_l2374_237484


namespace NUMINAMATH_CALUDE_tan_alpha_value_l2374_237462

theorem tan_alpha_value (α : Real) (h : Real.tan (α + π/4) = 9) : Real.tan α = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l2374_237462


namespace NUMINAMATH_CALUDE_even_product_sufficient_not_necessary_l2374_237483

-- Define the concept of an even function
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Define the product of two functions
def ProductFunc (f g : ℝ → ℝ) : ℝ → ℝ := λ x ↦ f x * g x

-- Theorem statement
theorem even_product_sufficient_not_necessary :
  (∀ f g : ℝ → ℝ, IsEven f ∧ IsEven g → IsEven (ProductFunc f g)) ∧
  (∃ f g : ℝ → ℝ, IsEven (ProductFunc f g) ∧ (¬IsEven f ∨ ¬IsEven g)) :=
sorry

end NUMINAMATH_CALUDE_even_product_sufficient_not_necessary_l2374_237483


namespace NUMINAMATH_CALUDE_geometric_sequence_middle_term_l2374_237493

theorem geometric_sequence_middle_term (a : ℝ) : 
  (∃ r : ℝ, 2 * r = a ∧ a * r = 8) → a = 4 ∨ a = -4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_middle_term_l2374_237493


namespace NUMINAMATH_CALUDE_secret_codes_count_l2374_237422

/-- The number of colors available in the game -/
def num_colors : ℕ := 8

/-- The number of slots in the game -/
def num_slots : ℕ := 5

/-- The total number of possible secret codes -/
def total_codes : ℕ := num_colors ^ num_slots

/-- Theorem stating that the total number of possible secret codes is 32768 -/
theorem secret_codes_count : total_codes = 32768 := by
  sorry

end NUMINAMATH_CALUDE_secret_codes_count_l2374_237422


namespace NUMINAMATH_CALUDE_exactly_two_valid_A_values_l2374_237467

/-- A function that checks if a number is divisible by 8 based on its last three digits -/
def isDivisibleBy8 (n : ℕ) : Prop :=
  n % 8 = 0

/-- A function that constructs the number 451,2A8 given A -/
def constructNumber (A : ℕ) : ℕ :=
  451200 + A * 10 + 8

/-- The main theorem stating that there are exactly 2 single-digit values of A satisfying both conditions -/
theorem exactly_two_valid_A_values :
  ∃! (S : Finset ℕ), S.card = 2 ∧ 
    (∀ A ∈ S, A < 10 ∧ 120 % A = 0 ∧ isDivisibleBy8 (constructNumber A)) ∧
    (∀ A < 10, 120 % A = 0 ∧ isDivisibleBy8 (constructNumber A) → A ∈ S) :=
sorry

end NUMINAMATH_CALUDE_exactly_two_valid_A_values_l2374_237467


namespace NUMINAMATH_CALUDE_family_ages_exist_and_unique_l2374_237455

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def digit_product (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) * digit_product (n / 10)

def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + digit_sum (n / 10)

theorem family_ages_exist_and_unique :
  ∃! (father mother daughter son : ℕ),
    is_perfect_square father ∧
    digit_product father = mother ∧
    digit_sum father = daughter ∧
    digit_sum mother = son ∧
    father ≤ 121 ∧
    mother > 0 ∧
    daughter > 0 ∧
    son > 0 :=
by sorry

end NUMINAMATH_CALUDE_family_ages_exist_and_unique_l2374_237455


namespace NUMINAMATH_CALUDE_first_turkey_weight_is_6_l2374_237460

/-- The weight of the first turkey in kilograms -/
def first_turkey_weight : ℝ := 6

/-- The weight of the second turkey in kilograms -/
def second_turkey_weight : ℝ := 9

/-- The weight of the third turkey in kilograms -/
def third_turkey_weight : ℝ := 2 * second_turkey_weight

/-- The cost of turkey per kilogram in dollars -/
def cost_per_kg : ℝ := 2

/-- The total amount spent on turkeys in dollars -/
def total_spent : ℝ := 66

/-- Theorem stating that the weight of the first turkey is 6 kilograms -/
theorem first_turkey_weight_is_6 :
  first_turkey_weight = 6 ∧
  second_turkey_weight = 9 ∧
  third_turkey_weight = 2 * second_turkey_weight ∧
  cost_per_kg = 2 ∧
  total_spent = 66 ∧
  total_spent = cost_per_kg * (first_turkey_weight + second_turkey_weight + third_turkey_weight) :=
by
  sorry

#check first_turkey_weight_is_6

end NUMINAMATH_CALUDE_first_turkey_weight_is_6_l2374_237460


namespace NUMINAMATH_CALUDE_box_width_proof_l2374_237404

/-- Proves that the width of a box with given dimensions and constraints is 18 cm -/
theorem box_width_proof (length height : ℝ) (cube_volume min_cubes : ℕ) :
  length = 7 →
  height = 3 →
  cube_volume = 9 →
  min_cubes = 42 →
  ∃ width : ℝ,
    width * length * height = min_cubes * cube_volume ∧
    width = 18 := by
  sorry

end NUMINAMATH_CALUDE_box_width_proof_l2374_237404


namespace NUMINAMATH_CALUDE_cafe_tables_needed_l2374_237400

def base5ToDecimal (n : Nat) : Nat :=
  (n / 100) * 25 + ((n / 10) % 10) * 5 + (n % 10)

def customersPerTable : Nat := 3

def cafeCapacity : Nat := 123

theorem cafe_tables_needed :
  let decimalCapacity := base5ToDecimal cafeCapacity
  ⌈(decimalCapacity : ℚ) / customersPerTable⌉ = 13 := by
  sorry

end NUMINAMATH_CALUDE_cafe_tables_needed_l2374_237400


namespace NUMINAMATH_CALUDE_line_slope_intercept_sum_l2374_237438

/-- Given a line with slope 4 passing through (2, -1), prove m + b = -5 -/
theorem line_slope_intercept_sum (m b : ℝ) : 
  m = 4 ∧ 
  -1 = m * 2 + b →
  m + b = -5 := by
sorry

end NUMINAMATH_CALUDE_line_slope_intercept_sum_l2374_237438


namespace NUMINAMATH_CALUDE_sally_payment_l2374_237453

/-- The amount Sally needs to pay out of pocket to buy books for her students -/
def sally_out_of_pocket (budget : ℚ) (num_students : ℕ) (reading_book_price : ℚ) 
  (math_book_price : ℚ) (discount_rate : ℚ) (discount_threshold : ℕ) : ℚ :=
  let total_reading_books := num_students * reading_book_price
  let discounted_reading_books := if num_students ≥ discount_threshold
    then total_reading_books * (1 - discount_rate)
    else total_reading_books
  let total_math_books := num_students * math_book_price
  let total_cost := discounted_reading_books + total_math_books
  max (total_cost - budget) 0

/-- Theorem stating that Sally needs to pay $467.50 out of pocket -/
theorem sally_payment : 
  sally_out_of_pocket 320 35 15 9 (1/10) 25 = 467.5 := by
  sorry

end NUMINAMATH_CALUDE_sally_payment_l2374_237453


namespace NUMINAMATH_CALUDE_equal_color_polygons_l2374_237491

/-- A color type to represent different colors of vertices -/
inductive Color

/-- A structure representing a regular polygon -/
structure RegularPolygon where
  vertices : Finset ℝ × ℝ
  is_regular : Bool

/-- A structure representing a colored regular n-gon -/
structure ColoredRegularNGon where
  n : ℕ
  vertices : Finset (ℝ × ℝ)
  colors : Finset Color
  vertex_coloring : (ℝ × ℝ) → Color
  is_regular : Bool
  num_vertices : vertices.card = n

/-- A function that returns the set of regular polygons formed by vertices of each color -/
def colorPolygons (ngon : ColoredRegularNGon) : Finset RegularPolygon :=
  sorry

/-- The main theorem statement -/
theorem equal_color_polygons (ngon : ColoredRegularNGon) :
  ∃ (p q : RegularPolygon), p ∈ colorPolygons ngon ∧ q ∈ colorPolygons ngon ∧ p ≠ q ∧ p.vertices = q.vertices :=
sorry

end NUMINAMATH_CALUDE_equal_color_polygons_l2374_237491


namespace NUMINAMATH_CALUDE_dinner_slices_count_l2374_237441

/-- Represents the number of slices of pie served at different times -/
structure PieSlices where
  lunch_today : ℕ
  total_today : ℕ
  dinner_today : ℕ

/-- Theorem stating that given 7 slices served at lunch and 12 slices served in total today,
    the number of slices served at dinner is 5 -/
theorem dinner_slices_count (ps : PieSlices) 
  (h1 : ps.lunch_today = 7)
  (h2 : ps.total_today = 12)
  : ps.dinner_today = 5 := by
  sorry

end NUMINAMATH_CALUDE_dinner_slices_count_l2374_237441


namespace NUMINAMATH_CALUDE_jose_investment_is_45000_l2374_237443

/-- Represents the investment and profit scenario of Tom and Jose's shop --/
structure ShopInvestment where
  tom_investment : ℕ
  tom_months : ℕ
  jose_months : ℕ
  total_profit : ℕ
  jose_profit : ℕ

/-- Calculates Jose's investment based on the given conditions --/
def calculate_jose_investment (s : ShopInvestment) : ℕ :=
  let tom_time_investment := s.tom_investment * s.tom_months
  let tom_profit := s.total_profit - s.jose_profit
  (tom_time_investment * s.jose_profit) / (tom_profit * s.jose_months)

/-- Theorem stating that Jose's investment is 45000 given the problem conditions --/
theorem jose_investment_is_45000 :
  let s : ShopInvestment := {
    tom_investment := 30000,
    tom_months := 12,
    jose_months := 10,
    total_profit := 63000,
    jose_profit := 35000
  }
  calculate_jose_investment s = 45000 := by sorry


end NUMINAMATH_CALUDE_jose_investment_is_45000_l2374_237443
