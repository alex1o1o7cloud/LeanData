import Mathlib

namespace NUMINAMATH_CALUDE_b_work_days_l1753_175320

/-- The number of days it takes for two workers to complete a job together -/
def combined_days : ℕ := 16

/-- The number of days it takes for worker 'a' to complete the job alone -/
def a_days : ℕ := 24

/-- The work rate of a worker is the fraction of the job they complete in one day -/
def work_rate (days : ℕ) : ℚ := 1 / days

/-- The number of days it takes for worker 'b' to complete the job alone -/
def b_days : ℕ := 48

theorem b_work_days : 
  work_rate combined_days = work_rate a_days + work_rate b_days :=
sorry

end NUMINAMATH_CALUDE_b_work_days_l1753_175320


namespace NUMINAMATH_CALUDE_line_equation_slope_intercept_l1753_175374

/-- Given a line equation, prove its slope and y-intercept -/
theorem line_equation_slope_intercept :
  let line_eq : ℝ × ℝ → ℝ := λ p => 3 * (p.1 + 2) + (-7) * (p.2 - 4)
  ∃ m b : ℝ, m = 3 / 7 ∧ b = -34 / 7 ∧
    ∀ x y : ℝ, line_eq (x, y) = 0 ↔ y = m * x + b := by
  sorry

end NUMINAMATH_CALUDE_line_equation_slope_intercept_l1753_175374


namespace NUMINAMATH_CALUDE_carlos_earnings_l1753_175334

/-- Carlos's work hours and earnings problem -/
theorem carlos_earnings (hours_week1 hours_week2 : ℕ) (extra_earnings : ℚ) :
  hours_week1 = 12 →
  hours_week2 = 18 →
  extra_earnings = 36 →
  ∃ (hourly_wage : ℚ),
    hourly_wage * (hours_week2 - hours_week1) = extra_earnings ∧
    hourly_wage * (hours_week1 + hours_week2) = 180 :=
by sorry


end NUMINAMATH_CALUDE_carlos_earnings_l1753_175334


namespace NUMINAMATH_CALUDE_tina_book_expense_l1753_175387

def savings_june : ℤ := 27
def savings_july : ℤ := 14
def savings_august : ℤ := 21
def spent_on_shoes : ℤ := 17
def money_left : ℤ := 40

theorem tina_book_expense :
  ∃ (book_expense : ℤ),
    savings_june + savings_july + savings_august - book_expense - spent_on_shoes = money_left ∧
    book_expense = 5 := by
  sorry

end NUMINAMATH_CALUDE_tina_book_expense_l1753_175387


namespace NUMINAMATH_CALUDE_modified_deck_choose_two_l1753_175329

/-- Represents a modified deck of cards -/
structure ModifiedDeck :=
  (normal_suits : Nat)  -- Number of suits with 13 cards
  (reduced_suit : Nat)  -- Number of suits with 12 cards

/-- Calculates the number of ways to choose 2 cards from different suits in a modified deck -/
def choose_two_cards (deck : ModifiedDeck) : Nat :=
  sorry

/-- The theorem to be proved -/
theorem modified_deck_choose_two (d : ModifiedDeck) :
  d.normal_suits = 3 ∧ d.reduced_suit = 1 → choose_two_cards d = 1443 :=
sorry

end NUMINAMATH_CALUDE_modified_deck_choose_two_l1753_175329


namespace NUMINAMATH_CALUDE_original_average_proof_l1753_175312

theorem original_average_proof (n : ℕ) (original_avg new_avg : ℚ) :
  n > 0 →
  new_avg = 2 * original_avg →
  new_avg = 72 →
  original_avg = 36 := by
sorry

end NUMINAMATH_CALUDE_original_average_proof_l1753_175312


namespace NUMINAMATH_CALUDE_spherical_coordinate_equivalence_l1753_175381

/-- Represents a point in spherical coordinates -/
structure SphericalPoint where
  ρ : ℝ
  θ : ℝ
  φ : ℝ

/-- Checks if a SphericalPoint is in standard representation -/
def isStandardRepresentation (p : SphericalPoint) : Prop :=
  p.ρ > 0 ∧ 0 ≤ p.θ ∧ p.θ < 2 * Real.pi ∧ 0 ≤ p.φ ∧ p.φ ≤ Real.pi

/-- Theorem stating the equivalence of the given spherical coordinates -/
theorem spherical_coordinate_equivalence :
  let p1 := SphericalPoint.mk 4 (5 * Real.pi / 6) (9 * Real.pi / 4)
  let p2 := SphericalPoint.mk 4 (11 * Real.pi / 6) (Real.pi / 4)
  (p1.ρ = p2.ρ) ∧ 
  (p1.θ % (2 * Real.pi) = p2.θ % (2 * Real.pi)) ∧ 
  (p1.φ % (2 * Real.pi) = p2.φ % (2 * Real.pi)) ∧
  isStandardRepresentation p2 :=
by sorry

end NUMINAMATH_CALUDE_spherical_coordinate_equivalence_l1753_175381


namespace NUMINAMATH_CALUDE_jake_current_weight_l1753_175342

/-- Jake's current weight in pounds -/
def jake_weight : ℕ := 219

/-- Jake's sister's current weight in pounds -/
def sister_weight : ℕ := 318 - jake_weight

theorem jake_current_weight : 
  (jake_weight + sister_weight = 318) ∧ 
  (jake_weight - 12 = 2 * (sister_weight + 4)) → 
  jake_weight = 219 := by sorry

end NUMINAMATH_CALUDE_jake_current_weight_l1753_175342


namespace NUMINAMATH_CALUDE_range_of_a_l1753_175397

def A (a : ℝ) : Set ℝ := {x | (a * x - 1) * (a - x) > 0}

theorem range_of_a :
  ∀ a : ℝ, (2 ∈ A a ∧ 3 ∉ A a) ↔ a ∈ (Set.Ioo 2 3 ∪ Set.Ico (1/3) (1/2)) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1753_175397


namespace NUMINAMATH_CALUDE_jessy_reading_plan_l1753_175338

/-- The number of pages Jessy initially plans to read each time -/
def pages_per_reading : ℕ := sorry

/-- The total number of pages in the book -/
def total_pages : ℕ := 140

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The number of times Jessy reads per day -/
def readings_per_day : ℕ := 3

/-- The additional pages Jessy needs to read per day to achieve her goal -/
def additional_pages_per_day : ℕ := 2

theorem jessy_reading_plan :
  pages_per_reading = 6 ∧
  days_in_week * readings_per_day * pages_per_reading + 
  days_in_week * additional_pages_per_day = total_pages := by
  sorry

end NUMINAMATH_CALUDE_jessy_reading_plan_l1753_175338


namespace NUMINAMATH_CALUDE_range_of_a_l1753_175319

theorem range_of_a (a : ℝ) : 
  (∃ x₀ : ℝ, x₀ ∈ Set.Icc 0 1 ∧ 2^x₀ * (3 * x₀ + a) < 1) → a < 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1753_175319


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l1753_175395

theorem polynomial_divisibility : ∃ z : Polynomial ℤ, 
  X^44 + X^33 + X^22 + X^11 + 1 = (X^4 + X^3 + X^2 + X + 1) * z :=
sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l1753_175395


namespace NUMINAMATH_CALUDE_round_robin_chess_tournament_l1753_175349

theorem round_robin_chess_tournament (n : Nat) (h : n = 10) :
  let total_games := n * (n - 1) / 2
  total_games = 45 := by
  sorry

end NUMINAMATH_CALUDE_round_robin_chess_tournament_l1753_175349


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_l1753_175360

theorem quadratic_roots_sum (p q : ℝ) : 
  p^2 - 6*p + 8 = 0 → q^2 - 6*q + 8 = 0 → p^3 + p^4*q^2 + p^2*q^4 + q^3 = 1352 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_l1753_175360


namespace NUMINAMATH_CALUDE_probability_knows_cpp_l1753_175388

/-- Given a software company with the following characteristics:
  * 600 total employees
  * 3/10 of employees know C++
  * 5/12 of employees know Java
  * 4/15 of employees know Python
  * 2/25 of employees know neither C++, Java, nor Python
  Prove that the probability of a randomly selected employee knowing C++ is 3/10. -/
theorem probability_knows_cpp (total_employees : ℕ) 
  (fraction_cpp : ℚ) (fraction_java : ℚ) (fraction_python : ℚ) (fraction_none : ℚ)
  (h1 : total_employees = 600)
  (h2 : fraction_cpp = 3 / 10)
  (h3 : fraction_java = 5 / 12)
  (h4 : fraction_python = 4 / 15)
  (h5 : fraction_none = 2 / 25) :
  (↑total_employees * fraction_cpp : ℚ) / total_employees = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_knows_cpp_l1753_175388


namespace NUMINAMATH_CALUDE_equal_population_after_17_years_l1753_175315

/-- The number of years needed for two villages' populations to become equal -/
def years_until_equal_population (x_initial : ℕ) (x_decrease : ℕ) (y_initial : ℕ) (y_increase : ℕ) : ℕ :=
  (x_initial - y_initial) / (x_decrease + y_increase)

/-- Theorem stating that the populations of Village X and Village Y will be equal after 17 years -/
theorem equal_population_after_17_years :
  years_until_equal_population 76000 1200 42000 800 = 17 := by
  sorry

end NUMINAMATH_CALUDE_equal_population_after_17_years_l1753_175315


namespace NUMINAMATH_CALUDE_inscribed_triangle_angle_60_l1753_175372

/-- Represents a triangle inscribed in a circle -/
structure InscribedTriangle where
  /-- Measure of arc PQ -/
  arc_pq : ℝ
  /-- Measure of arc QR -/
  arc_qr : ℝ
  /-- Measure of arc RP -/
  arc_rp : ℝ
  /-- The sum of all arcs is 360° -/
  sum_arcs : arc_pq + arc_qr + arc_rp = 360

/-- Theorem: If a triangle is inscribed in a circle with the given arc measures,
    then one of its interior angles is 60° -/
theorem inscribed_triangle_angle_60 (t : InscribedTriangle)
  (h1 : ∃ x : ℝ, t.arc_pq = x + 80 ∧ t.arc_qr = 3*x - 30 ∧ t.arc_rp = 2*x + 10) :
  ∃ θ : ℝ, θ = 60 ∧ (θ = t.arc_qr / 2 ∨ θ = t.arc_rp / 2 ∨ θ = t.arc_pq / 2) := by
  sorry

end NUMINAMATH_CALUDE_inscribed_triangle_angle_60_l1753_175372


namespace NUMINAMATH_CALUDE_sheep_transaction_gain_l1753_175325

/-- Calculates the percent gain on a sheep transaction given specific conditions. -/
theorem sheep_transaction_gain : ∀ (x : ℝ),
  x > 0 →  -- x represents the cost per sheep
  let total_cost : ℝ := 850 * x
  let first_sale_revenue : ℝ := total_cost
  let first_sale_price_per_sheep : ℝ := first_sale_revenue / 800
  let second_sale_price_per_sheep : ℝ := first_sale_price_per_sheep * 1.1
  let second_sale_revenue : ℝ := second_sale_price_per_sheep * 50
  let total_revenue : ℝ := first_sale_revenue + second_sale_revenue
  let profit : ℝ := total_revenue - total_cost
  let percent_gain : ℝ := (profit / total_cost) * 100
  percent_gain = 6.875 := by
  sorry


end NUMINAMATH_CALUDE_sheep_transaction_gain_l1753_175325


namespace NUMINAMATH_CALUDE_base_is_ten_l1753_175318

-- Define a function to convert a number from base h to decimal
def to_decimal (digits : List Nat) (h : Nat) : Nat :=
  digits.foldr (fun d acc => d + h * acc) 0

-- Define a function to check if the equation holds in base h
def equation_holds (h : Nat) : Prop :=
  to_decimal [5, 7, 3, 4] h + to_decimal [6, 4, 2, 1] h = to_decimal [1, 4, 1, 5, 5] h

-- Theorem statement
theorem base_is_ten : ∃ h, h = 10 ∧ equation_holds h := by
  sorry

end NUMINAMATH_CALUDE_base_is_ten_l1753_175318


namespace NUMINAMATH_CALUDE_sequence_property_l1753_175399

theorem sequence_property (a : ℕ → ℕ) (p : ℕ) : 
  (∀ (m n : ℕ), m ≥ n → a (m + n) + a (m - n) + 2 * m - 2 * n - 1 = (a (2 * m) + a (2 * n)) / 2) →
  a 1 = 0 →
  a p = 2019 * 2019 →
  p = 2020 := by
  sorry

end NUMINAMATH_CALUDE_sequence_property_l1753_175399


namespace NUMINAMATH_CALUDE_sqrt_five_addition_l1753_175375

theorem sqrt_five_addition : 2 * Real.sqrt 5 + 3 * Real.sqrt 5 = 5 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_five_addition_l1753_175375


namespace NUMINAMATH_CALUDE_power_division_result_l1753_175398

theorem power_division_result : 3^12 / 27^2 = 729 := by
  -- Define 27 as 3^3
  have h1 : 27 = 3^3 := by sorry
  
  -- Rewrite the division using the definition of 27
  have h2 : 3^12 / 27^2 = 3^12 / (3^3)^2 := by sorry
  
  -- Simplify the exponents
  have h3 : 3^12 / (3^3)^2 = 3^(12 - 3*2) := by sorry
  
  -- Evaluate the final result
  have h4 : 3^(12 - 3*2) = 3^6 := by sorry
  have h5 : 3^6 = 729 := by sorry
  
  -- Combine all steps
  sorry

#check power_division_result

end NUMINAMATH_CALUDE_power_division_result_l1753_175398


namespace NUMINAMATH_CALUDE_expression_evaluation_l1753_175357

theorem expression_evaluation : 
  Real.sin (π / 4) ^ 2 - Real.sqrt 27 + (1 / 2) * ((Real.sqrt 3 - 2006) ^ 0) + 6 * Real.tan (π / 6) = 1 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1753_175357


namespace NUMINAMATH_CALUDE_angle_sum_zero_l1753_175365

theorem angle_sum_zero (α β : ℝ) (h_acute_α : 0 < α ∧ α < π / 2) (h_acute_β : 0 < β ∧ β < π / 2)
  (h_eq1 : 4 * (Real.cos α)^2 + 3 * (Real.cos β)^2 = 2)
  (h_eq2 : 4 * Real.sin (2 * α) + 3 * Real.sin (2 * β) = 0) :
  α + 3 * β = 0 := by sorry

end NUMINAMATH_CALUDE_angle_sum_zero_l1753_175365


namespace NUMINAMATH_CALUDE_taxi_charge_calculation_l1753_175376

/-- Calculates the total charge for a taxi trip -/
def totalCharge (initialFee : ℚ) (additionalChargePerIncrement : ℚ) (incrementDistance : ℚ) (tripDistance : ℚ) : ℚ :=
  initialFee + (tripDistance / incrementDistance).floor * additionalChargePerIncrement

/-- Theorem: The total charge for a 3.6-mile trip with given fees is $5.50 -/
theorem taxi_charge_calculation :
  let initialFee : ℚ := 235 / 100
  let additionalChargePerIncrement : ℚ := 35 / 100
  let incrementDistance : ℚ := 2 / 5
  let tripDistance : ℚ := 36 / 10
  totalCharge initialFee additionalChargePerIncrement incrementDistance tripDistance = 550 / 100 := by
  sorry

#eval totalCharge (235/100) (35/100) (2/5) (36/10)

end NUMINAMATH_CALUDE_taxi_charge_calculation_l1753_175376


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l1753_175331

theorem smallest_n_congruence : ∃ (n : ℕ), n > 0 ∧ (5 * n) % 26 = 1463 % 26 ∧ ∀ (m : ℕ), m > 0 → (5 * m) % 26 = 1463 % 26 → n ≤ m := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l1753_175331


namespace NUMINAMATH_CALUDE_division_fraction_equality_l1753_175373

theorem division_fraction_equality : (2 / 7) / (1 / 14) = 4 := by sorry

end NUMINAMATH_CALUDE_division_fraction_equality_l1753_175373


namespace NUMINAMATH_CALUDE_intersection_point_coordinates_l1753_175354

-- Define the triangle ABC and points D, E, P
variable (A B C D E P : ℝ × ℝ)

-- Define the conditions
def D_on_BC_extended (A B C D : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, t > 1 ∧ D = B + t • (C - B)

def E_on_AC (A C E : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 < t ∧ t < 1 ∧ E = A + t • (C - A)

def BD_DC_ratio (B C D : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, t > 0 ∧ D = B + (2/3) • (C - B)

def AE_EC_ratio (A C E : ℝ × ℝ) : Prop :=
  E = A + (2/3) • (C - A)

def P_on_BE (B E P : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = B + t • (E - B)

def P_on_AD (A D P : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = A + t • (D - A)

-- State the theorem
theorem intersection_point_coordinates
  (h1 : D_on_BC_extended A B C D)
  (h2 : E_on_AC A C E)
  (h3 : BD_DC_ratio B C D)
  (h4 : AE_EC_ratio A C E)
  (h5 : P_on_BE B E P)
  (h6 : P_on_AD A D P) :
  P = (1/3) • A + (1/2) • B + (1/6) • C :=
sorry

end NUMINAMATH_CALUDE_intersection_point_coordinates_l1753_175354


namespace NUMINAMATH_CALUDE_original_equals_scientific_l1753_175345

/-- The number to be expressed in scientific notation -/
def original_number : ℝ := 1650000

/-- The scientific notation representation -/
def scientific_notation : ℝ := 1.65 * (10 ^ 6)

/-- Theorem stating that the original number is equal to its scientific notation representation -/
theorem original_equals_scientific : original_number = scientific_notation := by
  sorry

end NUMINAMATH_CALUDE_original_equals_scientific_l1753_175345


namespace NUMINAMATH_CALUDE_train_platform_equal_length_l1753_175326

/-- Given a train and platform with specific properties, prove that their lengths are equal --/
theorem train_platform_equal_length 
  (train_speed : ℝ) 
  (train_length : ℝ) 
  (crossing_time : ℝ) 
  (h1 : train_speed = 108 * 1000 / 60) -- 108 km/hr converted to m/min
  (h2 : train_length = 900)
  (h3 : crossing_time = 1) :
  train_length = train_speed * crossing_time - train_length := by
  sorry

#check train_platform_equal_length

end NUMINAMATH_CALUDE_train_platform_equal_length_l1753_175326


namespace NUMINAMATH_CALUDE_tan_negative_405_l1753_175362

-- Define the tangent function
noncomputable def tan (θ : ℝ) : ℝ := Real.tan θ

-- Define the property of tangent periodicity
axiom tan_periodic (θ : ℝ) (n : ℤ) : tan θ = tan (θ + n * 360)

-- Define the value of tan(45°)
axiom tan_45 : tan 45 = 1

-- Theorem to prove
theorem tan_negative_405 : tan (-405) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_negative_405_l1753_175362


namespace NUMINAMATH_CALUDE_interest_calculation_l1753_175301

def total_investment : ℝ := 33000
def rate1 : ℝ := 0.04
def rate2 : ℝ := 0.0225
def partial_investment : ℝ := 13000

theorem interest_calculation :
  ∃ (investment1 investment2 : ℝ),
    investment1 + investment2 = total_investment ∧
    (investment1 = partial_investment ∨ investment2 = partial_investment) ∧
    investment1 * rate1 + investment2 * rate2 = 970 :=
by sorry

end NUMINAMATH_CALUDE_interest_calculation_l1753_175301


namespace NUMINAMATH_CALUDE_unique_solution_system_l1753_175344

theorem unique_solution_system (x y z : ℝ) : 
  (x^2 - 23*y - 25*z = -681) ∧
  (y^2 - 21*x - 21*z = -419) ∧
  (z^2 - 19*x - 21*y = -313) ↔
  (x = 20 ∧ y = 22 ∧ z = 23) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_system_l1753_175344


namespace NUMINAMATH_CALUDE_six_by_six_grid_squares_l1753_175341

/-- The number of squares of a given size in a grid --/
def count_squares (grid_size : ℕ) (square_size : ℕ) : ℕ :=
  (grid_size + 1 - square_size) ^ 2

/-- The total number of squares in a 6x6 grid --/
def total_squares (grid_size : ℕ) : ℕ :=
  (count_squares grid_size 1) + (count_squares grid_size 2) +
  (count_squares grid_size 3) + (count_squares grid_size 4)

/-- Theorem: The total number of squares in a 6x6 grid is 86 --/
theorem six_by_six_grid_squares :
  total_squares 6 = 86 := by
  sorry


end NUMINAMATH_CALUDE_six_by_six_grid_squares_l1753_175341


namespace NUMINAMATH_CALUDE_no_valid_gnomon_tiling_l1753_175356

/-- A gnomon is a figure formed by removing one unit square from a 2x2 square -/
def Gnomon : Type := Unit

/-- Represents a tiling of an m × n rectangle with gnomons -/
def GnomonTiling (m n : ℕ) := Unit

/-- Predicate to check if a tiling satisfies the no-rectangle condition -/
def NoRectangleCondition (tiling : GnomonTiling m n) : Prop := sorry

/-- Predicate to check if a tiling satisfies the no-four-vertex condition -/
def NoFourVertexCondition (tiling : GnomonTiling m n) : Prop := sorry

theorem no_valid_gnomon_tiling (m n : ℕ) :
  ¬∃ (tiling : GnomonTiling m n), NoRectangleCondition tiling ∧ NoFourVertexCondition tiling := by
  sorry

end NUMINAMATH_CALUDE_no_valid_gnomon_tiling_l1753_175356


namespace NUMINAMATH_CALUDE_unique_solution_l1753_175347

/-- The equation from the problem -/
def equation (x y : ℝ) : Prop :=
  11 * x^2 + 2 * x * y + 9 * y^2 + 8 * x - 12 * y + 6 = 0

/-- There exists exactly one pair of real numbers (x, y) that satisfies the equation -/
theorem unique_solution : ∃! p : ℝ × ℝ, equation p.1 p.2 := by sorry

end NUMINAMATH_CALUDE_unique_solution_l1753_175347


namespace NUMINAMATH_CALUDE_triangle_properties_l1753_175392

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_properties (t : Triangle) 
  (h1 : 2 * Real.sin t.A * Real.cos t.B = 2 * Real.sin t.C - Real.sin t.B)
  (h2 : t.a = 4 * Real.sqrt 3)
  (h3 : t.b + t.c = 8) :
  t.A = π / 3 ∧ (1/2 * t.b * t.c * Real.sin t.A = 4 * Real.sqrt 3 / 3) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l1753_175392


namespace NUMINAMATH_CALUDE_original_number_proof_l1753_175396

theorem original_number_proof (x : ℚ) : (1 / x) - 2 = 5 / 2 → x = 2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l1753_175396


namespace NUMINAMATH_CALUDE_investment_plans_count_l1753_175313

/-- The number of cities available for investment -/
def num_cities : ℕ := 5

/-- The number of projects to be invested -/
def num_projects : ℕ := 3

/-- The maximum number of projects that can be invested in a single city -/
def max_projects_per_city : ℕ := 2

/-- The function that calculates the number of investment plans -/
def num_investment_plans : ℕ :=
  -- The actual calculation would go here
  120

/-- Theorem stating that the number of investment plans is 120 -/
theorem investment_plans_count :
  num_investment_plans = 120 := by sorry

end NUMINAMATH_CALUDE_investment_plans_count_l1753_175313


namespace NUMINAMATH_CALUDE_expression_simplification_l1753_175394

theorem expression_simplification 
  (x y z w : ℝ) 
  (hx : x ≠ 2) 
  (hy : y ≠ 3) 
  (hz : z ≠ 4) 
  (hw : w ≠ 5) 
  (h1 : y ≠ 6) 
  (h2 : w ≠ 4) 
  (h3 : z ≠ 6) :
  (x - 2) / (6 - y) * (y - 3) / (2 - x) * (z - 4) / (3 - y) * 
  (6 - z) / (4 - w) * (w - 5) / (z - 6) * (x + 1) / (5 - w) = 1 := by
  sorry

#check expression_simplification

end NUMINAMATH_CALUDE_expression_simplification_l1753_175394


namespace NUMINAMATH_CALUDE_min_sum_distances_to_lines_l1753_175314

/-- The minimum sum of distances from a point on the parabola y² = 4x to two specific lines -/
theorem min_sum_distances_to_lines : ∃ (a : ℝ),
  let P : ℝ × ℝ := (a^2, 2*a)
  let d₁ : ℝ := |4*a^2 - 6*a + 6| / 5  -- Distance to line 4x - 3y + 6 = 0
  let d₂ : ℝ := a^2                    -- Distance to line x = 0
  (∀ b : ℝ, d₁ + d₂ ≤ |4*b^2 - 6*b + 6| / 5 + b^2) ∧ 
  d₁ + d₂ = 1 :=
sorry

end NUMINAMATH_CALUDE_min_sum_distances_to_lines_l1753_175314


namespace NUMINAMATH_CALUDE_correct_fill_in_l1753_175302

def sentence (phrase : String) : String :=
  s!"It will not be {phrase} we meet again."

def correctPhrase : String := "long before"

theorem correct_fill_in :
  sentence correctPhrase = "It will not be long before we meet again." :=
by sorry

end NUMINAMATH_CALUDE_correct_fill_in_l1753_175302


namespace NUMINAMATH_CALUDE_monomial_degree_implications_l1753_175304

-- Define the condition
def is_monomial_of_degree_5 (a : ℝ) : Prop :=
  2 + (1 + a) = 5

-- Theorem statement
theorem monomial_degree_implications (a : ℝ) 
  (h : is_monomial_of_degree_5 a) : 
  a^3 + 1 = 9 ∧ 
  (a + 1) * (a^2 - a + 1) = 9 ∧ 
  a^3 + 1 = (a + 1) * (a^2 - a + 1) := by
  sorry

end NUMINAMATH_CALUDE_monomial_degree_implications_l1753_175304


namespace NUMINAMATH_CALUDE_average_monthly_balance_l1753_175379

def monthly_balances : List ℚ := [150, 250, 100, 200, 300]

theorem average_monthly_balance : 
  (monthly_balances.sum / monthly_balances.length : ℚ) = 200 := by
  sorry

end NUMINAMATH_CALUDE_average_monthly_balance_l1753_175379


namespace NUMINAMATH_CALUDE_blue_hat_cost_is_6_l1753_175317

-- Define the total number of hats
def total_hats : ℕ := 85

-- Define the cost of each green hat
def green_hat_cost : ℕ := 7

-- Define the total price
def total_price : ℕ := 540

-- Define the number of green hats
def green_hats : ℕ := 30

-- Define the number of blue hats
def blue_hats : ℕ := total_hats - green_hats

-- Define the cost of green hats
def green_hats_cost : ℕ := green_hats * green_hat_cost

-- Define the cost of blue hats
def blue_hats_cost : ℕ := total_price - green_hats_cost

-- Theorem: The cost of each blue hat is $6
theorem blue_hat_cost_is_6 : blue_hats_cost / blue_hats = 6 := by
  sorry

end NUMINAMATH_CALUDE_blue_hat_cost_is_6_l1753_175317


namespace NUMINAMATH_CALUDE_negative_fractions_comparison_l1753_175361

theorem negative_fractions_comparison : -4/5 < -2/3 := by
  sorry

end NUMINAMATH_CALUDE_negative_fractions_comparison_l1753_175361


namespace NUMINAMATH_CALUDE_floor_sum_inequality_l1753_175350

theorem floor_sum_inequality (x y : ℝ) : ⌊x + y⌋ ≤ ⌊x⌋ + ⌊y⌋ := by
  sorry

end NUMINAMATH_CALUDE_floor_sum_inequality_l1753_175350


namespace NUMINAMATH_CALUDE_set_equality_l1753_175311

-- Define sets A and B
def A : Set ℝ := {x | x < 4}
def B : Set ℝ := {x | x^2 - 4*x + 3 > 0}

-- Define the set we want to prove equal to our result
def S : Set ℝ := {x | x ∈ A ∧ x ∉ A ∩ B}

-- State the theorem
theorem set_equality : S = {x : ℝ | 1 ≤ x ∧ x ≤ 3} := by sorry

end NUMINAMATH_CALUDE_set_equality_l1753_175311


namespace NUMINAMATH_CALUDE_arithmetic_sequence_24th_term_l1753_175339

/-- An arithmetic sequence is a sequence where the difference between consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem states that for an arithmetic sequence where the 3rd term is 7 and the 10th term is 27,
    the 24th term is 67. -/
theorem arithmetic_sequence_24th_term
  (a : ℕ → ℚ)
  (h_arith : ArithmeticSequence a)
  (h_3rd : a 3 = 7)
  (h_10th : a 10 = 27) :
  a 24 = 67 := by
sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_24th_term_l1753_175339


namespace NUMINAMATH_CALUDE_cross_product_solution_l1753_175377

theorem cross_product_solution :
  let v1 : ℝ × ℝ × ℝ := (128/15, -2, 7/5)
  let v2 : ℝ × ℝ × ℝ := (4, 5, 3)
  let result : ℝ × ℝ × ℝ := (-13, -20, 23)
  (v1.2.1 * v2.2.2 - v1.2.2 * v2.2.1,
   v1.2.2 * v2.1 - v1.1 * v2.2.2,
   v1.1 * v2.2.1 - v1.2.1 * v2.1) = result :=
by sorry

end NUMINAMATH_CALUDE_cross_product_solution_l1753_175377


namespace NUMINAMATH_CALUDE_trigonometric_identities_l1753_175333

theorem trigonometric_identities (θ : ℝ) (h : Real.sin (θ - π/3) = 1/3) :
  (Real.sin (θ + 2*π/3) = -1/3) ∧ (Real.cos (θ - 5*π/6) = 1/3) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l1753_175333


namespace NUMINAMATH_CALUDE_coordinate_sum_with_slope_l1753_175386

/-- Given points A and B, where A is at (0, 0) and B is on the line y = 4,
    if the slope of segment AB is 3/4, then the sum of B's coordinates is 28/3. -/
theorem coordinate_sum_with_slope (x : ℝ) :
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (x, 4)
  let slope : ℝ := (B.2 - A.2) / (B.1 - A.1)
  slope = 3/4 → x + 4 = 28/3 := by
  sorry

end NUMINAMATH_CALUDE_coordinate_sum_with_slope_l1753_175386


namespace NUMINAMATH_CALUDE_jack_position_change_l1753_175300

-- Define the constants from the problem
def flights_up : ℕ := 3
def flights_down : ℕ := 6
def steps_per_flight : ℕ := 12
def inches_per_step : ℕ := 8
def inches_per_foot : ℕ := 12

-- Define the function to calculate the net change in position
def net_position_change : ℚ :=
  (flights_down - flights_up) * steps_per_flight * inches_per_step / inches_per_foot

-- Theorem statement
theorem jack_position_change :
  net_position_change = 24 := by sorry

end NUMINAMATH_CALUDE_jack_position_change_l1753_175300


namespace NUMINAMATH_CALUDE_inverse_function_range_l1753_175383

-- Define the function
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + a

-- State the theorem
theorem inverse_function_range (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 3, Function.Injective (fun x => f a x)) ∧ 
  (|a - 1| + |a - 3| ≤ 4) →
  a ∈ Set.Icc 0 1 ∪ Set.Icc 3 4 :=
sorry

end NUMINAMATH_CALUDE_inverse_function_range_l1753_175383


namespace NUMINAMATH_CALUDE_min_value_quadratic_form_l1753_175371

theorem min_value_quadratic_form (a b c d : ℝ) (h : 5*a + 6*b - 7*c + 4*d = 1) :
  3*a^2 + 2*b^2 + 5*c^2 + d^2 ≥ 15/782 ∧
  ∃ (a₀ b₀ c₀ d₀ : ℝ), 5*a₀ + 6*b₀ - 7*c₀ + 4*d₀ = 1 ∧ 3*a₀^2 + 2*b₀^2 + 5*c₀^2 + d₀^2 = 15/782 :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_form_l1753_175371


namespace NUMINAMATH_CALUDE_mika_birthday_stickers_l1753_175328

/-- The number of stickers Mika gets for her birthday -/
def birthday_stickers : ℕ := sorry

/-- The number of stickers Mika initially had -/
def initial_stickers : ℕ := 20

/-- The number of stickers Mika bought -/
def bought_stickers : ℕ := 26

/-- The number of stickers Mika gave to her sister -/
def given_stickers : ℕ := 6

/-- The number of stickers Mika used for the greeting card -/
def used_stickers : ℕ := 58

/-- The number of stickers Mika has left -/
def remaining_stickers : ℕ := 2

theorem mika_birthday_stickers :
  initial_stickers + bought_stickers + birthday_stickers - given_stickers - used_stickers = remaining_stickers ∧
  birthday_stickers = 20 :=
sorry

end NUMINAMATH_CALUDE_mika_birthday_stickers_l1753_175328


namespace NUMINAMATH_CALUDE_uncle_bob_can_park_l1753_175355

-- Define the number of total spaces and parked cars
def total_spaces : ℕ := 18
def parked_cars : ℕ := 14

-- Define a function to calculate the number of ways to distribute n items into k groups
def stars_and_bars (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

-- Define the probability of Uncle Bob finding a parking space
def uncle_bob_parking_probability : ℚ :=
  1 - (stars_and_bars 7 5 : ℚ) / (Nat.choose total_spaces parked_cars : ℚ)

-- Theorem statement
theorem uncle_bob_can_park : 
  uncle_bob_parking_probability = 91 / 102 :=
sorry

end NUMINAMATH_CALUDE_uncle_bob_can_park_l1753_175355


namespace NUMINAMATH_CALUDE_brick_height_calculation_l1753_175322

theorem brick_height_calculation (brick_length : ℝ) (brick_width : ℝ)
  (wall_length : ℝ) (wall_height : ℝ) (wall_width : ℝ)
  (num_bricks : ℕ) :
  brick_length = 25 →
  brick_width = 11.25 →
  wall_length = 800 →
  wall_height = 600 →
  wall_width = 22.5 →
  num_bricks = 6400 →
  ∃ brick_height : ℝ,
    brick_height = 6 ∧
    wall_length * wall_height * wall_width =
    num_bricks * (brick_length * brick_width * brick_height) :=
by
  sorry

#check brick_height_calculation

end NUMINAMATH_CALUDE_brick_height_calculation_l1753_175322


namespace NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l1753_175378

/-- Arithmetic sequence sum -/
def arithmetic_sum (a : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  n * (2 * a + (n - 1) * d) / 2

/-- The problem statement -/
theorem arithmetic_sequence_first_term
  (d : ℚ)
  (h_d : d = 5)
  (h_constant : ∃ (c : ℚ), ∀ (n : ℕ+),
    arithmetic_sum a d (3 * n) / arithmetic_sum a d n = c) :
  a = 5 / 2 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l1753_175378


namespace NUMINAMATH_CALUDE_expression_evaluation_l1753_175336

theorem expression_evaluation (a b c : ℝ) 
  (ha : a = 15) (hb : b = 19) (hc : c = 13) : 
  (a^2 * (1/c - 1/b) + b^2 * (1/a - 1/c) + c^2 * (1/b - 1/a)) / 
  (a * (1/c - 1/b) + b * (1/a - 1/c) + c * (1/b - 1/a)) = a + b + c := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1753_175336


namespace NUMINAMATH_CALUDE_subset_implies_a_values_l1753_175348

theorem subset_implies_a_values (a : ℝ) : 
  let M : Set ℝ := {x | x^2 = 1}
  let N : Set ℝ := {x | a * x = 1}
  N ⊆ M → a ∈ ({-1, 0, 1} : Set ℝ) := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_values_l1753_175348


namespace NUMINAMATH_CALUDE_ab_value_l1753_175343

theorem ab_value (a b : ℝ) (h : (a + 2)^2 + |b - 4| = 0) : a^b = 16 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l1753_175343


namespace NUMINAMATH_CALUDE_pure_imaginary_square_root_l1753_175307

theorem pure_imaginary_square_root (a : ℝ) :
  (∃ (b : ℝ), (a - Complex.I) ^ 2 = Complex.I * b) → (a = 1 ∨ a = -1) := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_square_root_l1753_175307


namespace NUMINAMATH_CALUDE_unique_solution_lcm_gcd_equation_l1753_175359

theorem unique_solution_lcm_gcd_equation : 
  ∃! n : ℕ+, Nat.lcm n 120 = Nat.gcd n 120 + 300 ∧ n = 180 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_lcm_gcd_equation_l1753_175359


namespace NUMINAMATH_CALUDE_edward_savings_l1753_175363

/-- Represents the amount of money Edward had saved before mowing lawns -/
def money_saved (earnings_per_lawn : ℕ) (lawns_mowed : ℕ) (total_money : ℕ) : ℕ :=
  total_money - (earnings_per_lawn * lawns_mowed)

/-- Theorem stating that Edward's savings before mowing can be calculated -/
theorem edward_savings :
  let earnings_per_lawn : ℕ := 8
  let lawns_mowed : ℕ := 5
  let total_money : ℕ := 47
  money_saved earnings_per_lawn lawns_mowed total_money = 7 := by
  sorry

end NUMINAMATH_CALUDE_edward_savings_l1753_175363


namespace NUMINAMATH_CALUDE_division_of_powers_l1753_175364

theorem division_of_powers (n : ℕ) : 19^11 / 19^8 = 6859 := by
  sorry

end NUMINAMATH_CALUDE_division_of_powers_l1753_175364


namespace NUMINAMATH_CALUDE_football_score_proof_l1753_175340

theorem football_score_proof :
  let hawks_touchdowns : ℕ := 4
  let hawks_successful_extra_points : ℕ := 2
  let hawks_failed_extra_points : ℕ := 2
  let hawks_field_goals : ℕ := 2
  let eagles_touchdowns : ℕ := 3
  let eagles_successful_extra_points : ℕ := 3
  let eagles_field_goals : ℕ := 3
  let touchdown_points : ℕ := 6
  let extra_point_points : ℕ := 1
  let field_goal_points : ℕ := 3
  
  let hawks_score : ℕ := hawks_touchdowns * touchdown_points + 
                         hawks_successful_extra_points * extra_point_points + 
                         hawks_field_goals * field_goal_points
  
  let eagles_score : ℕ := eagles_touchdowns * touchdown_points + 
                          eagles_successful_extra_points * extra_point_points + 
                          eagles_field_goals * field_goal_points
  
  let total_score : ℕ := hawks_score + eagles_score
  
  total_score = 62 := by
    sorry

end NUMINAMATH_CALUDE_football_score_proof_l1753_175340


namespace NUMINAMATH_CALUDE_erica_age_is_17_l1753_175367

def casper_age : ℕ := 18

def ivy_age (casper_age : ℕ) : ℕ := casper_age + 4

def erica_age (ivy_age : ℕ) : ℕ := ivy_age - 5

theorem erica_age_is_17 :
  erica_age (ivy_age casper_age) = 17 := by
  sorry

end NUMINAMATH_CALUDE_erica_age_is_17_l1753_175367


namespace NUMINAMATH_CALUDE_seven_digit_palindrome_count_l1753_175358

/-- A seven-digit palindrome is a number of the form abcdcba where a ≠ 0 -/
def SevenDigitPalindrome : Type := Nat

/-- The count of seven-digit palindromes -/
def countSevenDigitPalindromes : Nat := 9000

theorem seven_digit_palindrome_count :
  (Finset.filter (λ n : Nat => n ≥ 1000000 ∧ n ≤ 9999999 ∧ 
    (String.mk (List.reverse (String.toList (toString n)))) = toString n)
    (Finset.range 10000000)).card = countSevenDigitPalindromes := by
  sorry

end NUMINAMATH_CALUDE_seven_digit_palindrome_count_l1753_175358


namespace NUMINAMATH_CALUDE_exists_non_increasing_log_l1753_175324

-- Define the logarithmic function
noncomputable def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- State the theorem
theorem exists_non_increasing_log :
  ∃ (a : ℝ), a > 0 ∧ a ≠ 1 ∧ ¬(∀ (x y : ℝ), x < y → log a x < log a y) :=
sorry

end NUMINAMATH_CALUDE_exists_non_increasing_log_l1753_175324


namespace NUMINAMATH_CALUDE_minutes_conversion_l1753_175346

/-- The number of seconds in one minute -/
def seconds_per_minute : ℕ := 60

/-- The number of minutes in one hour -/
def minutes_per_hour : ℕ := 60

/-- Converts minutes to seconds -/
def minutes_to_seconds (minutes : ℚ) : ℚ :=
  minutes * seconds_per_minute

/-- Converts minutes to hours -/
def minutes_to_hours (minutes : ℚ) : ℚ :=
  minutes / minutes_per_hour

theorem minutes_conversion (minutes : ℚ) :
  minutes = 25/2 →
  minutes_to_seconds minutes = 750 ∧ minutes_to_hours minutes = 5/24 := by
  sorry

end NUMINAMATH_CALUDE_minutes_conversion_l1753_175346


namespace NUMINAMATH_CALUDE_li_point_parabola_range_l1753_175391

/-- A point (x, y) is a "Li point" if x and y have opposite signs -/
def is_li_point (x y : ℝ) : Prop := x * y < 0

/-- Parabola equation -/
def parabola (a c x : ℝ) : ℝ := a * x^2 - 7 * x + c

theorem li_point_parabola_range (a c : ℝ) :
  a > 1 →
  (∃! x : ℝ, is_li_point x (parabola a c x)) →
  0 < c ∧ c < 9 :=
by sorry

end NUMINAMATH_CALUDE_li_point_parabola_range_l1753_175391


namespace NUMINAMATH_CALUDE_quadrilateral_bd_value_l1753_175323

/-- Represents a quadrilateral ABCD with given side lengths and diagonal BD --/
structure Quadrilateral where
  AB : ℝ
  BC : ℝ
  CD : ℝ
  DA : ℝ
  BD : ℤ

/-- The quadrilateral satisfies the triangle inequality --/
def satisfies_triangle_inequality (q : Quadrilateral) : Prop :=
  q.AB + q.BD > q.DA ∧
  q.BC + q.CD > q.BD ∧
  q.DA + q.BD > q.AB ∧
  q.BD + q.CD > q.BC

/-- The theorem to be proved --/
theorem quadrilateral_bd_value (q : Quadrilateral) 
  (h1 : q.AB = 6)
  (h2 : q.BC = 19)
  (h3 : q.CD = 6)
  (h4 : q.DA = 10)
  (h5 : satisfies_triangle_inequality q) :
  q.BD = 15 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_bd_value_l1753_175323


namespace NUMINAMATH_CALUDE_josanna_minimum_score_l1753_175385

def josanna_scores : List ℕ := [82, 76, 91, 65, 87, 78]

def current_average : ℚ := (josanna_scores.sum : ℚ) / josanna_scores.length

def target_average : ℚ := current_average + 5

def minimum_score : ℕ := 116

theorem josanna_minimum_score :
  ∀ (new_score : ℕ),
    ((josanna_scores.sum + new_score : ℚ) / (josanna_scores.length + 1) ≥ target_average) →
    (new_score ≥ minimum_score) := by sorry

end NUMINAMATH_CALUDE_josanna_minimum_score_l1753_175385


namespace NUMINAMATH_CALUDE_different_arrangements_count_l1753_175352

def num_red_balls : ℕ := 6
def num_green_balls : ℕ := 3
def num_selected_balls : ℕ := 4

def num_arrangements (r g s : ℕ) : ℕ :=
  (Nat.choose s s) +
  (Nat.choose s 1) * 2 +
  (Nat.choose s 2)

theorem different_arrangements_count :
  num_arrangements num_red_balls num_green_balls num_selected_balls = 15 := by
  sorry

end NUMINAMATH_CALUDE_different_arrangements_count_l1753_175352


namespace NUMINAMATH_CALUDE_sin_seventeen_pi_sixths_l1753_175389

theorem sin_seventeen_pi_sixths : Real.sin (17 * π / 6) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_seventeen_pi_sixths_l1753_175389


namespace NUMINAMATH_CALUDE_inequality_range_l1753_175335

theorem inequality_range : 
  ∀ x : ℝ, (∀ a b : ℝ, a > 0 ∧ b > 0 → x^2 + x < a/b + b/a) ↔ -2 < x ∧ x < 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_range_l1753_175335


namespace NUMINAMATH_CALUDE_mango_rate_is_59_l1753_175309

/-- Calculates the rate per kg for mangoes given the total amount paid, grape price, grape weight, and mango weight. -/
def mango_rate (total_paid : ℕ) (grape_price : ℕ) (grape_weight : ℕ) (mango_weight : ℕ) : ℕ :=
  (total_paid - grape_price * grape_weight) / mango_weight

/-- Theorem stating that under the given conditions, the mango rate is 59 -/
theorem mango_rate_is_59 :
  mango_rate 975 74 6 9 = 59 := by
  sorry

#eval mango_rate 975 74 6 9

end NUMINAMATH_CALUDE_mango_rate_is_59_l1753_175309


namespace NUMINAMATH_CALUDE_f_bijection_l1753_175382

def f (n : ℤ) : ℤ := 2 * n

theorem f_bijection : Function.Bijective f := by sorry

end NUMINAMATH_CALUDE_f_bijection_l1753_175382


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l1753_175303

theorem sqrt_meaningful_range (x : ℝ) : (∃ y : ℝ, y ^ 2 = x - 1) → x ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l1753_175303


namespace NUMINAMATH_CALUDE_travis_cereal_expenditure_l1753_175366

theorem travis_cereal_expenditure 
  (boxes_per_week : ℕ) 
  (cost_per_box : ℚ) 
  (weeks_per_year : ℕ) 
  (h1 : boxes_per_week = 2) 
  (h2 : cost_per_box = 3) 
  (h3 : weeks_per_year = 52) : 
  (boxes_per_week * cost_per_box * weeks_per_year : ℚ) = 312 :=
by sorry

end NUMINAMATH_CALUDE_travis_cereal_expenditure_l1753_175366


namespace NUMINAMATH_CALUDE_complex_expression_simplification_l1753_175321

theorem complex_expression_simplification :
  (7 - 3*Complex.I) - 3*(2 + 4*Complex.I) + 4*(1 - Complex.I) = 5 - 19*Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_expression_simplification_l1753_175321


namespace NUMINAMATH_CALUDE_exists_subset_with_constant_gcd_l1753_175337

/-- A function that checks if a natural number is the product of at most 2000 distinct primes -/
def is_product_of_limited_primes (n : ℕ) : Prop :=
  ∃ (primes : Finset ℕ), (∀ p ∈ primes, Nat.Prime p) ∧ primes.card ≤ 2000 ∧ n = primes.prod id

/-- The main theorem -/
theorem exists_subset_with_constant_gcd 
  (A : Set ℕ) 
  (h_infinite : Set.Infinite A) 
  (h_limited_primes : ∀ a ∈ A, is_product_of_limited_primes a) :
  ∃ (B : Set ℕ) (k : ℕ), Set.Infinite B ∧ B ⊆ A ∧ 
    ∀ (b1 b2 : ℕ), b1 ∈ B → b2 ∈ B → b1 ≠ b2 → Nat.gcd b1 b2 = k :=
sorry

end NUMINAMATH_CALUDE_exists_subset_with_constant_gcd_l1753_175337


namespace NUMINAMATH_CALUDE_triangle_division_l1753_175370

/-- The number of triangles formed by n points inside a triangle -/
def numTriangles (n : ℕ) : ℕ := 2 * n + 1

/-- Theorem stating that 1997 points inside a triangle creates 3995 smaller triangles -/
theorem triangle_division (n : ℕ) (h : n = 1997) : numTriangles n = 3995 := by
  sorry

end NUMINAMATH_CALUDE_triangle_division_l1753_175370


namespace NUMINAMATH_CALUDE_binomial_expansion_largest_coeff_l1753_175384

theorem binomial_expansion_largest_coeff (n : ℕ+) :
  (∀ k : ℕ, k ≠ 5 → Nat.choose n 5 ≥ Nat.choose n k) → n = 10 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_largest_coeff_l1753_175384


namespace NUMINAMATH_CALUDE_cube_root_sum_zero_implies_opposite_l1753_175332

theorem cube_root_sum_zero_implies_opposite (x y : ℝ) : 
  (x^(1/3 : ℝ) + y^(1/3 : ℝ) = 0) → (x = -y) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_sum_zero_implies_opposite_l1753_175332


namespace NUMINAMATH_CALUDE_total_weekly_calories_l1753_175368

/-- Represents the number of each type of burger consumed on a given day -/
structure DailyConsumption where
  burgerA : ℕ
  burgerB : ℕ
  burgerC : ℕ

/-- Calculates the total calories for a given daily consumption -/
def dailyCalories (d : DailyConsumption) : ℕ :=
  d.burgerA * 350 + d.burgerB * 450 + d.burgerC * 550

/-- Represents Dimitri's burger consumption for the week -/
def weeklyConsumption : List DailyConsumption :=
  [
    ⟨2, 1, 0⟩,  -- Day 1
    ⟨1, 2, 1⟩,  -- Day 2
    ⟨1, 1, 2⟩,  -- Day 3
    ⟨0, 3, 0⟩,  -- Day 4
    ⟨1, 1, 1⟩,  -- Day 5
    ⟨2, 0, 3⟩,  -- Day 6
    ⟨0, 1, 2⟩   -- Day 7
  ]

/-- Theorem: The total calories consumed by Dimitri in a week is 11,450 -/
theorem total_weekly_calories : 
  (weeklyConsumption.map dailyCalories).sum = 11450 := by
  sorry


end NUMINAMATH_CALUDE_total_weekly_calories_l1753_175368


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l1753_175369

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_product (a : ℕ → ℝ) :
  geometric_sequence a →
  a 5 * a 14 = 5 →
  a 8 * a 9 * a 10 * a 11 = 25 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l1753_175369


namespace NUMINAMATH_CALUDE_calculate_expression_l1753_175351

theorem calculate_expression : 
  Real.sqrt 12 - 3 - ((1/3) * Real.sqrt 27 - Real.sqrt 9) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l1753_175351


namespace NUMINAMATH_CALUDE_sum_six_consecutive_integers_l1753_175390

theorem sum_six_consecutive_integers (n : ℤ) : 
  n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) = 6 * n + 15 := by
  sorry

end NUMINAMATH_CALUDE_sum_six_consecutive_integers_l1753_175390


namespace NUMINAMATH_CALUDE_solution_set_inequality_l1753_175330

theorem solution_set_inequality (x : ℝ) (h : x ≠ 0) :
  (2*x - 1) / x < 1 ↔ 0 < x ∧ x < 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l1753_175330


namespace NUMINAMATH_CALUDE_total_animal_eyes_pond_animal_eyes_l1753_175305

theorem total_animal_eyes (num_frogs num_crocodiles : ℕ) 
  (eyes_per_frog eyes_per_crocodile : ℕ) : ℕ :=
  num_frogs * eyes_per_frog + num_crocodiles * eyes_per_crocodile

theorem pond_animal_eyes : total_animal_eyes 20 6 2 2 = 52 := by
  sorry

end NUMINAMATH_CALUDE_total_animal_eyes_pond_animal_eyes_l1753_175305


namespace NUMINAMATH_CALUDE_cost_of_potting_soil_l1753_175393

def cost_of_seeds : ℝ := 2.00
def number_of_plants : ℕ := 20
def price_per_plant : ℝ := 5.00
def net_profit : ℝ := 90.00

theorem cost_of_potting_soil :
  ∃ (cost : ℝ), cost = (number_of_plants : ℝ) * price_per_plant - cost_of_seeds - net_profit :=
by sorry

end NUMINAMATH_CALUDE_cost_of_potting_soil_l1753_175393


namespace NUMINAMATH_CALUDE_mila_coin_collection_value_l1753_175316

/-- The total value of Mila's coin collection -/
def total_value (gold_coins silver_coins : ℕ) (gold_value silver_value : ℚ) : ℚ :=
  gold_coins * gold_value + silver_coins * silver_value

/-- Theorem stating the total value of Mila's coin collection -/
theorem mila_coin_collection_value :
  let gold_coins : ℕ := 20
  let silver_coins : ℕ := 15
  let gold_value : ℚ := 10 / 4
  let silver_value : ℚ := 15 / 5
  total_value gold_coins silver_coins gold_value silver_value = 95
  := by sorry

end NUMINAMATH_CALUDE_mila_coin_collection_value_l1753_175316


namespace NUMINAMATH_CALUDE_slower_speed_calculation_l1753_175380

theorem slower_speed_calculation (actual_distance : ℝ) (faster_speed : ℝ) (additional_distance : ℝ) :
  actual_distance = 40 →
  faster_speed = 12 →
  additional_distance = 20 →
  ∃ slower_speed : ℝ, 
    slower_speed > 0 ∧
    actual_distance / slower_speed = (actual_distance + additional_distance) / faster_speed ∧
    slower_speed = 8 := by
  sorry

end NUMINAMATH_CALUDE_slower_speed_calculation_l1753_175380


namespace NUMINAMATH_CALUDE_parabola_equation_l1753_175327

/-- A parabola is defined by its axis equation and standard form equation. -/
structure Parabola where
  /-- The x-coordinate of the axis of the parabola -/
  axis : ℝ
  /-- The coefficient in the standard form equation y² = 2px -/
  p : ℝ
  /-- Condition that p is positive -/
  p_pos : p > 0

/-- The standard form equation of a parabola is y² = 2px -/
def standard_form (para : Parabola) : Prop :=
  ∀ x y : ℝ, y^2 = 2 * para.p * x

/-- The axis equation of a parabola is x = -p/2 -/
def axis_equation (para : Parabola) : Prop :=
  para.axis = -para.p / 2

/-- Theorem: Given a parabola with axis equation x = -2, its standard form equation is y² = 8x -/
theorem parabola_equation (para : Parabola) 
  (h : axis_equation para) 
  (h_axis : para.axis = -2) : 
  standard_form para ∧ para.p = 4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_equation_l1753_175327


namespace NUMINAMATH_CALUDE_gcd_lcm_product_24_60_l1753_175353

theorem gcd_lcm_product_24_60 : Nat.gcd 24 60 * Nat.lcm 24 60 = 1440 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_24_60_l1753_175353


namespace NUMINAMATH_CALUDE_drums_hit_calculation_l1753_175310

/-- Represents the drumming contest scenario --/
structure DrummingContest where
  entryFee : ℝ
  costPerDrum : ℝ
  earningsStartDrum : ℕ
  earningsPerDrum : ℝ
  bonusRoundDrum : ℕ
  totalLoss : ℝ

/-- Calculates the number of drums hit in the contest --/
def drumsHit (contest : DrummingContest) : ℕ :=
  sorry

/-- Theorem stating the number of drums hit in the given scenario --/
theorem drums_hit_calculation (contest : DrummingContest) 
  (h1 : contest.entryFee = 10)
  (h2 : contest.costPerDrum = 0.02)
  (h3 : contest.earningsStartDrum = 200)
  (h4 : contest.earningsPerDrum = 0.025)
  (h5 : contest.bonusRoundDrum = 250)
  (h6 : contest.totalLoss = 7.5) :
  drumsHit contest = 4500 :=
sorry

end NUMINAMATH_CALUDE_drums_hit_calculation_l1753_175310


namespace NUMINAMATH_CALUDE_sin_cos_15_ratio_eq_neg_sqrt3_div_3_l1753_175306

theorem sin_cos_15_ratio_eq_neg_sqrt3_div_3 :
  (Real.sin (15 * π / 180) - Real.cos (15 * π / 180)) /
  (Real.sin (15 * π / 180) + Real.cos (15 * π / 180)) = -Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_15_ratio_eq_neg_sqrt3_div_3_l1753_175306


namespace NUMINAMATH_CALUDE_henrys_money_l1753_175308

/-- Henry's money calculation -/
theorem henrys_money (initial_amount : ℕ) (birthday_gift : ℕ) (spent_amount : ℕ) : 
  initial_amount = 11 → birthday_gift = 18 → spent_amount = 10 → 
  initial_amount + birthday_gift - spent_amount = 19 := by
  sorry

#check henrys_money

end NUMINAMATH_CALUDE_henrys_money_l1753_175308
