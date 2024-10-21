import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l394_39454

theorem simplify_expression (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (x + 2*y)^((-2) : ℝ) * (x^((-1) : ℝ) + 2*y^((-1) : ℝ)) = 
  (y + 2*x) * x^((-1) : ℝ) * y^((-1) : ℝ) * (x + 2*y)^((-2) : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l394_39454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_money_calculation_l394_39443

noncomputable def remaining_money (initial_amount : ℝ) : ℝ :=
  let after_tablet := initial_amount * (1 - 0.40)
  let after_phone_game := after_tablet * (1 - 1/3)
  let after_in_game := after_phone_game * (1 - 0.15)
  let after_tablet_case := after_in_game * (1 - 0.20)
  let after_phone_cover := after_tablet_case * (1 - 0.05)
  let after_power_bank := after_phone_cover * (1 - 0.12)
  after_power_bank

theorem remaining_money_calculation (initial_amount : ℝ) :
  initial_amount = 200 →
  remaining_money initial_amount = 45.4784 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_money_calculation_l394_39443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_power_eight_l394_39431

noncomputable def A : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![Real.sqrt 2 / 2, -Real.sqrt 2 / 2],
    ![Real.sqrt 2 / 2, Real.sqrt 2 / 2]]

theorem matrix_power_eight :
  A ^ 8 = ![![16, 0],
            ![0, 16]] :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_power_eight_l394_39431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_plane_example_l394_39488

noncomputable def distanceToPlane (M₀ M₁ M₂ M₃ : ℝ × ℝ × ℝ) : ℝ :=
  let (x₀, y₀, z₀) := M₀
  let (x₁, y₁, z₁) := M₁
  let (x₂, y₂, z₂) := M₂
  let (x₃, y₃, z₃) := M₃
  let A := (y₂ - y₁) * (z₃ - z₁) - (z₂ - z₁) * (y₃ - y₁)
  let B := (z₂ - z₁) * (x₃ - x₁) - (x₂ - x₁) * (z₃ - z₁)
  let C := (x₂ - x₁) * (y₃ - y₁) - (y₂ - y₁) * (x₃ - x₁)
  let D := -A * x₁ - B * y₁ - C * z₁
  abs (A * x₀ + B * y₀ + C * z₀ + D) / Real.sqrt (A * A + B * B + C * C)

theorem distance_to_plane_example :
  distanceToPlane (1, -1, 2) (1, 5, -7) (-3, 6, 3) (-2, 7, 3) = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_plane_example_l394_39488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_f_nonnegative_implies_b_range_l394_39452

open Real

/-- The function f(x) as defined in the problem -/
noncomputable def f (a b x : ℝ) : ℝ := log x + (1/2) * a * x^2 + b * x

/-- Part I: Monotonicity of f when a = -2 and b = 1 -/
theorem f_monotonicity :
  ∀ x₁ x₂, 0 < x₁ ∧ x₁ < 1 ∧ 1 < x₂ →
  deriv (f (-2) 1) x₁ > 0 ∧ deriv (f (-2) 1) x₂ < 0 :=
by
  sorry

/-- Part II: Range of b when a ∈ [1, +∞) and f(x) ≥ 0 for x ∈ [1, +∞) -/
theorem f_nonnegative_implies_b_range :
  ∀ a, a ≥ 1 →
  (∀ x, x ≥ 1 → f a b x ≥ 0) →
  b ≥ (-1/2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_f_nonnegative_implies_b_range_l394_39452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_relation_l394_39468

theorem proposition_relation (p q r : Prop) :
  (¬p ↔ q) → ((q → r) ↔ (¬r → ¬q)) → (p ↔ (r → p)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_relation_l394_39468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_l394_39472

-- Define the quadrilateral vertices
def A : ℝ × ℝ := (1, 3)
def B : ℝ × ℝ := (1, 1)
def C : ℝ × ℝ := (4, 1)
def D : ℝ × ℝ := (20, 22)

-- Define the quadrilateral
def quadrilateral : List (ℝ × ℝ) := [A, B, C, D]

-- Define a function to calculate the area of a quadrilateral
noncomputable def area (quad : List (ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem quadrilateral_area : 
  area quadrilateral = 50.5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_l394_39472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_strict_decreasing_interval_l394_39453

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.sqrt 3 * Real.cos x

-- Define the domain
def domain : Set ℝ := Set.Icc 0 (2 * Real.pi)

-- Define the decreasing interval
def decreasing_interval : Set ℝ := Set.Icc (Real.pi / 6) (7 * Real.pi / 6)

-- Theorem statement
theorem strict_decreasing_interval :
  ∀ x y, x ∈ domain → y ∈ domain → x < y → x ∈ decreasing_interval → y ∈ decreasing_interval → f y < f x :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_strict_decreasing_interval_l394_39453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tin_amount_in_new_alloy_l394_39491

/-- The amount of tin in a mixture of two alloys -/
noncomputable def tin_in_mixture (weight_a weight_b : ℝ) (ratio_a ratio_b : ℝ × ℝ) : ℝ :=
  let tin_a := weight_a * (ratio_a.2 / (ratio_a.1 + ratio_a.2))
  let tin_b := weight_b * (ratio_b.1 / (ratio_b.1 + ratio_b.2))
  tin_a + tin_b

/-- The theorem stating the amount of tin in the mixture of alloys A and B -/
theorem tin_amount_in_new_alloy :
  let weight_a : ℝ := 130
  let weight_b : ℝ := 160
  let ratio_a : ℝ × ℝ := (2, 3)  -- lead:tin ratio for alloy A
  let ratio_b : ℝ × ℝ := (3, 4)  -- tin:copper ratio for alloy B
  ‖tin_in_mixture weight_a weight_b ratio_a ratio_b - 146.57‖ < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tin_amount_in_new_alloy_l394_39491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_value_l394_39471

-- Define the function f
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.cos (ω * x)

-- State the theorem
theorem omega_value (ω : ℝ) (h1 : ω > 0) :
  (∀ x : ℝ, f ω x = f ω (3 * Real.pi / 2 - x)) →  -- Symmetry condition
  (∀ x y : ℝ, 0 < x ∧ x < y ∧ y < 2 * Real.pi / 3 → 
    (f ω x < f ω y ∨ ∀ z ∈ Set.Ioo x y, f ω x = f ω z)) →  -- Monotonicity condition
  ω = 2 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_value_l394_39471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_inverse_distance_sum_l394_39441

-- Define the curves C1 and C2
noncomputable def C1 (t α : ℝ) : ℝ × ℝ := (t * Real.cos α, 1 + t * Real.sin α)
noncomputable def C2 (θ : ℝ) : ℝ × ℝ := (2 * Real.cos θ * Real.cos θ, 2 * Real.cos θ * Real.sin θ)

-- Define point A
def A : ℝ × ℝ := (0, 1)

-- Define the theorem
theorem max_inverse_distance_sum (α : ℝ) (hα : 0 < α ∧ α < Real.pi) :
  ∃ (P Q : ℝ × ℝ) (t₁ t₂ : ℝ),
    P ≠ Q ∧
    C1 t₁ α = P ∧
    C1 t₂ α = Q ∧
    (∃ θ₁ θ₂, C2 θ₁ = P ∧ C2 θ₂ = Q) ∧
    (∀ R S : ℝ × ℝ,
      (∃ t₃ t₄ θ₃ θ₄, C1 t₃ α = R ∧ C1 t₄ α = S ∧ C2 θ₃ = R ∧ C2 θ₄ = S) →
      1 / dist A P + 1 / dist A Q ≥ 1 / dist A R + 1 / dist A S) ∧
    1 / dist A P + 1 / dist A Q = 2 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_inverse_distance_sum_l394_39441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_reverse_sum_l394_39470

/-- Two-digit integer -/
def TwoDigitInt (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

/-- Reverse digits of a two-digit number -/
def reverseDigits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

/-- Main theorem -/
theorem two_digit_reverse_sum (x y m : ℕ) (h : ℚ) : 
  TwoDigitInt x ∧ 
  y = reverseDigits x ∧ 
  x^2 - y^2 = 4 * m^2 ∧
  h = 137.5 →
  x + y + m = h := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_reverse_sum_l394_39470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_amanda_earnings_rounded_to_21_l394_39435

/-- Represents Amanda's work schedule for a week -/
structure WorkSchedule where
  tuesday_hours : ℚ
  wednesday_minutes : ℚ
  thursday_start : ℚ  -- in minutes after midnight
  thursday_end : ℚ    -- in minutes after midnight
  saturday_minutes : ℚ

/-- Calculates the total earnings based on the work schedule and hourly rate -/
noncomputable def calculate_earnings (schedule : WorkSchedule) (hourly_rate : ℚ) : ℚ :=
  let total_minutes : ℚ :=
    schedule.tuesday_hours * 60 +
    schedule.wednesday_minutes +
    (schedule.thursday_end - schedule.thursday_start) +
    schedule.saturday_minutes
  let total_hours : ℚ := total_minutes / 60
  total_hours * hourly_rate

/-- Amanda's actual work schedule for the week -/
def amanda_schedule : WorkSchedule :=
  { tuesday_hours := 3/2
  , wednesday_minutes := 45
  , thursday_start := 9 * 60 + 15  -- 9:15 AM
  , thursday_end := 11 * 60 + 30   -- 11:30 AM
  , saturday_minutes := 40 }

/-- Amanda's hourly rate -/
def amanda_hourly_rate : ℚ := 4

/-- Theorem stating that Amanda's earnings rounded to the nearest dollar is $21 -/
theorem amanda_earnings_rounded_to_21 :
  Int.floor (calculate_earnings amanda_schedule amanda_hourly_rate + 1/2) = 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_amanda_earnings_rounded_to_21_l394_39435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_expression_simplification_l394_39483

theorem trig_expression_simplification (α : ℝ) : 
  (Real.cos (2 * π + α) * Real.tan (π + α)) / Real.cos (π / 2 - α) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_expression_simplification_l394_39483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_texts_sent_monday_l394_39405

/-- The number of texts Sydney sent to Allison on Monday -/
def texts_to_allison_monday : ℕ := sorry

/-- The number of texts Sydney sent to Brittney on Monday -/
def texts_to_brittney_monday : ℕ := sorry

/-- The number of texts Sydney sent to each person on Tuesday -/
def texts_per_person_tuesday : ℕ := 15

/-- The total number of texts Sydney sent over both days -/
def total_texts : ℕ := 40

theorem texts_sent_monday :
  texts_to_allison_monday + texts_to_brittney_monday = 10 :=
by
  sorry

#check texts_sent_monday

end NUMINAMATH_CALUDE_ERRORFEEDBACK_texts_sent_monday_l394_39405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_interval_l394_39480

open Real

-- Define the function
noncomputable def f (x : ℝ) : ℝ := sin (2 * x - π / 6)

-- State the theorem
theorem f_monotone_increasing_interval :
  ∀ k : ℤ, StrictMonoOn f { x | -π/6 + (k : ℝ) * π ≤ x ∧ x ≤ π/3 + (k : ℝ) * π } :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_interval_l394_39480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_trees_same_needles_l394_39411

theorem two_trees_same_needles (forest : Finset ℕ) (needle_count : ℕ → ℕ) :
  (Finset.card forest = 1000000) →
  (∀ tree, tree ∈ forest → needle_count tree ≤ 600000) →
  ∃ tree1 tree2, tree1 ∈ forest ∧ tree2 ∈ forest ∧ tree1 ≠ tree2 ∧ needle_count tree1 = needle_count tree2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_trees_same_needles_l394_39411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_and_curve_properties_l394_39439

noncomputable def line_l (t : ℝ) : ℝ × ℝ := (3 - t, 1 + t)

noncomputable def curve_C (θ : ℝ) : ℝ := 2 * Real.sqrt 2 * Real.cos (θ - Real.pi / 4)

theorem line_and_curve_properties :
  -- 1. General equation of line l
  (∀ x y : ℝ, (∃ t : ℝ, line_l t = (x, y)) ↔ x + y = 4) ∧
  -- 2. Cartesian equation of curve C
  (∀ x y : ℝ, (∃ θ : ℝ, x^2 + y^2 = (curve_C θ)^2 ∧ 
                        x = (curve_C θ) * Real.cos θ ∧ 
                        y = (curve_C θ) * Real.sin θ) 
              ↔ (x - 1)^2 + (y - 1)^2 = 2) ∧
  -- 3. Maximum distance from a point on C to line l
  (∃ d : ℝ, d = 2 * Real.sqrt 2 ∧
    (∀ θ : ℝ, let (x, y) := (1 + Real.sqrt 2 * Real.cos θ, 1 + Real.sqrt 2 * Real.sin θ)
              |x + y - 4| / Real.sqrt 2 ≤ d) ∧
    (∃ θ₀ : ℝ, let (x₀, y₀) := (1 + Real.sqrt 2 * Real.cos θ₀, 1 + Real.sqrt 2 * Real.sin θ₀)
               |x₀ + y₀ - 4| / Real.sqrt 2 = d)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_and_curve_properties_l394_39439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l394_39455

noncomputable def x : ℝ := 3 + (1 : ℝ) / Real.rpow 4 (1/3)
noncomputable def y : ℝ := -1 - (1 : ℝ) / Real.rpow 4 (1/3)

theorem unique_solution :
  (x + y = 2) ∧
  (x^4 - y^4 = 5*x - 3*y) ∧
  (∀ a b : ℝ, (a + b = 2 ∧ a^4 - b^4 = 5*a - 3*b) → (a = x ∧ b = y)) := by
  sorry

#check unique_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l394_39455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base12_multiplication_addition_l394_39462

/-- Represents a number in base 12 --/
def Base12 : Type := List (Fin 12)

/-- Convert a base 12 number to a natural number --/
def base12ToNat (x : Base12) : ℕ :=
  x.foldr (λ d acc => acc * 12 + d.val) 0

/-- Convert a natural number to base 12 --/
def natToBase12 (n : ℕ) : Base12 :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) : Base12 :=
    if m = 0 then [] else (aux (m / 12)).append [Fin.ofNat (m % 12)]
  aux n

/-- Multiply a base 12 number by 2 --/
def multiplyBy2 (x : Base12) : Base12 :=
  natToBase12 (2 * (base12ToNat x))

/-- Add two base 12 numbers --/
def addBase12 (x y : Base12) : Base12 :=
  natToBase12 ((base12ToNat x) + (base12ToNat y))

theorem base12_multiplication_addition :
  let x : Base12 := [1, 5, 9]  -- 159 in base 12
  let y : Base12 := [7, 0, 4]  -- 704 in base 12
  let result : Base12 := [11, 5, 10]  -- B5A in base 12
  addBase12 (multiplyBy2 x) y = result := by
  sorry

#eval base12ToNat [1, 5, 9]
#eval natToBase12 (base12ToNat [1, 5, 9])
#eval multiplyBy2 [1, 5, 9]
#eval addBase12 (multiplyBy2 [1, 5, 9]) [7, 0, 4]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base12_multiplication_addition_l394_39462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_deal_correct_l394_39474

/-- The number of games in the original deal -/
def original_deal : ℕ := 3

/-- The total price of the original deal in cents -/
def total_price : ℕ := 3426

/-- The price for 2 games in cents -/
def price_for_two : ℕ := 2284

/-- The price per game in cents -/
def price_per_game : ℕ := price_for_two / 2

theorem original_deal_correct : 
  original_deal * price_per_game = total_price ∧ 
  original_deal > 0 :=
by
  sorry

#check original_deal_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_deal_correct_l394_39474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_interval_is_three_minutes_l394_39499

/-- Represents the travel time between stations -/
noncomputable def travel_time (route : Bool) : ℝ :=
  if route then 17 else 11

/-- Probability of boarding a clockwise train -/
noncomputable def p : ℝ := 7/12

/-- Expected travel time to work -/
noncomputable def E_to : ℝ := 17 - 6 * p

/-- Expected travel time from work -/
noncomputable def E_back : ℝ := 11 + 6 * p

/-- Interval between trains in one direction -/
noncomputable def T : ℝ := 3

/-- Theorem stating the interval between trains is 3 minutes -/
theorem train_interval_is_three_minutes :
  (E_back - E_to = 1) ∧
  (T * (1 - p) = 5/4) →
  T = 3 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_interval_is_three_minutes_l394_39499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_number_for_divisibility_l394_39417

theorem least_number_for_divisibility : ∃! x : ℕ, 
  (∀ d ∈ ({5, 6, 4, 3} : Set ℕ), (5432 + x) % d = 0) ∧ 
  (∀ y : ℕ, y < x → ∃ d ∈ ({5, 6, 4, 3} : Set ℕ), (5432 + y) % d ≠ 0) ∧
  x = 28 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_number_for_divisibility_l394_39417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_l394_39451

theorem geometric_series_sum :
  ∃ (S : ℝ), S = (∑' n, (1/2 : ℝ)^(n+2)) ∧ S = (1/2 : ℝ) := by
  -- We use ∑' for the infinite sum, and represent the series as (1/2)^(n+2) where n starts from 0
  -- This gives us 1/4, 1/8, 1/16, ... as required
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_l394_39451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_player_growth_exceeds_30000_l394_39478

/-- The growth rate constant for the player count -/
noncomputable def k : ℝ := Real.log 10 / 5

/-- The number of players at time t -/
noncomputable def R (t : ℝ) : ℝ := 100 * Real.exp (k * t)

/-- The minimum number of days required to exceed 30,000 players -/
def min_days : ℕ := 13

theorem player_growth_exceeds_30000 :
  (∀ t : ℝ, t < min_days → R t ≤ 30000) ∧
  (∀ ε > 0, R (min_days + ε) > 30000) := by
  sorry

#eval min_days

end NUMINAMATH_CALUDE_ERRORFEEDBACK_player_growth_exceeds_30000_l394_39478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_l394_39420

-- Define the line equation
def line_equation (x y : ℝ) : Prop := Real.sqrt 3 * x - y - 3 = 0

-- Define the angle of inclination
def angle_of_inclination (α : ℝ) : Prop := α = 60 * Real.pi / 180

-- Theorem statement
theorem line_inclination :
  ∃ (α : ℝ), (∀ x y : ℝ, line_equation x y → angle_of_inclination α) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_l394_39420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_price_calculation_l394_39449

noncomputable def selling_price : ℚ := 15000

noncomputable def discount_rate : ℚ := 1/10

noncomputable def profit_rate : ℚ := 2/25

noncomputable def discounted_price : ℚ := selling_price * (1 - discount_rate)

noncomputable def cost_price : ℚ := discounted_price / (1 + profit_rate)

theorem cost_price_calculation :
  cost_price = 12500 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_price_calculation_l394_39449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangency_condition_l394_39427

/-- Definition of circle M -/
def circle_M (a : ℝ) : ℝ × ℝ → Prop :=
  λ p => (p.1 - a)^2 + (p.2 + a - 3)^2 = 1

/-- Definition of point N on circle M -/
def point_on_circle_M (a : ℝ) (N : ℝ × ℝ) : Prop :=
  circle_M a N

/-- Definition of circle with center N and radius ON -/
def circle_N (O N : ℝ × ℝ) : ℝ × ℝ → Prop :=
  λ p => (p.1 - N.1)^2 + (p.2 - N.2)^2 = (N.1 - O.1)^2 + (N.2 - O.2)^2

/-- Definition of at most one common point between two circles -/
def at_most_one_common_point (C₁ C₂ : (ℝ × ℝ) → Prop) : Prop :=
  ∀ P Q : ℝ × ℝ, C₁ P → C₁ Q → C₂ P → C₂ Q → P = Q

/-- Main theorem -/
theorem circle_tangency_condition (a : ℝ) (O : ℝ × ℝ) :
  (a > 0) →
  (∀ N : ℝ × ℝ, point_on_circle_M a N →
    at_most_one_common_point (circle_M a) (circle_N O N)) →
  a ≥ 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangency_condition_l394_39427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_x_minus_3y_values_exactly_two_values_l394_39424

theorem cos_x_minus_3y_values (x y : ℝ) :
  (∃ (k : ℤ), x = -y - 2 * Real.pi * k ∨ x = 3 * y - Real.pi - 2 * Real.pi * k) →
  (Real.cos (3 * x) / ((2 * Real.cos (2 * x) - 1) * Real.cos y) = 2 / 3 + (Real.cos (x - y))^2) →
  (Real.sin (3 * x) / ((2 * Real.cos (2 * x) + 1) * Real.sin y) = -1 / 3 - (Real.sin (x - y))^2) →
  Real.cos (x - 3 * y) = -1 ∨ Real.cos (x - 3 * y) = -1 / 3 :=
by sorry

theorem exactly_two_values (x y : ℝ) :
  (∃ (k : ℤ), x = -y - 2 * Real.pi * k ∨ x = 3 * y - Real.pi - 2 * Real.pi * k) →
  (Real.cos (3 * x) / ((2 * Real.cos (2 * x) - 1) * Real.cos y) = 2 / 3 + (Real.cos (x - y))^2) →
  (Real.sin (3 * x) / ((2 * Real.cos (2 * x) + 1) * Real.sin y) = -1 / 3 - (Real.sin (x - y))^2) →
  ∃! (S : Set ℝ), S = {-1, -1/3} ∧ ∀ z ∈ S, ∃ x y, Real.cos (x - 3 * y) = z :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_x_minus_3y_values_exactly_two_values_l394_39424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_degree_of_g_l394_39416

-- Define polynomials f, g, and h
variable (f g h : Polynomial ℝ)

-- Define the relationship between f, g, and h
def relation (f g h : Polynomial ℝ) : Prop := 5 • f + 7 • g = h

-- Theorem statement
theorem min_degree_of_g (f g h : Polynomial ℝ) 
  (rel : relation f g h) 
  (deg_f : Polynomial.degree f = 10) 
  (deg_h : Polynomial.degree h = 12) : 
  Polynomial.degree g ≥ 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_degree_of_g_l394_39416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_satisfies_equation_l394_39407

/-- A function that satisfies f(x) + x * f(1-x) = x^2 for all real x -/
noncomputable def f (x : ℝ) : ℝ := (x^3 - 3*x^2 + x) / (x - x^2 - 1)

/-- Theorem stating that f satisfies the given equation -/
theorem f_satisfies_equation : ∀ x : ℝ, f x + x * f (1 - x) = x^2 := by
  intro x
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_satisfies_equation_l394_39407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_common_numbers_l394_39492

theorem count_common_numbers : ∃ (count : ℕ), count = 101 ∧
  count = (Finset.filter (λ x => x % 5 = 3 ∧ x % 4 = 1) (Finset.Icc 3 2021)).card :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_common_numbers_l394_39492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_plane_intersection_shapes_l394_39437

/-- A cube is a three-dimensional shape with six square faces. -/
structure Cube where
  faces : Fin 6 → Square

/-- A plane is a two-dimensional surface in three-dimensional space. -/
structure Plane

/-- Possible shapes of cross-sections when cutting a cube with a plane. -/
inductive CrossSectionShape
  | Triangle
  | Quadrilateral
  | Pentagon
  | Hexagon

/-- Represents the intersection of a cube and a plane. -/
def intersect (cube : Cube) (plane : Plane) : CrossSectionShape :=
  sorry -- Placeholder for the actual implementation

/-- The theorem stating that the possible shapes of cross-sections when cutting a cube 
    with a plane are exactly triangle, quadrilateral, pentagon, and hexagon. -/
theorem cube_plane_intersection_shapes (cube : Cube) (plane : Plane) :
  ∃ (shape : CrossSectionShape), intersect cube plane = shape ∧
  (shape = CrossSectionShape.Triangle ∨
   shape = CrossSectionShape.Quadrilateral ∨
   shape = CrossSectionShape.Pentagon ∨
   shape = CrossSectionShape.Hexagon) := by
  sorry -- Placeholder for the actual proof


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_plane_intersection_shapes_l394_39437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l394_39498

/-- The equation of the curve -/
def f (x : ℝ) : ℝ := -x^3 + 3*x^2

/-- The point of tangency -/
def point : ℝ × ℝ := (1, 2)

/-- The slope of the tangent line -/
def tangent_slope : ℝ := 3

/-- The y-intercept of the tangent line -/
def y_intercept : ℝ := -1

/-- Theorem: The tangent line to the curve y = -x^3 + 3x^2 at (1, 2) is y = 3x - 1 -/
theorem tangent_line_equation :
  let (x₀, y₀) := point
  let m := tangent_slope
  let b := y_intercept
  (∀ x y, y = f x → (x - x₀) * (deriv f x₀) = y - y₀) →
  f x₀ = y₀ →
  deriv f x₀ = m →
  y₀ = m * x₀ + b :=
by
  sorry

#check tangent_line_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l394_39498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l394_39484

-- Define the function as noncomputable due to the use of Real.sqrt
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (7 + 6*x - x^2)

-- Define the domain
def domain : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 7}

-- Theorem statement
theorem f_domain : 
  {x : ℝ | ∃ y, f x = y} = domain := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l394_39484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_inner_circle_radius_inner_circle_radius_value_l394_39486

/-- Two externally tangent circles with a common external tangent of length 20 -/
structure TangentCircles where
  R : ℝ
  r : ℝ
  tangent_length : R + r = 20

/-- A circle tangent to both circles and their common tangent -/
noncomputable def inner_circle_radius (tc : TangentCircles) : ℝ :=
  100 / ((tc.R.sqrt + tc.r.sqrt) ^ 2)

/-- The theorem stating that the radii of both circles must be 10 to maximize the inner circle radius -/
theorem max_inner_circle_radius (tc : TangentCircles) :
  inner_circle_radius tc ≤ inner_circle_radius ⟨10, 10, by norm_num⟩ := by
  sorry

/-- The theorem stating that when both radii are 10, the inner circle radius is 2.5 -/
theorem inner_circle_radius_value :
  inner_circle_radius ⟨10, 10, by norm_num⟩ = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_inner_circle_radius_inner_circle_radius_value_l394_39486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_xy_product_l394_39490

/-- Given that D = (4, -2) is the midpoint of EF, where E = (x, -6) and F = (10, y), prove that xy = -4 -/
theorem midpoint_xy_product (x y : ℝ) : 
  (4, -2) = ((x + 10) / 2, (-6 + y) / 2) → x * y = -4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_xy_product_l394_39490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_piravena_trip_cost_l394_39465

/-- Represents the cost structure for transportation --/
structure TransportCost where
  busCostPerKm : ℚ
  planeCostPerKm : ℚ
  planeBookingFee : ℚ

/-- Represents the distances between cities --/
structure CityDistances where
  XY : ℚ
  YZ : ℚ

/-- Calculates the total cost of Piravena's round trip --/
def totalTripCost (costs : TransportCost) (distances : CityDistances) : ℚ :=
  let ZX := (distances.XY ^ 2 + distances.YZ ^ 2).sqrt
  let flightCost := costs.planeBookingFee + costs.planeCostPerKm * distances.XY
  let busYZCost := costs.busCostPerKm * distances.YZ
  let busZXCost := costs.busCostPerKm * ZX
  flightCost + busYZCost + busZXCost

/-- The main theorem stating the total cost of Piravena's round trip --/
theorem piravena_trip_cost :
  let costs := TransportCost.mk (20/100) (12/100) 120
  let distances := CityDistances.mk 4000 3000
  totalTripCost costs distances = 2200 := by
  sorry

#eval totalTripCost (TransportCost.mk (20/100) (12/100) 120) (CityDistances.mk 4000 3000)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_piravena_trip_cost_l394_39465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_liquid_level_rise_ratio_is_half_l394_39404

/-- Represents a right circular cone --/
structure Cone where
  radius : ℝ
  height : ℝ

/-- Represents a spherical marble --/
structure Marble where
  radius : ℝ

/-- The ratio of liquid level rise in two cones after adding marbles --/
noncomputable def liquidLevelRiseRatio (narrowCone wideCone : Cone) (narrowMarble wideMarble : Marble) : ℝ :=
  let narrowRise := narrowMarble.radius^3 / (3 * narrowCone.radius^2)
  let wideRise := wideMarble.radius^3 / (3 * wideCone.radius^2)
  narrowRise / wideRise

theorem liquid_level_rise_ratio_is_half 
  (narrowCone wideCone : Cone) 
  (narrowMarble wideMarble : Marble) :
  narrowCone.radius = 4 →
  wideCone.radius = 8 →
  narrowMarble.radius = 1 →
  wideMarble.radius = 2 →
  narrowCone.height * narrowCone.radius^2 = wideCone.height * wideCone.radius^2 →
  liquidLevelRiseRatio narrowCone wideCone narrowMarble wideMarble = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_liquid_level_rise_ratio_is_half_l394_39404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_roots_iff_m_in_range_l394_39473

-- Define the function f(x) = (mx - 3 + √2)²
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m * x - 3 + Real.sqrt 2) ^ 2

-- Define the function g(x) = √(x + m)
noncomputable def g (m : ℝ) (x : ℝ) : ℝ := Real.sqrt (x + m)

-- State the theorem
theorem two_roots_iff_m_in_range (m : ℝ) :
  (m > 0 ∧
   ∃! (r₁ r₂ : ℝ), 0 ≤ r₁ ∧ r₁ < r₂ ∧ r₂ ≤ 1 ∧
     f m r₁ = g m r₁ ∧ f m r₂ = g m r₂ ∧
     ∀ x, 0 ≤ x ∧ x ≤ 1 → (f m x = g m x ↔ x = r₁ ∨ x = r₂))
  ↔
  (3 ≤ m ∧ m ≤ 193 - 132 * Real.sqrt 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_roots_iff_m_in_range_l394_39473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_f_equals_half_plus_pi_quarter_magnitude_z₁_equals_ten_thirds_l394_39482

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ∈ Set.Icc (-1) 0 then x + 1
  else if x ∈ Set.Ico 0 1 then Real.sqrt (1 - x^2)
  else 0

-- Define complex numbers z₁ and z₂
def z₁ (a : ℝ) : ℂ := a + 2*Complex.I
def z₂ : ℂ := 3 - 4*Complex.I

-- Theorem for the definite integral
theorem integral_f_equals_half_plus_pi_quarter :
  ∫ x in Set.Icc (-1) 1, f x = 1/2 + π/4 := by sorry

-- Theorem for the magnitude of z₁
theorem magnitude_z₁_equals_ten_thirds :
  ∃ a : ℝ, (z₁ a / z₂).im ≠ 0 ∧ (z₁ a / z₂).re = 0 ∧ Complex.abs (z₁ a) = 10/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_f_equals_half_plus_pi_quarter_magnitude_z₁_equals_ten_thirds_l394_39482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_perimeter_triangle_l394_39456

theorem min_perimeter_triangle (d e f : ℝ) : 
  d > 0 → e > 0 → f > 0 →
  (d^2 + e^2 - f^2) / (2 * d * e) = 9/16 →
  (e^2 + f^2 - d^2) / (2 * e * f) = 12/13 →
  (f^2 + d^2 - e^2) / (2 * f * d) = -1/3 →
  d + e + f ≥ 30 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_perimeter_triangle_l394_39456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_C_max_CP_value_l394_39421

-- Part (a)
def circle_center (z : ℂ) := {w : ℂ | ∃ (r : ℝ), Complex.abs (w - z) = r}

structure MovingPoint (ω : ℝ) where
  center : ℂ
  radius : ℝ
  initial_angle : ℝ

noncomputable def position (p : MovingPoint ω) (t : ℝ) : ℂ :=
  p.center + p.radius * Complex.exp (Complex.I * (ω * t + p.initial_angle))

def equilateral_constraint (A B C : ℂ) : Prop :=
  Complex.abs (B - A) = Complex.abs (C - B) ∧ Complex.abs (C - B) = Complex.abs (A - C)

theorem locus_of_C (ω : ℝ) (A B : MovingPoint ω) :
  ∃ (C : MovingPoint ω), ∀ t, equilateral_constraint (position A t) (position B t) (position C t) := by
  sorry

-- Part (b)
def equilateral_triangle (A B C : ℂ) : Prop :=
  Complex.abs (B - A) = Complex.abs (C - B) ∧ 
  Complex.abs (C - B) = Complex.abs (A - C) ∧ 
  Complex.abs (A - C) = Complex.abs (B - A)

theorem max_CP_value (A B C P : ℂ) (h1 : equilateral_triangle A B C) 
    (h2 : Complex.abs (P - A) = 2) (h3 : Complex.abs (P - B) = 3) :
  ∀ P', Complex.abs (P' - A) = 2 → Complex.abs (P' - B) = 3 → Complex.abs (P' - C) ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_C_max_CP_value_l394_39421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_exponential_function_l394_39433

theorem min_value_exponential_function :
  let f : ℝ → ℝ := fun x ↦ Real.exp x + 4 * Real.exp (-x)
  ∃ x₀ : ℝ, f x₀ = 4 ∧ ∀ x : ℝ, f x ≥ 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_exponential_function_l394_39433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l394_39419

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 3 * x^2 - 6 * a * x - 11
def g (x : ℝ) : ℝ := 3 * x^2 + 6 * x + 12

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 6 * x - 6 * a

-- Define the line m
def m (k : ℝ) (x : ℝ) : ℝ := k * x + 9

theorem problem_solution :
  (∃ a : ℝ, f' a (-1) = 0 ∧ a = -2) ∧
  (∃ k : ℝ, 
    (∀ x : ℝ, (m k x = f (-2) x → (∃ y : ℝ, f' (-2) y = k)) ∧
              (m k x = g x → (∃ y : ℝ, (deriv g) y = k))) ∧
    k = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l394_39419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_line_intersection_l394_39475

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Defines the hyperbola Γ: x^2 - y^2/8 = 1 -/
def hyperbola (p : Point) : Prop :=
  p.x^2 - p.y^2/8 = 1

/-- The right focus F2 of the hyperbola -/
def F2 : Point :=
  ⟨3, 0⟩

/-- The left vertex A of the hyperbola -/
def A : Point :=
  ⟨-1, 0⟩

/-- Checks if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  p.y = l.slope * (p.x - F2.x) + F2.y

/-- Calculates the slope between two points -/
noncomputable def slopeBetween (p1 p2 : Point) : ℝ :=
  (p2.y - p1.y) / (p2.x - p1.x)

/-- Main theorem -/
theorem hyperbola_line_intersection 
  (l : Line) (M N : Point) :
  hyperbola M ∧ hyperbola N ∧
  pointOnLine M l ∧ pointOnLine N l ∧
  slopeBetween A M + slopeBetween A N = -1/2 →
  l.slope = -8 ∧ l.intercept = 24 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_line_intersection_l394_39475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l394_39406

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (2 * x - 4) + (2 * x - 6) ^ (1/3 : ℝ)

-- State the theorem about the domain of f
theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x ≥ 2} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l394_39406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_multiples_of_4_and_9_l394_39409

theorem three_digit_multiples_of_4_and_9 : 
  Finset.card (Finset.filter (fun n => 100 ≤ n ∧ n ≤ 999 ∧ 4 ∣ n ∧ 9 ∣ n) (Finset.range 1000)) = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_multiples_of_4_and_9_l394_39409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_OAB_l394_39422

/-- Parabola C: y² = 3x -/
def C (x y : ℝ) : Prop := y^2 = 3*x

/-- Focus F of the parabola C -/
noncomputable def F : ℝ × ℝ := (3/4, 0)

/-- A line passing through F intersects C at points A and B -/
def line_intersects_C (A B : ℝ × ℝ) : Prop :=
  C A.1 A.2 ∧ C B.1 B.2 ∧ ∃ (t : ℝ), A = F + t • (B - F)

/-- The origin O -/
def O : ℝ × ℝ := (0, 0)

/-- Area of triangle OAB -/
noncomputable def area_OAB (A B : ℝ × ℝ) : ℝ :=
  abs ((A.1 * B.2 - B.1 * A.2) / 2)

/-- The minimum area of triangle OAB is 9/8 -/
theorem min_area_OAB :
  ∃ (min_area : ℝ), min_area = 9/8 ∧
  ∀ (A B : ℝ × ℝ), line_intersects_C A B →
  area_OAB A B ≥ min_area := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_OAB_l394_39422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_is_correct_l394_39400

/-- The complex polynomial whose roots form the rhombus -/
def f (z : ℂ) : ℂ := z^4 + (2 + 2*Complex.I)*z^3 + (1 - 3*Complex.I)*z^2 - (6 + 4*Complex.I)*z + (8 - 2*Complex.I)

/-- The roots of the polynomial f -/
noncomputable def roots : Finset ℂ := sorry

/-- The area of the rhombus formed by the roots of f -/
noncomputable def rhombusArea : ℝ := sorry

/-- Theorem stating that the area of the rhombus is 5√218 / 8 -/
theorem rhombus_area_is_correct : rhombusArea = 5 * Real.sqrt 218 / 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_is_correct_l394_39400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_volume_ratio_in_cone_l394_39444

/-- For a cone filled with water to 2/3 of its height, the ratio of the water volume to the cone's volume is 8/27. -/
theorem water_volume_ratio_in_cone (h r : ℝ) (h_pos : h > 0) (r_pos : r > 0) : 
  (1/3) * π * ((2/3) * r)^2 * ((2/3) * h) / ((1/3) * π * r^2 * h) = 8/27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_volume_ratio_in_cone_l394_39444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_243_three_fifths_l394_39467

theorem power_of_243_three_fifths : (243 : ℝ) ^ (3/5 : ℝ) = 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_243_three_fifths_l394_39467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infiniteSeriesSumValue_l394_39413

/-- The sum of the infinite series 1 - 3(1/999) + 5(1/999)^2 - 7(1/999)^3 + ... -/
noncomputable def infiniteSeriesSum : ℝ := ∑' n, ((-1)^n * (2*n + 1) * (1/999)^n)

/-- The sum of the infinite series is equal to 996995/497004 -/
theorem infiniteSeriesSumValue : infiniteSeriesSum = 996995/497004 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infiniteSeriesSumValue_l394_39413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plastic_bag_degradation_l394_39412

/-- The photodegradation coefficient -/
noncomputable def k : ℝ := Real.log (Real.sqrt 0.75) / 2

/-- The time it takes for the residual amount to be 10% of the initial amount -/
noncomputable def t : ℝ := Real.log 0.1 / Real.log (Real.sqrt 0.75)

/-- Theorem stating that t is approximately 16 years -/
theorem plastic_bag_degradation :
  (|t - 16| < 0.5) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plastic_bag_degradation_l394_39412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mowing_time_closest_to_2_2_l394_39464

/-- Represents the lawn mowing problem --/
structure LawnMowing where
  length : ℝ  -- length of the lawn in feet
  width : ℝ   -- width of the lawn in feet
  swath : ℝ   -- swath width in inches
  overlap : ℝ  -- overlap in inches
  rate : ℝ    -- mowing rate in feet per hour
  extra_time : ℝ  -- extra time in minutes

/-- Calculates the time needed to mow the lawn --/
noncomputable def mowing_time (l : LawnMowing) : ℝ :=
  let effective_swath := (l.swath - l.overlap) / 12  -- convert to feet
  let strips := l.width / effective_swath
  let total_distance := strips * l.length
  let mowing_hours := total_distance / l.rate
  mowing_hours + l.extra_time / 60  -- convert extra time to hours

/-- Theorem stating that the mowing time is closest to 2.2 hours --/
theorem mowing_time_closest_to_2_2 (l : LawnMowing) 
  (h1 : l.length = 100)
  (h2 : l.width = 180)
  (h3 : l.swath = 30)
  (h4 : l.overlap = 6)
  (h5 : l.rate = 4500)
  (h6 : l.extra_time = 10) :
  abs (mowing_time l - 2.2) ≤ abs (mowing_time l - 2.0) ∧ 
  abs (mowing_time l - 2.2) ≤ abs (mowing_time l - 2.1) ∧
  abs (mowing_time l - 2.2) ≤ abs (mowing_time l - 2.3) ∧
  abs (mowing_time l - 2.2) ≤ abs (mowing_time l - 2.5) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mowing_time_closest_to_2_2_l394_39464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_point_inside_circle_point_inside_implies_intersection_l394_39463

-- Define the circle C
def my_circle (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 3 = 0

-- Define the line l
def my_line (x y a : ℝ) : Prop := x + a*y + 2 - a = 0

-- Theorem statement
theorem line_intersects_circle :
  ∀ a : ℝ, ∃ x y : ℝ, my_circle x y ∧ my_line x y a :=
by
  sorry

-- Proof that the point (-2, 1) is inside the circle
theorem point_inside_circle :
  my_circle (-2) 1 = False :=
by
  unfold my_circle
  norm_num

-- Theorem stating that if a point is inside the circle, the line through it intersects the circle
theorem point_inside_implies_intersection :
  ∀ a : ℝ, my_circle (-2) 1 = False →
  ∃ x y : ℝ, my_circle x y ∧ my_line x y a :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_point_inside_circle_point_inside_implies_intersection_l394_39463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_ratio_range_l394_39442

-- Define the hyperbola
def is_on_hyperbola (x y : ℝ) : Prop := x^2 / 8 - y^2 / 4 = 1

-- Define the foci
noncomputable def F₁ : ℝ × ℝ := (-2 * Real.sqrt 3, 0)
noncomputable def F₂ : ℝ × ℝ := (2 * Real.sqrt 3, 0)

-- Define the origin
def O : ℝ × ℝ := (0, 0)

-- Define the distance function
noncomputable def distance (p₁ p₂ : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2)

-- Theorem statement
theorem hyperbola_ratio_range (x y : ℝ) :
  is_on_hyperbola x y →
  2 < (distance (x, y) F₁ + distance (x, y) F₂) / distance (x, y) O ∧
  (distance (x, y) F₁ + distance (x, y) F₂) / distance (x, y) O ≤ Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_ratio_range_l394_39442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_term_b_is_third_l394_39403

-- Define the sequences
def a (n : ℕ) : ℝ := sorry
def S (n : ℕ) : ℝ := sorry
def b (n : ℕ) : ℝ := sorry

-- Define the conditions
axiom condition1 : ∀ n : ℕ, n ≥ 1 → S n - a n = (n - 1)^2
axiom condition2 : ∀ n : ℕ, n ≥ 1 → b n = 2^(a n) / (S n)^2

-- State the theorem
theorem min_term_b_is_third : 
  ∀ n : ℕ, n ≥ 1 → b 3 ≤ b n :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_term_b_is_third_l394_39403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_inequality_l394_39477

theorem triangle_sine_inequality (a b c : ℝ) 
  (h1 : 0 < a ∧ 0 < b ∧ 0 < c)
  (h2 : a + b > c ∧ b + c > a ∧ c + a > b)
  (h3 : a + b + c ≤ 2 * Real.pi) :
  Real.sin a + Real.sin b > Real.sin c ∧ 
  Real.sin b + Real.sin c > Real.sin a ∧ 
  Real.sin c + Real.sin a > Real.sin b := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_inequality_l394_39477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inclination_angle_range_l394_39469

/-- A line that passes through the second and fourth quadrants -/
structure LineInSecondFourthQuadrants where
  slope : ℝ
  passes_through_second_fourth : slope < 0

/-- The inclination angle of a line -/
noncomputable def inclination_angle (l : LineInSecondFourthQuadrants) : ℝ :=
  Real.arctan l.slope + Real.pi / 2

/-- Theorem: The inclination angle of a line passing through the second and fourth quadrants
    is between 90° and 180° (exclusive) -/
theorem inclination_angle_range (l : LineInSecondFourthQuadrants) :
  Real.pi / 2 < inclination_angle l ∧ inclination_angle l < Real.pi := by
  sorry

#check inclination_angle_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inclination_angle_range_l394_39469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_angle_l394_39457

theorem parallel_vectors_angle (α : Real) : 
  α > 0 ∧ α < π / 2 → -- α is an acute angle
  (∃ (k : Real), k ≠ 0 ∧ k • (3 * Real.cos α, 2) = (3, 4 * Real.sin α)) → -- vectors are parallel
  α = π / 4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_angle_l394_39457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_sum_identity_l394_39481

theorem binomial_sum_identity (n m : ℕ) (h : 4*m - 3 ≤ n) :
  (Finset.range m).sum (λ i => Nat.choose n (4*i + 1)) = 
    (1/2 : ℝ) * (2^(n - 1) + 2^(n/2 : ℝ) * Real.sin (n * Real.pi / 4)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_sum_identity_l394_39481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_l394_39493

/-- Converts kilometers per hour to meters per second -/
noncomputable def km_per_hr_to_m_per_s (speed : ℝ) : ℝ :=
  speed * 1000 / 3600

/-- Calculates the time (in seconds) it takes for an object to travel a given distance at a given speed -/
noncomputable def time_to_pass (length : ℝ) (speed : ℝ) : ℝ :=
  length / speed

theorem train_passing_time (train_length : ℝ) (train_speed_km_hr : ℝ)
    (h1 : train_length = 280)
    (h2 : train_speed_km_hr = 36) :
    time_to_pass train_length (km_per_hr_to_m_per_s train_speed_km_hr) = 28 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_l394_39493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_arccos_cos_l394_39426

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.arccos (Real.cos x)

-- Define the area function
noncomputable def area_under_curve (a b : ℝ) (f : ℝ → ℝ) : ℝ :=
  ∫ x in a..b, f x

-- Theorem statement
theorem area_arccos_cos : area_under_curve 0 (2 * Real.pi) f = Real.pi ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_arccos_cos_l394_39426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_sum_l394_39446

/-- Represents an arithmetic progression -/
structure ArithmeticProgression where
  first : ℚ
  common_difference : ℚ

/-- Calculates the sum of the first n terms of an arithmetic progression -/
def sum_of_terms (ap : ArithmeticProgression) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (2 * ap.first + (n - 1 : ℚ) * ap.common_difference)

/-- The main theorem stating the properties of the specific arithmetic progression -/
theorem arithmetic_progression_sum :
  ∃ (ap : ArithmeticProgression),
    sum_of_terms ap 15 = 150 ∧
    sum_of_terms ap 90 = 15 ∧
    sum_of_terms ap 105 = -189 := by
  sorry

#eval sum_of_terms ⟨100, -1⟩ 15

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_sum_l394_39446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nice_numbers_l394_39445

def IsNice (k : ℕ) : Prop :=
  k > 1 ∧ ∀ m n : ℕ, m > 0 → n > 0 → (k * n + m ∣ k * m + n) → (n ∣ m)

theorem nice_numbers : {k : ℕ | IsNice k} = {2, 3, 5} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nice_numbers_l394_39445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l394_39401

-- Define the * operation
noncomputable def star (a b : ℝ) : ℝ := min a b

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := star (Real.sin x) (Real.cos x)

-- State the theorem
theorem range_of_f :
  Set.range f = Set.Icc (-1) (Real.sqrt 2 / 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l394_39401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_l394_39414

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- The circle (x-2)^2 + y^2 = 2 -/
def circle_eq (x y : ℝ) : Prop :=
  (x - 2)^2 + y^2 = 2

/-- The asymptotes of the hyperbola intersect with the circle -/
def asymptotes_intersect_circle (h : Hyperbola) : Prop :=
  ∃ x y : ℝ, circle_eq x y ∧ (h.b * x = h.a * y ∨ h.b * x = -h.a * y)

/-- The main theorem: if the asymptotes of the hyperbola intersect with the circle,
    then the eccentricity is between 1 and √2 -/
theorem eccentricity_range (h : Hyperbola) 
    (h_intersect : asymptotes_intersect_circle h) : 
    1 < eccentricity h ∧ eccentricity h < Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_l394_39414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_project_completion_theorem_l394_39494

/-- Represents the number of days it takes to complete the project -/
noncomputable def project_completion_time (a_time b_time c_time : ℝ) (a_quit b_quit : ℝ) : ℕ :=
  let x := (a_time * b_time * c_time) / (a_time * b_time + a_time * c_time + b_time * c_time - a_quit * b_time * c_time / a_time - b_quit * a_time * c_time / b_time)
  ⌈x⌉.toNat

/-- Theorem stating that under the given conditions, the project will be completed in 16 days -/
theorem project_completion_theorem :
  project_completion_time 20 30 40 10 5 = 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_project_completion_theorem_l394_39494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_l394_39408

theorem problem_1 : 12 - (-8) + (-11) = 9 := by
  -- Evaluate the expression step by step
  calc
    12 - (-8) + (-11) = 12 + 8 + (-11) := by ring
    _ = 20 + (-11) := by ring
    _ = 9 := by ring

-- QED

#eval 12 - (-8) + (-11)  -- This will output 9

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_l394_39408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2alpha_minus_sin_cos_l394_39423

theorem cos_2alpha_minus_sin_cos (α : ℝ) (h : Real.tan α = 2) : 
  Real.cos (2 * α) - Real.sin α * Real.cos α = -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2alpha_minus_sin_cos_l394_39423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_relation_l394_39425

-- Define a triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the length of a side
noncomputable def side_length (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the angle at a vertex
noncomputable def angle (p q r : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem triangle_side_relation (t : Triangle) (a b c : ℝ) :
  side_length t.B t.C = a →
  side_length t.A t.C = b →
  side_length t.A t.B = c →
  3 * angle t.B t.A t.C + angle t.A t.B t.C = π →
  3 * a = 2 * c →
  b = 5 * a / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_relation_l394_39425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lucy_peter_difference_l394_39495

/-- The amount of chestnuts picked by Mary, in kg -/
noncomputable def mary_amount : ℝ := 12

/-- The amount of chestnuts picked by Peter, in kg -/
noncomputable def peter_amount : ℝ := mary_amount / 2

/-- The total amount of chestnuts picked by all three, in kg -/
noncomputable def total_amount : ℝ := 26

/-- The amount of chestnuts picked by Lucy, in kg -/
noncomputable def lucy_amount : ℝ := total_amount - mary_amount - peter_amount

theorem lucy_peter_difference : lucy_amount - peter_amount = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lucy_peter_difference_l394_39495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_namjoon_korean_score_l394_39402

/-- Represents the test scores of a student -/
structure TestScores where
  korean : ℚ
  math : ℚ
  english : ℚ

/-- Calculates the average score of the three subjects -/
def average (scores : TestScores) : ℚ :=
  (scores.korean + scores.math + scores.english) / 3

/-- Theorem: Given the conditions, Namjoon's Korean test score is 90 -/
theorem namjoon_korean_score :
  ∃ (scores : TestScores),
    scores.math = 100 ∧
    scores.english = 95 ∧
    average scores = 95 ∧
    scores.korean = 90 :=
by
  -- Construct the TestScores
  let scores : TestScores := {
    korean := 90,
    math := 100,
    english := 95
  }
  
  -- Prove that this TestScores satisfies all conditions
  exists scores
  constructor
  · rfl
  constructor
  · rfl
  constructor
  · -- Prove that the average is 95
    unfold average
    norm_num
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_namjoon_korean_score_l394_39402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_basement_pumping_time_l394_39440

/-- Pumping time for a flooded basement --/
theorem basement_pumping_time
  (basement_length : ℝ)
  (basement_width : ℝ)
  (water_depth_inches : ℝ)
  (num_pumps : ℕ)
  (pump_rate : ℝ)
  (gallons_per_cubic_foot : ℝ)
  (h1 : basement_length = 40)
  (h2 : basement_width = 20)
  (h3 : water_depth_inches = 24)
  (h4 : num_pumps = 4)
  (h5 : pump_rate = 10)
  (h6 : gallons_per_cubic_foot = 7.5) :
  (basement_length * basement_width * (water_depth_inches / 12) * gallons_per_cubic_foot) /
  (↑num_pumps * pump_rate) = 300 := by
  sorry

#check basement_pumping_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_basement_pumping_time_l394_39440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_juan_stamp_collection_cost_l394_39496

/-- Represents a country with its stamp quantities and price --/
structure Country where
  name : String
  price : Nat
  stamps_80s : Nat
  stamps_90s : Nat
deriving BEq, Repr

/-- Calculates the cost of stamps for a given country and decade --/
def calculate_cost (country : Country) (decade : String) : Rat :=
  let stamps := if decade = "80s" then country.stamps_80s else country.stamps_90s
  (stamps * country.price : Rat) / 100

/-- Defines the set of European countries --/
def european_countries : List String := ["France", "Spain", "Italy"]

/-- Checks if a country is European --/
def is_european (country : Country) : Bool :=
  european_countries.contains country.name

/-- Calculates the total cost of European stamps for both decades --/
def total_european_cost (countries : List Country) : Rat :=
  countries.filter is_european
    |>.map (λ c => calculate_cost c "80s" + calculate_cost c "90s")
    |>.sum

/-- The main theorem to prove --/
theorem juan_stamp_collection_cost 
  (countries : List Country) 
  (h_france : countries.contains ⟨"France", 7, 15, 14⟩)
  (h_spain : countries.contains ⟨"Spain", 6, 11, 10⟩)
  (h_italy : countries.contains ⟨"Italy", 8, 14, 12⟩) :
  total_european_cost countries = 537 / 100 := by
  sorry

#eval total_european_cost [⟨"France", 7, 15, 14⟩, ⟨"Spain", 6, 11, 10⟩, ⟨"Italy", 8, 14, 12⟩]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_juan_stamp_collection_cost_l394_39496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_freight_departure_time_l394_39460

/-- The distance between points A and B in kilometers. -/
noncomputable def total_distance : ℝ := 300

/-- The speed of the passenger train in km/h. -/
noncomputable def passenger_speed : ℝ := 60

/-- The speed of the freight train in km/h. -/
noncomputable def freight_speed : ℝ := 40

/-- The time (in hours) it takes for the trains to meet after the passenger train departs. -/
noncomputable def meeting_time : ℝ := total_distance / (passenger_speed + freight_speed)

/-- The time (in hours) it takes for the passenger train to complete its journey. -/
noncomputable def passenger_travel_time : ℝ := total_distance / passenger_speed

/-- The time (in hours) it takes for the freight train to complete its journey. -/
noncomputable def freight_travel_time : ℝ := total_distance / freight_speed

/-- The departure time of the passenger train in hours after midnight. -/
noncomputable def passenger_departure_time : ℝ := 12

/-- Theorem stating that the freight train departed at 9:30 AM given the problem conditions. -/
theorem freight_departure_time : 
  passenger_departure_time + passenger_travel_time - freight_travel_time = 9.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_freight_departure_time_l394_39460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_trapezoid_area_l394_39415

-- Define the trapezoid
structure RightTrapezoid where
  lowerBase : ℝ
  upperBase : ℝ
  height : ℝ

-- Define the conditions
def trapezoidConditions (t : RightTrapezoid) : Prop :=
  t.upperBase = 0.6 * t.lowerBase ∧
  t.upperBase + 24 = t.lowerBase ∧
  t.height = t.lowerBase

-- Define the area calculation
noncomputable def trapezoidArea (t : RightTrapezoid) : ℝ :=
  (t.lowerBase + t.upperBase) * t.height / 2

-- The theorem to prove
theorem right_trapezoid_area : 
  ∀ t : RightTrapezoid, trapezoidConditions t → trapezoidArea t = 2880 := by
  intro t hypothesis
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_trapezoid_area_l394_39415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l394_39497

noncomputable def f (x : ℝ) := Real.sin x ^ 2 + Real.sin x * Real.cos x + 1

theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ (∀ x, f (x + T) = f x) ∧
    (∀ T' > 0, (∀ x, f (x + T') = f x) → T ≤ T')) ∧
  (∃ (m : ℝ), (∀ x, f x ≥ m) ∧ (∃ x₀, f x₀ = m)) ∧
  (let T := Real.pi;
   let m := (3 - Real.sqrt 2) / 2;
   (∀ x, f (x + T) = f x) ∧
   (∀ x, f x ≥ m) ∧
   (∃ x₀, f x₀ = m)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l394_39497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lawn_problem_l394_39436

/-- Represents the dimensions and cost of a lawn with intersecting roads -/
structure LawnWithRoads where
  length : ℝ
  width : ℝ
  road_width : ℝ
  total_cost : ℝ

/-- Calculates the cost per square meter of traveling the roads -/
noncomputable def cost_per_sq_meter (lawn : LawnWithRoads) : ℝ :=
  let road_area := lawn.length * lawn.road_width + lawn.width * lawn.road_width - lawn.road_width * lawn.road_width
  lawn.total_cost / road_area

/-- Theorem statement for the lawn problem -/
theorem lawn_problem (lawn : LawnWithRoads) 
  (h1 : lawn.length = 80)
  (h2 : lawn.width = 50)
  (h3 : lawn.road_width = 10)
  (h4 : lawn.total_cost = 3600) :
  cost_per_sq_meter lawn = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lawn_problem_l394_39436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_series_sum_l394_39438

/-- Sum of an arithmetic sequence -/
noncomputable def arithmetic_sum (a₁ : ℚ) (aₙ : ℚ) (n : ℕ) : ℚ :=
  n * (a₁ + aₙ) / 2

/-- Number of terms in an arithmetic sequence -/
def num_terms (a₁ : ℚ) (aₙ : ℚ) (d : ℚ) : ℕ :=
  Int.toNat ((aₙ - a₁) / d + 1).ceil

theorem arithmetic_series_sum :
  let a₁ : ℚ := 40
  let aₙ : ℚ := 80
  let d : ℚ := 1
  let n : ℕ := num_terms a₁ aₙ d
  arithmetic_sum a₁ aₙ n = 2460 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_series_sum_l394_39438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_l394_39461

-- Define the line l
def line_l (x y : ℝ) : Prop := 2 * x - y + 1 = 0

-- Define the circle C
noncomputable def circle_C (x y : ℝ) : Prop := x^2 + y^2 = 2 * Real.sqrt 2 * y

-- Define the center of the circle
noncomputable def circle_center : ℝ × ℝ := (0, Real.sqrt 2)

-- Define the radius of the circle
noncomputable def circle_radius : ℝ := Real.sqrt 2

-- Define the distance from a point to a line
noncomputable def distance_point_to_line (x y : ℝ) : ℝ :=
  abs (-y + 1) / Real.sqrt 5

-- Theorem statement
theorem line_intersects_circle :
  ∃ (x y : ℝ), line_l x y ∧ circle_C x y :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_l394_39461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_and_volume_formula_l394_39459

/-- Regular quadrilateral pyramid with base diagonal equal to lateral edge -/
structure RegularQuadPyramid where
  a : ℝ  -- length of base diagonal and lateral edge
  a_pos : a > 0

namespace RegularQuadPyramid

/-- The side length of the square base -/
noncomputable def baseSide (p : RegularQuadPyramid) : ℝ := p.a / Real.sqrt 2

/-- The height of the pyramid -/
noncomputable def height (p : RegularQuadPyramid) : ℝ := p.a * Real.sqrt 3 / 2

/-- The total surface area of the pyramid -/
noncomputable def totalSurfaceArea (p : RegularQuadPyramid) : ℝ := 
  (p.a^2 * (Real.sqrt 7 + 1)) / 2

/-- The volume of the pyramid -/
noncomputable def volume (p : RegularQuadPyramid) : ℝ := 
  (Real.sqrt 3 * p.a^3) / 12

theorem surface_area_and_volume_formula (p : RegularQuadPyramid) : 
  totalSurfaceArea p = (p.a^2 * (Real.sqrt 7 + 1)) / 2 ∧ 
  volume p = (Real.sqrt 3 * p.a^3) / 12 := by
  sorry

end RegularQuadPyramid

end NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_and_volume_formula_l394_39459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_guards_theorem_l394_39479

/-- Represents a configuration of guards in a heptagonal castle --/
def GuardConfiguration := Fin 7 → ℕ

/-- The sum of guards in a configuration --/
def total_guards (config : GuardConfiguration) : ℕ :=
  (Finset.univ.sum config)

/-- Checks if a configuration satisfies the guard requirement for each wall --/
def is_valid_configuration (config : GuardConfiguration) : Prop :=
  ∀ i : Fin 7, config i + config (i.succ) ≥ 7

/-- The minimum number of guards needed for a valid configuration --/
def min_guards : ℕ := 25

/-- Theorem stating the minimum number of guards required --/
theorem min_guards_theorem :
  ∀ config : GuardConfiguration,
    is_valid_configuration config →
    total_guards config ≥ min_guards :=
by
  sorry

#check min_guards_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_guards_theorem_l394_39479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_pentagon_jc_length_l394_39476

/-- A pentagon CGHJK with specific properties -/
structure SpecialPentagon where
  -- Points of the pentagon
  C : ℝ × ℝ
  G : ℝ × ℝ
  H : ℝ × ℝ
  J : ℝ × ℝ
  K : ℝ × ℝ
  J' : ℝ × ℝ
  -- GH and HJ are equal
  gh_hj_equal : dist G H = dist H J
  -- GH and HJ are perpendicular
  gh_hj_perpendicular : (G.1 - H.1) * (J.1 - H.1) + (G.2 - H.2) * (J.2 - H.2) = 0
  -- Pentagon is symmetric about JJ'
  symmetric_about_jj' : ∀ (P : ℝ × ℝ), dist P J = dist P J' → 
    dist P C = dist P K ∧ dist P G = dist P H

/-- The length of J'C that makes the pentagon equilateral -/
noncomputable def equilateral_jc_length (p : SpecialPentagon) : ℝ := (Real.sqrt 7 - 1) / 6

/-- Theorem: The length of J'C that makes the pentagon equilateral is (√7 - 1) / 6 -/
theorem equilateral_pentagon_jc_length (p : SpecialPentagon) :
  (∀ (X Y : ℝ × ℝ), X ∈ ({p.C, p.G, p.H, p.J, p.K} : Set (ℝ × ℝ)) → 
    Y ∈ ({p.C, p.G, p.H, p.J, p.K} : Set (ℝ × ℝ)) → dist X Y = dist p.C p.G) ↔
  dist p.J' p.C = equilateral_jc_length p :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_pentagon_jc_length_l394_39476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_of_three_digit_numbers_with_repeated_digits_l394_39489

theorem percentage_of_three_digit_numbers_with_repeated_digits :
  let total_three_digit_numbers : ℕ := 900
  let numbers_without_repeated_digits : ℕ := 9 * 9 * 8
  let numbers_with_repeated_digits : ℕ := total_three_digit_numbers - numbers_without_repeated_digits
  let percentage : ℚ := (numbers_with_repeated_digits : ℚ) / total_three_digit_numbers * 100
  ⌊percentage * 10⌋ / 10 = 28 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_of_three_digit_numbers_with_repeated_digits_l394_39489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_pyramid_volume_l394_39430

/-- Pyramid with equilateral triangular base -/
structure Pyramid where
  -- Side length of the equilateral triangular base
  base_side : ℝ
  -- Angle between two edges from apex to base vertices
  apex_angle : ℝ
  -- Condition: base side is positive
  base_positive : base_side > 0
  -- Condition: apex angle is between 0 and π
  angle_range : 0 < apex_angle ∧ apex_angle < Real.pi

/-- Volume of the pyramid -/
noncomputable def pyramid_volume (p : Pyramid) : ℝ :=
  1 / (3 * Real.cos p.apex_angle)

/-- Theorem: Volume of the specific pyramid -/
theorem specific_pyramid_volume :
  ∀ (p : Pyramid),
    p.base_side = 2 →
    pyramid_volume p = 1 / (3 * Real.cos p.apex_angle) :=
by
  intro p h
  rfl

#check specific_pyramid_volume

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_pyramid_volume_l394_39430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_vertices_l394_39487

-- Define the equation
noncomputable def equation (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + y^2) + abs (y + 2) = 5

-- Define the vertices of the parabolas
noncomputable def vertex1 : ℝ × ℝ := (0, 3/2)
noncomputable def vertex2 : ℝ × ℝ := (0, -7/2)

-- Theorem statement
theorem distance_between_vertices : 
  abs (vertex1.2 - vertex2.2) = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_vertices_l394_39487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_to_nearest_whole_number_l394_39434

def number : ℝ := 7564.49999997

theorem round_to_nearest_whole_number :
  Int.floor (number + 0.5) = 7564 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_to_nearest_whole_number_l394_39434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_ABCD_area_l394_39418

/-- Trapezoid in a 2D coordinate system -/
structure Trapezoid where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- Calculate the area of a trapezoid -/
noncomputable def trapezoidArea (t : Trapezoid) : ℝ :=
  let h := (t.C.1 - t.A.1)
  let b1 := abs (t.B.2 - t.A.2)
  let b2 := abs (t.C.2 - t.D.2)
  (b1 + b2) * h / 2

/-- The specific trapezoid ABCD from the problem -/
def ABCD : Trapezoid :=
  { A := (1, -2)
    B := (1, 1)
    C := (7, 7)
    D := (7, -1) }

theorem trapezoid_ABCD_area : trapezoidArea ABCD = 33 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_ABCD_area_l394_39418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_proof_l394_39458

structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

structure Trapezoid where
  a : ℝ
  b : ℝ
  h : ℝ

def isIsosceles (t : Triangle) : Prop := t.a = t.b

def areaTriangle (t : Triangle) : ℝ := sorry

def areaTrapezoid (t : Trapezoid) : ℝ := sorry

def similar (t1 t2 : Triangle) : Prop := sorry

theorem trapezoid_area_proof 
  (ABC : Triangle)
  (DBCE : Trapezoid)
  (h1 : isIsosceles ABC)
  (h2 : areaTriangle ABC = 80)
  (h3 : ∃ (smallTriangles : Finset Triangle), smallTriangles.card = 10 ∧ 
        ∀ t ∈ smallTriangles, areaTriangle t = 2 ∧ similar t ABC)
  : areaTrapezoid DBCE = 70 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_proof_l394_39458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_l394_39448

/-- The area of a parallelogram with vertices at (0, 0), (4, 0), (3, 5), and (7, 5) is 20 square units. -/
theorem parallelogram_area : ∃ (area : ℝ), area = 20 ∧
  let v1 : ℝ × ℝ := (0, 0)
  let v2 : ℝ × ℝ := (4, 0)
  let v3 : ℝ × ℝ := (3, 5)
  let v4 : ℝ × ℝ := (7, 5)
  area = (v2.1 - v1.1) * (v3.2 - v1.2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_l394_39448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_divisible_by_five_l394_39485

def transform_sequence (seq : List ℤ) : List ℤ :=
  let n := seq.length
  List.zipWith (λ a b => a - b) seq (seq.drop 1 ++ seq.take 1)

def iterate_transform (seq : List ℤ) : ℕ → List ℤ
  | 0 => seq
  | n + 1 => iterate_transform (transform_sequence seq) n

theorem sum_divisible_by_five (initial_sequence : List ℤ) :
  initial_sequence.length = 100 →
  (iterate_transform initial_sequence 5).sum % 5 = 0 := by
  sorry

#eval iterate_transform [1, 2, 3, 4, 5] 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_divisible_by_five_l394_39485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_prism_similarity_l394_39410

/-- A rectangular prism with edge lengths a, b, and c. -/
structure RectangularPrism where
  a : ℝ
  b : ℝ
  c : ℝ
  a_le_b : a ≤ b
  b_le_c : b ≤ c

/-- Similarity ratio between two rectangular prisms -/
noncomputable def similarityRatio (p1 p2 : RectangularPrism) : ℝ :=
  max (p1.a / p2.a) (max (p1.b / p2.b) (p1.c / p2.c))

/-- Two rectangular prisms are similar if their corresponding edge ratios are equal -/
def isSimilar (p1 p2 : RectangularPrism) : Prop :=
  p1.a / p2.a = p1.b / p2.b ∧ p1.b / p2.b = p1.c / p2.c

theorem rectangular_prism_similarity :
  ∃ (p : RectangularPrism),
    p.b / p.a = Real.rpow 2 (1/3) ∧
    p.c / p.b = Real.rpow 2 (1/3) ∧
    isSimilar p { a := p.a, b := p.b, c := p.c/2, a_le_b := p.a_le_b, b_le_c := sorry } :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_prism_similarity_l394_39410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_motorcycle_braking_distance_is_100_l394_39429

/-- Calculates the sum of an arithmetic sequence representing the distance traveled by a motorcycle while braking -/
def motorcycleBrakingDistance (a : ℕ) (d : ℤ) : ℕ :=
  let sequence := (List.range 5).map (fun n => (a : ℤ) + n * d)
  (sequence.filter (· > 0)).sum.toNat

/-- Theorem stating that the total distance traveled by the motorcycle during braking is 100 feet -/
theorem motorcycle_braking_distance_is_100 :
  motorcycleBrakingDistance 40 (-10) = 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_motorcycle_braking_distance_is_100_l394_39429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_min_distance_value_l394_39466

noncomputable def A : ℝ × ℝ := (-3, 5)
noncomputable def B : ℝ × ℝ := (2, 15)

def on_line (P : ℝ × ℝ) : Prop :=
  P.1 - P.2 + 5 = 0

noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

theorem min_distance_sum :
  ∃ (P : ℝ × ℝ), on_line P ∧
    ∀ (Q : ℝ × ℝ), on_line Q →
      distance P A + distance P B ≤ distance Q A + distance Q B :=
by sorry

theorem min_distance_value :
  ∃ (P : ℝ × ℝ), on_line P ∧
    distance P A + distance P B = Real.sqrt 593 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_min_distance_value_l394_39466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wire_length_for_specific_grid_l394_39450

/-- Represents a rectangular wire grid -/
structure RectangularGrid where
  squares : ℕ             -- Number of squares in the grid
  length : ℚ              -- Length of the grid
  width : ℕ               -- Number of squares in width
  height : ℕ              -- Number of squares in height

/-- Calculates the total length of wire needed for a rectangular grid -/
noncomputable def wireLength (grid : RectangularGrid) : ℚ :=
  let squareSize := grid.length / grid.width
  let horizontalWires := (grid.height + 1) * grid.length
  let verticalWires := (grid.width + 1) * (grid.height * squareSize)
  horizontalWires + verticalWires

/-- The theorem to be proved -/
theorem wire_length_for_specific_grid :
  ∃ (grid : RectangularGrid),
    grid.squares = 15 ∧
    grid.length = 10 ∧
    grid.width = 5 ∧
    grid.height = 3 ∧
    wireLength grid = 76 := by
  -- Construct the specific grid
  let grid : RectangularGrid := {
    squares := 15,
    length := 10,
    width := 5,
    height := 3
  }
  -- Prove that this grid satisfies all conditions
  have h1 : grid.squares = 15 := rfl
  have h2 : grid.length = 10 := rfl
  have h3 : grid.width = 5 := rfl
  have h4 : grid.height = 3 := rfl
  -- Calculate wire length
  have h5 : wireLength grid = 76 := by
    unfold wireLength
    -- Additional steps would be needed here to complete the proof
    sorry
  -- Conclude the existence of such a grid
  exact ⟨grid, h1, h2, h3, h4, h5⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wire_length_for_specific_grid_l394_39450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equal_parts_l394_39447

theorem complex_equal_parts (a : ℝ) : 
  (((a - Complex.I) / (1 + Complex.I)).re = 
   ((a - Complex.I) / (1 + Complex.I)).im) → 
  a = 0 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equal_parts_l394_39447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_line_segment_l394_39432

-- Define the fixed points F₁ and F₂
def F₁ : ℝ × ℝ := (-2, 0)
def F₂ : ℝ × ℝ := (2, 0)

-- Define the distance between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the set of points P satisfying the condition
def S : Set (ℝ × ℝ) := {P | distance P F₁ + distance P F₂ = 4}

-- Define the line segment between F₁ and F₂
def lineSegment : Set (ℝ × ℝ) := {P | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (t * F₂.1 + (1 - t) * F₁.1, t * F₂.2 + (1 - t) * F₁.2)}

-- Theorem statement
theorem trajectory_is_line_segment : S = lineSegment := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_line_segment_l394_39432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fruit_weights_l394_39428

-- Define the fruit weights
def banana : ℕ := sorry
def pear : ℕ := sorry
def melon : ℕ := sorry
def orange : ℕ := sorry
def kiwi : ℕ := sorry

-- Define the conditions
axiom weight_sum : banana + pear + melon + orange + kiwi = 140 + 150 + 160 + 170 + 1700
axiom melon_heaviest : melon > banana + pear + orange + kiwi
axiom banana_kiwi_eq_pear_orange : banana + kiwi = pear + orange
axiom pear_between_kiwi_orange : kiwi < pear ∧ pear < orange

-- Theorem to prove
theorem fruit_weights :
  banana = 170 ∧ pear = 150 ∧ melon = 1700 ∧ orange = 160 ∧ kiwi = 140 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fruit_weights_l394_39428
