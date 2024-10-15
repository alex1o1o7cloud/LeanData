import Mathlib

namespace NUMINAMATH_CALUDE_connor_sleep_time_l539_53924

theorem connor_sleep_time (puppy_sleep : ℕ) (luke_sleep : ℕ) (connor_sleep : ℕ) : 
  puppy_sleep = 16 →
  puppy_sleep = 2 * luke_sleep →
  luke_sleep = connor_sleep + 2 →
  connor_sleep = 6 :=
by sorry

end NUMINAMATH_CALUDE_connor_sleep_time_l539_53924


namespace NUMINAMATH_CALUDE_product_11_cubed_sum_l539_53906

theorem product_11_cubed_sum (a b c : ℕ+) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c → a * b * c = 11^3 → a + b + c = 133 := by sorry

end NUMINAMATH_CALUDE_product_11_cubed_sum_l539_53906


namespace NUMINAMATH_CALUDE_roque_bike_trips_l539_53999

/-- Represents the number of times Roque rides his bike to and from work per week -/
def bike_trips : ℕ := 2

/-- Represents the time it takes Roque to walk to work one way (in hours) -/
def walk_time : ℕ := 2

/-- Represents the time it takes Roque to bike to work one way (in hours) -/
def bike_time : ℕ := 1

/-- Represents the number of times Roque walks to and from work per week -/
def walk_trips : ℕ := 3

/-- Represents the total time Roque spends commuting per week (in hours) -/
def total_commute_time : ℕ := 16

theorem roque_bike_trips :
  walk_trips * (2 * walk_time) + bike_trips * (2 * bike_time) = total_commute_time :=
by sorry

end NUMINAMATH_CALUDE_roque_bike_trips_l539_53999


namespace NUMINAMATH_CALUDE_segment_division_sum_l539_53990

/-- Given a line segment AB with A = (1, 1) and B = (x, y), and a point C = (2, 4) that divides AB in the ratio 2:1, prove that x + y = 8 -/
theorem segment_division_sum (x y : ℝ) : 
  let A : ℝ × ℝ := (1, 1)
  let B : ℝ × ℝ := (x, y)
  let C : ℝ × ℝ := (2, 4)
  (C.1 - A.1) / (B.1 - C.1) = 2 ∧ 
  (C.2 - A.2) / (B.2 - C.2) = 2 →
  x + y = 8 := by
sorry

end NUMINAMATH_CALUDE_segment_division_sum_l539_53990


namespace NUMINAMATH_CALUDE_sum_a_t_equals_41_l539_53912

theorem sum_a_t_equals_41 (a t : ℝ) (ha : a > 0) (ht : t > 0) 
  (h : Real.sqrt (6 + a / t) = 6 * Real.sqrt (a / t)) : a + t = 41 :=
sorry

end NUMINAMATH_CALUDE_sum_a_t_equals_41_l539_53912


namespace NUMINAMATH_CALUDE_exam_students_count_l539_53916

theorem exam_students_count (N : ℕ) (T : ℕ) : 
  T = 88 * N ∧ 
  T - 8 * 50 = 92 * (N - 8) ∧ 
  T - 8 * 50 - 100 = 92 * (N - 9) → 
  N = 84 := by
sorry

end NUMINAMATH_CALUDE_exam_students_count_l539_53916


namespace NUMINAMATH_CALUDE_common_chord_equation_l539_53931

/-- The equation of the line where the common chord of two circles lies -/
theorem common_chord_equation (r : ℝ) (h : r > 0) :
  ∃ (ρ θ : ℝ), (ρ = r ∨ ρ = -2 * r * Real.sin (θ + π/4)) →
  Real.sqrt 2 * ρ * (Real.sin θ + Real.cos θ) = -r :=
sorry

end NUMINAMATH_CALUDE_common_chord_equation_l539_53931


namespace NUMINAMATH_CALUDE_fran_average_speed_l539_53974

/-- Proves that given Joann's average speed and time, and Fran's riding time,
    Fran's required average speed to travel the same distance as Joann is 14 mph. -/
theorem fran_average_speed 
  (joann_speed : ℝ) 
  (joann_time : ℝ) 
  (fran_time : ℝ) 
  (h1 : joann_speed = 16) 
  (h2 : joann_time = 3.5) 
  (h3 : fran_time = 4) : 
  (joann_speed * joann_time) / fran_time = 14 := by
  sorry

end NUMINAMATH_CALUDE_fran_average_speed_l539_53974


namespace NUMINAMATH_CALUDE_train_crossing_time_l539_53935

/-- Calculates the time taken for a train to cross a platform -/
theorem train_crossing_time (train_length : ℝ) (signal_pole_time : ℝ) (platform_length : ℝ) :
  train_length = 300 →
  signal_pole_time = 24 →
  platform_length = 187.5 →
  (train_length + platform_length) / (train_length / signal_pole_time) = 39 :=
by sorry

end NUMINAMATH_CALUDE_train_crossing_time_l539_53935


namespace NUMINAMATH_CALUDE_curve_sum_invariant_under_translation_l539_53981

-- Define a type for points in a plane
variable (P : Type) [AddCommGroup P]

-- Define a type for convex curves in the plane
variable (Curve : Type) [AddCommGroup Curve]

-- Define a parallel translation operation
variable (T : P → P)

-- Define an operation to apply translation to a curve
variable (applyTranslation : (P → P) → Curve → Curve)

-- Define a sum operation for curves
variable (curveSum : Curve → Curve → Curve)

-- Define a congruence relation for curves
variable (congruent : Curve → Curve → Prop)

-- Statement of the theorem
theorem curve_sum_invariant_under_translation 
  (K₁ K₂ : Curve) :
  congruent (curveSum K₁ K₂) (curveSum (applyTranslation T K₁) (applyTranslation T K₂)) :=
sorry

end NUMINAMATH_CALUDE_curve_sum_invariant_under_translation_l539_53981


namespace NUMINAMATH_CALUDE_expected_value_twelve_sided_die_l539_53917

/-- A twelve-sided die with faces numbered from 1 to 12 -/
structure TwelveSidedDie :=
  (faces : Finset ℕ)
  (face_count : faces.card = 12)
  (face_range : ∀ n, n ∈ faces ↔ 1 ≤ n ∧ n ≤ 12)

/-- The expected value of a roll of a twelve-sided die -/
def expected_value (d : TwelveSidedDie) : ℚ :=
  (d.faces.sum id) / 12

/-- Theorem: The expected value of a roll of a twelve-sided die with faces numbered from 1 to 12 is 6.5 -/
theorem expected_value_twelve_sided_die :
  ∀ d : TwelveSidedDie, expected_value d = 13/2 :=
sorry

end NUMINAMATH_CALUDE_expected_value_twelve_sided_die_l539_53917


namespace NUMINAMATH_CALUDE_grandmothers_gift_amount_l539_53986

theorem grandmothers_gift_amount (num_grandchildren : ℕ) (cards_per_year : ℕ) (money_per_card : ℕ) : 
  num_grandchildren = 3 → cards_per_year = 2 → money_per_card = 80 →
  num_grandchildren * cards_per_year * money_per_card = 480 := by
  sorry

end NUMINAMATH_CALUDE_grandmothers_gift_amount_l539_53986


namespace NUMINAMATH_CALUDE_tan_point_zero_l539_53932

theorem tan_point_zero (φ : ℝ) : 
  (fun x => Real.tan (x + φ)) (π / 3) = 0 → φ = -π / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_point_zero_l539_53932


namespace NUMINAMATH_CALUDE_petty_cash_for_support_staff_bonus_l539_53956

/-- Represents the number of staff members in each category -/
structure StaffCount where
  total : Nat
  administrative : Nat
  junior : Nat
  support : Nat

/-- Represents the daily bonus amounts for each staff category -/
structure DailyBonus where
  administrative : Nat
  junior : Nat
  support : Nat

/-- Represents the financial details of the bonus distribution -/
structure BonusDistribution where
  staff : StaffCount
  daily_bonus : DailyBonus
  bonus_days : Nat
  accountant_amount : Nat
  petty_cash_budget : Nat

/-- Calculates the amount needed from petty cash for support staff bonuses -/
def petty_cash_needed (bd : BonusDistribution) : Nat :=
  let total_bonus := bd.staff.administrative * bd.daily_bonus.administrative * bd.bonus_days +
                     bd.staff.junior * bd.daily_bonus.junior * bd.bonus_days +
                     bd.staff.support * bd.daily_bonus.support * bd.bonus_days
  total_bonus - bd.accountant_amount

/-- Theorem stating the amount needed from petty cash for support staff bonuses -/
theorem petty_cash_for_support_staff_bonus 
  (bd : BonusDistribution) 
  (h1 : bd.staff.total = 30)
  (h2 : bd.staff.administrative = 10)
  (h3 : bd.staff.junior = 10)
  (h4 : bd.staff.support = 10)
  (h5 : bd.daily_bonus.administrative = 100)
  (h6 : bd.daily_bonus.junior = 120)
  (h7 : bd.daily_bonus.support = 80)
  (h8 : bd.bonus_days = 30)
  (h9 : bd.accountant_amount = 85000)
  (h10 : bd.petty_cash_budget = 25000) :
  petty_cash_needed bd = 5000 := by
  sorry

end NUMINAMATH_CALUDE_petty_cash_for_support_staff_bonus_l539_53956


namespace NUMINAMATH_CALUDE_max_crates_first_trip_solution_l539_53946

/-- The maximum number of crates that can be carried in the first part of the trip -/
def max_crates_first_trip (total_crates : ℕ) (min_crate_weight : ℕ) (max_trip_weight : ℕ) : ℕ :=
  min (total_crates) (max_trip_weight / min_crate_weight)

theorem max_crates_first_trip_solution :
  max_crates_first_trip 12 120 600 = 5 := by
  sorry

end NUMINAMATH_CALUDE_max_crates_first_trip_solution_l539_53946


namespace NUMINAMATH_CALUDE_trader_gain_percentage_l539_53937

/-- Represents a trader selling pens -/
structure PenTrader where
  sold : ℕ
  gainInPens : ℕ

/-- Calculates the gain percentage for a pen trader -/
def gainPercentage (trader : PenTrader) : ℚ :=
  (trader.gainInPens : ℚ) / (trader.sold : ℚ) * 100

/-- Theorem stating that for a trader selling 250 pens and gaining the cost of 65 pens, 
    the gain percentage is 26% -/
theorem trader_gain_percentage :
  ∀ (trader : PenTrader), 
    trader.sold = 250 → 
    trader.gainInPens = 65 → 
    gainPercentage trader = 26 := by
  sorry

end NUMINAMATH_CALUDE_trader_gain_percentage_l539_53937


namespace NUMINAMATH_CALUDE_car_gasoline_usage_l539_53949

/-- Calculates the amount of gasoline used by a car given its efficiency, speed, and travel time. -/
def gasoline_used (efficiency : Real) (speed : Real) (time : Real) : Real :=
  efficiency * speed * time

theorem car_gasoline_usage :
  let efficiency : Real := 0.14  -- liters per kilometer
  let speed : Real := 93.6       -- kilometers per hour
  let time : Real := 2.5         -- hours
  gasoline_used efficiency speed time = 32.76 := by
  sorry

end NUMINAMATH_CALUDE_car_gasoline_usage_l539_53949


namespace NUMINAMATH_CALUDE_lcm_problem_l539_53918

theorem lcm_problem (a b : ℕ+) : 
  (Nat.gcd a b = 1) → 
  (∃ (a_max b_max a_min b_min : ℕ+), 
    (a_max - b_max) - (a_min - b_min) = 38 ∧ 
    ∀ (x y : ℕ+), (x - y) ≤ (a_max - b_max) ∧ (x - y) ≥ (a_min - b_min)) → 
  Nat.lcm a b = 40 := by
sorry

end NUMINAMATH_CALUDE_lcm_problem_l539_53918


namespace NUMINAMATH_CALUDE_net_income_for_130_tax_l539_53927

/-- Calculates the net income after tax given a pre-tax income -/
def net_income (pre_tax_income : ℝ) : ℝ :=
  pre_tax_income - ((pre_tax_income - 800) * 0.2)

/-- Theorem stating that for a pre-tax income resulting in 130 yuan tax, the net income is 1320 yuan -/
theorem net_income_for_130_tax :
  ∃ (pre_tax_income : ℝ),
    (pre_tax_income - 800) * 0.2 = 130 ∧
    net_income pre_tax_income = 1320 :=
by sorry

end NUMINAMATH_CALUDE_net_income_for_130_tax_l539_53927


namespace NUMINAMATH_CALUDE_day_relationship_l539_53921

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a specific day in a year -/
structure DayInYear where
  year : ℤ
  dayNumber : ℕ

/-- Function to determine the day of the week for a given day in a year -/
def dayOfWeek (d : DayInYear) : DayOfWeek := sorry

/-- Theorem stating the relationship between specific days and their days of the week -/
theorem day_relationship (M : ℤ) :
  (dayOfWeek ⟨M, 250⟩ = DayOfWeek.Friday) →
  (dayOfWeek ⟨M + 1, 150⟩ = DayOfWeek.Friday) →
  (dayOfWeek ⟨M - 1, 50⟩ = DayOfWeek.Wednesday) := by
  sorry

end NUMINAMATH_CALUDE_day_relationship_l539_53921


namespace NUMINAMATH_CALUDE_cody_tickets_l539_53948

def arcade_tickets (initial_tickets spent_tickets additional_tickets : ℕ) : ℕ :=
  initial_tickets - spent_tickets + additional_tickets

theorem cody_tickets : arcade_tickets 49 25 6 = 30 := by
  sorry

end NUMINAMATH_CALUDE_cody_tickets_l539_53948


namespace NUMINAMATH_CALUDE_congruence_solutions_count_l539_53955

theorem congruence_solutions_count : 
  (Finset.filter (fun x : ℕ => 
    x > 0 ∧ x < 150 ∧ (x + 20) % 45 = 75 % 45) 
    (Finset.range 150)).card = 4 := by sorry

end NUMINAMATH_CALUDE_congruence_solutions_count_l539_53955


namespace NUMINAMATH_CALUDE_circle_and_tangent_line_l539_53938

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/16 + y^2 = 1

-- Define the circle ⊙G
def circle_G (x y r : ℝ) : Prop := (x-2)^2 + y^2 = r^2

-- Define the incircle property
def is_incircle (r : ℝ) : Prop :=
  ∃ (A B C : ℝ × ℝ),
    ellipse A.1 A.2 ∧ ellipse B.1 B.2 ∧ ellipse C.1 C.2 ∧
    circle_G 2 0 r ∧
    A.1 = -4 -- Left vertex of ellipse

-- Define the tangent line EF
def line_EF (m b : ℝ) (x y : ℝ) : Prop := y = m*x + b

-- Define the tangency condition
def is_tangent (m b r : ℝ) : Prop :=
  abs (m*2 - b) / Real.sqrt (1 + m^2) = r

-- State the theorem
theorem circle_and_tangent_line :
  ∀ r : ℝ,
  is_incircle r →
  r = 2/3 ∧
  ∃ m b : ℝ,
    line_EF m b 0 1 ∧  -- Line passes through M(0,1)
    is_tangent m b r   -- Line is tangent to ⊙G
:= by sorry

end NUMINAMATH_CALUDE_circle_and_tangent_line_l539_53938


namespace NUMINAMATH_CALUDE_min_value_expression_l539_53911

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + 4*a + 2) * (b^2 + 4*b + 2) * (c^2 + 4*c + 2) / (a * b * c) ≥ 216 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l539_53911


namespace NUMINAMATH_CALUDE_apple_percentage_after_adding_oranges_l539_53995

/-- Given a basket with apples and oranges, calculate the percentage of apples after adding more oranges. -/
theorem apple_percentage_after_adding_oranges 
  (initial_apples initial_oranges added_oranges : ℕ) : 
  initial_apples = 10 → 
  initial_oranges = 5 → 
  added_oranges = 5 → 
  (initial_apples : ℚ) / (initial_apples + initial_oranges + added_oranges) * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_apple_percentage_after_adding_oranges_l539_53995


namespace NUMINAMATH_CALUDE_linear_combination_harmonic_l539_53903

/-- A function is harmonic if its value at each point is the average of its values at the four neighboring points. -/
def IsHarmonic (f : ℤ → ℤ → ℝ) : Prop :=
  ∀ x y : ℤ, f x y = (f (x + 1) y + f (x - 1) y + f x (y + 1) + f x (y - 1)) / 4

/-- The theorem states that a linear combination of two harmonic functions is also harmonic. -/
theorem linear_combination_harmonic
    (f g : ℤ → ℤ → ℝ) (hf : IsHarmonic f) (hg : IsHarmonic g) (a b : ℝ) :
    IsHarmonic (fun x y ↦ a * f x y + b * g x y) := by
  sorry

end NUMINAMATH_CALUDE_linear_combination_harmonic_l539_53903


namespace NUMINAMATH_CALUDE_quadratic_function_difference_l539_53993

/-- A quadratic function with the property g(x+1) - g(x) = 2x + 3 for all real x -/
def g_property (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (x + 1) - g x = 2 * x + 3

theorem quadratic_function_difference (g : ℝ → ℝ) (h : g_property g) : 
  g 2 - g 6 = -40 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_difference_l539_53993


namespace NUMINAMATH_CALUDE_sin_2theta_value_l539_53919

theorem sin_2theta_value (θ : Real) (h : Real.tan θ + 1 / Real.tan θ = 2) : 
  Real.sin (2 * θ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_2theta_value_l539_53919


namespace NUMINAMATH_CALUDE_remainder_problem_l539_53989

theorem remainder_problem (n : ℕ) : n % 13 = 11 → n = 349 → n % 17 = 9 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l539_53989


namespace NUMINAMATH_CALUDE_tan_plus_reciprocal_l539_53901

theorem tan_plus_reciprocal (α : Real) : 
  Real.tan α + (Real.tan α)⁻¹ = (Real.sin α * Real.cos α)⁻¹ :=
by sorry

end NUMINAMATH_CALUDE_tan_plus_reciprocal_l539_53901


namespace NUMINAMATH_CALUDE_max_value_expression_l539_53998

theorem max_value_expression (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) 
  (h_sum : a^2 + b^2 + c^2 = 3) : 
  3 * a * b * Real.sqrt 2 + 9 * b * c ≤ 3 * Real.sqrt 11 := by
  sorry

end NUMINAMATH_CALUDE_max_value_expression_l539_53998


namespace NUMINAMATH_CALUDE_baseball_card_value_decrease_l539_53909

theorem baseball_card_value_decrease (x : ℝ) :
  (1 - x / 100) * (1 - 10 / 100) = 1 - 28 / 100 →
  x = 20 := by sorry

end NUMINAMATH_CALUDE_baseball_card_value_decrease_l539_53909


namespace NUMINAMATH_CALUDE_paint_usage_l539_53979

theorem paint_usage (total_paint : ℝ) (first_week_fraction : ℝ) (second_week_fraction : ℝ) 
  (h1 : total_paint = 360)
  (h2 : first_week_fraction = 1/4)
  (h3 : second_week_fraction = 1/6) :
  let first_week_usage := first_week_fraction * total_paint
  let remaining_paint := total_paint - first_week_usage
  let second_week_usage := second_week_fraction * remaining_paint
  first_week_usage + second_week_usage = 135 := by
sorry

end NUMINAMATH_CALUDE_paint_usage_l539_53979


namespace NUMINAMATH_CALUDE_minimum_value_theorem_l539_53942

theorem minimum_value_theorem (x y : ℝ) (h1 : 0 < x) (h2 : x < 1) (h3 : 0 < y) (h4 : y < 1) (h5 : x * y = 1/2) :
  (2 / (1 - x) + 1 / (1 - y)) ≥ 10 := by
sorry

end NUMINAMATH_CALUDE_minimum_value_theorem_l539_53942


namespace NUMINAMATH_CALUDE_smallest_a_in_special_progression_l539_53994

theorem smallest_a_in_special_progression (a b c : ℤ) 
  (h1 : a < c ∧ c < b)
  (h2 : 2 * c = a + b)  -- arithmetic progression condition
  (h3 : b * b = a * c)  -- geometric progression condition
  : a ≥ -4 ∧ ∃ (a₀ b₀ c₀ : ℤ), a₀ = -4 ∧ b₀ = 2 ∧ c₀ = -1 ∧ 
    a₀ < c₀ ∧ c₀ < b₀ ∧ 
    2 * c₀ = a₀ + b₀ ∧ 
    b₀ * b₀ = a₀ * c₀ :=
by sorry

end NUMINAMATH_CALUDE_smallest_a_in_special_progression_l539_53994


namespace NUMINAMATH_CALUDE_problem_statement_l539_53987

theorem problem_statement (t : ℝ) (x y : ℝ) 
  (h1 : x = 3 - 2*t) 
  (h2 : y = 5*t + 6) 
  (h3 : x = 1) : 
  y = 11 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l539_53987


namespace NUMINAMATH_CALUDE_fraction_sum_inequality_l539_53965

theorem fraction_sum_inequality (a b : ℝ) (h : a * b ≠ 0) :
  (a * b > 0 → b / a + a / b ≥ 2) ∧
  (a * b < 0 → |b / a + a / b| ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_inequality_l539_53965


namespace NUMINAMATH_CALUDE_vector_problem_l539_53929

/-- Two-dimensional vector -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Dot product of two 2D vectors -/
def dot (v w : Vector2D) : ℝ := v.x * w.x + v.y * w.y

/-- Vectors are perpendicular if their dot product is zero -/
def perpendicular (v w : Vector2D) : Prop := dot v w = 0

/-- The angle between two vectors is acute if their dot product is positive -/
def acuteAngle (v w : Vector2D) : Prop := dot v w > 0

theorem vector_problem (x : ℝ) : 
  let a : Vector2D := ⟨1, 2⟩
  let b : Vector2D := ⟨x, 1⟩
  (acuteAngle a b ↔ x > -2 ∧ x ≠ 1/2) ∧ 
  (perpendicular (Vector2D.mk (1 + 2*x) 4) (Vector2D.mk (2 - x) 3) ↔ x = 7/2) := by
  sorry

end NUMINAMATH_CALUDE_vector_problem_l539_53929


namespace NUMINAMATH_CALUDE_kim_trip_time_kim_trip_time_is_120_l539_53930

/-- The total time Kim spends away from home given the described trip conditions -/
theorem kim_trip_time : ℝ :=
  let distance_to_friend : ℝ := 30
  let detour_percentage : ℝ := 0.2
  let time_at_friends : ℝ := 30
  let speed : ℝ := 44
  let distance_back : ℝ := distance_to_friend * (1 + detour_percentage)
  let total_distance : ℝ := distance_to_friend + distance_back
  let driving_time : ℝ := total_distance / speed * 60
  driving_time + time_at_friends

theorem kim_trip_time_is_120 : kim_trip_time = 120 := by
  sorry

end NUMINAMATH_CALUDE_kim_trip_time_kim_trip_time_is_120_l539_53930


namespace NUMINAMATH_CALUDE_prime_gap_2015_l539_53970

theorem prime_gap_2015 : ∃ p q : ℕ, 
  Prime p ∧ Prime q ∧ p < q ∧ q - p > 2015 ∧ 
  ∀ k : ℕ, p < k ∧ k < q → ¬(Prime k) :=
sorry

end NUMINAMATH_CALUDE_prime_gap_2015_l539_53970


namespace NUMINAMATH_CALUDE_tan_triangle_identity_l539_53905

theorem tan_triangle_identity (A B C : Real) (h₁ : 0 < A) (h₂ : 0 < B) (h₃ : 0 < C) 
  (h₄ : A + B + C = Real.pi) : 
  (Real.tan A * Real.tan B * Real.tan C) / (Real.tan A + Real.tan B + Real.tan C) = 1 :=
by sorry

end NUMINAMATH_CALUDE_tan_triangle_identity_l539_53905


namespace NUMINAMATH_CALUDE_min_value_fraction_l539_53914

theorem min_value_fraction (x : ℝ) (h : x > 8) : 
  x^2 / (x - 8)^2 ≥ 1 ∧ ∀ ε > 0, ∃ y > 8, y^2 / (y - 8)^2 < 1 + ε :=
sorry

end NUMINAMATH_CALUDE_min_value_fraction_l539_53914


namespace NUMINAMATH_CALUDE_sum_a_d_equals_negative_one_l539_53902

theorem sum_a_d_equals_negative_one
  (a b c d : ℤ)
  (eq1 : a + b = 11)
  (eq2 : b + c = 9)
  (eq3 : c + d = 3) :
  a + d = -1 := by sorry

end NUMINAMATH_CALUDE_sum_a_d_equals_negative_one_l539_53902


namespace NUMINAMATH_CALUDE_book_arrangement_l539_53933

theorem book_arrangement (n : ℕ) (k : ℕ) (h1 : n = 7) (h2 : k = 3) :
  (n! / k!) = 840 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_l539_53933


namespace NUMINAMATH_CALUDE_cone_sphere_volume_ratio_l539_53940

/-- Theorem: Ratio of cone height to base radius when cone volume is one-third of sphere volume -/
theorem cone_sphere_volume_ratio (r h : ℝ) (hr : r > 0) : 
  (1 / 3 : ℝ) * ((4 / 3 : ℝ) * Real.pi * r^3) = (1 / 3 : ℝ) * Real.pi * r^2 * h → h / r = 4 / 3 :=
by sorry

end NUMINAMATH_CALUDE_cone_sphere_volume_ratio_l539_53940


namespace NUMINAMATH_CALUDE_sqrt_x_minus_5_real_implies_x_geq_5_l539_53972

theorem sqrt_x_minus_5_real_implies_x_geq_5 (x : ℝ) : 
  (∃ y : ℝ, y^2 = x - 5) → x ≥ 5 := by sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_5_real_implies_x_geq_5_l539_53972


namespace NUMINAMATH_CALUDE_max_semicircle_intersections_l539_53985

/-- Given n distinct points on a line, the maximum number of intersection points
    of semicircles drawn on one side of the line with these points as endpoints
    is equal to (n choose 4). -/
theorem max_semicircle_intersections (n : ℕ) : ℕ :=
  Nat.choose n 4

#check max_semicircle_intersections

end NUMINAMATH_CALUDE_max_semicircle_intersections_l539_53985


namespace NUMINAMATH_CALUDE_selling_price_calculation_l539_53950

/-- Given an article with a gain of $15 and a gain percentage of 20%,
    prove that the selling price is $90. -/
theorem selling_price_calculation (gain : ℝ) (gain_percentage : ℝ) :
  gain = 15 →
  gain_percentage = 20 →
  ∃ (cost_price selling_price : ℝ),
    gain = (gain_percentage / 100) * cost_price ∧
    selling_price = cost_price + gain ∧
    selling_price = 90 := by
  sorry

end NUMINAMATH_CALUDE_selling_price_calculation_l539_53950


namespace NUMINAMATH_CALUDE_solution_implies_m_equals_three_l539_53983

/-- Given that x = 2 and y = 1 is a solution to the equation x + my = 5, prove that m = 3 -/
theorem solution_implies_m_equals_three (x y m : ℝ) 
  (h1 : x = 2) 
  (h2 : y = 1) 
  (h3 : x + m * y = 5) : 
  m = 3 := by
  sorry

end NUMINAMATH_CALUDE_solution_implies_m_equals_three_l539_53983


namespace NUMINAMATH_CALUDE_pure_imaginary_square_l539_53996

theorem pure_imaginary_square (a : ℝ) : 
  (Complex.I * ((1 : ℂ) + a * Complex.I)^2).re = 0 → a = 1 ∨ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_square_l539_53996


namespace NUMINAMATH_CALUDE_cube_sum_minus_triple_product_l539_53982

theorem cube_sum_minus_triple_product (p : ℕ) : 
  Prime p → 
  ({(x, y) : ℕ × ℕ | x^3 + y^3 - 3*x*y = p - 1} = 
    if p = 2 then {(1, 0), (0, 1)} 
    else if p = 5 then {(2, 2)} 
    else ∅) := by
sorry

end NUMINAMATH_CALUDE_cube_sum_minus_triple_product_l539_53982


namespace NUMINAMATH_CALUDE_dave_book_cost_l539_53988

/-- The cost per book given the total number of books and total amount spent -/
def cost_per_book (total_books : ℕ) (total_spent : ℚ) : ℚ :=
  total_spent / total_books

theorem dave_book_cost :
  let total_books : ℕ := 8 + 6 + 3
  let total_spent : ℚ := 102
  cost_per_book total_books total_spent = 6 := by
  sorry

end NUMINAMATH_CALUDE_dave_book_cost_l539_53988


namespace NUMINAMATH_CALUDE_nathan_tokens_l539_53939

/-- Calculates the total number of tokens used by Nathan at the arcade -/
def total_tokens (air_hockey_games basketball_games skee_ball_games shooting_games racing_games : ℕ)
  (air_hockey_cost basketball_cost skee_ball_cost shooting_cost racing_cost : ℕ) : ℕ :=
  air_hockey_games * air_hockey_cost +
  basketball_games * basketball_cost +
  skee_ball_games * skee_ball_cost +
  shooting_games * shooting_cost +
  racing_games * racing_cost

theorem nathan_tokens :
  total_tokens 7 12 9 6 5 6 8 4 7 5 = 241 := by
  sorry

end NUMINAMATH_CALUDE_nathan_tokens_l539_53939


namespace NUMINAMATH_CALUDE_john_school_year_hours_l539_53904

/-- Calculates the required working hours per week during school year -/
def school_year_hours_per_week (summer_weeks : ℕ) (summer_hours_per_week : ℕ) (summer_earnings : ℕ) 
  (school_year_weeks : ℕ) (school_year_target : ℕ) : ℚ :=
  let hourly_wage : ℚ := summer_earnings / (summer_weeks * summer_hours_per_week)
  let total_school_year_hours : ℚ := school_year_target / hourly_wage
  total_school_year_hours / school_year_weeks

theorem john_school_year_hours (summer_weeks : ℕ) (summer_hours_per_week : ℕ) (summer_earnings : ℕ) 
  (school_year_weeks : ℕ) (school_year_target : ℕ) 
  (h1 : summer_weeks = 8) 
  (h2 : summer_hours_per_week = 40) 
  (h3 : summer_earnings = 4000) 
  (h4 : school_year_weeks = 25) 
  (h5 : school_year_target = 5000) :
  school_year_hours_per_week summer_weeks summer_hours_per_week summer_earnings school_year_weeks school_year_target = 16 := by
  sorry

end NUMINAMATH_CALUDE_john_school_year_hours_l539_53904


namespace NUMINAMATH_CALUDE_twitter_to_insta_fb_ratio_l539_53975

/-- Represents the number of followers on each social media platform -/
structure Followers where
  instagram : ℕ
  facebook : ℕ
  twitter : ℕ
  tiktok : ℕ
  youtube : ℕ

/-- Conditions for Malcolm's social media followers -/
def malcolm_followers (f : Followers) : Prop :=
  f.instagram = 240 ∧
  f.facebook = 500 ∧
  f.tiktok = 3 * f.twitter ∧
  f.youtube = f.tiktok + 510 ∧
  f.instagram + f.facebook + f.twitter + f.tiktok + f.youtube = 3840

/-- Theorem stating the ratio of Twitter followers to Instagram and Facebook followers -/
theorem twitter_to_insta_fb_ratio (f : Followers) 
  (h : malcolm_followers f) : 
  f.twitter * 2 = f.instagram + f.facebook := by
  sorry

end NUMINAMATH_CALUDE_twitter_to_insta_fb_ratio_l539_53975


namespace NUMINAMATH_CALUDE_triangle_median_and_altitude_l539_53923

/-- Triangle ABC with given vertices -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Definition of median -/
def isMedian (t : Triangle) (l : LineEquation) : Prop :=
  -- The line passes through vertex B and the midpoint of AC
  sorry

/-- Definition of altitude -/
def isAltitude (t : Triangle) (l : LineEquation) : Prop :=
  -- The line passes through vertex A and is perpendicular to BC
  sorry

/-- Main theorem -/
theorem triangle_median_and_altitude (t : Triangle) 
    (h1 : t.A = (-5, 0))
    (h2 : t.B = (4, -4))
    (h3 : t.C = (0, 2)) :
    ∃ (median altitude : LineEquation),
      isMedian t median ∧
      isAltitude t altitude ∧
      median = LineEquation.mk 1 7 5 ∧
      altitude = LineEquation.mk 2 (-3) 10 := by
  sorry


end NUMINAMATH_CALUDE_triangle_median_and_altitude_l539_53923


namespace NUMINAMATH_CALUDE_inequality_proof_l539_53992

theorem inequality_proof (x y z : ℝ) 
  (hx : x > 1) (hy : y > 1) (hz : z > 1)
  (h : 1/x + 1/y + 1/z = 2) : 
  Real.sqrt (x + y + z) ≥ Real.sqrt (x - 1) + Real.sqrt (y - 1) + Real.sqrt (z - 1) := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l539_53992


namespace NUMINAMATH_CALUDE_sin_product_identity_l539_53964

theorem sin_product_identity :
  Real.sin (12 * π / 180) * Real.sin (48 * π / 180) * Real.sin (72 * π / 180) * Real.sin (84 * π / 180) =
  (1 / 8) * (1 + Real.cos (24 * π / 180)) := by
sorry

end NUMINAMATH_CALUDE_sin_product_identity_l539_53964


namespace NUMINAMATH_CALUDE_rational_function_value_l539_53957

/-- A rational function with specific properties -/
structure RationalFunction where
  p : ℝ → ℝ
  q : ℝ → ℝ
  linear_p : ∃ (a b : ℝ), ∀ x, p x = a * x + b
  quadratic_q : ∃ (a b c : ℝ), ∀ x, q x = a * x^2 + b * x + c
  asymptote_neg4 : q (-4) = 0
  asymptote_1 : q 1 = 0
  point_0 : p 0 / q 0 = -1
  point_1 : p 1 / q 1 = -2

/-- The main theorem -/
theorem rational_function_value (f : RationalFunction) : f.p 0 / f.q 0 = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_rational_function_value_l539_53957


namespace NUMINAMATH_CALUDE_binomial_square_constant_l539_53973

theorem binomial_square_constant (c : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 + 84*x + c = (x + a)^2) → c = 1764 := by
  sorry

end NUMINAMATH_CALUDE_binomial_square_constant_l539_53973


namespace NUMINAMATH_CALUDE_product_97_103_l539_53980

theorem product_97_103 : 97 * 103 = 9991 := by
  sorry

end NUMINAMATH_CALUDE_product_97_103_l539_53980


namespace NUMINAMATH_CALUDE_rectangular_box_with_spheres_l539_53969

theorem rectangular_box_with_spheres (h : ℝ) : 
  let box_base : ℝ := 4
  let large_sphere_radius : ℝ := 2
  let small_sphere_radius : ℝ := 1
  let num_small_spheres : ℕ := 8
  h > 0 ∧ 
  box_base > 0 ∧
  large_sphere_radius > 0 ∧
  small_sphere_radius > 0 ∧
  num_small_spheres > 0 ∧
  (∃ (box : Set (ℝ × ℝ × ℝ)) (large_sphere : Set (ℝ × ℝ × ℝ)) (small_spheres : Finset (Set (ℝ × ℝ × ℝ))),
    -- Box properties
    (∀ (x y z : ℝ), (x, y, z) ∈ box ↔ 0 ≤ x ∧ x ≤ box_base ∧ 0 ≤ y ∧ y ≤ box_base ∧ 0 ≤ z ∧ z ≤ h) ∧
    -- Large sphere properties
    (∃ (cx cy cz : ℝ), large_sphere = {(x, y, z) | (x - cx)^2 + (y - cy)^2 + (z - cz)^2 ≤ large_sphere_radius^2}) ∧
    -- Small spheres properties
    (small_spheres.card = num_small_spheres) ∧
    (∀ s ∈ small_spheres, ∃ (cx cy cz : ℝ), s = {(x, y, z) | (x - cx)^2 + (y - cy)^2 + (z - cz)^2 ≤ small_sphere_radius^2}) ∧
    -- Tangency conditions
    (∀ s ∈ small_spheres, ∃ (face1 face2 face3 : Set (ℝ × ℝ × ℝ)), face1 ∪ face2 ∪ face3 ⊆ box ∧ s ∩ face1 ≠ ∅ ∧ s ∩ face2 ≠ ∅ ∧ s ∩ face3 ≠ ∅) ∧
    (∀ s ∈ small_spheres, large_sphere ∩ s ≠ ∅)) →
  h = 2 + 2 * Real.sqrt 7 := by
sorry

end NUMINAMATH_CALUDE_rectangular_box_with_spheres_l539_53969


namespace NUMINAMATH_CALUDE_chairs_per_row_l539_53922

theorem chairs_per_row (total_rows : ℕ) (occupied_seats : ℕ) (unoccupied_seats : ℕ) :
  total_rows = 40 →
  occupied_seats = 790 →
  unoccupied_seats = 10 →
  (occupied_seats + unoccupied_seats) / total_rows = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_chairs_per_row_l539_53922


namespace NUMINAMATH_CALUDE_max_x_minus_y_on_circle_l539_53966

theorem max_x_minus_y_on_circle (x y : ℝ) (h : x^2 + y^2 - 4*x - 2*y - 4 = 0) :
  ∃ (z : ℝ), z = x - y ∧ z ≤ 1 + 3 * Real.sqrt 2 ∧
  ∀ (w : ℝ), (∃ (a b : ℝ), a^2 + b^2 - 4*a - 2*b - 4 = 0 ∧ w = a - b) → w ≤ 1 + 3 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_max_x_minus_y_on_circle_l539_53966


namespace NUMINAMATH_CALUDE_rectangle_max_area_l539_53968

theorem rectangle_max_area (l w : ℕ) : 
  (2 * l + 2 * w = 40) → 
  (∀ a b : ℕ, 2 * a + 2 * b = 40 → l * w ≥ a * b) →
  l * w = 100 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_max_area_l539_53968


namespace NUMINAMATH_CALUDE_doughnuts_per_person_l539_53925

/-- The number of doughnuts each person receives when Samuel and Cathy share their doughnuts with friends -/
theorem doughnuts_per_person :
  ∀ (samuel_dozens cathy_dozens num_friends : ℕ),
  samuel_dozens = 2 →
  cathy_dozens = 3 →
  num_friends = 8 →
  (samuel_dozens * 12 + cathy_dozens * 12) / (num_friends + 2) = 6 :=
by sorry

end NUMINAMATH_CALUDE_doughnuts_per_person_l539_53925


namespace NUMINAMATH_CALUDE_jennie_drive_time_l539_53963

def drive_time_proof (distance : ℝ) (time_with_traffic : ℝ) (speed_difference : ℝ) : Prop :=
  let speed_with_traffic := distance / time_with_traffic
  let speed_no_traffic := speed_with_traffic + speed_difference
  let time_no_traffic := distance / speed_no_traffic
  distance = 200 ∧ time_with_traffic = 5 ∧ speed_difference = 10 →
  time_no_traffic = 4

theorem jennie_drive_time : drive_time_proof 200 5 10 := by
  sorry

end NUMINAMATH_CALUDE_jennie_drive_time_l539_53963


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l539_53991

theorem sufficient_not_necessary (a : ℝ) :
  (∀ a, a > 2 → a^2 > 2*a) ∧
  (∃ a, a^2 > 2*a ∧ a ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l539_53991


namespace NUMINAMATH_CALUDE_second_candidate_votes_l539_53944

theorem second_candidate_votes (total_votes : ℕ) (first_candidate_percentage : ℚ) : 
  total_votes = 600 → 
  first_candidate_percentage = 60 / 100 → 
  (total_votes : ℚ) * (1 - first_candidate_percentage) = 240 := by
  sorry

end NUMINAMATH_CALUDE_second_candidate_votes_l539_53944


namespace NUMINAMATH_CALUDE_final_ring_count_is_225_l539_53926

/-- Calculates the final number of ornamental rings in the store after a series of transactions -/
def final_ring_count (initial_purchase : ℕ) (additional_purchase : ℕ) (final_sale : ℕ) : ℕ :=
  let initial_stock := initial_purchase / 2
  let total_stock := initial_purchase + initial_stock
  let remaining_after_first_sale := total_stock - (3 * total_stock / 4)
  let stock_after_additional_purchase := remaining_after_first_sale + additional_purchase
  stock_after_additional_purchase - final_sale

/-- The final number of ornamental rings in the store is 225 -/
theorem final_ring_count_is_225 : final_ring_count 200 300 150 = 225 := by
  sorry

end NUMINAMATH_CALUDE_final_ring_count_is_225_l539_53926


namespace NUMINAMATH_CALUDE_angle_conversion_l539_53900

theorem angle_conversion (π : ℝ) :
  (12 : ℝ) * (π / 180) = π / 15 :=
by sorry

end NUMINAMATH_CALUDE_angle_conversion_l539_53900


namespace NUMINAMATH_CALUDE_binomial_expansion_sum_l539_53959

theorem binomial_expansion_sum (a : ℝ) (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (a - x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₂ = 80 →
  a₀ + a₁ + a₂ + a₃ + a₄ + a₅ = 1 :=
by sorry

end NUMINAMATH_CALUDE_binomial_expansion_sum_l539_53959


namespace NUMINAMATH_CALUDE_special_sequence_lower_bound_l539_53920

/-- A sequence of n consecutive natural numbers with special divisor properties -/
structure SpecialSequence (n : ℕ) :=
  (original : Fin n → ℕ)
  (divisors : Fin n → ℕ)
  (original_ascending : ∀ i j, i < j → original i < original j)
  (divisors_ascending : ∀ i j, i < j → divisors i < divisors j)
  (divisor_property : ∀ i, 1 < divisors i ∧ divisors i < original i ∧ divisors i ∣ original i)

/-- All prime numbers smaller than n -/
def primes_less_than (n : ℕ) : Finset ℕ :=
  (Finset.range n).filter Nat.Prime

/-- The main theorem -/
theorem special_sequence_lower_bound (n : ℕ) (seq : SpecialSequence n) :
  ∀ i, seq.original i > (n ^ (primes_less_than n).card) / (primes_less_than n).prod id :=
sorry

end NUMINAMATH_CALUDE_special_sequence_lower_bound_l539_53920


namespace NUMINAMATH_CALUDE_max_xy_value_l539_53961

theorem max_xy_value (x y : ℕ+) (h : 7 * x + 4 * y = 150) : x * y ≤ 200 := by
  sorry

end NUMINAMATH_CALUDE_max_xy_value_l539_53961


namespace NUMINAMATH_CALUDE_store_price_reduction_l539_53976

theorem store_price_reduction (original_price : ℝ) (h1 : original_price > 0) :
  let first_reduction := 0.09
  let final_price_ratio := 0.819
  let price_after_first := original_price * (1 - first_reduction)
  let second_reduction := 1 - (final_price_ratio / (1 - first_reduction))
  second_reduction = 0.181 := by sorry

end NUMINAMATH_CALUDE_store_price_reduction_l539_53976


namespace NUMINAMATH_CALUDE_triangle_angle_sum_l539_53971

theorem triangle_angle_sum (A B C : ℝ) (h1 : A + B = 115) : 
  A + B + C = 180 → C = 65 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_sum_l539_53971


namespace NUMINAMATH_CALUDE_no_unique_solution_l539_53978

/-- The system of equations does not have a unique solution if and only if k = 3 -/
theorem no_unique_solution (k : ℝ) : 
  (∃ (x y : ℝ), 4 * (3 * x + 4 * y) = 48 ∧ k * x + 12 * y = 30) ∧ 
  ¬(∃! (x y : ℝ), 4 * (3 * x + 4 * y) = 48 ∧ k * x + 12 * y = 30) ↔ 
  k = 3 := by
sorry

end NUMINAMATH_CALUDE_no_unique_solution_l539_53978


namespace NUMINAMATH_CALUDE_single_root_condition_l539_53958

theorem single_root_condition (n : ℕ) (a : ℝ) (h : n > 1) :
  (∃! x : ℝ, (1 + x)^(1/n : ℝ) + (1 - x)^(1/n : ℝ) = a) ↔ a = 2 := by sorry

end NUMINAMATH_CALUDE_single_root_condition_l539_53958


namespace NUMINAMATH_CALUDE_function_through_points_l539_53997

/-- Given a function f(x) = a^x - k passing through (1,3) and (0,2), prove f(x) = 2^x + 1 -/
theorem function_through_points (a k : ℝ) (f : ℝ → ℝ) 
    (h1 : ∀ x, f x = a^x - k) 
    (h2 : f 1 = 3) 
    (h3 : f 0 = 2) : 
    ∀ x, f x = 2^x + 1 := by
  sorry

end NUMINAMATH_CALUDE_function_through_points_l539_53997


namespace NUMINAMATH_CALUDE_choose_three_from_nine_l539_53962

theorem choose_three_from_nine : Nat.choose 9 3 = 84 := by
  sorry

end NUMINAMATH_CALUDE_choose_three_from_nine_l539_53962


namespace NUMINAMATH_CALUDE_tan_sqrt_two_identity_l539_53907

theorem tan_sqrt_two_identity (α : Real) (h : Real.tan α = Real.sqrt 2) :
  1 + Real.sin (2 * α) + (Real.cos α)^2 = (4 + Real.sqrt 2) / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_sqrt_two_identity_l539_53907


namespace NUMINAMATH_CALUDE_tangent_line_x_intercept_l539_53913

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + 4*x + 5

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 + 4

theorem tangent_line_x_intercept :
  let slope := f' 1
  let point := (1, f 1)
  let m := slope
  let b := point.2 - m * point.1
  (0 - b) / m = -3/7 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_x_intercept_l539_53913


namespace NUMINAMATH_CALUDE_curve_tangent_perpendicular_l539_53953

-- Define the curve
def curve (a : ℝ) (x : ℝ) : ℝ := a * x^2 - a * x + 1

-- Define the tangent line
def tangent_slope (a : ℝ) : ℝ := -a

-- Define the perpendicular line
def perp_line (x y : ℝ) : Prop := 2 * x + y + 10 = 0

-- State the theorem
theorem curve_tangent_perpendicular (a : ℝ) (h : a ≠ 0) :
  (tangent_slope a * 2 = -1) → a = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_curve_tangent_perpendicular_l539_53953


namespace NUMINAMATH_CALUDE_total_expenses_calculation_l539_53954

-- Define the initial conditions
def initial_price : ℝ := 1.4
def daily_price_decrease : ℝ := 0.1
def first_purchase : ℝ := 10
def second_purchase : ℝ := 25
def total_trip_distance : ℝ := 320
def distance_before_friday : ℝ := 200
def fuel_efficiency : ℝ := 8

-- Define the theorem
theorem total_expenses_calculation :
  let friday_price := initial_price - 4 * daily_price_decrease
  let cost_monday := first_purchase * initial_price
  let cost_friday := second_purchase * friday_price
  let total_cost_35_liters := cost_monday + cost_friday
  let remaining_distance := total_trip_distance - distance_before_friday
  let additional_liters := remaining_distance / fuel_efficiency
  let cost_additional_liters := additional_liters * friday_price
  let total_expenses := total_cost_35_liters + cost_additional_liters
  total_expenses = 54 := by sorry

end NUMINAMATH_CALUDE_total_expenses_calculation_l539_53954


namespace NUMINAMATH_CALUDE_solve_equation_l539_53984

theorem solve_equation : ∃ x : ℝ, (0.5^3 - 0.1^3 / 0.5^2 + x + 0.1^2 = 0.4) ∧ (x = 0.269) := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l539_53984


namespace NUMINAMATH_CALUDE_min_speed_to_arrive_earlier_l539_53951

/-- Proves the minimum speed required for the second person to arrive before the first person --/
theorem min_speed_to_arrive_earlier (distance : ℝ) (speed_A : ℝ) (delay : ℝ) :
  distance = 120 →
  speed_A = 30 →
  delay = 1.5 →
  ∀ speed_B : ℝ, speed_B > 48 → 
    distance / speed_B < distance / speed_A - delay := by
  sorry

end NUMINAMATH_CALUDE_min_speed_to_arrive_earlier_l539_53951


namespace NUMINAMATH_CALUDE_sum_of_products_bounds_l539_53934

/-- Represents a table of -1s and 1s -/
def Table (n : ℕ) := Fin n → Fin n → Int

/-- Defines the valid entries for the table -/
def validEntry (x : Int) : Prop := x = 1 ∨ x = -1

/-- Defines a valid table where all entries are either 1 or -1 -/
def validTable (A : Table n) : Prop :=
  ∀ i j, validEntry (A i j)

/-- Product of elements in a row -/
def rowProduct (A : Table n) (i : Fin n) : Int :=
  (Finset.univ.prod fun j => A i j)

/-- Product of elements in a column -/
def colProduct (A : Table n) (j : Fin n) : Int :=
  (Finset.univ.prod fun i => A i j)

/-- Sum of products S for a given table -/
def sumOfProducts (A : Table n) : Int :=
  (Finset.univ.sum fun i => rowProduct A i) + (Finset.univ.sum fun j => colProduct A j)

/-- Theorem stating that the sum of products is always even and bounded -/
theorem sum_of_products_bounds (n : ℕ) (A : Table n) (h : validTable A) :
  ∃ k : Int, sumOfProducts A = 2 * k ∧ -n ≤ k ∧ k ≤ n :=
sorry

end NUMINAMATH_CALUDE_sum_of_products_bounds_l539_53934


namespace NUMINAMATH_CALUDE_average_mark_is_76_l539_53943

def marks : List ℝ := [80, 70, 60, 90, 80]

theorem average_mark_is_76 : (marks.sum / marks.length : ℝ) = 76 := by
  sorry

end NUMINAMATH_CALUDE_average_mark_is_76_l539_53943


namespace NUMINAMATH_CALUDE_circular_arrangement_theorem_l539_53915

/-- Represents a circular arrangement of people -/
structure CircularArrangement where
  n : ℕ  -- Total number of people
  dist : ℕ → ℕ → ℕ  -- Distance function between two positions

/-- The main theorem -/
theorem circular_arrangement_theorem (c : CircularArrangement) :
  (c.dist 31 7 = c.dist 31 14) → c.n = 41 := by
  sorry

/-- Helper function to calculate clockwise distance -/
def clockwise_distance (n : ℕ) (a b : ℕ) : ℕ :=
  if b ≥ a then b - a else n - a + b

/-- Axiom: The distance function in CircularArrangement is defined by clockwise_distance -/
axiom distance_defined (c : CircularArrangement) :
  ∀ a b, c.dist a b = clockwise_distance c.n a b

/-- Axiom: The arrangement is circular, so the distance from a to b equals the distance from b to a -/
axiom circular_symmetry (c : CircularArrangement) :
  ∀ a b, c.dist a b = c.dist b a

end NUMINAMATH_CALUDE_circular_arrangement_theorem_l539_53915


namespace NUMINAMATH_CALUDE_range_of_a_l539_53910

def p (x : ℝ) : Prop := |4*x - 3| ≤ 1

def q (x a : ℝ) : Prop := x^2 - (2*a + 1)*x + a*(a + 1) ≤ 0

theorem range_of_a :
  (∀ x a : ℝ, ¬(p x) → ¬(q x a)) ∧
  (∃ x a : ℝ, ¬(q x a) ∧ p x) →
  ∀ a : ℝ, (0 ≤ a ∧ a ≤ 1/2) ↔ (∀ x : ℝ, p x → q x a) ∧ (∃ x : ℝ, q x a ∧ ¬(p x)) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l539_53910


namespace NUMINAMATH_CALUDE_eighth_grade_ratio_l539_53941

theorem eighth_grade_ratio (total_students : Nat) (girls : Nat) :
  total_students = 68 →
  girls = 28 →
  (total_students - girls : Nat) / girls = 10 / 7 := by
  sorry

end NUMINAMATH_CALUDE_eighth_grade_ratio_l539_53941


namespace NUMINAMATH_CALUDE_water_bottles_cost_l539_53952

/-- The total cost of water bottles given the number of bottles, liters per bottle, and price per liter. -/
def total_cost (num_bottles : ℕ) (liters_per_bottle : ℕ) (price_per_liter : ℕ) : ℕ :=
  num_bottles * liters_per_bottle * price_per_liter

/-- Theorem stating that the total cost of six 2-liter bottles of water is $12 when the price is $1 per liter. -/
theorem water_bottles_cost :
  total_cost 6 2 1 = 12 := by
  sorry

end NUMINAMATH_CALUDE_water_bottles_cost_l539_53952


namespace NUMINAMATH_CALUDE_unique_function_satisfying_inequality_l539_53945

theorem unique_function_satisfying_inequality (a c d : ℝ) :
  ∃! f : ℝ → ℝ, ∀ x : ℝ, f (a * x + c) + d ≤ x ∧ x ≤ f (x + d) + c :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_function_satisfying_inequality_l539_53945


namespace NUMINAMATH_CALUDE_min_value_fraction_l539_53928

theorem min_value_fraction (x : ℝ) (h : x > 9) : 
  x^2 / (x - 9) ≥ 36 ∧ ∃ y > 9, y^2 / (y - 9) = 36 := by
sorry

end NUMINAMATH_CALUDE_min_value_fraction_l539_53928


namespace NUMINAMATH_CALUDE_gcd_n_power_7_minus_n_l539_53967

theorem gcd_n_power_7_minus_n (n : ℤ) : 42 ∣ (n^7 - n) := by
  sorry

end NUMINAMATH_CALUDE_gcd_n_power_7_minus_n_l539_53967


namespace NUMINAMATH_CALUDE_base_3_12021_equals_142_l539_53908

def base_3_to_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3^i)) 0

theorem base_3_12021_equals_142 :
  base_3_to_10 [1, 2, 0, 2, 1] = 142 := by
  sorry

end NUMINAMATH_CALUDE_base_3_12021_equals_142_l539_53908


namespace NUMINAMATH_CALUDE_number_of_triceratopses_l539_53936

/-- Represents the number of rhinoceroses -/
def r : ℕ := sorry

/-- Represents the number of triceratopses -/
def t : ℕ := sorry

/-- The total number of horns -/
def total_horns : ℕ := 31

/-- The total number of legs -/
def total_legs : ℕ := 48

/-- Theorem stating that the number of triceratopses is 7 -/
theorem number_of_triceratopses : t = 7 := by
  sorry

end NUMINAMATH_CALUDE_number_of_triceratopses_l539_53936


namespace NUMINAMATH_CALUDE_lawrence_county_camp_attendance_l539_53960

/-- The number of kids from Lawrence county who go to camp -/
def kids_at_camp (total : ℕ) (stay_home : ℕ) : ℕ :=
  total - stay_home

/-- Proof that 610769 kids from Lawrence county go to camp -/
theorem lawrence_county_camp_attendance :
  kids_at_camp 1201565 590796 = 610769 := by
  sorry

end NUMINAMATH_CALUDE_lawrence_county_camp_attendance_l539_53960


namespace NUMINAMATH_CALUDE_no_cracked_seashells_l539_53977

theorem no_cracked_seashells 
  (tom_initial : ℕ) 
  (fred_initial : ℕ) 
  (fred_more_than_tom : ℕ) 
  (h1 : tom_initial = 15)
  (h2 : fred_initial = 43)
  (h3 : fred_more_than_tom = 28)
  : ∃ (tom_final fred_final : ℕ),
    tom_initial + fred_initial = tom_final + fred_final ∧
    fred_final = tom_final + fred_more_than_tom ∧
    tom_initial - tom_final = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_no_cracked_seashells_l539_53977


namespace NUMINAMATH_CALUDE_cube_with_holes_volume_l539_53947

/-- The volume of a cube with holes drilled through it -/
theorem cube_with_holes_volume :
  let cube_edge : ℝ := 3
  let hole_side : ℝ := 1
  let cube_volume := cube_edge ^ 3
  let hole_volume := hole_side ^ 2 * cube_edge
  let num_hole_pairs := 3
  cube_volume - (num_hole_pairs * hole_volume) = 18 := by
  sorry

end NUMINAMATH_CALUDE_cube_with_holes_volume_l539_53947
