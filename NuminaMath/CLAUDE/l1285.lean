import Mathlib

namespace NUMINAMATH_CALUDE_min_cost_57_and_227_l1285_128591

/-- Calculates the minimum cost for notebooks given the pricing structure and number of notebooks -/
def min_cost (n : ℕ) : ℚ :=
  let single_price := 0.3
  let dozen_price := 3.0
  let bulk_dozen_price := 2.7
  let dozens := n / 12
  let singles := n % 12
  if dozens > 10 then
    bulk_dozen_price * dozens + single_price * singles
  else if singles = 0 then
    dozen_price * dozens
  else
    min (dozen_price * (dozens + 1)) (dozen_price * dozens + single_price * singles)

theorem min_cost_57_and_227 :
  min_cost 57 = 14.7 ∧ min_cost 227 = 51.3 := by sorry

end NUMINAMATH_CALUDE_min_cost_57_and_227_l1285_128591


namespace NUMINAMATH_CALUDE_power_tower_mod_500_l1285_128566

theorem power_tower_mod_500 : 5^(5^(5^5)) ≡ 125 [ZMOD 500] := by sorry

end NUMINAMATH_CALUDE_power_tower_mod_500_l1285_128566


namespace NUMINAMATH_CALUDE_ball_hitting_ground_time_l1285_128567

theorem ball_hitting_ground_time :
  let height (t : ℝ) := -16 * t^2 + 16 * t + 50
  ∃ t : ℝ, t > 0 ∧ height t = 0 ∧ t = (2 + 3 * Real.sqrt 6) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ball_hitting_ground_time_l1285_128567


namespace NUMINAMATH_CALUDE_parabola_vertex_sum_max_l1285_128549

theorem parabola_vertex_sum_max (a T : ℤ) (h_T : T ≠ 0) : 
  let parabola (x : ℝ) := a * x * (x - 2 * T)
  let N := T - a * T^2
  (parabola 0 = 0) → 
  (parabola (2 * T) = 0) → 
  (parabola (T + 2) = 36) → 
  (∀ (a' T' : ℤ), T' ≠ 0 → 
    let parabola' (x : ℝ) := a' * x * (x - 2 * T')
    let N' := T' - a' * T'^2
    (parabola' 0 = 0) → 
    (parabola' (2 * T') = 0) → 
    (parabola' (T' + 2) = 36) → 
    N ≥ N') → 
  N = 37 :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_sum_max_l1285_128549


namespace NUMINAMATH_CALUDE_no_prime_solution_l1285_128542

/-- Converts a number from base p to decimal --/
def to_decimal (digits : List Nat) (p : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * p^i) 0

/-- The equation that p must satisfy --/
def equation (p : Nat) : Prop :=
  to_decimal [9, 0, 0, 1] p + to_decimal [7, 0, 3] p + 
  to_decimal [5, 1, 1] p + to_decimal [6, 2, 1] p + 
  to_decimal [7] p = 
  to_decimal [3, 4, 1] p + to_decimal [4, 7, 2] p + 
  to_decimal [1, 6, 3] p

theorem no_prime_solution : ¬∃ p : Nat, Nat.Prime p ∧ equation p := by
  sorry

end NUMINAMATH_CALUDE_no_prime_solution_l1285_128542


namespace NUMINAMATH_CALUDE_nested_sqrt_fifteen_l1285_128594

theorem nested_sqrt_fifteen (x : ℝ) : x = Real.sqrt (15 + x) → x = (1 + Real.sqrt 61) / 2 := by
  sorry

end NUMINAMATH_CALUDE_nested_sqrt_fifteen_l1285_128594


namespace NUMINAMATH_CALUDE_graduation_day_after_85_days_l1285_128501

/-- Days of the week represented as integers mod 7 -/
inductive DayOfWeek : Type
| monday : DayOfWeek
| tuesday : DayOfWeek
| wednesday : DayOfWeek
| thursday : DayOfWeek
| friday : DayOfWeek
| saturday : DayOfWeek
| sunday : DayOfWeek

/-- Function to add days to a given day of the week -/
def addDays (start : DayOfWeek) (days : Nat) : DayOfWeek :=
  match (days % 7) with
  | 0 => start
  | 1 => match start with
    | DayOfWeek.monday => DayOfWeek.tuesday
    | DayOfWeek.tuesday => DayOfWeek.wednesday
    | DayOfWeek.wednesday => DayOfWeek.thursday
    | DayOfWeek.thursday => DayOfWeek.friday
    | DayOfWeek.friday => DayOfWeek.saturday
    | DayOfWeek.saturday => DayOfWeek.sunday
    | DayOfWeek.sunday => DayOfWeek.monday
  | _ => sorry -- Other cases omitted for brevity

theorem graduation_day_after_85_days : 
  addDays DayOfWeek.monday 85 = DayOfWeek.tuesday :=
by sorry


end NUMINAMATH_CALUDE_graduation_day_after_85_days_l1285_128501


namespace NUMINAMATH_CALUDE_tom_payment_l1285_128563

/-- The total amount Tom paid for apples and mangoes -/
def total_amount (apple_quantity : ℕ) (apple_rate : ℕ) (mango_quantity : ℕ) (mango_rate : ℕ) : ℕ :=
  apple_quantity * apple_rate + mango_quantity * mango_rate

/-- Theorem stating that Tom paid 965 for his purchase -/
theorem tom_payment : total_amount 8 70 9 45 = 965 := by
  sorry

end NUMINAMATH_CALUDE_tom_payment_l1285_128563


namespace NUMINAMATH_CALUDE_injective_function_inequality_l1285_128589

theorem injective_function_inequality (f : ℕ → ℕ) 
  (h_inj : Function.Injective f) 
  (h_ineq : ∀ n : ℕ, f (f n) ≤ (n + f n) / 2) : 
  ∀ n : ℕ, f n = n := by
  sorry

end NUMINAMATH_CALUDE_injective_function_inequality_l1285_128589


namespace NUMINAMATH_CALUDE_white_copy_cost_is_five_cents_l1285_128574

/- Define the problem parameters -/
def total_copies : ℕ := 400
def colored_copies : ℕ := 50
def colored_cost : ℚ := 10 / 100  -- 10 cents in dollars
def total_bill : ℚ := 225 / 10    -- $22.50

/- Define the cost of a white copy -/
def white_copy_cost : ℚ := (total_bill - colored_copies * colored_cost) / (total_copies - colored_copies)

/- Theorem statement -/
theorem white_copy_cost_is_five_cents : white_copy_cost = 5 / 100 := by
  sorry

end NUMINAMATH_CALUDE_white_copy_cost_is_five_cents_l1285_128574


namespace NUMINAMATH_CALUDE_car_travel_time_l1285_128522

theorem car_travel_time (speed_A speed_B : ℝ) (time_A : ℝ) (ratio : ℝ) 
  (h1 : speed_A = 50)
  (h2 : speed_B = 25)
  (h3 : time_A = 8)
  (h4 : ratio = 4)
  (h5 : speed_A > 0)
  (h6 : speed_B > 0)
  (h7 : time_A > 0)
  (h8 : ratio > 0) :
  (speed_A * time_A) / (speed_B * ((speed_A * time_A) / (ratio * speed_B))) = 4 := by
  sorry

#check car_travel_time

end NUMINAMATH_CALUDE_car_travel_time_l1285_128522


namespace NUMINAMATH_CALUDE_crayon_selection_l1285_128517

theorem crayon_selection (n k : ℕ) (h1 : n = 15) (h2 : k = 5) :
  Nat.choose n k = 3003 := by
  sorry

end NUMINAMATH_CALUDE_crayon_selection_l1285_128517


namespace NUMINAMATH_CALUDE_cos_value_given_sin_l1285_128508

theorem cos_value_given_sin (θ : ℝ) (h : Real.sin (θ - π/6) = Real.sqrt 3 / 3) :
  Real.cos (π/3 - 2*θ) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_cos_value_given_sin_l1285_128508


namespace NUMINAMATH_CALUDE_a_plus_b_value_l1285_128586

-- Define the functions f and h
def f (a b x : ℝ) : ℝ := a * x + b
def h (x : ℝ) : ℝ := 3 * x + 1

-- State the theorem
theorem a_plus_b_value (a b : ℝ) : 
  (∀ x, h (f a b x) = 5 * x - 8) → a + b = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_a_plus_b_value_l1285_128586


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l1285_128514

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, x ≠ 0 → f (-x) = f x

theorem solution_set_of_inequality
  (f : ℝ → ℝ) (f' : ℝ → ℝ)
  (h_even : is_even_function f)
  (h_deriv : ∀ x > 0, x * f' x > -2 * f x)
  (g : ℝ → ℝ) (h_g : ∀ x, g x = x^2 * f x) :
  {x : ℝ | g x < g (1 - x)} = {x : ℝ | x < 0 ∨ (0 < x ∧ x < 1/2)} :=
sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l1285_128514


namespace NUMINAMATH_CALUDE_defective_clock_correct_time_fraction_l1285_128582

/-- Represents a 12-hour digital clock with a defect that displays '7' instead of '1'. -/
structure DefectiveClock where
  /-- The number of hours in the clock cycle -/
  hours : Nat
  /-- The number of minutes in an hour -/
  minutes_per_hour : Nat
  /-- The number of hours displayed correctly -/
  correct_hours : Nat
  /-- The number of minutes displayed correctly in each hour -/
  correct_minutes : Nat

/-- The fraction of the day that the defective clock displays the correct time -/
def correct_time_fraction (clock : DefectiveClock) : ℚ :=
  (clock.correct_hours : ℚ) / clock.hours * (clock.correct_minutes : ℚ) / clock.minutes_per_hour

/-- Theorem stating that the fraction of the day the defective clock displays the correct time is 1/2 -/
theorem defective_clock_correct_time_fraction :
  ∃ (clock : DefectiveClock),
    clock.hours = 12 ∧
    clock.minutes_per_hour = 60 ∧
    clock.correct_hours = 8 ∧
    clock.correct_minutes = 45 ∧
    correct_time_fraction clock = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_defective_clock_correct_time_fraction_l1285_128582


namespace NUMINAMATH_CALUDE_coin_collection_dimes_l1285_128597

def penny : ℕ := 1
def nickel : ℕ := 5
def dime : ℕ := 10
def quarter : ℕ := 25
def half_dollar : ℕ := 50

theorem coin_collection_dimes :
  ∀ (p n d q h : ℕ),
    p ≥ 1 → n ≥ 1 → d ≥ 1 → q ≥ 1 → h ≥ 1 →
    p + n + d + q + h = 12 →
    p * penny + n * nickel + d * dime + q * quarter + h * half_dollar = 163 →
    d = 5 := by
  sorry

end NUMINAMATH_CALUDE_coin_collection_dimes_l1285_128597


namespace NUMINAMATH_CALUDE_polynomial_gp_roots_condition_l1285_128518

/-- A polynomial with coefficients a, b, and c -/
def polynomial (a b c x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

/-- Condition for three distinct real roots in geometric progression -/
def has_three_distinct_real_roots_in_gp (a b c : ℝ) : Prop :=
  ∃ x y z : ℝ, 
    x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    polynomial a b c x = 0 ∧
    polynomial a b c y = 0 ∧
    polynomial a b c z = 0 ∧
    ∃ r : ℝ, r ≠ 0 ∧ y = x * r ∧ z = y * r

/-- Theorem stating the conditions on coefficients a, b, and c -/
theorem polynomial_gp_roots_condition (a b c : ℝ) :
  has_three_distinct_real_roots_in_gp a b c ↔ 
    a^3 * c = b^3 ∧ -a^2 < b ∧ b < a^2 / 3 :=
sorry

end NUMINAMATH_CALUDE_polynomial_gp_roots_condition_l1285_128518


namespace NUMINAMATH_CALUDE_four_circle_plus_two_l1285_128502

-- Define the operation ⊕
def circle_plus (a b : ℝ) : ℝ := 4 * a + 5 * b

-- State the theorem
theorem four_circle_plus_two : circle_plus 4 2 = 26 := by
  sorry

end NUMINAMATH_CALUDE_four_circle_plus_two_l1285_128502


namespace NUMINAMATH_CALUDE_parallelogram_area_l1285_128595

/-- The area of a parallelogram with a diagonal of length 30 meters and an altitude of 20 meters to that diagonal is 600 square meters. -/
theorem parallelogram_area (d : ℝ) (h : ℝ) (h1 : d = 30) (h2 : h = 20) :
  d * h = 600 := by sorry

end NUMINAMATH_CALUDE_parallelogram_area_l1285_128595


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l1285_128584

-- Problem 1
theorem problem_1 : 42.67 - (12.67 - 2.87) = 32.87 := by sorry

-- Problem 2
theorem problem_2 : (4.8 - 4.8 * (3.2 - 2.7)) / 0.24 = 10 := by sorry

-- Problem 3
theorem problem_3 : 4.31 * 0.57 + 0.43 * 4.31 - 4.31 = 0 := by sorry

-- Problem 4
theorem problem_4 : 9.99 * 222 + 3.33 * 334 = 3330 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l1285_128584


namespace NUMINAMATH_CALUDE_black_chicken_daytime_theorem_l1285_128544

/-- The number of spots in the daytime program -/
def daytime_spots : ℕ := 2

/-- The number of spots in the evening program -/
def evening_spots : ℕ := 3

/-- The total number of spots available -/
def total_spots : ℕ := daytime_spots + evening_spots

/-- The number of black chickens applying -/
def black_chickens : ℕ := 3

/-- The number of white chickens applying -/
def white_chickens : ℕ := 1

/-- The total number of chickens applying -/
def total_chickens : ℕ := black_chickens + white_chickens

/-- The probability of a chicken choosing the daytime program when both are available -/
def daytime_probability : ℚ := 1/2

/-- The probability that at least one black chicken is admitted to the daytime program -/
def black_chicken_daytime_probability : ℚ := 63/64

theorem black_chicken_daytime_theorem :
  (total_spots = daytime_spots + evening_spots) →
  (total_chickens = black_chickens + white_chickens) →
  (total_chickens ≤ total_spots) →
  (daytime_probability = 1/2) →
  black_chicken_daytime_probability = 63/64 := by
  sorry

end NUMINAMATH_CALUDE_black_chicken_daytime_theorem_l1285_128544


namespace NUMINAMATH_CALUDE_only_two_special_triples_l1285_128550

/-- A structure representing a triple of positive integers (a, b, c) satisfying certain conditions. -/
structure SpecialTriple where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  h1 : a ≥ b
  h2 : b ≥ c
  h3 : ∃ x : ℕ, a^2 + 3*b = x^2
  h4 : ∃ y : ℕ, b^2 + 3*c = y^2
  h5 : ∃ z : ℕ, c^2 + 3*a = z^2

/-- The theorem stating that there are only two SpecialTriples. -/
theorem only_two_special_triples :
  {t : SpecialTriple | t.a = 1 ∧ t.b = 1 ∧ t.c = 1} ∪
  {t : SpecialTriple | t.a = 37 ∧ t.b = 25 ∧ t.c = 17} =
  {t : SpecialTriple | True} :=
sorry

end NUMINAMATH_CALUDE_only_two_special_triples_l1285_128550


namespace NUMINAMATH_CALUDE_greatest_multiple_of_four_l1285_128576

theorem greatest_multiple_of_four (y : ℕ) : 
  y > 0 ∧ 
  ∃ k : ℕ, y = 4 * k ∧ 
  y^3 < 4096 →
  y ≤ 12 ∧ 
  ∀ z : ℕ, z > 0 ∧ (∃ m : ℕ, z = 4 * m) ∧ z^3 < 4096 → z ≤ y :=
by sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_four_l1285_128576


namespace NUMINAMATH_CALUDE_polynomial_equality_l1285_128557

theorem polynomial_equality (a b c m n : ℝ) : 
  (∀ x : ℝ, m * x^2 - n * x + 3 = a * (x - 1)^2 + b * (x - 1) + c) →
  a - b + c = 3 := by
sorry

end NUMINAMATH_CALUDE_polynomial_equality_l1285_128557


namespace NUMINAMATH_CALUDE_cube_sum_product_l1285_128571

theorem cube_sum_product : ∃ (a b : ℤ), a^3 + b^3 = 91 ∧ a * b = 12 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_product_l1285_128571


namespace NUMINAMATH_CALUDE_missing_number_equation_l1285_128533

theorem missing_number_equation (x : ℝ) : 11 + Real.sqrt (-4 + 6 * x / 3) = 13 ↔ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_missing_number_equation_l1285_128533


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l1285_128510

def M : Set ℝ := {x | x^2 > 4}
def N : Set ℝ := {x | 1 < x ∧ x ≤ 3}

theorem intersection_complement_equality :
  N ∩ (Set.univ \ M) = {x : ℝ | 1 < x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l1285_128510


namespace NUMINAMATH_CALUDE_work_completion_time_l1285_128546

theorem work_completion_time (a b : ℕ) (h1 : a + b = 1/12) (h2 : a = 1/20) : b = 1/30 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l1285_128546


namespace NUMINAMATH_CALUDE_hallie_earnings_l1285_128539

/-- Calculates the total earnings for a waitress over three days given her hourly wage, hours worked, and tips for each day. -/
def total_earnings (hourly_wage : ℝ) (hours_day1 hours_day2 hours_day3 : ℝ) (tips_day1 tips_day2 tips_day3 : ℝ) : ℝ :=
  (hourly_wage * hours_day1 + tips_day1) +
  (hourly_wage * hours_day2 + tips_day2) +
  (hourly_wage * hours_day3 + tips_day3)

/-- Theorem stating that Hallie's total earnings over three days equal $240 given her work schedule and tips. -/
theorem hallie_earnings :
  total_earnings 10 7 5 7 18 12 20 = 240 :=
by
  sorry

end NUMINAMATH_CALUDE_hallie_earnings_l1285_128539


namespace NUMINAMATH_CALUDE_closest_integer_to_sqrt3_plus_1_l1285_128521

theorem closest_integer_to_sqrt3_plus_1 : 
  ∃ (n : ℤ), ∀ (m : ℤ), |n - (Real.sqrt 3 + 1)| ≤ |m - (Real.sqrt 3 + 1)| ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_closest_integer_to_sqrt3_plus_1_l1285_128521


namespace NUMINAMATH_CALUDE_thread_length_problem_l1285_128525

theorem thread_length_problem (current_length : ℝ) : 
  current_length + (3/4 * current_length) = 21 → current_length = 12 := by
  sorry

end NUMINAMATH_CALUDE_thread_length_problem_l1285_128525


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_l1285_128540

-- Define propositions p and q
def p (x : ℝ) : Prop := x > 4
def q (x : ℝ) : Prop := 4 < x ∧ x < 10

-- Theorem statement
theorem p_necessary_not_sufficient :
  (∀ x, q x → p x) ∧ (∃ x, p x ∧ ¬q x) :=
sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_l1285_128540


namespace NUMINAMATH_CALUDE_g_positive_f_local_min_iff_l1285_128543

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x - 1) * Real.exp x - (1/3) * a * x^3 - (1/2) * x^2

-- Define the derivative of f
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := x * Real.exp x - a * x^2 - x

-- Define the function g
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := (f' a x) / x

-- Theorem 1: When a > 0, g(a) > 0
theorem g_positive (a : ℝ) (h : a > 0) : g a a > 0 := by sorry

-- Theorem 2: f(x) has a local minimum if and only if a ∈ (0, +∞)
theorem f_local_min_iff (a : ℝ) :
  (∃ x : ℝ, IsLocalMin (f a) x) ↔ a > 0 := by sorry

end NUMINAMATH_CALUDE_g_positive_f_local_min_iff_l1285_128543


namespace NUMINAMATH_CALUDE_smallest_sum_of_factors_l1285_128516

theorem smallest_sum_of_factors (a b c d : ℕ+) 
  (h : a * b * c * d = Nat.factorial 10) : 
  a + b + c + d ≥ 175 := by
  sorry

end NUMINAMATH_CALUDE_smallest_sum_of_factors_l1285_128516


namespace NUMINAMATH_CALUDE_pure_imaginary_product_l1285_128532

theorem pure_imaginary_product (a : ℝ) : 
  (Complex.I : ℂ).im ≠ 0 →
  (Complex.ofReal 1 - Complex.I) * (Complex.ofReal a + Complex.I) ∈ {z : ℂ | z.re = 0 ∧ z.im ≠ 0} → 
  a = -1 := by
sorry

end NUMINAMATH_CALUDE_pure_imaginary_product_l1285_128532


namespace NUMINAMATH_CALUDE_triangle_vector_ratio_l1285_128579

/-- Given a triangle ABC with point E, prove that if AE = 3/4 * AB + 1/4 * AC, 
    then BE = 1/3 * EC -/
theorem triangle_vector_ratio (A B C E : ℝ × ℝ) : 
  (E - A) = 3/4 * (B - A) + 1/4 * (C - A) → 
  (E - B) = 1/3 * (C - E) := by
  sorry

end NUMINAMATH_CALUDE_triangle_vector_ratio_l1285_128579


namespace NUMINAMATH_CALUDE_parabola_circle_triangle_l1285_128570

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A parabola defined by x^2 = 2py -/
def Parabola (p : ℝ) : Set Point :=
  {pt : Point | pt.x^2 = 2 * p * pt.y}

/-- Check if three points form an equilateral triangle -/
def isEquilateralTriangle (a b c : Point) : Prop :=
  (a.x - b.x)^2 + (a.y - b.y)^2 = 
  (b.x - c.x)^2 + (b.y - c.y)^2 ∧
  (b.x - c.x)^2 + (b.y - c.y)^2 = 
  (c.x - a.x)^2 + (c.y - a.y)^2

/-- The origin point -/
def O : Point := ⟨0, 0⟩

/-- The given point M -/
def M : Point := ⟨0, 9⟩

theorem parabola_circle_triangle (p : ℝ) 
  (h_p_pos : p > 0)
  (A : Point)
  (h_A_on_parabola : A ∈ Parabola p)
  (B : Point)
  (h_B_on_parabola : B ∈ Parabola p)
  (h_circle : (A.x - M.x)^2 + (A.y - M.y)^2 = A.x^2 + A.y^2)
  (h_equilateral : isEquilateralTriangle A B O) :
  p = 3/4 := by
  sorry


end NUMINAMATH_CALUDE_parabola_circle_triangle_l1285_128570


namespace NUMINAMATH_CALUDE_profit_ratio_equals_investment_ratio_l1285_128561

/-- The ratio of profits is equal to the ratio of investments -/
theorem profit_ratio_equals_investment_ratio (p q : ℕ) (h : p = 60000 ∧ q = 90000) : 
  (p : ℚ) / q = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_profit_ratio_equals_investment_ratio_l1285_128561


namespace NUMINAMATH_CALUDE_factorization_proof_l1285_128520

theorem factorization_proof (x : ℝ) : 4 * x^2 - 1 = (2*x + 1) * (2*x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l1285_128520


namespace NUMINAMATH_CALUDE_cone_volume_approximation_l1285_128592

theorem cone_volume_approximation (r h : ℝ) (π : ℝ) : 
  (1/3) * π * r^2 * h = (2/75) * (2 * π * r)^2 * h → π = 25/8 := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_approximation_l1285_128592


namespace NUMINAMATH_CALUDE_mean_temperature_l1285_128593

def temperatures : List ℝ := [79, 81, 83, 85, 84, 86, 88, 87, 85, 84]

theorem mean_temperature : (temperatures.sum / temperatures.length) = 84.2 := by
  sorry

end NUMINAMATH_CALUDE_mean_temperature_l1285_128593


namespace NUMINAMATH_CALUDE_square_diff_sum_l1285_128507

theorem square_diff_sum : 19^2 - 17^2 + 15^2 - 13^2 + 11^2 - 9^2 + 7^2 - 5^2 + 3^2 - 1^2 = 200 := by
  sorry

end NUMINAMATH_CALUDE_square_diff_sum_l1285_128507


namespace NUMINAMATH_CALUDE_ellipse_m_range_l1285_128577

/-- An ellipse is represented by the equation x²/(25 - m) + y²/(m + 9) = 1 with foci on the y-axis -/
def is_ellipse_with_y_foci (m : ℝ) : Prop :=
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ b > a ∧ 
  (∀ (x y : ℝ), x^2 / (25 - m) + y^2 / (m + 9) = 1 ↔ x^2 / a^2 + y^2 / b^2 = 1)

/-- The range of m for the given ellipse -/
theorem ellipse_m_range :
  ∀ m : ℝ, is_ellipse_with_y_foci m ↔ 8 < m ∧ m < 25 := by sorry

end NUMINAMATH_CALUDE_ellipse_m_range_l1285_128577


namespace NUMINAMATH_CALUDE_picnic_gender_difference_l1285_128511

theorem picnic_gender_difference (total : ℕ) (men : ℕ) (adult_child_diff : ℕ) 
  (h_total : total = 240)
  (h_men : men = 90)
  (h_adult_child : adult_child_diff = 40) : 
  ∃ (women children : ℕ), 
    men + women + children = total ∧ 
    men + women = children + adult_child_diff ∧ 
    men - women = 40 := by
sorry

end NUMINAMATH_CALUDE_picnic_gender_difference_l1285_128511


namespace NUMINAMATH_CALUDE_roof_dimension_difference_l1285_128513

theorem roof_dimension_difference (width : ℝ) (length : ℝ) : 
  width > 0 →
  length = 4 * width →
  width * length = 768 →
  length - width = 24 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_roof_dimension_difference_l1285_128513


namespace NUMINAMATH_CALUDE_sixtieth_pair_is_five_seven_l1285_128560

/-- Represents a pair of integers in the sequence -/
structure IntPair :=
  (first : ℕ)
  (second : ℕ)

/-- Generates the nth pair in the sequence -/
def generatePair (n : ℕ) : IntPair :=
  sorry

/-- The main theorem stating that the 60th pair is (5,7) -/
theorem sixtieth_pair_is_five_seven : generatePair 60 = IntPair.mk 5 7 := by
  sorry

end NUMINAMATH_CALUDE_sixtieth_pair_is_five_seven_l1285_128560


namespace NUMINAMATH_CALUDE_circle_radius_c_value_l1285_128529

theorem circle_radius_c_value (c : ℝ) : 
  (∀ x y : ℝ, x^2 + 10*x + y^2 + 2*y + c = 0 ↔ (x+5)^2 + (y+1)^2 = 25) → 
  c = 51 := by
sorry

end NUMINAMATH_CALUDE_circle_radius_c_value_l1285_128529


namespace NUMINAMATH_CALUDE_roi_is_25_percent_l1285_128530

/-- Calculates the return on investment (ROI) percentage for an investor given the dividend rate, face value, and purchase price of shares. -/
def calculate_roi (dividend_rate : ℚ) (face_value : ℚ) (purchase_price : ℚ) : ℚ :=
  (dividend_rate * face_value / purchase_price) * 100

/-- Theorem stating that for the given conditions, the ROI is 25%. -/
theorem roi_is_25_percent :
  let dividend_rate : ℚ := 125 / 1000  -- 12.5%
  let face_value : ℚ := 50
  let purchase_price : ℚ := 25
  calculate_roi dividend_rate face_value purchase_price = 25 := by
  sorry

#eval calculate_roi (125/1000) 50 25  -- This should evaluate to 25

end NUMINAMATH_CALUDE_roi_is_25_percent_l1285_128530


namespace NUMINAMATH_CALUDE_tan_3_75_deg_sum_l1285_128572

theorem tan_3_75_deg_sum (a b c d : ℕ+) 
  (h1 : Real.tan (3.75 * π / 180) = (a : ℝ).sqrt - (b : ℝ).sqrt + (c : ℝ).sqrt - d)
  (h2 : a ≥ b) (h3 : b ≥ c) (h4 : c ≥ d) :
  a + b + c + d = 13 := by
  sorry

end NUMINAMATH_CALUDE_tan_3_75_deg_sum_l1285_128572


namespace NUMINAMATH_CALUDE_at_least_one_not_less_than_two_l1285_128587

theorem at_least_one_not_less_than_two
  (a b c : ℝ)
  (ha : a > 0)
  (hb : b > 0)
  (hc : c > 0)
  (sum_eq_three : a + b + c = 3) :
  ¬(a + 1/b < 2 ∧ b + 1/c < 2 ∧ c + 1/a < 2) :=
by sorry

end NUMINAMATH_CALUDE_at_least_one_not_less_than_two_l1285_128587


namespace NUMINAMATH_CALUDE_jellybean_box_theorem_l1285_128554

/-- The number of jellybeans in a box that is three times larger in each dimension
    compared to a box that holds 200 jellybeans -/
theorem jellybean_box_theorem (ella_jellybeans : ℕ) (scale_factor : ℕ) :
  ella_jellybeans = 200 →
  scale_factor = 3 →
  scale_factor ^ 3 * ella_jellybeans = 5400 :=
by sorry

end NUMINAMATH_CALUDE_jellybean_box_theorem_l1285_128554


namespace NUMINAMATH_CALUDE_a_3_equals_negative_10_l1285_128558

def a (n : ℕ) : ℤ := (-1)^n * (n^2 + 1)

theorem a_3_equals_negative_10 : a 3 = -10 := by
  sorry

end NUMINAMATH_CALUDE_a_3_equals_negative_10_l1285_128558


namespace NUMINAMATH_CALUDE_initial_trees_count_l1285_128565

/-- The number of walnut trees to be removed from the park -/
def trees_removed : ℕ := 4

/-- The number of walnut trees remaining after removal -/
def trees_remaining : ℕ := 2

/-- The initial number of walnut trees in the park -/
def initial_trees : ℕ := trees_removed + trees_remaining

theorem initial_trees_count : initial_trees = 6 := by sorry

end NUMINAMATH_CALUDE_initial_trees_count_l1285_128565


namespace NUMINAMATH_CALUDE_painted_cube_theorem_l1285_128575

/-- Represents a cube that has been painted on all sides and then cut into smaller cubes -/
structure PaintedCube where
  edge_length : ℕ
  small_cube_edge : ℕ

/-- Counts the number of small cubes with a given number of painted faces -/
def count_painted_faces (c : PaintedCube) (num_faces : ℕ) : ℕ := sorry

theorem painted_cube_theorem (c : PaintedCube) 
  (h1 : c.edge_length = 5) 
  (h2 : c.small_cube_edge = 1) : 
  (count_painted_faces c 3 = 8) ∧ 
  (count_painted_faces c 2 = 36) ∧ 
  (count_painted_faces c 1 = 54) := by sorry

end NUMINAMATH_CALUDE_painted_cube_theorem_l1285_128575


namespace NUMINAMATH_CALUDE_max_abs_sum_on_circle_l1285_128526

theorem max_abs_sum_on_circle : ∀ x y : ℝ, x^2 + y^2 = 4 → |x| + |y| ≤ 2 * Real.sqrt 2 ∧ ∃ x y : ℝ, x^2 + y^2 = 4 ∧ |x| + |y| = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_max_abs_sum_on_circle_l1285_128526


namespace NUMINAMATH_CALUDE_storm_rainfall_theorem_l1285_128551

/-- Represents the rainfall data for a city over three days -/
structure CityRainfall where
  day1 : ℝ
  day2 : ℝ
  day3 : ℝ

/-- Represents the rainfall data for two cities -/
structure StormData where
  cityA : CityRainfall
  cityB : CityRainfall
  X : ℝ  -- Combined rainfall on day 3
  Y : ℝ  -- Total rainfall over three days

/-- Defines the conditions of the storm and proves the results -/
theorem storm_rainfall_theorem (s : StormData) : 
  s.cityA.day1 = 4 ∧ 
  s.cityA.day2 = 5 * s.cityA.day1 ∧
  s.cityB.day2 = 3 * s.cityA.day1 ∧
  s.cityA.day3 = (s.cityA.day1 + s.cityA.day2) / 2 ∧
  s.cityB.day3 = s.cityB.day1 + s.cityB.day2 - 6 ∧
  s.X = s.cityA.day3 + s.cityB.day3 ∧
  s.Y = s.cityA.day1 + s.cityA.day2 + s.cityA.day3 + s.cityB.day1 + s.cityB.day2 + s.cityB.day3 →
  s.cityA.day3 = 12 ∧
  s.cityB.day3 = s.cityB.day1 + 6 ∧
  s.X = 18 + s.cityB.day1 ∧
  s.Y = 54 + 2 * s.cityB.day1 := by
  sorry


end NUMINAMATH_CALUDE_storm_rainfall_theorem_l1285_128551


namespace NUMINAMATH_CALUDE_sum_of_squares_and_products_l1285_128534

theorem sum_of_squares_and_products (x y z : ℝ) : 
  x ≥ 0 → y ≥ 0 → z ≥ 0 → 
  x^2 + y^2 + z^2 = 75 → 
  x*y + y*z + z*x = 32 → 
  x + y + z = Real.sqrt 139 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_and_products_l1285_128534


namespace NUMINAMATH_CALUDE_catering_company_comparison_l1285_128569

/-- Represents the cost function for a catering company -/
structure CateringCompany where
  basicFee : ℕ
  perPersonFee : ℕ

/-- Calculates the total cost for a given number of people -/
def totalCost (company : CateringCompany) (people : ℕ) : ℕ :=
  company.basicFee + company.perPersonFee * people

/-- The problem statement -/
theorem catering_company_comparison :
  let company1 : CateringCompany := ⟨120, 18⟩
  let company2 : CateringCompany := ⟨250, 15⟩
  ∀ n : ℕ, n < 44 → totalCost company1 n ≤ totalCost company2 n ∧
  totalCost company2 44 < totalCost company1 44 :=
by sorry

end NUMINAMATH_CALUDE_catering_company_comparison_l1285_128569


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_l1285_128509

/-- The asymptote of a hyperbola with specific properties -/
theorem hyperbola_asymptote (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (∃ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1 ∧ 
    (|x + c| - |x - c|) / (2 * c) = 1/3) →
  (∃ (k : ℝ), k = 2 * Real.sqrt 2 ∧ 
    ∀ (x y : ℝ), y = k * x ∨ y = -k * x) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_l1285_128509


namespace NUMINAMATH_CALUDE_units_digit_of_large_power_l1285_128528

theorem units_digit_of_large_power (n : ℕ) : n > 0 → (7^(8^5) : ℕ) % 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_large_power_l1285_128528


namespace NUMINAMATH_CALUDE_bottles_bought_l1285_128524

theorem bottles_bought (initial : ℕ) (drunk : ℕ) (final : ℕ) : 
  initial = 14 → drunk = 8 → final = 51 → final - (initial - drunk) = 45 := by
  sorry

end NUMINAMATH_CALUDE_bottles_bought_l1285_128524


namespace NUMINAMATH_CALUDE_workshop_workers_l1285_128512

theorem workshop_workers (total_avg : ℝ) (tech_count : ℕ) (tech_avg : ℝ) (rest_avg : ℝ)
  (h1 : total_avg = 8000)
  (h2 : tech_count = 7)
  (h3 : tech_avg = 16000)
  (h4 : rest_avg = 6000) :
  ∃ (total_workers : ℕ),
    (total_workers : ℝ) * total_avg = 
      (tech_count : ℝ) * tech_avg + ((total_workers - tech_count) : ℝ) * rest_avg ∧
    total_workers = 35 :=
by sorry

end NUMINAMATH_CALUDE_workshop_workers_l1285_128512


namespace NUMINAMATH_CALUDE_point_not_on_graph_l1285_128541

-- Define the function
def f (x : ℝ) : ℝ := 1 - 2 * x

-- Theorem statement
theorem point_not_on_graph :
  f (-1) ≠ 0 ∧ 
  f 1 = -1 ∧ 
  f 0 = 1 ∧ 
  f (-1/2) = 2 :=
by sorry

end NUMINAMATH_CALUDE_point_not_on_graph_l1285_128541


namespace NUMINAMATH_CALUDE_critical_points_product_bound_l1285_128599

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x * Real.log x - (1/2) * m * x^2 - x

theorem critical_points_product_bound (m : ℝ) (x₁ x₂ : ℝ) :
  x₁ < x₂ →
  (∃ (y : ℝ), y ∈ Set.Icc x₁ x₂ ∧ (deriv (f m)) y = 0) →
  (deriv (f m)) x₁ = 0 →
  (deriv (f m)) x₂ = 0 →
  x₁ * x₂ > Real.exp 2 := by
sorry

end NUMINAMATH_CALUDE_critical_points_product_bound_l1285_128599


namespace NUMINAMATH_CALUDE_unique_solution_l1285_128583

/-- The functional equation that f must satisfy for all x and y -/
def functional_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x - f y) = f (f y) + x * f y + f x - 1

/-- The theorem stating that f(x) = 1 - x²/2 is the unique solution -/
theorem unique_solution :
  ∃! f : ℝ → ℝ, functional_equation f ∧ ∀ x : ℝ, f x = 1 - x^2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l1285_128583


namespace NUMINAMATH_CALUDE_sum_geq_sqrt_three_l1285_128578

theorem sum_geq_sqrt_three (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : a * b + b * c + c * a = 1) : a + b + c ≥ Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_geq_sqrt_three_l1285_128578


namespace NUMINAMATH_CALUDE_jakes_weight_l1285_128515

theorem jakes_weight (jake kendra : ℝ) 
  (h1 : jake - 8 = 2 * kendra) 
  (h2 : jake + kendra = 293) : 
  jake = 198 := by
sorry

end NUMINAMATH_CALUDE_jakes_weight_l1285_128515


namespace NUMINAMATH_CALUDE_evaluate_expression_l1285_128573

theorem evaluate_expression : (2023 - 1984)^2 / 144 = 10 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1285_128573


namespace NUMINAMATH_CALUDE_equation_holds_for_all_x_l1285_128564

theorem equation_holds_for_all_x (m : ℝ) : 
  (∀ x : ℝ, (m - 5) * x = 0) → m = 5 := by
  sorry

end NUMINAMATH_CALUDE_equation_holds_for_all_x_l1285_128564


namespace NUMINAMATH_CALUDE_smallest_a_is_eight_l1285_128588

/-- A function that represents the expression x^4 + a^2 + x^2 --/
def f (a x : ℤ) : ℤ := x^4 + a^2 + x^2

/-- A predicate that checks if a number is composite --/
def is_composite (n : ℤ) : Prop := ∃ (p q : ℤ), p ≠ 1 ∧ q ≠ 1 ∧ n = p * q

theorem smallest_a_is_eight :
  (∀ x : ℤ, is_composite (f 8 x)) ∧
  (∀ a : ℤ, 0 < a → a < 8 → ∃ x : ℤ, ¬is_composite (f a x)) :=
sorry

end NUMINAMATH_CALUDE_smallest_a_is_eight_l1285_128588


namespace NUMINAMATH_CALUDE_carton_weight_is_three_l1285_128562

/-- The weight of one crate of vegetables in kilograms. -/
def crate_weight : ℝ := 4

/-- The number of crates in the load. -/
def num_crates : ℕ := 12

/-- The number of cartons in the load. -/
def num_cartons : ℕ := 16

/-- The total weight of the load in kilograms. -/
def total_weight : ℝ := 96

/-- The weight of one carton of vegetables in kilograms. -/
def carton_weight : ℝ := 3

/-- Theorem stating that the weight of one carton of vegetables is 3 kilograms. -/
theorem carton_weight_is_three :
  crate_weight * num_crates + carton_weight * num_cartons = total_weight :=
by sorry

end NUMINAMATH_CALUDE_carton_weight_is_three_l1285_128562


namespace NUMINAMATH_CALUDE_worker_count_l1285_128503

/-- Represents the number of workers in the factory -/
def num_workers : ℕ := sorry

/-- The initial average monthly salary of workers and supervisor -/
def initial_average_salary : ℚ := 430

/-- The initial supervisor's monthly salary -/
def initial_supervisor_salary : ℚ := 870

/-- The new average monthly salary after supervisor change -/
def new_average_salary : ℚ := 410

/-- The new supervisor's monthly salary -/
def new_supervisor_salary : ℚ := 690

/-- The total number of people (workers + new supervisor) after the change -/
def total_people : ℕ := 9

theorem worker_count :
  (num_workers + 1) * initial_average_salary - initial_supervisor_salary =
  total_people * new_average_salary - new_supervisor_salary ∧
  num_workers = 8 := by sorry

end NUMINAMATH_CALUDE_worker_count_l1285_128503


namespace NUMINAMATH_CALUDE_sequence_property_l1285_128505

theorem sequence_property (a : ℕ → ℝ) :
  (∀ n, a n > 0) →
  (∀ p q : ℕ, a (p + q) = a p * a q) →
  a 8 = 16 →
  a 10 = 32 := by
sorry

end NUMINAMATH_CALUDE_sequence_property_l1285_128505


namespace NUMINAMATH_CALUDE_expression_equality_l1285_128598

theorem expression_equality (x : ℝ) (h : x > 0) : 
  x^x - x^x = 0 ∧ (x - 1)^x = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l1285_128598


namespace NUMINAMATH_CALUDE_maya_total_pages_l1285_128568

/-- The number of books Maya read in the first week -/
def first_week_books : ℕ := 5

/-- The number of pages in each book Maya read in the first week -/
def first_week_pages_per_book : ℕ := 300

/-- The number of pages in each book Maya read in the second week -/
def second_week_pages_per_book : ℕ := 350

/-- The number of pages in each book Maya read in the third week -/
def third_week_pages_per_book : ℕ := 400

/-- The total number of pages Maya read over three weeks -/
def total_pages : ℕ :=
  (first_week_books * first_week_pages_per_book) +
  (2 * first_week_books * second_week_pages_per_book) +
  (3 * first_week_books * third_week_pages_per_book)

theorem maya_total_pages : total_pages = 11000 := by
  sorry

end NUMINAMATH_CALUDE_maya_total_pages_l1285_128568


namespace NUMINAMATH_CALUDE_integer_representation_l1285_128553

theorem integer_representation (n : ℤ) : 
  ∃ (a b c d : ℤ), n = a^2 + b^2 + c^2 + d^2 ∨ n = a^2 + b^2 + c^2 - d^2 ∨
                    n = a^2 + b^2 - c^2 - d^2 ∨ n = a^2 - b^2 - c^2 - d^2 :=
sorry

example : ∃ (a b c : ℤ), 1947 = a^2 - b^2 - c^2 :=
sorry

end NUMINAMATH_CALUDE_integer_representation_l1285_128553


namespace NUMINAMATH_CALUDE_min_workers_to_complete_job_l1285_128552

theorem min_workers_to_complete_job
  (total_days : ℕ)
  (days_worked : ℕ)
  (initial_workers : ℕ)
  (job_fraction_completed : ℚ)
  (h1 : total_days = 30)
  (h2 : days_worked = 6)
  (h3 : initial_workers = 8)
  (h4 : job_fraction_completed = 1/3)
  (h5 : days_worked < total_days) :
  ∃ (min_workers : ℕ),
    min_workers ≤ initial_workers ∧
    (min_workers : ℚ) * (total_days - days_worked : ℚ) * job_fraction_completed / days_worked ≥ 1 - job_fraction_completed ∧
    ∀ (w : ℕ), w < min_workers →
      (w : ℚ) * (total_days - days_worked : ℚ) * job_fraction_completed / days_worked < 1 - job_fraction_completed ∧
    min_workers = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_workers_to_complete_job_l1285_128552


namespace NUMINAMATH_CALUDE_aartis_work_time_l1285_128536

/-- Given that Aarti completes three times a piece of work in 27 days,
    prove that she can complete one piece of work in 9 days. -/
theorem aartis_work_time :
  ∀ (work_time : ℕ),
  (3 * work_time = 27) →
  (work_time = 9) :=
by sorry

end NUMINAMATH_CALUDE_aartis_work_time_l1285_128536


namespace NUMINAMATH_CALUDE_runners_alignment_time_l1285_128580

def steinLapTime : ℕ := 6
def roseLapTime : ℕ := 10
def schwartzLapTime : ℕ := 18

theorem runners_alignment_time :
  Nat.lcm steinLapTime (Nat.lcm roseLapTime schwartzLapTime) = 90 := by
  sorry

end NUMINAMATH_CALUDE_runners_alignment_time_l1285_128580


namespace NUMINAMATH_CALUDE_equal_sum_product_difference_N_l1285_128504

theorem equal_sum_product_difference_N (N : ℕ) :
  ∃ (a b c d : ℕ), (a + b = c + d) ∧ (c * d - a * b = N) := by
  sorry

end NUMINAMATH_CALUDE_equal_sum_product_difference_N_l1285_128504


namespace NUMINAMATH_CALUDE_negation_of_universal_nonnegative_square_l1285_128596

theorem negation_of_universal_nonnegative_square (P : ℝ → Prop) : 
  (¬ ∀ x : ℝ, x^2 ≥ 0) ↔ (∃ x₀ : ℝ, x₀^2 < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_nonnegative_square_l1285_128596


namespace NUMINAMATH_CALUDE_fractional_equation_positive_root_l1285_128556

theorem fractional_equation_positive_root (x m : ℝ) : 
  (2 / (x - 2) - (2 * x - m) / (2 - x) = 3) → 
  (x > 0) →
  (m = 6) := by
sorry

end NUMINAMATH_CALUDE_fractional_equation_positive_root_l1285_128556


namespace NUMINAMATH_CALUDE_discount_percentage_l1285_128585

theorem discount_percentage
  (MP : ℝ)
  (CP : ℝ)
  (SP : ℝ)
  (h1 : CP = 0.55 * MP)
  (h2 : (SP - CP) / CP = 0.5454545454545454)
  : (MP - SP) / MP = 0.15 := by
  sorry

end NUMINAMATH_CALUDE_discount_percentage_l1285_128585


namespace NUMINAMATH_CALUDE_optimal_price_for_target_profit_l1285_128506

-- Define the problem parameters
def cost : ℝ := 30
def initialPrice : ℝ := 40
def initialSales : ℝ := 600
def priceIncreaseSalesDrop : ℝ := 20
def priceDecreaseSalesIncrease : ℝ := 200
def stock : ℝ := 1210
def targetProfit : ℝ := 8400

-- Define the sales function based on price change
def sales (priceChange : ℝ) : ℝ :=
  initialSales + priceDecreaseSalesIncrease * priceChange

-- Define the profit function
def profit (priceChange : ℝ) : ℝ :=
  (initialPrice - priceChange - cost) * (sales priceChange)

-- Theorem statement
theorem optimal_price_for_target_profit :
  ∃ (priceChange : ℝ), profit priceChange = targetProfit ∧ 
  initialPrice - priceChange = 37 ∧
  sales priceChange ≤ stock :=
sorry

end NUMINAMATH_CALUDE_optimal_price_for_target_profit_l1285_128506


namespace NUMINAMATH_CALUDE_composite_iff_on_line_count_lines_l1285_128545

def S (a b : ℕ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 + (a * b + a) * p.2 - b - 1 = 0}

def A : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ a : ℕ, p = (0, 1 / (a : ℝ))}

def B : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ b : ℕ, p = ((b + 1 : ℝ), 0)}

def M (m : ℕ) : ℝ × ℝ := (m, -1)

def τ (m : ℕ) : ℕ := (Nat.divisors m).card

theorem composite_iff_on_line (m : ℕ) :
  (∃ a b : ℕ, m = (a + 1) * (b + 1)) ↔
  (∃ a b : ℕ, M m ∈ S a b) :=
sorry

theorem count_lines (m : ℕ) :
  (Nat.card {p : ℕ × ℕ | m = (p.1 + 1) * (p.2 + 1)}) = τ m - 2 :=
sorry

end NUMINAMATH_CALUDE_composite_iff_on_line_count_lines_l1285_128545


namespace NUMINAMATH_CALUDE_min_additional_weeks_equals_additional_wins_needed_l1285_128523

/-- Represents the number of dollars Bob has won so far -/
def initial_winnings : ℕ := 200

/-- Represents the number of additional wins needed to afford the puppy -/
def additional_wins_needed : ℕ := 8

/-- Represents the prize money for each win in dollars -/
def prize_money : ℕ := 100

/-- Proves that the minimum number of additional weeks Bob must win first place is equal to the number of additional wins needed -/
theorem min_additional_weeks_equals_additional_wins_needed :
  additional_wins_needed = additional_wins_needed := by sorry

end NUMINAMATH_CALUDE_min_additional_weeks_equals_additional_wins_needed_l1285_128523


namespace NUMINAMATH_CALUDE_annual_increase_y_l1285_128590

/-- The annual increase in price of commodity Y -/
def y : ℝ := sorry

/-- The price of commodity X in a given year -/
def price_x (year : ℕ) : ℝ :=
  4.20 + 0.30 * (year - 2001)

/-- The price of commodity Y in a given year -/
def price_y (year : ℕ) : ℝ :=
  4.40 + y * (year - 2001)

theorem annual_increase_y : y = 0.20 :=
  have h1 : price_x 2010 = price_y 2010 + 0.70 := by sorry
  sorry

end NUMINAMATH_CALUDE_annual_increase_y_l1285_128590


namespace NUMINAMATH_CALUDE_simplify_expression_l1285_128547

theorem simplify_expression :
  4 * Real.sqrt 5 + Real.sqrt 45 - Real.sqrt 8 + 4 * Real.sqrt 2 = 7 * Real.sqrt 5 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1285_128547


namespace NUMINAMATH_CALUDE_min_value_expression_l1285_128555

theorem min_value_expression (x : ℝ) : (x^2 + 13) / Real.sqrt (x^2 + 7) ≥ 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l1285_128555


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l1285_128559

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def are_parallel (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v = (k * w.1, k * w.2)

theorem parallel_vectors_m_value :
  let a : ℝ × ℝ := (-2, 1)
  let b : ℝ × ℝ := (m, 3)
  are_parallel a b → m = -6 :=
by sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l1285_128559


namespace NUMINAMATH_CALUDE_perfect_square_product_iff_factors_l1285_128537

theorem perfect_square_product_iff_factors (x y z : ℕ+) :
  ∃ (n : ℕ), (x * y + 1) * (y * z + 1) * (z * x + 1) = n ^ 2 ↔
  (∃ (a b c : ℕ), (x * y + 1 = a ^ 2) ∧ (y * z + 1 = b ^ 2) ∧ (z * x + 1 = c ^ 2)) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_product_iff_factors_l1285_128537


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1285_128535

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (∀ a b : ℝ, a > 1 ∧ b > 1 → a + b > 2) ∧
  (∃ a b : ℝ, a + b > 2 ∧ ¬(a > 1 ∧ b > 1)) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1285_128535


namespace NUMINAMATH_CALUDE_campaign_fund_distribution_l1285_128527

theorem campaign_fund_distribution (total : ℝ) (family_percent : ℝ) (own_savings : ℝ) :
  total = 10000 →
  family_percent = 0.3 →
  own_savings = 4200 →
  ∃ (friends_contribution : ℝ),
    friends_contribution = total * 0.4 ∧
    total = friends_contribution + (family_percent * (total - friends_contribution)) + own_savings :=
by sorry

end NUMINAMATH_CALUDE_campaign_fund_distribution_l1285_128527


namespace NUMINAMATH_CALUDE_four_person_four_office_assignment_l1285_128500

def number_of_assignments (n : ℕ) : ℕ := n.factorial

theorem four_person_four_office_assignment :
  number_of_assignments 4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_four_person_four_office_assignment_l1285_128500


namespace NUMINAMATH_CALUDE_factor_implies_c_value_l1285_128531

theorem factor_implies_c_value (c : ℝ) : 
  (∀ x : ℝ, (4*x + 14) ∣ (6*x^3 + 19*x^2 + c*x + 70)) → c = 13 :=
by sorry

end NUMINAMATH_CALUDE_factor_implies_c_value_l1285_128531


namespace NUMINAMATH_CALUDE_geese_to_ducks_ratio_l1285_128538

theorem geese_to_ducks_ratio (initial_ducks : ℕ) (arriving_ducks : ℕ) (leaving_geese : ℕ) (initial_geese : ℕ) :
  initial_ducks = 25 →
  arriving_ducks = 4 →
  leaving_geese = 10 →
  initial_geese - leaving_geese = initial_ducks + arriving_ducks + 1 →
  (initial_geese : ℚ) / initial_ducks = 8 / 5 := by
sorry

end NUMINAMATH_CALUDE_geese_to_ducks_ratio_l1285_128538


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l1285_128581

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y^2 = 6 + x) ↔ x ≥ -6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l1285_128581


namespace NUMINAMATH_CALUDE_four_liters_possible_l1285_128548

/-- Represents the state of water in two vessels -/
structure WaterState :=
  (small : ℕ)  -- Amount of water in the 3-liter vessel
  (large : ℕ)  -- Amount of water in the 5-liter vessel

/-- Represents a pouring operation between vessels -/
inductive PourOperation
  | FillSmall
  | FillLarge
  | EmptySmall
  | EmptyLarge
  | SmallToLarge
  | LargeToSmall

/-- Applies a pouring operation to a water state -/
def applyOperation (state : WaterState) (op : PourOperation) : WaterState :=
  match op with
  | PourOperation.FillSmall => ⟨3, state.large⟩
  | PourOperation.FillLarge => ⟨state.small, 5⟩
  | PourOperation.EmptySmall => ⟨0, state.large⟩
  | PourOperation.EmptyLarge => ⟨state.small, 0⟩
  | PourOperation.SmallToLarge =>
      let amount := min state.small (5 - state.large)
      ⟨state.small - amount, state.large + amount⟩
  | PourOperation.LargeToSmall =>
      let amount := min state.large (3 - state.small)
      ⟨state.small + amount, state.large - amount⟩

/-- Theorem stating that it's possible to obtain 4 liters in the 5-liter vessel -/
theorem four_liters_possible : ∃ (ops : List PourOperation),
  (ops.foldl applyOperation ⟨0, 0⟩).large = 4 :=
sorry

end NUMINAMATH_CALUDE_four_liters_possible_l1285_128548


namespace NUMINAMATH_CALUDE_largest_class_size_l1285_128519

/-- Proves that in a school with 5 classes, where each class has 2 students less than the previous class,
    and the total number of students is 140, the number of students in the largest class is 32. -/
theorem largest_class_size (num_classes : Nat) (student_difference : Nat) (total_students : Nat)
    (h1 : num_classes = 5)
    (h2 : student_difference = 2)
    (h3 : total_students = 140) :
    ∃ (x : Nat), x = 32 ∧ 
    (x + (x - student_difference) + (x - 2*student_difference) + 
     (x - 3*student_difference) + (x - 4*student_difference) = total_students) :=
  by sorry

end NUMINAMATH_CALUDE_largest_class_size_l1285_128519
