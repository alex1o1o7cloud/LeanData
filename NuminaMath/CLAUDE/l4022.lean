import Mathlib

namespace seed_cost_calculation_l4022_402285

/-- Given that 2 pounds of seed cost $44.68, prove that 6 pounds of seed will cost $134.04. -/
theorem seed_cost_calculation (cost_for_two_pounds : ℝ) (pounds_needed : ℝ) : 
  cost_for_two_pounds = 44.68 → pounds_needed = 6 → 
  (pounds_needed / 2) * cost_for_two_pounds = 134.04 := by
sorry

end seed_cost_calculation_l4022_402285


namespace good_numbers_in_set_l4022_402278

/-- A number n is a "good number" if there exists a permutation of 1..n such that
    k + a[k] is a perfect square for all k in 1..n -/
def is_good_number (n : ℕ) : Prop :=
  ∃ a : Fin n → Fin n, Function.Bijective a ∧
    ∀ k : Fin n, ∃ m : ℕ, (k : ℕ) + (a k : ℕ) + 1 = m * m

theorem good_numbers_in_set : 
  is_good_number 13 ∧ 
  is_good_number 15 ∧ 
  is_good_number 17 ∧ 
  is_good_number 19 ∧ 
  ¬is_good_number 11 := by
  sorry

#check good_numbers_in_set

end good_numbers_in_set_l4022_402278


namespace inscribed_circle_radius_l4022_402275

theorem inscribed_circle_radius (DE DF EF : ℝ) (h1 : DE = 26) (h2 : DF = 15) (h3 : EF = 17) :
  let s := (DE + DF + EF) / 2
  let K := Real.sqrt (s * (s - DE) * (s - DF) * (s - EF))
  K / s = Real.sqrt 6 := by sorry

end inscribed_circle_radius_l4022_402275


namespace arithmetic_progression_rth_term_l4022_402231

/-- The sum of the first n terms of an arithmetic progression -/
def S (n : ℕ) : ℝ := 4 * n + 5 * n^2

/-- The rth term of the arithmetic progression -/
def a (r : ℕ) : ℝ := S r - S (r - 1)

theorem arithmetic_progression_rth_term (r : ℕ) (hr : r > 0) : a r = 10 * r - 1 := by
  sorry

end arithmetic_progression_rth_term_l4022_402231


namespace circles_internally_tangent_l4022_402299

-- Define the circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 4*x + 8*y - 5 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 4*x + 4*y - 1 = 0

-- Define the centers and radii of the circles
def center1 : ℝ × ℝ := (-2, -4)
def radius1 : ℝ := 5
def center2 : ℝ × ℝ := (-2, -2)
def radius2 : ℝ := 3

-- Theorem statement
theorem circles_internally_tangent :
  let d := Real.sqrt ((center1.1 - center2.1)^2 + (center1.2 - center2.2)^2)
  d = abs (radius1 - radius2) ∧ d < radius1 + radius2 := by sorry

end circles_internally_tangent_l4022_402299


namespace agreed_period_is_18_months_prove_agreed_period_of_service_l4022_402271

/-- Represents the agreed period of service in months -/
def agreed_period : ℕ := 18

/-- Represents the actual period served in months -/
def actual_period : ℕ := 9

/-- Represents the full payment amount in rupees -/
def full_payment : ℕ := 800

/-- Represents the actual payment received in rupees -/
def actual_payment : ℕ := 400

/-- Theorem stating that the agreed period of service is 18 months -/
theorem agreed_period_is_18_months :
  (actual_payment = full_payment / 2) →
  (actual_period * 2 = agreed_period) :=
by
  sorry

/-- Main theorem proving the agreed period of service -/
theorem prove_agreed_period_of_service :
  agreed_period = 18 :=
by
  sorry

end agreed_period_is_18_months_prove_agreed_period_of_service_l4022_402271


namespace valid_sequences_of_length_16_l4022_402242

/-- Represents a sequence of C's and D's -/
inductive CDSequence
  | C : CDSequence → CDSequence
  | D : CDSequence → CDSequence
  | empty : CDSequence

/-- Returns true if the given sequence satisfies the conditions -/
def isValidSequence (s : CDSequence) : Bool :=
  sorry

/-- Returns the length of the given sequence -/
def sequenceLength (s : CDSequence) : Nat :=
  sorry

/-- Returns the number of valid sequences of a given length -/
def countValidSequences (n : Nat) : Nat :=
  sorry

theorem valid_sequences_of_length_16 :
  countValidSequences 16 = 55 := by sorry

end valid_sequences_of_length_16_l4022_402242


namespace william_has_more_money_l4022_402211

/-- Represents the amount of money in different currencies --/
structure Money where
  usd_20 : ℕ
  usd_10 : ℕ
  usd_5 : ℕ
  gbp_10 : ℕ
  eur_20 : ℕ

/-- Converts Money to USD --/
def to_usd (m : Money) (gbp_rate : ℚ) (eur_rate : ℚ) : ℚ :=
  (m.usd_20 * 20 + m.usd_10 * 10 + m.usd_5 * 5 + m.gbp_10 * 10 * gbp_rate + m.eur_20 * 20 * eur_rate : ℚ)

/-- Oliver's money --/
def oliver : Money := ⟨10, 0, 3, 12, 0⟩

/-- William's money --/
def william : Money := ⟨0, 15, 4, 0, 20⟩

/-- The exchange rates --/
def gbp_rate : ℚ := 138 / 100
def eur_rate : ℚ := 118 / 100

theorem william_has_more_money :
  to_usd william gbp_rate eur_rate - to_usd oliver gbp_rate eur_rate = 2614 / 10 := by
  sorry

end william_has_more_money_l4022_402211


namespace soccer_balls_count_l4022_402297

/-- The cost of a football in dollars -/
def football_cost : ℝ := 35

/-- The cost of a soccer ball in dollars -/
def soccer_ball_cost : ℝ := 50

/-- The number of soccer balls in the first set -/
def soccer_balls_in_first_set : ℕ := 1

theorem soccer_balls_count : 
  3 * football_cost + soccer_balls_in_first_set * soccer_ball_cost = 155 ∧
  2 * football_cost + 3 * soccer_ball_cost = 220 →
  soccer_balls_in_first_set = 1 := by
  sorry

end soccer_balls_count_l4022_402297


namespace partial_fraction_decomposition_l4022_402282

theorem partial_fraction_decomposition :
  ∃! (P Q R : ℚ),
    ∀ (x : ℝ), x ≠ 1 → x ≠ 4 → x ≠ -2 →
      (x^2 - 18) / ((x - 1) * (x - 4) * (x + 2)) =
      P / (x - 1) + Q / (x - 4) + R / (x + 2) ∧
      P = 17/9 ∧ Q = 1/9 ∧ R = -5/9 := by
sorry

end partial_fraction_decomposition_l4022_402282


namespace function_inequality_l4022_402277

theorem function_inequality (f : ℝ → ℝ) (h1 : Differentiable ℝ f) 
  (h2 : ∀ x : ℝ, f x < (deriv^[2] f) x) : 
  (Real.exp 2019 * f (-2019) < f 0) ∧ (f 2019 > Real.exp 2019 * f 0) := by
  sorry

end function_inequality_l4022_402277


namespace age_condition_l4022_402221

/-- Given three people A, B, and C, this theorem states that if A is older than B,
    then "C is older than B" is a necessary but not sufficient condition for
    "the sum of B and C's ages is greater than twice A's age". -/
theorem age_condition (a b c : ℕ) (h : a > b) :
  (c > b → b + c > 2 * a) ∧ ¬(b + c > 2 * a → c > b) := by
  sorry

end age_condition_l4022_402221


namespace point_in_fourth_quadrant_l4022_402210

-- Define the Cartesian coordinate system
def CartesianPoint := ℝ × ℝ

-- Define the fourth quadrant
def is_in_fourth_quadrant (p : CartesianPoint) : Prop :=
  p.1 > 0 ∧ p.2 < 0

-- Define pi as a real number
noncomputable def π : ℝ := Real.pi

-- Theorem statement
theorem point_in_fourth_quadrant :
  is_in_fourth_quadrant (π, -1) := by
  sorry

end point_in_fourth_quadrant_l4022_402210


namespace sin_minus_cos_special_angle_l4022_402284

/-- Given an angle α whose terminal side passes through the point (3a, -4a) where a < 0,
    prove that sinα - cosα = 7/5 -/
theorem sin_minus_cos_special_angle (a : ℝ) (α : Real) (h : a < 0) 
    (h_terminal : ∃ k : ℝ, k > 0 ∧ k * Real.cos α = 3 * a ∧ k * Real.sin α = -4 * a) :
    Real.sin α - Real.cos α = 7/5 := by
  sorry

end sin_minus_cos_special_angle_l4022_402284


namespace distance_between_stations_l4022_402283

/-- The distance between two stations given train travel information -/
theorem distance_between_stations
  (speed1 : ℝ) (time1 : ℝ) (speed2 : ℝ) (time2 : ℝ)
  (h1 : speed1 = 20)
  (h2 : time1 = 3)
  (h3 : speed2 = 25)
  (h4 : time2 = 2)
  (h5 : speed1 * time1 + speed2 * time2 = speed1 * time1 + speed2 * time2) :
  speed1 * time1 + speed2 * time2 = 110 := by
  sorry

#check distance_between_stations

end distance_between_stations_l4022_402283


namespace dice_roll_probability_l4022_402246

def is_valid_roll (m n : ℕ) : Prop := 1 ≤ m ∧ m ≤ 6 ∧ 1 ≤ n ∧ n ≤ 6

def angle_greater_than_90 (m n : ℕ) : Prop := m > n

def count_favorable_outcomes : ℕ := 15

def total_outcomes : ℕ := 36

theorem dice_roll_probability : 
  (count_favorable_outcomes : ℚ) / total_outcomes = 5 / 12 :=
sorry

end dice_roll_probability_l4022_402246


namespace max_vertex_sum_l4022_402296

/-- Represents a face of the dice -/
structure Face where
  value : Nat
  deriving Repr

/-- Represents a cubical dice -/
structure Dice where
  faces : List Face
  opposite_sum : Nat

/-- Defines a valid cubical dice where opposite faces sum to 8 -/
def is_valid_dice (d : Dice) : Prop :=
  d.faces.length = 6 ∧
  d.opposite_sum = 8 ∧
  ∀ (f1 f2 : Face), f1 ∈ d.faces → f2 ∈ d.faces → f1 ≠ f2 → f1.value + f2.value = d.opposite_sum

/-- Represents three faces meeting at a vertex -/
structure Vertex where
  f1 : Face
  f2 : Face
  f3 : Face

/-- Calculates the sum of face values at a vertex -/
def vertex_sum (v : Vertex) : Nat :=
  v.f1.value + v.f2.value + v.f3.value

/-- Defines a valid vertex of the dice -/
def is_valid_vertex (d : Dice) (v : Vertex) : Prop :=
  v.f1 ∈ d.faces ∧ v.f2 ∈ d.faces ∧ v.f3 ∈ d.faces ∧
  v.f1 ≠ v.f2 ∧ v.f1 ≠ v.f3 ∧ v.f2 ≠ v.f3

theorem max_vertex_sum (d : Dice) (h : is_valid_dice d) :
  ∀ (v : Vertex), is_valid_vertex d v → vertex_sum v ≤ 11 :=
sorry

end max_vertex_sum_l4022_402296


namespace pure_imaginary_product_l4022_402236

/-- A complex number z is pure imaginary if its real part is zero and its imaginary part is non-zero -/
def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

/-- The theorem states that if (a+i)(2+i) is a pure imaginary number, then a = 1/2 -/
theorem pure_imaginary_product (a : ℝ) : 
  is_pure_imaginary ((a + Complex.I) * (2 + Complex.I)) → a = 1/2 := by
  sorry

end pure_imaginary_product_l4022_402236


namespace given_equation_is_quadratic_l4022_402202

/-- Represents a quadratic equation in standard form -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  h : a ≠ 0

/-- The given equation: 3(x+1)^2 = 2(x+1) -/
def given_equation (x : ℝ) : Prop :=
  3 * (x + 1)^2 = 2 * (x + 1)

/-- Theorem stating that the given equation is equivalent to a quadratic equation -/
theorem given_equation_is_quadratic :
  ∃ (q : QuadraticEquation), ∀ x, given_equation x ↔ q.a * x^2 + q.b * x + q.c = 0 :=
sorry

end given_equation_is_quadratic_l4022_402202


namespace expression_evaluation_l4022_402224

theorem expression_evaluation : (4 * 6) / (12 * 14) * (8 * 12 * 14) / (4 * 6 * 8) = 1 := by
  sorry

end expression_evaluation_l4022_402224


namespace power_function_through_point_l4022_402295

theorem power_function_through_point (f : ℝ → ℝ) (n : ℝ) :
  (∀ x, f x = x^n) →  -- f is a power function
  f 2 = 8 →           -- f passes through (2,8)
  n = 3 :=            -- the power must be 3
by
  sorry


end power_function_through_point_l4022_402295


namespace max_m_F_theorem_l4022_402238

/-- The maximum value of m(F) for subsets F of {1, ..., 2n} with n elements -/
def max_m_F (n : ℕ) : ℕ :=
  if n = 2 ∨ n = 3 then 12
  else if n = 4 then 24
  else if n % 2 = 1 then 3 * (n + 1)
  else 3 * (n + 2)

/-- The theorem stating the maximum value of m(F) -/
theorem max_m_F_theorem (n : ℕ) (h : n ≥ 2) :
  ∀ (F : Finset ℕ),
    F ⊆ Finset.range (2 * n + 1) →
    F.card = n →
    (∀ (x y : ℕ), x ∈ F → y ∈ F → x ≠ y → Nat.lcm x y ≥ max_m_F n) :=
by sorry

end max_m_F_theorem_l4022_402238


namespace min_value_function_l4022_402218

theorem min_value_function (x : ℝ) (h : x > 0) : 2 + 4*x + 1/x ≥ 6 ∧ ∃ y > 0, 2 + 4*y + 1/y = 6 := by
  sorry

end min_value_function_l4022_402218


namespace hyperbola_equation_l4022_402203

theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (∀ x y : ℝ, x^2 / 12 + y^2 / 4 = 1) →
  (∃ c : ℝ, c > 0 ∧ c^2 = a^2 - b^2 ∧ c^2 = 12 - 4) →
  (∀ x : ℝ, ∃ y : ℝ, y = Real.sqrt 3 * x ∧ x^2 / a^2 - y^2 / b^2 = 1) →
  (∀ x y : ℝ, x^2 / 2 - y^2 / 6 = 1) :=
by sorry

end hyperbola_equation_l4022_402203


namespace seven_thousand_twenty_two_l4022_402261

theorem seven_thousand_twenty_two : 7000 + 22 = 7022 := by
  sorry

end seven_thousand_twenty_two_l4022_402261


namespace quadratic_roots_property_l4022_402235

theorem quadratic_roots_property (b c : ℝ) : 
  (3 * b^2 + 5 * b - 2 = 0) → 
  (3 * c^2 + 5 * c - 2 = 0) → 
  (b-1)*(c-1) = 2 := by
sorry

end quadratic_roots_property_l4022_402235


namespace geometric_sequence_problem_l4022_402274

theorem geometric_sequence_problem :
  ∃ (a r : ℝ), 
    -- Condition 1: a, ar, ar² form a geometric sequence
    (a * r * r - a * r = a * r - a) ∧
    -- Condition 2: ar² - a = 48
    (a * r * r - a = 48) ∧
    -- Condition 3: (ar²)² - a² = (208/217) * (a² + (ar)² + (ar²)²)
    ((a * r * r)^2 - a^2 = (208/217) * (a^2 + (a * r)^2 + (a * r * r)^2)) :=
by sorry

end geometric_sequence_problem_l4022_402274


namespace range_of_negative_values_l4022_402226

-- Define the properties of the function f
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def decreasing_on_neg (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y ∧ y ≤ 0 → f x ≥ f y

-- State the theorem
theorem range_of_negative_values
  (f : ℝ → ℝ)
  (h_even : is_even f)
  (h_decreasing : decreasing_on_neg f)
  (h_f2 : f 2 = 0) :
  {x : ℝ | f x < 0} = Set.Ioo (-2) 2 := by
  sorry

end range_of_negative_values_l4022_402226


namespace lori_beanie_babies_l4022_402213

theorem lori_beanie_babies (sydney_beanie_babies : ℕ) 
  (h1 : sydney_beanie_babies + 15 * sydney_beanie_babies = 320) : 
  15 * sydney_beanie_babies = 300 := by
  sorry

end lori_beanie_babies_l4022_402213


namespace ellipse_equation_hyperbola_equation_l4022_402252

-- Problem 1: Ellipse
theorem ellipse_equation (x y : ℝ) :
  let foci_ellipse : ℝ × ℝ → Prop := λ p => p.1^2 / 9 + p.2^2 / 4 = 1
  let passes_through : ℝ × ℝ → Prop := λ p => p = (-3, 2)
  let result_equation : ℝ × ℝ → Prop := λ p => p.1^2 / 15 + p.2^2 / 10 = 1
  (∃ e : Set (ℝ × ℝ), (∀ p ∈ e, result_equation p) ∧
    (∃ p ∈ e, passes_through p) ∧
    (∀ f : ℝ × ℝ, foci_ellipse f ↔ (∃ f' : ℝ × ℝ, (∀ p ∈ e, (p.1 - f.1)^2 + (p.2 - f.2)^2 =
                                                          (p.1 - f'.1)^2 + (p.2 - f'.2)^2) ∧
                                                 f.2 = f'.2 ∧ f.1 = -f'.1))) :=
by sorry

-- Problem 2: Hyperbola
theorem hyperbola_equation (x y : ℝ) :
  let passes_through : ℝ × ℝ → Prop := λ p => p = (2, -1)
  let has_asymptotes : (ℝ → ℝ) → Prop := λ f => f x = 3 * x ∨ f x = -3 * x
  let result_equation : ℝ × ℝ → Prop := λ p => p.1^2 / (35/9) - p.2^2 / 35 = 1
  (∃ h : Set (ℝ × ℝ), (∀ p ∈ h, result_equation p) ∧
    (∃ p ∈ h, passes_through p) ∧
    (∃ f g : ℝ → ℝ, has_asymptotes f ∧ has_asymptotes g ∧
      (∀ x : ℝ, (x, f x) ∉ h ∧ (x, g x) ∉ h) ∧
      (∀ ε > 0, ∃ δ > 0, ∀ p ∈ h, |p.1| > δ → (|p.2 - f p.1| < ε ∨ |p.2 - g p.1| < ε)))) :=
by sorry

end ellipse_equation_hyperbola_equation_l4022_402252


namespace cube_of_fraction_l4022_402256

theorem cube_of_fraction (x y : ℝ) : 
  (-2/3 * x * y^2)^3 = -8/27 * x^3 * y^6 := by
sorry

end cube_of_fraction_l4022_402256


namespace christmas_on_thursday_l4022_402292

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents dates in November and December -/
structure Date where
  month : Nat
  day : Nat

/-- Returns the day of the week for a given date, assuming November 27 is a Thursday -/
def dayOfWeek (date : Date) : DayOfWeek :=
  sorry

theorem christmas_on_thursday (thanksgiving : Date)
    (h1 : thanksgiving.month = 11)
    (h2 : thanksgiving.day = 27)
    (h3 : dayOfWeek thanksgiving = DayOfWeek.Thursday) :
    dayOfWeek ⟨12, 25⟩ = DayOfWeek.Thursday :=
  sorry

end christmas_on_thursday_l4022_402292


namespace initial_distance_adrian_colton_initial_distance_l4022_402251

/-- The initial distance between Adrian and Colton given their relative motion -/
theorem initial_distance (speed : ℝ) (time : ℝ) (final_distance : ℝ) : ℝ :=
  let distance_run := speed * time
  distance_run + final_distance

/-- Proof of the initial distance between Adrian and Colton -/
theorem adrian_colton_initial_distance : 
  initial_distance 17 13 68 = 289 := by
  sorry

end initial_distance_adrian_colton_initial_distance_l4022_402251


namespace g_of_nine_l4022_402245

/-- Given a function g(x) = ax^7 - bx^3 + cx - 7 where g(-9) = 9, prove that g(9) = -23 -/
theorem g_of_nine (a b c : ℝ) (g : ℝ → ℝ) 
  (h1 : ∀ x, g x = a * x^7 - b * x^3 + c * x - 7)
  (h2 : g (-9) = 9) : 
  g 9 = -23 := by sorry

end g_of_nine_l4022_402245


namespace intersection_of_sets_l4022_402253

theorem intersection_of_sets : 
  let A : Set ℤ := {-1, 0, 1, 2}
  let B : Set ℤ := {x | x ≥ 2}
  A ∩ B = {2} := by
  sorry

end intersection_of_sets_l4022_402253


namespace no_valid_positive_x_for_equal_volume_increase_l4022_402290

theorem no_valid_positive_x_for_equal_volume_increase (x : ℝ) : 
  x > 0 → 
  π * (5 + x)^2 * 10 - π * 5^2 * 10 ≠ π * 5^2 * (10 + x) - π * 5^2 * 10 := by
  sorry

end no_valid_positive_x_for_equal_volume_increase_l4022_402290


namespace parabola_directrix_l4022_402254

/-- The directrix of a parabola y² = 2x is x = -1/2 -/
theorem parabola_directrix (x y : ℝ) : y^2 = 2*x → (∃ (k : ℝ), k = -1/2 ∧ (∀ (x₀ y₀ : ℝ), y₀^2 = 2*x₀ → x₀ = k)) :=
sorry

end parabola_directrix_l4022_402254


namespace perfect_square_condition_l4022_402237

theorem perfect_square_condition (m : ℤ) : 
  (∃ k : ℤ, ∀ x : ℤ, x^2 + 2*(m-3) + 16 = (x + k)^2) → (m = -1 ∨ m = 7) :=
by sorry

end perfect_square_condition_l4022_402237


namespace perfect_square_primes_no_perfect_square_primes_l4022_402217

theorem perfect_square_primes (p : Nat) : Prime p → (∃ n : Nat, (7^(p-1) - 1) / p = n^2) ↔ p = 3 :=
sorry

theorem no_perfect_square_primes (p : Nat) : Prime p → ¬∃ n : Nat, (11^(p-1) - 1) / p = n^2 :=
sorry

end perfect_square_primes_no_perfect_square_primes_l4022_402217


namespace inequality_transformation_l4022_402232

theorem inequality_transformation (a b : ℝ) (h : a < b) : -3 * a > -3 * b := by
  sorry

end inequality_transformation_l4022_402232


namespace largest_prime_factor_of_1729_l4022_402204

theorem largest_prime_factor_of_1729 : 
  ∃ (p : Nat), Nat.Prime p ∧ p ∣ 1729 ∧ ∀ (q : Nat), Nat.Prime q → q ∣ 1729 → q ≤ p ∧ p = 19 := by
  sorry

end largest_prime_factor_of_1729_l4022_402204


namespace sum_of_solutions_l4022_402227

-- Define the equations
def equation1 (x : ℝ) : Prop := x + Real.log x = 3
def equation2 (x : ℝ) : Prop := x + (10 : ℝ) ^ x = 3

-- State the theorem
theorem sum_of_solutions (x₁ x₂ : ℝ) 
  (h1 : equation1 x₁) (h2 : equation2 x₂) : x₁ + x₂ = 6 := by
  sorry

end sum_of_solutions_l4022_402227


namespace soda_cost_l4022_402239

theorem soda_cost (bob_burgers bob_sodas bob_total carol_burgers carol_sodas carol_total : ℕ) 
  (h_bob : bob_burgers = 4 ∧ bob_sodas = 3 ∧ bob_total = 500)
  (h_carol : carol_burgers = 3 ∧ carol_sodas = 4 ∧ carol_total = 540) :
  ∃ (burger_cost soda_cost : ℕ), 
    burger_cost * bob_burgers + soda_cost * bob_sodas = bob_total ∧
    burger_cost * carol_burgers + soda_cost * carol_sodas = carol_total ∧
    soda_cost = 94 := by
  sorry

end soda_cost_l4022_402239


namespace iceland_visitors_l4022_402259

theorem iceland_visitors (total : ℕ) (norway : ℕ) (both : ℕ) (neither : ℕ) :
  total = 60 →
  norway = 23 →
  both = 31 →
  neither = 33 →
  ∃ iceland : ℕ, iceland = 35 ∧ total = iceland + norway - both + neither :=
by sorry

end iceland_visitors_l4022_402259


namespace work_completion_time_l4022_402230

/-- Given a work that can be completed by person A in 60 days, and together with person B in 24 days,
    this theorem proves that B can complete the remaining work alone in 40 days after A works for 15 days. -/
theorem work_completion_time (total_work : ℝ) : 
  (∃ (rate_a rate_b : ℝ), 
    rate_a * 60 = total_work ∧ 
    (rate_a + rate_b) * 24 = total_work ∧ 
    rate_b * 40 = total_work - rate_a * 15) := by
  sorry

end work_completion_time_l4022_402230


namespace complex_power_magnitude_l4022_402205

theorem complex_power_magnitude (z w : ℂ) (n : ℕ) :
  z = w ^ n → Complex.abs z ^ 2 = Complex.abs w ^ (2 * n) := by
  sorry

end complex_power_magnitude_l4022_402205


namespace ab_value_l4022_402264

theorem ab_value (a b : ℝ) (h1 : a + b = 5) (h2 : a^3 + b^3 = 125) : a * b = 0 := by
  sorry

end ab_value_l4022_402264


namespace hexagon_angle_measure_l4022_402281

theorem hexagon_angle_measure (a b c d e : ℝ) (h1 : a = 130) (h2 : b = 95) (h3 : c = 115) (h4 : d = 110) (h5 : e = 87) : 
  720 - (a + b + c + d + e) = 183 := by
  sorry

end hexagon_angle_measure_l4022_402281


namespace count_negative_rationals_l4022_402201

theorem count_negative_rationals : 
  let S : Finset ℚ := {-5, -(-3), 3.14, |-2/7|, -(2^3), 0}
  (S.filter (λ x => x < 0)).card = 2 := by sorry

end count_negative_rationals_l4022_402201


namespace joyful_point_properties_l4022_402222

-- Define a "joyful point"
def is_joyful_point (m n : ℝ) : Prop := 2 * m = 6 - n

-- Define the point P
def P (m n : ℝ) : ℝ × ℝ := (m, n + 2)

theorem joyful_point_properties :
  -- Part 1: (1, 6) is a joyful point
  is_joyful_point 1 4 ∧
  P 1 4 = (1, 6) ∧
  -- Part 2: If P(a, -a+3) is a joyful point, then a = 5 and P is in the fourth quadrant
  (∀ a : ℝ, is_joyful_point a (-a + 3) → a = 5 ∧ 5 > 0 ∧ -2 < 0) ∧
  -- Part 3: The midpoint of OP is (5/2, -1)
  (let O : ℝ × ℝ := (0, 0);
   let P : ℝ × ℝ := (5, -2);
   (O.1 + P.1) / 2 = 5 / 2 ∧ (O.2 + P.2) / 2 = -1) :=
by sorry

end joyful_point_properties_l4022_402222


namespace jack_morning_emails_l4022_402280

/-- The number of emails Jack received in the morning -/
def morning_emails : ℕ := sorry

/-- The number of emails Jack received in the afternoon -/
def afternoon_emails : ℕ := 3

/-- The number of emails Jack received in the evening -/
def evening_emails : ℕ := 16

/-- The difference between morning and afternoon emails -/
def email_difference : ℕ := 2

theorem jack_morning_emails : 
  morning_emails = afternoon_emails + email_difference := by sorry

end jack_morning_emails_l4022_402280


namespace max_valid_rectangles_l4022_402269

/-- Represents a grid with dimensions and unit square size -/
structure Grid :=
  (width : ℕ)
  (height : ℕ)
  (unit_size : ℕ)

/-- Represents a coloring of the grid -/
def Coloring := Grid → Fin 2

/-- Represents a cutting of the grid into rectangles -/
def Cutting := Grid → List (ℕ × ℕ)

/-- Counts the number of rectangles with at most one black square -/
def count_valid_rectangles (g : Grid) (c : Coloring) (cut : Cutting) : ℕ :=
  sorry

/-- The main theorem stating the maximum number of valid rectangles -/
theorem max_valid_rectangles (g : Grid) 
  (h1 : g.width = 2020)
  (h2 : g.height = 2020)
  (h3 : g.unit_size = 11)
  (h4 : g.width / g.unit_size * (g.height / g.unit_size) = 400) :
  ∃ (c : Coloring) (cut : Cutting), 
    ∀ (c' : Coloring) (cut' : Cutting), 
      count_valid_rectangles g c cut ≥ count_valid_rectangles g c' cut' ∧ 
      count_valid_rectangles g c cut = 20 :=
sorry

end max_valid_rectangles_l4022_402269


namespace gift_package_combinations_l4022_402263

theorem gift_package_combinations : 
  let wrapping_paper_varieties : ℕ := 10
  let ribbon_colors : ℕ := 5
  let gift_card_types : ℕ := 5
  let gift_tag_types : ℕ := 2
  wrapping_paper_varieties * ribbon_colors * gift_card_types * gift_tag_types = 500 :=
by
  sorry

end gift_package_combinations_l4022_402263


namespace island_puzzle_l4022_402257

/-- Represents the nature of a person on the island -/
inductive PersonNature
| Knight
| Liar

/-- Represents the statement made by person A -/
def statement (nature : PersonNature) (treasures : Prop) : Prop :=
  (nature = PersonNature.Knight) ↔ treasures

/-- The main theorem about A's statement and the existence of treasures -/
theorem island_puzzle :
  ∀ (A_nature : PersonNature) (treasures : Prop),
    statement A_nature treasures →
    (¬ (A_nature = PersonNature.Knight ∨ A_nature = PersonNature.Liar) ∧ treasures) :=
by sorry

end island_puzzle_l4022_402257


namespace power_comparison_l4022_402255

theorem power_comparison : 2^51 > 4^25 := by
  sorry

end power_comparison_l4022_402255


namespace no_number_decreases_58_times_when_first_digit_removed_l4022_402279

theorem no_number_decreases_58_times_when_first_digit_removed :
  ¬ ∃ (n : ℕ) (x y : ℕ), 
    n ≥ 2 ∧ 
    x > 0 ∧ x < 10 ∧
    y > 0 ∧
    x * 10^(n-1) + y = 58 * y :=
by sorry

end no_number_decreases_58_times_when_first_digit_removed_l4022_402279


namespace order_of_abc_l4022_402219

theorem order_of_abc : 
  let a : ℝ := (Real.exp 0.6)⁻¹
  let b : ℝ := 0.4
  let c : ℝ := (Real.log 1.4) / 1.4
  c < b ∧ b < a := by sorry

end order_of_abc_l4022_402219


namespace stratified_sampling_l4022_402289

theorem stratified_sampling (seniors juniors freshmen sampled_freshmen : ℕ) :
  seniors = 1000 →
  juniors = 1200 →
  freshmen = 1500 →
  sampled_freshmen = 75 →
  (seniors + juniors + freshmen) * sampled_freshmen / freshmen = 185 := by
sorry

end stratified_sampling_l4022_402289


namespace paiges_team_size_l4022_402212

theorem paiges_team_size (total_points : ℕ) (paige_points : ℕ) (other_player_points : ℕ) :
  total_points = 41 →
  paige_points = 11 →
  other_player_points = 6 →
  ∃ (team_size : ℕ), team_size = (total_points - paige_points) / other_player_points + 1 ∧ team_size = 6 :=
by sorry

end paiges_team_size_l4022_402212


namespace iphone_purchase_savings_l4022_402268

/-- The price of an iPhone X in dollars -/
def iphone_x_price : ℝ := 600

/-- The price of an iPhone Y in dollars -/
def iphone_y_price : ℝ := 800

/-- The discount rate for buying at least 2 smartphones of the same model -/
def same_model_discount : ℝ := 0.05

/-- The discount rate for mixed purchases of at least 3 smartphones -/
def mixed_purchase_discount : ℝ := 0.03

/-- The total cost of buying three iPhones individually -/
def individual_cost : ℝ := 2 * iphone_x_price + iphone_y_price

/-- The discounted price of two iPhone X models -/
def discounted_iphone_x : ℝ := 2 * (iphone_x_price * (1 - same_model_discount))

/-- The discounted price of one iPhone Y model -/
def discounted_iphone_y : ℝ := iphone_y_price * (1 - mixed_purchase_discount)

/-- The total cost of buying three iPhones together with discounts -/
def group_cost : ℝ := discounted_iphone_x + discounted_iphone_y

/-- The savings from buying three iPhones together vs. individually -/
def savings : ℝ := individual_cost - group_cost

theorem iphone_purchase_savings : savings = 84 := by sorry

end iphone_purchase_savings_l4022_402268


namespace production_consistency_gizmo_production_zero_l4022_402223

/-- Represents the production rate of gadgets and gizmos -/
structure ProductionRate where
  workers : ℕ
  hours : ℕ
  gadgets : ℕ
  gizmos : ℕ

/-- Given production rates -/
def rate1 : ProductionRate := ⟨120, 1, 360, 240⟩
def rate2 : ProductionRate := ⟨40, 3, 240, 360⟩
def rate3 : ProductionRate := ⟨60, 4, 240, 0⟩

/-- Time to produce one gadget -/
def gadgetTime (r : ProductionRate) : ℚ :=
  (r.workers * r.hours : ℚ) / r.gadgets

/-- Time to produce one gizmo -/
def gizmoTime (r : ProductionRate) : ℚ :=
  (r.workers * r.hours : ℚ) / r.gizmos

theorem production_consistency (r1 r2 : ProductionRate) :
  gadgetTime r1 = gadgetTime r2 ∧ gizmoTime r1 = gizmoTime r2 := by sorry

theorem gizmo_production_zero :
  rate3.gizmos = 0 := by sorry

end production_consistency_gizmo_production_zero_l4022_402223


namespace probability_one_good_product_l4022_402293

def total_products : ℕ := 5
def good_products : ℕ := 3
def defective_products : ℕ := 2
def selections : ℕ := 2

theorem probability_one_good_product : 
  (Nat.choose good_products 1 * Nat.choose defective_products 1) / 
  Nat.choose total_products selections = 3 / 5 := by
sorry

end probability_one_good_product_l4022_402293


namespace square_root_of_four_l4022_402294

theorem square_root_of_four :
  {x : ℝ | x^2 = 4} = {2, -2} := by sorry

end square_root_of_four_l4022_402294


namespace vector_equation_solution_l4022_402233

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem vector_equation_solution (a b x : V) (h : 3 • a + 4 • (b - x) = 0) : 
  x = (3 / 4) • a + b := by
  sorry

end vector_equation_solution_l4022_402233


namespace complex_number_equidistant_l4022_402215

theorem complex_number_equidistant (z : ℂ) :
  Complex.abs (z - Complex.I) = Complex.abs (z - 1) ∧
  Complex.abs (z - 1) = Complex.abs (z - 2015) →
  z = Complex.mk 1008 1008 :=
by sorry

end complex_number_equidistant_l4022_402215


namespace minimize_quadratic_l4022_402216

theorem minimize_quadratic (c : ℝ) :
  ∃ (b : ℝ), ∀ (x : ℝ), 3 * b^2 + 2 * b + c ≤ 3 * x^2 + 2 * x + c :=
by
  use (-1/3)
  sorry

end minimize_quadratic_l4022_402216


namespace number_of_apples_l4022_402220

/-- Given a box of fruit with the following properties:
  * The total number of fruit pieces is 56
  * One-fourth of the fruit are oranges
  * The number of peaches is half the number of oranges
  * The number of apples is five times the number of peaches
  This theorem proves that the number of apples in the box is 35. -/
theorem number_of_apples (total : ℕ) (oranges peaches apples : ℕ) : 
  total = 56 →
  oranges = total / 4 →
  peaches = oranges / 2 →
  apples = 5 * peaches →
  apples = 35 := by
  sorry

end number_of_apples_l4022_402220


namespace triangle_angle_C_l4022_402208

theorem triangle_angle_C (a b c A B C : ℝ) : 
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  10 * a * Real.cos B = 3 * b * Real.cos A →
  Real.cos A = (5 * Real.sqrt 26) / 26 →
  C = 3 * π / 4 := by sorry

end triangle_angle_C_l4022_402208


namespace triangle_cosine_value_l4022_402258

theorem triangle_cosine_value (A B C : ℝ) (h : 7 * Real.sin B ^ 2 + 3 * Real.sin C ^ 2 = 2 * Real.sin A ^ 2 + 2 * Real.sin A * Real.sin B * Real.sin C) :
  Real.cos (A - π / 4) = -Real.sqrt 10 / 10 := by
  sorry

end triangle_cosine_value_l4022_402258


namespace chemical_mixture_problem_l4022_402214

/-- Given two chemical solutions x and y, and their mixture, prove the percentage of chemical b in solution x -/
theorem chemical_mixture_problem (x_a : ℝ) (y_a y_b : ℝ) (mixture_a : ℝ) :
  x_a = 0.3 →
  y_a = 0.4 →
  y_b = 0.6 →
  mixture_a = 0.32 →
  0.8 * x_a + 0.2 * y_a = mixture_a →
  1 - x_a = 0.7 := by
  sorry

end chemical_mixture_problem_l4022_402214


namespace complex_equation_solution_l4022_402270

theorem complex_equation_solution (x y : ℝ) :
  (2 * x - 1 : ℂ) + I = -y - (3 - y) * I →
  x = -3/2 ∧ y = 4 := by
sorry

end complex_equation_solution_l4022_402270


namespace linear_function_intersection_k_range_l4022_402250

-- Define the linear function
def linear_function (k : ℝ) (x : ℝ) : ℝ := k * x + (2 - 2 * k)

-- Define the intersection function
def intersection_function (x : ℝ) : ℝ := -x + 3

-- Define the domain
def in_domain (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 3

-- Theorem statement
theorem linear_function_intersection_k_range :
  ∀ k : ℝ, 
  (∃ x : ℝ, in_domain x ∧ linear_function k x = intersection_function x) →
  ((k ≤ -2 ∨ k ≥ -1/2) ∧ k ≠ 0) :=
sorry

end linear_function_intersection_k_range_l4022_402250


namespace not_p_sufficient_but_not_necessary_for_not_q_l4022_402267

-- Define the conditions p and q as predicates on real numbers
def p (x : ℝ) : Prop := |x + 1| > 2
def q (x : ℝ) : Prop := 5*x - 6 > x^2

-- Define what it means for one condition to be sufficient but not necessary for another
def sufficient_but_not_necessary (A B : Prop) : Prop :=
  (A → B) ∧ ¬(B → A)

-- Theorem statement
theorem not_p_sufficient_but_not_necessary_for_not_q :
  sufficient_but_not_necessary (¬∃ x, p x) (¬∃ x, q x) := by
  sorry

end not_p_sufficient_but_not_necessary_for_not_q_l4022_402267


namespace rectangle_y_value_l4022_402249

/-- Given a rectangle with vertices at (-2, y), (8, y), (-2, 2), and (8, 2),
    with an area of 100 square units and y > 2, prove that y = 12. -/
theorem rectangle_y_value (y : ℝ) 
    (h1 : (8 - (-2)) * (y - 2) = 100)  -- Area of rectangle is 100
    (h2 : y > 2) :                     -- y is greater than 2
  y = 12 := by
sorry

end rectangle_y_value_l4022_402249


namespace trigonometric_identity_l4022_402298

theorem trigonometric_identity (α : Real) : 
  Real.sin α ^ 2 + Real.cos (π / 6 + α) ^ 2 + Real.sin α * Real.cos (π / 6 + α) = 3 / 4 := by
  sorry

end trigonometric_identity_l4022_402298


namespace remainder_theorem_polynomial_remainder_l4022_402262

def f (x : ℝ) : ℝ := x^4 - 8*x^3 + 12*x^2 + 20*x - 18

theorem remainder_theorem (f : ℝ → ℝ) (a : ℝ) :
  ∃ q : ℝ → ℝ, ∀ x, f x = (x - a) * q x + f a := sorry

theorem polynomial_remainder : 
  ∃ q : ℝ → ℝ, ∀ x, f x = (x - 2) * q x + 22 := by
  sorry

end remainder_theorem_polynomial_remainder_l4022_402262


namespace min_value_theorem_l4022_402291

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 1/b = 2) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + 1/y = 2 → 4/x + y ≥ 9/2) ∧
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + 1/y = 2 ∧ 4/x + y = 9/2) :=
by sorry

end min_value_theorem_l4022_402291


namespace domain_of_g_l4022_402229

-- Define the function f with domain [0,4]
def f : Set ℝ := Set.Icc 0 4

-- Define the function g
def g (f : Set ℝ) : Set ℝ := {x | x ∈ f ∧ x^2 ∈ f}

-- Theorem statement
theorem domain_of_g (f : Set ℝ) (hf : f = Set.Icc 0 4) : 
  g f = Set.Icc 0 2 := by sorry

end domain_of_g_l4022_402229


namespace clinton_wardrobe_problem_l4022_402209

/-- Clinton's wardrobe problem -/
theorem clinton_wardrobe_problem (hats belts shoes : ℕ) :
  hats = 5 →
  belts = hats + 2 →
  shoes = 2 * belts →
  shoes = 14 := by sorry

end clinton_wardrobe_problem_l4022_402209


namespace trig_identity_l4022_402207

theorem trig_identity (θ : Real) (h : Real.sin θ + Real.cos θ = Real.sqrt 2) :
  Real.tan θ + (Real.tan θ)⁻¹ = 2 := by
  sorry

end trig_identity_l4022_402207


namespace smallest_n_divisible_sixty_satisfies_smallest_n_is_sixty_l4022_402248

theorem smallest_n_divisible (n : ℕ) : n > 0 ∧ 24 ∣ n^2 ∧ 480 ∣ n^3 → n ≥ 60 :=
by sorry

theorem sixty_satisfies : 24 ∣ 60^2 ∧ 480 ∣ 60^3 :=
by sorry

theorem smallest_n_is_sixty : ∃ (n : ℕ), n > 0 ∧ 24 ∣ n^2 ∧ 480 ∣ n^3 ∧ ∀ (m : ℕ), (m > 0 ∧ 24 ∣ m^2 ∧ 480 ∣ m^3) → m ≥ n :=
by sorry

end smallest_n_divisible_sixty_satisfies_smallest_n_is_sixty_l4022_402248


namespace tangent_slope_implies_n_value_l4022_402272

/-- The function f(x) defined as x^n + 3^x --/
noncomputable def f (n : ℝ) (x : ℝ) : ℝ := x^n + 3^x

/-- The derivative of f(x) --/
noncomputable def f_derivative (n : ℝ) (x : ℝ) : ℝ := n * x^(n-1) + 3^x * Real.log 3

theorem tangent_slope_implies_n_value (n : ℝ) :
  f n 1 = 4 →
  f_derivative n 1 = 3 + 3 * Real.log 3 →
  n = 3 := by
  sorry

end tangent_slope_implies_n_value_l4022_402272


namespace paint_problem_l4022_402243

theorem paint_problem (initial_paint : ℚ) : 
  initial_paint = 1 →
  let first_day_used := initial_paint / 2
  let first_day_remaining := initial_paint - first_day_used
  let second_day_first_op := first_day_remaining / 4
  let second_day_mid_remaining := first_day_remaining - second_day_first_op
  let second_day_second_op := second_day_mid_remaining / 8
  let final_remaining := second_day_mid_remaining - second_day_second_op
  final_remaining = (21 : ℚ) / 64 * initial_paint :=
by sorry

end paint_problem_l4022_402243


namespace victor_insect_stickers_l4022_402240

/-- The number of insect stickers Victor has -/
def insect_stickers (flower_stickers : ℝ) (total_stickers : ℝ) : ℝ :=
  total_stickers - (2 * flower_stickers - 3.5) - (1.5 * flower_stickers + 5.5)

theorem victor_insect_stickers :
  insect_stickers 15 70 = 15.5 := by sorry

end victor_insect_stickers_l4022_402240


namespace intersection_sum_zero_l4022_402266

-- Define the parabolas
def parabola1 (x y : ℝ) : Prop := y = (x - 2)^2 + 1
def parabola2 (x y : ℝ) : Prop := x - 1 = (y + 2)^2

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | parabola1 p.1 p.2 ∧ parabola2 p.1 p.2}

-- Theorem statement
theorem intersection_sum_zero :
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ),
    (x₁, y₁) ∈ intersection_points ∧
    (x₂, y₂) ∈ intersection_points ∧
    (x₃, y₃) ∈ intersection_points ∧
    (x₄, y₄) ∈ intersection_points ∧
    (x₁, y₁) ≠ (x₂, y₂) ∧
    (x₁, y₁) ≠ (x₃, y₃) ∧
    (x₁, y₁) ≠ (x₄, y₄) ∧
    (x₂, y₂) ≠ (x₃, y₃) ∧
    (x₂, y₂) ≠ (x₄, y₄) ∧
    (x₃, y₃) ≠ (x₄, y₄) ∧
    x₁ + x₂ + x₃ + x₄ + y₁ + y₂ + y₃ + y₄ = 0 :=
by sorry

end intersection_sum_zero_l4022_402266


namespace farm_animal_ratio_l4022_402228

theorem farm_animal_ratio (total animals goats cows pigs : ℕ) : 
  total = 56 ∧ 
  goats = 11 ∧ 
  cows = goats + 4 ∧ 
  total = pigs + cows + goats → 
  pigs * 1 = cows * 2 := by
  sorry

end farm_animal_ratio_l4022_402228


namespace quadratic_minimum_l4022_402287

theorem quadratic_minimum : 
  (∀ x : ℝ, x^2 + 10*x ≥ -25) ∧ (∃ x : ℝ, x^2 + 10*x = -25) := by
  sorry

end quadratic_minimum_l4022_402287


namespace arithmetic_sequence_fifth_term_l4022_402225

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem to be proved -/
theorem arithmetic_sequence_fifth_term
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_sum : a 2 + a 4 = 16)
  (h_first : a 1 = 1) :
  a 5 = 15 := by
  sorry

end arithmetic_sequence_fifth_term_l4022_402225


namespace tank_filling_capacity_l4022_402276

/-- Given a tank that can be filled with 28 buckets of 13.5 litres each,
    prove that if the same tank can be filled with 42 buckets of equal capacity,
    then the capacity of each bucket in the second case is 9 litres. -/
theorem tank_filling_capacity (tank_volume : ℝ) (bucket_count_1 bucket_count_2 : ℕ) 
    (bucket_capacity_1 : ℝ) :
  tank_volume = bucket_count_1 * bucket_capacity_1 →
  bucket_count_1 = 28 →
  bucket_capacity_1 = 13.5 →
  bucket_count_2 = 42 →
  ∃ bucket_capacity_2 : ℝ, 
    tank_volume = bucket_count_2 * bucket_capacity_2 ∧
    bucket_capacity_2 = 9 := by
  sorry


end tank_filling_capacity_l4022_402276


namespace cyclic_sum_inequality_l4022_402206

theorem cyclic_sum_inequality (x y z : ℝ) (hpos_x : x > 0) (hpos_y : y > 0) (hpos_z : z > 0)
  (h_sum_squares : x^2 + y^2 + z^2 = 1) :
  (x*y/z) + (y*z/x) + (z*x/y) ≥ Real.sqrt 3 := by
  sorry

end cyclic_sum_inequality_l4022_402206


namespace german_enrollment_l4022_402286

theorem german_enrollment (total_students : ℕ) (both_subjects : ℕ) (only_english : ℕ) 
  (h1 : total_students = 45)
  (h2 : both_subjects = 12)
  (h3 : only_english = 23)
  (h4 : total_students = only_english + both_subjects + (total_students - (only_english + both_subjects))) :
  total_students - (only_english + both_subjects) + both_subjects = 22 := by
  sorry

end german_enrollment_l4022_402286


namespace square_ratio_problem_l4022_402244

theorem square_ratio_problem (area_ratio : ℚ) (a b c : ℕ) :
  area_ratio = 75 / 48 →
  (a : ℚ) * Real.sqrt b / c = Real.sqrt (area_ratio) →
  a + b + c = 10 :=
by sorry

end square_ratio_problem_l4022_402244


namespace hyperbola_asymptote_l4022_402247

/-- Given a hyperbola with equation x^2 - y^2/b^2 = 1 where b > 0,
    if one of its asymptotic lines is y = 2x, then b = 2 -/
theorem hyperbola_asymptote (b : ℝ) (h1 : b > 0) :
  (∃ (x y : ℝ), x^2 - y^2/b^2 = 1 ∧ y = 2*x) → b = 2 :=
by sorry

end hyperbola_asymptote_l4022_402247


namespace yellow_packs_bought_l4022_402273

/-- The number of bouncy balls in each package -/
def balls_per_pack : ℕ := 10

/-- The total number of bouncy balls Maggie kept -/
def total_balls : ℕ := 80

/-- The number of packs of green bouncy balls given away -/
def green_packs_given : ℕ := 4

/-- The number of packs of green bouncy balls bought -/
def green_packs_bought : ℕ := 4

/-- The theorem stating the number of packs of yellow bouncy balls Maggie bought -/
theorem yellow_packs_bought : 
  (total_balls / balls_per_pack : ℕ) = 8 :=
sorry

end yellow_packs_bought_l4022_402273


namespace complex_equation_solution_l4022_402260

theorem complex_equation_solution (z : ℂ) (h : z * (1 - Complex.I) = 2) : z = 1 + Complex.I := by
  sorry

end complex_equation_solution_l4022_402260


namespace vector_collinearity_l4022_402265

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v = (k * w.1, k * w.2) ∨ w = (k * v.1, k * v.2)

theorem vector_collinearity :
  let m : ℝ × ℝ := (0, -2)
  let n : ℝ × ℝ := (Real.sqrt 3, 1)
  let v : ℝ × ℝ := (-1, Real.sqrt 3)
  collinear (2 * m.1 + n.1, 2 * m.2 + n.2) v := by sorry

end vector_collinearity_l4022_402265


namespace seven_sided_die_perfect_square_probability_l4022_402234

/-- Represents a fair seven-sided die with faces numbered 1 through 7 -/
def SevenSidedDie : Finset ℕ := {1, 2, 3, 4, 5, 6, 7}

/-- Checks if a number is a perfect square -/
def isPerfectSquare (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

/-- The number of times the die is rolled -/
def numRolls : ℕ := 4

/-- The total number of possible outcomes when rolling the die numRolls times -/
def totalOutcomes : ℕ := SevenSidedDie.card ^ numRolls

/-- The number of favorable outcomes (product of rolls is a perfect square) -/
def favorableOutcomes : ℕ := 164

theorem seven_sided_die_perfect_square_probability :
  (favorableOutcomes : ℚ) / totalOutcomes = 164 / 2401 :=
sorry

end seven_sided_die_perfect_square_probability_l4022_402234


namespace divisibility_999_from_50_l4022_402200

/-- A function that extracts 50 consecutive digits from a 999-digit number starting at a given index -/
def extract_50_digits (n : ℕ) (start_index : ℕ) : ℕ := sorry

/-- Predicate to check if a number is a valid 999-digit number -/
def is_999_digit_number (n : ℕ) : Prop := sorry

theorem divisibility_999_from_50 (n : ℕ) (h1 : is_999_digit_number n)
  (h2 : ∀ i, i ≤ 950 → extract_50_digits n i % 2^50 = 0) :
  n % 2^999 = 0 := by sorry

end divisibility_999_from_50_l4022_402200


namespace chessboard_coloring_count_l4022_402241

/-- The number of ways to paint an N × N chessboard with 4 colors such that:
    1) Squares with a common side are painted with distinct colors
    2) Every 2 × 2 square is painted with the four colors -/
def chessboardColorings (N : ℕ) : ℕ := 24 * (2^(N-1) - 1)

/-- Theorem stating the number of valid colorings for an N × N chessboard -/
theorem chessboard_coloring_count (N : ℕ) (h : N > 1) : 
  chessboardColorings N = 24 * (2^(N-1) - 1) := by
  sorry


end chessboard_coloring_count_l4022_402241


namespace rectangle_area_equals_44_l4022_402288

/-- Given a triangle with sides a, b, c and a rectangle with one side length d,
    if the perimeters are equal and d = 8, then the area of the rectangle is 44. -/
theorem rectangle_area_equals_44 (a b c d : ℝ) : 
  a = 7.5 → b = 9 → c = 10.5 → d = 8 → 
  a + b + c = 2 * (d + (a + b + c) / 2 - d) → 
  d * ((a + b + c) / 2 - d) = 44 := by sorry

end rectangle_area_equals_44_l4022_402288
