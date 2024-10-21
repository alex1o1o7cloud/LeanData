import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_sum_triple_l990_99032

theorem divisor_sum_triple (a b n : ℕ) : 
  a > 0 → b > 0 → n > 0 →
  a ∣ n → b ∣ n → a + b = n / 2 → 
  (∃ t : ℕ, t > 0 ∧ ((a = t ∧ b = t ∧ n = 4 * t) ∨ 
             (a = 2 * t ∧ b = t ∧ n = 6 * t) ∨ 
             (a = t ∧ b = 2 * t ∧ n = 6 * t))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_sum_triple_l990_99032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_perimeter_l990_99088

/-- An isosceles right-angled triangle with specific properties -/
structure IsoscelesRightTriangle where
  /-- The shorter leg of the triangle -/
  a : ℝ
  /-- The longer leg of the triangle -/
  b : ℝ
  /-- The area of the triangle is 50 square units -/
  area_eq : a * b / 2 = 50
  /-- The ratio of the shorter leg to the longer leg is 3:5 -/
  leg_ratio : a / b = 3 / 5
  /-- The triangle is right-angled (Pythagorean theorem) -/
  right_angled : a^2 + b^2 = (a + b)^2 / 2

/-- The perimeter of the isosceles right-angled triangle -/
noncomputable def perimeter (t : IsoscelesRightTriangle) : ℝ :=
  t.a + t.b + Real.sqrt (t.a^2 + t.b^2)

/-- Theorem: The perimeter of the specified isosceles right-angled triangle is approximately 29.78 -/
theorem isosceles_right_triangle_perimeter :
  ∃ t : IsoscelesRightTriangle, |perimeter t - 29.78| < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_perimeter_l990_99088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_price_calculation_l990_99012

/-- Given a selling price and a profit percentage, calculate the cost price. -/
noncomputable def calculate_cost_price (selling_price : ℝ) (profit_percentage : ℝ) : ℝ :=
  selling_price / (1 + profit_percentage / 100)

/-- Theorem: The cost price of an article sold for $100 with 30% profit is approximately $76.92. -/
theorem cost_price_calculation :
  let selling_price : ℝ := 100
  let profit_percentage : ℝ := 30
  let cost_price := calculate_cost_price selling_price profit_percentage
  ∃ ε > 0, |cost_price - 76.92| < ε :=
by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval calculate_cost_price 100 30

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_price_calculation_l990_99012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_l990_99004

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the midpoint of AB
def midpoint_AB (x y : ℝ) : Prop := x = 3 ∧ y = 2

-- Define the line L
def line_L (x y : ℝ) : Prop := x - y - 1 = 0

-- Define the length of AB
def length_AB : ℝ := 8

-- Theorem statement
theorem parabola_line_intersection :
  ∀ (x_A y_A x_B y_B : ℝ),
  parabola x_A y_A ∧ parabola x_B y_B ∧  -- A and B are on the parabola
  line_L x_A y_A ∧ line_L x_B y_B ∧      -- A and B are on line L
  midpoint_AB ((x_A + x_B)/2) ((y_A + y_B)/2) → -- Midpoint of AB is (3, 2)
  (∀ x y : ℝ, line_L x y ↔ x - y - 1 = 0) ∧  -- Equation of line L
  Real.sqrt ((x_A - x_B)^2 + (y_A - y_B)^2) = length_AB  -- Length of AB
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_l990_99004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_brokers_commission_percentage_l990_99091

theorem brokers_commission_percentage 
  (original_price : ℝ) 
  (selling_price : ℝ) 
  (profit_percentage : ℝ) 
  (h1 : original_price = 80000)
  (h2 : selling_price = 100000)
  (h3 : profit_percentage = 20) :
  (selling_price - original_price * (1 + profit_percentage / 100)) / original_price * 100 = 5 := by
  sorry

-- Remove the #eval line as it's causing issues with universe levels
-- #eval brokers_commission_percentage 80000 100000 20

end NUMINAMATH_CALUDE_ERRORFEEDBACK_brokers_commission_percentage_l990_99091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_migration_increases_both_ratings_reverse_migration_cannot_increase_both_three_country_double_migration_increases_all_l990_99096

structure Country where
  population : ℕ
  total_score : ℕ

def average_score (c : Country) : ℚ :=
  c.total_score / c.population

def migrate (from_country to_country : Country) (migrants : ℕ) (migrant_score : ℕ) : Country × Country :=
  let new_from : Country := {
    population := from_country.population - migrants,
    total_score := from_country.total_score - migrants * migrant_score
  }
  let new_to : Country := {
    population := to_country.population + migrants,
    total_score := to_country.total_score + migrants * migrant_score
  }
  (new_from, new_to)

theorem migration_increases_both_ratings (a b : Country) :
  ∃ (migrants : ℕ) (migrant_score : ℕ),
    let (new_a, new_b) := migrate a b migrants migrant_score
    average_score new_a > average_score a ∧
    average_score new_b > average_score b := by
  sorry

theorem reverse_migration_cannot_increase_both (a b : Country) :
  ∀ (migrants1 migrants2 : ℕ) (migrant_score1 migrant_score2 : ℕ),
    let (a', b') := migrate a b migrants1 migrant_score1
    let (b'', a'') := migrate b' a' migrants2 migrant_score2
    ¬(average_score a'' > average_score a' ∧
      average_score b'' > average_score b') := by
  sorry

theorem three_country_double_migration_increases_all (a b c : Country) :
  ∃ (migrants_ab migrants_bc migrants_cb migrants_ba : ℕ)
    (score_ab score_bc score_cb score_ba : ℕ),
    let (a', b_temp) := migrate a b migrants_ab score_ab
    let (b', c') := migrate b_temp c migrants_bc score_bc
    let (c'', b_temp') := migrate c' b' migrants_cb score_cb
    let (b'', a'') := migrate b_temp' a' migrants_ba score_ba
    average_score a'' > average_score a' ∧
    average_score b'' > average_score b' ∧
    average_score c'' > average_score c' := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_migration_increases_both_ratings_reverse_migration_cannot_increase_both_three_country_double_migration_increases_all_l990_99096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_eq_neg_half_l990_99075

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then 1/2 * x - 1 else 1/x

theorem f_composition_eq_neg_half (a : ℝ) :
  f (f a) = -1/2 → a = 4 ∨ a = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_eq_neg_half_l990_99075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_symmetric_l990_99064

noncomputable def f (x : ℝ) : ℝ := Real.sin x * Real.cos x

theorem f_sum_symmetric : f (-1) + f 1 = 0 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_symmetric_l990_99064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_sum_no_small_prime_factors_counterexample_for_38_l990_99041

/-- A function that checks if a number has no prime factors less than 37 -/
def no_small_prime_factors (n : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → p < 37 → ¬(p ∣ n)

/-- The main theorem to be proved -/
theorem exists_sum_no_small_prime_factors 
  (S : Finset ℕ) 
  (h1 : S.card = 2019)
  (h2 : ∀ n ∈ S, n > 0 ∧ no_small_prime_factors n)
  (h3 : ∀ a b, a ∈ S → b ∈ S → a ≠ b) :
  ∃ a b, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ no_small_prime_factors (a + b) :=
sorry

/-- The counterexample for 38 -/
theorem counterexample_for_38
  (S : Finset ℕ)
  (h1 : S.card = 2019)
  (h2 : ∀ n ∈ S, n > 0 ∧ (∀ p : ℕ, Nat.Prime p → p < 38 → ¬(p ∣ n))) :
  ¬(∀ a b, a ∈ S → b ∈ S → a ≠ b → 
    (∀ p : ℕ, Nat.Prime p → p < 38 → ¬(p ∣ (a + b)))) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_sum_no_small_prime_factors_counterexample_for_38_l990_99041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_expression_equality_l990_99056

theorem trigonometric_expression_equality (a : Real) 
  (h1 : Real.tan (π / 6 - a) = -2)
  (h2 : a ∈ Set.Icc (π / 6) ((7 / 6) * π)) :
  let α := a + π / 3
  Real.sin (α / 2) * Real.cos (α / 2) + Real.sqrt 3 * (Real.cos (α / 2))^2 - Real.sqrt 3 / 2 = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_expression_equality_l990_99056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_painting_rate_is_three_l990_99083

-- Define the parameters
noncomputable def length : ℝ := 22
noncomputable def total_cost : ℝ := 484

-- Define the relationship between length and breadth
noncomputable def breadth : ℝ := length / 3

-- Define the area of the floor
noncomputable def area : ℝ := length * breadth

-- Define the painting rate
noncomputable def painting_rate : ℝ := total_cost / area

-- Theorem to prove
theorem painting_rate_is_three : 
  ∀ ε > 0, |painting_rate - 3| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_painting_rate_is_three_l990_99083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_perimeter_rectangle_l990_99028

-- Define the curve C
def curve_C (x y : ℝ) : Prop := x^2 / 3 + y^2 = 1

-- Define point R
def point_R : ℝ × ℝ := (2, 2)

-- Define the rectangle PQRS
def rectangle_PQRS (P : ℝ × ℝ) : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) :=
  let (px, py) := P
  let (rx, ry) := point_R
  ((px, py), (rx, py), point_R, (px, ry))

-- Calculate the perimeter of the rectangle
def perimeter (P : ℝ × ℝ) : ℝ :=
  let (px, py) := P
  let (rx, ry) := point_R
  2 * (abs (rx - px) + abs (ry - py))

-- Theorem statement
theorem min_perimeter_rectangle :
  ∃ (P : ℝ × ℝ),
    curve_C P.1 P.2 ∧
    perimeter P = 4 ∧
    (∀ (Q : ℝ × ℝ), curve_C Q.1 Q.2 → perimeter Q ≥ 4) ∧
    P = (3/2, 1/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_perimeter_rectangle_l990_99028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_unchanged_l990_99068

noncomputable def q (e x z : ℝ) : ℝ := 5 * e / (4 * x * z^2)

noncomputable def q_new (e x z : ℝ) : ℝ := 0.2222222222222222 * q (4 * e) (2 * x) z

theorem z_unchanged (e x z : ℝ) (h : x ≠ 0) (h' : z ≠ 0) : 
  q_new e x z = q e x z := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_unchanged_l990_99068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_leap_day_2024_l990_99046

/-- Represents days of the week -/
inductive DayOfWeek
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday
| Sunday

/-- Calculates the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday
  | DayOfWeek.Sunday => DayOfWeek.Monday

/-- Calculates the day of the week after a given number of days -/
def dayAfter (start : DayOfWeek) (days : Nat) : DayOfWeek :=
  match days with
  | 0 => start
  | n + 1 => dayAfter (nextDay start) n

theorem leap_day_2024 (leap_day_2000 : DayOfWeek) 
  (h : leap_day_2000 = DayOfWeek.Tuesday) :
  dayAfter leap_day_2000 8766 = DayOfWeek.Thursday :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_leap_day_2024_l990_99046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dissatisfied_passengers_properties_l990_99079

/-- Represents the probability distribution of dissatisfied passengers -/
def DissatisfiedPassengers (n : ℕ) : Type :=
  Fin (n + 1) → ℝ

/-- The probability that a passenger prefers chicken (or fish) -/
noncomputable def preference_probability : ℝ := 1 / 2

/-- The most likely number of dissatisfied passengers -/
def most_likely_dissatisfied (n : ℕ) : ℕ := 1

/-- The expected number of dissatisfied passengers -/
noncomputable def expected_dissatisfied (n : ℕ) : ℝ :=
  Real.sqrt (n / Real.pi)

/-- The variance of the number of dissatisfied passengers -/
noncomputable def variance_dissatisfied (n : ℕ) : ℝ :=
  0.182 * (n : ℝ)

/-- Main theorem about dissatisfied passengers -/
theorem dissatisfied_passengers_properties (n : ℕ) (h : n > 1) :
  ∃ (d : DissatisfiedPassengers n),
    (∀ k : Fin (n + 1), d k ≤ d (Fin.ofNat (most_likely_dissatisfied n))) ∧
    (|expected_dissatisfied n - Real.sqrt (n / Real.pi)| < 1 / n) ∧
    (|variance_dissatisfied n - 0.182 * n| < 1 / (n : ℝ)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dissatisfied_passengers_properties_l990_99079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l990_99013

-- Define the curve
def f (x : ℝ) : ℝ := x^3

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3 * x^2

-- Define the point of tangency
def tangentPoint : ℝ × ℝ := (1, 1)

-- Define the slope of the tangent line at the point
def tangentSlope : ℝ := f' tangentPoint.1

-- Theorem statement
theorem tangent_line_equation :
  ∀ x y : ℝ, y = tangentSlope * (x - tangentPoint.1) + tangentPoint.2 ↔ y = 3*x - 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l990_99013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_5614_to_hundredth_l990_99039

/-- Rounds a real number to the nearest hundredth -/
noncomputable def roundToHundredth (x : ℝ) : ℝ :=
  ⌊x * 100 + 0.5⌋ / 100

/-- The problem statement -/
theorem round_5614_to_hundredth :
  roundToHundredth 5.614 = 5.61 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_5614_to_hundredth_l990_99039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_AEKS_l990_99061

/-- The area of an equilateral triangle with side length 1 -/
noncomputable def unit_equilateral_area : ℝ := Real.sqrt 3 / 4

/-- The area of triangle AES in terms of unit equilateral triangles -/
noncomputable def area_AES : ℝ := 16 * unit_equilateral_area

/-- The area of triangle EKJ in terms of unit equilateral triangles -/
noncomputable def area_EKJ : ℝ := unit_equilateral_area

/-- The area of parallelogram KJST in terms of unit equilateral triangles -/
noncomputable def area_KJST : ℝ := 6 * unit_equilateral_area

/-- The area of triangle KJS is half the area of parallelogram KJST -/
noncomputable def area_KJS : ℝ := area_KJST / 2

/-- The theorem stating that the area of polygon AEKS is 5√3 -/
theorem area_AEKS : area_AES + area_EKJ + area_KJS = 5 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_AEKS_l990_99061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_roll_odd_is_half_l990_99090

/-- Represents an 8-sided die with dots numbered from 1 to 8 -/
structure Die where
  sides : Fin 8 → ℕ
  initial_dots : ∀ i, sides i = i.val + 1

/-- The total number of dots on the die -/
def total_dots (d : Die) : ℕ :=
  (Finset.sum Finset.univ (fun i => d.sides i))

/-- The probability of removing a single dot -/
def prob_remove_dot (d : Die) : ℚ :=
  2 / (total_dots d)

/-- The probability that a face with n dots ends up with an odd number of dots -/
def prob_odd_face (d : Die) (n : ℕ) : ℚ :=
  if n % 2 = 1
  then prob_remove_dot d * (1 - prob_remove_dot d)
  else (1 - prob_remove_dot d)

/-- The probability of rolling an odd number of dots after removing two dots -/
def prob_roll_odd (d : Die) : ℚ :=
  (1 / 8) * (Finset.sum Finset.univ (fun i => prob_odd_face d (d.sides i)))

theorem prob_roll_odd_is_half (d : Die) : prob_roll_odd d = 1/2 := by
  sorry

#check prob_roll_odd_is_half

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_roll_odd_is_half_l990_99090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_value_l990_99026

-- Define the function f
noncomputable def f : ℝ → ℝ := fun x => 
  if x > 0 then x * (1 - x) else -(-x * (1 - (-x)))

-- State the theorem
theorem odd_function_value : f (-3) = 6 := by
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_value_l990_99026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_train_speed_is_40_l990_99022

/-- Calculates the speed of the second train given the lengths of two trains,
    the speed of the first train, and the time they take to cross each other. -/
noncomputable def second_train_speed (length1 length2 : ℝ) (speed1 time_to_cross : ℝ) : ℝ :=
  let total_length := length1 + length2
  let total_length_km := total_length / 1000
  let time_to_cross_hours := time_to_cross / 3600
  let relative_speed := total_length_km / time_to_cross_hours
  relative_speed - speed1

/-- Theorem stating that under the given conditions, the speed of the second train is 40 km/hr. -/
theorem second_train_speed_is_40 :
  second_train_speed 180 160 60 12.239020878329734 = 40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_train_speed_is_40_l990_99022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_game_result_l990_99098

-- Define the number of participants
def num_participants : ℕ := 44

-- Define the operation for the first calculator (square root)
noncomputable def op1 (x : ℝ) : ℝ := Real.sqrt x

-- Define the operation for the second calculator (square)
def op2 (x : ℝ) : ℝ := x ^ 2

-- Define the operation for the third calculator (factorial for non-negative integers)
def op3 (x : ℤ) : ℤ :=
  if x ≥ 0 then Nat.factorial x.toNat else x

-- Define the final state of the calculators after all operations
noncomputable def final_state (n : ℕ) : ℝ × ℝ × ℤ :=
  (op1 2, op2 0, op3 (-2))

-- Theorem statement
theorem game_result :
  let (a, b, c) := final_state num_participants
  a + b + c = Real.sqrt 2 - 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_game_result_l990_99098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_disprove_claim_with_non_prime_l990_99099

-- Define the set of cards
def Card : Type := ℕ × Char

-- Define the set of visible cards
def visibleCards : List Card := [(1, 'X'), (2, 'X'), (4, 'X'), (8, 'X'), (0, 'A'), (0, 'B'), (0, 'S')]

-- Define what a consonant is (simplified for this problem)
def isConsonant (c : Char) : Prop := c = 'B' ∨ c = 'S'

-- Define primality
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

-- Tom's claim
def tomClaim (card : Card) : Prop :=
  isConsonant (card.snd) → isPrime card.fst

-- Define a function to check if a card disproves Tom's claim
def disprovesClaim (card : Card) : Prop :=
  isConsonant (card.snd) ∧ ¬(isPrime card.fst)

-- Theorem to prove
theorem disprove_claim_with_non_prime :
  ∀ (card : Card),
    (card ∈ visibleCards ∧ card.fst > 2 ∧ ¬(isPrime card.fst)) ↔
    (∃ (c : Char), disprovesClaim (card.fst, c)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_disprove_claim_with_non_prime_l990_99099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_composition_l990_99087

-- Define the function g and its inverse
def g : ℝ → ℝ := sorry

-- Define g^(-1) as the inverse of g
noncomputable def g_inv : ℝ → ℝ := Function.invFun g

-- State the given conditions
axiom g_4 : g 4 = 7
axiom g_6 : g 6 = 2
axiom g_3 : g 3 = 6

-- State that g_inv is the left inverse of g
axiom g_inv_left : Function.LeftInverse g_inv g

-- Theorem to prove
theorem inverse_composition : g_inv (g_inv 6 + g_inv 7) = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_composition_l990_99087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_empty_proper_subset_implies_a_nonnegative_l990_99029

def S (a : ℝ) := {x : ℝ | x^2 ≤ a}

theorem empty_proper_subset_implies_a_nonnegative (a : ℝ) :
  (∅ : Set ℝ).Subset (S a) ∧ ∅ ≠ S a → a ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_empty_proper_subset_implies_a_nonnegative_l990_99029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_is_three_l990_99082

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then 2^(-x) else Real.log x / Real.log 81

-- Theorem statement
theorem unique_solution_is_three :
  ∃! x, f x = 1/4 ∧ x = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_is_three_l990_99082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonically_decreasing_power_function_l990_99019

noncomputable def f (n : ℝ) (x : ℝ) : ℝ := (n^2 - 3*n + 3) * x^(n^2 - 2*n)

theorem monotonically_decreasing_power_function :
  ∃! n : ℝ, 
    (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ → f n x₁ > f n x₂) ∧ 
    (n^2 - 3*n + 3 > 0) ∧ 
    (n^2 - 2*n < 0) ∧
    n = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonically_decreasing_power_function_l990_99019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l990_99036

-- Define the hyperbola C
def hyperbola (a b x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

-- Define a point on the hyperbola
structure PointOnHyperbola (a b : ℝ) where
  x : ℝ
  y : ℝ
  on_hyperbola : hyperbola a b x y

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the eccentricity of a hyperbola
noncomputable def eccentricity (a c : ℝ) : ℝ := c / a

theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (M : PointOnHyperbola a b) (F : ℝ × ℝ) (P Q : ℝ × ℝ) (circle : Circle) :
  -- Circle center is on the hyperbola
  circle.center = (M.x, M.y) →
  -- Circle is tangent to x-axis at focus F
  F.2 = 0 ∧ circle.radius = F.1 - M.x →
  -- Circle intersects y-axis at P and Q
  P.1 = 0 ∧ Q.1 = 0 →
  -- Triangle MPQ is equilateral
  Real.sqrt ((M.x - P.1)^2 + (M.y - P.2)^2) = 
  Real.sqrt ((M.x - Q.1)^2 + (M.y - Q.2)^2) ∧
  Real.sqrt ((M.x - P.1)^2 + (M.y - P.2)^2) = 
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) →
  -- The eccentricity of the hyperbola is √3
  eccentricity a F.1 = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l990_99036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_theta_value_l990_99025

-- Define the vectors and angle
variable (a b c d : Euclidean 3)
variable (θ : ℝ)

-- State the conditions
axiom nonzero_vectors : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0
axiom not_parallel : ¬(∃ (k : ℝ), a = k • b ∨ a = k • c ∨ a = k • d ∨ b = k • c ∨ b = k • d ∨ c = k • d)
axiom vector_equation : (a.cross b).cross (c.cross d) = (1/4 : ℝ) * ‖b‖ * ‖d‖ * a
axiom angle_definition : θ = Real.arccos ((b.dot d) / (‖b‖ * ‖d‖))

-- State the theorem
theorem sin_theta_value : Real.sin θ = Real.sqrt 15 / 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_theta_value_l990_99025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_expr3_is_fraction_l990_99048

/-- Definition of a fraction -/
def is_fraction (f : ℝ → ℝ) : Prop :=
  ∃ (n d : ℝ → ℝ), ∀ x, f x = (n x) / (d x) ∧ d x ≠ 0 ∧ (∃ y, d y ≠ d 0)

/-- The given expressions -/
noncomputable def expr1 (x : ℝ) : ℝ := x / 2
noncomputable def expr2 (x : ℝ) : ℝ := x / Real.pi
noncomputable def expr3 (x : ℝ) : ℝ := x / (x + 1)
noncomputable def expr4 (x y : ℝ) : ℝ := x / 2 + y

theorem only_expr3_is_fraction :
  ¬ is_fraction expr1 ∧
  ¬ is_fraction expr2 ∧
  is_fraction expr3 ∧
  ¬ is_fraction (λ x => expr4 x 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_expr3_is_fraction_l990_99048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jerrys_feathers_correct_l990_99063

def jerrys_feathers (hawk_feathers : ℕ) (eagle_ratio : ℕ) (remaining_feathers : ℕ) : ℕ :=
  let total_feathers := hawk_feathers + hawk_feathers * eagle_ratio
  let before_selling := remaining_feathers * 2
  total_feathers - before_selling

theorem jerrys_feathers_correct (hawk_feathers eagle_ratio remaining_feathers : ℕ) :
  jerrys_feathers hawk_feathers eagle_ratio remaining_feathers =
  hawk_feathers + hawk_feathers * eagle_ratio - remaining_feathers * 2 := by
  simp [jerrys_feathers]

#eval jerrys_feathers 6 17 49

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jerrys_feathers_correct_l990_99063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_year_to_target_population_population_closest_to_target_l990_99002

/-- The population growth function for Isola -/
def population (year : ℕ) : ℕ :=
  250 * (4 ^ ((year - 2000) / 30))

/-- The target population -/
def target_population : ℕ := 8000

/-- The year we want to prove is closest to the target population -/
def target_year : ℕ := 2090

/-- Theorem stating that 2090 is the closest year to when the population reaches 8000 -/
theorem closest_year_to_target_population :
  ∀ y : ℕ, y ≠ target_year →
    (population target_year).dist target_population ≤ (population y).dist target_population :=
by sorry

/-- Theorem stating that the population in 2090 is the closest to 8000 -/
theorem population_closest_to_target :
  (population target_year).dist target_population < target_population / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_year_to_target_population_population_closest_to_target_l990_99002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jana_walking_distance_l990_99086

/-- Given Jana's walking rate, calculate the distance walked in 10 minutes -/
def distance_walked (rate : ℚ) (time : ℚ) : ℚ :=
  rate * time

/-- Round a rational number to the nearest tenth -/
def round_to_tenth (x : ℚ) : ℚ :=
  ⌊(x * 10 + 1/2)⌋ / 10

theorem jana_walking_distance :
  let rate : ℚ := 1 / 24  -- 1 mile per 24 minutes
  let time : ℚ := 10      -- 10 minutes
  round_to_tenth (distance_walked rate time) = 4 / 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jana_walking_distance_l990_99086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_101_value_l990_99040

def sequence_a : ℕ → ℚ
  | 0 => 2  -- Define the base case for 0
  | n + 1 => (2 * sequence_a n + 1) / 2

theorem a_101_value : sequence_a 100 = 52 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_101_value_l990_99040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_theorem_l990_99050

/-- An equilateral triangle -/
structure EquilateralTriangle where
  /-- The side length of the equilateral triangle -/
  side : ℝ
  /-- The side length is positive -/
  side_pos : side > 0

/-- A point within an equilateral triangle -/
structure PointInTriangle (t : EquilateralTriangle) where
  /-- The distances from the point to the three sides of the triangle -/
  distances : Fin 3 → ℝ
  /-- All distances are non-negative -/
  distances_nonneg : ∀ i, distances i ≥ 0
  /-- The sum of distances equals the altitude of the triangle -/
  sum_distances : (distances 0) + (distances 1) + (distances 2) = t.side * Real.sqrt 3 / 2

/-- The probability that the distances from a random point in an equilateral triangle
    to its sides form a triangle -/
noncomputable def probability_distances_form_triangle (t : EquilateralTriangle) : ℝ :=
  1 / 4

/-- The main theorem: The probability that the distances from a random point
    in an equilateral triangle to its sides form a triangle is 1/4 -/
theorem probability_theorem (t : EquilateralTriangle) :
  probability_distances_form_triangle t = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_theorem_l990_99050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l990_99005

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 6*x + 5

-- Define the interval
def interval : Set ℝ := Set.Icc (-2) 2

-- Theorem statement
theorem f_properties :
  -- Tangent line equation at x=1
  (∀ y : ℝ, (3 : ℝ) * 1 - y - 3 = 0 ↔ y = f 1 + (deriv f) 1 * (1 - 1)) ∧
  -- Maximum value in [-2, 2]
  (∃ x ∈ interval, ∀ y ∈ interval, f y ≤ f x) ∧
  (∃ x ∈ interval, f x = 5 + 4 * Real.sqrt 2) ∧
  -- Minimum value in [-2, 2]
  (∃ x ∈ interval, ∀ y ∈ interval, f x ≤ f y) ∧
  (∃ x ∈ interval, f x = 5 - 4 * Real.sqrt 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l990_99005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_pear_equivalence_l990_99015

/-- Represents the value of an apple in terms of pears -/
def apple_value : ℚ → ℚ := sorry

/-- Given that 2/3 of 12 apples are worth 10 pears, 
    prove that 1/3 of 6 apples are worth 2.5 pears -/
theorem apple_pear_equivalence (h : apple_value (2/3 * 12) = 10) :
  apple_value (1/3 * 6) = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_pear_equivalence_l990_99015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_traces_ellipse_l990_99080

/-- Triangle with vertices A, B, and C -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The centroid of a triangle -/
noncomputable def centroid (t : Triangle) : ℝ × ℝ :=
  ((t.A.1 + t.B.1 + t.C.1) / 3, (t.A.2 + t.B.2 + t.C.2) / 3)

/-- The diameter of the semi-circle -/
noncomputable def diameter : ℝ → ℝ := id

/-- Predicate to check if a point is on the semi-circle -/
def on_semi_circle (p : ℝ × ℝ) (d : ℝ) : Prop :=
  p.1^2 + p.2^2 = (d/2)^2 ∧ p.2 ≥ 0

/-- The path traced by the centroid -/
def centroid_path (d : ℝ) : Set (ℝ × ℝ) :=
  { p | ∃ t : Triangle, t.A = (0, 0) ∧ t.B = (d, 0) ∧ on_semi_circle t.C d ∧ p = centroid t }

/-- An ellipse centered at the origin -/
def ellipse (a b : ℝ) : Set (ℝ × ℝ) :=
  { p | (p.1 / a)^2 + (p.2 / b)^2 = 1 }

theorem centroid_traces_ellipse (d : ℝ) (h : d > 0) :
  centroid_path d = ellipse (d/3) (d/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_traces_ellipse_l990_99080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_and_constant_product_l990_99003

/-- An ellipse with given eccentricity and maximum distance to focus -/
structure Ellipse where
  a : ℝ
  b : ℝ
  e : ℝ
  max_dist : ℝ
  h1 : a > b
  h2 : b > 0
  h3 : e = 1/2
  h4 : max_dist = 3

/-- Point on the ellipse -/
structure PointOnEllipse (E : Ellipse) where
  x : ℝ
  y : ℝ
  h : (x^2 / E.a^2) + (y^2 / E.b^2) = 1

theorem ellipse_equation_and_constant_product (E : Ellipse) :
  (E.a = 2 ∧ E.b^2 = 3) ∧
  ∀ (A B P : PointOnEllipse E),
    A.y = -B.y ∧ A ≠ B ∧ P ≠ A ∧ P ≠ B →
    let M := (P.x * A.y - A.x * P.y) / (A.y - P.y)
    let N := (P.x * A.y + A.x * P.y) / (A.y + P.y)
    |M| * |N| = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_and_constant_product_l990_99003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_time_is_five_total_distance_covered_equals_initial_distance_l990_99097

/-- The time (in minutes) it takes for two people to meet, given their initial distance and movement patterns. -/
noncomputable def meetingTime (initialDistance : ℝ) (aSpeed : ℝ) (bInitialSpeed : ℝ) (bSpeedIncrease : ℝ) : ℝ :=
  5 -- The actual value is 5, but we'll prove this

/-- Theorem stating that under the given conditions, the meeting time is 5 minutes -/
theorem meeting_time_is_five :
  meetingTime 30 3 2 0.5 = 5 := by
  sorry

/-- Function to calculate the distance covered by person A in a given time -/
noncomputable def distanceCoveredByA (speed : ℝ) (time : ℝ) : ℝ :=
  speed * time

/-- Function to calculate the distance covered by person B in a given time -/
noncomputable def distanceCoveredByB (initialSpeed : ℝ) (speedIncrease : ℝ) (time : ℝ) : ℝ :=
  initialSpeed * time + speedIncrease * time * (time - 1) / 2

/-- Theorem stating that the total distance covered by both people after 5 minutes equals the initial distance -/
theorem total_distance_covered_equals_initial_distance :
  distanceCoveredByA 3 5 + distanceCoveredByB 2 0.5 5 = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_time_is_five_total_distance_covered_equals_initial_distance_l990_99097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_path_difference_approx_l990_99054

/-- Represents a rectangular field with given length and width -/
structure RectangularField where
  length : ℝ
  width : ℝ

/-- Calculates the distance of Jerry's path along the edges of the field -/
def jerryPath (field : RectangularField) : ℝ :=
  field.length + field.width

/-- Calculates the distance of Silvia's direct path using the Pythagorean theorem -/
noncomputable def silviaPath (field : RectangularField) : ℝ :=
  Real.sqrt (field.length ^ 2 + field.width ^ 2)

/-- Calculates the percentage difference between Jerry's and Silvia's paths -/
noncomputable def pathDifferencePercentage (field : RectangularField) : ℝ :=
  (jerryPath field - silviaPath field) / jerryPath field * 100

theorem path_difference_approx (field : RectangularField) 
    (h1 : field.length = 3) 
    (h2 : field.width = 4) : 
    ∃ ε > 0, |pathDifferencePercentage field - 100 * (2 / 7)| < ε ∧ ε < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_path_difference_approx_l990_99054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_distance_theorem_l990_99035

/-- The total horizontal distance traveled by the centers of two wheels -/
noncomputable def total_distance (r1 r2 : ℝ) : ℝ := 2 * Real.pi * r1 + 2 * Real.pi * r2

/-- Theorem: The total horizontal distance traveled by the centers of two wheels 
    with radii 1 m and 2 m, when rolled through one complete revolution on a flat surface, 
    is equal to 6π meters. -/
theorem wheel_distance_theorem : total_distance 1 2 = 6 * Real.pi := by
  unfold total_distance
  ring


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_distance_theorem_l990_99035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_1000_l990_99011

theorem divisors_of_1000 : 
  (Finset.filter (fun n => 1000 % n = 0) (Finset.range 1001)).card = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_1000_l990_99011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_of_f_l990_99055

-- Define the complex function f(z)
noncomputable def f (z : ℂ) : ℂ := (Complex.sin z)^4 / z^2

-- Define the set of zeros
def zeros : Set ℂ := {z | ∃ k : ℤ, k ≠ 0 ∧ z = k * Real.pi}

-- Theorem statement
theorem zeros_of_f :
  ∀ z ∈ zeros, f z = 0 ∧ (∃ g : ℂ → ℂ, ∃ ε > 0,
    (∀ w, Complex.abs (w - z) < ε → f w = g w * (w - z)^4) ∧
    g z ≠ 0) :=
by
  sorry

-- Additional lemma to support the main theorem
lemma sin_zero_at_pi_multiples (k : ℤ) (h : k ≠ 0) :
  Complex.sin (k * Real.pi) = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_of_f_l990_99055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_windmill_energy_production_l990_99058

/-- The amount of energy (in kW/h) produced by a single windmill on a typical day -/
def typical_energy : ℝ := 400

/-- The percentage increase in energy production on a stormy day -/
def storm_increase : ℝ := 0.20

/-- The number of windmills -/
def num_windmills : ℕ := 3

/-- The number of hours of operation -/
def num_hours : ℕ := 2

/-- Calculates the energy produced by multiple windmills on a stormy day over a given time period -/
def stormy_energy (typical : ℝ) (increase : ℝ) (windmills : ℕ) (hours : ℕ) : ℝ :=
  typical * (1 + increase) * (windmills : ℝ) * (hours : ℝ)

theorem windmill_energy_production :
  stormy_energy typical_energy storm_increase num_windmills num_hours = 2880 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_windmill_energy_production_l990_99058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l990_99076

-- Define the function f
def f (a b x : ℝ) : ℝ := x^2 + a*x + b

-- State the theorem
theorem function_properties
  (a b c : ℝ)
  (h1 : ∀ x, f a b x ≥ 0)
  (h2 : ∃ m, ∀ x, f a b x < m ↔ c < x ∧ x < c + 2 * Real.sqrt 2) :
  let m := Classical.choose h2
  ∃ m, (m = 2 ∧
    (∀ x y, x > 1 → y > 0 → x + y = m →
      (1 / (x - 1) + 2 / y ≥ 3 + 2 * Real.sqrt 2) ∧
      (∃ x₀ y₀, x₀ > 1 ∧ y₀ > 0 ∧ x₀ + y₀ = m ∧
        1 / (x₀ - 1) + 2 / y₀ = 3 + 2 * Real.sqrt 2))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l990_99076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circles_squares_area_ratio_l990_99092

theorem inscribed_circles_squares_area_ratio :
  let r : ℝ := 1  -- Radius of the smallest circle
  let small_circle_area := π * r^2
  let small_square_side := 2 * r
  let large_circle_radius := Real.sqrt 2 * r
  let large_square_side := 2 * large_circle_radius
  let large_square_area := large_square_side^2
  small_circle_area / large_square_area = π / 8 := by
  -- Proof steps will go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circles_squares_area_ratio_l990_99092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dog_distance_equals_12_l990_99038

/-- The distance between two points A and B -/
noncomputable def distance_AB : ℝ := 10

/-- The speed of person A in km/h -/
noncomputable def speed_A : ℝ := 6

/-- The speed of person B in km/h -/
noncomputable def speed_B : ℝ := 4

/-- The speed of the dog in km/h -/
noncomputable def speed_dog : ℝ := 12

/-- The time it takes for persons A and B to meet -/
noncomputable def meeting_time : ℝ := distance_AB / (speed_A + speed_B)

/-- The total distance run by the dog -/
noncomputable def dog_distance : ℝ := meeting_time * speed_dog

theorem dog_distance_equals_12 : dog_distance = 12 := by
  -- Unfold the definitions
  unfold dog_distance meeting_time distance_AB speed_A speed_B speed_dog
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dog_distance_equals_12_l990_99038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_remaining_numbers_is_perfect_square_l990_99020

theorem sum_of_remaining_numbers_is_perfect_square (k : ℕ) :
  let n := 50 * k
  let sum_all := n * (n + 1) / 2
  let sum_multiples_of_50 := k * (k + 1) * 25
  sum_all - sum_multiples_of_50 = (35 * k) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_remaining_numbers_is_perfect_square_l990_99020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_arrangement_count_l990_99000

-- Define the number of basil plants
def num_basil : ℕ := 5

-- Define the number of tomato plants
def num_tomato : ℕ := 3

-- Define the total number of groups to arrange (basil plants + tomato group)
def total_groups : ℕ := num_basil + 1

-- Theorem statement
theorem circular_arrangement_count :
  (Fact (total_groups > 1)) →
  (Nat.factorial (total_groups - 1) * Nat.factorial num_tomato = 720) :=
by
  intro h
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_arrangement_count_l990_99000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subsets_containing_five_l990_99062

theorem subsets_containing_five (S : Finset ℕ) (h : S = {1, 2, 3, 4, 5}) :
  (S.powerset.filter (λ A => 5 ∈ A)).card = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subsets_containing_five_l990_99062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_speed_theorem_l990_99031

/-- The speed of a particle at time t, given its position function -/
noncomputable def particleSpeed (t : ℝ) : ℝ :=
  Real.sqrt (36 * t^2 + 12 * t + 37)

/-- The position function of the particle -/
def particlePosition (t : ℝ) : ℝ × ℝ :=
  (3 * t^2 + t + 1, 6 * t + 2)

/-- Theorem stating that the speed of the particle at time t is equal to √(36t^2 + 12t + 37) -/
theorem particle_speed_theorem (t : ℝ) :
  particleSpeed t = Real.sqrt ((deriv (λ τ => (particlePosition τ).1) t)^2 +
                               (deriv (λ τ => (particlePosition τ).2) t)^2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_speed_theorem_l990_99031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_G_is_centroid_l990_99065

-- Define the triangle ABC
variable (A B C : EuclideanSpace ℝ (Fin 2))

-- Define an arbitrary point P in the plane
variable (P : EuclideanSpace ℝ (Fin 2))

-- Define point G
noncomputable def G (P A B C : EuclideanSpace ℝ (Fin 2)) : EuclideanSpace ℝ (Fin 2) :=
  (1/3 : ℝ) • (A + B + C - 2 • P)

-- Define the centroid of triangle ABC
noncomputable def centroid (A B C : EuclideanSpace ℝ (Fin 2)) : EuclideanSpace ℝ (Fin 2) :=
  (1/3 : ℝ) • (A + B + C)

-- Theorem statement
theorem G_is_centroid (P A B C : EuclideanSpace ℝ (Fin 2)) :
  G P A B C = centroid A B C := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_G_is_centroid_l990_99065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_specific_point_to_line_l990_99001

/-- The distance from a point in polar coordinates to a line in polar form -/
noncomputable def distance_point_to_line (r : ℝ) (θ : ℝ) (a : ℝ) (b : ℝ) : ℝ :=
  let x := r * Real.cos θ
  let y := r * Real.sin θ
  let numerator := |a * x + b * y + b|
  let denominator := Real.sqrt (a^2 + b^2)
  numerator / denominator

/-- Theorem: The distance from the point (√2, π/4) to the line ρsin(θ - π/3) = -√3/2 is 1/2 -/
theorem distance_specific_point_to_line :
  distance_point_to_line (Real.sqrt 2) (Real.pi/4) 3 (-Real.sqrt 3) = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_specific_point_to_line_l990_99001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l990_99023

noncomputable def f (x : ℝ) : ℝ := Real.cos x ^ 4 - 2 * Real.sin x * Real.cos x - Real.sin x ^ 4

theorem f_properties :
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ∈ Set.Icc (-Real.sqrt 2) 1) ∧
  (∀ x ∈ Set.Ioo 0 ((3 * Real.pi) / 8), ∀ y ∈ Set.Ioo 0 ((3 * Real.pi) / 8), x < y → f y < f x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l990_99023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_through_point_l990_99071

/-- Given a function f(x) = x - m/x that passes through the point (5, 4), prove that m = 5 -/
theorem function_through_point (m : ℝ) : (fun x : ℝ ↦ x - m / x) 5 = 4 → m = 5 := by
  intro h
  have eq : 5 - m / 5 = 4 := h
  linarith

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_through_point_l990_99071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_x_for_f_fifth_iterate_positive_integer_l990_99093

noncomputable def f (x : ℝ) : ℝ := 4/5 * (x - 1)

noncomputable def f_iterate (n : ℕ) (x : ℝ) : ℝ :=
  match n with
  | 0 => x
  | n + 1 => f (f_iterate n x)

theorem min_x_for_f_fifth_iterate_positive_integer :
  ∀ x : ℝ, (∃ k : ℤ, f_iterate 5 x = k) ∧ (f_iterate 5 x > 0) → x ≥ 3121 :=
by
  sorry

#check min_x_for_f_fifth_iterate_positive_integer

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_x_for_f_fifth_iterate_positive_integer_l990_99093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_bisecting_line_property_l990_99037

/-- Triangle ABC with vertices A(0, 6), B(2, 0), C(8, 0) -/
def triangle_ABC : Set (ℝ × ℝ) :=
  {⟨0, 6⟩, ⟨2, 0⟩, ⟨8, 0⟩}

/-- A line through point (2, 0) -/
def line_through_B (m b : ℝ) : Set (ℝ × ℝ) :=
  {⟨x, y⟩ | y = m * (x - 2) + b}

/-- Calculate the area of a triangle given its vertices -/
noncomputable def triangle_area (p₁ p₂ p₃ : ℝ × ℝ) : ℝ :=
  let (x₁, y₁) := p₁
  let (x₂, y₂) := p₂
  let (x₃, y₃) := p₃
  (1/2) * abs ((x₁ * (y₂ - y₃) + x₂ * (y₃ - y₁) + x₃ * (y₁ - y₂)))

/-- The line bisects the area of the triangle -/
def bisects_area (l : Set (ℝ × ℝ)) : Prop :=
  ∃ (A₁ A₂ : ℝ), A₁ = A₂ ∧ 
    A₁ + A₂ = triangle_area ⟨0, 6⟩ ⟨2, 0⟩ ⟨8, 0⟩

/-- Main theorem -/
theorem area_bisecting_line_property :
  ∃ (m b : ℝ), bisects_area (line_through_B m b) ∧ m * b = -9/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_bisecting_line_property_l990_99037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sheep_transaction_gain_l990_99016

/-- A farmer's sheep transaction problem -/
theorem sheep_transaction_gain :
  ∀ (x : ℝ), -- x represents the original cost per sheep
  let total_sheep : ℕ := 900
  let sold_at_original : ℕ := 850
  let sold_at_increased : ℕ := 50
  let increased_price_factor : ℝ := 1.2
  let total_cost : ℝ := x * total_sheep
  let revenue_original : ℝ := total_cost
  let revenue_increased : ℝ := x * increased_price_factor * sold_at_increased
  let total_revenue : ℝ := revenue_original + revenue_increased
  let profit : ℝ := total_revenue - total_cost
  let percent_gain : ℝ := (profit / total_cost) * 100
  abs (percent_gain - 11.11) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sheep_transaction_gain_l990_99016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coplanar_vectors_lambda_l990_99095

/-- Given three vectors a, b, and c in ℝ³, prove that if they are coplanar and
    a = (2, -1, 3), b = (-1, 4, -2), c = (7, 5, lambda), then lambda = 65/7. -/
theorem coplanar_vectors_lambda (a b c : ℝ × ℝ × ℝ) (lambda : ℝ) : 
  a = (2, -1, 3) →
  b = (-1, 4, -2) →
  c = (7, 5, lambda) →
  (∃ (x y : ℝ), c = x • a + y • b) →
  lambda = 65/7 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coplanar_vectors_lambda_l990_99095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_unique_good_arrangement_l990_99077

/-- A type representing a regular 1983-sided polygon -/
structure Polygon1983 :=
  (vertices : Fin 1983 → ℕ)

/-- A predicate that checks if a given arrangement is "good" with respect to all axes of symmetry -/
def is_good_arrangement (p : Polygon1983) : Prop :=
  ∀ (axis : Fin 991), 
    (∀ (i : Fin 991), p.vertices i > p.vertices (1982 - i)) ∨
    (∀ (i : Fin 991), p.vertices (1982 - i) > p.vertices i)

/-- The main theorem stating the existence and uniqueness of a good arrangement -/
theorem exists_unique_good_arrangement :
  ∃! (p : Polygon1983), is_good_arrangement p ∧ 
    (∀ (i j : Fin 1983), i ≠ j → p.vertices i ≠ p.vertices j) ∧
    (∀ (i : Fin 1983), p.vertices i ∈ Set.range (λ (n : Fin 1983) ↦ n.val + 1)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_unique_good_arrangement_l990_99077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_to_x_axis_l990_99045

/-- A polynomial function with integer coefficients -/
def IntPolynomial (n : ℕ) := Fin n → ℤ

/-- Evaluate a polynomial at a point -/
def evalPoly (p : IntPolynomial n) (x : ℤ) : ℤ :=
  (Finset.sum Finset.univ fun i => p i * x ^ i.val)

theorem parallel_to_x_axis 
  (n : ℕ) 
  (p : IntPolynomial n) 
  (c d : ℤ) 
  (h_distance : ∃ (k : ℤ), (c - d)^2 + (evalPoly p c - evalPoly p d)^2 = k^2) :
  evalPoly p c = evalPoly p d := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_to_x_axis_l990_99045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_is_even_l990_99051

noncomputable def e : ℝ := Real.exp 1

noncomputable def f (x : ℝ) : ℝ := e^x - e^(-x)

noncomputable def f' (x : ℝ) : ℝ := e^x + e^(-x)

theorem derivative_f_is_even : ∀ x : ℝ, f' x = f' (-x) := by
  intro x
  simp [f']
  ring_nf


end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_is_even_l990_99051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_l990_99049

/-- Proves that a train with the given length and time to cross a pole has the specified speed in km/hr -/
theorem train_speed (train_length : ℝ) (crossing_time : ℝ) (speed_kmh : ℝ) : 
  train_length = 200 →
  crossing_time = 3.3330666879982935 →
  speed_kmh = (train_length / crossing_time) * 3.6 →
  ‖speed_kmh - 216.00072‖ < 0.00001 := by
  sorry

#eval (200 / 3.3330666879982935) * 3.6

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_l990_99049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_usage_theorem_l990_99057

noncomputable def initial_paint : ℝ := 360

noncomputable def first_week_usage (total : ℝ) : ℝ := (1 / 4) * total

noncomputable def second_week_usage (remaining : ℝ) : ℝ := (1 / 3) * remaining

theorem paint_usage_theorem :
  let first_usage := first_week_usage initial_paint
  let remaining := initial_paint - first_usage
  let second_usage := second_week_usage remaining
  first_usage + second_usage = 180 := by
    -- Unfold definitions
    unfold first_week_usage second_week_usage initial_paint
    -- Simplify expressions
    simp
    -- Perform algebraic manipulations
    ring
    -- The proof is complete
    done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_usage_theorem_l990_99057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2004_bounds_l990_99070

noncomputable def a : ℕ → ℝ
  | 0 => 1
  | 1 => 2
  | (n + 2) => a (n + 1) + 1 / a (n + 1)

theorem a_2004_bounds : 63 < a 2004 ∧ a 2004 < 78 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2004_bounds_l990_99070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_of_angle_between_tangent_lines_l990_99073

/-- Two lines tangent to a circle with given intersection point --/
structure TangentLines where
  -- The circle is defined by x² + y² = 2
  circle : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 2}
  -- Two tangent lines
  l₁ : Set (ℝ × ℝ)
  l₂ : Set (ℝ × ℝ)
  -- Intersection point of l₁ and l₂
  intersection : ℝ × ℝ := (1, 3)
  -- l₁ and l₂ are tangent to the circle
  is_tangent_l₁ : ∀ p ∈ l₁, (p.1 - 0)^2 + (p.2 - 0)^2 = 2
  is_tangent_l₂ : ∀ p ∈ l₂, (p.1 - 0)^2 + (p.2 - 0)^2 = 2
  -- Intersection point is on both lines
  intersection_on_l₁ : intersection ∈ l₁
  intersection_on_l₂ : intersection ∈ l₂

/-- The angle between two lines --/
noncomputable def angleBetweenLines (l₁ l₂ : Set (ℝ × ℝ)) : ℝ := sorry

/-- Tangent of an angle --/
noncomputable def tan (θ : ℝ) : ℝ := sorry

/-- Main theorem: The tangent of the angle between the two tangent lines is 4/3 --/
theorem tangent_of_angle_between_tangent_lines (tl : TangentLines) :
  tan (angleBetweenLines tl.l₁ tl.l₂) = 4/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_of_angle_between_tangent_lines_l990_99073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_greater_than_n_l990_99024

/-- Given two quadratic expressions M and N, prove that M > N --/
theorem m_greater_than_n (x : ℝ) : 
  (x - 3) * (x - 4) > (x - 1) * (x - 6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_greater_than_n_l990_99024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_AOB_is_three_l990_99059

/-- Polar coordinate type --/
structure PolarCoord where
  r : ℝ
  θ : ℝ

/-- Function to calculate the area of a triangle given three points in polar coordinates --/
noncomputable def areaTriangle (O B C : PolarCoord) : ℝ :=
  (1/2) * B.r * C.r * Real.sin (C.θ - B.θ)

/-- Given two points A and B in polar coordinates, prove that the area of triangle AOB is 3 --/
theorem area_triangle_AOB_is_three :
  let A : PolarCoord := ⟨3, π/3⟩
  let B : PolarCoord := ⟨-4, 7*π/6⟩
  let O : PolarCoord := ⟨0, 0⟩
  (areaTriangle O A B) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_AOB_is_three_l990_99059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_over_y_equals_ten_l990_99053

theorem x_over_y_equals_ten (x y m : ℝ) (hx : x > 0) (hy : y > 0)
  (h1 : Real.log x / Real.log 10 = m) (h2 : y = 10^(m - 1)) : x / y = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_over_y_equals_ten_l990_99053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l990_99052

-- Define the hyperbola equation
def hyperbola_equation (x y : ℝ) : Prop := y^2 / 4 - x^2 = 1

-- Define the asymptote equations
def asymptote_equations (x y : ℝ) : Prop := y = 2*x ∨ y = -2*x

-- Define the eccentricity
noncomputable def eccentricity : ℝ := Real.sqrt 5 / 2

-- Theorem statement
theorem hyperbola_properties :
  (∀ x y : ℝ, hyperbola_equation x y → asymptote_equations x y) ∧
  (eccentricity = Real.sqrt 5 / 2) :=
by
  constructor
  · sorry  -- Proof for asymptote equations
  · rfl    -- Proof for eccentricity (reflexivity)


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l990_99052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_score_difference_l990_99074

/-- Represents the score distribution in a mathematics competition -/
structure ScoreDistribution where
  score60 : ℝ
  score75 : ℝ
  score85 : ℝ
  score95 : ℝ
  sum_to_one : score60 + score75 + score85 + score95 = 1
  non_negative : score60 ≥ 0 ∧ score75 ≥ 0 ∧ score85 ≥ 0 ∧ score95 ≥ 0

/-- Calculates the mean score given a score distribution -/
def meanScore (d : ScoreDistribution) : ℝ :=
  60 * d.score60 + 75 * d.score75 + 85 * d.score85 + 95 * d.score95

/-- Determines the median score given a score distribution -/
noncomputable def medianScore (d : ScoreDistribution) : ℝ :=
  if d.score60 + d.score75 > 0.5 then 75
  else if d.score60 + d.score75 + d.score85 > 0.5 then 85
  else 95

/-- Theorem stating that the difference between the median and mean score is 4 -/
theorem score_difference (d : ScoreDistribution) 
  (h1 : d.score60 = 0.15) 
  (h2 : d.score75 = 0.25) 
  (h3 : d.score85 = 0.40) : 
  ‖medianScore d - meanScore d‖ = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_score_difference_l990_99074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_range_l990_99067

/-- Given a line x - my - 8 = 0 intersecting a parabola y² = 8x at points A and B,
    with O being the origin, prove that the area S of triangle OAB satisfies S ∈ [64, +∞). -/
theorem triangle_area_range (m : ℝ) : 
  let line := {(x, y) : ℝ × ℝ | x - m * y - 8 = 0}
  let parabola := {(x, y) : ℝ × ℝ | y^2 = 8 * x}
  let O := (0, 0)
  ∃ A B : ℝ × ℝ, A ∈ line ∩ parabola ∧ B ∈ line ∩ parabola ∧ A ≠ B ∧
    let S := abs (A.1 * B.2 - A.2 * B.1) / 2
    S ∈ Set.Ici 64 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_range_l990_99067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_difference_l990_99044

-- Define the circle
def circle_region (x y : ℝ) : Prop := x^2 + y^2 ≤ 4

-- Define the line passing through P(1,1)
def line_through_P (m b : ℝ) (x y : ℝ) : Prop :=
  y = m * x + b ∧ 1 = m * 1 + b

-- Define the area difference function (noncomputable as it involves integration)
noncomputable def area_difference (m b : ℝ) : ℝ := sorry

-- Theorem statement
theorem max_area_difference :
  ∃ (m b : ℝ), line_through_P m b 1 1 ∧
  (∀ (m' b' : ℝ), line_through_P m' b' 1 1 →
    area_difference m b ≥ area_difference m' b') ∧
  (∀ (x y : ℝ), line_through_P m b x y ↔ y = x - 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_difference_l990_99044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_line_curve_l990_99017

/-- The line l in the Cartesian plane -/
def line_l (x y : ℝ) : Prop :=
  x + y - 6 = 0

/-- The curve C in the Cartesian plane -/
def curve_C (x y : ℝ) : Prop :=
  x^2 / 3 + y^2 = 1

/-- The distance between two points in the Cartesian plane -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

/-- The theorem stating the minimum distance between line l and curve C -/
theorem min_distance_line_curve :
  ∃ (min_dist : ℝ), min_dist = 2 * Real.sqrt 2 ∧
  ∀ (x1 y1 x2 y2 : ℝ),
    line_l x1 y1 → curve_C x2 y2 →
    distance x1 y1 x2 y2 ≥ min_dist := by
  sorry

#check min_distance_line_curve

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_line_curve_l990_99017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_return_speed_is_10_l990_99072

/-- Calculates the return speed given total distance, total time, and the fact that return speed is twice the outbound speed -/
noncomputable def calculate_return_speed (total_distance : ℝ) (total_time : ℝ) : ℝ :=
  let outbound_distance := total_distance / 2
  let return_distance := total_distance / 2
  let outbound_speed := outbound_distance / (total_time / 3)
  2 * outbound_speed

/-- Theorem stating that for a 40 km round trip with 6 hours total travel time and return speed twice the outbound speed, the return speed is 10 km/h -/
theorem return_speed_is_10 :
  calculate_return_speed 40 6 = 10 := by
  sorry

-- Remove the #eval statement as it's not necessary for the proof
-- and may cause issues with noncomputable definitions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_return_speed_is_10_l990_99072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_negative_seven_scaling_l990_99033

theorem matrix_negative_seven_scaling (M : Matrix (Fin 3) (Fin 3) ℝ) :
  (∀ v : Fin 3 → ℝ, M.mulVec v = (-7 : ℝ) • v) ↔ 
  M = ![![-7, 0, 0], ![0, -7, 0], ![0, 0, -7]] := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_negative_seven_scaling_l990_99033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_recurring_decimal_sum_l990_99027

theorem recurring_decimal_sum : 
  (∃ (x y : ℚ), x = 2/9 ∧ y = 101/3333 ∧ x + y = 827 / 3333) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_recurring_decimal_sum_l990_99027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_methods_statement_correct_l990_99043

/-- Represents the analytical method in problem-solving -/
def analytical_method : Prop := True

/-- Represents the synthetic method in problem-solving -/
def synthetic_method : Prop := True

/-- Represents the process of finding ideas and methods for solving problems -/
def find_ideas_and_methods : Prop := True

/-- Represents the process of demonstrating the problem-solving process -/
def demonstrate_process : Prop := True

/-- States that the analytical method is used to find ideas and methods for solving problems -/
axiom analytical_for_ideas : analytical_method → find_ideas_and_methods

/-- States that the synthetic method is used to demonstrate the process of solving problems -/
axiom synthetic_for_demonstration : synthetic_method → demonstrate_process

/-- Theorem stating that the given statement about analytical and synthetic methods is correct -/
theorem methods_statement_correct :
  (analytical_method → find_ideas_and_methods) ∧ (synthetic_method → demonstrate_process) := by
  apply And.intro
  exact analytical_for_ideas
  exact synthetic_for_demonstration


end NUMINAMATH_CALUDE_ERRORFEEDBACK_methods_statement_correct_l990_99043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_plus_b_is_zero_l990_99030

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := Real.log x - a * x - b

-- State the theorem
theorem min_a_plus_b_is_zero :
  ∀ a b : ℝ, (∀ x : ℝ, x > 0 → f a b x ≤ 0) →
  (∀ c d : ℝ, (∀ x : ℝ, x > 0 → f c d x ≤ 0) → a + b ≤ c + d) →
  a + b = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_plus_b_is_zero_l990_99030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_worker_de_time_l990_99060

/-- Represents a worker paving a path -/
structure Worker where
  speed : ℝ
  path : List (ℝ × ℝ)

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Calculates the total distance of a path -/
noncomputable def pathLength (path : List (ℝ × ℝ)) : ℝ :=
  match path with
  | [] => 0
  | [_] => 0
  | x :: y :: rest => distance x y + pathLength (y :: rest)

/-- The time taken to pave a segment of a path -/
noncomputable def segmentTime (w : Worker) (p1 p2 : ℝ × ℝ) : ℝ := 
  distance p1 p2 / w.speed

theorem second_worker_de_time 
  (w1 w2 : Worker) 
  (a b c d e f : ℝ × ℝ) 
  (h1 : w1.path = [a, b, c])
  (h2 : w2.path = [a, d, e, f, c])
  (h3 : pathLength w1.path / w1.speed = 9)
  (h4 : pathLength w2.path / w2.speed = 9)
  (h5 : w2.speed = 1.2 * w1.speed)
  : segmentTime w2 d e = 45 / 60 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_worker_de_time_l990_99060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_four_digit_binary_to_decimal_l990_99018

def fromDigits (base : ℕ) (digits : List Bool) : ℕ :=
  digits.foldl (fun acc d => base * acc + (if d then 1 else 0)) 0

theorem largest_four_digit_binary_to_decimal :
  (∀ n : ℕ, n ≤ 15 → ∃ b : List Bool, b.length = 4 ∧ fromDigits 2 b = n) ∧
  (∀ b : List Bool, b.length = 4 → fromDigits 2 b ≤ 15) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_four_digit_binary_to_decimal_l990_99018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transformation_result_l990_99009

noncomputable def initial_number : ℂ := -4 - 6 * Complex.I

noncomputable def rotation_60 (z : ℂ) : ℂ := z * Complex.exp (Complex.I * Real.pi / 3)

noncomputable def dilation_2 (z : ℂ) : ℂ := 2 * z

noncomputable def transformed_number : ℂ := dilation_2 (rotation_60 initial_number)

theorem transformation_result :
  transformed_number = (-4 + 6 * Real.sqrt 3) - (4 * Real.sqrt 3 + 6) * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transformation_result_l990_99009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orange_juice_production_l990_99084

/-- Calculates the amount of oranges used for juice production given total production and export percentage -/
noncomputable def oranges_for_juice (total_production : ℝ) (export_percentage : ℝ) (juice_percentage : ℝ) : ℝ :=
  total_production * (1 - export_percentage / 100) * (juice_percentage / 100)

/-- Rounds a real number to the nearest tenth -/
noncomputable def round_to_tenth (x : ℝ) : ℝ :=
  ↑(Int.floor (x * 10 + 0.5)) / 10

theorem orange_juice_production :
  let total_production : ℝ := 8.2
  let export_percentage : ℝ := 30
  let juice_percentage : ℝ := 40
  round_to_tenth (oranges_for_juice total_production export_percentage juice_percentage) = 2.3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_orange_juice_production_l990_99084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_truth_teller_l990_99069

/-- Represents the state of having watched the movie or not -/
inductive MovieWatched : Type
| yes : MovieWatched
| no : MovieWatched

/-- Represents a friend who is either always truthful or always lying -/
structure Friend :=
(isHonest : Bool)
(statement : MovieWatched → MovieWatched → MovieWatched → Bool)

/-- The scenario with three friends and their statements -/
def movieScenario (f1 f2 f3 : Friend) : Prop :=
  ∃ (m1 m2 m3 : MovieWatched),
    (f1.isHonest ↔ f1.statement m1 m2 m3 = (m1 = MovieWatched.no ∧ m2 = MovieWatched.no ∧ m3 = MovieWatched.no)) ∧
    (f2.isHonest ↔ f2.statement m1 m2 m3 = (m1 = MovieWatched.no ∧ m2 = MovieWatched.yes ∧ m3 = MovieWatched.no)) ∧
    (f3.isHonest ↔ f3.statement m1 m2 m3 = (m3 = MovieWatched.yes))

/-- The theorem to be proved -/
theorem one_truth_teller (f1 f2 f3 : Friend) :
  movieScenario f1 f2 f3 →
  (f1.isHonest ∨ f2.isHonest ∨ f3.isHonest) →
  (¬f1.isHonest ∨ ¬f2.isHonest ∨ ¬f3.isHonest) →
  (xor f1.isHonest (xor f2.isHonest f3.isHonest)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_truth_teller_l990_99069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_mean_relation_l990_99085

/-- Given two positive real numbers a and b, where a = kb and k > 1,
    if the arithmetic mean of a and b is triple their geometric mean,
    then k is approximately 34 (to the nearest integer). -/
theorem arithmetic_geometric_mean_relation (a b k : ℝ) : 
  a > 0 → b > 0 → k > 1 → a = k * b → (a + b) / 2 = 3 * Real.sqrt (a * b) →
  ⌊k + 0.5⌋ = 34 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_mean_relation_l990_99085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stone_stacking_game_l990_99008

/-- Represents the state of the game with the number of stones in each stack -/
def GameState := List Nat

/-- The rules of Pedrinho's stone stacking game -/
inductive game_rules (initial : Nat) : GameState → Prop
  | initial : game_rules initial [initial]
  | split {prev : GameState} (h : game_rules initial prev) 
          (i : Nat) (n m : Nat) :
    prev.get? i = some (n + m + 1) →
    n > 0 → m > 0 →
    game_rules initial (prev.removeNth i ++ [n, m])

/-- Theorem stating the solution to the stone stacking game -/
theorem stone_stacking_game :
  (∃ (final : GameState), game_rules 19 final ∧ final.all (· = 3)) ∧
  (¬∃ (final : GameState), game_rules 1001 final ∧ final.all (· = 3)) := by
  sorry

#check stone_stacking_game

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stone_stacking_game_l990_99008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_truncated_cone_volume_l990_99047

/-- The volume of a truncated right circular cone -/
noncomputable def truncated_cone_volume (R h r : ℝ) : ℝ :=
  (1/3) * Real.pi * h * (R^2 + r^2 + R*r)

/-- Theorem: Volume of a specific truncated right circular cone -/
theorem specific_truncated_cone_volume :
  truncated_cone_volume 10 10 5 = (1750/3) * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_truncated_cone_volume_l990_99047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_triangle_area_l990_99089

/-- The parabola defined by y^2 = 4x -/
def Parabola : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2^2 = 4 * p.1}

/-- The focus of the parabola y^2 = 4x -/
def F : ℝ × ℝ := (1, 0)

/-- A point P on the parabola -/
noncomputable def P : ℝ × ℝ := sorry

/-- The origin O -/
def O : ℝ × ℝ := (0, 0)

/-- The angle between PO and OF -/
noncomputable def angle_POF : ℝ := sorry

theorem parabola_triangle_area :
  P ∈ Parabola →
  ‖P - F‖ = 3 →
  (1/2 : ℝ) * ‖P - O‖ * ‖F - O‖ * Real.sin angle_POF = Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_triangle_area_l990_99089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_quadrant_l990_99078

theorem complex_number_quadrant (a : ℝ) (h : 3 < a ∧ a < 5) :
  (Complex.mk (a^2 - 8*a + 15) (a^2 - 5*a - 14)).re < 0 ∧
  (Complex.mk (a^2 - 8*a + 15) (a^2 - 5*a - 14)).im < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_quadrant_l990_99078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_suzhou_blood_donors_scientific_notation_l990_99007

-- Define scientific notation
noncomputable def scientific_notation (a : ℝ) (n : ℤ) : ℝ := a * (10 : ℝ) ^ n

-- Define the property of scientific notation
def is_scientific_notation (x : ℝ) (a : ℝ) (n : ℤ) : Prop :=
  x = scientific_notation a n ∧ 1 ≤ |a| ∧ |a| < 10

-- Theorem statement
theorem suzhou_blood_donors_scientific_notation :
  is_scientific_notation 124000 1.24 5 := by
  sorry

#check suzhou_blood_donors_scientific_notation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_suzhou_blood_donors_scientific_notation_l990_99007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_difference_squares_l990_99010

theorem absolute_difference_squares : |((110 : ℝ)^2 - (108 : ℝ)^2)| = 436 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_difference_squares_l990_99010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gold_length_approx_l990_99081

noncomputable section

-- Define the total length of the pencil
def total_length : ℝ := 15

-- Define the fraction of the pencil that is green
def green_fraction : ℝ := 7/10

-- Define the fraction of the remaining part that is gold
def gold_fraction : ℝ := Real.sqrt 2 / 2

-- Define the fraction of the remaining part after gold that is red
def red_fraction : ℝ := 1/4

-- Define the length of the green part
def green_length : ℝ := green_fraction * total_length

-- Define the remaining length after green
def remaining_after_green : ℝ := total_length - green_length

-- Define the length of the gold part
def gold_length : ℝ := gold_fraction * remaining_after_green

end noncomputable section

-- Theorem statement
theorem gold_length_approx : 
  ∃ ε > 0, abs (gold_length - 3.182) < ε := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gold_length_approx_l990_99081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_properties_min_area_and_equation_l990_99014

noncomputable section

/-- The line equation parameters -/
def line_equation (x y m : ℝ) : Prop := x / m + y / (4 - m) = 1

/-- The slope of the line -/
noncomputable def line_slope (m : ℝ) : ℝ := (4 - m) / (-m)

/-- The area of triangle AOB -/
noncomputable def triangle_area (m : ℝ) : ℝ := m * (4 - m) / 2

theorem line_properties (m : ℝ) :
  (∀ x y, line_equation x y m → line_slope m < 2) ↔ (m > 0 ∨ m < -4) ∧ m ≠ 4 :=
sorry

theorem min_area_and_equation :
  ∃ m, 0 < m ∧ m < 4 ∧
    (∀ m', 0 < m' ∧ m' < 4 → triangle_area m ≤ triangle_area m') ∧
    triangle_area m = 2 ∧
    (∀ x y, line_equation x y m ↔ x + y = 2) :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_properties_min_area_and_equation_l990_99014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l990_99094

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin x * Real.cos x + Real.sin x ^ 2

theorem f_properties :
  (∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, f (x + T) = f x ∧ ∀ S : ℝ, S > 0 ∧ (∀ x : ℝ, f (x + S) = f x) → T ≤ S) ∧
  (∀ x : ℝ, x ∈ Set.Ioo 0 (Real.pi / 2) → f x ∈ Set.Icc 0 (3 / 2)) ∧
  (∃ s₁ s₂ s₃ : Set ℝ,
    s₁ = Set.Icc 0 (Real.pi / 3) ∧
    s₂ = Set.Icc (5 * Real.pi / 6) (4 * Real.pi / 3) ∧
    s₃ = Set.Icc (11 * Real.pi / 6) (2 * Real.pi) ∧
    (∀ x y : ℝ, x ∈ s₁ ∪ s₂ ∪ s₃ → y ∈ s₁ ∪ s₂ ∪ s₃ → x < y → f x < f y) ∧
    (∀ s : Set ℝ, s ⊆ Set.Icc 0 (2 * Real.pi) →
      (∀ x y : ℝ, x ∈ s → y ∈ s → x < y → f x < f y) →
      s ⊆ s₁ ∪ s₂ ∪ s₃)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l990_99094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l990_99006

theorem inequality_proof (a b : ℝ) (n : ℕ) (x : ℝ) 
  (ha : a > 0) (hb : b > 0) (hn : n > 0) :
  a / (Real.sin x)^n + b / (Real.cos x)^n ≥ (a^(2/(n+2:ℝ)) + b^(2/(n+2:ℝ)))^((n+2:ℝ)/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l990_99006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l990_99034

/-- The time it takes for A and B to complete the work together -/
noncomputable def time_together (time_A time_B : ℝ) : ℝ :=
  1 / (1 / time_A + 1 / time_B)

/-- Theorem: A and B working together can complete the work in 2 days -/
theorem work_completion_time :
  let time_A : ℝ := 3
  let time_B : ℝ := 6
  time_together time_A time_B = 2 := by
  -- Unfold the definition of time_together
  unfold time_together
  -- Simplify the expression
  simp
  -- Perform algebraic manipulations
  ring
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l990_99034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_square_ABCD_is_16_l990_99021

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Triangle in 2D space -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Square in 2D space -/
structure Square where
  A : Point
  B : Point
  C : Point
  D : Point

noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

def isRightTriangle (t : Triangle) : Prop :=
  (distance t.A t.B)^2 + (distance t.B t.C)^2 = (distance t.A t.C)^2

def isVerticallyAbove (p1 p2 : Point) : Prop :=
  p1.x = p2.x ∧ p1.y > p2.y

def isParallel (p1 p2 p3 p4 : Point) : Prop :=
  (p2.y - p1.y) * (p4.x - p3.x) = (p4.y - p3.y) * (p2.x - p1.x)

noncomputable def squareArea (s : Square) : ℝ :=
  (distance s.A s.B)^2

theorem area_of_square_ABCD_is_16 
  (A B C E : Point)
  (D : Point)
  (h1 : B.x = 0 ∧ B.y = 0)
  (h2 : C.x = 4 ∧ C.y = 0)
  (h3 : E.x = 4 ∧ E.y = 3)
  (h4 : isRightTriangle ⟨B, C, E⟩)
  (h5 : isVerticallyAbove D C)
  (h6 : isParallel D C B E)
  (h7 : distance D A = distance B C)
  : squareArea ⟨A, B, C, D⟩ = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_square_ABCD_is_16_l990_99021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alternating_sum_to_101_l990_99042

theorem alternating_sum_to_101 : 
  (Finset.range 101).sum (fun i => (-1:ℤ)^i * (i + 1)) = 76 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alternating_sum_to_101_l990_99042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_projection_properties_l990_99066

-- Define the concept of horizontal projection
def horizontal_projection (shape : Type) : Type := sorry

-- Define geometric shapes
def rectangle : Type := sorry
def triangle : Type := sorry
def square : Type := sorry
def ellipse : Type := sorry  -- Changed from 'circle' to 'ellipse'

-- Define properties of projected shapes
def is_parallelogram (shape : Type) : Prop := sorry
def is_triangle (shape : Type) : Prop := sorry
def is_rhombus (shape : Type) : Prop := sorry
def is_ellipse (shape : Type) : Prop := sorry  -- Changed from 'is_circle' to 'is_ellipse'

-- Theorem to prove
theorem horizontal_projection_properties :
  (is_parallelogram (horizontal_projection rectangle)) ∧
  (is_triangle (horizontal_projection triangle)) ∧
  ¬(is_rhombus (horizontal_projection square)) ∧
  (is_ellipse (horizontal_projection ellipse)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_projection_properties_l990_99066
