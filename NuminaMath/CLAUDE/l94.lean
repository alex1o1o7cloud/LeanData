import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_maximum_value_l94_9441

/-- A quadratic function f(x) = ax² + bx + c -/
def QuadraticFunction (a b c : ℝ) := fun (x : ℝ) ↦ a * x^2 + b * x + c

/-- The derivative of a quadratic function -/
def QuadraticDerivative (a b : ℝ) := fun (x : ℝ) ↦ 2 * a * x + b

theorem quadratic_maximum_value (a b c : ℝ) :
  (∀ x : ℝ, QuadraticFunction a b c x ≥ QuadraticDerivative a b x) →
  (∃ M : ℝ, M = 2 * Real.sqrt 2 - 2 ∧
    (∀ k : ℝ, k ≤ M ↔ ∃ a' b' c' : ℝ, 
      (∀ x : ℝ, QuadraticFunction a' b' c' x ≥ QuadraticDerivative a' b' x) ∧
      k = b'^2 / (a'^2 + c'^2))) :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_maximum_value_l94_9441


namespace NUMINAMATH_CALUDE_greatest_of_three_consecutive_integers_l94_9469

theorem greatest_of_three_consecutive_integers (x y z : ℤ) : 
  (y = x + 1) → (z = y + 1) → (x + y + z = 39) → max x (max y z) = 14 := by
  sorry

end NUMINAMATH_CALUDE_greatest_of_three_consecutive_integers_l94_9469


namespace NUMINAMATH_CALUDE_tangent_product_l94_9448

theorem tangent_product (A B : ℝ) (hA : A = 30 * π / 180) (hB : B = 60 * π / 180) :
  (1 + Real.tan A) * (1 + Real.tan B) = 2 + 4 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_product_l94_9448


namespace NUMINAMATH_CALUDE_fraction_power_rule_l94_9470

theorem fraction_power_rule (a b : ℝ) (hb : b ≠ 0) : (a / b) ^ 4 = a ^ 4 / b ^ 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_power_rule_l94_9470


namespace NUMINAMATH_CALUDE_infinitely_many_odd_n_composite_l94_9485

theorem infinitely_many_odd_n_composite (n : ℕ) : 
  ∃ S : Set ℕ, (Set.Infinite S) ∧ 
  (∀ n ∈ S, Odd n ∧ ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ 2^n + n - 1 = a * b) :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_odd_n_composite_l94_9485


namespace NUMINAMATH_CALUDE_instantaneous_velocity_zero_at_two_l94_9445

-- Define the motion law
def motion_law (t : ℝ) : ℝ := t^2 - 4*t + 5

-- Define the instantaneous velocity (derivative of motion law)
def instantaneous_velocity (t : ℝ) : ℝ := 2*t - 4

-- Theorem statement
theorem instantaneous_velocity_zero_at_two :
  ∃ (t : ℝ), instantaneous_velocity t = 0 ∧ t = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_zero_at_two_l94_9445


namespace NUMINAMATH_CALUDE_emma_ball_lists_l94_9498

/-- The number of balls in the bin -/
def n : ℕ := 24

/-- The number of draws -/
def k : ℕ := 4

/-- The number of possible lists when drawing with replacement from n balls, k times -/
def num_lists (n k : ℕ) : ℕ := n^k

theorem emma_ball_lists : num_lists n k = 331776 := by
  sorry

end NUMINAMATH_CALUDE_emma_ball_lists_l94_9498


namespace NUMINAMATH_CALUDE_function_properties_l94_9467

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := -Real.sqrt a / (a^x + Real.sqrt a)

theorem function_properties (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x : ℝ, f a x + f a (1 - x) = -1) ∧
  (f a (-2) + f a (-1) + f a 0 + f a 1 + f a 2 + f a 3 = -3) := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l94_9467


namespace NUMINAMATH_CALUDE_equation_solution_l94_9466

theorem equation_solution (y : ℝ) : 
  (y / 6) / 3 = 9 / (y / 3) → y = 3 * Real.sqrt 54 ∨ y = -3 * Real.sqrt 54 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l94_9466


namespace NUMINAMATH_CALUDE_max_value_of_function_l94_9432

theorem max_value_of_function (x : ℝ) (h : x < 5/4) :
  (4 * x - 2 + 1 / (4 * x - 5)) ≤ 1 ∧ 
  ∃ y : ℝ, y < 5/4 ∧ 4 * y - 2 + 1 / (4 * y - 5) = 1 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_function_l94_9432


namespace NUMINAMATH_CALUDE_smallest_multiple_of_seven_greater_than_500_l94_9477

theorem smallest_multiple_of_seven_greater_than_500 :
  ∃ (n : ℕ), n * 7 = 504 ∧ 
  504 > 500 ∧
  ∀ (m : ℕ), m * 7 > 500 → m * 7 ≥ 504 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_seven_greater_than_500_l94_9477


namespace NUMINAMATH_CALUDE_smallest_common_difference_l94_9414

/-- Represents a quadratic equation ax^2 + bx + c = 0 --/
structure QuadraticEquation where
  a : Int
  b : Int
  c : Int

/-- Checks if a quadratic equation has two distinct roots --/
def hasTwoDistinctRoots (eq : QuadraticEquation) : Prop :=
  eq.b * eq.b - 4 * eq.a * eq.c > 0

/-- Generates all possible quadratic equations with coefficients a, b, 2c --/
def generateQuadraticEquations (a b c : Int) : List QuadraticEquation :=
  [
    ⟨a, b, 2*c⟩, ⟨a, 2*c, b⟩, ⟨b, a, 2*c⟩,
    ⟨b, 2*c, a⟩, ⟨2*c, a, b⟩, ⟨2*c, b, a⟩
  ]

theorem smallest_common_difference
  (a b c : Int)
  (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)
  (h_arithmetic : ∃ d : Int, b = a + d ∧ c = a + 2*d)
  (h_increasing : a < b ∧ b < c)
  (h_distinct_roots : ∀ eq ∈ generateQuadraticEquations a b c, hasTwoDistinctRoots eq) :
  ∃ d : Int, d = 4 ∧ a = -5 ∧ b = -1 ∧ c = 3 :=
sorry

end NUMINAMATH_CALUDE_smallest_common_difference_l94_9414


namespace NUMINAMATH_CALUDE_sequence_existence_l94_9412

theorem sequence_existence : ∃ (a : ℕ → ℕ+), 
  (∀ k : ℕ+, ∃ n : ℕ, a n = k) ∧ 
  (∀ k : ℕ+, (Finset.range k).sum (λ i => (a i.succ).val) % k = 0) :=
sorry

end NUMINAMATH_CALUDE_sequence_existence_l94_9412


namespace NUMINAMATH_CALUDE_second_race_lead_l94_9458

/-- Represents a runner in the race -/
structure Runner where
  speed : ℝ

/-- Represents the race conditions -/
structure RaceConditions where
  raceDistance : ℝ
  sunnyLeadFirstRace : ℝ
  sunnyStartSecondRace : ℝ
  windyDelaySecondRace : ℝ

/-- Calculates the lead of Sunny over Windy in the second race -/
def calculateSecondRaceLead (sunny : Runner) (windy : Runner) (conditions : RaceConditions) : ℝ :=
  sorry

/-- The main theorem stating the result of the second race -/
theorem second_race_lead
  (sunny : Runner)
  (windy : Runner)
  (conditions : RaceConditions)
  (h1 : conditions.raceDistance = 400)
  (h2 : conditions.sunnyLeadFirstRace = 50)
  (h3 : conditions.sunnyStartSecondRace = -50)
  (h4 : conditions.windyDelaySecondRace = 10)
  (h5 : sunny.speed / windy.speed = 8 / 7) :
  calculateSecondRaceLead sunny windy conditions = 56.25 :=
sorry

end NUMINAMATH_CALUDE_second_race_lead_l94_9458


namespace NUMINAMATH_CALUDE_difference_of_fractions_l94_9438

theorem difference_of_fractions : 
  (3 - 390 / 5) - (4 - 210 / 7) = -49 := by sorry

end NUMINAMATH_CALUDE_difference_of_fractions_l94_9438


namespace NUMINAMATH_CALUDE_convex_polygon_interior_angles_l94_9437

theorem convex_polygon_interior_angles (n : ℕ) : 
  n ≥ 3 →  -- Convex polygon has at least 3 sides
  (∀ k, k ∈ Finset.range n → 
    100 + k * 10 < 180) →  -- All interior angles are less than 180°
  (100 + (n - 1) * 10 ≥ 180) →  -- The largest angle is at least 180°
  n = 8 := by
  sorry

end NUMINAMATH_CALUDE_convex_polygon_interior_angles_l94_9437


namespace NUMINAMATH_CALUDE_ellipse_line_intersection_l94_9427

/-- The ellipse E -/
def E (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

/-- The line l -/
def l (k m x y : ℝ) : Prop := y = k*x + m

/-- Predicate to check if a point is on the ellipse E -/
def on_ellipse (x y : ℝ) : Prop := E x y

/-- Predicate to check if a point is on the line l -/
def on_line (k m x y : ℝ) : Prop := l k m x y

/-- The right vertex of the ellipse -/
def right_vertex : ℝ × ℝ := (2, 0)

/-- Predicate to check if two points are different -/
def different (p1 p2 : ℝ × ℝ) : Prop := p1 ≠ p2

theorem ellipse_line_intersection (k m : ℝ) :
  ∃ (M N : ℝ × ℝ),
    on_ellipse M.1 M.2 ∧
    on_ellipse N.1 N.2 ∧
    on_line k m M.1 M.2 ∧
    on_line k m N.1 N.2 ∧
    different M right_vertex ∧
    different N right_vertex ∧
    different M N →
    on_line k m (2/7) 0 :=
sorry

end NUMINAMATH_CALUDE_ellipse_line_intersection_l94_9427


namespace NUMINAMATH_CALUDE_height_difference_proof_l94_9440

/-- Proves that the height difference between Vlad and his sister is 104.14 cm -/
theorem height_difference_proof (vlad_height_m : ℝ) (sister_height_cm : ℝ) 
  (h1 : vlad_height_m = 1.905) (h2 : sister_height_cm = 86.36) : 
  vlad_height_m * 100 - sister_height_cm = 104.14 := by
  sorry

end NUMINAMATH_CALUDE_height_difference_proof_l94_9440


namespace NUMINAMATH_CALUDE_club_size_after_five_years_l94_9447

/-- Calculates the number of people in the club after a given number of years -/
def club_size (initial_size : ℕ) (years : ℕ) : ℕ :=
  match years with
  | 0 => initial_size
  | n + 1 => 4 * (club_size initial_size n - 7) + 7

theorem club_size_after_five_years :
  club_size 21 5 = 14343 := by
  sorry

#eval club_size 21 5

end NUMINAMATH_CALUDE_club_size_after_five_years_l94_9447


namespace NUMINAMATH_CALUDE_percent_of_x_l94_9418

theorem percent_of_x (x : ℝ) (h : x > 0) : (x / 5 + x / 25) / x = 24 / 100 := by
  sorry

end NUMINAMATH_CALUDE_percent_of_x_l94_9418


namespace NUMINAMATH_CALUDE_p_minus_m_equals_2010_l94_9425

-- Define the set of positive integers
def PositiveInt : Set ℕ := {n : ℕ | n > 0}

-- Define set M
def M : Set ℕ := {x ∈ PositiveInt | 1 ≤ x ∧ x ≤ 2009}

-- Define set P
def P : Set ℕ := {y ∈ PositiveInt | 2 ≤ y ∧ y ≤ 2010}

-- Define the set difference operation
def SetDifference (A B : Set ℕ) : Set ℕ := {x ∈ A | x ∉ B}

-- Theorem statement
theorem p_minus_m_equals_2010 : SetDifference P M = {2010} := by
  sorry

end NUMINAMATH_CALUDE_p_minus_m_equals_2010_l94_9425


namespace NUMINAMATH_CALUDE_odd_function_property_l94_9457

open Set

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_function_property
  (f : ℝ → ℝ)
  (f_diff : Differentiable ℝ f)
  (h_odd : is_odd f)
  (h_deriv : ∀ x ≤ 0, x * (deriv f x) < f (-x))
  (F : ℝ → ℝ)
  (h_F : ∀ x, F x = x * f x) :
  {x : ℝ | F 3 > F (2*x - 1)} = Ioo (-1) 2 := by
sorry

end NUMINAMATH_CALUDE_odd_function_property_l94_9457


namespace NUMINAMATH_CALUDE_unique_score_with_four_ways_l94_9499

/-- AMC scoring system -/
structure AMCScore where
  correct : ℕ
  unanswered : ℕ
  incorrect : ℕ
  total_questions : ℕ
  score : ℕ

/-- Predicate for valid AMC score -/
def is_valid_score (s : AMCScore) : Prop :=
  s.correct + s.unanswered + s.incorrect = s.total_questions ∧
  s.score = 7 * s.correct + 3 * s.unanswered

/-- Theorem: Unique score with exactly four distinct ways to achieve it -/
theorem unique_score_with_four_ways :
  ∃! S : ℕ, 
    (∃ scores : Finset AMCScore, 
      (∀ s ∈ scores, is_valid_score s ∧ s.total_questions = 30 ∧ s.score = S) ∧
      scores.card = 4 ∧
      (∀ s : AMCScore, is_valid_score s ∧ s.total_questions = 30 ∧ s.score = S → s ∈ scores)) ∧
    S = 240 := by
  sorry

end NUMINAMATH_CALUDE_unique_score_with_four_ways_l94_9499


namespace NUMINAMATH_CALUDE_candles_remaining_l94_9474

def total_candles : ℕ := 40
def alyssa_fraction : ℚ := 1/2
def chelsea_fraction : ℚ := 7/10

theorem candles_remaining (total : ℕ) (alyssa_frac chelsea_frac : ℚ) :
  total = total_candles →
  alyssa_frac = alyssa_fraction →
  chelsea_frac = chelsea_fraction →
  ↑total * (1 - alyssa_frac) * (1 - chelsea_frac) = 6 :=
by sorry

end NUMINAMATH_CALUDE_candles_remaining_l94_9474


namespace NUMINAMATH_CALUDE_new_cube_volume_l94_9407

theorem new_cube_volume (original_volume : ℝ) (scale_factor : ℝ) : 
  original_volume = 64 →
  scale_factor = 2 →
  (scale_factor ^ 3) * original_volume = 512 :=
by sorry

end NUMINAMATH_CALUDE_new_cube_volume_l94_9407


namespace NUMINAMATH_CALUDE_whale_prediction_correct_l94_9456

/-- The number of whales predicted for next year -/
def whales_next_year : ℕ := 8800

/-- The number of whales last year -/
def whales_last_year : ℕ := 4000

/-- The number of whales this year -/
def whales_this_year : ℕ := 2 * whales_last_year

/-- The predicted increase in the number of whales for next year -/
def predicted_increase : ℕ := whales_next_year - whales_this_year

theorem whale_prediction_correct : predicted_increase = 800 := by
  sorry

end NUMINAMATH_CALUDE_whale_prediction_correct_l94_9456


namespace NUMINAMATH_CALUDE_clinic_patient_count_l94_9406

theorem clinic_patient_count (original_count current_count diagnosed_count : ℕ) : 
  current_count = 2 * original_count →
  diagnosed_count = 13 →
  (4 : ℕ) * diagnosed_count = current_count →
  original_count = 26 := by
  sorry

end NUMINAMATH_CALUDE_clinic_patient_count_l94_9406


namespace NUMINAMATH_CALUDE_triangle_inequality_l94_9417

theorem triangle_inequality (a b x : ℝ) (ha : 0 < a) (hb : 0 < b) (hx : 0 < x) :
  (a = 4 ∧ b = 9) → (5 < x ∧ x < 13) ↔ (a + b > x ∧ a + x > b ∧ b + x > a) :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l94_9417


namespace NUMINAMATH_CALUDE_square_minus_one_l94_9449

theorem square_minus_one (x : ℤ) (h : x^2 = 1521) : (x + 1) * (x - 1) = 1520 := by
  sorry

end NUMINAMATH_CALUDE_square_minus_one_l94_9449


namespace NUMINAMATH_CALUDE_wedding_drinks_l94_9481

theorem wedding_drinks (total_guests : ℕ) (num_drink_types : ℕ) 
  (champagne_glasses_per_guest : ℕ) (champagne_servings_per_bottle : ℕ)
  (wine_glasses_per_guest : ℕ) (wine_servings_per_bottle : ℕ)
  (juice_glasses_per_guest : ℕ) (juice_servings_per_bottle : ℕ)
  (h1 : total_guests = 120)
  (h2 : num_drink_types = 3)
  (h3 : champagne_glasses_per_guest = 2)
  (h4 : champagne_servings_per_bottle = 6)
  (h5 : wine_glasses_per_guest = 1)
  (h6 : wine_servings_per_bottle = 5)
  (h7 : juice_glasses_per_guest = 1)
  (h8 : juice_servings_per_bottle = 4) :
  let guests_per_drink_type := total_guests / num_drink_types
  let juice_bottles_needed := (guests_per_drink_type * juice_glasses_per_guest + juice_servings_per_bottle - 1) / juice_servings_per_bottle
  juice_bottles_needed = 10 := by
sorry

end NUMINAMATH_CALUDE_wedding_drinks_l94_9481


namespace NUMINAMATH_CALUDE_roots_bound_implies_b_bound_l94_9428

-- Define the function f(x) = x^2 + ax + b
def f (a b x : ℝ) : ℝ := x^2 + a*x + b

-- Define the theorem
theorem roots_bound_implies_b_bound
  (a b x₁ x₂ : ℝ)
  (h1 : f a b x₁ = 0)  -- x₁ is a root of f
  (h2 : f a b x₂ = 0)  -- x₂ is a root of f
  (h3 : x₁ ≠ x₂)       -- The roots are distinct
  (h4 : |x₁| + |x₂| ≤ 2) :
  b ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_roots_bound_implies_b_bound_l94_9428


namespace NUMINAMATH_CALUDE_function_intersection_and_tangency_l94_9489

/-- Given two functions f and g, prove that under certain conditions, 
    the coefficients a, b, and c have specific values. -/
theorem function_intersection_and_tangency 
  (f : ℝ → ℝ) 
  (g : ℝ → ℝ) 
  (a b c : ℝ) 
  (h1 : ∀ x, f x = 2 * x^3 + a * x)
  (h2 : ∀ x, g x = b * x^2 + c)
  (h3 : f 2 = 0)
  (h4 : g 2 = 0)
  (h5 : (deriv f) 2 = (deriv g) 2) : 
  a = -8 ∧ b = 4 ∧ c = -16 := by
  sorry

end NUMINAMATH_CALUDE_function_intersection_and_tangency_l94_9489


namespace NUMINAMATH_CALUDE_unique_a_value_l94_9401

def A (a : ℚ) : Set ℚ := {a + 2, 2 * a^2 + a}

theorem unique_a_value : ∃! a : ℚ, 3 ∈ A a ∧ a = -3/2 := by sorry

end NUMINAMATH_CALUDE_unique_a_value_l94_9401


namespace NUMINAMATH_CALUDE_hyperbola_equation_l94_9491

theorem hyperbola_equation (a b c : ℝ) (h1 : 2 * a = 2) (h2 : c / a = Real.sqrt 2) :
  (∀ x y, x^2 - y^2 = 1 ∨ y^2 - x^2 = 1) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l94_9491


namespace NUMINAMATH_CALUDE_janice_stair_usage_l94_9451

/-- Represents the number of times Janice goes up or down the stairs in a day -/
structure StairUsage where
  up : Nat
  down : Nat

/-- Calculates the total number of flights of stairs walked -/
def totalFlights (usage : StairUsage) (flightsPerTrip : Nat) : Nat :=
  (usage.up + usage.down) * flightsPerTrip

theorem janice_stair_usage (flightsPerTrip : Nat) (totalFlightsWalked : Nat) :
  flightsPerTrip = 3 →
  totalFlightsWalked = 24 →
  ∃ (usage : StairUsage),
    usage.up = 5 ∧
    totalFlights usage flightsPerTrip = totalFlightsWalked ∧
    usage.down = 3 := by
  sorry

#check janice_stair_usage

end NUMINAMATH_CALUDE_janice_stair_usage_l94_9451


namespace NUMINAMATH_CALUDE_custom_mul_four_three_l94_9464

/-- Custom multiplication operation -/
def custom_mul (a b : ℝ) : ℝ := a^2 + a*b - b^2

/-- Theorem stating that 4 * 3 = 19 under the custom multiplication -/
theorem custom_mul_four_three : custom_mul 4 3 = 19 := by sorry

end NUMINAMATH_CALUDE_custom_mul_four_three_l94_9464


namespace NUMINAMATH_CALUDE_servant_payment_proof_l94_9480

/-- Calculates the cash payment for a servant who leaves early -/
def servant_payment (total_salary : ℚ) (turban_value : ℚ) (months_worked : ℚ) : ℚ :=
  (months_worked / 12) * total_salary - turban_value

/-- Proves that a servant working 9 months with given conditions receives Rs. 60 -/
theorem servant_payment_proof :
  let total_salary : ℚ := 120
  let turban_value : ℚ := 30
  let months_worked : ℚ := 9
  servant_payment total_salary turban_value months_worked = 60 := by
sorry

end NUMINAMATH_CALUDE_servant_payment_proof_l94_9480


namespace NUMINAMATH_CALUDE_multiple_of_x_l94_9446

theorem multiple_of_x (x y m : ℤ) : 
  (4 * x + y = 34) →
  (m * x - y = 20) →
  (y^2 = 4) →
  m = 2 := by sorry

end NUMINAMATH_CALUDE_multiple_of_x_l94_9446


namespace NUMINAMATH_CALUDE_smallest_third_altitude_l94_9478

/-- Represents a scalene triangle with altitudes --/
structure ScaleneTriangle where
  -- The lengths of the three sides
  a : ℝ
  b : ℝ
  c : ℝ
  -- The lengths of the three altitudes
  h_a : ℝ
  h_b : ℝ
  h_c : ℝ
  -- Conditions for a scalene triangle
  scalene : a ≠ b ∧ b ≠ c ∧ a ≠ c
  -- Triangle inequality
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b
  -- Area equality for altitudes
  area_equality : a * h_a = b * h_b ∧ b * h_b = c * h_c

/-- The theorem stating the smallest possible integer length for the third altitude --/
theorem smallest_third_altitude (t : ScaleneTriangle) 
  (h1 : t.h_a = 6 ∨ t.h_b = 6 ∨ t.h_c = 6)
  (h2 : t.h_a = 8 ∨ t.h_b = 8 ∨ t.h_c = 8)
  (h3 : ∃ (n : ℕ), t.h_a = n ∨ t.h_b = n ∨ t.h_c = n) :
  ∃ (h : ScaleneTriangle), h.h_a = 6 ∧ h.h_b = 8 ∧ h.h_c = 2 :=
sorry

end NUMINAMATH_CALUDE_smallest_third_altitude_l94_9478


namespace NUMINAMATH_CALUDE_shaded_area_calculation_l94_9462

/-- The area of the shaded regions in a figure with two rectangles and two semicircles removed -/
theorem shaded_area_calculation (small_radius : ℝ) (large_radius : ℝ)
  (h_small : small_radius = 3)
  (h_large : large_radius = 6) :
  let small_rect_area := small_radius * (2 * small_radius)
  let large_rect_area := large_radius * (2 * large_radius)
  let small_semicircle_area := π * small_radius^2 / 2
  let large_semicircle_area := π * large_radius^2 / 2
  small_rect_area + large_rect_area - small_semicircle_area - large_semicircle_area = 90 - 45 * π / 2 :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_calculation_l94_9462


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l94_9453

theorem complex_modulus_problem (a b : ℝ) (i : ℂ) (h1 : i * i = -1) (h2 : a + 2 * i = 2 - b * i) :
  Complex.abs (a + b * i) = 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l94_9453


namespace NUMINAMATH_CALUDE_polynomial_simplification_l94_9459

theorem polynomial_simplification (x : ℝ) : 
  (2 * x^6 + 3 * x^5 + x^4 + x^3 + x + 10) - (x^6 + 4 * x^5 + 2 * x^4 - x^3 + 12) = 
  x^6 - x^5 - x^4 + 2 * x^3 + x - 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l94_9459


namespace NUMINAMATH_CALUDE_unique_A_value_l94_9400

theorem unique_A_value (A : ℝ) (x₁ x₂ : ℂ) 
  (h_distinct : x₁ ≠ x₂)
  (h_eq1 : x₁ * (x₁ + 1) = A)
  (h_eq2 : x₂ * (x₂ + 1) = A)
  (h_eq3 : A * x₁^4 + 3 * x₁^3 + 5 * x₁ = x₂^4 + 3 * x₂^3 + 5 * x₂) :
  A = -7 := by
  sorry

end NUMINAMATH_CALUDE_unique_A_value_l94_9400


namespace NUMINAMATH_CALUDE_floor_sum_example_l94_9497

theorem floor_sum_example : ⌊(17.2 : ℝ)⌋ + ⌊(-17.2 : ℝ)⌋ = -1 := by sorry

end NUMINAMATH_CALUDE_floor_sum_example_l94_9497


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l94_9463

/-- Given a geometric sequence {a_n} where the sum of the first n terms
    is S_n = 3 × 2^n + m, prove that the common ratio is 2. -/
theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ) 
  (m : ℝ) 
  (h1 : ∀ n, S n = 3 * 2^n + m) 
  (h2 : ∀ n, n ≥ 2 → a n = S n - S (n-1)) :
  ∀ n, n ≥ 2 → a (n+1) / a n = 2 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l94_9463


namespace NUMINAMATH_CALUDE_yo_yo_count_l94_9461

theorem yo_yo_count 
  (x y z w : ℕ) 
  (h1 : x + y + w = 80)
  (h2 : (3/5 : ℚ) * 300 + (1/5 : ℚ) * 300 = x + y + z + w + 15)
  (h3 : x + y + z + w = 300 - ((3/5 : ℚ) * 300 + (1/5 : ℚ) * 300)) :
  z = 145 := by
  sorry

#check yo_yo_count

end NUMINAMATH_CALUDE_yo_yo_count_l94_9461


namespace NUMINAMATH_CALUDE_range_of_f_l94_9403

noncomputable def f (x : ℝ) : ℝ := 3 * (x - 4)

theorem range_of_f :
  ∀ y : ℝ, y ≠ -27 → ∃ x : ℝ, x ≠ -5 ∧ f x = y :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l94_9403


namespace NUMINAMATH_CALUDE_negative_300_coterminal_with_60_l94_9402

/-- An angle is coterminal with 60 degrees if it can be expressed as k * 360 + 60, where k is an integer -/
def is_coterminal_with_60 (angle : ℝ) : Prop :=
  ∃ k : ℤ, angle = k * 360 + 60

/-- Theorem stating that -300 degrees is coterminal with 60 degrees -/
theorem negative_300_coterminal_with_60 : is_coterminal_with_60 (-300) := by
  sorry

end NUMINAMATH_CALUDE_negative_300_coterminal_with_60_l94_9402


namespace NUMINAMATH_CALUDE_probability_white_given_popped_l94_9468

-- Define the probabilities
def P_white : ℝ := 0.4
def P_yellow : ℝ := 0.4
def P_red : ℝ := 0.2
def P_pop_given_white : ℝ := 0.7
def P_pop_given_yellow : ℝ := 0.5
def P_pop_given_red : ℝ := 0

-- Define the theorem
theorem probability_white_given_popped :
  let P_popped : ℝ := P_pop_given_white * P_white + P_pop_given_yellow * P_yellow + P_pop_given_red * P_red
  (P_pop_given_white * P_white) / P_popped = 7 / 12 := by
  sorry

end NUMINAMATH_CALUDE_probability_white_given_popped_l94_9468


namespace NUMINAMATH_CALUDE_jerry_book_pages_l94_9471

/-- Calculates the total number of pages in a book given the number of pages read on Saturday, Sunday, and the number of pages remaining. -/
def total_pages (pages_saturday : ℕ) (pages_sunday : ℕ) (pages_remaining : ℕ) : ℕ :=
  pages_saturday + pages_sunday + pages_remaining

/-- Theorem stating that the total number of pages in Jerry's book is 93. -/
theorem jerry_book_pages : total_pages 30 20 43 = 93 := by
  sorry

end NUMINAMATH_CALUDE_jerry_book_pages_l94_9471


namespace NUMINAMATH_CALUDE_large_balls_can_make_l94_9429

/-- The number of rubber bands in a small ball -/
def small_ball_bands : ℕ := 50

/-- The number of rubber bands in a large ball -/
def large_ball_bands : ℕ := 300

/-- The total number of rubber bands Michael brought to class -/
def total_bands : ℕ := 5000

/-- The number of small balls Michael has already made -/
def small_balls_made : ℕ := 22

/-- The number of large balls Michael can make with the remaining rubber bands -/
theorem large_balls_can_make : ℕ := by
  sorry

end NUMINAMATH_CALUDE_large_balls_can_make_l94_9429


namespace NUMINAMATH_CALUDE_smallest_divisor_of_427395_l94_9431

theorem smallest_divisor_of_427395 : 
  ∀ d : ℕ, d > 0 ∧ d < 5 → ¬(427395 % d = 0) ∧ 427395 % 5 = 0 := by sorry

end NUMINAMATH_CALUDE_smallest_divisor_of_427395_l94_9431


namespace NUMINAMATH_CALUDE_right_triangle_345_ratio_l94_9420

theorem right_triangle_345_ratio (a b c : ℝ) (h_positive : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_ratio : a / b = 3 / 4 ∧ b / c = 4 / 5) : a^2 + b^2 = c^2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_345_ratio_l94_9420


namespace NUMINAMATH_CALUDE_quadratic_polynomial_condition_l94_9444

/-- Given a polynomial of the form 2a*x^4 + 5a*x^3 - 13x^2 - x^4 + 2021 + 2x + b*x^3 - b*x^4 - 13x^3,
    if it is a quadratic polynomial, then a^2 + b^2 = 13 -/
theorem quadratic_polynomial_condition (a b : ℝ) : 
  (∀ x, (2*a - 1 - b) * x^4 + (5*a + b - 13) * x^3 - 13*x^2 + 2*x + 2021 = 0 → 
        ∃ p q r : ℝ, ∀ x, p*x^2 + q*x + r = 0) →
  a^2 + b^2 = 13 := by sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_condition_l94_9444


namespace NUMINAMATH_CALUDE_inequality_impossibility_l94_9430

theorem inequality_impossibility (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  ¬(a + b < c + d ∧ (a + b) * (c + d) < a * b + c * d ∧ (a + b) * c * d < a * b * (c + d)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_impossibility_l94_9430


namespace NUMINAMATH_CALUDE_equal_water_after_operations_l94_9409

theorem equal_water_after_operations (x : ℝ) (h : x > 0) :
  let barrel1 := x * 0.9 * 1.1
  let barrel2 := x * 1.1 * 0.9
  barrel1 = barrel2 := by sorry

end NUMINAMATH_CALUDE_equal_water_after_operations_l94_9409


namespace NUMINAMATH_CALUDE_inequality_proof_l94_9452

theorem inequality_proof (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) :
  (1/2) * (a + b)^2 + (1/4) * (a + b) ≥ a * Real.sqrt b + b * Real.sqrt a :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l94_9452


namespace NUMINAMATH_CALUDE_negation_equivalence_l94_9472

theorem negation_equivalence :
  (¬ ∃ (x₀ y₀ : ℝ), x₀^2 + y₀^2 - 1 ≤ 0) ↔ (∀ (x y : ℝ), x^2 + y^2 - 1 > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l94_9472


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l94_9419

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | x < 0}

-- Define set B
def B : Set ℝ := {x | x ≤ -1}

-- Theorem statement
theorem intersection_A_complement_B :
  A ∩ (Set.compl B) = {x | -1 < x ∧ x < 0} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l94_9419


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l94_9410

/-- Given that the solution set of ax^2 + bx + c > 0 is {x | 2 < x < 3},
    prove that the solution set of ax^2 - bx + c > 0 is {x | -3 < x < -2} -/
theorem quadratic_inequality_solution_sets
  (a b c : ℝ)
  (h : ∀ x, ax^2 + b*x + c > 0 ↔ 2 < x ∧ x < 3) :
  ∀ x, ax^2 - b*x + c > 0 ↔ -3 < x ∧ x < -2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l94_9410


namespace NUMINAMATH_CALUDE_q_div_p_equals_550_l94_9487

def total_slips : ℕ := 60
def numbers_per_slip : ℕ := 12
def slips_per_number : ℕ := 5
def drawn_slips : ℕ := 5

def p : ℚ := (numbers_per_slip : ℚ) / (Nat.choose total_slips drawn_slips)

def q : ℚ := (Nat.choose numbers_per_slip 2 * Nat.choose slips_per_number 3 * Nat.choose slips_per_number 2 : ℚ) / (Nat.choose total_slips drawn_slips)

theorem q_div_p_equals_550 : q / p = 550 := by sorry

end NUMINAMATH_CALUDE_q_div_p_equals_550_l94_9487


namespace NUMINAMATH_CALUDE_tangent_point_on_circle_l94_9460

theorem tangent_point_on_circle (a : ℝ) : 
  ((-1 - 1)^2 + a^2 = 4) ↔ (a = 0) := by sorry

end NUMINAMATH_CALUDE_tangent_point_on_circle_l94_9460


namespace NUMINAMATH_CALUDE_smallest_four_digit_unique_divisible_by_digits_with_five_l94_9490

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def has_unique_digits (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.length = 4 ∧ digits.toFinset.card = 4

def divisible_by_digits (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 10, d ≠ 0 → n % d = 0

def includes_digit_five (n : ℕ) : Prop :=
  5 ∈ n.digits 10

theorem smallest_four_digit_unique_divisible_by_digits_with_five :
  ∀ n : ℕ, is_four_digit n →
           has_unique_digits n →
           divisible_by_digits n →
           includes_digit_five n →
           1560 ≤ n :=
sorry

end NUMINAMATH_CALUDE_smallest_four_digit_unique_divisible_by_digits_with_five_l94_9490


namespace NUMINAMATH_CALUDE_cost_per_roof_tile_is_10_l94_9434

/-- Represents the construction costs for a house. -/
structure ConstructionCosts where
  landCostPerSqMeter : ℕ
  brickCostPer1000 : ℕ
  requiredLandArea : ℕ
  requiredBricks : ℕ
  requiredRoofTiles : ℕ
  totalCost : ℕ

/-- Calculates the cost per roof tile given the construction costs. -/
def costPerRoofTile (costs : ConstructionCosts) : ℕ :=
  let landCost := costs.landCostPerSqMeter * costs.requiredLandArea
  let brickCost := (costs.requiredBricks / 1000) * costs.brickCostPer1000
  let roofTileCost := costs.totalCost - (landCost + brickCost)
  roofTileCost / costs.requiredRoofTiles

/-- Theorem stating that the cost per roof tile is $10 given the specified construction costs. -/
theorem cost_per_roof_tile_is_10 (costs : ConstructionCosts)
    (h1 : costs.landCostPerSqMeter = 50)
    (h2 : costs.brickCostPer1000 = 100)
    (h3 : costs.requiredLandArea = 2000)
    (h4 : costs.requiredBricks = 10000)
    (h5 : costs.requiredRoofTiles = 500)
    (h6 : costs.totalCost = 106000) :
    costPerRoofTile costs = 10 := by
  sorry

end NUMINAMATH_CALUDE_cost_per_roof_tile_is_10_l94_9434


namespace NUMINAMATH_CALUDE_square_point_distance_probability_l94_9476

-- Define the square
def Square := {p : ℝ × ℝ | (0 ≤ p.1 ∧ p.1 ≤ 2 ∧ (p.2 = 0 ∨ p.2 = 2)) ∨ (0 ≤ p.2 ∧ p.2 ≤ 2 ∧ (p.1 = 0 ∨ p.1 = 2))}

-- Define the probability function
noncomputable def probability : ℝ := sorry

-- Define the gcd function
def gcd (a b c : ℕ) : ℕ := sorry

-- State the theorem
theorem square_point_distance_probability :
  ∃ (a b c : ℕ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    gcd a b c = 1 ∧
    probability = (a - b * Real.pi) / c ∧
    a = 28 ∧ b = 1 ∧ c = 1 := by sorry

end NUMINAMATH_CALUDE_square_point_distance_probability_l94_9476


namespace NUMINAMATH_CALUDE_a_beats_b_by_26_meters_l94_9482

/-- A beats B by 26 meters in a race -/
theorem a_beats_b_by_26_meters 
  (race_distance : ℝ) 
  (a_time : ℝ) 
  (b_time : ℝ) 
  (h1 : race_distance = 130)
  (h2 : a_time = 20)
  (h3 : b_time = 25) :
  race_distance - (race_distance / b_time * a_time) = 26 := by
  sorry

end NUMINAMATH_CALUDE_a_beats_b_by_26_meters_l94_9482


namespace NUMINAMATH_CALUDE_min_p_plus_q_l94_9421

theorem min_p_plus_q (p q : ℕ+) (h : 162 * p = q^3) : 
  ∀ (p' q' : ℕ+), 162 * p' = q'^3 → p + q ≤ p' + q' :=
sorry

end NUMINAMATH_CALUDE_min_p_plus_q_l94_9421


namespace NUMINAMATH_CALUDE_integer_fraction_characterization_l94_9495

theorem integer_fraction_characterization (p n : ℕ) :
  Nat.Prime p → n > 0 →
  (∃ k : ℕ, (n^p + 1 : ℕ) = k * (p^n + 1)) ↔
  ((p = 2 ∧ (n = 2 ∨ n = 4)) ∨ (p > 2 ∧ n = p)) := by
  sorry

end NUMINAMATH_CALUDE_integer_fraction_characterization_l94_9495


namespace NUMINAMATH_CALUDE_area_between_concentric_circles_l94_9405

/-- Given three concentric circles with radii r, s, and t, where r > s > t,
    and p, q as defined in the problem, prove that the area between the
    largest and smallest circles is π(p² + q²). -/
theorem area_between_concentric_circles
  (r s t p q : ℝ)
  (h_order : r > s ∧ s > t)
  (h_p : p = (r^2 - s^2).sqrt)
  (h_q : q = (s^2 - t^2).sqrt) :
  π * (r^2 - t^2) = π * (p^2 + q^2) := by
  sorry

end NUMINAMATH_CALUDE_area_between_concentric_circles_l94_9405


namespace NUMINAMATH_CALUDE_light_path_length_l94_9492

-- Define the cube side length
def cube_side : ℝ := 10

-- Define the reflection point coordinates relative to the face
def reflect_x : ℝ := 4
def reflect_y : ℝ := 6

-- Define the number of reflections needed
def num_reflections : ℕ := 10

-- Theorem statement
theorem light_path_length :
  let path_length := (num_reflections : ℝ) * Real.sqrt (cube_side^2 + reflect_x^2 + reflect_y^2)
  path_length = 10 * Real.sqrt 152 :=
by sorry

end NUMINAMATH_CALUDE_light_path_length_l94_9492


namespace NUMINAMATH_CALUDE_system_solutions_l94_9473

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- Define the system of equations
def system (x y : ℝ) : Prop :=
  lg (x^2 + y^2) = 2 - lg 5 ∧
  lg (x + y) + lg (x - y) = lg 1.2 + 1 ∧
  x + y > 0 ∧
  x - y > 0

-- Theorem statement
theorem system_solutions :
  ∀ x y : ℝ, system x y ↔ ((x = 4 ∧ y = 2) ∨ (x = 4 ∧ y = -2)) :=
by sorry

end NUMINAMATH_CALUDE_system_solutions_l94_9473


namespace NUMINAMATH_CALUDE_jesse_book_reading_l94_9413

theorem jesse_book_reading (pages_read pages_left : ℕ) 
  (h1 : pages_read = 83) 
  (h2 : pages_left = 166) : 
  (pages_read : ℚ) / (pages_read + pages_left) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_jesse_book_reading_l94_9413


namespace NUMINAMATH_CALUDE_idaho_to_nevada_distance_l94_9443

/-- Represents the road trip from Washington to Nevada via Idaho -/
structure RoadTrip where
  wash_to_idaho : ℝ     -- Distance from Washington to Idaho
  idaho_to_nevada : ℝ   -- Distance from Idaho to Nevada (to be proven)
  speed_to_idaho : ℝ    -- Speed from Washington to Idaho
  speed_to_nevada : ℝ   -- Speed from Idaho to Nevada
  total_time : ℝ        -- Total travel time

/-- The road trip satisfies the given conditions -/
def satisfies_conditions (trip : RoadTrip) : Prop :=
  trip.wash_to_idaho = 640 ∧
  trip.speed_to_idaho = 80 ∧
  trip.speed_to_nevada = 50 ∧
  trip.total_time = 19 ∧
  trip.total_time = trip.wash_to_idaho / trip.speed_to_idaho + trip.idaho_to_nevada / trip.speed_to_nevada

theorem idaho_to_nevada_distance (trip : RoadTrip) 
  (h : satisfies_conditions trip) : trip.idaho_to_nevada = 550 := by
  sorry

end NUMINAMATH_CALUDE_idaho_to_nevada_distance_l94_9443


namespace NUMINAMATH_CALUDE_alice_yard_side_length_l94_9411

/-- Given that Alice needs to buy 12 bushes to plant around three sides of her yard,
    and each bush fills 4 feet, prove that each side of her yard is 16 feet long. -/
theorem alice_yard_side_length
  (num_bushes : ℕ)
  (bush_length : ℕ)
  (num_sides : ℕ)
  (h1 : num_bushes = 12)
  (h2 : bush_length = 4)
  (h3 : num_sides = 3) :
  (num_bushes * bush_length) / num_sides = 16 := by
  sorry

end NUMINAMATH_CALUDE_alice_yard_side_length_l94_9411


namespace NUMINAMATH_CALUDE_fold_square_problem_l94_9416

/-- Given a square ABCD with side length 8 cm, if corner C is folded to point E
    which is one-third of the way along AD from D, and F is the point where the
    fold intersects CD, then the length of FD is 32/9 cm. -/
theorem fold_square_problem (A B C D E F G : ℝ × ℝ) : 
  (∀ (X Y : ℝ × ℝ), dist X Y = dist A B → dist X Y = 8) →  -- Square side length is 8
  dist A D = 8 →  -- AD is a side of the square
  dist D E = 8/3 →  -- E is one-third along AD from D
  F.1 = D.1 ∧ F.2 ≤ D.2 ∧ F.2 ≥ C.2 →  -- F is on CD
  dist C E = dist C F →  -- C is folded onto E
  dist F D = 32/9 := by
  sorry


end NUMINAMATH_CALUDE_fold_square_problem_l94_9416


namespace NUMINAMATH_CALUDE_class_size_l94_9488

theorem class_size (n : ℕ) (h1 : n > 0) :
  (∃ student_in_middle_row : ℕ, 
    student_in_middle_row > 0 ∧ 
    student_in_middle_row ≤ n ∧
    student_in_middle_row = 6 ∧ 
    n + 1 - student_in_middle_row = 7) →
  3 * n = 36 :=
by sorry

end NUMINAMATH_CALUDE_class_size_l94_9488


namespace NUMINAMATH_CALUDE_pq_passes_through_centroid_l94_9439

-- Define the points
variable (A B C D E F P Q : ℝ × ℝ)

-- Define the properties of the triangle and points
def is_right_triangle (A B C : ℝ × ℝ) : Prop :=
  (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0

def is_altitude_foot (A B C D : ℝ × ℝ) : Prop :=
  (B.1 - A.1) * (D.1 - A.1) + (B.2 - A.2) * (D.2 - A.2) = 0

def is_centroid (E A C D : ℝ × ℝ) : Prop :=
  E.1 = (A.1 + C.1 + D.1) / 3 ∧ E.2 = (A.2 + C.2 + D.2) / 3

def is_perpendicular (A B C : ℝ × ℝ) : Prop :=
  (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0

def equal_distance (A B C : ℝ × ℝ) : Prop :=
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = (C.1 - A.1)^2 + (C.2 - A.2)^2

def line_passes_through_point (P Q X : ℝ × ℝ) : Prop :=
  (Q.2 - P.2) * (X.1 - P.1) = (Q.1 - P.1) * (X.2 - P.2)

def centroid (G A B C : ℝ × ℝ) : Prop :=
  G.1 = (A.1 + B.1 + C.1) / 3 ∧ G.2 = (A.2 + B.2 + C.2) / 3

-- State the theorem
theorem pq_passes_through_centroid
  (h1 : is_right_triangle A B C)
  (h2 : is_altitude_foot A B C D)
  (h3 : is_centroid E A C D)
  (h4 : is_centroid F B C D)
  (h5 : is_perpendicular C E P)
  (h6 : equal_distance C P A)
  (h7 : is_perpendicular C F Q)
  (h8 : equal_distance C Q B) :
  ∃ G, centroid G A B C ∧ line_passes_through_point P Q G :=
sorry

end NUMINAMATH_CALUDE_pq_passes_through_centroid_l94_9439


namespace NUMINAMATH_CALUDE_linear_function_composition_l94_9426

-- Define a linear function
def LinearFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x + b

-- State the theorem
theorem linear_function_composition (f : ℝ → ℝ) :
  LinearFunction f → (∀ x, f (f x) = 4 * x + 1) →
  (∀ x, f x = 2 * x + 1/3) ∨ (∀ x, f x = -2 * x - 1) :=
by sorry

end NUMINAMATH_CALUDE_linear_function_composition_l94_9426


namespace NUMINAMATH_CALUDE_expression_equality_l94_9486

theorem expression_equality (a b c : ℤ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) : 
  (a^2 - b^2) / (a * b) - (a * b - b * c) / (a * b - a * c) = (c * a - (c - 1) * b) / b :=
by sorry

end NUMINAMATH_CALUDE_expression_equality_l94_9486


namespace NUMINAMATH_CALUDE_ruble_cashback_preferable_l94_9450

/-- Represents the value of a cashback reward -/
structure CashbackValue where
  nominal : ℝ
  usability : ℝ -- Represents the fraction of nominal value that can be effectively used (0 to 1)

/-- Cashback in rubles -/
def rubleCashback (amount : ℝ) : CashbackValue :=
  { nominal := amount, usability := 1 }

/-- Cashback in bonus points -/
def bonusPointsCashback (amount : ℝ) (usabilityFactor : ℝ) : CashbackValue :=
  { nominal := amount, usability := usabilityFactor }

/-- Theorem: Ruble cashback value is greater than or equal to bonus points cashback value -/
theorem ruble_cashback_preferable (amount : ℝ) (bonusUsabilityFactor : ℝ) 
    (h1 : amount > 0) (h2 : 0 ≤ bonusUsabilityFactor ∧ bonusUsabilityFactor ≤ 1) : 
    (rubleCashback amount).nominal * (rubleCashback amount).usability ≥ 
    (bonusPointsCashback amount bonusUsabilityFactor).nominal * (bonusPointsCashback amount bonusUsabilityFactor).usability :=
  sorry

end NUMINAMATH_CALUDE_ruble_cashback_preferable_l94_9450


namespace NUMINAMATH_CALUDE_soccer_team_selection_l94_9423

/-- The number of ways to choose an ordered selection of 5 players from a team of 15 players -/
def choose_squad (team_size : Nat) : Nat :=
  team_size * (team_size - 1) * (team_size - 2) * (team_size - 3) * (team_size - 4)

/-- Theorem stating that choosing 5 players from a team of 15 results in 360,360 possibilities -/
theorem soccer_team_selection :
  choose_squad 15 = 360360 := by
  sorry

end NUMINAMATH_CALUDE_soccer_team_selection_l94_9423


namespace NUMINAMATH_CALUDE_phone_profit_fraction_l94_9442

theorem phone_profit_fraction (num_phones : ℕ) (initial_investment : ℚ) (selling_price : ℚ) :
  num_phones = 200 →
  initial_investment = 3000 →
  selling_price = 20 →
  (num_phones * selling_price - initial_investment) / initial_investment = 1/3 := by
sorry

end NUMINAMATH_CALUDE_phone_profit_fraction_l94_9442


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_n_l94_9483

/-- 
A polynomial 4x^2 + 12x + n is a perfect square trinomial if and only if 
there exist real numbers a and b such that 4x^2 + 12x + n = (ax + b)^2 for all x
-/
def IsPerfectSquareTrinomial (n : ℝ) : Prop :=
  ∃ a b : ℝ, ∀ x : ℝ, 4 * x^2 + 12 * x + n = (a * x + b)^2

/-- 
If 4x^2 + 12x + n is a perfect square trinomial, then n = 9
-/
theorem perfect_square_trinomial_n (n : ℝ) :
  IsPerfectSquareTrinomial n → n = 9 := by
  sorry


end NUMINAMATH_CALUDE_perfect_square_trinomial_n_l94_9483


namespace NUMINAMATH_CALUDE_adam_remaining_candy_l94_9475

/-- The number of boxes of chocolate candy Adam initially bought -/
def initial_boxes : ℕ := 13

/-- The number of boxes Adam gave to his little brother -/
def given_boxes : ℕ := 7

/-- The number of pieces of candy in each box -/
def pieces_per_box : ℕ := 6

/-- Theorem: Adam still had 36 pieces of chocolate candy -/
theorem adam_remaining_candy : 
  (initial_boxes - given_boxes) * pieces_per_box = 36 := by
  sorry

end NUMINAMATH_CALUDE_adam_remaining_candy_l94_9475


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l94_9404

theorem quadratic_equation_solution (x : ℝ) : 
  -x^2 - (-18 + 12) * x - 8 = -(x - 2) * (x - 4) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l94_9404


namespace NUMINAMATH_CALUDE_cube_with_holes_surface_area_l94_9479

/-- Calculates the total surface area of a cube with holes --/
def totalSurfaceArea (cubeEdgeLength : ℝ) (holeEdgeLength : ℝ) : ℝ :=
  let originalSurfaceArea := 6 * cubeEdgeLength^2
  let holeArea := 6 * holeEdgeLength^2
  let exposedInsideArea := 6 * 4 * holeEdgeLength^2
  originalSurfaceArea - holeArea + exposedInsideArea

/-- The total surface area of a cube with edge length 4 and holes of side length 2 is 168 --/
theorem cube_with_holes_surface_area :
  totalSurfaceArea 4 2 = 168 := by
  sorry

#eval totalSurfaceArea 4 2

end NUMINAMATH_CALUDE_cube_with_holes_surface_area_l94_9479


namespace NUMINAMATH_CALUDE_village_population_l94_9455

theorem village_population (P : ℕ) : 
  (((((P * 95 / 100) * 85 / 100) * 93 / 100) * 80 / 100) * 90 / 100) * 75 / 100 = 3553 →
  P = 9262 := by
  sorry

end NUMINAMATH_CALUDE_village_population_l94_9455


namespace NUMINAMATH_CALUDE_inverse_proportion_ratio_l94_9435

-- Define the inverse proportionality relation
def inversely_proportional (a b : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x, a x * b x = k

theorem inverse_proportion_ratio 
  (a b : ℝ → ℝ) (a₁ a₂ b₁ b₂ : ℝ) :
  inversely_proportional a b →
  a₁ ≠ 0 → a₂ ≠ 0 → b₁ ≠ 0 → b₂ ≠ 0 →
  a₁ / a₂ = 3 / 4 →
  b₁ - b₂ = 5 →
  b₁ / b₂ = 4 / 3 := by
    sorry

end NUMINAMATH_CALUDE_inverse_proportion_ratio_l94_9435


namespace NUMINAMATH_CALUDE_cookie_distribution_l94_9494

theorem cookie_distribution (bags : ℕ) (cookies_per_bag : ℕ) (damaged_cookies : ℕ) (people : ℕ) :
  bags = 295 →
  cookies_per_bag = 738 →
  damaged_cookies = 13 →
  people = 125 →
  (bags * cookies_per_bag - bags * damaged_cookies) / people = 1711 :=
by sorry

end NUMINAMATH_CALUDE_cookie_distribution_l94_9494


namespace NUMINAMATH_CALUDE_total_miles_walked_l94_9408

/-- The number of ladies in the walking group -/
def num_ladies : ℕ := 5

/-- The number of miles walked together by the group each day -/
def group_miles_per_day : ℕ := 3

/-- The number of additional miles Jamie walks per day -/
def jamie_additional_miles_per_day : ℕ := 2

/-- The number of days they walk per week -/
def days_per_week : ℕ := 6

/-- The total miles walked by the ladies in 6 days -/
def total_miles : ℕ := num_ladies * group_miles_per_day * days_per_week + jamie_additional_miles_per_day * days_per_week

theorem total_miles_walked :
  total_miles = 120 := by sorry

end NUMINAMATH_CALUDE_total_miles_walked_l94_9408


namespace NUMINAMATH_CALUDE_probability_of_composite_product_l94_9465

/-- The number of sides on each die -/
def sides : ℕ := 8

/-- The number of dice rolled -/
def num_dice : ℕ := 6

/-- The set of prime numbers that can appear on an 8-sided die -/
def primes_on_die : Set ℕ := {2, 3, 5, 7}

/-- The total number of possible outcomes when rolling 6 8-sided dice -/
def total_outcomes : ℕ := sides ^ num_dice

/-- The number of ways to roll all 1's -/
def all_ones : ℕ := 1

/-- The number of ways to roll five 1's and one prime on a 6-sided die -/
def five_ones_one_prime : ℕ := 4 * 6

/-- The total number of favorable outcomes (product is 1 or prime) -/
def favorable_outcomes : ℕ := all_ones + five_ones_one_prime

/-- The probability that the product is composite -/
def prob_composite : ℚ := 1 - (favorable_outcomes : ℚ) / total_outcomes

theorem probability_of_composite_product :
  prob_composite = 262119 / 262144 := by sorry

end NUMINAMATH_CALUDE_probability_of_composite_product_l94_9465


namespace NUMINAMATH_CALUDE_parabola_coefficient_sum_l94_9454

/-- Represents a parabola of the form x = ay^2 + by + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The x-coordinate of a point on the parabola given its y-coordinate -/
def Parabola.x_coord (p : Parabola) (y : ℝ) : ℝ :=
  p.a * y^2 + p.b * y + p.c

theorem parabola_coefficient_sum (p : Parabola) :
  p.x_coord (-4) = 5 →
  p.x_coord (-2) = 3 →
  p.a + p.b + p.c = -15/2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_coefficient_sum_l94_9454


namespace NUMINAMATH_CALUDE_power_sum_equality_l94_9493

theorem power_sum_equality : 2^300 + (-2^301) = -2^300 := by sorry

end NUMINAMATH_CALUDE_power_sum_equality_l94_9493


namespace NUMINAMATH_CALUDE_certain_number_proof_l94_9484

theorem certain_number_proof (x : ℝ) (n : ℝ) : 
  x = 6 → 9 - n / x = 7 + 8 / x → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l94_9484


namespace NUMINAMATH_CALUDE_least_positive_integer_divisible_by_four_primes_l94_9433

theorem least_positive_integer_divisible_by_four_primes : ∃ n : ℕ, 
  (∃ p₁ p₂ p₃ p₄ : ℕ, Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄ ∧ 
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₃ ≠ p₄ ∧
    n % p₁ = 0 ∧ n % p₂ = 0 ∧ n % p₃ = 0 ∧ n % p₄ = 0) ∧
  (∀ m : ℕ, m < n → 
    ¬(∃ q₁ q₂ q₃ q₄ : ℕ, Prime q₁ ∧ Prime q₂ ∧ Prime q₃ ∧ Prime q₄ ∧ 
      q₁ ≠ q₂ ∧ q₁ ≠ q₃ ∧ q₁ ≠ q₄ ∧ q₂ ≠ q₃ ∧ q₂ ≠ q₄ ∧ q₃ ≠ q₄ ∧
      m % q₁ = 0 ∧ m % q₂ = 0 ∧ m % q₃ = 0 ∧ m % q₄ = 0)) ∧
  n = 210 :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_divisible_by_four_primes_l94_9433


namespace NUMINAMATH_CALUDE_inscribed_cube_volume_l94_9496

/-- A pyramid with a square base and right-angled isosceles triangular lateral faces -/
structure Pyramid :=
  (base_side : ℝ)

/-- A cube inscribed in a pyramid -/
structure InscribedCube :=
  (side_length : ℝ)

/-- The volume of a cube -/
def cube_volume (c : InscribedCube) : ℝ := c.side_length ^ 3

/-- Predicate for a cube being properly inscribed in the pyramid -/
def is_properly_inscribed (p : Pyramid) (c : InscribedCube) : Prop :=
  c.side_length > 0 ∧ c.side_length < p.base_side

theorem inscribed_cube_volume (p : Pyramid) (c : InscribedCube) 
  (h1 : p.base_side = 2)
  (h2 : is_properly_inscribed p c) :
  cube_volume c = 10 + 6 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_inscribed_cube_volume_l94_9496


namespace NUMINAMATH_CALUDE_cube_volume_problem_l94_9422

theorem cube_volume_problem (V₁ : ℝ) (A₂ : ℝ) : 
  V₁ = 8 → 
  A₂ = 3 * (6 * (V₁^(1/3))^2) → 
  (A₂ / 6)^(3/2) = 24 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_cube_volume_problem_l94_9422


namespace NUMINAMATH_CALUDE_fourth_side_length_l94_9436

/-- A quadrilateral inscribed in a circle -/
structure InscribedQuadrilateral where
  /-- The radius of the circumscribed circle -/
  radius : ℝ
  /-- The lengths of the four sides of the quadrilateral -/
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ

/-- The theorem statement -/
theorem fourth_side_length
  (q : InscribedQuadrilateral)
  (h1 : q.radius = 150)
  (h2 : q.side1 = 200)
  (h3 : q.side2 = 200)
  (h4 : q.side3 = 100) :
  q.side4 = 300 := by
  sorry


end NUMINAMATH_CALUDE_fourth_side_length_l94_9436


namespace NUMINAMATH_CALUDE_min_value_of_w_l94_9415

theorem min_value_of_w :
  ∀ (x y z w : ℝ),
    -2 ≤ x ∧ x ≤ 5 →
    -3 ≤ y ∧ y ≤ 7 →
    4 ≤ z ∧ z ≤ 8 →
    w = x * y - z →
    w ≥ -23 ∧ ∃ (x₀ y₀ z₀ : ℝ),
      -2 ≤ x₀ ∧ x₀ ≤ 5 ∧
      -3 ≤ y₀ ∧ y₀ ≤ 7 ∧
      4 ≤ z₀ ∧ z₀ ≤ 8 ∧
      x₀ * y₀ - z₀ = -23 :=
by sorry


end NUMINAMATH_CALUDE_min_value_of_w_l94_9415


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l94_9424

/-- The speed of a boat in still water, given that:
    1. It takes 90 minutes less to travel 36 miles downstream than upstream.
    2. The speed of the stream is 2 mph. -/
theorem boat_speed_in_still_water : ∃ (b : ℝ),
  (36 / (b - 2) - 36 / (b + 2) = 1.5) ∧ b = 10 := by sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l94_9424
