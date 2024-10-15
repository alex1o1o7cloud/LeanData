import Mathlib

namespace NUMINAMATH_CALUDE_hannah_strawberries_l3624_362411

theorem hannah_strawberries (daily_harvest : ℕ) (days : ℕ) (stolen : ℕ) (remaining : ℕ) :
  daily_harvest = 5 →
  days = 30 →
  stolen = 30 →
  remaining = 100 →
  daily_harvest * days - stolen - remaining = 20 :=
by sorry

end NUMINAMATH_CALUDE_hannah_strawberries_l3624_362411


namespace NUMINAMATH_CALUDE_chinese_chess_probability_l3624_362412

/-- The probability of player A winning a game of Chinese chess -/
def prob_A_win : ℝ := 0.2

/-- The probability of a draw between players A and B -/
def prob_draw : ℝ := 0.5

/-- The probability of player B winning a game of Chinese chess -/
def prob_B_win : ℝ := 1 - (prob_A_win + prob_draw)

theorem chinese_chess_probability :
  prob_B_win = 0.3 := by sorry

end NUMINAMATH_CALUDE_chinese_chess_probability_l3624_362412


namespace NUMINAMATH_CALUDE_cows_sold_l3624_362401

/-- The number of cows sold by a man last year, given the following conditions:
  * He initially had 39 cows
  * 25 cows died last year
  * The number of cows increased by 24 this year
  * He bought 43 more cows
  * His friend gave him 8 cows as a gift
  * He now has 83 cows -/
theorem cows_sold (initial : ℕ) (died : ℕ) (increased : ℕ) (bought : ℕ) (gifted : ℕ) (current : ℕ)
  (h_initial : initial = 39)
  (h_died : died = 25)
  (h_increased : increased = 24)
  (h_bought : bought = 43)
  (h_gifted : gifted = 8)
  (h_current : current = 83)
  (h_equation : current = initial - died - (initial - died - increased - bought - gifted)) :
  initial - died - increased - bought - gifted = 6 := by
  sorry

end NUMINAMATH_CALUDE_cows_sold_l3624_362401


namespace NUMINAMATH_CALUDE_not_product_of_consecutive_numbers_l3624_362415

theorem not_product_of_consecutive_numbers (n k : ℕ) :
  ¬ ∃ x : ℕ, x * (x + 1) = 2 * n^(3*k) + 4 * n^k + 10 := by
  sorry

end NUMINAMATH_CALUDE_not_product_of_consecutive_numbers_l3624_362415


namespace NUMINAMATH_CALUDE_race_parts_length_l3624_362494

/-- Given a race with 4 parts, prove that the length of each of the second and third parts is 21.5 km -/
theorem race_parts_length 
  (total_length : ℝ) 
  (first_part : ℝ) 
  (last_part : ℝ) 
  (h1 : total_length = 74.5)
  (h2 : first_part = 15.5)
  (h3 : last_part = 16)
  (h4 : ∃ (second_part third_part : ℝ), 
        second_part = third_part ∧ 
        total_length = first_part + second_part + third_part + last_part) :
  ∃ (second_part : ℝ), second_part = 21.5 ∧ 
    total_length = first_part + second_part + second_part + last_part :=
by
  sorry

end NUMINAMATH_CALUDE_race_parts_length_l3624_362494


namespace NUMINAMATH_CALUDE_rationalize_denominator_l3624_362423

/-- Proves that the rationalization of 1/(√5 + √7 + √11) is equal to (-√5 - √7 + √11 + 2√385)/139 -/
theorem rationalize_denominator :
  ∃ (A B C D E F : ℚ),
    (1 : ℝ) / (Real.sqrt 5 + Real.sqrt 7 + Real.sqrt 11) =
    (A * Real.sqrt 5 + B * Real.sqrt 7 + C * Real.sqrt 11 + D * Real.sqrt E) / F ∧
    A = -1 ∧
    B = -1 ∧
    C = 1 ∧
    D = 2 ∧
    E = 385 ∧
    F = 139 ∧
    F > 0 :=
by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l3624_362423


namespace NUMINAMATH_CALUDE_electricity_gasoline_ratio_l3624_362444

theorem electricity_gasoline_ratio (total : ℕ) (both : ℕ) (gas_only : ℕ) (neither : ℕ)
  (h_total : total = 300)
  (h_both : both = 120)
  (h_gas_only : gas_only = 60)
  (h_neither : neither = 24)
  (h_sum : total = both + gas_only + (total - both - gas_only - neither) + neither) :
  (total - both - gas_only - neither) / neither = 4 := by
sorry

end NUMINAMATH_CALUDE_electricity_gasoline_ratio_l3624_362444


namespace NUMINAMATH_CALUDE_parabola_intercept_minimum_l3624_362426

/-- Parabola defined by x^2 = 8y -/
def Parabola (x y : ℝ) : Prop := x^2 = 8*y

/-- Line with slope k passing through point (x, y) -/
def Line (k x y : ℝ) : Prop := y = k*x + 2

/-- Length of line segment intercepted by the parabola for a line with slope k -/
def InterceptLength (k : ℝ) : ℝ := 8*k^2 + 8

/-- The condition given in the problem relating k1 and k2 -/
def SlopeCondition (k1 k2 : ℝ) : Prop := 1/k1^2 + 4/k2^2 = 1

theorem parabola_intercept_minimum :
  ∀ k1 k2 : ℝ, 
  SlopeCondition k1 k2 →
  InterceptLength k1 + InterceptLength k2 ≥ 88 :=
sorry

end NUMINAMATH_CALUDE_parabola_intercept_minimum_l3624_362426


namespace NUMINAMATH_CALUDE_kho_kho_players_l3624_362481

theorem kho_kho_players (total : ℕ) (kabadi : ℕ) (both : ℕ) (kho_kho_only : ℕ) :
  total = 50 →
  kabadi = 10 →
  both = 5 →
  total = (kabadi - both) + kho_kho_only + both →
  kho_kho_only = 40 := by
sorry

end NUMINAMATH_CALUDE_kho_kho_players_l3624_362481


namespace NUMINAMATH_CALUDE_distance_probability_l3624_362414

-- Define the points and distances
def A : ℝ × ℝ := (0, -10)
def B : ℝ × ℝ := (0, 0)
def AB : ℝ := 10
def BC : ℝ := 6
def AC_max : ℝ := 8

-- Define the angle range
def angle_range : Set ℝ := Set.Ioo 0 Real.pi

-- Define the probability function
noncomputable def probability_AC_less_than_8 : ℝ :=
  (30 : ℝ) / 180

-- State the theorem
theorem distance_probability :
  probability_AC_less_than_8 = 1/6 := by sorry

end NUMINAMATH_CALUDE_distance_probability_l3624_362414


namespace NUMINAMATH_CALUDE_six_digit_nondecreasing_remainder_l3624_362432

theorem six_digit_nondecreasing_remainder (n : Nat) (k : Nat) : 
  n = 6 → k = 9 → (Nat.choose (n + k - 1) n) % 1000 = 3 := by
  sorry

end NUMINAMATH_CALUDE_six_digit_nondecreasing_remainder_l3624_362432


namespace NUMINAMATH_CALUDE_g_geq_one_l3624_362474

noncomputable section

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x^2 + Real.log x + 4

-- Define the function g
def g (x : ℝ) : ℝ := Real.exp (x - 1) + 3 * x^2 + 4 - f x

-- Theorem statement
theorem g_geq_one (x : ℝ) (h : x > 0) : g x ≥ 1 := by
  sorry

end

end NUMINAMATH_CALUDE_g_geq_one_l3624_362474


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_2_l3624_362467

-- Define the displacement function
def h (t : ℝ) : ℝ := 14 * t - t^2

-- Define the instantaneous velocity function (derivative of h)
def v (t : ℝ) : ℝ := 14 - 2 * t

-- Theorem statement
theorem instantaneous_velocity_at_2 : v 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_2_l3624_362467


namespace NUMINAMATH_CALUDE_sample_data_properties_l3624_362462

def median (s : Finset ℝ) : ℝ := sorry

theorem sample_data_properties (x₁ x₂ x₃ x₄ x₅ x₆ : ℝ) 
  (h : x₁ ≤ x₂ ∧ x₂ ≤ x₃ ∧ x₃ ≤ x₄ ∧ x₄ ≤ x₅ ∧ x₅ ≤ x₆) :
  let s₁ := {x₂, x₃, x₄, x₅}
  let s₂ := {x₁, x₂, x₃, x₄, x₅, x₆}
  (median s₁ = median s₂) ∧ 
  (x₅ - x₂ ≤ x₆ - x₁) := by
  sorry

end NUMINAMATH_CALUDE_sample_data_properties_l3624_362462


namespace NUMINAMATH_CALUDE_f_is_quadratic_l3624_362457

/-- Definition of a quadratic equation in one variable -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function representing the equation x^2 - 2 = x -/
def f (x : ℝ) : ℝ := x^2 - x - 2

/-- Theorem stating that f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry

end NUMINAMATH_CALUDE_f_is_quadratic_l3624_362457


namespace NUMINAMATH_CALUDE_banyan_tree_area_l3624_362447

theorem banyan_tree_area (C : Real) (h : C = 6.28) :
  let r := C / (2 * Real.pi)
  let S := Real.pi * r^2
  S = Real.pi := by
sorry

end NUMINAMATH_CALUDE_banyan_tree_area_l3624_362447


namespace NUMINAMATH_CALUDE_vector_equation_solution_l3624_362458

/-- Given vectors e₁ and e₂, and real numbers x and y satisfying the equation,
    prove that x - y = -3 -/
theorem vector_equation_solution (e₁ e₂ : ℝ × ℝ) (x y : ℝ) 
    (h₁ : e₁ = (1, 2))
    (h₂ : e₂ = (3, 4))
    (h₃ : x • e₁ + y • e₂ = (5, 6)) :
  x - y = -3 := by
  sorry

end NUMINAMATH_CALUDE_vector_equation_solution_l3624_362458


namespace NUMINAMATH_CALUDE_students_passed_l3624_362410

theorem students_passed (total : ℕ) (failure_rate : ℚ) : 
  total = 1000 → failure_rate = 0.4 → (total : ℚ) * (1 - failure_rate) = 600 := by
  sorry

end NUMINAMATH_CALUDE_students_passed_l3624_362410


namespace NUMINAMATH_CALUDE_two_possible_values_l3624_362437

def triangle (a b : ℕ) : ℕ := min a b

def nabla (a b : ℕ) : ℕ := max a b

theorem two_possible_values (x : ℕ) : 
  ∃ (s : Finset ℕ), (s.card = 2) ∧ 
  (triangle 6 (nabla 4 (triangle x 5)) ∈ s) := by
  sorry

end NUMINAMATH_CALUDE_two_possible_values_l3624_362437


namespace NUMINAMATH_CALUDE_find_b_value_l3624_362495

-- Define the inverse relationship between a^3 and √b
def inverse_relation (a b : ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ a^3 * Real.sqrt b = k

-- State the theorem
theorem find_b_value (a₁ b₁ a₂ b₂ : ℝ) 
  (h₁ : inverse_relation a₁ b₁)
  (h₂ : a₁ = 3 ∧ b₁ = 64)
  (h₃ : inverse_relation a₂ b₂)
  (h₄ : a₂ = 4)
  (h₅ : a₂ * Real.sqrt b₂ = 24) :
  b₂ = 11.390625 := by
sorry

end NUMINAMATH_CALUDE_find_b_value_l3624_362495


namespace NUMINAMATH_CALUDE_number_operations_l3624_362446

theorem number_operations (x : ℝ) : ((x - 2 + 3) * 2) / 3 = 6 ↔ x = 8 := by
  sorry

end NUMINAMATH_CALUDE_number_operations_l3624_362446


namespace NUMINAMATH_CALUDE_art_museum_exhibits_l3624_362425

/-- The number of exhibits in an art museum --/
def num_exhibits : ℕ := 4

/-- The number of pictures the museum currently has --/
def current_pictures : ℕ := 15

/-- The number of additional pictures needed for equal distribution --/
def additional_pictures : ℕ := 1

theorem art_museum_exhibits :
  (current_pictures + additional_pictures) % num_exhibits = 0 ∧
  current_pictures % num_exhibits ≠ 0 ∧
  num_exhibits > 1 :=
sorry

end NUMINAMATH_CALUDE_art_museum_exhibits_l3624_362425


namespace NUMINAMATH_CALUDE_impossibility_of_broken_line_l3624_362472

/-- Represents a segment in the figure -/
structure Segment where
  id : Nat

/-- Represents a region in the figure -/
structure Region where
  segments : Finset Segment

/-- Represents the entire figure -/
structure Figure where
  segments : Finset Segment
  regions : Finset Region

/-- A broken line (polygonal chain) -/
structure BrokenLine where
  intersections : Finset Segment

/-- The theorem statement -/
theorem impossibility_of_broken_line (fig : Figure) 
  (h1 : fig.segments.card = 16)
  (h2 : ∃ r1 r2 r3 : Region, r1 ∈ fig.regions ∧ r2 ∈ fig.regions ∧ r3 ∈ fig.regions ∧ 
        r1.segments.card = 5 ∧ r2.segments.card = 5 ∧ r3.segments.card = 5) :
  ¬∃ (bl : BrokenLine), bl.intersections = fig.segments :=
sorry

end NUMINAMATH_CALUDE_impossibility_of_broken_line_l3624_362472


namespace NUMINAMATH_CALUDE_range_of_k_line_equation_when_OB_2OA_l3624_362453

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 6)^2 + y^2 = 20

-- Define the line l
def line_l (k x y : ℝ) : Prop := y = k * x

-- Define the condition that line l intersects circle C at two distinct points
def intersects_at_two_points (k : ℝ) : Prop := 
  ∃ (x₁ y₁ x₂ y₂ : ℝ), x₁ ≠ x₂ ∧ circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧ line_l k x₁ y₁ ∧ line_l k x₂ y₂

-- Define the condition OB = 2OA
def OB_equals_2OA (k : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧ line_l k x₁ y₁ ∧ line_l k x₂ y₂ ∧
    x₂ = 2 * x₁ ∧ y₂ = 2 * y₁

-- Theorem for the range of k
theorem range_of_k (k : ℝ) : intersects_at_two_points k → -Real.sqrt 5 / 2 < k ∧ k < Real.sqrt 5 / 2 := 
  sorry

-- Theorem for the equation of line l when OB = 2OA
theorem line_equation_when_OB_2OA (k : ℝ) : OB_equals_2OA k → k = 1 ∨ k = -1 :=
  sorry

end NUMINAMATH_CALUDE_range_of_k_line_equation_when_OB_2OA_l3624_362453


namespace NUMINAMATH_CALUDE_fraction_equivalence_l3624_362497

theorem fraction_equivalence : (10 : ℝ) / (8 * 60) = 0.1 / (0.8 * 60) := by sorry

end NUMINAMATH_CALUDE_fraction_equivalence_l3624_362497


namespace NUMINAMATH_CALUDE_no_such_function_exists_l3624_362491

theorem no_such_function_exists :
  ¬∃ (f : ℕ+ → ℕ+), ∀ (n : ℕ+), f (f n) = n + 1987 := by
  sorry

end NUMINAMATH_CALUDE_no_such_function_exists_l3624_362491


namespace NUMINAMATH_CALUDE_max_candy_leftover_l3624_362409

theorem max_candy_leftover (x : ℕ) : ∃ (q r : ℕ), x = 11 * q + r ∧ r ≤ 10 ∧ ∀ (r' : ℕ), x = 11 * q + r' → r' ≤ r := by
  sorry

end NUMINAMATH_CALUDE_max_candy_leftover_l3624_362409


namespace NUMINAMATH_CALUDE_system_solution_l3624_362456

theorem system_solution (x y : ℝ) 
  (eq1 : x^2 - 4 * Real.sqrt (3*x - 2) + 6 = y)
  (eq2 : y^2 - 4 * Real.sqrt (3*y - 2) + 6 = x)
  (domain_x : 3*x - 2 ≥ 0)
  (domain_y : 3*y - 2 ≥ 0) :
  x = 2 ∧ y = 2 :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l3624_362456


namespace NUMINAMATH_CALUDE_set_A_enumeration_l3624_362434

def A : Set (ℤ × ℤ) := {p | p.1^2 = p.2 + 1 ∧ |p.1| < 2}

theorem set_A_enumeration : A = {(-1, 0), (0, -1), (1, 0)} := by sorry

end NUMINAMATH_CALUDE_set_A_enumeration_l3624_362434


namespace NUMINAMATH_CALUDE_domain_of_sqrt_2cos_plus_1_l3624_362461

open Real

theorem domain_of_sqrt_2cos_plus_1 (x : ℝ) (k : ℤ) :
  (∃ y : ℝ, y = Real.sqrt (2 * Real.cos x + 1)) ↔ 
  (x ∈ Set.Icc (2 * Real.pi * k - 2 * Real.pi / 3) (2 * Real.pi * k + 2 * Real.pi / 3)) :=
sorry

end NUMINAMATH_CALUDE_domain_of_sqrt_2cos_plus_1_l3624_362461


namespace NUMINAMATH_CALUDE_x_fourth_plus_reciprocal_l3624_362420

theorem x_fourth_plus_reciprocal (x : ℝ) (h : x + 1/x = 8) : x^4 + 1/x^4 = 3842 := by
  sorry

end NUMINAMATH_CALUDE_x_fourth_plus_reciprocal_l3624_362420


namespace NUMINAMATH_CALUDE_unique_negative_zero_l3624_362438

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 3 * x^2 + 1

-- State the theorem
theorem unique_negative_zero (a : ℝ) :
  (∃! x₀ : ℝ, f a x₀ = 0 ∧ x₀ < 0) ↔ a > 3/2 := by sorry

end NUMINAMATH_CALUDE_unique_negative_zero_l3624_362438


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3624_362469

theorem quadratic_equation_solution (m : ℝ) : 
  (∃ x : ℝ, 2 * x^2 + 5 * x - m = 0) → m ≥ -25/8 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3624_362469


namespace NUMINAMATH_CALUDE_total_carriages_l3624_362470

/-- The number of carriages in each town -/
structure TownCarriages where
  euston : ℕ
  norfolk : ℕ
  norwich : ℕ
  flying_scotsman : ℕ

/-- The conditions given in the problem -/
def problem_conditions (tc : TownCarriages) : Prop :=
  tc.euston = tc.norfolk + 20 ∧
  tc.norwich = 100 ∧
  tc.flying_scotsman = tc.norwich + 20 ∧
  tc.euston = 130

/-- The theorem stating that the total number of carriages is 460 -/
theorem total_carriages (tc : TownCarriages) 
  (h : problem_conditions tc) : 
  tc.euston + tc.norfolk + tc.norwich + tc.flying_scotsman = 460 := by
  sorry

end NUMINAMATH_CALUDE_total_carriages_l3624_362470


namespace NUMINAMATH_CALUDE_cost_price_of_ball_l3624_362441

theorem cost_price_of_ball (selling_price : ℕ) (num_balls_sold : ℕ) (num_balls_loss : ℕ) 
  (h1 : selling_price = 720)
  (h2 : num_balls_sold = 17)
  (h3 : num_balls_loss = 5) : 
  ∃ (cost_price : ℕ), 
    cost_price * num_balls_sold - cost_price * num_balls_loss = selling_price ∧ 
    cost_price = 60 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_of_ball_l3624_362441


namespace NUMINAMATH_CALUDE_stratified_sampling_most_appropriate_l3624_362431

/-- Represents the different sampling methods --/
inductive SamplingMethod
  | Lottery
  | Systematic
  | Stratified
  | RandomNumber

/-- Represents a grade level --/
inductive Grade
  | Third
  | Sixth
  | Ninth

/-- Represents the characteristics of the sampling problem --/
structure SamplingProblem where
  grades : List Grade
  proportionalSampling : Bool
  distinctGroups : Bool

/-- Determines the most appropriate sampling method for a given problem --/
def mostAppropriateMethod (problem : SamplingProblem) : SamplingMethod :=
  if problem.distinctGroups && problem.proportionalSampling then
    SamplingMethod.Stratified
  else
    SamplingMethod.Lottery  -- Default to Lottery for simplicity

/-- The specific problem described in the question --/
def schoolEyesightProblem : SamplingProblem :=
  { grades := [Grade.Third, Grade.Sixth, Grade.Ninth]
  , proportionalSampling := true
  , distinctGroups := true }

theorem stratified_sampling_most_appropriate :
  mostAppropriateMethod schoolEyesightProblem = SamplingMethod.Stratified := by
  sorry


end NUMINAMATH_CALUDE_stratified_sampling_most_appropriate_l3624_362431


namespace NUMINAMATH_CALUDE_function_value_at_2017_l3624_362490

/-- A function satisfying the given conditions -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ a b : ℝ, 3 * f ((a + 2 * b) / 3) = f a + 2 * f b) ∧
  f 1 = 1 ∧
  f 4 = 7

/-- The main theorem -/
theorem function_value_at_2017 (f : ℝ → ℝ) (h : special_function f) : f 2017 = 4033 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_2017_l3624_362490


namespace NUMINAMATH_CALUDE_f_zero_values_l3624_362498

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f x * f y

/-- Theorem stating that f(0) is either 0 or 1 for functions satisfying the functional equation -/
theorem f_zero_values (f : ℝ → ℝ) (h : FunctionalEquation f) :
  f 0 = 0 ∨ f 0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_f_zero_values_l3624_362498


namespace NUMINAMATH_CALUDE_min_sum_squares_l3624_362416

theorem min_sum_squares (x y z : ℝ) (h : 2*x + y + 2*z = 6) :
  ∃ (m : ℝ), m = 4 ∧ ∀ (a b c : ℝ), 2*a + b + 2*c = 6 → a^2 + b^2 + c^2 ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_sum_squares_l3624_362416


namespace NUMINAMATH_CALUDE_shaded_area_is_120_l3624_362483

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle with four vertices -/
structure Rectangle where
  p : Point
  r : Point
  s : Point
  v : Point

/-- Calculates the area of a rectangle -/
def rectangleArea (rect : Rectangle) : ℝ :=
  (rect.r.x - rect.p.x) * (rect.v.y - rect.p.y)

/-- Theorem: The shaded area in the given rectangle is 120 cm² -/
theorem shaded_area_is_120 (rect : Rectangle) 
  (h1 : rect.r.x - rect.p.x = 20) -- PR = 20 cm
  (h2 : rect.v.y - rect.p.y = 12) -- PV = 12 cm
  (u : Point) (t : Point) (q : Point)
  (h3 : u.x = rect.v.x ∧ u.y ≤ rect.v.y ∧ u.y ≥ rect.s.y) -- U is on VS
  (h4 : t.x = rect.v.x ∧ t.y ≤ rect.v.y ∧ t.y ≥ rect.s.y) -- T is on VS
  (h5 : q.y = rect.p.y ∧ q.x ≥ rect.p.x ∧ q.x ≤ rect.r.x) -- Q is on PR
  : rectangleArea rect - (rect.r.x - rect.p.x) * (rect.v.y - rect.p.y) / 2 = 120 :=
sorry

end NUMINAMATH_CALUDE_shaded_area_is_120_l3624_362483


namespace NUMINAMATH_CALUDE_sum_of_squares_of_coefficients_l3624_362407

def polynomial (x : ℝ) : ℝ := 4 * (x^4 + 3*x^2 + 1)

theorem sum_of_squares_of_coefficients :
  (4^2) + (12^2) + (4^2) = 176 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_coefficients_l3624_362407


namespace NUMINAMATH_CALUDE_num_triangles_on_circle_l3624_362476

/-- The number of ways to choose k elements from n elements --/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of distinct points on the circle --/
def num_points : ℕ := 10

/-- The number of points needed to form a triangle --/
def points_per_triangle : ℕ := 3

/-- Theorem: The number of different triangles that can be formed
    by choosing 3 points from 10 distinct points on a circle's circumference
    is equal to 120 --/
theorem num_triangles_on_circle :
  choose num_points points_per_triangle = 120 := by sorry

end NUMINAMATH_CALUDE_num_triangles_on_circle_l3624_362476


namespace NUMINAMATH_CALUDE_chore_division_proof_l3624_362406

/-- Time to sweep one room in minutes -/
def sweep_time_per_room : ℕ := 3

/-- Time to wash one dish in minutes -/
def wash_dish_time : ℕ := 2

/-- Number of rooms Anna sweeps -/
def anna_rooms : ℕ := 10

/-- Number of laundry loads Billy does -/
def billy_laundry_loads : ℕ := 2

/-- Number of dishes Billy washes -/
def billy_dishes : ℕ := 6

/-- Time to do one load of laundry in minutes -/
def laundry_time : ℕ := 9

theorem chore_division_proof :
  anna_rooms * sweep_time_per_room = 
  billy_laundry_loads * laundry_time + billy_dishes * wash_dish_time :=
by sorry

end NUMINAMATH_CALUDE_chore_division_proof_l3624_362406


namespace NUMINAMATH_CALUDE_proposition_implication_l3624_362468

theorem proposition_implication (p q : Prop) 
  (h1 : ¬(p ∧ q)) 
  (h2 : ¬p) : 
  (¬p ∧ q) ∨ (¬p ∧ ¬q) :=
sorry

end NUMINAMATH_CALUDE_proposition_implication_l3624_362468


namespace NUMINAMATH_CALUDE_f_2_equals_5_l3624_362440

def f (x : ℝ) : ℝ := x^3 - 2*x^2 + 3*x - 1

theorem f_2_equals_5 : f 2 = 5 := by sorry

end NUMINAMATH_CALUDE_f_2_equals_5_l3624_362440


namespace NUMINAMATH_CALUDE_cylinder_height_l3624_362421

/-- A cylinder with base diameter equal to height and volume 16π has height 4 -/
theorem cylinder_height (r h : ℝ) (h_positive : 0 < h) (r_positive : 0 < r) : 
  h = 2 * r → π * r^2 * h = 16 * π → h = 4 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_height_l3624_362421


namespace NUMINAMATH_CALUDE_sequence_general_term_formula_l3624_362496

/-- Given a quadratic equation with real roots, prove the general term formula for a sequence defined by a recurrence relation. -/
theorem sequence_general_term_formula 
  (p q : ℝ) 
  (hq : q ≠ 0) 
  (α β : ℝ) 
  (hroots : α^2 - p*α + q = 0 ∧ β^2 - p*β + q = 0) 
  (a : ℕ → ℝ) 
  (ha1 : a 1 = p) 
  (ha2 : a 2 = p^2 - q) 
  (han : ∀ n : ℕ, n ≥ 3 → a n = p * a (n-1) - q * a (n-2)) :
  ∀ n : ℕ, n ≥ 1 → a n = (α^(n+1) - β^(n+1)) / (α - β) :=
sorry

end NUMINAMATH_CALUDE_sequence_general_term_formula_l3624_362496


namespace NUMINAMATH_CALUDE_number_problem_l3624_362404

theorem number_problem : ∃ x : ℝ, x + 3 * x = 20 ∧ x = 5 := by sorry

end NUMINAMATH_CALUDE_number_problem_l3624_362404


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l3624_362452

theorem complex_number_quadrant : ∃ (z : ℂ), z = Complex.I * (2 - Complex.I) ∧ Complex.re z > 0 ∧ Complex.im z > 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l3624_362452


namespace NUMINAMATH_CALUDE_sin_cos_difference_equals_neg_sqrt3_half_l3624_362448

theorem sin_cos_difference_equals_neg_sqrt3_half :
  Real.sin (5 * π / 180) * Real.sin (25 * π / 180) - 
  Real.sin (95 * π / 180) * Real.sin (65 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_difference_equals_neg_sqrt3_half_l3624_362448


namespace NUMINAMATH_CALUDE_sum_58_46_rounded_to_hundred_l3624_362464

def round_to_nearest_hundred (n : ℤ) : ℤ :=
  (n + 50) / 100 * 100

theorem sum_58_46_rounded_to_hundred : 
  round_to_nearest_hundred (58 + 46) = 100 := by
  sorry

end NUMINAMATH_CALUDE_sum_58_46_rounded_to_hundred_l3624_362464


namespace NUMINAMATH_CALUDE_unique_modular_inverse_in_range_l3624_362459

theorem unique_modular_inverse_in_range (p : Nat) (a : Nat) 
  (h_prime : Nat.Prime p) 
  (h_odd : Odd p)
  (h_a_range : 2 ≤ a ∧ a ≤ p - 2) :
  ∃! i : Nat, 2 ≤ i ∧ i ≤ p - 2 ∧ 
    (i * a) % p = 1 ∧ 
    Nat.gcd i a = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_modular_inverse_in_range_l3624_362459


namespace NUMINAMATH_CALUDE_equation_solution_l3624_362439

theorem equation_solution (x : ℝ) : 
  Real.sqrt (x + 15) - 9 / Real.sqrt (x + 15) = 3 → x = 18 * Real.sqrt 5 / 4 - 6 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3624_362439


namespace NUMINAMATH_CALUDE_system_solution_l3624_362430

theorem system_solution (x y : ℝ) (k n : ℤ) : 
  (2 * (Real.cos x)^2 - 2 * Real.sqrt 2 * Real.cos x * (Real.cos (8 * x))^2 + (Real.cos (8 * x))^2 = 0 ∧
   Real.sin x = Real.cos y) ↔ 
  ((x = π/4 + 2*π*↑k ∧ (y = π/4 + 2*π*↑n ∨ y = -π/4 + 2*π*↑n)) ∨
   (x = -π/4 + 2*π*↑k ∧ (y = 3*π/4 + 2*π*↑n ∨ y = -3*π/4 + 2*π*↑n))) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l3624_362430


namespace NUMINAMATH_CALUDE_unique_line_through_point_l3624_362487

/-- A line in the xy-plane --/
structure Line where
  x_intercept : ℕ+
  y_intercept : ℕ+

/-- Checks if a natural number is prime --/
def isPrime (n : ℕ+) : Prop :=
  n > 1 ∧ ∀ m : ℕ+, m < n → m ∣ n → m = 1

/-- Checks if a line passes through the point (5,4) --/
def passesThrough (l : Line) : Prop :=
  5 / l.x_intercept.val + 4 / l.y_intercept.val = 1

/-- The main theorem --/
theorem unique_line_through_point :
  ∃! l : Line, passesThrough l ∧ isPrime l.y_intercept :=
sorry

end NUMINAMATH_CALUDE_unique_line_through_point_l3624_362487


namespace NUMINAMATH_CALUDE_total_tulips_is_308_l3624_362477

/-- The number of tulips needed for Anna's smiley face design --/
def total_tulips : ℕ :=
  let red_eye := 8
  let purple_eyebrow := 5
  let red_nose := 12
  let red_smile := 18
  let yellow_background := 9 * red_smile
  let purple_eyebrows := 4 * (2 * red_eye)
  let yellow_nose := 3 * red_nose
  
  let total_red := 2 * red_eye + red_nose + red_smile
  let total_purple := 2 * purple_eyebrow + (purple_eyebrows - 2 * purple_eyebrow)
  let total_yellow := yellow_background + yellow_nose
  
  total_red + total_purple + total_yellow

theorem total_tulips_is_308 : total_tulips = 308 := by
  sorry

end NUMINAMATH_CALUDE_total_tulips_is_308_l3624_362477


namespace NUMINAMATH_CALUDE_distance_between_trees_l3624_362418

def yard_length : ℝ := 300
def num_trees : ℕ := 26

theorem distance_between_trees :
  let num_intervals : ℕ := num_trees - 1
  let distance : ℝ := yard_length / num_intervals
  distance = 12 := by sorry

end NUMINAMATH_CALUDE_distance_between_trees_l3624_362418


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_l3624_362480

/-- 
For a quadratic equation kx² + 2x - 1 = 0 to have two distinct real roots,
k must satisfy k > -1 and k ≠ 0.
-/
theorem quadratic_distinct_roots (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ k * x₁^2 + 2 * x₁ - 1 = 0 ∧ k * x₂^2 + 2 * x₂ - 1 = 0) ↔ 
  (k > -1 ∧ k ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_l3624_362480


namespace NUMINAMATH_CALUDE_arrangements_count_is_2880_l3624_362443

/-- The number of arrangements of 4 students and 3 teachers in a row,
    where exactly two teachers are standing next to each other. -/
def arrangements_count : ℕ :=
  let num_students : ℕ := 4
  let num_teachers : ℕ := 3
  let num_units : ℕ := num_students + 1  -- 4 students + 1 teacher pair
  let teacher_pair_permutations : ℕ := 2  -- 2! ways to arrange 2 teachers in a pair
  let remaining_teacher_positions : ℕ := num_students + 1  -- positions for the remaining teacher
  let teacher_pair_combinations : ℕ := 3  -- number of ways to choose 2 teachers out of 3
  (Nat.factorial num_units) * teacher_pair_permutations * remaining_teacher_positions * teacher_pair_combinations

/-- Theorem stating that the number of arrangements is 2880 -/
theorem arrangements_count_is_2880 : arrangements_count = 2880 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_count_is_2880_l3624_362443


namespace NUMINAMATH_CALUDE_min_c_value_l3624_362449

theorem min_c_value (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (hab : a < b) (hbc : b < c) 
  (h_unique : ∃! (x y : ℝ), 2 * x + y = 3005 ∧ y = |x - a| + |x - b| + |x - c|) :
  c ≥ 3003 ∧ ∃ (a' b' : ℕ), 0 < a' ∧ 0 < b' ∧ a' < b' ∧ b' < 3003 ∧
    ∃! (x y : ℝ), 2 * x + y = 3005 ∧ y = |x - a'| + |x - b'| + |x - 3003| :=
by sorry

end NUMINAMATH_CALUDE_min_c_value_l3624_362449


namespace NUMINAMATH_CALUDE_min_cos_B_angle_A_values_l3624_362413

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.a + t.c = 6 * Real.sqrt 3 ∧ t.b = 6

-- Theorem for the minimum value of cos B
theorem min_cos_B (t : Triangle) (h : triangle_conditions t) :
  ∃ (min_cos_B : ℝ), min_cos_B = 1/3 ∧ ∀ (cos_B : ℝ), cos_B = (t.a^2 + t.c^2 - t.b^2) / (2 * t.a * t.c) → cos_B ≥ min_cos_B :=
sorry

-- Theorem for the possible values of angle A
theorem angle_A_values (t : Triangle) (h1 : triangle_conditions t) (h2 : t.a * t.b * Real.cos t.C = 12) :
  t.A = π/2 ∨ t.A = π/6 :=
sorry

end NUMINAMATH_CALUDE_min_cos_B_angle_A_values_l3624_362413


namespace NUMINAMATH_CALUDE_negative_integer_problem_l3624_362473

theorem negative_integer_problem (n : ℤ) : 
  n < 0 → n * (-8) + 5 = 93 → n = -11 := by
  sorry

end NUMINAMATH_CALUDE_negative_integer_problem_l3624_362473


namespace NUMINAMATH_CALUDE_binary_sum_equals_136_l3624_362400

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

def binary1 : List Bool := [true, false, true, false, true, false, true]
def binary2 : List Bool := [true, true, false, false, true, true]

theorem binary_sum_equals_136 :
  binary_to_decimal binary1 + binary_to_decimal binary2 = 136 := by
  sorry

end NUMINAMATH_CALUDE_binary_sum_equals_136_l3624_362400


namespace NUMINAMATH_CALUDE_garden_comparison_l3624_362463

-- Define the dimensions of the gardens
def chris_length : ℝ := 30
def chris_width : ℝ := 60
def jordan_length : ℝ := 35
def jordan_width : ℝ := 55

-- Define the area difference
def area_difference : ℝ := jordan_length * jordan_width - chris_length * chris_width

-- Define the perimeters
def chris_perimeter : ℝ := 2 * (chris_length + chris_width)
def jordan_perimeter : ℝ := 2 * (jordan_length + jordan_width)

-- Theorem statement
theorem garden_comparison :
  area_difference = 125 ∧ chris_perimeter = jordan_perimeter := by
  sorry

end NUMINAMATH_CALUDE_garden_comparison_l3624_362463


namespace NUMINAMATH_CALUDE_f_max_min_on_interval_l3624_362466

open Real

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.cos x - x

theorem f_max_min_on_interval :
  let a := 0
  let b := 2 * Real.pi / 3
  ∃ (max min : ℝ),
    (∀ x ∈ Set.Icc a b, f x ≤ max) ∧
    (∃ x ∈ Set.Icc a b, f x = max) ∧
    (∀ x ∈ Set.Icc a b, min ≤ f x) ∧
    (∃ x ∈ Set.Icc a b, f x = min) ∧
    max = 1 ∧
    min = -(1/2) * Real.exp (2 * Real.pi / 3) - 2 * Real.pi / 3 :=
by sorry

end NUMINAMATH_CALUDE_f_max_min_on_interval_l3624_362466


namespace NUMINAMATH_CALUDE_income_mean_difference_l3624_362479

theorem income_mean_difference (T : ℝ) (n : ℕ) : 
  n = 500 → 
  (T + 1100000) / n - (T + 110000) / n = 1980 :=
by sorry

end NUMINAMATH_CALUDE_income_mean_difference_l3624_362479


namespace NUMINAMATH_CALUDE_nth_letter_is_c_l3624_362499

def repeating_pattern : ℕ → Char
  | n => match n % 3 with
    | 0 => 'C'
    | 1 => 'A'
    | _ => 'B'

theorem nth_letter_is_c (n : ℕ) (h : n = 150) : repeating_pattern n = 'C' := by
  sorry

end NUMINAMATH_CALUDE_nth_letter_is_c_l3624_362499


namespace NUMINAMATH_CALUDE_complex_sum_to_polar_l3624_362492

theorem complex_sum_to_polar : 15 * Complex.exp (Complex.I * Real.pi / 6) + 15 * Complex.exp (Complex.I * 5 * Real.pi / 6) = 15 * Complex.exp (Complex.I * Real.pi / 2) := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_to_polar_l3624_362492


namespace NUMINAMATH_CALUDE_horner_first_step_value_l3624_362428

/-- Horner's Method first step for polynomial evaluation -/
def horner_first_step (a : ℕ → ℝ) (x : ℝ) : ℝ :=
  a 4 * x + a 3

/-- Polynomial coefficients -/
def f_coeff : ℕ → ℝ
  | 4 => 3
  | 3 => 0
  | 2 => 2
  | 1 => 1
  | 0 => 4
  | _ => 0

theorem horner_first_step_value :
  horner_first_step f_coeff 10 = 30 := by sorry

end NUMINAMATH_CALUDE_horner_first_step_value_l3624_362428


namespace NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l3624_362402

theorem sum_of_solutions_quadratic (x : ℝ) : 
  (63 - 21*x - x^2 = 0) → 
  (∃ r s : ℝ, (63 - 21*r - r^2 = 0) ∧ (63 - 21*s - s^2 = 0) ∧ (r + s = -21)) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l3624_362402


namespace NUMINAMATH_CALUDE_collinear_points_k_value_l3624_362442

-- Define the vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define basis vectors
variable (e₁ e₂ : V)

-- Define points and vectors
variable (A B C D : V)
variable (k : ℝ)

-- State the theorem
theorem collinear_points_k_value
  (h_basis : LinearIndependent ℝ ![e₁, e₂])
  (h_AB : B - A = e₁ - k • e₂)
  (h_CB : B - C = 2 • e₁ - e₂)
  (h_CD : D - C = 3 • e₁ - 3 • e₂)
  (h_collinear : ∃ (t : ℝ), D - A = t • (B - A)) :
  k = 2 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_k_value_l3624_362442


namespace NUMINAMATH_CALUDE_determine_english_marks_l3624_362435

/-- Represents a student's marks in 5 subjects -/
structure StudentMarks where
  english : ℕ
  mathematics : ℕ
  physics : ℕ
  chemistry : ℕ
  biology : ℕ

/-- Calculate the average marks -/
def average (marks : StudentMarks) : ℚ :=
  (marks.english + marks.mathematics + marks.physics + marks.chemistry + marks.biology) / 5

/-- Theorem: Given 4 subject marks and the average, the 5th subject mark is uniquely determined -/
theorem determine_english_marks (marks : StudentMarks) (avg : ℚ) 
    (h1 : marks.mathematics = 65)
    (h2 : marks.physics = 82)
    (h3 : marks.chemistry = 67)
    (h4 : marks.biology = 85)
    (h5 : average marks = avg)
    (h6 : avg = 75) :
  marks.english = 76 := by
  sorry


end NUMINAMATH_CALUDE_determine_english_marks_l3624_362435


namespace NUMINAMATH_CALUDE_square_difference_equals_400_l3624_362429

theorem square_difference_equals_400 : (25 + 8)^2 - (8^2 + 25^2) = 400 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equals_400_l3624_362429


namespace NUMINAMATH_CALUDE_andrew_bought_62_eggs_l3624_362482

/-- Represents the number of eggs Andrew has at different points -/
structure EggCount where
  initial : Nat
  final : Nat

/-- Calculates the number of eggs bought -/
def eggsBought (e : EggCount) : Nat :=
  e.final - e.initial

/-- Theorem stating that Andrew bought 62 eggs -/
theorem andrew_bought_62_eggs :
  let e : EggCount := { initial := 8, final := 70 }
  eggsBought e = 62 := by
  sorry

end NUMINAMATH_CALUDE_andrew_bought_62_eggs_l3624_362482


namespace NUMINAMATH_CALUDE_correct_quadratic_equation_l3624_362488

theorem correct_quadratic_equation 
  (a b c : ℝ) 
  (h1 : ∃ c', (a * 7^2 + b * 7 + c' = 0) ∧ (a * 3^2 + b * 3 + c' = 0))
  (h2 : ∃ b', (a * (-12)^2 + b' * (-12) + c = 0) ∧ (a * 3^2 + b' * 3 + c = 0)) :
  a = 1 ∧ b = -10 ∧ c = -36 := by
sorry

end NUMINAMATH_CALUDE_correct_quadratic_equation_l3624_362488


namespace NUMINAMATH_CALUDE_factor_expression_l3624_362478

theorem factor_expression (b : ℝ) : 280 * b^2 + 56 * b = 56 * b * (5 * b + 1) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l3624_362478


namespace NUMINAMATH_CALUDE_birthday_cake_icing_l3624_362485

/-- Represents a rectangular cake with given dimensions -/
structure Cake where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents a smaller cuboid piece of the cake -/
structure Piece where
  size : ℕ

/-- Calculates the number of pieces with icing on exactly two sides -/
def pieces_with_two_sided_icing (c : Cake) (p : Piece) : ℕ :=
  sorry

/-- Theorem stating that a 6 × 4 × 4 cake cut into 2 × 2 × 2 pieces has 16 pieces with icing on two sides -/
theorem birthday_cake_icing (c : Cake) (p : Piece) :
  c.length = 6 ∧ c.width = 4 ∧ c.height = 4 ∧ p.size = 2 →
  pieces_with_two_sided_icing c p = 16 :=
by sorry

end NUMINAMATH_CALUDE_birthday_cake_icing_l3624_362485


namespace NUMINAMATH_CALUDE_sqrt_two_plus_sqrt_three_root_l3624_362403

theorem sqrt_two_plus_sqrt_three_root : ∃ x : ℝ, x = Real.sqrt 2 + Real.sqrt 3 ∧ x^4 - 10*x^2 + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_plus_sqrt_three_root_l3624_362403


namespace NUMINAMATH_CALUDE_broken_lines_count_l3624_362465

/-- The number of paths on a grid with 2n steps, n horizontal and n vertical -/
def grid_paths (n : ℕ) : ℕ := (Nat.choose (2 * n) n) ^ 2

/-- Theorem stating that the number of broken lines of length 2n on a grid
    with cell side length 1 and vertices at intersections is (C_{2n}^{n})^2 -/
theorem broken_lines_count (n : ℕ) :
  grid_paths n = (Nat.choose (2 * n) n) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_broken_lines_count_l3624_362465


namespace NUMINAMATH_CALUDE_fractional_equation_solution_range_l3624_362451

theorem fractional_equation_solution_range (m : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ x ≠ 2 ∧ (x + m) / (x - 2) + (2 * m) / (2 - x) = 3) → 
  m < 6 ∧ m ≠ 2 := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_range_l3624_362451


namespace NUMINAMATH_CALUDE_second_largest_of_five_consecutive_odds_l3624_362445

theorem second_largest_of_five_consecutive_odds (a b c d e : ℕ) : 
  (∀ n : ℕ, n ∈ [a, b, c, d, e] → n % 2 = 1) →  -- all numbers are odd
  (b = a + 2 ∧ c = b + 2 ∧ d = c + 2 ∧ e = d + 2) →  -- consecutive
  a + b + c + d + e = 195 →  -- sum is 195
  d = 41 :=  -- 2nd largest (4th in sequence) is 41
by
  sorry

end NUMINAMATH_CALUDE_second_largest_of_five_consecutive_odds_l3624_362445


namespace NUMINAMATH_CALUDE_total_students_l3624_362471

theorem total_students (passed_first : ℕ) (passed_second : ℕ) (passed_both : ℕ) (failed_both : ℕ) 
  (h1 : passed_first = 60)
  (h2 : passed_second = 40)
  (h3 : passed_both = 20)
  (h4 : failed_both = 20) :
  passed_first + passed_second - passed_both + failed_both = 100 := by
  sorry

#check total_students

end NUMINAMATH_CALUDE_total_students_l3624_362471


namespace NUMINAMATH_CALUDE_intersection_line_equation_l3624_362408

/-- Represents a circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The equation of the line passing through the intersection points of two circles --/
def intersectionLine (c1 c2 : Circle) : ℝ → ℝ → Prop :=
  fun x y => x + y = -143/117

/-- The main theorem statement --/
theorem intersection_line_equation (c1 c2 : Circle) 
  (h1 : c1 = ⟨(-5, -6), 10⟩) 
  (h2 : c2 = ⟨(4, 7), Real.sqrt 85⟩) : 
  ∀ x y, (x - c1.center.1)^2 + (y - c1.center.2)^2 = c1.radius^2 ∧ 
         (x - c2.center.1)^2 + (y - c2.center.2)^2 = c2.radius^2 → 
  intersectionLine c1 c2 x y := by
  sorry

end NUMINAMATH_CALUDE_intersection_line_equation_l3624_362408


namespace NUMINAMATH_CALUDE_quadratic_roots_inequality_l3624_362424

theorem quadratic_roots_inequality (a : ℝ) (x₁ x₂ : ℝ) : 
  x₁^2 - a*x₁ + a = 0 → x₂^2 - a*x₂ + a = 0 → x₁ ≠ x₂ → x₁^2 + x₂^2 ≥ 2*(x₁ + x₂) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_inequality_l3624_362424


namespace NUMINAMATH_CALUDE_new_men_average_age_l3624_362422

/-- Given a group of 8 men, when two men aged 21 and 23 are replaced by two new men,
    and the average age of the group increases by 2 years,
    prove that the average age of the two new men is 30 years. -/
theorem new_men_average_age
  (initial_count : Nat)
  (replaced_age1 replaced_age2 : Nat)
  (age_increase : ℝ)
  (h_count : initial_count = 8)
  (h_replaced1 : replaced_age1 = 21)
  (h_replaced2 : replaced_age2 = 23)
  (h_increase : age_increase = 2)
  : (↑initial_count * age_increase + ↑replaced_age1 + ↑replaced_age2) / 2 = 30 :=
by sorry

end NUMINAMATH_CALUDE_new_men_average_age_l3624_362422


namespace NUMINAMATH_CALUDE_line_parameterization_l3624_362450

/-- Given a line y = 2x - 10 parameterized by (x, y) = (g(t), 20t - 8), 
    prove that g(t) = 10t + 1 -/
theorem line_parameterization (g : ℝ → ℝ) : 
  (∀ t x y, y = 2*x - 10 ∧ x = g t ∧ y = 20*t - 8) → 
  (∀ t, g t = 10*t + 1) :=
by sorry

end NUMINAMATH_CALUDE_line_parameterization_l3624_362450


namespace NUMINAMATH_CALUDE_new_basis_from_old_l3624_362493

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

def is_basis (v₁ v₂ v₃ : V) : Prop :=
  LinearIndependent ℝ ![v₁, v₂, v₃] ∧ Submodule.span ℝ {v₁, v₂, v₃} = ⊤

theorem new_basis_from_old (a b c p q : V) 
  (h₁ : is_basis a b c)
  (h₂ : p = a + b)
  (h₃ : q = a - b) :
  is_basis p q (a + 2 • c) := by
  sorry

end NUMINAMATH_CALUDE_new_basis_from_old_l3624_362493


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l3624_362455

theorem smallest_n_congruence : ∃ n : ℕ+, (∀ m : ℕ+, 813 * m ≡ 1224 * m [ZMOD 30] → n ≤ m) ∧ 813 * n ≡ 1224 * n [ZMOD 30] := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l3624_362455


namespace NUMINAMATH_CALUDE_minimum_pages_required_l3624_362486

-- Define the types of cards and pages
inductive CardType
| Rare
| LimitedEdition
| Regular

inductive PageType
| NineCard
| SevenCard
| FiveCard

-- Define the card counts
def rareCardCount : Nat := 18
def limitedEditionCardCount : Nat := 21
def regularCardCount : Nat := 45

-- Define the page capacities
def pageCapacity (pt : PageType) : Nat :=
  match pt with
  | PageType.NineCard => 9
  | PageType.SevenCard => 7
  | PageType.FiveCard => 5

-- Define a function to check if a page type is valid for a card type
def isValidPageType (ct : CardType) (pt : PageType) : Bool :=
  match ct, pt with
  | CardType.Rare, PageType.NineCard => true
  | CardType.Rare, PageType.SevenCard => true
  | CardType.LimitedEdition, PageType.NineCard => true
  | CardType.LimitedEdition, PageType.SevenCard => true
  | CardType.Regular, _ => true
  | _, _ => false

-- Define the theorem
theorem minimum_pages_required :
  ∃ (rarePages limitedPages regularPages : Nat),
    rarePages * pageCapacity PageType.NineCard = rareCardCount ∧
    limitedPages * pageCapacity PageType.SevenCard = limitedEditionCardCount ∧
    regularPages * pageCapacity PageType.NineCard = regularCardCount ∧
    rarePages + limitedPages + regularPages = 10 ∧
    (∀ (rp lp regalp : Nat),
      rp * pageCapacity PageType.NineCard ≥ rareCardCount →
      lp * pageCapacity PageType.SevenCard ≥ limitedEditionCardCount →
      regalp * pageCapacity PageType.NineCard ≥ regularCardCount →
      isValidPageType CardType.Rare PageType.NineCard →
      isValidPageType CardType.LimitedEdition PageType.SevenCard →
      isValidPageType CardType.Regular PageType.NineCard →
      rp + lp + regalp ≥ 10) :=
by sorry

end NUMINAMATH_CALUDE_minimum_pages_required_l3624_362486


namespace NUMINAMATH_CALUDE_product_evaluation_l3624_362419

theorem product_evaluation (n : ℕ) (h : n = 2) :
  (n - 1) * n * (n + 1) * (n + 2) * (n + 3) = 120 := by
  sorry

end NUMINAMATH_CALUDE_product_evaluation_l3624_362419


namespace NUMINAMATH_CALUDE_annas_lemonade_sales_l3624_362405

/-- Anna's lemonade sales problem -/
theorem annas_lemonade_sales 
  (plain_glasses : ℕ) 
  (plain_price : ℚ) 
  (plain_strawberry_difference : ℚ) 
  (h1 : plain_glasses = 36)
  (h2 : plain_price = 3/4)
  (h3 : plain_strawberry_difference = 11) :
  (plain_glasses : ℚ) * plain_price - plain_strawberry_difference = 16 := by
  sorry


end NUMINAMATH_CALUDE_annas_lemonade_sales_l3624_362405


namespace NUMINAMATH_CALUDE_classroom_pictures_l3624_362475

/-- The number of oil paintings on the walls of the classroom -/
def oil_paintings : ℕ := 9

/-- The number of watercolor paintings on the walls of the classroom -/
def watercolor_paintings : ℕ := 7

/-- The total number of pictures on the walls of the classroom -/
def total_pictures : ℕ := oil_paintings + watercolor_paintings

theorem classroom_pictures : total_pictures = 16 := by
  sorry

end NUMINAMATH_CALUDE_classroom_pictures_l3624_362475


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3624_362433

theorem imaginary_part_of_complex_fraction : Complex.im ((2 + 4 * Complex.I) / (1 + Complex.I)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3624_362433


namespace NUMINAMATH_CALUDE_distance_between_vertices_l3624_362427

-- Define the two quadratic functions
def f (x : ℝ) : ℝ := x^2 - 4*x + 5
def g (x : ℝ) : ℝ := x^2 + 6*x + 13

-- Define the vertices of the two parabolas
def vertex_f : ℝ × ℝ := (2, f 2)
def vertex_g : ℝ × ℝ := (-3, g (-3))

-- State the theorem
theorem distance_between_vertices : 
  Real.sqrt ((vertex_f.1 - vertex_g.1)^2 + (vertex_f.2 - vertex_g.2)^2) = Real.sqrt 34 :=
by sorry

end NUMINAMATH_CALUDE_distance_between_vertices_l3624_362427


namespace NUMINAMATH_CALUDE_mushroom_collection_l3624_362417

theorem mushroom_collection (total_mushrooms : ℕ) (h1 : total_mushrooms = 289) :
  ∃ (num_children : ℕ) (mushrooms_per_child : ℕ),
    num_children > 0 ∧
    mushrooms_per_child > 0 ∧
    num_children * mushrooms_per_child = total_mushrooms ∧
    num_children = 17 := by
  sorry

end NUMINAMATH_CALUDE_mushroom_collection_l3624_362417


namespace NUMINAMATH_CALUDE_factorization_difference_of_squares_factorization_quadratic_l3624_362489

-- Problem 1
theorem factorization_difference_of_squares (x y : ℝ) :
  x^2 - 4*y^2 = (x + 2*y) * (x - 2*y) := by sorry

-- Problem 2
theorem factorization_quadratic (a x : ℝ) :
  3*a*x^2 - 6*a*x + 3*a = 3*a*(x - 1)^2 := by sorry

end NUMINAMATH_CALUDE_factorization_difference_of_squares_factorization_quadratic_l3624_362489


namespace NUMINAMATH_CALUDE_square_plus_one_representation_l3624_362454

theorem square_plus_one_representation (x y z : ℕ+) (h : x * y = z^2 + 1) :
  ∃ (a b c d : ℤ), x = a^2 + b^2 ∧ y = c^2 + d^2 ∧ z = a * c + b * d :=
sorry

end NUMINAMATH_CALUDE_square_plus_one_representation_l3624_362454


namespace NUMINAMATH_CALUDE_sum_of_digits_5N_plus_2013_l3624_362436

/-- Sum of digits function in base 10 -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- The smallest positive integer with sum of digits 2013 -/
def N : ℕ := sorry

/-- Theorem stating the sum of digits of (5N + 2013) is 18 -/
theorem sum_of_digits_5N_plus_2013 :
  sum_of_digits (5 * N + 2013) = 18 ∧ 
  sum_of_digits N = 2013 ∧
  ∀ m : ℕ, m < N → sum_of_digits m ≠ 2013 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_5N_plus_2013_l3624_362436


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l3624_362460

open Set

-- Define sets A and B
def A : Set ℝ := {x | 1 < x ∧ x < 4}
def B : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = Icc (-1) 4 := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l3624_362460


namespace NUMINAMATH_CALUDE_apple_boxes_count_l3624_362484

def apples_per_crate : ℕ := 250
def number_of_crates : ℕ := 20
def rotten_apples : ℕ := 320
def apples_per_box : ℕ := 25

theorem apple_boxes_count :
  (apples_per_crate * number_of_crates - rotten_apples) / apples_per_box = 187 := by
  sorry

end NUMINAMATH_CALUDE_apple_boxes_count_l3624_362484
