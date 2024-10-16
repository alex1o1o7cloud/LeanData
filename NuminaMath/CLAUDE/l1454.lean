import Mathlib

namespace NUMINAMATH_CALUDE_compound_interest_rate_l1454_145440

/-- Compound interest calculation -/
theorem compound_interest_rate (P : ℝ) (t : ℝ) (I : ℝ) (r : ℝ) : 
  P = 6000 → t = 2 → I = 1260.000000000001 → 
  P * (1 + r)^t = P + I → r = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_compound_interest_rate_l1454_145440


namespace NUMINAMATH_CALUDE_original_integer_is_45_l1454_145499

theorem original_integer_is_45 (a b c d : ℤ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (eq1 : (b + c + d) / 3 + 10 = 37)
  (eq2 : (a + c + d) / 3 + 10 = 31)
  (eq3 : (a + b + d) / 3 + 10 = 25)
  (eq4 : (a + b + c) / 3 + 10 = 19) :
  a = 45 ∨ b = 45 ∨ c = 45 ∨ d = 45 :=
sorry

end NUMINAMATH_CALUDE_original_integer_is_45_l1454_145499


namespace NUMINAMATH_CALUDE_janet_dresses_pockets_l1454_145498

theorem janet_dresses_pockets :
  -- Total number of dresses
  ∀ total_dresses : ℕ,
  -- Number of dresses with pockets
  ∀ dresses_with_pockets : ℕ,
  -- Number of dresses with 2 pockets
  ∀ dresses_with_two_pockets : ℕ,
  -- Total number of pockets
  ∀ total_pockets : ℕ,
  -- Conditions
  total_dresses = 24 →
  dresses_with_pockets = total_dresses / 2 →
  dresses_with_two_pockets = dresses_with_pockets / 3 →
  total_pockets = 32 →
  -- Conclusion
  (total_pockets - 2 * dresses_with_two_pockets) / (dresses_with_pockets - dresses_with_two_pockets) = 3 :=
by sorry

end NUMINAMATH_CALUDE_janet_dresses_pockets_l1454_145498


namespace NUMINAMATH_CALUDE_aluminum_ball_radius_l1454_145418

theorem aluminum_ball_radius (small_radius : ℝ) (num_small_balls : ℕ) (large_radius : ℝ) :
  small_radius = 0.5 →
  num_small_balls = 12 →
  (4 / 3) * π * large_radius^3 = num_small_balls * ((4 / 3) * π * small_radius^3) →
  large_radius = (3 / 2)^(1 / 3) :=
by sorry

end NUMINAMATH_CALUDE_aluminum_ball_radius_l1454_145418


namespace NUMINAMATH_CALUDE_probability_joined_1890_to_1969_l1454_145422

def total_provinces : ℕ := 13
def joined_1890_to_1969 : ℕ := 4

theorem probability_joined_1890_to_1969 :
  (joined_1890_to_1969 : ℚ) / total_provinces = 4 / 13 := by
  sorry

end NUMINAMATH_CALUDE_probability_joined_1890_to_1969_l1454_145422


namespace NUMINAMATH_CALUDE_trigonometric_expression_equality_trigonometric_fraction_simplification_l1454_145454

-- Part 1
theorem trigonometric_expression_equality : 
  2 * Real.cos (π / 2) + Real.tan (π / 4) + 3 * Real.sin 0 + (Real.cos (π / 3))^2 + Real.sin (3 * π / 2) = 1 / 4 := by
  sorry

-- Part 2
theorem trigonometric_fraction_simplification (θ : ℝ) : 
  (Real.sin (2 * π - θ) * Real.cos (π + θ) * Real.cos (π / 2 + θ) * Real.cos (11 * π / 2 - θ)) /
  (Real.cos (π - θ) * Real.sin (3 * π - θ) * Real.sin (-π - θ) * Real.sin (9 * π / 2 + θ)) = -Real.tan θ := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_equality_trigonometric_fraction_simplification_l1454_145454


namespace NUMINAMATH_CALUDE_subtraction_of_decimals_l1454_145408

theorem subtraction_of_decimals : 3.57 - 1.14 - 0.23 = 2.20 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_of_decimals_l1454_145408


namespace NUMINAMATH_CALUDE_cubic_function_properties_l1454_145489

-- Define the function f
noncomputable def f (a b c x : ℝ) : ℝ := -x^3 + a*x^2 + b*x + c

-- Define the derivative of f
noncomputable def f' (a b x : ℝ) : ℝ := -3*x^2 + 2*a*x + b

theorem cubic_function_properties (a b c : ℝ) :
  (f' a b (-1) = 0 ∧ f' a b 3 = 0) →
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f a b c x = 20 ∧ ∀ y ∈ Set.Icc (-2 : ℝ) 2, f a b c y ≤ 20) →
  (a = 3 ∧ b = 9) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f a b c x = 13 ∧ ∀ y ∈ Set.Icc (-2 : ℝ) 2, f a b c y ≥ 13) ∨
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f a b c x = -7 ∧ ∀ y ∈ Set.Icc (-2 : ℝ) 2, f a b c y ≥ -7) :=
by sorry

end NUMINAMATH_CALUDE_cubic_function_properties_l1454_145489


namespace NUMINAMATH_CALUDE_cubic_fraction_inequality_l1454_145480

theorem cubic_fraction_inequality (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a^3 + b^3 + c^3) / (a + b + c) +
  (b^3 + c^3 + d^3) / (b + c + d) +
  (c^3 + d^3 + a^3) / (c + d + a) +
  (d^3 + a^3 + b^3) / (d + a + b) ≥
  a^2 + b^2 + c^2 + d^2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_fraction_inequality_l1454_145480


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l1454_145441

def M : Set ℝ := {x | x^2 + 2*x = 0}
def N : Set ℝ := {x | x^2 - 2*x = 0}

theorem union_of_M_and_N : M ∪ N = {-2, 0, 2} := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l1454_145441


namespace NUMINAMATH_CALUDE_calculation_proof_l1454_145459

theorem calculation_proof :
  ((-36) * (-(7/12) - 3/4 + 5/6) = 18) ∧
  (-3^2 / 4 * |-(4/3)| * 6 + (-2)^3 = -26) := by
sorry

end NUMINAMATH_CALUDE_calculation_proof_l1454_145459


namespace NUMINAMATH_CALUDE_winter_solstice_shadow_length_l1454_145488

/-- Given an arithmetic sequence of 12 terms, if the sum of the 1st, 4th, and 7th terms is 37.5
    and the 12th term is 4.5, then the 1st term is 15.5. -/
theorem winter_solstice_shadow_length 
  (a : ℕ → ℝ) 
  (h_arithmetic : ∀ n, a (n + 1) - a n = a 1 - a 0) 
  (h_sum : a 0 + a 3 + a 6 = 37.5) 
  (h_last : a 11 = 4.5) : 
  a 0 = 15.5 := by
sorry

end NUMINAMATH_CALUDE_winter_solstice_shadow_length_l1454_145488


namespace NUMINAMATH_CALUDE_jason_remaining_cards_l1454_145453

/-- The number of Pokemon cards Jason started with -/
def initial_cards : ℕ := 13

/-- The number of Pokemon cards Jason gave away -/
def cards_given_away : ℕ := 9

/-- The number of Pokemon cards Jason has now -/
def remaining_cards : ℕ := initial_cards - cards_given_away

theorem jason_remaining_cards : remaining_cards = 4 := by
  sorry

end NUMINAMATH_CALUDE_jason_remaining_cards_l1454_145453


namespace NUMINAMATH_CALUDE_either_equal_or_irrational_l1454_145496

theorem either_equal_or_irrational (m : ℤ) (n : ℝ) 
  (h : m^2 + 1/n = n^2 + 1/m) : n = m ∨ ¬(∃ (p q : ℤ), n = p / q) :=
by sorry

end NUMINAMATH_CALUDE_either_equal_or_irrational_l1454_145496


namespace NUMINAMATH_CALUDE_trigonometric_simplification_logarithmic_simplification_l1454_145406

theorem trigonometric_simplification (θ : Real) (h : Real.tan θ = 2) :
  (Real.sin (θ + π/2) * Real.cos (π/2 - θ) - Real.cos (π - θ)^2) / (1 + Real.sin θ^2) = 1/3 := by
  sorry

theorem logarithmic_simplification (x : Real) :
  Real.log (Real.sqrt (x^2 + 1) + x) + Real.log (Real.sqrt (x^2 + 1) - x) +
  (Real.log 2 / Real.log 10)^2 + (1 + Real.log 2 / Real.log 10) * (Real.log 5 / Real.log 10) -
  2 * Real.sin (30 * π / 180) = 0 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_logarithmic_simplification_l1454_145406


namespace NUMINAMATH_CALUDE_perfect_cube_pair_solution_l1454_145401

theorem perfect_cube_pair_solution : ∀ a b : ℕ+,
  (∃ k : ℕ+, (a ^ 3 + 6 * a * b + 1 : ℕ) = k ^ 3) →
  (∃ m : ℕ+, (b ^ 3 + 6 * a * b + 1 : ℕ) = m ^ 3) →
  a = 1 ∧ b = 1 := by
sorry

end NUMINAMATH_CALUDE_perfect_cube_pair_solution_l1454_145401


namespace NUMINAMATH_CALUDE_order_of_expressions_l1454_145447

theorem order_of_expressions :
  let x := Real.exp (-1/2)
  let y := (Real.log 2) / (Real.log 5)
  let z := Real.log 3
  z > x ∧ x > y := by sorry

end NUMINAMATH_CALUDE_order_of_expressions_l1454_145447


namespace NUMINAMATH_CALUDE_intersection_complement_theorem_l1454_145421

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}

def B : Set ℝ := {x | ∃ y, y = Real.log (1 - x)}

theorem intersection_complement_theorem : A ∩ (U \ B) = {x | 1 ≤ x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_theorem_l1454_145421


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l1454_145405

theorem cubic_roots_sum (m : ℤ) (p q r : ℤ) : 
  (∀ x, x^3 - 2023*x + m = (x - p) * (x - q) * (x - r)) →
  |p| + |q| + |r| = 100 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l1454_145405


namespace NUMINAMATH_CALUDE_pirate_treasure_l1454_145461

theorem pirate_treasure (m : ℕ) : 
  (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m → m = 120 := by
  sorry

end NUMINAMATH_CALUDE_pirate_treasure_l1454_145461


namespace NUMINAMATH_CALUDE_min_sum_squares_l1454_145463

theorem min_sum_squares (y₁ y₂ y₃ : ℝ) 
  (pos₁ : 0 < y₁) (pos₂ : 0 < y₂) (pos₃ : 0 < y₃)
  (sum_eq : y₁ + 3 * y₂ + 5 * y₃ = 120) :
  y₁^2 + y₂^2 + y₃^2 ≥ 43200 / 361 ∧
  ∃ y₁' y₂' y₃' : ℝ, 
    0 < y₁' ∧ 0 < y₂' ∧ 0 < y₃' ∧
    y₁' + 3 * y₂' + 5 * y₃' = 120 ∧
    y₁'^2 + y₂'^2 + y₃'^2 = 43200 / 361 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_squares_l1454_145463


namespace NUMINAMATH_CALUDE_skew_lines_sufficient_not_necessary_l1454_145423

-- Define the concept of a line in 3D space
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

-- Define what it means for two lines to be skew
def are_skew (l1 l2 : Line3D) : Prop := sorry

-- Define what it means for two lines to have no common point
def no_common_point (l1 l2 : Line3D) : Prop := sorry

-- State the theorem
theorem skew_lines_sufficient_not_necessary :
  ∀ (l1 l2 : Line3D),
    (are_skew l1 l2 → no_common_point l1 l2) ∧
    ¬(no_common_point l1 l2 → are_skew l1 l2) :=
by sorry

end NUMINAMATH_CALUDE_skew_lines_sufficient_not_necessary_l1454_145423


namespace NUMINAMATH_CALUDE_smallest_sum_of_squares_l1454_145455

theorem smallest_sum_of_squares (x y : ℕ) : 
  x^2 - y^2 = 133 → 
  (∀ a b : ℕ, a^2 - b^2 = 133 → x^2 + y^2 ≤ a^2 + b^2) → 
  x^2 + y^2 = 205 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_sum_of_squares_l1454_145455


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l1454_145426

theorem geometric_sequence_problem (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- geometric sequence definition
  (q > 0) →                     -- positive sequence
  (a 3 = 2) →                   -- given condition
  (a 4 = 8 * a 7) →             -- given condition
  (a 9 = 1 / 32) :=              -- conclusion to prove
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l1454_145426


namespace NUMINAMATH_CALUDE_emir_savings_correct_l1454_145436

/-- The amount Emir has saved from his allowance -/
def emirSavings (dictionaryCost cookbookCost dinosaurBookCost additionalNeeded : ℕ) : ℕ :=
  dictionaryCost + cookbookCost + dinosaurBookCost - additionalNeeded

theorem emir_savings_correct (dictionaryCost cookbookCost dinosaurBookCost additionalNeeded : ℕ) :
  emirSavings dictionaryCost cookbookCost dinosaurBookCost additionalNeeded =
  dictionaryCost + cookbookCost + dinosaurBookCost - additionalNeeded :=
by sorry

end NUMINAMATH_CALUDE_emir_savings_correct_l1454_145436


namespace NUMINAMATH_CALUDE_initial_speed_is_850_l1454_145444

/-- Represents the airplane's journey with given conditions -/
structure AirplaneJourney where
  totalDistance : ℝ
  distanceBeforeLanding : ℝ
  landingDuration : ℝ
  speedReduction : ℝ
  totalTime : ℝ

/-- Calculates the initial speed of the airplane given the journey parameters -/
def initialSpeed (journey : AirplaneJourney) : ℝ :=
  -- Implementation details omitted
  sorry

/-- Theorem stating that the initial speed is 850 km/h for the given conditions -/
theorem initial_speed_is_850 :
  let journey : AirplaneJourney := {
    totalDistance := 2900
    distanceBeforeLanding := 1700
    landingDuration := 1.5
    speedReduction := 50
    totalTime := 5
  }
  initialSpeed journey = 850 := by
  sorry

end NUMINAMATH_CALUDE_initial_speed_is_850_l1454_145444


namespace NUMINAMATH_CALUDE_garys_final_amount_l1454_145412

/-- Given Gary's initial amount and the amount he received from selling his snake, 
    calculate his final amount. -/
theorem garys_final_amount 
  (initial_amount : ℝ) 
  (snake_sale_amount : ℝ) 
  (h1 : initial_amount = 73.0) 
  (h2 : snake_sale_amount = 55.0) : 
  initial_amount + snake_sale_amount = 128.0 := by
  sorry

end NUMINAMATH_CALUDE_garys_final_amount_l1454_145412


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1454_145428

theorem quadratic_inequality (x : ℝ) : x^2 - 36*x + 323 ≤ 3 ↔ 16 ≤ x ∧ x ≤ 20 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1454_145428


namespace NUMINAMATH_CALUDE_initial_overs_is_ten_l1454_145464

/-- Represents a cricket game scenario -/
structure CricketGame where
  targetScore : ℕ
  initialRunRate : ℚ
  remainingOvers : ℕ
  requiredRunRate : ℚ

/-- Calculates the number of overs played initially -/
def initialOvers (game : CricketGame) : ℚ :=
  (game.targetScore - game.requiredRunRate * game.remainingOvers) / (game.initialRunRate - game.requiredRunRate)

/-- Theorem stating that the number of overs played initially is 10 -/
theorem initial_overs_is_ten (game : CricketGame) 
  (h1 : game.targetScore = 282)
  (h2 : game.initialRunRate = 4.8)
  (h3 : game.remainingOvers = 40)
  (h4 : game.requiredRunRate = 5.85) :
  initialOvers game = 10 := by
  sorry

end NUMINAMATH_CALUDE_initial_overs_is_ten_l1454_145464


namespace NUMINAMATH_CALUDE_quadrilateral_equality_l1454_145474

/-- Given a quadrilateral ABCD where AD is parallel to BC, 
    prove that AC^2 + BD^2 = AB^2 + CD^2 + 2AD · BC. -/
theorem quadrilateral_equality (A B C D : ℝ × ℝ) 
    (h_parallel : (D.2 - A.2) / (D.1 - A.1) = (C.2 - B.2) / (C.1 - B.1)) : 
    (A.1 - C.1)^2 + (A.2 - C.2)^2 + (B.1 - D.1)^2 + (B.2 - D.2)^2 = 
    (A.1 - B.1)^2 + (A.2 - B.2)^2 + (C.1 - D.1)^2 + (C.2 - D.2)^2 + 
    2 * ((D.1 - A.1) * (C.1 - B.1) + (D.2 - A.2) * (C.2 - B.2)) := by
  sorry


end NUMINAMATH_CALUDE_quadrilateral_equality_l1454_145474


namespace NUMINAMATH_CALUDE_f_composition_at_two_l1454_145491

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then -Real.sqrt x
  else (x - 1/x)^4

theorem f_composition_at_two : f (f 2) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_at_two_l1454_145491


namespace NUMINAMATH_CALUDE_solution_set_l1454_145460

-- Define the function f
def f (a b x : ℝ) : ℝ := x^2 + (a - b)*x + 1

-- Define the property of f being even
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Define the domain of f
def domain (a : ℝ) : Set ℝ := Set.Icc (a - 1) (2*a + 4)

-- State the theorem
theorem solution_set 
  (a b : ℝ) 
  (h1 : is_even (f a b))
  (h2 : domain a = Set.Icc (-2) 2) :
  {x | f a b x > f a b b} = 
    (Set.Ioc (-2) (-1) ∪ Set.Ioc 1 2) :=
sorry

end NUMINAMATH_CALUDE_solution_set_l1454_145460


namespace NUMINAMATH_CALUDE_paving_cost_calculation_l1454_145490

-- Define the room dimensions
def room_length : ℝ := 10
def room_width : ℝ := 4.75

-- Define the paving rate
def paving_rate : ℝ := 900

-- Calculate the area of the room
def room_area : ℝ := room_length * room_width

-- Calculate the total cost of paving
def paving_cost : ℝ := room_area * paving_rate

-- Theorem to prove
theorem paving_cost_calculation : paving_cost = 42750 := by
  sorry

end NUMINAMATH_CALUDE_paving_cost_calculation_l1454_145490


namespace NUMINAMATH_CALUDE_largest_number_with_digits_4_1_sum_14_l1454_145494

def is_valid_number (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 4 ∨ d = 1

def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem largest_number_with_digits_4_1_sum_14 :
  ∀ n : ℕ, is_valid_number n ∧ sum_of_digits n = 14 → n ≤ 4411 :=
by sorry

end NUMINAMATH_CALUDE_largest_number_with_digits_4_1_sum_14_l1454_145494


namespace NUMINAMATH_CALUDE_parabola_and_bisector_intercept_l1454_145471

-- Define the line l
def line_l (x y : ℝ) : Prop := y = (1/2) * (x + 4)

-- Define the parabola G
def parabola_G (p : ℝ) (x y : ℝ) : Prop := x^2 = 2 * p * y

-- Define the intersection points B and C
def intersection_points (xB yB xC yC : ℝ) : Prop :=
  line_l xB yB ∧ line_l xC yC ∧ 
  parabola_G 2 xB yB ∧ parabola_G 2 xC yC

-- Define the midpoint of BC
def midpoint_BC (x y : ℝ) : Prop :=
  ∃ xB yB xC yC, intersection_points xB yB xC yC ∧
  x = (xB + xC) / 2 ∧ y = (yB + yC) / 2

-- Define the perpendicular bisector of BC
def perp_bisector (x y : ℝ) : Prop :=
  ∃ x0 y0, midpoint_BC x0 y0 ∧ y - y0 = -2 * (x - x0)

-- Theorem statement
theorem parabola_and_bisector_intercept :
  (∃ p : ℝ, p > 0 ∧ ∀ x y, parabola_G p x y ↔ x^2 = 4 * y) ∧
  (∃ b : ℝ, b = 9/2 ∧ perp_bisector 0 b) ∧
  (∃ x, x = 1 ∧ midpoint_BC x ((1/2) * (x + 4))) :=
sorry

end NUMINAMATH_CALUDE_parabola_and_bisector_intercept_l1454_145471


namespace NUMINAMATH_CALUDE_new_member_age_l1454_145427

theorem new_member_age (n : ℕ) (initial_avg : ℝ) (new_avg : ℝ) : 
  n = 10 → initial_avg = 15 → new_avg = 17 → 
  ∃ (new_member_age : ℝ), 
    (n * initial_avg + new_member_age) / (n + 1) = new_avg ∧ 
    new_member_age = 37 := by
  sorry

end NUMINAMATH_CALUDE_new_member_age_l1454_145427


namespace NUMINAMATH_CALUDE_line_equation_through_M_intersecting_C_l1454_145485

-- Define the curve C
def curve_C (x y : ℝ) : Prop :=
  ∃ θ : ℝ, x = -1 + 2 * Real.cos θ ∧ y = 1 + 2 * Real.sin θ

-- Define the point M
def point_M : ℝ × ℝ := (-1, 2)

-- Define a line passing through two points
def line_through_points (x₁ y₁ x₂ y₂ x y : ℝ) : Prop :=
  (y - y₁) * (x₂ - x₁) = (y₂ - y₁) * (x - x₁)

-- State the theorem
theorem line_equation_through_M_intersecting_C :
  ∀ A B : ℝ × ℝ,
  curve_C A.1 A.2 →
  curve_C B.1 B.2 →
  A ≠ B →
  line_through_points point_M.1 point_M.2 A.1 A.2 B.1 B.2 →
  point_M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
  ∃ x y : ℝ,
    (Real.sqrt 15 * x - 5 * y + Real.sqrt 15 + 10 = 0) ∨
    (Real.sqrt 15 * x + 5 * y + Real.sqrt 15 - 10 = 0) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_through_M_intersecting_C_l1454_145485


namespace NUMINAMATH_CALUDE_charity_box_distribution_l1454_145437

/-- The charity organization's box distribution problem -/
theorem charity_box_distribution
  (box_cost : ℕ)
  (donation_multiplier : ℕ)
  (total_boxes : ℕ)
  (h1 : box_cost = 245)
  (h2 : donation_multiplier = 4)
  (h3 : total_boxes = 2000) :
  ∃ (initial_boxes : ℕ),
    initial_boxes * box_cost * (1 + donation_multiplier) = total_boxes * box_cost ∧
    initial_boxes = 400 := by
  sorry

end NUMINAMATH_CALUDE_charity_box_distribution_l1454_145437


namespace NUMINAMATH_CALUDE_bowling_ball_weight_l1454_145479

/-- Given that five identical bowling balls weigh the same as four identical canoes,
    and two canoes weigh 80 pounds, prove that one bowling ball weighs 32 pounds. -/
theorem bowling_ball_weight :
  ∀ (bowling_ball_weight canoe_weight : ℕ),
    5 * bowling_ball_weight = 4 * canoe_weight →
    2 * canoe_weight = 80 →
    bowling_ball_weight = 32 := by
  sorry

end NUMINAMATH_CALUDE_bowling_ball_weight_l1454_145479


namespace NUMINAMATH_CALUDE_max_sum_of_square_roots_l1454_145448

theorem max_sum_of_square_roots (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (sum_eq_one : a + b + c = 1) : 
  Real.sqrt a + Real.sqrt b + Real.sqrt c ≤ Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_of_square_roots_l1454_145448


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l1454_145419

open Set

def M : Set ℝ := {x | (x + 2) * (x - 1) < 0}
def N : Set ℝ := {x | x + 1 < 0}

theorem intersection_of_M_and_N : M ∩ N = {x : ℝ | -2 < x ∧ x < -1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l1454_145419


namespace NUMINAMATH_CALUDE_sum_even_implies_diff_even_l1454_145493

theorem sum_even_implies_diff_even (a b : ℤ) : 
  Even (a + b) → Even (a - b) := by
  sorry

end NUMINAMATH_CALUDE_sum_even_implies_diff_even_l1454_145493


namespace NUMINAMATH_CALUDE_teresas_class_size_l1454_145445

theorem teresas_class_size :
  ∃! n : ℕ, 50 < n ∧ n < 100 ∧ n % 3 = 2 ∧ n % 4 = 2 ∧ n % 5 = 2 ∧ n = 62 := by
  sorry

end NUMINAMATH_CALUDE_teresas_class_size_l1454_145445


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1454_145456

theorem quadratic_equation_roots (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ k^2*x₁^2 - (2*k+1)*x₁ + 1 = 0 ∧ k^2*x₂^2 - (2*k+1)*x₂ + 1 = 0) ↔ 
  (k ≥ -1/4 ∧ k ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1454_145456


namespace NUMINAMATH_CALUDE_student_multiplication_problem_l1454_145465

theorem student_multiplication_problem (x : ℝ) : 30 * x - 138 = 102 → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_student_multiplication_problem_l1454_145465


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l1454_145420

theorem cone_lateral_surface_area (radius : ℝ) (slant_height : ℝ) :
  radius = 3 → slant_height = 5 → π * radius * slant_height = 15 * π := by
  sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l1454_145420


namespace NUMINAMATH_CALUDE_min_value_expression_l1454_145409

theorem min_value_expression (a b c : ℝ) 
  (ha : -0.5 < a ∧ a < 0.5) 
  (hb : -0.5 < b ∧ b < 0.5) 
  (hc : -0.5 < c ∧ c < 0.5) : 
  1 / ((1 - a) * (1 - b) * (1 - c)) + 1 / ((1 + a) * (1 + b) * (1 + c)) ≥ 4.74 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l1454_145409


namespace NUMINAMATH_CALUDE_product_of_decimals_l1454_145476

theorem product_of_decimals : (0.7 : ℝ) * 0.8 = 0.56 := by
  sorry

end NUMINAMATH_CALUDE_product_of_decimals_l1454_145476


namespace NUMINAMATH_CALUDE_cloth_cost_price_theorem_l1454_145402

/-- Calculates the cost price per meter of cloth given the total meters sold,
    the total selling price, and the profit per meter. -/
def costPricePerMeter (totalMeters : ℕ) (sellingPrice : ℕ) (profitPerMeter : ℕ) : ℕ :=
  (sellingPrice - profitPerMeter * totalMeters) / totalMeters

/-- Proves that given the specified conditions, the cost price per meter of cloth is 95 Rs. -/
theorem cloth_cost_price_theorem (totalMeters sellingPrice profitPerMeter : ℕ)
    (h1 : totalMeters = 85)
    (h2 : sellingPrice = 8925)
    (h3 : profitPerMeter = 10) :
    costPricePerMeter totalMeters sellingPrice profitPerMeter = 95 := by
  sorry

#eval costPricePerMeter 85 8925 10

end NUMINAMATH_CALUDE_cloth_cost_price_theorem_l1454_145402


namespace NUMINAMATH_CALUDE_condition_sufficiency_not_necessity_l1454_145443

theorem condition_sufficiency_not_necessity (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∀ a b : ℝ, a > 0 → b > 0 → a^2 + b^2 < 1 → a * b + 1 > a + b) ∧ 
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a * b + 1 > a + b ∧ a^2 + b^2 ≥ 1) := by
sorry

end NUMINAMATH_CALUDE_condition_sufficiency_not_necessity_l1454_145443


namespace NUMINAMATH_CALUDE_definite_integral_2x_l1454_145416

theorem definite_integral_2x : ∫ x in (0)..(π/2), 2*x = π^2/4 := by
  sorry

end NUMINAMATH_CALUDE_definite_integral_2x_l1454_145416


namespace NUMINAMATH_CALUDE_keychain_manufacturing_cost_l1454_145450

theorem keychain_manufacturing_cost 
  (P : ℝ) -- Selling price
  (initial_cost : ℝ) -- Initial manufacturing cost
  (initial_profit_percentage : ℝ) -- Initial profit percentage
  (new_profit_percentage : ℝ) -- New profit percentage
  (h1 : initial_cost = 65) -- Initial cost is $65
  (h2 : P - initial_cost = initial_profit_percentage * P) -- Initial profit equation
  (h3 : initial_profit_percentage = 0.35) -- Initial profit is 35%
  (h4 : new_profit_percentage = 0.50) -- New profit is 50%
  : ∃ C, P - C = new_profit_percentage * P ∧ C = 50 := by
sorry

end NUMINAMATH_CALUDE_keychain_manufacturing_cost_l1454_145450


namespace NUMINAMATH_CALUDE_movie_production_l1454_145446

theorem movie_production (x : ℝ) : 
  (∃ y : ℝ, y = 1.25 * x ∧ 5 * (x + y) = 2475) → x = 220 :=
by sorry

end NUMINAMATH_CALUDE_movie_production_l1454_145446


namespace NUMINAMATH_CALUDE_solve_video_game_problem_l1454_145434

def video_game_problem (total_games : ℕ) (potential_earnings : ℕ) (price_per_game : ℕ) : Prop :=
  let working_games := potential_earnings / price_per_game
  let non_working_games := total_games - working_games
  non_working_games = 8

theorem solve_video_game_problem :
  video_game_problem 16 56 7 :=
sorry

end NUMINAMATH_CALUDE_solve_video_game_problem_l1454_145434


namespace NUMINAMATH_CALUDE_train_speed_l1454_145487

-- Define the train length in meters
def train_length : ℝ := 180

-- Define the time to cross in seconds
def crossing_time : ℝ := 12

-- Define the conversion factor from m/s to km/h
def ms_to_kmh : ℝ := 3.6

-- Theorem to prove
theorem train_speed :
  (train_length / crossing_time) * ms_to_kmh = 54 := by
  sorry


end NUMINAMATH_CALUDE_train_speed_l1454_145487


namespace NUMINAMATH_CALUDE_stones_division_impossible_l1454_145407

theorem stones_division_impossible (stones : List Nat) : 
  stones.length = 31 ∧ stones.sum = 660 → 
  ∃ (a b : Nat), a ∈ stones ∧ b ∈ stones ∧ a > 2 * b :=
by sorry

end NUMINAMATH_CALUDE_stones_division_impossible_l1454_145407


namespace NUMINAMATH_CALUDE_square_area_ratio_l1454_145495

theorem square_area_ratio (s₁ s₂ : ℝ) (h : s₁ = 2 * s₂ * Real.sqrt 2) :
  s₁^2 / s₂^2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_l1454_145495


namespace NUMINAMATH_CALUDE_carltons_outfits_l1454_145415

/-- Represents a person's wardrobe and outfit combinations -/
structure Wardrobe where
  buttonUpShirts : ℕ
  sweaterVests : ℕ
  outfits : ℕ

/-- Calculates the number of outfits for a given wardrobe -/
def calculateOutfits (w : Wardrobe) : Prop :=
  w.sweaterVests = 2 * w.buttonUpShirts ∧
  w.outfits = w.sweaterVests * w.buttonUpShirts

/-- Theorem: Carlton's wardrobe has 18 outfits -/
theorem carltons_outfits :
  ∃ (w : Wardrobe), w.buttonUpShirts = 3 ∧ calculateOutfits w ∧ w.outfits = 18 := by
  sorry


end NUMINAMATH_CALUDE_carltons_outfits_l1454_145415


namespace NUMINAMATH_CALUDE_johns_computer_cost_l1454_145492

/-- The total cost of John's computer setup -/
def total_cost (computer_cost peripherals_cost original_video_card_cost upgraded_video_card_cost : ℝ) : ℝ :=
  computer_cost + peripherals_cost + (upgraded_video_card_cost - original_video_card_cost)

/-- Theorem stating the total cost of John's computer setup -/
theorem johns_computer_cost :
  let computer_cost : ℝ := 1500
  let peripherals_cost : ℝ := computer_cost / 5
  let original_video_card_cost : ℝ := 300
  let upgraded_video_card_cost : ℝ := 2 * original_video_card_cost
  total_cost computer_cost peripherals_cost original_video_card_cost upgraded_video_card_cost = 2100 :=
by
  sorry

end NUMINAMATH_CALUDE_johns_computer_cost_l1454_145492


namespace NUMINAMATH_CALUDE_second_larger_perfect_square_l1454_145439

theorem second_larger_perfect_square (x : ℕ) (h : ∃ k : ℕ, x = k ^ 2) :
  ∃ n : ℕ, n > x ∧ (∃ m : ℕ, n = m ^ 2) ∧
  (∀ y : ℕ, y > x ∧ (∃ l : ℕ, y = l ^ 2) → y ≥ n) ∧
  n = x + 4 * (x.sqrt) + 4 :=
sorry

end NUMINAMATH_CALUDE_second_larger_perfect_square_l1454_145439


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l1454_145417

theorem fraction_equation_solution :
  ∃! x : ℚ, (x + 5) / (x - 3) = (x - 2) / (x + 4) :=
by
  use -1
  constructor
  · -- Prove that x = -1 satisfies the equation
    sorry
  · -- Prove uniqueness
    sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l1454_145417


namespace NUMINAMATH_CALUDE_sector_max_area_l1454_145484

/-- Given a sector with perimeter 20 cm, prove that the area is maximized when the central angle is 2 radians and the maximum area is 25 cm². -/
theorem sector_max_area (r : ℝ) (θ : ℝ) :
  r > 0 →
  r * θ + 2 * r = 20 →
  0 < θ →
  θ ≤ 2 * π →
  (∀ r' θ', r' > 0 → r' * θ' + 2 * r' = 20 → 0 < θ' → θ' ≤ 2 * π → 
    1/2 * r * r * θ ≥ 1/2 * r' * r' * θ') →
  θ = 2 ∧ 1/2 * r * r * θ = 25 :=
sorry

end NUMINAMATH_CALUDE_sector_max_area_l1454_145484


namespace NUMINAMATH_CALUDE_choose_two_from_three_l1454_145470

theorem choose_two_from_three (n : ℕ) (h : n = 3) :
  Nat.choose n 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_choose_two_from_three_l1454_145470


namespace NUMINAMATH_CALUDE_no_real_solutions_l1454_145432

theorem no_real_solutions :
  ∀ x : ℝ, (5 * x^2 - 3 * x + 2) / (x + 2) ≠ 2 * x - 3 :=
by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l1454_145432


namespace NUMINAMATH_CALUDE_sweater_price_theorem_l1454_145425

/-- The marked price of a sweater in yuan -/
def marked_price : ℝ := 150

/-- The selling price as a percentage of the marked price -/
def selling_percentage : ℝ := 0.8

/-- The profit percentage -/
def profit_percentage : ℝ := 0.2

/-- The purchase price of the sweater in yuan -/
def purchase_price : ℝ := 100

theorem sweater_price_theorem : 
  selling_percentage * marked_price = purchase_price * (1 + profit_percentage) :=
sorry

end NUMINAMATH_CALUDE_sweater_price_theorem_l1454_145425


namespace NUMINAMATH_CALUDE_complex_roots_sum_of_absolute_values_l1454_145457

theorem complex_roots_sum_of_absolute_values (a : ℝ) (x₁ x₂ : ℂ) : 
  (∀ x : ℂ, x^2 - 2*a*x + a^2 - 4*a + 4 = 0 ↔ x = x₁ ∨ x = x₂) →
  Complex.abs x₁ + Complex.abs x₂ = 3 →
  a = 1/2 := by
sorry

end NUMINAMATH_CALUDE_complex_roots_sum_of_absolute_values_l1454_145457


namespace NUMINAMATH_CALUDE_farm_animals_count_l1454_145475

/-- Represents the farm with goats and sheep -/
structure Farm where
  goats : ℕ
  sheep : ℕ

/-- Calculates the total number of animals on the farm -/
def Farm.total (f : Farm) : ℕ := f.goats + f.sheep

/-- Theorem: Given the conditions, the total number of animals on the farm is 1524 -/
theorem farm_animals_count (f : Farm) 
  (ratio : f.goats * 7 = f.sheep * 5)
  (sale_amount : (f.goats / 2) * 40 + (f.sheep * 2 / 3) * 30 = 7200) : 
  f.total = 1524 := by
  sorry


end NUMINAMATH_CALUDE_farm_animals_count_l1454_145475


namespace NUMINAMATH_CALUDE_large_cylinder_height_l1454_145433

-- Define constants
def small_cylinder_diameter : ℝ := 3
def small_cylinder_height : ℝ := 6
def large_cylinder_diameter : ℝ := 20
def small_cylinders_to_fill : ℝ := 74.07407407407408

-- Define the theorem
theorem large_cylinder_height :
  let small_cylinder_volume := π * (small_cylinder_diameter / 2)^2 * small_cylinder_height
  let large_cylinder_radius := large_cylinder_diameter / 2
  let large_cylinder_volume := small_cylinders_to_fill * small_cylinder_volume
  large_cylinder_volume = π * large_cylinder_radius^2 * 10 := by
  sorry

end NUMINAMATH_CALUDE_large_cylinder_height_l1454_145433


namespace NUMINAMATH_CALUDE_laura_park_time_l1454_145404

/-- The number of trips Laura took to the park -/
def num_trips : ℕ := 6

/-- The time (in hours) spent walking to and from the park for each trip -/
def walking_time : ℝ := 0.5

/-- The fraction of total time spent in the park -/
def park_time_fraction : ℝ := 0.8

/-- The time (in hours) Laura spent at the park during each trip -/
def park_time : ℝ := 2

theorem laura_park_time :
  park_time = (park_time_fraction * num_trips * (park_time + walking_time)) / num_trips := by
  sorry

end NUMINAMATH_CALUDE_laura_park_time_l1454_145404


namespace NUMINAMATH_CALUDE_tournament_games_32_teams_l1454_145431

/-- The number of games required in a single-elimination tournament --/
def games_required (n : ℕ) : ℕ := n - 1

/-- A theorem stating that a single-elimination tournament with 32 teams requires 31 games --/
theorem tournament_games_32_teams :
  games_required 32 = 31 :=
by sorry

end NUMINAMATH_CALUDE_tournament_games_32_teams_l1454_145431


namespace NUMINAMATH_CALUDE_mo_hot_chocolate_consumption_l1454_145442

/-- The number of cups of hot chocolate Mo drinks on rainy mornings -/
def cups_of_hot_chocolate : ℚ := 1.75

/-- The number of cups of tea Mo drinks on non-rainy mornings -/
def cups_of_tea_non_rainy : ℕ := 5

/-- The total number of cups of tea and hot chocolate Mo drank last week -/
def total_cups_last_week : ℕ := 22

/-- The difference between tea cups and hot chocolate cups Mo drank last week -/
def tea_minus_chocolate : ℕ := 8

/-- The number of rainy days last week -/
def rainy_days : ℕ := 4

/-- The number of days in a week -/
def days_in_week : ℕ := 7

theorem mo_hot_chocolate_consumption :
  cups_of_hot_chocolate * rainy_days = 
    total_cups_last_week - (cups_of_tea_non_rainy * (days_in_week - rainy_days)) - tea_minus_chocolate := by
  sorry

end NUMINAMATH_CALUDE_mo_hot_chocolate_consumption_l1454_145442


namespace NUMINAMATH_CALUDE_compute_M_v_minus_2w_l1454_145452

variable (M : Matrix (Fin 2) (Fin 2) ℝ)
variable (v w : Fin 2 → ℝ)

axiom Mv : M.mulVec v = ![4, 2]
axiom Mw : M.mulVec w = ![5, 1]

theorem compute_M_v_minus_2w :
  M.mulVec (v - 2 • w) = ![-6, 0] := by sorry

end NUMINAMATH_CALUDE_compute_M_v_minus_2w_l1454_145452


namespace NUMINAMATH_CALUDE_increasing_function_derivative_relation_l1454_145468

open Set
open Function
open Topology

theorem increasing_function_derivative_relation 
  {a b : ℝ} (hab : a < b) (f : ℝ → ℝ) (hf : DifferentiableOn ℝ f (Ioo a b)) :
  (∀ x ∈ Ioo a b, (deriv f) x > 0 → StrictMonoOn f (Ioo a b)) ∧
  ¬(StrictMonoOn f (Ioo a b) → ∀ x ∈ Ioo a b, (deriv f) x > 0) :=
sorry

end NUMINAMATH_CALUDE_increasing_function_derivative_relation_l1454_145468


namespace NUMINAMATH_CALUDE_tan_alpha_value_l1454_145472

theorem tan_alpha_value (α : Real) 
  (h : (Real.sin α - 2 * Real.cos α) / (3 * Real.sin α + 5 * Real.cos α) = -5) : 
  Real.tan α = -23/16 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l1454_145472


namespace NUMINAMATH_CALUDE_intersection_P_complement_Q_l1454_145481

-- Define the sets P and Q
def P : Set ℝ := {1, 2, 3, 4}
def Q : Set ℝ := {3, 4, 5}

-- Define the universal set U as the set of real numbers
def U : Type := ℝ

-- Theorem statement
theorem intersection_P_complement_Q :
  P ∩ (Set.univ \ Q) = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_P_complement_Q_l1454_145481


namespace NUMINAMATH_CALUDE_sticks_per_matchbox_l1454_145467

/-- Given the following:
  * num_boxes: The number of boxes ordered
  * matchboxes_per_box: The number of matchboxes in each box
  * total_sticks: The total number of match sticks ordered

  Prove that the number of match sticks in each matchbox is 300.
-/
theorem sticks_per_matchbox
  (num_boxes : ℕ)
  (matchboxes_per_box : ℕ)
  (total_sticks : ℕ)
  (h1 : num_boxes = 4)
  (h2 : matchboxes_per_box = 20)
  (h3 : total_sticks = 24000) :
  total_sticks / (num_boxes * matchboxes_per_box) = 300 := by
  sorry

end NUMINAMATH_CALUDE_sticks_per_matchbox_l1454_145467


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_equation_l1454_145429

theorem sum_of_roots_quadratic_equation : 
  let f : ℝ → ℝ := λ x => x^2 + x - 2
  ∃ r₁ r₂ : ℝ, f r₁ = 0 ∧ f r₂ = 0 ∧ r₁ + r₂ = -1 :=
sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_equation_l1454_145429


namespace NUMINAMATH_CALUDE_fraction_simplification_l1454_145486

theorem fraction_simplification (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  1 / a + 1 / b - (2 * a + b) / (2 * a * b) = 1 / (2 * a) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1454_145486


namespace NUMINAMATH_CALUDE_nuts_division_l1454_145478

theorem nuts_division (bags : ℕ) (nuts_per_bag : ℕ) (students : ℕ) 
  (h1 : bags = 65) 
  (h2 : nuts_per_bag = 15) 
  (h3 : students = 13) : 
  (bags * nuts_per_bag) / students = 75 := by
  sorry

#check nuts_division

end NUMINAMATH_CALUDE_nuts_division_l1454_145478


namespace NUMINAMATH_CALUDE_rectangular_region_ratio_l1454_145473

theorem rectangular_region_ratio (L W : ℝ) (k : ℝ) : 
  L > 0 → W > 0 → k > 0 →
  L = k * W →
  L * W = 200 →
  2 * W + L = 40 →
  L / W = 2 := by
sorry

end NUMINAMATH_CALUDE_rectangular_region_ratio_l1454_145473


namespace NUMINAMATH_CALUDE_pizza_toppings_l1454_145449

theorem pizza_toppings (total_slices pepperoni_slices mushroom_slices : ℕ) 
  (h_total : total_slices = 24)
  (h_pepperoni : pepperoni_slices = 15)
  (h_mushroom : mushroom_slices = 15)
  (h_at_least_one : ∀ slice, slice ∈ Finset.range total_slices → 
    (slice ∈ Finset.range pepperoni_slices ∨ slice ∈ Finset.range mushroom_slices)) :
  (Finset.range pepperoni_slices ∩ Finset.range mushroom_slices).card = 6 := by
sorry

end NUMINAMATH_CALUDE_pizza_toppings_l1454_145449


namespace NUMINAMATH_CALUDE_ship_passengers_l1454_145430

theorem ship_passengers : ∃ (P : ℕ), 
  P > 0 ∧ 
  (P : ℚ) * (1/3 + 1/8 + 1/5 + 1/6) + 42 = P ∧ 
  P = 240 := by
  sorry

end NUMINAMATH_CALUDE_ship_passengers_l1454_145430


namespace NUMINAMATH_CALUDE_bill_after_30_days_l1454_145482

/-- The amount owed after applying late charges -/
def amount_owed (initial_bill : ℝ) (late_charge_rate : ℝ) (days : ℕ) : ℝ :=
  initial_bill * (1 + late_charge_rate) ^ (days / 10)

/-- Theorem stating the amount owed after 30 days -/
theorem bill_after_30_days (initial_bill : ℝ) (late_charge_rate : ℝ) :
  initial_bill = 500 →
  late_charge_rate = 0.02 →
  amount_owed initial_bill late_charge_rate 30 = 530.604 :=
by
  sorry

#eval amount_owed 500 0.02 30

end NUMINAMATH_CALUDE_bill_after_30_days_l1454_145482


namespace NUMINAMATH_CALUDE_maria_carrots_thrown_out_l1454_145497

/-- The number of carrots Maria initially picked -/
def initial_carrots : ℕ := 48

/-- The number of additional carrots Maria picked the next day -/
def additional_carrots : ℕ := 15

/-- The total number of carrots Maria had after picking additional carrots -/
def total_carrots : ℕ := 52

/-- The number of carrots Maria threw out -/
def carrots_thrown_out : ℕ := 11

theorem maria_carrots_thrown_out : 
  initial_carrots - carrots_thrown_out + additional_carrots = total_carrots :=
sorry

end NUMINAMATH_CALUDE_maria_carrots_thrown_out_l1454_145497


namespace NUMINAMATH_CALUDE_min_class_size_class_size_32_achievable_l1454_145424

/-- Represents the number of people in each group of the class --/
structure ClassGroups where
  first : ℕ
  second : ℕ
  third : ℕ

/-- The conditions of the tree-planting problem --/
def TreePlantingConditions (g : ClassGroups) : Prop :=
  g.second = (g.first + g.third) / 3 ∧
  4 * g.second = 5 * g.first + 3 * g.third - 72

/-- The theorem stating the minimum number of people in the class --/
theorem min_class_size (g : ClassGroups) 
  (h : TreePlantingConditions g) : 
  g.first + g.second + g.third ≥ 32 := by
  sorry

/-- The theorem stating that 32 is achievable --/
theorem class_size_32_achievable : 
  ∃ g : ClassGroups, TreePlantingConditions g ∧ g.first + g.second + g.third = 32 := by
  sorry

end NUMINAMATH_CALUDE_min_class_size_class_size_32_achievable_l1454_145424


namespace NUMINAMATH_CALUDE_aunt_gift_amount_l1454_145451

theorem aunt_gift_amount (jade_initial julia_initial jack_initial total_after_gift : ℕ) : 
  jade_initial = 38 →
  julia_initial = jade_initial / 2 →
  jack_initial = 12 →
  total_after_gift = 132 →
  ∃ gift : ℕ, 
    jade_initial + julia_initial + jack_initial + 3 * gift = total_after_gift ∧
    gift = 21 := by
  sorry

end NUMINAMATH_CALUDE_aunt_gift_amount_l1454_145451


namespace NUMINAMATH_CALUDE_vector_difference_magnitude_l1454_145483

def a : Fin 2 → ℝ := ![2, 1]
def b : Fin 2 → ℝ := ![-2, 4]

theorem vector_difference_magnitude : ‖a - b‖ = 5 := by sorry

end NUMINAMATH_CALUDE_vector_difference_magnitude_l1454_145483


namespace NUMINAMATH_CALUDE_expression_value_l1454_145469

theorem expression_value (x y z : ℤ) (hx : x = 3) (hy : y = 2) (hz : z = 1) :
  3 * x - 4 * y + 2 * z = 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1454_145469


namespace NUMINAMATH_CALUDE_slower_train_speed_l1454_145403

/-- Calculates the speed of the slower train given the conditions of the problem -/
theorem slower_train_speed (train_length : ℝ) (faster_speed : ℝ) (passing_time : ℝ) : 
  train_length = 500 →
  faster_speed = 45 →
  passing_time = 60 / 3600 →
  (faster_speed + (2 * train_length / 1000) / passing_time) - faster_speed = 15 := by
  sorry

#check slower_train_speed

end NUMINAMATH_CALUDE_slower_train_speed_l1454_145403


namespace NUMINAMATH_CALUDE_seastar_arms_l1454_145410

theorem seastar_arms (num_starfish : ℕ) (arms_per_starfish : ℕ) (total_arms : ℕ) : 
  num_starfish = 7 → arms_per_starfish = 5 → total_arms = 49 → 
  total_arms - (num_starfish * arms_per_starfish) = 14 := by
sorry

end NUMINAMATH_CALUDE_seastar_arms_l1454_145410


namespace NUMINAMATH_CALUDE_michaels_pets_cats_percentage_l1454_145400

/-- Proves that the percentage of cats among Michael's pets is 50% -/
theorem michaels_pets_cats_percentage
  (total_pets : ℕ)
  (dog_percentage : ℚ)
  (bunny_count : ℕ)
  (h1 : total_pets = 36)
  (h2 : dog_percentage = 1/4)
  (h3 : bunny_count = 9)
  (h4 : (dog_percentage * total_pets).num + bunny_count + (total_pets - (dog_percentage * total_pets).num - bunny_count) = total_pets) :
  (total_pets - (dog_percentage * total_pets).num - bunny_count) / total_pets = 1/2 := by
  sorry

#check michaels_pets_cats_percentage

end NUMINAMATH_CALUDE_michaels_pets_cats_percentage_l1454_145400


namespace NUMINAMATH_CALUDE_y_in_terms_of_x_l1454_145435

theorem y_in_terms_of_x (p : ℝ) (x y : ℝ) 
  (hx : x = 1 + 3^p) (hy : y = 1 + 3^(-p)) : 
  y = x / (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_y_in_terms_of_x_l1454_145435


namespace NUMINAMATH_CALUDE_difference_of_squares_special_case_l1454_145414

theorem difference_of_squares_special_case : (527 : ℤ) * 527 - 526 * 528 = 1 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_special_case_l1454_145414


namespace NUMINAMATH_CALUDE_maria_berry_purchase_l1454_145477

/-- The number of cartons Maria needs to buy -/
def cartons_to_buy (total_needed : ℕ) (strawberries : ℕ) (blueberries : ℕ) : ℕ :=
  total_needed - (strawberries + blueberries)

/-- Theorem stating that Maria needs to buy 9 more cartons of berries -/
theorem maria_berry_purchase : cartons_to_buy 21 4 8 = 9 := by
  sorry

end NUMINAMATH_CALUDE_maria_berry_purchase_l1454_145477


namespace NUMINAMATH_CALUDE_circle_equations_correct_l1454_145413

-- Define the points A, B, and D
def A : ℝ × ℝ := (-1, 5)
def B : ℝ × ℝ := (5, 5)
def D : ℝ × ℝ := (5, -1)

-- Define circle C
def circle_C (x y : ℝ) : Prop :=
  (x - 2)^2 + (y - 2)^2 = 18

-- Define circle M
def circle_M (x y : ℝ) : Prop :=
  (x - 6)^2 + (y - 6)^2 = 2

-- Theorem statement
theorem circle_equations_correct :
  (circle_C A.1 A.2 ∧ circle_C B.1 B.2 ∧ circle_C D.1 D.2) ∧
  (∃ (t : ℝ), circle_C (B.1 + t) (B.2 + t) ∧ circle_M (B.1 + t) (B.2 + t)) ∧
  (∀ (x y : ℝ), circle_M x y → (x - B.1)^2 + (y - B.2)^2 = 2) :=
by sorry

end NUMINAMATH_CALUDE_circle_equations_correct_l1454_145413


namespace NUMINAMATH_CALUDE_point_on_line_l1454_145438

/-- Given a point P(x, b) on the line x + y = 30, if the slope of OP is 4 (where O is the origin), then b = 24. -/
theorem point_on_line (x b : ℝ) : 
  x + b = 30 →  -- P(x, b) is on the line x + y = 30
  (b / x = 4) →  -- The slope of OP is 4
  b = 24 := by
sorry

end NUMINAMATH_CALUDE_point_on_line_l1454_145438


namespace NUMINAMATH_CALUDE_sin_cos_value_l1454_145462

theorem sin_cos_value (x : ℝ) : 
  let a : ℝ × ℝ := (4 * Real.sin x, 1 - Real.cos x)
  let b : ℝ × ℝ := (1, -2)
  (a.1 * b.1 + a.2 * b.2 = -2) → (Real.sin x * Real.cos x = -2/5) := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_value_l1454_145462


namespace NUMINAMATH_CALUDE_cycle_original_price_l1454_145411

/-- Given a cycle sold at a 15% loss for Rs. 1190, prove that the original price was Rs. 1400 -/
theorem cycle_original_price (selling_price : ℝ) (loss_percentage : ℝ) 
  (h1 : selling_price = 1190)
  (h2 : loss_percentage = 15) : 
  ∃ (original_price : ℝ), 
    original_price * (1 - loss_percentage / 100) = selling_price ∧ 
    original_price = 1400 := by
  sorry

end NUMINAMATH_CALUDE_cycle_original_price_l1454_145411


namespace NUMINAMATH_CALUDE_part_one_part_two_l1454_145458

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 2}
def B : Set ℝ := {x | x^2 + x - 2 < 0}

-- Part 1
theorem part_one : A 0 ∩ (Set.univ \ B) = {x | 1 ≤ x ∧ x ≤ 2} := by sorry

-- Part 2
theorem part_two (a : ℝ) : (∀ x ∈ A a, x ∉ B) ↔ (a ≤ -4 ∨ a ≥ 1) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1454_145458


namespace NUMINAMATH_CALUDE_original_number_proof_l1454_145466

theorem original_number_proof (N : ℝ) (x y z : ℝ) : 
  (N * 1.2 = 480) →
  ((480 * 0.85) * x^2 = 5*x^3 + 24*x - 50) →
  ((N / y) * 0.75 = z) →
  (z = x * y) →
  N = 400 := by
sorry

end NUMINAMATH_CALUDE_original_number_proof_l1454_145466
