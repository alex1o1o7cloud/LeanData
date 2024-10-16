import Mathlib

namespace NUMINAMATH_CALUDE_singing_competition_stats_l4043_404383

def scores : List ℝ := [9.40, 9.40, 9.50, 9.50, 9.50, 9.60, 9.60, 9.60, 9.60, 9.60, 9.70, 9.70, 9.70, 9.70, 9.80, 9.80, 9.80, 9.90]

def median (l : List ℝ) : ℝ := sorry

def mode (l : List ℝ) : ℝ := sorry

theorem singing_competition_stats :
  median scores = 9.60 ∧ mode scores = 9.60 := by sorry

end NUMINAMATH_CALUDE_singing_competition_stats_l4043_404383


namespace NUMINAMATH_CALUDE_increasing_f_implies_t_geq_5_l4043_404379

/-- A cubic function with a parameter t -/
def f (t : ℝ) (x : ℝ) : ℝ := -x^3 + x^2 + t*x + t

/-- The derivative of f with respect to x -/
def f' (t : ℝ) (x : ℝ) : ℝ := -3*x^2 + 2*x + t

theorem increasing_f_implies_t_geq_5 :
  (∀ x ∈ Set.Ioo (-1 : ℝ) 1, Monotone (f t)) →
  t ≥ 5 := by sorry

end NUMINAMATH_CALUDE_increasing_f_implies_t_geq_5_l4043_404379


namespace NUMINAMATH_CALUDE_unique_root_implies_k_range_l4043_404381

/-- A function f(x) with parameter k -/
def f (k : ℝ) (x : ℝ) : ℝ := x^2 + (1-k)*x - k

/-- Theorem: If f(x) has exactly one root in (2,3), then k is in (2,3) -/
theorem unique_root_implies_k_range (k : ℝ) :
  (∃! x, x ∈ (Set.Ioo 2 3) ∧ f k x = 0) → k ∈ Set.Ioo 2 3 :=
by sorry

end NUMINAMATH_CALUDE_unique_root_implies_k_range_l4043_404381


namespace NUMINAMATH_CALUDE_mikes_games_this_year_l4043_404349

def total_games : ℕ := 54
def last_year_games : ℕ := 39
def missed_games : ℕ := 41

theorem mikes_games_this_year : 
  total_games - last_year_games = 15 := by sorry

end NUMINAMATH_CALUDE_mikes_games_this_year_l4043_404349


namespace NUMINAMATH_CALUDE_max_value_a_l4043_404319

theorem max_value_a (a b c e : ℕ+) 
  (h1 : a < 2 * b)
  (h2 : b < 3 * c)
  (h3 : c < 5 * e)
  (h4 : e < 100) :
  a ≤ 2961 ∧ ∃ (a' b' c' e' : ℕ+), 
    a' = 2961 ∧ 
    a' < 2 * b' ∧ 
    b' < 3 * c' ∧ 
    c' < 5 * e' ∧ 
    e' < 100 :=
by sorry

end NUMINAMATH_CALUDE_max_value_a_l4043_404319


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l4043_404300

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 2) :
  1/x + 1/y ≥ 2 ∧ (1/x + 1/y = 2 ↔ x = 1 ∧ y = 1) := by
  sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l4043_404300


namespace NUMINAMATH_CALUDE_prob_four_red_cards_standard_deck_l4043_404344

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (red_cards : ℕ)
  (black_cards : ℕ)

/-- Probability of drawing n red cards in a row from a deck -/
def prob_n_red_cards (d : Deck) (n : ℕ) : ℚ :=
  sorry

theorem prob_four_red_cards_standard_deck :
  let standard_deck : Deck := ⟨52, 26, 26⟩
  prob_n_red_cards standard_deck 4 = 276 / 9801 := by
  sorry

end NUMINAMATH_CALUDE_prob_four_red_cards_standard_deck_l4043_404344


namespace NUMINAMATH_CALUDE_football_team_linemen_l4043_404357

-- Define the constants from the problem
def cooler_capacity : ℕ := 126
def skill_players : ℕ := 10
def lineman_consumption : ℕ := 8
def skill_player_consumption : ℕ := 6
def skill_players_drinking : ℕ := 5

-- Define the number of linemen as a variable
def num_linemen : ℕ := sorry

-- Theorem statement
theorem football_team_linemen :
  num_linemen * lineman_consumption +
  skill_players_drinking * skill_player_consumption = cooler_capacity :=
by sorry

end NUMINAMATH_CALUDE_football_team_linemen_l4043_404357


namespace NUMINAMATH_CALUDE_prime_sum_product_l4043_404326

theorem prime_sum_product (p q : ℕ) : 
  Nat.Prime p → Nat.Prime q → p + q = 101 → p * q = 194 := by
sorry

end NUMINAMATH_CALUDE_prime_sum_product_l4043_404326


namespace NUMINAMATH_CALUDE_alice_acorn_price_l4043_404370

/-- Given the conditions of Alice and Bob's acorn purchases, prove that Alice paid $15 for each acorn. -/
theorem alice_acorn_price (alice_acorns : ℕ) (bob_price : ℝ) (alice_bob_ratio : ℝ) : 
  alice_acorns = 3600 → 
  bob_price = 6000 → 
  alice_bob_ratio = 9 → 
  (alice_bob_ratio * bob_price) / alice_acorns = 15 := by
sorry

end NUMINAMATH_CALUDE_alice_acorn_price_l4043_404370


namespace NUMINAMATH_CALUDE_three_heads_probability_l4043_404396

def prob_heads : ℚ := 1/2

theorem three_heads_probability :
  let prob_three_heads := prob_heads * prob_heads * prob_heads
  prob_three_heads = 1/8 := by sorry

end NUMINAMATH_CALUDE_three_heads_probability_l4043_404396


namespace NUMINAMATH_CALUDE_plane_equation_l4043_404391

/-- The plane passing through points (0,3,-1), (4,7,1), and (2,5,0) has the equation y - 2z - 5 = 0 -/
theorem plane_equation (p q r : ℝ × ℝ × ℝ) : 
  p = (0, 3, -1) → q = (4, 7, 1) → r = (2, 5, 0) →
  ∃ (A B C D : ℤ), 
    (A > 0) ∧ 
    (Int.gcd (Int.natAbs A) (Int.gcd (Int.natAbs B) (Int.gcd (Int.natAbs C) (Int.natAbs D))) = 1) ∧
    (∀ (x y z : ℝ), A * x + B * y + C * z + D = 0 ↔ y - 2 * z - 5 = 0) :=
by sorry

end NUMINAMATH_CALUDE_plane_equation_l4043_404391


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l4043_404363

theorem inverse_variation_problem (x y : ℝ) (k : ℝ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : x^2 * y = k) (h4 : 2^2 * 10 = k) (h5 : y = 4000) : x = 1/10 := by
  sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l4043_404363


namespace NUMINAMATH_CALUDE_max_value_abc_l4043_404313

theorem max_value_abc (a b c : ℕ+) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) (h4 : a * b * c = 16) :
  (∀ x y z : ℕ+, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ x * y * z = 16 →
    x ^ y.val - y ^ z.val + z ^ x.val ≤ a ^ b.val - b ^ c.val + c ^ a.val) →
  a ^ b.val - b ^ c.val + c ^ a.val = 263 :=
by sorry

end NUMINAMATH_CALUDE_max_value_abc_l4043_404313


namespace NUMINAMATH_CALUDE_added_number_after_doubling_l4043_404333

theorem added_number_after_doubling (original : ℕ) (added : ℕ) : 
  original = 9 → 3 * (2 * original + added) = 72 → added = 6 := by
  sorry

end NUMINAMATH_CALUDE_added_number_after_doubling_l4043_404333


namespace NUMINAMATH_CALUDE_arithmetic_combination_exists_l4043_404348

theorem arithmetic_combination_exists : ∃ (f : ℕ → ℕ → ℕ) (g : ℕ → ℕ → ℕ) (h : ℕ → ℕ → ℕ),
  (f 1 (g 2 3)) * (h 4 5) = 100 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_combination_exists_l4043_404348


namespace NUMINAMATH_CALUDE_number_exceeding_percentage_l4043_404358

theorem number_exceeding_percentage (x : ℝ) : x = 60 ↔ x = 0.12 * x + 52.8 := by
  sorry

end NUMINAMATH_CALUDE_number_exceeding_percentage_l4043_404358


namespace NUMINAMATH_CALUDE_expression_value_l4043_404360

theorem expression_value (x y : ℝ) (hx : x = 8) (hy : y = 3) :
  (x - 2*y) * (x + 2*y) = 28 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l4043_404360


namespace NUMINAMATH_CALUDE_lucky_number_2005_to_52000_l4043_404364

/-- A natural number is a lucky number if the sum of its digits is 7 -/
def is_lucky_number (n : ℕ) : Prop :=
  (n.digits 10).sum = 7

/-- The sequence of lucky numbers in ascending order -/
def lucky_number_sequence : ℕ → ℕ :=
  sorry

/-- The 2005th lucky number is the nth in the sequence -/
axiom a_2005_is_nth : ∃ n : ℕ, lucky_number_sequence n = 2005

theorem lucky_number_2005_to_52000 :
  ∃ n : ℕ, lucky_number_sequence n = 2005 ∧ lucky_number_sequence (5 * n) = 52000 :=
sorry

end NUMINAMATH_CALUDE_lucky_number_2005_to_52000_l4043_404364


namespace NUMINAMATH_CALUDE_greg_age_l4043_404304

/-- Given the ages and relationships of siblings, prove Greg's age -/
theorem greg_age (cindy_age : ℕ) (jan_age : ℕ) (marcia_age : ℕ) (greg_age : ℕ)
  (h1 : cindy_age = 5)
  (h2 : jan_age = cindy_age + 2)
  (h3 : marcia_age = 2 * jan_age)
  (h4 : greg_age = marcia_age + 2) :
  greg_age = 16 := by
sorry

end NUMINAMATH_CALUDE_greg_age_l4043_404304


namespace NUMINAMATH_CALUDE_right_triangles_on_circle_l4043_404314

/-- The number of right-angled triangles formed by 2n equally spaced points on a circle -/
theorem right_triangles_on_circle (n : ℕ) (h : n > 1) :
  (number_of_right_triangles : ℕ) = 2 * n * (n - 1) :=
by sorry

end NUMINAMATH_CALUDE_right_triangles_on_circle_l4043_404314


namespace NUMINAMATH_CALUDE_cost_of_450_candies_l4043_404305

/-- The cost of buying a given number of chocolate candies -/
def cost_of_candies (candies_per_box : ℕ) (cost_per_box : ℚ) (total_candies : ℕ) : ℚ :=
  (total_candies / candies_per_box : ℚ) * cost_per_box

/-- Theorem: The cost of 450 chocolate candies is $112.50, given that a box of 30 costs $7.50 -/
theorem cost_of_450_candies :
  cost_of_candies 30 (7.5 : ℚ) 450 = (112.5 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_cost_of_450_candies_l4043_404305


namespace NUMINAMATH_CALUDE_triangle_area_is_11_over_2_l4043_404389

-- Define the vertices of the triangle
def D : ℝ × ℝ := (2, -3)
def E : ℝ × ℝ := (0, 4)
def F : ℝ × ℝ := (3, -1)

-- Define the function to calculate the area of a triangle given its vertices
def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem triangle_area_is_11_over_2 : triangleArea D E F = 11 / 2 := by sorry

end NUMINAMATH_CALUDE_triangle_area_is_11_over_2_l4043_404389


namespace NUMINAMATH_CALUDE_sum_and_double_l4043_404356

theorem sum_and_double : 2 * (1324 + 4231 + 3124 + 2413) = 22184 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_double_l4043_404356


namespace NUMINAMATH_CALUDE_event_probability_estimate_l4043_404303

-- Define the frequency function
def frequency : ℕ → ℝ
| 20 => 0.300
| 50 => 0.360
| 100 => 0.350
| 300 => 0.350
| 500 => 0.352
| 1000 => 0.351
| 5000 => 0.351
| _ => 0  -- For other values, we set the frequency to 0

-- Define the set of trial numbers
def trialNumbers : Set ℕ := {20, 50, 100, 300, 500, 1000, 5000}

-- Theorem statement
theorem event_probability_estimate :
  ∀ ε > 0, ∃ N ∈ trialNumbers, ∀ n ∈ trialNumbers, n ≥ N → |frequency n - 0.35| < ε :=
sorry

end NUMINAMATH_CALUDE_event_probability_estimate_l4043_404303


namespace NUMINAMATH_CALUDE_polynomial_derivative_symmetry_l4043_404310

/-- Given a polynomial function f(x) = ax^4 + bx^2 + c, 
    if f'(1) = 2, then f'(-1) = -2 -/
theorem polynomial_derivative_symmetry 
  (a b c : ℝ) 
  (f : ℝ → ℝ)
  (h_f : ∀ x, f x = a * x^4 + b * x^2 + c)
  (h_f'_1 : (deriv f) 1 = 2) : 
  (deriv f) (-1) = -2 := by
sorry

end NUMINAMATH_CALUDE_polynomial_derivative_symmetry_l4043_404310


namespace NUMINAMATH_CALUDE_event_A_necessary_not_sufficient_for_B_l4043_404329

/- Define the bag contents -/
def total_balls : ℕ := 4
def red_balls : ℕ := 2
def white_balls : ℕ := 2

/- Define the events -/
def event_A (drawn_red : ℕ) : Prop := drawn_red ≥ 1
def event_B (drawn_red : ℕ) : Prop := drawn_red = 1

/- Define the relationship between events -/
theorem event_A_necessary_not_sufficient_for_B :
  (∀ (drawn_red : ℕ), event_B drawn_red → event_A drawn_red) ∧
  (∃ (drawn_red : ℕ), event_A drawn_red ∧ ¬event_B drawn_red) :=
sorry

end NUMINAMATH_CALUDE_event_A_necessary_not_sufficient_for_B_l4043_404329


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l4043_404343

/-- A hyperbola with center at the origin and axes of symmetry along the coordinate axes -/
structure CenteredHyperbola where
  /-- The angle of inclination of one of the asymptotes -/
  asymptote_angle : ℝ

/-- The eccentricity of a hyperbola -/
def eccentricity (h : CenteredHyperbola) : ℝ :=
  sorry

/-- Theorem stating the possible eccentricities of a hyperbola with an asymptote angle of π/3 -/
theorem hyperbola_eccentricity (h : CenteredHyperbola) 
  (h_angle : h.asymptote_angle = π / 3) : 
  eccentricity h = 2 ∨ eccentricity h = 2 * Real.sqrt 3 / 3 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l4043_404343


namespace NUMINAMATH_CALUDE_odd_periodic_function_sum_l4043_404372

-- Define the properties of the function f
def is_odd_periodic_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ 
  (f 1 = 2) ∧ 
  (∀ x, f (x + 1) = f (x + 5))

-- State the theorem
theorem odd_periodic_function_sum (f : ℝ → ℝ) 
  (h : is_odd_periodic_function f) : f 12 + f 3 = -2 := by
  sorry

end NUMINAMATH_CALUDE_odd_periodic_function_sum_l4043_404372


namespace NUMINAMATH_CALUDE_fraction_inequality_conditions_l4043_404338

theorem fraction_inequality_conditions (a b : ℝ) :
  (∀ x : ℝ, |((x^2 + a*x + b) / (x^2 + 2*x + 2))| < 1) ↔ (a = 2 ∧ 0 < b ∧ b < 2) := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_conditions_l4043_404338


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_equality_l4043_404353

/-- Represents a repeating decimal with a repeating part and a period length. -/
structure RepeatingDecimal where
  repeating_part : ℕ
  period_length : ℕ

/-- Converts a RepeatingDecimal to a rational number. -/
def to_rational (d : RepeatingDecimal) : ℚ :=
  d.repeating_part / (10^d.period_length - 1)

/-- The sum of the three given repeating decimals equals 10099098/29970003. -/
theorem repeating_decimal_sum_equality : 
  let d1 := RepeatingDecimal.mk 3 1
  let d2 := RepeatingDecimal.mk 4 3
  let d3 := RepeatingDecimal.mk 5 4
  to_rational d1 + to_rational d2 + to_rational d3 = 10099098 / 29970003 := by
  sorry

#eval (10099098 : ℚ) / 29970003

end NUMINAMATH_CALUDE_repeating_decimal_sum_equality_l4043_404353


namespace NUMINAMATH_CALUDE_unique_three_digit_reborn_number_l4043_404323

def is_reborn_number (n : ℕ) : Prop :=
  ∃ a b c : ℕ,
    n = 100 * a + 10 * b + c ∧
    0 ≤ a ∧ a ≤ 9 ∧
    0 ≤ b ∧ b ≤ 9 ∧
    0 ≤ c ∧ c ≤ 9 ∧
    (a ≠ b ∨ b ≠ c ∨ a ≠ c) ∧
    n = (100 * max a (max b c) + 10 * max (min a b) (max (min a c) (min b c)) + min a (min b c)) -
        (100 * min a (min b c) + 10 * min (max a b) (min (max a c) (max b c)) + max a (max b c))

theorem unique_three_digit_reborn_number :
  ∀ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ is_reborn_number n ↔ n = 495 := by
  sorry

end NUMINAMATH_CALUDE_unique_three_digit_reborn_number_l4043_404323


namespace NUMINAMATH_CALUDE_participation_plans_eq_48_l4043_404359

/-- The number of different participation plans for selecting 3 out of 5 students
    for math, physics, and chemistry competitions, where each student competes
    in one subject and student A cannot participate in the physics competition. -/
def participation_plans : ℕ :=
  let total_students : ℕ := 5
  let selected_students : ℕ := 3
  let competitions : ℕ := 3
  let student_a_options : ℕ := 2  -- math or chemistry

  let scenario1 : ℕ := (total_students - 1).factorial / (total_students - 1 - selected_students).factorial
  let scenario2 : ℕ := student_a_options * ((total_students - 1).factorial / (total_students - 1 - (selected_students - 1)).factorial)

  scenario1 + scenario2

theorem participation_plans_eq_48 :
  participation_plans = 48 := by
  sorry

end NUMINAMATH_CALUDE_participation_plans_eq_48_l4043_404359


namespace NUMINAMATH_CALUDE_arithmetic_mean_two_digit_multiples_of_eight_l4043_404335

theorem arithmetic_mean_two_digit_multiples_of_eight : 
  let first_multiple := 16
  let last_multiple := 96
  let number_of_multiples := (last_multiple - first_multiple) / 8 + 1
  (first_multiple + last_multiple) / 2 = 56 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_two_digit_multiples_of_eight_l4043_404335


namespace NUMINAMATH_CALUDE_cone_base_circumference_l4043_404351

theorem cone_base_circumference (V : ℝ) (h : ℝ) (r : ℝ) :
  V = 18 * Real.pi ∧ h = 3 ∧ V = (1/3) * Real.pi * r^2 * h →
  2 * Real.pi * r = 6 * Real.sqrt 2 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_cone_base_circumference_l4043_404351


namespace NUMINAMATH_CALUDE_latest_score_is_68_l4043_404301

def scores : List Int := [68, 75, 83, 94]

def is_integer_average (subset : List Int) : Prop :=
  subset.sum % subset.length = 0

theorem latest_score_is_68 :
  (∀ subset : List Int, subset ⊆ scores → is_integer_average subset) →
  scores.head? = some 68 :=
by sorry

end NUMINAMATH_CALUDE_latest_score_is_68_l4043_404301


namespace NUMINAMATH_CALUDE_quadratic_roots_l4043_404340

theorem quadratic_roots : 
  let f : ℝ → ℝ := λ x ↦ x^2 - 3
  ∃ x₁ x₂ : ℝ, x₁ = Real.sqrt 3 ∧ x₂ = -Real.sqrt 3 ∧ f x₁ = 0 ∧ f x₂ = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_l4043_404340


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l4043_404378

theorem complex_magnitude_problem (z : ℂ) (h : z + 3 * Complex.I = 3 - Complex.I) : 
  Complex.abs z = 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l4043_404378


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l4043_404320

theorem pure_imaginary_condition (m : ℝ) : 
  (∀ z : ℂ, z = Complex.mk (m^2 - 1) (m + 1) → z.re = 0 ∧ z.im ≠ 0) → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l4043_404320


namespace NUMINAMATH_CALUDE_no_rational_q_exists_l4043_404384

theorem no_rational_q_exists : ¬ ∃ (q : ℚ) (b c : ℚ),
  -- f(x) = x^2 + bx + c is a quadratic trinomial
  -- The coefficients 1, b, and c form a geometric progression with common ratio q
  ((1 = b ∧ b = c * q) ∨ (1 = c * q ∧ c * q = b) ∨ (b = 1 * q ∧ 1 * q = c)) ∧
  -- The difference between the roots of f(x) is q
  (b^2 - 4*c).sqrt = q := by
sorry

end NUMINAMATH_CALUDE_no_rational_q_exists_l4043_404384


namespace NUMINAMATH_CALUDE_dave_shows_per_week_l4043_404368

theorem dave_shows_per_week :
  let strings_per_show : ℕ := 2
  let total_weeks : ℕ := 12
  let total_strings : ℕ := 144
  let shows_per_week : ℕ := total_strings / (strings_per_show * total_weeks)
  shows_per_week = 6 :=
by sorry

end NUMINAMATH_CALUDE_dave_shows_per_week_l4043_404368


namespace NUMINAMATH_CALUDE_sixth_rack_dvds_l4043_404394

def dvd_sequence : ℕ → ℕ
  | 0 => 2
  | n + 1 => 2 * dvd_sequence n

theorem sixth_rack_dvds : dvd_sequence 5 = 64 := by
  sorry

end NUMINAMATH_CALUDE_sixth_rack_dvds_l4043_404394


namespace NUMINAMATH_CALUDE_circle_unique_dual_symmetry_l4043_404327

-- Define the shapes
inductive Shape
  | Parallelogram
  | Circle
  | EquilateralTriangle
  | RegularPentagon

-- Define symmetry properties
def isAxiallySymmetric (s : Shape) : Prop :=
  match s with
  | Shape.Circle => true
  | Shape.EquilateralTriangle => true
  | Shape.RegularPentagon => true
  | _ => false

def isCentrallySymmetric (s : Shape) : Prop :=
  match s with
  | Shape.Parallelogram => true
  | Shape.Circle => true
  | _ => false

-- Theorem statement
theorem circle_unique_dual_symmetry :
  ∀ s : Shape, (isAxiallySymmetric s ∧ isCentrallySymmetric s) ↔ s = Shape.Circle :=
by sorry

end NUMINAMATH_CALUDE_circle_unique_dual_symmetry_l4043_404327


namespace NUMINAMATH_CALUDE_distance_between_trees_l4043_404386

/-- Given a yard of length 275 meters with 26 trees planted at equal distances,
    including one at each end, the distance between consecutive trees is 11 meters. -/
theorem distance_between_trees (yard_length : ℝ) (num_trees : ℕ) :
  yard_length = 275 →
  num_trees = 26 →
  yard_length / (num_trees - 1) = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_distance_between_trees_l4043_404386


namespace NUMINAMATH_CALUDE_semicircle_perimeter_approx_l4043_404397

/-- The perimeter of a semicircle with radius 9 is approximately 46.27 units. -/
theorem semicircle_perimeter_approx : ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |((9 * Real.pi + 18) : ℝ) - 46.27| < ε := by
  sorry

end NUMINAMATH_CALUDE_semicircle_perimeter_approx_l4043_404397


namespace NUMINAMATH_CALUDE_container_volume_ratio_l4043_404324

theorem container_volume_ratio (V1 V2 V3 : ℚ) : 
  (3/7 : ℚ) * V1 = V2 →  -- First container's juice fills second container
  (3/5 : ℚ) * V3 + (2/3 : ℚ) * ((3/7 : ℚ) * V1) = (4/5 : ℚ) * V3 →  -- Third container's final state
  V1 / V2 = 7/3 := by
sorry

end NUMINAMATH_CALUDE_container_volume_ratio_l4043_404324


namespace NUMINAMATH_CALUDE_hiking_rate_ratio_l4043_404341

-- Define the given constants
def rate_up : ℝ := 6
def time_up : ℝ := 2
def distance_down : ℝ := 18

-- Define the theorem
theorem hiking_rate_ratio :
  let distance_up := rate_up * time_up
  let time_down := time_up
  let rate_down := distance_down / time_down
  rate_down / rate_up = 1.5 := by
sorry

end NUMINAMATH_CALUDE_hiking_rate_ratio_l4043_404341


namespace NUMINAMATH_CALUDE_beadshop_profit_ratio_l4043_404375

theorem beadshop_profit_ratio : 
  ∀ (total_profit monday_profit tuesday_profit wednesday_profit : ℝ),
    total_profit = 1200 →
    monday_profit = (1/3) * total_profit →
    wednesday_profit = 500 →
    tuesday_profit = total_profit - monday_profit - wednesday_profit →
    tuesday_profit / total_profit = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_beadshop_profit_ratio_l4043_404375


namespace NUMINAMATH_CALUDE_interest_rate_multiple_l4043_404346

theorem interest_rate_multiple (P r m : ℝ) 
  (h1 : P * r^2 = 40)
  (h2 : P * (m * r)^2 = 360) :
  m = 3 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_multiple_l4043_404346


namespace NUMINAMATH_CALUDE_sugar_price_increase_l4043_404365

theorem sugar_price_increase (original_price : ℝ) (consumption_reduction : ℝ) :
  original_price = 3 →
  consumption_reduction = 0.4 →
  let new_consumption := 1 - consumption_reduction
  let new_price := original_price / new_consumption
  new_price = 5 := by
  sorry

end NUMINAMATH_CALUDE_sugar_price_increase_l4043_404365


namespace NUMINAMATH_CALUDE_arithmetic_equalities_l4043_404376

theorem arithmetic_equalities :
  (187 / 12 - 63 / 12 - 52 / 12 = 6) ∧
  (321321 * 123 - 123123 * 321 = 0) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equalities_l4043_404376


namespace NUMINAMATH_CALUDE_radical_conjugate_sum_product_l4043_404325

theorem radical_conjugate_sum_product (c d : ℝ) : 
  (c + 2 * Real.sqrt d) + (c - 2 * Real.sqrt d) = 6 ∧ 
  (c + 2 * Real.sqrt d) * (c - 2 * Real.sqrt d) = 4 → 
  c + d = 17/4 := by
sorry

end NUMINAMATH_CALUDE_radical_conjugate_sum_product_l4043_404325


namespace NUMINAMATH_CALUDE_square_diff_ratio_equals_one_third_l4043_404347

theorem square_diff_ratio_equals_one_third :
  (2025^2 - 2018^2) / (2032^2 - 2011^2) = 1/3 := by
sorry

end NUMINAMATH_CALUDE_square_diff_ratio_equals_one_third_l4043_404347


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l4043_404339

theorem sum_of_coefficients (a b c d e : ℝ) : 
  (∀ x, 729 * x^3 + 8 = (a * x + b) * (c * x^2 + d * x + e)) →
  a + b + c + d + e = 78 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l4043_404339


namespace NUMINAMATH_CALUDE_puzzle_solution_l4043_404328

/-- A function that represents the puzzle rule --/
def puzzleRule (n : ℕ) : ℕ := sorry

/-- The puzzle conditions --/
axiom rule_111 : puzzleRule 111 = 9
axiom rule_444 : puzzleRule 444 = 12
axiom rule_777 : puzzleRule 777 = 15

/-- The theorem to prove --/
theorem puzzle_solution : ∃ (n : ℕ), puzzleRule n = 15 ∧ n = 777 := by sorry

end NUMINAMATH_CALUDE_puzzle_solution_l4043_404328


namespace NUMINAMATH_CALUDE_expression_has_17_digits_l4043_404393

-- Define a function to calculate the number of digits in a number
def numDigits (n : ℕ) : ℕ := sorry

-- Define the expression
def expression : ℕ := (8 * 10^10) * (10 * 10^5)

-- Theorem statement
theorem expression_has_17_digits : numDigits expression = 17 := by sorry

end NUMINAMATH_CALUDE_expression_has_17_digits_l4043_404393


namespace NUMINAMATH_CALUDE_geometric_sequence_proof_l4043_404367

def is_geometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_proof (a : ℕ → ℝ) 
  (h_positive : ∀ n, a n > 0)
  (h_ratio : ∀ n, 2 * a n = 3 * a (n + 1))
  (h_product : a 2 * a 5 = 8 / 27) :
  is_geometric a ∧ a 6 = 16 / 81 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_proof_l4043_404367


namespace NUMINAMATH_CALUDE_solve_strawberry_problem_l4043_404342

def strawberry_problem (christine_pounds rachel_pounds total_pies : ℕ) : Prop :=
  rachel_pounds = 2 * christine_pounds →
  christine_pounds = 10 →
  total_pies = 10 →
  (christine_pounds + rachel_pounds) / total_pies = 3

theorem solve_strawberry_problem :
  ∃ (christine_pounds rachel_pounds total_pies : ℕ),
    strawberry_problem christine_pounds rachel_pounds total_pies :=
by
  sorry

end NUMINAMATH_CALUDE_solve_strawberry_problem_l4043_404342


namespace NUMINAMATH_CALUDE_bruce_purchase_cost_l4043_404315

/-- The total cost of Bruce's purchase of grapes and mangoes -/
def total_cost (grape_quantity : ℕ) (grape_price : ℕ) (mango_quantity : ℕ) (mango_price : ℕ) : ℕ :=
  grape_quantity * grape_price + mango_quantity * mango_price

/-- Theorem stating the total cost of Bruce's purchase -/
theorem bruce_purchase_cost :
  total_cost 8 70 11 55 = 1165 := by
  sorry

#eval total_cost 8 70 11 55

end NUMINAMATH_CALUDE_bruce_purchase_cost_l4043_404315


namespace NUMINAMATH_CALUDE_max_days_to_eat_candies_l4043_404382

/-- The total number of candies Vasya received -/
def total_candies : ℕ := 777

/-- The function that calculates the total number of candies eaten over n days,
    where a is the number of candies eaten on the first day -/
def candies_eaten (n a : ℕ) : ℕ := n * (2 * a + n - 1) / 2

/-- The proposition that states 37 is the maximum number of days
    in which Vasya can eat all the candies -/
theorem max_days_to_eat_candies :
  ∃ (a : ℕ), candies_eaten 37 a = total_candies ∧
  ∀ (n : ℕ), n > 37 → ∀ (b : ℕ), candies_eaten n b ≠ total_candies :=
sorry

end NUMINAMATH_CALUDE_max_days_to_eat_candies_l4043_404382


namespace NUMINAMATH_CALUDE_smallest_m_for_integral_solutions_l4043_404345

theorem smallest_m_for_integral_solutions : 
  ∃ (m : ℕ), m > 0 ∧ 
  (∃ (x : ℤ), 12 * x^2 - m * x + 360 = 0) ∧
  (∀ (k : ℕ), 0 < k ∧ k < m → ¬∃ (x : ℤ), 12 * x^2 - k * x + 360 = 0) ∧
  m = 132 := by
sorry

end NUMINAMATH_CALUDE_smallest_m_for_integral_solutions_l4043_404345


namespace NUMINAMATH_CALUDE_sarahs_bowling_score_l4043_404355

theorem sarahs_bowling_score :
  ∀ (sarah_score greg_score : ℕ),
    sarah_score = greg_score + 50 →
    (sarah_score + greg_score) / 2 = 110 →
    sarah_score = 135 :=
by
  sorry

end NUMINAMATH_CALUDE_sarahs_bowling_score_l4043_404355


namespace NUMINAMATH_CALUDE_blue_sequin_rows_l4043_404369

/-- The number of sequins in each row of blue sequins -/
def blue_sequins_per_row : ℕ := 8

/-- The number of rows of purple sequins -/
def purple_rows : ℕ := 5

/-- The number of sequins in each row of purple sequins -/
def purple_sequins_per_row : ℕ := 12

/-- The number of rows of green sequins -/
def green_rows : ℕ := 9

/-- The number of sequins in each row of green sequins -/
def green_sequins_per_row : ℕ := 6

/-- The total number of sequins -/
def total_sequins : ℕ := 162

/-- Theorem: The number of rows of blue sequins is 6 -/
theorem blue_sequin_rows : 
  (total_sequins - (purple_rows * purple_sequins_per_row + green_rows * green_sequins_per_row)) / blue_sequins_per_row = 6 := by
  sorry

end NUMINAMATH_CALUDE_blue_sequin_rows_l4043_404369


namespace NUMINAMATH_CALUDE_problem_solution_l4043_404309

theorem problem_solution (a b c : ℝ) : 
  (-(a) = -2) → 
  (1 / b = -3/2) → 
  (abs c = 2) → 
  a + b + c^2 = 16/3 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l4043_404309


namespace NUMINAMATH_CALUDE_expression_equals_one_eighth_l4043_404306

theorem expression_equals_one_eighth :
  let a := 404445
  let b := 202222
  let c := 202223
  let d := 202224
  let e := 12639
  (a^2 / (b * c * d) - c / (b * d) - b / (c * d)) * e = 1/8 := by sorry

end NUMINAMATH_CALUDE_expression_equals_one_eighth_l4043_404306


namespace NUMINAMATH_CALUDE_y_not_less_than_four_by_at_least_one_l4043_404322

theorem y_not_less_than_four_by_at_least_one (y : ℝ) :
  (y ≥ 5) ↔ (y - 4 ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_y_not_less_than_four_by_at_least_one_l4043_404322


namespace NUMINAMATH_CALUDE_coefficient_x3y3_l4043_404354

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the expansion of (2x-y)^5
def expansion_term (r : ℕ) : ℤ := 
  (2^(5-r)) * ((-1)^r : ℤ) * (binomial 5 r)

-- Define the coefficient of x^3y^3 in (x+y)(2x-y)^5
def coefficient : ℤ := 
  expansion_term 3 + 2 * expansion_term 2

-- Theorem statement
theorem coefficient_x3y3 : coefficient = 40 := by sorry

end NUMINAMATH_CALUDE_coefficient_x3y3_l4043_404354


namespace NUMINAMATH_CALUDE_triangle_area_proof_l4043_404337

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (ω * x) * Real.cos (ω * x) - Real.sin (ω * x) ^ 2 + 1

theorem triangle_area_proof (ω : ℝ) (A B C : ℝ) (b : ℝ) :
  ω > 0 →
  (∀ x : ℝ, f ω (x + π / (2 * ω)) = f ω x) →
  b = 2 →
  f ω A = 1 →
  2 * Real.sin A = Real.sqrt 3 * Real.sin C →
  ∃ (a c : ℝ), a * b * Real.sin C / 2 = 2 * Real.sqrt 3 := by
  sorry

#check triangle_area_proof

end NUMINAMATH_CALUDE_triangle_area_proof_l4043_404337


namespace NUMINAMATH_CALUDE_no_intersection_l4043_404312

/-- Definition of a parabola -/
def is_parabola (x y : ℝ) : Prop := y^2 = 4*x

/-- Definition of a point inside the parabola -/
def is_inside_parabola (x₀ y₀ : ℝ) : Prop := y₀^2 < 4*x₀

/-- Definition of the line -/
def line_equation (x₀ y₀ x y : ℝ) : Prop := y₀*y = 2*(x + x₀)

/-- Theorem stating that a line passing through a point inside the parabola has no intersection with the parabola -/
theorem no_intersection (x₀ y₀ : ℝ) :
  is_inside_parabola x₀ y₀ →
  ∀ x y : ℝ, is_parabola x y ∧ line_equation x₀ y₀ x y → False :=
sorry

end NUMINAMATH_CALUDE_no_intersection_l4043_404312


namespace NUMINAMATH_CALUDE_exam_score_calculation_l4043_404321

theorem exam_score_calculation (total_questions : ℕ) (correct_answers : ℕ) 
  (marks_per_correct : ℕ) (marks_lost_per_wrong : ℕ) :
  total_questions = 50 →
  correct_answers = 36 →
  marks_per_correct = 4 →
  marks_lost_per_wrong = 1 →
  (correct_answers * marks_per_correct) - 
  ((total_questions - correct_answers) * marks_lost_per_wrong) = 130 := by
  sorry

end NUMINAMATH_CALUDE_exam_score_calculation_l4043_404321


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l4043_404374

/-- An infinite geometric sequence where any term is equal to the sum of all terms following it -/
def InfiniteGeometricSequence (a : ℝ) (r : ℝ) : Prop :=
  ∀ n : ℕ, a * r^n = (a * r^(n+1)) / (1 - r)

/-- Theorem: The common ratio of such a sequence is 1/2 -/
theorem geometric_sequence_common_ratio 
  (a : ℝ) (r : ℝ) (h : a ≠ 0) (seq : InfiniteGeometricSequence a r) : 
  r = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l4043_404374


namespace NUMINAMATH_CALUDE_blakes_initial_money_l4043_404318

/-- Blake's grocery shopping problem -/
theorem blakes_initial_money (orange_cost apples_cost mangoes_cost change : ℕ) 
  (h1 : orange_cost = 40)
  (h2 : apples_cost = 50)
  (h3 : mangoes_cost = 60)
  (h4 : change = 150) : 
  orange_cost + apples_cost + mangoes_cost + change = 300 := by
  sorry

#check blakes_initial_money

end NUMINAMATH_CALUDE_blakes_initial_money_l4043_404318


namespace NUMINAMATH_CALUDE_updated_average_weight_average_weight_proof_l4043_404302

theorem updated_average_weight (initial_avg : ℝ) (second_avg : ℝ) (third_avg : ℝ) 
  (correction1 : ℝ) (correction2 : ℝ) (correction3 : ℝ) : ℝ :=
  let initial_total := initial_avg * 5
  let second_total := second_avg * 9
  let third_total := third_avg * 12
  let corrected_total := third_total + correction1 + correction2 + correction3
  corrected_total / 12

theorem average_weight_proof :
  updated_average_weight 60 63 64 5 5 5 = 64.4167 := by
  sorry

end NUMINAMATH_CALUDE_updated_average_weight_average_weight_proof_l4043_404302


namespace NUMINAMATH_CALUDE_pet_ownership_l4043_404366

theorem pet_ownership (S D C B : Finset ℕ) (h1 : S.card = 50)
  (h2 : ∀ s ∈ S, s ∈ D ∨ s ∈ C ∨ s ∈ B)
  (h3 : D.card = 30) (h4 : C.card = 35) (h5 : B.card = 10)
  (h6 : (D ∩ C ∩ B).card = 5) :
  ((D ∩ C) \ B).card = 25 := by
  sorry

end NUMINAMATH_CALUDE_pet_ownership_l4043_404366


namespace NUMINAMATH_CALUDE_min_value_quadratic_l4043_404390

theorem min_value_quadratic (x : ℝ) : 
  4 * x^2 + 8 * x + 16 ≥ 12 ∧ ∃ y : ℝ, 4 * y^2 + 8 * y + 16 = 12 := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l4043_404390


namespace NUMINAMATH_CALUDE_circle_C_distance_range_l4043_404361

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  (x - 4)^2 + (y - 2)^2 = 25

-- Define the points A and B
def point_A : ℝ × ℝ := (1, 1)
def point_B : ℝ × ℝ := (7, 4)

-- Define the function for |PA|^2 + |PB|^2
def sum_of_squared_distances (P : ℝ × ℝ) : ℝ :=
  (P.1 - point_A.1)^2 + (P.2 - point_A.2)^2 +
  (P.1 - point_B.1)^2 + (P.2 - point_B.2)^2

-- Theorem statement
theorem circle_C_distance_range :
  ∀ P : ℝ × ℝ, circle_C P.1 P.2 →
  103 ≤ sum_of_squared_distances P ∧ sum_of_squared_distances P ≤ 123 :=
sorry

end NUMINAMATH_CALUDE_circle_C_distance_range_l4043_404361


namespace NUMINAMATH_CALUDE_light_could_be_green_l4043_404350

/-- Represents the state of a traffic light -/
inductive TrafficLightState
| Red
| Green
| Yellow

/-- Represents a traffic light with its cycle durations -/
structure TrafficLight where
  total_cycle : ℕ
  red_duration : ℕ
  green_duration : ℕ
  yellow_duration : ℕ
  cycle_valid : total_cycle = red_duration + green_duration + yellow_duration

/-- Defines the specific traffic light from the problem -/
def intersection_light : TrafficLight :=
  { total_cycle := 60
  , red_duration := 30
  , green_duration := 25
  , yellow_duration := 5
  , cycle_valid := by rfl }

/-- Theorem stating that the traffic light could be green at any random observation -/
theorem light_could_be_green (t : ℕ) : 
  ∃ (s : TrafficLightState), s = TrafficLightState.Green :=
sorry

end NUMINAMATH_CALUDE_light_could_be_green_l4043_404350


namespace NUMINAMATH_CALUDE_marble_problem_l4043_404388

theorem marble_problem (total : ℕ) (red : ℕ) (prob_red_or_white : ℚ) :
  total = 30 →
  red = 9 →
  prob_red_or_white = 5 / 6 →
  ∃ (blue white : ℕ), blue + red + white = total ∧ 
                       (red + white : ℚ) / total = prob_red_or_white ∧
                       blue = 5 := by
  sorry

end NUMINAMATH_CALUDE_marble_problem_l4043_404388


namespace NUMINAMATH_CALUDE_square_area_is_25_l4043_404392

/-- A square in the coordinate plane with given vertex coordinates -/
structure Square where
  x₁ : ℝ
  x₂ : ℝ

/-- The area of the square -/
def square_area (s : Square) : ℝ :=
  (5 : ℝ) ^ 2

/-- Theorem stating that the area of the square is 25 -/
theorem square_area_is_25 (s : Square) : square_area s = 25 := by
  sorry


end NUMINAMATH_CALUDE_square_area_is_25_l4043_404392


namespace NUMINAMATH_CALUDE_parabolas_intersection_l4043_404332

def parabola1 (x : ℝ) : ℝ := 2 * x^2 + 5 * x + 1
def parabola2 (x : ℝ) : ℝ := -x^2 + 4 * x + 6

theorem parabolas_intersection :
  ∃ (y1 y2 : ℝ),
    (∀ x : ℝ, parabola1 x = parabola2 x ↔ x = (-1 + Real.sqrt 61) / 6 ∨ x = (-1 - Real.sqrt 61) / 6) ∧
    parabola1 ((-1 + Real.sqrt 61) / 6) = y1 ∧
    parabola1 ((-1 - Real.sqrt 61) / 6) = y2 :=
by sorry

end NUMINAMATH_CALUDE_parabolas_intersection_l4043_404332


namespace NUMINAMATH_CALUDE_billy_bumper_car_rides_l4043_404316

/-- Calculates the number of bumper car rides given the number of ferris wheel rides,
    the cost per ride, and the total number of tickets used. -/
def bumper_car_rides (ferris_wheel_rides : ℕ) (cost_per_ride : ℕ) (total_tickets : ℕ) : ℕ :=
  (total_tickets - ferris_wheel_rides * cost_per_ride) / cost_per_ride

theorem billy_bumper_car_rides :
  bumper_car_rides 7 5 50 = 3 := by
  sorry

end NUMINAMATH_CALUDE_billy_bumper_car_rides_l4043_404316


namespace NUMINAMATH_CALUDE_diamond_set_eq_three_lines_l4043_404334

/-- Define the ⋄ operation -/
def diamond (a b : ℝ) : ℝ := a^2 * b - a * b^2

/-- The set of points (x, y) where x ⋄ y = y ⋄ x -/
def diamond_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | diamond p.1 p.2 = diamond p.2 p.1}

/-- The union of three lines: x = 0, y = 0, and x = y -/
def three_lines : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = 0 ∨ p.2 = 0 ∨ p.1 = p.2}

theorem diamond_set_eq_three_lines : diamond_set = three_lines := by
  sorry

end NUMINAMATH_CALUDE_diamond_set_eq_three_lines_l4043_404334


namespace NUMINAMATH_CALUDE_digital_city_activities_l4043_404308

-- Define the concept of a digital city
structure DigitalCity where
  is_part_of_digital_earth : Bool

-- Define possible activities in a digital city
inductive DigitalCityActivity
  | DistanceEducation
  | OnlineShopping
  | OnlineMedicalAdvice

-- Define a function that checks if an activity is enabled in a digital city
def is_enabled (city : DigitalCity) (activity : DigitalCityActivity) : Prop :=
  city.is_part_of_digital_earth

-- Theorem stating that digital cities enable specific activities
theorem digital_city_activities (city : DigitalCity) 
  (h : city.is_part_of_digital_earth = true) : 
  (is_enabled city DigitalCityActivity.DistanceEducation) ∧
  (is_enabled city DigitalCityActivity.OnlineShopping) ∧
  (is_enabled city DigitalCityActivity.OnlineMedicalAdvice) :=
by
  sorry


end NUMINAMATH_CALUDE_digital_city_activities_l4043_404308


namespace NUMINAMATH_CALUDE_quartic_polynomial_value_l4043_404362

def is_monic_quartic (q : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℝ, ∀ x, q x = x^4 + a*x^3 + b*x^2 + c*x + d

theorem quartic_polynomial_value (q : ℝ → ℝ) :
  is_monic_quartic q →
  q 1 = 3 →
  q 2 = 6 →
  q 3 = 11 →
  q 4 = 18 →
  q 5 = 51 := by
  sorry

end NUMINAMATH_CALUDE_quartic_polynomial_value_l4043_404362


namespace NUMINAMATH_CALUDE_inequality_holds_iff_one_l4043_404371

def is_valid (x : ℕ) : Prop := x > 0 ∧ x < 100

theorem inequality_holds_iff_one (x : ℕ) (h : is_valid x) :
  (2^x : ℚ) / x.factorial > x^2 ↔ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_one_l4043_404371


namespace NUMINAMATH_CALUDE_largest_n_less_than_2023_l4043_404385

-- Define the sequence a_n
def a (n : ℕ) : ℕ := 2 * n

-- Define the sequence b_n
def b (n : ℕ) : ℕ := 2^n

-- Define the sum of the first n terms of a_n
def S (n : ℕ) : ℕ := n^2 + n

-- Define T_n
def T (n : ℕ) : ℕ := (n - 1) * 2^(n + 2) + 4

theorem largest_n_less_than_2023 :
  (∀ n : ℕ, S n = n^2 + n) →
  (∀ n : ℕ, b n = 2^n) →
  (∀ n : ℕ, T n = (n - 1) * 2^(n + 2) + 4) →
  (∃ m : ℕ, m = 6 ∧ T m < 2023 ∧ ∀ k > m, T k ≥ 2023) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_less_than_2023_l4043_404385


namespace NUMINAMATH_CALUDE_video_game_earnings_l4043_404317

def total_games : ℕ := 15
def non_working_games : ℕ := 9
def price_per_game : ℕ := 5

theorem video_game_earnings : 
  (total_games - non_working_games) * price_per_game = 30 := by
  sorry

end NUMINAMATH_CALUDE_video_game_earnings_l4043_404317


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l4043_404330

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) := 
  ∃ (r : ℝ), ∀ n, a (n + 1) = r * a n

theorem geometric_sequence_property (a : ℕ → ℝ) :
  geometric_sequence a →
  a 1 * a 5 * a 7 = -3 * Real.sqrt 3 →
  a 2 * a 8 = 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l4043_404330


namespace NUMINAMATH_CALUDE_R_value_for_S_12_l4043_404387

theorem R_value_for_S_12 (g : ℝ) (R S : ℝ → ℝ) :
  (∀ x, R x = g * S x - 3) →
  R 5 = 17 →
  S 12 = 12 →
  R 12 = 45 := by
sorry

end NUMINAMATH_CALUDE_R_value_for_S_12_l4043_404387


namespace NUMINAMATH_CALUDE_problem_solution_l4043_404373

-- Define the function f
def f (a b x : ℝ) := |x - a| - |x + b|

-- Define the function g
def g (a b x : ℝ) := -x^2 - a*x - b

theorem problem_solution (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) 
  (hmax : ∀ x, f a b x ≤ 3) 
  (hmax_exists : ∃ x, f a b x = 3) 
  (hg_less_f : ∀ x ≥ a, g a b x < f a b x) : 
  (a + b = 3) ∧ (1/2 < a ∧ a < 3) := by
sorry


end NUMINAMATH_CALUDE_problem_solution_l4043_404373


namespace NUMINAMATH_CALUDE_prism_with_five_faces_has_nine_edges_l4043_404395

/-- A prism is a three-dimensional shape with two identical ends (bases) and flat sides. -/
structure Prism where
  faces : ℕ

/-- The number of edges in a prism given its number of faces. -/
def num_edges (p : Prism) : ℕ :=
  if p.faces = 5 then 9 else 0  -- We only define it for the case of 5 faces

theorem prism_with_five_faces_has_nine_edges (p : Prism) (h : p.faces = 5) : 
  num_edges p = 9 := by
  sorry

#check prism_with_five_faces_has_nine_edges

end NUMINAMATH_CALUDE_prism_with_five_faces_has_nine_edges_l4043_404395


namespace NUMINAMATH_CALUDE_highest_divisible_digit_l4043_404380

theorem highest_divisible_digit : ∃ (a : ℕ), a ≤ 9 ∧ 
  (365 * 10 + a) * 100 + 16 % 8 = 0 ∧ 
  ∀ (b : ℕ), b ≤ 9 → b > a → (365 * 10 + b) * 100 + 16 % 8 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_highest_divisible_digit_l4043_404380


namespace NUMINAMATH_CALUDE_central_angle_of_specific_sector_l4043_404377

/-- A circular sector with given circumference and area -/
structure CircularSector where
  circumference : ℝ
  area : ℝ

/-- The possible central angles of a circular sector -/
def central_angles (s : CircularSector) : Set ℝ :=
  {θ : ℝ | ∃ r : ℝ, r > 0 ∧ 2 * r + r * θ = s.circumference ∧ 1/2 * r^2 * θ = s.area}

/-- Theorem: The central angle of a sector with circumference 6 and area 2 is either 1 or 4 -/
theorem central_angle_of_specific_sector :
  let s : CircularSector := ⟨6, 2⟩
  central_angles s = {1, 4} := by sorry

end NUMINAMATH_CALUDE_central_angle_of_specific_sector_l4043_404377


namespace NUMINAMATH_CALUDE_cosine_value_from_tangent_half_l4043_404336

theorem cosine_value_from_tangent_half (α : Real) :
  (1 - Real.cos α) / Real.sin α = 3 → Real.cos α = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_cosine_value_from_tangent_half_l4043_404336


namespace NUMINAMATH_CALUDE_trains_meet_at_360km_l4043_404311

/-- Represents a train with its departure time and speed -/
structure Train where
  departureTime : ℕ  -- Departure time in hours after midnight
  speed : ℕ         -- Speed in km/h
  deriving Repr

/-- Calculates the meeting point of three trains -/
def meetingPoint (trainA trainB trainC : Train) : ℕ :=
  let t : ℕ := 18  -- 6 p.m. in 24-hour format
  let distanceA : ℕ := trainA.speed * (t - trainA.departureTime) 
  let distanceB : ℕ := trainB.speed * (t - trainB.departureTime)
  let timeAfterC : ℕ := (distanceB - distanceA) / (trainA.speed - trainB.speed)
  trainC.speed * timeAfterC

theorem trains_meet_at_360km :
  let trainA : Train := { departureTime := 9, speed := 30 }
  let trainB : Train := { departureTime := 15, speed := 40 }
  let trainC : Train := { departureTime := 18, speed := 60 }
  meetingPoint trainA trainB trainC = 360 := by
  sorry

#eval meetingPoint { departureTime := 9, speed := 30 } { departureTime := 15, speed := 40 } { departureTime := 18, speed := 60 }

end NUMINAMATH_CALUDE_trains_meet_at_360km_l4043_404311


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l4043_404399

theorem sqrt_equation_solution (x : ℝ) : 
  Real.sqrt (4 * x + 9) = 11 → x = 28 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l4043_404399


namespace NUMINAMATH_CALUDE_range_of_S_l4043_404398

theorem range_of_S (a b : ℝ) (h : ∀ x : ℝ, x ∈ Set.Icc 0 1 → |a * x + b| ≤ 1) :
  -2 ≤ (a + 1) * (b + 1) ∧ (a + 1) * (b + 1) ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_S_l4043_404398


namespace NUMINAMATH_CALUDE_words_with_vowels_count_l4043_404331

def alphabet : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F'}
def vowels : Finset Char := {'A', 'E'}
def consonants : Finset Char := alphabet \ vowels
def word_length : Nat := 5

def total_words : Nat := alphabet.card ^ word_length
def words_without_vowels : Nat := consonants.card ^ word_length

theorem words_with_vowels_count :
  total_words - words_without_vowels = 6752 :=
sorry

end NUMINAMATH_CALUDE_words_with_vowels_count_l4043_404331


namespace NUMINAMATH_CALUDE_rectangular_prism_surface_area_l4043_404352

/-- The surface area of a rectangular prism with given dimensions -/
def surface_area (length width height : ℝ) : ℝ :=
  2 * (length * width + width * height + height * length)

/-- Theorem: The surface area of a rectangular prism with length 5, width 4, and height 3 is 94 -/
theorem rectangular_prism_surface_area :
  surface_area 5 4 3 = 94 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_surface_area_l4043_404352


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l4043_404307

theorem arithmetic_calculation : 5 * (9 / 3) + 7 * 4 - 36 / 4 = 34 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l4043_404307
