import Mathlib

namespace NUMINAMATH_CALUDE_typist_margin_width_l1149_114927

/-- Proves that for a 20x30 cm sheet with 3 cm margins on top and bottom,
    if 64% is used for typing, the side margins are 2 cm wide. -/
theorem typist_margin_width (x : ℝ) : 
  x > 0 →                             -- side margin is positive
  x < 10 →                            -- side margin is less than half the sheet width
  (20 - 2*x) * 24 = 0.64 * 600 →      -- 64% of sheet is used for typing
  x = 2 := by
sorry

end NUMINAMATH_CALUDE_typist_margin_width_l1149_114927


namespace NUMINAMATH_CALUDE_apple_count_difference_l1149_114988

theorem apple_count_difference (initial_green : ℕ) (delivered_green : ℕ) (final_difference : ℕ) : 
  initial_green = 32 →
  delivered_green = 340 →
  initial_green + delivered_green = initial_green + final_difference + 140 →
  ∃ (initial_red : ℕ), initial_red - initial_green = 200 :=
by sorry

end NUMINAMATH_CALUDE_apple_count_difference_l1149_114988


namespace NUMINAMATH_CALUDE_johns_score_increase_l1149_114947

/-- Given John's four test scores, prove that the difference between
    the average of all four scores and the average of the first three scores is 0.92. -/
theorem johns_score_increase (score1 score2 score3 score4 : ℚ) 
    (h1 : score1 = 92)
    (h2 : score2 = 89)
    (h3 : score3 = 93)
    (h4 : score4 = 95) :
    (score1 + score2 + score3 + score4) / 4 - (score1 + score2 + score3) / 3 = 92 / 100 := by
  sorry

end NUMINAMATH_CALUDE_johns_score_increase_l1149_114947


namespace NUMINAMATH_CALUDE_min_columns_for_formation_l1149_114964

theorem min_columns_for_formation (n : ℕ) : n ≥ 141 → ∃ k : ℕ, 8 * n = 225 * k + 3 :=
sorry

end NUMINAMATH_CALUDE_min_columns_for_formation_l1149_114964


namespace NUMINAMATH_CALUDE_fridays_in_non_leap_year_starting_saturday_l1149_114918

/-- Represents a day of the week -/
inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

/-- Represents a year -/
structure Year where
  isLeapYear : Bool
  firstDayOfYear : DayOfWeek

/-- Counts the number of occurrences of a specific day in a year -/
def countDaysInYear (y : Year) (d : DayOfWeek) : Nat :=
  sorry

/-- Theorem: In a non-leap year where January 1st is a Saturday, there are 52 Fridays -/
theorem fridays_in_non_leap_year_starting_saturday (y : Year) 
  (h1 : y.isLeapYear = false) 
  (h2 : y.firstDayOfYear = DayOfWeek.Saturday) : 
  countDaysInYear y DayOfWeek.Friday = 52 :=
by sorry

end NUMINAMATH_CALUDE_fridays_in_non_leap_year_starting_saturday_l1149_114918


namespace NUMINAMATH_CALUDE_geometric_sequence_b6_l1149_114939

/-- A geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = r * b n

/-- The theorem statement -/
theorem geometric_sequence_b6 (b : ℕ → ℝ) :
  geometric_sequence b → b 3 * b 9 = 9 → b 6 = 3 ∨ b 6 = -3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_b6_l1149_114939


namespace NUMINAMATH_CALUDE_annes_speed_l1149_114996

theorem annes_speed (distance : ℝ) (time : ℝ) (speed : ℝ) 
  (h1 : distance = 6) 
  (h2 : time = 3) 
  (h3 : speed = distance / time) : 
  speed = 2 := by
sorry

end NUMINAMATH_CALUDE_annes_speed_l1149_114996


namespace NUMINAMATH_CALUDE_hyperbola_properties_l1149_114915

-- Define the hyperbola
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ x^2 / a^2 - y^2 / b^2 = 1

-- Define the point that lies on the hyperbola
def point_on_hyperbola (a b : ℝ) : Prop :=
  hyperbola a b (Real.sqrt 6) (Real.sqrt 3)

-- Define the focus of the hyperbola
def focus (a b : ℝ) : Prop :=
  hyperbola a b (-Real.sqrt 6) 0

-- Define the intersection line
def intersection_line (k : ℝ) (x y : ℝ) : Prop :=
  y = k * x + 2

-- Theorem statement
theorem hyperbola_properties :
  ∀ a b : ℝ,
  point_on_hyperbola a b →
  focus a b →
  (∀ x y : ℝ, hyperbola a b x y ↔ x^2 - y^2 = 3) ∧
  (∀ k : ℝ, (∃ x₁ x₂ y₁ y₂ : ℝ, 
    x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧
    hyperbola a b x₁ y₁ ∧ hyperbola a b x₂ y₂ ∧
    intersection_line k x₁ y₁ ∧ intersection_line k x₂ y₂)
    ↔ -Real.sqrt (21 / 9) < k ∧ k < -1) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l1149_114915


namespace NUMINAMATH_CALUDE_inequality_properties_l1149_114966

-- Define the inequality and its solution set
def inequality (a x : ℝ) : Prop := a * (x - 1) * (x - 3) + 2 > 0

def solution_set (x₁ x₂ : ℝ) : Set ℝ :=
  {x | x < x₁ ∨ x > x₂}

-- State the theorem
theorem inequality_properties
  (a x₁ x₂ : ℝ)
  (h_solution : ∀ x, inequality a x ↔ x ∈ solution_set x₁ x₂)
  (h_order : x₁ < x₂) :
  (x₁ + x₂ = 4) ∧
  (3 < x₁ * x₂ ∧ x₁ * x₂ < 4) ∧
  (∀ x, (3*a + 2) * x^2 - 4*a*x + a < 0 ↔ 1/x₂ < x ∧ x < 1/x₁) :=
by sorry

end NUMINAMATH_CALUDE_inequality_properties_l1149_114966


namespace NUMINAMATH_CALUDE_emily_sixth_score_l1149_114976

def emily_scores : List ℕ := [91, 94, 86, 88, 101]

theorem emily_sixth_score (target_mean : ℕ := 94) (sixth_score : ℕ := 104) :
  let all_scores := emily_scores ++ [sixth_score]
  (all_scores.sum / all_scores.length : ℚ) = target_mean := by
  sorry

end NUMINAMATH_CALUDE_emily_sixth_score_l1149_114976


namespace NUMINAMATH_CALUDE_odd_7x_plus_4_l1149_114982

theorem odd_7x_plus_4 (x : ℤ) (h : Even (3 * x + 1)) : Odd (7 * x + 4) := by
  sorry

end NUMINAMATH_CALUDE_odd_7x_plus_4_l1149_114982


namespace NUMINAMATH_CALUDE_no_three_digit_number_satisfies_conditions_l1149_114952

/-- Function to check if digits are different and in ascending order -/
def digits_ascending_different (n : ℕ) : Prop := sorry

/-- Theorem stating that no three-digit number satisfies the given conditions -/
theorem no_three_digit_number_satisfies_conditions :
  ¬ ∃ n : ℕ, 
    100 ≤ n ∧ n ≤ 999 ∧ 
    digits_ascending_different n ∧
    digits_ascending_different (n^2) ∧
    digits_ascending_different (n^3) := by
  sorry

end NUMINAMATH_CALUDE_no_three_digit_number_satisfies_conditions_l1149_114952


namespace NUMINAMATH_CALUDE_bike_cost_theorem_l1149_114903

def apple_price : ℚ := 1.25
def apples_sold : ℕ := 20
def repair_ratio : ℚ := 1/4
def remaining_ratio : ℚ := 1/5

def total_earnings : ℚ := apple_price * apples_sold

theorem bike_cost_theorem (h1 : total_earnings = apple_price * apples_sold)
                          (h2 : repair_ratio * (total_earnings * (1 - remaining_ratio)) = total_earnings * (1 - remaining_ratio)) :
  (total_earnings * (1 - remaining_ratio)) / repair_ratio = 80 := by sorry

end NUMINAMATH_CALUDE_bike_cost_theorem_l1149_114903


namespace NUMINAMATH_CALUDE_card_game_result_l1149_114948

/-- Represents the money distribution in a card game --/
structure MoneyDistribution where
  aldo : ℚ
  bernardo : ℚ
  carlos : ℚ

/-- The card game scenario --/
def CardGame : Type :=
  { game : MoneyDistribution × MoneyDistribution // 
    (game.1.aldo : ℚ) / (game.1.bernardo : ℚ) = 7/6 ∧
    (game.1.bernardo : ℚ) / (game.1.carlos : ℚ) = 6/5 ∧
    (game.2.aldo : ℚ) / (game.2.bernardo : ℚ) = 6/5 ∧
    (game.2.bernardo : ℚ) / (game.2.carlos : ℚ) = 5/4 ∧
    (game.2.aldo - game.1.aldo : ℚ) = 1200 ∨
    (game.2.bernardo - game.1.bernardo : ℚ) = 1200 ∨
    (game.2.carlos - game.1.carlos : ℚ) = 1200 }

/-- The theorem to be proved --/
theorem card_game_result (game : CardGame) :
  game.val.2.aldo = 43200 ∧
  game.val.2.bernardo = 36000 ∧
  game.val.2.carlos = 28800 := by
  sorry


end NUMINAMATH_CALUDE_card_game_result_l1149_114948


namespace NUMINAMATH_CALUDE_credit_card_problem_l1149_114965

/-- Calculates the amount added to a credit card in the second month given the initial balance,
    interest rate, and final balance after two months. -/
def amount_added (initial_balance : ℚ) (interest_rate : ℚ) (final_balance : ℚ) : ℚ :=
  let first_month_balance := initial_balance * (1 + interest_rate)
  let x := (final_balance - first_month_balance * (1 + interest_rate)) / (1 + interest_rate)
  x

theorem credit_card_problem :
  let initial_balance : ℚ := 50
  let interest_rate : ℚ := 1/5
  let final_balance : ℚ := 96
  amount_added initial_balance interest_rate final_balance = 20 := by
  sorry

end NUMINAMATH_CALUDE_credit_card_problem_l1149_114965


namespace NUMINAMATH_CALUDE_negation_equivalence_l1149_114960

theorem negation_equivalence :
  (¬ (∃ x : ℝ, x > 0 ∧ x^2 - 2*x - 3 > 0)) ↔ 
  (∀ x : ℝ, x > 0 → x^2 - 2*x - 3 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1149_114960


namespace NUMINAMATH_CALUDE_union_of_sets_l1149_114949

theorem union_of_sets : 
  let A : Set ℕ := {1, 2, 4}
  let B : Set ℕ := {2, 4}
  A ∪ B = {1, 2, 4} := by
sorry

end NUMINAMATH_CALUDE_union_of_sets_l1149_114949


namespace NUMINAMATH_CALUDE_quadratic_inequalities_intersection_l1149_114919

def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def B : Set ℝ := {x | x^2 + x - 6 < 0}

theorem quadratic_inequalities_intersection (a b : ℝ) :
  ({x : ℝ | x^2 + a*x + b < 0} = A ∩ B) →
  a + b = -3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequalities_intersection_l1149_114919


namespace NUMINAMATH_CALUDE_investigator_strategy_equivalence_l1149_114908

/-- Represents the investigator's questioning strategy -/
structure InvestigatorStrategy where
  num_questions : ℕ
  max_lie : ℕ

/-- Defines the original strategy with all truthful answers -/
def original_strategy : InvestigatorStrategy :=
  { num_questions := 91
  , max_lie := 0 }

/-- Defines the new strategy allowing for one possible lie -/
def new_strategy : InvestigatorStrategy :=
  { num_questions := 105
  , max_lie := 1 }

/-- Represents the information obtained from questioning -/
def Information : Type := Unit

/-- Function to obtain information given a strategy -/
def obtain_information (strategy : InvestigatorStrategy) : Information := sorry

theorem investigator_strategy_equivalence :
  obtain_information original_strategy = obtain_information new_strategy :=
by sorry

end NUMINAMATH_CALUDE_investigator_strategy_equivalence_l1149_114908


namespace NUMINAMATH_CALUDE_solve_equation_l1149_114992

-- Define the function f
def f (x : ℝ) : ℝ := abs (x - 1) - abs (x + 1)

-- State the theorem
theorem solve_equation (a : ℝ) : f (f a) = f 9 + 1 → a = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1149_114992


namespace NUMINAMATH_CALUDE_max_intersected_edges_l1149_114900

/-- A regular p-gonal prism -/
structure RegularPrism (p : ℕ) :=
  (p_pos : p > 0)

/-- A plane that does not pass through the vertices of the prism -/
structure NonVertexPlane (p : ℕ) (prism : RegularPrism p) :=

/-- The number of edges of a regular p-gonal prism intersected by a plane -/
def intersected_edges (p : ℕ) (prism : RegularPrism p) (plane : NonVertexPlane p prism) : ℕ :=
  sorry

/-- The maximum number of edges that can be intersected is 3p -/
theorem max_intersected_edges (p : ℕ) (prism : RegularPrism p) :
  ∃ (plane : NonVertexPlane p prism), intersected_edges p prism plane = 3 * p ∧
  ∀ (other_plane : NonVertexPlane p prism), intersected_edges p prism other_plane ≤ 3 * p :=
sorry

end NUMINAMATH_CALUDE_max_intersected_edges_l1149_114900


namespace NUMINAMATH_CALUDE_arithmetic_sequence_2015th_term_l1149_114922

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) + a n = 4 * n - 58

theorem arithmetic_sequence_2015th_term (a : ℕ → ℤ) 
  (h : arithmetic_sequence a) : a 2015 = 4000 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_2015th_term_l1149_114922


namespace NUMINAMATH_CALUDE_midpoint_coordinate_product_l1149_114942

/-- Given a line segment with endpoints (4, -7) and (-8, 9), 
    the product of the coordinates of its midpoint is -2. -/
theorem midpoint_coordinate_product : 
  let x1 : ℝ := 4
  let y1 : ℝ := -7
  let x2 : ℝ := -8
  let y2 : ℝ := 9
  let midpoint_x : ℝ := (x1 + x2) / 2
  let midpoint_y : ℝ := (y1 + y2) / 2
  midpoint_x * midpoint_y = -2 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_product_l1149_114942


namespace NUMINAMATH_CALUDE_parabola_equation_l1149_114910

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space of the form ax + by = c -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a parabola in general form ax^2 + bxy + cy^2 + dx + ey + f = 0 -/
structure Parabola where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ
  e : ℤ
  f : ℤ

/-- The greatest common divisor of the absolute values of all coefficients is 1 -/
def coefficientsAreCoprime (p : Parabola) : Prop :=
  Nat.gcd (Nat.gcd (Nat.gcd (Nat.gcd (Nat.gcd (Int.natAbs p.a) (Int.natAbs p.b)) (Int.natAbs p.c)) (Int.natAbs p.d)) (Int.natAbs p.e)) (Int.natAbs p.f) = 1

/-- The theorem stating the equation of the parabola -/
theorem parabola_equation (focus : Point) (directrix : Line) : 
  focus.x = 4 ∧ focus.y = 2 ∧ 
  directrix.a = 2 ∧ directrix.b = 5 ∧ directrix.c = 20 →
  ∃ (p : Parabola), 
    p.a = 25 ∧ p.b = -20 ∧ p.c = 4 ∧ p.d = -152 ∧ p.e = 84 ∧ p.f = -180 ∧
    p.a > 0 ∧
    coefficientsAreCoprime p :=
by sorry

end NUMINAMATH_CALUDE_parabola_equation_l1149_114910


namespace NUMINAMATH_CALUDE_ice_cream_flavors_l1149_114924

/-- The number of ways to distribute n indistinguishable items into k distinguishable categories -/
def distribute (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of new ice cream flavors -/
def new_flavors : ℕ := distribute 5 5

theorem ice_cream_flavors : new_flavors = 126 := by sorry

end NUMINAMATH_CALUDE_ice_cream_flavors_l1149_114924


namespace NUMINAMATH_CALUDE_expression_comparison_l1149_114990

theorem expression_comparison (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  ¬(∀ x y : ℝ, x > 0 → y > 0 → x ≠ y →
    ((x + 1/x) * (y + 1/y) > (Real.sqrt (x*y) + 1/Real.sqrt (x*y))^2 ∧
     (x + 1/x) * (y + 1/y) > ((x + y)/2 + 2/(x + y))^2) ∨
    ((Real.sqrt (x*y) + 1/Real.sqrt (x*y))^2 > (x + 1/x) * (y + 1/y) ∧
     (Real.sqrt (x*y) + 1/Real.sqrt (x*y))^2 > ((x + y)/2 + 2/(x + y))^2) ∨
    (((x + y)/2 + 2/(x + y))^2 > (x + 1/x) * (y + 1/y) ∧
     ((x + y)/2 + 2/(x + y))^2 > (Real.sqrt (x*y) + 1/Real.sqrt (x*y))^2)) :=
by sorry

end NUMINAMATH_CALUDE_expression_comparison_l1149_114990


namespace NUMINAMATH_CALUDE_fraction_sum_l1149_114926

theorem fraction_sum (m n : ℚ) (h : m / n = 3 / 7) : (m + n) / n = 10 / 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_l1149_114926


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1149_114957

theorem right_triangle_hypotenuse : 
  ∀ (a b c : ℝ), 
    a = 20 → 
    b = 21 → 
    c^2 = a^2 + b^2 → 
    c = 29 :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1149_114957


namespace NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l1149_114923

theorem complex_number_in_first_quadrant : ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ (Complex.mk a b) = (1 : ℂ) / (1 - Complex.I) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l1149_114923


namespace NUMINAMATH_CALUDE_min_value_f_plus_f_l1149_114999

/-- The function f(x) = -x³ + ax² - 4 -/
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + a*x^2 - 4

/-- The derivative of f(x) -/
def f' (a : ℝ) (x : ℝ) : ℝ := -3*x^2 + 2*a*x

theorem min_value_f_plus_f' (a : ℝ) :
  (f' a 1 = 0) →
  (∃ (m n : ℝ), m ∈ Set.Icc (-1 : ℝ) 1 ∧ n ∈ Set.Icc (-1 : ℝ) 1 ∧
    ∀ (m' n' : ℝ), m' ∈ Set.Icc (-1 : ℝ) 1 → n' ∈ Set.Icc (-1 : ℝ) 1 →
      f a m + f' a n ≤ f a m' + f' a n') →
  ∃ (m n : ℝ), m ∈ Set.Icc (-1 : ℝ) 1 ∧ n ∈ Set.Icc (-1 : ℝ) 1 ∧
    f a m + f' a n = -13 :=
by sorry

end NUMINAMATH_CALUDE_min_value_f_plus_f_l1149_114999


namespace NUMINAMATH_CALUDE_bicycle_price_problem_l1149_114929

theorem bicycle_price_problem (cp_a : ℝ) (sp_b sp_c : ℝ) : 
  sp_b = 1.5 * cp_a →
  sp_c = 1.25 * sp_b →
  sp_c = 225 →
  cp_a = 120 := by
sorry

end NUMINAMATH_CALUDE_bicycle_price_problem_l1149_114929


namespace NUMINAMATH_CALUDE_sine_is_periodic_l1149_114956

-- Define the properties
def IsTrigonometric (f : ℝ → ℝ) : Prop := sorry
def IsPeriodic (f : ℝ → ℝ) : Prop := sorry

-- Define the sine function
def sin : ℝ → ℝ := sorry

-- Theorem statement
theorem sine_is_periodic :
  (∀ f : ℝ → ℝ, IsTrigonometric f → IsPeriodic f) →
  IsTrigonometric sin →
  IsPeriodic sin := by sorry

end NUMINAMATH_CALUDE_sine_is_periodic_l1149_114956


namespace NUMINAMATH_CALUDE_harriett_us_dollars_l1149_114909

def quarter_value : ℚ := 0.25
def dime_value : ℚ := 0.10
def nickel_value : ℚ := 0.05
def penny_value : ℚ := 0.01
def half_dollar_value : ℚ := 0.50
def dollar_coin_value : ℚ := 1.00

def num_quarters : ℕ := 23
def num_dimes : ℕ := 15
def num_nickels : ℕ := 17
def num_pennies : ℕ := 29
def num_half_dollars : ℕ := 6
def num_dollar_coins : ℕ := 10

def total_us_dollars : ℚ := 
  num_quarters * quarter_value + 
  num_dimes * dime_value + 
  num_nickels * nickel_value + 
  num_pennies * penny_value + 
  num_half_dollars * half_dollar_value + 
  num_dollar_coins * dollar_coin_value

theorem harriett_us_dollars : total_us_dollars = 21.39 := by
  sorry

end NUMINAMATH_CALUDE_harriett_us_dollars_l1149_114909


namespace NUMINAMATH_CALUDE_probability_theorem_l1149_114984

def standard_dice : ℕ := 6

def roll_count : ℕ := 4

def probability_at_least_three_distinct_with_six : ℚ :=
  360 / (standard_dice ^ roll_count)

theorem probability_theorem :
  probability_at_least_three_distinct_with_six = 5 / 18 :=
by sorry

end NUMINAMATH_CALUDE_probability_theorem_l1149_114984


namespace NUMINAMATH_CALUDE_sin_double_angle_special_l1149_114944

/-- Given an angle θ with specific properties, prove that sin(2θ) = -√3/2 -/
theorem sin_double_angle_special (θ : Real) : 
  (∃ (x y : Real), x > 0 ∧ y = -Real.sqrt 3 * x ∧ 
    Real.cos θ = x / Real.sqrt (x^2 + y^2) ∧
    Real.sin θ = y / Real.sqrt (x^2 + y^2)) →
  Real.sin (2 * θ) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_double_angle_special_l1149_114944


namespace NUMINAMATH_CALUDE_celebration_attendees_l1149_114950

theorem celebration_attendees (men : ℕ) (women : ℕ) : 
  men = 15 →
  men * 4 = women * 3 →
  women = 20 :=
by sorry

end NUMINAMATH_CALUDE_celebration_attendees_l1149_114950


namespace NUMINAMATH_CALUDE_intersection_line_circle_l1149_114969

theorem intersection_line_circle (a : ℝ) : 
  (∃ A B : ℝ × ℝ, 
    (a * A.1 - A.2 + 3 = 0 ∧ (A.1 - 1)^2 + (A.2 - 2)^2 = 4) ∧
    (a * B.1 - B.2 + 3 = 0 ∧ (B.1 - 1)^2 + (B.2 - 2)^2 = 4) ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 12) →
  a = 0 := by
sorry

end NUMINAMATH_CALUDE_intersection_line_circle_l1149_114969


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l1149_114935

theorem imaginary_part_of_complex_fraction :
  Complex.im ((4 - 5 * Complex.I) / Complex.I) = -4 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l1149_114935


namespace NUMINAMATH_CALUDE_hyperbola_larger_y_focus_l1149_114946

def hyperbola_equation (x y : ℝ) : Prop :=
  (x - 5)^2 / 7^2 - (y - 10)^2 / 3^2 = 1

def is_focus (x y : ℝ) : Prop :=
  (x - 5)^2 + (y - 10)^2 = 58

def larger_y_focus (x y : ℝ) : Prop :=
  is_focus x y ∧ y > 10

theorem hyperbola_larger_y_focus :
  ∃ (x y : ℝ), hyperbola_equation x y ∧ larger_y_focus x y ∧ x = 5 ∧ y = 10 + Real.sqrt 58 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_larger_y_focus_l1149_114946


namespace NUMINAMATH_CALUDE_arcsin_cos_eq_x_div_3_solutions_l1149_114986

theorem arcsin_cos_eq_x_div_3_solutions (x : Real) :
  -3 * Real.pi / 2 ≤ x ∧ x ≤ 3 * Real.pi / 2 →
  (Real.arcsin (Real.cos x) = x / 3 ↔ (x = 3 * Real.pi / 10 ∨ x = 3 * Real.pi / 8)) :=
by sorry

end NUMINAMATH_CALUDE_arcsin_cos_eq_x_div_3_solutions_l1149_114986


namespace NUMINAMATH_CALUDE_circumcircle_radius_of_three_spheres_l1149_114954

/-- Given two spheres touching a plane at points B and C, with sum of radii 11 and distance between
    centers 5√17, and a third sphere of radius 8 at point A externally tangent to the other two,
    the radius of the circumcircle of triangle ABC is 2√19. -/
theorem circumcircle_radius_of_three_spheres (R1 R2 : ℝ) (d : ℝ) (R3 : ℝ) :
  R1 + R2 = 11 →
  d = 5 * Real.sqrt 17 →
  R3 = 8 →
  R1 + R2 + 2 * R3 = d →
  ∃ (R : ℝ), R = 2 * Real.sqrt 19 ∧ R = d / 2 :=
by sorry

end NUMINAMATH_CALUDE_circumcircle_radius_of_three_spheres_l1149_114954


namespace NUMINAMATH_CALUDE_emily_necklaces_l1149_114934

def beads_per_necklace : ℕ := 8
def total_beads : ℕ := 16

theorem emily_necklaces :
  total_beads / beads_per_necklace = 2 := by
  sorry

end NUMINAMATH_CALUDE_emily_necklaces_l1149_114934


namespace NUMINAMATH_CALUDE_greatest_prime_factor_of_341_l1149_114973

theorem greatest_prime_factor_of_341 :
  ∃ (p : ℕ), p.Prime ∧ p ∣ 341 ∧ p = 19 ∧ ∀ (q : ℕ), q.Prime → q ∣ 341 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_of_341_l1149_114973


namespace NUMINAMATH_CALUDE_adams_trivia_score_l1149_114998

/-- Adam's trivia game score calculation -/
theorem adams_trivia_score :
  ∀ (first_half second_half points_per_question : ℕ),
    first_half = 8 →
    second_half = 2 →
    points_per_question = 8 →
    (first_half + second_half) * points_per_question = 80 :=
by
  sorry

end NUMINAMATH_CALUDE_adams_trivia_score_l1149_114998


namespace NUMINAMATH_CALUDE_polynomial_uniqueness_l1149_114945

def is_valid_polynomial (P : ℝ → ℝ) (n : ℕ) : Prop :=
  (∀ k : ℕ, k ≤ n → P (2 * k) = 0) ∧
  (∀ k : ℕ, k < n → P (2 * k + 1) = 2) ∧
  (P (2 * n + 1) = -6)

theorem polynomial_uniqueness (P : ℝ → ℝ) (n : ℕ) :
  is_valid_polynomial P n →
  (∃ a b c : ℝ, ∀ x, P x = a * x^2 + b * x + c) →
  (n = 1 ∧ ∀ x, P x = -2 * x^2 + 4 * x) :=
sorry

end NUMINAMATH_CALUDE_polynomial_uniqueness_l1149_114945


namespace NUMINAMATH_CALUDE_expression_evaluation_l1149_114930

theorem expression_evaluation : 6^4 - 4 * 6^3 + 6^2 - 2 * 6 + 1 = 457 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1149_114930


namespace NUMINAMATH_CALUDE_bagel_count_is_three_l1149_114972

/-- Represents the number of items bought at each price point -/
structure PurchaseCount where
  sixtyCount : ℕ
  eightyCount : ℕ
  hundredCount : ℕ

/-- Calculates the total cost in cents for a given purchase count -/
def totalCost (p : PurchaseCount) : ℕ :=
  60 * p.sixtyCount + 80 * p.eightyCount + 100 * p.hundredCount

/-- Theorem stating that under the given conditions, the number of 80-cent items is 3 -/
theorem bagel_count_is_three :
  ∃ (p : PurchaseCount),
    p.sixtyCount + p.eightyCount + p.hundredCount = 5 ∧
    totalCost p = 400 ∧
    p.eightyCount = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_bagel_count_is_three_l1149_114972


namespace NUMINAMATH_CALUDE_quantity_cost_relation_l1149_114994

theorem quantity_cost_relation (Q : ℝ) (h1 : Q * 20 = 1) (h2 : 3.5 * Q * 28 = 1) :
  20 / 8 = 2.5 := by sorry

end NUMINAMATH_CALUDE_quantity_cost_relation_l1149_114994


namespace NUMINAMATH_CALUDE_right_triangle_third_side_l1149_114963

theorem right_triangle_third_side
  (m n : ℝ)
  (h1 : |m - 3| + Real.sqrt (n - 4) = 0)
  (h2 : m > 0 ∧ n > 0)
  (h3 : ∃ (a b c : ℝ), a^2 + b^2 = c^2 ∧ ((a = m ∧ b = n) ∨ (a = m ∧ c = n) ∨ (b = m ∧ c = n)))
  : ∃ (x : ℝ), (x = 5 ∨ x = Real.sqrt 7) ∧
    ∃ (a b c : ℝ), a^2 + b^2 = c^2 ∧ ((a = m ∧ b = n ∧ c = x) ∨ (a = m ∧ c = n ∧ b = x) ∨ (b = m ∧ c = n ∧ a = x)) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_third_side_l1149_114963


namespace NUMINAMATH_CALUDE_coefficient_x5_in_expansion_l1149_114997

theorem coefficient_x5_in_expansion : 
  (Finset.range 8).sum (fun k => (Nat.choose 7 k) * (2 ^ (7 - k)) * if k == 5 then 1 else 0) = 84 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x5_in_expansion_l1149_114997


namespace NUMINAMATH_CALUDE_purely_imaginary_condition_fourth_quadrant_condition_l1149_114901

def z (a : ℝ) : ℂ := Complex.mk (a^2 - 7*a + 6) (a^2 - 5*a - 6)

theorem purely_imaginary_condition (a : ℝ) : 
  (z a).re = 0 ∧ (z a).im ≠ 0 → a = 1 := by sorry

theorem fourth_quadrant_condition (a : ℝ) :
  (z a).re > 0 ∧ (z a).im < 0 → -1 < a ∧ a < 1 := by sorry

end NUMINAMATH_CALUDE_purely_imaginary_condition_fourth_quadrant_condition_l1149_114901


namespace NUMINAMATH_CALUDE_angle_inequality_l1149_114936

theorem angle_inequality : 
  let a := (2 * Real.tan (22.5 * π / 180)) / (1 - Real.tan (22.5 * π / 180) ^ 2)
  let b := 2 * Real.sin (13 * π / 180) * Real.cos (13 * π / 180)
  let c := Real.sqrt ((1 - Real.cos (50 * π / 180)) / 2)
  c < b ∧ b < a := by sorry

end NUMINAMATH_CALUDE_angle_inequality_l1149_114936


namespace NUMINAMATH_CALUDE_same_solutions_implies_a_equals_four_l1149_114937

theorem same_solutions_implies_a_equals_four :
  ∀ a : ℝ,
  (∀ x : ℝ, x^2 - a = 0 ↔ 3*x^4 - 48 = 0) →
  a = 4 := by
sorry

end NUMINAMATH_CALUDE_same_solutions_implies_a_equals_four_l1149_114937


namespace NUMINAMATH_CALUDE_sin_function_value_l1149_114978

/-- Given that the terminal side of angle φ passes through point P(3, -4),
    and the distance between two adjacent symmetry axes of the graph of
    the function f(x) = sin(ωx + φ) (ω > 0) is equal to π/2,
    prove that f(π/4) = 3/5 -/
theorem sin_function_value (φ ω : ℝ) (h1 : ω > 0) 
    (h2 : (3 : ℝ) / Real.sqrt (3^2 + 4^2) = Real.cos φ)
    (h3 : (-4 : ℝ) / Real.sqrt (3^2 + 4^2) = Real.sin φ)
    (h4 : π / (2 * ω) = π / 2) :
  Real.sin (ω * (π / 4) + φ) = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sin_function_value_l1149_114978


namespace NUMINAMATH_CALUDE_stamp_sale_value_l1149_114975

def total_stamps : ℕ := 75
def stamps_of_one_kind : ℕ := 40
def value_type1 : ℚ := 5 / 100
def value_type2 : ℚ := 8 / 100

theorem stamp_sale_value :
  ∃ (type1_count type2_count : ℕ),
    type1_count + type2_count = total_stamps ∧
    (type1_count = stamps_of_one_kind ∨ type2_count = stamps_of_one_kind) ∧
    type1_count * value_type1 + type2_count * value_type2 = 48 / 10 := by
  sorry

end NUMINAMATH_CALUDE_stamp_sale_value_l1149_114975


namespace NUMINAMATH_CALUDE_abs_m_minus_n_equals_five_l1149_114928

theorem abs_m_minus_n_equals_five (m n : ℝ) 
  (h1 : m * n = 6) 
  (h2 : m + n = 7) : 
  |m - n| = 5 := by
sorry

end NUMINAMATH_CALUDE_abs_m_minus_n_equals_five_l1149_114928


namespace NUMINAMATH_CALUDE_specific_quadrilateral_area_l1149_114938

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The area of a quadrilateral given its four vertices -/
def quadrilateralArea (a b c d : Point) : ℝ := sorry

/-- The theorem stating that the area of the specific quadrilateral is 4.5 -/
theorem specific_quadrilateral_area :
  let a : Point := ⟨0, 0⟩
  let b : Point := ⟨0, 2⟩
  let c : Point := ⟨3, 2⟩
  let d : Point := ⟨3, 3⟩
  quadrilateralArea a b c d = 4.5 := by sorry

end NUMINAMATH_CALUDE_specific_quadrilateral_area_l1149_114938


namespace NUMINAMATH_CALUDE_eccentricity_of_ellipse_l1149_114932

-- Define the complex polynomial
def polynomial (z : ℂ) : ℂ := (z - 2) * (z^2 + 3*z + 5) * (z^2 + 5*z + 8)

-- Define the set of solutions
def solutions : Set ℂ := {z : ℂ | polynomial z = 0}

-- Define the ellipse centered at the origin
def ellipse (a b : ℝ) : Set ℂ := {z : ℂ | (z.re^2 / a^2) + (z.im^2 / b^2) = 1}

-- Theorem statement
theorem eccentricity_of_ellipse :
  ∀ a b : ℝ, a > 0 → b > 0 →
  (∃ e : Set ℂ, e = ellipse a b ∧ solutions ⊆ e) →
  (a^2 - b^2) / a^2 = 5/16 :=
sorry

end NUMINAMATH_CALUDE_eccentricity_of_ellipse_l1149_114932


namespace NUMINAMATH_CALUDE_neg_p_sufficient_not_necessary_for_neg_q_l1149_114970

-- Define the propositions p and q
def p (x : ℝ) : Prop := abs x > 1
def q (x : ℝ) : Prop := x < -2

-- State the theorem
theorem neg_p_sufficient_not_necessary_for_neg_q :
  (∀ x, ¬(p x) → ¬(q x)) ∧ 
  (∃ x, ¬(p x) ∧ q x) :=
sorry

end NUMINAMATH_CALUDE_neg_p_sufficient_not_necessary_for_neg_q_l1149_114970


namespace NUMINAMATH_CALUDE_power_division_l1149_114913

theorem power_division (a : ℝ) : a^6 / a^3 = a^3 := by
  sorry

end NUMINAMATH_CALUDE_power_division_l1149_114913


namespace NUMINAMATH_CALUDE_cousins_ages_sum_l1149_114961

def is_single_digit (n : ℕ) : Prop := 0 < n ∧ n < 10

theorem cousins_ages_sum :
  ∀ (a b c d : ℕ),
    is_single_digit a ∧ is_single_digit b ∧ is_single_digit c ∧ is_single_digit d →
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
    ((a * b = 24 ∧ c * d = 30) ∨ (a * c = 24 ∧ b * d = 30) ∨ (a * d = 24 ∧ b * c = 30)) →
    a + b + c + d = 22 :=
by sorry

end NUMINAMATH_CALUDE_cousins_ages_sum_l1149_114961


namespace NUMINAMATH_CALUDE_admission_criteria_correct_l1149_114911

/-- Represents the admission score criteria for art students in a high school. -/
structure AdmissionCriteria where
  x : ℝ  -- Professional score
  y : ℝ  -- Total score of liberal arts
  z : ℝ  -- Physical education score

/-- Defines the correct admission criteria based on the given conditions. -/
def correct_criteria (c : AdmissionCriteria) : Prop :=
  c.x ≥ 95 ∧ c.y > 380 ∧ c.z > 45

/-- Theorem stating that the given inequalities correctly represent the admission criteria. -/
theorem admission_criteria_correct (c : AdmissionCriteria) :
  (c.x ≥ 95 ∧ c.y > 380 ∧ c.z > 45) ↔ correct_criteria c :=
by sorry

end NUMINAMATH_CALUDE_admission_criteria_correct_l1149_114911


namespace NUMINAMATH_CALUDE_simple_interest_time_period_l1149_114912

/-- Theorem: Simple Interest Time Period Calculation -/
theorem simple_interest_time_period 
  (P : ℝ) -- Principal amount
  (r : ℝ) -- Rate of interest per annum
  (t : ℝ) -- Time period in years
  (h1 : r = 12) -- Given rate is 12% per annum
  (h2 : (P * r * t) / 100 = (6/5) * P) -- Simple interest equation
  : t = 10 := by
sorry

end NUMINAMATH_CALUDE_simple_interest_time_period_l1149_114912


namespace NUMINAMATH_CALUDE_total_shells_eq_195_l1149_114971

/-- The number of shells David has -/
def david_shells : ℕ := 15

/-- The number of shells Mia has -/
def mia_shells : ℕ := 4 * david_shells

/-- The number of shells Ava has -/
def ava_shells : ℕ := mia_shells + 20

/-- The number of shells Alice has -/
def alice_shells : ℕ := ava_shells / 2

/-- The total number of shells -/
def total_shells : ℕ := david_shells + mia_shells + ava_shells + alice_shells

theorem total_shells_eq_195 : total_shells = 195 := by
  sorry

end NUMINAMATH_CALUDE_total_shells_eq_195_l1149_114971


namespace NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l1149_114907

theorem solution_set_quadratic_inequality :
  {x : ℝ | 4 - x^2 < 0} = Set.Ioi 2 ∪ Set.Iio (-2) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l1149_114907


namespace NUMINAMATH_CALUDE_combined_swimming_distance_is_1890_l1149_114979

/-- Calculates the combined swimming distance for Jamir, Sarah, and Julien over a week. -/
def combinedSwimmingDistance (julienDailyDistance : ℕ) (daysInWeek : ℕ) : ℕ :=
  let sarahDailyDistance := 2 * julienDailyDistance
  let jamirDailyDistance := sarahDailyDistance + 20
  (julienDailyDistance + sarahDailyDistance + jamirDailyDistance) * daysInWeek

/-- Proves that the combined swimming distance for Jamir, Sarah, and Julien over a week is 1890 meters. -/
theorem combined_swimming_distance_is_1890 :
  combinedSwimmingDistance 50 7 = 1890 := by
  sorry

end NUMINAMATH_CALUDE_combined_swimming_distance_is_1890_l1149_114979


namespace NUMINAMATH_CALUDE_mathematician_meeting_theorem_l1149_114917

theorem mathematician_meeting_theorem (n p q r : ℕ) (h1 : n = p - q * Real.sqrt r) 
  (h2 : 0 < p ∧ 0 < q ∧ 0 < r) (h3 : ∀ (prime : ℕ), Prime prime → ¬(prime^2 ∣ r)) 
  (h4 : ((120 - n : ℝ) / 120)^2 = 1/2) : p + q + r = 182 := by
sorry

end NUMINAMATH_CALUDE_mathematician_meeting_theorem_l1149_114917


namespace NUMINAMATH_CALUDE_margin_formula_l1149_114940

theorem margin_formula (n : ℝ) (C S M : ℝ) 
  (h1 : n > 0) 
  (h2 : M = (2/n) * C) 
  (h3 : S - M = C) : 
  M = (2/(n+2)) * S := 
by sorry

end NUMINAMATH_CALUDE_margin_formula_l1149_114940


namespace NUMINAMATH_CALUDE_complement_of_union_in_U_l1149_114921

def U : Set ℕ := {1, 2, 3, 4}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {2, 3}

theorem complement_of_union_in_U :
  (U \ (M ∪ N)) = {4} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_in_U_l1149_114921


namespace NUMINAMATH_CALUDE_smallest_number_with_conditions_l1149_114916

theorem smallest_number_with_conditions : ∃ A : ℕ,
  (A % 10 = 6) ∧
  (4 * A = 6 * (A / 10)) ∧
  (∀ B : ℕ, B < A → ¬(B % 10 = 6 ∧ 4 * B = 6 * (B / 10))) ∧
  A = 153846 := by
sorry

end NUMINAMATH_CALUDE_smallest_number_with_conditions_l1149_114916


namespace NUMINAMATH_CALUDE_min_value_of_expression_l1149_114974

theorem min_value_of_expression (x : ℝ) (h : x > 0) :
  x + 4 / (x + 1) ≥ 3 ∧ ∃ y > 0, y + 4 / (y + 1) = 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l1149_114974


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1149_114995

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  S : ℕ → ℝ  -- Sum function
  is_arithmetic : ∀ n, a (n + 1) = a n + d
  sum_def : ∀ n, S n = (n : ℝ) * (2 * a 1 + (n - 1) * d) / 2

/-- The main theorem -/
theorem arithmetic_sequence_common_difference
  (seq : ArithmeticSequence)
  (h : 2 * seq.S 3 = 3 * seq.S 2 + 6) :
  seq.d = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1149_114995


namespace NUMINAMATH_CALUDE_polynomial_remainder_l1149_114904

/-- Given a polynomial q(x) = Dx^4 + Ex^2 + Fx + 7, where the remainder when divided by x - 2 is 21,
    the remainder when divided by x + 2 is 21 - 2F -/
theorem polynomial_remainder (D E F : ℝ) : 
  let q : ℝ → ℝ := λ x ↦ D * x^4 + E * x^2 + F * x + 7
  (q 2 = 21) → 
  ∃ r : ℝ, ∀ x : ℝ, ∃ k : ℝ, q x = (x + 2) * k + r ∧ r = 21 - 2 * F :=
by sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l1149_114904


namespace NUMINAMATH_CALUDE_line_perpendicular_to_plane_l1149_114943

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem line_perpendicular_to_plane 
  (m n : Line) (α : Plane) :
  perpendicular m α → parallel m n → perpendicular n α :=
sorry

end NUMINAMATH_CALUDE_line_perpendicular_to_plane_l1149_114943


namespace NUMINAMATH_CALUDE_new_profit_percentage_l1149_114914

/-- Given the initial and new manufacturing costs, and the initial profit percentage,
    calculate the new profit percentage of the selling price. -/
theorem new_profit_percentage
  (initial_cost : ℝ)
  (new_cost : ℝ)
  (initial_profit_percentage : ℝ)
  (h_initial_cost : initial_cost = 70)
  (h_new_cost : new_cost = 50)
  (h_initial_profit_percentage : initial_profit_percentage = 30)
  : (1 - new_cost / (initial_cost / (1 - initial_profit_percentage / 100))) * 100 = 50 := by
  sorry

#check new_profit_percentage

end NUMINAMATH_CALUDE_new_profit_percentage_l1149_114914


namespace NUMINAMATH_CALUDE_inequality_proof_l1149_114991

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum_eq : a + b + c = 3) : 
  (a^2 + 6) / (2*a^2 + 2*b^2 + 2*c^2 + 2*a - 1) + 
  (b^2 + 6) / (2*a^2 + 2*b^2 + 2*c^2 + 2*b - 1) + 
  (c^2 + 6) / (2*a^2 + 2*b^2 + 2*c^2 + 2*c - 1) ≤ 3 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1149_114991


namespace NUMINAMATH_CALUDE_odd_function_implies_a_equals_negative_one_l1149_114983

def f (a : ℝ) (x : ℝ) : ℝ := x - a - 1

theorem odd_function_implies_a_equals_negative_one :
  (∀ x : ℝ, f a (-x) = -(f a x)) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_implies_a_equals_negative_one_l1149_114983


namespace NUMINAMATH_CALUDE_locus_of_point_P_l1149_114933

/-- The locus of point P given a line and specific conditions --/
theorem locus_of_point_P (x y m n : ℝ) :
  (m / 4 + n / 3 = 1) → -- M(m, n) is on the line l
  (x - m = -2 * x) →    -- Condition from AP = 2PB
  (y = 2 * n - 2 * y) → -- Condition from AP = 2PB
  (3 * x / 4 + y / 2 = 1) := by
sorry


end NUMINAMATH_CALUDE_locus_of_point_P_l1149_114933


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l1149_114925

theorem partial_fraction_decomposition (x : ℝ) (h1 : x ≠ 9) (h2 : x ≠ -4) :
  (5 * x + 7) / (x^2 - 5*x - 36) = 4 / (x - 9) + 1 / (x + 4) := by
  sorry

#check partial_fraction_decomposition

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l1149_114925


namespace NUMINAMATH_CALUDE_trailing_zeros_1_to_20_l1149_114980

/-- The number of factors of 5 in n! -/
def count_factors_of_5 (n : ℕ) : ℕ :=
  (n / 5) + (n / 25)

/-- The number of trailing zeros in the product of factorials from 1 to n -/
def trailing_zeros_factorial_product (n : ℕ) : ℕ :=
  count_factors_of_5 n

theorem trailing_zeros_1_to_20 :
  trailing_zeros_factorial_product 20 = 8 ∧
  trailing_zeros_factorial_product 20 % 100 = 8 := by
  sorry

end NUMINAMATH_CALUDE_trailing_zeros_1_to_20_l1149_114980


namespace NUMINAMATH_CALUDE_geometric_sequence_solution_l1149_114968

/-- A geometric sequence with three consecutive terms x, 2x+2, and 3x+3 has x = -4 -/
theorem geometric_sequence_solution (x : ℝ) : 
  (∃ r : ℝ, r ≠ 0 ∧ (2*x + 2) = x * r ∧ (3*x + 3) = (2*x + 2) * r) → x = -4 :=
by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_solution_l1149_114968


namespace NUMINAMATH_CALUDE_intersection_M_N_l1149_114920

def M : Set ℝ := {x | -2 < x ∧ x < 3}
def N : Set ℝ := {-2, -1, 0, 1}

theorem intersection_M_N : M ∩ N = {-1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1149_114920


namespace NUMINAMATH_CALUDE_square_perimeter_ratio_l1149_114951

theorem square_perimeter_ratio (s S : ℝ) (hs : s > 0) (hS : S > 0) : 
  S * Real.sqrt 2 = 7 * (s * Real.sqrt 2) → 
  (4 * S) / (4 * s) = 7 := by sorry

end NUMINAMATH_CALUDE_square_perimeter_ratio_l1149_114951


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_by_nine_l1149_114941

def original_number : ℕ := 228712

theorem least_addition_for_divisibility_by_nine :
  ∃ (n : ℕ), n = 5 ∧ 
  (∀ (m : ℕ), m < n → ¬((original_number + m) % 9 = 0)) ∧
  ((original_number + n) % 9 = 0) :=
sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_by_nine_l1149_114941


namespace NUMINAMATH_CALUDE_divisibility_of_sum_of_squares_l1149_114987

theorem divisibility_of_sum_of_squares (p k a b : ℤ) : 
  Prime p → 
  p = 4*k + 3 → 
  p ∣ (a^2 + b^2) → 
  p ∣ a ∧ p ∣ b := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_sum_of_squares_l1149_114987


namespace NUMINAMATH_CALUDE_income_ratio_proof_l1149_114905

/-- Given two persons P1 and P2 with the following conditions:
    1. The ratio of their expenditures is 3:2
    2. Each saves 2200 at the end of the year
    3. The income of P1 is 5500
    Prove that the ratio of their incomes is 5:4 -/
theorem income_ratio_proof (income_P1 income_P2 expenditure_P1 expenditure_P2 : ℕ) : 
  income_P1 = 5500 →
  expenditure_P1 = income_P1 - 2200 →
  expenditure_P2 = income_P2 - 2200 →
  3 * expenditure_P2 = 2 * expenditure_P1 →
  5 * income_P2 = 4 * income_P1 := by
  sorry

#check income_ratio_proof

end NUMINAMATH_CALUDE_income_ratio_proof_l1149_114905


namespace NUMINAMATH_CALUDE_triangle_area_with_given_base_and_height_l1149_114958

/-- The area of a triangle with base 12 cm and height 15 cm is 90 cm². -/
theorem triangle_area_with_given_base_and_height :
  let base : ℝ := 12
  let height : ℝ := 15
  let area : ℝ := (1 / 2) * base * height
  area = 90 := by sorry

end NUMINAMATH_CALUDE_triangle_area_with_given_base_and_height_l1149_114958


namespace NUMINAMATH_CALUDE_clothing_store_inventory_l1149_114959

theorem clothing_store_inventory (belts : ℕ) (black_shirts : ℕ) (white_shirts : ℕ) :
  belts = 40 →
  black_shirts = 63 →
  white_shirts = 42 →
  ∃ (ties : ℕ) (scarves : ℕ) (jeans : ℕ),
    jeans = (2 * (black_shirts + white_shirts)) / 3 ∧
    scarves = (ties + belts) / 2 ∧
    jeans = scarves + 33 ∧
    ties = 34 :=
by sorry

end NUMINAMATH_CALUDE_clothing_store_inventory_l1149_114959


namespace NUMINAMATH_CALUDE_rita_swimming_months_l1149_114993

/-- The number of months Rita needs to fulfill her coach's requirements -/
def months_to_fulfill_requirement (total_required_hours : ℕ) (hours_already_completed : ℕ) (hours_per_month : ℕ) : ℕ :=
  (total_required_hours - hours_already_completed) / hours_per_month

/-- Proof that Rita needs 6 months to fulfill her coach's requirements -/
theorem rita_swimming_months : 
  months_to_fulfill_requirement 1500 180 220 = 6 := by
sorry

end NUMINAMATH_CALUDE_rita_swimming_months_l1149_114993


namespace NUMINAMATH_CALUDE_inverse_proportion_comparison_l1149_114953

theorem inverse_proportion_comparison (k : ℝ) (y₁ y₂ : ℝ) 
  (h1 : k > 0) 
  (h2 : y₁ = k / (-2)) 
  (h3 : y₂ = k / (-1)) : 
  y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_comparison_l1149_114953


namespace NUMINAMATH_CALUDE_largest_five_digit_multiple_of_3_and_4_l1149_114955

theorem largest_five_digit_multiple_of_3_and_4 : 
  ∀ n : ℕ, n ≤ 99999 ∧ n ≥ 10000 ∧ 3 ∣ n ∧ 4 ∣ n → n ≤ 99996 :=
by sorry

end NUMINAMATH_CALUDE_largest_five_digit_multiple_of_3_and_4_l1149_114955


namespace NUMINAMATH_CALUDE_maria_bottles_l1149_114981

/-- The number of bottles Maria has at the end, given her initial number of bottles,
    the number she drinks, and the number she buys. -/
def final_bottles (initial : ℕ) (drunk : ℕ) (bought : ℕ) : ℕ :=
  initial - drunk + bought

/-- Theorem stating that Maria ends up with 51 bottles given the problem conditions -/
theorem maria_bottles : final_bottles 14 8 45 = 51 := by
  sorry

end NUMINAMATH_CALUDE_maria_bottles_l1149_114981


namespace NUMINAMATH_CALUDE_expansion_terms_count_l1149_114977

/-- The number of terms in the expansion of a product of two sums -/
def num_terms_in_expansion (n m : ℕ) : ℕ := n * m

/-- The first factor (a+b+c+d) has 4 terms -/
def first_factor_terms : ℕ := 4

/-- The second factor (e+f+g+h+i) has 5 terms -/
def second_factor_terms : ℕ := 5

theorem expansion_terms_count :
  num_terms_in_expansion first_factor_terms second_factor_terms = 20 := by
  sorry

end NUMINAMATH_CALUDE_expansion_terms_count_l1149_114977


namespace NUMINAMATH_CALUDE_january_first_is_tuesday_l1149_114985

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a calendar month -/
structure Month where
  days : Nat
  firstDay : DayOfWeek

/-- Counts the occurrences of a specific day in a month -/
def countDayInMonth (m : Month) (d : DayOfWeek) : Nat :=
  sorry

/-- Main theorem: If January has 31 days, and there are exactly four Fridays and four Mondays, then January 1st is a Tuesday -/
theorem january_first_is_tuesday (jan : Month) :
  jan.days = 31 →
  countDayInMonth jan DayOfWeek.Friday = 4 →
  countDayInMonth jan DayOfWeek.Monday = 4 →
  jan.firstDay = DayOfWeek.Tuesday :=
sorry

end NUMINAMATH_CALUDE_january_first_is_tuesday_l1149_114985


namespace NUMINAMATH_CALUDE_jebbs_take_home_pay_l1149_114931

/-- Calculates the take-home pay given the total pay and tax rate -/
def takeHomePay (totalPay : ℝ) (taxRate : ℝ) : ℝ :=
  totalPay * (1 - taxRate)

/-- Theorem stating that given a total pay of 650 and a tax rate of 10%, the take-home pay is 585 -/
theorem jebbs_take_home_pay :
  takeHomePay 650 0.1 = 585 := by
  sorry

end NUMINAMATH_CALUDE_jebbs_take_home_pay_l1149_114931


namespace NUMINAMATH_CALUDE_arcade_tickets_l1149_114902

theorem arcade_tickets (initial_tickets yoyo_cost : ℝ) 
  (h1 : initial_tickets = 48.5)
  (h2 : yoyo_cost = 11.7) : 
  initial_tickets - (initial_tickets - yoyo_cost) = yoyo_cost := by
sorry

end NUMINAMATH_CALUDE_arcade_tickets_l1149_114902


namespace NUMINAMATH_CALUDE_function_property_l1149_114989

variable (f : ℝ → ℝ)
variable (p q : ℝ)

theorem function_property
  (h1 : ∀ a b, f (a * b) = f a + f b)
  (h2 : f 2 = p)
  (h3 : f 3 = q) :
  f 12 = 2 * p + q :=
by sorry

end NUMINAMATH_CALUDE_function_property_l1149_114989


namespace NUMINAMATH_CALUDE_min_cooking_time_is_12_l1149_114967

/-- Represents the time taken for each step in the cooking process -/
structure CookingSteps where
  step1 : ℕ  -- Wash pot and fill with water
  step2 : ℕ  -- Wash vegetables
  step3 : ℕ  -- Prepare noodles and seasonings
  step4 : ℕ  -- Boil water
  step5 : ℕ  -- Cook noodles and vegetables

/-- Calculates the minimum cooking time given the cooking steps -/
def minCookingTime (steps : CookingSteps) : ℕ :=
  steps.step1 + max steps.step4 (steps.step2 + steps.step3) + steps.step5

/-- Theorem stating that the minimum cooking time is 12 minutes -/
theorem min_cooking_time_is_12 (steps : CookingSteps) 
  (h1 : steps.step1 = 2)
  (h2 : steps.step2 = 3)
  (h3 : steps.step3 = 2)
  (h4 : steps.step4 = 7)
  (h5 : steps.step5 = 3) :
  minCookingTime steps = 12 := by
  sorry


end NUMINAMATH_CALUDE_min_cooking_time_is_12_l1149_114967


namespace NUMINAMATH_CALUDE_new_class_mean_l1149_114906

theorem new_class_mean (total_students : ℕ) (initial_students : ℕ) (later_students : ℕ)
  (initial_mean : ℚ) (later_mean : ℚ) :
  total_students = initial_students + later_students →
  initial_students = 30 →
  later_students = 6 →
  initial_mean = 72 / 100 →
  later_mean = 78 / 100 →
  (initial_students * initial_mean + later_students * later_mean) / total_students = 73 / 100 :=
by sorry

end NUMINAMATH_CALUDE_new_class_mean_l1149_114906


namespace NUMINAMATH_CALUDE_weight_of_four_moles_l1149_114962

/-- Given a compound with a molecular weight of 260, prove that 4 moles of this compound weighs 1040 grams. -/
theorem weight_of_four_moles (molecular_weight : ℝ) (moles : ℝ) : 
  molecular_weight = 260 → moles = 4 → moles * molecular_weight = 1040 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_four_moles_l1149_114962
