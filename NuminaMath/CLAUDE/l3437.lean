import Mathlib

namespace NUMINAMATH_CALUDE_maria_savings_l3437_343767

/-- The amount left in Maria's savings after buying sweaters and scarves -/
def savings_left (sweater_price scarf_price sweater_count scarf_count initial_savings : ℕ) : ℕ :=
  initial_savings - (sweater_price * sweater_count + scarf_price * scarf_count)

/-- Theorem stating that Maria will have $200 left in her savings -/
theorem maria_savings : savings_left 30 20 6 6 500 = 200 := by
  sorry

end NUMINAMATH_CALUDE_maria_savings_l3437_343767


namespace NUMINAMATH_CALUDE_sum_of_square_areas_l3437_343711

theorem sum_of_square_areas (square1_side : ℝ) (square2_side : ℝ) 
  (h1 : square1_side = 11) (h2 : square2_side = 5) : 
  square1_side ^ 2 + square2_side ^ 2 = 146 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_square_areas_l3437_343711


namespace NUMINAMATH_CALUDE_gcd_of_specific_numbers_l3437_343753

theorem gcd_of_specific_numbers : Nat.gcd 33333333 666666666 = 2 := by sorry

end NUMINAMATH_CALUDE_gcd_of_specific_numbers_l3437_343753


namespace NUMINAMATH_CALUDE_f_of_two_eq_neg_eight_l3437_343790

/-- Given a function f(x) = x^5 + ax^3 + bx + 1 where f(-2) = 10, prove that f(2) = -8 -/
theorem f_of_two_eq_neg_eight (a b : ℝ) (f : ℝ → ℝ) 
    (h1 : ∀ x, f x = x^5 + a*x^3 + b*x + 1)
    (h2 : f (-2) = 10) : 
  f 2 = -8 := by
  sorry

end NUMINAMATH_CALUDE_f_of_two_eq_neg_eight_l3437_343790


namespace NUMINAMATH_CALUDE_train_length_problem_l3437_343792

/-- Proves that given two trains of equal length running on parallel lines in the same direction,
    with the faster train moving at 46 km/hr and the slower train at 36 km/hr,
    if the faster train passes the slower train in 72 seconds,
    then the length of each train is 100 meters. -/
theorem train_length_problem (faster_speed slower_speed : ℝ) (passing_time : ℝ) (train_length : ℝ) :
  faster_speed = 46 →
  slower_speed = 36 →
  passing_time = 72 →
  (faster_speed - slower_speed) * passing_time * (5 / 18) = 2 * train_length →
  train_length = 100 := by
  sorry

end NUMINAMATH_CALUDE_train_length_problem_l3437_343792


namespace NUMINAMATH_CALUDE_pizza_theorem_l3437_343759

def pizza_problem (total_served : ℕ) (successfully_served : ℕ) : Prop :=
  total_served - successfully_served = 6

theorem pizza_theorem : pizza_problem 9 3 := by
  sorry

end NUMINAMATH_CALUDE_pizza_theorem_l3437_343759


namespace NUMINAMATH_CALUDE_ellipse_min_area_l3437_343757

/-- An ellipse containing two specific circles has a minimum area of π -/
theorem ellipse_min_area (a b : ℝ) (h_positive : a > 0 ∧ b > 0) :
  (∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 →
    ((x - 2)^2 + y^2 = 4 ∨ (x + 2)^2 + y^2 = 4)) →
  ∃ k : ℝ, k = 1 ∧ π * a * b ≥ k * π :=
sorry

end NUMINAMATH_CALUDE_ellipse_min_area_l3437_343757


namespace NUMINAMATH_CALUDE_circle_and_tangent_properties_l3437_343768

/-- Given a circle with center C on the line x-y+1=0 and passing through points A(1,1) and B(2,-2) -/
structure CircleData where
  C : ℝ × ℝ
  center_on_line : C.1 - C.2 + 1 = 0
  passes_through_A : (C.1 - 1)^2 + (C.2 - 1)^2 = (C.1 - 2)^2 + (C.2 + 2)^2

/-- The standard equation of the circle is (x+3)^2 + (y+2)^2 = 25 -/
def circle_equation (x y : ℝ) : Prop :=
  (x + 3)^2 + (y + 2)^2 = 25

/-- The equation of the tangent line passing through point (1,1) is 4x + 3y - 7 = 0 -/
def tangent_line_equation (x y : ℝ) : Prop :=
  4*x + 3*y - 7 = 0

theorem circle_and_tangent_properties (data : CircleData) :
  (∀ x y, circle_equation x y ↔ ((x - data.C.1)^2 + (y - data.C.2)^2 = (1 - data.C.1)^2 + (1 - data.C.2)^2)) ∧
  tangent_line_equation 1 1 ∧
  (∀ x y, tangent_line_equation x y → (x - 1) * (1 - data.C.1) + (y - 1) * (1 - data.C.2) = 0) :=
sorry

end NUMINAMATH_CALUDE_circle_and_tangent_properties_l3437_343768


namespace NUMINAMATH_CALUDE_fibonacci_inequality_l3437_343774

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

theorem fibonacci_inequality (n : ℕ) (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  (min (fibonacci n / fibonacci (n - 1)) (fibonacci (n + 1) / fibonacci n) < a / b ∧
   a / b < max (fibonacci n / fibonacci (n - 1)) (fibonacci (n + 1) / fibonacci n)) →
  b ≥ fibonacci (n + 1) :=
by sorry

end NUMINAMATH_CALUDE_fibonacci_inequality_l3437_343774


namespace NUMINAMATH_CALUDE_inscribed_square_side_length_l3437_343729

/-- A regular triangle with an inscribed square -/
structure RegularTriangleWithInscribedSquare where
  -- Side length of the regular triangle
  triangleSide : ℝ
  -- Distance from a vertex of the triangle to the nearest vertex of the square on the opposite side
  vertexToSquareDistance : ℝ
  -- Assumption that the triangle is regular (equilateral)
  regular : triangleSide > 0
  -- Assumption that the square is inscribed (vertexToSquareDistance < triangleSide)
  inscribed : vertexToSquareDistance < triangleSide

/-- The side length of the inscribed square -/
def squareSideLength (t : RegularTriangleWithInscribedSquare) : ℝ := 
  t.triangleSide - t.vertexToSquareDistance

/-- Theorem stating that for a regular triangle with side length 30 and vertexToSquareDistance 29, 
    the side length of the inscribed square is 30 -/
theorem inscribed_square_side_length 
  (t : RegularTriangleWithInscribedSquare) 
  (h1 : t.triangleSide = 30) 
  (h2 : t.vertexToSquareDistance = 29) : 
  squareSideLength t = 30 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_square_side_length_l3437_343729


namespace NUMINAMATH_CALUDE_equal_piece_length_equal_piece_length_proof_l3437_343742

/-- Given a rope of 1165 cm cut into 154 pieces, where 4 pieces are 100mm each and the rest are equal,
    the length of each equal piece is 75 mm. -/
theorem equal_piece_length : ℕ → ℕ → ℕ → ℕ → ℕ → Prop :=
  fun (total_length_cm : ℕ) (total_pieces : ℕ) (equal_pieces : ℕ) (special_pieces : ℕ) (special_length_mm : ℕ) =>
    total_length_cm = 1165 ∧
    total_pieces = 154 ∧
    equal_pieces = 150 ∧
    special_pieces = 4 ∧
    special_length_mm = 100 →
    (total_length_cm * 10 - special_pieces * special_length_mm) / equal_pieces = 75

/-- Proof of the theorem -/
theorem equal_piece_length_proof : equal_piece_length 1165 154 150 4 100 := by
  sorry

end NUMINAMATH_CALUDE_equal_piece_length_equal_piece_length_proof_l3437_343742


namespace NUMINAMATH_CALUDE_infinitely_many_solutions_l3437_343713

theorem infinitely_many_solutions (c : ℚ) :
  (∀ y : ℚ, 3 * (2 + 2 * c * y) = 15 * y + 6) ↔ c = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_solutions_l3437_343713


namespace NUMINAMATH_CALUDE_monomial_sum_l3437_343752

/-- Given constants a and b, if the sum of 4xy^2, axy^b, and -5xy is a monomial, 
    then a+b = -2 or a+b = 6 -/
theorem monomial_sum (a b : ℝ) : 
  (∃ (x y : ℝ), ∀ (z : ℝ), z = 4*x*y^2 + a*x*y^b - 5*x*y → ∃ (c : ℝ), z = c*x*y^k) → 
  a + b = -2 ∨ a + b = 6 :=
sorry

end NUMINAMATH_CALUDE_monomial_sum_l3437_343752


namespace NUMINAMATH_CALUDE_total_cats_l3437_343747

theorem total_cats (asleep : ℕ) (awake : ℕ) (h1 : asleep = 92) (h2 : awake = 6) :
  asleep + awake = 98 := by
  sorry

end NUMINAMATH_CALUDE_total_cats_l3437_343747


namespace NUMINAMATH_CALUDE_abs_x_minus_two_geq_abs_x_l3437_343740

theorem abs_x_minus_two_geq_abs_x (x : ℝ) : |x - 2| ≥ |x| ↔ x ≤ 1 := by sorry

end NUMINAMATH_CALUDE_abs_x_minus_two_geq_abs_x_l3437_343740


namespace NUMINAMATH_CALUDE_circle_area_theorem_l3437_343761

theorem circle_area_theorem (r : ℝ) (h : 3 * (1 / (2 * Real.pi * r)) = r) : 
  Real.pi * r^2 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_theorem_l3437_343761


namespace NUMINAMATH_CALUDE_remaining_pie_portion_l3437_343719

theorem remaining_pie_portion (carlos_share maria_fraction : ℝ) : 
  carlos_share = 0.6 →
  maria_fraction = 0.25 →
  (1 - carlos_share) * (1 - maria_fraction) = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_remaining_pie_portion_l3437_343719


namespace NUMINAMATH_CALUDE_integer_triple_divisibility_l3437_343744

theorem integer_triple_divisibility :
  ∀ p q r : ℕ,
    1 < p → p < q → q < r →
    (p * q * r - 1) % ((p - 1) * (q - 1) * (r - 1)) = 0 →
    ((p = 2 ∧ q = 4 ∧ r = 8) ∨ (p = 3 ∧ q = 5 ∧ r = 15)) :=
by sorry

end NUMINAMATH_CALUDE_integer_triple_divisibility_l3437_343744


namespace NUMINAMATH_CALUDE_addition_proof_l3437_343741

theorem addition_proof : 72 + 15 = 87 := by
  sorry

end NUMINAMATH_CALUDE_addition_proof_l3437_343741


namespace NUMINAMATH_CALUDE_vector_operation_result_l3437_343798

theorem vector_operation_result :
  let v₁ : Fin 3 → ℝ := ![(-3), 2, (-5)]
  let v₂ : Fin 3 → ℝ := ![1, 6, (-3)]
  2 • v₁ + v₂ = ![-5, 10, (-13)] := by
sorry

end NUMINAMATH_CALUDE_vector_operation_result_l3437_343798


namespace NUMINAMATH_CALUDE_routes_to_n_2_l3437_343769

/-- The number of possible routes from (0, 0) to (x, y) moving only right or up -/
def f (x y : ℕ) : ℕ := sorry

/-- Theorem: The number of routes from (0, 0) to (n, 2) is (n^2 + 3n + 2) / 2 -/
theorem routes_to_n_2 (n : ℕ) : f n 2 = (n^2 + 3*n + 2) / 2 := by sorry

end NUMINAMATH_CALUDE_routes_to_n_2_l3437_343769


namespace NUMINAMATH_CALUDE_constant_term_expansion_l3437_343700

/-- The constant term in the expansion of (2x + 1/x)^6 -/
def constantTerm : ℕ := 160

/-- The binomial coefficient function -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The general term of the expansion -/
def generalTerm (r : ℕ) : ℚ :=
  2^(6 - r) * binomial 6 r * (1 : ℚ)

theorem constant_term_expansion :
  constantTerm = generalTerm 3 := by sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l3437_343700


namespace NUMINAMATH_CALUDE_tennis_tournament_has_three_cycle_l3437_343766

/-- Represents a tennis tournament as a directed graph -/
structure TennisTournament where
  -- The set of participants
  V : Type
  -- The "wins against" relation
  E : V → V → Prop
  -- There are at least three participants
  atleastThree : ∃ (a b c : V), a ≠ b ∧ b ≠ c ∧ a ≠ c
  -- Every participant plays against every other participant exactly once
  complete : ∀ (a b : V), a ≠ b → (E a b ∨ E b a) ∧ ¬(E a b ∧ E b a)
  -- Every participant wins at least one match
  hasWin : ∀ (a : V), ∃ (b : V), E a b

/-- A 3-cycle in the tournament -/
def HasThreeCycle (T : TennisTournament) : Prop :=
  ∃ (a b c : T.V), T.E a b ∧ T.E b c ∧ T.E c a

/-- The main theorem: every tennis tournament has a 3-cycle -/
theorem tennis_tournament_has_three_cycle (T : TennisTournament) : HasThreeCycle T := by
  sorry

end NUMINAMATH_CALUDE_tennis_tournament_has_three_cycle_l3437_343766


namespace NUMINAMATH_CALUDE_custom_mult_example_l3437_343716

-- Define the custom operation *
def custom_mult (a b : Int) : Int := a * b

-- Theorem statement
theorem custom_mult_example : custom_mult 2 (-3) = -6 := by
  sorry

end NUMINAMATH_CALUDE_custom_mult_example_l3437_343716


namespace NUMINAMATH_CALUDE_money_division_l3437_343725

theorem money_division (total : ℝ) (p q r : ℝ) : 
  p + q + r = total →
  p / q = 3 / 7 →
  q / r = 7 / 12 →
  q - p = 3200 →
  r - q = 4000 := by
sorry

end NUMINAMATH_CALUDE_money_division_l3437_343725


namespace NUMINAMATH_CALUDE_cube_difference_problem_l3437_343726

theorem cube_difference_problem (a b c : ℝ) 
  (positive_a : 0 < a) (positive_b : 0 < b) (positive_c : 0 < c)
  (sum_of_squares : a^2 + b^2 + c^2 = 160)
  (largest_sum : a = b + c)
  (difference : b - c = 4) :
  |b^3 - c^3| = 320 := by
sorry

end NUMINAMATH_CALUDE_cube_difference_problem_l3437_343726


namespace NUMINAMATH_CALUDE_calculate_a10_l3437_343727

/-- A sequence satisfying the given property -/
def special_sequence (a : ℕ+ → ℤ) : Prop :=
  ∀ (p q : ℕ+), a (p + q) = a p + a q

/-- The theorem to prove -/
theorem calculate_a10 (a : ℕ+ → ℤ) 
  (h1 : special_sequence a) 
  (h2 : a 2 = -6) : 
  a 10 = -30 := by
sorry

end NUMINAMATH_CALUDE_calculate_a10_l3437_343727


namespace NUMINAMATH_CALUDE_cube_surface_area_increase_l3437_343730

/-- If each edge of a cube increases by 20%, the surface area of the cube increases by 44%. -/
theorem cube_surface_area_increase (L : ℝ) (h : L > 0) :
  let original_area := 6 * L^2
  let new_edge := 1.2 * L
  let new_area := 6 * new_edge^2
  (new_area - original_area) / original_area = 0.44 := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_increase_l3437_343730


namespace NUMINAMATH_CALUDE_book_purchase_problem_l3437_343765

/-- Given information about book purchases, prove the number of people who purchased both books --/
theorem book_purchase_problem (A B AB : ℕ) 
  (h1 : A = 2 * B)
  (h2 : AB = 2 * (B - AB))
  (h3 : A - AB = 1000) :
  AB = 500 := by
  sorry

end NUMINAMATH_CALUDE_book_purchase_problem_l3437_343765


namespace NUMINAMATH_CALUDE_find_x_l3437_343782

theorem find_x : ∃ x : ℤ, 9873 + x = 13800 ∧ x = 3927 := by
  sorry

end NUMINAMATH_CALUDE_find_x_l3437_343782


namespace NUMINAMATH_CALUDE_m_range_for_fourth_quadrant_l3437_343736

/-- A point in the fourth quadrant has positive x-coordinate and negative y-coordinate -/
def is_in_fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

/-- The coordinates of point M are (m+2, m) -/
def point_M (m : ℝ) : ℝ × ℝ := (m + 2, m)

/-- Theorem stating the range of m for point M to be in the fourth quadrant -/
theorem m_range_for_fourth_quadrant :
  ∀ m : ℝ, is_in_fourth_quadrant (point_M m).1 (point_M m).2 ↔ -2 < m ∧ m < 0 := by
  sorry

end NUMINAMATH_CALUDE_m_range_for_fourth_quadrant_l3437_343736


namespace NUMINAMATH_CALUDE_factor_x_squared_minus_four_ninety_eight_squared_minus_four_exists_n_for_equation_three_nine_nine_nine_nine_nine_one_not_prime_l3437_343780

-- Part (a)
theorem factor_x_squared_minus_four (x : ℝ) : x^2 - 4 = (x - 2) * (x + 2) := by sorry

-- Part (b)
theorem ninety_eight_squared_minus_four : 98^2 - 4 = 100 * 96 := by sorry

-- Part (c)
theorem exists_n_for_equation : ∃ n : ℕ+, (20 - n) * (20 + n) = 391 ∧ n = 3 := by sorry

-- Part (d)
theorem three_nine_nine_nine_nine_nine_one_not_prime : ¬ Nat.Prime 3999991 := by sorry

end NUMINAMATH_CALUDE_factor_x_squared_minus_four_ninety_eight_squared_minus_four_exists_n_for_equation_three_nine_nine_nine_nine_nine_one_not_prime_l3437_343780


namespace NUMINAMATH_CALUDE_math_statements_l3437_343770

theorem math_statements :
  (∃ x : ℚ, x < -1 ∧ x > 1/x) ∧
  (∃ y : ℝ, y ≥ 0 ∧ -y ≥ y) ∧
  (∀ z : ℚ, z < 0 → z^2 > z) :=
by sorry

end NUMINAMATH_CALUDE_math_statements_l3437_343770


namespace NUMINAMATH_CALUDE_alternative_plan_more_expensive_l3437_343749

/-- Represents a phone plan with its pricing structure -/
structure PhonePlan where
  text_cost : ℚ  -- Cost per 30 texts
  call_cost : ℚ  -- Cost per 20 minutes of calls
  data_cost : ℚ  -- Cost per 2GB of data
  intl_cost : ℚ  -- Additional cost for international calls

/-- Represents a user's monthly usage -/
structure Usage where
  texts : ℕ
  call_minutes : ℕ
  data_gb : ℚ
  intl_calls : Bool

def calculate_cost (plan : PhonePlan) (usage : Usage) : ℚ :=
  let text_units := (usage.texts + 29) / 30
  let call_units := (usage.call_minutes + 19) / 20
  let data_units := ⌈usage.data_gb / 2⌉
  plan.text_cost * text_units +
  plan.call_cost * call_units +
  plan.data_cost * data_units +
  if usage.intl_calls then plan.intl_cost else 0

theorem alternative_plan_more_expensive :
  let current_plan_cost : ℚ := 12
  let alternative_plan := PhonePlan.mk 1 3 5 2
  let darnell_usage := Usage.mk 60 60 3 true
  calculate_cost alternative_plan darnell_usage - current_plan_cost = 11 := by
  sorry

end NUMINAMATH_CALUDE_alternative_plan_more_expensive_l3437_343749


namespace NUMINAMATH_CALUDE_simplify_sqrt_sum_l3437_343751

theorem simplify_sqrt_sum : 
  (Real.sqrt 726 / Real.sqrt 81) + (Real.sqrt 294 / Real.sqrt 49) = (33 * Real.sqrt 2 + 9 * Real.sqrt 6) / 9 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_sum_l3437_343751


namespace NUMINAMATH_CALUDE_rectangle_shorter_side_l3437_343763

/-- A rectangle with perimeter 60 meters and area 221 square meters has a shorter side of 13 meters. -/
theorem rectangle_shorter_side (a b : ℝ) : 
  a > 0 → b > 0 →  -- positive sides
  2 * (a + b) = 60 →  -- perimeter condition
  a * b = 221 →  -- area condition
  min a b = 13 := by
sorry

end NUMINAMATH_CALUDE_rectangle_shorter_side_l3437_343763


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l3437_343794

theorem quadratic_no_real_roots (a : ℝ) : 
  (¬ ∃ x : ℝ, x^2 - a*x + 1 = 0) ↔ -2 < a ∧ a < 2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l3437_343794


namespace NUMINAMATH_CALUDE_great_eighteen_hockey_league_games_l3437_343748

/-- Represents a sports league with the given structure -/
structure League where
  total_teams : ℕ
  divisions : ℕ
  teams_per_division : ℕ
  intra_division_games : ℕ
  inter_division_games : ℕ

/-- Calculates the total number of games in the league -/
def total_games (l : League) : ℕ :=
  (l.total_teams * (l.teams_per_division - 1) * l.intra_division_games +
   l.total_teams * (l.total_teams - l.teams_per_division) * l.inter_division_games) / 2

/-- Theorem stating that the given league structure results in 243 total games -/
theorem great_eighteen_hockey_league_games :
  ∃ (l : League),
    l.total_teams = 18 ∧
    l.divisions = 3 ∧
    l.teams_per_division = 6 ∧
    l.intra_division_games = 3 ∧
    l.inter_division_games = 1 ∧
    total_games l = 243 := by
  sorry


end NUMINAMATH_CALUDE_great_eighteen_hockey_league_games_l3437_343748


namespace NUMINAMATH_CALUDE_vector_sum_zero_l3437_343731

variable {V : Type*} [AddCommGroup V]

/-- Given four points A, B, C, and D in a vector space, 
    prove that AB + BD - AC - CD equals the zero vector -/
theorem vector_sum_zero (A B C D : V) : 
  (B - A) + (D - B) - (C - A) - (D - C) = (0 : V) := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_zero_l3437_343731


namespace NUMINAMATH_CALUDE_willy_stuffed_animals_l3437_343760

def stuffed_animals_total (initial : ℕ) (mom_gift : ℕ) (dad_multiplier : ℕ) : ℕ :=
  let after_mom := initial + mom_gift
  let dad_gift := after_mom * dad_multiplier
  after_mom + dad_gift

theorem willy_stuffed_animals :
  stuffed_animals_total 10 2 3 = 48 := by
  sorry

end NUMINAMATH_CALUDE_willy_stuffed_animals_l3437_343760


namespace NUMINAMATH_CALUDE_southern_tents_l3437_343787

/-- Represents the number of tents in different parts of the campsite -/
structure Campsite where
  total : ℕ
  north : ℕ
  east : ℕ
  center : ℕ
  south : ℕ

/-- Theorem stating the number of tents in the southern part of the campsite -/
theorem southern_tents (c : Campsite) 
  (h_total : c.total = 900)
  (h_north : c.north = 100)
  (h_east : c.east = 2 * c.north)
  (h_center : c.center = 4 * c.north)
  (h_sum : c.total = c.north + c.east + c.center + c.south) : 
  c.south = 200 := by
  sorry


end NUMINAMATH_CALUDE_southern_tents_l3437_343787


namespace NUMINAMATH_CALUDE_simplify_expression_l3437_343771

theorem simplify_expression (x y z : ℝ) : 
  (3 * x - (2 * y - 4 * z)) - ((3 * x - 2 * y) - 5 * z) = 9 * z := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3437_343771


namespace NUMINAMATH_CALUDE_smallest_largest_8digit_multiples_of_360_l3437_343777

/-- Checks if a number has all unique digits --/
def hasUniqueDigits (n : Nat) : Bool :=
  let digits := n.digits 10
  digits.length = digits.toFinset.card

/-- Checks if a number is a multiple of 360 --/
def isMultipleOf360 (n : Nat) : Bool :=
  n % 360 = 0

/-- Theorem: 12378960 and 98763120 are the smallest and largest 8-digit multiples of 360 with unique digits --/
theorem smallest_largest_8digit_multiples_of_360 :
  (∀ n : Nat, n ≥ 10000000 ∧ n < 100000000 ∧ isMultipleOf360 n ∧ hasUniqueDigits n →
    n ≥ 12378960) ∧
  (∀ n : Nat, n ≥ 10000000 ∧ n < 100000000 ∧ isMultipleOf360 n ∧ hasUniqueDigits n →
    n ≤ 98763120) ∧
  isMultipleOf360 12378960 ∧
  isMultipleOf360 98763120 ∧
  hasUniqueDigits 12378960 ∧
  hasUniqueDigits 98763120 :=
by sorry


end NUMINAMATH_CALUDE_smallest_largest_8digit_multiples_of_360_l3437_343777


namespace NUMINAMATH_CALUDE_spaceship_break_time_l3437_343704

/-- Represents the travel pattern of a spaceship -/
structure TravelPattern where
  initialTravel : ℕ
  initialBreak : ℕ
  secondTravel : ℕ
  secondBreak : ℕ
  subsequentTravel : ℕ
  subsequentBreak : ℕ

/-- Calculates the total break time for a spaceship journey -/
def calculateBreakTime (pattern : TravelPattern) (totalJourneyTime : ℕ) : ℕ :=
  sorry

/-- Theorem stating that for the given travel pattern and journey time, 
    the total break time is 8 hours -/
theorem spaceship_break_time :
  let pattern : TravelPattern := {
    initialTravel := 10,
    initialBreak := 3,
    secondTravel := 10,
    secondBreak := 1,
    subsequentTravel := 11,
    subsequentBreak := 1
  }
  let totalJourneyTime : ℕ := 72
  calculateBreakTime pattern totalJourneyTime = 8 := by
  sorry

end NUMINAMATH_CALUDE_spaceship_break_time_l3437_343704


namespace NUMINAMATH_CALUDE_license_plate_sampling_is_systematic_l3437_343739

/-- Represents a car's license plate --/
structure LicensePlate where
  number : ℕ

/-- Represents a sampling method --/
inductive SamplingMethod
  | SimpleRandom
  | Lottery
  | Systematic
  | Stratified

/-- Function to check if a license plate ends with a specific digit --/
def endsWithDigit (plate : LicensePlate) (digit : ℕ) : Prop :=
  plate.number % 10 = digit

/-- Definition of systematic sampling for this context --/
def isSystematicSampling (sample : Set LicensePlate) (digit : ℕ) : Prop :=
  ∀ plate, plate ∈ sample ↔ endsWithDigit plate digit

/-- Theorem stating that selecting cars with license plates ending in a specific digit
    is equivalent to systematic sampling --/
theorem license_plate_sampling_is_systematic (sample : Set LicensePlate) (digit : ℕ) :
  (∀ plate, plate ∈ sample ↔ endsWithDigit plate digit) →
  isSystematicSampling sample digit :=
by sorry

end NUMINAMATH_CALUDE_license_plate_sampling_is_systematic_l3437_343739


namespace NUMINAMATH_CALUDE_santinos_garden_fruit_count_l3437_343707

/-- Represents the number of trees for each fruit type in Santino's garden -/
structure TreeCounts where
  papaya : ℕ
  mango : ℕ
  apple : ℕ
  orange : ℕ

/-- Represents the fruit production rate for each tree type -/
structure FruitProduction where
  papaya : ℕ
  mango : ℕ
  apple : ℕ
  orange : ℕ

/-- Calculates the total number of fruits in Santino's garden -/
def totalFruits (trees : TreeCounts) (production : FruitProduction) : ℕ :=
  trees.papaya * production.papaya +
  trees.mango * production.mango +
  trees.apple * production.apple +
  trees.orange * production.orange

theorem santinos_garden_fruit_count :
  let trees : TreeCounts := ⟨2, 3, 4, 5⟩
  let production : FruitProduction := ⟨10, 20, 15, 25⟩
  totalFruits trees production = 265 := by
  sorry

end NUMINAMATH_CALUDE_santinos_garden_fruit_count_l3437_343707


namespace NUMINAMATH_CALUDE_sum_of_absolute_coefficients_l3437_343775

theorem sum_of_absolute_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ : ℝ) :
  (∀ x, (1 - 2*x)^8 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7 + a₈*x^8) →
  |a| + |a₁| + |a₂| + |a₃| + |a₄| + |a₅| + |a₆| + |a₇| + |a₈| = 3^8 := by
sorry

end NUMINAMATH_CALUDE_sum_of_absolute_coefficients_l3437_343775


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3437_343754

-- Define the quadratic function
def f (x : ℝ) : ℝ := -6 * x^2 + 36 * x - 48

-- State the theorem
theorem quadratic_function_properties :
  f 2 = 0 ∧ f 4 = 0 ∧ f 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l3437_343754


namespace NUMINAMATH_CALUDE_least_k_value_l3437_343722

/-- The function f(t) = t² - t + 1 -/
def f (t : ℝ) : ℝ := t^2 - t + 1

/-- The property that needs to be satisfied for all x, y, z that are not all positive -/
def satisfies_property (k : ℝ) : Prop :=
  ∀ x y z : ℝ, ¬(x > 0 ∧ y > 0 ∧ z > 0) →
    k * f x * f y * f z ≥ f (x * y * z)

/-- The theorem stating that 16/9 is the least value of k satisfying the property -/
theorem least_k_value : 
  (∀ k : ℝ, k < 16/9 → ¬(satisfies_property k)) ∧ 
  satisfies_property (16/9) := by sorry

end NUMINAMATH_CALUDE_least_k_value_l3437_343722


namespace NUMINAMATH_CALUDE_expression_simplification_l3437_343756

theorem expression_simplification (a : ℝ) (h1 : a ≠ 0) (h2 : a ≠ 3) (h3 : a ≠ -3) :
  (3 / (a - 3) - a / (a + 3)) * ((a^2 - 9) / a) = (-a^2 + 6*a + 9) / a := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3437_343756


namespace NUMINAMATH_CALUDE_max_value_of_g_l3437_343737

def g (x : ℝ) : ℝ := 4 * x - x^4

theorem max_value_of_g :
  ∃ (c : ℝ), c ∈ Set.Icc 0 2 ∧ 
  (∀ x, x ∈ Set.Icc 0 2 → g x ≤ g c) ∧
  g c = 3 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_g_l3437_343737


namespace NUMINAMATH_CALUDE_max_x_value_l3437_343720

theorem max_x_value (x y z : ℝ) (sum_eq : x + y + z = 6) (sum_prod_eq : x*y + x*z + y*z = 11) : 
  x ≤ 2 ∧ ∃ (a b : ℝ), a + b + 2 = 6 ∧ 2*a + 2*b + a*b = 11 := by
  sorry

end NUMINAMATH_CALUDE_max_x_value_l3437_343720


namespace NUMINAMATH_CALUDE_geometric_arithmetic_progression_l3437_343796

theorem geometric_arithmetic_progression (a b c : ℝ) (q : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Ensure positivity for decreasing sequence
  a > b ∧ b > c →  -- Decreasing sequence
  b = a * q ∧ c = a * q^2 →  -- Geometric progression
  2 * (2020 * b / 7) = 577 * a + c / 7 →  -- Arithmetic progression
  q = 1/2 := by sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_progression_l3437_343796


namespace NUMINAMATH_CALUDE_cuts_equality_l3437_343778

/-- Represents a bagel -/
structure Bagel :=
  (intact : Bool)

/-- Represents the result of cutting a bagel -/
inductive CutResult
  | Log
  | TwoSectors

/-- Function to cut a bagel -/
def cut_bagel (b : Bagel) (result : CutResult) : Nat :=
  match result with
  | CutResult.Log => 1
  | CutResult.TwoSectors => 1

/-- Theorem stating that the number of cuts is the same for both operations -/
theorem cuts_equality (b : Bagel) :
  cut_bagel b CutResult.Log = cut_bagel b CutResult.TwoSectors :=
by
  sorry

end NUMINAMATH_CALUDE_cuts_equality_l3437_343778


namespace NUMINAMATH_CALUDE_smallest_divisible_by_15_18_20_l3437_343710

theorem smallest_divisible_by_15_18_20 : ∃ (n : ℕ), n > 0 ∧ 15 ∣ n ∧ 18 ∣ n ∧ 20 ∣ n ∧ ∀ (m : ℕ), m > 0 → 15 ∣ m → 18 ∣ m → 20 ∣ m → n ≤ m :=
by
  use 180
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_15_18_20_l3437_343710


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l3437_343764

def A : Set ℤ := {-1, 0}
def B : Set ℤ := {0, 1}

theorem union_of_A_and_B : A ∪ B = {-1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l3437_343764


namespace NUMINAMATH_CALUDE_linear_function_existence_l3437_343783

theorem linear_function_existence (k : ℝ) (h1 : k ≠ 0) 
  (h2 : ∀ x1 x2 : ℝ, x1 < x2 → k * x1 + 3 < k * x2 + 3) :
  ∃ y : ℝ, y = k * (-2) + 3 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_existence_l3437_343783


namespace NUMINAMATH_CALUDE_inequality_equivalence_fraction_comparison_l3437_343705

-- Problem 1
theorem inequality_equivalence (m x : ℝ) (h : m > 2) :
  m * x + 4 < m^2 + 2 * x ↔ x < m + 2 := by sorry

-- Problem 2
theorem fraction_comparison (x y : ℝ) (h1 : x > y) (h2 : y > 0) :
  x / (1 + x) > y / (1 + y) := by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_fraction_comparison_l3437_343705


namespace NUMINAMATH_CALUDE_rachels_age_l3437_343703

theorem rachels_age (rachel leah sam alex : ℝ) 
  (h1 : rachel = leah + 4)
  (h2 : rachel + leah = 2 * sam)
  (h3 : alex = 2 * rachel)
  (h4 : rachel + leah + sam + alex = 92) :
  rachel = 24.5 := by
  sorry

end NUMINAMATH_CALUDE_rachels_age_l3437_343703


namespace NUMINAMATH_CALUDE_bakers_sales_comparison_l3437_343702

/-- Baker's sales comparison -/
theorem bakers_sales_comparison 
  (usual_pastries : ℕ) (usual_bread : ℕ) 
  (today_pastries : ℕ) (today_bread : ℕ) 
  (pastry_price : ℕ) (bread_price : ℕ) : 
  usual_pastries = 20 → 
  usual_bread = 10 → 
  today_pastries = 14 → 
  today_bread = 25 → 
  pastry_price = 2 → 
  bread_price = 4 → 
  (today_pastries * pastry_price + today_bread * bread_price) - 
  (usual_pastries * pastry_price + usual_bread * bread_price) = 48 := by
  sorry

end NUMINAMATH_CALUDE_bakers_sales_comparison_l3437_343702


namespace NUMINAMATH_CALUDE_reflection_across_y_axis_l3437_343750

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflects a point across the y-axis -/
def reflectAcrossYAxis (p : Point) : Point :=
  { x := -p.x, y := p.y }

theorem reflection_across_y_axis :
  let P : Point := { x := 4, y := -1 }
  reflectAcrossYAxis P = { x := -4, y := -1 } := by
  sorry

end NUMINAMATH_CALUDE_reflection_across_y_axis_l3437_343750


namespace NUMINAMATH_CALUDE_sqrt_product_equals_21_l3437_343788

theorem sqrt_product_equals_21 (x : ℝ) (h_pos : x > 0) 
  (h_eq : Real.sqrt (12 * x) * Real.sqrt (5 * x) * Real.sqrt (7 * x) * Real.sqrt (21 * x) = 21) : 
  x = 21 / 97 := by
sorry

end NUMINAMATH_CALUDE_sqrt_product_equals_21_l3437_343788


namespace NUMINAMATH_CALUDE_power_product_evaluation_l3437_343732

theorem power_product_evaluation (a : ℕ) (h : a = 2) : a^3 * a^4 = 128 := by
  sorry

end NUMINAMATH_CALUDE_power_product_evaluation_l3437_343732


namespace NUMINAMATH_CALUDE_circle_C_equation_l3437_343717

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  (x - 3)^2 + (y - 2)^2 = 13

-- Define the line l
def line_l (x y : ℝ) : Prop :=
  2*x - 7*y + 8 = 0

-- Define points A and B
def point_A : ℝ × ℝ := (6, 0)
def point_B : ℝ × ℝ := (1, 5)

-- Theorem statement
theorem circle_C_equation :
  ∀ (x y : ℝ),
    (∃ (cx cy : ℝ), line_l cx cy ∧ 
      (x - cx)^2 + (y - cy)^2 = (point_A.1 - cx)^2 + (point_A.2 - cy)^2 ∧
      (x - cx)^2 + (y - cy)^2 = (point_B.1 - cx)^2 + (point_B.2 - cy)^2) →
    circle_C x y :=
by
  sorry

end NUMINAMATH_CALUDE_circle_C_equation_l3437_343717


namespace NUMINAMATH_CALUDE_employee_count_l3437_343781

theorem employee_count (avg_salary : ℕ) (salary_increase : ℕ) (manager_salary : ℕ) :
  avg_salary = 1500 →
  salary_increase = 500 →
  manager_salary = 12000 →
  ∃ n : ℕ, n * avg_salary + manager_salary = (n + 1) * (avg_salary + salary_increase) ∧ n = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_employee_count_l3437_343781


namespace NUMINAMATH_CALUDE_jacket_final_price_l3437_343721

/-- The final price of a jacket after multiple discounts -/
theorem jacket_final_price (original_price : ℝ) (discount1 discount2 discount3 : ℝ) 
  (h1 : original_price = 25)
  (h2 : discount1 = 0.40)
  (h3 : discount2 = 0.25)
  (h4 : discount3 = 0.10) : 
  original_price * (1 - discount1) * (1 - discount2) * (1 - discount3) = 10.125 := by
  sorry

end NUMINAMATH_CALUDE_jacket_final_price_l3437_343721


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l3437_343755

theorem solution_set_of_inequality (x : ℝ) :
  (2 / (x - 3) ≤ 5) ↔ (x < 3 ∨ x ≥ 17/5) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l3437_343755


namespace NUMINAMATH_CALUDE_smallest_integer_l3437_343735

theorem smallest_integer (a b : ℕ) (ha : a = 60) (h_lcm_gcd : Nat.lcm a b / Nat.gcd a b = 45) :
  b ≥ 1080 ∧ ∀ c : ℕ, c < 1080 → Nat.lcm a c / Nat.gcd a c ≠ 45 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_l3437_343735


namespace NUMINAMATH_CALUDE_sum_reciprocals_bound_l3437_343784

theorem sum_reciprocals_bound (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 2) :
  1/x + 1/y ≥ 2 ∧ ∀ M : ℝ, ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 2 ∧ 1/x + 1/y > M :=
by sorry

end NUMINAMATH_CALUDE_sum_reciprocals_bound_l3437_343784


namespace NUMINAMATH_CALUDE_circle_tangency_l3437_343733

/-- Two circles are tangent internally if the distance between their centers
    equals the difference of their radii -/
def are_tangent_internally (c1_center c2_center : ℝ × ℝ) (r1 r2 : ℝ) : Prop :=
  (c1_center.1 - c2_center.1)^2 + (c1_center.2 - c2_center.2)^2 = (r1 - r2)^2

theorem circle_tangency (m : ℝ) :
  are_tangent_internally (m, 0) (-1, 2*m) 2 3 →
  m = 0 ∨ m = -2/5 := by
  sorry

end NUMINAMATH_CALUDE_circle_tangency_l3437_343733


namespace NUMINAMATH_CALUDE_largest_multiple_of_11_below_negative_150_l3437_343799

theorem largest_multiple_of_11_below_negative_150 :
  ∀ n : ℤ, n * 11 < -150 → n * 11 ≤ -154 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_11_below_negative_150_l3437_343799


namespace NUMINAMATH_CALUDE_spinner_probability_l3437_343728

theorem spinner_probability (p_A p_B p_C p_D : ℚ) : 
  p_A = 1/4 → p_B = 1/3 → p_D = 1/6 → p_A + p_B + p_C + p_D = 1 → p_C = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_spinner_probability_l3437_343728


namespace NUMINAMATH_CALUDE_journey_time_calculation_l3437_343724

theorem journey_time_calculation (total_distance : ℝ) (speed1 : ℝ) (speed2 : ℝ) 
    (h1 : total_distance = 560)
    (h2 : speed1 = 21)
    (h3 : speed2 = 24) : 
  (total_distance / 2 / speed1) + (total_distance / 2 / speed2) = 25 := by
  sorry

end NUMINAMATH_CALUDE_journey_time_calculation_l3437_343724


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3437_343785

/-- A quadratic function f(x) = ax^2 + bx + c where a > 0 and f(1) = 0 -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem sufficient_not_necessary_condition
  (a b c : ℝ)
  (h_a_pos : a > 0)
  (h_f_1_eq_0 : QuadraticFunction a b c 1 = 0) :
  (∀ a b c, b > 2 * a → QuadraticFunction a b c (-2) < 0) ∧
  (∃ a b c, QuadraticFunction a b c (-2) < 0 ∧ b ≤ 2 * a) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3437_343785


namespace NUMINAMATH_CALUDE_existence_of_a_l3437_343714

theorem existence_of_a : ∃ a : ℝ, a ≥ 1 ∧ 
  (∀ x : ℝ, |x - 1| > a → Real.log (x^2 - 3*x + 3) > 0) ∧
  (∃ x : ℝ, Real.log (x^2 - 3*x + 3) > 0 ∧ |x - 1| ≤ a) :=
by sorry

end NUMINAMATH_CALUDE_existence_of_a_l3437_343714


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3437_343701

/-- An arithmetic sequence with common difference d -/
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r ≠ 0, ∀ n, a (n + 1) = r * a n

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ) (d : ℝ) (h_d : d ≠ 0)
  (h_sum : a 1 + a 2 + a 5 = 13)
  (h_geometric : geometric_sequence (λ n ↦ a (2 * n - 1)))
  (h_arithmetic : arithmetic_sequence a d) :
  d = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3437_343701


namespace NUMINAMATH_CALUDE_distribute_4_3_l3437_343797

/-- The number of ways to distribute n indistinguishable objects into k distinct containers,
    with each container receiving at least one object. -/
def distribute (n k : ℕ) : ℕ := sorry

/-- There are 36 ways to distribute 4 indistinguishable objects into 3 distinct containers,
    with each container receiving at least one object. -/
theorem distribute_4_3 : distribute 4 3 = 36 := by sorry

end NUMINAMATH_CALUDE_distribute_4_3_l3437_343797


namespace NUMINAMATH_CALUDE_unique_solution_l3437_343773

/-- Represents the ages of three brothers -/
structure BrothersAges where
  older : ℕ
  xiaoyong : ℕ
  younger : ℕ

/-- The conditions of the problem -/
def satisfiesConditions (ages : BrothersAges) : Prop :=
  ages.older = 20 ∧
  ages.older > ages.xiaoyong ∧
  ages.xiaoyong > ages.younger ∧
  ages.younger ≥ 1 ∧
  2 * ages.xiaoyong + 5 * ages.younger = 97

/-- The theorem stating the unique solution to the problem -/
theorem unique_solution :
  ∃! ages : BrothersAges, satisfiesConditions ages ∧ ages.xiaoyong = 16 ∧ ages.younger = 13 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l3437_343773


namespace NUMINAMATH_CALUDE_book_sale_loss_percentage_l3437_343758

/-- Proves that the loss percentage on the first book is 15% given the problem conditions --/
theorem book_sale_loss_percentage
  (total_cost : ℝ)
  (cost_book1 : ℝ)
  (gain_percentage : ℝ)
  (h1 : total_cost = 360)
  (h2 : cost_book1 = 210)
  (h3 : gain_percentage = 19)
  (h4 : ∃ (selling_price : ℝ),
    selling_price = cost_book1 * (1 - (loss_percentage / 100)) ∧
    selling_price = (total_cost - cost_book1) * (1 + (gain_percentage / 100))) :
  ∃ (loss_percentage : ℝ), loss_percentage = 15 := by
sorry


end NUMINAMATH_CALUDE_book_sale_loss_percentage_l3437_343758


namespace NUMINAMATH_CALUDE_sin_315_degrees_l3437_343789

theorem sin_315_degrees : Real.sin (315 * π / 180) = -Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_sin_315_degrees_l3437_343789


namespace NUMINAMATH_CALUDE_ceiling_floor_difference_l3437_343708

theorem ceiling_floor_difference : 
  ⌈(15 : ℚ) / 8 * (-35 : ℚ) / 4⌉ - ⌊(15 : ℚ) / 8 * ⌊(-35 : ℚ) / 4⌋⌋ = 1 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_difference_l3437_343708


namespace NUMINAMATH_CALUDE_xy_value_l3437_343738

theorem xy_value (x y : ℝ) 
  (h1 : (4:ℝ)^x / (2:ℝ)^(x+y) = 16)
  (h2 : (9:ℝ)^(x+y) / (3:ℝ)^(5*y) = 81) : 
  x * y = 32 := by
sorry

end NUMINAMATH_CALUDE_xy_value_l3437_343738


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l3437_343712

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y^2 = x - 2) ↔ x ≥ 2 := by
sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l3437_343712


namespace NUMINAMATH_CALUDE_red_balls_unchanged_l3437_343791

/-- A box containing colored balls -/
structure Box where
  red : Nat
  blue : Nat
  yellow : Nat

/-- Remove one blue ball from the box -/
def removeOneBlueBall (b : Box) : Box :=
  { red := b.red, blue := b.blue - 1, yellow := b.yellow }

theorem red_balls_unchanged (initial : Box) (h : initial.blue ≥ 1) :
  (removeOneBlueBall initial).red = initial.red :=
by sorry

end NUMINAMATH_CALUDE_red_balls_unchanged_l3437_343791


namespace NUMINAMATH_CALUDE_job_completion_time_l3437_343718

theorem job_completion_time 
  (initial_men : ℕ) 
  (initial_days : ℕ) 
  (new_men : ℕ) 
  (prep_days : ℕ) 
  (h1 : initial_men = 10) 
  (h2 : initial_days = 15) 
  (h3 : new_men = 15) 
  (h4 : prep_days = 2) : 
  (initial_men * initial_days) / new_men + prep_days = 12 := by
sorry

end NUMINAMATH_CALUDE_job_completion_time_l3437_343718


namespace NUMINAMATH_CALUDE_inequality_proof_l3437_343734

theorem inequality_proof (a b c d : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) 
  (h_sum : a + b + c + d = 3) : 
  1/a^3 + 1/b^3 + 1/c^3 + 1/d^3 ≤ 1/(a^3 * b^3 * c^3 * d^3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3437_343734


namespace NUMINAMATH_CALUDE_complex_number_opposites_l3437_343779

theorem complex_number_opposites (b : ℝ) : 
  let z : ℂ := (2 - b * Complex.I) / (1 + 2 * Complex.I)
  (z.re = -z.im) → b = -2/3 := by
sorry

end NUMINAMATH_CALUDE_complex_number_opposites_l3437_343779


namespace NUMINAMATH_CALUDE_smallest_solution_floor_equation_l3437_343745

theorem smallest_solution_floor_equation :
  ∃ x : ℝ, (∀ y : ℝ, (⌊y^2⌋ : ℤ) - (⌊y⌋ : ℤ)^2 = 21 → x ≤ y) ∧
            (⌊x^2⌋ : ℤ) - (⌊x⌋ : ℤ)^2 = 21 ∧
            x > 11.5 ∧ x < 11.6 :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_floor_equation_l3437_343745


namespace NUMINAMATH_CALUDE_sum_of_powers_of_3_mod_5_l3437_343723

def sum_of_powers (base : ℕ) (exponent : ℕ) : ℕ :=
  Finset.sum (Finset.range (exponent + 1)) (fun i => base ^ i)

theorem sum_of_powers_of_3_mod_5 :
  sum_of_powers 3 2023 % 5 = 3 := by sorry

end NUMINAMATH_CALUDE_sum_of_powers_of_3_mod_5_l3437_343723


namespace NUMINAMATH_CALUDE_product_divisible_by_nine_l3437_343715

theorem product_divisible_by_nine : ∃ k : ℤ, 12345 * 54321 = 9 * k := by
  sorry

end NUMINAMATH_CALUDE_product_divisible_by_nine_l3437_343715


namespace NUMINAMATH_CALUDE_gcd_1729_867_l3437_343762

theorem gcd_1729_867 : Nat.gcd 1729 867 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1729_867_l3437_343762


namespace NUMINAMATH_CALUDE_quadratic_solution_sum_l3437_343786

theorem quadratic_solution_sum (a b : ℝ) : 
  (∀ x : ℂ, (3 * x^2 + 8 = 4 * x - 7) ↔ (x = a + b * I ∨ x = a - b * I)) →
  a + b^2 = 47/9 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_sum_l3437_343786


namespace NUMINAMATH_CALUDE_dodecahedron_path_count_l3437_343795

/-- Represents a path on a dodecahedron -/
structure DodecahedronPath where
  start : (Int × Int × Int)
  finish : (Int × Int × Int)
  length : Nat
  visitsAllCorners : Bool
  cannotReturnToStart : Bool

/-- The number of valid paths on a dodecahedron meeting specific conditions -/
def countValidPaths : Nat :=
  sorry

theorem dodecahedron_path_count :
  let validPath : DodecahedronPath :=
    { start := (0, 0, 0),
      finish := (1, 1, 0),
      length := 19,
      visitsAllCorners := true,
      cannotReturnToStart := true }
  countValidPaths = 90 := by
  sorry

end NUMINAMATH_CALUDE_dodecahedron_path_count_l3437_343795


namespace NUMINAMATH_CALUDE_parallel_lines_theorem_l3437_343793

/-- Two lines are parallel if their slopes are equal -/
def parallel_lines (m1 m2 : ℝ) : Prop := m1 = m2

/-- Slope of the line ax + y - 1 - a = 0 -/
def slope1 (a : ℝ) : ℝ := -a

/-- Slope of the line x - 1/2y = 0 -/
def slope2 : ℝ := 2

/-- Theorem: If ax + y - 1 - a = 0 is parallel to x - 1/2y = 0, then a = -2 -/
theorem parallel_lines_theorem (a : ℝ) : 
  parallel_lines (slope1 a) slope2 → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_theorem_l3437_343793


namespace NUMINAMATH_CALUDE_physics_to_music_ratio_l3437_343743

/-- Proves that the ratio of physics marks to music marks is 1:2 given the marks in other subjects and total marks -/
theorem physics_to_music_ratio (science music social_studies total : ℕ) (physics : ℚ) :
  science = 70 →
  music = 80 →
  social_studies = 85 →
  total = 275 →
  physics = music * (1 / 2) →
  science + music + social_studies + physics = total →
  physics / music = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_physics_to_music_ratio_l3437_343743


namespace NUMINAMATH_CALUDE_bob_distance_when_meeting_l3437_343709

/-- Prove that Bob walked 35 miles when he met Yolanda, given the following conditions:
  - The total distance between X and Y is 65 miles
  - Yolanda starts walking from X to Y
  - Bob starts walking from Y to X one hour after Yolanda
  - Yolanda's walking rate is 5 miles per hour
  - Bob's walking rate is 7 miles per hour
-/
theorem bob_distance_when_meeting (total_distance : ℝ) (yolanda_rate : ℝ) (bob_rate : ℝ)
  (h1 : total_distance = 65)
  (h2 : yolanda_rate = 5)
  (h3 : bob_rate = 7) :
  let time_to_meet := (total_distance - yolanda_rate) / (yolanda_rate + bob_rate)
  bob_rate * time_to_meet = 35 := by
  sorry

end NUMINAMATH_CALUDE_bob_distance_when_meeting_l3437_343709


namespace NUMINAMATH_CALUDE_orchids_unchanged_l3437_343746

/-- The number of orchids in a vase remains unchanged when only roses are added --/
theorem orchids_unchanged 
  (initial_roses : ℕ) 
  (initial_orchids : ℕ) 
  (final_roses : ℕ) 
  (roses_added : ℕ) : 
  initial_roses = 15 → 
  initial_orchids = 62 → 
  final_roses = 17 → 
  roses_added = 2 → 
  final_roses = initial_roses + roses_added → 
  initial_orchids = 62 := by
sorry

end NUMINAMATH_CALUDE_orchids_unchanged_l3437_343746


namespace NUMINAMATH_CALUDE_parabolas_intersection_l3437_343776

/-- The x-coordinates of the intersection points of two parabolas -/
def intersection_x : Set ℝ :=
  {x | 3 * x^2 - 4 * x + 2 = -x^2 + 2 * x + 3}

/-- The y-coordinates of the intersection points of two parabolas -/
def intersection_y (x : ℝ) : ℝ :=
  3 * x^2 - 4 * x + 2

/-- The first parabola -/
def parabola1 (x : ℝ) : ℝ :=
  3 * x^2 - 4 * x + 2

/-- The second parabola -/
def parabola2 (x : ℝ) : ℝ :=
  -x^2 + 2 * x + 3

theorem parabolas_intersection :
  intersection_x = {(3 - Real.sqrt 13) / 4, (3 + Real.sqrt 13) / 4} ∧
  ∀ x ∈ intersection_x, intersection_y x = (74 + 14 * Real.sqrt 13 * (if x < 0 then -1 else 1)) / 16 ∧
  ∀ x : ℝ, parabola1 x = parabola2 x ↔ x ∈ intersection_x :=
by sorry


end NUMINAMATH_CALUDE_parabolas_intersection_l3437_343776


namespace NUMINAMATH_CALUDE_elderly_selected_l3437_343706

/-- Given a population with the following properties:
  - Total population of 1500
  - Divided into three equal groups (children, elderly, middle-aged)
  - 60 people are selected using stratified sampling
  This theorem proves that the number of elderly people selected is 20. -/
theorem elderly_selected (total_population : ℕ) (sample_size : ℕ) (num_groups : ℕ) :
  total_population = 1500 →
  sample_size = 60 →
  num_groups = 3 →
  (total_population / num_groups : ℚ) * (sample_size / total_population : ℚ) = 20 := by
  sorry

end NUMINAMATH_CALUDE_elderly_selected_l3437_343706


namespace NUMINAMATH_CALUDE_function_k_value_l3437_343772

theorem function_k_value (f : ℝ → ℝ) (k : ℝ) :
  (∀ x, f x = k * x + 1) →
  f 2 = 3 →
  k = 1 := by
sorry

end NUMINAMATH_CALUDE_function_k_value_l3437_343772
