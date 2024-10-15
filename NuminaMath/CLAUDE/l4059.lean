import Mathlib

namespace NUMINAMATH_CALUDE_fraction_division_equals_seventeen_l4059_405934

theorem fraction_division_equals_seventeen :
  (-4/9 + 1/6 - 2/3) / (-1/18) = 17 := by sorry

end NUMINAMATH_CALUDE_fraction_division_equals_seventeen_l4059_405934


namespace NUMINAMATH_CALUDE_negation_disjunction_true_l4059_405943

theorem negation_disjunction_true (p q : Prop) : 
  (p ∧ q) = False → (¬p ∨ ¬q) = True := by sorry

end NUMINAMATH_CALUDE_negation_disjunction_true_l4059_405943


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_problem_l4059_405930

theorem arithmetic_geometric_sequence_problem (a b : ℕ → ℝ) (d : ℝ) :
  (∀ n, a (n + 1) - a n = d) →  -- a_n is arithmetic with common difference d
  d ≠ 0 →  -- d is not equal to 0
  a 2046 + a 1978 - (a 2012)^2 = 0 →  -- given condition
  (∃ r, ∀ n, b (n + 1) = r * b n) →  -- b_n is geometric
  b 2012 = a 2012 →  -- given condition
  b 2010 * b 2014 = 4 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_problem_l4059_405930


namespace NUMINAMATH_CALUDE_value_of_expression_l4059_405987

theorem value_of_expression (x y : ℤ) (h1 : x = -1) (h2 : y = 4) : 2 * (x + y) = 6 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l4059_405987


namespace NUMINAMATH_CALUDE_robie_chocolates_l4059_405971

/-- The number of bags of chocolates Robie has after her purchases and giveaway. -/
def final_bags (initial : ℕ) (given_away : ℕ) (bought_later : ℕ) : ℕ :=
  initial - given_away + bought_later

/-- Theorem stating that Robie ends up with 4 bags of chocolates. -/
theorem robie_chocolates : final_bags 3 2 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_robie_chocolates_l4059_405971


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l4059_405950

def M : Set ℤ := {0, 1, 2}
def N : Set ℤ := {x | -1 ≤ x ∧ x ≤ 1}

theorem intersection_of_M_and_N : M ∩ N = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l4059_405950


namespace NUMINAMATH_CALUDE_remaining_money_l4059_405998

-- Define the plant sales
def orchid_sales : ℕ := 30
def orchid_price : ℕ := 50
def money_plant_sales : ℕ := 25
def money_plant_price : ℕ := 30
def bonsai_sales : ℕ := 15
def bonsai_price : ℕ := 75
def cacti_sales : ℕ := 20
def cacti_price : ℕ := 20

-- Define the expenses
def num_workers : ℕ := 4
def worker_pay : ℕ := 60
def new_pots_cost : ℕ := 250
def utility_bill : ℕ := 200
def tax : ℕ := 500

-- Calculate total earnings
def total_earnings : ℕ := 
  orchid_sales * orchid_price + 
  money_plant_sales * money_plant_price + 
  bonsai_sales * bonsai_price + 
  cacti_sales * cacti_price

-- Calculate total expenses
def total_expenses : ℕ := 
  num_workers * worker_pay + 
  new_pots_cost + 
  utility_bill + 
  tax

-- Theorem to prove
theorem remaining_money : 
  total_earnings - total_expenses = 2585 := by
  sorry

end NUMINAMATH_CALUDE_remaining_money_l4059_405998


namespace NUMINAMATH_CALUDE_no_solution_for_functional_equation_l4059_405952

theorem no_solution_for_functional_equation :
  ¬∃ (f : ℕ → ℕ), ∀ (x : ℕ), f (f x) = x + 1 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_functional_equation_l4059_405952


namespace NUMINAMATH_CALUDE_apples_in_basket_l4059_405970

/-- The number of apples left in a basket after removals --/
def applesLeft (initial : ℕ) (rickiRemoves : ℕ) : ℕ :=
  initial - rickiRemoves - (2 * rickiRemoves)

/-- Theorem stating the number of apples left in the basket --/
theorem apples_in_basket : applesLeft 74 14 = 32 := by
  sorry

end NUMINAMATH_CALUDE_apples_in_basket_l4059_405970


namespace NUMINAMATH_CALUDE_square_of_107_l4059_405968

theorem square_of_107 : (107 : ℕ)^2 = 11449 := by
  sorry

end NUMINAMATH_CALUDE_square_of_107_l4059_405968


namespace NUMINAMATH_CALUDE_fraction_inequality_solution_set_l4059_405939

theorem fraction_inequality_solution_set (x : ℝ) (h : x ≠ 0) :
  1 / x ≤ 1 ↔ x ∈ Set.Ioo 0 1 ∪ Set.Ici 1 := by sorry

end NUMINAMATH_CALUDE_fraction_inequality_solution_set_l4059_405939


namespace NUMINAMATH_CALUDE_committee_permutations_count_l4059_405989

/-- The number of distinct permutations of the letters in "COMMITTEE" -/
def committee_permutations : ℕ := sorry

/-- The total number of letters in "COMMITTEE" -/
def total_letters : ℕ := 8

/-- The number of occurrences of each letter in "COMMITTEE" -/
def letter_occurrences : List ℕ := [2, 2, 3, 1, 1]

theorem committee_permutations_count : 
  committee_permutations = (total_letters.factorial) / (letter_occurrences.map Nat.factorial).prod := by
  sorry

end NUMINAMATH_CALUDE_committee_permutations_count_l4059_405989


namespace NUMINAMATH_CALUDE_gift_wrapping_theorem_l4059_405923

/-- Cagney's gift wrapping rate in gifts per second -/
def cagney_rate : ℚ := 1 / 45

/-- Lacey's gift wrapping rate in gifts per second -/
def lacey_rate : ℚ := 1 / 60

/-- Total time available in seconds -/
def total_time : ℚ := 15 * 60

/-- The number of gifts that can be wrapped collectively -/
def total_gifts : ℕ := 35

theorem gift_wrapping_theorem :
  (cagney_rate + lacey_rate) * total_time = total_gifts := by
  sorry

end NUMINAMATH_CALUDE_gift_wrapping_theorem_l4059_405923


namespace NUMINAMATH_CALUDE_number_exceeding_fraction_l4059_405999

theorem number_exceeding_fraction (x : ℚ) : x = (3 / 8) * x + 35 → x = 56 := by
  sorry

end NUMINAMATH_CALUDE_number_exceeding_fraction_l4059_405999


namespace NUMINAMATH_CALUDE_city_population_l4059_405995

theorem city_population (population_percentage : Real) (partial_population : ℕ) (total_population : ℕ) : 
  population_percentage = 0.85 →
  partial_population = 85000 →
  population_percentage * (total_population : Real) = partial_population →
  total_population = 100000 :=
by
  sorry

end NUMINAMATH_CALUDE_city_population_l4059_405995


namespace NUMINAMATH_CALUDE_kabadi_kho_kho_intersection_no_players_in_both_games_l4059_405988

/-- Proves that the number of people playing both kabadi and kho kho is 0 -/
theorem kabadi_kho_kho_intersection (total : ℕ) (kabadi : ℕ) (kho_kho_only : ℕ)
  (h_total : total = 30)
  (h_kabadi : kabadi = 10)
  (h_kho_kho_only : kho_kho_only = 20) :
  total = kabadi + kho_kho_only :=
by sorry

/-- The number of people playing both kabadi and kho kho -/
def both_games (total : ℕ) (kabadi : ℕ) (kho_kho_only : ℕ) : ℕ :=
  kabadi - (total - kho_kho_only)

theorem no_players_in_both_games (total : ℕ) (kabadi : ℕ) (kho_kho_only : ℕ)
  (h_total : total = 30)
  (h_kabadi : kabadi = 10)
  (h_kho_kho_only : kho_kho_only = 20) :
  both_games total kabadi kho_kho_only = 0 :=
by sorry

end NUMINAMATH_CALUDE_kabadi_kho_kho_intersection_no_players_in_both_games_l4059_405988


namespace NUMINAMATH_CALUDE_polygon_interior_angles_l4059_405929

theorem polygon_interior_angles (P : ℕ) (h1 : P > 2) : 
  (∃ (a d : ℝ), 
    a = 20 ∧ 
    a + (P - 1) * d = 160 ∧ 
    (P / 2 : ℝ) * (a + (a + (P - 1) * d)) = 180 * (P - 2)) → 
  P = 4 := by
sorry

end NUMINAMATH_CALUDE_polygon_interior_angles_l4059_405929


namespace NUMINAMATH_CALUDE_pokemon_cards_remaining_l4059_405912

theorem pokemon_cards_remaining (initial : ℕ) (given_away : ℕ) (remaining : ℕ) : 
  initial = 13 → given_away = 9 → remaining = initial - given_away → remaining = 4 := by
  sorry

end NUMINAMATH_CALUDE_pokemon_cards_remaining_l4059_405912


namespace NUMINAMATH_CALUDE_dave_guitar_strings_l4059_405962

/-- The number of guitar strings Dave breaks per night -/
def strings_per_night : ℕ := 2

/-- The number of shows Dave performs per week -/
def shows_per_week : ℕ := 6

/-- The number of weeks Dave performs -/
def total_weeks : ℕ := 12

/-- The total number of guitar strings Dave needs to replace -/
def total_strings : ℕ := strings_per_night * shows_per_week * total_weeks

theorem dave_guitar_strings :
  total_strings = 144 := by sorry

end NUMINAMATH_CALUDE_dave_guitar_strings_l4059_405962


namespace NUMINAMATH_CALUDE_equation_to_lines_l4059_405928

/-- The set of points satisfying the given equation is equivalent to the union of two lines -/
theorem equation_to_lines : 
  ∀ x y : ℝ, (2*x^2 + y^2 + 3*x*y + 3*x + y = 2) ↔ 
  (y = -x - 2 ∨ y = -2*x + 1) := by sorry

end NUMINAMATH_CALUDE_equation_to_lines_l4059_405928


namespace NUMINAMATH_CALUDE_max_inscribed_circle_area_l4059_405918

/-- Definition of the ellipse -/
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

/-- Left focus of the ellipse -/
def F1 : ℝ × ℝ := (-1, 0)

/-- Right focus of the ellipse -/
def F2 : ℝ × ℝ := (1, 0)

/-- A line passing through the right focus -/
def line_through_F2 (m : ℝ) (y : ℝ) : ℝ := m * y + 1

/-- Points of intersection between the line and the ellipse -/
def intersection_points (m : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ y, ellipse (line_through_F2 m y) y ∧ p = (line_through_F2 m y, y)}

/-- Triangle formed by F1 and two intersection points -/
def triangle_F1PQ (m : ℝ) : Set (ℝ × ℝ) :=
  {F1} ∪ intersection_points m

/-- The inscribed circle of a triangle -/
def inscribed_circle (t : Set (ℝ × ℝ)) : Set (ℝ × ℝ) :=
  sorry  -- Definition of inscribed circle

/-- The area of a circle -/
def circle_area (c : Set (ℝ × ℝ)) : ℝ :=
  sorry  -- Definition of circle area

/-- The theorem to be proved -/
theorem max_inscribed_circle_area :
  ∃ (m : ℝ), ∀ (n : ℝ),
    circle_area (inscribed_circle (triangle_F1PQ m)) ≥
    circle_area (inscribed_circle (triangle_F1PQ n)) ∧
    circle_area (inscribed_circle (triangle_F1PQ m)) = 9 * Real.pi / 16 :=
sorry

end NUMINAMATH_CALUDE_max_inscribed_circle_area_l4059_405918


namespace NUMINAMATH_CALUDE_textbook_page_ratio_l4059_405947

/-- Proves the ratio of math textbook pages to the sum of history and geography textbook pages -/
theorem textbook_page_ratio : ∀ (history geography math science : ℕ) (total : ℕ),
  history = 160 →
  geography = history + 70 →
  science = 2 * history →
  total = history + geography + math + science →
  total = 905 →
  (math : ℚ) / (history + geography : ℚ) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_textbook_page_ratio_l4059_405947


namespace NUMINAMATH_CALUDE_asymptote_coincidence_l4059_405951

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 3 - y^2 = 1

-- Define the parabola
def parabola (p x y : ℝ) : Prop := y^2 = 2 * p * x

-- Define the asymptote of the hyperbola
def hyperbola_asymptote (x : ℝ) : Prop := x = -3/2 ∨ x = 3/2

-- Define the asymptote of the parabola
def parabola_asymptote (p x : ℝ) : Prop := x = -p/2

-- State the theorem
theorem asymptote_coincidence (p : ℝ) :
  (p > 0) →
  (∃ x : ℝ, hyperbola_asymptote x ∧ parabola_asymptote p x) →
  p = 3 :=
sorry

end NUMINAMATH_CALUDE_asymptote_coincidence_l4059_405951


namespace NUMINAMATH_CALUDE_horror_movie_tickets_l4059_405953

theorem horror_movie_tickets (romance_tickets horror_tickets : ℕ) : 
  romance_tickets = 25 →
  horror_tickets = 3 * romance_tickets + 18 →
  horror_tickets = 93 := by
sorry

end NUMINAMATH_CALUDE_horror_movie_tickets_l4059_405953


namespace NUMINAMATH_CALUDE_cubic_fraction_zero_l4059_405982

theorem cubic_fraction_zero (a b c : ℝ) (h : a + b + c ≠ 0) :
  ((a^3 - b^3)^2 + (b^3 - c^3)^2 + (c^3 - a^3)^2) / (a^3 + b^3 + c^3 - 3*a*b*c) = 0 :=
by sorry

end NUMINAMATH_CALUDE_cubic_fraction_zero_l4059_405982


namespace NUMINAMATH_CALUDE_first_blend_cost_is_correct_l4059_405907

/-- The cost of the first blend of coffee in dollars per pound -/
def first_blend_cost : ℝ := 9

/-- The cost of the second blend of coffee in dollars per pound -/
def second_blend_cost : ℝ := 8

/-- The total weight of the mixed blend in pounds -/
def total_blend_weight : ℝ := 20

/-- The selling price of the mixed blend in dollars per pound -/
def mixed_blend_price : ℝ := 8.4

/-- The weight of the first blend used in the mixture in pounds -/
def first_blend_weight : ℝ := 8

/-- Theorem stating that the cost of the first blend is correct given the conditions -/
theorem first_blend_cost_is_correct :
  first_blend_cost * first_blend_weight + 
  second_blend_cost * (total_blend_weight - first_blend_weight) = 
  mixed_blend_price * total_blend_weight := by
  sorry

end NUMINAMATH_CALUDE_first_blend_cost_is_correct_l4059_405907


namespace NUMINAMATH_CALUDE_power_sum_theorem_l4059_405961

theorem power_sum_theorem (a : ℝ) (m : ℕ) (h : a^m = 2) : a^(2*m) + a^(3*m) = 12 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_theorem_l4059_405961


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l4059_405942

theorem sqrt_equation_solution :
  ∃! z : ℝ, Real.sqrt (8 + 3 * z) = 14 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l4059_405942


namespace NUMINAMATH_CALUDE_inverse_proportion_y_relationship_l4059_405933

theorem inverse_proportion_y_relationship (x₁ x₂ y₁ y₂ : ℝ) :
  x₁ > x₂ → x₂ > 0 → y₁ = -3 / x₁ → y₂ = -3 / x₂ → y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_y_relationship_l4059_405933


namespace NUMINAMATH_CALUDE_three_statements_true_l4059_405994

open Function

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties
def isOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def isPeriodic (f : ℝ → ℝ) : Prop := ∃ T ≠ 0, ∀ x, f (x + T) = f x
def isMonoDecreasing (f : ℝ → ℝ) : Prop := ∀ x y, x < y → f x > f y
def hasInverse (f : ℝ → ℝ) : Prop := ∃ g : ℝ → ℝ, LeftInverse g f ∧ RightInverse g f

-- The main theorem
theorem three_statements_true (f : ℝ → ℝ) : 
  (isOdd f → isOdd (f ∘ f)) ∧
  (isPeriodic f → isPeriodic (f ∘ f)) ∧
  ¬(isMonoDecreasing f → isMonoDecreasing (f ∘ f)) ∧
  (hasInverse f → (∃ x, f x = x)) :=
sorry

end NUMINAMATH_CALUDE_three_statements_true_l4059_405994


namespace NUMINAMATH_CALUDE_parabola_translation_l4059_405945

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Translates a parabola horizontally and vertically -/
def translate (p : Parabola) (h : ℝ) (k : ℝ) : Parabola :=
  { a := p.a
    b := -2 * p.a * h + p.b
    c := p.a * h^2 - p.b * h + p.c - k }

theorem parabola_translation (x y : ℝ) :
  let original := Parabola.mk 8 0 0
  let translated := translate original 3 (-5)
  y = 8 * x^2 → y = translated.a * (x - 3)^2 + translated.b * (x - 3) + translated.c :=
by sorry

end NUMINAMATH_CALUDE_parabola_translation_l4059_405945


namespace NUMINAMATH_CALUDE_ellipse_axis_endpoint_distance_l4059_405964

/-- Given an ellipse with equation 4(x-2)^2 + 16y^2 = 64, 
    the distance between an endpoint of its major axis 
    and an endpoint of its minor axis is 2√5. -/
theorem ellipse_axis_endpoint_distance : 
  ∃ (C D : ℝ × ℝ),
    (∀ (x y : ℝ), 4 * (x - 2)^2 + 16 * y^2 = 64 → 
      ((x = C.1 ∧ y = C.2) ∨ (x = D.1 ∧ y = D.2))) →
    (C.1 - 2)^2 / 16 + C.2^2 / 4 = 1 →
    (D.1 - 2)^2 / 16 + D.2^2 / 4 = 1 →
    C.1 ≠ D.1 →
    C.2 ≠ D.2 →
    Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) = 2 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_axis_endpoint_distance_l4059_405964


namespace NUMINAMATH_CALUDE_bakers_cakes_l4059_405910

/-- Baker's cake selling problem -/
theorem bakers_cakes (initial_cakes : ℕ) (cakes_left : ℕ) (h1 : initial_cakes = 48) (h2 : cakes_left = 4) :
  initial_cakes - cakes_left = 44 := by
  sorry

end NUMINAMATH_CALUDE_bakers_cakes_l4059_405910


namespace NUMINAMATH_CALUDE_hannah_strawberries_l4059_405957

theorem hannah_strawberries (x : ℕ) : 
  (30 * x - 20 - 30 = 100) → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_hannah_strawberries_l4059_405957


namespace NUMINAMATH_CALUDE_smallest_integer_y_l4059_405967

def is_integer (x : ℚ) : Prop := ∃ n : ℤ, x = n

def expression (y : ℤ) : ℚ := (y^2 - 3*y + 11) / (y - 5)

theorem smallest_integer_y : 
  (∀ y : ℤ, y < 6 → ¬(is_integer (expression y))) ∧ 
  (is_integer (expression 6)) := by sorry

end NUMINAMATH_CALUDE_smallest_integer_y_l4059_405967


namespace NUMINAMATH_CALUDE_expansion_coefficient_x_squared_l4059_405949

/-- The coefficient of x^2 in the expansion of (1 + x + x^(1/2018))^10 -/
def coefficient_x_squared : ℕ :=
  Nat.choose 10 2

theorem expansion_coefficient_x_squared :
  coefficient_x_squared = 45 := by sorry

end NUMINAMATH_CALUDE_expansion_coefficient_x_squared_l4059_405949


namespace NUMINAMATH_CALUDE_mary_peaches_cost_l4059_405913

/-- The amount Mary paid for berries in dollars -/
def berries_cost : ℚ := 7.19

/-- The total amount Mary paid with in dollars -/
def total_paid : ℚ := 20

/-- The amount Mary received as change in dollars -/
def change_received : ℚ := 5.98

/-- The amount Mary paid for peaches in dollars -/
def peaches_cost : ℚ := total_paid - change_received - berries_cost

theorem mary_peaches_cost : peaches_cost = 6.83 := by sorry

end NUMINAMATH_CALUDE_mary_peaches_cost_l4059_405913


namespace NUMINAMATH_CALUDE_multiplication_mistake_correction_l4059_405984

theorem multiplication_mistake_correction (α : ℝ) :
  1.2 * α = 1.23 * α - 0.3 → 1.23 * α = 111 := by
sorry

end NUMINAMATH_CALUDE_multiplication_mistake_correction_l4059_405984


namespace NUMINAMATH_CALUDE_stating_chess_team_arrangements_l4059_405960

/-- Represents the number of boys on the chess team -/
def num_boys : Nat := 3

/-- Represents the number of girls on the chess team -/
def num_girls : Nat := 3

/-- Represents the total number of students on the chess team -/
def total_students : Nat := num_boys + num_girls

/-- 
Represents the number of ways to arrange the chess team in a row 
such that all boys are at the ends and exactly one boy is in the middle
-/
def num_arrangements : Nat := 36

/-- 
Theorem stating that the number of arrangements of the chess team
satisfying the given conditions is equal to 36
-/
theorem chess_team_arrangements : 
  (num_boys = 3 ∧ num_girls = 3) → num_arrangements = 36 := by
  sorry

end NUMINAMATH_CALUDE_stating_chess_team_arrangements_l4059_405960


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l4059_405920

theorem complex_fraction_simplification :
  ∀ (i : ℂ), i^2 = -1 →
  (5 + 7*i) / (3 - 4*i) = (43 : ℚ)/25 + (41 : ℚ)/25 * i := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l4059_405920


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l4059_405922

theorem quadratic_two_distinct_roots (a b c : ℝ) (h : 2016 + a^2 + a*c < a*b) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a*x₁^2 + b*x₁ + c = 0 ∧ a*x₂^2 + b*x₂ + c = 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l4059_405922


namespace NUMINAMATH_CALUDE_polynomial_equality_unique_solution_l4059_405992

theorem polynomial_equality_unique_solution :
  ∃! (a b c : ℤ), ∀ (x : ℝ), (x - a) * (x - 11) + 2 = (x + b) * (x + c) ∧
  a = 13 ∧ b = -13 ∧ c = -12 :=
sorry

end NUMINAMATH_CALUDE_polynomial_equality_unique_solution_l4059_405992


namespace NUMINAMATH_CALUDE_preimage_of_4_3_l4059_405926

/-- The mapping f from R² to R² -/
def f (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 + 2*p.2, 2*p.1 - p.2)

/-- Theorem: The pre-image of (4,3) under the mapping f is (2,1) -/
theorem preimage_of_4_3 :
  f (2, 1) = (4, 3) := by
  sorry

end NUMINAMATH_CALUDE_preimage_of_4_3_l4059_405926


namespace NUMINAMATH_CALUDE_percentage_female_officers_on_duty_l4059_405990

def total_officers_on_duty : ℕ := 204
def female_ratio_on_duty : ℚ := 1/2
def total_female_officers : ℕ := 600

theorem percentage_female_officers_on_duty :
  (total_officers_on_duty * female_ratio_on_duty) / total_female_officers * 100 = 17 := by
  sorry

end NUMINAMATH_CALUDE_percentage_female_officers_on_duty_l4059_405990


namespace NUMINAMATH_CALUDE_gear_speed_proportion_l4059_405980

/-- Represents a gear with a number of teeth and angular speed -/
structure Gear where
  teeth : ℕ
  speed : ℝ

/-- Represents a system of four meshed gears -/
structure GearSystem where
  A : Gear
  B : Gear
  C : Gear
  D : Gear
  x : ℕ
  y : ℕ
  z : ℕ
  w : ℕ
  mesh_correctly : A.teeth * A.speed = B.teeth * B.speed ∧
                   B.teeth * B.speed = C.teeth * C.speed ∧
                   C.teeth * C.speed = D.teeth * D.speed

/-- Theorem stating the proportion of angular speeds in a gear system -/
theorem gear_speed_proportion (gs : GearSystem)
  (hA : gs.A.teeth = 10 * gs.x)
  (hB : gs.B.teeth = 15 * gs.y)
  (hC : gs.C.teeth = 12 * gs.z)
  (hD : gs.D.teeth = 20 * gs.w) :
  ∃ (k : ℝ), k > 0 ∧
    gs.A.speed = k * (12 * gs.y * gs.z * gs.w : ℝ) ∧
    gs.B.speed = k * (8 * gs.x * gs.z * gs.w : ℝ) ∧
    gs.C.speed = k * (10 * gs.x * gs.y * gs.w : ℝ) ∧
    gs.D.speed = k * (6 * gs.x * gs.y * gs.z : ℝ) :=
sorry

end NUMINAMATH_CALUDE_gear_speed_proportion_l4059_405980


namespace NUMINAMATH_CALUDE_english_spanish_difference_l4059_405935

/-- Ryan's learning schedule for three days -/
structure LearningSchedule :=
  (day1_english : ℕ) (day1_chinese : ℕ) (day1_spanish : ℕ)
  (day2_english : ℕ) (day2_chinese : ℕ) (day2_spanish : ℕ)
  (day3_english : ℕ) (day3_chinese : ℕ) (day3_spanish : ℕ)

/-- Ryan's actual learning schedule -/
def ryans_schedule : LearningSchedule :=
  { day1_english := 7, day1_chinese := 2, day1_spanish := 4,
    day2_english := 6, day2_chinese := 3, day2_spanish := 5,
    day3_english := 8, day3_chinese := 1, day3_spanish := 3 }

/-- Calculate the total hours spent on a language over three days -/
def total_hours (schedule : LearningSchedule) (language : String) : ℕ :=
  match language with
  | "English" => schedule.day1_english + schedule.day2_english + schedule.day3_english
  | "Spanish" => schedule.day1_spanish + schedule.day2_spanish + schedule.day3_spanish
  | _ => 0

/-- Theorem: Ryan spends 9 more hours on English than Spanish -/
theorem english_spanish_difference :
  total_hours ryans_schedule "English" - total_hours ryans_schedule "Spanish" = 9 := by
  sorry

end NUMINAMATH_CALUDE_english_spanish_difference_l4059_405935


namespace NUMINAMATH_CALUDE_right_triangle_arithmetic_sides_ratio_l4059_405946

-- Define a right-angled triangle with sides forming an arithmetic sequence
structure RightTriangleArithmeticSides where
  a : ℝ
  b : ℝ
  c : ℝ
  right_angle : c^2 = a^2 + b^2
  arithmetic_sequence : ∃ d : ℝ, b = a + d ∧ c = b + d

-- Theorem statement
theorem right_triangle_arithmetic_sides_ratio 
  (t : RightTriangleArithmeticSides) : 
  ∃ k : ℝ, t.a = 3*k ∧ t.b = 4*k ∧ t.c = 5*k := by
sorry

end NUMINAMATH_CALUDE_right_triangle_arithmetic_sides_ratio_l4059_405946


namespace NUMINAMATH_CALUDE_closest_integer_to_two_plus_sqrt_six_l4059_405979

theorem closest_integer_to_two_plus_sqrt_six (x : ℝ) : 
  x = 2 + Real.sqrt 6 → 
  ∃ (n : ℕ), n = 4 ∧ ∀ (m : ℕ), m ≠ 4 → |x - ↑n| < |x - ↑m| := by
  sorry

end NUMINAMATH_CALUDE_closest_integer_to_two_plus_sqrt_six_l4059_405979


namespace NUMINAMATH_CALUDE_number_divided_by_two_equals_number_minus_five_l4059_405985

theorem number_divided_by_two_equals_number_minus_five : ∃! x : ℝ, x / 2 = x - 5 := by
  sorry

end NUMINAMATH_CALUDE_number_divided_by_two_equals_number_minus_five_l4059_405985


namespace NUMINAMATH_CALUDE_correct_answer_l4059_405974

theorem correct_answer (x : ℤ) (h : x - 8 = 32) : x + 8 = 48 := by
  sorry

end NUMINAMATH_CALUDE_correct_answer_l4059_405974


namespace NUMINAMATH_CALUDE_solution_satisfies_system_l4059_405924

theorem solution_satisfies_system :
  let x : ℝ := 0
  let y : ℝ := 6
  let z : ℝ := 7
  let u : ℝ := 3
  let v : ℝ := -1
  (x - y + z = 1) ∧
  (y - z + u = 2) ∧
  (z - u + v = 3) ∧
  (u - v + x = 4) ∧
  (v - x + y = 5) := by
  sorry

end NUMINAMATH_CALUDE_solution_satisfies_system_l4059_405924


namespace NUMINAMATH_CALUDE_star_equality_implies_x_eight_l4059_405906

/-- Binary operation ★ on ordered pairs of integers -/
def star (a b c d : ℤ) : ℤ × ℤ := (a - c, b + d)

/-- Theorem stating that if (5,7) ★ (1,3) = (x,y) ★ (4,5), then x = 8 -/
theorem star_equality_implies_x_eight (x y : ℤ) :
  star 5 7 1 3 = star x y 4 5 → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_star_equality_implies_x_eight_l4059_405906


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_same_directrix_l4059_405975

/-- An ellipse with equation x^2 + k*y^2 = 1 -/
def ellipse (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + k * p.2^2 = 1}

/-- A hyperbola with equation x^2/4 - y^2/5 = 1 -/
def hyperbola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / 4 - p.2^2 / 5 = 1}

/-- The directrix of a conic section -/
def directrix (c : Set (ℝ × ℝ)) : Set ℝ := sorry

theorem ellipse_hyperbola_same_directrix (k : ℝ) :
  directrix (ellipse k) = directrix hyperbola → k = 16/7 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_same_directrix_l4059_405975


namespace NUMINAMATH_CALUDE_point_a_coordinates_l4059_405976

/-- A point on the x-axis at a distance of 3 units from the origin -/
structure PointA where
  x : ℝ
  y : ℝ
  on_x_axis : y = 0
  distance_from_origin : x^2 + y^2 = 3^2

theorem point_a_coordinates (A : PointA) : (A.x = 3 ∧ A.y = 0) ∨ (A.x = -3 ∧ A.y = 0) := by
  sorry

end NUMINAMATH_CALUDE_point_a_coordinates_l4059_405976


namespace NUMINAMATH_CALUDE_no_reverse_multiply_all_ones_l4059_405914

/-- Given a natural number, return the number with its digits reversed -/
def reverse_digits (n : ℕ) : ℕ := sorry

/-- Check if a natural number is composed of only ones -/
def all_ones (n : ℕ) : Prop := sorry

theorem no_reverse_multiply_all_ones :
  ∀ n : ℕ, n > 1 → ¬(all_ones (n * reverse_digits n)) := by
  sorry

end NUMINAMATH_CALUDE_no_reverse_multiply_all_ones_l4059_405914


namespace NUMINAMATH_CALUDE_modulo_17_residue_l4059_405965

theorem modulo_17_residue : (305 + 7 * 51 + 11 * 187 + 6 * 23) % 17 = 3 := by
  sorry

end NUMINAMATH_CALUDE_modulo_17_residue_l4059_405965


namespace NUMINAMATH_CALUDE_box_2_neg2_3_l4059_405932

def box (a b c : ℤ) : ℚ := (a ^ (2 * b) : ℚ) - (b ^ (2 * c) : ℚ) + (c ^ (2 * a) : ℚ)

theorem box_2_neg2_3 : box 2 (-2) 3 = 273 / 16 := by sorry

end NUMINAMATH_CALUDE_box_2_neg2_3_l4059_405932


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_l4059_405925

theorem quadratic_roots_sum (k₁ k₂ : ℝ) : 
  36 * k₁^2 - 200 * k₁ + 49 = 0 →
  36 * k₂^2 - 200 * k₂ + 49 = 0 →
  k₁ / k₂ + k₂ / k₁ = 6.25 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_l4059_405925


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l4059_405911

theorem complex_fraction_simplification :
  (2 / 5 + 3 / 4) / (4 / 9 + 1 / 6) = 207 / 110 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l4059_405911


namespace NUMINAMATH_CALUDE_square_of_sum_twice_x_plus_y_l4059_405966

theorem square_of_sum_twice_x_plus_y (x y : ℝ) : (2*x + y)^2 = (2*x + y)^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_sum_twice_x_plus_y_l4059_405966


namespace NUMINAMATH_CALUDE_complex_fraction_sum_l4059_405955

theorem complex_fraction_sum : 
  let S := 1 / (2 - Real.sqrt 3) - 1 / (Real.sqrt 3 - Real.sqrt 2) + 
           1 / (Real.sqrt 2 - 1) - 1 / (1 - Real.sqrt 3 + Real.sqrt 2)
  S = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_sum_l4059_405955


namespace NUMINAMATH_CALUDE_odd_prime_factor_form_l4059_405919

theorem odd_prime_factor_form (p q : ℕ) (hp : Nat.Prime p) (hp_odd : Odd p) (hq : Nat.Prime q) (h_div : q ∣ 2^p - 1) :
  ∃ k : ℕ, q = 2*k*p + 1 := by
sorry

end NUMINAMATH_CALUDE_odd_prime_factor_form_l4059_405919


namespace NUMINAMATH_CALUDE_jump_height_to_touch_hoop_l4059_405915

/-- Calculates the jump height needed to touch a basketball hoop -/
theorem jump_height_to_touch_hoop 
  (yao_height_ft : ℕ) 
  (yao_height_in : ℕ) 
  (hoop_height_ft : ℕ) 
  (inches_per_foot : ℕ) : 
  hoop_height_ft * inches_per_foot - (yao_height_ft * inches_per_foot + yao_height_in) = 31 :=
by
  sorry

#check jump_height_to_touch_hoop 7 5 10 12

end NUMINAMATH_CALUDE_jump_height_to_touch_hoop_l4059_405915


namespace NUMINAMATH_CALUDE_value_of_expression_l4059_405993

theorem value_of_expression (x : ℝ) (h : 6 * x^2 - 4 * x - 3 = 0) :
  (x - 1)^2 + x * (x + 2/3) = 2 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l4059_405993


namespace NUMINAMATH_CALUDE_james_game_preparation_time_l4059_405900

def time_before_main_game (download_time install_time update_time account_time 
  internet_issues_time discussion_time tutorial_video_time in_game_tutorial_time : ℕ) : ℕ :=
  download_time + install_time + update_time + account_time + internet_issues_time + 
  discussion_time + tutorial_video_time + in_game_tutorial_time

theorem james_game_preparation_time :
  let download_time := 10
  let install_time := download_time / 2
  let update_time := download_time * 2
  let account_time := 5
  let internet_issues_time := 15
  let discussion_time := 20
  let tutorial_video_time := 8
  let preparation_time := download_time + install_time + update_time + account_time + 
    internet_issues_time + discussion_time + tutorial_video_time
  let in_game_tutorial_time := preparation_time * 3
  time_before_main_game download_time install_time update_time account_time 
    internet_issues_time discussion_time tutorial_video_time in_game_tutorial_time = 332 := by
  sorry

end NUMINAMATH_CALUDE_james_game_preparation_time_l4059_405900


namespace NUMINAMATH_CALUDE_loot_box_average_loss_l4059_405986

/-- Represents the expected value calculation for a loot box system -/
def loot_box_expected_value (standard_value : ℝ) (rare_a_prob : ℝ) (rare_a_value : ℝ)
  (rare_b_prob : ℝ) (rare_b_value : ℝ) (rare_c_prob : ℝ) (rare_c_value : ℝ) : ℝ :=
  let standard_prob := 1 - (rare_a_prob + rare_b_prob + rare_c_prob)
  standard_prob * standard_value + rare_a_prob * rare_a_value +
  rare_b_prob * rare_b_value + rare_c_prob * rare_c_value

/-- Calculates the average loss per loot box -/
def average_loss_per_loot_box (box_cost : ℝ) (expected_value : ℝ) : ℝ :=
  box_cost - expected_value

/-- Theorem stating the average loss per loot box in the given scenario -/
theorem loot_box_average_loss :
  let box_cost : ℝ := 5
  let standard_value : ℝ := 3.5
  let rare_a_prob : ℝ := 0.05
  let rare_a_value : ℝ := 10
  let rare_b_prob : ℝ := 0.03
  let rare_b_value : ℝ := 15
  let rare_c_prob : ℝ := 0.02
  let rare_c_value : ℝ := 20
  let expected_value := loot_box_expected_value standard_value rare_a_prob rare_a_value
    rare_b_prob rare_b_value rare_c_prob rare_c_value
  average_loss_per_loot_box box_cost expected_value = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_loot_box_average_loss_l4059_405986


namespace NUMINAMATH_CALUDE_zoo_cost_l4059_405908

def goat_cost : ℕ := 400
def goat_count : ℕ := 3

def llama_count : ℕ := 2 * goat_count
def llama_cost : ℕ := goat_cost + (goat_cost / 2)

def kangaroo_count : ℕ := 3 * goat_count
def kangaroo_cost : ℕ := llama_cost - (llama_cost / 4)

def total_cost : ℕ := goat_cost * goat_count + llama_cost * llama_count + kangaroo_cost * kangaroo_count

theorem zoo_cost : total_cost = 8850 := by
  sorry

end NUMINAMATH_CALUDE_zoo_cost_l4059_405908


namespace NUMINAMATH_CALUDE_total_earnings_is_4350_l4059_405903

/-- Given investment ratios and return ratios for three investors a, b, and c,
    calculates their total earnings. -/
def total_earnings (invest_a invest_b invest_c : ℚ)
                   (return_a return_b return_c : ℚ)
                   (diff_b_a : ℚ) : ℚ :=
  let earnings_a := invest_a * return_a
  let earnings_b := invest_b * return_b
  let earnings_c := invest_c * return_c
  earnings_a + earnings_b + earnings_c

/-- Theorem stating that under given conditions, the total earnings are 4350. -/
theorem total_earnings_is_4350 :
  ∃ (x y : ℚ),
    let invest_a := 3 * x
    let invest_b := 4 * x
    let invest_c := 5 * x
    let return_a := 6 * y
    let return_b := 5 * y
    let return_c := 4 * y
    invest_b * return_b - invest_a * return_a = 150 ∧
    total_earnings invest_a invest_b invest_c return_a return_b return_c 150 = 4350 :=
by
  sorry

#check total_earnings_is_4350

end NUMINAMATH_CALUDE_total_earnings_is_4350_l4059_405903


namespace NUMINAMATH_CALUDE_crimson_valley_skirts_l4059_405948

theorem crimson_valley_skirts 
  (azure_skirts : ℕ) 
  (seafoam_skirts : ℕ) 
  (purple_skirts : ℕ) 
  (crimson_skirts : ℕ) 
  (h1 : azure_skirts = 90)
  (h2 : seafoam_skirts = 2 * azure_skirts / 3)
  (h3 : purple_skirts = seafoam_skirts / 4)
  (h4 : crimson_skirts = purple_skirts / 3) :
  crimson_skirts = 5 := by
  sorry

end NUMINAMATH_CALUDE_crimson_valley_skirts_l4059_405948


namespace NUMINAMATH_CALUDE_gcd_of_squares_gcd_130_215_310_131_216_309_l4059_405931

theorem gcd_of_squares (a b c d e f : ℤ) : 
  Int.gcd (a^2 + b^2 + c^2) (d^2 + e^2 + f^2) = 
  Int.gcd ((d^2 + e^2 + f^2) : ℤ) (|((a - d) * (a + d) + (b - e) * (b + e) + (c - f) * (c + f))|) :=
by sorry

theorem gcd_130_215_310_131_216_309 : 
  Int.gcd (130^2 + 215^2 + 310^2) (131^2 + 216^2 + 309^2) = 1 :=
by sorry

end NUMINAMATH_CALUDE_gcd_of_squares_gcd_130_215_310_131_216_309_l4059_405931


namespace NUMINAMATH_CALUDE_roots_sum_magnitude_l4059_405959

theorem roots_sum_magnitude (p : ℝ) (r₁ r₂ : ℝ) : 
  r₁ ≠ r₂ →
  r₁^2 + p*r₁ + 18 = 0 →
  r₂^2 + p*r₂ + 18 = 0 →
  |r₁ + r₂| > 6 := by
sorry

end NUMINAMATH_CALUDE_roots_sum_magnitude_l4059_405959


namespace NUMINAMATH_CALUDE_tan_A_value_l4059_405917

theorem tan_A_value (A : Real) (h1 : 0 < A ∧ A < π / 2) 
  (h2 : 4 * Real.sin A ^ 2 - 4 * Real.sin A * Real.cos A + Real.cos A ^ 2 = 0) : 
  Real.tan A = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_tan_A_value_l4059_405917


namespace NUMINAMATH_CALUDE_genuine_coin_remains_l4059_405902

/-- Represents the type of a coin -/
inductive CoinType
| Genuine
| Fake

/-- Represents the state of the coin selection process -/
structure CoinState where
  total : Nat
  genuine : Nat
  fake : Nat
  moves : Nat

/-- The initial state of coins -/
def initialState : CoinState :=
  { total := 2022
  , genuine := 1012  -- More than half of 2022
  , fake := 1010     -- Less than half of 2022
  , moves := 0 }

/-- Simulates a single move in the coin selection process -/
def move (state : CoinState) : CoinState :=
  { state with
    total := state.total - 1
    moves := state.moves + 1
    genuine := state.genuine - 1  -- Worst case: remove a genuine coin
  }

/-- Applies the move function n times -/
def applyMoves (n : Nat) (state : CoinState) : CoinState :=
  match n with
  | 0 => state
  | n + 1 => move (applyMoves n state)

theorem genuine_coin_remains : 
  (applyMoves 2021 initialState).genuine > 0 := by
  sorry

#check genuine_coin_remains

end NUMINAMATH_CALUDE_genuine_coin_remains_l4059_405902


namespace NUMINAMATH_CALUDE_evaluate_g_l4059_405977

def g (x : ℝ) : ℝ := 3 * x^2 - 5 * x + 8

theorem evaluate_g : 3 * g 3 + 2 * g (-3) = 160 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_g_l4059_405977


namespace NUMINAMATH_CALUDE_area_ratio_of_squares_l4059_405927

/-- Given three squares A, B, and C with specific perimeters, 
    this theorem proves the ratio of areas of A to C. -/
theorem area_ratio_of_squares (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (pa : 4 * a = 16) -- perimeter of A is 16
  (pb : 4 * b = 40) -- perimeter of B is 40
  (pc : 4 * c = 120) -- perimeter of C is 120 (3 times B's perimeter)
  : (a * a) / (c * c) = 4 / 225 := by
  sorry

#check area_ratio_of_squares

end NUMINAMATH_CALUDE_area_ratio_of_squares_l4059_405927


namespace NUMINAMATH_CALUDE_population_growth_proof_l4059_405901

/-- The annual growth rate of the population -/
def annual_growth_rate : ℝ := 0.1

/-- The population after 2 years -/
def population_after_2_years : ℕ := 15730

/-- The initial population of the town -/
def initial_population : ℕ := 13000

theorem population_growth_proof :
  (1 + annual_growth_rate) * (1 + annual_growth_rate) * initial_population = population_after_2_years := by
  sorry

end NUMINAMATH_CALUDE_population_growth_proof_l4059_405901


namespace NUMINAMATH_CALUDE_necessary_sufficient_condition_l4059_405958

theorem necessary_sufficient_condition (a b : ℝ) :
  a * |a + b| < |a| * (a + b) ↔ a < 0 ∧ b > -a := by
  sorry

end NUMINAMATH_CALUDE_necessary_sufficient_condition_l4059_405958


namespace NUMINAMATH_CALUDE_mrs_hilt_animal_legs_l4059_405972

theorem mrs_hilt_animal_legs :
  let num_dogs : ℕ := 2
  let num_chickens : ℕ := 2
  let dog_legs : ℕ := 4
  let chicken_legs : ℕ := 2
  num_dogs * dog_legs + num_chickens * chicken_legs = 12 :=
by sorry

end NUMINAMATH_CALUDE_mrs_hilt_animal_legs_l4059_405972


namespace NUMINAMATH_CALUDE_f_even_iff_a_zero_l4059_405921

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The quadratic function f(x) = x^2 + ax + 1 -/
def f (a : ℝ) : ℝ → ℝ := fun x ↦ x^2 + a*x + 1

/-- Theorem: f is an even function if and only if a = 0 -/
theorem f_even_iff_a_zero (a : ℝ) :
  IsEven (f a) ↔ a = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_even_iff_a_zero_l4059_405921


namespace NUMINAMATH_CALUDE_melissas_total_points_l4059_405956

/-- Calculates the total points scored in multiple games -/
def totalPoints (gamesPlayed : ℕ) (pointsPerGame : ℕ) : ℕ :=
  gamesPlayed * pointsPerGame

/-- Proves that Melissa's total points is 81 -/
theorem melissas_total_points :
  let gamesPlayed : ℕ := 3
  let pointsPerGame : ℕ := 27
  totalPoints gamesPlayed pointsPerGame = 81 := by
  sorry

end NUMINAMATH_CALUDE_melissas_total_points_l4059_405956


namespace NUMINAMATH_CALUDE_determinant_trig_matrix_equals_one_l4059_405909

theorem determinant_trig_matrix_equals_one (α β γ : ℝ) :
  let M : Matrix (Fin 3) (Fin 3) ℝ := λ i j ↦ 
    match i, j with
    | 0, 0 => Real.cos (α + γ) * Real.cos β
    | 0, 1 => Real.cos (α + γ) * Real.sin β
    | 0, 2 => -Real.sin (α + γ)
    | 1, 0 => -Real.sin β
    | 1, 1 => Real.cos β
    | 1, 2 => 0
    | 2, 0 => Real.sin (α + γ) * Real.cos β
    | 2, 1 => Real.sin (α + γ) * Real.sin β
    | 2, 2 => Real.cos (α + γ)
  Matrix.det M = 1 := by sorry

end NUMINAMATH_CALUDE_determinant_trig_matrix_equals_one_l4059_405909


namespace NUMINAMATH_CALUDE_eccentricity_of_hyperbola_with_diagonal_asymptotes_l4059_405940

/-- A hyperbola with given asymptotes -/
structure Hyperbola where
  -- Asymptotes of the hyperbola are y = ±x
  asymptotes : (ℝ → ℝ) × (ℝ → ℝ)
  asymptotes_prop : asymptotes = ((fun x => x), (fun x => -x))

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola) : ℝ := sorry

/-- Theorem: The eccentricity of a hyperbola with asymptotes y = ±x is √2 -/
theorem eccentricity_of_hyperbola_with_diagonal_asymptotes (h : Hyperbola) :
  eccentricity h = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_eccentricity_of_hyperbola_with_diagonal_asymptotes_l4059_405940


namespace NUMINAMATH_CALUDE_stratified_sampling_third_year_students_l4059_405973

theorem stratified_sampling_third_year_students 
  (total_students : ℕ) 
  (third_year_students : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_students = 1600) 
  (h2 : third_year_students = 400) 
  (h3 : sample_size = 160) :
  (sample_size * third_year_students) / total_students = 40 :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_third_year_students_l4059_405973


namespace NUMINAMATH_CALUDE_milk_water_mixture_l4059_405981

theorem milk_water_mixture (milk water : ℝ) : 
  milk / water = 2 →
  milk / (water + 10) = 6 / 5 →
  milk = 30 := by
sorry

end NUMINAMATH_CALUDE_milk_water_mixture_l4059_405981


namespace NUMINAMATH_CALUDE_find_regular_working_hours_l4059_405905

/-- Represents the problem of finding regular working hours per day --/
theorem find_regular_working_hours
  (working_days_per_week : ℕ)
  (regular_pay_rate : ℚ)
  (overtime_pay_rate : ℚ)
  (total_earnings : ℚ)
  (total_hours : ℕ)
  (h1 : working_days_per_week = 6)
  (h2 : regular_pay_rate = 21/10)
  (h3 : overtime_pay_rate = 42/10)
  (h4 : total_earnings = 525)
  (h5 : total_hours = 245) :
  ∃ (regular_hours_per_day : ℕ),
    regular_hours_per_day = 10 ∧
    regular_hours_per_day * working_days_per_week * 4 ≤ total_hours ∧
    regular_pay_rate * (regular_hours_per_day * working_days_per_week * 4) +
    overtime_pay_rate * (total_hours - regular_hours_per_day * working_days_per_week * 4) =
    total_earnings :=
by sorry

end NUMINAMATH_CALUDE_find_regular_working_hours_l4059_405905


namespace NUMINAMATH_CALUDE_partnership_investment_l4059_405954

/-- A partnership business problem -/
theorem partnership_investment (b_investment c_investment c_profit total_profit : ℕ) 
  (hb : b_investment = 72000)
  (hc : c_investment = 81000)
  (hcp : c_profit = 36000)
  (htp : total_profit = 80000) :
  ∃ a_investment : ℕ, 
    (c_profit : ℚ) / (total_profit : ℚ) = (c_investment : ℚ) / ((a_investment : ℚ) + (b_investment : ℚ) + (c_investment : ℚ)) ∧ 
    a_investment = 27000 := by
  sorry

end NUMINAMATH_CALUDE_partnership_investment_l4059_405954


namespace NUMINAMATH_CALUDE_original_number_proof_l4059_405991

theorem original_number_proof : ∃ N : ℕ, 
  (∀ m : ℕ, m < N → ¬(m - 6 ≡ 3 [MOD 5] ∧ m - 6 ≡ 3 [MOD 11] ∧ m - 6 ≡ 3 [MOD 13])) ∧
  (N - 6 ≡ 3 [MOD 5] ∧ N - 6 ≡ 3 [MOD 11] ∧ N - 6 ≡ 3 [MOD 13]) ∧
  N = 724 :=
by sorry

#check original_number_proof

end NUMINAMATH_CALUDE_original_number_proof_l4059_405991


namespace NUMINAMATH_CALUDE_F_composition_result_l4059_405996

def F (x : ℝ) : ℝ := 2 * x - 1

theorem F_composition_result : F (F (F (F (F 2)))) = 33 := by
  sorry

end NUMINAMATH_CALUDE_F_composition_result_l4059_405996


namespace NUMINAMATH_CALUDE_iron_conducts_electricity_l4059_405904

-- Define the universe of discourse
variable (Object : Type)

-- Define the predicates
variable (is_metal : Object → Prop)
variable (can_conduct_electricity : Object → Prop)

-- Define iron as a constant
variable (iron : Object)

-- Theorem statement
theorem iron_conducts_electricity 
  (all_metals_conduct : ∀ x, is_metal x → can_conduct_electricity x) 
  (iron_is_metal : is_metal iron) : 
  can_conduct_electricity iron := by
  sorry

end NUMINAMATH_CALUDE_iron_conducts_electricity_l4059_405904


namespace NUMINAMATH_CALUDE_fixed_point_parabola_l4059_405963

theorem fixed_point_parabola (k : ℝ) : 
  225 = 9 * (5 : ℝ)^2 + k * 5 - 5 * k := by sorry

end NUMINAMATH_CALUDE_fixed_point_parabola_l4059_405963


namespace NUMINAMATH_CALUDE_enrico_earnings_l4059_405997

/-- Calculates the earnings from selling roosters -/
def rooster_earnings (price_per_kg : ℚ) (weights : List ℚ) : ℚ :=
  (weights.map (· * price_per_kg)).sum

/-- Proves that Enrico's earnings from selling two roosters are $35 -/
theorem enrico_earnings : 
  let price_per_kg : ℚ := 1/2
  let weights : List ℚ := [30, 40]
  rooster_earnings price_per_kg weights = 35 := by
sorry

#eval rooster_earnings (1/2) [30, 40]

end NUMINAMATH_CALUDE_enrico_earnings_l4059_405997


namespace NUMINAMATH_CALUDE_managers_salary_l4059_405983

theorem managers_salary (num_employees : ℕ) (avg_salary : ℝ) (avg_increase : ℝ) : 
  num_employees = 20 → 
  avg_salary = 1500 → 
  avg_increase = 1000 → 
  (num_employees * avg_salary + (num_employees + 1) * avg_increase) / (num_employees + 1) - avg_salary = 22500 :=
by sorry

end NUMINAMATH_CALUDE_managers_salary_l4059_405983


namespace NUMINAMATH_CALUDE_function_equality_l4059_405938

theorem function_equality (f : ℝ → ℝ) : 
  (∀ x y : ℝ, f (2 * x + f y) = x + y + f x) → 
  (∀ x : ℝ, f x = x) := by
sorry

end NUMINAMATH_CALUDE_function_equality_l4059_405938


namespace NUMINAMATH_CALUDE_square_of_complex_number_l4059_405944

theorem square_of_complex_number : 
  let z : ℂ := 1 - 2*I
  z^2 = -3 - 4*I :=
by sorry

end NUMINAMATH_CALUDE_square_of_complex_number_l4059_405944


namespace NUMINAMATH_CALUDE_tree_height_proof_l4059_405969

/-- Proves that a tree with current height 180 inches, which is 50% taller than its original height, had an original height of 10 feet. -/
theorem tree_height_proof (current_height : ℝ) (height_increase_percent : ℝ) 
  (h1 : current_height = 180)
  (h2 : height_increase_percent = 50)
  (h3 : current_height = (1 + height_increase_percent / 100) * (12 * 10)) : 
  ∃ (original_height_feet : ℝ), original_height_feet = 10 :=
by
  sorry

#check tree_height_proof

end NUMINAMATH_CALUDE_tree_height_proof_l4059_405969


namespace NUMINAMATH_CALUDE_expected_original_positions_value_l4059_405916

/-- Represents the number of balls in the circle -/
def num_balls : ℕ := 7

/-- Represents the probability of a ball being in its original position after two transpositions -/
def prob_original_position : ℚ := 9 / 14

/-- The expected number of balls in their original positions after two transpositions -/
def expected_original_positions : ℚ := num_balls * prob_original_position

theorem expected_original_positions_value :
  expected_original_positions = 4.5 := by sorry

end NUMINAMATH_CALUDE_expected_original_positions_value_l4059_405916


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l4059_405978

/-- Given a hyperbola C and a circle F with specific properties, prove that the eccentricity of C is 2 -/
theorem hyperbola_eccentricity (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  let C := {(x, y) : ℝ × ℝ | x^2 / a^2 - y^2 / b^2 = 1}
  let F := {(x, y) : ℝ × ℝ | (x - c)^2 + y^2 = c^2}
  let l := {(x, y) : ℝ × ℝ | y = -(a / b) * (x - 2 * a / 3)}
  ∃ (chord_length : ℝ), 
    (∀ (p q : ℝ × ℝ), p ∈ F ∧ q ∈ F ∧ p ∈ l ∧ q ∈ l → ‖p - q‖ = chord_length) ∧
    chord_length = 4 * Real.sqrt 2 * c / 3 →
  c / a = 2 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l4059_405978


namespace NUMINAMATH_CALUDE_laser_reflection_distance_l4059_405936

def laser_path_distance : ℝ → ℝ → ℝ → ℝ → ℝ := sorry

theorem laser_reflection_distance :
  let start_x : ℝ := 2
  let start_y : ℝ := 4
  let end_x : ℝ := 10
  let end_y : ℝ := 4
  laser_path_distance start_x start_y end_x end_y = 6 + 2 * Real.sqrt 29 := by
  sorry

end NUMINAMATH_CALUDE_laser_reflection_distance_l4059_405936


namespace NUMINAMATH_CALUDE_min_button_presses_to_escape_l4059_405941

/-- Represents the state of the room with doors and mines -/
structure RoomState where
  armed_mines : ℕ
  closed_doors : ℕ

/-- Represents the actions of pressing buttons -/
structure ButtonPresses where
  red : ℕ
  yellow : ℕ
  green : ℕ

/-- Calculates the final state of the room after pressing buttons -/
def final_state (initial : RoomState) (presses : ButtonPresses) : RoomState :=
  { armed_mines := initial.armed_mines + presses.red - 2 * presses.yellow,
    closed_doors := initial.closed_doors + presses.yellow - 2 * presses.green }

/-- Checks if all mines are disarmed and all doors are open -/
def is_solved (state : RoomState) : Prop :=
  state.armed_mines = 0 ∧ state.closed_doors = 0

/-- The main theorem to prove -/
theorem min_button_presses_to_escape : 
  ∃ (presses : ButtonPresses),
    is_solved (final_state { armed_mines := 3, closed_doors := 3 } presses) ∧
    presses.red + presses.yellow + presses.green = 9 ∧
    ∀ (other_presses : ButtonPresses),
      is_solved (final_state { armed_mines := 3, closed_doors := 3 } other_presses) →
      other_presses.red + other_presses.yellow + other_presses.green ≥ 9 :=
by sorry

end NUMINAMATH_CALUDE_min_button_presses_to_escape_l4059_405941


namespace NUMINAMATH_CALUDE_chime_2023_date_l4059_405937

/-- Represents a date with year, month, and day -/
structure Date :=
  (year : Nat) (month : Nat) (day : Nat)

/-- Represents a time with hour and minute -/
structure Time :=
  (hour : Nat) (minute : Nat)

/-- Calculates the number of chimes for a given hour -/
def chimes_for_hour (hour : Nat) : Nat :=
  if hour ≤ 12 then hour else hour - 12

/-- Calculates the total number of chimes in a day with the malfunction -/
def daily_chimes : Nat := 101

/-- Calculates the number of chimes from a given start time to midnight -/
def chimes_until_midnight (start_time : Time) : Nat :=
  sorry -- Implementation details omitted

/-- Calculates the date of the nth chime given a start date and time -/
def date_of_nth_chime (start_date : Date) (start_time : Time) (n : Nat) : Date :=
  sorry -- Implementation details omitted

theorem chime_2023_date :
  let start_date := Date.mk 2003 2 26
  let start_time := Time.mk 14 15
  date_of_nth_chime start_date start_time 2023 = Date.mk 2003 3 18 := by
  sorry

end NUMINAMATH_CALUDE_chime_2023_date_l4059_405937
