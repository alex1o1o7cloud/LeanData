import Mathlib

namespace NUMINAMATH_CALUDE_davids_trip_spending_l2060_206036

theorem davids_trip_spending (initial_amount spent_amount remaining_amount : ℕ) : 
  initial_amount = 1800 →
  remaining_amount = spent_amount - 800 →
  initial_amount = spent_amount + remaining_amount →
  remaining_amount = 500 := by
  sorry

end NUMINAMATH_CALUDE_davids_trip_spending_l2060_206036


namespace NUMINAMATH_CALUDE_train_meeting_time_l2060_206091

/-- Represents the problem of two trains meeting on a journey from Delhi to Bombay -/
theorem train_meeting_time 
  (speed_first : ℝ) 
  (speed_second : ℝ) 
  (departure_second : ℝ) 
  (meeting_distance : ℝ) 
  (h1 : speed_first = 60) 
  (h2 : speed_second = 80) 
  (h3 : departure_second = 16.5) 
  (h4 : meeting_distance = 480) : 
  ∃ (departure_first : ℝ), 
    speed_first * (departure_second - departure_first) = meeting_distance ∧ 
    departure_first = 8.5 := by
  sorry

end NUMINAMATH_CALUDE_train_meeting_time_l2060_206091


namespace NUMINAMATH_CALUDE_power_mod_seventeen_l2060_206099

theorem power_mod_seventeen : 3^45 % 17 = 15 := by
  sorry

end NUMINAMATH_CALUDE_power_mod_seventeen_l2060_206099


namespace NUMINAMATH_CALUDE_absolute_value_fraction_l2060_206008

theorem absolute_value_fraction (x y : ℝ) 
  (h : y < Real.sqrt (x - 1) + Real.sqrt (1 - x) + 1) : 
  |y - 1| / (y - 1) = -1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_fraction_l2060_206008


namespace NUMINAMATH_CALUDE_sally_orange_balloons_l2060_206079

def initial_orange_balloons : ℕ := 9
def lost_orange_balloons : ℕ := 2

theorem sally_orange_balloons :
  initial_orange_balloons - lost_orange_balloons = 7 :=
by sorry

end NUMINAMATH_CALUDE_sally_orange_balloons_l2060_206079


namespace NUMINAMATH_CALUDE_wall_width_proof_l2060_206029

/-- Given a rectangular wall and a square mirror, if the mirror's area is half the wall's area,
    prove that the wall's width is 68 inches. -/
theorem wall_width_proof (wall_length wall_width mirror_side : ℝ) : 
  wall_length = 85.76470588235294 →
  mirror_side = 54 →
  (mirror_side * mirror_side) = (wall_length * wall_width) / 2 →
  wall_width = 68 := by sorry

end NUMINAMATH_CALUDE_wall_width_proof_l2060_206029


namespace NUMINAMATH_CALUDE_A_neither_sufficient_nor_necessary_l2060_206075

-- Define propositions A and B
def prop_A (a b : ℝ) : Prop := a + b ≠ 4
def prop_B (a b : ℝ) : Prop := a ≠ 1 ∧ b ≠ 3

-- Theorem stating that A is neither sufficient nor necessary for B
theorem A_neither_sufficient_nor_necessary :
  ¬(∀ a b : ℝ, prop_A a b → prop_B a b) ∧
  ¬(∀ a b : ℝ, prop_B a b → prop_A a b) :=
by sorry

end NUMINAMATH_CALUDE_A_neither_sufficient_nor_necessary_l2060_206075


namespace NUMINAMATH_CALUDE_distribution_and_points_correct_l2060_206020

/-- Represents a comparison between two tanks -/
structure TankComparison where
  siyan_name : String
  siyan_quality : Nat
  zvezda_quality : Nat
  zvezda_name : String

/-- Calculates the oil distribution and rating points -/
def calculate_distribution_and_points (comparisons : List TankComparison) (oil_quantity : Real) :
  (Real × Real × Nat × Nat) :=
  let process := λ (acc : Real × Real × Nat × Nat) (comp : TankComparison) =>
    let (hv_22, lv_426, siyan_points, zvezda_points) := acc
    let new_hv_22 := hv_22 + 
      (if comp.siyan_quality > 2 then oil_quantity else 0) +
      (if comp.zvezda_quality > 2 then oil_quantity else 0)
    let new_lv_426 := lv_426 + 
      (if comp.siyan_quality ≤ 2 then oil_quantity else 0) +
      (if comp.zvezda_quality ≤ 2 then oil_quantity else 0)
    let new_siyan_points := siyan_points +
      (if comp.siyan_quality > comp.zvezda_quality then 3
       else if comp.siyan_quality = comp.zvezda_quality then 1
       else 0)
    let new_zvezda_points := zvezda_points +
      (if comp.zvezda_quality > comp.siyan_quality then 3
       else if comp.siyan_quality = comp.zvezda_quality then 1
       else 0)
    (new_hv_22, new_lv_426, new_siyan_points, new_zvezda_points)
  comparisons.foldl process (0, 0, 0, 0)

/-- Theorem stating the correctness of the calculation -/
theorem distribution_and_points_correct (comparisons : List TankComparison) (oil_quantity : Real) :
  let (hv_22, lv_426, siyan_points, zvezda_points) := calculate_distribution_and_points comparisons oil_quantity
  (hv_22 ≥ 0 ∧ lv_426 ≥ 0 ∧ 
   hv_22 + lv_426 = oil_quantity * comparisons.length * 2 ∧
   siyan_points + zvezda_points = comparisons.length * 3) :=
by sorry

end NUMINAMATH_CALUDE_distribution_and_points_correct_l2060_206020


namespace NUMINAMATH_CALUDE_optimal_strategy_is_valid_l2060_206053

/-- Represents a chain of links -/
structure Chain where
  links : ℕ

/-- Represents a cut in the chain -/
structure Cut where
  position : ℕ

/-- Represents a payment strategy for the hotel stay -/
structure PaymentStrategy where
  cut : Cut
  dailyPayments : List ℕ

/-- Checks if a payment strategy is valid for the given chain and number of days -/
def isValidPaymentStrategy (c : Chain) (days : ℕ) (s : PaymentStrategy) : Prop :=
  c.links = days ∧
  s.cut.position > 0 ∧
  s.cut.position < c.links ∧
  s.dailyPayments.length = days ∧
  s.dailyPayments.sum = c.links

/-- The optimal payment strategy for a 7-day stay with a 7-link chain -/
def optimalStrategy : PaymentStrategy :=
  { cut := { position := 3 },
    dailyPayments := [1, 1, 1, 1, 1, 1, 1] }

/-- Theorem stating that the optimal strategy is valid for a 7-day stay with a 7-link chain -/
theorem optimal_strategy_is_valid :
  isValidPaymentStrategy { links := 7 } 7 optimalStrategy := by sorry

end NUMINAMATH_CALUDE_optimal_strategy_is_valid_l2060_206053


namespace NUMINAMATH_CALUDE_extreme_values_and_monotonicity_l2060_206050

-- Define the function f(x)
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x

-- Define the derivative of f(x)
def f' (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem extreme_values_and_monotonicity (a b : ℝ) :
  (f' a b (-1) = 0 ∧ f' a b 2 = 0) →
  (a = -3/2 ∧ b = -6) ∧
  (∀ x, x ∈ Set.Ioo (-1) 2 → (f' (-3/2) (-6) x < 0)) ∧
  (∀ x, (x < -1 ∨ x > 2) → (f' (-3/2) (-6) x > 0)) ∧
  (∀ m, (∀ x, x ∈ Set.Icc (-2) 3 → f (-3/2) (-6) x < m) ↔ m > 7/2) :=
by sorry


end NUMINAMATH_CALUDE_extreme_values_and_monotonicity_l2060_206050


namespace NUMINAMATH_CALUDE_solve_temperature_problem_l2060_206021

def temperature_problem (temps : List ℝ) (avg : ℝ) : Prop :=
  temps.length = 6 ∧
  temps = [99.1, 98.2, 98.7, 99.3, 99.8, 99] ∧
  avg = 99 ∧
  ∃ (saturday_temp : ℝ),
    (temps.sum + saturday_temp) / 7 = avg ∧
    saturday_temp = 98.9

theorem solve_temperature_problem (temps : List ℝ) (avg : ℝ)
  (h : temperature_problem temps avg) : 
  ∃ (saturday_temp : ℝ), saturday_temp = 98.9 := by
  sorry

end NUMINAMATH_CALUDE_solve_temperature_problem_l2060_206021


namespace NUMINAMATH_CALUDE_graph_connectivity_probability_l2060_206084

/-- A complete graph with 20 vertices -/
def complete_graph : Nat := 20

/-- Number of edges removed -/
def removed_edges : Nat := 35

/-- Total number of edges in the complete graph -/
def total_edges : Nat := complete_graph * (complete_graph - 1) / 2

/-- Number of edges remaining after removal -/
def remaining_edges : Nat := total_edges - removed_edges

/-- Probability that the graph remains connected after edge removal -/
def connected_probability : ℚ :=
  1 - (complete_graph * (Nat.choose (total_edges - complete_graph + 1) (removed_edges - complete_graph + 1))) / 
      (Nat.choose total_edges removed_edges)

theorem graph_connectivity_probability :
  connected_probability = 1 - (20 * (Nat.choose 171 16)) / (Nat.choose 190 35) :=
sorry

end NUMINAMATH_CALUDE_graph_connectivity_probability_l2060_206084


namespace NUMINAMATH_CALUDE_book_page_digit_sum_l2060_206039

/-- Sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Sum of digits for all page numbers from 1 to n -/
def total_digit_sum (n : ℕ) : ℕ := sorry

/-- Theorem: The sum of all digits in page numbers of a 2000-page book is 28002 -/
theorem book_page_digit_sum :
  total_digit_sum 2000 = 28002 := by sorry

end NUMINAMATH_CALUDE_book_page_digit_sum_l2060_206039


namespace NUMINAMATH_CALUDE_skitties_remainder_l2060_206041

theorem skitties_remainder (m : ℕ) (h : m % 7 = 5) : (4 * m) % 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_skitties_remainder_l2060_206041


namespace NUMINAMATH_CALUDE_trapezoid_division_l2060_206045

/-- Represents a trapezoid with given properties -/
structure Trapezoid where
  area : ℝ
  base_ratio : ℝ
  smaller_base : ℝ
  larger_base : ℝ
  height : ℝ
  area_eq : area = (smaller_base + larger_base) * height / 2
  base_ratio_eq : larger_base = base_ratio * smaller_base

/-- Represents the two smaller trapezoids formed by the median line -/
structure SmallerTrapezoids where
  top_area : ℝ
  bottom_area : ℝ

/-- The main theorem stating the areas of smaller trapezoids -/
theorem trapezoid_division (t : Trapezoid) 
  (h1 : t.area = 80)
  (h2 : t.base_ratio = 3) :
  ∃ (st : SmallerTrapezoids), st.top_area = 30 ∧ st.bottom_area = 50 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_division_l2060_206045


namespace NUMINAMATH_CALUDE_seed_survival_rate_l2060_206040

theorem seed_survival_rate 
  (germination_rate : ℝ) 
  (seedling_probability : ℝ) 
  (h1 : germination_rate = 0.9) 
  (h2 : seedling_probability = 0.81) : 
  ∃ p : ℝ, p = germination_rate ∧ p * germination_rate = seedling_probability :=
by
  sorry

end NUMINAMATH_CALUDE_seed_survival_rate_l2060_206040


namespace NUMINAMATH_CALUDE_sum_of_squares_cubic_roots_l2060_206086

theorem sum_of_squares_cubic_roots (p q r : ℝ) : 
  (3 * p^3 - 2 * p^2 + 5 * p - 7 = 0) →
  (3 * q^3 - 2 * q^2 + 5 * q - 7 = 0) →
  (3 * r^3 - 2 * r^2 + 5 * r - 7 = 0) →
  p^2 + q^2 + r^2 = -26/9 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_cubic_roots_l2060_206086


namespace NUMINAMATH_CALUDE_symbiotic_pair_negation_l2060_206023

theorem symbiotic_pair_negation (m n : ℚ) : 
  (m - n = m * n + 1) → (-n - (-m) = (-n) * (-m) + 1) := by
  sorry

end NUMINAMATH_CALUDE_symbiotic_pair_negation_l2060_206023


namespace NUMINAMATH_CALUDE_infinite_solutions_condition_l2060_206048

theorem infinite_solutions_condition (a b : ℝ) :
  (∀ x, a * (a - x) - b * (b - x) = 0) → a - b = 0 := by
  sorry

end NUMINAMATH_CALUDE_infinite_solutions_condition_l2060_206048


namespace NUMINAMATH_CALUDE_longest_side_length_l2060_206066

-- Define the feasible region
def FeasibleRegion (x y : ℝ) : Prop :=
  x + 2*y ≤ 4 ∧ 3*x + y ≥ 3 ∧ x ≥ 0 ∧ y ≥ 0

-- Define the vertices of the quadrilateral
def Vertices : Set (ℝ × ℝ) :=
  {(0, 3), (0.4, 1.8), (4, 0), (0, 0)}

-- State the theorem
theorem longest_side_length :
  ∃ (a b : ℝ × ℝ), a ∈ Vertices ∧ b ∈ Vertices ∧
    (∀ (c d : ℝ × ℝ), c ∈ Vertices → d ∈ Vertices →
      Real.sqrt ((c.1 - d.1)^2 + (c.2 - d.2)^2) ≤ Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2)) ∧
    Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = 5 :=
by sorry

end NUMINAMATH_CALUDE_longest_side_length_l2060_206066


namespace NUMINAMATH_CALUDE_soccer_team_games_theorem_l2060_206080

/-- Represents the ratio of wins, losses, and ties for a soccer team -/
structure GameRatio :=
  (wins : ℕ)
  (losses : ℕ)
  (ties : ℕ)

/-- Calculates the total number of games played given a game ratio and number of losses -/
def totalGames (ratio : GameRatio) (numLosses : ℕ) : ℕ :=
  let gamesPerPart := numLosses / ratio.losses
  (ratio.wins + ratio.losses + ratio.ties) * gamesPerPart

/-- Theorem stating that for a team with a 4:3:1 win:loss:tie ratio and 9 losses, 
    the total number of games played is 24 -/
theorem soccer_team_games_theorem :
  let ratio : GameRatio := ⟨4, 3, 1⟩
  totalGames ratio 9 = 24 := by
  sorry

end NUMINAMATH_CALUDE_soccer_team_games_theorem_l2060_206080


namespace NUMINAMATH_CALUDE_complex_modulus_l2060_206094

theorem complex_modulus (i : ℂ) (z : ℂ) : 
  i * i = -1 → z = 2 + 3 * i / (1 - i) → Complex.abs z = Real.sqrt 10 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l2060_206094


namespace NUMINAMATH_CALUDE_nested_fraction_equality_l2060_206072

theorem nested_fraction_equality : 
  1 / (3 - 1 / (3 - 1 / (3 - 1 / (3 - 1 / 3)))) = 21 / 55 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_equality_l2060_206072


namespace NUMINAMATH_CALUDE_price_decrease_percentage_l2060_206074

-- Define the original and discounted prices
def original_price : ℚ := 12 / 3
def discounted_price : ℚ := 10 / 4

-- Define the percentage decrease
def percentage_decrease : ℚ := (original_price - discounted_price) / original_price * 100

-- Theorem statement
theorem price_decrease_percentage :
  percentage_decrease = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_price_decrease_percentage_l2060_206074


namespace NUMINAMATH_CALUDE_triangle_distance_l2060_206055

/-- Given a triangle ABC with the following properties:
  - AB = x meters
  - BC = 3 meters
  - Angle B = 150°
  - Area of triangle ABC = 3√3/4 m²
  Prove that the length of AC is √3 meters. -/
theorem triangle_distance (x : ℝ) : 
  let a := x
  let b := 3
  let c := (a^2 + b^2 - 2*a*b*Real.cos (150 * π / 180))^(1/2)
  let s := 3 * Real.sqrt 3 / 4
  s = 1/2 * a * b * Real.sin (150 * π / 180) →
  c = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_distance_l2060_206055


namespace NUMINAMATH_CALUDE_lines_parallel_iff_x_eq_9_l2060_206035

/-- Two 2D vectors are parallel if one is a scalar multiple of the other -/
def are_parallel (v1 v2 : ℝ × ℝ) : Prop :=
  ∃ (c : ℝ), v1 = (c * v2.1, c * v2.2)

/-- Definition of the first line -/
def line1 (u : ℝ) : ℝ × ℝ := (1 + 6*u, 3 - 2*u)

/-- Definition of the second line -/
def line2 (x v : ℝ) : ℝ × ℝ := (-4 + x*v, 5 - 3*v)

/-- The theorem stating that the lines are parallel iff x = 9 -/
theorem lines_parallel_iff_x_eq_9 :
  ∀ x : ℝ, (∀ u v : ℝ, line1 u ≠ line2 x v) ↔ x = 9 :=
sorry

end NUMINAMATH_CALUDE_lines_parallel_iff_x_eq_9_l2060_206035


namespace NUMINAMATH_CALUDE_half_plus_six_equals_eleven_l2060_206081

theorem half_plus_six_equals_eleven (n : ℝ) : (1/2 : ℝ) * n + 6 = 11 → n = 10 := by
  sorry

end NUMINAMATH_CALUDE_half_plus_six_equals_eleven_l2060_206081


namespace NUMINAMATH_CALUDE_five_balls_three_boxes_l2060_206028

/-- Calculates the number of ways to distribute n distinct balls into k distinct boxes,
    with each box containing at least 1 ball. -/
def distributeBalls (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- The number of ways to distribute 5 distinct balls into 3 distinct boxes,
    with each box containing at least 1 ball, is equal to 150. -/
theorem five_balls_three_boxes : distributeBalls 5 3 = 150 := by
  sorry

end NUMINAMATH_CALUDE_five_balls_three_boxes_l2060_206028


namespace NUMINAMATH_CALUDE_smallest_n_for_sqrt_difference_smallest_n_is_626_l2060_206065

theorem smallest_n_for_sqrt_difference (n : ℕ) : n ≥ 626 ↔ Real.sqrt n - Real.sqrt (n - 1) < 0.02 := by
  sorry

theorem smallest_n_is_626 : ∀ k : ℕ, k < 626 → Real.sqrt k - Real.sqrt (k - 1) ≥ 0.02 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_sqrt_difference_smallest_n_is_626_l2060_206065


namespace NUMINAMATH_CALUDE_exists_infinite_set_satisfying_equation_l2060_206093

/-- A function from positive integers to positive integers -/
def PositiveIntegerFunction : Type := ℕ+ → ℕ+

/-- The property that f(x) + f(x+2) ≤ 2f(x+1) for all x -/
def SatisfiesInequality (f : PositiveIntegerFunction) : Prop :=
  ∀ x : ℕ+, f x + f (x + 2) ≤ 2 * f (x + 1)

/-- The property that (i-j)f(k) + (j-k)f(i) + (k-i)f(j) = 0 for all i, j, k in a set M -/
def SatisfiesEquation (f : PositiveIntegerFunction) (M : Set ℕ+) : Prop :=
  ∀ i j k : ℕ+, i ∈ M → j ∈ M → k ∈ M →
    (i - j : ℤ) * (f k : ℤ) + (j - k : ℤ) * (f i : ℤ) + (k - i : ℤ) * (f j : ℤ) = 0

/-- The main theorem -/
theorem exists_infinite_set_satisfying_equation
  (f : PositiveIntegerFunction) (h : SatisfiesInequality f) :
  ∃ M : Set ℕ+, Set.Infinite M ∧ SatisfiesEquation f M := by
  sorry

end NUMINAMATH_CALUDE_exists_infinite_set_satisfying_equation_l2060_206093


namespace NUMINAMATH_CALUDE_ones_digit_sum_powers_l2060_206022

theorem ones_digit_sum_powers (n : Nat) : n = 2023 → 
  (1^n + 2^n + 3^n + 4^n + 5^n) % 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ones_digit_sum_powers_l2060_206022


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l2060_206056

theorem simplify_fraction_product : (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l2060_206056


namespace NUMINAMATH_CALUDE_sum_of_7th_8th_9th_terms_l2060_206073

/-- A geometric sequence with sum of first n terms S_n -/
structure GeometricSequence where
  a : ℕ → ℝ
  S : ℕ → ℝ
  is_geometric : ∀ n, a (n + 1) = a n * (a 2 / a 1)
  sum_formula : ∀ n, S n = (a 1) * (1 - (a 2 / a 1)^n) / (1 - (a 2 / a 1))

/-- Theorem stating the sum of 7th, 8th, and 9th terms of the geometric sequence -/
theorem sum_of_7th_8th_9th_terms (seq : GeometricSequence) 
  (h1 : seq.S 3 = 8) (h2 : seq.S 6 = 7) : 
  seq.a 7 + seq.a 8 + seq.a 9 = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_7th_8th_9th_terms_l2060_206073


namespace NUMINAMATH_CALUDE_b_alone_time_l2060_206089

/-- The time (in days) it takes for A and B together to complete the work -/
def combined_time : ℚ := 12

/-- The time (in days) it takes for A alone to complete the work -/
def a_time : ℚ := 24

/-- The work rate of A (work per day) -/
def a_rate : ℚ := 1 / a_time

/-- The combined work rate of A and B (work per day) -/
def combined_rate : ℚ := 1 / combined_time

/-- The work rate of B (work per day) -/
def b_rate : ℚ := combined_rate - a_rate

/-- The time (in days) it takes for B alone to complete the work -/
def b_time : ℚ := 1 / b_rate

theorem b_alone_time : b_time = 24 := by
  sorry

end NUMINAMATH_CALUDE_b_alone_time_l2060_206089


namespace NUMINAMATH_CALUDE_rationalize_denominator_l2060_206088

theorem rationalize_denominator :
  ∃ (A B C D E : ℤ),
    (2 : ℝ) / (3 * Real.sqrt 7 + 2 * Real.sqrt 13) =
    (A * Real.sqrt B + C * Real.sqrt D) / E ∧
    B < D ∧
    A = 6 ∧
    B = 7 ∧
    C = -4 ∧
    D = 13 ∧
    E = 11 ∧
    Int.gcd A E = 1 ∧
    Int.gcd C E = 1 ∧
    ¬∃ (k : ℤ), k > 1 ∧ k ^ 2 ∣ B ∧
    ¬∃ (k : ℤ), k > 1 ∧ k ^ 2 ∣ D :=
by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l2060_206088


namespace NUMINAMATH_CALUDE_transportation_charges_calculation_l2060_206046

theorem transportation_charges_calculation (purchase_price repair_cost selling_price : ℕ) 
  (h1 : purchase_price = 14000)
  (h2 : repair_cost = 5000)
  (h3 : selling_price = 30000)
  (h4 : selling_price = (purchase_price + repair_cost + transportation_charges) * 3 / 2) :
  transportation_charges = 1000 :=
by
  sorry

#check transportation_charges_calculation

end NUMINAMATH_CALUDE_transportation_charges_calculation_l2060_206046


namespace NUMINAMATH_CALUDE_floor_equation_solution_l2060_206017

theorem floor_equation_solution (x : ℝ) :
  ⌊x * (⌊x⌋ - 1)⌋ = 8 ↔ 4 ≤ x ∧ x < 4.5 :=
by sorry

end NUMINAMATH_CALUDE_floor_equation_solution_l2060_206017


namespace NUMINAMATH_CALUDE_trapezoid_shorter_base_l2060_206044

/-- Represents a trapezoid -/
structure Trapezoid where
  long_base : ℝ
  short_base : ℝ
  midpoint_diagonal_length : ℝ

/-- 
Given a trapezoid where:
- The line joining the midpoints of the diagonals has length 5
- The longer base is 105
Then the shorter base must be 95
-/
theorem trapezoid_shorter_base 
  (t : Trapezoid) 
  (h1 : t.midpoint_diagonal_length = 5) 
  (h2 : t.long_base = 105) : 
  t.short_base = 95 := by
sorry

end NUMINAMATH_CALUDE_trapezoid_shorter_base_l2060_206044


namespace NUMINAMATH_CALUDE_quadratic_integer_value_l2060_206015

theorem quadratic_integer_value (a b c : ℚ) :
  (∀ n : ℤ, ∃ m : ℤ, a * n^2 + b * n + c = m) ↔ 
  (∃ k l m : ℤ, 2 * a = k ∧ a + b = l ∧ c = m) :=
sorry

end NUMINAMATH_CALUDE_quadratic_integer_value_l2060_206015


namespace NUMINAMATH_CALUDE_constant_term_g_l2060_206004

-- Define polynomials f, g, and h
variable (f g h : ℝ → ℝ)

-- Define the condition that h is the product of f and g
variable (h_def : ∀ x, h x = f x * g x)

-- Define the constant terms of f and h
variable (f_const : f 0 = 6)
variable (h_const : h 0 = -18)

-- Theorem statement
theorem constant_term_g : g 0 = -3 := by
  sorry

end NUMINAMATH_CALUDE_constant_term_g_l2060_206004


namespace NUMINAMATH_CALUDE_sam_study_time_l2060_206095

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- The time Sam spends studying Science in minutes -/
def science_time : ℕ := 60

/-- The time Sam spends studying Math in minutes -/
def math_time : ℕ := 80

/-- The time Sam spends studying Literature in minutes -/
def literature_time : ℕ := 40

/-- The total time Sam spends studying in hours -/
def total_study_time : ℚ :=
  (science_time + math_time + literature_time : ℚ) / minutes_per_hour

theorem sam_study_time :
  total_study_time = 3 := by sorry

end NUMINAMATH_CALUDE_sam_study_time_l2060_206095


namespace NUMINAMATH_CALUDE_tennis_balls_count_l2060_206026

-- Define the ratio of red to blue balls
def red_to_blue_ratio : ℚ := 3 / 7

-- Define the percentage of red balls that are tennis balls
def red_tennis_percentage : ℚ := 70 / 100

-- Define the percentage of blue balls that are tennis balls
def blue_tennis_percentage : ℚ := 30 / 100

-- Theorem statement
theorem tennis_balls_count (R : ℚ) :
  let B := R * (1 / red_to_blue_ratio)
  let red_tennis := R * red_tennis_percentage
  let blue_tennis := B * blue_tennis_percentage
  red_tennis + blue_tennis = 1.4 * R :=
by sorry

end NUMINAMATH_CALUDE_tennis_balls_count_l2060_206026


namespace NUMINAMATH_CALUDE_delta_u_zero_iff_k_ge_five_l2060_206000

def u (n : ℕ) : ℕ := n^4 + n^2

def Δ : (ℕ → ℕ) → (ℕ → ℕ)
  | f => fun n => f (n + 1) - f n

def iterateΔ : ℕ → (ℕ → ℕ) → (ℕ → ℕ)
  | 0 => id
  | k + 1 => Δ ∘ iterateΔ k

theorem delta_u_zero_iff_k_ge_five :
  ∀ k : ℕ, (∀ n : ℕ, iterateΔ k u n = 0) ↔ k ≥ 5 := by sorry

end NUMINAMATH_CALUDE_delta_u_zero_iff_k_ge_five_l2060_206000


namespace NUMINAMATH_CALUDE_five_circle_five_num_five_circle_seven_num_l2060_206085

-- Define the structure of the diagram
structure Diagram :=
  (n : ℕ)  -- number of circles
  (m : ℕ)  -- maximum number to be used

-- Define a valid filling of the diagram
def ValidFilling (d : Diagram) := Fin d.m → Fin d.n

-- Define the number of valid fillings
def NumValidFillings (d : Diagram) : ℕ := sorry

-- Theorem for the case with 5 circles and numbers 1 to 5
theorem five_circle_five_num :
  ∀ d : Diagram, d.n = 5 ∧ d.m = 5 → NumValidFillings d = 8 := by sorry

-- Theorem for the case with 5 circles and numbers 1 to 7
theorem five_circle_seven_num :
  ∀ d : Diagram, d.n = 5 ∧ d.m = 7 → NumValidFillings d = 48 := by sorry

end NUMINAMATH_CALUDE_five_circle_five_num_five_circle_seven_num_l2060_206085


namespace NUMINAMATH_CALUDE_luke_friend_games_l2060_206027

/-- The number of games Luke bought from his friend -/
def games_from_friend : ℕ := sorry

/-- The number of games Luke bought at the garage sale -/
def games_from_garage_sale : ℕ := 2

/-- The number of games that didn't work -/
def broken_games : ℕ := 2

/-- The number of good games Luke ended up with -/
def good_games : ℕ := 2

theorem luke_friend_games :
  games_from_friend + games_from_garage_sale - broken_games = good_games ∧
  games_from_friend = 2 :=
sorry

end NUMINAMATH_CALUDE_luke_friend_games_l2060_206027


namespace NUMINAMATH_CALUDE_marias_cupcakes_l2060_206090

/-- 
Given that Maria made some cupcakes, sold 5, made 10 more, and ended up with 24 cupcakes,
this theorem proves that she initially made 19 cupcakes.
-/
theorem marias_cupcakes (x : ℕ) 
  (h : x - 5 + 10 = 24) : x = 19 := by
  sorry

end NUMINAMATH_CALUDE_marias_cupcakes_l2060_206090


namespace NUMINAMATH_CALUDE_orange_cost_27_pounds_l2060_206001

/-- The cost of oranges in dollars per 3 pounds -/
def orange_rate : ℚ := 3

/-- The weight of oranges in pounds that we want to buy -/
def orange_weight : ℚ := 27

/-- The cost of oranges for a given weight -/
def orange_cost (weight : ℚ) : ℚ := (weight / 3) * orange_rate

theorem orange_cost_27_pounds :
  orange_cost orange_weight = 27 := by sorry

end NUMINAMATH_CALUDE_orange_cost_27_pounds_l2060_206001


namespace NUMINAMATH_CALUDE_enclosure_blocks_l2060_206032

/-- Calculates the number of blocks required for a rectangular enclosure --/
def blocks_required (length width height : ℕ) : ℕ :=
  let external_volume := length * width * height
  let internal_length := length - 2
  let internal_width := width - 2
  let internal_height := height - 2
  let internal_volume := internal_length * internal_width * internal_height
  external_volume - internal_volume

/-- Proves that the number of blocks required for the given dimensions is 598 --/
theorem enclosure_blocks : blocks_required 15 13 6 = 598 := by
  sorry

end NUMINAMATH_CALUDE_enclosure_blocks_l2060_206032


namespace NUMINAMATH_CALUDE_special_set_odd_sum_l2060_206098

def SpecialSet (S : Set (ℕ × ℕ)) : Prop :=
  (1, 0) ∈ S ∧
  ∀ (i j : ℕ), (i, j) ∈ S →
    (((i + 1, j) ∈ S ∧ (i, j + 1) ∉ S ∧ (i - 1, j - 1) ∉ S) ∨
     ((i + 1, j) ∉ S ∧ (i, j + 1) ∈ S ∧ (i - 1, j - 1) ∉ S) ∨
     ((i + 1, j) ∉ S ∧ (i, j + 1) ∉ S ∧ (i - 1, j - 1) ∈ S))

theorem special_set_odd_sum (S : Set (ℕ × ℕ)) (h : SpecialSet S) :
  ∀ (i j : ℕ), (i, j) ∈ S → Odd (i + j) := by
  sorry

end NUMINAMATH_CALUDE_special_set_odd_sum_l2060_206098


namespace NUMINAMATH_CALUDE_direction_cosines_sum_of_squares_l2060_206061

/-- Direction cosines of a vector in 3D space -/
structure DirectionCosines where
  α : ℝ
  β : ℝ
  γ : ℝ

/-- Theorem: The sum of squares of direction cosines equals 1 -/
theorem direction_cosines_sum_of_squares (dc : DirectionCosines) : 
  dc.α^2 + dc.β^2 + dc.γ^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_direction_cosines_sum_of_squares_l2060_206061


namespace NUMINAMATH_CALUDE_value_of_y_l2060_206087

theorem value_of_y : ∃ y : ℚ, (3 * y) / 7 = 12 ∧ y = 28 := by
  sorry

end NUMINAMATH_CALUDE_value_of_y_l2060_206087


namespace NUMINAMATH_CALUDE_employee_hourly_rate_l2060_206082

/-- Proves that the hourly pay rate for employees is $12 given the company's workforce and payment information. -/
theorem employee_hourly_rate (initial_employees : ℕ) (hours_per_day : ℕ) (days_per_week : ℕ) (weeks_per_month : ℕ) (additional_employees : ℕ) (total_monthly_payment : ℕ) :
  initial_employees = 500 →
  hours_per_day = 10 →
  days_per_week = 5 →
  weeks_per_month = 4 →
  additional_employees = 200 →
  total_monthly_payment = 1680000 →
  (total_monthly_payment : ℚ) / ((initial_employees + additional_employees) * hours_per_day * days_per_week * weeks_per_month) = 12 := by
  sorry


end NUMINAMATH_CALUDE_employee_hourly_rate_l2060_206082


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l2060_206057

theorem complex_modulus_problem (a : ℝ) (z : ℂ) : 
  z = (1 - a * Complex.I) / (1 + Complex.I) →
  z.re = -2 →
  Complex.abs z = Real.sqrt 13 :=
sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l2060_206057


namespace NUMINAMATH_CALUDE_cost_of_pens_l2060_206078

/-- Given a box of 150 pens costing $45, prove that the cost of 3600 pens is $1080 -/
theorem cost_of_pens (box_size : ℕ) (box_cost : ℚ) (total_pens : ℕ) :
  box_size = 150 →
  box_cost = 45 →
  total_pens = 3600 →
  (total_pens : ℚ) / box_size * box_cost = 1080 :=
by
  sorry


end NUMINAMATH_CALUDE_cost_of_pens_l2060_206078


namespace NUMINAMATH_CALUDE_division_with_remainder_l2060_206067

theorem division_with_remainder (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) :
  dividend = divisor * quotient + remainder →
  remainder < divisor →
  dividend = 11 →
  divisor = 3 →
  remainder = 2 →
  quotient = 3 := by
sorry

end NUMINAMATH_CALUDE_division_with_remainder_l2060_206067


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2060_206013

theorem complex_equation_solution :
  ∃ (x : ℂ), 5 - 3 * Complex.I * x = -2 + 6 * Complex.I * x ∧ x = -7 * Complex.I / 9 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2060_206013


namespace NUMINAMATH_CALUDE_smallest_bob_number_l2060_206030

def alice_number : ℕ := 24

def has_all_prime_factors (n m : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → (p ∣ n → p ∣ m)

theorem smallest_bob_number :
  ∃ (bob_number : ℕ),
    bob_number > 0 ∧
    has_all_prime_factors alice_number bob_number ∧
    (∀ m : ℕ, m > 0 → has_all_prime_factors alice_number m → bob_number ≤ m) ∧
    bob_number = 6 :=
sorry

end NUMINAMATH_CALUDE_smallest_bob_number_l2060_206030


namespace NUMINAMATH_CALUDE_max_value_of_f_l2060_206010

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := -1/3 * x^3 + 1/2 * x^2 + 2*a*x

theorem max_value_of_f (a : ℝ) (h1 : 0 < a) (h2 : a < 2) :
  (∀ x ∈ Set.Icc 1 4, f a x ≥ -16/3) →
  (∃ x ∈ Set.Icc 1 4, f a x = -16/3) →
  (∃ x ∈ Set.Icc 1 4, f a x = 10/3) ∧
  (∀ x ∈ Set.Icc 1 4, f a x ≤ 10/3) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_f_l2060_206010


namespace NUMINAMATH_CALUDE_sqrt_sum_2014_l2060_206060

theorem sqrt_sum_2014 (a b c : ℕ) : 
  Real.sqrt a + Real.sqrt b + Real.sqrt c = Real.sqrt 2014 →
  ((a = 0 ∧ b = 0 ∧ c = 2014) ∨
   (a = 0 ∧ b = 2014 ∧ c = 0) ∨
   (a = 2014 ∧ b = 0 ∧ c = 0)) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_2014_l2060_206060


namespace NUMINAMATH_CALUDE_square_sum_nonzero_iff_one_nonzero_l2060_206062

theorem square_sum_nonzero_iff_one_nonzero (a b : ℝ) : 
  a^2 + b^2 ≠ 0 ↔ a ≠ 0 ∨ b ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_nonzero_iff_one_nonzero_l2060_206062


namespace NUMINAMATH_CALUDE_world_grain_ratio_l2060_206064

def world_grain_supply : ℝ := 1800000
def world_grain_demand : ℝ := 2400000

theorem world_grain_ratio : 
  world_grain_supply / world_grain_demand = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_world_grain_ratio_l2060_206064


namespace NUMINAMATH_CALUDE_sculpture_cost_in_cny_l2060_206034

/-- Exchange rate from US dollars to Namibian dollars -/
def usd_to_nad : ℝ := 8

/-- Exchange rate from US dollars to Chinese yuan -/
def usd_to_cny : ℝ := 5

/-- Cost of the sculpture in Namibian dollars -/
def sculpture_cost_nad : ℝ := 160

/-- Theorem stating the cost of the sculpture in Chinese yuan -/
theorem sculpture_cost_in_cny :
  (sculpture_cost_nad / usd_to_nad) * usd_to_cny = 100 := by
  sorry

end NUMINAMATH_CALUDE_sculpture_cost_in_cny_l2060_206034


namespace NUMINAMATH_CALUDE_rectangle_longest_side_l2060_206019

/-- A rectangle with perimeter 240 feet and area equal to eight times its perimeter has its longest side equal to 101 feet. -/
theorem rectangle_longest_side (l w : ℝ) : 
  l > 0 ∧ w > 0 ∧  -- positive length and width
  2 * (l + w) = 240 ∧  -- perimeter is 240 feet
  l * w = 8 * 240 →  -- area is eight times perimeter
  max l w = 101 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_longest_side_l2060_206019


namespace NUMINAMATH_CALUDE_motor_lifespan_probability_l2060_206014

variable (X : Real → Real)  -- Random variable representing motor lifespan

-- Define the expected value of X
def expected_value : Real := 4

-- Define the theorem
theorem motor_lifespan_probability :
  (∫ x, X x) = expected_value →  -- The expected value of X is 4
  (∫ x in {x | x < 20}, X x) ≥ 0.8 := by
  sorry

end NUMINAMATH_CALUDE_motor_lifespan_probability_l2060_206014


namespace NUMINAMATH_CALUDE_A_intersect_B_l2060_206077

def A : Set ℝ := {-1, 0, 1, 2, 3}

def B : Set ℝ := {x : ℝ | (x + 1) * (x - 2) < 0}

theorem A_intersect_B : A ∩ B = {0, 1} := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_l2060_206077


namespace NUMINAMATH_CALUDE_stickers_at_end_of_week_l2060_206051

def initial_stickers : ℝ := 39.0
def given_away_stickers : ℝ := 22.0

theorem stickers_at_end_of_week : 
  initial_stickers - given_away_stickers = 17.0 := by
  sorry

end NUMINAMATH_CALUDE_stickers_at_end_of_week_l2060_206051


namespace NUMINAMATH_CALUDE_jackson_hermit_crabs_l2060_206005

/-- Given the conditions of Jackson's souvenir collection, prove that he collected 45 hermit crabs. -/
theorem jackson_hermit_crabs :
  ∀ (hermit_crabs spiral_shells starfish : ℕ),
  spiral_shells = 3 * hermit_crabs →
  starfish = 2 * spiral_shells →
  hermit_crabs + spiral_shells + starfish = 450 →
  hermit_crabs = 45 := by
sorry

end NUMINAMATH_CALUDE_jackson_hermit_crabs_l2060_206005


namespace NUMINAMATH_CALUDE_food_supply_duration_l2060_206043

/-- Proves that given a food supply for 760 men that lasts for x days, 
    if after 2 days 1140 more men join and the food lasts for 8 more days, 
    then x = 20. -/
theorem food_supply_duration (x : ℝ) : 
  (760 * x = (760 + 1140) * 8) → x = 20 := by
  sorry

end NUMINAMATH_CALUDE_food_supply_duration_l2060_206043


namespace NUMINAMATH_CALUDE_apple_tv_cost_l2060_206076

theorem apple_tv_cost (iphone_count : ℕ) (iphone_cost : ℝ)
                      (ipad_count : ℕ) (ipad_cost : ℝ)
                      (apple_tv_count : ℕ)
                      (total_avg_cost : ℝ) :
  iphone_count = 100 →
  iphone_cost = 1000 →
  ipad_count = 20 →
  ipad_cost = 900 →
  apple_tv_count = 80 →
  total_avg_cost = 670 →
  (iphone_count * iphone_cost + ipad_count * ipad_cost + apple_tv_count * (iphone_count * iphone_cost + ipad_count * ipad_cost + apple_tv_count * 200) / (iphone_count + ipad_count + apple_tv_count)) / (iphone_count + ipad_count + apple_tv_count) = total_avg_cost →
  (iphone_count * iphone_cost + ipad_count * ipad_cost + apple_tv_count * 200) / (iphone_count + ipad_count + apple_tv_count) = total_avg_cost :=
by sorry

#check apple_tv_cost

end NUMINAMATH_CALUDE_apple_tv_cost_l2060_206076


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2060_206037

theorem complex_fraction_simplification :
  ∀ (i : ℂ), i^2 = -1 →
  (11 - 3*i) / (1 + 2*i) = 3 - 5*i := by
sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2060_206037


namespace NUMINAMATH_CALUDE_stock_sale_loss_l2060_206052

/-- Calculates the overall loss from selling a stock with given conditions -/
def calculate_overall_loss (stock_worth : ℝ) : ℝ :=
  let profit_portion := 0.2 * stock_worth
  let loss_portion := 0.8 * stock_worth
  let profit := 0.1 * profit_portion
  let loss := 0.05 * loss_portion
  loss - profit

/-- Theorem stating the overall loss for the given stock and conditions -/
theorem stock_sale_loss (stock_worth : ℝ) (h : stock_worth = 12500) :
  calculate_overall_loss stock_worth = 250 := by
  sorry

end NUMINAMATH_CALUDE_stock_sale_loss_l2060_206052


namespace NUMINAMATH_CALUDE_power_of_product_l2060_206011

theorem power_of_product (a : ℝ) : (2 * a^2)^3 = 8 * a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_l2060_206011


namespace NUMINAMATH_CALUDE_forty_students_in_music_l2060_206092

/-- Represents the number of students in various categories in a high school. -/
structure SchoolData where
  total : ℕ
  art : ℕ
  both : ℕ
  neither : ℕ

/-- Calculates the number of students taking music based on the given school data. -/
def studentsInMusic (data : SchoolData) : ℕ :=
  data.total - data.neither - (data.art - data.both)

/-- Theorem stating that given the specific school data, 40 students are taking music. -/
theorem forty_students_in_music :
  let data : SchoolData := {
    total := 500,
    art := 20,
    both := 10,
    neither := 450
  }
  studentsInMusic data = 40 := by
  sorry


end NUMINAMATH_CALUDE_forty_students_in_music_l2060_206092


namespace NUMINAMATH_CALUDE_investment_split_l2060_206069

/-- Proves the amount invested at 6% given total investment, interest rates, and total interest earned --/
theorem investment_split (total_investment : ℝ) (rate1 rate2 : ℝ) (total_interest : ℝ) 
  (h1 : total_investment = 15000)
  (h2 : rate1 = 0.06)
  (h3 : rate2 = 0.075)
  (h4 : total_interest = 1023)
  (h5 : ∃ (x y : ℝ), x + y = total_investment ∧ 
                     rate1 * x + rate2 * y = total_interest) :
  ∃ (x : ℝ), x = 6800 ∧ 
              ∃ (y : ℝ), y = total_investment - x ∧
                          rate1 * x + rate2 * y = total_interest :=
sorry

end NUMINAMATH_CALUDE_investment_split_l2060_206069


namespace NUMINAMATH_CALUDE_inequality_holds_iff_x_eq_pi_div_2_l2060_206012

theorem inequality_holds_iff_x_eq_pi_div_2 : 
  ∀ x : ℝ, 0 < x → x < π → 
  ((8 / (3 * Real.sin x - Real.sin (3 * x))) + 3 * (Real.sin x)^2 ≤ 5 ↔ x = π / 2) := by
sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_x_eq_pi_div_2_l2060_206012


namespace NUMINAMATH_CALUDE_problem_solution_l2060_206007

theorem problem_solution (m n : ℝ) 
  (h1 : m^2 + 2*m*n = 384) 
  (h2 : 3*m*n + 2*n^2 = 560) : 
  2*m^2 + 13*m*n + 6*n^2 - 444 = 2004 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2060_206007


namespace NUMINAMATH_CALUDE_car_travel_distance_l2060_206018

/-- Proves that a car traveling at a constant rate of 3 kilometers every 4 minutes
    will cover 90 kilometers in 2 hours. -/
theorem car_travel_distance (rate : ℝ) (time : ℝ) : 
  rate = 3 / 4 → time = 120 → rate * time = 90 := by
  sorry

end NUMINAMATH_CALUDE_car_travel_distance_l2060_206018


namespace NUMINAMATH_CALUDE_fraction_power_five_l2060_206033

theorem fraction_power_five : (3 / 4) ^ 5 = 243 / 1024 := by
  sorry

end NUMINAMATH_CALUDE_fraction_power_five_l2060_206033


namespace NUMINAMATH_CALUDE_f_property_l2060_206049

noncomputable def f (x : ℝ) : ℝ := (4^x) / (4^x + 2)

theorem f_property (x : ℝ) :
  f x + f (1 - x) = 1 ∧
  (2 * (f x)^2 < f (1 - x) ↔ x < 1/2) :=
by sorry

end NUMINAMATH_CALUDE_f_property_l2060_206049


namespace NUMINAMATH_CALUDE_sports_competition_results_l2060_206083

/-- Represents the outcome of a single event -/
inductive EventOutcome
| SchoolAWins
| SchoolBWins

/-- Represents the outcome of the entire championship -/
inductive ChampionshipOutcome
| SchoolAWins
| SchoolBWins

/-- The probability of School A winning each event -/
def probSchoolAWins : Fin 3 → ℝ
| 0 => 0.5
| 1 => 0.4
| 2 => 0.8

/-- The score awarded for winning an event -/
def winningScore : ℕ := 10

/-- Calculate the probability of School A winning the championship -/
def probSchoolAWinsChampionship : ℝ := sorry

/-- Calculate the expectation of School B's total score -/
def expectationSchoolBScore : ℝ := sorry

/-- Theorem stating the main results -/
theorem sports_competition_results :
  probSchoolAWinsChampionship = 0.6 ∧ expectationSchoolBScore = 13 := by sorry

end NUMINAMATH_CALUDE_sports_competition_results_l2060_206083


namespace NUMINAMATH_CALUDE_two_inequalities_true_l2060_206058

theorem two_inequalities_true (x y a b : ℝ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (ha : a ≠ 0) (hb : b ≠ 0)
  (hxa : x^2 < a^2) (hyb : y^2 < b^2) : 
  ∃! n : ℕ, n = 2 ∧ 
    (n = (if x^2 + y^2 < a^2 + b^2 then 1 else 0) +
         (if x^2 - y^2 < a^2 - b^2 then 1 else 0) +
         (if x^2 * y^2 < a^2 * b^2 then 1 else 0) +
         (if x^2 / y^2 < a^2 / b^2 then 1 else 0)) :=
by sorry

end NUMINAMATH_CALUDE_two_inequalities_true_l2060_206058


namespace NUMINAMATH_CALUDE_harold_marbles_distribution_l2060_206016

def marble_distribution (total_marbles : ℕ) (kept_marbles : ℕ) (num_friends : ℕ) : ℕ :=
  (total_marbles - kept_marbles) / num_friends

theorem harold_marbles_distribution :
  marble_distribution 100 20 5 = 16 := by
  sorry

end NUMINAMATH_CALUDE_harold_marbles_distribution_l2060_206016


namespace NUMINAMATH_CALUDE_expression_equals_36_l2060_206054

theorem expression_equals_36 : ∃ (expr : ℝ), 
  (expr = 13 * (3 - 3 / 13)) ∧ (expr = 36) :=
by sorry

end NUMINAMATH_CALUDE_expression_equals_36_l2060_206054


namespace NUMINAMATH_CALUDE_expression_evaluation_l2060_206002

theorem expression_evaluation (x y : ℝ) (h : (x - 2)^2 + |y - 3| = 0) :
  ((x - 2*y) * (x + 2*y) - (x - y)^2 + y * (y + 2*x)) / (-2*y) = 2 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2060_206002


namespace NUMINAMATH_CALUDE_max_value_of_f_max_value_of_expression_one_is_max_value_l2060_206071

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| - 2 * |x + 1|

-- Theorem for the maximum value of f
theorem max_value_of_f : ∃ m : ℝ, m = 2 ∧ ∀ x : ℝ, f x ≤ m := by sorry

-- Theorem for the maximum value of ab + 2bc
theorem max_value_of_expression {a b c : ℝ} 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : a^2 + 3*b^2 + 2*c^2 = 2) : 
  a*b + 2*b*c ≤ 1 := by sorry

-- Theorem stating that 1 is indeed the maximum value
theorem one_is_max_value {a b c : ℝ} 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : a^2 + 3*b^2 + 2*c^2 = 2) : 
  ∃ a₀ b₀ c₀ : ℝ, a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧ 
  a₀^2 + 3*b₀^2 + 2*c₀^2 = 2 ∧ 
  a₀*b₀ + 2*b₀*c₀ = 1 := by sorry

end NUMINAMATH_CALUDE_max_value_of_f_max_value_of_expression_one_is_max_value_l2060_206071


namespace NUMINAMATH_CALUDE_one_perm_scheduled_l2060_206096

/-- Represents the salon's pricing and scheduling for a day --/
structure SalonDay where
  haircut_price : ℕ
  perm_price : ℕ
  dye_job_price : ℕ
  dye_cost : ℕ
  num_haircuts : ℕ
  num_dye_jobs : ℕ
  tips : ℕ
  total_revenue : ℕ

/-- Calculates the number of perms scheduled given the salon day information --/
def calculate_perms (day : SalonDay) : ℕ :=
  let revenue_without_perms := day.haircut_price * day.num_haircuts +
                               (day.dye_job_price - day.dye_cost) * day.num_dye_jobs +
                               day.tips
  (day.total_revenue - revenue_without_perms) / day.perm_price

/-- Theorem stating that for the given salon day, exactly one perm is scheduled --/
theorem one_perm_scheduled (day : SalonDay) 
  (h1 : day.haircut_price = 30)
  (h2 : day.perm_price = 40)
  (h3 : day.dye_job_price = 60)
  (h4 : day.dye_cost = 10)
  (h5 : day.num_haircuts = 4)
  (h6 : day.num_dye_jobs = 2)
  (h7 : day.tips = 50)
  (h8 : day.total_revenue = 310) :
  calculate_perms day = 1 := by
  sorry

end NUMINAMATH_CALUDE_one_perm_scheduled_l2060_206096


namespace NUMINAMATH_CALUDE_not_divisible_by_five_l2060_206038

theorem not_divisible_by_five (n : ℤ) : ¬ (5 ∣ (n^2 - 8)) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_five_l2060_206038


namespace NUMINAMATH_CALUDE_joels_age_proof_l2060_206025

/-- Joel's current age -/
def joels_current_age : ℕ := 5

/-- Joel's dad's current age -/
def dads_current_age : ℕ := 32

/-- The age Joel will be when his dad is twice his age -/
def joels_future_age : ℕ := 27

theorem joels_age_proof :
  joels_current_age = 5 ∧
  dads_current_age = 32 ∧
  joels_future_age = 27 ∧
  dads_current_age + (joels_future_age - joels_current_age) = 2 * joels_future_age :=
by sorry

end NUMINAMATH_CALUDE_joels_age_proof_l2060_206025


namespace NUMINAMATH_CALUDE_left_handed_fraction_l2060_206059

/-- Represents the number of participants from each world -/
structure Participants where
  red : ℚ
  blue : ℚ
  green : ℚ

/-- Calculates the total number of participants -/
def total_participants (p : Participants) : ℚ :=
  p.red + p.blue + p.green

/-- Calculates the number of left-handed participants -/
def left_handed_participants (p : Participants) : ℚ :=
  p.red / 3 + 2 * p.blue / 3

/-- The main theorem stating the fraction of left-handed participants -/
theorem left_handed_fraction (p : Participants) 
  (h1 : p.red = 3 * p.blue / 2)  -- ratio of red to blue is 3:2
  (h2 : p.blue = 5 * p.green / 4)  -- ratio of blue to green is 5:4
  : left_handed_participants p / total_participants p = 35 / 99 := by
  sorry

end NUMINAMATH_CALUDE_left_handed_fraction_l2060_206059


namespace NUMINAMATH_CALUDE_train_passing_platform_l2060_206047

-- Define the train's length
def train_length : ℝ := 240

-- Define the time to pass a pole
def time_to_pass_pole : ℝ := 24

-- Define the platform length
def platform_length : ℝ := 650

-- Theorem statement
theorem train_passing_platform :
  let train_speed := train_length / time_to_pass_pole
  let total_distance := train_length + platform_length
  total_distance / train_speed = 89 := by
sorry

end NUMINAMATH_CALUDE_train_passing_platform_l2060_206047


namespace NUMINAMATH_CALUDE_tmobile_additional_line_cost_l2060_206042

theorem tmobile_additional_line_cost 
  (tmobile_base : ℕ) 
  (mmobile_base : ℕ) 
  (mmobile_additional : ℕ) 
  (total_lines : ℕ) 
  (price_difference : ℕ) 
  (h1 : tmobile_base = 50)
  (h2 : mmobile_base = 45)
  (h3 : mmobile_additional = 14)
  (h4 : total_lines = 5)
  (h5 : price_difference = 11)
  (h6 : tmobile_base + (total_lines - 2) * x = 
        mmobile_base + (total_lines - 2) * mmobile_additional + price_difference) :
  x = 16 := by
  sorry

#check tmobile_additional_line_cost

end NUMINAMATH_CALUDE_tmobile_additional_line_cost_l2060_206042


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l2060_206024

theorem arithmetic_sequence_fifth_term 
  (x y : ℝ) 
  (seq : ℕ → ℝ)
  (h1 : seq 0 = x + 2*y)
  (h2 : seq 1 = x - 2*y)
  (h3 : seq 2 = x^2 * y)
  (h4 : seq 3 = x / (2*y))
  (h_arith : ∀ n, seq (n+1) - seq n = seq 1 - seq 0)
  (hy : y = 1)
  (hx : x = 20) :
  seq 4 = 6 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l2060_206024


namespace NUMINAMATH_CALUDE_book_purchase_total_price_l2060_206006

/-- Calculates the total price of books given the following conditions:
  * Total number of books
  * Number of math books
  * Price of a math book
  * Price of a history book
-/
def total_price (total_books : ℕ) (math_books : ℕ) (math_price : ℕ) (history_price : ℕ) : ℕ :=
  math_books * math_price + (total_books - math_books) * history_price

/-- Theorem stating that given the specific conditions in the problem,
    the total price of books is $390. -/
theorem book_purchase_total_price :
  total_price 80 10 4 5 = 390 := by
  sorry

end NUMINAMATH_CALUDE_book_purchase_total_price_l2060_206006


namespace NUMINAMATH_CALUDE_polynomial_equality_l2060_206063

theorem polynomial_equality (s t : ℝ) : -1/4 * s * t + 0.25 * s * t = 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l2060_206063


namespace NUMINAMATH_CALUDE_randys_final_amount_l2060_206068

/-- Calculates Randy's remaining money after a series of transactions --/
def randys_remaining_money (initial_amount : ℝ) (smith_gift : ℝ) 
  (sally_percentage : ℝ) (stock_percentage : ℝ) (crypto_percentage : ℝ) : ℝ :=
  let new_total := initial_amount + smith_gift
  let after_sally := new_total * (1 - sally_percentage)
  let after_stocks := after_sally * (1 - stock_percentage)
  after_stocks * (1 - crypto_percentage)

/-- Theorem stating that Randy's remaining money is $1,008 --/
theorem randys_final_amount :
  randys_remaining_money 3000 200 0.25 0.40 0.30 = 1008 := by
  sorry

#eval randys_remaining_money 3000 200 0.25 0.40 0.30

end NUMINAMATH_CALUDE_randys_final_amount_l2060_206068


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2060_206097

/-- A geometric sequence with specific properties -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), ∀ (n : ℕ), a (n + 1) = r * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  a 4 + a 7 = 2 →
  a 5 * a 6 = -8 →
  a 1 + a 10 = -7 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2060_206097


namespace NUMINAMATH_CALUDE_lcm_factor_proof_l2060_206031

theorem lcm_factor_proof (A B : ℕ+) (X : ℕ+) (h1 : Nat.gcd A B = 23)
  (h2 : Nat.lcm A B = 23 * X * 16) (h3 : A = 368) : X = 1 := by
  sorry

end NUMINAMATH_CALUDE_lcm_factor_proof_l2060_206031


namespace NUMINAMATH_CALUDE_third_row_chairs_l2060_206070

def chair_sequence (n : ℕ) : ℕ → ℕ
  | 1 => 14
  | 2 => 23
  | 3 => n
  | 4 => 41
  | 5 => 50
  | 6 => 59
  | _ => 0

theorem third_row_chairs :
  ∃ n : ℕ, 
    chair_sequence n 2 - chair_sequence n 1 = 9 ∧
    chair_sequence n 4 - chair_sequence n 2 = 18 ∧
    chair_sequence n 5 - chair_sequence n 4 = 9 ∧
    chair_sequence n 6 - chair_sequence n 5 = 9 ∧
    n = 32 := by
  sorry

end NUMINAMATH_CALUDE_third_row_chairs_l2060_206070


namespace NUMINAMATH_CALUDE_all_integers_are_integers_l2060_206003

theorem all_integers_are_integers (n : ℕ) (a : Fin n → ℕ+) 
  (h : ∀ i j : Fin n, i ≠ j → 
    (((a i).val + (a j).val) / 2 : ℚ).den = 1 ∨ 
    (((a i).val * (a j).val : ℕ).sqrt : ℚ).den = 1) : 
  ∀ i : Fin n, (a i).val = (a i : ℕ) := by
sorry

end NUMINAMATH_CALUDE_all_integers_are_integers_l2060_206003


namespace NUMINAMATH_CALUDE_trigonometric_roots_theorem_l2060_206009

-- Define the equation and its roots
def equation (m : ℝ) (x : ℝ) : Prop := 8 * x^2 + 6 * m * x + 2 * m + 1 = 0

-- Define the interval for α
def alpha_in_interval (α : ℝ) : Prop := 0 < α ∧ α < Real.pi

-- Theorem statement
theorem trigonometric_roots_theorem (α : ℝ) (m : ℝ) 
  (h1 : alpha_in_interval α)
  (h2 : equation m (Real.sin α))
  (h3 : equation m (Real.cos α)) :
  m = -10/9 ∧ 
  (Real.cos α + Real.sin α) * Real.tan α / (1 - Real.tan α^2) = 11 * Real.sqrt 47 / 564 :=
sorry

end NUMINAMATH_CALUDE_trigonometric_roots_theorem_l2060_206009
