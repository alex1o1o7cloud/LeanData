import Mathlib

namespace convergence_of_derived_series_l1618_161833

theorem convergence_of_derived_series (a : ℕ → ℝ) 
  (h_monotonic : Monotone a) 
  (h_convergent : Summable a) :
  Summable (fun n => n * (a n - a (n + 1))) :=
sorry

end convergence_of_derived_series_l1618_161833


namespace roberto_salary_raise_l1618_161866

def starting_salary : ℝ := 80000
def previous_salary : ℝ := starting_salary * 1.4
def current_salary : ℝ := 134400

theorem roberto_salary_raise :
  (current_salary - previous_salary) / previous_salary * 100 = 20 := by
sorry

end roberto_salary_raise_l1618_161866


namespace distinct_scores_l1618_161821

def goals : ℕ := 7

def possible_scores (n : ℕ) : Finset ℕ :=
  Finset.image (λ y : ℕ => y + n) (Finset.range (n + 1))

theorem distinct_scores : Finset.card (possible_scores goals) = 8 := by
  sorry

end distinct_scores_l1618_161821


namespace curve_C_properties_l1618_161826

/-- Definition of the curve C -/
def curve_C (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / (25 - k) + p.2^2 / (k - 9) = 1}

/-- Definition of an ellipse -/
def is_ellipse (S : Set (ℝ × ℝ)) : Prop :=
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ ∀ p ∈ S, p.1^2 / a^2 + p.2^2 / b^2 = 1

/-- Definition of a hyperbola with foci on the x-axis -/
def is_hyperbola_x_axis (S : Set (ℝ × ℝ)) : Prop :=
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ ∀ p ∈ S, p.1^2 / a^2 - p.2^2 / b^2 = 1

theorem curve_C_properties (k : ℝ) :
  (9 < k ∧ k < 25 → is_ellipse (curve_C k)) ∧
  (is_hyperbola_x_axis (curve_C k) → k < 9) :=
sorry

end curve_C_properties_l1618_161826


namespace G_in_third_quadrant_implies_x_negative_l1618_161894

/-- A point in the rectangular coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of a point being in the third quadrant -/
def isInThirdQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y < 0

/-- The specific point G(x, x-5) -/
def G (x : ℝ) : Point :=
  { x := x, y := x - 5 }

theorem G_in_third_quadrant_implies_x_negative (x : ℝ) :
  isInThirdQuadrant (G x) → x < 0 := by
  sorry

end G_in_third_quadrant_implies_x_negative_l1618_161894


namespace roots_of_polynomial_l1618_161808

def p (x : ℝ) : ℝ := 4 * x^4 + 11 * x^3 - 37 * x^2 + 18 * x

theorem roots_of_polynomial : 
  (p 0 = 0) ∧ (p (1/2) = 0) ∧ (p (3/2) = 0) ∧ (p (-6) = 0) :=
by sorry

end roots_of_polynomial_l1618_161808


namespace vessel_volume_ratio_l1618_161868

theorem vessel_volume_ratio : 
  ∀ (V₁ V₂ : ℝ), V₁ > 0 → V₂ > 0 →
  (3/4 : ℝ) * V₁ = (5/8 : ℝ) * V₂ →
  V₁ / V₂ = 5/6 := by
sorry

end vessel_volume_ratio_l1618_161868


namespace cans_needed_proof_l1618_161889

/-- The number of cans Martha collected -/
def martha_cans : ℕ := 90

/-- The number of cans Diego collected -/
def diego_cans : ℕ := martha_cans / 2 + 10

/-- The total number of cans needed for the project -/
def project_goal : ℕ := 150

/-- The number of additional cans needed -/
def additional_cans : ℕ := project_goal - (martha_cans + diego_cans)

theorem cans_needed_proof : additional_cans = 5 := by sorry

end cans_needed_proof_l1618_161889


namespace min_colors_shapes_for_distribution_centers_l1618_161874

theorem min_colors_shapes_for_distribution_centers :
  ∃ (C S : ℕ),
    (C = 3 ∧ S = 3) ∧
    (∀ (C' S' : ℕ),
      C' + C' * (C' - 1) / 2 + S' + S' * (S' - 1) ≥ 12 →
      C' ≥ C ∧ S' ≥ S) ∧
    C + C * (C - 1) / 2 + S + S * (S - 1) ≥ 12 :=
by sorry

end min_colors_shapes_for_distribution_centers_l1618_161874


namespace horse_net_earnings_zero_l1618_161820

/-- Represents the chessboard and the horse's movement rules --/
structure ChessboardGame where
  /-- The number of black squares the horse lands on --/
  black_squares : ℕ
  /-- The number of white squares the horse lands on --/
  white_squares : ℕ
  /-- Ensures the horse starts and ends on a white square --/
  start_end_white : white_squares > 0
  /-- Ensures the number of black and white squares are equal --/
  equal_squares : black_squares = white_squares
  /-- Represents the rule that the horse earns 2 carrots for each black square --/
  carrots_earned : ℕ := 2 * black_squares
  /-- Represents the rule that the horse pays 1 carrot for each move --/
  carrots_paid : ℕ := black_squares + white_squares

/-- The theorem stating that the net earnings of the horse is always 0 --/
theorem horse_net_earnings_zero (game : ChessboardGame) :
  game.carrots_earned - game.carrots_paid = 0 := by
  sorry

#check horse_net_earnings_zero

end horse_net_earnings_zero_l1618_161820


namespace M_geq_N_l1618_161813

theorem M_geq_N (x : ℝ) : 
  let M := 2 * x^2 - 12 * x + 15
  let N := x^2 - 8 * x + 11
  M ≥ N := by
sorry

end M_geq_N_l1618_161813


namespace arithmetic_sequence_fifth_term_l1618_161864

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_fifth_term
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum : a 2 + a 8 = 12) :
  a 5 = 6 := by
sorry

end arithmetic_sequence_fifth_term_l1618_161864


namespace power_division_equality_l1618_161845

theorem power_division_equality : (3^18 : ℕ) / (27^3 : ℕ) = 19683 := by
  sorry

end power_division_equality_l1618_161845


namespace largest_perimeter_incenter_l1618_161848

/-- A triangle in a plane with a fixed point P --/
structure TriangleWithFixedPoint where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  P : ℝ × ℝ

/-- Distance between two points in a plane --/
def distance (p q : ℝ × ℝ) : ℝ := sorry

/-- Perimeter of a triangle --/
def perimeter (t : TriangleWithFixedPoint) : ℝ :=
  distance t.A t.B + distance t.B t.C + distance t.C t.A

/-- Predicate to check if a point is the incenter of a triangle --/
def is_incenter (t : TriangleWithFixedPoint) : Prop := sorry

/-- Theorem: Triangles with largest perimeter have P as incenter --/
theorem largest_perimeter_incenter (t : TriangleWithFixedPoint) 
  (h1 : distance t.P t.A = 3)
  (h2 : distance t.P t.B = 5)
  (h3 : distance t.P t.C = 7) :
  (∀ t' : TriangleWithFixedPoint, 
    distance t'.P t'.A = 3 → 
    distance t'.P t'.B = 5 → 
    distance t'.P t'.C = 7 → 
    perimeter t ≥ perimeter t') ↔ 
  is_incenter t := by sorry

end largest_perimeter_incenter_l1618_161848


namespace zhuzhuxia_defeats_l1618_161823

/-- Represents the game state after a certain number of rounds -/
structure GameState where
  rounds : ℕ
  monsters_defeated : ℕ

/-- Theorem stating that after 8 rounds with 20 monsters defeated, Zhuzhuxia has been defeated 8 times -/
theorem zhuzhuxia_defeats (game : GameState) 
  (h1 : game.rounds = 8) 
  (h2 : game.monsters_defeated = 20) : 
  (game.rounds : ℕ) = 8 := by sorry

end zhuzhuxia_defeats_l1618_161823


namespace octagon_area_l1618_161899

/-- The area of a regular octagon inscribed in a square -/
theorem octagon_area (s : ℝ) (h : s = 4 + 2 * Real.sqrt 2) :
  let octagon_side := 2 * Real.sqrt 2
  let square_area := s^2
  let triangle_area := 2
  square_area - 4 * triangle_area = 16 + 8 * Real.sqrt 2 := by
  sorry

end octagon_area_l1618_161899


namespace romanov_family_savings_l1618_161832

/-- Represents the electricity cost calculation problem for the Romanov family -/
def electricity_cost_problem (multi_tariff_meter_cost : ℝ) (installation_cost : ℝ)
  (monthly_consumption : ℝ) (night_consumption : ℝ) (day_rate : ℝ) (night_rate : ℝ)
  (standard_rate : ℝ) : Prop :=
  let day_consumption := monthly_consumption - night_consumption
  let multi_tariff_yearly_cost := (night_consumption * night_rate + day_consumption * day_rate) * 12
  let standard_yearly_cost := monthly_consumption * standard_rate * 12
  let multi_tariff_total_cost := multi_tariff_meter_cost + installation_cost + multi_tariff_yearly_cost * 3
  let standard_total_cost := standard_yearly_cost * 3
  standard_total_cost - multi_tariff_total_cost = 3824

/-- The theorem stating the savings of the Romanov family -/
theorem romanov_family_savings :
  electricity_cost_problem 3500 1100 300 230 5.2 3.4 4.6 :=
by
  sorry

end romanov_family_savings_l1618_161832


namespace megans_eggs_per_meal_megans_eggs_problem_l1618_161896

theorem megans_eggs_per_meal (initial_eggs : ℕ) (neighbor_eggs : ℕ) 
  (omelet_eggs : ℕ) (cake_eggs : ℕ) (meals : ℕ) : ℕ :=
  let total_eggs := initial_eggs + neighbor_eggs
  let used_eggs := omelet_eggs + cake_eggs
  let remaining_eggs := total_eggs - used_eggs
  let eggs_after_aunt := remaining_eggs / 2
  let final_eggs := eggs_after_aunt
  final_eggs / meals

theorem megans_eggs_problem :
  megans_eggs_per_meal 12 12 2 4 3 = 3 := by
  sorry

end megans_eggs_per_meal_megans_eggs_problem_l1618_161896


namespace counterexample_exists_l1618_161891

theorem counterexample_exists : ∃ n : ℕ, 
  (¬ Nat.Prime n) ∧ (¬ Nat.Prime (n - 1) ∨ Nat.Prime (n - 2)) := by
  sorry

end counterexample_exists_l1618_161891


namespace quad_func_order_l1618_161880

/-- A quadratic function with the given properties -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  h_a_neg : a < 0
  f : ℝ → ℝ
  h_f_def : ∀ x, f x = a * x^2 + b * x + c
  h_f_sym : ∀ x, f (x + 2) = f (2 - x)

/-- The main theorem stating the order of function values -/
theorem quad_func_order (qf : QuadraticFunction) :
  qf.f (-1992) < qf.f 1992 ∧ qf.f 1992 < qf.f 0 := by
  sorry

end quad_func_order_l1618_161880


namespace arcade_spending_l1618_161892

theorem arcade_spending (allowance : ℝ) (f : ℝ) : 
  allowance = 3.75 →
  (1 - f) * allowance - (1/3) * ((1 - f) * allowance) = 1 →
  f = 3/5 := by
  sorry

end arcade_spending_l1618_161892


namespace sum_of_numbers_less_than_three_tenths_l1618_161804

def numbers : List ℚ := [8/10, 1/2, 9/10, 2/10, 1/3]

theorem sum_of_numbers_less_than_three_tenths :
  (numbers.filter (λ x => x < 3/10)).sum = 2/10 := by
  sorry

end sum_of_numbers_less_than_three_tenths_l1618_161804


namespace square_side_length_l1618_161861

theorem square_side_length (x : ℝ) : 
  x > 0 ∧ 
  x + 17 > 0 ∧ 
  x + 11 > 0 ∧ 
  x + (x + 17) + (x + 11) = 52 → 
  x = 8 := by
sorry

end square_side_length_l1618_161861


namespace stack_of_books_average_pages_l1618_161886

/-- Calculates the average number of pages per book given the total thickness,
    pages per inch, and number of books. -/
def averagePagesPerBook (totalThickness : ℕ) (pagesPerInch : ℕ) (numBooks : ℕ) : ℕ :=
  (totalThickness * pagesPerInch) / numBooks

/-- Theorem stating that for a stack of books 12 inches thick,
    with 80 pages per inch and 6 books in total,
    the average number of pages per book is 160. -/
theorem stack_of_books_average_pages :
  averagePagesPerBook 12 80 6 = 160 := by
  sorry

end stack_of_books_average_pages_l1618_161886


namespace M_intersect_N_eq_M_l1618_161855

-- Define set M
def M : Set ℝ := {y | ∃ x : ℝ, y = x^2 + 1}

-- Define set N
def N : Set ℝ := {x | ∃ a : ℝ, x = 2*a^2 - 4*a + 1}

-- Theorem statement
theorem M_intersect_N_eq_M : M ∩ N = M := by sorry

end M_intersect_N_eq_M_l1618_161855


namespace function_equality_implies_a_range_l1618_161800

-- Define the functions f and g
def f (a x : ℝ) : ℝ := |x + a| + |x + 3|
def g (x : ℝ) : ℝ := |x - 1| + 2

-- State the theorem
theorem function_equality_implies_a_range (a : ℝ) :
  (∀ x₁ : ℝ, ∃ x₂ : ℝ, f a x₁ = g x₂) →
  a ≥ 5 ∨ a ≤ 1 :=
by sorry

end function_equality_implies_a_range_l1618_161800


namespace solution_set_of_inequality_l1618_161802

theorem solution_set_of_inequality (x : ℝ) :
  (|2 * x^2 - 1| ≤ 1) ↔ (-1 ≤ x ∧ x ≤ 1) := by
  sorry

end solution_set_of_inequality_l1618_161802


namespace susan_book_count_l1618_161811

theorem susan_book_count (susan_books : ℕ) (lidia_books : ℕ) : 
  lidia_books = 4 * susan_books → 
  susan_books + lidia_books = 3000 → 
  susan_books = 600 := by
sorry

end susan_book_count_l1618_161811


namespace price_reduction_achieves_target_profit_l1618_161815

/-- Represents the price reduction in yuan -/
def price_reduction : ℝ := 20

/-- Average daily sale before price reduction -/
def initial_sales : ℝ := 20

/-- Initial profit per piece in yuan -/
def initial_profit_per_piece : ℝ := 40

/-- Increase in sales per yuan of price reduction -/
def sales_increase_rate : ℝ := 2

/-- Target daily profit in yuan -/
def target_profit : ℝ := 1200

/-- Theorem stating that the given price reduction achieves the target profit -/
theorem price_reduction_achieves_target_profit :
  (initial_sales + sales_increase_rate * price_reduction) * 
  (initial_profit_per_piece - price_reduction) = target_profit :=
by sorry

end price_reduction_achieves_target_profit_l1618_161815


namespace converse_statement_is_false_l1618_161827

/-- Represents an ellipse with foci on the x-axis -/
structure Ellipse where
  a : ℝ
  eq : ∀ (x y : ℝ), x^2 / a^2 + y^2 / 4 = 1
  foci_on_x : True  -- We assume this property is satisfied

/-- The converse statement is false -/
theorem converse_statement_is_false : 
  ¬(∀ e : Ellipse, e.a = 4) :=
sorry

end converse_statement_is_false_l1618_161827


namespace rectangular_prism_problem_l1618_161893

theorem rectangular_prism_problem :
  ∀ x y : ℕ,
  x < 4 →
  y < 15 →
  15 * 5 * 4 - y * 5 * x = 120 →
  (x = 3 ∧ y = 12) :=
by sorry

end rectangular_prism_problem_l1618_161893


namespace intersection_of_A_and_B_l1618_161806

def A : Set ℝ := {x | |x| ≤ 2}
def B : Set ℝ := {x | x ≤ 1}

theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | -2 ≤ x ∧ x ≤ 1} := by sorry

end intersection_of_A_and_B_l1618_161806


namespace alcohol_mixture_problem_l1618_161853

/-- Proves that given a mixture of x litres with 20% alcohol, when 5 litres of water are added 
    resulting in a new mixture with 15% alcohol, the value of x is 15 litres. -/
theorem alcohol_mixture_problem (x : ℝ) 
  (h1 : x > 0)  -- Ensure x is positive
  (h2 : 0.20 * x = 0.15 * (x + 5)) : x = 15 := by
  sorry

end alcohol_mixture_problem_l1618_161853


namespace qq_fish_tank_theorem_l1618_161816

/-- Represents the fish tank scenario --/
structure FishTank where
  total_fish : Nat
  blue_fish : Nat
  black_fish : Nat
  daily_catch : Nat

/-- Probability that Mr. QQ eats at least a certain number of fish --/
def prob_eat_at_least (tank : FishTank) (n : Nat) : ℚ :=
  sorry

/-- Expected number of fish eaten by Mr. QQ --/
def expected_fish_eaten (tank : FishTank) : ℚ :=
  sorry

/-- The main theorem about Mr. QQ's fish tank --/
theorem qq_fish_tank_theorem (tank : FishTank) 
    (h1 : tank.total_fish = 7)
    (h2 : tank.blue_fish = 6)
    (h3 : tank.black_fish = 1)
    (h4 : tank.daily_catch = 1) :
  (prob_eat_at_least tank 5 = 19/35) ∧ 
  (expected_fish_eaten tank = 5) :=
by sorry

end qq_fish_tank_theorem_l1618_161816


namespace arithmetic_calculations_l1618_161824

theorem arithmetic_calculations :
  (1 - 3 - (-4) = 1) ∧
  (-1/3 + (-4/3) = -5/3) ∧
  ((-2) * (-3) * (-5) = -30) ∧
  (15 / 4 * (-1/4) = -15/16) := by
  sorry

end arithmetic_calculations_l1618_161824


namespace characterization_of_special_numbers_l1618_161883

/-- A function that checks if a real number has only two distinct non-zero digits, one of which is 3 -/
def hasTwoDistinctNonZeroDigitsWithThree (N : ℝ) : Prop := sorry

/-- A function that checks if a real number is a perfect square -/
def isPerfectSquare (N : ℝ) : Prop := sorry

/-- Theorem stating the characterization of numbers satisfying the given conditions -/
theorem characterization_of_special_numbers (N : ℝ) : 
  (hasTwoDistinctNonZeroDigitsWithThree N ∧ isPerfectSquare N) ↔ 
  ∃ n : ℕ, N = 36 * (100 : ℝ) ^ n :=
sorry

end characterization_of_special_numbers_l1618_161883


namespace apartment_cost_comparison_l1618_161825

/-- Proves that the average cost per mile driven is $0.58 given the conditions of the apartment comparison problem. -/
theorem apartment_cost_comparison (rent1 rent2 utilities1 utilities2 : ℕ)
  (miles_per_day1 miles_per_day2 work_days_per_month : ℕ)
  (total_cost_difference : ℚ) :
  rent1 = 800 →
  rent2 = 900 →
  utilities1 = 260 →
  utilities2 = 200 →
  miles_per_day1 = 31 →
  miles_per_day2 = 21 →
  work_days_per_month = 20 →
  total_cost_difference = 76 →
  let total_miles1 := miles_per_day1 * work_days_per_month
  let total_miles2 := miles_per_day2 * work_days_per_month
  let cost_per_mile := (rent1 + utilities1 - rent2 - utilities2 + total_cost_difference) / (total_miles1 - total_miles2)
  cost_per_mile = 29/50 := by
  sorry

end apartment_cost_comparison_l1618_161825


namespace max_value_expression_max_value_achievable_l1618_161859

theorem max_value_expression (x y : ℝ) :
  (3 * x + 4 * y + 5) / Real.sqrt (5 * x^2 + 3 * y^2 + 4) ≤ 5 * Real.sqrt 2 :=
sorry

theorem max_value_achievable :
  ∃ (x y : ℝ), (3 * x + 4 * y + 5) / Real.sqrt (5 * x^2 + 3 * y^2 + 4) = 5 * Real.sqrt 2 :=
sorry

end max_value_expression_max_value_achievable_l1618_161859


namespace total_luggage_calculation_l1618_161856

def passengers : ℕ := 4
def luggage_per_passenger : ℕ := 8

theorem total_luggage_calculation : passengers * luggage_per_passenger = 32 := by
  sorry

end total_luggage_calculation_l1618_161856


namespace arithmetic_sequence_difference_l1618_161879

/-- An arithmetic sequence with its properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  sum : ℕ → ℝ -- Sum function
  is_arithmetic : ∀ n, a (n + 1) = a n + d
  sum_formula : ∀ n, sum n = n * (2 * a 1 + (n - 1) * d) / 2

/-- Theorem: Given an arithmetic sequence with S_3 = 6 and a_4 = 8, the common difference is 3 -/
theorem arithmetic_sequence_difference (seq : ArithmeticSequence) 
  (h1 : seq.sum 3 = 6) (h2 : seq.a 4 = 8) : seq.d = 3 := by
  sorry

#check arithmetic_sequence_difference

end arithmetic_sequence_difference_l1618_161879


namespace scientific_notation_of_small_number_l1618_161843

theorem scientific_notation_of_small_number :
  ∃ (a : ℝ) (n : ℤ), 0.000815 = a * 10^n ∧ 1 ≤ a ∧ a < 10 ∧ n = -4 := by
  sorry

end scientific_notation_of_small_number_l1618_161843


namespace unique_positive_solution_l1618_161839

theorem unique_positive_solution (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x * y = z) (h2 : y * z = x) (h3 : z * x = y) :
  x = 1 ∧ y = 1 ∧ z = 1 := by
sorry

end unique_positive_solution_l1618_161839


namespace intersection_of_sets_l1618_161858

theorem intersection_of_sets : 
  let A : Set ℕ := {1, 2, 3}
  let B : Set ℕ := {1, 2, 5}
  A ∩ B = {1, 2} := by
sorry

end intersection_of_sets_l1618_161858


namespace complement_of_A_union_B_l1618_161835

def U : Set Nat := {1, 2, 3, 4, 5}

def A : Set Nat := {x ∈ U | x^2 - 3*x + 2 = 0}

def B : Set Nat := {x ∈ U | ∃ a ∈ A, x = 2*a}

theorem complement_of_A_union_B (x : Nat) : 
  x ∈ (U \ (A ∪ B)) ↔ x = 3 ∨ x = 5 := by
  sorry

end complement_of_A_union_B_l1618_161835


namespace min_colors_theorem_l1618_161838

/-- The minimum number of colors needed for a regular n-gon -/
def min_colors (n : ℕ) : ℕ :=
  if n % 2 = 1 then n else n - 1

/-- Theorem stating the minimum number of colors needed for a regular n-gon -/
theorem min_colors_theorem (n : ℕ) (h : n ≥ 3) :
  ∃ (m : ℕ), m = min_colors n ∧
  (∀ (k : ℕ), k < m → ¬(∀ (coloring : Fin n → Fin n → Fin k),
    ∀ (v : Fin n), ∀ (i j : Fin n), i ≠ j → coloring v i ≠ coloring v j)) ∧
  (∃ (coloring : Fin n → Fin n → Fin m),
    ∀ (v : Fin n), ∀ (i j : Fin n), i ≠ j → coloring v i ≠ coloring v j) :=
by sorry

#check min_colors_theorem

end min_colors_theorem_l1618_161838


namespace total_coughs_equation_georgia_coughs_five_times_l1618_161865

/-- Georgia's coughs per minute -/
def G : ℕ := sorry

/-- The total number of coughs after 20 minutes -/
def total_coughs : ℕ := 300

/-- Robert coughs twice as much as Georgia -/
def roberts_coughs_per_minute : ℕ := 2 * G

/-- The total coughs after 20 minutes equals 300 -/
theorem total_coughs_equation : 20 * (G + roberts_coughs_per_minute) = total_coughs := by sorry

/-- Georgia coughs 5 times per minute -/
theorem georgia_coughs_five_times : G = 5 := by sorry

end total_coughs_equation_georgia_coughs_five_times_l1618_161865


namespace marble_fraction_after_tripling_l1618_161807

theorem marble_fraction_after_tripling (total : ℚ) (h1 : total > 0) : 
  let initial_blue := (4 : ℚ) / 7 * total
  let initial_red := total - initial_blue
  let final_red := 3 * initial_red
  let final_total := initial_blue + final_red
  final_red / final_total = (9 : ℚ) / 13 :=
by sorry

end marble_fraction_after_tripling_l1618_161807


namespace smallest_fraction_between_l1618_161873

theorem smallest_fraction_between (p q : ℕ+) : 
  (5 : ℚ) / 9 < (p : ℚ) / q ∧ 
  (p : ℚ) / q < (4 : ℚ) / 7 ∧ 
  (∀ (p' q' : ℕ+), (5 : ℚ) / 9 < (p' : ℚ) / q' ∧ (p' : ℚ) / q' < (4 : ℚ) / 7 → q ≤ q') →
  q - p = 7 := by
sorry

end smallest_fraction_between_l1618_161873


namespace right_triangle_arithmetic_sequence_l1618_161876

theorem right_triangle_arithmetic_sequence (b k : ℝ) (h_k_pos : k > 0) : 
  (b - k) > 0 ∧ (b + k)^2 = (b - k)^2 + b^2 → b = 2 * k := by
  sorry

end right_triangle_arithmetic_sequence_l1618_161876


namespace forty_percent_of_jacqueline_candy_bars_l1618_161817

def fred_candy_bars : ℕ := 12
def uncle_bob_extra_candy_bars : ℕ := 6
def jacqueline_multiplier : ℕ := 10

def uncle_bob_candy_bars : ℕ := fred_candy_bars + uncle_bob_extra_candy_bars
def total_fred_and_bob : ℕ := fred_candy_bars + uncle_bob_candy_bars
def jacqueline_candy_bars : ℕ := jacqueline_multiplier * total_fred_and_bob

theorem forty_percent_of_jacqueline_candy_bars : 
  (40 : ℕ) * jacqueline_candy_bars / 100 = 120 := by
  sorry

end forty_percent_of_jacqueline_candy_bars_l1618_161817


namespace malcolm_route_fraction_l1618_161887

/-- Represents the fraction of time spent on the last stage of the first route -/
def last_stage_fraction (uphill_time flat_time : ℕ) (route_difference : ℕ) : ℚ :=
  let first_two_stages := uphill_time + 2 * uphill_time
  let second_route_time := flat_time + 2 * flat_time
  (second_route_time - first_two_stages - route_difference) / first_two_stages

theorem malcolm_route_fraction :
  last_stage_fraction 6 14 18 = 1/3 := by
  sorry

end malcolm_route_fraction_l1618_161887


namespace time_per_furniture_piece_l1618_161863

theorem time_per_furniture_piece (chairs tables total_time : ℕ) 
  (h1 : chairs = 7)
  (h2 : tables = 3)
  (h3 : total_time = 40) : 
  total_time / (chairs + tables) = 4 := by
  sorry

end time_per_furniture_piece_l1618_161863


namespace probability_three_out_of_five_dice_less_than_six_l1618_161830

/-- The probability of exactly three out of five fair 10-sided dice showing a number less than 6 -/
theorem probability_three_out_of_five_dice_less_than_six :
  let n : ℕ := 5  -- number of dice
  let k : ℕ := 3  -- number of successes (dice showing less than 6)
  let p : ℚ := 1/2  -- probability of a single die showing less than 6
  Nat.choose n k * p^k * (1-p)^(n-k) = 5/16 := by sorry

end probability_three_out_of_five_dice_less_than_six_l1618_161830


namespace triangle_inequality_constant_l1618_161852

theorem triangle_inequality_constant (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) :
  (a^2 + b^2 + c^2) / (a*b + b*c + c*a) < 2 ∧
  ∀ N : ℝ, (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 →
    (a^2 + b^2 + c^2) / (a*b + b*c + c*a) < N) →
  2 ≤ N :=
sorry

end triangle_inequality_constant_l1618_161852


namespace four_heads_before_three_tails_l1618_161810

/-- The probability of getting heads in a fair coin flip -/
def p_heads : ℚ := 1/2

/-- The probability of getting tails in a fair coin flip -/
def p_tails : ℚ := 1/2

/-- The probability of encountering 4 heads before 3 tails in repeated fair coin flips -/
noncomputable def q : ℚ := sorry

theorem four_heads_before_three_tails : q = 15/23 := by sorry

end four_heads_before_three_tails_l1618_161810


namespace area_of_shaded_region_l1618_161854

/-- A line defined by two points -/
structure Line where
  x1 : ℝ
  y1 : ℝ
  x2 : ℝ
  y2 : ℝ

/-- The area bounded by two lines -/
def area_between_lines (l1 l2 : Line) : ℝ := sorry

/-- The first line passing through (0,3) and (10,2) -/
def line1 : Line := { x1 := 0, y1 := 3, x2 := 10, y2 := 2 }

/-- The second line passing through (0,5) and (5,0) -/
def line2 : Line := { x1 := 0, y1 := 5, x2 := 5, y2 := 0 }

theorem area_of_shaded_region :
  area_between_lines line1 line2 = 5/4 := by sorry

end area_of_shaded_region_l1618_161854


namespace equation_solution_l1618_161850

theorem equation_solution : 
  ∃ x : ℝ, x + (x + 1) + (x + 2) + (x + 3) = 34 ∧ x = 7 := by
  sorry

end equation_solution_l1618_161850


namespace number_equation_solution_l1618_161881

theorem number_equation_solution : ∃ x : ℝ, x - 3 / (1/3) + 3 = 3 ∧ x = 9 := by
  sorry

end number_equation_solution_l1618_161881


namespace max_demand_decrease_l1618_161851

theorem max_demand_decrease (price_increase : ℝ) (revenue_increase : ℝ) : 
  price_increase = 0.20 →
  revenue_increase = 0.10 →
  (1 + price_increase) * (1 - (1 / 12 : ℝ)) ≥ 1 + revenue_increase :=
by sorry

end max_demand_decrease_l1618_161851


namespace b_value_l1618_161871

/-- A probability distribution for a random variable X -/
structure ProbDist where
  a : ℝ
  b : ℝ
  c : ℝ
  sum_to_one : a + b + c = 1
  b_is_mean : b = (a + c) / 2

/-- The value of b in the probability distribution is 1/3 -/
theorem b_value (p : ProbDist) : p.b = 1/3 := by
  sorry

end b_value_l1618_161871


namespace least_with_eight_factors_l1618_161822

/-- A function that returns the number of distinct positive factors of a positive integer -/
def number_of_factors (n : ℕ+) : ℕ := sorry

/-- A function that returns the set of all distinct positive factors of a positive integer -/
def factors (n : ℕ+) : Finset ℕ+ := sorry

/-- The theorem stating that 24 is the least positive integer with exactly eight distinct positive factors -/
theorem least_with_eight_factors :
  ∀ n : ℕ+, number_of_factors n = 8 → n ≥ 24 ∧ 
  (n = 24 → number_of_factors 24 = 8 ∧ factors 24 = {1, 2, 3, 4, 6, 8, 12, 24}) := by
  sorry

end least_with_eight_factors_l1618_161822


namespace sum_max_min_f_on_interval_l1618_161836

def f (x : ℝ) : ℝ := x^2 - 4*x + 1

theorem sum_max_min_f_on_interval : 
  ∃ (max min : ℝ), 
    (∀ x ∈ Set.Icc 0 5, f x ≤ max) ∧ 
    (∃ x ∈ Set.Icc 0 5, f x = max) ∧
    (∀ x ∈ Set.Icc 0 5, min ≤ f x) ∧ 
    (∃ x ∈ Set.Icc 0 5, f x = min) ∧
    max + min = 3 :=
by sorry

end sum_max_min_f_on_interval_l1618_161836


namespace balloon_theorem_l1618_161831

def balloon_problem (brooke_initial : ℕ) (brooke_added : ℕ) (tracy_initial : ℕ) (tracy_added : ℕ) : ℕ :=
  let brooke_total := brooke_initial + brooke_added
  let tracy_total := tracy_initial + tracy_added
  let tracy_after_popping := tracy_total - (tracy_total / 5 * 2)
  let brooke_after_giving := brooke_total - (brooke_total / 4)
  (tracy_after_popping - 5) + (brooke_after_giving - 5)

theorem balloon_theorem :
  balloon_problem 25 22 16 42 = 61 := by
  sorry

end balloon_theorem_l1618_161831


namespace sets_intersection_union_l1618_161803

-- Define the sets A, B, and C
def A : Set ℝ := {x | x^2 + x - 2 ≤ 0}
def B : Set ℝ := {x | 2 < x + 1 ∧ x + 1 ≤ 4}
def C (b c : ℝ) : Set ℝ := {x | x^2 + b*x + c > 0}

-- State the theorem
theorem sets_intersection_union (b c : ℝ) :
  (A ∪ B) ∩ C b c = ∅ ∧ (A ∪ B) ∪ C b c = Set.univ →
  b = -1 ∧ c = -6 := by
  sorry

end sets_intersection_union_l1618_161803


namespace find_x_l1618_161888

theorem find_x : ∃ x : ℝ, 0.65 * x = 0.20 * 682.50 ∧ x = 210 := by sorry

end find_x_l1618_161888


namespace parallel_planes_distance_equivalence_l1618_161869

-- Define the types for planes and points
variable (Plane Point : Type)

-- Define the distance function between planes
variable (distance : Plane → Plane → ℝ)

-- Define the length function between points
variable (length : Point → Point → ℝ)

-- Define the parallel relation between planes
variable (parallel : Plane → Plane → Prop)

-- Define the intersection of a line with a plane
variable (intersect : Plane → Point)

-- Given three parallel planes
variable (α₁ α₂ α₃ : Plane)
variable (h_parallel : parallel α₁ α₂ ∧ parallel α₂ α₃ ∧ parallel α₁ α₃)

-- Define the distances between planes
variable (d₁ d₂ : ℝ)
variable (h_d₁ : distance α₁ α₂ = d₁)
variable (h_d₂ : distance α₂ α₃ = d₂)

-- Define the intersection points
variable (P₁ P₂ P₃ : Point)
variable (h_P₁ : P₁ = intersect α₁)
variable (h_P₂ : P₂ = intersect α₂)
variable (h_P₃ : P₃ = intersect α₃)

-- State the theorem
theorem parallel_planes_distance_equivalence :
  (length P₁ P₂ = length P₂ P₃) ↔ (d₁ = d₂) :=
sorry

end parallel_planes_distance_equivalence_l1618_161869


namespace no_real_roots_l1618_161857

def f (x : ℝ) : ℝ := x^2 - 2*x + 3

theorem no_real_roots : ∀ x : ℝ, f x ≠ 0 := by
  sorry

end no_real_roots_l1618_161857


namespace unique_four_digit_number_l1618_161884

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

def hundreds_digit (n : ℕ) : ℕ := (n / 100) % 10

def units_digit (n : ℕ) : ℕ := n % 10

def first_two_digits (n : ℕ) : ℕ := n / 100

def first_last_digits (n : ℕ) : ℕ := (n / 1000) * 10 + (n % 10)

theorem unique_four_digit_number :
  ∃! n : ℕ, is_four_digit n ∧
            ¬(n % 7 = 0) ∧
            tens_digit n = hundreds_digit n + units_digit n ∧
            first_two_digits n = 15 * units_digit n ∧
            Nat.Prime (first_last_digits n) := by
  sorry

end unique_four_digit_number_l1618_161884


namespace intersection_problem_complement_problem_l1618_161867

def A : Set ℝ := {x | x^2 - 3*x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + 2*(a+1)*x + (a^2-5) = 0}
def U : Set ℝ := Set.univ

theorem intersection_problem (a : ℝ) :
  (A ∩ B a = {2}) → (a = -1 ∨ a = -3) :=
by sorry

theorem complement_problem (a : ℝ) :
  (A ∩ (U \ B a) = A) → 
  (a < -3 ∨ (-3 < a ∧ a < -1-Real.sqrt 3) ∨ 
   (-1-Real.sqrt 3 < a ∧ a < -1) ∨ 
   (-1 < a ∧ a < -1+Real.sqrt 3) ∨ 
   a > -1+Real.sqrt 3) :=
by sorry

end intersection_problem_complement_problem_l1618_161867


namespace ya_interval_neg_sqrt_seven_value_of_c_l1618_161828

-- Definition of Ya interval
def ya_interval (T : ℝ) : Set ℝ :=
  {x | ∃ m n : ℤ, m < T ∧ T < n ∧ x ∈ Set.Ioo (↑m : ℝ) (↑n : ℝ) ∧
    ∀ k : ℤ, (k : ℝ) ≤ T → k ≤ m}

-- Theorem 1: Ya interval of -√7
theorem ya_interval_neg_sqrt_seven :
  ya_interval (-Real.sqrt 7) = Set.Ioo (-3 : ℝ) (-2 : ℝ) := by sorry

-- Theorem 2: Value of c in the equation
theorem value_of_c (m n : ℕ) (h1 : ya_interval (Real.sqrt n - m) = Set.Ioo (↑m : ℝ) (↑n : ℝ))
  (h2 : 0 < m + Real.sqrt n) (h3 : m + Real.sqrt n < 12)
  (h4 : ∃ (x y : ℕ), x = m ∧ y^2 = n ∧ m*x - n*y = c) :
  c = 1 ∨ c = 37 := by sorry

end ya_interval_neg_sqrt_seven_value_of_c_l1618_161828


namespace matrix_operation_example_l1618_161805

def matrix_operation (a b c d : ℚ) : ℚ := a * d - b * c

theorem matrix_operation_example : matrix_operation 1 2 3 4 = -2 := by
  sorry

end matrix_operation_example_l1618_161805


namespace seven_people_circular_permutations_l1618_161834

/-- The number of distinct seating arrangements for n people around a round table,
    where rotations are considered the same. -/
def circularPermutations (n : ℕ) : ℕ := (n - 1).factorial

/-- Theorem: The number of distinct seating arrangements for 7 people around a round table,
    where rotations are considered the same, is equal to 720. -/
theorem seven_people_circular_permutations :
  circularPermutations 7 = 720 := by
  sorry

end seven_people_circular_permutations_l1618_161834


namespace tangent_and_fixed_line_l1618_161829

-- Define the curves C₁ and C₂
def C₁ (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 16
def C₂ (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define points M and N
def M : ℝ × ℝ := (-1, 0)
def N : ℝ × ℝ := (1, 0)

-- Define a line through M with non-zero slope
def line_through_M (k : ℝ) (x y : ℝ) : Prop := y = k * (x + 1)

-- Define the intersection points A and B
def intersect_C₁_line (k : ℝ) : Set (ℝ × ℝ) :=
  {p | C₁ p.1 p.2 ∧ line_through_M k p.1 p.2}

-- Define perpendicular bisector
def perp_bisector (p₁ p₂ : ℝ × ℝ) (x y : ℝ) : Prop :=
  (x - (p₁.1 + p₂.1) / 2) * (p₂.1 - p₁.1) + (y - (p₁.2 + p₂.2) / 2) * (p₂.2 - p₁.2) = 0

theorem tangent_and_fixed_line 
  (k : ℝ) 
  (hk : k ≠ 0) 
  (A B : ℝ × ℝ) 
  (hA : A ∈ intersect_C₁_line k) 
  (hB : B ∈ intersect_C₁_line k) 
  (hAB : A ≠ B) :
  (∃ (P : ℝ × ℝ), 
    (∀ (x y : ℝ), perp_bisector A N x y → C₂ x y) ∧ 
    (∀ (x y : ℝ), perp_bisector B N x y → C₂ x y) ∧
    perp_bisector A N P.1 P.2 ∧ 
    perp_bisector B N P.1 P.2 ∧
    P.1 = -4) :=
sorry

end tangent_and_fixed_line_l1618_161829


namespace special_triangle_side_length_l1618_161882

/-- An equilateral triangle with a point inside satisfying certain distances -/
structure SpecialTriangle where
  /-- Side length of the equilateral triangle -/
  t : ℝ
  /-- Point A of the triangle -/
  A : ℝ × ℝ
  /-- Point B of the triangle -/
  B : ℝ × ℝ
  /-- Point C of the triangle -/
  C : ℝ × ℝ
  /-- Point Q inside the triangle -/
  Q : ℝ × ℝ
  /-- Triangle ABC is equilateral with side length t -/
  equilateral : ‖A - B‖ = t ∧ ‖B - C‖ = t ∧ ‖C - A‖ = t
  /-- Distance AQ is 2 -/
  AQ_dist : ‖A - Q‖ = 2
  /-- Distance BQ is 2√2 -/
  BQ_dist : ‖B - Q‖ = 2 * Real.sqrt 2
  /-- Distance CQ is 3 -/
  CQ_dist : ‖C - Q‖ = 3

/-- Theorem stating that the side length of the special triangle is √15 -/
theorem special_triangle_side_length (tri : SpecialTriangle) : tri.t = Real.sqrt 15 := by
  sorry

end special_triangle_side_length_l1618_161882


namespace max_a_minus_b_is_seven_l1618_161841

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - x^2 - x + 2

-- Define the theorem
theorem max_a_minus_b_is_seven :
  ∃ (a b : ℝ),
    (∀ x ∈ Set.Icc (-1 : ℝ) 2, -3 ≤ a * f x + b ∧ a * f x + b ≤ 3) ∧
    (a - b = 7) ∧
    (∀ a' b' : ℝ, (∀ x ∈ Set.Icc (-1 : ℝ) 2, -3 ≤ a' * f x + b' ∧ a' * f x + b' ≤ 3) → a' - b' ≤ 7) :=
by sorry

end max_a_minus_b_is_seven_l1618_161841


namespace marcy_spear_count_l1618_161895

/-- Represents the number of spears that can be made from different resources --/
structure SpearYield where
  sapling : ℕ
  log : ℕ
  branches : ℕ
  trunk : ℕ

/-- Represents the exchange rates between resources --/
structure ExchangeRates where
  saplings_to_logs : ℕ × ℕ
  branches_to_trunk : ℕ × ℕ

/-- Represents the initial resources Marcy has --/
structure InitialResources where
  saplings : ℕ
  logs : ℕ
  branches : ℕ

/-- Calculates the maximum number of spears Marcy can make --/
def max_spears (yield : SpearYield) (rates : ExchangeRates) (initial : InitialResources) : ℕ :=
  sorry

/-- Theorem stating that Marcy can make 81 spears given the problem conditions --/
theorem marcy_spear_count :
  let yield : SpearYield := { sapling := 3, log := 9, branches := 7, trunk := 15 }
  let rates : ExchangeRates := { saplings_to_logs := (5, 2), branches_to_trunk := (3, 1) }
  let initial : InitialResources := { saplings := 12, logs := 1, branches := 6 }
  max_spears yield rates initial = 81 := by
  sorry

end marcy_spear_count_l1618_161895


namespace solution_sum_l1618_161814

theorem solution_sum (a b : ℚ) : 
  (2 * a + b = 14 ∧ a + 2 * b = 21) → a + b = 35 / 3 := by
  sorry

end solution_sum_l1618_161814


namespace line_through_point_l1618_161846

theorem line_through_point (k : ℚ) : 
  (2 * k * 3 - 5 = -4 * (-4)) → k = 7/2 := by
  sorry

end line_through_point_l1618_161846


namespace product_sum_inequality_l1618_161842

theorem product_sum_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_prod : a * b * c = 1) :
  (a + b) * (b + c) * (c + a) ≥ 4 * (a + b + c - 1) := by
  sorry

end product_sum_inequality_l1618_161842


namespace wall_width_proof_elijah_wall_width_l1618_161840

theorem wall_width_proof (total_walls : ℕ) (known_wall_width : ℝ) (known_wall_count : ℕ) 
  (total_tape_needed : ℝ) : ℝ :=
  let remaining_walls := total_walls - known_wall_count
  let known_walls_tape := known_wall_width * known_wall_count
  let remaining_tape := total_tape_needed - known_walls_tape
  remaining_tape / remaining_walls

theorem elijah_wall_width : wall_width_proof 4 4 2 20 = 6 := by
  sorry

end wall_width_proof_elijah_wall_width_l1618_161840


namespace marbles_found_vs_lost_l1618_161898

theorem marbles_found_vs_lost (initial : ℕ) (lost : ℕ) (found : ℕ) :
  initial = 7 → lost = 8 → found = 10 → found - lost = 2 := by
  sorry

end marbles_found_vs_lost_l1618_161898


namespace smokers_percentage_is_five_percent_l1618_161847

/-- Represents the survey setup and results -/
structure SurveyData where
  total_students : ℕ
  white_balls : ℕ
  red_balls : ℕ
  stones_in_box : ℕ

/-- Calculates the estimated percentage of smokers based on the survey data -/
def estimate_smokers_percentage (data : SurveyData) : ℚ :=
  let total_balls := data.white_balls + data.red_balls
  let prob_question1 := data.white_balls / total_balls
  let expected_yes_question1 := data.total_students * prob_question1 * (1 / 2)
  let smokers := data.stones_in_box - expected_yes_question1
  let students_answering_question2 := data.total_students * (data.red_balls / total_balls)
  (smokers / students_answering_question2) * 100

/-- The main theorem stating that given the survey conditions, 
    the estimated percentage of smokers is 5% -/
theorem smokers_percentage_is_five_percent 
  (data : SurveyData) 
  (h1 : data.total_students = 200)
  (h2 : data.white_balls = 5)
  (h3 : data.red_balls = 5)
  (h4 : data.stones_in_box = 55) :
  estimate_smokers_percentage data = 5 := by
  sorry


end smokers_percentage_is_five_percent_l1618_161847


namespace total_catch_l1618_161844

def johnny_catch : ℕ := 8

def sony_catch (johnny : ℕ) : ℕ := 4 * johnny

theorem total_catch (johnny : ℕ) (sony : ℕ → ℕ) 
  (h1 : johnny = johnny_catch) 
  (h2 : sony = sony_catch) : 
  sony johnny + johnny = 40 := by
  sorry

end total_catch_l1618_161844


namespace chessboard_problem_l1618_161818

/-- Distance function on the infinite chessboard -/
def distance (p q : ℤ × ℤ) : ℕ :=
  (Int.natAbs (p.1 - q.1)).max (Int.natAbs (p.2 - q.2))

/-- The problem statement -/
theorem chessboard_problem (A B C : ℤ × ℤ) 
  (hAB : distance A B = 100)
  (hBC : distance B C = 100)
  (hAC : distance A C = 100) :
  ∃! X : ℤ × ℤ, distance X A = 50 ∧ distance X B = 50 ∧ distance X C = 50 :=
sorry

end chessboard_problem_l1618_161818


namespace line_x_intercept_l1618_161885

/-- Given a straight line passing through points (2, -2) and (-3, 7), 
    its x-intercept is 8/9 -/
theorem line_x_intercept : 
  ∀ (f : ℝ → ℝ), 
  (∃ m b : ℝ, ∀ x, f x = m * x + b) → -- f is a linear function
  f 2 = -2 →                         -- f passes through (2, -2)
  f (-3) = 7 →                       -- f passes through (-3, 7)
  ∃ x : ℝ, x = 8/9 ∧ f x = 0 :=      -- x-intercept is 8/9
by
  sorry


end line_x_intercept_l1618_161885


namespace inequality_solution_and_parameters_l1618_161890

theorem inequality_solution_and_parameters :
  ∀ (a b : ℝ),
  (∀ x : ℝ, x^2 - 3*a*x + b > 0 ↔ x < 1 ∨ x > 2) →
  (a = 1 ∧ b = 2) ∧
  (∀ m : ℝ, 
    (m < 2 → ∀ x : ℝ, (x - 2)*(x - m) < 0 ↔ m < x ∧ x < 2) ∧
    (m = 2 → ∀ x : ℝ, ¬((x - 2)*(x - m) < 0)) ∧
    (m > 2 → ∀ x : ℝ, (x - 2)*(x - m) < 0 ↔ 2 < x ∧ x < m)) :=
by sorry

end inequality_solution_and_parameters_l1618_161890


namespace subcommittees_count_l1618_161837

def total_members : ℕ := 12
def teacher_count : ℕ := 5
def subcommittee_size : ℕ := 4

def subcommittees_with_teacher : ℕ := Nat.choose total_members subcommittee_size - Nat.choose (total_members - teacher_count) subcommittee_size

theorem subcommittees_count : subcommittees_with_teacher = 460 := by
  sorry

end subcommittees_count_l1618_161837


namespace inequality_proof_l1618_161878

theorem inequality_proof (x y : ℝ) (hx : x ≠ -1) (hy : y ≠ -1) (hxy : x * y = 1) :
  ((2 + x) / (1 + x))^2 + ((2 + y) / (1 + y))^2 ≥ 9/2 := by
  sorry

end inequality_proof_l1618_161878


namespace sin_C_value_side_lengths_l1618_161801

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.c = 13 ∧ Real.cos t.A = 5/13

-- Theorem 1
theorem sin_C_value (t : Triangle) (h : triangle_conditions t) (ha : t.a = 36) :
  Real.sin t.C = 1/3 := by
  sorry

-- Theorem 2
theorem side_lengths (t : Triangle) (h : triangle_conditions t) (harea : (1/2) * t.b * t.c * Real.sin t.A = 6) :
  t.a = 4 * Real.sqrt 10 ∧ t.b = 1 := by
  sorry

end sin_C_value_side_lengths_l1618_161801


namespace mean_of_special_set_l1618_161872

def is_valid_set (S : Finset ℝ) : Prop :=
  let n := S.card
  let s := S.sum id
  (s + 1) / (n + 1) = s / n - 13 ∧
  (s + 2001) / (n + 1) = s / n + 27

theorem mean_of_special_set (S : Finset ℝ) (h : is_valid_set S) :
  S.sum id / S.card = 651 := by
  sorry

end mean_of_special_set_l1618_161872


namespace angle_trisection_l1618_161877

/-- Given an angle of 54°, prove that its trisection results in three equal angles of 18° each. -/
theorem angle_trisection (θ : Real) (h : θ = 54) : 
  ∃ (α β γ : Real), α = β ∧ β = γ ∧ α + β + γ = θ ∧ α = 18 := by
  sorry

end angle_trisection_l1618_161877


namespace circle_O_diameter_l1618_161819

/-- The circle O with equation x^2 + y^2 - 2x + my - 4 = 0 -/
def circle_O (m : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x + m*y - 4 = 0

/-- The line with equation 2x + y = 0 -/
def symmetry_line (x y : ℝ) : Prop :=
  2*x + y = 0

/-- Two points are symmetric about a line if the line is the perpendicular bisector of the segment connecting the points -/
def symmetric_points (M N : ℝ × ℝ) (line : ℝ → ℝ → Prop) : Prop :=
  ∃ (midpoint : ℝ × ℝ), 
    line midpoint.1 midpoint.2 ∧ 
    (midpoint.1 = (M.1 + N.1) / 2) ∧ 
    (midpoint.2 = (M.2 + N.2) / 2) ∧
    ((N.1 - M.1) * 2 + (N.2 - M.2) = 0)

theorem circle_O_diameter : 
  ∃ (m : ℝ) (M N : ℝ × ℝ),
    circle_O m M.1 M.2 ∧ 
    circle_O m N.1 N.2 ∧ 
    symmetric_points M N symmetry_line →
    ∃ (center : ℝ × ℝ) (radius : ℝ),
      center = (1, -m/2) ∧ 
      radius = 3 ∧ 
      2 * radius = 6 :=
sorry

end circle_O_diameter_l1618_161819


namespace four_numbers_with_avg_six_l1618_161897

theorem four_numbers_with_avg_six (a b c d : ℕ+) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  (a + b + c + d : ℚ) / 4 = 6 →
  ∀ w x y z : ℕ+, w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z →
    (w + x + y + z : ℚ) / 4 = 6 →
    max a (max b (max c d)) - min a (min b (min c d)) ≥ max w (max x (max y z)) - min w (min x (min y z)) →
  (((max a (max b (max c d)) + min a (min b (min c d))) - (a + b + c + d)) / 2 : ℚ) = 7/2 :=
by sorry

end four_numbers_with_avg_six_l1618_161897


namespace max_min_sum_l1618_161849

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := a * x^3 - b * x + 2

-- Define the maximum and minimum values
def M (a b : ℝ) : ℝ := sorry
def m (a b : ℝ) : ℝ := sorry

-- Theorem statement
theorem max_min_sum (a b : ℝ) (h : a ≠ 0) : M a b + m a b = 4 := by sorry

end max_min_sum_l1618_161849


namespace defective_part_probability_l1618_161875

theorem defective_part_probability (p : ℝ) : 
  (0 ≤ p) ∧ (p ≤ 1) →
  (1 - 0.01) * (1 - p) = 0.9603 →
  p = 0.03 := by
sorry

end defective_part_probability_l1618_161875


namespace correct_sampling_methods_l1618_161860

/-- Represents different sampling methods -/
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified

/-- Represents a survey with its characteristics -/
structure Survey where
  total_population : ℕ
  sample_size : ℕ
  has_subgroups : Bool
  subgroup_sizes : List ℕ

/-- Determines the most appropriate sampling method for a given survey -/
def best_sampling_method (s : Survey) : SamplingMethod :=
  if s.has_subgroups then
    SamplingMethod.Stratified
  else if s.total_population % s.sample_size = 0 then
    SamplingMethod.Systematic
  else
    SamplingMethod.SimpleRandom

/-- The three surveys from the problem -/
def survey1 : Survey := ⟨10, 3, false, []⟩
def survey2 : Survey := ⟨3000, 30, false, []⟩
def survey3 : Survey := ⟨160, 40, true, [120, 16, 24]⟩

/-- Theorem stating the correct sampling methods for the given surveys -/
theorem correct_sampling_methods :
  best_sampling_method survey1 = SamplingMethod.SimpleRandom ∧
  best_sampling_method survey2 = SamplingMethod.Systematic ∧
  best_sampling_method survey3 = SamplingMethod.Stratified :=
  sorry

end correct_sampling_methods_l1618_161860


namespace fluffy_arrangements_eq_72_l1618_161812

/-- The number of distinct four-letter arrangements from the letters in "FLUFFY" -/
def fluffy_arrangements : ℕ :=
  let f_count := 3  -- Number of F's in FLUFFY
  let other_letters := 3  -- Number of other distinct letters (L, U, Y)
  let arrangement_size := 4  -- Size of each arrangement

  -- Case 1: Using 1 F
  let case1 := (arrangement_size.factorial) *
               (Nat.choose other_letters (arrangement_size - 1))

  -- Case 2: Using 2 F's
  let case2 := (arrangement_size.factorial / 2) *
               (Nat.choose other_letters (arrangement_size - 2))

  -- Case 3: Using 3 F's
  let case3 := (arrangement_size.factorial / 6) *
               (Nat.choose other_letters (arrangement_size - 3))

  -- Sum of all cases
  case1 + case2 + case3

/-- The number of distinct four-letter arrangements from the letters in "FLUFFY" is 72 -/
theorem fluffy_arrangements_eq_72 : fluffy_arrangements = 72 := by
  sorry

end fluffy_arrangements_eq_72_l1618_161812


namespace multiple_is_two_l1618_161809

-- Define the depths of the pools
def johns_pool_depth : ℝ := 15
def sarahs_pool_depth : ℝ := 5

-- Define the relationship between the pool depths
def depth_relation (x : ℝ) : Prop :=
  johns_pool_depth = x * sarahs_pool_depth + 5

-- Theorem statement
theorem multiple_is_two :
  ∃ x : ℝ, depth_relation x ∧ x = 2 := by sorry

end multiple_is_two_l1618_161809


namespace triangle_vector_sum_zero_l1618_161862

-- Define a triangle as a structure with three points
structure Triangle (V : Type*) [AddCommGroup V] :=
  (A B C : V)

-- Theorem statement
theorem triangle_vector_sum_zero {V : Type*} [AddCommGroup V] (t : Triangle V) :
  t.B - t.A + t.C - t.B + t.A - t.C = (0 : V) := by sorry

end triangle_vector_sum_zero_l1618_161862


namespace joe_total_cars_l1618_161870

def initial_cars : ℕ := 50
def additional_cars : ℕ := 12

theorem joe_total_cars : initial_cars + additional_cars = 62 := by
  sorry

end joe_total_cars_l1618_161870
