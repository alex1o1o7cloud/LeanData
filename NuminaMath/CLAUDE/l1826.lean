import Mathlib

namespace grocery_solution_l1826_182643

/-- Represents the grocery shopping problem --/
def grocery_problem (initial_money : ℝ) (mustard_oil_price : ℝ) (mustard_oil_quantity : ℝ)
  (pasta_price : ℝ) (pasta_quantity : ℝ) (sauce_price : ℝ) (money_left : ℝ) : Prop :=
  let total_spent := initial_money - money_left
  let mustard_oil_cost := mustard_oil_price * mustard_oil_quantity
  let pasta_cost := pasta_price * pasta_quantity
  let sauce_cost := total_spent - mustard_oil_cost - pasta_cost
  sauce_cost / sauce_price = 1

/-- Theorem stating the solution to the grocery problem --/
theorem grocery_solution :
  grocery_problem 50 13 2 4 3 5 7 := by
  sorry

#check grocery_solution

end grocery_solution_l1826_182643


namespace player1_can_win_l1826_182630

/-- Represents a square on the game board -/
structure Square where
  x : Fin 2021
  y : Fin 2021

/-- Represents a domino placement on the game board -/
structure Domino where
  square1 : Square
  square2 : Square

/-- The game state -/
structure GameState where
  board : Fin 2021 → Fin 2021 → Bool
  dominoes : List Domino

/-- A player's strategy -/
def Strategy := GameState → Domino

/-- The game play function -/
def play (player1Strategy player2Strategy : Strategy) : GameState :=
  sorry

theorem player1_can_win :
  ∃ (player1Strategy : Strategy),
    ∀ (player2Strategy : Strategy),
      let finalState := play player1Strategy player2Strategy
      ∃ (s1 s2 : Square), s1 ≠ s2 ∧ finalState.board s1.x s1.y = false ∧ finalState.board s2.x s2.y = false :=
  sorry


end player1_can_win_l1826_182630


namespace petya_bonus_points_l1826_182612

def calculate_bonus (final_score : ℕ) : ℕ :=
  if final_score < 1000 then
    (final_score * 20) / 100
  else if final_score < 2000 then
    200 + ((final_score - 1000) * 30) / 100
  else
    200 + 300 + ((final_score - 2000) * 50) / 100

theorem petya_bonus_points :
  calculate_bonus 2370 = 685 := by sorry

end petya_bonus_points_l1826_182612


namespace die_expected_value_l1826_182647

/-- Represents a fair six-sided die -/
def Die := Fin 6

/-- The strategy for two rolls -/
def strategy2 (d : Die) : Bool :=
  d.val ≥ 4

/-- The strategy for three rolls -/
def strategy3 (d : Die) : Bool :=
  d.val ≥ 5

/-- Expected value of a single roll -/
def E1 : ℚ := 3.5

/-- Expected value with two opportunities to roll -/
def E2 : ℚ := 4.25

/-- Expected value with three opportunities to roll -/
def E3 : ℚ := 14/3

theorem die_expected_value :
  (E2 = 4.25) ∧ (E3 = 14/3) := by
  sorry

end die_expected_value_l1826_182647


namespace angle_sum_pi_over_two_l1826_182653

theorem angle_sum_pi_over_two (a b : Real) (h1 : 0 < a ∧ a < π / 2) (h2 : 0 < b ∧ b < π / 2)
  (eq1 : 2 * Real.sin a ^ 3 + 3 * Real.sin b ^ 2 = 1)
  (eq2 : 2 * Real.sin (3 * a) - 3 * Real.sin (3 * b) = 0) :
  a + 3 * b = π / 2 := by
  sorry

end angle_sum_pi_over_two_l1826_182653


namespace divisibility_implies_zero_product_l1826_182600

theorem divisibility_implies_zero_product (p q r : ℝ) : 
  (∀ x, ∃ k, x^4 + 6*x^3 + 4*p*x^2 + 2*q*x + r = k * (x^3 + 4*x^2 + 2*x + 1)) →
  (p + q) * r = 0 := by
sorry

end divisibility_implies_zero_product_l1826_182600


namespace five_balls_three_boxes_l1826_182607

def indistinguishable_distributions (n : ℕ) (k : ℕ) : ℕ :=
  sorry

theorem five_balls_three_boxes : 
  indistinguishable_distributions 5 3 = 5 := by
  sorry

end five_balls_three_boxes_l1826_182607


namespace storage_to_total_ratio_l1826_182659

def total_planks : ℕ := 200
def friends_planks : ℕ := 20
def store_planks : ℕ := 30

def parents_planks : ℕ := total_planks / 2

def storage_planks : ℕ := total_planks - parents_planks - friends_planks - store_planks

theorem storage_to_total_ratio :
  (storage_planks : ℚ) / total_planks = 1 / 2 := by
  sorry

end storage_to_total_ratio_l1826_182659


namespace tangent_circles_slope_l1826_182617

/-- Definition of circle w1 -/
def w1 (x y : ℝ) : Prop := x^2 + y^2 + 10*x - 20*y - 77 = 0

/-- Definition of circle w2 -/
def w2 (x y : ℝ) : Prop := x^2 + y^2 - 10*x - 20*y + 193 = 0

/-- Definition of a line y = mx -/
def line (m x y : ℝ) : Prop := y = m * x

/-- Definition of internal tangency -/
def internallyTangent (x y r : ℝ) : Prop := (x - 5)^2 + (y - 10)^2 = (8 - r)^2

/-- Definition of external tangency -/
def externallyTangent (x y r : ℝ) : Prop := (x + 5)^2 + (y - 10)^2 = (r + 12)^2

/-- Main theorem -/
theorem tangent_circles_slope : 
  ∃ (m : ℝ), m > 0 ∧ 
  (∀ (m' : ℝ), m' > 0 → 
    (∃ (x y r : ℝ), line m' x y ∧ internallyTangent x y r ∧ externallyTangent x y r) 
    → m' ≥ m) ∧ 
  m^2 = 81/4 := by
sorry

end tangent_circles_slope_l1826_182617


namespace max_value_trig_expression_l1826_182650

theorem max_value_trig_expression (a b c : ℝ) :
  (∃ (θ : ℝ), ∀ (φ : ℝ), a * Real.cos θ + b * Real.sin θ + c * Real.sin (2 * θ) ≥ 
                         a * Real.cos φ + b * Real.sin φ + c * Real.sin (2 * φ)) →
  (∃ (θ : ℝ), a * Real.cos θ + b * Real.sin θ + c * Real.sin (2 * θ) = Real.sqrt (2 * (a^2 + b^2 + c^2))) :=
by sorry

end max_value_trig_expression_l1826_182650


namespace range_of_f_l1826_182664

-- Define the function
def f (x : ℝ) : ℝ := |x + 5| - |x - 3|

-- State the theorem
theorem range_of_f :
  Set.range f = Set.Icc (-8 : ℝ) 8 :=
sorry

end range_of_f_l1826_182664


namespace percentage_of_160_to_50_l1826_182620

theorem percentage_of_160_to_50 : ∀ (x y : ℝ), x = 160 ∧ y = 50 → (x / y) * 100 = 320 := by
  sorry

end percentage_of_160_to_50_l1826_182620


namespace smile_area_l1826_182696

/-- The area of the "smile" region formed by two sectors and a semicircle -/
theorem smile_area : 
  ∀ (r₁ r₂ : ℝ) (θ : ℝ),
  r₁ = 3 → r₂ = 2 → θ = π/4 →
  2 * (1/2 * r₁^2 * θ) + 1/2 * π * r₂^2 = 17*π/4 :=
by sorry

end smile_area_l1826_182696


namespace distance_range_for_intersecting_circles_l1826_182633

-- Define the radii of the two circles
def r₁ : ℝ := 3
def r₂ : ℝ := 5

-- Define the property of intersection
def intersecting (d : ℝ) : Prop := d < r₁ + r₂ ∧ d > abs (r₁ - r₂)

-- Theorem statement
theorem distance_range_for_intersecting_circles (d : ℝ) 
  (h : intersecting d) : 2 < d ∧ d < 8 := by sorry

end distance_range_for_intersecting_circles_l1826_182633


namespace chessboard_color_swap_theorem_l1826_182661

/-- A color is represented by a natural number -/
def Color := ℕ

/-- A chessboard is represented by a function from coordinates to colors -/
def Chessboard (n : ℕ) := Fin (2*n) → Fin (2*n) → Color

/-- A rectangle on the chessboard is defined by its corner coordinates -/
structure Rectangle (n : ℕ) where
  i1 : Fin (2*n)
  j1 : Fin (2*n)
  i2 : Fin (2*n)
  j2 : Fin (2*n)

/-- Predicate to check if all corners of a rectangle have the same color -/
def same_color_corners (board : Chessboard n) (rect : Rectangle n) : Prop :=
  board rect.i1 rect.j1 = board rect.i1 rect.j2 ∧
  board rect.i1 rect.j1 = board rect.i2 rect.j1 ∧
  board rect.i1 rect.j1 = board rect.i2 rect.j2

/-- Main theorem: There exist two tiles in the same column such that swapping
    their colors creates a rectangle with all four corners of the same color -/
theorem chessboard_color_swap_theorem (n : ℕ) (board : Chessboard n) :
  ∃ (i1 i2 j : Fin (2*n)) (rect : Rectangle n),
    i1 ≠ i2 ∧
    (∀ (i : Fin (2*n)), board i j ≠ board i1 j → board i j = board i2 j) →
    same_color_corners board rect :=
  sorry

end chessboard_color_swap_theorem_l1826_182661


namespace sum_25_36_in_base3_l1826_182678

/-- Converts a natural number from base 10 to base 3 -/
def toBase3 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a list of digits in base 3 to a natural number in base 10 -/
def fromBase3 (digits : List ℕ) : ℕ :=
  sorry

theorem sum_25_36_in_base3 :
  toBase3 (25 + 36) = [2, 0, 2, 1] :=
sorry

end sum_25_36_in_base3_l1826_182678


namespace barbaras_score_l1826_182685

theorem barbaras_score (total_students : ℕ) (students_without_barbara : ℕ) 
  (avg_without_barbara : ℚ) (avg_with_barbara : ℚ) :
  total_students = 20 →
  students_without_barbara = 19 →
  avg_without_barbara = 78 →
  avg_with_barbara = 79 →
  (total_students * avg_with_barbara - students_without_barbara * avg_without_barbara : ℚ) = 98 := by
  sorry

#check barbaras_score

end barbaras_score_l1826_182685


namespace arithmetic_sequence_problem_l1826_182657

/-- Given an arithmetic sequence {a_n} with a_2 = 7 and a_11 = a_9 + 6, prove a_1 = 4 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) : 
  (∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) → -- arithmetic sequence condition
  a 2 = 7 →
  a 11 = a 9 + 6 →
  a 1 = 4 := by
sorry

end arithmetic_sequence_problem_l1826_182657


namespace equiangular_and_equilateral_implies_regular_polygon_l1826_182611

/-- A figure is equiangular if all its angles are equal. -/
def IsEquiangular (figure : Type) : Prop := sorry

/-- A figure is equilateral if all its sides are equal. -/
def IsEquilateral (figure : Type) : Prop := sorry

/-- A figure is a regular polygon if it is both equiangular and equilateral. -/
def IsRegularPolygon (figure : Type) : Prop := 
  IsEquiangular figure ∧ IsEquilateral figure

/-- Theorem: If a figure is both equiangular and equilateral, then it is a regular polygon. -/
theorem equiangular_and_equilateral_implies_regular_polygon 
  (figure : Type) 
  (h1 : IsEquiangular figure) 
  (h2 : IsEquilateral figure) : 
  IsRegularPolygon figure := by
  sorry


end equiangular_and_equilateral_implies_regular_polygon_l1826_182611


namespace simplify_expression_l1826_182652

theorem simplify_expression (b : ℝ) : 3*b*(3*b^2 + 2*b) - 2*b^2 + b*(2*b+1) = 9*b^3 + 6*b^2 + b := by
  sorry

end simplify_expression_l1826_182652


namespace garden_area_l1826_182635

-- Define the rectangle garden
def rectangular_garden (length width : ℝ) := length * width

-- Theorem statement
theorem garden_area : rectangular_garden 12 5 = 60 := by
  sorry

end garden_area_l1826_182635


namespace max_perimeter_triangle_is_isosceles_l1826_182638

/-- Given a fixed base length and a fixed angle at one vertex, 
    the triangle with maximum perimeter is isosceles. -/
theorem max_perimeter_triangle_is_isosceles 
  (b : ℝ) 
  (β : ℝ) 
  (h1 : b > 0) 
  (h2 : 0 < β ∧ β < π) : 
  ∃ (a c : ℝ), 
    a > 0 ∧ c > 0 ∧
    a = c ∧
    ∀ (a' c' : ℝ), 
      a' > 0 → c' > 0 → 
      a' + b + c' ≤ a + b + c := by
  sorry

end max_perimeter_triangle_is_isosceles_l1826_182638


namespace converse_of_zero_product_is_false_l1826_182694

theorem converse_of_zero_product_is_false :
  ¬ (∀ (a b : ℝ), a * b = 0 → a = 0) :=
sorry

end converse_of_zero_product_is_false_l1826_182694


namespace distribution_schemes_correct_l1826_182616

/-- The number of ways to distribute 5 volunteers to 3 different Olympic venues,
    with at least one volunteer assigned to each venue. -/
def distributionSchemes : ℕ := 150

/-- Theorem stating that the number of distribution schemes is correct. -/
theorem distribution_schemes_correct : distributionSchemes = 150 := by sorry

end distribution_schemes_correct_l1826_182616


namespace lucia_weekly_dance_cost_l1826_182686

/-- Represents the cost of dance classes for a week -/
def total_dance_cost (hip_hop_classes ballet_classes jazz_classes : ℕ) 
  (hip_hop_cost ballet_cost jazz_cost : ℕ) : ℕ :=
  hip_hop_classes * hip_hop_cost + ballet_classes * ballet_cost + jazz_classes * jazz_cost

/-- Proves that Lucia's weekly dance class cost is $52 -/
theorem lucia_weekly_dance_cost : 
  total_dance_cost 2 2 1 10 12 8 = 52 := by
  sorry

end lucia_weekly_dance_cost_l1826_182686


namespace complex_number_simplification_l1826_182683

theorem complex_number_simplification :
  (2 - 5 * Complex.I) - (-3 + 7 * Complex.I) - 4 * (-1 + 2 * Complex.I) = 1 - 4 * Complex.I := by
  sorry

end complex_number_simplification_l1826_182683


namespace a_left_after_three_days_l1826_182637

/-- The number of days it takes A to complete the work alone -/
def a_days : ℝ := 21

/-- The number of days it takes B to complete the work alone -/
def b_days : ℝ := 28

/-- The number of days B worked alone to complete the remaining work -/
def b_remaining_days : ℝ := 21

/-- The number of days A worked before leaving -/
def x : ℝ := 3

theorem a_left_after_three_days :
  (x / (a_days⁻¹ + b_days⁻¹)⁻¹) + (b_remaining_days / b_days) = 1 :=
sorry

end a_left_after_three_days_l1826_182637


namespace exists_k_undecided_tournament_l1826_182684

/-- A tournament is a complete directed graph where each edge represents a match outcome. -/
def Tournament (n : ℕ) := Fin n → Fin n → Bool

/-- A tournament is k-undecided if for every set of k players, there exists a player who has defeated all of them. -/
def IsKUndecided (k : ℕ) (n : ℕ) (t : Tournament n) : Prop :=
  ∀ (A : Finset (Fin n)), A.card = k →
    ∃ (p : Fin n), p ∉ A ∧ ∀ (a : Fin n), a ∈ A → t p a = true

/-- For every positive integer k, there exists a k-undecided tournament with more than k players. -/
theorem exists_k_undecided_tournament (k : ℕ+) :
  ∃ (n : ℕ) (t : Tournament n), n > k ∧ IsKUndecided k n t :=
sorry

end exists_k_undecided_tournament_l1826_182684


namespace infinitely_many_primes_6k_plus_1_l1826_182634

theorem infinitely_many_primes_6k_plus_1 :
  ∀ S : Finset Nat, (∀ p ∈ S, Prime p ∧ ∃ k, p = 6*k + 1) →
  ∃ q, Prime q ∧ (∃ m, q = 6*m + 1) ∧ q ∉ S :=
by sorry

end infinitely_many_primes_6k_plus_1_l1826_182634


namespace total_students_l1826_182658

theorem total_students (general : ℕ) (biology : ℕ) (chemistry : ℕ) (math : ℕ) (arts : ℕ) 
  (physics : ℕ) (history : ℕ) (literature : ℕ) : 
  general = 30 →
  biology = 2 * general →
  chemistry = general + 10 →
  math = (3 * (general + biology + chemistry)) / 5 →
  arts * 20 / 100 = general →
  physics = general + chemistry - 5 →
  history = (3 * general) / 4 →
  literature = history + 15 →
  general + biology + chemistry + math + arts + physics + history + literature = 484 :=
by sorry

end total_students_l1826_182658


namespace sum_base6_100_l1826_182632

/-- Converts a number from base 10 to base 6 -/
def toBase6 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 6 to base 10 -/
def fromBase6 (n : ℕ) : ℕ := sorry

/-- Sum of numbers from 1 to n in base 6 -/
def sumBase6 (n : ℕ) : ℕ := sorry

theorem sum_base6_100 : sumBase6 100 = toBase6 (fromBase6 6110) := by sorry

end sum_base6_100_l1826_182632


namespace c_absolute_value_l1826_182618

def g (a b c : ℤ) (x : ℂ) : ℂ := a * x^4 + b * x^3 + c * x^2 + b * x + a

theorem c_absolute_value (a b c : ℤ) :
  (∀ d : ℤ, d ≠ 1 → d ∣ a → d ∣ b → d ∣ c → False) →
  g a b c (3 + I) = 0 →
  |c| = 142 := by sorry

end c_absolute_value_l1826_182618


namespace pebble_collection_sum_l1826_182687

/-- The sum of an arithmetic sequence with first term 2, common difference 3, and 15 terms -/
def arithmetic_sum : ℕ → ℕ
| n => n * (4 + 3 * (n - 1)) / 2

/-- Theorem stating that the sum of the first 15 terms of the arithmetic sequence is 345 -/
theorem pebble_collection_sum : arithmetic_sum 15 = 345 := by
  sorry

end pebble_collection_sum_l1826_182687


namespace bus_driver_regular_rate_l1826_182601

/-- Represents the bus driver's compensation structure and work hours -/
structure BusDriverCompensation where
  regularRate : ℝ
  overtimeRate : ℝ
  regularHours : ℝ
  overtimeHours : ℝ
  totalCompensation : ℝ

/-- Calculates the total compensation based on the given rates and hours -/
def calculateTotalCompensation (c : BusDriverCompensation) : ℝ :=
  c.regularRate * c.regularHours + c.overtimeRate * c.overtimeHours

/-- Theorem stating that the bus driver's regular rate is $16 per hour -/
theorem bus_driver_regular_rate :
  ∃ (c : BusDriverCompensation),
    c.regularHours = 40 ∧
    c.overtimeHours = 8 ∧
    c.overtimeRate = 1.75 * c.regularRate ∧
    c.totalCompensation = 864 ∧
    calculateTotalCompensation c = c.totalCompensation ∧
    c.regularRate = 16 := by
  sorry


end bus_driver_regular_rate_l1826_182601


namespace simplify_expression_l1826_182627

theorem simplify_expression (b : ℝ) : (1 : ℝ) * (3 * b) * (4 * b^2) * (5 * b^3) * (6 * b^4) = 360 * b^10 := by
  sorry

end simplify_expression_l1826_182627


namespace P_infimum_and_no_minimum_l1826_182651

/-- The function P : ℝ² → ℝ defined by P(X₁, X₂) = X₁² + (1 - X₁X₂)² -/
def P : ℝ × ℝ → ℝ := fun (X₁, X₂) ↦ X₁^2 + (1 - X₁ * X₂)^2

theorem P_infimum_and_no_minimum :
  (∀ ε > 0, ∃ (X₁ X₂ : ℝ), P (X₁, X₂) < ε) ∧
  (¬∃ (X₁ X₂ : ℝ), ∀ (Y₁ Y₂ : ℝ), P (X₁, X₂) ≤ P (Y₁, Y₂)) := by
  sorry

end P_infimum_and_no_minimum_l1826_182651


namespace power_function_decreasing_m_l1826_182665

/-- A power function y = ax^b where a and b are constants and x > 0 -/
def isPowerFunction (y : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, ∀ x : ℝ, x > 0 → y x = a * x ^ b

/-- A decreasing function on (0, +∞) -/
def isDecreasingOn (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → f x₂ < f x₁

theorem power_function_decreasing_m (m : ℝ) :
  isPowerFunction (fun x => (m^2 - 2*m - 2) * x^(-4*m - 2)) →
  isDecreasingOn (fun x => (m^2 - 2*m - 2) * x^(-4*m - 2)) →
  m = 3 := by
  sorry

end power_function_decreasing_m_l1826_182665


namespace daily_rental_cost_satisfies_conditions_l1826_182639

/-- Represents the daily rental cost of a car in dollars -/
def daily_rental_cost : ℝ := 30

/-- Represents the cost per mile in dollars -/
def cost_per_mile : ℝ := 0.18

/-- Represents the total budget in dollars -/
def total_budget : ℝ := 75

/-- Represents the number of miles that can be driven -/
def miles_driven : ℝ := 250

/-- Theorem stating that the daily rental cost satisfies the given conditions -/
theorem daily_rental_cost_satisfies_conditions :
  daily_rental_cost + (cost_per_mile * miles_driven) = total_budget :=
by sorry

end daily_rental_cost_satisfies_conditions_l1826_182639


namespace percentage_problem_l1826_182699

theorem percentage_problem : ∃ x : ℝ, (0.001 * x = 0.24) ∧ (x = 240) := by sorry

end percentage_problem_l1826_182699


namespace quadratic_inequality_l1826_182693

theorem quadratic_inequality (b c : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = x^2 + b*x + c) 
  (h2 : f (-1) = f 3) : 
  f 1 < c ∧ c < f (-1) := by
sorry

end quadratic_inequality_l1826_182693


namespace log_base_three_squared_l1826_182606

theorem log_base_three_squared (m : ℝ) (b : ℝ) (h : 3^m = b) : 
  Real.log b / Real.log (3^2) = m / 2 := by
  sorry

end log_base_three_squared_l1826_182606


namespace law_school_applicants_l1826_182619

theorem law_school_applicants (PS : ℕ) (GPA_high : ℕ) (not_PS_GPA_low : ℕ) (PS_GPA_high : ℕ) :
  PS = 15 →
  GPA_high = 20 →
  not_PS_GPA_low = 10 →
  PS_GPA_high = 5 →
  PS + GPA_high - PS_GPA_high + not_PS_GPA_low = 40 :=
by sorry

end law_school_applicants_l1826_182619


namespace fifth_month_sales_l1826_182697

def sales_1 : ℕ := 6435
def sales_2 : ℕ := 6927
def sales_3 : ℕ := 6855
def sales_4 : ℕ := 7230
def sales_6 : ℕ := 7991
def average_sales : ℕ := 7000
def num_months : ℕ := 6

theorem fifth_month_sales :
  ∃ (sales_5 : ℕ),
    (sales_1 + sales_2 + sales_3 + sales_4 + sales_5 + sales_6) / num_months = average_sales ∧
    sales_5 = 6562 := by
  sorry

end fifth_month_sales_l1826_182697


namespace part1_part2_l1826_182609

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * (1 / (2^x - 1) + a / 2)

-- Part 1: If f(x) is even and f(1) = 3/2, then f(-1) = 3/2
theorem part1 (a : ℝ) : 
  (∀ x : ℝ, x ≠ 0 → f a x = f a (-x)) → 
  f a 1 = 3/2 → 
  f a (-1) = 3/2 := by sorry

-- Part 2: If a = 1, then f(x) is an even function
theorem part2 : 
  ∀ x : ℝ, x ≠ 0 → f 1 x = f 1 (-x) := by sorry

end part1_part2_l1826_182609


namespace stripe_difference_l1826_182642

/-- The number of stripes on one of Olga's shoes -/
def olga_stripes_per_shoe : ℕ := 3

/-- The total number of stripes on Olga's shoes -/
def olga_total_stripes : ℕ := 2 * olga_stripes_per_shoe

/-- The total number of stripes on Hortense's shoes -/
def hortense_total_stripes : ℕ := 2 * olga_total_stripes

/-- The total number of stripes on all their shoes -/
def total_stripes : ℕ := 22

/-- The number of stripes on Rick's shoes -/
def rick_total_stripes : ℕ := total_stripes - olga_total_stripes - hortense_total_stripes

theorem stripe_difference : olga_total_stripes - rick_total_stripes = 2 := by
  sorry

end stripe_difference_l1826_182642


namespace sandal_price_proof_l1826_182622

/-- Proves that the price of each pair of sandals is $3 given the conditions of Yanna's purchase. -/
theorem sandal_price_proof (num_shirts : ℕ) (shirt_price : ℕ) (num_sandals : ℕ) (bill_paid : ℕ) (change_received : ℕ) :
  num_shirts = 10 →
  shirt_price = 5 →
  num_sandals = 3 →
  bill_paid = 100 →
  change_received = 41 →
  (bill_paid - change_received - num_shirts * shirt_price) / num_sandals = 3 :=
by sorry

end sandal_price_proof_l1826_182622


namespace quartic_polynomial_property_l1826_182629

def Q (x : ℝ) (e f : ℝ) : ℝ := 3 * x^4 + 24 * x^3 + e * x^2 + f * x + 16

theorem quartic_polynomial_property (e f : ℝ) :
  (∀ r₁ r₂ r₃ r₄ : ℝ, Q r₁ e f = 0 ∧ Q r₂ e f = 0 ∧ Q r₃ e f = 0 ∧ Q r₄ e f = 0 →
    (-24 / 12 = e / 3) ∧
    (-24 / 12 = 3 + 24 + e + f + 16) ∧
    (e / 3 = 3 + 24 + e + f + 16)) →
  f = -39 := by
sorry

end quartic_polynomial_property_l1826_182629


namespace evaluate_expression_l1826_182660

theorem evaluate_expression : (8^6 / 8^4) * 3^10 = 3783136 := by
  sorry

end evaluate_expression_l1826_182660


namespace hexagon_perimeter_l1826_182666

/-- Hexagon ABCDEF with given side lengths -/
structure Hexagon where
  AB : ℝ
  BC : ℝ
  CD : ℝ
  DE : ℝ
  EF : ℝ
  AF : ℝ

/-- The perimeter of a hexagon -/
def perimeter (h : Hexagon) : ℝ :=
  h.AB + h.BC + h.CD + h.DE + h.EF + h.AF

/-- Theorem: The perimeter of the given hexagon is 7 + √10 -/
theorem hexagon_perimeter : 
  ∀ (h : Hexagon), 
  h.AB = 1 → h.BC = 1 → h.CD = 2 → h.DE = 2 → h.EF = 1 → h.AF = Real.sqrt 10 →
  perimeter h = 7 + Real.sqrt 10 := by
  sorry

end hexagon_perimeter_l1826_182666


namespace expand_expression_l1826_182662

theorem expand_expression (x : ℝ) : (20 * x - 25) * (3 * x) = 60 * x^2 - 75 * x := by
  sorry

end expand_expression_l1826_182662


namespace derivative_f_at_zero_l1826_182690

/-- The function f(x) = (2x+1)^2 -/
def f (x : ℝ) : ℝ := (2*x + 1)^2

/-- The derivative of f at x = 0 is 4 -/
theorem derivative_f_at_zero : 
  deriv f 0 = 4 := by sorry

end derivative_f_at_zero_l1826_182690


namespace find_n_l1826_182654

/-- Given that P = s / ((1 + k)^n + m), prove that n = (log((s/P) - m)) / (log(1 + k)) -/
theorem find_n (P s k m n : ℝ) (h : P = s / ((1 + k)^n + m)) (h1 : k > -1) (h2 : P > 0) (h3 : s > 0) :
  n = (Real.log ((s/P) - m)) / (Real.log (1 + k)) := by
  sorry

end find_n_l1826_182654


namespace function_b_increasing_on_negative_reals_l1826_182649

/-- The function f(x) = 1 - 1/x is increasing on the interval (-∞,0) -/
theorem function_b_increasing_on_negative_reals :
  ∀ x y : ℝ, x < y → x < 0 → y < 0 → (1 - 1/x) < (1 - 1/y) := by
sorry

end function_b_increasing_on_negative_reals_l1826_182649


namespace factorial_sum_ratio_l1826_182626

theorem factorial_sum_ratio (N : ℕ) (h : N > 0) : 
  (Nat.factorial (N + 1) + Nat.factorial (N - 1)) / Nat.factorial (N + 2) = 
  (N^2 + N + 1) / (N^3 + 3*N^2 + 2*N) := by
  sorry

end factorial_sum_ratio_l1826_182626


namespace quadratic_solution_l1826_182610

theorem quadratic_solution (b : ℚ) : 
  ((-4 : ℚ)^2 + b * (-4) - 45 = 0) → b = -29/4 := by
  sorry

end quadratic_solution_l1826_182610


namespace correct_average_points_l1826_182641

/-- Represents Melissa's basketball season statistics -/
structure BasketballSeason where
  totalGames : ℕ
  totalPoints : ℕ
  wonGames : ℕ
  averagePointDifference : ℕ

/-- Calculates the average points scored in won and lost games -/
def calculateAveragePoints (season : BasketballSeason) : ℕ × ℕ :=
  sorry

/-- Theorem stating the correct average points for won and lost games -/
theorem correct_average_points (season : BasketballSeason) 
  (h1 : season.totalGames = 20)
  (h2 : season.totalPoints = 400)
  (h3 : season.wonGames = 8)
  (h4 : season.averagePointDifference = 15) :
  calculateAveragePoints season = (29, 14) := by
  sorry

end correct_average_points_l1826_182641


namespace two_n_squared_lt_three_to_n_l1826_182681

theorem two_n_squared_lt_three_to_n (n : ℕ+) : 2 * n.val ^ 2 < 3 ^ n.val := by sorry

end two_n_squared_lt_three_to_n_l1826_182681


namespace evaluate_expression_l1826_182691

theorem evaluate_expression : 5 * 399 + 4 * 399 + 3 * 399 + 397 = 5185 := by
  sorry

end evaluate_expression_l1826_182691


namespace cosine_shift_equals_sine_l1826_182603

open Real

theorem cosine_shift_equals_sine (m : ℝ) : (∀ x, cos (x + m) = sin x) → m = 3 * π / 2 := by
  sorry

end cosine_shift_equals_sine_l1826_182603


namespace compute_expression_l1826_182682

theorem compute_expression : 3 * 3^4 - 4^55 / 4^54 = 239 := by sorry

end compute_expression_l1826_182682


namespace wang_processing_time_l1826_182679

/-- Given that Master Wang processes 92 parts in 4 days, 
    prove that it takes 9 days to process 207 parts using proportion. -/
theorem wang_processing_time 
  (parts_per_four_days : ℕ) 
  (h_parts : parts_per_four_days = 92) 
  (new_parts : ℕ) 
  (h_new_parts : new_parts = 207) : 
  (4 : ℚ) * new_parts / parts_per_four_days = 9 := by
  sorry

end wang_processing_time_l1826_182679


namespace smallest_number_l1826_182688

def digits : List Nat := [1, 4, 5]

def is_permutation (n : Nat) : Prop :=
  let digits_of_n := n.digits 10
  digits_of_n.length = digits.length ∧ digits_of_n.toFinset = digits.toFinset

theorem smallest_number :
  ∀ n : Nat, is_permutation n → 145 ≤ n := by
  sorry

end smallest_number_l1826_182688


namespace larger_number_proof_l1826_182656

theorem larger_number_proof (L S : ℕ) (h1 : L - S = 1365) (h2 : L = 4 * S + 15) : L = 1815 := by
  sorry

end larger_number_proof_l1826_182656


namespace door_purchase_savings_l1826_182640

/-- Calculates the cost of purchasing doors with the "buy 3 get 1 free" offer -/
def cost_with_offer (num_doors : ℕ) (price_per_door : ℕ) : ℕ :=
  ((num_doors + 3) / 4) * 3 * price_per_door

/-- Calculates the regular cost of purchasing doors without any offer -/
def regular_cost (num_doors : ℕ) (price_per_door : ℕ) : ℕ :=
  num_doors * price_per_door

/-- Calculates the savings when purchasing doors with the offer -/
def savings (num_doors : ℕ) (price_per_door : ℕ) : ℕ :=
  regular_cost num_doors price_per_door - cost_with_offer num_doors price_per_door

theorem door_purchase_savings :
  let alice_doors := 6
  let bob_doors := 9
  let price_per_door := 120
  let total_doors := alice_doors + bob_doors
  savings total_doors price_per_door = 600 :=
by sorry

end door_purchase_savings_l1826_182640


namespace log_simplification_l1826_182631

-- Define the base-10 logarithm
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_simplification : (log10 2)^2 + log10 2 * log10 5 + log10 5 = 1 := by
  sorry

end log_simplification_l1826_182631


namespace chord_length_concentric_circles_l1826_182636

/-- Given two concentric circles with radii a and b (a > b), 
    if the area of the ring between them is 16π,
    then the length of a chord of the larger circle 
    that is tangent to the smaller circle is 8. -/
theorem chord_length_concentric_circles 
  (a b : ℝ) (h1 : a > b) (h2 : a^2 - b^2 = 16) : 
  ∃ c : ℝ, c = 8 ∧ c^2 = 4 * (a^2 - b^2) := by
sorry

end chord_length_concentric_circles_l1826_182636


namespace prob_over_60_and_hypertension_is_9_percent_l1826_182698

/-- The probability of a person being over 60 years old in the region -/
def prob_over_60 : ℝ := 0.2

/-- The probability of a person having hypertension given they are over 60 -/
def prob_hypertension_given_over_60 : ℝ := 0.45

/-- The probability of a person being both over 60 and having hypertension -/
def prob_over_60_and_hypertension : ℝ := prob_over_60 * prob_hypertension_given_over_60

theorem prob_over_60_and_hypertension_is_9_percent :
  prob_over_60_and_hypertension = 0.09 := by
  sorry

end prob_over_60_and_hypertension_is_9_percent_l1826_182698


namespace lamp_probability_l1826_182670

/-- The number of red lamps -/
def num_red_lamps : ℕ := 4

/-- The number of blue lamps -/
def num_blue_lamps : ℕ := 4

/-- The total number of lamps -/
def total_lamps : ℕ := num_red_lamps + num_blue_lamps

/-- The number of lamps turned on -/
def num_on_lamps : ℕ := 4

/-- The probability of the leftmost lamp being blue and on, and the rightmost lamp being red and off -/
theorem lamp_probability : 
  (num_red_lamps : ℚ) * (num_blue_lamps : ℚ) * (Nat.choose (total_lamps - 2) (num_on_lamps - 1)) / 
  ((Nat.choose total_lamps num_red_lamps) * (Nat.choose total_lamps num_on_lamps)) = 5 / 7 := by
  sorry

end lamp_probability_l1826_182670


namespace geraldine_dolls_count_l1826_182602

theorem geraldine_dolls_count (jazmin_dolls total_dolls : ℕ) 
  (h1 : jazmin_dolls = 1209)
  (h2 : total_dolls = 3395) :
  total_dolls - jazmin_dolls = 2186 :=
by sorry

end geraldine_dolls_count_l1826_182602


namespace gcd_of_36_and_54_l1826_182689

theorem gcd_of_36_and_54 : Nat.gcd 36 54 = 18 := by
  sorry

end gcd_of_36_and_54_l1826_182689


namespace spring_mass_for_32cm_l1826_182614

/-- Represents the relationship between spring length and mass -/
def spring_length (initial_length : ℝ) (extension_rate : ℝ) (mass : ℝ) : ℝ :=
  initial_length + extension_rate * mass

/-- Theorem: For a spring with initial length 18 cm and extension rate 2 cm/kg,
    a length of 32 cm corresponds to a mass of 7 kg -/
theorem spring_mass_for_32cm :
  spring_length 18 2 7 = 32 :=
by sorry

end spring_mass_for_32cm_l1826_182614


namespace triangular_array_coin_sum_l1826_182669

/-- The number of coins in a triangular array with n rows -/
def triangular_sum (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem triangular_array_coin_sum :
  ∃ (N : ℕ), triangular_sum N = 2211 ∧ sum_of_digits N = 12 :=
sorry

end triangular_array_coin_sum_l1826_182669


namespace NaHCO3_moles_equal_H2O_moles_NaHCO3_moles_proof_l1826_182645

-- Define the molar masses and quantities
def molar_mass_H2O : ℝ := 18
def HNO3_moles : ℝ := 2
def H2O_grams : ℝ := 36

-- Define the reaction stoichiometry
def HNO3_to_H2O_ratio : ℝ := 1
def NaHCO3_to_H2O_ratio : ℝ := 1

-- Theorem statement
theorem NaHCO3_moles_equal_H2O_moles : ℝ → Prop :=
  fun NaHCO3_moles =>
    let H2O_moles := H2O_grams / molar_mass_H2O
    NaHCO3_moles = H2O_moles ∧ NaHCO3_moles = HNO3_moles

-- Proof (skipped)
theorem NaHCO3_moles_proof : ∃ (x : ℝ), NaHCO3_moles_equal_H2O_moles x :=
sorry

end NaHCO3_moles_equal_H2O_moles_NaHCO3_moles_proof_l1826_182645


namespace number_equation_solution_l1826_182623

theorem number_equation_solution : 
  ∃ x : ℝ, (5020 - (x / 20.08) = 4970) ∧ (x = 1004) := by sorry

end number_equation_solution_l1826_182623


namespace direct_variation_with_constant_l1826_182648

/-- A function that varies directly as x with an additional constant term -/
def f (k c : ℝ) (x : ℝ) : ℝ := k * x + c

/-- Theorem stating that if f(3) = 9 and f(4) = 12, then f(-5) = -15 -/
theorem direct_variation_with_constant 
  (k c : ℝ) 
  (h1 : f k c 3 = 9) 
  (h2 : f k c 4 = 12) : 
  f k c (-5) = -15 := by
  sorry

#check direct_variation_with_constant

end direct_variation_with_constant_l1826_182648


namespace vectors_orthogonal_l1826_182608

/-- Two vectors in ℝ² are orthogonal if their dot product is zero -/
def orthogonal (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

/-- The first vector -/
def v : ℝ × ℝ := (3, 4)

/-- The second vector -/
def w (x : ℝ) : ℝ × ℝ := (x, -7)

/-- The theorem stating that the vectors are orthogonal when x = 28/3 -/
theorem vectors_orthogonal : orthogonal v (w (28/3)) := by
  sorry

end vectors_orthogonal_l1826_182608


namespace distinct_hexagon_colorings_l1826_182624

-- Define the number of disks and colors
def num_disks : ℕ := 6
def num_blue : ℕ := 3
def num_red : ℕ := 2
def num_green : ℕ := 1

-- Define the symmetry group of a hexagon
def hexagon_symmetries : ℕ := 12

-- Define the function to calculate the number of distinct colorings
def distinct_colorings : ℕ :=
  let total_colorings := (num_disks.choose num_blue) * ((num_disks - num_blue).choose num_red)
  let fixed_points_identity := total_colorings
  let fixed_points_reflection := 3 * (3 * 2 * 1)  -- 3 reflections, each with 6 fixed points
  let fixed_points_rotation := 0  -- 120° and 240° rotations have no fixed points
  (fixed_points_identity + fixed_points_reflection + fixed_points_rotation) / hexagon_symmetries

-- Theorem statement
theorem distinct_hexagon_colorings :
  distinct_colorings = 13 :=
sorry

end distinct_hexagon_colorings_l1826_182624


namespace fifth_square_area_l1826_182673

theorem fifth_square_area (s : ℝ) (h : s + 5 = 11) : s^2 = 36 := by
  sorry

end fifth_square_area_l1826_182673


namespace solve_equation_and_evaluate_l1826_182628

theorem solve_equation_and_evaluate (x : ℝ) : 
  2*x - 7 = 8*x - 1 → 5*(x - 3) = -20 := by
  sorry

end solve_equation_and_evaluate_l1826_182628


namespace nabla_computation_l1826_182674

-- Define the nabla operation
def nabla (a b : ℕ) : ℕ := 3 + b^a

-- Theorem statement
theorem nabla_computation : nabla (nabla 1 3) 2 = 67 := by
  sorry

end nabla_computation_l1826_182674


namespace sphere_volume_from_surface_area_l1826_182671

theorem sphere_volume_from_surface_area :
  ∀ (R : ℝ), 
  R > 0 →
  4 * π * R^2 = 24 * π →
  (4 / 3) * π * R^3 = 8 * Real.sqrt 6 * π :=
by
  sorry

end sphere_volume_from_surface_area_l1826_182671


namespace evaluate_expression_l1826_182677

theorem evaluate_expression (x y : ℝ) (hx : x = 3) (hy : y = 2) : y * (y - 3 * x) = -14 := by
  sorry

end evaluate_expression_l1826_182677


namespace hockey_league_games_l1826_182672

/-- The number of games played in a hockey league season -/
def number_of_games (n : ℕ) (m : ℕ) : ℕ :=
  n * (n - 1) / 2 * m

theorem hockey_league_games :
  number_of_games 17 10 = 1360 := by
  sorry

end hockey_league_games_l1826_182672


namespace fifth_day_income_l1826_182692

def cab_driver_income (income_4_days : List ℝ) (average_income : ℝ) : ℝ :=
  5 * average_income - income_4_days.sum

theorem fifth_day_income 
  (income_4_days : List ℝ) 
  (average_income : ℝ) 
  (h1 : income_4_days.length = 4) 
  (h2 : average_income = (income_4_days.sum + cab_driver_income income_4_days average_income) / 5) :
  cab_driver_income income_4_days average_income = 
    5 * average_income - income_4_days.sum :=
by
  sorry

#eval cab_driver_income [300, 150, 750, 200] 400

end fifth_day_income_l1826_182692


namespace first_class_males_count_l1826_182604

/-- Represents the number of male students in the first class -/
def first_class_males : ℕ := sorry

/-- Represents the number of female students in the first class -/
def first_class_females : ℕ := 13

/-- Represents the number of male students in the second class -/
def second_class_males : ℕ := 14

/-- Represents the number of female students in the second class -/
def second_class_females : ℕ := 18

/-- Represents the number of male students in the third class -/
def third_class_males : ℕ := 15

/-- Represents the number of female students in the third class -/
def third_class_females : ℕ := 17

/-- Represents the number of students unable to partner with the opposite gender -/
def unpartnered_students : ℕ := 2

theorem first_class_males_count : first_class_males = 21 := by
  sorry

end first_class_males_count_l1826_182604


namespace tangent_line_equation_l1826_182676

theorem tangent_line_equation (a : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * Real.cos x
  let slope : ℝ := -a * Real.sin (π / 6)
  slope = 1 / 2 →
  let x₀ : ℝ := π / 6
  let y₀ : ℝ := f x₀
  ∀ x y : ℝ, (y - y₀ = slope * (x - x₀)) ↔ (x - 2 * y - Real.sqrt 3 - π / 6 = 0) :=
by sorry

end tangent_line_equation_l1826_182676


namespace circle_centers_distance_l1826_182615

-- Define the circles
def circle_O₁ : ℝ := 3
def circle_O₂ : ℝ := 7

-- Define the condition of at most one common point
def at_most_one_common_point (d : ℝ) : Prop :=
  d ≥ circle_O₁ + circle_O₂ ∨ d ≤ abs (circle_O₁ - circle_O₂)

-- State the theorem
theorem circle_centers_distance (d : ℝ) :
  at_most_one_common_point d → d ≠ 8 :=
by sorry

end circle_centers_distance_l1826_182615


namespace continuous_third_derivative_product_nonnegative_l1826_182605

/-- A real function with continuous third derivative has a point where the product of
    the function value and its first three derivatives is non-negative. -/
theorem continuous_third_derivative_product_nonnegative (f : ℝ → ℝ) 
  (hf : ContDiff ℝ 3 f) :
  ∃ a : ℝ, f a * (deriv f a) * (deriv^[2] f a) * (deriv^[3] f a) ≥ 0 := by
  sorry

end continuous_third_derivative_product_nonnegative_l1826_182605


namespace elenas_garden_tulips_l1826_182613

/-- Represents Elena's garden with lilies and tulips. -/
structure Garden where
  lilies : ℕ
  tulips : ℕ
  lily_petals : ℕ
  tulip_petals : ℕ
  total_petals : ℕ

/-- Theorem stating the number of tulips in Elena's garden. -/
theorem elenas_garden_tulips (g : Garden)
  (h1 : g.lilies = 8)
  (h2 : g.lily_petals = 6)
  (h3 : g.tulip_petals = 3)
  (h4 : g.total_petals = 63)
  (h5 : g.total_petals = g.lilies * g.lily_petals + g.tulips * g.tulip_petals) :
  g.tulips = 5 := by
  sorry


end elenas_garden_tulips_l1826_182613


namespace inequality_solution_l1826_182646

theorem inequality_solution (x : ℝ) :
  (x^2 - 9) / (x^2 - 1) > 0 ↔ x > 3 ∨ x < -3 ∨ (-1 < x ∧ x < 1) :=
by sorry

end inequality_solution_l1826_182646


namespace sum_geq_sqrt_product_of_sum_products_eq_27_l1826_182675

theorem sum_geq_sqrt_product_of_sum_products_eq_27
  (x y z : ℝ)
  (pos_x : 0 < x)
  (pos_y : 0 < y)
  (pos_z : 0 < z)
  (sum_products : x * y + y * z + z * x = 27) :
  x + y + z ≥ Real.sqrt (3 * x * y * z) ∧
  (x + y + z = Real.sqrt (3 * x * y * z) ↔ x = 3 ∧ y = 3 ∧ z = 3) :=
by sorry

end sum_geq_sqrt_product_of_sum_products_eq_27_l1826_182675


namespace least_positive_integer_satisfying_congruences_l1826_182621

theorem least_positive_integer_satisfying_congruences : ∃ x : ℕ, x > 0 ∧
  ((x : ℤ) + 127 ≡ 53 [ZMOD 15]) ∧
  ((x : ℤ) + 104 ≡ 76 [ZMOD 7]) ∧
  (∀ y : ℕ, y > 0 →
    ((y : ℤ) + 127 ≡ 53 [ZMOD 15]) →
    ((y : ℤ) + 104 ≡ 76 [ZMOD 7]) →
    x ≤ y) ∧
  x = 91 := by
sorry

end least_positive_integer_satisfying_congruences_l1826_182621


namespace building_height_problem_l1826_182668

theorem building_height_problem (h_taller h_shorter : ℝ) : 
  h_taller - h_shorter = 36 →
  h_shorter / h_taller = 5 / 7 →
  h_taller = 126 := by
  sorry

end building_height_problem_l1826_182668


namespace probability_three_primes_l1826_182680

def num_dice : ℕ := 5
def num_primes : ℕ := 3
def prob_prime : ℚ := 2/5

theorem probability_three_primes :
  (num_dice.choose num_primes : ℚ) * prob_prime^num_primes * (1 - prob_prime)^(num_dice - num_primes) = 720/3125 := by
  sorry

end probability_three_primes_l1826_182680


namespace sibling_count_product_l1826_182625

/-- Represents a family with a given number of boys and girls -/
structure Family :=
  (boys : ℕ)
  (girls : ℕ)

/-- Represents a sibling in a family -/
structure Sibling :=
  (family : Family)
  (isBoy : Bool)

/-- Counts the number of sisters a sibling has -/
def sisterCount (s : Sibling) : ℕ :=
  s.family.girls - if s.isBoy then 0 else 1

/-- Counts the number of brothers a sibling has -/
def brotherCount (s : Sibling) : ℕ :=
  s.family.boys - if s.isBoy then 1 else 0

theorem sibling_count_product (f : Family) (h : Sibling) (henry : Sibling)
    (henry_sisters : sisterCount henry = 4)
    (henry_brothers : brotherCount henry = 7)
    (h_family : h.family = f)
    (henry_family : henry.family = f)
    (h_girl : h.isBoy = false)
    (henry_boy : henry.isBoy = true) :
    sisterCount h * brotherCount h = 28 := by
  sorry

end sibling_count_product_l1826_182625


namespace jims_bulb_purchase_l1826_182695

theorem jims_bulb_purchase : 
  let lamp_cost : ℕ := 7
  let bulb_cost : ℕ := lamp_cost - 4
  let num_lamps : ℕ := 2
  let total_cost : ℕ := 32
  let bulbs_cost : ℕ := total_cost - (num_lamps * lamp_cost)
  ∃ (num_bulbs : ℕ), num_bulbs * bulb_cost = bulbs_cost ∧ num_bulbs = 6
  := by sorry

end jims_bulb_purchase_l1826_182695


namespace hyperbola_and_asymptotes_l1826_182663

/-- Given an ellipse and a hyperbola with the same foci, prove the equation of the hyperbola and its asymptotes -/
theorem hyperbola_and_asymptotes (x y : ℝ) : 
  (∃ (a b c : ℝ), 
    -- Ellipse equation
    (x^2 / 36 + y^2 / 27 = 1) ∧ 
    -- Hyperbola has same foci as ellipse
    (c^2 = a^2 + b^2) ∧ 
    -- Length of conjugate axis of hyperbola
    (2 * b = 4) ∧ 
    -- Foci on x-axis
    (c = 3)) →
  -- Equation of hyperbola
  (x^2 / 5 - y^2 / 4 = 1) ∧
  -- Equations of asymptotes
  (y = (2 * Real.sqrt 5 / 5) * x ∨ y = -(2 * Real.sqrt 5 / 5) * x) :=
by sorry

end hyperbola_and_asymptotes_l1826_182663


namespace corn_row_length_l1826_182667

/-- Calculates the length of a row of seeds in feet, given the space per seed and the number of seeds. -/
def row_length_in_feet (space_per_seed_inches : ℕ) (num_seeds : ℕ) : ℚ :=
  (space_per_seed_inches * num_seeds : ℚ) / 12

/-- Theorem stating that a row with 80 seeds, each requiring 18 inches of space, is 120 feet long. -/
theorem corn_row_length :
  row_length_in_feet 18 80 = 120 := by
  sorry

#eval row_length_in_feet 18 80

end corn_row_length_l1826_182667


namespace quartic_root_product_l1826_182644

theorem quartic_root_product (k : ℝ) : 
  (∃ a b c d : ℝ, 
    (a^4 - 18*a^3 + k*a^2 + 200*a - 1984 = 0) ∧
    (b^4 - 18*b^3 + k*b^2 + 200*b - 1984 = 0) ∧
    (c^4 - 18*c^3 + k*c^2 + 200*c - 1984 = 0) ∧
    (d^4 - 18*d^3 + k*d^2 + 200*d - 1984 = 0) ∧
    (a * b = -32 ∨ a * c = -32 ∨ a * d = -32 ∨ b * c = -32 ∨ b * d = -32 ∨ c * d = -32)) →
  k = 86 := by
sorry

end quartic_root_product_l1826_182644


namespace david_scott_age_difference_l1826_182655

/-- Represents the ages of three brothers -/
structure BrothersAges where
  richard : ℕ
  david : ℕ
  scott : ℕ

/-- The conditions of the problem -/
def satisfiesConditions (ages : BrothersAges) : Prop :=
  ages.richard = ages.david + 6 ∧
  ages.richard + 8 = 2 * (ages.scott + 8) ∧
  ages.david = 14

/-- The theorem to prove -/
theorem david_scott_age_difference (ages : BrothersAges) 
  (h : satisfiesConditions ages) : ages.david - ages.scott = 8 := by
  sorry

end david_scott_age_difference_l1826_182655
