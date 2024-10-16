import Mathlib

namespace NUMINAMATH_CALUDE_shaded_area_calculation_l2787_278788

/-- The area of a square with side length 12 cm, minus the area of four quarter circles 
    with radius 4 cm (one-third of the square's side length) drawn at each corner, 
    is equal to 144 - 16π cm². -/
theorem shaded_area_calculation (π : Real) : 
  let square_side : Real := 12
  let circle_radius : Real := square_side / 3
  let square_area : Real := square_side ^ 2
  let quarter_circles_area : Real := π * circle_radius ^ 2
  square_area - quarter_circles_area = 144 - 16 * π := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_calculation_l2787_278788


namespace NUMINAMATH_CALUDE_saree_pricing_l2787_278771

/-- Calculates the final price of a saree given the original price and discount options --/
def calculate_final_price (original_price : ℚ) : ℚ × ℚ × ℚ := by
  -- Define the discount options
  let option_a : ℚ := (original_price * (1 - 0.18) - 100) * (1 - 0.05) * (1 + 0.0325) + 50
  let option_b : ℚ := original_price * (1 - 0.25) * (1 + 0.0275) * (1 + 0.0175)
  let option_c : ℚ := original_price * (1 - 0.12) * (1 - 0.06) * (1 + 0.035) * (1 + 0.0225)
  
  exact (option_a, option_b, option_c)

/-- Theorem stating the final prices for each option --/
theorem saree_pricing (original_price : ℚ) :
  original_price = 1200 →
  let (price_a, price_b, price_c) := calculate_final_price original_price
  price_a = 917.09 ∧ price_b = 940.93 ∧ price_c = 1050.50 := by
  sorry

end NUMINAMATH_CALUDE_saree_pricing_l2787_278771


namespace NUMINAMATH_CALUDE_subtract_point_five_from_47_point_two_l2787_278704

theorem subtract_point_five_from_47_point_two : 47.2 - 0.5 = 46.7 := by
  sorry

end NUMINAMATH_CALUDE_subtract_point_five_from_47_point_two_l2787_278704


namespace NUMINAMATH_CALUDE_number_of_students_l2787_278741

theorem number_of_students : ∃ (x : ℕ), 
  (∃ (total : ℕ), total = 3 * x + 8) ∧ 
  (5 * (x - 1) + 3 > 3 * x + 8) ∧
  (3 * x + 8 ≥ 5 * (x - 1)) ∧
  x = 6 := by
  sorry

end NUMINAMATH_CALUDE_number_of_students_l2787_278741


namespace NUMINAMATH_CALUDE_p_or_not_q_l2787_278702

def p : Prop := ∃ α : ℝ, Real.sin (Real.pi - α) = Real.cos α

def q : Prop := ∀ m : ℝ, m > 0 → 
  (∀ x y : ℝ, x^2/m^2 - y^2/m^2 = 1 → 
    Real.sqrt (1 + (m^2/m^2)) = Real.sqrt 2) ∧
  (∃ n : ℝ, n ≤ 0 ∧ 
    (∀ x y : ℝ, x^2/n^2 - y^2/n^2 = 1 → 
      Real.sqrt (1 + (n^2/n^2)) = Real.sqrt 2))

theorem p_or_not_q : (¬p) ∨ q := by sorry

end NUMINAMATH_CALUDE_p_or_not_q_l2787_278702


namespace NUMINAMATH_CALUDE_triangle_inequality_l2787_278744

theorem triangle_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) ≤ 3 * a * b * c :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2787_278744


namespace NUMINAMATH_CALUDE_election_votes_proof_l2787_278746

theorem election_votes_proof (total_votes : ℕ) 
  (h1 : total_votes > 0)
  (h2 : (62 * total_votes) / 100 - (38 * total_votes) / 100 = 360) : 
  (62 * total_votes) / 100 = 930 := by
sorry

end NUMINAMATH_CALUDE_election_votes_proof_l2787_278746


namespace NUMINAMATH_CALUDE_males_listening_to_station_l2787_278743

theorem males_listening_to_station (total_surveyed : ℕ) (males_not_listening : ℕ) (females_listening : ℕ) (total_listeners : ℕ) (total_non_listeners : ℕ) :
  total_surveyed = total_listeners + total_non_listeners →
  males_not_listening = 98 →
  females_listening = 76 →
  total_listeners = 160 →
  total_non_listeners = 180 →
  total_surveyed - (females_listening + (total_non_listeners - males_not_listening)) - males_not_listening = 84 :=
by
  sorry

end NUMINAMATH_CALUDE_males_listening_to_station_l2787_278743


namespace NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l2787_278748

theorem greatest_divisor_with_remainders : ∃! d : ℕ,
  d > 0 ∧
  (∀ k : ℕ, k > 0 ∧ (∃ q₁ : ℕ, 13976 = k * q₁ + 23) ∧ (∃ q₂ : ℕ, 20868 = k * q₂ + 37) → k ≤ d) ∧
  (∃ q₁ : ℕ, 13976 = d * q₁ + 23) ∧
  (∃ q₂ : ℕ, 20868 = d * q₂ + 37) ∧
  d = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l2787_278748


namespace NUMINAMATH_CALUDE_gcd_of_4557_1953_5115_l2787_278723

theorem gcd_of_4557_1953_5115 : Nat.gcd 4557 (Nat.gcd 1953 5115) = 93 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_4557_1953_5115_l2787_278723


namespace NUMINAMATH_CALUDE_fraction_equality_l2787_278797

theorem fraction_equality (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (h : a / 2 = b / 3) :
  3 / b = 2 / a := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2787_278797


namespace NUMINAMATH_CALUDE_part1_part2_l2787_278726

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given condition c - b = 2b cos A -/
def satisfiesCondition (t : Triangle) : Prop :=
  t.c - t.b = 2 * t.b * Real.cos t.A

theorem part1 (t : Triangle) (h : satisfiesCondition t) 
    (ha : t.a = 2 * Real.sqrt 6) (hb : t.b = 3) : 
  t.c = 5 := by
  sorry

theorem part2 (t : Triangle) (h : satisfiesCondition t) 
    (hc : t.C = Real.pi / 2) : 
  t.B = Real.pi / 6 := by
  sorry

end NUMINAMATH_CALUDE_part1_part2_l2787_278726


namespace NUMINAMATH_CALUDE_candy_box_distribution_l2787_278795

theorem candy_box_distribution :
  ∃ (x y z : ℕ), 
    x * 16 + y * 17 + z * 21 = 185 ∧ 
    x = 5 ∧ 
    y = 0 ∧ 
    z = 5 :=
by sorry

end NUMINAMATH_CALUDE_candy_box_distribution_l2787_278795


namespace NUMINAMATH_CALUDE_divisibility_theorem_l2787_278758

theorem divisibility_theorem (n : ℕ) (a : ℝ) (h : n > 0) :
  ∃ k : ℤ, a^(2*n + 1) + (a - 1)^(n + 2) = k * (a^2 - a + 1) := by
sorry

end NUMINAMATH_CALUDE_divisibility_theorem_l2787_278758


namespace NUMINAMATH_CALUDE_add_12345_seconds_to_5_15_00_l2787_278782

/-- Represents time in hours, minutes, and seconds -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat
  deriving Repr

/-- Adds seconds to a given time -/
def addSeconds (t : Time) (s : Nat) : Time :=
  sorry

theorem add_12345_seconds_to_5_15_00 :
  addSeconds (Time.mk 5 15 0) 12345 = Time.mk 9 0 45 := by
  sorry

end NUMINAMATH_CALUDE_add_12345_seconds_to_5_15_00_l2787_278782


namespace NUMINAMATH_CALUDE_boys_playing_marbles_l2787_278780

theorem boys_playing_marbles (total_marbles : ℕ) (marbles_per_boy : ℕ) (h1 : total_marbles = 35) (h2 : marbles_per_boy = 7) :
  total_marbles / marbles_per_boy = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_boys_playing_marbles_l2787_278780


namespace NUMINAMATH_CALUDE_infinite_triples_sum_of_squares_l2787_278750

/-- A number that can be expressed as the sum of one or two squares. -/
def IsSumOfTwoSquares (k : ℤ) : Prop :=
  ∃ a b : ℤ, k = a^2 + b^2

theorem infinite_triples_sum_of_squares (n : ℤ) :
  let N := 2 * n^2 * (n + 1)^2
  IsSumOfTwoSquares N ∧
  IsSumOfTwoSquares (N + 1) ∧
  IsSumOfTwoSquares (N + 2) := by
  sorry


end NUMINAMATH_CALUDE_infinite_triples_sum_of_squares_l2787_278750


namespace NUMINAMATH_CALUDE_rectangle_on_W_perimeter_l2787_278711

/-- The locus W is defined by the equation y = x^2 + 1/4 -/
def W : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1^2 + 1/4}

/-- A rectangle is defined by its four vertices -/
structure Rectangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- The perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ :=
  2 * (dist r.A r.B + dist r.B r.C)

/-- Theorem: Any rectangle with three vertices on W has perimeter greater than 3√3 -/
theorem rectangle_on_W_perimeter (r : Rectangle) 
  (h1 : r.A ∈ W) (h2 : r.B ∈ W) (h3 : r.C ∈ W ∨ r.D ∈ W) : 
  perimeter r > 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_on_W_perimeter_l2787_278711


namespace NUMINAMATH_CALUDE_merchant_pricing_strategy_l2787_278785

theorem merchant_pricing_strategy 
  (list_price : ℝ) 
  (purchase_price_ratio : ℝ) 
  (discount_ratio : ℝ) 
  (profit_ratio : ℝ) 
  (marked_price_ratio : ℝ) 
  (h1 : purchase_price_ratio = 0.7) 
  (h2 : discount_ratio = 0.25) 
  (h3 : profit_ratio = 0.3) 
  (h4 : marked_price_ratio * (1 - discount_ratio) * list_price - 
        purchase_price_ratio * list_price = 
        profit_ratio * marked_price_ratio * (1 - discount_ratio) * list_price) :
  marked_price_ratio = 1.33 := by
  sorry

#check merchant_pricing_strategy

end NUMINAMATH_CALUDE_merchant_pricing_strategy_l2787_278785


namespace NUMINAMATH_CALUDE_least_number_for_divisibility_l2787_278772

theorem least_number_for_divisibility (n m : ℕ) (h : n = 1056 ∧ m = 27) :
  ∃ x : ℕ, (n + x) % m = 0 ∧ ∀ y : ℕ, y < x → (n + y) % m ≠ 0 ∧ x = 24 :=
sorry

end NUMINAMATH_CALUDE_least_number_for_divisibility_l2787_278772


namespace NUMINAMATH_CALUDE_company_average_service_l2787_278790

/-- Represents a department in the company -/
structure Department where
  employees : ℕ
  total_service : ℕ

/-- The company with two departments -/
structure Company where
  dept_a : Department
  dept_b : Department

/-- Average years of service for a department -/
def avg_service (d : Department) : ℚ :=
  d.total_service / d.employees

/-- Average years of service for the entire company -/
def company_avg_service (c : Company) : ℚ :=
  (c.dept_a.total_service + c.dept_b.total_service) / (c.dept_a.employees + c.dept_b.employees)

theorem company_average_service (k : ℕ) (h_k : k > 0) :
  let c : Company := {
    dept_a := { employees := 7 * k, total_service := 56 * k },
    dept_b := { employees := 5 * k, total_service := 30 * k }
  }
  avg_service c.dept_a = 8 ∧
  avg_service c.dept_b = 6 ∧
  company_avg_service c = 7 + 1/6 :=
by sorry

end NUMINAMATH_CALUDE_company_average_service_l2787_278790


namespace NUMINAMATH_CALUDE_polygon_diagonal_division_l2787_278786

/-- 
For an n-sided polygon, if a diagonal drawn from a vertex can divide it into 
at most 2023 triangles, then n = 2025.
-/
theorem polygon_diagonal_division (n : ℕ) : 
  (∃ (d : ℕ), d ≤ 2023 ∧ d = n - 2) → n = 2025 := by
  sorry

end NUMINAMATH_CALUDE_polygon_diagonal_division_l2787_278786


namespace NUMINAMATH_CALUDE_convergence_bound_minimal_k_smallest_k_is_five_l2787_278794

def v : ℕ → ℚ
  | 0 => 1/8
  | n + 1 => 3 * v n - 3 * (v n)^2

def M : ℚ := 1/2

theorem convergence_bound (k : ℕ) : k ≥ 5 → |v k - M| ≤ 1/2^500 := by sorry

theorem minimal_k : ∀ j : ℕ, j < 5 → |v j - M| > 1/2^500 := by sorry

theorem smallest_k_is_five : 
  (∃ k : ℕ, |v k - M| ≤ 1/2^500) ∧ 
  (∀ j : ℕ, |v j - M| ≤ 1/2^500 → j ≥ 5) := by sorry

end NUMINAMATH_CALUDE_convergence_bound_minimal_k_smallest_k_is_five_l2787_278794


namespace NUMINAMATH_CALUDE_f_is_quadratic_l2787_278742

-- Define what it means for an equation to be quadratic in x
def is_quadratic_in_x (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

-- Define the function representing the equation x^2 + x = 4
def f (x : ℝ) : ℝ := x^2 + x - 4

-- Theorem stating that f is quadratic in x
theorem f_is_quadratic : is_quadratic_in_x f := by
  sorry


end NUMINAMATH_CALUDE_f_is_quadratic_l2787_278742


namespace NUMINAMATH_CALUDE_curve_self_intersection_l2787_278707

/-- A point on the curve defined by t --/
def curve_point (t : ℝ) : ℝ × ℝ :=
  (t^2 - 4, t^3 - 6*t + 7)

/-- The curve crosses itself if there exist two distinct real numbers that map to the same point --/
def self_intersection (p : ℝ × ℝ) : Prop :=
  ∃ a b : ℝ, a ≠ b ∧ curve_point a = p ∧ curve_point b = p

theorem curve_self_intersection :
  self_intersection (2, 7) := by
  sorry

end NUMINAMATH_CALUDE_curve_self_intersection_l2787_278707


namespace NUMINAMATH_CALUDE_profit_and_max_profit_l2787_278751

/-- Represents the average daily profit as a function of price reduction --/
def averageDailyProfit (x : ℝ) : ℝ := -2 * x^2 + 60 * x + 800

/-- The price reduction that results in $1200 average daily profit --/
def priceReductionFor1200Profit : ℝ := 20

/-- The price reduction that maximizes average daily profit --/
def priceReductionForMaxProfit : ℝ := 15

/-- The maximum average daily profit --/
def maxAverageDailyProfit : ℝ := 1250

theorem profit_and_max_profit :
  (averageDailyProfit priceReductionFor1200Profit = 1200) ∧
  (∀ x : ℝ, averageDailyProfit x ≤ maxAverageDailyProfit) ∧
  (averageDailyProfit priceReductionForMaxProfit = maxAverageDailyProfit) := by
  sorry


end NUMINAMATH_CALUDE_profit_and_max_profit_l2787_278751


namespace NUMINAMATH_CALUDE_incorrect_exponent_operation_l2787_278732

theorem incorrect_exponent_operation (a : ℝ) (h : a ≠ 0 ∧ a ≠ 1) : (a^2)^3 ≠ a^5 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_exponent_operation_l2787_278732


namespace NUMINAMATH_CALUDE_characterization_of_special_numbers_l2787_278731

def is_power_of_two (n : ℕ+) : Prop :=
  ∃ k : ℕ, n = 2^k

def greatest_odd_divisor (n : ℕ+) : ℕ+ :=
  sorry

def smallest_odd_divisor (n : ℕ+) : ℕ+ :=
  sorry

def is_odd_prime (p : ℕ+) : Prop :=
  Nat.Prime p.val ∧ p.val % 2 = 1

theorem characterization_of_special_numbers (n : ℕ+) :
  ¬is_power_of_two n →
  (n = 3 * greatest_odd_divisor n + 5 * smallest_odd_divisor n ↔
    (∃ p : ℕ+, is_odd_prime p ∧ n = 8 * p) ∨ n = 60 ∨ n = 100) :=
  sorry

end NUMINAMATH_CALUDE_characterization_of_special_numbers_l2787_278731


namespace NUMINAMATH_CALUDE_inequality_solution_l2787_278727

noncomputable section

variables (a x : ℝ)

def inequality := (a * (x - 1)) / (x - 2) > 1

def solution : Prop :=
  (0 < a ∧ a < 1 → 2 < x ∧ x < (a - 2) / (a - 1)) ∧
  (a = 1 → x > 2) ∧
  (a > 1 → x > 2 ∨ x < (a - 2) / (a - 1))

theorem inequality_solution (h : a > 0) : inequality a x ↔ solution a x := by sorry

end

end NUMINAMATH_CALUDE_inequality_solution_l2787_278727


namespace NUMINAMATH_CALUDE_factor_polynomial_l2787_278798

theorem factor_polynomial (x y : ℝ) : 
  66 * x^5 - 165 * x^9 + 99 * x^5 * y = 33 * x^5 * (2 - 5 * x^4 + 3 * y) := by
  sorry

end NUMINAMATH_CALUDE_factor_polynomial_l2787_278798


namespace NUMINAMATH_CALUDE_positive_number_squared_plus_self_l2787_278737

theorem positive_number_squared_plus_self (n : ℝ) : n > 0 ∧ n^2 + n = 210 → n = 14 := by
  sorry

end NUMINAMATH_CALUDE_positive_number_squared_plus_self_l2787_278737


namespace NUMINAMATH_CALUDE_exam_students_count_l2787_278796

/-- Proves that given the conditions of the exam results, the total number of students is 14 -/
theorem exam_students_count (total_average : ℝ) (excluded_count : ℕ) (excluded_average : ℝ) (remaining_average : ℝ)
  (h1 : total_average = 65)
  (h2 : excluded_count = 5)
  (h3 : excluded_average = 20)
  (h4 : remaining_average = 90) :
  ∃ (n : ℕ), n = 14 ∧ 
    (n : ℝ) * total_average = 
      ((n - excluded_count) : ℝ) * remaining_average + (excluded_count : ℝ) * excluded_average :=
by sorry

end NUMINAMATH_CALUDE_exam_students_count_l2787_278796


namespace NUMINAMATH_CALUDE_plot_area_in_acres_l2787_278774

/-- Conversion factor from square miles to acres -/
def miles_to_acres : ℝ := 640

/-- Length of the plot in miles -/
def length : ℝ := 12

/-- Width of the plot in miles -/
def width : ℝ := 8

/-- Theorem stating that the area of the rectangular plot in acres is 61440 -/
theorem plot_area_in_acres :
  length * width * miles_to_acres = 61440 := by sorry

end NUMINAMATH_CALUDE_plot_area_in_acres_l2787_278774


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l2787_278713

theorem polynomial_division_remainder (k : ℚ) : 
  (∃! k, ∀ x, (3 * x^3 + k * x^2 + 5 * x - 8) % (3 * x + 4) = 10) ↔ k = 31/4 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l2787_278713


namespace NUMINAMATH_CALUDE_line_slope_l2787_278717

theorem line_slope (x y : ℝ) :
  x + Real.sqrt 3 * y + 2 = 0 →
  (y - (-2 / Real.sqrt 3)) / (x - 0) = - Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_line_slope_l2787_278717


namespace NUMINAMATH_CALUDE_toms_game_sale_l2787_278777

/-- Calculates the sale amount of games given initial cost, value increase factor, and sale percentage -/
def gameSaleAmount (initialCost : ℝ) (valueIncreaseFactor : ℝ) (salePercentage : ℝ) : ℝ :=
  initialCost * valueIncreaseFactor * salePercentage

/-- Proves that Tom's game sale amount is $240 given the specified conditions -/
theorem toms_game_sale : gameSaleAmount 200 3 0.4 = 240 := by
  sorry

end NUMINAMATH_CALUDE_toms_game_sale_l2787_278777


namespace NUMINAMATH_CALUDE_min_value_expression_l2787_278754

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 27) :
  x^2 + 6*x*y + 9*y^2 + (3/2)*z^2 ≥ 102 ∧
  ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ x₀ * y₀ * z₀ = 27 ∧
    x₀^2 + 6*x₀*y₀ + 9*y₀^2 + (3/2)*z₀^2 = 102 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2787_278754


namespace NUMINAMATH_CALUDE_average_exists_l2787_278709

theorem average_exists : ∃ N : ℝ, 13 < N ∧ N < 21 ∧ (8 + 12 + N) / 3 = 12 := by
  sorry

end NUMINAMATH_CALUDE_average_exists_l2787_278709


namespace NUMINAMATH_CALUDE_triangle_area_l2787_278703

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.sin x * Real.cos x + 2 * (Real.cos x)^2

theorem triangle_area (a b c A B C : ℝ) : 
  a = 2 * Real.sqrt 3 →
  f A = 2 →
  b + c = 6 →
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  a * Real.sin B = b * Real.sin A →
  a * Real.sin C = c * Real.sin A →
  (1/2) * b * c * Real.sin A = 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l2787_278703


namespace NUMINAMATH_CALUDE_both_miss_probability_l2787_278793

/-- The probability that both shooters miss the target given their individual hit probabilities -/
theorem both_miss_probability (p_hit_A p_hit_B : ℝ) (h_A : p_hit_A = 0.85) (h_B : p_hit_B = 0.8) :
  (1 - p_hit_A) * (1 - p_hit_B) = 0.03 := by
  sorry

end NUMINAMATH_CALUDE_both_miss_probability_l2787_278793


namespace NUMINAMATH_CALUDE_shirt_cost_relationship_l2787_278769

/-- Represents the relationship between two types of shirts and their costs. -/
theorem shirt_cost_relationship (x : ℝ) 
  (h1 : x > 0)  -- Ensure x is positive (number of shirts can't be negative or zero)
  (h2 : 1.5 * x > 0)  -- Ensure 1.5x is positive
  : 7800 / (1.5 * x) + 30 = 6400 / x := by
  sorry

#check shirt_cost_relationship

end NUMINAMATH_CALUDE_shirt_cost_relationship_l2787_278769


namespace NUMINAMATH_CALUDE_alcohol_mixture_concentration_l2787_278718

-- Define the concentrations and volumes
def x_concentration : ℝ := 0.10
def y_concentration : ℝ := 0.30
def target_concentration : ℝ := 0.22
def x_volume : ℝ := 300
def y_volume : ℝ := 450

-- Theorem statement
theorem alcohol_mixture_concentration :
  (x_concentration * x_volume + y_concentration * y_volume) / (x_volume + y_volume) = target_concentration := by
  sorry

end NUMINAMATH_CALUDE_alcohol_mixture_concentration_l2787_278718


namespace NUMINAMATH_CALUDE_prime_dividing_polynomial_congruence_l2787_278764

theorem prime_dividing_polynomial_congruence (n : ℕ) (p : ℕ) (hn : n > 0) (hp : Nat.Prime p) :
  p ∣ (5^(4*n) - 5^(3*n) + 5^(2*n) - 5^n + 1) → p % 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_prime_dividing_polynomial_congruence_l2787_278764


namespace NUMINAMATH_CALUDE_abs_sum_lt_abs_diff_for_opposite_signs_l2787_278734

theorem abs_sum_lt_abs_diff_for_opposite_signs (a b : ℝ) (h : a * b < 0) :
  |a + b| < |a - b| := by sorry

end NUMINAMATH_CALUDE_abs_sum_lt_abs_diff_for_opposite_signs_l2787_278734


namespace NUMINAMATH_CALUDE_consecutive_numbers_sum_l2787_278779

theorem consecutive_numbers_sum (n : ℕ) :
  (n + 1) + (n + 2) + (n + 3) = 2 * (n + (n - 1) + (n - 2)) →
  n + 3 = 7 ∧ (n - 2) + (n - 1) + n + (n + 1) + (n + 2) + (n + 3) = 27 :=
by sorry

end NUMINAMATH_CALUDE_consecutive_numbers_sum_l2787_278779


namespace NUMINAMATH_CALUDE_barycentric_coords_exist_and_unique_l2787_278712

-- Define a triangle in 2D space
structure Triangle where
  A₁ : ℝ × ℝ
  A₂ : ℝ × ℝ
  A₃ : ℝ × ℝ

-- Define barycentric coordinates
structure BarycentricCoords where
  m₁ : ℝ
  m₂ : ℝ
  m₃ : ℝ

-- Define a point in 2D space
def Point := ℝ × ℝ

-- State the theorem
theorem barycentric_coords_exist_and_unique (t : Triangle) (X : Point) :
  ∃! (b : BarycentricCoords),
    b.m₁ + b.m₂ + b.m₃ = 1 ∧
    X = (b.m₁ * t.A₁.1 + b.m₂ * t.A₂.1 + b.m₃ * t.A₃.1,
         b.m₁ * t.A₁.2 + b.m₂ * t.A₂.2 + b.m₃ * t.A₃.2) :=
  sorry

end NUMINAMATH_CALUDE_barycentric_coords_exist_and_unique_l2787_278712


namespace NUMINAMATH_CALUDE_arccos_equation_solution_l2787_278729

theorem arccos_equation_solution (x : ℝ) : 
  Real.arccos (2 * x - 1) = π / 4 → x = (Real.sqrt 2 + 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_arccos_equation_solution_l2787_278729


namespace NUMINAMATH_CALUDE_m_range_l2787_278767

/-- Proposition p: m is a real number and m + 1 ≤ 0 -/
def p (m : ℝ) : Prop := m + 1 ≤ 0

/-- Proposition q: For all real x, x² + mx + 1 > 0 -/
def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 + m*x + 1 > 0

/-- The range of m satisfying the given conditions -/
theorem m_range (m : ℝ) : 
  (p m ∧ q m → False) →  -- p ∧ q is false
  (p m ∨ q m) →          -- p ∨ q is true
  (m ≤ -2 ∨ (-1 < m ∧ m < 2)) := by
sorry

end NUMINAMATH_CALUDE_m_range_l2787_278767


namespace NUMINAMATH_CALUDE_strawberries_per_basket_is_15_l2787_278783

/-- The number of strawberries in each basket picked by Kimberly's brother -/
def strawberries_per_basket (kimberly_amount : ℕ) (brother_baskets : ℕ) (parents_amount : ℕ) (total_amount : ℕ) : ℕ :=
  (total_amount / 4) / brother_baskets

/-- Theorem stating the number of strawberries in each basket picked by Kimberly's brother -/
theorem strawberries_per_basket_is_15 
  (kimberly_amount : ℕ) 
  (brother_baskets : ℕ) 
  (parents_amount : ℕ) 
  (total_amount : ℕ) 
  (h1 : kimberly_amount = 8 * (brother_baskets * strawberries_per_basket kimberly_amount brother_baskets parents_amount total_amount))
  (h2 : parents_amount = kimberly_amount - 93)
  (h3 : brother_baskets = 3)
  (h4 : total_amount = 4 * 168)
  : strawberries_per_basket kimberly_amount brother_baskets parents_amount total_amount = 15 :=
sorry


end NUMINAMATH_CALUDE_strawberries_per_basket_is_15_l2787_278783


namespace NUMINAMATH_CALUDE_oranges_picked_l2787_278768

theorem oranges_picked (michaela_full : ℕ) (cassandra_full : ℕ) (remaining : ℕ) : 
  michaela_full = 20 → 
  cassandra_full = 2 * michaela_full → 
  remaining = 30 → 
  michaela_full + cassandra_full + remaining = 90 := by
  sorry

end NUMINAMATH_CALUDE_oranges_picked_l2787_278768


namespace NUMINAMATH_CALUDE_base_ten_to_five_235_l2787_278714

/-- Converts a number from base 10 to base 5 -/
def toBaseFive (n : ℕ) : List ℕ :=
  sorry

theorem base_ten_to_five_235 :
  toBaseFive 235 = [1, 4, 2, 0] :=
sorry

end NUMINAMATH_CALUDE_base_ten_to_five_235_l2787_278714


namespace NUMINAMATH_CALUDE_base_number_proof_l2787_278701

theorem base_number_proof (x y : ℝ) (h1 : x ^ y = 3 ^ 16) (h2 : y = 8) : x = 9 := by
  sorry

end NUMINAMATH_CALUDE_base_number_proof_l2787_278701


namespace NUMINAMATH_CALUDE_linear_function_y_axis_intersection_l2787_278735

/-- The coordinates of the intersection point of y = (1/2)x + 1 with the y-axis -/
theorem linear_function_y_axis_intersection :
  let f : ℝ → ℝ := λ x ↦ (1/2) * x + 1
  ∃! p : ℝ × ℝ, p.1 = 0 ∧ p.2 = f p.1 ∧ p = (0, 1) := by
  sorry

end NUMINAMATH_CALUDE_linear_function_y_axis_intersection_l2787_278735


namespace NUMINAMATH_CALUDE_circle_center_sum_l2787_278775

theorem circle_center_sum (x y : ℝ) : 
  x^2 + y^2 = 4*x + 10*y - 12 → x + y = 7 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_sum_l2787_278775


namespace NUMINAMATH_CALUDE_regular_polygon_108_degrees_has_5_sides_l2787_278705

/-- A regular polygon with interior angles measuring 108 degrees has 5 sides. -/
theorem regular_polygon_108_degrees_has_5_sides :
  ∀ n : ℕ,
  n ≥ 3 →
  (180 * (n - 2) : ℝ) = (108 * n : ℝ) →
  n = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_108_degrees_has_5_sides_l2787_278705


namespace NUMINAMATH_CALUDE_impossible_to_use_all_stock_l2787_278791

/-- Represents the number of units required for each product type -/
structure ProductRequirements where
  alpha_A : Nat
  alpha_B : Nat
  beta_B : Nat
  beta_C : Nat
  gamma_A : Nat
  gamma_C : Nat

/-- Represents the current stock levels after production -/
structure StockLevels where
  remaining_A : Nat
  remaining_B : Nat
  remaining_C : Nat

/-- Theorem stating the impossibility of using up all stocks exactly -/
theorem impossible_to_use_all_stock 
  (req : ProductRequirements)
  (stock : StockLevels)
  (h_req : req = { 
    alpha_A := 2, alpha_B := 2, 
    beta_B := 1, beta_C := 1, 
    gamma_A := 2, gamma_C := 1 
  })
  (h_stock : stock = { remaining_A := 2, remaining_B := 1, remaining_C := 0 }) :
  ∀ (p q r : Nat), ∃ (total_A total_B total_C : Nat),
    (2 * p + 2 * r + stock.remaining_A ≠ total_A) ∨
    (2 * p + q + stock.remaining_B ≠ total_B) ∨
    (q + r ≠ total_C) :=
sorry

end NUMINAMATH_CALUDE_impossible_to_use_all_stock_l2787_278791


namespace NUMINAMATH_CALUDE_unique_solution_iff_a_nonpositive_l2787_278770

/-- The system of equations has at most one real solution if and only if a ≤ 0 -/
theorem unique_solution_iff_a_nonpositive (a : ℝ) :
  (∃! x y z : ℝ, x^4 = y*z - x^2 + a ∧ y^4 = z*x - y^2 + a ∧ z^4 = x*y - z^2 + a) ↔ a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_iff_a_nonpositive_l2787_278770


namespace NUMINAMATH_CALUDE_average_people_per_hour_rounded_l2787_278757

/-- The number of people moving to Alaska in 5 days -/
def total_people : ℕ := 4000

/-- The number of days -/
def num_days : ℕ := 5

/-- The number of hours in a day -/
def hours_per_day : ℕ := 24

/-- Calculate the average number of people moving to Alaska per hour -/
def average_per_hour : ℚ :=
  total_people / (num_days * hours_per_day)

/-- Round a rational number to the nearest integer -/
def round_to_nearest (x : ℚ) : ℤ :=
  ⌊x + 1/2⌋

theorem average_people_per_hour_rounded : 
  round_to_nearest average_per_hour = 33 := by
  sorry

end NUMINAMATH_CALUDE_average_people_per_hour_rounded_l2787_278757


namespace NUMINAMATH_CALUDE_triangle_isosceles_l2787_278745

theorem triangle_isosceles (A B C : ℝ) (h : 2 * Real.sin A * Real.cos B = Real.sin C) : A = B :=
sorry

end NUMINAMATH_CALUDE_triangle_isosceles_l2787_278745


namespace NUMINAMATH_CALUDE_polynomial_roots_magnitude_l2787_278716

theorem polynomial_roots_magnitude (c : ℂ) : 
  let p : ℂ → ℂ := λ x => (x^2 - 2*x + 2) * (x^2 - c*x + 4) * (x^2 - 4*x + 8)
  (∃ (s : Finset ℂ), s.card = 4 ∧ (∀ z ∈ s, p z = 0) ∧ (∀ z, p z = 0 → z ∈ s)) →
  Complex.abs c = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_roots_magnitude_l2787_278716


namespace NUMINAMATH_CALUDE_rows_of_nine_l2787_278715

/-- Given 74 people seated in rows of either 7 or 9 seats, with all seats occupied,
    there are exactly 2 rows seating 9 people. -/
theorem rows_of_nine (total_people : ℕ) (rows_of_seven : ℕ) (rows_of_nine : ℕ) : 
  total_people = 74 →
  total_people = 7 * rows_of_seven + 9 * rows_of_nine →
  rows_of_nine = 2 := by
  sorry

end NUMINAMATH_CALUDE_rows_of_nine_l2787_278715


namespace NUMINAMATH_CALUDE_exponential_plus_x_increasing_l2787_278765

open Real

theorem exponential_plus_x_increasing (x : ℝ) : exp (x + 1) + (x + 1) > (exp x + x) + 1 := by
  sorry

end NUMINAMATH_CALUDE_exponential_plus_x_increasing_l2787_278765


namespace NUMINAMATH_CALUDE_train_speed_fraction_l2787_278700

/-- Given a train journey where:
  1. The train reached its destination in 8 hours at a certain fraction of its own speed.
  2. If the train had run at its full speed, it would have taken 4 hours less.
  This theorem proves that the fraction of the train's own speed at which it was running is 1/2. -/
theorem train_speed_fraction (full_speed : ℝ) (fraction : ℝ) 
  (h1 : fraction * full_speed * 8 = full_speed * 4) : fraction = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_fraction_l2787_278700


namespace NUMINAMATH_CALUDE_triangle_3_4_6_l2787_278789

/-- A function that checks if three line segments can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Theorem stating that the line segments 3, 4, and 6 can form a triangle -/
theorem triangle_3_4_6 : can_form_triangle 3 4 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_3_4_6_l2787_278789


namespace NUMINAMATH_CALUDE_lt_iff_forall_add_lt_l2787_278755

theorem lt_iff_forall_add_lt (a b : ℝ) : a < b ↔ ∀ x ∈ Set.Ioo 0 1, a + x < b := by sorry

end NUMINAMATH_CALUDE_lt_iff_forall_add_lt_l2787_278755


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l2787_278749

theorem simplify_and_rationalize : 
  (Real.sqrt 5 / Real.sqrt 6) * (Real.sqrt 10 / Real.sqrt 15) * (Real.sqrt 12 / Real.sqrt 20) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l2787_278749


namespace NUMINAMATH_CALUDE_x_eq_2_sufficient_not_necessary_l2787_278728

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 * b.2 = k * a.2 * b.1

/-- The statement that x = 2 is sufficient but not necessary for a ∥ b -/
theorem x_eq_2_sufficient_not_necessary (x : ℝ) :
  (∀ x, x = 2 → are_parallel (1, x - 1) (x + 1, 3)) ∧
  (∃ x, x ≠ 2 ∧ are_parallel (1, x - 1) (x + 1, 3)) := by
  sorry

end NUMINAMATH_CALUDE_x_eq_2_sufficient_not_necessary_l2787_278728


namespace NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l2787_278740

/-- Given a geometric sequence where the first term is 512 and the 8th term is 2,
    prove that the 6th term is 16. -/
theorem geometric_sequence_sixth_term
  (a : ℝ) -- First term
  (r : ℝ) -- Common ratio
  (h1 : a = 512) -- First term is 512
  (h2 : a * r^7 = 2) -- 8th term is 2
  : a * r^5 = 16 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l2787_278740


namespace NUMINAMATH_CALUDE_only_B_is_random_event_l2787_278733

-- Define the type for a die roll
def DieRoll := Fin 6

-- Define the type for a pair of die rolls
def TwoDiceRoll := DieRoll × DieRoll

-- Define the sum of two dice
def diceSum (roll : TwoDiceRoll) : Nat := roll.1.val + roll.2.val + 2

-- Define the sample space
def Ω : Set TwoDiceRoll := Set.univ

-- Define the events
def A : Set TwoDiceRoll := {roll | diceSum roll = 1}
def B : Set TwoDiceRoll := {roll | diceSum roll = 6}
def C : Set TwoDiceRoll := {roll | diceSum roll > 12}
def D : Set TwoDiceRoll := {roll | diceSum roll < 13}

-- Theorem statement
theorem only_B_is_random_event :
  (A = ∅ ∧ B ≠ ∅ ∧ B ≠ Ω ∧ C = ∅ ∧ D = Ω) := by sorry

end NUMINAMATH_CALUDE_only_B_is_random_event_l2787_278733


namespace NUMINAMATH_CALUDE_quadratic_root_problem_l2787_278759

theorem quadratic_root_problem (m : ℝ) :
  (2 : ℝ)^2 + 2 + m = 0 → ∃ (x : ℝ), x^2 + x + m = 0 ∧ x ≠ 2 ∧ x = -3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_problem_l2787_278759


namespace NUMINAMATH_CALUDE_gcd_840_1764_l2787_278763

theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := by
  sorry

end NUMINAMATH_CALUDE_gcd_840_1764_l2787_278763


namespace NUMINAMATH_CALUDE_evaluate_expression_l2787_278720

theorem evaluate_expression : -(14 / 2 * 9 - 60 + 3 * 9) = -30 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2787_278720


namespace NUMINAMATH_CALUDE_circle_equations_l2787_278724

-- Define the circle N
def circle_N (x y : ℝ) : Prop := (x - 2)^2 + (y - 4)^2 = 10

-- Define the symmetric circle
def symmetric_circle (x y : ℝ) : Prop := (x - 1)^2 + (y - 5)^2 = 10

-- Define the trajectory of midpoint M
def trajectory_M (x y : ℝ) : Prop := (x - 5/2)^2 + (y - 2)^2 = 5/2

-- Define points A, B, and C
def point_A : ℝ × ℝ := (3, 1)
def point_B : ℝ × ℝ := (-1, 3)
def point_C : ℝ × ℝ := (3, 0)

-- Define the line that contains the center of circle N
def center_line (x y : ℝ) : Prop := 3*x - y - 2 = 0

-- Define a point D on circle N
def point_D (x y : ℝ) : Prop := circle_N x y

-- Define the midpoint M of segment CD
def midpoint_M (x y x_D y_D : ℝ) : Prop := x = (x_D + 3)/2 ∧ y = y_D/2

theorem circle_equations :
  (∀ x y, circle_N x y ↔ (x - 2)^2 + (y - 4)^2 = 10) ∧
  (∀ x y, symmetric_circle x y ↔ (x - 1)^2 + (y - 5)^2 = 10) ∧
  (∀ x y, (∃ x_D y_D, point_D x_D y_D ∧ midpoint_M x y x_D y_D) → trajectory_M x y) :=
sorry

end NUMINAMATH_CALUDE_circle_equations_l2787_278724


namespace NUMINAMATH_CALUDE_chickens_bought_l2787_278792

def eggCount : ℕ := 20
def eggPrice : ℕ := 2
def chickenPrice : ℕ := 8
def totalSpent : ℕ := 88

theorem chickens_bought :
  (totalSpent - eggCount * eggPrice) / chickenPrice = 6 := by sorry

end NUMINAMATH_CALUDE_chickens_bought_l2787_278792


namespace NUMINAMATH_CALUDE_reliable_plumbing_hourly_charge_l2787_278784

/-- Paul's Plumbing visit charge -/
def paul_visit : ℕ := 55

/-- Paul's Plumbing hourly labor charge -/
def paul_hourly : ℕ := 35

/-- Reliable Plumbing visit charge -/
def reliable_visit : ℕ := 75

/-- Number of labor hours -/
def labor_hours : ℕ := 4

/-- Reliable Plumbing's hourly labor charge -/
def reliable_hourly : ℕ := 30

theorem reliable_plumbing_hourly_charge :
  paul_visit + labor_hours * paul_hourly = reliable_visit + labor_hours * reliable_hourly :=
by sorry

end NUMINAMATH_CALUDE_reliable_plumbing_hourly_charge_l2787_278784


namespace NUMINAMATH_CALUDE_exists_m_composite_l2787_278778

theorem exists_m_composite (n : ℕ) : ∃ m : ℕ, ∃ k : ℕ, k > 1 ∧ k < n * m + 1 ∧ (n * m + 1) % k = 0 := by
  sorry

end NUMINAMATH_CALUDE_exists_m_composite_l2787_278778


namespace NUMINAMATH_CALUDE_equation_represents_hyperbola_l2787_278761

/-- The equation |y-3| = √((x+4)² + 4y²) represents a hyperbola -/
theorem equation_represents_hyperbola :
  ∃ (x y : ℝ), |y - 3| = Real.sqrt ((x + 4)^2 + 4*y^2) →
  ∃ (A B C D E : ℝ), A ≠ 0 ∧ C ≠ 0 ∧ A * C < 0 ∧
    A * y^2 + B * y + C * x^2 + D * x + E = 0 :=
by sorry

end NUMINAMATH_CALUDE_equation_represents_hyperbola_l2787_278761


namespace NUMINAMATH_CALUDE_not_all_tetrahedra_altitudes_intersect_l2787_278756

/-- A tetrahedron is represented by its four vertices in 3D space -/
def Tetrahedron := Fin 4 → ℝ × ℝ × ℝ

/-- An altitude of a tetrahedron is a line segment from a vertex perpendicular to the opposite face -/
def Altitude (t : Tetrahedron) (v : Fin 4) : Set (ℝ × ℝ × ℝ) :=
  sorry

/-- Predicate to check if all altitudes of a tetrahedron intersect at a single point -/
def altitudesIntersectAtPoint (t : Tetrahedron) : Prop :=
  ∃ p : ℝ × ℝ × ℝ, ∀ v : Fin 4, p ∈ Altitude t v

/-- Theorem stating that not all tetrahedra have altitudes intersecting at a single point -/
theorem not_all_tetrahedra_altitudes_intersect :
  ∃ t : Tetrahedron, ¬ altitudesIntersectAtPoint t :=
sorry

end NUMINAMATH_CALUDE_not_all_tetrahedra_altitudes_intersect_l2787_278756


namespace NUMINAMATH_CALUDE_wang_shifu_not_yuan_dramatist_l2787_278747

/-- The set of four great dramatists of the Yuan Dynasty -/
def YuanDramatists : Set String :=
  {"Guan Hanqing", "Zheng Guangzu", "Bai Pu", "Ma Zhiyuan"}

/-- Wang Shifu -/
def WangShifu : String := "Wang Shifu"

/-- Theorem stating that Wang Shifu is not one of the four great dramatists of the Yuan Dynasty -/
theorem wang_shifu_not_yuan_dramatist :
  WangShifu ∉ YuanDramatists := by
  sorry

end NUMINAMATH_CALUDE_wang_shifu_not_yuan_dramatist_l2787_278747


namespace NUMINAMATH_CALUDE_original_price_per_acre_l2787_278760

/-- Proves that the original price per acre was $140 --/
theorem original_price_per_acre 
  (total_area : ℕ)
  (sold_area : ℕ)
  (selling_price : ℕ)
  (profit : ℕ)
  (h1 : total_area = 200)
  (h2 : sold_area = total_area / 2)
  (h3 : selling_price = 200)
  (h4 : profit = 6000)
  : (selling_price * sold_area - profit) / sold_area = 140 := by
  sorry

end NUMINAMATH_CALUDE_original_price_per_acre_l2787_278760


namespace NUMINAMATH_CALUDE_sean_sunday_spending_l2787_278721

/-- Represents Sean's Sunday purchases and their costs --/
structure SundayPurchases where
  almond_croissant_price : ℝ
  salami_cheese_croissant_price : ℝ
  plain_croissant_price : ℝ
  focaccia_price : ℝ
  latte_price : ℝ
  almond_croissant_quantity : ℕ
  salami_cheese_croissant_quantity : ℕ
  plain_croissant_quantity : ℕ
  focaccia_quantity : ℕ
  latte_quantity : ℕ

/-- Calculates the total cost of Sean's Sunday purchases --/
def total_cost (purchases : SundayPurchases) : ℝ :=
  purchases.almond_croissant_price * purchases.almond_croissant_quantity +
  purchases.salami_cheese_croissant_price * purchases.salami_cheese_croissant_quantity +
  purchases.plain_croissant_price * purchases.plain_croissant_quantity +
  purchases.focaccia_price * purchases.focaccia_quantity +
  purchases.latte_price * purchases.latte_quantity

/-- Theorem stating that Sean's total spending on Sunday is $21.00 --/
theorem sean_sunday_spending (purchases : SundayPurchases)
  (h1 : purchases.almond_croissant_price = 4.5)
  (h2 : purchases.salami_cheese_croissant_price = 4.5)
  (h3 : purchases.plain_croissant_price = 3)
  (h4 : purchases.focaccia_price = 4)
  (h5 : purchases.latte_price = 2.5)
  (h6 : purchases.almond_croissant_quantity = 1)
  (h7 : purchases.salami_cheese_croissant_quantity = 1)
  (h8 : purchases.plain_croissant_quantity = 1)
  (h9 : purchases.focaccia_quantity = 1)
  (h10 : purchases.latte_quantity = 2)
  : total_cost purchases = 21 := by
  sorry

end NUMINAMATH_CALUDE_sean_sunday_spending_l2787_278721


namespace NUMINAMATH_CALUDE_raft_capacity_l2787_278773

theorem raft_capacity (total_capacity : ℕ) (reduction_with_jackets : ℕ) (people_needing_jackets : ℕ) : 
  total_capacity = 21 → 
  reduction_with_jackets = 7 → 
  people_needing_jackets = 8 → 
  (total_capacity - (reduction_with_jackets * people_needing_jackets / (total_capacity - reduction_with_jackets))) = 17 := by
sorry

end NUMINAMATH_CALUDE_raft_capacity_l2787_278773


namespace NUMINAMATH_CALUDE_logans_score_l2787_278776

theorem logans_score (total_students : ℕ) (average_without_logan : ℚ) (average_with_logan : ℚ) :
  total_students = 20 →
  average_without_logan = 85 →
  average_with_logan = 86 →
  (total_students * average_with_logan - (total_students - 1) * average_without_logan : ℚ) = 105 :=
by sorry

end NUMINAMATH_CALUDE_logans_score_l2787_278776


namespace NUMINAMATH_CALUDE_population_reaches_capacity_l2787_278730

-- Define the constants
def land_area : ℕ := 40000
def acres_per_person : ℕ := 1
def base_population : ℕ := 500
def years_to_quadruple : ℕ := 20

-- Define the population growth function
def population (years : ℕ) : ℕ :=
  base_population * (4 ^ (years / years_to_quadruple))

-- Define the maximum capacity
def max_capacity : ℕ := land_area / acres_per_person

-- Theorem to prove
theorem population_reaches_capacity : 
  population 60 ≥ max_capacity ∧ population 40 < max_capacity :=
sorry

end NUMINAMATH_CALUDE_population_reaches_capacity_l2787_278730


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l2787_278799

/-- 
Given an arithmetic sequence with:
- First term a₁ = -5
- Last term aₙ = 40
- Common difference d = 3

Prove that the sequence has 16 terms.
-/
theorem arithmetic_sequence_length :
  ∀ (a : ℕ → ℤ),
  (a 0 = -5) →  -- First term
  (∀ n, a (n + 1) - a n = 3) →  -- Common difference
  (∃ k, a k = 40) →  -- Last term
  (∃ n, n = 16 ∧ a (n - 1) = 40) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l2787_278799


namespace NUMINAMATH_CALUDE_minimal_n_for_square_product_set_l2787_278738

theorem minimal_n_for_square_product_set (m : ℕ+) (p : ℕ) (h1 : p.Prime) (h2 : p ∣ m) 
  (h3 : p > Real.sqrt (2 * m) + 1) :
  ∃ (n : ℕ), n = m + p ∧
  (∀ (k : ℕ), k < n → 
    ¬∃ (S : Finset ℕ), 
      (∀ x ∈ S, m ≤ x ∧ x ≤ k) ∧ 
      (∃ y : ℕ, (S.prod id : ℕ) = y * y)) ∧
  ∃ (S : Finset ℕ), 
    (∀ x ∈ S, m ≤ x ∧ x ≤ n) ∧ 
    (∃ y : ℕ, (S.prod id : ℕ) = y * y) :=
by sorry

end NUMINAMATH_CALUDE_minimal_n_for_square_product_set_l2787_278738


namespace NUMINAMATH_CALUDE_virus_spread_l2787_278752

def infection_rate : ℕ → ℕ
  | 0 => 1
  | n + 1 => infection_rate n * 9

theorem virus_spread (x : ℕ) :
  (∃ n : ℕ, infection_rate n = 81) →
  (∀ n : ℕ, infection_rate (n + 1) = infection_rate n * 9) →
  infection_rate 2 = 81 →
  infection_rate 3 > 700 :=
by sorry

#check virus_spread

end NUMINAMATH_CALUDE_virus_spread_l2787_278752


namespace NUMINAMATH_CALUDE_zero_sufficient_for_perpendicular_zero_not_necessary_for_perpendicular_zero_sufficient_not_necessary_for_perpendicular_l2787_278753

/-- Line l1 with equation x + ay - a = 0 -/
def line1 (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 + a * p.2 - a = 0}

/-- Line l2 with equation ax - (2a - 3)y - 1 = 0 -/
def line2 (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | a * p.1 - (2 * a - 3) * p.2 - 1 = 0}

/-- Two lines are perpendicular -/
def perpendicular (l1 l2 : Set (ℝ × ℝ)) : Prop :=
  ∃ (m1 m2 : ℝ), (∀ (p q : ℝ × ℝ), p ∈ l1 → q ∈ l1 → p ≠ q → (q.2 - p.2) = m1 * (q.1 - p.1)) ∧
                 (∀ (p q : ℝ × ℝ), p ∈ l2 → q ∈ l2 → p ≠ q → (q.2 - p.2) = m2 * (q.1 - p.1)) ∧
                 m1 * m2 = -1

/-- a=0 is a sufficient condition for perpendicularity -/
theorem zero_sufficient_for_perpendicular :
  perpendicular (line1 0) (line2 0) :=
sorry

/-- a=0 is not a necessary condition for perpendicularity -/
theorem zero_not_necessary_for_perpendicular :
  ∃ a : ℝ, a ≠ 0 ∧ perpendicular (line1 a) (line2 a) :=
sorry

/-- Main theorem: a=0 is sufficient but not necessary for perpendicularity -/
theorem zero_sufficient_not_necessary_for_perpendicular :
  (perpendicular (line1 0) (line2 0)) ∧
  (∃ a : ℝ, a ≠ 0 ∧ perpendicular (line1 a) (line2 a)) :=
sorry

end NUMINAMATH_CALUDE_zero_sufficient_for_perpendicular_zero_not_necessary_for_perpendicular_zero_sufficient_not_necessary_for_perpendicular_l2787_278753


namespace NUMINAMATH_CALUDE_interest_rate_problem_l2787_278708

/-- 
Given a principal amount P and an interest rate R,
if increasing the rate by 1% for 3 years results in Rs. 63 more interest,
then P = Rs. 2100.
-/
theorem interest_rate_problem (P R : ℚ) : 
  (P * (R + 1) * 3 / 100 - P * R * 3 / 100 = 63) → P = 2100 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_problem_l2787_278708


namespace NUMINAMATH_CALUDE_remaining_money_l2787_278725

def base_8_to_10 (n : ℕ) : ℕ := 
  4 * 8^3 + 4 * 8^2 + 4 * 8^1 + 4 * 8^0

def savings : ℕ := base_8_to_10 4444

def ticket_cost : ℕ := 1000

theorem remaining_money : 
  savings - ticket_cost = 1340 := by sorry

end NUMINAMATH_CALUDE_remaining_money_l2787_278725


namespace NUMINAMATH_CALUDE_shade_in_three_folds_l2787_278719

/-- Represents a square grid -/
structure Grid :=
  (size : Nat)
  (shaded : Set (Nat × Nat))

/-- Represents a fold along a grid line -/
inductive Fold
  | Vertical (col : Nat)
  | Horizontal (row : Nat)

/-- Apply a fold to a grid -/
def applyFold (g : Grid) (f : Fold) : Grid :=
  sorry

/-- Check if the entire grid is shaded -/
def isFullyShaded (g : Grid) : Prop :=
  sorry

/-- Theorem stating that it's possible to shade the entire grid in 3 or fewer folds -/
theorem shade_in_three_folds (g : Grid) :
  ∃ (folds : List Fold), folds.length ≤ 3 ∧ isFullyShaded (folds.foldl applyFold g) :=
sorry

end NUMINAMATH_CALUDE_shade_in_three_folds_l2787_278719


namespace NUMINAMATH_CALUDE_shortened_area_l2787_278781

/-- Represents the dimensions of a rectangle --/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle --/
def area (r : Rectangle) : ℝ := r.length * r.width

/-- The original rectangle --/
def original : Rectangle := { length := 5, width := 7 }

/-- The rectangle after shortening one side --/
def shortened : Rectangle := { length := 3, width := 7 }

/-- Theorem stating the relationship between the original rectangle and the shortened rectangle --/
theorem shortened_area (h : area shortened = 21) :
  ∃ (r : Rectangle), r.length = original.length ∧ r.width = original.width - 2 ∧ area r = 25 := by
  sorry


end NUMINAMATH_CALUDE_shortened_area_l2787_278781


namespace NUMINAMATH_CALUDE_min_balls_for_three_same_color_l2787_278736

/-- Represents the number of balls of each color in the bag -/
structure BagContents where
  white : Nat
  black : Nat
  blue : Nat

/-- Calculates the minimum number of balls to draw to ensure at least three of the same color -/
def minBallsToEnsureThreeSameColor (bag : BagContents) : Nat :=
  7

/-- Theorem stating that for a bag with 5 white, 5 black, and 2 blue balls,
    the minimum number of balls to draw to ensure at least three of the same color is 7 -/
theorem min_balls_for_three_same_color :
  let bag : BagContents := { white := 5, black := 5, blue := 2 }
  minBallsToEnsureThreeSameColor bag = 7 := by
  sorry

end NUMINAMATH_CALUDE_min_balls_for_three_same_color_l2787_278736


namespace NUMINAMATH_CALUDE_equal_interest_rate_equal_interest_l2787_278766

/-- The rate at which a principal of 200 invested for 12 years produces the same
    interest as 400 invested for 5 years at 12% annual interest rate -/
theorem equal_interest_rate : ℝ :=
  let principal1 : ℝ := 200
  let time1 : ℝ := 12
  let principal2 : ℝ := 400
  let time2 : ℝ := 5
  let rate2 : ℝ := 12 / 100
  let interest2 : ℝ := principal2 * rate2 * time2
  10 / 100

/-- Proof that the calculated rate produces equal interest -/
theorem equal_interest (rate : ℝ) (h : rate = equal_interest_rate) :
  200 * rate * 12 = 400 * (12 / 100) * 5 := by
  sorry

#check equal_interest
#check equal_interest_rate

end NUMINAMATH_CALUDE_equal_interest_rate_equal_interest_l2787_278766


namespace NUMINAMATH_CALUDE_pool_dimensions_l2787_278722

/-- Represents the dimensions and costs of a rectangular open-top swimming pool. -/
structure Pool where
  shortSide : ℝ  -- Length of the shorter side of the rectangular bottom
  depth : ℝ      -- Depth of the pool
  bottomCost : ℝ -- Cost per square meter for constructing the bottom
  wallCost : ℝ   -- Cost per square meter for constructing the walls
  totalCost : ℝ  -- Total construction cost

/-- Calculates the total cost of constructing the pool. -/
def calculateCost (p : Pool) : ℝ :=
  p.bottomCost * p.shortSide * (2 * p.shortSide) + 
  p.wallCost * (p.shortSide + 2 * p.shortSide) * 2 * p.depth

/-- Theorem stating that the pool with given specifications has sides of 3m and 6m. -/
theorem pool_dimensions (p : Pool) 
  (h1 : p.depth = 2)
  (h2 : p.bottomCost = 200)
  (h3 : p.wallCost = 100)
  (h4 : p.totalCost = 7200)
  (h5 : calculateCost p = p.totalCost) :
  p.shortSide = 3 ∧ 2 * p.shortSide = 6 := by
  sorry

#check pool_dimensions

end NUMINAMATH_CALUDE_pool_dimensions_l2787_278722


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l2787_278739

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 32) :
  1 / x + 1 / y = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l2787_278739


namespace NUMINAMATH_CALUDE_f_100_of_1990_eq_11_l2787_278710

/-- Sum of digits of a natural number in base 10 -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Function f as defined in the problem -/
def f (n : ℕ) : ℕ := sumOfDigits (n^2 + 1)

/-- Iterated application of f, k times -/
def fIter (k : ℕ) (n : ℕ) : ℕ :=
  match k with
  | 0 => n
  | k+1 => f (fIter k n)

/-- The main theorem to prove -/
theorem f_100_of_1990_eq_11 : fIter 100 1990 = 11 := by sorry

end NUMINAMATH_CALUDE_f_100_of_1990_eq_11_l2787_278710


namespace NUMINAMATH_CALUDE_largest_number_problem_l2787_278787

theorem largest_number_problem (a b c d : ℕ) 
  (sum_abc : a + b + c = 222)
  (sum_abd : a + b + d = 208)
  (sum_acd : a + c + d = 197)
  (sum_bcd : b + c + d = 180) :
  max a (max b (max c d)) = 89 := by
  sorry

end NUMINAMATH_CALUDE_largest_number_problem_l2787_278787


namespace NUMINAMATH_CALUDE_square_difference_value_l2787_278706

theorem square_difference_value (x y : ℚ) 
  (h1 : x + y = 2/5) 
  (h2 : x - y = 1/10) : 
  x^2 - y^2 = 1/25 := by
sorry

end NUMINAMATH_CALUDE_square_difference_value_l2787_278706


namespace NUMINAMATH_CALUDE_square_root_of_square_l2787_278762

theorem square_root_of_square (x : ℝ) : Real.sqrt (x^2) = |x| := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_square_l2787_278762
