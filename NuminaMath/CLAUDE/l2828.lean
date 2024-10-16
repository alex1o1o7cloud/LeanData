import Mathlib

namespace NUMINAMATH_CALUDE_speed_in_still_water_l2828_282886

def upstream_speed : ℝ := 25
def downstream_speed : ℝ := 31

theorem speed_in_still_water :
  (upstream_speed + downstream_speed) / 2 = 28 := by
  sorry

end NUMINAMATH_CALUDE_speed_in_still_water_l2828_282886


namespace NUMINAMATH_CALUDE_mistaken_polynomial_calculation_l2828_282899

/-- Given a polynomial P such that P + (x^2 - 3x + 5) = 5x^2 - 2x + 4,
    prove that P = 4x^2 + x - 1 and P - (x^2 - 3x + 5) = 3x^2 + 4x - 6 -/
theorem mistaken_polynomial_calculation (P : ℝ → ℝ) 
  (h : ∀ x, P x + (x^2 - 3*x + 5) = 5*x^2 - 2*x + 4) : 
  (∀ x, P x = 4*x^2 + x - 1) ∧ 
  (∀ x, P x - (x^2 - 3*x + 5) = 3*x^2 + 4*x - 6) := by
  sorry

end NUMINAMATH_CALUDE_mistaken_polynomial_calculation_l2828_282899


namespace NUMINAMATH_CALUDE_function_zero_between_consecutive_integers_l2828_282815

theorem function_zero_between_consecutive_integers :
  ∃ (a b : ℤ), 
    (∀ x ∈ Set.Ioo a b, (Real.log x + x - 3 : ℝ) ≠ 0) ∧
    b = a + 1 ∧
    a + b = 5 := by
  sorry

end NUMINAMATH_CALUDE_function_zero_between_consecutive_integers_l2828_282815


namespace NUMINAMATH_CALUDE_three_digit_number_divisibility_l2828_282827

theorem three_digit_number_divisibility : ∃! x : ℕ, 
  100 ≤ x ∧ x ≤ 999 ∧ 
  (x - 6) % 7 = 0 ∧ 
  (x - 7) % 8 = 0 ∧ 
  (x - 8) % 9 = 0 ∧ 
  x = 503 := by sorry

end NUMINAMATH_CALUDE_three_digit_number_divisibility_l2828_282827


namespace NUMINAMATH_CALUDE_arithmetic_sequence_max_sum_l2828_282829

/-- Given an arithmetic sequence, prove that under certain conditions, 
    the maximum sum occurs at the 8th term -/
theorem arithmetic_sequence_max_sum 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ) 
  (h_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) 
  (h_sum : ∀ n, S n = (n * (2 * a 1 + (n - 1) * (a 2 - a 1))) / 2) 
  (h_15 : S 15 > 0) 
  (h_16 : S 16 < 0) : 
  ∃ (n : ℕ), ∀ (m : ℕ), S m ≤ S n ∧ n = 8 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_max_sum_l2828_282829


namespace NUMINAMATH_CALUDE_opposite_solutions_system_l2828_282873

theorem opposite_solutions_system (x y m : ℝ) : 
  x - 2*y = -3 → 
  2*x + 3*y = m - 1 → 
  x = -y → 
  m = 2 := by
sorry

end NUMINAMATH_CALUDE_opposite_solutions_system_l2828_282873


namespace NUMINAMATH_CALUDE_smallest_number_with_given_remainders_l2828_282863

theorem smallest_number_with_given_remainders : ∃ (b : ℕ), 
  b > 0 ∧
  b % 4 = 2 ∧
  b % 3 = 2 ∧
  b % 5 = 3 ∧
  (∀ (x : ℕ), x > 0 ∧ x % 4 = 2 ∧ x % 3 = 2 ∧ x % 5 = 3 → x ≥ b) ∧
  b = 38 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_given_remainders_l2828_282863


namespace NUMINAMATH_CALUDE_max_rooks_on_chessboard_l2828_282841

/-- Represents a chessboard --/
def Chessboard := Fin 8 → Fin 8 → Bool

/-- Checks if a rook at position (x, y) attacks an odd number of rooks on the board --/
def attacks_odd (board : Chessboard) (x y : Fin 8) : Bool :=
  sorry

/-- Returns the number of rooks on the board --/
def count_rooks (board : Chessboard) : Nat :=
  sorry

/-- Checks if a board configuration is valid according to the rules --/
def is_valid_configuration (board : Chessboard) : Prop :=
  sorry

theorem max_rooks_on_chessboard :
  ∃ (board : Chessboard),
    is_valid_configuration board ∧
    count_rooks board = 63 ∧
    ∀ (other_board : Chessboard),
      is_valid_configuration other_board →
      count_rooks other_board ≤ 63 :=
by sorry

end NUMINAMATH_CALUDE_max_rooks_on_chessboard_l2828_282841


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2828_282897

theorem complex_equation_solution (a b : ℝ) :
  (Complex.I : ℂ) * 2 + 1 = (Complex.I + 1) * (Complex.I * b + a) →
  a = 3/2 ∧ b = 1/2 :=
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2828_282897


namespace NUMINAMATH_CALUDE_stone_breadth_proof_l2828_282898

/-- Given a hall and stones with specific dimensions, prove the breadth of each stone. -/
theorem stone_breadth_proof (hall_length : ℝ) (hall_width : ℝ) (stone_length : ℝ) (stone_count : ℕ) 
  (h1 : hall_length = 36)
  (h2 : hall_width = 15)
  (h3 : stone_length = 0.8)
  (h4 : stone_count = 1350) :
  ∃ (stone_width : ℝ), 
    stone_width = 0.5 ∧ 
    (hall_length * hall_width * 100) = (stone_count : ℝ) * stone_length * stone_width * 100 := by
  sorry


end NUMINAMATH_CALUDE_stone_breadth_proof_l2828_282898


namespace NUMINAMATH_CALUDE_mixed_strategy_optimal_mixed_strategy_optimal_at_60_l2828_282874

/-- Represents the cost function for purchasing heaters from a store -/
structure StoreCost where
  typeA : ℝ  -- Cost per unit of Type A heater (including shipping)
  typeB : ℝ  -- Cost per unit of Type B heater (including shipping)

/-- Calculates the total cost for a store given the number of Type A heaters -/
def totalCost (store : StoreCost) (x : ℝ) : ℝ :=
  store.typeA * x + store.typeB * (100 - x)

/-- Store A's cost structure -/
def storeA : StoreCost := { typeA := 110, typeB := 210 }

/-- Store B's cost structure -/
def storeB : StoreCost := { typeA := 120, typeB := 202 }

/-- Cost function for buying Type A from Store A and Type B from Store B -/
def mixedCost (x : ℝ) : ℝ := storeA.typeA * x + storeB.typeB * (100 - x)

/-- Theorem: The mixed purchasing strategy is always the most cost-effective -/
theorem mixed_strategy_optimal (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 100) : 
  mixedCost x ≤ min (totalCost storeA x) (totalCost storeB x) := by
  sorry

/-- Corollary: When x = 60, the mixed strategy is more cost-effective than buying from a single store -/
theorem mixed_strategy_optimal_at_60 : 
  mixedCost 60 < min (totalCost storeA 60) (totalCost storeB 60) := by
  sorry

end NUMINAMATH_CALUDE_mixed_strategy_optimal_mixed_strategy_optimal_at_60_l2828_282874


namespace NUMINAMATH_CALUDE_boat_journey_time_l2828_282817

/-- The boat's journey time given specific conditions -/
theorem boat_journey_time 
  (stream_velocity : ℝ) 
  (boat_speed_still : ℝ) 
  (distance_AB : ℝ) 
  (h1 : stream_velocity = 4)
  (h2 : boat_speed_still = 14)
  (h3 : distance_AB = 180) :
  let downstream_speed := boat_speed_still + stream_velocity
  let upstream_speed := boat_speed_still - stream_velocity
  let time_downstream := distance_AB / downstream_speed
  let time_upstream := (distance_AB / 2) / upstream_speed
  time_downstream + time_upstream = 19 := by
sorry

end NUMINAMATH_CALUDE_boat_journey_time_l2828_282817


namespace NUMINAMATH_CALUDE_jogger_train_distance_l2828_282831

/-- Calculates the distance a jogger is ahead of a train's engine given their speeds and the time it takes for the train to pass the jogger. -/
theorem jogger_train_distance
  (jogger_speed : ℝ)
  (train_speed : ℝ)
  (train_length : ℝ)
  (passing_time : ℝ)
  (h1 : jogger_speed = 9 / 3.6)  -- Convert 9 km/hr to m/s
  (h2 : train_speed = 45 / 3.6)  -- Convert 45 km/hr to m/s
  (h3 : train_length = 120)
  (h4 : passing_time = 32) :
  train_speed * passing_time - jogger_speed * passing_time - train_length = 200 :=
by sorry

end NUMINAMATH_CALUDE_jogger_train_distance_l2828_282831


namespace NUMINAMATH_CALUDE_shaded_square_area_ratio_l2828_282818

theorem shaded_square_area_ratio :
  ∀ (n : ℕ) (large_square_side : ℝ) (small_square_side : ℝ),
    n = 4 →
    large_square_side = n * small_square_side →
    small_square_side > 0 →
    (2 * small_square_side^2) / (large_square_side^2) = 1/8 :=
by sorry

end NUMINAMATH_CALUDE_shaded_square_area_ratio_l2828_282818


namespace NUMINAMATH_CALUDE_calculator_cost_l2828_282862

/-- Given information about calculator purchases, prove the cost of each graphing calculator. -/
theorem calculator_cost (total_cost : ℕ) (total_calculators : ℕ) (scientific_cost : ℕ)
  (scientific_count : ℕ) (graphing_count : ℕ)
  (h1 : total_cost = 1625)
  (h2 : total_calculators = 45)
  (h3 : scientific_cost = 10)
  (h4 : scientific_count = 20)
  (h5 : graphing_count = 25)
  (h6 : total_calculators = scientific_count + graphing_count) :
  (total_cost - scientific_cost * scientific_count) / graphing_count = 57 := by
  sorry

#eval (1625 - 10 * 20) / 25  -- Should output 57

end NUMINAMATH_CALUDE_calculator_cost_l2828_282862


namespace NUMINAMATH_CALUDE_success_permutations_l2828_282828

/-- The number of distinct permutations of a word with repeated letters -/
def permutationsWithRepetition (totalLetters : ℕ) (repetitions : List ℕ) : ℕ :=
  Nat.factorial totalLetters / (repetitions.map Nat.factorial).prod

/-- The word "SUCCESS" has 7 letters with 'S' appearing 3 times, 'C' appearing 2 times, 
    and 'U' and 'E' appearing once each -/
def successWord : (ℕ × List ℕ) :=
  (7, [3, 2, 1, 1])

theorem success_permutations :
  permutationsWithRepetition successWord.1 successWord.2 = 420 := by
  sorry

end NUMINAMATH_CALUDE_success_permutations_l2828_282828


namespace NUMINAMATH_CALUDE_parabola_focus_coordinates_l2828_282850

/-- The focus of the parabola x = -8y^2 has coordinates (-1/32, 0) -/
theorem parabola_focus_coordinates :
  let f : ℝ × ℝ → ℝ := fun (x, y) ↦ x + 8 * y^2
  ∃! p : ℝ × ℝ, p = (-1/32, 0) ∧ 
    (∀ q : ℝ × ℝ, f q = 0 → (q.1 - p.1)^2 + (q.2 - p.2)^2 = (q.2 - 0)^2 + (1/16)^2) :=
by sorry

end NUMINAMATH_CALUDE_parabola_focus_coordinates_l2828_282850


namespace NUMINAMATH_CALUDE_japanese_selectors_l2828_282857

theorem japanese_selectors (j c f : ℕ) : 
  j = 3 * c →
  c = f + 15 →
  j + c + f = 165 →
  j = 108 := by
sorry

end NUMINAMATH_CALUDE_japanese_selectors_l2828_282857


namespace NUMINAMATH_CALUDE_function_value_problem_l2828_282848

theorem function_value_problem (f : ℝ → ℝ) (h : ∀ x, f (x + 1) = 2 * x - 1) : 
  f 1 = -1 := by
sorry

end NUMINAMATH_CALUDE_function_value_problem_l2828_282848


namespace NUMINAMATH_CALUDE_multiply_102_98_l2828_282834

theorem multiply_102_98 : 102 * 98 = 9996 := by
  sorry

end NUMINAMATH_CALUDE_multiply_102_98_l2828_282834


namespace NUMINAMATH_CALUDE_solution_equality_l2828_282895

theorem solution_equality (a b c d : ℝ) 
  (eq1 : a - Real.sqrt (1 - b^2) + Real.sqrt (1 - c^2) = d)
  (eq2 : b - Real.sqrt (1 - c^2) + Real.sqrt (1 - d^2) = a)
  (eq3 : c - Real.sqrt (1 - d^2) + Real.sqrt (1 - a^2) = b)
  (eq4 : d - Real.sqrt (1 - a^2) + Real.sqrt (1 - b^2) = c)
  (nonneg1 : 1 - a^2 ≥ 0)
  (nonneg2 : 1 - b^2 ≥ 0)
  (nonneg3 : 1 - c^2 ≥ 0)
  (nonneg4 : 1 - d^2 ≥ 0) :
  a = b ∧ b = c ∧ c = d := by
  sorry

end NUMINAMATH_CALUDE_solution_equality_l2828_282895


namespace NUMINAMATH_CALUDE_circle_areas_in_right_triangle_l2828_282877

theorem circle_areas_in_right_triangle (a b c : Real) (r : Real) :
  a = 3 ∧ b = 4 ∧ c = 5 ∧ r = 1 →
  a^2 + b^2 = c^2 →
  let α := Real.arctan (a / b)
  let β := Real.arctan (b / a)
  let γ := π / 2
  (α + β + γ = π) →
  (α / 2 + β / 2 + γ / 2) * r^2 = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_areas_in_right_triangle_l2828_282877


namespace NUMINAMATH_CALUDE_system_solution_l2828_282864

theorem system_solution :
  let f (x y z : ℚ) := (x * y = x + 2 * y) ∧ (y * z = y + 3 * z) ∧ (z * x = z + 4 * x)
  ∀ x y z : ℚ, f x y z ↔ (x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 25/9 ∧ y = 25/7 ∧ z = 25/4) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l2828_282864


namespace NUMINAMATH_CALUDE_largest_odd_digit_multiple_of_5_is_correct_l2828_282823

/-- A function that checks if a positive integer has only odd digits -/
def has_only_odd_digits (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d % 2 = 1

/-- The largest positive integer less than 10000 with only odd digits that is a multiple of 5 -/
def largest_odd_digit_multiple_of_5 : ℕ := 9995

theorem largest_odd_digit_multiple_of_5_is_correct :
  (largest_odd_digit_multiple_of_5 < 10000) ∧
  (has_only_odd_digits largest_odd_digit_multiple_of_5) ∧
  (largest_odd_digit_multiple_of_5 % 5 = 0) ∧
  (∀ n : ℕ, n < 10000 → has_only_odd_digits n → n % 5 = 0 → n ≤ largest_odd_digit_multiple_of_5) :=
by sorry

#eval largest_odd_digit_multiple_of_5

end NUMINAMATH_CALUDE_largest_odd_digit_multiple_of_5_is_correct_l2828_282823


namespace NUMINAMATH_CALUDE_two_thousand_five_power_l2828_282806

theorem two_thousand_five_power : ∃ a b : ℕ, (2005 : ℕ)^2005 = a^2 + b^2 ∧ ¬∃ c d : ℕ, (2005 : ℕ)^2005 = c^3 + d^3 := by
  sorry

end NUMINAMATH_CALUDE_two_thousand_five_power_l2828_282806


namespace NUMINAMATH_CALUDE_angle_system_solution_l2828_282896

theorem angle_system_solution (α β : Real) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2)
  (eq1 : 2 * Real.sin (2 * β) = 3 * Real.sin (2 * α))
  (eq2 : Real.tan β = 3 * Real.tan α) :
  α = Real.arctan (Real.sqrt 7 / 7) ∧ β = Real.arctan (3 * Real.sqrt 7 / 7) := by
sorry

end NUMINAMATH_CALUDE_angle_system_solution_l2828_282896


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2828_282809

/-- Given a hyperbola C: mx^2 + ny^2 = 1 (m > 0, n < 0) with one of its asymptotes
    tangent to the circle x^2 + y^2 - 6x - 2y + 9 = 0, 
    the eccentricity of C is 5/4. -/
theorem hyperbola_eccentricity (m n : ℝ) (hm : m > 0) (hn : n < 0) :
  let C := {(x, y) : ℝ × ℝ | m * x^2 + n * y^2 = 1}
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 - 6*x - 2*y + 9 = 0}
  let asymptote := {(x, y) : ℝ × ℝ | Real.sqrt m * x - Real.sqrt (-n) * y = 0}
  (∃ (p : ℝ × ℝ), p ∈ asymptote ∧ p ∈ circle) →
  let a := 1 / Real.sqrt m
  let b := 1 / Real.sqrt (-n)
  let e := Real.sqrt (1 + (b/a)^2)
  e = 5/4 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2828_282809


namespace NUMINAMATH_CALUDE_crafts_club_necklaces_l2828_282842

theorem crafts_club_necklaces 
  (members : ℕ) 
  (beads_per_necklace : ℕ) 
  (total_beads : ℕ) 
  (h1 : members = 9)
  (h2 : beads_per_necklace = 50)
  (h3 : total_beads = 900) :
  total_beads / beads_per_necklace / members = 2 := by
sorry

end NUMINAMATH_CALUDE_crafts_club_necklaces_l2828_282842


namespace NUMINAMATH_CALUDE_cubic_polynomial_sum_l2828_282835

/-- Given a cubic polynomial Q with specific values at 0, 1, and -1, prove that Q(2) + Q(-2) = 20m -/
theorem cubic_polynomial_sum (m : ℝ) (Q : ℝ → ℝ) :
  (∃ a b c : ℝ, ∀ x, Q x = a * x^3 + b * x^2 + c * x + 2 * m) →
  Q 0 = 2 * m →
  Q 1 = 3 * m →
  Q (-1) = 5 * m →
  Q 2 + Q (-2) = 20 * m :=
by sorry

end NUMINAMATH_CALUDE_cubic_polynomial_sum_l2828_282835


namespace NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l2828_282810

theorem greatest_divisor_with_remainders : Nat.gcd (28572 - 142) (39758 - 84) = 2 := by
  sorry

end NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l2828_282810


namespace NUMINAMATH_CALUDE_sum_of_even_and_multiples_of_seven_l2828_282866

/-- The number of five-digit even numbers -/
def X : ℕ := 45000

/-- The number of five-digit multiples of 7 -/
def Y : ℕ := 12857

/-- The sum of five-digit even numbers and five-digit multiples of 7 -/
theorem sum_of_even_and_multiples_of_seven : X + Y = 57857 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_even_and_multiples_of_seven_l2828_282866


namespace NUMINAMATH_CALUDE_trig_identities_for_point_l2828_282893

/-- Given a point P(1, -3) on the terminal side of angle α, prove trigonometric identities. -/
theorem trig_identities_for_point (α : Real) :
  let P : ℝ × ℝ := (1, -3)
  (P.1 = Real.cos α * Real.sqrt (P.1^2 + P.2^2) ∧ 
   P.2 = Real.sin α * Real.sqrt (P.1^2 + P.2^2)) →
  Real.sin α = -3 * Real.sqrt 10 / 10 ∧ 
  Real.sqrt 10 * Real.cos α + Real.tan α = -2 := by
sorry

end NUMINAMATH_CALUDE_trig_identities_for_point_l2828_282893


namespace NUMINAMATH_CALUDE_cloth_profit_theorem_l2828_282870

/-- Calculates the profit per meter of cloth (rounded to the nearest rupee) -/
def profit_per_meter (meters : ℕ) (total_selling_price : ℚ) (cost_price_per_meter : ℚ) : ℕ :=
  let total_cost_price := meters * cost_price_per_meter
  let total_profit := total_selling_price - total_cost_price
  let profit_per_meter := total_profit / meters
  (profit_per_meter + 1/2).floor.toNat

/-- The profit per meter of cloth is 29 rupees -/
theorem cloth_profit_theorem :
  profit_per_meter 78 6788 (58.02564102564102) = 29 := by
  sorry

end NUMINAMATH_CALUDE_cloth_profit_theorem_l2828_282870


namespace NUMINAMATH_CALUDE_unique_a_l2828_282825

/-- The equation is quadratic in x -/
def is_quadratic (a : ℝ) : Prop :=
  |a - 1| = 2

/-- The coefficient of the quadratic term is non-zero -/
def coeff_nonzero (a : ℝ) : Prop :=
  a - 3 ≠ 0

/-- The value of a that satisfies the conditions -/
theorem unique_a : ∃! a : ℝ, is_quadratic a ∧ coeff_nonzero a :=
  sorry

end NUMINAMATH_CALUDE_unique_a_l2828_282825


namespace NUMINAMATH_CALUDE_binomial_eight_zero_l2828_282865

theorem binomial_eight_zero : Nat.choose 8 0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_eight_zero_l2828_282865


namespace NUMINAMATH_CALUDE_online_store_problem_l2828_282847

/-- Represents the purchase and selling prices of products A and B -/
structure Prices where
  purchaseA : ℝ
  purchaseB : ℝ
  sellingA : ℝ
  sellingB : ℝ

/-- Represents the first purchase conditions -/
structure FirstPurchase where
  totalItems : ℕ
  totalCost : ℝ

/-- Represents the second purchase conditions -/
structure SecondPurchase where
  totalItems : ℕ
  maxCost : ℝ

/-- Represents the sales conditions for product B -/
structure BSales where
  initialSales : ℕ
  additionalSalesPerReduction : ℕ

/-- Main theorem stating the solutions to the problem -/
theorem online_store_problem 
  (prices : Prices)
  (firstPurchase : FirstPurchase)
  (secondPurchase : SecondPurchase)
  (bSales : BSales)
  (h1 : prices.purchaseA = 30)
  (h2 : prices.purchaseB = 25)
  (h3 : prices.sellingA = 45)
  (h4 : prices.sellingB = 37)
  (h5 : firstPurchase.totalItems = 30)
  (h6 : firstPurchase.totalCost = 850)
  (h7 : secondPurchase.totalItems = 80)
  (h8 : secondPurchase.maxCost = 2200)
  (h9 : bSales.initialSales = 4)
  (h10 : bSales.additionalSalesPerReduction = 2) :
  (∃ (x y : ℕ), x + y = firstPurchase.totalItems ∧ 
    prices.purchaseA * x + prices.purchaseB * y = firstPurchase.totalCost ∧ 
    x = 20 ∧ y = 10) ∧
  (∃ (m : ℕ), m ≤ secondPurchase.totalItems ∧ 
    prices.purchaseA * m + prices.purchaseB * (secondPurchase.totalItems - m) ≤ secondPurchase.maxCost ∧
    (prices.sellingA - prices.purchaseA) * m + (prices.sellingB - prices.purchaseB) * (secondPurchase.totalItems - m) = 2520 ∧
    m = 40) ∧
  (∃ (a₁ a₂ : ℝ), (12 - a₁) * (bSales.initialSales + 2 * a₁) = 90 ∧
    (12 - a₂) * (bSales.initialSales + 2 * a₂) = 90 ∧
    a₁ = 3 ∧ a₂ = 7) := by
  sorry

end NUMINAMATH_CALUDE_online_store_problem_l2828_282847


namespace NUMINAMATH_CALUDE_salary_change_l2828_282885

theorem salary_change (x : ℝ) :
  (1 - x / 100) * (1 + x / 100) = 0.75 → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_salary_change_l2828_282885


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2828_282804

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x ↦ x^2 - 6*x - 3
  ∃ x1 x2 : ℝ, x1 = 3 + 2*Real.sqrt 3 ∧ 
             x2 = 3 - 2*Real.sqrt 3 ∧ 
             f x1 = 0 ∧ f x2 = 0 ∧
             ∀ x : ℝ, f x = 0 → x = x1 ∨ x = x2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2828_282804


namespace NUMINAMATH_CALUDE_root_relationship_l2828_282816

-- Define the first polynomial equation
def f (x : ℝ) : ℝ := x^3 - 6*x^2 - 39*x - 10

-- Define the second polynomial equation
def g (x : ℝ) : ℝ := x^3 + x^2 - 20*x - 50

-- State the theorem
theorem root_relationship :
  (∃ (x y : ℝ), f x = 0 ∧ g y = 0 ∧ x = 2*y) →
  f 10 = 0 ∧ g 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_root_relationship_l2828_282816


namespace NUMINAMATH_CALUDE_exists_disjoint_graphs_l2828_282875

open Set

/-- The graph of a function f: [0, 1] → ℝ -/
def Graph (f : ℝ → ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 ∈ Icc 0 1 ∧ p.2 = f p.1}

/-- The graph of the translated function f(x-a) -/
def GraphTranslated (f : ℝ → ℝ) (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 ∈ Icc a (a+1) ∧ p.2 = f (p.1 - a)}

theorem exists_disjoint_graphs :
  ∀ a ∈ Ioo 0 1, ∃ f : ℝ → ℝ,
    Continuous f ∧
    f 0 = 0 ∧ f 1 = 0 ∧
    (Graph f) ∩ (GraphTranslated f a) = ∅ :=
sorry

end NUMINAMATH_CALUDE_exists_disjoint_graphs_l2828_282875


namespace NUMINAMATH_CALUDE_job_completion_time_l2828_282868

/-- The time taken for three workers to complete a job together, given their individual efficiencies -/
theorem job_completion_time 
  (sakshi_time : ℝ) 
  (tanya_efficiency : ℝ) 
  (rahul_efficiency : ℝ) 
  (h1 : sakshi_time = 20) 
  (h2 : tanya_efficiency = 1.25) 
  (h3 : rahul_efficiency = 1.5) : 
  (1 / (1 / sakshi_time + tanya_efficiency * (1 / sakshi_time) + rahul_efficiency * tanya_efficiency * (1 / sakshi_time))) = 160 / 33 :=
by sorry

end NUMINAMATH_CALUDE_job_completion_time_l2828_282868


namespace NUMINAMATH_CALUDE_line_through_circle_center_l2828_282840

theorem line_through_circle_center (a : ℝ) : 
  (∃ x y : ℝ, x^2 + y^2 + 2*x - 4*y = 0 ∧ 3*x + y + a = 0 ∧ 
   ∀ cx cy : ℝ, (cx - x)^2 + (cy - y)^2 ≤ (cx + 1)^2 + (cy - 2)^2) → 
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_line_through_circle_center_l2828_282840


namespace NUMINAMATH_CALUDE_dividing_line_halves_area_l2828_282891

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The T-shaped region -/
def TRegion : Set Point := {p | 
  (0 ≤ p.x ∧ p.x ≤ 4 ∧ 0 ≤ p.y ∧ p.y ≤ 4) ∨
  (4 < p.x ∧ p.x ≤ 7 ∧ 0 ≤ p.y ∧ p.y ≤ 2)
}

/-- The line y = (1/2)x -/
def DividingLine (p : Point) : Prop :=
  p.y = (1/2) * p.x

/-- The area of a region -/
noncomputable def area (s : Set Point) : ℝ := sorry

/-- The part of the region above the line -/
def UpperRegion : Set Point :=
  {p ∈ TRegion | p.y > (1/2) * p.x}

/-- The part of the region below the line -/
def LowerRegion : Set Point :=
  {p ∈ TRegion | p.y < (1/2) * p.x}

/-- The theorem stating that the line y = (1/2)x divides the T-shaped region in half -/
theorem dividing_line_halves_area : 
  area UpperRegion = area LowerRegion := by sorry

end NUMINAMATH_CALUDE_dividing_line_halves_area_l2828_282891


namespace NUMINAMATH_CALUDE_unique_solution_exponential_equation_l2828_282805

theorem unique_solution_exponential_equation :
  ∃! x : ℝ, (2 : ℝ) ^ (x^2 - 6*x + 8) = (1/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_exponential_equation_l2828_282805


namespace NUMINAMATH_CALUDE_cindy_calculation_l2828_282851

theorem cindy_calculation (x : ℚ) : 
  ((x - 5) * 3 / 7 = 10) → ((3 * x - 5) / 7 = 80 / 7) := by
  sorry

end NUMINAMATH_CALUDE_cindy_calculation_l2828_282851


namespace NUMINAMATH_CALUDE_polynomial_sum_l2828_282808

theorem polynomial_sum (f g : ℝ → ℝ) :
  (∀ x, f x + g x = 3 * x - x^2) →
  (∀ x, f x = x^2 - 4 * x + 3) →
  (∀ x, g x = -2 * x^2 + 7 * x - 3) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_sum_l2828_282808


namespace NUMINAMATH_CALUDE_blue_balls_unchanged_l2828_282890

/-- Represents the number of balls of each color in the box -/
structure BallCount where
  red : Nat
  blue : Nat
  yellow : Nat

/-- The operation of adding yellow balls to the box -/
def addYellowBalls (initial : BallCount) (added : Nat) : BallCount :=
  { red := initial.red,
    blue := initial.blue,
    yellow := initial.yellow + added }

theorem blue_balls_unchanged (initial : BallCount) (added : Nat) :
  (addYellowBalls initial added).blue = initial.blue :=
by sorry

end NUMINAMATH_CALUDE_blue_balls_unchanged_l2828_282890


namespace NUMINAMATH_CALUDE_comparison_proofs_l2828_282846

theorem comparison_proofs :
  (-2.3 < 2.4) ∧ (-3/4 > -5/6) ∧ (0 > -Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_comparison_proofs_l2828_282846


namespace NUMINAMATH_CALUDE_count_integers_in_range_l2828_282813

theorem count_integers_in_range : 
  ∃! n : ℕ, n = (Finset.filter (fun x : ℕ => 
    50 < x^2 + 6*x + 9 ∧ x^2 + 6*x + 9 < 100) (Finset.range 100)).card ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_count_integers_in_range_l2828_282813


namespace NUMINAMATH_CALUDE_truck_rental_charge_per_mile_l2828_282802

/-- Given a truck rental scenario, calculate the charge per mile. -/
theorem truck_rental_charge_per_mile
  (rental_fee : ℚ)
  (total_paid : ℚ)
  (miles_driven : ℕ)
  (h1 : rental_fee = 2099 / 100)
  (h2 : total_paid = 9574 / 100)
  (h3 : miles_driven = 299)
  : (total_paid - rental_fee) / miles_driven = 1 / 4 := by
  sorry

#eval (9574 / 100 : ℚ) - (2099 / 100 : ℚ)
#eval ((9574 / 100 : ℚ) - (2099 / 100 : ℚ)) / 299

end NUMINAMATH_CALUDE_truck_rental_charge_per_mile_l2828_282802


namespace NUMINAMATH_CALUDE_student_age_problem_l2828_282855

theorem student_age_problem (num_students : ℕ) (teacher_age : ℕ) 
  (h1 : num_students = 20)
  (h2 : teacher_age = 42)
  (h3 : ∀ (student_avg : ℝ), 
    (num_students * student_avg + teacher_age) / (num_students + 1) = student_avg + 1) :
  ∃ (student_avg : ℝ), student_avg = 21 := by
sorry

end NUMINAMATH_CALUDE_student_age_problem_l2828_282855


namespace NUMINAMATH_CALUDE_odd_even_function_sum_l2828_282822

-- Define the properties of odd and even functions
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def IsEven (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

-- State the theorem
theorem odd_even_function_sum (f g : ℝ → ℝ) (h : ℝ → ℝ) 
  (hf : IsOdd f) (hg : IsEven g) 
  (sum_eq : ∀ x ≠ 1, f x + g x = 1 / (x - 1)) :
  ∀ x ≠ 1, f x = x / (x^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_odd_even_function_sum_l2828_282822


namespace NUMINAMATH_CALUDE_bird_count_correct_bird_count_l2828_282833

theorem bird_count (total_heads : ℕ) (total_legs : ℕ) : ℕ :=
  let birds : ℕ := total_heads - (total_legs - 2 * total_heads) / 2
  birds

theorem correct_bird_count :
  bird_count 300 980 = 110 := by
  sorry

end NUMINAMATH_CALUDE_bird_count_correct_bird_count_l2828_282833


namespace NUMINAMATH_CALUDE_music_stand_cost_l2828_282811

/-- The cost of Jason's music stand purchase --/
theorem music_stand_cost (flute_cost song_book_cost total_spent : ℝ) 
  (h1 : flute_cost = 142.46)
  (h2 : song_book_cost = 7)
  (h3 : total_spent = 158.35) :
  total_spent - (flute_cost + song_book_cost) = 8.89 := by
  sorry

end NUMINAMATH_CALUDE_music_stand_cost_l2828_282811


namespace NUMINAMATH_CALUDE_transformed_area_theorem_l2828_282872

-- Define the transformation matrix
def A : Matrix (Fin 2) (Fin 2) ℝ := !![3, 4; 8, -2]

-- Define the original region's area
def original_area : ℝ := 15

-- Theorem statement
theorem transformed_area_theorem :
  let transformed_area := original_area * abs (Matrix.det A)
  transformed_area = 570 := by sorry

end NUMINAMATH_CALUDE_transformed_area_theorem_l2828_282872


namespace NUMINAMATH_CALUDE_tangent_line_proof_l2828_282819

-- Define the given curve
def f (x : ℝ) : ℝ := x^3 + 3*x^2 - 5

-- Define the given line
def line1 (x y : ℝ) : Prop := 2*x - 6*y + 1 = 0

-- Define the tangent line we want to prove
def line2 (x y : ℝ) : Prop := 3*x + y + 6 = 0

-- Theorem statement
theorem tangent_line_proof :
  ∃ (x₀ y₀ : ℝ),
    -- The point (x₀, y₀) is on the curve
    f x₀ = y₀ ∧
    -- The point (x₀, y₀) is on line2
    line2 x₀ y₀ ∧
    -- line2 is tangent to the curve at (x₀, y₀)
    (deriv f x₀ = -3) ∧
    -- line1 and line2 are perpendicular
    (∀ (x₁ y₁ x₂ y₂ : ℝ),
      line1 x₁ y₁ → line1 x₂ y₂ → x₁ ≠ x₂ →
      line2 x₁ y₁ → line2 x₂ y₂ → x₁ ≠ x₂ →
      (y₂ - y₁) / (x₂ - x₁) * (y₂ - y₁) / (x₂ - x₁) = -1) :=
by
  sorry

end NUMINAMATH_CALUDE_tangent_line_proof_l2828_282819


namespace NUMINAMATH_CALUDE_coordinate_sum_theorem_l2828_282859

theorem coordinate_sum_theorem (g : ℝ → ℝ) (h : g 4 = 7) :
  ∃ (x y : ℝ), 3 * y = 2 * g (3 * x) + 6 ∧ x + y = 8 := by
  sorry

end NUMINAMATH_CALUDE_coordinate_sum_theorem_l2828_282859


namespace NUMINAMATH_CALUDE_ellipse_perpendicular_sum_l2828_282881

/-- Theorem about perpendicular distances in an ellipse -/
theorem ellipse_perpendicular_sum (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_a_ge_b : a ≥ b) :
  let e := Real.sqrt (a^2 - b^2)
  ∀ (x₀ y₀ : ℝ), x₀^2 / a^2 + y₀^2 / b^2 = 1 →
    let d₁ := |y₀ - b| / b / Real.sqrt ((x₀/a^2)^2 + (y₀/b^2)^2)
    let d₂ := |y₀ + b| / b / Real.sqrt ((x₀/a^2)^2 + (y₀/b^2)^2)
    d₁^2 + d₂^2 = 2 * a^2 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_perpendicular_sum_l2828_282881


namespace NUMINAMATH_CALUDE_harolds_books_ratio_l2828_282820

theorem harolds_books_ratio (h m : ℝ) : 
  h > 0 ∧ m > 0 → 
  (1/3 : ℝ) * h + (1/2 : ℝ) * m = (5/6 : ℝ) * m → 
  h / m = 1 := by
sorry

end NUMINAMATH_CALUDE_harolds_books_ratio_l2828_282820


namespace NUMINAMATH_CALUDE_solution_sum_of_squares_l2828_282803

theorem solution_sum_of_squares (x y : ℝ) : 
  x * y = 8 → 
  x^2 * y + x * y^2 + 2*x + 2*y = 108 → 
  x^2 + y^2 = 100.64 := by
sorry

end NUMINAMATH_CALUDE_solution_sum_of_squares_l2828_282803


namespace NUMINAMATH_CALUDE_triangle_angle_sum_l2828_282814

theorem triangle_angle_sum (A B C : ℝ) (h1 : A + B + C = 180) (h2 : 180 - C = 130) :
  A + B = 130 := by sorry

end NUMINAMATH_CALUDE_triangle_angle_sum_l2828_282814


namespace NUMINAMATH_CALUDE_paper_folding_ratio_l2828_282860

theorem paper_folding_ratio : 
  let square_side : ℝ := 8
  let folded_height : ℝ := square_side / 2
  let folded_width : ℝ := square_side
  let cut_height : ℝ := folded_height / 3
  let small_rect_height : ℝ := cut_height
  let small_rect_width : ℝ := folded_width
  let large_rect_height : ℝ := folded_height - cut_height
  let large_rect_width : ℝ := folded_width
  let small_rect_perimeter : ℝ := 2 * (small_rect_height + small_rect_width)
  let large_rect_perimeter : ℝ := 2 * (large_rect_height + large_rect_width)
  small_rect_perimeter / large_rect_perimeter = 7 / 11 := by
sorry

end NUMINAMATH_CALUDE_paper_folding_ratio_l2828_282860


namespace NUMINAMATH_CALUDE_correct_divisor_l2828_282856

/-- Represents a person with their age in years -/
structure Person where
  name : String
  age : Nat

/-- The divisor that gives Gokul's age when (Arun's age - 6) is divided by it -/
def divisor (arun gokul : Person) : Nat :=
  (arun.age - 6) / gokul.age

theorem correct_divisor (arun madan gokul : Person) : 
  arun.name = "Arun" → 
  arun.age = 60 →
  madan.name = "Madan" → 
  madan.age = 5 →
  gokul.name = "Gokul" →
  gokul.age = madan.age - 2 →
  divisor arun gokul = 18 := by
  sorry

#check correct_divisor

end NUMINAMATH_CALUDE_correct_divisor_l2828_282856


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l2828_282889

theorem geometric_sequence_product (a : ℕ → ℝ) :
  (∀ n, a (n + 1) / a n = a 2 / a 1) →  -- Geometric sequence condition
  a 1 = 2 →                            -- First term is 2
  a 5 = 8 →                            -- Fifth term is 8
  a 2 * a 3 * a 4 = 64 := by            -- Product of middle terms is 64
sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l2828_282889


namespace NUMINAMATH_CALUDE_ellipse_m_value_l2828_282894

/-- An ellipse with equation x²/(10-m) + y²/(m-2) = 1, major axis on x-axis, and focal distance 4 -/
structure Ellipse (m : ℝ) :=
  (eq : ∀ x y : ℝ, x^2 / (10 - m) + y^2 / (m - 2) = 1)
  (major_axis_x : ℝ → ℝ)
  (focal_distance : ℝ)
  (h_focal_distance : focal_distance = 4)

/-- The value of m for the given ellipse is 4 -/
theorem ellipse_m_value (m : ℝ) (e : Ellipse m) : m = 4 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_m_value_l2828_282894


namespace NUMINAMATH_CALUDE_min_value_xy_plus_reciprocal_l2828_282858

theorem min_value_xy_plus_reciprocal (x y : ℝ) 
  (h1 : x + y = -1) 
  (h2 : x < 0) 
  (h3 : y < 0) : 
  ∃ (min : ℝ), min = 17/4 ∧ ∀ z, z = x*y + 1/(x*y) → z ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_xy_plus_reciprocal_l2828_282858


namespace NUMINAMATH_CALUDE_stratified_sample_sum_eq_six_l2828_282839

/-- Represents the number of varieties in each food category -/
def food_categories : List Nat := [40, 10, 30, 20]

/-- The total number of food varieties -/
def total_varieties : Nat := food_categories.sum

/-- The sample size for food safety inspection -/
def sample_size : Nat := 20

/-- Calculates the number of samples for a given category size -/
def stratified_sample (category_size : Nat) : Nat :=
  (sample_size * category_size) / total_varieties

/-- Theorem: The sum of stratified samples from the second and fourth categories is 6 -/
theorem stratified_sample_sum_eq_six :
  stratified_sample (food_categories[1]) + stratified_sample (food_categories[3]) = 6 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_sum_eq_six_l2828_282839


namespace NUMINAMATH_CALUDE_ln2_largest_l2828_282883

theorem ln2_largest (h1 : Real.log ℯ = 1) (h2 : Real.log 2 < 1) (h3 : Real.log 2 > 0) :
  Real.log 2 > (Real.log 2)^2 ∧ Real.log 2 > Real.log (Real.log 2) := by
  sorry

end NUMINAMATH_CALUDE_ln2_largest_l2828_282883


namespace NUMINAMATH_CALUDE_range_of_a_l2828_282871

def A (a : ℝ) : Set ℝ := {x | -2 - a < x ∧ x < a}

theorem range_of_a (a : ℝ) :
  (a > 0) →
  ((1 ∈ A a) ∨ (2 ∈ A a)) ∧
  ¬((1 ∈ A a) ∧ (2 ∈ A a)) →
  1 < a ∧ a ≤ 2 :=
by
  sorry

#check range_of_a

end NUMINAMATH_CALUDE_range_of_a_l2828_282871


namespace NUMINAMATH_CALUDE_system_solution_l2828_282884

theorem system_solution : 
  ∃ (x y : ℚ), (4 * x - 3 * y = 2) ∧ (6 * x + 5 * y = 1) ∧ (x = 13/38) ∧ (y = -4/19) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2828_282884


namespace NUMINAMATH_CALUDE_line_equal_intercepts_l2828_282888

/-- A line with equation ax + y - 2 - a = 0 has equal intercepts on the x-axis and y-axis if and only if a = -2 or a = 1 -/
theorem line_equal_intercepts (a : ℝ) : 
  (∃ (x y : ℝ), a * x + y - 2 - a = 0 ∧ x = y) ↔ (a = -2 ∨ a = 1) :=
sorry

end NUMINAMATH_CALUDE_line_equal_intercepts_l2828_282888


namespace NUMINAMATH_CALUDE_circle_radius_l2828_282832

theorem circle_radius (A C : ℝ) (h : A / C = 15) : 
  ∃ r : ℝ, r > 0 ∧ A = π * r^2 ∧ C = 2 * π * r ∧ r = 30 := by
sorry

end NUMINAMATH_CALUDE_circle_radius_l2828_282832


namespace NUMINAMATH_CALUDE_nested_radical_value_l2828_282887

def nested_radical (x : ℝ) : Prop := x = Real.sqrt (20 + x)

theorem nested_radical_value :
  ∃ x : ℝ, nested_radical x ∧ x = 5 :=
sorry

end NUMINAMATH_CALUDE_nested_radical_value_l2828_282887


namespace NUMINAMATH_CALUDE_exists_arrangement_for_23_l2828_282861

/-- Fibonacci-like sequence defined by F_0 = 0, F_1 = 1, F_i = 3F_{i-1} - F_{i-2} for i ≥ 2 -/
def F : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 3 * F (n + 1) - F n

/-- Theorem stating the existence of a sequence satisfying the required conditions for P = 23 -/
theorem exists_arrangement_for_23 : ∃ (F : ℕ → ℤ), F 0 = 0 ∧ F 1 = 1 ∧ 
  (∀ n : ℕ, n ≥ 2 → F n = 3 * F (n - 1) - F (n - 2)) ∧ F 12 % 23 = 0 :=
sorry

end NUMINAMATH_CALUDE_exists_arrangement_for_23_l2828_282861


namespace NUMINAMATH_CALUDE_profit_maximization_l2828_282807

noncomputable def g (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 30 then
    -5 * x^2 + 420 * x - 3
  else if 30 < x ∧ x ≤ 110 then
    -2 * x - 20000 / (x + 10) + 597
  else
    0

theorem profit_maximization :
  ∃ (x : ℝ), 0 < x ∧ x ≤ 110 ∧
  g x = 9320 ∧
  ∀ (y : ℝ), 0 < y ∧ y ≤ 110 → g y ≤ g x :=
by sorry

end NUMINAMATH_CALUDE_profit_maximization_l2828_282807


namespace NUMINAMATH_CALUDE_z_max_min_l2828_282854

def z (x y : ℝ) : ℝ := 2 * x + y

theorem z_max_min (x y : ℝ) (h1 : x + y ≤ 2) (h2 : x ≥ 1) (h3 : y ≥ 0) :
  (∀ a b : ℝ, a + b ≤ 2 → a ≥ 1 → b ≥ 0 → z a b ≤ 4) ∧
  (∀ a b : ℝ, a + b ≤ 2 → a ≥ 1 → b ≥ 0 → z a b ≥ 2) ∧
  (∃ a b : ℝ, a + b ≤ 2 ∧ a ≥ 1 ∧ b ≥ 0 ∧ z a b = 4) ∧
  (∃ a b : ℝ, a + b ≤ 2 ∧ a ≥ 1 ∧ b ≥ 0 ∧ z a b = 2) :=
by sorry

end NUMINAMATH_CALUDE_z_max_min_l2828_282854


namespace NUMINAMATH_CALUDE_equal_area_division_l2828_282812

-- Define a type for points in a plane
variable (Point : Type)

-- Define a type for lines in a plane
variable (Line : Type)

-- Define a type for figures in a plane
variable (Figure : Type)

-- Function to check if two lines are parallel
variable (parallel : Line → Line → Prop)

-- Function to measure the area of a figure
variable (area : Figure → ℝ)

-- Function to get the part of a figure on one side of a line
variable (figurePart : Figure → Line → Figure)

-- Theorem statement
theorem equal_area_division 
  (Φ : Figure) (l₀ : Line) : 
  ∃ (l : Line), 
    parallel l l₀ ∧ 
    area (figurePart Φ l) = (area Φ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_equal_area_division_l2828_282812


namespace NUMINAMATH_CALUDE_probability_at_least_two_white_l2828_282879

def total_balls : ℕ := 17
def white_balls : ℕ := 8
def black_balls : ℕ := 9
def drawn_balls : ℕ := 3

theorem probability_at_least_two_white :
  (Nat.choose white_balls 2 * black_balls +
   Nat.choose white_balls 3) /
  Nat.choose total_balls drawn_balls =
  154 / 340 := by
sorry

end NUMINAMATH_CALUDE_probability_at_least_two_white_l2828_282879


namespace NUMINAMATH_CALUDE_journey_speed_proof_l2828_282826

/-- Proves that given a journey in three equal parts with speeds 5 km/hr, v km/hr, and 15 km/hr,
    where the total time is 11 minutes and the total distance is 1.5 km, the value of v is 10 km/hr. -/
theorem journey_speed_proof (v : ℝ) : 
  let total_distance : ℝ := 1.5 -- km
  let part_distance : ℝ := total_distance / 3
  let total_time : ℝ := 11 / 60 -- hours
  let time1 : ℝ := part_distance / 5
  let time2 : ℝ := part_distance / v
  let time3 : ℝ := part_distance / 15
  time1 + time2 + time3 = total_time → v = 10 := by sorry

end NUMINAMATH_CALUDE_journey_speed_proof_l2828_282826


namespace NUMINAMATH_CALUDE_window_savings_theorem_l2828_282821

/-- Represents the savings when purchasing windows together vs separately --/
def windowSavings (windowPrice : ℕ) (daveWindows : ℕ) (dougWindows : ℕ) : ℕ :=
  let batchSize := 10
  let freeWindows := 2
  let separateCost := 
    (((daveWindows + batchSize - 1) / batchSize * batchSize - freeWindows) * windowPrice)
    + (((dougWindows + batchSize - 1) / batchSize * batchSize - freeWindows) * windowPrice)
  let jointWindows := daveWindows + dougWindows
  let jointCost := ((jointWindows + batchSize - 1) / batchSize * batchSize - freeWindows * (jointWindows / batchSize)) * windowPrice
  separateCost - jointCost

/-- Theorem stating the savings when Dave and Doug purchase windows together --/
theorem window_savings_theorem : 
  windowSavings 120 9 11 = 120 := by
  sorry

end NUMINAMATH_CALUDE_window_savings_theorem_l2828_282821


namespace NUMINAMATH_CALUDE_orange_pear_weight_balance_l2828_282853

/-- Given that 4 oranges weigh the same as 3 pears, prove that 36 oranges weigh the same as 27 pears -/
theorem orange_pear_weight_balance :
  ∀ (orange_weight pear_weight : ℝ),
  orange_weight > 0 →
  pear_weight > 0 →
  4 * orange_weight = 3 * pear_weight →
  36 * orange_weight = 27 * pear_weight := by
sorry

end NUMINAMATH_CALUDE_orange_pear_weight_balance_l2828_282853


namespace NUMINAMATH_CALUDE_cost_per_person_l2828_282849

theorem cost_per_person (num_friends : ℕ) (total_cost : ℚ) (cost_per_person : ℚ) : 
  num_friends = 15 → 
  total_cost = 13500 → 
  cost_per_person = total_cost / num_friends → 
  cost_per_person = 900 := by
sorry

end NUMINAMATH_CALUDE_cost_per_person_l2828_282849


namespace NUMINAMATH_CALUDE_zachary_pushups_l2828_282830

theorem zachary_pushups (david_pushups : ℕ) (difference : ℕ) (zachary_pushups : ℕ) :
  david_pushups = 37 →
  david_pushups = zachary_pushups + difference →
  difference = 30 →
  zachary_pushups = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_zachary_pushups_l2828_282830


namespace NUMINAMATH_CALUDE_f_monotone_decreasing_no_minimum_l2828_282800

noncomputable def f (x : ℝ) : ℝ := 1 / (x^2 + 1)

theorem f_monotone_decreasing_no_minimum :
  (∀ x y : ℝ, x < y → f x > f y) ∧ 
  (∀ ε > 0, ∃ x : ℝ, f x < ε) :=
by sorry

end NUMINAMATH_CALUDE_f_monotone_decreasing_no_minimum_l2828_282800


namespace NUMINAMATH_CALUDE_interest_rate_proof_l2828_282869

/-- Given simple interest and compound interest for 2 years, prove the interest rate -/
theorem interest_rate_proof (P : ℝ) (R : ℝ) : 
  (2 * P * R / 100 = 600) →  -- Simple interest condition
  (P * ((1 + R / 100)^2 - 1) = 630) →  -- Compound interest condition
  R = 10 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_proof_l2828_282869


namespace NUMINAMATH_CALUDE_max_subsets_with_intersection_property_l2828_282867

/-- The maximum number of distinct subsets satisfying the intersection property -/
theorem max_subsets_with_intersection_property (n : ℕ) :
  (∃ (t : ℕ) (A : Fin t → Finset (Fin n)),
    (∀ i j k, i < j → j < k → (A i ∩ A k) ⊆ A j) ∧
    (∀ i j, i ≠ j → A i ≠ A j)) →
  (∀ (t : ℕ) (A : Fin t → Finset (Fin n)),
    (∀ i j k, i < j → j < k → (A i ∩ A k) ⊆ A j) ∧
    (∀ i j, i ≠ j → A i ≠ A j) →
    t ≤ 2 * n + 1) :=
by sorry

end NUMINAMATH_CALUDE_max_subsets_with_intersection_property_l2828_282867


namespace NUMINAMATH_CALUDE_extrema_not_necessarily_unique_l2828_282876

-- Define a function type
def RealFunction := ℝ → ℝ

-- Define what it means for a point to be an extremum
def IsExtremum (f : RealFunction) (x : ℝ) (a b : ℝ) : Prop :=
  ∀ y ∈ Set.Icc a b, f x ≥ f y ∨ f x ≤ f y

-- Theorem statement
theorem extrema_not_necessarily_unique :
  ∃ (f : RealFunction) (a b x₁ x₂ : ℝ),
    x₁ ≠ x₂ ∧ a < x₁ ∧ x₁ < b ∧ a < x₂ ∧ x₂ < b ∧
    IsExtremum f x₁ a b ∧ IsExtremum f x₂ a b :=
sorry

end NUMINAMATH_CALUDE_extrema_not_necessarily_unique_l2828_282876


namespace NUMINAMATH_CALUDE_sum_of_fractions_l2828_282836

theorem sum_of_fractions : (1 / 1.01) + (1 / 1.1) + (1 / 1) + (1 / 11) + (1 / 101) = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l2828_282836


namespace NUMINAMATH_CALUDE_sum_of_coefficients_is_two_l2828_282882

/-- The polynomial expression -/
def p (x : ℝ) : ℝ := (x - 2)^6 - (x - 1)^7 + (3*x - 2)^8

/-- Theorem: The sum of coefficients of the polynomial p is 2 -/
theorem sum_of_coefficients_is_two : (p 1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_is_two_l2828_282882


namespace NUMINAMATH_CALUDE_initial_books_count_l2828_282892

theorem initial_books_count (action_figures : ℕ) (added_books : ℕ) (difference : ℕ) : 
  action_figures = 7 →
  added_books = 4 →
  difference = 1 →
  ∃ (initial_books : ℕ), 
    initial_books + added_books + difference = action_figures ∧
    initial_books = 2 := by
  sorry

end NUMINAMATH_CALUDE_initial_books_count_l2828_282892


namespace NUMINAMATH_CALUDE_complex_modulus_equality_l2828_282843

theorem complex_modulus_equality (x y : ℝ) :
  (Complex.I + 1) * x = Complex.I * y + 1 →
  Complex.abs (x + Complex.I * y) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_equality_l2828_282843


namespace NUMINAMATH_CALUDE_line_equation_from_intersections_and_midpoint_l2828_282880

/-- The equation of line l given its intersections with two other lines and its midpoint -/
theorem line_equation_from_intersections_and_midpoint 
  (l₁ : Set (ℝ × ℝ)) 
  (l₂ : Set (ℝ × ℝ)) 
  (P : ℝ × ℝ) :
  (∀ x y, (x, y) ∈ l₁ ↔ 4 * x + y + 3 = 0) →
  (∀ x y, (x, y) ∈ l₂ ↔ 3 * x - 5 * y - 5 = 0) →
  P = (-1, 2) →
  ∃ A B : ℝ × ℝ, A ∈ l₁ ∧ B ∈ l₂ ∧ P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
  ∃ l : Set (ℝ × ℝ), (∀ x y, (x, y) ∈ l ↔ 3 * x + y + 1 = 0) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_from_intersections_and_midpoint_l2828_282880


namespace NUMINAMATH_CALUDE_complex_ratio_theorem_l2828_282838

theorem complex_ratio_theorem (z₁ z₂ z₃ : ℂ) 
  (h₁ : Complex.abs z₁ = Real.sqrt 2)
  (h₂ : Complex.abs z₂ = Real.sqrt 2)
  (h₃ : Complex.abs z₃ = Real.sqrt 2) :
  Complex.abs (1 / z₁ + 1 / z₂ + 1 / z₃) / Complex.abs (z₁ + z₂ + z₃) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_ratio_theorem_l2828_282838


namespace NUMINAMATH_CALUDE_total_pencils_l2828_282845

/-- Given that Reeta has 20 pencils and Anika has 4 more than twice the number of pencils as Reeta,
    prove that they have 64 pencils in total. -/
theorem total_pencils (reeta_pencils : ℕ) (anika_pencils : ℕ) : 
  reeta_pencils = 20 →
  anika_pencils = 2 * reeta_pencils + 4 →
  anika_pencils + reeta_pencils = 64 := by
sorry

end NUMINAMATH_CALUDE_total_pencils_l2828_282845


namespace NUMINAMATH_CALUDE_triangle_area_ratio_l2828_282844

theorem triangle_area_ratio (a b c : ℝ) (A : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  0 < A ∧ A < π →
  let S := a^2 - (b-c)^2
  S = (1/2) * b * c * Real.sin A →
  a^2 = b^2 + c^2 - 2*b*c*(Real.cos A) →
  (Real.sin A) / (1 - Real.cos A) = 4 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_ratio_l2828_282844


namespace NUMINAMATH_CALUDE_f_max_at_zero_l2828_282837

-- Define the function f and its derivative
noncomputable def f : ℝ → ℝ := λ x => x^4 - 2*x^2 - 5
def f' : ℝ → ℝ := λ x => 4*x^3 - 4*x

-- State the theorem
theorem f_max_at_zero :
  (∀ x : ℝ, (f' x) = 4*x^3 - 4*x) →
  f 0 = -5 →
  (∀ x : ℝ, f x ≤ -5) ∧ f 0 = -5 :=
by sorry

end NUMINAMATH_CALUDE_f_max_at_zero_l2828_282837


namespace NUMINAMATH_CALUDE_triangle_theorem_l2828_282852

/-- Triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem about the triangle -/
theorem triangle_theorem (t : Triangle) 
  (h1 : t.b^2 + t.c^2 - t.a^2 = t.b * t.c) 
  (h2 : t.a = Real.sqrt 2) 
  (h3 : Real.sin t.B * Real.sin t.C = (Real.sin t.A)^2) :
  t.A = π/3 ∧ t.a + t.b + t.c = 3 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_triangle_theorem_l2828_282852


namespace NUMINAMATH_CALUDE_focus_to_line_distance_l2828_282801

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the line
def line (x y : ℝ) : Prop := Real.sqrt 3 * x - y = 0

-- Define the focus of the parabola
def focus : ℝ × ℝ := (2, 0)

-- State the theorem
theorem focus_to_line_distance :
  let (fx, fy) := focus
  ∃ d : ℝ, d = Real.sqrt 3 ∧
    d = |Real.sqrt 3 * fx - fy| / Real.sqrt (3 + 1) :=
sorry

end NUMINAMATH_CALUDE_focus_to_line_distance_l2828_282801


namespace NUMINAMATH_CALUDE_molecular_weight_N2O5_is_108_l2828_282878

/-- The molecular weight of N2O5 in grams per mole. -/
def molecular_weight_N2O5 : ℝ := 108

/-- The number of moles used in the given condition. -/
def given_moles : ℝ := 10

/-- The total weight of the given number of moles in grams. -/
def given_total_weight : ℝ := 1080

/-- Theorem stating that the molecular weight of N2O5 is 108 grams/mole,
    given that 10 moles of N2O5 weigh 1080 grams. -/
theorem molecular_weight_N2O5_is_108 :
  molecular_weight_N2O5 = given_total_weight / given_moles :=
by sorry

end NUMINAMATH_CALUDE_molecular_weight_N2O5_is_108_l2828_282878


namespace NUMINAMATH_CALUDE_triangle_problem_l2828_282824

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →  -- positive side lengths
  0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2 →  -- acute angles
  A + B + C = π →  -- angles sum to π
  b = Real.sqrt 2 * a * Real.sin B →  -- given condition
  (A = π/4 ∧ 
   (b = Real.sqrt 6 ∧ c = Real.sqrt 3 + 1 → a = 2)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l2828_282824
