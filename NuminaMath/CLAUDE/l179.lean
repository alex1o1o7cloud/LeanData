import Mathlib

namespace NUMINAMATH_CALUDE_simplify_fraction_l179_17996

theorem simplify_fraction : (5^5 + 5^3) / (5^4 - 5^2) = 65 / 12 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l179_17996


namespace NUMINAMATH_CALUDE_range_of_m_l179_17918

/-- Represents the condition for proposition p -/
def is_hyperbola_y_axis (m : ℝ) : Prop :=
  (2 - m < 0) ∧ (m - 1 > 0)

/-- Represents the condition for proposition q -/
def has_no_real_roots (m : ℝ) : Prop :=
  16 * (m - 2)^2 - 16 < 0

/-- The main theorem stating the range of m -/
theorem range_of_m (m : ℝ) 
  (h_p_or_q : is_hyperbola_y_axis m ∨ has_no_real_roots m)
  (h_not_q : ¬has_no_real_roots m) :
  m ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l179_17918


namespace NUMINAMATH_CALUDE_problem_proof_l179_17982

def problem (aunt_gift : ℝ) : Prop :=
  let jade_initial : ℝ := 38
  let julia_initial : ℝ := jade_initial / 2
  let jade_final : ℝ := jade_initial + aunt_gift
  let julia_final : ℝ := julia_initial + aunt_gift
  let total : ℝ := jade_final + julia_final
  total = 57 + 2 * aunt_gift

theorem problem_proof (aunt_gift : ℝ) : problem aunt_gift :=
  sorry

end NUMINAMATH_CALUDE_problem_proof_l179_17982


namespace NUMINAMATH_CALUDE_mean_temperature_l179_17924

def temperatures : List ℝ := [82, 83, 78, 86, 88, 90, 88]

theorem mean_temperature : 
  (List.sum temperatures) / temperatures.length = 84.5714 := by
  sorry

end NUMINAMATH_CALUDE_mean_temperature_l179_17924


namespace NUMINAMATH_CALUDE_parabola_slope_theorem_l179_17979

/-- A parabola with equation y² = 2px, where p > 0 -/
structure Parabola where
  p : ℝ
  h_pos : p > 0

/-- A point on the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Given a parabola and three points on it, prove that the slopes of the lines
    formed by these points satisfy a specific equation -/
theorem parabola_slope_theorem (C : Parabola) (A B P M N : Point) 
  (h_A : A.y^2 = 2 * C.p * A.x) 
  (h_A_x : A.x = 1)
  (h_B : B.y = 0 ∧ B.x = -C.p/2)
  (h_AB : (A.x - B.x)^2 + (A.y - B.y)^2 = 8)
  (h_P : P.y^2 = 2 * C.p * P.x ∧ P.y = 2)
  (h_M : M.y^2 = 2 * C.p * M.x)
  (h_N : N.y^2 = 2 * C.p * N.x)
  (k₁ k₂ k₃ : ℝ)
  (h_k₁ : k₁ = (M.y - P.y) / (M.x - P.x))
  (h_k₂ : k₂ = (N.y - P.y) / (N.x - P.x))
  (h_k₃ : k₃ = (N.y - M.y) / (N.x - M.x)) :
  1/k₁ + 1/k₂ - 1/k₃ = 1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_slope_theorem_l179_17979


namespace NUMINAMATH_CALUDE_compute_expression_l179_17950

theorem compute_expression : 3 * 3^4 - 9^27 / 9^25 = 162 := by sorry

end NUMINAMATH_CALUDE_compute_expression_l179_17950


namespace NUMINAMATH_CALUDE_derivative_at_zero_does_not_exist_l179_17919

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 
  if x = 0 then 0 else Real.sin x * Real.cos (5 / x)

-- State the theorem
theorem derivative_at_zero_does_not_exist :
  ¬ ∃ (L : ℝ), HasDerivAt f L 0 := by sorry

end NUMINAMATH_CALUDE_derivative_at_zero_does_not_exist_l179_17919


namespace NUMINAMATH_CALUDE_valid_seating_arrangements_count_l179_17960

/-- Represents a seating arrangement for two people -/
structure SeatingArrangement :=
  (front : Fin 4 → Bool)
  (back : Fin 5 → Bool)

/-- Checks if a seating arrangement is valid (two people not adjacent) -/
def is_valid (s : SeatingArrangement) : Bool :=
  sorry

/-- Counts the number of valid seating arrangements -/
def count_valid_arrangements : Nat :=
  sorry

/-- Theorem stating that the number of valid seating arrangements is 58 -/
theorem valid_seating_arrangements_count :
  count_valid_arrangements = 58 := by sorry

end NUMINAMATH_CALUDE_valid_seating_arrangements_count_l179_17960


namespace NUMINAMATH_CALUDE_unique_right_triangle_18_l179_17978

/-- Represents a triple of positive integers (a, b, c) that form a right triangle with perimeter 18. -/
structure RightTriangle18 where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  right_triangle : a^2 + b^2 = c^2
  perimeter_18 : a + b + c = 18

/-- There exists exactly one right triangle with integer side lengths and perimeter 18. -/
theorem unique_right_triangle_18 : ∃! t : RightTriangle18, True := by sorry

end NUMINAMATH_CALUDE_unique_right_triangle_18_l179_17978


namespace NUMINAMATH_CALUDE_stock_price_is_102_l179_17985

/-- Given an income, dividend rate, and investment amount, calculate the price of a stock. -/
def stock_price (income : ℚ) (dividend_rate : ℚ) (investment : ℚ) : ℚ :=
  let face_value := income / dividend_rate
  (investment / face_value) * 100

/-- Theorem stating that given the specific conditions, the stock price is 102. -/
theorem stock_price_is_102 :
  stock_price 900 (20 / 100) 4590 = 102 := by
  sorry

#eval stock_price 900 (20 / 100) 4590

end NUMINAMATH_CALUDE_stock_price_is_102_l179_17985


namespace NUMINAMATH_CALUDE_power_two_eq_square_plus_one_solutions_power_two_plus_one_eq_square_solution_l179_17944

theorem power_two_eq_square_plus_one_solutions (x n : ℕ) :
  2^n = x^2 + 1 ↔ (x = 0 ∧ n = 0) ∨ (x = 1 ∧ n = 1) := by sorry

theorem power_two_plus_one_eq_square_solution (x n : ℕ) :
  2^n + 1 = x^2 ↔ x = 3 ∧ n = 3 := by sorry

end NUMINAMATH_CALUDE_power_two_eq_square_plus_one_solutions_power_two_plus_one_eq_square_solution_l179_17944


namespace NUMINAMATH_CALUDE_sin_cos_symmetry_l179_17906

open Real

theorem sin_cos_symmetry :
  ∃ (k : ℤ), (∀ x : ℝ, sin (2 * x - π / 6) = sin (π / 2 - (2 * x - π / 6))) ∧
             (∀ x : ℝ, cos (x - π / 3) = cos (π - (x - π / 3))) ∧
  ¬ ∃ (c : ℝ), (∀ x : ℝ, sin (2 * (x + c) - π / 6) = -sin (2 * (x - c) - π / 6)) ∧
                (∀ x : ℝ, cos ((x + c) - π / 3) = cos ((x - c) - π / 3)) :=
by sorry

end NUMINAMATH_CALUDE_sin_cos_symmetry_l179_17906


namespace NUMINAMATH_CALUDE_product_equals_zero_l179_17907

theorem product_equals_zero (n : ℤ) (h : n = 3) :
  (n - 3) * (n - 2) * (n - 1) * n * (n + 1) * (n + 4) = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_zero_l179_17907


namespace NUMINAMATH_CALUDE_train_length_l179_17965

/-- The length of a train given specific conditions -/
theorem train_length (jogger_speed : ℝ) (train_speed : ℝ) (initial_distance : ℝ) (passing_time : ℝ) :
  jogger_speed = 9 * (1000 / 3600) →
  train_speed = 45 * (1000 / 3600) →
  initial_distance = 230 →
  passing_time = 35 →
  (train_speed - jogger_speed) * passing_time + initial_distance = 580 :=
by sorry

end NUMINAMATH_CALUDE_train_length_l179_17965


namespace NUMINAMATH_CALUDE_y1_greater_than_y2_l179_17942

/-- A linear function f(x) = -3x + 1 -/
def f (x : ℝ) : ℝ := -3 * x + 1

theorem y1_greater_than_y2 (y1 y2 : ℝ) 
  (h1 : f 2 = y1) 
  (h2 : f 3 = y2) : 
  y1 > y2 := by
  sorry

end NUMINAMATH_CALUDE_y1_greater_than_y2_l179_17942


namespace NUMINAMATH_CALUDE_newspaper_pieces_l179_17905

theorem newspaper_pieces (petya_tears : ℕ) (vasya_tears : ℕ) (found_pieces : ℕ) :
  petya_tears = 5 →
  vasya_tears = 9 →
  found_pieces = 1988 →
  ∃ n : ℕ, (1 + n * (petya_tears - 1) + m * (vasya_tears - 1)) ≠ found_pieces :=
by sorry

end NUMINAMATH_CALUDE_newspaper_pieces_l179_17905


namespace NUMINAMATH_CALUDE_basketball_conference_games_l179_17946

/-- The number of divisions in the basketball conference -/
def num_divisions : ℕ := 3

/-- The number of teams in each division -/
def teams_per_division : ℕ := 4

/-- The number of times each team plays other teams in its own division -/
def intra_division_games : ℕ := 3

/-- The number of times each team plays teams from other divisions -/
def inter_division_games : ℕ := 2

/-- The total number of scheduled games in the basketball conference -/
def total_games : ℕ := 150

theorem basketball_conference_games :
  (num_divisions * (teams_per_division.choose 2) * intra_division_games) +
  (num_divisions * teams_per_division * (num_divisions - 1) * teams_per_division * inter_division_games / 2) = total_games :=
by sorry

end NUMINAMATH_CALUDE_basketball_conference_games_l179_17946


namespace NUMINAMATH_CALUDE_triangle_third_angle_l179_17943

theorem triangle_third_angle (A B C : ℝ) (h : A + B = 90) : C = 90 :=
  by
  sorry

end NUMINAMATH_CALUDE_triangle_third_angle_l179_17943


namespace NUMINAMATH_CALUDE_inscribed_rectangles_area_sum_l179_17993

-- Define a rectangle
structure Rectangle where
  a : ℝ
  b : ℝ
  h_positive : a > 0 ∧ b > 0

-- Define an inscribed rectangle
structure InscribedRectangle (R : Rectangle) where
  x : ℝ
  y : ℝ
  h_x_bounds : 0 ≤ x ∧ x ≤ R.a
  h_y_bounds : 0 ≤ y ∧ y ≤ R.b

-- Define the area of a rectangle
def area (R : Rectangle) : ℝ := R.a * R.b

-- Define the area of an inscribed rectangle
def inscribed_area (R : Rectangle) (IR : InscribedRectangle R) : ℝ :=
  IR.x * IR.y + (R.a - IR.x) * (R.b - IR.y)

-- Theorem statement
theorem inscribed_rectangles_area_sum (R : Rectangle) 
  (IR1 IR2 : InscribedRectangle R) (h : IR1.x = IR2.x) :
  inscribed_area R IR1 + inscribed_area R IR2 = area R := by
  sorry

end NUMINAMATH_CALUDE_inscribed_rectangles_area_sum_l179_17993


namespace NUMINAMATH_CALUDE_optimal_purchase_plan_l179_17927

/-- Represents the daily carrying capacity and cost of robots --/
structure Robot where
  capacity : ℕ  -- daily carrying capacity in tons
  cost : ℕ      -- cost in yuan

/-- Represents the purchase plan for robots --/
structure PurchasePlan where
  typeA : ℕ  -- number of type A robots
  typeB : ℕ  -- number of type B robots

/-- Calculates the total daily carrying capacity for a given purchase plan --/
def totalCapacity (a b : Robot) (plan : PurchasePlan) : ℕ :=
  plan.typeA * a.capacity + plan.typeB * b.capacity

/-- Calculates the total cost for a given purchase plan --/
def totalCost (a b : Robot) (plan : PurchasePlan) : ℕ :=
  plan.typeA * a.cost + plan.typeB * b.cost

/-- Theorem stating the optimal purchase plan --/
theorem optimal_purchase_plan (a b : Robot) :
  a.capacity = b.capacity + 20 →
  3 * a.capacity + 2 * b.capacity = 460 →
  a.cost = 30000 →
  b.cost = 20000 →
  (∀ plan : PurchasePlan, plan.typeA + plan.typeB = 20 →
    totalCapacity a b plan ≥ 1820 →
    totalCost a b plan ≥ 510000) ∧
  (∃ plan : PurchasePlan, plan.typeA = 11 ∧ plan.typeB = 9 ∧
    totalCapacity a b plan ≥ 1820 ∧
    totalCost a b plan = 510000) :=
by sorry

end NUMINAMATH_CALUDE_optimal_purchase_plan_l179_17927


namespace NUMINAMATH_CALUDE_inverse_matrices_sum_l179_17989

open Matrix

theorem inverse_matrices_sum (x y z w p q r s : ℝ) : 
  let A : Matrix (Fin 3) (Fin 3) ℝ := !![x, 2, y; 3, 4, 5; z, 6, w]
  let B : Matrix (Fin 3) (Fin 3) ℝ := !![-7, p, -13; q, -15, r; 3, s, 6]
  A * B = 1 → x + y + z + w + p + q + r + s = -5.5 := by
sorry

end NUMINAMATH_CALUDE_inverse_matrices_sum_l179_17989


namespace NUMINAMATH_CALUDE_fraction_subtraction_l179_17939

theorem fraction_subtraction (a b c d x : ℚ) 
  (h1 : a ≠ b) 
  (h2 : b ≠ 0) 
  (h3 : (a - x) / (b - x) = c / d) 
  (h4 : d ≠ c) : 
  x = (b * c - a * d) / (d - c) := by
sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l179_17939


namespace NUMINAMATH_CALUDE_investment_interest_proof_l179_17976

/-- Calculates the total interest earned on an investment -/
def totalInterestEarned (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal * ((1 + rate) ^ years - 1)

/-- Proves that the total interest earned on $2,000 invested at 5% annually for 5 years is $552.56 -/
theorem investment_interest_proof :
  let principal := 2000
  let rate := 0.05
  let years := 5
  ∃ ε > 0, abs (totalInterestEarned principal rate years - 552.56) < ε :=
by sorry

end NUMINAMATH_CALUDE_investment_interest_proof_l179_17976


namespace NUMINAMATH_CALUDE_final_payment_calculation_final_payment_is_861_90_l179_17966

/-- Calculates the final payment amount for a product purchase given specific deposit and discount conditions --/
theorem final_payment_calculation (total_cost : ℝ) (first_deposit : ℝ) (second_deposit : ℝ) 
  (promotional_discount_rate : ℝ) (interest_rate : ℝ) : ℝ :=
  let remaining_balance_before_discount := total_cost - (first_deposit + second_deposit)
  let promotional_discount := total_cost * promotional_discount_rate
  let remaining_balance_after_discount := remaining_balance_before_discount - promotional_discount
  let interest := remaining_balance_after_discount * interest_rate
  remaining_balance_after_discount + interest

/-- Proves that the final payment amount is $861.90 given the specific conditions of the problem --/
theorem final_payment_is_861_90 : 
  let total_cost := 1300
  let first_deposit := 130
  let second_deposit := 260
  let promotional_discount_rate := 0.05
  let interest_rate := 0.02
  (final_payment_calculation total_cost first_deposit second_deposit promotional_discount_rate interest_rate) = 861.90 := by
  sorry

end NUMINAMATH_CALUDE_final_payment_calculation_final_payment_is_861_90_l179_17966


namespace NUMINAMATH_CALUDE_mityas_age_l179_17934

/-- Represents the ages of Mitya and Shura -/
structure Ages where
  mitya : ℕ
  shura : ℕ

/-- The conditions of the problem -/
def problem_conditions (ages : Ages) : Prop :=
  (ages.mitya = ages.shura + 11) ∧
  (ages.shura = 2 * (ages.shura - (ages.mitya - ages.shura)))

/-- The theorem stating Mitya's age -/
theorem mityas_age :
  ∃ (ages : Ages), problem_conditions ages ∧ ages.mitya = 33 :=
sorry

end NUMINAMATH_CALUDE_mityas_age_l179_17934


namespace NUMINAMATH_CALUDE_polynomial_simplification_l179_17911

/-- The given polynomial is equal to its simplified form for all x. -/
theorem polynomial_simplification :
  ∀ x : ℝ, 3 - 5*x - 7*x^2 + 9 + 11*x - 13*x^2 - 15 + 17*x + 19*x^2 + 2*x^3 - 4*x^3 =
            -2*x^3 - x^2 + 23*x - 3 :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l179_17911


namespace NUMINAMATH_CALUDE_rectangle_count_l179_17937

theorem rectangle_count (a : ℝ) (ha : a > 0) : 
  ∃! (x y : ℝ), x < 2*a ∧ y < 2*a ∧ 
  2*(x + y) = 2*((2*a + 3*a) * (2/3)) ∧ 
  x*y = (2*a * 3*a) * (2/9) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_count_l179_17937


namespace NUMINAMATH_CALUDE_max_volume_difference_l179_17967

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a rectangular box -/
def boxVolume (d : BoxDimensions) : ℝ :=
  d.length * d.width * d.height

/-- The measured dimensions of the box -/
def measuredDimensions : BoxDimensions :=
  { length := 150, width := 150, height := 225 }

/-- The maximum error in each measurement -/
def maxError : ℝ := 1

/-- Theorem: The maximum possible difference between the actual capacity
    and the computed capacity of the box is 90726 cubic centimeters -/
theorem max_volume_difference :
  ∃ (actualDimensions : BoxDimensions),
    actualDimensions.length ≤ measuredDimensions.length + maxError ∧
    actualDimensions.length ≥ measuredDimensions.length - maxError ∧
    actualDimensions.width ≤ measuredDimensions.width + maxError ∧
    actualDimensions.width ≥ measuredDimensions.width - maxError ∧
    actualDimensions.height ≤ measuredDimensions.height + maxError ∧
    actualDimensions.height ≥ measuredDimensions.height - maxError ∧
    (boxVolume actualDimensions - boxVolume measuredDimensions) ≤ 90726 ∧
    ∀ (d : BoxDimensions),
      d.length ≤ measuredDimensions.length + maxError →
      d.length ≥ measuredDimensions.length - maxError →
      d.width ≤ measuredDimensions.width + maxError →
      d.width ≥ measuredDimensions.width - maxError →
      d.height ≤ measuredDimensions.height + maxError →
      d.height ≥ measuredDimensions.height - maxError →
      (boxVolume d - boxVolume measuredDimensions) ≤ 90726 :=
by sorry

end NUMINAMATH_CALUDE_max_volume_difference_l179_17967


namespace NUMINAMATH_CALUDE_sqrt_inequality_l179_17955

theorem sqrt_inequality (a : ℝ) (h : a ≥ 2) : 
  Real.sqrt (a + 1) - Real.sqrt a < Real.sqrt (a - 1) - Real.sqrt (a - 2) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l179_17955


namespace NUMINAMATH_CALUDE_total_covered_area_l179_17981

/-- Represents a rectangular strip with length and width -/
structure Strip where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangular strip -/
def Strip.area (s : Strip) : ℝ := s.length * s.width

/-- Calculates the area of overlap between two strips -/
def overlap_area (width : ℝ) (overlap_length : ℝ) : ℝ := width * overlap_length

/-- Theorem: The total area covered by three intersecting strips -/
theorem total_covered_area (s : Strip) (overlap_length : ℝ) : 
  s.length = 12 → s.width = 2 → overlap_length = 2 →
  3 * s.area - 3 * overlap_area s.width overlap_length = 60 := by
  sorry

end NUMINAMATH_CALUDE_total_covered_area_l179_17981


namespace NUMINAMATH_CALUDE_min_set_size_with_mean_constraints_l179_17912

theorem min_set_size_with_mean_constraints (n : ℕ) (S : Finset ℕ) : 
  n > 0 ∧ 
  S.card = n ∧ 
  (∃ m L P : ℕ, 
    L ∈ S ∧ 
    P ∈ S ∧ 
    (∀ x ∈ S, x ≤ L ∧ x ≥ P) ∧
    (S.sum id) / n = m ∧
    m = (2 * L) / 5 ∧ 
    m = (7 * P) / 4) →
  n ≥ 5 :=
by sorry

end NUMINAMATH_CALUDE_min_set_size_with_mean_constraints_l179_17912


namespace NUMINAMATH_CALUDE_passengers_boarded_in_north_carolina_l179_17980

/-- Represents the number of passengers at different stages of the flight --/
structure FlightPassengers where
  initial : Nat
  afterTexas : Nat
  afterNorthCarolina : Nat
  final : Nat

/-- Represents the changes in passenger numbers during layovers --/
structure LayoverChanges where
  texasOff : Nat
  texasOn : Nat
  northCarolinaOff : Nat

/-- The main theorem about the flight --/
theorem passengers_boarded_in_north_carolina 
  (fp : FlightPassengers) 
  (lc : LayoverChanges) 
  (crew : Nat) 
  (h1 : fp.initial = 124)
  (h2 : lc.texasOff = 58)
  (h3 : lc.texasOn = 24)
  (h4 : lc.northCarolinaOff = 47)
  (h5 : crew = 10)
  (h6 : fp.final + crew = 67)
  (h7 : fp.afterTexas = fp.initial - lc.texasOff + lc.texasOn)
  (h8 : fp.afterNorthCarolina = fp.afterTexas - lc.northCarolinaOff)
  : fp.final - fp.afterNorthCarolina = 14 := by
  sorry

#check passengers_boarded_in_north_carolina

end NUMINAMATH_CALUDE_passengers_boarded_in_north_carolina_l179_17980


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l179_17925

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : List ℕ :=
  List.range n |>.map (λ i => a₁ + i * d)

def sum_list (L : List ℕ) : ℕ :=
  L.foldl (· + ·) 0

theorem arithmetic_sequence_sum : 
  2 * (sum_list (arithmetic_sequence 102 2 10)) = 2220 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l179_17925


namespace NUMINAMATH_CALUDE_binomial_expansion_positive_integer_powers_l179_17948

theorem binomial_expansion_positive_integer_powers (x : ℝ) : 
  (Finset.filter (fun r : ℕ => (10 - 3*r) / 2 > 0 ∧ (10 - 3*r) % 2 = 0) (Finset.range 11)).card = 2 :=
sorry

end NUMINAMATH_CALUDE_binomial_expansion_positive_integer_powers_l179_17948


namespace NUMINAMATH_CALUDE_contrapositive_theorem_negation_theorem_l179_17995

def M : Set ℝ := {x | 0 < x ∧ x ≤ 3}
def N : Set ℝ := {x | 0 < x ∧ x ≤ 2}

theorem contrapositive_theorem (a b : ℝ) :
  (a ∈ M → b ∉ M) ↔ (b ∈ M → a ∉ M) :=
sorry

theorem negation_theorem :
  (∃ x : ℝ, x^2 - x - 1 > 0) ↔ ¬(∀ x : ℝ, x^2 - x - 1 ≤ 0) :=
sorry

end NUMINAMATH_CALUDE_contrapositive_theorem_negation_theorem_l179_17995


namespace NUMINAMATH_CALUDE_line_x_intercept_l179_17908

/-- Given a line passing through points (10, 3) and (-4, -4), 
    prove that its x-intercept is 4 -/
theorem line_x_intercept : 
  let p1 : ℝ × ℝ := (10, 3)
  let p2 : ℝ × ℝ := (-4, -4)
  let m : ℝ := (p2.2 - p1.2) / (p2.1 - p1.1)
  let b : ℝ := p1.2 - m * p1.1
  (0 : ℝ) = m * 4 + b :=
by sorry

end NUMINAMATH_CALUDE_line_x_intercept_l179_17908


namespace NUMINAMATH_CALUDE_plane_contains_line_and_parallel_to_intersection_l179_17904

-- Define the line L
def L : Set (ℝ × ℝ × ℝ) :=
  {(x, y, z) | (x - 1) / 2 = -y / 3 ∧ (x - 1) / 2 = 3 - z}

-- Define the two planes
def plane1 : Set (ℝ × ℝ × ℝ) := {(x, y, z) | 4*x + 5*z - 3 = 0}
def plane2 : Set (ℝ × ℝ × ℝ) := {(x, y, z) | 2*x + y + 2*z = 0}

-- Define the plane P we want to prove
def P : Set (ℝ × ℝ × ℝ) := {(x, y, z) | 2*x - y + 7*z - 23 = 0}

-- Theorem statement
theorem plane_contains_line_and_parallel_to_intersection :
  (∀ p ∈ L, p ∈ P) ∧
  (∃ v : ℝ × ℝ × ℝ, v ≠ 0 ∧
    (∀ p q : ℝ × ℝ × ℝ, p ∈ plane1 ∧ q ∈ plane1 ∧ p ∈ plane2 ∧ q ∈ plane2 → 
      ∃ t : ℝ, q - p = t • v) ∧
    (∀ p q : ℝ × ℝ × ℝ, p ∈ P ∧ q ∈ P → 
      ∃ u : ℝ × ℝ × ℝ, u ≠ 0 ∧ q - p = u • v)) :=
by
  sorry

end NUMINAMATH_CALUDE_plane_contains_line_and_parallel_to_intersection_l179_17904


namespace NUMINAMATH_CALUDE_grace_lee_calculation_difference_l179_17932

theorem grace_lee_calculation_difference : 
  (12 - (3 * 4 - 2)) - (12 - 3 * 4 - 2) = -32 := by
  sorry

end NUMINAMATH_CALUDE_grace_lee_calculation_difference_l179_17932


namespace NUMINAMATH_CALUDE_triangle_is_equilateral_l179_17963

/-- Given a triangle ABC with side lengths a, b, c and angles A, B, C, 
    if b^2 + c^2 - bc = a^2 and b/c = tan(B) / tan(C), 
    then the triangle is equilateral. -/
theorem triangle_is_equilateral 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h1 : b^2 + c^2 - b*c = a^2) 
  (h2 : b/c = Real.tan B / Real.tan C) 
  (h3 : 0 < a ∧ 0 < b ∧ 0 < c)
  (h4 : 0 < A ∧ A < π)
  (h5 : 0 < B ∧ B < π)
  (h6 : 0 < C ∧ C < π)
  (h7 : A + B + C = π) :
  a = b ∧ b = c := by
  sorry


end NUMINAMATH_CALUDE_triangle_is_equilateral_l179_17963


namespace NUMINAMATH_CALUDE_bianca_birthday_money_l179_17971

/-- The amount of money Bianca received for her birthday -/
def birthday_money (num_friends : ℕ) (amount_per_friend : ℕ) : ℕ :=
  num_friends * amount_per_friend

/-- Theorem: Bianca received 120 dollars for her birthday -/
theorem bianca_birthday_money :
  birthday_money 8 15 = 120 := by
  sorry

end NUMINAMATH_CALUDE_bianca_birthday_money_l179_17971


namespace NUMINAMATH_CALUDE_solution_exists_l179_17974

def f (x : ℝ) := x^3 + x - 3

theorem solution_exists : ∃ c ∈ Set.Icc 1 2, f c = 0 := by
  sorry

end NUMINAMATH_CALUDE_solution_exists_l179_17974


namespace NUMINAMATH_CALUDE_sequence_inequality_l179_17922

theorem sequence_inequality (a : ℕ → ℕ) 
  (h0 : ∀ n, a n > 0)
  (h1 : a 1 > a 0)
  (h2 : ∀ n, n ≥ 2 ∧ n ≤ 100 → a n = 3 * a (n - 1) - 2 * a (n - 2)) :
  a 100 > 2^99 := by
  sorry

end NUMINAMATH_CALUDE_sequence_inequality_l179_17922


namespace NUMINAMATH_CALUDE_square_area_from_diagonal_l179_17969

/-- The area of a square with a diagonal of 10 meters is 50 square meters. -/
theorem square_area_from_diagonal (d : ℝ) (h : d = 10) : 
  let s := d / Real.sqrt 2
  s ^ 2 = 50 := by sorry

end NUMINAMATH_CALUDE_square_area_from_diagonal_l179_17969


namespace NUMINAMATH_CALUDE_increasing_function_range_l179_17952

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (6 - a) * x - 4 * a else Real.log x / Real.log a

-- State the theorem
theorem increasing_function_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) ↔ (6/5 < a ∧ a < 6) :=
sorry

end NUMINAMATH_CALUDE_increasing_function_range_l179_17952


namespace NUMINAMATH_CALUDE_circle_parabola_intersection_l179_17961

/-- The number of intersection points between a circle and a parabola -/
def intersection_count (b : ℝ) : ℕ :=
  sorry

/-- The curves x^2 + y^2 = b^2 and y = x^2 - b + 1 intersect at exactly 4 points
    if and only if b > 2 -/
theorem circle_parabola_intersection (b : ℝ) :
  intersection_count b = 4 ↔ b > 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_parabola_intersection_l179_17961


namespace NUMINAMATH_CALUDE_fruit_combinations_l179_17914

theorem fruit_combinations (n r : ℕ) (h1 : n = 5) (h2 : r = 2) :
  (n + r - 1).choose r = 15 := by
sorry

end NUMINAMATH_CALUDE_fruit_combinations_l179_17914


namespace NUMINAMATH_CALUDE_lighter_ball_problem_l179_17909

/-- Represents the maximum number of balls that can be checked in a given number of weighings -/
def max_balls (weighings : ℕ) : ℕ := 3^weighings

/-- The problem statement -/
theorem lighter_ball_problem (n : ℕ) :
  (∀ m : ℕ, m > n → max_balls 5 < m) →
  (∃ strategy : Unit, true) →  -- placeholder for the existence of a strategy
  n ≤ max_balls 5 :=
sorry

end NUMINAMATH_CALUDE_lighter_ball_problem_l179_17909


namespace NUMINAMATH_CALUDE_first_divisor_l179_17991

theorem first_divisor (k : ℕ) (h1 : k > 0) (h2 : k % 5 = 2) (h3 : k % 6 = 5) (h4 : k % 7 = 3) (h5 : k < 42) :
  min 5 (min 6 7) = 5 :=
by sorry

end NUMINAMATH_CALUDE_first_divisor_l179_17991


namespace NUMINAMATH_CALUDE_f_minimum_l179_17951

/-- The function to be minimized -/
def f (x y : ℝ) : ℝ := x^2 - 2*x*y + 6*y^2 - 14*x - 6*y + 72

/-- Theorem stating that f attains its minimum at (15/2, 1/2) -/
theorem f_minimum : 
  ∀ (x y : ℝ), f x y ≥ f (15/2) (1/2) := by sorry

end NUMINAMATH_CALUDE_f_minimum_l179_17951


namespace NUMINAMATH_CALUDE_candy_boxes_total_l179_17921

theorem candy_boxes_total (x y z : ℕ) : 
  x = y / 2 → 
  x + z = 24 → 
  y + z = 34 → 
  x + y + z = 44 :=
by
  sorry

end NUMINAMATH_CALUDE_candy_boxes_total_l179_17921


namespace NUMINAMATH_CALUDE_problem_stack_total_logs_l179_17901

/-- Represents a stack of logs -/
structure LogStack where
  bottomRowCount : ℕ
  topRowCount : ℕ
  rowDifference : ℕ

/-- Calculates the total number of logs in the stack -/
def totalLogs (stack : LogStack) : ℕ :=
  sorry

/-- The specific log stack described in the problem -/
def problemStack : LogStack :=
  { bottomRowCount := 20
  , topRowCount := 4
  , rowDifference := 2 }

theorem problem_stack_total_logs :
  totalLogs problemStack = 108 := by
  sorry

end NUMINAMATH_CALUDE_problem_stack_total_logs_l179_17901


namespace NUMINAMATH_CALUDE_workers_days_per_week_l179_17913

/-- The number of toys produced per week -/
def toys_per_week : ℕ := 5505

/-- The number of toys produced per day -/
def toys_per_day : ℕ := 1101

/-- The number of days worked in a week -/
def days_worked : ℕ := toys_per_week / toys_per_day

theorem workers_days_per_week :
  days_worked = 5 :=
by sorry

end NUMINAMATH_CALUDE_workers_days_per_week_l179_17913


namespace NUMINAMATH_CALUDE_product_of_four_integers_l179_17998

theorem product_of_four_integers (P Q R S : ℕ+) : 
  (P : ℚ) + (Q : ℚ) + (R : ℚ) + (S : ℚ) = 50 →
  (P : ℚ) + 4 = (Q : ℚ) - 4 ∧ 
  (P : ℚ) + 4 = (R : ℚ) * 3 ∧ 
  (P : ℚ) + 4 = (S : ℚ) / 3 →
  (P : ℚ) * (Q : ℚ) * (R : ℚ) * (S : ℚ) = (43 * 107 * 75 * 225) / 1536 := by
  sorry

#check product_of_four_integers

end NUMINAMATH_CALUDE_product_of_four_integers_l179_17998


namespace NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l179_17997

theorem arithmetic_sequence_middle_term : ∀ (a b c : ℤ),
  (a = 2^2 ∧ c = 2^4 ∧ b - a = c - b) → b = 10 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l179_17997


namespace NUMINAMATH_CALUDE_sum_of_binary_digits_345_l179_17900

/-- Returns the binary representation of a natural number as a list of bits -/
def toBinary (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec go (m : ℕ) : List ℕ :=
    if m = 0 then [] else (m % 2) :: go (m / 2)
  go n

/-- Sums the elements of a list of natural numbers -/
def sumList (l : List ℕ) : ℕ :=
  l.foldl (· + ·) 0

theorem sum_of_binary_digits_345 : sumList (toBinary 345) = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_binary_digits_345_l179_17900


namespace NUMINAMATH_CALUDE_tangent_line_hyperbola_l179_17956

/-- The equation of the tangent line to the hyperbola x^2 - y^2/2 = 1 at the point (√2, √2) is 2x - y - √2 = 0 -/
theorem tangent_line_hyperbola (x y : ℝ) :
  (x^2 - y^2/2 = 1) →
  let P : ℝ × ℝ := (Real.sqrt 2, Real.sqrt 2)
  let tangent_line := fun (x y : ℝ) ↦ 2*x - y - Real.sqrt 2 = 0
  (x = P.1 ∧ y = P.2) →
  tangent_line x y :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_hyperbola_l179_17956


namespace NUMINAMATH_CALUDE_min_value_f_l179_17984

/-- The function f(x) = 12x - x³ -/
def f (x : ℝ) : ℝ := 12 * x - x^3

/-- The theorem stating that the minimum value of f(x) on [-3, 3] is -16 -/
theorem min_value_f : 
  ∃ (x₀ : ℝ), x₀ ∈ Set.Icc (-3) 3 ∧ 
  (∀ (x : ℝ), x ∈ Set.Icc (-3) 3 → f x ≥ f x₀) ∧
  f x₀ = -16 := by
  sorry


end NUMINAMATH_CALUDE_min_value_f_l179_17984


namespace NUMINAMATH_CALUDE_reach_probability_l179_17968

-- Define the type for a point in the coordinate plane
structure Point where
  x : Int
  y : Int

-- Define the type for a step direction
inductive Direction
  | Left
  | Right
  | Up
  | Down

-- Define the function to calculate the probability
def probability_reach_target (start : Point) (target : Point) (max_steps : Nat) : Rat :=
  sorry

-- Theorem statement
theorem reach_probability :
  probability_reach_target ⟨0, 0⟩ ⟨2, 3⟩ 7 = 179 / 8192 := by sorry

end NUMINAMATH_CALUDE_reach_probability_l179_17968


namespace NUMINAMATH_CALUDE_not_necessarily_linear_l179_17910

open Set MeasureTheory

-- Define the type of real-valued functions
def RealFunction := ℝ → ℝ

-- Define the Minkowski sum of graphs
def minkowskiSumGraphs (f g : RealFunction) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ x y : ℝ, p = (x + y, f x + g y)}

-- State the theorem
theorem not_necessarily_linear :
  ∃ (f g : RealFunction),
    Continuous f ∧
    (volume (minkowskiSumGraphs f g) = 0) ∧
    ¬∃ (a b : ℝ), ∀ x, f x = a * x + b :=
by sorry

end NUMINAMATH_CALUDE_not_necessarily_linear_l179_17910


namespace NUMINAMATH_CALUDE_absolute_difference_of_roots_l179_17986

-- Define the quadratic equation
def quadratic_equation (k : ℝ) (x : ℝ) : ℝ := x^2 - (k+3)*x + k

-- Define the roots of the quadratic equation
def roots (k : ℝ) : ℝ × ℝ := sorry

-- Theorem statement
theorem absolute_difference_of_roots (k : ℝ) :
  let (r₁, r₂) := roots k
  |r₁ - r₂| = Real.sqrt (k^2 + 2*k + 9) := by sorry

end NUMINAMATH_CALUDE_absolute_difference_of_roots_l179_17986


namespace NUMINAMATH_CALUDE_coeff_4th_term_of_1_minus_2x_to_15_l179_17988

/-- The coefficient of the 4th term in the expansion of (1-2x)^15 -/
def coeff_4th_term : ℤ := -3640

/-- The binomial coefficient (n choose k) -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem coeff_4th_term_of_1_minus_2x_to_15 :
  coeff_4th_term = (-2)^3 * (binomial 15 3) := by sorry

end NUMINAMATH_CALUDE_coeff_4th_term_of_1_minus_2x_to_15_l179_17988


namespace NUMINAMATH_CALUDE_hyperbola_satisfies_conditions_l179_17933

/-- A hyperbola is defined by its equation and properties -/
structure Hyperbola where
  equation : ℝ → ℝ → Prop
  passes_through : ℝ × ℝ
  asymptotes : ℝ → ℝ → Prop

/-- The given hyperbola with equation x²/2 - y² = 1 -/
def given_hyperbola : Hyperbola where
  equation := fun x y => x^2 / 2 - y^2 = 1
  passes_through := (2, -2)
  asymptotes := fun x y => x^2 / 2 - y^2 = 0

/-- The hyperbola we need to prove -/
def our_hyperbola : Hyperbola where
  equation := fun x y => y^2 / 2 - x^2 / 4 = 1
  passes_through := (2, -2)
  asymptotes := fun x y => x^2 / 2 - y^2 = 0

/-- Theorem stating that our_hyperbola satisfies the required conditions -/
theorem hyperbola_satisfies_conditions :
  (our_hyperbola.equation our_hyperbola.passes_through.1 our_hyperbola.passes_through.2) ∧
  (∀ x y, our_hyperbola.asymptotes x y ↔ given_hyperbola.asymptotes x y) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_satisfies_conditions_l179_17933


namespace NUMINAMATH_CALUDE_decimal_places_product_specific_case_l179_17931

/-- Given two real numbers a and b, this function returns the number of decimal places in their product. -/
def decimal_places_in_product (a b : ℝ) : ℕ :=
  sorry

/-- This function returns the number of decimal places in a real number. -/
def count_decimal_places (x : ℝ) : ℕ :=
  sorry

theorem decimal_places_product (a b : ℝ) :
  decimal_places_in_product a b = count_decimal_places a + count_decimal_places b :=
sorry

theorem specific_case : 
  decimal_places_in_product 0.38 0.26 = 4 :=
sorry

end NUMINAMATH_CALUDE_decimal_places_product_specific_case_l179_17931


namespace NUMINAMATH_CALUDE_nail_polish_theorem_l179_17945

def nail_polish_problem (kim heidi karen : ℕ) : Prop :=
  kim = 25 ∧
  heidi = kim + 8 ∧
  karen = kim - 6 ∧
  heidi + karen = 52

theorem nail_polish_theorem :
  ∃ (kim heidi karen : ℕ), nail_polish_problem kim heidi karen := by
  sorry

end NUMINAMATH_CALUDE_nail_polish_theorem_l179_17945


namespace NUMINAMATH_CALUDE_max_gcd_sum_1729_l179_17992

theorem max_gcd_sum_1729 :
  ∃ (x y : ℕ+), x + y = 1729 ∧ 
  ∀ (a b : ℕ+), a + b = 1729 → Nat.gcd x y ≥ Nat.gcd a b ∧
  Nat.gcd x y = 247 :=
sorry

end NUMINAMATH_CALUDE_max_gcd_sum_1729_l179_17992


namespace NUMINAMATH_CALUDE_quadratic_completion_l179_17999

/-- The quadratic function we're working with -/
def f (x : ℝ) : ℝ := x^2 - 24*x + 50

/-- The completed square form of our quadratic -/
def g (x b c : ℝ) : ℝ := (x + b)^2 + c

/-- Theorem stating that f can be written in the form of g, and b + c = -106 -/
theorem quadratic_completion (b c : ℝ) : 
  (∀ x, f x = g x b c) → b + c = -106 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_completion_l179_17999


namespace NUMINAMATH_CALUDE_point_d_and_k_value_l179_17920

/-- Given four points in a plane, prove the coordinates of D and the value of k. -/
theorem point_d_and_k_value 
  (A B C D : ℝ × ℝ)
  (hA : A = (1, 3))
  (hB : B = (2, -2))
  (hC : C = (4, 1))
  (h_AB_CD : B - A = D - C)
  (a b : ℝ × ℝ)
  (ha : a = B - A)
  (hb : b = C - B)
  (h_parallel : ∃ (t : ℝ), t ≠ 0 ∧ t • (k • a - b) = a + 3 • b) :
  D = (5, -4) ∧ k = -1/3 := by sorry

end NUMINAMATH_CALUDE_point_d_and_k_value_l179_17920


namespace NUMINAMATH_CALUDE_may_day_travel_scientific_notation_l179_17964

/-- Expresses a number in scientific notation -/
def scientific_notation (n : ℝ) : ℝ × ℤ :=
  sorry

theorem may_day_travel_scientific_notation :
  scientific_notation (56.99 * 1000000) = (5.699, 7) :=
sorry

end NUMINAMATH_CALUDE_may_day_travel_scientific_notation_l179_17964


namespace NUMINAMATH_CALUDE_intersection_and_complement_union_condition_implies_m_range_l179_17941

-- Define the sets
def U : Set ℝ := {x | 1 < x ∧ x < 7}
def A1 : Set ℝ := {x | 2 ≤ x ∧ x < 5}
def B1 : Set ℝ := {x | 3*x - 7 ≥ 8 - 2*x}

def A2 : Set ℝ := {x | -2 ≤ x ∧ x ≤ 7}
def B2 (m : ℝ) : Set ℝ := {x | m + 1 < x ∧ x < 2*m - 1}

-- Theorem for the first part
theorem intersection_and_complement :
  (A1 ∩ B1 = {x | 3 ≤ x ∧ x < 5}) ∧
  (U \ A1 = {x | (1 < x ∧ x < 2) ∨ (5 ≤ x ∧ x < 7)}) :=
sorry

-- Theorem for the second part
theorem union_condition_implies_m_range :
  ∀ m, (A2 ∪ B2 m = A2) → m ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_intersection_and_complement_union_condition_implies_m_range_l179_17941


namespace NUMINAMATH_CALUDE_villa_tournament_correct_l179_17954

/-- A tournament where each player plays with a fixed number of other players. -/
structure Tournament where
  num_players : ℕ
  games_per_player : ℕ
  total_games : ℕ

/-- The specific tournament described in the problem. -/
def villa_tournament : Tournament :=
  { num_players := 6,
    games_per_player := 4,
    total_games := 10 }

/-- Theorem stating that the total number of games in the Villa tournament is correct. -/
theorem villa_tournament_correct :
  villa_tournament.total_games = (villa_tournament.num_players * villa_tournament.games_per_player) / 2 :=
by sorry

end NUMINAMATH_CALUDE_villa_tournament_correct_l179_17954


namespace NUMINAMATH_CALUDE_f_prime_at_i_l179_17928

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the function f
def f (x : ℂ) : ℂ := x^4 - x^2

-- State the theorem
theorem f_prime_at_i : 
  (deriv f) i = -6 * i := by sorry

end NUMINAMATH_CALUDE_f_prime_at_i_l179_17928


namespace NUMINAMATH_CALUDE_paint_mixture_intensity_l179_17972

/-- Calculates the intensity of a paint mixture when a fraction of the original paint is replaced with a different paint. -/
def mixedPaintIntensity (originalIntensity addedIntensity : ℝ) (fractionReplaced : ℝ) : ℝ :=
  originalIntensity * (1 - fractionReplaced) + addedIntensity * fractionReplaced

/-- Theorem stating that mixing 45% intensity paint with 25% intensity paint in a 3:1 ratio results in 40% intensity paint. -/
theorem paint_mixture_intensity :
  let originalIntensity : ℝ := 0.45
  let addedIntensity : ℝ := 0.25
  let fractionReplaced : ℝ := 0.25
  mixedPaintIntensity originalIntensity addedIntensity fractionReplaced = 0.40 := by
  sorry

end NUMINAMATH_CALUDE_paint_mixture_intensity_l179_17972


namespace NUMINAMATH_CALUDE_evaluate_expression_l179_17929

theorem evaluate_expression (x y : ℝ) (hx : x = 2) (hy : y = 4) : y * (2 * y - x) = 24 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l179_17929


namespace NUMINAMATH_CALUDE_square_difference_equality_l179_17902

theorem square_difference_equality : (43 + 15)^2 - (43^2 + 15^2) = 1290 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equality_l179_17902


namespace NUMINAMATH_CALUDE_polygon_interior_exterior_angles_equal_l179_17977

theorem polygon_interior_exterior_angles_equal (n : ℕ) : 
  (n - 2) * 180 = 360 → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_polygon_interior_exterior_angles_equal_l179_17977


namespace NUMINAMATH_CALUDE_geometric_sequence_308th_term_l179_17994

theorem geometric_sequence_308th_term
  (a₁ : ℝ)
  (a₂ : ℝ)
  (h₁ : a₁ = 10)
  (h₂ : a₂ = -10) :
  let r := a₂ / a₁
  let aₙ := a₁ * r^(308 - 1)
  aₙ = -10 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_308th_term_l179_17994


namespace NUMINAMATH_CALUDE_raines_change_l179_17958

/-- Calculates the change Raine receives after purchasing items from a gift shop --/
theorem raines_change (bracelet_price necklace_price mug_price : ℕ)
  (bracelet_count necklace_count mug_count : ℕ)
  (payment : ℕ)
  (h1 : bracelet_price = 15)
  (h2 : necklace_price = 10)
  (h3 : mug_price = 20)
  (h4 : bracelet_count = 3)
  (h5 : necklace_count = 2)
  (h6 : mug_count = 1)
  (h7 : payment = 100) :
  payment - (bracelet_price * bracelet_count + necklace_price * necklace_count + mug_price * mug_count) = 15 := by
  sorry

#check raines_change

end NUMINAMATH_CALUDE_raines_change_l179_17958


namespace NUMINAMATH_CALUDE_total_amount_is_fifteen_l179_17903

/-- Represents the share of each person in Rupees -/
structure Share where
  w : ℚ
  x : ℚ
  y : ℚ

/-- The total amount of the sum -/
def total_amount (s : Share) : ℚ :=
  s.w + s.x + s.y

/-- The condition that for each rupee w gets, x gets 30 paisa and y gets 20 paisa -/
def share_ratio (s : Share) : Prop :=
  s.x = (3/10) * s.w ∧ s.y = (1/5) * s.w

/-- The theorem stating that if w's share is 10 rupees and the share ratio is maintained,
    then the total amount is 15 rupees -/
theorem total_amount_is_fifteen (s : Share) 
    (h1 : s.w = 10)
    (h2 : share_ratio s) : 
    total_amount s = 15 := by
  sorry


end NUMINAMATH_CALUDE_total_amount_is_fifteen_l179_17903


namespace NUMINAMATH_CALUDE_larger_number_proof_l179_17940

theorem larger_number_proof (L S : ℕ) (h1 : L > S) (h2 : L - S = 1515) (h3 : L = 16 * S + 15) : L = 1617 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l179_17940


namespace NUMINAMATH_CALUDE_tan_roots_and_angle_sum_cosine_product_l179_17936

theorem tan_roots_and_angle_sum_cosine_product
  (α β : Real)
  (h1 : ∀ x, x^2 + 3 * Real.sqrt 3 * x + 4 = 0 ↔ x = Real.tan α ∨ x = Real.tan β)
  (h2 : α ∈ Set.Ioo (-π/2) (π/2))
  (h3 : β ∈ Set.Ioo (-π/2) (π/2)) :
  (α + β = -2*π/3) ∧ (Real.cos α * Real.cos β = 1/6) := by
sorry

end NUMINAMATH_CALUDE_tan_roots_and_angle_sum_cosine_product_l179_17936


namespace NUMINAMATH_CALUDE_treatment_volume_is_120_ml_l179_17953

/-- Calculates the total volume of treatment received from a saline drip. -/
def total_treatment_volume (drops_per_minute : ℕ) (treatment_hours : ℕ) (ml_per_100_drops : ℕ) : ℕ :=
  let minutes_per_hour : ℕ := 60
  let drops_per_100 : ℕ := 100
  let total_minutes : ℕ := treatment_hours * minutes_per_hour
  let total_drops : ℕ := drops_per_minute * total_minutes
  (total_drops * ml_per_100_drops) / drops_per_100

/-- The theorem stating that the total treatment volume is 120 ml under given conditions. -/
theorem treatment_volume_is_120_ml :
  total_treatment_volume 20 2 5 = 120 :=
by
  sorry

#eval total_treatment_volume 20 2 5

end NUMINAMATH_CALUDE_treatment_volume_is_120_ml_l179_17953


namespace NUMINAMATH_CALUDE_root_triple_relation_l179_17957

theorem root_triple_relation (p q r : ℝ) (h : ∃ x y : ℝ, p * x^2 + q * x + r = 0 ∧ p * y^2 + q * y + r = 0 ∧ y = 3 * x) :
  3 * q^2 = 8 * p * r := by
sorry

end NUMINAMATH_CALUDE_root_triple_relation_l179_17957


namespace NUMINAMATH_CALUDE_acacia_arrangement_probability_l179_17947

/-- The number of fir trees -/
def num_fir : ℕ := 4

/-- The number of pine trees -/
def num_pine : ℕ := 5

/-- The number of acacia trees -/
def num_acacia : ℕ := 6

/-- The total number of trees -/
def total_trees : ℕ := num_fir + num_pine + num_acacia

/-- The probability of no two acacia trees being next to each other -/
def prob_no_adjacent_acacia : ℚ := 84 / 159

theorem acacia_arrangement_probability :
  let total_arrangements := Nat.choose total_trees num_acacia
  let valid_arrangements := Nat.choose (num_fir + num_pine + 1) num_acacia * Nat.choose (num_fir + num_pine) num_fir
  (valid_arrangements : ℚ) / total_arrangements = prob_no_adjacent_acacia := by
  sorry

end NUMINAMATH_CALUDE_acacia_arrangement_probability_l179_17947


namespace NUMINAMATH_CALUDE_stratified_sampling_l179_17930

theorem stratified_sampling (total_employees : ℕ) (total_sample : ℕ) (dept_employees : ℕ) :
  total_employees = 240 →
  total_sample = 20 →
  dept_employees = 60 →
  (dept_employees * total_sample) / total_employees = 5 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_l179_17930


namespace NUMINAMATH_CALUDE_infinitely_many_n_squared_plus_one_divides_factorial_l179_17938

/-- The set of positive integers n for which n^2 + 1 divides n! is infinite -/
theorem infinitely_many_n_squared_plus_one_divides_factorial :
  Set.Infinite {n : ℕ+ | (n^2 + 1) ∣ n!} := by sorry

end NUMINAMATH_CALUDE_infinitely_many_n_squared_plus_one_divides_factorial_l179_17938


namespace NUMINAMATH_CALUDE_reciprocal_and_absolute_value_l179_17975

theorem reciprocal_and_absolute_value :
  (1 / (- (-2))) = 1/2 ∧ 
  {x : ℝ | |x| = 5} = {-5, 5} := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_and_absolute_value_l179_17975


namespace NUMINAMATH_CALUDE_sin_product_zero_l179_17990

theorem sin_product_zero : Real.sin (12 * π / 180) * Real.sin (36 * π / 180) * Real.sin (60 * π / 180) * Real.sin (84 * π / 180) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sin_product_zero_l179_17990


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l179_17915

theorem polynomial_divisibility (A B : ℝ) : 
  (∀ x : ℂ, x^2 + x + 1 = 0 → x^103 + A*x^2 + B = 0) → 
  A + B = 2 := by
sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l179_17915


namespace NUMINAMATH_CALUDE_negation_of_existential_proposition_l179_17949

theorem negation_of_existential_proposition :
  (¬ ∃ x : ℤ, x^2 + 2*x - 1 < 0) ↔ (∀ x : ℤ, x^2 + 2*x - 1 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existential_proposition_l179_17949


namespace NUMINAMATH_CALUDE_complex_equation_sum_l179_17959

theorem complex_equation_sum (a b : ℝ) (i : ℂ) (h : i * i = -1) :
  (1 - 2*i) * (2 + a*i) = b - 2*i → a + b = 8 :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l179_17959


namespace NUMINAMATH_CALUDE_fireworks_per_letter_l179_17917

/-- The number of fireworks needed to display a digit --/
def fireworks_per_digit : ℕ := 6

/-- The number of digits in the year display --/
def year_digits : ℕ := 4

/-- The number of letters in "HAPPY NEW YEAR" --/
def phrase_letters : ℕ := 12

/-- The number of additional boxes of fireworks --/
def additional_boxes : ℕ := 50

/-- The number of fireworks in each box --/
def fireworks_per_box : ℕ := 8

/-- The total number of fireworks lit during the display --/
def total_fireworks : ℕ := 484

/-- Theorem: The number of fireworks needed to display a letter is 5 --/
theorem fireworks_per_letter :
  ∃ (x : ℕ), 
    x * phrase_letters + 
    fireworks_per_digit * year_digits + 
    additional_boxes * fireworks_per_box = 
    total_fireworks ∧ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_fireworks_per_letter_l179_17917


namespace NUMINAMATH_CALUDE_hexagon_diagonals_l179_17983

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A hexagon has 6 sides -/
def hexagon_sides : ℕ := 6

/-- Theorem: The number of diagonals in a hexagon is 9 -/
theorem hexagon_diagonals : num_diagonals hexagon_sides = 9 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_diagonals_l179_17983


namespace NUMINAMATH_CALUDE_task_completion_rate_l179_17926

/-- Given two people A and B who can complete a task in x and y days respectively,
    this theorem proves that together they can complete a fraction of 1/x + 1/y of the task in one day. -/
theorem task_completion_rate (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (1 : ℝ) / x + (1 : ℝ) / y = (x + y) / (x * y) := by
  sorry


end NUMINAMATH_CALUDE_task_completion_rate_l179_17926


namespace NUMINAMATH_CALUDE_rectangle_width_is_five_l179_17970

/-- Represents the dimensions of a rectangle --/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of a rectangle --/
def perimeter (r : Rectangle) : ℝ :=
  2 * (r.length + r.width)

/-- Theorem: A rectangle with length 6 and perimeter 22 has width 5 --/
theorem rectangle_width_is_five :
  ∀ r : Rectangle, r.length = 6 → perimeter r = 22 → r.width = 5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_width_is_five_l179_17970


namespace NUMINAMATH_CALUDE_sales_profit_equation_max_profit_selling_price_range_l179_17923

-- Define the cost to produce each item
def production_cost : ℝ := 50

-- Define the daily sales volume as a function of price
def sales_volume (x : ℝ) : ℝ := 50 + 5 * (100 - x)

-- Define the daily sales profit function
def sales_profit (x : ℝ) : ℝ := (x - production_cost) * sales_volume x

-- Theorem 1: The daily sales profit function
theorem sales_profit_equation (x : ℝ) :
  sales_profit x = -5 * x^2 + 800 * x - 27500 := by sorry

-- Theorem 2: The maximum daily sales profit
theorem max_profit :
  ∃ (x : ℝ), x = 80 ∧ sales_profit x = 4500 ∧
  ∀ (y : ℝ), 50 ≤ y ∧ y ≤ 100 → sales_profit y ≤ sales_profit x := by sorry

-- Theorem 3: The range of selling prices satisfying the conditions
theorem selling_price_range :
  ∀ (x : ℝ), (sales_profit x ≥ 4000 ∧ production_cost * sales_volume x ≤ 7000) ↔
  (82 ≤ x ∧ x ≤ 90) := by sorry

end NUMINAMATH_CALUDE_sales_profit_equation_max_profit_selling_price_range_l179_17923


namespace NUMINAMATH_CALUDE_probability_divisor_of_12_on_8_sided_die_l179_17916

def is_divisor_of_12 (n : ℕ) : Prop := 12 % n = 0

def die_sides : ℕ := 8

def favorable_outcomes : Finset ℕ := {1, 2, 3, 4, 6}

theorem probability_divisor_of_12_on_8_sided_die :
  (favorable_outcomes.card : ℚ) / die_sides = 5 / 8 :=
sorry

end NUMINAMATH_CALUDE_probability_divisor_of_12_on_8_sided_die_l179_17916


namespace NUMINAMATH_CALUDE_system_solution_range_l179_17962

theorem system_solution_range (a x y : ℝ) : 
  (5 * x + 2 * y = 11 * a + 18) →
  (2 * x - 3 * y = 12 * a - 8) →
  (x > 0) →
  (y > 0) →
  (-2/3 < a ∧ a < 2) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_range_l179_17962


namespace NUMINAMATH_CALUDE_expression_satisfies_equation_l179_17935

theorem expression_satisfies_equation (x : ℝ) (E : ℝ → ℝ) : 
  x = 4 → (7 * E x = 21) → E = fun y ↦ y - 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_satisfies_equation_l179_17935


namespace NUMINAMATH_CALUDE_min_rectangle_side_l179_17973

/-- Given a rectangle with one side of length 1, divided into four smaller rectangles
    by two perpendicular lines, where three of the smaller rectangles have areas of
    at least 1 and the fourth has an area of at least 2, the minimum length of the
    other side of the original rectangle is 3 + 2√2. -/
theorem min_rectangle_side (a b c d : ℝ) : 
  a + b = 1 →
  a * c ≥ 1 →
  a * d ≥ 1 →
  b * c ≥ 1 →
  b * d ≥ 2 →
  c + d ≥ 3 + 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_rectangle_side_l179_17973


namespace NUMINAMATH_CALUDE_arcsin_equation_solution_l179_17987

theorem arcsin_equation_solution :
  ∃ x : ℝ, x = Real.sqrt 102 / 51 ∧ 
    Real.arcsin x + Real.arcsin (3 * x) = π / 4 ∧
    -1 < x ∧ x < 1 ∧ -1 < 3 * x ∧ 3 * x < 1 :=
by sorry

end NUMINAMATH_CALUDE_arcsin_equation_solution_l179_17987
