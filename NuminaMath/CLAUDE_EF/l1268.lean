import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_at_40_l1268_126865

/-- The power consumption function for an electric bicycle -/
noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - (39/2) * x^2 - 40 * x

/-- Theorem stating that f has a minimum at x = 40 for x > 0 -/
theorem f_min_at_40 :
  ∀ x > 0, f x ≥ f 40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_at_40_l1268_126865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l1268_126814

def parabola_equation (x y : ℝ) : ℝ := 2 * y^2 - x - 24 * y + 72

theorem parabola_properties :
  -- The parabola passes through the point (2,7)
  parabola_equation 2 7 = 0 ∧
  -- The y-coordinate of the focus is 6
  ∃ (x : ℝ), parabola_equation x 6 = 0 ∧
  -- Its axis of symmetry is parallel to the x-axis
  ∀ (y : ℝ), ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ parabola_equation x1 y = 0 ∧ parabola_equation x2 y = 0 ∧
  -- Its vertex lies on the y-axis
  ∃ (y : ℝ), parabola_equation 0 y = 0 ∧
  -- The equation is in the form ax^2 + bxy + cy^2 + dx + ey + f = 0
  -- where a, b, c, d, e, f are integers, c is positive, and gcd(|a|,|b|,|c|,|d|,|e|,|f|) = 1
  ∃ (a b c d e f : ℤ), c > 0 ∧
    (∀ (x y : ℝ), a * x^2 + b * x * y + c * y^2 + d * x + e * y + f = parabola_equation x y) ∧
    Nat.gcd (Int.natAbs a) (Nat.gcd (Int.natAbs b) (Nat.gcd (Int.natAbs c) (Nat.gcd (Int.natAbs d) (Nat.gcd (Int.natAbs e) (Int.natAbs f))))) = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l1268_126814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_bisecting_C1_C2_l1268_126853

-- Define the circles C1 and C2
def C1 (p : ℝ × ℝ) : Prop := (p.1 - 4)^2 + (p.2 - 8)^2 = 1
def C2 (p : ℝ × ℝ) : Prop := (p.1 - 6)^2 + (p.2 + 6)^2 = 9

-- Define a circle C with center on the x-axis
def C (x₀ r : ℝ) (p : ℝ × ℝ) : Prop := (p.1 - x₀)^2 + p.2^2 = r^2

-- Define the property of C bisecting C1 and C2
def bisects (c c1 c2 : (ℝ × ℝ) → Prop) : Prop :=
  ∀ p, c p → (c1 p ∨ c2 p) → 
    ∃ p1 p2, c1 p1 ∧ c2 p2 ∧ 
      (p.1 - p1.1)^2 + (p.2 - p1.2)^2 = (p.1 - p2.1)^2 + (p.2 - p2.2)^2

theorem circle_bisecting_C1_C2 :
  ∃ x₀ : ℝ, bisects (C x₀ 9) C1 C2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_bisecting_C1_C2_l1268_126853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_scaling_l1268_126817

theorem determinant_scaling (x y z w : ℝ) :
  Matrix.det !![x, y; z, w] = -3 →
  Matrix.det !![3*x, 3*y; 3*z, 3*w] = -27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_scaling_l1268_126817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_average_speed_l1268_126819

/-- Calculates the average speed of a car given its speeds for three equal parts of a journey -/
noncomputable def averageSpeed (s1 s2 s3 : ℝ) : ℝ :=
  3 / ((1 / s1) + (1 / s2) + (1 / s3))

/-- Theorem: The average speed of a car traveling in three equal parts at 80 km/h, 24 km/h, and 44 km/h is 2640/67 km/h -/
theorem car_average_speed :
  averageSpeed 80 24 44 = 2640 / 67 := by
  -- Unfold the definition of averageSpeed
  unfold averageSpeed
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_average_speed_l1268_126819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_tripling_l1268_126848

/-- The annual interest rate as a decimal -/
def r : ℝ := 0.341

/-- The compound interest factor -/
def factor : ℝ := 1 + r

/-- The function that calculates the value of the investment after t years -/
noncomputable def investment_value (initial_value : ℝ) (t : ℝ) : ℝ := initial_value * (factor ^ t)

/-- The smallest integer number of years for the investment to more than triple -/
def smallest_tripling_period : ℕ := 4

/-- Theorem stating that the investment more than triples after the smallest_tripling_period -/
theorem investment_tripling :
  ∀ (initial_value : ℝ), initial_value > 0 →
  (∀ (t : ℝ), t < smallest_tripling_period → investment_value initial_value t ≤ 3 * initial_value) ∧
  investment_value initial_value (smallest_tripling_period : ℝ) > 3 * initial_value :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_tripling_l1268_126848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_trig_expression_l1268_126877

theorem min_value_trig_expression (θ : Real) (h : 0 < θ ∧ θ < Real.pi / 2) :
  ∃ (min_val : Real), min_val = 6 * (Real.sqrt 3) ^ (1/3) ∧
  ∀ θ', 0 < θ' ∧ θ' < Real.pi / 2 →
    3 * Real.sin θ' + 2 / Real.cos θ' + 2 * Real.sqrt 3 / Real.tan θ' ≥ min_val :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_trig_expression_l1268_126877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flower_shop_max_profit_l1268_126825

/-- Represents the purchase and sale of flowers to maximize gross profit --/
theorem flower_shop_max_profit :
  ∀ (rose_count lily_count : ℕ),
  (1000 ≤ rose_count) ∧ (rose_count ≤ 1500) →
  (if rose_count > 1200 then rose_count * 3 else rose_count * 4) + lily_count * 5 = 9000 →
  ∀ (other_rose_count other_lily_count : ℕ),
  (1000 ≤ other_rose_count) ∧ (other_rose_count ≤ 1500) →
  (if other_rose_count > 1200 then other_rose_count * 3 else other_rose_count * 4) + other_lily_count * 5 = 9000 →
  (rose_count * 5 + lily_count * 13/2 : ℚ) - 9000 ≤ (1500 * 5 + 900 * 13/2 : ℚ) - 9000 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flower_shop_max_profit_l1268_126825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_machine_present_value_l1268_126801

/-- The present value of a machine given its future value and depreciation rate. -/
noncomputable def present_value (future_value : ℝ) (depreciation_rate : ℝ) (years : ℕ) : ℝ :=
  future_value / ((1 - depreciation_rate) ^ years)

/-- Theorem stating that the present value of a machine with given conditions is $2500. -/
theorem machine_present_value :
  let future_value : ℝ := 2256.25
  let depreciation_rate : ℝ := 0.05
  let years : ℕ := 2
  abs (present_value future_value depreciation_rate years - 2500) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_machine_present_value_l1268_126801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_fraction_l1268_126802

/-- The sum of the first n terms of a geometric series with first term a and common ratio r -/
noncomputable def geometricSum (a : ℝ) (r : ℝ) (n : ℕ) : ℝ := a * (1 - r^n) / (1 - r)

/-- The fraction of the sum of the first 17 terms to the sum of the first 21 terms
    for a geometric series with first term 10 and common ratio 2 -/
theorem geometric_series_fraction :
  (geometricSum 10 2 17) / (geometricSum 10 2 21) = (2^17 - 1) / (2^21 - 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_fraction_l1268_126802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_black_has_winning_strategy_l1268_126878

/-- Represents a player in the game -/
inductive Player : Type where
  | White : Player
  | Black : Player

/-- Represents a position on the board -/
structure Position where
  row : Fin 8
  col : Fin 8

/-- Represents the game state -/
structure GameState where
  whitePieces : List Position
  blackPieces : List Position
  currentPlayer : Player

/-- Represents a move in the game -/
structure Move where
  fromPos : Position
  toPos : Position

/-- Checks if a move is valid according to the game rules -/
def isValidMove (state : GameState) (move : Move) : Bool :=
  sorry

/-- Applies a move to the game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  sorry

/-- Checks if a player has any valid moves -/
def hasValidMoves (state : GameState) : Bool :=
  sorry

/-- The initial game state -/
def initialState : GameState :=
  sorry

/-- Theorem stating that Black has a winning strategy -/
theorem black_has_winning_strategy :
  ∃ (strategy : GameState → Move),
    ∀ (game : List Move),
      game.length % 2 = 0 →
      let finalState := game.foldl applyMove initialState
      (¬hasValidMoves finalState ∧ finalState.currentPlayer = Player.White) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_black_has_winning_strategy_l1268_126878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bridge_pass_time_l1268_126871

/-- The time (in seconds) it takes for a train to pass a bridge -/
noncomputable def train_pass_time (train_length bridge_length : ℝ) (train_speed_kmh : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  total_distance / train_speed_ms

/-- Theorem stating the time it takes for the train to pass the bridge -/
theorem train_bridge_pass_time :
  let train_length := (300 : ℝ)
  let bridge_length := (115 : ℝ)
  let train_speed_kmh := (35 : ℝ)
  let result := train_pass_time train_length bridge_length train_speed_kmh
  abs (result - 42.7) < 0.1 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval train_pass_time 300 115 35

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bridge_pass_time_l1268_126871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_death_probability_first_three_months_l1268_126808

/-- The probability that an animal will die during each of the first 3 months,
    given the survival rate of newborns over 3 months. -/
theorem death_probability_first_three_months 
  (total_newborns : ℕ) 
  (expected_survivors : ℝ) 
  (probability_of_death_in_month : ℕ → ℝ)
  (h_total : total_newborns = 200)
  (h_survivors : expected_survivors = 84.375)
  (h_prob_constant : ∀ month, month ≤ 3 → 
    probability_of_death_in_month month = probability_of_death_in_month 1) :
  probability_of_death_in_month 1 = 0.25 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_death_probability_first_three_months_l1268_126808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bakery_purchase_l1268_126852

/-- Represents the total number of items purchased given the conditions of the problem -/
def total_items (total_money cookie_cost coffee_cost : ℚ) : ℕ :=
  let max_cookies := (total_money / cookie_cost).floor.toNat
  let remaining_money := total_money - (max_cookies : ℚ) * cookie_cost
  let max_coffees := (remaining_money / coffee_cost).floor.toNat
  max_cookies + max_coffees

/-- Theorem stating that under the given conditions, the total items purchased is 13 -/
theorem bakery_purchase : total_items 40 3 2.5 = 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bakery_purchase_l1268_126852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_factor_less_than_six_l1268_126862

theorem probability_factor_less_than_six (n : ℕ) (h : n = 36) :
  (Finset.filter (λ x => x < 6) (Nat.divisors n)).card / (Nat.divisors n).card = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_factor_less_than_six_l1268_126862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_star_circle_area_ratio_approx_l1268_126897

/-- The ratio of the area of a star-like figure formed from 8 congruent arcs of a circle
    to the area of the original circle -/
noncomputable def star_circle_area_ratio (r : ℝ) (n : ℕ) : ℝ :=
  2 * (1 + Real.sqrt 2) / Real.pi

/-- Theorem stating that for a circle with radius 3 divided into 8 congruent arcs and
    rearranged into a star-like figure, the ratio of the area of the star-like figure
    to the area of the original circle is approximately 8(1+√2)/π -/
theorem star_circle_area_ratio_approx :
  ∃ (ε : ℝ), ε > 0 ∧ |star_circle_area_ratio 3 8 - 8 * (1 + Real.sqrt 2) / Real.pi| < ε :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_star_circle_area_ratio_approx_l1268_126897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_height_is_one_l1268_126873

/-- A parabola represented by the equation y = x^2 -/
def Parabola : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1^2}

/-- A right triangle with vertices on a parabola and hypotenuse parallel to x-axis -/
structure RightTriangleOnParabola where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  h_on_parabola : A ∈ Parabola ∧ B ∈ Parabola ∧ C ∈ Parabola
  h_hypotenuse_parallel : A.2 = B.2
  h_right_angle : (C.1 - A.1) * (C.1 - B.1) = -(C.2 - A.2) * (C.2 - B.2)

/-- The height of a right triangle from the right angle to the hypotenuse -/
def triangleHeight (t : RightTriangleOnParabola) : ℝ := t.C.2 - t.A.2

/-- The main theorem: the height of the right triangle on the parabola is 1 -/
theorem height_is_one (t : RightTriangleOnParabola) : triangleHeight t = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_height_is_one_l1268_126873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_squirrel_families_l1268_126863

theorem remaining_squirrel_families (initial_deer : ℕ) (initial_rabbit : ℕ) (initial_squirrel : ℕ)
  (deer_moved_percent : ℚ) (deer_plan_percent : ℚ)
  (rabbit_moved_percent : ℚ) (rabbit_plan_percent : ℚ)
  (squirrel_moved_percent : ℚ) (squirrel_plan_percent : ℚ)
  (h1 : initial_deer = 79)
  (h2 : initial_rabbit = 55)
  (h3 : initial_squirrel = 40)
  (h4 : deer_moved_percent = 30/100)
  (h5 : deer_plan_percent = 25/100)
  (h6 : rabbit_moved_percent = 15/100)
  (h7 : rabbit_plan_percent = 35/100)
  (h8 : squirrel_moved_percent = 10/100)
  (h9 : squirrel_plan_percent = 40/100) :
  initial_squirrel - Int.floor (↑initial_squirrel * (squirrel_moved_percent + squirrel_plan_percent)) = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_squirrel_families_l1268_126863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_median_lines_l1268_126839

/-- A structure representing a right triangle in the coordinate plane with one leg along the x-axis and the other along the y-axis -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  h_positive : a > 0 ∧ b > 0

/-- The unique value of m that satisfies the conditions -/
noncomputable def unique_m : ℝ := -1/4

/-- Theorem stating that there exists a unique value of m satisfying the given conditions -/
theorem right_triangle_median_lines (t : RightTriangle) :
  ∃! m : ℝ,
    (∃ x y : ℝ, y = 4 * x + 3 ∧ 2 * x = -t.a ∧ 2 * y = t.b) ∧
    (∃ x y : ℝ, y = m * x + 5 ∧ 2 * x = 0 ∧ 2 * y = t.b) ∧
    m = unique_m :=
by
  sorry

#check right_triangle_median_lines

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_median_lines_l1268_126839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_ABC_l1268_126810

noncomputable def A : ℝ × ℝ := (0, 0)
noncomputable def B : ℝ × ℝ := (2, 0)
noncomputable def C : ℝ × ℝ := (0, 3)

noncomputable def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (1/2) * abs ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

theorem area_of_triangle_ABC : triangle_area A B C = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_ABC_l1268_126810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_exists_sin_greater_than_one_l1268_126896

theorem negation_of_exists_sin_greater_than_one :
  (¬ ∃ x : ℝ, Real.sin x > 1) ↔ (∀ x : ℝ, Real.sin x ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_exists_sin_greater_than_one_l1268_126896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_special_integers_is_30_l1268_126891

/-- A function that counts the number of positive three-digit integers 
    that are divisible by 15 and have 5 in the units place -/
def count_special_integers : ℕ :=
  (Finset.filter (λ n : ℕ ↦ 
    100 ≤ n ∧ n < 1000 ∧  -- three-digit
    n % 10 = 5 ∧          -- 5 in the units place
    n % 15 = 0            -- divisible by 15
  ) (Finset.range 1000)).card

/-- Theorem stating that the count of special integers is 30 -/
theorem count_special_integers_is_30 : 
  count_special_integers = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_special_integers_is_30_l1268_126891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_inequality_theorem_l1268_126881

open Real

theorem vector_inequality_theorem (a : ℝ) :
  (∀ x θ : ℝ, 0 ≤ θ ∧ θ ≤ π/2 →
    ((x + 3 + 2 * sin θ * cos θ)^2 + (x + a * (sin θ + cos θ))^2) ≥ 2) ↔
  (a ≤ 1 ∨ a ≥ 5) := by
  sorry

#check vector_inequality_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_inequality_theorem_l1268_126881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sound_pressure_ratio_l1268_126867

/-- Sound Pressure Level (SPL) calculation -/
noncomputable def SPL (P : ℝ) (Pref : ℝ) : ℝ := 20 * Real.log (P / Pref) / Real.log 10

/-- Reference sound pressure -/
def Pref : ℝ := 20e-6

/-- Theorem: Ratio of effective sound pressure between day and night -/
theorem sound_pressure_ratio (SPL_day SPL_night : ℝ) 
  (h1 : SPL_day = 50)
  (h2 : SPL_night = 30) :
  (20 * (10 : ℝ) ^ (SPL_day / 20)) / (20 * (10 : ℝ) ^ (SPL_night / 20)) = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sound_pressure_ratio_l1268_126867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_division_result_l1268_126856

/-- Given polynomials f and d, prove that q(1) + r(-1) = -2 when f = q * d + r and deg r < deg d -/
theorem polynomial_division_result (f d q r : Polynomial ℝ) : 
  f = 2 * X^4 + 8 * X^3 - 5 * X^2 + 2 * X + 5 →
  d = X^2 + 2 * X - 1 →
  f = q * d + r →
  r.degree < d.degree →
  (q.eval 1 + r.eval (-1) : ℝ) = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_division_result_l1268_126856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lisa_iphone_cost_l1268_126859

/-- Calculates the total cost of Lisa's iPhone and related expenses over 3 years --/
noncomputable def total_iphone_cost (
  initial_iphone_cost : ℝ)
  (monthly_contract_cost : ℝ)
  (case_cost_percentage : ℝ)
  (portable_charger_cost : ℝ)
  (extended_warranty_cost : ℝ)
  (case_headphones_discount : ℝ)
  (monthly_insurance_cost : ℝ)
  (insurance_deductible : ℝ)
  (new_iphone_discount : ℝ)
  : ℝ :=
  let case_cost := initial_iphone_cost * case_cost_percentage
  let headphones_cost := case_cost / 2
  let discounted_case_cost := case_cost * (1 - case_headphones_discount)
  let discounted_headphones_cost := headphones_cost * (1 - case_headphones_discount)
  let total_contract_cost := monthly_contract_cost * 12 * 3
  let total_insurance_cost := monthly_insurance_cost * 12 * 3
  let new_iphone_cost := initial_iphone_cost * (1 - new_iphone_discount)
  initial_iphone_cost + total_contract_cost + discounted_case_cost + discounted_headphones_cost +
  portable_charger_cost + extended_warranty_cost + total_insurance_cost + insurance_deductible +
  new_iphone_cost

/-- Theorem stating that the total cost of Lisa's iPhone and related expenses over 3 years is $9120 --/
theorem lisa_iphone_cost :
  total_iphone_cost 1000 200 0.2 60 150 0.1 15 200 0.3 = 9120 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lisa_iphone_cost_l1268_126859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l1268_126895

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  AB : ℝ
  BC : ℝ
  AC : ℝ
  r : ℝ -- inradius

-- Define the given triangle
noncomputable def givenTriangle : Triangle where
  A := Real.pi / 3
  AB := 16 -- Assuming AC = 10 for simplicity
  AC := 10
  r := 2 * Real.sqrt 3
  B := 0 -- Placeholder value
  C := 0 -- Placeholder value
  BC := 14 -- Calculated from AB and AC

-- Theorem statement
theorem triangle_area (t : Triangle) (h1 : t = givenTriangle) : 
  (1/2 * t.AB * t.AC * Real.sin t.A : ℝ) = 40 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l1268_126895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_spend_theorem_l1268_126833

/-- Represents the minimum amount Alice needs to spend for free delivery -/
def minimum_spend_for_free_delivery (
  chicken_price : ℝ)
  (chicken_weight : ℝ)
  (lettuce_price : ℝ)
  (tomatoes_price : ℝ)
  (potato_price : ℝ)
  (potato_quantity : ℕ)
  (broccoli_price : ℝ)
  (broccoli_quantity : ℕ)
  (sprouts_price : ℝ)
  (additional_amount : ℝ) : ℝ :=
  chicken_price * chicken_weight +
  lettuce_price +
  tomatoes_price +
  potato_price * (potato_quantity : ℝ) +
  broccoli_price * (broccoli_quantity : ℝ) +
  sprouts_price +
  additional_amount

/-- Theorem stating the minimum amount Alice needs to spend for free delivery -/
theorem minimum_spend_theorem :
  minimum_spend_for_free_delivery 6 1.5 3 2.5 0.75 4 2 2 2.5 11 = 35 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_spend_theorem_l1268_126833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_between_ticks_at_6_l1268_126854

/-- Represents a clock with ticking behavior -/
structure Clock where
  ticks_at_6 : ℕ
  time_at_12 : ℚ

/-- The time between the first and last ticks at 6 o'clock -/
noncomputable def time_between_ticks (c : Clock) : ℚ :=
  (c.time_at_12 / (c.ticks_at_6 * 2 - 1)) * (c.ticks_at_6 - 1)

/-- Theorem stating the time between the first and last ticks at 6 o'clock is 40 seconds -/
theorem time_between_ticks_at_6 (c : Clock) 
  (h1 : c.ticks_at_6 = 6) 
  (h2 : c.time_at_12 = 88) : 
  time_between_ticks c = 40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_between_ticks_at_6_l1268_126854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_count_even_quadratic_function_count_l1268_126876

/-- The set of possible coefficients -/
def CoeffSet : Finset ℤ := {-1, 0, 1, 2}

/-- A quadratic function represented by its coefficients -/
structure QuadraticFunction where
  a : ℤ
  b : ℤ
  c : ℤ
  a_nonzero : a ≠ 0
  coeff_in_set : a ∈ CoeffSet ∧ b ∈ CoeffSet ∧ c ∈ CoeffSet
  distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c

/-- An even quadratic function -/
def EvenQuadraticFunction (f : QuadraticFunction) : Prop :=
  f.b = 0

theorem quadratic_function_count :
  ∃ (S : Finset QuadraticFunction), Finset.card S = 18 := by
  sorry

theorem even_quadratic_function_count (S : Finset QuadraticFunction) :
  Finset.card S = 18 →
  ∃ (E : Finset QuadraticFunction), E ⊆ S ∧ (∀ f ∈ E, EvenQuadraticFunction f) ∧ Finset.card E = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_count_even_quadratic_function_count_l1268_126876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_special_property_l1268_126864

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the property of arithmetic sequence for the given expressions
def isArithmeticSequence (t : Triangle) : Prop :=
  ∃ (d : Real), t.a * Real.cos t.C - t.b * Real.cos t.B = t.b * Real.cos t.B - t.c * Real.cos t.A

-- State the theorem
theorem triangle_special_property (t : Triangle) 
  (h : isArithmeticSequence t) : t.B = Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_special_property_l1268_126864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_deduction_approx_five_percent_l1268_126872

/-- Calculates the percentage deduction from the marked price to achieve the desired profit. -/
noncomputable def calculate_percentage_deduction (cost_price : ℝ) (marked_price : ℝ) (profit_percentage : ℝ) : ℝ :=
  let selling_price := cost_price * (1 + profit_percentage / 100)
  let deduction := marked_price - selling_price
  (deduction / marked_price) * 100

/-- Theorem stating that the percentage deduction is approximately 5% given the specified conditions. -/
theorem percentage_deduction_approx_five_percent :
  let cost_price : ℝ := 100
  let marked_price : ℝ := 131.58
  let profit_percentage : ℝ := 25
  let calculated_deduction := calculate_percentage_deduction cost_price marked_price profit_percentage
  abs (calculated_deduction - 5) < 0.1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_deduction_approx_five_percent_l1268_126872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_sum_of_bounds_l1268_126847

/-- Parabola P with equation y = x^2 -/
def P : ℝ → ℝ := λ x => x^2

/-- Point Q -/
def Q : ℝ × ℝ := (12, 8)

/-- Line passing through Q with slope m -/
def line (m : ℝ) (x : ℝ) : ℝ := m * (x - Q.1) + Q.2

/-- Condition for line not intersecting parabola P -/
def no_intersection (m : ℝ) : Prop :=
  ∀ x, P x ≠ line m x

/-- Lower bound of non-intersecting slopes -/
noncomputable def a : ℝ := (48 - Real.sqrt 2176) / 2

/-- Upper bound of non-intersecting slopes -/
noncomputable def b : ℝ := (48 + Real.sqrt 2176) / 2

theorem parabola_line_intersection :
  ∀ m, no_intersection m ↔ a < m ∧ m < b := by
  sorry

theorem sum_of_bounds :
  a + b = 49 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_sum_of_bounds_l1268_126847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arcsin_plus_arccos_equals_pi_half_l1268_126812

theorem arcsin_plus_arccos_equals_pi_half (x : ℝ) :
  x ∈ Set.Icc (-1 : ℝ) 1 →
  (∀ y ∈ Set.Icc (-1 : ℝ) 1, HasDerivAt (λ t ↦ Real.arcsin t + Real.arccos t) 0 y) →
  Real.arcsin x + Real.arccos x = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arcsin_plus_arccos_equals_pi_half_l1268_126812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_is_perfect_square_l1268_126869

def a : ℕ → ℤ
  | 0 => 1
  | 1 => 1
  | (n + 2) => 7 * a (n + 1) - a n - 2

theorem a_is_perfect_square : ∀ n : ℕ, ∃ b : ℤ, a n = b ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_is_perfect_square_l1268_126869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_convergence_l1268_126851

theorem sequence_convergence (x : ℕ → ℝ) (h_bounded : ∃ M, ∀ n, |x n| ≤ M)
  (h_inequality : ∀ n, x n + x (n + 1) ≥ 2 * x (n + 2)) :
  ∃ L, Filter.Tendsto x Filter.atTop (nhds L) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_convergence_l1268_126851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_f_determination_l1268_126807

/-- Given functions f and g with specific properties, prove that f(x) = 2x^2 - 1 -/
theorem function_f_determination (f g : ℝ → ℝ) (a b : ℝ) (h_a : a > 0) :
  (∀ x : ℝ, f x = 2 * a * x^2 - 2 * b * x - a + b) →
  (∀ x : ℝ, g x = 2 * a * x - 2 * b) →
  (∀ θ : ℝ, |f (Real.sin θ)| ≤ 1) →
  (∀ θ : ℝ, g (Real.sin θ) ≤ 2) →
  (∃ θ : ℝ, g (Real.sin θ) = 2) →
  (∀ x : ℝ, f x = 2 * x^2 - 1) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_f_determination_l1268_126807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_power_five_expansion_l1268_126831

theorem cos_power_five_expansion (a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ θ : ℝ, (Real.cos θ)^5 = a₁ * Real.cos θ + a₂ * Real.cos (2*θ) + a₃ * Real.cos (3*θ) + a₄ * Real.cos (4*θ) + a₅ * Real.cos (5*θ)) →
  a₁^2 + a₂^2 + a₃^2 + a₄^2 + a₅^2 = 63/128 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_power_five_expansion_l1268_126831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_theorem_l1268_126893

/-- Given a projection that takes (3, 3) to (45/10, 9/10), 
    prove that it takes (-3, 3) to (-30/13, -6/13) -/
theorem projection_theorem (P : ℝ × ℝ → ℝ × ℝ) 
  (h : P (3, 3) = (45/10, 9/10)) : 
  P (-3, 3) = (-30/13, -6/13) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_theorem_l1268_126893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_shift_l1268_126840

-- Define the two functions as noncomputable
noncomputable def f (x : ℝ) : ℝ := 2^(x + 2)
noncomputable def g (x : ℝ) : ℝ := 2^(x - 1)

-- State the theorem
theorem graph_shift : ∀ x : ℝ, f x = g (x + 3) := by
  intro x
  -- Unfold the definitions of f and g
  unfold f g
  -- Use real number properties to prove the equality
  simp [Real.rpow_add, Real.rpow_neg]
  -- The rest of the proof is skipped
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_shift_l1268_126840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_and_slope_l1268_126890

-- Define the hyperbola
def hyperbola (b : ℝ) (x y : ℝ) : Prop := x^2 - y^2/b^2 = 1

-- Define the foci
noncomputable def right_focus (b : ℝ) : ℝ × ℝ := (Real.sqrt (1 + b^2), 0)
noncomputable def left_focus (b : ℝ) : ℝ × ℝ := (-Real.sqrt (1 + b^2), 0)

-- Define a line passing through the right focus
noncomputable def line_through_right_focus (b k : ℝ) (x : ℝ) : ℝ := k * (x - Real.sqrt (1 + b^2))

-- Define the length of a line segment
noncomputable def segment_length (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem hyperbola_asymptotes_and_slope (b : ℝ) (h1 : b > 0) :
  (∃ (x y : ℝ), hyperbola b x y ∧ 
    segment_length x y (left_focus b).1 (left_focus b).2 = 
    segment_length x y (right_focus b).1 (right_focus b).2 ∧
    segment_length x y (left_focus b).1 (left_focus b).2 = 
    segment_length (left_focus b).1 (left_focus b).2 (right_focus b).1 (right_focus b).2) →
  ((∃ (k : ℝ), ∀ (x : ℝ), hyperbola b x (k*x)) ∧ k = Real.sqrt 2 ∨ k = -Real.sqrt 2) ∧
  (b = Real.sqrt 3 →
    ∃ (k : ℝ), ∀ (x1 y1 x2 y2 : ℝ), 
      hyperbola b x1 y1 ∧ hyperbola b x2 y2 ∧
      y1 = line_through_right_focus b k x1 ∧
      y2 = line_through_right_focus b k x2 ∧
      segment_length x1 y1 x2 y2 = 4 →
      k = Real.sqrt 15 / 5 ∨ k = -Real.sqrt 15 / 5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_and_slope_l1268_126890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_positive_integers_are_clever_l1268_126836

/-- A positive integer n is clever if there exist positive integers a, b, c, and d
    such that n = (a^2 - b^2) / (c^2 + d^2) -/
def IsClever (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ 
    n * (c^2 + d^2) = a^2 - b^2

/-- All positive integers are clever -/
theorem all_positive_integers_are_clever :
  ∀ (n : ℕ), n > 0 → IsClever n := by
  sorry

#check all_positive_integers_are_clever

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_positive_integers_are_clever_l1268_126836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cars_meet_at_correct_time_and_place_l1268_126843

/-- Represents the distance between two places in kilometers -/
noncomputable def route_length : ℝ := 12

/-- Represents Car A's speed in km/min -/
noncomputable def speed_a : ℝ := route_length / 15

/-- Represents Car B's speed in km/min -/
noncomputable def speed_b : ℝ := route_length / 10

/-- Represents Car B's departure delay in minutes -/
noncomputable def delay : ℝ := 2

/-- Represents the position of Car A at time t -/
noncomputable def pos_a (t : ℝ) : ℝ := speed_a * t

/-- Represents the position of Car B at time t -/
noncomputable def pos_b (t : ℝ) : ℝ := max 0 (speed_b * (t - delay))

/-- The time when the cars meet -/
noncomputable def meeting_time : ℝ := 6

/-- The distance from the starting point where the cars meet -/
noncomputable def meeting_distance : ℝ := 7.2

theorem cars_meet_at_correct_time_and_place :
  pos_a meeting_time = pos_b meeting_time ∧
  pos_a meeting_time = meeting_distance := by
  sorry

#eval "Theorem statement compiled successfully."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cars_meet_at_correct_time_and_place_l1268_126843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1268_126824

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Icc (-2) 3 ∧ 2 * x - x^2 ≥ a) ↔ a ∈ Set.Iic 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1268_126824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_l1268_126842

/-- The function f(x) defined for positive real numbers -/
noncomputable def f (x : ℝ) : ℝ := x^2 + 1/x^2 + 1/(x^2 + 1/x^2)

/-- Theorem stating that f(x) has a minimum value of 2.5 for all positive real x -/
theorem f_min_value (x : ℝ) (hx : x > 0) : f x ≥ 2.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_l1268_126842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_from_powers_l1268_126837

theorem arithmetic_sequence_from_powers (a b c : ℝ) : 
  (2 : ℝ)^a = 3 → (2 : ℝ)^b = 6 → (2 : ℝ)^c = 12 → 
  (b - a = c - b) ∧ (b - a = 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_from_powers_l1268_126837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_product_l1268_126834

theorem max_value_product (x₁ x₂ x₃ : ℝ) (h : x₁ + x₂ + x₃ = 1) :
  x₁^3 * x₂^2 * x₃ ≤ 1 / (2^4 * 3^3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_product_l1268_126834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_6723_499_l1268_126849

noncomputable def round_to_nearest (x : ℝ) : ℤ :=
  if x - ⌊x⌋ < 0.5 then ⌊x⌋ else ⌈x⌉

theorem round_6723_499 :
  let x : ℝ := 6723.499
  (6723 : ℝ) ≤ x ∧ x < 6724 ∧ (x - 6723) < 0.5 →
  round_to_nearest x = 6723 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_6723_499_l1268_126849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l1268_126886

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x^2 - 3*x + 2 * Real.log x

-- State the theorem
theorem tangent_line_at_one :
  ∃ (m b : ℝ), ∀ (x y : ℝ),
    (y - f 1 = m * (x - 1)) ↔ (x - y - 3 = 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l1268_126886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_plane_without_points_l1268_126805

-- Define a type for points in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a type for planes in 3D space
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

-- Define a function to check if three points are collinear
def collinear (p1 p2 p3 : Point3D) : Prop := sorry

-- Define a function to check if four points are coplanar
def coplanar (p1 p2 p3 p4 : Point3D) : Prop := sorry

-- Define a function to create a plane from three points
noncomputable def plane_from_points (p1 p2 p3 : Point3D) : Plane3D := sorry

-- Define a function to check if a point lies on a plane
def point_on_plane (p : Point3D) (plane : Plane3D) : Prop := sorry

-- Main theorem
theorem exists_plane_without_points 
  (n : ℕ) 
  (points : Finset Point3D) 
  (h_card : points.card = n) 
  (h_general_position : ∀ p1 p2 p3 p4, p1 ∈ points → p2 ∈ points → p3 ∈ points → p4 ∈ points → 
    p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 → ¬collinear p1 p2 p3 ∧ 
    p1 ≠ p4 ∧ p2 ≠ p4 ∧ p3 ≠ p4 → ¬coplanar p1 p2 p3 p4) :
  ∀ (selected : Finset Point3D), selected ⊆ points → selected.card = n - 3 →
  ∃ (p1 p2 p3 : Point3D), p1 ∈ points ∧ p2 ∈ points ∧ p3 ∈ points ∧
    p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧
    ∀ p ∈ selected, ¬point_on_plane p (plane_from_points p1 p2 p3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_plane_without_points_l1268_126805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_cubic_difference_l1268_126838

theorem sin_cos_cubic_difference (α : ℝ) 
  (h1 : Real.sin α + Real.cos α = 1/3) 
  (h2 : α ∈ Set.Ioo (-π/2) (π/2)) : 
  Real.sin α ^ 3 - Real.cos α ^ 3 = -5 * Real.sqrt 17 / 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_cubic_difference_l1268_126838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_formTeams_with_girls_l1268_126809

/-- The number of ways to form two teams of 5 players from 7 boys and 3 girls, 
    such that both teams have at least one girl -/
def formTeams (numBoys numGirls : ℕ) : ℕ :=
  Nat.choose numGirls 1 * Nat.choose 2 2 * Nat.choose numBoys 4

theorem formTeams_with_girls (numBoys numGirls : ℕ) 
  (h1 : numBoys = 7) (h2 : numGirls = 3) : 
  formTeams numBoys numGirls = 105 := by
  sorry

#eval formTeams 7 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_formTeams_with_girls_l1268_126809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_middle_term_arithmetic_sequence_l1268_126806

/-- The middle term of an arithmetic sequence with first term 1 and last term 5 is 3 -/
theorem middle_term_arithmetic_sequence :
  ∀ m : ℝ, (∃ (a d : ℝ) (n : ℕ), n > 2 ∧ 
    (∀ k : ℕ, k < n → (a + k * d = 1 + (k / (n - 1 : ℝ)) * 4)) ∧
    (a = 1) ∧ (a + (n - 1 : ℝ) * d = 5) ∧ (m = a + ((n - 1 : ℝ) / 2) * d)) →
  m = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_middle_term_arithmetic_sequence_l1268_126806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tree_planting_pigeonhole_l1268_126861

theorem tree_planting_pigeonhole (n : ℕ) (total : ℕ) (min max : ℕ) 
  (h_n : n = 204)
  (h_total : total = 15301)
  (h_min : min = 50)
  (h_max : max = 100)
  (h_range : ∀ s, s ∈ Finset.range (max - min + 1) → min ≤ s + min ∧ s + min ≤ max) :
  ∃ (k : ℕ), k ≥ 5 ∧ ∃ (S : Finset ℕ), S.card = k ∧ 
    (∀ i j, i ∈ S → j ∈ S → (i : ℕ) = (j : ℕ)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tree_planting_pigeonhole_l1268_126861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_calculations_l1268_126823

theorem sqrt_calculations :
  (∀ (a b : ℝ), a = 8 ∧ b = 2 → Real.sqrt a - Real.sqrt b = Real.sqrt b) ∧
  (∀ (c d e f : ℝ), c = 12 ∧ d = 3 ∧ e = 4 ∧ f = 2 →
      -2 * Real.sqrt c * (Real.sqrt d / e) / Real.sqrt f = -(3 * Real.sqrt f / 2)) :=
by
  constructor
  · intro a b hab
    sorry
  · intro c d e f hcdef
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_calculations_l1268_126823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_vertices_l1268_126858

/-- The distance between the vertices of two quadratic functions -/
theorem distance_between_vertices (a b c d e f : ℝ) : 
  let vertex1 := (-(b / (2 * a)), c - (b / (2 * a))^2)
  let vertex2 := (-(e / (2 * d)), f - (e / (2 * d))^2)
  let distance := Real.sqrt ((vertex1.1 - vertex2.1)^2 + (vertex1.2 - vertex2.2)^2)
  (∀ x y, y = a * x^2 + b * x + c) →
  (∀ x y, y = d * x^2 + e * x + f) →
  a = 1 → d = 1 → b = 6 → e = -8 → c = 5 → f = 20 →
  distance = Real.sqrt 113 := by
  sorry

#check distance_between_vertices

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_vertices_l1268_126858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_share_difference_l1268_126870

/-- Given a business partnership with three investors A, B, and C, this theorem
    calculates the difference between the profit shares of A and C based on
    their initial investments and B's known profit share. -/
theorem profit_share_difference
  (investment_A investment_B investment_C : ℕ)
  (profit_share_B : ℕ)
  (h1 : investment_A = 8000)
  (h2 : investment_B = 10000)
  (h3 : investment_C = 12000)
  (h4 : profit_share_B = 3500) :
  (let total_investment := investment_A + investment_B + investment_C
   let total_profit := profit_share_B * total_investment / investment_B
   let profit_share_A := total_profit * investment_A / total_investment
   let profit_share_C := total_profit * investment_C / total_investment
   profit_share_C - profit_share_A) = 1400 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_share_difference_l1268_126870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_arrangement_l1268_126826

-- Define the fruits
inductive Fruit
| Apple
| Pear
| Orange
| Banana

-- Define a type for box numbers
inductive BoxNumber
| One
| Two
| Three
| Four

-- Define a function to represent the arrangement of fruits in boxes
def arrangement : BoxNumber → Fruit := sorry

-- Define the labeling function (which is known to be incorrect for all boxes)
def label : BoxNumber → Fruit := sorry

-- State that all labels are incorrect
axiom all_labels_incorrect :
  ∀ (b : BoxNumber), label b ≠ arrangement b

-- Define the labels for each box
axiom label_box_one : label BoxNumber.One = Fruit.Orange
axiom label_box_two : label BoxNumber.Two = Fruit.Pear
axiom label_box_four : label BoxNumber.Four = Fruit.Apple

-- Define the conditional label for box three
axiom label_box_three :
  arrangement BoxNumber.One = Fruit.Banana →
  (arrangement BoxNumber.Three = Fruit.Apple ∨ arrangement BoxNumber.Three = Fruit.Pear)

-- State that each fruit is in exactly one box
axiom fruit_in_one_box :
  ∀ (f : Fruit), ∃! (b : BoxNumber), arrangement b = f

-- The theorem to prove
theorem correct_arrangement :
  arrangement BoxNumber.One = Fruit.Banana ∧
  arrangement BoxNumber.Two = Fruit.Apple ∧
  arrangement BoxNumber.Three = Fruit.Orange ∧
  arrangement BoxNumber.Four = Fruit.Pear := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_arrangement_l1268_126826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l1268_126889

/-- An ellipse with foci on the y-axis -/
structure Ellipse where
  a : ℝ
  c : ℝ
  h : c < a

/-- A line intersecting the ellipse -/
structure IntersectingLine (E : Ellipse) where
  k : ℝ
  m : ℝ

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (E : Ellipse) : ℝ := E.c / E.a

/-- The distance from the upper focus to the upper vertex -/
noncomputable def upperFocusToVertex (E : Ellipse) : ℝ := E.a - E.c

/-- The area of the triangle formed by the origin and two points -/
noncomputable def triangleArea (x₁ y₁ x₂ y₂ : ℝ) : ℝ := abs (x₁ * y₂ - x₂ * y₁) / 2

/-- The theorem to be proved -/
theorem ellipse_theorem (E : Ellipse) (l : IntersectingLine E) :
  eccentricity E = Real.sqrt 3 / 2 →
  upperFocusToVertex E = 2 - Real.sqrt 3 →
  (∃ x₁ y₁ x₂ y₂ : ℝ,
    x₁^2 + y₁^2/4 = 1 ∧
    x₂^2 + y₂^2/4 = 1 ∧
    (y₁ = l.k * x₁ + l.m) ∧
    (y₂ = l.k * x₂ + l.m) ∧
    triangleArea x₁ y₁ x₂ y₂ = 1) →
  (∀ x₁ y₁ x₂ y₂ : ℝ,
    x₁^2 + y₁^2/4 = 1 →
    x₂^2 + y₂^2/4 = 1 →
    y₁ = l.k * x₁ + l.m →
    y₂ = l.k * x₂ + l.m →
    triangleArea x₁ y₁ x₂ y₂ = 1 →
    x₁^2 + y₁^2 + x₂^2 + y₂^2 = 5) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l1268_126889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_is_42_l1268_126888

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def valid_grid (a b c d e f : ℕ) : Prop :=
  ({a, b, c, d, e, f} : Finset ℕ) = {2, 5, 8, 11, 14, 17} ∧
  is_prime (a + b) ∧ is_prime (c + d) ∧ is_prime (e + f) ∧
  is_prime (a + c + e) ∧ is_prime (b + d + f)

theorem max_sum_is_42 :
  ∀ a b c d e f : ℕ,
    valid_grid a b c d e f →
    (∀ sum : ℕ, sum ∈ ({a + b, c + d, e + f, a + c + e, b + d + f} : Finset ℕ) → sum ≤ 42) ∧
    (∃ sum : ℕ, sum ∈ ({a + b, c + d, e + f, a + c + e, b + d + f} : Finset ℕ) ∧ sum = 42) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_is_42_l1268_126888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_l1268_126815

noncomputable def f (x : ℝ) : ℝ := 2^x

noncomputable def g (x : ℝ) : ℝ := (1/2)^x

theorem problem_1 (x : ℝ) : f x = 4 * g x + 3 → x = 2 := by sorry

theorem problem_2 (a : ℝ) : 
  (∃ x ∈ Set.Icc 0 4, f (a + x) - g (-2 * x) ≥ 3) → 
  a ≥ 1 + Real.log 3 / Real.log 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_l1268_126815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l1268_126835

/-- Calculates the length of a train given its speed and time to cross a pole. -/
noncomputable def train_length (speed : ℝ) (time : ℝ) : ℝ :=
  speed * (1000 / 3600) * time

/-- Theorem: A train running at 60 km/hr that crosses a pole in 21 seconds has a length of approximately 350.07 meters. -/
theorem train_length_calculation :
  let speed := (60 : ℝ)
  let time := (21 : ℝ)
  ∃ ε > 0, |train_length speed time - 350.07| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l1268_126835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_lattice_points_satisfying_centroid_condition_l1268_126803

/-- A lattice point in the plane is a point whose coordinates are both integral. -/
def LatticePoint (p : ℚ × ℚ) : Prop :=
  ∃ (x y : ℤ), p = (↑x, ↑y)

/-- The centroid of four points in the plane. -/
noncomputable def Centroid (p1 p2 p3 p4 : ℚ × ℚ) : ℚ × ℚ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  let (x4, y4) := p4
  ((x1 + x2 + x3 + x4) / 4, (y1 + y2 + y3 + y4) / 4)

/-- A set of points satisfies the centroid condition if the centroid of any four points in the set is not a lattice point. -/
def SatisfiesCentroidCondition (S : Set (ℚ × ℚ)) : Prop :=
  ∀ (p1 p2 p3 p4 : ℚ × ℚ), p1 ∈ S → p2 ∈ S → p3 ∈ S → p4 ∈ S → 
    p1 ≠ p2 → p1 ≠ p3 → p1 ≠ p4 → p2 ≠ p3 → p2 ≠ p4 → p3 ≠ p4 → 
    ¬(LatticePoint (Centroid p1 p2 p3 p4))

theorem largest_lattice_points_satisfying_centroid_condition :
  ∃ (S : Finset (ℚ × ℚ)), (∀ p ∈ S, LatticePoint p) ∧ SatisfiesCentroidCondition S ∧ 
    (∀ (T : Finset (ℚ × ℚ)), (∀ p ∈ T, LatticePoint p) → SatisfiesCentroidCondition T → 
      T.card ≤ S.card) ∧
    S.card = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_lattice_points_satisfying_centroid_condition_l1268_126803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1268_126884

theorem triangle_properties (a b c : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_equality : |a - b| + |b - c| = 0) : 
  (a = b ∧ b = c) ∧ |a + b - c| + |b - c - a| = 2 * a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1268_126884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_quadrilateral_l1268_126866

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a quadrilateral -/
structure Quadrilateral where
  P : Point
  Q : Point
  R : Point
  S : Point

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Calculate the angle between three points -/
noncomputable def angle (p1 p2 p3 : Point) : ℝ :=
  Real.arccos ((distance p1 p2)^2 + (distance p2 p3)^2 - (distance p1 p3)^2) / (2 * distance p1 p2 * distance p2 p3)

/-- Calculate the area of a quadrilateral -/
noncomputable def area (q : Quadrilateral) : ℝ :=
  sorry

/-- Theorem: Area of quadrilateral PQRS -/
theorem area_of_quadrilateral (PQRS : Quadrilateral) :
  distance PQRS.P PQRS.Q = 10 →
  distance PQRS.Q PQRS.R = 5 →
  distance PQRS.R PQRS.S = 12 →
  distance PQRS.S PQRS.P = 12 →
  angle PQRS.R PQRS.S PQRS.P = π/3 →
  area PQRS = 36 * Real.sqrt 3 + 3 * Real.sqrt 3601 / 4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_quadrilateral_l1268_126866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_implies_a_range_l1268_126850

theorem no_solution_implies_a_range (a : ℝ) :
  (∀ x : ℝ, ¬(|x - 5| + |x + 3| < a)) → a ≤ 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_implies_a_range_l1268_126850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_for_f_geq_24_l1268_126887

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 5 * (x + 1)^2 + a / (x + 1)^5

theorem min_a_for_f_geq_24 (a : ℝ) :
  (a > 0) →
  (∀ x : ℝ, x ≥ 0 → f a x ≥ 24) ↔
  a ≥ 2 * Real.sqrt ((24/7)^7) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_for_f_geq_24_l1268_126887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_function_correct_l1268_126828

/-- The cost function for sending a package -/
noncomputable def cost (P : ℝ) : ℝ :=
  if P = 0 then 0
  else max 50 (30 + 7 * (P - 1))

/-- Theorem stating the correctness of the cost function -/
theorem cost_function_correct (P : ℝ) (h : P ≥ 0) :
  cost P = if P = 0 then 0 else max 50 (30 + 7 * (P - 1)) :=
by
  unfold cost
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_function_correct_l1268_126828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_P_sqrt3PA_eq_PB_min_PA_plus_PB_eq_5sqrt2_max_abs_PA_minus_PB_eq_2sqrt5_l1268_126880

noncomputable section

-- Define points A and B
def A : ℝ × ℝ := (3, 1)
def B : ℝ × ℝ := (1, -3)

-- Define the line l: x - y + 1 = 0
def l (x y : ℝ) : Prop := x - y + 1 = 0

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Statement B
theorem exists_P_sqrt3PA_eq_PB :
  ∃ P : ℝ × ℝ, l P.1 P.2 ∧ Real.sqrt 3 * distance P A = distance P B := by
  sorry

-- Statement C
theorem min_PA_plus_PB_eq_5sqrt2 :
  ∀ P : ℝ × ℝ, l P.1 P.2 → distance P A + distance P B ≥ 5 * Real.sqrt 2 := by
  sorry

-- Statement D
theorem max_abs_PA_minus_PB_eq_2sqrt5 :
  ∃ P : ℝ × ℝ, l P.1 P.2 ∧ |distance P A - distance P B| = 2 * Real.sqrt 5 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_P_sqrt3PA_eq_PB_min_PA_plus_PB_eq_5sqrt2_max_abs_PA_minus_PB_eq_2sqrt5_l1268_126880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_theorem_l1268_126844

/-- A right prism with regular pentagon bases -/
structure RegularPentagonPrism where
  height : ℝ
  base_side_length : ℝ

/-- Midpoint of an edge in the prism -/
structure Midpoint where
  edge : String

/-- Triangle formed by three midpoints in the prism -/
structure MidpointTriangle where
  m : Midpoint
  n : Midpoint
  o : Midpoint

/-- Calculate the perimeter of the midpoint triangle -/
noncomputable def perimeter_midpoint_triangle (prism : RegularPentagonPrism) (triangle : MidpointTriangle) : ℝ :=
  14 * Real.cos (54 * Real.pi / 180) + 2 * Real.sqrt 449

theorem perimeter_theorem (prism : RegularPentagonPrism) (triangle : MidpointTriangle) 
  (h1 : prism.height = 20)
  (h2 : prism.base_side_length = 14)
  (h3 : triangle.m = ⟨"AB"⟩)
  (h4 : triangle.n = ⟨"BC"⟩)
  (h5 : triangle.o = ⟨"CG"⟩) :
  perimeter_midpoint_triangle prism triangle = 14 * Real.cos (54 * Real.pi / 180) + 2 * Real.sqrt 449 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_theorem_l1268_126844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_x_value_l1268_126860

theorem arithmetic_progression_x_value (x : ℝ) : 
  let a₁ := x^2 - 3
  let a₂ := x^2 + 1
  let a₃ := 2*x^2 - 1
  (a₂ - a₁ = a₃ - a₂) → (x = Real.sqrt 6 ∨ x = -Real.sqrt 6) :=
by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_x_value_l1268_126860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_regions_three_planes_l1268_126898

/-- A plane in 3D space --/
structure Plane3D where
  -- We don't need to define the internal structure of a plane for this problem

/-- The number of regions formed by a set of planes in 3D space --/
def num_regions (planes : Finset Plane3D) : ℕ := sorry

/-- Theorem: The maximum number of regions formed by three planes in 3D space is 8 --/
theorem max_regions_three_planes :
  ∃ (planes : Finset Plane3D), (planes.card = 3 ∧ num_regions planes = 8) ∧
  ∀ (other_planes : Finset Plane3D), other_planes.card = 3 → num_regions other_planes ≤ 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_regions_three_planes_l1268_126898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_convergence_l1268_126892

noncomputable def a (a₀ : ℝ) : ℕ → ℝ
  | 0 => a₀
  | n + 1 => a a₀ n / Real.sqrt (1 + 2020 * (a a₀ n)^2)

theorem sequence_convergence (a₀ : ℝ) (h : a₀ > 0) :
  a a₀ 2020 < 1 / 2020 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_convergence_l1268_126892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_problem_l1268_126804

/-- The length of a rectangle in a configuration of six identical rectangles forming a larger rectangle --/
noncomputable def rectangle_length (total_area : ℝ) : ℝ := 
  Real.sqrt (2 * total_area / 9)

/-- Rounds a real number to the nearest integer --/
noncomputable def round_to_nearest (x : ℝ) : ℤ :=
  ⌊x + 0.5⌋

theorem rectangle_problem (total_area : ℝ) (h1 : total_area = 4500) :
  round_to_nearest (rectangle_length total_area) = 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_problem_l1268_126804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_shaded_area_l1268_126816

noncomputable section

/-- The side length of the square in all figures -/
def side_length : ℝ := 3

/-- The shaded area of Figure A -/
noncomputable def shaded_area_A : ℝ := side_length^2 - (Real.pi * (side_length / 2)^2)

/-- The shaded area of Figure B -/
noncomputable def shaded_area_B : ℝ := side_length^2 - Real.pi

/-- The shaded area of Figure C -/
noncomputable def shaded_area_C : ℝ := side_length^2 - (Real.pi * (side_length * Real.sqrt 2 / 2)^2 / 2)

theorem largest_shaded_area :
  shaded_area_B > shaded_area_A ∧ shaded_area_B > shaded_area_C := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_shaded_area_l1268_126816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_angle_HAF_l1268_126827

/-- Given a rectangle ABCD, F is the midpoint of BC, and H is the trisection point of CD closer to C -/
structure Rectangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  F : ℝ × ℝ
  H : ℝ × ℝ
  is_rectangle : (A.1 = B.1) ∧ (C.1 = D.1) ∧ (A.2 = D.2) ∧ (B.2 = C.2)
  F_midpoint : F = ((B.1 + C.1) / 2, B.2)
  H_trisection : H = (C.1, C.2 + (D.2 - C.2) / 3)

/-- Helper function to calculate the angle HAF -/
noncomputable def angle_HAF (rect : Rectangle) : ℝ :=
  sorry

/-- The maximum measure of angle HAF is π/6 -/
theorem max_angle_HAF (rect : Rectangle) : 
  ∃ (θ : ℝ), θ = Real.pi / 6 ∧ ∀ (α : ℝ), angle_HAF rect ≤ α → α ≤ θ :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_angle_HAF_l1268_126827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_identity_l1268_126841

/-- Represents the sum of the first n terms of a geometric sequence -/
noncomputable def S (a₁ q : ℝ) (n : ℕ) : ℝ := a₁ * (1 - q^n) / (1 - q)

/-- Theorem stating that for any geometric sequence, S_n^2 + S_{2n}^2 = S_n (S_{2n} + S_{3n}) -/
theorem geometric_sequence_sum_identity (a₁ q : ℝ) (n : ℕ) (hq : q ≠ 1) :
  (S a₁ q n)^2 + (S a₁ q (2*n))^2 = (S a₁ q n) * ((S a₁ q (2*n)) + (S a₁ q (3*n))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_identity_l1268_126841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limits_as_x_approaches_zero_l1268_126855

/-- Theorem about various limits as x approaches 0 -/
theorem limits_as_x_approaches_zero :
  (∀ (x : ℝ), x ≠ 0 → 
    (∃ (ε : ℝ), ε > 0 ∧ 
      (abs x < ε → 
        (abs ((Real.sin (3*x) / (5*x)) - (3/5)) < ε) ∧
        (abs ((Real.sin (9*x) / Real.sin (5*x)) - (9/5)) < ε) ∧
        (abs (((Real.cos x - Real.cos (2*x)) / x^2) - (-3/2)) < ε) ∧
        (∀ (p q : ℝ), q ≠ 0 → abs ((Real.tan (p*x) / Real.tan (q*x)) - (p/q)) < ε) ∧
        (abs ((Real.log (1 + 3*x) / (7*x)) - (3/7)) < ε) ∧
        (abs ((Real.arcsin (5*x) / Real.arcsin (2*x)) - (5/2)) < ε)
      )
    )
  ) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_limits_as_x_approaches_zero_l1268_126855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_max_side_length_l1268_126846

theorem triangle_max_side_length 
  (A B C : ℝ) 
  (a b c : ℝ) 
  (h1 : Real.cos (2 * A) + Real.cos (2 * B) + Real.cos (2 * C) = 1) 
  (h2 : a = 9) 
  (h3 : b = 12) 
  (h4 : c ≤ Real.sqrt (a^2 + b^2)) : 
  c ≤ 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_max_side_length_l1268_126846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_force_for_different_lengths_l1268_126885

/-- Represents the force needed to loosen a bolt -/
noncomputable def force_needed (length : ℝ) : ℝ := 2880 / length

/-- The initial condition: 12-inch wrench requires 240 pounds of force -/
axiom initial_condition : force_needed 12 = 240

/-- Theorem stating the force needed for 16-inch and 8-inch wrenches -/
theorem force_for_different_lengths :
  force_needed 16 = 180 ∧ force_needed 8 = 360 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_force_for_different_lengths_l1268_126885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_squares_l1268_126845

theorem min_sum_squares (a b : ℝ) : 
  (∃ k : ℕ, k = 20 ∧ k = (Nat.choose 6 3) * a^3 * b^3) →
  ∀ x y : ℝ, x^2 + y^2 ≥ 2 ∧ (a^2 + b^2 = 2 → x^2 + y^2 ≥ a^2 + b^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_squares_l1268_126845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_fixed_point_l1268_126874

-- Define the function h
noncomputable def h (x : ℝ) : ℝ := (3 * x + 54) / 5

-- State the theorem
theorem h_fixed_point : 
  (∀ x, h (5 * x - 3) = 3 * x + 9) → 
  (∃! x, h x = x) ∧ (h 27 = 27) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_fixed_point_l1268_126874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_comparisons_l1268_126868

theorem sine_comparisons :
  (Real.sin (-π/10) > Real.sin (-π/8)) ∧ (Real.sin (7*π/8) < Real.sin (5*π/8)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_comparisons_l1268_126868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_point_distance_l1268_126829

-- Define the race parameters
noncomputable def total_distance : ℝ := 12
noncomputable def uphill_distance : ℝ := 6
noncomputable def downhill_distance : ℝ := 6
noncomputable def head_start : ℝ := 8 / 60 -- 8 minutes in hours

-- Define speeds for Elliot and Emily
noncomputable def elliot_uphill_speed : ℝ := 12
noncomputable def elliot_downhill_speed : ℝ := 18
noncomputable def emily_uphill_speed : ℝ := 14
noncomputable def emily_downhill_speed : ℝ := 20

-- Theorem to prove
theorem meeting_point_distance (t : ℝ) : 
  t > head_start ∧ t < (uphill_distance / elliot_uphill_speed + head_start) →
  ∃ x : ℝ, 
    x = uphill_distance - emily_uphill_speed * (t - head_start) ∧ 
    x = elliot_downhill_speed * (t - uphill_distance / elliot_uphill_speed) ∧
    x = 169 / 48 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_point_distance_l1268_126829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l1268_126879

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def triangle_condition (t : Triangle) : Prop :=
  (t.a + t.b) / Real.sin (t.A + t.B) = (t.a - t.c) / (Real.sin t.A - Real.sin t.B)

-- Theorem statement
theorem triangle_theorem (t : Triangle) 
  (h1 : triangle_condition t)
  (h2 : t.b = 3)
  (h3 : Real.cos t.A = Real.sqrt 6 / 3) :
  t.B = π / 3 ∧ 
  (1 / 2 : ℝ) * t.a * t.b * Real.sin t.C = (Real.sqrt 3 + 3 * Real.sqrt 2) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l1268_126879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_ellipse_eccentricity_l1268_126882

/-- An ellipse with foci and a point satisfying specific conditions -/
structure SpecialEllipse where
  a : ℝ
  b : ℝ
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ
  P : ℝ × ℝ
  h₁ : a > b
  h₂ : b > 0
  h₃ : F₁.1 < 0 ∧ F₁.2 = 0  -- Left focus
  h₄ : F₂.1 > 0 ∧ F₂.2 = 0  -- Right focus
  h₅ : (P.1^2 / a^2) + (P.2^2 / b^2) = 1  -- P is on the ellipse
  h₆ : (P.1 + F₁.1) / 2 = 0  -- Midpoint of PF₁ is on y-axis
  h₇ : ∃ (θ : ℝ), θ = 30 * π / 180 ∧ 
       Real.tan θ = (P.2 - F₁.2) / (P.1 - F₁.1)  -- ∠PF₁F₂ = 30°

/-- The eccentricity of a SpecialEllipse is √3/3 -/
theorem special_ellipse_eccentricity (e : SpecialEllipse) :
  let c := Real.sqrt (e.a^2 - e.b^2)
  c / e.a = Real.sqrt 3 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_ellipse_eccentricity_l1268_126882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1268_126875

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 + 2*x else -x^2 + 2*x

-- State the theorem
theorem f_properties :
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x, f (2*x - 1) + f (x + 1) ≤ 0 ↔ x ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1268_126875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alley_width_l1268_126830

/-- The width of an alley given a ladder's length and angles --/
theorem alley_width (L : ℝ) (h : L > 0) : 
  ∃ (d : ℝ), d = L * (1 + Real.sqrt 3) / 2 ∧ 
  ∃ (H : ℝ), H > 0 ∧
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧
  H = L * Real.sqrt 3 / 2 ∧
  x = L / 2 ∧
  y = L * Real.sqrt 3 / 2 ∧
  d = x + y ∧
  Real.sin (60 * π / 180) = H / L ∧
  Real.cos (60 * π / 180) = x / L ∧
  Real.cos (30 * π / 180) = y / L := by
  sorry

#check alley_width

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alley_width_l1268_126830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_by_3804_l1268_126899

theorem divisibility_by_3804 (n : ℕ) : 
  3804 ∣ (n^3 - n) * (5^(8*n+4) + 3^(4*n+2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_by_3804_l1268_126899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_XPN_l1268_126822

-- Define the point type
structure Point where
  x : ℝ
  y : ℝ

-- Define the triangle and its properties
structure Triangle (X Y Z : Point) where
  area : ℝ
  area_positive : area > 0

-- Define the midpoint property
def is_midpoint (M A B : Point) : Prop :=
  M.x = (A.x + B.x) / 2 ∧ M.y = (A.y + B.y) / 2

-- Define the theorem
theorem area_of_XPN (X Y Z M N P : Point) (t : Triangle X Y Z) :
  t.area = 180 →
  is_midpoint M X Y →
  is_midpoint N Y Z →
  is_midpoint P X M →
  ∃ (XPN : Triangle X P N), XPN.area = 22.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_XPN_l1268_126822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_and_max_sum_l1268_126811

noncomputable def sequence_a (n : ℕ) : ℝ := 2 * n - 1

noncomputable def sequence_b (n : ℕ) : ℝ := 10 - sequence_a n

noncomputable def sum_s (n : ℕ) : ℝ := ((sequence_a n + 1) / 2) ^ 2

noncomputable def sum_T (n : ℕ) : ℝ := n * (sequence_b 1 + sequence_b n) / 2

theorem arithmetic_sequence_and_max_sum :
  (∀ n : ℕ, sequence_a n > 0) →
  (∀ n : ℕ, sum_s n = ((sequence_a n + 1) / 2) ^ 2) →
  (∀ n : ℕ, sequence_b n = 10 - sequence_a n) →
  (∀ n : ℕ, n ≥ 1 → sequence_a n = 2 * n - 1) ∧
  (∃ m : ℝ, m = 25 ∧ ∀ n : ℕ, sum_T n ≤ m) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_and_max_sum_l1268_126811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equality_l1268_126813

def a : ℝ × ℝ := (2, -1)
def b (lambda : ℝ) : ℝ × ℝ := (lambda, 1)

def vector_add (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)
def vector_sub (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 - w.1, v.2 - w.2)

noncomputable def vector_magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

theorem vector_equality (lambda : ℝ) : 
  vector_magnitude (vector_add a (b lambda)) = vector_magnitude (vector_sub a (b lambda)) ↔ lambda = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equality_l1268_126813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_snail_reaches_destination_l1268_126857

/-- The snail's distance from point B as a function of time -/
noncomputable def S : ℝ → ℝ := sorry

/-- The time at which the snail reaches point B -/
noncomputable def t_star : ℝ := sorry

/-- The initial distance of the snail from point B in centimeters -/
def initial_distance : ℝ := 1000

/-- The derivative of S with respect to time -/
axiom dS_dt (t : ℝ) : deriv S t = S t / (t + 1) - 1

/-- The initial condition for S -/
axiom S_initial : S 0 = initial_distance

/-- t_star is positive -/
axiom t_star_positive : t_star > 0

/-- The snail reaches point B at time t_star -/
axiom snail_reaches_B : S t_star = 0

/-- Theorem: The snail eventually reaches point B -/
theorem snail_reaches_destination : ∃ t : ℝ, t > 0 ∧ S t = 0 := by
  use t_star
  constructor
  · exact t_star_positive
  · exact snail_reaches_B

end NUMINAMATH_CALUDE_ERRORFEEDBACK_snail_reaches_destination_l1268_126857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xiao_wang_fourth_l1268_126821

-- Define the set of participants
inductive Participant : Type
  | XiaoWang : Participant
  | XiaoZhang : Participant
  | XiaoZhao : Participant
  | XiaoLi : Participant

-- Define the ranking function
def ranking : Participant → ℕ := sorry

-- No ties condition
axiom no_ties : ∀ p q : Participant, p ≠ q → ranking p ≠ ranking q

-- Predictions
def prediction_A (p : Participant) : Prop :=
  (p = Participant.XiaoWang ∧ ranking p = 1) ∨
  (p = Participant.XiaoZhang ∧ ranking p = 3)

def prediction_B (p : Participant) : Prop :=
  (p = Participant.XiaoLi ∧ ranking p = 1) ∨
  (p = Participant.XiaoZhao ∧ ranking p = 4)

def prediction_C (p : Participant) : Prop :=
  (p = Participant.XiaoZhao ∧ ranking p = 2) ∨
  (p = Participant.XiaoWang ∧ ranking p = 3)

-- Half of the predictions are correct
axiom half_correct_A : (∃ p : Participant, prediction_A p) ∧
                       (∃ p : Participant, ¬prediction_A p)

axiom half_correct_B : (∃ p : Participant, prediction_B p) ∧
                       (∃ p : Participant, ¬prediction_B p)

axiom half_correct_C : (∃ p : Participant, prediction_C p) ∧
                       (∃ p : Participant, ¬prediction_C p)

-- Theorem to prove
theorem xiao_wang_fourth : ranking Participant.XiaoWang = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_xiao_wang_fourth_l1268_126821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_decomposition_l1268_126800

def x : ℝ × ℝ × ℝ := (8, 1, 12)
def p : ℝ × ℝ × ℝ := (1, 2, -1)
def q : ℝ × ℝ × ℝ := (3, 0, 2)
def r : ℝ × ℝ × ℝ := (-1, 1, 1)

theorem vector_decomposition :
  x = (-1 : ℝ) • p + (4 : ℝ) • q + (3 : ℝ) • r := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_decomposition_l1268_126800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_asphalt_and_green_costs_l1268_126818

/-- The cost to pave 10,000 square meters of streets with asphalt in millions of yuan -/
def x : ℝ := sorry

/-- The cost to green 10,000 square meters of open spaces in millions of yuan -/
def y : ℝ := sorry

/-- The total area in 10,000 square meter units -/
def total_area : ℝ := 80

/-- The cost of scenario 1 in millions of yuan -/
def scenario1_cost : ℝ := 3

/-- The cost of scenario 2 in millions of yuan -/
def scenario2_cost : ℝ := 2.8

theorem asphalt_and_green_costs :
  (total_area * 0.5 * x + total_area * 0.5 * y = scenario1_cost) ∧
  (total_area * 0.4 * x + total_area * 0.6 * y = scenario2_cost) →
  x = 0.5 ∧ y = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_asphalt_and_green_costs_l1268_126818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l1268_126832

/-- A hyperbola with foci on the x-axis -/
structure Hyperbola where
  real_axis : ℝ
  imaginary_axis : ℝ

/-- An ellipse with foci on the x-axis -/
structure Ellipse where
  semi_major_axis : ℝ
  semi_minor_axis : ℝ

def Hyperbola.standard_equation (h : Hyperbola) : Prop :=
  ∀ x y : ℝ, x^2 / (h.real_axis^2 / 4) - y^2 / (h.imaginary_axis^2 / 4) = 1

noncomputable def Hyperbola.eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt ((h.real_axis^2 / 4) + (h.imaginary_axis^2 / 4)) / (h.real_axis / 2)

def Ellipse.standard_equation (e : Ellipse) : Prop :=
  ∀ x y : ℝ, x^2 / e.semi_major_axis^2 + y^2 / e.semi_minor_axis^2 = 1

theorem hyperbola_properties (h : Hyperbola) 
    (h_real : h.real_axis = 10) 
    (h_imaginary : h.imaginary_axis = 8) : 
  h.standard_equation ∧ 
  h.eccentricity = Real.sqrt 41 / 5 ∧
  (let e : Ellipse := ⟨Real.sqrt 41, 4⟩; e.standard_equation) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l1268_126832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_AB_BD_collinear_k_solution_l1268_126883

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

variable (a b : V)

-- Define the vectors
def AB (a b : V) : V := a + b
def BC (a b : V) : V := 2 • a + 8 • b
def CD (a b : V) : V := 3 • (a - b)

-- Helper definition for BD
def BD (a b : V) : V := BC a b + CD a b

-- Theorem 1: AB and BD are collinear
theorem AB_BD_collinear (a b : V) (h : a ≠ 0 ∧ b ≠ 0 ∧ ¬∃ (r : ℝ), a = r • b) : 
  ∃ (t : ℝ), BD a b = t • AB a b := by
  sorry

-- Theorem 2: k = ±1 is the only solution for collinearity
theorem k_solution (a b : V) (h : a ≠ 0 ∧ b ≠ 0 ∧ ¬∃ (r : ℝ), a = r • b) : 
  ∀ k : ℝ, (∃ t : ℝ, k • a + b = t • (a + k • b)) ↔ k = 1 ∨ k = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_AB_BD_collinear_k_solution_l1268_126883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_zeros_g_l1268_126894

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
variable (hf_cont : Continuous f)
variable (hf_diff : Differentiable ℝ f)
variable (hf_ineq : ∀ x, x * (deriv f x) + f x > deriv f x)

-- Define the function g
noncomputable def g (x : ℝ) := (x - 1) * f x + 1/2

-- State the theorem
theorem no_zeros_g : ∀ x > 1, g f x ≠ 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_zeros_g_l1268_126894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_recoloring_theorem_l1268_126820

/-- Represents the color of a number -/
inductive Color
| White
| Black

/-- Represents the state of coloring for numbers from 1 to n -/
def ColorState (n : ℕ) := Fin n → Color

/-- Recoloring operation on a specific number -/
def recolor (state : ColorState n) (k : Fin n) : ColorState n :=
  fun i => if i = k ∨ ¬(((k : ℕ) + 1) ∣ ((i : ℕ) + 1)) then
    match state i with
    | Color.White => Color.Black
    | Color.Black => Color.White
  else
    state i

/-- Initial state where all numbers are white -/
def initialState (n : ℕ) : ColorState n :=
  fun _ => Color.White

/-- Checks if all numbers in the state are black -/
def allBlack (state : ColorState n) : Prop :=
  ∀ i, state i = Color.Black

/-- Main theorem: For all natural numbers n, there exists a sequence of recoloring operations
    that can turn all numbers from 1 to n black -/
theorem recoloring_theorem (n : ℕ) :
  ∃ (seq : List (Fin n)), allBlack (seq.foldl recolor (initialState n)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_recoloring_theorem_l1268_126820
