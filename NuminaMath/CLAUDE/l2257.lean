import Mathlib

namespace NUMINAMATH_CALUDE_hotel_charge_difference_l2257_225788

theorem hotel_charge_difference (G R P : ℝ) 
  (hR : R = G * (1 + 0.125))
  (hP : P = R * (1 - 0.2)) :
  P = G * 0.9 := by
sorry

end NUMINAMATH_CALUDE_hotel_charge_difference_l2257_225788


namespace NUMINAMATH_CALUDE_students_not_enrolled_l2257_225769

theorem students_not_enrolled (total : ℕ) (french : ℕ) (german : ℕ) (both : ℕ) 
  (h1 : total = 60)
  (h2 : french = 41)
  (h3 : german = 22)
  (h4 : both = 9) :
  total - (french + german - both) = 6 := by
  sorry

end NUMINAMATH_CALUDE_students_not_enrolled_l2257_225769


namespace NUMINAMATH_CALUDE_arithmetic_mean_reciprocals_first_four_primes_l2257_225701

theorem arithmetic_mean_reciprocals_first_four_primes :
  let primes : List ℕ := [2, 3, 5, 7]
  let reciprocals := primes.map (λ x => (1 : ℚ) / x)
  let sum := reciprocals.sum
  let mean := sum / 4
  mean = 247 / 840 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_reciprocals_first_four_primes_l2257_225701


namespace NUMINAMATH_CALUDE_problem_statement_l2257_225742

theorem problem_statement (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x^2 - y^2 = 3*x*y) :
  x^2 / y^2 + y^2 / x^2 - 2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2257_225742


namespace NUMINAMATH_CALUDE_inequality_solution_fractional_equation_no_solution_l2257_225737

-- Part 1: Inequality
theorem inequality_solution (x : ℝ) : 
  (1 - x) / 3 - x < 3 - (x + 2) / 4 ↔ x > -2 :=
sorry

-- Part 2: Fractional equation
theorem fractional_equation_no_solution :
  ¬∃ (x : ℝ), (x - 2) / (2 * x - 1) + 1 = 3 / (2 * (1 - 2 * x)) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_fractional_equation_no_solution_l2257_225737


namespace NUMINAMATH_CALUDE_min_moves_for_identical_contents_l2257_225709

/-- Represents a ball color -/
inductive BallColor
| White
| Black

/-- Represents a box containing balls -/
structure Box where
  white : Nat
  black : Nat

/-- Represents a move: taking a ball from a box and either discarding it or transferring it -/
inductive Move
| Discard : BallColor → Move
| Transfer : BallColor → Move

/-- The initial state of the boxes -/
def initialState : (Box × Box) :=
  ({white := 4, black := 6}, {white := 0, black := 10})

/-- Predicate to check if two boxes have identical contents -/
def identicalContents (box1 box2 : Box) : Prop :=
  box1.white = box2.white ∧ box1.black = box2.black

/-- The minimum number of moves required to guarantee identical contents -/
def minMovesForIdenticalContents : Nat := 15

theorem min_moves_for_identical_contents :
  ∀ (sequence : List Move),
  (∃ (finalState : Box × Box),
    finalState.1.white + finalState.1.black + finalState.2.white + finalState.2.black ≤ 
      initialState.1.white + initialState.1.black + initialState.2.white + initialState.2.black ∧
    identicalContents finalState.1 finalState.2) →
  sequence.length ≥ minMovesForIdenticalContents :=
sorry

end NUMINAMATH_CALUDE_min_moves_for_identical_contents_l2257_225709


namespace NUMINAMATH_CALUDE_quadratic_root_sum_product_l2257_225779

theorem quadratic_root_sum_product (p q : ℝ) : 
  (∃ x y : ℝ, 3 * x^2 - p * x + q = 0 ∧ 3 * y^2 - p * y + q = 0 ∧ x + y = 8 ∧ x * y = 12) →
  p + q = 60 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_sum_product_l2257_225779


namespace NUMINAMATH_CALUDE_geometric_progression_condition_l2257_225793

def is_geometric_progression (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_progression_condition
  (a : ℕ → ℝ)
  (h : ∀ n : ℕ, a (n + 2) = a n * a (n + 1) - c)
  (c : ℝ) :
  is_geometric_progression a ↔ (a 1 = a 2 ∧ c = 0) :=
sorry

end NUMINAMATH_CALUDE_geometric_progression_condition_l2257_225793


namespace NUMINAMATH_CALUDE_f_minus_g_at_7_l2257_225728

def f : ℝ → ℝ := fun _ ↦ 3

def g : ℝ → ℝ := fun _ ↦ 5

theorem f_minus_g_at_7 : f 7 - g 7 = -2 := by
  sorry

end NUMINAMATH_CALUDE_f_minus_g_at_7_l2257_225728


namespace NUMINAMATH_CALUDE_complement_intersection_cardinality_l2257_225745

def U : Finset ℕ := {3,4,5,7,8,9}
def A : Finset ℕ := {4,5,7,8}
def B : Finset ℕ := {3,4,7,8}

theorem complement_intersection_cardinality :
  Finset.card (U \ (A ∩ B)) = 3 := by sorry

end NUMINAMATH_CALUDE_complement_intersection_cardinality_l2257_225745


namespace NUMINAMATH_CALUDE_cube_sum_theorem_l2257_225713

theorem cube_sum_theorem (a b : ℝ) (h1 : a + b = 12) (h2 : a * b = 20) : 
  a^3 + b^3 = 1008 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_theorem_l2257_225713


namespace NUMINAMATH_CALUDE_smallest_even_triangle_perimeter_l2257_225722

/-- A triangle with consecutive even integer side lengths -/
structure EvenTriangle where
  n : ℕ
  side1 : ℕ := 2 * n
  side2 : ℕ := 2 * n + 2
  side3 : ℕ := 2 * n + 4

/-- The triangle inequality for EvenTriangle -/
def satisfiesTriangleInequality (t : EvenTriangle) : Prop :=
  t.side1 + t.side2 > t.side3 ∧
  t.side1 + t.side3 > t.side2 ∧
  t.side2 + t.side3 > t.side1

/-- The perimeter of an EvenTriangle -/
def perimeter (t : EvenTriangle) : ℕ :=
  t.side1 + t.side2 + t.side3

/-- The theorem stating the smallest possible perimeter -/
theorem smallest_even_triangle_perimeter :
  ∀ t : EvenTriangle, satisfiesTriangleInequality t →
  ∃ t_min : EvenTriangle, satisfiesTriangleInequality t_min ∧
    perimeter t_min = 18 ∧
    ∀ t' : EvenTriangle, satisfiesTriangleInequality t' →
      perimeter t' ≥ perimeter t_min :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_even_triangle_perimeter_l2257_225722


namespace NUMINAMATH_CALUDE_valve_flow_rate_difference_l2257_225708

/-- The problem of calculating the difference in water flow rates between two valves filling a pool. -/
theorem valve_flow_rate_difference (pool_capacity : ℝ) (both_valves_time : ℝ) (first_valve_time : ℝ) :
  pool_capacity = 12000 ∧ 
  both_valves_time = 48 ∧ 
  first_valve_time = 120 →
  (pool_capacity / both_valves_time) - (pool_capacity / first_valve_time) = 50 := by
sorry

end NUMINAMATH_CALUDE_valve_flow_rate_difference_l2257_225708


namespace NUMINAMATH_CALUDE_quadratic_solution_l2257_225718

/-- A quadratic function passing through specific points with given conditions -/
def QuadraticProblem (f : ℝ → ℝ) : Prop :=
  (∃ b c : ℝ, ∀ x, f x = x^2 + b*x + c) ∧ 
  f 0 = -1 ∧
  f 2 = 7 ∧
  ∃ y₁ y₂ : ℝ, f (-5) = y₁ ∧ ∃ m : ℝ, f m = y₂ ∧ y₁ + y₂ = 28

/-- The solution to the quadratic problem -/
theorem quadratic_solution (f : ℝ → ℝ) (h : QuadraticProblem f) :
  (∀ x, f x = x^2 + 2*x - 1) ∧ 
  (- (2 / (2 * 1)) = -1) ∧
  (∃ m : ℝ, f m = 14 ∧ m = 3) :=
sorry

end NUMINAMATH_CALUDE_quadratic_solution_l2257_225718


namespace NUMINAMATH_CALUDE_monic_quadratic_root_l2257_225710

theorem monic_quadratic_root (x : ℂ) :
  let p : ℂ → ℂ := λ x => x^2 + 6*x + 12
  p (-3 - Complex.I * Real.sqrt 3) = 0 := by
  sorry

end NUMINAMATH_CALUDE_monic_quadratic_root_l2257_225710


namespace NUMINAMATH_CALUDE_sports_purchase_equation_l2257_225767

/-- Represents the cost of sports equipment purchases -/
structure SportsPurchase where
  volleyball_cost : ℝ  -- Cost of one volleyball in yuan
  shot_put_cost : ℝ    -- Cost of one shot put ball in yuan

/-- Conditions of the sports equipment purchase problem -/
def purchase_conditions (p : SportsPurchase) : Prop :=
  2 * p.volleyball_cost + 3 * p.shot_put_cost = 95 ∧
  5 * p.volleyball_cost + 7 * p.shot_put_cost = 230

/-- The theorem stating that the given system of linear equations 
    correctly represents the sports equipment purchase problem -/
theorem sports_purchase_equation (p : SportsPurchase) :
  purchase_conditions p ↔ 
  (2 * p.volleyball_cost + 3 * p.shot_put_cost = 95 ∧
   5 * p.volleyball_cost + 7 * p.shot_put_cost = 230) :=
by sorry

end NUMINAMATH_CALUDE_sports_purchase_equation_l2257_225767


namespace NUMINAMATH_CALUDE_unique_number_property_l2257_225789

theorem unique_number_property : ∃! x : ℚ, x / 3 = x - 5 := by sorry

end NUMINAMATH_CALUDE_unique_number_property_l2257_225789


namespace NUMINAMATH_CALUDE_complex_product_quadrant_l2257_225787

theorem complex_product_quadrant : 
  let z : ℂ := (1 + 3*I) * (3 - I)
  (0 < z.re) ∧ (0 < z.im) :=
by
  sorry

end NUMINAMATH_CALUDE_complex_product_quadrant_l2257_225787


namespace NUMINAMATH_CALUDE_divisibility_and_smallest_m_l2257_225733

def E (x y m : ℕ) : ℤ := (72 / x)^m + (72 / y)^m - x^m - y^m

theorem divisibility_and_smallest_m :
  ∀ k : ℕ,
  let m := 400 * k + 200
  2005 ∣ E 3 12 m ∧
  2005 ∣ E 9 6 m ∧
  (∀ m' : ℕ, m' > 0 ∧ m' < 200 → ¬(2005 ∣ E 3 12 m' ∧ 2005 ∣ E 9 6 m')) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_and_smallest_m_l2257_225733


namespace NUMINAMATH_CALUDE_line_through_intersection_and_parallel_l2257_225768

-- Define the lines
def line1 (x y : ℝ) : Prop := 2*x + 3*y - 7 = 0
def line2 (x y : ℝ) : Prop := 7*x + 15*y + 1 = 0
def line3 (x y : ℝ) : Prop := x + 2*y - 3 = 0
def result_line (x y : ℝ) : Prop := 3*x + 6*y - 2 = 0

-- Theorem statement
theorem line_through_intersection_and_parallel :
  ∃ (x₀ y₀ : ℝ),
    (line1 x₀ y₀ ∧ line2 x₀ y₀) ∧  -- Intersection point satisfies both line1 and line2
    (∀ (x y : ℝ), line3 x y ↔ ∃ (k : ℝ), y - y₀ = k * (x - x₀) ∧ y - y₀ = -1/2 * (x - x₀)) ∧  -- line3 has slope -1/2
    (∀ (x y : ℝ), result_line x y ↔ ∃ (k : ℝ), y - y₀ = k * (x - x₀) ∧ y - y₀ = -1/2 * (x - x₀)) ∧  -- result_line has slope -1/2
    result_line x₀ y₀  -- result_line passes through the intersection point
  := by sorry

end NUMINAMATH_CALUDE_line_through_intersection_and_parallel_l2257_225768


namespace NUMINAMATH_CALUDE_cloth_cost_price_l2257_225748

/-- Calculates the cost price per meter of cloth given the total selling price,
    number of meters sold, and profit per meter. -/
def cost_price_per_meter (total_selling_price : ℕ) (meters_sold : ℕ) (profit_per_meter : ℕ) : ℕ :=
  (total_selling_price - profit_per_meter * meters_sold) / meters_sold

/-- Proves that the cost price of one meter of cloth is Rs. 100, given the conditions. -/
theorem cloth_cost_price :
  cost_price_per_meter 8925 85 5 = 100 := by
  sorry

end NUMINAMATH_CALUDE_cloth_cost_price_l2257_225748


namespace NUMINAMATH_CALUDE_vector_properties_l2257_225731

/-- Given vectors a, b, c and x ∈ [0,π], prove two statements about x and sin(x + π/6) -/
theorem vector_properties (x : Real) 
  (hx : x ∈ Set.Icc 0 Real.pi)
  (a : Fin 2 → Real)
  (ha : a = fun i => if i = 0 then Real.sin x else Real.sqrt 3 * Real.cos x)
  (b : Fin 2 → Real)
  (hb : b = fun i => if i = 0 then -1 else 1)
  (c : Fin 2 → Real)
  (hc : c = fun i => if i = 0 then 1 else -1) :
  (∃ (k : Real), (a + b) = k • c → x = 5 * Real.pi / 6) ∧
  (a • b = 1 / 2 → Real.sin (x + Real.pi / 6) = Real.sqrt 15 / 4) := by
sorry

end NUMINAMATH_CALUDE_vector_properties_l2257_225731


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l2257_225726

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = q * a n

-- State the theorem
theorem geometric_sequence_ratio 
  (a : ℕ → ℝ) 
  (q : ℝ) 
  (h1 : geometric_sequence a q) 
  (h2 : ∀ n, a n > 0) 
  (h3 : q^2 = 4) : 
  (a 3 + a 4) / (a 5 + a 6) = 1/4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l2257_225726


namespace NUMINAMATH_CALUDE_sequence_inequality_l2257_225754

theorem sequence_inequality (a : ℕ → ℝ) (h1 : ∀ n, a n ≥ 0) 
  (h2 : ∀ m n, a (m + n) ≤ a m + a n) :
  ∀ m n, m ≤ n → a n ≤ m * a 1 + (n / m - 1) * a m :=
by sorry

end NUMINAMATH_CALUDE_sequence_inequality_l2257_225754


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2257_225783

theorem polynomial_factorization (x : ℝ) : 3 * x^2 + 3 * x - 18 = 3 * (x + 3) * (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2257_225783


namespace NUMINAMATH_CALUDE_julias_preferred_number_l2257_225780

def is_multiple (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

def digit_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

theorem julias_preferred_number :
  ∃! n : ℕ,
    100 < n ∧ n < 200 ∧
    is_multiple n 13 ∧
    ¬ is_multiple n 3 ∧
    is_multiple (digit_sum n) 5 ∧
    n = 104 := by
  sorry

end NUMINAMATH_CALUDE_julias_preferred_number_l2257_225780


namespace NUMINAMATH_CALUDE_square_sum_product_l2257_225751

theorem square_sum_product (a b : ℝ) (h1 : a + b = 3) (h2 : a * b = 2) :
  a^2 + b^2 + a * b = 7 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_product_l2257_225751


namespace NUMINAMATH_CALUDE_iphone_defects_l2257_225732

theorem iphone_defects (
  initial_samsung : ℕ)
  (initial_iphone : ℕ)
  (final_samsung : ℕ)
  (final_iphone : ℕ)
  (total_sold : ℕ)
  (h1 : initial_samsung = 14)
  (h2 : initial_iphone = 8)
  (h3 : final_samsung = 10)
  (h4 : final_iphone = 5)
  (h5 : total_sold = 4)
  : initial_iphone - final_iphone - (total_sold - (initial_samsung - final_samsung)) = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_iphone_defects_l2257_225732


namespace NUMINAMATH_CALUDE_sports_club_overlap_l2257_225704

/-- Given a sports club with the following properties:
  * There are 30 total members
  * 17 members play badminton
  * 21 members play tennis
  * 2 members play neither badminton nor tennis
  This theorem proves that 10 members play both badminton and tennis. -/
theorem sports_club_overlap :
  ∀ (total badminton tennis neither : ℕ),
  total = 30 →
  badminton = 17 →
  tennis = 21 →
  neither = 2 →
  badminton + tennis - total + neither = 10 :=
by sorry

end NUMINAMATH_CALUDE_sports_club_overlap_l2257_225704


namespace NUMINAMATH_CALUDE_square_perimeter_proof_l2257_225705

theorem square_perimeter_proof (p1 p2 p3 : ℝ) : 
  p1 = 60 ∧ p2 = 48 ∧ p3 = 36 →
  (p1 / 4)^2 - (p2 / 4)^2 = (p3 / 4)^2 →
  p3 = 36 := by
sorry

end NUMINAMATH_CALUDE_square_perimeter_proof_l2257_225705


namespace NUMINAMATH_CALUDE_coefficient_x_cubed_in_expansion_l2257_225730

theorem coefficient_x_cubed_in_expansion : 
  (Finset.range 37).sum (fun k => (Nat.choose 36 k) * (1 ^ (36 - k)) * (1 ^ k)) = 7140 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_cubed_in_expansion_l2257_225730


namespace NUMINAMATH_CALUDE_chenny_spoons_count_l2257_225746

/-- Proves that Chenny bought 4 spoons given the conditions of the problem -/
theorem chenny_spoons_count : 
  ∀ (num_plates : ℕ) (plate_cost spoon_cost total_cost : ℚ),
    num_plates = 9 →
    plate_cost = 2 →
    spoon_cost = 3/2 →
    total_cost = 24 →
    (total_cost - (↑num_plates * plate_cost)) / spoon_cost = 4 :=
by sorry

end NUMINAMATH_CALUDE_chenny_spoons_count_l2257_225746


namespace NUMINAMATH_CALUDE_complex_power_difference_l2257_225736

theorem complex_power_difference (i : ℂ) (h : i^2 = -1) :
  (1 + i)^16 - (1 - i)^16 = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_difference_l2257_225736


namespace NUMINAMATH_CALUDE_solve_cubic_equation_l2257_225752

theorem solve_cubic_equation (m : ℝ) : (m - 4)^3 = (1/8)⁻¹ ↔ m = 6 := by
  sorry

end NUMINAMATH_CALUDE_solve_cubic_equation_l2257_225752


namespace NUMINAMATH_CALUDE_coefficient_x7y_is_20_l2257_225770

/-- The coefficient of x^7y in the expansion of (x^2 + x + y)^5 -/
def coefficient_x7y (x y : ℕ) : ℕ :=
  (Nat.choose 5 1) * (Nat.choose 4 1) * (Nat.choose 3 3)

/-- Theorem stating that the coefficient of x^7y in (x^2 + x + y)^5 is 20 -/
theorem coefficient_x7y_is_20 :
  ∀ x y, coefficient_x7y x y = 20 := by
  sorry

#eval coefficient_x7y 0 0

end NUMINAMATH_CALUDE_coefficient_x7y_is_20_l2257_225770


namespace NUMINAMATH_CALUDE_hyperbola_tangent_angle_bisector_parabola_tangent_angle_bisector_l2257_225750

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a hyperbola -/
structure Hyperbola where
  f1 : Point2D  -- First focus
  f2 : Point2D  -- Second focus
  a : ℝ         -- Distance from center to vertex

/-- Represents a parabola -/
structure Parabola where
  f : Point2D    -- Focus
  directrix : Line2D

/-- Returns the angle bisector of three points -/
def angleBisector (p1 p2 p3 : Point2D) : Line2D :=
  sorry

/-- Returns the tangent line to a hyperbola at a given point -/
def hyperbolaTangent (h : Hyperbola) (p : Point2D) : Line2D :=
  sorry

/-- Returns the tangent line to a parabola at a given point -/
def parabolaTangent (p : Parabola) (pt : Point2D) : Line2D :=
  sorry

/-- Theorem: The angle bisector property holds for hyperbola tangents -/
theorem hyperbola_tangent_angle_bisector (h : Hyperbola) (p : Point2D) :
  hyperbolaTangent h p = angleBisector h.f1 p h.f2 :=
sorry

/-- Theorem: The angle bisector property holds for parabola tangents -/
theorem parabola_tangent_angle_bisector (p : Parabola) (pt : Point2D) :
  parabolaTangent p pt = angleBisector p.f pt (Point2D.mk 0 0) :=  -- Assuming (0,0) is on the directrix
sorry

end NUMINAMATH_CALUDE_hyperbola_tangent_angle_bisector_parabola_tangent_angle_bisector_l2257_225750


namespace NUMINAMATH_CALUDE_additional_interest_proof_l2257_225753

/-- Calculate the simple interest given principal, rate, and time -/
def simpleInterest (principal : ℚ) (rate : ℚ) (time : ℚ) : ℚ :=
  principal * rate * time

theorem additional_interest_proof :
  let principal : ℚ := 2500
  let time : ℚ := 2
  let higherRate : ℚ := 18 / 100
  let lowerRate : ℚ := 12 / 100
  simpleInterest principal higherRate time - simpleInterest principal lowerRate time = 300 := by
sorry

end NUMINAMATH_CALUDE_additional_interest_proof_l2257_225753


namespace NUMINAMATH_CALUDE_inequality_system_solutions_l2257_225763

theorem inequality_system_solutions :
  {x : ℕ | 3 * (x - 1) < 5 * x + 1 ∧ (x - 1) / 2 ≥ 2 * x - 4} = {0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solutions_l2257_225763


namespace NUMINAMATH_CALUDE_positive_real_sum_one_inequality_l2257_225723

theorem positive_real_sum_one_inequality (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 1) : 
  (1/a - 1) * (1/b - 1) * (1/c - 1) ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_positive_real_sum_one_inequality_l2257_225723


namespace NUMINAMATH_CALUDE_tile_pricing_problem_l2257_225724

/-- Represents the price and discount information for tiles --/
structure TileInfo where
  basePrice : ℝ
  discountRate : ℝ
  discountThreshold : ℕ

/-- Calculates the price for a given quantity of tiles --/
def calculatePrice (info : TileInfo) (quantity : ℕ) : ℝ :=
  if quantity ≥ info.discountThreshold
  then info.basePrice * (1 - info.discountRate) * quantity
  else info.basePrice * quantity

/-- Theorem statement for the tile pricing problem --/
theorem tile_pricing_problem
  (redInfo bluInfo : TileInfo)
  (h1 : calculatePrice redInfo 4000 + calculatePrice bluInfo 6000 = 86000)
  (h2 : calculatePrice redInfo 10000 + calculatePrice bluInfo 3500 = 99000)
  (h3 : redInfo.discountRate = 0.2)
  (h4 : bluInfo.discountRate = 0.1)
  (h5 : redInfo.discountThreshold = 5000)
  (h6 : bluInfo.discountThreshold = 5000) :
  redInfo.basePrice = 8 ∧ bluInfo.basePrice = 10 ∧
  (∃ (redQty bluQty : ℕ),
    redQty + bluQty = 12000 ∧
    bluQty ≥ redQty / 2 ∧
    bluQty ≤ 6000 ∧
    calculatePrice redInfo redQty + calculatePrice bluInfo bluQty = 89800 ∧
    ∀ (r b : ℕ), r + b = 12000 → b ≥ r / 2 → b ≤ 6000 →
      calculatePrice redInfo r + calculatePrice bluInfo b ≥ 89800) :=
sorry

end NUMINAMATH_CALUDE_tile_pricing_problem_l2257_225724


namespace NUMINAMATH_CALUDE_problem_statement_l2257_225714

theorem problem_statement (a b c d e : ℕ+) 
  (h1 : a * b + a + b = 624)
  (h2 : b * c + b + c = 234)
  (h3 : c * d + c + d = 156)
  (h4 : d * e + d + e = 80)
  (h5 : a * b * c * d * e = 3628800) : -- 3628800 is 10!
  a - e = 22 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2257_225714


namespace NUMINAMATH_CALUDE_triangle_inequality_l2257_225729

/-- Given a triangle with sides a, b, c and area S, 
    the sum of squares of the sides is greater than or equal to 
    4 times the area multiplied by the square root of 3. 
    Equality holds if and only if the triangle is equilateral. -/
theorem triangle_inequality (a b c S : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_area : S > 0)
  (h_S : S = Real.sqrt (s * (s - a) * (s - b) * (s - c))) 
  (h_s : s = (a + b + c) / 2) : 
  a^2 + b^2 + c^2 ≥ 4 * S * Real.sqrt 3 ∧ 
  (a^2 + b^2 + c^2 = 4 * S * Real.sqrt 3 ↔ a = b ∧ b = c) :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2257_225729


namespace NUMINAMATH_CALUDE_factorization_equality_l2257_225758

theorem factorization_equality (a : ℝ) : a^2 - 3*a = a*(a - 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2257_225758


namespace NUMINAMATH_CALUDE_two_equal_intercept_lines_l2257_225799

/-- A line passing through (5,2) with equal x and y intercepts -/
structure EqualInterceptLine where
  /-- The slope of the line -/
  m : ℝ
  /-- The y-intercept of the line -/
  b : ℝ
  /-- The line passes through (5,2) -/
  passes_through : 2 = m * 5 + b
  /-- The line has equal x and y intercepts -/
  equal_intercepts : b = m * b

/-- There are exactly two distinct lines passing through (5,2) with equal x and y intercepts -/
theorem two_equal_intercept_lines : 
  ∃ (l₁ l₂ : EqualInterceptLine), l₁ ≠ l₂ ∧ 
  ∀ (l : EqualInterceptLine), l = l₁ ∨ l = l₂ :=
sorry

end NUMINAMATH_CALUDE_two_equal_intercept_lines_l2257_225799


namespace NUMINAMATH_CALUDE_limit_point_sequence_a_l2257_225794

def sequence_a (n : ℕ) : ℚ := (n + 1) / n

theorem limit_point_sequence_a :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |sequence_a n - 1| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_point_sequence_a_l2257_225794


namespace NUMINAMATH_CALUDE_intersection_empty_iff_intersection_equals_A_iff_l2257_225757

-- Define sets A and B
def A (a : ℝ) : Set ℝ := { x | a ≤ x ∧ x ≤ a + 2 }
def B : Set ℝ := { x | x ≤ 0 ∨ x ≥ 4 }

-- Theorem 1
theorem intersection_empty_iff (a : ℝ) : A a ∩ B = ∅ ↔ 0 < a ∧ a < 2 := by sorry

-- Theorem 2
theorem intersection_equals_A_iff (a : ℝ) : A a ∩ B = A a ↔ a ≤ -2 ∨ a ≥ 4 := by sorry

end NUMINAMATH_CALUDE_intersection_empty_iff_intersection_equals_A_iff_l2257_225757


namespace NUMINAMATH_CALUDE_system_one_solution_system_two_solution_l2257_225716

-- System 1
theorem system_one_solution (x y : ℚ) : 
  x + y = 4 ∧ 5 * (x - y) - 2 * (x + y) = -1 → x = 27/10 ∧ y = 13/10 := by sorry

-- System 2
theorem system_two_solution (x y : ℚ) :
  2 * (x - y) / 3 - (x + y) / 4 = -1/12 ∧ 3 * (x + y) - 2 * (2 * x - y) = 3 → x = 2 ∧ y = 1 := by sorry

end NUMINAMATH_CALUDE_system_one_solution_system_two_solution_l2257_225716


namespace NUMINAMATH_CALUDE_dot_product_equals_one_l2257_225771

/-- Given two vectors a and b in ℝ², prove that their dot product is 1. -/
theorem dot_product_equals_one (a b : ℝ × ℝ) : 
  a = (2, 1) → a - 2 • b = (1, 1) → a.fst * b.fst + a.snd * b.snd = 1 := by sorry

end NUMINAMATH_CALUDE_dot_product_equals_one_l2257_225771


namespace NUMINAMATH_CALUDE_parallel_vectors_sin_cos_product_l2257_225797

theorem parallel_vectors_sin_cos_product (α : ℝ) : 
  let a : ℝ × ℝ := (4, 3)
  let b : ℝ × ℝ := (Real.sin α, Real.cos α)
  (∃ (k : ℝ), a.1 = k * b.1 ∧ a.2 = k * b.2) →
  Real.sin α * Real.cos α = 12 / 25 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_sin_cos_product_l2257_225797


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l2257_225739

theorem repeating_decimal_sum (a b : ℕ+) : 
  (a.val : ℚ) / b.val = 4 / 11 → Nat.gcd a.val b.val = 1 → a.val + b.val = 15 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l2257_225739


namespace NUMINAMATH_CALUDE_root_exists_in_interval_l2257_225761

def f (x : ℝ) : ℝ := x^5 - x - 1

theorem root_exists_in_interval :
  ∃ r ∈ Set.Ioo 1 2, f r = 0 :=
by sorry

end NUMINAMATH_CALUDE_root_exists_in_interval_l2257_225761


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2257_225762

-- Define the inequality
def inequality (x : ℝ) : Prop := (x - 2) * (x + 3) > 0

-- Define the solution set
def solution_set : Set ℝ := {x | x < -3 ∨ x > 2}

-- Theorem statement
theorem inequality_solution_set :
  {x : ℝ | inequality x} = solution_set := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2257_225762


namespace NUMINAMATH_CALUDE_john_apple_sales_l2257_225743

/-- Calculates the total money earned from selling apples -/
def apple_sales_revenue 
  (trees_x : ℕ) 
  (trees_y : ℕ) 
  (apples_per_tree : ℕ) 
  (price_per_apple : ℚ) : ℚ :=
  (trees_x * trees_y * apples_per_tree : ℚ) * price_per_apple

/-- Proves that John's apple sales revenue is $30 -/
theorem john_apple_sales : 
  apple_sales_revenue 3 4 5 (1/2) = 30 := by
  sorry

end NUMINAMATH_CALUDE_john_apple_sales_l2257_225743


namespace NUMINAMATH_CALUDE_exponent_multiplication_l2257_225744

theorem exponent_multiplication (a : ℝ) : a^4 * a^3 = a^7 := by sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l2257_225744


namespace NUMINAMATH_CALUDE_problem_solution_l2257_225798

theorem problem_solution :
  ∃ (a b c : ℕ),
    (∃ (x : ℝ), x > 0 ∧ (1 - 2 * a : ℝ) ^ 2 = x ∧ (a + 4 : ℝ) ^ 2 = x) ∧
    (4 * a + 2 * b - 1 : ℝ) ^ (1/3 : ℝ) = 3 ∧
    c = ⌊Real.sqrt 13⌋ ∧
    a = 5 ∧
    b = 4 ∧
    c = 3 ∧
    Real.sqrt (a + 2 * b + c : ℝ) = 4 :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2257_225798


namespace NUMINAMATH_CALUDE_sum_of_constants_l2257_225772

theorem sum_of_constants (a b : ℝ) : 
  (∀ x : ℝ, x ≠ 0 → (a + b / x = 1 ↔ x = -1)) ∧
  (∀ x : ℝ, x ≠ 0 → (a + b / x = 7 ↔ x = -3)) →
  a + b = 19 := by
sorry

end NUMINAMATH_CALUDE_sum_of_constants_l2257_225772


namespace NUMINAMATH_CALUDE_greatest_root_of_g_l2257_225792

-- Define the polynomial g(x)
def g (x : ℝ) : ℝ := 20 * x^4 - 18 * x^2 + 3

-- State the theorem
theorem greatest_root_of_g :
  ∃ (r : ℝ), r = Real.sqrt 15 / 5 ∧
  g r = 0 ∧
  ∀ (x : ℝ), g x = 0 → x ≤ r :=
by sorry

end NUMINAMATH_CALUDE_greatest_root_of_g_l2257_225792


namespace NUMINAMATH_CALUDE_least_positive_integer_divisible_by_53_l2257_225734

theorem least_positive_integer_divisible_by_53 :
  ∃ (x : ℕ), x > 0 ∧ 
  (∀ (y : ℕ), y > 0 → y < x → ¬(53 ∣ (3*y)^2 + 2*41*3*y + 41^2)) ∧
  (53 ∣ (3*x)^2 + 2*41*3*x + 41^2) ∧ 
  x = 4 := by
  sorry

end NUMINAMATH_CALUDE_least_positive_integer_divisible_by_53_l2257_225734


namespace NUMINAMATH_CALUDE_area_of_triangle_MOI_l2257_225707

/-- Triangle ABC with given side lengths --/
structure Triangle :=
  (A B C : ℝ × ℝ)
  (AB_length : Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 15)
  (AC_length : Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 8)
  (BC_length : Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) = 17)

/-- Circumcenter of a triangle --/
def circumcenter (t : Triangle) : ℝ × ℝ := sorry

/-- Incenter of a triangle --/
def incenter (t : Triangle) : ℝ × ℝ := sorry

/-- Center of circle tangent to AC, BC, and circumcircle --/
def tangent_circle_center (t : Triangle) : ℝ × ℝ := sorry

/-- Check if a point lies on the internal bisector of angle A --/
def on_angle_bisector (t : Triangle) (p : ℝ × ℝ) : Prop := sorry

/-- Area of a triangle given its vertices --/
def triangle_area (p q r : ℝ × ℝ) : ℝ := sorry

/-- Main theorem --/
theorem area_of_triangle_MOI (t : Triangle) :
  let O := circumcenter t
  let I := incenter t
  let M := tangent_circle_center t
  on_angle_bisector t M →
  triangle_area M O I = 4.5 := by sorry

end NUMINAMATH_CALUDE_area_of_triangle_MOI_l2257_225707


namespace NUMINAMATH_CALUDE_inequality_range_l2257_225786

theorem inequality_range (m : ℝ) : 
  (∀ x : ℝ, x ≤ 0 → m * (x^2 - 2*x) * Real.exp x + 1 ≥ Real.exp x) ↔ 
  m ≥ -1/2 := by
sorry

end NUMINAMATH_CALUDE_inequality_range_l2257_225786


namespace NUMINAMATH_CALUDE_stating_min_problems_olympiad_l2257_225738

/-- The number of students in the olympiad -/
def num_students : ℕ := 55

/-- 
The function that calculates the maximum number of distinct pairs of "+" and "-" scores
for a given number of problems.
-/
def max_distinct_pairs (num_problems : ℕ) : ℕ :=
  (num_problems + 1) * (num_problems + 2) / 2

/-- 
Theorem stating that the minimum number of problems needed in the olympiad is 9,
given that there are 55 students and no two students can have the same number of "+" and "-" scores.
-/
theorem min_problems_olympiad :
  ∃ (n : ℕ), n = 9 ∧ max_distinct_pairs n = num_students ∧
  ∀ (m : ℕ), m < n → max_distinct_pairs m < num_students :=
by sorry

end NUMINAMATH_CALUDE_stating_min_problems_olympiad_l2257_225738


namespace NUMINAMATH_CALUDE_function_characterization_l2257_225740

-- Define the function type
def RealFunction := ℝ → ℝ

-- Define the conditions
def Condition1 (f : RealFunction) : Prop :=
  ∀ u v : ℝ, f (2 * u) = f (u + v) * f (v - u) + f (u - v) * f (-u - v)

def Condition2 (f : RealFunction) : Prop :=
  ∀ u : ℝ, f u ≥ 0

-- State the theorem
theorem function_characterization (f : RealFunction) 
  (h1 : Condition1 f) (h2 : Condition2 f) :
  (∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = 1/2) :=
sorry

end NUMINAMATH_CALUDE_function_characterization_l2257_225740


namespace NUMINAMATH_CALUDE_percentage_chain_l2257_225700

theorem percentage_chain (n : ℝ) : 
  (0.20 * 0.15 * 0.40 * 0.30 * 0.50 * n = 180) → n = 1000000 := by
  sorry

end NUMINAMATH_CALUDE_percentage_chain_l2257_225700


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2257_225727

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (i : ℂ) / (1 + i) = (1 : ℂ) / 2 + (i : ℂ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2257_225727


namespace NUMINAMATH_CALUDE_parabola_distance_theorem_l2257_225796

/-- Parabola type representing y^2 = 8x -/
structure Parabola where
  equation : ℝ → ℝ → Prop
  focus : ℝ × ℝ
  directrix : ℝ → ℝ → Prop

/-- Represents a point on the parabola -/
structure ParabolaPoint (p : Parabola) where
  point : ℝ × ℝ
  on_parabola : p.equation point.1 point.2

theorem parabola_distance_theorem (p : Parabola) 
  (h_equation : p.equation = fun x y ↦ y^2 = 8*x)
  (P : ParabolaPoint p)
  (A : ℝ × ℝ)
  (h_perpendicular : (P.point.1 - A.1) * (P.point.2 - A.2) = 0)
  (h_on_directrix : p.directrix A.1 A.2)
  (h_slope : (A.2 - p.focus.2) / (A.1 - p.focus.1) = -Real.sqrt 3) :
  Real.sqrt ((P.point.1 - p.focus.1)^2 + (P.point.2 - p.focus.2)^2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_parabola_distance_theorem_l2257_225796


namespace NUMINAMATH_CALUDE_function_properties_l2257_225766

noncomputable def f (x : ℝ) : ℝ := Real.sin (x + 7 * Real.pi / 4) + Real.cos (x - 3 * Real.pi / 4)

theorem function_properties (α : ℝ) 
  (h1 : 0 < α) (h2 : α < 3 * Real.pi / 4) (h3 : f α = 6 / 5) :
  (∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, f (x + T) = f x ∧ 
    ∀ S : ℝ, S > 0 ∧ (∀ x : ℝ, f (x + S) = f x) → T ≤ S) ∧
  (∀ x : ℝ, f x ≥ -2) ∧
  (∃ x : ℝ, f x = -2) ∧
  f (2 * α) = 31 * Real.sqrt 2 / 25 :=
sorry

end NUMINAMATH_CALUDE_function_properties_l2257_225766


namespace NUMINAMATH_CALUDE_product_of_five_consecutive_integers_l2257_225774

theorem product_of_five_consecutive_integers (n : ℤ) :
  (n - 2) * (n - 1) * n * (n + 1) * (n + 2) = n^5 - n^4 - 5*n^3 + 4*n^2 + 4*n :=
by sorry

end NUMINAMATH_CALUDE_product_of_five_consecutive_integers_l2257_225774


namespace NUMINAMATH_CALUDE_infinite_chain_resistance_l2257_225711

/-- The resistance of a single resistor in the chain -/
def R₀ : ℝ := 50

/-- The resistance of an infinitely long chain of identical resistors -/
noncomputable def R_X : ℝ := R₀ * (1 + Real.sqrt 5) / 2

/-- Theorem stating that R_X satisfies the equation for the infinite chain resistance -/
theorem infinite_chain_resistance : R_X = R₀ + (R₀ * R_X) / (R₀ + R_X) := by
  sorry

end NUMINAMATH_CALUDE_infinite_chain_resistance_l2257_225711


namespace NUMINAMATH_CALUDE_calories_per_shake_johns_shake_calories_l2257_225712

/-- Calculates the calories in each shake given John's daily meal plan. -/
theorem calories_per_shake (breakfast : ℕ) (total_daily : ℕ) : ℕ :=
  let lunch := breakfast + breakfast / 4
  let dinner := 2 * lunch
  let meals_total := breakfast + lunch + dinner
  let shakes_total := total_daily - meals_total
  shakes_total / 3

/-- Proves that each shake contains 300 calories given John's meal plan. -/
theorem johns_shake_calories :
  calories_per_shake 500 3275 = 300 := by
  sorry

end NUMINAMATH_CALUDE_calories_per_shake_johns_shake_calories_l2257_225712


namespace NUMINAMATH_CALUDE_coin_toss_count_l2257_225777

theorem coin_toss_count (total_tosses : ℕ) (tail_count : ℕ) (head_count : ℕ) :
  total_tosses = 14 →
  tail_count = 5 →
  total_tosses = head_count + tail_count →
  head_count = 9 := by
  sorry

end NUMINAMATH_CALUDE_coin_toss_count_l2257_225777


namespace NUMINAMATH_CALUDE_gcd_18_30_l2257_225703

theorem gcd_18_30 : Nat.gcd 18 30 = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcd_18_30_l2257_225703


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2257_225715

theorem complex_fraction_simplification :
  (3 + 4 * Complex.I) / (1 - 2 * Complex.I) = -1 + 2 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2257_225715


namespace NUMINAMATH_CALUDE_total_height_increase_l2257_225760

-- Define the increase in height per decade
def height_increase_per_decade : ℝ := 75

-- Define the number of centuries
def num_centuries : ℕ := 4

-- Define the number of decades in a century
def decades_per_century : ℕ := 10

-- Theorem statement
theorem total_height_increase : 
  height_increase_per_decade * (num_centuries * decades_per_century) = 3000 :=
by sorry

end NUMINAMATH_CALUDE_total_height_increase_l2257_225760


namespace NUMINAMATH_CALUDE_election_vote_ratio_l2257_225706

theorem election_vote_ratio (Vx Vy : ℝ) 
  (h1 : 0.72 * Vx + 0.36 * Vy = 0.6 * (Vx + Vy)) 
  (h2 : Vx > 0) 
  (h3 : Vy > 0) : 
  Vx / Vy = 2 := by
sorry

end NUMINAMATH_CALUDE_election_vote_ratio_l2257_225706


namespace NUMINAMATH_CALUDE_representation_inequality_l2257_225735

/-- The smallest number of 1s needed to represent a positive integer using only 1s, +, ×, and brackets -/
noncomputable def f (n : ℕ) : ℕ := sorry

/-- The inequality holds for all n > 1 -/
theorem representation_inequality (n : ℕ) (hn : n > 1) :
  3 * Real.log n ≤ Real.log 3 * (f n : ℝ) ∧ Real.log 3 * (f n : ℝ) ≤ 5 * Real.log n := by
  sorry

end NUMINAMATH_CALUDE_representation_inequality_l2257_225735


namespace NUMINAMATH_CALUDE_expression_simplification_l2257_225719

theorem expression_simplification (a : ℝ) (h : a = Real.sqrt 3 - 1) :
  (a - 1) / (a^2 - 2*a + 1) / ((a^2 + a) / (a^2 - 1) + 1 / (a - 1)) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2257_225719


namespace NUMINAMATH_CALUDE_valid_paths_count_l2257_225795

/-- Represents a point in the 2D lattice --/
structure Point where
  x : ℕ
  y : ℕ

/-- Represents a move in the lattice --/
inductive Move
  | Right : Move
  | Up : Move
  | Diagonal : Move
  | LongRight : Move

/-- Checks if a sequence of moves is valid (no right angle turns) --/
def isValidPath (path : List Move) : Bool :=
  sorry

/-- Checks if a path leads from (0,0) to (7,5) --/
def leadsTo7_5 (path : List Move) : Bool :=
  sorry

/-- Counts the number of valid paths from (0,0) to (7,5) --/
def countValidPaths : ℕ :=
  sorry

/-- The main theorem stating that the number of valid paths is N --/
theorem valid_paths_count :
  ∃ N : ℕ, countValidPaths = N :=
sorry

end NUMINAMATH_CALUDE_valid_paths_count_l2257_225795


namespace NUMINAMATH_CALUDE_upward_parabola_m_value_l2257_225749

/-- If y=(m-1)x^2-2mx+1 is an upward-opening parabola, then m = 2 -/
theorem upward_parabola_m_value (m : ℝ) : 
  (∀ x : ℝ, (m - 1) * x^2 - 2 * m * x + 1 = 0 → (m - 1) > 0) → 
  m = 2 := by sorry

end NUMINAMATH_CALUDE_upward_parabola_m_value_l2257_225749


namespace NUMINAMATH_CALUDE_line_parallel_to_AB_through_P_circumcircle_OAB_l2257_225717

-- Define the points
def A : ℝ × ℝ := (4, 0)
def B : ℝ × ℝ := (0, 2)
def P : ℝ × ℝ := (2, 3)
def O : ℝ × ℝ := (0, 0)

-- Define the line equation
def line_equation (x y : ℝ) : Prop := x + 2*y - 8 = 0

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 5

-- Theorem for the line equation
theorem line_parallel_to_AB_through_P :
  ∀ x y : ℝ, line_equation x y ↔ 
  (∃ t : ℝ, x = 2 + t * (B.1 - A.1) ∧ y = 3 + t * (B.2 - A.2)) :=
sorry

-- Theorem for the circle equation
theorem circumcircle_OAB :
  ∀ x y : ℝ, circle_equation x y ↔
  (x - O.1)^2 + (y - O.2)^2 = (x - A.1)^2 + (y - A.2)^2 ∧
  (x - O.1)^2 + (y - O.2)^2 = (x - B.1)^2 + (y - B.2)^2 :=
sorry

end NUMINAMATH_CALUDE_line_parallel_to_AB_through_P_circumcircle_OAB_l2257_225717


namespace NUMINAMATH_CALUDE_integral_approximation_l2257_225720

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
variable (hf_continuous : ContinuousOn f (Set.Icc 0 1))
variable (hf_range : ∀ x ∈ Set.Icc 0 1, 0 ≤ f x ∧ f x ≤ 1)

-- Define N and N_1
variable (N N_1 : ℕ)

-- Define the theorem
theorem integral_approximation :
  ∃ ε > 0, |∫ x in Set.Icc 0 1, f x - (N_1 : ℝ) / N| < ε :=
sorry

end NUMINAMATH_CALUDE_integral_approximation_l2257_225720


namespace NUMINAMATH_CALUDE_second_number_calculation_l2257_225765

theorem second_number_calculation (A : ℝ) (X : ℝ) (h1 : A = 1280) 
  (h2 : 0.25 * A = 0.20 * X + 190) : X = 650 := by
  sorry

end NUMINAMATH_CALUDE_second_number_calculation_l2257_225765


namespace NUMINAMATH_CALUDE_jerrys_action_figures_l2257_225764

theorem jerrys_action_figures (initial : ℕ) : 
  initial + 11 - 10 = 8 → initial = 7 := by
  sorry

end NUMINAMATH_CALUDE_jerrys_action_figures_l2257_225764


namespace NUMINAMATH_CALUDE_different_color_probability_l2257_225702

/-- The probability of drawing two balls of different colors from a box -/
theorem different_color_probability (total_balls : ℕ) (white_balls : ℕ) (black_balls : ℕ) :
  total_balls = white_balls + black_balls →
  white_balls = 3 →
  black_balls = 2 →
  (white_balls * black_balls : ℚ) / ((total_balls * (total_balls - 1)) / 2 : ℚ) = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_different_color_probability_l2257_225702


namespace NUMINAMATH_CALUDE_remainder_proof_l2257_225778

theorem remainder_proof (x y r : ℤ) : 
  x > 0 →
  x = 7 * y + r →
  0 ≤ r →
  r < 7 →
  2 * x = 18 * y + 2 →
  11 * y - x = 1 →
  r = 3 := by sorry

end NUMINAMATH_CALUDE_remainder_proof_l2257_225778


namespace NUMINAMATH_CALUDE_square_area_to_side_length_ratio_l2257_225791

theorem square_area_to_side_length_ratio (a b : ℝ) (h : a > 0 ∧ b > 0) :
  (a^2 / b^2 = 72 / 98) → (a / b = 6 / 7) := by
  sorry

end NUMINAMATH_CALUDE_square_area_to_side_length_ratio_l2257_225791


namespace NUMINAMATH_CALUDE_gcd_lcm_identity_l2257_225741

theorem gcd_lcm_identity (a b c : ℕ+) :
  (Nat.lcm (Nat.lcm a b) c)^2 / (Nat.lcm a b * Nat.lcm b c * Nat.lcm c a) =
  (Nat.gcd (Nat.gcd a b) c)^2 / (Nat.gcd a b * Nat.gcd b c * Nat.gcd c a) :=
by sorry

end NUMINAMATH_CALUDE_gcd_lcm_identity_l2257_225741


namespace NUMINAMATH_CALUDE_rectangle_area_with_inscribed_circle_rectangle_area_is_588_l2257_225776

/-- The area of a rectangle with an inscribed circle of radius 7 and length-to-width ratio of 3:1 -/
theorem rectangle_area_with_inscribed_circle : ℝ :=
  let radius : ℝ := 7
  let length_width_ratio : ℝ := 3
  let diameter : ℝ := 2 * radius
  let width : ℝ := diameter
  let length : ℝ := length_width_ratio * width
  let area : ℝ := length * width
  area

/-- Proof that the area of the rectangle is 588 -/
theorem rectangle_area_is_588 : rectangle_area_with_inscribed_circle = 588 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_with_inscribed_circle_rectangle_area_is_588_l2257_225776


namespace NUMINAMATH_CALUDE_water_ratio_corn_to_pig_l2257_225755

def water_pumping_rate : ℚ := 3
def pumping_time : ℕ := 25
def corn_rows : ℕ := 4
def corn_plants_per_row : ℕ := 15
def num_pigs : ℕ := 10
def water_per_pig : ℚ := 4
def num_ducks : ℕ := 20
def water_per_duck : ℚ := 1/4

theorem water_ratio_corn_to_pig :
  let total_water := water_pumping_rate * pumping_time
  let total_corn_plants := corn_rows * corn_plants_per_row
  let water_for_pigs := num_pigs * water_per_pig
  let water_for_ducks := num_ducks * water_per_duck
  let water_for_corn := total_water - water_for_pigs - water_for_ducks
  let water_per_corn := water_for_corn / total_corn_plants
  water_per_corn / water_per_pig = 1/8 := by sorry

end NUMINAMATH_CALUDE_water_ratio_corn_to_pig_l2257_225755


namespace NUMINAMATH_CALUDE_pawn_placement_count_l2257_225747

/-- The number of ways to place distinct pawns on a square chess board -/
def placePawns (n : ℕ) : ℕ :=
  (n.factorial) ^ 2

/-- The size of the chess board -/
def boardSize : ℕ := 5

/-- The number of pawns to be placed -/
def numPawns : ℕ := 5

theorem pawn_placement_count :
  placePawns boardSize = 14400 :=
sorry

end NUMINAMATH_CALUDE_pawn_placement_count_l2257_225747


namespace NUMINAMATH_CALUDE_morgan_sat_score_l2257_225756

theorem morgan_sat_score (second_score : ℝ) (improvement_rate : ℝ) :
  second_score = 1100 →
  improvement_rate = 0.1 →
  ∃ (first_score : ℝ), first_score * (1 + improvement_rate) = second_score ∧ first_score = 1000 :=
by
  sorry

end NUMINAMATH_CALUDE_morgan_sat_score_l2257_225756


namespace NUMINAMATH_CALUDE_right_triangle_segment_ratio_l2257_225790

theorem right_triangle_segment_ratio (a b c r s : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 ∧ r > 0 ∧ s > 0 →
  a^2 + b^2 = c^2 →
  r + s = c →
  a^2 = r * c →
  b^2 = s * c →
  a / b = 2 / 5 →
  r / s = 4 / 25 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_segment_ratio_l2257_225790


namespace NUMINAMATH_CALUDE_dans_trip_l2257_225773

/-- The distance from Dan's home to his workplace -/
def distance : ℝ := 160

/-- The time of the usual trip in minutes -/
def usual_time : ℝ := 240

/-- The time spent driving at normal speed on the particular day -/
def normal_speed_time : ℝ := 120

/-- The speed reduction factor due to heavy traffic -/
def speed_reduction : ℝ := 0.75

/-- The total trip time on the particular day -/
def total_time : ℝ := 330

theorem dans_trip :
  distance = distance * (normal_speed_time / usual_time + 
    (total_time - normal_speed_time) / (usual_time / speed_reduction)) := by
  sorry

end NUMINAMATH_CALUDE_dans_trip_l2257_225773


namespace NUMINAMATH_CALUDE_smallest_value_absolute_equation_l2257_225725

theorem smallest_value_absolute_equation :
  (∃ x : ℝ, |x - 8| = 15) ∧
  (∀ x : ℝ, |x - 8| = 15 → x ≥ -7) ∧
  |-7 - 8| = 15 := by
  sorry

end NUMINAMATH_CALUDE_smallest_value_absolute_equation_l2257_225725


namespace NUMINAMATH_CALUDE_no_integer_points_in_sphere_intersection_l2257_225782

theorem no_integer_points_in_sphere_intersection : 
  ¬∃ (x y z : ℤ), (x^2 + y^2 + (z - 10)^2 ≤ 9) ∧ (x^2 + y^2 + (z - 2)^2 ≤ 16) :=
by sorry

end NUMINAMATH_CALUDE_no_integer_points_in_sphere_intersection_l2257_225782


namespace NUMINAMATH_CALUDE_min_value_theorem_l2257_225785

theorem min_value_theorem (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 2 * a + 3 * b = 6) :
  (2 / a + 3 / b) ≥ 25 / 6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2257_225785


namespace NUMINAMATH_CALUDE_gcd_of_315_and_2016_l2257_225784

theorem gcd_of_315_and_2016 : Nat.gcd 315 2016 = 63 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_315_and_2016_l2257_225784


namespace NUMINAMATH_CALUDE_bowling_record_proof_l2257_225775

/-- The old record average score per player per round in a bowling league -/
def old_record : ℝ := 287

/-- Number of players in a team -/
def players_per_team : ℕ := 4

/-- Number of rounds in a season -/
def rounds_per_season : ℕ := 10

/-- Total score of the team after 9 rounds -/
def score_after_nine_rounds : ℕ := 10440

/-- Difference between old record and minimum average needed in final round -/
def score_difference : ℕ := 27

theorem bowling_record_proof :
  old_record = 
    (score_after_nine_rounds + players_per_team * (old_record - score_difference)) / 
    (players_per_team * rounds_per_season) := by
  sorry

end NUMINAMATH_CALUDE_bowling_record_proof_l2257_225775


namespace NUMINAMATH_CALUDE_unequal_grandchildren_probability_l2257_225781

def num_grandchildren : ℕ := 12

def prob_male : ℚ := 1/2

def prob_female : ℚ := 1/2

theorem unequal_grandchildren_probability :
  let total_outcomes := 2^num_grandchildren
  let equal_outcomes := (num_grandchildren.choose (num_grandchildren / 2))
  (total_outcomes - equal_outcomes) / total_outcomes = 3172/4096 :=
sorry

end NUMINAMATH_CALUDE_unequal_grandchildren_probability_l2257_225781


namespace NUMINAMATH_CALUDE_max_residents_per_apartment_is_four_l2257_225721

/-- Represents a block of flats -/
structure BlockOfFlats where
  floors : ℕ
  apartments_per_floor_type1 : ℕ
  apartments_per_floor_type2 : ℕ
  max_residents : ℕ

/-- Calculates the maximum number of residents per apartment -/
def max_residents_per_apartment (block : BlockOfFlats) : ℕ :=
  block.max_residents / ((block.floors / 2) * block.apartments_per_floor_type1 + 
                         (block.floors / 2) * block.apartments_per_floor_type2)

/-- Theorem stating the maximum number of residents per apartment -/
theorem max_residents_per_apartment_is_four (block : BlockOfFlats) 
  (h1 : block.floors = 12)
  (h2 : block.apartments_per_floor_type1 = 6)
  (h3 : block.apartments_per_floor_type2 = 5)
  (h4 : block.max_residents = 264) :
  max_residents_per_apartment block = 4 := by
  sorry

#eval max_residents_per_apartment { 
  floors := 12, 
  apartments_per_floor_type1 := 6, 
  apartments_per_floor_type2 := 5, 
  max_residents := 264 
}

end NUMINAMATH_CALUDE_max_residents_per_apartment_is_four_l2257_225721


namespace NUMINAMATH_CALUDE_garden_expansion_l2257_225759

/-- Given a rectangular garden with dimensions 50 feet by 20 feet, 
    prove that adding 40 feet of fencing and reshaping into a square 
    results in a garden 1025 square feet larger than the original. -/
theorem garden_expansion (original_length : ℝ) (original_width : ℝ) 
  (additional_fence : ℝ) (h1 : original_length = 50) 
  (h2 : original_width = 20) (h3 : additional_fence = 40) : 
  let original_area := original_length * original_width
  let original_perimeter := 2 * (original_length + original_width)
  let new_perimeter := original_perimeter + additional_fence
  let new_side := new_perimeter / 4
  let new_area := new_side * new_side
  new_area - original_area = 1025 := by
sorry

end NUMINAMATH_CALUDE_garden_expansion_l2257_225759
