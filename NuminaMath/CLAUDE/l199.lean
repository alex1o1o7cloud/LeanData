import Mathlib

namespace NUMINAMATH_CALUDE_triangle_expression_simplification_l199_19945

-- Define a triangle with side lengths a, b, c
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  -- Triangle inequality conditions
  ab_gt_c : a + b > c
  ac_gt_b : a + c > b
  bc_gt_a : b + c > a

-- Define the theorem
theorem triangle_expression_simplification (t : Triangle) :
  |t.a - t.b - t.c| + |t.b - t.a - t.c| - |t.c - t.a + t.b| = t.a - t.b + t.c :=
by sorry

end NUMINAMATH_CALUDE_triangle_expression_simplification_l199_19945


namespace NUMINAMATH_CALUDE_a_value_l199_19983

def set_A : Set ℝ := {1, -2}

def set_B (a b : ℝ) : Set ℝ := {x | x^2 + a*x + b = 0}

theorem a_value (b : ℝ) : set_A = set_B 1 b → 1 ∈ set_A ∧ -2 ∈ set_A := by sorry

end NUMINAMATH_CALUDE_a_value_l199_19983


namespace NUMINAMATH_CALUDE_range_of_a_l199_19962

-- Define the set P
def P : Set ℝ := {x : ℝ | x^2 ≤ 1}

-- Define the set M
def M (a : ℝ) : Set ℝ := {a}

-- Theorem statement
theorem range_of_a (a : ℝ) : P ∪ M a = P → a ∈ Set.Icc (-1) 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l199_19962


namespace NUMINAMATH_CALUDE_smallest_valid_number_l199_19992

def is_valid_number (N : ℕ) : Prop :=
  ∃ X : ℕ, 
    X > 0 ∧
    (N - 12) % 8 = 0 ∧
    (N - 12) % 12 = 0 ∧
    (N - 12) % 24 = 0 ∧
    (N - 12) % X = 0 ∧
    (N - 12) / Nat.lcm 24 X = 276

theorem smallest_valid_number : 
  is_valid_number 6636 ∧ ∀ n < 6636, ¬ is_valid_number n :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_number_l199_19992


namespace NUMINAMATH_CALUDE_line_of_sight_condition_l199_19984

-- Define the curve C
def C (x : ℝ) : ℝ := 2 * x^2

-- Define point A
def A : ℝ × ℝ := (0, -2)

-- Define point B
def B (a : ℝ) : ℝ × ℝ := (3, a)

-- Define the condition for line of sight not being blocked
def lineOfSightNotBlocked (a : ℝ) : Prop :=
  ∀ x : ℝ, 0 < x → x < 3 → 
    (A.2 + (B a).2 - A.2) / 3 * x + A.2 > C x

-- State the theorem
theorem line_of_sight_condition :
  ∀ a : ℝ, lineOfSightNotBlocked a ↔ a < 10 := by sorry

end NUMINAMATH_CALUDE_line_of_sight_condition_l199_19984


namespace NUMINAMATH_CALUDE_kylie_coins_left_l199_19941

/-- Calculates the number of coins Kylie has left after giving half to Laura -/
def kyliesRemainingCoins (piggyBank : ℕ) (brotherCoins : ℕ) (sofaCoins : ℕ) : ℕ :=
  let fatherCoins := 2 * brotherCoins
  let totalCoins := piggyBank + brotherCoins + fatherCoins + sofaCoins
  totalCoins - (totalCoins / 2)

/-- Theorem stating that Kylie has 62 coins left -/
theorem kylie_coins_left :
  kyliesRemainingCoins 30 26 15 = 62 := by
  sorry

#eval kyliesRemainingCoins 30 26 15

end NUMINAMATH_CALUDE_kylie_coins_left_l199_19941


namespace NUMINAMATH_CALUDE_inequality_chain_l199_19950

theorem inequality_chain (x a : ℝ) (h1 : x < a) (h2 : a < 0) : x^2 > a*x ∧ a*x > a^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_chain_l199_19950


namespace NUMINAMATH_CALUDE_probability_intersection_bounds_l199_19930

theorem probability_intersection_bounds (A B : Set Ω) (P : Set Ω → ℝ) 
  (hA : P A = 3/4) (hB : P B = 2/3) :
  5/12 ≤ P (A ∩ B) ∧ P (A ∩ B) ≤ 2/3 := by
sorry

end NUMINAMATH_CALUDE_probability_intersection_bounds_l199_19930


namespace NUMINAMATH_CALUDE_one_thirds_in_nine_halves_l199_19973

theorem one_thirds_in_nine_halves :
  (9 / 2) / (1 / 3) = 27 / 2 := by
  sorry

end NUMINAMATH_CALUDE_one_thirds_in_nine_halves_l199_19973


namespace NUMINAMATH_CALUDE_order_of_exponential_expressions_l199_19985

theorem order_of_exponential_expressions :
  let a := Real.exp (2 * Real.log 3 * Real.log 2)
  let b := Real.exp (3 * Real.log 2 * Real.log 3)
  let c := Real.exp (Real.log 5 * Real.log 5)
  a < b ∧ b < c := by
  sorry

end NUMINAMATH_CALUDE_order_of_exponential_expressions_l199_19985


namespace NUMINAMATH_CALUDE_probability_four_twos_value_l199_19993

def num_dice : ℕ := 8
def num_sides : ℕ := 8
def num_success : ℕ := 4

def probability_exactly_four_twos : ℚ :=
  (num_dice.choose num_success) * 
  ((1 : ℚ) / num_sides) ^ num_success * 
  ((num_sides - 1 : ℚ) / num_sides) ^ (num_dice - num_success)

theorem probability_four_twos_value : 
  probability_exactly_four_twos = 168070 / 16777216 := by sorry

end NUMINAMATH_CALUDE_probability_four_twos_value_l199_19993


namespace NUMINAMATH_CALUDE_ten_percent_relation_l199_19932

/-- If 10% of s is equal to t, then s equals 10t -/
theorem ten_percent_relation (s t : ℝ) (h : (10 : ℝ) / 100 * s = t) : s = 10 * t := by
  sorry

end NUMINAMATH_CALUDE_ten_percent_relation_l199_19932


namespace NUMINAMATH_CALUDE_tangent_line_sum_l199_19954

/-- Given a differentiable function f : ℝ → ℝ, if the graph of f is tangent to the line 2x+y-1=0 at the point (1, f(1)), then f(1) + f'(1) = -3 -/
theorem tangent_line_sum (f : ℝ → ℝ) (hf : Differentiable ℝ f) :
  (∀ x, 2 * x + f x - 1 = 0 ↔ x = 1) →  -- tangency condition
  (2 * 1 + f 1 - 1 = 0) →               -- point (1, f(1)) lies on the line
  f 1 + deriv f 1 = -3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_sum_l199_19954


namespace NUMINAMATH_CALUDE_boat_downstream_time_l199_19919

def boat_problem (boat_speed : ℝ) (stream_speed : ℝ) (upstream_time : ℝ) : Prop :=
  let downstream_speed := boat_speed + stream_speed
  let upstream_speed := boat_speed - stream_speed
  let distance := upstream_speed * upstream_time
  let downstream_time := distance / downstream_speed
  downstream_time = 1

theorem boat_downstream_time :
  boat_problem 15 3 1.5 := by sorry

end NUMINAMATH_CALUDE_boat_downstream_time_l199_19919


namespace NUMINAMATH_CALUDE_exists_monomial_with_conditions_l199_19902

/-- A monomial is a product of a coefficient and variables raised to non-negative integer powers. -/
structure Monomial (α : Type*) [Semiring α] where
  coeff : α
  powers : List (Nat × Nat)

/-- The degree of a monomial is the sum of the exponents of its variables. -/
def Monomial.degree {α : Type*} [Semiring α] (m : Monomial α) : Nat :=
  m.powers.foldl (fun acc (_, pow) => acc + pow) 0

/-- A monomial contains specific variables if they appear in its power list. -/
def Monomial.containsVariables {α : Type*} [Semiring α] (m : Monomial α) (vars : List Nat) : Prop :=
  ∀ v ∈ vars, ∃ (pow : Nat), (v, pow) ∈ m.powers

/-- There exists a monomial with coefficient 3, containing variables x and y, and having a total degree of 3. -/
theorem exists_monomial_with_conditions :
  ∃ (m : Monomial ℕ),
    m.coeff = 3 ∧
    m.containsVariables [1, 2] ∧  -- Let 1 represent x and 2 represent y
    m.degree = 3 := by
  sorry

end NUMINAMATH_CALUDE_exists_monomial_with_conditions_l199_19902


namespace NUMINAMATH_CALUDE_total_selling_price_l199_19981

-- Define the cost and loss percentage for each item
def cost1 : ℕ := 750
def cost2 : ℕ := 1200
def cost3 : ℕ := 500
def loss_percent1 : ℚ := 10 / 100
def loss_percent2 : ℚ := 15 / 100
def loss_percent3 : ℚ := 5 / 100

-- Calculate the selling price of an item
def selling_price (cost : ℕ) (loss_percent : ℚ) : ℚ :=
  cost - (cost * loss_percent)

-- Define the theorem
theorem total_selling_price :
  selling_price cost1 loss_percent1 +
  selling_price cost2 loss_percent2 +
  selling_price cost3 loss_percent3 = 2170 := by
  sorry

end NUMINAMATH_CALUDE_total_selling_price_l199_19981


namespace NUMINAMATH_CALUDE_function_decreasing_interval_l199_19967

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := x^2 * (a*x + b)

-- Define the derivative of f(x)
def f_derivative (a b x : ℝ) : ℝ := 3*a*x^2 + 2*b*x

theorem function_decreasing_interval (a b : ℝ) :
  (∀ x, f_derivative a b x = 0 → x = 2) →  -- Extremum at x = 2
  (f_derivative a b 1 = -3) →              -- Tangent line parallel to 3x + y = 0
  (∀ x ∈ (Set.Ioo 0 2), f_derivative a b x < 0) ∧ 
  (∀ x ∉ (Set.Icc 0 2), f_derivative a b x ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_function_decreasing_interval_l199_19967


namespace NUMINAMATH_CALUDE_value_of_y_l199_19987

theorem value_of_y (x y : ℝ) (h1 : x^2 - 3*x + 2 = y + 2) (h2 : x = -5) : y = 40 := by
  sorry

end NUMINAMATH_CALUDE_value_of_y_l199_19987


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_simplify_with_condition_l199_19953

-- Part 1
theorem simplify_and_evaluate (x y : ℤ) (h1 : x = -2) (h2 : y = -3) :
  x^2 - 2*(x^2 - 3*y) - 3*(2*x^2 + 5*y) = -1 := by sorry

-- Part 2
theorem simplify_with_condition (a b : ℝ) (h : a - b = 2*b^2) :
  2*(a^3 - 2*b^2) - (2*b - a) + a - 2*a^3 = 0 := by sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_simplify_with_condition_l199_19953


namespace NUMINAMATH_CALUDE_min_boxes_to_eliminate_l199_19908

/-- The total number of boxes in the game -/
def total_boxes : ℕ := 30

/-- The number of boxes containing at least $250,000 -/
def high_value_boxes : ℕ := 6

/-- The desired probability of holding a high-value box -/
def desired_probability : ℚ := 1/3

/-- The function to calculate the minimum number of boxes to eliminate -/
def boxes_to_eliminate : ℕ := total_boxes - (high_value_boxes * 3)

/-- Theorem stating the minimum number of boxes to eliminate -/
theorem min_boxes_to_eliminate :
  boxes_to_eliminate = 12 := by sorry

end NUMINAMATH_CALUDE_min_boxes_to_eliminate_l199_19908


namespace NUMINAMATH_CALUDE_union_complement_equality_l199_19979

def U : Set Nat := {0,1,2,3,4,5,6}
def A : Set Nat := {2,4,5}
def B : Set Nat := {0,1,3,5}

theorem union_complement_equality : A ∪ (U \ B) = {2,4,5,6} := by sorry

end NUMINAMATH_CALUDE_union_complement_equality_l199_19979


namespace NUMINAMATH_CALUDE_upstream_speed_is_26_l199_19964

/-- The speed of a man rowing in a stream -/
structure RowingSpeed :=
  (stillWater : ℝ)
  (downstream : ℝ)

/-- Calculate the speed of the man rowing upstream -/
def upstreamSpeed (s : RowingSpeed) : ℝ :=
  2 * s.stillWater - s.downstream

/-- Theorem: The speed of the man rowing upstream is 26 kmph -/
theorem upstream_speed_is_26 (s : RowingSpeed)
  (h1 : s.stillWater = 28)
  (h2 : s.downstream = 30) :
  upstreamSpeed s = 26 := by
  sorry

end NUMINAMATH_CALUDE_upstream_speed_is_26_l199_19964


namespace NUMINAMATH_CALUDE_student_ticket_price_l199_19943

theorem student_ticket_price 
  (total_tickets : ℕ) 
  (total_revenue : ℕ) 
  (general_tickets : ℕ) 
  (general_price : ℕ) 
  (h1 : total_tickets = 525)
  (h2 : total_revenue = 2876)
  (h3 : general_tickets = 388)
  (h4 : general_price = 6) :
  ∃ (student_price : ℕ),
    student_price = 4 ∧
    (total_tickets - general_tickets) * student_price + general_tickets * general_price = total_revenue :=
by sorry

end NUMINAMATH_CALUDE_student_ticket_price_l199_19943


namespace NUMINAMATH_CALUDE_min_value_product_l199_19937

theorem min_value_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 1) :
  (2 * x + 3 * y) * (2 * y + 3 * z) * (2 * x * z + 1) ≥ 24 ∧
  ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ x₀ * y₀ * z₀ = 1 ∧
    (2 * x₀ + 3 * y₀) * (2 * y₀ + 3 * z₀) * (2 * x₀ * z₀ + 1) = 24 :=
by sorry

end NUMINAMATH_CALUDE_min_value_product_l199_19937


namespace NUMINAMATH_CALUDE_square_of_rational_l199_19924

theorem square_of_rational (x y : ℚ) (h : x^5 + y^5 = 2*x^2*y^2) :
  ∃ z : ℚ, 1 - x*y = z^2 := by
sorry

end NUMINAMATH_CALUDE_square_of_rational_l199_19924


namespace NUMINAMATH_CALUDE_x_plus_y_value_l199_19989

theorem x_plus_y_value (x y : ℝ) 
  (eq1 : x + Real.cos y = 2010)
  (eq2 : x + 2010 * Real.sin y = 2011)
  (y_range : 0 ≤ y ∧ y ≤ Real.pi) :
  x + y = 2011 + Real.pi := by
  sorry

end NUMINAMATH_CALUDE_x_plus_y_value_l199_19989


namespace NUMINAMATH_CALUDE_quadratic_inequality_theorem_l199_19991

theorem quadratic_inequality_theorem (a : ℝ) :
  (∀ m > a, ∀ x : ℝ, x^2 + 2*x + m > 0) → a = 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_theorem_l199_19991


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l199_19971

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  (∀ x : ℝ, (2*x - 1)^6 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6) →
  a₁ + 2*a₂ + 3*a₃ + 4*a₄ + 5*a₅ + 6*a₆ = 12 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l199_19971


namespace NUMINAMATH_CALUDE_length_function_is_linear_alpha_is_rate_of_change_l199_19969

/-- Represents the length of a metal rod as a function of temperature -/
def length_function (l₀ α : ℝ) (t : ℝ) : ℝ := l₀ * (1 + α * t)

/-- States that the length function is linear in t -/
theorem length_function_is_linear (l₀ α : ℝ) : 
  ∃ m b : ℝ, ∀ t : ℝ, length_function l₀ α t = m * t + b :=
sorry

/-- Defines α as the rate of change of length with respect to temperature -/
theorem alpha_is_rate_of_change (l₀ α : ℝ) : 
  α = (length_function l₀ α 1 - length_function l₀ α 0) / l₀ :=
sorry

end NUMINAMATH_CALUDE_length_function_is_linear_alpha_is_rate_of_change_l199_19969


namespace NUMINAMATH_CALUDE_last_digit_base4_last_digit_390_base4_l199_19988

/-- Convert a natural number to its base-4 representation as a list of digits -/
def toBase4 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) :=
    if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
  aux n []

/-- The last digit of a number in base-4 is the same as the remainder when divided by 4 -/
theorem last_digit_base4 (n : ℕ) : 
  (toBase4 n).getLast? = some (n % 4) :=
sorry

/-- The last digit of 390 in base-4 is 2 -/
theorem last_digit_390_base4 : 
  (toBase4 390).getLast? = some 2 :=
sorry

end NUMINAMATH_CALUDE_last_digit_base4_last_digit_390_base4_l199_19988


namespace NUMINAMATH_CALUDE_chinese_sturgeon_probability_l199_19990

theorem chinese_sturgeon_probability (p_maturity p_spawn_reproduce : ℝ) 
  (h_maturity : p_maturity = 0.15)
  (h_spawn_reproduce : p_spawn_reproduce = 0.05) :
  p_spawn_reproduce / p_maturity = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_chinese_sturgeon_probability_l199_19990


namespace NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l199_19968

theorem complex_number_in_fourth_quadrant :
  let z : ℂ := (1 - 2*I) * (2 + I)
  (z.re > 0) ∧ (z.im < 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l199_19968


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_min_value_sum_reciprocals_equality_condition_l199_19933

theorem min_value_sum_reciprocals (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (sum_eq_9 : a + b + c = 9) : 
  2/a + 2/b + 2/c ≥ 2 := by
  sorry

theorem min_value_sum_reciprocals_equality_condition (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (sum_eq_9 : a + b + c = 9) : 
  2/a + 2/b + 2/c = 2 ↔ a = b ∧ b = c := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_min_value_sum_reciprocals_equality_condition_l199_19933


namespace NUMINAMATH_CALUDE_intersection_distance_l199_19918

/-- The distance between the intersection points of y = -2 and y = 3x^2 + 2x - 5 -/
theorem intersection_distance : 
  let f (x : ℝ) := 3 * x^2 + 2 * x - 5
  let y := -2
  let roots := {x : ℝ | f x = y}
  ∃ (x₁ x₂ : ℝ), x₁ ∈ roots ∧ x₂ ∈ roots ∧ x₁ ≠ x₂ ∧ |x₁ - x₂| = 2 * Real.sqrt 10 / 3 :=
by sorry

end NUMINAMATH_CALUDE_intersection_distance_l199_19918


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l199_19972

theorem imaginary_part_of_complex_fraction : 
  Complex.im (Complex.I^3 / (2 * Complex.I - 1)) = 1/5 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l199_19972


namespace NUMINAMATH_CALUDE_function_composition_value_l199_19994

/-- Given a function g and a composition f[g(x)], prove that f(0) = 4/5 -/
theorem function_composition_value (g : ℝ → ℝ) (f : ℝ → ℝ) :
  (∀ x, g x = 1 - 3 * x) →
  (∀ x, f (g x) = (1 - x^2) / (1 + x^2)) →
  f 0 = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_function_composition_value_l199_19994


namespace NUMINAMATH_CALUDE_password_length_l199_19999

-- Define the structure of the password
structure PasswordStructure where
  lowercase_letters : Nat
  uppercase_and_numbers : Nat
  digits : Nat
  symbols : Nat

-- Define Pat's password structure
def pats_password : PasswordStructure :=
  { lowercase_letters := 12
  , uppercase_and_numbers := 6
  , digits := 4
  , symbols := 2 }

-- Theorem to prove the total number of characters
theorem password_length :
  (pats_password.lowercase_letters +
   pats_password.uppercase_and_numbers +
   pats_password.digits +
   pats_password.symbols) = 24 := by
  sorry

end NUMINAMATH_CALUDE_password_length_l199_19999


namespace NUMINAMATH_CALUDE_card_number_factorization_l199_19978

/-- Represents a set of 90 cards with 10 each of digits 1 through 9 -/
def CardSet := Finset (Fin 9)

/-- Predicate to check if a number can be formed from the given card set -/
def canBeFormedFromCards (n : ℕ) (cards : CardSet) : Prop := sorry

/-- Predicate to check if a number can be factored into four natural factors each greater than one -/
def hasEligibleFactorization (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ), a > 1 ∧ b > 1 ∧ c > 1 ∧ d > 1 ∧ n = a * b * c * d

/-- Main theorem statement -/
theorem card_number_factorization (cards : CardSet) (A B : ℕ) :
  (canBeFormedFromCards A cards) →
  (canBeFormedFromCards B cards) →
  B = 3 * A →
  A > 0 →
  (hasEligibleFactorization A ∨ hasEligibleFactorization B) := by sorry

end NUMINAMATH_CALUDE_card_number_factorization_l199_19978


namespace NUMINAMATH_CALUDE_tan_and_g_alpha_l199_19928

open Real

theorem tan_and_g_alpha (α : ℝ) 
  (h1 : π / 2 < α) (h2 : α < π) 
  (h3 : tan α - (tan α)⁻¹ = -8/3) : 
  tan α = -3 ∧ 
  (sin (π + α) + 4 * cos (2*π + α)) / (sin (π/2 - α) - 4 * sin (-α)) = -7/11 := by
  sorry

end NUMINAMATH_CALUDE_tan_and_g_alpha_l199_19928


namespace NUMINAMATH_CALUDE_probability_of_two_red_balls_l199_19909

-- Define the number of balls of each color
def red_balls : ℕ := 5
def blue_balls : ℕ := 4
def green_balls : ℕ := 3

-- Define the total number of balls
def total_balls : ℕ := red_balls + blue_balls + green_balls

-- Define the number of balls to be picked
def balls_picked : ℕ := 2

-- Theorem statement
theorem probability_of_two_red_balls :
  (Nat.choose red_balls balls_picked : ℚ) / (Nat.choose total_balls balls_picked) = 5 / 33 :=
sorry

end NUMINAMATH_CALUDE_probability_of_two_red_balls_l199_19909


namespace NUMINAMATH_CALUDE_distribute_balls_theorem_l199_19975

/-- The number of ways to distribute 4 different balls into 3 labeled boxes, with no box left empty -/
def distributeWays : ℕ := 36

/-- The number of ways to choose 2 balls from 4 different balls -/
def chooseTwo : ℕ := 6

/-- The number of ways to permute 3 groups -/
def permuteThree : ℕ := 6

theorem distribute_balls_theorem :
  distributeWays = chooseTwo * permuteThree := by sorry

end NUMINAMATH_CALUDE_distribute_balls_theorem_l199_19975


namespace NUMINAMATH_CALUDE_attraction_visit_orders_l199_19935

theorem attraction_visit_orders (n : ℕ) (h : n = 5) : 
  (n! / 2 : ℕ) = 60 := by
  sorry

end NUMINAMATH_CALUDE_attraction_visit_orders_l199_19935


namespace NUMINAMATH_CALUDE_total_cards_l199_19939

theorem total_cards (hockey_cards : ℕ) 
  (h1 : hockey_cards = 200)
  (h2 : ∃ football_cards : ℕ, football_cards = 4 * hockey_cards)
  (h3 : ∃ baseball_cards : ℕ, baseball_cards = football_cards - 50) :
  ∃ total_cards : ℕ, total_cards = hockey_cards + football_cards + baseball_cards ∧ total_cards = 1750 :=
by
  sorry

end NUMINAMATH_CALUDE_total_cards_l199_19939


namespace NUMINAMATH_CALUDE_value_of_c_l199_19957

theorem value_of_c (a b c : ℝ) : 
  8 = 0.04 * a → 
  4 = 0.08 * b → 
  c = b / a → 
  c = 0.25 := by
sorry

end NUMINAMATH_CALUDE_value_of_c_l199_19957


namespace NUMINAMATH_CALUDE_triangle_segment_calculation_l199_19982

/-- Given a triangle ABC with point D on AB and point E on AD, prove that FC has the specified value. -/
theorem triangle_segment_calculation (DC CB AD : ℝ) (h1 : DC = 9) (h2 : CB = 10) 
  (h3 : (1 : ℝ)/5 * AD = AD - DC - CB) (h4 : (3 : ℝ)/4 * AD = ED) : 
  let CA := CB + AD - DC - CB
  let FC := ED * CA / AD
  FC = 11.025 := by sorry

end NUMINAMATH_CALUDE_triangle_segment_calculation_l199_19982


namespace NUMINAMATH_CALUDE_edgar_cookies_count_l199_19904

/-- The number of cookies a paper bag can hold -/
def cookies_per_bag : ℕ := 16

/-- The number of paper bags Edgar needs -/
def bags_needed : ℕ := 19

/-- The total number of cookies Edgar bought -/
def total_cookies : ℕ := cookies_per_bag * bags_needed

theorem edgar_cookies_count : total_cookies = 304 := by
  sorry

end NUMINAMATH_CALUDE_edgar_cookies_count_l199_19904


namespace NUMINAMATH_CALUDE_max_min_sum_zero_l199_19942

def f (x : ℝ) := x^3 - 3*x

theorem max_min_sum_zero :
  ∃ (m n : ℝ), (∀ x, f x ≤ m) ∧ (∃ x₁, f x₁ = m) ∧
                (∀ x, n ≤ f x) ∧ (∃ x₂, f x₂ = n) ∧
                (m + n = 0) := by sorry

end NUMINAMATH_CALUDE_max_min_sum_zero_l199_19942


namespace NUMINAMATH_CALUDE_ad_transmission_cost_l199_19921

/-- The cost of transmitting advertisements during a race -/
theorem ad_transmission_cost
  (num_ads : ℕ)
  (ad_duration : ℕ)
  (cost_per_minute : ℕ)
  (h1 : num_ads = 5)
  (h2 : ad_duration = 3)
  (h3 : cost_per_minute = 4000) :
  num_ads * ad_duration * cost_per_minute = 60000 :=
by sorry

end NUMINAMATH_CALUDE_ad_transmission_cost_l199_19921


namespace NUMINAMATH_CALUDE_theta_range_l199_19917

theorem theta_range (θ : ℝ) : 
  (∀ x ∈ Set.Icc 0 1, x^2 * Real.cos θ - x * (1 - x) + (1 - x^2) * Real.sin θ > 0) →
  ∃ k : ℤ, 2 * k * Real.pi + Real.pi / 6 < θ ∧ θ < 2 * k * Real.pi + Real.pi / 2 :=
by sorry

end NUMINAMATH_CALUDE_theta_range_l199_19917


namespace NUMINAMATH_CALUDE_total_animals_on_yacht_l199_19931

theorem total_animals_on_yacht (cows foxes zebras sheep : ℕ) : 
  cows = 20 → 
  foxes = 15 → 
  zebras = 3 * foxes → 
  sheep = 20 → 
  cows + foxes + zebras + sheep = 100 := by
  sorry

end NUMINAMATH_CALUDE_total_animals_on_yacht_l199_19931


namespace NUMINAMATH_CALUDE_chocobites_remainder_l199_19955

theorem chocobites_remainder (m : ℕ) : 
  m % 8 = 5 → (4 * m) % 8 = 4 := by
sorry

end NUMINAMATH_CALUDE_chocobites_remainder_l199_19955


namespace NUMINAMATH_CALUDE_system_solution_ratio_l199_19946

/-- The system of equations has a nontrivial solution with the given ratio -/
theorem system_solution_ratio :
  ∃ (x y z : ℚ), x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧
  x + (95/9)*y + 4*z = 0 ∧
  4*x + (95/9)*y - 3*z = 0 ∧
  3*x + 5*y - 4*z = 0 ∧
  x*z / (y^2) = 175/81 := by
sorry

end NUMINAMATH_CALUDE_system_solution_ratio_l199_19946


namespace NUMINAMATH_CALUDE_pq_length_in_30_60_90_triangle_l199_19997

/-- Represents a 30-60-90 triangle -/
structure Triangle30_60_90 where
  /-- The length of the hypotenuse -/
  hypotenuse : ℝ
  /-- The length of the side opposite to the 30° angle -/
  short_side : ℝ
  /-- The length of the side opposite to the 60° angle -/
  long_side : ℝ
  /-- The hypotenuse is twice the short side -/
  hypotenuse_twice_short : hypotenuse = 2 * short_side
  /-- The long side is √3 times the short side -/
  long_side_sqrt3_short : long_side = Real.sqrt 3 * short_side

/-- Theorem: In a 30-60-90 triangle PQR where PR = 6√3 and angle QPR = 30°, PQ = 6√3 -/
theorem pq_length_in_30_60_90_triangle (t : Triangle30_60_90) 
  (h : t.hypotenuse = 6 * Real.sqrt 3) : t.long_side = 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_pq_length_in_30_60_90_triangle_l199_19997


namespace NUMINAMATH_CALUDE_earth_surface_cultivation_l199_19929

theorem earth_surface_cultivation (total : ℝ) (water_percentage : ℝ) (land_percentage : ℝ)
  (desert_ice_fraction : ℝ) (pasture_forest_mountain_fraction : ℝ) :
  water_percentage = 70 →
  land_percentage = 30 →
  water_percentage + land_percentage = 100 →
  desert_ice_fraction = 2/5 →
  pasture_forest_mountain_fraction = 1/3 →
  (land_percentage / 100 * total * (1 - desert_ice_fraction - pasture_forest_mountain_fraction)) / total * 100 = 8 :=
by sorry

end NUMINAMATH_CALUDE_earth_surface_cultivation_l199_19929


namespace NUMINAMATH_CALUDE_sequence_and_max_sum_l199_19915

def f (x : ℝ) := -x^2 + 7*x

def S (n : ℕ) := f n

def a (n : ℕ+) := S n - S (n-1)

theorem sequence_and_max_sum :
  (∀ n : ℕ+, a n = -2*(n:ℝ) + 8) ∧
  (∃ n : ℕ+, S n = 12 ∧ ∀ m : ℕ+, S m ≤ 12) :=
sorry

end NUMINAMATH_CALUDE_sequence_and_max_sum_l199_19915


namespace NUMINAMATH_CALUDE_m_plus_n_values_l199_19960

theorem m_plus_n_values (m n : ℤ) 
  (h1 : |m - n| = n - m) 
  (h2 : |m| = 4) 
  (h3 : |n| = 3) : 
  m + n = -1 ∨ m + n = -7 := by
sorry

end NUMINAMATH_CALUDE_m_plus_n_values_l199_19960


namespace NUMINAMATH_CALUDE_hyperbola_center_l199_19995

/-- The center of a hyperbola is the midpoint of its foci -/
theorem hyperbola_center (f₁ f₂ c : ℝ × ℝ) : 
  f₁ = (3, 2) → f₂ = (11, 6) → c = (7, 4) → 
  c = ((f₁.1 + f₂.1) / 2, (f₁.2 + f₂.2) / 2) := by
sorry

end NUMINAMATH_CALUDE_hyperbola_center_l199_19995


namespace NUMINAMATH_CALUDE_haley_trees_l199_19913

/-- The number of trees that died after the typhoon -/
def dead_trees : ℕ := 2

/-- The difference between survived trees and dead trees -/
def survival_difference : ℕ := 7

/-- The total number of trees Haley initially grew -/
def total_trees : ℕ := dead_trees + (dead_trees + survival_difference)

theorem haley_trees : total_trees = 11 := by
  sorry

end NUMINAMATH_CALUDE_haley_trees_l199_19913


namespace NUMINAMATH_CALUDE_distribute_identical_items_l199_19947

theorem distribute_identical_items (n : ℕ) (k : ℕ) :
  n = 10 → k = 3 → Nat.choose (n + k - 1) k = 220 := by
  sorry

end NUMINAMATH_CALUDE_distribute_identical_items_l199_19947


namespace NUMINAMATH_CALUDE_intersection_point_l199_19900

/-- A parabola defined by x = -3y^2 - 4y + 7 -/
def parabola (y : ℝ) : ℝ := -3 * y^2 - 4 * y + 7

/-- The line x = m -/
def line (m : ℝ) : ℝ := m

/-- The condition for a single intersection point -/
def single_intersection (m : ℝ) : Prop :=
  ∃! y : ℝ, parabola y = line m

theorem intersection_point (m : ℝ) : single_intersection m ↔ m = 25/3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_l199_19900


namespace NUMINAMATH_CALUDE_select_real_coins_l199_19980

/-- Represents the outcome of a weighing on a balance scale -/
inductive WeighingResult
| Equal : WeighingResult
| LeftHeavier : WeighingResult
| RightHeavier : WeighingResult

/-- Represents a group of coins -/
structure CoinGroup where
  total : Nat
  counterfeit : Nat

/-- Represents a weighing action on the balance scale -/
def weighing (left right : CoinGroup) : WeighingResult :=
  sorry

/-- Represents the process of selecting coins -/
def selectCoins (coins : CoinGroup) (weighings : Nat) : Option (Finset Nat) :=
  sorry

theorem select_real_coins 
  (total_coins : Nat)
  (counterfeit_coins : Nat)
  (max_weighings : Nat)
  (coins_to_select : Nat)
  (h1 : total_coins = 40)
  (h2 : counterfeit_coins = 3)
  (h3 : max_weighings = 3)
  (h4 : coins_to_select = 16)
  (h5 : counterfeit_coins < total_coins) :
  ∃ (selected : Finset Nat), 
    (selected.card = coins_to_select) ∧ 
    (∀ c ∈ selected, c ≤ total_coins - counterfeit_coins) ∧
    (selectCoins ⟨total_coins, counterfeit_coins⟩ max_weighings = some selected) :=
sorry

end NUMINAMATH_CALUDE_select_real_coins_l199_19980


namespace NUMINAMATH_CALUDE_cherry_price_proof_l199_19922

-- Define the discount rate
def discount_rate : ℝ := 0.3

-- Define the discounted price for a quarter-pound package
def discounted_quarter_pound_price : ℝ := 2

-- Define the weight of a full pound in terms of quarter-pounds
def full_pound_weight : ℝ := 4

-- Define the regular price for a full pound of cherries
def regular_full_pound_price : ℝ := 11.43

theorem cherry_price_proof :
  (1 - discount_rate) * regular_full_pound_price / full_pound_weight = discounted_quarter_pound_price := by
  sorry

end NUMINAMATH_CALUDE_cherry_price_proof_l199_19922


namespace NUMINAMATH_CALUDE_stationery_costs_l199_19966

theorem stationery_costs : ∃ (x y z : ℕ+), 
  (x : ℤ) % 2 = 0 ∧
  x + 3*y + 2*z = 98 ∧
  3*x + y = 5*z - 36 ∧
  x = 4 ∧ y = 22 ∧ z = 14 := by
sorry

end NUMINAMATH_CALUDE_stationery_costs_l199_19966


namespace NUMINAMATH_CALUDE_range_x_when_p_false_range_m_when_p_sufficient_for_q_l199_19938

-- Define propositions p and q
def p (x : ℝ) : Prop := |x - 3| < 1
def q (x m : ℝ) : Prop := m - 2 < x ∧ x < m + 1

-- Part 1: Range of x when p is false
theorem range_x_when_p_false (x : ℝ) :
  ¬(p x) → x ≤ 2 ∨ x ≥ 4 :=
sorry

-- Part 2: Range of m when p is a sufficient condition for q
theorem range_m_when_p_sufficient_for_q (m : ℝ) :
  (∀ x, p x → q x m) → 3 ≤ m ∧ m ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_range_x_when_p_false_range_m_when_p_sufficient_for_q_l199_19938


namespace NUMINAMATH_CALUDE_opposite_to_gold_is_silver_l199_19970

-- Define the colors
inductive Color
| P | M | C | S | G | V | L

-- Define a cube face
structure Face where
  color : Color

-- Define a cube
structure Cube where
  top : Face
  bottom : Face
  front : Face
  back : Face
  left : Face
  right : Face

-- Define the theorem
theorem opposite_to_gold_is_silver (cube : Cube) : 
  cube.top.color = Color.P → 
  cube.bottom.color = Color.V → 
  (cube.front.color = Color.G ∨ cube.back.color = Color.G ∨ cube.left.color = Color.G ∨ cube.right.color = Color.G) → 
  ((cube.front.color = Color.G → cube.back.color = Color.S) ∧ 
   (cube.back.color = Color.G → cube.front.color = Color.S) ∧ 
   (cube.left.color = Color.G → cube.right.color = Color.S) ∧ 
   (cube.right.color = Color.G → cube.left.color = Color.S)) := by
  sorry


end NUMINAMATH_CALUDE_opposite_to_gold_is_silver_l199_19970


namespace NUMINAMATH_CALUDE_arithmetic_sequence_before_five_l199_19956

/-- Given an arithmetic sequence with first term 105 and common difference -5,
    prove that there are 20 terms before the term with value 5. -/
theorem arithmetic_sequence_before_five (n : ℕ) : 
  (105 : ℤ) - 5 * n = 5 → n - 1 = 20 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_before_five_l199_19956


namespace NUMINAMATH_CALUDE_ryan_english_study_hours_l199_19907

/-- Ryan's daily study schedule -/
structure StudySchedule where
  chinese_hours : ℕ
  english_hours : ℕ
  hours_difference : ℕ

/-- Theorem: Ryan's English study hours -/
theorem ryan_english_study_hours (schedule : StudySchedule)
  (h1 : schedule.chinese_hours = 5)
  (h2 : schedule.hours_difference = 2)
  (h3 : schedule.english_hours = schedule.chinese_hours + schedule.hours_difference) :
  schedule.english_hours = 7 := by
  sorry

end NUMINAMATH_CALUDE_ryan_english_study_hours_l199_19907


namespace NUMINAMATH_CALUDE_greatest_divisor_four_consecutive_integers_l199_19976

theorem greatest_divisor_four_consecutive_integers :
  ∀ n : ℕ, n > 0 → 
  ∃ m : ℕ, m = 12 ∧ 
  (∀ k : ℕ, k > m → ¬(k ∣ (n * (n + 1) * (n + 2) * (n + 3)))) ∧
  (12 ∣ (n * (n + 1) * (n + 2) * (n + 3))) :=
by sorry


end NUMINAMATH_CALUDE_greatest_divisor_four_consecutive_integers_l199_19976


namespace NUMINAMATH_CALUDE_smallest_digit_sum_of_sum_l199_19923

/-- Two different two-digit positive integers -/
def is_valid_pair (x y : ℕ) : Prop :=
  10 ≤ x ∧ x < 100 ∧ 10 ≤ y ∧ y < 100 ∧ x ≠ y

/-- All four digits in the two numbers are unique -/
def has_unique_digits (x y : ℕ) : Prop :=
  let digits := [x / 10, x % 10, y / 10, y % 10]
  List.Nodup digits

/-- The sum is a two-digit number -/
def is_two_digit_sum (x y : ℕ) : Prop :=
  10 ≤ x + y ∧ x + y < 100

/-- The sum of digits of a number -/
def digit_sum (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

/-- The main theorem -/
theorem smallest_digit_sum_of_sum :
  ∃ (x y : ℕ), 
    is_valid_pair x y ∧ 
    has_unique_digits x y ∧ 
    is_two_digit_sum x y ∧
    ∀ (a b : ℕ), 
      is_valid_pair a b → 
      has_unique_digits a b → 
      is_two_digit_sum a b → 
      digit_sum (x + y) ≤ digit_sum (a + b) ∧
      digit_sum (x + y) = 10 :=
sorry

end NUMINAMATH_CALUDE_smallest_digit_sum_of_sum_l199_19923


namespace NUMINAMATH_CALUDE_pet_ratio_l199_19963

theorem pet_ratio (dogs : ℕ) (cats : ℕ) (total_pets : ℕ) : 
  dogs = 2 → cats = 3 → total_pets = 15 → 
  (total_pets - (dogs + cats)) * 1 = 2 * (dogs + cats) := by
  sorry

end NUMINAMATH_CALUDE_pet_ratio_l199_19963


namespace NUMINAMATH_CALUDE_min_value_fraction_sum_l199_19951

theorem min_value_fraction_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 3) :
  1 / (a + 1) + 1 / b ≥ 1 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + b₀ = 3 ∧ 1 / (a₀ + 1) + 1 / b₀ = 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_fraction_sum_l199_19951


namespace NUMINAMATH_CALUDE_circular_garden_area_ratio_l199_19927

theorem circular_garden_area_ratio :
  ∀ (original_diameter : ℝ) (original_area enlarged_area : ℝ),
  original_diameter > 0 →
  original_area = π * (original_diameter / 2)^2 →
  enlarged_area = π * ((3 * original_diameter) / 2)^2 →
  original_area / enlarged_area = 1 / 9 :=
by
  sorry

end NUMINAMATH_CALUDE_circular_garden_area_ratio_l199_19927


namespace NUMINAMATH_CALUDE_other_number_when_five_l199_19906

/-- Represents the invariant relation between Peter's numbers -/
def peterInvariant (a b : ℚ) : Prop :=
  2 * a * b - 5 * a - 5 * b = -11

/-- Peter's initial numbers satisfy the invariant -/
axiom initial_invariant : peterInvariant 1 2

/-- The invariant is preserved after each update -/
axiom invariant_preserved (a b c d : ℚ) :
  peterInvariant a b → peterInvariant c d → 
  ∀ p q : ℚ, (∃ m : ℚ, m * (p - a) * (p - b) = (p - c) * (p - d)) →
  peterInvariant p q

/-- When one of Peter's numbers is 5, the other satisfies the invariant -/
theorem other_number_when_five :
  ∃ b : ℚ, peterInvariant 5 b ∧ b = 14/5 := by sorry

end NUMINAMATH_CALUDE_other_number_when_five_l199_19906


namespace NUMINAMATH_CALUDE_sin_cos_difference_equals_half_l199_19920

theorem sin_cos_difference_equals_half : 
  Real.sin (43 * π / 180) * Real.cos (13 * π / 180) - 
  Real.sin (13 * π / 180) * Real.cos (43 * π / 180) = 1/2 := by sorry

end NUMINAMATH_CALUDE_sin_cos_difference_equals_half_l199_19920


namespace NUMINAMATH_CALUDE_set_equality_implies_a_value_l199_19936

/-- Given two sets are equal, prove that a must be either 1 or -1 -/
theorem set_equality_implies_a_value (a : ℝ) : 
  ({0, -1, 2*a} : Set ℝ) = ({a-1, -abs a, a+1} : Set ℝ) → 
  a = 1 ∨ a = -1 := by
sorry

end NUMINAMATH_CALUDE_set_equality_implies_a_value_l199_19936


namespace NUMINAMATH_CALUDE_expression_simplification_l199_19912

theorem expression_simplification (a b : ℝ) 
  (ha : a = Real.sqrt 3 - Real.sqrt 11) 
  (hb : b = Real.sqrt 3 + Real.sqrt 11) : 
  (a^2 - b^2) / (a^2 * b - a * b^2) / (1 + (a^2 + b^2) / (2 * a * b)) = Real.sqrt 3 / 6 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l199_19912


namespace NUMINAMATH_CALUDE_container_capacity_problem_l199_19903

/-- Represents a rectangular container with dimensions and capacity -/
structure Container where
  height : ℝ
  width : ℝ
  length : ℝ
  capacity : ℝ

/-- The problem statement -/
theorem container_capacity_problem (c1 c2 : Container) :
  c1.height = 4 →
  c1.width = 2 →
  c1.length = 8 →
  c1.capacity = 64 →
  c2.height = 3 * c1.height →
  c2.width = 2 * c1.width →
  c2.length = c1.length →
  c2.capacity = 384 := by
  sorry

end NUMINAMATH_CALUDE_container_capacity_problem_l199_19903


namespace NUMINAMATH_CALUDE_premier_pups_count_l199_19944

theorem premier_pups_count :
  let fetch : ℕ := 70
  let jump : ℕ := 40
  let bark : ℕ := 45
  let fetch_and_jump : ℕ := 25
  let jump_and_bark : ℕ := 15
  let fetch_and_bark : ℕ := 20
  let all_three : ℕ := 12
  let none : ℕ := 15
  
  let fetch_only : ℕ := fetch - (fetch_and_jump + fetch_and_bark - all_three)
  let jump_only : ℕ := jump - (fetch_and_jump + jump_and_bark - all_three)
  let bark_only : ℕ := bark - (fetch_and_bark + jump_and_bark - all_three)
  let fetch_jump_only : ℕ := fetch_and_jump - all_three
  let jump_bark_only : ℕ := jump_and_bark - all_three
  let fetch_bark_only : ℕ := fetch_and_bark - all_three

  fetch_only + jump_only + bark_only + fetch_jump_only + jump_bark_only + fetch_bark_only + all_three + none = 122 := by
  sorry

end NUMINAMATH_CALUDE_premier_pups_count_l199_19944


namespace NUMINAMATH_CALUDE_regular_pay_is_three_l199_19959

/-- Calculates the regular hourly pay rate given total pay, regular hours, overtime hours, and overtime pay rate multiplier. -/
def regularHourlyPay (totalPay : ℚ) (regularHours : ℚ) (overtimeHours : ℚ) (overtimeMultiplier : ℚ) : ℚ :=
  totalPay / (regularHours + overtimeHours * overtimeMultiplier)

/-- Proves that the regular hourly pay is $3 given the problem conditions. -/
theorem regular_pay_is_three :
  let totalPay : ℚ := 192
  let regularHours : ℚ := 40
  let overtimeHours : ℚ := 12
  let overtimeMultiplier : ℚ := 2
  regularHourlyPay totalPay regularHours overtimeHours overtimeMultiplier = 3 := by
  sorry

#eval regularHourlyPay 192 40 12 2

end NUMINAMATH_CALUDE_regular_pay_is_three_l199_19959


namespace NUMINAMATH_CALUDE_estate_value_l199_19974

/-- Represents the distribution of Mr. T's estate -/
structure EstateDistribution where
  total : ℝ
  wife_share : ℝ
  daughter1_share : ℝ
  daughter2_share : ℝ
  son_share : ℝ
  gardener_share : ℝ

/-- Defines the conditions of Mr. T's estate distribution -/
def valid_distribution (e : EstateDistribution) : Prop :=
  -- Two daughters and son received 3/4 of the estate
  e.daughter1_share + e.daughter2_share + e.son_share = 3/4 * e.total ∧
  -- Daughters shared their portion in the ratio of 5:3
  e.daughter1_share / e.daughter2_share = 5/3 ∧
  -- Wife received thrice as much as the son
  e.wife_share = 3 * e.son_share ∧
  -- Gardener received $600
  e.gardener_share = 600 ∧
  -- Sum of wife and gardener's shares was 1/4 of the estate
  e.wife_share + e.gardener_share = 1/4 * e.total ∧
  -- Total is sum of all shares
  e.total = e.wife_share + e.daughter1_share + e.daughter2_share + e.son_share + e.gardener_share

/-- Theorem stating that Mr. T's estate value is $2400 -/
theorem estate_value (e : EstateDistribution) (h : valid_distribution e) : e.total = 2400 :=
  sorry


end NUMINAMATH_CALUDE_estate_value_l199_19974


namespace NUMINAMATH_CALUDE_arithmetic_progression_max_first_term_l199_19916

theorem arithmetic_progression_max_first_term 
  (b₁ : ℚ) 
  (d : ℚ) 
  (S₄ S₉ : ℕ) :
  (4 * b₁ + 6 * d = S₄) →
  (9 * b₁ + 36 * d = S₉) →
  (b₁ ≤ 3/4) →
  (∀ b₁' d' S₄' S₉' : ℚ, 
    (4 * b₁' + 6 * d' = S₄') →
    (9 * b₁' + 36 * d' = S₉') →
    (b₁' ≤ 3/4) →
    (b₁' ≤ b₁)) →
  b₁ = 11/15 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_max_first_term_l199_19916


namespace NUMINAMATH_CALUDE_tom_gave_balloons_to_fred_l199_19986

/-- The number of balloons Tom gave to Fred -/
def balloons_given (initial : ℕ) (remaining : ℕ) : ℕ :=
  initial - remaining

theorem tom_gave_balloons_to_fred (initial : ℕ) (remaining : ℕ) 
  (h1 : initial = 30) (h2 : remaining = 14) :
  balloons_given initial remaining = 16 := by
  sorry

end NUMINAMATH_CALUDE_tom_gave_balloons_to_fred_l199_19986


namespace NUMINAMATH_CALUDE_field_trip_buses_l199_19910

theorem field_trip_buses (total_people : ℕ) (num_vans : ℕ) (people_per_van : ℕ) (people_per_bus : ℕ) 
  (h1 : total_people = 76)
  (h2 : num_vans = 2)
  (h3 : people_per_van = 8)
  (h4 : people_per_bus = 20) :
  (total_people - num_vans * people_per_van) / people_per_bus = 3 := by
  sorry

end NUMINAMATH_CALUDE_field_trip_buses_l199_19910


namespace NUMINAMATH_CALUDE_order_of_a_b_c_l199_19911

theorem order_of_a_b_c :
  let a := (2 : ℝ) ^ (9/10)
  let b := (3 : ℝ) ^ (2/3)
  let c := Real.log 3 / Real.log (1/2)
  b > a ∧ a > c := by sorry

end NUMINAMATH_CALUDE_order_of_a_b_c_l199_19911


namespace NUMINAMATH_CALUDE_min_value_theorem_l199_19977

theorem min_value_theorem (x y : ℝ) (h : x + y = 5) :
  ∃ m : ℝ, m = (6100 : ℝ) / 17 ∧ 
  ∀ z : ℝ, z ≥ m ∧ ∃ a b : ℝ, a + b = 5 ∧ 
  z = a^5*b + a^4*b + a^3*b + a^2*b + a*b + a*b^2 + a*b^3 + a*b^4 + 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l199_19977


namespace NUMINAMATH_CALUDE_triangle_inequality_l199_19901

theorem triangle_inequality (A B C a b c : Real) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧ 
  A < π/2 ∧ B < π/2 ∧ C < π/2 ∧
  a = 2 * Real.sin A ∧
  b = 2 * Real.sin B ∧
  c = 2 * Real.sin C →
  (a / (1 - Real.sin A)) + (b / (1 - Real.sin B)) + (c / (1 - Real.sin C)) ≥ 18 + 12 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_l199_19901


namespace NUMINAMATH_CALUDE_sara_remaining_pears_l199_19998

-- Define the initial number of pears Sara picked
def initial_pears : ℕ := 35

-- Define the number of pears Sara gave to Dan
def pears_given : ℕ := 28

-- Theorem to prove
theorem sara_remaining_pears :
  initial_pears - pears_given = 7 := by
  sorry

end NUMINAMATH_CALUDE_sara_remaining_pears_l199_19998


namespace NUMINAMATH_CALUDE_probability_negative_product_l199_19952

def S : Finset Int := {-6, -3, -1, 2, 5, 8}

def negative_product_pairs (S : Finset Int) : Finset (Int × Int) :=
  S.product S |>.filter (fun (a, b) => a ≠ b ∧ a * b < 0)

def total_pairs (S : Finset Int) : Finset (Int × Int) :=
  S.product S |>.filter (fun (a, b) => a ≠ b)

theorem probability_negative_product :
  (negative_product_pairs S).card / (total_pairs S).card = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_negative_product_l199_19952


namespace NUMINAMATH_CALUDE_binary_multiplication_correct_l199_19914

/-- Represents a binary number as a list of booleans, where true represents 1 and false represents 0. 
    The least significant bit is at the head of the list. -/
def BinaryNumber := List Bool

/-- Converts a binary number to its decimal representation -/
def binary_to_decimal (b : BinaryNumber) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

/-- Multiplies two binary numbers -/
def binary_multiply (a b : BinaryNumber) : BinaryNumber :=
  sorry -- Implementation details omitted

theorem binary_multiplication_correct :
  let a : BinaryNumber := [true, true, false, true]  -- 1011₂
  let b : BinaryNumber := [true, false, true]        -- 101₂
  let result : BinaryNumber := [true, true, true, false, true, true]  -- 110111₂
  binary_multiply a b = result ∧ 
  binary_to_decimal (binary_multiply a b) = binary_to_decimal a * binary_to_decimal b :=
by sorry

end NUMINAMATH_CALUDE_binary_multiplication_correct_l199_19914


namespace NUMINAMATH_CALUDE_ab_plus_cd_value_l199_19965

theorem ab_plus_cd_value (a b c d : ℝ) 
  (eq1 : a + b + c = 5)
  (eq2 : a + b + d = 1)
  (eq3 : a + c + d = 16)
  (eq4 : b + c + d = 9) :
  a * b + c * d = 734 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ab_plus_cd_value_l199_19965


namespace NUMINAMATH_CALUDE_f_is_integer_l199_19905

def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

def f (m n : ℕ) : ℚ :=
  (factorial (2 * m) * factorial (2 * n)) / (factorial m * factorial n * factorial (m + n))

theorem f_is_integer (m n : ℕ) : ∃ k : ℤ, f m n = k :=
sorry

end NUMINAMATH_CALUDE_f_is_integer_l199_19905


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l199_19925

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h : a > b
  k : b > 0

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola) : ℝ := sorry

/-- Theorem: Eccentricity of a special hyperbola -/
theorem hyperbola_eccentricity (h : Hyperbola) 
  (hexagon_condition : ∃ (c : ℝ), c > 0 ∧ 
    (∃ (x y : ℝ), x^2 / h.a^2 - y^2 / h.b^2 = 1 ∧ 
      x^2 + y^2 = c^2 ∧ 
      -- The following condition represents that the intersections form a regular hexagon
      2 * h.a = (Real.sqrt 3 - 1) * c)) :
  eccentricity h = Real.sqrt 3 + 1 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l199_19925


namespace NUMINAMATH_CALUDE_height_difference_l199_19996

def pine_height : ℚ := 12 + 4/5
def birch_height : ℚ := 18 + 1/2
def maple_height : ℚ := 14 + 3/5

def tallest_height : ℚ := max (max pine_height birch_height) maple_height
def shortest_height : ℚ := min (min pine_height birch_height) maple_height

theorem height_difference :
  tallest_height - shortest_height = 7 + 7/10 := by sorry

end NUMINAMATH_CALUDE_height_difference_l199_19996


namespace NUMINAMATH_CALUDE_total_texts_sent_l199_19934

/-- The number of texts Sydney sent to Allison, Brittney, and Carol over three days -/
theorem total_texts_sent (
  monday_allison monday_brittney monday_carol : ℕ)
  (tuesday_allison tuesday_brittney tuesday_carol : ℕ)
  (wednesday_allison wednesday_brittney wednesday_carol : ℕ)
  (h1 : monday_allison = 5 ∧ monday_brittney = 5 ∧ monday_carol = 5)
  (h2 : tuesday_allison = 15 ∧ tuesday_brittney = 10 ∧ tuesday_carol = 12)
  (h3 : wednesday_allison = 20 ∧ wednesday_brittney = 18 ∧ wednesday_carol = 7) :
  monday_allison + monday_brittney + monday_carol +
  tuesday_allison + tuesday_brittney + tuesday_carol +
  wednesday_allison + wednesday_brittney + wednesday_carol = 97 :=
by sorry

end NUMINAMATH_CALUDE_total_texts_sent_l199_19934


namespace NUMINAMATH_CALUDE_two_plus_three_equals_twentysix_l199_19940

/-- Defines the sequence operation for two consecutive terms -/
def sequenceOperation (a b : ℕ) : ℕ := (a + b)^2 + 1

/-- Theorem stating that 2 + 3 in the given sequence equals 26 -/
theorem two_plus_three_equals_twentysix :
  sequenceOperation 2 3 = 26 := by
  sorry

end NUMINAMATH_CALUDE_two_plus_three_equals_twentysix_l199_19940


namespace NUMINAMATH_CALUDE_angle_A_is_pi_over_three_perimeter_range_l199_19948

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  -- Triangle inequality
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  ineq_a : a < b + c
  ineq_b : b < a + c
  ineq_c : c < a + b
  -- Angle sum is π
  angle_sum : A + B + C = π
  -- Sine rule
  sine_rule_a : a / Real.sin A = b / Real.sin B
  sine_rule_b : b / Real.sin B = c / Real.sin C
  -- Cosine rule
  cosine_rule_a : a^2 = b^2 + c^2 - 2*b*c*Real.cos A
  cosine_rule_b : b^2 = a^2 + c^2 - 2*a*c*Real.cos B
  cosine_rule_c : c^2 = a^2 + b^2 - 2*a*b*Real.cos C

theorem angle_A_is_pi_over_three (t : Triangle) (h : t.a * Real.cos t.C + (1/2) * t.c = t.b) :
  t.A = π/3 := by sorry

theorem perimeter_range (t : Triangle) (h1 : t.a = 1) (h2 : t.a * Real.cos t.C + (1/2) * t.c = t.b) :
  2 < t.a + t.b + t.c ∧ t.a + t.b + t.c ≤ 3 := by sorry

end NUMINAMATH_CALUDE_angle_A_is_pi_over_three_perimeter_range_l199_19948


namespace NUMINAMATH_CALUDE_max_value_constraint_l199_19961

theorem max_value_constraint (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 3 * x + 4 * y < 72) :
  x * y * (72 - 3 * x - 4 * y) ≤ 1152 := by
sorry

end NUMINAMATH_CALUDE_max_value_constraint_l199_19961


namespace NUMINAMATH_CALUDE_jello_cost_is_270_l199_19926

/-- The cost to fill a bathtub with jello -/
def jello_cost (jello_per_pound : Real) (tub_volume : Real) (cubic_foot_to_gallon : Real) (pounds_per_gallon : Real) (cost_per_tablespoon : Real) : Real :=
  jello_per_pound * tub_volume * cubic_foot_to_gallon * pounds_per_gallon * cost_per_tablespoon

/-- Theorem: The cost to fill the bathtub with jello is $270 -/
theorem jello_cost_is_270 :
  jello_cost 1.5 6 7.5 8 0.5 = 270 := by
  sorry

end NUMINAMATH_CALUDE_jello_cost_is_270_l199_19926


namespace NUMINAMATH_CALUDE_hundred_to_fifty_zeros_l199_19949

theorem hundred_to_fifty_zeros (n : ℕ) : 100^50 = 10^100 := by
  sorry

end NUMINAMATH_CALUDE_hundred_to_fifty_zeros_l199_19949


namespace NUMINAMATH_CALUDE_f_2021_2_l199_19958

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem f_2021_2 (f : ℝ → ℝ) 
  (h1 : is_even_function f)
  (h2 : ∀ x, f (x + 2) = -f x)
  (h3 : ∀ x ∈ Set.Ioo 1 2, f x = 2^x) :
  f (2021/2) = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_f_2021_2_l199_19958
