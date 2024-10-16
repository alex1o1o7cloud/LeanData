import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l429_42935

theorem quadratic_equation_roots (p q : ℝ) (a b : ℝ) : 
  (a^2 + p*a + q = 0) → 
  (b^2 + p*b + q = 0) → 
  ∃ y₁ y₂ : ℝ, 
    (y₁ = (a+b)^2 ∧ y₂ = (a-b)^2) ∧ 
    (y₁^2 - 2*(p^2 - 2*q)*y₁ + (p^4 - 4*q*p^2) = 0) ∧
    (y₂^2 - 2*(p^2 - 2*q)*y₂ + (p^4 - 4*q*p^2) = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l429_42935


namespace NUMINAMATH_CALUDE_find_a_and_b_l429_42981

/-- Set A defined by the equation ax - y² + b = 0 -/
def A (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | a * p.1 - p.2^2 + b = 0}

/-- Set B defined by the equation x² - ay - b = 0 -/
def B (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 - a * p.2 - b = 0}

/-- Theorem stating that a = -3 and b = 7 given the conditions -/
theorem find_a_and_b :
  ∃ (a b : ℝ), (1, 2) ∈ A a b ∩ B a b ∧ a = -3 ∧ b = 7 := by
  sorry

end NUMINAMATH_CALUDE_find_a_and_b_l429_42981


namespace NUMINAMATH_CALUDE_complement_of_M_l429_42996

def M : Set ℝ := {a : ℝ | a^2 - 2*a > 0}

theorem complement_of_M : 
  {a : ℝ | a ∉ M} = Set.Icc 0 2 := by sorry

end NUMINAMATH_CALUDE_complement_of_M_l429_42996


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_one_range_of_a_for_all_real_solution_l429_42995

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |a*x - 1| + |a*x - 3*a|

-- Part 1
theorem solution_set_when_a_is_one :
  {x : ℝ | |x - 1| + |x - 3| ≥ 5} = {x : ℝ | x ≥ 9/2 ∨ x ≤ -1/2} := by sorry

-- Part 2
theorem range_of_a_for_all_real_solution :
  {a : ℝ | a > 0 ∧ ∀ x, f a x ≥ 5} = {a : ℝ | a ≥ 2} := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_one_range_of_a_for_all_real_solution_l429_42995


namespace NUMINAMATH_CALUDE_gcf_lcm_sum_l429_42954

def numbers : List Nat := [9, 15, 27]

/-- The greatest common factor of 9, 15, and 27 -/
def A : Nat := numbers.foldl Nat.gcd 0

/-- The least common multiple of 9, 15, and 27 -/
def B : Nat := numbers.foldl Nat.lcm 1

/-- Theorem stating that the sum of the greatest common factor and 
    the least common multiple of 9, 15, and 27 is equal to 138 -/
theorem gcf_lcm_sum : A + B = 138 := by
  sorry

end NUMINAMATH_CALUDE_gcf_lcm_sum_l429_42954


namespace NUMINAMATH_CALUDE_sum_of_y_values_l429_42986

theorem sum_of_y_values (y : ℝ) : 
  (∃ (y₁ y₂ : ℝ), 
    (Real.sqrt ((y₁ - 2)^2) = 9 ∧ 
     Real.sqrt ((y₂ - 2)^2) = 9 ∧ 
     y₁ ≠ y₂ ∧
     (∀ y', Real.sqrt ((y' - 2)^2) = 9 → y' = y₁ ∨ y' = y₂)) →
    y₁ + y₂ = 4) :=
sorry

end NUMINAMATH_CALUDE_sum_of_y_values_l429_42986


namespace NUMINAMATH_CALUDE_broken_flagpole_tip_height_l429_42977

/-- Represents a broken flagpole -/
structure BrokenFlagpole where
  initial_height : ℝ
  break_height : ℝ
  folds_in_half : Bool

/-- Calculates the height of the tip of a broken flagpole from the ground -/
def tip_height (f : BrokenFlagpole) : ℝ :=
  if f.folds_in_half then f.break_height else f.initial_height

/-- Theorem stating that the height of the tip of a broken flagpole is equal to the break height -/
theorem broken_flagpole_tip_height 
  (f : BrokenFlagpole) 
  (h1 : f.initial_height = 12)
  (h2 : f.break_height = 7)
  (h3 : f.folds_in_half = true) :
  tip_height f = 7 := by
  sorry

end NUMINAMATH_CALUDE_broken_flagpole_tip_height_l429_42977


namespace NUMINAMATH_CALUDE_sheep_purchase_equation_l429_42962

/-- Represents a group of people jointly buying sheep -/
structure SheepPurchase where
  x : ℕ  -- number of people
  price : ℕ  -- price of the sheep

/-- The equation holds for the given conditions -/
theorem sheep_purchase_equation (sp : SheepPurchase) : 
  (5 * sp.x + 45 = sp.price) ∧ (7 * sp.x - 3 = sp.price) → 5 * sp.x + 45 = 7 * sp.x + 3 :=
by
  sorry

end NUMINAMATH_CALUDE_sheep_purchase_equation_l429_42962


namespace NUMINAMATH_CALUDE_sum_x_y_z_l429_42972

def x : ℕ := (List.range 11).map (· + 30) |>.sum

def y : ℕ := (List.range 11).map (· + 30) |>.filter (· % 2 = 0) |>.length

def z : ℕ := (List.range 11).map (· + 30) |>.filter (· % 2 ≠ 0) |>.prod

theorem sum_x_y_z : x + y + z = 51768016 := by
  sorry

end NUMINAMATH_CALUDE_sum_x_y_z_l429_42972


namespace NUMINAMATH_CALUDE_tangent_parallel_point_l429_42973

theorem tangent_parallel_point (x y : ℝ) : 
  y = Real.exp x → -- Point A (x, y) is on the curve y = e^x
  (Real.exp x) = 1 → -- Tangent at A is parallel to x - y + 3 = 0 (slope = 1)
  x = 0 ∧ y = 1 := by
sorry

end NUMINAMATH_CALUDE_tangent_parallel_point_l429_42973


namespace NUMINAMATH_CALUDE_evaluate_expression_l429_42975

theorem evaluate_expression : (1023 : ℕ) * 1023 - 1022 * 1024 = 1 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l429_42975


namespace NUMINAMATH_CALUDE_penguin_count_l429_42960

/-- The number of penguins in a zoo can be determined by the sum of penguins
    that have already been fed and those that still need to be fed. -/
theorem penguin_count (total_fish : ℕ) (fed_penguins : ℕ) (to_be_fed : ℕ) :
  total_fish ≥ fed_penguins + to_be_fed →
  fed_penguins + to_be_fed = fed_penguins + to_be_fed :=
by
  sorry

#check penguin_count

end NUMINAMATH_CALUDE_penguin_count_l429_42960


namespace NUMINAMATH_CALUDE_william_final_napkins_l429_42938

def napkin_problem (initial_napkins : ℕ) (olivia_napkins : ℕ) : ℕ :=
  let amelia_napkins := 2 * olivia_napkins
  let charlie_napkins := amelia_napkins / 2
  let georgia_napkins := 3 * charlie_napkins
  initial_napkins + olivia_napkins + amelia_napkins + charlie_napkins + georgia_napkins

theorem william_final_napkins :
  napkin_problem 15 10 = 85 := by
  sorry

end NUMINAMATH_CALUDE_william_final_napkins_l429_42938


namespace NUMINAMATH_CALUDE_max_u_coordinate_is_two_l429_42941

-- Define the transformation
def transform (x y : ℝ) : ℝ × ℝ :=
  (x^2 + y^2, x - y)

-- Define the unit square vertices
def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (1, 1)
def C : ℝ × ℝ := (0, 1)

-- Define the set of points in the unit square
def unitSquare : Set (ℝ × ℝ) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1}

-- Theorem: The maximum u-coordinate of the transformed unit square is 2
theorem max_u_coordinate_is_two :
  ∃ (p : ℝ × ℝ), p ∈ unitSquare ∧
    (∀ (q : ℝ × ℝ), q ∈ unitSquare →
      (transform p.1 p.2).1 ≥ (transform q.1 q.2).1) ∧
    (transform p.1 p.2).1 = 2 :=
  sorry

end NUMINAMATH_CALUDE_max_u_coordinate_is_two_l429_42941


namespace NUMINAMATH_CALUDE_g_of_3_eq_200_l429_42925

-- Define the function g
def g (x : ℝ) : ℝ := 9 * x^3 - 5 * x^2 + 3 * x - 7

-- Theorem statement
theorem g_of_3_eq_200 : g 3 = 200 := by
  sorry

end NUMINAMATH_CALUDE_g_of_3_eq_200_l429_42925


namespace NUMINAMATH_CALUDE_least_N_mod_1000_l429_42914

/-- Sum of digits in base-five representation -/
def f (n : ℕ) : ℕ := sorry

/-- Sum of digits in base-seven representation -/
def g (n : ℕ) : ℕ := sorry

/-- The least value of n such that g(n) ≥ 10 -/
def N : ℕ := sorry

theorem least_N_mod_1000 : N % 1000 = 781 := by sorry

end NUMINAMATH_CALUDE_least_N_mod_1000_l429_42914


namespace NUMINAMATH_CALUDE_sin_pi_minus_alpha_l429_42943

theorem sin_pi_minus_alpha (α : Real) :
  (∃ (x y : Real), x = -4 ∧ y = 3 ∧ x = r * Real.cos α ∧ y = r * Real.sin α ∧ r > 0) →
  Real.sin (Real.pi - α) = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_sin_pi_minus_alpha_l429_42943


namespace NUMINAMATH_CALUDE_josh_marbles_l429_42936

/-- The number of marbles Josh initially had -/
def initial_marbles : ℕ := 19

/-- The number of marbles Josh lost -/
def lost_marbles : ℕ := 11

/-- The number of marbles Josh has now -/
def current_marbles : ℕ := initial_marbles - lost_marbles

/-- Theorem stating that Josh now has 8 marbles -/
theorem josh_marbles : current_marbles = 8 := by
  sorry

end NUMINAMATH_CALUDE_josh_marbles_l429_42936


namespace NUMINAMATH_CALUDE_max_profit_at_one_MP_decreasing_max_profit_x_eq_one_l429_42921

-- Define the profit function
def P (x : ℕ) : ℚ := -0.2 * x^2 + 25 * x - 40

-- Define the marginal profit function
def MP (x : ℕ) : ℚ := P (x + 1) - P x

-- State the theorem
theorem max_profit_at_one :
  ∀ x : ℕ, 1 ≤ x → x ≤ 100 → P 1 ≥ P x ∧ P 1 = 24.4 := by
  sorry

-- Prove that MP is decreasing
theorem MP_decreasing :
  ∀ x y : ℕ, x < y → MP x > MP y := by
  sorry

-- Prove that maximum profit occurs at x = 1
theorem max_profit_x_eq_one :
  ∃ x : ℕ, 1 ≤ x ∧ x ≤ 100 ∧ ∀ y : ℕ, 1 ≤ y ∧ y ≤ 100 → P x ≥ P y := by
  sorry

end NUMINAMATH_CALUDE_max_profit_at_one_MP_decreasing_max_profit_x_eq_one_l429_42921


namespace NUMINAMATH_CALUDE_age_sum_l429_42909

theorem age_sum (patrick michael monica : ℕ) 
  (h1 : 3 * michael = 5 * patrick)
  (h2 : 3 * monica = 5 * michael)
  (h3 : monica - patrick = 32) : 
  patrick + michael + monica = 98 := by
sorry

end NUMINAMATH_CALUDE_age_sum_l429_42909


namespace NUMINAMATH_CALUDE_triangle_ratio_theorem_l429_42966

theorem triangle_ratio_theorem (A B C : Real) (a b c : Real) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  -- Sides a, b, c are opposite to angles A, B, C respectively
  a > 0 ∧ b > 0 ∧ c > 0 →
  -- Given conditions
  b * Real.cos C + c * Real.sin B = a →
  b = 6 →
  -- Theorem statement
  (a + 2*b) / (Real.sin A + 2 * Real.sin B) = 6 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_ratio_theorem_l429_42966


namespace NUMINAMATH_CALUDE_stock_price_increase_l429_42971

theorem stock_price_increase (initial_price : ℝ) (h : initial_price > 0) : 
  let price_after_year1 := initial_price * 1.2
  let price_after_year2 := price_after_year1 * 0.75
  let price_after_year3 := initial_price * 1.26
  (price_after_year3 / price_after_year2 - 1) * 100 = 40 := by
sorry

end NUMINAMATH_CALUDE_stock_price_increase_l429_42971


namespace NUMINAMATH_CALUDE_inequality_equivalence_l429_42988

theorem inequality_equivalence (x : ℝ) (h : x ≠ 2) :
  (x - 1) / (x - 2) ≤ 0 ↔ (x^3 - x^2 + x - 1) / (x - 2) ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l429_42988


namespace NUMINAMATH_CALUDE_largest_multiple_of_18_with_8_and_0_l429_42917

def is_valid_number (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d = 8 ∨ d = 0

theorem largest_multiple_of_18_with_8_and_0 :
  ∃ m : ℕ,
    m > 0 ∧
    m % 18 = 0 ∧
    is_valid_number m ∧
    (∀ k : ℕ, k > m → k % 18 = 0 → ¬is_valid_number k) ∧
    m / 18 = 493826048 :=
sorry

end NUMINAMATH_CALUDE_largest_multiple_of_18_with_8_and_0_l429_42917


namespace NUMINAMATH_CALUDE_functional_equation_solution_l429_42958

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f x * f y = f (x - y)) →
  ((∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = 1)) :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l429_42958


namespace NUMINAMATH_CALUDE_mersenne_prime_implies_exponent_prime_l429_42920

theorem mersenne_prime_implies_exponent_prime (n : ℕ) : 
  Prime (2^n - 1) → Prime n := by
  sorry

end NUMINAMATH_CALUDE_mersenne_prime_implies_exponent_prime_l429_42920


namespace NUMINAMATH_CALUDE_bag_to_items_ratio_l429_42947

/-- The cost of a shirt in dollars -/
def shirt_cost : ℚ := 7

/-- The cost of a pair of shoes in dollars -/
def shoes_cost : ℚ := shirt_cost + 3

/-- The total cost of 2 shirts and a pair of shoes in dollars -/
def total_cost_without_bag : ℚ := 2 * shirt_cost + shoes_cost

/-- The total cost of all items (including the bag) in dollars -/
def total_cost : ℚ := 36

/-- The cost of the bag in dollars -/
def bag_cost : ℚ := total_cost - total_cost_without_bag

/-- Theorem stating that the ratio of the bag cost to the total cost without bag is 1:2 -/
theorem bag_to_items_ratio :
  bag_cost / total_cost_without_bag = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_bag_to_items_ratio_l429_42947


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_difference_l429_42985

theorem absolute_value_equation_solution_difference : ∃ (x₁ x₂ : ℝ), 
  (|x₁ + 3| = 15) ∧ 
  (|x₂ + 3| = 15) ∧ 
  (x₁ ≠ x₂) ∧
  (x₁ - x₂ = 30 ∨ x₂ - x₁ = 30) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_difference_l429_42985


namespace NUMINAMATH_CALUDE_correct_calculation_l429_42959

theorem correct_calculation (m : ℝ) : 2 * m^3 * 3 * m^2 = 6 * m^5 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l429_42959


namespace NUMINAMATH_CALUDE_average_salary_increase_proof_l429_42905

def average_salary_increase 
  (initial_employees : ℕ) 
  (initial_average_salary : ℚ) 
  (manager_salary : ℚ) : ℚ :=
  let total_initial_salary := initial_employees * initial_average_salary
  let new_total_salary := total_initial_salary + manager_salary
  let new_average_salary := new_total_salary / (initial_employees + 1)
  new_average_salary - initial_average_salary

theorem average_salary_increase_proof :
  average_salary_increase 24 1500 11500 = 400 := by
  sorry

end NUMINAMATH_CALUDE_average_salary_increase_proof_l429_42905


namespace NUMINAMATH_CALUDE_polygon_sides_count_l429_42952

-- Define a convex polygon with n sides
def ConvexPolygon (n : ℕ) := n ≥ 3

-- Define the sum of interior angles of a polygon
def SumOfInteriorAngles (n : ℕ) : ℝ := 180 * (n - 2)

-- Define the theorem
theorem polygon_sides_count 
  (n : ℕ) 
  (h_convex : ConvexPolygon n) 
  (h_sum : SumOfInteriorAngles n - (2 * (SumOfInteriorAngles n / (n - 1)) - 20) = 2790) :
  n = 18 := by sorry

end NUMINAMATH_CALUDE_polygon_sides_count_l429_42952


namespace NUMINAMATH_CALUDE_max_product_at_12_l429_42993

def geometric_sequence (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ := a₁ * q^(n - 1)

def product_of_terms (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  (a₁^n) * (q^(n * (n - 1) / 2))

theorem max_product_at_12 (a₁ q : ℝ) (h₁ : a₁ = 1536) (h₂ : q = -1/2) :
  product_of_terms a₁ q 12 > product_of_terms a₁ q 9 ∧
  product_of_terms a₁ q 12 > product_of_terms a₁ q 13 := by
  sorry

end NUMINAMATH_CALUDE_max_product_at_12_l429_42993


namespace NUMINAMATH_CALUDE_olga_aquarium_fish_count_l429_42942

/-- The number of fish in Olga's aquarium -/
def fish_count (yellow blue green : ℕ) : ℕ := yellow + blue + green

/-- Theorem stating the total number of fish in Olga's aquarium -/
theorem olga_aquarium_fish_count :
  ∀ (yellow blue green : ℕ),
    yellow = 12 →
    blue = yellow / 2 →
    green = yellow * 2 →
    fish_count yellow blue green = 42 :=
by
  sorry

#check olga_aquarium_fish_count

end NUMINAMATH_CALUDE_olga_aquarium_fish_count_l429_42942


namespace NUMINAMATH_CALUDE_max_roses_for_680_l429_42992

/-- Represents the pricing options for roses -/
structure RosePrices where
  individual : ℝ  -- Price of an individual rose
  dozen : ℝ       -- Price of a dozen roses
  twoDozen : ℝ    -- Price of two dozen roses

/-- Calculates the maximum number of roses that can be purchased given a budget and pricing options -/
def maxRoses (budget : ℝ) (prices : RosePrices) : ℕ :=
  sorry

/-- Theorem stating that given specific pricing options and a budget of $680, the maximum number of roses that can be purchased is 325 -/
theorem max_roses_for_680 :
  let prices : RosePrices := { individual := 2.30, dozen := 36, twoDozen := 50 }
  maxRoses 680 prices = 325 := by
  sorry

end NUMINAMATH_CALUDE_max_roses_for_680_l429_42992


namespace NUMINAMATH_CALUDE_g_18_value_l429_42984

-- Define the properties of g
def is_valid_g (g : ℕ+ → ℕ+) : Prop :=
  (∀ n : ℕ+, g (n + 1) > g n) ∧ 
  (∀ m n : ℕ+, g (m * n) = g m * g n) ∧
  (∀ m n : ℕ+, m ≠ n → m ^ (n : ℕ) = n ^ (m : ℕ) → g m = n ^ 2 ∨ g n = m ^ 2)

-- State the theorem
theorem g_18_value (g : ℕ+ → ℕ+) (h : is_valid_g g) : g 18 = 104976 := by
  sorry

end NUMINAMATH_CALUDE_g_18_value_l429_42984


namespace NUMINAMATH_CALUDE_square_plus_one_geq_double_for_all_reals_l429_42955

theorem square_plus_one_geq_double_for_all_reals :
  ∀ a : ℝ, a^2 + 1 ≥ 2*a := by sorry

end NUMINAMATH_CALUDE_square_plus_one_geq_double_for_all_reals_l429_42955


namespace NUMINAMATH_CALUDE_square_region_perimeter_l429_42980

theorem square_region_perimeter (area : ℝ) (num_squares : ℕ) (rows : ℕ) (cols : ℕ) :
  area = 392 →
  num_squares = 8 →
  rows = 2 →
  cols = 4 →
  let side_length := Real.sqrt (area / num_squares)
  let perimeter := 2 * (rows * side_length + cols * side_length)
  perimeter = 126 := by
  sorry

end NUMINAMATH_CALUDE_square_region_perimeter_l429_42980


namespace NUMINAMATH_CALUDE_function_with_bounded_difference_is_constant_l429_42964

/-- A function f: ℝ → ℝ that satisfies |f(x) - f(y)| ≤ (x - y)² for all x, y ∈ ℝ is constant. -/
theorem function_with_bounded_difference_is_constant (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, |f x - f y| ≤ (x - y)^2) : 
  ∃ c : ℝ, ∀ x : ℝ, f x = c :=
by sorry

end NUMINAMATH_CALUDE_function_with_bounded_difference_is_constant_l429_42964


namespace NUMINAMATH_CALUDE_log_function_k_range_l429_42970

theorem log_function_k_range (a : ℝ) (h_a : a > 0) :
  {k : ℝ | ∀ x > a, x > max a (k * a)} = {k : ℝ | -1 ≤ k ∧ k ≤ 1} := by
sorry

end NUMINAMATH_CALUDE_log_function_k_range_l429_42970


namespace NUMINAMATH_CALUDE_inequalities_with_distinct_positive_reals_l429_42951

theorem inequalities_with_distinct_positive_reals 
  (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) : 
  (a^4 + b^4 > a^3*b + a*b^3) ∧ (a^5 + b^5 > a^3*b^2 + a^2*b^3) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_with_distinct_positive_reals_l429_42951


namespace NUMINAMATH_CALUDE_brownies_left_l429_42931

/-- Calculates the number of brownies left after consumption by Tina, her husband, and dinner guests. -/
theorem brownies_left (total : ℝ) (tina_lunch : ℝ) (tina_dinner : ℝ) (husband : ℝ) (guests : ℝ) 
  (days : ℕ) (guest_days : ℕ) : 
  total = 24 ∧ 
  tina_lunch = 1.5 ∧ 
  tina_dinner = 0.5 ∧ 
  husband = 0.75 ∧ 
  guests = 2.5 ∧ 
  days = 5 ∧ 
  guest_days = 2 → 
  total - ((tina_lunch + tina_dinner) * days + husband * days + guests * guest_days) = 5.25 := by
  sorry

end NUMINAMATH_CALUDE_brownies_left_l429_42931


namespace NUMINAMATH_CALUDE_equation_solution_l429_42950

theorem equation_solution : 
  ∃! x : ℝ, 4 * (4 ^ x) + Real.sqrt (16 * (16 ^ x)) = 64 ∧ x = (3 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l429_42950


namespace NUMINAMATH_CALUDE_school_distance_l429_42990

theorem school_distance (speed_to_school : ℝ) (speed_from_school : ℝ) (total_time : ℝ) 
  (h1 : speed_to_school = 3)
  (h2 : speed_from_school = 2)
  (h3 : total_time = 5) :
  ∃ (distance : ℝ), distance = 6 ∧ 
    (distance / speed_to_school + distance / speed_from_school = total_time) :=
by
  sorry

end NUMINAMATH_CALUDE_school_distance_l429_42990


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a2_l429_42916

/-- An arithmetic sequence with a_1 = 1 and a_{n+2} - a_n = 3 has a_2 = 5/2 -/
theorem arithmetic_sequence_a2 (a : ℕ → ℚ) :
  a 1 = 1 →
  (∀ n : ℕ, a (n + 2) - a n = 3) →
  (∀ n : ℕ, ∃ d : ℚ, a (n + 1) = a n + d) →
  a 2 = 5/2 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a2_l429_42916


namespace NUMINAMATH_CALUDE_right_triangle_area_l429_42987

theorem right_triangle_area (a b c : ℝ) (h1 : a^2 = 64) (h2 : b^2 = 49) (h3 : c^2 = 113) 
  (h4 : a^2 + b^2 = c^2) : (1/2) * a * b = 28 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l429_42987


namespace NUMINAMATH_CALUDE_plot_area_calculation_l429_42982

/-- Represents the area of a rectangular plot of land in acres, given its dimensions in miles. -/
def plot_area (length width : ℝ) : ℝ :=
  length * width * 640

/-- Theorem stating that a rectangular plot of land with dimensions 20 miles by 30 miles has an area of 384000 acres. -/
theorem plot_area_calculation :
  plot_area 30 20 = 384000 := by
  sorry


end NUMINAMATH_CALUDE_plot_area_calculation_l429_42982


namespace NUMINAMATH_CALUDE_rectangle_triangles_l429_42940

/-- Represents a rectangle divided into triangles -/
structure DividedRectangle where
  horizontal_divisions : Nat
  vertical_divisions : Nat

/-- Counts the number of triangles in a divided rectangle -/
def count_triangles (rect : DividedRectangle) : Nat :=
  sorry

/-- Theorem: A rectangle divided into 6 horizontal and 3 vertical parts contains 48 triangles -/
theorem rectangle_triangles :
  let rect : DividedRectangle := { horizontal_divisions := 6, vertical_divisions := 3 }
  count_triangles rect = 48 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_triangles_l429_42940


namespace NUMINAMATH_CALUDE_max_pages_copied_l429_42983

/-- The cost in cents to copy one page -/
def cost_per_page : ℕ := 3

/-- The available amount in dollars -/
def available_dollars : ℕ := 25

/-- Convert dollars to cents -/
def dollars_to_cents (dollars : ℕ) : ℕ := dollars * 100

/-- Calculate the number of full pages that can be copied -/
def pages_copied (cents : ℕ) (cost : ℕ) : ℕ := cents / cost

theorem max_pages_copied : 
  pages_copied (dollars_to_cents available_dollars) cost_per_page = 833 := by
  sorry

end NUMINAMATH_CALUDE_max_pages_copied_l429_42983


namespace NUMINAMATH_CALUDE_solve_cost_problem_l429_42999

def cost_problem (shirt_cost jacket_cost carrie_payment : ℕ) 
                 (num_shirts num_pants num_jackets : ℕ) : Prop :=
  let total_cost := 2 * carrie_payment
  let pants_cost := (total_cost - num_shirts * shirt_cost - num_jackets * jacket_cost) / num_pants
  pants_cost = 18

theorem solve_cost_problem :
  cost_problem 8 60 94 4 2 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_cost_problem_l429_42999


namespace NUMINAMATH_CALUDE_hexagon_sixth_angle_l429_42923

/-- A hexagon with given angle measures -/
structure Hexagon where
  Q : ℝ
  R : ℝ
  S : ℝ
  T : ℝ
  U : ℝ
  V : ℝ
  sum_angles : Q + R + S + T + U + V = 720
  Q_value : Q = 110
  R_value : R = 135
  S_value : S = 140
  T_value : T = 95
  U_value : U = 100

/-- The sixth angle of a hexagon with five known angles measures 140° -/
theorem hexagon_sixth_angle (h : Hexagon) : h.V = 140 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_sixth_angle_l429_42923


namespace NUMINAMATH_CALUDE_product_of_powers_equals_hundred_l429_42913

theorem product_of_powers_equals_hundred : 
  (10 ^ 0.6) * (10 ^ 0.2) * (10 ^ 0.1) * (10 ^ 0.3) * (10 ^ 0.7) * (10 ^ 0.1) = 100 := by
sorry

end NUMINAMATH_CALUDE_product_of_powers_equals_hundred_l429_42913


namespace NUMINAMATH_CALUDE_smallest_composite_no_small_factors_l429_42976

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def has_no_prime_factors_less_than (n k : ℕ) : Prop :=
  ∀ p, p < k → Prime p → ¬(p ∣ n)

theorem smallest_composite_no_small_factors :
  (is_composite 221) ∧
  (has_no_prime_factors_less_than 221 12) ∧
  (∀ m : ℕ, m < 221 → ¬(is_composite m ∧ has_no_prime_factors_less_than m 12)) :=
sorry

end NUMINAMATH_CALUDE_smallest_composite_no_small_factors_l429_42976


namespace NUMINAMATH_CALUDE_proposition_count_l429_42937

theorem proposition_count : 
  (∃ (correct : Finset (Fin 6)) (h : correct.card = 5),
    (∀ i : Fin 6, i ∈ correct ↔
      (i = 0 ∧ ∀ a b : ℝ, b < a ∧ a < 0 → |a| > |b|) ∨
      (i = 1 ∧ ∀ a b : ℝ, b < a ∧ a < 0 → a + b < a * b) ∨
      (i = 2 ∧ ∀ a b : ℝ, b < a ∧ a < 0 → b / a + a / b > 2) ∨
      (i = 3 ∧ ∀ a b : ℝ, b < a ∧ a < 0 → a^2 / b < 2 * a - b) ∨
      (i = 4 ∧ ∀ a b : ℝ, b < a ∧ a < 0 → (2 * a + b) / (a + 2 * b) > a / b) ∨
      (i = 5 ∧ ∀ a b : ℝ, a + b = 1 → a^2 + b^2 ≥ 1 / 2))) :=
by sorry

end NUMINAMATH_CALUDE_proposition_count_l429_42937


namespace NUMINAMATH_CALUDE_michaels_initial_money_proof_l429_42908

/-- Michael's initial amount of money -/
def michaels_initial_money : ℕ := 152

/-- Amount Michael's brother had initially -/
def brothers_initial_money : ℕ := 17

/-- Amount spent on candy -/
def candy_cost : ℕ := 3

/-- Amount Michael's brother has left after buying candy -/
def brothers_remaining_money : ℕ := 35

theorem michaels_initial_money_proof :
  michaels_initial_money = 
    2 * (brothers_remaining_money + candy_cost + brothers_initial_money - brothers_initial_money) :=
by sorry

end NUMINAMATH_CALUDE_michaels_initial_money_proof_l429_42908


namespace NUMINAMATH_CALUDE_elective_subjects_theorem_l429_42989

def subjects := 6
def chosen := 3

def choose (n : ℕ) (r : ℕ) : ℕ := Nat.choose n r

theorem elective_subjects_theorem :
  -- Statement A
  (choose 5 3 = choose 5 2) ∧
  -- Statement C
  (choose subjects chosen - choose 4 1 = choose subjects chosen - choose (subjects - 2) (chosen - 2)) :=
sorry

end NUMINAMATH_CALUDE_elective_subjects_theorem_l429_42989


namespace NUMINAMATH_CALUDE_max_blocks_in_box_l429_42979

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents the dimensions of a block -/
structure BlockDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the maximum number of blocks that can fit in a box -/
def maxBlocks (box : BoxDimensions) (block : BlockDimensions) : ℕ :=
  sorry

/-- The theorem stating the maximum number of blocks that can fit in the given box -/
theorem max_blocks_in_box :
  let box := BoxDimensions.mk 4 3 2
  let block := BlockDimensions.mk 3 1 1
  maxBlocks box block = 6 :=
sorry

end NUMINAMATH_CALUDE_max_blocks_in_box_l429_42979


namespace NUMINAMATH_CALUDE_special_sequence_a10_l429_42915

/-- A sequence of positive real numbers satisfying aₚ₊ₖ = aₚ · aₖ for all positive integers p and q -/
def SpecialSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ 
  (∀ p q : ℕ, p > 0 → q > 0 → a (p + q) = a p * a q)

theorem special_sequence_a10 (a : ℕ → ℝ) (h : SpecialSequence a) (h8 : a 8 = 16) : 
  a 10 = 32 := by
  sorry

end NUMINAMATH_CALUDE_special_sequence_a10_l429_42915


namespace NUMINAMATH_CALUDE_jane_is_26_l429_42968

/-- Given Danny's current age and the age difference between Danny and Jane 19 years ago,
    calculates Jane's current age. -/
def janes_current_age (dannys_current_age : ℕ) (years_ago : ℕ) : ℕ :=
  let dannys_age_then := dannys_current_age - years_ago
  let janes_age_then := dannys_age_then / 3
  janes_age_then + years_ago

/-- Proves that Jane's current age is 26, given the problem conditions. -/
theorem jane_is_26 :
  janes_current_age 40 19 = 26 := by
  sorry


end NUMINAMATH_CALUDE_jane_is_26_l429_42968


namespace NUMINAMATH_CALUDE_train_length_l429_42918

/-- The length of a train that passes a stationary man in 8 seconds and crosses a 270-meter platform in 20 seconds is 180 meters. -/
theorem train_length : ℝ → Prop :=
  fun L : ℝ =>
    (L / 8 = (L + 270) / 20) →
    L = 180

/-- Proof of the train length theorem -/
lemma train_length_proof : ∃ L : ℝ, train_length L :=
by
  sorry

end NUMINAMATH_CALUDE_train_length_l429_42918


namespace NUMINAMATH_CALUDE_divisibility_of_quadratic_form_l429_42963

theorem divisibility_of_quadratic_form (n : ℕ) (h : 0 < n) :
  ∃ (a b : ℕ), 0 < a ∧ 0 < b ∧ (n ∣ 4 * a^2 + 9 * b^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_quadratic_form_l429_42963


namespace NUMINAMATH_CALUDE_square_root_of_121_l429_42956

theorem square_root_of_121 : ∀ x : ℝ, x^2 = 121 ↔ x = 11 ∨ x = -11 := by sorry

end NUMINAMATH_CALUDE_square_root_of_121_l429_42956


namespace NUMINAMATH_CALUDE_base_prime_repr_360_l429_42932

/-- Base prime representation of a natural number -/
def base_prime_repr (n : ℕ) : ℕ := sorry

/-- Prime factorization of a natural number -/
def prime_factorization (n : ℕ) : List (ℕ × ℕ) := sorry

theorem base_prime_repr_360 :
  base_prime_repr 360 = 321 := by sorry

end NUMINAMATH_CALUDE_base_prime_repr_360_l429_42932


namespace NUMINAMATH_CALUDE_butter_theorem_l429_42904

def butter_problem (total_butter : ℝ) (chocolate_chip : ℝ) (peanut_butter : ℝ) (sugar : ℝ) (oatmeal : ℝ) (spilled : ℝ) : Prop :=
  let used_butter := chocolate_chip * total_butter + peanut_butter * total_butter + sugar * total_butter + oatmeal * total_butter
  let remaining_before_spill := total_butter - used_butter
  let remaining_after_spill := remaining_before_spill - spilled
  remaining_after_spill = 0.375

theorem butter_theorem : 
  butter_problem 15 (2/5) (1/6) (1/8) (1/4) 0.5 := by
  sorry

end NUMINAMATH_CALUDE_butter_theorem_l429_42904


namespace NUMINAMATH_CALUDE_roots_sum_and_product_l429_42922

def absolute_value_equation (x : ℝ) : Prop :=
  |x|^3 + |x|^2 - 4*|x| - 12 = 0

theorem roots_sum_and_product :
  ∃ (roots : Finset ℝ), 
    (∀ x ∈ roots, absolute_value_equation x) ∧
    (roots.sum id = 0) ∧
    (roots.prod id = -4) :=
sorry

end NUMINAMATH_CALUDE_roots_sum_and_product_l429_42922


namespace NUMINAMATH_CALUDE_park_perimeter_l429_42924

/-- Given a square park with a road inside, proves that the perimeter is 600 meters -/
theorem park_perimeter (s : ℝ) : 
  s > 0 →  -- The side length is positive
  s^2 - (s - 6)^2 = 1764 →  -- The area of the road is 1764 sq meters
  4 * s = 600 :=  -- The perimeter is 600 meters
by
  sorry

end NUMINAMATH_CALUDE_park_perimeter_l429_42924


namespace NUMINAMATH_CALUDE_line_slope_through_points_l429_42998

/-- The slope of a line passing through points (1,3) and (5,7) is 1 -/
theorem line_slope_through_points : 
  let x1 : ℝ := 1
  let y1 : ℝ := 3
  let x2 : ℝ := 5
  let y2 : ℝ := 7
  let slope := (y2 - y1) / (x2 - x1)
  slope = 1 := by sorry

end NUMINAMATH_CALUDE_line_slope_through_points_l429_42998


namespace NUMINAMATH_CALUDE_midpoint_distance_theorem_l429_42997

theorem midpoint_distance_theorem (t : ℝ) : 
  let A : ℝ × ℝ := (t - 3, 0)
  let B : ℝ × ℝ := (-1, t + 2)
  let midpoint := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  ((midpoint.1 - A.1)^2 + (midpoint.2 - A.2)^2 = t^2 + 1) →
  (t = Real.sqrt 2 ∨ t = -Real.sqrt 2) := by
sorry

end NUMINAMATH_CALUDE_midpoint_distance_theorem_l429_42997


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l429_42907

def A : Set ℝ := {x | 3 ≤ x ∧ x ≤ 10}
def B : Set ℝ := {x | 2 < x ∧ x < 7}

theorem intersection_of_A_and_B :
  A ∩ B = {x | 3 ≤ x ∧ x < 7} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l429_42907


namespace NUMINAMATH_CALUDE_sum_of_cube_roots_bounded_l429_42944

theorem sum_of_cube_roots_bounded (a₁ a₂ a₃ a₄ : ℝ) 
  (h_pos : a₁ > 0 ∧ a₂ > 0 ∧ a₃ > 0 ∧ a₄ > 0) 
  (h_sum : a₁ + a₂ + a₃ + a₄ = 1) : 
  5 < (7 * a₁ + 1) ^ (1/3) + (7 * a₂ + 1) ^ (1/3) + 
      (7 * a₃ + 1) ^ (1/3) + (7 * a₄ + 1) ^ (1/3) ∧
      (7 * a₁ + 1) ^ (1/3) + (7 * a₂ + 1) ^ (1/3) + 
      (7 * a₃ + 1) ^ (1/3) + (7 * a₄ + 1) ^ (1/3) < 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cube_roots_bounded_l429_42944


namespace NUMINAMATH_CALUDE_fraction_evaluation_l429_42974

theorem fraction_evaluation (a b : ℚ) (ha : a = 5) (hb : b = -2) : 
  5 / (a + b) = 5 / 3 := by sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l429_42974


namespace NUMINAMATH_CALUDE_trig_expression_equals_sqrt_two_l429_42957

theorem trig_expression_equals_sqrt_two :
  (Real.cos (-585 * π / 180)) / (Real.tan (495 * π / 180) + Real.sin (-690 * π / 180)) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equals_sqrt_two_l429_42957


namespace NUMINAMATH_CALUDE_largeSum_congruence_l429_42930

/-- A function that calculates the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- The property that a number is congruent to the sum of its digits modulo 9 -/
axiom sum_of_digits_congruence (n : ℕ) : n ≡ sumOfDigits n [MOD 9]

/-- The sum we want to evaluate -/
def largeSum : ℕ := 2 + 55 + 444 + 3333 + 66666 + 777777 + 8888888 + 99999999

/-- Theorem stating that the large sum is congruent to 2 modulo 9 -/
theorem largeSum_congruence : largeSum ≡ 2 [MOD 9] := by sorry

end NUMINAMATH_CALUDE_largeSum_congruence_l429_42930


namespace NUMINAMATH_CALUDE_no_factors_l429_42900

def main_polynomial (z : ℂ) : ℂ := z^6 + 3*z^3 + 18

def option1 (z : ℂ) : ℂ := z^3 + 6
def option2 (z : ℂ) : ℂ := z - 2
def option3 (z : ℂ) : ℂ := z^3 - 6
def option4 (z : ℂ) : ℂ := z^3 - 3*z - 9

theorem no_factors :
  (∀ z, main_polynomial z ≠ 0 → option1 z ≠ 0) ∧
  (∀ z, main_polynomial z ≠ 0 → option2 z ≠ 0) ∧
  (∀ z, main_polynomial z ≠ 0 → option3 z ≠ 0) ∧
  (∀ z, main_polynomial z ≠ 0 → option4 z ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_no_factors_l429_42900


namespace NUMINAMATH_CALUDE_weight_of_barium_fluoride_l429_42948

/-- The atomic weight of barium in g/mol -/
def atomic_weight_Ba : ℝ := 137.33

/-- The atomic weight of fluorine in g/mol -/
def atomic_weight_F : ℝ := 19.00

/-- The number of moles of Barium fluoride -/
def moles_BaF2 : ℝ := 3

/-- The molecular weight of Barium fluoride (BaF2) in g/mol -/
def molecular_weight_BaF2 : ℝ := atomic_weight_Ba + 2 * atomic_weight_F

/-- The weight of Barium fluoride in grams -/
def weight_BaF2 : ℝ := moles_BaF2 * molecular_weight_BaF2

theorem weight_of_barium_fluoride : weight_BaF2 = 525.99 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_barium_fluoride_l429_42948


namespace NUMINAMATH_CALUDE_range_of_a_l429_42991

-- Define the sets P and Q
def P : Set ℝ := {x | x ≤ -3 ∨ x ≥ 0}
def Q (a : ℝ) : Set ℝ := {x | x ≤ a}

-- Define the conditions
def condition_not_p (x : ℝ) : Prop := -3 < x ∧ x < 0
def condition_not_q (x a : ℝ) : Prop := x > a

-- Define the relationship between q and p
def q_sufficient_not_necessary_for_p (a : ℝ) : Prop :=
  Q a ⊂ P ∧ Q a ≠ P

-- The main theorem
theorem range_of_a :
  ∀ a : ℝ, (∀ x : ℝ, condition_not_p x → condition_not_q x a) →
  q_sufficient_not_necessary_for_p a →
  a ≤ -3 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l429_42991


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l429_42949

theorem trigonometric_equation_solution (α : Real) 
  (h : 3 * Real.sin α + Real.cos α = 0) : 
  1 / (Real.cos α ^ 2 + 2 * Real.sin α * Real.cos α) = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l429_42949


namespace NUMINAMATH_CALUDE_same_solution_implies_zero_power_l429_42903

theorem same_solution_implies_zero_power (a b : ℝ) :
  (∃ x y : ℝ, 4*x + 3*y = 11 ∧ a*x + b*y = -2 ∧ 2*x - y = 3 ∧ b*x - a*y = 6) →
  (a + b)^2023 = 0 := by
sorry

end NUMINAMATH_CALUDE_same_solution_implies_zero_power_l429_42903


namespace NUMINAMATH_CALUDE_equation_solution_l429_42969

theorem equation_solution (x : ℝ) : 
  (x^3 - 3*x^2) / (x^2 - 4*x + 3) + 2*x = 0 → x = 0 ∨ x = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l429_42969


namespace NUMINAMATH_CALUDE_total_pay_is_550_l429_42961

/-- The total weekly pay for two employees, where one is paid 150% of the other -/
def total_weekly_pay (b_pay : ℚ) : ℚ :=
  b_pay + (150 / 100) * b_pay

/-- Theorem: Given B is paid 220 per week, the total pay for both employees is 550 -/
theorem total_pay_is_550 : total_weekly_pay 220 = 550 := by
  sorry

end NUMINAMATH_CALUDE_total_pay_is_550_l429_42961


namespace NUMINAMATH_CALUDE_ceiling_floor_difference_l429_42929

theorem ceiling_floor_difference : 
  ⌈(18 : ℚ) / 11 * (-33 : ℚ) / 4⌉ - ⌊(18 : ℚ) / 11 * ⌊(-33 : ℚ) / 4⌋⌋ = 2 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_difference_l429_42929


namespace NUMINAMATH_CALUDE_archery_competition_theorem_l429_42933

/-- Represents the point system for the archery competition -/
def PointSystem : Fin 4 → ℕ
  | 0 => 11  -- 1st place
  | 1 => 7   -- 2nd place
  | 2 => 5   -- 3rd place
  | 3 => 2   -- 4th place

/-- Represents the participation counts for each place -/
structure Participation where
  first : ℕ
  second : ℕ
  third : ℕ
  fourth : ℕ

/-- Calculates the product of points based on participation -/
def pointProduct (p : Participation) : ℕ :=
  (PointSystem 0) ^ p.first * 
  (PointSystem 1) ^ p.second * 
  (PointSystem 2) ^ p.third * 
  (PointSystem 3) ^ p.fourth

/-- Calculates the total number of participations -/
def totalParticipations (p : Participation) : ℕ :=
  p.first + p.second + p.third + p.fourth

/-- Theorem: If the product of points is 38500, then the total participations is 7 -/
theorem archery_competition_theorem (p : Participation) :
  pointProduct p = 38500 → totalParticipations p = 7 := by
  sorry


end NUMINAMATH_CALUDE_archery_competition_theorem_l429_42933


namespace NUMINAMATH_CALUDE_gcd_12345_67890_l429_42946

theorem gcd_12345_67890 : Nat.gcd 12345 67890 = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcd_12345_67890_l429_42946


namespace NUMINAMATH_CALUDE_remainder_of_1493824_div_4_l429_42927

theorem remainder_of_1493824_div_4 : 1493824 % 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_1493824_div_4_l429_42927


namespace NUMINAMATH_CALUDE_snow_difference_l429_42919

def mrs_hilt_snow : ℕ := 29
def brecknock_snow : ℕ := 17

theorem snow_difference : mrs_hilt_snow - brecknock_snow = 12 := by
  sorry

end NUMINAMATH_CALUDE_snow_difference_l429_42919


namespace NUMINAMATH_CALUDE_students_in_both_clubs_l429_42953

theorem students_in_both_clubs
  (total_students : ℕ)
  (drama_club : ℕ)
  (science_club : ℕ)
  (in_either_club : ℕ)
  (h1 : total_students = 300)
  (h2 : drama_club = 100)
  (h3 : science_club = 140)
  (h4 : in_either_club = 220) :
  drama_club + science_club - in_either_club = 20 :=
by sorry

end NUMINAMATH_CALUDE_students_in_both_clubs_l429_42953


namespace NUMINAMATH_CALUDE_dinner_bill_proof_l429_42902

theorem dinner_bill_proof (n : ℕ) (extra : ℝ) (total : ℝ) : 
  n = 10 →
  extra = 3 →
  (n - 1) * (total / n + extra) = total →
  total = 270 := by
sorry

end NUMINAMATH_CALUDE_dinner_bill_proof_l429_42902


namespace NUMINAMATH_CALUDE_tv_weekly_cost_l429_42910

/-- Calculate the weekly cost of running a TV -/
theorem tv_weekly_cost 
  (watt_per_hour : ℕ) 
  (hours_per_day : ℕ) 
  (cents_per_kwh : ℕ) 
  (h1 : watt_per_hour = 125)
  (h2 : hours_per_day = 4)
  (h3 : cents_per_kwh = 14) : 
  (watt_per_hour * hours_per_day * 7 * cents_per_kwh : ℚ) / 1000 = 49 := by
sorry

end NUMINAMATH_CALUDE_tv_weekly_cost_l429_42910


namespace NUMINAMATH_CALUDE_debate_team_boys_l429_42965

theorem debate_team_boys (girls : ℕ) (groups : ℕ) (group_size : ℕ) (boys : ℕ) : 
  girls = 46 → 
  groups = 8 → 
  group_size = 9 → 
  boys + girls = groups * group_size → 
  boys = 26 := by sorry

end NUMINAMATH_CALUDE_debate_team_boys_l429_42965


namespace NUMINAMATH_CALUDE_donut_distribution_l429_42967

/-- The number of ways to distribute n indistinguishable objects among k distinct boxes -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to choose r items from n items -/
def choose (n r : ℕ) : ℕ := sorry

theorem donut_distribution :
  let n : ℕ := 3  -- number of additional donuts to distribute
  let k : ℕ := 5  -- number of donut kinds
  distribute n k = choose (n + k - 1) n ∧
  choose (n + k - 1) n = 35 :=
by sorry

end NUMINAMATH_CALUDE_donut_distribution_l429_42967


namespace NUMINAMATH_CALUDE_equation_solution_l429_42912

theorem equation_solution : ∃! x : ℝ, (16 : ℝ)^(x - 1) / (8 : ℝ)^(x - 1) = (64 : ℝ)^(x + 2) ∧ x = -13/5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l429_42912


namespace NUMINAMATH_CALUDE_contrapositive_quadratic_inequality_l429_42906

theorem contrapositive_quadratic_inequality :
  (∀ x : ℝ, x^2 + x - 6 > 0 → x < -3 ∨ x > 2) ↔
  (∀ x : ℝ, x ≥ -3 ∧ x ≤ 2 → x^2 + x - 6 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_quadratic_inequality_l429_42906


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l429_42978

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1)
  third_term : a 3 = 2
  product_46 : a 4 * a 6 = 16

/-- The main theorem -/
theorem geometric_sequence_ratio 
  (seq : GeometricSequence) :
  (seq.a 9 - seq.a 11) / (seq.a 5 - seq.a 7) = 4 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_ratio_l429_42978


namespace NUMINAMATH_CALUDE_arc_length_calculation_l429_42945

theorem arc_length_calculation (circumference : ℝ) (central_angle : ℝ) 
  (h1 : circumference = 72) 
  (h2 : central_angle = 45) : 
  (central_angle / 360) * circumference = 9 := by
  sorry

end NUMINAMATH_CALUDE_arc_length_calculation_l429_42945


namespace NUMINAMATH_CALUDE_cubic_roots_sum_squares_l429_42939

theorem cubic_roots_sum_squares (p q r : ℝ) : 
  (p + q + r = 15) → (p * q + q * r + r * p = 25) → 
  (p^3 - 15*p^2 + 25*p - 10 = 0) → 
  (q^3 - 15*q^2 + 25*q - 10 = 0) → 
  (r^3 - 15*r^2 + 25*r - 10 = 0) → 
  (p + q)^2 + (q + r)^2 + (r + p)^2 = 350 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_squares_l429_42939


namespace NUMINAMATH_CALUDE_real_part_of_sum_l429_42926

theorem real_part_of_sum (z₁ z₂ : ℂ) (h₁ : z₁ = 4 + 19 * Complex.I) (h₂ : z₂ = 6 + 9 * Complex.I) :
  (z₁ + z₂).re = 10 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_sum_l429_42926


namespace NUMINAMATH_CALUDE_stamps_total_proof_l429_42928

/-- The number of stamps Lizette has -/
def lizette_stamps : ℕ := 813

/-- The number of stamps Lizette has more than Minerva -/
def lizette_minerva_diff : ℕ := 125

/-- The number of stamps Jermaine has more than Lizette -/
def jermaine_lizette_diff : ℕ := 217

/-- The total number of stamps Minerva, Lizette, and Jermaine have -/
def total_stamps : ℕ := lizette_stamps + (lizette_stamps - lizette_minerva_diff) + (lizette_stamps + jermaine_lizette_diff)

theorem stamps_total_proof : total_stamps = 2531 := by
  sorry

end NUMINAMATH_CALUDE_stamps_total_proof_l429_42928


namespace NUMINAMATH_CALUDE_babylon_sphere_properties_l429_42994

structure Sphere :=
  (holes : Nat)
  (angle_step : Real)

def ray_pairs (s : Sphere) : Nat :=
  (s.holes * (s.holes - 1)) / 2

def angle_between_rays (s : Sphere) (r1 r2 : Nat × Nat) : Real :=
  sorry  -- Function to calculate angle between two rays

def count_angle_pairs (s : Sphere) (angle : Real) : Nat :=
  sorry  -- Function to count pairs of rays forming a specific angle

def can_construct_polyhedron (s : Sphere) (polyhedron : String) : Prop :=
  sorry  -- Predicate to determine if a polyhedron can be constructed

theorem babylon_sphere_properties (s : Sphere) 
  (h1 : s.holes = 26) 
  (h2 : s.angle_step = 45) : 
  (count_angle_pairs s (45 : Real) = 40) ∧ 
  (count_angle_pairs s (60 : Real) = 48) ∧ 
  (can_construct_polyhedron s "tetrahedron") ∧ 
  (can_construct_polyhedron s "octahedron") ∧ 
  ¬(can_construct_polyhedron s "dual_tetrahedron") :=
by
  sorry

end NUMINAMATH_CALUDE_babylon_sphere_properties_l429_42994


namespace NUMINAMATH_CALUDE_circle_center_l429_42911

/-- Given a circle with polar equation ρ = 2cos(θ), its center in the Cartesian coordinate system is at (1,0) -/
theorem circle_center (ρ θ : ℝ) : ρ = 2 * Real.cos θ → ∃ (x y : ℝ), x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ ∧ (x - 1)^2 + y^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_l429_42911


namespace NUMINAMATH_CALUDE_unique_c_l429_42901

-- Define the quadratic function
def f (c : ℝ) (x : ℝ) : ℝ := -x^2 + c*x - 12

-- Define the condition for the inequality
def condition (c : ℝ) : Prop :=
  ∀ x : ℝ, f c x < 0 ↔ (x < 2 ∨ x > 7)

-- Theorem statement
theorem unique_c : ∃! c : ℝ, condition c :=
  sorry

end NUMINAMATH_CALUDE_unique_c_l429_42901


namespace NUMINAMATH_CALUDE_negative_cube_squared_l429_42934

theorem negative_cube_squared (a : ℝ) : (-a^3)^2 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_negative_cube_squared_l429_42934
