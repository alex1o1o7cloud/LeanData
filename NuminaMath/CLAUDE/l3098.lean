import Mathlib

namespace NUMINAMATH_CALUDE_cos_is_even_and_has_zero_point_l3098_309809

-- Define what it means for a function to be even
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

-- Define what it means for a function to have a zero point
def HasZeroPoint (f : ℝ → ℝ) : Prop :=
  ∃ x, f x = 0

theorem cos_is_even_and_has_zero_point :
  IsEven Real.cos ∧ HasZeroPoint Real.cos := by sorry

end NUMINAMATH_CALUDE_cos_is_even_and_has_zero_point_l3098_309809


namespace NUMINAMATH_CALUDE_sum_of_factors_l3098_309882

theorem sum_of_factors (a b c d e : ℤ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ 
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ 
  c ≠ d ∧ c ≠ e ∧ 
  d ≠ e → 
  (7 - a) * (7 - b) * (7 - c) * (7 - d) * (7 - e) = 60 →
  a + b + c + d + e = 24 := by
sorry

end NUMINAMATH_CALUDE_sum_of_factors_l3098_309882


namespace NUMINAMATH_CALUDE_unique_fixed_point_for_rotationally_invariant_function_l3098_309808

-- Define a function that remains unchanged when its graph is rotated by π/2
def RotationallyInvariant (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f x = y ↔ f (-y) = x

-- Theorem statement
theorem unique_fixed_point_for_rotationally_invariant_function
  (f : ℝ → ℝ) (h : RotationallyInvariant f) :
  ∃! x : ℝ, f x = x :=
by sorry

end NUMINAMATH_CALUDE_unique_fixed_point_for_rotationally_invariant_function_l3098_309808


namespace NUMINAMATH_CALUDE_larry_cards_l3098_309812

theorem larry_cards (initial_cards final_cards taken_cards : ℕ) : 
  final_cards = initial_cards - taken_cards → 
  taken_cards = 9 → 
  final_cards = 58 → 
  initial_cards = 67 := by
sorry

end NUMINAMATH_CALUDE_larry_cards_l3098_309812


namespace NUMINAMATH_CALUDE_worker_travel_time_l3098_309864

/-- If a worker walking at 5/6 of her normal speed arrives 12 minutes later than usual, 
    then her usual time to reach the office is 60 minutes. -/
theorem worker_travel_time (S : ℝ) (T : ℝ) (h1 : S > 0) (h2 : T > 0) : 
  S * T = (5/6 * S) * (T + 12) → T = 60 := by
  sorry

end NUMINAMATH_CALUDE_worker_travel_time_l3098_309864


namespace NUMINAMATH_CALUDE_elder_age_is_30_l3098_309848

/-- The age difference between two persons -/
def age_difference : ℕ := 16

/-- The number of years ago when the elder was 3 times as old as the younger -/
def years_ago : ℕ := 6

/-- The present age of the younger person -/
def younger_age : ℕ := 14

/-- The present age of the elder person -/
def elder_age : ℕ := younger_age + age_difference

theorem elder_age_is_30 :
  (elder_age - years_ago = 3 * (younger_age - years_ago)) →
  elder_age = 30 :=
by sorry

end NUMINAMATH_CALUDE_elder_age_is_30_l3098_309848


namespace NUMINAMATH_CALUDE_inequality_reversal_l3098_309899

theorem inequality_reversal (x y : ℝ) (h : x > y) : ¬(1 - x > 1 - y) := by
  sorry

end NUMINAMATH_CALUDE_inequality_reversal_l3098_309899


namespace NUMINAMATH_CALUDE_expression_evaluation_l3098_309872

theorem expression_evaluation (a b : ℤ) (h1 : a = 4) (h2 : b = -2) :
  2 * a^2 - 3 * b^2 + a * b = 12 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3098_309872


namespace NUMINAMATH_CALUDE_series_sum_l3098_309804

theorem series_sum : ∑' n, (3 * n - 1 : ℝ) / 2^n = 5 := by sorry

end NUMINAMATH_CALUDE_series_sum_l3098_309804


namespace NUMINAMATH_CALUDE_intersection_equals_N_implies_t_range_l3098_309811

-- Define the sets M and N
def M : Set ℝ := {x | -4 < x ∧ x < 3}
def N (t : ℝ) : Set ℝ := {x | t + 2 < x ∧ x < 2*t - 1}

-- State the theorem
theorem intersection_equals_N_implies_t_range (t : ℝ) : 
  M ∩ N t = N t → t ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_intersection_equals_N_implies_t_range_l3098_309811


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_6889_l3098_309854

theorem largest_prime_factor_of_6889 : ∃ p : ℕ, p.Prime ∧ p ∣ 6889 ∧ ∀ q : ℕ, q.Prime → q ∣ 6889 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_6889_l3098_309854


namespace NUMINAMATH_CALUDE_num_colorings_is_162_l3098_309834

/-- Represents the three colors available for coloring --/
inductive Color
| Red
| White
| Blue

/-- Represents a coloring of a single triangle --/
structure TriangleColoring :=
  (a b c : Color)
  (different_colors : a ≠ b ∧ b ≠ c ∧ a ≠ c)

/-- Represents a coloring of the entire figure (four triangles) --/
structure FigureColoring :=
  (t1 t2 t3 t4 : TriangleColoring)
  (connected_different : t1.c = t2.a ∧ t2.c = t3.a ∧ t3.c = t4.a)

/-- The number of valid colorings for the figure --/
def num_colorings : ℕ := sorry

/-- Theorem stating that the number of valid colorings is 162 --/
theorem num_colorings_is_162 : num_colorings = 162 := by sorry

end NUMINAMATH_CALUDE_num_colorings_is_162_l3098_309834


namespace NUMINAMATH_CALUDE_fundraiser_group_composition_l3098_309820

theorem fundraiser_group_composition (initial_total : ℕ) : 
  let initial_girls : ℕ := (initial_total * 3) / 10
  let final_total : ℕ := initial_total
  let final_girls : ℕ := initial_girls - 3
  (initial_girls : ℚ) / initial_total = 3 / 10 →
  (final_girls : ℚ) / final_total = 1 / 4 →
  initial_girls = 18 :=
by
  sorry

#check fundraiser_group_composition

end NUMINAMATH_CALUDE_fundraiser_group_composition_l3098_309820


namespace NUMINAMATH_CALUDE_triangle_theorem_l3098_309852

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem -/
theorem triangle_theorem (t : Triangle) (h : 4 * t.a^2 = t.b * t.c * Real.cos t.A + t.a * t.c * Real.cos t.B) :
  (t.a / t.c = 1 / 2) ∧
  (t.a = 1 → Real.cos t.B = 3 / 4 → ∃ D : ℝ × ℝ, 
    (D.1 = (t.a + t.c) / 2 ∧ D.2 = 0) → 
    Real.sqrt ((D.1 - t.a)^2 + D.2^2) = Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_theorem_l3098_309852


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3098_309859

/-- Given an arithmetic sequence with first term 3 and common difference 12,
    prove that the sum of the first 30 terms is 5310. -/
theorem arithmetic_sequence_sum : 
  let a : ℕ → ℤ := fun n => 3 + (n - 1) * 12
  let S : ℕ → ℤ := fun n => n * (a 1 + a n) / 2
  S 30 = 5310 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3098_309859


namespace NUMINAMATH_CALUDE_f_min_at_one_l3098_309867

/-- The quadratic function that we're analyzing -/
def f (x : ℝ) : ℝ := (x - 1)^2 - 3

/-- Theorem stating that f reaches its minimum value when x = 1 -/
theorem f_min_at_one : ∀ x : ℝ, f x ≥ f 1 := by
  sorry

end NUMINAMATH_CALUDE_f_min_at_one_l3098_309867


namespace NUMINAMATH_CALUDE_smallest_quadratic_coefficient_l3098_309822

theorem smallest_quadratic_coefficient (a : ℕ) : 
  (∃ (b c : ℤ) (x₁ x₂ : ℝ), 
    x₁ ≠ x₂ ∧ 
    0 < x₁ ∧ x₁ < 1 ∧ 
    0 < x₂ ∧ x₂ < 1 ∧ 
    (a : ℝ) * x₁^2 + (b : ℝ) * x₁ + (c : ℝ) = 0 ∧ 
    (a : ℝ) * x₂^2 + (b : ℝ) * x₂ + (c : ℝ) = 0) →
  a ≥ 5 :=
sorry

end NUMINAMATH_CALUDE_smallest_quadratic_coefficient_l3098_309822


namespace NUMINAMATH_CALUDE_stockholm_uppsala_distance_l3098_309816

/-- The actual distance between two cities given their distance on a map and the map's scale. -/
def actual_distance (map_distance : ℝ) (scale : ℝ) : ℝ :=
  map_distance * scale

/-- Theorem: The actual distance between Stockholm and Uppsala is 450 km. -/
theorem stockholm_uppsala_distance : 
  let map_distance : ℝ := 45
  let scale : ℝ := 10
  actual_distance map_distance scale = 450 :=
by sorry

end NUMINAMATH_CALUDE_stockholm_uppsala_distance_l3098_309816


namespace NUMINAMATH_CALUDE_opposite_of_three_l3098_309845

theorem opposite_of_three : (-(3 : ℝ)) = -3 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_three_l3098_309845


namespace NUMINAMATH_CALUDE_add_like_terms_l3098_309843

theorem add_like_terms (a : ℝ) : 3 * a + 2 * a = 5 * a := by
  sorry

end NUMINAMATH_CALUDE_add_like_terms_l3098_309843


namespace NUMINAMATH_CALUDE_expression_bounds_l3098_309807

theorem expression_bounds (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ∃ (m M : ℝ),
    (∀ x y : ℝ, x ≠ 0 → y ≠ 0 → m ≤ (3 * |x + y|) / (|x| + |y|) ∧ (3 * |x + y|) / (|x| + |y|) ≤ M) ∧
    m = 0 ∧ M = 3 :=
by sorry

end NUMINAMATH_CALUDE_expression_bounds_l3098_309807


namespace NUMINAMATH_CALUDE_correct_pricing_strategy_l3098_309887

/-- Represents the cost and pricing structure of items A and B -/
structure ItemPricing where
  cost_A : ℝ
  cost_B : ℝ
  initial_price_A : ℝ
  price_reduction_A : ℝ

/-- Represents the sales data for items A and B -/
structure SalesData where
  initial_sales_A : ℕ
  sales_increase_rate : ℝ
  revenue_B : ℝ

/-- Theorem stating the correct pricing and reduction strategy -/
theorem correct_pricing_strategy 
  (p : ItemPricing) 
  (s : SalesData) 
  (h1 : 5 * p.cost_A + 3 * p.cost_B = 450)
  (h2 : 10 * p.cost_A + 8 * p.cost_B = 1000)
  (h3 : p.initial_price_A = 80)
  (h4 : s.initial_sales_A = 100)
  (h5 : s.sales_increase_rate = 20)
  (h6 : s.initial_sales_A + s.sales_increase_rate * p.price_reduction_A > 200)
  (h7 : s.revenue_B = 7000)
  (h8 : (p.initial_price_A - p.price_reduction_A) * 
        (s.initial_sales_A + s.sales_increase_rate * p.price_reduction_A) + 
        s.revenue_B = 10000) :
  p.cost_A = 60 ∧ p.cost_B = 50 ∧ p.price_reduction_A = 10 := by
  sorry

end NUMINAMATH_CALUDE_correct_pricing_strategy_l3098_309887


namespace NUMINAMATH_CALUDE_unique_minimum_condition_l3098_309874

/-- The objective function z(x,y) = ax + 2y has its unique minimum at (1,0) for all real x and y
    if and only if a is in the open interval (-4, -2) -/
theorem unique_minimum_condition (a : ℝ) :
  (∀ x y : ℝ, (a * x + 2 * y) ≥ (a * 1 + 2 * 0) ∧ 
   (∀ x' y' : ℝ, (x', y') ≠ (1, 0) → (a * x' + 2 * y') > (a * 1 + 2 * 0)))
  ↔ 
  (-4 < a ∧ a < -2) :=
sorry

end NUMINAMATH_CALUDE_unique_minimum_condition_l3098_309874


namespace NUMINAMATH_CALUDE_max_distinct_sum_100_l3098_309835

/-- The sum of the first n positive integers -/
def triangular_sum (n : ℕ) : ℕ := n * (n + 1) / 2

/-- A function that checks if a number is the maximum number of distinct positive integers that sum to 100 -/
def is_max_distinct_sum (k : ℕ) : Prop :=
  triangular_sum k ≤ 100 ∧ 
  triangular_sum (k + 1) > 100

theorem max_distinct_sum_100 : is_max_distinct_sum 13 := by
  sorry

#check max_distinct_sum_100

end NUMINAMATH_CALUDE_max_distinct_sum_100_l3098_309835


namespace NUMINAMATH_CALUDE_power_multiplication_l3098_309896

theorem power_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l3098_309896


namespace NUMINAMATH_CALUDE_cubic_roots_same_abs_value_iff_l3098_309814

-- Define the polynomial type
def CubicPolynomial (a b c : ℝ) : ℝ → ℝ := fun x ↦ x^3 + a*x^2 + b*x + c

-- Define the property that all roots have the same absolute value
def AllRootsSameAbsValue (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ z : ℂ, f z.re = 0 → Complex.abs z = k

-- Theorem statement
theorem cubic_roots_same_abs_value_iff (a b c : ℝ) :
  AllRootsSameAbsValue (CubicPolynomial a b c) → (a = 0 ↔ b = 0) := by
  sorry

end NUMINAMATH_CALUDE_cubic_roots_same_abs_value_iff_l3098_309814


namespace NUMINAMATH_CALUDE_total_profit_is_3872_l3098_309823

/-- Represents the investment and duration for each person -/
structure Investment where
  amount : ℕ
  duration : ℕ

/-- Calculates the total profit given the investments and profit difference -/
def calculateTotalProfit (suresh rohan sudhir : Investment) (profitDifference : ℕ) : ℕ :=
  sorry

/-- Theorem stating that the total profit is 3872 given the problem conditions -/
theorem total_profit_is_3872 :
  let suresh : Investment := ⟨18000, 12⟩
  let rohan : Investment := ⟨12000, 9⟩
  let sudhir : Investment := ⟨9000, 8⟩
  let profitDifference : ℕ := 352
  calculateTotalProfit suresh rohan sudhir profitDifference = 3872 :=
by sorry

end NUMINAMATH_CALUDE_total_profit_is_3872_l3098_309823


namespace NUMINAMATH_CALUDE_min_questions_for_phone_number_l3098_309842

theorem min_questions_for_phone_number (n : ℕ) (h : n = 100000) :
  ∃ k : ℕ, k = 17 ∧ 2^k ≥ n ∧ ∀ m : ℕ, m < k → 2^m < n :=
by sorry

end NUMINAMATH_CALUDE_min_questions_for_phone_number_l3098_309842


namespace NUMINAMATH_CALUDE_abs_neg_abs_square_minus_one_eq_zero_l3098_309894

theorem abs_neg_abs_square_minus_one_eq_zero :
  |(-|(-1 + 2)|)^2 - 1| = 0 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_abs_square_minus_one_eq_zero_l3098_309894


namespace NUMINAMATH_CALUDE_max_steps_to_empty_l3098_309801

/-- A function that checks if a natural number has repeated digits -/
def has_repeated_digits (n : ℕ) : Bool :=
  sorry

/-- A function that represents one step of the process -/
def step (list : List ℕ) : List ℕ :=
  sorry

/-- The initial list of the first 1000 positive integers -/
def initial_list : List ℕ :=
  sorry

/-- The number of steps required to empty the list -/
def steps_to_empty (list : List ℕ) : ℕ :=
  sorry

theorem max_steps_to_empty : steps_to_empty initial_list = 11 :=
  sorry

end NUMINAMATH_CALUDE_max_steps_to_empty_l3098_309801


namespace NUMINAMATH_CALUDE_product_of_digits_7891_base7_is_zero_l3098_309826

/-- Converts a natural number from base 10 to base 7 -/
def toBase7 (n : ℕ) : List ℕ :=
  sorry

/-- Calculates the product of a list of natural numbers -/
def productOfList (l : List ℕ) : ℕ :=
  sorry

/-- Theorem: The product of the digits in the base 7 representation of 7891 is 0 -/
theorem product_of_digits_7891_base7_is_zero :
  productOfList (toBase7 7891) = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_of_digits_7891_base7_is_zero_l3098_309826


namespace NUMINAMATH_CALUDE_g_value_at_3056_l3098_309870

theorem g_value_at_3056 (g : ℝ → ℝ) 
  (h1 : ∀ x > 0, g x > 0)
  (h2 : ∀ x y, x > y ∧ y > 0 → g (x - y) = Real.sqrt (g (x * y) + 4))
  (h3 : ∃ x y, x > y ∧ y > 0 ∧ x - y = x * y ∧ x * y = 3056) :
  g 3056 = 2 := by
sorry

end NUMINAMATH_CALUDE_g_value_at_3056_l3098_309870


namespace NUMINAMATH_CALUDE_tree_planting_seedlings_l3098_309840

theorem tree_planting_seedlings : 
  ∃ (x : ℕ), 
    (∃ (n : ℕ), x - 6 = 5 * n) ∧ 
    (∃ (m : ℕ), x + 9 = 6 * m) ∧ 
    x = 81 := by
  sorry

end NUMINAMATH_CALUDE_tree_planting_seedlings_l3098_309840


namespace NUMINAMATH_CALUDE_marie_bike_distance_l3098_309893

/-- The distance traveled given speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Proof that Marie biked 31 miles -/
theorem marie_bike_distance :
  let speed := 12.0
  let time := 2.583333333
  distance speed time = 31 := by
sorry

end NUMINAMATH_CALUDE_marie_bike_distance_l3098_309893


namespace NUMINAMATH_CALUDE_prime_rational_sum_l3098_309883

theorem prime_rational_sum (p : ℕ) (hp : Prime p) :
  (∃ (x y : ℚ) (n : ℕ), x > 0 ∧ y > 0 ∧ x + y + p / x + p / y = 3 * n) ↔ 3 ∣ (p + 1) :=
sorry

end NUMINAMATH_CALUDE_prime_rational_sum_l3098_309883


namespace NUMINAMATH_CALUDE_binary_ternary_equality_l3098_309829

theorem binary_ternary_equality (a b : ℕ) : 
  a ∈ ({0, 1, 2} : Set ℕ) → 
  b ∈ ({0, 1} : Set ℕ) → 
  (8 + 2 * b + 1 = 9 * a + 2) → 
  (a = 1 ∧ b = 1) :=
by sorry

end NUMINAMATH_CALUDE_binary_ternary_equality_l3098_309829


namespace NUMINAMATH_CALUDE_geometric_series_sum_l3098_309824

/-- The sum of a geometric series with 6 terms, first term a, and common ratio r -/
def geometric_sum (a : ℚ) (r : ℚ) : ℚ :=
  a * (1 - r^6) / (1 - r)

/-- The problem statement -/
theorem geometric_series_sum :
  let a : ℚ := 1/5
  let r : ℚ := -1/2
  geometric_sum a r = 21/160 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l3098_309824


namespace NUMINAMATH_CALUDE_functional_equation_solution_l3098_309819

-- Define the function type
def FunctionType := ℝ → ℝ

-- State the theorem
theorem functional_equation_solution (f : FunctionType) 
  (h1 : ∀ a b : ℝ, f (a + b) + f (a - b) = 3 * f a + f b) 
  (h2 : f 1 = 1) : 
  ∀ x : ℝ, f x = if x = 1 then 1 else 0 := by
  sorry


end NUMINAMATH_CALUDE_functional_equation_solution_l3098_309819


namespace NUMINAMATH_CALUDE_probability_not_purple_l3098_309869

/-- Given a bag of marbles where the odds of pulling a purple marble are 5:6,
    prove that the probability of not pulling a purple marble is 6/11. -/
theorem probability_not_purple (total : ℕ) (purple : ℕ) (not_purple : ℕ) :
  total = purple + not_purple →
  purple = 5 →
  not_purple = 6 →
  (not_purple : ℚ) / total = 6 / 11 :=
by sorry

end NUMINAMATH_CALUDE_probability_not_purple_l3098_309869


namespace NUMINAMATH_CALUDE_investment_loss_l3098_309827

/-- Given two investors with capitals in ratio 1:9 and proportional loss distribution,
    if one investor's loss is 603, then the total loss is 670. -/
theorem investment_loss (capital_ratio : ℚ) (investor1_loss : ℚ) (total_loss : ℚ) :
  capital_ratio = 1 / 9 →
  investor1_loss = 603 →
  total_loss = investor1_loss / (capital_ratio / (capital_ratio + 1)) →
  total_loss = 670 := by
  sorry

end NUMINAMATH_CALUDE_investment_loss_l3098_309827


namespace NUMINAMATH_CALUDE_lcm_problem_l3098_309803

theorem lcm_problem (a b : ℕ+) (h : Nat.gcd a b = 9) (p : a * b = 1800) :
  Nat.lcm a b = 200 := by
  sorry

end NUMINAMATH_CALUDE_lcm_problem_l3098_309803


namespace NUMINAMATH_CALUDE_min_value_expression_l3098_309800

theorem min_value_expression (x y z : ℝ) 
  (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) (hz : 0 ≤ z ∧ z ≤ 1) :
  (1 / ((1 - x) * (1 - y) * (1 - z)) + 
   1 / ((1 + x) * (1 + y) * (1 + z)) + 
   1 / ((1 - x^2) * (1 - y^2) * (1 - z^2))) ≥ 3 ∧
  (1 / ((1 - 0) * (1 - 0) * (1 - 0)) + 
   1 / ((1 + 0) * (1 + 0) * (1 + 0)) + 
   1 / ((1 - 0^2) * (1 - 0^2) * (1 - 0^2))) = 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3098_309800


namespace NUMINAMATH_CALUDE_net_loss_calculation_l3098_309849

def initial_value : ℝ := 12000
def loss_percentage : ℝ := 0.1
def gain_percentage : ℝ := 0.15

def first_sale_price : ℝ := initial_value * (1 - loss_percentage)
def second_sale_price : ℝ := first_sale_price * (1 + gain_percentage)

theorem net_loss_calculation :
  second_sale_price - initial_value = 420 := by sorry

end NUMINAMATH_CALUDE_net_loss_calculation_l3098_309849


namespace NUMINAMATH_CALUDE_average_age_decrease_l3098_309895

theorem average_age_decrease (original_average : ℝ) (new_students : ℕ) (new_average : ℝ) (original_strength : ℕ) :
  original_average = 40 →
  new_students = 15 →
  new_average = 32 →
  original_strength = 15 →
  let total_students := original_strength + new_students
  let new_total_age := original_average * original_strength + new_average * new_students
  let final_average := new_total_age / total_students
  40 - final_average = 4 :=
by sorry

end NUMINAMATH_CALUDE_average_age_decrease_l3098_309895


namespace NUMINAMATH_CALUDE_det_related_matrix_l3098_309873

/-- Given a 2x2 matrix with determinant 4, prove that the determinant of a related matrix is 12 -/
theorem det_related_matrix (a b c d : ℝ) (h : a * d - b * c = 4) :
  a * (7 * c + 3 * d) - c * (7 * a + 3 * b) = 12 := by
  sorry


end NUMINAMATH_CALUDE_det_related_matrix_l3098_309873


namespace NUMINAMATH_CALUDE_fraction_sum_equals_decimal_l3098_309897

theorem fraction_sum_equals_decimal : (3 / 50) + (5 / 500) + (7 / 5000) = 0.0714 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_decimal_l3098_309897


namespace NUMINAMATH_CALUDE_rectangle_p_value_l3098_309856

/-- Rectangle PQRS with given vertices and area -/
structure Rectangle where
  P : ℝ × ℝ
  S : ℝ × ℝ
  Q : ℝ × ℝ
  area : ℝ

/-- The theorem stating that if a rectangle PQRS has the given properties, then p = 15 -/
theorem rectangle_p_value (rect : Rectangle)
  (h1 : rect.P = (2, 3))
  (h2 : rect.S = (12, 3))
  (h3 : rect.Q.2 = 15)
  (h4 : rect.area = 120) :
  rect.Q.1 = 15 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_p_value_l3098_309856


namespace NUMINAMATH_CALUDE_expression_simplification_l3098_309818

theorem expression_simplification (a : ℝ) : 
  a^3 * a^5 + (a^2)^4 + (-2*a^4)^2 - 10*a^10 / (5*a^2) = 4*a^8 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3098_309818


namespace NUMINAMATH_CALUDE_stone_division_impossibility_l3098_309892

theorem stone_division_impossibility :
  ¬ ∃ (n : ℕ), n > 0 ∧ 3 * n = 1001 - (n - 1) :=
by
  sorry

end NUMINAMATH_CALUDE_stone_division_impossibility_l3098_309892


namespace NUMINAMATH_CALUDE_eleven_by_eleven_grid_segment_length_l3098_309857

/-- Represents a grid of lattice points -/
structure LatticeGrid where
  rows : ℕ
  columns : ℕ

/-- Calculates the total length of segments in a lattice grid -/
def totalSegmentLength (grid : LatticeGrid) : ℕ :=
  (grid.rows - 1) * grid.columns + (grid.columns - 1) * grid.rows

/-- Theorem: The total length of segments in an 11x11 lattice grid is 220 -/
theorem eleven_by_eleven_grid_segment_length :
  totalSegmentLength ⟨11, 11⟩ = 220 := by
  sorry

#eval totalSegmentLength ⟨11, 11⟩

end NUMINAMATH_CALUDE_eleven_by_eleven_grid_segment_length_l3098_309857


namespace NUMINAMATH_CALUDE_teachers_distribution_arrangements_l3098_309841

/-- The number of ways to distribute teachers between two classes -/
def distribute_teachers (total_teachers : ℕ) (max_per_class : ℕ) : ℕ :=
  let equal_distribution := 1
  let unequal_distribution := 2 * (Nat.choose total_teachers max_per_class)
  equal_distribution + unequal_distribution

/-- Theorem stating that distributing 6 teachers with a maximum of 4 per class results in 31 arrangements -/
theorem teachers_distribution_arrangements :
  distribute_teachers 6 4 = 31 := by
  sorry

end NUMINAMATH_CALUDE_teachers_distribution_arrangements_l3098_309841


namespace NUMINAMATH_CALUDE_handshake_theorem_l3098_309831

theorem handshake_theorem (n : ℕ) (total_handshakes : ℕ) :
  n = 10 →
  total_handshakes = 45 →
  total_handshakes = n * (n - 1) / 2 →
  ∀ boy : Fin n, (n - 1 : ℕ) = total_handshakes / n :=
by sorry

end NUMINAMATH_CALUDE_handshake_theorem_l3098_309831


namespace NUMINAMATH_CALUDE_total_players_on_ground_l3098_309875

def cricket_players : ℕ := 35
def hockey_players : ℕ := 28
def football_players : ℕ := 42
def softball_players : ℕ := 25
def basketball_players : ℕ := 18
def volleyball_players : ℕ := 30

theorem total_players_on_ground : 
  cricket_players + hockey_players + football_players + 
  softball_players + basketball_players + volleyball_players = 178 := by
  sorry

end NUMINAMATH_CALUDE_total_players_on_ground_l3098_309875


namespace NUMINAMATH_CALUDE_xy_value_l3098_309890

theorem xy_value (x y : ℝ) (h : |x - 1| + (y + 2)^2 = 0) : x * y = -2 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l3098_309890


namespace NUMINAMATH_CALUDE_min_separating_edges_l3098_309884

/-- Represents a color in the grid -/
inductive Color
| Red
| Green
| Blue

/-- Represents a cell in the grid -/
structure Cell :=
  (row : Fin 33)
  (col : Fin 33)
  (color : Color)

/-- Represents the grid -/
def Grid := Array (Array Cell)

/-- Checks if two cells are adjacent -/
def isAdjacent (c1 c2 : Cell) : Bool :=
  (c1.row = c2.row ∧ (c1.col.val + 1 = c2.col.val ∨ c1.col.val = c2.col.val + 1)) ∨
  (c1.col = c2.col ∧ (c1.row.val + 1 = c2.row.val ∨ c1.row.val = c2.row.val + 1))

/-- Counts the number of separating edges in the grid -/
def countSeparatingEdges (grid : Grid) : Nat :=
  sorry

/-- Checks if the grid has an equal number of cells for each color -/
def hasEqualColorDistribution (grid : Grid) : Prop :=
  sorry

/-- Theorem: The minimum number of separating edges in a 33x33 grid with three equally distributed colors is 56 -/
theorem min_separating_edges (grid : Grid) 
  (h : hasEqualColorDistribution grid) : 
  countSeparatingEdges grid ≥ 56 := by
  sorry

end NUMINAMATH_CALUDE_min_separating_edges_l3098_309884


namespace NUMINAMATH_CALUDE_chameleon_color_change_l3098_309853

/-- The number of chameleons that changed color in the grove --/
def chameleons_changed : ℕ := 80

/-- The total number of chameleons in the grove --/
def total_chameleons : ℕ := 140

/-- The number of blue chameleons after the color change --/
def blue_after : ℕ → ℕ
| n => n

/-- The number of blue chameleons before the color change --/
def blue_before : ℕ → ℕ
| n => 5 * n

/-- The number of red chameleons before the color change --/
def red_before : ℕ → ℕ
| n => total_chameleons - blue_before n

/-- The number of red chameleons after the color change --/
def red_after : ℕ → ℕ
| n => 3 * (red_before n)

theorem chameleon_color_change :
  ∃ n : ℕ, 
    blue_after n + red_after n = total_chameleons ∧ 
    chameleons_changed = blue_before n - blue_after n :=
  sorry

end NUMINAMATH_CALUDE_chameleon_color_change_l3098_309853


namespace NUMINAMATH_CALUDE_exists_set_without_triangle_l3098_309813

/-- A set of 10 segment lengths --/
def SegmentSet : Type := Fin 10 → ℝ

/-- Predicate to check if three segments can form a triangle --/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

/-- Theorem stating that there exists a set of 10 segments where no three can form a triangle --/
theorem exists_set_without_triangle : 
  ∃ (s : SegmentSet), ∀ (i j k : Fin 10), i ≠ j → j ≠ k → i ≠ k → 
    ¬(can_form_triangle (s i) (s j) (s k)) := by
  sorry

end NUMINAMATH_CALUDE_exists_set_without_triangle_l3098_309813


namespace NUMINAMATH_CALUDE_translation_company_min_employees_l3098_309866

/-- The number of languages offered by the company -/
def num_languages : ℕ := 4

/-- The number of languages each employee must learn -/
def languages_per_employee : ℕ := 2

/-- The minimum number of employees with identical training -/
def min_identical_training : ℕ := 5

/-- The number of possible language combinations -/
def num_combinations : ℕ := Nat.choose num_languages languages_per_employee

/-- The minimum number of employees in the company -/
def min_employees : ℕ := 25

theorem translation_company_min_employees :
  ∀ n : ℕ, n ≥ min_employees →
    ∃ (group : Finset (Finset (Fin num_languages))),
      (∀ e ∈ group, Finset.card e = languages_per_employee) ∧
      (Finset.card group ≥ min_identical_training) :=
by sorry

end NUMINAMATH_CALUDE_translation_company_min_employees_l3098_309866


namespace NUMINAMATH_CALUDE_fixed_point_of_function_l3098_309881

theorem fixed_point_of_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a * (x + 1) + 2
  f (-1) = 3 := by
sorry

end NUMINAMATH_CALUDE_fixed_point_of_function_l3098_309881


namespace NUMINAMATH_CALUDE_simplify_expression_l3098_309878

theorem simplify_expression (a b : ℝ) : a * b^2 * (-2 * a^3 * b) = -2 * a^4 * b^3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3098_309878


namespace NUMINAMATH_CALUDE_olivia_chocolate_sales_l3098_309833

def chocolate_problem (cost_per_bar total_bars unsold_bars : ℕ) : Prop :=
  let sold_bars := total_bars - unsold_bars
  let money_made := sold_bars * cost_per_bar
  money_made = 9

theorem olivia_chocolate_sales : chocolate_problem 3 7 4 := by
  sorry

end NUMINAMATH_CALUDE_olivia_chocolate_sales_l3098_309833


namespace NUMINAMATH_CALUDE_quiz_max_correct_answers_l3098_309880

theorem quiz_max_correct_answers :
  ∀ (correct blank incorrect : ℕ),
    correct + blank + incorrect = 60 →
    5 * correct - 2 * incorrect = 150 →
    correct ≤ 38 ∧
    ∃ (c b i : ℕ), c + b + i = 60 ∧ 5 * c - 2 * i = 150 ∧ c = 38 := by
  sorry

end NUMINAMATH_CALUDE_quiz_max_correct_answers_l3098_309880


namespace NUMINAMATH_CALUDE_kendra_hat_purchase_l3098_309886

theorem kendra_hat_purchase (toy_price hat_price initial_money change toys_bought : ℕ) 
  (h1 : toy_price = 20)
  (h2 : hat_price = 10)
  (h3 : initial_money = 100)
  (h4 : toys_bought = 2)
  (h5 : change = 30) :
  (initial_money - change - toy_price * toys_bought) / hat_price = 3 := by
  sorry

end NUMINAMATH_CALUDE_kendra_hat_purchase_l3098_309886


namespace NUMINAMATH_CALUDE_custom_mul_result_l3098_309865

/-- Custom multiplication operation for rational numbers -/
noncomputable def custom_mul (a b : ℚ) (x y : ℚ) : ℚ := a * x + b * y

theorem custom_mul_result 
  (a b : ℚ) 
  (h1 : custom_mul a b 1 2 = 1) 
  (h2 : custom_mul a b (-3) 3 = 6) :
  custom_mul a b 2 (-5) = -7 := by
  sorry

end NUMINAMATH_CALUDE_custom_mul_result_l3098_309865


namespace NUMINAMATH_CALUDE_probability_divisible_by_10_and_5_l3098_309851

def TwoDigitNumber (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def DivisibleBy10 (n : ℕ) : Prop := n % 10 = 0

def DivisibleBy5 (n : ℕ) : Prop := n % 5 = 0

def CountTwoDigitNumbers : ℕ := 90

def CountTwoDigitDivisibleBy10 : ℕ := 9

theorem probability_divisible_by_10_and_5 :
  (CountTwoDigitDivisibleBy10 : ℚ) / CountTwoDigitNumbers = 1 / 10 := by sorry

end NUMINAMATH_CALUDE_probability_divisible_by_10_and_5_l3098_309851


namespace NUMINAMATH_CALUDE_y_divisibility_l3098_309858

def y : ℕ := 48 + 72 + 144 + 216 + 432 + 648 + 2592

theorem y_divisibility :
  (∃ k : ℕ, y = 3 * k) ∧
  (∃ k : ℕ, y = 6 * k) ∧
  ¬(∀ k : ℕ, y = 9 * k) ∧
  ¬(∃ k : ℕ, y = 18 * k) :=
by sorry

end NUMINAMATH_CALUDE_y_divisibility_l3098_309858


namespace NUMINAMATH_CALUDE_multiplication_formula_examples_l3098_309862

theorem multiplication_formula_examples :
  (203 * 197 = 39991) ∧ ((-69.9)^2 = 4886.01) := by sorry

end NUMINAMATH_CALUDE_multiplication_formula_examples_l3098_309862


namespace NUMINAMATH_CALUDE_inequality_proof_l3098_309885

theorem inequality_proof (a b c d : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > d) :
  (1 / (a - b)) + (4 / (b - c)) + (9 / (c - d)) ≥ 36 / (a - d) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3098_309885


namespace NUMINAMATH_CALUDE_cost_of_dozen_pens_l3098_309877

theorem cost_of_dozen_pens (pen_cost pencil_cost : ℚ) : 
  (3 * pen_cost + 5 * pencil_cost = 150) →
  (pen_cost = 5 * pencil_cost) →
  (12 * pen_cost = 450) :=
by
  sorry

end NUMINAMATH_CALUDE_cost_of_dozen_pens_l3098_309877


namespace NUMINAMATH_CALUDE_hyperbola_dot_product_range_l3098_309815

/-- The hyperbola with center at origin and left focus at (-2,0) -/
structure Hyperbola where
  a : ℝ
  h_pos : a > 0

/-- A point on the right branch of the hyperbola -/
structure HyperbolaPoint (h : Hyperbola) where
  x : ℝ
  y : ℝ
  h_on_hyperbola : x^2 / h.a^2 - y^2 = 1
  h_right_branch : x ≥ h.a

/-- The theorem stating the range of the dot product -/
theorem hyperbola_dot_product_range (h : Hyperbola) (p : HyperbolaPoint h) :
  p.x * (p.x + 2) + p.y * p.y ≥ 3 + 2 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_hyperbola_dot_product_range_l3098_309815


namespace NUMINAMATH_CALUDE_ball_placement_count_is_42_l3098_309876

/-- The number of ways to place four distinct balls into three labeled boxes
    such that exactly one box remains empty. -/
def ballPlacementCount : ℕ := 42

/-- Theorem stating that the number of ways to place four distinct balls
    into three labeled boxes such that exactly one box remains empty is 42. -/
theorem ball_placement_count_is_42 : ballPlacementCount = 42 := by
  sorry

end NUMINAMATH_CALUDE_ball_placement_count_is_42_l3098_309876


namespace NUMINAMATH_CALUDE_linear_equation_transformation_l3098_309838

theorem linear_equation_transformation (x y : ℝ) :
  (3 * x + 4 * y = 5) ↔ (x = (5 - 4 * y) / 3) :=
by sorry

end NUMINAMATH_CALUDE_linear_equation_transformation_l3098_309838


namespace NUMINAMATH_CALUDE_crayons_in_box_l3098_309846

def crayons_problem (given_away lost : ℕ) (difference : ℤ) : Prop :=
  given_away = 90 ∧
  lost = 412 ∧
  difference = lost - given_away ∧
  difference = 322

theorem crayons_in_box (given_away lost : ℕ) (difference : ℤ) 
  (h : crayons_problem given_away lost difference) : 
  given_away + lost = 502 := by
  sorry

end NUMINAMATH_CALUDE_crayons_in_box_l3098_309846


namespace NUMINAMATH_CALUDE_class_ratio_problem_l3098_309855

theorem class_ratio_problem (total : ℕ) (boys : ℕ) (h_total : total > 0) (h_boys : boys ≤ total) :
  let p_boy := boys / total
  let p_girl := (total - boys) / total
  (p_boy = (2 : ℚ) / 3 * p_girl) → (boys : ℚ) / total = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_class_ratio_problem_l3098_309855


namespace NUMINAMATH_CALUDE_probability_three_even_dice_l3098_309879

def num_dice : ℕ := 6
def sides_per_die : ℕ := 12

theorem probability_three_even_dice :
  let p := (num_dice.choose 3) * (1 / 2) ^ num_dice / 1
  p = 5 / 16 := by sorry

end NUMINAMATH_CALUDE_probability_three_even_dice_l3098_309879


namespace NUMINAMATH_CALUDE_roof_ratio_l3098_309828

/-- Proves that a rectangular roof with given area and length-width difference has a specific length-to-width ratio -/
theorem roof_ratio (length width : ℝ) 
  (area_eq : length * width = 675)
  (diff_eq : length - width = 30) :
  length / width = 3 := by
sorry

end NUMINAMATH_CALUDE_roof_ratio_l3098_309828


namespace NUMINAMATH_CALUDE_number_equation_proof_l3098_309821

theorem number_equation_proof (x : ℤ) : 
  x - (28 - (37 - (15 - 15))) = 54 → x = 45 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_proof_l3098_309821


namespace NUMINAMATH_CALUDE_max_bw_edges_grid_l3098_309844

/-- Represents a square grid with corners removed and colored squares. -/
structure ColoredGrid :=
  (size : ℕ)
  (corner_size : ℕ)
  (coloring : ℕ → ℕ → Bool)

/-- Checks if a 2x2 square forms a checkerboard pattern. -/
def is_checkerboard (g : ColoredGrid) (x y : ℕ) : Prop :=
  g.coloring x y ≠ g.coloring (x+1) y ∧
  g.coloring x y ≠ g.coloring x (y+1) ∧
  g.coloring x y = g.coloring (x+1) (y+1)

/-- Counts the number of black-white edges in the grid. -/
def count_bw_edges (g : ColoredGrid) : ℕ := sorry

/-- The main theorem statement. -/
theorem max_bw_edges_grid (g : ColoredGrid) :
  g.size = 300 →
  g.corner_size = 100 →
  (∀ x y, x < g.size - g.corner_size ∧ y < g.size - g.corner_size →
    ¬is_checkerboard g x y) →
  count_bw_edges g ≤ 49998 :=
sorry

end NUMINAMATH_CALUDE_max_bw_edges_grid_l3098_309844


namespace NUMINAMATH_CALUDE_area_of_rectangle_l3098_309805

/-- The area of rectangle ABCD given the described configuration of squares and triangle -/
theorem area_of_rectangle (
  shaded_square_area : ℝ) 
  (h1 : shaded_square_area = 4) 
  (h2 : ∃ (side : ℝ), side^2 = shaded_square_area) 
  (h3 : ∃ (triangle_height : ℝ), triangle_height = Real.sqrt shaded_square_area) : 
  shaded_square_area + shaded_square_area + (2 * Real.sqrt shaded_square_area * Real.sqrt shaded_square_area / 2) = 12 := by
  sorry

end NUMINAMATH_CALUDE_area_of_rectangle_l3098_309805


namespace NUMINAMATH_CALUDE_company_match_percentage_l3098_309810

/-- Proves that the company's 401K match percentage is 6% given the problem conditions --/
theorem company_match_percentage (
  paychecks_per_year : ℕ)
  (contribution_per_paycheck : ℚ)
  (total_contribution : ℚ)
  (h1 : paychecks_per_year = 26)
  (h2 : contribution_per_paycheck = 100)
  (h3 : total_contribution = 2756) :
  (total_contribution - (paychecks_per_year : ℚ) * contribution_per_paycheck) /
  ((paychecks_per_year : ℚ) * contribution_per_paycheck) * 100 = 6 :=
by sorry

end NUMINAMATH_CALUDE_company_match_percentage_l3098_309810


namespace NUMINAMATH_CALUDE_existence_implies_bound_l3098_309871

theorem existence_implies_bound :
  (∃ (m : ℝ), ∃ (x : ℝ), 4^x + m * 2^x + 1 = 0) →
  (∀ (m : ℝ), (∃ (x : ℝ), 4^x + m * 2^x + 1 = 0) → m ≤ -2) :=
by sorry

end NUMINAMATH_CALUDE_existence_implies_bound_l3098_309871


namespace NUMINAMATH_CALUDE_largest_x_floor_div_l3098_309861

theorem largest_x_floor_div (x : ℝ) : 
  (∀ y : ℝ, (↑⌊y⌋ : ℝ) / y = 6 / 7 → y ≤ x) ↔ x = 35 / 6 :=
sorry

end NUMINAMATH_CALUDE_largest_x_floor_div_l3098_309861


namespace NUMINAMATH_CALUDE_product_after_digit_reversal_l3098_309868

/-- Reverses the digits of a two-digit number -/
def reverse_digits (n : ℕ) : ℕ :=
  10 * (n % 10) + (n / 10)

/-- The theorem statement -/
theorem product_after_digit_reversal (x y : ℕ) :
  x ≥ 10 ∧ x < 100 ∧  -- x is a two-digit number
  y > 0 ∧  -- y is positive
  (reverse_digits x) * y = 221 →  -- erroneous product condition
  x * y = 527 ∨ x * y = 923 :=
by sorry

end NUMINAMATH_CALUDE_product_after_digit_reversal_l3098_309868


namespace NUMINAMATH_CALUDE_louisa_travel_l3098_309891

/-- Louisa's travel problem -/
theorem louisa_travel (first_day_distance : ℝ) (speed : ℝ) (time_difference : ℝ) 
  (h1 : first_day_distance = 200)
  (h2 : speed = 50)
  (h3 : time_difference = 3)
  (h4 : first_day_distance / speed + time_difference = second_day_distance / speed) :
  second_day_distance = 350 :=
by
  sorry

end NUMINAMATH_CALUDE_louisa_travel_l3098_309891


namespace NUMINAMATH_CALUDE_reciprocal_square_inequality_l3098_309832

theorem reciprocal_square_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x ≤ y) : 
  1 / y^2 ≤ 1 / x^2 := by
sorry

end NUMINAMATH_CALUDE_reciprocal_square_inequality_l3098_309832


namespace NUMINAMATH_CALUDE_g_expression_l3098_309802

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2*x + 1

-- Define the property of g being a linear function
def is_linear (g : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), ∀ x, g x = a * x + b

-- State the theorem
theorem g_expression (g : ℝ → ℝ) (h_linear : is_linear g) 
    (h_comp : ∀ x, f (g x) = 4 * x^2) :
  (∀ x, g x = 2 * x + 1) ∨ (∀ x, g x = -2 * x + 1) := by sorry

end NUMINAMATH_CALUDE_g_expression_l3098_309802


namespace NUMINAMATH_CALUDE_second_wing_rooms_per_hall_l3098_309806

/-- Represents a hotel wing -/
structure Wing where
  floors : Nat
  hallsPerFloor : Nat
  roomsPerHall : Nat

/-- Represents a hotel with two wings -/
structure Hotel where
  wing1 : Wing
  wing2 : Wing
  totalRooms : Nat

def Hotel.secondWingRoomsPerHall (h : Hotel) : Nat :=
  (h.totalRooms - h.wing1.floors * h.wing1.hallsPerFloor * h.wing1.roomsPerHall) / 
  (h.wing2.floors * h.wing2.hallsPerFloor)

theorem second_wing_rooms_per_hall :
  let h : Hotel := {
    wing1 := { floors := 9, hallsPerFloor := 6, roomsPerHall := 32 },
    wing2 := { floors := 7, hallsPerFloor := 9, roomsPerHall := 0 }, -- roomsPerHall is unknown
    totalRooms := 4248
  }
  h.secondWingRoomsPerHall = 40 := by
  sorry

end NUMINAMATH_CALUDE_second_wing_rooms_per_hall_l3098_309806


namespace NUMINAMATH_CALUDE_equal_ratios_sum_l3098_309863

theorem equal_ratios_sum (M N : ℚ) : 
  (5 : ℚ) / 7 = M / 63 ∧ (5 : ℚ) / 7 = 70 / N → M + N = 143 := by
  sorry

end NUMINAMATH_CALUDE_equal_ratios_sum_l3098_309863


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l3098_309817

theorem necessary_but_not_sufficient
  (A B C : Set α)
  (hAnonempty : A.Nonempty)
  (hBnonempty : B.Nonempty)
  (hCnonempty : C.Nonempty)
  (hUnion : A ∪ B = C)
  (hNotSubset : ¬(B ⊆ A)) :
  (∀ x, x ∈ A → x ∈ C) ∧ (∃ x, x ∈ C ∧ x ∉ A) := by
  sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l3098_309817


namespace NUMINAMATH_CALUDE_lawrence_county_kids_count_lawrence_county_kids_count_proof_l3098_309860

theorem lawrence_county_kids_count : ℕ → ℕ → ℕ → Prop :=
  fun kids_home kids_camp total_kids =>
    kids_home = 274865 ∧ 
    kids_camp = 38608 ∧ 
    total_kids = kids_home + kids_camp → 
    total_kids = 313473

-- The proof is omitted
theorem lawrence_county_kids_count_proof : 
  ∃ (total_kids : ℕ), lawrence_county_kids_count 274865 38608 total_kids :=
sorry

end NUMINAMATH_CALUDE_lawrence_county_kids_count_lawrence_county_kids_count_proof_l3098_309860


namespace NUMINAMATH_CALUDE_smallest_benches_arrangement_l3098_309889

theorem smallest_benches_arrangement (M : ℕ+) (n : ℕ+) : 
  (9 * M.val = n ∧ 14 * M.val = n) → M.val ≥ 9 :=
by sorry

end NUMINAMATH_CALUDE_smallest_benches_arrangement_l3098_309889


namespace NUMINAMATH_CALUDE_smallest_absolute_value_l3098_309836

theorem smallest_absolute_value : ∃ x : ℝ, ∀ y : ℝ, abs x ≤ abs y :=
  sorry

end NUMINAMATH_CALUDE_smallest_absolute_value_l3098_309836


namespace NUMINAMATH_CALUDE_consecutive_integers_product_990_l3098_309837

theorem consecutive_integers_product_990 (a b c : ℤ) : 
  b = a + 1 ∧ c = b + 1 ∧ a * b * c = 990 → a + b + c = 30 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_990_l3098_309837


namespace NUMINAMATH_CALUDE_least_prime_factor_of_5_cubed_minus_5_squared_l3098_309825

theorem least_prime_factor_of_5_cubed_minus_5_squared : 
  (Nat.minFac (5^3 - 5^2) = 2) := by sorry

end NUMINAMATH_CALUDE_least_prime_factor_of_5_cubed_minus_5_squared_l3098_309825


namespace NUMINAMATH_CALUDE_circle_m_range_l3098_309839

-- Define the equation
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 + x + y - m = 0

-- Define what it means for the equation to represent a circle
def represents_circle (m : ℝ) : Prop :=
  ∃ (center_x center_y radius : ℝ), 
    ∀ (x y : ℝ), circle_equation x y m ↔ (x - center_x)^2 + (y - center_y)^2 = radius^2

-- Theorem statement
theorem circle_m_range (m : ℝ) :
  represents_circle m → m > -1/2 := by sorry

end NUMINAMATH_CALUDE_circle_m_range_l3098_309839


namespace NUMINAMATH_CALUDE_accident_insurance_probability_l3098_309898

theorem accident_insurance_probability (p1 p2 : ℝ) 
  (h1 : p1 = 1 / 20)
  (h2 : p2 = 1 / 21)
  (h3 : 0 ≤ p1 ∧ p1 ≤ 1)
  (h4 : 0 ≤ p2 ∧ p2 ≤ 1) :
  1 - (1 - p1) * (1 - p2) = 2 / 21 := by
sorry


end NUMINAMATH_CALUDE_accident_insurance_probability_l3098_309898


namespace NUMINAMATH_CALUDE_sum_of_digits_7_power_1500_l3098_309888

-- Define a function to get the last two digits of a number
def lastTwoDigits (n : ℕ) : ℕ := n % 100

-- Define a function to get the tens digit of a two-digit number
def tensDigit (n : ℕ) : ℕ := (n / 10) % 10

-- Define a function to get the units digit of a number
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Theorem statement
theorem sum_of_digits_7_power_1500 :
  tensDigit (lastTwoDigits (7^1500)) + unitsDigit (lastTwoDigits (7^1500)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_7_power_1500_l3098_309888


namespace NUMINAMATH_CALUDE_regular_ngon_parallel_pairs_l3098_309850

/-- Represents a regular n-gon with a connected path visiting each vertex exactly once -/
structure RegularNGonPath (n : ℕ) where
  path : List ℕ
  is_valid : path.length = n ∧ path.toFinset.card = n

/-- Two edges (i, j) and (p, q) are parallel in a regular n-gon if and only if i + j ≡ p + q (mod n) -/
def parallel_edges (n : ℕ) (i j p q : ℕ) : Prop :=
  (i + j) % n = (p + q) % n

/-- Counts the number of parallel pairs in a path -/
def count_parallel_pairs (n : ℕ) (path : RegularNGonPath n) : ℕ :=
  sorry

theorem regular_ngon_parallel_pairs (n : ℕ) (path : RegularNGonPath n) :
  (Even n → count_parallel_pairs n path > 0) ∧
  (Odd n → count_parallel_pairs n path ≠ 1) :=
sorry

end NUMINAMATH_CALUDE_regular_ngon_parallel_pairs_l3098_309850


namespace NUMINAMATH_CALUDE_A_intersect_B_equals_three_l3098_309847

def A : Set ℝ := {0, 1, 3}
def B : Set ℝ := {x | ∃ y, y = Real.log (x - 1)}

theorem A_intersect_B_equals_three : A ∩ B = {3} := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_equals_three_l3098_309847


namespace NUMINAMATH_CALUDE_apple_cost_calculation_l3098_309830

/-- The total cost of apples given weight, price per kg, and packaging fee -/
def total_cost (weight : ℝ) (price_per_kg : ℝ) (packaging_fee : ℝ) : ℝ :=
  weight * (price_per_kg + packaging_fee)

/-- Theorem stating that the total cost of 2.5 kg of apples is 38.875 -/
theorem apple_cost_calculation :
  total_cost 2.5 15.3 0.25 = 38.875 := by
  sorry

end NUMINAMATH_CALUDE_apple_cost_calculation_l3098_309830
