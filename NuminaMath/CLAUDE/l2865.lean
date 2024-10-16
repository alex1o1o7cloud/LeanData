import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_roots_range_l2865_286512

theorem quadratic_roots_range (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
   x₁ > 2 ∧ x₂ > 2 ∧ 
   x₁^2 + (k-2)*x₁ + 5 - k = 0 ∧ 
   x₂^2 + (k-2)*x₂ + 5 - k = 0) → 
  -5 < k ∧ k < -4 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_range_l2865_286512


namespace NUMINAMATH_CALUDE_sin_square_inequality_l2865_286523

theorem sin_square_inequality (n : ℕ+) (x : ℝ) :
  n * (Real.sin x)^2 ≥ Real.sin x * Real.sin (n * x) := by
  sorry

end NUMINAMATH_CALUDE_sin_square_inequality_l2865_286523


namespace NUMINAMATH_CALUDE_jian_has_second_most_l2865_286562

-- Define the number of notebooks for each person
def jian_notebooks : ℕ := 3
def doyun_notebooks : ℕ := 5
def siu_notebooks : ℕ := 2

-- Define a function to determine if a person has the second most notebooks
def has_second_most (x y z : ℕ) : Prop :=
  (x > y ∧ x < z) ∨ (x > z ∧ x < y)

-- Theorem statement
theorem jian_has_second_most :
  has_second_most jian_notebooks siu_notebooks doyun_notebooks :=
sorry

end NUMINAMATH_CALUDE_jian_has_second_most_l2865_286562


namespace NUMINAMATH_CALUDE_ceiling_floor_product_l2865_286507

theorem ceiling_floor_product (y : ℝ) : 
  y < 0 → ⌈y⌉ * ⌊y⌋ = 72 → -9 < y ∧ y < -8 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_product_l2865_286507


namespace NUMINAMATH_CALUDE_pig_count_l2865_286526

/-- Given a group of pigs and hens, if the total number of legs is 22 more than twice 
    the total number of heads, then the number of pigs is 11. -/
theorem pig_count (pigs hens : ℕ) : 
  4 * pigs + 2 * hens = 2 * (pigs + hens) + 22 → pigs = 11 := by
  sorry

end NUMINAMATH_CALUDE_pig_count_l2865_286526


namespace NUMINAMATH_CALUDE_trig_problem_l2865_286554

theorem trig_problem (α β : Real) 
  (h1 : Real.sin (Real.pi - α) - 2 * Real.sin (Real.pi / 2 + α) = 0) 
  (h2 : Real.tan (α + β) = -1) : 
  Real.sin α * Real.cos α + Real.sin α ^ 2 = 6 / 5 ∧ Real.tan β = 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_problem_l2865_286554


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l2865_286511

theorem quadratic_roots_condition (a : ℝ) (h1 : a ≠ 0) (h2 : a < -1) :
  ∃ (x1 x2 : ℝ), x1 > 0 ∧ x2 < 0 ∧ 
  (a * x1^2 + 2 * x1 + 1 = 0) ∧ 
  (a * x2^2 + 2 * x2 + 1 = 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l2865_286511


namespace NUMINAMATH_CALUDE_unique_quadratic_solution_l2865_286539

theorem unique_quadratic_solution (a c : ℤ) : 
  (∃! x : ℝ, a * x^2 - 6 * x + c = 0) →  -- exactly one solution
  (a + c = 12) →                        -- sum condition
  (a < c) →                             -- order condition
  (a = 3 ∧ c = 9) :=                    -- unique solution
by sorry

end NUMINAMATH_CALUDE_unique_quadratic_solution_l2865_286539


namespace NUMINAMATH_CALUDE_tan_alpha_two_expressions_l2865_286545

theorem tan_alpha_two_expressions (α : Real) (h : Real.tan α = 2) :
  (2 * Real.sin α + Real.cos α) / (Real.sin α - 3 * Real.cos α) = -5 ∧
  Real.sin α * (Real.sin α + Real.cos α) = 6/5 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_two_expressions_l2865_286545


namespace NUMINAMATH_CALUDE_cookies_left_for_birthday_l2865_286560

theorem cookies_left_for_birthday 
  (pans : ℕ) 
  (cookies_per_pan : ℕ) 
  (eaten_cookies : ℕ) 
  (burnt_cookies : ℕ) 
  (h1 : pans = 12)
  (h2 : cookies_per_pan = 15)
  (h3 : eaten_cookies = 9)
  (h4 : burnt_cookies = 6) :
  (pans * cookies_per_pan) - (eaten_cookies + burnt_cookies) = 165 := by
  sorry

end NUMINAMATH_CALUDE_cookies_left_for_birthday_l2865_286560


namespace NUMINAMATH_CALUDE_fundraiser_goal_reached_l2865_286563

/-- Proves that the total amount raised by the group is equal to the total amount needed for the trip. -/
theorem fundraiser_goal_reached (
  num_students : ℕ) 
  (individual_cost : ℕ)
  (collective_expenses : ℕ)
  (day1_raised : ℕ)
  (day2_raised : ℕ)
  (day3_raised : ℕ)
  (num_half_days : ℕ)
  (h1 : num_students = 6)
  (h2 : individual_cost = 450)
  (h3 : collective_expenses = 3000)
  (h4 : day1_raised = 600)
  (h5 : day2_raised = 900)
  (h6 : day3_raised = 400)
  (h7 : num_half_days = 4) :
  (num_students * individual_cost + collective_expenses) = 
  (day1_raised + day2_raised + day3_raised + 
   num_half_days * ((day1_raised + day2_raised + day3_raised) / 2)) := by
  sorry

#eval 6 * 450 + 3000 -- Total needed
#eval 600 + 900 + 400 + 4 * ((600 + 900 + 400) / 2) -- Total raised

end NUMINAMATH_CALUDE_fundraiser_goal_reached_l2865_286563


namespace NUMINAMATH_CALUDE_p_3_eq_10_p_condition_l2865_286553

/-- A polynomial function p: ℝ → ℝ satisfying specific conditions -/
def p : ℝ → ℝ := fun x ↦ x^2 + 1

/-- The first condition: p(3) = 10 -/
theorem p_3_eq_10 : p 3 = 10 := by sorry

/-- The second condition: p(x)p(y) = p(x) + p(y) + p(xy) - 2 for all real x and y -/
theorem p_condition (x y : ℝ) : p x * p y = p x + p y + p (x * y) - 2 := by sorry

end NUMINAMATH_CALUDE_p_3_eq_10_p_condition_l2865_286553


namespace NUMINAMATH_CALUDE_baker_cakes_sold_l2865_286572

theorem baker_cakes_sold (pastries_sold : ℕ) (pastry_cake_difference : ℕ) 
  (h1 : pastries_sold = 154)
  (h2 : pastry_cake_difference = 76) :
  pastries_sold - pastry_cake_difference = 78 := by
  sorry

end NUMINAMATH_CALUDE_baker_cakes_sold_l2865_286572


namespace NUMINAMATH_CALUDE_arithmetic_sum_10_l2865_286505

/-- An arithmetic sequence with given first and second terms -/
def arithmetic_sequence (a₁ a₂ : ℤ) (n : ℕ) : ℤ :=
  a₁ + (n - 1) * (a₂ - a₁)

/-- Sum of the first n terms of an arithmetic sequence -/
def arithmetic_sum (a₁ a₂ : ℤ) (n : ℕ) : ℤ :=
  n * (a₁ + arithmetic_sequence a₁ a₂ n) / 2

/-- Theorem: The sum of the first 10 terms of the arithmetic sequence
    with a₁ = 1 and a₂ = -3 is -170 -/
theorem arithmetic_sum_10 :
  arithmetic_sum 1 (-3) 10 = -170 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sum_10_l2865_286505


namespace NUMINAMATH_CALUDE_percentage_increase_girls_to_total_l2865_286583

def boys : ℕ := 2000
def girls : ℕ := 5000

theorem percentage_increase_girls_to_total : 
  (((boys + girls) - girls : ℚ) / girls) * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_girls_to_total_l2865_286583


namespace NUMINAMATH_CALUDE_cookie_distribution_l2865_286580

theorem cookie_distribution (total : ℚ) (blue green orange red : ℚ) : 
  blue + green + orange + red = total →
  blue + green + orange = 11 / 12 * total →
  red = 1 / 12 * total →
  blue = 1 / 6 * total →
  green = 5 / 12 * total →
  orange = 1 / 3 * total :=
by sorry

end NUMINAMATH_CALUDE_cookie_distribution_l2865_286580


namespace NUMINAMATH_CALUDE_cycle_price_calculation_l2865_286532

theorem cycle_price_calculation (selling_price : ℝ) (gain_percentage : ℝ) 
  (h1 : selling_price = 1125)
  (h2 : gain_percentage = 25) : 
  ∃ original_price : ℝ, 
    original_price * (1 + gain_percentage / 100) = selling_price ∧ 
    original_price = 900 := by
sorry

end NUMINAMATH_CALUDE_cycle_price_calculation_l2865_286532


namespace NUMINAMATH_CALUDE_claire_photos_l2865_286598

theorem claire_photos (lisa robert claire : ℕ) 
  (h1 : lisa = robert) 
  (h2 : lisa = 3 * claire) 
  (h3 : robert = claire + 16) : 
  claire = 8 := by
sorry

end NUMINAMATH_CALUDE_claire_photos_l2865_286598


namespace NUMINAMATH_CALUDE_meters_examined_l2865_286513

/-- The percentage of meters rejected as defective -/
def rejection_rate : ℝ := 0.10

/-- The number of defective meters found -/
def defective_meters : ℕ := 10

/-- The total number of meters examined -/
def total_meters : ℕ := 100

/-- Theorem stating that if the rejection rate is 10% and 10 defective meters are found,
    then the total number of meters examined is 100 -/
theorem meters_examined (h : ℝ) (defective : ℕ) (total : ℕ) 
  (h_rate : h = rejection_rate)
  (h_defective : defective = defective_meters)
  (h_total : total = total_meters) :
  ↑defective = h * ↑total := by
  sorry

end NUMINAMATH_CALUDE_meters_examined_l2865_286513


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l2865_286595

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y^2 = x - 3) → x ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l2865_286595


namespace NUMINAMATH_CALUDE_unique_valid_integer_l2865_286527

-- Define a type for 10-digit integers
def TenDigitInteger := Fin 10 → Fin 10

-- Define a property for strictly increasing sequence
def StrictlyIncreasing (n : TenDigitInteger) : Prop :=
  ∀ i j : Fin 10, i < j → n i < n j

-- Define a property for using each digit exactly once
def UsesEachDigitOnce (n : TenDigitInteger) : Prop :=
  ∀ d : Fin 10, ∃! i : Fin 10, n i = d

-- Define the set of valid integers
def ValidIntegers : Set TenDigitInteger :=
  {n | n 0 ≠ 0 ∧ StrictlyIncreasing n ∧ UsesEachDigitOnce n}

-- Theorem statement
theorem unique_valid_integer : ∃! n : TenDigitInteger, n ∈ ValidIntegers := by
  sorry

end NUMINAMATH_CALUDE_unique_valid_integer_l2865_286527


namespace NUMINAMATH_CALUDE_lcm_20_45_75_l2865_286566

theorem lcm_20_45_75 : Nat.lcm 20 (Nat.lcm 45 75) = 900 := by
  sorry

end NUMINAMATH_CALUDE_lcm_20_45_75_l2865_286566


namespace NUMINAMATH_CALUDE_prove_correct_statements_l2865_286594

-- Define the types of relationships
inductive Relationship
| Functional
| Correlation

-- Define the properties of relationships
def isDeterministic (r : Relationship) : Prop :=
  match r with
  | Relationship.Functional => True
  | Relationship.Correlation => False

-- Define regression analysis
def regressionAnalysis (r : Relationship) : Prop :=
  match r with
  | Relationship.Functional => False
  | Relationship.Correlation => True

-- Define the set of correct statements
def correctStatements : Set Nat :=
  {1, 2, 4}

-- Theorem to prove
theorem prove_correct_statements :
  (isDeterministic Relationship.Functional) ∧
  (¬isDeterministic Relationship.Correlation) ∧
  (regressionAnalysis Relationship.Correlation) →
  correctStatements = {1, 2, 4} := by
  sorry

end NUMINAMATH_CALUDE_prove_correct_statements_l2865_286594


namespace NUMINAMATH_CALUDE_rogers_money_l2865_286548

theorem rogers_money (initial amount_spent final : ℤ) 
  (h1 : initial = 45)
  (h2 : amount_spent = 20)
  (h3 : final = 71) :
  final - (initial - amount_spent) = 46 := by
  sorry

end NUMINAMATH_CALUDE_rogers_money_l2865_286548


namespace NUMINAMATH_CALUDE_original_prices_theorem_l2865_286564

def shirt_discount : Float := 0.20
def shoes_discount : Float := 0.30
def jacket_discount : Float := 0.10

def discounted_shirt_price : Float := 780
def discounted_shoes_price : Float := 2100
def discounted_jacket_price : Float := 2700

def original_shirt_price : Float := discounted_shirt_price / (1 - shirt_discount)
def original_shoes_price : Float := discounted_shoes_price / (1 - shoes_discount)
def original_jacket_price : Float := discounted_jacket_price / (1 - jacket_discount)

theorem original_prices_theorem :
  original_shirt_price = 975 ∧
  original_shoes_price = 3000 ∧
  original_jacket_price = 3000 :=
by sorry

end NUMINAMATH_CALUDE_original_prices_theorem_l2865_286564


namespace NUMINAMATH_CALUDE_function_inequality_l2865_286590

/-- The function f(x) = x^2 - 2x -/
def f (x : ℝ) : ℝ := x^2 - 2*x

/-- The function g(x) = ax + 2, where a > 0 -/
def g (a : ℝ) (x : ℝ) : ℝ := a*x + 2

/-- The theorem statement -/
theorem function_inequality (a : ℝ) (h_a : a > 0) :
  (∀ x₁ : ℝ, ∃ x₂ : ℝ, x₂ ≥ -2 ∧ f x₁ > g a x₂) →
  a > 3/2 :=
by sorry

end NUMINAMATH_CALUDE_function_inequality_l2865_286590


namespace NUMINAMATH_CALUDE_g_of_5_equals_22_l2865_286555

-- Define the function g
def g (x : ℝ) : ℝ := 4 * x + 2

-- Theorem statement
theorem g_of_5_equals_22 : g 5 = 22 := by
  sorry

end NUMINAMATH_CALUDE_g_of_5_equals_22_l2865_286555


namespace NUMINAMATH_CALUDE_toys_per_week_l2865_286536

/-- A factory produces toys with the following conditions:
    1. Workers work 5 days a week
    2. Workers make the same number of toys every day
    3. Workers produce 680 toys each day -/
theorem toys_per_week (days_per_week : ℕ) (toys_per_day : ℕ) 
  (h1 : days_per_week = 5)
  (h2 : toys_per_day = 680) :
  days_per_week * toys_per_day = 3400 := by
  sorry

end NUMINAMATH_CALUDE_toys_per_week_l2865_286536


namespace NUMINAMATH_CALUDE_solve_linear_equation_l2865_286519

theorem solve_linear_equation (x : ℝ) (h : 5 * x + 3 = 10 * x - 22) : x = 5 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l2865_286519


namespace NUMINAMATH_CALUDE_distribute_five_among_three_l2865_286558

/-- The number of ways to distribute n distinct objects among k distinct groups,
    with each group receiving at least one object -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 5 distinct objects among 3 distinct groups,
    with each group receiving at least one object, is 150 -/
theorem distribute_five_among_three :
  distribute 5 3 = 150 := by sorry

end NUMINAMATH_CALUDE_distribute_five_among_three_l2865_286558


namespace NUMINAMATH_CALUDE_fourth_power_sum_l2865_286571

theorem fourth_power_sum (a b c : ℝ) 
  (h1 : a + b + c = 1) 
  (h2 : a^2 + b^2 + c^2 = 3) 
  (h3 : a^3 + b^3 + c^3 = 3) : 
  a^4 + b^4 + c^4 = 37/6 := by
sorry

end NUMINAMATH_CALUDE_fourth_power_sum_l2865_286571


namespace NUMINAMATH_CALUDE_max_value_constraint_l2865_286537

theorem max_value_constraint (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 2) :
  x + y^3 + z^2 ≤ 8 ∧ ∃ (a b c : ℝ), a + b^3 + c^2 = 8 ∧ 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ a + b + c = 2 := by
sorry

end NUMINAMATH_CALUDE_max_value_constraint_l2865_286537


namespace NUMINAMATH_CALUDE_point_not_in_transformed_plane_l2865_286574

/-- A plane in 3D space -/
structure Plane where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ

/-- A point in 3D space -/
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Similarity transformation of a plane -/
def transformPlane (p : Plane) (k : ℝ) : Plane :=
  { A := p.A, B := p.B, C := p.C, D := k * p.D }

/-- Check if a point satisfies a plane equation -/
def satisfiesPlane (point : Point) (plane : Plane) : Prop :=
  plane.A * point.x + plane.B * point.y + plane.C * point.z + plane.D = 0

theorem point_not_in_transformed_plane :
  let originalPlane : Plane := { A := 7, B := -6, C := 1, D := -5 }
  let k : ℝ := -2
  let A : Point := { x := 1, y := 1, z := 1 }
  let transformedPlane := transformPlane originalPlane k
  ¬ satisfiesPlane A transformedPlane := by
  sorry

end NUMINAMATH_CALUDE_point_not_in_transformed_plane_l2865_286574


namespace NUMINAMATH_CALUDE_three_custom_op_three_equals_six_l2865_286582

-- Define the custom operation
def customOp (m n : ℕ) : ℕ := n ^ 2 - m

-- State the theorem
theorem three_custom_op_three_equals_six :
  customOp 3 3 = 6 := by sorry

end NUMINAMATH_CALUDE_three_custom_op_three_equals_six_l2865_286582


namespace NUMINAMATH_CALUDE_optimal_bouquet_l2865_286506

/-- Represents the number of flowers Kyle picked last year -/
def last_year_flowers : ℕ := 12

/-- Represents the total number of flowers needed this year -/
def total_flowers : ℕ := 2 * last_year_flowers

/-- Represents the number of roses Kyle picked from his garden this year -/
def picked_roses : ℕ := last_year_flowers / 2

/-- Represents the cost of a rose -/
def rose_cost : ℕ := 3

/-- Represents the cost of a tulip -/
def tulip_cost : ℕ := 2

/-- Represents the cost of a daisy -/
def daisy_cost : ℕ := 1

/-- Represents Kyle's budget constraint -/
def budget : ℕ := 30

/-- Represents the number of additional flowers Kyle needs to buy -/
def flowers_to_buy : ℕ := total_flowers - picked_roses

theorem optimal_bouquet (roses tulips daisies : ℕ) :
  roses + tulips + daisies = flowers_to_buy →
  rose_cost * roses + tulip_cost * tulips + daisy_cost * daisies ≤ budget →
  roses ≤ 9 ∧
  (roses = 9 → tulips = 1 ∧ daisies = 1) :=
sorry

end NUMINAMATH_CALUDE_optimal_bouquet_l2865_286506


namespace NUMINAMATH_CALUDE_ab_nonzero_sufficient_for_a_nonzero_l2865_286561

theorem ab_nonzero_sufficient_for_a_nonzero (a b : ℝ) : 
  (∀ a b, a * b ≠ 0 → a ≠ 0) ∧ 
  ¬(∀ a b, a ≠ 0 → a * b ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_ab_nonzero_sufficient_for_a_nonzero_l2865_286561


namespace NUMINAMATH_CALUDE_cindy_marbles_l2865_286570

theorem cindy_marbles (initial_marbles : ℕ) (friends : ℕ) (remaining_multiplier : ℕ) (remaining_total : ℕ) :
  initial_marbles = 500 →
  friends = 4 →
  remaining_multiplier = 4 →
  remaining_total = 720 →
  remaining_multiplier * (initial_marbles - friends * (initial_marbles - (remaining_total / remaining_multiplier))) = remaining_total →
  initial_marbles - (remaining_total / remaining_multiplier) = friends * 80 :=
by sorry

end NUMINAMATH_CALUDE_cindy_marbles_l2865_286570


namespace NUMINAMATH_CALUDE_M_is_range_of_f_l2865_286503

def M : Set ℝ := {y | ∃ x, y = x^2}

def f (x : ℝ) : ℝ := x^2

theorem M_is_range_of_f : M = Set.range f := by
  sorry

end NUMINAMATH_CALUDE_M_is_range_of_f_l2865_286503


namespace NUMINAMATH_CALUDE_rectangle_width_l2865_286535

/-- Given a rectangle where the length is 2 cm shorter than the width and the perimeter is 16 cm, 
    the width of the rectangle is 5 cm. -/
theorem rectangle_width (w : ℝ) (h1 : 2 * w + 2 * (w - 2) = 16) : w = 5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_width_l2865_286535


namespace NUMINAMATH_CALUDE_inequalities_for_positive_reals_l2865_286541

theorem inequalities_for_positive_reals (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  (1 / (a * c) + a / (b^2 * c) + b * c ≥ 2 * Real.sqrt 2) ∧
  (a + b + c ≥ Real.sqrt (2 * a * b) + Real.sqrt (2 * a * c)) ∧
  (a^2 + b^2 + c^2 ≥ 2 * a * b + 2 * b * c - 2 * a * c) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_for_positive_reals_l2865_286541


namespace NUMINAMATH_CALUDE_arithmetic_sequence_increasing_iff_sum_inequality_l2865_286599

theorem arithmetic_sequence_increasing_iff_sum_inequality 
  (a : ℕ → ℝ) (d : ℝ) (S : ℕ → ℝ) : 
  (∀ n : ℕ, a n = a 1 + (n - 1) * d) →
  (∀ n : ℕ, S n = n * a 1 + n * (n - 1) / 2 * d) →
  (∀ n : ℕ, n ≥ 2 → S n < n * a n) ↔ d > 0 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_increasing_iff_sum_inequality_l2865_286599


namespace NUMINAMATH_CALUDE_geometric_sequence_and_parabola_vertex_l2865_286567

/-- Given that a, b, c, and d form a geometric sequence, and the vertex of the curve y = x^2 - 2x + 3 is (b, c), then ad = 2 -/
theorem geometric_sequence_and_parabola_vertex (a b c d : ℝ) : 
  (∃ (r : ℝ), r ≠ 0 ∧ b = a * r ∧ c = b * r ∧ d = c * r) →  -- geometric sequence condition
  (∀ x : ℝ, x^2 - 2*x + 3 ≥ c) →  -- vertex condition
  (b^2 - 2*b + 3 = c) →  -- vertex condition
  a * d = 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_and_parabola_vertex_l2865_286567


namespace NUMINAMATH_CALUDE_prime_sqrt_sum_integer_implies_equal_l2865_286544

theorem prime_sqrt_sum_integer_implies_equal (p q : ℕ) : 
  Prime p → Prime q → 
  ∃ (z : ℤ), (Int.sqrt (p^2 + 7*p*q + q^2) + Int.sqrt (p^2 + 14*p*q + q^2) = z) → 
  p = q :=
by sorry

end NUMINAMATH_CALUDE_prime_sqrt_sum_integer_implies_equal_l2865_286544


namespace NUMINAMATH_CALUDE_pigeon_chicks_count_l2865_286565

/-- Proves that each pigeon has 6 chicks given the problem conditions -/
theorem pigeon_chicks_count :
  ∀ (total_pigeons : ℕ) (adult_pigeons : ℕ) (remaining_pigeons : ℕ),
    adult_pigeons = 40 →
    remaining_pigeons = 196 →
    (remaining_pigeons : ℚ) = 0.7 * total_pigeons →
    (total_pigeons - adult_pigeons) / adult_pigeons = 6 := by
  sorry


end NUMINAMATH_CALUDE_pigeon_chicks_count_l2865_286565


namespace NUMINAMATH_CALUDE_smallest_integer_y_five_satisfies_inequality_smallest_integer_is_five_l2865_286577

theorem smallest_integer_y (y : ℤ) : (7 + 3 * y < 25) ↔ y ≤ 5 := by sorry

theorem five_satisfies_inequality : 7 + 3 * 5 < 25 := by sorry

theorem smallest_integer_is_five : ∃ (y : ℤ), y = 5 ∧ (7 + 3 * y < 25) ∧ ∀ (z : ℤ), (7 + 3 * z < 25) → z ≥ y := by sorry

end NUMINAMATH_CALUDE_smallest_integer_y_five_satisfies_inequality_smallest_integer_is_five_l2865_286577


namespace NUMINAMATH_CALUDE_negative_64_to_7_6th_l2865_286510

theorem negative_64_to_7_6th : ∃ (z : ℂ), z^6 = (-64)^7 ∧ z = 128 * Complex.I :=
by
  sorry

end NUMINAMATH_CALUDE_negative_64_to_7_6th_l2865_286510


namespace NUMINAMATH_CALUDE_triangle_and_function_problem_l2865_286509

open Real

theorem triangle_and_function_problem 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (m α : ℝ) :
  (2 * c * cos B = 2 * a + b) →
  (∀ x, 2 * sin (2 * x + π / 6) + m * cos (2 * x) = 2 * sin (2 * (C / 2 - x) + π / 6) + m * cos (2 * (C / 2 - x))) →
  (2 * sin (α + π / 6) + m * cos α = 6 / 5) →
  cos (2 * α + C) = -7 / 25 := by
sorry

end NUMINAMATH_CALUDE_triangle_and_function_problem_l2865_286509


namespace NUMINAMATH_CALUDE_ellipse_m_range_l2865_286586

/-- Given an ellipse represented by the equation x²/(m-1) + y²/(2-m) = 1 with foci on the y-axis,
    the range of values for m is (1, 3/2). -/
theorem ellipse_m_range (x y m : ℝ) :
  (∀ x y, x^2 / (m - 1) + y^2 / (2 - m) = 1) →  -- Ellipse equation
  (∃ c : ℝ, ∀ x, x^2 / (m - 1) + 0^2 / (2 - m) = 1 → x^2 ≤ c^2) →  -- Foci on y-axis
  1 < m ∧ m < 3/2 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_m_range_l2865_286586


namespace NUMINAMATH_CALUDE_simplify_polynomial_l2865_286547

theorem simplify_polynomial (x : ℝ) : 3 * (3 * x^2 + 9 * x - 4) - 2 * (x^2 + 7 * x - 14) = 7 * x^2 + 13 * x + 16 := by
  sorry

end NUMINAMATH_CALUDE_simplify_polynomial_l2865_286547


namespace NUMINAMATH_CALUDE_C_power_50_is_identity_l2865_286518

def C : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![3, 2],
    ![-8, -5]]

theorem C_power_50_is_identity :
  C ^ 50 = (1 : Matrix (Fin 2) (Fin 2) ℤ) := by
  sorry

end NUMINAMATH_CALUDE_C_power_50_is_identity_l2865_286518


namespace NUMINAMATH_CALUDE_show_length_is_52_hours_l2865_286540

-- Define the number of hours in a day
def hours_in_day : ℕ := 24

-- Define the watching time for each day
def monday_hours : ℕ := hours_in_day / 2
def tuesday_hours : ℕ := 4
def wednesday_hours : ℕ := hours_in_day / 4
def thursday_hours : ℕ := (monday_hours + tuesday_hours + wednesday_hours) / 2
def friday_hours : ℕ := 19

-- Define the total show length
def total_show_length : ℕ := monday_hours + tuesday_hours + wednesday_hours + thursday_hours + friday_hours

-- Theorem to prove
theorem show_length_is_52_hours : total_show_length = 52 := by
  sorry

end NUMINAMATH_CALUDE_show_length_is_52_hours_l2865_286540


namespace NUMINAMATH_CALUDE_total_additions_in_half_hour_l2865_286524

/-- The number of additions a single computer can perform per second -/
def additions_per_second : ℕ := 15000

/-- The number of computers -/
def num_computers : ℕ := 3

/-- The number of seconds in half an hour -/
def seconds_in_half_hour : ℕ := 1800

/-- The total number of additions performed by all computers in half an hour -/
def total_additions : ℕ := additions_per_second * num_computers * seconds_in_half_hour

theorem total_additions_in_half_hour :
  total_additions = 81000000 := by sorry

end NUMINAMATH_CALUDE_total_additions_in_half_hour_l2865_286524


namespace NUMINAMATH_CALUDE_simplify_fraction_l2865_286525

theorem simplify_fraction : (4^5 + 4^3) / (4^4 - 4^2 + 2) = 544 / 121 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2865_286525


namespace NUMINAMATH_CALUDE_min_quotient_three_digit_number_l2865_286585

theorem min_quotient_three_digit_number : 
  ∀ a b c : ℕ, 
  1 ≤ a ∧ a ≤ 9 ∧ 
  1 ≤ b ∧ b ≤ 9 ∧ 
  1 ≤ c ∧ c ≤ 9 ∧ 
  a ≠ b ∧ b ≠ c ∧ a ≠ c → 
  (100 * a + 10 * b + c : ℚ) / (a + b + c) ≥ 10.5 :=
by sorry

end NUMINAMATH_CALUDE_min_quotient_three_digit_number_l2865_286585


namespace NUMINAMATH_CALUDE_farmer_cow_count_farmer_has_52_cows_l2865_286514

/-- The number of cows a farmer has, given milk production data. -/
theorem farmer_cow_count : ℕ :=
  let milk_per_cow_per_day : ℕ := 5
  let days_per_week : ℕ := 7
  let total_milk_per_week : ℕ := 1820
  let milk_per_cow_per_week : ℕ := milk_per_cow_per_day * days_per_week
  total_milk_per_week / milk_per_cow_per_week

/-- Proof that the farmer has 52 cows. -/
theorem farmer_has_52_cows : farmer_cow_count = 52 := by
  sorry

end NUMINAMATH_CALUDE_farmer_cow_count_farmer_has_52_cows_l2865_286514


namespace NUMINAMATH_CALUDE_water_bottle_problem_l2865_286589

def water_bottle_capacity (initial_capacity : ℝ) : Prop :=
  let remaining_after_first_drink := (3/4) * initial_capacity
  let remaining_after_second_drink := (1/3) * remaining_after_first_drink
  remaining_after_second_drink = 1

theorem water_bottle_problem : ∃ (c : ℝ), water_bottle_capacity c ∧ c = 4 := by
  sorry

end NUMINAMATH_CALUDE_water_bottle_problem_l2865_286589


namespace NUMINAMATH_CALUDE_intersection_M_N_l2865_286500

def M : Set ℝ := {y | ∃ x, y = |Real.cos x ^ 2 - Real.sin x ^ 2|}

def N : Set ℝ := {x | ∃ y, y = Real.log (1 - x^2)}

theorem intersection_M_N : M ∩ N = {x | 0 ≤ x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2865_286500


namespace NUMINAMATH_CALUDE_find_w_l2865_286520

/-- Given that ( √ 1.21 ) / ( √ 0.81 ) + ( √ 1.44 ) / ( √ w ) = 2.9365079365079367, prove that w = 0.49 -/
theorem find_w (w : ℝ) (h : Real.sqrt 1.21 / Real.sqrt 0.81 + Real.sqrt 1.44 / Real.sqrt w = 2.9365079365079367) : 
  w = 0.49 := by
  sorry

end NUMINAMATH_CALUDE_find_w_l2865_286520


namespace NUMINAMATH_CALUDE_exists_fib_divisible_l2865_286517

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- For any natural number m, there exists a Fibonacci number divisible by m -/
theorem exists_fib_divisible (m : ℕ) : ∃ n : ℕ, n ≥ 1 ∧ m ∣ fib n := by
  sorry

end NUMINAMATH_CALUDE_exists_fib_divisible_l2865_286517


namespace NUMINAMATH_CALUDE_simplify_expression_l2865_286575

theorem simplify_expression (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (x + y + z)⁻¹ * (x⁻¹ + y⁻¹ + z⁻¹) = (x*y*z)⁻¹ * (x + y + z)⁻¹ :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l2865_286575


namespace NUMINAMATH_CALUDE_solution_implies_k_value_l2865_286531

theorem solution_implies_k_value (k : ℝ) :
  (∃ x y : ℝ, k * x + y = 5 ∧ x = 2 ∧ y = 1) → k = 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_implies_k_value_l2865_286531


namespace NUMINAMATH_CALUDE_second_box_capacity_l2865_286579

/-- Represents the amount of clay a box can hold based on its dimensions -/
def clay_capacity (height width length : ℝ) : ℝ := sorry

theorem second_box_capacity :
  let first_box_height : ℝ := 2
  let first_box_width : ℝ := 3
  let first_box_length : ℝ := 5
  let first_box_capacity : ℝ := 40
  let second_box_height : ℝ := 2 * first_box_height
  let second_box_width : ℝ := 3 * first_box_width
  let second_box_length : ℝ := first_box_length
  clay_capacity first_box_height first_box_width first_box_length = first_box_capacity →
  clay_capacity second_box_height second_box_width second_box_length = 240 :=
by sorry

end NUMINAMATH_CALUDE_second_box_capacity_l2865_286579


namespace NUMINAMATH_CALUDE_parallel_through_common_parallel_l2865_286576

-- Define the types for lines and the parallel relation
variable {Line : Type}
variable (parallel : Line → Line → Prop)

-- State the axiom of parallels
axiom parallel_transitive {x y z : Line} : parallel x z → parallel y z → parallel x y

-- Theorem statement
theorem parallel_through_common_parallel (a b c : Line) :
  parallel a c → parallel b c → parallel a b :=
by sorry

end NUMINAMATH_CALUDE_parallel_through_common_parallel_l2865_286576


namespace NUMINAMATH_CALUDE_isosceles_if_root_one_right_angled_if_equal_roots_l2865_286542

-- Define the triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (ha : a > 0)
  (hb : b > 0)
  (hc : c > 0)

-- Define the quadratic equation
def quadratic_equation (t : Triangle) (x : ℝ) : Prop :=
  (t.a - t.c) * x^2 - 2 * t.b * x + (t.a + t.c) = 0

-- Define isosceles triangle
def is_isosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.a = t.c

-- Define right-angled triangle
def is_right_angled (t : Triangle) : Prop :=
  t.a^2 = t.b^2 + t.c^2 ∨ t.b^2 = t.a^2 + t.c^2 ∨ t.c^2 = t.a^2 + t.b^2

-- Theorem 1
theorem isosceles_if_root_one (t : Triangle) :
  quadratic_equation t 1 → is_isosceles t :=
sorry

-- Theorem 2
theorem right_angled_if_equal_roots (t : Triangle) :
  (∃ x : ℝ, ∀ y : ℝ, quadratic_equation t y ↔ y = x) → is_right_angled t :=
sorry

end NUMINAMATH_CALUDE_isosceles_if_root_one_right_angled_if_equal_roots_l2865_286542


namespace NUMINAMATH_CALUDE_square_area_ratio_l2865_286534

theorem square_area_ratio (x : ℝ) (h : x > 0) : 
  (x^2) / ((4*x)^2) = 1 / 16 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_l2865_286534


namespace NUMINAMATH_CALUDE_total_cost_is_360_l2865_286502

def calculate_total_cost (sale_prices : List ℝ) (discounts : List ℝ) 
  (installation_fee : ℝ) (disposal_fee : ℝ) : ℝ :=
  let discounted_prices := List.zipWith (·-·) sale_prices discounts
  let with_installation := List.map (·+installation_fee) discounted_prices
  let total_per_tire := List.map (·+disposal_fee) with_installation
  List.sum total_per_tire

theorem total_cost_is_360 :
  let sale_prices := [75, 90, 120, 150]
  let discounts := [20, 30, 45, 60]
  let installation_fee := 15
  let disposal_fee := 5
  calculate_total_cost sale_prices discounts installation_fee disposal_fee = 360 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_360_l2865_286502


namespace NUMINAMATH_CALUDE_largest_divisor_n4_minus_n_l2865_286587

/-- A positive integer n is composite if it has a factor other than 1 and itself -/
def IsComposite (n : ℕ) : Prop :=
  ∃ m : ℕ, 1 < m ∧ m < n ∧ n % m = 0

/-- The largest integer that always divides n^4 - n for all composite n is 6 -/
theorem largest_divisor_n4_minus_n (n : ℕ) (h : IsComposite n) :
  (∀ k : ℕ, k > 6 → ∃ m : ℕ, IsComposite m ∧ (m^4 - m) % k ≠ 0) ∧
  (∀ n : ℕ, IsComposite n → (n^4 - n) % 6 = 0) :=
sorry

end NUMINAMATH_CALUDE_largest_divisor_n4_minus_n_l2865_286587


namespace NUMINAMATH_CALUDE_quadratic_inequality_l2865_286557

theorem quadratic_inequality (x : ℝ) : x^2 - 4*x ≥ 0 ↔ x ≥ 4 ∨ x ≤ 0 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l2865_286557


namespace NUMINAMATH_CALUDE_quadratic_common_root_l2865_286530

-- Define the quadratic functions
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c
def g (a b c x : ℝ) : ℝ := c * x^2 + b * x + a

-- State the theorem
theorem quadratic_common_root (a b c : ℝ) :
  (∃! x, f a b c x + g a b c x = 0) →
  (∃ x, f a b c x = 0 ∧ g a b c x = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_common_root_l2865_286530


namespace NUMINAMATH_CALUDE_bmw_sales_l2865_286522

theorem bmw_sales (total : ℕ) (mercedes_percent : ℚ) (nissan_percent : ℚ) (ford_percent : ℚ) (chevrolet_percent : ℚ) 
  (h_total : total = 300)
  (h_mercedes : mercedes_percent = 20 / 100)
  (h_nissan : nissan_percent = 25 / 100)
  (h_ford : ford_percent = 10 / 100)
  (h_chevrolet : chevrolet_percent = 18 / 100) :
  ↑(total - (mercedes_percent + nissan_percent + ford_percent + chevrolet_percent).num * total / (mercedes_percent + nissan_percent + ford_percent + chevrolet_percent).den) = 81 := by
  sorry


end NUMINAMATH_CALUDE_bmw_sales_l2865_286522


namespace NUMINAMATH_CALUDE_denis_neighbors_l2865_286559

-- Define the students
inductive Student : Type
| Anya : Student
| Borya : Student
| Vera : Student
| Gena : Student
| Denis : Student

-- Define the line as a function from position (1 to 5) to Student
def Line : Type := Fin 5 → Student

-- Define what it means for two students to be adjacent
def adjacent (s1 s2 : Student) (line : Line) : Prop :=
  ∃ i : Fin 4, (line i = s1 ∧ line (i.succ) = s2) ∨ (line i = s2 ∧ line (i.succ) = s1)

-- State the theorem
theorem denis_neighbors (line : Line) : 
  (line 0 = Student.Borya) →  -- Borya is at the beginning
  (adjacent Student.Vera Student.Anya line ∧ ¬adjacent Student.Vera Student.Gena line) →  -- Vera next to Anya but not Gena
  (¬adjacent Student.Anya Student.Borya line ∧ ¬adjacent Student.Anya Student.Gena line ∧ ¬adjacent Student.Borya Student.Gena line) →  -- Anya, Borya, Gena not adjacent
  (adjacent Student.Denis Student.Anya line ∧ adjacent Student.Denis Student.Gena line) :=  -- Denis is next to Anya and Gena
by sorry

end NUMINAMATH_CALUDE_denis_neighbors_l2865_286559


namespace NUMINAMATH_CALUDE_sqrt_seven_minus_one_half_less_than_one_l2865_286578

theorem sqrt_seven_minus_one_half_less_than_one :
  (Real.sqrt 7 - 1) / 2 < 1 := by sorry

end NUMINAMATH_CALUDE_sqrt_seven_minus_one_half_less_than_one_l2865_286578


namespace NUMINAMATH_CALUDE_pushups_total_l2865_286584

def zachary_pushups : ℕ := 47

def david_pushups (zachary : ℕ) : ℕ := zachary + 15

def emily_pushups (david : ℕ) : ℕ := 2 * david

def total_pushups (zachary david emily : ℕ) : ℕ := zachary + david + emily

theorem pushups_total :
  total_pushups zachary_pushups (david_pushups zachary_pushups) (emily_pushups (david_pushups zachary_pushups)) = 233 := by
  sorry

end NUMINAMATH_CALUDE_pushups_total_l2865_286584


namespace NUMINAMATH_CALUDE_plate_on_square_table_l2865_286533

/-- The distance from the edge of a round plate to the bottom edge of a square table -/
def plate_to_bottom_edge (top_margin left_margin right_margin : ℝ) : ℝ :=
  left_margin + right_margin - top_margin

theorem plate_on_square_table 
  (top_margin left_margin right_margin : ℝ) 
  (h_top : top_margin = 10)
  (h_left : left_margin = 63)
  (h_right : right_margin = 20) :
  plate_to_bottom_edge top_margin left_margin right_margin = 73 := by
sorry

end NUMINAMATH_CALUDE_plate_on_square_table_l2865_286533


namespace NUMINAMATH_CALUDE_company_female_employees_l2865_286588

theorem company_female_employees :
  ∀ (total_employees : ℕ) 
    (advanced_degrees : ℕ) 
    (males_college_only : ℕ) 
    (females_advanced : ℕ),
  total_employees = 180 →
  advanced_degrees = 90 →
  males_college_only = 35 →
  females_advanced = 55 →
  ∃ (female_employees : ℕ),
    female_employees = 110 ∧
    female_employees = (total_employees - advanced_degrees - males_college_only) + females_advanced :=
by
  sorry

end NUMINAMATH_CALUDE_company_female_employees_l2865_286588


namespace NUMINAMATH_CALUDE_spinner_probability_l2865_286592

/-- Represents the outcomes of the spinner -/
inductive SpinnerOutcome
| one
| two
| three
| five

/-- Represents a three-digit number formed by three spins -/
structure ThreeDigitNumber where
  hundreds : SpinnerOutcome
  tens : SpinnerOutcome
  units : SpinnerOutcome

def isDivisibleByFive (n : ThreeDigitNumber) : Prop :=
  n.units = SpinnerOutcome.five

def totalOutcomes : ℕ := 4^3

def favorableOutcomes : ℕ := 4 * 4

theorem spinner_probability :
  (favorableOutcomes : ℚ) / totalOutcomes = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_spinner_probability_l2865_286592


namespace NUMINAMATH_CALUDE_arithmetic_sum_l2865_286556

def arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 1 = 2 →
  a 2 + a 3 = 13 →
  a 4 + a 5 + a 6 = 42 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sum_l2865_286556


namespace NUMINAMATH_CALUDE_rectangle_area_l2865_286552

/-- Given a rectangle with perimeter 80 meters and length three times the width, 
    prove that its area is 300 square meters. -/
theorem rectangle_area (l w : ℝ) 
  (perimeter_eq : 2 * l + 2 * w = 80)
  (length_width_relation : l = 3 * w) : 
  l * w = 300 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l2865_286552


namespace NUMINAMATH_CALUDE_max_value_quadratic_l2865_286543

theorem max_value_quadratic (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, x^2 + 2*x + a^2 - 1 ≤ 16) ∧ 
  (∃ x ∈ Set.Icc 1 2, x^2 + 2*x + a^2 - 1 = 16) → 
  a = 3 ∨ a = -3 := by
sorry

end NUMINAMATH_CALUDE_max_value_quadratic_l2865_286543


namespace NUMINAMATH_CALUDE_female_officers_count_l2865_286550

theorem female_officers_count (total_on_duty : ℕ) (female_on_duty_ratio : ℚ) 
  (female_duty_percentage : ℚ) (h1 : total_on_duty = 300) 
  (h2 : female_on_duty_ratio = 1/2) (h3 : female_duty_percentage = 15/100) : 
  ∃ (total_female : ℕ), total_female = 1000 ∧ 
  (total_on_duty : ℚ) * female_on_duty_ratio * (1/female_duty_percentage) = total_female := by
sorry

end NUMINAMATH_CALUDE_female_officers_count_l2865_286550


namespace NUMINAMATH_CALUDE_morning_pear_sales_l2865_286573

/-- Represents the sale of pears by a salesman in a day. -/
structure PearSales where
  morning : ℝ
  afternoon : ℝ
  total : ℝ

/-- Theorem stating the number of kilograms of pears sold in the morning. -/
theorem morning_pear_sales (sales : PearSales) 
  (h1 : sales.afternoon = 2 * sales.morning)
  (h2 : sales.total = 360)
  (h3 : sales.total = sales.morning + sales.afternoon) :
  sales.morning = 120 := by
  sorry

end NUMINAMATH_CALUDE_morning_pear_sales_l2865_286573


namespace NUMINAMATH_CALUDE_reservoir_capacity_shortage_l2865_286529

/-- Proves that the normal level of a reservoir is 7 million gallons short of total capacity
    given specific conditions about the current amount and capacity. -/
theorem reservoir_capacity_shortage :
  ∀ (current_amount normal_level total_capacity : ℝ),
  current_amount = 6 →
  current_amount = 2 * normal_level →
  current_amount = 0.6 * total_capacity →
  total_capacity - normal_level = 7 := by
sorry

end NUMINAMATH_CALUDE_reservoir_capacity_shortage_l2865_286529


namespace NUMINAMATH_CALUDE_size_relationship_l2865_286596

theorem size_relationship : 5^30 < 3^50 ∧ 3^50 < 4^40 := by
  sorry

end NUMINAMATH_CALUDE_size_relationship_l2865_286596


namespace NUMINAMATH_CALUDE_polygon_vertices_from_diagonals_l2865_286597

/-- The number of diagonals that can be drawn from one vertex of a polygon. -/
def diagonals_from_vertex (n : ℕ) : ℕ := n - 3

/-- Theorem: A polygon with 6 diagonals from one vertex has 9 vertices. -/
theorem polygon_vertices_from_diagonals :
  ∃ (n : ℕ), n > 2 ∧ diagonals_from_vertex n = 6 → n = 9 :=
by sorry

end NUMINAMATH_CALUDE_polygon_vertices_from_diagonals_l2865_286597


namespace NUMINAMATH_CALUDE_M_values_l2865_286546

theorem M_values (a b : ℚ) (h : a * b ≠ 0) :
  let M := (2 * abs a) / a + (3 * b) / abs b
  M = 1 ∨ M = -1 ∨ M = 5 ∨ M = -5 :=
sorry

end NUMINAMATH_CALUDE_M_values_l2865_286546


namespace NUMINAMATH_CALUDE_tournament_games_l2865_286538

/-- The number of games played in a single-elimination tournament -/
def games_played (n : ℕ) : ℕ :=
  n - 1

/-- Theorem: In a single-elimination tournament with 32 teams, 31 games are played -/
theorem tournament_games : games_played 32 = 31 := by
  sorry

end NUMINAMATH_CALUDE_tournament_games_l2865_286538


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l2865_286515

theorem polynomial_evaluation (x : ℝ) (h1 : x > 0) (h2 : x^2 - 4*x - 12 = 0) :
  x^3 - 4*x^2 - 12*x + 16 = 16 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l2865_286515


namespace NUMINAMATH_CALUDE_trapezoid_circle_area_ratio_l2865_286501

/-- Given a trapezoid inscribed in a circle, where the larger base forms an angle α 
    with a lateral side and an angle β with the diagonal, the ratio of the area of 
    the circle to the area of the trapezoid is π / (2 sin²α sin(2β)). -/
theorem trapezoid_circle_area_ratio (α β : Real) 
  (h1 : 0 < α ∧ α < π / 2) 
  (h2 : 0 < β ∧ β < π / 2) : 
  ∃ (S_circle S_trapezoid : Real),
    S_circle > 0 ∧ S_trapezoid > 0 ∧
    S_circle / S_trapezoid = π / (2 * Real.sin α ^ 2 * Real.sin (2 * β)) :=
by sorry

end NUMINAMATH_CALUDE_trapezoid_circle_area_ratio_l2865_286501


namespace NUMINAMATH_CALUDE_three_planes_max_parts_l2865_286549

/-- The maximum number of parts that can be created by n planes in 3D space -/
def maxParts (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | k + 1 => maxParts k + k + 1

/-- Theorem: Three planes can divide 3D space into at most 8 parts -/
theorem three_planes_max_parts :
  maxParts 3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_three_planes_max_parts_l2865_286549


namespace NUMINAMATH_CALUDE_treasure_chest_gems_l2865_286568

theorem treasure_chest_gems (diamonds : ℕ) (rubies : ℕ) : 
  diamonds = 45 → rubies = 5110 → diamonds + rubies = 5155 := by
  sorry

end NUMINAMATH_CALUDE_treasure_chest_gems_l2865_286568


namespace NUMINAMATH_CALUDE_truck_toll_theorem_l2865_286551

/-- Calculates the number of axles for a truck given the total number of wheels,
    the number of wheels on the front axle, and the number of wheels on each other axle -/
def calculateAxles (totalWheels : ℕ) (frontAxleWheels : ℕ) (otherAxleWheels : ℕ) : ℕ :=
  1 + (totalWheels - frontAxleWheels) / otherAxleWheels

/-- Calculates the toll for a truck given the number of axles -/
def calculateToll (axles : ℕ) : ℚ :=
  3.5 + 0.5 * (axles - 2)

theorem truck_toll_theorem :
  let totalWheels : ℕ := 18
  let frontAxleWheels : ℕ := 2
  let otherAxleWheels : ℕ := 4
  let axles : ℕ := calculateAxles totalWheels frontAxleWheels otherAxleWheels
  calculateToll axles = 5 := by
  sorry

end NUMINAMATH_CALUDE_truck_toll_theorem_l2865_286551


namespace NUMINAMATH_CALUDE_sequence_sum_l2865_286504

def a : ℕ → ℕ
  | 0 => 2
  | 1 => 3
  | 2 => 6
  | (n + 3) => (n + 7) * a (n + 2) - 4 * (n + 3) * a (n + 1) + (4 * (n + 3) - 8) * a n

theorem sequence_sum (n : ℕ) : a n = n.factorial + 2^n := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_l2865_286504


namespace NUMINAMATH_CALUDE_function_symmetry_l2865_286528

/-- Given a real-valued function f(x) = x³ + sin(x) + 1 and a real number a such that f(a) = 2,
    prove that f(-a) = 0. -/
theorem function_symmetry (a : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = x^3 + Real.sin x + 1) 
    (h2 : f a = 2) : f (-a) = 0 := by
  sorry

end NUMINAMATH_CALUDE_function_symmetry_l2865_286528


namespace NUMINAMATH_CALUDE_expand_and_simplify_l2865_286508

theorem expand_and_simplify (x : ℝ) : (5*x - 3)*(2*x + 4) = 10*x^2 + 14*x - 12 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l2865_286508


namespace NUMINAMATH_CALUDE_lucas_payment_l2865_286593

/-- Calculates the payment for window cleaning based on given conditions -/
def calculate_payment (windows_per_floor : ℕ) (num_floors : ℕ) (pay_per_window : ℕ) 
                      (deduction_per_period : ℕ) (days_per_period : ℕ) (days_taken : ℕ) : ℕ :=
  let total_windows := windows_per_floor * num_floors
  let gross_pay := total_windows * pay_per_window
  let num_periods := days_taken / days_per_period
  let total_deduction := num_periods * deduction_per_period
  gross_pay - total_deduction

/-- Theorem stating that Lucas will be paid $16 for cleaning windows -/
theorem lucas_payment : 
  calculate_payment 3 3 2 1 3 6 = 16 := by
  sorry

end NUMINAMATH_CALUDE_lucas_payment_l2865_286593


namespace NUMINAMATH_CALUDE_quadratic_congruences_equivalence_l2865_286521

theorem quadratic_congruences_equivalence (p : Nat) (h : Nat.Prime p) :
  (∃ x, (x^2 + x + 3) % p = 0 → ∃ y, (y^2 + y + 25) % p = 0) ∧
  (¬∃ x, (x^2 + x + 3) % p = 0 → ¬∃ y, (y^2 + y + 25) % p = 0) ∧
  (∃ y, (y^2 + y + 25) % p = 0 → ∃ x, (x^2 + x + 3) % p = 0) ∧
  (¬∃ y, (y^2 + y + 25) % p = 0 → ¬∃ x, (x^2 + x + 3) % p = 0) :=
by sorry


end NUMINAMATH_CALUDE_quadratic_congruences_equivalence_l2865_286521


namespace NUMINAMATH_CALUDE_janice_purchase_problem_l2865_286591

/-- Represents the number of items at each price point -/
structure ItemCounts where
  cents50 : ℕ
  dollar150 : ℕ
  dollar250 : ℕ
  dollar350 : ℕ

/-- The problem statement -/
theorem janice_purchase_problem (items : ItemCounts) : 
  (items.cents50 + items.dollar150 + items.dollar250 + items.dollar350 = 50) →
  (50 * items.cents50 + 150 * items.dollar150 + 250 * items.dollar250 + 350 * items.dollar350 = 10000) →
  items.cents50 = 5 := by
  sorry

#check janice_purchase_problem

end NUMINAMATH_CALUDE_janice_purchase_problem_l2865_286591


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2865_286581

theorem quadratic_equation_roots (k : ℚ) : 
  (∀ x : ℚ, 2 * x^2 + 14 * x + k = 0 ↔ x = (-14 + Real.sqrt 10) / 4 ∨ x = (-14 - Real.sqrt 10) / 4) →
  k = 93 / 4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2865_286581


namespace NUMINAMATH_CALUDE_no_rational_square_in_sequence_l2865_286516

def sequence_a : ℕ → ℚ
  | 0 => 2016
  | n + 1 => sequence_a n + 2 / sequence_a n

theorem no_rational_square_in_sequence :
  ∀ n : ℕ, ¬ ∃ r : ℚ, sequence_a n = r ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_no_rational_square_in_sequence_l2865_286516


namespace NUMINAMATH_CALUDE_roller_derby_team_size_l2865_286569

theorem roller_derby_team_size :
  ∀ (num_teams : ℕ) (skates_per_member : ℕ) (laces_per_skate : ℕ) (total_laces : ℕ),
    num_teams = 4 →
    skates_per_member = 2 →
    laces_per_skate = 3 →
    total_laces = 240 →
    ∃ (members_per_team : ℕ),
      members_per_team * num_teams * skates_per_member * laces_per_skate = total_laces ∧
      members_per_team = 10 :=
by sorry

end NUMINAMATH_CALUDE_roller_derby_team_size_l2865_286569
