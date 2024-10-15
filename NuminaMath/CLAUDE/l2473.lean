import Mathlib

namespace NUMINAMATH_CALUDE_square_floor_tiles_l2473_247334

theorem square_floor_tiles (n : ℕ) (h : 2 * n - 1 = 25) : n ^ 2 = 169 := by
  sorry

end NUMINAMATH_CALUDE_square_floor_tiles_l2473_247334


namespace NUMINAMATH_CALUDE_locus_is_equidistant_l2473_247330

/-- The locus of points equidistant from the x-axis and point F(0, 2) -/
def locus_equation (x y : ℝ) : Prop :=
  y = x^2 / 4 + 1

/-- A point is equidistant from the x-axis and F(0, 2) -/
def is_equidistant (x y : ℝ) : Prop :=
  abs y = Real.sqrt (x^2 + (y - 2)^2)

/-- Theorem: The locus equation represents points equidistant from x-axis and F(0, 2) -/
theorem locus_is_equidistant :
  ∀ x y : ℝ, locus_equation x y ↔ is_equidistant x y :=
by sorry

end NUMINAMATH_CALUDE_locus_is_equidistant_l2473_247330


namespace NUMINAMATH_CALUDE_lamplighter_monkey_distance_l2473_247348

/-- Represents the speed and time of a monkey's movement --/
structure MonkeyMovement where
  speed : ℝ
  time : ℝ

/-- Calculates the total distance traveled by a Lamplighter monkey --/
def totalDistance (running : MonkeyMovement) (swinging : MonkeyMovement) : ℝ :=
  running.speed * running.time + swinging.speed * swinging.time

/-- Theorem stating the total distance traveled by the Lamplighter monkey --/
theorem lamplighter_monkey_distance :
  let running := MonkeyMovement.mk 15 5
  let swinging := MonkeyMovement.mk 10 10
  totalDistance running swinging = 175 := by sorry

end NUMINAMATH_CALUDE_lamplighter_monkey_distance_l2473_247348


namespace NUMINAMATH_CALUDE_zeros_properties_l2473_247313

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a / x + Real.log x - 3

theorem zeros_properties (a : ℝ) (x₁ x₂ : ℝ) 
  (h₁ : f a x₁ = 0) 
  (h₂ : f a x₂ = 0) 
  (h₃ : x₁ < x₂) : 
  (0 < a ∧ a < Real.exp 2) ∧ x₁ + x₂ > 2 * a := by
  sorry

end NUMINAMATH_CALUDE_zeros_properties_l2473_247313


namespace NUMINAMATH_CALUDE_factorization_proof_l2473_247354

theorem factorization_proof (x : ℝ) : 4*x*(x-5) + 7*(x-5) + 12*(x-5) = (4*x + 19)*(x-5) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l2473_247354


namespace NUMINAMATH_CALUDE_cos_15_cos_30_minus_sin_15_sin_150_l2473_247341

theorem cos_15_cos_30_minus_sin_15_sin_150 :
  Real.cos (15 * π / 180) * Real.cos (30 * π / 180) - 
  Real.sin (15 * π / 180) * Real.sin (150 * π / 180) = 
  Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_cos_15_cos_30_minus_sin_15_sin_150_l2473_247341


namespace NUMINAMATH_CALUDE_food_distribution_proof_l2473_247337

/-- The initial number of men in the group -/
def initial_men : ℕ := 760

/-- The number of additional men who join after 2 days -/
def additional_men : ℕ := 190

/-- The initial number of days the food would last -/
def initial_days : ℕ := 22

/-- The number of days that pass before additional men join -/
def days_before_addition : ℕ := 2

/-- The number of days the food lasts after additional men join -/
def remaining_days : ℕ := 16

theorem food_distribution_proof :
  initial_men * initial_days = 
  (initial_men * days_before_addition) + 
  ((initial_men + additional_men) * remaining_days) := by
  sorry

end NUMINAMATH_CALUDE_food_distribution_proof_l2473_247337


namespace NUMINAMATH_CALUDE_distribution_count_theorem_l2473_247328

/-- Represents a boat with its capacity -/
structure Boat where
  capacity : Nat

/-- Represents the distribution of people on boats -/
structure Distribution where
  adults : Nat
  children : Nat

/-- Checks if a distribution is valid (i.e., has an adult if there's a child) -/
def is_valid_distribution (d : Distribution) : Bool :=
  d.children > 0 → d.adults > 0

/-- Counts the number of valid ways to distribute people on boats -/
def count_valid_distributions (boats : List Boat) (total_adults total_children : Nat) : Nat :=
  sorry -- The actual implementation would go here

/-- The main theorem to prove -/
theorem distribution_count_theorem :
  let boats := [Boat.mk 3, Boat.mk 2, Boat.mk 1]
  count_valid_distributions boats 3 2 = 33 := by
  sorry

#check distribution_count_theorem

end NUMINAMATH_CALUDE_distribution_count_theorem_l2473_247328


namespace NUMINAMATH_CALUDE_loan_principal_is_1200_l2473_247335

/-- Calculates the principal amount of a loan given the interest rate, time period, and total interest paid. -/
def calculate_principal (rate : ℚ) (time : ℚ) (interest : ℚ) : ℚ :=
  (interest * 100) / (rate * time)

/-- Theorem stating that under the given conditions, the loan principal is $1200. -/
theorem loan_principal_is_1200 :
  let rate : ℚ := 4
  let time : ℚ := rate
  let interest : ℚ := 192
  calculate_principal rate time interest = 1200 := by sorry

end NUMINAMATH_CALUDE_loan_principal_is_1200_l2473_247335


namespace NUMINAMATH_CALUDE_fourth_rectangle_area_determined_l2473_247376

/-- Represents a rectangle with given dimensions -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.length * r.width

/-- Represents the division of a large rectangle into four smaller rectangles -/
structure DividedRectangle where
  large : Rectangle
  efgh : Rectangle
  efij : Rectangle
  ijkl : Rectangle
  ghkl : Rectangle
  h_division : 
    large.length = efgh.length + ijkl.length ∧ 
    large.width = efgh.width + efij.width

/-- Theorem stating that the area of the fourth rectangle (GHKL) is uniquely determined -/
theorem fourth_rectangle_area_determined (dr : DividedRectangle) : 
  ∃! a : ℝ, a = dr.ghkl.area ∧ 
    dr.large.area = dr.efgh.area + dr.efij.area + dr.ijkl.area + a :=
sorry

end NUMINAMATH_CALUDE_fourth_rectangle_area_determined_l2473_247376


namespace NUMINAMATH_CALUDE_largest_four_digit_divisible_by_88_and_prime_l2473_247342

theorem largest_four_digit_divisible_by_88_and_prime : ∃ (p : ℕ), 
  p.Prime ∧ 
  p > 100 ∧ 
  9944 % 88 = 0 ∧ 
  9944 % p = 0 ∧ 
  ∀ (n : ℕ), n > 9944 → n < 10000 → ¬(n % 88 = 0 ∧ ∃ (q : ℕ), q.Prime ∧ q > 100 ∧ n % q = 0) :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_divisible_by_88_and_prime_l2473_247342


namespace NUMINAMATH_CALUDE_complement_union_equals_set_l2473_247378

universe u

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {2, 3, 5}
def N : Set ℕ := {4, 5}

theorem complement_union_equals_set : (U \ (M ∪ N)) = {1, 6} := by sorry

end NUMINAMATH_CALUDE_complement_union_equals_set_l2473_247378


namespace NUMINAMATH_CALUDE_apple_price_proof_l2473_247359

def grocery_problem (total_spent milk_price cereal_price banana_price cookie_multiplier
                     milk_qty cereal_qty banana_qty cookie_qty apple_qty : ℚ) : Prop :=
  let cereal_total := cereal_price * cereal_qty
  let banana_total := banana_price * banana_qty
  let cookie_price := milk_price * cookie_multiplier
  let cookie_total := cookie_price * cookie_qty
  let known_items_total := milk_price * milk_qty + cereal_total + banana_total + cookie_total
  let apple_total := total_spent - known_items_total
  let apple_price := apple_total / apple_qty
  apple_price = 0.5

theorem apple_price_proof :
  grocery_problem 25 3 3.5 0.25 2 1 2 4 2 4 := by
  sorry

end NUMINAMATH_CALUDE_apple_price_proof_l2473_247359


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2473_247353

/-- An isosceles triangle with sides of length 4 and 8 has a perimeter of 20 -/
theorem isosceles_triangle_perimeter (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- positive side lengths
  (a = 4 ∧ b = 8 ∧ c = 8) ∨ (a = 8 ∧ b = 4 ∧ c = 8) →  -- possible configurations
  a + b > c ∧ b + c > a ∧ a + c > b →  -- triangle inequality
  a + b + c = 20 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2473_247353


namespace NUMINAMATH_CALUDE_lesser_fraction_proof_l2473_247319

theorem lesser_fraction_proof (x y : ℚ) : 
  x + y = 11/12 → x * y = 1/6 → min x y = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_lesser_fraction_proof_l2473_247319


namespace NUMINAMATH_CALUDE_triangle_abc_property_l2473_247345

theorem triangle_abc_property (A B C : Real) (a b c : Real) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  -- Sides a, b, c are positive
  0 < a ∧ 0 < b ∧ 0 < c →
  -- Given condition
  b * Real.sin B - a * Real.sin A = c →
  -- Additional conditions
  c = Real.sqrt 3 →
  C = π / 3 →
  -- Conclusions
  B - A = π / 2 ∧
  (1 / 2 : Real) * a * c * Real.sin B = Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_property_l2473_247345


namespace NUMINAMATH_CALUDE_alice_coin_difference_l2473_247323

/-- Proves that given the conditions of Alice's coin collection, she has 3 more 10-cent coins than 25-cent coins -/
theorem alice_coin_difference :
  ∀ (n d q : ℕ),
  n + d + q = 30 →
  5 * n + 10 * d + 25 * q = 435 →
  d = n + 6 →
  q = 10 →
  d - q = 3 := by
sorry

end NUMINAMATH_CALUDE_alice_coin_difference_l2473_247323


namespace NUMINAMATH_CALUDE_no_real_solutions_for_composition_l2473_247336

/-- A quadratic function f(x) = ax^2 + bx + c -/
def quadratic_function (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

/-- Theorem: If f(x) = ax^2 + bx + c is a quadratic function and f(x) = x has no real solutions,
    then f(f(x)) = x also has no real solutions -/
theorem no_real_solutions_for_composition
  (a b c : ℝ)
  (h1 : a ≠ 0)
  (h2 : ∀ x : ℝ, quadratic_function a b c x ≠ x) :
  ∀ x : ℝ, quadratic_function a b c (quadratic_function a b c x) ≠ x :=
by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_for_composition_l2473_247336


namespace NUMINAMATH_CALUDE_bags_collection_l2473_247360

/-- Calculates the total number of bags collected over three days -/
def totalBags (initial : ℕ) (day2 : ℕ) (day3 : ℕ) : ℕ :=
  initial + day2 + day3

/-- Theorem stating that the total number of bags is 20 given the specific conditions -/
theorem bags_collection :
  totalBags 10 3 7 = 20 := by
  sorry

end NUMINAMATH_CALUDE_bags_collection_l2473_247360


namespace NUMINAMATH_CALUDE_triangle_area_from_medians_l2473_247368

theorem triangle_area_from_medians (a b : ℝ) (cos_angle : ℝ) (h1 : a = 3) (h2 : b = 2 * Real.sqrt 7) (h3 : cos_angle = -3/4) :
  let sin_angle := Real.sqrt (1 - cos_angle^2)
  let sub_triangle_area := 1/2 * (2/3 * a) * (1/3 * b) * sin_angle
  6 * sub_triangle_area = 7 :=
sorry

end NUMINAMATH_CALUDE_triangle_area_from_medians_l2473_247368


namespace NUMINAMATH_CALUDE_line_intercepts_sum_l2473_247302

/-- Proves that for the line 2x - 3y - 6k = 0, if the sum of its x-intercept and y-intercept is 1, then k = 1 -/
theorem line_intercepts_sum (k : ℝ) : 
  (∃ x y : ℝ, 2*x - 3*y - 6*k = 0 ∧ 
   (2*(3*k) - 3*0 - 6*k = 0) ∧ 
   (2*0 - 3*(-2*k) - 6*k = 0) ∧ 
   3*k + (-2*k) = 1) → 
  k = 1 := by
  sorry

end NUMINAMATH_CALUDE_line_intercepts_sum_l2473_247302


namespace NUMINAMATH_CALUDE_monotonicity_intervals_min_value_l2473_247375

-- Define the function f
def f (x a : ℝ) : ℝ := -x^3 + 3*x^2 + 9*x + a

-- Define the derivative of f
def f' (x : ℝ) : ℝ := -3*x^2 + 6*x + 9

-- Theorem for monotonicity intervals
theorem monotonicity_intervals (a : ℝ) :
  (∀ x < -1, (f' x) < 0) ∧
  (∀ x > 3, (f' x) < 0) ∧
  (∀ x ∈ Set.Ioo (-1) 3, (f' x) > 0) :=
sorry

-- Theorem for minimum value
theorem min_value (a : ℝ) :
  (∃ x ∈ Set.Icc (-2) 2, f x a = 20) →
  (∃ y ∈ Set.Icc (-2) 2, f y a = -7 ∧ ∀ z ∈ Set.Icc (-2) 2, f z a ≥ -7) :=
sorry

end NUMINAMATH_CALUDE_monotonicity_intervals_min_value_l2473_247375


namespace NUMINAMATH_CALUDE_billiard_path_equals_diagonals_l2473_247374

/-- Represents a rectangle in a 2D plane -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents a point on the rectangle's perimeter -/
structure PerimeterPoint where
  x : ℝ
  y : ℝ

/-- Calculates the length of the billiard ball's path -/
def billiardPathLength (rect : Rectangle) (start : PerimeterPoint) : ℝ :=
  sorry

/-- Calculates the sum of the diagonals of the rectangle -/
def sumOfDiagonals (rect : Rectangle) : ℝ :=
  sorry

/-- Theorem: The billiard path length equals the sum of the rectangle's diagonals -/
theorem billiard_path_equals_diagonals (rect : Rectangle) (start : PerimeterPoint) :
  billiardPathLength rect start = sumOfDiagonals rect :=
  sorry

end NUMINAMATH_CALUDE_billiard_path_equals_diagonals_l2473_247374


namespace NUMINAMATH_CALUDE_ball_cost_l2473_247365

/-- Given that 3 balls cost $4.62, prove that each ball costs $1.54. -/
theorem ball_cost (total_cost : ℝ) (num_balls : ℕ) (cost_per_ball : ℝ) 
  (h1 : total_cost = 4.62)
  (h2 : num_balls = 3)
  (h3 : cost_per_ball = total_cost / num_balls) : 
  cost_per_ball = 1.54 := by
  sorry

end NUMINAMATH_CALUDE_ball_cost_l2473_247365


namespace NUMINAMATH_CALUDE_consumption_increase_l2473_247329

theorem consumption_increase (original_tax original_consumption : ℝ) 
  (h1 : original_tax > 0) (h2 : original_consumption > 0) : 
  let new_tax := 0.65 * original_tax
  let new_revenue := 0.7475 * (original_tax * original_consumption)
  ∃ (new_consumption : ℝ), 
    new_revenue = new_tax * new_consumption ∧ 
    new_consumption = 1.15 * original_consumption :=
by
  sorry

end NUMINAMATH_CALUDE_consumption_increase_l2473_247329


namespace NUMINAMATH_CALUDE_tan_product_seventh_pi_l2473_247312

theorem tan_product_seventh_pi : 
  Real.tan (π / 7) * Real.tan (2 * π / 7) * Real.tan (3 * π / 7) = Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_seventh_pi_l2473_247312


namespace NUMINAMATH_CALUDE_inequality_proof_l2473_247332

theorem inequality_proof (a b c d e f : ℝ) 
  (h : ∀ x : ℝ, a * x^2 + b * x + c ≥ |d * x^2 + e * x + f|) : 
  4 * a * c - b^2 ≥ |4 * d * f - e^2| := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2473_247332


namespace NUMINAMATH_CALUDE_linear_equation_implies_a_squared_plus_a_minus_one_equals_one_l2473_247396

theorem linear_equation_implies_a_squared_plus_a_minus_one_equals_one (a : ℝ) :
  (∀ x, ∃ k, (a + 4) * x^(|a + 3|) + 8 = k * x + 8) →
  a^2 + a - 1 = 1 :=
by sorry

end NUMINAMATH_CALUDE_linear_equation_implies_a_squared_plus_a_minus_one_equals_one_l2473_247396


namespace NUMINAMATH_CALUDE_only_234_and_468_satisfy_l2473_247351

def sumOfDigits (n : Nat) : Nat :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

def satisfiesCondition (n : Nat) : Prop :=
  n < 10000 ∧ n = 26 * sumOfDigits n

theorem only_234_and_468_satisfy :
  ∀ n : Nat, satisfiesCondition n ↔ n = 234 ∨ n = 468 := by
  sorry

end NUMINAMATH_CALUDE_only_234_and_468_satisfy_l2473_247351


namespace NUMINAMATH_CALUDE_quadratic_equations_integer_solutions_l2473_247321

theorem quadratic_equations_integer_solutions 
  (b c : ℤ) 
  (hb : b ≠ 0) 
  (hc : c ≠ 0) 
  (h1 : ∃ x y : ℤ, x ≠ y ∧ x^2 + b*x + c = 0 ∧ y^2 + b*y + c = 0)
  (h2 : ∃ u v : ℤ, u ≠ v ∧ u^2 + b*u - c = 0 ∧ v^2 + b*v - c = 0) :
  (∃ p q : ℕ+, p ≠ q ∧ 2*b^2 = p^2 + q^2) ∧
  (∃ r s : ℕ+, r ≠ s ∧ b^2 = r^2 + s^2) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equations_integer_solutions_l2473_247321


namespace NUMINAMATH_CALUDE_fraction_expansion_invariance_l2473_247322

theorem fraction_expansion_invariance (m n : ℝ) (h : m ≠ n) :
  (2 * (3 * m)) / ((3 * m) - (3 * n)) = (2 * m) / (m - n) := by
  sorry

end NUMINAMATH_CALUDE_fraction_expansion_invariance_l2473_247322


namespace NUMINAMATH_CALUDE_probability_a_speaks_truth_l2473_247326

theorem probability_a_speaks_truth 
  (prob_b : ℝ) 
  (prob_both : ℝ) 
  (h1 : prob_b = 0.60)
  (h2 : prob_both = 0.33)
  (h3 : prob_both = prob_a * prob_b)
  : prob_a = 0.55 :=
by sorry

end NUMINAMATH_CALUDE_probability_a_speaks_truth_l2473_247326


namespace NUMINAMATH_CALUDE_school_population_proof_l2473_247340

theorem school_population_proof (x : ℝ) (h1 : 162 = (x / 100) * (0.5 * x)) : x = 180 := by
  sorry

end NUMINAMATH_CALUDE_school_population_proof_l2473_247340


namespace NUMINAMATH_CALUDE_calculate_expression_l2473_247324

theorem calculate_expression : -1^4 - (1 - 0.4) * (1/3) * (2 - 3^2) = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l2473_247324


namespace NUMINAMATH_CALUDE_max_value_of_expression_l2473_247361

theorem max_value_of_expression (t : ℝ) : 
  ∃ (max : ℝ), max = (1 / 16) ∧ ∀ t, ((3^t - 4*t) * t) / (9^t) ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l2473_247361


namespace NUMINAMATH_CALUDE_fruit_store_profit_l2473_247395

-- Define the cost price
def cost_price : ℝ := 40

-- Define the linear function for weekly sales quantity
def sales_quantity (x : ℝ) : ℝ := -2 * x + 200

-- Define the profit function
def profit (x : ℝ) : ℝ := (x - cost_price) * sales_quantity x

-- Define the new profit function with increased cost
def new_profit (x m : ℝ) : ℝ := (x - cost_price - m) * sales_quantity x

theorem fruit_store_profit :
  -- 1. The selling price that maximizes profit is 70 yuan/kg
  (∀ x : ℝ, profit x ≤ profit 70) ∧
  -- 2. The maximum profit is 1800 yuan
  (profit 70 = 1800) ∧
  -- 3. When the cost price increases by m yuan/kg (m > 0), and the profit decreases
  --    for selling prices > 76 yuan/kg, then 0 < m ≤ 12
  (∀ m : ℝ, m > 0 →
    (∀ x : ℝ, x > 76 → (∀ y : ℝ, y > x → new_profit y m < new_profit x m)) →
    m ≤ 12) :=
by sorry

end NUMINAMATH_CALUDE_fruit_store_profit_l2473_247395


namespace NUMINAMATH_CALUDE_ben_savings_days_l2473_247358

/-- Calculates the number of days elapsed given Ben's savings scenario --/
def days_elapsed (daily_start : ℕ) (daily_spend : ℕ) (final_amount : ℕ) : ℕ :=
  let daily_save := daily_start - daily_spend
  let d : ℕ := (final_amount - 10) / (2 * daily_save)
  d

/-- Theorem stating that the number of days elapsed is 7 --/
theorem ben_savings_days : days_elapsed 50 15 500 = 7 := by
  sorry

end NUMINAMATH_CALUDE_ben_savings_days_l2473_247358


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2473_247327

theorem complex_equation_solution :
  ∃ (z : ℂ), z = -3/4 * I ∧ (2 : ℂ) - I * z = -1 + 3 * I * z :=
by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2473_247327


namespace NUMINAMATH_CALUDE_ellipse_properties_l2473_247315

/-- Ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0
  sum_focal_distances : ℝ → ℝ → ℝ
  eccentricity : ℝ
  focal_sum_eq : ∀ x y, x^2/a^2 + y^2/b^2 = 1 → sum_focal_distances x y = 2 * Real.sqrt 3
  ecc_eq : eccentricity = Real.sqrt 3 / 3

/-- Point on the ellipse -/
structure PointOnEllipse (E : Ellipse) where
  x : ℝ
  y : ℝ
  h : x^2/E.a^2 + y^2/E.b^2 = 1

/-- Theorem about the standard form of the ellipse and slope product -/
theorem ellipse_properties (E : Ellipse) :
  (E.a^2 = 3 ∧ E.b^2 = 2) ∧
  ∀ (P : PointOnEllipse E) (Q : PointOnEllipse E),
    P.x = 3 →
    (Q.x - 1) * (P.y - 0) + (Q.y - 0) * (P.x - 1) = 0 →
    (Q.y / Q.x) * ((Q.y - P.y) / (Q.x - P.x)) = -2/3 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_properties_l2473_247315


namespace NUMINAMATH_CALUDE_arithmetic_sum_specific_l2473_247366

def arithmetic_sum (a₁ l d : ℤ) : ℤ :=
  let n := (l - a₁) / d + 1
  n * (a₁ + l) / 2

theorem arithmetic_sum_specific : arithmetic_sum (-45) 1 2 = -528 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sum_specific_l2473_247366


namespace NUMINAMATH_CALUDE_hyperbola_vertices_distance_l2473_247349

/-- The distance between the vertices of a hyperbola with equation (y^2 / 27) - (x^2 / 11) = 1 is 6√3. -/
theorem hyperbola_vertices_distance :
  let hyperbola := {p : ℝ × ℝ | (p.2^2 / 27) - (p.1^2 / 11) = 1}
  ∃ v₁ v₂ : ℝ × ℝ, v₁ ∈ hyperbola ∧ v₂ ∈ hyperbola ∧ 
    ∀ p ∈ hyperbola, dist p v₁ ≤ dist v₁ v₂ ∧ dist p v₂ ≤ dist v₁ v₂ ∧
    dist v₁ v₂ = 6 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_vertices_distance_l2473_247349


namespace NUMINAMATH_CALUDE_zeros_of_f_l2473_247307

def f (x : ℝ) := -x^2 + 5*x - 6

theorem zeros_of_f :
  ∃ (a b : ℝ), (∀ x, f x = 0 ↔ x = a ∨ x = b) ∧ a = 2 ∧ b = 3 := by
  sorry

end NUMINAMATH_CALUDE_zeros_of_f_l2473_247307


namespace NUMINAMATH_CALUDE_range_of_a_l2473_247373

-- Define the propositions p and q
def p (x : ℝ) : Prop := (x - 5) / (x - 3) ≥ 2
def q (x a : ℝ) : Prop := x^2 - a*x ≤ x - a

-- Define the condition that not p implies not q
def not_p_implies_not_q (a : ℝ) : Prop :=
  ∀ x : ℝ, ¬(p x) → ¬(q x a)

-- State the theorem
theorem range_of_a :
  ∀ a : ℝ, (∀ x : ℝ, not_p_implies_not_q a) → a ∈ Set.Icc 1 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2473_247373


namespace NUMINAMATH_CALUDE_bakery_roll_combinations_l2473_247300

theorem bakery_roll_combinations :
  let n : ℕ := 4  -- number of remaining rolls
  let k : ℕ := 4  -- number of kinds of rolls
  let total : ℕ := 8  -- total number of rolls
  Nat.choose (n + k - 1) (k - 1) = 35 :=
by
  sorry

end NUMINAMATH_CALUDE_bakery_roll_combinations_l2473_247300


namespace NUMINAMATH_CALUDE_annie_original_seat_l2473_247389

-- Define the type for seats
inductive Seat
| one
| two
| three
| four
| five

-- Define the type for friends
inductive Friend
| Annie
| Beth
| Cass
| Dana
| Ella

-- Define the function type for seating arrangement
def SeatingArrangement := Seat → Friend

-- Define the movement function type
def Movement := SeatingArrangement → SeatingArrangement

-- Define the specific movements
def bethMove : Movement := sorry
def cassDanaSwap : Movement := sorry
def ellaMove : Movement := sorry

-- Define the property of Ella ending in an end seat
def ellaInEndSeat (arrangement : SeatingArrangement) : Prop := sorry

-- Define the theorem
theorem annie_original_seat (initial : SeatingArrangement) :
  (∃ (final : SeatingArrangement),
    final = ellaMove (cassDanaSwap (bethMove initial)) ∧
    ellaInEndSeat final) →
  initial Seat.one = Friend.Annie := by sorry

end NUMINAMATH_CALUDE_annie_original_seat_l2473_247389


namespace NUMINAMATH_CALUDE_student_rank_from_right_l2473_247394

/-- Given a student ranked 8th from the left in a group of 20 students, 
    their rank from the right is 13th. -/
theorem student_rank_from_right 
  (total_students : ℕ) 
  (rank_from_left : ℕ) 
  (h1 : total_students = 20) 
  (h2 : rank_from_left = 8) : 
  total_students - (rank_from_left - 1) = 13 := by
sorry

end NUMINAMATH_CALUDE_student_rank_from_right_l2473_247394


namespace NUMINAMATH_CALUDE_exactly_seven_numbers_satisfy_condition_l2473_247304

/-- A two-digit number is a natural number between 10 and 99 inclusive. -/
def TwoDigitNumber (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- The tens digit of a natural number. -/
def tensDigit (n : ℕ) : ℕ := n / 10

/-- The units digit of a natural number. -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The sum of digits of a two-digit number. -/
def digitSum (n : ℕ) : ℕ := tensDigit n + unitsDigit n

/-- The condition specified in the problem. -/
def satisfiesCondition (n : ℕ) : Prop :=
  TwoDigitNumber n ∧ unitsDigit (n - 2 * digitSum n) = 4

/-- The main theorem stating that exactly 7 two-digit numbers satisfy the condition. -/
theorem exactly_seven_numbers_satisfy_condition :
  ∃! (s : Finset ℕ), (∀ n ∈ s, satisfiesCondition n) ∧ s.card = 7 := by
  sorry

end NUMINAMATH_CALUDE_exactly_seven_numbers_satisfy_condition_l2473_247304


namespace NUMINAMATH_CALUDE_calculation_one_calculation_two_l2473_247362

-- Define the logarithm base 10 function
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Theorem for the first calculation
theorem calculation_one :
  (1 / Real.sqrt 0.04) + (1 / Real.sqrt 27) ^ (1/3) + (Real.sqrt 2 + 1)⁻¹ - 2 ^ (1/2) + (-2) ^ 0 = 8 := by sorry

-- Theorem for the second calculation
theorem calculation_two :
  (2/5) * lg 32 + lg 50 + Real.sqrt ((lg 3)^2 - lg 9 + 1) - lg (2/3) = 3 := by sorry

end NUMINAMATH_CALUDE_calculation_one_calculation_two_l2473_247362


namespace NUMINAMATH_CALUDE_quadratic_rewrite_l2473_247380

theorem quadratic_rewrite (a b c : ℤ) :
  (∀ x : ℝ, 16 * x^2 - 48 * x - 72 = (a * x + b)^2 + c) →
  a * b = -24 := by
sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_l2473_247380


namespace NUMINAMATH_CALUDE_yellow_balls_count_l2473_247316

theorem yellow_balls_count (total : ℕ) (red yellow green : ℕ) : 
  total = 68 →
  2 * red = yellow →
  3 * green = 4 * yellow →
  red + yellow + green = total →
  yellow = 24 := by
sorry

end NUMINAMATH_CALUDE_yellow_balls_count_l2473_247316


namespace NUMINAMATH_CALUDE_sawyer_cut_difference_l2473_247338

/-- Represents a sawyer with their stick length and number of sections sawed -/
structure Sawyer where
  stickLength : Nat
  sectionsSawed : Nat

/-- Calculates the number of cuts made by a sawyer -/
def calculateCuts (s : Sawyer) : Nat :=
  (s.stickLength / 2 - 1) * (s.sectionsSawed / (s.stickLength / 2))

theorem sawyer_cut_difference (a b c : Sawyer)
  (h1 : a.stickLength = 8 ∧ b.stickLength = 10 ∧ c.stickLength = 6)
  (h2 : a.sectionsSawed = 24 ∧ b.sectionsSawed = 25 ∧ c.sectionsSawed = 27) :
  (max (max (calculateCuts a) (calculateCuts b)) (calculateCuts c) -
   min (min (calculateCuts a) (calculateCuts b)) (calculateCuts c)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sawyer_cut_difference_l2473_247338


namespace NUMINAMATH_CALUDE_cube_root_problem_l2473_247386

theorem cube_root_problem (a : ℝ) : 
  (27 : ℝ) ^ (1/3) = a + 3 → (a + 4).sqrt = 2 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_problem_l2473_247386


namespace NUMINAMATH_CALUDE_planted_fraction_of_field_l2473_247308

theorem planted_fraction_of_field (a b c x : ℝ) (h1 : a = 5) (h2 : b = 12) 
  (h3 : c^2 = a^2 + b^2) (h4 : x^2 * (a - x) * (b - x) = 3 * c * x^2) : 
  (a * b / 2 - x^2) / (a * b / 2) = 1461 / 1470 := by
  sorry

end NUMINAMATH_CALUDE_planted_fraction_of_field_l2473_247308


namespace NUMINAMATH_CALUDE_circle_intersection_range_l2473_247347

-- Define the circles
def circle1 (x y m : ℝ) : Prop := x^2 + y^2 = m
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 6*x - 8*y - 11 = 0

-- Define the intersection of the circles
def circles_intersect (m : ℝ) : Prop :=
  ∃ x y : ℝ, circle1 x y m ∧ circle2 x y

-- Theorem statement
theorem circle_intersection_range (m : ℝ) :
  circles_intersect m ↔ 1 < m ∧ m < 121 :=
sorry

end NUMINAMATH_CALUDE_circle_intersection_range_l2473_247347


namespace NUMINAMATH_CALUDE_family_milk_consumption_l2473_247367

/-- Represents the milk consumption of a family member -/
structure MilkConsumption where
  regular : ℝ
  soy : ℝ
  almond : ℝ
  cashew : ℝ
  oat : ℝ
  coconut : ℝ
  lactoseFree : ℝ

/-- Calculates the total milk consumption excluding lactose-free milk -/
def totalConsumption (c : MilkConsumption) : ℝ :=
  c.regular + c.soy + c.almond + c.cashew + c.oat + c.coconut

/-- Represents the family's milk consumption -/
structure FamilyConsumption where
  mitch : MilkConsumption
  sister : MilkConsumption
  mother : MilkConsumption
  father : MilkConsumption
  extraSoyMilk : ℝ

theorem family_milk_consumption (family : FamilyConsumption)
    (h_mitch : family.mitch = ⟨3, 2, 1, 0, 0, 0, 0⟩)
    (h_sister : family.sister = ⟨1.5, 3, 1.5, 1, 0, 0, 0⟩)
    (h_mother : family.mother = ⟨0.5, 2.5, 0, 0, 1, 0, 0.5⟩)
    (h_father : family.father = ⟨2, 1, 3, 0, 0, 1, 0⟩)
    (h_extra_soy : family.extraSoyMilk = 7.5) :
    totalConsumption family.mitch +
    totalConsumption family.sister +
    totalConsumption family.mother +
    totalConsumption family.father +
    family.extraSoyMilk = 31.5 := by
  sorry


end NUMINAMATH_CALUDE_family_milk_consumption_l2473_247367


namespace NUMINAMATH_CALUDE_jakes_test_scores_l2473_247393

theorem jakes_test_scores (average : ℝ) (first_test : ℝ) (second_test : ℝ) (third_test : ℝ) :
  average = 75 →
  first_test = 80 →
  second_test = 90 →
  (first_test + second_test + third_test + third_test) / 4 = average →
  third_test = 65 := by
sorry

end NUMINAMATH_CALUDE_jakes_test_scores_l2473_247393


namespace NUMINAMATH_CALUDE_min_cubes_for_specific_block_l2473_247311

/-- The minimum number of cubes needed to create a hollow block -/
def min_cubes_hollow_block (length width depth : ℕ) : ℕ :=
  let total_cubes := length * width * depth
  let hollow_length := length - 2
  let hollow_width := width - 2
  let hollow_depth := depth - 2
  let hollow_cubes := hollow_length * hollow_width * hollow_depth
  total_cubes - hollow_cubes

/-- Theorem stating the minimum number of cubes needed for the specific block -/
theorem min_cubes_for_specific_block :
  min_cubes_hollow_block 4 10 7 = 200 := by
  sorry

#eval min_cubes_hollow_block 4 10 7

end NUMINAMATH_CALUDE_min_cubes_for_specific_block_l2473_247311


namespace NUMINAMATH_CALUDE_tom_total_games_l2473_247370

/-- The number of hockey games Tom attended over two years -/
def total_games (games_this_year games_last_year : ℕ) : ℕ :=
  games_this_year + games_last_year

/-- Theorem: Tom attended 13 hockey games in total over two years -/
theorem tom_total_games :
  total_games 4 9 = 13 := by
  sorry

end NUMINAMATH_CALUDE_tom_total_games_l2473_247370


namespace NUMINAMATH_CALUDE_hyperbola_real_axis_length_l2473_247318

-- Define the hyperbola and parabola
def hyperbola (a b x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1 ∧ a > 0 ∧ b > 0
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the intersection points A and B
def intersectionPoints (a b : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), hyperbola a b x₁ y₁ ∧ parabola x₁ y₁ ∧
                       hyperbola a b x₂ y₂ ∧ parabola x₂ y₂ ∧
                       (x₁ ≠ x₂ ∨ y₁ ≠ y₂)

-- Define the common focus F
def commonFocus (a b : ℝ) : Prop :=
  ∃ (xf yf : ℝ), (xf = a ∧ yf = 0) ∧ (xf = 1 ∧ yf = 0)

-- Define that line AB passes through F
def lineABThroughF (a b : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ xf yf : ℝ),
    hyperbola a b x₁ y₁ ∧ parabola x₁ y₁ ∧
    hyperbola a b x₂ y₂ ∧ parabola x₂ y₂ ∧
    commonFocus a b ∧
    (y₂ - y₁) * (xf - x₁) = (yf - y₁) * (x₂ - x₁)

-- Theorem statement
theorem hyperbola_real_axis_length
  (a b : ℝ)
  (h_intersect : intersectionPoints a b)
  (h_focus : commonFocus a b)
  (h_line : lineABThroughF a b) :
  2 * a = 2 * Real.sqrt 2 - 2 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_real_axis_length_l2473_247318


namespace NUMINAMATH_CALUDE_antonieta_initial_tickets_l2473_247301

/-- The number of tickets required for the Ferris wheel -/
def ferris_wheel_tickets : ℕ := 6

/-- The number of tickets required for the roller coaster -/
def roller_coaster_tickets : ℕ := 5

/-- The number of tickets required for the log ride -/
def log_ride_tickets : ℕ := 7

/-- The number of additional tickets Antonieta needs to buy -/
def additional_tickets_needed : ℕ := 16

/-- The initial number of tickets Antonieta has -/
def initial_tickets : ℕ := 2

theorem antonieta_initial_tickets :
  initial_tickets + additional_tickets_needed =
  ferris_wheel_tickets + roller_coaster_tickets + log_ride_tickets :=
by sorry

end NUMINAMATH_CALUDE_antonieta_initial_tickets_l2473_247301


namespace NUMINAMATH_CALUDE_line_segment_length_l2473_247377

theorem line_segment_length : Real.sqrt ((8 - 3)^2 + (16 - 4)^2) = 13 := by
  sorry

end NUMINAMATH_CALUDE_line_segment_length_l2473_247377


namespace NUMINAMATH_CALUDE_alyssa_kittens_l2473_247371

/-- The number of kittens Alyssa initially had -/
def initial_kittens : ℕ := 8

/-- The number of kittens Alyssa gave away -/
def kittens_given_away : ℕ := 4

/-- The number of kittens Alyssa now has -/
def remaining_kittens : ℕ := initial_kittens - kittens_given_away

theorem alyssa_kittens : remaining_kittens = 4 := by
  sorry

end NUMINAMATH_CALUDE_alyssa_kittens_l2473_247371


namespace NUMINAMATH_CALUDE_least_five_digit_square_cube_l2473_247339

theorem least_five_digit_square_cube : ∃ n : ℕ,
  (10000 ≤ n ∧ n < 100000) ∧  -- five-digit number
  (∃ a : ℕ, n = a^2) ∧        -- perfect square
  (∃ b : ℕ, n = b^3) ∧        -- perfect cube
  n = 15625 ∧                 -- the specific number
  (∀ m : ℕ, 
    (10000 ≤ m ∧ m < 100000) ∧ 
    (∃ x : ℕ, m = x^2) ∧ 
    (∃ y : ℕ, m = y^3) → 
    n ≤ m) :=                 -- least such number
by sorry

end NUMINAMATH_CALUDE_least_five_digit_square_cube_l2473_247339


namespace NUMINAMATH_CALUDE_annual_population_increase_rounded_l2473_247392

/-- The number of hours in a day -/
def hours_per_day : ℕ := 24

/-- The number of hours between births -/
def hours_per_birth : ℕ := 6

/-- The number of hours between deaths -/
def hours_per_death : ℕ := 10

/-- The number of days in a year -/
def days_per_year : ℕ := 365

/-- Calculate the annual population increase -/
def annual_population_increase : ℕ :=
  (hours_per_day / hours_per_birth - hours_per_day / hours_per_death) * days_per_year

/-- Round to the nearest hundred -/
def round_to_hundred (n : ℕ) : ℕ :=
  ((n + 50) / 100) * 100

/-- Theorem stating the annual population increase rounded to the nearest hundred -/
theorem annual_population_increase_rounded :
  round_to_hundred annual_population_increase = 700 := by
  sorry

end NUMINAMATH_CALUDE_annual_population_increase_rounded_l2473_247392


namespace NUMINAMATH_CALUDE_triangles_from_parallel_lines_l2473_247364

/-- The number of triangles formed by points on two parallel lines -/
theorem triangles_from_parallel_lines (n m : ℕ) (hn : n = 6) (hm : m = 8) :
  n.choose 2 * m + n * m.choose 2 = 288 := by
  sorry

end NUMINAMATH_CALUDE_triangles_from_parallel_lines_l2473_247364


namespace NUMINAMATH_CALUDE_center_is_five_l2473_247344

-- Define the grid type
def Grid := Matrix (Fin 3) (Fin 3) Nat

-- Define the property of consecutive numbers sharing an edge
def ConsecutiveShareEdge (g : Grid) : Prop :=
  ∀ i j k l : Fin 3, g i j = g k l + 1 →
    ((i = k ∧ j.val + 1 = l.val) ∨ (i = k ∧ j.val = l.val + 1) ∨
     (i.val + 1 = k.val ∧ j = l) ∨ (i.val = k.val + 1 ∧ j = l))

-- Define the sum of corner numbers
def CornerSum (g : Grid) : Nat :=
  g 0 0 + g 0 2 + g 2 0 + g 2 2

-- Define the sum of numbers along one diagonal
def DiagonalSum (g : Grid) : Nat :=
  g 0 0 + g 1 1 + g 2 2

-- Theorem statement
theorem center_is_five (g : Grid) 
  (grid_nums : ∀ i j, g i j ∈ Finset.range 9)
  (consecutive_edge : ConsecutiveShareEdge g)
  (corner_sum : CornerSum g = 20)
  (diagonal_sum : DiagonalSum g = 15) :
  g 1 1 = 5 := by
  sorry

end NUMINAMATH_CALUDE_center_is_five_l2473_247344


namespace NUMINAMATH_CALUDE_solve_linear_system_l2473_247314

/-- Given a system of linear equations:
     a + b = c
     b + c = 7
     c - a = 2
    Prove that b = 2 -/
theorem solve_linear_system (a b c : ℤ) 
  (eq1 : a + b = c) 
  (eq2 : b + c = 7) 
  (eq3 : c - a = 2) : 
  b = 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_system_l2473_247314


namespace NUMINAMATH_CALUDE_pace_ratio_l2473_247383

/-- The ratio of a man's pace on a day he was late to his usual pace -/
theorem pace_ratio (usual_time : ℝ) (late_time : ℝ) (h1 : usual_time = 2) 
  (h2 : late_time = usual_time + 1/3) : 
  (usual_time / late_time) = 6/7 := by
  sorry

end NUMINAMATH_CALUDE_pace_ratio_l2473_247383


namespace NUMINAMATH_CALUDE_perimeter_difference_l2473_247303

-- Define the perimeter of the first figure
def perimeter_figure1 : ℕ :=
  -- Outer rectangle perimeter
  2 * (5 + 2) +
  -- Middle vertical rectangle contribution
  2 * 3 +
  -- Inner vertical rectangle contribution
  2 * 2

-- Define the perimeter of the second figure
def perimeter_figure2 : ℕ :=
  -- Outer rectangle perimeter
  2 * (5 + 3) +
  -- Vertical lines contribution
  5 * 2

-- Theorem statement
theorem perimeter_difference : perimeter_figure2 - perimeter_figure1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_difference_l2473_247303


namespace NUMINAMATH_CALUDE_reading_assignment_solution_l2473_247382

/-- Represents the reading assignment for Mrs. Reed's English class -/
structure ReadingAssignment where
  total_pages : ℕ
  alice_speed : ℕ  -- seconds per page
  bob_speed : ℕ    -- seconds per page
  chandra_speed : ℕ -- seconds per page
  x : ℕ  -- last page Alice reads
  y : ℕ  -- last page Chandra reads

/-- Checks if the reading assignment satisfies the given conditions -/
def is_valid_assignment (r : ReadingAssignment) : Prop :=
  r.total_pages = 910 ∧
  r.alice_speed = 30 ∧
  r.bob_speed = 60 ∧
  r.chandra_speed = 45 ∧
  r.x < r.y ∧
  r.y < r.total_pages ∧
  r.alice_speed * r.x = r.chandra_speed * (r.y - r.x) ∧
  r.chandra_speed * (r.y - r.x) = r.bob_speed * (r.total_pages - r.y)

/-- Theorem stating the unique solution for the reading assignment -/
theorem reading_assignment_solution (r : ReadingAssignment) :
  is_valid_assignment r → r.x = 420 ∧ r.y = 700 := by
  sorry


end NUMINAMATH_CALUDE_reading_assignment_solution_l2473_247382


namespace NUMINAMATH_CALUDE_savings_percentage_l2473_247309

theorem savings_percentage (income : ℝ) (savings_rate : ℝ) : 
  savings_rate = 0.35 →
  (2 : ℝ) * (income * (1 - savings_rate)) = 
    income * (1 - savings_rate) + income * (1 - 2 * savings_rate) →
  savings_rate = 0.35 := by
  sorry

end NUMINAMATH_CALUDE_savings_percentage_l2473_247309


namespace NUMINAMATH_CALUDE_election_winner_votes_l2473_247325

theorem election_winner_votes 
  (total_votes : ℕ) 
  (winner_percentage : ℚ) 
  (vote_difference : ℕ) 
  (h1 : winner_percentage = 62 / 100) 
  (h2 : vote_difference = 336) 
  (h3 : ↑total_votes * winner_percentage - ↑total_votes * (1 - winner_percentage) = vote_difference) :
  ↑total_votes * winner_percentage = 868 :=
sorry

end NUMINAMATH_CALUDE_election_winner_votes_l2473_247325


namespace NUMINAMATH_CALUDE_smallest_natural_ending_2012_l2473_247381

theorem smallest_natural_ending_2012 : 
  ∃ (n : ℕ), n = 1716 ∧ 
  (∀ (m : ℕ), m < n → (m * 7) % 10000 ≠ 2012) ∧ 
  (n * 7) % 10000 = 2012 := by
sorry

end NUMINAMATH_CALUDE_smallest_natural_ending_2012_l2473_247381


namespace NUMINAMATH_CALUDE_circular_garden_radius_l2473_247391

/-- 
Given a circular garden with radius r, if the length of the fence (circumference) 
is 1/8 of the area of the garden, then r = 16.
-/
theorem circular_garden_radius (r : ℝ) (h : r > 0) : 
  2 * π * r = (1 / 8) * π * r^2 → r = 16 := by sorry

end NUMINAMATH_CALUDE_circular_garden_radius_l2473_247391


namespace NUMINAMATH_CALUDE_floor_of_5_7_l2473_247350

theorem floor_of_5_7 : ⌊(5.7 : ℝ)⌋ = 5 := by sorry

end NUMINAMATH_CALUDE_floor_of_5_7_l2473_247350


namespace NUMINAMATH_CALUDE_a4b4_value_l2473_247397

theorem a4b4_value (a₁ a₂ a₃ a₄ b₁ b₂ b₃ b₄ : ℝ) 
  (eq1 : a₁ * b₁ + a₂ * b₃ = 1)
  (eq2 : a₁ * b₂ + a₂ * b₄ = 0)
  (eq3 : a₃ * b₁ + a₄ * b₃ = 0)
  (eq4 : a₃ * b₂ + a₄ * b₄ = 1)
  (eq5 : a₂ * b₃ = 7) :
  a₄ * b₄ = -6 := by
  sorry

end NUMINAMATH_CALUDE_a4b4_value_l2473_247397


namespace NUMINAMATH_CALUDE_correct_calculation_l2473_247306

theorem correct_calculation : -5 * (-4) * (-2) * (-2) = 80 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l2473_247306


namespace NUMINAMATH_CALUDE_maglev_train_speed_l2473_247357

/-- Proves that the average speed of a maglev train is 225 km/h given specific conditions --/
theorem maglev_train_speed :
  ∀ (subway_speed : ℝ),
    subway_speed > 0 →
    let maglev_speed := 6.25 * subway_speed
    let distance := 30
    let subway_time := distance / subway_speed
    let maglev_time := distance / maglev_speed
    subway_time - maglev_time = 0.7 →
    maglev_speed = 225 := by
  sorry

#check maglev_train_speed

end NUMINAMATH_CALUDE_maglev_train_speed_l2473_247357


namespace NUMINAMATH_CALUDE_xiaoGang_weight_not_80_grams_l2473_247320

-- Define a person
structure Person where
  name : String
  weight : Float  -- weight in kilograms

-- Define Xiao Gang
def xiaoGang : Person := { name := "Xiao Gang", weight := 80 }

-- Theorem to prove
theorem xiaoGang_weight_not_80_grams : 
  xiaoGang.weight ≠ 0.08 := by sorry

end NUMINAMATH_CALUDE_xiaoGang_weight_not_80_grams_l2473_247320


namespace NUMINAMATH_CALUDE_problem_solution_l2473_247317

theorem problem_solution (x : ℝ) (h : x + Real.sqrt (x^2 - 1) + 1 / (x + Real.sqrt (x^2 - 1)) = 12) :
  x^3 + Real.sqrt (x^6 - 1) + 1 / (x^3 + Real.sqrt (x^6 - 1)) = 432 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2473_247317


namespace NUMINAMATH_CALUDE_equation_impossible_l2473_247363

-- Define the set of digits from 1 to 9
def Digits : Set Nat := {1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Define the property that all variables are distinct
def AllDistinct (K T U Ch O H : Nat) : Prop :=
  K ≠ T ∧ K ≠ U ∧ K ≠ Ch ∧ K ≠ O ∧ K ≠ H ∧
  T ≠ U ∧ T ≠ Ch ∧ T ≠ O ∧ T ≠ H ∧
  U ≠ Ch ∧ U ≠ O ∧ U ≠ H ∧
  Ch ≠ O ∧ Ch ≠ H ∧
  O ≠ H

theorem equation_impossible :
  ∀ (K T U Ch O H : Nat),
    K ∈ Digits → T ∈ Digits → U ∈ Digits → Ch ∈ Digits → O ∈ Digits → H ∈ Digits →
    AllDistinct K T U Ch O H →
    K * 0 * T ≠ U * Ch * O * H * H * U :=
by sorry

end NUMINAMATH_CALUDE_equation_impossible_l2473_247363


namespace NUMINAMATH_CALUDE_spelling_homework_time_l2473_247331

theorem spelling_homework_time (total_time math_time reading_time : ℕ) 
  (h1 : total_time = 60)
  (h2 : math_time = 15)
  (h3 : reading_time = 27) :
  total_time - math_time - reading_time = 18 := by
  sorry

end NUMINAMATH_CALUDE_spelling_homework_time_l2473_247331


namespace NUMINAMATH_CALUDE_intersection_points_l2473_247384

-- Define the functions f and g
def f (t x : ℝ) : ℝ := t * x^2 - x + 1
def g (t x : ℝ) : ℝ := 2 * t * x - 1

-- Define the discriminant function
def discriminant (t : ℝ) : ℝ := (2 * t - 1)^2

-- Theorem statement
theorem intersection_points (t : ℝ) :
  (∃ x : ℝ, f t x = g t x) ∧
  (∀ x y : ℝ, f t x = g t x ∧ f t y = g t y → x = y ∨ (∃ z : ℝ, f t z = g t z ∧ z ≠ x ∧ z ≠ y)) :=
by sorry

end NUMINAMATH_CALUDE_intersection_points_l2473_247384


namespace NUMINAMATH_CALUDE_april_rainfall_calculation_l2473_247388

/-- Given the March rainfall and the difference between March and April rainfall,
    calculate the April rainfall. -/
def april_rainfall (march_rainfall : ℝ) (rainfall_difference : ℝ) : ℝ :=
  march_rainfall - rainfall_difference

/-- Theorem stating that given the specific March rainfall and difference,
    the April rainfall is 0.46 inches. -/
theorem april_rainfall_calculation :
  april_rainfall 0.81 0.35 = 0.46 := by
  sorry

end NUMINAMATH_CALUDE_april_rainfall_calculation_l2473_247388


namespace NUMINAMATH_CALUDE_number_of_bowls_l2473_247343

theorem number_of_bowls (n : ℕ) 
  (h1 : n > 0)  -- There is at least one bowl
  (h2 : 12 ≤ n)  -- There are at least 12 bowls to add grapes to
  (h3 : (96 : ℝ) / n = 6)  -- The average increase is 6
  : n = 16 := by
sorry

end NUMINAMATH_CALUDE_number_of_bowls_l2473_247343


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l2473_247352

theorem decimal_to_fraction : 
  (2.35 : ℚ) = 47 / 20 := by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l2473_247352


namespace NUMINAMATH_CALUDE_classroom_notebooks_l2473_247355

theorem classroom_notebooks (total_students : ℕ) 
  (notebooks_group1 : ℕ) (notebooks_group2 : ℕ) : 
  total_students = 28 → 
  notebooks_group1 = 5 → 
  notebooks_group2 = 3 → 
  (total_students / 2) * notebooks_group1 + (total_students / 2) * notebooks_group2 = 112 := by
  sorry

end NUMINAMATH_CALUDE_classroom_notebooks_l2473_247355


namespace NUMINAMATH_CALUDE_prob_three_correct_is_one_twelfth_l2473_247333

def number_of_houses : ℕ := 5

def probability_three_correct_deliveries : ℚ :=
  (number_of_houses.choose 3 * 1) / number_of_houses.factorial

theorem prob_three_correct_is_one_twelfth :
  probability_three_correct_deliveries = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_correct_is_one_twelfth_l2473_247333


namespace NUMINAMATH_CALUDE_hyperbola_transverse_axis_range_l2473_247387

theorem hyperbola_transverse_axis_range (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (∃ y : ℝ, y^2 / a^2 - (2*y)^2 / b^2 = 1) →
  b^2 = 1 - a^2 →
  0 < 2*a ∧ 2*a < 2*Real.sqrt 5 / 5 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_transverse_axis_range_l2473_247387


namespace NUMINAMATH_CALUDE_inequality_proof_l2473_247346

theorem inequality_proof (a b c : ℝ) (h1 : c > b) (h2 : b > a) (h3 : c + b + a = 0) : b * c > a * c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2473_247346


namespace NUMINAMATH_CALUDE_elevator_max_additional_weight_l2473_247398

/-- The maximum weight a person can have to enter an elevator without overloading it,
    given the current occupants and the elevator's weight limit. -/
def max_additional_weight (adult_count : ℕ) (adult_avg_weight : ℝ)
                          (child_count : ℕ) (child_avg_weight : ℝ)
                          (max_elevator_weight : ℝ) : ℝ :=
  max_elevator_weight - (adult_count * adult_avg_weight + child_count * child_avg_weight)

/-- Theorem stating the maximum weight of the next person to enter the elevator
    without overloading it, given the specific conditions. -/
theorem elevator_max_additional_weight :
  max_additional_weight 3 140 2 64 600 = 52 := by
  sorry

end NUMINAMATH_CALUDE_elevator_max_additional_weight_l2473_247398


namespace NUMINAMATH_CALUDE_stratified_sampling_seniors_l2473_247390

/-- Represents the number of students in each grade -/
structure GradePopulation where
  freshmen : ℕ
  sophomores : ℕ
  seniors : ℕ

/-- Calculates the total number of students -/
def totalStudents (pop : GradePopulation) : ℕ :=
  pop.freshmen + pop.sophomores + pop.seniors

/-- Calculates the number of students to sample from each grade -/
def stratifiedSample (pop : GradePopulation) (sampleSize : ℕ) : GradePopulation :=
  let total := totalStudents pop
  let factor := sampleSize / total
  { freshmen := pop.freshmen * factor,
    sophomores := pop.sophomores * factor,
    seniors := pop.seniors * factor }

theorem stratified_sampling_seniors
  (pop : GradePopulation)
  (h1 : pop.freshmen = 520)
  (h2 : pop.sophomores = 500)
  (h3 : pop.seniors = 580)
  (h4 : totalStudents pop = 1600)
  (sampleSize : ℕ)
  (h5 : sampleSize = 80) :
  (stratifiedSample pop sampleSize).seniors = 29 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_seniors_l2473_247390


namespace NUMINAMATH_CALUDE_price_increase_percentage_l2473_247305

theorem price_increase_percentage (original_price new_price : ℝ) 
  (h1 : original_price = 300)
  (h2 : new_price = 420) :
  (new_price - original_price) / original_price * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_price_increase_percentage_l2473_247305


namespace NUMINAMATH_CALUDE_product_of_five_integers_l2473_247399

theorem product_of_five_integers (E F G H I : ℕ) : 
  E > 0 → F > 0 → G > 0 → H > 0 → I > 0 →
  E + F + G + H + I = 80 →
  E + 2 = F - 2 →
  E + 2 = G * 2 →
  E + 2 = H * 3 →
  E + 2 = I / 2 →
  E * F * G * H * I = 5120000 / 81 := by
sorry

end NUMINAMATH_CALUDE_product_of_five_integers_l2473_247399


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_eccentricity_l2473_247310

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A conic section (ellipse or hyperbola) -/
structure Conic where
  center : Point
  a : ℝ  -- semi-major axis
  b : ℝ  -- semi-minor axis
  isEllipse : Bool

/-- The eccentricity of a conic section -/
def eccentricity (c : Conic) : ℝ :=
  sorry

/-- The foci of a conic section -/
def foci (c : Conic) : (Point × Point) :=
  sorry

/-- The angle between three points -/
def angle (p1 p2 p3 : Point) : ℝ :=
  sorry

/-- The intersection points of two conic sections -/
def intersection (c1 c2 : Conic) : Set Point :=
  sorry

theorem ellipse_hyperbola_eccentricity 
  (C₁ : Conic) (C₂ : Conic) (F₁ F₂ P : Point) :
  C₁.isEllipse = true →
  C₂.isEllipse = false →
  foci C₁ = (F₁, F₂) →
  foci C₂ = (F₁, F₂) →
  P ∈ intersection C₁ C₂ →
  P.x > 0 ∧ P.y > 0 →
  eccentricity C₁ * eccentricity C₂ = 1 →
  angle F₁ P F₂ = π / 3 →
  eccentricity C₁ = Real.sqrt 3 / 3 :=
sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_eccentricity_l2473_247310


namespace NUMINAMATH_CALUDE_milk_consumption_l2473_247369

theorem milk_consumption (initial_milk : ℚ) (rachel_fraction : ℚ) (monica_fraction : ℚ) : 
  initial_milk = 3/4 →
  rachel_fraction = 1/2 →
  monica_fraction = 1/3 →
  let rachel_consumption := rachel_fraction * initial_milk
  let remaining_milk := initial_milk - rachel_consumption
  let monica_consumption := monica_fraction * remaining_milk
  rachel_consumption + monica_consumption = 1/2 := by
sorry

end NUMINAMATH_CALUDE_milk_consumption_l2473_247369


namespace NUMINAMATH_CALUDE_cistern_length_is_eight_l2473_247379

/-- Represents a rectangular cistern with water --/
structure Cistern where
  length : ℝ
  width : ℝ
  depth : ℝ
  wetSurfaceArea : ℝ

/-- Calculates the wet surface area of a cistern --/
def wetSurfaceArea (c : Cistern) : ℝ :=
  c.length * c.width + 2 * c.length * c.depth + 2 * c.width * c.depth

/-- Theorem stating that a cistern with given dimensions has a length of 8 meters --/
theorem cistern_length_is_eight (c : Cistern) 
    (h1 : c.width = 4)
    (h2 : c.depth = 1.25)
    (h3 : c.wetSurfaceArea = 62)
    (h4 : wetSurfaceArea c = c.wetSurfaceArea) : 
    c.length = 8 := by
  sorry


end NUMINAMATH_CALUDE_cistern_length_is_eight_l2473_247379


namespace NUMINAMATH_CALUDE_point_not_on_graph_l2473_247356

theorem point_not_on_graph : ¬ ∃ (y : ℝ), y = (-2 - 1) / (-2 + 2) ∧ y = 1 := by sorry

end NUMINAMATH_CALUDE_point_not_on_graph_l2473_247356


namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l2473_247385

theorem unique_solution_for_equation (x y : ℝ) :
  (x - 14)^2 + (y - 15)^2 + (x - y)^2 = 1/3 ↔ x = 14 + 1/3 ∧ y = 14 + 2/3 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_equation_l2473_247385


namespace NUMINAMATH_CALUDE_triangle_theorem_l2473_247372

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  2 * Real.cos (2 * t.A) + 4 * Real.cos (t.B + t.C) + 3 = 0 ∧
  t.a = Real.sqrt 3 ∧
  t.b + t.c = 3

-- Theorem statement
theorem triangle_theorem (t : Triangle) (h : triangle_conditions t) :
  t.A = π / 3 ∧ ((t.b = 2 ∧ t.c = 1) ∨ (t.b = 1 ∧ t.c = 2)) := by
  sorry

end NUMINAMATH_CALUDE_triangle_theorem_l2473_247372
