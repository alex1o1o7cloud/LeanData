import Mathlib

namespace NUMINAMATH_CALUDE_basketball_game_students_l1062_106254

/-- The total number of students in a basketball game given the number of 5th graders and a ratio of 6th to 5th graders -/
def total_students (fifth_graders : ℕ) (ratio : ℕ) : ℕ :=
  fifth_graders + ratio * fifth_graders

/-- Theorem stating that given 12 5th graders and 6 times as many 6th graders, the total number of students is 84 -/
theorem basketball_game_students :
  total_students 12 6 = 84 := by
  sorry

end NUMINAMATH_CALUDE_basketball_game_students_l1062_106254


namespace NUMINAMATH_CALUDE_factorization_problem1_factorization_problem2_factorization_problem3_factorization_problem4_l1062_106248

-- Problem 1
theorem factorization_problem1 (x y : ℝ) : 
  8 * x^2 + 26 * x * y - 15 * y^2 = (2 * x - y) * (4 * x + 15 * y) := by sorry

-- Problem 2
theorem factorization_problem2 (x y : ℝ) : 
  x^6 - y^6 - 2 * x^3 + 1 = (x^3 - y^3 - 1) * (x^3 + y^3 - 1) := by sorry

-- Problem 3
theorem factorization_problem3 (a b c : ℝ) : 
  a^3 + a^2 * c + b^2 * c - a * b * c + b^3 = (a + b + c) * (a^2 - a * b + b^2) := by sorry

-- Problem 4
theorem factorization_problem4 (x : ℝ) : 
  x^3 - 11 * x^2 + 31 * x - 21 = (x - 1) * (x - 3) * (x - 7) := by sorry

end NUMINAMATH_CALUDE_factorization_problem1_factorization_problem2_factorization_problem3_factorization_problem4_l1062_106248


namespace NUMINAMATH_CALUDE_range_of_a_l1062_106272

theorem range_of_a (a b c : ℝ) 
  (sum_eq : a + b + c = 2)
  (sum_sq_eq : a^2 + b^2 + c^2 = 4)
  (order : a > b ∧ b > c) :
  a ∈ Set.Ioo (2/3) 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1062_106272


namespace NUMINAMATH_CALUDE_opposite_number_theorem_l1062_106267

theorem opposite_number_theorem (a b c : ℝ) : 
  -((-a + b - c)) = c - a - b := by sorry

end NUMINAMATH_CALUDE_opposite_number_theorem_l1062_106267


namespace NUMINAMATH_CALUDE_bottom_face_points_l1062_106252

/-- Represents the number of points on each face of a cube -/
structure CubePoints where
  front : ℕ
  back : ℕ
  left : ℕ
  right : ℕ
  top : ℕ
  bottom : ℕ

/-- Theorem stating the number of points on the bottom face of the cube -/
theorem bottom_face_points (c : CubePoints) 
  (opposite_sum : c.front + c.back = 13 ∧ c.left + c.right = 13 ∧ c.top + c.bottom = 13)
  (front_left_top_sum : c.front + c.left + c.top = 16)
  (top_right_back_sum : c.top + c.right + c.back = 24) :
  c.bottom = 6 := by
  sorry

end NUMINAMATH_CALUDE_bottom_face_points_l1062_106252


namespace NUMINAMATH_CALUDE_paper_pieces_l1062_106211

/-- The number of pieces of paper after n tears -/
def num_pieces (n : ℕ) : ℕ := 3 * n + 1

/-- Theorem stating the number of pieces after n tears -/
theorem paper_pieces (n : ℕ) : 
  (∀ k : ℕ, k ≤ n → num_pieces k = num_pieces (k - 1) + 3) → 
  num_pieces n = 3 * n + 1 :=
by sorry

end NUMINAMATH_CALUDE_paper_pieces_l1062_106211


namespace NUMINAMATH_CALUDE_smallest_equal_packs_l1062_106289

theorem smallest_equal_packs (pencil_pack : Nat) (eraser_pack : Nat) : 
  pencil_pack = 5 → eraser_pack = 7 → 
  (∃ n : Nat, n > 0 ∧ ∃ m : Nat, n * eraser_pack = m * pencil_pack ∧ 
  ∀ k : Nat, k > 0 → k * eraser_pack = m * pencil_pack → n ≤ k) → n = 5 := by
sorry

end NUMINAMATH_CALUDE_smallest_equal_packs_l1062_106289


namespace NUMINAMATH_CALUDE_dolphin_training_hours_l1062_106206

theorem dolphin_training_hours (num_dolphins : ℕ) (training_hours_per_dolphin : ℕ) (num_trainers : ℕ) 
  (h1 : num_dolphins = 12)
  (h2 : training_hours_per_dolphin = 5)
  (h3 : num_trainers = 4)
  (h4 : num_trainers > 0) :
  (num_dolphins * training_hours_per_dolphin) / num_trainers = 15 := by
sorry

end NUMINAMATH_CALUDE_dolphin_training_hours_l1062_106206


namespace NUMINAMATH_CALUDE_john_net_profit_l1062_106265

def gross_income : ℝ := 30000
def car_purchase_price : ℝ := 20000
def monthly_maintenance_cost : ℝ := 300
def annual_insurance_cost : ℝ := 1200
def tire_replacement_cost : ℝ := 400
def car_trade_in_value : ℝ := 6000
def tax_rate : ℝ := 0.15

def total_maintenance_cost : ℝ := monthly_maintenance_cost * 12
def car_depreciation : ℝ := car_purchase_price - car_trade_in_value
def total_expenses : ℝ := total_maintenance_cost + annual_insurance_cost + tire_replacement_cost + car_depreciation
def taxes : ℝ := tax_rate * gross_income
def net_profit : ℝ := gross_income - total_expenses - taxes

theorem john_net_profit : net_profit = 6300 := by
  sorry

end NUMINAMATH_CALUDE_john_net_profit_l1062_106265


namespace NUMINAMATH_CALUDE_cubic_polynomial_problem_l1062_106216

/-- Given a cubic equation and a polynomial P satisfying certain conditions, 
    prove that P has a specific form. -/
theorem cubic_polynomial_problem (a b c : ℝ) (P : ℝ → ℝ) : 
  (∀ x, x^3 - 4*x^2 + x - 1 = 0 ↔ (x = a ∨ x = b ∨ x = c)) →
  (∃ p q r s : ℝ, ∀ x, P x = p*x^3 + q*x^2 + r*x + s) →
  P a = b + c →
  P b = a + c →
  P c = a + b →
  P (a + b + c) = -20 →
  ∀ x, P x = (-20*x^3 + 80*x^2 - 23*x + 32) / 3 := by
sorry

end NUMINAMATH_CALUDE_cubic_polynomial_problem_l1062_106216


namespace NUMINAMATH_CALUDE_goods_train_speed_l1062_106208

/-- The speed of a goods train passing a woman in an opposite moving train -/
theorem goods_train_speed
  (woman_train_speed : ℝ)
  (passing_time : ℝ)
  (goods_train_length : ℝ)
  (h1 : woman_train_speed = 25)
  (h2 : passing_time = 3)
  (h3 : goods_train_length = 140) :
  ∃ (goods_train_speed : ℝ),
    goods_train_speed = 143 ∧
    (goods_train_length / passing_time) * 3.6 = woman_train_speed + goods_train_speed :=
by sorry

end NUMINAMATH_CALUDE_goods_train_speed_l1062_106208


namespace NUMINAMATH_CALUDE_initial_men_count_l1062_106251

/-- The number of days it takes for the initial group to complete the work -/
def initial_days : ℕ := 70

/-- The number of days it takes for 40 men to complete the work -/
def new_days : ℕ := 63

/-- The number of men in the new group -/
def new_men : ℕ := 40

/-- The amount of work is constant and can be represented as men * days -/
axiom work_constant (m1 m2 : ℕ) (d1 d2 : ℕ) : m1 * d1 = m2 * d2

/-- The theorem to be proved -/
theorem initial_men_count : ∃ x : ℕ, x * initial_days = new_men * new_days ∧ x = 36 := by
  sorry

end NUMINAMATH_CALUDE_initial_men_count_l1062_106251


namespace NUMINAMATH_CALUDE_park_trees_after_planting_l1062_106201

/-- The total number of dogwood trees after planting -/
def total_trees (current : ℕ) (planted_today : ℕ) (planted_tomorrow : ℕ) : ℕ :=
  current + planted_today + planted_tomorrow

/-- Theorem: The park will have 100 dogwood trees when the workers are finished -/
theorem park_trees_after_planting :
  total_trees 39 41 20 = 100 := by
  sorry

end NUMINAMATH_CALUDE_park_trees_after_planting_l1062_106201


namespace NUMINAMATH_CALUDE_hyperbola_equation_and_minimum_distance_l1062_106209

structure Hyperbola where
  a : ℝ
  b : ℝ
  h_positive : a > 0 ∧ b > 0

def on_hyperbola (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / h.a^2 - y^2 / h.b^2 = 1

def asymptotic_equation (h : Hyperbola) : Prop :=
  h.b = Real.sqrt 3 * h.a

def point_on_hyperbola (h : Hyperbola) : Prop :=
  on_hyperbola h (Real.sqrt 5) (Real.sqrt 3)

def perpendicular_vectors (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ * x₂ + y₁ * y₂ = 0

theorem hyperbola_equation_and_minimum_distance 
  (h : Hyperbola) 
  (h_asymptotic : asymptotic_equation h)
  (h_point : point_on_hyperbola h) :
  (h.a = 2 ∧ h.b = 2 * Real.sqrt 3) ∧
  (∀ (x₁ y₁ x₂ y₂ : ℝ), 
    on_hyperbola h x₁ y₁ → 
    on_hyperbola h x₂ y₂ → 
    perpendicular_vectors x₁ y₁ x₂ y₂ → 
    x₁^2 + y₁^2 + x₂^2 + y₂^2 ≥ 24) :=
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_and_minimum_distance_l1062_106209


namespace NUMINAMATH_CALUDE_largest_quantity_l1062_106233

theorem largest_quantity (a b c d e : ℝ) 
  (eq1 : a = b + 3)
  (eq2 : b = c - 4)
  (eq3 : c = d + 5)
  (eq4 : d = e - 6) :
  e ≥ a ∧ e ≥ b ∧ e ≥ c ∧ e ≥ d := by
  sorry

end NUMINAMATH_CALUDE_largest_quantity_l1062_106233


namespace NUMINAMATH_CALUDE_tower_of_threes_greater_than_tower_of_twos_l1062_106229

-- Define a function to represent the tower of exponents
def tower (base : ℕ) (height : ℕ) : ℕ :=
  match height with
  | 0 => 1
  | n + 1 => base ^ (tower base n)

-- State the theorem
theorem tower_of_threes_greater_than_tower_of_twos :
  tower 3 99 > tower 2 100 :=
sorry

end NUMINAMATH_CALUDE_tower_of_threes_greater_than_tower_of_twos_l1062_106229


namespace NUMINAMATH_CALUDE_simplify_fraction_multiplication_l1062_106276

theorem simplify_fraction_multiplication : (405 : ℚ) / 1215 * 27 = 9 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_multiplication_l1062_106276


namespace NUMINAMATH_CALUDE_triangle_acute_if_tan_product_positive_l1062_106286

/-- Given a triangle ABC with internal angles A, B, and C, 
    if the product of their tangents is positive, 
    then the triangle is acute. -/
theorem triangle_acute_if_tan_product_positive 
  (A B C : Real) 
  (h_triangle : A + B + C = π) 
  (h_positive : Real.tan A * Real.tan B * Real.tan C > 0) : 
  A < π/2 ∧ B < π/2 ∧ C < π/2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_acute_if_tan_product_positive_l1062_106286


namespace NUMINAMATH_CALUDE_ratio_problem_l1062_106296

theorem ratio_problem (a b c d : ℚ) 
  (hab : a / b = 5 / 4)
  (hcd : c / d = 4 / 3)
  (hdb : d / b = 1 / 5) :
  a / c = 75 / 16 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l1062_106296


namespace NUMINAMATH_CALUDE_unique_two_digit_number_mod_13_l1062_106207

theorem unique_two_digit_number_mod_13 :
  ∃! n : ℕ, 10 ≤ n ∧ n < 100 ∧ (13 * n) % 100 = 42 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_two_digit_number_mod_13_l1062_106207


namespace NUMINAMATH_CALUDE_slices_per_pizza_l1062_106256

theorem slices_per_pizza (total_pizzas : ℕ) (total_slices : ℕ) (h1 : total_pizzas = 7) (h2 : total_slices = 14) :
  total_slices / total_pizzas = 2 := by
  sorry

end NUMINAMATH_CALUDE_slices_per_pizza_l1062_106256


namespace NUMINAMATH_CALUDE_number_of_girls_in_class_l1062_106273

theorem number_of_girls_in_class (total_students : ℕ) (girls_ratio : ℚ) : 
  total_students = 35 →
  girls_ratio = 0.4 →
  ∃ (boys girls : ℕ), 
    boys + girls = total_students ∧ 
    girls = (girls_ratio * boys).floor ∧
    girls = 10 := by
  sorry

end NUMINAMATH_CALUDE_number_of_girls_in_class_l1062_106273


namespace NUMINAMATH_CALUDE_refrigerator_savings_l1062_106293

def cash_price : ℕ := 8000
def deposit : ℕ := 3000
def num_installments : ℕ := 30
def installment_amount : ℕ := 300

theorem refrigerator_savings : 
  deposit + num_installments * installment_amount - cash_price = 4000 := by
  sorry

end NUMINAMATH_CALUDE_refrigerator_savings_l1062_106293


namespace NUMINAMATH_CALUDE_smallest_solution_quartic_equation_l1062_106225

theorem smallest_solution_quartic_equation :
  let f : ℝ → ℝ := λ x => x^4 - 40*x^2 + 144
  ∃ (x : ℝ), f x = 0 ∧ (∀ y : ℝ, f y = 0 → x ≤ y) ∧ x = -6 :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_quartic_equation_l1062_106225


namespace NUMINAMATH_CALUDE_ratio_of_sums_l1062_106231

theorem ratio_of_sums (p q r u v w : ℝ) 
  (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) (hu : 0 < u) (hv : 0 < v) (hw : 0 < w)
  (h1 : p^2 + q^2 + r^2 = 49)
  (h2 : u^2 + v^2 + w^2 = 64)
  (h3 : p*u + q*v + r*w = 56) :
  (p + q + r) / (u + v + w) = 7/8 := by
sorry

end NUMINAMATH_CALUDE_ratio_of_sums_l1062_106231


namespace NUMINAMATH_CALUDE_cricketer_wickets_before_match_l1062_106217

/-- Represents a cricketer's bowling statistics -/
structure CricketerStats where
  wickets : ℕ
  runs : ℕ
  avg : ℚ

/-- Calculates the new average after a match -/
def newAverage (stats : CricketerStats) (newWickets : ℕ) (newRuns : ℕ) : ℚ :=
  (stats.runs + newRuns) / (stats.wickets + newWickets)

theorem cricketer_wickets_before_match 
  (stats : CricketerStats)
  (h1 : stats.avg = 12.4)
  (h2 : newAverage stats 5 26 = 12) :
  stats.wickets = 85 := by
  sorry

end NUMINAMATH_CALUDE_cricketer_wickets_before_match_l1062_106217


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l1062_106235

/-- A rectangle with given area and width has a specific perimeter -/
theorem rectangle_perimeter (area width : ℝ) (h_area : area = 200) (h_width : width = 10) :
  2 * (area / width + width) = 60 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l1062_106235


namespace NUMINAMATH_CALUDE_remainder_of_quadratic_l1062_106249

theorem remainder_of_quadratic (a : ℤ) : 
  let n : ℤ := 40 * a + 2
  (n^2 - 3*n + 5) % 40 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_quadratic_l1062_106249


namespace NUMINAMATH_CALUDE_watch_cost_price_l1062_106257

theorem watch_cost_price (loss_percent : ℚ) (gain_percent : ℚ) (price_difference : ℚ) :
  loss_percent = 16 →
  gain_percent = 4 →
  price_difference = 140 →
  ∃ (cost_price : ℚ),
    (cost_price * (1 - loss_percent / 100)) + price_difference = cost_price * (1 + gain_percent / 100) ∧
    cost_price = 700 :=
by sorry

end NUMINAMATH_CALUDE_watch_cost_price_l1062_106257


namespace NUMINAMATH_CALUDE_max_sum_of_squares_l1062_106274

/-- The distance from a real number to the nearest integer -/
noncomputable def distToNearestInt (x : ℝ) : ℝ := min (x - ⌊x⌋) (⌈x⌉ - x)

/-- The sum of squares of x * (distance of x to nearest integer) -/
noncomputable def sumOfSquares (xs : Finset ℝ) : ℝ :=
  Finset.sum xs (λ x => (x * distToNearestInt x)^2)

/-- The maximum value of the sum of squares given the constraints -/
theorem max_sum_of_squares (n : ℕ) :
  ∃ (xs : Finset ℝ),
    (∀ x ∈ xs, 0 ≤ x) ∧
    (Finset.sum xs id = n) ∧
    (Finset.card xs = n) ∧
    (∀ ys : Finset ℝ,
      (∀ y ∈ ys, 0 ≤ y) →
      (Finset.sum ys id = n) →
      (Finset.card ys = n) →
      sumOfSquares ys ≤ sumOfSquares xs) ∧
    (sumOfSquares xs = (n^2 - n + 1/2) / 4) := by
  sorry

end NUMINAMATH_CALUDE_max_sum_of_squares_l1062_106274


namespace NUMINAMATH_CALUDE_min_value_quadratic_min_value_achieved_min_value_points_l1062_106240

theorem min_value_quadratic (x y : ℝ) : 2*x^2 + 2*y^2 - 8*x + 6*y + 28 ≥ 10.5 :=
by sorry

theorem min_value_achieved : ∃ (x y : ℝ), 2*x^2 + 2*y^2 - 8*x + 6*y + 28 = 10.5 :=
by sorry

theorem min_value_points : 2*2^2 + 2*(-3/2)^2 - 8*2 + 6*(-3/2) + 28 = 10.5 :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_min_value_achieved_min_value_points_l1062_106240


namespace NUMINAMATH_CALUDE_smallest_sum_of_four_consecutive_primes_divisible_by_four_l1062_106242

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def consecutive_primes (p₁ p₂ p₃ p₄ : ℕ) : Prop :=
  is_prime p₁ ∧ is_prime p₂ ∧ is_prime p₃ ∧ is_prime p₄ ∧
  ∀ q : ℕ, (is_prime q ∧ p₁ < q ∧ q < p₄) → (q = p₂ ∨ q = p₃)

theorem smallest_sum_of_four_consecutive_primes_divisible_by_four :
  ∃ p₁ p₂ p₃ p₄ : ℕ,
    consecutive_primes p₁ p₂ p₃ p₄ ∧
    (p₁ + p₂ + p₃ + p₄) % 4 = 0 ∧
    p₁ + p₂ + p₃ + p₄ = 36 ∧
    ∀ q₁ q₂ q₃ q₄ : ℕ,
      consecutive_primes q₁ q₂ q₃ q₄ →
      (q₁ + q₂ + q₃ + q₄) % 4 = 0 →
      q₁ + q₂ + q₃ + q₄ ≥ 36 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_four_consecutive_primes_divisible_by_four_l1062_106242


namespace NUMINAMATH_CALUDE_x_intercept_of_line_l1062_106205

/-- The x-intercept of the line 4x + 7y = 28 is (7, 0) -/
theorem x_intercept_of_line (x y : ℚ) : 4 * x + 7 * y = 28 → y = 0 → x = 7 := by
  sorry

end NUMINAMATH_CALUDE_x_intercept_of_line_l1062_106205


namespace NUMINAMATH_CALUDE_third_side_length_valid_l1062_106266

theorem third_side_length_valid (a b c : ℝ) : 
  a = 2 → b = 4 → c = 4 → 
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a) := by
  sorry

end NUMINAMATH_CALUDE_third_side_length_valid_l1062_106266


namespace NUMINAMATH_CALUDE_arithmetic_progression_same_digit_sum_l1062_106281

/-- Sum of digits of a natural number -/
def digitSum (n : ℕ) : ℕ := sorry

/-- Arithmetic progression with first term a and common difference d -/
def arithmeticProgression (a d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d

theorem arithmetic_progression_same_digit_sum (a d : ℕ) :
  ∃ m n : ℕ, m ≠ n ∧ 
    digitSum (arithmeticProgression a d m) = digitSum (arithmeticProgression a d n) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_same_digit_sum_l1062_106281


namespace NUMINAMATH_CALUDE_initial_term_range_l1062_106220

/-- A strictly increasing sequence satisfying the given recursive formula -/
def StrictlyIncreasingSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a (n + 1) > a n) ∧ 
  (∀ n, a (n + 1) = (4 * a n - 2) / (a n + 1))

/-- The theorem stating that the initial term of the sequence must be in (1, 2) -/
theorem initial_term_range (a : ℕ → ℝ) :
  StrictlyIncreasingSequence a → 1 < a 1 ∧ a 1 < 2 := by
  sorry


end NUMINAMATH_CALUDE_initial_term_range_l1062_106220


namespace NUMINAMATH_CALUDE_tree_planting_theorem_l1062_106278

/-- The number of trees planted by 4th graders -/
def trees_4th : ℕ := 30

/-- The number of trees planted by 5th graders -/
def trees_5th : ℕ := 2 * trees_4th

/-- The number of trees planted by 6th graders -/
def trees_6th : ℕ := 3 * trees_5th - 30

/-- The total number of trees planted by all grades -/
def total_trees : ℕ := trees_4th + trees_5th + trees_6th

theorem tree_planting_theorem : total_trees = 240 := by
  sorry

end NUMINAMATH_CALUDE_tree_planting_theorem_l1062_106278


namespace NUMINAMATH_CALUDE_f_geq_f1_iff_a_in_range_l1062_106283

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x < 1 then (1/3) * x^3 - a * x + 1
  else if x ≥ 1 then a * Real.log x
  else 0  -- This case should never occur in our problem, but Lean requires it for completeness

-- State the theorem
theorem f_geq_f1_iff_a_in_range (a : ℝ) :
  (∀ x : ℝ, x ≥ 0 → f a x ≥ f a 1) ↔ (0 < a ∧ a ≤ 4/3) :=
by sorry

end NUMINAMATH_CALUDE_f_geq_f1_iff_a_in_range_l1062_106283


namespace NUMINAMATH_CALUDE_sum_of_squares_l1062_106234

/-- A structure representing a set of four-digit numbers formed from four distinct digits. -/
structure FourDigitSet where
  digits : Finset Nat
  first_number : Nat
  second_last_number : Nat
  (digit_count : digits.card = 4)
  (distinct_digits : ∀ d ∈ digits, d < 10)
  (number_count : (digits.powerset.filter (λ s : Finset Nat => s.card = 4)).card = 18)
  (ascending_order : first_number < second_last_number)
  (first_is_square : ∃ n : Nat, first_number = n ^ 2)
  (second_last_is_square : ∃ n : Nat, second_last_number = n ^ 2)

/-- The theorem stating that the sum of the first and second-last numbers is 10890. -/
theorem sum_of_squares (s : FourDigitSet) : s.first_number + s.second_last_number = 10890 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l1062_106234


namespace NUMINAMATH_CALUDE_min_dimension_sum_for_2310_volume_l1062_106221

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ+
  width : ℕ+
  height : ℕ+

/-- The volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℕ := d.length * d.width * d.height

/-- The sum of the dimensions of a box -/
def dimensionSum (d : BoxDimensions) : ℕ := d.length + d.width + d.height

/-- Theorem stating that the minimum sum of dimensions for a box with volume 2310 is 42 -/
theorem min_dimension_sum_for_2310_volume :
  (∃ (d : BoxDimensions), boxVolume d = 2310) →
  (∀ (d : BoxDimensions), boxVolume d = 2310 → dimensionSum d ≥ 42) ∧
  (∃ (d : BoxDimensions), boxVolume d = 2310 ∧ dimensionSum d = 42) :=
by sorry

end NUMINAMATH_CALUDE_min_dimension_sum_for_2310_volume_l1062_106221


namespace NUMINAMATH_CALUDE_vietnam_2007_solution_l1062_106202

open Real

/-- The functional equation from the 2007 Vietnam Mathematical Olympiad -/
def functional_equation (f : ℝ → ℝ) (b : ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f x * 3^(b + f y - 1) + b^x * 3^(b^3 + f y - 1) - b^(x + y)

/-- The theorem statement for the 2007 Vietnam Mathematical Olympiad problem -/
theorem vietnam_2007_solution (b : ℝ) (hb : b > 0) :
  ∀ f : ℝ → ℝ, functional_equation f b ↔ (∀ x, f x = -b^x) ∨ (∀ x, f x = 1 - b^x) :=
sorry

end NUMINAMATH_CALUDE_vietnam_2007_solution_l1062_106202


namespace NUMINAMATH_CALUDE_expression_evaluation_l1062_106299

theorem expression_evaluation :
  let x : ℕ := 3
  (x + x * x^(x^2)) * 3 = 177156 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1062_106299


namespace NUMINAMATH_CALUDE_quadratic_solution_set_l1062_106269

/-- Given a quadratic function f(x) = x^2 + bx + c, 
    this theorem states that if the solution set of f(x) < 0 
    is the open interval (1, 3), then b + c = -1. -/
theorem quadratic_solution_set (b c : ℝ) : 
  ({x : ℝ | x^2 + b*x + c < 0} = {x : ℝ | 1 < x ∧ x < 3}) → 
  b + c = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_set_l1062_106269


namespace NUMINAMATH_CALUDE_prime_divisor_problem_l1062_106238

theorem prime_divisor_problem (p : ℕ) (h_prime : Nat.Prime p) : 
  (∃ k : ℕ, 635 = 7 * k * p + 11) → p = 89 := by sorry

end NUMINAMATH_CALUDE_prime_divisor_problem_l1062_106238


namespace NUMINAMATH_CALUDE_dogs_left_over_l1062_106264

theorem dogs_left_over (total_dogs : ℕ) (num_houses : ℕ) (h1 : total_dogs = 50) (h2 : num_houses = 17) : 
  total_dogs - (num_houses * (total_dogs / num_houses)) = 16 := by
sorry

end NUMINAMATH_CALUDE_dogs_left_over_l1062_106264


namespace NUMINAMATH_CALUDE_jenn_savings_problem_l1062_106232

/-- Given information about Jenn's savings for a bike purchase --/
theorem jenn_savings_problem (num_jars : ℕ) (bike_cost : ℕ) (leftover : ℕ) :
  num_jars = 5 →
  bike_cost = 180 →
  leftover = 20 →
  (∃ (quarters_per_jar : ℕ),
    quarters_per_jar * num_jars = (bike_cost + leftover) * 4 ∧
    quarters_per_jar = 160) :=
by sorry

end NUMINAMATH_CALUDE_jenn_savings_problem_l1062_106232


namespace NUMINAMATH_CALUDE_jerry_logs_count_l1062_106290

/-- The number of logs Jerry gets from cutting trees -/
def total_logs : ℕ :=
  let pine_logs_per_tree : ℕ := 80
  let maple_logs_per_tree : ℕ := 60
  let walnut_logs_per_tree : ℕ := 100
  let pine_trees_cut : ℕ := 8
  let maple_trees_cut : ℕ := 3
  let walnut_trees_cut : ℕ := 4
  pine_logs_per_tree * pine_trees_cut +
  maple_logs_per_tree * maple_trees_cut +
  walnut_logs_per_tree * walnut_trees_cut

theorem jerry_logs_count : total_logs = 1220 := by
  sorry

end NUMINAMATH_CALUDE_jerry_logs_count_l1062_106290


namespace NUMINAMATH_CALUDE_condition_relation_l1062_106244

theorem condition_relation (a : ℝ) :
  (∀ a, a > 0 → a^2 + a ≥ 0) ∧
  (∃ a, a^2 + a ≥ 0 ∧ ¬(a > 0)) :=
by sorry

end NUMINAMATH_CALUDE_condition_relation_l1062_106244


namespace NUMINAMATH_CALUDE_five_year_compound_interest_l1062_106262

/-- Calculates the final amount after compound interest --/
def compound_interest (m : ℝ) (a : ℝ) (n : ℕ) : ℝ :=
  m * (1 + a) ^ n

/-- Theorem: After 5 years of compound interest, the final amount is m(1+a)^5 --/
theorem five_year_compound_interest (m : ℝ) (a : ℝ) :
  compound_interest m a 5 = m * (1 + a) ^ 5 :=
by
  sorry

end NUMINAMATH_CALUDE_five_year_compound_interest_l1062_106262


namespace NUMINAMATH_CALUDE_number_divided_by_002_equals_50_l1062_106279

theorem number_divided_by_002_equals_50 :
  ∃ x : ℝ, x / 0.02 = 50 ∧ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_number_divided_by_002_equals_50_l1062_106279


namespace NUMINAMATH_CALUDE_two_thirds_of_number_l1062_106213

theorem two_thirds_of_number (y : ℝ) : (2 / 3) * y = 40 → y = 60 := by
  sorry

end NUMINAMATH_CALUDE_two_thirds_of_number_l1062_106213


namespace NUMINAMATH_CALUDE_equation_roots_imply_a_range_l1062_106285

open Real

theorem equation_roots_imply_a_range (m : ℝ) (a : ℝ) (e : ℝ) :
  m > 0 →
  e > 0 →
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
    x₁ + a * (2 * x₁ + 2 * m - 4 * e * x₁) * (log (x₁ + m) - log x₁) = 0 ∧
    x₂ + a * (2 * x₂ + 2 * m - 4 * e * x₂) * (log (x₂ + m) - log x₂) = 0) →
  a > 1 / (2 * e) :=
by sorry

end NUMINAMATH_CALUDE_equation_roots_imply_a_range_l1062_106285


namespace NUMINAMATH_CALUDE_solutions_of_equation_l1062_106277

theorem solutions_of_equation (x : ℝ) : x * (x - 1) = x ↔ x = 0 ∨ x = 2 := by sorry

end NUMINAMATH_CALUDE_solutions_of_equation_l1062_106277


namespace NUMINAMATH_CALUDE_book_sale_loss_percentage_l1062_106282

/-- Given two books with specified costs and selling conditions, prove the loss percentage on the first book. -/
theorem book_sale_loss_percentage
  (total_cost : ℝ)
  (cost_book1 : ℝ)
  (gain_percentage : ℝ)
  (h_total_cost : total_cost = 480)
  (h_cost_book1 : cost_book1 = 280)
  (h_gain_percentage : gain_percentage = 19)
  (h_same_selling_price : ∃ (selling_price : ℝ),
    selling_price = cost_book1 * (1 - (loss_percentage / 100)) ∧
    selling_price = (total_cost - cost_book1) * (1 + (gain_percentage / 100)))
  : ∃ (loss_percentage : ℝ), loss_percentage = 15 := by
  sorry


end NUMINAMATH_CALUDE_book_sale_loss_percentage_l1062_106282


namespace NUMINAMATH_CALUDE_rational_cubic_polynomial_existence_l1062_106287

theorem rational_cubic_polynomial_existence :
  ∃ (b c d : ℚ), 
    let P := fun (x : ℚ) => x^3 + b*x^2 + c*x + d
    let P' := fun (x : ℚ) => 3*x^2 + 2*b*x + c
    ∃ (r₁ r₂ r₃ : ℚ), r₁ ≠ r₂ ∧ r₂ ≠ r₃ ∧ r₃ ≠ r₁ ∧
    P r₁ = 0 ∧ P r₂ = 0 ∧ P r₃ = 0 ∧
    ∃ (c₁ c₂ : ℚ), P' c₁ = 0 ∧ P' c₂ = 0 := by
  sorry

end NUMINAMATH_CALUDE_rational_cubic_polynomial_existence_l1062_106287


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l1062_106203

theorem partial_fraction_decomposition (M₁ M₂ : ℚ) :
  (∀ x : ℚ, x ≠ 1 → x ≠ 2 →
    (48 * x^2 + 26 * x - 35) / (x^2 - 3 * x + 2) = M₁ / (x - 1) + M₂ / (x - 2)) →
  M₁ * M₂ = -1056 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l1062_106203


namespace NUMINAMATH_CALUDE_polyhedron_relations_l1062_106230

structure Polyhedron where
  E : ℕ  -- number of edges
  F : ℕ  -- number of faces
  V : ℕ  -- number of vertices
  n : ℕ  -- number of sides in each face
  m : ℕ  -- number of edges meeting at each vertex

theorem polyhedron_relations (P : Polyhedron) : 
  (P.n * P.F = 2 * P.E) ∧ 
  (P.m * P.V = 2 * P.E) ∧ 
  (P.V + P.F = P.E + 2) ∧ 
  ¬(P.m * P.F = 2 * P.E) := by
  sorry

end NUMINAMATH_CALUDE_polyhedron_relations_l1062_106230


namespace NUMINAMATH_CALUDE_total_roses_planted_l1062_106245

/-- The number of roses planted by Uncle Welly over three days -/
def roses_planted (day1 day2 day3 : ℕ) : ℕ := day1 + day2 + day3

/-- Theorem stating the total number of roses planted -/
theorem total_roses_planted :
  let day1 := 50
  let day2 := day1 + 20
  let day3 := 2 * day1
  roses_planted day1 day2 day3 = 220 := by sorry

end NUMINAMATH_CALUDE_total_roses_planted_l1062_106245


namespace NUMINAMATH_CALUDE_largest_package_size_l1062_106284

theorem largest_package_size (alex_folders jamie_folders : ℕ) 
  (h1 : alex_folders = 60) (h2 : jamie_folders = 90) : 
  Nat.gcd alex_folders jamie_folders = 30 := by
  sorry

end NUMINAMATH_CALUDE_largest_package_size_l1062_106284


namespace NUMINAMATH_CALUDE_unpainted_cubes_in_6x6x6_l1062_106214

/-- Represents a cube composed of unit cubes -/
structure Cube where
  size : Nat
  total_units : Nat
  painted_per_face : Nat

/-- Calculates the number of unpainted unit cubes in a cube with painted cross patterns on each face -/
def unpainted_cubes (c : Cube) : Nat :=
  c.total_units - (c.painted_per_face * 6 - 24 - 12)

/-- Theorem stating the number of unpainted cubes in the specific 6x6x6 cube -/
theorem unpainted_cubes_in_6x6x6 :
  let c : Cube := { size := 6, total_units := 216, painted_per_face := 10 }
  unpainted_cubes c = 180 := by
  sorry

end NUMINAMATH_CALUDE_unpainted_cubes_in_6x6x6_l1062_106214


namespace NUMINAMATH_CALUDE_intermediate_root_existence_l1062_106212

theorem intermediate_root_existence (a b c x₁ x₂ : ℝ) 
  (ha : a ≠ 0)
  (hx₁ : a * x₁^2 + b * x₁ + c = 0)
  (hx₂ : -a * x₂^2 + b * x₂ + c = 0) :
  ∃ x₃ : ℝ, (a / 2) * x₃^2 + b * x₃ + c = 0 ∧ 
    ((x₁ ≤ x₃ ∧ x₃ ≤ x₂) ∨ (x₁ ≥ x₃ ∧ x₃ ≥ x₂)) := by
  sorry

end NUMINAMATH_CALUDE_intermediate_root_existence_l1062_106212


namespace NUMINAMATH_CALUDE_intersection_theorem_l1062_106204

-- Define the curves
def curve1 (x y a : ℝ) : Prop := (x - 1)^2 + y^2 = a^2
def curve2 (x y a : ℝ) : Prop := y = x^2 - a

-- Define the intersection points
def intersection_points (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | curve1 p.1 p.2 a ∧ curve2 p.1 p.2 a}

-- Define the condition for exactly three intersection points
def has_exactly_three_intersections (a : ℝ) : Prop :=
  ∃ p q r : ℝ × ℝ, p ≠ q ∧ q ≠ r ∧ p ≠ r ∧
  intersection_points a = {p, q, r}

-- Theorem statement
theorem intersection_theorem :
  ∀ a : ℝ, has_exactly_three_intersections a ↔ 
  (a = (3 + Real.sqrt 5) / 2 ∨ a = (3 - Real.sqrt 5) / 2) :=
sorry

end NUMINAMATH_CALUDE_intersection_theorem_l1062_106204


namespace NUMINAMATH_CALUDE_gold_coin_percentage_is_45_5_percent_l1062_106297

/-- Represents the composition of items in an urn --/
structure UrnComposition where
  beadPercentage : ℝ
  bronzeCoinPercentage : ℝ

/-- Calculates the percentage of gold coins in the urn --/
def goldCoinPercentage (urn : UrnComposition) : ℝ :=
  (1 - urn.beadPercentage) * (1 - urn.bronzeCoinPercentage)

/-- Theorem: The percentage of gold coins in the urn is 45.5% --/
theorem gold_coin_percentage_is_45_5_percent (urn : UrnComposition)
  (h1 : urn.beadPercentage = 0.35)
  (h2 : urn.bronzeCoinPercentage = 0.30) :
  goldCoinPercentage urn = 0.455 := by
  sorry

#eval goldCoinPercentage { beadPercentage := 0.35, bronzeCoinPercentage := 0.30 }

end NUMINAMATH_CALUDE_gold_coin_percentage_is_45_5_percent_l1062_106297


namespace NUMINAMATH_CALUDE_largest_prime_divisor_of_n_l1062_106294

/-- Represents a number in base 5 -/
def BaseNumber (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- The number 201032021 in base 5 -/
def n : Nat := BaseNumber [1, 2, 0, 2, 3, 0, 1, 0, 2]

/-- 31 is a prime number -/
axiom thirty_one_prime : Prime 31

/-- 31 divides n -/
axiom thirty_one_divides_n : 31 ∣ n

theorem largest_prime_divisor_of_n :
  ∀ p : Nat, Prime p → p ∣ n → p ≤ 31 := by
  sorry

#check largest_prime_divisor_of_n

end NUMINAMATH_CALUDE_largest_prime_divisor_of_n_l1062_106294


namespace NUMINAMATH_CALUDE_matchsticks_left_l1062_106246

def totalMatchsticks : ℕ := 50
def elvisSquareMatchsticks : ℕ := 4
def ralphSquareMatchsticks : ℕ := 8
def zoeyTriangleMatchsticks : ℕ := 6
def elvisMaxMatchsticks : ℕ := 20
def ralphMaxMatchsticks : ℕ := 20
def zoeyMaxMatchsticks : ℕ := 15
def maxTotalShapes : ℕ := 9

theorem matchsticks_left : 
  ∃ (elvisShapes ralphShapes zoeyShapes : ℕ),
    elvisShapes * elvisSquareMatchsticks ≤ elvisMaxMatchsticks ∧
    ralphShapes * ralphSquareMatchsticks ≤ ralphMaxMatchsticks ∧
    zoeyShapes * zoeyTriangleMatchsticks ≤ zoeyMaxMatchsticks ∧
    elvisShapes + ralphShapes + zoeyShapes = maxTotalShapes ∧
    totalMatchsticks - (elvisShapes * elvisSquareMatchsticks + 
                        ralphShapes * ralphSquareMatchsticks + 
                        zoeyShapes * zoeyTriangleMatchsticks) = 2 :=
by sorry

end NUMINAMATH_CALUDE_matchsticks_left_l1062_106246


namespace NUMINAMATH_CALUDE_tire_price_proof_l1062_106247

/-- The regular price of a tire -/
def regular_price : ℝ := 126

/-- The promotional price for three tires -/
def promotional_price : ℝ := 315

/-- The promotion discount on the third tire -/
def discount : ℝ := 0.5

theorem tire_price_proof :
  2 * regular_price + discount * regular_price = promotional_price :=
by sorry

end NUMINAMATH_CALUDE_tire_price_proof_l1062_106247


namespace NUMINAMATH_CALUDE_smallest_value_of_expression_l1062_106291

theorem smallest_value_of_expression (p q t : ℕ) : 
  Nat.Prime p → Nat.Prime q → Nat.Prime t →
  p < q → q < t → p ≠ q → q ≠ t → p ≠ t →
  (∀ p' q' t' : ℕ, Nat.Prime p' → Nat.Prime q' → Nat.Prime t' →
    p' < q' → q' < t' → p' ≠ q' → q' ≠ t' → p' ≠ t' →
    p' * q' * t' + p' * t' + q' * t' + q' * t' ≥ p * q * t + p * t + q * t + q * t) →
  p * q * t + p * t + q * t + q * t = 70 :=
by sorry

end NUMINAMATH_CALUDE_smallest_value_of_expression_l1062_106291


namespace NUMINAMATH_CALUDE_solution_pairs_l1062_106271

theorem solution_pairs : 
  ∀ (x y : ℕ), 2^(2*x+1) + 2^x + 1 = y^2 ↔ (x = 0 ∧ y = 2) ∨ (x = 4 ∧ y = 23) :=
by sorry

end NUMINAMATH_CALUDE_solution_pairs_l1062_106271


namespace NUMINAMATH_CALUDE_unique_preimage_of_triple_l1062_106241

-- Define v₂ function
def v₂ (n : ℕ) : ℕ :=
  if n = 0 then 0 else (n.log 2)

-- Define the properties of function f
def has_properties (f : ℕ → ℕ) : Prop :=
  (∀ x : ℕ, f x ≤ 3 * x) ∧ 
  (∀ x y : ℕ, v₂ (f x + f y) = v₂ (x + y))

-- State the theorem
theorem unique_preimage_of_triple (f : ℕ → ℕ) (h : has_properties f) :
  ∀ a : ℕ, ∃! x : ℕ, f x = 3 * a :=
sorry

end NUMINAMATH_CALUDE_unique_preimage_of_triple_l1062_106241


namespace NUMINAMATH_CALUDE_alice_cookies_l1062_106288

/-- Given that Alice can make 24 cookies with 4 cups of flour,
    this theorem proves that she can make 30 cookies with 5 cups of flour. -/
theorem alice_cookies (cookies_four : ℕ) (flour_four : ℕ) (flour_five : ℕ)
  (h1 : cookies_four = 24)
  (h2 : flour_four = 4)
  (h3 : flour_five = 5)
  : (cookies_four * flour_five) / flour_four = 30 := by
  sorry

end NUMINAMATH_CALUDE_alice_cookies_l1062_106288


namespace NUMINAMATH_CALUDE_log_2_base_10_bound_l1062_106263

theorem log_2_base_10_bound (h1 : 10^3 = 1000) (h2 : 10^5 = 100000)
  (h3 : 2^12 = 4096) (h4 : 2^15 = 32768) (h5 : 2^17 = 131072) :
  5/17 < Real.log 2 / Real.log 10 := by
  sorry

end NUMINAMATH_CALUDE_log_2_base_10_bound_l1062_106263


namespace NUMINAMATH_CALUDE_sqrt_simplification_exists_l1062_106268

theorem sqrt_simplification_exists :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a * Real.sqrt (b / a) = Real.sqrt (a * b) :=
sorry

end NUMINAMATH_CALUDE_sqrt_simplification_exists_l1062_106268


namespace NUMINAMATH_CALUDE_intersection_implies_m_equals_three_l1062_106292

def A (m : ℝ) : Set ℝ := {1, 2, m}
def B : Set ℝ := {3, 4}

theorem intersection_implies_m_equals_three (m : ℝ) :
  A m ∩ B = {3} → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_m_equals_three_l1062_106292


namespace NUMINAMATH_CALUDE_correct_recommendation_count_l1062_106226

/-- Represents the number of recommendation spots for each language -/
structure RecommendationSpots :=
  (russian : Nat)
  (japanese : Nat)
  (spanish : Nat)

/-- Represents the number of male and female candidates -/
structure Candidates :=
  (males : Nat)
  (females : Nat)

/-- Calculate the number of different recommendation plans -/
def countRecommendationPlans (spots : RecommendationSpots) (candidates : Candidates) : Nat :=
  sorry

theorem correct_recommendation_count :
  let spots := RecommendationSpots.mk 2 2 1
  let candidates := Candidates.mk 3 2
  countRecommendationPlans spots candidates = 24 :=
by sorry

end NUMINAMATH_CALUDE_correct_recommendation_count_l1062_106226


namespace NUMINAMATH_CALUDE_initial_markup_percentage_l1062_106237

theorem initial_markup_percentage (initial_price : ℝ) (price_increase : ℝ) : 
  initial_price = 24 →
  price_increase = 6 →
  let final_price := initial_price + price_increase
  let wholesale_price := final_price / 2
  let initial_markup := initial_price - wholesale_price
  initial_markup / wholesale_price = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_initial_markup_percentage_l1062_106237


namespace NUMINAMATH_CALUDE_right_triangle_area_l1062_106228

/-- A right triangle with vertices at (0, 0), (0, 10), and (-10, 0), 
    and two points (-3, 7) and (-7, 3) on its hypotenuse. -/
structure RightTriangle where
  -- Define the vertices
  v1 : ℝ × ℝ := (0, 0)
  v2 : ℝ × ℝ := (0, 10)
  v3 : ℝ × ℝ := (-10, 0)
  -- Define the points on the hypotenuse
  p1 : ℝ × ℝ := (-3, 7)
  p2 : ℝ × ℝ := (-7, 3)
  -- Ensure the triangle is right-angled
  is_right_angle : (v2.1 - v1.1) * (v3.1 - v1.1) + (v2.2 - v1.2) * (v3.2 - v1.2) = 0
  -- Ensure the points lie on the hypotenuse
  p1_on_hypotenuse : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ p1 = (t * v2.1 + (1 - t) * v3.1, t * v2.2 + (1 - t) * v3.2)
  p2_on_hypotenuse : ∃ s : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ p2 = (s * v2.1 + (1 - s) * v3.1, s * v2.2 + (1 - s) * v3.2)

/-- The area of the right triangle is 50 square units. -/
theorem right_triangle_area (t : RightTriangle) : 
  (1/2) * abs (t.v2.1 * t.v3.2 - t.v3.1 * t.v2.2) = 50 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l1062_106228


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1062_106250

theorem inequality_solution_set (x : ℝ) : (3 * x - 1) / (2 - x) ≥ 1 ↔ 3 / 4 ≤ x ∧ x < 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1062_106250


namespace NUMINAMATH_CALUDE_perfect_squares_condition_l1062_106200

theorem perfect_squares_condition (n : ℕ+) : 
  (∃ a b : ℕ, (8 * n.val - 7 = a ^ 2) ∧ (18 * n.val - 35 = b ^ 2)) ↔ (n.val = 2 ∨ n.val = 22) := by
  sorry

end NUMINAMATH_CALUDE_perfect_squares_condition_l1062_106200


namespace NUMINAMATH_CALUDE_parabola_directrix_l1062_106260

/-- The directrix of the parabola y = 3x^2 + 6x + 5 is y = 23/12 -/
theorem parabola_directrix : 
  ∀ (x y : ℝ), y = 3 * x^2 + 6 * x + 5 → 
  ∃ (k : ℝ), k = 23/12 ∧ (∀ (x₀ : ℝ), (x - x₀)^2 = 4 * (1/12) * (y - k)) :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l1062_106260


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a7_l1062_106239

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a7 (a : ℕ → ℝ) (h_arith : ArithmeticSequence a)
    (h_a4 : a 4 = 4) (h_sum : a 3 + a 8 = 5) : a 7 = 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a7_l1062_106239


namespace NUMINAMATH_CALUDE_train_bridge_crossing_time_l1062_106223

/-- Proves that a train of given length, traveling at a given speed, will take the calculated time to cross a bridge of given length. -/
theorem train_bridge_crossing_time
  (train_length : ℝ)
  (bridge_length : ℝ)
  (train_speed_kmph : ℝ)
  (h1 : train_length = 100)
  (h2 : bridge_length = 200)
  (h3 : train_speed_kmph = 36)
  : (train_length + bridge_length) / (train_speed_kmph * 1000 / 3600) = 30 :=
by sorry

end NUMINAMATH_CALUDE_train_bridge_crossing_time_l1062_106223


namespace NUMINAMATH_CALUDE_complex_rational_sum_l1062_106280

def separate_and_sum (a b c d : ℚ) : ℚ :=
  let int_part := a.floor + b.floor + c.floor + d.floor
  let frac_part := (a - a.floor) + (b - b.floor) + (c - c.floor) + (d - d.floor)
  int_part + frac_part

theorem complex_rational_sum :
  separate_and_sum (-206) (401 + 3/4) (-204 - 2/3) (-1 - 1/2) = -10 - 5/12 :=
by sorry

end NUMINAMATH_CALUDE_complex_rational_sum_l1062_106280


namespace NUMINAMATH_CALUDE_factorization_equality_l1062_106243

theorem factorization_equality (a : ℝ) : -3*a + 12*a^2 - 12*a^3 = -3*a*(1-2*a)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1062_106243


namespace NUMINAMATH_CALUDE_max_value_of_x_plus_inverse_l1062_106258

theorem max_value_of_x_plus_inverse (x : ℝ) (h : 13 = x^2 + 1/x^2) :
  (∀ y : ℝ, y > 0 → 13 = y^2 + 1/y^2 → x + 1/x ≥ y + 1/y) → x + 1/x = Real.sqrt 15 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_x_plus_inverse_l1062_106258


namespace NUMINAMATH_CALUDE_square_sum_problem_l1062_106215

theorem square_sum_problem (a b : ℝ) (h1 : a + b = -9) (h2 : a = 30 / b) : a^2 + b^2 = 61 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_problem_l1062_106215


namespace NUMINAMATH_CALUDE_safari_park_animal_difference_l1062_106219

theorem safari_park_animal_difference :
  let safari_lions : ℕ := 100
  let safari_snakes : ℕ := safari_lions / 2
  let savanna_lions : ℕ := safari_lions * 2
  let savanna_snakes : ℕ := safari_snakes * 3
  let safari_giraffes : ℕ := safari_snakes - (savanna_lions + savanna_snakes + safari_giraffes + 20 - 410)
  safari_snakes - safari_giraffes = 10 := by
  sorry

end NUMINAMATH_CALUDE_safari_park_animal_difference_l1062_106219


namespace NUMINAMATH_CALUDE_edward_final_lives_l1062_106261

/-- Calculates the final number of lives Edward has after completing three stages of a game. -/
def final_lives (initial_lives : ℕ) 
                (stage1_loss stage1_gain : ℕ) 
                (stage2_loss stage2_gain : ℕ) 
                (stage3_loss stage3_gain : ℕ) : ℕ :=
  initial_lives - stage1_loss + stage1_gain - stage2_loss + stage2_gain - stage3_loss + stage3_gain

/-- Theorem stating that Edward's final number of lives is 23 given the specified conditions. -/
theorem edward_final_lives : 
  final_lives 50 18 7 10 5 13 2 = 23 := by
  sorry


end NUMINAMATH_CALUDE_edward_final_lives_l1062_106261


namespace NUMINAMATH_CALUDE_scientific_notation_of_400_million_l1062_106270

theorem scientific_notation_of_400_million :
  (400000000 : ℝ) = 4 * 10^8 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_400_million_l1062_106270


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1062_106236

/-- An arithmetic sequence with general term formula aₙ = -n + 5 -/
def arithmeticSequence (n : ℕ) : ℤ := -n + 5

/-- The common difference of an arithmetic sequence -/
def commonDifference (a : ℕ → ℤ) : ℤ := a (1 : ℕ) - a 0

theorem arithmetic_sequence_common_difference :
  commonDifference arithmeticSequence = -1 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1062_106236


namespace NUMINAMATH_CALUDE_energy_usage_is_96_watts_l1062_106295

/-- Calculate total energy usage for three lights over a given time period -/
def totalEnergyUsage (baseWatts : ℕ) (hours : ℕ) : ℕ :=
  let lightA := baseWatts * hours
  let lightB := 3 * lightA
  let lightC := 4 * lightA
  lightA + lightB + lightC

/-- Theorem: The total energy usage for the given scenario is 96 watts -/
theorem energy_usage_is_96_watts :
  totalEnergyUsage 6 2 = 96 := by sorry

end NUMINAMATH_CALUDE_energy_usage_is_96_watts_l1062_106295


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1062_106275

theorem quadratic_inequality_range (k : ℝ) : 
  (∀ x : ℝ, 2 * k * x^2 + k * x - 3/8 < 0) → -3 < k ∧ k < 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1062_106275


namespace NUMINAMATH_CALUDE_a_100_value_l1062_106298

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n, a n - a (n + 1) = 2

theorem a_100_value (a : ℕ → ℤ) 
  (h1 : arithmetic_sequence a) 
  (h2 : a 3 = 6) : 
  a 100 = -188 := by
  sorry

end NUMINAMATH_CALUDE_a_100_value_l1062_106298


namespace NUMINAMATH_CALUDE_consecutive_four_plus_one_is_square_l1062_106222

theorem consecutive_four_plus_one_is_square (n : ℕ) :
  n * (n + 1) * (n + 2) * (n + 3) + 1 = (n^2 + 3*n + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_four_plus_one_is_square_l1062_106222


namespace NUMINAMATH_CALUDE_tv_show_sampling_interval_l1062_106218

/-- Calculate the sampling interval for system sampling --/
def sampling_interval (total_population : ℕ) (sample_size : ℕ) : ℕ :=
  total_population / sample_size

/-- Theorem: The sampling interval for selecting 10 viewers from 10,000 is 1000 --/
theorem tv_show_sampling_interval :
  sampling_interval 10000 10 = 1000 := by
  sorry

#eval sampling_interval 10000 10

end NUMINAMATH_CALUDE_tv_show_sampling_interval_l1062_106218


namespace NUMINAMATH_CALUDE_exponent_zero_equals_one_f_equals_S_l1062_106227

-- Option C
theorem exponent_zero_equals_one (x : ℝ) (h : x ≠ 0) : x^0 = 1 := by sorry

-- Option D
def f (x : ℝ) : ℝ := x^2
def S (t : ℝ) : ℝ := t^2

theorem f_equals_S : f = S := by sorry

end NUMINAMATH_CALUDE_exponent_zero_equals_one_f_equals_S_l1062_106227


namespace NUMINAMATH_CALUDE_brocard_angle_inequalities_l1062_106255

/-- The Brocard angle of a triangle -/
def brocard_angle (α β γ : ℝ) : ℝ := sorry

/-- Theorem: Brocard angle inequalities -/
theorem brocard_angle_inequalities (α β γ : ℝ) (hα : 0 < α) (hβ : 0 < β) (hγ : 0 < γ) 
  (hsum : α + β + γ = π) :
  let φ := brocard_angle α β γ
  (φ^3 ≤ (α - φ) * (β - φ) * (γ - φ)) ∧ (8 * φ^3 ≤ α * β * γ) := by sorry

end NUMINAMATH_CALUDE_brocard_angle_inequalities_l1062_106255


namespace NUMINAMATH_CALUDE_golden_ratio_logarithm_l1062_106224

theorem golden_ratio_logarithm (r s : ℝ) (hr : r > 0) (hs : s > 0) :
  (Real.log r / Real.log 4 = Real.log s / Real.log 18) ∧
  (Real.log s / Real.log 18 = Real.log (r + s) / Real.log 24) →
  s / r = (1 + Real.sqrt 5) / 2 := by
sorry

end NUMINAMATH_CALUDE_golden_ratio_logarithm_l1062_106224


namespace NUMINAMATH_CALUDE_fred_paper_count_l1062_106210

theorem fred_paper_count (initial_sheets received_sheets given_sheets : ℕ) :
  initial_sheets = 212 →
  received_sheets = 307 →
  given_sheets = 156 →
  initial_sheets + received_sheets - given_sheets = 363 := by
  sorry

end NUMINAMATH_CALUDE_fred_paper_count_l1062_106210


namespace NUMINAMATH_CALUDE_triangle_perimeter_and_shape_l1062_106253

/-- Given a triangle ABC with side lengths a, b, and c satisfying certain conditions,
    prove that its perimeter is 17 and it is an isosceles triangle. -/
theorem triangle_perimeter_and_shape (a b c : ℝ) : 
  (b - 5)^2 + (c - 7)^2 = 0 →
  |a - 3| = 2 →
  a + b + c = 17 ∧ a = b := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_and_shape_l1062_106253


namespace NUMINAMATH_CALUDE_conic_is_circle_l1062_106259

-- Define the equation
def conic_equation (x y : ℝ) : Prop := (x - 3)^2 + (y + 4)^2 = 49

-- Theorem stating that the equation represents a circle
theorem conic_is_circle :
  ∃ (h k r : ℝ), r > 0 ∧ 
  (∀ (x y : ℝ), conic_equation x y ↔ (x - h)^2 + (y - k)^2 = r^2) :=
sorry

end NUMINAMATH_CALUDE_conic_is_circle_l1062_106259
