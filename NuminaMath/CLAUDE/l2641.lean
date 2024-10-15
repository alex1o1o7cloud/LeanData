import Mathlib

namespace NUMINAMATH_CALUDE_problem_statement_l2641_264127

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : (a+2)*(b+2) = 18) :
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ (x+2)*(y+2) = 18 ∧ 3/(x+2) + 3/(y+2) < 3/(a+2) + 3/(b+2)) ∨
  (3/(a+2) + 3/(b+2) = Real.sqrt 2) ∧
  (∀ (x y : ℝ), x > 0 → y > 0 → (x+2)*(y+2) = 18 → 2*x + y ≥ 6) ∧
  (∀ (x y : ℝ), x > 0 → y > 0 → (x+2)*(y+2) = 18 → (x+1)*y ≤ 8) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2641_264127


namespace NUMINAMATH_CALUDE_ten_consecutive_composites_l2641_264105

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

/-- A function that checks if a number is composite -/
def isComposite (n : ℕ) : Prop := n > 1 ∧ ¬(isPrime n)

theorem ten_consecutive_composites :
  ∃ (start : ℕ), 
    start + 9 < 500 ∧
    (∀ i : ℕ, i ∈ Finset.range 10 → isComposite (start + i)) ∧
    start + 9 = 489 := by
  sorry

end NUMINAMATH_CALUDE_ten_consecutive_composites_l2641_264105


namespace NUMINAMATH_CALUDE_exp_addition_property_l2641_264129

-- Define the function f(x) = e^x
noncomputable def f (x : ℝ) : ℝ := Real.exp x

-- State the theorem
theorem exp_addition_property (x y : ℝ) : f (x + y) = f x * f y := by sorry

end NUMINAMATH_CALUDE_exp_addition_property_l2641_264129


namespace NUMINAMATH_CALUDE_min_sum_positive_reals_l2641_264123

theorem min_sum_positive_reals (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / (2 * b) + b / (4 * c) + c / (8 * a)) ≥ (3 / 4) :=
by sorry

end NUMINAMATH_CALUDE_min_sum_positive_reals_l2641_264123


namespace NUMINAMATH_CALUDE_solution_of_equation_l2641_264109

theorem solution_of_equation :
  ∃! y : ℚ, 6 * (4 * y - 1) - 3 = 3 * (2 - 5 * y) ∧ y = 5 / 13 := by
  sorry

end NUMINAMATH_CALUDE_solution_of_equation_l2641_264109


namespace NUMINAMATH_CALUDE_flooring_cost_is_14375_l2641_264145

/-- Represents the dimensions and cost of a rectangular room -/
structure RectRoom where
  length : Float
  width : Float
  cost_per_sqm : Float

/-- Represents the dimensions and cost of an L-shaped room -/
structure LShapeRoom where
  rect1_length : Float
  rect1_width : Float
  rect2_length : Float
  rect2_width : Float
  cost_per_sqm : Float

/-- Represents the dimensions and cost of a triangular room -/
structure TriRoom where
  base : Float
  height : Float
  cost_per_sqm : Float

/-- Calculates the total cost of flooring for all rooms -/
def total_flooring_cost (room1 : RectRoom) (room2 : LShapeRoom) (room3 : TriRoom) : Float :=
  (room1.length * room1.width * room1.cost_per_sqm) +
  ((room2.rect1_length * room2.rect1_width + room2.rect2_length * room2.rect2_width) * room2.cost_per_sqm) +
  (0.5 * room3.base * room3.height * room3.cost_per_sqm)

/-- Theorem stating that the total flooring cost for the given rooms is $14,375 -/
theorem flooring_cost_is_14375 
  (room1 : RectRoom)
  (room2 : LShapeRoom)
  (room3 : TriRoom)
  (h1 : room1 = { length := 5.5, width := 3.75, cost_per_sqm := 400 })
  (h2 : room2 = { rect1_length := 4, rect1_width := 2.5, rect2_length := 2, rect2_width := 1.5, cost_per_sqm := 350 })
  (h3 : room3 = { base := 3.5, height := 2, cost_per_sqm := 450 }) :
  total_flooring_cost room1 room2 room3 = 14375 := by
  sorry

end NUMINAMATH_CALUDE_flooring_cost_is_14375_l2641_264145


namespace NUMINAMATH_CALUDE_linear_function_proof_l2641_264130

-- Define the linear function
def linear_function (k b : ℝ) (x : ℝ) : ℝ := k * x + b

-- Define the theorem
theorem linear_function_proof :
  ∃ (k b : ℝ),
    (linear_function k b 1 = 5) ∧
    (linear_function k b (-1) = -1) ∧
    (∀ (x : ℝ), linear_function k b x = 3 * x + 2) ∧
    (linear_function k b 2 = 8) := by
  sorry


end NUMINAMATH_CALUDE_linear_function_proof_l2641_264130


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l2641_264141

/-- The slope of the asymptotes of a hyperbola with specific properties -/
theorem hyperbola_asymptote_slope (a b c : ℝ) (ha : a > 0) (hb : b > 0) :
  let A₁ : ℝ × ℝ := (-a, 0)
  let A₂ : ℝ × ℝ := (a, 0)
  let B : ℝ × ℝ := (c, b^2/a)
  let C : ℝ × ℝ := (c, -b^2/a)
  (∀ x y, x^2/a^2 - y^2/b^2 = 1 → (x = c ∧ (y = b^2/a ∨ y = -b^2/a))) →
  ((B.2 - A₁.2) * (C.2 - A₂.2) = -(B.1 - A₁.1) * (C.1 - A₂.1)) →
  (∀ x, (x : ℝ) = x ∨ (x : ℝ) = -x) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l2641_264141


namespace NUMINAMATH_CALUDE_tan_theta_solution_l2641_264162

theorem tan_theta_solution (θ : Real) (h1 : 0 < θ * (180 / Real.pi)) 
  (h2 : θ * (180 / Real.pi) < 30) 
  (h3 : Real.tan θ + Real.tan (4 * θ) + Real.tan (6 * θ) = 0) : 
  Real.tan θ = 1 / Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_tan_theta_solution_l2641_264162


namespace NUMINAMATH_CALUDE_certain_number_problem_l2641_264140

theorem certain_number_problem (x : ℕ) (certain_number : ℕ) 
  (h1 : 9873 + x = certain_number) (h2 : x = 3327) : 
  certain_number = 13200 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l2641_264140


namespace NUMINAMATH_CALUDE_remaining_marbles_l2641_264112

def initial_marbles : ℕ := 64
def marbles_given : ℕ := 14

theorem remaining_marbles :
  initial_marbles - marbles_given = 50 :=
by sorry

end NUMINAMATH_CALUDE_remaining_marbles_l2641_264112


namespace NUMINAMATH_CALUDE_car_selection_proof_l2641_264172

theorem car_selection_proof (num_cars : ℕ) (num_clients : ℕ) (selections_per_client : ℕ)
  (h1 : num_cars = 10)
  (h2 : num_clients = 15)
  (h3 : selections_per_client = 2)
  (h4 : ∀ i, 1 ≤ i ∧ i ≤ num_cars → ∃ (x : ℕ), x > 0) :
  ∀ i, 1 ≤ i ∧ i ≤ num_cars → ∃ (x : ℕ), x = 3 :=
by sorry

end NUMINAMATH_CALUDE_car_selection_proof_l2641_264172


namespace NUMINAMATH_CALUDE_min_sum_reciprocals_roots_l2641_264167

theorem min_sum_reciprocals_roots (k : ℝ) (x₁ x₂ : ℝ) : 
  x₁ > 0 ∧ x₂ > 0 ∧ 
  x₁^2 - k*x₁ + k + 3 = 0 ∧ 
  x₂^2 - k*x₂ + k + 3 = 0 ∧ 
  x₁ ≠ x₂ →
  (∃ (s : ℝ), s = 1/x₁ + 1/x₂ ∧ s ≥ 2/3 ∧ ∀ (t : ℝ), t = 1/x₁ + 1/x₂ → t ≥ 2/3) :=
sorry

end NUMINAMATH_CALUDE_min_sum_reciprocals_roots_l2641_264167


namespace NUMINAMATH_CALUDE_find_x_l2641_264173

theorem find_x : ∃ X : ℤ, X - (5 - (6 + 2 * (7 - 8 - 5))) = 89 ∧ X = 100 := by
  sorry

end NUMINAMATH_CALUDE_find_x_l2641_264173


namespace NUMINAMATH_CALUDE_days_without_calls_l2641_264133

/-- The number of days in the year -/
def total_days : ℕ := 365

/-- The calling frequencies of the three grandchildren -/
def call_frequencies : List ℕ := [4, 6, 8]

/-- Calculate the number of days with at least one call -/
def days_with_calls (frequencies : List ℕ) (total : ℕ) : ℕ :=
  sorry

/-- The main theorem to prove -/
theorem days_without_calls (frequencies : List ℕ) (total : ℕ) :
  frequencies = call_frequencies → total = total_days →
  total - days_with_calls frequencies total = 244 :=
by sorry

end NUMINAMATH_CALUDE_days_without_calls_l2641_264133


namespace NUMINAMATH_CALUDE_calculator_probability_l2641_264147

/-- The probability of a specific number M appearing on the display
    when starting from a number N, where M < N -/
def prob_appear (N M : ℕ) : ℚ :=
  if M < N then 1 / (M + 1 : ℚ) else 0

/-- The probability of all numbers in a list appearing on the display
    when starting from a given number -/
def prob_all_appear (start : ℕ) (numbers : List ℕ) : ℚ :=
  numbers.foldl (fun acc n => acc * prob_appear start n) 1

theorem calculator_probability :
  prob_all_appear 2003 [1000, 100, 10, 1] = 1 / 2224222 := by
  sorry

end NUMINAMATH_CALUDE_calculator_probability_l2641_264147


namespace NUMINAMATH_CALUDE_range_of_f_l2641_264196

def f (x : ℕ) : ℤ := x^2 - 2*x

def domain : Set ℕ := {0, 1, 2, 3}

theorem range_of_f : 
  {y : ℤ | ∃ x ∈ domain, f x = y} = {-1, 0, 3} := by sorry

end NUMINAMATH_CALUDE_range_of_f_l2641_264196


namespace NUMINAMATH_CALUDE_steel_making_experiment_l2641_264132

/-- The 0.618 method calculation for steel-making experiment --/
theorem steel_making_experiment (lower upper : ℝ) (h1 : lower = 500) (h2 : upper = 1000) :
  lower + (upper - lower) * 0.618 = 809 :=
by sorry

end NUMINAMATH_CALUDE_steel_making_experiment_l2641_264132


namespace NUMINAMATH_CALUDE_taylor_family_reunion_l2641_264103

theorem taylor_family_reunion (kids : ℕ) (tables : ℕ) (people_per_table : ℕ) (adults : ℕ) : 
  kids = 45 → tables = 14 → people_per_table = 12 → 
  adults = tables * people_per_table - kids → adults = 123 :=
by
  sorry

end NUMINAMATH_CALUDE_taylor_family_reunion_l2641_264103


namespace NUMINAMATH_CALUDE_quadratic_factorization_l2641_264153

theorem quadratic_factorization (c d : ℤ) :
  (∀ x : ℝ, (5*x + c) * (5*x + d) = 25*x^2 - 135*x - 150) →
  c + 2*d = -59 := by
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l2641_264153


namespace NUMINAMATH_CALUDE_repeating_decimal_47_l2641_264188

/-- The repeating decimal 0.474747... is equal to 47/99 -/
theorem repeating_decimal_47 : ∀ (x : ℚ), (∃ (n : ℕ), x * 10^n = ⌊x * 10^n⌋ + 0.47) → x = 47 / 99 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_47_l2641_264188


namespace NUMINAMATH_CALUDE_target_parabola_satisfies_conditions_l2641_264163

/-- Represents a parabola with specific properties -/
structure Parabola where
  -- Equation coefficients
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ
  e : ℤ
  f : ℤ
  -- Conditions
  passes_through : a * 2^2 + b * 2 * 8 + c * 8^2 + d * 2 + e * 8 + f = 0
  focus_y : ℤ := 4
  vertex_on_y_axis : a * 0^2 + b * 0 * 4 + c * 4^2 + d * 0 + e * 4 + f = 0
  c_positive : c > 0
  gcd_one : Nat.gcd (Nat.gcd (Nat.gcd (Nat.gcd (Nat.gcd (Int.natAbs a) (Int.natAbs b)) (Int.natAbs c)) (Int.natAbs d)) (Int.natAbs e)) (Int.natAbs f) = 1

/-- The specific parabola we want to prove -/
def target_parabola : Parabola :=
  { a := 0,
    b := 0,
    c := 1,
    d := -8,
    e := -8,
    f := 16,
    passes_through := sorry,
    focus_y := 4,
    vertex_on_y_axis := sorry,
    c_positive := sorry,
    gcd_one := sorry }

/-- Theorem stating that the target parabola satisfies all conditions -/
theorem target_parabola_satisfies_conditions : 
  ∃ (p : Parabola), p = target_parabola := by sorry

end NUMINAMATH_CALUDE_target_parabola_satisfies_conditions_l2641_264163


namespace NUMINAMATH_CALUDE_triangle_angle_inequalities_l2641_264149

theorem triangle_angle_inequalities (α β γ : Real) 
  (h_triangle : α + β + γ = Real.pi) 
  (h_positive : 0 < α ∧ 0 < β ∧ 0 < γ) : 
  (Real.sin α * Real.sin β * Real.sin γ ≤ 3 * Real.sqrt 3 / 8) ∧
  (Real.cos (α/2) * Real.cos (β/2) * Real.cos (γ/2) ≤ 3 * Real.sqrt 3 / 8) := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_inequalities_l2641_264149


namespace NUMINAMATH_CALUDE_connors_date_cost_is_36_l2641_264151

/-- The cost of Connor's movie date --/
def connors_date_cost : ℝ :=
  let ticket_price : ℝ := 10
  let ticket_quantity : ℕ := 2
  let combo_meal_price : ℝ := 11
  let candy_price : ℝ := 2.5
  let candy_quantity : ℕ := 2
  ticket_price * ticket_quantity + combo_meal_price + candy_price * candy_quantity

/-- Theorem stating the total cost of Connor's date --/
theorem connors_date_cost_is_36 : connors_date_cost = 36 := by
  sorry

end NUMINAMATH_CALUDE_connors_date_cost_is_36_l2641_264151


namespace NUMINAMATH_CALUDE_no_solution_exists_l2641_264108

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem no_solution_exists : ¬ ∃ n : ℕ, n * sum_of_digits n = 20222022 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l2641_264108


namespace NUMINAMATH_CALUDE_puzzle_solution_l2641_264199

theorem puzzle_solution (x y z w : ℕ+) 
  (h1 : x^3 = y^2) 
  (h2 : z^4 = w^3) 
  (h3 : z - x = 17) : 
  w - y = 229 := by
  sorry

end NUMINAMATH_CALUDE_puzzle_solution_l2641_264199


namespace NUMINAMATH_CALUDE_positive_real_solutions_range_l2641_264143

theorem positive_real_solutions_range (a : ℝ) : 
  (∀ x : ℝ, x > 0 → Real.pi ^ x = (a + 1) / (2 - a)) ↔ 1/2 < a ∧ a < 2 := by
  sorry

end NUMINAMATH_CALUDE_positive_real_solutions_range_l2641_264143


namespace NUMINAMATH_CALUDE_remi_tomato_seedlings_l2641_264142

theorem remi_tomato_seedlings (day1 : ℕ) (total : ℕ) : 
  day1 = 200 →
  total = 5000 →
  (day1 + 2 * day1 + 3 * (2 * day1) + 4 * (2 * day1) = total) →
  3 * (2 * day1) + 4 * (2 * day1) = 4400 :=
by
  sorry

end NUMINAMATH_CALUDE_remi_tomato_seedlings_l2641_264142


namespace NUMINAMATH_CALUDE_solve_equation_l2641_264150

theorem solve_equation (x : ℝ) :
  Real.sqrt (3 / x + 3) = 5 / 3 → x = -27 / 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2641_264150


namespace NUMINAMATH_CALUDE_parabola_circle_intersection_l2641_264166

/-- Theorem: For a parabola y = x^2 - ax - 3 (a ∈ ℝ) intersecting the x-axis at points A and B,
    and passing through point C(0, -3), if a circle passing through A, B, and C intersects
    the y-axis at point D(0, b), then b = 1. -/
theorem parabola_circle_intersection (a : ℝ) (A B : ℝ × ℝ) (b : ℝ) :
  let f : ℝ → ℝ := λ x ↦ x^2 - a*x - 3
  (f A.1 = 0 ∧ f B.1 = 0) →  -- A and B are on the x-axis
  (∃ D E F : ℝ, (D^2 + E^2 - 4*F > 0) ∧  -- Circle equation coefficients
    (∀ x y : ℝ, x^2 + y^2 + D*x + E*y + F = 0 ↔
      ((x = A.1 ∧ y = A.2) ∨ (x = B.1 ∧ y = B.2) ∨ (x = 0 ∧ y = -3) ∨ (x = 0 ∧ y = b)))) →
  b = 1 :=
by sorry

end NUMINAMATH_CALUDE_parabola_circle_intersection_l2641_264166


namespace NUMINAMATH_CALUDE_solve_for_c_l2641_264198

theorem solve_for_c (y : ℝ) (c : ℝ) (h1 : y > 0) (h2 : (6 * y) / 20 + (c * y) / 10 = 0.6 * y) : c = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_c_l2641_264198


namespace NUMINAMATH_CALUDE_problem1_solution_problem2_solution_l2641_264124

-- Problem 1
def problem1 (x y : ℤ) : ℤ :=
  (2 * x^2 * y - 4 * x * y^2) - (-3 * x * y^2 + x^2 * y)

theorem problem1_solution :
  problem1 (-1) 2 = 6 := by sorry

-- Problem 2
def A (x y : ℤ) : ℤ := x^2 - x*y + y^2
def B (x y : ℤ) : ℤ := -x^2 + 2*x*y + y^2

theorem problem2_solution :
  A 2010 (-1) + B 2010 (-1) = -2008 := by sorry

end NUMINAMATH_CALUDE_problem1_solution_problem2_solution_l2641_264124


namespace NUMINAMATH_CALUDE_smallest_n_divisible_sixty_satisfies_conditions_sixty_is_smallest_smallest_n_is_sixty_l2641_264160

theorem smallest_n_divisible (n : ℕ) : n > 0 ∧ 45 ∣ n^2 ∧ 1152 ∣ n^4 → n ≥ 60 :=
by sorry

theorem sixty_satisfies_conditions : 45 ∣ 60^2 ∧ 1152 ∣ 60^4 :=
by sorry

theorem sixty_is_smallest : ∀ m : ℕ, m > 0 ∧ 45 ∣ m^2 ∧ 1152 ∣ m^4 → m ≥ 60 :=
by sorry

theorem smallest_n_is_sixty : ∃! n : ℕ, n > 0 ∧ 45 ∣ n^2 ∧ 1152 ∣ n^4 ∧ ∀ m : ℕ, (m > 0 ∧ 45 ∣ m^2 ∧ 1152 ∣ m^4 → m ≥ n) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_divisible_sixty_satisfies_conditions_sixty_is_smallest_smallest_n_is_sixty_l2641_264160


namespace NUMINAMATH_CALUDE_circle_properties_l2641_264122

-- Define the circle C
def C : Set (ℝ × ℝ) := {(x, y) | x^2 + (y - 1)^2 = 9}

-- Define the line intercepting the chord
def L : Set (ℝ × ℝ) := {(x, y) | 12*x - 5*y - 8 = 0}

-- Define a general line through the origin
def l (k : ℝ) : Set (ℝ × ℝ) := {(x, y) | y = k*x}

-- Define point Q
def Q : ℝ × ℝ := (1, 2)

theorem circle_properties :
  -- Part 1: Length of the chord
  ∃ (A B : ℝ × ℝ), A ∈ C ∩ L ∧ B ∈ C ∩ L ∧ (A.1 - B.1)^2 + (A.2 - B.2)^2 = 32 ∧
  -- Part 2: Sum of reciprocals of y-coordinates is constant
  ∀ (k : ℝ) (A B : ℝ × ℝ), k ≠ 0 → A ∈ C ∩ l k → B ∈ C ∩ l k → A ≠ B →
    1 / A.2 + 1 / B.2 = -1/4 ∧
  -- Part 3: Slope of line l when sum of squared distances is 22
  ∃ (k : ℝ) (A B : ℝ × ℝ), k ≠ 0 ∧ A ∈ C ∩ l k ∧ B ∈ C ∩ l k ∧ A ≠ B ∧
    (A.1 - Q.1)^2 + (A.2 - Q.2)^2 + (B.1 - Q.1)^2 + (B.2 - Q.2)^2 = 22 ∧ k = 1 :=
sorry

end NUMINAMATH_CALUDE_circle_properties_l2641_264122


namespace NUMINAMATH_CALUDE_line_properties_l2641_264104

def line_equation (x y : ℝ) : Prop := 3 * y = 4 * x - 9

theorem line_properties :
  (∃ m : ℝ, m = 4/3 ∧ ∀ x y : ℝ, line_equation x y → y = m * x + (-3)) ∧
  line_equation 3 1 :=
sorry

end NUMINAMATH_CALUDE_line_properties_l2641_264104


namespace NUMINAMATH_CALUDE_certain_number_divisor_l2641_264197

theorem certain_number_divisor : ∃ n : ℕ, 
  n > 1 ∧ 
  n < 509 - 5 ∧ 
  (509 - 5) % n = 0 ∧ 
  ∀ m : ℕ, m > n → m < 509 - 5 → (509 - 5) % m ≠ 0 ∧
  ∀ k : ℕ, k < 5 → (509 - k) % n ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_certain_number_divisor_l2641_264197


namespace NUMINAMATH_CALUDE_monochromatic_triangle_exists_l2641_264191

/-- A coloring of the edges of a complete graph using three colors. -/
def ThreeColoring (n : ℕ) := Fin n → Fin n → Fin 3

/-- A complete graph with n vertices. -/
def CompleteGraph (n : ℕ) := Fin n

/-- A triangle in a graph is a set of three distinct vertices. -/
def Triangle (n : ℕ) := { t : Fin n × Fin n × Fin n // t.1 ≠ t.2.1 ∧ t.1 ≠ t.2.2 ∧ t.2.1 ≠ t.2.2 }

/-- A triangle is monochromatic if all its edges have the same color. -/
def IsMonochromatic (n : ℕ) (coloring : ThreeColoring n) (t : Triangle n) : Prop :=
  coloring t.val.1 t.val.2.1 = coloring t.val.1 t.val.2.2 ∧
  coloring t.val.1 t.val.2.1 = coloring t.val.2.1 t.val.2.2

/-- The main theorem: any 3-coloring of K₁₇ contains a monochromatic triangle. -/
theorem monochromatic_triangle_exists :
  ∀ (coloring : ThreeColoring 17),
  ∃ (t : Triangle 17), IsMonochromatic 17 coloring t :=
sorry


end NUMINAMATH_CALUDE_monochromatic_triangle_exists_l2641_264191


namespace NUMINAMATH_CALUDE_divisibility_implies_value_l2641_264101

-- Define the polynomials
def f (x : ℝ) : ℝ := x^4 + 4*x^3 + 6*x^2 + 4*x + 1
def g (x p q r s : ℝ) : ℝ := x^5 + 5*x^4 + 10*p*x^3 + 10*q*x^2 + 5*r*x + s

-- State the theorem
theorem divisibility_implies_value (p q r s : ℝ) :
  (∃ h : ℝ → ℝ, ∀ x, g x p q r s = f x * h x) →
  (p + q) * r = -1.5 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_implies_value_l2641_264101


namespace NUMINAMATH_CALUDE_equation_roots_reciprocal_l2641_264177

theorem equation_roots_reciprocal (a : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ 
    (a^2 - 1) * x^2 - (a + 1) * x + 1 = 0 ∧ 
    (a^2 - 1) * y^2 - (a + 1) * y + 1 = 0 ∧ 
    x * y = 1) → 
  a = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_equation_roots_reciprocal_l2641_264177


namespace NUMINAMATH_CALUDE_expand_expression_l2641_264152

theorem expand_expression (a : ℝ) : 4 * a^2 * (3*a - 1) = 12*a^3 - 4*a^2 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l2641_264152


namespace NUMINAMATH_CALUDE_land_reaping_l2641_264157

/-- Given that 4 men can reap 40 acres in 15 days, prove that 16 men can reap 320 acres in 30 days. -/
theorem land_reaping (men_initial : ℕ) (acres_initial : ℕ) (days_initial : ℕ)
                     (men_final : ℕ) (days_final : ℕ) :
  men_initial = 4 →
  acres_initial = 40 →
  days_initial = 15 →
  men_final = 16 →
  days_final = 30 →
  (men_final * days_final * acres_initial) / (men_initial * days_initial) = 320 := by
  sorry

#check land_reaping

end NUMINAMATH_CALUDE_land_reaping_l2641_264157


namespace NUMINAMATH_CALUDE_quadratic_two_real_roots_l2641_264113

theorem quadratic_two_real_roots (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 - x + k + 1 = 0 ∧ y^2 - y + k + 1 = 0) → k ≤ -3/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_two_real_roots_l2641_264113


namespace NUMINAMATH_CALUDE_quadratic_equation_coefficients_l2641_264137

theorem quadratic_equation_coefficients :
  let original_eq : ℝ → Prop := λ x => 3 * x^2 - 1 = 6 * x
  let general_form : ℝ → ℝ → ℝ → ℝ → Prop := λ a b c x => a * x^2 + b * x + c = 0
  ∃ (a b c : ℝ), (∀ x, original_eq x ↔ general_form a b c x) ∧ a = 3 ∧ b = -6 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_coefficients_l2641_264137


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_nine_l2641_264185

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n, S n = n * (a 1 + a n) / 2

/-- Theorem: For an arithmetic sequence with a_5 = 2, S_9 = 18 -/
theorem arithmetic_sequence_sum_nine 
  (seq : ArithmeticSequence) 
  (h : seq.a 5 = 2) : 
  seq.S 9 = 18 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_sum_nine_l2641_264185


namespace NUMINAMATH_CALUDE_inscribed_circle_segment_ratio_l2641_264121

-- Define the triangle and circle
def Triangle (a b c : ℝ) := a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

def InscribedCircle (t : Triangle a b c) := 
  ∃ (r : ℝ), r > 0 ∧ 
  ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧
  x + y = a ∧ y + z = b ∧ z + x = c ∧
  x + y + z = (a + b + c) / 2

-- Define the theorem
theorem inscribed_circle_segment_ratio 
  (t : Triangle 10 15 19) 
  (c : InscribedCircle t) :
  ∃ (r s : ℝ), r > 0 ∧ s > 0 ∧ r < s ∧ r + s = 10 ∧ r / s = 3 / 7 :=
sorry

end NUMINAMATH_CALUDE_inscribed_circle_segment_ratio_l2641_264121


namespace NUMINAMATH_CALUDE_initial_breads_count_l2641_264119

/-- The number of thieves -/
def num_thieves : ℕ := 5

/-- The number of breads remaining after all thieves -/
def remaining_breads : ℕ := 3

/-- Function to calculate the number of breads after a thief takes their share -/
def breads_after_thief (x : ℚ) : ℚ := x / 2 - 1 / 2

/-- Function to calculate the number of breads after n thieves -/
def breads_after_n_thieves : ℕ → ℚ → ℚ
  | 0, x => x
  | n + 1, x => breads_after_n_thieves n (breads_after_thief x)

/-- Theorem stating that the initial number of breads was 127 -/
theorem initial_breads_count : 
  breads_after_n_thieves num_thieves 127 = remaining_breads := by sorry

end NUMINAMATH_CALUDE_initial_breads_count_l2641_264119


namespace NUMINAMATH_CALUDE_arithmetic_sequence_proof_l2641_264110

/-- Given a sequence of real numbers satisfying the condition
    |a_m + a_n - a_(m+n)| ≤ 1 / (m + n) for all m and n,
    prove that the sequence is arithmetic with a_k = k * a_1 for all k. -/
theorem arithmetic_sequence_proof (a : ℕ → ℝ) 
    (h : ∀ m n : ℕ, |a m + a n - a (m + n)| ≤ 1 / (m + n)) :
  ∀ k : ℕ, a k = k * a 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_proof_l2641_264110


namespace NUMINAMATH_CALUDE_lcm_of_5_6_8_21_l2641_264126

theorem lcm_of_5_6_8_21 : Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 8 21)) = 840 := by sorry

end NUMINAMATH_CALUDE_lcm_of_5_6_8_21_l2641_264126


namespace NUMINAMATH_CALUDE_product_of_place_values_l2641_264156

/-- The place value of a digit in a decimal number -/
def placeValue (digit : ℕ) (position : ℤ) : ℚ :=
  (digit : ℚ) * (10 : ℚ) ^ position

/-- The numeral under consideration -/
def numeral : ℚ := 7804830.88

/-- The product of place values of the three 8's in the numeral -/
def productOfPlaceValues : ℚ :=
  placeValue 8 5 * placeValue 8 1 * placeValue 8 (-2)

theorem product_of_place_values :
  productOfPlaceValues = 5120000 := by sorry

end NUMINAMATH_CALUDE_product_of_place_values_l2641_264156


namespace NUMINAMATH_CALUDE_magic_8_ball_probability_l2641_264131

/-- The probability of getting exactly 3 successes in 7 independent trials,
    where each trial has a success probability of 3/8. -/
theorem magic_8_ball_probability : 
  (Nat.choose 7 3 : ℚ) * (3/8)^3 * (5/8)^4 = 590625/2097152 := by
  sorry

end NUMINAMATH_CALUDE_magic_8_ball_probability_l2641_264131


namespace NUMINAMATH_CALUDE_departmental_store_average_salary_l2641_264134

def average_salary (num_managers : ℕ) (num_associates : ℕ) (avg_salary_managers : ℚ) (avg_salary_associates : ℚ) : ℚ :=
  let total_salary := num_managers * avg_salary_managers + num_associates * avg_salary_associates
  let total_employees := num_managers + num_associates
  total_salary / total_employees

theorem departmental_store_average_salary :
  average_salary 9 18 1300 12000 = 8433.33 := by
  sorry

end NUMINAMATH_CALUDE_departmental_store_average_salary_l2641_264134


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l2641_264186

theorem cubic_roots_sum (p q r : ℝ) : 
  p^3 - 8*p^2 + 10*p - 3 = 0 ∧ 
  q^3 - 8*q^2 + 10*q - 3 = 0 ∧ 
  r^3 - 8*r^2 + 10*r - 3 = 0 → 
  (p / (q*r + 2)) + (q / (p*r + 2)) + (r / (p*q + 2)) = 367/183 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l2641_264186


namespace NUMINAMATH_CALUDE_original_denominator_proof_l2641_264194

theorem original_denominator_proof (d : ℤ) : 
  (3 : ℚ) / d ≠ 0 →
  (3 + 7 : ℚ) / (d + 7) = 2 / 5 →
  d = 18 := by
sorry

end NUMINAMATH_CALUDE_original_denominator_proof_l2641_264194


namespace NUMINAMATH_CALUDE_polynomial_characterization_l2641_264189

-- Define a real polynomial
def RealPolynomial := ℝ → ℝ

-- Define the condition for a, b, c
def SatisfiesCondition (a b c : ℝ) : Prop := a * b + b * c + c * a = 0

-- Define the property that P must satisfy
def SatisfiesProperty (P : RealPolynomial) : Prop :=
  ∀ a b c : ℝ, SatisfiesCondition a b c →
    P (a - b) + P (b - c) + P (c - a) = 2 * P (a + b + c)

-- Define the form of the polynomial we want to prove
def IsQuarticQuadratic (P : RealPolynomial) : Prop :=
  ∃ α β : ℝ, ∀ x : ℝ, P x = α * x^4 + β * x^2

-- The main theorem
theorem polynomial_characterization (P : RealPolynomial) :
  SatisfiesProperty P → IsQuarticQuadratic P :=
sorry

end NUMINAMATH_CALUDE_polynomial_characterization_l2641_264189


namespace NUMINAMATH_CALUDE_regular_polygon_perimeters_l2641_264155

/-- Regular polygon perimeters for a unit circle -/
noncomputable def RegularPolygonPerimeters (n : ℕ) : ℝ × ℝ :=
  sorry

/-- Circumscribed polygon perimeter -/
noncomputable def P (n : ℕ) : ℝ := (RegularPolygonPerimeters n).1

/-- Inscribed polygon perimeter -/
noncomputable def p (n : ℕ) : ℝ := (RegularPolygonPerimeters n).2

theorem regular_polygon_perimeters :
  (P 4 = 8 ∧ p 4 = 4 * Real.sqrt 2 ∧ P 6 = 4 * Real.sqrt 3 ∧ p 6 = 6) ∧
  (∀ n ≥ 3, P (2 * n) = (2 * P n * p n) / (P n + p n) ∧
            p (2 * n) = Real.sqrt (p n * P (2 * n))) ∧
  (3^10 / 71 < Real.pi ∧ Real.pi < 22 / 7) :=
sorry

end NUMINAMATH_CALUDE_regular_polygon_perimeters_l2641_264155


namespace NUMINAMATH_CALUDE_finite_good_not_divisible_by_k_l2641_264170

/-- The number of divisors of an integer n -/
def τ (n : ℕ) : ℕ := sorry

/-- An integer n is "good" if for all m < n, we have τ(m) < τ(n) -/
def is_good (n : ℕ) : Prop :=
  ∀ m < n, τ m < τ n

/-- The set of good integers not divisible by k is finite -/
theorem finite_good_not_divisible_by_k (k : ℕ) (h : k ≥ 1) :
  {n : ℕ | is_good n ∧ ¬k ∣ n}.Finite :=
sorry

end NUMINAMATH_CALUDE_finite_good_not_divisible_by_k_l2641_264170


namespace NUMINAMATH_CALUDE_max_c_value_l2641_264180

-- Define the function f
def f (a b c d x : ℝ) : ℝ := a * x^3 + 2 * b * x^2 + 3 * c * x + 4 * d

-- Define the conditions
def is_valid_function (a b c d : ℝ) : Prop :=
  a < 0 ∧ c > 0 ∧
  (∀ x, f a b c d x = -f a b c d (-x)) ∧
  (∀ x ∈ Set.Icc 0 1, f a b c d x ∈ Set.Icc 0 1)

-- Theorem statement
theorem max_c_value (a b c d : ℝ) (h : is_valid_function a b c d) :
  c ≤ Real.sqrt 3 / 2 ∧ ∃ a₀ b₀ d₀, is_valid_function a₀ b₀ (Real.sqrt 3 / 2) d₀ :=
sorry

end NUMINAMATH_CALUDE_max_c_value_l2641_264180


namespace NUMINAMATH_CALUDE_congruence_solution_l2641_264154

theorem congruence_solution (m : ℕ) : m ∈ Finset.range 47 → (13 * m ≡ 9 [ZMOD 47]) ↔ m = 29 := by
  sorry

end NUMINAMATH_CALUDE_congruence_solution_l2641_264154


namespace NUMINAMATH_CALUDE_may_day_travelers_l2641_264114

def scientific_notation (n : ℕ) (c : ℝ) (e : ℤ) : Prop :=
  (1 ≤ c) ∧ (c < 10) ∧ (n = c * (10 : ℝ) ^ e)

theorem may_day_travelers :
  scientific_notation 213000000 2.13 8 :=
by sorry

end NUMINAMATH_CALUDE_may_day_travelers_l2641_264114


namespace NUMINAMATH_CALUDE_original_savings_l2641_264102

def lindas_savings : ℚ → Prop :=
  λ s => (1 / 4 : ℚ) * s = 450

theorem original_savings : ∃ s : ℚ, lindas_savings s ∧ s = 1800 :=
  sorry

end NUMINAMATH_CALUDE_original_savings_l2641_264102


namespace NUMINAMATH_CALUDE_tree_spacing_l2641_264125

theorem tree_spacing (yard_length : ℝ) (num_trees : ℕ) :
  yard_length = 350 →
  num_trees = 26 →
  yard_length / (num_trees - 1) = 14 :=
by sorry

end NUMINAMATH_CALUDE_tree_spacing_l2641_264125


namespace NUMINAMATH_CALUDE_polynomial_expansion_l2641_264171

theorem polynomial_expansion (x : ℝ) : 
  (x - 3) * (x + 5) * (x^2 + 9) = x^4 + 2*x^3 - 6*x^2 + 18*x - 135 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l2641_264171


namespace NUMINAMATH_CALUDE_vasily_expected_salary_l2641_264117

/-- Represents the salary distribution for graduates --/
structure SalaryDistribution where
  high : ℝ  -- Salary for 1/5 of graduates
  medium : ℝ  -- Salary for 1/10 of graduates
  low : ℝ  -- Salary for 1/20 of graduates
  default : ℝ  -- Salary for the rest

/-- Represents the given conditions of the problem --/
structure ProblemConditions where
  total_students : ℕ
  successful_graduates : ℕ
  non_graduate_salary : ℝ
  graduate_salary_dist : SalaryDistribution
  education_duration : ℕ

def expected_salary (conditions : ProblemConditions) : ℝ :=
  sorry

theorem vasily_expected_salary 
  (conditions : ProblemConditions)
  (h1 : conditions.total_students = 300)
  (h2 : conditions.successful_graduates = 270)
  (h3 : conditions.non_graduate_salary = 25000)
  (h4 : conditions.graduate_salary_dist.high = 60000)
  (h5 : conditions.graduate_salary_dist.medium = 80000)
  (h6 : conditions.graduate_salary_dist.low = 25000)
  (h7 : conditions.graduate_salary_dist.default = 40000)
  (h8 : conditions.education_duration = 4) :
  expected_salary conditions = 45025 :=
sorry

end NUMINAMATH_CALUDE_vasily_expected_salary_l2641_264117


namespace NUMINAMATH_CALUDE_ellipse_focus_distance_range_l2641_264135

/-- An ellipse with equation x²/4 + y²/t = 1 -/
structure Ellipse (t : ℝ) where
  x : ℝ
  y : ℝ
  eq : x^2/4 + y^2/t = 1

/-- The distance from a point on the ellipse to one of its foci -/
noncomputable def distance_to_focus (t : ℝ) (e : Ellipse t) : ℝ :=
  sorry  -- Definition omitted as it's not directly given in the problem

/-- The theorem stating the range of t for which the distance to a focus is always greater than 1 -/
theorem ellipse_focus_distance_range :
  ∀ t : ℝ, (∀ e : Ellipse t, distance_to_focus t e > 1) →
    t ∈ Set.union (Set.Ioo 3 4) (Set.Ioo 4 (25/4)) :=
sorry

end NUMINAMATH_CALUDE_ellipse_focus_distance_range_l2641_264135


namespace NUMINAMATH_CALUDE_inequality_proof_l2641_264178

theorem inequality_proof (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) 
  (h_sum_squares : a^2 + b^2 + c^2 = 1) : 
  (a * b / c) + (b * c / a) + (c * a / b) ≥ Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2641_264178


namespace NUMINAMATH_CALUDE_relay_team_orders_l2641_264100

/-- The number of permutations of n distinct objects -/
def permutations (n : ℕ) : ℕ := Nat.factorial n

/-- There are 5 team members -/
def team_size : ℕ := 5

/-- Lara always runs the last lap, so we need to arrange the other 4 runners -/
def runners_to_arrange : ℕ := team_size - 1

theorem relay_team_orders : permutations runners_to_arrange = 24 := by
  sorry

end NUMINAMATH_CALUDE_relay_team_orders_l2641_264100


namespace NUMINAMATH_CALUDE_problem_solution_l2641_264176

theorem problem_solution (x : ℝ) (n : ℝ) (h1 : x > 0) 
  (h2 : x / n + x / 25 = 0.24000000000000004 * x) : n = 5 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2641_264176


namespace NUMINAMATH_CALUDE_share_price_increase_l2641_264159

theorem share_price_increase (initial_price : ℝ) (q1_increase : ℝ) (q2_increase : ℝ) :
  q1_increase = 0.25 →
  q2_increase = 0.44 →
  ((initial_price * (1 + q1_increase) * (1 + q2_increase) - initial_price) / initial_price) = 0.80 :=
by sorry

end NUMINAMATH_CALUDE_share_price_increase_l2641_264159


namespace NUMINAMATH_CALUDE_smallest_satisfying_number_l2641_264138

/-- Returns the number of digits in a positive integer -/
def num_digits (n : ℕ) : ℕ :=
  if n < 10 then 1 else 1 + num_digits (n / 10)

/-- Returns the leftmost digit of a positive integer -/
def leftmost_digit (n : ℕ) : ℕ :=
  n / (10 ^ (num_digits n - 1))

/-- Returns the number after removing the leftmost digit -/
def remove_leftmost_digit (n : ℕ) : ℕ :=
  n % (10 ^ (num_digits n - 1))

/-- Checks if a number satisfies the condition -/
def satisfies_condition (n : ℕ) : Prop :=
  remove_leftmost_digit n = n / 19

theorem smallest_satisfying_number :
  ∀ n : ℕ, n > 0 → n < 1350 → ¬(satisfies_condition n) ∧ satisfies_condition 1350 :=
sorry

end NUMINAMATH_CALUDE_smallest_satisfying_number_l2641_264138


namespace NUMINAMATH_CALUDE_choir_members_l2641_264179

theorem choir_members (n : ℕ) : 
  50 ≤ n ∧ n ≤ 150 ∧ 
  n % 6 = 4 ∧ 
  n % 10 = 4 → 
  n = 64 ∨ n = 94 ∨ n = 124 := by
sorry

end NUMINAMATH_CALUDE_choir_members_l2641_264179


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l2641_264146

theorem quadratic_roots_property : 
  ∀ x₁ x₂ : ℝ, 
  x₁^2 - 3*x₁ + 1 = 0 → 
  x₂^2 - 3*x₂ + 1 = 0 → 
  x₁ ≠ x₂ → 
  x₁^2 + 3*x₂ + x₁*x₂ - 2 = 7 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l2641_264146


namespace NUMINAMATH_CALUDE_kilmer_park_tree_height_l2641_264148

/-- Calculates the height of a tree in inches after a given number of years -/
def tree_height_in_inches (initial_height : ℕ) (annual_growth : ℕ) (years : ℕ) : ℕ :=
  (initial_height + annual_growth * years) * 12

/-- Theorem: The height of the tree in Kilmer Park after 8 years is 1104 inches -/
theorem kilmer_park_tree_height : tree_height_in_inches 52 5 8 = 1104 := by
  sorry

end NUMINAMATH_CALUDE_kilmer_park_tree_height_l2641_264148


namespace NUMINAMATH_CALUDE_dividend_calculation_l2641_264175

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 14)
  (h2 : quotient = 9)
  (h3 : remainder = 5) :
  divisor * quotient + remainder = 131 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l2641_264175


namespace NUMINAMATH_CALUDE_solve_diamond_equation_l2641_264187

-- Define the binary operation ◊
noncomputable def diamond (a b : ℝ) : ℝ := a / b

-- State the properties of the operation
axiom diamond_prop1 (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  diamond a (diamond b c) = diamond a b * c

axiom diamond_prop2 (a : ℝ) (ha : a ≠ 0) :
  diamond a a = 1

-- State the theorem to be proved
theorem solve_diamond_equation :
  ∃ x : ℝ, x ≠ 0 ∧ diamond 2016 (diamond 6 x) = 100 ∧ x = 25 / 84 := by
  sorry

end NUMINAMATH_CALUDE_solve_diamond_equation_l2641_264187


namespace NUMINAMATH_CALUDE_cube_sum_greater_than_product_sum_l2641_264169

theorem cube_sum_greater_than_product_sum {a b : ℝ} (ha : 0 < a) (hb : 0 < b) (hab : a ≠ b) : 
  a^3 + b^3 > a^2 * b + a * b^2 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_greater_than_product_sum_l2641_264169


namespace NUMINAMATH_CALUDE_math_club_election_l2641_264139

theorem math_club_election (total_candidates : ℕ) (past_officers : ℕ) (positions : ℕ) :
  total_candidates = 20 →
  past_officers = 9 →
  positions = 6 →
  (Nat.choose total_candidates positions - Nat.choose (total_candidates - past_officers) positions) = 38298 :=
by sorry

end NUMINAMATH_CALUDE_math_club_election_l2641_264139


namespace NUMINAMATH_CALUDE_age_difference_proof_l2641_264184

def elder_age : ℕ := 30

theorem age_difference_proof (younger_age : ℕ) 
  (h1 : elder_age - 6 = 3 * (younger_age - 6)) : 
  elder_age - younger_age = 16 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_proof_l2641_264184


namespace NUMINAMATH_CALUDE_factorial_sum_equality_l2641_264128

theorem factorial_sum_equality : 7 * Nat.factorial 7 + 6 * Nat.factorial 6 + 2 * Nat.factorial 5 + Nat.factorial 5 = 39960 := by
  sorry

end NUMINAMATH_CALUDE_factorial_sum_equality_l2641_264128


namespace NUMINAMATH_CALUDE_not_sufficient_nor_necessary_l2641_264107

theorem not_sufficient_nor_necessary (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ¬(∀ a b : ℝ, a > b → (1 / a) < (1 / b)) ∧
  ¬(∀ a b : ℝ, (1 / a) < (1 / b) → a > b) :=
sorry

end NUMINAMATH_CALUDE_not_sufficient_nor_necessary_l2641_264107


namespace NUMINAMATH_CALUDE_largest_stamps_per_page_l2641_264181

theorem largest_stamps_per_page (a b c : ℕ) 
  (ha : a = 924) (hb : b = 1386) (hc : c = 1848) : 
  Nat.gcd a (Nat.gcd b c) = 462 := by
  sorry

end NUMINAMATH_CALUDE_largest_stamps_per_page_l2641_264181


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l2641_264115

theorem triangle_abc_properties (A B C : Real) (a b c : Real) (S : Real) :
  -- Given conditions
  (3 * Real.cos B * Real.cos C + 2 = 3 * Real.sin B * Real.sin C + 2 * Real.cos (2 * A)) →
  (S = 5 * Real.sqrt 3) →
  (b = 5) →
  -- Triangle inequality and positive side lengths
  (a + b > c ∧ b + c > a ∧ c + a > b) →
  (a > 0 ∧ b > 0 ∧ c > 0) →
  -- Angle sum in a triangle
  (A + B + C = Real.pi) →
  -- Area formula
  (S = 1/2 * b * c * Real.sin A) →
  -- Conclusions
  (A = Real.pi / 3 ∧ Real.sin B * Real.sin C = 5 / 7) := by
  sorry


end NUMINAMATH_CALUDE_triangle_abc_properties_l2641_264115


namespace NUMINAMATH_CALUDE_sum_of_roots_l2641_264174

theorem sum_of_roots (α β : ℝ) 
  (hα : α^3 - 3*α^2 + 5*α - 4 = 0)
  (hβ : β^3 - 3*β^2 + 5*β - 2 = 0) : 
  α + β = 2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l2641_264174


namespace NUMINAMATH_CALUDE_function_proof_l2641_264165

noncomputable section

-- Define the function f
def f (a b : ℝ) (x : ℝ) := a * x^3 + b * x^2 - 3 * x

-- Define the derivative of f
def f' (a b : ℝ) (x : ℝ) := 3 * a * x^2 + 2 * b * x - 3

-- Define the function g
def g (a b : ℝ) (x : ℝ) := (1/3) * f a b x - 6 * Real.log x

-- Define the curve y = xf(x)
def curve (a b : ℝ) (x : ℝ) := x * (f a b x)

theorem function_proof (a b : ℝ) :
  (∀ x, f' a b x = f' a b (-x)) →  -- f' is even
  f' a b 1 = 0 →                   -- f'(1) = 0
  (∃ c, ∀ x₁ x₂, x₁ ∈ Set.Icc 1 2 → x₂ ∈ Set.Icc 1 2 → 
    |g a b x₁ - g a b x₂| ≤ c) →   -- |g(x₁) - g(x₂)| ≤ c for x₁, x₂ ∈ [1, 2]
  (∀ x, f a b x = x^3 - 3*x) ∧     -- f(x) = x³ - 3x
  (∃ c_min, c_min = -4/3 + 6 * Real.log 2 ∧ 
    ∀ c', (∀ x₁ x₂, x₁ ∈ Set.Icc 1 2 → x₂ ∈ Set.Icc 1 2 → 
      |g a b x₁ - g a b x₂| ≤ c') → c' ≥ c_min) ∧  -- Minimum value of c
  (∃ s : Set ℝ, s = {4, 3/4 - 4 * Real.sqrt 2} ∧ 
    ∀ m, m ∈ s ↔ (∃ x₁ x₂ x₃, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
      curve a b x₁ = m * x₁ - 2 * m ∧
      curve a b x₂ = m * x₂ - 2 * m ∧
      curve a b x₃ = m * x₃ - 2 * m)) -- Set of m values for three tangent lines
  := by sorry

end NUMINAMATH_CALUDE_function_proof_l2641_264165


namespace NUMINAMATH_CALUDE_marys_income_percentage_l2641_264111

theorem marys_income_percentage (juan tim mary : ℝ) 
  (h1 : tim = juan * 0.9) 
  (h2 : mary = tim * 1.6) : 
  mary = juan * 1.44 := by
sorry

end NUMINAMATH_CALUDE_marys_income_percentage_l2641_264111


namespace NUMINAMATH_CALUDE_problem_solution_l2641_264182

noncomputable section

def f (x : ℝ) : ℝ := Real.log x + 1 / x
def g (x : ℝ) : ℝ := x - Real.log x

theorem problem_solution :
  (∃ (a : ℝ), ∀ (x : ℝ), x > 0 → f x ≥ a ∧ 
    ∀ (b : ℝ), (∀ (y : ℝ), y > 0 → f y ≥ b) → b ≤ a) ∧
  (∀ (x : ℝ), x > 1 → f x < g x) ∧
  (∀ (x₁ x₂ : ℝ), x₁ > x₂ ∧ x₂ > 0 ∧ g x₁ = g x₂ → x₁ * x₂ < 1) :=
by sorry

end

end NUMINAMATH_CALUDE_problem_solution_l2641_264182


namespace NUMINAMATH_CALUDE_container_fullness_l2641_264136

def container_capacity : ℝ := 120
def initial_fullness : ℝ := 0.35
def added_water : ℝ := 48

theorem container_fullness :
  let initial_water := initial_fullness * container_capacity
  let total_water := initial_water + added_water
  let final_fullness := total_water / container_capacity
  final_fullness = 0.75 := by sorry

end NUMINAMATH_CALUDE_container_fullness_l2641_264136


namespace NUMINAMATH_CALUDE_range_of_a_l2641_264120

-- Define the linear function
def f (a x : ℝ) : ℝ := (2 + a) * x + (5 - a)

-- Define the condition for the graph to pass through the first, second, and third quadrants
def passes_through_123_quadrants (a : ℝ) : Prop :=
  (2 + a > 0) ∧ (5 - a > 0)

-- Theorem statement
theorem range_of_a (a : ℝ) :
  passes_through_123_quadrants a → -2 < a ∧ a < 5 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2641_264120


namespace NUMINAMATH_CALUDE_lollipops_left_l2641_264183

theorem lollipops_left (initial : ℕ) (eaten : ℕ) (h1 : initial = 12) (h2 : eaten = 5) :
  initial - eaten = 7 := by
  sorry

end NUMINAMATH_CALUDE_lollipops_left_l2641_264183


namespace NUMINAMATH_CALUDE_baseball_average_runs_l2641_264144

theorem baseball_average_runs (games : ℕ) (runs_once : ℕ) (runs_twice : ℕ) (runs_thrice : ℕ)
  (h_games : games = 6)
  (h_once : runs_once = 1)
  (h_twice : runs_twice = 4)
  (h_thrice : runs_thrice = 5)
  (h_pattern : 1 * runs_once + 2 * runs_twice + 3 * runs_thrice = games * 4) :
  (1 * runs_once + 2 * runs_twice + 3 * runs_thrice) / games = 4 := by
sorry

end NUMINAMATH_CALUDE_baseball_average_runs_l2641_264144


namespace NUMINAMATH_CALUDE_volleyball_team_selection_l2641_264192

def total_players : ℕ := 16
def triplets : ℕ := 3
def team_size : ℕ := 7

def choose_with_triplets (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem volleyball_team_selection :
  (choose_with_triplets 3 1 * choose_with_triplets 13 6) +
  (choose_with_triplets 3 2 * choose_with_triplets 13 5) +
  (choose_with_triplets 3 3 * choose_with_triplets 13 4) = 9724 :=
by
  sorry

#check volleyball_team_selection

end NUMINAMATH_CALUDE_volleyball_team_selection_l2641_264192


namespace NUMINAMATH_CALUDE_continuous_function_property_l2641_264161

open Real Set

theorem continuous_function_property (d : ℝ) (h_d : d ∈ Ioc 0 1) :
  (∀ f : ℝ → ℝ, ContinuousOn f (Icc 0 1) → f 0 = f 1 →
    ∃ x₀ ∈ Icc 0 (1 - d), f x₀ = f (x₀ + d)) ↔
  ∃ k : ℕ, d = 1 / k :=
by sorry


end NUMINAMATH_CALUDE_continuous_function_property_l2641_264161


namespace NUMINAMATH_CALUDE_average_equals_seven_implies_x_equals_twelve_l2641_264106

theorem average_equals_seven_implies_x_equals_twelve :
  let numbers : List ℝ := [1, 2, 4, 5, 6, 9, 9, 10, 12, x]
  (List.sum numbers) / (List.length numbers) = 7 →
  x = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_average_equals_seven_implies_x_equals_twelve_l2641_264106


namespace NUMINAMATH_CALUDE_max_value_of_f_l2641_264168

-- Define the function
def f (x : ℝ) : ℝ := -4 * x^2 + 12 * x + 1

-- State the theorem
theorem max_value_of_f :
  ∃ (m : ℝ), (∀ (x : ℝ), f x ≤ m) ∧ (∃ (x : ℝ), f x = m) ∧ m = 10 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_f_l2641_264168


namespace NUMINAMATH_CALUDE_total_peppers_calculation_l2641_264116

/-- The amount of green peppers bought by Dale's Vegetarian Restaurant in pounds -/
def green_peppers : ℝ := 2.8333333333333335

/-- The amount of red peppers bought by Dale's Vegetarian Restaurant in pounds -/
def red_peppers : ℝ := 2.8333333333333335

/-- The total amount of peppers bought by Dale's Vegetarian Restaurant in pounds -/
def total_peppers : ℝ := green_peppers + red_peppers

theorem total_peppers_calculation :
  total_peppers = 5.666666666666667 := by sorry

end NUMINAMATH_CALUDE_total_peppers_calculation_l2641_264116


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l2641_264195

theorem fixed_point_on_line (m : ℝ) : (2 * m + 1) * 3 + (m + 1) * 1 - 7 * m - 4 = 0 := by
  sorry

#check fixed_point_on_line

end NUMINAMATH_CALUDE_fixed_point_on_line_l2641_264195


namespace NUMINAMATH_CALUDE_absolute_value_equation_l2641_264193

theorem absolute_value_equation (x z : ℝ) (h : |2*x - Real.sqrt z| = 2*x + Real.sqrt z) :
  x ≥ 0 ∧ z = 0 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_l2641_264193


namespace NUMINAMATH_CALUDE_solve_system_l2641_264190

theorem solve_system (x y : ℝ) 
  (eq1 : 3 * x - y = 7)
  (eq2 : x + 3 * y = 7) :
  x = 2.8 := by
sorry

end NUMINAMATH_CALUDE_solve_system_l2641_264190


namespace NUMINAMATH_CALUDE_loan_interest_time_l2641_264118

/-- 
Given:
- A loan of 1000 at 3% per year
- A loan of 1400 at 5% per year
- The total interest is 350

Prove that the number of years required for the total interest to reach 350 is 3.5
-/
theorem loan_interest_time (principal1 principal2 rate1 rate2 total_interest : ℝ) 
  (h1 : principal1 = 1000)
  (h2 : principal2 = 1400)
  (h3 : rate1 = 0.03)
  (h4 : rate2 = 0.05)
  (h5 : total_interest = 350) :
  (total_interest / (principal1 * rate1 + principal2 * rate2)) = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_loan_interest_time_l2641_264118


namespace NUMINAMATH_CALUDE_canoe_kayak_difference_l2641_264158

/-- Represents the rental information for canoes and kayaks --/
structure RentalInfo where
  canoe_cost : ℕ  -- Cost of renting a canoe per day
  kayak_cost : ℕ  -- Cost of renting a kayak per day
  canoe_kayak_ratio : Rat  -- Ratio of canoes to kayaks rented
  total_revenue : ℕ  -- Total revenue from rentals

/-- 
Given rental information, proves that the difference between 
the number of canoes and kayaks rented is 4
--/
theorem canoe_kayak_difference (info : RentalInfo) 
  (h1 : info.canoe_cost = 14)
  (h2 : info.kayak_cost = 15)
  (h3 : info.canoe_kayak_ratio = 3/2)
  (h4 : info.total_revenue = 288) :
  ∃ (c k : ℕ), c = k + 4 ∧ 
    c * info.canoe_cost + k * info.kayak_cost = info.total_revenue ∧
    (c : Rat) / k = info.canoe_kayak_ratio := by
  sorry

end NUMINAMATH_CALUDE_canoe_kayak_difference_l2641_264158


namespace NUMINAMATH_CALUDE_circular_seating_theorem_l2641_264164

/-- The number of people seated at a circular table. -/
def n : ℕ := sorry

/-- The distance between two positions in a circular arrangement. -/
def circularDistance (a b : ℕ) : ℕ :=
  min ((a - b + n) % n) ((b - a + n) % n)

/-- The theorem stating that if the distance from 31 to 7 equals the distance from 31 to 14
    in a circular arrangement of n people, then n must be 41. -/
theorem circular_seating_theorem :
  circularDistance 31 7 = circularDistance 31 14 → n = 41 := by
  sorry

end NUMINAMATH_CALUDE_circular_seating_theorem_l2641_264164
