import Mathlib

namespace NUMINAMATH_CALUDE_remainder_problem_l4004_400481

theorem remainder_problem (x : ℕ+) : (6 * x.val) % 9 = 3 → x.val % 9 = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l4004_400481


namespace NUMINAMATH_CALUDE_gcf_lcm_problem_l4004_400430

-- Define GCF (Greatest Common Factor)
def GCF (a b : ℕ) : ℕ := Nat.gcd a b

-- Define LCM (Least Common Multiple)
def LCM (a b : ℕ) : ℕ := Nat.lcm a b

-- Theorem statement
theorem gcf_lcm_problem : GCF (LCM 9 21) (LCM 8 15) = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcf_lcm_problem_l4004_400430


namespace NUMINAMATH_CALUDE_exists_congruent_triangle_with_same_color_on_sides_l4004_400449

/-- A color type with 1992 different colors -/
inductive Color : Type
| mk : Fin 1992 → Color

/-- A point in the plane -/
structure Point : Type :=
  (x y : ℝ)

/-- A triangle in the plane -/
structure Triangle : Type :=
  (a b c : Point)

/-- A coloring of the plane -/
def Coloring : Type := Point → Color

/-- A predicate to check if a point is on a line segment -/
def OnSegment (p q r : Point) : Prop := sorry

/-- A predicate to check if two triangles are congruent -/
def Congruent (t1 t2 : Triangle) : Prop := sorry

/-- Main theorem -/
theorem exists_congruent_triangle_with_same_color_on_sides
  (coloring : Coloring)
  (all_colors_used : ∀ c : Color, ∃ p : Point, coloring p = c)
  (t : Triangle) :
  ∃ t' : Triangle, Congruent t t' ∧
    ∃ (p1 p2 p3 : Point) (c : Color),
      OnSegment p1 t'.a t'.b ∧
      OnSegment p2 t'.b t'.c ∧
      OnSegment p3 t'.c t'.a ∧
      coloring p1 = c ∧
      coloring p2 = c ∧
      coloring p3 = c :=
sorry

end NUMINAMATH_CALUDE_exists_congruent_triangle_with_same_color_on_sides_l4004_400449


namespace NUMINAMATH_CALUDE_sin_130_equals_sin_50_l4004_400494

theorem sin_130_equals_sin_50 : Real.sin (130 * π / 180) = Real.sin (50 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_sin_130_equals_sin_50_l4004_400494


namespace NUMINAMATH_CALUDE_candy_distribution_l4004_400475

def possible_totals : Set ℕ := {18, 16, 14, 12}

def is_valid_distribution (vitya masha sasha : ℕ) : Prop :=
  vitya = 5 ∧ masha < vitya ∧ sasha = vitya + masha

theorem candy_distribution :
  ∀ total : ℕ,
  total ∈ possible_totals ↔
  ∃ vitya masha sasha : ℕ,
    is_valid_distribution vitya masha sasha ∧
    total = vitya + masha + sasha :=
by sorry

end NUMINAMATH_CALUDE_candy_distribution_l4004_400475


namespace NUMINAMATH_CALUDE_sum_of_perimeters_theorem_l4004_400493

/-- The sum of perimeters of all polygons in the sequence formed by repeatedly
    joining mid-points of an n-sided regular polygon with initial side length 60 cm. -/
def sum_of_perimeters (n : ℕ) : ℝ :=
  n * 120

/-- Theorem: The sum of perimeters of all polygons in the sequence formed by repeatedly
    joining mid-points of an n-sided regular polygon with initial side length 60 cm
    is equal to n * 120 cm. -/
theorem sum_of_perimeters_theorem (n : ℕ) (h : n > 0) :
  let initial_side_length : ℝ := 60
  let perimeter_sequence : ℕ → ℝ := λ k => n * (initial_side_length / 2^(k - 1))
  (∑' k, perimeter_sequence k) = sum_of_perimeters n :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_perimeters_theorem_l4004_400493


namespace NUMINAMATH_CALUDE_ratio_proof_l4004_400413

theorem ratio_proof (a b : ℕ+) (h1 : a.val * 4 = b.val * 3) (h2 : Nat.gcd a.val b.val = 5) (h3 : Nat.lcm a.val b.val = 60) : a.val * 4 = b.val * 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_proof_l4004_400413


namespace NUMINAMATH_CALUDE_simple_interest_calculation_l4004_400405

/-- Simple interest calculation -/
theorem simple_interest_calculation
  (principal : ℝ)
  (rate : ℝ)
  (time : ℝ)
  (h1 : principal = 10000)
  (h2 : rate = 0.09)
  (h3 : time = 1) :
  principal * rate * time = 900 :=
by sorry

end NUMINAMATH_CALUDE_simple_interest_calculation_l4004_400405


namespace NUMINAMATH_CALUDE_calculation_proof_l4004_400472

theorem calculation_proof : (-3) / (-1 - 3/4) * (3/4) / (3/7) = 3 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l4004_400472


namespace NUMINAMATH_CALUDE_horner_rule_V₁_l4004_400418

-- Define the polynomial coefficients
def a₄ : ℝ := 3
def a₃ : ℝ := 0
def a₂ : ℝ := 2
def a₁ : ℝ := 1
def a₀ : ℝ := 4

-- Define the x value
def x : ℝ := 10

-- Define Horner's Rule first step
def V₀ : ℝ := a₄

-- Define Horner's Rule second step (V₁)
def V₁ : ℝ := V₀ * x + a₃

-- Theorem statement
theorem horner_rule_V₁ : V₁ = 32 := by
  sorry

end NUMINAMATH_CALUDE_horner_rule_V₁_l4004_400418


namespace NUMINAMATH_CALUDE_curler_ratio_l4004_400466

theorem curler_ratio (total : ℕ) (pink : ℕ) (green : ℕ) (blue : ℕ) : 
  total = 16 → 
  pink = total / 4 → 
  green = 4 → 
  blue = total - pink - green →
  blue / pink = 2 := by
sorry

end NUMINAMATH_CALUDE_curler_ratio_l4004_400466


namespace NUMINAMATH_CALUDE_estevan_blanket_ratio_l4004_400437

/-- The ratio of polka-dot blankets to total blankets before Estevan's birthday -/
theorem estevan_blanket_ratio :
  let total_blankets : ℕ := 24
  let new_polka_dot_blankets : ℕ := 2
  let total_polka_dot_blankets : ℕ := 10
  let initial_polka_dot_blankets : ℕ := total_polka_dot_blankets - new_polka_dot_blankets
  (initial_polka_dot_blankets : ℚ) / total_blankets = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_estevan_blanket_ratio_l4004_400437


namespace NUMINAMATH_CALUDE_smallest_integer_y_minus_five_smallest_l4004_400401

theorem smallest_integer_y (y : ℤ) : (7 - 3 * y < 25) ↔ y ≥ -5 := by sorry

theorem minus_five_smallest : ∃ (y : ℤ), (7 - 3 * y < 25) ∧ (∀ (z : ℤ), z < y → (7 - 3 * z ≥ 25)) := by sorry

end NUMINAMATH_CALUDE_smallest_integer_y_minus_five_smallest_l4004_400401


namespace NUMINAMATH_CALUDE_toaster_pricing_theorem_l4004_400425

/-- Represents the relationship between cost and number of purchasers for toasters -/
def toaster_relation (c p : ℝ) : Prop := c * p = 6000

theorem toaster_pricing_theorem :
  -- Given condition
  toaster_relation 300 20 →
  -- Proofs to show
  (toaster_relation 600 10 ∧ toaster_relation 400 15) :=
by
  sorry

end NUMINAMATH_CALUDE_toaster_pricing_theorem_l4004_400425


namespace NUMINAMATH_CALUDE_smallest_divisible_by_1_to_12_l4004_400426

theorem smallest_divisible_by_1_to_12 : ∃ n : ℕ, n > 0 ∧ (∀ k : ℕ, 1 ≤ k ∧ k ≤ 12 → k ∣ n) ∧ (∀ m : ℕ, m > 0 ∧ (∀ k : ℕ, 1 ≤ k ∧ k ≤ 12 → k ∣ m) → n ≤ m) ∧ n = 27720 := by
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_1_to_12_l4004_400426


namespace NUMINAMATH_CALUDE_prob_wind_given_rain_l4004_400444

theorem prob_wind_given_rain (prob_rain prob_wind_and_rain : ℚ) 
  (h1 : prob_rain = 4/15)
  (h2 : prob_wind_and_rain = 1/10) :
  prob_wind_and_rain / prob_rain = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_prob_wind_given_rain_l4004_400444


namespace NUMINAMATH_CALUDE_michael_truck_meetings_l4004_400499

/-- Represents the number of meetings between Michael and the garbage truck --/
def number_of_meetings : ℕ := 7

/-- Michael's walking speed in feet per second --/
def michael_speed : ℝ := 6

/-- Distance between trash pails in feet --/
def pail_distance : ℝ := 200

/-- Garbage truck's speed in feet per second --/
def truck_speed : ℝ := 10

/-- Time the truck stops at each pail in seconds --/
def truck_stop_time : ℝ := 40

/-- Initial distance between Michael and the truck in feet --/
def initial_distance : ℝ := 250

/-- Theorem stating that Michael and the truck will meet 7 times --/
theorem michael_truck_meetings :
  ∃ (t : ℝ), t > 0 ∧
  (michael_speed * t = truck_speed * (t - truck_stop_time * (number_of_meetings - 1)) + initial_distance) :=
sorry

end NUMINAMATH_CALUDE_michael_truck_meetings_l4004_400499


namespace NUMINAMATH_CALUDE_g_one_value_l4004_400490

-- Define the polynomial f(x)
def f (a b c x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

-- Define the conditions
structure Conditions (a b c : ℝ) : Prop :=
  (a_lt_b : a < b)
  (b_lt_c : b < c)
  (one_lt_a : 1 < a)

-- Define the theorem
theorem g_one_value (a b c : ℝ) (h : Conditions a b c) :
  ∃ g : ℝ → ℝ,
    (∀ x, g x = 0 ↔ ∃ y, f a b c y = 0 ∧ x = 1 / y) →
    (∃ k, k ≠ 0 ∧ ∀ x, g x = k * (x^3 + (c/k)*x^2 + (b/k)*x + a/k)) →
    g 1 = (1 + a + b + c) / c :=
sorry

end NUMINAMATH_CALUDE_g_one_value_l4004_400490


namespace NUMINAMATH_CALUDE_power_division_l4004_400422

theorem power_division (n : ℕ) : n = 3^4053 → n / 3^2 = 3^4051 := by
  sorry

end NUMINAMATH_CALUDE_power_division_l4004_400422


namespace NUMINAMATH_CALUDE_car_speed_ratio_l4004_400415

-- Define the variables and constants
variable (v : ℝ) -- speed of car A
variable (k : ℝ) -- speed multiplier for car B
variable (AB CD AD : ℝ) -- distances

-- Define the theorem
theorem car_speed_ratio (h1 : k > 1) (h2 : AD = AB / 2) (h3 : CD / AD = 1 / 2) : k = 2 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_ratio_l4004_400415


namespace NUMINAMATH_CALUDE_system_solution_l4004_400434

/-- The system of equations:
    1. 3x² - xy = 1
    2. 9xy + y² = 22
    has exactly four solutions: (1,2), (-1,-2), (-1/6, 5.5), and (1/6, -5.5) -/
theorem system_solution :
  let f (x y : ℝ) := 3 * x^2 - x * y - 1
  let g (x y : ℝ) := 9 * x * y + y^2 - 22
  ∀ x y : ℝ, f x y = 0 ∧ g x y = 0 ↔
    (x = 1 ∧ y = 2) ∨
    (x = -1 ∧ y = -2) ∨
    (x = -1/6 ∧ y = 11/2) ∨
    (x = 1/6 ∧ y = -11/2) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l4004_400434


namespace NUMINAMATH_CALUDE_chocolate_cost_l4004_400474

theorem chocolate_cost (total_cost candy_price_difference : ℚ)
  (h1 : total_cost = 7)
  (h2 : candy_price_difference = 4) : 
  ∃ (chocolate_cost : ℚ), 
    chocolate_cost + (chocolate_cost + candy_price_difference) = total_cost ∧ 
    chocolate_cost = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_cost_l4004_400474


namespace NUMINAMATH_CALUDE_problem_solution_l4004_400484

theorem problem_solution (a b : ℚ) :
  (∀ x y : ℚ, y = a + b / x) →
  (2 = a + b / (-2)) →
  (3 = a + b / (-6)) →
  a + b = 13/2 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l4004_400484


namespace NUMINAMATH_CALUDE_tangent_perpendicular_implies_a_solutions_of_g_eq_zero_l4004_400467

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := -2 * x * Real.log x + x^2 - 2 * a * x + a^2

-- Define the derivative of f as g
def g (a : ℝ) (x : ℝ) : ℝ := -2 * (1 + Real.log x) + 2 * x - 2 * a

theorem tangent_perpendicular_implies_a (a : ℝ) :
  (g a 1 = -1) → a = -1/2 := by sorry

theorem solutions_of_g_eq_zero (a : ℝ) :
  (a < 0 → ∀ x, g a x ≠ 0) ∧
  (a = 0 → ∃! x, g a x = 0) ∧
  (a > 0 → ∃ x y, x ≠ y ∧ g a x = 0 ∧ g a y = 0) := by sorry

end

end NUMINAMATH_CALUDE_tangent_perpendicular_implies_a_solutions_of_g_eq_zero_l4004_400467


namespace NUMINAMATH_CALUDE_min_value_expression_l4004_400482

theorem min_value_expression (x y : ℝ) : (3*x*y - 1)^2 + (x - y)^2 ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l4004_400482


namespace NUMINAMATH_CALUDE_system_solution_unique_l4004_400496

theorem system_solution_unique (x y : ℝ) : 
  2 * x - 5 * y = 2 ∧ x + 3 * y = 12 ↔ x = 6 ∧ y = 2 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_unique_l4004_400496


namespace NUMINAMATH_CALUDE_sum_of_y_values_l4004_400469

theorem sum_of_y_values (x y : ℝ) : 
  x^2 + x^2*y^2 + x^2*y^4 = 525 ∧ x + x*y + x*y^2 = 35 →
  ∃ (y1 y2 : ℝ), y = y1 ∨ y = y2 ∧ y1 + y2 = 5/2 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_y_values_l4004_400469


namespace NUMINAMATH_CALUDE_complex_multiplication_division_l4004_400495

theorem complex_multiplication_division (P F G : ℂ) :
  P = 3 + 4 * Complex.I ∧
  F = -Complex.I ∧
  G = 3 - 4 * Complex.I →
  (P * F * G) / (-3 * Complex.I) = 25 / 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_division_l4004_400495


namespace NUMINAMATH_CALUDE_horner_method_operations_l4004_400478

/-- Horner's method for polynomial evaluation -/
def horner_eval (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The number of operations required by Horner's method -/
def horner_operations (n : ℕ) : ℕ × ℕ :=
  (n, n)

/-- The polynomial f(x) = 5x^5 + 4x^4 + 3x^3 + 2x^2 + x + 1 -/
def f_coeffs : List ℝ := [5, 4, 3, 2, 1, 1]

theorem horner_method_operations :
  let (mults, adds) := horner_operations (f_coeffs.length - 1)
  mults ≤ 5 ∧ adds = 5 := by sorry

end NUMINAMATH_CALUDE_horner_method_operations_l4004_400478


namespace NUMINAMATH_CALUDE_problem_statement_l4004_400417

theorem problem_statement : (2222 - 2002)^2 / 144 = 3025 / 9 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l4004_400417


namespace NUMINAMATH_CALUDE_work_completion_time_l4004_400438

theorem work_completion_time (a_time b_time : ℝ) (ha : a_time = 10) (hb : b_time = 10) :
  1 / (1 / a_time + 1 / b_time) = 5 := by sorry

end NUMINAMATH_CALUDE_work_completion_time_l4004_400438


namespace NUMINAMATH_CALUDE_decimal_23_to_binary_l4004_400454

-- Define a function to convert decimal to binary
def decimalToBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec toBinary (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: toBinary (m / 2)
  toBinary n

-- Theorem statement
theorem decimal_23_to_binary :
  decimalToBinary 23 = [true, true, true, false, true] := by
  sorry

end NUMINAMATH_CALUDE_decimal_23_to_binary_l4004_400454


namespace NUMINAMATH_CALUDE_sector_area_l4004_400436

theorem sector_area (r : ℝ) (θ : ℝ) (h : r = 2) (k : θ = π / 3) :
  (1 / 2) * r^2 * θ = (2 * π) / 3 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l4004_400436


namespace NUMINAMATH_CALUDE_rice_problem_l4004_400432

theorem rice_problem (total : ℚ) : 
  (21 : ℚ) / 50 * total = 210 → total = 500 := by
  sorry

end NUMINAMATH_CALUDE_rice_problem_l4004_400432


namespace NUMINAMATH_CALUDE_quadratic_roots_ratio_l4004_400462

theorem quadratic_roots_ratio (x₁ x₂ : ℝ) : 
  x₁^2 - 2*x₁ - 8 = 0 → x₂^2 - 2*x₂ - 8 = 0 → (x₁ + x₂) / (x₁ * x₂) = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_ratio_l4004_400462


namespace NUMINAMATH_CALUDE_playground_girls_l4004_400457

theorem playground_girls (total_children : ℕ) (boys : ℕ) (girls : ℕ) : 
  total_children = 117 → boys = 40 → girls = total_children - boys → girls = 77 := by
  sorry

end NUMINAMATH_CALUDE_playground_girls_l4004_400457


namespace NUMINAMATH_CALUDE_quadratic_equation_rational_solutions_l4004_400453

theorem quadratic_equation_rational_solutions : 
  ∃ (c₁ c₂ : ℕ+), 
    (∀ (c : ℕ+), (∃ (x : ℚ), 3 * x^2 + 7 * x + c.val = 0) ↔ (c = c₁ ∨ c = c₂)) ∧
    (c₁.val * c₂.val = 8) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_rational_solutions_l4004_400453


namespace NUMINAMATH_CALUDE_rogers_money_l4004_400470

/-- Roger's money calculation -/
theorem rogers_money (initial_amount spent_amount received_amount : ℕ) :
  initial_amount = 45 →
  spent_amount = 20 →
  received_amount = 46 →
  initial_amount - spent_amount + received_amount = 71 := by
  sorry

end NUMINAMATH_CALUDE_rogers_money_l4004_400470


namespace NUMINAMATH_CALUDE_function_identity_implies_zero_function_l4004_400400

def IsPositive (n : ℤ) : Prop := n > 0

theorem function_identity_implies_zero_function 
  (f : ℤ → ℝ) 
  (h : ∀ (n m : ℤ), IsPositive n → IsPositive m → n ≥ m → 
       f (n + m) + f (n - m) = f (3 * n)) :
  ∀ (n : ℤ), IsPositive n → f n = 0 := by
sorry

end NUMINAMATH_CALUDE_function_identity_implies_zero_function_l4004_400400


namespace NUMINAMATH_CALUDE_second_class_average_l4004_400414

theorem second_class_average (n₁ n₂ : ℕ) (avg₁ avg_total : ℚ) : 
  n₁ = 30 →
  n₂ = 50 →
  avg₁ = 50 →
  avg_total = 56.25 →
  (n₁ * avg₁ + n₂ * (n₁ * avg₁ + n₂ * avg_total - n₁ * avg₁) / n₂) / (n₁ + n₂) = avg_total →
  (n₁ * avg₁ + n₂ * avg_total - n₁ * avg₁) / n₂ = 60 := by
sorry

end NUMINAMATH_CALUDE_second_class_average_l4004_400414


namespace NUMINAMATH_CALUDE_increase_by_percentage_increase_500_by_30_percent_l4004_400423

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) : 
  initial * (1 + percentage / 100) = initial + initial * (percentage / 100) :=
by sorry

theorem increase_500_by_30_percent : 
  500 * (1 + 30 / 100) = 650 :=
by sorry

end NUMINAMATH_CALUDE_increase_by_percentage_increase_500_by_30_percent_l4004_400423


namespace NUMINAMATH_CALUDE_arithmetic_sequence_calculation_l4004_400448

theorem arithmetic_sequence_calculation : 
  let n := 2023
  let sum_to_n (k : ℕ) := k * (k + 1) / 2
  let diff_from_one_to (k : ℕ) := 1 - (sum_to_n k - 1)
  (diff_from_one_to (n - 1)) * (sum_to_n n - 1) - 
  (diff_from_one_to n) * (sum_to_n (n - 1) - 1) = n :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_calculation_l4004_400448


namespace NUMINAMATH_CALUDE_power_product_equals_four_l4004_400451

theorem power_product_equals_four (x y : ℝ) (h : x + 2 * y = 2) :
  (2 : ℝ) ^ x * (4 : ℝ) ^ y = 4 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equals_four_l4004_400451


namespace NUMINAMATH_CALUDE_wallet_problem_l4004_400456

/-- The number of quarters in the wallet -/
def num_quarters : ℕ := 15

/-- The number of dimes in the wallet -/
def num_dimes : ℕ := 25

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The number of dimes that equal the value of the quarters -/
def n : ℕ := 38

theorem wallet_problem :
  (num_quarters * quarter_value : ℕ) = n * dime_value :=
by sorry

end NUMINAMATH_CALUDE_wallet_problem_l4004_400456


namespace NUMINAMATH_CALUDE_triangle_area_l4004_400402

/-- Given a right triangle with side lengths in the ratio 2:3:4 inscribed in a circle of radius 4,
    prove that its area is 12. -/
theorem triangle_area (a b c : ℝ) (r : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Ensure positive side lengths
  r = 4 →  -- Circle radius
  a^2 + b^2 = c^2 →  -- Right triangle condition
  c = 2 * r →  -- Hypotenuse is diameter
  b / a = 3 / 2 →  -- Side ratio condition
  c / a = 2 →  -- Side ratio condition
  (1 / 2) * a * b = 12 :=  -- Area formula
by sorry

end NUMINAMATH_CALUDE_triangle_area_l4004_400402


namespace NUMINAMATH_CALUDE_digit_79_is_2_l4004_400480

/-- The sequence of digits obtained by concatenating consecutive integers from 60 down to 1 -/
def digit_sequence : List Nat := sorry

/-- The 79th digit in the sequence -/
def digit_79 : Nat := sorry

/-- Theorem stating that the 79th digit in the sequence is 2 -/
theorem digit_79_is_2 : digit_79 = 2 := by sorry

end NUMINAMATH_CALUDE_digit_79_is_2_l4004_400480


namespace NUMINAMATH_CALUDE_union_with_empty_set_l4004_400465

theorem union_with_empty_set (A B : Set ℕ) : 
  A = {1, 2} → B = ∅ → A ∪ B = {1, 2} := by sorry

end NUMINAMATH_CALUDE_union_with_empty_set_l4004_400465


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l4004_400492

/-- Given a geometric sequence {a_n} with the specified conditions, prove that a₆ + a₇ + a₈ = 32 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (h_geom : ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q)
  (h_sum1 : a 1 + a 2 + a 3 = 1)
  (h_sum2 : a 2 + a 3 + a 4 = 2) :
  a 6 + a 7 + a 8 = 32 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l4004_400492


namespace NUMINAMATH_CALUDE_calculation_proof_l4004_400476

theorem calculation_proof : 120 / (6 / 2) * 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l4004_400476


namespace NUMINAMATH_CALUDE_stork_comparison_l4004_400406

def initial_sparrows : ℕ := 12
def initial_pigeons : ℕ := 5
def initial_crows : ℕ := 9
def initial_storks : ℕ := 8
def additional_storks : ℕ := 15
def additional_pigeons : ℕ := 4

def final_storks : ℕ := initial_storks + additional_storks
def final_pigeons : ℕ := initial_pigeons + additional_pigeons
def final_other_birds : ℕ := initial_sparrows + final_pigeons + initial_crows

theorem stork_comparison : 
  (final_storks : ℤ) - (final_other_birds : ℤ) = -7 := by sorry

end NUMINAMATH_CALUDE_stork_comparison_l4004_400406


namespace NUMINAMATH_CALUDE_kennedy_benedict_house_difference_l4004_400450

theorem kennedy_benedict_house_difference (kennedy_house : ℕ) (benedict_house : ℕ)
  (h1 : kennedy_house = 10000)
  (h2 : benedict_house = 2350) :
  kennedy_house - 4 * benedict_house = 600 :=
by sorry

end NUMINAMATH_CALUDE_kennedy_benedict_house_difference_l4004_400450


namespace NUMINAMATH_CALUDE_emily_egg_collection_l4004_400410

theorem emily_egg_collection (total_baskets : ℕ) (first_group_baskets : ℕ) (second_group_baskets : ℕ)
  (eggs_per_first_basket : ℕ) (eggs_per_second_basket : ℕ) :
  total_baskets = first_group_baskets + second_group_baskets →
  first_group_baskets = 450 →
  second_group_baskets = 405 →
  eggs_per_first_basket = 36 →
  eggs_per_second_basket = 42 →
  first_group_baskets * eggs_per_first_basket + second_group_baskets * eggs_per_second_basket = 33210 := by
sorry

end NUMINAMATH_CALUDE_emily_egg_collection_l4004_400410


namespace NUMINAMATH_CALUDE_functional_equation_solution_l4004_400441

/-- The functional equation problem -/
theorem functional_equation_solution 
  (f : ℝ → ℝ) 
  (h : ∀ (x y : ℝ), x > 0 → y > 0 → f (x * y + f x) = x * f y + 2) : 
  ∀ (x : ℝ), x > 0 → f x = x + 1 := by
sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l4004_400441


namespace NUMINAMATH_CALUDE_function_value_difference_bound_l4004_400429

theorem function_value_difference_bound
  (f : Set.Icc 0 1 → ℝ)
  (h₁ : f ⟨0, by norm_num⟩ = f ⟨1, by norm_num⟩)
  (h₂ : ∀ (x₁ x₂ : Set.Icc 0 1), x₁ ≠ x₂ → |f x₂ - f x₁| < |x₂.val - x₁.val|) :
  ∀ (x₁ x₂ : Set.Icc 0 1), |f x₂ - f x₁| < (1/2 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_function_value_difference_bound_l4004_400429


namespace NUMINAMATH_CALUDE_midpoint_of_complex_line_segment_l4004_400455

theorem midpoint_of_complex_line_segment : 
  let z₁ : ℂ := 2 + 4 * Complex.I
  let z₂ : ℂ := -6 + 10 * Complex.I
  let midpoint := (z₁ + z₂) / 2
  midpoint = -2 + 7 * Complex.I := by
sorry

end NUMINAMATH_CALUDE_midpoint_of_complex_line_segment_l4004_400455


namespace NUMINAMATH_CALUDE_unique_solution_for_exponential_equation_l4004_400445

theorem unique_solution_for_exponential_equation :
  ∀ n m : ℕ+, 5^(n : ℕ) = 6*(m : ℕ)^2 + 1 ↔ n = 2 ∧ m = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_exponential_equation_l4004_400445


namespace NUMINAMATH_CALUDE_sphere_surface_area_l4004_400485

theorem sphere_surface_area (V : ℝ) (r : ℝ) (A : ℝ) : 
  V = 72 * Real.pi → 
  V = (4/3) * Real.pi * r^3 → 
  A = 4 * Real.pi * r^2 → 
  A = 36 * Real.pi * 2^(2/3) :=
by sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l4004_400485


namespace NUMINAMATH_CALUDE_rhombus_area_in_square_l4004_400498

/-- The area of a rhombus formed by intersecting equilateral triangles in a square -/
theorem rhombus_area_in_square (square_side : ℝ) (h_square_side : square_side = 4) :
  let triangle_height : ℝ := square_side * (Real.sqrt 3) / 2
  let rhombus_diagonal1 : ℝ := 2 * triangle_height - square_side
  let rhombus_diagonal2 : ℝ := square_side
  let rhombus_area : ℝ := (rhombus_diagonal1 * rhombus_diagonal2) / 2
  rhombus_area = 8 * Real.sqrt 3 - 8 := by
  sorry


end NUMINAMATH_CALUDE_rhombus_area_in_square_l4004_400498


namespace NUMINAMATH_CALUDE_integer_solutions_of_equation_l4004_400468

theorem integer_solutions_of_equation :
  ∀ x y : ℤ, x^4 = y^2 + 2*y + 2 ↔ (x = 1 ∧ y = -1) ∨ (x = -1 ∧ y = -1) := by
  sorry

end NUMINAMATH_CALUDE_integer_solutions_of_equation_l4004_400468


namespace NUMINAMATH_CALUDE_line_intersection_x_axis_l4004_400427

/-- A point in the 2D plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- A line defined by two points -/
structure Line :=
  (p1 : Point)
  (p2 : Point)

/-- The x-axis -/
def x_axis : Line :=
  { p1 := ⟨0, 0⟩, p2 := ⟨1, 0⟩ }

/-- Function to determine if a point lies on a given line -/
def point_on_line (p : Point) (l : Line) : Prop :=
  (p.y - l.p1.y) * (l.p2.x - l.p1.x) = (p.x - l.p1.x) * (l.p2.y - l.p1.y)

/-- Function to determine if a point lies on the x-axis -/
def point_on_x_axis (p : Point) : Prop :=
  p.y = 0

/-- The main theorem -/
theorem line_intersection_x_axis :
  let l : Line := { p1 := ⟨7, 3⟩, p2 := ⟨3, 7⟩ }
  let intersection_point : Point := ⟨10, 0⟩
  point_on_line intersection_point l ∧ point_on_x_axis intersection_point :=
by
  sorry

end NUMINAMATH_CALUDE_line_intersection_x_axis_l4004_400427


namespace NUMINAMATH_CALUDE_insurance_premium_theorem_l4004_400459

/-- Represents an insurance policy -/
structure InsurancePolicy where
  payout : ℝ  -- The amount paid out if the event occurs
  probability : ℝ  -- The probability of the event occurring
  premium : ℝ  -- The premium charged to the customer

/-- Calculates the expected revenue for an insurance policy -/
def expectedRevenue (policy : InsurancePolicy) : ℝ :=
  policy.premium - policy.payout * policy.probability

/-- Theorem: Given an insurance policy with payout 'a' and event probability 'p',
    if the company wants an expected revenue of 10% of 'a',
    then the required premium is a(p + 0.1) -/
theorem insurance_premium_theorem (a p : ℝ) (h1 : 0 ≤ p) (h2 : p ≤ 1) :
  let policy := InsurancePolicy.mk a p (a * (p + 0.1))
  expectedRevenue policy = 0.1 * a := by
  sorry

#check insurance_premium_theorem

end NUMINAMATH_CALUDE_insurance_premium_theorem_l4004_400459


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l4004_400479

theorem sufficient_not_necessary (x : ℝ) :
  (∀ x, (abs x = x → x^2 ≥ -x)) ∧
  (∃ x, x^2 ≥ -x ∧ abs x ≠ x) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l4004_400479


namespace NUMINAMATH_CALUDE_max_value_of_f_on_interval_l4004_400412

-- Define the function f(x) = x^2
def f (x : ℝ) : ℝ := x^2

-- State the theorem
theorem max_value_of_f_on_interval :
  ∃ (M : ℝ), M = 4 ∧ ∀ x ∈ Set.Icc (-1) 2, f x ≤ M :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_of_f_on_interval_l4004_400412


namespace NUMINAMATH_CALUDE_at_least_one_acute_angle_not_greater_than_45_l4004_400409

-- Define a right triangle ABC
structure RightTriangle where
  A : Real
  B : Real
  C : Real
  right_angle : C = 90
  angle_sum : A + B + C = 180

-- Theorem statement
theorem at_least_one_acute_angle_not_greater_than_45 (t : RightTriangle) :
  t.A ≤ 45 ∨ t.B ≤ 45 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_acute_angle_not_greater_than_45_l4004_400409


namespace NUMINAMATH_CALUDE_households_with_both_count_l4004_400447

/-- Represents the distribution of car and bike ownership in a neighborhood -/
structure Neighborhood where
  total : ℕ
  neither : ℕ
  with_car : ℕ
  only_bike : ℕ

/-- Calculates the number of households with both a car and a bike -/
def households_with_both (n : Neighborhood) : ℕ :=
  n.with_car - n.only_bike

/-- Theorem stating the number of households with both a car and a bike -/
theorem households_with_both_count (n : Neighborhood) 
  (h1 : n.total = 90)
  (h2 : n.neither = 11)
  (h3 : n.with_car = 44)
  (h4 : n.only_bike = 35)
  (h5 : n.total = n.neither + n.with_car + n.only_bike) :
  households_with_both n = 9 := by
  sorry

#eval households_with_both { total := 90, neither := 11, with_car := 44, only_bike := 35 }

end NUMINAMATH_CALUDE_households_with_both_count_l4004_400447


namespace NUMINAMATH_CALUDE_graph_is_pair_of_straight_lines_l4004_400421

/-- The equation of the graph -/
def equation (x y : ℝ) : Prop := x^2 - 9*y^2 = 0

/-- Definition of a straight line -/
def is_straight_line (f : ℝ → ℝ) : Prop :=
  ∃ m b : ℝ, ∀ x : ℝ, f x = m * x + b

/-- Theorem stating that the graph of x^2 - 9y^2 = 0 is a pair of straight lines -/
theorem graph_is_pair_of_straight_lines :
  ∃ f g : ℝ → ℝ, 
    (is_straight_line f ∧ is_straight_line g) ∧
    (∀ x y : ℝ, equation x y ↔ (y = f x ∨ y = g x)) :=
sorry

end NUMINAMATH_CALUDE_graph_is_pair_of_straight_lines_l4004_400421


namespace NUMINAMATH_CALUDE_wall_volume_is_86436_l4004_400435

def wall_volume (width : ℝ) : ℝ :=
  let height := 6 * width
  let length := 7 * height
  width * height * length

theorem wall_volume_is_86436 :
  wall_volume 7 = 86436 :=
by sorry

end NUMINAMATH_CALUDE_wall_volume_is_86436_l4004_400435


namespace NUMINAMATH_CALUDE_base3_subtraction_l4004_400483

/-- Represents a number in base 3 as a list of digits (least significant first) -/
def Base3 : Type := List Nat

/-- Converts a base 3 number to a natural number -/
def to_nat (b : Base3) : Nat :=
  b.enum.foldl (fun acc (i, d) => acc + d * (3 ^ i)) 0

/-- Adds two base 3 numbers -/
def add (a b : Base3) : Base3 :=
  sorry

/-- Subtracts two base 3 numbers -/
def sub (a b : Base3) : Base3 :=
  sorry

theorem base3_subtraction :
  let a : Base3 := [0, 1, 0]  -- 10₃
  let b : Base3 := [1, 0, 1, 1]  -- 1101₃
  let c : Base3 := [2, 0, 1, 2]  -- 2102₃
  let d : Base3 := [2, 1, 2]  -- 212₃
  let result : Base3 := [1, 0, 1, 1]  -- 1101₃
  sub (add (add a b) c) d = result := by
  sorry

end NUMINAMATH_CALUDE_base3_subtraction_l4004_400483


namespace NUMINAMATH_CALUDE_inequality_chain_l4004_400404

theorem inequality_chain (a b x : ℝ) (h1 : 0 < b) (h2 : b < x) (h3 : x < a) :
  b * x < x^2 ∧ x^2 < a^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_chain_l4004_400404


namespace NUMINAMATH_CALUDE_palmer_photos_l4004_400403

def total_photos (initial : ℕ) (first_week : ℕ) (third_fourth_week : ℕ) : ℕ :=
  initial + first_week + 2 * first_week + third_fourth_week

theorem palmer_photos : total_photos 100 50 80 = 330 := by
  sorry

end NUMINAMATH_CALUDE_palmer_photos_l4004_400403


namespace NUMINAMATH_CALUDE_sum_of_roots_equal_l4004_400452

-- Define the original equation
def original_equation (x : ℝ) : Prop :=
  10 = (x^3 - 3*x^2 - 4*x) / (x + 3)

-- Define the derived polynomial
def derived_polynomial (x : ℝ) : ℝ :=
  x^3 - 3*x^2 - 14*x - 30

-- Theorem statement
theorem sum_of_roots_equal :
  ∃ (r₁ r₂ r₃ : ℝ),
    (∀ x, derived_polynomial x = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃) →
    (∀ x, original_equation x ↔ x = r₁ ∨ x = r₂ ∨ x = r₃) →
    r₁ + r₂ + r₃ = 3 := by sorry

end NUMINAMATH_CALUDE_sum_of_roots_equal_l4004_400452


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_53_ending_3_l4004_400486

theorem smallest_four_digit_divisible_by_53_ending_3 :
  ∀ n : ℕ,
  1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ n % 10 = 3 →
  1113 ≤ n :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_53_ending_3_l4004_400486


namespace NUMINAMATH_CALUDE_johns_candy_store_spending_l4004_400491

theorem johns_candy_store_spending (allowance : ℚ) (arcade_fraction : ℚ) (toy_fraction : ℚ) :
  allowance = 3.60 ∧ 
  arcade_fraction = 3/5 ∧ 
  toy_fraction = 1/3 →
  allowance * (1 - arcade_fraction) * (1 - toy_fraction) = 0.96 := by
  sorry

end NUMINAMATH_CALUDE_johns_candy_store_spending_l4004_400491


namespace NUMINAMATH_CALUDE_johns_expenses_l4004_400431

/-- Given that John spent 40% of his earnings on rent and had 32% left over,
    prove that he spent 30% less on the dishwasher compared to the rent. -/
theorem johns_expenses (earnings : ℝ) (rent_percent : ℝ) (leftover_percent : ℝ)
  (h1 : rent_percent = 40)
  (h2 : leftover_percent = 32) :
  let dishwasher_percent := 100 - rent_percent - leftover_percent
  (rent_percent - dishwasher_percent) / rent_percent * 100 = 30 := by
  sorry

end NUMINAMATH_CALUDE_johns_expenses_l4004_400431


namespace NUMINAMATH_CALUDE_prob_B_not_lose_l4004_400424

/-- The probability of player A winning in a chess game -/
def prob_A_win : ℝ := 0.3

/-- The probability of a draw in a chess game -/
def prob_draw : ℝ := 0.5

/-- Theorem: The probability of player B not losing in a chess game -/
theorem prob_B_not_lose : prob_A_win + prob_draw + (1 - prob_A_win - prob_draw) = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_prob_B_not_lose_l4004_400424


namespace NUMINAMATH_CALUDE_max_blocks_in_box_l4004_400497

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a rectangular object given its dimensions -/
def volume (d : Dimensions) : ℕ := d.length * d.width * d.height

def box : Dimensions := ⟨3, 4, 3⟩
def block : Dimensions := ⟨3, 1, 1⟩

/-- The maximum number of blocks that can fit in the box -/
def max_blocks : ℕ := volume box / volume block

theorem max_blocks_in_box : max_blocks = 12 := by sorry

end NUMINAMATH_CALUDE_max_blocks_in_box_l4004_400497


namespace NUMINAMATH_CALUDE_age_difference_is_51_l4004_400420

/-- The age difference between Milena's cousin Alex and her grandfather -/
def age_difference : ℕ :=
  let milena_age : ℕ := 7
  let grandmother_age : ℕ := 9 * milena_age
  let grandfather_age : ℕ := grandmother_age + 2
  let alex_age : ℕ := 2 * milena_age
  grandfather_age - alex_age

theorem age_difference_is_51 : age_difference = 51 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_is_51_l4004_400420


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l4004_400442

/-- Given a geometric sequence {a_n} where a_3 = 6 and the sum of the first three terms S_3 = 18,
    prove that the common ratio q is either 1 or -1/2. -/
theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- Definition of geometric sequence
  a 3 = 6 →                     -- Third term is 6
  a 1 + a 2 + a 3 = 18 →        -- Sum of first three terms is 18
  q = 1 ∨ q = -1/2 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l4004_400442


namespace NUMINAMATH_CALUDE_count_digit_six_is_280_l4004_400411

/-- Count of digit 6 in integers from 100 to 999 -/
def count_digit_six : ℕ :=
  let hundreds := 100  -- 600 to 699
  let tens := 9 * 10   -- 10 numbers per hundred, 9 hundreds
  let ones := 9 * 10   -- 10 numbers per hundred, 9 hundreds
  hundreds + tens + ones

/-- The count of digit 6 in integers from 100 to 999 is 280 -/
theorem count_digit_six_is_280 : count_digit_six = 280 := by
  sorry

end NUMINAMATH_CALUDE_count_digit_six_is_280_l4004_400411


namespace NUMINAMATH_CALUDE_circle_equation_l4004_400416

-- Define the circle C
def Circle (center : ℝ × ℝ) (radius : ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the line 4x + 3y = 0
def Line : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 4 * p.1 + 3 * p.2 = 0}

-- Define the y-axis
def YAxis : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = 0}

-- Theorem statement
theorem circle_equation :
  ∀ (C : Set (ℝ × ℝ)) (center : ℝ × ℝ),
    -- Conditions
    C = Circle center 1 →  -- Radius is 1
    center.1 < 0 ∧ center.2 > 0 →  -- Center is in second quadrant
    ∃ (p : ℝ × ℝ), p ∈ C ∩ Line →  -- Tangent to 4x + 3y = 0
    ∃ (q : ℝ × ℝ), q ∈ C ∩ YAxis →  -- Tangent to y-axis
    -- Conclusion
    C = {p : ℝ × ℝ | (p.1 + 1)^2 + (p.2 - 3)^2 = 1} :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l4004_400416


namespace NUMINAMATH_CALUDE_length_width_ratio_l4004_400407

-- Define the rectangle
def rectangle (width : ℝ) (length : ℝ) : Prop :=
  width > 0 ∧ length > 0

-- Define the area of the rectangle
def area (width : ℝ) (length : ℝ) : ℝ :=
  width * length

-- Theorem statement
theorem length_width_ratio (width : ℝ) (length : ℝ) :
  rectangle width length →
  width = 6 →
  area width length = 108 →
  length / width = 3 := by
  sorry


end NUMINAMATH_CALUDE_length_width_ratio_l4004_400407


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l4004_400439

theorem polynomial_division_remainder : ∃ q : Polynomial ℚ, 
  (X^5 - 1) * (X^3 - 1) = (X^3 + X^2 + 1) * q + (-2*X^2 + X + 1) := by sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l4004_400439


namespace NUMINAMATH_CALUDE_smallest_greater_perfect_square_l4004_400460

theorem smallest_greater_perfect_square (a : ℕ) (h : ∃ k : ℕ, a = k^2) :
  (∀ n : ℕ, n > a ∧ (∃ m : ℕ, n = m^2) → n ≥ a + 2*Int.sqrt a + 1) ∧
  (∃ m : ℕ, a + 2*Int.sqrt a + 1 = m^2) :=
sorry

end NUMINAMATH_CALUDE_smallest_greater_perfect_square_l4004_400460


namespace NUMINAMATH_CALUDE_orange_apple_difference_l4004_400473

def apples : ℕ := 14
def dozen : ℕ := 12
def oranges : ℕ := 2 * dozen

theorem orange_apple_difference : oranges - apples = 10 := by
  sorry

end NUMINAMATH_CALUDE_orange_apple_difference_l4004_400473


namespace NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l4004_400428

-- Define the arithmetic sequence
def arithmetic_seq (a₁ a₂ : ℝ) : Prop :=
  ∃ d : ℝ, a₂ - a₁ = d ∧ a₁ - (-2) = d ∧ (-8) - a₂ = d

-- Define the geometric sequence
def geometric_seq (b₁ b₂ b₃ : ℝ) : Prop :=
  ∃ r : ℝ, b₁ / (-2) = r ∧ b₂ / b₁ = r ∧ b₃ / b₂ = r ∧ (-8) / b₃ = r

theorem arithmetic_geometric_ratio
  (a₁ a₂ b₁ b₂ b₃ : ℝ)
  (h_arith : arithmetic_seq a₁ a₂)
  (h_geom : geometric_seq b₁ b₂ b₃) :
  (a₂ - a₁) / b₂ = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l4004_400428


namespace NUMINAMATH_CALUDE_unique_solution_x4_y2_71_l4004_400464

theorem unique_solution_x4_y2_71 :
  ∀ x y : ℕ+, x^4 = y^2 + 71 → x = 6 ∧ y = 35 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_x4_y2_71_l4004_400464


namespace NUMINAMATH_CALUDE_cube_edge_length_l4004_400443

theorem cube_edge_length (surface_area : ℝ) (edge_length : ℝ) : 
  surface_area = 54 → 
  surface_area = 6 * edge_length ^ 2 → 
  edge_length = 3 := by
  sorry

end NUMINAMATH_CALUDE_cube_edge_length_l4004_400443


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l4004_400489

theorem polynomial_division_remainder :
  ∃ q : Polynomial ℝ, (X^6 - 1) * (X^2 - 1) = (X^3 - 1) * q + (X^2 - 1) :=
sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l4004_400489


namespace NUMINAMATH_CALUDE_shirt_profit_theorem_l4004_400471

/-- Represents the daily profit function for a shirt department -/
def daily_profit (initial_sales : ℕ) (initial_profit : ℝ) (price_reduction : ℝ) : ℝ :=
  (initial_profit - price_reduction) * (initial_sales + 2 * price_reduction)

theorem shirt_profit_theorem 
  (initial_sales : ℕ) 
  (initial_profit : ℝ) 
  (h_initial_sales : initial_sales = 30)
  (h_initial_profit : initial_profit = 40) :
  (∃ (x : ℝ), daily_profit initial_sales initial_profit x = 1200) ∧
  (∀ (y : ℝ), daily_profit initial_sales initial_profit y ≠ 1600) :=
sorry

#check shirt_profit_theorem

end NUMINAMATH_CALUDE_shirt_profit_theorem_l4004_400471


namespace NUMINAMATH_CALUDE_triangle_is_obtuse_l4004_400461

theorem triangle_is_obtuse (a b c : ℝ) (h_a : a = 5) (h_b : b = 6) (h_c : c = 8) :
  c^2 > a^2 + b^2 := by
  sorry

#check triangle_is_obtuse

end NUMINAMATH_CALUDE_triangle_is_obtuse_l4004_400461


namespace NUMINAMATH_CALUDE_counterexample_exists_l4004_400477

theorem counterexample_exists : ∃ x : ℝ, x > 1 ∧ x + 1 / (x - 1) ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l4004_400477


namespace NUMINAMATH_CALUDE_number_of_boys_l4004_400487

theorem number_of_boys (total_pupils : ℕ) (girls : ℕ) (teachers : ℕ) 
  (h1 : total_pupils = 626) 
  (h2 : girls = 308) 
  (h3 : teachers = 36) : 
  total_pupils - girls - teachers = 282 := by
  sorry

end NUMINAMATH_CALUDE_number_of_boys_l4004_400487


namespace NUMINAMATH_CALUDE_katie_game_difference_l4004_400433

theorem katie_game_difference (katie_games friends_games : ℕ) 
  (h1 : katie_games = 81) (h2 : friends_games = 59) : 
  katie_games - friends_games = 22 := by
sorry

end NUMINAMATH_CALUDE_katie_game_difference_l4004_400433


namespace NUMINAMATH_CALUDE_largest_n_for_sin_cos_inequality_l4004_400408

open Real

theorem largest_n_for_sin_cos_inequality :
  ∃ (n : ℕ), n = 3 ∧
  (∀ x : ℝ, 0 < x ∧ x < π / 2 → sin x ^ n + cos x ^ n > 1 / 2) ∧
  ¬(∀ x : ℝ, 0 < x ∧ x < π / 2 → sin x ^ (n + 1) + cos x ^ (n + 1) > 1 / 2) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_for_sin_cos_inequality_l4004_400408


namespace NUMINAMATH_CALUDE_prob_at_least_two_women_l4004_400419

/-- The probability of selecting at least two women from a group of 9 men and 6 women when choosing 4 people at random -/
theorem prob_at_least_two_women (num_men : ℕ) (num_women : ℕ) (num_selected : ℕ) : 
  num_men = 9 → num_women = 6 → num_selected = 4 →
  (Nat.choose (num_men + num_women) num_selected - 
   (Nat.choose num_men num_selected + 
    num_women * Nat.choose num_men (num_selected - 1))) / 
  Nat.choose (num_men + num_women) num_selected = 7 / 13 :=
by sorry

end NUMINAMATH_CALUDE_prob_at_least_two_women_l4004_400419


namespace NUMINAMATH_CALUDE_art_gallery_problem_l4004_400463

theorem art_gallery_problem (total_pieces : ℕ) 
  (h1 : total_pieces / 3 = total_pieces - (total_pieces * 2 / 3))  -- 1/3 of pieces are displayed
  (h2 : (total_pieces / 3) / 6 = (total_pieces / 3) - (5 * total_pieces / 18))  -- 1/6 of displayed pieces are sculptures
  (h3 : (total_pieces * 2 / 3) / 3 = (total_pieces * 2 / 3) - (2 * total_pieces / 3))  -- 1/3 of not displayed pieces are paintings
  (h4 : 2 * (total_pieces * 2 / 3) / 3 = 1200)  -- 1200 sculptures are not on display
  : total_pieces = 2700 := by
  sorry

end NUMINAMATH_CALUDE_art_gallery_problem_l4004_400463


namespace NUMINAMATH_CALUDE_initial_milk_water_ratio_l4004_400488

/-- Given a mixture of milk and water, proves that the initial ratio is 4:1 --/
theorem initial_milk_water_ratio 
  (total_volume : ℝ) 
  (added_water : ℝ) 
  (final_ratio : ℝ) 
  (h1 : total_volume = 45) 
  (h2 : added_water = 11) 
  (h3 : final_ratio = 1.8) : 
  ∃ (milk water : ℝ), 
    milk + water = total_volume ∧ 
    milk / (water + added_water) = final_ratio ∧ 
    milk / water = 4 := by
sorry

end NUMINAMATH_CALUDE_initial_milk_water_ratio_l4004_400488


namespace NUMINAMATH_CALUDE_discount_calculation_l4004_400440

-- Define the discount rate
def discount_rate : ℝ := 0.2

-- Define the original price (can be any positive real number)
variable (original_price : ℝ)
variable (original_price_positive : original_price > 0)

-- Define the purchase price
def purchase_price (original_price : ℝ) : ℝ := original_price * (1 - discount_rate)

-- Define the selling price
def selling_price (original_price : ℝ) : ℝ := original_price * 1.24

-- Theorem statement
theorem discount_calculation (original_price : ℝ) (original_price_positive : original_price > 0) :
  selling_price original_price = purchase_price original_price * 1.55 := by
  sorry

#check discount_calculation

end NUMINAMATH_CALUDE_discount_calculation_l4004_400440


namespace NUMINAMATH_CALUDE_cone_height_ratio_l4004_400446

theorem cone_height_ratio (original_circumference : ℝ) (original_height : ℝ) (new_volume : ℝ) :
  original_circumference = 20 * Real.pi →
  original_height = 40 →
  new_volume = 800 * Real.pi →
  ∃ (new_height : ℝ),
    (1 / 3) * Real.pi * (original_circumference / (2 * Real.pi))^2 * new_height = new_volume ∧
    new_height / original_height = 3 / 5 :=
by sorry

end NUMINAMATH_CALUDE_cone_height_ratio_l4004_400446


namespace NUMINAMATH_CALUDE_union_of_sets_l4004_400458

def A (a : ℝ) : Set ℝ := {0, a}
def B (a : ℝ) : Set ℝ := {3^a, 1}

theorem union_of_sets (a : ℝ) (h : A a ∩ B a = {1}) : A a ∪ B a = {0, 1, 3} := by
  sorry

end NUMINAMATH_CALUDE_union_of_sets_l4004_400458
