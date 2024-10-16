import Mathlib

namespace NUMINAMATH_CALUDE_sin_right_angle_l579_57934

theorem sin_right_angle (D E F : ℝ) (h1 : D = 90) (h2 : DE = 12) (h3 : EF = 35) : Real.sin D = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_right_angle_l579_57934


namespace NUMINAMATH_CALUDE_greatest_prime_factor_eleven_l579_57909

def f (m : ℕ) : ℕ := Finset.prod (Finset.range (m/2)) (fun i => 2*(i+1))

theorem greatest_prime_factor_eleven (m : ℕ) (h1 : m > 0) (h2 : Even m) :
  (∀ p : ℕ, Prime p → p ∣ f m → p ≤ 11) ∧
  (11 ∣ f m) →
  m = 22 := by sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_eleven_l579_57909


namespace NUMINAMATH_CALUDE_expression_value_l579_57922

theorem expression_value : (4 - 2)^3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l579_57922


namespace NUMINAMATH_CALUDE_books_left_over_l579_57951

/-- Calculates the number of books left over after filling a bookcase -/
theorem books_left_over
  (initial_books : ℕ)
  (shelves : ℕ)
  (books_per_shelf : ℕ)
  (new_books : ℕ)
  (h1 : initial_books = 56)
  (h2 : shelves = 4)
  (h3 : books_per_shelf = 20)
  (h4 : new_books = 26) :
  initial_books + new_books - (shelves * books_per_shelf) = 2 :=
by
  sorry

#check books_left_over

end NUMINAMATH_CALUDE_books_left_over_l579_57951


namespace NUMINAMATH_CALUDE_mean_equality_implies_z_value_l579_57963

theorem mean_equality_implies_z_value : 
  (∃ z : ℝ, (8 + 10 + 24) / 3 = (16 + z) / 2) → 
  (∃ z : ℝ, (8 + 10 + 24) / 3 = (16 + z) / 2 ∧ z = 12) :=
by sorry

end NUMINAMATH_CALUDE_mean_equality_implies_z_value_l579_57963


namespace NUMINAMATH_CALUDE_eliminate_y_implies_opposite_coefficients_l579_57984

/-- Given a system of linear equations in two variables x and y,
    prove that if the sum of the equations directly eliminates y,
    then the coefficients of y in the two equations are opposite numbers. -/
theorem eliminate_y_implies_opposite_coefficients 
  (a b c d : ℝ) (k₁ k₂ : ℝ) : 
  (∀ x y : ℝ, a * x + b * y = k₁ ∧ c * x + d * y = k₂) →
  (∀ x : ℝ, (a + c) * x = k₁ + k₂) →
  b + d = 0 :=
sorry

end NUMINAMATH_CALUDE_eliminate_y_implies_opposite_coefficients_l579_57984


namespace NUMINAMATH_CALUDE_roots_sum_powers_l579_57906

theorem roots_sum_powers (α β : ℝ) : 
  α^2 - 4*α + 1 = 0 → β^2 - 4*β + 1 = 0 → 7*α^3 + 3*β^4 = 1019 := by
sorry

end NUMINAMATH_CALUDE_roots_sum_powers_l579_57906


namespace NUMINAMATH_CALUDE_x_twelfth_power_l579_57968

theorem x_twelfth_power (x : ℝ) (h : x + 1/x = 2) : x^12 = 1 := by
  sorry

end NUMINAMATH_CALUDE_x_twelfth_power_l579_57968


namespace NUMINAMATH_CALUDE_max_value_theorem_l579_57931

theorem max_value_theorem (x y a b : ℝ) 
  (ha : a > 1) (hb : b > 1)
  (hax : a^x = 2) (hby : b^y = 2) (hab : a + Real.sqrt b = 4) :
  ∃ (M : ℝ), M = 4 ∧ ∀ (z : ℝ), (2/x + 1/y) ≤ z → z ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l579_57931


namespace NUMINAMATH_CALUDE_complex_number_properties_l579_57972

theorem complex_number_properties (z : ℂ) (h : z * (1 + Complex.I) = 2) :
  (Complex.abs z = Real.sqrt 2) ∧
  (∀ p : ℝ, z^2 - p*z + 2 = 0 → p = 2) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_properties_l579_57972


namespace NUMINAMATH_CALUDE_flower_count_l579_57989

theorem flower_count (bees : ℕ) (diff : ℕ) : bees = 3 → diff = 2 → bees + diff = 5 := by
  sorry

end NUMINAMATH_CALUDE_flower_count_l579_57989


namespace NUMINAMATH_CALUDE_camping_trip_percentage_l579_57912

theorem camping_trip_percentage (total_students : ℕ) 
  (h1 : (16 : ℝ) / 100 * total_students = (25 : ℝ) / 100 * (64 : ℝ) / 100 * total_students)
  (h2 : (75 : ℝ) / 100 * (64 : ℝ) / 100 * total_students = (64 : ℝ) / 100 * total_students - (16 : ℝ) / 100 * total_students) :
  (64 : ℝ) / 100 * total_students = (64 : ℝ) / 100 * total_students :=
by sorry

end NUMINAMATH_CALUDE_camping_trip_percentage_l579_57912


namespace NUMINAMATH_CALUDE_coefficient_of_x_squared_l579_57914

theorem coefficient_of_x_squared (x : ℝ) : 
  let expr := 2*(x^2 - 5) + 6*(3*x^2 - 2*x + 4) - 4*(x^2 - 3*x)
  ∃ (a b c : ℝ), expr = 16*x^2 + a*x + b + c*x^3 :=
by sorry

end NUMINAMATH_CALUDE_coefficient_of_x_squared_l579_57914


namespace NUMINAMATH_CALUDE_no_real_solutions_condition_l579_57901

theorem no_real_solutions_condition (a : ℝ) : 
  (∀ x : ℝ, (a^2 + 2*a)*x^2 + 3*a*x + 1 ≠ 0) ↔ (0 < a ∧ a < 8/5) :=
by sorry

end NUMINAMATH_CALUDE_no_real_solutions_condition_l579_57901


namespace NUMINAMATH_CALUDE_tangent_line_at_point_l579_57900

/-- The equation of the tangent line to y = 2x² at (1, 2) is y = 4x - 2 -/
theorem tangent_line_at_point (x y : ℝ) :
  (y = 2 * x^2) →  -- Given curve
  (∃ P : ℝ × ℝ, P = (1, 2)) →  -- Given point
  (∃ m b : ℝ, ∀ x y : ℝ, y = m * x + b ∧ y = 4 * x - 2) -- Tangent line equation
  := by sorry

end NUMINAMATH_CALUDE_tangent_line_at_point_l579_57900


namespace NUMINAMATH_CALUDE_star_equality_implies_x_equals_six_l579_57907

-- Define the binary operation ★
def star (a b c d : ℤ) : ℤ × ℤ := (a + c, b - d)

theorem star_equality_implies_x_equals_six :
  ∀ x y : ℤ, star 5 5 2 2 = star x y 1 3 → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_star_equality_implies_x_equals_six_l579_57907


namespace NUMINAMATH_CALUDE_sin_585_degrees_l579_57978

theorem sin_585_degrees : Real.sin (585 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_585_degrees_l579_57978


namespace NUMINAMATH_CALUDE_angle_relationships_l579_57933

theorem angle_relationships (A B C : ℝ) : 
  A + B = 180 →  -- A and B are supplementary
  C = B / 2 →    -- C is half of B
  A = 6 * B →    -- A is 6 times B
  (A = 1080 / 7 ∧ B = 180 / 7 ∧ C = 90 / 7) := by
  sorry

end NUMINAMATH_CALUDE_angle_relationships_l579_57933


namespace NUMINAMATH_CALUDE_square_root_calculations_l579_57952

theorem square_root_calculations :
  (3 * Real.sqrt 8 - Real.sqrt 32 = 2 * Real.sqrt 2) ∧
  (Real.sqrt 6 * Real.sqrt 2 / Real.sqrt 3 = 2) ∧
  ((Real.sqrt 24 + Real.sqrt (1/6)) / Real.sqrt 3 = 13 * Real.sqrt 2 / 6) ∧
  (Real.sqrt 27 - Real.sqrt 12) / Real.sqrt 3 - 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_square_root_calculations_l579_57952


namespace NUMINAMATH_CALUDE_exists_multiple_sum_of_digits_divides_l579_57947

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: For all positive integers n, there exists a multiple k of n such that 
    the sum of digits of k divides k. -/
theorem exists_multiple_sum_of_digits_divides (n : ℕ+) : 
  ∃ k : ℕ+, n ∣ k ∧ sum_of_digits k ∣ k := by
  sorry

end NUMINAMATH_CALUDE_exists_multiple_sum_of_digits_divides_l579_57947


namespace NUMINAMATH_CALUDE_chocolate_distribution_l579_57932

/-- Calculates the number of chocolate squares each student receives when:
  * Gerald brings 7 chocolate bars
  * Each bar contains 8 squares
  * For every bar Gerald brings, the teacher brings 2 more identical ones
  * There are 24 students in class
-/
theorem chocolate_distribution (gerald_bars : Nat) (squares_per_bar : Nat) (teacher_ratio : Nat) (num_students : Nat)
    (h1 : gerald_bars = 7)
    (h2 : squares_per_bar = 8)
    (h3 : teacher_ratio = 2)
    (h4 : num_students = 24) :
    (gerald_bars + gerald_bars * teacher_ratio) * squares_per_bar / num_students = 7 := by
  sorry


end NUMINAMATH_CALUDE_chocolate_distribution_l579_57932


namespace NUMINAMATH_CALUDE_tan_sin_function_property_l579_57925

/-- Given a function f(x) = tan x + sin x + 1, prove that if f(b) = 2, then f(-b) = 0 -/
theorem tan_sin_function_property (b : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ Real.tan x + Real.sin x + 1
  f b = 2 → f (-b) = 0 := by
sorry

end NUMINAMATH_CALUDE_tan_sin_function_property_l579_57925


namespace NUMINAMATH_CALUDE_Mp_not_perfect_square_l579_57935

/-- A prime number p congruent to 3 modulo 4 -/
def p : ℕ := sorry

/-- Assumption that p is prime -/
axiom p_prime : Nat.Prime p

/-- Assumption that p is congruent to 3 modulo 4 -/
axiom p_mod_4 : p % 4 = 3

/-- Definition of a balanced sequence -/
def BalancedSequence (seq : List ℤ) : Prop :=
  (∀ x ∈ seq, ∃ y ∈ seq, x = -y) ∧
  (∀ x ∈ seq, |x| ≤ (p - 1) / 2) ∧
  (seq.length ≤ p - 1)

/-- The number of balanced sequences for prime p -/
def Mp : ℕ := sorry

/-- Theorem: Mp is not a perfect square -/
theorem Mp_not_perfect_square : ¬ ∃ (n : ℕ), Mp = n ^ 2 := by sorry

end NUMINAMATH_CALUDE_Mp_not_perfect_square_l579_57935


namespace NUMINAMATH_CALUDE_solve_for_q_l579_57958

theorem solve_for_q (n m q : ℚ) 
  (h1 : 5/6 = n/60)
  (h2 : 5/6 = (m+n)/90)
  (h3 : 5/6 = (q-m)/150) : q = 150 := by sorry

end NUMINAMATH_CALUDE_solve_for_q_l579_57958


namespace NUMINAMATH_CALUDE_shelves_used_l579_57910

def initial_stock : ℕ := 40
def books_sold : ℕ := 20
def books_per_shelf : ℕ := 4

theorem shelves_used : (initial_stock - books_sold) / books_per_shelf = 5 :=
by sorry

end NUMINAMATH_CALUDE_shelves_used_l579_57910


namespace NUMINAMATH_CALUDE_base2_to_base4_conversion_l579_57919

/-- Converts a natural number from base 2 to base 4 -/
def base2ToBase4 (n : ℕ) : ℕ := sorry

/-- The base 2 representation of the number -/
def base2Number : ℕ := 1011101100

/-- The expected base 4 representation of the number -/
def expectedBase4Number : ℕ := 23230

theorem base2_to_base4_conversion :
  base2ToBase4 base2Number = expectedBase4Number := by sorry

end NUMINAMATH_CALUDE_base2_to_base4_conversion_l579_57919


namespace NUMINAMATH_CALUDE_binary_sum_equals_1945_l579_57966

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : Nat :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

theorem binary_sum_equals_1945 :
  let num1 := binary_to_decimal [true, true, true, true, true, true, true, true, true, true]
  let num2 := binary_to_decimal [false, true, false, true, false, true, false, true, false, true]
  let num3 := binary_to_decimal [false, false, false, false, true, true, true, true]
  num1 + num2 + num3 = 1945 := by
  sorry

end NUMINAMATH_CALUDE_binary_sum_equals_1945_l579_57966


namespace NUMINAMATH_CALUDE_treasure_chest_coins_l579_57992

theorem treasure_chest_coins : ∃ n : ℕ, 
  (n > 0) ∧
  (n % 8 = 2) ∧ 
  (n % 7 = 6) ∧ 
  (∀ m : ℕ, m > 0 ∧ m < n → (m % 8 ≠ 2 ∨ m % 7 ≠ 6)) →
  (n % 9 = 7) := by
sorry

end NUMINAMATH_CALUDE_treasure_chest_coins_l579_57992


namespace NUMINAMATH_CALUDE_shooting_probabilities_l579_57908

/-- Probability of a person hitting a target -/
def prob_hit (p : ℝ) : ℝ := p

/-- Probability of missing at least once in n shots -/
def prob_miss_at_least_once (p : ℝ) (n : ℕ) : ℝ := 1 - p^n

/-- Probability of stopping exactly after n shots, given stopping after two consecutive misses -/
def prob_stop_after_n_shots (p : ℝ) (n : ℕ) : ℝ :=
  if n < 2 then 0
  else if n = 2 then (1 - p)^2
  else p * (prob_stop_after_n_shots p (n - 1)) + (1 - p) * p * (1 - p)^2

theorem shooting_probabilities :
  let pA := prob_hit (2/3)
  let pB := prob_hit (3/4)
  (prob_miss_at_least_once pA 4 = 65/81) ∧
  (prob_stop_after_n_shots pB 5 = 45/1024) :=
by sorry

end NUMINAMATH_CALUDE_shooting_probabilities_l579_57908


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l579_57959

theorem simplify_fraction_product : 8 * (15 / 4) * (-40 / 45) = -2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l579_57959


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l579_57982

open Set

-- Define the sets A and B
def A : Set ℝ := {x | 0 < x ∧ x < 2}
def B : Set ℝ := {x | x^2 - 1 < 0}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = Ioo (-1) 2 := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l579_57982


namespace NUMINAMATH_CALUDE_kylie_picked_558_apples_l579_57967

/-- Represents the number of apples picked in each hour -/
structure ApplesPicked where
  first_hour : ℕ
  second_hour : ℕ
  third_hour : ℕ

/-- Calculates the total number of apples picked -/
def total_apples (ap : ApplesPicked) : ℕ :=
  ap.first_hour + ap.second_hour + ap.third_hour

/-- Represents the first three Fibonacci numbers -/
def first_three_fibonacci : List ℕ := [1, 1, 2]

/-- Represents the first three terms of the arithmetic progression -/
def arithmetic_progression (a₁ d : ℕ) : List ℕ :=
  [a₁, a₁ + d, a₁ + 2*d]

/-- Kylie's apple picking scenario -/
def kylie_apples : ApplesPicked where
  first_hour := 66
  second_hour := (List.sum first_three_fibonacci) * 66
  third_hour := List.sum (arithmetic_progression 66 10)

/-- Theorem stating that Kylie picked 558 apples in total -/
theorem kylie_picked_558_apples :
  total_apples kylie_apples = 558 := by
  sorry


end NUMINAMATH_CALUDE_kylie_picked_558_apples_l579_57967


namespace NUMINAMATH_CALUDE_point_on_x_axis_l579_57913

/-- If point P with coordinates (4-a, 3a+9) lies on the x-axis, then its coordinates are (7, 0) -/
theorem point_on_x_axis (a : ℝ) :
  let P : ℝ × ℝ := (4 - a, 3 * a + 9)
  (P.2 = 0) → P = (7, 0) := by
  sorry

end NUMINAMATH_CALUDE_point_on_x_axis_l579_57913


namespace NUMINAMATH_CALUDE_trigonometric_equation_solutions_l579_57942

theorem trigonometric_equation_solutions (x : ℝ) : 
  (1 + Real.sin x + Real.cos (3 * x) = Real.cos x + Real.sin (2 * x) + Real.cos (2 * x)) ↔ 
  (∃ k : ℤ, x = k * Real.pi ∨ 
            x = (-1)^(k+1) * Real.pi / 6 + k * Real.pi ∨ 
            x = Real.pi / 3 + 2 * k * Real.pi ∨ 
            x = -Real.pi / 3 + 2 * k * Real.pi) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solutions_l579_57942


namespace NUMINAMATH_CALUDE_quadratic_function_m_range_l579_57969

def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 5

theorem quadratic_function_m_range (a : ℝ) (m : ℝ) :
  (∀ t, f a t = f a (-4 - t)) →
  (∀ x ∈ Set.Icc m 0, f a x ≤ 5) →
  (∃ x ∈ Set.Icc m 0, f a x = 5) →
  (∀ x ∈ Set.Icc m 0, f a x ≥ 1) →
  (∃ x ∈ Set.Icc m 0, f a x = 1) →
  -4 ≤ m ∧ m ≤ -2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_m_range_l579_57969


namespace NUMINAMATH_CALUDE_add_three_preserves_inequality_l579_57936

theorem add_three_preserves_inequality (a b : ℝ) (h : a > b) : a + 3 > b + 3 := by
  sorry

end NUMINAMATH_CALUDE_add_three_preserves_inequality_l579_57936


namespace NUMINAMATH_CALUDE_shopkeeper_sales_l579_57927

/-- The number of articles sold by a shopkeeper -/
def articles_sold (cost_price : ℝ) : ℕ :=
  72

/-- The profit percentage made by the shopkeeper -/
def profit_percentage : ℝ :=
  20

/-- The number of articles whose cost price equals the selling price -/
def equivalent_articles : ℕ :=
  60

theorem shopkeeper_sales :
  ∀ (cost_price : ℝ),
  cost_price > 0 →
  (articles_sold cost_price : ℝ) * cost_price =
    equivalent_articles * cost_price * (1 + profit_percentage / 100) :=
by sorry

end NUMINAMATH_CALUDE_shopkeeper_sales_l579_57927


namespace NUMINAMATH_CALUDE_rescue_net_sag_l579_57915

/-- Represents the sag of a rescue net under different conditions -/
theorem rescue_net_sag
  (m₁ : Real) (h₁ : Real) (x₁ : Real)  -- Mass, height, and sag for first jump
  (m₂ : Real) (h₂ : Real)              -- Mass and height for second jump
  (g : Real)                           -- Gravitational acceleration
  (hm₁ : m₁ = 78.75)                   -- Mass of first jumper (athlete)
  (hh₁ : h₁ = 15)                      -- Height of first jump
  (hx₁ : x₁ = 1)                       -- Sag for first jump
  (hm₂ : m₂ = 45)                      -- Mass of second jumper (person)
  (hh₂ : h₂ = 29)                      -- Height of second jump
  (hg : g > 0)                         -- Gravity is positive
  : ∃ x₂ : Real, 
    abs (x₂ - 1.38) < 0.01 ∧           -- Allow for small numerical difference
    m₁ * g * (h₁ + x₁) * x₂^2 = m₂ * g * (h₂ + x₂) * x₁^2 := by
  sorry


end NUMINAMATH_CALUDE_rescue_net_sag_l579_57915


namespace NUMINAMATH_CALUDE_special_quadratic_a_range_l579_57905

/-- A quadratic function with specific properties -/
structure SpecialQuadratic where
  f : ℝ → ℝ
  is_quadratic : ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c
  symmetry : ∀ x, f (2 + x) = f (2 - x)
  inequality : ∃ a : ℝ, f a ≤ f 0 ∧ f 0 < f 1

/-- The range of 'a' for a special quadratic function -/
def range_of_a (q : SpecialQuadratic) : Set ℝ :=
  {x | x ≤ 0 ∨ x ≥ 4}

/-- Theorem stating the range of 'a' for a special quadratic function -/
theorem special_quadratic_a_range (q : SpecialQuadratic) :
  ∀ a : ℝ, q.f a ≤ q.f 0 → a ∈ range_of_a q := by
  sorry

end NUMINAMATH_CALUDE_special_quadratic_a_range_l579_57905


namespace NUMINAMATH_CALUDE_ingredient_problem_l579_57960

/-- Represents the quantities and prices of ingredients A and B -/
structure Ingredients where
  total_quantity : ℕ
  price_a : ℕ
  price_b_base : ℕ
  price_b_decrease : ℚ
  quantity_b : ℕ

/-- The total cost function for the ingredients -/
def total_cost (i : Ingredients) : ℚ :=
  if i.quantity_b ≤ 300 then
    (i.total_quantity - i.quantity_b) * i.price_a + i.quantity_b * i.price_b_base
  else
    (i.total_quantity - i.quantity_b) * i.price_a + 
    i.quantity_b * (i.price_b_base - (i.quantity_b - 300) / 10 * i.price_b_decrease)

/-- The main theorem encompassing all parts of the problem -/
theorem ingredient_problem (i : Ingredients) 
  (h_total : i.total_quantity = 600)
  (h_price_a : i.price_a = 5)
  (h_price_b_base : i.price_b_base = 9)
  (h_price_b_decrease : i.price_b_decrease = 0.1)
  (h_quantity_b_multiple : i.quantity_b % 10 = 0) :
  (∃ (x : ℕ), x < 300 ∧ i.quantity_b = x ∧ total_cost i = 3800 → 
    i.total_quantity - x = 400 ∧ x = 200) ∧
  (∃ (x : ℕ), x > 300 ∧ i.quantity_b = x ∧ 2 * (i.total_quantity - x) ≥ x → 
    ∃ (min_cost : ℚ), min_cost = 4200 ∧ 
    ∀ (y : ℕ), y > 300 ∧ 2 * (i.total_quantity - y) ≥ y → 
      total_cost { i with quantity_b := y } ≥ min_cost) ∧
  (∃ (m : ℕ), m < 250 ∧ 
    (∀ (x : ℕ), x > 300 ∧ i.total_quantity - x ≤ m → 
      total_cost { i with quantity_b := x } ≤ 4000) ∧
    (∃ (x : ℕ), x > 300 ∧ i.total_quantity - x ≤ m ∧ 
      total_cost { i with quantity_b := x } = 4000) →
    m = 100) := by
  sorry

end NUMINAMATH_CALUDE_ingredient_problem_l579_57960


namespace NUMINAMATH_CALUDE_cubic_term_simplification_l579_57987

theorem cubic_term_simplification (a : ℝ) : a^3 + 7*a^3 - 5*a^3 = 3*a^3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_term_simplification_l579_57987


namespace NUMINAMATH_CALUDE_grassy_plot_width_l579_57903

/-- Proves that the width of a rectangular grassy plot is 60 meters given the specified conditions --/
theorem grassy_plot_width :
  ∀ (w : ℝ),
  let plot_length : ℝ := 100
  let path_width : ℝ := 2.5
  let gravel_cost_per_sqm : ℝ := 0.9  -- 90 paise = 0.9 rupees
  let total_gravel_cost : ℝ := 742.5
  let total_length : ℝ := plot_length + 2 * path_width
  let total_width : ℝ := w + 2 * path_width
  let path_area : ℝ := total_length * total_width - plot_length * w
  gravel_cost_per_sqm * path_area = total_gravel_cost →
  w = 60 := by
sorry

end NUMINAMATH_CALUDE_grassy_plot_width_l579_57903


namespace NUMINAMATH_CALUDE_second_butcher_delivery_l579_57911

/-- Represents the number of packages delivered by each butcher -/
structure ButcherDelivery where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Represents the weight of each package and the total weight delivered -/
structure DeliveryInfo where
  package_weight : ℕ
  total_weight : ℕ

/-- Given the delivery information and the number of packages from the first and third butchers,
    proves that the second butcher delivered 7 packages -/
theorem second_butcher_delivery 
  (delivery : ButcherDelivery)
  (info : DeliveryInfo)
  (h1 : delivery.first = 10)
  (h2 : delivery.third = 8)
  (h3 : info.package_weight = 4)
  (h4 : info.total_weight = 100)
  (h5 : info.total_weight = 
    (delivery.first + delivery.second + delivery.third) * info.package_weight) :
  delivery.second = 7 := by
  sorry


end NUMINAMATH_CALUDE_second_butcher_delivery_l579_57911


namespace NUMINAMATH_CALUDE_average_of_w_and_x_l579_57994

theorem average_of_w_and_x (w x y : ℝ) 
  (h1 : 7 / w + 7 / x = 7 / y) 
  (h2 : w * x = y) : 
  (w + x) / 2 = y / 2 := by
  sorry

end NUMINAMATH_CALUDE_average_of_w_and_x_l579_57994


namespace NUMINAMATH_CALUDE_product_of_parts_l579_57962

theorem product_of_parts (z : ℂ) : z = 1 - I → (z.re * z.im = -1) := by
  sorry

end NUMINAMATH_CALUDE_product_of_parts_l579_57962


namespace NUMINAMATH_CALUDE_sum_of_altitudes_equals_2432_div_17_l579_57917

-- Define the line equation
def line_equation (x y : ℝ) : Prop := 15 * x + 8 * y = 120

-- Define the triangle formed by the line and coordinate axes
def triangle : Set (ℝ × ℝ) :=
  {p | p.1 ≥ 0 ∧ p.2 ≥ 0 ∧ line_equation p.1 p.2}

-- Define the function to calculate the sum of altitudes
def sum_of_altitudes (t : Set (ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem sum_of_altitudes_equals_2432_div_17 :
  sum_of_altitudes triangle = 2432 / 17 := by sorry

end NUMINAMATH_CALUDE_sum_of_altitudes_equals_2432_div_17_l579_57917


namespace NUMINAMATH_CALUDE_circle_area_equality_l579_57950

theorem circle_area_equality (r₁ r₂ : ℝ) (h₁ : r₁ = 25) (h₂ : r₂ = 17) :
  ∃ r : ℝ, π * r^2 = π * r₁^2 - π * r₂^2 ∧ r = 4 * Real.sqrt 21 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_equality_l579_57950


namespace NUMINAMATH_CALUDE_smallest_value_l579_57940

def Q (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x + 2

theorem smallest_value (x₁ x₂ x₃ : ℝ) (hzeros : Q x₁ = 0 ∧ Q x₂ = 0 ∧ Q x₃ = 0) :
  min (min (Q (-1)) (1 + (-3) + (-9) + 2)) (min (x₁ * x₂ * x₃) (Q 1)) = x₁ * x₂ * x₃ :=
sorry

end NUMINAMATH_CALUDE_smallest_value_l579_57940


namespace NUMINAMATH_CALUDE_fraction_not_simplifiable_l579_57904

theorem fraction_not_simplifiable (n : ℕ) : 
  (∃ k : ℕ, n = 6 * k + 1 ∨ n = 6 * k + 3) ↔ 
  ¬∃ (a b : ℕ), (n^2 + 2 : ℚ) / (n * (n + 1)) = (a : ℚ) / b ∧ 
                gcd a b = 1 ∧ 
                b < n * (n + 1) :=
sorry

end NUMINAMATH_CALUDE_fraction_not_simplifiable_l579_57904


namespace NUMINAMATH_CALUDE_betty_has_winning_strategy_l579_57986

/-- Represents the state of a bowl -/
structure BowlState :=
  (redBalls : Nat)
  (blueBalls : Nat)

/-- Represents the state of the game -/
structure GameState :=
  (blueBowl : BowlState)
  (redBowl : BowlState)

/-- Enum for the possible moves in the game -/
inductive Move
  | TakeRedFromBlue
  | TakeBlueFromRed
  | ThrowAway

/-- Enum for the players -/
inductive Player
  | Albert
  | Betty

/-- Function to check if a game state is winning for the current player -/
def isWinningState (state : GameState) : Bool :=
  state.blueBowl.redBalls = 0 || state.redBowl.blueBalls = 0

/-- Function to get the next player -/
def nextPlayer (player : Player) : Player :=
  match player with
  | Player.Albert => Player.Betty
  | Player.Betty => Player.Albert

/-- Function to apply a move to the game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  match move with
  | Move.TakeRedFromBlue => 
      { blueBowl := { redBalls := state.blueBowl.redBalls - 2, blueBalls := state.blueBowl.blueBalls },
        redBowl := { redBalls := state.redBowl.redBalls + 2, blueBalls := state.redBowl.blueBalls } }
  | Move.TakeBlueFromRed => 
      { blueBowl := { redBalls := state.blueBowl.redBalls, blueBalls := state.blueBowl.blueBalls + 2 },
        redBowl := { redBalls := state.redBowl.redBalls, blueBalls := state.redBowl.blueBalls - 2 } }
  | Move.ThrowAway => 
      { blueBowl := { redBalls := state.blueBowl.redBalls - 1, blueBalls := state.blueBowl.blueBalls - 1 },
        redBowl := state.redBowl }

/-- The initial state of the game -/
def initialState : GameState :=
  { blueBowl := { redBalls := 100, blueBalls := 0 },
    redBowl := { redBalls := 0, blueBalls := 100 } }

/-- Theorem stating that Betty has a winning strategy -/
theorem betty_has_winning_strategy :
  ∃ (strategy : Player → GameState → Move),
    ∀ (game : Nat → GameState),
      game 0 = initialState →
      (∀ n, game (n + 1) = applyMove (game n) (strategy (if n % 2 = 0 then Player.Albert else Player.Betty) (game n))) →
      ∃ n, isWinningState (game n) ∧ n % 2 = 1 :=
sorry


end NUMINAMATH_CALUDE_betty_has_winning_strategy_l579_57986


namespace NUMINAMATH_CALUDE_ladder_length_l579_57974

/-- Proves that the length of a ladder is 9.2 meters, given specific conditions. -/
theorem ladder_length (angle : Real) (foot_distance : Real) (length : Real) : 
  angle = 60 * π / 180 →
  foot_distance = 4.6 →
  Real.cos angle = foot_distance / length →
  length = 9.2 := by
sorry

end NUMINAMATH_CALUDE_ladder_length_l579_57974


namespace NUMINAMATH_CALUDE_function_inequality_l579_57995

-- Define a function f that is differentiable on ℝ
variable (f : ℝ → ℝ)
variable (hf : Differentiable ℝ f)

-- Define the condition that 2f(x) - f'(x) > 0 for all x in ℝ
variable (h : ∀ x : ℝ, 2 * f x - deriv f x > 0)

-- State the theorem
theorem function_inequality : f 1 > f 2 / Real.exp 2 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l579_57995


namespace NUMINAMATH_CALUDE_cube_root_simplification_l579_57979

theorem cube_root_simplification : 
  (50^3 + 60^3 + 70^3 : ℝ)^(1/3) = 10 * 684^(1/3) := by sorry

end NUMINAMATH_CALUDE_cube_root_simplification_l579_57979


namespace NUMINAMATH_CALUDE_regression_validity_l579_57990

/-- Represents a linear regression equation -/
structure LinearRegression where
  slope : ℝ
  intercept : ℝ

/-- Represents the sample means of x and y -/
structure SampleMeans where
  x : ℝ
  y : ℝ

/-- Checks if the linear regression equation is valid for the given sample means -/
def isValidRegression (reg : LinearRegression) (means : SampleMeans) : Prop :=
  means.y = reg.slope * means.x + reg.intercept

/-- Theorem stating that the given linear regression is valid for the provided sample means -/
theorem regression_validity (means : SampleMeans) 
    (h_corr : 0 < 0.4) -- Positive correlation between x and y
    (h_means_x : means.x = 3)
    (h_means_y : means.y = 3.5) :
    isValidRegression ⟨0.4, 2.3⟩ means := by
  sorry

end NUMINAMATH_CALUDE_regression_validity_l579_57990


namespace NUMINAMATH_CALUDE_quadratic_negative_range_l579_57991

/-- The quadratic function f(x) = ax^2 + 2ax + m -/
def f (a m : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 * a * x + m

theorem quadratic_negative_range (a m : ℝ) (h1 : a < 0) (h2 : f a m 2 = 0) :
  {x : ℝ | f a m x < 0} = {x : ℝ | x < -4 ∨ x > 2} := by
  sorry

end NUMINAMATH_CALUDE_quadratic_negative_range_l579_57991


namespace NUMINAMATH_CALUDE_base_nine_addition_l579_57938

/-- Represents a number in base 9 --/
def BaseNine : Type := List (Fin 9)

/-- Converts a base 9 number to a natural number --/
def to_nat (b : BaseNine) : ℕ :=
  b.foldr (λ d acc => 9 * acc + d.val) 0

/-- Adds two base 9 numbers --/
def add_base_nine (a b : BaseNine) : BaseNine :=
  sorry

theorem base_nine_addition :
  let a : BaseNine := [2, 5, 6]
  let b : BaseNine := [8, 5]
  let c : BaseNine := [1, 5, 5]
  let result : BaseNine := [5, 1, 7, 6]
  add_base_nine (add_base_nine a b) c = result := by
  sorry

end NUMINAMATH_CALUDE_base_nine_addition_l579_57938


namespace NUMINAMATH_CALUDE_nineteen_team_tournament_games_l579_57973

/-- Represents a single-elimination tournament. -/
structure Tournament where
  num_teams : ℕ
  is_single_elimination : Bool
  no_ties : Bool

/-- Calculates the number of games needed to determine a winner in a tournament. -/
def games_to_determine_winner (t : Tournament) : ℕ :=
  t.num_teams - 1

/-- Theorem stating that a single-elimination tournament with 19 teams and no ties requires 18 games to determine a winner. -/
theorem nineteen_team_tournament_games (t : Tournament) 
  (h1 : t.num_teams = 19) 
  (h2 : t.is_single_elimination = true) 
  (h3 : t.no_ties = true) : 
  games_to_determine_winner t = 18 := by
  sorry


end NUMINAMATH_CALUDE_nineteen_team_tournament_games_l579_57973


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l579_57902

theorem polynomial_division_remainder : ∃ q : Polynomial ℚ, 
  3 * X^4 + 13 * X^3 + 5 * X^2 - 10 * X + 20 = 
  (X^2 + 5 * X + 1) * q + (-68 * X + 8) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l579_57902


namespace NUMINAMATH_CALUDE_min_c_value_l579_57920

theorem min_c_value (a b c k : ℕ+) (h1 : b = a + k) (h2 : c = a + 2*k) 
  (h3 : a < b ∧ b < c) 
  (h4 : ∃! (x y : ℝ), 3*x + y = 3005 ∧ y = |x - a| + |x - (a + k)| + |x - (a + 2*k)|) :
  c ≥ 6005 ∧ ∃ (a₀ b₀ c₀ k₀ : ℕ+), 
    b₀ = a₀ + k₀ ∧ c₀ = a₀ + 2*k₀ ∧ a₀ < b₀ ∧ b₀ < c₀ ∧ c₀ = 6005 ∧
    ∃! (x y : ℝ), 3*x + y = 3005 ∧ y = |x - a₀| + |x - (a₀ + k₀)| + |x - (a₀ + 2*k₀)| :=
by sorry

end NUMINAMATH_CALUDE_min_c_value_l579_57920


namespace NUMINAMATH_CALUDE_paint_calculation_l579_57924

theorem paint_calculation (num_bedrooms : ℕ) (num_other_rooms : ℕ) 
  (total_cans : ℕ) (white_can_size : ℚ) :
  num_bedrooms = 3 →
  num_other_rooms = 2 * num_bedrooms →
  total_cans = 10 →
  white_can_size = 3 →
  (total_cans - num_bedrooms) * white_can_size / num_other_rooms = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_paint_calculation_l579_57924


namespace NUMINAMATH_CALUDE_black_tiles_to_total_tiles_l579_57948

/-- Represents a square room tiled with congruent square tiles -/
structure TiledRoom where
  side_length : ℕ

/-- Counts the number of black tiles in the room -/
def count_black_tiles (room : TiledRoom) : ℕ :=
  4 * room.side_length - 3

/-- Counts the total number of tiles in the room -/
def count_total_tiles (room : TiledRoom) : ℕ :=
  room.side_length * room.side_length

/-- Theorem stating the relationship between black tiles and total tiles -/
theorem black_tiles_to_total_tiles :
  ∃ (room : TiledRoom), count_black_tiles room = 201 ∧ count_total_tiles room = 2601 :=
sorry

end NUMINAMATH_CALUDE_black_tiles_to_total_tiles_l579_57948


namespace NUMINAMATH_CALUDE_smallest_money_sum_l579_57999

/-- Represents a sum of money in pounds, shillings, pence, and farthings -/
structure Money where
  pounds : ℕ
  shillings : ℕ
  pence : ℕ
  farthings : ℕ
  shillings_valid : shillings < 20
  pence_valid : pence < 12
  farthings_valid : farthings < 4

/-- Checks if a list of digits contains each of 1 to 9 exactly once -/
def valid_digits (digits : List ℕ) : Prop :=
  digits.length = 9 ∧ (∀ d, d ∈ digits → d ≥ 1 ∧ d ≤ 9) ∧ digits.Nodup

/-- Converts a Money value to its total value in farthings -/
def to_farthings (m : Money) : ℕ :=
  m.pounds * 960 + m.shillings * 48 + m.pence * 4 + m.farthings

/-- The theorem to be proved -/
theorem smallest_money_sum :
  ∃ (m : Money) (digits : List ℕ),
    valid_digits digits ∧
    to_farthings m = to_farthings ⟨2567, 18, 9, 3, by sorry, by sorry, by sorry⟩ ∧
    (∀ (m' : Money) (digits' : List ℕ),
      valid_digits digits' →
      to_farthings m' ≥ to_farthings m) :=
by sorry

end NUMINAMATH_CALUDE_smallest_money_sum_l579_57999


namespace NUMINAMATH_CALUDE_unique_stutterer_square_l579_57945

/-- A function that checks if a number is a stutterer (first two digits are the same and last two digits are the same) --/
def is_stutterer (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n < 10000 ∧
  (n / 1000 = (n / 100) % 10) ∧
  ((n / 10) % 10 = n % 10)

/-- The theorem stating that 7744 is the only four-digit stutterer number that is a perfect square --/
theorem unique_stutterer_square : ∀ n : ℕ, 
  is_stutterer n ∧ ∃ k : ℕ, n = k^2 ↔ n = 7744 :=
sorry

end NUMINAMATH_CALUDE_unique_stutterer_square_l579_57945


namespace NUMINAMATH_CALUDE_man_downstream_speed_l579_57965

/-- Calculates the downstream speed given upstream and still water speeds -/
def downstream_speed (upstream_speed still_water_speed : ℝ) : ℝ :=
  2 * still_water_speed - upstream_speed

/-- Theorem stating that given the specified upstream and still water speeds, 
    the downstream speed is 80 kmph -/
theorem man_downstream_speed :
  downstream_speed 20 50 = 80 := by
  sorry

end NUMINAMATH_CALUDE_man_downstream_speed_l579_57965


namespace NUMINAMATH_CALUDE_x_intercept_of_line_l579_57957

/-- The x-intercept of the line 2x + y - 2 = 0 is at x = 1 -/
theorem x_intercept_of_line (x y : ℝ) : 2*x + y - 2 = 0 → y = 0 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_x_intercept_of_line_l579_57957


namespace NUMINAMATH_CALUDE_floor_equation_equivalence_l579_57929

/-- The floor function -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

/-- The solution set for the equation -/
def solution_set : Set ℝ :=
  {x | x < 0 ∨ x ≥ 2.5}

/-- Theorem stating the equivalence of the equation and the solution set -/
theorem floor_equation_equivalence (x : ℝ) :
  floor (1 / (1 - x)) = floor (1 / (1.5 - x)) ↔ x ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_floor_equation_equivalence_l579_57929


namespace NUMINAMATH_CALUDE_geometry_relations_l579_57997

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (contains : Plane → Line → Prop)
variable (parallel : Plane → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)

-- State the theorem
theorem geometry_relations 
  (l m : Line) (α β : Plane) 
  (h1 : perpendicular l α) 
  (h2 : contains β m) :
  (parallel α β → perpendicular_lines l m) ∧
  (parallel_lines l m → perpendicular_planes α β) ∧
  ¬(perpendicular_planes α β → parallel_lines l m) ∧
  ¬(perpendicular_lines l m → parallel α β) :=
by sorry

end NUMINAMATH_CALUDE_geometry_relations_l579_57997


namespace NUMINAMATH_CALUDE_total_ants_is_twenty_l579_57985

/-- The number of ants found by Abe -/
def abe_ants : ℕ := 4

/-- The number of ants found by Beth -/
def beth_ants : ℕ := abe_ants + abe_ants / 2

/-- The number of ants found by CeCe -/
def cece_ants : ℕ := 2 * abe_ants

/-- The number of ants found by Duke -/
def duke_ants : ℕ := abe_ants / 2

/-- The total number of ants found by all four children -/
def total_ants : ℕ := abe_ants + beth_ants + cece_ants + duke_ants

/-- Theorem stating that the total number of ants found is 20 -/
theorem total_ants_is_twenty : total_ants = 20 := by sorry

end NUMINAMATH_CALUDE_total_ants_is_twenty_l579_57985


namespace NUMINAMATH_CALUDE_doughnuts_left_l579_57976

theorem doughnuts_left (total_doughnuts : ℕ) (staff_count : ℕ) (doughnuts_per_staff : ℕ) :
  total_doughnuts = 50 →
  staff_count = 19 →
  doughnuts_per_staff = 2 →
  total_doughnuts - (staff_count * doughnuts_per_staff) = 12 := by
sorry

end NUMINAMATH_CALUDE_doughnuts_left_l579_57976


namespace NUMINAMATH_CALUDE_lower_bound_sum_squares_roots_l579_57930

/-- A monic polynomial of degree 4 with real coefficients -/
structure MonicPolynomial4 where
  coeffs : Fin 4 → ℝ
  monic : coeffs 0 = 1

/-- The sum of the squares of the roots of a polynomial -/
def sumSquaresRoots (p : MonicPolynomial4) : ℝ := sorry

/-- The theorem statement -/
theorem lower_bound_sum_squares_roots (p : MonicPolynomial4)
  (h1 : p.coeffs 1 = 0)  -- No cubic term
  (h2 : ∃ a₂ : ℝ, p.coeffs 2 = a₂ ∧ p.coeffs 3 = 2 * a₂) :  -- a₃ = 2a₂
  |sumSquaresRoots p| ≥ (1/4 : ℝ) := by sorry

end NUMINAMATH_CALUDE_lower_bound_sum_squares_roots_l579_57930


namespace NUMINAMATH_CALUDE_largest_smallest_valid_numbers_l579_57975

def is_valid_number (n : ℕ) : Prop :=
  (n ≥ 1000000000 ∧ n < 10000000000) ∧  -- 10-digit number
  (n % 11 = 0) ∧                        -- divisible by 11
  (∀ i j, i ≠ j → (n / 10^i % 10) ≠ (n / 10^j % 10))  -- no repeated digits

theorem largest_smallest_valid_numbers :
  (∀ n : ℕ, is_valid_number n → n ≤ 9876524130) ∧
  (∀ n : ℕ, is_valid_number n → n ≥ 1024375869) ∧
  is_valid_number 9876524130 ∧
  is_valid_number 1024375869 :=
sorry

end NUMINAMATH_CALUDE_largest_smallest_valid_numbers_l579_57975


namespace NUMINAMATH_CALUDE_cone_central_angle_l579_57941

/-- Given a circular piece of paper with radius 18 cm, when partially cut to form a cone
    with radius 8 cm and volume 128π cm³, the central angle of the sector used to create
    the cone is approximately 53 degrees. -/
theorem cone_central_angle (paper_radius : ℝ) (cone_radius : ℝ) (cone_volume : ℝ) :
  paper_radius = 18 →
  cone_radius = 8 →
  cone_volume = 128 * Real.pi →
  ∃ (central_angle : ℝ), 52 < central_angle ∧ central_angle < 54 := by
  sorry

end NUMINAMATH_CALUDE_cone_central_angle_l579_57941


namespace NUMINAMATH_CALUDE_positive_X_value_l579_57961

-- Define the # operation
def hash (X Y : ℝ) : ℝ := X^2 + Y^2

-- State the theorem
theorem positive_X_value :
  ∃ X : ℝ, X > 0 ∧ hash X 7 = 85 ∧ X = 6 := by
  sorry

end NUMINAMATH_CALUDE_positive_X_value_l579_57961


namespace NUMINAMATH_CALUDE_max_volume_cutout_length_l579_57970

/-- The side length of the original square sheet of iron in centimeters -/
def original_side_length : ℝ := 36

/-- The volume of the box as a function of the side length of the cut-out square -/
def volume (x : ℝ) : ℝ := x * (original_side_length - 2*x)^2

/-- The derivative of the volume function -/
def volume_derivative (x : ℝ) : ℝ := 12 * (18 - x) * (6 - x)

theorem max_volume_cutout_length :
  ∃ (x : ℝ), 0 < x ∧ x < original_side_length / 2 ∧
  volume_derivative x = 0 ∧
  (∀ y, 0 < y → y < original_side_length / 2 → volume y ≤ volume x) ∧
  x = 6 := by sorry

end NUMINAMATH_CALUDE_max_volume_cutout_length_l579_57970


namespace NUMINAMATH_CALUDE_king_crown_cost_l579_57937

/-- Calculates the total cost of a purchase with a tip -/
def totalCostWithTip (originalCost tipPercentage : ℚ) : ℚ :=
  originalCost * (1 + tipPercentage / 100)

/-- Proves that the king pays $22,000 for a $20,000 crown with a 10% tip -/
theorem king_crown_cost :
  totalCostWithTip 20000 10 = 22000 := by
  sorry

end NUMINAMATH_CALUDE_king_crown_cost_l579_57937


namespace NUMINAMATH_CALUDE_functional_equation_solution_l579_57918

-- Define the function type
def FunctionType := ℝ → ℝ

-- State the theorem
theorem functional_equation_solution (c : ℝ) (h_c : c > 1) (f : FunctionType) 
  (h_f : ∀ x y : ℝ, f (x + y) = f x * f y - c * Real.sin x * Real.sin y) :
  (∀ t : ℝ, f t = Real.sqrt (c - 1) * Real.sin t + Real.cos t) ∨ 
  (∀ t : ℝ, f t = -Real.sqrt (c - 1) * Real.sin t + Real.cos t) :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l579_57918


namespace NUMINAMATH_CALUDE_total_cost_calculation_l579_57964

def cabinet_price : ℝ := 1200
def cabinet_discount : ℝ := 0.15
def dining_table_price : ℝ := 1800
def dining_table_discount : ℝ := 0.20
def sofa_price : ℝ := 2500
def sofa_discount : ℝ := 0.10
def sales_tax : ℝ := 0.08

def discounted_price (price : ℝ) (discount : ℝ) : ℝ :=
  price * (1 - discount)

def total_discounted_price : ℝ :=
  discounted_price cabinet_price cabinet_discount +
  discounted_price dining_table_price dining_table_discount +
  discounted_price sofa_price sofa_discount

def total_cost : ℝ :=
  total_discounted_price * (1 + sales_tax)

theorem total_cost_calculation :
  total_cost = 5086.80 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_calculation_l579_57964


namespace NUMINAMATH_CALUDE_expand_and_evaluate_l579_57980

theorem expand_and_evaluate : 
  ∀ x : ℝ, (x + 3) * (4 * x - 8) = 4 * x^2 + 4 * x - 24 ∧ 
  (let x : ℝ := 5; 4 * x^2 + 4 * x - 24) = 96 := by sorry

end NUMINAMATH_CALUDE_expand_and_evaluate_l579_57980


namespace NUMINAMATH_CALUDE_world_cup_gifts_l579_57928

/-- Calculates the number of gifts needed for a world cup inauguration event. -/
def gifts_needed (num_teams : ℕ) : ℕ :=
  num_teams * 2

/-- Theorem: The number of gifts needed for the world cup inauguration event with 7 teams is 14. -/
theorem world_cup_gifts : gifts_needed 7 = 14 := by
  sorry

end NUMINAMATH_CALUDE_world_cup_gifts_l579_57928


namespace NUMINAMATH_CALUDE_problem_triangle_integer_lengths_l579_57971

/-- Represents a right triangle with integer side lengths -/
structure RightTriangle where
  de : ℕ
  ef : ℕ

/-- Counts the number of distinct integer lengths of line segments
    that can be drawn from a vertex to the opposite side in a right triangle -/
def countIntegerLengths (t : RightTriangle) : ℕ :=
  sorry

/-- The specific right triangle in the problem -/
def problemTriangle : RightTriangle :=
  { de := 15, ef := 36 }

/-- Theorem stating that the number of distinct integer lengths
    in the problem triangle is 24 -/
theorem problem_triangle_integer_lengths :
  countIntegerLengths problemTriangle = 24 := by
  sorry

end NUMINAMATH_CALUDE_problem_triangle_integer_lengths_l579_57971


namespace NUMINAMATH_CALUDE_multiple_implies_equal_l579_57921

theorem multiple_implies_equal (a b : ℕ+) (h : ∃ k : ℕ, (a^2 + a*b + 1 : ℕ) = k * (b^2 + a*b + 1)) : a = b := by
  sorry

end NUMINAMATH_CALUDE_multiple_implies_equal_l579_57921


namespace NUMINAMATH_CALUDE_equation_solution_l579_57926

theorem equation_solution : ∃ x : ℝ, (x - 1) / (2 * x + 1) = 1 ∧ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l579_57926


namespace NUMINAMATH_CALUDE_shelby_rain_time_l579_57983

/-- Represents the speed of Shelby's scooter in miles per hour -/
structure ScooterSpeed where
  normal : ℝ  -- Speed when not raining
  rain : ℝ    -- Speed when raining

/-- Represents Shelby's journey -/
structure Journey where
  total_distance : ℝ  -- Total distance covered in miles
  total_time : ℝ      -- Total time taken in minutes
  rain_time : ℝ       -- Time driven in rain in minutes

/-- Checks if the given journey satisfies the conditions of Shelby's ride -/
def is_valid_journey (speed : ScooterSpeed) (j : Journey) : Prop :=
  speed.normal = 40 ∧
  speed.rain = 25 ∧
  j.total_distance = 20 ∧
  j.total_time = 40 ∧
  j.total_distance = (speed.normal / 60) * (j.total_time - j.rain_time) + (speed.rain / 60) * j.rain_time

theorem shelby_rain_time (speed : ScooterSpeed) (j : Journey) 
  (h : is_valid_journey speed j) : j.rain_time = 27 := by
  sorry

end NUMINAMATH_CALUDE_shelby_rain_time_l579_57983


namespace NUMINAMATH_CALUDE_complex_number_location_l579_57949

theorem complex_number_location (z : ℂ) (h : (1 - Complex.I) * z = 2 * Complex.I) :
  z.re < 0 ∧ z.im > 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_location_l579_57949


namespace NUMINAMATH_CALUDE_nested_fraction_evaluation_l579_57988

theorem nested_fraction_evaluation :
  1 / (2 + 1 / (3 + 1 / 4)) = 13 / 30 := by
sorry

end NUMINAMATH_CALUDE_nested_fraction_evaluation_l579_57988


namespace NUMINAMATH_CALUDE_laundry_time_proof_l579_57953

/-- Proves that the time to wash one load of laundry is 45 minutes -/
theorem laundry_time_proof (wash_time : ℕ) : 
  (2 * wash_time + 75 = 165) → wash_time = 45 := by
  sorry

end NUMINAMATH_CALUDE_laundry_time_proof_l579_57953


namespace NUMINAMATH_CALUDE_convex_polygon_24_sides_diagonals_l579_57956

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

theorem convex_polygon_24_sides_diagonals :
  num_diagonals 24 = 126 := by
  sorry

end NUMINAMATH_CALUDE_convex_polygon_24_sides_diagonals_l579_57956


namespace NUMINAMATH_CALUDE_starting_number_for_prime_factors_of_210_l579_57998

def isPrime (n : Nat) : Prop := sorry

def isFactor (a b : Nat) : Prop := sorry

theorem starting_number_for_prime_factors_of_210 :
  ∃ (start : Nat),
    start ≤ 100 ∧
    (∀ p, isPrime p → p > start → p ≤ 100 → isFactor p 210 →
      ∃ (primes : Finset Nat),
        primes.card = 4 ∧
        (∀ q ∈ primes, isPrime q ∧ q > start ∧ q ≤ 100 ∧ isFactor q 210)) ∧
    (∀ start' > start,
      ¬(∃ (primes : Finset Nat),
        primes.card = 4 ∧
        (∀ q ∈ primes, isPrime q ∧ q > start' ∧ q ≤ 100 ∧ isFactor q 210))) ∧
    start = 1 :=
by sorry

end NUMINAMATH_CALUDE_starting_number_for_prime_factors_of_210_l579_57998


namespace NUMINAMATH_CALUDE_probability_of_blue_ball_l579_57993

theorem probability_of_blue_ball (p_red p_yellow p_blue : ℝ) : 
  p_red = 0.48 → p_yellow = 0.35 → p_red + p_yellow + p_blue = 1 → p_blue = 0.17 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_blue_ball_l579_57993


namespace NUMINAMATH_CALUDE_toy_poodle_height_is_14_l579_57981

def standard_poodle_height : ℕ := 28

def height_difference_standard_miniature : ℕ := 8

def height_difference_miniature_toy : ℕ := 6

def toy_poodle_height : ℕ := standard_poodle_height - height_difference_standard_miniature - height_difference_miniature_toy

theorem toy_poodle_height_is_14 : toy_poodle_height = 14 := by
  sorry

end NUMINAMATH_CALUDE_toy_poodle_height_is_14_l579_57981


namespace NUMINAMATH_CALUDE_barkley_bones_theorem_l579_57954

/-- Calculates the number of bones Barkley has available after a given number of months -/
def bones_available (bones_per_month : ℕ) (months : ℕ) (buried_bones : ℕ) : ℕ :=
  bones_per_month * months - buried_bones

/-- Theorem: Barkley has 8 bones available after 5 months -/
theorem barkley_bones_theorem :
  bones_available 10 5 42 = 8 := by
  sorry

end NUMINAMATH_CALUDE_barkley_bones_theorem_l579_57954


namespace NUMINAMATH_CALUDE_bucket_fill_time_l579_57977

/-- Given that two-thirds of a bucket is filled in 100 seconds,
    prove that it takes 150 seconds to fill the bucket completely. -/
theorem bucket_fill_time :
  let partial_fill_time : ℝ := 100
  let partial_fill_fraction : ℝ := 2/3
  let complete_fill_time : ℝ := 150
  (partial_fill_fraction * complete_fill_time = partial_fill_time) →
  complete_fill_time = 150 :=
by sorry

end NUMINAMATH_CALUDE_bucket_fill_time_l579_57977


namespace NUMINAMATH_CALUDE_sin_value_fourth_quadrant_l579_57996

theorem sin_value_fourth_quadrant (α : Real) (h1 : α ∈ Set.Icc (3 * Real.pi / 2) (2 * Real.pi)) 
  (h2 : Real.tan α = -5/12) : Real.sin α = -5/13 := by
  sorry

end NUMINAMATH_CALUDE_sin_value_fourth_quadrant_l579_57996


namespace NUMINAMATH_CALUDE_quadratic_equations_solution_l579_57955

def A (p : ℝ) : Set ℝ := {x | x^2 + p*x - 2 = 0}
def B (q r : ℝ) : Set ℝ := {x | x^2 + q*x + r = 0}

theorem quadratic_equations_solution (p q r : ℝ) :
  (A p ∪ B q r = {-2, 1, 5}) ∧
  (A p ∩ B q r = {-2}) →
  p = -1 ∧ q = -3 ∧ r = -10 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equations_solution_l579_57955


namespace NUMINAMATH_CALUDE_lous_shoes_monthly_goal_l579_57939

/-- The number of shoes Lou's Shoes must sell each month -/
def monthly_goal (last_week : ℕ) (this_week : ℕ) (remaining : ℕ) : ℕ :=
  last_week + this_week + remaining

/-- Theorem stating the total number of shoes Lou's Shoes must sell each month -/
theorem lous_shoes_monthly_goal :
  monthly_goal 27 12 41 = 80 := by
  sorry

end NUMINAMATH_CALUDE_lous_shoes_monthly_goal_l579_57939


namespace NUMINAMATH_CALUDE_equation_represents_two_intersecting_lines_l579_57944

-- Define the equation
def equation (x y : ℝ) : Prop := (x - y)^2 = 3 * x^2 - y^2

-- Theorem statement
theorem equation_represents_two_intersecting_lines :
  ∃ (m₁ m₂ : ℝ), m₁ ≠ m₂ ∧ 
  (∀ (x y : ℝ), equation x y ↔ (y = m₁ * x ∨ y = m₂ * x)) :=
sorry

end NUMINAMATH_CALUDE_equation_represents_two_intersecting_lines_l579_57944


namespace NUMINAMATH_CALUDE_parabola_equation_l579_57946

-- Define the parabola structure
structure Parabola where
  equation : ℝ → ℝ → Prop

-- Define the focus of a parabola
def Focus := ℝ × ℝ

-- Define the line x - y + 4 = 0
def LineEquation (x y : ℝ) : Prop := x - y + 4 = 0

-- Define the condition that the focus is on the line
def FocusOnLine (f : Focus) : Prop := LineEquation f.1 f.2

-- Define the condition that the vertex is at the origin
def VertexAtOrigin (p : Parabola) : Prop := p.equation 0 0

-- Define the condition that the axis of symmetry is one of the coordinate axes
def AxisIsCoordinateAxis (p : Parabola) : Prop :=
  (∀ x y : ℝ, p.equation x y ↔ p.equation x (-y)) ∨
  (∀ x y : ℝ, p.equation x y ↔ p.equation (-x) y)

-- Theorem statement
theorem parabola_equation (p : Parabola) (f : Focus) :
  VertexAtOrigin p →
  AxisIsCoordinateAxis p →
  FocusOnLine f →
  (∀ x y : ℝ, p.equation x y ↔ y^2 = -16*x) ∨
  (∀ x y : ℝ, p.equation x y ↔ x^2 = 16*y) :=
sorry

end NUMINAMATH_CALUDE_parabola_equation_l579_57946


namespace NUMINAMATH_CALUDE_pet_store_dogs_l579_57943

/-- Calculates the number of dogs in a pet store after a series of events --/
def final_dog_count (initial : ℕ) 
  (sunday_received sunday_sold : ℕ)
  (monday_received monday_returned : ℕ)
  (tuesday_received tuesday_sold : ℕ) : ℕ :=
  initial + sunday_received - sunday_sold + 
  monday_received + monday_returned +
  tuesday_received - tuesday_sold

/-- Theorem stating the final number of dogs in the pet store --/
theorem pet_store_dogs : 
  final_dog_count 2 5 2 3 1 4 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_dogs_l579_57943


namespace NUMINAMATH_CALUDE_northern_village_population_l579_57923

theorem northern_village_population :
  let western_village : ℕ := 7488
  let southern_village : ℕ := 6912
  let total_conscripted : ℕ := 300
  let northern_conscripted : ℕ := 108
  let northern_village : ℕ := 4206
  (northern_conscripted : ℚ) / (total_conscripted : ℚ) = 
    (northern_village : ℚ) / ((northern_village + western_village + southern_village) : ℚ) →
  northern_village = 4206 := by
sorry

end NUMINAMATH_CALUDE_northern_village_population_l579_57923


namespace NUMINAMATH_CALUDE_peyton_juice_boxes_l579_57916

/-- Calculate the total number of juice boxes needed for Peyton's children for the school year -/
def total_juice_boxes (num_children : ℕ) (school_days_per_week : ℕ) (weeks_in_school_year : ℕ) : ℕ :=
  num_children * school_days_per_week * weeks_in_school_year

/-- Proof that Peyton needs 375 juice boxes for the entire school year for all of her children -/
theorem peyton_juice_boxes :
  total_juice_boxes 3 5 25 = 375 := by
  sorry

end NUMINAMATH_CALUDE_peyton_juice_boxes_l579_57916
