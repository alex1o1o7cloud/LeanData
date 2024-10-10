import Mathlib

namespace black_block_is_t_shaped_l218_21889

/-- Represents the shape of a block --/
inductive BlockShape
  | L
  | T
  | S
  | I

/-- Represents a block in the rectangular prism --/
structure Block where
  shape : BlockShape
  visible : Bool
  inLowestLayer : Bool

/-- Represents the rectangular prism --/
structure RectangularPrism where
  blocks : Fin 4 → Block
  threeFullyVisible : ∃ (a b c : Fin 4), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    (blocks a).visible ∧ (blocks b).visible ∧ (blocks c).visible
  onePartiallyVisible : ∃ (d : Fin 4), ¬(blocks d).visible
  blackBlockInLowestLayer : ∃ (d : Fin 4), ¬(blocks d).visible ∧ (blocks d).inLowestLayer

/-- The main theorem --/
theorem black_block_is_t_shaped (prism : RectangularPrism) : 
  ∃ (d : Fin 4), ¬(prism.blocks d).visible ∧ (prism.blocks d).shape = BlockShape.T := by
  sorry

end black_block_is_t_shaped_l218_21889


namespace number_of_people_entered_l218_21876

/-- The number of placards each person takes -/
def placards_per_person : ℕ := 2

/-- The total number of placards the basket can hold -/
def basket_capacity : ℕ := 823

/-- The number of people who entered the stadium -/
def people_entered : ℕ := basket_capacity / placards_per_person

/-- Theorem stating the number of people who entered the stadium -/
theorem number_of_people_entered : people_entered = 411 := by sorry

end number_of_people_entered_l218_21876


namespace jade_transactions_l218_21822

theorem jade_transactions (mabel anthony cal jade : ℕ) : 
  mabel = 90 →
  anthony = mabel + mabel / 10 →
  cal = anthony * 2 / 3 →
  jade = cal + 16 →
  jade = 82 := by
sorry

end jade_transactions_l218_21822


namespace total_weekly_prayers_l218_21818

/-- The number of prayers Pastor Paul makes on a regular day -/
def paul_regular_prayers : ℕ := 20

/-- The number of prayers Pastor Caroline makes on a regular day -/
def caroline_regular_prayers : ℕ := 10

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The number of weekdays (non-Sunday days) in a week -/
def weekdays : ℕ := 6

/-- Calculate Pastor Paul's total prayers for a week -/
def paul_weekly_prayers : ℕ :=
  paul_regular_prayers * weekdays + 2 * paul_regular_prayers

/-- Calculate Pastor Bruce's total prayers for a week -/
def bruce_weekly_prayers : ℕ :=
  (paul_regular_prayers / 2) * weekdays + 2 * (2 * paul_regular_prayers)

/-- Calculate Pastor Caroline's total prayers for a week -/
def caroline_weekly_prayers : ℕ :=
  caroline_regular_prayers * weekdays + 3 * caroline_regular_prayers

/-- The main theorem: total prayers of all pastors in a week -/
theorem total_weekly_prayers :
  paul_weekly_prayers + bruce_weekly_prayers + caroline_weekly_prayers = 390 := by
  sorry

end total_weekly_prayers_l218_21818


namespace division_remainder_problem_l218_21881

theorem division_remainder_problem (a b : ℕ) (h1 : a - b = 1365) (h2 : a = 1634) 
  (h3 : ∃ (q : ℕ), q = 6 ∧ a = q * b + (a % b) ∧ a % b < b) : a % b = 20 := by
  sorry

end division_remainder_problem_l218_21881


namespace star_equation_solution_l218_21832

/-- Custom binary operation ⋆ -/
def star (a b : ℝ) : ℝ := a * b + 2 * b - a

/-- Theorem stating that if 7 ⋆ y = 85, then y = 92/9 -/
theorem star_equation_solution (y : ℝ) (h : star 7 y = 85) : y = 92 / 9 := by
  sorry

end star_equation_solution_l218_21832


namespace net_population_increase_l218_21810

/-- The net population increase in one day given specific birth and death rates -/
theorem net_population_increase (birth_rate : ℚ) (death_rate : ℚ) (seconds_per_day : ℕ) : 
  birth_rate = 5 / 2 → death_rate = 3 / 2 → seconds_per_day = 24 * 60 * 60 →
  (birth_rate - death_rate) * seconds_per_day = 86400 := by
  sorry

end net_population_increase_l218_21810


namespace remainder_equality_l218_21827

def r (n : ℕ) : ℕ := n % 6

theorem remainder_equality (n : ℕ) : 
  r (2 * n + 3) = r (5 * n + 6) ↔ ∃ k : ℤ, n = 2 * k - 1 := by sorry

end remainder_equality_l218_21827


namespace geometric_sum_n1_l218_21898

theorem geometric_sum_n1 (x : ℝ) (h : x ≠ 1) :
  1 + x + x^2 = (1 - x^3) / (1 - x) := by
  sorry

end geometric_sum_n1_l218_21898


namespace henry_initial_amount_l218_21843

/-- Henry's initial amount of money -/
def henry_initial : ℕ := sorry

/-- Amount Henry earned from chores -/
def chores_earnings : ℕ := 2

/-- Amount of money Henry's friend had -/
def friend_money : ℕ := 13

/-- Total amount when they put their money together -/
def total_money : ℕ := 20

theorem henry_initial_amount :
  henry_initial + chores_earnings + friend_money = total_money ∧
  henry_initial = 5 := by sorry

end henry_initial_amount_l218_21843


namespace hyperbola_eccentricity_l218_21857

/-- Triangle ABC with sides a, b, c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A hyperbola with foci at the vertices of a triangle -/
structure Hyperbola (t : Triangle) where
  /-- The hyperbola passes through point A of the triangle -/
  passes_through_A : Bool

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola t) : ℝ := sorry

/-- The main theorem -/
theorem hyperbola_eccentricity (t : Triangle) (h : Hyperbola t) :
  t.a = 4 ∧ t.b = 5 ∧ t.c = Real.sqrt 21 ∧ h.passes_through_A = true →
  eccentricity h = 5 + Real.sqrt 21 := by
  sorry

end hyperbola_eccentricity_l218_21857


namespace common_difference_is_negative_three_l218_21805

/-- An arithmetic sequence with given terms -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  third_term : a 3 = 7
  seventh_term : a 7 = -5

/-- The common difference of an arithmetic sequence -/
def common_difference (seq : ArithmeticSequence) : ℝ :=
  seq.a 2 - seq.a 1

theorem common_difference_is_negative_three (seq : ArithmeticSequence) :
  common_difference seq = -3 := by
  sorry

end common_difference_is_negative_three_l218_21805


namespace probability_adjacent_circular_probability_two_adjacent_in_six_l218_21893

def num_people : ℕ := 6

def total_arrangements (n : ℕ) : ℕ := Nat.factorial (n - 1)

def adjacent_arrangements (n : ℕ) : ℕ := 2 * Nat.factorial (n - 2)

theorem probability_adjacent_circular (n : ℕ) (h : n ≥ 3) :
  (adjacent_arrangements n : ℚ) / (total_arrangements n : ℚ) = 2 / (n - 1 : ℚ) :=
sorry

theorem probability_two_adjacent_in_six :
  (adjacent_arrangements num_people : ℚ) / (total_arrangements num_people : ℚ) = 2 / 5 :=
sorry

end probability_adjacent_circular_probability_two_adjacent_in_six_l218_21893


namespace fraction_zero_implies_x_equals_seven_l218_21840

theorem fraction_zero_implies_x_equals_seven :
  ∀ x : ℝ, (x^2 - 49) / (x + 7) = 0 → x = 7 :=
by
  sorry

end fraction_zero_implies_x_equals_seven_l218_21840


namespace random_selection_result_l218_21865

/-- Represents a random number table --/
def RandomNumberTable := List (List Nat)

/-- Represents a position in the random number table --/
structure TablePosition where
  row : Nat
  column : Nat

/-- Function to select numbers from the random number table --/
def selectNumbers (table : RandomNumberTable) (start : TablePosition) (count : Nat) (maxNumber : Nat) : List Nat :=
  sorry

/-- The theorem to prove --/
theorem random_selection_result (table : RandomNumberTable) (studentCount : Nat) (selectionCount : Nat) (startPosition : TablePosition) :
  studentCount = 247 →
  selectionCount = 4 →
  startPosition = ⟨4, 9⟩ →
  selectNumbers table startPosition selectionCount studentCount = [050, 121, 014, 218] :=
sorry

end random_selection_result_l218_21865


namespace basket_probability_l218_21870

def binomial_probability (n : ℕ) (k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

theorem basket_probability : 
  let n : ℕ := 6
  let k : ℕ := 2
  let p : ℝ := 2/3
  binomial_probability n k p = 20/243 := by sorry

end basket_probability_l218_21870


namespace min_value_of_power_difference_l218_21820

theorem min_value_of_power_difference (m n : ℕ) : 12^m - 5^n ≥ 7 ∧ ∃ m n : ℕ, 12^m - 5^n = 7 := by
  sorry

end min_value_of_power_difference_l218_21820


namespace min_value_sum_l218_21809

theorem min_value_sum (a b c d e f : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (pos_d : 0 < d) (pos_e : 0 < e) (pos_f : 0 < f)
  (sum_eq_9 : a + b + c + d + e + f = 9) :
  (1/a) + (9/b) + (16/c) + (25/d) + (36/e) + (49/f) ≥ 676/9 := by
  sorry

end min_value_sum_l218_21809


namespace repeating_decimal_sqrt_pairs_l218_21835

def is_valid_pair (a b : Nat) : Prop :=
  a ≤ 9 ∧ b ≤ 9 ∧ (b * b = 9 * a)

theorem repeating_decimal_sqrt_pairs :
  ∀ a b : Nat, is_valid_pair a b ↔ 
    (a = 0 ∧ b = 0) ∨ (a = 1 ∧ b = 3) ∨ (a = 4 ∧ b = 6) ∨ (a = 9 ∧ b = 9) := by
  sorry

end repeating_decimal_sqrt_pairs_l218_21835


namespace people_per_car_l218_21859

/-- Given 3.0 cars and 189 people going to the zoo, prove that there are 63 people in each car. -/
theorem people_per_car (total_cars : Float) (total_people : Nat) : 
  total_cars = 3.0 → total_people = 189 → (total_people.toFloat / total_cars).round = 63 := by
  sorry

end people_per_car_l218_21859


namespace egyptian_fraction_solutions_l218_21871

def EgyptianFractionSolutions : Set (ℕ × ℕ × ℕ × ℕ) := {
  (2, 3, 7, 42), (2, 3, 8, 24), (2, 3, 9, 18), (2, 3, 10, 15), (2, 3, 12, 12),
  (2, 4, 5, 20), (2, 4, 6, 12), (2, 4, 8, 8), (2, 5, 5, 10), (2, 6, 6, 6),
  (3, 3, 4, 12), (3, 3, 6, 6), (3, 4, 4, 6), (4, 4, 4, 4)
}

theorem egyptian_fraction_solutions :
  {(x, y, z, t) : ℕ × ℕ × ℕ × ℕ | 
    x > 0 ∧ y > 0 ∧ z > 0 ∧ t > 0 ∧
    (1 : ℚ) / x + (1 : ℚ) / y + (1 : ℚ) / z + (1 : ℚ) / t = 1 ∧
    x ≤ y ∧ y ≤ z ∧ z ≤ t} = EgyptianFractionSolutions := by
  sorry

end egyptian_fraction_solutions_l218_21871


namespace quadratic_equation_solution_l218_21866

theorem quadratic_equation_solution :
  let x₁ : ℝ := (3 + Real.sqrt 15) / 3
  let x₂ : ℝ := (3 - Real.sqrt 15) / 3
  (3 * x₁^2 - 6 * x₁ - 2 = 0) ∧ (3 * x₂^2 - 6 * x₂ - 2 = 0) := by
  sorry

end quadratic_equation_solution_l218_21866


namespace book_cost_price_l218_21821

theorem book_cost_price (selling_price : ℝ) (profit_percentage : ℝ) (cost_price : ℝ) : 
  selling_price = 270 → 
  profit_percentage = 20 → 
  selling_price = cost_price * (1 + profit_percentage / 100) → 
  cost_price = 225 := by
sorry

end book_cost_price_l218_21821


namespace real_roots_quadratic_equation_l218_21864

theorem real_roots_quadratic_equation (m : ℝ) :
  (∃ x : ℝ, (m - 1) * x^2 + 4 * x + 1 = 0) → m ≤ 5 :=
by sorry

end real_roots_quadratic_equation_l218_21864


namespace exponent_calculation_l218_21861

theorem exponent_calculation : (1 / ((-5^2)^4)) * ((-5)^9) = -5 := by
  sorry

end exponent_calculation_l218_21861


namespace hyperbola_equation_l218_21841

/-- Given a hyperbola with the equation x²/a² - y²/b² = 1, where a > 0 and b > 0,
    one of its asymptotes is perpendicular to the line l: x - 2y - 5 = 0,
    and one of its foci lies on line l,
    prove that the equation of the hyperbola is x²/5 - y²/20 = 1 -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h_asymptote : ∃ (m : ℝ), m * (1/2) = -1 ∧ m = b/a)
  (h_focus : ∃ (x y : ℝ), x - 2*y - 5 = 0 ∧ x^2/a^2 - y^2/b^2 = 1 ∧ x^2 - (a^2 + b^2) = 0) :
  a^2 = 5 ∧ b^2 = 20 :=
sorry

end hyperbola_equation_l218_21841


namespace triple_composition_even_l218_21895

/-- A function f : ℝ → ℝ is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- Given an even function f, prove that f(f(f(x))) is also even -/
theorem triple_composition_even (f : ℝ → ℝ) (hf : IsEven f) : IsEven (fun x ↦ f (f (f x))) := by
  sorry

end triple_composition_even_l218_21895


namespace pen_purchase_cost_l218_21874

/-- The cost of a single brand X pen -/
def brand_x_cost : ℚ := 4

/-- The cost of a single brand Y pen -/
def brand_y_cost : ℚ := 14/5

/-- The number of brand X pens purchased -/
def num_brand_x : ℕ := 8

/-- The total number of pens purchased -/
def total_pens : ℕ := 12

/-- The number of brand Y pens purchased -/
def num_brand_y : ℕ := total_pens - num_brand_x

/-- The total cost of all pens purchased -/
def total_cost : ℚ := num_brand_x * brand_x_cost + num_brand_y * brand_y_cost

theorem pen_purchase_cost : total_cost = 216/5 := by sorry

end pen_purchase_cost_l218_21874


namespace simplify_trig_expression_130_degrees_simplify_trig_expression_second_quadrant_l218_21806

-- Part 1
theorem simplify_trig_expression_130_degrees :
  (Real.sqrt (1 - 2 * Real.sin (130 * π / 180) * Real.cos (130 * π / 180))) /
  (Real.sin (130 * π / 180) + Real.sqrt (1 - Real.sin (130 * π / 180) ^ 2)) = 1 := by
sorry

-- Part 2
theorem simplify_trig_expression_second_quadrant (α : Real) 
  (h : π / 2 < α ∧ α < π) :
  Real.cos α * Real.sqrt ((1 - Real.sin α) / (1 + Real.sin α)) +
  Real.sin α * Real.sqrt ((1 - Real.cos α) / (1 + Real.cos α)) =
  Real.sin α - Real.cos α := by
sorry

end simplify_trig_expression_130_degrees_simplify_trig_expression_second_quadrant_l218_21806


namespace factorial_divisibility_l218_21823

theorem factorial_divisibility (k n : ℕ) (hk : 0 < k ∧ k ≤ 2020) (hn : 0 < n) :
  ¬ (3^((k-1)*n+1) ∣ ((Nat.factorial (k*n) / Nat.factorial n)^2)) := by
  sorry

end factorial_divisibility_l218_21823


namespace midpoint_path_difference_l218_21847

/-- Given a rectangle with sides a and b, and a segment AB of length 4 inside it,
    the path traced by the midpoint C of AB as A completes one revolution around 
    the perimeter is shorter than the perimeter by 16 - 4π. -/
theorem midpoint_path_difference (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hl : 4 < min a b) :
  2 * (a + b) - (2 * (a + b) - 16 + 4 * Real.pi) = 16 - 4 * Real.pi := by
  sorry

end midpoint_path_difference_l218_21847


namespace perfect_square_trinomial_m_l218_21858

/-- A trinomial ax^2 + bx + c is a perfect square if there exists a real number r
    such that ax^2 + bx + c = (√a * x + r)^2 for all x. -/
def IsPerfectSquareTrinomial (a b c : ℝ) : Prop :=
  ∃ r : ℝ, ∀ x : ℝ, a * x^2 + b * x + c = (Real.sqrt a * x + r)^2

theorem perfect_square_trinomial_m (m : ℝ) :
  IsPerfectSquareTrinomial 1 (-m) 16 → m = 8 ∨ m = -8 := by
  sorry

end perfect_square_trinomial_m_l218_21858


namespace complex_equation_proof_l218_21872

theorem complex_equation_proof (z : ℂ) (h : z = 1 - Complex.I) : z^2 - 2*z + 2 = 0 := by
  sorry

end complex_equation_proof_l218_21872


namespace intersection_equals_open_interval_l218_21856

def A : Set ℝ := {x | x^2 - 4*x + 3 < 0}
def B : Set ℝ := {x | 2 < x ∧ x < 4}

theorem intersection_equals_open_interval : A ∩ B = Set.Ioo 2 3 := by sorry

end intersection_equals_open_interval_l218_21856


namespace function_inequality_l218_21883

theorem function_inequality (a : ℝ) : 
  let f (x : ℝ) := (1/3) * x^3 - Real.log (x + 1)
  let g (x : ℝ) := x^2 - 2 * a * x
  (∃ (x₁ : ℝ) (x₂ : ℝ), x₁ ∈ Set.Icc 0 1 ∧ x₂ ∈ Set.Icc 1 2 ∧ 
    (deriv f x₁) ≥ g x₂) →
  a ≥ (1/4 : ℝ) :=
by sorry

end function_inequality_l218_21883


namespace phone_price_calculation_l218_21882

/-- Proves that given specific conditions on phone accessories and contract,
    the phone price that results in a total yearly cost of $3700 is $1000. -/
theorem phone_price_calculation (phone_price : ℝ) : 
  (∀ (monthly_contract case_cost headphones_cost : ℝ),
    monthly_contract = 200 ∧
    case_cost = 0.2 * phone_price ∧
    headphones_cost = 0.5 * case_cost ∧
    phone_price + 12 * monthly_contract + case_cost + headphones_cost = 3700) →
  phone_price = 1000 := by
  sorry

end phone_price_calculation_l218_21882


namespace four_row_grid_has_sixteen_triangles_l218_21819

/-- Represents a triangular grid with a given number of rows at the base -/
structure TriangularGrid where
  baseRows : Nat

/-- Calculates the number of small triangles in a triangular grid -/
def smallTriangles (grid : TriangularGrid) : Nat :=
  (grid.baseRows * (grid.baseRows + 1)) / 2

/-- Calculates the number of medium triangles in a triangular grid -/
def mediumTriangles (grid : TriangularGrid) : Nat :=
  ((grid.baseRows - 1) * grid.baseRows) / 2

/-- Calculates the number of large triangles in a triangular grid -/
def largeTriangles (grid : TriangularGrid) : Nat :=
  if grid.baseRows ≥ 3 then 1 else 0

/-- Calculates the total number of triangles in a triangular grid -/
def totalTriangles (grid : TriangularGrid) : Nat :=
  smallTriangles grid + mediumTriangles grid + largeTriangles grid

/-- Theorem: A triangular grid with 4 rows at the base has 16 total triangles -/
theorem four_row_grid_has_sixteen_triangles :
  totalTriangles { baseRows := 4 } = 16 := by
  sorry

end four_row_grid_has_sixteen_triangles_l218_21819


namespace largest_room_width_l218_21848

theorem largest_room_width (width smallest_width smallest_length largest_length area_difference : ℝ) :
  smallest_width = 15 →
  smallest_length = 8 →
  largest_length = 30 →
  area_difference = 1230 →
  width * largest_length - smallest_width * smallest_length = area_difference →
  width = 45 := by
sorry

end largest_room_width_l218_21848


namespace quadratic_roots_nature_l218_21853

theorem quadratic_roots_nature (k : ℂ) (h : k.re = 0 ∧ k.im ≠ 0) :
  ∃ (z₁ z₂ : ℂ), 10 * z₁^2 - 5 * z₁ - k = 0 ∧
                 10 * z₂^2 - 5 * z₂ - k = 0 ∧
                 z₁.im = 0 ∧
                 z₂.re = 0 ∧ z₂.im ≠ 0 :=
by sorry

end quadratic_roots_nature_l218_21853


namespace sum_of_digits_seven_pow_nineteen_l218_21863

/-- The sum of the tens digit and the ones digit of 7^19 is 7 -/
theorem sum_of_digits_seven_pow_nineteen : 
  (((7^19) / 10) % 10) + ((7^19) % 10) = 7 := by
  sorry

end sum_of_digits_seven_pow_nineteen_l218_21863


namespace water_canteen_count_l218_21830

def water_problem (flow_rate : ℚ) (duration : ℚ) (additional_water : ℚ) (small_canteen_capacity : ℚ) : ℕ :=
  let total_water := flow_rate * duration + additional_water
  (total_water / small_canteen_capacity).ceil.toNat

theorem water_canteen_count :
  water_problem 9 8 7 6 = 14 := by sorry

end water_canteen_count_l218_21830


namespace library_experience_l218_21824

/-- Given two employees' years of experience satisfying certain conditions,
    prove that one employee has 10 years of experience. -/
theorem library_experience (b j : ℝ) 
  (h1 : j - 5 = 3 * (b - 5))
  (h2 : j = 2 * b) : 
  b = 10 := by sorry

end library_experience_l218_21824


namespace proportional_function_graph_l218_21852

/-- A proportional function with coefficient 2 -/
def f (x : ℝ) : ℝ := 2 * x

theorem proportional_function_graph (x y : ℝ) :
  y = f x → (∃ k : ℝ, k > 0 ∧ y = k * x) ∧ f 0 = 0 := by
  sorry

#check proportional_function_graph

end proportional_function_graph_l218_21852


namespace next_number_with_property_l218_21812

/-- A function that splits a four-digit number into its hundreds and tens-ones parts -/
def split_number (n : ℕ) : ℕ × ℕ :=
  (n / 100, n % 100)

/-- A function that checks if a number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

/-- The property we're looking for in the number -/
def has_property (n : ℕ) : Prop :=
  let (a, b) := split_number n
  is_perfect_square (a * b)

theorem next_number_with_property :
  ∀ n : ℕ, 1818 < n → n < 1832 → ¬(has_property n) ∧ has_property 1832 := by
  sorry

#check next_number_with_property

end next_number_with_property_l218_21812


namespace polynomial_factorization_l218_21814

theorem polynomial_factorization (x : ℝ) : 
  x^6 + 2*x^4 - x^2 - 2 = (x - 1) * (x + 1) * (x^2 + 1) * (x^2 + 2) := by
  sorry

end polynomial_factorization_l218_21814


namespace projection_property_l218_21811

def projection (v : ℝ × ℝ) (w : ℝ × ℝ) : ℝ × ℝ := sorry

theorem projection_property :
  ∀ (p : (ℝ × ℝ) → (ℝ × ℝ)),
  (p (2, -4) = (3, -3)) →
  (p = projection (1, -1)) →
  (p (-8, 2) = (-5, 5)) := by sorry

end projection_property_l218_21811


namespace equation_roots_l218_21855

theorem equation_roots (m n : ℝ) (hm : m ≠ 0) 
  (h : 2 * m * (-3)^2 - n * (-3) + 2 = 0) : 
  ∃ (x y : ℝ), 2 * m * x^2 + n * x + 2 = 0 ∧ 2 * m * y^2 + n * y + 2 = 0 :=
sorry

end equation_roots_l218_21855


namespace sequence_formula_main_theorem_l218_21801

def a (n : ℕ+) : ℚ := 1 / ((2 * n.val - 1) * (2 * n.val + 1))

def S (n : ℕ+) : ℚ := sorry

theorem sequence_formula (n : ℕ+) :
  S n / (n.val * (2 * n.val - 1)) = a n ∧ 
  S 1 / (1 * (2 * 1 - 1)) = 1 / 3 :=
by sorry

theorem main_theorem (n : ℕ+) : 
  S n / (n.val * (2 * n.val - 1)) = 1 / ((2 * n.val - 1) * (2 * n.val + 1)) :=
by sorry

end sequence_formula_main_theorem_l218_21801


namespace inequality_proof_l218_21885

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a / Real.sqrt b + b / Real.sqrt a ≥ Real.sqrt a + Real.sqrt b := by
  sorry

end inequality_proof_l218_21885


namespace interview_problem_l218_21899

theorem interview_problem (n : ℕ) 
  (h1 : n ≥ 2) 
  (h2 : (Nat.choose 2 2 * Nat.choose (n - 2) 1) / Nat.choose n 3 = 1 / 70) : 
  n = 21 := by
sorry

end interview_problem_l218_21899


namespace workshop_workers_count_l218_21838

/-- Proves that the total number of workers in a workshop is 24 given specific salary conditions -/
theorem workshop_workers_count : ∀ (W : ℕ) (N : ℕ),
  (W : ℚ) * 8000 = (8 : ℚ) * 12000 + (N : ℚ) * 6000 →  -- total salary equation
  W = 8 + N →                                          -- total workers equation
  W = 24 :=
by
  sorry

end workshop_workers_count_l218_21838


namespace triangular_array_coin_sum_l218_21879

/-- The sum of the first n odd numbers -/
def triangular_sum (n : ℕ) : ℕ := n^2

/-- The sum of the digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem triangular_array_coin_sum :
  ∃ (n : ℕ), triangular_sum n = 3081 ∧ sum_of_digits n = 10 := by
  sorry

end triangular_array_coin_sum_l218_21879


namespace inequality_proof_l218_21828

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x / (x + 5*y)) + (y / (y + 5*x)) ≤ 1 := by
  sorry

end inequality_proof_l218_21828


namespace max_daily_net_income_l218_21850

/-- Represents the daily rental fee for each electric car -/
def x : ℕ → ℕ := fun n => n

/-- Represents the daily net income from renting out electric cars -/
def y : ℕ → ℤ
| n =>
  if 60 ≤ n ∧ n ≤ 90 then
    750 * n - 1700
  else if 90 < n ∧ n ≤ 300 then
    -3 * n * n + 1020 * n - 1700
  else
    0

/-- The theorem stating the maximum daily net income and the corresponding rental fee -/
theorem max_daily_net_income :
  ∃ (n : ℕ), 60 ≤ n ∧ n ≤ 300 ∧ y n = 85000 ∧ n = 170 ∧
  ∀ (m : ℕ), 60 ≤ m ∧ m ≤ 300 → y m ≤ y n :=
sorry

end max_daily_net_income_l218_21850


namespace roden_gold_fish_l218_21836

/-- The number of gold fish Roden bought -/
def gold_fish : ℕ := 22 - 7

/-- Theorem stating that Roden bought 15 gold fish -/
theorem roden_gold_fish : gold_fish = 15 := by
  sorry

end roden_gold_fish_l218_21836


namespace min_value_abc_l218_21845

theorem min_value_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : 1/a + 1/b + 1/c = 9) :
  ∃ (min : ℝ), min = 1/2916 ∧ ∀ x, x = a^3 * b^2 * c → x ≥ min :=
sorry

end min_value_abc_l218_21845


namespace max_abs_f_implies_sum_l218_21877

def f (a b x : ℝ) := x^2 + a*x + b

theorem max_abs_f_implies_sum (a b : ℝ) :
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, |f a b x| ≤ (1/2 : ℝ)) →
  (∃ x ∈ Set.Icc (-1 : ℝ) 1, |f a b x| = (1/2 : ℝ)) →
  4*a + 3*b = -(3/2 : ℝ) := by
sorry

end max_abs_f_implies_sum_l218_21877


namespace homework_time_calculation_l218_21869

theorem homework_time_calculation (jacob_time greg_time patrick_time : ℕ) : 
  jacob_time = 18 →
  greg_time = jacob_time - 6 →
  patrick_time = 2 * greg_time - 4 →
  jacob_time + greg_time + patrick_time = 50 := by
  sorry

end homework_time_calculation_l218_21869


namespace smallest_shift_l218_21839

/-- A function with period 30 -/
def periodic_function (g : ℝ → ℝ) : Prop :=
  ∀ x, g (x - 30) = g x

/-- The shift property for g(x/4) -/
def shift_property (g : ℝ → ℝ) (b : ℝ) : Prop :=
  ∀ x, g ((x - b) / 4) = g (x / 4)

/-- The theorem stating the smallest positive b is 120 -/
theorem smallest_shift (g : ℝ → ℝ) (h : periodic_function g) :
  ∃ b : ℝ, b > 0 ∧ shift_property g b ∧ ∀ b' : ℝ, b' > 0 → shift_property g b' → b ≤ b' :=
sorry

end smallest_shift_l218_21839


namespace max_sin_cos_sum_l218_21800

theorem max_sin_cos_sum (A : Real) : 2 * Real.sin (A / 2) + Real.cos (A / 2) ≤ Real.sqrt 3 := by
  sorry

end max_sin_cos_sum_l218_21800


namespace probability_at_least_one_head_and_three_l218_21837

def coin_flip : Nat := 2
def die_sides : Nat := 8

def coin_success : Nat := 3  -- number of successful coin flip outcomes (HH, HT, TH)
def die_success : Nat := 1   -- number of successful die roll outcomes (3)

def total_outcomes : Nat := coin_flip^2 * die_sides
def successful_outcomes : Nat := coin_success * die_success

theorem probability_at_least_one_head_and_three :
  (successful_outcomes : ℚ) / total_outcomes = 3 / 32 := by
  sorry

end probability_at_least_one_head_and_three_l218_21837


namespace thirty_percent_less_than_80_l218_21875

theorem thirty_percent_less_than_80 : 
  80 * (1 - 0.3) = (224 / 5) * (1 + 1 / 4) := by sorry

end thirty_percent_less_than_80_l218_21875


namespace deepak_present_age_l218_21802

/-- The ratio of ages between Rahul, Deepak, and Sameer -/
def age_ratio : Fin 3 → ℕ
  | 0 => 4  -- Rahul
  | 1 => 3  -- Deepak
  | 2 => 5  -- Sameer

/-- The number of years in the future we're considering -/
def years_future : ℕ := 6

/-- Rahul's age after the specified number of years -/
def rahul_future_age : ℕ := 26

/-- Proves that given the age ratio and Rahul's future age, Deepak's present age is 15 years -/
theorem deepak_present_age :
  ∃ (k : ℕ),
    (age_ratio 0 * k + years_future = rahul_future_age) ∧
    (age_ratio 1 * k = 15) := by
  sorry

end deepak_present_age_l218_21802


namespace max_value_x_4_minus_3x_l218_21888

theorem max_value_x_4_minus_3x :
  ∃ (max : ℝ), max = 4/3 ∧
  (∀ x : ℝ, 0 < x → x < 4/3 → x * (4 - 3 * x) ≤ max) ∧
  (∃ x : ℝ, 0 < x ∧ x < 4/3 ∧ x * (4 - 3 * x) = max) := by
  sorry

end max_value_x_4_minus_3x_l218_21888


namespace negation_existence_real_l218_21892

theorem negation_existence_real : 
  (¬ ∃ x : ℝ, x > 1) ↔ (∀ x : ℝ, x ≤ 1) := by sorry

end negation_existence_real_l218_21892


namespace max_area_square_with_perimeter_32_l218_21834

/-- The maximum area of a square with a perimeter of 32 meters is 64 square meters. -/
theorem max_area_square_with_perimeter_32 :
  let perimeter : ℝ := 32
  let side_length : ℝ := perimeter / 4
  let area : ℝ := side_length ^ 2
  area = 64 := by sorry

end max_area_square_with_perimeter_32_l218_21834


namespace fraction_equality_l218_21890

theorem fraction_equality (p q r s : ℝ) 
  (h : (p - q) * (r - s) / ((q - r) * (s - p)) = -3/7) : 
  (p - r) * (q - s) / ((p - q) * (r - s)) = 1 := by
  sorry

end fraction_equality_l218_21890


namespace article_pages_count_l218_21842

-- Define the constants
def total_word_limit : ℕ := 48000
def large_font_words_per_page : ℕ := 1800
def small_font_words_per_page : ℕ := 2400
def large_font_pages : ℕ := 4

-- Define the theorem
theorem article_pages_count :
  let words_in_large_font := large_font_pages * large_font_words_per_page
  let remaining_words := total_word_limit - words_in_large_font
  let small_font_pages := remaining_words / small_font_words_per_page
  large_font_pages + small_font_pages = 21 := by
sorry

end article_pages_count_l218_21842


namespace cookie_distribution_probability_l218_21804

/-- Represents the number of cookies of each type -/
def num_cookies_per_type : ℕ := 4

/-- Represents the total number of cookies -/
def total_cookies : ℕ := 3 * num_cookies_per_type

/-- Represents the number of students -/
def num_students : ℕ := 4

/-- Represents the number of cookies each student receives -/
def cookies_per_student : ℕ := 3

/-- Calculates the probability of a specific distribution of cookies -/
def probability_distribution (n : ℕ) : ℚ :=
  (num_cookies_per_type * (num_cookies_per_type - 1) * (num_cookies_per_type - 2)) /
  ((total_cookies - n * cookies_per_student + 2) *
   (total_cookies - n * cookies_per_student + 1) *
   (total_cookies - n * cookies_per_student))

/-- The main theorem stating the probability of each student getting one cookie of each type -/
theorem cookie_distribution_probability :
  (probability_distribution 0 * probability_distribution 1 * probability_distribution 2) = 81 / 3850 := by
  sorry

end cookie_distribution_probability_l218_21804


namespace cricket_team_captain_age_l218_21897

theorem cricket_team_captain_age 
  (team_size : ℕ) 
  (captain_age wicket_keeper_age : ℕ) 
  (team_average_age : ℚ) 
  (remaining_players_average_age : ℚ) :
  team_size = 11 →
  wicket_keeper_age = captain_age + 7 →
  team_average_age = 23 →
  remaining_players_average_age = team_average_age - 1 →
  (team_size : ℚ) * team_average_age = 
    ((team_size - 2) : ℚ) * remaining_players_average_age + captain_age + wicket_keeper_age →
  captain_age = 24 := by
sorry

end cricket_team_captain_age_l218_21897


namespace minimum_amount_is_1000_l218_21894

/-- The minimum amount of the sell to get a discount -/
def minimum_amount_for_discount (
  item_count : ℕ) 
  (item_cost : ℚ) 
  (discounted_total : ℚ) 
  (discount_rate : ℚ) : ℚ :=
  item_count * item_cost - (item_count * item_cost - discounted_total) / discount_rate

/-- Theorem stating the minimum amount for discount is $1000 -/
theorem minimum_amount_is_1000 : 
  minimum_amount_for_discount 7 200 1360 (1/10) = 1000 := by
  sorry

end minimum_amount_is_1000_l218_21894


namespace max_subsets_l218_21815

-- Define the set S
def S : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

-- Define the property for subsets A₁, A₂, ..., Aₖ
def valid_subsets (A : Finset (Finset ℕ)) : Prop :=
  ∀ X ∈ A, X ⊆ S ∧ X.card = 5 ∧ ∀ Y ∈ A, X ≠ Y → (X ∩ Y).card ≤ 2

-- Theorem statement
theorem max_subsets :
  ∀ A : Finset (Finset ℕ), valid_subsets A → A.card ≤ 6 :=
sorry

end max_subsets_l218_21815


namespace second_smallest_three_digit_in_pascal_l218_21816

/-- Pascal's Triangle coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

/-- Checks if a number is three digits -/
def isThreeDigit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

/-- The smallest three-digit number in Pascal's Triangle -/
def smallestThreeDigit : ℕ := 100

/-- The row where the smallest three-digit number first appears -/
def smallestThreeDigitRow : ℕ := 100

theorem second_smallest_three_digit_in_pascal :
  ∃ (n : ℕ), isThreeDigit n ∧
    (∀ (m : ℕ), isThreeDigit m → m < n → m = smallestThreeDigit) ∧
    (∃ (row : ℕ), binomial row 1 = n ∧
      ∀ (r : ℕ), r < row → ¬(∃ (k : ℕ), isThreeDigit (binomial r k) ∧ binomial r k = n)) ∧
    n = 101 ∧ row = 101 :=
sorry

end second_smallest_three_digit_in_pascal_l218_21816


namespace max_value_of_f_l218_21891

/-- The function we're maximizing -/
def f (t : ℤ) : ℚ := (3^t - 2*t) * t / 9^t

/-- The theorem stating the maximum value of the function -/
theorem max_value_of_f :
  ∃ (max : ℚ), max = 1/8 ∧ ∀ (t : ℤ), f t ≤ max :=
sorry

end max_value_of_f_l218_21891


namespace discount_is_25_percent_l218_21867

-- Define the cost of one photocopy
def cost_per_copy : ℚ := 2 / 100

-- Define the number of copies for each person
def copies_per_person : ℕ := 80

-- Define the total number of copies in the combined order
def total_copies : ℕ := 2 * copies_per_person

-- Define the savings per person
def savings_per_person : ℚ := 40 / 100

-- Define the total savings
def total_savings : ℚ := 2 * savings_per_person

-- Define the total cost without discount
def total_cost_without_discount : ℚ := total_copies * cost_per_copy

-- Define the total cost with discount
def total_cost_with_discount : ℚ := total_cost_without_discount - total_savings

-- Define the discount percentage
def discount_percentage : ℚ := (total_savings / total_cost_without_discount) * 100

-- Theorem statement
theorem discount_is_25_percent : discount_percentage = 25 := by
  sorry

end discount_is_25_percent_l218_21867


namespace phone_number_theorem_l218_21860

def phone_number_count (n : ℕ) (k : ℕ) : ℕ := 2^n

theorem phone_number_theorem : phone_number_count 5 2 = 32 := by
  sorry

end phone_number_theorem_l218_21860


namespace four_digit_number_property_l218_21849

theorem four_digit_number_property : ∃ (a b c d : ℕ), 
  (a ≥ 1 ∧ a ≤ 9) ∧ 
  (b ≥ 0 ∧ b ≤ 9) ∧ 
  (c ≥ 0 ∧ c ≤ 9) ∧ 
  (d ≥ 0 ∧ d ≤ 9) ∧ 
  (a * 1000 + b * 100 + c * 10 + d ≥ 1000) ∧
  (a * 1000 + b * 100 + c * 10 + d ≤ 9999) ∧
  ((a + b + c + d) * (a * b * c * d) = 3990) :=
by sorry

end four_digit_number_property_l218_21849


namespace max_floors_is_fourteen_fourteen_floors_is_feasible_l218_21831

/-- Represents a building with elevators -/
structure Building where
  num_elevators : ℕ
  num_floors : ℕ
  stops_per_elevator : ℕ
  every_two_floors_connected : Bool

/-- The conditions of our specific building -/
def our_building : Building := {
  num_elevators := 7,
  num_floors := 14,  -- We'll prove this is the maximum
  stops_per_elevator := 6,
  every_two_floors_connected := true
}

/-- The theorem stating that 14 is the maximum number of floors -/
theorem max_floors_is_fourteen (b : Building) 
  (h1 : b.num_elevators = 7)
  (h2 : b.stops_per_elevator = 6)
  (h3 : b.every_two_floors_connected = true) :
  b.num_floors ≤ 14 := by
  sorry

/-- The theorem stating that 14 floors is feasible -/
theorem fourteen_floors_is_feasible (b : Building) 
  (h1 : b.num_elevators = 7)
  (h2 : b.stops_per_elevator = 6)
  (h3 : b.every_two_floors_connected = true) :
  ∃ (b' : Building), b'.num_floors = 14 ∧ 
    b'.num_elevators = b.num_elevators ∧ 
    b'.stops_per_elevator = b.stops_per_elevator ∧ 
    b'.every_two_floors_connected = b.every_two_floors_connected := by
  sorry

end max_floors_is_fourteen_fourteen_floors_is_feasible_l218_21831


namespace sum_of_fractions_inequality_l218_21854

theorem sum_of_fractions_inequality (a b c : ℝ) (h : a * b * c = 1) :
  (1 / (2 * a^2 + b^2 + 3)) + (1 / (2 * b^2 + c^2 + 3)) + (1 / (2 * c^2 + a^2 + 3)) ≤ 1/2 := by
  sorry

end sum_of_fractions_inequality_l218_21854


namespace Q_subset_P_l218_21833

def P : Set ℝ := {x | x ≥ -1}
def Q : Set ℝ := {y | y ≥ 0}

theorem Q_subset_P : Q ⊆ P := by sorry

end Q_subset_P_l218_21833


namespace birthday_party_ratio_l218_21886

theorem birthday_party_ratio (total_guests : ℕ) (men : ℕ) (stayed : ℕ) : 
  total_guests = 60 →
  men = 15 →
  stayed = 50 →
  (total_guests / 2 : ℕ) + men + (total_guests - (total_guests / 2 + men)) = total_guests →
  (total_guests - stayed - 5 : ℕ) / men = 1 / 3 :=
by sorry

end birthday_party_ratio_l218_21886


namespace rectangular_hall_area_l218_21880

/-- Calculates the area of a rectangular hall given its length and breadth ratio. -/
def hall_area (length : ℝ) (breadth_ratio : ℝ) : ℝ :=
  length * (breadth_ratio * length)

/-- Theorem: The area of a rectangular hall with length 60 meters and breadth
    two-thirds of its length is 2400 square meters. -/
theorem rectangular_hall_area :
  hall_area 60 (2/3) = 2400 := by
  sorry

end rectangular_hall_area_l218_21880


namespace problem_statement_l218_21884

-- Define lg as the base-10 logarithm
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem problem_statement : 8^(2/3) + lg 25 - lg (1/4) = 6 := by
  sorry

end problem_statement_l218_21884


namespace rectangle_diagonal_l218_21887

/-- Given a rectangle with perimeter 72 meters and length-to-width ratio of 5:2,
    its diagonal length is 194/7 meters. -/
theorem rectangle_diagonal (length width : ℝ) : 
  (2 * (length + width) = 72) →
  (length / width = 5 / 2) →
  Real.sqrt (length^2 + width^2) = 194 / 7 := by
sorry

end rectangle_diagonal_l218_21887


namespace fraction_zero_implies_x_equals_one_l218_21808

theorem fraction_zero_implies_x_equals_one (x : ℝ) :
  (x^2 - 1) / (x + 1) = 0 ∧ x + 1 ≠ 0 → x = 1 := by
  sorry

end fraction_zero_implies_x_equals_one_l218_21808


namespace sqrt_11_parts_sum_l218_21825

theorem sqrt_11_parts_sum (x y : ℝ) : 
  (x = ⌊Real.sqrt 11⌋) → 
  (y = Real.sqrt 11 - x) → 
  (2 * x * y + y^2 = 2) := by
  sorry

end sqrt_11_parts_sum_l218_21825


namespace alex_sandwiches_l218_21851

/-- The number of different sandwiches Alex can make -/
def num_sandwiches (num_meats : ℕ) (num_cheeses : ℕ) : ℕ :=
  num_meats * (num_cheeses.choose 3)

/-- Theorem: Alex can make 1760 different sandwiches -/
theorem alex_sandwiches :
  num_sandwiches 8 12 = 1760 := by
  sorry

end alex_sandwiches_l218_21851


namespace friendly_iff_ge_seven_l218_21826

def is_friendly (n : ℕ) : Prop :=
  n ≥ 2 ∧
  ∃ (A : Fin n → Set (Fin n)),
    (∀ i, i ∉ A i) ∧
    (∀ i j, i ≠ j → (i ∈ A j ↔ j ∉ A i)) ∧
    (∀ i j, (A i ∩ A j).Nonempty)

theorem friendly_iff_ge_seven :
  ∀ n : ℕ, is_friendly n ↔ n ≥ 7 := by sorry

end friendly_iff_ge_seven_l218_21826


namespace power_of_two_sum_l218_21878

theorem power_of_two_sum (x : ℕ) : 2^x + 2^x + 2^x + 2^x + 2^x = 2048 → x = 9 := by
  sorry

end power_of_two_sum_l218_21878


namespace gcd_power_minus_one_l218_21896

theorem gcd_power_minus_one (m n : ℕ+) :
  Nat.gcd ((2 ^ m.val) - 1) ((2 ^ n.val) - 1) = (2 ^ Nat.gcd m.val n.val) - 1 := by
  sorry

end gcd_power_minus_one_l218_21896


namespace min_nSn_l218_21817

/-- Represents an arithmetic sequence and its properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum function
  m : ℕ      -- Given index
  h_m : m ≥ 2
  h_sum_pred : S (m - 1) = -2
  h_sum : S m = 0
  h_sum_succ : S (m + 1) = 3

/-- The minimum value of nS_n for the given arithmetic sequence -/
theorem min_nSn (seq : ArithmeticSequence) : 
  ∃ (k : ℝ), k = -9 ∧ ∀ (n : ℕ), n * seq.S n ≥ k :=
sorry

end min_nSn_l218_21817


namespace similar_triangle_perimeter_l218_21844

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- Checks if a triangle is isosceles -/
def Triangle.isIsosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.c = t.a

/-- Calculates the perimeter of a triangle -/
def Triangle.perimeter (t : Triangle) : ℝ :=
  t.a + t.b + t.c

/-- Checks if two triangles are similar -/
def areSimilar (t1 t2 : Triangle) : Prop :=
  ∃ (k : ℝ), k > 0 ∧ t2.a = k * t1.a ∧ t2.b = k * t1.b ∧ t2.c = k * t1.c

theorem similar_triangle_perimeter
  (t1 : Triangle)
  (h1 : t1.isIsosceles)
  (h2 : t1.a = 12 ∧ t1.b = 12 ∧ t1.c = 18)
  (t2 : Triangle)
  (h3 : areSimilar t1 t2)
  (h4 : min t2.a (min t2.b t2.c) = 30) :
  t2.perimeter = 120 := by
  sorry

end similar_triangle_perimeter_l218_21844


namespace abc_def_ratio_l218_21829

theorem abc_def_ratio (a b c d e f : ℝ) 
  (h1 : a / b = 1 / 3)
  (h2 : b / c = 2)
  (h3 : c / d = 1 / 2)
  (h4 : d / e = 3)
  (h5 : e / f = 1 / 10) :
  a * b * c / (d * e * f) = 1 / 20 := by
  sorry

end abc_def_ratio_l218_21829


namespace theater_attendance_l218_21813

theorem theater_attendance
  (adult_ticket_price : ℕ)
  (child_ticket_price : ℕ)
  (total_revenue : ℕ)
  (num_children : ℕ)
  (h1 : adult_ticket_price = 8)
  (h2 : child_ticket_price = 1)
  (h3 : total_revenue = 50)
  (h4 : num_children = 18) :
  adult_ticket_price * (total_revenue - child_ticket_price * num_children) / adult_ticket_price + num_children = 22 :=
by
  sorry

end theater_attendance_l218_21813


namespace AE_length_l218_21846

-- Define the points A, B, C, D, E, and M on a line
variable (A B C D E M : ℝ)

-- Define the conditions
axiom divide_four_equal : B - A = C - B ∧ C - B = D - C ∧ D - C = E - D
axiom M_midpoint : M - A = E - M
axiom MC_length : M - C = 12

-- Theorem to prove
theorem AE_length : E - A = 48 := by
  sorry

end AE_length_l218_21846


namespace brittany_rebecca_age_difference_l218_21803

/-- The age difference between Brittany and Rebecca -/
def ageDifference (rebecca_age : ℕ) (brittany_age_after_vacation : ℕ) (vacation_duration : ℕ) : ℕ :=
  brittany_age_after_vacation - vacation_duration - rebecca_age

/-- Proof that Brittany is 3 years older than Rebecca -/
theorem brittany_rebecca_age_difference :
  ageDifference 25 32 4 = 3 := by
  sorry

end brittany_rebecca_age_difference_l218_21803


namespace function_max_value_l218_21873

open Real

theorem function_max_value (a : ℝ) (h1 : a > 0) :
  let f : ℝ → ℝ := λ x ↦ Real.log x - a * x
  (∀ x ∈ Set.Icc 1 (Real.exp 1), f x ≤ -4) ∧
  (∃ x ∈ Set.Icc 1 (Real.exp 1), f x = -4) →
  a = 4 := by
  sorry

end function_max_value_l218_21873


namespace data_transmission_time_l218_21862

/-- Proves that the time to send 80 blocks of 400 chunks each at 160 chunks per second is 3 minutes -/
theorem data_transmission_time :
  let num_blocks : ℕ := 80
  let chunks_per_block : ℕ := 400
  let transmission_rate : ℕ := 160
  let total_chunks : ℕ := num_blocks * chunks_per_block
  let transmission_time_seconds : ℕ := total_chunks / transmission_rate
  let transmission_time_minutes : ℚ := transmission_time_seconds / 60
  transmission_time_minutes = 3 := by
  sorry

end data_transmission_time_l218_21862


namespace value_two_std_dev_below_mean_l218_21868

/-- Given a normal distribution with mean 14.5 and standard deviation 1.7,
    the value that is exactly 2 standard deviations less than the mean is 11.1 -/
theorem value_two_std_dev_below_mean :
  let μ : ℝ := 14.5  -- mean
  let σ : ℝ := 1.7   -- standard deviation
  μ - 2 * σ = 11.1 := by
  sorry

end value_two_std_dev_below_mean_l218_21868


namespace jumps_per_meter_l218_21807

/-- Given the relationships between different units of length, 
    this theorem proves how many jumps are in one meter. -/
theorem jumps_per_meter 
  (x y a b p q s t : ℚ) 
  (hops_to_skips : x * 1 = y)
  (jumps_to_hops : a = b * 1)
  (skips_to_leaps : p * 1 = q)
  (leaps_to_meters : s = t * 1)
  (x_pos : 0 < x) (y_pos : 0 < y) (a_pos : 0 < a) (b_pos : 0 < b)
  (p_pos : 0 < p) (q_pos : 0 < q) (s_pos : 0 < s) (t_pos : 0 < t) :
  1 = (s * p * x * a) / (t * q * y * b) :=
sorry

end jumps_per_meter_l218_21807
