import Mathlib

namespace parallel_vectors_sum_l3313_331393

theorem parallel_vectors_sum (m n : ℝ) : 
  let a : Fin 3 → ℝ := ![(-2 : ℝ), 3, -1]
  let b : Fin 3 → ℝ := ![4, m, n]
  (∃ (k : ℝ), k ≠ 0 ∧ (∀ i, b i = k * a i)) →
  m + n = -4 := by
sorry

end parallel_vectors_sum_l3313_331393


namespace angle_conversion_correct_l3313_331399

/-- The number of clerts in a full circle on Mars -/
def mars_full_circle : ℕ := 400

/-- The number of degrees in a full circle on Earth -/
def earth_full_circle : ℕ := 360

/-- The number of degrees in the angle we're converting -/
def angle_to_convert : ℕ := 45

/-- The number of clerts corresponding to the given angle on Earth -/
def clerts_in_angle : ℕ := 50

theorem angle_conversion_correct : 
  (angle_to_convert : ℚ) / earth_full_circle * mars_full_circle = clerts_in_angle :=
sorry

end angle_conversion_correct_l3313_331399


namespace special_polygon_properties_l3313_331347

/-- A polygon where the sum of interior angles is twice the sum of exterior angles -/
structure SpecialPolygon where
  sides : ℕ
  sum_interior_angles : ℝ
  sum_exterior_angles : ℝ
  interior_exterior_relation : sum_interior_angles = 2 * sum_exterior_angles

theorem special_polygon_properties (p : SpecialPolygon) :
  p.sum_interior_angles = 720 ∧ p.sides = 6 := by
  sorry

end special_polygon_properties_l3313_331347


namespace problem_solution_l3313_331329

theorem problem_solution (x y : ℝ) (h : |x - 3| + Real.sqrt (x - y + 1) = 0) :
  Real.sqrt (x^2 * y + x * y^2 + 1/4 * y^3) = 10 := by
  sorry

end problem_solution_l3313_331329


namespace faye_age_l3313_331373

/-- Represents the ages of the people in the problem -/
structure Ages where
  diana : ℕ
  eduardo : ℕ
  chad : ℕ
  faye : ℕ

/-- The conditions of the problem -/
def problem_conditions (ages : Ages) : Prop :=
  ages.diana = ages.eduardo - 4 ∧
  ages.eduardo = ages.chad + 5 ∧
  ages.faye = ages.chad + 4 ∧
  ages.diana = 18

/-- The theorem stating that under the given conditions, Faye is 21 years old -/
theorem faye_age (ages : Ages) : problem_conditions ages → ages.faye = 21 := by
  sorry

end faye_age_l3313_331373


namespace clock_angles_at_3_and_6_l3313_331309

/-- The angle between the hour hand and minute hand of a clock at a given time -/
def clock_angle (hour : ℕ) (minute : ℕ) : ℝ :=
  sorry

theorem clock_angles_at_3_and_6 :
  (clock_angle 3 0 = 90) ∧ (clock_angle 6 0 = 180) := by
  sorry

end clock_angles_at_3_and_6_l3313_331309


namespace cost_per_person_l3313_331362

theorem cost_per_person (num_friends : ℕ) (total_cost : ℚ) (cost_per_person : ℚ) : 
  num_friends = 15 → 
  total_cost = 13500 → 
  cost_per_person = total_cost / num_friends → 
  cost_per_person = 900 := by
sorry

end cost_per_person_l3313_331362


namespace min_value_of_quadratic_squared_l3313_331377

theorem min_value_of_quadratic_squared (x : ℝ) : 
  ∃ (y : ℝ), (x^2 + 6*x + 2)^2 ≥ 0 ∧ (y^2 + 6*y + 2)^2 = 0 := by
  sorry

end min_value_of_quadratic_squared_l3313_331377


namespace complex_sum_zero_l3313_331382

theorem complex_sum_zero : 
  let x : ℂ := 2 * Complex.I / (1 - Complex.I)
  let n : ℕ := 2016
  (Finset.sum (Finset.range n) (fun k => Nat.choose n (k + 1) * x ^ (k + 1))) = 0 := by
  sorry

end complex_sum_zero_l3313_331382


namespace triangle_inequality_l3313_331374

theorem triangle_inequality (a b c : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_area : (1/2) * a * b * Real.sqrt (1 - ((a^2 + b^2 - c^2) / (2*a*b))^2) = 1/4)
  (h_circumradius : (a * b * c) / (4 * (1/4)) = 1) : 
  (1/a + 1/b + 1/c) > (Real.sqrt a + Real.sqrt b + Real.sqrt c) := by
  sorry

#check triangle_inequality

end triangle_inequality_l3313_331374


namespace serenity_new_shoes_l3313_331397

theorem serenity_new_shoes (pairs_bought : ℕ) (shoes_per_pair : ℕ) :
  pairs_bought = 3 →
  shoes_per_pair = 2 →
  pairs_bought * shoes_per_pair = 6 :=
by
  sorry

end serenity_new_shoes_l3313_331397


namespace younger_person_age_is_29_l3313_331388

/-- The age difference between Brittany and the other person -/
def age_difference : ℕ := 3

/-- The duration of Brittany's vacation -/
def vacation_duration : ℕ := 4

/-- Brittany's age when she returns from vacation -/
def brittany_age_after_vacation : ℕ := 32

/-- The age of the person who is younger than Brittany -/
def younger_person_age : ℕ := brittany_age_after_vacation - vacation_duration - age_difference

theorem younger_person_age_is_29 : younger_person_age = 29 := by
  sorry

end younger_person_age_is_29_l3313_331388


namespace complex_multiplication_l3313_331307

theorem complex_multiplication (i : ℂ) : i * i = -1 → i * (3 + 4*i) = -4 + 3*i := by
  sorry

end complex_multiplication_l3313_331307


namespace triangle_side_length_l3313_331372

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  a / Real.sin A = b / Real.sin B ∧
  a = 1 ∧
  A = π / 6 ∧
  B = π / 3 →
  b = Real.sqrt 3 := by
sorry

end triangle_side_length_l3313_331372


namespace sin_160_equals_sin_20_l3313_331346

theorem sin_160_equals_sin_20 : Real.sin (160 * π / 180) = Real.sin (20 * π / 180) := by
  sorry

end sin_160_equals_sin_20_l3313_331346


namespace special_gp_ratio_equation_special_gp_ratio_value_l3313_331389

/-- A geometric progression with positive terms where any term is equal to the square of the sum of the next two following terms -/
structure SpecialGeometricProgression where
  a : ℝ  -- first term
  r : ℝ  -- common ratio
  a_pos : a > 0
  r_pos : r > 0
  special_property : ∀ n : ℕ, a * r^n = (a * r^(n+1) + a * r^(n+2))^2

/-- The common ratio of a special geometric progression satisfies a specific equation -/
theorem special_gp_ratio_equation (gp : SpecialGeometricProgression) :
  gp.r^4 + 2 * gp.r^3 + gp.r^2 - 1 = 0 :=
sorry

/-- The positive solution to the equation r^4 + 2r^3 + r^2 - 1 = 0 is approximately 0.618 -/
theorem special_gp_ratio_value :
  ∃ r : ℝ, r > 0 ∧ r^4 + 2 * r^3 + r^2 - 1 = 0 ∧ abs (r - 0.618) < 0.001 :=
sorry

end special_gp_ratio_equation_special_gp_ratio_value_l3313_331389


namespace line_passes_through_fixed_point_l3313_331350

/-- A line in the form kx - y - k + 1 = 0 passes through the point (1, 1) for all real k -/
theorem line_passes_through_fixed_point :
  ∀ (k : ℝ), (k * 1 - 1 - k + 1 = 0) := by sorry

end line_passes_through_fixed_point_l3313_331350


namespace cubic_polynomial_theorem_l3313_331337

/-- Given a, b, c are roots of x³ + 4x² + 6x + 8 = 0 -/
def cubic_roots (a b c : ℝ) : Prop :=
  a^3 + 4*a^2 + 6*a + 8 = 0 ∧
  b^3 + 4*b^2 + 6*b + 8 = 0 ∧
  c^3 + 4*c^2 + 6*c + 8 = 0

/-- Q is a cubic polynomial satisfying the given conditions -/
def Q_conditions (Q : ℝ → ℝ) (a b c : ℝ) : Prop :=
  (∃ p q r s : ℝ, ∀ x, Q x = p*x^3 + q*x^2 + r*x + s) ∧
  Q a = b + c ∧
  Q b = a + c ∧
  Q c = a + b ∧
  Q (a + b + c) = -20

theorem cubic_polynomial_theorem (a b c : ℝ) (Q : ℝ → ℝ) 
  (h1 : cubic_roots a b c) (h2 : Q_conditions Q a b c) :
  ∀ x, Q x = 5/4*x^3 + 4*x^2 + 17/4*x + 2 :=
by sorry

end cubic_polynomial_theorem_l3313_331337


namespace max_gcd_triangular_number_l3313_331385

def triangular_number (n : ℕ+) : ℕ := (n * (n + 1)) / 2

theorem max_gcd_triangular_number :
  ∃ (k : ℕ+), ∀ (n : ℕ+), Nat.gcd (6 * triangular_number n) (n - 2) ≤ 12 ∧
  Nat.gcd (6 * triangular_number k) (k - 2) = 12 :=
sorry

end max_gcd_triangular_number_l3313_331385


namespace population_function_time_to_reach_1_2_million_max_growth_rate_20_years_l3313_331392

-- Define the initial population and growth rate
def initial_population : ℝ := 1000000
def annual_growth_rate : ℝ := 0.012

-- Define the population function
def population (years : ℕ) : ℝ := initial_population * (1 + annual_growth_rate) ^ years

-- Theorem 1: Population function
theorem population_function (years : ℕ) : 
  population years = 100 * (1.012 ^ years) * 10000 := by sorry

-- Theorem 2: Time to reach 1.2 million
theorem time_to_reach_1_2_million : 
  ∃ y : ℕ, y ≥ 16 ∧ y < 17 ∧ population y ≥ 1200000 ∧ population (y-1) < 1200000 := by sorry

-- Theorem 3: Maximum growth rate for 20 years
theorem max_growth_rate_20_years (max_rate : ℝ) : 
  (∀ rate : ℝ, rate ≤ max_rate → initial_population * (1 + rate) ^ 20 ≤ 1200000) ↔ 
  max_rate ≤ 0.009 := by sorry

end population_function_time_to_reach_1_2_million_max_growth_rate_20_years_l3313_331392


namespace arithmetic_sequence_middle_term_l3313_331375

/-- 
Given an arithmetic sequence where the first term is 3^2 and the third term is 3^4,
prove that the second term is 45.
-/
theorem arithmetic_sequence_middle_term : 
  ∀ (a : ℕ → ℤ), 
  (a 0 = 3^2) → 
  (a 2 = 3^4) → 
  (∀ i j k, i < j → j < k → a j - a i = a k - a j) → 
  (a 1 = 45) := by
sorry

end arithmetic_sequence_middle_term_l3313_331375


namespace sum_of_min_max_FGH_is_23_l3313_331334

/-- Represents a single digit (0-9) -/
def SingleDigit : Type := { n : ℕ // n < 10 }

/-- Represents a number in the form F861G20H -/
def NumberFGH (F G H : SingleDigit) : ℕ := 
  F.1 * 100000000 + 861 * 100000 + G.1 * 10000 + 20 * 100 + H.1

/-- Condition that F861G20H is divisible by 11 -/
def IsDivisibleBy11 (F G H : SingleDigit) : Prop :=
  NumberFGH F G H % 11 = 0

theorem sum_of_min_max_FGH_is_23 :
  ∃ (Fmin Gmin Hmin Fmax Gmax Hmax : SingleDigit),
    (∀ F G H : SingleDigit, IsDivisibleBy11 F G H →
      Fmin.1 + Gmin.1 + Hmin.1 ≤ F.1 + G.1 + H.1 ∧
      F.1 + G.1 + H.1 ≤ Fmax.1 + Gmax.1 + Hmax.1) ∧
    Fmin.1 + Gmin.1 + Hmin.1 + Fmax.1 + Gmax.1 + Hmax.1 = 23 :=
sorry

end sum_of_min_max_FGH_is_23_l3313_331334


namespace area_ABC_is_72_l3313_331339

def X : ℝ × ℝ := (6, 0)
def Y : ℝ × ℝ := (8, 4)
def Z : ℝ × ℝ := (10, 0)

def area_triangle (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  0.5 * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

theorem area_ABC_is_72 :
  let area_XYZ := area_triangle X Y Z
  let area_ABC := area_XYZ / 0.1111111111111111
  area_ABC = 72 := by sorry

end area_ABC_is_72_l3313_331339


namespace water_fountain_build_time_l3313_331379

/-- Represents the work rate for building water fountains -/
def work_rate (men : ℕ) (length : ℕ) (days : ℕ) : ℚ :=
  length / (men * days)

/-- Theorem stating the relationship between different teams building water fountains -/
theorem water_fountain_build_time 
  (men1 : ℕ) (length1 : ℕ) (days1 : ℕ)
  (men2 : ℕ) (length2 : ℕ) (days2 : ℕ)
  (h1 : men1 = 20) (h2 : length1 = 56) (h3 : days1 = 7)
  (h4 : men2 = 35) (h5 : length2 = 42) (h6 : days2 = 3) :
  work_rate men1 length1 days1 = work_rate men2 length2 days2 :=
by sorry

#check water_fountain_build_time

end water_fountain_build_time_l3313_331379


namespace sine_fraction_simplification_l3313_331312

theorem sine_fraction_simplification (b : Real) (h : b = 2 * Real.pi / 13) :
  (Real.sin (4 * b) * Real.sin (8 * b) * Real.sin (10 * b) * Real.sin (12 * b) * Real.sin (14 * b)) /
  (Real.sin b * Real.sin (2 * b) * Real.sin (4 * b) * Real.sin (6 * b) * Real.sin (10 * b)) =
  Real.sin (10 * Real.pi / 13) / Real.sin (4 * Real.pi / 13) := by
  sorry

end sine_fraction_simplification_l3313_331312


namespace complex_division_pure_imaginary_l3313_331376

theorem complex_division_pure_imaginary (a : ℝ) : 
  let z₁ : ℂ := a + 3 * Complex.I
  let z₂ : ℂ := 3 - 4 * Complex.I
  (∃ (b : ℝ), z₁ / z₂ = b * Complex.I) → a = 4 := by
  sorry

end complex_division_pure_imaginary_l3313_331376


namespace marble_combinations_l3313_331313

-- Define the number of marbles
def total_marbles : ℕ := 9

-- Define the number of marbles to choose
def chosen_marbles : ℕ := 4

-- Define the combination function
def combination (n k : ℕ) : ℕ := Nat.choose n k

-- Theorem statement
theorem marble_combinations : combination total_marbles chosen_marbles = 126 := by
  sorry

end marble_combinations_l3313_331313


namespace repeating_decimal_sum_l3313_331305

/-- Represents a repeating decimal where the digit repeats indefinitely after the decimal point. -/
def RepeatingDecimal (digit : ℕ) : ℚ :=
  (digit : ℚ) / 9

/-- The sum of 0.4444... and 0.7777... is equal to 11/9. -/
theorem repeating_decimal_sum :
  RepeatingDecimal 4 + RepeatingDecimal 7 = 11 / 9 := by
  sorry

end repeating_decimal_sum_l3313_331305


namespace angle_equation_solutions_l3313_331381

theorem angle_equation_solutions (θ : Real) : 
  0 ≤ θ ∧ θ ≤ π ∧ Real.sqrt 2 * (Real.cos (2 * θ)) = Real.cos θ + Real.sin θ → 
  θ = π / 12 ∨ θ = 3 * π / 4 := by
  sorry

end angle_equation_solutions_l3313_331381


namespace eight_integer_pairs_satisfy_equation_l3313_331371

theorem eight_integer_pairs_satisfy_equation :
  ∃! (s : Finset (ℕ × ℕ)), 
    (∀ (x y : ℕ), (x, y) ∈ s ↔ x > 0 ∧ y > 0 ∧ Real.sqrt (x * y) - 71 * Real.sqrt x + 30 = 0) ∧
    s.card = 8 := by
  sorry

end eight_integer_pairs_satisfy_equation_l3313_331371


namespace range_of_f_l3313_331352

def f (x : Int) : Int := x + 1

def domain : Set Int := {-1, 1, 2}

theorem range_of_f :
  {y : Int | ∃ x ∈ domain, f x = y} = {0, 2, 3} := by
  sorry

end range_of_f_l3313_331352


namespace complex_repair_cost_is_50_l3313_331394

/-- Represents Jim's bike shop financials for a month -/
structure BikeShop where
  tire_repair_price : ℕ
  tire_repair_cost : ℕ
  tire_repairs_count : ℕ
  complex_repair_price : ℕ
  complex_repairs_count : ℕ
  retail_profit : ℕ
  fixed_expenses : ℕ
  total_profit : ℕ

/-- Calculates the cost of parts for each complex repair -/
def complex_repair_cost (shop : BikeShop) : ℕ :=
  let tire_repair_profit := (shop.tire_repair_price - shop.tire_repair_cost) * shop.tire_repairs_count
  let complex_repairs_revenue := shop.complex_repair_price * shop.complex_repairs_count
  let total_revenue := tire_repair_profit + shop.retail_profit + complex_repairs_revenue
  let profit_before_complex_costs := total_revenue - shop.fixed_expenses
  let complex_repairs_profit := shop.total_profit - (profit_before_complex_costs - complex_repairs_revenue)
  (complex_repairs_revenue - complex_repairs_profit) / shop.complex_repairs_count

theorem complex_repair_cost_is_50 (shop : BikeShop)
  (h1 : shop.tire_repair_price = 20)
  (h2 : shop.tire_repair_cost = 5)
  (h3 : shop.tire_repairs_count = 300)
  (h4 : shop.complex_repair_price = 300)
  (h5 : shop.complex_repairs_count = 2)
  (h6 : shop.retail_profit = 2000)
  (h7 : shop.fixed_expenses = 4000)
  (h8 : shop.total_profit = 3000) :
  complex_repair_cost shop = 50 := by
  sorry

end complex_repair_cost_is_50_l3313_331394


namespace maggie_plant_books_l3313_331302

/-- The number of books about plants Maggie bought -/
def num_plant_books : ℕ := 9

/-- The number of books about fish Maggie bought -/
def num_fish_books : ℕ := 1

/-- The number of science magazines Maggie bought -/
def num_magazines : ℕ := 10

/-- The cost of each book in dollars -/
def book_cost : ℕ := 15

/-- The cost of each magazine in dollars -/
def magazine_cost : ℕ := 2

/-- The total amount Maggie spent in dollars -/
def total_spent : ℕ := 170

theorem maggie_plant_books :
  num_plant_books * book_cost + num_fish_books * book_cost + num_magazines * magazine_cost = total_spent :=
by sorry

end maggie_plant_books_l3313_331302


namespace sum_of_prime_factors_of_3_pow_6_minus_1_l3313_331300

theorem sum_of_prime_factors_of_3_pow_6_minus_1 : 
  (Finset.sum (Finset.filter Nat.Prime (Finset.range ((3^6 - 1) + 1))) id) = 22 := by
  sorry

end sum_of_prime_factors_of_3_pow_6_minus_1_l3313_331300


namespace polynomial_remainder_l3313_331348

-- Define the polynomial
def f (x : ℝ) : ℝ := 8 * x^3 - 20 * x^2 + 28 * x - 30

-- Define the divisor
def g (x : ℝ) : ℝ := 4 * x - 8

-- Theorem statement
theorem polynomial_remainder : 
  ∃ (q : ℝ → ℝ), ∀ x, f x = g x * q x + 10 := by
  sorry

end polynomial_remainder_l3313_331348


namespace no_valid_list_exists_l3313_331330

theorem no_valid_list_exists : ¬ ∃ (list : List ℤ), 
  (list.length = 10) ∧ 
  (∀ i j k, i + 1 = j ∧ j + 1 = k → i < list.length ∧ k < list.length → 
    (list.get ⟨i, sorry⟩ * list.get ⟨j, sorry⟩ * list.get ⟨k, sorry⟩) % 6 = 0) ∧
  (∀ i j, i + 1 = j → j < list.length → 
    (list.get ⟨i, sorry⟩ * list.get ⟨j, sorry⟩) % 6 ≠ 0) :=
by sorry

end no_valid_list_exists_l3313_331330


namespace symmetric_point_l3313_331317

/-- The line of symmetry -/
def line_of_symmetry (x y : ℝ) : Prop := x - y - 1 = 0

/-- Check if two points are symmetric with respect to a line -/
def is_symmetric (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  -- The line connecting the points is perpendicular to the line of symmetry
  (y₂ - y₁) / (x₂ - x₁) = -1 ∧
  -- The midpoint of the two points lies on the line of symmetry
  line_of_symmetry ((x₁ + x₂) / 2) ((y₁ + y₂) / 2)

/-- Theorem: The point (5, -4) is symmetric to (-3, 4) with respect to the line x-y-1=0 -/
theorem symmetric_point : is_symmetric (-3) 4 5 (-4) :=
sorry

end symmetric_point_l3313_331317


namespace cross_arrangement_sum_l3313_331365

/-- A type representing digits from 0 to 9 -/
inductive Digit : Type
| zero | one | two | three | four | five | six | seven | eight | nine

/-- Convert a Digit to its natural number value -/
def digitToNat (d : Digit) : Nat :=
  match d with
  | Digit.zero => 0
  | Digit.one => 1
  | Digit.two => 2
  | Digit.three => 3
  | Digit.four => 4
  | Digit.five => 5
  | Digit.six => 6
  | Digit.seven => 7
  | Digit.eight => 8
  | Digit.nine => 9

/-- The cross shape arrangement of digits -/
structure CrossArrangement :=
  (a b c d e f g : Digit)
  (all_different : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧
                   b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧
                   c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧
                   d ≠ e ∧ d ≠ f ∧ d ≠ g ∧
                   e ≠ f ∧ e ≠ g ∧
                   f ≠ g)
  (vertical_sum : digitToNat a + digitToNat b + digitToNat c = 25)
  (horizontal_sum : digitToNat d + digitToNat e + digitToNat f + digitToNat g = 17)

theorem cross_arrangement_sum (arr : CrossArrangement) :
  digitToNat arr.a + digitToNat arr.b + digitToNat arr.c +
  digitToNat arr.d + digitToNat arr.e + digitToNat arr.f + digitToNat arr.g = 33 :=
by sorry

end cross_arrangement_sum_l3313_331365


namespace calculate_initial_weight_l3313_331321

/-- Calculates the initial weight of a person on a constant weight loss diet -/
theorem calculate_initial_weight 
  (current_weight : ℝ) 
  (future_weight : ℝ) 
  (months_to_future : ℝ) 
  (months_on_diet : ℝ) 
  (h1 : current_weight > future_weight) 
  (h2 : months_to_future > 0) 
  (h3 : months_on_diet > 0) :
  ∃ (initial_weight : ℝ),
    initial_weight = current_weight + (current_weight - future_weight) / months_to_future * months_on_diet :=
by
  sorry

#check calculate_initial_weight

end calculate_initial_weight_l3313_331321


namespace chocolate_gum_pricing_l3313_331338

theorem chocolate_gum_pricing (c g : ℝ) 
  (h : (2 * c > 5 * g ∧ 3 * c ≤ 8 * g) ∨ (2 * c ≤ 5 * g ∧ 3 * c > 8 * g)) :
  7 * c < 19 * g := by
  sorry

end chocolate_gum_pricing_l3313_331338


namespace exchange_rate_scaling_l3313_331322

theorem exchange_rate_scaling (x : ℝ) :
  2994 * 14.5 = 177 → 29.94 * 1.45 = 0.177 := by
  sorry

end exchange_rate_scaling_l3313_331322


namespace two_tshirts_per_package_l3313_331395

/-- Given a number of packages and a total number of t-shirts, 
    calculate the number of t-shirts per package -/
def tshirts_per_package (num_packages : ℕ) (total_tshirts : ℕ) : ℕ :=
  total_tshirts / num_packages

/-- Theorem: Given 28 packages and 56 total t-shirts, 
    each package contains 2 t-shirts -/
theorem two_tshirts_per_package :
  tshirts_per_package 28 56 = 2 := by
  sorry

end two_tshirts_per_package_l3313_331395


namespace arc_length_240_degrees_l3313_331351

theorem arc_length_240_degrees (r : ℝ) (θ : ℝ) (l : ℝ) : 
  r = 10 → θ = 240 → l = (θ * π * r) / 180 → l = (40 / 3) * π :=
by sorry

end arc_length_240_degrees_l3313_331351


namespace sector_central_angle_l3313_331344

theorem sector_central_angle (arc_length : Real) (radius : Real) (central_angle : Real) :
  arc_length = 4 * Real.pi ∧ radius = 8 →
  arc_length = (central_angle * Real.pi * radius) / 180 →
  central_angle = 90 := by
  sorry

end sector_central_angle_l3313_331344


namespace point_p_final_position_point_q_initial_position_l3313_331391

-- Define the movement of point P
def point_p_movement : ℝ := 2

-- Define the movement of point Q
def point_q_movement : ℝ := 3

-- Theorem for point P's final position
theorem point_p_final_position :
  point_p_movement = 2 → 0 + point_p_movement = 2 :=
by sorry

-- Theorem for point Q's initial position
theorem point_q_initial_position :
  point_q_movement = 3 →
  (0 + point_q_movement = 3 ∨ 0 - point_q_movement = -3) :=
by sorry

end point_p_final_position_point_q_initial_position_l3313_331391


namespace minimum_groups_l3313_331310

theorem minimum_groups (n : Nat) (h : n = 29) : 
  Nat.ceil (n / 4 : ℚ) = 8 := by
  sorry

end minimum_groups_l3313_331310


namespace expansion_coefficient_l3313_331340

theorem expansion_coefficient (n : ℕ) : 
  ((-2)^n : ℤ) + ((-2)^(n-1) : ℤ) * n = -128 ↔ n = 6 := by
  sorry

end expansion_coefficient_l3313_331340


namespace arithmetic_equality_l3313_331370

theorem arithmetic_equality : 8 / 2 - 5 + 3^2 * 2 = 17 := by
  sorry

end arithmetic_equality_l3313_331370


namespace outfit_combinations_l3313_331316

theorem outfit_combinations (short_sleeve : ℕ) (long_sleeve : ℕ) (jeans : ℕ) (formal_trousers : ℕ) :
  short_sleeve = 5 →
  long_sleeve = 3 →
  jeans = 6 →
  formal_trousers = 2 →
  (short_sleeve + long_sleeve) * (jeans + formal_trousers) = 64 :=
by
  sorry

end outfit_combinations_l3313_331316


namespace pencil_distribution_l3313_331366

/-- Given a total number of pencils and pencils per row, calculate the number of rows -/
def calculate_rows (total_pencils : ℕ) (pencils_per_row : ℕ) : ℕ :=
  total_pencils / pencils_per_row

/-- Theorem: Given 6 pencils distributed equally into rows of 3 pencils each, 
    the number of rows created is 2 -/
theorem pencil_distribution :
  calculate_rows 6 3 = 2 := by
  sorry

end pencil_distribution_l3313_331366


namespace parabola_circle_tangency_l3313_331342

-- Define the parabola C
def parabola_C (x y : ℝ) : Prop := y^2 = x

-- Define the circle M
def circle_M (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 1

-- Define a point on the parabola
def point_on_parabola (p : ℝ × ℝ) : Prop := parabola_C p.1 p.2

-- Define tangency of a line to the circle
def line_tangent_to_circle (p q : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), circle_M (p.1 + t * (q.1 - p.1)) (p.2 + t * (q.2 - p.2)) ∧
             ∀ (s : ℝ), s ≠ t → ¬circle_M (p.1 + s * (q.1 - p.1)) (p.2 + s * (q.2 - p.2))

theorem parabola_circle_tangency 
  (A₁ A₂ A₃ : ℝ × ℝ) 
  (h₁ : point_on_parabola A₁) 
  (h₂ : point_on_parabola A₂) 
  (h₃ : point_on_parabola A₃) 
  (h₄ : line_tangent_to_circle A₁ A₂) 
  (h₅ : line_tangent_to_circle A₁ A₃) : 
  line_tangent_to_circle A₂ A₃ := by
  sorry

end parabola_circle_tangency_l3313_331342


namespace constant_speed_walking_time_l3313_331328

/-- Represents the time taken to walk a certain distance at a constant speed -/
structure WalkingTime where
  distance : ℝ
  time : ℝ

/-- Given a constant walking speed, prove that if it takes 30 minutes to walk 4 kilometers,
    then it will take 15 minutes to walk 2 kilometers -/
theorem constant_speed_walking_time 
  (speed : ℝ) 
  (library : WalkingTime) 
  (school : WalkingTime) 
  (h1 : speed > 0)
  (h2 : library.distance = 4)
  (h3 : library.time = 30)
  (h4 : school.distance = 2)
  (h5 : library.distance / library.time = speed)
  (h6 : school.distance / school.time = speed) :
  school.time = 15 := by
  sorry

end constant_speed_walking_time_l3313_331328


namespace product_of_smallest_primes_l3313_331319

def smallest_one_digit_primes : List Nat := [2, 3]
def smallest_two_digit_prime : Nat := 11

theorem product_of_smallest_primes : 
  (smallest_one_digit_primes.prod * smallest_two_digit_prime) = 66 := by
  sorry

end product_of_smallest_primes_l3313_331319


namespace ramsey_theorem_l3313_331327

-- Define a type for people
variable (Person : Type)

-- Define the acquaintance relation
variable (knows : Person → Person → Prop)

-- Axiom: The acquaintance relation is symmetric (mutual)
axiom knows_symmetric : ∀ (a b : Person), knows a b ↔ knows b a

-- Define a group of 6 people
variable (group : Finset Person)
axiom group_size : group.card = 6

-- Main theorem
theorem ramsey_theorem :
  ∃ (subset : Finset Person),
    subset.card = 3 ∧
    subset ⊆ group ∧
    (∀ (a b : Person), a ∈ subset → b ∈ subset → a ≠ b → knows a b) ∨
    (∀ (a b : Person), a ∈ subset → b ∈ subset → a ≠ b → ¬knows a b) :=
sorry

end ramsey_theorem_l3313_331327


namespace fraction_simplification_l3313_331359

theorem fraction_simplification :
  (21 : ℚ) / 25 * 35 / 45 * 75 / 63 = 35 / 9 := by
  sorry

end fraction_simplification_l3313_331359


namespace binary_101101_equals_octal_55_l3313_331383

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

/-- Converts a decimal number to its octal representation -/
def decimal_to_octal (n : ℕ) : List ℕ :=
  if n < 8 then [n]
  else (n % 8) :: decimal_to_octal (n / 8)

/-- The binary representation of 101101 -/
def binary_101101 : List Bool := [true, false, true, true, false, true]

theorem binary_101101_equals_octal_55 :
  decimal_to_octal (binary_to_decimal binary_101101) = [5, 5] := by
  sorry

end binary_101101_equals_octal_55_l3313_331383


namespace lottery_probabilities_l3313_331306

/-- Represents the outcome of a customer's lottery participation -/
inductive LotteryResult
  | Gold
  | Silver
  | NoWin

/-- Models the lottery promotion scenario -/
structure LotteryPromotion where
  totalTickets : Nat
  surveySize : Nat
  noWinRatio : Rat
  silverRatioAmongWinners : Rat

/-- Calculates the probability of at least one gold prize winner among 3 randomly selected customers -/
def probAtLeastOneGold (lp : LotteryPromotion) : Rat :=
  sorry

/-- Calculates the probability that the number of gold prize winners is not more than 
    the number of silver prize winners among 3 randomly selected customers -/
def probGoldNotMoreThanSilver (lp : LotteryPromotion) : Rat :=
  sorry

/-- The main theorem stating the probabilities for the given lottery promotion scenario -/
theorem lottery_probabilities (lp : LotteryPromotion) 
  (h1 : lp.totalTickets = 2000)
  (h2 : lp.surveySize = 30)
  (h3 : lp.noWinRatio = 2/3)
  (h4 : lp.silverRatioAmongWinners = 3/5) :
  probAtLeastOneGold lp = 73/203 ∧ 
  probGoldNotMoreThanSilver lp = 157/203 := by
  sorry

end lottery_probabilities_l3313_331306


namespace solve_equation_l3313_331386

theorem solve_equation (y : ℝ) (h : Real.sqrt (3 / y + 3) = 5 / 3) : y = -27 / 2 := by
  sorry

end solve_equation_l3313_331386


namespace mean_equality_implies_x_value_l3313_331315

theorem mean_equality_implies_x_value :
  let mean1 := (3 + 7 + 15) / 3
  let mean2 := (x + 10) / 2
  mean1 = mean2 → x = 20 / 3 :=
by sorry

end mean_equality_implies_x_value_l3313_331315


namespace isosceles_triangle_perimeter_l3313_331325

theorem isosceles_triangle_perimeter (a b : ℝ) (h1 : a = 2 ∨ a = 5) (h2 : b = 2 ∨ b = 5) (h3 : a ≠ b) :
  ∃ (c : ℝ), c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a ∧ a + b + c = 12 := by
  sorry

end isosceles_triangle_perimeter_l3313_331325


namespace function_property_l3313_331361

-- Define the function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem function_property (h1 : ∀ x y : ℝ, f (x + y) = f x + f y) (h2 : f 6 = 3) : f 7 = 7/2 := by
  sorry

end function_property_l3313_331361


namespace faster_train_length_l3313_331396

/-- Calculates the length of a faster train given the speeds of two trains and the time it takes for the faster train to pass a man in the slower train. -/
theorem faster_train_length
  (faster_speed slower_speed : ℝ)
  (passing_time : ℝ)
  (h1 : faster_speed = 72)
  (h2 : slower_speed = 36)
  (h3 : passing_time = 12)
  (h4 : faster_speed > slower_speed) :
  let relative_speed := faster_speed - slower_speed
  let speed_ms := relative_speed * (5 / 18)
  let train_length := speed_ms * passing_time
  train_length = 120 := by sorry

end faster_train_length_l3313_331396


namespace earth_moon_distance_scientific_notation_l3313_331308

/-- Represents the distance from Earth to Moon in kilometers -/
def earth_moon_distance : ℝ := 384401

/-- Converts a real number to scientific notation with given significant figures -/
def to_scientific_notation (x : ℝ) (sig_figs : ℕ) : ℝ × ℤ :=
  sorry

theorem earth_moon_distance_scientific_notation :
  to_scientific_notation earth_moon_distance 3 = (3.84, 5) :=
sorry

end earth_moon_distance_scientific_notation_l3313_331308


namespace charlie_calculator_problem_l3313_331398

theorem charlie_calculator_problem :
  let original_factor1 : ℚ := 75 / 10000
  let original_factor2 : ℚ := 256 / 10
  let incorrect_result : ℕ := 19200
  (original_factor1 * original_factor2 = 192 / 1000) ∧
  (75 * 256 = incorrect_result) := by
  sorry

end charlie_calculator_problem_l3313_331398


namespace sneakers_discount_proof_l3313_331349

/-- Calculates the membership discount percentage given the original price,
    coupon discount, and final price after both discounts are applied. -/
def membership_discount_percentage (original_price coupon_discount final_price : ℚ) : ℚ :=
  let price_after_coupon := original_price - coupon_discount
  let discount_amount := price_after_coupon - final_price
  (discount_amount / price_after_coupon) * 100

/-- Proves that the membership discount percentage is 10% for the given scenario. -/
theorem sneakers_discount_proof :
  membership_discount_percentage 120 10 99 = 10 := by
  sorry

end sneakers_discount_proof_l3313_331349


namespace division_remainder_and_primality_l3313_331357

theorem division_remainder_and_primality : 
  let dividend := 5432109
  let divisor := 125
  let remainder := dividend % divisor
  (remainder = 84) ∧ ¬(Nat.Prime remainder) := by
  sorry

end division_remainder_and_primality_l3313_331357


namespace tangent_line_parabola_l3313_331369

/-- The value of d for which the line y = 3x + d is tangent to the parabola y^2 = 12x -/
theorem tangent_line_parabola : 
  ∃ d : ℝ, (∀ x y : ℝ, y = 3*x + d ∧ y^2 = 12*x → 
    ∃! x₀ : ℝ, 3*x₀ + d = (12*x₀).sqrt ∧ 
    ∀ x : ℝ, x ≠ x₀ → 3*x + d ≠ (12*x).sqrt) → 
  d = 1 :=
sorry

end tangent_line_parabola_l3313_331369


namespace petyas_fruits_l3313_331390

theorem petyas_fruits (total : ℕ) (apples oranges tangerines : ℕ) : 
  total = 20 →
  apples = 6 * tangerines →
  apples > oranges →
  apples + oranges + tangerines = total →
  oranges = 6 :=
by
  sorry

end petyas_fruits_l3313_331390


namespace no_zeros_in_larger_interval_l3313_331311

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the property of having a unique zero in the given intervals
def has_unique_zero_in_intervals (f : ℝ → ℝ) : Prop :=
  ∃! x, f x = 0 ∧ 
    0 < x ∧ x < 16 ∧
    0 < x ∧ x < 8 ∧
    0 < x ∧ x < 4 ∧
    0 < x ∧ x < 2

-- State the theorem
theorem no_zeros_in_larger_interval 
  (h : has_unique_zero_in_intervals f) : 
  ∀ x ∈ Set.Icc 2 16, f x ≠ 0 := by
  sorry


end no_zeros_in_larger_interval_l3313_331311


namespace triangle_distance_sum_l3313_331387

-- Define a triangle in 2D space
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define a function to calculate distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define a function to check if a point is inside a triangle
def isInside (t : Triangle) (p : ℝ × ℝ) : Prop := sorry

-- Define a function to calculate the perimeter of a triangle
def perimeter (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem triangle_distance_sum (t : Triangle) (M : ℝ × ℝ) :
  isInside t M →
  distance M t.A + distance M t.B + distance M t.C > perimeter t / 2 := by
  sorry

end triangle_distance_sum_l3313_331387


namespace angle_B_measure_l3313_331335

theorem angle_B_measure :
  ∀ (A B : ℝ),
  A + B = 180 →  -- complementary angles sum to 180°
  B = 4 * A →    -- B is 4 times A
  B = 144 :=     -- B measures 144°
by
  sorry

end angle_B_measure_l3313_331335


namespace sin_cos_45_degrees_l3313_331356

theorem sin_cos_45_degrees : 
  Real.sin (π / 4) = 1 / Real.sqrt 2 ∧ Real.cos (π / 4) = 1 / Real.sqrt 2 := by
  sorry

end sin_cos_45_degrees_l3313_331356


namespace ned_video_game_earnings_l3313_331343

/-- Calculates the total money earned from selling video games --/
def totalEarnings (totalGames : ℕ) (nonWorkingGames : ℕ) 
                  (firstGroupSize : ℕ) (firstGroupPrice : ℕ)
                  (secondGroupSize : ℕ) (secondGroupPrice : ℕ)
                  (remainingPrice : ℕ) : ℕ :=
  let workingGames := totalGames - nonWorkingGames
  let remainingGames := workingGames - firstGroupSize - secondGroupSize
  firstGroupSize * firstGroupPrice + 
  secondGroupSize * secondGroupPrice + 
  remainingGames * remainingPrice

/-- Theorem stating the total earnings from selling the working games --/
theorem ned_video_game_earnings : 
  totalEarnings 25 8 5 9 7 12 15 = 204 := by
  sorry

end ned_video_game_earnings_l3313_331343


namespace number_division_problem_l3313_331304

theorem number_division_problem : ∃ x : ℚ, x / 11 + 156 = 178 ∧ x = 242 := by
  sorry

end number_division_problem_l3313_331304


namespace probability_different_suits_l3313_331378

def deck_size : ℕ := 60
def num_suits : ℕ := 5
def cards_per_suit : ℕ := 12

theorem probability_different_suits :
  let remaining_cards : ℕ := deck_size - 1
  let cards_not_same_suit : ℕ := remaining_cards - (cards_per_suit - 1)
  (cards_not_same_suit : ℚ) / remaining_cards = 48 / 59 :=
by sorry

end probability_different_suits_l3313_331378


namespace calculation_proof_l3313_331323

theorem calculation_proof :
  (5 / (-5/3) * (-2) = 6) ∧
  (-(1^2) + 3 * (-2)^2 + (-9) / (-1/3)^2 = -70) := by
  sorry

end calculation_proof_l3313_331323


namespace ab_value_l3313_331380

theorem ab_value (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 33) : a * b = 18 := by
  sorry

end ab_value_l3313_331380


namespace star_three_five_l3313_331364

-- Define the star operation
def star (x y : ℝ) : ℝ := x^2 + 2*x*y + y^2

-- State the theorem
theorem star_three_five : star 3 5 = 64 := by sorry

end star_three_five_l3313_331364


namespace quadratic_equation_distinct_roots_l3313_331320

theorem quadratic_equation_distinct_roots (k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ k * x^2 - 2 * x - 1 = 0 ∧ k * y^2 - 2 * y - 1 = 0) ↔ 
  (k > -1 ∧ k ≠ 0) :=
by sorry

end quadratic_equation_distinct_roots_l3313_331320


namespace merck_hourly_rate_l3313_331384

/-- Represents the babysitting data for Layla --/
structure BabysittingData where
  donaldson_hours : ℕ
  merck_hours : ℕ
  hille_hours : ℕ
  total_earnings : ℚ

/-- Calculates the hourly rate for babysitting --/
def hourly_rate (data : BabysittingData) : ℚ :=
  data.total_earnings / (data.donaldson_hours + data.merck_hours + data.hille_hours)

/-- Theorem stating that the hourly rate for the Merck family is $17.0625 --/
theorem merck_hourly_rate (data : BabysittingData) 
  (h1 : data.donaldson_hours = 7)
  (h2 : data.merck_hours = 6)
  (h3 : data.hille_hours = 3)
  (h4 : data.total_earnings = 273) :
  hourly_rate data = 17.0625 := by
  sorry

end merck_hourly_rate_l3313_331384


namespace boat_distance_main_theorem_l3313_331326

/-- The distance between two boats given specific angles and fort height -/
theorem boat_distance (fort_height : ℝ) (angle1 angle2 base_angle : ℝ) : ℝ :=
  let boat_distance := 30
  by
    -- Assuming fort_height = 30, angle1 = 45°, angle2 = 30°, base_angle = 30°
    sorry

/-- Main theorem stating the distance between the boats is 30 meters -/
theorem main_theorem : boat_distance 30 (45 * π / 180) (30 * π / 180) (30 * π / 180) = 30 :=
by
  sorry

end boat_distance_main_theorem_l3313_331326


namespace smallest_integer_above_root_sum_sixth_power_l3313_331336

theorem smallest_integer_above_root_sum_sixth_power :
  ∃ n : ℕ, n = 3323 ∧ (∀ m : ℕ, m < n → (m : ℝ) ≤ (Real.sqrt 5 + Real.sqrt 3)^6) ∧
  n > (Real.sqrt 5 + Real.sqrt 3)^6 := by
  sorry

end smallest_integer_above_root_sum_sixth_power_l3313_331336


namespace square_increasing_on_positive_reals_l3313_331331

theorem square_increasing_on_positive_reals :
  ∀ x₁ x₂ : ℝ, 0 < x₁ → 0 < x₂ → x₁ < x₂ → x₁^2 < x₂^2 := by
  sorry

end square_increasing_on_positive_reals_l3313_331331


namespace race_head_start_l3313_331332

theorem race_head_start (Va Vb D H : ℝ) :
  Va = (30 / 17) * Vb →
  D / Va = (D - H) / Vb →
  H = (13 / 30) * D :=
by sorry

end race_head_start_l3313_331332


namespace handshakes_for_four_and_n_l3313_331368

/-- Number of handshakes for n people when every two people shake hands once -/
def handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

theorem handshakes_for_four_and_n :
  (handshakes 4 = 6) ∧
  (∀ n : ℕ, handshakes n = n * (n - 1) / 2) :=
by sorry

end handshakes_for_four_and_n_l3313_331368


namespace parabola_focus_coordinates_l3313_331363

/-- The focus of the parabola x = -8y^2 has coordinates (-1/32, 0) -/
theorem parabola_focus_coordinates :
  let f : ℝ × ℝ → ℝ := fun (x, y) ↦ x + 8 * y^2
  ∃! p : ℝ × ℝ, p = (-1/32, 0) ∧ 
    (∀ q : ℝ × ℝ, f q = 0 → (q.1 - p.1)^2 + (q.2 - p.2)^2 = (q.2 - 0)^2 + (1/16)^2) :=
by sorry

end parabola_focus_coordinates_l3313_331363


namespace smallest_five_digit_mod_five_l3313_331345

theorem smallest_five_digit_mod_five : ∃ n : ℕ, 
  (n ≥ 10000 ∧ n < 100000) ∧ 
  n % 5 = 4 ∧ 
  ∀ m : ℕ, (m ≥ 10000 ∧ m < 100000 ∧ m % 5 = 4) → n ≤ m :=
by sorry

end smallest_five_digit_mod_five_l3313_331345


namespace find_divisor_l3313_331353

theorem find_divisor (dividend quotient remainder divisor : ℕ) : 
  dividend = 144 →
  quotient = 13 →
  remainder = 1 →
  dividend = divisor * quotient + remainder →
  divisor = 11 :=
by
  sorry

end find_divisor_l3313_331353


namespace min_value_of_abs_sum_l3313_331354

theorem min_value_of_abs_sum (x : ℝ) : 
  |x - 4| + |x + 2| + |x - 5| ≥ -1 ∧ ∃ y : ℝ, |y - 4| + |y + 2| + |y - 5| = -1 := by
  sorry

end min_value_of_abs_sum_l3313_331354


namespace projection_matrix_values_l3313_331367

/-- A 2x2 matrix is a projection matrix if and only if Q^2 = Q -/
def is_projection_matrix (Q : Matrix (Fin 2) (Fin 2) ℚ) : Prop :=
  Q * Q = Q

/-- The specific matrix we're working with -/
def Q (x y : ℚ) : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![x, 21/49], ![y, 35/49]]

/-- The theorem stating the values of x and y that make Q a projection matrix -/
theorem projection_matrix_values :
  ∃ (x y : ℚ), is_projection_matrix (Q x y) ∧ x = 666/2401 ∧ y = (49 * 2401) / 1891 := by
  sorry

end projection_matrix_values_l3313_331367


namespace deposit_calculation_l3313_331318

/-- Calculates the deposit amount given an initial amount -/
def calculateDeposit (initialAmount : ℚ) : ℚ :=
  initialAmount * (30 / 100) * (25 / 100) * (20 / 100)

/-- Proves that the deposit calculation for Rs. 50,000 results in Rs. 750 -/
theorem deposit_calculation :
  calculateDeposit 50000 = 750 := by
  sorry

end deposit_calculation_l3313_331318


namespace peanuts_in_box_l3313_331303

/-- The number of peanuts in a box after adding more -/
def total_peanuts (initial : ℕ) (added : ℕ) : ℕ :=
  initial + added

/-- Theorem: If a box initially contains 4 peanuts and 2 more are added, the total is 6 -/
theorem peanuts_in_box : total_peanuts 4 2 = 6 := by
  sorry

end peanuts_in_box_l3313_331303


namespace three_distinct_roots_condition_l3313_331301

theorem three_distinct_roots_condition (a : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    (|x^3 - a^3| = x - a) ∧
    (|y^3 - a^3| = y - a) ∧
    (|z^3 - a^3| = z - a)) ↔
  (-2 / Real.sqrt 3 < a ∧ a < -1 / Real.sqrt 3) :=
sorry

end three_distinct_roots_condition_l3313_331301


namespace e_neg_4i_in_second_quadrant_l3313_331341

-- Define the complex exponential function
noncomputable def cexp (z : ℂ) : ℂ := Real.exp z.re * (Complex.cos z.im + Complex.I * Complex.sin z.im)

-- Define the quadrants of the complex plane
def in_second_quadrant (z : ℂ) : Prop := z.re < 0 ∧ z.im > 0

-- Theorem statement
theorem e_neg_4i_in_second_quadrant : 
  in_second_quadrant (cexp (-4 * Complex.I)) :=
sorry

end e_neg_4i_in_second_quadrant_l3313_331341


namespace eliminate_denominator_l3313_331358

theorem eliminate_denominator (x : ℝ) : 
  (x + 1) / 3 - 3 = 2 * x + 7 → (x + 1) - 9 = 3 * (2 * x + 7) := by
  sorry

end eliminate_denominator_l3313_331358


namespace community_center_chairs_l3313_331314

/-- Converts a number from base 6 to base 10 -/
def base6ToBase10 (a b c : ℕ) : ℕ := a * 6^2 + b * 6 + c

/-- Calculates the number of chairs needed given the total people and people per chair -/
def calculateChairs (totalPeople : ℕ) (peoplePerChair : ℕ) : ℚ :=
  (totalPeople : ℚ) / peoplePerChair

theorem community_center_chairs :
  let seatingCapacity := base6ToBase10 2 3 1
  let peoplePerChair := 3
  calculateChairs seatingCapacity peoplePerChair = 30.33 := by sorry

end community_center_chairs_l3313_331314


namespace nathan_ate_twenty_gumballs_l3313_331360

/-- The number of gumballs in each package -/
def gumballs_per_package : ℕ := 5

/-- The number of whole boxes Nathan consumed -/
def boxes_consumed : ℕ := 4

/-- The total number of gumballs Nathan ate -/
def gumballs_eaten : ℕ := gumballs_per_package * boxes_consumed

theorem nathan_ate_twenty_gumballs : gumballs_eaten = 20 := by
  sorry

end nathan_ate_twenty_gumballs_l3313_331360


namespace circular_sign_diameter_ratio_l3313_331355

theorem circular_sign_diameter_ratio (d₁ d₂ : ℝ) (h : d₁ > 0 ∧ d₂ > 0) :
  (π * (d₂ / 2)^2) = 49 * (π * (d₁ / 2)^2) → d₂ = 7 * d₁ := by
  sorry

end circular_sign_diameter_ratio_l3313_331355


namespace largest_n_for_sin_cos_inequality_l3313_331324

theorem largest_n_for_sin_cos_inequality : 
  (∀ n : ℕ, n > 8 → ∃ x : ℝ, (Real.sin x)^n + (Real.cos x)^n < 1 / (2 * n)) ∧ 
  (∀ x : ℝ, (Real.sin x)^8 + (Real.cos x)^8 ≥ 1 / 16) := by
  sorry

end largest_n_for_sin_cos_inequality_l3313_331324


namespace midpoint_coordinate_sum_l3313_331333

/-- Given a point M that is the midpoint of AB, and point A,
    prove that the sum of coordinates of B is as expected. -/
theorem midpoint_coordinate_sum (M A B : ℝ × ℝ) : 
  M = (3, 5) →  -- M has coordinates (3,5)
  A = (6, 8) →  -- A has coordinates (6,8)
  M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →  -- M is the midpoint of AB
  B.1 + B.2 = 2 :=  -- The sum of B's coordinates is 2
by
  sorry

#check midpoint_coordinate_sum

end midpoint_coordinate_sum_l3313_331333
