import Mathlib

namespace g_negative_two_equals_negative_fifteen_l2053_205352

-- Define the functions f and g
def f (a b x : ℝ) : ℝ := a * x^3 + b * x - 7
def g (a b x : ℝ) : ℝ := f a b x + 2

-- State the theorem
theorem g_negative_two_equals_negative_fifteen 
  (a b : ℝ) (h : f a b 2 = 3) : g a b (-2) = -15 := by
  sorry

end g_negative_two_equals_negative_fifteen_l2053_205352


namespace floor_sqrt_30_squared_l2053_205347

theorem floor_sqrt_30_squared : ⌊Real.sqrt 30⌋^2 = 25 := by
  sorry

end floor_sqrt_30_squared_l2053_205347


namespace intersection_x_coordinate_l2053_205375

/-- The x-coordinate of the intersection point of y = 9 / (x^2 + 3) and x + y = 3 is 0 -/
theorem intersection_x_coordinate : ∃ y : ℝ, 
  y = 9 / (0^2 + 3) ∧ 0 + y = 3 :=
by sorry

end intersection_x_coordinate_l2053_205375


namespace servant_service_duration_l2053_205396

def yearly_payment : ℕ := 800
def uniform_price : ℕ := 300
def actual_payment : ℕ := 600

def months_served : ℕ := 7

theorem servant_service_duration :
  yearly_payment = 800 ∧
  uniform_price = 300 ∧
  actual_payment = 600 →
  months_served = 7 :=
by sorry

end servant_service_duration_l2053_205396


namespace balanced_domino_config_exists_l2053_205316

/-- A domino configuration on an n × n board. -/
structure DominoConfig (n : ℕ) where
  /-- The number of dominoes in the configuration. -/
  num_dominoes : ℕ
  /-- Predicate that the configuration is balanced. -/
  is_balanced : Prop

/-- The minimum number of dominoes needed for a balanced configuration. -/
def min_dominoes (n : ℕ) : ℕ :=
  if n % 3 = 0 then 2 * n / 3 else 2 * n

/-- Theorem stating the existence of a balanced configuration and the minimum number of dominoes needed. -/
theorem balanced_domino_config_exists (n : ℕ) (h : n ≥ 3) :
  ∃ (config : DominoConfig n), config.is_balanced ∧ config.num_dominoes = min_dominoes n :=
by sorry

end balanced_domino_config_exists_l2053_205316


namespace circle_triangle_perimeter_l2053_205319

structure Circle :=
  (points : Fin 6 → ℝ × ℝ)

structure Triangle :=
  (vertices : Fin 3 → ℝ × ℝ)

def perimeter (t : Triangle) : ℝ := sorry

theorem circle_triangle_perimeter
  (c : Circle)
  (x y z : ℝ × ℝ)
  (h1 : x ∈ Set.Icc (c.points 0) (c.points 3) ∩ Set.Icc (c.points 1) (c.points 4))
  (h2 : y ∈ Set.Icc (c.points 0) (c.points 3) ∩ Set.Icc (c.points 2) (c.points 5))
  (h3 : z ∈ Set.Icc (c.points 2) (c.points 5) ∩ Set.Icc (c.points 1) (c.points 4))
  (h4 : x ∈ Set.Icc z (c.points 1))
  (h5 : x ∈ Set.Icc y (c.points 0))
  (h6 : y ∈ Set.Icc z (c.points 2))
  (h7 : dist (c.points 0) x = 3)
  (h8 : dist (c.points 1) x = 2)
  (h9 : dist (c.points 2) y = 4)
  (h10 : dist (c.points 3) y = 10)
  (h11 : dist (c.points 4) z = 16)
  (h12 : dist (c.points 5) z = 12)
  : perimeter { vertices := ![x, y, z] } = 25/2 := by
  sorry

end circle_triangle_perimeter_l2053_205319


namespace freshman_sample_size_l2053_205358

/-- Calculates the number of students to be sampled from a specific stratum in stratified sampling -/
def stratifiedSampleSize (totalPopulation sampleSize stratumSize : ℕ) : ℕ :=
  (stratumSize * sampleSize) / totalPopulation

/-- The number of students to be sampled from the freshman year in a stratified sampling -/
theorem freshman_sample_size :
  let totalPopulation : ℕ := 4500
  let sampleSize : ℕ := 150
  let freshmanSize : ℕ := 1200
  stratifiedSampleSize totalPopulation sampleSize freshmanSize = 40 := by
sorry

#eval stratifiedSampleSize 4500 150 1200

end freshman_sample_size_l2053_205358


namespace price_decrease_revenue_unchanged_l2053_205332

theorem price_decrease_revenue_unchanged (P U : ℝ) (h_positive : P > 0 ∧ U > 0) :
  let new_price := 0.8 * P
  let new_units := U / 0.8
  let percent_decrease_price := 20
  let percent_increase_units := (new_units - U) / U * 100
  P * U = new_price * new_units →
  percent_increase_units / percent_decrease_price = 1.25 := by
sorry

end price_decrease_revenue_unchanged_l2053_205332


namespace decrease_by_one_point_five_l2053_205360

/-- Represents a linear regression equation of the form y = a + bx -/
structure LinearRegression where
  a : ℝ
  b : ℝ

/-- The change in y when x increases by one unit in a linear regression -/
def change_in_y (lr : LinearRegression) : ℝ := -lr.b

/-- Theorem: In the given linear regression, when x increases by one unit, y decreases by 1.5 units -/
theorem decrease_by_one_point_five :
  let lr : LinearRegression := { a := 2, b := -1.5 }
  change_in_y lr = -1.5 := by sorry

end decrease_by_one_point_five_l2053_205360


namespace fraction_equality_sum_l2053_205340

theorem fraction_equality_sum (P Q : ℚ) : 
  (4 : ℚ) / 7 = P / 49 ∧ (4 : ℚ) / 7 = 56 / Q → P + Q = 126 := by
  sorry

end fraction_equality_sum_l2053_205340


namespace exponential_inequality_l2053_205385

theorem exponential_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : Real.exp a + 2 * a = Real.exp b + 3 * b) : a < b := by
  sorry

end exponential_inequality_l2053_205385


namespace stephen_pizza_percentage_l2053_205323

theorem stephen_pizza_percentage (total_slices : ℕ) (stephen_percentage : ℚ) (pete_percentage : ℚ) (remaining_slices : ℕ) : 
  total_slices = 24 →
  pete_percentage = 1/2 →
  remaining_slices = 9 →
  (1 - stephen_percentage) * total_slices * (1 - pete_percentage) = remaining_slices →
  stephen_percentage = 1/4 := by
sorry

end stephen_pizza_percentage_l2053_205323


namespace complex_number_quadrant_l2053_205373

theorem complex_number_quadrant : ∃ (z : ℂ), z = Complex.I * (1 - Complex.I) ∧ 
  Complex.re z > 0 ∧ Complex.im z > 0 := by
  sorry

end complex_number_quadrant_l2053_205373


namespace triangle_side_value_l2053_205365

theorem triangle_side_value (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = π →
  a / (Real.sin A) = b / (Real.sin B) →
  a / (Real.sin A) = c / (Real.sin C) →
  a^2 - c^2 = 2*b →
  Real.sin A * Real.cos C = 3 * Real.cos A * Real.sin A →
  b = 4 := by sorry

end triangle_side_value_l2053_205365


namespace logarithmic_equation_solution_l2053_205392

theorem logarithmic_equation_solution :
  ∃! x : ℝ, x > 0 ∧ 3^x = x + 50 := by sorry

end logarithmic_equation_solution_l2053_205392


namespace abc_and_fourth_power_sum_l2053_205307

theorem abc_and_fourth_power_sum (a b c : ℝ) 
  (h1 : a + b + c = 1) 
  (h2 : a^2 + b^2 + c^2 = 2) 
  (h3 : a^3 + b^3 + c^3 = 3) : 
  a * b * c = 1/6 ∧ a^4 + b^4 + c^4 = 25/6 := by
  sorry

end abc_and_fourth_power_sum_l2053_205307


namespace megan_popsicles_l2053_205388

/-- The number of Popsicles Megan can finish in a given time period --/
def popsicles_finished (total_minutes : ℕ) (popsicle_time : ℕ) (break_time : ℕ) (break_interval : ℕ) : ℕ :=
  let effective_minutes := total_minutes - (total_minutes / (break_interval * 60)) * break_time
  (effective_minutes / popsicle_time : ℕ)

/-- Theorem stating the number of Popsicles Megan can finish in 5 hours and 40 minutes --/
theorem megan_popsicles :
  popsicles_finished 340 20 5 1 = 15 := by
  sorry

end megan_popsicles_l2053_205388


namespace quadratic_inequality_l2053_205381

theorem quadratic_inequality (a b c : ℝ) : a^2 + a*b + a*c < 0 → b^2 > 4*a*c := by
  sorry

end quadratic_inequality_l2053_205381


namespace difference_of_squares_l2053_205395

theorem difference_of_squares (a b : ℝ) : (a + b) * (a - b) = a^2 - b^2 := by
  sorry

end difference_of_squares_l2053_205395


namespace hyperbola_asymptotes_l2053_205300

/-- Given a hyperbola and a line passing through its right focus, 
    prove the equations of the asymptotes -/
theorem hyperbola_asymptotes 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (C : ℝ × ℝ → Prop) 
  (l : ℝ × ℝ → Prop) 
  (F : ℝ × ℝ) :
  (C = λ (x, y) ↦ x^2 / a^2 - y^2 / b^2 = 1) →
  (l = λ (x, y) ↦ x + 3*y - 2*b = 0) →
  (∃ c, F = (c, 0) ∧ l F) →
  (∃ f : ℝ → ℝ, f x = Real.sqrt 3 / 3 * x ∧ 
   ∀ (x y : ℝ), (C (x, y) → (y = f x ∨ y = -f x))) :=
by sorry

end hyperbola_asymptotes_l2053_205300


namespace cube_iff_greater_l2053_205389

theorem cube_iff_greater (a b : ℝ) : a > b ↔ a^3 > b^3 := by sorry

end cube_iff_greater_l2053_205389


namespace intersection_A_B_l2053_205322

def A : Set ℕ := {1, 2, 3}

def B : Set ℕ := {y | ∃ x ∈ A, y = 2 * x - 1}

theorem intersection_A_B : A ∩ B = {1, 3} := by sorry

end intersection_A_B_l2053_205322


namespace product_and_quotient_cube_square_l2053_205397

theorem product_and_quotient_cube_square (a b k : ℕ) : 
  100 ≤ a * b ∧ a * b < 1000 →  -- three-digit number condition
  a * b = k^3 →                 -- product is cube of k
  (a : ℚ) / b = k^2 →           -- quotient is square of k
  a = 243 ∧ b = 3 ∧ k = 9 := by
sorry

end product_and_quotient_cube_square_l2053_205397


namespace sum_289_37_base4_l2053_205390

/-- Converts a natural number to its base 4 representation as a list of digits -/
def toBase4 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 4) ((m % 4) :: acc)
    aux n []

/-- Checks if a list of natural numbers represents a valid base 4 number -/
def isValidBase4 (l : List ℕ) : Prop :=
  ∀ d ∈ l, d < 4

theorem sum_289_37_base4 :
  let sum := 289 + 37
  let base4Sum := toBase4 sum
  isValidBase4 base4Sum ∧ base4Sum = [1, 1, 0, 1, 2] := by sorry

end sum_289_37_base4_l2053_205390


namespace no_integer_solution_for_ten_l2053_205325

theorem no_integer_solution_for_ten :
  ¬ ∃ (x y z : ℤ), 3 * x^2 + 4 * y^2 - 5 * z^2 = 10 := by
  sorry

end no_integer_solution_for_ten_l2053_205325


namespace cindy_calculation_l2053_205324

theorem cindy_calculation (x : ℚ) : (x - 7) / 5 = 25 → (x - 5) / 7 = 18 + 1 / 7 := by
  sorry

end cindy_calculation_l2053_205324


namespace sum_of_divisors_882_prime_factors_l2053_205329

def sum_of_divisors (n : ℕ) : ℕ := sorry

def count_distinct_prime_factors (n : ℕ) : ℕ := sorry

theorem sum_of_divisors_882_prime_factors :
  count_distinct_prime_factors (sum_of_divisors 882) = 3 := by sorry

end sum_of_divisors_882_prime_factors_l2053_205329


namespace greatest_power_of_seven_in_50_factorial_l2053_205313

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def highest_power_of_seven (n : ℕ) : ℕ :=
  if n < 7 then 0
  else (n / 7) + highest_power_of_seven (n / 7)

theorem greatest_power_of_seven_in_50_factorial :
  ∃ (z : ℕ), z = highest_power_of_seven 50 ∧
  (7^z : ℕ) ∣ factorial 50 ∧
  ∀ (y : ℕ), y > z → ¬((7^y : ℕ) ∣ factorial 50) :=
sorry

end greatest_power_of_seven_in_50_factorial_l2053_205313


namespace four_integer_b_values_l2053_205377

/-- A function that checks if a given integer b results in integer roots for the quadratic equation x^2 + bx + 7b = 0 -/
def has_integer_roots (b : ℤ) : Prop :=
  ∃ p q : ℤ, p + q = -b ∧ p * q = 7 * b

/-- The theorem stating that there are exactly 4 integer values of b for which the quadratic equation x^2 + bx + 7b = 0 always has integer roots -/
theorem four_integer_b_values :
  ∃! (s : Finset ℤ), s.card = 4 ∧ ∀ b : ℤ, has_integer_roots b ↔ b ∈ s :=
sorry

end four_integer_b_values_l2053_205377


namespace sqrt_450_simplification_l2053_205361

theorem sqrt_450_simplification : Real.sqrt 450 = 15 * Real.sqrt 2 := by sorry

end sqrt_450_simplification_l2053_205361


namespace june_birth_percentage_l2053_205334

/-- The total number of scientists -/
def total_scientists : ℕ := 150

/-- The number of scientists born in June -/
def june_scientists : ℕ := 15

/-- The percentage of scientists born in June -/
def june_percentage : ℚ := (june_scientists : ℚ) / (total_scientists : ℚ) * 100

theorem june_birth_percentage :
  june_percentage = 10 := by sorry

end june_birth_percentage_l2053_205334


namespace trapezoid_length_in_divided_square_l2053_205351

/-- Given a square with side length 2 meters, divided into two congruent trapezoids and a quadrilateral,
    where the trapezoids have bases on two sides of the square and their other bases meet at the square's center,
    and all three shapes have equal areas, the length of the longer parallel side of each trapezoid is 5/3 meters. -/
theorem trapezoid_length_in_divided_square :
  let square_side : ℝ := 2
  let total_area : ℝ := square_side ^ 2
  let shape_area : ℝ := total_area / 3
  let shorter_base : ℝ := square_side / 2
  ∃ (longer_base : ℝ),
    longer_base = 5 / 3 ∧
    shape_area = (longer_base + shorter_base) * square_side / 4 :=
by sorry

end trapezoid_length_in_divided_square_l2053_205351


namespace class_size_calculation_l2053_205353

theorem class_size_calculation (E T B N : ℕ) 
  (h1 : E = 55)
  (h2 : T = 85)
  (h3 : N = 30)
  (h4 : B = 20) :
  E + T - B + N = 150 := by
  sorry

end class_size_calculation_l2053_205353


namespace four_birds_joined_l2053_205394

/-- The number of birds that joined the fence -/
def birds_joined (initial_birds final_birds : ℕ) : ℕ :=
  final_birds - initial_birds

/-- Proof that 4 birds joined the fence -/
theorem four_birds_joined :
  let initial_birds : ℕ := 2
  let final_birds : ℕ := 6
  birds_joined initial_birds final_birds = 4 := by
  sorry

end four_birds_joined_l2053_205394


namespace derivative_x_ln_x_l2053_205371

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem derivative_x_ln_x (x : ℝ) (h : x > 0) : 
  deriv f x = 1 + Real.log x :=
sorry

end derivative_x_ln_x_l2053_205371


namespace equation_solution_l2053_205379

theorem equation_solution (x : ℝ) (a b : ℕ) :
  (x^2 + 5*x + 5/x + 1/x^2 = 40) →
  (x = a + Real.sqrt b) →
  (a > 0 ∧ b > 0) →
  (a + b = 11) := by
sorry

end equation_solution_l2053_205379


namespace vertical_throw_meeting_conditions_l2053_205346

/-- Two objects thrown vertically upwards meet under specific conditions -/
theorem vertical_throw_meeting_conditions 
  (g a b τ : ℝ) (τ' : ℝ) (h_g_pos : g > 0) (h_a_pos : a > 0) (h_τ_pos : τ > 0) (h_τ'_pos : τ' > 0) :
  (b > a - g * τ) ∧ 
  (b > a + (g * τ^2 / 2) / (a/g - τ)) ∧ 
  (b ≥ a / Real.sqrt 2) ∧
  (b ≥ -g * τ' / 2 + Real.sqrt (2 * a^2 - g^2 * τ'^2) / 2) := by
  sorry


end vertical_throw_meeting_conditions_l2053_205346


namespace cube_sum_problem_l2053_205301

theorem cube_sum_problem (x y : ℝ) (h1 : x + y = 10) (h2 : x * y = 12) :
  x^3 + y^3 = 640 := by sorry

end cube_sum_problem_l2053_205301


namespace transmission_time_l2053_205303

/-- Proves that given the specified conditions, the transmission time is 5 minutes -/
theorem transmission_time (blocks : ℕ) (chunks_per_block : ℕ) (transmission_rate : ℕ) :
  blocks = 80 →
  chunks_per_block = 640 →
  transmission_rate = 160 →
  (blocks * chunks_per_block : ℝ) / transmission_rate / 60 = 5 := by
  sorry

end transmission_time_l2053_205303


namespace decaf_coffee_percentage_l2053_205391

/-- Proves that the percentage of decaffeinated coffee in the initial stock is 40% --/
theorem decaf_coffee_percentage
  (initial_stock : ℝ)
  (additional_purchase : ℝ)
  (decaf_percent_additional : ℝ)
  (decaf_percent_total : ℝ)
  (h1 : initial_stock = 400)
  (h2 : additional_purchase = 100)
  (h3 : decaf_percent_additional = 60)
  (h4 : decaf_percent_total = 44)
  (h5 : decaf_percent_total / 100 * (initial_stock + additional_purchase) =
        (initial_stock * x / 100) + (additional_purchase * decaf_percent_additional / 100)) :
  x = 40 := by
  sorry

end decaf_coffee_percentage_l2053_205391


namespace circle_bisection_and_symmetric_points_l2053_205318

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 4*y - 4 = 0

-- Define the line l
def line_l (x y k : ℝ) : Prop := y = k*x - 1

-- Define symmetry with respect to a line
def symmetric_wrt_line (x1 y1 x2 y2 k : ℝ) : Prop :=
  (x1 + x2) * (k + 1/k) = (y1 + y2) * (1 - 1/k)

-- Define perpendicularity
def perpendicular (x1 y1 x2 y2 : ℝ) : Prop :=
  x1 * x2 + y1 * y2 = 0

theorem circle_bisection_and_symmetric_points :
  -- Part 1: The line y = -x - 1 bisects the circle
  (∀ x y : ℝ, circle_C x y → line_l x y (-1)) ∧
  -- Part 2: There exist points A and B on the circle satisfying the conditions
  ∃ x1 y1 x2 y2 : ℝ,
    circle_C x1 y1 ∧ circle_C x2 y2 ∧
    symmetric_wrt_line x1 y1 x2 y2 (-1) ∧
    perpendicular x1 y1 x2 y2 ∧
    ((x1 - y1 + 1 = 0 ∧ x2 - y2 + 1 = 0) ∨ (x1 - y1 - 4 = 0 ∧ x2 - y2 - 4 = 0)) :=
sorry

end circle_bisection_and_symmetric_points_l2053_205318


namespace polar_curve_is_parabola_l2053_205310

/-- The curve defined by the polar equation r = 1 / (1 - sin θ) is a parabola -/
theorem polar_curve_is_parabola :
  ∀ r θ x y : ℝ,
  (r = 1 / (1 - Real.sin θ)) →
  (x = r * Real.cos θ) →
  (y = r * Real.sin θ) →
  ∃ a b : ℝ, x^2 = a * y + b :=
by sorry

end polar_curve_is_parabola_l2053_205310


namespace num_tilings_div_by_eight_l2053_205399

/-- A tromino is an L-shaped tile covering exactly three cells -/
structure Tromino :=
  (shape : List (Int × Int))
  (shape_size : shape.length = 3)

/-- A tiling of a square grid using trominos -/
def Tiling (n : Nat) := List (List (Option Tromino))

/-- The size of the square grid -/
def gridSize : Nat := 999

/-- The number of distinct tilings of an n x n grid using trominos -/
def numDistinctTilings (n : Nat) : Nat :=
  sorry

/-- Theorem: The number of distinct tilings of a 999x999 grid using trominos is divisible by 8 -/
theorem num_tilings_div_by_eight :
  ∃ k : Nat, numDistinctTilings gridSize = 8 * k :=
sorry

end num_tilings_div_by_eight_l2053_205399


namespace mari_buttons_l2053_205326

theorem mari_buttons (sue_buttons : ℕ) (kendra_buttons : ℕ) (mari_buttons : ℕ) 
  (h1 : sue_buttons = 6)
  (h2 : sue_buttons = kendra_buttons / 2)
  (h3 : mari_buttons = 5 * kendra_buttons + 4) :
  mari_buttons = 64 := by
  sorry

end mari_buttons_l2053_205326


namespace omega_roots_quadratic_equation_l2053_205372

theorem omega_roots_quadratic_equation :
  ∀ (ω : ℂ) (α β : ℂ),
    ω^5 = 1 →
    ω ≠ 1 →
    α = ω + ω^2 →
    β = ω^3 + ω^4 →
    ∃ (a b : ℝ), ∀ (x : ℂ), x = α ∨ x = β → x^2 + a*x + b = 0 :=
by sorry

end omega_roots_quadratic_equation_l2053_205372


namespace shape_sum_theorem_l2053_205305

-- Define the shapes as real numbers
variable (triangle : ℝ) (circle : ℝ) (square : ℝ)

-- Define the conditions from the problem
def condition1 : Prop := 2 * triangle + 2 * circle + square = 27
def condition2 : Prop := 2 * circle + triangle + square = 26
def condition3 : Prop := 2 * square + triangle + circle = 23

-- Define the theorem
theorem shape_sum_theorem 
  (h1 : condition1 triangle circle square)
  (h2 : condition2 triangle circle square)
  (h3 : condition3 triangle circle square) :
  2 * triangle + 3 * circle + square = 45.5 := by
  sorry

end shape_sum_theorem_l2053_205305


namespace pentagon_coverage_l2053_205320

-- Define a pentagon as a set of 5 points in 2D space
def Pentagon : Type := Fin 5 → ℝ × ℝ

-- Define a function to check if a pentagon is convex
def isConvex (p : Pentagon) : Prop := sorry

-- Define a function to check if all interior angles of a pentagon are obtuse
def allAnglesObtuse (p : Pentagon) : Prop := sorry

-- Define a function to check if a point is inside or on a circle
def isInsideOrOnCircle (center : ℝ × ℝ) (radius : ℝ) (point : ℝ × ℝ) : Prop := sorry

-- Define a function to check if a circle covers a point of the pentagon
def circleCoversPoint (p : Pentagon) (diagonal : Fin 5 × Fin 5) (point : Fin 5) : Prop := sorry

-- Main theorem
theorem pentagon_coverage (p : Pentagon) 
  (h_convex : isConvex p) 
  (h_obtuse : allAnglesObtuse p) : 
  ∃ (d1 d2 : Fin 5 × Fin 5), ∀ (point : Fin 5), 
    circleCoversPoint p d1 point ∨ circleCoversPoint p d2 point := by
  sorry

end pentagon_coverage_l2053_205320


namespace our_equation_is_linear_l2053_205378

/-- Definition of a linear equation in two variables -/
def is_linear_equation (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), ∀ x y, f x y = a * x + b * y - c

/-- The specific equation we want to prove is linear -/
def our_equation (x y : ℝ) : ℝ := x + y - 5

theorem our_equation_is_linear :
  is_linear_equation our_equation :=
sorry

end our_equation_is_linear_l2053_205378


namespace arithmetic_sequence_fourth_term_l2053_205355

/-- Given an arithmetic sequence {a_n} where a_1 = 2, a_2 = 4, and a_3 = 6, 
    prove that the fourth term a_4 = 8. -/
theorem arithmetic_sequence_fourth_term 
  (a : ℕ → ℝ) 
  (h1 : a 1 = 2) 
  (h2 : a 2 = 4) 
  (h3 : a 3 = 6) 
  (h_arith : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1) : 
  a 4 = 8 := by
sorry

end arithmetic_sequence_fourth_term_l2053_205355


namespace segments_5_6_10_form_triangle_l2053_205309

/-- Triangle inequality theorem: the sum of the lengths of any two sides of a triangle
    must be greater than the length of the remaining side. -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- A function that determines if three line segments can form a triangle. -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ triangle_inequality a b c

/-- Theorem stating that the line segments 5, 6, and 10 can form a triangle. -/
theorem segments_5_6_10_form_triangle :
  can_form_triangle 5 6 10 := by sorry

end segments_5_6_10_form_triangle_l2053_205309


namespace geometric_sequence_statements_l2053_205341

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = q * a n

/-- An increasing sequence -/
def IncreasingSequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, n < m → a n < a m

theorem geometric_sequence_statements
    (a : ℕ → ℝ) (q : ℝ) (h : GeometricSequence a q) :
    (¬ (q > 1 → IncreasingSequence a)) ∧
    (¬ (IncreasingSequence a → q > 1)) ∧
    (¬ (q ≤ 1 → ¬IncreasingSequence a)) ∧
    (¬ (¬IncreasingSequence a → q ≤ 1)) :=
  sorry

end geometric_sequence_statements_l2053_205341


namespace sqrt_x_minus_one_defined_l2053_205383

theorem sqrt_x_minus_one_defined (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 1) ↔ x ≥ 1 := by sorry

end sqrt_x_minus_one_defined_l2053_205383


namespace women_married_fraction_l2053_205343

theorem women_married_fraction (total : ℕ) (h1 : total > 0) :
  let women := (64 : ℚ) / 100 * total
  let married := (60 : ℚ) / 100 * total
  let men := total - women
  let single_men := (2 : ℚ) / 3 * men
  let married_men := men - single_men
  let married_women := married - married_men
  married_women / women = (3 : ℚ) / 4 :=
by sorry

end women_married_fraction_l2053_205343


namespace problem_solution_l2053_205362

-- Define the propositions
def p : Prop := ∃ k : ℤ, 0 = 2 * k
def q : Prop := ∃ k : ℤ, 3 = 2 * k

-- Theorem to prove
theorem problem_solution : p ∨ q := by
  sorry

end problem_solution_l2053_205362


namespace fraction_simplification_l2053_205350

theorem fraction_simplification (n : ℕ+) : (n : ℚ) * (3 : ℚ)^(n : ℕ) / (3 : ℚ)^(n : ℕ) = n := by
  sorry

end fraction_simplification_l2053_205350


namespace earthquake_ratio_l2053_205344

def initial_collapse : ℕ := 4
def total_earthquakes : ℕ := 4
def total_collapsed : ℕ := 60

def geometric_sum (a : ℕ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem earthquake_ratio :
  ∃ (r : ℚ), 
    r > 0 ∧ 
    geometric_sum initial_collapse r total_earthquakes = total_collapsed ∧
    r = 2 := by
  sorry

end earthquake_ratio_l2053_205344


namespace max_apartments_five_by_five_l2053_205357

/-- Represents a building with a given number of floors and windows per floor. -/
structure Building where
  floors : ℕ
  windowsPerFloor : ℕ

/-- Calculates the maximum number of apartments in a building. -/
def maxApartments (b : Building) : ℕ :=
  b.floors * b.windowsPerFloor

/-- Theorem stating that for a 5-story building with 5 windows per floor,
    the maximum number of apartments is 25. -/
theorem max_apartments_five_by_five :
  ∀ (b : Building),
    b.floors = 5 →
    b.windowsPerFloor = 5 →
    maxApartments b = 25 := by
  sorry

#check max_apartments_five_by_five

end max_apartments_five_by_five_l2053_205357


namespace train_passing_jogger_time_train_passing_jogger_time_approx_l2053_205348

/-- Time for a train to pass a jogger -/
theorem train_passing_jogger_time 
  (jogger_speed : ℝ) 
  (train_speed : ℝ) 
  (train_length : ℝ) 
  (initial_distance : ℝ) : ℝ :=
  let jogger_speed_ms := jogger_speed * 1000 / 3600
  let train_speed_ms := train_speed * 1000 / 3600
  let relative_speed := train_speed_ms - jogger_speed_ms
  let total_distance := initial_distance + train_length
  total_distance / relative_speed

/-- The time for the train to pass the jogger is approximately 38.75 seconds -/
theorem train_passing_jogger_time_approx :
  ∃ ε > 0, abs (train_passing_jogger_time 8 60 200 360 - 38.75) < ε :=
sorry

end train_passing_jogger_time_train_passing_jogger_time_approx_l2053_205348


namespace f_derivative_at_zero_l2053_205386

def f (x : ℝ) : ℝ := x * (x - 1) * (x - 2)

theorem f_derivative_at_zero : 
  (deriv f) 0 = 2 := by sorry

end f_derivative_at_zero_l2053_205386


namespace range_of_a_l2053_205369

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x - 1| + |x + 1| ≥ 3 * a) →
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → (2 * a - 1) ^ x₁ > (2 * a - 1) ^ x₂) →
  1/2 < a ∧ a ≤ 2/3 :=
by sorry

end range_of_a_l2053_205369


namespace dvd_cost_l2053_205376

/-- Given that two identical DVDs cost $40, prove that five DVDs cost $100. -/
theorem dvd_cost (cost_of_two : ℝ) (h : cost_of_two = 40) :
  5 / 2 * cost_of_two = 100 := by
  sorry

end dvd_cost_l2053_205376


namespace events_mutually_exclusive_to_both_red_l2053_205349

/-- Represents the color of a card -/
inductive Color
  | Red
  | Green
  | Blue

/-- Represents a pair of cards drawn from the bag -/
structure DrawnCards :=
  (first : Color)
  (second : Color)

/-- The bag containing 2 red, 2 green, and 2 blue cards -/
def bag : Multiset Color := 
  2 • {Color.Red} + 2 • {Color.Green} + 2 • {Color.Blue}

/-- Event: Both cards are red -/
def bothRed (draw : DrawnCards) : Prop :=
  draw.first = Color.Red ∧ draw.second = Color.Red

/-- Event: Neither of the 2 cards is red -/
def neitherRed (draw : DrawnCards) : Prop :=
  draw.first ≠ Color.Red ∧ draw.second ≠ Color.Red

/-- Event: Exactly one card is blue -/
def exactlyOneBlue (draw : DrawnCards) : Prop :=
  (draw.first = Color.Blue ∧ draw.second ≠ Color.Blue) ∨
  (draw.first ≠ Color.Blue ∧ draw.second = Color.Blue)

/-- Event: Both cards are green -/
def bothGreen (draw : DrawnCards) : Prop :=
  draw.first = Color.Green ∧ draw.second = Color.Green

theorem events_mutually_exclusive_to_both_red :
  ∀ (draw : DrawnCards),
    (bothRed draw → ¬(neitherRed draw)) ∧
    (bothRed draw → ¬(exactlyOneBlue draw)) ∧
    (bothRed draw → ¬(bothGreen draw)) :=
  sorry

end events_mutually_exclusive_to_both_red_l2053_205349


namespace intersection_distance_l2053_205370

theorem intersection_distance : ∃ (C D : ℝ × ℝ),
  (C.2 = 2 ∧ C.2 = 3 * C.1^2 + 2 * C.1 - 5) ∧
  (D.2 = 2 ∧ D.2 = 3 * D.1^2 + 2 * D.1 - 5) ∧
  C ≠ D ∧
  |C.1 - D.1| = 2 * Real.sqrt 22 / 3 :=
by sorry

end intersection_distance_l2053_205370


namespace hansol_weight_l2053_205315

/-- Given two people, Hanbyul and Hansol, with the following conditions:
    1. The sum of their weights is 88 kg.
    2. Hanbyul weighs 4 kg more than Hansol.
    Prove that Hansol weighs 42 kg. -/
theorem hansol_weight (hanbyul hansol : ℝ) 
    (sum_weight : hanbyul + hansol = 88)
    (weight_diff : hanbyul = hansol + 4) : 
  hansol = 42 := by
  sorry

end hansol_weight_l2053_205315


namespace probability_is_one_fifth_l2053_205304

/-- The probability of finding the last defective product on the fourth inspection -/
def probability_last_defective_fourth_inspection (total : ℕ) (qualified : ℕ) (defective : ℕ) : ℚ :=
  let p1 := qualified / total * (qualified - 1) / (total - 1) * defective / (total - 2) * 1 / (total - 3)
  let p2 := qualified / total * defective / (total - 1) * (qualified - 1) / (total - 2) * 1 / (total - 3)
  let p3 := defective / total * qualified / (total - 1) * (qualified - 1) / (total - 2) * 1 / (total - 3)
  p1 + p2 + p3

/-- Theorem stating that the probability is 1/5 for the given conditions -/
theorem probability_is_one_fifth :
  probability_last_defective_fourth_inspection 6 4 2 = 1/5 := by
  sorry

end probability_is_one_fifth_l2053_205304


namespace frames_cost_l2053_205345

theorem frames_cost (lens_cost : ℝ) (insurance_coverage : ℝ) (coupon : ℝ) (total_cost : ℝ)
  (h1 : lens_cost = 500)
  (h2 : insurance_coverage = 0.8)
  (h3 : coupon = 50)
  (h4 : total_cost = 250)
  : ∃ (frame_cost : ℝ), frame_cost = 200 ∧
    total_cost = (frame_cost - coupon) + (lens_cost * (1 - insurance_coverage)) := by
  sorry

end frames_cost_l2053_205345


namespace cookie_sales_ratio_l2053_205398

theorem cookie_sales_ratio : 
  ∀ (goal : ℕ) (first third fourth fifth left : ℕ),
    goal ≥ 150 →
    first = 5 →
    third = 10 →
    fifth = 10 →
    left = 75 →
    goal - left = first + 4 * first + third + fourth + fifth →
    fourth / third = 3 :=
by
  sorry

end cookie_sales_ratio_l2053_205398


namespace rectangular_pen_max_area_l2053_205368

/-- The perimeter of the rectangular pen -/
def perimeter : ℝ := 60

/-- The maximum possible area of a rectangular pen with the given perimeter -/
def max_area : ℝ := 225

/-- Theorem: The maximum area of a rectangular pen with a perimeter of 60 feet is 225 square feet -/
theorem rectangular_pen_max_area : 
  ∀ (width height : ℝ), 
  width > 0 → height > 0 → 
  2 * (width + height) = perimeter → 
  width * height ≤ max_area :=
by
  sorry

end rectangular_pen_max_area_l2053_205368


namespace fraction_to_decimal_l2053_205338

theorem fraction_to_decimal : (47 : ℚ) / (2 * 5^3) = 0.188 := by sorry

end fraction_to_decimal_l2053_205338


namespace parabola_standard_equation_l2053_205364

/-- Given a parabola with focus F(a,0) where a < 0, its standard equation is y^2 = 4ax -/
theorem parabola_standard_equation (a : ℝ) (h : a < 0) :
  ∃ (x y : ℝ), y^2 = 4*a*x :=
sorry

end parabola_standard_equation_l2053_205364


namespace smallest_multiple_l2053_205336

theorem smallest_multiple (n : ℕ) : n = 187 ↔ 
  n > 0 ∧ 
  17 ∣ n ∧ 
  n % 53 = 7 ∧ 
  ∀ m : ℕ, m > 0 → 17 ∣ m → m % 53 = 7 → n ≤ m :=
by sorry

end smallest_multiple_l2053_205336


namespace sum_of_ages_is_100_l2053_205384

/-- Given the conditions about Alice, Ben, and Charlie's ages, prove that the sum of their ages is 100. -/
theorem sum_of_ages_is_100 (A B C : ℕ) 
  (h1 : A = 20 + B + C) 
  (h2 : A^2 = 2000 + (B + C)^2) : 
  A + B + C = 100 := by
  sorry

end sum_of_ages_is_100_l2053_205384


namespace deck_cost_is_32_l2053_205306

/-- Calculates the total cost of Tom's deck of cards. -/
def deck_cost : ℝ :=
  let rare_count : ℕ := 19
  let uncommon_count : ℕ := 11
  let common_count : ℕ := 30
  let rare_cost : ℝ := 1
  let uncommon_cost : ℝ := 0.5
  let common_cost : ℝ := 0.25
  rare_count * rare_cost + uncommon_count * uncommon_cost + common_count * common_cost

/-- Proves that the total cost of Tom's deck is $32. -/
theorem deck_cost_is_32 : deck_cost = 32 := by
  sorry

end deck_cost_is_32_l2053_205306


namespace rightmost_three_digits_of_5_to_1994_l2053_205387

theorem rightmost_three_digits_of_5_to_1994 : 5^1994 % 1000 = 625 := by
  sorry

end rightmost_three_digits_of_5_to_1994_l2053_205387


namespace tan_30_degrees_l2053_205308

theorem tan_30_degrees : Real.tan (30 * π / 180) = Real.sqrt 3 / 3 := by
  sorry

end tan_30_degrees_l2053_205308


namespace mandatory_work_effect_l2053_205374

/-- Represents the labor market for doctors -/
structure DoctorLaborMarket where
  state_supply : ℝ → ℝ  -- Supply function for state sector
  state_demand : ℝ → ℝ  -- Demand function for state sector
  private_supply : ℝ → ℝ  -- Supply function for private sector
  private_demand : ℝ → ℝ  -- Demand function for private sector

/-- Represents the policy of mandatory work in public healthcare -/
structure MandatoryWorkPolicy where
  years_required : ℕ  -- Number of years required in public healthcare

/-- The equilibrium wage in the state sector -/
def state_equilibrium_wage (market : DoctorLaborMarket) : ℝ :=
  sorry

/-- The equilibrium price in the private healthcare sector -/
def private_equilibrium_price (market : DoctorLaborMarket) : ℝ :=
  sorry

/-- The effect of the mandatory work policy on the labor market -/
def apply_policy (market : DoctorLaborMarket) (policy : MandatoryWorkPolicy) : DoctorLaborMarket :=
  sorry

theorem mandatory_work_effect (initial_market : DoctorLaborMarket) (policy : MandatoryWorkPolicy) :
  let final_market := apply_policy initial_market policy
  state_equilibrium_wage final_market > state_equilibrium_wage initial_market ∧
  private_equilibrium_price final_market < private_equilibrium_price initial_market :=
sorry

end mandatory_work_effect_l2053_205374


namespace bob_has_81_robots_l2053_205312

/-- The number of car robots Tom and Michael have together -/
def tom_and_michael_robots : ℕ := 9

/-- The factor by which Bob has more car robots than Tom and Michael -/
def bob_factor : ℕ := 9

/-- The total number of car robots Bob has -/
def bob_robots : ℕ := tom_and_michael_robots * bob_factor

/-- Theorem stating that Bob has 81 car robots -/
theorem bob_has_81_robots : bob_robots = 81 := by
  sorry

end bob_has_81_robots_l2053_205312


namespace fran_speed_to_match_joann_l2053_205382

/-- Proves that Fran needs to ride at 30 mph to cover the same distance as Joann -/
theorem fran_speed_to_match_joann (joann_speed : ℝ) (joann_time : ℝ) (fran_time : ℝ) :
  joann_speed = 15 →
  joann_time = 4 →
  fran_time = 2 →
  (fran_time * (joann_speed * joann_time / fran_time) = joann_speed * joann_time) ∧
  (joann_speed * joann_time / fran_time = 30) := by
  sorry

end fran_speed_to_match_joann_l2053_205382


namespace binary_addition_correct_l2053_205354

/-- Represents a binary number as a list of bits (least significant bit first) -/
def BinaryNumber := List Bool

/-- Converts a binary number to its decimal representation -/
def binary_to_decimal (b : BinaryNumber) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

/-- The four binary numbers given in the problem -/
def num1 : BinaryNumber := [true, false, true, true]
def num2 : BinaryNumber := [false, true, true]
def num3 : BinaryNumber := [true, true, false, true]
def num4 : BinaryNumber := [false, false, true, true, true]

/-- The expected sum in binary -/
def expected_sum : BinaryNumber := [true, false, true, false, false, true]

theorem binary_addition_correct :
  binary_to_decimal num1 + binary_to_decimal num2 + 
  binary_to_decimal num3 + binary_to_decimal num4 = 
  binary_to_decimal expected_sum := by
  sorry

end binary_addition_correct_l2053_205354


namespace total_strings_is_72_l2053_205342

/-- Calculates the total number of strings John needs to restring his instruments. -/
def total_strings : ℕ :=
  let num_basses : ℕ := 3
  let strings_per_bass : ℕ := 4
  let num_guitars : ℕ := 2 * num_basses
  let strings_per_guitar : ℕ := 6
  let num_eight_string_guitars : ℕ := num_guitars - 3
  let strings_per_eight_string_guitar : ℕ := 8
  
  (num_basses * strings_per_bass) +
  (num_guitars * strings_per_guitar) +
  (num_eight_string_guitars * strings_per_eight_string_guitar)

theorem total_strings_is_72 : total_strings = 72 := by
  sorry

end total_strings_is_72_l2053_205342


namespace arithmetic_reciprocal_sequence_l2053_205393

theorem arithmetic_reciprocal_sequence (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hdistinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) (harith : ∃ d ≠ 0, b = a + d ∧ c = a + 2*d) :
  ¬(∃ r ≠ 0, (1/b - 1/a) = r ∧ (1/c - 1/b) = r) ∧
  ¬(∃ q ≠ 1, (1/b) / (1/a) = q ∧ (1/c) / (1/b) = q) :=
by sorry

end arithmetic_reciprocal_sequence_l2053_205393


namespace max_regions_40_parabolas_l2053_205333

/-- Represents a parabola of the form y = ax^2 + bx + c -/
structure VerticalParabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a parabola of the form x = ay^2 + by + c -/
structure HorizontalParabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculates the maximum number of regions created by a set of vertical and horizontal parabolas -/
def max_regions (vertical_parabolas : Finset VerticalParabola) (horizontal_parabolas : Finset HorizontalParabola) : ℕ :=
  sorry

/-- Theorem stating the maximum number of regions created by 20 vertical and 20 horizontal parabolas -/
theorem max_regions_40_parabolas :
  ∀ (v : Finset VerticalParabola) (h : Finset HorizontalParabola),
  v.card = 20 → h.card = 20 →
  max_regions v h = 2422 :=
by sorry

end max_regions_40_parabolas_l2053_205333


namespace value_of_y_l2053_205363

theorem value_of_y (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (eq1 : x = 2 + 1/y) (eq2 : y = 1 + 1/x) : y = (1 + Real.sqrt 3) / 2 := by
  sorry

end value_of_y_l2053_205363


namespace regular_polygon_perimeter_l2053_205335

/-- A regular polygon with exterior angle 20° and side length 10 has perimeter 180 -/
theorem regular_polygon_perimeter (n : ℕ) (exterior_angle : ℝ) (side_length : ℝ) : 
  n > 2 →
  exterior_angle = 20 →
  side_length = 10 →
  n * exterior_angle = 360 →
  n * side_length = 180 := by
sorry

end regular_polygon_perimeter_l2053_205335


namespace sarah_hair_product_usage_l2053_205337

/-- Calculates the total volume of hair care products used over a given number of days -/
def total_hair_product_usage (shampoo_daily : ℝ) (conditioner_ratio : ℝ) (days : ℕ) : ℝ :=
  let conditioner_daily := shampoo_daily * conditioner_ratio
  let total_daily := shampoo_daily + conditioner_daily
  total_daily * days

/-- Proves that Sarah's total hair product usage over two weeks is 21 ounces -/
theorem sarah_hair_product_usage : 
  total_hair_product_usage 1 0.5 14 = 21 := by
sorry

#eval total_hair_product_usage 1 0.5 14

end sarah_hair_product_usage_l2053_205337


namespace functional_equation_solution_l2053_205359

theorem functional_equation_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, (f (x + y))^2 = (f x)^2 + (f y)^2) : 
  ∀ x : ℝ, f x = 0 := by
  sorry

end functional_equation_solution_l2053_205359


namespace sphere_identical_views_l2053_205380

/-- A geometric body in 3D space -/
inductive GeometricBody
  | Sphere
  | Cylinder
  | TriangularPrism
  | Cone

/-- Represents a 2D view of a geometric body -/
structure View where
  shape : Type
  size : ℝ

/-- Returns true if all views are identical -/
def identicalViews (front side top : View) : Prop :=
  front = side ∧ side = top

/-- Returns the front, side, and top views of a geometric body -/
def getViews (body : GeometricBody) : (View × View × View) :=
  sorry

theorem sphere_identical_views :
  ∀ (body : GeometricBody),
    (∃ (front side top : View), 
      getViews body = (front, side, top) ∧ 
      identicalViews front side top) 
    ↔ 
    body = GeometricBody.Sphere :=
  sorry

end sphere_identical_views_l2053_205380


namespace rope_length_proof_l2053_205366

/-- The length of a rope after being folded in half twice -/
def folded_length : ℝ := 10

/-- The number of times the rope is folded in half -/
def fold_count : ℕ := 2

/-- Calculates the original length of the rope before folding -/
def original_length : ℝ := folded_length * (2 ^ fold_count)

/-- Proves that the original length of the rope is 40 centimeters -/
theorem rope_length_proof : original_length = 40 := by
  sorry

end rope_length_proof_l2053_205366


namespace binomial_10_choose_5_l2053_205330

theorem binomial_10_choose_5 : Nat.choose 10 5 = 252 := by
  sorry

end binomial_10_choose_5_l2053_205330


namespace complex_fraction_equality_l2053_205317

theorem complex_fraction_equality (a b : ℝ) : 
  (1 + I : ℂ) / (1 - I) = a + b * I → b = 1 := by
  sorry

end complex_fraction_equality_l2053_205317


namespace least_x_for_even_prime_quotient_l2053_205328

theorem least_x_for_even_prime_quotient :
  ∃ (x p q : ℕ),
    x > 0 ∧
    Prime p ∧
    Prime q ∧
    p ≠ q ∧
    q - p = 3 ∧
    x / (11 * p * q) = 2 ∧
    (∀ y, y > 0 → y / (11 * p * q) = 2 → y ≥ x) ∧
    x = 770 :=
by sorry

end least_x_for_even_prime_quotient_l2053_205328


namespace decimal_addition_l2053_205321

theorem decimal_addition : (7.15 : ℝ) + 2.639 = 9.789 := by
  sorry

end decimal_addition_l2053_205321


namespace function_graph_overlap_l2053_205367

theorem function_graph_overlap (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x : ℝ, a^(x/2) = 2^(-x/2)) → a = 1/2 := by
  sorry

end function_graph_overlap_l2053_205367


namespace binary_11011_equals_27_l2053_205356

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_11011_equals_27 :
  binary_to_decimal [true, true, false, true, true] = 27 := by
  sorry

end binary_11011_equals_27_l2053_205356


namespace lauren_earnings_l2053_205331

/-- Represents the earnings for a single day --/
structure DayEarnings where
  commercial_rate : ℝ
  subscription_rate : ℝ
  commercial_views : ℕ
  subscriptions : ℕ

/-- Calculates the total earnings for a single day --/
def day_total (d : DayEarnings) : ℝ :=
  d.commercial_rate * d.commercial_views + d.subscription_rate * d.subscriptions

/-- Represents the earnings for the weekend --/
structure WeekendEarnings where
  merchandise_sales : ℝ
  merchandise_rate : ℝ

/-- Calculates the total earnings for the weekend --/
def weekend_total (w : WeekendEarnings) : ℝ :=
  w.merchandise_sales * w.merchandise_rate

/-- Represents Lauren's earnings for the entire period --/
structure PeriodEarnings where
  monday : DayEarnings
  tuesday : DayEarnings
  weekend : WeekendEarnings

/-- Calculates the total earnings for the entire period --/
def period_total (p : PeriodEarnings) : ℝ :=
  day_total p.monday + day_total p.tuesday + weekend_total p.weekend

/-- Theorem stating that Lauren's total earnings for the period equal $140.00 --/
theorem lauren_earnings :
  let p : PeriodEarnings := {
    monday := {
      commercial_rate := 0.40,
      subscription_rate := 0.80,
      commercial_views := 80,
      subscriptions := 20
    },
    tuesday := {
      commercial_rate := 0.50,
      subscription_rate := 1.00,
      commercial_views := 100,
      subscriptions := 27
    },
    weekend := {
      merchandise_sales := 150,
      merchandise_rate := 0.10
    }
  }
  period_total p = 140
:= by sorry

end lauren_earnings_l2053_205331


namespace river_lengths_theorem_l2053_205339

/-- The lengths of the Danube, Dnieper, and Don rivers satisfy the given conditions -/
theorem river_lengths_theorem (danube dnieper don : ℝ) : 
  (dnieper / danube = 5 / (19 / 3)) →
  (don / danube = 6.5 / 9.5) →
  (dnieper - don = 300) →
  (danube = 2850 ∧ dnieper = 2250 ∧ don = 1950) :=
by sorry

end river_lengths_theorem_l2053_205339


namespace intersection_A_B_when_m_3_range_of_m_when_A_subset_B_l2053_205327

-- Define set A
def A : Set ℝ := {x | ∃ y, y = Real.sqrt (3 - 2*x - x^2)}

-- Define set B
def B (m : ℝ) : Set ℝ := {x | x^2 - 2*x + 1 - m^2 ≤ 0}

-- Part 1
theorem intersection_A_B_when_m_3 : 
  A ∩ B 3 = {x | -2 ≤ x ∧ x ≤ 1} := by sorry

-- Part 2
theorem range_of_m_when_A_subset_B : 
  ∀ m > 0, A ⊆ B m → m ≥ 4 := by sorry

end intersection_A_B_when_m_3_range_of_m_when_A_subset_B_l2053_205327


namespace unique_stamp_denomination_l2053_205302

/-- Given stamps of denominations 6, n, and n+2 cents, 
    this function returns the greatest postage that cannot be formed. -/
def greatest_unattainable_postage (n : ℕ) : ℕ :=
  6 * n * (n + 2) - (6 + n + (n + 2))

/-- This theorem states that there exists a unique positive integer n 
    such that the greatest unattainable postage is 120 cents, 
    and this n is equal to 8. -/
theorem unique_stamp_denomination :
  ∃! n : ℕ, n > 0 ∧ greatest_unattainable_postage n = 120 ∧ n = 8 :=
sorry

end unique_stamp_denomination_l2053_205302


namespace sum_of_cubes_and_fourth_powers_l2053_205311

theorem sum_of_cubes_and_fourth_powers (a b : ℝ) 
  (sum_eq : a + b = 2) 
  (sum_squares_eq : a^2 + b^2 = 2) : 
  a^3 + b^3 = 2 ∧ a^4 + b^4 = 2 := by
  sorry

end sum_of_cubes_and_fourth_powers_l2053_205311


namespace total_hunt_is_21_l2053_205314

/-- The number of animals hunted by Sam in a day -/
def sam_hunt : ℕ := 6

/-- The number of animals hunted by Rob in a day -/
def rob_hunt : ℕ := sam_hunt / 2

/-- The number of animals hunted by Mark in a day -/
def mark_hunt : ℕ := (sam_hunt + rob_hunt) / 3

/-- The number of animals hunted by Peter in a day -/
def peter_hunt : ℕ := 3 * mark_hunt

/-- The total number of animals hunted by all four in a day -/
def total_hunt : ℕ := sam_hunt + rob_hunt + mark_hunt + peter_hunt

theorem total_hunt_is_21 : total_hunt = 21 := by
  sorry

end total_hunt_is_21_l2053_205314
