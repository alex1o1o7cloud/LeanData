import Mathlib

namespace NUMINAMATH_CALUDE_f_one_eq_zero_f_x_minus_one_lt_zero_iff_f_inequality_iff_l3131_313170

/-- An increasing function f satisfying f(x/y) = f(x) - f(y) -/
class SpecialFunction (f : ℝ → ℝ) : Prop where
  domain : ∀ x, x > 0 → f x ≠ 0
  increasing : ∀ x y, x < y → f x < f y
  special_prop : ∀ x y, x > 0 → y > 0 → f (x / y) = f x - f y

variable (f : ℝ → ℝ) [SpecialFunction f]

/-- f(1) = 0 -/
theorem f_one_eq_zero : f 1 = 0 := by sorry

/-- f(x-1) < 0 iff x ∈ (1, 2) -/
theorem f_x_minus_one_lt_zero_iff (x : ℝ) : f (x - 1) < 0 ↔ 1 < x ∧ x < 2 := by sorry

/-- If f(2) = 1, then f(x+3) - f(1/x) < 2 iff x ∈ (0, 1) -/
theorem f_inequality_iff (h : f 2 = 1) (x : ℝ) : f (x + 3) - f (1 / x) < 2 ↔ 0 < x ∧ x < 1 := by sorry

end NUMINAMATH_CALUDE_f_one_eq_zero_f_x_minus_one_lt_zero_iff_f_inequality_iff_l3131_313170


namespace NUMINAMATH_CALUDE_stack_probability_theorem_l3131_313176

/-- Represents the dimensions of a crate -/
structure CrateDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents the configuration of crates in the stack -/
structure StackConfiguration where
  count_3ft : ℕ
  count_4ft : ℕ
  count_6ft : ℕ

def crate_dimensions : CrateDimensions :=
  { length := 3, width := 4, height := 6 }

def total_crates : ℕ := 12

def target_height : ℕ := 50

def valid_configuration (config : StackConfiguration) : Prop :=
  config.count_3ft + config.count_4ft + config.count_6ft = total_crates ∧
  3 * config.count_3ft + 4 * config.count_4ft + 6 * config.count_6ft = target_height

def count_valid_configurations : ℕ := 30690

def total_possible_configurations : ℕ := 3^total_crates

theorem stack_probability_theorem :
  (count_valid_configurations : ℚ) / total_possible_configurations = 10230 / 531441 :=
sorry

end NUMINAMATH_CALUDE_stack_probability_theorem_l3131_313176


namespace NUMINAMATH_CALUDE_halfway_fraction_l3131_313142

theorem halfway_fraction (a b c : ℚ) (ha : a = 1/4) (hb : b = 1/6) (hc : c = 1/3) :
  (a + b + c) / 3 = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_halfway_fraction_l3131_313142


namespace NUMINAMATH_CALUDE_power_not_all_ones_l3131_313193

theorem power_not_all_ones (a n : ℕ) : a > 1 → n > 1 → ¬∃ s : ℕ, a^n = 2^s - 1 := by
  sorry

end NUMINAMATH_CALUDE_power_not_all_ones_l3131_313193


namespace NUMINAMATH_CALUDE_patricia_barrels_l3131_313156

/-- Given a scenario where Patricia has some barrels, proves that the number of barrels is 4 -/
theorem patricia_barrels : 
  ∀ (barrel_capacity : ℝ) (flow_rate : ℝ) (fill_time : ℝ) (num_barrels : ℕ),
  barrel_capacity = 7 →
  flow_rate = 3.5 →
  fill_time = 8 →
  (flow_rate * fill_time : ℝ) = (↑num_barrels * barrel_capacity) →
  num_barrels = 4 := by
sorry

end NUMINAMATH_CALUDE_patricia_barrels_l3131_313156


namespace NUMINAMATH_CALUDE_number_equation_solution_l3131_313132

theorem number_equation_solution : 
  ∃ x : ℚ, x^2 + 145 = (x - 19)^2 ∧ x = 108/19 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_solution_l3131_313132


namespace NUMINAMATH_CALUDE_double_age_proof_l3131_313166

/-- The number of years in the future when Richard will be twice as old as Scott -/
def years_until_double_age : ℕ := 8

theorem double_age_proof (david_current_age richard_current_age scott_current_age : ℕ) 
  (h1 : david_current_age = 14)
  (h2 : richard_current_age = david_current_age + 6)
  (h3 : david_current_age = scott_current_age + 8) :
  richard_current_age + years_until_double_age = 2 * (scott_current_age + years_until_double_age) := by
  sorry

#check double_age_proof

end NUMINAMATH_CALUDE_double_age_proof_l3131_313166


namespace NUMINAMATH_CALUDE_simple_interest_problem_l3131_313118

/-- Proves that for a principal of 800 at simple interest, if increasing the
    interest rate by 5% results in 400 more interest, then the time period is 10 years. -/
theorem simple_interest_problem (r : ℝ) (t : ℝ) :
  (800 * r * t / 100) + 400 = 800 * (r + 5) * t / 100 →
  t = 10 :=
by sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l3131_313118


namespace NUMINAMATH_CALUDE_germination_probability_l3131_313114

/-- The germination rate of the seeds -/
def germination_rate : ℝ := 0.7

/-- The number of seeds -/
def total_seeds : ℕ := 3

/-- The number of seeds we want to germinate -/
def target_germination : ℕ := 2

/-- The probability of exactly 2 out of 3 seeds germinating -/
def probability_2_out_of_3 : ℝ := 
  (Nat.choose total_seeds target_germination : ℝ) * 
  germination_rate ^ target_germination * 
  (1 - germination_rate) ^ (total_seeds - target_germination)

theorem germination_probability : 
  probability_2_out_of_3 = 0.441 := by sorry

end NUMINAMATH_CALUDE_germination_probability_l3131_313114


namespace NUMINAMATH_CALUDE_marble_count_l3131_313133

theorem marble_count (total : ℕ) (yellow : ℕ) (blue_ratio : ℕ) (red_ratio : ℕ) 
  (h1 : total = 19)
  (h2 : yellow = 5)
  (h3 : blue_ratio = 3)
  (h4 : red_ratio = 4) :
  let remaining := total - yellow
  let share := remaining / (blue_ratio + red_ratio)
  let red := red_ratio * share
  red - yellow = 3 := by sorry

end NUMINAMATH_CALUDE_marble_count_l3131_313133


namespace NUMINAMATH_CALUDE_square_binomial_constant_l3131_313140

theorem square_binomial_constant (c : ℝ) : 
  (∃ a b : ℝ, ∀ x : ℝ, x^2 + 50*x + c = (x + a)^2 + b) → c = 625 := by
  sorry

end NUMINAMATH_CALUDE_square_binomial_constant_l3131_313140


namespace NUMINAMATH_CALUDE_same_remainder_mod_ten_l3131_313105

theorem same_remainder_mod_ten (a b c : ℕ) 
  (h : ∃ r : ℕ, (2*a + b) % 10 = r ∧ (2*b + c) % 10 = r ∧ (2*c + a) % 10 = r) :
  ∃ s : ℕ, a % 10 = s ∧ b % 10 = s ∧ c % 10 = s := by
  sorry

end NUMINAMATH_CALUDE_same_remainder_mod_ten_l3131_313105


namespace NUMINAMATH_CALUDE_binomial_10_3_l3131_313198

theorem binomial_10_3 : Nat.choose 10 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_binomial_10_3_l3131_313198


namespace NUMINAMATH_CALUDE_crazy_silly_school_books_l3131_313158

theorem crazy_silly_school_books (num_movies : ℕ) (movie_book_diff : ℕ) : 
  num_movies = 17 → movie_book_diff = 6 → num_movies - movie_book_diff = 11 := by
  sorry

end NUMINAMATH_CALUDE_crazy_silly_school_books_l3131_313158


namespace NUMINAMATH_CALUDE_inequality_proof_l3131_313130

theorem inequality_proof (u v x y a b c d : ℝ) 
  (hu : u > 0) (hv : v > 0) (hx : x > 0) (hy : y > 0)
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (u / x + v / y ≥ 4 * (u * y + v * x) / ((x + y) ^ 2)) ∧
  (a / (b + 2 * c + d) + b / (c + 2 * d + a) + c / (d + 2 * a + b) + d / (a + 2 * b + c) ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3131_313130


namespace NUMINAMATH_CALUDE_sum_of_solutions_is_zero_l3131_313185

theorem sum_of_solutions_is_zero : 
  ∃ (x₁ x₂ : ℝ), (8 * x₁) / 40 = 7 / x₁ ∧ 
                 (8 * x₂) / 40 = 7 / x₂ ∧ 
                 x₁ + x₂ = 0 ∧
                 ∀ (y : ℝ), (8 * y) / 40 = 7 / y → y = x₁ ∨ y = x₂ := by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_is_zero_l3131_313185


namespace NUMINAMATH_CALUDE_floor_sum_abcd_l3131_313157

theorem floor_sum_abcd (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h1 : a^2 + b^2 = 2010) (h2 : c^2 + d^2 = 2010) (h3 : a * c = 1020) (h4 : b * d = 1020) :
  ⌊a + b + c + d⌋ = 126 := by
  sorry

end NUMINAMATH_CALUDE_floor_sum_abcd_l3131_313157


namespace NUMINAMATH_CALUDE_smallest_n_perfect_powers_l3131_313144

theorem smallest_n_perfect_powers : ∃ (n : ℕ), 
  (n = 151875) ∧ 
  (∀ m : ℕ, m > 0 → m < n → 
    (∃ k : ℕ, 3 * m = k^2) → 
    (∃ l : ℕ, 5 * m = l^5) → False) ∧
  (∃ k : ℕ, 3 * n = k^2) ∧
  (∃ l : ℕ, 5 * n = l^5) := by
sorry

end NUMINAMATH_CALUDE_smallest_n_perfect_powers_l3131_313144


namespace NUMINAMATH_CALUDE_range_of_m_for_false_proposition_l3131_313171

theorem range_of_m_for_false_proposition :
  (∀ x : ℝ, x^2 - m*x - m > 0) → m ∈ Set.Ioo (-4 : ℝ) 0 :=
by
  sorry

end NUMINAMATH_CALUDE_range_of_m_for_false_proposition_l3131_313171


namespace NUMINAMATH_CALUDE_linear_function_property_l3131_313122

/-- Given a linear function f(x) = ax + b, if f(1) = 2 and f'(1) = 2, then f(2) = 4 -/
theorem linear_function_property (a b : ℝ) (f : ℝ → ℝ) 
    (h1 : ∀ x, f x = a * x + b)
    (h2 : f 1 = 2)
    (h3 : (deriv f) 1 = 2) : 
  f 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_property_l3131_313122


namespace NUMINAMATH_CALUDE_proposition_q_false_l3131_313106

theorem proposition_q_false (p q : Prop) 
  (h1 : ¬p) 
  (h2 : ¬((¬p) ∧ q)) : 
  ¬q := by
  sorry

end NUMINAMATH_CALUDE_proposition_q_false_l3131_313106


namespace NUMINAMATH_CALUDE_stating_distribution_schemes_count_l3131_313135

/-- Represents the number of schools --/
def num_schools : ℕ := 5

/-- Represents the number of computers --/
def num_computers : ℕ := 6

/-- Represents the number of schools that must receive at least 2 computers --/
def num_special_schools : ℕ := 2

/-- Represents the minimum number of computers each special school must receive --/
def min_computers_per_special_school : ℕ := 2

/-- 
Calculates the number of ways to distribute computers to schools 
under the given constraints
--/
def distribution_schemes : ℕ := sorry

/-- 
Theorem stating that the number of distribution schemes is 15
--/
theorem distribution_schemes_count : distribution_schemes = 15 := by sorry

end NUMINAMATH_CALUDE_stating_distribution_schemes_count_l3131_313135


namespace NUMINAMATH_CALUDE_smallest_scalene_triangle_with_prime_perimeter_l3131_313169

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop := sorry

/-- A function that checks if three numbers form a valid triangle -/
def isValidTriangle (a b c : ℕ) : Prop := a + b > c ∧ a + c > b ∧ b + c > a

/-- A function that checks if three numbers are consecutive odd primes -/
def areConsecutiveOddPrimes (a b c : ℕ) : Prop :=
  isPrime a ∧ isPrime b ∧ isPrime c ∧
  a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 1 ∧
  b = a + 2 ∧ c = b + 2

theorem smallest_scalene_triangle_with_prime_perimeter :
  ∃ (a b c : ℕ),
    areConsecutiveOddPrimes a b c ∧
    isValidTriangle a b c ∧
    isPrime (a + b + c) ∧
    a + b + c = 23 ∧
    (∀ (x y z : ℕ),
      areConsecutiveOddPrimes x y z →
      isValidTriangle x y z →
      isPrime (x + y + z) →
      x + y + z ≥ 23) :=
sorry

end NUMINAMATH_CALUDE_smallest_scalene_triangle_with_prime_perimeter_l3131_313169


namespace NUMINAMATH_CALUDE_area_of_triangle_AEB_l3131_313123

-- Define the points
variable (A B C D E F G : Euclidean_plane)

-- Define the rectangle ABCD
def is_rectangle (A B C D : Euclidean_plane) : Prop := sorry

-- Define the lengths
def length (P Q : Euclidean_plane) : ℝ := sorry

-- Define a point being on a line segment
def on_segment (P Q R : Euclidean_plane) : Prop := sorry

-- Define line intersection
def intersect (P Q R S : Euclidean_plane) : Euclidean_plane := sorry

-- Define triangle area
def triangle_area (P Q R : Euclidean_plane) : ℝ := sorry

theorem area_of_triangle_AEB 
  (h_rect : is_rectangle A B C D)
  (h_AB : length A B = 10)
  (h_BC : length B C = 4)
  (h_F_on_CD : on_segment C D F)
  (h_G_on_CD : on_segment C D G)
  (h_DF : length D F = 2)
  (h_GC : length G C = 3)
  (h_E : E = intersect A F B G) :
  triangle_area A E B = 40 := by sorry

end NUMINAMATH_CALUDE_area_of_triangle_AEB_l3131_313123


namespace NUMINAMATH_CALUDE_largest_multiple_of_9_below_75_l3131_313155

theorem largest_multiple_of_9_below_75 : ∃ n : ℕ, n * 9 = 72 ∧ 72 < 75 ∧ ∀ m : ℕ, m * 9 < 75 → m * 9 ≤ 72 :=
by sorry

end NUMINAMATH_CALUDE_largest_multiple_of_9_below_75_l3131_313155


namespace NUMINAMATH_CALUDE_solution_and_minimum_value_l3131_313101

-- Define the function f
def f (x m : ℝ) : ℝ := |x - m|

-- State the theorem
theorem solution_and_minimum_value :
  (∀ x : ℝ, f x 2 ≤ 3 ↔ x ∈ Set.Icc (-1) 5) ∧
  (∀ a b c : ℝ, a - 2*b + c = 2 → a^2 + b^2 + c^2 ≥ 2/3) ∧
  (∃ a b c : ℝ, a - 2*b + c = 2 ∧ a^2 + b^2 + c^2 = 2/3) :=
by sorry

end NUMINAMATH_CALUDE_solution_and_minimum_value_l3131_313101


namespace NUMINAMATH_CALUDE_piggy_bank_pennies_l3131_313184

theorem piggy_bank_pennies (num_compartments : ℕ) (initial_pennies : ℕ) (added_pennies : ℕ) : 
  num_compartments = 12 → 
  initial_pennies = 2 → 
  added_pennies = 6 → 
  (num_compartments * (initial_pennies + added_pennies)) = 96 := by
sorry

end NUMINAMATH_CALUDE_piggy_bank_pennies_l3131_313184


namespace NUMINAMATH_CALUDE_calculate_income_l3131_313121

/-- Given a person's income and expenditure ratio, and their savings, calculate their income. -/
theorem calculate_income (income expenditure savings : ℕ) : 
  income * 4 = expenditure * 5 →  -- income to expenditure ratio is 5:4
  income - expenditure = savings → -- savings definition
  savings = 4000 → -- given savings amount
  income = 20000 := by
  sorry

end NUMINAMATH_CALUDE_calculate_income_l3131_313121


namespace NUMINAMATH_CALUDE_lucy_flour_problem_l3131_313141

/-- The amount of flour Lucy had at the start of the week -/
def initial_flour : ℝ := 500

/-- The amount of flour Lucy used for baking cookies -/
def used_flour : ℝ := 240

/-- The amount of flour Lucy needs to buy to have a full bag -/
def flour_to_buy : ℝ := 370

theorem lucy_flour_problem :
  (initial_flour - used_flour) / 2 + flour_to_buy = initial_flour :=
by sorry

end NUMINAMATH_CALUDE_lucy_flour_problem_l3131_313141


namespace NUMINAMATH_CALUDE_Q_space_diagonals_l3131_313111

-- Define the structure of our polyhedron
structure Polyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ
  triangular_faces : ℕ
  quadrilateral_faces : ℕ

-- Define our specific polyhedron Q
def Q : Polyhedron := {
  vertices := 30,
  edges := 72,
  faces := 38,
  triangular_faces := 20,
  quadrilateral_faces := 18
}

-- Function to calculate the number of space diagonals
def space_diagonals (p : Polyhedron) : ℕ :=
  let total_pairs := p.vertices.choose 2
  let face_diagonals := 2 * p.quadrilateral_faces
  total_pairs - p.edges - face_diagonals

-- Theorem statement
theorem Q_space_diagonals : space_diagonals Q = 327 := by
  sorry


end NUMINAMATH_CALUDE_Q_space_diagonals_l3131_313111


namespace NUMINAMATH_CALUDE_equation_system_solution_l3131_313195

theorem equation_system_solution :
  ∀ (x y z : ℤ),
    (4 : ℝ) ^ (x^2 + 2*x*y + 1) = (z + 2 : ℝ) * 7^(|y| - 1) →
    Real.sin ((3 * Real.pi * ↑z) / 2) = 1 →
    ((x = 1 ∧ y = -1 ∧ z = -1) ∨ (x = -1 ∧ y = 1 ∧ z = -1)) :=
by sorry

end NUMINAMATH_CALUDE_equation_system_solution_l3131_313195


namespace NUMINAMATH_CALUDE_circle_C_equation_l3131_313107

/-- A circle C in the xy-plane -/
structure CircleC where
  /-- x-coordinate of a point on the circle -/
  x : ℝ → ℝ
  /-- y-coordinate of a point on the circle -/
  y : ℝ → ℝ
  /-- The parameter θ ranges over all real numbers -/
  θ : ℝ
  /-- x-coordinate is defined as 2 + 2cos(θ) -/
  x_eq : x θ = 2 + 2 * Real.cos θ
  /-- y-coordinate is defined as 2sin(θ) -/
  y_eq : y θ = 2 * Real.sin θ

/-- The standard equation of circle C -/
def standard_equation (c : CircleC) (x y : ℝ) : Prop :=
  (x - 2)^2 + y^2 = 4

/-- Theorem stating that the parametric equations of CircleC satisfy its standard equation -/
theorem circle_C_equation (c : CircleC) :
  ∀ θ, standard_equation c (c.x θ) (c.y θ) := by
  sorry

end NUMINAMATH_CALUDE_circle_C_equation_l3131_313107


namespace NUMINAMATH_CALUDE_number_ratio_problem_l3131_313129

theorem number_ratio_problem (x y z : ℝ) : 
  x = 18 →  -- The smallest number is 18
  y = 4 * x →  -- The second number is 4 times the first
  ∃ k : ℝ, z = k * y →  -- The third number is some multiple of the second
  (x + y + z) / 3 = 78 →  -- Their average is 78
  z / y = 2 :=  -- The ratio of the third to the second is 2
by sorry

end NUMINAMATH_CALUDE_number_ratio_problem_l3131_313129


namespace NUMINAMATH_CALUDE_exists_cost_price_l3131_313168

/-- The cost price of a watch satisfying the given conditions --/
def cost_price : ℝ → Prop := fun C =>
  3 * (0.925 * C + 265) = 3 * C * 1.053

/-- Theorem stating the existence of a cost price satisfying the conditions --/
theorem exists_cost_price : ∃ C : ℝ, cost_price C := by
  sorry

end NUMINAMATH_CALUDE_exists_cost_price_l3131_313168


namespace NUMINAMATH_CALUDE_factorization_of_2a_5_minus_8a_l3131_313159

theorem factorization_of_2a_5_minus_8a (a : ℝ) : 
  2 * a^5 - 8 * a = 2 * a * (a^2 + 2) * (a + Real.sqrt 2) * (a - Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_2a_5_minus_8a_l3131_313159


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3131_313150

theorem polynomial_factorization (x y : ℝ) : 
  (x - 2*y) * (x - 2*y + 1) = x^2 - 4*x*y - 2*y + x + 4*y^2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3131_313150


namespace NUMINAMATH_CALUDE_tangent_and_normal_equations_l3131_313120

/-- The curve equation -/
def curve (x y : ℝ) : Prop := x^2 - 2*x*y + 3*y^2 - 2*y - 16 = 0

/-- The point on the curve -/
def point : ℝ × ℝ := (1, 3)

/-- Tangent line equation -/
def tangent_line (x y : ℝ) : Prop := 2*x - 7*y + 19 = 0

/-- Normal line equation -/
def normal_line (x y : ℝ) : Prop := 7*x + 2*y - 13 = 0

theorem tangent_and_normal_equations :
  curve point.1 point.2 →
  (∀ x y, tangent_line x y ↔ 
    (y - point.2) = (2/7) * (x - point.1)) ∧
  (∀ x y, normal_line x y ↔ 
    (y - point.2) = (-7/2) * (x - point.1)) := by
  sorry

end NUMINAMATH_CALUDE_tangent_and_normal_equations_l3131_313120


namespace NUMINAMATH_CALUDE_B_power_15_minus_3_power_14_l3131_313137

def B : Matrix (Fin 2) (Fin 2) ℝ := !![3, 4; 0, 2]

theorem B_power_15_minus_3_power_14 :
  B^15 - 3 • B^14 = !![0, 4; 0, -1] := by sorry

end NUMINAMATH_CALUDE_B_power_15_minus_3_power_14_l3131_313137


namespace NUMINAMATH_CALUDE_bottles_left_l3131_313194

theorem bottles_left (initial : Float) (maria_drank : Float) (sister_drank : Float) :
  initial = 45.0 →
  maria_drank = 14.0 →
  sister_drank = 8.0 →
  initial - maria_drank - sister_drank = 23.0 :=
by sorry

end NUMINAMATH_CALUDE_bottles_left_l3131_313194


namespace NUMINAMATH_CALUDE_polygon_with_20_diagonals_has_8_sides_l3131_313117

/-- The number of diagonals in a polygon -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A polygon with 20 diagonals has 8 sides -/
theorem polygon_with_20_diagonals_has_8_sides :
  ∃ (n : ℕ), n > 0 ∧ num_diagonals n = 20 → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_polygon_with_20_diagonals_has_8_sides_l3131_313117


namespace NUMINAMATH_CALUDE_gcf_lcm_product_8_12_l3131_313109

theorem gcf_lcm_product_8_12 : Nat.gcd 8 12 * Nat.lcm 8 12 = 96 := by
  sorry

end NUMINAMATH_CALUDE_gcf_lcm_product_8_12_l3131_313109


namespace NUMINAMATH_CALUDE_trick_deck_total_spent_l3131_313173

/-- The total amount spent by Tom and his friend on trick decks -/
theorem trick_deck_total_spent 
  (price_per_deck : ℕ)
  (tom_decks : ℕ)
  (friend_decks : ℕ)
  (h1 : price_per_deck = 8)
  (h2 : tom_decks = 3)
  (h3 : friend_decks = 5) :
  price_per_deck * (tom_decks + friend_decks) = 64 :=
by sorry

end NUMINAMATH_CALUDE_trick_deck_total_spent_l3131_313173


namespace NUMINAMATH_CALUDE_total_earning_calculation_l3131_313160

theorem total_earning_calculation (days_a days_b days_c : ℕ) 
  (wage_ratio_a wage_ratio_b wage_ratio_c : ℕ) (wage_c : ℕ) :
  days_a = 6 →
  days_b = 9 →
  days_c = 4 →
  wage_ratio_a = 3 →
  wage_ratio_b = 4 →
  wage_ratio_c = 5 →
  wage_c = 110 →
  (days_a * (wage_c * wage_ratio_a / wage_ratio_c) +
   days_b * (wage_c * wage_ratio_b / wage_ratio_c) +
   days_c * wage_c) = 1628 :=
by sorry

end NUMINAMATH_CALUDE_total_earning_calculation_l3131_313160


namespace NUMINAMATH_CALUDE_triangle_arithmetic_sides_cosine_identity_l3131_313153

/-- In a triangle ABC where the sides form an arithmetic sequence, 
    5 cos A - 4 cos A cos C + 5 cos C equals 8 -/
theorem triangle_arithmetic_sides_cosine_identity 
  (A B C : ℝ) (a b c : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →
  A + B + C = π →
  2 * b = a + c →
  5 * Real.cos A - 4 * Real.cos A * Real.cos C + 5 * Real.cos C = 8 := by
  sorry

end NUMINAMATH_CALUDE_triangle_arithmetic_sides_cosine_identity_l3131_313153


namespace NUMINAMATH_CALUDE_union_equals_B_l3131_313128

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x ≤ 0}
def B (a : ℝ) : Set ℝ := {x | x - a ≤ 0}

-- State the theorem
theorem union_equals_B (a : ℝ) : A ∪ B a = B a → a ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_union_equals_B_l3131_313128


namespace NUMINAMATH_CALUDE_tan_theta_plus_pi_fourth_l3131_313149

theorem tan_theta_plus_pi_fourth (θ : Real) 
  (h1 : θ > 3 * Real.pi / 2 ∧ θ < 2 * Real.pi) 
  (h2 : Real.cos (θ - Real.pi / 4) = 3 / 5) : 
  Real.tan (θ + Real.pi / 4) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_tan_theta_plus_pi_fourth_l3131_313149


namespace NUMINAMATH_CALUDE_tory_sold_seven_guns_l3131_313100

/-- The number of toy guns Tory sold -/
def tory_guns : ℕ := sorry

/-- The price of each toy phone Bert sold -/
def bert_phone_price : ℕ := 18

/-- The number of toy phones Bert sold -/
def bert_phones : ℕ := 8

/-- The price of each toy gun Tory sold -/
def tory_gun_price : ℕ := 20

/-- The difference in earnings between Bert and Tory -/
def earning_difference : ℕ := 4

theorem tory_sold_seven_guns :
  tory_guns = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_tory_sold_seven_guns_l3131_313100


namespace NUMINAMATH_CALUDE_area_ratio_AMK_ABC_l3131_313102

/-- Triangle ABC with points M on AB and K on AC -/
structure TriangleWithPoints where
  /-- The area of triangle ABC -/
  area_ABC : ℝ
  /-- The ratio of AM to MB -/
  ratio_AM_MB : ℝ × ℝ
  /-- The ratio of AK to KC -/
  ratio_AK_KC : ℝ × ℝ

/-- The theorem stating the area ratio of triangle AMK to triangle ABC -/
theorem area_ratio_AMK_ABC (t : TriangleWithPoints) (h1 : t.area_ABC = 50) 
  (h2 : t.ratio_AM_MB = (1, 5)) (h3 : t.ratio_AK_KC = (3, 2)) : 
  (∃ (area_AMK : ℝ), area_AMK / t.area_ABC = 1 / 10) :=
sorry

end NUMINAMATH_CALUDE_area_ratio_AMK_ABC_l3131_313102


namespace NUMINAMATH_CALUDE_exists_number_with_digit_sum_property_l3131_313116

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Theorem: There exists a natural number n such that the sum of digits of (n + 18) 
    is equal to the sum of digits of n minus 18 -/
theorem exists_number_with_digit_sum_property : 
  ∃ n : ℕ, sumOfDigits (n + 18) = sumOfDigits n - 18 := by sorry

end NUMINAMATH_CALUDE_exists_number_with_digit_sum_property_l3131_313116


namespace NUMINAMATH_CALUDE_completing_square_quadratic_l3131_313147

theorem completing_square_quadratic (x : ℝ) : 
  x^2 + 8*x - 3 = 0 ↔ (x + 4)^2 = 19 :=
sorry

end NUMINAMATH_CALUDE_completing_square_quadratic_l3131_313147


namespace NUMINAMATH_CALUDE_least_marbles_nine_marbles_marbles_solution_l3131_313190

theorem least_marbles (n : ℕ) : n > 0 ∧ n % 6 = 3 ∧ n % 4 = 1 → n ≥ 9 :=
by sorry

theorem nine_marbles : 9 % 6 = 3 ∧ 9 % 4 = 1 :=
by sorry

theorem marbles_solution : ∃ (n : ℕ), n > 0 ∧ n % 6 = 3 ∧ n % 4 = 1 ∧ 
  ∀ (m : ℕ), m > 0 ∧ m % 6 = 3 ∧ m % 4 = 1 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_least_marbles_nine_marbles_marbles_solution_l3131_313190


namespace NUMINAMATH_CALUDE_square_area_on_parabola_l3131_313179

/-- A square with two vertices on a parabola and one side on a line -/
structure SquareOnParabola where
  /-- The side length of the square -/
  side : ℝ
  /-- The y-intercept of the line parallel to y = 2x - 22 that contains two vertices of the square -/
  b : ℝ
  /-- Two vertices of the square lie on the parabola y = x^2 -/
  vertices_on_parabola : ∃ (x₁ x₂ : ℝ), x₁^2 = (2 * x₁ + b) ∧ x₂^2 = (2 * x₂ + b) ∧ (x₁ - x₂)^2 + (x₁^2 - x₂^2)^2 = side^2
  /-- One side of the square lies on the line y = 2x - 22 -/
  side_on_line : side = |b + 22| / Real.sqrt 5

/-- The theorem stating the possible areas of the square -/
theorem square_area_on_parabola (s : SquareOnParabola) :
  s.side^2 = 115.2 ∨ s.side^2 = 156.8 := by
  sorry

end NUMINAMATH_CALUDE_square_area_on_parabola_l3131_313179


namespace NUMINAMATH_CALUDE_parallel_lines_m_value_l3131_313189

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m1 m2 b1 b2 : ℝ} :
  (∀ x y : ℝ, m1 * x + y + b1 = 0 ↔ m2 * x + y + b2 = 0) ↔ m1 = m2

/-- The problem statement -/
theorem parallel_lines_m_value (m : ℝ) :
  (∀ x y : ℝ, 2 * x + (m + 1) * y + 4 = 0 ↔ m * x + 3 * y + 4 = 0) →
  m = -3 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_m_value_l3131_313189


namespace NUMINAMATH_CALUDE_game_ends_after_54_rounds_l3131_313154

/-- Represents a player in the token game -/
structure Player where
  tokens : ℕ

/-- Represents the state of the game -/
structure GameState where
  playerA : Player
  playerB : Player
  playerC : Player
  rounds : ℕ

/-- Simulates one round of the game -/
def playRound (state : GameState) : GameState :=
  sorry

/-- Checks if the game has ended (any player has 0 tokens) -/
def gameEnded (state : GameState) : Bool :=
  sorry

/-- Main theorem: The game ends after exactly 54 rounds -/
theorem game_ends_after_54_rounds :
  let initialState : GameState := {
    playerA := { tokens := 20 },
    playerB := { tokens := 19 },
    playerC := { tokens := 18 },
    rounds := 0
  }
  ∃ (finalState : GameState),
    (finalState.rounds = 54) ∧
    (gameEnded finalState) ∧
    (∀ (intermediateState : GameState),
      intermediateState.rounds < 54 →
      ¬(gameEnded intermediateState)) :=
  sorry

end NUMINAMATH_CALUDE_game_ends_after_54_rounds_l3131_313154


namespace NUMINAMATH_CALUDE_purchase_savings_l3131_313152

/-- Calculates the total savings on a purchase given the original and discounted prices -/
def calculateSavings (originalPrice discountedPrice : ℚ) (quantity : ℕ) : ℚ :=
  (originalPrice - discountedPrice) * quantity

/-- Calculates the discounted price given the original price and discount percentage -/
def calculateDiscountedPrice (originalPrice : ℚ) (discountPercentage : ℚ) : ℚ :=
  originalPrice * (1 - discountPercentage)

theorem purchase_savings :
  let folderQuantity : ℕ := 7
  let folderPrice : ℚ := 3
  let folderDiscount : ℚ := 0.25
  let penQuantity : ℕ := 4
  let penPrice : ℚ := 1.5
  let penDiscount : ℚ := 0.1
  let folderSavings := calculateSavings folderPrice (calculateDiscountedPrice folderPrice folderDiscount) folderQuantity
  let penSavings := calculateSavings penPrice (calculateDiscountedPrice penPrice penDiscount) penQuantity
  folderSavings + penSavings = 5.85 := by
  sorry


end NUMINAMATH_CALUDE_purchase_savings_l3131_313152


namespace NUMINAMATH_CALUDE_all_students_visiting_one_student_visiting_l3131_313113

-- Define the probabilities of each student visiting
def prob_A : ℚ := 2/3
def prob_B : ℚ := 1/4
def prob_C : ℚ := 2/5

-- Theorem for the probability of all three students visiting
theorem all_students_visiting : 
  prob_A * prob_B * prob_C = 1/15 := by
  sorry

-- Theorem for the probability of exactly one student visiting
theorem one_student_visiting : 
  prob_A * (1 - prob_B) * (1 - prob_C) + 
  (1 - prob_A) * prob_B * (1 - prob_C) + 
  (1 - prob_A) * (1 - prob_B) * prob_C = 9/20 := by
  sorry

end NUMINAMATH_CALUDE_all_students_visiting_one_student_visiting_l3131_313113


namespace NUMINAMATH_CALUDE_horner_method_f_at_3_l3131_313124

/-- Horner's method for polynomial evaluation -/
def horner (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = x^5 - 2x^3 + 3x^2 - x + 1 -/
def f : List ℝ := [1, -2, 3, 0, -1, 1]

/-- Theorem: Horner's method applied to f(x) at x = 3 yields v₃ = 24 -/
theorem horner_method_f_at_3 :
  horner f 3 = 24 := by
  sorry

#eval horner f 3

end NUMINAMATH_CALUDE_horner_method_f_at_3_l3131_313124


namespace NUMINAMATH_CALUDE_P_on_y_axis_l3131_313143

/-- A point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the y-axis -/
def is_on_y_axis (p : Point) : Prop :=
  p.x = 0

/-- The point P with coordinates (0, -4) -/
def P : Point :=
  { x := 0, y := -4 }

/-- Theorem: The point P(0, -4) lies on the y-axis -/
theorem P_on_y_axis : is_on_y_axis P := by
  sorry

end NUMINAMATH_CALUDE_P_on_y_axis_l3131_313143


namespace NUMINAMATH_CALUDE_largest_three_digit_base5_l3131_313145

-- Define a function to convert a three-digit base-5 number to base-10
def base5ToBase10 (a b c : Nat) : Nat :=
  a * 5^2 + b * 5^1 + c * 5^0

-- Theorem statement
theorem largest_three_digit_base5 : 
  base5ToBase10 4 4 4 = 124 := by sorry

end NUMINAMATH_CALUDE_largest_three_digit_base5_l3131_313145


namespace NUMINAMATH_CALUDE_quiz_win_probability_l3131_313177

/-- Represents a quiz with multiple-choice questions. -/
structure Quiz where
  num_questions : ℕ
  num_choices : ℕ

/-- Represents the outcome of a quiz attempt. -/
structure QuizOutcome where
  correct_answers : ℕ

/-- The probability of getting a single question correct. -/
def single_question_probability (q : Quiz) : ℚ :=
  1 / q.num_choices

/-- The probability of winning the quiz. -/
def win_probability (q : Quiz) : ℚ :=
  let p := single_question_probability q
  (p ^ q.num_questions) +  -- All correct
  q.num_questions * (p ^ 3 * (1 - p))  -- Exactly 3 correct

/-- The theorem stating the probability of winning the quiz. -/
theorem quiz_win_probability (q : Quiz) (h1 : q.num_questions = 4) (h2 : q.num_choices = 3) :
  win_probability q = 1 / 9 := by
  sorry

#eval win_probability {num_questions := 4, num_choices := 3}

end NUMINAMATH_CALUDE_quiz_win_probability_l3131_313177


namespace NUMINAMATH_CALUDE_tom_stamp_collection_tom_final_collection_l3131_313119

theorem tom_stamp_collection (tom_initial : ℕ) (mike_gift : ℕ) : ℕ :=
  let harry_gift := 2 * mike_gift + 10
  let sarah_gift := 3 * mike_gift - 5
  let total_gifts := mike_gift + harry_gift + sarah_gift
  tom_initial + total_gifts

theorem tom_final_collection :
  tom_stamp_collection 3000 17 = 3107 := by
  sorry

end NUMINAMATH_CALUDE_tom_stamp_collection_tom_final_collection_l3131_313119


namespace NUMINAMATH_CALUDE_parabola_roots_difference_l3131_313199

def parabola (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem parabola_roots_difference (a b c : ℝ) :
  (∃ (h k : ℝ), h = 3 ∧ k = -3 ∧ ∀ x, parabola a b c x = a * (x - h)^2 + k) →
  parabola a b c 5 = 9 →
  (∃ (m n : ℝ), m > n ∧ parabola a b c m = 0 ∧ parabola a b c n = 0) →
  ∃ (m n : ℝ), m - n = 2 :=
sorry

end NUMINAMATH_CALUDE_parabola_roots_difference_l3131_313199


namespace NUMINAMATH_CALUDE_lamp_distribution_and_profit_l3131_313127

/-- Represents the types of lamps --/
inductive LampType
| A
| B

/-- Represents the purchase price of a lamp --/
def purchasePrice (t : LampType) : ℕ :=
  match t with
  | LampType.A => 40
  | LampType.B => 65

/-- Represents the selling price of a lamp --/
def sellingPrice (t : LampType) : ℕ :=
  match t with
  | LampType.A => 60
  | LampType.B => 100

/-- Represents the profit from selling a lamp --/
def profit (t : LampType) : ℕ := sellingPrice t - purchasePrice t

/-- The total number of lamps --/
def totalLamps : ℕ := 50

/-- The total purchase cost --/
def totalPurchaseCost : ℕ := 2500

/-- The minimum total profit --/
def minTotalProfit : ℕ := 1400

theorem lamp_distribution_and_profit :
  (∃ (x y : ℕ),
    x + y = totalLamps ∧
    x * purchasePrice LampType.A + y * purchasePrice LampType.B = totalPurchaseCost ∧
    x = 30 ∧ y = 20) ∧
  (∃ (m : ℕ),
    m * profit LampType.B + (totalLamps - m) * profit LampType.A ≥ minTotalProfit ∧
    m ≥ 27 ∧
    ∀ (n : ℕ), n * profit LampType.B + (totalLamps - n) * profit LampType.A ≥ minTotalProfit → n ≥ m) :=
by sorry

end NUMINAMATH_CALUDE_lamp_distribution_and_profit_l3131_313127


namespace NUMINAMATH_CALUDE_initial_tagged_fish_calculation_l3131_313146

/-- Calculates the number of initially tagged fish in a pond -/
def initiallyTaggedFish (totalFish : ℕ) (secondCatchTotal : ℕ) (secondCatchTagged : ℕ) : ℕ :=
  (totalFish * secondCatchTagged) / secondCatchTotal

theorem initial_tagged_fish_calculation :
  initiallyTaggedFish 1250 50 2 = 50 := by
  sorry

end NUMINAMATH_CALUDE_initial_tagged_fish_calculation_l3131_313146


namespace NUMINAMATH_CALUDE_min_cuts_correct_l3131_313115

/-- The minimum number of cuts required to transform a square into 100 20-gons -/
def min_cuts : ℕ := 1699

/-- The number of sides in a square -/
def square_sides : ℕ := 4

/-- The number of sides in a 20-gon -/
def twenty_gon_sides : ℕ := 20

/-- The number of 20-gons we want to obtain -/
def target_polygons : ℕ := 100

/-- The maximum increase in the number of sides per cut -/
def max_side_increase : ℕ := 4

/-- The total number of sides in the final configuration -/
def total_final_sides : ℕ := target_polygons * twenty_gon_sides

theorem min_cuts_correct :
  min_cuts = (total_final_sides - square_sides) / max_side_increase + 
             (target_polygons - 1) :=
by sorry

end NUMINAMATH_CALUDE_min_cuts_correct_l3131_313115


namespace NUMINAMATH_CALUDE_george_turning_25_l3131_313164

/-- Represents George's age and bill exchange scenario --/
def GeorgeBirthdayProblem (n : ℕ) : Prop :=
  let billsReceived : ℕ := n
  let billsRemaining : ℚ := 0.8 * n
  let exchangeRate : ℚ := 1.5
  let totalExchange : ℚ := 12
  (exchangeRate * billsRemaining = totalExchange) ∧ (n + 15 = 25)

/-- Theorem stating that George is turning 25 years old --/
theorem george_turning_25 : ∃ n : ℕ, GeorgeBirthdayProblem n := by
  sorry

end NUMINAMATH_CALUDE_george_turning_25_l3131_313164


namespace NUMINAMATH_CALUDE_problem_statement_l3131_313167

theorem problem_statement (a b : ℝ) : 
  ({a, b/a, 1} : Set ℝ) = {a^2, a+b, 0} → a^2015 + b^2016 = -1 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l3131_313167


namespace NUMINAMATH_CALUDE_max_moves_card_game_l3131_313196

/-- Represents the state of the cards as a natural number -/
def initial_state : Nat := 43690

/-- Represents a valid move in the game -/
def is_valid_move (n : Nat) : Prop :=
  ∃ k, 0 < k ∧ k ≤ 16 ∧ n.mod (2^k) = 2^(k-1)

/-- The game ends when no valid move can be made -/
def game_ended (n : Nat) : Prop :=
  ¬∃ m, is_valid_move n ∧ m < n

/-- Theorem stating the maximum number of moves in the game -/
theorem max_moves_card_game :
  ∃ moves : Nat, moves = initial_state ∧
  (∀ n, n > moves → ¬∃ seq : Nat → Nat, seq 0 = initial_state ∧
    (∀ i < n, is_valid_move (seq i) ∧ seq (i+1) < seq i) ∧
    game_ended (seq n)) :=
sorry

end NUMINAMATH_CALUDE_max_moves_card_game_l3131_313196


namespace NUMINAMATH_CALUDE_sum_of_roots_l3131_313148

theorem sum_of_roots (p q r s : ℝ) : 
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s →
  (∀ x : ℝ, x^2 - 12*p*x - 13*q = 0 ↔ x = r ∨ x = s) →
  (∀ x : ℝ, x^2 - 12*r*x - 13*s = 0 ↔ x = p ∨ x = q) →
  p + q + r + s = 2028 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_l3131_313148


namespace NUMINAMATH_CALUDE_recycling_program_earnings_l3131_313125

/-- Represents the referral program structure and earnings --/
structure ReferralProgram where
  initial_signup_bonus : ℚ
  first_tier_referral_bonus : ℚ
  second_tier_referral_bonus : ℚ
  friend_signup_bonus : ℚ
  friend_referral_bonus : ℚ
  first_day_referrals : ℕ
  first_day_friends_referrals : ℕ
  week_end_friends_referrals : ℕ
  third_day_referrals : ℕ
  fourth_day_friends_referrals : ℕ

/-- Calculates the total earnings for Katrina and her friends --/
def total_earnings (program : ReferralProgram) : ℚ :=
  sorry

/-- The recycling program referral structure --/
def recycling_program : ReferralProgram := {
  initial_signup_bonus := 5,
  first_tier_referral_bonus := 8,
  second_tier_referral_bonus := 3/2,
  friend_signup_bonus := 5,
  friend_referral_bonus := 2,
  first_day_referrals := 5,
  first_day_friends_referrals := 3,
  week_end_friends_referrals := 2,
  third_day_referrals := 2,
  fourth_day_friends_referrals := 1
}

/-- Theorem stating that the total earnings for Katrina and her friends is $190.50 --/
theorem recycling_program_earnings :
  total_earnings recycling_program = 381/2 := by
  sorry

end NUMINAMATH_CALUDE_recycling_program_earnings_l3131_313125


namespace NUMINAMATH_CALUDE_a_range_l3131_313151

/-- A cubic function with a parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 3 * x^2 - 2

/-- Theorem stating the range of a given the conditions -/
theorem a_range (a : ℝ) :
  (∃! x, f a x = 0) ∧ 
  (∀ x, f a x = 0 → x < 0) →
  a < -Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_a_range_l3131_313151


namespace NUMINAMATH_CALUDE_square_of_85_l3131_313181

theorem square_of_85 : 85^2 = 7225 := by
  sorry

end NUMINAMATH_CALUDE_square_of_85_l3131_313181


namespace NUMINAMATH_CALUDE_little_d_can_win_l3131_313192

/-- Represents a point in the 3D lattice grid -/
structure LatticePoint where
  x : Int
  y : Int
  z : Int

/-- Represents a plane perpendicular to a coordinate axis -/
inductive Plane
  | X (y z : Int)
  | Y (x z : Int)
  | Z (x y : Int)

/-- Represents the state of the game -/
structure GameState where
  markedPoints : Set LatticePoint
  munchedPlanes : Set Plane

/-- Represents a move by Little D -/
def LittleDMove := LatticePoint

/-- Represents a move by Big Z -/
def BigZMove := Plane

/-- A strategy for Little D is a function that takes the current game state
    and returns the next move -/
def LittleDStrategy := GameState → LittleDMove

/-- Check if n consecutive points are marked on a line parallel to a coordinate axis -/
def hasConsecutiveMarkedPoints (state : GameState) (n : Nat) : Prop :=
  ∃ (start : LatticePoint) (axis : Fin 3),
    ∀ i : Fin n,
      let point : LatticePoint :=
        match axis with
        | 0 => ⟨start.x + i.val, start.y, start.z⟩
        | 1 => ⟨start.x, start.y + i.val, start.z⟩
        | 2 => ⟨start.x, start.y, start.z + i.val⟩
      point ∈ state.markedPoints

/-- The main theorem: Little D can win for any n -/
theorem little_d_can_win (n : Nat) :
  ∃ (strategy : LittleDStrategy),
    ∀ (bigZMoves : Nat → BigZMove),
      ∃ (finalState : GameState),
        hasConsecutiveMarkedPoints finalState n :=
  sorry

end NUMINAMATH_CALUDE_little_d_can_win_l3131_313192


namespace NUMINAMATH_CALUDE_ellipse_and_line_properties_l3131_313180

/-- Represents an ellipse in the Cartesian coordinate system -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

/-- Represents a line in the Cartesian coordinate system -/
structure Line where
  k : ℝ

/-- Theorem stating the properties of the ellipse and line -/
theorem ellipse_and_line_properties
  (C : Ellipse)
  (h1 : C.a^2 / C.b^2 = 4 / 3)  -- Eccentricity condition
  (h2 : 1 / C.a^2 + (9/4) / C.b^2 = 1)  -- Point (1, 3/2) lies on the ellipse
  (l : Line)
  (h3 : ∀ x y, y = l.k * (x - 1))  -- Line equation
  (h4 : ∃ x1 y1 x2 y2, 
    x1^2 / C.a^2 + y1^2 / C.b^2 = 1 ∧
    x2^2 / C.a^2 + y2^2 / C.b^2 = 1 ∧
    y1 = l.k * (x1 - 1) ∧
    y2 = l.k * (x2 - 1) ∧
    x1 * x2 + y1 * y2 = -2)  -- Intersection points and dot product condition
  : C.a^2 = 4 ∧ C.b^2 = 3 ∧ l.k^2 = 2 := by sorry

end NUMINAMATH_CALUDE_ellipse_and_line_properties_l3131_313180


namespace NUMINAMATH_CALUDE_gas_fill_calculation_l3131_313163

theorem gas_fill_calculation (cost_today : ℝ) (rollback : ℝ) (friday_fill : ℝ) 
  (total_spend : ℝ) (total_liters : ℝ) 
  (h1 : cost_today = 1.4)
  (h2 : rollback = 0.4)
  (h3 : friday_fill = 25)
  (h4 : total_spend = 39)
  (h5 : total_liters = 35) :
  ∃ (today_fill : ℝ), 
    today_fill = 10 ∧ 
    cost_today * today_fill + (cost_today - rollback) * friday_fill = total_spend ∧
    today_fill + friday_fill = total_liters :=
by sorry

end NUMINAMATH_CALUDE_gas_fill_calculation_l3131_313163


namespace NUMINAMATH_CALUDE_dead_to_total_ratio_is_three_to_five_l3131_313197

/-- Represents the ratio of two natural numbers -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

/-- Represents the problem setup -/
structure FlowerProblem where
  desired_flowers : ℕ
  seeds_per_pack : ℕ
  price_per_pack : ℕ
  total_spent : ℕ

/-- Calculates the ratio of dead seeds to total seeds -/
def dead_to_total_ratio (p : FlowerProblem) : Ratio :=
  let total_packs := p.total_spent / p.price_per_pack
  let total_seeds := total_packs * p.seeds_per_pack
  let dead_seeds := total_seeds - p.desired_flowers
  { numerator := dead_seeds, denominator := total_seeds }

/-- Simplifies a ratio by dividing both numerator and denominator by their GCD -/
def simplify_ratio (r : Ratio) : Ratio :=
  let gcd := Nat.gcd r.numerator r.denominator
  { numerator := r.numerator / gcd, denominator := r.denominator / gcd }

/-- The main theorem to prove -/
theorem dead_to_total_ratio_is_three_to_five (p : FlowerProblem) 
    (h1 : p.desired_flowers = 20)
    (h2 : p.seeds_per_pack = 25)
    (h3 : p.price_per_pack = 5)
    (h4 : p.total_spent = 10) : 
    simplify_ratio (dead_to_total_ratio p) = { numerator := 3, denominator := 5 } := by
  sorry

end NUMINAMATH_CALUDE_dead_to_total_ratio_is_three_to_five_l3131_313197


namespace NUMINAMATH_CALUDE_fib_matrix_power_eq_fib_relation_l3131_313131

/-- Fibonacci matrix -/
def fib_matrix : Matrix (Fin 2) (Fin 2) ℕ := !![1, 1; 1, 0]

/-- n-th power of Fibonacci matrix -/
def fib_matrix_power (n : ℕ) : Matrix (Fin 2) (Fin 2) ℕ := fib_matrix ^ n

/-- n-th Fibonacci number -/
def F (n : ℕ) : ℕ := (fib_matrix_power n) 0 1

/-- Theorem stating the relation between Fibonacci numbers and matrix power -/
theorem fib_matrix_power_eq (n : ℕ) :
  fib_matrix_power n = !![F (n + 1), F n; F n, F (n - 1)] := by sorry

/-- Main theorem to prove -/
theorem fib_relation :
  F 1001 * F 1003 - F 1002 * F 1002 = 1 := by sorry

end NUMINAMATH_CALUDE_fib_matrix_power_eq_fib_relation_l3131_313131


namespace NUMINAMATH_CALUDE_initial_principal_is_500_l3131_313112

/-- Simple interest calculation function -/
def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

/-- Theorem stating that given the conditions, the initial principal must be $500 -/
theorem initial_principal_is_500 :
  ∃ (rate : ℝ),
    simpleInterest 500 rate 2 = 590 ∧
    simpleInterest 500 rate 7 = 815 :=
by
  sorry

#check initial_principal_is_500

end NUMINAMATH_CALUDE_initial_principal_is_500_l3131_313112


namespace NUMINAMATH_CALUDE_paco_initial_cookies_l3131_313139

/-- The number of cookies Paco ate -/
def cookies_eaten : ℕ := 21

/-- The number of cookies Paco had left -/
def cookies_left : ℕ := 7

/-- The initial number of cookies Paco had -/
def initial_cookies : ℕ := cookies_eaten + cookies_left

theorem paco_initial_cookies : initial_cookies = 28 := by sorry

end NUMINAMATH_CALUDE_paco_initial_cookies_l3131_313139


namespace NUMINAMATH_CALUDE_ferris_wheel_problem_l3131_313174

theorem ferris_wheel_problem (capacity : ℕ) (waiting : ℕ) (h1 : capacity = 56) (h2 : waiting = 92) :
  waiting - capacity = 36 := by
  sorry

end NUMINAMATH_CALUDE_ferris_wheel_problem_l3131_313174


namespace NUMINAMATH_CALUDE_arithmetic_mean_geq_product_l3131_313134

theorem arithmetic_mean_geq_product (x y : ℝ) : x * y ≤ (x^2 + y^2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_geq_product_l3131_313134


namespace NUMINAMATH_CALUDE_inequality_range_l3131_313138

/-- Given that m < (e^x) / (x*e^x - x + 1) has exactly two integer solutions, 
    prove that the range of m is [e^2 / (2e^2 - 1), 1) -/
theorem inequality_range (m : ℝ) : 
  (∃! (a b : ℤ), ∀ (x : ℤ), m < (Real.exp x) / (x * Real.exp x - x + 1) ↔ x = a ∨ x = b) →
  m ∈ Set.Ici (Real.exp 2 / (2 * Real.exp 2 - 1)) ∩ Set.Iio 1 :=
by sorry

end NUMINAMATH_CALUDE_inequality_range_l3131_313138


namespace NUMINAMATH_CALUDE_one_weighing_sufficient_l3131_313187

/-- Represents a coin, which can be either real or counterfeit -/
inductive Coin
  | Real
  | Counterfeit

/-- Represents the result of weighing two coins -/
inductive WeighResult
  | Left  -- Left side is lighter
  | Right -- Right side is lighter
  | Equal -- Both sides are equal

/-- A function that simulates weighing two coins -/
def weigh (a b : Coin) : WeighResult :=
  match a, b with
  | Coin.Counterfeit, Coin.Real    => WeighResult.Left
  | Coin.Real, Coin.Counterfeit    => WeighResult.Right
  | Coin.Real, Coin.Real           => WeighResult.Equal
  | Coin.Counterfeit, Coin.Counterfeit => WeighResult.Equal

/-- A function that determines the counterfeit coin given three coins -/
def findCounterfeit (a b c : Coin) : Coin :=
  match weigh a b with
  | WeighResult.Left  => a
  | WeighResult.Right => b
  | WeighResult.Equal => c

theorem one_weighing_sufficient :
  ∀ (a b c : Coin),
  (∃! x, x = Coin.Counterfeit) →
  (a = Coin.Counterfeit ∨ b = Coin.Counterfeit ∨ c = Coin.Counterfeit) →
  findCounterfeit a b c = Coin.Counterfeit :=
by sorry

end NUMINAMATH_CALUDE_one_weighing_sufficient_l3131_313187


namespace NUMINAMATH_CALUDE_cylinder_volume_relation_l3131_313162

/-- Given two cylinders X and Y with the following properties:
    1. The height of X equals the diameter of Y
    2. The diameter of X equals the height of Y (denoted as k)
    3. The volume of X is three times the volume of Y
    This theorem states that the volume of Y can be expressed as (1/4) π k^3 cubic units. -/
theorem cylinder_volume_relation (k : ℝ) (hk : k > 0) :
  ∃ (r_x h_x r_y : ℝ),
    r_x > 0 ∧ h_x > 0 ∧ r_y > 0 ∧
    h_x = 2 * r_y ∧
    2 * r_x = k ∧
    π * r_x^2 * h_x = 3 * (π * r_y^2 * k) ∧
    π * r_y^2 * k = (1/4) * π * k^3 :=
sorry

end NUMINAMATH_CALUDE_cylinder_volume_relation_l3131_313162


namespace NUMINAMATH_CALUDE_min_value_theorem_l3131_313104

-- Define the function f(x) = ax^2 - 4x + c
def f (a c x : ℝ) : ℝ := a * x^2 - 4 * x + c

-- State the theorem
theorem min_value_theorem (a c : ℝ) (h1 : a > 0) 
  (h2 : Set.range (f a c) = Set.Ici 1) :
  ∃ (m : ℝ), m = 3 ∧ ∀ (x : ℝ), (1 / (c - 1)) + (9 / a) ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3131_313104


namespace NUMINAMATH_CALUDE_shifted_sine_function_proof_l3131_313103

open Real

theorem shifted_sine_function_proof (φ : ℝ) (h1 : 0 < φ) (h2 : φ < π/2) : 
  (∃ x₁ x₂ : ℝ, |sin (2*x₁) - sin (2*(x₂ - φ))| = 2 ∧ 
   (∀ y₁ y₂ : ℝ, |sin (2*y₁) - sin (2*(y₂ - φ))| = 2 → |x₁ - x₂| ≤ |y₁ - y₂|) ∧
   |x₁ - x₂| = π/3) →
  φ = π/6 := by
sorry

end NUMINAMATH_CALUDE_shifted_sine_function_proof_l3131_313103


namespace NUMINAMATH_CALUDE_cube_greater_than_one_iff_l3131_313183

theorem cube_greater_than_one_iff (x : ℝ) : x > 1 ↔ x^3 > 1 := by sorry

end NUMINAMATH_CALUDE_cube_greater_than_one_iff_l3131_313183


namespace NUMINAMATH_CALUDE_arithmetic_sequence_and_sum_l3131_313175

-- Define the arithmetic sequence a_n and its sum S_n
def a (n : ℕ) : ℝ := sorry
def S (n : ℕ) : ℝ := sorry

-- Define T_n as the sum of first n terms of 1/S_n
def T (n : ℕ) : ℝ := sorry

-- State the given conditions
axiom S_3_eq_15 : S 3 = 15
axiom a_3_plus_a_8 : a 3 + a 8 = 2 * a 5 + 2

-- State the theorem to be proved
theorem arithmetic_sequence_and_sum (n : ℕ) : 
  a n = 2 * n + 1 ∧ T n < 3/4 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_and_sum_l3131_313175


namespace NUMINAMATH_CALUDE_farm_leg_count_l3131_313161

/-- The number of legs for animals on a farm --/
def farm_legs (total_animals : ℕ) (num_ducks : ℕ) (duck_legs : ℕ) (dog_legs : ℕ) : ℕ :=
  let num_dogs := total_animals - num_ducks
  num_ducks * duck_legs + num_dogs * dog_legs

/-- Theorem stating the total number of legs on the farm --/
theorem farm_leg_count : farm_legs 11 6 2 4 = 32 := by
  sorry

end NUMINAMATH_CALUDE_farm_leg_count_l3131_313161


namespace NUMINAMATH_CALUDE_quadratic_discriminant_l3131_313186

theorem quadratic_discriminant : 
  let a : ℝ := 1
  let b : ℝ := -4
  let c : ℝ := 3
  b^2 - 4*a*c = 4 := by sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_l3131_313186


namespace NUMINAMATH_CALUDE_combined_tax_rate_l3131_313165

/-- Combined tax rate calculation -/
theorem combined_tax_rate 
  (mork_rate : ℝ) 
  (mindy_rate : ℝ) 
  (income_ratio : ℝ) 
  (h1 : mork_rate = 0.4) 
  (h2 : mindy_rate = 0.3) 
  (h3 : income_ratio = 3) :
  (mork_rate + mindy_rate * income_ratio) / (1 + income_ratio) = 0.325 := by
  sorry

end NUMINAMATH_CALUDE_combined_tax_rate_l3131_313165


namespace NUMINAMATH_CALUDE_rice_weight_l3131_313126

/-- Given rice divided equally into 4 containers, with 70 ounces in each container,
    and 1 pound equaling 16 ounces, the total amount of rice is 17.5 pounds. -/
theorem rice_weight (containers : Nat) (ounces_per_container : Nat) (ounces_per_pound : Nat)
    (h1 : containers = 4)
    (h2 : ounces_per_container = 70)
    (h3 : ounces_per_pound = 16) :
    (containers * ounces_per_container : Rat) / ounces_per_pound = 17.5 := by
  sorry

end NUMINAMATH_CALUDE_rice_weight_l3131_313126


namespace NUMINAMATH_CALUDE_fifteenth_term_is_53_l3131_313172

/-- An arithmetic sequence with given first three terms -/
def arithmetic_sequence (a₁ a₂ a₃ : ℤ) : ℕ → ℤ :=
  λ n => a₁ + (n - 1) * (a₂ - a₁)

/-- Theorem: The 15th term of the specific arithmetic sequence is 53 -/
theorem fifteenth_term_is_53 :
  arithmetic_sequence (-3) 1 5 15 = 53 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_term_is_53_l3131_313172


namespace NUMINAMATH_CALUDE_marble_problem_l3131_313188

theorem marble_problem (a : ℚ) 
  (angela : ℚ) (brian : ℚ) (caden : ℚ) (daryl : ℚ)
  (h1 : angela = a)
  (h2 : brian = 2 * a)
  (h3 : caden = 3 * brian)
  (h4 : daryl = 6 * caden)
  (h5 : angela + brian + caden + daryl = 150) :
  a = 10 / 3 := by
sorry

end NUMINAMATH_CALUDE_marble_problem_l3131_313188


namespace NUMINAMATH_CALUDE_banana_arrangements_l3131_313110

theorem banana_arrangements :
  let total_letters : ℕ := 6
  let freq_b : ℕ := 1
  let freq_n : ℕ := 2
  let freq_a : ℕ := 3
  (total_letters = freq_b + freq_n + freq_a) →
  (Nat.factorial total_letters) / (Nat.factorial freq_b * Nat.factorial freq_n * Nat.factorial freq_a) = 60 :=
by sorry

end NUMINAMATH_CALUDE_banana_arrangements_l3131_313110


namespace NUMINAMATH_CALUDE_gwen_homework_l3131_313178

def homework_problem (math_problems science_problems finished_problems : ℕ) : Prop :=
  let total_problems := math_problems + science_problems
  total_problems - finished_problems = 5

theorem gwen_homework :
  homework_problem 18 11 24 := by sorry

end NUMINAMATH_CALUDE_gwen_homework_l3131_313178


namespace NUMINAMATH_CALUDE_sqrt_of_nine_l3131_313108

theorem sqrt_of_nine : {x : ℝ | x ^ 2 = 9} = {3, -3} := by sorry

end NUMINAMATH_CALUDE_sqrt_of_nine_l3131_313108


namespace NUMINAMATH_CALUDE_flagpole_height_correct_l3131_313191

/-- The height of the flagpole in feet -/
def flagpole_height : ℝ := 48

/-- The length of the flagpole's shadow in feet -/
def flagpole_shadow : ℝ := 72

/-- The height of the reference pole in feet -/
def reference_pole_height : ℝ := 18

/-- The length of the reference pole's shadow in feet -/
def reference_pole_shadow : ℝ := 27

/-- Theorem stating that the flagpole height is correct given the shadow lengths -/
theorem flagpole_height_correct :
  flagpole_height * reference_pole_shadow = reference_pole_height * flagpole_shadow :=
by sorry

end NUMINAMATH_CALUDE_flagpole_height_correct_l3131_313191


namespace NUMINAMATH_CALUDE_phone_profit_maximization_l3131_313182

theorem phone_profit_maximization
  (profit_A_B : ℕ → ℕ → ℕ)
  (h1 : profit_A_B 1 1 = 600)
  (h2 : profit_A_B 3 2 = 1400)
  (total_phones : ℕ)
  (h3 : total_phones = 20)
  (h4 : ∀ x y : ℕ, x + y = total_phones → 3 * y ≤ 2 * x) :
  ∃ (x y : ℕ),
    x + y = total_phones ∧
    3 * y ≤ 2 * x ∧
    ∀ (a b : ℕ), a + b = total_phones → 3 * b ≤ 2 * a →
      profit_A_B x y ≥ profit_A_B a b ∧
      profit_A_B x y = 5600 ∧
      x = 12 ∧ y = 8 :=
by sorry

end NUMINAMATH_CALUDE_phone_profit_maximization_l3131_313182


namespace NUMINAMATH_CALUDE_ella_reads_500_pages_l3131_313136

/-- Represents the reading task for Ella and John -/
structure ReadingTask where
  total_pages : ℕ
  ella_pace : ℕ  -- seconds per page
  john_pace : ℕ  -- seconds per page

/-- Calculates the number of pages Ella should read -/
def pages_for_ella (task : ReadingTask) : ℕ :=
  (task.total_pages * task.john_pace) / (task.ella_pace + task.john_pace)

/-- Theorem stating that Ella should read 500 pages given the conditions -/
theorem ella_reads_500_pages (task : ReadingTask) 
  (h1 : task.total_pages = 900)
  (h2 : task.ella_pace = 40)
  (h3 : task.john_pace = 50) : 
  pages_for_ella task = 500 := by
  sorry

#eval pages_for_ella ⟨900, 40, 50⟩

end NUMINAMATH_CALUDE_ella_reads_500_pages_l3131_313136
