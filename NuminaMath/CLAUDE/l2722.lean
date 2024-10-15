import Mathlib

namespace NUMINAMATH_CALUDE_largest_number_l2722_272256

theorem largest_number : ∀ (a b c d : ℝ), 
  a = -|(-4)| ∧ b = 0 ∧ c = 1 ∧ d = -(-3) →
  d ≥ a ∧ d ≥ b ∧ d ≥ c :=
by sorry

end NUMINAMATH_CALUDE_largest_number_l2722_272256


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l2722_272224

theorem quadratic_roots_property (r s : ℝ) : 
  (3 * r^2 - 5 * r - 7 = 0) → 
  (3 * s^2 - 5 * s - 7 = 0) → 
  (r ≠ s) →
  (4 * r^2 - 4 * s^2) / (r - s) = 20 / 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l2722_272224


namespace NUMINAMATH_CALUDE_vector_magnitude_l2722_272226

/-- Given two vectors a and b in a 2D space, if the angle between them is 120°,
    |a| = 2, and |a + b| = √7, then |b| = 3. -/
theorem vector_magnitude (a b : ℝ × ℝ) : 
  let θ := Real.arccos (-1/2)  -- 120° in radians
  (a.1 * b.1 + a.2 * b.2) / (Real.sqrt ((a.1^2 + a.2^2) * (b.1^2 + b.2^2))) = Real.cos θ →
  Real.sqrt (a.1^2 + a.2^2) = 2 →
  Real.sqrt ((a.1 + b.1)^2 + (a.2 + b.2)^2) = Real.sqrt 7 →
  Real.sqrt (b.1^2 + b.2^2) = 3 := by
sorry

end NUMINAMATH_CALUDE_vector_magnitude_l2722_272226


namespace NUMINAMATH_CALUDE_anna_toy_production_l2722_272296

/-- Anna's toy production problem -/
theorem anna_toy_production (t : ℕ) : 
  let w : ℕ := 3 * t
  let monday_production : ℕ := w * t
  let tuesday_production : ℕ := (w + 5) * (t - 3)
  monday_production - tuesday_production = 4 * t + 15 := by
sorry

end NUMINAMATH_CALUDE_anna_toy_production_l2722_272296


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_exp_greater_than_x_l2722_272206

theorem negation_of_existence (p : ℝ → Prop) :
  (¬ ∃ x, p x) ↔ (∀ x, ¬ p x) :=
by sorry

theorem negation_of_exp_greater_than_x :
  (¬ ∃ x : ℝ, Real.exp x > x) ↔ (∀ x : ℝ, Real.exp x ≤ x) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_exp_greater_than_x_l2722_272206


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l2722_272211

theorem sufficient_but_not_necessary :
  (∃ x : ℝ, (|x - 1| < 4 ∧ ¬(x * (x - 5) < 0))) ∧
  (∀ x : ℝ, (x * (x - 5) < 0) → |x - 1| < 4) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l2722_272211


namespace NUMINAMATH_CALUDE_common_chord_equation_l2722_272222

/-- The equation of the common chord of two circles -/
def common_chord (c1 c2 : ℝ × ℝ → Prop) : ℝ × ℝ → Prop :=
  fun p => c1 p ∧ c2 p

/-- First circle equation -/
def circle1 : ℝ × ℝ → Prop :=
  fun (x, y) => x^2 + y^2 + 2*x = 0

/-- Second circle equation -/
def circle2 : ℝ × ℝ → Prop :=
  fun (x, y) => x^2 + y^2 - 4*y = 0

/-- The proposed common chord equation -/
def proposed_chord : ℝ × ℝ → Prop :=
  fun (x, y) => x + 2*y = 0

theorem common_chord_equation :
  common_chord circle1 circle2 = proposed_chord := by
  sorry

end NUMINAMATH_CALUDE_common_chord_equation_l2722_272222


namespace NUMINAMATH_CALUDE_cookie_box_calories_l2722_272228

theorem cookie_box_calories (bags_per_box : ℕ) (cookies_per_bag : ℕ) (calories_per_cookie : ℕ) 
  (h1 : bags_per_box = 6)
  (h2 : cookies_per_bag = 25)
  (h3 : calories_per_cookie = 18) :
  bags_per_box * cookies_per_bag * calories_per_cookie = 2700 := by
  sorry

end NUMINAMATH_CALUDE_cookie_box_calories_l2722_272228


namespace NUMINAMATH_CALUDE_triangle_side_angle_relation_l2722_272220

-- Define a triangle structure
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  α : ℝ
  β : ℝ
  γ : ℝ

-- Define the theorem
theorem triangle_side_angle_relation (t : Triangle) 
  (h1 : t.α > 0 ∧ t.β > 0 ∧ t.γ > 0)
  (h2 : t.α + t.β + t.γ = Real.pi)
  (h3 : 3 * t.α + 2 * t.β = Real.pi) :
  t.a^2 + t.b * t.c - t.c^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_angle_relation_l2722_272220


namespace NUMINAMATH_CALUDE_triangle_side_ratio_l2722_272280

open Real

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if 9sin²B = 4sin²A and cosC = 1/4, then c/a = √(10)/3 -/
theorem triangle_side_ratio (a b c A B C : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : 9 * (sin B)^2 = 4 * (sin A)^2) (h5 : cos C = 1/4) :
  c / a = sqrt 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_ratio_l2722_272280


namespace NUMINAMATH_CALUDE_alice_gives_no_stickers_to_charlie_l2722_272284

/-- Represents the sticker distribution problem --/
def sticker_distribution (c : ℕ) : Prop :=
  let alice_initial := 12 * c
  let bob_initial := 3 * c
  let charlie_initial := c
  let dave_initial := c
  let alice_final := alice_initial - (2 * c - bob_initial) - (3 * c - dave_initial)
  let bob_final := 2 * c
  let charlie_final := c
  let dave_final := 3 * c
  (alice_final - alice_initial) / alice_initial = 0

/-- Theorem stating that Alice gives 0 fraction of her stickers to Charlie --/
theorem alice_gives_no_stickers_to_charlie (c : ℕ) (hc : c > 0) :
  sticker_distribution c :=
sorry

end NUMINAMATH_CALUDE_alice_gives_no_stickers_to_charlie_l2722_272284


namespace NUMINAMATH_CALUDE_power_function_inequality_l2722_272200

theorem power_function_inequality : let f : ℝ → ℝ := fun x ↦ x^3
  let a : ℝ := f (Real.sqrt 3 / 3)
  let b : ℝ := f (Real.log π)
  let c : ℝ := f (Real.sqrt 2 / 2)
  a < c ∧ c < b := by sorry

end NUMINAMATH_CALUDE_power_function_inequality_l2722_272200


namespace NUMINAMATH_CALUDE_evaluate_expression_l2722_272263

theorem evaluate_expression : 9^6 * 3^3 / 3^15 = 1 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2722_272263


namespace NUMINAMATH_CALUDE_jills_age_l2722_272260

/-- Given that the sum of Henry and Jill's present ages is 40,
    and 11 years ago Henry was twice the age of Jill,
    prove that Jill's present age is 17 years. -/
theorem jills_age (henry_age jill_age : ℕ) 
  (sum_ages : henry_age + jill_age = 40)
  (past_relation : henry_age - 11 = 2 * (jill_age - 11)) :
  jill_age = 17 := by
  sorry

end NUMINAMATH_CALUDE_jills_age_l2722_272260


namespace NUMINAMATH_CALUDE_range_of_m_l2722_272255

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, x^2 + 3*x - 4 < 0 → (x - m)*(x - m - 3) > 0) ∧ 
  (∃ x : ℝ, (x - m)*(x - m - 3) > 0 ∧ x^2 + 3*x - 4 ≥ 0) →
  m ≤ -7 ∨ m ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l2722_272255


namespace NUMINAMATH_CALUDE_binary_subtraction_equiv_l2722_272288

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

def decimal_to_binary (n : Nat) : List Bool :=
  if n = 0 then [false] else
  let rec aux (m : Nat) (acc : List Bool) : List Bool :=
    if m = 0 then acc else aux (m / 2) ((m % 2 = 1) :: acc)
  aux n []

theorem binary_subtraction_equiv :
  let a := [true, true, false, true, true]  -- 11011 in binary
  let b := [true, false, true]              -- 101 in binary
  let result := [true, true, false, true]   -- 1011 in binary (11 in decimal)
  binary_to_decimal a - binary_to_decimal b = binary_to_decimal result :=
by
  sorry

#eval binary_to_decimal [true, true, false, true, true]  -- Should output 27
#eval binary_to_decimal [true, false, true]              -- Should output 5
#eval binary_to_decimal [true, true, false, true]        -- Should output 11

end NUMINAMATH_CALUDE_binary_subtraction_equiv_l2722_272288


namespace NUMINAMATH_CALUDE_praveen_initial_investment_l2722_272205

-- Define the initial parameters
def haris_investment : ℕ := 8280
def praveens_time : ℕ := 12
def haris_time : ℕ := 7
def profit_ratio_praveen : ℕ := 2
def profit_ratio_hari : ℕ := 3

-- Define Praveen's investment as a function
def praveens_investment : ℕ := 
  (haris_investment * haris_time * profit_ratio_praveen) / (praveens_time * profit_ratio_hari)

-- Theorem statement
theorem praveen_initial_investment :
  praveens_investment = 3220 :=
sorry

end NUMINAMATH_CALUDE_praveen_initial_investment_l2722_272205


namespace NUMINAMATH_CALUDE_carissa_street_crossing_l2722_272246

/-- Carissa's street crossing problem -/
theorem carissa_street_crossing 
  (walking_speed : ℝ) 
  (street_width : ℝ) 
  (total_time : ℝ) 
  (n : ℝ) 
  (h1 : walking_speed = 2) 
  (h2 : street_width = 260) 
  (h3 : total_time = 30) 
  (h4 : n > 0) :
  let running_speed := n * walking_speed
  let walking_time := total_time / (1 + n)
  let running_time := n * walking_time
  walking_speed * walking_time + running_speed * running_time = street_width →
  running_speed = 10 := by sorry

end NUMINAMATH_CALUDE_carissa_street_crossing_l2722_272246


namespace NUMINAMATH_CALUDE_fraction_comparison_l2722_272249

theorem fraction_comparison (x : ℝ) (hx : x > 0) :
  ∀ n : ℕ, (x^n + 1) / (x^(n+1) + 1) > (x^(n+1) + 1) / (x^(n+2) + 1) :=
by sorry

end NUMINAMATH_CALUDE_fraction_comparison_l2722_272249


namespace NUMINAMATH_CALUDE_dodge_trucks_count_l2722_272242

/-- Represents the number of vehicles of each type in the parking lot -/
structure ParkingLot where
  dodge : ℚ
  ford : ℚ
  toyota : ℚ
  nissan : ℚ
  volkswagen : ℚ
  honda : ℚ
  mazda : ℚ
  chevrolet : ℚ
  subaru : ℚ
  fiat : ℚ

/-- The conditions of the parking lot -/
def validParkingLot (p : ParkingLot) : Prop :=
  p.ford = (1/3) * p.dodge ∧
  p.ford = 2 * p.toyota ∧
  p.toyota = (7/9) * p.nissan ∧
  p.volkswagen = (1/2) * p.toyota ∧
  p.honda = (3/4) * p.ford ∧
  p.mazda = (2/5) * p.nissan ∧
  p.chevrolet = (2/3) * p.honda ∧
  p.subaru = 4 * p.dodge ∧
  p.fiat = (1/2) * p.mazda ∧
  p.volkswagen = 5

theorem dodge_trucks_count (p : ParkingLot) (h : validParkingLot p) : p.dodge = 60 := by
  sorry

end NUMINAMATH_CALUDE_dodge_trucks_count_l2722_272242


namespace NUMINAMATH_CALUDE_remainder_of_valid_polynomials_l2722_272251

/-- The number of elements in the tuple -/
def tuple_size : ℕ := 2011

/-- The upper bound for each element in the tuple -/
def upper_bound : ℕ := 2011^2

/-- The degree of the polynomial -/
def poly_degree : ℕ := 4019

/-- The modulus for the divisibility conditions -/
def modulus : ℕ := 2011^2

/-- The final modulus for the remainder -/
def final_modulus : ℕ := 1000

/-- The expected remainder -/
def expected_remainder : ℕ := 281

/-- A function representing the conditions on the polynomial -/
def valid_polynomial (f : ℤ → ℤ) : Prop :=
  (∀ n : ℤ, ∃ k : ℤ, f n = k) ∧
  (∀ i : ℕ, i ≤ tuple_size → ∃ k : ℤ, f i - k = modulus * (f i / modulus)) ∧
  (∀ n : ℤ, ∃ k : ℤ, f (n + tuple_size) - f n = modulus * k)

/-- The main theorem -/
theorem remainder_of_valid_polynomials :
  (upper_bound ^ (poly_degree + 1)) % final_modulus = expected_remainder := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_valid_polynomials_l2722_272251


namespace NUMINAMATH_CALUDE_sector_arc_length_l2722_272225

/-- Given a sector with circumference 8 and central angle 2 radians, 
    the length of its arc is 4 -/
theorem sector_arc_length (c : ℝ) (θ : ℝ) (l : ℝ) (r : ℝ) : 
  c = 8 →  -- circumference of the sector
  θ = 2 →  -- central angle in radians
  c = l + 2 * r →  -- circumference formula for a sector
  l = r * θ →  -- arc length formula
  l = 4 := by
  sorry

end NUMINAMATH_CALUDE_sector_arc_length_l2722_272225


namespace NUMINAMATH_CALUDE_sqrt_2_simplest_l2722_272247

-- Define a function to represent the simplicity of a square root
def is_simplest_sqrt (x : ℝ) : Prop :=
  ∀ y : ℝ, y > 0 → y ≠ 1 → x ≠ y * y * (x / (y * y))

-- State the theorem
theorem sqrt_2_simplest : 
  is_simplest_sqrt (Real.sqrt 2) ∧ 
  ¬ is_simplest_sqrt (Real.sqrt 20) ∧ 
  ¬ is_simplest_sqrt (Real.sqrt (1/2)) ∧ 
  ¬ is_simplest_sqrt (Real.sqrt 0.2) :=
sorry

end NUMINAMATH_CALUDE_sqrt_2_simplest_l2722_272247


namespace NUMINAMATH_CALUDE_reciprocal_sum_of_roots_l2722_272227

theorem reciprocal_sum_of_roots (γ δ : ℝ) : 
  (∃ r s : ℝ, 6 * r^2 - 11 * r + 7 = 0 ∧ 
              6 * s^2 - 11 * s + 7 = 0 ∧ 
              γ = 1 / r ∧ 
              δ = 1 / s) → 
  γ + δ = 11 / 7 := by
sorry

end NUMINAMATH_CALUDE_reciprocal_sum_of_roots_l2722_272227


namespace NUMINAMATH_CALUDE_perp_condition_for_parallel_l2722_272238

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Plane → Prop)
variable (perpLine : Line → Line → Prop)
variable (parallel : Plane → Plane → Prop)
variable (subset : Line → Plane → Prop)

-- Define the given lines and planes
variable (a b : Line) (α β : Plane)

-- State the theorem
theorem perp_condition_for_parallel 
  (h1 : perp a α) 
  (h2 : subset b β) :
  (∀ α β, parallel α β → perpLine a b) ∧ 
  (∃ α β, perpLine a b ∧ ¬parallel α β) :=
sorry

end NUMINAMATH_CALUDE_perp_condition_for_parallel_l2722_272238


namespace NUMINAMATH_CALUDE_workshop_average_salary_l2722_272258

-- Define the total number of workers
def total_workers : ℕ := 14

-- Define the number of technicians
def num_technicians : ℕ := 7

-- Define the average salary of technicians
def avg_salary_technicians : ℕ := 12000

-- Define the average salary of other workers
def avg_salary_others : ℕ := 8000

-- Theorem statement
theorem workshop_average_salary :
  (num_technicians * avg_salary_technicians + (total_workers - num_technicians) * avg_salary_others) / total_workers = 10000 := by
  sorry

end NUMINAMATH_CALUDE_workshop_average_salary_l2722_272258


namespace NUMINAMATH_CALUDE_consecutive_number_pair_l2722_272202

theorem consecutive_number_pair (a b : ℤ) : 
  (a = 18 ∨ b = 18) → -- One of the numbers is 18
  abs (a - b) = 1 → -- The numbers are consecutive
  a + b = 35 → -- Their sum is 35
  (a + b) % 5 = 0 → -- The sum is divisible by 5
  (a = 17 ∨ b = 17) := by sorry

end NUMINAMATH_CALUDE_consecutive_number_pair_l2722_272202


namespace NUMINAMATH_CALUDE_fixed_point_theorem_l2722_272295

-- Define the curve E
def E (x y : ℝ) : Prop := y^2 = 4*x

-- Define a line passing through two points
def Line (x1 y1 x2 y2 : ℝ) (x y : ℝ) : Prop :=
  (y - y1) * (x2 - x1) = (x - x1) * (y2 - y1)

-- Define a line with a given slope passing through a point
def LineWithSlope (x0 y0 m : ℝ) (x y : ℝ) : Prop :=
  y - y0 = m * (x - x0)

theorem fixed_point_theorem (xA yA xB yB xC yC : ℝ) :
  E xA yA →
  E xB yB →
  E xC yC →
  Line (-3) 2 xA yA xB yB →
  LineWithSlope xA yA 1 xC yC →
  Line xB yB xC yC 5 2 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_theorem_l2722_272295


namespace NUMINAMATH_CALUDE_parallel_line_theorem_perpendicular_line_theorem_l2722_272257

-- Define the given line
def given_line : Set (ℝ × ℝ) := {(x, y) | 2 * x + 3 * y + 5 = 0}

-- Define the point that line l passes through
def point : ℝ × ℝ := (1, -4)

-- Define parallel line
def parallel_line (m : ℝ) : Set (ℝ × ℝ) := {(x, y) | 2 * x + 3 * y + m = 0}

-- Define perpendicular line
def perpendicular_line (n : ℝ) : Set (ℝ × ℝ) := {(x, y) | 3 * x - 2 * y - n = 0}

-- Theorem for parallel case
theorem parallel_line_theorem :
  ∃ m : ℝ, point ∈ parallel_line m ∧ m = 10 :=
sorry

-- Theorem for perpendicular case
theorem perpendicular_line_theorem :
  ∃ n : ℝ, point ∈ perpendicular_line n ∧ n = 11 :=
sorry

end NUMINAMATH_CALUDE_parallel_line_theorem_perpendicular_line_theorem_l2722_272257


namespace NUMINAMATH_CALUDE_system_solution_ratio_l2722_272239

/-- Given a system of linear equations with non-zero solutions x, y, and z,
    prove that xz/y^2 = 175 -/
theorem system_solution_ratio (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) 
  (eq1 : x + (95/3)*y + 4*z = 0)
  (eq2 : 4*x + (95/3)*y - 3*z = 0)
  (eq3 : 3*x + 5*y - 4*z = 0) :
  x*z/y^2 = 175 := by
sorry


end NUMINAMATH_CALUDE_system_solution_ratio_l2722_272239


namespace NUMINAMATH_CALUDE_ratio_fraction_equality_l2722_272218

theorem ratio_fraction_equality (a b c : ℝ) (h : a ≠ 0) :
  (a : ℝ) / 2 = b / 3 ∧ a / 2 = c / 4 →
  (a - b + c) / b = 1 := by
sorry

end NUMINAMATH_CALUDE_ratio_fraction_equality_l2722_272218


namespace NUMINAMATH_CALUDE_eight_squares_exist_l2722_272232

/-- Represents a 3x3 square of digits -/
def Square := Matrix (Fin 3) (Fin 3) Nat

/-- Checks if a square uses all digits from 1 to 9 exactly once -/
def uses_all_digits (s : Square) : Prop :=
  ∀ d : Fin 9, ∃! (i j : Fin 3), s i j = d.val + 1

/-- Calculates the sum of a row in a square -/
def row_sum (s : Square) (i : Fin 3) : Nat :=
  (s i 0) + (s i 1) + (s i 2)

/-- Checks if the sum of the first two rows equals the sum of the third row -/
def sum_property (s : Square) : Prop :=
  row_sum s 0 + row_sum s 1 = row_sum s 2

/-- Calculates the difference between row sums -/
def row_sum_diff (s : Square) : Nat :=
  (row_sum s 2) - (row_sum s 1)

/-- The main theorem statement -/
theorem eight_squares_exist : 
  ∃ (squares : Fin 8 → Square),
    (∀ i : Fin 8, uses_all_digits (squares i)) ∧
    (∀ i : Fin 8, sum_property (squares i)) ∧
    (∀ i j : Fin 8, row_sum_diff (squares i) = row_sum_diff (squares j)) ∧
    (∀ i : Fin 8, row_sum_diff (squares i) = 9) :=
  sorry

end NUMINAMATH_CALUDE_eight_squares_exist_l2722_272232


namespace NUMINAMATH_CALUDE_perpendicular_tangents_intersection_l2722_272281

/-- Given two curves y = x^2 - 1 and y = 1 + x^3 with perpendicular tangents at x = x₀, 
    prove that x₀ = - ∛(36) / 6 -/
theorem perpendicular_tangents_intersection (x₀ : ℝ) : 
  (∀ x, (2 * x) * (3 * x^2) = -1 → x = x₀) →
  x₀ = - (36 : ℝ)^(1/3) / 6 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_tangents_intersection_l2722_272281


namespace NUMINAMATH_CALUDE_yolanda_scoring_l2722_272265

/-- Yolanda's basketball scoring problem -/
theorem yolanda_scoring (total_points : ℕ) (num_games : ℕ) (avg_free_throws : ℕ) (avg_two_pointers : ℕ) 
  (h1 : total_points = 345)
  (h2 : num_games = 15)
  (h3 : avg_free_throws = 4)
  (h4 : avg_two_pointers = 5) :
  (total_points / num_games - (avg_free_throws * 1 + avg_two_pointers * 2)) / 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_yolanda_scoring_l2722_272265


namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l2722_272216

theorem unique_solution_for_equation :
  ∃! (m n : ℝ), 21 * (m^2 + n) + 21 * Real.sqrt n = 21 * (-m^3 + n^2) + 21 * m^2 * n ∧ m = -1 ∧ n = 0 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_for_equation_l2722_272216


namespace NUMINAMATH_CALUDE_a_divides_next_squared_plus_next_plus_one_l2722_272285

def a : ℕ → ℤ
  | 0 => 1
  | 1 => 1
  | (n + 2) => 5 * a (n + 1) - a n - 1

theorem a_divides_next_squared_plus_next_plus_one :
  ∀ n : ℕ, (a n) ∣ ((a (n + 1))^2 + a (n + 1) + 1) :=
by sorry

end NUMINAMATH_CALUDE_a_divides_next_squared_plus_next_plus_one_l2722_272285


namespace NUMINAMATH_CALUDE_hannahs_remaining_money_l2722_272235

/-- The problem of calculating Hannah's remaining money after selling cookies and cupcakes and buying measuring spoons. -/
theorem hannahs_remaining_money :
  let cookie_count : ℕ := 40
  let cookie_price : ℚ := 4/5  -- $0.8 expressed as a rational number
  let cupcake_count : ℕ := 30
  let cupcake_price : ℚ := 2
  let spoon_set_count : ℕ := 2
  let spoon_set_price : ℚ := 13/2  -- $6.5 expressed as a rational number
  
  let total_sales := cookie_count * cookie_price + cupcake_count * cupcake_price
  let total_spent := spoon_set_count * spoon_set_price
  let remaining_money := total_sales - total_spent

  remaining_money = 79 := by
  sorry

end NUMINAMATH_CALUDE_hannahs_remaining_money_l2722_272235


namespace NUMINAMATH_CALUDE_girls_not_participating_count_l2722_272209

/-- Represents the number of students in an extracurricular activity -/
structure Activity where
  total : ℕ
  boys : ℕ
  girls : ℕ

/-- Represents the school's student body and activities -/
structure School where
  total_students : ℕ
  total_boys : ℕ
  total_girls : ℕ
  soccer : Activity
  basketball : Activity
  chess : Activity
  math : Activity
  glee : Activity
  absent_boys : ℕ
  absent_girls : ℕ

/-- The number of girls not participating in any extracurricular activities -/
def girls_not_participating (s : School) : ℕ :=
  s.total_girls - s.soccer.girls - s.basketball.girls - s.chess.girls - s.math.girls - s.glee.girls - s.absent_girls

theorem girls_not_participating_count (s : School) :
  s.total_students = 800 ∧
  s.total_boys = 420 ∧
  s.total_girls = 380 ∧
  s.soccer.total = 320 ∧
  s.soccer.boys = 224 ∧
  s.basketball.total = 280 ∧
  s.basketball.girls = 182 ∧
  s.chess.total = 70 ∧
  s.chess.boys = 56 ∧
  s.math.total = 50 ∧
  s.math.boys = 25 ∧
  s.math.girls = 25 ∧
  s.absent_boys = 21 ∧
  s.absent_girls = 30 →
  girls_not_participating s = 33 := by
  sorry


end NUMINAMATH_CALUDE_girls_not_participating_count_l2722_272209


namespace NUMINAMATH_CALUDE_weight_of_replaced_person_l2722_272293

theorem weight_of_replaced_person
  (n : ℕ) 
  (avg_increase : ℝ)
  (new_person_weight : ℝ) :
  n = 8 →
  avg_increase = 2.5 →
  new_person_weight = 95 →
  new_person_weight - n * avg_increase = 75 :=
by sorry

end NUMINAMATH_CALUDE_weight_of_replaced_person_l2722_272293


namespace NUMINAMATH_CALUDE_ninety_sixth_digit_of_5_div_37_l2722_272299

/-- The decimal representation of 5/37 has a repeating pattern of length 3 -/
def decimal_repeat_length : ℕ := 3

/-- The repeating pattern in the decimal representation of 5/37 -/
def decimal_pattern : Fin 3 → ℕ
| 0 => 1
| 1 => 3
| 2 => 5

/-- The 96th digit after the decimal point in the decimal representation of 5/37 is 5 -/
theorem ninety_sixth_digit_of_5_div_37 : 
  decimal_pattern ((96 : ℕ) % decimal_repeat_length) = 5 := by
  sorry

end NUMINAMATH_CALUDE_ninety_sixth_digit_of_5_div_37_l2722_272299


namespace NUMINAMATH_CALUDE_train_speed_l2722_272276

/-- The speed of two trains crossing each other -/
theorem train_speed (train_length : ℝ) (crossing_time : ℝ) (h1 : train_length = 120) (h2 : crossing_time = 16) :
  let relative_speed := 2 * train_length / crossing_time
  let train_speed := relative_speed / 2
  let train_speed_kmh := train_speed * 3.6
  train_speed_kmh = 27 := by sorry

end NUMINAMATH_CALUDE_train_speed_l2722_272276


namespace NUMINAMATH_CALUDE_polynomial_equality_l2722_272221

theorem polynomial_equality (k : ℝ) : 
  (∀ x : ℝ, (x + 6) * (x - 5) = x^2 + k*x - 30) → k = 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l2722_272221


namespace NUMINAMATH_CALUDE_quadratic_roots_range_l2722_272245

theorem quadratic_roots_range (a : ℝ) (x₁ x₂ : ℝ) :
  (∀ x, x^2 + (3*a - 1)*x + a + 8 = 0 ↔ x = x₁ ∨ x = x₂) →
  x₁ ≠ x₂ →
  x₁ < 1 →
  x₂ > 1 →
  a < -2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_range_l2722_272245


namespace NUMINAMATH_CALUDE_both_selected_probability_l2722_272207

theorem both_selected_probability (ram_prob ravi_prob : ℚ) 
  (h1 : ram_prob = 2/7)
  (h2 : ravi_prob = 1/5) :
  ram_prob * ravi_prob = 2/35 := by
  sorry

end NUMINAMATH_CALUDE_both_selected_probability_l2722_272207


namespace NUMINAMATH_CALUDE_jake_watching_show_l2722_272219

theorem jake_watching_show (total_show_length : ℝ) (friday_watch_time : ℝ)
  (monday_fraction : ℝ) (tuesday_watch_time : ℝ) (thursday_fraction : ℝ) :
  total_show_length = 52 →
  friday_watch_time = 19 →
  monday_fraction = 1/2 →
  tuesday_watch_time = 4 →
  thursday_fraction = 1/2 →
  ∃ (wednesday_fraction : ℝ),
    wednesday_fraction = 1/4 ∧
    total_show_length = 
      (monday_fraction * 24 + tuesday_watch_time + wednesday_fraction * 24 +
       thursday_fraction * (monday_fraction * 24 + tuesday_watch_time + wednesday_fraction * 24)) +
      friday_watch_time :=
by
  sorry

#check jake_watching_show

end NUMINAMATH_CALUDE_jake_watching_show_l2722_272219


namespace NUMINAMATH_CALUDE_max_value_abc_l2722_272290

theorem max_value_abc (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) (hsum : a + b + c = 3) :
  a + Real.sqrt (a * b) + (a * b * c) ^ (1/4) ≤ 3 ∧
  ∃ a' b' c' : ℝ, a' ≥ 0 ∧ b' ≥ 0 ∧ c' ≥ 0 ∧ a' + b' + c' = 3 ∧
    a' + Real.sqrt (a' * b') + (a' * b' * c') ^ (1/4) = 3 :=
sorry

end NUMINAMATH_CALUDE_max_value_abc_l2722_272290


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l2722_272294

theorem inverse_variation_problem (k : ℝ) (h1 : k > 0) :
  (∀ x y : ℝ, x ≠ 0 → y * x^2 = k) →
  (2 * 3^2 = k) →
  (∃ x : ℝ, x > 0 ∧ 8 * x^2 = k) →
  (∃ x : ℝ, x > 0 ∧ 8 * x^2 = k ∧ x = 3/2) :=
by sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l2722_272294


namespace NUMINAMATH_CALUDE_unique_n_divisible_by_11_l2722_272269

theorem unique_n_divisible_by_11 : ∃! n : ℕ, 0 < n ∧ n < 11 ∧ (18888 - n) % 11 = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_n_divisible_by_11_l2722_272269


namespace NUMINAMATH_CALUDE_debt_work_hours_l2722_272286

def initial_debt : ℝ := 100
def payment : ℝ := 40
def hourly_rate : ℝ := 15

theorem debt_work_hours : 
  (initial_debt - payment) / hourly_rate = 4 := by sorry

end NUMINAMATH_CALUDE_debt_work_hours_l2722_272286


namespace NUMINAMATH_CALUDE_geometric_sum_five_terms_l2722_272234

theorem geometric_sum_five_terms (a r : ℚ) (h1 : a = 1/4) (h2 : r = 1/4) :
  let S := a + a*r + a*r^2 + a*r^3 + a*r^4
  S = 341/1024 := by sorry

end NUMINAMATH_CALUDE_geometric_sum_five_terms_l2722_272234


namespace NUMINAMATH_CALUDE_min_value_theorem_l2722_272215

theorem min_value_theorem (x : ℝ) (h : x > 0) :
  x^2 + 12*x + 108/x^4 ≥ 36 ∧ 
  (x^2 + 12*x + 108/x^4 = 36 ↔ x = 3) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2722_272215


namespace NUMINAMATH_CALUDE_g_is_even_l2722_272278

/-- A function satisfying the given functional equation -/
def FunctionalEq (f : ℝ → ℝ) : Prop :=
  ∀ a b : ℝ, f (a + b) + f (a - b) = 2 * f a + 2 * f b

/-- The function g defined in terms of f -/
def g (f : ℝ → ℝ) (x : ℝ) : ℝ := x^2 * f x

/-- Theorem stating that g is an even function -/
theorem g_is_even (f : ℝ → ℝ) (h : FunctionalEq f) :
  ∀ x : ℝ, g f (-x) = g f x :=
by sorry

end NUMINAMATH_CALUDE_g_is_even_l2722_272278


namespace NUMINAMATH_CALUDE_perfect_square_binomial_l2722_272266

theorem perfect_square_binomial (c : ℚ) : 
  (∃ t u : ℚ, ∀ x : ℚ, c * x^2 + (45/2) * x + 1 = (t * x + u)^2) → 
  c = 2025/16 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_binomial_l2722_272266


namespace NUMINAMATH_CALUDE_square_roots_theorem_l2722_272248

theorem square_roots_theorem (x a : ℝ) (hx : x > 0) 
  (h1 : (a + 1) ^ 2 = x) (h2 : (2 * a - 7) ^ 2 = x) : a = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_roots_theorem_l2722_272248


namespace NUMINAMATH_CALUDE_coin_container_total_l2722_272214

theorem coin_container_total : ∃ (x : ℕ),
  (x * 1 + x * 3 * 10 + x * 3 * 5 * 25) = 63000 :=
by
  sorry

end NUMINAMATH_CALUDE_coin_container_total_l2722_272214


namespace NUMINAMATH_CALUDE_quadratic_root_sqrt5_minus_2_l2722_272270

theorem quadratic_root_sqrt5_minus_2 :
  ∃ (a b c : ℚ), a = 1 ∧ 
    (∀ x : ℝ, x^2 + b*x + c = 0 ↔ x = Real.sqrt 5 - 2 ∨ x = -(Real.sqrt 5) - 2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_root_sqrt5_minus_2_l2722_272270


namespace NUMINAMATH_CALUDE_line_through_points_l2722_272291

/-- Given a line y = ax + b passing through points (3, 2) and (7, 14), prove that 2a - b = 13 -/
theorem line_through_points (a b : ℝ) : 
  (2 : ℝ) = a * 3 + b → 
  (14 : ℝ) = a * 7 + b → 
  2 * a - b = 13 := by
  sorry

end NUMINAMATH_CALUDE_line_through_points_l2722_272291


namespace NUMINAMATH_CALUDE_base3_even_iff_sum_even_base10_multiple_of_7_iff_sum_congruent_l2722_272267

/- Define a function to convert a list of digits to a number in base 3 -/
def toBase3 (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 3 * acc + d) 0

/- Define a function to check if a number is even -/
def isEven (n : Nat) : Bool :=
  n % 2 = 0

/- Define a function to sum the digits of a list -/
def sumDigits (digits : List Nat) : Nat :=
  digits.sum

/- Theorem for base 3 even numbers -/
theorem base3_even_iff_sum_even (digits : List Nat) :
  isEven (toBase3 digits) ↔ isEven (sumDigits digits) := by
  sorry

/- Define a function to convert a list of digits to a number in base 10 -/
def toBase10 (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 10 * acc + d) 0

/- Define a function to check if a number is divisible by 7 -/
def isDivisibleBy7 (n : Nat) : Bool :=
  n % 7 = 0

/- Define a function to compute the sum of digits multiplied by powers of 10 mod 7 -/
def sumDigitsPowersOf10Mod7 (digits : List Nat) : Nat :=
  (List.range digits.length).zip digits
  |> List.foldl (fun acc (i, d) => (acc + d * (10^i % 7)) % 7) 0

/- Theorem for base 10 multiples of 7 -/
theorem base10_multiple_of_7_iff_sum_congruent (digits : List Nat) :
  isDivisibleBy7 (toBase10 digits) ↔ sumDigitsPowersOf10Mod7 digits = 0 := by
  sorry

end NUMINAMATH_CALUDE_base3_even_iff_sum_even_base10_multiple_of_7_iff_sum_congruent_l2722_272267


namespace NUMINAMATH_CALUDE_average_weight_increase_l2722_272268

theorem average_weight_increase (initial_average : ℝ) : 
  let initial_total := 5 * initial_average
  let final_total := initial_total - 40 + 90
  let final_average := final_total / 5
  final_average - initial_average = 10 := by
sorry

end NUMINAMATH_CALUDE_average_weight_increase_l2722_272268


namespace NUMINAMATH_CALUDE_part_one_part_two_l2722_272217

/-- The function f(x) = ax^2 - 3ax + 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 3 * a * x + 2

/-- Part 1: Given the solution set of f(x) > 0, prove a and b -/
theorem part_one (a b : ℝ) : 
  (∀ x, f a x > 0 ↔ (x < 1 ∨ x > b)) → a = 1 ∧ b = 2 :=
sorry

/-- Part 2: Given f(x) > 0 for all x, prove the range of a -/
theorem part_two (a : ℝ) :
  (∀ x, f a x > 0) → 0 ≤ a ∧ a < 8/9 :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2722_272217


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l2722_272237

theorem min_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + 2*b)/c + (2*a + c)/b + (b + 3*c)/a ≥ 6 * 12^(1/6) :=
by sorry

theorem min_value_achievable :
  ∃ (a b c : ℝ), 0 < a ∧ 0 < b ∧ 0 < c ∧
  (a + 2*b)/c + (2*a + c)/b + (b + 3*c)/a = 6 * 12^(1/6) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l2722_272237


namespace NUMINAMATH_CALUDE_grass_area_in_square_plot_l2722_272201

theorem grass_area_in_square_plot (perimeter : ℝ) (h_perimeter : perimeter = 40) :
  let side_length := perimeter / 4
  let square_area := side_length ^ 2
  let circle_radius := side_length / 2
  let circle_area := π * circle_radius ^ 2
  let grass_area := square_area - circle_area
  grass_area = 100 - 25 * π :=
by sorry

end NUMINAMATH_CALUDE_grass_area_in_square_plot_l2722_272201


namespace NUMINAMATH_CALUDE_polynomial_identity_l2722_272230

theorem polynomial_identity (P : ℝ → ℝ) : 
  (∀ x, P x - 3 * x = 5 * x^2 - 3 * x - 5) → 
  (∀ x, P x = 5 * x^2 - 5) := by
sorry

end NUMINAMATH_CALUDE_polynomial_identity_l2722_272230


namespace NUMINAMATH_CALUDE_square_congruence_mod_four_l2722_272283

theorem square_congruence_mod_four (n : ℤ) : (n^2) % 4 = 0 ∨ (n^2) % 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_congruence_mod_four_l2722_272283


namespace NUMINAMATH_CALUDE_phone_number_guess_probability_l2722_272261

theorem phone_number_guess_probability : 
  ∀ (total_digits : ℕ) (correct_digit : ℕ),
  total_digits = 10 →
  correct_digit < total_digits →
  (1 - 1 / total_digits) * (1 / (total_digits - 1)) = 1 / 10 :=
by
  sorry

end NUMINAMATH_CALUDE_phone_number_guess_probability_l2722_272261


namespace NUMINAMATH_CALUDE_point_translation_l2722_272204

def initial_point : ℝ × ℝ := (0, 1)
def downward_translation : ℝ := 2
def leftward_translation : ℝ := 4

theorem point_translation :
  (initial_point.1 - leftward_translation, initial_point.2 - downward_translation) = (-4, -1) := by
  sorry

end NUMINAMATH_CALUDE_point_translation_l2722_272204


namespace NUMINAMATH_CALUDE_restaurant_hiring_l2722_272271

theorem restaurant_hiring (initial_ratio_cooks initial_ratio_waiters new_ratio_cooks new_ratio_waiters num_cooks : ℕ) 
  (h1 : initial_ratio_cooks = 3)
  (h2 : initial_ratio_waiters = 10)
  (h3 : new_ratio_cooks = 3)
  (h4 : new_ratio_waiters = 14)
  (h5 : num_cooks = 9) :
  ∃ (initial_waiters hired_waiters : ℕ),
    initial_ratio_cooks * initial_waiters = initial_ratio_waiters * num_cooks ∧
    new_ratio_cooks * (initial_waiters + hired_waiters) = new_ratio_waiters * num_cooks ∧
    hired_waiters = 12 := by
  sorry


end NUMINAMATH_CALUDE_restaurant_hiring_l2722_272271


namespace NUMINAMATH_CALUDE_inverse_inequality_l2722_272289

theorem inverse_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a < b) :
  1 / a > 1 / b := by
  sorry

end NUMINAMATH_CALUDE_inverse_inequality_l2722_272289


namespace NUMINAMATH_CALUDE_pure_imaginary_implies_a_eq_three_l2722_272208

/-- A complex number z is pure imaginary if its real part is zero and its imaginary part is non-zero. -/
def IsPureImaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

/-- Given that z = (a^2 - 9) + (a + 3)i is a pure imaginary number, prove that a = 3. -/
theorem pure_imaginary_implies_a_eq_three (a : ℝ) 
    (h : IsPureImaginary ((a^2 - 9) + (a + 3)*I)) : a = 3 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_implies_a_eq_three_l2722_272208


namespace NUMINAMATH_CALUDE_hyperbola_parameters_l2722_272259

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- The shortest distance from a point on the hyperbola to one of its foci -/
def shortest_focal_distance (h : Hyperbola) : ℝ := 2

/-- A point on the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point lies on an asymptote of the hyperbola -/
def on_asymptote (h : Hyperbola) (p : Point) : Prop :=
  p.y / p.x = h.b / h.a ∨ p.y / p.x = -h.b / h.a

/-- The given point P -/
def P : Point := ⟨3, 4⟩

theorem hyperbola_parameters (h : Hyperbola) 
  (h_focal : shortest_focal_distance h = 2)
  (h_asymptote : on_asymptote h P) :
  h.a = 3 ∧ h.b = 4 := by sorry

end NUMINAMATH_CALUDE_hyperbola_parameters_l2722_272259


namespace NUMINAMATH_CALUDE_smallest_solution_quartic_equation_l2722_272203

theorem smallest_solution_quartic_equation :
  ∃ (x : ℝ), x^4 - 40*x^2 + 400 = 0 ∧
  x = -2 * Real.sqrt 5 ∧
  ∀ (y : ℝ), y^4 - 40*y^2 + 400 = 0 → y ≥ x :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_quartic_equation_l2722_272203


namespace NUMINAMATH_CALUDE_sphere_volume_circumscribing_cube_l2722_272231

/-- The volume of a sphere that circumscribes a cube with side length 2 is 4√3π. -/
theorem sphere_volume_circumscribing_cube (cube_side : ℝ) (sphere_volume : ℝ) : 
  cube_side = 2 →
  sphere_volume = (4 / 3) * Real.pi * (Real.sqrt 3 * cube_side / 2)^3 →
  sphere_volume = 4 * Real.sqrt 3 * Real.pi :=
by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_circumscribing_cube_l2722_272231


namespace NUMINAMATH_CALUDE_son_age_is_eighteen_l2722_272253

/-- Represents the ages of a father and son -/
structure FatherSonAges where
  fatherAge : ℕ
  sonAge : ℕ

/-- The condition that the father is 20 years older than the son -/
def ageDifference (ages : FatherSonAges) : Prop :=
  ages.fatherAge = ages.sonAge + 20

/-- The condition that in two years, the father's age will be twice the son's age -/
def futureAgeRelation (ages : FatherSonAges) : Prop :=
  ages.fatherAge + 2 = 2 * (ages.sonAge + 2)

/-- Theorem stating that given the conditions, the son's present age is 18 -/
theorem son_age_is_eighteen (ages : FatherSonAges) 
  (h1 : ageDifference ages) (h2 : futureAgeRelation ages) : ages.sonAge = 18 := by
  sorry

end NUMINAMATH_CALUDE_son_age_is_eighteen_l2722_272253


namespace NUMINAMATH_CALUDE_smallest_with_eight_factors_l2722_272250

/-- The number of distinct positive factors of a positive integer -/
def num_factors (n : ℕ+) : ℕ := sorry

/-- The set of distinct positive factors of a positive integer -/
def factors (n : ℕ+) : Finset ℕ := sorry

theorem smallest_with_eight_factors : 
  (∀ m : ℕ+, m < 24 → num_factors m ≠ 8) ∧ 
  num_factors 24 = 8 ∧
  factors 24 = {1, 2, 3, 4, 6, 8, 12, 24} := by sorry

end NUMINAMATH_CALUDE_smallest_with_eight_factors_l2722_272250


namespace NUMINAMATH_CALUDE_angle_B_is_60_l2722_272292

-- Define a scalene triangle ABC
structure ScaleneTriangle where
  A : Real
  B : Real
  C : Real
  scalene : A ≠ B ∧ B ≠ C ∧ C ≠ A
  sum_180 : A + B + C = 180

-- Define the specific triangle with given angle relationships
def SpecificTriangle (t : ScaleneTriangle) : Prop :=
  t.C = 3 * t.A ∧ t.B = 2 * t.A

-- Theorem statement
theorem angle_B_is_60 (t : ScaleneTriangle) (h : SpecificTriangle t) : t.B = 60 := by
  sorry

end NUMINAMATH_CALUDE_angle_B_is_60_l2722_272292


namespace NUMINAMATH_CALUDE_systematic_sample_sum_l2722_272264

/-- Systematic sampling function that returns the nth element in a sample of size k from a population of size n -/
def systematicSample (n k : ℕ) (i : ℕ) : ℕ :=
  i * (n / k) + 1

theorem systematic_sample_sum (a b : ℕ) :
  systematicSample 60 5 0 = 4 ∧
  systematicSample 60 5 1 = a ∧
  systematicSample 60 5 2 = 28 ∧
  systematicSample 60 5 3 = b ∧
  systematicSample 60 5 4 = 52 →
  a + b = 56 := by
sorry

end NUMINAMATH_CALUDE_systematic_sample_sum_l2722_272264


namespace NUMINAMATH_CALUDE_circle_center_proof_l2722_272213

/-- A line in the 2D plane represented by ax + by = c --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A circle in the 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Check if a point lies on a line --/
def pointOnLine (p : ℝ × ℝ) (l : Line) : Prop :=
  l.a * p.1 + l.b * p.2 = l.c

/-- Check if a circle is tangent to a line --/
def circleTangentToLine (c : Circle) (l : Line) : Prop :=
  abs (l.a * c.center.1 + l.b * c.center.2 - l.c) = c.radius * (l.a^2 + l.b^2).sqrt

theorem circle_center_proof :
  let line1 : Line := { a := 5, b := -2, c := 40 }
  let line2 : Line := { a := 5, b := -2, c := 10 }
  let line3 : Line := { a := 3, b := -4, c := 0 }
  let center : ℝ × ℝ := (50/7, 75/14)
  ∃ (r : ℝ), 
    let c : Circle := { center := center, radius := r }
    circleTangentToLine c line1 ∧ 
    circleTangentToLine c line2 ∧ 
    pointOnLine center line3 := by
  sorry


end NUMINAMATH_CALUDE_circle_center_proof_l2722_272213


namespace NUMINAMATH_CALUDE_right_triangle_area_l2722_272229

/-- The area of a right triangle with hypotenuse 12 inches and one angle 30° is 18√3 square inches -/
theorem right_triangle_area (h : ℝ) (θ : ℝ) (area : ℝ) : 
  h = 12 →  -- hypotenuse is 12 inches
  θ = 30 * π / 180 →  -- one angle is 30°
  area = h * h * Real.sin θ * Real.cos θ / 2 →  -- area formula for right triangle
  area = 18 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_area_l2722_272229


namespace NUMINAMATH_CALUDE_claire_balloons_given_away_l2722_272244

/-- The number of balloons Claire gave away during the fair -/
def balloons_given_away (initial_balloons : ℕ) (floated_away : ℕ) (grabbed_from_coworker : ℕ) (final_balloons : ℕ) : ℕ :=
  initial_balloons - floated_away + grabbed_from_coworker - final_balloons

/-- Theorem stating the number of balloons Claire gave away during the fair -/
theorem claire_balloons_given_away :
  balloons_given_away 50 12 11 39 = 10 := by
  sorry

#eval balloons_given_away 50 12 11 39

end NUMINAMATH_CALUDE_claire_balloons_given_away_l2722_272244


namespace NUMINAMATH_CALUDE_qr_distance_l2722_272277

/-- Right triangle DEF with given side lengths -/
structure RightTriangle where
  DE : ℝ
  EF : ℝ
  DF : ℝ
  right_angle : DE^2 + EF^2 = DF^2

/-- Circle centered at Q tangent to DE at D and passing through F -/
structure CircleQ where
  Q : ℝ × ℝ
  tangent_DE : True  -- Simplified representation of tangency
  passes_through_F : True  -- Simplified representation of passing through F

/-- Circle centered at R tangent to EF at E and passing through F -/
structure CircleR where
  R : ℝ × ℝ
  tangent_EF : True  -- Simplified representation of tangency
  passes_through_F : True  -- Simplified representation of passing through F

/-- The main theorem statement -/
theorem qr_distance (t : RightTriangle) (cq : CircleQ) (cr : CircleR) 
  (h1 : t.DE = 5) (h2 : t.EF = 12) (h3 : t.DF = 13) :
  Real.sqrt ((cq.Q.1 - cr.R.1)^2 + (cq.Q.2 - cr.R.2)^2) = 13.54 := by
  sorry

end NUMINAMATH_CALUDE_qr_distance_l2722_272277


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainders_l2722_272254

theorem smallest_integer_with_remainders : ∃ n : ℕ, 
  n > 1 ∧
  n % 3 = 2 ∧
  n % 7 = 2 ∧
  n % 8 = 2 ∧
  (∀ m : ℕ, m > 1 ∧ m % 3 = 2 ∧ m % 7 = 2 ∧ m % 8 = 2 → n ≤ m) ∧
  n = 170 ∧
  131 ≤ n ∧ n ≤ 170 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainders_l2722_272254


namespace NUMINAMATH_CALUDE_number_categorization_l2722_272233

def given_numbers : List ℚ := [-18, -3/5, 0, 2023, -22/7, -0.142857, 95/100]

def is_positive (x : ℚ) : Prop := x > 0
def is_negative (x : ℚ) : Prop := x < 0
def is_integer (x : ℚ) : Prop := ∃ n : ℤ, x = n
def is_fraction (x : ℚ) : Prop := ∃ a b : ℤ, b ≠ 0 ∧ x = a / b

def positive_set : Set ℚ := {x | is_positive x}
def negative_set : Set ℚ := {x | is_negative x}
def integer_set : Set ℚ := {x | is_integer x}
def fraction_set : Set ℚ := {x | is_fraction x}

theorem number_categorization :
  (positive_set ∩ given_numbers.toFinset = {2023, 95/100}) ∧
  (negative_set ∩ given_numbers.toFinset = {-18, -3/5, -22/7, -0.142857}) ∧
  (integer_set ∩ given_numbers.toFinset = {-18, 0, 2023}) ∧
  (fraction_set ∩ given_numbers.toFinset = {-3/5, -22/7, -0.142857, 95/100}) :=
by sorry

end NUMINAMATH_CALUDE_number_categorization_l2722_272233


namespace NUMINAMATH_CALUDE_count_three_digit_divisible_by_nine_l2722_272287

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem count_three_digit_divisible_by_nine :
  let min_num : ℕ := 108
  let max_num : ℕ := 999
  let common_diff : ℕ := 9
  (∀ n, is_three_digit n ∧ n % 9 = 0 → min_num ≤ n ∧ n ≤ max_num) →
  (∀ n, min_num ≤ n ∧ n ≤ max_num ∧ n % 9 = 0 → is_three_digit n) →
  (∀ n m, min_num ≤ n ∧ n < m ∧ m ≤ max_num ∧ n % 9 = 0 ∧ m % 9 = 0 → m - n = common_diff) →
  (Finset.filter (λ n => n % 9 = 0) (Finset.range (max_num - min_num + 1))).card + 1 = 100 :=
by sorry

end NUMINAMATH_CALUDE_count_three_digit_divisible_by_nine_l2722_272287


namespace NUMINAMATH_CALUDE_athlete_stop_point_l2722_272272

/-- Represents a rectangular square with length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Represents a point on the perimeter of a rectangle -/
structure PerimeterPoint where
  distance : ℝ  -- Distance from a chosen starting point

/-- The athlete's run around the rectangular square -/
def athleteRun (rect : Rectangle) (start : PerimeterPoint) (distance : ℝ) : PerimeterPoint :=
  sorry

theorem athlete_stop_point (rect : Rectangle) (start : PerimeterPoint) :
  let totalDistance : ℝ := 15500  -- 15.5 km in meters
  rect.length = 900 ∧ rect.width = 600 ∧ start.distance = 550 →
  (athleteRun rect start totalDistance).distance = 150 :=
sorry

end NUMINAMATH_CALUDE_athlete_stop_point_l2722_272272


namespace NUMINAMATH_CALUDE_no_odd_white_columns_exists_odd_black_columns_l2722_272212

/-- Represents a 3x3x3 cube composed of white and black unit cubes -/
structure Cube :=
  (white_count : Nat)
  (black_count : Nat)
  (total_count : Nat)
  (is_valid : white_count + black_count = total_count ∧ total_count = 27)

/-- Represents a column in the cube -/
structure Column :=
  (white_count : Nat)
  (black_count : Nat)
  (is_valid : white_count + black_count = 3)

/-- Checks if a number is odd -/
def is_odd (n : Nat) : Prop := n % 2 = 1

/-- Theorem: It is impossible for each column to contain an odd number of white cubes -/
theorem no_odd_white_columns (c : Cube) (h : c.white_count = 14 ∧ c.black_count = 13) :
  ¬ (∀ col : Column, is_odd col.white_count) :=
sorry

/-- Theorem: It is possible for each column to contain an odd number of black cubes -/
theorem exists_odd_black_columns (c : Cube) (h : c.white_count = 14 ∧ c.black_count = 13) :
  ∃ (arrangement : List Column), (∀ col ∈ arrangement, is_odd col.black_count) ∧ 
    arrangement.length = 27 ∧ (arrangement.map Column.black_count).sum = 13 :=
sorry

end NUMINAMATH_CALUDE_no_odd_white_columns_exists_odd_black_columns_l2722_272212


namespace NUMINAMATH_CALUDE_middle_part_of_proportional_division_l2722_272273

theorem middle_part_of_proportional_division (total : ℕ) (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  total = 120 ∧ a = 3 ∧ b = 5 ∧ c = 7 →
  ∃ (x : ℚ), x > 0 ∧ a * x + b * x + c * x = total ∧ (∃ (n : ℕ), a * x = n ∨ b * x = n ∨ c * x = n) →
  b * x = 40 := by
  sorry

end NUMINAMATH_CALUDE_middle_part_of_proportional_division_l2722_272273


namespace NUMINAMATH_CALUDE_twice_x_greater_than_five_l2722_272223

theorem twice_x_greater_than_five (x : ℝ) : (2 * x > 5) ↔ (2 * x > 5) := by sorry

end NUMINAMATH_CALUDE_twice_x_greater_than_five_l2722_272223


namespace NUMINAMATH_CALUDE_walking_speed_calculation_l2722_272298

/-- Proves that the walking speed is 4 km/hr given the conditions of the problem -/
theorem walking_speed_calculation (run_speed : ℝ) (total_distance : ℝ) (total_time : ℝ)
  (h1 : run_speed = 8)
  (h2 : total_distance = 20)
  (h3 : total_time = 3.75) :
  ∃ (walk_speed : ℝ),
    walk_speed = 4 ∧
    (total_distance / 2) / walk_speed + (total_distance / 2) / run_speed = total_time :=
by sorry

end NUMINAMATH_CALUDE_walking_speed_calculation_l2722_272298


namespace NUMINAMATH_CALUDE_diagonals_perpendicular_l2722_272279

/-- Given four points A, B, C, and D in a 2D plane, prove that the diagonals of the quadrilateral ABCD are perpendicular. -/
theorem diagonals_perpendicular (A B C D : ℝ × ℝ) 
  (hA : A = (-2, 3))
  (hB : B = (2, 6))
  (hC : C = (6, -1))
  (hD : D = (-3, -4)) : 
  (C.1 - A.1) * (D.1 - B.1) + (C.2 - A.2) * (D.2 - B.2) = 0 := by
  sorry

#check diagonals_perpendicular

end NUMINAMATH_CALUDE_diagonals_perpendicular_l2722_272279


namespace NUMINAMATH_CALUDE_point_P_coordinates_l2722_272252

def M : ℝ × ℝ := (2, 2)
def N : ℝ × ℝ := (5, -2)

def is_right_angle (A B C : ℝ × ℝ) : Prop :=
  (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0

theorem point_P_coordinates :
  ∀ x : ℝ,
    let P : ℝ × ℝ := (x, 0)
    is_right_angle M P N → x = 1 ∨ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_point_P_coordinates_l2722_272252


namespace NUMINAMATH_CALUDE_range_of_a_l2722_272240

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 - 2*x + 5 ≥ a^2 - 3*a) ↔ -1 ≤ a ∧ a ≤ 4 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l2722_272240


namespace NUMINAMATH_CALUDE_hyperbola_equation_correct_l2722_272275

/-- A hyperbola with foci on the X-axis, distance between vertices of 6, and asymptote equations y = ± 3/2 x -/
structure Hyperbola where
  foci_on_x_axis : Bool
  vertex_distance : ℝ
  asymptote_slope : ℝ

/-- The equation of a hyperbola in the form (x²/a² - y²/b² = 1) -/
def hyperbola_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / 9 - 4 * y^2 / 81 = 1

/-- Theorem stating that the given hyperbola has the specified equation -/
theorem hyperbola_equation_correct (h : Hyperbola) 
  (h_foci : h.foci_on_x_axis = true)
  (h_vertex : h.vertex_distance = 6)
  (h_asymptote : h.asymptote_slope = 3/2) :
  ∀ x y : ℝ, hyperbola_equation h x y ↔ x^2 / 9 - 4 * y^2 / 81 = 1 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_correct_l2722_272275


namespace NUMINAMATH_CALUDE_inequality_proof_l2722_272274

theorem inequality_proof (x : ℝ) (n : ℕ) (h1 : 0 ≤ x) (h2 : x ≤ 1) (h3 : 0 < n) :
  (1 + x)^n ≥ (1 - x)^n + 2 * n * x * (1 - x^2)^((n - 1) / 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2722_272274


namespace NUMINAMATH_CALUDE_function_increasing_iff_a_nonpositive_l2722_272241

/-- The function f(x) = (1/3)x^3 - ax is increasing on ℝ if and only if a ≤ 0 -/
theorem function_increasing_iff_a_nonpositive (a : ℝ) :
  (∀ x : ℝ, HasDerivAt (fun x => (1/3) * x^3 - a * x) (x^2 - a) x) →
  (∀ x y : ℝ, x < y → ((1/3) * x^3 - a * x) < ((1/3) * y^3 - a * y)) ↔
  a ≤ 0 := by sorry

end NUMINAMATH_CALUDE_function_increasing_iff_a_nonpositive_l2722_272241


namespace NUMINAMATH_CALUDE_prove_annes_cleaning_time_l2722_272210

/-- Represents the time it takes Anne to clean the house alone -/
def annes_cleaning_time : ℝ := 12

/-- Represents Bruce's cleaning rate (houses per hour) -/
noncomputable def bruce_rate : ℝ := sorry

/-- Represents Anne's cleaning rate (houses per hour) -/
noncomputable def anne_rate : ℝ := sorry

theorem prove_annes_cleaning_time :
  -- Bruce and Anne can clean the house in 4 hours together
  (bruce_rate + anne_rate) * 4 = 1 →
  -- If Anne's speed is doubled, they can clean the house in 3 hours
  (bruce_rate + 2 * anne_rate) * 3 = 1 →
  -- Then Anne's individual cleaning time is 12 hours
  annes_cleaning_time = 1 / anne_rate :=
by sorry

end NUMINAMATH_CALUDE_prove_annes_cleaning_time_l2722_272210


namespace NUMINAMATH_CALUDE_total_turnips_after_selling_l2722_272262

/-- The total number of turnips after selling some -/
def totalTurnipsAfterSelling (melanieTurnips bennyTurnips sarahTurnips davidTurnips melanieSold davidSold : ℕ) : ℕ :=
  (melanieTurnips - melanieSold) + bennyTurnips + sarahTurnips + (davidTurnips - davidSold)

/-- Theorem stating the total number of turnips after selling -/
theorem total_turnips_after_selling :
  totalTurnipsAfterSelling 139 113 195 87 32 15 = 487 := by
  sorry

#eval totalTurnipsAfterSelling 139 113 195 87 32 15

end NUMINAMATH_CALUDE_total_turnips_after_selling_l2722_272262


namespace NUMINAMATH_CALUDE_erin_has_90_dollars_l2722_272282

/-- The amount of money Erin has after emptying all machines in her launderette -/
def erins_money_after_emptying (quarters_per_machine : ℕ) (dimes_per_machine : ℕ) (num_machines : ℕ) : ℚ :=
  (quarters_per_machine * (25 : ℚ) / 100 + dimes_per_machine * (10 : ℚ) / 100) * num_machines

/-- Theorem stating that Erin will have $90.00 after emptying all machines -/
theorem erin_has_90_dollars :
  erins_money_after_emptying 80 100 3 = 90 :=
by sorry

end NUMINAMATH_CALUDE_erin_has_90_dollars_l2722_272282


namespace NUMINAMATH_CALUDE_counterexample_exists_l2722_272236

theorem counterexample_exists : ∃ (a b c : ℤ), a > b ∧ b > c ∧ a + b ≤ c := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l2722_272236


namespace NUMINAMATH_CALUDE_min_perimeter_triangle_ABC_l2722_272297

/-- Triangle ABC with integer side lengths, BD angle bisector of ∠ABC, AD = 4, DC = 6, D on AC -/
structure TriangleABC where
  AB : ℕ
  BC : ℕ
  AC : ℕ
  AD : ℕ
  DC : ℕ
  hAD : AD = 4
  hDC : DC = 6
  hAC : AC = AD + DC
  hAngleBisector : AB * DC = BC * AD

/-- The minimum possible perimeter of triangle ABC is 25 -/
theorem min_perimeter_triangle_ABC (t : TriangleABC) : 
  (∀ t' : TriangleABC, t'.AB + t'.BC + t'.AC ≥ t.AB + t.BC + t.AC) → 
  t.AB + t.BC + t.AC = 25 := by
  sorry

end NUMINAMATH_CALUDE_min_perimeter_triangle_ABC_l2722_272297


namespace NUMINAMATH_CALUDE_smallest_n_for_inequality_l2722_272243

theorem smallest_n_for_inequality : ∃ (n : ℕ), n = 2 ∧ 
  (∀ (k : ℕ), k < n → (10 : ℝ) ^ (2 ^ (k + 1)) < 1000) ∧
  (10 : ℝ) ^ (2 ^ (n + 1)) ≥ 1000 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_inequality_l2722_272243
