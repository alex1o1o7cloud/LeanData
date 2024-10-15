import Mathlib

namespace NUMINAMATH_CALUDE_fibonacci_divisibility_l3973_397351

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Divisibility relation -/
def divides (a b : ℕ) : Prop := ∃ k, b = a * k

theorem fibonacci_divisibility (m n : ℕ) (h : m > 2) :
  divides (fib m) (fib n) ↔ divides m n := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_divisibility_l3973_397351


namespace NUMINAMATH_CALUDE_stating_rectangular_box_area_diagonal_product_l3973_397344

/-- Represents a rectangular box with dimensions a, b, and c -/
structure RectangularBox (a b c : ℝ) where
  bottom_area : ℝ := a * b
  side_area : ℝ := b * c
  front_area : ℝ := c * a
  diagonal_squared : ℝ := a^2 + b^2 + c^2

/-- 
Theorem stating that for a rectangular box, the product of its face areas 
multiplied by the square of its diagonal equals a²b²c² · (a² + b² + c²)
-/
theorem rectangular_box_area_diagonal_product 
  (a b c : ℝ) (box : RectangularBox a b c) : 
  box.bottom_area * box.side_area * box.front_area * box.diagonal_squared = 
  a^2 * b^2 * c^2 * (a^2 + b^2 + c^2) := by
  sorry

end NUMINAMATH_CALUDE_stating_rectangular_box_area_diagonal_product_l3973_397344


namespace NUMINAMATH_CALUDE_john_laptop_savings_l3973_397394

def octal_to_decimal (n : ℕ) : ℕ :=
  (n / 1000) * 512 + ((n / 100) % 10) * 64 + ((n / 10) % 10) * 8 + (n % 10)

theorem john_laptop_savings :
  octal_to_decimal 5555 - 1500 = 1425 := by
  sorry

end NUMINAMATH_CALUDE_john_laptop_savings_l3973_397394


namespace NUMINAMATH_CALUDE_sum_of_squares_coefficients_l3973_397361

/-- The sum of squares of coefficients in the simplified form of 6(x³-2x²+x-3)-5(x⁴-4x²+3x+2) is 990 -/
theorem sum_of_squares_coefficients : 
  let expression := fun x : ℝ => 6 * (x^3 - 2*x^2 + x - 3) - 5 * (x^4 - 4*x^2 + 3*x + 2)
  let coefficients := [-5, 6, 8, -9, -28]
  (coefficients.map (fun c => c^2)).sum = 990 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_coefficients_l3973_397361


namespace NUMINAMATH_CALUDE_max_z_value_l3973_397385

theorem max_z_value (x y z : ℕ) : 
  7 < x → x < 9 → 9 < y → y < 15 → 
  0 < z → 
  Nat.Prime x → Nat.Prime y → Nat.Prime z →
  (y - x) % z = 0 →
  z ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_max_z_value_l3973_397385


namespace NUMINAMATH_CALUDE_z_coordinate_is_zero_l3973_397370

/-- A line passing through two points in 3D space -/
structure Line3D where
  point1 : ℝ × ℝ × ℝ
  point2 : ℝ × ℝ × ℝ

/-- A point on a line with a specific x-coordinate -/
def point_on_line (l : Line3D) (x : ℝ) : ℝ × ℝ × ℝ := sorry

theorem z_coordinate_is_zero : 
  let l : Line3D := { point1 := (1, 3, 2), point2 := (4, 2, -1) }
  let p := point_on_line l 3
  p.2.2 = 0 := by sorry

end NUMINAMATH_CALUDE_z_coordinate_is_zero_l3973_397370


namespace NUMINAMATH_CALUDE_arithmetic_geometric_progression_l3973_397312

theorem arithmetic_geometric_progression (a b : ℝ) : 
  (1 = (a + b) / 2) →  -- arithmetic progression condition
  (1 = |a * b|) →      -- geometric progression condition
  ((a = 1 ∧ b = 1) ∨ 
   (a = 1 + Real.sqrt 2 ∧ b = 1 - Real.sqrt 2) ∨ 
   (a = 1 - Real.sqrt 2 ∧ b = 1 + Real.sqrt 2)) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_progression_l3973_397312


namespace NUMINAMATH_CALUDE_minkowski_sum_convex_l3973_397316

-- Define a type for points in a 2D space
variable {α : Type*} [AddCommGroup α] [Module ℝ α]

-- Define a convex figure as a set of points
def ConvexFigure (S : Set α) : Prop :=
  ∀ (x y : α), x ∈ S → y ∈ S → ∀ (t : ℝ), 0 ≤ t ∧ t ≤ 1 →
    (1 - t) • x + t • y ∈ S

-- Define Minkowski sum of two sets
def MinkowskiSum (S T : Set α) : Set α :=
  {z | ∃ (x y : α), x ∈ S ∧ y ∈ T ∧ z = x + y}

-- Theorem statement
theorem minkowski_sum_convex
  (Φ₁ Φ₂ : Set α) (h1 : ConvexFigure Φ₁) (h2 : ConvexFigure Φ₂) :
  ConvexFigure (MinkowskiSum Φ₁ Φ₂) :=
sorry

end NUMINAMATH_CALUDE_minkowski_sum_convex_l3973_397316


namespace NUMINAMATH_CALUDE_square_product_eq_sum_squares_solution_l3973_397345

theorem square_product_eq_sum_squares_solution (a b : ℤ) :
  a^2 * b^2 = a^2 + b^2 → a = 0 ∧ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_square_product_eq_sum_squares_solution_l3973_397345


namespace NUMINAMATH_CALUDE_fourth_largest_common_divisor_l3973_397341

def is_divisor (d n : ℕ) : Prop := n % d = 0

def common_divisors (a b : ℕ) : Set ℕ :=
  {d : ℕ | is_divisor d a ∧ is_divisor d b}

theorem fourth_largest_common_divisor :
  let cd := common_divisors 72 120
  ∃ (l : List ℕ), (∀ x ∈ cd, x ∈ l) ∧
                  (∀ x ∈ l, x ∈ cd) ∧
                  l.Sorted (· > ·) ∧
                  l.get? 3 = some 6 :=
sorry

end NUMINAMATH_CALUDE_fourth_largest_common_divisor_l3973_397341


namespace NUMINAMATH_CALUDE_inequality_proof_l3973_397318

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a^2 = b^2 + c^2) :
  a^3 + b^3 + c^3 ≥ (2 * Real.sqrt 2 + 1) / 7 * (a^2 * (b + c) + b^2 * (c + a) + c^2 * (a + b)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3973_397318


namespace NUMINAMATH_CALUDE_probability_of_drawing_two_l3973_397365

/-- Represents a card with a number -/
structure Card where
  number : ℕ

/-- Represents the set of cards -/
def cardSet : Finset Card := sorry

/-- The total number of cards -/
def totalCards : ℕ := 5

/-- The number of cards with the number 2 -/
def cardsWithTwo : ℕ := 2

/-- The probability of drawing a card with the number 2 -/
def probabilityOfTwo : ℚ := cardsWithTwo / totalCards

theorem probability_of_drawing_two :
  probabilityOfTwo = 2 / 5 := by sorry

end NUMINAMATH_CALUDE_probability_of_drawing_two_l3973_397365


namespace NUMINAMATH_CALUDE_complex_fraction_calculation_l3973_397309

theorem complex_fraction_calculation : 
  (2 + 5/8 - 2/3 * (2 + 5/14)) / ((3 + 1/12 + 4.375) / (19 + 8/9)) = 2 + 17/21 := by sorry

end NUMINAMATH_CALUDE_complex_fraction_calculation_l3973_397309


namespace NUMINAMATH_CALUDE_min_shapes_for_square_l3973_397340

/-- The area of each shape in square units -/
def shape_area : ℕ := 3

/-- The side length of the smallest possible square that can be formed -/
def square_side : ℕ := 6

/-- The theorem stating the minimum number of shapes required -/
theorem min_shapes_for_square :
  let total_area : ℕ := square_side * square_side
  let num_shapes : ℕ := total_area / shape_area
  (∀ n : ℕ, n < num_shapes → n * shape_area < square_side * square_side) ∧
  (num_shapes * shape_area = square_side * square_side) ∧
  (∃ (arrangement : ℕ → ℕ → ℕ),
    (∀ i j : ℕ, i < square_side ∧ j < square_side →
      ∃ k : ℕ, k < num_shapes ∧ arrangement i j = k)) :=
by sorry

end NUMINAMATH_CALUDE_min_shapes_for_square_l3973_397340


namespace NUMINAMATH_CALUDE_sexagesimal_cubes_correct_l3973_397375

/-- Converts a sexagesimal number to decimal -/
def sexagesimal_to_decimal (whole : ℕ) (frac : ℕ) : ℕ :=
  whole * 60 + frac

/-- Checks if a number is a perfect cube -/
def is_perfect_cube (n : ℕ) : Prop :=
  ∃ m : ℕ, m^3 = n

/-- Represents a sexagesimal number as a pair of natural numbers -/
structure Sexagesimal :=
  (whole : ℕ)
  (frac : ℕ)

/-- Theorem stating that the sexagesimal representation of cubes is correct for numbers from 1 to 32 -/
theorem sexagesimal_cubes_correct :
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 32 →
    ∃ s : Sexagesimal, 
      sexagesimal_to_decimal s.whole s.frac = n^3 ∧
      is_perfect_cube (sexagesimal_to_decimal s.whole s.frac) :=
by sorry

end NUMINAMATH_CALUDE_sexagesimal_cubes_correct_l3973_397375


namespace NUMINAMATH_CALUDE_money_sharing_l3973_397393

theorem money_sharing (total : ℚ) (per_person : ℚ) (num_people : ℕ) : 
  total = 3.75 ∧ per_person = 1.25 → num_people = 3 ∧ total = num_people * per_person :=
by sorry

end NUMINAMATH_CALUDE_money_sharing_l3973_397393


namespace NUMINAMATH_CALUDE_arithmetic_and_geometric_is_nonzero_constant_l3973_397377

/-- A sequence that is both arithmetic and geometric is non-zero constant -/
theorem arithmetic_and_geometric_is_nonzero_constant (a : ℕ → ℝ) : 
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d) →  -- arithmetic sequence condition
  (∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n) →  -- geometric sequence condition
  (∃ c : ℝ, c ≠ 0 ∧ ∀ n : ℕ, a n = c) :=      -- non-zero constant sequence
by sorry

end NUMINAMATH_CALUDE_arithmetic_and_geometric_is_nonzero_constant_l3973_397377


namespace NUMINAMATH_CALUDE_delta_triple_72_l3973_397356

-- Define the Δ function
def Δ (N : ℝ) : ℝ := 0.4 * N + 2

-- Theorem statement
theorem delta_triple_72 : Δ (Δ (Δ 72)) = 7.728 := by
  sorry

end NUMINAMATH_CALUDE_delta_triple_72_l3973_397356


namespace NUMINAMATH_CALUDE_inverse_composition_equals_six_l3973_397354

-- Define the function f
def f : ℕ → ℕ
| 1 => 4
| 2 => 6
| 3 => 2
| 4 => 1
| 5 => 5
| 6 => 3
| _ => 0  -- Default case for other inputs

-- Define the inverse function f⁻¹
def f_inv : ℕ → ℕ
| 1 => 4
| 2 => 3
| 3 => 6
| 4 => 1
| 5 => 5
| 6 => 2
| _ => 0  -- Default case for other inputs

-- State the theorem
theorem inverse_composition_equals_six :
  f_inv (f_inv (f_inv 6)) = 6 := by sorry

end NUMINAMATH_CALUDE_inverse_composition_equals_six_l3973_397354


namespace NUMINAMATH_CALUDE_probability_zero_l3973_397366

def P (x : ℝ) : ℝ := x^3 - 4*x^2 - x + 20

theorem probability_zero :
  ∀ x : ℝ, 3 ≤ x ∧ x ≤ 10 →
  (⌊(P x)^(1/3)⌋ : ℝ) ≠ (P ⌊x⌋)^(1/3) :=
by sorry

end NUMINAMATH_CALUDE_probability_zero_l3973_397366


namespace NUMINAMATH_CALUDE_celebrity_match_probability_l3973_397355

/-- The number of celebrities -/
def n : ℕ := 4

/-- The probability of correctly matching all celebrities with their pictures and hobbies -/
def correct_match_probability : ℚ := 1 / (n.factorial * n.factorial)

/-- Theorem: The probability of correctly matching all celebrities with their pictures and hobbies is 1/576 -/
theorem celebrity_match_probability :
  correct_match_probability = 1 / 576 := by sorry

end NUMINAMATH_CALUDE_celebrity_match_probability_l3973_397355


namespace NUMINAMATH_CALUDE_system_solution_pairs_l3973_397326

theorem system_solution_pairs :
  ∀ x y : ℝ, x > 0 ∧ y > 0 →
  (x - 3 * Real.sqrt (x * y) - 2 * Real.sqrt (x / y) = 0 ∧
   x^2 * y^2 + x^4 = 82) →
  ((x = 3 ∧ y = 1/3) ∨ (x = Real.rpow 66 (1/4) ∧ y = 4 / Real.rpow 66 (1/4))) := by
sorry

end NUMINAMATH_CALUDE_system_solution_pairs_l3973_397326


namespace NUMINAMATH_CALUDE_fifth_root_unity_sum_l3973_397314

theorem fifth_root_unity_sum (x : ℂ) : x^5 = 1 → 1 + x^4 + x^8 + x^12 + x^16 = 0 := by
  sorry

end NUMINAMATH_CALUDE_fifth_root_unity_sum_l3973_397314


namespace NUMINAMATH_CALUDE_little_twelve_games_l3973_397321

/-- Represents a basketball conference with two divisions -/
structure BasketballConference :=
  (teams_per_division : ℕ)
  (divisions : ℕ)
  (intra_division_games : ℕ)
  (inter_division_games : ℕ)

/-- Calculates the total number of games in the conference -/
def total_games (conf : BasketballConference) : ℕ :=
  let total_teams := conf.teams_per_division * conf.divisions
  let games_per_team := (conf.teams_per_division - 1) * conf.intra_division_games + 
                        conf.teams_per_division * conf.inter_division_games
  total_teams * games_per_team / 2

/-- Theorem stating that the Little Twelve Basketball Conference schedules 96 games -/
theorem little_twelve_games : 
  ∀ (conf : BasketballConference), 
    conf.teams_per_division = 6 ∧ 
    conf.divisions = 2 ∧ 
    conf.intra_division_games = 2 ∧ 
    conf.inter_division_games = 1 → 
    total_games conf = 96 := by
  sorry

end NUMINAMATH_CALUDE_little_twelve_games_l3973_397321


namespace NUMINAMATH_CALUDE_product_of_roots_cubic_l3973_397376

theorem product_of_roots_cubic (a b c : ℂ) : 
  (3 * a^3 - 4 * a^2 + 9 * a - 18 = 0) ∧ 
  (3 * b^3 - 4 * b^2 + 9 * b - 18 = 0) ∧ 
  (3 * c^3 - 4 * c^2 + 9 * c - 18 = 0) → 
  a * b * c = 6 := by
sorry

end NUMINAMATH_CALUDE_product_of_roots_cubic_l3973_397376


namespace NUMINAMATH_CALUDE_population_size_l3973_397358

/-- Given a population with specified birth and death rates, and a net growth rate,
    prove that the initial population size is 3000. -/
theorem population_size (birth_rate death_rate net_growth_rate : ℝ) 
  (h1 : birth_rate = 52)
  (h2 : death_rate = 16)
  (h3 : net_growth_rate = 0.012)
  (h4 : birth_rate - death_rate = net_growth_rate * 100) : 
  (birth_rate - death_rate) / net_growth_rate = 3000 := by
  sorry

end NUMINAMATH_CALUDE_population_size_l3973_397358


namespace NUMINAMATH_CALUDE_f_monotonic_increasing_interval_l3973_397367

-- Define the function f(x) = -x^2 + 1
def f (x : ℝ) : ℝ := -x^2 + 1

-- State the theorem
theorem f_monotonic_increasing_interval :
  ∀ x y, x < y ∧ y ≤ 0 → f x < f y :=
by
  sorry

end NUMINAMATH_CALUDE_f_monotonic_increasing_interval_l3973_397367


namespace NUMINAMATH_CALUDE_subset_P_l3973_397348

def P : Set ℝ := {x | x ≤ 3}

theorem subset_P : {-1} ⊆ P := by sorry

end NUMINAMATH_CALUDE_subset_P_l3973_397348


namespace NUMINAMATH_CALUDE_hyperbola_inequality_l3973_397334

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 = 1

-- Define the asymptotes
def asymptote (x y : ℝ) : Prop := y = x / 2 ∨ y = -x / 2

-- Define points A and B
def A : ℝ × ℝ := (2, 1)
def B : ℝ × ℝ := (2, -1)

-- Define any point P on the hyperbola
def P (a b : ℝ) : ℝ × ℝ := (2*a + 2*b, a - b)

theorem hyperbola_inequality (a b : ℝ) :
  hyperbola (P a b).1 (P a b).2 →
  |a + b| ≥ 1 := by sorry

end NUMINAMATH_CALUDE_hyperbola_inequality_l3973_397334


namespace NUMINAMATH_CALUDE_inequality_solution_l3973_397324

theorem inequality_solution (x : ℝ) : 
  (3 - 2 / (3 * x + 4) < 5) ↔ (x < -5/3 ∧ x ≠ -4/3) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3973_397324


namespace NUMINAMATH_CALUDE_sum_of_roots_l3973_397336

/-- The function f(x) = x^3 - 6x^2 + 17x - 5 -/
def f (x : ℝ) : ℝ := x^3 - 6*x^2 + 17*x - 5

/-- Theorem: If f(a) = 3 and f(b) = 23, then a + b = 4 -/
theorem sum_of_roots (a b : ℝ) (ha : f a = 3) (hb : f b = 23) : a + b = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_l3973_397336


namespace NUMINAMATH_CALUDE_nina_homework_calculation_l3973_397378

/-- Nina's homework calculation -/
theorem nina_homework_calculation
  (ruby_math : ℕ) (ruby_reading : ℕ)
  (nina_math_multiplier : ℕ) (nina_reading_multiplier : ℕ)
  (h_ruby_math : ruby_math = 6)
  (h_ruby_reading : ruby_reading = 2)
  (h_nina_math : nina_math_multiplier = 4)
  (h_nina_reading : nina_reading_multiplier = 8) :
  (ruby_math * nina_math_multiplier + ruby_math) +
  (ruby_reading * nina_reading_multiplier + ruby_reading) = 48 := by
  sorry

#check nina_homework_calculation

end NUMINAMATH_CALUDE_nina_homework_calculation_l3973_397378


namespace NUMINAMATH_CALUDE_scaled_roots_polynomial_l3973_397364

theorem scaled_roots_polynomial (r₁ r₂ r₃ : ℂ) : 
  (r₁^3 - 4*r₁^2 + 10 = 0) → 
  (r₂^3 - 4*r₂^2 + 10 = 0) → 
  (r₃^3 - 4*r₃^2 + 10 = 0) → 
  (∀ x : ℂ, x^3 - 12*x^2 + 270 = (x - 3*r₁) * (x - 3*r₂) * (x - 3*r₃)) := by
sorry

end NUMINAMATH_CALUDE_scaled_roots_polynomial_l3973_397364


namespace NUMINAMATH_CALUDE_root_sum_power_five_l3973_397335

theorem root_sum_power_five (ζ₁ ζ₂ ζ₃ : ℂ) : 
  (ζ₁^3 - ζ₁^2 - 2*ζ₁ - 2 = 0) →
  (ζ₂^3 - ζ₂^2 - 2*ζ₂ - 2 = 0) →
  (ζ₃^3 - ζ₃^2 - 2*ζ₃ - 2 = 0) →
  (ζ₁ + ζ₂ + ζ₃ = 1) →
  (ζ₁^2 + ζ₂^2 + ζ₃^2 = 5) →
  (ζ₁^3 + ζ₂^3 + ζ₃^3 = 11) →
  (ζ₁^5 + ζ₂^5 + ζ₃^5 = 55) :=
by sorry

end NUMINAMATH_CALUDE_root_sum_power_five_l3973_397335


namespace NUMINAMATH_CALUDE_flag_distribution_theorem_l3973_397372

structure FlagDistribution where
  total_flags : ℕ
  blue_percentage : ℚ
  red_percentage : ℚ
  green_percentage : ℚ

def children_with_both_blue_and_red (fd : FlagDistribution) : ℚ :=
  fd.blue_percentage + fd.red_percentage - 1

theorem flag_distribution_theorem (fd : FlagDistribution) 
  (h1 : Even fd.total_flags)
  (h2 : fd.blue_percentage = 1/2)
  (h3 : fd.red_percentage = 3/5)
  (h4 : fd.green_percentage = 2/5)
  (h5 : fd.blue_percentage + fd.red_percentage + fd.green_percentage = 3/2) :
  children_with_both_blue_and_red fd = 1/10 := by
  sorry

end NUMINAMATH_CALUDE_flag_distribution_theorem_l3973_397372


namespace NUMINAMATH_CALUDE_range_of_a_minus_abs_b_l3973_397304

-- Define the conditions
def a_condition (a : ℝ) : Prop := 1 < a ∧ a < 3
def b_condition (b : ℝ) : Prop := -4 < b ∧ b < 2

-- Define the range of a - |b|
def range_a_minus_abs_b (x : ℝ) : Prop :=
  ∃ (a b : ℝ), a_condition a ∧ b_condition b ∧ x = a - |b|

-- Theorem statement
theorem range_of_a_minus_abs_b :
  ∀ x, range_a_minus_abs_b x ↔ -3 < x ∧ x < 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_minus_abs_b_l3973_397304


namespace NUMINAMATH_CALUDE_remaining_garlic_cloves_l3973_397363

-- Define the initial number of garlic cloves
def initial_cloves : ℕ := 93

-- Define the number of cloves used
def used_cloves : ℕ := 86

-- Theorem stating that the remaining cloves is 7
theorem remaining_garlic_cloves : initial_cloves - used_cloves = 7 := by
  sorry

end NUMINAMATH_CALUDE_remaining_garlic_cloves_l3973_397363


namespace NUMINAMATH_CALUDE_l_shape_area_and_perimeter_l3973_397357

/-- Represents the dimensions of a rectangle -/
structure RectangleDimensions where
  length : Real
  width : Real

/-- Calculates the area of a rectangle -/
def rectangleArea (d : RectangleDimensions) : Real :=
  d.length * d.width

/-- Calculates the perimeter of a rectangle -/
def rectanglePerimeter (d : RectangleDimensions) : Real :=
  2 * (d.length + d.width)

/-- Represents an L-shaped region formed by two rectangles -/
structure LShape where
  rect1 : RectangleDimensions
  rect2 : RectangleDimensions

/-- Calculates the area of an L-shaped region -/
def lShapeArea (l : LShape) : Real :=
  rectangleArea l.rect1 + rectangleArea l.rect2

/-- Calculates the perimeter of an L-shaped region -/
def lShapePerimeter (l : LShape) : Real :=
  rectanglePerimeter l.rect1 + rectanglePerimeter l.rect2 - 2 * l.rect1.length

theorem l_shape_area_and_perimeter :
  let l : LShape := {
    rect1 := { length := 0.5, width := 0.3 },
    rect2 := { length := 0.2, width := 0.5 }
  }
  lShapeArea l = 0.25 ∧ lShapePerimeter l = 2.0 := by sorry

end NUMINAMATH_CALUDE_l_shape_area_and_perimeter_l3973_397357


namespace NUMINAMATH_CALUDE_ratio_calculation_l3973_397388

theorem ratio_calculation (A B C : ℚ) (h : A / B = 3 / 2 ∧ B / C = 2 / 5) : 
  (2 * A + 3 * B) / (A + 5 * C) = 3 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ratio_calculation_l3973_397388


namespace NUMINAMATH_CALUDE_S_bounds_l3973_397339

def S : Set ℝ := {y | ∃ x : ℝ, x ≥ 0 ∧ y = (3*x + 2)/(x + 1)}

theorem S_bounds : 
  ∃ (M m : ℝ), 
    (∀ y ∈ S, y ≤ M) ∧ 
    (∀ y ∈ S, y ≥ m) ∧ 
    (M ∉ S) ∧ 
    (m ∈ S) ∧
    (M = 3) ∧ 
    (m = 2) :=
by sorry

end NUMINAMATH_CALUDE_S_bounds_l3973_397339


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3973_397342

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (i : ℂ) / (1 - i) = -1/2 + (1/2 : ℂ) * i :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3973_397342


namespace NUMINAMATH_CALUDE_number_solution_l3973_397306

theorem number_solution : ∃ x : ℝ, (3034 - (1002 / x) = 2984) ∧ x = 20.04 := by
  sorry

end NUMINAMATH_CALUDE_number_solution_l3973_397306


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3973_397374

/-- An arithmetic sequence with common difference d -/
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

/-- Three terms form a geometric sequence -/
def geometric_sequence (x y z : ℝ) : Prop :=
  y^2 = x * z

theorem arithmetic_sequence_problem (a : ℕ → ℝ) (d : ℝ) :
  arithmetic_sequence a d →
  a 2 = 4 →
  geometric_sequence (1 + a 3) (a 6) (4 + a 10) →
  d = 3 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3973_397374


namespace NUMINAMATH_CALUDE_f_monotonicity_and_extremum_l3973_397397

noncomputable def f (x : ℝ) := x * Real.exp (-x)

theorem f_monotonicity_and_extremum :
  (∀ x y : ℝ, x < y ∧ y < 1 → f x < f y) ∧
  (∀ x y : ℝ, 1 < x ∧ x < y → f y < f x) ∧
  (∀ x : ℝ, x ≠ 1 → f x < f 1) ∧
  f 1 = Real.exp (-1) := by sorry

end NUMINAMATH_CALUDE_f_monotonicity_and_extremum_l3973_397397


namespace NUMINAMATH_CALUDE_total_pins_used_l3973_397301

/-- The number of sides of a rectangle -/
def rectangle_sides : ℕ := 4

/-- The number of pins used on each side of the cardboard -/
def pins_per_side : ℕ := 35

/-- Theorem: The total number of pins used to attach a rectangular cardboard to a box -/
theorem total_pins_used : rectangle_sides * pins_per_side = 140 := by
  sorry

end NUMINAMATH_CALUDE_total_pins_used_l3973_397301


namespace NUMINAMATH_CALUDE_problem_statement_l3973_397343

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  (0 < a * b ∧ a * b ≤ 1) ∧ (a^2 + b^2 ≥ 2) ∧ (0 < b ∧ b < 2) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3973_397343


namespace NUMINAMATH_CALUDE_fold_square_problem_l3973_397323

-- Define the square ABCD
def Square (A B C D : ℝ × ℝ) : Prop :=
  let distAB := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  distAB = 8 ∧ 
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = (C.1 - B.1)^2 + (C.2 - B.2)^2 ∧
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = (D.1 - C.1)^2 + (D.2 - C.2)^2 ∧
  (D.1 - C.1)^2 + (D.2 - C.2)^2 = (A.1 - D.1)^2 + (A.2 - D.2)^2

-- Define point E as the midpoint of AD
def Midpoint (E A D : ℝ × ℝ) : Prop :=
  E.1 = (A.1 + D.1) / 2 ∧ E.2 = (A.2 + D.2) / 2

-- Define point F on BD such that BF = EF
def PointF (F B D E : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ 
  F.1 = B.1 + t * (D.1 - B.1) ∧ 
  F.2 = B.2 + t * (D.2 - B.2) ∧
  (F.1 - B.1)^2 + (F.2 - B.2)^2 = (F.1 - E.1)^2 + (F.2 - E.2)^2

-- Theorem statement
theorem fold_square_problem (A B C D E F : ℝ × ℝ) :
  Square A B C D → Midpoint E A D → PointF F B D E →
  (F.1 - D.1)^2 + (F.2 - D.2)^2 = 3^2 := by
  sorry

end NUMINAMATH_CALUDE_fold_square_problem_l3973_397323


namespace NUMINAMATH_CALUDE_cubic_root_implies_coefficients_l3973_397313

theorem cubic_root_implies_coefficients 
  (a b : ℝ) 
  (h : (2 - 3*Complex.I)^3 + a*(2 - 3*Complex.I)^2 - 2*(2 - 3*Complex.I) + b = 0) : 
  a = -1/4 ∧ b = 195/4 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_implies_coefficients_l3973_397313


namespace NUMINAMATH_CALUDE_point_in_inequality_region_implies_B_range_l3973_397330

/-- Given a point A (1, 2) inside the plane region corresponding to the linear inequality 2x - By + 3 ≥ 0, 
    prove that the range of the real number B is B ≤ 2.5. -/
theorem point_in_inequality_region_implies_B_range (B : ℝ) : 
  (2 * 1 - B * 2 + 3 ≥ 0) → B ≤ 2.5 := by
  sorry

end NUMINAMATH_CALUDE_point_in_inequality_region_implies_B_range_l3973_397330


namespace NUMINAMATH_CALUDE_existence_of_square_root_of_minus_one_l3973_397396

theorem existence_of_square_root_of_minus_one (p : ℕ) (hp : Nat.Prime p) :
  (∃ a : ℤ, a^2 ≡ -1 [ZMOD p]) ↔ p ≡ 1 [MOD 4] := by sorry

end NUMINAMATH_CALUDE_existence_of_square_root_of_minus_one_l3973_397396


namespace NUMINAMATH_CALUDE_min_n_for_sqrt_12n_integer_l3973_397391

theorem min_n_for_sqrt_12n_integer (n : ℕ) (h1 : n > 0) (h2 : ∃ k : ℕ, k > 0 ∧ k^2 = 12*n) :
  ∀ m : ℕ, m > 0 → (∃ j : ℕ, j > 0 ∧ j^2 = 12*m) → m ≥ 3 :=
by sorry

end NUMINAMATH_CALUDE_min_n_for_sqrt_12n_integer_l3973_397391


namespace NUMINAMATH_CALUDE_james_music_beats_l3973_397305

/-- Calculate the number of beats heard in a week given the beats per minute,
    hours of listening per day, and days in a week. -/
def beats_per_week (beats_per_minute : ℕ) (hours_per_day : ℕ) (days_per_week : ℕ) : ℕ :=
  beats_per_minute * 60 * hours_per_day * days_per_week

/-- Theorem stating that listening to 200 beats per minute music for 2 hours a day
    for 7 days results in hearing 168,000 beats in a week. -/
theorem james_music_beats :
  beats_per_week 200 2 7 = 168000 := by
  sorry

end NUMINAMATH_CALUDE_james_music_beats_l3973_397305


namespace NUMINAMATH_CALUDE_romeo_profit_l3973_397359

/-- Calculates the profit for selling chocolate bars -/
def chocolate_profit (num_bars : ℕ) (cost_per_bar : ℕ) (selling_price : ℕ) (packaging_cost : ℕ) : ℕ :=
  selling_price - (num_bars * cost_per_bar + num_bars * packaging_cost)

/-- Theorem: Romeo's profit is $55 -/
theorem romeo_profit : 
  chocolate_profit 5 5 90 2 = 55 := by
  sorry

end NUMINAMATH_CALUDE_romeo_profit_l3973_397359


namespace NUMINAMATH_CALUDE_ellipse_intersection_right_triangle_l3973_397319

/-- Defines an ellipse with equation x²/4 + y²/2 = 1 -/
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2/2 = 1

/-- Defines a line with equation y = x + m -/
def line (x y m : ℝ) : Prop := y = x + m

/-- Defines the intersection points of the ellipse and the line -/
def intersection (x₁ y₁ x₂ y₂ m : ℝ) : Prop :=
  ellipse x₁ y₁ ∧ ellipse x₂ y₂ ∧ line x₁ y₁ m ∧ line x₂ y₂ m

/-- Defines a point Q on the y-axis -/
def point_on_y_axis (y : ℝ) : Prop := true

/-- Defines a right triangle formed by points Q, A, and B -/
def right_triangle (x₁ y₁ x₂ y₂ y : ℝ) : Prop :=
  (x₂ - x₁)*(0 - x₁) + (y₂ - y₁)*(y - y₁) = 0

/-- Main theorem: If there exists a point Q on the y-axis such that △QAB is a right triangle,
    where A and B are the intersection points of the ellipse x²/4 + y²/2 = 1 and the line y = x + m,
    then m = ±(3√10)/5 -/
theorem ellipse_intersection_right_triangle (m : ℝ) :
  (∃ x₁ y₁ x₂ y₂ y, intersection x₁ y₁ x₂ y₂ m ∧ point_on_y_axis y ∧ right_triangle x₁ y₁ x₂ y₂ y) →
  m = 3*Real.sqrt 10/5 ∨ m = -3*Real.sqrt 10/5 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_intersection_right_triangle_l3973_397319


namespace NUMINAMATH_CALUDE_problems_left_to_grade_l3973_397360

theorem problems_left_to_grade 
  (total_worksheets : ℕ) 
  (graded_worksheets : ℕ) 
  (problems_per_worksheet : ℕ) 
  (h1 : total_worksheets = 9) 
  (h2 : graded_worksheets = 5) 
  (h3 : problems_per_worksheet = 4) : 
  (total_worksheets - graded_worksheets) * problems_per_worksheet = 16 :=
by sorry

end NUMINAMATH_CALUDE_problems_left_to_grade_l3973_397360


namespace NUMINAMATH_CALUDE_exam_score_l3973_397379

theorem exam_score (total_questions : ℕ) (correct_answers : ℕ) 
  (marks_per_correct : ℕ) (marks_lost_per_wrong : ℕ) : 
  total_questions = 80 → 
  correct_answers = 40 → 
  marks_per_correct = 4 → 
  marks_lost_per_wrong = 1 → 
  (correct_answers * marks_per_correct) - 
    ((total_questions - correct_answers) * marks_lost_per_wrong) = 120 := by
  sorry

#check exam_score

end NUMINAMATH_CALUDE_exam_score_l3973_397379


namespace NUMINAMATH_CALUDE_right_triangle_trig_l3973_397347

theorem right_triangle_trig (P Q R : ℝ × ℝ) : 
  let pq := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)
  let pr := Real.sqrt ((P.1 - R.1)^2 + (P.2 - R.2)^2)
  let qr := Real.sqrt ((Q.1 - R.1)^2 + (Q.2 - R.2)^2)
  (∀ S, S ≠ R → (Real.sqrt ((R.1 - S.1)^2 + (R.2 - S.2)^2))^2 = (S.1 - R.1)^2 + (S.2 - R.2)^2) →
  pq = 15 →
  pr = 9 →
  qr^2 + pr^2 = pq^2 →
  (qr / pq) = 4/5 ∧ (pr / pq) = 3/5 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_trig_l3973_397347


namespace NUMINAMATH_CALUDE_group_size_problem_l3973_397331

theorem group_size_problem (total_paise : ℕ) (h1 : total_paise = 5776) : ∃ n : ℕ, n * n = total_paise ∧ n = 76 := by
  sorry

end NUMINAMATH_CALUDE_group_size_problem_l3973_397331


namespace NUMINAMATH_CALUDE_bank_interest_rate_problem_l3973_397303

theorem bank_interest_rate_problem
  (initial_investment : ℝ)
  (additional_investment : ℝ)
  (additional_rate : ℝ)
  (total_rate : ℝ)
  (h1 : initial_investment = 2200)
  (h2 : additional_investment = 1099.9999999999998)
  (h3 : additional_rate = 0.08)
  (h4 : total_rate = 0.06)
  (h5 : total_rate * (initial_investment + additional_investment) =
        initial_investment * x + additional_investment * additional_rate) :
  x = 0.05 :=
by sorry

end NUMINAMATH_CALUDE_bank_interest_rate_problem_l3973_397303


namespace NUMINAMATH_CALUDE_tims_sleep_schedule_l3973_397384

/-- Tim's sleep schedule and total sleep calculation -/
theorem tims_sleep_schedule (weekday_sleep : ℕ) (weekend_sleep : ℕ) (weekdays : ℕ) (weekend_days : ℕ) :
  weekday_sleep = 6 →
  weekend_sleep = 10 →
  weekdays = 5 →
  weekend_days = 2 →
  weekday_sleep * weekdays + weekend_sleep * weekend_days = 50 := by
  sorry

#check tims_sleep_schedule

end NUMINAMATH_CALUDE_tims_sleep_schedule_l3973_397384


namespace NUMINAMATH_CALUDE_largest_solution_quadratic_l3973_397369

theorem largest_solution_quadratic : 
  let f : ℝ → ℝ := λ x => 6 * x^2 - 31 * x + 35
  ∃ x : ℝ, f x = 0 ∧ ∀ y : ℝ, f y = 0 → y ≤ x ∧ x = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_largest_solution_quadratic_l3973_397369


namespace NUMINAMATH_CALUDE_f_2018_l3973_397327

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
axiom periodic : ∀ x : ℝ, f (x + 4) = -f x
axiom symmetric : ∀ x : ℝ, f (1 - (x - 1)) = f (x - 1)
axiom f_2 : f 2 = 2

-- The theorem to prove
theorem f_2018 : f 2018 = 2 := by sorry

end NUMINAMATH_CALUDE_f_2018_l3973_397327


namespace NUMINAMATH_CALUDE_bart_survey_earnings_l3973_397386

theorem bart_survey_earnings :
  let questions_per_survey : ℕ := 10
  let earnings_per_question : ℚ := 0.2
  let monday_surveys : ℕ := 3
  let tuesday_surveys : ℕ := 4
  
  let total_questions := questions_per_survey * (monday_surveys + tuesday_surveys)
  let total_earnings := (total_questions : ℚ) * earnings_per_question

  total_earnings = 14 :=
by sorry

end NUMINAMATH_CALUDE_bart_survey_earnings_l3973_397386


namespace NUMINAMATH_CALUDE_range_a_theorem_l3973_397315

-- Define the propositions P and Q
def P (a : ℝ) : Prop := ∀ x ∈ Set.Icc (-1) 2, x^2 - a ≥ 0
def Q (a : ℝ) : Prop := ∀ x : ℝ, 2*x^2 + a*x + 1 > 0

-- Define the range of a
def range_of_a (a : ℝ) : Prop := a ≤ -2*Real.sqrt 2 ∨ (0 < a ∧ a < 2*Real.sqrt 2)

-- State the theorem
theorem range_a_theorem (a : ℝ) : 
  (¬(P a ∧ Q a) ∧ (P a ∨ Q a)) → range_of_a a :=
by sorry

end NUMINAMATH_CALUDE_range_a_theorem_l3973_397315


namespace NUMINAMATH_CALUDE_discarded_numbers_l3973_397349

-- Define the set of numbers
def numbers : Finset ℕ := Finset.range 11 \ {0}

-- Define the type for a distribution on a rectangular block
structure BlockDistribution where
  vertices : Finset ℕ
  face_sum : ℕ
  is_valid : vertices ⊆ numbers ∧ vertices.card = 8 ∧ face_sum = 18

-- Theorem statement
theorem discarded_numbers (d : BlockDistribution) :
  numbers \ d.vertices = {9, 10} := by
  sorry

end NUMINAMATH_CALUDE_discarded_numbers_l3973_397349


namespace NUMINAMATH_CALUDE_unique_coin_combination_l3973_397395

/-- Represents the number of coins of each denomination -/
structure CoinCounts where
  five : ℕ
  ten : ℕ
  twentyFive : ℕ

/-- Calculates the number of different values obtainable from a given set of coins -/
def differentValues (coins : CoinCounts) : ℕ :=
  14 + coins.ten + 4 * coins.twentyFive

/-- The main theorem -/
theorem unique_coin_combination :
  ∀ (coins : CoinCounts),
    coins.five + coins.ten + coins.twentyFive = 15 →
    differentValues coins = 21 →
    coins.twentyFive = 1 := by
  sorry

#check unique_coin_combination

end NUMINAMATH_CALUDE_unique_coin_combination_l3973_397395


namespace NUMINAMATH_CALUDE_vasya_numbers_l3973_397310

theorem vasya_numbers : ∃! (x y : ℝ), x + y = x * y ∧ x + y = x / y ∧ x = (1 : ℝ) / 2 ∧ y = -(1 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_vasya_numbers_l3973_397310


namespace NUMINAMATH_CALUDE_kody_age_is_32_l3973_397346

-- Define Mohamed's current age
def mohamed_current_age : ℕ := 2 * 30

-- Define Mohamed's age four years ago
def mohamed_past_age : ℕ := mohamed_current_age - 4

-- Define Kody's age four years ago
def kody_past_age : ℕ := mohamed_past_age / 2

-- Define Kody's current age
def kody_current_age : ℕ := kody_past_age + 4

-- Theorem stating Kody's current age
theorem kody_age_is_32 : kody_current_age = 32 := by
  sorry

end NUMINAMATH_CALUDE_kody_age_is_32_l3973_397346


namespace NUMINAMATH_CALUDE_nth_equation_holds_l3973_397332

theorem nth_equation_holds (n : ℕ) : 
  (n : ℚ) / (n + 1) = (n + 3 * 2 * n) / (n + 1 + 3 * 2 * (n + 1)) := by
  sorry

end NUMINAMATH_CALUDE_nth_equation_holds_l3973_397332


namespace NUMINAMATH_CALUDE_pool_fill_time_ab_l3973_397398

/-- Represents the time it takes for a valve to fill the pool individually -/
structure ValveTime where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents the conditions given in the problem -/
structure PoolFillConditions where
  vt : ValveTime
  all_valves_time : (1 / vt.a + 1 / vt.b + 1 / vt.c) = 1
  ac_time : (1 / vt.a + 1 / vt.c) * 1.5 = 1
  bc_time : (1 / vt.b + 1 / vt.c) * 2 = 1

/-- Theorem stating that given the conditions, the time to fill the pool with valves A and B is 1.2 hours -/
theorem pool_fill_time_ab (conditions : PoolFillConditions) : 
  (1 / conditions.vt.a + 1 / conditions.vt.b) * 1.2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_pool_fill_time_ab_l3973_397398


namespace NUMINAMATH_CALUDE_origami_stars_per_bottle_l3973_397317

/-- Represents the problem of determining the number of origami stars per bottle -/
theorem origami_stars_per_bottle
  (total_bottles : ℕ)
  (total_stars : ℕ)
  (h1 : total_bottles = 5)
  (h2 : total_stars = 75) :
  total_stars / total_bottles = 15 := by
  sorry

end NUMINAMATH_CALUDE_origami_stars_per_bottle_l3973_397317


namespace NUMINAMATH_CALUDE_last_islander_is_knight_l3973_397338

/-- Represents the type of an islander: either a knight or a liar -/
inductive IslanderType
  | Knight
  | Liar

/-- Represents what an islander says about their neighbor -/
inductive Statement
  | Knight
  | Liar

/-- The number of islanders around the table -/
def numIslanders : Nat := 50

/-- Function that determines what an islander at a given position says -/
def statement (position : Nat) : Statement :=
  if position % 2 == 1 then Statement.Knight else Statement.Liar

/-- Function that determines the actual type of an islander based on their statement and the type of their neighbor -/
def actualType (position : Nat) (neighborType : IslanderType) : IslanderType :=
  match (statement position, neighborType) with
  | (Statement.Knight, IslanderType.Knight) => IslanderType.Knight
  | (Statement.Knight, IslanderType.Liar) => IslanderType.Liar
  | (Statement.Liar, IslanderType.Knight) => IslanderType.Liar
  | (Statement.Liar, IslanderType.Liar) => IslanderType.Knight

theorem last_islander_is_knight : 
  ∀ (first : IslanderType), actualType numIslanders first = IslanderType.Knight :=
by sorry

end NUMINAMATH_CALUDE_last_islander_is_knight_l3973_397338


namespace NUMINAMATH_CALUDE_problem_solution_l3973_397350

theorem problem_solution (p_xavier p_yvonne p_zelda : ℚ)
  (h_xavier : p_xavier = 1/3)
  (h_yvonne : p_yvonne = 1/2)
  (h_zelda : p_zelda = 5/8) :
  p_xavier * p_yvonne * (1 - p_zelda) = 1/16 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3973_397350


namespace NUMINAMATH_CALUDE_a_gt_one_sufficient_not_necessary_l3973_397381

theorem a_gt_one_sufficient_not_necessary (a : ℝ) :
  (a > 1 → 1/a < 1) ∧ ∃ b : ℝ, 1/b < 1 ∧ ¬(b > 1) :=
by sorry

end NUMINAMATH_CALUDE_a_gt_one_sufficient_not_necessary_l3973_397381


namespace NUMINAMATH_CALUDE_team_a_wins_l3973_397322

theorem team_a_wins (total_matches : ℕ) (team_a_points : ℕ) : 
  total_matches = 10 → 
  team_a_points = 22 → 
  ∃ (wins draws : ℕ), 
    wins + draws = total_matches ∧ 
    3 * wins + draws = team_a_points ∧ 
    wins = 6 :=
by sorry

end NUMINAMATH_CALUDE_team_a_wins_l3973_397322


namespace NUMINAMATH_CALUDE_games_not_working_l3973_397329

theorem games_not_working (games_from_friend : ℕ) (games_from_garage_sale : ℕ) (good_games : ℕ) :
  games_from_friend = 2 →
  games_from_garage_sale = 2 →
  good_games = 2 →
  games_from_friend + games_from_garage_sale - good_games = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_games_not_working_l3973_397329


namespace NUMINAMATH_CALUDE_cubic_root_sum_cubes_l3973_397320

theorem cubic_root_sum_cubes (a b c : ℝ) : 
  (a^3 - 2*a^2 - a + 3 = 0) →
  (b^3 - 2*b^2 - b + 3 = 0) →
  (c^3 - 2*c^2 - c + 3 = 0) →
  a^3 + b^3 + c^3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_sum_cubes_l3973_397320


namespace NUMINAMATH_CALUDE_determine_coins_in_38_bags_l3973_397383

/-- Represents a bag of coins -/
structure Bag where
  coins : ℕ
  inv : coins ≥ 1000

/-- Represents the state of all bags -/
def BagState := Fin 40 → Bag

/-- An operation that checks two bags and potentially removes a coin from one of them -/
def CheckOperation (state : BagState) (i j : Fin 40) : BagState := sorry

/-- Predicate to check if we know the exact number of coins in a bag -/
def KnowExactCoins (state : BagState) (i : Fin 40) : Prop := sorry

/-- The main theorem stating that it's possible to determine the number of coins in 38 out of 40 bags -/
theorem determine_coins_in_38_bags :
  ∃ (operations : List (Fin 40 × Fin 40)),
    operations.length ≤ 100 ∧
    ∀ (initial_state : BagState),
      let final_state := operations.foldl (fun state (i, j) => CheckOperation state i j) initial_state
      (∃ (unknown1 unknown2 : Fin 40), ∀ (i : Fin 40),
        i ≠ unknown1 → i ≠ unknown2 → KnowExactCoins final_state i) :=
sorry

end NUMINAMATH_CALUDE_determine_coins_in_38_bags_l3973_397383


namespace NUMINAMATH_CALUDE_dealer_profit_percentage_l3973_397302

/-- The profit percentage of a dealer who sells 900 grams of goods for the price of 1000 grams -/
theorem dealer_profit_percentage : 
  let actual_weight : ℝ := 900
  let claimed_weight : ℝ := 1000
  let profit_percentage := (claimed_weight / actual_weight - 1) * 100
  profit_percentage = (1 / 9) * 100 := by sorry

end NUMINAMATH_CALUDE_dealer_profit_percentage_l3973_397302


namespace NUMINAMATH_CALUDE_kishore_rent_expenditure_l3973_397368

def monthly_salary (savings : ℕ) : ℕ := savings * 10

def total_expenses (salary : ℕ) : ℕ := (salary * 9) / 10

def other_expenses : ℕ := 1500 + 4500 + 2500 + 2000 + 3940

def rent_expenditure (total_exp other_exp : ℕ) : ℕ := total_exp - other_exp

theorem kishore_rent_expenditure (savings : ℕ) (h : savings = 2160) :
  rent_expenditure (total_expenses (monthly_salary savings)) other_expenses = 5000 := by
  sorry

end NUMINAMATH_CALUDE_kishore_rent_expenditure_l3973_397368


namespace NUMINAMATH_CALUDE_quadratic_inequality_l3973_397300

theorem quadratic_inequality (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) 
  (ha₁ : 0 < a₁) (hb₁ : 0 < b₁) (hc₁ : 0 < c₁)
  (ha₂ : 0 < a₂) (hb₂ : 0 < b₂) (hc₂ : 0 < c₂)
  (h₁ : b₁^2 ≤ a₁*c₁) (h₂ : b₂^2 ≤ a₂*c₂) :
  (a₁ + a₂ + 5) * (c₁ + c₂ + 2) > (b₁ + b₂ + 3)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l3973_397300


namespace NUMINAMATH_CALUDE_abs_diff_gt_cube_root_product_l3973_397362

theorem abs_diff_gt_cube_root_product (a b : ℤ) : 
  a ≠ b → (a^2 + a*b + b^2) ∣ (a*b*(a + b)) → |a - b| > (a*b : ℝ)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_abs_diff_gt_cube_root_product_l3973_397362


namespace NUMINAMATH_CALUDE_proposition_induction_l3973_397325

theorem proposition_induction (P : ℕ → Prop) :
  (∀ k : ℕ, k > 0 → (P k → P (k + 1))) →
  ¬ P 7 →
  ¬ P 6 := by
  sorry

end NUMINAMATH_CALUDE_proposition_induction_l3973_397325


namespace NUMINAMATH_CALUDE_cubic_sum_equals_27_l3973_397337

theorem cubic_sum_equals_27 (a b : ℝ) (h : a + b = 3) : a^3 + b^3 + 9*a*b = 27 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_equals_27_l3973_397337


namespace NUMINAMATH_CALUDE_average_speed_proof_l3973_397389

/-- Prove that the average speed of a trip with given conditions is 40 miles per hour -/
theorem average_speed_proof (total_distance : ℝ) (first_half_speed : ℝ) (second_half_time_factor : ℝ) :
  total_distance = 640 →
  first_half_speed = 80 →
  second_half_time_factor = 3 →
  (total_distance / (total_distance / (2 * first_half_speed) * (1 + second_half_time_factor))) = 40 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_proof_l3973_397389


namespace NUMINAMATH_CALUDE_cube_volume_surface_area_l3973_397352

theorem cube_volume_surface_area (x : ℝ) :
  (∃ (s : ℝ), s > 0 ∧ s^3 = 8*x ∧ 6*s^2 = 2*x) → x = 1728 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_surface_area_l3973_397352


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3973_397328

theorem inequality_solution_set :
  ∀ x : ℝ, (((2*x - 1) / (x + 1) ≤ 1 ∧ x + 1 ≠ 0) ↔ x ∈ Set.Ioo (-1 : ℝ) 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3973_397328


namespace NUMINAMATH_CALUDE_fermat_mod_large_prime_l3973_397311

theorem fermat_mod_large_prime (n : ℕ) (hn : n > 0) :
  ∃ M : ℕ, ∀ p : ℕ, p > M → Prime p →
    ∃ x y z : ℤ, (x^n + y^n) % p = z^n % p ∧ (x * y * z) % p ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_fermat_mod_large_prime_l3973_397311


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainders_l3973_397333

theorem smallest_integer_with_remainders : ∃! M : ℕ,
  (M > 0) ∧
  (M % 7 = 6) ∧
  (M % 8 = 7) ∧
  (M % 9 = 8) ∧
  (M % 10 = 9) ∧
  (M % 11 = 10) ∧
  (M % 12 = 11) ∧
  (∀ n : ℕ, n > 0 ∧
    n % 7 = 6 ∧
    n % 8 = 7 ∧
    n % 9 = 8 ∧
    n % 10 = 9 ∧
    n % 11 = 10 ∧
    n % 12 = 11 → n ≥ M) ∧
  M = 27719 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainders_l3973_397333


namespace NUMINAMATH_CALUDE_jimmy_stairs_l3973_397373

/-- The sum of an arithmetic sequence -/
def arithmetic_sum (a : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

/-- Jimmy's stair climbing problem -/
theorem jimmy_stairs : arithmetic_sum 30 10 8 = 520 := by
  sorry

end NUMINAMATH_CALUDE_jimmy_stairs_l3973_397373


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l3973_397307

theorem pure_imaginary_complex_number (m : ℝ) : 
  (m * (m + 2)) / (m - 1) = 0 → m = 0 := by sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l3973_397307


namespace NUMINAMATH_CALUDE_white_area_is_42_l3973_397380

/-- The area of a rectangle -/
def rectangle_area (width : ℕ) (height : ℕ) : ℕ := width * height

/-- The area of the letter C -/
def c_area : ℕ := 2 * (6 * 1) + 1 * 4

/-- The area of the letter O -/
def o_area : ℕ := 2 * (6 * 1) + 2 * 4

/-- The area of the letter L -/
def l_area : ℕ := 1 * (6 * 1) + 1 * 4

/-- The total black area of the word COOL -/
def cool_area : ℕ := c_area + 2 * o_area + l_area

/-- The width of the sign -/
def sign_width : ℕ := 18

/-- The height of the sign -/
def sign_height : ℕ := 6

theorem white_area_is_42 : 
  rectangle_area sign_width sign_height - cool_area = 42 := by
  sorry

end NUMINAMATH_CALUDE_white_area_is_42_l3973_397380


namespace NUMINAMATH_CALUDE_latest_start_time_l3973_397353

def movie_start_time : ℕ := 20 -- 8 pm in 24-hour format
def home_time : ℕ := 17 -- 5 pm in 24-hour format
def dinner_duration : ℕ := 45
def homework_duration : ℕ := 30
def clean_room_duration : ℕ := 30
def trash_duration : ℕ := 5
def dishwasher_duration : ℕ := 10

def total_task_duration : ℕ := 
  dinner_duration + homework_duration + clean_room_duration + trash_duration + dishwasher_duration

theorem latest_start_time (start_time : ℕ) :
  start_time + total_task_duration / 60 = movie_start_time →
  start_time ≥ home_time →
  start_time = 18 := by sorry

end NUMINAMATH_CALUDE_latest_start_time_l3973_397353


namespace NUMINAMATH_CALUDE_ellipse_line_intersection_l3973_397392

/-- The ellipse equation -/
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

/-- The line equation -/
def line (x y m : ℝ) : Prop := y = x + m

/-- The line intersects the ellipse at two distinct points -/
def intersects_at_two_points (m : ℝ) : Prop :=
  ∃ x₁ y₁ x₂ y₂ : ℝ, x₁ ≠ x₂ ∧ 
    ellipse x₁ y₁ ∧ ellipse x₂ y₂ ∧
    line x₁ y₁ m ∧ line x₂ y₂ m

/-- Main theorem -/
theorem ellipse_line_intersection (m : ℝ) :
  intersects_at_two_points m ↔ m ∈ Set.Ioo (-Real.sqrt 7) (Real.sqrt 7) :=
sorry

end NUMINAMATH_CALUDE_ellipse_line_intersection_l3973_397392


namespace NUMINAMATH_CALUDE_symmetric_curves_l3973_397308

/-- The original curve E -/
def E (x y : ℝ) : Prop :=
  5 * x^2 + 12 * x * y - 22 * x - 12 * y - 19 = 0

/-- The line of symmetry l -/
def l (x y : ℝ) : Prop :=
  x - y + 2 = 0

/-- The symmetric curve E' -/
def E' (x y : ℝ) : Prop :=
  12 * x * y + 5 * y^2 - 78 * y + 45 = 0

/-- Theorem stating that E' is symmetric to E with respect to l -/
theorem symmetric_curves : ∀ (x y x' y' : ℝ),
  l ((x + x') / 2) ((y + y') / 2) →
  E x y ↔ E' x' y' :=
sorry

end NUMINAMATH_CALUDE_symmetric_curves_l3973_397308


namespace NUMINAMATH_CALUDE_complement_union_M_N_l3973_397382

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def M : Set Nat := {1, 4}
def N : Set Nat := {2, 3}

theorem complement_union_M_N :
  (M ∪ N)ᶜ = {5, 6} :=
by sorry

end NUMINAMATH_CALUDE_complement_union_M_N_l3973_397382


namespace NUMINAMATH_CALUDE_ladybug_count_l3973_397387

/-- The number of ladybugs with spots -/
def ladybugs_with_spots : ℕ := 12170

/-- The number of ladybugs without spots -/
def ladybugs_without_spots : ℕ := 54912

/-- The total number of ladybugs -/
def total_ladybugs : ℕ := ladybugs_with_spots + ladybugs_without_spots

theorem ladybug_count : total_ladybugs = 67082 := by
  sorry

end NUMINAMATH_CALUDE_ladybug_count_l3973_397387


namespace NUMINAMATH_CALUDE_instructors_next_meeting_l3973_397371

theorem instructors_next_meeting (f g h i j : ℕ) 
  (hf : f = 5) (hg : g = 3) (hh : h = 9) (hi : i = 2) (hj : j = 8) :
  Nat.lcm f (Nat.lcm g (Nat.lcm h (Nat.lcm i j))) = 360 :=
by sorry

end NUMINAMATH_CALUDE_instructors_next_meeting_l3973_397371


namespace NUMINAMATH_CALUDE_simple_interest_problem_l3973_397399

/-- Calculate simple interest given principal, rate, and time -/
def simple_interest (principal : ℚ) (rate : ℚ) (time : ℚ) : ℚ :=
  principal * rate * time / 100

/-- The problem statement -/
theorem simple_interest_problem :
  let principal : ℚ := 26775
  let rate : ℚ := 3
  let time : ℚ := 5
  simple_interest principal rate time = 803.25 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l3973_397399


namespace NUMINAMATH_CALUDE_circle_diameter_endpoint_l3973_397390

/-- Given a circle with center (5, -4) and one endpoint of a diameter at (0, -9),
    the other endpoint of the diameter is at (10, 1). -/
theorem circle_diameter_endpoint :
  ∀ (P : ℝ × ℝ) (A : ℝ × ℝ) (Q : ℝ × ℝ),
  P = (5, -4) →  -- Center of the circle
  A = (0, -9) →  -- One endpoint of the diameter
  (P.1 - A.1)^2 + (P.2 - A.2)^2 = (Q.1 - P.1)^2 + (Q.2 - P.2)^2 →  -- A and Q are equidistant from P
  P.1 - A.1 = Q.1 - P.1 ∧ P.2 - A.2 = Q.2 - P.2 →  -- A, P, and Q are collinear
  Q = (10, 1) :=
by sorry

end NUMINAMATH_CALUDE_circle_diameter_endpoint_l3973_397390
