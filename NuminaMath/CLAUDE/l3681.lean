import Mathlib

namespace NUMINAMATH_CALUDE_christina_age_fraction_l3681_368159

/-- Christina's current age -/
def christina_age : ℕ := sorry

/-- Oscar's current age -/
def oscar_age : ℕ := 6

/-- The fraction of Christina's age in 5 years to 80 years -/
def christina_fraction : ℚ := (christina_age + 5) / 80

theorem christina_age_fraction :
  (oscar_age + 15 = 3 * christina_age / 5) →
  christina_fraction = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_christina_age_fraction_l3681_368159


namespace NUMINAMATH_CALUDE_problem_solution_l3681_368114

-- Define proposition p
def p : Prop := ∀ x : ℝ, x^2 + x + 1 ≥ 0

-- Define proposition q
def q : Prop := ∀ x : ℝ, (x > 1 → x > 2) ∧ ¬(x > 2 → x > 1)

-- Theorem to prove
theorem problem_solution : p ∧ ¬q := by sorry

end NUMINAMATH_CALUDE_problem_solution_l3681_368114


namespace NUMINAMATH_CALUDE_part1_part2_l3681_368130

-- Define the quadratic function
def f (k : ℝ) (x : ℝ) : ℝ := k * x^2 - 2 * x + 6 * k

-- Part 1
theorem part1 (k : ℝ) : 
  (∀ x, f k x < 0 ↔ 2 < x ∧ x < 3) → k = 2/5 := by sorry

-- Part 2
theorem part2 (k : ℝ) :
  k > 0 ∧ (∀ x, 2 < x ∧ x < 3 → f k x < 0) → 0 < k ∧ k ≤ 2/5 := by sorry

end NUMINAMATH_CALUDE_part1_part2_l3681_368130


namespace NUMINAMATH_CALUDE_line_circle_separate_l3681_368167

theorem line_circle_separate (x₀ y₀ a : ℝ) (h1 : x₀^2 + y₀^2 < a^2) (h2 : a > 0) (h3 : (x₀, y₀) ≠ (0, 0)) :
  ∀ x y, x₀*x + y₀*y = a^2 → x^2 + y^2 ≠ a^2 :=
sorry

end NUMINAMATH_CALUDE_line_circle_separate_l3681_368167


namespace NUMINAMATH_CALUDE_complex_point_l3681_368145

theorem complex_point (i : ℂ) (h : i ^ 2 = -1) :
  let z : ℂ := i + 2 * i^2 + 3 * i^3
  (z.re = -2) ∧ (z.im = -2) := by sorry

end NUMINAMATH_CALUDE_complex_point_l3681_368145


namespace NUMINAMATH_CALUDE_no_solution_for_diophantine_equation_l3681_368115

theorem no_solution_for_diophantine_equation :
  ¬ ∃ (m n : ℕ+), 5 * m.val^2 - 6 * m.val * n.val + 7 * n.val^2 = 2006 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_diophantine_equation_l3681_368115


namespace NUMINAMATH_CALUDE_greatest_integer_prime_absolute_value_l3681_368119

theorem greatest_integer_prime_absolute_value : 
  ∃ (x : ℤ), (∀ (y : ℤ), y > x → ¬(Nat.Prime (Int.natAbs (8 * y^2 - 56 * y + 21)))) ∧ 
  (Nat.Prime (Int.natAbs (8 * x^2 - 56 * x + 21))) ∧ 
  x = 4 :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_prime_absolute_value_l3681_368119


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3681_368112

theorem complex_equation_solution (i : ℂ) (h_i : i^2 = -1) :
  ∀ z : ℂ, i * z = 1 → z = -i := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3681_368112


namespace NUMINAMATH_CALUDE_standard_form_is_quadratic_expanded_form_is_quadratic_l3681_368103

/-- Definition of a quadratic equation -/
def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation ax^2 + bx + c = 0 (where a ≠ 0) is quadratic -/
theorem standard_form_is_quadratic (a b c : ℝ) (ha : a ≠ 0) :
  is_quadratic (λ x => a * x^2 + b * x + c) :=
sorry

/-- The equation (x-2)^2 - 4 = 0 is quadratic -/
theorem expanded_form_is_quadratic :
  is_quadratic (λ x => (x - 2)^2 - 4) :=
sorry

end NUMINAMATH_CALUDE_standard_form_is_quadratic_expanded_form_is_quadratic_l3681_368103


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3681_368135

theorem polynomial_factorization (x : ℤ) :
  x^12 + x^6 + 1 = (x^2 + x + 1) * (x^10 - x^9 + x^8 - x^7 + x^6 - x^5 + x^4 - x^3 + x^2 - x + 1) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3681_368135


namespace NUMINAMATH_CALUDE_mabels_tomatoes_l3681_368160

/-- The number of tomatoes Mabel has -/
def total_tomatoes (plant1 plant2 plant3 plant4 : ℕ) : ℕ :=
  plant1 + plant2 + plant3 + plant4

/-- Theorem stating the total number of tomatoes Mabel has -/
theorem mabels_tomatoes :
  ∃ (plant1 plant2 plant3 plant4 : ℕ),
    plant1 = 8 ∧
    plant2 = plant1 + 4 ∧
    plant3 = 3 * (plant1 + plant2) ∧
    plant4 = 3 * (plant1 + plant2) ∧
    total_tomatoes plant1 plant2 plant3 plant4 = 140 :=
by
  sorry

end NUMINAMATH_CALUDE_mabels_tomatoes_l3681_368160


namespace NUMINAMATH_CALUDE_cubic_polynomial_value_l3681_368173

/-- A cubic polynomial function. -/
def CubicPolynomial (f : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℝ, ∀ x, f x = a * x^3 + b * x^2 + c * x + d

/-- The main theorem stating that a cubic polynomial with given properties has f(1) = -23. -/
theorem cubic_polynomial_value (f : ℝ → ℝ) 
  (hcubic : CubicPolynomial f)
  (h1 : f (-2) = -4)
  (h2 : f 3 = -9)
  (h3 : f (-4) = -16) :
  f 1 = -23 := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_value_l3681_368173


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3681_368118

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x ≥ 0 → x^2 + 4*x + 3 > 0) ↔ (∃ x : ℝ, x ≥ 0 ∧ x^2 + 4*x + 3 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3681_368118


namespace NUMINAMATH_CALUDE_tan_alpha_plus_pi_12_l3681_368185

theorem tan_alpha_plus_pi_12 (α : Real) 
  (h : Real.sin α = 3 * Real.sin (α + π/6)) : 
  Real.tan (α + π/12) = 2 * Real.sqrt 3 - 4 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_plus_pi_12_l3681_368185


namespace NUMINAMATH_CALUDE_problem_solution_l3681_368120

theorem problem_solution (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 7 * x^2 + 21 * x * y = x^3 + 3 * x^2 * y^2) : x = 7 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3681_368120


namespace NUMINAMATH_CALUDE_count_four_digit_integers_eq_six_l3681_368142

def digits : Multiset ℕ := {2, 2, 9, 9}

/-- The number of different positive, four-digit integers that can be formed using the digits 2, 2, 9, and 9 -/
def count_four_digit_integers : ℕ :=
  Nat.factorial 4 / (Nat.factorial 2 * Nat.factorial 2)

theorem count_four_digit_integers_eq_six :
  count_four_digit_integers = 6 := by
  sorry

end NUMINAMATH_CALUDE_count_four_digit_integers_eq_six_l3681_368142


namespace NUMINAMATH_CALUDE_sum_of_triangles_l3681_368136

-- Define the triangle operation
def triangle (a b c : ℤ) : ℤ := a + 2*b - c

-- Theorem statement
theorem sum_of_triangles : triangle 3 5 7 + triangle 6 1 8 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_triangles_l3681_368136


namespace NUMINAMATH_CALUDE_eggs_per_meal_l3681_368141

def initial_eggs : ℕ := 24
def used_eggs : ℕ := 6
def meals : ℕ := 3

theorem eggs_per_meal :
  let remaining_after_use := initial_eggs - used_eggs
  let remaining_after_sharing := remaining_after_use / 2
  remaining_after_sharing / meals = 3 :=
by sorry

end NUMINAMATH_CALUDE_eggs_per_meal_l3681_368141


namespace NUMINAMATH_CALUDE_ampersand_eight_two_squared_l3681_368144

def ampersand (a b : ℝ) : ℝ := (a + b) * (a - b)

theorem ampersand_eight_two_squared :
  (ampersand 8 2)^2 = 3600 := by
  sorry

end NUMINAMATH_CALUDE_ampersand_eight_two_squared_l3681_368144


namespace NUMINAMATH_CALUDE_exactly_two_solutions_l3681_368163

-- Define the system of equations
def satisfies_system (x y : ℝ) : Prop :=
  x + 2*y = 2 ∧ |abs x - 2*(abs y)| = 2

-- Define the solution set
def solution_set : Set (ℝ × ℝ) :=
  {pair | satisfies_system pair.1 pair.2}

-- Theorem statement
theorem exactly_two_solutions :
  ∃ (a b c d : ℝ), 
    solution_set = {(a, b), (c, d)} ∧
    (a, b) ≠ (c, d) ∧
    ∀ (x y : ℝ), (x, y) ∈ solution_set → (x, y) = (a, b) ∨ (x, y) = (c, d) :=
sorry

end NUMINAMATH_CALUDE_exactly_two_solutions_l3681_368163


namespace NUMINAMATH_CALUDE_two_thousand_one_in_first_column_l3681_368191

-- Define the column patterns
def first_column (n : ℕ) : Prop := ∃ k : ℕ, n = 16 * k + 1
def second_column (n : ℕ) : Prop := ∃ k : ℕ, n = 16 * k + 3
def third_column (n : ℕ) : Prop := ∃ k : ℕ, n = 16 * k + 5
def fourth_column (n : ℕ) : Prop := ∃ k : ℕ, n = 16 * k + 7

-- Define the theorem
theorem two_thousand_one_in_first_column : 
  first_column 2001 ∧ ¬(second_column 2001 ∨ third_column 2001 ∨ fourth_column 2001) :=
by sorry

end NUMINAMATH_CALUDE_two_thousand_one_in_first_column_l3681_368191


namespace NUMINAMATH_CALUDE_reunion_handshakes_l3681_368183

/-- The number of handshakes when n boys each shake hands once with every other boy -/
def handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: For 9 boys at a reunion, where each boy shakes hands exactly once with each of the others, 
    the total number of handshakes is 36. -/
theorem reunion_handshakes : handshakes 9 = 36 := by
  sorry

end NUMINAMATH_CALUDE_reunion_handshakes_l3681_368183


namespace NUMINAMATH_CALUDE_ship_typhoon_probability_l3681_368146

/-- The probability of a ship being affected by a typhoon -/
theorem ship_typhoon_probability 
  (OA OB : ℝ) 
  (h_OA : OA = 100) 
  (h_OB : OB = 100) 
  (r_min r_max : ℝ) 
  (h_r_min : r_min = 50) 
  (h_r_max : r_max = 100) : 
  ∃ (P : ℝ), P = 1 - Real.sqrt 2 / 2 ∧ 
  P = (r_max - Real.sqrt (OA^2 + OB^2) / 2) / (r_max - r_min) := by
  sorry

#check ship_typhoon_probability

end NUMINAMATH_CALUDE_ship_typhoon_probability_l3681_368146


namespace NUMINAMATH_CALUDE_reflection_point_l3681_368186

/-- A function that passes through a given point when shifted -/
def passes_through (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  f (a - 1) = b

/-- Reflection of a function across the x-axis -/
def reflect_x (f : ℝ → ℝ) : ℝ → ℝ :=
  λ x => -f x

/-- A function passes through a point -/
def function_at_point (f : ℝ → ℝ) (x y : ℝ) : Prop :=
  f x = y

theorem reflection_point (f : ℝ → ℝ) :
  passes_through f 3 2 →
  function_at_point (reflect_x f) 4 (-2) := by
  sorry

end NUMINAMATH_CALUDE_reflection_point_l3681_368186


namespace NUMINAMATH_CALUDE_storks_vs_birds_l3681_368157

theorem storks_vs_birds (initial_birds : ℕ) (additional_birds : ℕ) (storks : ℕ) : 
  initial_birds = 3 → additional_birds = 2 → storks = 6 → 
  storks - (initial_birds + additional_birds) = 1 := by
sorry

end NUMINAMATH_CALUDE_storks_vs_birds_l3681_368157


namespace NUMINAMATH_CALUDE_corn_plants_multiple_of_max_l3681_368192

/-- Represents the number of plants in a garden -/
structure GardenPlants where
  sunflowers : ℕ
  corn : ℕ
  tomatoes : ℕ

/-- Represents the constraints for planting in the garden -/
structure GardenConstraints where
  max_plants_per_row : ℕ
  same_plants_per_row : Bool
  one_type_per_row : Bool

/-- Theorem stating that the number of corn plants must be a multiple of the maximum plants per row -/
theorem corn_plants_multiple_of_max (garden : GardenPlants) (constraints : GardenConstraints) 
  (h1 : garden.sunflowers = 45)
  (h2 : garden.tomatoes = 63)
  (h3 : constraints.max_plants_per_row = 9)
  (h4 : constraints.same_plants_per_row = true)
  (h5 : constraints.one_type_per_row = true) :
  ∃ k : ℕ, garden.corn = k * constraints.max_plants_per_row := by
  sorry

end NUMINAMATH_CALUDE_corn_plants_multiple_of_max_l3681_368192


namespace NUMINAMATH_CALUDE_roger_coin_count_l3681_368178

/-- The total number of coins in Roger's collection -/
def total_coins (quarters : List Nat) (dimes : List Nat) (nickels : List Nat) (pennies : List Nat) : Nat :=
  quarters.sum + dimes.sum + nickels.sum + pennies.sum

/-- Theorem stating that Roger has 93 coins in total -/
theorem roger_coin_count :
  let quarters := [8, 6, 7, 5]
  let dimes := [7, 5, 9]
  let nickels := [4, 6]
  let pennies := [10, 3, 8, 2, 13]
  total_coins quarters dimes nickels pennies = 93 := by
  sorry

#eval total_coins [8, 6, 7, 5] [7, 5, 9] [4, 6] [10, 3, 8, 2, 13]

end NUMINAMATH_CALUDE_roger_coin_count_l3681_368178


namespace NUMINAMATH_CALUDE_fruit_distribution_ways_l3681_368179

def num_apples : ℕ := 2
def num_pears : ℕ := 3
def num_days : ℕ := 5

theorem fruit_distribution_ways :
  (Nat.choose num_days num_apples) = 10 := by
  sorry

end NUMINAMATH_CALUDE_fruit_distribution_ways_l3681_368179


namespace NUMINAMATH_CALUDE_correct_forecast_interpretation_l3681_368133

-- Define the probability of rainfall
def rainfall_probability : ℝ := 0.9

-- Define the event of getting wet when going out without rain gear
def might_get_wet (p : ℝ) : Prop :=
  p > 0 ∧ p < 1

-- Theorem statement
theorem correct_forecast_interpretation :
  might_get_wet rainfall_probability := by
  sorry

end NUMINAMATH_CALUDE_correct_forecast_interpretation_l3681_368133


namespace NUMINAMATH_CALUDE_positive_less_than_one_inequality_l3681_368139

theorem positive_less_than_one_inequality (a b : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (ha1 : a < 1) (hb1 : b < 1) : 
  1 + a^2 + b^2 > 3 * a * b := by
  sorry

end NUMINAMATH_CALUDE_positive_less_than_one_inequality_l3681_368139


namespace NUMINAMATH_CALUDE_parabola_point_x_coordinate_l3681_368196

/-- A point on a parabola with a specific distance to the focus -/
structure ParabolaPoint where
  y : ℝ
  x : ℝ
  parabola_eq : x = 4 * y^2
  focus_distance : Real.sqrt ((x - 1/4)^2 + y^2) = 1/2

/-- The x-coordinate of a point on a parabola with a specific distance to the focus -/
theorem parabola_point_x_coordinate (M : ParabolaPoint) : M.x = 7/16 := by
  sorry

end NUMINAMATH_CALUDE_parabola_point_x_coordinate_l3681_368196


namespace NUMINAMATH_CALUDE_automotive_test_distance_l3681_368148

/-- Calculates the total distance driven in an automotive test -/
theorem automotive_test_distance (d : ℝ) (t : ℝ) : 
  t = d / 4 + d / 5 + d / 6 ∧ t = 37 → 3 * d = 180 := by
  sorry

#check automotive_test_distance

end NUMINAMATH_CALUDE_automotive_test_distance_l3681_368148


namespace NUMINAMATH_CALUDE_bike_distance_proof_l3681_368190

/-- Calculates the distance traveled given speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Proves that a bike traveling at 90 km/h for 5 hours covers 450 km -/
theorem bike_distance_proof :
  let speed : ℝ := 90
  let time : ℝ := 5
  distance speed time = 450 := by sorry

end NUMINAMATH_CALUDE_bike_distance_proof_l3681_368190


namespace NUMINAMATH_CALUDE_profit_and_marginal_profit_max_not_equal_l3681_368153

def marginal_function (f : ℕ → ℝ) : ℕ → ℝ := λ x => f (x + 1) - f x

def revenue (a : ℝ) : ℕ → ℝ := λ x => 3000 * x + a * x^2

def cost (k : ℝ) : ℕ → ℝ := λ x => k * x + 4000

def profit (a k : ℝ) : ℕ → ℝ := λ x => revenue a x - cost k x

def marginal_profit (a k : ℝ) : ℕ → ℝ := marginal_function (profit a k)

theorem profit_and_marginal_profit_max_not_equal :
  ∃ (a k : ℝ),
    (∀ x : ℕ, 0 < x ∧ x ≤ 100 → profit a k x ≤ 74120) ∧
    (∃ x : ℕ, 0 < x ∧ x ≤ 100 ∧ profit a k x = 74120) ∧
    (∀ x : ℕ, 0 < x ∧ x ≤ 100 → marginal_profit a k x ≤ 2440) ∧
    (∃ x : ℕ, 0 < x ∧ x ≤ 100 ∧ marginal_profit a k x = 2440) ∧
    (cost k 10 = 9000) ∧
    (profit a k 10 = 19000) ∧
    74120 ≠ 2440 :=
by
  sorry

end NUMINAMATH_CALUDE_profit_and_marginal_profit_max_not_equal_l3681_368153


namespace NUMINAMATH_CALUDE_x_less_than_y_less_than_zero_l3681_368107

theorem x_less_than_y_less_than_zero (x y : ℝ) 
  (h1 : x^2 - y^2 > 2*x) 
  (h2 : x*y < y) : 
  x < y ∧ y < 0 := by
sorry

end NUMINAMATH_CALUDE_x_less_than_y_less_than_zero_l3681_368107


namespace NUMINAMATH_CALUDE_parabola_properties_l3681_368189

/-- Parabola C: y² = 2px with focus F(2,0) and point A(6,3) -/
def Parabola (p : ℝ) : Set (ℝ × ℝ) :=
  {point : ℝ × ℝ | point.2^2 = 2 * p * point.1}

def F : ℝ × ℝ := (2, 0)
def A : ℝ × ℝ := (6, 3)

/-- The value of p for the given parabola -/
def p_value : ℝ := 4

/-- The minimum value of |MA| + |MF| where M is on the parabola -/
def min_distance : ℝ := 8

theorem parabola_properties :
  ∃ (p : ℝ), p = p_value ∧
  (∀ (M : ℝ × ℝ), M ∈ Parabola p →
    ∀ (d : ℝ), d = Real.sqrt ((M.1 - A.1)^2 + (M.2 - A.2)^2) + Real.sqrt ((M.1 - F.1)^2 + (M.2 - F.2)^2) →
    d ≥ min_distance) :=
sorry

end NUMINAMATH_CALUDE_parabola_properties_l3681_368189


namespace NUMINAMATH_CALUDE_square_roots_problem_l3681_368187

theorem square_roots_problem (n : ℝ) (x : ℝ) (hn : n > 0) 
  (h1 : x + 1 = Real.sqrt n) (h2 : x - 5 = Real.sqrt n) : n = 9 := by
  sorry

end NUMINAMATH_CALUDE_square_roots_problem_l3681_368187


namespace NUMINAMATH_CALUDE_meet_once_l3681_368172

/-- Represents the movement of Michael and the garbage truck --/
structure Movement where
  michael_speed : ℝ
  truck_speed : ℝ
  pail_distance : ℝ
  truck_stop_time : ℝ
  initial_distance : ℝ

/-- Calculates the number of meetings between Michael and the garbage truck --/
def number_of_meetings (m : Movement) : ℕ :=
  sorry

/-- The theorem stating that Michael and the garbage truck meet exactly once --/
theorem meet_once (m : Movement) 
  (h1 : m.michael_speed = 3)
  (h2 : m.truck_speed = 6)
  (h3 : m.pail_distance = 100)
  (h4 : m.truck_stop_time = 20)
  (h5 : m.initial_distance = 100) : 
  number_of_meetings m = 1 :=
sorry

end NUMINAMATH_CALUDE_meet_once_l3681_368172


namespace NUMINAMATH_CALUDE_equation_represents_two_lines_l3681_368177

/-- The equation x^2 - 72y^2 - 16x + 64 = 0 represents two lines in the xy-plane. -/
theorem equation_represents_two_lines :
  ∃ (a b c d : ℝ), ∀ (x y : ℝ),
    (x^2 - 72*y^2 - 16*x + 64 = 0) ↔ ((x = a*y + b) ∨ (x = c*y + d)) :=
by sorry

end NUMINAMATH_CALUDE_equation_represents_two_lines_l3681_368177


namespace NUMINAMATH_CALUDE_isosceles_triangle_leg_length_l3681_368184

/-- Given an isosceles triangle ABC with area √3/2 and sin(A) = √3 * sin(B),
    prove that the length of one of the legs is √2. -/
theorem isosceles_triangle_leg_length 
  (A B C : Real) -- Angles of the triangle
  (a b c : Real) -- Sides of the triangle
  (h : Real) -- Height of the triangle
  (area : Real) -- Area of the triangle
  (is_isosceles : b = c) -- Triangle is isosceles
  (area_value : area = Real.sqrt 3 / 2) -- Area is √3/2
  (sin_relation : Real.sin A = Real.sqrt 3 * Real.sin B) -- sin(A) = √3 * sin(B)
  : b = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_leg_length_l3681_368184


namespace NUMINAMATH_CALUDE_milk_water_ratio_problem_l3681_368105

/-- Proves that the initial ratio of milk to water was 4:1 given the conditions of the mixture problem. -/
theorem milk_water_ratio_problem (initial_volume : ℝ) (added_water : ℝ) (final_ratio : ℝ) :
  initial_volume = 45 →
  added_water = 21 →
  final_ratio = 1.2 →
  ∃ (initial_milk initial_water : ℝ),
    initial_milk + initial_water = initial_volume ∧
    initial_milk / (initial_water + added_water) = final_ratio ∧
    initial_milk / initial_water = 4 / 1 :=
by
  sorry

#check milk_water_ratio_problem

end NUMINAMATH_CALUDE_milk_water_ratio_problem_l3681_368105


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l3681_368152

theorem absolute_value_inequality (x y z : ℝ) :
  |x| + |y| + |z| - |x + y| - |y + z| - |z + x| + |x + y + z| ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l3681_368152


namespace NUMINAMATH_CALUDE_fraction_simplification_l3681_368122

theorem fraction_simplification 
  (d e f : ℝ) 
  (h : d + e + f ≠ 0) : 
  (d^2 + e^2 - f^2 + 2*d*e) / (d^2 + f^2 - e^2 + 3*d*f) = (d + e - f) / (d + f - e) :=
by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3681_368122


namespace NUMINAMATH_CALUDE_geometric_series_sum_l3681_368108

theorem geometric_series_sum (a b : ℝ) (h : ∑' n, a / b^n = 5) : 
  ∑' n, a / (a + b)^n = 5/6 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l3681_368108


namespace NUMINAMATH_CALUDE_sequence_property_l3681_368111

theorem sequence_property (a : ℕ → ℕ) 
  (h_bijective : Function.Bijective a) 
  (h_positive : ∀ n, a n > 0) : 
  ∃ ℓ m : ℕ, 1 < ℓ ∧ ℓ < m ∧ a 1 + a m = 2 * a ℓ := by
  sorry

end NUMINAMATH_CALUDE_sequence_property_l3681_368111


namespace NUMINAMATH_CALUDE_part1_part2_l3681_368106

/-- Given vectors in R^2 -/
def a : ℝ × ℝ := (1, 0)
def b : ℝ × ℝ := (2, 1)

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (v w : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), v = (t * w.1, t * w.2)

/-- Theorem for part 1 -/
theorem part1 :
  ∃ (k : ℝ), k = -1/2 ∧ collinear ((k * a.1 - b.1, k * a.2 - b.2)) (a.1 + 2 * b.1, a.2 + 2 * b.2) :=
sorry

/-- Theorem for part 2 -/
theorem part2 :
  ∃ (m : ℝ), m = 3/2 ∧
  (∃ (t : ℝ), (2 * a.1 + 3 * b.1, 2 * a.2 + 3 * b.2) = (t * (a.1 + m * b.1), t * (a.2 + m * b.2))) :=
sorry

end NUMINAMATH_CALUDE_part1_part2_l3681_368106


namespace NUMINAMATH_CALUDE_midpoint_coordinate_sum_l3681_368116

/-- Given a triangle in the Cartesian plane with vertices (a, d), (b, e), and (c, f),
    if the sum of x-coordinates (a + b + c) is 15 and the sum of y-coordinates (d + e + f) is 12,
    then the sum of x-coordinates of the midpoints of its sides is 15 and
    the sum of y-coordinates of the midpoints of its sides is 12. -/
theorem midpoint_coordinate_sum (a b c d e f : ℝ) 
  (h1 : a + b + c = 15) (h2 : d + e + f = 12) : 
  (a + b) / 2 + (a + c) / 2 + (b + c) / 2 = 15 ∧ 
  (d + e) / 2 + (d + f) / 2 + (e + f) / 2 = 12 := by
  sorry


end NUMINAMATH_CALUDE_midpoint_coordinate_sum_l3681_368116


namespace NUMINAMATH_CALUDE_planes_lines_relations_l3681_368121

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (contained_in : Line → Plane → Prop)
variable (line_perpendicular : Line → Line → Prop)
variable (line_parallel : Line → Line → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- State the theorem
theorem planes_lines_relations 
  (α β : Plane) (l m : Line) 
  (h1 : perpendicular l α) 
  (h2 : contained_in m β) :
  (parallel α β → line_perpendicular l m) ∧ 
  (line_parallel l m → plane_perpendicular α β) :=
sorry

end NUMINAMATH_CALUDE_planes_lines_relations_l3681_368121


namespace NUMINAMATH_CALUDE_max_side_length_of_special_triangle_l3681_368123

theorem max_side_length_of_special_triangle (a b c : ℕ) : 
  a < b → b < c →                 -- Three different side lengths
  a + b + c = 24 →                -- Perimeter is 24
  a + b > c → b + c > a → c + a > b →  -- Triangle inequality
  c ≤ 11 := by
sorry

end NUMINAMATH_CALUDE_max_side_length_of_special_triangle_l3681_368123


namespace NUMINAMATH_CALUDE_smallest_number_l3681_368197

def base_to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * base ^ i) 0

def number_a : List Nat := [2, 0]
def number_b : List Nat := [3, 0]
def number_c : List Nat := [2, 3]
def number_d : List Nat := [3, 1]

theorem smallest_number :
  let a := base_to_decimal number_a 7
  let b := base_to_decimal number_b 5
  let c := base_to_decimal number_c 6
  let d := base_to_decimal number_d 4
  d < a ∧ d < b ∧ d < c := by sorry

end NUMINAMATH_CALUDE_smallest_number_l3681_368197


namespace NUMINAMATH_CALUDE_cubic_three_roots_range_l3681_368194

/-- The cubic polynomial function -/
def f (x : ℝ) := x^3 - 6*x^2 + 9*x

/-- The derivative of f -/
def f' (x : ℝ) := 3*x^2 - 12*x + 9

/-- Theorem: The range of m for which x^3 - 6x^2 + 9x + m = 0 has exactly three distinct real roots is (-4, 0) -/
theorem cubic_three_roots_range :
  ∀ m : ℝ, (∃! (r₁ r₂ r₃ : ℝ), r₁ ≠ r₂ ∧ r₂ ≠ r₃ ∧ r₁ ≠ r₃ ∧ 
    f r₁ + m = 0 ∧ f r₂ + m = 0 ∧ f r₃ + m = 0) ↔ 
  -4 < m ∧ m < 0 :=
sorry

end NUMINAMATH_CALUDE_cubic_three_roots_range_l3681_368194


namespace NUMINAMATH_CALUDE_x_squared_minus_y_squared_plus_8y_equals_16_l3681_368169

theorem x_squared_minus_y_squared_plus_8y_equals_16 
  (x y : ℝ) (h : x + y = 4) : x^2 - y^2 + 8*y = 16 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_minus_y_squared_plus_8y_equals_16_l3681_368169


namespace NUMINAMATH_CALUDE_task_completion_choices_l3681_368113

theorem task_completion_choices (method1 method2 : Finset Nat) : 
  method1.card = 3 → method2.card = 5 → method1 ∩ method2 = ∅ → 
  (method1 ∪ method2).card = 8 :=
by sorry

end NUMINAMATH_CALUDE_task_completion_choices_l3681_368113


namespace NUMINAMATH_CALUDE_age_ratio_theorem_l3681_368158

/-- Given a person's present age is 14 years, this theorem proves that the ratio of their age 
    16 years hence to their age 4 years ago is 3:1. -/
theorem age_ratio_theorem (present_age : ℕ) (h : present_age = 14) : 
  (present_age + 16) / (present_age - 4) = 3 := by
  sorry

#check age_ratio_theorem

end NUMINAMATH_CALUDE_age_ratio_theorem_l3681_368158


namespace NUMINAMATH_CALUDE_number_problem_l3681_368151

theorem number_problem (A B C : ℝ) 
  (h1 : A - B = 1620)
  (h2 : 0.075 * A = 0.125 * B)
  (h3 : 0.06 * B = 0.10 * C) :
  A = 4050 ∧ B = 2430 ∧ C = 1458 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l3681_368151


namespace NUMINAMATH_CALUDE_min_operations_to_identify_controllers_l3681_368102

/-- The number of light bulbs and buttons -/
def n : ℕ := 64

/-- An operation consists of pressing a set of buttons and recording the on/off state of each light bulb -/
def Operation := Fin n → Bool

/-- A sequence of operations -/
def OperationSequence := List Operation

/-- The result of applying a sequence of operations to all light bulbs -/
def ApplyOperations (ops : OperationSequence) : Fin n → List Bool :=
  fun i => ops.map (fun op => op i)

/-- A mapping from light bulbs to their controlling buttons -/
def ControlMapping := Fin n → Fin n

theorem min_operations_to_identify_controllers :
  ∃ (k : ℕ), 
    (∃ (ops : OperationSequence), ops.length = k ∧
      (∀ (m : ControlMapping), Function.Injective m →
        Function.Injective (ApplyOperations ops ∘ m))) ∧
    (∀ (j : ℕ), j < k →
      ¬∃ (ops : OperationSequence), ops.length = j ∧
        (∀ (m : ControlMapping), Function.Injective m →
          Function.Injective (ApplyOperations ops ∘ m))) ∧
    k = 6 :=
  sorry

end NUMINAMATH_CALUDE_min_operations_to_identify_controllers_l3681_368102


namespace NUMINAMATH_CALUDE_continuous_stripe_probability_l3681_368127

/-- Represents a cube with stripes on its faces -/
structure StripedCube where
  faces : Fin 6 → Fin 3

/-- The probability of a continuous stripe encircling the cube -/
def probability_continuous_stripe : ℚ := 2 / 81

/-- The total number of possible stripe configurations -/
def total_configurations : ℕ := 3^6

/-- The number of configurations that result in a continuous stripe -/
def favorable_configurations : ℕ := 18

theorem continuous_stripe_probability :
  probability_continuous_stripe = favorable_configurations / total_configurations :=
sorry

end NUMINAMATH_CALUDE_continuous_stripe_probability_l3681_368127


namespace NUMINAMATH_CALUDE_bankers_gain_calculation_l3681_368164

/-- Banker's gain calculation -/
theorem bankers_gain_calculation 
  (time : ℝ) 
  (rate : ℝ) 
  (true_discount : ℝ) 
  (ε : ℝ) 
  (h1 : time = 1) 
  (h2 : rate = 12) 
  (h3 : true_discount = 55) 
  (h4 : ε > 0) : 
  ∃ (bankers_gain : ℝ), 
    abs (bankers_gain - 6.60) < ε ∧ 
    bankers_gain = 
      (((true_discount * 100) / (rate * time) + true_discount) * rate * time) / 100 - 
      true_discount :=
sorry

end NUMINAMATH_CALUDE_bankers_gain_calculation_l3681_368164


namespace NUMINAMATH_CALUDE_at_least_one_leq_neg_two_l3681_368110

theorem at_least_one_leq_neg_two (a b c : ℝ) 
  (ha : a < 0) (hb : b < 0) (hc : c < 0) : 
  (a + 1/b ≤ -2) ∨ (b + 1/c ≤ -2) ∨ (c + 1/a ≤ -2) := by
sorry

end NUMINAMATH_CALUDE_at_least_one_leq_neg_two_l3681_368110


namespace NUMINAMATH_CALUDE_red_balls_count_l3681_368126

theorem red_balls_count (total : ℕ) (white : ℕ) (green : ℕ) (yellow : ℕ) (purple : ℕ) (p : ℚ) :
  total = 100 →
  white = 20 →
  green = 30 →
  yellow = 10 →
  purple = 3 →
  p = 0.6 →
  p = (white + green + yellow : ℚ) / total →
  ∃ red : ℕ, red = 3 ∧ total = white + green + yellow + red + purple :=
by sorry

end NUMINAMATH_CALUDE_red_balls_count_l3681_368126


namespace NUMINAMATH_CALUDE_abs_neg_one_tenth_l3681_368188

theorem abs_neg_one_tenth : |(-1/10 : ℚ)| = 1/10 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_one_tenth_l3681_368188


namespace NUMINAMATH_CALUDE_no_real_solutions_for_inequality_l3681_368150

theorem no_real_solutions_for_inequality :
  ¬ ∃ x : ℝ, -x^2 + 2*x - 3 > 0 := by
sorry

end NUMINAMATH_CALUDE_no_real_solutions_for_inequality_l3681_368150


namespace NUMINAMATH_CALUDE_investment_return_calculation_l3681_368129

theorem investment_return_calculation (total_investment small_investment large_investment : ℝ)
  (combined_return_rate small_return_rate : ℝ) :
  total_investment = small_investment + large_investment →
  small_investment = 500 →
  large_investment = 1500 →
  combined_return_rate = 0.085 →
  small_return_rate = 0.07 →
  (small_return_rate * small_investment + 
   (combined_return_rate * total_investment - small_return_rate * small_investment) / large_investment)
  = 0.09 := by
sorry

end NUMINAMATH_CALUDE_investment_return_calculation_l3681_368129


namespace NUMINAMATH_CALUDE_tank_capacity_l3681_368155

theorem tank_capacity : 
  ∀ (initial_fraction final_fraction added_water : ℚ),
  initial_fraction = 1/8 →
  final_fraction = 2/3 →
  added_water = 150 →
  ∃ (total_capacity : ℚ),
  (final_fraction - initial_fraction) * total_capacity = added_water ∧
  total_capacity = 3600/13 := by
sorry

end NUMINAMATH_CALUDE_tank_capacity_l3681_368155


namespace NUMINAMATH_CALUDE_triangle_area_theorem_l3681_368195

theorem triangle_area_theorem (x : ℝ) (h1 : x > 0) : 
  (1/2 : ℝ) * x * (5*x) = 36 → x = 3.6 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_theorem_l3681_368195


namespace NUMINAMATH_CALUDE_corresponding_angles_equality_incomplete_l3681_368125

-- Define the concept of corresponding angles
def corresponding_angles (α β : ℝ) : Prop := sorry

-- Define the concept of parallel lines
def parallel_lines (l₁ l₂ : Set (ℝ × ℝ)) : Prop := sorry

-- Theorem stating that the proposition "corresponding angles are equal" is false
-- when not explicitly specifying that the lines are parallel
theorem corresponding_angles_equality_incomplete :
  ¬ ∀ (α β : ℝ) (l₁ l₂ : Set (ℝ × ℝ)), corresponding_angles α β → α = β :=
sorry

end NUMINAMATH_CALUDE_corresponding_angles_equality_incomplete_l3681_368125


namespace NUMINAMATH_CALUDE_intersection_in_fourth_quadrant_l3681_368149

/-- The intersection point of two lines is in the fourth quadrant if and only if k is within a specific range -/
theorem intersection_in_fourth_quadrant (k : ℝ) :
  (∃ x y : ℝ, y = -2 * x + 3 * k + 14 ∧ x - 4 * y = -3 * k - 2 ∧ x > 0 ∧ y < 0) ↔ 
  -6 < k ∧ k < -2 := by
sorry

end NUMINAMATH_CALUDE_intersection_in_fourth_quadrant_l3681_368149


namespace NUMINAMATH_CALUDE_shortest_path_length_l3681_368166

/-- Represents a frustum of a right circular cone -/
structure ConeFrustum where
  lower_circumference : ℝ
  upper_circumference : ℝ
  inclination_angle : ℝ

/-- The shortest path from a point on the lower base to the upper base and back -/
def shortest_return_path (cf : ConeFrustum) : ℝ := sorry

theorem shortest_path_length (cf : ConeFrustum) 
  (h1 : cf.lower_circumference = 8)
  (h2 : cf.upper_circumference = 6)
  (h3 : cf.inclination_angle = π / 3) :
  shortest_return_path cf = 4 * Real.sqrt 3 / π := by sorry

end NUMINAMATH_CALUDE_shortest_path_length_l3681_368166


namespace NUMINAMATH_CALUDE_distance_QR_l3681_368156

-- Define the triangle DEF
def Triangle (D E F : ℝ × ℝ) : Prop :=
  let de := Real.sqrt ((E.1 - D.1)^2 + (E.2 - D.2)^2)
  let ef := Real.sqrt ((F.1 - E.1)^2 + (F.2 - E.2)^2)
  let df := Real.sqrt ((F.1 - D.1)^2 + (F.2 - D.2)^2)
  de = 9 ∧ ef = 12 ∧ df = 15 ∧ de^2 + ef^2 = df^2

-- Define the circle centered at Q
def CircleQ (Q D E : ℝ × ℝ) : Prop :=
  let qd := Real.sqrt ((D.1 - Q.1)^2 + (D.2 - Q.2)^2)
  let qe := Real.sqrt ((E.1 - Q.1)^2 + (E.2 - Q.2)^2)
  qd = qe

-- Define the circle centered at R
def CircleR (R D F : ℝ × ℝ) : Prop :=
  let rd := Real.sqrt ((D.1 - R.1)^2 + (D.2 - R.2)^2)
  let rf := Real.sqrt ((F.1 - R.1)^2 + (F.2 - R.2)^2)
  rd = rf

-- State the theorem
theorem distance_QR (D E F Q R : ℝ × ℝ) :
  Triangle D E F → CircleQ Q D E → CircleR R D F →
  Real.sqrt ((R.1 - Q.1)^2 + (R.2 - Q.2)^2) = 18.75 := by
  sorry

end NUMINAMATH_CALUDE_distance_QR_l3681_368156


namespace NUMINAMATH_CALUDE_inequality_of_powers_l3681_368165

theorem inequality_of_powers (a n k : ℕ) (ha : a > 1) (hnk : 0 < n ∧ n < k) :
  (a^n - 1) / n < (a^k - 1) / k := by
  sorry

end NUMINAMATH_CALUDE_inequality_of_powers_l3681_368165


namespace NUMINAMATH_CALUDE_base_value_l3681_368180

theorem base_value (some_base : ℕ) : 
  (1/2)^16 * (1/81)^8 = 1/(some_base^16) → some_base = 18 := by sorry

end NUMINAMATH_CALUDE_base_value_l3681_368180


namespace NUMINAMATH_CALUDE_average_temperature_proof_l3681_368193

theorem average_temperature_proof (temp_first_3_days : ℝ) (temp_thur_fri : ℝ) (temp_remaining : ℝ) :
  temp_first_3_days = 40 →
  temp_thur_fri = 80 →
  (3 * temp_first_3_days + 2 * temp_thur_fri + temp_remaining) / 7 = 60 := by
  sorry

#check average_temperature_proof

end NUMINAMATH_CALUDE_average_temperature_proof_l3681_368193


namespace NUMINAMATH_CALUDE_expand_product_l3681_368175

theorem expand_product (y : ℝ) : 5 * (y - 3) * (y + 10) = 5 * y^2 + 35 * y - 150 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l3681_368175


namespace NUMINAMATH_CALUDE_min_fold_length_l3681_368170

theorem min_fold_length (width height : ℝ) (hw : width = 8) (hh : height = 11) :
  let min_length := fun y : ℝ => Real.sqrt (width^2 + (y - height)^2)
  ∃ (y : ℝ), y ∈ Set.Icc 0 height ∧
    ∀ (z : ℝ), z ∈ Set.Icc 0 height → min_length y ≤ min_length z ∧
    min_length y = width :=
by sorry

end NUMINAMATH_CALUDE_min_fold_length_l3681_368170


namespace NUMINAMATH_CALUDE_quadratic_composition_roots_l3681_368101

/-- Given two quadratic trinomials f and g such that f(g(x)) = 0 and g(f(x)) = 0 have no real roots,
    at least one of f(f(x)) = 0 or g(g(x)) = 0 has no real roots. -/
theorem quadratic_composition_roots
  (f g : ℝ → ℝ)
  (hf : ∀ x, ∃ a b c : ℝ, f x = a * x^2 + b * x + c)
  (hg : ∀ x, ∃ d e f : ℝ, g x = d * x^2 + e * x + f)
  (hfg : ¬∃ x, f (g x) = 0)
  (hgf : ¬∃ x, g (f x) = 0) :
  (¬∃ x, f (f x) = 0) ∨ (¬∃ x, g (g x) = 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_composition_roots_l3681_368101


namespace NUMINAMATH_CALUDE_number_of_students_selected_l3681_368174

/-- Given a class with boys and girls, prove that the number of students selected is 3 -/
theorem number_of_students_selected
  (num_boys : ℕ)
  (num_girls : ℕ)
  (num_ways : ℕ)
  (h_boys : num_boys = 13)
  (h_girls : num_girls = 10)
  (h_ways : num_ways = 780)
  (h_combination : num_ways = (num_girls.choose 1) * (num_boys.choose 2)) :
  3 = 1 + 2 := by
  sorry

#check number_of_students_selected

end NUMINAMATH_CALUDE_number_of_students_selected_l3681_368174


namespace NUMINAMATH_CALUDE_minimum_value_quadratic_l3681_368147

theorem minimum_value_quadratic (x : ℝ) :
  (4 * x^2 + 8 * x + 3 = 5) → x ≥ -1 - Real.sqrt 6 / 2 :=
by sorry

end NUMINAMATH_CALUDE_minimum_value_quadratic_l3681_368147


namespace NUMINAMATH_CALUDE_no_delightful_eight_digit_integers_l3681_368181

/-- Represents an 8-digit positive integer as a list of its digits -/
def EightDigitInteger := List Nat

/-- Checks if a list of digits forms a valid 8-digit integer -/
def isValid (n : EightDigitInteger) : Prop :=
  n.length = 8 ∧ n.toFinset = Finset.range 9 \ {0}

/-- Checks if the sum of the first k digits is divisible by k for all k from 1 to 8 -/
def isDelightful (n : EightDigitInteger) : Prop :=
  ∀ k : Nat, k ∈ Finset.range 9 \ {0} → (n.take k).sum % k = 0

/-- The main theorem: there are no delightful 8-digit integers -/
theorem no_delightful_eight_digit_integers :
  ¬∃ n : EightDigitInteger, isValid n ∧ isDelightful n := by
  sorry

end NUMINAMATH_CALUDE_no_delightful_eight_digit_integers_l3681_368181


namespace NUMINAMATH_CALUDE_geometric_arithmetic_sequence_common_difference_l3681_368128

/-- Given a geometric sequence {a_n} where a_1+1, a_3+4, a_5+7 form an arithmetic sequence,
    the common difference of this arithmetic sequence is 3. -/
theorem geometric_arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (h_geometric : ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q)
  (h_arithmetic : ∃ d : ℝ, (a 3 + 4) - (a 1 + 1) = d ∧ (a 5 + 7) - (a 3 + 4) = d) :
  ∃ d : ℝ, (a 3 + 4) - (a 1 + 1) = d ∧ d = 3 :=
by sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_sequence_common_difference_l3681_368128


namespace NUMINAMATH_CALUDE_smallest_four_digit_all_different_l3681_368137

/-- A function that checks if a natural number has all digits different --/
def allDigitsDifferent (n : ℕ) : Prop :=
  let digits := n.digits 10
  List.Nodup digits

/-- The smallest four-digit number with all digits different --/
def smallestFourDigitAllDifferent : ℕ := 1023

/-- Theorem: 1023 is the smallest four-digit number with all digits different --/
theorem smallest_four_digit_all_different :
  (∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ allDigitsDifferent n → smallestFourDigitAllDifferent ≤ n) ∧
  1000 ≤ smallestFourDigitAllDifferent ∧
  smallestFourDigitAllDifferent < 10000 ∧
  allDigitsDifferent smallestFourDigitAllDifferent :=
sorry

end NUMINAMATH_CALUDE_smallest_four_digit_all_different_l3681_368137


namespace NUMINAMATH_CALUDE_house_rent_percentage_l3681_368140

-- Define the percentages as real numbers
def food_percentage : ℝ := 0.50
def education_percentage : ℝ := 0.15
def remaining_percentage : ℝ := 0.175

-- Define the theorem
theorem house_rent_percentage :
  let total_income : ℝ := 100
  let remaining_after_food_education : ℝ := total_income * (1 - food_percentage - education_percentage)
  let spent_on_rent : ℝ := remaining_after_food_education - (total_income * remaining_percentage)
  (spent_on_rent / remaining_after_food_education) = 0.5 := by sorry

end NUMINAMATH_CALUDE_house_rent_percentage_l3681_368140


namespace NUMINAMATH_CALUDE_box_dimensions_solution_l3681_368154

/-- Represents the dimensions of a box --/
structure BoxDimensions where
  a : ℝ
  b : ℝ
  c : ℝ
  h1 : a + c = 17
  h2 : a + b = 13
  h3 : b + c = 20
  h4 : a < b
  h5 : b < c

/-- Proves that the dimensions of the box are 5, 8, and 12 --/
theorem box_dimensions_solution (box : BoxDimensions) : 
  box.a = 5 ∧ box.b = 8 ∧ box.c = 12 := by
  sorry

end NUMINAMATH_CALUDE_box_dimensions_solution_l3681_368154


namespace NUMINAMATH_CALUDE_at_least_one_goes_probability_l3681_368171

theorem at_least_one_goes_probability 
  (prob_A prob_B : ℚ)
  (h_prob_A : prob_A = 1 / 4)
  (h_prob_B : prob_B = 2 / 5)
  (h_independent : True)  -- Assumption of independence
  : 1 - (1 - prob_A) * (1 - prob_B) = 11 / 20 :=
by sorry

end NUMINAMATH_CALUDE_at_least_one_goes_probability_l3681_368171


namespace NUMINAMATH_CALUDE_product_of_eight_consecutive_integers_divisible_by_80_l3681_368162

theorem product_of_eight_consecutive_integers_divisible_by_80 (n : ℕ) : 
  80 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4) * (n + 5) * (n + 6) * (n + 7)) := by
  sorry

end NUMINAMATH_CALUDE_product_of_eight_consecutive_integers_divisible_by_80_l3681_368162


namespace NUMINAMATH_CALUDE_nonnegative_solutions_count_l3681_368198

theorem nonnegative_solutions_count : 
  ∃! (n : ℕ), ∃ (x : ℝ), x ≥ 0 ∧ x^2 = -6*x ∧ n = 1 := by sorry

end NUMINAMATH_CALUDE_nonnegative_solutions_count_l3681_368198


namespace NUMINAMATH_CALUDE_work_completion_time_l3681_368199

-- Define the work completion times for Paul and Rose
def paul_time : ℝ := 80
def rose_time : ℝ := 120

-- Define the theorem
theorem work_completion_time : 
  let paul_rate := 1 / paul_time
  let rose_rate := 1 / rose_time
  let combined_rate := paul_rate + rose_rate
  (1 / combined_rate) = 48 := by sorry

end NUMINAMATH_CALUDE_work_completion_time_l3681_368199


namespace NUMINAMATH_CALUDE_marble_problem_l3681_368104

theorem marble_problem (M : ℕ) 
  (h1 : M > 0)
  (h2 : (M - M / 3) / 4 > 0)
  (h3 : M - M / 3 - (M - M / 3) / 4 - 2 * ((M - M / 3) / 4) = 7) : 
  M = 42 := by
sorry

end NUMINAMATH_CALUDE_marble_problem_l3681_368104


namespace NUMINAMATH_CALUDE_positive_sum_greater_than_abs_difference_l3681_368134

theorem positive_sum_greater_than_abs_difference (x y : ℝ) :
  x + y > |x - y| ↔ x > 0 ∧ y > 0 := by sorry

end NUMINAMATH_CALUDE_positive_sum_greater_than_abs_difference_l3681_368134


namespace NUMINAMATH_CALUDE_largest_absolute_value_l3681_368132

theorem largest_absolute_value : let S : Finset Int := {2, 3, -3, -4}
  ∃ x ∈ S, ∀ y ∈ S, |y| ≤ |x| ∧ x = -4 := by
  sorry

end NUMINAMATH_CALUDE_largest_absolute_value_l3681_368132


namespace NUMINAMATH_CALUDE_parallelogram_height_l3681_368176

theorem parallelogram_height (area base height : ℝ) : 
  area = 96 ∧ base = 12 ∧ area = base * height → height = 8 := by sorry

end NUMINAMATH_CALUDE_parallelogram_height_l3681_368176


namespace NUMINAMATH_CALUDE_complex_equality_implies_modulus_l3681_368143

/-- The imaginary unit -/
noncomputable def i : ℂ := Complex.I

/-- Given that (1+i)x = 1+yi, where x and y are real numbers and i is the imaginary unit,
    prove that |x+yi| = √2 -/
theorem complex_equality_implies_modulus (x y : ℝ) 
  (h : (1 + i) * (x : ℂ) = 1 + y * i) : 
  Complex.abs (x + y * i) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equality_implies_modulus_l3681_368143


namespace NUMINAMATH_CALUDE_rowing_speed_problem_l3681_368117

/-- The rowing speed problem -/
theorem rowing_speed_problem (v c : ℝ) (h1 : c = 1.1)
  (h2 : (v + c) * t = (v - c) * (2 * t) → t ≠ 0) : v = 3.3 := by
  sorry

end NUMINAMATH_CALUDE_rowing_speed_problem_l3681_368117


namespace NUMINAMATH_CALUDE_speed_ratio_proof_l3681_368124

/-- The speed of A in yards per minute -/
def speed_A : ℝ := 333.33

/-- The speed of B in yards per minute -/
def speed_B : ℝ := 433.33

/-- The initial distance of B from point O in yards -/
def initial_distance_B : ℝ := 1000

/-- The time when A and B are first equidistant from O in minutes -/
def time_first_equidistant : ℝ := 3

/-- The time when A and B are second equidistant from O in minutes -/
def time_second_equidistant : ℝ := 10

theorem speed_ratio_proof :
  (∀ t : ℝ, t = time_first_equidistant → 
    (speed_A * t)^2 = (initial_distance_B - speed_B * t)^2) ∧
  (∀ t : ℝ, t = time_second_equidistant → 
    (speed_A * t)^2 = (initial_distance_B - speed_B * t)^2) →
  speed_A / speed_B = 333 / 433 := by
  sorry

end NUMINAMATH_CALUDE_speed_ratio_proof_l3681_368124


namespace NUMINAMATH_CALUDE_lcm_of_primes_l3681_368100

theorem lcm_of_primes (p₁ p₂ p₃ p₄ : Nat) (h₁ : p₁ = 97) (h₂ : p₂ = 193) (h₃ : p₃ = 419) (h₄ : p₄ = 673) :
  Nat.lcm p₁ (Nat.lcm p₂ (Nat.lcm p₃ p₄)) = 5280671387 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_primes_l3681_368100


namespace NUMINAMATH_CALUDE_ice_cream_sundaes_l3681_368161

/-- The number of ice cream flavors -/
def n : ℕ := 8

/-- The number of scoops in each sundae -/
def k : ℕ := 2

/-- The number of unique two scoop sundaes -/
def unique_sundaes : ℕ := Nat.choose n k

theorem ice_cream_sundaes :
  unique_sundaes = 28 := by sorry

end NUMINAMATH_CALUDE_ice_cream_sundaes_l3681_368161


namespace NUMINAMATH_CALUDE_two_digit_number_theorem_l3681_368109

/-- Represents a two-digit integer with its tens and units digits. -/
structure TwoDigitNumber where
  tens : ℕ
  units : ℕ
  is_valid : tens ≥ 1 ∧ tens ≤ 9 ∧ units ≤ 9

/-- Theorem stating the relationship between j and k for a two-digit number. -/
theorem two_digit_number_theorem (n : TwoDigitNumber) (k j : ℚ) :
  (10 * n.tens + n.units : ℚ) = k * (n.tens + n.units) →
  (20 * n.units + n.tens : ℚ) = j * (n.tens + n.units) →
  j = (199 + k) / 10 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_theorem_l3681_368109


namespace NUMINAMATH_CALUDE_ellipse_problem_l3681_368182

def given_ellipse (x y : ℝ) : Prop :=
  8 * x^2 / 81 + y^2 / 36 = 1

def reference_ellipse (x y : ℝ) : Prop :=
  x^2 / 9 + y^2 / 4 = 1

def required_ellipse (x y : ℝ) : Prop :=
  x^2 / 15 + y^2 / 10 = 1

theorem ellipse_problem (x₀ : ℝ) (h1 : given_ellipse x₀ 2) (h2 : x₀ < 0) :
  x₀ = -3 ∧
  ∀ (x y : ℝ), (x = x₀ ∧ y = 2 → required_ellipse x y) ∧
  (∃ (c : ℝ), ∀ (x y : ℝ), reference_ellipse x y ↔ x^2 + y^2 = 9 + 4 - c^2 ∧ c^2 = 5) ∧
  (∃ (c : ℝ), ∀ (x y : ℝ), required_ellipse x y ↔ x^2 + y^2 = 15 + 10 - c^2 ∧ c^2 = 5) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_problem_l3681_368182


namespace NUMINAMATH_CALUDE_sheila_hourly_rate_l3681_368131

/-- Represents Sheila's work schedule and earnings --/
structure WorkSchedule where
  mon_wed_fri_hours : ℕ  -- Hours worked on Monday, Wednesday, and Friday
  tue_thu_hours : ℕ      -- Hours worked on Tuesday and Thursday
  weekly_earnings : ℕ    -- Weekly earnings in dollars

/-- Calculates the total hours worked in a week --/
def total_hours (schedule : WorkSchedule) : ℕ :=
  3 * schedule.mon_wed_fri_hours + 2 * schedule.tue_thu_hours

/-- Calculates the hourly rate --/
def hourly_rate (schedule : WorkSchedule) : ℚ :=
  schedule.weekly_earnings / (total_hours schedule)

/-- Sheila's work schedule --/
def sheila_schedule : WorkSchedule :=
  { mon_wed_fri_hours := 8
  , tue_thu_hours := 6
  , weekly_earnings := 288 }

/-- Theorem stating that Sheila's hourly rate is $8 --/
theorem sheila_hourly_rate :
  hourly_rate sheila_schedule = 8 := by
  sorry

end NUMINAMATH_CALUDE_sheila_hourly_rate_l3681_368131


namespace NUMINAMATH_CALUDE_largest_n_polynomials_l3681_368138

/-- A type representing real polynomials -/
def RealPolynomial := ℝ → ℝ

/-- Predicate to check if a real polynomial has no real roots -/
def HasNoRealRoots (p : RealPolynomial) : Prop :=
  ∀ x : ℝ, p x ≠ 0

/-- Predicate to check if a real polynomial has at least one real root -/
def HasRealRoot (p : RealPolynomial) : Prop :=
  ∃ x : ℝ, p x = 0

/-- The main theorem statement -/
theorem largest_n_polynomials :
  (∃ (n : ℕ) (P : Fin n → RealPolynomial),
    (∀ (i j : Fin n) (hij : i ≠ j), HasNoRealRoots (fun x ↦ P i x + P j x)) ∧
    (∀ (i j k : Fin n) (hijk : i ≠ j ∧ j ≠ k ∧ i ≠ k), HasRealRoot (fun x ↦ P i x + P j x + P k x))) →
  (∃ (P : Fin 3 → RealPolynomial),
    (∀ (i j : Fin 3) (hij : i ≠ j), HasNoRealRoots (fun x ↦ P i x + P j x)) ∧
    (∀ (i j k : Fin 3) (hijk : i ≠ j ∧ j ≠ k ∧ i ≠ k), HasRealRoot (fun x ↦ P i x + P j x + P k x))) ∧
  (∀ (n : ℕ) (hn : n > 3),
    ¬∃ (P : Fin n → RealPolynomial),
      (∀ (i j : Fin n) (hij : i ≠ j), HasNoRealRoots (fun x ↦ P i x + P j x)) ∧
      (∀ (i j k : Fin n) (hijk : i ≠ j ∧ j ≠ k ∧ i ≠ k), HasRealRoot (fun x ↦ P i x + P j x + P k x))) :=
by
  sorry

end NUMINAMATH_CALUDE_largest_n_polynomials_l3681_368138


namespace NUMINAMATH_CALUDE_decoration_time_is_320_l3681_368168

/-- Represents the time in minutes for a single step in nail decoration -/
def step_time : ℕ := 20

/-- Represents the time in minutes for pattern creation -/
def pattern_time : ℕ := 40

/-- Represents the number of coating steps (base, paint, glitter) -/
def num_coats : ℕ := 3

/-- Represents the number of people getting their nails decorated -/
def num_people : ℕ := 2

/-- Calculates the total time for nail decoration -/
def total_decoration_time : ℕ :=
  num_people * (2 * num_coats * step_time + pattern_time)

/-- Theorem stating that the total decoration time is 320 minutes -/
theorem decoration_time_is_320 :
  total_decoration_time = 320 :=
sorry

end NUMINAMATH_CALUDE_decoration_time_is_320_l3681_368168
