import Mathlib

namespace NUMINAMATH_CALUDE_nearest_integer_to_x_plus_2y_l2001_200141

theorem nearest_integer_to_x_plus_2y (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : |x| + 2 * y = 6) (h2 : |x| * y + x^3 = 2) :
  ∃ (n : ℤ), n = 6 ∧ ∀ (m : ℤ), |x + 2 * y - ↑n| ≤ |x + 2 * y - ↑m| :=
sorry

end NUMINAMATH_CALUDE_nearest_integer_to_x_plus_2y_l2001_200141


namespace NUMINAMATH_CALUDE_baking_time_ratio_l2001_200189

def usual_assembly_time : ℝ := 1
def usual_baking_time : ℝ := 1.5
def usual_decorating_time : ℝ := 1
def total_time_on_failed_day : ℝ := 5

theorem baking_time_ratio :
  let usual_total_time := usual_assembly_time + usual_baking_time + usual_decorating_time
  let baking_time_on_failed_day := total_time_on_failed_day - usual_assembly_time - usual_decorating_time
  baking_time_on_failed_day / usual_baking_time = 2 := by
sorry

end NUMINAMATH_CALUDE_baking_time_ratio_l2001_200189


namespace NUMINAMATH_CALUDE_cakes_left_l2001_200102

def cakes_per_day : ℕ := 20
def baking_days : ℕ := 9
def total_cakes : ℕ := cakes_per_day * baking_days
def sold_cakes : ℕ := total_cakes / 2
def remaining_cakes : ℕ := total_cakes - sold_cakes

theorem cakes_left : remaining_cakes = 90 := by
  sorry

end NUMINAMATH_CALUDE_cakes_left_l2001_200102


namespace NUMINAMATH_CALUDE_pythagorean_chord_l2001_200168

theorem pythagorean_chord (m : ℕ) (h : m ≥ 3) : 
  let width := 2 * m
  let height := m^2 - 1
  let diagonal := height + 2
  width^2 + height^2 = diagonal^2 ∧ diagonal = m^2 + 1 := by
sorry

end NUMINAMATH_CALUDE_pythagorean_chord_l2001_200168


namespace NUMINAMATH_CALUDE_min_value_a5_plus_a6_l2001_200177

/-- A positive arithmetic geometric sequence -/
def ArithmeticGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ (r d : ℝ), r > 1 ∧ d > 0 ∧ ∀ n, a n > 0 ∧ a (n + 1) = r * a n + d

theorem min_value_a5_plus_a6 (a : ℕ → ℝ) :
  ArithmeticGeometricSequence a →
  a 4 + a 3 - 2 * a 2 - 2 * a 1 = 6 →
  ∃ (min : ℝ), min = 48 ∧ ∀ x, (∃ b, ArithmeticGeometricSequence b ∧ 
    b 4 + b 3 - 2 * b 2 - 2 * b 1 = 6 ∧ b 5 + b 6 = x) → x ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_a5_plus_a6_l2001_200177


namespace NUMINAMATH_CALUDE_gold_found_per_hour_l2001_200140

def diving_time : ℕ := 8
def chest_gold : ℕ := 100
def num_small_bags : ℕ := 2

def gold_per_hour : ℚ :=
  let small_bag_gold := chest_gold / 2
  let total_gold := chest_gold + num_small_bags * small_bag_gold
  total_gold / diving_time

theorem gold_found_per_hour :
  gold_per_hour = 25 := by sorry

end NUMINAMATH_CALUDE_gold_found_per_hour_l2001_200140


namespace NUMINAMATH_CALUDE_jihoons_class_size_l2001_200100

theorem jihoons_class_size :
  ∃! n : ℕ, 35 < n ∧ n < 70 ∧ n % 6 = 3 ∧ n % 8 = 1 ∧ n = 57 := by
  sorry

end NUMINAMATH_CALUDE_jihoons_class_size_l2001_200100


namespace NUMINAMATH_CALUDE_small_cubes_in_large_cube_l2001_200113

/-- Converts decimeters to centimeters -/
def dm_to_cm (dm : ℕ) : ℕ := dm * 10

/-- Calculates the number of small cubes that fit in a large cube -/
def num_small_cubes (large_side_dm : ℕ) (small_side_cm : ℕ) : ℕ :=
  let large_side_cm := dm_to_cm large_side_dm
  let num_cubes_per_edge := large_side_cm / small_side_cm
  num_cubes_per_edge ^ 3

theorem small_cubes_in_large_cube :
  num_small_cubes 8 4 = 8000 := by
  sorry

end NUMINAMATH_CALUDE_small_cubes_in_large_cube_l2001_200113


namespace NUMINAMATH_CALUDE_problem_solution_l2001_200172

theorem problem_solution : 
  let M : ℤ := 2007 / 3
  let N : ℤ := M / 3
  let X : ℤ := M - N
  X = 446 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l2001_200172


namespace NUMINAMATH_CALUDE_terminating_decimal_expansion_of_13_375_l2001_200191

theorem terminating_decimal_expansion_of_13_375 :
  ∃ (n : ℕ) (k : ℕ), (13 : ℚ) / 375 = (34666 : ℚ) / 10^6 + k / (10^6 * 10^n) :=
sorry

end NUMINAMATH_CALUDE_terminating_decimal_expansion_of_13_375_l2001_200191


namespace NUMINAMATH_CALUDE_friends_new_games_l2001_200173

theorem friends_new_games (katie_new : ℕ) (total_new : ℕ) (h1 : katie_new = 84) (h2 : total_new = 92) :
  total_new - katie_new = 8 := by
  sorry

end NUMINAMATH_CALUDE_friends_new_games_l2001_200173


namespace NUMINAMATH_CALUDE_smallest_n_with_conditions_n_3584_satisfies_conditions_smallest_n_is_3584_l2001_200153

/-- A function to check if a number contains the digit 9 -/
def contains_nine (n : ℕ) : Prop :=
  ∃ k : ℕ, n / (10^k) % 10 = 9

/-- A function to check if a fraction terminates -/
def is_terminating (n : ℕ) : Prop :=
  ∃ a b : ℕ, n = 2^a * 5^b

theorem smallest_n_with_conditions :
  ∀ n : ℕ, n > 0 →
    (is_terminating n ∧ contains_nine n ∧ n % 7 = 0) →
    n ≥ 3584 :=
sorry

theorem n_3584_satisfies_conditions :
  is_terminating 3584 ∧ contains_nine 3584 ∧ 3584 % 7 = 0 :=
sorry

theorem smallest_n_is_3584 :
  ∀ n : ℕ, n > 0 →
    (is_terminating n ∧ contains_nine n ∧ n % 7 = 0) →
    n = 3584 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_with_conditions_n_3584_satisfies_conditions_smallest_n_is_3584_l2001_200153


namespace NUMINAMATH_CALUDE_stock_value_return_l2001_200106

theorem stock_value_return (initial_value : ℝ) (h : initial_value > 0) :
  let first_year_value := initial_value * 1.4
  let second_year_decrease := 2 / 7
  first_year_value * (1 - second_year_decrease) = initial_value :=
by sorry

end NUMINAMATH_CALUDE_stock_value_return_l2001_200106


namespace NUMINAMATH_CALUDE_min_value_implies_a_l2001_200186

/-- The function f(x) defined as |x+1| + |2x+a| -/
def f (a : ℝ) (x : ℝ) : ℝ := |x + 1| + |2*x + a|

/-- The theorem stating that if the minimum value of f(x) is 3, then a = -4 or a = 8 -/
theorem min_value_implies_a (a : ℝ) : (∀ x, f a x ≥ 3) ∧ (∃ x, f a x = 3) → a = -4 ∨ a = 8 := by
  sorry


end NUMINAMATH_CALUDE_min_value_implies_a_l2001_200186


namespace NUMINAMATH_CALUDE_locus_of_angle_bisector_intersection_l2001_200138

-- Define the points and constants
variable (a : ℝ) -- distance between A and B
variable (x₀ y₀ : ℝ) -- coordinates of point C
variable (x y : ℝ) -- coordinates of point P

-- State the theorem
theorem locus_of_angle_bisector_intersection 
  (h1 : a > 0) -- A and B are distinct points
  (h2 : x₀^2 + y₀^2 = 1) -- C is on the unit circle centered at A
  (h3 : x = (a * x₀) / (1 + a)) -- x-coordinate of P
  (h4 : y = (a * y₀) / (1 + a)) -- y-coordinate of P
  : x^2 + y^2 = (a^2) / ((1 + a)^2) := by
  sorry

end NUMINAMATH_CALUDE_locus_of_angle_bisector_intersection_l2001_200138


namespace NUMINAMATH_CALUDE_quadratic_rational_solutions_l2001_200124

/-- For a positive integer k, the equation kx^2 + 30x + k = 0 has rational solutions
    if and only if k = 9 or k = 15. -/
theorem quadratic_rational_solutions (k : ℕ+) :
  (∃ x : ℚ, k * x^2 + 30 * x + k = 0) ↔ k = 9 ∨ k = 15 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_rational_solutions_l2001_200124


namespace NUMINAMATH_CALUDE_sum_of_x_coordinates_l2001_200111

/-- Given three points X, Y, and Z in a plane satisfying certain conditions, 
    prove that the sum of X's coordinates is 34. -/
theorem sum_of_x_coordinates (X Y Z : ℝ × ℝ) : 
  (dist X Z) / (dist X Y) = 2/3 →
  (dist Z Y) / (dist X Y) = 1/3 →
  Y = (1, 9) →
  Z = (-1, 3) →
  X.1 + X.2 = 34 := by sorry


end NUMINAMATH_CALUDE_sum_of_x_coordinates_l2001_200111


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l2001_200104

/-- The perimeter of a rhombus with diagonals measuring 24 feet and 10 feet is 52 feet. -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 24) (h2 : d2 = 10) : 
  4 * Real.sqrt ((d1/2)^2 + (d2/2)^2) = 52 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l2001_200104


namespace NUMINAMATH_CALUDE_coinciding_rest_days_l2001_200158

/-- Al's schedule cycle length -/
def al_cycle : ℕ := 7

/-- Barb's schedule cycle length -/
def barb_cycle : ℕ := 5

/-- Total number of days -/
def total_days : ℕ := 1200

/-- Number of rest days Al has in one cycle -/
def al_rest_days : ℕ := 2

/-- Number of rest days Barb has in one cycle -/
def barb_rest_days : ℕ := 1

/-- The theorem to prove -/
theorem coinciding_rest_days : 
  ∃ (n : ℕ), n = (total_days / (Nat.lcm al_cycle barb_cycle)) * 
    (al_rest_days * barb_rest_days) ∧ n = 34 := by
  sorry

end NUMINAMATH_CALUDE_coinciding_rest_days_l2001_200158


namespace NUMINAMATH_CALUDE_student_test_score_l2001_200187

theorem student_test_score (max_marks : ℕ) (pass_percentage : ℚ) (fail_margin : ℕ) : 
  max_marks = 300 → 
  pass_percentage = 60 / 100 → 
  fail_margin = 100 → 
  ∃ (student_score : ℕ), 
    student_score = max_marks * pass_percentage - fail_margin ∧ 
    student_score = 80 := by
  sorry

end NUMINAMATH_CALUDE_student_test_score_l2001_200187


namespace NUMINAMATH_CALUDE_gold_coin_percentage_is_48_percent_l2001_200192

/-- Represents the composition of objects in an urn --/
structure UrnComposition where
  bead_percentage : ℝ
  silver_coin_percentage : ℝ
  gold_coin_percentage : ℝ

/-- Calculates the percentage of gold coins in the urn --/
def gold_coin_percentage (u : UrnComposition) : ℝ :=
  u.gold_coin_percentage

/-- The urn composition satisfies the given conditions --/
def valid_urn_composition (u : UrnComposition) : Prop :=
  u.bead_percentage = 0.2 ∧
  u.silver_coin_percentage + u.gold_coin_percentage = 0.8 ∧
  u.silver_coin_percentage = 0.4 * (u.silver_coin_percentage + u.gold_coin_percentage)

theorem gold_coin_percentage_is_48_percent (u : UrnComposition) 
  (h : valid_urn_composition u) : gold_coin_percentage u = 0.48 := by
  sorry

end NUMINAMATH_CALUDE_gold_coin_percentage_is_48_percent_l2001_200192


namespace NUMINAMATH_CALUDE_min_value_exponential_sum_l2001_200154

theorem min_value_exponential_sum (x y : ℝ) (h : x + y = 5) :
  3^x + 3^y ≥ 18 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_exponential_sum_l2001_200154


namespace NUMINAMATH_CALUDE_nth_ring_area_l2001_200193

/-- Represents the area of a ring in a square garden system -/
def ring_area (n : ℕ) : ℝ :=
  36 * n

/-- Theorem stating the area of the nth ring in a square garden system -/
theorem nth_ring_area (n : ℕ) :
  ring_area n = 36 * n :=
by
  -- The proof goes here
  sorry

/-- The area of the 50th ring -/
def area_50th_ring : ℝ := ring_area 50

#eval area_50th_ring  -- Should evaluate to 1800

end NUMINAMATH_CALUDE_nth_ring_area_l2001_200193


namespace NUMINAMATH_CALUDE_cricket_average_l2001_200199

theorem cricket_average (current_innings : Nat) (next_innings_runs : Nat) (average_increase : Nat) (current_average : Nat) : 
  current_innings = 20 →
  next_innings_runs = 116 →
  average_increase = 4 →
  (current_innings * current_average + next_innings_runs) / (current_innings + 1) = current_average + average_increase →
  current_average = 32 := by
sorry

end NUMINAMATH_CALUDE_cricket_average_l2001_200199


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_from_max_ratio_l2001_200125

/-- An ellipse with foci F₁ and F₂ -/
structure Ellipse where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ

/-- The eccentricity of an ellipse -/
def eccentricity (e : Ellipse) : ℝ := sorry

/-- The distance between two points in ℝ² -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

theorem ellipse_eccentricity_from_max_ratio (e : Ellipse) :
  (∀ p : ℝ × ℝ, ∃ q : ℝ × ℝ, distance p e.F₁ / distance p e.F₂ ≤ distance q e.F₁ / distance q e.F₂) →
  (∃ p : ℝ × ℝ, distance p e.F₁ / distance p e.F₂ = 3) →
  eccentricity e = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_from_max_ratio_l2001_200125


namespace NUMINAMATH_CALUDE_probability_of_drawing_math_books_l2001_200180

/-- The number of Chinese books -/
def chinese_books : ℕ := 10

/-- The number of math books -/
def math_books : ℕ := 2

/-- The total number of books -/
def total_books : ℕ := chinese_books + math_books

/-- The number of books to be drawn -/
def books_drawn : ℕ := 2

theorem probability_of_drawing_math_books :
  (Nat.choose total_books books_drawn - Nat.choose chinese_books books_drawn) / Nat.choose total_books books_drawn = 7 / 22 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_drawing_math_books_l2001_200180


namespace NUMINAMATH_CALUDE_compound_molecular_weight_l2001_200167

-- Define atomic weights
def atomic_weight_Al : ℝ := 26.98
def atomic_weight_P : ℝ := 30.97
def atomic_weight_O : ℝ := 16.00

-- Define the number of atoms for each element
def num_Al : ℕ := 1
def num_P : ℕ := 1
def num_O : ℕ := 4

-- Define the molecular weight calculation function
def molecular_weight (w_Al w_P w_O : ℝ) (n_Al n_P n_O : ℕ) : ℝ :=
  w_Al * n_Al + w_P * n_P + w_O * n_O

-- Theorem statement
theorem compound_molecular_weight :
  molecular_weight atomic_weight_Al atomic_weight_P atomic_weight_O num_Al num_P num_O = 121.95 := by
  sorry

end NUMINAMATH_CALUDE_compound_molecular_weight_l2001_200167


namespace NUMINAMATH_CALUDE_theodore_tax_rate_l2001_200132

/-- Calculates the tax rate for Theodore's statue business --/
theorem theodore_tax_rate :
  let stone_statues : ℕ := 10
  let wooden_statues : ℕ := 20
  let stone_price : ℚ := 20
  let wooden_price : ℚ := 5
  let after_tax_earnings : ℚ := 270
  let before_tax_earnings := stone_statues * stone_price + wooden_statues * wooden_price
  let tax_rate := (before_tax_earnings - after_tax_earnings) / before_tax_earnings
  tax_rate = 1/10 := by sorry

end NUMINAMATH_CALUDE_theodore_tax_rate_l2001_200132


namespace NUMINAMATH_CALUDE_smallest_power_is_four_l2001_200126

def rotation_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![0, -1],
    ![1,  0]]

theorem smallest_power_is_four :
  (∀ k : ℕ, k > 0 ∧ k < 4 → rotation_matrix ^ k ≠ 1) ∧
  rotation_matrix ^ 4 = 1 :=
sorry

end NUMINAMATH_CALUDE_smallest_power_is_four_l2001_200126


namespace NUMINAMATH_CALUDE_train_length_l2001_200161

/-- The length of a train given its speed and time to cross an electric pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 90 →
  time_s = 3.9996800255979523 →
  ∃ (length_m : ℝ), abs (length_m - 99.992) < 0.001 ∧ 
    length_m = speed_kmh * (1000 / 3600) * time_s :=
by sorry

end NUMINAMATH_CALUDE_train_length_l2001_200161


namespace NUMINAMATH_CALUDE_sum_mod_nine_l2001_200169

theorem sum_mod_nine : (2 + 44 + 666 + 8888 + 111110 + 13131312 + 1414141414) % 9 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_mod_nine_l2001_200169


namespace NUMINAMATH_CALUDE_same_color_sock_pairs_l2001_200159

/-- The number of ways to choose a pair of socks of the same color from a drawer -/
theorem same_color_sock_pairs (white brown blue red : ℕ) 
  (h_white : white = 5)
  (h_brown : brown = 6)
  (h_blue : blue = 3)
  (h_red : red = 2) : 
  Nat.choose white 2 + Nat.choose brown 2 + Nat.choose blue 2 + Nat.choose red 2 = 29 := by
  sorry

end NUMINAMATH_CALUDE_same_color_sock_pairs_l2001_200159


namespace NUMINAMATH_CALUDE_max_product_sum_2024_l2001_200166

theorem max_product_sum_2024 : 
  (∃ (a b : ℤ), a + b = 2024 ∧ a * b = 1024144) ∧ 
  (∀ (x y : ℤ), x + y = 2024 → x * y ≤ 1024144) := by
  sorry

end NUMINAMATH_CALUDE_max_product_sum_2024_l2001_200166


namespace NUMINAMATH_CALUDE_smallest_integer_fraction_l2001_200179

theorem smallest_integer_fraction (y : ℤ) : (8 : ℚ) / 12 < (y : ℚ) / 15 ↔ 11 ≤ y := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_fraction_l2001_200179


namespace NUMINAMATH_CALUDE_quadratic_function_max_value_l2001_200176

theorem quadratic_function_max_value (m n : ℝ) : 
  m^2 - 4*n ≥ 0 →
  (m - 1)^2 + (n - 1)^2 + (m - n)^2 ≤ 9/8 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_max_value_l2001_200176


namespace NUMINAMATH_CALUDE_box_price_difference_l2001_200162

/-- The price difference between box C and box A -/
def price_difference (a b c : ℚ) : ℚ := c - a

/-- The conditions from the problem -/
def problem_conditions (a b c : ℚ) : Prop :=
  a + b + c = 9 ∧ 3*a + 2*b + c = 16

theorem box_price_difference :
  ∀ a b c : ℚ, problem_conditions a b c → price_difference a b c = 2 := by
  sorry

end NUMINAMATH_CALUDE_box_price_difference_l2001_200162


namespace NUMINAMATH_CALUDE_fraction_subtraction_l2001_200130

theorem fraction_subtraction (a : ℝ) (ha : a ≠ 0) : 1 / a - 3 / a = -2 / a := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l2001_200130


namespace NUMINAMATH_CALUDE_system_solution_unique_l2001_200152

def system_solution (a₁ a₂ a₃ a₄ : ℝ) (x₁ x₂ x₃ x₄ : ℝ) : Prop :=
  a₁ < a₂ ∧ a₂ < a₃ ∧ a₃ < a₄ ∧
  (|a₁ - a₂| * x₂ + |a₁ - a₃| * x₃ + |a₁ - a₄| * x₄ = 1) ∧
  (|a₂ - a₁| * x₁ + |a₂ - a₃| * x₃ + |a₂ - a₄| * x₄ = 1) ∧
  (|a₃ - a₁| * x₁ + |a₃ - a₂| * x₂ + |a₃ - a₄| * x₄ = 1) ∧
  (|a₄ - a₁| * x₁ + |a₄ - a₂| * x₂ + |a₄ - a₃| * x₃ = 1)

theorem system_solution_unique (a₁ a₂ a₃ a₄ : ℝ) :
  ∃! (x₁ x₂ x₃ x₄ : ℝ), system_solution a₁ a₂ a₃ a₄ x₁ x₂ x₃ x₄ ∧
    x₁ = 1 / (a₄ - a₁) ∧ x₂ = 0 ∧ x₃ = 0 ∧ x₄ = 1 / (a₄ - a₁) :=
by
  sorry

end NUMINAMATH_CALUDE_system_solution_unique_l2001_200152


namespace NUMINAMATH_CALUDE_binomial_probability_one_l2001_200164

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h_p_range : 0 ≤ p ∧ p ≤ 1

/-- The expectation of a binomial random variable -/
def expectation (X : BinomialRV) : ℝ := X.n * X.p

/-- The probability of a binomial random variable taking a specific value -/
def probability (X : BinomialRV) (k : ℕ) : ℝ :=
  (X.n.choose k) * (X.p ^ k) * ((1 - X.p) ^ (X.n - k))

theorem binomial_probability_one (η : BinomialRV) 
  (h_p : η.p = 0.6) 
  (h_expectation : expectation η = 3) :
  probability η 1 = 3 * 0.4^4 := by
  sorry

end NUMINAMATH_CALUDE_binomial_probability_one_l2001_200164


namespace NUMINAMATH_CALUDE_candy_redistribution_theorem_l2001_200196

/-- Represents the number of candies each friend has at a given stage -/
structure CandyState where
  vasya : ℕ
  petya : ℕ
  kolya : ℕ

/-- Represents a round of candy redistribution -/
def redistribute (state : CandyState) (giver : Fin 3) : CandyState :=
  match giver with
  | 0 => ⟨state.vasya - (state.petya + state.kolya), 2 * state.petya, 2 * state.kolya⟩
  | 1 => ⟨2 * state.vasya, state.petya - (state.vasya + state.kolya), 2 * state.kolya⟩
  | 2 => ⟨2 * state.vasya, 2 * state.petya, state.kolya - (state.vasya + state.petya)⟩

theorem candy_redistribution_theorem (initial : CandyState) :
  initial.kolya = 36 →
  (redistribute (redistribute (redistribute initial 0) 1) 2).kolya = 36 →
  initial.vasya + initial.petya + initial.kolya = 252 := by
  sorry


end NUMINAMATH_CALUDE_candy_redistribution_theorem_l2001_200196


namespace NUMINAMATH_CALUDE_simplify_expression_l2001_200109

theorem simplify_expression (x : ℝ) : x - 2*(1+x) + 3*(1-x) - 4*(1+2*x) = -12*x - 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2001_200109


namespace NUMINAMATH_CALUDE_binary_1011011_equals_base7_160_l2001_200112

/-- Converts a binary number represented as a list of bits to a natural number. -/
def binary_to_nat (bits : List Bool) : ℕ :=
  bits.foldr (fun b n => 2 * n + if b then 1 else 0) 0

/-- Converts a natural number to its representation in base 7. -/
def nat_to_base7 (n : ℕ) : List ℕ :=
  if n < 7 then [n]
  else (n % 7) :: nat_to_base7 (n / 7)

/-- The binary representation of 1011011. -/
def binary_1011011 : List Bool :=
  [true, false, true, true, false, true, true]

/-- The base 7 representation of 160. -/
def base7_160 : List ℕ :=
  [0, 6, 1]

theorem binary_1011011_equals_base7_160 :
  nat_to_base7 (binary_to_nat binary_1011011) = base7_160 := by
  sorry

#eval binary_to_nat binary_1011011
#eval nat_to_base7 (binary_to_nat binary_1011011)

end NUMINAMATH_CALUDE_binary_1011011_equals_base7_160_l2001_200112


namespace NUMINAMATH_CALUDE_reservoir_capacity_l2001_200101

theorem reservoir_capacity (x : ℝ) 
  (h1 : x > 0) -- Ensure the capacity is positive
  (h2 : (1/4) * x + 100 = (3/8) * x) -- Condition from initial state to final state
  : x = 800 := by
  sorry

end NUMINAMATH_CALUDE_reservoir_capacity_l2001_200101


namespace NUMINAMATH_CALUDE_spadesuit_example_l2001_200147

-- Define the spadesuit operation
def spadesuit (a b : ℝ) : ℝ := |a^2 - b^2|

-- Theorem statement
theorem spadesuit_example : spadesuit 3 (spadesuit 5 2) = 432 := by
  sorry

end NUMINAMATH_CALUDE_spadesuit_example_l2001_200147


namespace NUMINAMATH_CALUDE_transformed_graph_point_l2001_200135

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the theorem
theorem transformed_graph_point (h : f 4 = 7) : 
  ∃ (x y : ℝ), 2 * y = 3 * f (4 * x) + 5 ∧ x + y = 14 := by
sorry

end NUMINAMATH_CALUDE_transformed_graph_point_l2001_200135


namespace NUMINAMATH_CALUDE_no_real_solution_l2001_200119

theorem no_real_solution :
  ¬∃ (x : ℝ), 4 * (3 * x)^2 + (3 * x) + 3 = 2 * (9 * x^2 + (3 * x) + 1) := by
  sorry

end NUMINAMATH_CALUDE_no_real_solution_l2001_200119


namespace NUMINAMATH_CALUDE_speed_of_k_l2001_200110

-- Define the speeds and time delay
def speed_a : ℝ := 30
def speed_b : ℝ := 40
def delay : ℝ := 5

-- Define the theorem
theorem speed_of_k (speed_k : ℝ) : 
  -- a, b, k start from the same place and travel in the same direction
  -- a travels at speed_a km/hr
  -- b travels at speed_b km/hr
  -- b starts delay hours after a
  -- b and k overtake a at the same instant
  -- k starts at the same time as a
  (∃ (t : ℝ), t > 0 ∧ 
    speed_b * t = speed_a * (t + delay) ∧
    speed_k * (t + delay) = speed_a * (t + delay)) →
  -- Then the speed of k is 35 km/hr
  speed_k = 35 := by
sorry

end NUMINAMATH_CALUDE_speed_of_k_l2001_200110


namespace NUMINAMATH_CALUDE_lcm_of_20_28_45_l2001_200194

theorem lcm_of_20_28_45 : Nat.lcm (Nat.lcm 20 28) 45 = 1260 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_20_28_45_l2001_200194


namespace NUMINAMATH_CALUDE_sum_of_largest_and_smallest_prime_factors_l2001_200170

def number : ℕ := 1386

theorem sum_of_largest_and_smallest_prime_factors :
  ∃ (smallest largest : ℕ),
    smallest.Prime ∧ 
    largest.Prime ∧
    smallest ∣ number ∧ 
    largest ∣ number ∧
    (∀ p : ℕ, p.Prime → p ∣ number → p ≤ largest) ∧
    (∀ p : ℕ, p.Prime → p ∣ number → p ≥ smallest) ∧
    smallest + largest = 13 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_largest_and_smallest_prime_factors_l2001_200170


namespace NUMINAMATH_CALUDE_problem_equivalence_l2001_200143

theorem problem_equivalence (y x : ℝ) (h : x ≠ -1) :
  (y + 2) * (y - 2) + (y - 1) * (y + 3) = 2 * y^2 + 2 * y - 7 ∧
  (1 + 2 / (x + 1)) / ((x^2 + 6*x + 9) / (x + 1)) = 1 / (x + 3) := by
  sorry

end NUMINAMATH_CALUDE_problem_equivalence_l2001_200143


namespace NUMINAMATH_CALUDE_cubic_roots_property_l2001_200165

theorem cubic_roots_property (a b c t u v : ℝ) : 
  (∀ x : ℝ, x^3 + a*x^2 + b*x + c = 0 ↔ x = t ∨ x = u ∨ x = v) →
  (∀ x : ℝ, x^3 + a^3*x^2 + b^3*x + c^3 = 0 ↔ x = t^3 ∨ x = u^3 ∨ x = v^3) ↔
  ∃ t : ℝ, a = t ∧ b = 0 ∧ c = 0 :=
by sorry

end NUMINAMATH_CALUDE_cubic_roots_property_l2001_200165


namespace NUMINAMATH_CALUDE_sum_of_xyz_l2001_200195

theorem sum_of_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x^2 + y^2 + x*y = 1)
  (eq2 : y^2 + z^2 + y*z = 4)
  (eq3 : z^2 + x^2 + z*x = 5) :
  x + y + z = Real.sqrt (5 + 2 * Real.sqrt 3) := by
sorry

end NUMINAMATH_CALUDE_sum_of_xyz_l2001_200195


namespace NUMINAMATH_CALUDE_identify_counterfeit_coin_l2001_200146

/-- Represents the result of a weighing -/
inductive WeighResult
  | Left  : WeighResult  -- Left pan is heavier
  | Right : WeighResult  -- Right pan is heavier
  | Equal : WeighResult  -- Pans are balanced

/-- Represents a coin -/
inductive Coin
  | A : Coin
  | B : Coin
  | C : Coin
  | D : Coin

/-- Represents the state of a coin -/
inductive CoinState
  | Genuine : CoinState
  | Counterfeit : CoinState

/-- Represents whether the counterfeit coin is heavier or lighter -/
inductive CounterfeitWeight
  | Heavier : CounterfeitWeight
  | Lighter : CounterfeitWeight

/-- Function to perform a weighing -/
def weigh (left : List Coin) (right : List Coin) : WeighResult := sorry

/-- Function to determine the state of a coin -/
def determineCoinState (c : Coin) : CoinState := sorry

/-- Function to determine if the counterfeit coin is heavier or lighter -/
def determineCounterfeitWeight : CounterfeitWeight := sorry

/-- Theorem stating that the counterfeit coin can be identified in at most 3 weighings -/
theorem identify_counterfeit_coin :
  ∃ (counterfeit : Coin) (weight : CounterfeitWeight),
    (∀ c : Coin, c ≠ counterfeit → determineCoinState c = CoinState.Genuine) ∧
    (determineCoinState counterfeit = CoinState.Counterfeit) ∧
    (weight = determineCounterfeitWeight) ∧
    (∃ (w₁ w₂ w₃ : WeighResult),
      w₁ = weigh [Coin.A, Coin.B] [Coin.C, Coin.D] ∧
      w₂ = weigh [Coin.A, Coin.C] [Coin.B, Coin.D] ∧
      w₃ = weigh [Coin.A, Coin.D] [Coin.B, Coin.C]) :=
by
  sorry

end NUMINAMATH_CALUDE_identify_counterfeit_coin_l2001_200146


namespace NUMINAMATH_CALUDE_cafeteria_apples_l2001_200188

/-- The number of apples initially in the cafeteria -/
def initial_apples : ℕ := sorry

/-- The number of apples handed out to students -/
def apples_to_students : ℕ := 8

/-- The number of pies made -/
def pies_made : ℕ := 6

/-- The number of apples required for each pie -/
def apples_per_pie : ℕ := 9

/-- Theorem stating that the initial number of apples is 62 -/
theorem cafeteria_apples : initial_apples = 62 := by
  sorry

end NUMINAMATH_CALUDE_cafeteria_apples_l2001_200188


namespace NUMINAMATH_CALUDE_courier_cost_formula_l2001_200171

def courier_cost (P : ℕ+) : ℕ :=
  if P ≤ 2 then 15 else 15 + 5 * (P - 2)

theorem courier_cost_formula (P : ℕ+) :
  courier_cost P = if P ≤ 2 then 15 else 15 + 5 * (P - 2) :=
by sorry

end NUMINAMATH_CALUDE_courier_cost_formula_l2001_200171


namespace NUMINAMATH_CALUDE_common_element_in_sets_l2001_200163

theorem common_element_in_sets (n : ℕ) (S : Finset (Finset ℕ)) : 
  n = 50 →
  S.card = n →
  (∀ s ∈ S, s.card = 30) →
  (∀ T ⊆ S, T.card = 30 → ∃ x, ∀ s ∈ T, x ∈ s) →
  ∃ x, ∀ s ∈ S, x ∈ s :=
by sorry

end NUMINAMATH_CALUDE_common_element_in_sets_l2001_200163


namespace NUMINAMATH_CALUDE_dinner_cakes_count_l2001_200108

def lunch_cakes : ℕ := 6
def dinner_difference : ℕ := 3

theorem dinner_cakes_count : lunch_cakes + dinner_difference = 9 := by
  sorry

end NUMINAMATH_CALUDE_dinner_cakes_count_l2001_200108


namespace NUMINAMATH_CALUDE_sum_of_zeros_negative_l2001_200175

/-- The function f(x) = ln(x) - x + m -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.log x - x + m

/-- The function g(x) = f(x+m) -/
noncomputable def g (m : ℝ) (x : ℝ) : ℝ := f m (x + m)

/-- Theorem: Given f(x) = ln(x) - x + m, m > 1, g(x) = f(x+m), 
    and x₁, x₂ are zeros of g(x), then x₁ + x₂ < 0 -/
theorem sum_of_zeros_negative (m : ℝ) (x₁ x₂ : ℝ) 
  (hm : m > 1) 
  (hx₁ : g m x₁ = 0) 
  (hx₂ : g m x₂ = 0) : 
  x₁ + x₂ < 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_zeros_negative_l2001_200175


namespace NUMINAMATH_CALUDE_bug_crawl_distance_l2001_200184

-- Define the bug's starting position
def start : Int := 3

-- Define the bug's first destination
def first_dest : Int := 9

-- Define the bug's final destination
def final_dest : Int := -4

-- Define the function to calculate distance between two points
def distance (a b : Int) : Nat := Int.natAbs (b - a)

-- Theorem statement
theorem bug_crawl_distance : 
  distance start first_dest + distance first_dest final_dest = 19 := by
  sorry

end NUMINAMATH_CALUDE_bug_crawl_distance_l2001_200184


namespace NUMINAMATH_CALUDE_omega_range_l2001_200120

theorem omega_range (ω : ℝ) (h1 : ω > 1/4) :
  let f : ℝ → ℝ := λ x ↦ Real.sin (ω * x) - Real.cos (ω * x)
  (∀ k : ℤ, (3*π/4 + k*π) / ω ∉ Set.Ioo (2*π) (3*π)) →
  ω ∈ Set.union (Set.Icc (3/8) (7/12)) (Set.Icc (7/8) (11/12)) :=
by sorry

end NUMINAMATH_CALUDE_omega_range_l2001_200120


namespace NUMINAMATH_CALUDE_factorization_equality_l2001_200128

theorem factorization_equality (a b : ℝ) : a * b^2 - 2*a*b + a = a * (b - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2001_200128


namespace NUMINAMATH_CALUDE_ascending_order_for_negative_x_l2001_200137

theorem ascending_order_for_negative_x (x : ℝ) (h : -1 < x ∧ x < 0) : 
  5 * x < 0.5 * x ∧ 0.5 * x < 5 - x := by
  sorry

end NUMINAMATH_CALUDE_ascending_order_for_negative_x_l2001_200137


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2001_200157

-- Define the function f
def f (x : ℝ) : ℝ := sorry

-- State the theorem
theorem sum_of_coefficients :
  (∀ x, f (x + 5) = 4 * x^2 + 9 * x + 6) →
  (∃ a b c, ∀ x, f x = a * x^2 + b * x + c) →
  (∃ a b c, (∀ x, f x = a * x^2 + b * x + c) ∧ a + b + c = 34) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2001_200157


namespace NUMINAMATH_CALUDE_no_real_solution_l2001_200122

theorem no_real_solution (a b c : ℝ) : ¬∃ (x y z : ℝ), 
  (a^2 + b^2 + c^2 + 3*(x^2 + y^2 + z^2) = 6) ∧ (a*x + b*y + c*z = 2) := by
  sorry

end NUMINAMATH_CALUDE_no_real_solution_l2001_200122


namespace NUMINAMATH_CALUDE_cookfire_logs_after_three_hours_l2001_200151

/-- Calculates the number of logs left in a cookfire after a given number of hours. -/
def logs_left (initial_logs : ℕ) (burn_rate : ℕ) (add_rate : ℕ) (hours : ℕ) : ℕ :=
  initial_logs + hours * add_rate - hours * burn_rate

/-- Proves that after 3 hours, the cookfire will have 3 logs left. -/
theorem cookfire_logs_after_three_hours :
  logs_left 6 3 2 3 = 3 := by
  sorry

#eval logs_left 6 3 2 3

end NUMINAMATH_CALUDE_cookfire_logs_after_three_hours_l2001_200151


namespace NUMINAMATH_CALUDE_f_composition_negative_three_l2001_200129

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then 1 / Real.sqrt x else x^2

-- State the theorem
theorem f_composition_negative_three : f (f (-3)) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_negative_three_l2001_200129


namespace NUMINAMATH_CALUDE_percentage_relationship_l2001_200115

theorem percentage_relationship (j p t m n x y : ℕ) (r : ℚ) : 
  j > 0 ∧ p > 0 ∧ t > 0 ∧ m > 0 ∧ n > 0 ∧ x > 0 ∧ y > 0 →
  j = (3 / 4 : ℚ) * p →
  j = (4 / 5 : ℚ) * t →
  t = p - (r / 100) * p →
  m = (11 / 10 : ℚ) * p →
  n = (7 / 10 : ℚ) * m →
  j + p + t = m * n →
  x = (23 / 20 : ℚ) * j →
  y = (4 / 5 : ℚ) * n →
  x * y = (j + p + t)^2 →
  r = (25 / 4 : ℚ) := by
sorry

end NUMINAMATH_CALUDE_percentage_relationship_l2001_200115


namespace NUMINAMATH_CALUDE_grandfather_age_proof_l2001_200198

/-- The age of the grandfather -/
def grandfather_age : ℕ := 84

/-- The age of the older grandson -/
def older_grandson_age : ℕ := grandfather_age / 3

/-- The age of the younger grandson -/
def younger_grandson_age : ℕ := grandfather_age / 4

theorem grandfather_age_proof :
  (grandfather_age = 3 * older_grandson_age) ∧
  (grandfather_age = 4 * younger_grandson_age) ∧
  (older_grandson_age + younger_grandson_age = 49) →
  grandfather_age = 84 :=
by sorry

end NUMINAMATH_CALUDE_grandfather_age_proof_l2001_200198


namespace NUMINAMATH_CALUDE_g_of_2_eq_1_l2001_200131

/-- The function that keeps only the last k digits of a number --/
def lastKDigits (n : ℕ) (k : ℕ) : ℕ :=
  n % (10^k)

/-- The sequence of numbers on the board for a given k --/
def boardSequence (k : ℕ) : List ℕ :=
  sorry

/-- The function g(k) as defined in the problem --/
def g (k : ℕ) : ℕ :=
  sorry

/-- The theorem to be proved --/
theorem g_of_2_eq_1 : g 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_g_of_2_eq_1_l2001_200131


namespace NUMINAMATH_CALUDE_equal_cost_for_54_students_option2_cheaper_for_50_students_students_more_than_40_l2001_200181

/-- Represents the discount options for movie tickets. -/
inductive DiscountOption
  | Option1
  | Option2

/-- Calculates the total cost for a given number of students and discount option. -/
def calculateCost (students : ℕ) (option : DiscountOption) : ℚ :=
  match option with
  | DiscountOption.Option1 => 30 * students * (1 - 1/5)
  | DiscountOption.Option2 => 30 * (students - 6) * (1 - 1/10)

/-- Theorem stating that both discount options result in the same cost for 54 students. -/
theorem equal_cost_for_54_students :
  calculateCost 54 DiscountOption.Option1 = calculateCost 54 DiscountOption.Option2 :=
by sorry

/-- Theorem stating that Option 2 is cheaper for 50 students. -/
theorem option2_cheaper_for_50_students :
  calculateCost 50 DiscountOption.Option2 < calculateCost 50 DiscountOption.Option1 :=
by sorry

/-- Theorem stating that the number of students is more than 40. -/
theorem students_more_than_40 : 54 > 40 :=
by sorry

end NUMINAMATH_CALUDE_equal_cost_for_54_students_option2_cheaper_for_50_students_students_more_than_40_l2001_200181


namespace NUMINAMATH_CALUDE_min_value_of_squares_l2001_200117

theorem min_value_of_squares (a b t c : ℝ) (h : a + b = t) :
  ∃ m : ℝ, m = (t^2 + c^2 - 2*t*c + 2*c^2) / 2 ∧ 
  ∀ x y : ℝ, x + y = t → x^2 + (y + c)^2 ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_squares_l2001_200117


namespace NUMINAMATH_CALUDE_parabola_properties_l2001_200114

def f (x : ℝ) := -(x + 1)^2 + 3

theorem parabola_properties :
  (∀ x y : ℝ, f x ≤ f y → (x - (-1))^2 ≥ (y - (-1))^2) ∧ 
  (∀ x y : ℝ, x + y = -2 → f x = f y) ∧
  (f (-1) = 3 ∧ ∀ x : ℝ, f x ≤ f (-1)) ∧
  (∀ x y : ℝ, x > 1 ∧ y > 1 ∧ x > y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_parabola_properties_l2001_200114


namespace NUMINAMATH_CALUDE_workshop_workers_l2001_200183

/-- The total number of workers in a workshop -/
def total_workers : ℕ := 49

/-- The average salary of all workers -/
def avg_salary_all : ℕ := 8000

/-- The number of technicians -/
def num_technicians : ℕ := 7

/-- The average salary of technicians -/
def avg_salary_technicians : ℕ := 20000

/-- The average salary of non-technicians -/
def avg_salary_others : ℕ := 6000

/-- Theorem stating that the total number of workers is 49 -/
theorem workshop_workers :
  total_workers = 49 ∧
  avg_salary_all * total_workers = 
    avg_salary_technicians * num_technicians + 
    avg_salary_others * (total_workers - num_technicians) :=
by sorry

end NUMINAMATH_CALUDE_workshop_workers_l2001_200183


namespace NUMINAMATH_CALUDE_polynomial_roots_comparison_l2001_200136

theorem polynomial_roots_comparison (a₁ a₂ a₃ b₁ b₂ b₃ : ℝ) 
  (h_order_a : a₁ ≤ a₂ ∧ a₂ ≤ a₃)
  (h_order_b : b₁ ≤ b₂ ∧ b₂ ≤ b₃)
  (h_sum : a₁ + a₂ + a₃ = b₁ + b₂ + b₃)
  (h_prod : a₁*a₂ + a₂*a₃ + a₁*a₃ = b₁*b₂ + b₂*b₃ + b₁*b₃)
  (h_first : a₁ ≤ b₁) :
  a₃ ≤ b₃ := by sorry

end NUMINAMATH_CALUDE_polynomial_roots_comparison_l2001_200136


namespace NUMINAMATH_CALUDE_multiply_82519_by_9999_l2001_200123

theorem multiply_82519_by_9999 : 82519 * 9999 = 825117481 := by
  sorry

end NUMINAMATH_CALUDE_multiply_82519_by_9999_l2001_200123


namespace NUMINAMATH_CALUDE_meadow_trees_l2001_200149

/-- Represents the number of trees around the meadow. -/
def num_trees : ℕ := sorry

/-- Represents Serezha's count of a specific tree. -/
def serezha_count1 : ℕ := 20

/-- Represents Misha's count of the same tree as serezha_count1. -/
def misha_count1 : ℕ := 7

/-- Represents Serezha's count of another specific tree. -/
def serezha_count2 : ℕ := 7

/-- Represents Misha's count of the same tree as serezha_count2. -/
def misha_count2 : ℕ := 94

/-- The theorem stating that the number of trees around the meadow is 100. -/
theorem meadow_trees : num_trees = 100 := by sorry

end NUMINAMATH_CALUDE_meadow_trees_l2001_200149


namespace NUMINAMATH_CALUDE_encyclopedia_total_pages_l2001_200116

/-- Represents a chapter in the encyclopedia -/
structure Chapter where
  main_pages : ℕ
  sub_chapters : ℕ
  sub_chapter_pages : ℕ

/-- The encyclopedia with 12 chapters -/
def encyclopedia : Vector Chapter 12 :=
  Vector.ofFn fun i =>
    match i with
    | 0 => ⟨450, 3, 90⟩
    | 1 => ⟨650, 5, 68⟩
    | 2 => ⟨712, 4, 75⟩
    | 3 => ⟨820, 6, 120⟩
    | 4 => ⟨530, 2, 110⟩
    | 5 => ⟨900, 7, 95⟩
    | 6 => ⟨680, 4, 80⟩
    | 7 => ⟨555, 3, 180⟩
    | 8 => ⟨990, 5, 53⟩
    | 9 => ⟨825, 6, 150⟩
    | 10 => ⟨410, 2, 200⟩
    | 11 => ⟨1014, 7, 69⟩

/-- Total pages in a chapter -/
def total_pages_in_chapter (c : Chapter) : ℕ :=
  c.main_pages + c.sub_chapters * c.sub_chapter_pages

/-- Total pages in the encyclopedia -/
def total_pages : ℕ :=
  (encyclopedia.toList.map total_pages_in_chapter).sum

theorem encyclopedia_total_pages :
  total_pages = 13659 := by sorry

end NUMINAMATH_CALUDE_encyclopedia_total_pages_l2001_200116


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l2001_200144

theorem cube_volume_from_surface_area :
  ∀ s : ℝ, 
    s > 0 →
    6 * s^2 = 150 →
    s^3 = 125 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l2001_200144


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_75_4500_l2001_200190

theorem gcd_lcm_sum_75_4500 : Nat.gcd 75 4500 + Nat.lcm 75 4500 = 4575 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_75_4500_l2001_200190


namespace NUMINAMATH_CALUDE_no_natural_solutions_l2001_200182

theorem no_natural_solutions :
  ∀ (x y : ℕ), y^2 ≠ x^2 + x + 1 := by
  sorry

end NUMINAMATH_CALUDE_no_natural_solutions_l2001_200182


namespace NUMINAMATH_CALUDE_construction_time_theorem_l2001_200178

/-- Represents the time taken to construct a wall given the number of boys and girls -/
def constructionTime (boys : ℕ) (girls : ℕ) : ℝ :=
  sorry

/-- Theorem stating that if 16 boys or 24 girls can construct a wall in 6 days,
    then 8 boys and 4 girls will take 12 days to construct the same wall -/
theorem construction_time_theorem :
  (constructionTime 16 0 = 6 ∧ constructionTime 0 24 = 6) →
  constructionTime 8 4 = 12 :=
sorry

end NUMINAMATH_CALUDE_construction_time_theorem_l2001_200178


namespace NUMINAMATH_CALUDE_citrus_grove_total_orchards_l2001_200118

/-- Represents the number of orchards for each fruit type and the total -/
structure CitrusGrove where
  lemons : ℕ
  oranges : ℕ
  grapefruits : ℕ
  limes : ℕ
  total : ℕ

/-- Theorem stating the total number of orchards in the citrus grove -/
theorem citrus_grove_total_orchards (cg : CitrusGrove) : cg.total = 16 :=
  by
  have h1 : cg.lemons = 8 := by sorry
  have h2 : cg.oranges = cg.lemons / 2 := by sorry
  have h3 : cg.grapefruits = 2 := by sorry
  have h4 : cg.limes + cg.grapefruits = cg.total - (cg.lemons + cg.oranges) := by sorry
  sorry

#check citrus_grove_total_orchards

end NUMINAMATH_CALUDE_citrus_grove_total_orchards_l2001_200118


namespace NUMINAMATH_CALUDE_picture_book_shelves_l2001_200197

theorem picture_book_shelves 
  (books_per_shelf : ℕ) 
  (mystery_shelves : ℕ) 
  (total_books : ℕ) : 
  books_per_shelf = 4 → 
  mystery_shelves = 5 → 
  total_books = 32 → 
  (total_books - mystery_shelves * books_per_shelf) / books_per_shelf = 3 := by
sorry

end NUMINAMATH_CALUDE_picture_book_shelves_l2001_200197


namespace NUMINAMATH_CALUDE_factorization_x_10_minus_1024_l2001_200174

theorem factorization_x_10_minus_1024 (x : ℝ) : 
  x^10 - 1024 = (x^5 + 32) * (x^5 - 32) :=
by
  sorry

end NUMINAMATH_CALUDE_factorization_x_10_minus_1024_l2001_200174


namespace NUMINAMATH_CALUDE_expression_evaluation_l2001_200133

theorem expression_evaluation (x y : ℚ) (hx : x = -1/2) (hy : y = 2022) :
  ((2*x - y)^2 - (2*x + y)*(2*x - y)) / (2*y) = 2023 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2001_200133


namespace NUMINAMATH_CALUDE_hexagon_triangle_ratio_l2001_200145

/-- A regular hexagon divided into six equal triangles -/
structure RegularHexagon where
  /-- The area of one of the six triangles -/
  s : ℝ
  /-- The area of a region formed by two adjacent triangles -/
  r : ℝ
  /-- The hexagon is divided into six equal triangles -/
  triangle_count : ℕ
  triangle_count_eq : triangle_count = 6
  /-- r is the area of two adjacent triangles -/
  r_eq : r = 2 * s

/-- The ratio of the area of two adjacent triangles to the area of one triangle in a regular hexagon is 2 -/
theorem hexagon_triangle_ratio (h : RegularHexagon) : r / s = 2 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_triangle_ratio_l2001_200145


namespace NUMINAMATH_CALUDE_cashback_strategies_reduce_losses_l2001_200103

/-- Represents a bank's cashback program -/
structure CashbackProgram where
  name : String
  maxCashbackPercentage : Float
  monthlyCapExists : Bool
  variableRate : Bool
  nonMonetaryRewards : Bool

/-- Represents a customer's behavior -/
structure CustomerBehavior where
  financialLiteracy : Float
  prefersHighCashbackCategories : Bool

/-- Calculates the profitability of a cashback program -/
def calculateProfitability (program : CashbackProgram) (customer : CustomerBehavior) : Float :=
  sorry

/-- Theorem: Implementing certain cashback strategies can reduce potential losses for banks -/
theorem cashback_strategies_reduce_losses 
  (program : CashbackProgram) 
  (customer : CustomerBehavior) :
  (program.monthlyCapExists ∨ program.variableRate ∨ program.nonMonetaryRewards) →
  (customer.financialLiteracy > 0.8 ∧ customer.prefersHighCashbackCategories) →
  calculateProfitability program customer > 0 := by
  sorry

#check cashback_strategies_reduce_losses

end NUMINAMATH_CALUDE_cashback_strategies_reduce_losses_l2001_200103


namespace NUMINAMATH_CALUDE_systemC_is_linear_l2001_200160

/-- A linear equation in two variables is of the form ax + by = c, where a, b, and c are constants. -/
def IsLinearEquationInTwoVars (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), ∀ x y, f x y = a * x + b * y - c

/-- A system of two equations in two variables -/
structure SystemOfTwoEquations :=
  (eq1 : ℝ → ℝ → ℝ)
  (eq2 : ℝ → ℝ → ℝ)

/-- Definition of the specific system of equations given in Option C -/
def systemC : SystemOfTwoEquations :=
  { eq1 := λ x y => x - 3,
    eq2 := λ x y => 2 * x - y - 7 }

/-- Theorem stating that the given system is a system of linear equations in two variables -/
theorem systemC_is_linear : 
  IsLinearEquationInTwoVars systemC.eq1 ∧ IsLinearEquationInTwoVars systemC.eq2 :=
sorry

end NUMINAMATH_CALUDE_systemC_is_linear_l2001_200160


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2001_200155

/-- The sum of an arithmetic sequence with first term a₁ = k^2 - k + 1 and common difference d = 2 for k^2 terms -/
theorem arithmetic_sequence_sum (k : ℕ) : 
  let a₁ : ℤ := k^2 - k + 1
  let d : ℤ := 2
  let n : ℕ := k^2
  let Sₙ : ℤ := n * (2 * a₁ + (n - 1) * d) / 2
  Sₙ = 2 * k^4 - k^3 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2001_200155


namespace NUMINAMATH_CALUDE_range_of_x_l2001_200121

theorem range_of_x (x : ℝ) : (1 / x < 3) ∧ (1 / x > -4) → x > 1/3 ∨ x < -1/4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_x_l2001_200121


namespace NUMINAMATH_CALUDE_total_distance_jogged_l2001_200139

/-- The total distance jogged by Kyle and Sarah -/
def total_distance (inner_track_length outer_track_length : ℝ)
  (kyle_inner_laps kyle_outer_laps : ℝ)
  (sarah_inner_laps sarah_outer_laps : ℝ) : ℝ :=
  (kyle_inner_laps * inner_track_length + kyle_outer_laps * outer_track_length) +
  (sarah_inner_laps * inner_track_length + sarah_outer_laps * outer_track_length)

/-- Theorem stating the total distance jogged by Kyle and Sarah -/
theorem total_distance_jogged :
  total_distance 250 400 1.12 1.78 2.73 1.36 = 2218.5 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_jogged_l2001_200139


namespace NUMINAMATH_CALUDE_jennys_number_l2001_200150

theorem jennys_number (y : ℝ) : 10 * (y / 2 - 6) = 70 → y = 26 := by
  sorry

end NUMINAMATH_CALUDE_jennys_number_l2001_200150


namespace NUMINAMATH_CALUDE_triangle_inequality_l2001_200107

theorem triangle_inequality (a b c S : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hS : S > 0)
  (h_triangle : S = Real.sqrt (((a + b + c) * (b + c - a) * (c + a - b) * (a + b - c)) / 16)) :
  4 * Real.sqrt 3 * S + (a - b)^2 + (b - c)^2 + (c - a)^2 ≤ a^2 + b^2 + c^2 ∧
  a^2 + b^2 + c^2 ≤ 4 * Real.sqrt 3 * S + 3 * ((a - b)^2 + (b - c)^2 + (c - a)^2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2001_200107


namespace NUMINAMATH_CALUDE_no_real_solution_for_equation_l2001_200105

theorem no_real_solution_for_equation :
  ¬ ∃ (x : ℝ), x ≠ 0 ∧ (2 / x - (3 / x) * (6 / x) = 0.5) := by
  sorry

end NUMINAMATH_CALUDE_no_real_solution_for_equation_l2001_200105


namespace NUMINAMATH_CALUDE_professors_age_l2001_200127

def guesses : List Nat := [34, 37, 39, 41, 43, 46, 48, 51, 53, 56]

def is_prime (n : Nat) : Prop := Nat.Prime n

theorem professors_age :
  ∃ (age : Nat),
    age ∈ guesses ∧
    is_prime age ∧
    (guesses.filter (· < age)).length ≥ guesses.length / 2 ∧
    (guesses.filter (fun x => x = age - 1 ∨ x = age + 1)).length = 2 ∧
    age = 47 :=
  sorry

end NUMINAMATH_CALUDE_professors_age_l2001_200127


namespace NUMINAMATH_CALUDE_unique_n_exists_l2001_200142

-- Define S(n) as the sum of digits of n
def S (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem unique_n_exists : ∃! n : ℕ, n > 0 ∧ n + S n + S (S n) = 2023 := by
  sorry

end NUMINAMATH_CALUDE_unique_n_exists_l2001_200142


namespace NUMINAMATH_CALUDE_best_fit_model_l2001_200148

/-- Represents a regression model with its R² value -/
structure RegressionModel where
  r_squared : ℝ

/-- Determines if a model has the best fitting effect among a list of models -/
def has_best_fit (model : RegressionModel) (models : List RegressionModel) : Prop :=
  ∀ m ∈ models, model.r_squared ≥ m.r_squared

theorem best_fit_model (models : List RegressionModel) 
  (h1 : RegressionModel.mk 0.95 ∈ models)
  (h2 : RegressionModel.mk 0.70 ∈ models)
  (h3 : RegressionModel.mk 0.55 ∈ models)
  (h4 : RegressionModel.mk 0.30 ∈ models)
  (h5 : models.length = 4) :
  has_best_fit (RegressionModel.mk 0.95) models :=
sorry

end NUMINAMATH_CALUDE_best_fit_model_l2001_200148


namespace NUMINAMATH_CALUDE_postman_june_distance_l2001_200185

/-- Represents a step counter with a maximum count before resetting -/
structure StepCounter where
  max_count : ℕ
  resets : ℕ
  final_count : ℕ

/-- Calculates the total number of steps based on the step counter data -/
def total_steps (counter : StepCounter) : ℕ :=
  counter.max_count * counter.resets + counter.final_count

/-- Converts steps to miles given the number of steps per mile -/
def steps_to_miles (steps : ℕ) (steps_per_mile : ℕ) : ℕ :=
  steps / steps_per_mile

/-- Theorem stating that given the specified conditions, the total distance walked is 2615 miles -/
theorem postman_june_distance :
  let counter : StepCounter := ⟨100000, 52, 30000⟩
  let steps_per_mile : ℕ := 2000
  steps_to_miles (total_steps counter) steps_per_mile = 2615 := by
  sorry

end NUMINAMATH_CALUDE_postman_june_distance_l2001_200185


namespace NUMINAMATH_CALUDE_dot_product_of_vectors_l2001_200134

theorem dot_product_of_vectors (a b : ℝ × ℝ) 
  (h1 : a + b = (1, -3)) 
  (h2 : a - b = (3, 7)) : 
  a.1 * b.1 + a.2 * b.2 = -12 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_of_vectors_l2001_200134


namespace NUMINAMATH_CALUDE_pyramid_volume_in_cone_l2001_200156

/-- The volume of a pyramid inscribed in a cone, where the pyramid's base is an isosceles triangle -/
theorem pyramid_volume_in_cone (V : ℝ) (α : ℝ) :
  let cone_volume := V
  let base_angle := α
  let pyramid_volume := (2 * V / Real.pi) * Real.sin α * (Real.cos (α / 2))^2
  0 < V → 0 < α → α < π →
  pyramid_volume = (2 * cone_volume / Real.pi) * Real.sin base_angle * (Real.cos (base_angle / 2))^2 :=
by sorry

end NUMINAMATH_CALUDE_pyramid_volume_in_cone_l2001_200156
