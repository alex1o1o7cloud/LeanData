import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_equation_implication_l3543_354345

theorem quadratic_equation_implication (x : ℝ) : 2 * x^2 + 1 = 17 → 4 * x^2 + 1 = 33 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_implication_l3543_354345


namespace NUMINAMATH_CALUDE_cos_2x_min_value_in_interval_l3543_354373

theorem cos_2x_min_value_in_interval :
  ∃ x ∈ Set.Ioo 0 π, ∀ y ∈ Set.Ioo 0 π, Real.cos (2 * x) ≤ Real.cos (2 * y) ∧
  Real.cos (2 * x) = -1 :=
sorry

end NUMINAMATH_CALUDE_cos_2x_min_value_in_interval_l3543_354373


namespace NUMINAMATH_CALUDE_difference_10_6_l3543_354330

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  prop1 : a 5 * a 7 = 6
  prop2 : a 2 + a 10 = 5

/-- The difference between the 10th and 6th terms is either 2 or -2 -/
theorem difference_10_6 (seq : ArithmeticSequence) : 
  seq.a 10 - seq.a 6 = 2 ∨ seq.a 10 - seq.a 6 = -2 := by
  sorry

end NUMINAMATH_CALUDE_difference_10_6_l3543_354330


namespace NUMINAMATH_CALUDE_opposite_of_negative_mixed_number_l3543_354369

theorem opposite_of_negative_mixed_number : 
  -(-(7/4)) = 7/4 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_mixed_number_l3543_354369


namespace NUMINAMATH_CALUDE_min_value_a_plus_2b_l3543_354363

theorem min_value_a_plus_2b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 1/b = 1) :
  ∀ x y : ℝ, x > 0 → y > 0 → 1/x + 1/y = 1 → x + 2*y ≥ 3 + 2*Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_a_plus_2b_l3543_354363


namespace NUMINAMATH_CALUDE_linear_function_properties_l3543_354308

def linear_function (k b : ℝ) : ℝ → ℝ := λ x => k * x + b

theorem linear_function_properties :
  ∃ (k b : ℝ), 
    k ≠ 0 ∧
    linear_function k b 1 = 2 ∧
    linear_function k b (-1) = 4 ∧
    linear_function (-1) 3 = linear_function k b ∧
    linear_function (-1) 3 2 ≠ 3 ∧
    linear_function (-1) 3 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_properties_l3543_354308


namespace NUMINAMATH_CALUDE_mushroom_problem_solution_l3543_354307

/-- Represents a basket of mushrooms with two types: ryzhiki and gruzdi -/
structure MushroomBasket where
  total : ℕ
  ryzhiki : ℕ
  gruzdi : ℕ
  sum_eq_total : ryzhiki + gruzdi = total

/-- Predicate to check if the basket satisfies the ryzhiki condition -/
def has_ryzhik_in_12 (b : MushroomBasket) : Prop :=
  ∀ n : ℕ, n ≤ b.total - 12 → b.ryzhiki > n

/-- Predicate to check if the basket satisfies the gruzdi condition -/
def has_gruzd_in_20 (b : MushroomBasket) : Prop :=
  ∀ n : ℕ, n ≤ b.total - 20 → b.gruzdi > n

/-- Theorem stating the solution to the mushroom problem -/
theorem mushroom_problem_solution :
  ∀ b : MushroomBasket,
  b.total = 30 →
  has_ryzhik_in_12 b →
  has_gruzd_in_20 b →
  b.ryzhiki = 19 ∧ b.gruzdi = 11 := by
  sorry


end NUMINAMATH_CALUDE_mushroom_problem_solution_l3543_354307


namespace NUMINAMATH_CALUDE_f_properties_l3543_354375

noncomputable def f (x : ℝ) : ℝ := 
  Real.cos (2 * x - Real.pi) + 2 * Real.sin (x - Real.pi / 2) * Real.sin (x + Real.pi / 2)

theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
    (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T')) ∧
  (∀ (y : ℝ), y ∈ Set.Icc (-1 : ℝ) 1 ↔ ∃ (x : ℝ), x ∈ Set.Icc (-Real.pi) Real.pi ∧ f x = y) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l3543_354375


namespace NUMINAMATH_CALUDE_mikeys_jelly_beans_l3543_354364

theorem mikeys_jelly_beans (napoleon : ℕ) (sedrich : ℕ) (mikey : ℕ) : 
  napoleon = 17 →
  sedrich = napoleon + 4 →
  2 * (napoleon + sedrich) = 4 * mikey →
  mikey = 19 := by
sorry

end NUMINAMATH_CALUDE_mikeys_jelly_beans_l3543_354364


namespace NUMINAMATH_CALUDE_min_sum_of_squares_with_diff_l3543_354343

theorem min_sum_of_squares_with_diff (x y : ℕ+) : 
  x.val^2 - y.val^2 = 145 → x.val^2 + y.val^2 ≥ 433 := by
sorry

end NUMINAMATH_CALUDE_min_sum_of_squares_with_diff_l3543_354343


namespace NUMINAMATH_CALUDE_square_pattern_properties_l3543_354357

/-- Represents the number of squares in Figure n of the pattern --/
def num_squares (n : ℕ+) : ℕ := 3 + 2 * (n - 1)

/-- Represents the perimeter of Figure n of the pattern --/
def perimeter (n : ℕ+) : ℕ := 8 + 4 * (n - 1)

/-- Theorem stating the properties of the square pattern --/
theorem square_pattern_properties (n : ℕ+) :
  (num_squares n = 3 + 2 * (n - 1)) ∧ (perimeter n = 8 + 4 * (n - 1)) := by
  sorry

end NUMINAMATH_CALUDE_square_pattern_properties_l3543_354357


namespace NUMINAMATH_CALUDE_value_of_a_l3543_354331

theorem value_of_a (a : ℝ) : (0.005 * a = 0.80) → (a = 160) := by sorry

end NUMINAMATH_CALUDE_value_of_a_l3543_354331


namespace NUMINAMATH_CALUDE_compare_quadratic_expressions_l3543_354374

theorem compare_quadratic_expressions (x : ℝ) : 2*x^2 - 2*x + 1 > x^2 - 2*x := by
  sorry

end NUMINAMATH_CALUDE_compare_quadratic_expressions_l3543_354374


namespace NUMINAMATH_CALUDE_max_quarters_sasha_l3543_354383

/-- Represents the value of a quarter in dollars -/
def quarter_value : ℚ := 25 / 100

/-- Represents the value of a dime in dollars -/
def dime_value : ℚ := 10 / 100

/-- The total amount Sasha has in dollars -/
def total_amount : ℚ := 48 / 10

theorem max_quarters_sasha (q : ℕ) : 
  q * quarter_value + (2 * q) * dime_value ≤ total_amount → 
  q ≤ 10 :=
sorry

end NUMINAMATH_CALUDE_max_quarters_sasha_l3543_354383


namespace NUMINAMATH_CALUDE_ac_length_l3543_354319

/-- Given a quadrilateral ABCD with specified side lengths, prove the length of AC --/
theorem ac_length (AB DC AD : ℝ) (h1 : AB = 13) (h2 : DC = 15) (h3 : AD = 12) :
  ∃ (AC : ℝ), abs (AC - Real.sqrt (369 + 240 * Real.sqrt 2)) < 0.05 := by
  sorry

end NUMINAMATH_CALUDE_ac_length_l3543_354319


namespace NUMINAMATH_CALUDE_inverse_variation_proof_l3543_354320

/-- Given that x varies inversely as the square of y, prove that y = 6 when x = 0.1111111111111111,
    given that y = 2 when x = 1. -/
theorem inverse_variation_proof (x y : ℝ) (k : ℝ) 
    (h1 : ∀ x y, x = k / (y * y))  -- x varies inversely as square of y
    (h2 : 1 = k / (2 * 2))         -- y = 2 when x = 1
    : y = 6 ↔ x = 0.1111111111111111 :=
by sorry

end NUMINAMATH_CALUDE_inverse_variation_proof_l3543_354320


namespace NUMINAMATH_CALUDE_four_solutions_l3543_354381

/-- S(n) denotes the sum of the digits of n -/
def S (n : ℕ) : ℕ := sorry

/-- The number of positive integers n such that n + S(n) + S(S(n)) = 2010 -/
def count_solutions : ℕ := sorry

/-- Theorem stating that there are exactly 4 solutions -/
theorem four_solutions : count_solutions = 4 := by sorry

end NUMINAMATH_CALUDE_four_solutions_l3543_354381


namespace NUMINAMATH_CALUDE_largest_cube_surface_area_l3543_354327

/-- Given a cuboid with dimensions 12 cm, 16 cm, and 14 cm, 
    the surface area of the largest cube that can be cut from it is 864 cm^2 -/
theorem largest_cube_surface_area 
  (width : ℝ) (length : ℝ) (height : ℝ)
  (h_width : width = 12)
  (h_length : length = 16)
  (h_height : height = 14) :
  6 * (min width (min length height))^2 = 864 := by
  sorry

end NUMINAMATH_CALUDE_largest_cube_surface_area_l3543_354327


namespace NUMINAMATH_CALUDE_complex_number_property_l3543_354344

theorem complex_number_property : 
  let z : ℂ := (-2 * Complex.I) / (1 + Complex.I)
  (z + 1).im ≠ 0 ∧ (z + 1).re = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_property_l3543_354344


namespace NUMINAMATH_CALUDE_polyhedron_volume_l3543_354318

theorem polyhedron_volume (prism_volume pyramid_volume : ℝ) 
  (h1 : prism_volume = Real.sqrt 2 - 1)
  (h2 : pyramid_volume = 1/6) :
  prism_volume + 2 * pyramid_volume = Real.sqrt 2 - 2/3 := by
  sorry

end NUMINAMATH_CALUDE_polyhedron_volume_l3543_354318


namespace NUMINAMATH_CALUDE_solution_of_linear_equation_l3543_354376

theorem solution_of_linear_equation (x y m : ℝ) 
  (h1 : x = -1)
  (h2 : y = 2)
  (h3 : m * x + 2 * y = 1) :
  m = 3 := by
sorry

end NUMINAMATH_CALUDE_solution_of_linear_equation_l3543_354376


namespace NUMINAMATH_CALUDE_floor_times_x_eq_54_l3543_354382

theorem floor_times_x_eq_54 :
  ∃! (x : ℝ), x > 0 ∧ (⌊x⌋ : ℝ) * x = 54 ∧ abs (x - 7.7143) < 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_floor_times_x_eq_54_l3543_354382


namespace NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l3543_354338

theorem least_positive_integer_with_remainders : ∃! n : ℕ,
  (n > 0) ∧
  (n % 11 = 10) ∧
  (n % 12 = 11) ∧
  (n % 13 = 12) ∧
  (n % 14 = 13) ∧
  (n % 15 = 14) ∧
  (n % 16 = 15) ∧
  (∀ m : ℕ, m > 0 ∧ 
    (m % 11 = 10) ∧
    (m % 12 = 11) ∧
    (m % 13 = 12) ∧
    (m % 14 = 13) ∧
    (m % 15 = 14) ∧
    (m % 16 = 15) → m ≥ n) ∧
  n = 720719 :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l3543_354338


namespace NUMINAMATH_CALUDE_min_cuts_correct_l3543_354399

/-- The minimum number of cuts required to divide a cube of edge length 4 into 64 unit cubes -/
def min_cuts : ℕ := 6

/-- The edge length of the initial cube -/
def initial_edge_length : ℕ := 4

/-- The number of smaller cubes we want to create -/
def target_num_cubes : ℕ := 64

/-- The edge length of the smaller cubes -/
def target_edge_length : ℕ := 1

/-- Theorem stating that min_cuts is the minimum number of cuts required -/
theorem min_cuts_correct :
  (2 ^ min_cuts = target_num_cubes) ∧
  (∀ n : ℕ, n < min_cuts → 2 ^ n < target_num_cubes) :=
sorry

end NUMINAMATH_CALUDE_min_cuts_correct_l3543_354399


namespace NUMINAMATH_CALUDE_boys_girls_arrangement_l3543_354301

/-- The number of ways to arrange boys and girls in a row with alternating genders -/
def alternating_arrangements (num_boys : ℕ) (num_girls : ℕ) : ℕ :=
  (Nat.factorial num_boys) * (Nat.factorial num_girls)

/-- Theorem stating that there are 144 ways to arrange 4 boys and 3 girls
    in a row such that no two boys or two girls stand next to each other -/
theorem boys_girls_arrangement :
  alternating_arrangements 4 3 = 144 := by
  sorry

#check boys_girls_arrangement

end NUMINAMATH_CALUDE_boys_girls_arrangement_l3543_354301


namespace NUMINAMATH_CALUDE_seventh_term_is_29_3_l3543_354329

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  -- First term
  a : ℚ
  -- Common difference
  d : ℚ
  -- Sum of first five terms is 15
  sum_first_five : a + (a + d) + (a + 2*d) + (a + 3*d) + (a + 4*d) = 15
  -- Sixth term is 8
  sixth_term : a + 5*d = 8

/-- The seventh term of the arithmetic sequence is 29/3 -/
theorem seventh_term_is_29_3 (seq : ArithmeticSequence) : seq.a + 6*seq.d = 29/3 := by
  sorry


end NUMINAMATH_CALUDE_seventh_term_is_29_3_l3543_354329


namespace NUMINAMATH_CALUDE_complex_conjugate_roots_l3543_354371

/-- The quadratic equation z^2 + (12 + ci)z + (45 + di) = 0 has complex conjugate roots if and only if c = 0 and d = 0 -/
theorem complex_conjugate_roots (c d : ℝ) : 
  (∀ z : ℂ, z^2 + (12 + c * Complex.I) * z + (45 + d * Complex.I) = 0 → 
    ∃ x y : ℝ, z = x + y * Complex.I ∧ x - y * Complex.I ∈ {w : ℂ | w^2 + (12 + c * Complex.I) * w + (45 + d * Complex.I) = 0}) ↔ 
  c = 0 ∧ d = 0 := by
sorry

end NUMINAMATH_CALUDE_complex_conjugate_roots_l3543_354371


namespace NUMINAMATH_CALUDE_laborer_income_proof_l3543_354372

/-- Represents the monthly income of a laborer -/
def monthly_income : ℝ := 75

/-- Represents the debt after 6 months -/
def debt : ℝ := 30

theorem laborer_income_proof :
  let initial_period := 6
  let initial_monthly_expenditure := 80
  let later_period := 4
  let later_monthly_expenditure := 60
  let savings := 30
  (initial_period * monthly_income < initial_period * initial_monthly_expenditure) ∧
  (later_period * monthly_income = later_period * later_monthly_expenditure + debt + savings) →
  monthly_income = 75 := by
sorry

end NUMINAMATH_CALUDE_laborer_income_proof_l3543_354372


namespace NUMINAMATH_CALUDE_average_weight_b_c_l3543_354321

/-- Given the weights of three people a, b, and c, prove that the average weight of b and c is 43 kg -/
theorem average_weight_b_c (a b c : ℝ) : 
  (a + b + c) / 3 = 42 →  -- The average weight of a, b, and c is 42 kg
  (a + b) / 2 = 40 →      -- The average weight of a and b is 40 kg
  b = 40 →                -- The weight of b is 40 kg
  (b + c) / 2 = 43 :=     -- The average weight of b and c is 43 kg
by sorry

end NUMINAMATH_CALUDE_average_weight_b_c_l3543_354321


namespace NUMINAMATH_CALUDE_largest_four_digit_divisible_by_5_6_2_l3543_354347

theorem largest_four_digit_divisible_by_5_6_2 :
  ∀ n : ℕ, n ≤ 9999 ∧ n ≥ 1000 ∧ 5 ∣ n ∧ 6 ∣ n → n ≤ 9990 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_four_digit_divisible_by_5_6_2_l3543_354347


namespace NUMINAMATH_CALUDE_printer_cost_l3543_354315

/-- The cost of a single printer given the total cost of keyboards and printers, 
    the number of each item, and the cost of a single keyboard. -/
theorem printer_cost 
  (total_cost : ℕ) 
  (num_keyboards num_printers : ℕ) 
  (keyboard_cost : ℕ) 
  (h1 : total_cost = 2050)
  (h2 : num_keyboards = 15)
  (h3 : num_printers = 25)
  (h4 : keyboard_cost = 20) :
  (total_cost - num_keyboards * keyboard_cost) / num_printers = 70 :=
by sorry

end NUMINAMATH_CALUDE_printer_cost_l3543_354315


namespace NUMINAMATH_CALUDE_xiaoming_walking_speed_l3543_354340

theorem xiaoming_walking_speed (distance : ℝ) (min_time max_time : ℝ) (h1 : distance = 3500)
  (h2 : min_time = 40) (h3 : max_time = 50) :
  let speed_range := {x : ℝ | distance / max_time ≤ x ∧ x ≤ distance / min_time}
  ∀ x ∈ speed_range, 70 ≤ x ∧ x ≤ 87.5 :=
by sorry

end NUMINAMATH_CALUDE_xiaoming_walking_speed_l3543_354340


namespace NUMINAMATH_CALUDE_claires_weight_l3543_354336

theorem claires_weight (alice_weight claire_weight : ℚ) : 
  alice_weight + claire_weight = 200 →
  claire_weight - alice_weight = claire_weight / 3 →
  claire_weight = 1400 / 9 := by
sorry

end NUMINAMATH_CALUDE_claires_weight_l3543_354336


namespace NUMINAMATH_CALUDE_johnson_family_seating_l3543_354309

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem johnson_family_seating (boys girls : ℕ) (total : ℕ) :
  boys = 5 →
  girls = 4 →
  total = boys + girls →
  factorial total - (factorial boys * factorial girls) = 360000 :=
by
  sorry

end NUMINAMATH_CALUDE_johnson_family_seating_l3543_354309


namespace NUMINAMATH_CALUDE_rectangular_garden_area_l3543_354328

/-- Represents a rectangular garden -/
structure RectangularGarden where
  width : ℝ
  length : ℝ

/-- The area of a rectangular garden -/
def area (g : RectangularGarden) : ℝ := g.width * g.length

/-- Theorem: The area of a rectangular garden with width 16 meters and length three times its width is 768 square meters -/
theorem rectangular_garden_area : 
  ∀ (g : RectangularGarden), 
  g.width = 16 → 
  g.length = 3 * g.width → 
  area g = 768 := by
sorry

end NUMINAMATH_CALUDE_rectangular_garden_area_l3543_354328


namespace NUMINAMATH_CALUDE_hippocrates_lunes_l3543_354304

theorem hippocrates_lunes (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a^2 + b^2 = c^2) :
  let semicircle_area (d : ℝ) := π * d^2 / 8
  let triangle_area := a * b / 2
  let lunes_area := semicircle_area a + semicircle_area b - (semicircle_area c - triangle_area)
  lunes_area = triangle_area := by
sorry

end NUMINAMATH_CALUDE_hippocrates_lunes_l3543_354304


namespace NUMINAMATH_CALUDE_max_absolute_value_z_l3543_354396

theorem max_absolute_value_z (z : ℂ) (h : Complex.abs (z + 3 + 4 * I) ≤ 2) :
  ∃ (M : ℝ), M = 7 ∧ Complex.abs z ≤ M ∧ ∀ (N : ℝ), Complex.abs z ≤ N → M ≤ N :=
sorry

end NUMINAMATH_CALUDE_max_absolute_value_z_l3543_354396


namespace NUMINAMATH_CALUDE_repeating_decimal_three_six_equals_eleven_thirtieths_l3543_354311

def repeating_decimal (a b : ℕ) : ℚ :=
  (a : ℚ) / 10 + (b : ℚ) / (9 * 10)

theorem repeating_decimal_three_six_equals_eleven_thirtieths :
  repeating_decimal 3 6 = 11 / 30 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_three_six_equals_eleven_thirtieths_l3543_354311


namespace NUMINAMATH_CALUDE_jaylen_green_beans_l3543_354397

/-- Prove that Jaylen has 7 green beans given the conditions of the vegetable problem. -/
theorem jaylen_green_beans :
  ∀ (jaylen_carrots jaylen_cucumbers jaylen_bell_peppers jaylen_green_beans kristin_bell_peppers kristin_green_beans : ℕ),
  jaylen_carrots = 5 →
  jaylen_cucumbers = 2 →
  kristin_bell_peppers = 2 →
  jaylen_bell_peppers = 2 * kristin_bell_peppers →
  kristin_green_beans = 20 →
  jaylen_carrots + jaylen_cucumbers + jaylen_bell_peppers + jaylen_green_beans = 18 →
  jaylen_green_beans = kristin_green_beans / 2 - 3 →
  jaylen_green_beans = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_jaylen_green_beans_l3543_354397


namespace NUMINAMATH_CALUDE_arithmetic_sequence_remainder_l3543_354385

theorem arithmetic_sequence_remainder (a₁ : ℕ) (d : ℕ) (aₙ : ℕ) (n : ℕ) :
  a₁ = 3 →
  d = 8 →
  aₙ = 283 →
  n = (aₙ - a₁) / d + 1 →
  (n * (a₁ + aₙ) / 2) % 8 = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_remainder_l3543_354385


namespace NUMINAMATH_CALUDE_parallelogram_area_l3543_354390

/-- The area of a parallelogram with base 32 cm and height 18 cm is 576 square centimeters. -/
theorem parallelogram_area : 
  ∀ (base height area : ℝ), 
  base = 32 → 
  height = 18 → 
  area = base * height → 
  area = 576 := by sorry

end NUMINAMATH_CALUDE_parallelogram_area_l3543_354390


namespace NUMINAMATH_CALUDE_drama_club_theorem_l3543_354339

theorem drama_club_theorem (total : ℕ) (math : ℕ) (physics : ℕ) (both : ℕ) 
  (h1 : total = 50)
  (h2 : math = 36)
  (h3 : physics = 27)
  (h4 : both = 20) :
  total - (math - both + physics - both + both) = 7 :=
by sorry

end NUMINAMATH_CALUDE_drama_club_theorem_l3543_354339


namespace NUMINAMATH_CALUDE_smallest_three_digit_prime_with_prime_reverse_l3543_354337

/-- A function that reverses the digits of a natural number -/
def reverseDigits (n : ℕ) : ℕ := sorry

/-- A function that checks if a natural number is prime -/
def isPrime (n : ℕ) : Prop := sorry

/-- A function that checks if a natural number has three digits -/
def hasThreeDigits (n : ℕ) : Prop := sorry

theorem smallest_three_digit_prime_with_prime_reverse : 
  (∀ n : ℕ, hasThreeDigits n → isPrime n → isPrime (reverseDigits n) → 107 ≤ n) ∧ 
  hasThreeDigits 107 ∧ 
  isPrime 107 ∧ 
  isPrime (reverseDigits 107) := by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_prime_with_prime_reverse_l3543_354337


namespace NUMINAMATH_CALUDE_prank_combinations_l3543_354314

/-- Represents the number of choices for each day of the week --/
def choices : List Nat := [1, 2, 6, 3, 1]

/-- Calculates the total number of combinations --/
def totalCombinations (choices : List Nat) : Nat :=
  choices.prod

/-- Theorem: The total number of combinations for the given choices is 36 --/
theorem prank_combinations :
  totalCombinations choices = 36 := by
  sorry

end NUMINAMATH_CALUDE_prank_combinations_l3543_354314


namespace NUMINAMATH_CALUDE_scarves_difference_formula_l3543_354365

/-- Calculates the difference in scarves produced between a normal day and a tiring day. -/
def scarves_difference (h : ℝ) : ℝ :=
  let s := 3 * h
  let normal_day := s * h
  let tiring_day := (s - 2) * (h - 3)
  normal_day - tiring_day

/-- Theorem stating that the difference in scarves produced is 11h - 6. -/
theorem scarves_difference_formula (h : ℝ) :
  scarves_difference h = 11 * h - 6 := by
  sorry

end NUMINAMATH_CALUDE_scarves_difference_formula_l3543_354365


namespace NUMINAMATH_CALUDE_fourth_post_length_l3543_354313

/-- Given a total rope length and the lengths used for the first three posts,
    calculate the length of rope used for the fourth post. -/
def rope_for_fourth_post (total : ℕ) (first : ℕ) (second : ℕ) (third : ℕ) : ℕ :=
  total - (first + second + third)

/-- Theorem stating that given the specific lengths in the problem,
    the rope used for the fourth post is 12 inches. -/
theorem fourth_post_length :
  rope_for_fourth_post 70 24 20 14 = 12 := by
  sorry

end NUMINAMATH_CALUDE_fourth_post_length_l3543_354313


namespace NUMINAMATH_CALUDE_compound_oxygen_atoms_l3543_354391

/-- Represents the number of atoms of each element in the compound -/
structure CompoundComposition where
  carbon : ℕ
  hydrogen : ℕ
  oxygen : ℕ

/-- Represents the atomic weights of elements -/
structure AtomicWeights where
  carbon : ℝ
  hydrogen : ℝ
  oxygen : ℝ

def molecularWeight (comp : CompoundComposition) (weights : AtomicWeights) : ℝ :=
  comp.carbon * weights.carbon + comp.hydrogen * weights.hydrogen + comp.oxygen * weights.oxygen

theorem compound_oxygen_atoms 
  (comp : CompoundComposition)
  (weights : AtomicWeights)
  (h1 : comp.carbon = 4)
  (h2 : comp.hydrogen = 8)
  (h3 : weights.carbon = 12.01)
  (h4 : weights.hydrogen = 1.008)
  (h5 : weights.oxygen = 16.00)
  (h6 : molecularWeight comp weights = 88) :
  comp.oxygen = 2 := by
sorry

end NUMINAMATH_CALUDE_compound_oxygen_atoms_l3543_354391


namespace NUMINAMATH_CALUDE_swan_population_after_ten_years_l3543_354354

/-- The number of swans after a given number of years, given an initial population and a doubling period -/
def swan_population (initial_population : ℕ) (doubling_period : ℕ) (years : ℕ) : ℕ :=
  initial_population * 2 ^ (years / doubling_period)

/-- Theorem stating that the swan population after 10 years will be 480, given the initial conditions -/
theorem swan_population_after_ten_years :
  swan_population 15 2 10 = 480 := by
sorry

end NUMINAMATH_CALUDE_swan_population_after_ten_years_l3543_354354


namespace NUMINAMATH_CALUDE_inequality_system_solution_l3543_354334

theorem inequality_system_solution (x : ℝ) :
  (2 * x + 1 > x + 3 ∧ 2 * x - 4 < x) ↔ (2 < x ∧ x < 4) := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l3543_354334


namespace NUMINAMATH_CALUDE_heartsuit_three_five_l3543_354348

-- Define the ⊛ operation
def heartsuit (x y : ℝ) : ℝ := 4 * x + 6 * y

-- State the theorem
theorem heartsuit_three_five : heartsuit 3 5 = 42 := by
  sorry

end NUMINAMATH_CALUDE_heartsuit_three_five_l3543_354348


namespace NUMINAMATH_CALUDE_intersection_complement_equals_l3543_354393

def U : Set ℕ := {x | x > 0 ∧ x < 9}
def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {3, 4, 5, 6}

theorem intersection_complement_equals : A ∩ (U \ B) = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equals_l3543_354393


namespace NUMINAMATH_CALUDE_donut_problem_l3543_354398

theorem donut_problem (D : ℕ) : (D - 6) / 2 = 22 ↔ D = 50 := by
  sorry

end NUMINAMATH_CALUDE_donut_problem_l3543_354398


namespace NUMINAMATH_CALUDE_original_number_proof_l3543_354392

theorem original_number_proof (x : ℝ) : x * 1.5 = 135 → x = 90 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l3543_354392


namespace NUMINAMATH_CALUDE_train_problem_solution_l3543_354335

/-- Represents the day of the week -/
inductive Day
  | Saturday
  | Monday

/-- Represents a date in a month -/
structure Date where
  day : Day
  number : Nat

/-- Represents a train car -/
structure TrainCar where
  number : Nat
  seat : Nat

/-- The problem setup -/
def TrainProblem (d1 d2 : Date) (car : TrainCar) : Prop :=
  d1.day = Day.Saturday ∧
  d2.day = Day.Monday ∧
  d2.number = car.number ∧
  car.seat < car.number ∧
  d1.number > car.number ∧
  d1.number ≠ d2.number ∧
  car.number < 10

theorem train_problem_solution :
  ∀ (d1 d2 : Date) (car : TrainCar),
    TrainProblem d1 d2 car →
    car.number = 2 ∧ car.seat = 1 :=
by
  sorry


end NUMINAMATH_CALUDE_train_problem_solution_l3543_354335


namespace NUMINAMATH_CALUDE_picnic_cost_is_60_l3543_354356

/-- Calculates the total cost of a picnic basket given the number of people and item costs. -/
def picnic_cost (num_people : ℕ) (sandwich_cost fruit_salad_cost soda_cost snack_cost : ℕ) 
  (num_sodas_per_person num_snack_bags : ℕ) : ℕ :=
  num_people * (sandwich_cost + fruit_salad_cost + num_sodas_per_person * soda_cost) + 
  num_snack_bags * snack_cost

/-- Theorem stating that the total cost of the picnic basket is $60. -/
theorem picnic_cost_is_60 : 
  picnic_cost 4 5 3 2 4 2 3 = 60 := by
  sorry

end NUMINAMATH_CALUDE_picnic_cost_is_60_l3543_354356


namespace NUMINAMATH_CALUDE_prob_10_or_9_prob_at_least_7_l3543_354361

/-- Represents the probabilities of hitting different rings in a shooting event -/
structure ShootingProbabilities where
  ring10 : ℝ
  ring9 : ℝ
  ring8 : ℝ
  ring7 : ℝ
  below7 : ℝ

/-- The probabilities sum to 1 -/
axiom prob_sum_to_one (p : ShootingProbabilities) : 
  p.ring10 + p.ring9 + p.ring8 + p.ring7 + p.below7 = 1

/-- All probabilities are non-negative -/
axiom prob_non_negative (p : ShootingProbabilities) : 
  p.ring10 ≥ 0 ∧ p.ring9 ≥ 0 ∧ p.ring8 ≥ 0 ∧ p.ring7 ≥ 0 ∧ p.below7 ≥ 0

/-- Given probabilities for a shooting event -/
def shooter_probs : ShootingProbabilities := {
  ring10 := 0.1,
  ring9 := 0.2,
  ring8 := 0.3,
  ring7 := 0.3,
  below7 := 0.1
}

/-- Theorem: The probability of hitting the 10 or 9 ring is 0.3 -/
theorem prob_10_or_9 : shooter_probs.ring10 + shooter_probs.ring9 = 0.3 := by sorry

/-- Theorem: The probability of hitting at least the 7 ring is 0.9 -/
theorem prob_at_least_7 : 1 - shooter_probs.below7 = 0.9 := by sorry

end NUMINAMATH_CALUDE_prob_10_or_9_prob_at_least_7_l3543_354361


namespace NUMINAMATH_CALUDE_intersection_points_properties_l3543_354366

/-- The curve equation -/
def curve (x : ℝ) : ℝ := x^2 - 5*x + 4

/-- The line equation -/
def line (p : ℝ) : ℝ := p

/-- Theorem stating the properties of the intersection points -/
theorem intersection_points_properties (a b p : ℝ) : 
  (curve a = line p) ∧ 
  (curve b = line p) ∧ 
  (a ≠ b) ∧
  (a^4 + b^4 = 1297) →
  (a = 6 ∧ b = -1) ∨ (a = -1 ∧ b = 6) := by
  sorry

#check intersection_points_properties

end NUMINAMATH_CALUDE_intersection_points_properties_l3543_354366


namespace NUMINAMATH_CALUDE_eight_weavers_eight_days_l3543_354303

/-- Represents the number of mats woven by a given number of mat-weavers in a given number of days. -/
def mats_woven (weavers : ℕ) (days : ℕ) : ℕ := sorry

/-- The rate at which mat-weavers work is constant. -/
axiom constant_rate : mats_woven 4 4 = 4

/-- Theorem stating that 8 mat-weavers can weave 16 mats in 8 days. -/
theorem eight_weavers_eight_days : mats_woven 8 8 = 16 := by sorry

end NUMINAMATH_CALUDE_eight_weavers_eight_days_l3543_354303


namespace NUMINAMATH_CALUDE_min_children_for_all_colors_l3543_354317

/-- Represents the distribution of pencils among children -/
structure PencilDistribution where
  total_pencils : ℕ
  num_colors : ℕ
  pencils_per_color : ℕ
  num_children : ℕ
  pencils_per_child : ℕ

/-- Theorem stating the minimum number of children to select to guarantee all colors -/
theorem min_children_for_all_colors (d : PencilDistribution) 
  (h1 : d.total_pencils = 24)
  (h2 : d.num_colors = 4)
  (h3 : d.pencils_per_color = 6)
  (h4 : d.num_children = 6)
  (h5 : d.pencils_per_child = 4)
  (h6 : d.total_pencils = d.num_colors * d.pencils_per_color)
  (h7 : d.total_pencils = d.num_children * d.pencils_per_child) :
  ∃ (n : ℕ), n = 5 ∧ 
  (∀ (m : ℕ), m < n → ¬(∀ (selection : Finset (Fin d.num_children)), 
    selection.card = m → 
    (∃ (colors : Finset (Fin d.num_colors)), colors.card = d.num_colors ∧
      ∀ (c : Fin d.num_colors), c ∈ colors → 
        ∃ (child : Fin d.num_children), child ∈ selection ∧ 
          ∃ (pencil : Fin d.pencils_per_child), pencil.val < d.pencils_per_child ∧
            (child.val * d.pencils_per_child + pencil.val) % d.num_colors = c.val))) ∧
  (∀ (selection : Finset (Fin d.num_children)), 
    selection.card = n → 
    (∃ (colors : Finset (Fin d.num_colors)), colors.card = d.num_colors ∧
      ∀ (c : Fin d.num_colors), c ∈ colors → 
        ∃ (child : Fin d.num_children), child ∈ selection ∧ 
          ∃ (pencil : Fin d.pencils_per_child), pencil.val < d.pencils_per_child ∧
            (child.val * d.pencils_per_child + pencil.val) % d.num_colors = c.val)) :=
by sorry

end NUMINAMATH_CALUDE_min_children_for_all_colors_l3543_354317


namespace NUMINAMATH_CALUDE_strawberry_cake_cost_l3543_354362

/-- Proves that the cost of each strawberry cake is $22 given the order details --/
theorem strawberry_cake_cost
  (num_chocolate : ℕ)
  (price_chocolate : ℕ)
  (num_strawberry : ℕ)
  (total_cost : ℕ)
  (h1 : num_chocolate = 3)
  (h2 : price_chocolate = 12)
  (h3 : num_strawberry = 6)
  (h4 : total_cost = 168)
  : (total_cost - num_chocolate * price_chocolate) / num_strawberry = 22 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_cake_cost_l3543_354362


namespace NUMINAMATH_CALUDE_angle_sum_is_345_l3543_354341

def is_acute_angle (θ : ℝ) : Prop := 0 < θ ∧ θ < 90

def is_obtuse_angle (θ : ℝ) : Prop := 90 < θ ∧ θ < 180

theorem angle_sum_is_345 (α β γ : ℝ) 
  (h1 : is_acute_angle α ∨ is_acute_angle β ∨ is_acute_angle γ)
  (h2 : is_acute_angle α ∨ is_acute_angle β ∨ is_acute_angle γ)
  (h3 : is_obtuse_angle α ∨ is_obtuse_angle β ∨ is_obtuse_angle γ)
  (h4 : (α + β + γ) / 15 = 23 ∨ (α + β + γ) / 15 = 24 ∨ (α + β + γ) / 15 = 25) :
  α + β + γ = 345 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_is_345_l3543_354341


namespace NUMINAMATH_CALUDE_x_squared_minus_y_squared_l3543_354395

theorem x_squared_minus_y_squared (x y : ℝ) 
  (h1 : x + y = 12) 
  (h2 : 3 * x + y = 18) : 
  x^2 - y^2 = -72 := by
sorry

end NUMINAMATH_CALUDE_x_squared_minus_y_squared_l3543_354395


namespace NUMINAMATH_CALUDE_average_age_of_nine_students_l3543_354353

theorem average_age_of_nine_students 
  (total_students : ℕ)
  (total_average : ℚ)
  (five_students : ℕ)
  (five_average : ℚ)
  (fifteenth_student_age : ℕ)
  (h1 : total_students = 15)
  (h2 : total_average = 15)
  (h3 : five_students = 5)
  (h4 : five_average = 13)
  (h5 : fifteenth_student_age = 16) :
  (total_students * total_average - five_students * five_average - fifteenth_student_age) / (total_students - five_students - 1) = 16 := by
  sorry

end NUMINAMATH_CALUDE_average_age_of_nine_students_l3543_354353


namespace NUMINAMATH_CALUDE_coin_found_in_33_moves_l3543_354352

/-- Represents the game state with thimbles and a hidden coin. -/
structure ThimbleGame where
  num_thimbles : Nat
  thimbles_per_move : Nat

/-- Calculates the maximum number of moves needed to guarantee finding the coin. -/
def max_moves_to_find_coin (game : ThimbleGame) : Nat :=
  sorry

/-- Theorem stating that for 100 thimbles and 4 checks per move, 33 moves are sufficient. -/
theorem coin_found_in_33_moves :
  let game : ThimbleGame := { num_thimbles := 100, thimbles_per_move := 4 }
  max_moves_to_find_coin game ≤ 33 := by
  sorry

end NUMINAMATH_CALUDE_coin_found_in_33_moves_l3543_354352


namespace NUMINAMATH_CALUDE_simple_interest_problem_l3543_354324

/-- Given a principal amount P and an interest rate R (as a percentage),
    if the amount after 2 years is 720 and after 7 years is 1020,
    then the principal amount P is 600. -/
theorem simple_interest_problem (P R : ℚ) : 
  P + (P * R * 2) / 100 = 720 →
  P + (P * R * 7) / 100 = 1020 →
  P = 600 := by
sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l3543_354324


namespace NUMINAMATH_CALUDE_banana_cost_l3543_354355

-- Define the rate of bananas
def banana_rate : ℚ := 3 / 4

-- Define the amount of bananas to buy
def banana_amount : ℚ := 20

-- Theorem to prove
theorem banana_cost : banana_amount * banana_rate = 15 := by
  sorry

end NUMINAMATH_CALUDE_banana_cost_l3543_354355


namespace NUMINAMATH_CALUDE_expected_value_girls_selected_l3543_354342

/-- The expected value of girls selected in a hypergeometric distribution -/
theorem expected_value_girls_selected (total : ℕ) (girls : ℕ) (sample : ℕ)
  (h_total : total = 8)
  (h_girls : girls = 3)
  (h_sample : sample = 2) :
  (girls : ℚ) / total * sample = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_expected_value_girls_selected_l3543_354342


namespace NUMINAMATH_CALUDE_pentagon_square_angle_sum_l3543_354349

theorem pentagon_square_angle_sum : 
  ∀ (pentagon_angle square_angle : ℝ),
  (pentagon_angle = 180 * (5 - 2) / 5) →
  (square_angle = 180 * (4 - 2) / 4) →
  pentagon_angle + square_angle = 198 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_square_angle_sum_l3543_354349


namespace NUMINAMATH_CALUDE_sum_of_digits_theorem_l3543_354389

-- Define the set A
def A : Set Int := {m | ∃ p q : Int, p > 0 ∧ q > 0 ∧ p * q = 2020 ∧ p + q = -m}

-- Define the set B
def B : Set Int := {n | ∃ r s : Int, r > 0 ∧ s > 0 ∧ r * s = n ∧ r + s = 2020}

-- Define a function to calculate the sum of digits
def sumOfDigits (n : Int) : Nat :=
  (n.natAbs.digits 10).sum

-- State the theorem
theorem sum_of_digits_theorem :
  ∃ a b : Int, a ∈ A ∧ b ∈ B ∧ (∀ m ∈ A, m ≤ a) ∧ (∀ n ∈ B, b ≤ n) ∧
  sumOfDigits (a + b) = 27 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_theorem_l3543_354389


namespace NUMINAMATH_CALUDE_hyperbola_center_l3543_354325

/-- The center of a hyperbola is the midpoint of its foci -/
theorem hyperbola_center (f1 f2 : ℝ × ℝ) :
  let center := ((f1.1 + f2.1) / 2, (f1.2 + f2.2) / 2)
  f1 = (2, 0) ∧ f2 = (8, 6) →
  center = (5, 3) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_center_l3543_354325


namespace NUMINAMATH_CALUDE_infinitely_many_n_satisfying_conditions_l3543_354394

theorem infinitely_many_n_satisfying_conditions :
  ∀ k : ℕ, k > 0 →
  let n := k * (k + 1)
  ∃ m : ℕ, m^2 < n ∧ n < (m + 1)^2 ∧ n % m = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_n_satisfying_conditions_l3543_354394


namespace NUMINAMATH_CALUDE_function_transformation_l3543_354350

-- Define the given function
def f (x : ℝ) : ℝ := sorry

-- State the theorem
theorem function_transformation :
  (∀ x, f (2 * x + 1) = x^2 - 2*x) →
  (∀ x, f x = x^2 / 4 - (3/2) * x + 5/4) := by sorry

end NUMINAMATH_CALUDE_function_transformation_l3543_354350


namespace NUMINAMATH_CALUDE_math_club_teams_l3543_354379

theorem math_club_teams (girls boys : ℕ) (h1 : girls = 4) (h2 : boys = 6) :
  (girls.choose 2) * (boys.choose 2) = 90 := by
sorry

end NUMINAMATH_CALUDE_math_club_teams_l3543_354379


namespace NUMINAMATH_CALUDE_keys_for_52_phones_l3543_354359

/-- Represents the warehouse setup and the task of retrieving phones -/
structure WarehouseSetup where
  total_cabinets : ℕ
  boxes_per_cabinet : ℕ
  phones_per_box : ℕ
  phones_to_retrieve : ℕ

/-- Calculates the minimum number of keys required to retrieve the specified number of phones -/
def min_keys_required (setup : WarehouseSetup) : ℕ :=
  let boxes_needed := (setup.phones_to_retrieve + setup.phones_per_box - 1) / setup.phones_per_box
  let cabinets_needed := (boxes_needed + setup.boxes_per_cabinet - 1) / setup.boxes_per_cabinet
  boxes_needed + cabinets_needed + 1

/-- The theorem stating that for the given setup, 9 keys are required -/
theorem keys_for_52_phones :
  let setup : WarehouseSetup := {
    total_cabinets := 8,
    boxes_per_cabinet := 4,
    phones_per_box := 10,
    phones_to_retrieve := 52
  }
  min_keys_required setup = 9 := by
  sorry

end NUMINAMATH_CALUDE_keys_for_52_phones_l3543_354359


namespace NUMINAMATH_CALUDE_sequence_existence_iff_k_in_range_l3543_354305

theorem sequence_existence_iff_k_in_range (n : ℕ) :
  (∃ (x : ℕ → ℕ), (∀ i j, i < j → i ≤ n → j ≤ n → x i < x j)) ↔
  (∀ k : ℕ, k ≤ n → ∃ (x : ℕ → ℕ), (∀ i j, i < j → i ≤ k → j ≤ k → x i < x j)) :=
by sorry

end NUMINAMATH_CALUDE_sequence_existence_iff_k_in_range_l3543_354305


namespace NUMINAMATH_CALUDE_overall_profit_percentage_l3543_354387

/-- Calculate the overall profit percentage for four items --/
theorem overall_profit_percentage
  (sp_a : ℝ) (cp_percent_a : ℝ)
  (sp_b : ℝ) (cp_percent_b : ℝ)
  (sp_c : ℝ) (cp_percent_c : ℝ)
  (sp_d : ℝ) (cp_percent_d : ℝ)
  (h_sp_a : sp_a = 120)
  (h_cp_percent_a : cp_percent_a = 30)
  (h_sp_b : sp_b = 200)
  (h_cp_percent_b : cp_percent_b = 20)
  (h_sp_c : sp_c = 75)
  (h_cp_percent_c : cp_percent_c = 40)
  (h_sp_d : sp_d = 180)
  (h_cp_percent_d : cp_percent_d = 25) :
  let cp_a := sp_a * (cp_percent_a / 100)
  let cp_b := sp_b * (cp_percent_b / 100)
  let cp_c := sp_c * (cp_percent_c / 100)
  let cp_d := sp_d * (cp_percent_d / 100)
  let total_cp := cp_a + cp_b + cp_c + cp_d
  let total_sp := sp_a + sp_b + sp_c + sp_d
  let total_profit := total_sp - total_cp
  let profit_percentage := (total_profit / total_cp) * 100
  abs (profit_percentage - 280.79) < 0.01 := by
sorry

end NUMINAMATH_CALUDE_overall_profit_percentage_l3543_354387


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l3543_354316

/-- The quadratic equation x^2 + ax + a = 0 has no real roots -/
def has_no_real_roots (a : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + a*x + a ≠ 0

/-- The condition 0 ≤ a ≤ 4 is necessary but not sufficient for x^2 + ax + a = 0 to have no real roots -/
theorem necessary_but_not_sufficient :
  (∀ a : ℝ, has_no_real_roots a → 0 ≤ a ∧ a ≤ 4) ∧
  (∃ a : ℝ, 0 ≤ a ∧ a ≤ 4 ∧ ¬(has_no_real_roots a)) :=
sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l3543_354316


namespace NUMINAMATH_CALUDE_three_digit_palindrome_squares_l3543_354377

/-- A number is a 3-digit palindrome square if it satisfies these conditions:
1. It is between 100 and 999 (inclusive).
2. It is a perfect square.
3. It reads the same forward and backward. -/
def is_three_digit_palindrome_square (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧
  ∃ k, n = k^2 ∧
  (n / 100 = n % 10) ∧ (n / 10 % 10 = (n / 10) % 10)

/-- There are exactly 3 numbers that are 3-digit palindrome squares. -/
theorem three_digit_palindrome_squares :
  ∃! (s : Finset ℕ), (∀ n ∈ s, is_three_digit_palindrome_square n) ∧ s.card = 3 :=
sorry

end NUMINAMATH_CALUDE_three_digit_palindrome_squares_l3543_354377


namespace NUMINAMATH_CALUDE_range_of_sin_cos_function_l3543_354370

theorem range_of_sin_cos_function : 
  ∀ x : ℝ, 3/4 ≤ Real.sin x ^ 4 + Real.cos x ^ 2 ∧ 
  Real.sin x ^ 4 + Real.cos x ^ 2 ≤ 1 ∧
  ∃ y z : ℝ, Real.sin y ^ 4 + Real.cos y ^ 2 = 3/4 ∧
            Real.sin z ^ 4 + Real.cos z ^ 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_sin_cos_function_l3543_354370


namespace NUMINAMATH_CALUDE_haleigh_leggings_count_l3543_354384

/-- The number of leggings needed for Haleigh's pets -/
def total_leggings : ℕ :=
  let num_dogs := 4
  let num_cats := 3
  let num_spiders := 2
  let num_parrots := 1
  let dog_legs := 4
  let cat_legs := 4
  let spider_legs := 8
  let parrot_legs := 2
  num_dogs * dog_legs + num_cats * cat_legs + num_spiders * spider_legs + num_parrots * parrot_legs

theorem haleigh_leggings_count : total_leggings = 46 := by
  sorry

end NUMINAMATH_CALUDE_haleigh_leggings_count_l3543_354384


namespace NUMINAMATH_CALUDE_log_relation_l3543_354310

theorem log_relation (y : ℝ) (m : ℝ) : 
  (Real.log 5 / Real.log 8 = y) → 
  (Real.log 125 / Real.log 2 = m * y) → 
  m = 9 := by
  sorry

end NUMINAMATH_CALUDE_log_relation_l3543_354310


namespace NUMINAMATH_CALUDE_mia_fruit_probability_l3543_354380

def num_fruit_types : ℕ := 4
def num_meals : ℕ := 4

/-- The probability of choosing the same fruit for all meals -/
def prob_same_fruit : ℚ := (1 / num_fruit_types) ^ num_meals

/-- The probability of eating at least two different kinds of fruit -/
def prob_different_fruits : ℚ := 1 - (num_fruit_types * prob_same_fruit)

theorem mia_fruit_probability :
  prob_different_fruits = 63 / 64 :=
sorry

end NUMINAMATH_CALUDE_mia_fruit_probability_l3543_354380


namespace NUMINAMATH_CALUDE_f_8_equals_60_l3543_354358

-- Define the function f
def f (n : ℤ) : ℤ := n^2 - 3*n + 20

-- Theorem statement
theorem f_8_equals_60 : f 8 = 60 := by
  sorry

end NUMINAMATH_CALUDE_f_8_equals_60_l3543_354358


namespace NUMINAMATH_CALUDE_leaves_per_frond_l3543_354322

theorem leaves_per_frond (num_ferns : ℕ) (fronds_per_fern : ℕ) (total_leaves : ℕ) :
  num_ferns = 6 →
  fronds_per_fern = 7 →
  total_leaves = 1260 →
  total_leaves / (num_ferns * fronds_per_fern) = 30 :=
by sorry

end NUMINAMATH_CALUDE_leaves_per_frond_l3543_354322


namespace NUMINAMATH_CALUDE_fencing_calculation_l3543_354346

/-- Represents a rectangular field with fencing on three sides -/
structure FencedField where
  length : ℝ
  width : ℝ
  area : ℝ
  uncovered_side : ℝ

/-- Calculates the required fencing for a given field -/
def required_fencing (field : FencedField) : ℝ :=
  field.length + 2 * field.width

theorem fencing_calculation (field : FencedField) 
  (h1 : field.area = 680)
  (h2 : field.uncovered_side = 34)
  (h3 : field.area = field.length * field.width)
  (h4 : field.length = field.uncovered_side) :
  required_fencing field = 74 := by
  sorry

#check fencing_calculation

end NUMINAMATH_CALUDE_fencing_calculation_l3543_354346


namespace NUMINAMATH_CALUDE_sin_tan_product_l3543_354386

/-- Given an angle α whose terminal side intersects the unit circle at point P(-1/2, y),
    prove that sinα•tanα = -3/2 -/
theorem sin_tan_product (α : Real) (y : Real) 
    (h1 : Real.cos α = -1/2)  -- x-coordinate of P is -1/2
    (h2 : Real.sin α = y)     -- y-coordinate of P is y
    (h3 : (-1/2)^2 + y^2 = 1) -- P is on the unit circle
    : Real.sin α * Real.tan α = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_tan_product_l3543_354386


namespace NUMINAMATH_CALUDE_toms_original_amount_l3543_354333

theorem toms_original_amount (tom sara jim : ℝ) : 
  tom + sara + jim = 1200 →
  (tom - 200) + (3 * sara) + (2 * jim) = 1800 →
  tom = 400 := by
sorry

end NUMINAMATH_CALUDE_toms_original_amount_l3543_354333


namespace NUMINAMATH_CALUDE_rhombus_existence_and_uniqueness_l3543_354360

/-- Represents a rhombus -/
structure Rhombus where
  side : ℝ
  diag1 : ℝ
  diag2 : ℝ
  angle : ℝ

/-- Given the sum of diagonals and an opposite angle, a unique rhombus can be determined -/
theorem rhombus_existence_and_uniqueness 
  (diag_sum : ℝ) 
  (opp_angle : ℝ) 
  (h_pos : diag_sum > 0) 
  (h_angle : 0 < opp_angle ∧ opp_angle < π) :
  ∃! r : Rhombus, r.diag1 + r.diag2 = diag_sum ∧ r.angle = opp_angle :=
sorry

end NUMINAMATH_CALUDE_rhombus_existence_and_uniqueness_l3543_354360


namespace NUMINAMATH_CALUDE_power_of_i_2023_l3543_354323

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem power_of_i_2023 : i ^ 2023 = -i := by
  sorry

end NUMINAMATH_CALUDE_power_of_i_2023_l3543_354323


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3543_354306

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  x₁ = -1 + Real.sqrt 2 ∧ 
  x₂ = -1 - Real.sqrt 2 ∧ 
  x₁^2 + 2*x₁ - 1 = 0 ∧ 
  x₂^2 + 2*x₂ - 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3543_354306


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l3543_354326

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- State the theorem
theorem geometric_sequence_property (a : ℕ → ℝ) :
  is_geometric_sequence a → a 2 * a 6 = 4 → (a 4 = 2 ∨ a 4 = -2) :=
by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_property_l3543_354326


namespace NUMINAMATH_CALUDE_compare_powers_l3543_354300

theorem compare_powers (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a < 1) (hba : 1 < b) :
  a^4 < 1 ∧ 1 < b^(1/4) := by
  sorry

end NUMINAMATH_CALUDE_compare_powers_l3543_354300


namespace NUMINAMATH_CALUDE_volunteer_allocation_l3543_354351

theorem volunteer_allocation (n : ℕ) (k : ℕ) (h1 : n = 10) (h2 : k = 4) :
  (Nat.choose (n - 1) (k - 1) : ℕ) = 84 := by
  sorry

end NUMINAMATH_CALUDE_volunteer_allocation_l3543_354351


namespace NUMINAMATH_CALUDE_combined_work_theorem_l3543_354312

/-- The number of days it takes for three workers to complete a task together,
    given their individual completion times. -/
def combinedWorkDays (raviDays prakashDays seemaDays : ℚ) : ℚ :=
  1 / (1 / raviDays + 1 / prakashDays + 1 / seemaDays)

/-- Theorem stating that if Ravi can do the work in 50 days, Prakash in 75 days,
    and Seema in 60 days, they will finish the work together in 20 days. -/
theorem combined_work_theorem :
  combinedWorkDays 50 75 60 = 20 := by sorry

end NUMINAMATH_CALUDE_combined_work_theorem_l3543_354312


namespace NUMINAMATH_CALUDE_final_cat_count_l3543_354302

/-- Represents the number of cats of each breed -/
structure CatInventory where
  siamese : ℕ
  house : ℕ
  persian : ℕ
  sphynx : ℕ

/-- Calculates the total number of cats -/
def totalCats (inventory : CatInventory) : ℕ :=
  inventory.siamese + inventory.house + inventory.persian + inventory.sphynx

/-- Represents a sale event -/
structure SaleEvent where
  siamese : ℕ
  house : ℕ
  persian : ℕ
  sphynx : ℕ

/-- Applies a sale event to the inventory -/
def applySale (inventory : CatInventory) (sale : SaleEvent) : CatInventory where
  siamese := inventory.siamese - sale.siamese
  house := inventory.house - sale.house
  persian := inventory.persian - sale.persian
  sphynx := inventory.sphynx - sale.sphynx

/-- Adds new cats to the inventory -/
def addNewCats (inventory : CatInventory) (newSiamese newPersian : ℕ) : CatInventory where
  siamese := inventory.siamese + newSiamese
  house := inventory.house
  persian := inventory.persian + newPersian
  sphynx := inventory.sphynx

theorem final_cat_count (initialInventory : CatInventory)
    (sale1 sale2 : SaleEvent) (newSiamese newPersian : ℕ)
    (h1 : initialInventory = CatInventory.mk 12 20 8 18)
    (h2 : sale1 = SaleEvent.mk 6 4 5 0)
    (h3 : sale2 = SaleEvent.mk 0 15 0 10)
    (h4 : newSiamese = 5)
    (h5 : newPersian = 3) :
    totalCats (addNewCats (applySale (applySale initialInventory sale1) sale2) newSiamese newPersian) = 26 := by
  sorry


end NUMINAMATH_CALUDE_final_cat_count_l3543_354302


namespace NUMINAMATH_CALUDE_largest_unexpressible_sum_l3543_354378

def min_num : ℕ := 135
def max_num : ℕ := 144
def target : ℕ := 2024

theorem largest_unexpressible_sum : 
  (∀ n : ℕ, n > target → ∃ k : ℕ, k * min_num ≤ n ∧ n ≤ k * max_num) ∧
  (∀ k : ℕ, k * min_num > target ∨ target > k * max_num) :=
sorry

end NUMINAMATH_CALUDE_largest_unexpressible_sum_l3543_354378


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_l3543_354388

theorem gcd_lcm_sum : Nat.gcd 30 81 + Nat.lcm 36 12 = 39 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_l3543_354388


namespace NUMINAMATH_CALUDE_alternative_plan_savings_l3543_354368

/-- Proves that the alternative phone plan is $1 cheaper than the current plan -/
theorem alternative_plan_savings :
  ∀ (current_plan_cost : ℚ)
    (texts_sent : ℕ)
    (call_minutes : ℕ)
    (text_package_size : ℕ)
    (call_package_size : ℕ)
    (text_package_cost : ℚ)
    (call_package_cost : ℚ),
  current_plan_cost = 12 →
  texts_sent = 60 →
  call_minutes = 60 →
  text_package_size = 30 →
  call_package_size = 20 →
  text_package_cost = 1 →
  call_package_cost = 3 →
  current_plan_cost - 
    ((texts_sent / text_package_size : ℚ) * text_package_cost +
     (call_minutes / call_package_size : ℚ) * call_package_cost) = 1 :=
by
  sorry

#check alternative_plan_savings

end NUMINAMATH_CALUDE_alternative_plan_savings_l3543_354368


namespace NUMINAMATH_CALUDE_at_least_one_not_less_than_two_l3543_354332

theorem at_least_one_not_less_than_two 
  (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  (a + 1/b ≥ 2) ∨ (b + 1/c ≥ 2) ∨ (c + 1/a ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_not_less_than_two_l3543_354332


namespace NUMINAMATH_CALUDE_cubic_equation_roots_l3543_354367

theorem cubic_equation_roots (p q : ℝ) : 
  (Complex.I : ℂ).re = 0 ∧ (Complex.I : ℂ).im = 1 →
  (2 - Complex.I : ℂ) ^ 3 + p * (2 - Complex.I : ℂ) ^ 2 + q * (2 - Complex.I : ℂ) - 6 = 0 →
  p = -26/5 ∧ q = 49/5 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_l3543_354367
