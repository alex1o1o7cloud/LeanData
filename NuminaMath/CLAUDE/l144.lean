import Mathlib

namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l144_14434

theorem cube_volume_from_surface_area (S : ℝ) (h : S = 294) :
  let s := Real.sqrt (S / 6)
  s ^ 3 = 343 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l144_14434


namespace NUMINAMATH_CALUDE_probability_of_matching_pair_l144_14483

def blue_socks : ℕ := 12
def red_socks : ℕ := 10

def total_socks : ℕ := blue_socks + red_socks

def ways_to_pick_two (n : ℕ) : ℕ := n * (n - 1) / 2

def matching_pairs : ℕ := ways_to_pick_two blue_socks + ways_to_pick_two red_socks

def total_combinations : ℕ := ways_to_pick_two total_socks

theorem probability_of_matching_pair :
  (matching_pairs : ℚ) / total_combinations = 111 / 231 := by sorry

end NUMINAMATH_CALUDE_probability_of_matching_pair_l144_14483


namespace NUMINAMATH_CALUDE_problem_solution_l144_14460

theorem problem_solution (x y : ℝ) (h1 : y = Real.log (2 * x)) (h2 : x + y = 2) :
  (Real.exp x + Real.exp y > 2 * Real.exp 1) ∧ (x * Real.log x + y * Real.log y > 0) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l144_14460


namespace NUMINAMATH_CALUDE_water_amount_in_new_recipe_l144_14417

/-- Represents the ratio of ingredients in a recipe --/
structure RecipeRatio :=
  (flour : ℚ)
  (water : ℚ)
  (sugar : ℚ)

/-- The original recipe ratio --/
def original_ratio : RecipeRatio :=
  { flour := 11, water := 8, sugar := 1 }

/-- The new recipe ratio --/
def new_ratio : RecipeRatio :=
  { flour := 22, water := 8, sugar := 4 }

/-- The amount of sugar in the new recipe --/
def new_sugar_amount : ℚ := 2

theorem water_amount_in_new_recipe :
  let water_amount := (new_ratio.water / new_ratio.sugar) * new_sugar_amount
  water_amount = 4 := by
  sorry

#check water_amount_in_new_recipe

end NUMINAMATH_CALUDE_water_amount_in_new_recipe_l144_14417


namespace NUMINAMATH_CALUDE_lennon_reimbursement_l144_14414

/-- Calculates the total reimbursement for a sales rep given daily mileage and reimbursement rate -/
def calculate_reimbursement (monday : ℕ) (tuesday : ℕ) (wednesday : ℕ) (thursday : ℕ) (friday : ℕ) (rate : ℚ) : ℚ :=
  (monday + tuesday + wednesday + thursday + friday : ℚ) * rate

/-- Proves that the total reimbursement for Lennon's mileage is $36 -/
theorem lennon_reimbursement :
  calculate_reimbursement 18 26 20 20 16 (36/100) = 36 := by
  sorry

end NUMINAMATH_CALUDE_lennon_reimbursement_l144_14414


namespace NUMINAMATH_CALUDE_trajectory_of_Q_l144_14447

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the point M on circle C
def point_M (x₀ y₀ : ℝ) : Prop := circle_C x₀ y₀

-- Define the vector ON
def vector_ON (y₀ : ℝ) : ℝ × ℝ := (0, y₀)

-- Define the vector OQ as the sum of OM and ON
def vector_OQ (x₀ y₀ : ℝ) : ℝ × ℝ := (x₀, 2 * y₀)

-- State the theorem
theorem trajectory_of_Q (x y : ℝ) :
  (∃ x₀ y₀ : ℝ, point_M x₀ y₀ ∧ vector_OQ x₀ y₀ = (x, y)) →
  x^2/4 + y^2/16 = 1 :=
sorry

end NUMINAMATH_CALUDE_trajectory_of_Q_l144_14447


namespace NUMINAMATH_CALUDE_pure_imaginary_sixth_power_l144_14443

theorem pure_imaginary_sixth_power (a : ℝ) (z : ℂ) :
  z = a + (a + 1) * Complex.I →
  z.im ≠ 0 →
  z.re = 0 →
  z^6 = -1 := by sorry

end NUMINAMATH_CALUDE_pure_imaginary_sixth_power_l144_14443


namespace NUMINAMATH_CALUDE_parabola_coefficient_b_l144_14408

/-- Given a parabola y = ax^2 + bx + c with vertex at (h, -h) and y-intercept at (0, h),
    where h ≠ 0, the coefficient b equals -4. -/
theorem parabola_coefficient_b (a b c h : ℝ) : 
  h ≠ 0 →
  (∀ x, a * x^2 + b * x + c = a * (x - h)^2 - h) →
  c = h →
  b = -4 := by sorry

end NUMINAMATH_CALUDE_parabola_coefficient_b_l144_14408


namespace NUMINAMATH_CALUDE_amulet_price_is_40_l144_14450

/-- Calculates the selling price of amulets given the following conditions:
  * Dirk sells amulets for 2 days
  * Each day he sells 25 amulets
  * Each amulet costs him $30 to make
  * He gives 10% of his revenue to the faire
  * He made a profit of $300
-/
def amulet_price (days : ℕ) (amulets_per_day : ℕ) (cost_per_amulet : ℕ) 
                 (faire_percentage : ℚ) (profit : ℕ) : ℚ :=
  let total_amulets := days * amulets_per_day
  let total_cost := total_amulets * cost_per_amulet
  let x := (profit + total_cost) / (total_amulets * (1 - faire_percentage))
  x

theorem amulet_price_is_40 :
  amulet_price 2 25 30 (1/10) 300 = 40 := by
  sorry

end NUMINAMATH_CALUDE_amulet_price_is_40_l144_14450


namespace NUMINAMATH_CALUDE_odd_numbers_equality_l144_14487

theorem odd_numbers_equality (a b c d k m : ℕ) : 
  Odd a → Odd b → Odd c → Odd d →
  0 < a → a < b → b < c → c < d →
  a * d = b * c →
  a + d = 2^k →
  b + c = 2^m →
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_odd_numbers_equality_l144_14487


namespace NUMINAMATH_CALUDE_continuous_piecewise_function_sum_l144_14477

noncomputable def g (a c : ℝ) (x : ℝ) : ℝ :=
  if x > 3 then 2 * a * x + 4
  else if -3 ≤ x ∧ x ≤ 3 then x^2 - 7
  else 3 * x - c

def IsContinuous (f : ℝ → ℝ) : Prop :=
  ∀ x₀ : ℝ, ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, |x - x₀| < δ → |f x - f x₀| < ε

theorem continuous_piecewise_function_sum (a c : ℝ) :
  IsContinuous (g a c) → a + c = -34/3 := by
  sorry

end NUMINAMATH_CALUDE_continuous_piecewise_function_sum_l144_14477


namespace NUMINAMATH_CALUDE_complex_magnitude_l144_14438

theorem complex_magnitude (z : ℂ) (h : (z + 2) / (z - 2) = Complex.I) : Complex.abs z = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l144_14438


namespace NUMINAMATH_CALUDE_shirt_discount_percentage_l144_14406

/-- Calculates the discount percentage for a shirt given its cost price, profit margin, and sale price. -/
theorem shirt_discount_percentage
  (cost_price : ℝ)
  (profit_margin : ℝ)
  (sale_price : ℝ)
  (h1 : cost_price = 20)
  (h2 : profit_margin = 0.3)
  (h3 : sale_price = 13) :
  (cost_price * (1 + profit_margin) - sale_price) / (cost_price * (1 + profit_margin)) = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_shirt_discount_percentage_l144_14406


namespace NUMINAMATH_CALUDE_semicircle_perimeter_approx_l144_14492

/-- The perimeter of a semicircle with radius 12 units is approximately 61.7 units. -/
theorem semicircle_perimeter_approx : ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ 
  |π * 12 + 24 - 61.7| < ε := by
  sorry

end NUMINAMATH_CALUDE_semicircle_perimeter_approx_l144_14492


namespace NUMINAMATH_CALUDE_real_part_of_complex_fraction_l144_14431

theorem real_part_of_complex_fraction : 
  let z : ℂ := (2 * Complex.I) / (1 - Complex.I)
  Complex.re z = -1 := by sorry

end NUMINAMATH_CALUDE_real_part_of_complex_fraction_l144_14431


namespace NUMINAMATH_CALUDE_x_fourth_coefficient_equals_a_9_l144_14429

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The sequence a_n = 2n + 2 -/
def a (n : ℕ) : ℕ := 2 * n + 2

/-- The theorem to prove -/
theorem x_fourth_coefficient_equals_a_9 :
  binomial 5 4 + binomial 6 4 = a 9 := by sorry

end NUMINAMATH_CALUDE_x_fourth_coefficient_equals_a_9_l144_14429


namespace NUMINAMATH_CALUDE_average_of_combined_results_l144_14467

theorem average_of_combined_results (n₁ : ℕ) (n₂ : ℕ) (avg₁ : ℝ) (avg₂ : ℝ) :
  n₁ = 60 →
  n₂ = 40 →
  avg₁ = 40 →
  avg₂ = 60 →
  (n₁ * avg₁ + n₂ * avg₂) / (n₁ + n₂) = 48 := by
  sorry

end NUMINAMATH_CALUDE_average_of_combined_results_l144_14467


namespace NUMINAMATH_CALUDE_valid_selections_count_l144_14439

/-- Represents a 6x6 grid of blocks -/
def Grid := Fin 6 → Fin 6 → Bool

/-- Represents a selection of 4 blocks on the grid -/
def Selection := Fin 4 → (Fin 6 × Fin 6)

/-- Checks if a selection forms an L shape -/
def is_L_shape (s : Selection) : Prop := sorry

/-- Checks if no two blocks in the selection share a row or column -/
def no_shared_row_col (s : Selection) : Prop := sorry

/-- The number of valid selections -/
def num_valid_selections : ℕ := sorry

theorem valid_selections_count :
  num_valid_selections = 1800 := by sorry

end NUMINAMATH_CALUDE_valid_selections_count_l144_14439


namespace NUMINAMATH_CALUDE_extrema_relations_l144_14416

noncomputable def f (x : ℝ) : ℝ := (x - 1) / (x^2 + 1)

theorem extrema_relations (a b : ℝ) 
  (h1 : ∀ x, f x ≥ a) 
  (h2 : ∃ x, f x = a)
  (h3 : ∀ x, f x ≤ b) 
  (h4 : ∃ x, f x = b) :
  (∀ x, (x^3 - 1) / (x^6 + 1) ≥ a) ∧
  (∃ x, (x^3 - 1) / (x^6 + 1) = a) ∧
  (∀ x, (x^3 - 1) / (x^6 + 1) ≤ b) ∧
  (∃ x, (x^3 - 1) / (x^6 + 1) = b) ∧
  (∀ x, (x + 1) / (x^2 + 1) ≥ -b) ∧
  (∃ x, (x + 1) / (x^2 + 1) = -b) ∧
  (∀ x, (x + 1) / (x^2 + 1) ≤ -a) ∧
  (∃ x, (x + 1) / (x^2 + 1) = -a) :=
by sorry

end NUMINAMATH_CALUDE_extrema_relations_l144_14416


namespace NUMINAMATH_CALUDE_smallest_solution_absolute_value_equation_l144_14444

theorem smallest_solution_absolute_value_equation :
  let x : ℝ := (-3 - Real.sqrt 17) / 2
  (∀ y : ℝ, y * |y| = 3 * y - 2 → x ≤ y) ∧ (x * |x| = 3 * x - 2) := by
  sorry

end NUMINAMATH_CALUDE_smallest_solution_absolute_value_equation_l144_14444


namespace NUMINAMATH_CALUDE_smallest_dual_base_representation_l144_14475

def is_valid_representation (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a > 2 ∧ b > 2 ∧
  n = 2 * a + 1 ∧
  n = b + 2

theorem smallest_dual_base_representation :
  (is_valid_representation 7) ∧
  (∀ m : ℕ, m < 7 → ¬(is_valid_representation m)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_dual_base_representation_l144_14475


namespace NUMINAMATH_CALUDE_det_special_matrix_l144_14486

theorem det_special_matrix (x : ℝ) : 
  Matrix.det !![x + 2, x, x; x, x + 2, x; x, x, x + 2] = 8 * x + 8 := by
  sorry

end NUMINAMATH_CALUDE_det_special_matrix_l144_14486


namespace NUMINAMATH_CALUDE_arithmetic_triangle_inradius_l144_14422

/-- A triangle with sides in arithmetic progression and an inscribed circle -/
structure ArithmeticTriangle where
  -- The three sides of the triangle
  a : ℝ
  b : ℝ
  c : ℝ
  -- The sides form an arithmetic progression
  progression : ∃ d : ℝ, b = a + d ∧ c = a + 2*d
  -- The triangle is valid (sum of any two sides is greater than the third)
  valid : a + b > c ∧ b + c > a ∧ c + a > b
  -- The triangle has positive area
  positive_area : a > 0 ∧ b > 0 ∧ c > 0
  -- The inscribed circle exists
  inradius : ℝ
  -- One of the altitudes
  altitude : ℝ

/-- 
The radius of the inscribed circle of a triangle with sides in arithmetic progression 
is equal to 1/3 of one of its altitudes
-/
theorem arithmetic_triangle_inradius (t : ArithmeticTriangle) : 
  t.inradius = (1/3) * t.altitude := by sorry

end NUMINAMATH_CALUDE_arithmetic_triangle_inradius_l144_14422


namespace NUMINAMATH_CALUDE_unique_four_digit_palindromic_square_l144_14461

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

theorem unique_four_digit_palindromic_square : 
  ∃! n : ℕ, is_four_digit n ∧ is_perfect_square n ∧ is_palindrome n :=
sorry

end NUMINAMATH_CALUDE_unique_four_digit_palindromic_square_l144_14461


namespace NUMINAMATH_CALUDE_die_faces_count_l144_14413

-- Define the probability of all five dice showing the same number
def probability : ℝ := 0.0007716049382716049

-- Define the number of dice
def num_dice : ℕ := 5

-- Theorem: The number of faces on each die is 10
theorem die_faces_count : 
  ∃ (n : ℕ), n > 0 ∧ (1 : ℝ) / n ^ num_dice = probability ∧ n = 10 :=
sorry

end NUMINAMATH_CALUDE_die_faces_count_l144_14413


namespace NUMINAMATH_CALUDE_equation_represents_hyperbola_l144_14405

/-- The equation (x-y)^2 = 3(x^2 - y^2) represents a hyperbola -/
theorem equation_represents_hyperbola : 
  ∃ (a b c : ℝ), a ≠ 0 ∧ b^2 - 4*a*c > 0 ∧
  ∀ (x y : ℝ), (x - y)^2 = 3*(x^2 - y^2) ↔ a*x^2 + b*x*y + c*y^2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_equation_represents_hyperbola_l144_14405


namespace NUMINAMATH_CALUDE_geometric_sequence_increasing_condition_l144_14499

/-- A geometric sequence with first term a and common ratio r -/
def GeometricSequence (a r : ℝ) : ℕ → ℝ :=
  fun n => a * r^(n - 1)

/-- An increasing sequence -/
def IsIncreasing (f : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, n < m → f n < f m

theorem geometric_sequence_increasing_condition (a r : ℝ) :
  (IsIncreasing (GeometricSequence a r) → 
    GeometricSequence a r 1 < GeometricSequence a r 3 ∧ 
    GeometricSequence a r 3 < GeometricSequence a r 5) ∧
  (∃ a r : ℝ, 
    GeometricSequence a r 1 < GeometricSequence a r 3 ∧ 
    GeometricSequence a r 3 < GeometricSequence a r 5 ∧
    ¬IsIncreasing (GeometricSequence a r)) :=
by sorry


end NUMINAMATH_CALUDE_geometric_sequence_increasing_condition_l144_14499


namespace NUMINAMATH_CALUDE_max_value_of_expression_l144_14454

-- Define the functions f and g
def f : ℝ → ℝ := sorry
def g : ℝ → ℝ := sorry

-- State the theorem
theorem max_value_of_expression (hf : ∀ x, f x ∈ Set.Icc (-3) 5) 
                                 (hg : ∀ x, g x ∈ Set.Icc (-4) 2) :
  ∃ d, d = 45 ∧ ∀ x, 2 * f x * g x + f x ≤ d ∧ 
  ∃ y, 2 * f y * g y + f y = d :=
sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l144_14454


namespace NUMINAMATH_CALUDE_system_inequalities_solution_equation_solution_l144_14493

-- Define the system of inequalities
def system_inequalities (x : ℝ) : Prop :=
  2 * (x - 1) ≥ -4 ∧ (3 * x - 6) / 2 < x - 1

-- Define the set of positive integer solutions
def positive_integer_solutions : Set ℕ :=
  {1, 2, 3}

-- Define the equation
def equation (x : ℝ) : Prop :=
  x ≠ 2 ∧ 3 / (x - 2) = 5 / (2 - x) - 1

-- Theorem for the system of inequalities
theorem system_inequalities_solution :
  ∀ n : ℕ, n ∈ positive_integer_solutions ↔ system_inequalities (n : ℝ) :=
sorry

-- Theorem for the equation
theorem equation_solution :
  ∀ x : ℝ, equation x ↔ x = -6 :=
sorry

end NUMINAMATH_CALUDE_system_inequalities_solution_equation_solution_l144_14493


namespace NUMINAMATH_CALUDE_largest_n_for_inequality_l144_14423

theorem largest_n_for_inequality : ∃ (n : ℕ), n = 2 ∧ 
  (∀ (a b c d : ℝ), 
    (n + 2) * Real.sqrt (a^2 + b^2) + (n + 1) * Real.sqrt (a^2 + c^2) + (n + 1) * Real.sqrt (a^2 + d^2) ≥ n * (a + b + c + d)) ∧
  (∀ (m : ℕ), m > n → 
    ∃ (a b c d : ℝ), 
      (m + 2) * Real.sqrt (a^2 + b^2) + (m + 1) * Real.sqrt (a^2 + c^2) + (m + 1) * Real.sqrt (a^2 + d^2) < m * (a + b + c + d)) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_for_inequality_l144_14423


namespace NUMINAMATH_CALUDE_salem_poem_lines_per_stanza_l144_14412

/-- Represents a poem with stanzas, lines, and words. -/
structure Poem where
  num_stanzas : ℕ
  words_per_line : ℕ
  total_words : ℕ

/-- Calculates the number of lines per stanza in a poem. -/
def lines_per_stanza (p : Poem) : ℕ :=
  (p.total_words / p.words_per_line) / p.num_stanzas

/-- Theorem stating that for a poem with 20 stanzas, 8 words per line, 
    and 1600 total words, each stanza has 10 lines. -/
theorem salem_poem_lines_per_stanza :
  let p : Poem := { num_stanzas := 20, words_per_line := 8, total_words := 1600 }
  lines_per_stanza p = 10 := by
  sorry

end NUMINAMATH_CALUDE_salem_poem_lines_per_stanza_l144_14412


namespace NUMINAMATH_CALUDE_f_seven_eq_neg_seventeen_l144_14426

/-- Given a function f(x) = ax^7 + bx^3 + cx - 5, if f(-7) = 7, then f(7) = -17 -/
theorem f_seven_eq_neg_seventeen 
  (a b c : ℝ) 
  (f : ℝ → ℝ)
  (h1 : ∀ x, f x = a * x^7 + b * x^3 + c * x - 5)
  (h2 : f (-7) = 7) : 
  f 7 = -17 := by
  sorry

end NUMINAMATH_CALUDE_f_seven_eq_neg_seventeen_l144_14426


namespace NUMINAMATH_CALUDE_even_number_of_fours_l144_14415

theorem even_number_of_fours (n₃ n₄ n₅ : ℕ) : 
  n₃ + n₄ + n₅ = 80 →
  3 * n₃ + 4 * n₄ + 5 * n₅ = 276 →
  Even n₄ := by
sorry

end NUMINAMATH_CALUDE_even_number_of_fours_l144_14415


namespace NUMINAMATH_CALUDE_product_sequence_equals_32_l144_14437

theorem product_sequence_equals_32 : 
  (1/4 : ℚ) * 8 * (1/16 : ℚ) * 32 * (1/64 : ℚ) * 128 * (1/256 : ℚ) * 512 * (1/1024 : ℚ) * 2048 = 32 := by
  sorry

end NUMINAMATH_CALUDE_product_sequence_equals_32_l144_14437


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l144_14489

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - x + 2 ≥ 0) ↔ (∃ x : ℝ, x^2 - x + 2 < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l144_14489


namespace NUMINAMATH_CALUDE_tiling_8x2_equals_fib_9_l144_14479

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Number of ways to tile a 2 × n rectangle with 1 × 2 dominoes -/
def tiling_ways (n : ℕ) : ℕ := fib (n + 1)

/-- Theorem: The number of ways to tile an 8 × 2 rectangle with 1 × 2 dominoes
    is equal to the 9th Fibonacci number -/
theorem tiling_8x2_equals_fib_9 :
  tiling_ways 8 = fib 9 := by
  sorry

#eval tiling_ways 8  -- Expected output: 34

end NUMINAMATH_CALUDE_tiling_8x2_equals_fib_9_l144_14479


namespace NUMINAMATH_CALUDE_helen_total_cookies_l144_14456

/-- The number of cookies Helen baked yesterday -/
def cookies_yesterday : ℕ := 435

/-- The number of cookies Helen baked this morning -/
def cookies_this_morning : ℕ := 139

/-- The total number of cookies Helen baked -/
def total_cookies : ℕ := cookies_yesterday + cookies_this_morning

/-- Theorem stating that the total number of cookies Helen baked is 574 -/
theorem helen_total_cookies : total_cookies = 574 := by
  sorry

end NUMINAMATH_CALUDE_helen_total_cookies_l144_14456


namespace NUMINAMATH_CALUDE_basketball_team_sales_l144_14430

/-- The number of cupcakes sold -/
def cupcakes : ℕ := 50

/-- The price of each cupcake in dollars -/
def cupcake_price : ℚ := 2

/-- The price of each cookie in dollars -/
def cookie_price : ℚ := 1/2

/-- The number of basketballs bought -/
def basketballs : ℕ := 2

/-- The price of each basketball in dollars -/
def basketball_price : ℚ := 40

/-- The number of energy drinks bought -/
def energy_drinks : ℕ := 20

/-- The price of each energy drink in dollars -/
def energy_drink_price : ℚ := 2

/-- The number of cookies sold -/
def cookies_sold : ℕ := 40

theorem basketball_team_sales :
  cookies_sold * cookie_price = 
    basketballs * basketball_price + 
    energy_drinks * energy_drink_price - 
    cupcakes * cupcake_price :=
by sorry

end NUMINAMATH_CALUDE_basketball_team_sales_l144_14430


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l144_14421

theorem quadratic_inequality_solution (x : ℝ) : x^2 - 8*x + 12 < 0 ↔ 2 < x ∧ x < 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l144_14421


namespace NUMINAMATH_CALUDE_divisibility_condition_l144_14448

theorem divisibility_condition (x y : ℤ) : 
  (x^3 + y) % (x^2 + y^2) = 0 ∧ (x + y^3) % (x^2 + y^2) = 0 →
  (x = 1 ∧ y = 1) ∨ (x = 1 ∧ y = 0) ∨ (x = 1 ∧ y = -1) ∨
  (x = 0 ∧ y = 1) ∨ (x = 0 ∧ y = -1) ∨
  (x = -1 ∧ y = 1) ∨ (x = -1 ∧ y = 0) ∨ (x = -1 ∧ y = -1) := by
sorry

end NUMINAMATH_CALUDE_divisibility_condition_l144_14448


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l144_14478

/-- Configuration of semicircles and inscribed circle -/
structure SemicircleConfig where
  R : ℝ  -- Radius of larger semicircle
  r : ℝ  -- Radius of smaller semicircle
  x : ℝ  -- Radius of inscribed circle

/-- The inscribed circle touches both semicircles and the diameter -/
def touches_all (c : SemicircleConfig) : Prop :=
  ∃ (O O₁ O₂ : ℝ × ℝ) (P : ℝ × ℝ),
    let (xₒ, yₒ) := O
    let (x₁, y₁) := O₁
    let (x₂, y₂) := O₂
    let (xₚ, yₚ) := P
    (xₒ - x₂)^2 + (yₒ - y₂)^2 = (c.R - c.x)^2 ∧  -- Larger semicircle touches inscribed circle
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = (c.r + c.x)^2 ∧  -- Smaller semicircle touches inscribed circle
    (x₂ - xₚ)^2 + (y₂ - yₚ)^2 = c.x^2           -- Inscribed circle touches diameter

/-- Main theorem: The radius of the inscribed circle is 8 cm -/
theorem inscribed_circle_radius
  (c : SemicircleConfig)
  (h₁ : c.R = 18)
  (h₂ : c.r = 9)
  (h₃ : touches_all c) :
  c.x = 8 :=
sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l144_14478


namespace NUMINAMATH_CALUDE_acceptable_quality_probability_l144_14435

theorem acceptable_quality_probability (p1 p2 : ℝ) 
  (h1 : p1 = 0.01) 
  (h2 : p2 = 0.03) 
  (h3 : 0 ≤ p1 ∧ p1 ≤ 1) 
  (h4 : 0 ≤ p2 ∧ p2 ≤ 1) :
  (1 - p1) * (1 - p2) = 0.960 := by
  sorry

end NUMINAMATH_CALUDE_acceptable_quality_probability_l144_14435


namespace NUMINAMATH_CALUDE_frequency_calculation_l144_14495

theorem frequency_calculation (sample_size : ℕ) (frequency_rate : ℚ) (h1 : sample_size = 1000) (h2 : frequency_rate = 0.4) :
  (sample_size : ℚ) * frequency_rate = 400 := by
  sorry

end NUMINAMATH_CALUDE_frequency_calculation_l144_14495


namespace NUMINAMATH_CALUDE_distance_between_trees_l144_14494

/-- Given a yard of length 150 meters with 11 trees planted at equal distances,
    including one at each end, the distance between two consecutive trees is 15 meters. -/
theorem distance_between_trees (yard_length : ℝ) (num_trees : ℕ) :
  yard_length = 150 →
  num_trees = 11 →
  let num_segments := num_trees - 1
  yard_length / num_segments = 15 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_trees_l144_14494


namespace NUMINAMATH_CALUDE_vodka_mixture_profit_l144_14442

/-- Profit percentage of a mixture of two vodkas -/
def mixture_profit_percentage (profit1 profit2 : ℚ) (increase1 increase2 : ℚ) : ℚ :=
  ((profit1 * increase1 + profit2 * increase2) / 2)

theorem vodka_mixture_profit :
  let initial_profit1 : ℚ := 40 / 100
  let initial_profit2 : ℚ := 20 / 100
  let increase1 : ℚ := 4 / 3
  let increase2 : ℚ := 5 / 3
  mixture_profit_percentage initial_profit1 initial_profit2 increase1 increase2 = 13 / 30 := by
  sorry

#eval (13 / 30 : ℚ)

end NUMINAMATH_CALUDE_vodka_mixture_profit_l144_14442


namespace NUMINAMATH_CALUDE_boy_running_speed_l144_14432

theorem boy_running_speed (side_length : ℝ) (time : ℝ) (speed : ℝ) : 
  side_length = 30 →
  time = 36 →
  speed = (4 * side_length / 1000) / (time / 3600) →
  speed = 12 := by
sorry

end NUMINAMATH_CALUDE_boy_running_speed_l144_14432


namespace NUMINAMATH_CALUDE_water_depth_calculation_l144_14463

def water_depth (dean_height ron_height : ℝ) : ℝ :=
  2 * dean_height

theorem water_depth_calculation (ron_height : ℝ) (h1 : ron_height = 14) :
  water_depth (ron_height - 8) ron_height = 12 := by
  sorry

end NUMINAMATH_CALUDE_water_depth_calculation_l144_14463


namespace NUMINAMATH_CALUDE_max_sum_on_circle_l144_14401

theorem max_sum_on_circle : ∀ x y : ℤ, x^2 + y^2 = 20 → x + y ≤ 6 := by sorry

end NUMINAMATH_CALUDE_max_sum_on_circle_l144_14401


namespace NUMINAMATH_CALUDE_no_arithmetic_mean_l144_14469

theorem no_arithmetic_mean (f1 f2 f3 : ℚ) : 
  f1 = 5/8 ∧ f2 = 9/12 ∧ f3 = 7/10 →
  (f1 ≠ (f2 + f3) / 2) ∧ (f2 ≠ (f1 + f3) / 2) ∧ (f3 ≠ (f1 + f2) / 2) := by
  sorry

#check no_arithmetic_mean

end NUMINAMATH_CALUDE_no_arithmetic_mean_l144_14469


namespace NUMINAMATH_CALUDE_middle_book_pages_l144_14445

def longest_book : ℕ := 396

def shortest_book : ℕ := longest_book / 4

def middle_book : ℕ := 3 * shortest_book

theorem middle_book_pages : middle_book = 297 := by
  sorry

end NUMINAMATH_CALUDE_middle_book_pages_l144_14445


namespace NUMINAMATH_CALUDE_max_product_constrained_l144_14407

theorem max_product_constrained (x y : ℕ+) (h : 7 * x + 5 * y = 140) :
  x * y ≤ 140 := by
  sorry

end NUMINAMATH_CALUDE_max_product_constrained_l144_14407


namespace NUMINAMATH_CALUDE_correct_factorization_l144_14452

theorem correct_factorization (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_correct_factorization_l144_14452


namespace NUMINAMATH_CALUDE_x_intercept_of_line_l144_14459

/-- The x-intercept of the line 4x + 7y = 28 is (7, 0) -/
theorem x_intercept_of_line (x y : ℝ) :
  4 * x + 7 * y = 28 ∧ y = 0 → x = 7 := by
  sorry

end NUMINAMATH_CALUDE_x_intercept_of_line_l144_14459


namespace NUMINAMATH_CALUDE_lcm_of_4_6_10_18_l144_14457

theorem lcm_of_4_6_10_18 : Nat.lcm 4 (Nat.lcm 6 (Nat.lcm 10 18)) = 180 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_4_6_10_18_l144_14457


namespace NUMINAMATH_CALUDE_probability_even_sum_l144_14462

def set_a : Finset ℕ := {11, 44, 55}
def set_b : Finset ℕ := {1}

def is_sum_even (x : ℕ) (y : ℕ) : Bool :=
  (x + y) % 2 = 0

def count_even_sums : ℕ :=
  (set_a.filter (λ x => is_sum_even x 1)).card

theorem probability_even_sum :
  (count_even_sums : ℚ) / (set_a.card : ℚ) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_probability_even_sum_l144_14462


namespace NUMINAMATH_CALUDE_indicator_light_signals_l144_14485

/-- The number of indicator lights in a row -/
def num_lights : ℕ := 8

/-- The number of lights displayed at a time -/
def lights_displayed : ℕ := 4

/-- The number of adjacent lights among those displayed -/
def adjacent_lights : ℕ := 3

/-- The number of colors each light can display -/
def colors_per_light : ℕ := 2

/-- The total number of different signals that can be displayed -/
def total_signals : ℕ := 320

theorem indicator_light_signals :
  (num_lights = 8) →
  (lights_displayed = 4) →
  (adjacent_lights = 3) →
  (colors_per_light = 2) →
  total_signals = 320 := by sorry

end NUMINAMATH_CALUDE_indicator_light_signals_l144_14485


namespace NUMINAMATH_CALUDE_target_probability_value_l144_14419

/-- The probability of hitting the target on a single shot -/
def hit_probability : ℝ := 0.85

/-- The probability of missing the target on a single shot -/
def miss_probability : ℝ := 1 - hit_probability

/-- The probability of missing the first two shots and hitting the third shot -/
def target_probability : ℝ := miss_probability * miss_probability * hit_probability

theorem target_probability_value : target_probability = 0.019125 := by
  sorry

end NUMINAMATH_CALUDE_target_probability_value_l144_14419


namespace NUMINAMATH_CALUDE_common_ratio_is_two_l144_14441

def geometric_sequence (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ := a₁ * q ^ (n - 1)

theorem common_ratio_is_two 
  (a₁ : ℝ) 
  (q : ℝ) 
  (h_positive : ∀ n : ℕ, geometric_sequence a₁ q n > 0)
  (h_product : geometric_sequence a₁ q 1 * geometric_sequence a₁ q 5 = 16)
  (h_first_term : a₁ = 2) :
  q = 2 := by
sorry

end NUMINAMATH_CALUDE_common_ratio_is_two_l144_14441


namespace NUMINAMATH_CALUDE_arrangement_count_is_150_l144_14458

/-- The number of ways to arrange volunteers among events --/
def arrange_volunteers (n : ℕ) (k : ℕ) : ℕ :=
  k^n - k * (k-1)^n + (k * (k-1) / 2) * (k-2)^n

/-- The number of arrangements for 5 volunteers and 3 events --/
def arrangement_count : ℕ := arrange_volunteers 5 3

/-- Theorem: The number of arrangements for 5 volunteers and 3 events,
    such that each event has at least one participant, is 150 --/
theorem arrangement_count_is_150 : arrangement_count = 150 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_count_is_150_l144_14458


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l144_14446

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := (x + 1)^2 + (y - 2)^2 = 9

-- State the theorem
theorem circle_center_and_radius :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (-1, 2) ∧
    radius = 3 ∧
    ∀ (x y : ℝ), circle_equation x y ↔ ((x - center.1)^2 + (y - center.2)^2 = radius^2) :=
sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_l144_14446


namespace NUMINAMATH_CALUDE_harry_fish_count_harry_fish_count_proof_l144_14453

/-- Given three friends with fish, prove Harry has 224 fish -/
theorem harry_fish_count : ℕ → ℕ → ℕ → Prop :=
  fun sam_fish joe_fish harry_fish =>
    sam_fish = 7 ∧
    joe_fish = 8 * sam_fish ∧
    harry_fish = 4 * joe_fish →
    harry_fish = 224

/-- Proof of the theorem -/
theorem harry_fish_count_proof : ∃ (sam_fish joe_fish harry_fish : ℕ),
  harry_fish_count sam_fish joe_fish harry_fish :=
by
  sorry

end NUMINAMATH_CALUDE_harry_fish_count_harry_fish_count_proof_l144_14453


namespace NUMINAMATH_CALUDE_parallel_lines_to_plane_not_always_parallel_l144_14449

structure Plane where

structure Line where

def parallel_line_plane (l : Line) (p : Plane) : Prop := sorry

def parallel_lines (l1 l2 : Line) : Prop := sorry

theorem parallel_lines_to_plane_not_always_parallel (m n : Line) (α : Plane) : 
  m ≠ n → 
  ¬(parallel_line_plane m α → parallel_line_plane n α → parallel_lines m n) := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_to_plane_not_always_parallel_l144_14449


namespace NUMINAMATH_CALUDE_min_cost_for_89_coins_l144_14410

/-- Represents the cost structure for the coin problem -/
structure CoinProblem where
  total_coins : Nat
  coin_cost : Nat
  yes_fee : Nat
  no_fee : Nat

/-- Calculates the minimum cost to guarantee obtaining the lucky coin -/
def min_cost_to_get_lucky_coin (problem : CoinProblem) : Nat :=
  sorry

/-- Theorem stating the minimum cost for the specific problem instance -/
theorem min_cost_for_89_coins :
  let problem : CoinProblem := {
    total_coins := 89,
    coin_cost := 30,
    yes_fee := 20,
    no_fee := 10
  }
  min_cost_to_get_lucky_coin problem = 130 := by
  sorry

end NUMINAMATH_CALUDE_min_cost_for_89_coins_l144_14410


namespace NUMINAMATH_CALUDE_part_one_part_two_l144_14491

-- Define the propositions p and q
def p (a x : ℝ) : Prop := a < x ∧ x < 3 * a

def q (x : ℝ) : Prop := 2 < x ∧ x < 3

-- Part 1
theorem part_one (a x : ℝ) (h1 : a > 0) (h2 : a = 1) (h3 : p a x ∧ q x) :
  2 < x ∧ x < 3 := by sorry

-- Part 2
theorem part_two (a : ℝ) (h1 : a > 0) 
  (h2 : ∀ x, q x → p a x) 
  (h3 : ∃ x, p a x ∧ ¬q x) :
  1 ≤ a ∧ a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l144_14491


namespace NUMINAMATH_CALUDE_diagonals_not_parallel_32gon_l144_14400

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- The number of diagonals parallel to sides in a regular polygon with n sides -/
def num_parallel_diagonals (n : ℕ) : ℕ := (n / 2) * ((n - 4) / 2)

/-- The number of diagonals not parallel to any side in a regular 32-sided polygon -/
theorem diagonals_not_parallel_32gon : 
  num_diagonals 32 - num_parallel_diagonals 32 = 240 := by
  sorry


end NUMINAMATH_CALUDE_diagonals_not_parallel_32gon_l144_14400


namespace NUMINAMATH_CALUDE_pizza_toppings_l144_14418

theorem pizza_toppings (total_slices : ℕ) (pepperoni_slices : ℕ) (mushroom_slices : ℕ)
  (h_total : total_slices = 16)
  (h_pepperoni : pepperoni_slices = 8)
  (h_mushroom : mushroom_slices = 14)
  (h_at_least_one : ∀ slice, slice ≤ total_slices → (slice ≤ pepperoni_slices ∨ slice ≤ mushroom_slices)) :
  ∃ both_toppings : ℕ, 
    both_toppings = pepperoni_slices + mushroom_slices - total_slices ∧
    both_toppings = 6 :=
by sorry

end NUMINAMATH_CALUDE_pizza_toppings_l144_14418


namespace NUMINAMATH_CALUDE_T_is_three_rays_l144_14464

-- Define the set T
def T : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p
               (4 = x + 3 ∧ y - 2 ≤ 4) ∨
               (4 = y - 2 ∧ x + 3 ≤ 4) ∨
               (x + 3 = y - 2 ∧ 4 ≤ x + 3)}

-- Define the three rays with common endpoint (1,6)
def ray1 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = 1 ∧ p.2 ≤ 6}
def ray2 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 ≤ 1 ∧ p.2 = 6}
def ray3 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 ≥ 1 ∧ p.2 = p.1 + 5}

-- Theorem statement
theorem T_is_three_rays : T = ray1 ∪ ray2 ∪ ray3 :=
  sorry

end NUMINAMATH_CALUDE_T_is_three_rays_l144_14464


namespace NUMINAMATH_CALUDE_first_floor_bedrooms_l144_14403

theorem first_floor_bedrooms 
  (total : ℕ) 
  (second_floor : ℕ) 
  (third_floor : ℕ) 
  (fourth_floor : ℕ) 
  (h1 : total = 22) 
  (h2 : second_floor = 6) 
  (h3 : third_floor = 4) 
  (h4 : fourth_floor = 3) : 
  total - (second_floor + third_floor + fourth_floor) = 9 := by
  sorry

end NUMINAMATH_CALUDE_first_floor_bedrooms_l144_14403


namespace NUMINAMATH_CALUDE_prob_two_defective_consignment_l144_14476

/-- Represents a consignment of picture tubes -/
structure Consignment where
  total : ℕ
  defective : ℕ
  h_defective_le_total : defective ≤ total

/-- Calculates the probability of selecting two defective tubes without replacement -/
def prob_two_defective (c : Consignment) : ℚ :=
  (c.defective : ℚ) / (c.total : ℚ) * ((c.defective - 1) : ℚ) / ((c.total - 1) : ℚ)

theorem prob_two_defective_consignment :
  let c : Consignment := ⟨20, 5, by norm_num⟩
  prob_two_defective c = 1 / 19 := by sorry

end NUMINAMATH_CALUDE_prob_two_defective_consignment_l144_14476


namespace NUMINAMATH_CALUDE_twins_age_problem_l144_14420

theorem twins_age_problem (age : ℕ) : 
  (age + 1) * (age + 1) = age * age + 15 → age = 7 := by
  sorry

end NUMINAMATH_CALUDE_twins_age_problem_l144_14420


namespace NUMINAMATH_CALUDE_fiftieth_term_is_296_l144_14402

/-- The nth term of an arithmetic sequence -/
def arithmeticSequenceTerm (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

/-- The 50th term of the specific arithmetic sequence -/
def fiftiethTerm : ℝ :=
  arithmeticSequenceTerm 2 6 50

theorem fiftieth_term_is_296 : fiftiethTerm = 296 := by
  sorry

end NUMINAMATH_CALUDE_fiftieth_term_is_296_l144_14402


namespace NUMINAMATH_CALUDE_median_in_group_two_l144_14484

/-- Represents the labor time groups --/
inductive LaborGroup
  | One
  | Two
  | Three
  | Four

/-- The frequency of each labor group --/
def frequency : LaborGroup → Nat
  | LaborGroup.One => 10
  | LaborGroup.Two => 20
  | LaborGroup.Three => 12
  | LaborGroup.Four => 8

/-- The total number of surveyed students --/
def totalStudents : Nat := 50

/-- The cumulative frequency up to and including a given group --/
def cumulativeFrequency (g : LaborGroup) : Nat :=
  match g with
  | LaborGroup.One => frequency LaborGroup.One
  | LaborGroup.Two => frequency LaborGroup.One + frequency LaborGroup.Two
  | LaborGroup.Three => frequency LaborGroup.One + frequency LaborGroup.Two + frequency LaborGroup.Three
  | LaborGroup.Four => totalStudents

/-- The median position --/
def medianPosition : Nat := totalStudents / 2

theorem median_in_group_two :
  cumulativeFrequency LaborGroup.One < medianPosition ∧
  medianPosition ≤ cumulativeFrequency LaborGroup.Two :=
sorry

end NUMINAMATH_CALUDE_median_in_group_two_l144_14484


namespace NUMINAMATH_CALUDE_five_card_selection_with_constraints_l144_14472

/-- The number of cards in a standard deck -/
def standard_deck_size : ℕ := 52

/-- The number of suits in a standard deck -/
def number_of_suits : ℕ := 4

/-- The number of cards in each suit -/
def cards_per_suit : ℕ := 13

/-- The number of cards to be chosen -/
def cards_to_choose : ℕ := 5

/-- The number of cards that must share a suit -/
def cards_sharing_suit : ℕ := 2

/-- 
  The number of ways to choose 5 cards from a standard deck of 52 cards, 
  where exactly two cards share a suit and the remaining three are of different suits.
-/
theorem five_card_selection_with_constraints : 
  (number_of_suits) * 
  (Nat.choose cards_per_suit cards_sharing_suit) * 
  (Nat.choose (number_of_suits - 1) (cards_to_choose - cards_sharing_suit)) * 
  (cards_per_suit ^ (cards_to_choose - cards_sharing_suit)) = 684684 := by
  sorry

end NUMINAMATH_CALUDE_five_card_selection_with_constraints_l144_14472


namespace NUMINAMATH_CALUDE_quadratic_inequality_theorem_l144_14496

-- Define the quadratic function
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_inequality_theorem (a b c : ℝ) :
  (∀ x : ℝ, a * x^2 + b * x + c > 0 ↔ x < -2 ∨ x > 4) →
  f a b c 2 < f a b c (-1) ∧ f a b c (-1) < f a b c 5 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_theorem_l144_14496


namespace NUMINAMATH_CALUDE_sum_equals_negative_six_l144_14473

theorem sum_equals_negative_six (a b c d : ℤ) 
  (h : a + 2 = b + 3 ∧ b + 3 = c + 4 ∧ c + 4 = d + 5 ∧ d + 5 = a + b + c + d + 8) : 
  a + b + c + d = -6 := by
sorry

end NUMINAMATH_CALUDE_sum_equals_negative_six_l144_14473


namespace NUMINAMATH_CALUDE_upper_limit_of_b_l144_14466

theorem upper_limit_of_b (a b : ℤ) (h1 : 6 < a) (h2 : a < 17) (h3 : 3 < b) 
  (h4 : (a : ℚ) / b ≤ 3.75) (h5 : 3.75 ≤ (a : ℚ) / b) : b ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_upper_limit_of_b_l144_14466


namespace NUMINAMATH_CALUDE_probability_different_digits_l144_14409

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def has_different_digits (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.length = 3 ∧ digits.toFinset.card = 3

def count_valid_numbers : ℕ :=
  999 - 100 + 1

def count_different_digit_numbers : ℕ :=
  9 * 9 * 8

theorem probability_different_digits :
  (count_different_digit_numbers : ℚ) / count_valid_numbers = 18 / 25 := by
  sorry

end NUMINAMATH_CALUDE_probability_different_digits_l144_14409


namespace NUMINAMATH_CALUDE_max_value_of_expression_l144_14455

/-- Given that a, b, and c are distinct elements from the set {1, 2, 4},
    the maximum value of (a / 2) / (b / c) is 8. -/
theorem max_value_of_expression (a b c : ℕ) : 
  a ∈ ({1, 2, 4} : Set ℕ) →
  b ∈ ({1, 2, 4} : Set ℕ) →
  c ∈ ({1, 2, 4} : Set ℕ) →
  a ≠ b → b ≠ c → a ≠ c →
  (a / 2 : ℚ) / (b / c : ℚ) ≤ 8 ∧ 
  ∃ (x y z : ℕ), x ∈ ({1, 2, 4} : Set ℕ) ∧ 
                 y ∈ ({1, 2, 4} : Set ℕ) ∧ 
                 z ∈ ({1, 2, 4} : Set ℕ) ∧ 
                 x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
                 (x / 2 : ℚ) / (y / z : ℚ) = 8 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l144_14455


namespace NUMINAMATH_CALUDE_pet_store_birds_l144_14490

/-- The number of birds in a cage -/
def birds_in_cage (parrots parakeets finches cockatiels canaries lovebirds toucans : ℕ) : ℕ :=
  parrots + parakeets + finches + cockatiels + canaries + lovebirds + toucans

/-- The total number of birds in the pet store -/
def total_birds : ℕ :=
  birds_in_cage 6 2 0 0 0 0 0 +  -- Cage 1
  birds_in_cage 4 3 5 0 0 0 0 +  -- Cage 2
  birds_in_cage 2 4 0 1 0 0 0 +  -- Cage 3
  birds_in_cage 3 5 0 0 2 0 0 +  -- Cage 4
  birds_in_cage 7 0 0 0 0 4 0 +  -- Cage 5
  birds_in_cage 4 2 3 0 0 0 1    -- Cage 6

theorem pet_store_birds : total_birds = 58 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_birds_l144_14490


namespace NUMINAMATH_CALUDE_max_sum_on_circle_l144_14433

theorem max_sum_on_circle (x y : ℤ) (h : x^2 + y^2 = 100) : 
  x + y ≤ 14 ∧ ∃ (a b : ℤ), a^2 + b^2 = 100 ∧ a + b = 14 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_on_circle_l144_14433


namespace NUMINAMATH_CALUDE_min_value_problem_l144_14440

theorem min_value_problem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 2/b = 1) :
  2/(a-1) + 1/(b-2) ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_problem_l144_14440


namespace NUMINAMATH_CALUDE_not_pythagorean_triple_l144_14427

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

theorem not_pythagorean_triple : ¬ is_pythagorean_triple 7 25 26 := by
  sorry

end NUMINAMATH_CALUDE_not_pythagorean_triple_l144_14427


namespace NUMINAMATH_CALUDE_odd_function_sum_l144_14480

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def periodic_4 (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 4) = f x

theorem odd_function_sum (f : ℝ → ℝ) 
  (h_odd : is_odd_function f)
  (h_periodic : ∀ x, f x + f (4 - x) = 0)
  (h_f_1 : f 1 = 8) :
  f 2010 + f 2011 + f 2012 = -8 := by
sorry

end NUMINAMATH_CALUDE_odd_function_sum_l144_14480


namespace NUMINAMATH_CALUDE_partition_condition_l144_14482

/-- A partition of ℕ* into n sets satisfying the given conditions -/
structure Partition (a : ℝ) where
  n : ℕ+
  sets : Fin n → Set ℕ+
  disjoint : ∀ i j, i ≠ j → Disjoint (sets i) (sets j)
  cover : (⋃ i, sets i) = Set.univ
  infinite : ∀ i, Set.Infinite (sets i)
  difference : ∀ i x y, x ∈ sets i → y ∈ sets i → x > y → x - y ≥ a ^ (i : ℕ)

/-- The main theorem -/
theorem partition_condition (a : ℝ) : 
  (∃ p : Partition a, True) → a < 2 := by
  sorry

end NUMINAMATH_CALUDE_partition_condition_l144_14482


namespace NUMINAMATH_CALUDE_no_integer_solutions_for_equation_l144_14488

theorem no_integer_solutions_for_equation :
  ¬ ∃ (x y : ℤ), 15 * x^2 - 7 * y^2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_for_equation_l144_14488


namespace NUMINAMATH_CALUDE_angle_range_l144_14470

theorem angle_range (α : Real) 
  (h1 : α > 0 ∧ α < 2 * Real.pi) 
  (h2 : Real.sin α > 0) 
  (h3 : Real.cos α < 0) : 
  α > Real.pi / 2 ∧ α < Real.pi :=
by sorry

end NUMINAMATH_CALUDE_angle_range_l144_14470


namespace NUMINAMATH_CALUDE_rectangular_box_volume_l144_14436

theorem rectangular_box_volume (l w h : ℝ) 
  (area1 : l * w = 30)
  (area2 : w * h = 20)
  (area3 : l * h = 12) :
  l * w * h = 60 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_box_volume_l144_14436


namespace NUMINAMATH_CALUDE_value_of_expression_l144_14474

theorem value_of_expression (x : ℝ) (h : x = 5) : 3 * x + 4 = 19 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l144_14474


namespace NUMINAMATH_CALUDE_matrix_equation_solution_l144_14451

-- Define the matrix evaluation rule
def matrix_value (a b c d : ℝ) : ℝ := a * b - c * d

-- Define our specific matrix as a function of x
def M (x : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![3*x+1, 2; 2*x, x+1]

-- State the theorem
theorem matrix_equation_solution :
  ∀ x : ℝ, matrix_value (M x 0 0) (M x 1 1) (M x 0 1) (M x 1 0) = 5 ↔ 
  x = 2 * Real.sqrt 3 / 3 ∨ x = -2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_matrix_equation_solution_l144_14451


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_min_value_is_two_l144_14468

theorem min_reciprocal_sum (a b : ℝ) (h1 : b > 0) (h2 : a + b = 2) :
  ∀ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 2 → 1/a + 1/b ≤ 1/x + 1/y :=
by sorry

theorem min_value_is_two (a b : ℝ) (h1 : b > 0) (h2 : a + b = 2) :
  1/a + 1/b = 2 :=
by sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_min_value_is_two_l144_14468


namespace NUMINAMATH_CALUDE_star_vertex_angle_formula_l144_14481

/-- The angle measure at the vertices of a star formed by extending the sides of a regular n-sided polygon -/
def starVertexAngle (n : ℕ) : ℚ :=
  (n - 4) * 180 / n

/-- Theorem stating the angle measure at the vertices of a star formed by extending the sides of a regular n-sided polygon -/
theorem star_vertex_angle_formula (n : ℕ) (h : n > 2) :
  starVertexAngle n = (n - 4) * 180 / n :=
by sorry

end NUMINAMATH_CALUDE_star_vertex_angle_formula_l144_14481


namespace NUMINAMATH_CALUDE_a_2n_is_perfect_square_l144_14425

/-- Definition of a_n: number of natural numbers with digit sum n and digits in {1,3,4} -/
def a (n : ℕ) : ℕ := sorry

/-- Theorem: a_{2n} is a perfect square for all natural numbers n -/
theorem a_2n_is_perfect_square (n : ℕ) : ∃ k : ℕ, a (2 * n) = k ^ 2 := by sorry

end NUMINAMATH_CALUDE_a_2n_is_perfect_square_l144_14425


namespace NUMINAMATH_CALUDE_dandelion_ratio_l144_14411

theorem dandelion_ratio : 
  ∀ (billy_initial george_initial billy_additional george_additional : ℕ) 
    (average : ℚ),
  billy_initial = 36 →
  billy_additional = 10 →
  george_additional = 10 →
  average = 34 →
  (billy_initial + george_initial + billy_additional + george_additional : ℚ) / 2 = average →
  george_initial * 3 = billy_initial :=
by sorry

end NUMINAMATH_CALUDE_dandelion_ratio_l144_14411


namespace NUMINAMATH_CALUDE_exists_number_with_specific_digit_sum_l144_14465

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Theorem stating the existence of a number with specific digit sum properties -/
theorem exists_number_with_specific_digit_sum : 
  ∃ m : ℕ, sumOfDigits m = 1990 ∧ sumOfDigits (m^2) = 1990^2 := by sorry

end NUMINAMATH_CALUDE_exists_number_with_specific_digit_sum_l144_14465


namespace NUMINAMATH_CALUDE_negation_equivalence_l144_14424

-- Define the original proposition
def original_proposition (a : ℝ) : Prop :=
  ∀ x : ℝ, x > 0 → x^2 + a*x + 3 ≥ 0

-- Define the negation of the proposition
def negation_proposition (a : ℝ) : Prop :=
  ∃ x : ℝ, x > 0 ∧ x^2 + a*x + 3 < 0

-- Theorem stating that the negation is correct
theorem negation_equivalence (a : ℝ) :
  ¬(original_proposition a) ↔ negation_proposition a :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l144_14424


namespace NUMINAMATH_CALUDE_min_value_of_f_l144_14497

noncomputable def f (b : ℝ) (x : ℝ) : ℝ := (x^2 - 2*x + 1/b) / (2*x^2 + 2*x + 1)

theorem min_value_of_f (b : ℝ) (h : b > 0) :
  ∃ c : ℝ, c = -4 ∧ ∀ x : ℝ, f b x ≥ c :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l144_14497


namespace NUMINAMATH_CALUDE_height_difference_approx_10_inches_l144_14404

-- Define constants
def mark_height_cm : ℝ := 160
def mike_height_cm : ℝ := 185
def cm_to_m : ℝ := 0.01
def m_to_ft : ℝ := 3.28084
def ft_to_in : ℝ := 12

-- Define the height difference function
def height_difference_inches (h1 h2 : ℝ) : ℝ :=
  (h2 - h1) * cm_to_m * m_to_ft * ft_to_in

-- Theorem statement
theorem height_difference_approx_10_inches :
  ∃ ε > 0, abs (height_difference_inches mark_height_cm mike_height_cm - 10) < ε :=
sorry

end NUMINAMATH_CALUDE_height_difference_approx_10_inches_l144_14404


namespace NUMINAMATH_CALUDE_factors_180_multiples_15_l144_14428

/-- A function that returns the number of positive integers that are both factors of n and multiples of m -/
def count_common_factors_multiples (n m : ℕ) : ℕ :=
  (Finset.filter (λ x => n % x = 0 ∧ x % m = 0) (Finset.range n)).card

/-- Theorem stating that the number of positive integers that are both factors of 180 and multiples of 15 is 6 -/
theorem factors_180_multiples_15 : count_common_factors_multiples 180 15 = 6 := by
  sorry

end NUMINAMATH_CALUDE_factors_180_multiples_15_l144_14428


namespace NUMINAMATH_CALUDE_abs_neg_one_eq_one_l144_14498

theorem abs_neg_one_eq_one : |(-1 : ℚ)| = 1 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_one_eq_one_l144_14498


namespace NUMINAMATH_CALUDE_function_inequality_condition_l144_14471

theorem function_inequality_condition (f : ℝ → ℝ) (a b : ℝ) :
  (∀ x, f x = x^2 - 4*x + 3) →
  a > 0 →
  b > 0 →
  (∀ x, |x + 2| < b → |f x + 1| < a) ↔
  b ≤ Real.sqrt a := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_condition_l144_14471
