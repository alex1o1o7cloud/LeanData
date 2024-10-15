import Mathlib

namespace NUMINAMATH_CALUDE_division_simplification_l876_87686

theorem division_simplification (a : ℝ) (h : a ≠ 0) :
  (a - 1/a) / ((a - 1)/a) = a + 1 := by
sorry

end NUMINAMATH_CALUDE_division_simplification_l876_87686


namespace NUMINAMATH_CALUDE_banana_orange_equivalence_l876_87605

/-- Given that 3/4 of 12 bananas are worth as much as 9 oranges,
    prove that 3/5 of 15 bananas are worth as much as 9 oranges. -/
theorem banana_orange_equivalence (banana_value : ℚ) :
  (3/4 : ℚ) * 12 * banana_value = 9 →
  (3/5 : ℚ) * 15 * banana_value = 9 :=
by sorry

end NUMINAMATH_CALUDE_banana_orange_equivalence_l876_87605


namespace NUMINAMATH_CALUDE_power_division_l876_87637

theorem power_division (a b c d : ℕ) (h : b = a^2) :
  a^(2*c+1) / b^c = a :=
sorry

end NUMINAMATH_CALUDE_power_division_l876_87637


namespace NUMINAMATH_CALUDE_fraction_doubling_l876_87638

theorem fraction_doubling (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (1.4 * x) / (0.7 * y) = 2 * (x / y) :=
by sorry

end NUMINAMATH_CALUDE_fraction_doubling_l876_87638


namespace NUMINAMATH_CALUDE_product_as_sum_of_squares_l876_87669

theorem product_as_sum_of_squares : 
  85 * 135 = 85^2 + 50^2 + 35^2 + 15^2 + 15^2 + 5^2 + 5^2 + 5^ 2 := by
  sorry

end NUMINAMATH_CALUDE_product_as_sum_of_squares_l876_87669


namespace NUMINAMATH_CALUDE_business_value_l876_87657

/-- Given a man who owns 2/3 of a business and sells 3/4 of his shares for 75,000 Rs,
    the total value of the business is 150,000 Rs. -/
theorem business_value (owned_fraction : ℚ) (sold_fraction : ℚ) (sale_price : ℕ) :
  owned_fraction = 2/3 →
  sold_fraction = 3/4 →
  sale_price = 75000 →
  (owned_fraction * sold_fraction * (sale_price : ℚ) / (owned_fraction * sold_fraction)) = 150000 :=
by sorry

end NUMINAMATH_CALUDE_business_value_l876_87657


namespace NUMINAMATH_CALUDE_inequality_holds_for_all_reals_l876_87600

theorem inequality_holds_for_all_reals (a b : ℝ) (h : |a - b| > 2) :
  ∀ x : ℝ, |x - a| + |x - b| > 2 := by
sorry

end NUMINAMATH_CALUDE_inequality_holds_for_all_reals_l876_87600


namespace NUMINAMATH_CALUDE_valentino_farm_birds_l876_87650

/-- The number of birds on Mr. Valentino's farm -/
def total_birds (chickens : ℕ) : ℕ :=
  let ducks := 2 * chickens
  let turkeys := 3 * ducks
  chickens + ducks + turkeys

/-- Theorem stating the total number of birds on Mr. Valentino's farm -/
theorem valentino_farm_birds :
  total_birds 200 = 1800 := by
  sorry

end NUMINAMATH_CALUDE_valentino_farm_birds_l876_87650


namespace NUMINAMATH_CALUDE_polynomial_transformation_c_values_l876_87695

/-- The number of distinct possible values of c in a polynomial transformation. -/
theorem polynomial_transformation_c_values
  (a b r s t : ℂ)
  (h_distinct : r ≠ s ∧ s ≠ t ∧ r ≠ t)
  (h_transform : ∀ z, (z - r) * (z - s) * (z - t) =
                      ((a * z + b) - c * r) * ((a * z + b) - c * s) * ((a * z + b) - c * t)) :
  ∃! (values : Finset ℂ), values.card = 4 ∧ ∀ c, c ∈ values ↔ 
    ∃ z, (z - r) * (z - s) * (z - t) =
         ((a * z + b) - c * r) * ((a * z + b) - c * s) * ((a * z + b) - c * t) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_transformation_c_values_l876_87695


namespace NUMINAMATH_CALUDE_sum_difference_theorem_l876_87683

def jo_sum (n : ℕ) : ℕ := n * (n + 1) / 2

def round_to_nearest_ten (x : ℕ) : ℕ :=
  let r := x % 10
  if r < 5 then x - r else x + (10 - r)

def kate_sum (n : ℕ) : ℕ :=
  List.range n |> List.map (λ x => round_to_nearest_ten (x + 1)) |> List.sum

theorem sum_difference_theorem :
  jo_sum 60 - kate_sum 60 = 1530 := by sorry

end NUMINAMATH_CALUDE_sum_difference_theorem_l876_87683


namespace NUMINAMATH_CALUDE_expression_equality_l876_87647

theorem expression_equality (x : ℝ) :
  3 * (x + 2)^2 + 2 * (x + 2) * (5 - x) + (5 - x)^2 = 
  ((Real.sqrt 3 - 1) * x + 5 + 2 * Real.sqrt 3)^2 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l876_87647


namespace NUMINAMATH_CALUDE_simplest_form_iff_odd_l876_87672

theorem simplest_form_iff_odd (n : ℤ) : 
  (∀ d : ℤ, d ∣ (3*n + 10) ∧ d ∣ (5*n + 16) → d = 1 ∨ d = -1) ↔ Odd n :=
sorry

end NUMINAMATH_CALUDE_simplest_form_iff_odd_l876_87672


namespace NUMINAMATH_CALUDE_special_circle_standard_equation_l876_87617

/-- A circle passing through two points with its center on a given line -/
structure SpecialCircle where
  -- Center coordinates
  h : ℝ
  k : ℝ
  -- Radius
  r : ℝ
  -- The circle passes through (0,4)
  passes_through_A : h^2 + (k - 4)^2 = r^2
  -- The circle passes through (4,6)
  passes_through_B : (h - 4)^2 + (k - 6)^2 = r^2
  -- The center lies on the line x-2y-2=0
  center_on_line : h - 2*k - 2 = 0

/-- The standard equation of the special circle -/
def special_circle_equation (c : SpecialCircle) : Prop :=
  ∀ (x y : ℝ), (x - 4)^2 + (y - 1)^2 = 25 ↔ (x - c.h)^2 + (y - c.k)^2 = c.r^2

/-- The main theorem: proving the standard equation of the special circle -/
theorem special_circle_standard_equation :
  ∃ (c : SpecialCircle), special_circle_equation c :=
sorry

end NUMINAMATH_CALUDE_special_circle_standard_equation_l876_87617


namespace NUMINAMATH_CALUDE_square_of_binomial_coefficient_l876_87623

theorem square_of_binomial_coefficient (a : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, 9*x^2 - 18*x + a = (3*x + b)^2) → a = 9 := by
  sorry

end NUMINAMATH_CALUDE_square_of_binomial_coefficient_l876_87623


namespace NUMINAMATH_CALUDE_max_eel_coverage_l876_87619

/-- An eel is a polyomino formed by a path of unit squares which makes two turns in opposite directions -/
def Eel : Type := Unit

/-- A configuration of non-overlapping eels on a grid -/
def EelConfiguration (n : ℕ) : Type := Unit

/-- The area covered by a configuration of eels -/
def coveredArea (n : ℕ) (config : EelConfiguration n) : ℕ := sorry

theorem max_eel_coverage :
  ∃ (config : EelConfiguration 1000),
    coveredArea 1000 config = 999998 ∧
    ∀ (other_config : EelConfiguration 1000),
      coveredArea 1000 other_config ≤ 999998 := by sorry

end NUMINAMATH_CALUDE_max_eel_coverage_l876_87619


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l876_87687

def a : ℝ × ℝ := (1, 2)
def b (m : ℝ) : ℝ × ℝ := (m, m + 1)

theorem parallel_vectors_m_value :
  (∃ (k : ℝ), k ≠ 0 ∧ b m = k • a) → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l876_87687


namespace NUMINAMATH_CALUDE_students_left_on_bus_l876_87648

theorem students_left_on_bus (initial_students : ℕ) (students_off : ℕ) : 
  initial_students = 10 → students_off = 3 → initial_students - students_off = 7 := by
  sorry

end NUMINAMATH_CALUDE_students_left_on_bus_l876_87648


namespace NUMINAMATH_CALUDE_odd_decreasing_function_properties_l876_87621

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
variable (h_odd : ∀ x, f (-x) = -f x)
variable (h_decreasing : ∀ x y, x < y → f x > f y)

-- Define the theorem
theorem odd_decreasing_function_properties
  (a b : ℝ)
  (h_sum_neg : a + b < 0) :
  f (a + b) > 0 ∧ f a + f b > 0 := by
sorry


end NUMINAMATH_CALUDE_odd_decreasing_function_properties_l876_87621


namespace NUMINAMATH_CALUDE_solve_batting_problem_l876_87665

def batting_problem (pitches_per_token : ℕ) (macy_tokens : ℕ) (piper_tokens : ℕ) 
  (macy_hits : ℕ) (total_misses : ℕ) : Prop :=
  let total_pitches := pitches_per_token * (macy_tokens + piper_tokens)
  let total_hits := total_pitches - total_misses
  let piper_hits := total_hits - macy_hits
  piper_hits = 55

theorem solve_batting_problem :
  batting_problem 15 11 17 50 315 := by
  sorry

end NUMINAMATH_CALUDE_solve_batting_problem_l876_87665


namespace NUMINAMATH_CALUDE_sqrt_difference_equals_one_l876_87696

theorem sqrt_difference_equals_one : 
  Real.sqrt 9 - Real.sqrt ((-2)^2) = 1 := by sorry

end NUMINAMATH_CALUDE_sqrt_difference_equals_one_l876_87696


namespace NUMINAMATH_CALUDE_chocolate_distribution_l876_87645

theorem chocolate_distribution (minho : ℕ) (taemin kibum : ℕ) : 
  taemin = 5 * minho →
  kibum = 3 * minho →
  taemin + kibum = 160 →
  minho = 20 := by
sorry

end NUMINAMATH_CALUDE_chocolate_distribution_l876_87645


namespace NUMINAMATH_CALUDE_sum_of_powers_equals_six_to_power_three_l876_87677

theorem sum_of_powers_equals_six_to_power_three :
  3^3 + 4^3 + 5^3 = 6^3 := by sorry

end NUMINAMATH_CALUDE_sum_of_powers_equals_six_to_power_three_l876_87677


namespace NUMINAMATH_CALUDE_smallest_n_value_l876_87660

/-- Represents the dimensions of a rectangular block -/
structure BlockDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the total number of cubes in a block given its dimensions -/
def totalCubes (d : BlockDimensions) : ℕ :=
  d.length * d.width * d.height

/-- Calculates the number of invisible cubes when three faces are shown -/
def invisibleCubes (d : BlockDimensions) : ℕ :=
  (d.length - 1) * (d.width - 1) * (d.height - 1)

/-- Theorem stating the smallest possible value of N -/
theorem smallest_n_value (d : BlockDimensions) : 
  invisibleCubes d = 143 → totalCubes d ≥ 336 ∧ ∃ d', invisibleCubes d' = 143 ∧ totalCubes d' = 336 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_value_l876_87660


namespace NUMINAMATH_CALUDE_no_real_solutions_l876_87627

theorem no_real_solutions :
  ∀ x : ℝ, x ≠ 2 → (3 * x^2) / (x - 2) - (3 * x + 9) / 4 + (5 - 9 * x) / (x - 2) + 2 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l876_87627


namespace NUMINAMATH_CALUDE_magic_deck_price_is_two_l876_87624

/-- The price of a magic card deck given initial and final quantities and total earnings -/
def magic_deck_price (initial : ℕ) (final : ℕ) (earnings : ℕ) : ℚ :=
  earnings / (initial - final)

/-- Theorem: The price of each magic card deck is 2 dollars -/
theorem magic_deck_price_is_two :
  magic_deck_price 5 3 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_magic_deck_price_is_two_l876_87624


namespace NUMINAMATH_CALUDE_dot_product_range_l876_87641

/-- Given vectors a and b in a plane such that their magnitudes and the magnitude of their difference
    are between 2 and 6 (inclusive), prove that their dot product is between -14 and 34 (inclusive). -/
theorem dot_product_range (a b : ℝ × ℝ) 
  (ha : 2 ≤ ‖a‖ ∧ ‖a‖ ≤ 6)
  (hb : 2 ≤ ‖b‖ ∧ ‖b‖ ≤ 6)
  (hab : 2 ≤ ‖a - b‖ ∧ ‖a - b‖ ≤ 6) :
  -14 ≤ a • b ∧ a • b ≤ 34 :=
by sorry

end NUMINAMATH_CALUDE_dot_product_range_l876_87641


namespace NUMINAMATH_CALUDE_parallel_vectors_t_value_l876_87636

/-- Two vectors in ℝ² -/
def Vector2 := ℝ × ℝ

/-- Check if two vectors are parallel -/
def are_parallel (v w : Vector2) : Prop :=
  ∃ (k : ℝ), v.1 = k * w.1 ∧ v.2 = k * w.2

theorem parallel_vectors_t_value :
  ∀ (t : ℝ),
  let a : Vector2 := (t, -6)
  let b : Vector2 := (-3, 2)
  are_parallel a b → t = 9 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_t_value_l876_87636


namespace NUMINAMATH_CALUDE_complex_expression_equals_two_l876_87606

theorem complex_expression_equals_two :
  (Complex.I * (1 - Complex.I)^2 : ℂ) = 2 := by sorry

end NUMINAMATH_CALUDE_complex_expression_equals_two_l876_87606


namespace NUMINAMATH_CALUDE_chemistry_books_count_l876_87666

def number_of_biology_books : ℕ := 13
def total_combinations : ℕ := 2184

theorem chemistry_books_count (C : ℕ) : 
  (number_of_biology_books.choose 2) * (C.choose 2) = total_combinations → C = 8 := by
  sorry

end NUMINAMATH_CALUDE_chemistry_books_count_l876_87666


namespace NUMINAMATH_CALUDE_kayla_apple_count_l876_87628

/-- Given that Kylie and Kayla picked a total of 340 apples, and Kayla picked 10 more than 4 times 
    the amount of apples that Kylie picked, prove that Kayla picked 274 apples. -/
theorem kayla_apple_count :
  ∀ (kylie_apples : ℕ),
  kylie_apples + (10 + 4 * kylie_apples) = 340 →
  10 + 4 * kylie_apples = 274 :=
by
  sorry

end NUMINAMATH_CALUDE_kayla_apple_count_l876_87628


namespace NUMINAMATH_CALUDE_f_properties_l876_87673

def f (x : ℝ) : ℝ := x^3 + x^2 - 8*x + 6

theorem f_properties :
  (∀ x, deriv f x = 3*x^2 + 2*x - 8) ∧
  deriv f (-2) = 0 ∧
  deriv f 1 = -3 ∧
  f 1 = 0 ∧
  (∀ x, x < -2 ∨ x > 4/3 → deriv f x > 0) ∧
  (∀ x, -2 < x ∧ x < 4/3 → deriv f x < 0) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l876_87673


namespace NUMINAMATH_CALUDE_A_minus_3B_equals_x_cubed_plus_y_cubed_l876_87611

variable (x y : ℝ)

def A : ℝ := x^3 + 3*x^2*y + y^3 - 3*x*y^2
def B : ℝ := x^2*y - x*y^2

theorem A_minus_3B_equals_x_cubed_plus_y_cubed :
  A x y - 3 * B x y = x^3 + y^3 := by sorry

end NUMINAMATH_CALUDE_A_minus_3B_equals_x_cubed_plus_y_cubed_l876_87611


namespace NUMINAMATH_CALUDE_smallest_four_digit_pascal_l876_87640

/-- Pascal's triangle function -/
def pascal (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose n k

/-- Predicate for four-digit numbers -/
def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

/-- Theorem: 1001 is the smallest four-digit number in Pascal's triangle -/
theorem smallest_four_digit_pascal :
  (∃ n k, pascal n k = 1001) ∧
  (∀ n k, pascal n k < 1001 → ¬is_four_digit (pascal n k)) :=
sorry

end NUMINAMATH_CALUDE_smallest_four_digit_pascal_l876_87640


namespace NUMINAMATH_CALUDE_tangent_line_at_point_one_l876_87668

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - 2*x

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 - 2

-- Theorem statement
theorem tangent_line_at_point_one (x y : ℝ) :
  f 1 = -1 →
  f' 1 = 1 →
  (y - f 1 = f' 1 * (x - 1)) ↔ x - y - 2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_point_one_l876_87668


namespace NUMINAMATH_CALUDE_chess_tournament_games_l876_87671

/-- The number of games in a chess tournament --/
def num_games (n : ℕ) (games_per_pair : ℕ) : ℕ :=
  n * (n - 1) * games_per_pair / 2

/-- Theorem stating the number of games in the specific tournament --/
theorem chess_tournament_games :
  num_games 20 3 = 570 := by sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l876_87671


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l876_87646

theorem trigonometric_equation_solution (θ : ℝ) :
  3 * Real.sin (-3 * Real.pi + θ) + Real.cos (Real.pi - θ) = 0 →
  (Real.sin θ * Real.cos θ) / Real.cos (2 * θ) = -3/8 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l876_87646


namespace NUMINAMATH_CALUDE_complex_expression_value_l876_87601

theorem complex_expression_value (x y : ℂ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : x^2 + x*y + y^2 = 0) : 
  (x / (x + y))^1990 + (y / (x + y))^1990 = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_value_l876_87601


namespace NUMINAMATH_CALUDE_alien_mineral_collection_l876_87639

/-- Converts a base-6 number represented as a list of digits to its base-10 equivalent -/
def base6ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (6 ^ i)) 0

/-- The base-6 representation of the number -/
def alienCollection : List Nat := [5, 3, 2]

theorem alien_mineral_collection :
  base6ToBase10 alienCollection = 95 := by
  sorry

end NUMINAMATH_CALUDE_alien_mineral_collection_l876_87639


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ninth_term_l876_87604

/-- Given an arithmetic sequence where the third term is 23 and the sixth term is 29,
    prove that the ninth term is 35. -/
theorem arithmetic_sequence_ninth_term
  (a : ℕ → ℝ)  -- a is the arithmetic sequence
  (h1 : a 3 = 23)  -- third term is 23
  (h2 : a 6 = 29)  -- sixth term is 29
  (h3 : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1)  -- definition of arithmetic sequence
  : a 9 = 35 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ninth_term_l876_87604


namespace NUMINAMATH_CALUDE_unique_solution_g_equals_g_inv_l876_87698

-- Define the function g
def g (x : ℝ) : ℝ := 5 * x - 12

-- Define the inverse function g⁻¹
noncomputable def g_inv (x : ℝ) : ℝ := (x + 12) / 5

-- Theorem statement
theorem unique_solution_g_equals_g_inv :
  ∃! x : ℝ, g x = g_inv x :=
by
  sorry

end NUMINAMATH_CALUDE_unique_solution_g_equals_g_inv_l876_87698


namespace NUMINAMATH_CALUDE_investment_calculation_l876_87651

/-- Given a total investment split between a savings account and mutual funds,
    where the investment in mutual funds is 6 times the investment in the savings account,
    calculate the total investment in mutual funds. -/
theorem investment_calculation (total : ℝ) (savings : ℝ) (mutual_funds : ℝ)
    (h1 : total = 320000)
    (h2 : mutual_funds = 6 * savings)
    (h3 : total = savings + mutual_funds) :
  mutual_funds = 274285.74 := by
  sorry

end NUMINAMATH_CALUDE_investment_calculation_l876_87651


namespace NUMINAMATH_CALUDE_cubic_equation_solutions_l876_87625

theorem cubic_equation_solutions :
  let f : ℂ → ℂ := λ x => (x^3 + 3*x^2*Real.sqrt 2 + 6*x + 2*Real.sqrt 2) + (x + Real.sqrt 2)
  ∀ x : ℂ, f x = 0 ↔ x = -Real.sqrt 2 ∨ x = -Real.sqrt 2 + Complex.I ∨ x = -Real.sqrt 2 - Complex.I :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_solutions_l876_87625


namespace NUMINAMATH_CALUDE_range_of_m_l876_87608

theorem range_of_m (m : ℝ) : 
  (∀ x y : ℝ, x > 0 → y > 0 → (2 * y / x + 8 * x / y) > m^2 + 2*m) → 
  -4 < m ∧ m < 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l876_87608


namespace NUMINAMATH_CALUDE_odd_function_symmetry_l876_87652

/-- A function f is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_function_symmetry (f : ℝ → ℝ) (a : ℝ) (h : IsOdd f) :
  f (-a) = -f a := by sorry

end NUMINAMATH_CALUDE_odd_function_symmetry_l876_87652


namespace NUMINAMATH_CALUDE_clique_six_and_best_degree_l876_87694

/-- A graph with 1991 points where every point has degree at least 1593 -/
structure Graph1991 where
  vertices : Finset (Fin 1991)
  edges : Finset (Fin 1991 × Fin 1991)
  degree_condition : ∀ v : Fin 1991, (edges.filter (λ e => e.1 = v ∨ e.2 = v)).card ≥ 1593

/-- A clique is a subset of vertices where every two distinct vertices are adjacent -/
def is_clique (G : Graph1991) (S : Finset (Fin 1991)) : Prop :=
  ∀ u v : Fin 1991, u ∈ S → v ∈ S → u ≠ v → (u, v) ∈ G.edges ∨ (v, u) ∈ G.edges

/-- The main theorem stating that there exists a clique of size 6 and 1593 is the best possible -/
theorem clique_six_and_best_degree (G : Graph1991) :
  (∃ S : Finset (Fin 1991), S.card = 6 ∧ is_clique G S) ∧
  ∀ d < 1593, ∃ H : Graph1991, ¬∃ S : Finset (Fin 1991), S.card = 6 ∧ is_clique H S :=
sorry

end NUMINAMATH_CALUDE_clique_six_and_best_degree_l876_87694


namespace NUMINAMATH_CALUDE_task_completion_time_l876_87685

/-- The time required for Sumin and Junwoo to complete a task together, given their individual work rates -/
theorem task_completion_time (sumin_rate junwoo_rate : ℚ) 
  (h_sumin : sumin_rate = 1 / 10)
  (h_junwoo : junwoo_rate = 1 / 15) :
  (1 : ℚ) / (sumin_rate + junwoo_rate) = 6 := by
  sorry

end NUMINAMATH_CALUDE_task_completion_time_l876_87685


namespace NUMINAMATH_CALUDE_percent_increase_decrease_l876_87658

theorem percent_increase_decrease (p q M : ℝ) 
  (hp : p > 0) (hq : q > 0) (hq_bound : q < 100) (hM : M > 0) :
  (M * (1 + p/100) * (1 - q/100) > M) ↔ (p > 100*q / (100 - q)) := by
  sorry

end NUMINAMATH_CALUDE_percent_increase_decrease_l876_87658


namespace NUMINAMATH_CALUDE_number_percentage_problem_l876_87664

theorem number_percentage_problem (N : ℝ) : 
  (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * N = 35 → 0.40 * N = 420 := by
  sorry

end NUMINAMATH_CALUDE_number_percentage_problem_l876_87664


namespace NUMINAMATH_CALUDE_no_solution_equation_l876_87699

theorem no_solution_equation : ∀ (a b : ℤ), a^4 + 6 ≠ b^3 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_equation_l876_87699


namespace NUMINAMATH_CALUDE_other_root_of_complex_quadratic_l876_87688

theorem other_root_of_complex_quadratic (z : ℂ) :
  z^2 = -75 + 100*I ∧ z = 5 + 10*I → (-z)^2 = -75 + 100*I :=
by
  sorry

end NUMINAMATH_CALUDE_other_root_of_complex_quadratic_l876_87688


namespace NUMINAMATH_CALUDE_problem_solution_l876_87629

def f (a : ℝ) (x : ℝ) : ℝ := |a * x - 1|

theorem problem_solution (a : ℝ) :
  (∀ x, f a x ≤ 2 ↔ -1/2 ≤ x ∧ x ≤ 3/2) →
  (a = 2 ∧
   ∀ x, f 2 x + f 2 (x/2 - 1) ≥ 5 ↔ x ≥ 3 ∨ x ≤ -1/3) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l876_87629


namespace NUMINAMATH_CALUDE_unique_symmetric_solutions_l876_87614

theorem unique_symmetric_solutions (a b α β : ℝ) :
  (α * β = a ∧ α + β = b) →
  (∀ x y : ℝ, x * y = a ∧ x + y = b ↔ (x = α ∧ y = β) ∨ (x = β ∧ y = α)) :=
sorry

end NUMINAMATH_CALUDE_unique_symmetric_solutions_l876_87614


namespace NUMINAMATH_CALUDE_sum_of_solutions_eq_four_l876_87609

theorem sum_of_solutions_eq_four :
  let f : ℝ → ℝ := λ N => N * (N - 4)
  let solutions := {N : ℝ | f N = -21}
  (∃ N₁ N₂, N₁ ∈ solutions ∧ N₂ ∈ solutions ∧ N₁ ≠ N₂) →
  (∀ N, N ∈ solutions → N₁ = N ∨ N₂ = N) →
  N₁ + N₂ = 4
  := by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_eq_four_l876_87609


namespace NUMINAMATH_CALUDE_chord_equation_l876_87642

/-- Given positive real numbers m, n, s, t satisfying certain conditions,
    prove that the equation of the line containing a chord of the hyperbola
    x²/4 - y²/2 = 1 with midpoint (m, n) is x - 2y + 1 = 0. -/
theorem chord_equation (m n s t : ℝ) 
    (hm : m > 0) (hn : n > 0) (hs : s > 0) (ht : t > 0)
    (h1 : m + n = 2)
    (h2 : m / s + n / t = 9)
    (h3 : s + t = 4 / 9)
    (h4 : ∀ s' t' : ℝ, s' > 0 → t' > 0 → m / s' + n / t' = 9 → s' + t' ≥ 4 / 9)
    (h5 : ∃ x₁ y₁ x₂ y₂ : ℝ, 
      x₁^2 / 4 - y₁^2 / 2 = 1 ∧
      x₂^2 / 4 - y₂^2 / 2 = 1 ∧
      (x₁ + x₂) / 2 = m ∧
      (y₁ + y₂) / 2 = n) :
  ∃ a b c : ℝ, a * m + b * n + c = 0 ∧
             ∀ x y : ℝ, x^2 / 4 - y^2 / 2 = 1 →
               (∃ t : ℝ, x = m + t * a ∧ y = n + t * b) →
               a * x + b * y + c = 0 ∧
               a = 1 ∧ b = -2 ∧ c = 1 :=
by sorry

end NUMINAMATH_CALUDE_chord_equation_l876_87642


namespace NUMINAMATH_CALUDE_antonio_hamburger_usage_l876_87603

/-- Calculates the total amount of hamburger used for meatballs given the number of family members,
    meatballs per person, and amount of hamburger per meatball. -/
def hamburger_used (family_members : ℕ) (meatballs_per_person : ℕ) (hamburger_per_meatball : ℚ) : ℚ :=
  (family_members * meatballs_per_person : ℚ) * hamburger_per_meatball

/-- Proves that given the conditions in the problem, Antonio used 4 pounds of hamburger. -/
theorem antonio_hamburger_usage :
  let family_members : ℕ := 8
  let meatballs_per_person : ℕ := 4
  let hamburger_per_meatball : ℚ := 1/8
  hamburger_used family_members meatballs_per_person hamburger_per_meatball = 4 := by
  sorry


end NUMINAMATH_CALUDE_antonio_hamburger_usage_l876_87603


namespace NUMINAMATH_CALUDE_x_in_open_interval_one_two_l876_87670

/-- A monotonically increasing function on (0,+∞) -/
def MonoIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x ∧ 0 < y ∧ x < y → f x < f y

theorem x_in_open_interval_one_two
  (f : ℝ → ℝ)
  (h_mono : MonoIncreasing f)
  (h_gt : ∀ x, 0 < x → f x > f (2 - x)) :
  ∃ x, 1 < x ∧ x < 2 :=
sorry

end NUMINAMATH_CALUDE_x_in_open_interval_one_two_l876_87670


namespace NUMINAMATH_CALUDE_cubic_extrema_l876_87682

-- Define a cubic function
def cubic_function (a b c d : ℝ) : ℝ → ℝ := λ x => a * x^3 + b * x^2 + c * x + d

-- Define the derivative of the cubic function
def cubic_derivative (a b c : ℝ) : ℝ → ℝ := λ x => 3 * a * x^2 + 2 * b * x + c

-- State the theorem
theorem cubic_extrema (a b c d : ℝ) :
  let f := cubic_function a b c d
  let f' := cubic_derivative (3*a) (2*b) c
  (∀ x, x * f' x = 0 ↔ x = 0 ∨ x = 2 ∨ x = -2) →
  (∀ x, f x ≤ f (-2)) ∧ (∀ x, f 2 ≤ f x) :=
sorry

end NUMINAMATH_CALUDE_cubic_extrema_l876_87682


namespace NUMINAMATH_CALUDE_not_divides_2007_l876_87626

theorem not_divides_2007 : ¬(2007 ∣ (2009^3 - 2009)) := by sorry

end NUMINAMATH_CALUDE_not_divides_2007_l876_87626


namespace NUMINAMATH_CALUDE_sylvester_theorem_l876_87691

-- Define coprimality
def coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

-- Define the theorem
theorem sylvester_theorem (a b : ℕ) (h : coprime a b) :
  -- Part 1: Unique solution in the strip
  (∀ c : ℕ, ∃! p : ℕ × ℕ, p.1 < b ∧ a * p.1 + b * p.2 = c) ∧
  -- Part 2: Largest value without non-negative solutions
  (∀ c : ℕ, c > a * b - a - b → ∃ x y : ℕ, a * x + b * y = c) ∧
  (¬∃ x y : ℕ, a * x + b * y = a * b - a - b) := by
  sorry

end NUMINAMATH_CALUDE_sylvester_theorem_l876_87691


namespace NUMINAMATH_CALUDE_gecko_insect_consumption_l876_87654

theorem gecko_insect_consumption (geckos lizards total_insects : ℕ) 
  (h1 : geckos = 5)
  (h2 : lizards = 3)
  (h3 : total_insects = 66) :
  ∃ (gecko_consumption : ℕ), 
    gecko_consumption * geckos + (2 * gecko_consumption) * lizards = total_insects ∧ 
    gecko_consumption = 6 := by
  sorry

end NUMINAMATH_CALUDE_gecko_insect_consumption_l876_87654


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l876_87653

-- Define repeating decimals
def repeating_decimal_2 : ℚ := 2/9
def repeating_decimal_03 : ℚ := 1/33

-- Theorem statement
theorem sum_of_repeating_decimals :
  repeating_decimal_2 + repeating_decimal_03 = 25/99 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l876_87653


namespace NUMINAMATH_CALUDE_beadshop_profit_l876_87662

theorem beadshop_profit (monday_profit_ratio : ℚ) (tuesday_profit_ratio : ℚ) (wednesday_profit : ℚ) 
  (h1 : monday_profit_ratio = 1/3)
  (h2 : tuesday_profit_ratio = 1/4)
  (h3 : wednesday_profit = 500) :
  ∃ total_profit : ℚ, 
    total_profit * (1 - monday_profit_ratio - tuesday_profit_ratio) = wednesday_profit ∧
    total_profit = 1200 := by
sorry

end NUMINAMATH_CALUDE_beadshop_profit_l876_87662


namespace NUMINAMATH_CALUDE_proposition_truth_l876_87612

theorem proposition_truth (p q : Prop) 
  (h1 : ¬(p ∧ q)) 
  (h2 : ¬¬q) : 
  (¬p ∧ q) := by
  sorry

end NUMINAMATH_CALUDE_proposition_truth_l876_87612


namespace NUMINAMATH_CALUDE_intersection_line_equation_l876_87630

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 10
def circle2 (x y : ℝ) : Prop := (x - 1)^2 + (y - 3)^2 = 20

-- Define the line
def line (x y : ℝ) : Prop := x + 3*y = 0

-- Theorem statement
theorem intersection_line_equation :
  ∀ (A B : ℝ × ℝ),
  (circle1 A.1 A.2 ∧ circle2 A.1 A.2) →
  (circle1 B.1 B.2 ∧ circle2 B.1 B.2) →
  A ≠ B →
  line A.1 A.2 ∧ line B.1 B.2 :=
sorry

end NUMINAMATH_CALUDE_intersection_line_equation_l876_87630


namespace NUMINAMATH_CALUDE_john_travel_distance_l876_87615

/-- Calculates the total distance traveled given a constant speed and two driving periods -/
def totalDistance (speed : ℝ) (time1 : ℝ) (time2 : ℝ) : ℝ :=
  speed * (time1 + time2)

/-- Proves that the total distance traveled is 225 miles -/
theorem john_travel_distance :
  let speed := 45
  let time1 := 2
  let time2 := 3
  totalDistance speed time1 time2 = 225 := by
sorry

end NUMINAMATH_CALUDE_john_travel_distance_l876_87615


namespace NUMINAMATH_CALUDE_exists_integer_fifth_power_less_than_one_l876_87684

theorem exists_integer_fifth_power_less_than_one :
  ∃ x : ℤ, x^5 < 1 := by sorry

end NUMINAMATH_CALUDE_exists_integer_fifth_power_less_than_one_l876_87684


namespace NUMINAMATH_CALUDE_power_fraction_simplification_l876_87620

theorem power_fraction_simplification : (3^100 + 3^98) / (3^100 - 3^98) = 5/4 := by sorry

end NUMINAMATH_CALUDE_power_fraction_simplification_l876_87620


namespace NUMINAMATH_CALUDE_currency_notes_total_l876_87678

theorem currency_notes_total (total_notes : ℕ) (denom_1 denom_2 : ℕ) (amount_denom_2 : ℕ) : 
  total_notes = 100 → 
  denom_1 = 70 → 
  denom_2 = 50 → 
  amount_denom_2 = 100 →
  ∃ (notes_denom_1 notes_denom_2 : ℕ),
    notes_denom_1 + notes_denom_2 = total_notes ∧
    notes_denom_2 * denom_2 = amount_denom_2 ∧
    notes_denom_1 * denom_1 + notes_denom_2 * denom_2 = 6960 :=
by sorry

end NUMINAMATH_CALUDE_currency_notes_total_l876_87678


namespace NUMINAMATH_CALUDE_sum_of_decimals_l876_87644

theorem sum_of_decimals : 0.305 + 0.089 + 0.007 = 0.401 := by sorry

end NUMINAMATH_CALUDE_sum_of_decimals_l876_87644


namespace NUMINAMATH_CALUDE_prob_girl_from_E_expected_value_X_l876_87681

/-- Represents a family with a number of boys and girls -/
structure Family :=
  (boys : Nat)
  (girls : Nat)

/-- The set of all families -/
def Families : Finset (Fin 5) := Finset.univ

/-- The number of boys and girls in each family -/
def familyData : Fin 5 → Family
  | ⟨0, _⟩ => ⟨0, 0⟩  -- Family A
  | ⟨1, _⟩ => ⟨1, 0⟩  -- Family B
  | ⟨2, _⟩ => ⟨0, 1⟩  -- Family C
  | ⟨3, _⟩ => ⟨1, 1⟩  -- Family D
  | ⟨4, _⟩ => ⟨1, 2⟩  -- Family E

/-- The total number of children -/
def totalChildren : Nat := Finset.sum Families (λ i => (familyData i).boys + (familyData i).girls)

/-- The total number of girls -/
def totalGirls : Nat := Finset.sum Families (λ i => (familyData i).girls)

/-- Probability of selecting a girl from family E given that a girl is selected -/
theorem prob_girl_from_E : 
  (familyData 4).girls / totalGirls = 1 / 2 := by sorry

/-- Probability distribution of X when selecting 3 families -/
def probDistX (x : Fin 3) : Rat :=
  match x with
  | ⟨0, _⟩ => 1 / 10
  | ⟨1, _⟩ => 3 / 5
  | ⟨2, _⟩ => 3 / 10

/-- Expected value of X -/
def expectedX : Rat := Finset.sum (Finset.range 3) (λ i => i * probDistX i)

theorem expected_value_X : expectedX = 6 / 5 := by sorry

end NUMINAMATH_CALUDE_prob_girl_from_E_expected_value_X_l876_87681


namespace NUMINAMATH_CALUDE_speed_conversion_l876_87675

theorem speed_conversion (speed_ms : ℝ) (speed_kmh : ℝ) : 
  speed_ms = 9/36 → speed_kmh = 0.9 → (1 : ℝ) * 3.6 = speed_kmh / speed_ms :=
by
  sorry

end NUMINAMATH_CALUDE_speed_conversion_l876_87675


namespace NUMINAMATH_CALUDE_ab_value_l876_87697

theorem ab_value (a b : ℝ) (h : (a + 3)^2 + (b - 3)^2 = 0) : a^b = -27 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l876_87697


namespace NUMINAMATH_CALUDE_smallest_number_with_remainders_l876_87679

theorem smallest_number_with_remainders : ∃ n : ℕ, 
  (n > 0) ∧ 
  (n % 3 = 2) ∧ 
  (n % 5 = 3) ∧ 
  (∀ m : ℕ, m > 0 → m % 3 = 2 → m % 5 = 3 → n ≤ m) ∧
  n = 8 := by
sorry

end NUMINAMATH_CALUDE_smallest_number_with_remainders_l876_87679


namespace NUMINAMATH_CALUDE_power_function_m_value_l876_87656

def is_power_function (f : ℝ → ℝ) : Prop :=
  ∃ (a : ℝ) (n : ℝ), ∀ x, f x = a * x^n

theorem power_function_m_value (m : ℝ) :
  is_power_function (λ x => (3*m - 1) * x^m) → m = 2/3 :=
by
  sorry

end NUMINAMATH_CALUDE_power_function_m_value_l876_87656


namespace NUMINAMATH_CALUDE_four_numbers_product_sum_prime_l876_87659

theorem four_numbers_product_sum_prime :
  ∃ (a b c d : ℕ), a < b ∧ b < c ∧ c < d ∧
  Nat.Prime (a * b + c * d) ∧
  Nat.Prime (a * c + b * d) ∧
  Nat.Prime (a * d + b * c) := by
  sorry

end NUMINAMATH_CALUDE_four_numbers_product_sum_prime_l876_87659


namespace NUMINAMATH_CALUDE_amount_decreased_l876_87680

theorem amount_decreased (x y : ℝ) (h1 : x = 50.0) (h2 : 0.20 * x - y = 6) : y = 4 := by
  sorry

end NUMINAMATH_CALUDE_amount_decreased_l876_87680


namespace NUMINAMATH_CALUDE_coin_flip_probability_l876_87607

theorem coin_flip_probability :
  let n : ℕ := 12  -- number of coin flips
  let k : ℕ := 9   -- number of heads we want
  let p : ℚ := 1/2 -- probability of heads for a fair coin
  Nat.choose n k * p^k * (1-p)^(n-k) = 55/1024 := by
sorry

end NUMINAMATH_CALUDE_coin_flip_probability_l876_87607


namespace NUMINAMATH_CALUDE_perpendicular_chords_sum_bounds_l876_87633

/-- Given a circle with radius R and an interior point P at distance kR from the center,
    where 0 ≤ k ≤ 1, the sum of the lengths of two perpendicular chords passing through P
    is bounded above by 2R√(2(1 - k²)) and below by 0. -/
theorem perpendicular_chords_sum_bounds (R k : ℝ) (h_R_pos : R > 0) (h_k_range : 0 ≤ k ∧ k ≤ 1) :
  ∃ (chord_sum : ℝ), 0 ≤ chord_sum ∧ chord_sum ≤ 2 * R * Real.sqrt (2 * (1 - k^2)) := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_chords_sum_bounds_l876_87633


namespace NUMINAMATH_CALUDE_f_no_extreme_points_l876_87655

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + 3*x^2 + 3*x - a

-- Theorem stating that f has no extreme points for any real a
theorem f_no_extreme_points (a : ℝ) : 
  ∀ x : ℝ, ∃ ε > 0, ∀ y ∈ Set.Ioo (x - ε) (x + ε), f a y ≠ f a x ∨ (f a y < f a x ∧ y < x) ∨ (f a y > f a x ∧ y > x) :=
sorry

end NUMINAMATH_CALUDE_f_no_extreme_points_l876_87655


namespace NUMINAMATH_CALUDE_freddy_travel_time_l876_87635

/-- Represents the travel details of a person --/
structure TravelDetails where
  startCity : String
  endCity : String
  distance : Real
  time : Real

/-- The problem setup --/
def problem : Prop := ∃ (eddySpeed freddySpeed : Real),
  let eddy : TravelDetails := ⟨"A", "B", 900, 3⟩
  let freddy : TravelDetails := ⟨"A", "C", 300, freddySpeed / 300⟩
  eddySpeed = eddy.distance / eddy.time ∧
  eddySpeed / freddySpeed = 4 ∧
  freddy.time = 4

/-- The theorem to be proved --/
theorem freddy_travel_time : problem := by sorry

end NUMINAMATH_CALUDE_freddy_travel_time_l876_87635


namespace NUMINAMATH_CALUDE_solve_system_and_find_perimeter_l876_87689

/-- Given a system of equations, prove the values of a and b, and the perimeter of an isosceles triangle with these side lengths. -/
theorem solve_system_and_find_perimeter :
  ∃ (a b : ℝ),
    (4 * a - 3 * b = 22) ∧
    (2 * a + b = 16) ∧
    (a = 7) ∧
    (b = 2) ∧
    (2 * max a b + min a b = 16) := by
  sorry


end NUMINAMATH_CALUDE_solve_system_and_find_perimeter_l876_87689


namespace NUMINAMATH_CALUDE_profit_doubling_l876_87634

theorem profit_doubling (cost : ℝ) (original_price : ℝ) :
  original_price = cost * 1.6 →
  let double_price := 2 * original_price
  (double_price - cost) / cost * 100 = 220 := by
sorry

end NUMINAMATH_CALUDE_profit_doubling_l876_87634


namespace NUMINAMATH_CALUDE_goldfish_preference_total_l876_87622

/-- Calculates the total number of students preferring goldfish across three classes -/
theorem goldfish_preference_total (class_size : ℕ) 
  (johnson_fraction : ℚ) (feldstein_fraction : ℚ) (henderson_fraction : ℚ)
  (h1 : class_size = 30)
  (h2 : johnson_fraction = 1 / 6)
  (h3 : feldstein_fraction = 2 / 3)
  (h4 : henderson_fraction = 1 / 5) :
  ⌊class_size * johnson_fraction⌋ + ⌊class_size * feldstein_fraction⌋ + ⌊class_size * henderson_fraction⌋ = 31 :=
by sorry

end NUMINAMATH_CALUDE_goldfish_preference_total_l876_87622


namespace NUMINAMATH_CALUDE_regular_polygon_perimeter_l876_87610

/-- A regular polygon with exterior angle 90 degrees and side length 7 units has a perimeter of 28 units. -/
theorem regular_polygon_perimeter (n : ℕ) (side_length : ℝ) (exterior_angle : ℝ) :
  n > 0 ∧ 
  side_length = 7 ∧ 
  exterior_angle = 90 ∧ 
  n * exterior_angle = 360 →
  n * side_length = 28 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_perimeter_l876_87610


namespace NUMINAMATH_CALUDE_zhang_li_age_ratio_l876_87663

-- Define the ages
def li_age : ℕ := 12
def jung_age : ℕ := 26

-- Define Zhang's age based on Jung's age
def zhang_age : ℕ := jung_age - 2

-- Define the ratio of Zhang's age to Li's age
def age_ratio : ℚ := zhang_age / li_age

-- Theorem statement
theorem zhang_li_age_ratio :
  age_ratio = 2 := by sorry

end NUMINAMATH_CALUDE_zhang_li_age_ratio_l876_87663


namespace NUMINAMATH_CALUDE_hiltons_marbles_l876_87693

theorem hiltons_marbles (initial_marbles : ℕ) : 
  (initial_marbles + 6 - 10 + 2 * 10 = 42) → initial_marbles = 26 := by
  sorry

end NUMINAMATH_CALUDE_hiltons_marbles_l876_87693


namespace NUMINAMATH_CALUDE_number_of_people_l876_87692

/-- Given a group of people, prove that there are 5 people based on the given conditions. -/
theorem number_of_people (n : ℕ) (total_age : ℕ) : n = 5 :=
  by
  /- Define the average age of all people -/
  have avg_age : total_age = n * 30 := by sorry
  
  /- Define the total age when the youngest was born -/
  have prev_total_age : total_age - 6 = (n - 1) * 24 := by sorry
  
  /- The main proof -/
  sorry

end NUMINAMATH_CALUDE_number_of_people_l876_87692


namespace NUMINAMATH_CALUDE_evelyn_found_caps_l876_87618

/-- The number of bottle caps Evelyn started with -/
def starting_caps : ℕ := 18

/-- The number of bottle caps Evelyn ended up with -/
def total_caps : ℕ := 81

/-- The number of bottle caps Evelyn found -/
def found_caps : ℕ := total_caps - starting_caps

theorem evelyn_found_caps : found_caps = 63 := by
  sorry

end NUMINAMATH_CALUDE_evelyn_found_caps_l876_87618


namespace NUMINAMATH_CALUDE_preceding_number_in_base_three_l876_87643

/-- Converts a base-3 number to decimal --/
def baseThreeToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ i)) 0

/-- Converts a decimal number to base-3 --/
def decimalToBaseThree (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) :=
      if m = 0 then acc else aux (m / 3) ((m % 3) :: acc)
    aux n []

theorem preceding_number_in_base_three (M : List Nat) (h : M = [2, 1, 0, 2, 1]) :
  decimalToBaseThree (baseThreeToDecimal M - 1) = [2, 1, 0, 2, 0] := by
  sorry

end NUMINAMATH_CALUDE_preceding_number_in_base_three_l876_87643


namespace NUMINAMATH_CALUDE_trigonometric_expression_equality_l876_87661

theorem trigonometric_expression_equality : 
  (Real.sin (92 * π / 180) - Real.sin (32 * π / 180) * Real.cos (60 * π / 180)) / Real.cos (32 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_equality_l876_87661


namespace NUMINAMATH_CALUDE_circle_diameter_ratio_l876_87616

theorem circle_diameter_ratio (D C : ℝ) : 
  D = 20 → -- Diameter of circle D is 20 cm
  C > 0 → -- Diameter of circle C is positive
  C < D → -- Circle C is inside circle D
  (π * D^2 / 4 - π * C^2 / 4) / (π * C^2 / 4) = 4 → -- Ratio of shaded area to area of C is 4:1
  C = 8 * Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_circle_diameter_ratio_l876_87616


namespace NUMINAMATH_CALUDE_peter_bought_nine_kilos_of_tomatoes_l876_87632

/-- Represents the purchase of groceries by Peter -/
structure Groceries where
  initialMoney : ℕ
  potatoPrice : ℕ
  potatoKilos : ℕ
  tomatoPrice : ℕ
  cucumberPrice : ℕ
  cucumberKilos : ℕ
  bananaPrice : ℕ
  bananaKilos : ℕ
  remainingMoney : ℕ

/-- Calculates the number of kilos of tomatoes bought -/
def tomatoKilos (g : Groceries) : ℕ :=
  (g.initialMoney - g.remainingMoney - 
   (g.potatoPrice * g.potatoKilos + 
    g.cucumberPrice * g.cucumberKilos + 
    g.bananaPrice * g.bananaKilos)) / g.tomatoPrice

/-- Theorem stating that Peter bought 9 kilos of tomatoes -/
theorem peter_bought_nine_kilos_of_tomatoes (g : Groceries) 
  (h1 : g.initialMoney = 500)
  (h2 : g.potatoPrice = 2)
  (h3 : g.potatoKilos = 6)
  (h4 : g.tomatoPrice = 3)
  (h5 : g.cucumberPrice = 4)
  (h6 : g.cucumberKilos = 5)
  (h7 : g.bananaPrice = 5)
  (h8 : g.bananaKilos = 3)
  (h9 : g.remainingMoney = 426) :
  tomatoKilos g = 9 := by
  sorry

end NUMINAMATH_CALUDE_peter_bought_nine_kilos_of_tomatoes_l876_87632


namespace NUMINAMATH_CALUDE_parabola_vertex_first_quadrant_l876_87676

/-- A parabola with equation y = (x-m)^2 + (m-1) has its vertex in the first quadrant if and only if m > 1 -/
theorem parabola_vertex_first_quadrant (m : ℝ) : 
  (∃ x y : ℝ, y = (x - m)^2 + (m - 1) ∧ x = m ∧ y = m - 1 ∧ x > 0 ∧ y > 0) ↔ m > 1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_first_quadrant_l876_87676


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_thirteen_sixths_l876_87631

theorem sqrt_sum_equals_thirteen_sixths : 
  Real.sqrt (9 / 4) + Real.sqrt (4 / 9) = 13 / 6 := by sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_thirteen_sixths_l876_87631


namespace NUMINAMATH_CALUDE_lateral_edge_length_l876_87674

-- Define the regular triangular pyramid
structure RegularTriangularPyramid where
  baseEdge : ℝ
  lateralEdge : ℝ

-- Define the property of medians not intersecting and lying on cube edges
def mediansPropertyHolds (pyramid : RegularTriangularPyramid) : Prop :=
  -- This is a placeholder for the complex geometric condition
  -- In a real implementation, this would involve more detailed geometric definitions
  sorry

-- Theorem statement
theorem lateral_edge_length
  (pyramid : RegularTriangularPyramid)
  (h1 : pyramid.baseEdge = 1)
  (h2 : mediansPropertyHolds pyramid) :
  pyramid.lateralEdge = Real.sqrt 6 / 2 :=
sorry

end NUMINAMATH_CALUDE_lateral_edge_length_l876_87674


namespace NUMINAMATH_CALUDE_barry_total_amount_l876_87690

/-- Calculates the total amount Barry needs to pay for his purchase --/
def calculate_total_amount (shirt_price pants_price tie_price : ℝ)
  (shirt_discount pants_discount coupon_discount sales_tax : ℝ) : ℝ :=
  let discounted_shirt := shirt_price * (1 - shirt_discount)
  let discounted_pants := pants_price * (1 - pants_discount)
  let subtotal := discounted_shirt + discounted_pants + tie_price
  let after_coupon := subtotal * (1 - coupon_discount)
  let total := after_coupon * (1 + sales_tax)
  total

/-- Theorem stating that the total amount Barry needs to pay is $201.27 --/
theorem barry_total_amount : 
  calculate_total_amount 80 100 40 0.15 0.10 0.05 0.07 = 201.27 := by
  sorry

end NUMINAMATH_CALUDE_barry_total_amount_l876_87690


namespace NUMINAMATH_CALUDE_find_k_l876_87602

theorem find_k (x y z k : ℝ) 
  (h1 : 7 / (x + y) = k / (x + z)) 
  (h2 : k / (x + z) = 11 / (z - y)) : k = 18 := by
  sorry

end NUMINAMATH_CALUDE_find_k_l876_87602


namespace NUMINAMATH_CALUDE_exam_maximum_marks_l876_87667

theorem exam_maximum_marks 
  (passing_percentage : ℝ)
  (student_score : ℕ)
  (failing_margin : ℕ)
  (h1 : passing_percentage = 0.45)
  (h2 : student_score = 40)
  (h3 : failing_margin = 40) :
  ∃ (max_marks : ℕ), max_marks = 180 ∧ 
    (passing_percentage * max_marks : ℝ) = (student_score + failing_margin) :=
by sorry

end NUMINAMATH_CALUDE_exam_maximum_marks_l876_87667


namespace NUMINAMATH_CALUDE_rhombus_sides_equal_l876_87613

/-- A rhombus is a quadrilateral with all sides equal -/
structure Rhombus where
  sides : Fin 4 → ℝ
  is_quadrilateral : True
  all_sides_equal : ∀ (i j : Fin 4), sides i = sides j

/-- All four sides of a rhombus are equal -/
theorem rhombus_sides_equal (r : Rhombus) : 
  ∀ (i j : Fin 4), r.sides i = r.sides j := by
  sorry

end NUMINAMATH_CALUDE_rhombus_sides_equal_l876_87613


namespace NUMINAMATH_CALUDE_geometric_subsequence_contains_342_l876_87649

/-- An arithmetic sequence with a non-zero common difference -/
structure ArithmeticSequence :=
  (a : ℕ → ℝ)
  (d : ℝ)
  (h_arith : ∀ n : ℕ, a (n + 1) = a n + d)
  (h_d_nonzero : d ≠ 0)

/-- A geometric sequence extracted from an arithmetic sequence -/
structure GeometricSubsequence (as : ArithmeticSequence) :=
  (seq : ℕ → ℝ)
  (h_geom : ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, seq (n + 1) = q * seq n)
  (h_sub : ∃ f : ℕ → ℕ, ∀ n : ℕ, seq n = as.a (f n))
  (h_2_6_22 : ∃ k₁ k₂ k₃ : ℕ, seq k₁ = as.a 2 ∧ seq k₂ = as.a 6 ∧ seq k₃ = as.a 22)

/-- The main theorem -/
theorem geometric_subsequence_contains_342 (as : ArithmeticSequence) 
  (gs : GeometricSubsequence as) : 
  ∃ k : ℕ, gs.seq k = as.a 342 := by
  sorry

end NUMINAMATH_CALUDE_geometric_subsequence_contains_342_l876_87649
