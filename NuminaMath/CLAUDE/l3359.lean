import Mathlib

namespace NUMINAMATH_CALUDE_salary_increase_l3359_335952

theorem salary_increase (num_employees : ℕ) (avg_salary : ℝ) (manager_salary : ℝ) :
  num_employees = 20 →
  avg_salary = 1500 →
  manager_salary = 12000 →
  let total_salary := num_employees * avg_salary
  let new_total := total_salary + manager_salary
  let new_avg := new_total / (num_employees + 1)
  new_avg - avg_salary = 500 := by
  sorry

end NUMINAMATH_CALUDE_salary_increase_l3359_335952


namespace NUMINAMATH_CALUDE_odd_times_even_is_odd_l3359_335921

-- Define the properties of odd and even functions
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- State the theorem
theorem odd_times_even_is_odd
  (f g : ℝ → ℝ) (hf : IsOdd f) (hg : IsEven g) :
  IsOdd (fun x ↦ f x * g x) := by
  sorry

#check odd_times_even_is_odd

end NUMINAMATH_CALUDE_odd_times_even_is_odd_l3359_335921


namespace NUMINAMATH_CALUDE_modular_equivalence_in_range_l3359_335938

theorem modular_equivalence_in_range (a b : ℤ) (h1 : a ≡ 54 [ZMOD 53]) (h2 : b ≡ 98 [ZMOD 53]) :
  ∃! n : ℤ, 150 ≤ n ∧ n ≤ 200 ∧ (a - b) ≡ n [ZMOD 53] ∧ n = 168 := by
  sorry

end NUMINAMATH_CALUDE_modular_equivalence_in_range_l3359_335938


namespace NUMINAMATH_CALUDE_sum_of_xyz_l3359_335996

theorem sum_of_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x * y = 40) (h2 : x * z = 80) (h3 : y * z = 160) :
  x + y + z = 14 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_xyz_l3359_335996


namespace NUMINAMATH_CALUDE_probability_one_unit_apart_l3359_335943

/-- The number of points around the square -/
def num_points : ℕ := 12

/-- The number of pairs of points that are one unit apart -/
def favorable_pairs : ℕ := 12

/-- The total number of ways to choose two points from num_points -/
def total_pairs : ℕ := num_points.choose 2

/-- The probability of choosing two points one unit apart -/
def probability : ℚ := favorable_pairs / total_pairs

theorem probability_one_unit_apart : probability = 2 / 11 := by sorry

end NUMINAMATH_CALUDE_probability_one_unit_apart_l3359_335943


namespace NUMINAMATH_CALUDE_ginger_flower_sales_l3359_335966

/-- Represents the number of flowers sold of each type -/
structure FlowerSales where
  lilacs : ℕ
  roses : ℕ
  gardenias : ℕ

/-- Calculates the total number of flowers sold -/
def totalFlowers (sales : FlowerSales) : ℕ :=
  sales.lilacs + sales.roses + sales.gardenias

/-- Theorem: Given the conditions of Ginger's flower sales, the total number of flowers sold is 45 -/
theorem ginger_flower_sales :
  ∀ (sales : FlowerSales),
    sales.lilacs = 10 →
    sales.roses = 3 * sales.lilacs →
    sales.gardenias = sales.lilacs / 2 →
    totalFlowers sales = 45 := by
  sorry


end NUMINAMATH_CALUDE_ginger_flower_sales_l3359_335966


namespace NUMINAMATH_CALUDE_meaningful_range_l3359_335910

def is_meaningful (x : ℝ) : Prop :=
  x + 1 ≥ 0 ∧ x ≠ 0

theorem meaningful_range :
  ∀ x : ℝ, is_meaningful x ↔ x ≥ -1 ∧ x ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_meaningful_range_l3359_335910


namespace NUMINAMATH_CALUDE_monotonicity_intervals_two_zeros_condition_l3359_335939

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 + ((1-a)/2) * x^2 - a*x - a

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := x^2 + (1-a)*x - a

-- Theorem for monotonicity intervals
theorem monotonicity_intervals (a : ℝ) (h : a > 0) :
  (∀ x < -1, (f' a x > 0)) ∧
  (∀ x ∈ Set.Ioo (-1) a, (f' a x < 0)) ∧
  (∀ x > a, (f' a x > 0)) :=
sorry

-- Theorem for the range of a when f has exactly two zeros in (-2, 0)
theorem two_zeros_condition (a : ℝ) :
  (∃ x y : ℝ, x ∈ Set.Ioo (-2) 0 ∧ y ∈ Set.Ioo (-2) 0 ∧ x ≠ y ∧ f a x = 0 ∧ f a y = 0 ∧
   ∀ z ∈ Set.Ioo (-2) 0, f a z = 0 → (z = x ∨ z = y)) ↔
  (a > 0 ∧ a < 1/3) :=
sorry

end

end NUMINAMATH_CALUDE_monotonicity_intervals_two_zeros_condition_l3359_335939


namespace NUMINAMATH_CALUDE_complex_number_coordinates_l3359_335931

theorem complex_number_coordinates (a : ℝ) : 
  let z₁ : ℂ := a + Complex.I
  let z₂ : ℂ := 1 - Complex.I
  (∃ b : ℝ, z₁ / z₂ = Complex.I * b) → z₁ = 1 + Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_number_coordinates_l3359_335931


namespace NUMINAMATH_CALUDE_eighth_minus_seventh_difference_l3359_335982

/-- The number of tiles in the nth square of the sequence -/
def tiles (n : ℕ) : ℕ := n^2 + 2*n

/-- The difference in tiles between the 8th and 7th squares -/
def tile_difference : ℕ := tiles 8 - tiles 7

theorem eighth_minus_seventh_difference :
  tile_difference = 17 := by sorry

end NUMINAMATH_CALUDE_eighth_minus_seventh_difference_l3359_335982


namespace NUMINAMATH_CALUDE_negation_of_existence_original_proposition_negation_l3359_335909

theorem negation_of_existence (p : ℝ → Prop) :
  (¬ ∃ x > 1, p x) ↔ ∀ x > 1, ¬ p x := by sorry

theorem original_proposition_negation :
  (¬ ∃ x > 1, 3*x + 1 > 5) ↔ (∀ x > 1, 3*x + 1 ≤ 5) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_original_proposition_negation_l3359_335909


namespace NUMINAMATH_CALUDE_weight_of_replaced_person_l3359_335953

theorem weight_of_replaced_person
  (n : ℕ)
  (average_increase : ℝ)
  (new_person_weight : ℝ)
  (h1 : n = 10)
  (h2 : average_increase = 2.5)
  (h3 : new_person_weight = 90)
  : ∃ (replaced_weight : ℝ),
    replaced_weight = new_person_weight - n * average_increase :=
by
  sorry

end NUMINAMATH_CALUDE_weight_of_replaced_person_l3359_335953


namespace NUMINAMATH_CALUDE_subcommittee_count_l3359_335949

theorem subcommittee_count (n k : ℕ) (hn : n = 8) (hk : k = 3) : 
  Nat.choose n k = 56 := by
  sorry

end NUMINAMATH_CALUDE_subcommittee_count_l3359_335949


namespace NUMINAMATH_CALUDE_kyuhyung_cards_l3359_335942

/-- The number of cards in Kyuhyung's possession -/
def total_cards : ℕ := 103

/-- The side length of the square arrangement -/
def side_length : ℕ := 10

/-- The number of cards left over after forming the square -/
def leftover_cards : ℕ := 3

/-- The number of additional cards needed to fill the outer perimeter -/
def perimeter_cards : ℕ := 44

theorem kyuhyung_cards :
  total_cards = side_length^2 + leftover_cards ∧
  (side_length + 2)^2 - side_length^2 = perimeter_cards :=
by sorry

end NUMINAMATH_CALUDE_kyuhyung_cards_l3359_335942


namespace NUMINAMATH_CALUDE_triangle_similarity_problem_l3359_335934

theorem triangle_similarity_problem (DC CB AD : ℝ) (h1 : DC = 9) (h2 : CB = 10) 
  (h3 : AD > 0) (h4 : (1 : ℝ) / 3 * AD = (DC + CB) * 2 / 3) (h5 : (3 : ℝ) / 4 * AD = 21.375) : 
  ∃ FC : ℝ, FC = 14.625 := by
  sorry

end NUMINAMATH_CALUDE_triangle_similarity_problem_l3359_335934


namespace NUMINAMATH_CALUDE_orange_stack_problem_l3359_335950

/-- Calculates the number of oranges in a pyramid-like stack --/
def orangeStackSum (base_width : ℕ) (base_length : ℕ) : ℕ :=
  let layers := min base_width base_length
  let layerSum (n : ℕ) : ℕ := (base_width - n + 1) * (base_length - n + 1)
  (List.range layers).map layerSum |>.sum

/-- The pyramid-like stack of oranges problem --/
theorem orange_stack_problem :
  orangeStackSum 5 8 = 100 := by
  sorry

end NUMINAMATH_CALUDE_orange_stack_problem_l3359_335950


namespace NUMINAMATH_CALUDE_same_terminal_side_angle_with_same_terminal_side_l3359_335901

theorem same_terminal_side (x y : Real) : 
  x = y + 2 * Real.pi * ↑(Int.floor ((x - y) / (2 * Real.pi))) → 
  ∃ k : ℤ, y = x + 2 * Real.pi * k := by
  sorry

theorem angle_with_same_terminal_side : 
  ∃ k : ℤ, -π/3 = 5*π/3 + 2*π*k := by
  sorry

end NUMINAMATH_CALUDE_same_terminal_side_angle_with_same_terminal_side_l3359_335901


namespace NUMINAMATH_CALUDE_configuration_permutations_l3359_335916

/-- The number of distinct arrangements of the letters in "CONFIGURATION" -/
def configuration_arrangements : ℕ := 389188800

/-- The total number of letters in "CONFIGURATION" -/
def total_letters : ℕ := 13

/-- The number of times each of O, I, N, and U appears in "CONFIGURATION" -/
def repeated_letter_count : ℕ := 2

/-- The number of letters that repeat in "CONFIGURATION" -/
def repeating_letters : ℕ := 4

theorem configuration_permutations :
  configuration_arrangements = (Nat.factorial total_letters) / (Nat.factorial repeated_letter_count ^ repeating_letters) :=
sorry

end NUMINAMATH_CALUDE_configuration_permutations_l3359_335916


namespace NUMINAMATH_CALUDE_inequality_properties_l3359_335915

theorem inequality_properties (a b : ℝ) (h : 1/a < 1/b ∧ 1/b < 0) :
  a^2 < b^2 ∧ a*b < b^2 ∧ a/b + b/a > 2 ∧ |a| + |b| = |a + b| := by
  sorry

end NUMINAMATH_CALUDE_inequality_properties_l3359_335915


namespace NUMINAMATH_CALUDE_chess_tournament_games_l3359_335985

/-- Calculate the number of games in a chess tournament --/
def tournament_games (n : ℕ) : ℕ := n * (n - 1)

/-- The number of players in the tournament --/
def num_players : ℕ := 7

/-- Theorem: In a chess tournament with 7 players, where each player plays twice
    with every other player, the total number of games played is 84. --/
theorem chess_tournament_games :
  2 * tournament_games num_players = 84 := by
  sorry


end NUMINAMATH_CALUDE_chess_tournament_games_l3359_335985


namespace NUMINAMATH_CALUDE_min_value_sum_squares_l3359_335911

theorem min_value_sum_squares (a b c : ℝ) (h : a + 2*b + 3*c = 6) :
  ∃ (min : ℝ), min = 12 ∧ a^2 + 4*b^2 + 9*c^2 ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_squares_l3359_335911


namespace NUMINAMATH_CALUDE_geometric_sequence_max_value_l3359_335960

theorem geometric_sequence_max_value (a b c d : ℝ) : 
  (∃ r : ℝ, a * r = b ∧ b * r = c ∧ c * r = d) →  -- geometric sequence condition
  (∀ x : ℝ, Real.log (x + 2) - x ≤ c) →           -- maximum value condition
  (Real.log (b + 2) - b = c) →                    -- maximum occurs at x = b
  a * d = -1 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_max_value_l3359_335960


namespace NUMINAMATH_CALUDE_number_thought_of_l3359_335997

theorem number_thought_of (x : ℝ) : (x / 4) + 9 = 15 → x = 24 := by
  sorry

end NUMINAMATH_CALUDE_number_thought_of_l3359_335997


namespace NUMINAMATH_CALUDE_square_sum_given_product_and_sum_l3359_335920

theorem square_sum_given_product_and_sum (x y : ℝ) 
  (h1 : x * y = 16) 
  (h2 : x + y = 8) : 
  x^2 + y^2 = 32 := by
sorry

end NUMINAMATH_CALUDE_square_sum_given_product_and_sum_l3359_335920


namespace NUMINAMATH_CALUDE_calculate_expression_l3359_335995

theorem calculate_expression : 3 * 301 + 4 * 301 + 5 * 301 + 300 = 3912 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l3359_335995


namespace NUMINAMATH_CALUDE_pastries_sum_is_147_l3359_335990

/-- The total number of pastries made by Lola, Lulu, and Lila -/
def total_pastries (lola_cupcakes lola_poptarts lola_pies lola_eclairs
                    lulu_cupcakes lulu_poptarts lulu_pies lulu_eclairs
                    lila_cupcakes lila_poptarts lila_pies lila_eclairs : ℕ) : ℕ :=
  lola_cupcakes + lola_poptarts + lola_pies + lola_eclairs +
  lulu_cupcakes + lulu_poptarts + lulu_pies + lulu_eclairs +
  lila_cupcakes + lila_poptarts + lila_pies + lila_eclairs

/-- Theorem stating that the total number of pastries is 147 -/
theorem pastries_sum_is_147 :
  total_pastries 13 10 8 6 16 12 14 9 22 15 10 12 = 147 := by
  sorry

end NUMINAMATH_CALUDE_pastries_sum_is_147_l3359_335990


namespace NUMINAMATH_CALUDE_altered_detergent_theorem_l3359_335917

/-- Represents the ratio of components in a cleaning solution -/
structure CleaningSolution where
  bleach : ℚ
  detergent : ℚ
  water : ℚ

/-- Calculates the amount of detergent in the altered solution -/
def altered_detergent_amount (original : CleaningSolution) (water_amount : ℚ) : ℚ :=
  let original_detergent_water_ratio := original.detergent / original.water
  let new_detergent_water_ratio := original_detergent_water_ratio / 2
  water_amount * new_detergent_water_ratio

/-- Theorem stating the amount of detergent in the altered solution -/
theorem altered_detergent_theorem (original : CleaningSolution) 
    (h1 : original.bleach = 2)
    (h2 : original.detergent = 25)
    (h3 : original.water = 100)
    (h4 : altered_detergent_amount original 300 = 37.5) : 
  altered_detergent_amount original 300 = 37.5 := by
  sorry

#check altered_detergent_theorem

end NUMINAMATH_CALUDE_altered_detergent_theorem_l3359_335917


namespace NUMINAMATH_CALUDE_quadratic_inequality_existence_l3359_335948

theorem quadratic_inequality_existence (m : ℝ) : 
  (∃ x : ℝ, x^2 - m*x + 1 ≤ 0) ↔ (m ≥ 2 ∨ m ≤ -2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_existence_l3359_335948


namespace NUMINAMATH_CALUDE_min_alterations_for_equal_sum_l3359_335913

def initial_matrix : Matrix (Fin 3) (Fin 3) ℕ := !![1,2,3; 4,5,6; 7,8,9]

def row_sum (M : Matrix (Fin 3) (Fin 3) ℕ) (i : Fin 3) : ℕ :=
  M i 0 + M i 1 + M i 2

def col_sum (M : Matrix (Fin 3) (Fin 3) ℕ) (j : Fin 3) : ℕ :=
  M 0 j + M 1 j + M 2 j

def all_sums_different (M : Matrix (Fin 3) (Fin 3) ℕ) : Prop :=
  ∀ i j i' j', i ≠ i' ∨ j ≠ j' → row_sum M i ≠ row_sum M j ∧ col_sum M i ≠ col_sum M j'

theorem min_alterations_for_equal_sum :
  all_sums_different initial_matrix ∧
  (∃ M : Matrix (Fin 3) (Fin 3) ℕ, ∃ i j : Fin 3,
    (∀ x y, (M x y ≠ initial_matrix x y) → (x = i ∧ y = j)) ∧
    (∃ r c, row_sum M r = col_sum M c)) ∧
  ¬(∃ r c, row_sum initial_matrix r = col_sum initial_matrix c) :=
by sorry

end NUMINAMATH_CALUDE_min_alterations_for_equal_sum_l3359_335913


namespace NUMINAMATH_CALUDE_gcd_180_270_l3359_335936

theorem gcd_180_270 : Nat.gcd 180 270 = 90 := by
  sorry

end NUMINAMATH_CALUDE_gcd_180_270_l3359_335936


namespace NUMINAMATH_CALUDE_purely_imaginary_solution_l3359_335967

theorem purely_imaginary_solution (z : ℂ) :
  (∃ b : ℝ, z = Complex.I * b) →
  (∃ c : ℝ, (z - 2)^2 - Complex.I * 8 = Complex.I * c) →
  z = Complex.I * 2 := by
sorry

end NUMINAMATH_CALUDE_purely_imaginary_solution_l3359_335967


namespace NUMINAMATH_CALUDE_roots_quadratic_equation_l3359_335914

theorem roots_quadratic_equation (a b : ℝ) : 
  (a^2 + a - 2014 = 0) → 
  (b^2 + b - 2014 = 0) → 
  a^2 + 2*a + b = 2013 := by
sorry

end NUMINAMATH_CALUDE_roots_quadratic_equation_l3359_335914


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3359_335935

theorem polynomial_simplification (x : ℝ) : 
  3 - 5*x - 7*x^2 + 9 + 11*x - 13*x^2 - 15 + 17*x + 19*x^2 + 6*x^3 = 
  6*x^3 - x^2 + 23*x - 3 := by
sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3359_335935


namespace NUMINAMATH_CALUDE_symmetric_point_of_P_l3359_335905

-- Define a point in 2D Cartesian coordinate system
def Point := ℝ × ℝ

-- Define the origin
def origin : Point := (0, 0)

-- Define the given point P
def P : Point := (-1, 2)

-- Define symmetry with respect to the origin
def symmetricPoint (p : Point) : Point :=
  (-p.1, -p.2)

-- Theorem statement
theorem symmetric_point_of_P :
  symmetricPoint P = (1, -2) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_of_P_l3359_335905


namespace NUMINAMATH_CALUDE_alyssa_plums_count_l3359_335955

/-- The number of plums picked by Jason -/
def jason_plums : ℕ := 10

/-- The total number of plums picked -/
def total_plums : ℕ := 27

/-- The number of plums picked by Alyssa -/
def alyssa_plums : ℕ := total_plums - jason_plums

theorem alyssa_plums_count : alyssa_plums = 17 := by
  sorry

end NUMINAMATH_CALUDE_alyssa_plums_count_l3359_335955


namespace NUMINAMATH_CALUDE_knicks_knacks_knocks_conversion_l3359_335961

/-- Given the conversion rates between knicks, knacks, and knocks, 
    prove that 30 knocks are equal to 20 knicks. -/
theorem knicks_knacks_knocks_conversion :
  ∀ (knicks knacks knocks : ℚ),
    (5 * knicks = 3 * knacks) →
    (2 * knacks = 5 * knocks) →
    (30 * knocks = 20 * knicks) :=
by
  sorry

end NUMINAMATH_CALUDE_knicks_knacks_knocks_conversion_l3359_335961


namespace NUMINAMATH_CALUDE_tower_of_hanoi_l3359_335933

/-- Minimal number of moves required to solve the Tower of Hanoi problem with n discs -/
def hanoi_moves (n : ℕ) : ℕ :=
  2^n - 1

/-- The Tower of Hanoi theorem -/
theorem tower_of_hanoi (n : ℕ) : 
  hanoi_moves n = 2^n - 1 := by
  sorry

#eval hanoi_moves 64

end NUMINAMATH_CALUDE_tower_of_hanoi_l3359_335933


namespace NUMINAMATH_CALUDE_cauchy_not_dense_implies_linear_l3359_335989

/-- A function satisfying the Cauchy functional equation -/
def CauchyFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f x + f y

/-- The graph of a function is not dense in the plane -/
def NotDenseGraph (f : ℝ → ℝ) : Prop :=
  ∃ U : Set (ℝ × ℝ), IsOpen U ∧ U.Nonempty ∧ ∀ x : ℝ, (x, f x) ∉ U

/-- A function is linear -/
def LinearFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, ∀ x : ℝ, f x = a * x + b

theorem cauchy_not_dense_implies_linear (f : ℝ → ℝ) 
  (h_cauchy : CauchyFunction f) (h_not_dense : NotDenseGraph f) : 
  LinearFunction f := by
  sorry

end NUMINAMATH_CALUDE_cauchy_not_dense_implies_linear_l3359_335989


namespace NUMINAMATH_CALUDE_chicken_rabbit_problem_l3359_335969

theorem chicken_rabbit_problem :
  ∀ (chickens rabbits : ℕ),
    chickens + rabbits = 15 →
    2 * chickens + 4 * rabbits = 40 →
    chickens = 10 ∧ rabbits = 5 := by
  sorry

end NUMINAMATH_CALUDE_chicken_rabbit_problem_l3359_335969


namespace NUMINAMATH_CALUDE_acorns_given_calculation_l3359_335902

/-- The number of acorns Megan gave to her sister -/
def acorns_given : ℕ := sorry

/-- The initial number of acorns Megan had -/
def initial_acorns : ℕ := 16

/-- The number of acorns Megan has left -/
def acorns_left : ℕ := 9

/-- Theorem stating that the number of acorns given is the difference between
    the initial number and the number left -/
theorem acorns_given_calculation : acorns_given = initial_acorns - acorns_left := by
  sorry

end NUMINAMATH_CALUDE_acorns_given_calculation_l3359_335902


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3359_335929

theorem quadratic_inequality_solution_set 
  (α β a b c : ℝ) 
  (h1 : 0 < α) (h2 : α < β)
  (h3 : ∀ x, a * x^2 + b * x + c > 0 ↔ α < x ∧ x < β) :
  ∀ x, (a + c - b) * x^2 + (b - 2*a) * x + a > 0 ↔ 
    1 / (1 + β) < x ∧ x < 1 / (1 + α) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3359_335929


namespace NUMINAMATH_CALUDE_min_value_is_neg_one_l3359_335979

/-- The system of equations and inequalities -/
def system (x y : ℝ) : Prop :=
  3^(-x) * y^4 - 2*y^2 + 3^x ≤ 0 ∧ 27^x + y^4 - 3^x - 1 = 0

/-- The expression to be minimized -/
def expression (x y : ℝ) : ℝ := x^3 + y^3

/-- The theorem stating that the minimum value of the expression is -1 -/
theorem min_value_is_neg_one :
  ∃ (x y : ℝ), system x y ∧
  ∀ (a b : ℝ), system a b → expression x y ≤ expression a b ∧
  expression x y = -1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_is_neg_one_l3359_335979


namespace NUMINAMATH_CALUDE_stone_price_calculation_l3359_335968

/-- The price per stone when selling a collection of precious stones -/
def price_per_stone (total_amount : ℕ) (num_stones : ℕ) : ℚ :=
  (total_amount : ℚ) / (num_stones : ℚ)

/-- Theorem stating that the price per stone is $1785 when 8 stones are sold for $14280 -/
theorem stone_price_calculation :
  price_per_stone 14280 8 = 1785 := by
  sorry

end NUMINAMATH_CALUDE_stone_price_calculation_l3359_335968


namespace NUMINAMATH_CALUDE_tv_watching_time_equivalence_l3359_335988

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- The number of hours Ava watched television -/
def hours_watched : ℕ := 4

/-- The theorem stating that watching TV for 4 hours is equivalent to 240 minutes -/
theorem tv_watching_time_equivalence : 
  hours_watched * minutes_per_hour = 240 := by
  sorry

end NUMINAMATH_CALUDE_tv_watching_time_equivalence_l3359_335988


namespace NUMINAMATH_CALUDE_complement_of_union_l3359_335977

def U : Set Nat := {1,2,3,4,5}
def M : Set Nat := {1,3}
def N : Set Nat := {1,2}

theorem complement_of_union (U M N : Set Nat) : 
  U \ (M ∪ N) = {4,5} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_union_l3359_335977


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l3359_335908

def U : Set Int := {-2, -1, 0, 1, 2}
def A : Set Int := {-2, -1, 1, 2}

theorem complement_of_A_in_U : 
  {x : Int | x ∈ U ∧ x ∉ A} = {0} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l3359_335908


namespace NUMINAMATH_CALUDE_quadratic_sum_l3359_335963

theorem quadratic_sum (a b : ℝ) : 
  a^2 - 2*a + 8 = 24 →
  b^2 - 2*b + 8 = 24 →
  a ≥ b →
  3*a + 2*b = 5 + Real.sqrt 17 := by
sorry

end NUMINAMATH_CALUDE_quadratic_sum_l3359_335963


namespace NUMINAMATH_CALUDE_budget_calculation_l3359_335986

/-- The original budget in Euros -/
def original_budget : ℝ := sorry

/-- The amount left after spending -/
def amount_left : ℝ := 13500

/-- The fraction of budget spent on clothes -/
def clothes_fraction : ℝ := 0.25

/-- The discount on clothes -/
def clothes_discount : ℝ := 0.1

/-- The fraction of budget spent on groceries -/
def groceries_fraction : ℝ := 0.15

/-- The sales tax on groceries -/
def groceries_tax : ℝ := 0.05

/-- The fraction of budget spent on electronics -/
def electronics_fraction : ℝ := 0.1

/-- The exchange rate for electronics (EUR to USD) -/
def exchange_rate : ℝ := 1.2

/-- The fraction of budget spent on dining -/
def dining_fraction : ℝ := 0.05

/-- The service charge on dining -/
def dining_service_charge : ℝ := 0.12

theorem budget_calculation :
  amount_left = original_budget * (1 - (
    clothes_fraction * (1 - clothes_discount) +
    groceries_fraction * (1 + groceries_tax) +
    electronics_fraction * exchange_rate +
    dining_fraction * (1 + dining_service_charge)
  )) := by sorry

end NUMINAMATH_CALUDE_budget_calculation_l3359_335986


namespace NUMINAMATH_CALUDE_unique_coin_combination_l3359_335975

/-- Represents a coin with a value in kopecks -/
structure Coin where
  value : ℕ

/-- Represents a wallet containing two coins -/
structure Wallet where
  coin1 : Coin
  coin2 : Coin

/-- The total value of coins in a wallet -/
def walletValue (w : Wallet) : ℕ := w.coin1.value + w.coin2.value

/-- Predicate to check if a coin is not a five-kopeck coin -/
def isNotFiveKopecks (c : Coin) : Prop := c.value ≠ 5

/-- Theorem stating the only possible combination of coins -/
theorem unique_coin_combination (w : Wallet) 
  (h1 : walletValue w = 15)
  (h2 : isNotFiveKopecks w.coin1 ∨ isNotFiveKopecks w.coin2) :
  (w.coin1.value = 5 ∧ w.coin2.value = 10) ∨ (w.coin1.value = 10 ∧ w.coin2.value = 5) :=
sorry

end NUMINAMATH_CALUDE_unique_coin_combination_l3359_335975


namespace NUMINAMATH_CALUDE_simplify_expression_l3359_335980

theorem simplify_expression : (18 * (10^10)) / (6 * (10^4)) * 2 = 6 * (10^6) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3359_335980


namespace NUMINAMATH_CALUDE_max_value_2x_minus_y_l3359_335959

theorem max_value_2x_minus_y (x y : ℝ) 
  (h1 : x - y + 1 ≥ 0)
  (h2 : y + 1 ≥ 0)
  (h3 : x + y + 1 ≤ 0) :
  ∃ (max : ℝ), max = 1 ∧ ∀ x' y' : ℝ, 
    x' - y' + 1 ≥ 0 → y' + 1 ≥ 0 → x' + y' + 1 ≤ 0 → 
    2 * x' - y' ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_2x_minus_y_l3359_335959


namespace NUMINAMATH_CALUDE_loaves_sold_l3359_335973

/-- The number of loaves sold in a supermarket given initial, delivered, and final counts. -/
theorem loaves_sold (initial : ℕ) (delivered : ℕ) (final : ℕ) :
  initial = 2355 →
  delivered = 489 →
  final = 2215 →
  initial + delivered - final = 629 := by
  sorry

#check loaves_sold

end NUMINAMATH_CALUDE_loaves_sold_l3359_335973


namespace NUMINAMATH_CALUDE_days_before_reinforcement_l3359_335957

/-- 
Given a garrison with initial provisions and a reinforcement, 
calculate the number of days that passed before the reinforcement arrived.
-/
theorem days_before_reinforcement 
  (initial_garrison : ℕ) 
  (initial_provisions : ℕ) 
  (reinforcement : ℕ) 
  (remaining_provisions : ℕ) 
  (h1 : initial_garrison = 150)
  (h2 : initial_provisions = 31)
  (h3 : reinforcement = 300)
  (h4 : remaining_provisions = 5) : 
  ∃ (x : ℕ), x = 16 ∧ 
    initial_garrison * (initial_provisions - x) = 
    (initial_garrison + reinforcement) * remaining_provisions :=
by sorry

end NUMINAMATH_CALUDE_days_before_reinforcement_l3359_335957


namespace NUMINAMATH_CALUDE_wire_ratio_proof_l3359_335983

/-- Given a wire of length 70 cm cut into two pieces, where the shorter piece is 27.999999999999993 cm long,
    prove that the ratio of the shorter piece to the longer piece is 2:3. -/
theorem wire_ratio_proof (total_length : ℝ) (shorter_piece : ℝ) (longer_piece : ℝ) :
  total_length = 70 →
  shorter_piece = 27.999999999999993 →
  longer_piece = total_length - shorter_piece →
  (shorter_piece / longer_piece) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_wire_ratio_proof_l3359_335983


namespace NUMINAMATH_CALUDE_new_ratio_second_term_l3359_335971

def original_ratio : Rat × Rat := (4, 15)
def number_to_add : ℕ := 29

theorem new_ratio_second_term :
  let new_ratio := (original_ratio.1 + number_to_add, original_ratio.2 + number_to_add)
  new_ratio.2 = 44 := by sorry

end NUMINAMATH_CALUDE_new_ratio_second_term_l3359_335971


namespace NUMINAMATH_CALUDE_rice_bags_weight_analysis_l3359_335981

def standard_weight : ℝ := 50
def num_bags : ℕ := 10
def weight_deviations : List ℝ := [0.5, 0.3, 0, -0.2, -0.3, 1.1, -0.7, -0.2, 0.6, 0.7]

theorem rice_bags_weight_analysis :
  let total_deviation : ℝ := weight_deviations.sum
  let total_weight : ℝ := (standard_weight * num_bags) + total_deviation
  let average_weight : ℝ := total_weight / num_bags
  (total_deviation = 1.7) ∧ 
  (total_weight = 501.7) ∧ 
  (average_weight = 50.17) := by
sorry

end NUMINAMATH_CALUDE_rice_bags_weight_analysis_l3359_335981


namespace NUMINAMATH_CALUDE_soldier_target_practice_l3359_335907

theorem soldier_target_practice (total_shots : ℕ) (total_score : ℕ) (tens : ℕ) (tens_score : ℕ) :
  total_shots = 10 →
  total_score = 90 →
  tens = 4 →
  tens_score = 10 →
  ∃ (sevens eights nines : ℕ),
    sevens + eights + nines = total_shots - tens ∧
    7 * sevens + 8 * eights + 9 * nines = total_score - tens * tens_score ∧
    sevens = 1 :=
by sorry

end NUMINAMATH_CALUDE_soldier_target_practice_l3359_335907


namespace NUMINAMATH_CALUDE_g_of_2_eq_5_l3359_335962

def g (x : ℝ) : ℝ := x^3 - x^2 + 1

theorem g_of_2_eq_5 : g 2 = 5 := by sorry

end NUMINAMATH_CALUDE_g_of_2_eq_5_l3359_335962


namespace NUMINAMATH_CALUDE_expression_value_l3359_335947

theorem expression_value (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (sum_zero : a + b + c = 0) (sum_squares_nonzero : a^2 + b^2 + c^2 ≠ 0) :
  (a^7 + b^7 + c^7) / (a * b * c * (a^2 + b^2 + c^2)) = -7 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3359_335947


namespace NUMINAMATH_CALUDE_max_expression_c_value_l3359_335900

theorem max_expression_c_value (a b c : ℕ) : 
  a ∈ ({1, 2, 4} : Set ℕ) →
  b ∈ ({1, 2, 4} : Set ℕ) →
  c ∈ ({1, 2, 4} : Set ℕ) →
  a ≠ b → b ≠ c → a ≠ c →
  (∀ x y z : ℕ, x ∈ ({1, 2, 4} : Set ℕ) → y ∈ ({1, 2, 4} : Set ℕ) → z ∈ ({1, 2, 4} : Set ℕ) →
    x ≠ y → y ≠ z → x ≠ z → (x / 2) / (y / z : ℚ) ≤ (a / 2) / (b / c : ℚ)) →
  (a / 2) / (b / c : ℚ) = 4 →
  c = 2 := by sorry

end NUMINAMATH_CALUDE_max_expression_c_value_l3359_335900


namespace NUMINAMATH_CALUDE_units_digit_is_nine_l3359_335964

/-- The product of digits of a two-digit number -/
def P (n : ℕ) : ℕ := (n / 10) * (n % 10)

/-- The sum of digits of a two-digit number -/
def S (n : ℕ) : ℕ := (n / 10) + (n % 10)

/-- A two-digit number is between 10 and 99, inclusive -/
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

theorem units_digit_is_nine (N : ℕ) (h1 : is_two_digit N) (h2 : N = P N + S N) :
  N % 10 = 9 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_is_nine_l3359_335964


namespace NUMINAMATH_CALUDE_sphere_tangency_relation_l3359_335954

/-- Given three mutually tangent spheres touching a plane at three points on a circle of radius R,
    and two spheres of radii r and ρ (ρ > r) each tangent to the three given spheres and the plane,
    prove that 1/r - 1/ρ = 2√3/R. -/
theorem sphere_tangency_relation (R r ρ : ℝ) (h1 : r > 0) (h2 : ρ > 0) (h3 : ρ > r) :
  1 / r - 1 / ρ = 2 * Real.sqrt 3 / R :=
by sorry

end NUMINAMATH_CALUDE_sphere_tangency_relation_l3359_335954


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l3359_335956

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def A : Set Nat := {1, 2}
def B : Set Nat := {2, 3, 4}

theorem intersection_A_complement_B : A ∩ (U \ B) = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l3359_335956


namespace NUMINAMATH_CALUDE_barnyard_owls_count_l3359_335904

/-- The number of hoot sounds one barnyard owl makes per minute. -/
def hoots_per_owl : ℕ := 5

/-- The total number of hoots heard per minute. -/
def total_hoots : ℕ := 20 - 5

/-- The number of barnyard owls making the noise. -/
def num_owls : ℕ := total_hoots / hoots_per_owl

theorem barnyard_owls_count : num_owls = 3 := by
  sorry

end NUMINAMATH_CALUDE_barnyard_owls_count_l3359_335904


namespace NUMINAMATH_CALUDE_quadratic_with_prime_roots_l3359_335924

theorem quadratic_with_prime_roots (m : ℕ) : 
  (∃ x y : ℕ, x.Prime ∧ y.Prime ∧ x ≠ y ∧ x^2 - 1999*x + m = 0 ∧ y^2 - 1999*y + m = 0) → 
  m = 3994 := by
sorry

end NUMINAMATH_CALUDE_quadratic_with_prime_roots_l3359_335924


namespace NUMINAMATH_CALUDE_solution_set_implies_sum_l3359_335944

-- Define the inequality solution set
def SolutionSet (a b : ℝ) : Set ℝ :=
  {x | 1 < x ∧ x < 2}

-- State the theorem
theorem solution_set_implies_sum (a b : ℝ) :
  SolutionSet a b = {x | (x - a) * (x - b) < 0} →
  a + b = 3 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_implies_sum_l3359_335944


namespace NUMINAMATH_CALUDE_sin_300_degrees_l3359_335987

theorem sin_300_degrees : Real.sin (300 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_300_degrees_l3359_335987


namespace NUMINAMATH_CALUDE_triangle_area_l3359_335978

-- Define the point P
def P : ℝ × ℝ := (2, 5)

-- Define the slopes of the two lines
def slope1 : ℝ := -1
def slope2 : ℝ := 1.5

-- Define Q and R as the x-intercepts of the lines
def Q : ℝ × ℝ := (-3, 0)
def R : ℝ × ℝ := (5.33, 0)

-- Theorem statement
theorem triangle_area : 
  let triangle_area := (1/2) * (R.1 - Q.1) * P.2
  triangle_area = 20.825 := by sorry

end NUMINAMATH_CALUDE_triangle_area_l3359_335978


namespace NUMINAMATH_CALUDE_angle_BED_is_120_l3359_335919

/-- A point in a 2D plane -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A polygon defined by its vertices -/
structure Polygon :=
  (vertices : List Point)

/-- Checks if a polygon is a square -/
def is_square (p : Polygon) : Prop :=
  sorry

/-- Checks if a polygon is an equilateral triangle -/
def is_equilateral_triangle (p : Polygon) : Prop :=
  sorry

/-- Checks if a point is outside a polygon -/
def is_outside (pt : Point) (p : Polygon) : Prop :=
  sorry

/-- Calculates the angle between three points -/
def angle (p1 p2 p3 : Point) : ℝ :=
  sorry

theorem angle_BED_is_120 (A B C D E : Point) :
  let ABCD := Polygon.mk [A, B, C, D]
  let ABE := Polygon.mk [A, B, E]
  let ECD := Polygon.mk [E, C, D]
  is_square ABCD →
  is_equilateral_triangle ABE →
  is_equilateral_triangle ECD →
  is_outside E ABCD →
  C ∈ ABCD.vertices →
  C ∈ ECD.vertices →
  angle B E D = 120 :=
sorry

end NUMINAMATH_CALUDE_angle_BED_is_120_l3359_335919


namespace NUMINAMATH_CALUDE_sufficient_condition_range_l3359_335998

theorem sufficient_condition_range (m : ℝ) : 
  (∀ x : ℝ, |x - 4| ≤ 6 → x ≤ 1 + m) ∧ 
  (∃ x : ℝ, x ≤ 1 + m ∧ |x - 4| > 6) → 
  m ≥ 9 := by sorry

end NUMINAMATH_CALUDE_sufficient_condition_range_l3359_335998


namespace NUMINAMATH_CALUDE_perpendicular_line_to_plane_l3359_335940

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the intersection operation for planes
variable (intersect : Plane → Plane → Line)

-- Define the perpendicular relation between planes and between lines and planes
variable (perp : Plane → Plane → Prop)
variable (perpL : Line → Plane → Prop)

-- Define the planes and lines
variable (α β γ : Plane) (m n : Line)

-- State the theorem
theorem perpendicular_line_to_plane
  (h1 : intersect α γ = m)
  (h2 : perp β α)
  (h3 : perp β γ)
  (h4 : perpL n α)
  (h5 : perpL n β)
  (h6 : perpL m α) :
  perpL m β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_to_plane_l3359_335940


namespace NUMINAMATH_CALUDE_average_of_abc_l3359_335918

theorem average_of_abc (a b c : ℝ) : 
  (4 + 6 + 9 + a + b + c) / 6 = 18 → (a + b + c) / 3 = 29 + 2/3 := by
  sorry

end NUMINAMATH_CALUDE_average_of_abc_l3359_335918


namespace NUMINAMATH_CALUDE_logarithm_sum_equals_two_l3359_335976

theorem logarithm_sum_equals_two : 2 * Real.log 10 / Real.log 5 + Real.log 0.25 / Real.log 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_sum_equals_two_l3359_335976


namespace NUMINAMATH_CALUDE_smallest_n_for_sqrt_63n_l3359_335993

theorem smallest_n_for_sqrt_63n (n : ℕ) : n > 0 ∧ ∃ k : ℕ, k > 0 ∧ k^2 = 63 * n → n ≥ 7 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_sqrt_63n_l3359_335993


namespace NUMINAMATH_CALUDE_parallel_implies_a_values_l_passes_through_point_l3359_335984

-- Define the lines l and n
def l (a x y : ℝ) : Prop := (a + 2) * x + a * y - 2 = 0
def n (a x y : ℝ) : Prop := (a - 2) * x + 3 * y - 6 = 0

-- Define parallel lines
def parallel (f g : ℝ → ℝ → ℝ → Prop) : Prop :=
  ∀ a, (∃ k ≠ 0, ∀ x y, f a x y ↔ g a (k * x) (k * y))

-- Theorem 1: If l is parallel to n, then a = 6 or a = -1
theorem parallel_implies_a_values :
  parallel l n → ∀ a, (a = 6 ∨ a = -1) :=
sorry

-- Theorem 2: Line l always passes through the point (1, -1)
theorem l_passes_through_point :
  ∀ a, l a 1 (-1) :=
sorry

end NUMINAMATH_CALUDE_parallel_implies_a_values_l_passes_through_point_l3359_335984


namespace NUMINAMATH_CALUDE_shortest_side_of_right_triangle_l3359_335965

/-- Given a right triangle with sides of length 6 and 8, 
    the length of the third side is 2√7 -/
theorem shortest_side_of_right_triangle : ∃ (a b c : ℝ), 
  a = 6 ∧ b = 8 ∧ c = 2 * Real.sqrt 7 ∧ 
  a^2 + c^2 = b^2 ∧ 
  ∀ (x : ℝ), (x^2 + a^2 = b^2 → x ≥ c) :=
by sorry

end NUMINAMATH_CALUDE_shortest_side_of_right_triangle_l3359_335965


namespace NUMINAMATH_CALUDE_sum_simplification_l3359_335937

theorem sum_simplification : -1^2022 + (-1)^2023 + 1^2024 - 1^2025 = -2 := by
  sorry

end NUMINAMATH_CALUDE_sum_simplification_l3359_335937


namespace NUMINAMATH_CALUDE_min_value_function_l3359_335999

theorem min_value_function (x y : ℝ) (hx : x > 1) (hy : y > 1) :
  (x^2 + y) / (y^2 - 1) + (y^2 + x) / (x^2 - 1) ≥ 8/3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_function_l3359_335999


namespace NUMINAMATH_CALUDE_marble_count_theorem_l3359_335970

theorem marble_count_theorem (g y : ℚ) :
  (g - 3) / (g + y - 3) = 1 / 6 →
  g / (g + y - 4) = 1 / 4 →
  g + y = 18 :=
by sorry

end NUMINAMATH_CALUDE_marble_count_theorem_l3359_335970


namespace NUMINAMATH_CALUDE_quadratic_properties_l3359_335945

/-- A quadratic function passing through specific points -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_properties (a b c : ℝ) (h : a ≠ 0) :
  let f := QuadraticFunction a b c
  (f (-3) = 15) ∧ (f (-1) = 3) ∧ (f 0 = 0) ∧ (f 1 = -1) ∧ (f 2 = 0) ∧ (f 4 = 8) →
  (∀ x, f (1 + x) = f (1 - x)) ∧  -- Axis of symmetry at x = 1
  (f (-2) = 8) ∧ (f 3 = 3) ∧      -- Values at x = -2 and x = 3
  (f 0 = 0) ∧ (f 2 = 0)           -- Roots at x = 0 and x = 2
  := by sorry

end NUMINAMATH_CALUDE_quadratic_properties_l3359_335945


namespace NUMINAMATH_CALUDE_inequality_proof_l3359_335925

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (a / (1 + a * b))^2 + (b / (1 + b * c))^2 + (c / (1 + c * a))^2 ≥ 3/4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3359_335925


namespace NUMINAMATH_CALUDE_marcy_makeup_count_l3359_335992

/-- The number of people Marcy can paint with one tube of lip gloss -/
def people_per_tube : ℕ := 3

/-- The number of tubs of lip gloss Marcy brings -/
def tubs : ℕ := 6

/-- The number of tubes of lip gloss in each tub -/
def tubes_per_tub : ℕ := 2

/-- The total number of people Marcy is painting with makeup -/
def total_people : ℕ := tubs * tubes_per_tub * people_per_tube

theorem marcy_makeup_count : total_people = 36 := by
  sorry

end NUMINAMATH_CALUDE_marcy_makeup_count_l3359_335992


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l3359_335906

def U : Set Int := {0, -1, -2, -3, -4}
def M : Set Int := {0, -1, -2}
def N : Set Int := {0, -3, -4}

theorem complement_intersection_theorem :
  ((U \ M) ∩ N) = {-3, -4} := by
  sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l3359_335906


namespace NUMINAMATH_CALUDE_difference_of_squares_multiplication_l3359_335974

theorem difference_of_squares_multiplication (a b : ℕ) :
  58 * 42 = 2352 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_multiplication_l3359_335974


namespace NUMINAMATH_CALUDE_square_area_ratio_l3359_335928

theorem square_area_ratio (s₂ : ℝ) (h : s₂ > 0) : 
  let s₁ := 2 * s₂ * Real.sqrt 2
  (s₁ ^ 2) / (s₂ ^ 2) = 8 := by
sorry

end NUMINAMATH_CALUDE_square_area_ratio_l3359_335928


namespace NUMINAMATH_CALUDE_sqrt_two_subset_P_l3359_335951

def P : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

theorem sqrt_two_subset_P : {Real.sqrt 2} ⊆ P := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_subset_P_l3359_335951


namespace NUMINAMATH_CALUDE_downstream_speed_l3359_335926

/-- The speed of a man rowing downstream, given his upstream speed and still water speed -/
theorem downstream_speed (upstream_speed still_water_speed : ℝ) :
  upstream_speed = 20 →
  still_water_speed = 40 →
  still_water_speed + (still_water_speed - upstream_speed) = 60 :=
by sorry

end NUMINAMATH_CALUDE_downstream_speed_l3359_335926


namespace NUMINAMATH_CALUDE_swing_rope_length_proof_l3359_335958

/-- The length of a swing rope satisfying specific conditions -/
def swing_rope_length : ℝ := 14.5

/-- The initial height of the swing's footboard off the ground -/
def initial_height : ℝ := 1

/-- The distance the swing is pushed forward -/
def push_distance : ℝ := 10

/-- The height of the person -/
def person_height : ℝ := 5

theorem swing_rope_length_proof :
  ∃ (rope_length : ℝ),
    rope_length = swing_rope_length ∧
    rope_length^2 = push_distance^2 + (rope_length - person_height + initial_height)^2 :=
by sorry

end NUMINAMATH_CALUDE_swing_rope_length_proof_l3359_335958


namespace NUMINAMATH_CALUDE_smallest_positive_m_squared_l3359_335922

/-- Definition of circle w₁ -/
def w₁ (x y : ℝ) : Prop := x^2 + y^2 + 10*x - 24*y - 87 = 0

/-- Definition of circle w₂ -/
def w₂ (x y : ℝ) : Prop := x^2 + y^2 - 10*x - 24*y + 153 = 0

/-- Definition of a line y = ax -/
def line (a x y : ℝ) : Prop := y = a * x

/-- Definition of external tangency to w₂ -/
def externally_tangent_w₂ (x y r : ℝ) : Prop :=
  (x - 5)^2 + (y - 12)^2 = (r + 4)^2

/-- Definition of internal tangency to w₁ -/
def internally_tangent_w₁ (x y r : ℝ) : Prop :=
  (x + 5)^2 + (y - 12)^2 = (16 - r)^2

/-- The main theorem -/
theorem smallest_positive_m_squared (m : ℝ) : 
  (∀ a : ℝ, a > 0 → (∃ x y r : ℝ, line a x y ∧ externally_tangent_w₂ x y r ∧ internally_tangent_w₁ x y r) → m ≤ a) ∧
  (∃ x y r : ℝ, line m x y ∧ externally_tangent_w₂ x y r ∧ internally_tangent_w₁ x y r) →
  m^2 = 69/100 :=
sorry

end NUMINAMATH_CALUDE_smallest_positive_m_squared_l3359_335922


namespace NUMINAMATH_CALUDE_point_not_in_transformed_plane_l3359_335994

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Applies a similarity transformation to a plane -/
def transformPlane (p : Plane3D) (k : ℝ) : Plane3D :=
  { a := p.a, b := p.b, c := p.c, d := k * p.d }

/-- Checks if a point satisfies the equation of a plane -/
def pointSatisfiesPlane (point : Point3D) (plane : Plane3D) : Prop :=
  plane.a * point.x + plane.b * point.y + plane.c * point.z + plane.d = 0

/-- The main theorem to be proved -/
theorem point_not_in_transformed_plane :
  let A : Point3D := { x := -1, y := 1, z := -2 }
  let a : Plane3D := { a := 4, b := -1, c := 3, d := -6 }
  let k : ℝ := -5/3
  let transformedPlane := transformPlane a k
  ¬ pointSatisfiesPlane A transformedPlane :=
by
  sorry


end NUMINAMATH_CALUDE_point_not_in_transformed_plane_l3359_335994


namespace NUMINAMATH_CALUDE_sevens_to_hundred_l3359_335923

theorem sevens_to_hundred : ∃ (expr : ℕ), 
  (expr = 100) ∧ 
  (∃ (a b c d e f g h i : ℕ), 
    (a ≤ 7 ∧ b ≤ 7 ∧ c ≤ 7 ∧ d ≤ 7 ∧ e ≤ 7 ∧ f ≤ 7 ∧ g ≤ 7 ∧ h ≤ 7 ∧ i ≤ 7) ∧
    (expr = a * b - c * d + e * f + g + h + i) ∧
    (a + b + c + d + e + f + g + h + i < 10 * 7)) :=
by sorry

end NUMINAMATH_CALUDE_sevens_to_hundred_l3359_335923


namespace NUMINAMATH_CALUDE_gp_sum_ratio_l3359_335941

/-- For a geometric progression with common ratio 3, the ratio of the sum
    of the first 6 terms to the sum of the first 3 terms is 28. -/
theorem gp_sum_ratio (a : ℝ) : 
  let r := 3
  let S₃ := a * (1 - r^3) / (1 - r)
  let S₆ := a * (1 - r^6) / (1 - r)
  S₆ / S₃ = 28 := by
sorry


end NUMINAMATH_CALUDE_gp_sum_ratio_l3359_335941


namespace NUMINAMATH_CALUDE_triangle_side_length_l3359_335903

theorem triangle_side_length (A B C : Real) (a b c : Real) :
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Angles are positive
  A + B + C = π ∧  -- Sum of angles in a triangle
  a > 0 ∧ b > 0 ∧ c > 0 ∧  -- Side lengths are positive
  a = Real.sqrt 3 ∧  -- Given condition
  Real.sin B = 1 / 2 ∧  -- Given condition
  C = π / 6 ∧  -- Given condition
  a / Real.sin A = b / Real.sin B ∧  -- Law of sines
  a / Real.sin A = c / Real.sin C  -- Law of sines
  →
  b = 1 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3359_335903


namespace NUMINAMATH_CALUDE_inequality_proof_l3359_335946

theorem inequality_proof (x₁ x₂ x₃ : ℝ) 
  (h_pos₁ : x₁ > 0) (h_pos₂ : x₂ > 0) (h_pos₃ : x₃ > 0)
  (h_sum : x₁ + x₂ + x₃ = 1) :
  x₂^2 / x₁ + x₃^2 / x₂ + x₁^2 / x₃ ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3359_335946


namespace NUMINAMATH_CALUDE_largest_four_digit_negative_congruent_to_one_mod_23_l3359_335932

theorem largest_four_digit_negative_congruent_to_one_mod_23 :
  ∀ n : ℤ, -9999 ≤ n ∧ n < -999 ∧ n ≡ 1 [ZMOD 23] → n ≤ -1011 :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_negative_congruent_to_one_mod_23_l3359_335932


namespace NUMINAMATH_CALUDE_semicircle_perimeter_l3359_335912

/-- The perimeter of a semicircle with radius 20 is equal to 20π + 40. -/
theorem semicircle_perimeter : 
  ∀ (r : ℝ), r = 20 → (r * π + r) = 20 * π + 40 := by
  sorry

end NUMINAMATH_CALUDE_semicircle_perimeter_l3359_335912


namespace NUMINAMATH_CALUDE_train_passing_time_l3359_335930

/-- The time it takes for two trains to completely pass each other -/
theorem train_passing_time (length_A length_B speed_A speed_B : ℝ) 
  (h1 : length_A = 125)
  (h2 : length_B = 150)
  (h3 : speed_A = 54 * (5/18))
  (h4 : speed_B = 36 * (5/18))
  (h5 : speed_A > 0)
  (h6 : speed_B > 0) :
  (length_A + length_B) / (speed_A + speed_B) = 11 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_time_l3359_335930


namespace NUMINAMATH_CALUDE_min_value_complex_expression_l3359_335972

theorem min_value_complex_expression (z : ℂ) (h : Complex.abs z = 1) :
  Complex.abs (z^2 - 2*z + 3) ≥ 2 * Real.sqrt 6 / 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_complex_expression_l3359_335972


namespace NUMINAMATH_CALUDE_range_of_x_l3359_335991

theorem range_of_x (x : ℝ) : Real.sqrt ((x - 3)^2) = x - 3 → x ≥ 3 := by sorry

end NUMINAMATH_CALUDE_range_of_x_l3359_335991


namespace NUMINAMATH_CALUDE_expression_equality_l3359_335927

theorem expression_equality (y : ℝ) (Q : ℝ) (h : 5 * (3 * y - 7 * Real.pi) = Q) :
  10 * (6 * y - 14 * Real.pi) = 4 * Q := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l3359_335927
