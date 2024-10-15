import Mathlib

namespace NUMINAMATH_CALUDE_product_of_sum_equals_three_times_product_l132_13289

theorem product_of_sum_equals_three_times_product (x y : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) 
  (h3 : x + y = 3 * x * y) (h4 : x + y ≠ 0) : x * y = (x + y) / 3 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sum_equals_three_times_product_l132_13289


namespace NUMINAMATH_CALUDE_max_brownies_l132_13254

theorem max_brownies (m n : ℕ) (h1 : m > 0) (h2 : n > 0) : 
  (2 * (m - 2) * (n - 2) = 2 * m + 2 * n - 4) → m * n ≤ 84 := by
  sorry

end NUMINAMATH_CALUDE_max_brownies_l132_13254


namespace NUMINAMATH_CALUDE_triangle_properties_l132_13285

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Given conditions for the triangle -/
def satisfies_conditions (t : Triangle) : Prop :=
  t.c = 2 * Real.sqrt 3 ∧
  t.a * Real.sin t.A - t.c * Real.sin t.C = (t.a - t.b) * Real.sin t.B ∧
  t.c + t.b * Real.cos t.A = t.a * (4 * Real.cos t.A + Real.cos t.B)

/-- Theorem stating the conclusions -/
theorem triangle_properties (t : Triangle) (h : satisfies_conditions t) :
  t.C = Real.pi / 3 ∧ t.a * t.b * Real.sin t.C / 2 = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l132_13285


namespace NUMINAMATH_CALUDE_solve_for_x_l132_13248

theorem solve_for_x (x y : ℝ) (h1 : x + 2*y = 100) (h2 : y = 25) : x = 50 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_x_l132_13248


namespace NUMINAMATH_CALUDE_abs_c_value_l132_13215

def f (a b c : ℤ) (x : ℂ) : ℂ := a * x^4 + b * x^3 + c * x^2 + b * x + a

theorem abs_c_value (a b c : ℤ) (h1 : Int.gcd a (Int.gcd b c) = 1) 
  (h2 : f a b c (2 + Complex.I) = 0) : 
  Int.natAbs c = 42 := by
  sorry

end NUMINAMATH_CALUDE_abs_c_value_l132_13215


namespace NUMINAMATH_CALUDE_some_board_game_masters_enjoy_logic_puzzles_l132_13207

-- Define the universe
variable (U : Type)

-- Define predicates
variable (M : U → Prop)  -- M x means x is a mathematics enthusiast
variable (B : U → Prop)  -- B x means x is a board game master
variable (L : U → Prop)  -- L x means x enjoys logic puzzles

-- State the theorem
theorem some_board_game_masters_enjoy_logic_puzzles
  (h1 : ∀ x, M x → L x)  -- All mathematics enthusiasts enjoy logic puzzles
  (h2 : ∃ x, B x ∧ M x)  -- Some board game masters are mathematics enthusiasts
  : ∃ x, B x ∧ L x :=    -- Some board game masters enjoy logic puzzles
by
  sorry


end NUMINAMATH_CALUDE_some_board_game_masters_enjoy_logic_puzzles_l132_13207


namespace NUMINAMATH_CALUDE_square_completion_l132_13225

theorem square_completion (x : ℝ) : x^2 + 6*x - 5 = 0 ↔ (x + 3)^2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_square_completion_l132_13225


namespace NUMINAMATH_CALUDE_polynomial_value_theorem_l132_13260

-- Define the polynomial P
def P (a b c d : ℝ) (x : ℝ) : ℝ := x^4 + a*x^3 + b*x^2 + c*x + d

-- State the theorem
theorem polynomial_value_theorem (a b c d : ℝ) :
  P a b c d 1 = 7 →
  P a b c d 2 = 52 →
  P a b c d 3 = 97 →
  (P a b c d 9 + P a b c d (-5)) / 4 = 1202 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_value_theorem_l132_13260


namespace NUMINAMATH_CALUDE_todays_production_l132_13214

def average_production (total_production : ℕ) (days : ℕ) : ℚ :=
  (total_production : ℚ) / (days : ℚ)

theorem todays_production
  (h1 : average_production (9 * 50) 9 = 50)
  (h2 : average_production ((9 * 50) + x) 10 = 55)
  : x = 100 := by
  sorry

end NUMINAMATH_CALUDE_todays_production_l132_13214


namespace NUMINAMATH_CALUDE_chess_tournament_games_l132_13250

/-- The number of games played in a chess tournament --/
def num_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a chess group of 30 players, where each player plays every other player exactly once,
    and each game involves two players, the total number of games played is 435. --/
theorem chess_tournament_games :
  num_games 30 = 435 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l132_13250


namespace NUMINAMATH_CALUDE_Q_has_exactly_one_negative_root_l132_13237

def Q (x : ℝ) : ℝ := x^7 + 5*x^5 + 5*x^4 - 6*x^3 - 2*x^2 - 10*x + 12

theorem Q_has_exactly_one_negative_root :
  ∃! x : ℝ, x < 0 ∧ Q x = 0 :=
sorry

end NUMINAMATH_CALUDE_Q_has_exactly_one_negative_root_l132_13237


namespace NUMINAMATH_CALUDE_internal_resistance_of_current_source_l132_13293

/-- Given an electric circuit with resistors R₁ and R₂, and a current source
    with internal resistance r, prove that r = 30 Ω when R₁ = 10 Ω, R₂ = 30 Ω,
    and the current ratio I₂/I₁ = 1.5 when the polarity is reversed. -/
theorem internal_resistance_of_current_source
  (R₁ R₂ r : ℝ)
  (h₁ : R₁ = 10)
  (h₂ : R₂ = 30)
  (h₃ : (R₁ + r) / (R₂ + r) = 1.5) :
  r = 30 := by
  sorry

#check internal_resistance_of_current_source

end NUMINAMATH_CALUDE_internal_resistance_of_current_source_l132_13293


namespace NUMINAMATH_CALUDE_train_speed_crossing_bridge_l132_13246

/-- The speed of a train crossing a bridge -/
theorem train_speed_crossing_bridge (train_length bridge_length : ℝ) (crossing_time : ℝ) :
  train_length = 165 →
  bridge_length = 850 →
  crossing_time = 67.66125376636536 →
  ∃ (speed : ℝ), (abs (speed - 54.018) < 0.001 ∧ 
    speed * crossing_time / 3.6 = train_length + bridge_length) := by
  sorry

end NUMINAMATH_CALUDE_train_speed_crossing_bridge_l132_13246


namespace NUMINAMATH_CALUDE_divisor_problem_l132_13226

theorem divisor_problem (x : ℕ) : 
  (95 / x = 6 ∧ 95 % x = 5) → x = 15 := by
  sorry

end NUMINAMATH_CALUDE_divisor_problem_l132_13226


namespace NUMINAMATH_CALUDE_greatest_integer_difference_l132_13292

theorem greatest_integer_difference (x y : ℝ) (hx : 3 < x ∧ x < 6) (hy : 6 < y ∧ y < 7) :
  (∀ (a b : ℝ), 3 < a ∧ a < 6 ∧ 6 < b ∧ b < 7 → ⌊b - a⌋ ≤ 2) ∧
  (∃ (a b : ℝ), 3 < a ∧ a < 6 ∧ 6 < b ∧ b < 7 ∧ ⌊b - a⌋ = 2) :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_difference_l132_13292


namespace NUMINAMATH_CALUDE_tourist_ratio_l132_13290

theorem tourist_ratio (initial_tourists : ℕ) (eaten_by_anaconda : ℕ) (final_tourists : ℕ) :
  initial_tourists = 30 →
  eaten_by_anaconda = 2 →
  final_tourists = 16 →
  ∃ (poisoned_tourists : ℕ),
    poisoned_tourists * 1 = (initial_tourists - eaten_by_anaconda - final_tourists) * 2 :=
by sorry

end NUMINAMATH_CALUDE_tourist_ratio_l132_13290


namespace NUMINAMATH_CALUDE_second_shop_payment_l132_13236

/-- The amount Rahim paid for the books from the second shop -/
def second_shop_amount (first_shop_books : ℕ) (second_shop_books : ℕ) (first_shop_amount : ℕ) (average_price : ℕ) : ℕ :=
  (first_shop_books + second_shop_books) * average_price - first_shop_amount

/-- Theorem stating the amount Rahim paid for the books from the second shop -/
theorem second_shop_payment :
  second_shop_amount 40 20 600 14 = 240 := by
  sorry

end NUMINAMATH_CALUDE_second_shop_payment_l132_13236


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l132_13255

theorem complex_fraction_simplification :
  (2 - Complex.I) / (1 + 2 * Complex.I) = -Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l132_13255


namespace NUMINAMATH_CALUDE_product_of_solutions_l132_13274

theorem product_of_solutions (y : ℝ) : 
  (∃ y₁ y₂ : ℝ, (|3 * y₁| = 2 * (|3 * y₁| - 1) ∧ 
                 |3 * y₂| = 2 * (|3 * y₂| - 1) ∧ 
                 y₁ ≠ y₂ ∧
                 (∀ y₃ : ℝ, |3 * y₃| = 2 * (|3 * y₃| - 1) → y₃ = y₁ ∨ y₃ = y₂)) →
                 y₁ * y₂ = -4/9) :=
by sorry

end NUMINAMATH_CALUDE_product_of_solutions_l132_13274


namespace NUMINAMATH_CALUDE_shane_leftover_bread_l132_13253

/-- The number of slices of bread leftover after making sandwiches --/
def bread_leftover (bread_packages : ℕ) (slices_per_bread_package : ℕ) 
                   (ham_packages : ℕ) (slices_per_ham_package : ℕ) 
                   (bread_per_sandwich : ℕ) : ℕ :=
  let total_bread := bread_packages * slices_per_bread_package
  let total_ham := ham_packages * slices_per_ham_package
  let sandwiches := total_ham
  let bread_used := sandwiches * bread_per_sandwich
  total_bread - bread_used

/-- Theorem stating that Shane will have 8 slices of bread leftover --/
theorem shane_leftover_bread : 
  bread_leftover 2 20 2 8 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_shane_leftover_bread_l132_13253


namespace NUMINAMATH_CALUDE_divisibility_property_l132_13244

theorem divisibility_property (n : ℕ) (a : Fin n → ℕ+) 
  (h_n : n ≥ 3)
  (h_gcd : Nat.gcd (Finset.univ.prod (fun i => (a i).val)) = 1)
  (h_div : ∀ i : Fin n, (a i).val ∣ (Finset.univ.sum (fun j => (a j).val))) :
  (Finset.univ.prod (fun i => (a i).val)) ∣ (Finset.univ.sum (fun i => (a i).val))^(n-2) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_property_l132_13244


namespace NUMINAMATH_CALUDE_even_odd_periodic_properties_l132_13240

-- Define the properties of even and odd functions
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define the property of periodic functions
def IsPeriodic (f : ℝ → ℝ) : Prop := ∃ t : ℝ, t ≠ 0 ∧ ∀ x, f (x + t) = f x

-- State the theorem
theorem even_odd_periodic_properties 
  (f g : ℝ → ℝ) 
  (hf_even : IsEven f) 
  (hg_odd : IsOdd g) 
  (hf_periodic : IsPeriodic f) 
  (hg_periodic : IsPeriodic g) : 
  IsOdd (λ x ↦ g (g x)) ∧ IsPeriodic (λ x ↦ f x * g x) := by
  sorry

end NUMINAMATH_CALUDE_even_odd_periodic_properties_l132_13240


namespace NUMINAMATH_CALUDE_shaded_to_unshaded_ratio_is_five_thirds_l132_13298

/-- A structure representing a nested square figure -/
structure NestedSquareFigure where
  /-- The number of nested squares in the figure -/
  num_squares : ℕ
  /-- Predicate ensuring inner squares have vertices at midpoints of outer squares -/
  midpoint_property : num_squares > 1 → True

/-- The ratio of shaded to unshaded area in a nested square figure -/
def shaded_to_unshaded_ratio (figure : NestedSquareFigure) : Rat :=
  5 / 3

/-- Theorem stating the ratio of shaded to unshaded area is 5:3 -/
theorem shaded_to_unshaded_ratio_is_five_thirds (figure : NestedSquareFigure) :
  shaded_to_unshaded_ratio figure = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_shaded_to_unshaded_ratio_is_five_thirds_l132_13298


namespace NUMINAMATH_CALUDE_expand_expression_l132_13247

theorem expand_expression (y : ℝ) : 5 * (y + 6) * (y - 3) = 5 * y^2 + 15 * y - 90 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l132_13247


namespace NUMINAMATH_CALUDE_nonagon_diagonal_intersection_probability_l132_13200

/-- A regular nonagon is a 9-sided polygon with all sides and angles equal -/
structure RegularNonagon where
  -- We don't need to define the structure explicitly for this problem

/-- The number of diagonals in a regular nonagon -/
def num_diagonals (n : RegularNonagon) : ℕ := 27

/-- The number of ways to choose 4 vertices from 9 vertices -/
def num_four_vertices_choices (n : RegularNonagon) : ℕ := Nat.choose 9 4

/-- The number of ways to choose 2 diagonals from the total number of diagonals -/
def num_diagonal_pairs (n : RegularNonagon) : ℕ := Nat.choose (num_diagonals n) 2

/-- The probability of two randomly chosen diagonals intersecting inside the nonagon -/
def intersection_probability (n : RegularNonagon) : ℚ :=
  (num_four_vertices_choices n : ℚ) / (num_diagonal_pairs n : ℚ)

theorem nonagon_diagonal_intersection_probability (n : RegularNonagon) :
  intersection_probability n = 14 / 39 := by
  sorry

end NUMINAMATH_CALUDE_nonagon_diagonal_intersection_probability_l132_13200


namespace NUMINAMATH_CALUDE_beautiful_arrangements_theorem_l132_13235

/-- A beautiful arrangement of numbers 0 to n is a circular arrangement where 
    for any four distinct numbers a, b, c, d with a + c = b + d, 
    the chord joining a and c does not intersect the chord joining b and d -/
def is_beautiful_arrangement (n : ℕ) (arrangement : List ℕ) : Prop :=
  sorry

/-- M is the number of beautiful arrangements of numbers 0 to n -/
def M (n : ℕ) : ℕ :=
  sorry

/-- N is the number of pairs (x, y) of positive integers such that x + y ≤ n and gcd(x, y) = 1 -/
def N (n : ℕ) : ℕ :=
  sorry

/-- For any integer n ≥ 2, M(n) = N(n) + 1 -/
theorem beautiful_arrangements_theorem (n : ℕ) (h : n ≥ 2) : M n = N n + 1 :=
  sorry

end NUMINAMATH_CALUDE_beautiful_arrangements_theorem_l132_13235


namespace NUMINAMATH_CALUDE_rahul_deepak_age_ratio_l132_13270

/-- Proves that the ratio of Rahul's current age to Deepak's current age is 4:3 -/
theorem rahul_deepak_age_ratio :
  let rahul_future_age : ℕ := 42
  let years_until_future : ℕ := 6
  let deepak_current_age : ℕ := 27
  let rahul_current_age : ℕ := rahul_future_age - years_until_future
  (rahul_current_age : ℚ) / deepak_current_age = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_rahul_deepak_age_ratio_l132_13270


namespace NUMINAMATH_CALUDE_cubic_equation_solutions_l132_13272

theorem cubic_equation_solutions :
  let z₁ : ℂ := -3
  let z₂ : ℂ := (3/2) + (3/2) * Complex.I * Real.sqrt 3
  let z₃ : ℂ := (3/2) - (3/2) * Complex.I * Real.sqrt 3
  (∀ z : ℂ, z^3 = -27 ↔ z = z₁ ∨ z = z₂ ∨ z = z₃) := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solutions_l132_13272


namespace NUMINAMATH_CALUDE_slope_product_for_30_degree_angle_l132_13286

theorem slope_product_for_30_degree_angle (m₁ m₂ : ℝ) :
  m₁ ≠ 0 →
  m₂ = 4 * m₁ →
  |((m₂ - m₁) / (1 + m₁ * m₂))| = 1 / Real.sqrt 3 →
  m₁ * m₂ = (38 - 6 * Real.sqrt 33) / 16 :=
by sorry

end NUMINAMATH_CALUDE_slope_product_for_30_degree_angle_l132_13286


namespace NUMINAMATH_CALUDE_units_digit_of_42_cubed_plus_27_squared_l132_13201

theorem units_digit_of_42_cubed_plus_27_squared : ∃ n : ℕ, 42^3 + 27^2 = 10 * n + 7 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_42_cubed_plus_27_squared_l132_13201


namespace NUMINAMATH_CALUDE_sharp_2_5_3_equals_1_l132_13281

-- Define the # operation for real numbers
def sharp (a b c : ℝ) : ℝ := b^2 - 4*a*c

-- Theorem statement
theorem sharp_2_5_3_equals_1 : sharp 2 5 3 = 1 := by sorry

end NUMINAMATH_CALUDE_sharp_2_5_3_equals_1_l132_13281


namespace NUMINAMATH_CALUDE_fourth_number_l132_13251

def sequence_property (a : ℕ → ℕ) : Prop :=
  ∀ n, n ≥ 3 → n ≤ 10 → a n = a (n - 1) + a (n - 2)

theorem fourth_number (a : ℕ → ℕ) (h : sequence_property a) 
  (h7 : a 7 = 42) (h9 : a 9 = 110) : a 4 = 10 := by
  sorry

end NUMINAMATH_CALUDE_fourth_number_l132_13251


namespace NUMINAMATH_CALUDE_circle_area_tripled_l132_13222

theorem circle_area_tripled (r n : ℝ) : 
  (π * (r + n)^2 = 3 * π * r^2) → (r = n * (Real.sqrt 3 - 1) / 2) :=
by sorry

end NUMINAMATH_CALUDE_circle_area_tripled_l132_13222


namespace NUMINAMATH_CALUDE_three_students_same_group_probability_l132_13227

theorem three_students_same_group_probability 
  (total_groups : ℕ) 
  (student_count : ℕ) 
  (h1 : total_groups = 4) 
  (h2 : student_count ≥ 3) 
  (h3 : student_count % total_groups = 0) :
  (1 : ℚ) / (total_groups ^ 2) = 1 / 16 :=
sorry

end NUMINAMATH_CALUDE_three_students_same_group_probability_l132_13227


namespace NUMINAMATH_CALUDE_eighth_term_is_128_l132_13242

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, n > 0 → ∃ q : ℝ, a (n + 1) = q * a n
  second_term : a 2 = 2
  product_condition : a 3 * a 4 = 32

/-- The 8th term of the geometric sequence is 128 -/
theorem eighth_term_is_128 (seq : GeometricSequence) : seq.a 8 = 128 := by
  sorry

end NUMINAMATH_CALUDE_eighth_term_is_128_l132_13242


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l132_13216

theorem arithmetic_calculation : 3127 + 240 / 60 * 5 - 227 = 2920 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l132_13216


namespace NUMINAMATH_CALUDE_total_cost_is_correct_l132_13264

-- Define the package details
def package_A_price : ℝ := 10
def package_A_months : ℕ := 6
def package_A_discount : ℝ := 0.10

def package_B_price : ℝ := 12
def package_B_months : ℕ := 9
def package_B_discount : ℝ := 0.15

-- Define the tax rate
def sales_tax_rate : ℝ := 0.08

-- Define the function to calculate the total cost
def total_cost (package_A_price package_A_months package_A_discount
                package_B_price package_B_months package_B_discount
                sales_tax_rate : ℝ) : ℝ :=
  let package_A_total := package_A_price * package_A_months
  let package_B_total := package_B_price * package_B_months
  let package_A_discounted := package_A_total * (1 - package_A_discount)
  let package_B_discounted := package_B_total * (1 - package_B_discount)
  let package_A_tax := package_A_total * sales_tax_rate
  let package_B_tax := package_B_total * sales_tax_rate
  package_A_discounted + package_A_tax + package_B_discounted + package_B_tax

-- Theorem statement
theorem total_cost_is_correct :
  total_cost package_A_price package_A_months package_A_discount
             package_B_price package_B_months package_B_discount
             sales_tax_rate = 159.24 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_correct_l132_13264


namespace NUMINAMATH_CALUDE_retail_price_calculation_l132_13266

theorem retail_price_calculation (wholesale_price : ℝ) (discount_rate : ℝ) (profit_rate : ℝ) : 
  wholesale_price = 90 ∧ 
  discount_rate = 0.1 ∧ 
  profit_rate = 0.2 →
  ∃ retail_price : ℝ, 
    retail_price * (1 - discount_rate) = wholesale_price * (1 + profit_rate) ∧
    retail_price = 120 := by
sorry

end NUMINAMATH_CALUDE_retail_price_calculation_l132_13266


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_probability_l132_13269

/-- The type of integers with absolute value less than or equal to 5 -/
def IntWithinFive : Type := {n : ℤ // n.natAbs ≤ 5}

/-- The sample space of ordered pairs (b, c) -/
def SampleSpace : Type := IntWithinFive × IntWithinFive

/-- Predicate for when a quadratic equation has no real roots -/
def NoRealRoots (p : SampleSpace) : Prop :=
  let b := p.1.val
  let c := p.2.val
  b^2 < 4*c

/-- The number of elements in the sample space -/
def TotalCount : ℕ := 121

/-- The count of pairs (b, c) where the quadratic has no real roots -/
def FavorableCount : ℕ := 70

/-- The probability of the quadratic having no real roots -/
def Probability : ℚ := FavorableCount / TotalCount

theorem quadratic_no_real_roots_probability :
  Probability = 70 / 121 := by sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_probability_l132_13269


namespace NUMINAMATH_CALUDE_function_value_at_negative_one_l132_13209

/-- Given a function f(x) = a*sin(x) + b*tan(x) + 3 where a and b are real numbers,
    if f(1) = 1, then f(-1) = 5. -/
theorem function_value_at_negative_one 
  (a b : ℝ) 
  (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = a * Real.sin x + b * Real.tan x + 3) 
  (h2 : f 1 = 1) : 
  f (-1) = 5 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_negative_one_l132_13209


namespace NUMINAMATH_CALUDE_correct_answers_statistics_probability_two_multiple_choice_A_l132_13228

-- Define the data for schools A and B
def school_A_students : ℕ := 12
def school_A_mean : ℚ := 1
def school_A_variance : ℚ := 1

def school_B_students : ℕ := 8
def school_B_mean : ℚ := 3/2
def school_B_variance : ℚ := 1/4

-- Define the boxes
def box_A_multiple_choice : ℕ := 4
def box_A_fill_blank : ℕ := 2
def box_B_multiple_choice : ℕ := 3
def box_B_fill_blank : ℕ := 3

-- Part 1: Mean and Variance Calculation
def total_students : ℕ := school_A_students + school_B_students

theorem correct_answers_statistics : 
  let total_mean : ℚ := (school_A_students * school_A_mean + school_B_students * school_B_mean) / total_students
  let total_variance : ℚ := (school_A_students * (school_A_variance + (school_A_mean - total_mean)^2) + 
                             school_B_students * (school_B_variance + (school_B_mean - total_mean)^2)) / total_students
  total_mean = 6/5 ∧ total_variance = 19/25 := by sorry

-- Part 2: Probability Calculation
def prob_two_multiple_choice_A : ℚ := 2/5
def prob_one_multiple_one_fill_A : ℚ := 8/15
def prob_two_fill_A : ℚ := 1/15

def prob_B_multiple_given_A_two_multiple : ℚ := 5/8
def prob_B_multiple_given_A_one_each : ℚ := 8/15
def prob_B_multiple_given_A_two_fill : ℚ := 3/8

theorem probability_two_multiple_choice_A : 
  let prob_B_multiple : ℚ := prob_two_multiple_choice_A * prob_B_multiple_given_A_two_multiple + 
                              prob_one_multiple_one_fill_A * prob_B_multiple_given_A_one_each + 
                              prob_two_fill_A * prob_B_multiple_given_A_two_fill
  let prob_A_two_multiple_given_B_multiple : ℚ := (prob_two_multiple_choice_A * prob_B_multiple_given_A_two_multiple) / prob_B_multiple
  prob_A_two_multiple_given_B_multiple = 6/13 := by sorry

end NUMINAMATH_CALUDE_correct_answers_statistics_probability_two_multiple_choice_A_l132_13228


namespace NUMINAMATH_CALUDE_tims_books_l132_13223

theorem tims_books (mike_books : ℕ) (total_books : ℕ) (h1 : mike_books = 20) (h2 : total_books = 42) :
  total_books - mike_books = 22 := by
sorry

end NUMINAMATH_CALUDE_tims_books_l132_13223


namespace NUMINAMATH_CALUDE_fraction_value_theorem_l132_13245

theorem fraction_value_theorem (x : ℝ) :
  2 / (x - 3) = 2 → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_theorem_l132_13245


namespace NUMINAMATH_CALUDE_angle_terminal_side_l132_13265

/-- Given an angle α whose terminal side passes through the point (m, -3) 
    and whose cosine is -4/5, prove that m = -4. -/
theorem angle_terminal_side (α : Real) (m : Real) : 
  (∃ (x y : Real), x = m ∧ y = -3 ∧ x = Real.cos α * Real.sqrt (x^2 + y^2) ∧ y = Real.sin α * Real.sqrt (x^2 + y^2)) →
  Real.cos α = -4/5 →
  m = -4 :=
by sorry

end NUMINAMATH_CALUDE_angle_terminal_side_l132_13265


namespace NUMINAMATH_CALUDE_A_initial_investment_l132_13271

/-- Represents the initial investment of A in rupees -/
def A_investment : ℝ := sorry

/-- Represents B's investment in rupees -/
def B_investment : ℝ := 21000

/-- Represents the number of months A invested -/
def A_months : ℝ := 12

/-- Represents the number of months B invested -/
def B_months : ℝ := 3

/-- Represents A's share in the profit ratio -/
def A_share : ℝ := 2

/-- Represents B's share in the profit ratio -/
def B_share : ℝ := 3

/-- Theorem stating that A's initial investment is 3500 rupees -/
theorem A_initial_investment : 
  (A_investment * A_months) / (B_investment * B_months) = A_share / B_share → 
  A_investment = 3500 := by sorry

end NUMINAMATH_CALUDE_A_initial_investment_l132_13271


namespace NUMINAMATH_CALUDE_point_p_coordinates_and_b_range_l132_13210

/-- The system of equations defining point P -/
def system_of_equations (x y a b : ℝ) : Prop :=
  x + y = 2*a - b - 4 ∧ x - y = b - 4

/-- Point P is in the second quadrant -/
def second_quadrant (x y : ℝ) : Prop :=
  x < 0 ∧ y > 0

/-- There are only three integers that satisfy the requirements for a -/
def three_integers_for_a (a : ℝ) : Prop :=
  a = 1 ∨ a = 2 ∨ a = 3

theorem point_p_coordinates_and_b_range :
  (∀ x y : ℝ, system_of_equations x y 1 1 → x = -3 ∧ y = 0) ∧
  (∀ a b : ℝ, (∃ x y : ℝ, system_of_equations x y a b ∧ second_quadrant x y) →
    three_integers_for_a a → 0 ≤ b ∧ b < 1) :=
sorry

end NUMINAMATH_CALUDE_point_p_coordinates_and_b_range_l132_13210


namespace NUMINAMATH_CALUDE_solution_characterization_l132_13277

theorem solution_characterization (x y z : ℝ) :
  (x - y + z)^2 = x^2 - y^2 + z^2 ↔ (x = y ∧ z = 0) ∨ (x = 0 ∧ y = z) := by
  sorry

end NUMINAMATH_CALUDE_solution_characterization_l132_13277


namespace NUMINAMATH_CALUDE_locus_is_conic_locus_degeneration_l132_13219

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square in 2D space -/
structure Square where
  sideLength : ℝ
  vertex1 : Point
  vertex2 : Point

/-- The locus equation of point P on square S -/
def locusEquation (a b c x y : ℝ) : Prop :=
  (b^2 + c^2) * x^2 - 4 * a * c * x * y + (4 * a^2 + b^2 + c^2 - 4 * a * b) * y^2 = (b^2 + c^2 - 2 * a * b)^2

/-- The condition for locus degeneration -/
def degenerationCondition (a b c : ℝ) : Prop :=
  (a - b)^2 + c^2 = a^2

/-- Theorem stating that the locus of point P on square S is part of a conic -/
theorem locus_is_conic (S : Square) (P : Point) :
  S.sideLength = 2 * a →
  S.vertex1.x ≥ 0 →
  S.vertex1.y = 0 →
  S.vertex2.x = 0 →
  S.vertex2.y ≥ 0 →
  P.x = b →
  P.y = c →
  locusEquation a b c P.x P.y :=
by sorry

/-- Theorem stating the condition for locus degeneration -/
theorem locus_degeneration (S : Square) (P : Point) :
  S.sideLength = 2 * a →
  S.vertex1.x ≥ 0 →
  S.vertex1.y = 0 →
  S.vertex2.x = 0 →
  S.vertex2.y ≥ 0 →
  P.x = b →
  P.y = c →
  degenerationCondition a b c →
  ∃ (m k : ℝ), P.y = m * P.x + k :=
by sorry

end NUMINAMATH_CALUDE_locus_is_conic_locus_degeneration_l132_13219


namespace NUMINAMATH_CALUDE_quadratic_equality_l132_13241

theorem quadratic_equality (p q : ℝ) : 
  (∀ x : ℝ, (x + 4) * (x - 1) = x^2 + p*x + q) → 
  (p = 3 ∧ q = -4) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equality_l132_13241


namespace NUMINAMATH_CALUDE_hyperbola_equation_l132_13212

theorem hyperbola_equation (x y : ℝ) :
  (∃ (f : ℝ × ℝ), (f.1^2 / 16 - f.2^2 / 4 = 1) ∧
   ((x^2 / 15 - y^2 / 5 = 1) → (f = (x, y) ∨ f = (-x, y)))) →
  (x^2 / 15 - y^2 / 5 = 1) →
  ((3 * Real.sqrt 2)^2 / 15 - 2^2 / 5 = 1) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l132_13212


namespace NUMINAMATH_CALUDE_fishing_competition_result_l132_13234

/-- The number of days in the fishing competition -/
def competition_days : ℕ := 5

/-- The number of fishes caught per day by the first person -/
def fishes_per_day_1 : ℕ := 6

/-- The number of fishes caught per day by the second person -/
def fishes_per_day_2 : ℕ := 4

/-- The number of fishes caught per day by the third person -/
def fishes_per_day_3 : ℕ := 8

/-- The total number of fishes caught by the team throughout the competition -/
def total_fishes : ℕ := competition_days * (fishes_per_day_1 + fishes_per_day_2 + fishes_per_day_3)

theorem fishing_competition_result : total_fishes = 90 := by
  sorry

end NUMINAMATH_CALUDE_fishing_competition_result_l132_13234


namespace NUMINAMATH_CALUDE_intersection_empty_range_l132_13224

theorem intersection_empty_range (a : ℝ) : 
  (∀ x : ℝ, (|x - a| < 1 → ¬(1 < x ∧ x < 5))) ↔ (a ≤ 0 ∨ a ≥ 6) := by
  sorry

end NUMINAMATH_CALUDE_intersection_empty_range_l132_13224


namespace NUMINAMATH_CALUDE_everett_work_weeks_l132_13211

/-- Given that Everett worked 5 hours every day and a total of 140 hours,
    prove that he worked for 4 weeks. -/
theorem everett_work_weeks :
  let hours_per_day : ℕ := 5
  let total_hours : ℕ := 140
  let days_per_week : ℕ := 7
  let hours_per_week : ℕ := hours_per_day * days_per_week
  total_hours / hours_per_week = 4 := by
sorry

end NUMINAMATH_CALUDE_everett_work_weeks_l132_13211


namespace NUMINAMATH_CALUDE_quadratic_roots_to_coefficients_l132_13230

/-- The quadratic equation x^2 - bx + c = 0 with roots 1 and -2 has b = -1 and c = -2 -/
theorem quadratic_roots_to_coefficients :
  ∀ (b c : ℝ),
  (∀ x : ℝ, x^2 - b*x + c = 0 ↔ x = 1 ∨ x = -2) →
  b = -1 ∧ c = -2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_to_coefficients_l132_13230


namespace NUMINAMATH_CALUDE_lettuce_purchase_proof_l132_13252

/-- Calculates the total pounds of lettuce bought given the costs of green and red lettuce and the price per pound. -/
def total_lettuce_pounds (green_cost red_cost price_per_pound : ℚ) : ℚ :=
  (green_cost + red_cost) / price_per_pound

/-- Proves that given the specified costs and price per pound, the total pounds of lettuce is 7. -/
theorem lettuce_purchase_proof :
  let green_cost : ℚ := 8
  let red_cost : ℚ := 6
  let price_per_pound : ℚ := 2
  total_lettuce_pounds green_cost red_cost price_per_pound = 7 := by
sorry

#eval total_lettuce_pounds 8 6 2

end NUMINAMATH_CALUDE_lettuce_purchase_proof_l132_13252


namespace NUMINAMATH_CALUDE_honey_servings_calculation_l132_13217

/-- The number of servings in a container of honey -/
def number_of_servings (container_volume : ℚ) (serving_size : ℚ) : ℚ :=
  container_volume / serving_size

/-- Proof that a container with 37 2/3 tablespoons of honey contains 25 1/9 servings when each serving is 1 1/2 tablespoons -/
theorem honey_servings_calculation :
  let container_volume : ℚ := 113/3  -- 37 2/3 as an improper fraction
  let serving_size : ℚ := 3/2        -- 1 1/2 as an improper fraction
  number_of_servings container_volume serving_size = 226/9
  := by sorry

end NUMINAMATH_CALUDE_honey_servings_calculation_l132_13217


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l132_13205

theorem imaginary_part_of_z (z : ℂ) (h : z * (2 + Complex.I) = 1) :
  z.im = -1/5 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l132_13205


namespace NUMINAMATH_CALUDE_equal_even_odd_probability_l132_13202

/-- The number of dice being rolled -/
def numDice : ℕ := 6

/-- The number of sides on each die -/
def numSides : ℕ := 6

/-- The probability of rolling an even number on a single die -/
def probEven : ℚ := 1/2

/-- The probability of rolling an odd number on a single die -/
def probOdd : ℚ := 1/2

/-- The number of dice that need to show even (and odd) numbers for the desired outcome -/
def numEven : ℕ := numDice / 2

theorem equal_even_odd_probability :
  (Nat.choose numDice numEven : ℚ) * probEven ^ numDice = 5/16 := by
  sorry

end NUMINAMATH_CALUDE_equal_even_odd_probability_l132_13202


namespace NUMINAMATH_CALUDE_new_average_weight_l132_13221

def original_team_size : ℕ := 7
def original_team_avg_weight : ℚ := 94
def first_team_size : ℕ := 5
def first_team_avg_weight : ℚ := 100
def second_team_size : ℕ := 8
def second_team_avg_weight : ℚ := 90
def third_team_size : ℕ := 4
def third_team_avg_weight : ℚ := 120

theorem new_average_weight :
  let total_players := original_team_size + first_team_size + second_team_size + third_team_size
  let total_weight := original_team_size * original_team_avg_weight +
                      first_team_size * first_team_avg_weight +
                      second_team_size * second_team_avg_weight +
                      third_team_size * third_team_avg_weight
  (total_weight / total_players : ℚ) = 98.25 := by
  sorry

end NUMINAMATH_CALUDE_new_average_weight_l132_13221


namespace NUMINAMATH_CALUDE_remainder_problem_l132_13231

theorem remainder_problem (k : ℕ+) (h : 90 % k.val^2 = 18) : 130 % k.val = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l132_13231


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formula_l132_13276

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  first_term : a 1 = 3
  is_arithmetic : ∃ d ≠ 0, ∀ n, a (n + 1) = a n + d
  is_geometric : ∃ r ≠ 0, (a 4) ^ 2 = (a 1) * (a 13)

/-- The theorem stating the general formula for the sequence -/
theorem arithmetic_sequence_formula (seq : ArithmeticSequence) : 
  ∃ d : ℝ, d ≠ 0 ∧ ∀ n : ℕ, seq.a n = 2 * n + 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_formula_l132_13276


namespace NUMINAMATH_CALUDE_largest_common_divisor_of_consecutive_squares_l132_13213

theorem largest_common_divisor_of_consecutive_squares (n : ℤ) (h : Even n) :
  (∃ (k : ℤ), k > 1 ∧ ∀ (b : ℤ), k ∣ ((n + 1)^2 - n^2)) → False :=
by sorry

end NUMINAMATH_CALUDE_largest_common_divisor_of_consecutive_squares_l132_13213


namespace NUMINAMATH_CALUDE_max_sum_on_ellipse_l132_13275

-- Define the ellipse
def on_ellipse (x y : ℝ) : Prop := x^2/3 + y^2 = 1

-- Define the sum function S
def S (x y : ℝ) : ℝ := x + y

-- Theorem statement
theorem max_sum_on_ellipse :
  (∀ x y : ℝ, on_ellipse x y → S x y ≤ 2) ∧
  (∃ x y : ℝ, on_ellipse x y ∧ S x y = 2) := by
  sorry


end NUMINAMATH_CALUDE_max_sum_on_ellipse_l132_13275


namespace NUMINAMATH_CALUDE_both_systematic_sampling_l132_13243

/-- Represents a sampling method --/
inductive SamplingMethod
| Systematic
| SimpleRandom
| Stratified

/-- Represents a reporter conducting interviews --/
structure Reporter where
  name : String
  interval : Nat
  intervalType : String

/-- Represents the interview setup at the train station --/
structure InterviewSetup where
  reporterA : Reporter
  reporterB : Reporter
  constantFlow : Bool

/-- Determines the sampling method based on the interview setup --/
def determineSamplingMethod (reporter : Reporter) (setup : InterviewSetup) : SamplingMethod :=
  if setup.constantFlow && (reporter.intervalType = "time" || reporter.intervalType = "people") then
    SamplingMethod.Systematic
  else
    SamplingMethod.SimpleRandom

/-- Theorem: Both reporters are using systematic sampling --/
theorem both_systematic_sampling (setup : InterviewSetup) 
  (h1 : setup.reporterA = { name := "A", interval := 10, intervalType := "time" })
  (h2 : setup.reporterB = { name := "B", interval := 1000, intervalType := "people" })
  (h3 : setup.constantFlow = true) :
  determineSamplingMethod setup.reporterA setup = SamplingMethod.Systematic ∧
  determineSamplingMethod setup.reporterB setup = SamplingMethod.Systematic := by
  sorry


end NUMINAMATH_CALUDE_both_systematic_sampling_l132_13243


namespace NUMINAMATH_CALUDE_parallelogram_theorem_l132_13229

/-- A point in a 2D plane -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A quadrilateral defined by four points -/
structure Quadrilateral :=
  (A B C D : Point)

/-- Checks if a quadrilateral is convex -/
def is_convex (q : Quadrilateral) : Prop := sorry

/-- Checks if a point is the midpoint of a line segment -/
def is_midpoint (M : Point) (A B : Point) : Prop := sorry

/-- Checks if a point is inside a quadrilateral -/
def is_inside (M : Point) (q : Quadrilateral) : Prop := sorry

/-- Checks if four points form a parallelogram -/
def is_parallelogram (A B C D : Point) : Prop := sorry

/-- The main theorem -/
theorem parallelogram_theorem (ABCD : Quadrilateral) (P Q R S M : Point) :
  is_convex ABCD →
  is_midpoint P ABCD.A ABCD.B →
  is_midpoint Q ABCD.B ABCD.C →
  is_midpoint R ABCD.C ABCD.D →
  is_midpoint S ABCD.D ABCD.A →
  is_inside M ABCD →
  is_parallelogram ABCD.A P M S →
  is_parallelogram ABCD.C R M Q := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_theorem_l132_13229


namespace NUMINAMATH_CALUDE_triangle_side_length_l132_13220

theorem triangle_side_length (a b c : ℝ) (C : ℝ) : 
  b = Real.sqrt 7 → 
  a = 3 → 
  Real.tan C = Real.sqrt 3 / 2 → 
  c = Real.sqrt (a^2 + b^2 - 2*a*b*(Real.cos C)) → 
  c = 2 := by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l132_13220


namespace NUMINAMATH_CALUDE_vector_sum_zero_l132_13288

variable {V : Type*} [AddCommGroup V]
variable (A C D E : V)

theorem vector_sum_zero :
  (E - C) + (C - A) - (E - D) - (D - A) = (0 : V) := by sorry

end NUMINAMATH_CALUDE_vector_sum_zero_l132_13288


namespace NUMINAMATH_CALUDE_largest_integer_with_remainder_ninety_four_satisfies_conditions_ninety_four_is_largest_l132_13296

theorem largest_integer_with_remainder (n : ℕ) : n < 100 ∧ n % 6 = 4 → n ≤ 94 :=
by sorry

theorem ninety_four_satisfies_conditions : 94 < 100 ∧ 94 % 6 = 4 :=
by sorry

theorem ninety_four_is_largest : ∃ (n : ℕ), n = 94 ∧ n < 100 ∧ n % 6 = 4 ∧ ∀ (m : ℕ), m < 100 ∧ m % 6 = 4 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_with_remainder_ninety_four_satisfies_conditions_ninety_four_is_largest_l132_13296


namespace NUMINAMATH_CALUDE_partnership_profit_l132_13279

/-- Calculates the total profit of a partnership given investments and one partner's share -/
def calculate_total_profit (investment_a investment_b investment_c c_share : ℕ) : ℕ :=
  let total_parts := investment_a / investment_c + investment_b / investment_c + 1
  total_parts * c_share

/-- Proves that given the investments and C's share, the total profit is 252000 -/
theorem partnership_profit (investment_a investment_b investment_c c_share : ℕ) 
  (h1 : investment_a = 8000)
  (h2 : investment_b = 4000)
  (h3 : investment_c = 2000)
  (h4 : c_share = 36000) :
  calculate_total_profit investment_a investment_b investment_c c_share = 252000 := by
  sorry

#eval calculate_total_profit 8000 4000 2000 36000

end NUMINAMATH_CALUDE_partnership_profit_l132_13279


namespace NUMINAMATH_CALUDE_marie_keeps_remainder_l132_13238

/-- The number of lollipops Marie keeps for herself -/
def lollipops_kept (total_lollipops : ℕ) (num_friends : ℕ) : ℕ :=
  total_lollipops % num_friends

/-- The total number of lollipops Marie has -/
def total_lollipops : ℕ := 75 + 132 + 9 + 315

/-- The number of friends Marie has -/
def num_friends : ℕ := 13

theorem marie_keeps_remainder :
  lollipops_kept total_lollipops num_friends = 11 := by
  sorry

end NUMINAMATH_CALUDE_marie_keeps_remainder_l132_13238


namespace NUMINAMATH_CALUDE_distribute_five_balls_three_boxes_l132_13259

/-- The number of ways to distribute n indistinguishable balls into k indistinguishable boxes with no empty boxes -/
def distribute_balls (n k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 3 ways to distribute 5 indistinguishable balls into 3 indistinguishable boxes with no empty boxes -/
theorem distribute_five_balls_three_boxes : distribute_balls 5 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_distribute_five_balls_three_boxes_l132_13259


namespace NUMINAMATH_CALUDE_other_number_proof_l132_13258

theorem other_number_proof (x : Float) : 
  (0.5 : Float) = x + 0.33333333333333337 → x = 0.16666666666666663 := by
  sorry

end NUMINAMATH_CALUDE_other_number_proof_l132_13258


namespace NUMINAMATH_CALUDE_polygon_sides_doubled_l132_13233

/-- The number of diagonals in a polygon with n sides -/
def diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: If doubling the sides of a polygon increases the diagonals by 45, the polygon has 6 sides -/
theorem polygon_sides_doubled (n : ℕ) (h : n > 3) :
  diagonals (2 * n) - diagonals n = 45 → n = 6 := by sorry

end NUMINAMATH_CALUDE_polygon_sides_doubled_l132_13233


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l132_13232

theorem quadratic_equation_solution :
  ∃! y : ℝ, y^2 + 6*y + 8 = -(y + 4)*(y + 6) :=
by
  -- The unique solution is y = -4
  use -4
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l132_13232


namespace NUMINAMATH_CALUDE_joan_football_games_l132_13299

/-- The number of football games Joan went to this year -/
def games_this_year : ℕ := 4

/-- The number of football games Joan went to last year -/
def games_last_year : ℕ := 9

/-- The total number of football games Joan went to -/
def total_games : ℕ := games_this_year + games_last_year

theorem joan_football_games : total_games = 13 := by
  sorry

end NUMINAMATH_CALUDE_joan_football_games_l132_13299


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l132_13218

theorem complex_fraction_simplification :
  (Complex.I * 3 - 1) / (1 + Complex.I * 3) = Complex.mk (4/5) (3/5) := by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l132_13218


namespace NUMINAMATH_CALUDE_quadratic_negative_root_l132_13278

theorem quadratic_negative_root (a : ℝ) (h : a < 0) :
  ∃ (condition : Prop), condition → ∃ x : ℝ, x < 0 ∧ a * x^2 + 2*x + 1 = 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_negative_root_l132_13278


namespace NUMINAMATH_CALUDE_gcd_factorial_problem_l132_13256

theorem gcd_factorial_problem : Nat.gcd (Nat.factorial 7) ((Nat.factorial 10) / (Nat.factorial 4)) = 5040 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_problem_l132_13256


namespace NUMINAMATH_CALUDE_estimated_value_reasonable_l132_13208

/-- The lower bound of the scale -/
def lower_bound : ℝ := 9.80

/-- The upper bound of the scale -/
def upper_bound : ℝ := 10.0

/-- The estimated value -/
def estimated_value : ℝ := 9.95

/-- Theorem stating that the estimated value is a reasonable approximation -/
theorem estimated_value_reasonable :
  lower_bound < estimated_value ∧
  estimated_value < upper_bound ∧
  (estimated_value - lower_bound) > (upper_bound - estimated_value) :=
by sorry

end NUMINAMATH_CALUDE_estimated_value_reasonable_l132_13208


namespace NUMINAMATH_CALUDE_appropriate_sampling_methods_l132_13283

/-- Represents different sampling methods -/
inductive SamplingMethod
  | SimpleRandom
  | Stratified
  | Systematic

/-- Represents a survey with its characteristics -/
structure Survey where
  totalUnits : ℕ
  sampleSize : ℕ
  hasSignificantDifferences : Bool

/-- Determines the most appropriate sampling method for a given survey -/
def mostAppropriateSamplingMethod (s : Survey) : SamplingMethod :=
  if s.hasSignificantDifferences then
    SamplingMethod.Stratified
  else
    SamplingMethod.SimpleRandom

/-- The first survey of high school classes -/
def survey1 : Survey :=
  { totalUnits := 15
  , sampleSize := 2
  , hasSignificantDifferences := false }

/-- The second survey of stores in the city -/
def survey2 : Survey :=
  { totalUnits := 1500
  , sampleSize := 15
  , hasSignificantDifferences := true }

theorem appropriate_sampling_methods :
  (mostAppropriateSamplingMethod survey1 = SamplingMethod.SimpleRandom) ∧
  (mostAppropriateSamplingMethod survey2 = SamplingMethod.Stratified) := by
  sorry

end NUMINAMATH_CALUDE_appropriate_sampling_methods_l132_13283


namespace NUMINAMATH_CALUDE_triangle_side_length_l132_13203

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) :
  b = Real.sqrt 3 →
  A = π / 4 →
  B = π / 3 →
  (Real.sin A) / a = (Real.sin B) / b →
  a = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l132_13203


namespace NUMINAMATH_CALUDE_modified_lucas_105_mod_9_l132_13291

def modifiedLucas : ℕ → ℕ
  | 0 => 2
  | 1 => 3
  | n + 2 => modifiedLucas n + modifiedLucas (n + 1)

theorem modified_lucas_105_mod_9 : modifiedLucas 104 % 9 = 8 := by
  sorry

end NUMINAMATH_CALUDE_modified_lucas_105_mod_9_l132_13291


namespace NUMINAMATH_CALUDE_one_female_selection_count_l132_13297

/-- The number of male students in Group A -/
def group_a_male : ℕ := 5

/-- The number of female students in Group A -/
def group_a_female : ℕ := 3

/-- The number of male students in Group B -/
def group_b_male : ℕ := 6

/-- The number of female students in Group B -/
def group_b_female : ℕ := 2

/-- The number of students to be selected from each group -/
def students_per_group : ℕ := 2

/-- The total number of selections with exactly one female student -/
def total_selections : ℕ := 345

theorem one_female_selection_count :
  (Nat.choose group_a_male 1 * Nat.choose group_a_female 1 * Nat.choose group_b_male 2) +
  (Nat.choose group_a_male 2 * Nat.choose group_b_male 1 * Nat.choose group_b_female 1) = total_selections :=
by sorry

end NUMINAMATH_CALUDE_one_female_selection_count_l132_13297


namespace NUMINAMATH_CALUDE_smallest_other_integer_l132_13273

theorem smallest_other_integer (m n x : ℕ) : 
  m > 0 → n > 0 → x > 0 →
  (m = 72 ∨ n = 72) →
  Nat.gcd m n = x + 7 →
  Nat.lcm m n = x^2 * (x + 7) →
  (m ≠ 72 → m ≥ 15309) ∧ (n ≠ 72 → n ≥ 15309) :=
sorry

end NUMINAMATH_CALUDE_smallest_other_integer_l132_13273


namespace NUMINAMATH_CALUDE_positive_root_of_cubic_equation_l132_13257

theorem positive_root_of_cubic_equation :
  ∃ x : ℝ, x > 0 ∧ x^3 - 5*x^2 + 2*x - Real.sqrt 3 = 0 :=
by
  use 3 + Real.sqrt 3
  sorry

end NUMINAMATH_CALUDE_positive_root_of_cubic_equation_l132_13257


namespace NUMINAMATH_CALUDE_cassini_identity_l132_13261

/-- Fibonacci sequence -/
def fib : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Cassini's identity for Fibonacci numbers -/
theorem cassini_identity (n : ℕ) (h : n > 0) : 
  (fib (n + 1) * fib (n - 1) - fib n ^ 2 : ℤ) = (-1) ^ n := by
  sorry

end NUMINAMATH_CALUDE_cassini_identity_l132_13261


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l132_13206

theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x, ax^2 + bx + 1 > 0 ↔ -1 < x ∧ x < 1/3) → a + b = -5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l132_13206


namespace NUMINAMATH_CALUDE_find_number_l132_13282

theorem find_number : ∃ x : ℝ, 0.123 + 0.321 + x = 1.794 ∧ x = 1.350 := by sorry

end NUMINAMATH_CALUDE_find_number_l132_13282


namespace NUMINAMATH_CALUDE_sequence_periodicity_l132_13262

def units_digit (n : ℕ) : ℕ := n % 10

def a (n : ℕ) : ℕ := units_digit (n^n)

theorem sequence_periodicity : ∀ n : ℕ, a (n + 20) = a n := by
  sorry

end NUMINAMATH_CALUDE_sequence_periodicity_l132_13262


namespace NUMINAMATH_CALUDE_complement_N_star_in_N_l132_13263

def N : Set ℕ := {n : ℕ | True}
def N_star : Set ℕ := {n : ℕ | n > 0}

theorem complement_N_star_in_N : N \ N_star = {0} := by sorry

end NUMINAMATH_CALUDE_complement_N_star_in_N_l132_13263


namespace NUMINAMATH_CALUDE_krishans_money_l132_13287

/-- Given the ratios of money between Ram, Gopal, and Krishan, and Ram's amount,
    prove that Krishan has Rs. 3468. -/
theorem krishans_money (ram gopal krishan : ℕ) : 
  (ram : ℚ) / gopal = 7 / 17 →
  (gopal : ℚ) / krishan = 7 / 17 →
  ram = 588 →
  krishan = 3468 := by
sorry

end NUMINAMATH_CALUDE_krishans_money_l132_13287


namespace NUMINAMATH_CALUDE_inequality_solution_set_l132_13284

-- Define the inequality function
def f (x : ℝ) := (3*x + 1) * (2*x - 1)

-- Define the solution set
def solution_set := {x : ℝ | x < -1/3 ∨ x > 1/2}

-- Theorem statement
theorem inequality_solution_set :
  {x : ℝ | f x > 0} = solution_set := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l132_13284


namespace NUMINAMATH_CALUDE_starting_lineup_combinations_l132_13280

def total_players : ℕ := 18
def lineup_size : ℕ := 8
def triplets : ℕ := 3
def twins : ℕ := 2

def remaining_players : ℕ := total_players - (triplets + twins)
def players_to_choose : ℕ := lineup_size - (triplets + twins)

theorem starting_lineup_combinations : 
  Nat.choose remaining_players players_to_choose = 286 := by
  sorry

end NUMINAMATH_CALUDE_starting_lineup_combinations_l132_13280


namespace NUMINAMATH_CALUDE_xyz_product_l132_13268

theorem xyz_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x * (y + z) = 120)
  (eq2 : y * (z + x) = 156)
  (eq3 : z * (x + y) = 144) :
  x * y * z = 360 := by
sorry

end NUMINAMATH_CALUDE_xyz_product_l132_13268


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l132_13204

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_property (a : ℕ → ℝ) :
  is_geometric_sequence a →
  (a 1 * a 2 * a 3 = 3) →
  (a 10 * a 11 * a 12 = 24) →
  a 13 * a 14 * a 15 = 48 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l132_13204


namespace NUMINAMATH_CALUDE_trailing_zeros_of_square_l132_13267

/-- The number of trailing zeros in (10^12 - 5)^2 is 12 -/
theorem trailing_zeros_of_square : ∃ n : ℕ, (10^12 - 5)^2 = n * 10^12 ∧ n % 10 ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_trailing_zeros_of_square_l132_13267


namespace NUMINAMATH_CALUDE_sandwich_theorem_l132_13295

/-- The number of sandwiches Samson ate on different days and meals --/
def sandwich_count : Prop :=
  let monday_lunch := 3
  let monday_dinner := 2 * monday_lunch
  let tuesday_lunch := 4
  let tuesday_dinner := tuesday_lunch / 2
  let wednesday_lunch := 2 * tuesday_lunch
  let wednesday_dinner := 3 * tuesday_lunch
  let monday_total := monday_lunch + monday_dinner
  let tuesday_total := tuesday_lunch + tuesday_dinner
  let wednesday_total := wednesday_lunch + wednesday_dinner
  wednesday_total - (monday_total + tuesday_total) = 5

theorem sandwich_theorem : sandwich_count := by
  sorry

end NUMINAMATH_CALUDE_sandwich_theorem_l132_13295


namespace NUMINAMATH_CALUDE_open_sets_l132_13249

-- Define the concept of an open set in a plane
def is_open_set (A : Set (ℝ × ℝ)) : Prop :=
  ∀ (x₀ y₀ : ℝ), (x₀, y₀) ∈ A → 
    ∃ (r : ℝ), r > 0 ∧ {(x, y) | (x - x₀)^2 + (y - y₀)^2 < r^2} ⊆ A

-- Define the four sets
def set1 : Set (ℝ × ℝ) := {(x, y) | x^2 + y^2 = 1}
def set2 : Set (ℝ × ℝ) := {(x, y) | |x + y + 2| ≥ 1}
def set3 : Set (ℝ × ℝ) := {(x, y) | |x| + |y| < 1}
def set4 : Set (ℝ × ℝ) := {(x, y) | 0 < x^2 + (y - 1)^2 ∧ x^2 + (y - 1)^2 < 1}

-- State the theorem
theorem open_sets : 
  ¬(is_open_set set1) ∧ 
  ¬(is_open_set set2) ∧ 
  (is_open_set set3) ∧ 
  (is_open_set set4) := by
  sorry

end NUMINAMATH_CALUDE_open_sets_l132_13249


namespace NUMINAMATH_CALUDE_surjective_sum_iff_constant_l132_13239

/-- A function is surjective if every element in the codomain is mapped to by at least one element in the domain. -/
def Surjective (f : ℤ → ℤ) : Prop :=
  ∀ y : ℤ, ∃ x : ℤ, f x = y

/-- The sum of two functions -/
def FunctionSum (f g : ℤ → ℤ) : ℤ → ℤ := λ x => f x + g x

/-- A function is constant if it maps all inputs to the same output -/
def ConstantFunction (f : ℤ → ℤ) : Prop :=
  ∃ c : ℤ, ∀ x : ℤ, f x = c

/-- The main theorem: a function f preserves surjectivity of g when added to it
    if and only if f is constant -/
theorem surjective_sum_iff_constant (f : ℤ → ℤ) :
  (∀ g : ℤ → ℤ, Surjective g → Surjective (FunctionSum f g)) ↔ ConstantFunction f :=
sorry

end NUMINAMATH_CALUDE_surjective_sum_iff_constant_l132_13239


namespace NUMINAMATH_CALUDE_train_speed_l132_13294

/-- The speed of a train given its length, time to pass a man, and the man's speed in the opposite direction -/
theorem train_speed (train_length : Real) (passing_time : Real) (man_speed : Real) :
  train_length = 240 ∧ 
  passing_time = 13.090909090909092 ∧ 
  man_speed = 6 →
  (train_length / 1000) / (passing_time / 3600) - man_speed = 60 := by
  sorry

#check train_speed

end NUMINAMATH_CALUDE_train_speed_l132_13294
