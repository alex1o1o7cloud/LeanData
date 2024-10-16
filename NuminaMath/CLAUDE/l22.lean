import Mathlib

namespace NUMINAMATH_CALUDE_mean_temperature_l22_2280

def temperatures : List ℝ := [82, 83, 78, 86, 88, 90, 88]

theorem mean_temperature : 
  (List.sum temperatures) / temperatures.length = 84.5714 := by
  sorry

end NUMINAMATH_CALUDE_mean_temperature_l22_2280


namespace NUMINAMATH_CALUDE_weight_of_b_l22_2277

-- Define variables for weights and heights
variable (W_a W_b W_c : ℚ)
variable (h_a h_b h_c : ℚ)

-- Define the conditions
def condition1 : Prop := (W_a + W_b + W_c) / 3 = 45
def condition2 : Prop := (W_a + W_b) / 2 = 40
def condition3 : Prop := (W_b + W_c) / 2 = 47
def condition4 : Prop := h_a + h_c = 2 * h_b
def condition5 : Prop := ∃ (n : ℤ), W_a + W_b + W_c = 2 * n + 1

-- Theorem statement
theorem weight_of_b 
  (h1 : condition1 W_a W_b W_c)
  (h2 : condition2 W_a W_b)
  (h3 : condition3 W_b W_c)
  (h4 : condition4 h_a h_b h_c)
  (h5 : condition5 W_a W_b W_c) :
  W_b = 39 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_b_l22_2277


namespace NUMINAMATH_CALUDE_simplify_expression_l22_2242

theorem simplify_expression : (2^8 + 4^5) * (2^3 - (-2)^2)^11 = 5368709120 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l22_2242


namespace NUMINAMATH_CALUDE_original_pencils_count_l22_2278

/-- The number of pencils Mike placed in the drawer -/
def pencils_added : ℕ := 30

/-- The total number of pencils now in the drawer -/
def total_pencils : ℕ := 71

/-- The original number of pencils in the drawer -/
def original_pencils : ℕ := total_pencils - pencils_added

theorem original_pencils_count : original_pencils = 41 := by
  sorry

end NUMINAMATH_CALUDE_original_pencils_count_l22_2278


namespace NUMINAMATH_CALUDE_f_properties_l22_2247

def f (x : ℝ) := x^3 - 3*x

theorem f_properties :
  (∀ y, (∃ x, x = 0 ∧ y = f x) → y = 0) ∧
  (∀ x, x < -1 → (∀ h > 0, f (x + h) > f x)) ∧
  (∀ x, x > 1 → (∀ h > 0, f (x + h) > f x)) ∧
  (∀ x, -1 < x ∧ x < 1 → (∀ h > 0, f (x + h) < f x)) ∧
  (f (-1) = 2) ∧
  (f 1 = -2) ∧
  (∀ x, f x ≤ 2) ∧
  (∀ x, f x ≥ -2) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l22_2247


namespace NUMINAMATH_CALUDE_smoothie_combinations_l22_2230

theorem smoothie_combinations (n_flavors : ℕ) (n_supplements : ℕ) : 
  n_flavors = 5 → n_supplements = 8 → n_flavors * (n_supplements.choose 3) = 280 := by
  sorry

end NUMINAMATH_CALUDE_smoothie_combinations_l22_2230


namespace NUMINAMATH_CALUDE_triangle_problem_l22_2296

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  S : Real

-- Define the theorem
theorem triangle_problem (t : Triangle) 
  (h1 : t.C.cos = 2 * t.A.cos * (t.B - π/6).sin)
  (h2 : t.S = 2 * Real.sqrt 3)
  (h3 : t.b - t.c = 2) :
  t.A = π/3 ∧ t.a = 2 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_triangle_problem_l22_2296


namespace NUMINAMATH_CALUDE_unique_integer_pair_l22_2210

theorem unique_integer_pair : 
  ∃! (a b : ℕ), 
    0 < a ∧ 0 < b ∧ 
    a < b ∧ 
    (2020 - a : ℚ) / a * (2020 - b : ℚ) / b = 2 ∧
    a = 505 ∧ 
    b = 1212 := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_pair_l22_2210


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l22_2281

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : List ℕ :=
  List.range n |>.map (λ i => a₁ + i * d)

def sum_list (L : List ℕ) : ℕ :=
  L.foldl (· + ·) 0

theorem arithmetic_sequence_sum : 
  2 * (sum_list (arithmetic_sequence 102 2 10)) = 2220 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l22_2281


namespace NUMINAMATH_CALUDE_rectangle_width_is_five_l22_2266

/-- Represents the dimensions of a rectangle --/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of a rectangle --/
def perimeter (r : Rectangle) : ℝ :=
  2 * (r.length + r.width)

/-- Theorem: A rectangle with length 6 and perimeter 22 has width 5 --/
theorem rectangle_width_is_five :
  ∀ r : Rectangle, r.length = 6 → perimeter r = 22 → r.width = 5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_width_is_five_l22_2266


namespace NUMINAMATH_CALUDE_jack_reading_pages_l22_2288

/-- The number of pages in each booklet -/
def pages_per_booklet : ℕ := 13

/-- The number of booklets in the short story section -/
def number_of_booklets : ℕ := 67

/-- The total number of pages Jack needs to read -/
def total_pages : ℕ := pages_per_booklet * number_of_booklets

theorem jack_reading_pages : total_pages = 871 := by
  sorry

end NUMINAMATH_CALUDE_jack_reading_pages_l22_2288


namespace NUMINAMATH_CALUDE_absolute_value_nonnegative_l22_2223

theorem absolute_value_nonnegative (a : ℝ) : |a| ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_nonnegative_l22_2223


namespace NUMINAMATH_CALUDE_dirichlet_approximation_l22_2255

theorem dirichlet_approximation (x : ℝ) (h_irr : Irrational x) (h_pos : 0 < x) :
  ∀ N : ℕ, ∃ p q : ℤ, N < q ∧ 0 < q ∧ |x - (p : ℝ) / (q : ℝ)| < 1 / (q : ℝ)^2 := by
  sorry

end NUMINAMATH_CALUDE_dirichlet_approximation_l22_2255


namespace NUMINAMATH_CALUDE_jean_calories_eaten_l22_2213

def pages_written : ℕ := 12
def pages_per_donut : ℕ := 2
def calories_per_donut : ℕ := 150

theorem jean_calories_eaten : 
  (pages_written / pages_per_donut) * calories_per_donut = 900 := by
  sorry

end NUMINAMATH_CALUDE_jean_calories_eaten_l22_2213


namespace NUMINAMATH_CALUDE_max_value_of_function_max_value_achieved_l22_2218

theorem max_value_of_function (x : ℝ) (h : x < 0) : x + 4/x ≤ -4 := by
  sorry

theorem max_value_achieved (x : ℝ) (h : x < 0) : ∃ x₀, x₀ < 0 ∧ x₀ + 4/x₀ = -4 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_function_max_value_achieved_l22_2218


namespace NUMINAMATH_CALUDE_equation_solutions_l22_2285

theorem equation_solutions :
  (∃ x : ℝ, 3 * x * (x - 1) = 2 - 2 * x) ∧
  (∃ x : ℝ, 3 * x^2 - 6 * x + 2 = 0) ∧
  (∀ x : ℝ, 3 * x * (x - 1) = 2 - 2 * x ↔ (x = 1 ∨ x = -2/3)) ∧
  (∀ x : ℝ, 3 * x^2 - 6 * x + 2 = 0 ↔ (x = 1 + Real.sqrt 3 / 3 ∨ x = 1 - Real.sqrt 3 / 3)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l22_2285


namespace NUMINAMATH_CALUDE_function_bounds_bounds_achievable_l22_2228

theorem function_bounds (x : ℝ) : 
  6 ≤ 7 - 4 * Real.sin x * Real.cos x + 4 * (Real.cos x)^2 - 4 * (Real.cos x)^4 ∧ 
  7 - 4 * Real.sin x * Real.cos x + 4 * (Real.cos x)^2 - 4 * (Real.cos x)^4 ≤ 10 :=
by sorry

theorem bounds_achievable : 
  (∃ x : ℝ, 7 - 4 * Real.sin x * Real.cos x + 4 * (Real.cos x)^2 - 4 * (Real.cos x)^4 = 6) ∧
  (∃ x : ℝ, 7 - 4 * Real.sin x * Real.cos x + 4 * (Real.cos x)^2 - 4 * (Real.cos x)^4 = 10) :=
by sorry

end NUMINAMATH_CALUDE_function_bounds_bounds_achievable_l22_2228


namespace NUMINAMATH_CALUDE_exponential_function_fixed_point_l22_2208

theorem exponential_function_fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f := fun (x : ℝ) ↦ a^x + 1
  f 0 = 2 := by
sorry

end NUMINAMATH_CALUDE_exponential_function_fixed_point_l22_2208


namespace NUMINAMATH_CALUDE_square_minus_product_equals_one_l22_2226

theorem square_minus_product_equals_one (x : ℝ) : (x + 2)^2 - (x + 1) * (x + 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_minus_product_equals_one_l22_2226


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l22_2258

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 2 * a * x + 1 ≥ 0) ↔ (0 ≤ a ∧ a ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l22_2258


namespace NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l22_2271

theorem arithmetic_sequence_middle_term : ∀ (a b c : ℤ),
  (a = 2^2 ∧ c = 2^4 ∧ b - a = c - b) → b = 10 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l22_2271


namespace NUMINAMATH_CALUDE_two_digit_sum_property_l22_2233

def is_valid_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧
  ∃ (x y : ℕ),
    n = 10 * x + y ∧
    x < 10 ∧ y < 10 ∧
    (x + 1 + y + 2 - 10) / 2 = x + y ∧
    y + 2 ≥ 10

theorem two_digit_sum_property :
  ∀ n : ℕ, is_valid_number n ↔ (n = 68 ∨ n = 59) :=
sorry

end NUMINAMATH_CALUDE_two_digit_sum_property_l22_2233


namespace NUMINAMATH_CALUDE_smallest_number_of_students_l22_2253

/-- Represents the number of students in each grade --/
structure Students where
  grade8 : ℕ
  grade9 : ℕ
  grade10 : ℕ

/-- The ratios given in the problem --/
def ratio_10_to_8 : ℚ := 7 / 4
def ratio_10_to_9 : ℚ := 9 / 5

/-- The function to calculate the total number of students --/
def total_students (s : Students) : ℕ := s.grade8 + s.grade9 + s.grade10

/-- The theorem stating the smallest possible number of students --/
theorem smallest_number_of_students :
  ∃ (s : Students),
    (s.grade10 : ℚ) / s.grade8 = ratio_10_to_8 ∧
    (s.grade10 : ℚ) / s.grade9 = ratio_10_to_9 ∧
    (∀ (t : Students),
      (t.grade10 : ℚ) / t.grade8 = ratio_10_to_8 →
      (t.grade10 : ℚ) / t.grade9 = ratio_10_to_9 →
      total_students s ≤ total_students t) ∧
    total_students s = 134 :=
sorry

end NUMINAMATH_CALUDE_smallest_number_of_students_l22_2253


namespace NUMINAMATH_CALUDE_molecular_weight_AlOH3_is_correct_l22_2216

/-- The molecular weight of Al(OH)3 -/
def molecular_weight_AlOH3 : ℝ := 78

/-- The number of moles given in the problem -/
def given_moles : ℝ := 7

/-- The total molecular weight for the given number of moles -/
def total_weight : ℝ := 546

/-- Theorem stating that the molecular weight of Al(OH)3 is correct -/
theorem molecular_weight_AlOH3_is_correct :
  molecular_weight_AlOH3 = total_weight / given_moles :=
by sorry

end NUMINAMATH_CALUDE_molecular_weight_AlOH3_is_correct_l22_2216


namespace NUMINAMATH_CALUDE_park_length_l22_2291

/-- The length of a rectangular park given its perimeter and breadth -/
theorem park_length (perimeter breadth : ℝ) (h1 : perimeter = 1000) (h2 : breadth = 200) :
  2 * (perimeter / 2 - breadth) = 300 := by
  sorry

end NUMINAMATH_CALUDE_park_length_l22_2291


namespace NUMINAMATH_CALUDE_villa_tournament_correct_l22_2263

/-- A tournament where each player plays with a fixed number of other players. -/
structure Tournament where
  num_players : ℕ
  games_per_player : ℕ
  total_games : ℕ

/-- The specific tournament described in the problem. -/
def villa_tournament : Tournament :=
  { num_players := 6,
    games_per_player := 4,
    total_games := 10 }

/-- Theorem stating that the total number of games in the Villa tournament is correct. -/
theorem villa_tournament_correct :
  villa_tournament.total_games = (villa_tournament.num_players * villa_tournament.games_per_player) / 2 :=
by sorry

end NUMINAMATH_CALUDE_villa_tournament_correct_l22_2263


namespace NUMINAMATH_CALUDE_simplify_fraction_l22_2270

theorem simplify_fraction : (5^5 + 5^3) / (5^4 - 5^2) = 65 / 12 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l22_2270


namespace NUMINAMATH_CALUDE_two_digit_number_square_difference_l22_2229

theorem two_digit_number_square_difference (a b : ℤ) 
  (h1 : a > b) (h2 : a + b = 10) : 
  ∃ k : ℤ, (9*a + 10)^2 - (100 - 9*a)^2 = 20 * k := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_square_difference_l22_2229


namespace NUMINAMATH_CALUDE_complex_multiplication_l22_2200

theorem complex_multiplication (i : ℂ) (h : i * i = -1) : i * (1 + i) = -1 + i := by sorry

end NUMINAMATH_CALUDE_complex_multiplication_l22_2200


namespace NUMINAMATH_CALUDE_third_year_interest_l22_2217

/-- Calculates the compound interest for a given principal, rate, and time -/
def compoundInterest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time - principal

/-- Represents the loan scenario with given parameters -/
structure LoanScenario where
  initialLoan : ℝ
  rate1 : ℝ
  rate2 : ℝ
  rate3 : ℝ

/-- Theorem stating the interest paid in the third year of the loan -/
theorem third_year_interest (loan : LoanScenario) 
  (h1 : loan.initialLoan = 9000)
  (h2 : loan.rate1 = 0.09)
  (h3 : loan.rate2 = 0.105)
  (h4 : loan.rate3 = 0.085) :
  let firstYearTotal := loan.initialLoan * (1 + loan.rate1)
  let secondYearTotal := firstYearTotal * (1 + loan.rate2)
  compoundInterest secondYearTotal loan.rate3 1 = 922.18 := by
  sorry

end NUMINAMATH_CALUDE_third_year_interest_l22_2217


namespace NUMINAMATH_CALUDE_julia_payment_l22_2238

def snickers_price : ℝ := 1.5
def snickers_quantity : ℕ := 2
def mm_quantity : ℕ := 3
def change : ℝ := 8

def mm_price : ℝ := 2 * snickers_price

def total_cost : ℝ := snickers_price * snickers_quantity + mm_price * mm_quantity

theorem julia_payment : total_cost + change = 20 := by
  sorry

end NUMINAMATH_CALUDE_julia_payment_l22_2238


namespace NUMINAMATH_CALUDE_probability_all_even_sum_l22_2264

/-- The number of tiles -/
def num_tiles : ℕ := 10

/-- The number of players -/
def num_players : ℕ := 3

/-- The number of tiles each player selects -/
def tiles_per_player : ℕ := 3

/-- The set of tile numbers -/
def tile_set : Finset ℕ := Finset.range num_tiles

/-- A function that returns true if a sum is even -/
def is_even_sum (sum : ℕ) : Prop := sum % 2 = 0

/-- The probability of a single player getting an even sum -/
def prob_even_sum_single : ℚ := 70 / 120

theorem probability_all_even_sum :
  (prob_even_sum_single ^ num_players : ℚ) = 343 / 1728 := by sorry

end NUMINAMATH_CALUDE_probability_all_even_sum_l22_2264


namespace NUMINAMATH_CALUDE_hexagon_diagonals_l22_2241

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A hexagon has 6 sides -/
def hexagon_sides : ℕ := 6

/-- Theorem: The number of diagonals in a hexagon is 9 -/
theorem hexagon_diagonals : num_diagonals hexagon_sides = 9 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_diagonals_l22_2241


namespace NUMINAMATH_CALUDE_square_area_from_diagonal_l22_2265

/-- The area of a square with a diagonal of 10 meters is 50 square meters. -/
theorem square_area_from_diagonal (d : ℝ) (h : d = 10) : 
  let s := d / Real.sqrt 2
  s ^ 2 = 50 := by sorry

end NUMINAMATH_CALUDE_square_area_from_diagonal_l22_2265


namespace NUMINAMATH_CALUDE_circle_radius_tangent_to_lines_l22_2224

/-- Given a circle with center (0,k) where k > 6, if the circle is tangent to the lines y = x, y = -x, and y = 6, then its radius is 6√2 + 6. -/
theorem circle_radius_tangent_to_lines (k : ℝ) (h : k > 6) :
  let C := { p : ℝ × ℝ | (p.1 - 0)^2 + (p.2 - k)^2 = r^2 }
  let L1 := { p : ℝ × ℝ | p.2 = p.1 }
  let L2 := { p : ℝ × ℝ | p.2 = -p.1 }
  let L3 := { p : ℝ × ℝ | p.2 = 6 }
  (∃ (p1 : ℝ × ℝ), p1 ∈ C ∧ p1 ∈ L1) →
  (∃ (p2 : ℝ × ℝ), p2 ∈ C ∧ p2 ∈ L2) →
  (∃ (p3 : ℝ × ℝ), p3 ∈ C ∧ p3 ∈ L3) →
  r = 6 * (Real.sqrt 2 + 1) :=
by
  sorry

end NUMINAMATH_CALUDE_circle_radius_tangent_to_lines_l22_2224


namespace NUMINAMATH_CALUDE_correct_product_l22_2236

-- Define a function to reverse the digits of a three-digit number
def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 100 + ((n / 10) % 10) * 10 + (n / 100)

-- Define the theorem
theorem correct_product (a b : ℕ) : 
  (100 ≤ a ∧ a < 1000) →  -- a is a three-digit number
  (0 < b) →               -- b is positive
  (reverse_digits a * b = 396) →  -- erroneous product condition
  (a * b = 693) :=        -- correct product
by sorry

end NUMINAMATH_CALUDE_correct_product_l22_2236


namespace NUMINAMATH_CALUDE_M_N_not_subset_l22_2293

def M : Set ℕ := {x | ∃ n : ℕ, x = 3^n}
def N : Set ℕ := {x | ∃ n : ℕ, x = 3*n}

theorem M_N_not_subset : (¬ (M ⊆ N)) ∧ (¬ (N ⊆ M)) := by
  sorry

end NUMINAMATH_CALUDE_M_N_not_subset_l22_2293


namespace NUMINAMATH_CALUDE_min_value_m_exists_l22_2211

theorem min_value_m_exists (m : ℝ) : 
  (∃ x₀ : ℝ, |x₀ + 1| + |x₀ - 1| ≤ m) ↔ m ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_m_exists_l22_2211


namespace NUMINAMATH_CALUDE_choose_three_cooks_from_ten_l22_2234

theorem choose_three_cooks_from_ten (n : ℕ) (k : ℕ) : n = 10 ∧ k = 3 → Nat.choose n k = 120 := by
  sorry

end NUMINAMATH_CALUDE_choose_three_cooks_from_ten_l22_2234


namespace NUMINAMATH_CALUDE_min_distance_to_line_l22_2205

theorem min_distance_to_line (x y : ℝ) (h1 : 8 * x + 15 * y = 120) (h2 : x ≥ 0) (h3 : y ≥ 0) :
  ∃ (x₀ y₀ : ℝ), x₀ ≥ 0 ∧ y₀ ≥ 0 ∧ 8 * x₀ + 15 * y₀ = 120 ∧
  (∀ (x' y' : ℝ), x' ≥ 0 → y' ≥ 0 → 8 * x' + 15 * y' = 120 → 
    Real.sqrt (x₀^2 + y₀^2) ≤ Real.sqrt (x'^2 + y'^2)) ∧
  Real.sqrt (x₀^2 + y₀^2) = 120 / 17 :=
sorry

end NUMINAMATH_CALUDE_min_distance_to_line_l22_2205


namespace NUMINAMATH_CALUDE_min_perimeter_l22_2275

/-- Represents an isosceles triangle with integer side lengths -/
structure IsoscelesTriangle where
  equalSide : ℕ
  base : ℕ

/-- Calculates the perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℕ := 2 * t.equalSide + t.base

/-- Represents the pair of isosceles triangles in the problem -/
structure TrianglePair where
  t1 : IsoscelesTriangle
  t2 : IsoscelesTriangle

/-- The conditions given in the problem -/
def satisfiesConditions (pair : TrianglePair) : Prop :=
  let t1 := pair.t1
  let t2 := pair.t2
  -- Same perimeter
  perimeter t1 = perimeter t2 ∧
  -- Ratio of bases is 10:9
  10 * t2.base = 9 * t1.base ∧
  -- Base relations
  t1.base = 2 * t1.equalSide - 12 ∧
  t2.base = 3 * t2.equalSide - 30 ∧
  -- Non-congruent
  t1 ≠ t2

theorem min_perimeter (pair : TrianglePair) :
  satisfiesConditions pair → perimeter pair.t1 ≥ 228 :=
sorry

end NUMINAMATH_CALUDE_min_perimeter_l22_2275


namespace NUMINAMATH_CALUDE_inner_triangle_area_l22_2222

/-- Given a triangle with area T, prove that the area of the triangle formed by
    joining the points that divide each side into three equal segments is T/9. -/
theorem inner_triangle_area (T : ℝ) (h : T > 0) :
  ∃ (M : ℝ), M = T / 9 ∧ M / T = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_inner_triangle_area_l22_2222


namespace NUMINAMATH_CALUDE_right_triangle_cos_c_l22_2215

theorem right_triangle_cos_c (A B C : ℝ) (h1 : A + B + C = π) (h2 : A = π/2) (h3 : Real.sin B = 3/5) : 
  Real.cos C = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_cos_c_l22_2215


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l22_2246

theorem complex_fraction_simplification :
  let z₁ : ℂ := Complex.mk 5 7
  let z₂ : ℂ := Complex.mk 2 3
  z₁ / z₂ = Complex.mk (31 / 13) (-1 / 13) := by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l22_2246


namespace NUMINAMATH_CALUDE_triangle_perimeter_l22_2292

/-- Given a triangle with sides in the ratio 1/2 : 1/3 : 1/4 and longest side 48 cm, its perimeter is 104 cm -/
theorem triangle_perimeter (a b c : ℝ) (h1 : a > 0 ∧ b > 0 ∧ c > 0) 
  (h2 : a / b = 3 / 2 ∧ a / c = 2 ∧ b / c = 4 / 3) (h3 : a = 48) : 
  a + b + c = 104 := by
sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l22_2292


namespace NUMINAMATH_CALUDE_largest_three_digit_product_l22_2239

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → n % d ≠ 0

def has_no_repeated_prime_factors (n : ℕ) : Prop :=
  ∀ p : ℕ, is_prime p → (n % (p * p) ≠ 0)

def is_valid_triple (x y : ℕ) : Prop :=
  x < 10 ∧ y < 10 ∧ x ≠ y ∧ is_prime (10 * x + y)

theorem largest_three_digit_product :
  ∃ m x y : ℕ,
    m = x * y * (10 * x + y) ∧
    is_valid_triple x y ∧
    has_no_repeated_prime_factors m ∧
    m < 1000 ∧
    (∀ m' x' y' : ℕ,
      m' = x' * y' * (10 * x' + y') →
      is_valid_triple x' y' →
      has_no_repeated_prime_factors m' →
      m' < 1000 →
      m' ≤ m) ∧
    m = 777 :=
sorry

end NUMINAMATH_CALUDE_largest_three_digit_product_l22_2239


namespace NUMINAMATH_CALUDE_same_color_probability_l22_2298

def red_plates : ℕ := 6
def blue_plates : ℕ := 5
def total_plates : ℕ := red_plates + blue_plates
def selected_plates : ℕ := 3

theorem same_color_probability :
  (Nat.choose red_plates selected_plates + Nat.choose blue_plates selected_plates) / 
  Nat.choose total_plates selected_plates = 2 / 11 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_l22_2298


namespace NUMINAMATH_CALUDE_roots_vs_ellipse_l22_2248

def has_two_positive_roots (m n : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧ x₁^2 - m*x₁ + n = 0 ∧ x₂^2 - m*x₂ + n = 0

def is_ellipse (m n : ℝ) : Prop :=
  m > 0 ∧ n > 0 ∧ m ≠ n

theorem roots_vs_ellipse (m n : ℝ) :
  ¬(has_two_positive_roots m n → is_ellipse m n) ∧
  ¬(is_ellipse m n → has_two_positive_roots m n) :=
sorry

end NUMINAMATH_CALUDE_roots_vs_ellipse_l22_2248


namespace NUMINAMATH_CALUDE_hall_length_l22_2232

/-- The length of a hall given its width, number of stones, and stone dimensions -/
theorem hall_length (hall_width : ℝ) (num_stones : ℕ) (stone_length stone_width : ℝ) :
  hall_width = 15 ∧ 
  num_stones = 1350 ∧ 
  stone_length = 0.8 ∧ 
  stone_width = 0.5 →
  (stone_length * stone_width * num_stones) / hall_width = 36 :=
by sorry

end NUMINAMATH_CALUDE_hall_length_l22_2232


namespace NUMINAMATH_CALUDE_max_y_coordinate_ellipse_l22_2273

theorem max_y_coordinate_ellipse :
  ∀ x y : ℝ, x^2/25 + (y-3)^2/25 = 0 → y ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_max_y_coordinate_ellipse_l22_2273


namespace NUMINAMATH_CALUDE_sum_of_powers_equals_power_of_sum_l22_2294

theorem sum_of_powers_equals_power_of_sum : 2^2 + 2^2 + 2^2 + 2^2 = 2^4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_equals_power_of_sum_l22_2294


namespace NUMINAMATH_CALUDE_car_pedestrian_speed_ratio_l22_2201

-- Define the bridge length
variable (L : ℝ)

-- Define the speeds of the pedestrian and the car
variable (v_p v_c : ℝ)

-- Assume positive speeds and bridge length
variable (h_pos_L : L > 0)
variable (h_pos_v_p : v_p > 0)
variable (h_pos_v_c : v_c > 0)

-- Define the theorem
theorem car_pedestrian_speed_ratio
  (h1 : v_c * (L / (5 * v_p)) = L) -- Car covers full bridge in time pedestrian covers 1/5
  (h2 : v_p * (L / (5 * v_p)) = L / 5) -- Pedestrian covers 1/5 bridge in same time
  : v_c / v_p = 5 :=
by sorry

end NUMINAMATH_CALUDE_car_pedestrian_speed_ratio_l22_2201


namespace NUMINAMATH_CALUDE_circle_x_plus_y_bounds_l22_2297

-- Define the circle in polar form
def polar_circle (ρ θ : ℝ) : Prop :=
  ρ^2 - 4 * Real.sqrt 2 * ρ * Real.cos (θ - Real.pi/4) + 6 = 0

-- Define a point on the circle in Cartesian coordinates
def point_on_circle (x y : ℝ) : Prop :=
  (x - 2)^2 + (y - 2)^2 = 4

-- Theorem statement
theorem circle_x_plus_y_bounds :
  ∀ x y : ℝ, point_on_circle x y →
  (∃ θ : ℝ, polar_circle (Real.sqrt (x^2 + y^2)) θ) →
  2 ≤ x + y ∧ x + y ≤ 6 :=
by sorry

end NUMINAMATH_CALUDE_circle_x_plus_y_bounds_l22_2297


namespace NUMINAMATH_CALUDE_savings_ratio_l22_2289

def debt : ℕ := 40
def lulu_savings : ℕ := 6
def nora_savings : ℕ := 5 * lulu_savings
def remaining_per_person : ℕ := 2

theorem savings_ratio (tamara_savings : ℕ) 
  (h1 : nora_savings + lulu_savings + tamara_savings = debt + 3 * remaining_per_person) :
  nora_savings / tamara_savings = 3 := by
  sorry

end NUMINAMATH_CALUDE_savings_ratio_l22_2289


namespace NUMINAMATH_CALUDE_express_y_in_terms_of_x_l22_2235

theorem express_y_in_terms_of_x (n : ℕ) (x y : ℝ) : 
  x = 3^n → y = 2 + 9^n → y = 2 + x^2 := by
sorry

end NUMINAMATH_CALUDE_express_y_in_terms_of_x_l22_2235


namespace NUMINAMATH_CALUDE_percent_difference_l22_2295

theorem percent_difference (y u w z : ℝ) 
  (hw : w = 0.6 * u) 
  (hu : u = 0.6 * y) 
  (hz : z = 0.54 * y) : 
  (z - w) / w = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_percent_difference_l22_2295


namespace NUMINAMATH_CALUDE_smallest_cube_divisible_by_primes_l22_2245

theorem smallest_cube_divisible_by_primes (p q r : ℕ) : 
  Prime p → Prime q → Prime r → p ≠ q → p ≠ r → q ≠ r → p ≠ 1 → q ≠ 1 → r ≠ 1 →
  (∀ m : ℕ, m > 0 → (p^2 * q^3 * r^4) ∣ m → m = m^3 → m ≥ (p^2 * q^2 * r^2)^3) ∧
  (p^2 * q^3 * r^4) ∣ (p^2 * q^2 * r^2)^3 ∧
  ((p^2 * q^2 * r^2)^3)^(1/3) = p^2 * q^2 * r^2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_cube_divisible_by_primes_l22_2245


namespace NUMINAMATH_CALUDE_tommy_makes_twelve_loaves_l22_2279

/-- Represents the number of loaves Tommy can make given the flour costs and his budget --/
def tommys_loaves (flour_per_loaf : ℝ) (small_bag_weight : ℝ) (small_bag_cost : ℝ) 
  (large_bag_weight : ℝ) (large_bag_cost : ℝ) (budget : ℝ) : ℕ :=
  sorry

/-- Theorem stating that Tommy can make 12 loaves of bread --/
theorem tommy_makes_twelve_loaves :
  tommys_loaves 4 10 10 12 13 50 = 12 := by
  sorry

end NUMINAMATH_CALUDE_tommy_makes_twelve_loaves_l22_2279


namespace NUMINAMATH_CALUDE_last_digit_sum_l22_2240

theorem last_digit_sum (n : ℕ) : (3^1991 + 1991^3) % 10 = 8 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_sum_l22_2240


namespace NUMINAMATH_CALUDE_gcd_119_34_l22_2256

theorem gcd_119_34 : Nat.gcd 119 34 = 17 := by
  sorry

end NUMINAMATH_CALUDE_gcd_119_34_l22_2256


namespace NUMINAMATH_CALUDE_P_greater_than_Q_l22_2212

theorem P_greater_than_Q : 
  let P : ℝ := Real.sqrt 7 - 1
  let Q : ℝ := Real.sqrt 11 - Real.sqrt 5
  P > Q := by sorry

end NUMINAMATH_CALUDE_P_greater_than_Q_l22_2212


namespace NUMINAMATH_CALUDE_no_geometric_progression_with_11_12_13_l22_2227

theorem no_geometric_progression_with_11_12_13 :
  ¬ ∃ (a q : ℝ) (k l n : ℕ), 
    (k < l ∧ l < n) ∧
    (a * q ^ k = 11) ∧
    (a * q ^ l = 12) ∧
    (a * q ^ n = 13) :=
sorry

end NUMINAMATH_CALUDE_no_geometric_progression_with_11_12_13_l22_2227


namespace NUMINAMATH_CALUDE_teacher_age_l22_2204

/-- Proves that given a class of 50 students with an average age of 18 years, 
    if including the teacher's age changes the average to 19.5 years, 
    then the teacher's age is 94.5 years. -/
theorem teacher_age (num_students : ℕ) (student_avg_age : ℝ) (new_avg_age : ℝ) :
  num_students = 50 →
  student_avg_age = 18 →
  new_avg_age = 19.5 →
  (num_students * student_avg_age + (new_avg_age * (num_students + 1) - num_students * student_avg_age)) = 94.5 :=
by
  sorry

#check teacher_age

end NUMINAMATH_CALUDE_teacher_age_l22_2204


namespace NUMINAMATH_CALUDE_radical_axis_existence_l22_2261

structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def power (p : ℝ × ℝ) (c : Circle) : ℝ :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 - c.radius^2

def radical_axis (c1 c2 : Circle) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | power p c1 = power p c2}

def intersect (c1 c2 : Circle) : Prop :=
  ∃ p : ℝ × ℝ, power p c1 = 0 ∧ power p c2 = 0

def line_of_centers (c1 c2 : Circle) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ t : ℝ, p = (t * c1.center.1 + (1 - t) * c2.center.1, 
                           t * c1.center.2 + (1 - t) * c2.center.2)}

def perpendicular (l1 l2 : Set (ℝ × ℝ)) : Prop :=
  ∃ p : ℝ × ℝ, p ∈ l1 ∧ p ∈ l2 ∧
    ∀ q r : ℝ × ℝ, q ∈ l1 → r ∈ l2 → 
      (q.1 - p.1) * (r.1 - p.1) + (q.2 - p.2) * (r.2 - p.2) = 0

theorem radical_axis_existence (c1 c2 : Circle) :
  (intersect c1 c2 → 
    ∃ p1 p2 : ℝ × ℝ, p1 ≠ p2 ∧ power p1 c1 = 0 ∧ power p1 c2 = 0 ∧
                     power p2 c1 = 0 ∧ power p2 c2 = 0 ∧
                     radical_axis c1 c2 = {p : ℝ × ℝ | ∃ t : ℝ, p = (t * p1.1 + (1 - t) * p2.1, 
                                                                   t * p1.2 + (1 - t) * p2.2)}) ∧
  (¬intersect c1 c2 → 
    ∃ c3 : Circle, intersect c1 c3 ∧ intersect c2 c3 ∧
    ∃ p : ℝ × ℝ, power p c1 = power p c2 ∧ power p c2 = power p c3 ∧
    perpendicular (radical_axis c1 c2) (line_of_centers c1 c2) ∧
    p ∈ radical_axis c1 c2) :=
sorry

end NUMINAMATH_CALUDE_radical_axis_existence_l22_2261


namespace NUMINAMATH_CALUDE_smallest_difference_is_one_l22_2276

/-- Triangle with integer side lengths and specific ordering -/
structure OrderedTriangle where
  de : ℕ
  ef : ℕ
  fd : ℕ
  de_lt_ef : de < ef
  ef_le_fd : ef ≤ fd

/-- The perimeter of the triangle is 2050 -/
def hasPerimeter2050 (t : OrderedTriangle) : Prop :=
  t.de + t.ef + t.fd = 2050

/-- The triangle inequality holds -/
def satisfiesTriangleInequality (t : OrderedTriangle) : Prop :=
  t.de + t.ef > t.fd ∧ t.ef + t.fd > t.de ∧ t.fd + t.de > t.ef

theorem smallest_difference_is_one :
  ∃ (t : OrderedTriangle), 
    hasPerimeter2050 t ∧ 
    satisfiesTriangleInequality t ∧
    (∀ (u : OrderedTriangle), 
      hasPerimeter2050 u → satisfiesTriangleInequality u → 
      u.ef - u.de ≥ t.ef - t.de) ∧
    t.ef - t.de = 1 :=
  sorry

end NUMINAMATH_CALUDE_smallest_difference_is_one_l22_2276


namespace NUMINAMATH_CALUDE_investment_growth_l22_2282

/-- Compound interest calculation -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- Proof that $30,697 grows to at least $50,000 in 10 years at 5% interest -/
theorem investment_growth :
  let initial_deposit : ℝ := 30697
  let interest_rate : ℝ := 0.05
  let years : ℕ := 10
  let target_amount : ℝ := 50000
  compound_interest initial_deposit interest_rate years ≥ target_amount :=
by
  sorry

#check investment_growth

end NUMINAMATH_CALUDE_investment_growth_l22_2282


namespace NUMINAMATH_CALUDE_complex_fraction_sum_complex_product_imaginary_l22_2260

-- Problem 1
theorem complex_fraction_sum : (1 / (1 - Complex.I)) + (1 / (2 + 3 * Complex.I)) = Complex.mk (17/26) (7/26) := by sorry

-- Problem 2
theorem complex_product_imaginary (z₁ z₂ : ℂ) :
  z₁ = Complex.mk 3 4 →
  Complex.abs z₂ = 5 →
  (Complex.re (z₁ * z₂) = 0 ∧ Complex.im (z₁ * z₂) ≠ 0) →
  z₂ = Complex.mk 4 3 ∨ z₂ = Complex.mk (-4) (-3) := by sorry

end NUMINAMATH_CALUDE_complex_fraction_sum_complex_product_imaginary_l22_2260


namespace NUMINAMATH_CALUDE_knights_in_gamma_quarter_l22_2237

/-- Represents a resident of the town -/
inductive Resident
| Knight
| Liar

/-- The total number of residents in the town -/
def total_residents : ℕ := 200

/-- The total number of affirmative answers received -/
def total_affirmative_answers : ℕ := 430

/-- The number of affirmative answers received in quarter Γ -/
def gamma_quarter_affirmative_answers : ℕ := 119

/-- The number of affirmative answers a knight gives to every four questions -/
def knight_affirmative_rate : ℚ := 1/4

/-- The number of affirmative answers a liar gives to every four questions -/
def liar_affirmative_rate : ℚ := 3/4

/-- The total number of liars in the town -/
def total_liars : ℕ := (total_affirmative_answers - total_residents) / 2

theorem knights_in_gamma_quarter : 
  ∃ (k : ℕ), k = 4 ∧ 
  k ≤ gamma_quarter_affirmative_answers ∧
  (gamma_quarter_affirmative_answers - k : ℤ) = total_liars - (k - 4 : ℤ) ∧
  ∀ (other_quarter : ℕ), other_quarter ≠ gamma_quarter_affirmative_answers →
    (other_quarter : ℤ) - (total_residents - total_liars) > (total_residents - total_liars : ℤ) :=
sorry

end NUMINAMATH_CALUDE_knights_in_gamma_quarter_l22_2237


namespace NUMINAMATH_CALUDE_exists_m_between_alpha_beta_l22_2251

theorem exists_m_between_alpha_beta (α β : ℝ) (h1 : 0 ≤ α) (h2 : α < β) (h3 : β ≤ 1) :
  ∃ m : ℕ, α < (Nat.totient m : ℝ) / m ∧ (Nat.totient m : ℝ) / m < β := by
  sorry

end NUMINAMATH_CALUDE_exists_m_between_alpha_beta_l22_2251


namespace NUMINAMATH_CALUDE_bus_capacity_theorem_l22_2225

/-- Represents the capacity of a bus in terms of children -/
def bus_capacity (rows : ℕ) (children_per_row : ℕ) : ℕ :=
  rows * children_per_row

/-- Theorem stating that a bus with 9 rows and 4 children per row can accommodate 36 children -/
theorem bus_capacity_theorem :
  bus_capacity 9 4 = 36 := by
  sorry

end NUMINAMATH_CALUDE_bus_capacity_theorem_l22_2225


namespace NUMINAMATH_CALUDE_discriminant_positive_roots_difference_implies_m_values_l22_2209

-- Define the quadratic equation
def quadratic (m x : ℝ) : ℝ := x^2 + (m + 3) * x + m + 1

-- Define the discriminant of the quadratic equation
def discriminant (m : ℝ) : ℝ := (m + 3)^2 - 4 * (m + 1)

-- Theorem 1: The discriminant is always positive for any real m
theorem discriminant_positive (m : ℝ) : discriminant m > 0 := by sorry

-- Define the roots of the quadratic equation
noncomputable def α (m : ℝ) : ℝ := sorry
noncomputable def β (m : ℝ) : ℝ := sorry

-- Theorem 2: If α - β = 2√2, then m = -3 or m = 1
theorem roots_difference_implies_m_values (m : ℝ) (h : α m - β m = 2 * Real.sqrt 2) : 
  m = -3 ∨ m = 1 := by sorry

end NUMINAMATH_CALUDE_discriminant_positive_roots_difference_implies_m_values_l22_2209


namespace NUMINAMATH_CALUDE_line_ellipse_no_intersection_l22_2286

/-- Given a line y = 2x + b and an ellipse x^2/4 + y^2 = 1,
    if the line has no point in common with the ellipse,
    then b < -2√2 or b > 2√2 -/
theorem line_ellipse_no_intersection (b : ℝ) : 
  (∀ x y : ℝ, y = 2*x + b → x^2/4 + y^2 ≠ 1) → 
  (b < -2 * Real.sqrt 2 ∨ b > 2 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_line_ellipse_no_intersection_l22_2286


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_two_sqrt_31_l22_2283

theorem sqrt_sum_equals_two_sqrt_31 :
  Real.sqrt (24 - 10 * Real.sqrt 5) + Real.sqrt (24 + 10 * Real.sqrt 5) = 2 * Real.sqrt 31 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_two_sqrt_31_l22_2283


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_equation_l22_2250

theorem negation_of_existence (P : ℝ → Prop) : 
  (¬ ∃ x : ℝ, P x) ↔ (∀ x : ℝ, ¬ P x) :=
by sorry

theorem negation_of_quadratic_equation : 
  (¬ ∃ x : ℝ, x^2 + 2*x - 8 = 0) ↔ (∀ x : ℝ, x^2 + 2*x - 8 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_equation_l22_2250


namespace NUMINAMATH_CALUDE_mod_twelve_six_eight_l22_2274

theorem mod_twelve_six_eight (m : ℕ) : 12^6 ≡ m [ZMOD 8] → 0 ≤ m → m < 8 → m = 0 := by
  sorry

end NUMINAMATH_CALUDE_mod_twelve_six_eight_l22_2274


namespace NUMINAMATH_CALUDE_f_derivative_at_one_l22_2243

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x + x

theorem f_derivative_at_one :
  deriv f 1 = 3 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_one_l22_2243


namespace NUMINAMATH_CALUDE_smallest_abs_rational_l22_2244

theorem smallest_abs_rational : ∀ q : ℚ, |0| ≤ |q| := by
  sorry

end NUMINAMATH_CALUDE_smallest_abs_rational_l22_2244


namespace NUMINAMATH_CALUDE_short_trees_after_planting_l22_2249

/-- The number of short trees in a park after planting new trees -/
def total_short_trees (initial_short_trees newly_planted_short_trees : ℕ) : ℕ :=
  initial_short_trees + newly_planted_short_trees

/-- Theorem stating that the total number of short trees after planting
    is equal to the sum of initial short trees and newly planted short trees -/
theorem short_trees_after_planting
  (initial_short_trees : ℕ)
  (initial_tall_trees : ℕ)
  (newly_planted_short_trees : ℕ) :
  total_short_trees initial_short_trees newly_planted_short_trees =
  initial_short_trees + newly_planted_short_trees :=
by
  sorry

/-- Example calculation for the specific problem -/
def park_short_trees : ℕ :=
  total_short_trees 3 9

#eval park_short_trees

end NUMINAMATH_CALUDE_short_trees_after_planting_l22_2249


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l22_2268

/-- The perimeter of a rhombus with diagonals of 8 inches and 30 inches is 4√241 inches. -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 8) (h2 : d2 = 30) : 
  4 * Real.sqrt ((d1/2)^2 + (d2/2)^2) = 4 * Real.sqrt 241 :=
by sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l22_2268


namespace NUMINAMATH_CALUDE_intersection_of_P_and_Q_l22_2257

def P : Set ℝ := {1, 3, 5, 7}
def Q : Set ℝ := {x | 2 * x - 1 > 5}

theorem intersection_of_P_and_Q : P ∩ Q = {5, 7} := by sorry

end NUMINAMATH_CALUDE_intersection_of_P_and_Q_l22_2257


namespace NUMINAMATH_CALUDE_sandra_beignet_consumption_l22_2269

/-- The number of beignets Sandra eats per day -/
def daily_beignets : ℕ := 3

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- The number of weeks -/
def num_weeks : ℕ := 16

/-- The total number of beignets Sandra eats in the given period -/
def total_beignets : ℕ := daily_beignets * days_per_week * num_weeks

theorem sandra_beignet_consumption :
  total_beignets = 336 := by sorry

end NUMINAMATH_CALUDE_sandra_beignet_consumption_l22_2269


namespace NUMINAMATH_CALUDE_x_value_l22_2206

theorem x_value (x y z : ℚ) : x = y / 3 → y = z / 4 → z = 80 → x = 20 / 3 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l22_2206


namespace NUMINAMATH_CALUDE_glass_pane_area_is_4900_l22_2254

/-- The area of a square glass pane inside a square frame -/
def glass_pane_area (frame_side_length : ℝ) (frame_width : ℝ) : ℝ :=
  (frame_side_length - 2 * frame_width) ^ 2

/-- Theorem: The area of the square glass pane is 4900 cm² -/
theorem glass_pane_area_is_4900 :
  glass_pane_area 100 15 = 4900 := by
  sorry

end NUMINAMATH_CALUDE_glass_pane_area_is_4900_l22_2254


namespace NUMINAMATH_CALUDE_additional_water_needed_l22_2284

/-- Represents the capacity of a tank in liters -/
def TankCapacity : ℝ := 1000

/-- Represents the volume of water in the first tank in liters -/
def FirstTankVolume : ℝ := 300

/-- Represents the volume of water in the second tank in liters -/
def SecondTankVolume : ℝ := 450

/-- Represents the percentage of the second tank that is filled -/
def SecondTankPercentage : ℝ := 0.45

theorem additional_water_needed : 
  let remaining_first := TankCapacity - FirstTankVolume
  let remaining_second := TankCapacity - SecondTankVolume
  remaining_first + remaining_second = 1250 := by sorry

end NUMINAMATH_CALUDE_additional_water_needed_l22_2284


namespace NUMINAMATH_CALUDE_second_divisor_problem_l22_2214

theorem second_divisor_problem :
  ∃! D : ℤ, 19 < D ∧ D < 242 ∧
  (∃ N : ℤ, N % 242 = 100 ∧ N % D = 19) ∧
  D = 27 := by
sorry

end NUMINAMATH_CALUDE_second_divisor_problem_l22_2214


namespace NUMINAMATH_CALUDE_solution_product_l22_2252

theorem solution_product (p q : ℝ) : 
  p ≠ q ∧ 
  (p - 7) * (3 * p + 11) = p^2 - 20 * p + 63 ∧ 
  (q - 7) * (3 * q + 11) = q^2 - 20 * q + 63 →
  (p + 2) * (q + 2) = -72 := by
sorry

end NUMINAMATH_CALUDE_solution_product_l22_2252


namespace NUMINAMATH_CALUDE_marks_quiz_goal_l22_2262

theorem marks_quiz_goal (total_quizzes : ℕ) (goal_percentage : ℚ) 
  (completed_quizzes : ℕ) (as_earned : ℕ) (h1 : total_quizzes = 60) 
  (h2 : goal_percentage = 85 / 100) (h3 : completed_quizzes = 40) 
  (h4 : as_earned = 30) : 
  Nat.ceil (↑total_quizzes * goal_percentage) - as_earned ≥ total_quizzes - completed_quizzes := by
  sorry

end NUMINAMATH_CALUDE_marks_quiz_goal_l22_2262


namespace NUMINAMATH_CALUDE_jims_investment_l22_2287

/-- 
Given an investment scenario with three investors and a total investment,
calculate the investment amount for one specific investor.
-/
theorem jims_investment
  (total_investment : ℕ) 
  (john_ratio : ℕ) 
  (james_ratio : ℕ) 
  (jim_ratio : ℕ) 
  (h1 : total_investment = 80000)
  (h2 : john_ratio = 4)
  (h3 : james_ratio = 7)
  (h4 : jim_ratio = 9) : 
  jim_ratio * (total_investment / (john_ratio + james_ratio + jim_ratio)) = 36000 := by
  sorry

end NUMINAMATH_CALUDE_jims_investment_l22_2287


namespace NUMINAMATH_CALUDE_remainder_problem_l22_2290

theorem remainder_problem (N : ℤ) (h : N % 296 = 75) : N % 37 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l22_2290


namespace NUMINAMATH_CALUDE_quadratic_roots_difference_l22_2203

theorem quadratic_roots_difference (a b c : ℝ) (h : a ≠ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  a = 1 ∧ b = 30 ∧ c = 225 → |r₁ - r₂| = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_difference_l22_2203


namespace NUMINAMATH_CALUDE_product_of_four_integers_l22_2272

theorem product_of_four_integers (P Q R S : ℕ+) : 
  (P : ℚ) + (Q : ℚ) + (R : ℚ) + (S : ℚ) = 50 →
  (P : ℚ) + 4 = (Q : ℚ) - 4 ∧ 
  (P : ℚ) + 4 = (R : ℚ) * 3 ∧ 
  (P : ℚ) + 4 = (S : ℚ) / 3 →
  (P : ℚ) * (Q : ℚ) * (R : ℚ) * (S : ℚ) = (43 * 107 * 75 * 225) / 1536 := by
  sorry

#check product_of_four_integers

end NUMINAMATH_CALUDE_product_of_four_integers_l22_2272


namespace NUMINAMATH_CALUDE_evaluate_expression_l22_2299

theorem evaluate_expression (x y z : ℚ) (hx : x = 1/4) (hy : y = 2/3) (hz : z = -3) :
  x^3 * y^2 * z^2 = 1/16 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l22_2299


namespace NUMINAMATH_CALUDE_identity_function_unique_l22_2259

def C : ℕ := 2022^2022

theorem identity_function_unique :
  ∀ f : ℕ → ℕ,
  (∀ x y : ℕ, x > 0 → y > 0 → 
    ∃ k : ℕ, k > 0 ∧ k ≤ C ∧ f (x + y) = f x + k * f y) →
  f = id :=
by sorry

end NUMINAMATH_CALUDE_identity_function_unique_l22_2259


namespace NUMINAMATH_CALUDE_peytons_children_l22_2202

theorem peytons_children (juice_boxes_per_week : ℕ) (weeks_in_school_year : ℕ) (total_juice_boxes : ℕ) : 
  juice_boxes_per_week = 5 → 
  weeks_in_school_year = 25 → 
  total_juice_boxes = 375 → 
  total_juice_boxes / (juice_boxes_per_week * weeks_in_school_year) = 3 := by
sorry

end NUMINAMATH_CALUDE_peytons_children_l22_2202


namespace NUMINAMATH_CALUDE_original_number_of_people_l22_2207

theorem original_number_of_people (x : ℕ) : 
  (x / 3 : ℚ) - (x / 3 : ℚ) / 4 = 15 → x = 60 := by
  sorry

end NUMINAMATH_CALUDE_original_number_of_people_l22_2207


namespace NUMINAMATH_CALUDE_base_conversion_512_l22_2221

/-- Converts a base-10 number to its base-6 representation -/
def toBase6 (n : ℕ) : List ℕ :=
  sorry

theorem base_conversion_512 :
  toBase6 512 = [2, 2, 1, 2] :=
sorry

end NUMINAMATH_CALUDE_base_conversion_512_l22_2221


namespace NUMINAMATH_CALUDE_condition_satisfied_pairs_l22_2231

/-- Checks if a pair of positive integers (m, n) satisfies the given condition -/
def satisfies_condition (m n : ℕ+) : Prop :=
  ∀ x y : ℝ, m ≤ x ∧ x ≤ n ∧ m ≤ y ∧ y ≤ n → m ≤ (5/x + 7/y) ∧ (5/x + 7/y) ≤ n

/-- The only positive integer pairs (m, n) satisfying the condition are (1,12), (2,6), and (3,4) -/
theorem condition_satisfied_pairs :
  ∀ m n : ℕ+, satisfies_condition m n ↔ (m = 1 ∧ n = 12) ∨ (m = 2 ∧ n = 6) ∨ (m = 3 ∧ n = 4) :=
sorry

end NUMINAMATH_CALUDE_condition_satisfied_pairs_l22_2231


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_formula_l22_2219

/-- An arithmetic-geometric sequence -/
def ArithGeomSeq (a : ℕ → ℝ) : Prop :=
  ∃ (q : ℝ), ∀ n, a (n + 1) = a n * q

/-- The general term formula for the sequence -/
def GeneralTerm (a : ℕ → ℝ) : Prop :=
  (∃ (c : ℝ), ∀ n, a n = 125 * (2/5)^(n-1)) ∨
  (∃ (c : ℝ), ∀ n, a n = 8 * (5/2)^(n-1))

theorem arithmetic_geometric_sequence_formula (a : ℕ → ℝ) 
  (h1 : ArithGeomSeq a)
  (h2 : a 1 + a 4 = 133)
  (h3 : a 2 + a 3 = 70) :
  GeneralTerm a :=
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_formula_l22_2219


namespace NUMINAMATH_CALUDE_compound_interest_calculation_l22_2267

/-- Calculates the total amount and interest earned for a compound interest investment --/
theorem compound_interest_calculation 
  (initial_investment : ℝ) 
  (annual_rate : ℝ) 
  (years : ℕ) 
  (h1 : initial_investment = 500)
  (h2 : annual_rate = 0.02)
  (h3 : years = 3) :
  ∃ (total_amount interest_earned : ℝ),
    total_amount = initial_investment * (1 + annual_rate) ^ years ∧
    interest_earned = total_amount - initial_investment ∧
    (⌊total_amount⌋ : ℤ) = 531 ∧
    (⌊interest_earned⌋ : ℤ) = 31 := by
  sorry

end NUMINAMATH_CALUDE_compound_interest_calculation_l22_2267


namespace NUMINAMATH_CALUDE_moose_population_canada_l22_2220

/-- The moose population in Canada, given the ratio of moose to beavers to humans -/
theorem moose_population_canada (total_humans : ℕ) (moose_to_beaver : ℕ) (beaver_to_human : ℕ) :
  total_humans = 38000000 →
  moose_to_beaver = 2 →
  beaver_to_human = 19 →
  (total_humans / (moose_to_beaver * beaver_to_human) : ℚ) = 1000000 := by
  sorry

#check moose_population_canada

end NUMINAMATH_CALUDE_moose_population_canada_l22_2220
