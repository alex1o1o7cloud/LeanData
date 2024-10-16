import Mathlib

namespace NUMINAMATH_CALUDE_triangle_inequality_l3031_303175

/-- Given a triangle with side lengths a, b, and c, 
    the inequality a^2 b(a-b) + b^2 c(b-c) + c^2 a(c-a) ≥ 0 holds. -/
theorem triangle_inequality (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_l3031_303175


namespace NUMINAMATH_CALUDE_mod_product_241_398_l3031_303187

theorem mod_product_241_398 (m : ℕ) : 
  (241 * 398 ≡ m [ZMOD 50]) → 
  0 ≤ m → m < 50 → 
  m = 18 := by sorry

end NUMINAMATH_CALUDE_mod_product_241_398_l3031_303187


namespace NUMINAMATH_CALUDE_existence_equivalence_l3031_303160

theorem existence_equivalence (a b : ℤ) :
  (∃ c d : ℤ, a + b + c + d = 0 ∧ a * c + b * d = 0) ↔ (∃ k : ℤ, 2 * a * b = k * (a - b)) :=
by sorry

end NUMINAMATH_CALUDE_existence_equivalence_l3031_303160


namespace NUMINAMATH_CALUDE_birthday_money_theorem_l3031_303190

def birthday_money_problem (initial_amount : ℚ) (video_game_fraction : ℚ) (goggles_fraction : ℚ) : ℚ :=
  let remaining_after_game := initial_amount * (1 - video_game_fraction)
  remaining_after_game * (1 - goggles_fraction)

theorem birthday_money_theorem :
  birthday_money_problem 100 (1/4) (1/5) = 60 := by
  sorry

end NUMINAMATH_CALUDE_birthday_money_theorem_l3031_303190


namespace NUMINAMATH_CALUDE_three_digit_number_property_l3031_303178

theorem three_digit_number_property (A : ℕ) : 
  100 ≤ A → A < 1000 → 
  let B := 1001 * A
  (((B / 7) / 11) / 13) = A := by
  sorry

end NUMINAMATH_CALUDE_three_digit_number_property_l3031_303178


namespace NUMINAMATH_CALUDE_cube_sum_divisible_implies_product_divisible_l3031_303138

theorem cube_sum_divisible_implies_product_divisible (a b c : ℤ) :
  7 ∣ (a^3 + b^3 + c^3) → 7 ∣ (a * b * c) := by
sorry

end NUMINAMATH_CALUDE_cube_sum_divisible_implies_product_divisible_l3031_303138


namespace NUMINAMATH_CALUDE_optimal_labeled_price_l3031_303131

/-- Represents the pricing strategy of a retailer --/
structure RetailPricing where
  list_price : ℝ
  purchase_discount : ℝ
  sale_discount : ℝ
  profit_margin : ℝ
  labeled_price : ℝ

/-- The pricing strategy satisfies the retailer's conditions --/
def satisfies_conditions (rp : RetailPricing) : Prop :=
  rp.purchase_discount = 0.3 ∧
  rp.sale_discount = 0.25 ∧
  rp.profit_margin = 0.3 ∧
  rp.labeled_price > 0 ∧
  rp.list_price > 0

/-- The final selling price after discount --/
def selling_price (rp : RetailPricing) : ℝ :=
  rp.labeled_price * (1 - rp.sale_discount)

/-- The purchase price for the retailer --/
def purchase_price (rp : RetailPricing) : ℝ :=
  rp.list_price * (1 - rp.purchase_discount)

/-- The profit calculation --/
def profit (rp : RetailPricing) : ℝ :=
  selling_price rp - purchase_price rp

/-- The theorem stating that the labeled price should be 135% of the list price --/
theorem optimal_labeled_price (rp : RetailPricing) 
  (h : satisfies_conditions rp) : 
  rp.labeled_price = 1.35 * rp.list_price ↔ 
  profit rp = rp.profit_margin * selling_price rp :=
sorry

end NUMINAMATH_CALUDE_optimal_labeled_price_l3031_303131


namespace NUMINAMATH_CALUDE_max_value_y_plus_one_squared_l3031_303171

theorem max_value_y_plus_one_squared (y : ℝ) : 
  (4 * y^2 + 4 * y + 3 = 1) → ((y + 1)^2 ≤ (1/4 : ℝ)) ∧ (∃ y : ℝ, 4 * y^2 + 4 * y + 3 = 1 ∧ (y + 1)^2 = (1/4 : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_max_value_y_plus_one_squared_l3031_303171


namespace NUMINAMATH_CALUDE_sequence_inequality_l3031_303176

theorem sequence_inequality (a : Fin 9 → ℝ) 
  (h_nonneg : ∀ i, 0 ≤ a i) 
  (h_first : a 0 = 0) 
  (h_last : a 8 = 0) 
  (h_nonzero : ∃ i, a i ≠ 0) : 
  (∃ i : Fin 9, 1 < i.val ∧ i.val < 9 ∧ a (i - 1) + a (i + 1) < 2 * a i) ∧
  (∃ i : Fin 9, 1 < i.val ∧ i.val < 9 ∧ a (i - 1) + a (i + 1) < 1.9 * a i) :=
sorry

end NUMINAMATH_CALUDE_sequence_inequality_l3031_303176


namespace NUMINAMATH_CALUDE_value_of_a_l3031_303123

theorem value_of_a : 
  let a := Real.sqrt ((19.19^2) + (39.19^2) - (38.38 * 39.19))
  a = 20 := by sorry

end NUMINAMATH_CALUDE_value_of_a_l3031_303123


namespace NUMINAMATH_CALUDE_y_intercept_of_line_l3031_303111

/-- The y-intercept of the line 6x - 4y = 24 is (0, -6) -/
theorem y_intercept_of_line (x y : ℝ) : 
  (6 * x - 4 * y = 24) → (x = 0 → y = -6) := by
  sorry

end NUMINAMATH_CALUDE_y_intercept_of_line_l3031_303111


namespace NUMINAMATH_CALUDE_max_distance_ratio_l3031_303110

/-- The maximum ratio of distances from a point on a circle to two fixed points -/
theorem max_distance_ratio : 
  let A : ℝ × ℝ := (0, -2)
  let B : ℝ × ℝ := (1, -1)
  let circle := {P : ℝ × ℝ | P.1^2 + P.2^2 = 2}
  ∃ (max : ℝ), max = (3 * Real.sqrt 2) / 2 ∧ 
    ∀ P ∈ circle, 
      Real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2) / Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2) ≤ max :=
by sorry


end NUMINAMATH_CALUDE_max_distance_ratio_l3031_303110


namespace NUMINAMATH_CALUDE_amys_house_height_l3031_303158

/-- The height of Amy's house given shadow lengths -/
theorem amys_house_height (house_shadow : ℝ) (tree_height : ℝ) (tree_shadow : ℝ)
  (h1 : house_shadow = 63)
  (h2 : tree_height = 14)
  (h3 : tree_shadow = 28)
  : ∃ (house_height : ℝ), 
    (house_height / tree_height = house_shadow / tree_shadow) ∧ 
    (round house_height = 32) := by
  sorry

end NUMINAMATH_CALUDE_amys_house_height_l3031_303158


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3031_303165

-- Define set A
def A : Set ℝ := {x | x^2 - 2*x ≤ 0}

-- Define set B
def B : Set ℝ := {x | ∃ y, y = Real.log (1 - x)}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 0 ≤ x ∧ x < 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3031_303165


namespace NUMINAMATH_CALUDE_exponent_relations_l3031_303132

theorem exponent_relations (a : ℝ) (m n k : ℤ) 
  (h1 : a ≠ 0) 
  (h2 : a^m = 2) 
  (h3 : a^n = 4) 
  (h4 : a^k = 32) : 
  (a^(3*m + 2*n - k) = 4) ∧ (k - 3*m - n = 0) := by
  sorry

end NUMINAMATH_CALUDE_exponent_relations_l3031_303132


namespace NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l3031_303122

theorem geometric_sequence_fifth_term 
  (t : ℕ → ℝ) 
  (h_positive : ∀ n, t n > 0) 
  (h_decreasing : t 1 > t 2) 
  (h_sum : t 1 + t 2 = 15/2) 
  (h_sum_squares : t 1^2 + t 2^2 = 153/4) 
  (h_geometric : ∃ r : ℝ, ∀ n, t (n+1) = t n * r) :
  t 5 = 3/128 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l3031_303122


namespace NUMINAMATH_CALUDE_johns_allowance_l3031_303106

/-- John's weekly allowance problem -/
theorem johns_allowance (A : ℚ) : A = 32.4 ↔
  ∃ (arcade toy book candy : ℚ),
    -- Spending at the arcade
    arcade = 7 / 12 * A ∧
    -- Spending at the toy store
    toy = 5 / 9 * (A - arcade) ∧
    -- Spending at the bookstore
    book = 3 / 4 * (A - arcade - toy) ∧
    -- Spending at the candy store
    candy = 3 / 2 ∧
    -- Total spending equals the allowance
    arcade + toy + book + candy = A := by
  sorry

end NUMINAMATH_CALUDE_johns_allowance_l3031_303106


namespace NUMINAMATH_CALUDE_stratified_sample_sum_l3031_303114

/-- Represents the number of items in each category -/
def categories : List ℕ := [40, 10, 30, 20]

/-- Total number of items -/
def total : ℕ := categories.sum

/-- Sample size -/
def sample_size : ℕ := 20

/-- Calculates the number of items sampled from a category -/
def sampled_items (category_size : ℕ) : ℕ :=
  (category_size * sample_size) / total

/-- Theorem stating that the sum of sampled items from categories with 10 and 20 items is 6 -/
theorem stratified_sample_sum :
  sampled_items (categories[1]) + sampled_items (categories[3]) = 6 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_sum_l3031_303114


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_is_one_seventh_l3031_303152

noncomputable def ellipse_eccentricity (a : ℝ) : ℝ :=
  let b := Real.sqrt 3
  let c := (1 : ℝ) / 4
  c / a

theorem ellipse_eccentricity_is_one_seventh :
  ∃ a : ℝ, (a > 0) ∧ 
  ((1 : ℝ) / 4)^2 / a^2 + (0 : ℝ)^2 / 3 = 1 ∧
  ellipse_eccentricity a = (1 : ℝ) / 7 := by
sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_is_one_seventh_l3031_303152


namespace NUMINAMATH_CALUDE_breakfast_expectation_l3031_303191

/-- Represents the possible outcomes of rolling a fair six-sided die, excluding 1 (which leads to a reroll) -/
inductive DieOutcome
| two
| three
| four
| five
| six

/-- The probability of rolling an even number (2, 4, or 6) after accounting for rerolls on 1 -/
def prob_even : ℚ := 3/5

/-- The probability of rolling an odd number (3 or 5) after accounting for rerolls on 1 -/
def prob_odd : ℚ := 2/5

/-- The number of days in a non-leap year -/
def days_in_year : ℕ := 365

/-- The expected difference between days eating pancakes and days eating oatmeal -/
def expected_difference : ℚ := prob_even * days_in_year - prob_odd * days_in_year

theorem breakfast_expectation :
  expected_difference = 73 := by sorry

end NUMINAMATH_CALUDE_breakfast_expectation_l3031_303191


namespace NUMINAMATH_CALUDE_age_difference_l3031_303125

/-- Given that the total age of A and B is 20 years more than the total age of B and C,
    prove that C is 20 years younger than A. -/
theorem age_difference (A B C : ℕ) (h : A + B = B + C + 20) : A = C + 20 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l3031_303125


namespace NUMINAMATH_CALUDE_ball_probability_l3031_303145

theorem ball_probability (m n : ℕ) : 
  (10 : ℝ) / (m + 10 + n : ℝ) = (m + n : ℝ) / (m + 10 + n : ℝ) → m + n = 10 := by
  sorry

end NUMINAMATH_CALUDE_ball_probability_l3031_303145


namespace NUMINAMATH_CALUDE_dividend_problem_l3031_303164

/-- Given a total amount of 585 to be divided among three people (a, b, c) such that
    4 times a's share equals 6 times b's share, which equals 3 times c's share,
    prove that c's share is equal to 135. -/
theorem dividend_problem (total : ℕ) (a b c : ℚ) 
    (h_total : total = 585)
    (h_sum : a + b + c = total)
    (h_prop : (4 * a = 6 * b) ∧ (6 * b = 3 * c)) :
  c = 135 := by
  sorry

end NUMINAMATH_CALUDE_dividend_problem_l3031_303164


namespace NUMINAMATH_CALUDE_max_third_altitude_l3031_303194

/-- A triangle with two known altitudes and one unknown integer altitude -/
structure TriangleWithAltitudes where
  /-- The length of the first known altitude -/
  h₁ : ℝ
  /-- The length of the second known altitude -/
  h₂ : ℝ
  /-- The length of the unknown altitude (assumed to be an integer) -/
  h₃ : ℤ
  /-- Condition that h₁ and h₂ are 3 and 9 (in either order) -/
  known_altitudes : (h₁ = 3 ∧ h₂ = 9) ∨ (h₁ = 9 ∧ h₂ = 3)

/-- The theorem stating that the maximum possible integer length for h₃ is 4 -/
theorem max_third_altitude (t : TriangleWithAltitudes) : t.h₃ ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_max_third_altitude_l3031_303194


namespace NUMINAMATH_CALUDE_right_triangle_third_side_length_l3031_303117

theorem right_triangle_third_side_length 
  (a b c : ℝ) 
  (ha : a = 5) 
  (hb : b = 13) 
  (hc : c * c = a * a + b * b) 
  (hright : a < b ∧ b > c) : c = 12 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_third_side_length_l3031_303117


namespace NUMINAMATH_CALUDE_even_sum_sufficient_not_necessary_l3031_303198

-- Define the concept of an even function
def IsEven (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x

-- Define the sum of two functions
def SumFunc (f g : ℝ → ℝ) : ℝ → ℝ := λ x ↦ f x + g x

theorem even_sum_sufficient_not_necessary :
  (∀ f g : ℝ → ℝ, IsEven f ∧ IsEven g → IsEven (SumFunc f g)) ∧ 
  (∃ f g : ℝ → ℝ, IsEven (SumFunc f g) ∧ ¬(IsEven f) ∧ ¬(IsEven g)) :=
sorry

end NUMINAMATH_CALUDE_even_sum_sufficient_not_necessary_l3031_303198


namespace NUMINAMATH_CALUDE_quarter_to_fourth_power_decimal_l3031_303147

theorem quarter_to_fourth_power_decimal : (1 / 4 : ℚ) ^ 4 = 0.00390625 := by
  sorry

end NUMINAMATH_CALUDE_quarter_to_fourth_power_decimal_l3031_303147


namespace NUMINAMATH_CALUDE_connectivity_determination_bound_l3031_303179

/-- A graph with n vertices -/
structure Graph (n : ℕ) where
  adj : Fin n → Fin n → Bool

/-- Distance between two vertices in a graph -/
def distance (G : Graph n) (u v : Fin n) : ℕ := sorry

/-- Whether a graph is connected -/
def is_connected (G : Graph n) : Prop := sorry

/-- A query about the distance between two vertices -/
structure Query (n : ℕ) where
  u : Fin n
  v : Fin n

/-- Result of a query -/
inductive QueryResult
  | LessThan
  | EqualTo
  | GreaterThan

/-- Function to determine if a graph is connected using queries -/
def determine_connectivity (n k : ℕ) (h : k ≤ n) (G : Graph n) : 
  ∃ (queries : List (Query n)), 
    queries.length ≤ 2 * n^2 / k ∧ 
    (∀ q : Query n, q ∈ queries → ∃ r : QueryResult, r = sorry) → 
    ∃ b : Bool, b = is_connected G := sorry

/-- Main theorem -/
theorem connectivity_determination_bound (n k : ℕ) (h : k ≤ n) :
  ∀ G : Graph n, ∃ (queries : List (Query n)), 
    queries.length ≤ 2 * n^2 / k ∧ 
    (∀ q : Query n, q ∈ queries → ∃ r : QueryResult, r = sorry) → 
    ∃ b : Bool, b = is_connected G := by
  sorry

end NUMINAMATH_CALUDE_connectivity_determination_bound_l3031_303179


namespace NUMINAMATH_CALUDE_f_properties_l3031_303185

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 1 then 1/4 + Real.log x / Real.log 4
  else 2^(-x) - 1/4

theorem f_properties :
  (∀ x, f x ≥ 1/4) ∧
  (∀ x, f x = 3/4 ↔ x = 0 ∨ x = 2) := by
sorry

end NUMINAMATH_CALUDE_f_properties_l3031_303185


namespace NUMINAMATH_CALUDE_shaded_area_is_14_l3031_303192

/-- Represents the grid dimensions --/
structure GridDimensions where
  width : ℕ
  height : ℕ

/-- Calculates the area of a rectangle --/
def rectangleArea (w h : ℕ) : ℕ := w * h

/-- Calculates the area of a right-angled triangle --/
def triangleArea (base height : ℕ) : ℕ := base * height / 2

/-- Theorem stating that the shaded area in the grid is 14 square units --/
theorem shaded_area_is_14 (grid : GridDimensions) 
    (h1 : grid.width = 12)
    (h2 : grid.height = 4) : 
  rectangleArea grid.width grid.height - triangleArea grid.width grid.height = 14 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_is_14_l3031_303192


namespace NUMINAMATH_CALUDE_quadratic_function_property_l3031_303150

/-- A quadratic function with real coefficients -/
def QuadraticFunction (a b : ℝ) : ℝ → ℝ := fun x ↦ x^2 + a*x + b

/-- The property that the range of a function is [0, +∞) -/
def HasNonnegativeRange (f : ℝ → ℝ) : Prop :=
  ∀ y, (∃ x, f x = y) → y ≥ 0

/-- The property that the solution set of f(x) < c is (m, m+8) -/
def HasSolutionSet (f : ℝ → ℝ) (c m : ℝ) : Prop :=
  ∀ x, f x < c ↔ m < x ∧ x < m + 8

theorem quadratic_function_property (a b c m : ℝ) :
  HasNonnegativeRange (QuadraticFunction a b) →
  HasSolutionSet (QuadraticFunction a b) c m →
  c = 16 := by sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l3031_303150


namespace NUMINAMATH_CALUDE_smallest_mu_inequality_l3031_303113

theorem smallest_mu_inequality (a b c d : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) (hd : d ≥ 0) :
  (∃ (μ : ℝ), ∀ (a b c d : ℝ), a ≥ 0 → b ≥ 0 → c ≥ 0 → d ≥ 0 → 
    a^2 + b^2 + c^2 + d^2 ≥ a*b + b*c + μ*c*d) ∧
  (∀ (μ : ℝ), (∀ (a b c d : ℝ), a ≥ 0 → b ≥ 0 → c ≥ 0 → d ≥ 0 → 
    a^2 + b^2 + c^2 + d^2 ≥ a*b + b*c + μ*c*d) → μ ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_smallest_mu_inequality_l3031_303113


namespace NUMINAMATH_CALUDE_square_binomial_constant_l3031_303116

/-- If x^2 + 50x + d is equal to the square of a binomial, then d = 625 -/
theorem square_binomial_constant (d : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, x^2 + 50*x + d = (x + b)^2) → d = 625 := by
  sorry

end NUMINAMATH_CALUDE_square_binomial_constant_l3031_303116


namespace NUMINAMATH_CALUDE_ln_abs_even_and_increasing_l3031_303112

noncomputable def f (x : ℝ) : ℝ := Real.log (abs x)

theorem ln_abs_even_and_increasing :
  (∀ x : ℝ, f x = f (-x)) ∧
  (∀ x y : ℝ, 0 < x → x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_ln_abs_even_and_increasing_l3031_303112


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l3031_303157

theorem cube_root_equation_solution (x : ℝ) (h : (3 - 1 / x^2)^(1/3) = -4) : 
  x = 1 / Real.sqrt 67 ∨ x = -1 / Real.sqrt 67 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l3031_303157


namespace NUMINAMATH_CALUDE_constant_covered_area_l3031_303128

/-- Represents a square in 2D space -/
structure Square where
  side_length : ℝ
  center : ℝ × ℝ

/-- Represents the configuration of two squares as described in the problem -/
structure TwoSquaresConfig where
  bottom_square : Square
  top_square : Square
  rotation_angle : ℝ

/-- Calculates the total area covered by two squares in the given configuration -/
noncomputable def total_covered_area (config : TwoSquaresConfig) : ℝ :=
  sorry

/-- Theorem: The total covered area is constant regardless of the rotation angle -/
theorem constant_covered_area
  (bottom_square : Square)
  (top_square : Square)
  (h_identical : bottom_square.side_length = top_square.side_length)
  (h_diagonal_intersection : top_square.center = (bottom_square.center.1 + bottom_square.side_length / 2, bottom_square.center.2 + bottom_square.side_length / 2)) :
  ∀ θ₁ θ₂ : ℝ,
    total_covered_area { bottom_square := bottom_square, top_square := top_square, rotation_angle := θ₁ } =
    total_covered_area { bottom_square := bottom_square, top_square := top_square, rotation_angle := θ₂ } :=
  sorry

end NUMINAMATH_CALUDE_constant_covered_area_l3031_303128


namespace NUMINAMATH_CALUDE_max_m_value_l3031_303135

/-- A quadratic function satisfying specific conditions -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0
  symmetry : ∀ x : ℝ, a * (x - 4)^2 + b * (x - 4) + c = a * (2 - x)^2 + b * (2 - x) + c
  inequality : ∀ x ∈ Set.Ioo 0 2, a * x^2 + b * x + c ≤ ((x + 1) / 2)^2
  min_value : ∃ x : ℝ, ∀ y : ℝ, a * x^2 + b * x + c ≤ a * y^2 + b * y + c ∧ a * x^2 + b * x + c = 0

/-- The theorem stating the maximum value of m -/
theorem max_m_value (f : QuadraticFunction) :
  ∃ m : ℝ, m = 9 ∧ m > 1 ∧
  (∀ m' > m, ¬∃ t : ℝ, ∀ x ∈ Set.Icc 1 m', f.a * (x + t)^2 + f.b * (x + t) + f.c ≤ x) ∧
  (∃ t : ℝ, ∀ x ∈ Set.Icc 1 m, f.a * (x + t)^2 + f.b * (x + t) + f.c ≤ x) :=
sorry

end NUMINAMATH_CALUDE_max_m_value_l3031_303135


namespace NUMINAMATH_CALUDE_inequality_holds_iff_k_geq_four_l3031_303119

theorem inequality_holds_iff_k_geq_four :
  ∀ k : ℝ, k > 0 →
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 →
    a / (b + c) + b / (c + a) + k * c / (a + b) ≥ 2) ↔
  k ≥ 4 := by sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_k_geq_four_l3031_303119


namespace NUMINAMATH_CALUDE_skylar_donation_amount_l3031_303107

theorem skylar_donation_amount (start_age : ℕ) (current_age : ℕ) (total_donation : ℕ) : 
  start_age = 13 →
  current_age = 33 →
  total_donation = 105000 →
  (total_donation : ℚ) / ((current_age - start_age) : ℚ) = 5250 := by
  sorry

end NUMINAMATH_CALUDE_skylar_donation_amount_l3031_303107


namespace NUMINAMATH_CALUDE_unique_b_value_l3031_303196

/-- The configuration of a circle and parabola with specific intersection properties -/
structure CircleParabolaConfig where
  b : ℝ
  circle_center : ℝ × ℝ
  parabola : ℝ → ℝ
  line : ℝ → ℝ
  intersect_origin : Bool
  intersect_line : Bool

/-- The theorem stating the unique value of b for the given configuration -/
theorem unique_b_value (config : CircleParabolaConfig) : 
  config.parabola = (λ x => (12/5) * x^2) →
  config.line = (λ x => (12/5) * x + config.b) →
  config.circle_center.2 = config.b →
  config.intersect_origin = true →
  config.intersect_line = true →
  config.b = 169/60 := by
  sorry

#check unique_b_value

end NUMINAMATH_CALUDE_unique_b_value_l3031_303196


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3031_303189

theorem quadratic_inequality_range (a : ℝ) : 
  (¬ ∀ x : ℝ, x^2 - 5*x + (5/4)*a > 0) ↔ a ≤ 5 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l3031_303189


namespace NUMINAMATH_CALUDE_class_average_mark_l3031_303133

theorem class_average_mark (students1 students2 : ℕ) (avg2 avg_combined : ℚ) 
  (h1 : students1 = 30)
  (h2 : students2 = 50)
  (h3 : avg2 = 70)
  (h4 : avg_combined = 58.75)
  (h5 : (students1 : ℚ) * x + (students2 : ℚ) * avg2 = ((students1 : ℚ) + (students2 : ℚ)) * avg_combined) :
  x = 40 := by
  sorry

end NUMINAMATH_CALUDE_class_average_mark_l3031_303133


namespace NUMINAMATH_CALUDE_median_room_number_l3031_303134

/-- Given a list of integers from 1 to n with two consecutive numbers removed,
    this function returns the median of the remaining numbers. -/
def medianWithGap (n : ℕ) (gap_start : ℕ) : ℕ :=
  if gap_start ≤ (n + 1) / 2
  then (n + 1) / 2 + 1
  else (n + 1) / 2

theorem median_room_number :
  medianWithGap 23 14 = 13 :=
by sorry

end NUMINAMATH_CALUDE_median_room_number_l3031_303134


namespace NUMINAMATH_CALUDE_monotone_increasing_condition_l3031_303168

/-- The function f(x) = kx - 2ln(x) is monotonically increasing on [1, +∞) iff k ≥ 2 -/
theorem monotone_increasing_condition (k : ℝ) :
  (∀ x ≥ 1, Monotone (λ x => k * x - 2 * Real.log x)) ↔ k ≥ 2 := by
sorry

end NUMINAMATH_CALUDE_monotone_increasing_condition_l3031_303168


namespace NUMINAMATH_CALUDE_cosine_sum_17_l3031_303181

theorem cosine_sum_17 : 
  Real.cos (2 * Real.pi / 17) + Real.cos (6 * Real.pi / 17) + Real.cos (8 * Real.pi / 17) = (Real.sqrt 13 - 1) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sum_17_l3031_303181


namespace NUMINAMATH_CALUDE_lawrence_county_kids_count_lawrence_county_total_kids_l3031_303143

theorem lawrence_county_kids_count : ℕ → ℕ → ℕ
  | kids_at_home, kids_at_camp =>
    kids_at_home + kids_at_camp

theorem lawrence_county_total_kids :
  lawrence_county_kids_count 907611 455682 = 1363293 := by
  sorry

end NUMINAMATH_CALUDE_lawrence_county_kids_count_lawrence_county_total_kids_l3031_303143


namespace NUMINAMATH_CALUDE_coordinate_sum_of_point_d_l3031_303184

/-- Given a point C at (0, 0) and a point D on the line y = 5,
    if the slope of segment CD is 3/4,
    then the sum of the x- and y-coordinates of point D is 35/3. -/
theorem coordinate_sum_of_point_d (D : ℝ × ℝ) : 
  D.2 = 5 →                  -- D is on the line y = 5
  (D.2 - 0) / (D.1 - 0) = 3/4 →  -- slope of CD is 3/4
  D.1 + D.2 = 35/3 := by
sorry

end NUMINAMATH_CALUDE_coordinate_sum_of_point_d_l3031_303184


namespace NUMINAMATH_CALUDE_units_digit_of_fraction_l3031_303170

theorem units_digit_of_fraction (n : ℕ) : n = 30 * 31 * 32 * 33 * 34 / 120 → n % 10 = 4 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_fraction_l3031_303170


namespace NUMINAMATH_CALUDE_mass_CO2_from_CO_combustion_l3031_303193

/-- The mass of CO2 produced from the complete combustion of CO -/
def mass_CO2_produced (initial_moles_CO : ℝ) (molar_mass_CO2 : ℝ) : ℝ :=
  initial_moles_CO * molar_mass_CO2

/-- The balanced chemical reaction coefficient for CO2 -/
def CO2_coefficient : ℚ := 2

/-- The balanced chemical reaction coefficient for CO -/
def CO_coefficient : ℚ := 2

theorem mass_CO2_from_CO_combustion 
  (initial_moles_CO : ℝ)
  (molar_mass_CO2 : ℝ)
  (h1 : initial_moles_CO = 3)
  (h2 : molar_mass_CO2 = 44.01) :
  mass_CO2_produced initial_moles_CO molar_mass_CO2 = 132.03 := by
  sorry

end NUMINAMATH_CALUDE_mass_CO2_from_CO_combustion_l3031_303193


namespace NUMINAMATH_CALUDE_briellesClockRings_l3031_303124

/-- Calculates the number of times a clock rings in a day -/
def ringsPerDay (startHour interval : ℕ) : ℕ :=
  (24 - startHour + interval - 1) / interval

/-- Represents the clock's ringing pattern over three days -/
structure ClockPattern where
  day1Interval : ℕ
  day1Start : ℕ
  day2Interval : ℕ
  day2Start : ℕ
  day3Interval : ℕ
  day3Start : ℕ

/-- Calculates the total number of rings for the given clock pattern -/
def totalRings (pattern : ClockPattern) : ℕ :=
  ringsPerDay pattern.day1Start pattern.day1Interval +
  ringsPerDay pattern.day2Start pattern.day2Interval +
  ringsPerDay pattern.day3Start pattern.day3Interval

/-- The specific clock pattern from the problem -/
def briellesClockPattern : ClockPattern :=
  { day1Interval := 3
    day1Start := 1
    day2Interval := 4
    day2Start := 2
    day3Interval := 5
    day3Start := 3 }

theorem briellesClockRings :
  totalRings briellesClockPattern = 19 := by
  sorry


end NUMINAMATH_CALUDE_briellesClockRings_l3031_303124


namespace NUMINAMATH_CALUDE_basketball_shooting_test_l3031_303105

-- Define the probabilities of making a basket for students A and B
def prob_A : ℚ := 1/2
def prob_B : ℚ := 2/3

-- Define the number of shots for Part I
def shots_part_I : ℕ := 3

-- Define the number of chances for Part II
def chances_part_II : ℕ := 4

-- Define the function to calculate the probability of exactly k successes in n trials
def binomial_probability (n : ℕ) (k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k) * p^k * (1-p)^(n-k)

-- Define the probability of student A meeting the standard in Part I
def prob_A_meets_standard : ℚ :=
  binomial_probability shots_part_I 2 prob_A + binomial_probability shots_part_I 3 prob_A

-- Define the probability distribution of X (number of shots taken by B) in Part II
def prob_X (x : ℕ) : ℚ :=
  if x = 2 then prob_B^2
  else if x = 3 then prob_B * (1-prob_B) * prob_B + prob_B^2 * (1-prob_B) + (1-prob_B)^3
  else if x = 4 then (1-prob_B) * prob_B^2 + prob_B * (1-prob_B) * prob_B
  else 0

-- Define the expected value of X
def expected_X : ℚ :=
  2 * prob_X 2 + 3 * prob_X 3 + 4 * prob_X 4

-- Theorem statement
theorem basketball_shooting_test :
  prob_A_meets_standard = 1/2 ∧ expected_X = 25/9 := by sorry

end NUMINAMATH_CALUDE_basketball_shooting_test_l3031_303105


namespace NUMINAMATH_CALUDE_rectangle_area_diagonal_l3031_303162

/-- Given a rectangle with length to width ratio of 5:2 and diagonal d,
    prove that its area A can be expressed as A = (10/29) * d^2 -/
theorem rectangle_area_diagonal (d : ℝ) (h : d > 0) :
  ∃ (l w : ℝ), l > 0 ∧ w > 0 ∧ l / w = 5 / 2 ∧ l^2 + w^2 = d^2 ∧ l * w = (10/29) * d^2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_diagonal_l3031_303162


namespace NUMINAMATH_CALUDE_seat_difference_is_two_l3031_303177

/-- Represents an airplane with first-class and coach class seats. -/
structure Airplane where
  total_seats : ℕ
  coach_seats : ℕ
  first_class_seats : ℕ
  h1 : total_seats = first_class_seats + coach_seats
  h2 : coach_seats > 4 * first_class_seats

/-- The difference between coach seats and 4 times first-class seats. -/
def seat_difference (a : Airplane) : ℕ :=
  a.coach_seats - 4 * a.first_class_seats

/-- Theorem stating the seat difference for a specific airplane configuration. -/
theorem seat_difference_is_two (a : Airplane)
  (h3 : a.total_seats = 387)
  (h4 : a.coach_seats = 310) :
  seat_difference a = 2 := by
  sorry


end NUMINAMATH_CALUDE_seat_difference_is_two_l3031_303177


namespace NUMINAMATH_CALUDE_prob_A_and_B_l3031_303102

/-- The probability of event A occurring -/
def prob_A : ℝ := 0.75

/-- The probability of event B occurring -/
def prob_B : ℝ := 0.60

/-- The theorem stating that the probability of both A and B occurring is 0.45 -/
theorem prob_A_and_B : prob_A * prob_B = 0.45 := by
  sorry

end NUMINAMATH_CALUDE_prob_A_and_B_l3031_303102


namespace NUMINAMATH_CALUDE_train_journey_theorem_l3031_303100

/-- Represents the properties of a train journey -/
structure TrainJourney where
  reducedSpeed : ℝ  -- Speed at which the train actually travels
  speedFraction : ℝ  -- Fraction of the train's own speed at which it travels
  time : ℝ  -- Time taken for the journey
  distance : ℝ  -- Distance traveled

/-- The problem setup -/
def trainProblem (trainA trainB : TrainJourney) : Prop :=
  trainA.speedFraction = 2/3 ∧
  trainA.time = 12 ∧
  trainA.distance = 360 ∧
  trainB.speedFraction = 1/2 ∧
  trainB.time = 8 ∧
  trainB.distance = trainA.distance

/-- The theorem to be proved -/
theorem train_journey_theorem (trainA trainB : TrainJourney) 
  (h : trainProblem trainA trainB) : 
  (trainA.time * (1 - trainA.speedFraction) + trainB.time * (1 - trainB.speedFraction) = 8) ∧
  (trainB.distance = 360) := by
  sorry

end NUMINAMATH_CALUDE_train_journey_theorem_l3031_303100


namespace NUMINAMATH_CALUDE_largest_divisor_of_difference_of_squares_l3031_303121

theorem largest_divisor_of_difference_of_squares (m n : ℕ) : 
  Odd m → Odd n → n < m → m - n > 2 → 
  (∀ k : ℕ, k > 4 → ∃ x y : ℕ, Odd x ∧ Odd y ∧ y < x ∧ x - y > 2 ∧ ¬(k ∣ x^2 - y^2)) ∧ 
  (∀ x y : ℕ, Odd x → Odd y → y < x → x - y > 2 → (4 ∣ x^2 - y^2)) := by
sorry

end NUMINAMATH_CALUDE_largest_divisor_of_difference_of_squares_l3031_303121


namespace NUMINAMATH_CALUDE_conic_section_eccentricity_l3031_303137

/-- The eccentricity of the conic section defined by 10x - 2xy - 2y + 1 = 0 is √2 -/
theorem conic_section_eccentricity :
  let P : ℝ × ℝ → Prop := λ (x, y) ↦ 10 * x - 2 * x * y - 2 * y + 1 = 0
  ∃ e : ℝ, e = Real.sqrt 2 ∧
    ∀ (x y : ℝ), P (x, y) →
      (Real.sqrt ((x - 2)^2 + (y - 2)^2)) / (|x - y + 3| / Real.sqrt 2) = e :=
by sorry

end NUMINAMATH_CALUDE_conic_section_eccentricity_l3031_303137


namespace NUMINAMATH_CALUDE_least_months_to_double_debt_l3031_303126

def initial_amount : ℝ := 1200
def interest_rate : ℝ := 0.06

def compound_factor : ℝ := 1 + interest_rate

theorem least_months_to_double_debt : 
  (∀ n : ℕ, n < 12 → compound_factor ^ n ≤ 2) ∧ 
  compound_factor ^ 12 > 2 := by
  sorry

end NUMINAMATH_CALUDE_least_months_to_double_debt_l3031_303126


namespace NUMINAMATH_CALUDE_stop_signs_per_mile_l3031_303172

-- Define the distance traveled
def distance : ℝ := 5 + 2

-- Define the number of stop signs encountered
def stop_signs : ℕ := 17 - 3

-- Theorem to prove
theorem stop_signs_per_mile : (stop_signs : ℝ) / distance = 2 := by
  sorry

end NUMINAMATH_CALUDE_stop_signs_per_mile_l3031_303172


namespace NUMINAMATH_CALUDE_jerky_order_fulfillment_l3031_303104

/-- Calculates the number of days required to fulfill a jerky order -/
def days_to_fulfill_order (order : ℕ) (in_stock : ℕ) (production_rate : ℕ) : ℕ :=
  ((order - in_stock) + production_rate - 1) / production_rate

/-- Theorem: Given the specific conditions, it takes 4 days to fulfill the order -/
theorem jerky_order_fulfillment :
  days_to_fulfill_order 60 20 10 = 4 := by
  sorry

end NUMINAMATH_CALUDE_jerky_order_fulfillment_l3031_303104


namespace NUMINAMATH_CALUDE_pages_per_day_l3031_303129

/-- Given a book with 576 pages read over 72 days, prove that the number of pages read per day is 8 -/
theorem pages_per_day (total_pages : ℕ) (total_days : ℕ) (h1 : total_pages = 576) (h2 : total_days = 72) :
  total_pages / total_days = 8 := by
  sorry

end NUMINAMATH_CALUDE_pages_per_day_l3031_303129


namespace NUMINAMATH_CALUDE_problem_solution_l3031_303118

def f (x : ℝ) := x^2 - 2*x

theorem problem_solution :
  (∀ x : ℝ, (|f x| + |x^2 + 2*x| ≥ 6*|x|) ↔ (x ≤ -3 ∨ x ≥ 3 ∨ x = 0)) ∧
  (∀ x a : ℝ, |x - a| < 1 → |f x - f a| < 2*|a| + 3) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l3031_303118


namespace NUMINAMATH_CALUDE_volume_ratio_l3031_303146

/-- Represents a square with side length 2 -/
structure Square :=
  (side : ℝ)
  (is_two : side = 2)

/-- Represents a pyramid formed by folding a square along its diagonal -/
structure Pyramid :=
  (base : Square)

/-- Represents a sphere circumscribing a pyramid -/
structure CircumscribedSphere :=
  (pyramid : Pyramid)

/-- The volume of a pyramid -/
def pyramid_volume (p : Pyramid) : ℝ := sorry

/-- The volume of a circumscribed sphere -/
def sphere_volume (s : CircumscribedSphere) : ℝ := sorry

/-- The theorem stating the ratio of volumes -/
theorem volume_ratio (s : CircumscribedSphere) :
  sphere_volume s / pyramid_volume s.pyramid = 4 * Real.pi := by sorry

end NUMINAMATH_CALUDE_volume_ratio_l3031_303146


namespace NUMINAMATH_CALUDE_only_translation_preserves_pattern_l3031_303166

-- Define the pattern
structure Pattern where
  square_side : ℝ
  triangle_hypotenuse : ℝ
  line_segments_length : ℝ
  total_length : ℝ
  triangle_faces_away : Bool

-- Define the line and the repeating pattern
def infinite_line_with_pattern : Pattern :=
  { square_side := 1
  , triangle_hypotenuse := 1
  , line_segments_length := 2
  , total_length := 4
  , triangle_faces_away := true
  }

-- Define the rigid motion transformations
inductive RigidMotion
  | Rotation (center : ℝ × ℝ) (angle : ℝ)
  | Translation (distance : ℝ)
  | ReflectionAcross
  | ReflectionPerpendicular (point : ℝ)

-- Theorem statement
theorem only_translation_preserves_pattern :
  ∀ (motion : RigidMotion),
    (∃ (k : ℤ), motion = RigidMotion.Translation (↑k * infinite_line_with_pattern.total_length)) ↔
    (motion ≠ RigidMotion.ReflectionAcross ∧
     (∀ (center : ℝ × ℝ) (angle : ℝ), motion ≠ RigidMotion.Rotation center angle) ∧
     (∀ (point : ℝ), motion ≠ RigidMotion.ReflectionPerpendicular point) ∧
     (∃ (distance : ℝ), motion = RigidMotion.Translation distance ∧
        distance = ↑k * infinite_line_with_pattern.total_length)) :=
by sorry

end NUMINAMATH_CALUDE_only_translation_preserves_pattern_l3031_303166


namespace NUMINAMATH_CALUDE_joyce_apples_l3031_303153

theorem joyce_apples (initial : Real) (received : Real) : 
  initial = 75.0 → received = 52.0 → initial + received = 127.0 := by
  sorry

end NUMINAMATH_CALUDE_joyce_apples_l3031_303153


namespace NUMINAMATH_CALUDE_color_grid_in_three_folds_l3031_303186

/-- Represents a square grid -/
structure Grid :=
  (size : Nat)
  (colored : Fin size → Fin size → Bool)

/-- Represents a fold operation along a grid line -/
inductive Fold
  | Vertical (column : Nat)
  | Horizontal (row : Nat)

/-- Applies a fold operation to a grid -/
def applyFold (g : Grid) (f : Fold) : Grid :=
  sorry

/-- Checks if the entire grid is colored -/
def isFullyColored (g : Grid) : Bool :=
  sorry

/-- The main theorem stating that any grid can be fully colored in 3 or fewer folds -/
theorem color_grid_in_three_folds (g : Grid) :
  ∃ (folds : List Fold), folds.length ≤ 3 ∧ isFullyColored (folds.foldl applyFold g) :=
sorry

end NUMINAMATH_CALUDE_color_grid_in_three_folds_l3031_303186


namespace NUMINAMATH_CALUDE_gravitational_force_at_distance_l3031_303174

/-- Represents the gravitational force at a given distance -/
structure GravitationalForce where
  distance : ℝ
  force : ℝ

/-- The gravitational constant k = f * d^2 -/
def gravitational_constant (gf : GravitationalForce) : ℝ :=
  gf.force * gf.distance^2

theorem gravitational_force_at_distance 
  (surface_force : GravitationalForce) 
  (space_force : GravitationalForce) :
  surface_force.distance = 5000 →
  surface_force.force = 800 →
  space_force.distance = 300000 →
  gravitational_constant surface_force = gravitational_constant space_force →
  space_force.force = 1/45 := by
sorry

end NUMINAMATH_CALUDE_gravitational_force_at_distance_l3031_303174


namespace NUMINAMATH_CALUDE_quadratic_factorization_l3031_303161

theorem quadratic_factorization (a b : ℕ) (h1 : a ≥ b) 
  (h2 : ∀ x : ℝ, x^2 - 18*x + 72 = (x - a)*(x - b)) : 
  4*b - a = 27 := by sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l3031_303161


namespace NUMINAMATH_CALUDE_book_sale_loss_percentage_l3031_303130

/-- Calculates the percentage loss when selling an item -/
def percentageLoss (costPrice sellingPrice : ℚ) : ℚ :=
  (costPrice - sellingPrice) / costPrice * 100

/-- Proves that the percentage loss is 10% given the conditions of the problem -/
theorem book_sale_loss_percentage
  (sellingPrice : ℚ)
  (gainPrice : ℚ)
  (gainPercentage : ℚ)
  (h1 : sellingPrice = 540)
  (h2 : gainPrice = 660)
  (h3 : gainPercentage = 10)
  (h4 : gainPrice = (100 + gainPercentage) / 100 * (gainPrice / (1 + gainPercentage / 100))) :
  percentageLoss (gainPrice / (1 + gainPercentage / 100)) sellingPrice = 10 := by
  sorry

#eval percentageLoss (660 / (1 + 10 / 100)) 540

end NUMINAMATH_CALUDE_book_sale_loss_percentage_l3031_303130


namespace NUMINAMATH_CALUDE_tom_time_ratio_l3031_303127

/-- The duration of the BS program in years -/
def bs_duration : ℕ := 3

/-- The duration of the Ph.D. program in years -/
def phd_duration : ℕ := 5

/-- Tom's total time to complete both programs in years -/
def tom_total_time : ℕ := 6

/-- The normal time to complete both programs -/
def normal_time : ℕ := bs_duration + phd_duration

theorem tom_time_ratio :
  (tom_total_time : ℚ) / (normal_time : ℚ) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_tom_time_ratio_l3031_303127


namespace NUMINAMATH_CALUDE_simple_interest_rate_calculation_l3031_303144

/-- Simple interest rate calculation -/
theorem simple_interest_rate_calculation
  (principal amount : ℚ)
  (time : ℕ)
  (h_principal : principal = 2500)
  (h_amount : amount = 3875)
  (h_time : time = 12)
  (h_positive : principal > 0 ∧ amount > principal ∧ time > 0) :
  (amount - principal) * 100 / (principal * time) = 55 / 12 :=
by sorry

end NUMINAMATH_CALUDE_simple_interest_rate_calculation_l3031_303144


namespace NUMINAMATH_CALUDE_grid_paths_6_4_l3031_303173

/-- The number of paths on a grid from (0,0) to (m,n) using exactly m+n steps -/
def grid_paths (m n : ℕ) : ℕ := Nat.choose (m + n) m

theorem grid_paths_6_4 : grid_paths 6 4 = 210 := by
  sorry

end NUMINAMATH_CALUDE_grid_paths_6_4_l3031_303173


namespace NUMINAMATH_CALUDE_angle_in_fourth_quadrant_l3031_303101

theorem angle_in_fourth_quadrant (θ : Real) 
  (h1 : Real.sin θ < Real.cos θ) 
  (h2 : Real.sin θ * Real.cos θ < 0) : 
  0 < θ ∧ θ < Real.pi / 2 ∧ Real.sin θ < 0 ∧ Real.cos θ > 0 := by
  sorry

end NUMINAMATH_CALUDE_angle_in_fourth_quadrant_l3031_303101


namespace NUMINAMATH_CALUDE_triangle_side_equality_l3031_303140

-- Define the triangle ABC
structure Triangle (α : Type) where
  A : α
  B : α
  C : α

-- Define the sides of the triangle
def side_AB (a b : ℤ) : ℤ := b^2 - 1
def side_BC (a b : ℤ) : ℤ := a^2
def side_CA (a b : ℤ) : ℤ := 2*a

-- State the theorem
theorem triangle_side_equality (a b : ℤ) (ABC : Triangle ℤ) :
  a > 1 ∧ b > 1 ∧ 
  side_AB a b = b^2 - 1 ∧
  side_BC a b = a^2 ∧
  side_CA a b = 2*a →
  b - a = 0 :=
sorry

end NUMINAMATH_CALUDE_triangle_side_equality_l3031_303140


namespace NUMINAMATH_CALUDE_complex_quadrant_l3031_303151

theorem complex_quadrant (z : ℂ) : (z + 2*I) * (3 + I) = 7 - I →
  (z.re > 0 ∧ z.im < 0) :=
sorry

end NUMINAMATH_CALUDE_complex_quadrant_l3031_303151


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3031_303195

theorem complex_equation_solution (a b : ℝ) : 
  (a : ℂ) + 3 * Complex.I = (b + Complex.I) * Complex.I → a = -1 ∧ b = 3 :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3031_303195


namespace NUMINAMATH_CALUDE_specific_bill_amount_l3031_303103

/-- Calculates the amount of a bill given its true discount, due time, and interest rate. -/
def bill_amount (true_discount : ℚ) (due_time : ℚ) (interest_rate : ℚ) : ℚ :=
  (true_discount * (100 + interest_rate * due_time)) / (interest_rate * due_time)

/-- Theorem stating that given the specific conditions, the bill amount is 1680. -/
theorem specific_bill_amount :
  let true_discount : ℚ := 180
  let due_time : ℚ := 9 / 12  -- 9 months expressed in years
  let interest_rate : ℚ := 16 -- 16% per annum
  bill_amount true_discount due_time interest_rate = 1680 :=
by sorry


end NUMINAMATH_CALUDE_specific_bill_amount_l3031_303103


namespace NUMINAMATH_CALUDE_negative_quartic_count_l3031_303155

theorem negative_quartic_count : ∃ (S : Finset ℤ), (∀ x ∈ S, x^4 - 62*x^2 + 60 < 0) ∧ S.card = 12 ∧ 
  ∀ x : ℤ, x^4 - 62*x^2 + 60 < 0 → x ∈ S :=
sorry

end NUMINAMATH_CALUDE_negative_quartic_count_l3031_303155


namespace NUMINAMATH_CALUDE_pollard_complexity_l3031_303159

/-- Represents the state of the algorithm at each iteration -/
structure AlgorithmState where
  u : Nat
  v : Nat

/-- The update function for the algorithm -/
def update (state : AlgorithmState) : AlgorithmState := sorry

/-- The main loop of the algorithm -/
def mainLoop (n : Nat) (initialState : AlgorithmState) : Nat := sorry

theorem pollard_complexity {n p : Nat} (hprime : Nat.Prime p) (hfactor : p ∣ n) :
  ∃ (c : Nat), mainLoop n (AlgorithmState.mk 1 1) ≤ 2 * p ∧
  mainLoop n (AlgorithmState.mk 1 1) ≤ c * p * (Nat.log n)^2 := by
  sorry

end NUMINAMATH_CALUDE_pollard_complexity_l3031_303159


namespace NUMINAMATH_CALUDE_clothing_tax_rate_l3031_303169

theorem clothing_tax_rate (total_spent : ℝ) (clothing_spent : ℝ) (food_spent : ℝ) (other_spent : ℝ)
  (clothing_tax : ℝ) (other_tax : ℝ) (total_tax : ℝ) :
  clothing_spent = 0.4 * total_spent →
  food_spent = 0.3 * total_spent →
  other_spent = 0.3 * total_spent →
  other_tax = 0.08 * other_spent →
  total_tax = 0.04 * total_spent →
  total_tax = clothing_tax + other_tax →
  clothing_tax / clothing_spent = 0.04 :=
by sorry

end NUMINAMATH_CALUDE_clothing_tax_rate_l3031_303169


namespace NUMINAMATH_CALUDE_find_k_l3031_303108

-- Define the functions f and g
def f (x : ℝ) : ℝ := 4 * x^2 - 3 * x + 5
def g (k : ℝ) (x : ℝ) : ℝ := 2 * x^2 - k * x + 7

-- State the theorem
theorem find_k : ∃ k : ℝ, f 5 - g k 5 = 40 ∧ k = 1.4 := by sorry

end NUMINAMATH_CALUDE_find_k_l3031_303108


namespace NUMINAMATH_CALUDE_triangle_problem_l3031_303141

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_problem (t : Triangle) 
  (h1 : (2 * t.a - t.c) * Real.cos t.B = t.b * Real.cos t.C)
  (h2 : t.a = 3)
  (h3 : (1/2) * t.a * t.c * Real.sin t.B = (3 * Real.sqrt 3) / 2) :
  t.B = π/3 ∧ t.a * t.c * Real.cos (π - t.A) = -1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l3031_303141


namespace NUMINAMATH_CALUDE_square_sum_equals_five_l3031_303182

theorem square_sum_equals_five (x y : ℝ) : (x + 2*y)^2 + |y - 1| = 0 → x^2 + y^2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equals_five_l3031_303182


namespace NUMINAMATH_CALUDE_classroom_boys_count_l3031_303109

/-- Represents the number of desks with one boy and one girl -/
def x : ℕ := 2

/-- The number of desks with two girls -/
def desks_two_girls : ℕ := 2 * x

/-- The number of desks with two boys -/
def desks_two_boys : ℕ := 2 * desks_two_girls

/-- The total number of girls in the classroom -/
def total_girls : ℕ := 10

/-- The total number of boys in the classroom -/
def total_boys : ℕ := 2 * desks_two_boys + x

theorem classroom_boys_count :
  total_girls = 5 * x ∧ total_boys = 18 := by
  sorry

#check classroom_boys_count

end NUMINAMATH_CALUDE_classroom_boys_count_l3031_303109


namespace NUMINAMATH_CALUDE_z_in_third_quadrant_l3031_303115

def complex_number_quadrant (z : ℂ) : Prop :=
  Real.sign z.re = -1 ∧ Real.sign z.im = -1

theorem z_in_third_quadrant :
  let z : ℂ := (-2 - Complex.I) * (3 + Complex.I)
  complex_number_quadrant z := by
  sorry

end NUMINAMATH_CALUDE_z_in_third_quadrant_l3031_303115


namespace NUMINAMATH_CALUDE_max_product_of_three_primes_l3031_303180

theorem max_product_of_three_primes (x y z : ℕ) : 
  Prime x → Prime y → Prime z →
  x ≠ y → x ≠ z → y ≠ z →
  x + y + z = 49 →
  x * y * z ≤ 4199 := by
sorry

end NUMINAMATH_CALUDE_max_product_of_three_primes_l3031_303180


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3031_303149

theorem quadratic_inequality_solution (a : ℝ) : 
  (∀ x : ℝ, ax^2 + 4*x + 2 > 0 ↔ -1/3 < x ∧ x < 1) → a = -6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3031_303149


namespace NUMINAMATH_CALUDE_questions_to_complete_l3031_303163

/-- Calculates the number of questions Sasha still needs to complete -/
theorem questions_to_complete 
  (rate : ℕ)        -- Questions completed per hour
  (total : ℕ)       -- Total questions to complete
  (time_worked : ℕ) -- Hours worked
  (h1 : rate = 15)  -- Sasha's rate is 15 questions per hour
  (h2 : total = 60) -- Total questions is 60
  (h3 : time_worked = 2) -- Time worked is 2 hours
  : total - (rate * time_worked) = 30 :=
by sorry

end NUMINAMATH_CALUDE_questions_to_complete_l3031_303163


namespace NUMINAMATH_CALUDE_number_theory_problem_l3031_303142

theorem number_theory_problem :
  (∃ n : ℤ, 35 = 5 * n) ∧
  (∃ n : ℤ, 252 = 21 * n) ∧ ¬(∃ m : ℤ, 48 = 21 * m) ∧
  (∃ k : ℤ, 180 = 9 * k) := by
  sorry

end NUMINAMATH_CALUDE_number_theory_problem_l3031_303142


namespace NUMINAMATH_CALUDE_linear_function_not_in_third_quadrant_l3031_303156

/-- A linear function that does not pass through the third quadrant -/
def linear_function (k : ℝ) (x : ℝ) : ℝ := (k - 2) * x + k

/-- Predicate to check if a point is in the third quadrant -/
def in_third_quadrant (x y : ℝ) : Prop := x < 0 ∧ y < 0

/-- Theorem stating the range of k for which the linear function does not pass through the third quadrant -/
theorem linear_function_not_in_third_quadrant (k : ℝ) :
  (∀ x, ¬(in_third_quadrant x (linear_function k x))) ↔ (0 ≤ k ∧ k < 2) :=
sorry

end NUMINAMATH_CALUDE_linear_function_not_in_third_quadrant_l3031_303156


namespace NUMINAMATH_CALUDE_problem_statement_l3031_303139

theorem problem_statement (p q r : ℝ) 
  (h1 : p * r / (p + q) + q * p / (q + r) + r * q / (r + p) = -8)
  (h2 : q * r / (p + q) + r * p / (q + r) + p * q / (r + p) = 9) :
  q / (p + q) + r / (q + r) + p / (r + p) = 10 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3031_303139


namespace NUMINAMATH_CALUDE_lock_rings_count_l3031_303148

theorem lock_rings_count : ∃ (n : ℕ), n > 0 ∧ 6^n - 1 ≤ 215 ∧ ∀ (m : ℕ), m > 0 → 6^m - 1 ≤ 215 → m ≤ n :=
sorry

end NUMINAMATH_CALUDE_lock_rings_count_l3031_303148


namespace NUMINAMATH_CALUDE_surface_area_of_rearranged_cube_l3031_303154

-- Define the cube and its properties
def cube_volume : ℝ := 8
def cube_side_length : ℝ := 2

-- Define the cuts
def first_cut_distance : ℝ := 1
def second_cut_distance : ℝ := 0.5

-- Define the heights of the pieces
def height_X : ℝ := first_cut_distance
def height_Y : ℝ := second_cut_distance
def height_Z : ℝ := cube_side_length - (first_cut_distance + second_cut_distance)

-- Define the total width of the rearranged pieces
def total_width : ℝ := height_X + height_Y + height_Z

-- Theorem statement
theorem surface_area_of_rearranged_cube :
  cube_volume = cube_side_length ^ 3 →
  (2 * cube_side_length * cube_side_length +    -- Top and bottom surfaces
   2 * total_width * cube_side_length +         -- Side surfaces
   2 * cube_side_length * cube_side_length) = 46 := by
sorry

end NUMINAMATH_CALUDE_surface_area_of_rearranged_cube_l3031_303154


namespace NUMINAMATH_CALUDE_jersey_cost_l3031_303188

theorem jersey_cost (initial_amount : ℕ) (num_jerseys : ℕ) (basketball_cost : ℕ) (shorts_cost : ℕ) (remaining_amount : ℕ) :
  initial_amount = 50 ∧
  num_jerseys = 5 ∧
  basketball_cost = 18 ∧
  shorts_cost = 8 ∧
  remaining_amount = 14 →
  ∃ (jersey_cost : ℕ), jersey_cost = 2 ∧ initial_amount = num_jerseys * jersey_cost + basketball_cost + shorts_cost + remaining_amount :=
by sorry

end NUMINAMATH_CALUDE_jersey_cost_l3031_303188


namespace NUMINAMATH_CALUDE_range_of_x_l3031_303136

def p (x : ℝ) : Prop := x^2 - 5*x + 6 ≥ 0

def q (x : ℝ) : Prop := 0 < x ∧ x < 4

theorem range_of_x (x : ℝ) (h1 : p x ∨ q x) (h2 : ¬q x) :
  x ≤ 0 ∨ x ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_x_l3031_303136


namespace NUMINAMATH_CALUDE_tangent_slope_at_zero_l3031_303167

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.exp x

theorem tangent_slope_at_zero : 
  (deriv f) 0 = 2 :=
by sorry

end NUMINAMATH_CALUDE_tangent_slope_at_zero_l3031_303167


namespace NUMINAMATH_CALUDE_sine_range_theorem_l3031_303120

theorem sine_range_theorem (x : ℝ) :
  x ∈ Set.Icc (0 : ℝ) (2 * Real.pi) →
  (Set.Icc (0 : ℝ) (2 * Real.pi) ∩ {x | Real.sin x ≥ Real.sqrt 3 / 2}) =
  Set.Icc (Real.pi / 3) ((2 * Real.pi) / 3) :=
by sorry

end NUMINAMATH_CALUDE_sine_range_theorem_l3031_303120


namespace NUMINAMATH_CALUDE_max_tiles_on_floor_l3031_303183

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Calculates the maximum number of tiles that can fit in a given direction -/
def maxTilesInDirection (floor : Rectangle) (tile : Rectangle) : ℕ :=
  (floor.width / tile.width) * (floor.height / tile.height)

/-- Theorem: The maximum number of 20x30 tiles on a 100x150 floor is 25 -/
theorem max_tiles_on_floor :
  let floor := Rectangle.mk 100 150
  let tile := Rectangle.mk 20 30
  let maxTiles := max (maxTilesInDirection floor tile) (maxTilesInDirection floor (Rectangle.mk tile.height tile.width))
  maxTiles = 25 := by
  sorry

#check max_tiles_on_floor

end NUMINAMATH_CALUDE_max_tiles_on_floor_l3031_303183


namespace NUMINAMATH_CALUDE_sample_size_calculation_l3031_303197

/-- Given a factory producing three product models A, B, and C with quantities in the ratio 3:4:7,
    prove that a sample containing 15 units of product A has a total size of 70. -/
theorem sample_size_calculation (ratio_A ratio_B ratio_C : ℕ) (sample_A : ℕ) (n : ℕ) : 
  ratio_A = 3 → ratio_B = 4 → ratio_C = 7 → sample_A = 15 →
  n = (ratio_A + ratio_B + ratio_C) * sample_A / ratio_A → n = 70 := by
  sorry

#check sample_size_calculation

end NUMINAMATH_CALUDE_sample_size_calculation_l3031_303197


namespace NUMINAMATH_CALUDE_circle_reassembly_possible_l3031_303199

/-- A circle with a marked point -/
structure MarkedCircle where
  center : ℝ × ℝ
  radius : ℝ
  marked_point : ℝ × ℝ

/-- A piece of a circle -/
structure CirclePiece

/-- Represents the process of cutting a circle into pieces -/
def cut_circle (c : MarkedCircle) (n : ℕ) : List CirclePiece :=
  sorry

/-- Represents the process of assembling pieces into a new circle -/
def assemble_circle (pieces : List CirclePiece) : MarkedCircle :=
  sorry

/-- Theorem stating that it's possible to cut and reassemble the circle as required -/
theorem circle_reassembly_possible (c : MarkedCircle) :
  ∃ (pieces : List CirclePiece),
    (pieces.length = 3) ∧
    (∃ (new_circle : MarkedCircle),
      (assemble_circle pieces = new_circle) ∧
      (new_circle.marked_point = new_circle.center)) :=
  sorry

end NUMINAMATH_CALUDE_circle_reassembly_possible_l3031_303199
