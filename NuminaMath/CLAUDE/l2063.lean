import Mathlib

namespace NUMINAMATH_CALUDE_complex_norm_squared_l2063_206388

theorem complex_norm_squared (z : ℂ) (h : z^2 + Complex.normSq z = 5 - 7*I) : Complex.normSq z = (74:ℝ)/10 := by
  sorry

end NUMINAMATH_CALUDE_complex_norm_squared_l2063_206388


namespace NUMINAMATH_CALUDE_odd_periodic_function_property_l2063_206307

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem odd_periodic_function_property
  (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_period : has_period f (π / 2))
  (h_value : f (π / 3) = 1) :
  f (-5 * π / 6) = -1 :=
sorry

end NUMINAMATH_CALUDE_odd_periodic_function_property_l2063_206307


namespace NUMINAMATH_CALUDE_pta_fundraising_savings_l2063_206310

theorem pta_fundraising_savings (initial_amount : ℚ) : 
  (3/4 : ℚ) * initial_amount - (1/2 : ℚ) * ((3/4 : ℚ) * initial_amount) = 150 →
  initial_amount = 400 := by
  sorry

end NUMINAMATH_CALUDE_pta_fundraising_savings_l2063_206310


namespace NUMINAMATH_CALUDE_semicircle_shaded_area_l2063_206338

/-- Given two adjacent semicircles sharing a diameter of length 2, prove that the area of the
    rectangle formed by the diameter and the vertical line through the intersection of the
    midpoints of the semicircle arcs is 2 square units. -/
theorem semicircle_shaded_area (X Y W Z M N P : ℝ × ℝ) : 
  -- Diameter XY is 2 units long
  (X.1 - Y.1)^2 + (X.2 - Y.2)^2 = 4 →
  -- M is on semicircle WXY
  (M.1 - X.1)^2 + (M.2 - X.2)^2 = 1 →
  -- N is on semicircle ZXY
  (N.1 - X.1)^2 + (N.2 - X.2)^2 = 1 →
  -- M is midpoint of arc WX
  (M.1 - W.1)^2 + (M.2 - W.2)^2 = (X.1 - W.1)^2 + (X.2 - W.2)^2 →
  -- N is midpoint of arc ZY
  (N.1 - Z.1)^2 + (N.2 - Z.2)^2 = (Y.1 - Z.1)^2 + (Y.2 - Z.2)^2 →
  -- P is on the vertical line from M
  P.1 = M.1 →
  -- P is on the vertical line from N
  P.1 = N.1 →
  -- P is the midpoint of MN
  2 * P.1 = M.1 + N.1 ∧ 2 * P.2 = M.2 + N.2 →
  -- The area of the rectangle is 2 square units
  (X.1 - Y.1) * (P.2 - X.2) = 2 :=
by sorry

end NUMINAMATH_CALUDE_semicircle_shaded_area_l2063_206338


namespace NUMINAMATH_CALUDE_cover_properties_l2063_206333

-- Define a type for points in the plane
variable {Point : Type}

-- Define a type for sets of points
variable {Set : Type}

-- Define the cover operation
variable (cover : Set → Set)

-- Define the subset relation
variable (subset : Set → Set → Prop)

-- Define the union operation
variable (union : Set → Set → Set)

-- Axiom for the given condition
axiom cover_union_superset (X Y : Set) :
  subset (union (cover (union X Y)) Y) (union (union (cover (cover X)) (cover Y)) Y)

-- Statement to prove
theorem cover_properties (X Y : Set) :
  (subset X (cover X)) ∧ 
  (cover (cover X) = cover X) ∧
  (subset X Y → subset (cover X) (cover Y)) :=
sorry

end NUMINAMATH_CALUDE_cover_properties_l2063_206333


namespace NUMINAMATH_CALUDE_square_8y_minus_5_l2063_206363

theorem square_8y_minus_5 (y : ℝ) (h : 4 * y^2 + 7 = 2 * y + 14) : (8 * y - 5)^2 = 125 := by
  sorry

end NUMINAMATH_CALUDE_square_8y_minus_5_l2063_206363


namespace NUMINAMATH_CALUDE_sequence_ratio_proof_l2063_206322

/-- Given an arithmetic sequence and a geometric sequence with specific properties,
    prove that the common ratio of the geometric sequence is (√5 - 1) / 2. -/
theorem sequence_ratio_proof (d : ℚ) (q : ℚ) (h_d : d ≠ 0) (h_q : 0 < q ∧ q < 1) :
  let a : ℕ → ℚ := λ n => d * n
  let b : ℕ → ℚ := λ n => d^2 * q^(n-1)
  (∃ k : ℕ+, (a 1)^2 + (a 2)^2 + (a 3)^2 = k * (b 1 + b 2 + b 3)) →
  q = (Real.sqrt 5 - 1) / 2 :=
by sorry

end NUMINAMATH_CALUDE_sequence_ratio_proof_l2063_206322


namespace NUMINAMATH_CALUDE_triangle_inequality_and_equality_l2063_206395

/-- Given a triangle with side lengths a, b, and c, 
    prove the inequality and equality condition --/
theorem triangle_inequality_and_equality (a b c : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) : 
  (a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ≥ 0) ∧ 
  (a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) = 0 ↔ a = b ∧ b = c) :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_and_equality_l2063_206395


namespace NUMINAMATH_CALUDE_fraction_denominator_value_l2063_206391

theorem fraction_denominator_value (p q : ℚ) (x : ℚ) 
  (h1 : p / q = 4 / 5)
  (h2 : 11 / 7 + (2 * q - p) / x = 2) : x = 14 := by
  sorry

end NUMINAMATH_CALUDE_fraction_denominator_value_l2063_206391


namespace NUMINAMATH_CALUDE_z_plus_inv_z_magnitude_l2063_206379

theorem z_plus_inv_z_magnitude (r : ℝ) (z : ℂ) (h1 : |r| < 3) (h2 : z + 1/z = r) : 
  Complex.abs z = 1 := by sorry

end NUMINAMATH_CALUDE_z_plus_inv_z_magnitude_l2063_206379


namespace NUMINAMATH_CALUDE_ratio_x_to_y_l2063_206384

theorem ratio_x_to_y (x y : ℝ) (h : (12*x - 5*y) / (15*x - 3*y) = 4/7) : 
  x / y = 23/24 := by
sorry

end NUMINAMATH_CALUDE_ratio_x_to_y_l2063_206384


namespace NUMINAMATH_CALUDE_chucks_team_lead_l2063_206301

/-- The lead of Chuck's team over the Yellow Team -/
def lead (chuck_score yellow_score : ℕ) : ℕ := chuck_score - yellow_score

/-- Theorem stating that Chuck's team's lead over the Yellow Team is 17 points -/
theorem chucks_team_lead : lead 72 55 = 17 := by
  sorry

end NUMINAMATH_CALUDE_chucks_team_lead_l2063_206301


namespace NUMINAMATH_CALUDE_fraction_equivalences_l2063_206335

theorem fraction_equivalences : 
  ∃ (n : ℕ) (p : ℕ) (d : ℚ),
    (n : ℚ) / 15 = 4 / 5 ∧
    (4 : ℚ) / 5 = p / 100 ∧
    (4 : ℚ) / 5 = d ∧
    d = 0.8 ∧
    p = 80 :=
sorry

end NUMINAMATH_CALUDE_fraction_equivalences_l2063_206335


namespace NUMINAMATH_CALUDE_queen_of_hearts_favorites_l2063_206375

/-- The Queen of Hearts' pretzel distribution problem -/
theorem queen_of_hearts_favorites (total_guests : ℕ) (total_pretzels : ℕ) 
  (favorite_pretzels : ℕ) (non_favorite_pretzels : ℕ) (favorite_guests : ℕ) :
  total_guests = 30 →
  total_pretzels = 100 →
  favorite_pretzels = 4 →
  non_favorite_pretzels = 3 →
  favorite_guests * favorite_pretzels + (total_guests - favorite_guests) * non_favorite_pretzels = total_pretzels →
  favorite_guests = 10 := by
sorry

end NUMINAMATH_CALUDE_queen_of_hearts_favorites_l2063_206375


namespace NUMINAMATH_CALUDE_f_divides_m_minus_n_prime_condition_l2063_206369

def f (x : ℚ) : ℕ :=
  let p := x.num.natAbs
  let q := x.den
  p + q

theorem f_divides_m_minus_n (x : ℚ) (m n : ℕ) (h : m > 0) (k : n > 0) (hx : x > 0) :
  f x = f ((m : ℚ) * x / n) → (f x : ℤ) ∣ |m - n| :=
by sorry

theorem prime_condition (n : ℕ) (hn : n > 1) :
  (∀ x : ℚ, x > 0 → f x = f ((2^n : ℚ) * x) → f x = 2^n - 1) ↔ Nat.Prime n :=
by sorry

end NUMINAMATH_CALUDE_f_divides_m_minus_n_prime_condition_l2063_206369


namespace NUMINAMATH_CALUDE_sin_alpha_minus_beta_l2063_206396

theorem sin_alpha_minus_beta (α β : Real) 
  (h1 : Real.sin α - Real.cos β = -2/3)
  (h2 : Real.cos α + Real.sin β = 1/3) :
  Real.sin (α - β) = 13/18 := by
  sorry

end NUMINAMATH_CALUDE_sin_alpha_minus_beta_l2063_206396


namespace NUMINAMATH_CALUDE_mandy_reading_age_ratio_l2063_206306

/-- Represents Mandy's reading progression over time -/
structure ReadingProgression where
  starting_age : ℕ
  starting_pages : ℕ
  middle_age_multiplier : ℕ
  middle_pages_multiplier : ℕ
  later_years : ℕ
  later_pages_multiplier : ℕ
  current_pages_multiplier : ℕ
  current_pages : ℕ

/-- Theorem stating the ratio of Mandy's age when she started reading 40-page books to her starting age -/
theorem mandy_reading_age_ratio 
  (rp : ReadingProgression)
  (h1 : rp.starting_age = 6)
  (h2 : rp.starting_pages = 8)
  (h3 : rp.middle_pages_multiplier = 5)
  (h4 : rp.later_pages_multiplier = 3)
  (h5 : rp.later_years = 8)
  (h6 : rp.current_pages_multiplier = 4)
  (h7 : rp.current_pages = 480) :
  (rp.starting_age * rp.middle_pages_multiplier) / rp.starting_age = 5 := by
  sorry

#check mandy_reading_age_ratio

end NUMINAMATH_CALUDE_mandy_reading_age_ratio_l2063_206306


namespace NUMINAMATH_CALUDE_optimal_price_maximizes_profit_max_profit_at_optimal_price_profit_function_is_quadratic_l2063_206313

/-- Represents the profit function for a product sale scenario -/
def profit_function (x : ℝ) : ℝ :=
  (x - 8) * (100 - 10 * (x - 10))

/-- The optimal selling price that maximizes profit -/
def optimal_price : ℝ := 14

/-- The maximum daily profit -/
def max_profit : ℝ := 360

/-- Theorem stating that the optimal price maximizes the profit function -/
theorem optimal_price_maximizes_profit :
  ∀ x, x > 10 → profit_function x ≤ profit_function optimal_price :=
sorry

/-- Theorem stating that the maximum profit is achieved at the optimal price -/
theorem max_profit_at_optimal_price :
  profit_function optimal_price = max_profit :=
sorry

/-- Theorem stating that the profit function is a quadratic function -/
theorem profit_function_is_quadratic :
  ∃ a b c, ∀ x, profit_function x = a * x^2 + b * x + c ∧ a < 0 :=
sorry

end NUMINAMATH_CALUDE_optimal_price_maximizes_profit_max_profit_at_optimal_price_profit_function_is_quadratic_l2063_206313


namespace NUMINAMATH_CALUDE_product_of_roots_l2063_206341

theorem product_of_roots : (16 : ℝ) ^ (1/4) * (32 : ℝ) ^ (1/5) = 4 := by sorry

end NUMINAMATH_CALUDE_product_of_roots_l2063_206341


namespace NUMINAMATH_CALUDE_remaining_milk_james_remaining_milk_l2063_206325

/-- Calculates the remaining milk in ounces and liters after consumption --/
theorem remaining_milk (initial_gallons : ℕ) (ounces_per_gallon : ℕ) 
  (james_consumed : ℕ) (sarah_consumed : ℕ) (mark_consumed : ℕ) 
  (ounce_to_liter : ℝ) : ℕ × ℝ :=
  let initial_ounces := initial_gallons * ounces_per_gallon
  let total_consumed := james_consumed + sarah_consumed + mark_consumed
  let remaining_ounces := initial_ounces - total_consumed
  let remaining_liters := (remaining_ounces : ℝ) * ounce_to_liter
  (remaining_ounces, remaining_liters)

/-- Proves that James has 326 ounces and approximately 9.64 liters of milk left --/
theorem james_remaining_milk :
  remaining_milk 3 128 13 20 25 0.0295735 = (326, 9.641051) :=
by sorry

end NUMINAMATH_CALUDE_remaining_milk_james_remaining_milk_l2063_206325


namespace NUMINAMATH_CALUDE_triangle_area_with_median_l2063_206318

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the median AM
def median (t : Triangle) : ℝ × ℝ := sorry

-- Define the length of a line segment
def length (p q : ℝ × ℝ) : ℝ := sorry

-- Define the area of a triangle
def area (t : Triangle) : ℝ := sorry

theorem triangle_area_with_median :
  ∀ (t : Triangle),
    length (t.A) (t.B) = 9 →
    length (t.A) (t.C) = 17 →
    length (t.A) (median t) = 12 →
    area t = 20 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_with_median_l2063_206318


namespace NUMINAMATH_CALUDE_hours_to_seconds_l2063_206329

-- Define constants
def minutes_per_hour : ℕ := 60
def seconds_per_minute : ℕ := 60

-- Define the theorem
theorem hours_to_seconds (hours : ℚ) : 
  hours * (minutes_per_hour * seconds_per_minute) = 12600 ↔ hours = 3.5 :=
by sorry

end NUMINAMATH_CALUDE_hours_to_seconds_l2063_206329


namespace NUMINAMATH_CALUDE_max_stamps_with_50_dollars_l2063_206390

/-- The maximum number of stamps that can be purchased with a given amount of money and stamp price. -/
def maxStamps (totalMoney stampPrice : ℕ) : ℕ :=
  totalMoney / stampPrice

/-- Theorem stating that with $50 and stamps costing 25 cents each, the maximum number of stamps that can be purchased is 200. -/
theorem max_stamps_with_50_dollars : 
  let dollarAmount : ℕ := 50
  let stampPriceCents : ℕ := 25
  let totalCents : ℕ := dollarAmount * 100
  maxStamps totalCents stampPriceCents = 200 := by
  sorry

#eval maxStamps (50 * 100) 25

end NUMINAMATH_CALUDE_max_stamps_with_50_dollars_l2063_206390


namespace NUMINAMATH_CALUDE_max_books_theorem_l2063_206317

def single_book_cost : ℕ := 3
def four_pack_cost : ℕ := 10
def seven_pack_cost : ℕ := 15
def budget : ℕ := 32

def max_books_bought (budget single four seven : ℕ) : ℕ :=
  sorry

theorem max_books_theorem :
  max_books_bought budget single_book_cost four_pack_cost seven_pack_cost = 14 :=
by sorry

end NUMINAMATH_CALUDE_max_books_theorem_l2063_206317


namespace NUMINAMATH_CALUDE_almas_test_score_l2063_206385

/-- Given two people, Alma and Melina, where Melina's age is 60 and three times Alma's age,
    and the sum of their ages is twice Alma's test score, prove that Alma's test score is 40. -/
theorem almas_test_score (alma_age melina_age alma_score : ℕ) : 
  melina_age = 60 →
  melina_age = 3 * alma_age →
  alma_age + melina_age = 2 * alma_score →
  alma_score = 40 := by
  sorry

end NUMINAMATH_CALUDE_almas_test_score_l2063_206385


namespace NUMINAMATH_CALUDE_percentage_change_condition_l2063_206377

theorem percentage_change_condition (r s N : ℝ) 
  (hr : r > 0) (hs : s > 0) (hN : N > 0) (hs_bound : s < 50) :
  (N * (1 + r / 100) * (1 - s / 100) > N) ↔ (r > 100 * s / (100 - s)) :=
sorry

end NUMINAMATH_CALUDE_percentage_change_condition_l2063_206377


namespace NUMINAMATH_CALUDE_evaluate_expression_l2063_206304

theorem evaluate_expression : (3^3)^2 * 3^2 = 6561 := by sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2063_206304


namespace NUMINAMATH_CALUDE_expanded_binomial_equals_x_power_minus_one_l2063_206374

theorem expanded_binomial_equals_x_power_minus_one (x : ℝ) :
  (x - 1)^5 + 5*(x - 1)^4 + 10*(x - 1)^3 + 10*(x - 1)^2 + 5*(x - 1) = x^5 - 1 := by
  sorry

end NUMINAMATH_CALUDE_expanded_binomial_equals_x_power_minus_one_l2063_206374


namespace NUMINAMATH_CALUDE_yellow_flowers_count_l2063_206343

/-- Represents a garden with yellow, green, and red flowers. -/
structure Garden where
  yellow : ℕ
  green : ℕ
  red : ℕ

/-- Proves that a garden with the given conditions has 12 yellow flowers. -/
theorem yellow_flowers_count (g : Garden) : 
  g.yellow + g.green + g.red = 78 → 
  g.red = 42 → 
  g.green = 2 * g.yellow → 
  g.yellow = 12 := by
  sorry

#check yellow_flowers_count

end NUMINAMATH_CALUDE_yellow_flowers_count_l2063_206343


namespace NUMINAMATH_CALUDE_sum_of_digits_b_l2063_206387

/-- The number with 2^n digits '9' in base 10 -/
def a (n : ℕ) : ℕ := 10^(2^n) - 1

/-- The product of the first n+1 terms of a_k -/
def b : ℕ → ℕ
  | 0 => a 0
  | n+1 => (a (n+1)) * (b n)

/-- The sum of digits of a natural number -/
def sum_of_digits : ℕ → ℕ := sorry

theorem sum_of_digits_b (n : ℕ) : sum_of_digits (b n) = 9 * 2^n := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_b_l2063_206387


namespace NUMINAMATH_CALUDE_pages_per_chapter_l2063_206372

/-- Given a book with 555 pages equally distributed over 5 chapters,
    prove that each chapter contains 111 pages. -/
theorem pages_per_chapter (total_pages : ℕ) (num_chapters : ℕ) (pages_per_chapter : ℕ) :
  total_pages = 555 →
  num_chapters = 5 →
  total_pages = num_chapters * pages_per_chapter →
  pages_per_chapter = 111 := by
  sorry

end NUMINAMATH_CALUDE_pages_per_chapter_l2063_206372


namespace NUMINAMATH_CALUDE_expand_and_simplify_l2063_206365

theorem expand_and_simplify (a : ℝ) : a * (a + 2) - 2 * a = a^2 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l2063_206365


namespace NUMINAMATH_CALUDE_phoenix_equation_equal_roots_l2063_206364

theorem phoenix_equation_equal_roots (a b c : ℝ) (h1 : a ≠ 0) 
  (h2 : a + b + c = 0) (h3 : ∃ x : ℝ, a * x^2 + b * x + c = 0 ∧ 
  ∀ y : ℝ, a * y^2 + b * y + c = 0 → y = x) : a = c :=
sorry

end NUMINAMATH_CALUDE_phoenix_equation_equal_roots_l2063_206364


namespace NUMINAMATH_CALUDE_melanie_grew_more_turnips_l2063_206332

/-- The number of turnips Melanie grew -/
def melanie_turnips : ℕ := 139

/-- The number of turnips Benny grew -/
def benny_turnips : ℕ := 113

/-- The difference in turnips between Melanie and Benny -/
def turnip_difference : ℕ := melanie_turnips - benny_turnips

theorem melanie_grew_more_turnips : turnip_difference = 26 := by
  sorry

end NUMINAMATH_CALUDE_melanie_grew_more_turnips_l2063_206332


namespace NUMINAMATH_CALUDE_arithmetic_sequence_count_l2063_206380

theorem arithmetic_sequence_count (a₁ d : ℤ) (n : ℕ) :
  a₁ = -3 →
  d = 4 →
  a₁ + (n - 1) * d ≤ 40 →
  (∀ k : ℕ, k < n → a₁ + (k - 1) * d ≤ 40) →
  (∀ k : ℕ, k > n → a₁ + (k - 1) * d > 40) →
  n = 11 := by
sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_count_l2063_206380


namespace NUMINAMATH_CALUDE_sqrt_sum_fractions_l2063_206366

theorem sqrt_sum_fractions : 
  Real.sqrt ((9 : ℝ) / 16 + 25 / 9) = Real.sqrt 481 / 12 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_fractions_l2063_206366


namespace NUMINAMATH_CALUDE_min_sum_of_product_l2063_206370

theorem min_sum_of_product (a b c : ℕ+) (h : a * b * c = 3432) :
  ∃ (x y z : ℕ+), x * y * z = 3432 ∧ x + y + z ≤ a + b + c ∧ x + y + z = 56 := by
sorry

end NUMINAMATH_CALUDE_min_sum_of_product_l2063_206370


namespace NUMINAMATH_CALUDE_inscribed_rhombus_rectangle_perimeter_l2063_206308

/-- A rhombus inscribed in a rectangle -/
structure InscribedRhombus where
  /-- Length of EA -/
  ea : ℝ
  /-- Length of FB -/
  fb : ℝ
  /-- Length of AD (rhombus side) -/
  ad : ℝ
  /-- Length of BC (rhombus side) -/
  bc : ℝ
  /-- EA is positive -/
  ea_pos : 0 < ea
  /-- FB is positive -/
  fb_pos : 0 < fb
  /-- AD is positive -/
  ad_pos : 0 < ad
  /-- BC is positive -/
  bc_pos : 0 < bc

/-- The perimeter of the rectangle containing the inscribed rhombus -/
def rectangle_perimeter (r : InscribedRhombus) : ℝ :=
  2 * (r.ea + r.ad + r.fb + r.bc)

/-- Theorem stating that for the given measurements, the rectangle perimeter is 238 -/
theorem inscribed_rhombus_rectangle_perimeter :
  ∃ r : InscribedRhombus, r.ea = 12 ∧ r.fb = 25 ∧ r.ad = 37 ∧ r.bc = 45 ∧ rectangle_perimeter r = 238 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_rhombus_rectangle_perimeter_l2063_206308


namespace NUMINAMATH_CALUDE_pascal_triangle_25th_row_5th_number_l2063_206337

theorem pascal_triangle_25th_row_5th_number : 
  let n : ℕ := 24  -- The row number (0-indexed) for a row with 25 numbers
  let k : ℕ := 4   -- The index (0-indexed) of the 5th number
  Nat.choose n k = 12650 := by
sorry

end NUMINAMATH_CALUDE_pascal_triangle_25th_row_5th_number_l2063_206337


namespace NUMINAMATH_CALUDE_quartic_polynomial_root_l2063_206312

theorem quartic_polynomial_root (a b c d : ℚ) : 
  (∃ (x : ℝ), x^4 + a*x^3 + b*x^2 + c*x + d = 0 ∧ 
               x = 3 - Real.sqrt 5 ∨ x = 3 + Real.sqrt 5) →
  (∀ (x₁ x₂ x₃ x₄ : ℝ), 
    (x₁^4 + a*x₁^3 + b*x₁^2 + c*x₁ + d = 0) ∧
    (x₂^4 + a*x₂^3 + b*x₂^2 + c*x₂ + d = 0) ∧
    (x₃^4 + a*x₃^3 + b*x₃^2 + c*x₃ + d = 0) ∧
    (x₄^4 + a*x₄^3 + b*x₄^2 + c*x₄ + d = 0) →
    x₁ + x₂ + x₃ + x₄ = 0) →
  ∃ (x : ℤ), x^4 + a*x^3 + b*x^2 + c*x + d = 0 ∧ x = -3 :=
by sorry

end NUMINAMATH_CALUDE_quartic_polynomial_root_l2063_206312


namespace NUMINAMATH_CALUDE_days_for_one_piece_correct_l2063_206321

/-- The number of days Aarti needs to complete one piece of work -/
def days_for_one_piece : ℝ := 6

/-- The number of days Aarti needs to complete three pieces of work -/
def days_for_three_pieces : ℝ := 18

/-- Theorem stating that the number of days for one piece of work is correct -/
theorem days_for_one_piece_correct : 
  days_for_one_piece * 3 = days_for_three_pieces :=
sorry

end NUMINAMATH_CALUDE_days_for_one_piece_correct_l2063_206321


namespace NUMINAMATH_CALUDE_cos_150_deg_l2063_206309

theorem cos_150_deg : Real.cos (150 * π / 180) = - (Real.sqrt 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_cos_150_deg_l2063_206309


namespace NUMINAMATH_CALUDE_min_product_abc_l2063_206350

theorem min_product_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hsum : a + b + c = 2)
  (hbound1 : a ≤ 3*b ∧ a ≤ 3*c)
  (hbound2 : b ≤ 3*a ∧ b ≤ 3*c)
  (hbound3 : c ≤ 3*a ∧ c ≤ 3*b) :
  a * b * c ≥ 2/9 :=
sorry

end NUMINAMATH_CALUDE_min_product_abc_l2063_206350


namespace NUMINAMATH_CALUDE_winning_strategy_l2063_206397

/-- Represents the state of the game -/
structure GameState :=
  (pieces : ℕ)

/-- Represents a valid move in the game -/
def ValidMove (n : ℕ) : Prop :=
  1 ≤ n ∧ n ≤ 6

/-- Applies a move to the game state -/
def applyMove (state : GameState) (move : ℕ) : GameState :=
  { pieces := state.pieces - move }

/-- Checks if the game is over -/
def isGameOver (state : GameState) : Prop :=
  state.pieces = 0

/-- Represents a winning strategy for the first player -/
def isWinningStrategy (firstMove : ℕ) : Prop :=
  ∀ (opponentMoves : ℕ → ℕ), 
    (∀ n, ValidMove (opponentMoves n)) →
    ∃ (playerMoves : ℕ → ℕ),
      (∀ n, ValidMove (playerMoves n)) ∧
      isGameOver (applyMove (applyMove (GameState.mk 32) firstMove) (opponentMoves 0))

/-- The main theorem stating that removing 4 pieces is a winning strategy -/
theorem winning_strategy : isWinningStrategy 4 := by
  sorry

end NUMINAMATH_CALUDE_winning_strategy_l2063_206397


namespace NUMINAMATH_CALUDE_quadratic_roots_distinct_l2063_206359

-- Define the proportional function
def proportional_function (k : ℝ) (x : ℝ) : ℝ := k * x

-- Theorem statement
theorem quadratic_roots_distinct 
  (k : ℝ) 
  (h1 : k ≠ 0)
  (h2 : ∀ x1 x2 : ℝ, x1 < x2 → proportional_function k x1 > proportional_function k x2) :
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 - x1 + k - 1 = 0 ∧ x2^2 - x2 + k - 1 = 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_distinct_l2063_206359


namespace NUMINAMATH_CALUDE_min_distance_circle_line_l2063_206316

theorem min_distance_circle_line :
  let C : Set (ℝ × ℝ) := {p | (p.1 - 2)^2 + p.2^2 = 5}
  let L : Set (ℝ × ℝ) := {q | q.1 - 2*q.2 + 4 = 0}
  (∀ p ∈ C, ∃ q ∈ L, ∀ r ∈ L, dist p q ≤ dist p r) →
  (∃ p ∈ C, ∃ q ∈ L, dist p q = 3 * Real.sqrt 5 / 5) ∧
  (∀ p ∈ C, ∀ q ∈ L, dist p q ≥ 3 * Real.sqrt 5 / 5) :=
by sorry


end NUMINAMATH_CALUDE_min_distance_circle_line_l2063_206316


namespace NUMINAMATH_CALUDE_third_group_men_count_l2063_206376

/-- The work rate of one man -/
def man_rate : ℝ := sorry

/-- The work rate of one woman -/
def woman_rate : ℝ := sorry

/-- The number of men in the third group -/
def x : ℝ := sorry

theorem third_group_men_count : x = 4 :=
  by
  have h1 : 3 * man_rate + 8 * woman_rate = 6 * man_rate + 2 * woman_rate := by sorry
  have h2 : x * man_rate + 5 * woman_rate = 0.9285714285714286 * (6 * man_rate + 2 * woman_rate) := by sorry
  sorry

end NUMINAMATH_CALUDE_third_group_men_count_l2063_206376


namespace NUMINAMATH_CALUDE_dvd_book_theorem_l2063_206305

/-- Represents a DVD book with a given capacity and current number of DVDs -/
structure DVDBook where
  capacity : ℕ
  current : ℕ

/-- Calculates the number of additional DVDs that can be put in the book -/
def additionalDVDs (book : DVDBook) : ℕ :=
  book.capacity - book.current

theorem dvd_book_theorem (book : DVDBook) 
  (h1 : book.capacity = 126) 
  (h2 : book.current = 81) : 
  additionalDVDs book = 45 := by
  sorry

end NUMINAMATH_CALUDE_dvd_book_theorem_l2063_206305


namespace NUMINAMATH_CALUDE_circle_tangency_problem_l2063_206398

theorem circle_tangency_problem : 
  let divisors := (Finset.range 150).filter (λ x => x > 0 ∧ 150 % x = 0)
  (divisors.card : ℕ) = 11 := by
  sorry

end NUMINAMATH_CALUDE_circle_tangency_problem_l2063_206398


namespace NUMINAMATH_CALUDE_sqrt_17_bounds_l2063_206339

theorem sqrt_17_bounds : 4 < Real.sqrt 17 ∧ Real.sqrt 17 < 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_17_bounds_l2063_206339


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2063_206357

/-- The solution set of a quadratic inequality -/
def SolutionSet (a b : ℝ) : Set ℝ := {x | a * x^2 + b * x + 2 > 0}

/-- The open interval (-1/2, 1/3) -/
def OpenInterval : Set ℝ := {x | -1/2 < x ∧ x < 1/3}

/-- 
If the solution set of ax^2 + bx + 2 > 0 is (-1/2, 1/3), then a + b = -14
-/
theorem quadratic_inequality_solution (a b : ℝ) :
  SolutionSet a b = OpenInterval → a + b = -14 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2063_206357


namespace NUMINAMATH_CALUDE_benny_birthday_money_l2063_206389

/-- The amount of money Benny spent on baseball gear -/
def money_spent : ℕ := 34

/-- The amount of money Benny had left over -/
def money_left : ℕ := 33

/-- The total amount of money Benny received for his birthday -/
def total_money : ℕ := money_spent + money_left

theorem benny_birthday_money :
  total_money = 67 := by sorry

end NUMINAMATH_CALUDE_benny_birthday_money_l2063_206389


namespace NUMINAMATH_CALUDE_like_terms_imply_x_y_equal_one_l2063_206383

/-- Two terms are like terms if they have the same variables raised to the same powers. -/
def like_terms (term1 term2 : ℕ → ℕ → ℕ) : Prop :=
  ∀ a b, ∃ k, term1 a b = k * term2 a b ∨ term2 a b = k * term1 a b

/-- Given two terms 3a^(x+1)b^2 and 7a^2b^(x+y), if they are like terms, then x = 1 and y = 1. -/
theorem like_terms_imply_x_y_equal_one (x y : ℕ) :
  like_terms (λ a b => 3 * a^(x + 1) * b^2) (λ a b => 7 * a^2 * b^(x + y)) →
  x = 1 ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_imply_x_y_equal_one_l2063_206383


namespace NUMINAMATH_CALUDE_quadratic_inequality_k_range_l2063_206381

theorem quadratic_inequality_k_range :
  (∀ x : ℝ, ∀ k : ℝ, k * x^2 - k * x + 1 > 0) ↔ k ∈ Set.Ici 0 ∩ Set.Iio 4 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_k_range_l2063_206381


namespace NUMINAMATH_CALUDE_probability_three_red_balls_l2063_206358

def total_balls : ℕ := 15
def red_balls : ℕ := 10
def blue_balls : ℕ := 5
def drawn_balls : ℕ := 5
def target_red : ℕ := 3

theorem probability_three_red_balls :
  (Nat.choose red_balls target_red * Nat.choose blue_balls (drawn_balls - target_red)) /
  Nat.choose total_balls drawn_balls = 1200 / 3003 :=
sorry

end NUMINAMATH_CALUDE_probability_three_red_balls_l2063_206358


namespace NUMINAMATH_CALUDE_quadratic_one_solution_l2063_206371

theorem quadratic_one_solution (n : ℝ) : 
  (n > 0 ∧ ∃! x, 4 * x^2 + n * x + 1 = 0) ↔ n = 4 :=
sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_l2063_206371


namespace NUMINAMATH_CALUDE_starting_lineup_count_l2063_206355

/-- Represents a football team with a given number of total players and offensive linemen --/
structure FootballTeam where
  total_players : Nat
  offensive_linemen : Nat
  h_offensive_linemen_le_total : offensive_linemen ≤ total_players

/-- Calculates the number of ways to choose a starting lineup --/
def chooseStartingLineup (team : FootballTeam) : Nat :=
  team.offensive_linemen * (team.total_players - 1) * (team.total_players - 2) * (team.total_players - 3)

/-- Theorem stating that for a team of 10 players with 3 offensive linemen, 
    there are 1512 ways to choose a starting lineup --/
theorem starting_lineup_count (team : FootballTeam) 
  (h_total : team.total_players = 10) 
  (h_offensive : team.offensive_linemen = 3) : 
  chooseStartingLineup team = 1512 := by
  sorry

#eval chooseStartingLineup ⟨10, 3, by norm_num⟩

end NUMINAMATH_CALUDE_starting_lineup_count_l2063_206355


namespace NUMINAMATH_CALUDE_odd_prime_equality_l2063_206336

theorem odd_prime_equality (p m : ℕ) (x y : ℕ) 
  (h_prime : Nat.Prime p)
  (h_odd : Odd p)
  (h_x : x > 1)
  (h_y : y > 1)
  (h_eq : (x^p + y^p) / 2 = ((x + y) / 2)^m) :
  m = p := by
  sorry

end NUMINAMATH_CALUDE_odd_prime_equality_l2063_206336


namespace NUMINAMATH_CALUDE_overall_correct_percent_l2063_206323

def math_problems : ℕ := 30
def science_problems : ℕ := 20
def history_problems : ℕ := 50

def math_correct_percent : ℚ := 85 / 100
def science_correct_percent : ℚ := 75 / 100
def history_correct_percent : ℚ := 65 / 100

def total_problems : ℕ := math_problems + science_problems + history_problems

def total_correct : ℚ := 
  math_problems * math_correct_percent + 
  science_problems * science_correct_percent + 
  history_problems * history_correct_percent

theorem overall_correct_percent : 
  (total_correct / total_problems) * 100 = 73 := by sorry

end NUMINAMATH_CALUDE_overall_correct_percent_l2063_206323


namespace NUMINAMATH_CALUDE_min_squares_exceeding_300_l2063_206346

/-- The function that represents repeated squaring of a number -/
def repeated_square (x : ℕ) (n : ℕ) : ℕ :=
  match n with
  | 0 => x
  | n + 1 => (repeated_square x n) ^ 2

/-- The theorem stating that 3 is the smallest positive integer n for which
    repeated squaring of 3, n times, exceeds 300 -/
theorem min_squares_exceeding_300 :
  ∀ n : ℕ, n > 0 → (
    (repeated_square 3 n > 300 ∧ ∀ m : ℕ, m > 0 → m < n → repeated_square 3 m ≤ 300) ↔ n = 3
  ) :=
sorry

end NUMINAMATH_CALUDE_min_squares_exceeding_300_l2063_206346


namespace NUMINAMATH_CALUDE_basketball_expected_score_l2063_206362

/-- The expected score of a basketball player making two independent free throws -/
def expected_score (p : ℝ) : ℝ :=
  2 * p

theorem basketball_expected_score :
  let p : ℝ := 0.7  -- Probability of making a free throw
  expected_score p = 1.4 := by
  sorry

end NUMINAMATH_CALUDE_basketball_expected_score_l2063_206362


namespace NUMINAMATH_CALUDE_quadratic_solution_sum_l2063_206373

theorem quadratic_solution_sum (x : ℝ → Prop) :
  (∀ x, x * (4 * x - 5) = -4) →
  (∃ m n p : ℕ,
    (Nat.gcd m (Nat.gcd n p) = 1) ∧
    (∀ x, x = (m + Real.sqrt n) / p ∨ x = (m - Real.sqrt n) / p)) →
  (∃ m n p : ℕ, m + n + p = 52) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_solution_sum_l2063_206373


namespace NUMINAMATH_CALUDE_binomial_coefficient_bounds_l2063_206399

theorem binomial_coefficient_bounds (n : ℕ) : 2^n ≤ Nat.choose (2*n) n ∧ Nat.choose (2*n) n ≤ 2^(2*n) := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_bounds_l2063_206399


namespace NUMINAMATH_CALUDE_standard_spherical_coordinates_example_l2063_206311

/-- 
Given a point in spherical coordinates (ρ, θ, φ), this function returns the 
standard representation coordinates (ρ', θ', φ') where:
- ρ' = ρ
- θ' is θ adjusted to be in the range [0, 2π)
- φ' is φ adjusted to be in the range [0, π]
-/
def standardSphericalCoordinates (ρ θ φ : Real) : Real × Real × Real :=
  sorry

theorem standard_spherical_coordinates_example :
  let (ρ, θ, φ) := (5, 3 * Real.pi / 8, 9 * Real.pi / 5)
  let (ρ', θ', φ') := standardSphericalCoordinates ρ θ φ
  ρ' = 5 ∧ θ' = 3 * Real.pi / 8 ∧ φ' = Real.pi / 5 :=
by sorry

end NUMINAMATH_CALUDE_standard_spherical_coordinates_example_l2063_206311


namespace NUMINAMATH_CALUDE_no_natural_solutions_for_equation_l2063_206386

theorem no_natural_solutions_for_equation : 
  ∀ (x y z : ℕ), x^x + 2*y^y ≠ z^z := by
  sorry

end NUMINAMATH_CALUDE_no_natural_solutions_for_equation_l2063_206386


namespace NUMINAMATH_CALUDE_identity_proof_l2063_206326

theorem identity_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum : a + b + c = 1) : 
  Real.sqrt ((a + b * c) * (b + c * a) / (c + a * b)) + 
  Real.sqrt ((b + c * a) * (c + a * b) / (a + b * c)) + 
  Real.sqrt ((c + a * b) * (a + b * c) / (b + c * a)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_identity_proof_l2063_206326


namespace NUMINAMATH_CALUDE_constant_functions_from_functional_equation_l2063_206347

theorem constant_functions_from_functional_equation 
  (f : ℝ → ℝ) (g : ℝ → ℝ) 
  (h : ∀ (x y : ℝ), x > 0 → y > 0 → f (x^2 + y^2) = g (x * y)) :
  ∃ (c : ℝ), (∀ (x : ℝ), x > 0 → f x = c) ∧ (∀ (x : ℝ), x > 0 → g x = c) :=
sorry

end NUMINAMATH_CALUDE_constant_functions_from_functional_equation_l2063_206347


namespace NUMINAMATH_CALUDE_geometric_sequence_third_term_l2063_206392

/-- A geometric sequence with first term 3 and the sum of first, third, and fifth terms equal to 21 has its third term equal to 6. -/
theorem geometric_sequence_third_term (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a n = 3 * q^(n-1)) →  -- Definition of geometric sequence
  a 1 = 3 →                   -- First term is 3
  a 1 + a 3 + a 5 = 21 →      -- Sum of first, third, and fifth terms is 21
  a 3 = 6 :=                  -- Conclusion: third term is 6
by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_third_term_l2063_206392


namespace NUMINAMATH_CALUDE_circle_center_and_sum_l2063_206314

/-- The equation of a circle in the xy-plane -/
def CircleEquation (x y : ℝ) : Prop :=
  x^2 + y^2 = 4*x - 6*y + 9

/-- The center of a circle given by its equation -/
def CircleCenter (eq : ℝ → ℝ → Prop) : ℝ × ℝ := sorry

/-- Theorem stating that the center of the given circle is (2, -3) 
    and the sum of its coordinates is -1 -/
theorem circle_center_and_sum :
  let center := CircleCenter CircleEquation
  center = (2, -3) ∧ center.1 + center.2 = -1 := by sorry

end NUMINAMATH_CALUDE_circle_center_and_sum_l2063_206314


namespace NUMINAMATH_CALUDE_unsafe_overtaking_l2063_206351

/-- Represents the safety of car A overtaking car B before meeting car C -/
def is_overtaking_safe (V_A V_B V_C d_AB d_AC : ℝ) : Prop :=
  let rel_vel_AB := V_A - V_B
  let rel_vel_AC := V_A + V_C
  let t_AB := d_AB / rel_vel_AB
  let t_AC := d_AC / rel_vel_AC
  t_AB < t_AC

/-- Theorem stating that car A cannot safely overtake car B before meeting car C
    under the given conditions -/
theorem unsafe_overtaking :
  let V_A : ℝ := 55  -- mph
  let V_B : ℝ := 45  -- mph
  let V_C : ℝ := 50  -- mph (10% less than flat road velocity)
  let d_AB : ℝ := 50 -- ft
  let d_AC : ℝ := 200 -- ft
  ¬(is_overtaking_safe V_A V_B V_C d_AB d_AC) := by
  sorry

#check unsafe_overtaking

end NUMINAMATH_CALUDE_unsafe_overtaking_l2063_206351


namespace NUMINAMATH_CALUDE_greatest_integer_for_all_real_domain_l2063_206342

theorem greatest_integer_for_all_real_domain (b : ℤ) : 
  (∀ x : ℝ, (x^2 + b*x + 9 : ℝ) ≠ 0) ↔ b ≤ 5 :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_for_all_real_domain_l2063_206342


namespace NUMINAMATH_CALUDE_circle_radius_when_area_circumference_ratio_is_15_l2063_206393

theorem circle_radius_when_area_circumference_ratio_is_15 
  (M N : ℝ) (h1 : M > 0) (h2 : N > 0) (h3 : M / N = 15) : 
  ∃ r : ℝ, r > 0 ∧ M = π * r^2 ∧ N = 2 * π * r ∧ r = 30 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_when_area_circumference_ratio_is_15_l2063_206393


namespace NUMINAMATH_CALUDE_student_expected_score_l2063_206315

theorem student_expected_score :
  let total_questions : ℕ := 12
  let points_per_question : ℝ := 5
  let confident_questions : ℕ := 6
  let eliminate_one_questions : ℕ := 3
  let eliminate_two_questions : ℕ := 2
  let random_guess_questions : ℕ := 1
  let prob_correct_confident : ℝ := 1
  let prob_correct_eliminate_one : ℝ := 1/4
  let prob_correct_eliminate_two : ℝ := 1/3
  let prob_correct_random : ℝ := 1/4

  let expected_score : ℝ :=
    points_per_question * (
      confident_questions * prob_correct_confident +
      eliminate_one_questions * prob_correct_eliminate_one +
      eliminate_two_questions * prob_correct_eliminate_two +
      random_guess_questions * prob_correct_random
    )

  total_questions = confident_questions + eliminate_one_questions + eliminate_two_questions + random_guess_questions →
  expected_score = 41.25 :=
by sorry

end NUMINAMATH_CALUDE_student_expected_score_l2063_206315


namespace NUMINAMATH_CALUDE_data_median_and_mode_l2063_206344

def data : List ℕ := [15, 17, 14, 10, 15, 17, 17, 16, 14, 12]

def median (l : List ℕ) : ℚ := sorry

def mode (l : List ℕ) : ℕ := sorry

theorem data_median_and_mode :
  median data = 14.5 ∧ mode data = 17 := by sorry

end NUMINAMATH_CALUDE_data_median_and_mode_l2063_206344


namespace NUMINAMATH_CALUDE_line_passes_through_point_l2063_206360

/-- The line y+2=k(x+1) always passes through the point (-1, -2) -/
theorem line_passes_through_point :
  ∀ (k : ℝ), ((-1) : ℝ) + 2 = k * ((-1) + 1) ∧ (-2 : ℝ) + 2 = k * ((-1) + 1) := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_point_l2063_206360


namespace NUMINAMATH_CALUDE_equation_solution_l2063_206378

theorem equation_solution : ∃ x : ℝ, 3 * x - 6 = |(-23 + 5)| ∧ x = 8 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2063_206378


namespace NUMINAMATH_CALUDE_crease_lines_set_l2063_206348

/-- Given a circle with center O, radius R, and a point A inside the circle such that OA = a,
    the set of points P that satisfy |PO| + |PA| ≥ R is equivalent to the set
    {(x, y): ((x - a/2)^2) / ((R/2)^2) + (y^2) / ((R/2)^2 - (a/2)^2) ≥ 1} -/
theorem crease_lines_set (O A : ℝ × ℝ) (R a : ℝ) (h1 : R > 0) (h2 : 0 < a ∧ a < R) :
  let d (P : ℝ × ℝ) (Q : ℝ × ℝ) := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)
  {P : ℝ × ℝ | d P O + d P A ≥ R} =
  {P : ℝ × ℝ | ((P.1 - a/2)^2) / ((R/2)^2) + (P.2^2) / ((R/2)^2 - (a/2)^2) ≥ 1} :=
by sorry

end NUMINAMATH_CALUDE_crease_lines_set_l2063_206348


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l2063_206345

open Real

theorem fixed_point_of_exponential_function (a : ℝ) (h : a > 0) :
  let f : ℝ → ℝ := fun x ↦ a^(x - 3) + 3
  f 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l2063_206345


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l2063_206340

theorem diophantine_equation_solution (x y z : ℤ) :
  x^2 + y^2 + z^2 - x*y - y*z - z*x = 3 ↔ 
  (∃ k : ℤ, (x = k + 2 ∧ y = k + 1 ∧ z = k) ∨ (x = k - 2 ∧ y = k - 1 ∧ z = k)) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l2063_206340


namespace NUMINAMATH_CALUDE_shielas_classmates_l2063_206302

theorem shielas_classmates (total_stars : ℕ) (stars_per_bottle : ℕ) (num_classmates : ℕ) : 
  total_stars = 45 → stars_per_bottle = 5 → num_classmates = total_stars / stars_per_bottle → num_classmates = 9 := by
  sorry

end NUMINAMATH_CALUDE_shielas_classmates_l2063_206302


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l2063_206382

theorem sqrt_product_equality : Real.sqrt 2 * Real.sqrt 3 = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l2063_206382


namespace NUMINAMATH_CALUDE_trader_weight_manipulation_l2063_206300

theorem trader_weight_manipulation :
  ∀ (supplier_weight : ℝ) (cost_price : ℝ),
  supplier_weight > 0 → cost_price > 0 →
  let actual_bought_weight := supplier_weight * 1.1
  let claimed_sell_weight := actual_bought_weight
  let actual_sell_weight := claimed_sell_weight / 1.65
  let weight_difference := claimed_sell_weight - actual_sell_weight
  (cost_price * actual_sell_weight) * 1.65 = cost_price * claimed_sell_weight →
  weight_difference / actual_sell_weight = 0.65 := by
  sorry

end NUMINAMATH_CALUDE_trader_weight_manipulation_l2063_206300


namespace NUMINAMATH_CALUDE_abc_fraction_value_l2063_206303

theorem abc_fraction_value (a b c : ℝ) 
  (eq1 : a * b / (a + b) = 3)
  (eq2 : b * c / (b + c) = 5)
  (eq3 : c * a / (c + a) = 8) :
  a * b * c / (a * b + b * c + c * a) = 240 / 79 := by
  sorry

end NUMINAMATH_CALUDE_abc_fraction_value_l2063_206303


namespace NUMINAMATH_CALUDE_bucket_capacity_reduction_l2063_206331

theorem bucket_capacity_reduction (original_buckets : ℕ) (capacity_ratio : ℚ) : 
  original_buckets = 42 →
  capacity_ratio = 2 / 5 →
  (original_buckets : ℚ) / capacity_ratio = 105 := by sorry

end NUMINAMATH_CALUDE_bucket_capacity_reduction_l2063_206331


namespace NUMINAMATH_CALUDE_kathleen_july_savings_l2063_206328

/-- Represents Kathleen's savings and expenses --/
structure KathleenFinances where
  june_savings : ℕ
  august_savings : ℕ
  school_supplies_expense : ℕ
  clothes_expense : ℕ
  money_left : ℕ
  aunt_bonus_threshold : ℕ

/-- Calculates Kathleen's savings in July --/
def july_savings (k : KathleenFinances) : ℕ :=
  k.school_supplies_expense + k.clothes_expense + k.money_left - k.june_savings - k.august_savings

/-- Theorem stating that Kathleen's savings in July is $46 --/
theorem kathleen_july_savings :
  ∀ (k : KathleenFinances),
    k.june_savings = 21 →
    k.august_savings = 45 →
    k.school_supplies_expense = 12 →
    k.clothes_expense = 54 →
    k.money_left = 46 →
    k.aunt_bonus_threshold = 125 →
    july_savings k = 46 :=
  sorry


end NUMINAMATH_CALUDE_kathleen_july_savings_l2063_206328


namespace NUMINAMATH_CALUDE_hypercube_diagonal_count_l2063_206349

/-- A hypercube is a 4-dimensional cube -/
structure Hypercube where
  vertices : Finset (Fin 16)
  edges : Finset (Fin 16 × Fin 16)
  vertex_count : vertices.card = 16
  edge_count : edges.card = 32

/-- A diagonal in a hypercube is a segment joining two vertices not joined by an edge -/
def Diagonal (h : Hypercube) (v w : Fin 16) : Prop :=
  v ∈ h.vertices ∧ w ∈ h.vertices ∧ v ≠ w ∧ (v, w) ∉ h.edges

/-- The number of diagonals in a hypercube -/
def DiagonalCount (h : Hypercube) : ℕ :=
  (h.vertices.card.choose 2) - h.edges.card

/-- Theorem: A hypercube has 408 diagonals -/
theorem hypercube_diagonal_count (h : Hypercube) : DiagonalCount h = 408 := by
  sorry

end NUMINAMATH_CALUDE_hypercube_diagonal_count_l2063_206349


namespace NUMINAMATH_CALUDE_topology_classification_l2063_206356

def X : Set Char := {'a', 'b', 'c'}

def τ₁ : Set (Set Char) := {∅, {'a'}, {'c'}, {'a', 'b', 'c'}}
def τ₂ : Set (Set Char) := {∅, {'b'}, {'c'}, {'b', 'c'}, {'a', 'b', 'c'}}
def τ₃ : Set (Set Char) := {∅, {'a'}, {'a', 'b'}, {'a', 'c'}}
def τ₄ : Set (Set Char) := {∅, {'a', 'c'}, {'b', 'c'}, {'c'}, {'a', 'b', 'c'}}

def IsTopology (τ : Set (Set Char)) : Prop :=
  X ∈ τ ∧ ∅ ∈ τ ∧
  (∀ S : Set (Set Char), S ⊆ τ → ⋃₀ S ∈ τ) ∧
  (∀ S : Set (Set Char), S ⊆ τ → ⋂₀ S ∈ τ)

theorem topology_classification :
  IsTopology τ₂ ∧ IsTopology τ₄ ∧ ¬IsTopology τ₁ ∧ ¬IsTopology τ₃ :=
sorry

end NUMINAMATH_CALUDE_topology_classification_l2063_206356


namespace NUMINAMATH_CALUDE_dissimilar_terms_expansion_l2063_206352

/-- The number of dissimilar terms in the expansion of (a + b + c + d)^12 -/
def dissimilar_terms : ℕ := 455

/-- The number of variables in the expansion -/
def num_variables : ℕ := 4

/-- The power to which the sum is raised -/
def power : ℕ := 12

/-- Theorem stating that the number of dissimilar terms in (a + b + c + d)^12 is 455 -/
theorem dissimilar_terms_expansion :
  dissimilar_terms = Nat.choose (power + num_variables - 1) (num_variables - 1) := by
  sorry

end NUMINAMATH_CALUDE_dissimilar_terms_expansion_l2063_206352


namespace NUMINAMATH_CALUDE_license_plate_increase_l2063_206354

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The number of digits (0-9) -/
def digit_size : ℕ := 10

/-- The number of letters in a license plate -/
def letter_count : ℕ := 2

/-- The number of digits in the old license plate format -/
def old_digit_count : ℕ := 3

/-- The number of digits in the new license plate format -/
def new_digit_count : ℕ := 4

/-- The ratio of new possible license plates to old possible license plates -/
def license_plate_ratio : ℚ :=
  (alphabet_size ^ letter_count * digit_size ^ new_digit_count) /
  (alphabet_size ^ letter_count * digit_size ^ old_digit_count)

theorem license_plate_increase : license_plate_ratio = 10 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_increase_l2063_206354


namespace NUMINAMATH_CALUDE_not_divisible_by_four_l2063_206319

theorem not_divisible_by_four (n : ℤ) : ¬(4 ∣ (1 + n + n^2 + n^3 + n^4)) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_four_l2063_206319


namespace NUMINAMATH_CALUDE_sphere_radius_l2063_206320

/-- Given a sphere and a pole under parallel sun rays, where the sphere's shadow extends
    12 meters from the point of contact with the ground, and a 3-meter tall pole casts
    a 4-meter shadow, the radius of the sphere is 9 meters. -/
theorem sphere_radius (shadow_length : ℝ) (pole_height : ℝ) (pole_shadow : ℝ) :
  shadow_length = 12 →
  pole_height = 3 →
  pole_shadow = 4 →
  ∃ (sphere_radius : ℝ), sphere_radius = 9 :=
by sorry

end NUMINAMATH_CALUDE_sphere_radius_l2063_206320


namespace NUMINAMATH_CALUDE_unique_product_l2063_206368

theorem unique_product (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a * b * c = 8 * (a + b + c))
  (h2 : c = a + b)
  (h3 : b = 2 * a) :
  a * b * c = 96 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_unique_product_l2063_206368


namespace NUMINAMATH_CALUDE_rock_paper_scissors_tournament_l2063_206353

/-- Represents the number of players in the tournament as a power of 2 -/
def n : ℕ := 4

/-- The number of players in the tournament -/
def num_players : ℕ := 2^n

/-- The number of matches in the tournament -/
def num_matches : ℕ := num_players - 1

/-- The number of possible choices for each player -/
def num_choices : ℕ := 3

/-- The number of possible tournament outcomes -/
def tournament_outcomes : ℕ := 2^num_matches

/-- The total number of possible choices for all players that result in a completed tournament -/
def total_variations : ℕ := num_choices * tournament_outcomes

theorem rock_paper_scissors_tournament :
  total_variations = 3 * 2^15 := by sorry

end NUMINAMATH_CALUDE_rock_paper_scissors_tournament_l2063_206353


namespace NUMINAMATH_CALUDE_complex_exp_210_deg_60th_power_l2063_206394

theorem complex_exp_210_deg_60th_power : 
  (Complex.exp (210 * π / 180 * Complex.I)) ^ 60 = 1 := by sorry

end NUMINAMATH_CALUDE_complex_exp_210_deg_60th_power_l2063_206394


namespace NUMINAMATH_CALUDE_common_internal_tangent_length_l2063_206327

theorem common_internal_tangent_length
  (center_distance : ℝ)
  (radius1 : ℝ)
  (radius2 : ℝ)
  (h1 : center_distance = 50)
  (h2 : radius1 = 7)
  (h3 : radius2 = 10) :
  Real.sqrt (center_distance^2 - (radius1 + radius2)^2) = Real.sqrt 2211 :=
by sorry

end NUMINAMATH_CALUDE_common_internal_tangent_length_l2063_206327


namespace NUMINAMATH_CALUDE_ticket_price_is_eight_l2063_206334

/-- Represents a movie theater with its capacity, tickets sold, and lost revenue --/
structure MovieTheater where
  capacity : ℕ
  ticketsSold : ℕ
  lostRevenue : ℚ

/-- Calculates the ticket price given the theater's information --/
def calculateTicketPrice (theater : MovieTheater) : ℚ :=
  theater.lostRevenue / (theater.capacity - theater.ticketsSold)

/-- Theorem stating that the ticket price is $8 for the given conditions --/
theorem ticket_price_is_eight :
  let theater : MovieTheater := { capacity := 50, ticketsSold := 24, lostRevenue := 208 }
  calculateTicketPrice theater = 8 := by sorry

end NUMINAMATH_CALUDE_ticket_price_is_eight_l2063_206334


namespace NUMINAMATH_CALUDE_quadratic_solution_product_l2063_206330

theorem quadratic_solution_product (x : ℝ) : 
  (49 = -2 * x^2 - 8 * x) → (∃ y : ℝ, (49 = -2 * y^2 - 8 * y) ∧ x * y = 49/2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_solution_product_l2063_206330


namespace NUMINAMATH_CALUDE_inscribing_square_area_is_one_l2063_206361

/-- The circle equation -/
def circle_equation (x y : ℝ) : Prop :=
  3 * x^2 + 3 * y^2 - 15 * x + 9 * y + 27 = 0

/-- The square that inscribes the circle -/
structure InscribingSquare where
  side_length : ℝ
  center_x : ℝ
  center_y : ℝ
  parallel_to_y_axis : Prop

/-- The theorem stating that the area of the inscribing square is 1 -/
theorem inscribing_square_area_is_one :
  ∃ (s : InscribingSquare), s.side_length ^ 2 = 1 ∧
  (∀ (x y : ℝ), circle_equation x y →
    (|x - s.center_x| ≤ s.side_length / 2 ∧
     |y - s.center_y| ≤ s.side_length / 2)) :=
sorry

end NUMINAMATH_CALUDE_inscribing_square_area_is_one_l2063_206361


namespace NUMINAMATH_CALUDE_factorization_equality_l2063_206367

theorem factorization_equality (x y : ℝ) : 2 * x^2 - 8 * y^2 = 2 * (x + 2*y) * (x - 2*y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2063_206367


namespace NUMINAMATH_CALUDE_remainder_95_equals_12_l2063_206324

theorem remainder_95_equals_12 (x : ℤ) : x % 19 = 12 → x % 95 = 12 := by
  sorry

end NUMINAMATH_CALUDE_remainder_95_equals_12_l2063_206324
