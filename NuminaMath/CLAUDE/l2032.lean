import Mathlib

namespace NUMINAMATH_CALUDE_ellipse_k_range_l2032_203221

-- Define the equation of the ellipse
def is_ellipse (k : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (k - 1) + y^2 / (9 - k) = 1

-- Define the range of k
def valid_k_range (k : ℝ) : Prop :=
  (1 < k ∧ k < 5) ∨ (5 < k ∧ k < 9)

-- Theorem stating the relationship between the ellipse equation and the range of k
theorem ellipse_k_range :
  ∀ k : ℝ, is_ellipse k ↔ valid_k_range k :=
sorry

end NUMINAMATH_CALUDE_ellipse_k_range_l2032_203221


namespace NUMINAMATH_CALUDE_gary_egg_collection_l2032_203241

/-- The number of chickens Gary starts with -/
def initial_chickens : ℕ := 4

/-- The factor by which the number of chickens increases after two years -/
def growth_factor : ℕ := 8

/-- The number of eggs each chicken lays per day -/
def eggs_per_chicken_per_day : ℕ := 6

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The number of eggs Gary collects every week after two years -/
def weekly_egg_collection : ℕ := initial_chickens * growth_factor * eggs_per_chicken_per_day * days_in_week

theorem gary_egg_collection : weekly_egg_collection = 1344 := by
  sorry

end NUMINAMATH_CALUDE_gary_egg_collection_l2032_203241


namespace NUMINAMATH_CALUDE_power_and_division_equality_l2032_203289

theorem power_and_division_equality : (12 : ℕ)^2 * 6^4 / 432 = 432 := by
  sorry

end NUMINAMATH_CALUDE_power_and_division_equality_l2032_203289


namespace NUMINAMATH_CALUDE_sequence_characterization_l2032_203232

theorem sequence_characterization (a : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 1) / a n = (a (n + 2) - a (n + 1)) / (a (n + 1) - a n)) →
  (∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n) :=
by sorry

end NUMINAMATH_CALUDE_sequence_characterization_l2032_203232


namespace NUMINAMATH_CALUDE_extreme_points_property_l2032_203272

theorem extreme_points_property (a : ℝ) (f : ℝ → ℝ) (x₁ x₂ : ℝ) :
  0 < a → a < 1/2 →
  (∀ x, f x = x * (Real.log x - a * x)) →
  x₁ < x₂ →
  (∃ ε > 0, ∀ x ∈ Set.Ioo (x₁ - ε) (x₁ + ε), f x ≤ f x₁) →
  (∃ ε > 0, ∀ x ∈ Set.Ioo (x₂ - ε) (x₂ + ε), f x ≤ f x₂) →
  f x₁ < 0 ∧ f x₂ > -1/2 := by
sorry

end NUMINAMATH_CALUDE_extreme_points_property_l2032_203272


namespace NUMINAMATH_CALUDE_m_range_l2032_203226

theorem m_range (m : ℝ) : 
  (¬(∃ x₀ : ℝ, m * x₀^2 + 1 < 1) ∧ ∀ x : ℝ, x^2 + m*x + 1 ≥ 0) ↔ 
  -2 ≤ m ∧ m ≤ 2 := by
sorry

end NUMINAMATH_CALUDE_m_range_l2032_203226


namespace NUMINAMATH_CALUDE_infinitely_many_silesian_infinitely_many_non_silesian_l2032_203231

/-- An integer n is Silesian if there exist positive integers a, b, c such that
    n = (a² + b² + c²) / (ab + bc + ca) -/
def is_silesian (n : ℤ) : Prop :=
  ∃ (a b c : ℕ+), n = (a.val^2 + b.val^2 + c.val^2) / (a.val * b.val + b.val * c.val + c.val * a.val)

/-- There are infinitely many Silesian integers -/
theorem infinitely_many_silesian : ∀ N : ℕ, ∃ n : ℤ, n > N ∧ is_silesian n :=
sorry

/-- There are infinitely many positive integers that are not Silesian -/
theorem infinitely_many_non_silesian : ∀ N : ℕ, ∃ k : ℕ, k > N ∧ ¬is_silesian (3 * k) :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_silesian_infinitely_many_non_silesian_l2032_203231


namespace NUMINAMATH_CALUDE_circle_area_solution_l2032_203203

theorem circle_area_solution :
  ∃! (x y z : ℕ), 6 * x + 15 * y + 83 * z = 220 ∧ x = 4 ∧ y = 2 ∧ z = 2 := by
sorry

end NUMINAMATH_CALUDE_circle_area_solution_l2032_203203


namespace NUMINAMATH_CALUDE_store_inventory_price_l2032_203252

theorem store_inventory_price (total_items : ℕ) (discount_rate : ℚ) (sold_rate : ℚ)
  (debt : ℕ) (remaining : ℕ) :
  total_items = 2000 →
  discount_rate = 80 / 100 →
  sold_rate = 90 / 100 →
  debt = 15000 →
  remaining = 3000 →
  ∃ (price : ℚ), price = 50 ∧
    (1 - discount_rate) * (sold_rate * total_items) * price = debt + remaining :=
by sorry

end NUMINAMATH_CALUDE_store_inventory_price_l2032_203252


namespace NUMINAMATH_CALUDE_rational_expression_iff_zero_l2032_203217

theorem rational_expression_iff_zero (x : ℝ) : 
  ∃ (q : ℚ), x + Real.sqrt (x^2 + 4) - 1 / (x + Real.sqrt (x^2 + 4)) = q ↔ x = 0 := by
  sorry

end NUMINAMATH_CALUDE_rational_expression_iff_zero_l2032_203217


namespace NUMINAMATH_CALUDE_student_wrestling_match_l2032_203269

theorem student_wrestling_match (n : ℕ) : n * (n - 1) / 2 = 91 → n = 14 := by
  sorry

end NUMINAMATH_CALUDE_student_wrestling_match_l2032_203269


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l2032_203291

theorem polynomial_division_remainder (x : ℝ) :
  ∃ q : ℝ → ℝ, x^4 + 2 = (x - 2)^2 * q x + (32*x - 46) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l2032_203291


namespace NUMINAMATH_CALUDE_least_sum_m_n_l2032_203242

theorem least_sum_m_n (m n : ℕ+) 
  (h1 : Nat.gcd (m + n) 330 = 1)
  (h2 : ∃ k : ℕ, m^m.val = k * n^n.val)
  (h3 : ¬ ∃ k : ℕ, m = k * n) :
  ∀ p q : ℕ+, 
    (Nat.gcd (p + q) 330 = 1) → 
    (∃ k : ℕ, p^p.val = k * q^q.val) → 
    (¬ ∃ k : ℕ, p = k * q) → 
    m + n ≤ p + q :=
by
  sorry

end NUMINAMATH_CALUDE_least_sum_m_n_l2032_203242


namespace NUMINAMATH_CALUDE_janet_tile_savings_l2032_203214

/-- Calculates the cost difference between two tile options for a given wall area and tile density -/
def tile_cost_difference (
  wall1_length wall1_width wall2_length wall2_width : ℝ)
  (tiles_per_sqft : ℝ)
  (turquoise_cost purple_cost : ℝ) : ℝ :=
  let total_area := wall1_length * wall1_width + wall2_length * wall2_width
  let total_tiles := total_area * tiles_per_sqft
  let cost_diff_per_tile := turquoise_cost - purple_cost
  total_tiles * cost_diff_per_tile

/-- The cost difference between turquoise and purple tiles for Janet's bathroom -/
theorem janet_tile_savings : 
  tile_cost_difference 5 8 7 8 4 13 11 = 768 := by
  sorry

end NUMINAMATH_CALUDE_janet_tile_savings_l2032_203214


namespace NUMINAMATH_CALUDE_sin_75_cos_15_minus_1_l2032_203299

theorem sin_75_cos_15_minus_1 : 
  2 * Real.sin (75 * π / 180) * Real.cos (15 * π / 180) - 1 = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_75_cos_15_minus_1_l2032_203299


namespace NUMINAMATH_CALUDE_problem_solution_l2032_203290

theorem problem_solution : -1^2015 + |(-3)| - (1/2)^2 * 8 + (-2)^3 / 4 = -2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2032_203290


namespace NUMINAMATH_CALUDE_race_time_difference_l2032_203270

def race_length : ℝ := 15
def malcolm_speed : ℝ := 6
def joshua_speed : ℝ := 7

theorem race_time_difference : 
  let malcolm_time := race_length * malcolm_speed
  let joshua_time := race_length * joshua_speed
  joshua_time - malcolm_time = 15 := by
sorry

end NUMINAMATH_CALUDE_race_time_difference_l2032_203270


namespace NUMINAMATH_CALUDE_right_triangle_division_l2032_203209

theorem right_triangle_division (n : ℝ) (h : n > 0) :
  ∀ (rect_area rect_short rect_long small_triangle1_area : ℝ),
    rect_short > 0 →
    rect_long > 0 →
    rect_area > 0 →
    small_triangle1_area > 0 →
    rect_long = 3 * rect_short →
    rect_area = rect_short * rect_long →
    small_triangle1_area = n * rect_area →
    ∃ (small_triangle2_area : ℝ),
      small_triangle2_area > 0 ∧
      small_triangle2_area / rect_area = 1 / (4 * n) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_division_l2032_203209


namespace NUMINAMATH_CALUDE_square_2007_position_l2032_203223

-- Define the possible square positions
inductive SquarePosition
  | ABCD
  | DCBA

-- Define the transformations
def rotate180 (pos : SquarePosition) : SquarePosition :=
  match pos with
  | SquarePosition.ABCD => SquarePosition.DCBA
  | SquarePosition.DCBA => SquarePosition.ABCD

def reflectHorizontal (pos : SquarePosition) : SquarePosition := pos

-- Define the sequence of transformations
def transformSquare (n : Nat) : SquarePosition :=
  if n % 2 = 1 then
    rotate180 SquarePosition.ABCD
  else
    reflectHorizontal (rotate180 SquarePosition.ABCD)

-- State the theorem
theorem square_2007_position :
  transformSquare 2007 = SquarePosition.DCBA := by sorry

end NUMINAMATH_CALUDE_square_2007_position_l2032_203223


namespace NUMINAMATH_CALUDE_helmet_costs_and_profit_l2032_203250

/-- Represents the cost and sales information for helmets --/
structure HelmetData where
  costA3B4 : ℕ  -- Cost of 3 type A and 4 type B helmets
  costA6B2 : ℕ  -- Cost of 6 type A and 2 type B helmets
  basePrice : ℝ  -- Base selling price of type A helmet
  baseSales : ℕ  -- Number of helmets sold at base price
  priceIncrement : ℝ  -- Price increment
  salesDecrement : ℕ  -- Sales decrement per price increment

/-- Theorem about helmet costs and profit --/
theorem helmet_costs_and_profit (data : HelmetData)
  (h1 : data.costA3B4 = 288)
  (h2 : data.costA6B2 = 306)
  (h3 : data.basePrice = 50)
  (h4 : data.baseSales = 100)
  (h5 : data.priceIncrement = 5)
  (h6 : data.salesDecrement = 10) :
  ∃ (costA costB : ℕ) (profitFunc : ℝ → ℝ) (maxProfit : ℝ),
    costA = 36 ∧
    costB = 45 ∧
    (∀ x, 50 ≤ x ∧ x ≤ 100 → profitFunc x = -2 * x^2 + 272 * x - 7200) ∧
    maxProfit = 2048 := by
  sorry

end NUMINAMATH_CALUDE_helmet_costs_and_profit_l2032_203250


namespace NUMINAMATH_CALUDE_laptop_sale_price_l2032_203230

def original_price : ℝ := 1000.00
def discount1 : ℝ := 0.10
def discount2 : ℝ := 0.20
def discount3 : ℝ := 0.15

theorem laptop_sale_price :
  original_price * (1 - discount1) * (1 - discount2) * (1 - discount3) = 612.00 := by
sorry

end NUMINAMATH_CALUDE_laptop_sale_price_l2032_203230


namespace NUMINAMATH_CALUDE_unique_number_l2032_203245

theorem unique_number : ∃! x : ℚ, x / 3 = x - 4 := by sorry

end NUMINAMATH_CALUDE_unique_number_l2032_203245


namespace NUMINAMATH_CALUDE_g_is_zero_l2032_203265

noncomputable def g (x : ℝ) : ℝ := Real.sqrt (2 * Real.sin x ^ 4 + 3 * Real.cos x ^ 2) - Real.sqrt (2 * Real.cos x ^ 4 + 3 * Real.sin x ^ 2)

theorem g_is_zero : ∀ x : ℝ, g x = 0 := by
  sorry

end NUMINAMATH_CALUDE_g_is_zero_l2032_203265


namespace NUMINAMATH_CALUDE_fruit_seller_apples_l2032_203267

theorem fruit_seller_apples (initial_apples : ℕ) : 
  (initial_apples * 40 / 100 = 560) → initial_apples = 1400 := by
  sorry

end NUMINAMATH_CALUDE_fruit_seller_apples_l2032_203267


namespace NUMINAMATH_CALUDE_article_profit_l2032_203208

/-- If selling an article at 2/3 of its original price results in a 20% loss,
    then selling it at the original price results in a 20% profit. -/
theorem article_profit (original_price : ℝ) (cost_price : ℝ) 
    (h1 : original_price > 0) 
    (h2 : cost_price > 0)
    (h3 : (2/3) * original_price = 0.8 * cost_price) : 
  (original_price - cost_price) / cost_price = 0.2 := by
  sorry

#check article_profit

end NUMINAMATH_CALUDE_article_profit_l2032_203208


namespace NUMINAMATH_CALUDE_solution_set_inequality_l2032_203256

theorem solution_set_inequality (x : ℝ) :
  (Set.Iio (1/3 : ℝ)) = {x | Real.sqrt (x^2 - 2*x + 1) > 2*x} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l2032_203256


namespace NUMINAMATH_CALUDE_solve_equation_l2032_203280

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^2 - 1

-- State the theorem
theorem solve_equation (a : ℝ) (h1 : 0 < a) (h2 : a < 3) (h3 : f a = 7) : a = 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2032_203280


namespace NUMINAMATH_CALUDE_negation_of_all_is_some_not_l2032_203236

variable (α : Type)
variable (HonorStudent : α → Prop)
variable (ReceivesScholarship : α → Prop)

theorem negation_of_all_is_some_not :
  ¬(∀ x, HonorStudent x → ReceivesScholarship x) ↔ 
  (∃ x, HonorStudent x ∧ ¬ReceivesScholarship x) :=
sorry

end NUMINAMATH_CALUDE_negation_of_all_is_some_not_l2032_203236


namespace NUMINAMATH_CALUDE_number_pair_uniqueness_l2032_203216

theorem number_pair_uniqueness (S P : ℝ) (h : S^2 ≥ 4*P) :
  let x₁ := (S + Real.sqrt (S^2 - 4*P)) / 2
  let x₂ := (S - Real.sqrt (S^2 - 4*P)) / 2
  let y₁ := S - x₁
  let y₂ := S - x₂
  ∀ x y : ℝ, (x + y = S ∧ x * y = P) ↔ ((x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂)) :=
by
  sorry

end NUMINAMATH_CALUDE_number_pair_uniqueness_l2032_203216


namespace NUMINAMATH_CALUDE_sum_of_cubics_degree_at_most_3_l2032_203218

-- Define a cubic polynomial
def CubicPolynomial (R : Type*) [CommRing R] := {p : Polynomial R // p.degree ≤ 3}

-- Theorem statement
theorem sum_of_cubics_degree_at_most_3 {R : Type*} [CommRing R] 
  (A B : CubicPolynomial R) : 
  (A.val + B.val).degree ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubics_degree_at_most_3_l2032_203218


namespace NUMINAMATH_CALUDE_hexagon_unit_triangles_l2032_203294

/-- The number of unit equilateral triangles in a regular hexagon -/
def num_unit_triangles_in_hexagon (side_length : ℕ) : ℕ :=
  6 * side_length^2

/-- Theorem: A regular hexagon with side length 5 contains 150 unit equilateral triangles -/
theorem hexagon_unit_triangles :
  num_unit_triangles_in_hexagon 5 = 150 := by
  sorry

#eval num_unit_triangles_in_hexagon 5

end NUMINAMATH_CALUDE_hexagon_unit_triangles_l2032_203294


namespace NUMINAMATH_CALUDE_unique_magnitude_for_complex_roots_l2032_203215

theorem unique_magnitude_for_complex_roots (z : ℂ) : 
  z^2 - 6*z + 20 = 0 → ∃! m : ℝ, ∃ z : ℂ, z^2 - 6*z + 20 = 0 ∧ Complex.abs z = m :=
by
  sorry

end NUMINAMATH_CALUDE_unique_magnitude_for_complex_roots_l2032_203215


namespace NUMINAMATH_CALUDE_intersection_points_theorem_l2032_203268

theorem intersection_points_theorem : 
  ∃ (k₁ k₂ k₃ k₄ : ℕ+), 
    k₁ + k₂ + k₃ + k₄ = 100 ∧ 
    k₁^2 + k₂^2 + k₃^2 + k₄^2 = 5996 ∧
    k₁ * k₂ + k₁ * k₃ + k₁ * k₄ + k₂ * k₃ + k₂ * k₄ + k₃ * k₄ = 2002 :=
by sorry

end NUMINAMATH_CALUDE_intersection_points_theorem_l2032_203268


namespace NUMINAMATH_CALUDE_evaluate_expression_l2032_203211

theorem evaluate_expression : 
  let sixteen : ℝ := 2^4
  let eight : ℝ := 2^3
  ∀ ε > 0, |Real.sqrt ((sixteen^15 + eight^20) / (sixteen^7 + eight^21)) - (1/2)| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2032_203211


namespace NUMINAMATH_CALUDE_james_money_theorem_l2032_203229

/-- The amount of money James has after finding some bills -/
def jamesTotal (billsFound : ℕ) (billValue : ℕ) (walletAmount : ℕ) : ℕ :=
  billsFound * billValue + walletAmount

/-- Theorem stating that James has $135 after finding 3 $20 bills -/
theorem james_money_theorem :
  jamesTotal 3 20 75 = 135 := by
  sorry

end NUMINAMATH_CALUDE_james_money_theorem_l2032_203229


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2032_203219

def A : Set ℤ := {x | ∃ k : ℤ, x = 2 * k - 1}
def B : Set ℤ := {x | ∃ k : ℕ, x = 2 * k + 1 ∧ k < 3}

theorem intersection_of_A_and_B : A ∩ B = {1, 3, 5} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2032_203219


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2032_203202

/-- A quadratic equation x^2 + x + a = 0 has one positive root and one negative root -/
def has_one_positive_one_negative_root (a : ℝ) : Prop :=
  ∃ x y : ℝ, x > 0 ∧ y < 0 ∧ x^2 + x + a = 0 ∧ y^2 + y + a = 0

/-- The condition a < -1 is sufficient but not necessary for x^2 + x + a = 0 
    to have one positive and one negative root -/
theorem sufficient_not_necessary_condition :
  (∀ a : ℝ, a < -1 → has_one_positive_one_negative_root a) ∧
  (∃ a : ℝ, -1 ≤ a ∧ a < 0 ∧ has_one_positive_one_negative_root a) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2032_203202


namespace NUMINAMATH_CALUDE_min_value_x_plus_4y_l2032_203278

theorem min_value_x_plus_4y (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 2*x*y) :
  x + 4*y ≥ 9/2 ∧ (x + 4*y = 9/2 ↔ x = 3/2 ∧ y = 3/4) := by
  sorry

end NUMINAMATH_CALUDE_min_value_x_plus_4y_l2032_203278


namespace NUMINAMATH_CALUDE_min_value_theorem_l2032_203277

theorem min_value_theorem (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 2 * a + 3 * b = 6) :
  (2 / a + 3 / b) ≥ 25 / 6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2032_203277


namespace NUMINAMATH_CALUDE_abcd_sum_l2032_203238

def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

theorem abcd_sum (A B C D : ℕ) : 
  is_digit A → is_digit B → is_digit C → is_digit D →
  A ≠ B → A ≠ C → A ≠ D → B ≠ C → B ≠ D → C ≠ D →
  C ≠ 0 → D ≠ 0 →
  (A + B) % (C + D) = 0 →
  ∀ E F, is_digit E → is_digit F → E ≠ 0 → F ≠ 0 → E + F < C + D →
  A + B = 15 :=
sorry

end NUMINAMATH_CALUDE_abcd_sum_l2032_203238


namespace NUMINAMATH_CALUDE_tour_group_composition_l2032_203213

/-- Given a group of 18 people where selecting one male (excluding two ineligible men) 
    and one female results in 64 different combinations, prove that there are 10 men 
    and 8 women in the group. -/
theorem tour_group_composition :
  ∀ (num_men : ℕ),
    (num_men - 2) * (18 - num_men) = 64 →
    num_men = 10 ∧ 18 - num_men = 8 := by
  sorry

end NUMINAMATH_CALUDE_tour_group_composition_l2032_203213


namespace NUMINAMATH_CALUDE_remainder_sum_mod_seven_l2032_203212

theorem remainder_sum_mod_seven : (9^7 + 6^9 + 5^11) % 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_mod_seven_l2032_203212


namespace NUMINAMATH_CALUDE_sum_black_eq_sum_white_l2032_203205

/-- Represents a frame in a multiplication table -/
structure Frame (m n : ℕ) :=
  (is_odd_m : Odd m)
  (is_odd_n : Odd n)

/-- The sum of numbers in black squares of the frame -/
def sum_black (f : Frame m n) : ℕ := sorry

/-- The sum of numbers in white squares of the frame -/
def sum_white (f : Frame m n) : ℕ := sorry

/-- Theorem stating that the sum of numbers in black squares equals the sum of numbers in white squares -/
theorem sum_black_eq_sum_white (m n : ℕ) (f : Frame m n) :
  sum_black f = sum_white f := by sorry

end NUMINAMATH_CALUDE_sum_black_eq_sum_white_l2032_203205


namespace NUMINAMATH_CALUDE_trays_needed_to_replace_ice_l2032_203227

def ice_cubes_in_glass : ℕ := 8
def ice_cubes_in_pitcher : ℕ := 2 * ice_cubes_in_glass
def spaces_per_tray : ℕ := 12

theorem trays_needed_to_replace_ice : 
  (ice_cubes_in_glass + ice_cubes_in_pitcher) / spaces_per_tray = 2 := by
  sorry

end NUMINAMATH_CALUDE_trays_needed_to_replace_ice_l2032_203227


namespace NUMINAMATH_CALUDE_russian_football_championship_l2032_203261

/-- Represents a football championship. -/
structure Championship where
  teams : ℕ
  matches_per_pair : ℕ

/-- Calculate the number of matches a single team plays. -/
def matches_per_team (c : Championship) : ℕ :=
  (c.teams - 1) * c.matches_per_pair

/-- Calculate the total number of matches in the championship. -/
def total_matches (c : Championship) : ℕ :=
  (c.teams * matches_per_team c) / 2

theorem russian_football_championship 
  (c : Championship) 
  (h1 : c.teams = 16) 
  (h2 : c.matches_per_pair = 2) : 
  matches_per_team c = 30 ∧ total_matches c = 240 := by
  sorry

#eval matches_per_team ⟨16, 2⟩
#eval total_matches ⟨16, 2⟩

end NUMINAMATH_CALUDE_russian_football_championship_l2032_203261


namespace NUMINAMATH_CALUDE_intersection_range_l2032_203296

theorem intersection_range (k : ℝ) : 
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    x₁ ≠ x₂ ∧ 
    y₁ = k * x₁ + 2 ∧ 
    y₂ = k * x₂ + 2 ∧ 
    x₁ = Real.sqrt (y₁^2 + 6) ∧ 
    x₂ = Real.sqrt (y₂^2 + 6)) →
  -Real.sqrt 15 / 3 < k ∧ k < -1 := by
sorry

end NUMINAMATH_CALUDE_intersection_range_l2032_203296


namespace NUMINAMATH_CALUDE_gcd_power_two_minus_one_l2032_203240

theorem gcd_power_two_minus_one : 
  Nat.gcd (2^1023 - 1) (2^1034 - 1) = 2^11 - 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_power_two_minus_one_l2032_203240


namespace NUMINAMATH_CALUDE_smallest_two_digit_with_product_12_l2032_203262

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_product (n : ℕ) : ℕ :=
  (n / 10) * (n % 10)

theorem smallest_two_digit_with_product_12 :
  ∀ n : ℕ, is_two_digit n → digit_product n = 12 → 26 ≤ n :=
sorry

end NUMINAMATH_CALUDE_smallest_two_digit_with_product_12_l2032_203262


namespace NUMINAMATH_CALUDE_square_sum_given_sum_and_product_l2032_203293

theorem square_sum_given_sum_and_product (x y : ℝ) 
  (h1 : x + y = 5) (h2 : x * y = 2) : x^2 + y^2 = 21 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_given_sum_and_product_l2032_203293


namespace NUMINAMATH_CALUDE_algebraic_expression_equality_l2032_203266

theorem algebraic_expression_equality (x : ℝ) : 
  x^2 + 2*x + 5 = 6 → 2*x^2 + 4*x + 15 = 17 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_equality_l2032_203266


namespace NUMINAMATH_CALUDE_range_of_k_l2032_203234

/-- Represents an ellipse equation -/
def is_ellipse (k : ℝ) : Prop :=
  2 * k - 1 > 0 ∧ k - 1 > 0

/-- Represents a hyperbola equation -/
def is_hyperbola (k : ℝ) : Prop :=
  (4 - k) * (k - 3) < 0

/-- The main theorem stating the range of k -/
theorem range_of_k :
  (∀ k : ℝ, (is_ellipse k ∨ is_hyperbola k) ∧ ¬(is_ellipse k ∧ is_hyperbola k)) →
  (∀ k : ℝ, k ≤ 1 ∨ (3 ≤ k ∧ k ≤ 4)) :=
sorry

end NUMINAMATH_CALUDE_range_of_k_l2032_203234


namespace NUMINAMATH_CALUDE_shari_walked_13_miles_l2032_203247

/-- Represents Shari's walking pattern -/
structure WalkingPattern where
  rate1 : ℝ  -- Rate for the first phase in miles per hour
  time1 : ℝ  -- Time for the first phase in hours
  rate2 : ℝ  -- Rate for the second phase in miles per hour
  time2 : ℝ  -- Time for the second phase in hours

/-- Calculates the total distance walked given a WalkingPattern -/
def totalDistance (w : WalkingPattern) : ℝ :=
  w.rate1 * w.time1 + w.rate2 * w.time2

/-- Shari's actual walking pattern -/
def sharisWalk : WalkingPattern :=
  { rate1 := 4
    time1 := 2
    rate2 := 5
    time2 := 1 }

/-- Theorem stating that Shari walked 13 miles in total -/
theorem shari_walked_13_miles :
  totalDistance sharisWalk = 13 := by
  sorry


end NUMINAMATH_CALUDE_shari_walked_13_miles_l2032_203247


namespace NUMINAMATH_CALUDE_percentage_decrease_l2032_203201

theorem percentage_decrease (x y z : ℝ) : 
  x = 1.2 * y → x = 0.48 * z → y = 0.4 * z :=
by sorry

end NUMINAMATH_CALUDE_percentage_decrease_l2032_203201


namespace NUMINAMATH_CALUDE_quadratic_always_positive_implies_a_range_l2032_203274

theorem quadratic_always_positive_implies_a_range (a : ℝ) :
  (∀ x : ℝ, 2 * x^2 + (a - 1) * x + 1/2 > 0) ↔ -1 < a ∧ a < 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_implies_a_range_l2032_203274


namespace NUMINAMATH_CALUDE_overall_profit_calculation_l2032_203233

/-- Calculate the overall profit from selling a refrigerator and a mobile phone -/
theorem overall_profit_calculation (refrigerator_cost mobile_cost : ℕ) 
  (refrigerator_loss_percent mobile_profit_percent : ℚ) :
  refrigerator_cost = 15000 →
  mobile_cost = 8000 →
  refrigerator_loss_percent = 5 / 100 →
  mobile_profit_percent = 10 / 100 →
  (refrigerator_cost * (1 - refrigerator_loss_percent) + 
   mobile_cost * (1 + mobile_profit_percent)) - 
  (refrigerator_cost + mobile_cost) = 50 := by
  sorry

end NUMINAMATH_CALUDE_overall_profit_calculation_l2032_203233


namespace NUMINAMATH_CALUDE_bicyclist_distance_l2032_203206

/-- Represents the bicyclist's journey --/
structure Journey where
  speed : ℝ
  time : ℝ
  distance : ℝ

/-- The original journey satisfies the problem conditions --/
def satisfies_conditions (j : Journey) : Prop :=
  j.distance = j.speed * j.time ∧
  j.distance = (j.speed + 1) * (2/3 * j.time) ∧
  j.distance = (j.speed - 1) * (j.time + 1)

/-- The theorem to be proved --/
theorem bicyclist_distance :
  ∃ j : Journey, satisfies_conditions j ∧ j.distance = 2 := by
  sorry

end NUMINAMATH_CALUDE_bicyclist_distance_l2032_203206


namespace NUMINAMATH_CALUDE_remainder_problem_l2032_203225

theorem remainder_problem (n : ℤ) (h : n % 8 = 3) : (4 * n - 10) % 8 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l2032_203225


namespace NUMINAMATH_CALUDE_complex_simplification_l2032_203282

/-- The imaginary unit -/
axiom I : ℂ

/-- The property of the imaginary unit -/
axiom I_squared : I^2 = -1

/-- Theorem stating the equality of the complex expressions -/
theorem complex_simplification : 7 * (4 - 2*I) - 2*I * (7 - 3*I) = 22 - 28*I := by
  sorry

end NUMINAMATH_CALUDE_complex_simplification_l2032_203282


namespace NUMINAMATH_CALUDE_carries_tshirt_purchase_l2032_203260

/-- The cost of a single t-shirt in dollars -/
def tshirt_cost : ℚ := 9.95

/-- The number of t-shirts Carrie bought -/
def num_tshirts : ℕ := 20

/-- The total cost of Carrie's t-shirt purchase -/
def total_cost : ℚ := tshirt_cost * num_tshirts

/-- Theorem stating that the total cost of Carrie's t-shirt purchase is $199 -/
theorem carries_tshirt_purchase : total_cost = 199 := by
  sorry

end NUMINAMATH_CALUDE_carries_tshirt_purchase_l2032_203260


namespace NUMINAMATH_CALUDE_janet_horses_count_l2032_203222

def fertilizer_per_horse_per_day : ℕ := 5
def total_acres : ℕ := 20
def fertilizer_per_acre : ℕ := 400
def acres_fertilized_per_day : ℕ := 4
def days_to_fertilize : ℕ := 25

def janet_horses : ℕ := 64

theorem janet_horses_count : janet_horses = 
  (total_acres * fertilizer_per_acre) / 
  (fertilizer_per_horse_per_day * days_to_fertilize) := by
  sorry

end NUMINAMATH_CALUDE_janet_horses_count_l2032_203222


namespace NUMINAMATH_CALUDE_probability_three_girls_l2032_203246

/-- The probability of choosing 3 girls from a group of 15 members (8 girls and 7 boys) -/
theorem probability_three_girls (total : ℕ) (girls : ℕ) (boys : ℕ) (chosen : ℕ) : 
  total = 15 → girls = 8 → boys = 7 → chosen = 3 →
  (Nat.choose girls chosen : ℚ) / (Nat.choose total chosen : ℚ) = 8 / 65 := by
sorry

end NUMINAMATH_CALUDE_probability_three_girls_l2032_203246


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2032_203275

theorem polynomial_factorization (x : ℝ) : 3 * x^2 + 3 * x - 18 = 3 * (x + 3) * (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2032_203275


namespace NUMINAMATH_CALUDE_P_in_fourth_quadrant_iff_l2032_203207

/-- A point in the Cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of a point P(2x+6, 5x) -/
def P (x : ℝ) : Point :=
  { x := 2*x + 6, y := 5*x }

/-- Definition of being in the fourth quadrant -/
def in_fourth_quadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- Theorem stating the range of x for P to be in the fourth quadrant -/
theorem P_in_fourth_quadrant_iff (x : ℝ) :
  in_fourth_quadrant (P x) ↔ -3 < x ∧ x < 0 := by
  sorry

end NUMINAMATH_CALUDE_P_in_fourth_quadrant_iff_l2032_203207


namespace NUMINAMATH_CALUDE_positive_sum_product_iff_l2032_203220

theorem positive_sum_product_iff (a b : ℝ) : (a > 0 ∧ b > 0) ↔ (a + b > 0 ∧ a * b > 0) := by
  sorry

end NUMINAMATH_CALUDE_positive_sum_product_iff_l2032_203220


namespace NUMINAMATH_CALUDE_positive_real_sum_one_inequality_l2032_203284

theorem positive_real_sum_one_inequality (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 1) : 
  (1/a - 1) * (1/b - 1) * (1/c - 1) ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_positive_real_sum_one_inequality_l2032_203284


namespace NUMINAMATH_CALUDE_remaining_uncracked_seashells_l2032_203288

def tom_seashells : ℕ := 15
def fred_seashells : ℕ := 43
def cracked_seashells : ℕ := 29
def giveaway_percentage : ℚ := 40 / 100

theorem remaining_uncracked_seashells :
  let total_seashells := tom_seashells + fred_seashells
  let uncracked_seashells := total_seashells - cracked_seashells
  let seashells_to_giveaway := ⌊(giveaway_percentage * uncracked_seashells : ℚ)⌋
  uncracked_seashells - seashells_to_giveaway = 18 := by sorry

end NUMINAMATH_CALUDE_remaining_uncracked_seashells_l2032_203288


namespace NUMINAMATH_CALUDE_white_washing_cost_l2032_203281

/-- Calculate the cost of white washing a room with given dimensions and openings. -/
theorem white_washing_cost
  (room_length room_width room_height : ℝ)
  (door_width door_height : ℝ)
  (window_width window_height : ℝ)
  (num_windows : ℕ)
  (cost_per_sqft : ℝ)
  (h_room_length : room_length = 25)
  (h_room_width : room_width = 15)
  (h_room_height : room_height = 12)
  (h_door_width : door_width = 6)
  (h_door_height : door_height = 3)
  (h_window_width : window_width = 4)
  (h_window_height : window_height = 3)
  (h_num_windows : num_windows = 3)
  (h_cost_per_sqft : cost_per_sqft = 6) :
  let total_wall_area := 2 * (room_length + room_width) * room_height
  let door_area := door_width * door_height
  let window_area := window_width * window_height
  let total_opening_area := door_area + num_windows * window_area
  let paintable_area := total_wall_area - total_opening_area
  let total_cost := paintable_area * cost_per_sqft
  total_cost = 5436 := by sorry


end NUMINAMATH_CALUDE_white_washing_cost_l2032_203281


namespace NUMINAMATH_CALUDE_recycling_drive_target_l2032_203298

/-- Calculates the target amount of kilos for a recycling drive given the number of sections,
    amount collected per section in two weeks, and additional amount needed. -/
def recycling_target (sections : ℕ) (kilos_per_section_two_weeks : ℕ) (additional_kilos : ℕ) : ℕ :=
  let kilos_per_section_per_week := kilos_per_section_two_weeks / 2
  let kilos_per_section_three_weeks := kilos_per_section_per_week * 3
  let total_collected := kilos_per_section_three_weeks * sections
  total_collected + additional_kilos

/-- The recycling drive target matches the calculated amount. -/
theorem recycling_drive_target :
  recycling_target 6 280 320 = 2840 := by
  sorry

end NUMINAMATH_CALUDE_recycling_drive_target_l2032_203298


namespace NUMINAMATH_CALUDE_find_divisor_l2032_203253

theorem find_divisor (dividend quotient remainder : ℕ) (h : dividend = quotient * 4 + remainder) :
  ∃ (divisor : ℕ), dividend = quotient * divisor + remainder ∧ divisor = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_find_divisor_l2032_203253


namespace NUMINAMATH_CALUDE_probability_estimate_l2032_203243

def is_hit (n : ℕ) : Bool :=
  n ≥ 3 ∧ n ≤ 9

def group_has_three_hits (group : List ℕ) : Bool :=
  (group.filter is_hit).length ≥ 3

def count_successful_groups (groups : List (List ℕ)) : ℕ :=
  (groups.filter group_has_three_hits).length

theorem probability_estimate (groups : List (List ℕ)) 
  (h1 : groups.length = 20) 
  (h2 : ∀ g ∈ groups, g.length = 4) 
  (h3 : ∀ g ∈ groups, ∀ n ∈ g, n ≥ 0 ∧ n ≤ 9) : 
  (count_successful_groups groups : ℚ) / groups.length = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_probability_estimate_l2032_203243


namespace NUMINAMATH_CALUDE_sandro_children_l2032_203279

/-- Calculates the total number of children Sandro has -/
def total_children (sons : ℕ) (daughter_ratio : ℕ) : ℕ :=
  sons + sons * daughter_ratio

theorem sandro_children :
  let sons := 3
  let daughter_ratio := 6
  total_children sons daughter_ratio = 21 := by
  sorry

end NUMINAMATH_CALUDE_sandro_children_l2032_203279


namespace NUMINAMATH_CALUDE_not_divides_power_minus_one_l2032_203251

theorem not_divides_power_minus_one (n : ℕ) (h : n > 1) : ¬(n ∣ (2^n - 1)) := by
  sorry

end NUMINAMATH_CALUDE_not_divides_power_minus_one_l2032_203251


namespace NUMINAMATH_CALUDE_f_range_l2032_203292

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 2*x

-- State the theorem
theorem f_range :
  ∀ x ∈ Set.Icc (-2 : ℝ) 1,
    -1 ≤ f x ∧ f x ≤ 3 ∧
    (∃ x₁ ∈ Set.Icc (-2 : ℝ) 1, f x₁ = -1) ∧
    (∃ x₂ ∈ Set.Icc (-2 : ℝ) 1, f x₂ = 3) :=
by sorry

end NUMINAMATH_CALUDE_f_range_l2032_203292


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l2032_203257

/-- An arithmetic sequence with a non-zero common difference -/
def ArithmeticSequence (a : ℕ → ℚ) (d : ℚ) : Prop :=
  d ≠ 0 ∧ ∀ n, a (n + 1) = a n + d

/-- Three terms of a sequence form a geometric sequence -/
def FormGeometricSequence (a : ℕ → ℚ) (i j k : ℕ) : Prop :=
  (a j) ^ 2 = a i * a k

theorem arithmetic_sequence_ratio (a : ℕ → ℚ) (d : ℚ) :
  ArithmeticSequence a d →
  FormGeometricSequence a 2 3 9 →
  (a 4 + a 5 + a 6) / (a 2 + a 3 + a 4) = 8 / 3 := by
  sorry

#check arithmetic_sequence_ratio

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l2032_203257


namespace NUMINAMATH_CALUDE_gcd_of_315_and_2016_l2032_203276

theorem gcd_of_315_and_2016 : Nat.gcd 315 2016 = 63 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_315_and_2016_l2032_203276


namespace NUMINAMATH_CALUDE_largest_angle_of_triangle_l2032_203258

/-- Given a triangle DEF with side lengths d, e, and f satisfying certain conditions,
    prove that its largest angle is 120°. -/
theorem largest_angle_of_triangle (d e f : ℝ) (h1 : d + 3*e + 3*f = d^2) (h2 : d + 3*e - 3*f = -4) :
  ∃ (A B C : ℝ), A + B + C = 180 ∧ A ≤ 120 ∧ B ≤ 120 ∧ max A (max B C) = 120 := by
  sorry

end NUMINAMATH_CALUDE_largest_angle_of_triangle_l2032_203258


namespace NUMINAMATH_CALUDE_randy_store_spending_l2032_203204

/-- Proves that Randy spends $2 per store trip -/
theorem randy_store_spending (initial_amount : ℕ) (final_amount : ℕ) (trips_per_month : ℕ) (months_per_year : ℕ) :
  initial_amount = 200 →
  final_amount = 104 →
  trips_per_month = 4 →
  months_per_year = 12 →
  (initial_amount - final_amount) / (trips_per_month * months_per_year) = 2 := by
  sorry

end NUMINAMATH_CALUDE_randy_store_spending_l2032_203204


namespace NUMINAMATH_CALUDE_square_inequality_negative_l2032_203244

theorem square_inequality_negative (a b : ℝ) (h1 : a < b) (h2 : b < 0) : a^2 > b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_inequality_negative_l2032_203244


namespace NUMINAMATH_CALUDE_f_min_at_neg_seven_l2032_203264

/-- The quadratic function f(x) = x^2 + 14x + 24 -/
def f (x : ℝ) : ℝ := x^2 + 14*x + 24

/-- Theorem: The function f(x) = x^2 + 14x + 24 attains its minimum value when x = -7 -/
theorem f_min_at_neg_seven :
  ∀ x : ℝ, f x ≥ f (-7) :=
by sorry

end NUMINAMATH_CALUDE_f_min_at_neg_seven_l2032_203264


namespace NUMINAMATH_CALUDE_shopkeeper_ornaments_profit_least_possible_n_l2032_203239

theorem shopkeeper_ornaments_profit (n d : ℕ) (h1 : d > 0) : 
  (3 * (d / (3 * n)) + (n - 3) * (d / n + 10) - d = 150) → n ≥ 18 :=
by
  sorry

theorem least_possible_n : 
  ∃ (n d : ℕ), d > 0 ∧ 3 * (d / (3 * n)) + (n - 3) * (d / n + 10) - d = 150 ∧ n = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_shopkeeper_ornaments_profit_least_possible_n_l2032_203239


namespace NUMINAMATH_CALUDE_equation_solution_expression_result_l2032_203295

-- Problem 1
theorem equation_solution :
  ∃ y : ℝ, 4 * (y - 1) = 1 - 3 * (y - 3) ∧ y = 2 := by sorry

-- Problem 2
theorem expression_result :
  (-2)^3 / 4 + 6 * |1/3 - 1| - 1/2 * 14 = -5 := by sorry

end NUMINAMATH_CALUDE_equation_solution_expression_result_l2032_203295


namespace NUMINAMATH_CALUDE_sum_of_angles_in_four_intersecting_lines_l2032_203235

-- Define the angles as real numbers
variable (p q r s : ℝ)

-- Define the property of four intersecting lines
def four_intersecting_lines (p q r s : ℝ) : Prop :=
  -- Add any additional properties that define four intersecting lines
  True

-- Theorem statement
theorem sum_of_angles_in_four_intersecting_lines 
  (h : four_intersecting_lines p q r s) : 
  p + q + r + s = 540 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_angles_in_four_intersecting_lines_l2032_203235


namespace NUMINAMATH_CALUDE_number_ratio_l2032_203248

theorem number_ratio (A B C : ℝ) : 
  A + B + C = 110 → A = 2 * B → B = 30 → C / A = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_number_ratio_l2032_203248


namespace NUMINAMATH_CALUDE_smallest_even_triangle_perimeter_l2032_203283

/-- A triangle with consecutive even integer side lengths -/
structure EvenTriangle where
  n : ℕ
  side1 : ℕ := 2 * n
  side2 : ℕ := 2 * n + 2
  side3 : ℕ := 2 * n + 4

/-- The triangle inequality for EvenTriangle -/
def satisfiesTriangleInequality (t : EvenTriangle) : Prop :=
  t.side1 + t.side2 > t.side3 ∧
  t.side1 + t.side3 > t.side2 ∧
  t.side2 + t.side3 > t.side1

/-- The perimeter of an EvenTriangle -/
def perimeter (t : EvenTriangle) : ℕ :=
  t.side1 + t.side2 + t.side3

/-- The theorem stating the smallest possible perimeter -/
theorem smallest_even_triangle_perimeter :
  ∀ t : EvenTriangle, satisfiesTriangleInequality t →
  ∃ t_min : EvenTriangle, satisfiesTriangleInequality t_min ∧
    perimeter t_min = 18 ∧
    ∀ t' : EvenTriangle, satisfiesTriangleInequality t' →
      perimeter t' ≥ perimeter t_min :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_even_triangle_perimeter_l2032_203283


namespace NUMINAMATH_CALUDE_fraction_less_than_decimal_l2032_203285

theorem fraction_less_than_decimal : (7 : ℚ) / 24 < (3 : ℚ) / 10 := by
  sorry

end NUMINAMATH_CALUDE_fraction_less_than_decimal_l2032_203285


namespace NUMINAMATH_CALUDE_sugar_profit_percentage_l2032_203297

theorem sugar_profit_percentage 
  (total_sugar : ℝ) 
  (sugar_at_18_percent : ℝ) 
  (overall_profit_percentage : ℝ) :
  total_sugar = 1000 →
  sugar_at_18_percent = 600 →
  overall_profit_percentage = 14 →
  ∃ (unknown_profit_percentage : ℝ),
    unknown_profit_percentage = 80 ∧
    sugar_at_18_percent * (18 / 100) + 
    (total_sugar - sugar_at_18_percent) * (unknown_profit_percentage / 100) = 
    total_sugar * (overall_profit_percentage / 100) :=
by
  sorry

end NUMINAMATH_CALUDE_sugar_profit_percentage_l2032_203297


namespace NUMINAMATH_CALUDE_probability_of_white_ball_l2032_203237

/-- The probability of drawing a white ball from a box containing white and black balls. -/
theorem probability_of_white_ball (white_balls black_balls : ℕ) 
  (h_white : white_balls = 5) (h_black : black_balls = 6) :
  (white_balls : ℚ) / (white_balls + black_balls) = 5 / 11 := by
  sorry

#check probability_of_white_ball

end NUMINAMATH_CALUDE_probability_of_white_ball_l2032_203237


namespace NUMINAMATH_CALUDE_arccos_zero_equals_pi_half_l2032_203254

theorem arccos_zero_equals_pi_half : Real.arccos 0 = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_arccos_zero_equals_pi_half_l2032_203254


namespace NUMINAMATH_CALUDE_sqrt_14_less_than_4_l2032_203200

theorem sqrt_14_less_than_4 : Real.sqrt 14 < 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_14_less_than_4_l2032_203200


namespace NUMINAMATH_CALUDE_equation_solution_implies_m_range_l2032_203271

theorem equation_solution_implies_m_range :
  ∀ m : ℝ,
  (∃ x : ℝ, 2^(2*x) + (m^2 - 2*m - 5)*2^x + 1 = 0) →
  m ∈ Set.Icc (-1 : ℝ) 3 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_implies_m_range_l2032_203271


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_property_l2032_203287

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_property : ∀ n, S n = (n : ℝ) * (a 1 + a n) / 2

/-- The main theorem -/
theorem arithmetic_sequence_sum_property
  (seq : ArithmeticSequence)
  (m : ℕ)
  (h_m_pos : m > 0)
  (h_sum_m : seq.S m = -2)
  (h_sum_m1 : seq.S (m + 1) = 0)
  (h_sum_m2 : seq.S (m + 2) = 3) :
  m = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_property_l2032_203287


namespace NUMINAMATH_CALUDE_song_ratio_after_deletion_l2032_203224

theorem song_ratio_after_deletion (total : ℕ) (deletion_percentage : ℚ) 
  (h1 : total = 720) 
  (h2 : deletion_percentage = 1/5) : 
  (total - (deletion_percentage * total).floor) / (deletion_percentage * total).floor = 4 := by
  sorry

end NUMINAMATH_CALUDE_song_ratio_after_deletion_l2032_203224


namespace NUMINAMATH_CALUDE_perpendicular_feet_circle_area_l2032_203263

/-- Given two points in the plane, calculate the area of the circle described by the perpendicular
    feet and find its floor. -/
theorem perpendicular_feet_circle_area (B C : ℝ × ℝ) (h_B : B = (20, 14)) (h_C : C = (18, 0)) :
  let midpoint := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
  let radius := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) / 2
  let area := π * radius^2
  area = 50 * π ∧ Int.floor area = 157 := by sorry

end NUMINAMATH_CALUDE_perpendicular_feet_circle_area_l2032_203263


namespace NUMINAMATH_CALUDE_digit_sum_puzzle_l2032_203273

def digit_set : Finset Nat := {0, 2, 3, 4, 5, 7, 8, 9}

theorem digit_sum_puzzle :
  ∃ (a b c d e f : Nat),
    a ∈ digit_set ∧ b ∈ digit_set ∧ c ∈ digit_set ∧
    d ∈ digit_set ∧ e ∈ digit_set ∧ f ∈ digit_set ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
    d ≠ e ∧ d ≠ f ∧
    e ≠ f ∧
    a + b + c = 24 ∧
    b + d + e + f = 14 ∧
    a + b + c + d + e + f = 31 :=
by sorry

end NUMINAMATH_CALUDE_digit_sum_puzzle_l2032_203273


namespace NUMINAMATH_CALUDE_cafeteria_problem_l2032_203286

theorem cafeteria_problem (n : ℕ) (h : n = 6) :
  (∃ (max_days : ℕ) (avg_dishes : ℚ),
    max_days = 2^n ∧
    avg_dishes = n / 2 ∧
    max_days = 64 ∧
    avg_dishes = 3) := by
  sorry

end NUMINAMATH_CALUDE_cafeteria_problem_l2032_203286


namespace NUMINAMATH_CALUDE_cos_negative_300_degrees_l2032_203210

theorem cos_negative_300_degrees : Real.cos (-(300 * π / 180)) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_negative_300_degrees_l2032_203210


namespace NUMINAMATH_CALUDE_inequality_solution_l2032_203255

theorem inequality_solution (x : ℝ) : 
  (x^2 - 4*x - 5) / (x^2 + 3*x + 2) < 0 ↔ x ∈ Set.Ioo (-2 : ℝ) (-1) ∪ Set.Ioo (-1 : ℝ) 5 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2032_203255


namespace NUMINAMATH_CALUDE_volume_of_T_l2032_203228

/-- The solid T in ℝ³ -/
def T : Set (ℝ × ℝ × ℝ) :=
  {p : ℝ × ℝ × ℝ | let (x, y, z) := p
                   (|x| + |y| ≤ 2) ∧ (|x| + |z| ≤ 2) ∧ (|y| + |z| ≤ 2)}

/-- The volume of a set in ℝ³ -/
noncomputable def volume (S : Set (ℝ × ℝ × ℝ)) : ℝ := sorry

/-- The theorem stating the volume of T -/
theorem volume_of_T : volume T = 32 * Real.sqrt 3 / 9 := by sorry

end NUMINAMATH_CALUDE_volume_of_T_l2032_203228


namespace NUMINAMATH_CALUDE_min_sum_with_geometric_mean_l2032_203259

theorem min_sum_with_geometric_mean (a b : ℝ) : 
  a > 0 → b > 0 → (Real.sqrt (3^a * 3^b) = Real.sqrt (3^a * 3^b)) → 
  ∃ (min : ℝ), min = 4 ∧ ∀ (x y : ℝ), x > 0 → y > 0 → 
  Real.sqrt (3^x * 3^y) = Real.sqrt (3^x * 3^y) → x + y ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_sum_with_geometric_mean_l2032_203259


namespace NUMINAMATH_CALUDE_intersection_M_N_l2032_203249

def M : Set ℤ := {m | -3 < m ∧ m < 2}
def N : Set ℤ := {n | -1 ≤ n ∧ n ≤ 3}

theorem intersection_M_N : M ∩ N = {-1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2032_203249
