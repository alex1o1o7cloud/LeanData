import Mathlib

namespace NUMINAMATH_CALUDE_infinitely_many_non_representable_l182_18246

def sum_of_powers (x₁ x₂ x₃ x₄ x₅ : ℕ) : ℕ :=
  x₁^3 + x₂^5 + x₃^7 + x₄^9 + x₅^11

def representable (n : ℕ) : Prop :=
  ∃ x₁ x₂ x₃ x₄ x₅ : ℕ, sum_of_powers x₁ x₂ x₃ x₄ x₅ = n

theorem infinitely_many_non_representable :
  ∃ S : Set ℕ, (Set.Infinite S) ∧ (∀ n ∈ S, ¬representable n) := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_non_representable_l182_18246


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l182_18219

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x * y = 4) :
  (1 / x + 1 / y) ≥ 1 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ x * y = 4 ∧ 1 / x + 1 / y = 1 :=
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l182_18219


namespace NUMINAMATH_CALUDE_society_member_property_l182_18277

theorem society_member_property (n : ℕ) (h : n = 1978) :
  ∀ (f : Fin n → Fin 6),
  ∃ (i j k : Fin n),
    f i = f j ∧ f i = f k ∧
    (i.val = j.val + k.val ∨ i.val = 2 * j.val) :=
by sorry

end NUMINAMATH_CALUDE_society_member_property_l182_18277


namespace NUMINAMATH_CALUDE_sum_integers_between_two_and_eleven_l182_18289

theorem sum_integers_between_two_and_eleven : 
  (Finset.range 8).sum (fun i => i + 3) = 52 := by
  sorry

end NUMINAMATH_CALUDE_sum_integers_between_two_and_eleven_l182_18289


namespace NUMINAMATH_CALUDE_dictionary_cost_l182_18266

def total_cost : ℕ := 8 + 29
def dinosaur_book_cost : ℕ := 19
def cookbook_cost : ℕ := 7
def savings : ℕ := 8
def additional_needed : ℕ := 29

theorem dictionary_cost : 
  total_cost - (dinosaur_book_cost + cookbook_cost) = 11 := by
sorry

end NUMINAMATH_CALUDE_dictionary_cost_l182_18266


namespace NUMINAMATH_CALUDE_coat_price_proof_l182_18276

theorem coat_price_proof (price : ℝ) : 
  (price - 250 = price * 0.5) → price = 500 := by
  sorry

end NUMINAMATH_CALUDE_coat_price_proof_l182_18276


namespace NUMINAMATH_CALUDE_right_triangle_third_side_product_l182_18221

theorem right_triangle_third_side_product (a b : ℝ) (ha : a = 6) (hb : b = 8) :
  let c := Real.sqrt (a^2 + b^2)
  let d := Real.sqrt (max a b ^ 2 - min a b ^ 2)
  c * d = 20 * Real.sqrt 7 := by sorry

end NUMINAMATH_CALUDE_right_triangle_third_side_product_l182_18221


namespace NUMINAMATH_CALUDE_class_size_problem_l182_18203

theorem class_size_problem (x y : ℕ) : 
  y = x / 6 →  -- Initial condition: absent = 1/6 of present
  y = (x - 1) / 5 →  -- Condition after one student leaves
  x + y = 7  -- Total number of students
  := by sorry

end NUMINAMATH_CALUDE_class_size_problem_l182_18203


namespace NUMINAMATH_CALUDE_largest_non_representable_l182_18291

/-- Coin denominations in Limonia -/
def coin_denominations (n : ℕ) : List ℕ :=
  List.range (n + 1) |> List.map (λ i => 2^(n - i) * 3^i)

/-- A number is representable if it can be expressed as a sum of coin denominations -/
def is_representable (s : ℕ) (n : ℕ) : Prop :=
  ∃ (coeffs : List ℕ), s = List.sum (List.zipWith (·*·) coeffs (coin_denominations n))

/-- The largest non-representable amount in Limonia's currency system -/
theorem largest_non_representable (n : ℕ) :
  ¬ is_representable (3^(n+1) - 2^(n+2)) n ∧
  ∀ s, s > 3^(n+1) - 2^(n+2) → is_representable s n :=
sorry

end NUMINAMATH_CALUDE_largest_non_representable_l182_18291


namespace NUMINAMATH_CALUDE_strip_coloring_problem_l182_18250

/-- The number of valid colorings for a strip of length n -/
def validColorings : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | n + 2 => validColorings (n + 1) + validColorings n

/-- The problem statement -/
theorem strip_coloring_problem :
  validColorings 9 = 89 := by sorry

end NUMINAMATH_CALUDE_strip_coloring_problem_l182_18250


namespace NUMINAMATH_CALUDE_grid_equal_sums_l182_18214

/-- Given a, b, c, prove that there exist x, y, z, t, u, v such that all rows, columns, and diagonals in a 3x3 grid sum to the same value -/
theorem grid_equal_sums (a b c : ℚ) : ∃ (x y z t u v : ℚ),
  (x + a + b = x + y + c) ∧
  (y + z + t = b + z + c) ∧
  (u + t + v = a + t + c) ∧
  (x + y + c = y + z + t) ∧
  (x + a + b = a + z + v) ∧
  (x + y + c = u + t + v) ∧
  (x + a + b = b + z + c) :=
by sorry

end NUMINAMATH_CALUDE_grid_equal_sums_l182_18214


namespace NUMINAMATH_CALUDE_fraction_of_students_with_As_l182_18288

theorem fraction_of_students_with_As (fraction_B : ℝ) (fraction_A_or_B : ℝ) 
  (h1 : fraction_B = 0.2) 
  (h2 : fraction_A_or_B = 0.9) : 
  ∃ fraction_A : ℝ, fraction_A + fraction_B = fraction_A_or_B ∧ fraction_A = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_students_with_As_l182_18288


namespace NUMINAMATH_CALUDE_line_intercepts_and_point_l182_18259

/-- Given a line 3x + 5y + c = 0 where the sum of x- and y-intercepts is 16,
    prove that c = -30 and the point (2, 24/5) lies on the line. -/
theorem line_intercepts_and_point (c : ℝ) : 
  (∃ (x y : ℝ), 3*x + 5*y + c = 0 ∧ x + y = 16) → 
  (c = -30 ∧ 3*2 + 5*(24/5) + c = 0) :=
by sorry

end NUMINAMATH_CALUDE_line_intercepts_and_point_l182_18259


namespace NUMINAMATH_CALUDE_prob_not_adjacent_seven_chairs_l182_18220

/-- The number of chairs in the row -/
def n : ℕ := 7

/-- The number of ways two people can sit next to each other in a row of n chairs -/
def adjacent_seating (n : ℕ) : ℕ := n - 1

/-- The total number of ways two people can choose seats from n chairs -/
def total_seating (n : ℕ) : ℕ := n.choose 2

/-- The probability that Mary and James don't sit next to each other
    when randomly choosing seats in a row of n chairs -/
def prob_not_adjacent (n : ℕ) : ℚ :=
  1 - (adjacent_seating n : ℚ) / (total_seating n : ℚ)

theorem prob_not_adjacent_seven_chairs :
  prob_not_adjacent n = 5/7 := by sorry

end NUMINAMATH_CALUDE_prob_not_adjacent_seven_chairs_l182_18220


namespace NUMINAMATH_CALUDE_total_trees_equals_sum_our_park_total_is_55_l182_18228

/-- Represents the number of walnut trees in a park -/
structure WalnutPark where
  initial : Nat  -- Initial number of walnut trees
  planted : Nat  -- Number of walnut trees planted

/-- Calculates the total number of walnut trees after planting -/
def total_trees (park : WalnutPark) : Nat :=
  park.initial + park.planted

/-- Theorem: The total number of walnut trees after planting is the sum of initial and planted trees -/
theorem total_trees_equals_sum (park : WalnutPark) : 
  total_trees park = park.initial + park.planted := by
  sorry

/-- The specific park instance from the problem -/
def our_park : WalnutPark := { initial := 22, planted := 33 }

/-- Theorem: The total number of walnut trees in our park after planting is 55 -/
theorem our_park_total_is_55 : total_trees our_park = 55 := by
  sorry

end NUMINAMATH_CALUDE_total_trees_equals_sum_our_park_total_is_55_l182_18228


namespace NUMINAMATH_CALUDE_square_overlap_area_difference_l182_18254

theorem square_overlap_area_difference :
  ∀ (side_large side_small overlap_area : ℝ),
    side_large = 10 →
    side_small = 7 →
    overlap_area = 9 →
    side_large > 0 →
    side_small > 0 →
    overlap_area > 0 →
    (side_large^2 - overlap_area) - (side_small^2 - overlap_area) = 51 :=
by
  sorry

end NUMINAMATH_CALUDE_square_overlap_area_difference_l182_18254


namespace NUMINAMATH_CALUDE_egg_price_decrease_impact_l182_18279

-- Define the types
structure CakeShop where
  eggDemand : ℝ
  productionScale : ℝ
  cakeOutput : ℝ
  marketSupply : ℝ

-- Define the egg price change
def eggPriceDecrease : ℝ := 0.05

-- Define the impact of egg price decrease on cake shops
def impactOnCakeShop (shop : CakeShop) (priceDecrease : ℝ) : CakeShop :=
  { eggDemand := shop.eggDemand * (1 + priceDecrease),
    productionScale := shop.productionScale * (1 + priceDecrease),
    cakeOutput := shop.cakeOutput * (1 + priceDecrease),
    marketSupply := shop.marketSupply * (1 + priceDecrease) }

-- Theorem statement
theorem egg_price_decrease_impact (shop : CakeShop) :
  let newShop := impactOnCakeShop shop eggPriceDecrease
  newShop.eggDemand > shop.eggDemand ∧
  newShop.productionScale > shop.productionScale ∧
  newShop.cakeOutput > shop.cakeOutput ∧
  newShop.marketSupply > shop.marketSupply :=
by sorry

end NUMINAMATH_CALUDE_egg_price_decrease_impact_l182_18279


namespace NUMINAMATH_CALUDE_max_intersection_points_quadrilateral_circle_l182_18249

/-- A circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A quadrilateral in a plane -/
structure Quadrilateral where
  vertices : Fin 4 → ℝ × ℝ

/-- The number of intersection points between a line segment and a circle -/
def intersectionPointsLineSegmentCircle (segment : (ℝ × ℝ) × (ℝ × ℝ)) (circle : Circle) : ℕ :=
  sorry

/-- The number of intersection points between a quadrilateral and a circle -/
def intersectionPointsQuadrilateralCircle (quad : Quadrilateral) (circle : Circle) : ℕ :=
  sorry

/-- Theorem: The maximum number of intersection points between a quadrilateral and a circle is 8 -/
theorem max_intersection_points_quadrilateral_circle :
  ∀ (quad : Quadrilateral) (circle : Circle),
    intersectionPointsQuadrilateralCircle quad circle ≤ 8 ∧
    ∃ (quad' : Quadrilateral) (circle' : Circle),
      intersectionPointsQuadrilateralCircle quad' circle' = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_max_intersection_points_quadrilateral_circle_l182_18249


namespace NUMINAMATH_CALUDE_diophantine_logarithm_equation_l182_18293

theorem diophantine_logarithm_equation : ∃ (X Y Z : ℕ+), 
  (Nat.gcd X.val (Nat.gcd Y.val Z.val) = 1) ∧ 
  (X.val : ℝ) * (Real.log 5 / Real.log 100) + (Y.val : ℝ) * (Real.log 4 / Real.log 100) = Z.val ∧
  X.val + Y.val + Z.val = 4 := by
  sorry

end NUMINAMATH_CALUDE_diophantine_logarithm_equation_l182_18293


namespace NUMINAMATH_CALUDE_definite_integral_2x_minus_1_l182_18295

theorem definite_integral_2x_minus_1 :
  ∫ x in (0:ℝ)..3, (2*x - 1) = 6 := by sorry

end NUMINAMATH_CALUDE_definite_integral_2x_minus_1_l182_18295


namespace NUMINAMATH_CALUDE_inequality_theorem_equality_condition_l182_18286

theorem inequality_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  1 / (2 * a) + 1 / (2 * b) + 1 / (2 * c) ≥ 1 / (b + c) + 1 / (c + a) + 1 / (a + b) :=
by sorry

theorem equality_condition (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (1 / (2 * a) + 1 / (2 * b) + 1 / (2 * c) = 1 / (b + c) + 1 / (c + a) + 1 / (a + b)) ↔ 
  (a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_inequality_theorem_equality_condition_l182_18286


namespace NUMINAMATH_CALUDE_one_rupee_coins_count_l182_18271

/-- Represents the types of coins in the bag -/
inductive CoinType
  | OneRupee
  | FiftyPaise
  | TwentyFivePaise

/-- The value of a coin in paise -/
def coinValue (c : CoinType) : ℕ :=
  match c with
  | CoinType.OneRupee => 100
  | CoinType.FiftyPaise => 50
  | CoinType.TwentyFivePaise => 25

/-- The total value of all coins in the bag in paise -/
def totalValue : ℕ := 105 * 100

/-- The number of each type of coin in the bag -/
def numEachCoin : ℕ := 60

/-- The total number of coins in the bag -/
def totalCoins : ℕ := 3 * numEachCoin

theorem one_rupee_coins_count :
  ∃ (n : ℕ), n = numEachCoin ∧
    n * coinValue CoinType.OneRupee +
    n * coinValue CoinType.FiftyPaise +
    n * coinValue CoinType.TwentyFivePaise = totalValue ∧
    3 * n = totalCoins := by
  sorry

end NUMINAMATH_CALUDE_one_rupee_coins_count_l182_18271


namespace NUMINAMATH_CALUDE_max_x_value_l182_18207

theorem max_x_value (x : ℝ) : 
  ((5*x - 20)/(4*x - 5))^2 + (5*x - 20)/(4*x - 5) = 20 → x ≤ 9/5 :=
by sorry

end NUMINAMATH_CALUDE_max_x_value_l182_18207


namespace NUMINAMATH_CALUDE_predict_sales_at_34_degrees_l182_18274

/-- Represents the linear regression model for cold drink sales -/
structure ColdDrinkSalesModel where
  /-- Calculates the predicted sales volume based on temperature -/
  predict : ℝ → ℝ

/-- Theorem: Given the linear regression model ŷ = 2x + 60, 
    the predicted sales volume for a day with highest temperature 34°C is 128 cups -/
theorem predict_sales_at_34_degrees 
  (model : ColdDrinkSalesModel)
  (h_model : ∀ x, model.predict x = 2 * x + 60) :
  model.predict 34 = 128 := by
  sorry

end NUMINAMATH_CALUDE_predict_sales_at_34_degrees_l182_18274


namespace NUMINAMATH_CALUDE_tangent_lines_range_l182_18225

/-- The function f(x) = x^3 - 3x --/
def f (x : ℝ) : ℝ := x^3 - 3*x

/-- The derivative of f(x) --/
def f' (x : ℝ) : ℝ := 3*x^2 - 3

/-- The function g(x) = 2x^3 - 6x^2 --/
def g (x : ℝ) : ℝ := 2*x^3 - 6*x^2

/-- The derivative of g(x) --/
def g' (x : ℝ) : ℝ := 6*x^2 - 12*x

/-- Theorem: If three distinct tangent lines to f(x) pass through A(2, m), then -6 < m < 2 --/
theorem tangent_lines_range (m : ℝ) : 
  (∃ a b c : ℝ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
    (m - (f a)) = (f' a) * (2 - a) ∧
    (m - (f b)) = (f' b) * (2 - b) ∧
    (m - (f c)) = (f' c) * (2 - c)) →
  -6 < m ∧ m < 2 :=
by sorry

end NUMINAMATH_CALUDE_tangent_lines_range_l182_18225


namespace NUMINAMATH_CALUDE_almonds_problem_l182_18232

theorem almonds_problem (lily_almonds jack_almonds : ℕ) : 
  lily_almonds = jack_almonds + 8 →
  jack_almonds = lily_almonds / 3 →
  lily_almonds = 12 := by
sorry

end NUMINAMATH_CALUDE_almonds_problem_l182_18232


namespace NUMINAMATH_CALUDE_polynomial_no_real_roots_l182_18213

theorem polynomial_no_real_roots :
  ∀ x : ℝ, 4 * x^8 - 2 * x^7 + x^6 - 3 * x^4 + x^2 - x + 1 > 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_no_real_roots_l182_18213


namespace NUMINAMATH_CALUDE_equation_solutions_l182_18287

theorem equation_solutions :
  (∃ x : ℚ, 3 * x - 1 = x + 7 ∧ x = 4) ∧
  (∃ x : ℚ, (x + 1) / 2 - 1 = (1 - 2 * x) / 3 ∧ x = 5 / 7) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l182_18287


namespace NUMINAMATH_CALUDE_package_volume_calculation_l182_18234

/-- Calculates the total volume needed to package a collection given box dimensions and cost constraints. -/
theorem package_volume_calculation 
  (box_length : ℝ) 
  (box_width : ℝ) 
  (box_height : ℝ) 
  (box_cost : ℝ) 
  (total_cost : ℝ) 
  (h1 : box_length = 20) 
  (h2 : box_width = 20) 
  (h3 : box_height = 15) 
  (h4 : box_cost = 0.9) 
  (h5 : total_cost = 459) :
  (total_cost / box_cost) * (box_length * box_width * box_height) = 3060000 := by
  sorry

end NUMINAMATH_CALUDE_package_volume_calculation_l182_18234


namespace NUMINAMATH_CALUDE_siena_bookmarks_l182_18241

/-- The number of bookmarked pages Siena will have at the end of March -/
def bookmarks_end_of_march (daily_rate : ℕ) (current_bookmarks : ℕ) : ℕ :=
  current_bookmarks + daily_rate * 31

/-- Theorem stating that Siena will have 1330 bookmarked pages at the end of March -/
theorem siena_bookmarks :
  bookmarks_end_of_march 30 400 = 1330 := by
  sorry

end NUMINAMATH_CALUDE_siena_bookmarks_l182_18241


namespace NUMINAMATH_CALUDE_cone_height_ratio_l182_18285

/-- Proves the ratio of heights for a cone with reduced height --/
theorem cone_height_ratio (base_circumference : ℝ) (original_height : ℝ) (shorter_volume : ℝ) :
  base_circumference = 18 * Real.pi →
  original_height = 36 →
  shorter_volume = 270 * Real.pi →
  ∃ (shorter_height : ℝ),
    shorter_height / original_height = 5 / 18 ∧
    shorter_volume = (1 / 3) * Real.pi * (base_circumference / (2 * Real.pi))^2 * shorter_height :=
by sorry

end NUMINAMATH_CALUDE_cone_height_ratio_l182_18285


namespace NUMINAMATH_CALUDE_work_completion_original_men_l182_18261

theorem work_completion_original_men (initial_days : ℕ) (absent_men : ℕ) (final_days : ℕ) : 
  initial_days = 55 → absent_men = 15 → final_days = 60 → 
  ∃ (original_men : ℕ), 
    original_men * initial_days = (original_men - absent_men) * final_days ∧
    original_men = 180 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_original_men_l182_18261


namespace NUMINAMATH_CALUDE_F_2_f_3_equals_341_l182_18204

-- Define the functions f and F
def f (a : ℝ) : ℝ := a^2 - 2
def F (a b : ℝ) : ℝ := b^3 - a

-- State the theorem
theorem F_2_f_3_equals_341 : F 2 (f 3) = 341 := by sorry

end NUMINAMATH_CALUDE_F_2_f_3_equals_341_l182_18204


namespace NUMINAMATH_CALUDE_residue_calculation_l182_18209

theorem residue_calculation (m : ℕ) (h : m = 17) : 
  (220 * 18 - 28 * 5 + 4) % m = 12 := by
  sorry

end NUMINAMATH_CALUDE_residue_calculation_l182_18209


namespace NUMINAMATH_CALUDE_unique_number_equality_l182_18243

theorem unique_number_equality : ∃! x : ℝ, (x / 2) + 6 = 2 * x - 6 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_equality_l182_18243


namespace NUMINAMATH_CALUDE_sweater_shirt_price_difference_l182_18245

theorem sweater_shirt_price_difference : 
  let shirt_total : ℚ := 400
  let shirt_count : ℕ := 25
  let sweater_total : ℚ := 1500
  let sweater_count : ℕ := 75
  let shirt_avg : ℚ := shirt_total / shirt_count
  let sweater_avg : ℚ := sweater_total / sweater_count
  sweater_avg - shirt_avg = 4 := by
sorry

end NUMINAMATH_CALUDE_sweater_shirt_price_difference_l182_18245


namespace NUMINAMATH_CALUDE_function_inequality_l182_18206

open Real

theorem function_inequality (a x : ℝ) (ha : a ≥ 1) (hx : x > 0) :
  a * exp x + 2 * x - 1 ≥ (x + a * exp 1) * x := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l182_18206


namespace NUMINAMATH_CALUDE_correct_change_l182_18212

/-- The change Sandy received after buying a football and a baseball -/
def sandys_change (football_cost baseball_cost payment : ℚ) : ℚ :=
  payment - (football_cost + baseball_cost)

theorem correct_change : sandys_change 9.14 6.81 20 = 4.05 := by
  sorry

end NUMINAMATH_CALUDE_correct_change_l182_18212


namespace NUMINAMATH_CALUDE_gcf_of_120_180_300_l182_18260

theorem gcf_of_120_180_300 : Nat.gcd 120 (Nat.gcd 180 300) = 60 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_120_180_300_l182_18260


namespace NUMINAMATH_CALUDE_equation_solution_l182_18253

theorem equation_solution : ∃ x : ℝ, (10 - x)^2 = (x - 2)^2 + 8 ∧ x = 5.5 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l182_18253


namespace NUMINAMATH_CALUDE_polygon_sides_from_angle_sum_l182_18229

theorem polygon_sides_from_angle_sum (n : ℕ) (angle_sum : ℝ) : 
  angle_sum = 720 → (n - 2) * 180 = angle_sum → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_from_angle_sum_l182_18229


namespace NUMINAMATH_CALUDE_part_one_disproof_part_two_proof_l182_18210

-- Part 1
theorem part_one_disproof (x y z : ℝ) (h_pos : x > 0 ∧ y > 0 ∧ z > 0) (h_sum : x + y + z ≥ 3) :
  ¬ (∀ x y z : ℝ, x > 0 → y > 0 → z > 0 → x + y + z ≥ 3 → 1/x + 1/y + 1/z ≤ 3) :=
sorry

-- Part 2
theorem part_two_proof (x y z : ℝ) (h_pos : x > 0 ∧ y > 0 ∧ z > 0) (h_sum : x + y + z ≤ 3) :
  1/x + 1/y + 1/z ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_part_one_disproof_part_two_proof_l182_18210


namespace NUMINAMATH_CALUDE_last_locker_opened_l182_18278

/-- Represents the state of a locker (open or closed) -/
inductive LockerState
| Open
| Closed

/-- Represents the locker-opening process -/
def lockerProcess (n : Nat) : Nat → Nat :=
  sorry

/-- The number of lockers in the hall -/
def totalLockers : Nat := 512

/-- Theorem stating that the last locker opened is number 509 -/
theorem last_locker_opened :
  ∃ (process : Nat → Nat → LockerState),
    (∀ k, k ≤ totalLockers → process totalLockers k = LockerState.Open) ∧
    (∀ k, k < 509 → ∃ m, m < 509 ∧ process totalLockers m = LockerState.Open ∧ m > k) ∧
    process totalLockers 509 = LockerState.Open :=
  sorry

end NUMINAMATH_CALUDE_last_locker_opened_l182_18278


namespace NUMINAMATH_CALUDE_chris_money_before_birthday_l182_18231

/-- The amount of money Chris had before his birthday -/
def money_before_birthday : ℕ := sorry

/-- The amount Chris received from his grandmother -/
def grandmother_gift : ℕ := 25

/-- The amount Chris received from his aunt and uncle -/
def aunt_uncle_gift : ℕ := 20

/-- The amount Chris received from his parents -/
def parents_gift : ℕ := 75

/-- The total amount Chris has now -/
def total_money_now : ℕ := 279

/-- Theorem stating that Chris had $159 before his birthday -/
theorem chris_money_before_birthday :
  money_before_birthday = 159 :=
by sorry

end NUMINAMATH_CALUDE_chris_money_before_birthday_l182_18231


namespace NUMINAMATH_CALUDE_correct_calculation_l182_18235

theorem correct_calculation (x : ℝ) (h : (x * 5) + 7 = 27) : (x + 5) * 7 = 63 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l182_18235


namespace NUMINAMATH_CALUDE_geometric_sequence_minimum_value_l182_18268

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ (q : ℝ), q > 0 ∧ ∀ n, a (n + 1) = q * a n

theorem geometric_sequence_minimum_value
  (a : ℕ → ℝ)
  (h_pos : ∀ n, a n > 0)
  (h_geo : GeometricSequence a)
  (h_relation : a 7 = a 6 + 2 * a 5)
  (h_product : ∃ (m n : ℕ), m ≠ n ∧ a m * a n = 16 * (a 1)^2) :
  (∃ (m n : ℕ), m ≠ n ∧ 1 / m + 4 / n = 3 / 2) ∧
  (∀ (m n : ℕ), m ≠ n → 1 / m + 4 / n ≥ 3 / 2) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_minimum_value_l182_18268


namespace NUMINAMATH_CALUDE_union_of_sets_l182_18298

theorem union_of_sets : 
  let A : Set ℕ := {1, 2, 3}
  let B : Set ℕ := {2, 4, 5}
  A ∪ B = {1, 2, 3, 4, 5} := by
sorry

end NUMINAMATH_CALUDE_union_of_sets_l182_18298


namespace NUMINAMATH_CALUDE_dan_seashells_given_l182_18255

/-- The number of seashells Dan gave to Jessica -/
def seashells_given (initial : ℕ) (left : ℕ) : ℕ :=
  initial - left

theorem dan_seashells_given :
  seashells_given 56 22 = 34 := by
  sorry

end NUMINAMATH_CALUDE_dan_seashells_given_l182_18255


namespace NUMINAMATH_CALUDE_x_times_one_minus_f_eq_one_l182_18252

noncomputable def x : ℝ := (3 + Real.sqrt 8) ^ 1000
noncomputable def n : ℤ := ⌊x⌋
noncomputable def f : ℝ := x - n

theorem x_times_one_minus_f_eq_one : x * (1 - f) = 1 := by sorry

end NUMINAMATH_CALUDE_x_times_one_minus_f_eq_one_l182_18252


namespace NUMINAMATH_CALUDE_town_population_problem_l182_18215

theorem town_population_problem (original_population : ℕ) : 
  (((original_population + 2000) * 85 / 100) : ℕ) = original_population - 50 →
  original_population = 11667 :=
by
  sorry

end NUMINAMATH_CALUDE_town_population_problem_l182_18215


namespace NUMINAMATH_CALUDE_big_bottle_volume_proof_l182_18262

/-- The volume of a big bottle of mango juice in ounces -/
def big_bottle_volume : ℝ := 30

/-- The cost of a big bottle in pesetas -/
def big_bottle_cost : ℝ := 2700

/-- The volume of a small bottle in ounces -/
def small_bottle_volume : ℝ := 6

/-- The cost of a small bottle in pesetas -/
def small_bottle_cost : ℝ := 600

/-- The amount saved by buying a big bottle instead of equivalent small bottles in pesetas -/
def savings : ℝ := 300

theorem big_bottle_volume_proof :
  big_bottle_volume = 30 :=
by sorry

end NUMINAMATH_CALUDE_big_bottle_volume_proof_l182_18262


namespace NUMINAMATH_CALUDE_ferry_journey_difference_l182_18239

/-- Represents a ferry with its speed and travel time -/
structure Ferry where
  speed : ℝ
  time : ℝ

/-- Calculates the distance traveled by a ferry -/
def distance (f : Ferry) : ℝ := f.speed * f.time

theorem ferry_journey_difference :
  let ferry_p : Ferry := { speed := 8, time := 3 }
  let ferry_q : Ferry := { speed := ferry_p.speed + 1, time := (3 * distance ferry_p) / (ferry_p.speed + 1) }
  ferry_q.time - ferry_p.time = 5 := by
  sorry

end NUMINAMATH_CALUDE_ferry_journey_difference_l182_18239


namespace NUMINAMATH_CALUDE_history_paper_pages_l182_18236

/-- The number of days Stacy has to complete her history paper -/
def days_to_complete : ℕ := 3

/-- The number of pages Stacy must write per day to finish on time -/
def pages_per_day : ℕ := 11

/-- The total number of pages in Stacy's history paper -/
def total_pages : ℕ := days_to_complete * pages_per_day

theorem history_paper_pages : total_pages = 33 := by
  sorry

end NUMINAMATH_CALUDE_history_paper_pages_l182_18236


namespace NUMINAMATH_CALUDE_figures_per_shelf_is_eleven_l182_18273

/-- The number of shelves in Adam's room -/
def num_shelves : ℕ := 4

/-- The total number of action figures that can fit on all shelves -/
def total_figures : ℕ := 44

/-- The number of action figures that can fit on each shelf -/
def figures_per_shelf : ℕ := total_figures / num_shelves

/-- Theorem: The number of action figures that can fit on each shelf is 11 -/
theorem figures_per_shelf_is_eleven : figures_per_shelf = 11 := by
  sorry

end NUMINAMATH_CALUDE_figures_per_shelf_is_eleven_l182_18273


namespace NUMINAMATH_CALUDE_largest_divisor_of_n_l182_18247

theorem largest_divisor_of_n (n : ℕ) (h1 : n > 0) (h2 : 36 ∣ n^3) : 
  ∀ d : ℕ, d ∣ n → d ≤ 6 :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_n_l182_18247


namespace NUMINAMATH_CALUDE_adams_collection_worth_80_dollars_l182_18284

/-- The value of Adam's coin collection -/
def adams_collection_value (total_coins : ℕ) (sample_coins : ℕ) (sample_value : ℕ) : ℕ :=
  total_coins * (sample_value / sample_coins)

/-- Theorem: Adam's coin collection is worth 80 dollars -/
theorem adams_collection_worth_80_dollars :
  adams_collection_value 20 5 20 = 80 :=
by sorry

end NUMINAMATH_CALUDE_adams_collection_worth_80_dollars_l182_18284


namespace NUMINAMATH_CALUDE_mod_equivalence_problem_l182_18292

theorem mod_equivalence_problem : ∃! m : ℤ, 0 ≤ m ∧ m ≤ 8 ∧ m ≡ 500000 [ZMOD 9] ∧ m = 5 := by
  sorry

end NUMINAMATH_CALUDE_mod_equivalence_problem_l182_18292


namespace NUMINAMATH_CALUDE_green_shirt_pairs_l182_18238

theorem green_shirt_pairs (total_students : ℕ) (red_students : ℕ) (green_students : ℕ) 
  (total_pairs : ℕ) (red_red_pairs : ℕ) :
  total_students = 150 →
  red_students = 60 →
  green_students = 90 →
  total_pairs = 75 →
  red_red_pairs = 28 →
  total_students = red_students + green_students →
  ∃ (green_green_pairs : ℕ), green_green_pairs = 43 ∧ 
    green_green_pairs + red_red_pairs + (total_students - 2 * red_red_pairs - 2 * green_green_pairs) / 2 = total_pairs :=
by sorry

end NUMINAMATH_CALUDE_green_shirt_pairs_l182_18238


namespace NUMINAMATH_CALUDE_chosen_number_l182_18272

theorem chosen_number (x : ℝ) : (x / 8) - 160 = 12 → x = 1376 := by
  sorry

end NUMINAMATH_CALUDE_chosen_number_l182_18272


namespace NUMINAMATH_CALUDE_product_of_divisors_equal_implies_equal_l182_18226

/-- Product of divisors function -/
def p (x : ℤ) : ℤ := sorry

/-- Theorem: If the product of divisors of two integers are equal, then the integers are equal -/
theorem product_of_divisors_equal_implies_equal (m n : ℤ) : p m = p n → m = n := by
  sorry

end NUMINAMATH_CALUDE_product_of_divisors_equal_implies_equal_l182_18226


namespace NUMINAMATH_CALUDE_smallest_bounded_area_l182_18294

/-- The area of the smallest region bounded by y = x^2 and x^2 + y^2 = 9 -/
theorem smallest_bounded_area : 
  let f (x : ℝ) := x^2
  let g (x y : ℝ) := x^2 + y^2 = 9
  let intersection_x := Real.sqrt ((Real.sqrt 37 - 1) / 2)
  let bounded_area := (2/3) * (((-1 + Real.sqrt 37) / 2)^(3/2))
  ∃ (area : ℝ), area = bounded_area ∧ 
    (∀ x y, -intersection_x ≤ x ∧ x ≤ intersection_x ∧ 
            y = f x ∧ g x y → 
            area = ∫ x in -intersection_x..intersection_x, f x) :=
by sorry


end NUMINAMATH_CALUDE_smallest_bounded_area_l182_18294


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l182_18227

/-- Given a quadratic equation x^2 - 3x + k - 2 = 0 with two real roots x1 and x2 -/
theorem quadratic_equation_properties (k : ℝ) (x1 x2 : ℝ) 
  (h1 : x1^2 - 3*x1 + k - 2 = 0)
  (h2 : x2^2 - 3*x2 + k - 2 = 0)
  (h3 : x1 ≠ x2) :
  (k ≤ 17/4) ∧ 
  (x1 + x2 - x1*x2 = 1 → k = 4) := by
  sorry


end NUMINAMATH_CALUDE_quadratic_equation_properties_l182_18227


namespace NUMINAMATH_CALUDE_magic_king_episodes_l182_18240

/-- The total number of episodes in the Magic King show -/
def total_episodes : ℕ :=
  let first_three_seasons := 3 * 20
  let seasons_four_to_eight := 5 * 25
  let seasons_nine_to_eleven := 3 * 30
  let last_three_seasons := 3 * 15
  let holiday_specials := 5
  first_three_seasons + seasons_four_to_eight + seasons_nine_to_eleven + last_three_seasons + holiday_specials

/-- Theorem stating that the total number of episodes is 325 -/
theorem magic_king_episodes : total_episodes = 325 := by
  sorry

end NUMINAMATH_CALUDE_magic_king_episodes_l182_18240


namespace NUMINAMATH_CALUDE_rectangular_solid_surface_area_l182_18217

/-- A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself. -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- The surface area of a rectangular solid with dimensions a, b, and c. -/
def surface_area (a b c : ℕ) : ℕ :=
  2 * (a * b + b * c + c * a)

/-- The volume of a rectangular solid with dimensions a, b, and c. -/
def volume (a b c : ℕ) : ℕ :=
  a * b * c

theorem rectangular_solid_surface_area (a b c : ℕ) :
  is_prime a ∧ is_prime b ∧ is_prime c ∧ volume a b c = 308 →
  surface_area a b c = 226 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_surface_area_l182_18217


namespace NUMINAMATH_CALUDE_proportion_solution_l182_18201

theorem proportion_solution (x : ℝ) : (0.6 / x = 5 / 8) → x = 0.96 := by
  sorry

end NUMINAMATH_CALUDE_proportion_solution_l182_18201


namespace NUMINAMATH_CALUDE_ed_has_27_pets_l182_18211

/-- Represents the number of pets Ed has -/
structure Pets where
  dogs : ℕ
  cats : ℕ
  fish : ℕ
  birds : ℕ
  turtles : ℕ

/-- The conditions given in the problem -/
def petConditions (p : Pets) : Prop :=
  p.dogs = 2 ∧
  p.cats = 3 ∧
  p.fish = 3 * p.birds ∧
  p.fish = 2 * (p.dogs + p.cats) ∧
  p.turtles = p.birds / 2

/-- The total number of pets -/
def totalPets (p : Pets) : ℕ :=
  p.dogs + p.cats + p.fish + p.birds + p.turtles

/-- Theorem stating that given the conditions, Ed has 27 pets in total -/
theorem ed_has_27_pets :
  ∃ p : Pets, petConditions p ∧ totalPets p = 27 := by
  sorry


end NUMINAMATH_CALUDE_ed_has_27_pets_l182_18211


namespace NUMINAMATH_CALUDE_T_minus_n_is_even_l182_18297

/-- The number of non-empty subsets with integer average -/
def T (n : ℕ) : ℕ := sorry

/-- Theorem: T_n - n is even for all n > 1 -/
theorem T_minus_n_is_even (n : ℕ) (h : n > 1) : Even (T n - n) := by
  sorry

end NUMINAMATH_CALUDE_T_minus_n_is_even_l182_18297


namespace NUMINAMATH_CALUDE_fraction_calculation_l182_18296

theorem fraction_calculation : (500^2 : ℝ) / (152^2 - 148^2) = 208.333 := by sorry

end NUMINAMATH_CALUDE_fraction_calculation_l182_18296


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l182_18223

theorem rectangle_perimeter (L W : ℝ) (h : L * W - (L - 4) * (W - 4) = 168) :
  2 * (L + W) = 92 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l182_18223


namespace NUMINAMATH_CALUDE_cost_per_sandwich_is_correct_l182_18224

-- Define the problem parameters
def sandwiches_per_loaf : ℕ := 10
def total_sandwiches : ℕ := 50
def bread_cost : ℚ := 4
def meat_cost : ℚ := 5
def cheese_cost : ℚ := 4
def meat_packs_per_loaf : ℕ := 2
def cheese_packs_per_loaf : ℕ := 2
def cheese_coupon : ℚ := 1
def meat_coupon : ℚ := 1
def discount_threshold : ℚ := 60
def discount_rate : ℚ := 0.1

-- Define the function to calculate the cost per sandwich
def cost_per_sandwich : ℚ :=
  let loaves := total_sandwiches / sandwiches_per_loaf
  let meat_packs := loaves * meat_packs_per_loaf
  let cheese_packs := loaves * cheese_packs_per_loaf
  let total_cost := loaves * bread_cost + meat_packs * meat_cost + cheese_packs * cheese_cost
  let discounted_cost := total_cost - cheese_coupon - meat_coupon
  let final_cost := if discounted_cost > discount_threshold
                    then discounted_cost * (1 - discount_rate)
                    else discounted_cost
  final_cost / total_sandwiches

-- Theorem to prove
theorem cost_per_sandwich_is_correct :
  cost_per_sandwich = 1.944 := by sorry

end NUMINAMATH_CALUDE_cost_per_sandwich_is_correct_l182_18224


namespace NUMINAMATH_CALUDE_parabola_point_distance_l182_18256

/-- Given a parabola y^2 = x with focus F(1/4, 0), prove that for any point A(x₀, y₀) on the parabola,
    if AF = |5/4 * x₀|, then x₀ = 1. -/
theorem parabola_point_distance (x₀ y₀ : ℝ) : 
  y₀^2 = x₀ →  -- Point A is on the parabola
  ((x₀ - 1/4)^2 + y₀^2)^(1/2) = |5/4 * x₀| →  -- AF = |5/4 * x₀|
  x₀ = 1 := by
sorry

end NUMINAMATH_CALUDE_parabola_point_distance_l182_18256


namespace NUMINAMATH_CALUDE_divide_and_power_l182_18202

theorem divide_and_power : (5 / (1 / 5)) ^ 3 = 15625 := by sorry

end NUMINAMATH_CALUDE_divide_and_power_l182_18202


namespace NUMINAMATH_CALUDE_min_fraction_sum_l182_18218

def Digits := Finset.range 8

theorem min_fraction_sum (A B C D : ℕ) 
  (hA : A ∈ Digits) (hB : B ∈ Digits) (hC : C ∈ Digits) (hD : D ∈ Digits)
  (hAB : A ≠ B) (hAC : A ≠ C) (hAD : A ≠ D) (hBC : B ≠ C) (hBD : B ≠ D) (hCD : C ≠ D)
  (hB_pos : B > 0) (hD_pos : D > 0) :
  (A : ℚ) / B + (C : ℚ) / D ≥ 11 / 28 :=
sorry

end NUMINAMATH_CALUDE_min_fraction_sum_l182_18218


namespace NUMINAMATH_CALUDE_dorchester_puppies_washed_l182_18257

/-- Calculates the number of puppies washed given the daily base pay, per-puppy pay, and total earnings. -/
def puppies_washed (base_pay per_puppy_pay total_earnings : ℚ) : ℚ :=
  (total_earnings - base_pay) / per_puppy_pay

/-- Proves that Dorchester washed 16 puppies given the specified conditions. -/
theorem dorchester_puppies_washed :
  puppies_washed 40 2.25 76 = 16 := by
  sorry

#eval puppies_washed 40 2.25 76

end NUMINAMATH_CALUDE_dorchester_puppies_washed_l182_18257


namespace NUMINAMATH_CALUDE_tangent_line_y_intercept_l182_18230

/-- The function representing the curve y = x^3 + 2x - 1 -/
def f (x : ℝ) : ℝ := x^3 + 2*x - 1

/-- The derivative of the function f -/
def f' (x : ℝ) : ℝ := 3*x^2 + 2

/-- The point P on the curve -/
def P : ℝ × ℝ := (1, 2)

/-- The slope of the tangent line at point P -/
def k : ℝ := f' P.1

/-- The y-intercept of the tangent line -/
def b : ℝ := P.2 - k * P.1

theorem tangent_line_y_intercept :
  b = -3 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_y_intercept_l182_18230


namespace NUMINAMATH_CALUDE_two_burritos_five_quesadillas_cost_l182_18248

/-- The price of a burrito in dollars -/
def burrito_price : ℝ := sorry

/-- The price of a quesadilla in dollars -/
def quesadilla_price : ℝ := sorry

/-- The condition that one burrito and four quesadillas cost $3.50 -/
axiom condition1 : burrito_price + 4 * quesadilla_price = 3.50

/-- The condition that four burritos and one quesadilla cost $4.10 -/
axiom condition2 : 4 * burrito_price + quesadilla_price = 4.10

/-- The theorem stating that two burritos and five quesadillas cost $5.02 -/
theorem two_burritos_five_quesadillas_cost :
  2 * burrito_price + 5 * quesadilla_price = 5.02 := by sorry

end NUMINAMATH_CALUDE_two_burritos_five_quesadillas_cost_l182_18248


namespace NUMINAMATH_CALUDE_euler_identity_complex_power_exp_sum_bound_l182_18233

-- Define the complex exponential function
noncomputable def cexp (x : ℝ) : ℂ := Complex.exp (x * Complex.I)

-- Euler's formula
axiom euler_formula (x : ℝ) : cexp x = Complex.cos x + Complex.I * Complex.sin x

-- Theorems to prove
theorem euler_identity : cexp π + 1 = 0 := by sorry

theorem complex_power : (1/2 + Complex.I * (Real.sqrt 3)/2) ^ 2022 = 1 := by sorry

theorem exp_sum_bound (x : ℝ) : Complex.abs (cexp x + cexp (-x)) ≤ 2 := by sorry

end NUMINAMATH_CALUDE_euler_identity_complex_power_exp_sum_bound_l182_18233


namespace NUMINAMATH_CALUDE_carlos_has_largest_answer_l182_18290

def alice_calculation (x : ℕ) : ℕ := ((x - 3) * 3) + 5

def bob_calculation (x : ℕ) : ℕ := (x^2 - 4) + 5

def carlos_calculation (x : ℕ) : ℕ := (x - 2 + 3)^2

theorem carlos_has_largest_answer :
  let initial_number := 12
  carlos_calculation initial_number > alice_calculation initial_number ∧
  carlos_calculation initial_number > bob_calculation initial_number :=
by sorry

end NUMINAMATH_CALUDE_carlos_has_largest_answer_l182_18290


namespace NUMINAMATH_CALUDE_range_of_sine_function_l182_18251

open Real

theorem range_of_sine_function :
  let f : ℝ → ℝ := λ x ↦ sin (2 * x + π / 4)
  let domain : Set ℝ := { x | π / 4 ≤ x ∧ x ≤ π / 2 }
  ∀ y ∈ Set.range (f ∘ (λ x ↦ x : domain → ℝ)),
    -Real.sqrt 2 / 2 ≤ y ∧ y ≤ Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_sine_function_l182_18251


namespace NUMINAMATH_CALUDE_product_divisible_by_4_probability_l182_18270

def is_divisible_by_4 (n : ℕ) : Prop := ∃ k : ℕ, n = 4 * k

def count_pairs_divisible_by_4 (n : ℕ) : ℕ :=
  let even_count := n / 2
  let multiple_of_4_count := n / 4
  (even_count.choose 2) + multiple_of_4_count * (even_count - multiple_of_4_count)

theorem product_divisible_by_4_probability (n : ℕ) (hn : n = 20) :
  (count_pairs_divisible_by_4 n : ℚ) / (n.choose 2) = 7 / 19 :=
sorry

end NUMINAMATH_CALUDE_product_divisible_by_4_probability_l182_18270


namespace NUMINAMATH_CALUDE_rationalize_denominator_l182_18216

theorem rationalize_denominator : 3 / Real.sqrt 48 = Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l182_18216


namespace NUMINAMATH_CALUDE_mod_product_253_649_l182_18280

theorem mod_product_253_649 (n : ℕ) : 
  253 * 649 ≡ n [ZMOD 100] → 0 ≤ n → n < 100 → n = 97 := by
  sorry

end NUMINAMATH_CALUDE_mod_product_253_649_l182_18280


namespace NUMINAMATH_CALUDE_zoo_trip_admission_cost_l182_18283

theorem zoo_trip_admission_cost 
  (total_budget : ℕ) 
  (bus_rental_cost : ℕ) 
  (num_students : ℕ) 
  (h1 : total_budget = 350) 
  (h2 : bus_rental_cost = 100) 
  (h3 : num_students = 25) :
  (total_budget - bus_rental_cost) / num_students = 10 :=
by sorry

end NUMINAMATH_CALUDE_zoo_trip_admission_cost_l182_18283


namespace NUMINAMATH_CALUDE_complex_power_magnitude_l182_18269

theorem complex_power_magnitude : 
  Complex.abs ((1/2 : ℂ) + (Complex.I * (Real.sqrt 3)/2))^12 = 1 := by sorry

end NUMINAMATH_CALUDE_complex_power_magnitude_l182_18269


namespace NUMINAMATH_CALUDE_gold_coin_percentage_l182_18265

theorem gold_coin_percentage 
  (total_objects : ℝ) 
  (beads_and_rings_percent : ℝ) 
  (beads_ratio : ℝ) 
  (silver_coins_percent : ℝ) 
  (h1 : beads_and_rings_percent = 30) 
  (h2 : beads_ratio = 1/2) 
  (h3 : silver_coins_percent = 35) : 
  let coins_percent := 100 - beads_and_rings_percent
  let gold_coins_percent := coins_percent * (100 - silver_coins_percent) / 100
  gold_coins_percent = 45.5 := by
sorry

end NUMINAMATH_CALUDE_gold_coin_percentage_l182_18265


namespace NUMINAMATH_CALUDE_round_trip_average_speed_l182_18242

/-- Calculates the average speed for a round trip boat journey on a river -/
theorem round_trip_average_speed
  (upstream_speed : ℝ)
  (downstream_speed : ℝ)
  (river_current : ℝ)
  (h1 : upstream_speed = 4)
  (h2 : downstream_speed = 7)
  (h3 : river_current = 2)
  : (2 / ((1 / (upstream_speed - river_current)) + (1 / (downstream_speed + river_current)))) = 36 / 11 := by
  sorry

end NUMINAMATH_CALUDE_round_trip_average_speed_l182_18242


namespace NUMINAMATH_CALUDE_sequence_difference_l182_18282

/-- The sequence a_n with sum S_n = n^2 + 2n for n ∈ ℕ* -/
def S (n : ℕ+) : ℕ := n.val^2 + 2*n.val

/-- The n-th term of the sequence -/
def a (n : ℕ+) : ℕ := 2*n.val + 1

theorem sequence_difference (n m : ℕ+) (h : m.val - n.val = 5) :
  a m - a n = 10 := by
  sorry

end NUMINAMATH_CALUDE_sequence_difference_l182_18282


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l182_18208

theorem quadratic_equation_roots (m : ℝ) : 
  (∃ x : ℝ, x^2 - 4*x + m = 0 ∧ x = -2) → 
  m = -12 ∧ ∃ y : ℝ, y^2 - 4*y + m = 0 ∧ y = 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l182_18208


namespace NUMINAMATH_CALUDE_cone_height_ratio_l182_18281

theorem cone_height_ratio (base_circumference : Real) (original_height : Real) (new_volume : Real) :
  base_circumference = 24 * Real.pi →
  original_height = 36 →
  new_volume = 432 * Real.pi →
  ∃ (new_height : Real),
    (1 / 3) * Real.pi * ((base_circumference / (2 * Real.pi)) ^ 2) * new_height = new_volume ∧
    new_height / original_height = 1 / 4 :=
by sorry

end NUMINAMATH_CALUDE_cone_height_ratio_l182_18281


namespace NUMINAMATH_CALUDE_chips_and_juice_weight_l182_18263

/-- Given the weight of chips and juice bottles, calculate the total weight of a specific quantity -/
theorem chips_and_juice_weight
  (chip_weight : ℝ) -- Weight of a bag of chips
  (juice_weight : ℝ) -- Weight of a bottle of juice
  (h1 : 2 * chip_weight = 800) -- Weight of 2 bags of chips is 800 g
  (h2 : chip_weight = juice_weight + 350) -- A bag of chips is 350 g heavier than a bottle of juice
  : 5 * chip_weight + 4 * juice_weight = 2200 := by
  sorry

end NUMINAMATH_CALUDE_chips_and_juice_weight_l182_18263


namespace NUMINAMATH_CALUDE_credits_needed_is_84_l182_18267

/-- The number of credits needed to buy cards for a game -/
def credits_needed : ℕ :=
  let red_card_cost : ℕ := 3
  let blue_card_cost : ℕ := 5
  let total_cards_required : ℕ := 20
  let red_cards_owned : ℕ := 8
  let blue_cards_needed : ℕ := total_cards_required - red_cards_owned
  red_card_cost * red_cards_owned + blue_card_cost * blue_cards_needed

theorem credits_needed_is_84 : credits_needed = 84 := by
  sorry

end NUMINAMATH_CALUDE_credits_needed_is_84_l182_18267


namespace NUMINAMATH_CALUDE_triangle_formation_theorem_l182_18264

/-- Triangle inequality theorem checker -/
def satisfies_triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Set of three line segments -/
structure TriangleSet where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Theorem: Only the set (5, 13, 12) can form a triangle -/
theorem triangle_formation_theorem :
  let set_a := TriangleSet.mk 3 10 5
  let set_b := TriangleSet.mk 4 8 4
  let set_c := TriangleSet.mk 5 13 12
  let set_d := TriangleSet.mk 2 7 4
  satisfies_triangle_inequality set_c.a set_c.b set_c.c ∧
  ¬satisfies_triangle_inequality set_a.a set_a.b set_a.c ∧
  ¬satisfies_triangle_inequality set_b.a set_b.b set_b.c ∧
  ¬satisfies_triangle_inequality set_d.a set_d.b set_d.c :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_formation_theorem_l182_18264


namespace NUMINAMATH_CALUDE_jake_peaches_l182_18237

/-- 
Given that Steven has 13 peaches and Jake has six fewer peaches than Steven,
prove that Jake has 7 peaches.
-/
theorem jake_peaches (steven_peaches : ℕ) (jake_peaches : ℕ) 
  (h1 : steven_peaches = 13)
  (h2 : jake_peaches = steven_peaches - 6) :
  jake_peaches = 7 := by
  sorry

end NUMINAMATH_CALUDE_jake_peaches_l182_18237


namespace NUMINAMATH_CALUDE_order_of_numbers_l182_18275

theorem order_of_numbers (m n : ℝ) (hm : m < 0) (hn : n > 0) (hmn : m + n < 0) :
  -m > n ∧ n > -n ∧ -n > m := by sorry

end NUMINAMATH_CALUDE_order_of_numbers_l182_18275


namespace NUMINAMATH_CALUDE_purple_nails_count_l182_18244

/-- The number of nails painted purple -/
def purple_nails : ℕ := sorry

/-- The number of nails painted blue -/
def blue_nails : ℕ := 8

/-- The number of nails painted striped -/
def striped_nails : ℕ := sorry

/-- The total number of nails -/
def total_nails : ℕ := 20

/-- The difference in percentage points between blue and striped nails -/
def percentage_difference : ℝ := 10

theorem purple_nails_count : purple_nails = 6 := by
  have h1 : purple_nails + blue_nails + striped_nails = total_nails := sorry
  have h2 : (blue_nails : ℝ) / total_nails * 100 - (striped_nails : ℝ) / total_nails * 100 = percentage_difference := sorry
  sorry

end NUMINAMATH_CALUDE_purple_nails_count_l182_18244


namespace NUMINAMATH_CALUDE_units_digit_of_72_cubed_minus_24_cubed_l182_18299

theorem units_digit_of_72_cubed_minus_24_cubed : ∃ n : ℕ, 72^3 - 24^3 ≡ 4 [MOD 10] ∧ n * 10 + 4 = 72^3 - 24^3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_72_cubed_minus_24_cubed_l182_18299


namespace NUMINAMATH_CALUDE_bus_interval_is_30_minutes_l182_18258

/-- Represents a bus station schedule -/
structure BusSchedule where
  operatingHoursPerDay : ℕ
  operatingDays : ℕ
  totalBuses : ℕ

/-- Calculates the time interval between bus departures in minutes -/
def calculateInterval (schedule : BusSchedule) : ℕ :=
  let minutesPerDay := schedule.operatingHoursPerDay * 60
  let busesPerDay := schedule.totalBuses / schedule.operatingDays
  minutesPerDay / busesPerDay

/-- Theorem: The time interval between bus departures is 30 minutes -/
theorem bus_interval_is_30_minutes (schedule : BusSchedule) 
    (h1 : schedule.operatingHoursPerDay = 12)
    (h2 : schedule.operatingDays = 5)
    (h3 : schedule.totalBuses = 120) :
  calculateInterval schedule = 30 := by
  sorry

#eval calculateInterval { operatingHoursPerDay := 12, operatingDays := 5, totalBuses := 120 }

end NUMINAMATH_CALUDE_bus_interval_is_30_minutes_l182_18258


namespace NUMINAMATH_CALUDE_min_sum_abs_values_l182_18205

def matrix_condition (a b c d : ℤ) : Prop :=
  let M : Matrix (Fin 2) (Fin 2) ℤ := !![a, b; c, d]
  M ^ 2 = !![5, 0; 0, 5]

theorem min_sum_abs_values (a b c d : ℤ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
  (h_matrix : matrix_condition a b c d) :
  (∀ a' b' c' d' : ℤ, a' ≠ 0 → b' ≠ 0 → c' ≠ 0 → d' ≠ 0 → 
    matrix_condition a' b' c' d' → 
    |a| + |b| + |c| + |d| ≤ |a'| + |b'| + |c'| + |d'|) ∧
  |a| + |b| + |c| + |d| = 6 :=
sorry

end NUMINAMATH_CALUDE_min_sum_abs_values_l182_18205


namespace NUMINAMATH_CALUDE_tank_empty_time_l182_18200

/- Define the tank capacity in liters -/
def tank_capacity : ℝ := 5760

/- Define the time it takes for the leak to empty the tank in hours -/
def leak_empty_time : ℝ := 6

/- Define the inlet pipe fill rate in liters per minute -/
def inlet_fill_rate : ℝ := 4

/- Define the time it takes to empty the tank with inlet open in hours -/
def empty_time_with_inlet : ℝ := 8

/- Theorem statement -/
theorem tank_empty_time :
  let leak_rate := tank_capacity / leak_empty_time
  let inlet_rate := inlet_fill_rate * 60
  let net_empty_rate := leak_rate - inlet_rate
  tank_capacity / net_empty_rate = empty_time_with_inlet :=
by sorry

end NUMINAMATH_CALUDE_tank_empty_time_l182_18200


namespace NUMINAMATH_CALUDE_block_has_twelve_floors_l182_18222

/-- Represents a block of flats with the given conditions -/
structure BlockOfFlats where
  half_floors : ℕ
  apartments_per_floor_1 : ℕ
  apartments_per_floor_2 : ℕ
  max_residents_per_apartment : ℕ
  max_total_residents : ℕ

/-- The number of floors in the block of flats -/
def total_floors (b : BlockOfFlats) : ℕ := 2 * b.half_floors

/-- The theorem stating that the block of flats has 12 floors -/
theorem block_has_twelve_floors (b : BlockOfFlats)
  (h1 : b.apartments_per_floor_1 = 6)
  (h2 : b.apartments_per_floor_2 = 5)
  (h3 : b.max_residents_per_apartment = 4)
  (h4 : b.max_total_residents = 264) :
  total_floors b = 12 := by
  sorry

#check block_has_twelve_floors

end NUMINAMATH_CALUDE_block_has_twelve_floors_l182_18222
