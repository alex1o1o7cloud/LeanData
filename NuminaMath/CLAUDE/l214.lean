import Mathlib

namespace NUMINAMATH_CALUDE_smallest_addition_for_divisibility_l214_21467

theorem smallest_addition_for_divisibility (n : ℕ) (h : n = 8261955) :
  ∃ x : ℕ, x = 2 ∧ 
  (∀ y : ℕ, y < x → ¬(11 ∣ (n + y))) ∧
  (11 ∣ (n + x)) := by
  sorry

end NUMINAMATH_CALUDE_smallest_addition_for_divisibility_l214_21467


namespace NUMINAMATH_CALUDE_expected_attacked_squares_theorem_l214_21416

/-- The number of squares on a chessboard -/
def chessboardSize : ℕ := 64

/-- The number of rooks placed on the board -/
def numRooks : ℕ := 3

/-- The probability that a specific square is not attacked by a single rook -/
def probNotAttacked : ℚ := (49 : ℚ) / 64

/-- The expected number of squares under attack by rooks on a chessboard -/
def expectedAttackedSquares : ℚ :=
  chessboardSize * (1 - probNotAttacked ^ numRooks)

/-- Theorem stating the expected number of squares under attack -/
theorem expected_attacked_squares_theorem :
  expectedAttackedSquares = 64 * (1 - (49/64)^3) :=
sorry

end NUMINAMATH_CALUDE_expected_attacked_squares_theorem_l214_21416


namespace NUMINAMATH_CALUDE_quadratic_roots_product_l214_21462

theorem quadratic_roots_product (a b : ℝ) : 
  (3 * a^2 + 9 * a - 21 = 0) → 
  (3 * b^2 + 9 * b - 21 = 0) → 
  (3*a - 4) * (5*b - 6) = -27 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_product_l214_21462


namespace NUMINAMATH_CALUDE_simplify_expression_l214_21460

theorem simplify_expression (n : ℕ) : (2^(n+4) - 3*(2^n)) / (2*(2^(n+3))) = 13/16 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l214_21460


namespace NUMINAMATH_CALUDE_ninas_money_l214_21484

theorem ninas_money (x : ℚ) 
  (h1 : 10 * x = 14 * (x - 1)) : 10 * x = 35 :=
by sorry

end NUMINAMATH_CALUDE_ninas_money_l214_21484


namespace NUMINAMATH_CALUDE_mean_score_is_215_div_11_l214_21415

def points : List ℕ := [15, 20, 25, 30]
def players : List ℕ := [5, 3, 2, 1]

theorem mean_score_is_215_div_11 : 
  (List.sum (List.zipWith (· * ·) points players)) / (List.sum players) = 215 / 11 := by
  sorry

end NUMINAMATH_CALUDE_mean_score_is_215_div_11_l214_21415


namespace NUMINAMATH_CALUDE_percentage_increase_l214_21470

theorem percentage_increase (original : ℝ) (new : ℝ) :
  original = 30 →
  new = 40 →
  (new - original) / original = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_percentage_increase_l214_21470


namespace NUMINAMATH_CALUDE_chessboard_placements_l214_21494

/-- Represents a standard 8x8 chessboard -/
def Chessboard := Fin 8 × Fin 8

/-- Represents the different types of chess pieces -/
inductive ChessPiece
| Rook
| King
| Bishop
| Knight
| Queen

/-- Returns true if two pieces of the given type at the given positions do not attack each other -/
def not_attacking (piece : ChessPiece) (pos1 pos2 : Chessboard) : Prop := sorry

/-- Counts the number of ways to place two identical pieces on the chessboard without attacking each other -/
def count_placements (piece : ChessPiece) : ℕ := sorry

theorem chessboard_placements :
  (count_placements ChessPiece.Rook = 1568) ∧
  (count_placements ChessPiece.King = 1806) ∧
  (count_placements ChessPiece.Bishop = 1736) ∧
  (count_placements ChessPiece.Knight = 1848) ∧
  (count_placements ChessPiece.Queen = 1288) := by sorry

end NUMINAMATH_CALUDE_chessboard_placements_l214_21494


namespace NUMINAMATH_CALUDE_maria_age_l214_21466

theorem maria_age (maria ann : ℕ) : 
  maria = ann - 3 →
  maria - 4 = (ann - 4) / 2 →
  maria = 7 := by sorry

end NUMINAMATH_CALUDE_maria_age_l214_21466


namespace NUMINAMATH_CALUDE_complex_cube_theorem_l214_21474

theorem complex_cube_theorem (z : ℂ) (h1 : Complex.abs (z - 2) = 2) (h2 : Complex.abs z = 2) : 
  z^3 = -8 := by sorry

end NUMINAMATH_CALUDE_complex_cube_theorem_l214_21474


namespace NUMINAMATH_CALUDE_parallelogram_fourth_vertex_l214_21445

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A vector in 2D space -/
def Vector2D := Point2D

/-- Addition of two vectors -/
def vectorAdd (v w : Vector2D) : Vector2D :=
  ⟨v.x + w.x, v.y + w.y⟩

/-- Subtraction of two vectors -/
def vectorSub (v w : Vector2D) : Vector2D :=
  ⟨v.x - w.x, v.y - w.y⟩

/-- Negation of a vector -/
def vectorNeg (v : Vector2D) : Vector2D :=
  ⟨-v.x, -v.y⟩

theorem parallelogram_fourth_vertex 
  (A B C : Point2D)
  (h : A = ⟨-1, -2⟩)
  (h' : B = ⟨3, 1⟩)
  (h'' : C = ⟨0, 2⟩) :
  let D := vectorAdd C (vectorAdd A (vectorNeg B))
  D = ⟨-4, -1⟩ := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_fourth_vertex_l214_21445


namespace NUMINAMATH_CALUDE_sum_of_x_solutions_is_zero_l214_21475

theorem sum_of_x_solutions_is_zero :
  ∀ x₁ x₂ : ℝ,
  (∃ y : ℝ, y = 7 ∧ x₁^2 + y^2 = 100 ∧ x₂^2 + y^2 = 100) →
  x₁ + x₂ = 0 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_x_solutions_is_zero_l214_21475


namespace NUMINAMATH_CALUDE_solution_range_l214_21499

-- Define the system of inequalities
def system (x m : ℝ) : Prop :=
  (6 - 3*(x + 1) < x - 9) ∧ 
  (x - m > -1) ∧ 
  (x > 3)

-- Theorem statement
theorem solution_range (m : ℝ) : 
  (∀ x, system x m → x > 3) → m ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_solution_range_l214_21499


namespace NUMINAMATH_CALUDE_grocery_solution_l214_21419

/-- Represents the grocery shopping problem --/
def grocery_problem (mustard_oil_price : ℝ) (pasta_price : ℝ) (pasta_amount : ℝ) 
  (sauce_price : ℝ) (sauce_amount : ℝ) (initial_money : ℝ) (remaining_money : ℝ) : Prop :=
  ∃ (mustard_oil_amount : ℝ),
    mustard_oil_amount ≥ 0 ∧
    mustard_oil_price > 0 ∧
    pasta_price > 0 ∧
    pasta_amount > 0 ∧
    sauce_price > 0 ∧
    sauce_amount > 0 ∧
    initial_money > 0 ∧
    remaining_money ≥ 0 ∧
    initial_money - remaining_money = 
      mustard_oil_amount * mustard_oil_price + pasta_amount * pasta_price + sauce_amount * sauce_price ∧
    mustard_oil_amount = 2

/-- Theorem stating the solution to the grocery problem --/
theorem grocery_solution : 
  grocery_problem 13 4 3 5 1 50 7 :=
by sorry

end NUMINAMATH_CALUDE_grocery_solution_l214_21419


namespace NUMINAMATH_CALUDE_journey_time_difference_l214_21401

theorem journey_time_difference 
  (speed : ℝ) 
  (distance1 : ℝ) 
  (distance2 : ℝ) 
  (h1 : speed = 40) 
  (h2 : distance1 = 360) 
  (h3 : distance2 = 400) :
  (distance2 - distance1) / speed * 60 = 60 := by
  sorry

end NUMINAMATH_CALUDE_journey_time_difference_l214_21401


namespace NUMINAMATH_CALUDE_binomial_product_l214_21407

theorem binomial_product (x : ℝ) : (4 * x - 3) * (x + 7) = 4 * x^2 + 25 * x - 21 := by
  sorry

end NUMINAMATH_CALUDE_binomial_product_l214_21407


namespace NUMINAMATH_CALUDE_reach_64_from_2_cannot_reach_2_2011_from_2_l214_21473

def cube (x : ℚ) : ℚ := x^3

def div_by_8 (x : ℚ) : ℚ := x / 8

inductive Operation
| Cube
| DivBy8

def apply_operation (x : ℚ) (op : Operation) : ℚ :=
  match op with
  | Operation.Cube => cube x
  | Operation.DivBy8 => div_by_8 x

def can_reach (start : ℚ) (target : ℚ) : Prop :=
  ∃ (ops : List Operation), target = ops.foldl apply_operation start

theorem reach_64_from_2 : can_reach 2 64 := by sorry

theorem cannot_reach_2_2011_from_2 : ¬ can_reach 2 (2^2011) := by sorry

end NUMINAMATH_CALUDE_reach_64_from_2_cannot_reach_2_2011_from_2_l214_21473


namespace NUMINAMATH_CALUDE_isabella_currency_exchange_l214_21483

theorem isabella_currency_exchange (d : ℕ) : 
  (11 * d / 8 : ℚ) - 80 = d →
  (d / 100 + (d / 10) % 10 + d % 10 : ℕ) = 6 :=
by sorry

end NUMINAMATH_CALUDE_isabella_currency_exchange_l214_21483


namespace NUMINAMATH_CALUDE_imaginary_roots_condition_l214_21425

/-- The quadratic equation kx^2 + mx + k = 0 (where k ≠ 0) has imaginary roots
    if and only if m^2 < 4k^2 -/
theorem imaginary_roots_condition (k m : ℝ) (hk : k ≠ 0) :
  (∀ x : ℝ, k * x^2 + m * x + k ≠ 0) ↔ m^2 < 4 * k^2 := by sorry

end NUMINAMATH_CALUDE_imaginary_roots_condition_l214_21425


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_simplify_rational_expression_l214_21412

-- Problem 1
theorem simplify_sqrt_expression :
  3 * Real.sqrt 12 - 6 * Real.sqrt (1/3) + Real.sqrt 48 = 8 * Real.sqrt 3 := by sorry

-- Problem 2
theorem simplify_rational_expression (m : ℝ) (h : m^2 + 3*m - 4 = 0) :
  (m - 3) / (3 * m^2 - 6 * m) / (m + 2 - 5 / (m - 2)) = 1 / 12 := by sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_simplify_rational_expression_l214_21412


namespace NUMINAMATH_CALUDE_inequality_proof_l214_21431

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum_squares : a^2 + b^2 + c^2 = 1) : 
  a*b/c + b*c/a + c*a/b ≥ Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l214_21431


namespace NUMINAMATH_CALUDE_business_value_calculation_l214_21432

/-- Calculates the total value of a business given partial ownership and sale information. -/
theorem business_value_calculation (total_ownership : ℚ) (sold_fraction : ℚ) (sale_price : ℕ) :
  total_ownership = 2 / 3 →
  sold_fraction = 3 / 4 →
  sale_price = 30000 →
  (total_ownership * sold_fraction * sale_price) / (total_ownership * sold_fraction) = 60000 := by
  sorry

end NUMINAMATH_CALUDE_business_value_calculation_l214_21432


namespace NUMINAMATH_CALUDE_enclosing_polygons_sides_l214_21449

/-- The number of sides of the central polygon -/
def m : ℕ := 12

/-- The number of polygons enclosing the central polygon -/
def enclosing_polygons : ℕ := 12

/-- The number of sides of each enclosing polygon -/
def n : ℕ := 12

/-- The interior angle of a regular polygon with k sides -/
def interior_angle (k : ℕ) : ℚ := (k - 2) * 180 / k

/-- The exterior angle of a regular polygon with k sides -/
def exterior_angle (k : ℕ) : ℚ := 360 / k

/-- Theorem: In a configuration where a regular polygon with m sides is exactly
    enclosed by 'enclosing_polygons' number of regular polygons each with n sides,
    the value of n must be equal to the number of sides of the central polygon. -/
theorem enclosing_polygons_sides (h1 : m = 12) (h2 : enclosing_polygons = 12) :
  n = m := by sorry

end NUMINAMATH_CALUDE_enclosing_polygons_sides_l214_21449


namespace NUMINAMATH_CALUDE_roots_cubic_expression_l214_21454

theorem roots_cubic_expression (γ δ : ℝ) : 
  (γ^2 - 3*γ + 2 = 0) → 
  (δ^2 - 3*δ + 2 = 0) → 
  8*γ^3 - 6*δ^2 = 48 := by
sorry

end NUMINAMATH_CALUDE_roots_cubic_expression_l214_21454


namespace NUMINAMATH_CALUDE_min_y_value_l214_21405

theorem min_y_value (x y : ℝ) (h : x^2 + y^2 = 30*x + 20*y) :
  ∃ (y_min : ℝ), y_min = 10 - 5 * Real.sqrt 13 ∧ ∀ (y' : ℝ), ∃ (x' : ℝ), x'^2 + y'^2 = 30*x' + 20*y' → y' ≥ y_min :=
sorry

end NUMINAMATH_CALUDE_min_y_value_l214_21405


namespace NUMINAMATH_CALUDE_unique_row_with_29_l214_21427

def pascal_coeff (n k : ℕ) : ℕ := Nat.choose n k

def contains_29 (row : ℕ) : Prop :=
  ∃ k, k ≤ row ∧ pascal_coeff row k = 29

theorem unique_row_with_29 :
  ∃! row, contains_29 row :=
sorry

end NUMINAMATH_CALUDE_unique_row_with_29_l214_21427


namespace NUMINAMATH_CALUDE_ceiling_negative_three_point_seven_l214_21422

theorem ceiling_negative_three_point_seven : ⌈(-3.7 : ℝ)⌉ = -3 := by sorry

end NUMINAMATH_CALUDE_ceiling_negative_three_point_seven_l214_21422


namespace NUMINAMATH_CALUDE_ratio_from_mean_ratio_l214_21442

theorem ratio_from_mean_ratio (x y : ℝ) (h : x > 0 ∧ y > 0) :
  (x + y) / 2 / Real.sqrt (x * y) = 25 / 24 →
  (x / y = 16 / 9 ∨ x / y = 9 / 16) := by
  sorry

end NUMINAMATH_CALUDE_ratio_from_mean_ratio_l214_21442


namespace NUMINAMATH_CALUDE_cos_sin_75_product_equality_l214_21461

theorem cos_sin_75_product_equality : 
  (Real.cos (75 * π / 180) + Real.sin (75 * π / 180)) * 
  (Real.cos (75 * π / 180) - Real.sin (75 * π / 180)) = 
  - (Real.sqrt 3) / 2 := by sorry

end NUMINAMATH_CALUDE_cos_sin_75_product_equality_l214_21461


namespace NUMINAMATH_CALUDE_simplify_expression_l214_21496

theorem simplify_expression (x y : ℝ) (h : x ≠ 0) :
  y * (x⁻¹ - 2) = (y * (1 - 2*x)) / x := by sorry

end NUMINAMATH_CALUDE_simplify_expression_l214_21496


namespace NUMINAMATH_CALUDE_pascal_triangle_45th_number_l214_21423

theorem pascal_triangle_45th_number (n : ℕ) : n = 46 →
  Nat.choose n 2 = 1035 :=
by sorry

end NUMINAMATH_CALUDE_pascal_triangle_45th_number_l214_21423


namespace NUMINAMATH_CALUDE_guitar_price_theorem_l214_21455

theorem guitar_price_theorem (hendricks_price : ℝ) (discount_percentage : ℝ) (gerald_price : ℝ) : 
  hendricks_price = 200 →
  discount_percentage = 20 →
  hendricks_price = gerald_price * (1 - discount_percentage / 100) →
  gerald_price = 250 :=
by sorry

end NUMINAMATH_CALUDE_guitar_price_theorem_l214_21455


namespace NUMINAMATH_CALUDE_fair_ride_cost_l214_21472

/-- Represents the fair entrance and ride costs --/
structure FairCosts where
  under18Fee : ℚ
  adultFeeIncrease : ℚ
  totalSpent : ℚ
  numRides : ℕ
  numUnder18 : ℕ
  numAdults : ℕ

/-- Calculates the cost per ride given the fair costs --/
def costPerRide (costs : FairCosts) : ℚ :=
  let adultFee := costs.under18Fee * (1 + costs.adultFeeIncrease)
  let totalEntrance := costs.under18Fee * costs.numUnder18 + adultFee * costs.numAdults
  let totalRideCost := costs.totalSpent - totalEntrance
  totalRideCost / costs.numRides

/-- Theorem stating that the cost per ride is $0.50 given the problem conditions --/
theorem fair_ride_cost :
  let costs : FairCosts := {
    under18Fee := 5,
    adultFeeIncrease := 1/5,
    totalSpent := 41/2,
    numRides := 9,
    numUnder18 := 2,
    numAdults := 1
  }
  costPerRide costs = 1/2 := by sorry


end NUMINAMATH_CALUDE_fair_ride_cost_l214_21472


namespace NUMINAMATH_CALUDE_train_speed_l214_21463

-- Define the train's parameters
def train_length : Real := 240  -- in meters
def crossing_time : Real := 16  -- in seconds

-- Define the conversion factor from m/s to km/h
def mps_to_kmh : Real := 3.6

-- Theorem statement
theorem train_speed :
  let speed_mps := train_length / crossing_time
  let speed_kmh := speed_mps * mps_to_kmh
  speed_kmh = 54 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l214_21463


namespace NUMINAMATH_CALUDE_triangle_area_theorem_l214_21444

theorem triangle_area_theorem (x : ℝ) (h1 : x > 0) : 
  (1/2 : ℝ) * x * (3*x) = 72 → x = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_theorem_l214_21444


namespace NUMINAMATH_CALUDE_product_of_three_numbers_l214_21434

theorem product_of_three_numbers (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a * b = 32) (hac : a * c = 48) (hbc : b * c = 80) :
  a * b * c = 64 * Real.sqrt 30 := by
  sorry

end NUMINAMATH_CALUDE_product_of_three_numbers_l214_21434


namespace NUMINAMATH_CALUDE_complex_modulus_range_l214_21458

theorem complex_modulus_range (a : ℝ) (z : ℂ) (h1 : 0 < a) (h2 : a < 2) (h3 : z = a + Complex.I) :
  1 < Complex.abs z ∧ Complex.abs z < Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_range_l214_21458


namespace NUMINAMATH_CALUDE_integral_inverse_cube_l214_21430

theorem integral_inverse_cube (x : ℝ) (h : x ≠ 0) :
  ∫ (t : ℝ) in Set.Ioo 0 x, 1 / (t^3) = -1 / (2 * x^2) + 1 / 2 := by sorry

end NUMINAMATH_CALUDE_integral_inverse_cube_l214_21430


namespace NUMINAMATH_CALUDE_prove_new_average_weight_l214_21438

def average_weight_problem (num_boys num_girls : ℕ) 
                           (avg_weight_boys avg_weight_girls : ℚ)
                           (lightest_boy_weight lightest_girl_weight : ℚ) : Prop :=
  let total_weight_boys := num_boys * avg_weight_boys
  let total_weight_girls := num_girls * avg_weight_girls
  let remaining_weight_boys := total_weight_boys - lightest_boy_weight
  let remaining_weight_girls := total_weight_girls - lightest_girl_weight
  let total_remaining_weight := remaining_weight_boys + remaining_weight_girls
  let remaining_children := num_boys + num_girls - 2
  let new_average_weight := total_remaining_weight / remaining_children
  new_average_weight = 161.5

theorem prove_new_average_weight : 
  average_weight_problem 8 5 155 125 140 110 := by
  sorry

end NUMINAMATH_CALUDE_prove_new_average_weight_l214_21438


namespace NUMINAMATH_CALUDE_largest_divisor_of_consecutive_odd_product_l214_21487

theorem largest_divisor_of_consecutive_odd_product (n : ℕ) (h : Even n) (h' : n > 0) :
  ∃ (k : ℕ), k = 15 ∧ 
  (∀ m : ℕ, m > k → ¬(m ∣ (n+1)*(n+3)*(n+5)*(n+7)*(n+9)*(n+11)*(n+13))) ∧
  (k ∣ (n+1)*(n+3)*(n+5)*(n+7)*(n+9)*(n+11)*(n+13)) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_consecutive_odd_product_l214_21487


namespace NUMINAMATH_CALUDE_f_properties_l214_21441

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 
  (a / (a^2 - 1)) * (a^x - a^(-x))

theorem f_properties (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  -- 1. f is an increasing odd function on ℝ
  (∀ x : ℝ, f a (-x) = -f a x) ∧
  (∀ x y : ℝ, x < y → f a x < f a y) ∧
  -- 2. For x ∈ (-1, 1), f(1-m) + f(1-m^2) < 0 implies m ∈ (1, √2)
  (∀ m : ℝ, (∀ x : ℝ, -1 < x ∧ x < 1 → f a (1-m) + f a (1-m^2) < 0) → 
    1 < m ∧ m < Real.sqrt 2) ∧
  -- 3. For x ∈ (-∞, 2), f(x) - 4 < 0 implies a ∈ (2 - √3, 2 + √3) \ {1}
  ((∀ x : ℝ, x < 2 → f a x - 4 < 0) → 
    2 - Real.sqrt 3 < a ∧ a < 2 + Real.sqrt 3 ∧ a ≠ 1) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l214_21441


namespace NUMINAMATH_CALUDE_square_diagonal_property_l214_21471

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a square -/
structure Square :=
  (p q r s : Point)

/-- The small square PQRS is contained in the big square -/
def small_square_in_big_square (small : Square) (big : Square) : Prop := sorry

/-- Point A lies on the extension of PQ -/
def point_on_extension (p q a : Point) : Prop := sorry

/-- Points A, B, C, D lie on the sides of the big square in order -/
def points_on_sides (a b c d : Point) (big : Square) : Prop := sorry

/-- Two line segments are equal -/
def segments_equal (p1 q1 p2 q2 : Point) : Prop := sorry

/-- Two line segments are perpendicular -/
def segments_perpendicular (p1 q1 p2 q2 : Point) : Prop := sorry

theorem square_diagonal_property (small big : Square) (a b c d : Point) :
  small_square_in_big_square small big →
  point_on_extension small.p small.q a →
  point_on_extension small.q small.r b →
  point_on_extension small.r small.s c →
  point_on_extension small.s small.p d →
  points_on_sides a b c d big →
  segments_equal a c b d ∧ segments_perpendicular a c b d := by
  sorry

end NUMINAMATH_CALUDE_square_diagonal_property_l214_21471


namespace NUMINAMATH_CALUDE_mothers_offer_l214_21411

def bike_cost : ℕ := 600
def maria_savings : ℕ := 120
def maria_earnings : ℕ := 230

theorem mothers_offer :
  bike_cost - (maria_savings + maria_earnings) = 250 :=
by sorry

end NUMINAMATH_CALUDE_mothers_offer_l214_21411


namespace NUMINAMATH_CALUDE_six_couples_handshakes_l214_21402

/-- The number of handshakes exchanged at a gathering of couples -/
def handshakes_at_gathering (num_couples : ℕ) : ℕ :=
  let total_people := 2 * num_couples
  let handshakes_per_person := total_people - 3
  (total_people * handshakes_per_person) / 2

/-- Theorem stating that for 6 couples, the number of handshakes is 54 -/
theorem six_couples_handshakes :
  handshakes_at_gathering 6 = 54 := by
  sorry

end NUMINAMATH_CALUDE_six_couples_handshakes_l214_21402


namespace NUMINAMATH_CALUDE_no_consecutive_squares_l214_21476

def t (n : ℕ) : ℕ := (Nat.divisors n).card

def a : ℕ → ℕ
  | 0 => 1  -- Arbitrary starting value
  | n + 1 => a n + 2 * t n

theorem no_consecutive_squares (n k : ℕ) :
  a n = k^2 → ¬∃ m : ℕ, a (n + 1) = (k + m)^2 :=
by sorry

end NUMINAMATH_CALUDE_no_consecutive_squares_l214_21476


namespace NUMINAMATH_CALUDE_solution_set_part1_min_value_condition_l214_21443

-- Define the function f
def f (x a b : ℝ) : ℝ := 2 * abs (x + a) + abs (3 * x - b)

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f x 1 0 ≥ 3 * abs x + 1} = {x : ℝ | x ≥ -1/2 ∨ x ≤ -3/2} := by sorry

-- Part 2
theorem min_value_condition :
  ∀ a b : ℝ, a > 0 → b > 0 → (∀ x : ℝ, f x a b ≥ 2) → (∃ x : ℝ, f x a b = 2) →
  3 * a + b = 3 := by sorry

end NUMINAMATH_CALUDE_solution_set_part1_min_value_condition_l214_21443


namespace NUMINAMATH_CALUDE_complex_multiplication_l214_21435

theorem complex_multiplication (z : ℂ) (h : z = 1 + 2 * I) : I * z = -2 + I := by sorry

end NUMINAMATH_CALUDE_complex_multiplication_l214_21435


namespace NUMINAMATH_CALUDE_gcd_special_numbers_l214_21492

theorem gcd_special_numbers : Nat.gcd (2^2010 - 3) (2^2001 - 3) = 1533 := by
  sorry

end NUMINAMATH_CALUDE_gcd_special_numbers_l214_21492


namespace NUMINAMATH_CALUDE_third_term_of_specific_series_l214_21428

/-- Represents an infinite geometric series -/
structure InfiniteGeometricSeries where
  firstTerm : ℝ
  commonRatio : ℝ
  sum : ℝ
  hSum : sum = firstTerm / (1 - commonRatio)
  hRatio : abs commonRatio < 1

/-- The third term of a geometric sequence -/
def thirdTerm (s : InfiniteGeometricSeries) : ℝ :=
  s.firstTerm * s.commonRatio ^ 2

/-- Theorem: In an infinite geometric series with common ratio 1/4 and sum 40, the third term is 15/8 -/
theorem third_term_of_specific_series :
  ∃ s : InfiniteGeometricSeries, 
    s.commonRatio = 1/4 ∧ 
    s.sum = 40 ∧ 
    thirdTerm s = 15/8 := by
  sorry

end NUMINAMATH_CALUDE_third_term_of_specific_series_l214_21428


namespace NUMINAMATH_CALUDE_power_inequality_l214_21424

theorem power_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a^b * b^c * c^a ≤ a^a * b^b * c^c := by
  sorry

end NUMINAMATH_CALUDE_power_inequality_l214_21424


namespace NUMINAMATH_CALUDE_orc_sword_weight_l214_21439

/-- Given a total weight of swords, number of squads, and orcs per squad,
    calculates the weight each orc must carry. -/
def weight_per_orc (total_weight : ℕ) (num_squads : ℕ) (orcs_per_squad : ℕ) : ℚ :=
  total_weight / (num_squads * orcs_per_squad)

/-- Proves that given 1200 pounds of swords, 10 squads, and 8 orcs per squad,
    each orc must carry 15 pounds of swords. -/
theorem orc_sword_weight :
  weight_per_orc 1200 10 8 = 15 := by
  sorry

#eval weight_per_orc 1200 10 8

end NUMINAMATH_CALUDE_orc_sword_weight_l214_21439


namespace NUMINAMATH_CALUDE_wilfred_carrots_l214_21413

/-- The number of carrots Wilfred ate on Tuesday, Wednesday, and Thursday -/
def carrots_tuesday : ℕ → ℕ → ℕ → ℕ → Prop :=
  fun tuesday wednesday thursday total =>
    tuesday + wednesday + thursday = total

theorem wilfred_carrots :
  ∃ (tuesday : ℕ),
    carrots_tuesday tuesday 6 5 15 ∧ tuesday = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_wilfred_carrots_l214_21413


namespace NUMINAMATH_CALUDE_water_surface_scientific_notation_correct_l214_21486

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- The area of water surface in China in km² -/
def water_surface_area : ℕ := 370000

/-- The scientific notation representation of the water surface area -/
def water_surface_scientific : ScientificNotation :=
  { coefficient := 3.7
    exponent := 5
    is_valid := by sorry }

/-- Theorem stating that the water surface area is correctly represented in scientific notation -/
theorem water_surface_scientific_notation_correct :
  (water_surface_scientific.coefficient * (10 : ℝ) ^ water_surface_scientific.exponent) = water_surface_area := by
  sorry

end NUMINAMATH_CALUDE_water_surface_scientific_notation_correct_l214_21486


namespace NUMINAMATH_CALUDE_bottle_cost_l214_21480

theorem bottle_cost (total : ℕ) (wine_extra : ℕ) (h1 : total = 30) (h2 : wine_extra = 26) : 
  ∃ (bottle : ℕ), bottle + (bottle + wine_extra) = total ∧ bottle = 2 := by
  sorry

end NUMINAMATH_CALUDE_bottle_cost_l214_21480


namespace NUMINAMATH_CALUDE_total_pencils_specific_pencil_case_l214_21479

/-- Given an initial number of pencils and a number of pencils added, 
    the total number of pencils is equal to the sum of the initial number and the added number. -/
theorem total_pencils (initial : ℕ) (added : ℕ) : 
  initial + added = initial + added :=
by sorry

/-- In the specific case of 2 initial pencils and 3 added pencils, the total is 5. -/
theorem specific_pencil_case : 
  2 + 3 = 5 :=
by sorry

end NUMINAMATH_CALUDE_total_pencils_specific_pencil_case_l214_21479


namespace NUMINAMATH_CALUDE_sum_of_cubes_l214_21490

theorem sum_of_cubes (a b : ℝ) (h1 : a + b = 4) (h2 : a * b = -5) : a^3 + b^3 = 124 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l214_21490


namespace NUMINAMATH_CALUDE_fraction_calculation_l214_21465

theorem fraction_calculation : (1 / 4 : ℚ) * (1 / 3 : ℚ) * (1 / 6 : ℚ) * 144 + (1 / 2 : ℚ) = (5 / 2 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_fraction_calculation_l214_21465


namespace NUMINAMATH_CALUDE_fraction_inequality_l214_21433

theorem fraction_inequality (a b t : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : t > 0) :
  a / b > (a + t) / (b + t) := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l214_21433


namespace NUMINAMATH_CALUDE_rectangular_field_width_l214_21469

theorem rectangular_field_width (width : ℝ) (length : ℝ) (perimeter : ℝ) : 
  length = (7 / 5) * width →
  perimeter = 240 →
  2 * length + 2 * width = perimeter →
  width = 50 := by
sorry

end NUMINAMATH_CALUDE_rectangular_field_width_l214_21469


namespace NUMINAMATH_CALUDE_city_population_l214_21478

/-- Represents the population distribution of a city -/
structure CityPopulation where
  total : ℕ
  under18 : ℕ
  between18and65 : ℕ
  over65 : ℕ
  belowPovertyLine : ℕ
  middleClass : ℕ
  wealthy : ℕ
  menUnder18 : ℕ
  womenUnder18 : ℕ

/-- Theorem stating the total population of the city given the conditions -/
theorem city_population (c : CityPopulation) : c.total = 500000 :=
  by
  have h1 : c.under18 = c.total / 4 := sorry
  have h2 : c.between18and65 = c.total * 11 / 20 := sorry
  have h3 : c.over65 = c.total / 5 := sorry
  have h4 : c.belowPovertyLine = c.total * 3 / 20 := sorry
  have h5 : c.middleClass = c.total * 13 / 20 := sorry
  have h6 : c.wealthy = c.total / 5 := sorry
  have h7 : c.menUnder18 = c.under18 * 3 / 5 := sorry
  have h8 : c.womenUnder18 = c.under18 * 2 / 5 := sorry
  have h9 : c.wealthy * 1 / 5 = 20000 := sorry
  sorry

#check city_population

end NUMINAMATH_CALUDE_city_population_l214_21478


namespace NUMINAMATH_CALUDE_triangle_arithmetic_angle_sequence_l214_21403

theorem triangle_arithmetic_angle_sequence (A B C : ℝ) : 
  A + B + C = 180 →  -- Sum of angles in a triangle is 180°
  2 * B = A + C →    -- Angles form an arithmetic sequence
  (max A (max B C) + min A (min B C) = 120) := by
sorry

end NUMINAMATH_CALUDE_triangle_arithmetic_angle_sequence_l214_21403


namespace NUMINAMATH_CALUDE_no_real_solutions_cubic_equation_l214_21477

theorem no_real_solutions_cubic_equation :
  ∀ x : ℝ, x^3 + 2*(x+1)^3 + 3*(x+2)^3 ≠ 6*(x+4)^3 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_cubic_equation_l214_21477


namespace NUMINAMATH_CALUDE_matrix_fourth_power_l214_21446

def A : Matrix (Fin 2) (Fin 2) ℤ := !![2, -1; 1, 1]

theorem matrix_fourth_power :
  A ^ 4 = !![0, -9; 9, -9] := by sorry

end NUMINAMATH_CALUDE_matrix_fourth_power_l214_21446


namespace NUMINAMATH_CALUDE_no_eulerian_path_in_picture_graph_l214_21481

/-- A graph representing the regions in the picture --/
structure PictureGraph where
  vertices : Finset (Fin 6)
  edges : Finset (Fin 6 × Fin 6)
  adjacent : (a b : Fin 6) → (a, b) ∈ edges → a ≠ b

/-- The degree of a vertex in the graph --/
def degree (G : PictureGraph) (v : Fin 6) : Nat :=
  (G.edges.filter (fun e => e.1 = v ∨ e.2 = v)).card

/-- An Eulerian path visits each edge exactly once --/
def hasEulerianPath (G : PictureGraph) : Prop :=
  ∃ path : List (Fin 6), path.length = G.edges.card + 1 ∧
    (∀ e ∈ G.edges, ∃ i, path[i]? = some e.1 ∧ path[i+1]? = some e.2)

/-- The main theorem: No Eulerian path exists in this graph --/
theorem no_eulerian_path_in_picture_graph (G : PictureGraph) 
  (h1 : ∃ v1 v2 v3 : Fin 6, v1 ≠ v2 ∧ v2 ≠ v3 ∧ v1 ≠ v3 ∧ 
    degree G v1 = 5 ∧ degree G v2 = 5 ∧ degree G v3 = 9) :
  ¬ hasEulerianPath G := by
  sorry


end NUMINAMATH_CALUDE_no_eulerian_path_in_picture_graph_l214_21481


namespace NUMINAMATH_CALUDE_checkerboard_probability_l214_21436

/-- The size of one side of the checkerboard -/
def board_size : ℕ := 9

/-- The total number of squares on the checkerboard -/
def total_squares : ℕ := board_size * board_size

/-- The number of squares on the perimeter of the checkerboard -/
def perimeter_squares : ℕ := 4 * board_size - 4

/-- The number of squares not on the perimeter of the checkerboard -/
def non_perimeter_squares : ℕ := total_squares - perimeter_squares

/-- The probability of choosing a square not on the perimeter -/
def prob_non_perimeter : ℚ := non_perimeter_squares / total_squares

theorem checkerboard_probability :
  prob_non_perimeter = 49 / 81 := by sorry

end NUMINAMATH_CALUDE_checkerboard_probability_l214_21436


namespace NUMINAMATH_CALUDE_abs_y_plus_sqrt_y_plus_two_squared_l214_21418

theorem abs_y_plus_sqrt_y_plus_two_squared (y : ℝ) (h : y > 1) :
  |y + Real.sqrt ((y + 2)^2)| = 2*y + 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_y_plus_sqrt_y_plus_two_squared_l214_21418


namespace NUMINAMATH_CALUDE_modular_inverse_of_4_mod_21_l214_21485

theorem modular_inverse_of_4_mod_21 : ∃ x : ℕ, x ≤ 20 ∧ (4 * x) % 21 = 1 :=
by
  use 16
  sorry

end NUMINAMATH_CALUDE_modular_inverse_of_4_mod_21_l214_21485


namespace NUMINAMATH_CALUDE_taehyung_has_most_points_l214_21497

def yoongi_points : ℕ := 7
def jungkook_points : ℕ := 6
def yuna_points : ℕ := 9
def yoojung_points : ℕ := 8
def taehyung_points : ℕ := 10

theorem taehyung_has_most_points :
  taehyung_points ≥ yoongi_points ∧
  taehyung_points ≥ jungkook_points ∧
  taehyung_points ≥ yuna_points ∧
  taehyung_points ≥ yoojung_points :=
by sorry

end NUMINAMATH_CALUDE_taehyung_has_most_points_l214_21497


namespace NUMINAMATH_CALUDE_two_by_one_parallelepiped_removals_l214_21448

/-- Represents a position on the net of a parallelepiped --/
structure Position :=
  (x : ℕ)
  (y : ℕ)

/-- Represents the net of a parallelepiped --/
structure ParallelepipedNet :=
  (squares : List Position)
  (width : ℕ)
  (height : ℕ)

/-- Checks if a given position is valid for removal --/
def is_valid_removal (net : ParallelepipedNet) (pos : Position) : Prop := sorry

/-- Counts the number of valid removal positions --/
def count_valid_removals (net : ParallelepipedNet) : ℕ := sorry

/-- Creates a 2x1 parallelepiped net --/
def create_2x1_net : ParallelepipedNet := sorry

theorem two_by_one_parallelepiped_removals :
  count_valid_removals (create_2x1_net) = 5 := by sorry

end NUMINAMATH_CALUDE_two_by_one_parallelepiped_removals_l214_21448


namespace NUMINAMATH_CALUDE_remainder_of_3_600_mod_17_l214_21459

theorem remainder_of_3_600_mod_17 : 3^600 % 17 = 9 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_3_600_mod_17_l214_21459


namespace NUMINAMATH_CALUDE_fifth_term_is_negative_one_l214_21437

/-- An arithmetic sequence with specific first four terms -/
def arithmetic_sequence (x y : ℚ) : ℕ → ℚ
  | 0 => x + 2*y
  | 1 => x - y
  | 2 => 2*x*y
  | 3 => x / (2*y)
  | n + 4 => arithmetic_sequence x y 3 + (n + 1) * (arithmetic_sequence x y 1 - arithmetic_sequence x y 0)

/-- The theorem stating that the fifth term of the specific arithmetic sequence is -1 -/
theorem fifth_term_is_negative_one :
  let x : ℚ := 4
  let y : ℚ := 1
  arithmetic_sequence x y 4 = -1 := by sorry

end NUMINAMATH_CALUDE_fifth_term_is_negative_one_l214_21437


namespace NUMINAMATH_CALUDE_janet_action_figures_l214_21440

def action_figure_count (initial : ℕ) (sold : ℕ) (bought : ℕ) (brother_factor : ℕ) : ℕ :=
  let remaining := initial - sold
  let after_buying := remaining + bought
  let brother_collection := after_buying * brother_factor
  after_buying + brother_collection

theorem janet_action_figures :
  action_figure_count 10 6 4 2 = 24 := by
  sorry

end NUMINAMATH_CALUDE_janet_action_figures_l214_21440


namespace NUMINAMATH_CALUDE_equal_fractions_imply_equal_values_l214_21414

theorem equal_fractions_imply_equal_values (a b c : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (h1 : (a + b) / c = (c + b) / a) 
  (h2 : (c + b) / a = (a + c) / b) : 
  a = b ∧ b = c := by
  sorry

end NUMINAMATH_CALUDE_equal_fractions_imply_equal_values_l214_21414


namespace NUMINAMATH_CALUDE_hyperbola_circle_range_l214_21464

theorem hyperbola_circle_range (a : ℝ) : 
  let P := (a > 1 ∨ a < -3)
  let Q := (-1 < a ∧ a < 3)
  (¬(P ∧ Q) ∧ ¬(¬Q)) → (-1 < a ∧ a ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_circle_range_l214_21464


namespace NUMINAMATH_CALUDE_investment_rate_proof_l214_21489

/-- Proves that the required interest rate for the remaining investment is 6.4% --/
theorem investment_rate_proof (total_investment : ℝ) (first_investment : ℝ) (second_investment : ℝ)
  (first_rate : ℝ) (second_rate : ℝ) (desired_income : ℝ)
  (h1 : total_investment = 10000)
  (h2 : first_investment = 4000)
  (h3 : second_investment = 3500)
  (h4 : first_rate = 0.05)
  (h5 : second_rate = 0.04)
  (h6 : desired_income = 500) :
  (desired_income - (first_investment * first_rate + second_investment * second_rate)) / 
  (total_investment - first_investment - second_investment) = 0.064 := by
sorry

end NUMINAMATH_CALUDE_investment_rate_proof_l214_21489


namespace NUMINAMATH_CALUDE_singing_competition_result_l214_21408

def singing_competition (total_contestants : ℕ) 
                        (female_solo_percent : ℚ) 
                        (male_solo_percent : ℚ) 
                        (group_percent : ℚ) 
                        (male_young_percent : ℚ) 
                        (female_young_percent : ℚ) : Prop :=
  let female_solo := ⌊(female_solo_percent * total_contestants : ℚ)⌋
  let male_solo := ⌊(male_solo_percent * total_contestants : ℚ)⌋
  let male_young := ⌊(male_young_percent * male_solo : ℚ)⌋
  let female_young := ⌊(female_young_percent * female_solo : ℚ)⌋
  total_contestants = 18 ∧
  female_solo_percent = 35/100 ∧
  male_solo_percent = 25/100 ∧
  group_percent = 40/100 ∧
  male_young_percent = 30/100 ∧
  female_young_percent = 20/100 ∧
  male_young = 1 ∧
  female_young = 1

theorem singing_competition_result : 
  singing_competition 18 (35/100) (25/100) (40/100) (30/100) (20/100) := by
  sorry

end NUMINAMATH_CALUDE_singing_competition_result_l214_21408


namespace NUMINAMATH_CALUDE_misery_ratio_bound_l214_21456

/-- Represents a room with its total load -/
structure Room where
  load : ℝ
  load_positive : load > 0

/-- Represents a student with their download request -/
structure Student where
  bits : ℝ
  bits_positive : bits > 0

/-- Calculates the displeasure of a student in a given room -/
def displeasure (s : Student) (r : Room) : ℝ := s.bits * r.load

/-- Calculates the total misery for a given configuration -/
def misery (students : List Student) (rooms : List Room) (assignment : Student → Room) : ℝ :=
  (students.map (fun s => displeasure s (assignment s))).sum

/-- Defines a balanced configuration -/
def is_balanced (students : List Student) (rooms : List Room) (assignment : Student → Room) : Prop :=
  ∀ s : Student, ∀ r : Room, displeasure s (assignment s) ≤ displeasure s r

theorem misery_ratio_bound 
  (students : List Student) 
  (rooms : List Room) 
  (balanced_assignment : Student → Room)
  (other_assignment : Student → Room)
  (h_balanced : is_balanced students rooms balanced_assignment) :
  let M1 := misery students rooms balanced_assignment
  let M2 := misery students rooms other_assignment
  M1 / M2 ≤ 9 / 8 := by
  sorry

end NUMINAMATH_CALUDE_misery_ratio_bound_l214_21456


namespace NUMINAMATH_CALUDE_max_length_sequence_l214_21482

def sequence_term (n : ℕ) (x : ℕ) : ℤ :=
  match n with
  | 0 => 5000
  | 1 => x
  | n + 2 => sequence_term n x - sequence_term (n + 1) x

def is_positive (n : ℤ) : Prop := n > 0

theorem max_length_sequence (x : ℕ) : 
  (∀ n : ℕ, n < 11 → is_positive (sequence_term n x)) ∧ 
  ¬(is_positive (sequence_term 11 x)) ↔ 
  x = 3089 :=
sorry

end NUMINAMATH_CALUDE_max_length_sequence_l214_21482


namespace NUMINAMATH_CALUDE_sum_of_first_seven_primes_mod_eighth_prime_l214_21468

def first_eight_primes : List Nat := [2, 3, 5, 7, 11, 13, 17, 19]

theorem sum_of_first_seven_primes_mod_eighth_prime :
  (List.sum (List.take 7 first_eight_primes)) % (List.get! first_eight_primes 7) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_first_seven_primes_mod_eighth_prime_l214_21468


namespace NUMINAMATH_CALUDE_polar_to_rectangular_l214_21498

theorem polar_to_rectangular :
  let r : ℝ := 3 * Real.sqrt 2
  let θ : ℝ := π / 4
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  (x = 3 ∧ y = 3) := by sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_l214_21498


namespace NUMINAMATH_CALUDE_find_k_l214_21488

theorem find_k (f g : ℝ → ℝ) (k : ℝ) : 
  (∀ x, f x = 7 * x^3 - 1/x + 5) →
  (∀ x, g x = x^3 - k) →
  f 3 - g 3 = 5 →
  k = -485/3 := by sorry

end NUMINAMATH_CALUDE_find_k_l214_21488


namespace NUMINAMATH_CALUDE_sector_max_area_l214_21450

/-- 
Given a sector with circumference c, this theorem states that:
1. The maximum area of the sector is c^2/16
2. The maximum area occurs when the arc length is c/2
-/
theorem sector_max_area (c : ℝ) (h : c > 0) :
  ∃ (max_area arc_length : ℝ),
    max_area = c^2 / 16 ∧
    arc_length = c / 2 ∧
    ∀ (area : ℝ), area ≤ max_area :=
by sorry

end NUMINAMATH_CALUDE_sector_max_area_l214_21450


namespace NUMINAMATH_CALUDE_intersection_condition_m_values_l214_21451

theorem intersection_condition_m_values (m : ℝ) : 
  let A : Set ℝ := {x | x^2 - x - 6 = 0}
  let B : Set ℝ := {x | x * m - 1 = 0}
  (A ∩ B = B) ↔ (m = 0 ∨ m = -1/2 ∨ m = 1/3) := by
  sorry

end NUMINAMATH_CALUDE_intersection_condition_m_values_l214_21451


namespace NUMINAMATH_CALUDE_f_three_l214_21420

/-- A function satisfying the given property -/
def f (x : ℝ) : ℝ := sorry

/-- The property of the function f -/
axiom f_property (x y : ℝ) : f (x + y) = f x + f y + x * y

/-- The given condition that f(1) = 1 -/
axiom f_one : f 1 = 1

/-- Theorem stating that f(3) = 6 -/
theorem f_three : f 3 = 6 := by sorry

end NUMINAMATH_CALUDE_f_three_l214_21420


namespace NUMINAMATH_CALUDE_floor_product_theorem_l214_21453

theorem floor_product_theorem :
  ∃ (x : ℝ), x > 0 ∧ (↑⌊x⌋ : ℝ) * x = 90 → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_floor_product_theorem_l214_21453


namespace NUMINAMATH_CALUDE_leading_coeff_of_polynomial_l214_21457

/-- Given a polynomial f such that f(x + 1) - f(x) = 8x^2 + 6x + 4 for all real x,
    the leading coefficient of f is 8/3 -/
theorem leading_coeff_of_polynomial (f : ℝ → ℝ) :
  (∀ x : ℝ, f (x + 1) - f x = 8 * x^2 + 6 * x + 4) →
  ∃ (a b c d : ℝ), (∀ x : ℝ, f x = (8/3) * x^3 + a * x^2 + b * x + c) ∧ a ≠ (8/3) :=
by sorry

end NUMINAMATH_CALUDE_leading_coeff_of_polynomial_l214_21457


namespace NUMINAMATH_CALUDE_melanie_selling_four_gumballs_l214_21491

/-- The number of gumballs Melanie is selling -/
def num_gumballs : ℕ := 32 / 8

/-- The price of each gumball in cents -/
def price_per_gumball : ℕ := 8

/-- The total amount Melanie gets from selling gumballs in cents -/
def total_amount : ℕ := 32

/-- Theorem stating that Melanie is selling 4 gumballs -/
theorem melanie_selling_four_gumballs :
  num_gumballs = 4 :=
by sorry

end NUMINAMATH_CALUDE_melanie_selling_four_gumballs_l214_21491


namespace NUMINAMATH_CALUDE_fourth_plus_fifth_sum_l214_21406

/-- A geometric sequence with a negative common ratio satisfying certain conditions -/
structure GeometricSequence where
  a : ℕ → ℝ
  q : ℝ
  q_neg : q < 0
  second_term : a 2 = 1 - a 1
  fourth_term : a 4 = 4 - a 3
  geom_seq : ∀ n : ℕ, a (n + 1) = a n * q

/-- The sum of the fourth and fifth terms of the geometric sequence is -8 -/
theorem fourth_plus_fifth_sum (seq : GeometricSequence) : seq.a 4 + seq.a 5 = -8 := by
  sorry

end NUMINAMATH_CALUDE_fourth_plus_fifth_sum_l214_21406


namespace NUMINAMATH_CALUDE_root_ratio_sum_l214_21400

theorem root_ratio_sum (k₁ k₂ : ℝ) : 
  (∃ a b : ℝ, k₁ * (a^2 - 2*a) + 3*a + 7 = 0 ∧ 
              k₁ * (b^2 - 2*b) + 3*b + 7 = 0 ∧ 
              a/b + b/a = 9/7) →
  (∃ a b : ℝ, k₂ * (a^2 - 2*a) + 3*a + 7 = 0 ∧ 
              k₂ * (b^2 - 2*b) + 3*b + 7 = 0 ∧ 
              a/b + b/a = 9/7) →
  k₁/k₂ + k₂/k₁ = 64/9 := by
sorry

end NUMINAMATH_CALUDE_root_ratio_sum_l214_21400


namespace NUMINAMATH_CALUDE_unique_triangle_arrangement_l214_21447

/-- Represents the arrangement of numbers in the triangle --/
structure TriangleArrangement where
  A : Nat
  B : Nat
  C : Nat
  D : Nat

/-- Checks if the given arrangement is valid according to the problem conditions --/
def is_valid_arrangement (arr : TriangleArrangement) : Prop :=
  -- All numbers are between 6 and 9
  (arr.A ≥ 6 ∧ arr.A ≤ 9) ∧
  (arr.B ≥ 6 ∧ arr.B ≤ 9) ∧
  (arr.C ≥ 6 ∧ arr.C ≤ 9) ∧
  (arr.D ≥ 6 ∧ arr.D ≤ 9) ∧
  -- All numbers are different
  arr.A ≠ arr.B ∧ arr.A ≠ arr.C ∧ arr.A ≠ arr.D ∧
  arr.B ≠ arr.C ∧ arr.B ≠ arr.D ∧
  arr.C ≠ arr.D ∧
  -- Sum of numbers on each side is equal
  arr.A + arr.C + 3 + 4 = 5 + arr.D + 2 + 4 ∧
  5 + 1 + arr.B + arr.A = 5 + arr.D + 2 + 4 ∧
  arr.A + arr.C + 3 + 4 = 5 + 1 + arr.B + arr.A

theorem unique_triangle_arrangement :
  ∃! arr : TriangleArrangement, is_valid_arrangement arr ∧
    arr.A = 6 ∧ arr.B = 8 ∧ arr.C = 7 ∧ arr.D = 9 :=
by sorry

end NUMINAMATH_CALUDE_unique_triangle_arrangement_l214_21447


namespace NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l214_21493

/-- Given a geometric sequence with first term 243 and eighth term 32,
    the sixth term of the sequence is 1. -/
theorem geometric_sequence_sixth_term :
  ∀ (a : ℝ) (r : ℝ),
    a = 243 →
    a * r^7 = 32 →
    a * r^5 = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l214_21493


namespace NUMINAMATH_CALUDE_cube_diff_of_squares_l214_21426

theorem cube_diff_of_squares (a : ℕ+) : ∃ x y : ℤ, x^2 - y^2 = (a : ℤ)^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_diff_of_squares_l214_21426


namespace NUMINAMATH_CALUDE_max_four_digit_product_of_primes_l214_21495

def is_prime (p : ℕ) : Prop := Nat.Prime p

theorem max_four_digit_product_of_primes :
  ∃ (n x y : ℕ),
    n = x * y * (10 * x + y) ∧
    is_prime x ∧
    is_prime y ∧
    is_prime (10 * x + y) ∧
    x < 5 ∧
    y < 5 ∧
    x ≠ y ∧
    1000 ≤ n ∧
    n < 10000 ∧
    (∀ (m x' y' : ℕ),
      m = x' * y' * (10 * x' + y') →
      is_prime x' →
      is_prime y' →
      is_prime (10 * x' + y') →
      x' < 5 →
      y' < 5 →
      x' ≠ y' →
      1000 ≤ m →
      m < 10000 →
      m ≤ n) ∧
    n = 138 :=
by sorry

end NUMINAMATH_CALUDE_max_four_digit_product_of_primes_l214_21495


namespace NUMINAMATH_CALUDE_compare_powers_l214_21421

theorem compare_powers : 5^333 < 3^555 ∧ 3^555 < 4^444 := by
  sorry

end NUMINAMATH_CALUDE_compare_powers_l214_21421


namespace NUMINAMATH_CALUDE_return_trip_duration_l214_21417

def time_to_park : ℕ := 20 + 10

def return_trip_factor : ℕ := 3

theorem return_trip_duration : 
  return_trip_factor * time_to_park = 90 :=
by sorry

end NUMINAMATH_CALUDE_return_trip_duration_l214_21417


namespace NUMINAMATH_CALUDE_binary_conversion_theorem_l214_21409

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (λ acc (i, bit) => acc + if bit then 2^i else 0) 0

def decimal_to_base7 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) :=
    if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
  aux n []

theorem binary_conversion_theorem :
  let binary : List Bool := [true, false, true, true, false, true]
  let decimal : ℕ := binary_to_decimal binary
  let base7 : List ℕ := decimal_to_base7 decimal
  decimal = 45 ∧ base7 = [6, 3] := by sorry

end NUMINAMATH_CALUDE_binary_conversion_theorem_l214_21409


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l214_21410

theorem quadratic_coefficient (a b c : ℝ) (h1 : a ≠ 0) : 
  let f := fun x => a * x^2 + b * x + c
  let Δ := b^2 - 4*a*c
  (∃ x y, x ≠ y ∧ f x = 0 ∧ f y = 0 ∧ (x - y)^2 = 1) →
  Δ = 1/4 →
  a = -1/2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l214_21410


namespace NUMINAMATH_CALUDE_preimage_of_4_neg2_l214_21429

def f (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 + p.2, p.1 - p.2)

theorem preimage_of_4_neg2 :
  ∃ (p : ℝ × ℝ), f p = (4, -2) ∧ p = (1, 3) :=
sorry

end NUMINAMATH_CALUDE_preimage_of_4_neg2_l214_21429


namespace NUMINAMATH_CALUDE_oliver_games_l214_21452

def number_of_games (initial_money : ℕ) (money_spent : ℕ) (game_cost : ℕ) : ℕ :=
  (initial_money - money_spent) / game_cost

theorem oliver_games : number_of_games 35 7 4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_oliver_games_l214_21452


namespace NUMINAMATH_CALUDE_min_value_fraction_l214_21404

theorem min_value_fraction (a b c : ℕ) 
  (ha : 1 ≤ a ∧ a ≤ 2011) 
  (hb : 1 ≤ b ∧ b ≤ 2011) 
  (hc : 1 ≤ c ∧ c ≤ 2011) : 
  (a * b + c : ℚ) / (a + b + c) ≥ 2/3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_fraction_l214_21404
