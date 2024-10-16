import Mathlib

namespace NUMINAMATH_CALUDE_triangle_theorem_l3459_345946

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_theorem (t : Triangle) 
  (h1 : Real.tan t.C / Real.tan t.B = -t.c / (2 * t.a + t.c))
  (h2 : t.b = 2 * Real.sqrt 3)
  (h3 : t.a + t.c = 4) :
  t.B = 2 * Real.pi / 3 ∧ 
  (1/2 * t.a * t.c * Real.sin t.B : ℝ) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_theorem_l3459_345946


namespace NUMINAMATH_CALUDE_novel_writing_speed_l3459_345929

/-- Given a novel with 40,000 words written in 80 hours, 
    the average number of words written per hour is 500. -/
theorem novel_writing_speed (total_words : ℕ) (total_hours : ℕ) 
  (h1 : total_words = 40000) (h2 : total_hours = 80) :
  total_words / total_hours = 500 := by
  sorry

#check novel_writing_speed

end NUMINAMATH_CALUDE_novel_writing_speed_l3459_345929


namespace NUMINAMATH_CALUDE_product_equals_square_l3459_345958

theorem product_equals_square : 100 * 29.98 * 2.998 * 1000 = (2998 : ℝ)^2 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_square_l3459_345958


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l3459_345926

theorem sufficient_but_not_necessary : 
  (∀ x : ℝ, x^2 - 2*x < 0 → abs (x - 2) < 2) ∧ 
  (∃ x : ℝ, abs (x - 2) < 2 ∧ ¬(x^2 - 2*x < 0)) := by
sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l3459_345926


namespace NUMINAMATH_CALUDE_tangent_points_parallel_to_line_tangent_points_on_curve_unique_tangent_points_l3459_345980

-- Define the function f(x) = x^3 + x - 2
def f (x : ℝ) : ℝ := x^3 + x - 2

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

theorem tangent_points_parallel_to_line (x : ℝ) :
  (f' x = 4) ↔ (x = 1 ∨ x = -1) :=
sorry

theorem tangent_points_on_curve :
  f 1 = 0 ∧ f (-1) = -4 :=
sorry

theorem unique_tangent_points :
  ∀ x : ℝ, f' x = 4 → (x = 1 ∨ x = -1) :=
sorry

end NUMINAMATH_CALUDE_tangent_points_parallel_to_line_tangent_points_on_curve_unique_tangent_points_l3459_345980


namespace NUMINAMATH_CALUDE_zero_in_interval_l3459_345927

noncomputable def f (x : ℝ) : ℝ := 6 / x - Real.log x / Real.log 2

theorem zero_in_interval :
  ∃ c ∈ Set.Ioo 2 4, f c = 0 :=
sorry

end NUMINAMATH_CALUDE_zero_in_interval_l3459_345927


namespace NUMINAMATH_CALUDE_pasta_sauce_free_percentage_l3459_345934

/-- Given a pasta dish weighing 200 grams with 50 grams of sauce,
    prove that 75% of the dish is sauce-free. -/
theorem pasta_sauce_free_percentage
  (total_weight : ℝ)
  (sauce_weight : ℝ)
  (h_total : total_weight = 200)
  (h_sauce : sauce_weight = 50) :
  (total_weight - sauce_weight) / total_weight * 100 = 75 := by
  sorry

end NUMINAMATH_CALUDE_pasta_sauce_free_percentage_l3459_345934


namespace NUMINAMATH_CALUDE_inscribed_circle_tangent_triangle_area_l3459_345931

/-- Given a right triangle with hypotenuse c, area T, and an inscribed circle of radius ρ,
    the area of the triangle formed by the points where the inscribed circle touches the sides
    of the right triangle is equal to (ρ/c) * T. -/
theorem inscribed_circle_tangent_triangle_area
  (c T ρ : ℝ)
  (h_positive : c > 0 ∧ T > 0 ∧ ρ > 0)
  (h_right_triangle : ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a^2 + b^2 = c^2)
  (h_area : T = (a + b + c) * ρ / 2)
  (h_inscribed : ρ = T / (a + b + c)) :
  (ρ / c) * T = (area_of_tangent_triangle : ℝ) :=
sorry

end NUMINAMATH_CALUDE_inscribed_circle_tangent_triangle_area_l3459_345931


namespace NUMINAMATH_CALUDE_buckingham_visitors_theorem_l3459_345967

/-- Represents the number of visitors to Buckingham Palace -/
structure BuckinghamVisitors where
  total_85_days : ℕ
  previous_day : ℕ

/-- Calculates the number of visitors on a specific day -/
def visitors_on_day (bv : BuckinghamVisitors) : ℕ :=
  bv.total_85_days - bv.previous_day

/-- Theorem statement for the Buckingham Palace visitor calculation -/
theorem buckingham_visitors_theorem (bv : BuckinghamVisitors) 
  (h1 : bv.total_85_days = 829)
  (h2 : bv.previous_day = 45) :
  visitors_on_day bv = 784 := by
  sorry

#eval visitors_on_day { total_85_days := 829, previous_day := 45 }

end NUMINAMATH_CALUDE_buckingham_visitors_theorem_l3459_345967


namespace NUMINAMATH_CALUDE_chess_game_duration_l3459_345940

theorem chess_game_duration (game_hours : ℕ) (game_minutes : ℕ) (analysis_minutes : ℕ) : 
  game_hours = 20 → game_minutes = 15 → analysis_minutes = 22 →
  game_hours * 60 + game_minutes + analysis_minutes = 1237 := by
  sorry

end NUMINAMATH_CALUDE_chess_game_duration_l3459_345940


namespace NUMINAMATH_CALUDE_cheryl_same_color_probability_l3459_345986

def total_marbles : ℕ := 12
def marbles_per_color : ℕ := 3
def num_colors : ℕ := 4
def marbles_taken_each_turn : ℕ := 3

def probability_cheryl_same_color : ℚ := 2 / 55

theorem cheryl_same_color_probability :
  let total_outcomes := Nat.choose total_marbles marbles_taken_each_turn *
                        Nat.choose (total_marbles - marbles_taken_each_turn) marbles_taken_each_turn *
                        Nat.choose (total_marbles - 2 * marbles_taken_each_turn) marbles_taken_each_turn
  let favorable_outcomes := num_colors * Nat.choose (total_marbles - marbles_taken_each_turn) marbles_taken_each_turn *
                            Nat.choose (total_marbles - 2 * marbles_taken_each_turn) marbles_taken_each_turn
  (favorable_outcomes : ℚ) / total_outcomes = probability_cheryl_same_color := by
  sorry

end NUMINAMATH_CALUDE_cheryl_same_color_probability_l3459_345986


namespace NUMINAMATH_CALUDE_barrel_division_l3459_345985

theorem barrel_division (length width height : ℝ) (volume_small_barrel : ℝ) : 
  length = 6.4 ∧ width = 9 ∧ height = 5.2 ∧ volume_small_barrel = 1 →
  ⌈length * width * height / volume_small_barrel⌉ = 300 := by
  sorry

end NUMINAMATH_CALUDE_barrel_division_l3459_345985


namespace NUMINAMATH_CALUDE_complex_quadrant_l3459_345974

theorem complex_quadrant (z : ℂ) (h : (1 + 2*I)/z = 1 - I) : 
  z.re < 0 ∧ z.im > 0 := by
sorry

end NUMINAMATH_CALUDE_complex_quadrant_l3459_345974


namespace NUMINAMATH_CALUDE_stating_tom_initial_investment_l3459_345909

/-- Represents the profit sharing scenario between Tom and Jose -/
structure ProfitSharing where
  tom_investment : ℕ  -- Tom's initial investment
  jose_investment : ℕ := 45000  -- Jose's investment
  total_profit : ℕ := 72000  -- Total profit after one year
  jose_profit : ℕ := 40000  -- Jose's share of the profit
  tom_months : ℕ := 12  -- Months Tom was in business
  jose_months : ℕ := 10  -- Months Jose was in business

/-- 
Theorem stating that given the conditions of the profit sharing scenario, 
Tom's initial investment was 30000.
-/
theorem tom_initial_investment (ps : ProfitSharing) : ps.tom_investment = 30000 := by
  sorry

#check tom_initial_investment

end NUMINAMATH_CALUDE_stating_tom_initial_investment_l3459_345909


namespace NUMINAMATH_CALUDE_only_set_C_not_in_proportion_l3459_345933

def is_in_proportion (a b c d : ℝ) : Prop := a * d = b * c

theorem only_set_C_not_in_proportion :
  (is_in_proportion 4 8 5 10) ∧
  (is_in_proportion 2 (2 * Real.sqrt 5) (Real.sqrt 5) 5) ∧
  ¬(is_in_proportion 1 2 3 4) ∧
  (is_in_proportion 1 2 2 4) :=
by sorry

end NUMINAMATH_CALUDE_only_set_C_not_in_proportion_l3459_345933


namespace NUMINAMATH_CALUDE_f_domain_f_property_f_one_eq_zero_l3459_345989

/-- A function f with the given properties -/
def f : ℝ → ℝ :=
  sorry

theorem f_domain (x : ℝ) : x ≠ 0 → f x ≠ 0 :=
  sorry

theorem f_property (x₁ x₂ : ℝ) (h₁ : x₁ ≠ 0) (h₂ : x₂ ≠ 0) :
  f (x₁ * x₂) = f x₁ + f x₂ :=
  sorry

theorem f_one_eq_zero : f 1 = 0 :=
  sorry

end NUMINAMATH_CALUDE_f_domain_f_property_f_one_eq_zero_l3459_345989


namespace NUMINAMATH_CALUDE_platform_length_l3459_345948

/-- The length of a platform given a goods train's speed, length, and time to cross the platform. -/
theorem platform_length (train_speed : ℝ) (train_length : ℝ) (crossing_time : ℝ) : 
  train_speed = 72 → 
  train_length = 280.0416 → 
  crossing_time = 26 → 
  (train_speed * 1000 / 3600 * crossing_time) - train_length = 239.9584 := by
  sorry

end NUMINAMATH_CALUDE_platform_length_l3459_345948


namespace NUMINAMATH_CALUDE_expression_factorization_l3459_345988

theorem expression_factorization (x : ℝ) : 
  (9 * x^4 - 138 * x^3 + 49 * x^2) - (-3 * x^4 + 27 * x^3 - 14 * x^2) = 
  3 * x^2 * (4 * x - 3) * (x - 7) := by
sorry

end NUMINAMATH_CALUDE_expression_factorization_l3459_345988


namespace NUMINAMATH_CALUDE_sample_grade_10_is_15_l3459_345922

/-- Represents a school with stratified sampling -/
structure School where
  total_students : ℕ
  grade_10_students : ℕ
  sample_size : ℕ

/-- Calculates the number of Grade 10 students to be sampled -/
def sample_grade_10 (school : School) : ℕ :=
  (school.sample_size * school.grade_10_students) / school.total_students

/-- Theorem stating that for the given school parameters, 
    the number of Grade 10 students to be sampled is 15 -/
theorem sample_grade_10_is_15 (school : School) 
  (h1 : school.total_students = 2000)
  (h2 : school.grade_10_students = 600)
  (h3 : school.sample_size = 50) :
  sample_grade_10 school = 15 := by
  sorry

#eval sample_grade_10 ⟨2000, 600, 50⟩

end NUMINAMATH_CALUDE_sample_grade_10_is_15_l3459_345922


namespace NUMINAMATH_CALUDE_min_x_plus_y_min_value_is_9_4_min_achieved_l3459_345902

-- Define the optimization problem
theorem min_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 4 / y + 1 / x = 4) :
  ∀ x' y' : ℝ, x' > 0 → y' > 0 → 4 / y' + 1 / x' = 4 → x + y ≤ x' + y' :=
by sorry

-- State the minimum value
theorem min_value_is_9_4 (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 4 / y + 1 / x = 4) :
  x + y ≥ 9 / 4 :=
by sorry

-- Prove the minimum is achieved
theorem min_achieved (ε : ℝ) (hε : ε > 0) :
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 4 / y + 1 / x = 4 ∧ x + y < 9 / 4 + ε :=
by sorry

end NUMINAMATH_CALUDE_min_x_plus_y_min_value_is_9_4_min_achieved_l3459_345902


namespace NUMINAMATH_CALUDE_product_9_to_11_l3459_345930

theorem product_9_to_11 : (List.range 3).foldl (·*·) 1 * 9 = 990 := by
  sorry

end NUMINAMATH_CALUDE_product_9_to_11_l3459_345930


namespace NUMINAMATH_CALUDE_min_value_of_f_l3459_345905

-- Define the function f(x)
def f (x : ℝ) : ℝ := 27 * x - x^3

-- State the theorem
theorem min_value_of_f :
  ∃ (x : ℝ), x ∈ Set.Icc (-4 : ℝ) 2 ∧
  (∀ y ∈ Set.Icc (-4 : ℝ) 2, f y ≥ f x) ∧
  f x = -54 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l3459_345905


namespace NUMINAMATH_CALUDE_function_value_plus_derivative_l3459_345965

open Real

/-- Given a differentiable function f : ℝ → ℝ satisfying f x = 2 * x * f.deriv 1 + log x for all x > 0,
    prove that f 1 + f.deriv 1 = -3 -/
theorem function_value_plus_derivative (f : ℝ → ℝ) (hf : Differentiable ℝ f)
    (h : ∀ x > 0, f x = 2 * x * (deriv f 1) + log x) :
  f 1 + deriv f 1 = -3 := by
  sorry

end NUMINAMATH_CALUDE_function_value_plus_derivative_l3459_345965


namespace NUMINAMATH_CALUDE_sum_of_prime_factors_2_pow_10_minus_1_l3459_345912

theorem sum_of_prime_factors_2_pow_10_minus_1 :
  ∃ (p q r : Nat), Prime p ∧ Prime q ∧ Prime r ∧
  p ≠ q ∧ p ≠ r ∧ q ≠ r ∧
  (2^10 - 1) % p = 0 ∧ (2^10 - 1) % q = 0 ∧ (2^10 - 1) % r = 0 ∧
  p + q + r = 45 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_prime_factors_2_pow_10_minus_1_l3459_345912


namespace NUMINAMATH_CALUDE_jaces_debt_jaces_debt_value_l3459_345998

theorem jaces_debt (earned : ℝ) (gave_away_cents : ℕ) (current_balance : ℝ) : ℝ :=
  let gave_away : ℝ := (gave_away_cents : ℝ) / 100
  let debt : ℝ := earned - (current_balance + gave_away)
  debt

theorem jaces_debt_value : jaces_debt 1000 358 642 = 354.42 := by sorry

end NUMINAMATH_CALUDE_jaces_debt_jaces_debt_value_l3459_345998


namespace NUMINAMATH_CALUDE_children_toothpaste_sales_amount_l3459_345921

/-- Calculates the total sales amount for children's toothpaste. -/
def total_sales_amount (num_boxes : ℕ) (packs_per_box : ℕ) (price_per_pack : ℕ) : ℕ :=
  num_boxes * packs_per_box * price_per_pack

/-- Proves that the total sales amount for the given conditions is 1200 yuan. -/
theorem children_toothpaste_sales_amount :
  total_sales_amount 12 25 4 = 1200 := by
  sorry

end NUMINAMATH_CALUDE_children_toothpaste_sales_amount_l3459_345921


namespace NUMINAMATH_CALUDE_floor_sqrt_equation_solutions_l3459_345995

theorem floor_sqrt_equation_solutions : 
  ∃! (S : Finset ℕ), 
    (∀ n ∈ S, n > 0 ∧ (n + 1000) / 70 = ⌊Real.sqrt n⌋) ∧ 
    Finset.card S = 6 := by sorry

end NUMINAMATH_CALUDE_floor_sqrt_equation_solutions_l3459_345995


namespace NUMINAMATH_CALUDE_polynomial_symmetry_l3459_345954

/-- Given a polynomial function g(x) = ax^5 + bx^3 + cx - 3 where g(-5) = 3, prove that g(5) = -9 -/
theorem polynomial_symmetry (a b c : ℝ) :
  let g : ℝ → ℝ := λ x => a * x^5 + b * x^3 + c * x - 3
  g (-5) = 3 → g 5 = -9 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_symmetry_l3459_345954


namespace NUMINAMATH_CALUDE_complex_argument_cube_l3459_345916

theorem complex_argument_cube (z₁ z₂ : ℂ) 
  (h1 : Complex.abs z₁ = 3)
  (h2 : Complex.abs z₂ = 5)
  (h3 : Complex.abs (z₁ + z₂) = 7) :
  Complex.arg ((z₂ / z₁) ^ 3) = π := by sorry

end NUMINAMATH_CALUDE_complex_argument_cube_l3459_345916


namespace NUMINAMATH_CALUDE_complex_fraction_power_l3459_345939

theorem complex_fraction_power (i : ℂ) (h : i^2 = -1) :
  ((1 + i) / (1 - i))^2006 = -1 := by sorry

end NUMINAMATH_CALUDE_complex_fraction_power_l3459_345939


namespace NUMINAMATH_CALUDE_bishopArrangements_isPerfectSquare_l3459_345978

/-- The size of the chessboard -/
def boardSize : ℕ := 8

/-- The number of squares of one color on the board -/
def squaresPerColor : ℕ := boardSize * boardSize / 2

/-- The maximum number of non-threatening bishops on squares of one color -/
def maxBishopsPerColor : ℕ := boardSize

/-- The number of ways to arrange the maximum number of non-threatening bishops on an 8x8 chessboard -/
def totalArrangements : ℕ := (Nat.choose squaresPerColor maxBishopsPerColor) ^ 2

/-- Theorem stating that the number of arrangements is a perfect square -/
theorem bishopArrangements_isPerfectSquare : 
  ∃ n : ℕ, totalArrangements = n ^ 2 := by
sorry

end NUMINAMATH_CALUDE_bishopArrangements_isPerfectSquare_l3459_345978


namespace NUMINAMATH_CALUDE_perpendicular_foot_coordinates_l3459_345983

/-- Given a point P(1, √2, √3) in a 3-D Cartesian coordinate system and a perpendicular line PQ 
    drawn from P to the plane xOy with Q as the foot of the perpendicular, 
    prove that the coordinates of point Q are (1, √2, 0). -/
theorem perpendicular_foot_coordinates :
  let P : ℝ × ℝ × ℝ := (1, Real.sqrt 2, Real.sqrt 3)
  let xOy_plane : Set (ℝ × ℝ × ℝ) := {p | p.2.2 = 0}
  ∃ Q : ℝ × ℝ × ℝ, Q ∈ xOy_plane ∧ 
    (Q.1 = P.1 ∧ Q.2.1 = P.2.1 ∧ Q.2.2 = 0) ∧
    (∀ R ∈ xOy_plane, (P.1 - R.1)^2 + (P.2.1 - R.2.1)^2 + (P.2.2 - R.2.2)^2 ≥
                      (P.1 - Q.1)^2 + (P.2.1 - Q.2.1)^2 + (P.2.2 - Q.2.2)^2) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_foot_coordinates_l3459_345983


namespace NUMINAMATH_CALUDE_total_chair_cost_l3459_345959

/-- Represents the cost calculation for chairs in a room -/
structure RoomChairs where
  count : Nat
  price : Nat

/-- Calculates the total cost for a set of room chairs -/
def totalCost (rc : RoomChairs) : Nat :=
  rc.count * rc.price

/-- Theorem: The total cost of chairs for the entire house is $2045 -/
theorem total_chair_cost (livingRoom kitchen diningRoom patio : RoomChairs)
    (h1 : livingRoom = ⟨3, 75⟩)
    (h2 : kitchen = ⟨6, 50⟩)
    (h3 : diningRoom = ⟨8, 100⟩)
    (h4 : patio = ⟨12, 60⟩) :
    totalCost livingRoom + totalCost kitchen + totalCost diningRoom + totalCost patio = 2045 := by
  sorry

#eval totalCost ⟨3, 75⟩ + totalCost ⟨6, 50⟩ + totalCost ⟨8, 100⟩ + totalCost ⟨12, 60⟩

end NUMINAMATH_CALUDE_total_chair_cost_l3459_345959


namespace NUMINAMATH_CALUDE_fraction_simplification_l3459_345992

theorem fraction_simplification (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (a^2 * b^2) / (b/a)^2 = a^4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3459_345992


namespace NUMINAMATH_CALUDE_factorization_difference_l3459_345932

theorem factorization_difference (y : ℝ) (a b : ℤ) : 
  3 * y^2 - y - 24 = (3*y + a) * (y + b) → a - b = 11 := by
sorry

end NUMINAMATH_CALUDE_factorization_difference_l3459_345932


namespace NUMINAMATH_CALUDE_exist_three_numbers_not_exist_four_numbers_l3459_345973

/-- A function that checks if a number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

/-- Theorem stating the existence of three different natural numbers satisfying the condition -/
theorem exist_three_numbers :
  ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    is_perfect_square (a * b + 10) ∧
    is_perfect_square (a * c + 10) ∧
    is_perfect_square (b * c + 10) :=
sorry

/-- Theorem stating the non-existence of four different natural numbers satisfying the condition -/
theorem not_exist_four_numbers :
  ¬∃ a b c d : ℕ, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    is_perfect_square (a * b + 10) ∧
    is_perfect_square (a * c + 10) ∧
    is_perfect_square (a * d + 10) ∧
    is_perfect_square (b * c + 10) ∧
    is_perfect_square (b * d + 10) ∧
    is_perfect_square (c * d + 10) :=
sorry

end NUMINAMATH_CALUDE_exist_three_numbers_not_exist_four_numbers_l3459_345973


namespace NUMINAMATH_CALUDE_min_blocking_tiles_18x8_l3459_345996

/-- Represents an L-shaped tile that covers exactly 3 squares --/
structure LTile :=
  (covers : Nat)

/-- Represents a chessboard --/
structure Chessboard :=
  (rows : Nat)
  (cols : Nat)

/-- Calculates the total number of squares on the chessboard --/
def totalSquares (board : Chessboard) : Nat :=
  board.rows * board.cols

/-- Defines the minimum number of L-tiles needed to block further placement --/
def minBlockingTiles (board : Chessboard) (tile : LTile) : Nat :=
  11

/-- Main theorem: The minimum number of L-tiles to block further placement on an 18x8 board is 11 --/
theorem min_blocking_tiles_18x8 :
  let board : Chessboard := ⟨18, 8⟩
  let tile : LTile := ⟨3⟩
  minBlockingTiles board tile = 11 := by
  sorry

end NUMINAMATH_CALUDE_min_blocking_tiles_18x8_l3459_345996


namespace NUMINAMATH_CALUDE_f_magnitude_relationship_l3459_345972

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
axiom f_even : ∀ x, f x = f (-x)
axiom f_decreasing : ∀ x₁ x₂, x₁ ∈ Set.Ici (0 : ℝ) → x₂ ∈ Set.Ici (0 : ℝ) → x₁ ≠ x₂ → 
  (x₁ - x₂) * (f x₁ - f x₂) < 0

-- State the theorem to be proved
theorem f_magnitude_relationship : f 0 > f (-2) ∧ f (-2) > f 3 :=
sorry

end NUMINAMATH_CALUDE_f_magnitude_relationship_l3459_345972


namespace NUMINAMATH_CALUDE_quadratic_root_implies_a_l3459_345957

theorem quadratic_root_implies_a (a : ℝ) : 
  let S := {x : ℝ | x^2 + 2*x + a = 0}
  (-1 : ℝ) ∈ S → a = 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_a_l3459_345957


namespace NUMINAMATH_CALUDE_first_half_speed_l3459_345919

/-- Proves that given a 60-mile trip where the speed increases by 16 mph halfway through,
    and the average speed for the entire trip is 30 mph,
    the average speed during the first half of the trip is 24 mph. -/
theorem first_half_speed (v : ℝ) : 
  (60 : ℝ) / ((30 / v) + (30 / (v + 16))) = 30 → v = 24 := by
  sorry

end NUMINAMATH_CALUDE_first_half_speed_l3459_345919


namespace NUMINAMATH_CALUDE_sum_of_reciprocal_relations_l3459_345915

theorem sum_of_reciprocal_relations (x y : ℚ) 
  (h1 : x ≠ 0) (h2 : y ≠ 0)
  (eq1 : 1 / x + 1 / y = 4) 
  (eq2 : 1 / x - 1 / y = -5) : 
  x + y = -16 / 9 := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocal_relations_l3459_345915


namespace NUMINAMATH_CALUDE_exponential_fraction_simplification_l3459_345979

theorem exponential_fraction_simplification :
  (3^1008 + 3^1006) / (3^1008 - 3^1006) = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_exponential_fraction_simplification_l3459_345979


namespace NUMINAMATH_CALUDE_teresa_pencil_distribution_l3459_345968

/-- Given Teresa's pencil collection and distribution rules, prove each sibling gets 13 pencils -/
theorem teresa_pencil_distribution :
  let colored_pencils : ℕ := 14
  let black_pencils : ℕ := 35
  let total_pencils : ℕ := colored_pencils + black_pencils
  let pencils_to_keep : ℕ := 10
  let number_of_siblings : ℕ := 3
  let pencils_to_distribute : ℕ := total_pencils - pencils_to_keep
  pencils_to_distribute / number_of_siblings = 13 :=
by
  sorry

#eval (14 + 35 - 10) / 3  -- This should output 13

end NUMINAMATH_CALUDE_teresa_pencil_distribution_l3459_345968


namespace NUMINAMATH_CALUDE_unique_solution_l3459_345987

theorem unique_solution : ∃! (x p : ℕ), 
  Prime p ∧ 
  x * (x + 1) * (x + 2) * (x + 3) = 1679^(p - 1) + 1680^(p - 1) + 1681^(p - 1) ∧
  x = 4 ∧ p = 2 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_l3459_345987


namespace NUMINAMATH_CALUDE_cosine_sine_sum_equals_half_l3459_345900

theorem cosine_sine_sum_equals_half : 
  Real.cos (80 * π / 180) * Real.cos (20 * π / 180) + 
  Real.sin (100 * π / 180) * Real.sin (380 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sine_sum_equals_half_l3459_345900


namespace NUMINAMATH_CALUDE_thirty_percent_less_problem_l3459_345925

theorem thirty_percent_less_problem (x : ℝ) : 
  (63 = 90 - 0.3 * 90) → (x + 0.25 * x = 63) → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_thirty_percent_less_problem_l3459_345925


namespace NUMINAMATH_CALUDE_company_works_four_weeks_per_month_l3459_345990

/-- Represents the company's employee and payroll information -/
structure Company where
  initial_employees : ℕ
  additional_employees : ℕ
  hourly_wage : ℚ
  hours_per_day : ℕ
  days_per_week : ℕ
  total_monthly_pay : ℚ

/-- Calculates the number of weeks worked per month -/
def weeks_per_month (c : Company) : ℚ :=
  let total_employees := c.initial_employees + c.additional_employees
  let daily_pay := c.hourly_wage * c.hours_per_day
  let weekly_pay := daily_pay * c.days_per_week
  let total_weekly_pay := weekly_pay * total_employees
  c.total_monthly_pay / total_weekly_pay

/-- Theorem stating that the company's employees work 4 weeks per month -/
theorem company_works_four_weeks_per_month :
  let c : Company := {
    initial_employees := 500,
    additional_employees := 200,
    hourly_wage := 12,
    hours_per_day := 10,
    days_per_week := 5,
    total_monthly_pay := 1680000
  }
  weeks_per_month c = 4 := by
  sorry


end NUMINAMATH_CALUDE_company_works_four_weeks_per_month_l3459_345990


namespace NUMINAMATH_CALUDE_second_fraction_greater_l3459_345945

/-- Define the first fraction -/
def fraction1 : ℚ := (77 * 10^2009 + 7) / (77.77 * 10^2010)

/-- Define the second fraction -/
def fraction2 : ℚ := (33 * (10^2010 - 1) / 9) / (33 * (10^2011 - 1) / 99)

/-- Theorem stating that the second fraction is greater than the first -/
theorem second_fraction_greater : fraction2 > fraction1 := by
  sorry

end NUMINAMATH_CALUDE_second_fraction_greater_l3459_345945


namespace NUMINAMATH_CALUDE_movie_duration_l3459_345952

theorem movie_duration (screens : ℕ) (open_hours : ℕ) (total_movies : ℕ) 
  (h1 : screens = 6) 
  (h2 : open_hours = 8) 
  (h3 : total_movies = 24) : 
  (screens * open_hours) / total_movies = 2 := by
  sorry

end NUMINAMATH_CALUDE_movie_duration_l3459_345952


namespace NUMINAMATH_CALUDE_sqrt_16_equals_4_l3459_345949

theorem sqrt_16_equals_4 : Real.sqrt 16 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_16_equals_4_l3459_345949


namespace NUMINAMATH_CALUDE_profit_sharing_ratio_l3459_345944

/-- Represents the business partnership between Praveen and Hari -/
structure Partnership where
  praveen_initial : ℚ
  hari_initial : ℚ
  total_months : ℕ
  hari_join_month : ℕ

/-- Calculates the effective contribution of a partner -/
def effective_contribution (initial : ℚ) (months : ℕ) : ℚ :=
  initial * months

/-- Theorem stating the profit-sharing ratio between Praveen and Hari -/
theorem profit_sharing_ratio (p : Partnership) 
  (h1 : p.praveen_initial = 3780)
  (h2 : p.hari_initial = 9720)
  (h3 : p.total_months = 12)
  (h4 : p.hari_join_month = 5) :
  (effective_contribution p.praveen_initial p.total_months) / 
  (effective_contribution p.hari_initial (p.total_months - p.hari_join_month)) = 2 / 3 := by
  sorry


end NUMINAMATH_CALUDE_profit_sharing_ratio_l3459_345944


namespace NUMINAMATH_CALUDE_f_of_two_eq_two_fifths_l3459_345975

noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 2 - Real.sin x * Real.cos x

theorem f_of_two_eq_two_fifths : f (Real.arctan 2) = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_f_of_two_eq_two_fifths_l3459_345975


namespace NUMINAMATH_CALUDE_more_books_than_maddie_l3459_345962

/-- Proves that Amy and Luisa have 9 more books than Maddie -/
theorem more_books_than_maddie 
  (maddie_books : ℕ) 
  (luisa_books : ℕ) 
  (amy_books : ℕ)
  (h1 : maddie_books = 15)
  (h2 : luisa_books = 18)
  (h3 : amy_books = 6) :
  luisa_books + amy_books - maddie_books = 9 := by
sorry

end NUMINAMATH_CALUDE_more_books_than_maddie_l3459_345962


namespace NUMINAMATH_CALUDE_macks_round_trip_l3459_345993

/-- Mack's round trip problem -/
theorem macks_round_trip 
  (total_time : ℝ) 
  (time_to_office : ℝ) 
  (return_speed : ℝ) 
  (h1 : total_time = 3) 
  (h2 : time_to_office = 1.4) 
  (h3 : return_speed = 62) :
  ∃ speed_to_office : ℝ, 
    speed_to_office * time_to_office = return_speed * (total_time - time_to_office) ∧ 
    (speed_to_office ≥ 70.85 ∧ speed_to_office ≤ 70.87) :=
by
  sorry


end NUMINAMATH_CALUDE_macks_round_trip_l3459_345993


namespace NUMINAMATH_CALUDE_smallest_multiple_l3459_345907

theorem smallest_multiple : ∃ n : ℕ, 
  (n ≥ 100 ∧ n < 1000) ∧ 
  (n % 5 = 0) ∧ 
  (n % 8 = 0) ∧ 
  (n % 2 = 0) ∧ 
  (∀ m : ℕ, (m ≥ 100 ∧ m < 1000) ∧ (m % 5 = 0) ∧ (m % 8 = 0) ∧ (m % 2 = 0) → n ≤ m) ∧
  n = 120 := by
sorry

end NUMINAMATH_CALUDE_smallest_multiple_l3459_345907


namespace NUMINAMATH_CALUDE_intersection_M_N_l3459_345904

def M : Set ℤ := {m : ℤ | -3 < m ∧ m < 2}
def N : Set ℤ := {n : ℤ | -1 ≤ n ∧ n ≤ 3}

theorem intersection_M_N : M ∩ N = {-1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3459_345904


namespace NUMINAMATH_CALUDE_intersection_solution_l3459_345999

/-- Given two linear functions that intersect at x = 2, prove that the solution
    to their system of equations is (2, 2) -/
theorem intersection_solution (a b : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := fun x ↦ 2 * x - 2
  let g : ℝ → ℝ := fun x ↦ a * x + b
  (f 2 = g 2) →
  (∃! p : ℝ × ℝ, p.1 = 2 ∧ p.2 = 2 ∧ 2 * p.1 - p.2 = 2 ∧ p.2 = a * p.1 + b) :=
by sorry

end NUMINAMATH_CALUDE_intersection_solution_l3459_345999


namespace NUMINAMATH_CALUDE_trig_identity_l3459_345950

theorem trig_identity (x y : ℝ) :
  Real.sin x ^ 2 + Real.sin (x + y) ^ 2 - 2 * Real.sin x * Real.sin y * Real.cos (x + y) =
  Real.sin x ^ 2 + Real.sin y ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l3459_345950


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3459_345969

theorem sqrt_equation_solution (x : ℝ) : 
  Real.sqrt (3 + Real.sqrt (x^2)) = 4 ↔ x = 13 ∨ x = -13 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3459_345969


namespace NUMINAMATH_CALUDE_chord_length_l3459_345991

-- Define the line l
def line_l (x y : ℝ) : Prop := y = -3/4 * x + 5/4

-- Define the circle O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 2*y + 1 = 0

-- Theorem statement
theorem chord_length :
  ∃ (chord_length : ℝ),
    (∀ x y : ℝ, line_l x y → circle_O x y → 
      chord_length = 2 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_chord_length_l3459_345991


namespace NUMINAMATH_CALUDE_distribute_six_to_four_l3459_345942

/-- The number of ways to distribute n distinct objects into k distinct groups,
    where each group must contain at least one object. -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 6 distinct objects into 4 distinct groups,
    where each group must contain at least one object, is 1560. -/
theorem distribute_six_to_four : distribute 6 4 = 1560 := by sorry

end NUMINAMATH_CALUDE_distribute_six_to_four_l3459_345942


namespace NUMINAMATH_CALUDE_magnificent_class_size_l3459_345943

theorem magnificent_class_size :
  ∀ (girls boys chocolates_given : ℕ),
    girls + boys = 33 →
    boys = girls + 3 →
    girls * girls + boys * boys = chocolates_given →
    chocolates_given = 540 - 12 →
    True :=
by
  sorry

end NUMINAMATH_CALUDE_magnificent_class_size_l3459_345943


namespace NUMINAMATH_CALUDE_lexi_run_distance_l3459_345941

/-- Proves that running 13 laps on a quarter-mile track equals 3.25 miles -/
theorem lexi_run_distance (lap_length : ℚ) (num_laps : ℕ) : 
  lap_length = 1/4 → num_laps = 13 → lap_length * num_laps = 3.25 := by
  sorry

end NUMINAMATH_CALUDE_lexi_run_distance_l3459_345941


namespace NUMINAMATH_CALUDE_angle_bisector_theorem_AX_length_l3459_345928

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the point X on BC
def X (t : Triangle) : ℝ × ℝ := sorry

-- Define the length function
def length (p q : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem angle_bisector_theorem (t : Triangle) :
  -- CX bisects ∠ACB
  (length (t.A) (X t)) * (length (t.B) (t.C)) =
  (length (t.A) (t.C)) * (length (t.B) (X t)) :=
sorry

-- State the main theorem
theorem AX_length (t : Triangle) :
  -- Conditions
  length (t.B) (t.C) = 50 →
  length (t.A) (t.C) = 40 →
  length (t.B) (X t) = 35 →
  -- CX bisects ∠ACB (using angle_bisector_theorem)
  (length (t.A) (X t)) * (length (t.B) (t.C)) =
  (length (t.A) (t.C)) * (length (t.B) (X t)) →
  -- Conclusion
  length (t.A) (X t) = 28 :=
sorry

end NUMINAMATH_CALUDE_angle_bisector_theorem_AX_length_l3459_345928


namespace NUMINAMATH_CALUDE_shaded_area_is_54_l3459_345908

/-- The area of a right triangle with base 12 cm and height 9 cm is 54 cm². -/
theorem shaded_area_is_54 :
  let base : ℝ := 12
  let height : ℝ := 9
  (1 / 2 : ℝ) * base * height = 54 := by sorry

end NUMINAMATH_CALUDE_shaded_area_is_54_l3459_345908


namespace NUMINAMATH_CALUDE_number_multiplying_a_l3459_345963

theorem number_multiplying_a (a b : ℝ) (h1 : ∃ x, x * a = 8 * b) (h2 : a ≠ 0 ∧ b ≠ 0) (h3 : a / 8 = b / 7) :
  ∃ x, x * a = 8 * b ∧ x = 7 := by
  sorry

end NUMINAMATH_CALUDE_number_multiplying_a_l3459_345963


namespace NUMINAMATH_CALUDE_tan_600_degrees_l3459_345964

theorem tan_600_degrees : Real.tan (600 * π / 180) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_600_degrees_l3459_345964


namespace NUMINAMATH_CALUDE_hundred_power_ten_as_sum_of_tens_l3459_345955

theorem hundred_power_ten_as_sum_of_tens (n : ℕ) : (100 ^ 10 : ℕ) = n * 10 → n = 10 ^ 19 := by
  sorry

end NUMINAMATH_CALUDE_hundred_power_ten_as_sum_of_tens_l3459_345955


namespace NUMINAMATH_CALUDE_certain_number_proof_l3459_345994

theorem certain_number_proof (m : ℤ) (x : ℝ) (h1 : m = 6) (h2 : x^(2*m) = 2^(18 - m)) : x = 2 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l3459_345994


namespace NUMINAMATH_CALUDE_total_balloons_count_l3459_345981

/-- The number of yellow balloons Tom has -/
def tom_balloons : ℕ := 9

/-- The number of yellow balloons Sara has -/
def sara_balloons : ℕ := 8

/-- The total number of yellow balloons Tom and Sara have -/
def total_balloons : ℕ := tom_balloons + sara_balloons

theorem total_balloons_count : total_balloons = 17 := by
  sorry

end NUMINAMATH_CALUDE_total_balloons_count_l3459_345981


namespace NUMINAMATH_CALUDE_inscribed_circle_rectangle_area_l3459_345956

theorem inscribed_circle_rectangle_area :
  ∀ (r : ℝ) (ratio : ℝ),
    r = 7 →
    ratio = 3 →
    let d := 2 * r
    let w := d
    let l := ratio * w
    l * w = 588 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_rectangle_area_l3459_345956


namespace NUMINAMATH_CALUDE_intersection_range_value_range_on_curve_l3459_345961

-- Define the line l
def line_l (α : Real) : Set (Real × Real) :=
  {(x, y) | ∃ t, x = -2 + t * Real.cos α ∧ y = t * Real.sin α}

-- Define the curve C
def curve_C : Set (Real × Real) :=
  {(x, y) | (x - 2)^2 + y^2 = 4}

-- Theorem for part (I)
theorem intersection_range (α : Real) :
  (∃ p, p ∈ line_l α ∧ p ∈ curve_C) ↔ 
  (0 ≤ α ∧ α ≤ Real.pi/6) ∨ (5*Real.pi/6 ≤ α ∧ α ≤ Real.pi) :=
sorry

-- Theorem for part (II)
theorem value_range_on_curve :
  ∀ (x y : Real), (x, y) ∈ curve_C → -2 ≤ x + Real.sqrt 3 * y ∧ x + Real.sqrt 3 * y ≤ 6 :=
sorry

end NUMINAMATH_CALUDE_intersection_range_value_range_on_curve_l3459_345961


namespace NUMINAMATH_CALUDE_function_properties_l3459_345951

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * y) = y^2 * f x + x^2 * f y

/-- The main theorem stating the properties of the function -/
theorem function_properties (f : ℝ → ℝ) (h : FunctionalEquation f) :
  f 0 = 0 ∧ f 1 = 0 ∧ ∀ x : ℝ, f (-x) = f x := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l3459_345951


namespace NUMINAMATH_CALUDE_total_questions_in_contest_l3459_345920

/-- Represents a participant in the spelling contest -/
structure Participant where
  name : String
  round1_correct : Nat
  round1_wrong : Nat
  round2_correct : Nat
  round2_wrong : Nat
  round3_correct : Nat
  round3_wrong : Nat

/-- Calculates the total number of questions for a participant in all rounds -/
def totalQuestions (p : Participant) : Nat :=
  p.round1_correct + p.round1_wrong +
  p.round2_correct + p.round2_wrong +
  p.round3_correct + p.round3_wrong

/-- Represents the spelling contest -/
structure SpellingContest where
  drew : Participant
  carla : Participant
  blake : Participant

/-- Theorem stating the total number of questions in the spelling contest -/
theorem total_questions_in_contest (contest : SpellingContest)
  (h1 : contest.drew.round1_correct = 20)
  (h2 : contest.drew.round1_wrong = 6)
  (h3 : contest.carla.round1_correct = 14)
  (h4 : contest.carla.round1_wrong = 2 * contest.drew.round1_wrong)
  (h5 : contest.drew.round2_correct = 24)
  (h6 : contest.drew.round2_wrong = 9)
  (h7 : contest.carla.round2_correct = 21)
  (h8 : contest.carla.round2_wrong = 8)
  (h9 : contest.blake.round2_correct = 18)
  (h10 : contest.blake.round2_wrong = 11)
  (h11 : contest.drew.round3_correct = 28)
  (h12 : contest.drew.round3_wrong = 14)
  (h13 : contest.carla.round3_correct = 22)
  (h14 : contest.carla.round3_wrong = 10)
  (h15 : contest.blake.round3_correct = 15)
  (h16 : contest.blake.round3_wrong = 16)
  : totalQuestions contest.drew + totalQuestions contest.carla + totalQuestions contest.blake = 248 := by
  sorry


end NUMINAMATH_CALUDE_total_questions_in_contest_l3459_345920


namespace NUMINAMATH_CALUDE_problem_29_AHSME_1978_l3459_345910

theorem problem_29_AHSME_1978 (a b c x : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h1 : (a + b - c) / c = (a - b + c) / b)
  (h2 : (a + b - c) / c = (-a + b + c) / a)
  (h3 : x = ((a + b) * (b + c) * (c + a)) / (a * b * c))
  (h4 : x < 0) :
  x = -1 := by sorry

end NUMINAMATH_CALUDE_problem_29_AHSME_1978_l3459_345910


namespace NUMINAMATH_CALUDE_problem_solution_l3459_345903

-- Define the region D
def D : Set (ℝ × ℝ) := {(x, y) | (x - 1)^2 + (y - 2)^2 ≤ 4}

-- Define proposition p
def p : Prop := ∀ (x y : ℝ), (x, y) ∈ D → 2*x + y ≤ 8

-- Define proposition q
def q : Prop := ∃ (x y : ℝ), (x, y) ∈ D ∧ 2*x + y ≤ -1

-- Theorem to prove
theorem problem_solution : (¬p ∨ q) ∧ (¬p ∧ ¬q) := by sorry

end NUMINAMATH_CALUDE_problem_solution_l3459_345903


namespace NUMINAMATH_CALUDE_max_min_sum_absolute_value_l3459_345966

theorem max_min_sum_absolute_value (x y : ℝ) 
  (h1 : x - y + 1 ≥ 0)
  (h2 : x + y - 1 ≥ 0)
  (h3 : 3 * x - y - 3 ≤ 0) :
  ∃ (z_max z_min : ℝ),
    (∀ (x' y' : ℝ), 
      x' - y' + 1 ≥ 0 → 
      x' + y' - 1 ≥ 0 → 
      3 * x' - y' - 3 ≤ 0 → 
      |x' - 4 * y' + 1| ≤ z_max ∧ 
      |x' - 4 * y' + 1| ≥ z_min) ∧
    z_max + z_min = 11 / Real.sqrt 17 :=
by sorry

end NUMINAMATH_CALUDE_max_min_sum_absolute_value_l3459_345966


namespace NUMINAMATH_CALUDE_constant_relationship_l3459_345935

theorem constant_relationship (a b c d : ℝ) :
  (∀ θ : ℝ, 0 < θ ∧ θ < π / 2 →
    (a * Real.sin θ + b * Real.cos θ - c = 0) ∧
    (a * Real.cos θ - b * Real.sin θ + d = 0)) →
  a^2 + b^2 = c^2 + d^2 := by sorry

end NUMINAMATH_CALUDE_constant_relationship_l3459_345935


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3459_345976

theorem quadratic_equation_roots : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
  (x₁^2 - 2*x₁ - 6 = 0) ∧ (x₂^2 - 2*x₂ - 6 = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3459_345976


namespace NUMINAMATH_CALUDE_no_solution_cubic_system_l3459_345938

theorem no_solution_cubic_system (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  ¬∃ x : ℝ, (x^3 - a*x^2 + b^3 = 0) ∧ (x^3 - b*x^2 + c^3 = 0) ∧ (x^3 - c*x^2 + a^3 = 0) :=
by sorry

end NUMINAMATH_CALUDE_no_solution_cubic_system_l3459_345938


namespace NUMINAMATH_CALUDE_dessert_preference_l3459_345937

theorem dessert_preference (total : ℕ) (apple : ℕ) (chocolate : ℕ) (neither : ℕ) :
  total = 50 →
  apple = 22 →
  chocolate = 20 →
  neither = 17 →
  ∃ (both : ℕ), both = apple + chocolate - (total - neither) :=
by sorry

end NUMINAMATH_CALUDE_dessert_preference_l3459_345937


namespace NUMINAMATH_CALUDE_range_of_a_l3459_345960

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + a * x + 1 > 0) ∨ 
  (∃ x₀ : ℝ, x₀^2 - x₀ + a = 0) ∧ 
  ¬((∀ x : ℝ, a * x^2 + a * x + 1 > 0) ∧ 
    (∃ x₀ : ℝ, x₀^2 - x₀ + a = 0)) →
  a < 0 ∨ (1/4 < a ∧ a < 4) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3459_345960


namespace NUMINAMATH_CALUDE_quarter_circles_sum_approaches_diameter_l3459_345970

/-- The sum of the lengths of the arcs of quarter circles approaches the diameter as n approaches infinity -/
theorem quarter_circles_sum_approaches_diameter (D : ℝ) (h : D > 0) :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |n * (π * D / (4 * n)) - D| < ε :=
sorry

end NUMINAMATH_CALUDE_quarter_circles_sum_approaches_diameter_l3459_345970


namespace NUMINAMATH_CALUDE_intersection_complement_M_and_N_l3459_345913

-- Define the universal set U
def U : Set ℤ := {-2, -1, 0, 1, 2, 3}

-- Define set M
def M : Set ℤ := {0, 1, 2}

-- Define set N
def N : Set ℤ := {0, 1, 2, 3}

-- Theorem statement
theorem intersection_complement_M_and_N :
  (U \ M) ∩ N = {3} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_M_and_N_l3459_345913


namespace NUMINAMATH_CALUDE_collinear_vectors_n_value_l3459_345918

/-- Two vectors in ℝ² are collinear if their cross product is zero -/
def collinear (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem collinear_vectors_n_value :
  ∀ n : ℝ, collinear (n, 1) (4, n) → n = 2 ∨ n = -2 := by
  sorry

end NUMINAMATH_CALUDE_collinear_vectors_n_value_l3459_345918


namespace NUMINAMATH_CALUDE_consecutive_good_numbers_l3459_345906

/-- A number is good if it can be expressed as 2^x + y^2 for nonnegative integers x and y. -/
def IsGood (n : ℕ) : Prop :=
  ∃ x y : ℕ, n = 2^x + y^2

/-- A set of 5 consecutive numbers is a good set if all numbers in the set are good. -/
def IsGoodSet (s : Fin 5 → ℕ) : Prop :=
  (∀ i : Fin 5, IsGood (s i)) ∧ (∀ i : Fin 4, s (Fin.succ i) = s i + 1)

/-- The theorem states that there are only six sets of 5 consecutive good numbers. -/
theorem consecutive_good_numbers :
  ∀ s : Fin 5 → ℕ, IsGoodSet s →
    (s = ![1, 2, 3, 4, 5]) ∨
    (s = ![2, 3, 4, 5, 6]) ∨
    (s = ![8, 9, 10, 11, 12]) ∨
    (s = ![9, 10, 11, 12, 13]) ∨
    (s = ![288, 289, 290, 291, 292]) ∨
    (s = ![289, 290, 291, 292, 293]) :=
by sorry


end NUMINAMATH_CALUDE_consecutive_good_numbers_l3459_345906


namespace NUMINAMATH_CALUDE_triple_composition_fixed_point_implies_fixed_point_l3459_345924

theorem triple_composition_fixed_point_implies_fixed_point
  (f : ℝ → ℝ) (hf : Continuous f)
  (h : ∃ x, f (f (f x)) = x) :
  ∃ x₀, f x₀ = x₀ := by
  sorry

end NUMINAMATH_CALUDE_triple_composition_fixed_point_implies_fixed_point_l3459_345924


namespace NUMINAMATH_CALUDE_class_average_score_l3459_345936

theorem class_average_score (total_students : Nat) (group1_students : Nat) (group1_average : ℚ)
  (score1 score2 score3 score4 : ℚ) :
  total_students = 30 →
  group1_students = 26 →
  group1_average = 82 →
  score1 = 90 →
  score2 = 85 →
  score3 = 88 →
  score4 = 80 →
  let group1_total := group1_students * group1_average
  let group2_total := score1 + score2 + score3 + score4
  let class_total := group1_total + group2_total
  class_total / total_students = 82.5 := by
sorry

end NUMINAMATH_CALUDE_class_average_score_l3459_345936


namespace NUMINAMATH_CALUDE_arithmetic_sequence_inequality_l3459_345923

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: For an arithmetic sequence, if 0 < a_1 < a_2, then a_2 > √(a_1 * a_3) -/
theorem arithmetic_sequence_inequality (a : ℕ → ℝ) (h : arithmetic_sequence a) :
  0 < a 1 → a 1 < a 2 → a 2 > Real.sqrt (a 1 * a 3) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_inequality_l3459_345923


namespace NUMINAMATH_CALUDE_square_area_and_perimeter_comparison_l3459_345982

theorem square_area_and_perimeter_comparison (a b : ℝ) :
  let square_I_diagonal := 2 * (a + b)
  let square_II_area := 4 * (square_I_diagonal^2 / 4)
  let square_II_perimeter := 4 * Real.sqrt square_II_area
  let rectangle_perimeter := 2 * (4 * (a + b) + (a + b))
  square_II_area = 8 * (a + b)^2 ∧ square_II_perimeter > rectangle_perimeter :=
by sorry

end NUMINAMATH_CALUDE_square_area_and_perimeter_comparison_l3459_345982


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3459_345953

theorem complex_equation_solution (z : ℂ) : (Complex.I * z = 3 - 4 * Complex.I) → z = -4 - 3 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3459_345953


namespace NUMINAMATH_CALUDE_heart_ratio_two_four_four_two_l3459_345977

def heart (n m : ℕ) : ℕ := n^(3+m) * m^(2+n)

theorem heart_ratio_two_four_four_two :
  (heart 2 4 : ℚ) / (heart 4 2) = 1/2 := by sorry

end NUMINAMATH_CALUDE_heart_ratio_two_four_four_two_l3459_345977


namespace NUMINAMATH_CALUDE_constant_sequence_l3459_345984

theorem constant_sequence (a : ℕ → ℕ) 
  (h : ∀ i j : ℕ, i > j → ((i - j)^(2*(i - j)) + 1) ∣ (a i - a j)) :
  ∀ n : ℕ, n ≥ 1 → a n = a 1 :=
by sorry

end NUMINAMATH_CALUDE_constant_sequence_l3459_345984


namespace NUMINAMATH_CALUDE_probability_of_white_ball_l3459_345901

theorem probability_of_white_ball (p_red p_black p_white : ℝ) : 
  p_red = 0.3 → p_black = 0.5 → p_red + p_black + p_white = 1 → p_white = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_white_ball_l3459_345901


namespace NUMINAMATH_CALUDE_equal_chord_lengths_l3459_345971

/-- A circle in the 2D plane -/
structure Circle where
  D : ℝ
  E : ℝ
  F : ℝ

/-- The equation of the circle: x^2 + y^2 + Dx + Ey + F = 0 -/
def Circle.equation (c : Circle) (x y : ℝ) : Prop :=
  x^2 + y^2 + c.D * x + c.E * y + c.F = 0

/-- The length of the chord cut by the x-axis -/
def Circle.xChordLength (c : Circle) : ℝ := sorry

/-- The length of the chord cut by the y-axis -/
def Circle.yChordLength (c : Circle) : ℝ := sorry

/-- Theorem: If D^2 ≠ E^2 > 4F, then the lengths of the chords cut by the two coordinate axes are equal -/
theorem equal_chord_lengths (c : Circle) 
    (h1 : c.D^2 ≠ c.E^2) 
    (h2 : c.D^2 > 4 * c.F) 
    (h3 : c.E^2 > 4 * c.F) : 
    c.xChordLength = c.yChordLength := by
  sorry

end NUMINAMATH_CALUDE_equal_chord_lengths_l3459_345971


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l3459_345917

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  X^6 + X^5 + 2*X^3 - X^2 + 3 = (X + 2) * (X - 1) * q + (-X + 5) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l3459_345917


namespace NUMINAMATH_CALUDE_positive_real_inequality_l3459_345911

theorem positive_real_inequality (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x * y / z) + (y * z / x) + (z * x / y) > 2 * (x^3 + y^3 + z^3)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_positive_real_inequality_l3459_345911


namespace NUMINAMATH_CALUDE_exists_valid_chain_l3459_345914

/-- A chain of integers satisfying the divisibility condition -/
def ValidChain (chain : List ℕ) : Prop :=
  ∀ i : ℕ, i + 1 < chain.length →
    (chain[i]! * chain[i + 1]!) % (chain[i]! + chain[i + 1]!) = 0

/-- The existence of a valid chain between any two integers greater than 2 -/
theorem exists_valid_chain (m n : ℕ) (hm : m > 2) (hn : n > 2) :
    ∃ chain : List ℕ, chain.head! = m ∧ chain.getLast! = n ∧ ValidChain chain :=
  sorry

end NUMINAMATH_CALUDE_exists_valid_chain_l3459_345914


namespace NUMINAMATH_CALUDE_isosceles_triangle_with_50_degree_angle_l3459_345947

-- Define an isosceles triangle
structure IsoscelesTriangle where
  -- We only need to define two angles, as the third can be derived
  angle1 : ℝ
  angle2 : ℝ
  -- Condition: The triangle is isosceles (two angles are equal)
  isIsosceles : angle1 = angle2 ∨ angle1 = 180 - angle1 - angle2 ∨ angle2 = 180 - angle1 - angle2
  -- Condition: The sum of angles in a triangle is 180°
  sumIs180 : angle1 + angle2 + (180 - angle1 - angle2) = 180

-- Define our theorem
theorem isosceles_triangle_with_50_degree_angle 
  (triangle : IsoscelesTriangle) 
  (has50DegreeAngle : triangle.angle1 = 50 ∨ triangle.angle2 = 50 ∨ (180 - triangle.angle1 - triangle.angle2) = 50) :
  triangle.angle1 = 50 ∨ triangle.angle1 = 65 ∨ triangle.angle2 = 50 ∨ triangle.angle2 = 65 :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_with_50_degree_angle_l3459_345947


namespace NUMINAMATH_CALUDE_roots_of_equation_l3459_345997

def equation (x : ℝ) : ℝ := (x^2 - 5*x + 6) * (x - 3) * (x + 2)

theorem roots_of_equation : 
  {x : ℝ | equation x = 0} = {-2, 2, 3} := by sorry

end NUMINAMATH_CALUDE_roots_of_equation_l3459_345997
