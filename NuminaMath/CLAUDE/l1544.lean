import Mathlib

namespace NUMINAMATH_CALUDE_complement_of_union_l1544_154438

open Set

theorem complement_of_union (U S T : Set ℕ) : 
  U = {1,2,3,4,5,6,7,8} →
  S = {1,3,5} →
  T = {3,6} →
  (Sᶜ ∩ Tᶜ) = {2,4,7,8} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_union_l1544_154438


namespace NUMINAMATH_CALUDE_absolute_value_equals_sqrt_square_l1544_154466

theorem absolute_value_equals_sqrt_square (x : ℝ) : |x - 3| = Real.sqrt ((x - 3)^2) := by sorry

end NUMINAMATH_CALUDE_absolute_value_equals_sqrt_square_l1544_154466


namespace NUMINAMATH_CALUDE_line_through_points_l1544_154446

/-- Given a line y = ax + b passing through points (3,6) and (7,26), prove that a - b = 14 -/
theorem line_through_points (a b : ℝ) : 
  (6 : ℝ) = a * 3 + b ∧ (26 : ℝ) = a * 7 + b → a - b = 14 := by
  sorry

end NUMINAMATH_CALUDE_line_through_points_l1544_154446


namespace NUMINAMATH_CALUDE_house_painting_cans_l1544_154456

/-- Calculates the number of paint cans needed for a house painting job -/
def paint_cans_needed (num_bedrooms : ℕ) (paint_per_room : ℕ) 
  (color_can_size : ℕ) (white_can_size : ℕ) : ℕ :=
  let num_other_rooms := 2 * num_bedrooms
  let total_color_paint := num_bedrooms * paint_per_room
  let total_white_paint := num_other_rooms * paint_per_room
  let color_cans := (total_color_paint + color_can_size - 1) / color_can_size
  let white_cans := (total_white_paint + white_can_size - 1) / white_can_size
  color_cans + white_cans

/-- Theorem stating that the number of paint cans needed for the given conditions is 10 -/
theorem house_painting_cans : paint_cans_needed 3 2 1 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_house_painting_cans_l1544_154456


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1544_154486

theorem quadratic_equation_roots (a : ℝ) : 
  (∃ x : ℝ, x^2 + 2*x - a = 0 ∧ x = 2) → 
  (∃ y : ℝ, y^2 + 2*y - a = 0 ∧ y = -4) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1544_154486


namespace NUMINAMATH_CALUDE_exactly_one_true_proposition_l1544_154459

-- Define parallel vectors
def parallel (a b : ℝ × ℝ) : Prop := ∃ k : ℝ, a = k • b ∨ b = k • a

theorem exactly_one_true_proposition : ∃! n : Fin 4, match n with
  | 0 => ∀ a b : ℝ, (a * b)^2 = a^2 * b^2
  | 1 => ∀ a b : ℝ, |a + b| > |a - b|
  | 2 => ∀ a b : ℝ, |a + b|^2 = (a + b)^2
  | 3 => ∀ a b : ℝ × ℝ, parallel a b → a.1 * b.1 + a.2 * b.2 = Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)
  := by sorry

end NUMINAMATH_CALUDE_exactly_one_true_proposition_l1544_154459


namespace NUMINAMATH_CALUDE_triangle_properties_l1544_154426

open Real

theorem triangle_properties (A B C : ℝ) (a b : ℝ) :
  let D := (A + B) / 2
  2 * sin A * cos B + b * sin (2 * A) + 2 * sqrt 3 * a * cos C = 0 →
  2 = 2 →
  sqrt 3 = sqrt 3 →
  C = 2 * π / 3 ∧
  (1/2) * (1/2) * a * 2 * sin C = sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l1544_154426


namespace NUMINAMATH_CALUDE_different_color_chips_probability_l1544_154442

theorem different_color_chips_probability :
  let total_chips := 6 + 5 + 4 + 3
  let blue_chips := 6
  let red_chips := 5
  let yellow_chips := 4
  let green_chips := 3
  let prob_different_colors := 
    (blue_chips * (total_chips - blue_chips) +
     red_chips * (total_chips - red_chips) +
     yellow_chips * (total_chips - yellow_chips) +
     green_chips * (total_chips - green_chips)) / (total_chips * total_chips)
  prob_different_colors = 119 / 162 := by
  sorry

end NUMINAMATH_CALUDE_different_color_chips_probability_l1544_154442


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l1544_154482

def M : Set ℝ := {x | 0 < x ∧ x < 1}
def N : Set ℝ := {x | -2 < x ∧ x < 1}

theorem necessary_but_not_sufficient :
  (∀ a : ℝ, a ∈ M → a ∈ N) ∧
  (∃ a : ℝ, a ∈ N ∧ a ∉ M) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l1544_154482


namespace NUMINAMATH_CALUDE_solution_set_range_l1544_154494

theorem solution_set_range (t : ℝ) : 
  let A := {x : ℝ | x^2 - 4*x + t ≤ 0}
  (∃ x ∈ Set.Iic t, x ∈ A) → t ∈ Set.Icc 0 4 :=
by sorry

end NUMINAMATH_CALUDE_solution_set_range_l1544_154494


namespace NUMINAMATH_CALUDE_revenue_change_after_price_and_quantity_change_l1544_154452

theorem revenue_change_after_price_and_quantity_change 
  (original_price original_quantity : ℝ) 
  (price_increase_percentage : ℝ) 
  (quantity_decrease_percentage : ℝ) :
  let new_price := original_price * (1 + price_increase_percentage)
  let new_quantity := original_quantity * (1 - quantity_decrease_percentage)
  let original_revenue := original_price * original_quantity
  let new_revenue := new_price * new_quantity
  price_increase_percentage = 0.7 →
  quantity_decrease_percentage = 0.2 →
  (new_revenue - original_revenue) / original_revenue = 0.36 := by
sorry

end NUMINAMATH_CALUDE_revenue_change_after_price_and_quantity_change_l1544_154452


namespace NUMINAMATH_CALUDE_cannot_reach_54_from_12_l1544_154410

/-- Represents the possible operations that can be performed on the number -/
inductive Operation
  | MultiplyBy2
  | DivideBy2
  | MultiplyBy3
  | DivideBy3

/-- Applies a single operation to a number -/
def applyOperation (n : ℕ) (op : Operation) : ℕ :=
  match op with
  | Operation.MultiplyBy2 => n * 2
  | Operation.DivideBy2 => n / 2
  | Operation.MultiplyBy3 => n * 3
  | Operation.DivideBy3 => n / 3

/-- Applies a sequence of operations to a number -/
def applyOperations (initial : ℕ) (ops : List Operation) : ℕ :=
  ops.foldl applyOperation initial

/-- Theorem stating that it's impossible to reach 54 from 12 after 60 operations -/
theorem cannot_reach_54_from_12 (ops : List Operation) :
  ops.length = 60 → applyOperations 12 ops ≠ 54 := by
  sorry


end NUMINAMATH_CALUDE_cannot_reach_54_from_12_l1544_154410


namespace NUMINAMATH_CALUDE_second_product_of_98_l1544_154499

def second_digit_product (n : ℕ) : ℕ :=
  let first_product := (n / 10) * (n % 10)
  (first_product / 10) * (first_product % 10)

theorem second_product_of_98 :
  second_digit_product 98 = 14 := by
  sorry

end NUMINAMATH_CALUDE_second_product_of_98_l1544_154499


namespace NUMINAMATH_CALUDE_cube_sphere_volume_ratio_l1544_154423

theorem cube_sphere_volume_ratio (a r : ℝ) (h : a > 0) (k : r > 0) :
  6 * a^2 = 4 * Real.pi * r^2 →
  (a^3) / ((4/3) * Real.pi * r^3) = Real.sqrt 6 / 6 := by
sorry

end NUMINAMATH_CALUDE_cube_sphere_volume_ratio_l1544_154423


namespace NUMINAMATH_CALUDE_inequality_proof_l1544_154483

theorem inequality_proof (a b c A B C k : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (pos_A : 0 < A) (pos_B : 0 < B) (pos_C : 0 < C)
  (sum_a : a + A = k) (sum_b : b + B = k) (sum_c : c + C = k) : 
  a * B + b * C + c * A < k^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1544_154483


namespace NUMINAMATH_CALUDE_correct_food_suggestion_ratio_l1544_154428

/-- The ratio of food suggestions by students -/
def food_suggestion_ratio (sushi mashed_potatoes bacon tomatoes : ℕ) : List ℕ :=
  [sushi, mashed_potatoes, bacon, tomatoes]

/-- Theorem stating the correct ratio of food suggestions -/
theorem correct_food_suggestion_ratio :
  food_suggestion_ratio 297 144 467 79 = [297, 144, 467, 79] := by
  sorry

end NUMINAMATH_CALUDE_correct_food_suggestion_ratio_l1544_154428


namespace NUMINAMATH_CALUDE_expression_equals_three_l1544_154479

theorem expression_equals_three : (-1)^2 + Real.sqrt 16 - |(-3)| + 2 + (-1) = 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_three_l1544_154479


namespace NUMINAMATH_CALUDE_gcd_8251_6105_l1544_154478

theorem gcd_8251_6105 : Nat.gcd 8251 6105 = 37 := by
  sorry

end NUMINAMATH_CALUDE_gcd_8251_6105_l1544_154478


namespace NUMINAMATH_CALUDE_min_sum_squares_l1544_154441

theorem min_sum_squares (a b : ℝ) (ha : a ≠ 0) :
  (∃ x ∈ Set.Icc 3 4, (a + 2) / x = a * x + 2 * b + 1) →
  (∀ c d : ℝ, (∃ x ∈ Set.Icc 3 4, (c + 2) / x = c * x + 2 * d + 1) → c^2 + d^2 ≥ 1/100) ∧
  (∃ c d : ℝ, (∃ x ∈ Set.Icc 3 4, (c + 2) / x = c * x + 2 * d + 1) ∧ c^2 + d^2 = 1/100) :=
by sorry

end NUMINAMATH_CALUDE_min_sum_squares_l1544_154441


namespace NUMINAMATH_CALUDE_white_squares_in_20th_row_l1544_154476

/-- Represents the number of squares in the nth row of the modified stair-step figure -/
def squares_in_row (n : ℕ) : ℕ := 3 * n

/-- Represents the number of white squares in the nth row of the modified stair-step figure -/
def white_squares_in_row (n : ℕ) : ℕ := squares_in_row n / 2

theorem white_squares_in_20th_row :
  white_squares_in_row 20 = 30 := by
  sorry

#eval white_squares_in_row 20

end NUMINAMATH_CALUDE_white_squares_in_20th_row_l1544_154476


namespace NUMINAMATH_CALUDE_investment_theorem_l1544_154436

/-- Calculates the total investment with interest after one year -/
def total_investment_with_interest (initial_investment : ℝ) (amount_at_5_percent : ℝ) (rate_5_percent : ℝ) (rate_6_percent : ℝ) : ℝ :=
  let amount_at_6_percent := initial_investment - amount_at_5_percent
  let interest_5_percent := amount_at_5_percent * rate_5_percent
  let interest_6_percent := amount_at_6_percent * rate_6_percent
  initial_investment + interest_5_percent + interest_6_percent

/-- Theorem stating that the total investment with interest is $1,054 -/
theorem investment_theorem :
  total_investment_with_interest 1000 600 0.05 0.06 = 1054 := by
  sorry

end NUMINAMATH_CALUDE_investment_theorem_l1544_154436


namespace NUMINAMATH_CALUDE_specific_trapezoid_ratio_l1544_154408

/-- Represents a trapezoid with extended legs -/
structure ExtendedTrapezoid where
  -- Base lengths
  ab : ℝ
  cd : ℝ
  -- Height
  h : ℝ
  -- Condition that it's a valid trapezoid (cd > ab)
  h_valid : cd > ab

/-- The ratio of the area of triangle EAB to the area of trapezoid ABCD -/
def area_ratio (t : ExtendedTrapezoid) : ℝ :=
  -- Definition to be filled
  sorry

/-- Theorem stating the ratio for the specific trapezoid in the problem -/
theorem specific_trapezoid_ratio :
  let t : ExtendedTrapezoid := ⟨5, 20, 12, by norm_num⟩
  area_ratio t = 1 / 15 := by
  sorry

end NUMINAMATH_CALUDE_specific_trapezoid_ratio_l1544_154408


namespace NUMINAMATH_CALUDE_fred_weekend_earnings_l1544_154470

/-- Fred's earnings from car washing over the weekend -/
def fred_earnings (initial_amount : ℝ) (final_amount : ℝ) (percentage_cars_washed : ℝ) : ℝ :=
  final_amount - initial_amount

/-- Theorem stating Fred's earnings over the weekend -/
theorem fred_weekend_earnings :
  fred_earnings 19 40 0.35 = 21 := by
  sorry

end NUMINAMATH_CALUDE_fred_weekend_earnings_l1544_154470


namespace NUMINAMATH_CALUDE_fraction_equality_sum_l1544_154495

theorem fraction_equality_sum (P Q : ℚ) : 
  (4 : ℚ) / 7 = P / 49 ∧ (4 : ℚ) / 7 = 84 / Q → P + Q = 175 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_sum_l1544_154495


namespace NUMINAMATH_CALUDE_xy_sum_problem_l1544_154455

theorem xy_sum_problem (x y : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : x < y) (h4 : x + y + x * y = 119) :
  x + y ∈ ({20, 21, 24, 27} : Set ℕ) := by
sorry

end NUMINAMATH_CALUDE_xy_sum_problem_l1544_154455


namespace NUMINAMATH_CALUDE_final_sum_theorem_l1544_154457

/-- The number of participants in the game -/
def num_participants : ℕ := 42

/-- The initial value on the first calculator -/
def initial_val1 : ℤ := 2

/-- The initial value on the second calculator -/
def initial_val2 : ℤ := -2

/-- The initial value on the third calculator -/
def initial_val3 : ℤ := 3

/-- The operation performed on the first calculator -/
def op1 (n : ℤ) : ℤ := n ^ 2

/-- The operation performed on the second calculator -/
def op2 (n : ℤ) : ℤ := -n

/-- The operation performed on the third calculator -/
def op3 (n : ℤ) : ℤ := n ^ 3

/-- The final value on the first calculator after all iterations -/
noncomputable def final_val1 : ℤ := initial_val1 ^ (2 ^ num_participants)

/-- The final value on the second calculator after all iterations -/
def final_val2 : ℤ := initial_val2

/-- The final value on the third calculator after all iterations -/
noncomputable def final_val3 : ℤ := initial_val3 ^ (3 ^ num_participants)

/-- The theorem stating the sum of the final values on all calculators -/
theorem final_sum_theorem : 
  final_val1 + final_val2 + final_val3 = 2^(2^num_participants) - 2 + 3^(3^num_participants) :=
by sorry

end NUMINAMATH_CALUDE_final_sum_theorem_l1544_154457


namespace NUMINAMATH_CALUDE_hyperbola_real_axis_length_l1544_154491

/-- The length of the real axis of the hyperbola x²/4 - y² = 1 is 4. -/
theorem hyperbola_real_axis_length :
  let h : ℝ → ℝ → Prop := λ x y ↦ x^2/4 - y^2 = 1
  ∃ a : ℝ, a > 0 ∧ (∀ x y, h x y ↔ x^2/a^2 - y^2 = 1) ∧ 2*a = 4 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_real_axis_length_l1544_154491


namespace NUMINAMATH_CALUDE_sin_2023pi_over_3_l1544_154496

theorem sin_2023pi_over_3 : Real.sin (2023 * Real.pi / 3) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_2023pi_over_3_l1544_154496


namespace NUMINAMATH_CALUDE_point_product_y_coordinates_l1544_154477

theorem point_product_y_coordinates : 
  ∀ y₁ y₂ : ℝ, 
  (3 - 1)^2 + (-1 - y₁)^2 = 10^2 →
  (3 - 1)^2 + (-1 - y₂)^2 = 10^2 →
  y₁ * y₂ = -95 := by
sorry

end NUMINAMATH_CALUDE_point_product_y_coordinates_l1544_154477


namespace NUMINAMATH_CALUDE_no_natural_squares_l1544_154467

theorem no_natural_squares (x y : ℕ) : ¬(∃ a b : ℕ, x^2 + y = a^2 ∧ x - y = b^2) := by
  sorry

end NUMINAMATH_CALUDE_no_natural_squares_l1544_154467


namespace NUMINAMATH_CALUDE_chess_piece_arrangements_l1544_154411

def num_arrangements (n : ℕ) (k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial 2)^k

theorem chess_piece_arrangements :
  num_arrangements 6 3 = 90 := by
  sorry

end NUMINAMATH_CALUDE_chess_piece_arrangements_l1544_154411


namespace NUMINAMATH_CALUDE_original_number_of_people_l1544_154443

theorem original_number_of_people (n : ℕ) : 
  (n / 3 : ℚ) = 18 → n = 54 := by sorry

end NUMINAMATH_CALUDE_original_number_of_people_l1544_154443


namespace NUMINAMATH_CALUDE_one_prime_in_sequence_l1544_154439

/-- The number of digits in a natural number -/
def digits (n : ℕ) : ℕ :=
  if n < 10 then 1 else 1 + digits (n / 10)

/-- The nth term of the sequence -/
def a : ℕ → ℕ
  | 0 => 37
  | n + 1 => 5 * 10^(digits (a n)) + a n

/-- The statement that there is exactly one prime in the sequence -/
theorem one_prime_in_sequence : ∃! k, k ∈ Set.range a ∧ Nat.Prime (a k) := by
  sorry

end NUMINAMATH_CALUDE_one_prime_in_sequence_l1544_154439


namespace NUMINAMATH_CALUDE_x_value_when_y_is_two_l1544_154420

theorem x_value_when_y_is_two (x y : ℝ) : 
  y = 1 / (4 * x + 2) → y = 2 → x = -3/8 := by
sorry

end NUMINAMATH_CALUDE_x_value_when_y_is_two_l1544_154420


namespace NUMINAMATH_CALUDE_jamie_min_score_l1544_154497

/-- The minimum average score required on the last two tests to qualify for a geometry class. -/
def min_average_score (score1 score2 score3 : ℚ) (required_average : ℚ) (num_tests : ℕ) : ℚ :=
  ((required_average * num_tests) - (score1 + score2 + score3)) / 2

/-- Theorem stating the minimum average score Jamie must achieve on the next two tests. -/
theorem jamie_min_score : 
  min_average_score 80 90 78 85 5 = 88.5 := by sorry

end NUMINAMATH_CALUDE_jamie_min_score_l1544_154497


namespace NUMINAMATH_CALUDE_power_function_exponent_l1544_154468

/-- A power function passing through (1/4, 1/2) has exponent 1/2 -/
theorem power_function_exponent (m : ℝ) (a : ℝ) :
  m * (1/4 : ℝ)^a = 1/2 → a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_power_function_exponent_l1544_154468


namespace NUMINAMATH_CALUDE_sum_of_integers_l1544_154474

theorem sum_of_integers (x y : ℕ+) 
  (h1 : x.val - y.val = 15) 
  (h2 : x.val * y.val = 54) : 
  x.val + y.val = 21 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l1544_154474


namespace NUMINAMATH_CALUDE_normal_distribution_mean_half_l1544_154461

-- Define a random variable following normal distribution
def normal_distribution (μ : ℝ) (σ : ℝ) (hσ : σ > 0) : Type := ℝ

-- Define the probability function
noncomputable def P (ξ : normal_distribution μ σ hσ) (pred : ℝ → Prop) : ℝ := sorry

-- Theorem statement
theorem normal_distribution_mean_half 
  (μ σ : ℝ) (hσ : σ > 0) (ξ : normal_distribution μ σ hσ) 
  (h : P ξ (λ x => x < 0) + P ξ (λ x => x < 1) = 1) : 
  μ = 1/2 := by sorry

end NUMINAMATH_CALUDE_normal_distribution_mean_half_l1544_154461


namespace NUMINAMATH_CALUDE_quilt_shaded_fraction_l1544_154487

/-- Represents a square quilt block -/
structure QuiltBlock where
  total_squares : ℕ
  corner_squares : ℕ
  center_half_triangles : ℕ

/-- Calculates the shaded fraction of a quilt block -/
def shaded_fraction (q : QuiltBlock) : ℚ :=
  let corner_area := q.corner_squares
  let center_area := q.center_half_triangles / 2
  (corner_area + center_area) / q.total_squares

/-- Theorem stating that the shaded fraction of the described quilt block is 3/8 -/
theorem quilt_shaded_fraction :
  let q := QuiltBlock.mk 16 4 4
  shaded_fraction q = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_quilt_shaded_fraction_l1544_154487


namespace NUMINAMATH_CALUDE_tetrahedron_max_lateral_area_l1544_154437

/-- Given a tetrahedron A-BCD where AB, AC, AD are mutually perpendicular
    and the radius of the circumscribed sphere is 2,
    prove that the maximum lateral surface area S of the tetrahedron is 8. -/
theorem tetrahedron_max_lateral_area :
  ∀ (a b c : ℝ),
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 + c^2 = 16 →
  (∀ (S : ℝ), S = (a * b + b * c + a * c) / 2 → S ≤ 8) ∧
  (∃ (S : ℝ), S = (a * b + b * c + a * c) / 2 ∧ S = 8) :=
by sorry

end NUMINAMATH_CALUDE_tetrahedron_max_lateral_area_l1544_154437


namespace NUMINAMATH_CALUDE_f_range_l1544_154463

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 2*x

-- State the theorem
theorem f_range :
  ∀ x ∈ Set.Icc (-2 : ℝ) 1,
    -1 ≤ f x ∧ f x ≤ 3 ∧
    (∃ x₁ ∈ Set.Icc (-2 : ℝ) 1, f x₁ = -1) ∧
    (∃ x₂ ∈ Set.Icc (-2 : ℝ) 1, f x₂ = 3) :=
by sorry

end NUMINAMATH_CALUDE_f_range_l1544_154463


namespace NUMINAMATH_CALUDE_complex_modulus_l1544_154493

theorem complex_modulus (z : ℂ) (h : z * (2 - 3*I) = 6 + 4*I) : Complex.abs z = 26/5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l1544_154493


namespace NUMINAMATH_CALUDE_indeterminate_or_l1544_154473

theorem indeterminate_or (p q : Prop) 
  (h1 : ¬p) 
  (h2 : ¬(p ∧ q)) : 
  ¬∀ (b : Bool), (p ∨ q) = b :=
by
  sorry

end NUMINAMATH_CALUDE_indeterminate_or_l1544_154473


namespace NUMINAMATH_CALUDE_min_value_of_expression_l1544_154417

/-- The line equation ax + 2by - 2 = 0 -/
def line_equation (a b x y : ℝ) : Prop := a * x + 2 * b * y - 2 = 0

/-- The circle equation x^2 + y^2 - 4x - 2y - 8 = 0 -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 2*y - 8 = 0

/-- The line bisects the circumference of the circle -/
def line_bisects_circle (a b : ℝ) : Prop :=
  ∀ x y : ℝ, line_equation a b x y → circle_equation x y

theorem min_value_of_expression (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (h_bisect : line_bisects_circle a b) :
  (1 / a + 2 / b) ≥ 3 + 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l1544_154417


namespace NUMINAMATH_CALUDE_room_053_selected_l1544_154448

/-- Represents a room number in the range [1, 64] -/
def RoomNumber := Fin 64

/-- Systematic sampling function -/
def systematicSample (totalRooms sampleSize : ℕ) (firstSample : RoomNumber) : List RoomNumber :=
  let interval := totalRooms / sampleSize
  (List.range sampleSize).map (fun i => ⟨(firstSample.val + i * interval) % totalRooms + 1, sorry⟩)

theorem room_053_selected :
  let totalRooms := 64
  let sampleSize := 8
  let firstSample : RoomNumber := ⟨5, sorry⟩
  let sampledRooms := systematicSample totalRooms sampleSize firstSample
  53 ∈ sampledRooms.map (fun r => r.val) := by
  sorry

#eval systematicSample 64 8 ⟨5, sorry⟩

end NUMINAMATH_CALUDE_room_053_selected_l1544_154448


namespace NUMINAMATH_CALUDE_area_ABCGDE_value_l1544_154444

/-- Shape ABCGDE formed by an equilateral triangle ABC and a square DEFG -/
structure ShapeABCGDE where
  /-- Side length of equilateral triangle ABC -/
  triangle_side : ℝ
  /-- Side length of square DEFG -/
  square_side : ℝ
  /-- Point D is at the midpoint of BC -/
  d_midpoint : Bool

/-- Calculate the area of shape ABCGDE -/
def area_ABCGDE (shape : ShapeABCGDE) : ℝ :=
  sorry

/-- Theorem: The area of shape ABCGDE is 27 + 9√3 -/
theorem area_ABCGDE_value :
  ∀ (shape : ShapeABCGDE),
  shape.triangle_side = 6 ∧ 
  shape.square_side = 6 ∧ 
  shape.d_midpoint = true →
  area_ABCGDE shape = 27 + 9 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_CALUDE_area_ABCGDE_value_l1544_154444


namespace NUMINAMATH_CALUDE_min_sum_distances_l1544_154401

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define line l₁
def line_l₁ (x y : ℝ) : Prop := 4*x - 3*y + 6 = 0

-- Define line l₂
def line_l₂ (x : ℝ) : Prop := x = -1

-- Define the distance function from a point to a line
noncomputable def dist_point_to_line (px py : ℝ) (a b c : ℝ) : ℝ :=
  abs (a * px + b * py + c) / Real.sqrt (a^2 + b^2)

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Theorem statement
theorem min_sum_distances :
  ∃ (d : ℝ), d = 2 ∧
  ∀ (px py : ℝ), parabola px py →
    d ≤ dist_point_to_line px py 4 (-3) 6 + abs (px + 1) :=
by sorry

end NUMINAMATH_CALUDE_min_sum_distances_l1544_154401


namespace NUMINAMATH_CALUDE_sequence_length_l1544_154427

theorem sequence_length (m : ℕ+) (a : ℕ → ℝ) 
  (h0 : a 0 = 37)
  (h1 : a 1 = 72)
  (hm : a m = 0)
  (h_rec : ∀ k ∈ Finset.range (m - 1), a (k + 2) = a k - 3 / a (k + 1)) :
  m = 889 := by
  sorry

end NUMINAMATH_CALUDE_sequence_length_l1544_154427


namespace NUMINAMATH_CALUDE_problem_solution_l1544_154418

theorem problem_solution (a : ℚ) : a + a/3 + a/4 = 11/4 → a = 33/19 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1544_154418


namespace NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l1544_154484

theorem geometric_sequence_sixth_term 
  (a : ℕ → ℝ)  -- The geometric sequence
  (h1 : a 1 = 4)  -- First term is 4
  (h2 : a 9 = 39304)  -- Last term is 39304
  (h3 : ∀ n : ℕ, 1 < n → n < 9 → a n = a 1 * (a 2 / a 1) ^ (n - 1))  -- Geometric sequence property
  : a 6 = 31104 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l1544_154484


namespace NUMINAMATH_CALUDE_tangent_slope_at_one_two_l1544_154492

/-- The slope of the tangent line to y = 2x^2 at (1, 2) is 4 -/
theorem tangent_slope_at_one_two : 
  let f : ℝ → ℝ := fun x ↦ 2 * x^2
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  (deriv f) x₀ = 4 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_at_one_two_l1544_154492


namespace NUMINAMATH_CALUDE_silver_car_percentage_l1544_154422

theorem silver_car_percentage (initial_cars : ℕ) (initial_silver_percent : ℚ)
  (new_cars : ℕ) (new_non_silver_percent : ℚ) :
  initial_cars = 40 →
  initial_silver_percent = 1/5 →
  new_cars = 80 →
  new_non_silver_percent = 1/2 →
  let total_cars := initial_cars + new_cars
  let initial_silver := initial_cars * initial_silver_percent
  let new_silver := new_cars * (1 - new_non_silver_percent)
  let total_silver := initial_silver + new_silver
  (total_silver / total_cars) = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_silver_car_percentage_l1544_154422


namespace NUMINAMATH_CALUDE_max_value_x_cubed_over_y_fourth_l1544_154450

theorem max_value_x_cubed_over_y_fourth (x y : ℝ) 
  (h1 : 3 ≤ x * y^2) (h2 : x * y^2 ≤ 8) 
  (h3 : 4 ≤ x^2 / y) (h4 : x^2 / y ≤ 9) : 
  (∃ (a b : ℝ), 3 ≤ a * b^2 ∧ a * b^2 ≤ 8 ∧ 4 ≤ a^2 / b ∧ a^2 / b ≤ 9 ∧ a^3 / b^4 = 27) ∧ 
  (∀ (z w : ℝ), 3 ≤ z * w^2 → z * w^2 ≤ 8 → 4 ≤ z^2 / w → z^2 / w ≤ 9 → z^3 / w^4 ≤ 27) :=
sorry

end NUMINAMATH_CALUDE_max_value_x_cubed_over_y_fourth_l1544_154450


namespace NUMINAMATH_CALUDE_polar_bear_salmon_consumption_l1544_154498

/-- The daily diet of a polar bear at Richmond's zoo -/
structure PolarBearDiet where
  trout : ℝ
  salmon : ℝ
  total_fish : ℝ

/-- Properties of the polar bear's diet -/
def is_valid_diet (d : PolarBearDiet) : Prop :=
  d.trout = 0.2 ∧ d.total_fish = 0.6 ∧ d.total_fish = d.trout + d.salmon

theorem polar_bear_salmon_consumption (d : PolarBearDiet) 
  (h : is_valid_diet d) : d.salmon = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_polar_bear_salmon_consumption_l1544_154498


namespace NUMINAMATH_CALUDE_point_in_planar_region_l1544_154465

def planar_region (x y : ℝ) : Prop := x + 2 * y - 1 > 0

theorem point_in_planar_region :
  planar_region 0 1 ∧ 
  ¬ planar_region 1 (-1) ∧ 
  ¬ planar_region 1 0 ∧ 
  ¬ planar_region (-2) 0 :=
by sorry

end NUMINAMATH_CALUDE_point_in_planar_region_l1544_154465


namespace NUMINAMATH_CALUDE_player5_score_l1544_154430

/-- Represents a basketball player's score breakdown -/
structure PlayerScore where
  twoPointers : Nat
  threePointers : Nat
  freeThrows : Nat

/-- Calculates the total points scored by a player -/
def totalPoints (score : PlayerScore) : Nat :=
  2 * score.twoPointers + 3 * score.threePointers + score.freeThrows

theorem player5_score 
  (teamAScore : Nat)
  (player1 : PlayerScore)
  (player2 : PlayerScore)
  (player3 : PlayerScore)
  (player4 : PlayerScore)
  (h1 : teamAScore = 75)
  (h2 : player1 = ⟨0, 5, 0⟩)
  (h3 : player2 = ⟨5, 0, 5⟩)
  (h4 : player3 = ⟨0, 3, 3⟩)
  (h5 : player4 = ⟨6, 0, 0⟩) :
  teamAScore - (totalPoints player1 + totalPoints player2 + totalPoints player3 + totalPoints player4) = 14 := by
  sorry

#eval totalPoints ⟨0, 5, 0⟩  -- Player 1
#eval totalPoints ⟨5, 0, 5⟩  -- Player 2
#eval totalPoints ⟨0, 3, 3⟩  -- Player 3
#eval totalPoints ⟨6, 0, 0⟩  -- Player 4

end NUMINAMATH_CALUDE_player5_score_l1544_154430


namespace NUMINAMATH_CALUDE_person_age_l1544_154415

theorem person_age : ∃ x : ℕ, x = 30 ∧ 3 * (x + 5) - 3 * (x - 5) = x := by sorry

end NUMINAMATH_CALUDE_person_age_l1544_154415


namespace NUMINAMATH_CALUDE_clock_hands_straight_period_l1544_154425

/-- Represents the number of times clock hands are straight in a given period -/
def straight_hands (period : ℝ) : ℕ := sorry

/-- Represents the number of times clock hands coincide in a given period -/
def coinciding_hands (period : ℝ) : ℕ := sorry

/-- Represents the number of times clock hands are opposite in a given period -/
def opposite_hands (period : ℝ) : ℕ := sorry

theorem clock_hands_straight_period :
  straight_hands 12 = 22 ∧
  (∀ period : ℝ, straight_hands period = coinciding_hands period + opposite_hands period) ∧
  coinciding_hands 12 = 11 ∧
  opposite_hands 12 = 11 :=
by sorry

end NUMINAMATH_CALUDE_clock_hands_straight_period_l1544_154425


namespace NUMINAMATH_CALUDE_C₂_function_l1544_154432

-- Define the original function f
variable (f : ℝ → ℝ)

-- Define C as the graph of y = f(x)
def C (f : ℝ → ℝ) : Set (ℝ × ℝ) := {(x, y) | y = f x}

-- Define C₁ as symmetric to C with respect to x = 1
def C₁ (f : ℝ → ℝ) : Set (ℝ × ℝ) := {(x, y) | y = f (2 - x)}

-- Define C₂ as C₁ shifted one unit to the left
def C₂ (f : ℝ → ℝ) : Set (ℝ × ℝ) := {(x, y) | ∃ x', x = x' - 1 ∧ (x', y) ∈ C₁ f}

-- Theorem: The function corresponding to C₂ is y = f(1 - x)
theorem C₂_function (f : ℝ → ℝ) : C₂ f = {(x, y) | y = f (1 - x)} := by sorry

end NUMINAMATH_CALUDE_C₂_function_l1544_154432


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l1544_154462

theorem complex_magnitude_problem (z : ℂ) (h : (1 - I) / I * z = 1) : Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l1544_154462


namespace NUMINAMATH_CALUDE_max_red_points_in_grid_l1544_154451

/-- 
Given a rectangular grid of m × n points where m and n are integers greater than 7,
this theorem states that the maximum number of points that can be colored red
such that no right-angled triangle with sides parallel to the rectangle's sides
has all three vertices colored red is m + n - 2.
-/
theorem max_red_points_in_grid (m n : ℕ) (hm : m > 7) (hn : n > 7) :
  (∃ (k : ℕ), k = m + n - 2 ∧
    ∀ (S : Finset (ℕ × ℕ)), S.card = k →
      (∀ (a b c : ℕ × ℕ), a ∈ S → b ∈ S → c ∈ S →
        (a.1 = b.1 ∧ a.2 = c.2 ∧ b.2 = c.1) → false) →
    (∀ (T : Finset (ℕ × ℕ)), T.card > k →
      ∃ (a b c : ℕ × ℕ), a ∈ T ∧ b ∈ T ∧ c ∈ T ∧
        a.1 = b.1 ∧ a.2 = c.2 ∧ b.2 = c.1)) :=
by sorry

end NUMINAMATH_CALUDE_max_red_points_in_grid_l1544_154451


namespace NUMINAMATH_CALUDE_determine_bal_meaning_l1544_154405

/-- Represents the possible responses from a native --/
inductive Response
| Bal
| Da

/-- Represents the possible meanings of a word --/
inductive Meaning
| Yes
| No

/-- A native person who can respond to questions --/
structure Native where
  response : String → Response

/-- The meaning of the word "bal" --/
def balMeaning (n : Native) : Meaning :=
  match n.response "Are you a human?" with
  | Response.Bal => Meaning.Yes
  | Response.Da => Meaning.No

/-- Theorem stating that it's possible to determine the meaning of "bal" with a single question --/
theorem determine_bal_meaning (n : Native) :
  (∀ q : String, n.response q = Response.Bal ∨ n.response q = Response.Da) →
  (n.response "Are you a human?" = Response.Da → Meaning.Yes = Meaning.Yes) →
  (∀ q : String, n.response q = Response.Bal → Meaning.Yes = balMeaning n) :=
by
  sorry


end NUMINAMATH_CALUDE_determine_bal_meaning_l1544_154405


namespace NUMINAMATH_CALUDE_closest_integer_to_cube_root_200_l1544_154419

theorem closest_integer_to_cube_root_200 : 
  ∃ (n : ℤ), n = 6 ∧ ∀ (m : ℤ), |m^3 - 200| ≥ |n^3 - 200| := by
  sorry

end NUMINAMATH_CALUDE_closest_integer_to_cube_root_200_l1544_154419


namespace NUMINAMATH_CALUDE_fraction_equality_l1544_154409

theorem fraction_equality (x y : ℚ) (hx : x = 4/7) (hy : y = 5/11) : 
  (7*x + 11*y) / (77*x*y) = 9/20 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l1544_154409


namespace NUMINAMATH_CALUDE_olivia_earnings_l1544_154454

/-- Calculates the earnings for a tutor based on their hours worked and payment conditions. -/
def calculate_earnings (tuesday_hours : ℚ) (wednesday_minutes : ℕ) (thursday_start_hour : ℕ) (thursday_start_minute : ℕ) (thursday_end_hour : ℕ) (thursday_end_minute : ℕ) (saturday_minutes : ℕ) (hourly_rate : ℚ) (bonus_threshold : ℚ) (bonus_rate : ℚ) : ℚ :=
  sorry

/-- Proves that Olivia's earnings for the week are $28.17 given her tutoring schedule and payment conditions. -/
theorem olivia_earnings : 
  let tuesday_hours : ℚ := 3/2
  let wednesday_minutes : ℕ := 40
  let thursday_start_hour : ℕ := 9
  let thursday_start_minute : ℕ := 15
  let thursday_end_hour : ℕ := 11
  let thursday_end_minute : ℕ := 30
  let saturday_minutes : ℕ := 45
  let hourly_rate : ℚ := 5
  let bonus_threshold : ℚ := 4
  let bonus_rate : ℚ := 2
  calculate_earnings tuesday_hours wednesday_minutes thursday_start_hour thursday_start_minute thursday_end_hour thursday_end_minute saturday_minutes hourly_rate bonus_threshold bonus_rate = 28.17 := by
  sorry

end NUMINAMATH_CALUDE_olivia_earnings_l1544_154454


namespace NUMINAMATH_CALUDE_probability_two_thirds_l1544_154460

/-- The probability of drawing two balls of different colors from a bag containing 
    2 red balls and 2 yellow balls when randomly selecting 2 balls at once. -/
def probability_different_colors (total_balls : ℕ) (red_balls : ℕ) (yellow_balls : ℕ) : ℚ :=
  let total_ways := Nat.choose total_balls 2
  let different_color_ways := red_balls * yellow_balls
  different_color_ways / total_ways

/-- Theorem stating that the probability of drawing two balls of different colors 
    from a bag with 2 red balls and 2 yellow balls is 2/3. -/
theorem probability_two_thirds : 
  probability_different_colors 4 2 2 = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_thirds_l1544_154460


namespace NUMINAMATH_CALUDE_problem1_problem1_evaluation_l1544_154458

theorem problem1 (x : ℝ) : 
  3 * x^3 - (x^3 + (6 * x^2 - 7 * x)) - 2 * (x^3 - 3 * x^2 - 4 * x) = 15 * x :=
by sorry

theorem problem1_evaluation : 
  3 * (-1)^3 - ((-1)^3 + (6 * (-1)^2 - 7 * (-1))) - 2 * ((-1)^3 - 3 * (-1)^2 - 4 * (-1)) = -15 :=
by sorry

end NUMINAMATH_CALUDE_problem1_problem1_evaluation_l1544_154458


namespace NUMINAMATH_CALUDE_negation_equivalence_l1544_154421

theorem negation_equivalence :
  (¬ (∀ x : ℝ, x^2 + x + 1 > 0)) ↔ (∃ x : ℝ, x^2 + x + 1 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1544_154421


namespace NUMINAMATH_CALUDE_fuel_station_total_cost_l1544_154489

/-- Calculates the total cost for filling up vehicles at a fuel station -/
def total_cost (service_cost : ℝ) 
                (minivan_price minivan_capacity : ℝ) 
                (pickup_price pickup_capacity : ℝ)
                (semitruck_price : ℝ)
                (minivan_count pickup_count semitruck_count : ℕ) : ℝ :=
  let semitruck_capacity := pickup_capacity * 2.2
  let minivan_total := (service_cost + minivan_price * minivan_capacity) * minivan_count
  let pickup_total := (service_cost + pickup_price * pickup_capacity) * pickup_count
  let semitruck_total := (service_cost + semitruck_price * semitruck_capacity) * semitruck_count
  minivan_total + pickup_total + semitruck_total

/-- The total cost for filling up 4 mini-vans, 2 pick-up trucks, and 3 semi-trucks is $998.80 -/
theorem fuel_station_total_cost : 
  total_cost 2.20 0.70 65 0.85 100 0.95 4 2 3 = 998.80 := by
  sorry

end NUMINAMATH_CALUDE_fuel_station_total_cost_l1544_154489


namespace NUMINAMATH_CALUDE_total_lists_is_forty_l1544_154469

/-- The number of elements in the first set (Bin A) -/
def set_A_size : ℕ := 8

/-- The number of elements in the second set (Bin B) -/
def set_B_size : ℕ := 5

/-- The total number of possible lists -/
def total_lists : ℕ := set_A_size * set_B_size

/-- Theorem stating that the total number of possible lists is 40 -/
theorem total_lists_is_forty : total_lists = 40 := by
  sorry

end NUMINAMATH_CALUDE_total_lists_is_forty_l1544_154469


namespace NUMINAMATH_CALUDE_odd_prime_power_equality_l1544_154475

theorem odd_prime_power_equality (p m : ℕ) (hp : Nat.Prime p) (hp_odd : Odd p) :
  (∃ x y : ℕ, x > 1 ∧ y > 1 ∧ (x^p + y^p : ℚ) / 2 = ((x + y : ℚ) / 2)^m) → m = p := by
  sorry

end NUMINAMATH_CALUDE_odd_prime_power_equality_l1544_154475


namespace NUMINAMATH_CALUDE_tony_remaining_money_l1544_154433

def initial_money : ℕ := 20
def ticket_cost : ℕ := 8
def hotdog_cost : ℕ := 3

theorem tony_remaining_money :
  initial_money - ticket_cost - hotdog_cost = 9 := by
  sorry

end NUMINAMATH_CALUDE_tony_remaining_money_l1544_154433


namespace NUMINAMATH_CALUDE_club_leadership_selection_l1544_154485

theorem club_leadership_selection (num_girls num_boys : ℕ) 
  (h1 : num_girls = 15) 
  (h2 : num_boys = 15) : 
  num_girls * num_boys = 225 := by
  sorry

end NUMINAMATH_CALUDE_club_leadership_selection_l1544_154485


namespace NUMINAMATH_CALUDE_inverse_function_c_value_l1544_154472

/-- Given a function f and its inverse, prove the value of c -/
theorem inverse_function_c_value 
  (f : ℝ → ℝ) 
  (c : ℝ) 
  (h1 : ∀ x, f x = 1 / (3 * x + c)) 
  (h2 : ∀ x, Function.invFun f x = (2 - 3 * x) / (3 * x)) : 
  c = 1 := by
sorry

end NUMINAMATH_CALUDE_inverse_function_c_value_l1544_154472


namespace NUMINAMATH_CALUDE_gcd_of_three_numbers_l1544_154416

theorem gcd_of_three_numbers :
  Nat.gcd 13642 (Nat.gcd 19236 34176) = 2 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_three_numbers_l1544_154416


namespace NUMINAMATH_CALUDE_surjective_injective_ge_equal_l1544_154435

theorem surjective_injective_ge_equal (f g : ℕ → ℕ) 
  (hf : Function.Surjective f)
  (hg : Function.Injective g)
  (h : ∀ n : ℕ, f n ≥ g n) :
  f = g := by
  sorry

end NUMINAMATH_CALUDE_surjective_injective_ge_equal_l1544_154435


namespace NUMINAMATH_CALUDE_katie_speed_calculation_l1544_154403

-- Define the running speeds
def eugene_speed : ℚ := 4
def brianna_speed : ℚ := (2/3) * eugene_speed
def katie_speed : ℚ := (7/5) * brianna_speed

-- Theorem to prove
theorem katie_speed_calculation :
  katie_speed = 56/15 := by sorry

end NUMINAMATH_CALUDE_katie_speed_calculation_l1544_154403


namespace NUMINAMATH_CALUDE_wong_valentines_l1544_154424

/-- The number of Valentines Mrs. Wong gave away -/
def valentines_given : ℕ := 8

/-- The number of Valentines Mrs. Wong had left -/
def valentines_left : ℕ := 22

/-- The initial number of Valentines Mrs. Wong had -/
def initial_valentines : ℕ := valentines_given + valentines_left

theorem wong_valentines : initial_valentines = 30 := by
  sorry

end NUMINAMATH_CALUDE_wong_valentines_l1544_154424


namespace NUMINAMATH_CALUDE_books_sold_l1544_154431

theorem books_sold (initial_books : ℕ) (added_books : ℕ) (final_books : ℕ) : 
  initial_books = 4 → added_books = 10 → final_books = 11 → 
  initial_books - (final_books - added_books) = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_books_sold_l1544_154431


namespace NUMINAMATH_CALUDE_max_reciprocal_sum_exists_l1544_154429

/-- Given a quadratic polynomial x^2 - px + q with roots r₁ and r₂ satisfying
    r₁ + r₂ = r₁² + r₂² = r₁⁴ + r₂⁴, there exists a maximum value for 1/r₁⁵ + 1/r₂⁵ -/
theorem max_reciprocal_sum_exists (p q r₁ r₂ : ℝ) : 
  (r₁ * r₁ - p * r₁ + q = 0) →
  (r₂ * r₂ - p * r₂ + q = 0) →
  (r₁ + r₂ = r₁^2 + r₂^2) →
  (r₁ + r₂ = r₁^4 + r₂^4) →
  ∃ (M : ℝ), ∀ (s₁ s₂ : ℝ), 
    (s₁ * s₁ - p * s₁ + q = 0) →
    (s₂ * s₂ - p * s₂ + q = 0) →
    (s₁ + s₂ = s₁^2 + s₂^2) →
    (s₁ + s₂ = s₁^4 + s₂^4) →
    1/s₁^5 + 1/s₂^5 ≤ M :=
by
  sorry


end NUMINAMATH_CALUDE_max_reciprocal_sum_exists_l1544_154429


namespace NUMINAMATH_CALUDE_min_value_theorem_l1544_154449

theorem min_value_theorem (x : ℝ) (h : x > -1) : 
  x + 4 / (x + 1) ≥ 3 ∧ ∃ y > -1, y + 4 / (y + 1) = 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1544_154449


namespace NUMINAMATH_CALUDE_equilateral_triangle_and_regular_pentagon_not_similar_l1544_154400

-- Define an equilateral triangle
structure EquilateralTriangle where
  side : ℝ
  side_positive : side > 0

-- Define a regular pentagon
structure RegularPentagon where
  side : ℝ
  side_positive : side > 0

-- Define similarity between shapes
def similar (shape1 shape2 : Type) : Prop := sorry

-- Theorem statement
theorem equilateral_triangle_and_regular_pentagon_not_similar :
  ∀ (t : EquilateralTriangle) (p : RegularPentagon), ¬(similar EquilateralTriangle RegularPentagon) :=
by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_and_regular_pentagon_not_similar_l1544_154400


namespace NUMINAMATH_CALUDE_cannot_tile_modified_checkerboard_l1544_154440

/-- Represents a checkerboard with two opposite corners removed -/
structure ModifiedCheckerboard :=
  (size : Nat)
  (cornersRemoved : Nat)

/-- Represents a domino used for tiling -/
structure Domino :=
  (length : Nat)
  (width : Nat)

/-- Defines the property of a checkerboard being tileable by dominoes -/
def is_tileable (board : ModifiedCheckerboard) (domino : Domino) : Prop :=
  ∃ (tiling : Nat), tiling > 0

/-- The main theorem stating that an 8x8 checkerboard with opposite corners removed cannot be tiled by 2x1 dominoes -/
theorem cannot_tile_modified_checkerboard :
  ¬ is_tileable (ModifiedCheckerboard.mk 8 2) (Domino.mk 2 1) := by
  sorry

end NUMINAMATH_CALUDE_cannot_tile_modified_checkerboard_l1544_154440


namespace NUMINAMATH_CALUDE_reaction_gibbs_free_energy_change_l1544_154434

/-- The standard Gibbs free energy of formation of NaOH in kJ/mol -/
def ΔG_f_NaOH : ℝ := -381.1

/-- The standard Gibbs free energy of formation of Na₂O in kJ/mol -/
def ΔG_f_Na2O : ℝ := -378

/-- The standard Gibbs free energy of formation of H₂O (liquid) in kJ/mol -/
def ΔG_f_H2O : ℝ := -237

/-- The temperature in Kelvin -/
def T : ℝ := 298

/-- 
The standard Gibbs free energy change (ΔG°₂₉₈) for the reaction Na₂O + H₂O → 2NaOH at 298 K
is equal to -147.2 kJ/mol, given the standard Gibbs free energies of formation for NaOH, Na₂O, and H₂O.
-/
theorem reaction_gibbs_free_energy_change : 
  2 * ΔG_f_NaOH - (ΔG_f_Na2O + ΔG_f_H2O) = -147.2 := by sorry

end NUMINAMATH_CALUDE_reaction_gibbs_free_energy_change_l1544_154434


namespace NUMINAMATH_CALUDE_square_sum_given_sum_and_product_l1544_154464

theorem square_sum_given_sum_and_product (x y : ℝ) 
  (h1 : x + y = 5) (h2 : x * y = 2) : x^2 + y^2 = 21 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_given_sum_and_product_l1544_154464


namespace NUMINAMATH_CALUDE_selling_price_calculation_l1544_154453

theorem selling_price_calculation (cost_price : ℝ) (gain_percent : ℝ) (selling_price : ℝ) : 
  cost_price = 100 → gain_percent = 10 → selling_price = cost_price * (1 + gain_percent / 100) → selling_price = 110 := by
  sorry

end NUMINAMATH_CALUDE_selling_price_calculation_l1544_154453


namespace NUMINAMATH_CALUDE_total_pages_to_read_l1544_154413

def pages_read : ℕ := 113
def days_left : ℕ := 5
def pages_per_day : ℕ := 59

theorem total_pages_to_read : pages_read + days_left * pages_per_day = 408 := by
  sorry

end NUMINAMATH_CALUDE_total_pages_to_read_l1544_154413


namespace NUMINAMATH_CALUDE_ellipse_triangle_perimeter_l1544_154445

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := 3 * x^2 + 4 * y^2 = 12

-- Define the foci
def is_focus (F : ℝ × ℝ) : Prop := 
  ∃ (c : ℝ), F.1^2 + F.2^2 = c^2 ∧ c^2 = 4 - 3

-- Define a point on the ellipse
def on_ellipse (P : ℝ × ℝ) : Prop := ellipse P.1 P.2

-- Define the property of points on the ellipse
def ellipse_property (A B : ℝ × ℝ) (F1 F2 : ℝ × ℝ) : Prop :=
  on_ellipse A ∧ on_ellipse B ∧ is_focus F1 ∧ is_focus F2 →
  dist A F1 + dist A F2 = dist B F1 + dist B F2

-- Theorem statement
theorem ellipse_triangle_perimeter 
  (A B : ℝ × ℝ) (F1 F2 : ℝ × ℝ) :
  ellipse_property A B F1 F2 →
  (∃ (t : ℝ), A = F1 + t • (B - F1)) →
  dist A B + dist A F2 + dist B F2 = 8 := by
sorry

end NUMINAMATH_CALUDE_ellipse_triangle_perimeter_l1544_154445


namespace NUMINAMATH_CALUDE_largest_x_absolute_value_inequality_l1544_154488

theorem largest_x_absolute_value_inequality : 
  ∃ (x_max : ℝ), x_max = 199 ∧ 
  (∀ (x : ℝ), abs (x^2 - 4*x - 39601) ≥ abs (x^2 + 4*x - 39601) → x ≤ x_max) ∧
  abs (x_max^2 - 4*x_max - 39601) ≥ abs (x_max^2 + 4*x_max - 39601) :=
by sorry

end NUMINAMATH_CALUDE_largest_x_absolute_value_inequality_l1544_154488


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l1544_154412

/-- Given two 2D vectors a and b, where a = (2, 1) and b = (m, -1),
    and a is parallel to b, prove that m = -2. -/
theorem parallel_vectors_m_value :
  ∀ (m : ℝ),
  let a : Fin 2 → ℝ := ![2, 1]
  let b : Fin 2 → ℝ := ![m, -1]
  (∃ (k : ℝ), k ≠ 0 ∧ (∀ i, a i = k * b i)) →
  m = -2 :=
by sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l1544_154412


namespace NUMINAMATH_CALUDE_negation_of_implication_l1544_154447

theorem negation_of_implication (a b : ℝ) :
  ¬(a > b → a^2 > b^2) ↔ (a ≤ b → a^2 ≤ b^2) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l1544_154447


namespace NUMINAMATH_CALUDE_triangle_angle_A_l1544_154481

theorem triangle_angle_A (a b c : ℝ) (A B C : ℝ) (S : ℝ) :
  b = 8 →
  c = 8 * Real.sqrt 3 →
  S = 16 * Real.sqrt 3 →
  S = 1/2 * b * c * Real.sin A →
  A = π/6 ∨ A = 5*π/6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_A_l1544_154481


namespace NUMINAMATH_CALUDE_vector_collinearity_l1544_154471

/-- Given vectors a, b, and c in ℝ², prove that if k*a + b is collinear with c, then k = -1 -/
theorem vector_collinearity (a b c : ℝ × ℝ) (k : ℝ) 
    (ha : a = (1, 1)) 
    (hb : b = (-1, 0)) 
    (hc : c = (2, 1)) 
    (hcollinear : ∃ (t : ℝ), t • c = k • a + b) : 
  k = -1 := by
  sorry

end NUMINAMATH_CALUDE_vector_collinearity_l1544_154471


namespace NUMINAMATH_CALUDE_van_speed_problem_l1544_154406

theorem van_speed_problem (distance : ℝ) (original_time : ℝ) (time_factor : ℝ) 
  (h1 : distance = 600)
  (h2 : original_time = 5)
  (h3 : time_factor = 3 / 2) :
  distance / (original_time * time_factor) = 80 := by
sorry

end NUMINAMATH_CALUDE_van_speed_problem_l1544_154406


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1544_154480

/-- A function f(x) = ax + 3 has a zero point in the interval [-1, 2] -/
def has_zero_point (a : ℝ) : Prop :=
  ∃ x : ℝ, -1 ≤ x ∧ x ≤ 2 ∧ a * x + 3 = 0

/-- The condition a < -3 is sufficient but not necessary for the function to have a zero point in [-1, 2] -/
theorem sufficient_not_necessary :
  (∀ a : ℝ, a < -3 → has_zero_point a) ∧
  ¬(∀ a : ℝ, has_zero_point a → a < -3) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1544_154480


namespace NUMINAMATH_CALUDE_incorrect_step_identification_l1544_154414

theorem incorrect_step_identification :
  (2 * Real.sqrt 3 = Real.sqrt (2^2 * 3)) ∧
  (2 * Real.sqrt 3 ≠ -2 * Real.sqrt 3) ∧
  (Real.sqrt ((-2)^2 * 3) ≠ -2 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_incorrect_step_identification_l1544_154414


namespace NUMINAMATH_CALUDE_sin_315_degrees_l1544_154402

/-- Proves that sin 315° = -√2/2 -/
theorem sin_315_degrees : Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_315_degrees_l1544_154402


namespace NUMINAMATH_CALUDE_circle_areas_equal_l1544_154490

theorem circle_areas_equal (x y : Real) 
  (hx : 2 * Real.pi * x = 10 * Real.pi) 
  (hy : y / 2 = 2.5) : 
  Real.pi * x^2 = Real.pi * y^2 := by
sorry

end NUMINAMATH_CALUDE_circle_areas_equal_l1544_154490


namespace NUMINAMATH_CALUDE_terry_bottle_caps_l1544_154404

def bottle_cap_collection (num_groups : ℕ) (caps_per_group : ℕ) : ℕ :=
  num_groups * caps_per_group

theorem terry_bottle_caps : 
  bottle_cap_collection 80 7 = 560 := by
  sorry

end NUMINAMATH_CALUDE_terry_bottle_caps_l1544_154404


namespace NUMINAMATH_CALUDE_circle_equation_l1544_154407

-- Define the circle C
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the lines
def line1 (x y : ℝ) : Prop := x - y + 1 = 0
def line2 (x y : ℝ) : Prop := x + y + 1 = 0
def y_axis (x : ℝ) : Prop := x = 0

-- State the theorem
theorem circle_equation (C : Circle) : 
  (∃ x y : ℝ, line1 x y ∧ y_axis x ∧ C.center = (x, y)) →  -- Center condition
  (∃ x y : ℝ, line2 x y ∧ (x - C.center.1)^2 + (y - C.center.2)^2 = C.radius^2) →  -- Tangent condition
  ∀ x y : ℝ, (x - C.center.1)^2 + (y - C.center.2)^2 = C.radius^2 ↔ x^2 + (y-1)^2 = 2 :=
by sorry


end NUMINAMATH_CALUDE_circle_equation_l1544_154407
