import Mathlib

namespace NUMINAMATH_CALUDE_range_of_a_l2876_287617

theorem range_of_a (a : ℝ) : (∃ x : ℝ, x + a * x + a = 0) → a ∈ Set.Iic 0 ∪ Set.Ici 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2876_287617


namespace NUMINAMATH_CALUDE_consecutive_even_sum_squares_l2876_287687

theorem consecutive_even_sum_squares (a b c d : ℕ) : 
  (∃ n : ℕ, a = 2*n ∧ b = 2*n + 2 ∧ c = 2*n + 4 ∧ d = 2*n + 6) →
  (a + b + c + d = 36) →
  (a^2 + b^2 + c^2 + d^2 = 344) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_even_sum_squares_l2876_287687


namespace NUMINAMATH_CALUDE_area_of_triangle_AEC_l2876_287688

-- Define the points
variable (A B C D E : ℝ × ℝ)

-- Define the properties of the rectangle and point E
def is_rectangle (A B C D : ℝ × ℝ) : Prop := sorry

def on_segment (E C D : ℝ × ℝ) : Prop := sorry

def segment_ratio (D E C : ℝ × ℝ) (r : ℚ) : Prop := sorry

def triangle_area (A D E : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem area_of_triangle_AEC 
  (h_rectangle : is_rectangle A B C D)
  (h_on_segment : on_segment E C D)
  (h_ratio : segment_ratio D E C (3/2))
  (h_area_ADE : triangle_area A D E = 27) :
  triangle_area A E C = 18 := by sorry

end NUMINAMATH_CALUDE_area_of_triangle_AEC_l2876_287688


namespace NUMINAMATH_CALUDE_circle_equation_l2876_287691

/-- Given a circle with center (1,2) and a point (-2,6) on the circle,
    prove that its standard equation is (x-1)^2 + (y-2)^2 = 25 -/
theorem circle_equation (x y : ℝ) :
  let center := (1, 2)
  let point := (-2, 6)
  let on_circle := (point.1 - center.1)^2 + (point.2 - center.2)^2 = (x - center.1)^2 + (y - center.2)^2
  on_circle → (x - 1)^2 + (y - 2)^2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_l2876_287691


namespace NUMINAMATH_CALUDE_chocolate_topping_proof_l2876_287673

/-- Proves that adding 220 ounces of pure chocolate to the initial mixture 
    results in a 75% chocolate topping -/
theorem chocolate_topping_proof 
  (initial_total : ℝ) 
  (initial_chocolate : ℝ) 
  (initial_other : ℝ) 
  (target_percentage : ℝ) 
  (h1 : initial_total = 220)
  (h2 : initial_chocolate = 110)
  (h3 : initial_other = 110)
  (h4 : initial_total = initial_chocolate + initial_other)
  (h5 : target_percentage = 0.75) : 
  let added_chocolate : ℝ := 220
  let final_chocolate : ℝ := initial_chocolate + added_chocolate
  let final_total : ℝ := initial_total + added_chocolate
  final_chocolate / final_total = target_percentage :=
by sorry

end NUMINAMATH_CALUDE_chocolate_topping_proof_l2876_287673


namespace NUMINAMATH_CALUDE_f_properties_l2876_287697

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x)

theorem f_properties :
  (∀ x : ℝ, f (-x) = -f x) ∧
  (∀ x : ℝ, deriv f x > 0) ∧
  (∀ k : ℝ, (∀ x : ℝ, x > 0 → f x > k * x) ↔ k ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l2876_287697


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_system_l2876_287696

theorem unique_solution_quadratic_system :
  ∃! x : ℚ, (8 * x^2 + 7 * x - 1 = 0) ∧ (40 * x^2 + 89 * x - 9 = 0) ∧ (x = 1/8) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_system_l2876_287696


namespace NUMINAMATH_CALUDE_sibling_height_l2876_287609

/-- Given information about Eliza and her siblings' heights, prove the height of the sibling with unknown height -/
theorem sibling_height (total_height : ℕ) (eliza_height : ℕ) (sibling1_height : ℕ) (sibling2_height : ℕ) :
  total_height = 330 ∧
  eliza_height = 68 ∧
  sibling1_height = 66 ∧
  sibling2_height = 66 ∧
  ∃ (unknown_sibling_height last_sibling_height : ℕ),
    unknown_sibling_height = eliza_height + 2 ∧
    total_height = eliza_height + sibling1_height + sibling2_height + unknown_sibling_height + last_sibling_height →
  ∃ (unknown_sibling_height : ℕ), unknown_sibling_height = 70 :=
by sorry

end NUMINAMATH_CALUDE_sibling_height_l2876_287609


namespace NUMINAMATH_CALUDE_subtraction_preserves_inequality_l2876_287684

theorem subtraction_preserves_inequality (a b c : ℝ) : a > b → a - c > b - c := by
  sorry

end NUMINAMATH_CALUDE_subtraction_preserves_inequality_l2876_287684


namespace NUMINAMATH_CALUDE_perpendicular_lines_coefficient_l2876_287668

/-- Two lines in the form y = mx + b are perpendicular if and only if the product of their slopes is -1 -/
axiom perpendicular_lines_slope_product (m₁ m₂ : ℝ) : 
  m₁ * m₂ = -1 ↔ (∃ (b₁ b₂ : ℝ), ∀ (x y : ℝ), y = m₁ * x + b₁ ↔ y = -1/m₂ * x + b₂)

/-- Given two lines ax + y + 2 = 0 and 3x - y - 2 = 0 that are perpendicular, prove that a = 2/3 -/
theorem perpendicular_lines_coefficient (a : ℝ) :
  (∀ (x y : ℝ), y = -a * x - 2 ↔ y = 3 * x - 2) →
  a = 2/3 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_coefficient_l2876_287668


namespace NUMINAMATH_CALUDE_smallest_n_for_book_pricing_l2876_287674

theorem smallest_n_for_book_pricing : 
  ∀ n : ℕ+, (∃ x : ℕ+, (105 * x : ℕ) = 100 * n) → n ≥ 21 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_book_pricing_l2876_287674


namespace NUMINAMATH_CALUDE_intersection_implies_x_zero_l2876_287681

def A : Set ℝ := {0, 1, 2, 4, 5}
def B (x : ℝ) : Set ℝ := {x-2, x, x+2}

theorem intersection_implies_x_zero (x : ℝ) (h : A ∩ B x = {0, 2}) : x = 0 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_x_zero_l2876_287681


namespace NUMINAMATH_CALUDE_angle_C_is_pi_third_area_of_triangle_l2876_287627

namespace TriangleProof

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The determinant condition from the problem -/
def determinant_condition (t : Triangle) : Prop :=
  2 * t.c * Real.sin t.C = (2 * t.a - t.b) * Real.sin t.A + (2 * t.b - t.a) * Real.sin t.B

/-- Theorem 1: If the determinant condition holds, then C = π/3 -/
theorem angle_C_is_pi_third (t : Triangle) 
  (h : determinant_condition t) : t.C = Real.pi / 3 := by
  sorry

/-- Theorem 2: Area of the triangle under given conditions -/
theorem area_of_triangle (t : Triangle) 
  (h1 : Real.sin t.A = 4/5)
  (h2 : t.C = 2 * Real.pi / 3)
  (h3 : t.c = Real.sqrt 3) : 
  (1/2) * t.a * t.c * Real.sin t.B = (18 - 8 * Real.sqrt 3) / 25 := by
  sorry

end TriangleProof

end NUMINAMATH_CALUDE_angle_C_is_pi_third_area_of_triangle_l2876_287627


namespace NUMINAMATH_CALUDE_class_size_l2876_287604

/-- The position of Xiao Ming from the front of the line -/
def position_from_front : ℕ := 23

/-- The position of Xiao Ming from the back of the line -/
def position_from_back : ℕ := 23

/-- The total number of students in the class -/
def total_students : ℕ := position_from_front + position_from_back - 1

theorem class_size :
  total_students = 45 :=
sorry

end NUMINAMATH_CALUDE_class_size_l2876_287604


namespace NUMINAMATH_CALUDE_expand_polynomial_l2876_287653

theorem expand_polynomial (a b : ℝ) : (a - b) * (a + b) * (a^2 - b^2) = a^4 - 2*a^2*b^2 + b^4 := by
  sorry

end NUMINAMATH_CALUDE_expand_polynomial_l2876_287653


namespace NUMINAMATH_CALUDE_product_rule_l2876_287643

theorem product_rule (b a : ℤ) (h : 0 ≤ a ∧ a < 10) : 
  (10 * b + a) * (10 * b + 10 - a) = 100 * b * (b + 1) + a * (10 - a) := by
  sorry

end NUMINAMATH_CALUDE_product_rule_l2876_287643


namespace NUMINAMATH_CALUDE_sqrt_sum_difference_equality_l2876_287616

theorem sqrt_sum_difference_equality : 
  Real.sqrt 27 + Real.sqrt (1/3) - Real.sqrt 2 * Real.sqrt 6 = (4 * Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_difference_equality_l2876_287616


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l2876_287635

def U : Set Int := {-2, -1, 0, 1, 2}

def A : Set Int := {x | ∃ n : Int, x = 2 / (n - 1) ∧ x ∈ U}

theorem complement_of_A_in_U :
  (U \ A) = {0} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l2876_287635


namespace NUMINAMATH_CALUDE_race_heartbeats_l2876_287659

/-- Calculates the total number of heartbeats during a race -/
def total_heartbeats (heart_rate : ℕ) (pace : ℕ) (distance : ℕ) : ℕ :=
  heart_rate * pace * distance

/-- Proves that the total number of heartbeats during the race is 27000 -/
theorem race_heartbeats :
  let heart_rate : ℕ := 180  -- heartbeats per minute
  let pace : ℕ := 3          -- minutes per kilometer
  let distance : ℕ := 50     -- kilometers
  total_heartbeats heart_rate pace distance = 27000 := by
  sorry


end NUMINAMATH_CALUDE_race_heartbeats_l2876_287659


namespace NUMINAMATH_CALUDE_pauline_total_spend_l2876_287629

/-- The total amount Pauline will spend, including sales tax -/
def total_amount (pre_tax_amount : ℝ) (tax_rate : ℝ) : ℝ :=
  pre_tax_amount * (1 + tax_rate)

/-- Proof that Pauline will spend $162 on all items, including sales tax -/
theorem pauline_total_spend :
  total_amount 150 0.08 = 162 := by
  sorry

end NUMINAMATH_CALUDE_pauline_total_spend_l2876_287629


namespace NUMINAMATH_CALUDE_angle_range_for_point_in_first_quadrant_l2876_287630

def is_in_first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

theorem angle_range_for_point_in_first_quadrant (α : ℝ) :
  0 ≤ α ∧ α ≤ 2 * Real.pi →
  is_in_first_quadrant (Real.tan α) (Real.sin α - Real.cos α) →
  (α ∈ Set.Ioo (Real.pi / 4) (Real.pi / 2)) ∨ (α ∈ Set.Ioo Real.pi (5 * Real.pi / 4)) :=
by sorry

end NUMINAMATH_CALUDE_angle_range_for_point_in_first_quadrant_l2876_287630


namespace NUMINAMATH_CALUDE_right_triangle_with_perimeter_80_l2876_287677

theorem right_triangle_with_perimeter_80 :
  ∀ a b c : ℕ+,
  a^2 + b^2 = c^2 →
  a + b + c = 80 →
  (a = 30 ∧ b = 16 ∧ c = 34) ∨ 
  (a = 16 ∧ b = 30 ∧ c = 34) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_with_perimeter_80_l2876_287677


namespace NUMINAMATH_CALUDE_xiaoming_money_problem_l2876_287686

theorem xiaoming_money_problem (price_left : ℕ) (price_right : ℕ) :
  price_right = price_left - 1 →
  12 * price_left = 14 * price_right →
  12 * price_left = 84 :=
by sorry

end NUMINAMATH_CALUDE_xiaoming_money_problem_l2876_287686


namespace NUMINAMATH_CALUDE_diagonal_passes_through_600_cubes_l2876_287606

/-- The number of cubes an internal diagonal passes through in a rectangular solid -/
def cubes_passed_by_diagonal (l w h : ℕ) : ℕ :=
  l + w + h - (Nat.gcd l w + Nat.gcd w h + Nat.gcd h l) + Nat.gcd l (Nat.gcd w h)

/-- Theorem: An internal diagonal of a 120 × 280 × 360 rectangular solid passes through 600 cubes -/
theorem diagonal_passes_through_600_cubes :
  cubes_passed_by_diagonal 120 280 360 = 600 := by
  sorry

end NUMINAMATH_CALUDE_diagonal_passes_through_600_cubes_l2876_287606


namespace NUMINAMATH_CALUDE_exist_four_distinct_numbers_perfect_squares_l2876_287610

theorem exist_four_distinct_numbers_perfect_squares : 
  ∃ (a b c d : ℕ), 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    ∃ (m n : ℕ), 
      a^2 + 2*c*d + b^2 = m^2 ∧
      c^2 + 2*a*b + d^2 = n^2 :=
by sorry

end NUMINAMATH_CALUDE_exist_four_distinct_numbers_perfect_squares_l2876_287610


namespace NUMINAMATH_CALUDE_fraction_equality_sum_l2876_287613

theorem fraction_equality_sum (p q : ℚ) : p / q = 2 / 7 → 2 * p + q = 11 * p / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_sum_l2876_287613


namespace NUMINAMATH_CALUDE_relation_between_exponents_l2876_287670

/-- Given real numbers a, b, c, d, x, y, p satisfying certain equations,
    prove that y = (3 * p^2) / 2 -/
theorem relation_between_exponents
  (a b c d x y p : ℝ)
  (h1 : a^x = c^(3*p))
  (h2 : c^(3*p) = b^2)
  (h3 : c^y = b^p)
  (h4 : b^p = d^3)
  : y = (3 * p^2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_relation_between_exponents_l2876_287670


namespace NUMINAMATH_CALUDE_fermat_divisibility_l2876_287660

/-- Fermat number -/
def F (n : ℕ) : ℕ := 2^(2^n) + 1

/-- Theorem: For all natural numbers n, F_n divides 2^F_n - 2 -/
theorem fermat_divisibility (n : ℕ) : (F n) ∣ (2^(F n) - 2) := by
  sorry

end NUMINAMATH_CALUDE_fermat_divisibility_l2876_287660


namespace NUMINAMATH_CALUDE_cycling_distance_l2876_287611

/-- Proves that cycling at a constant rate of 4 miles per hour for 2 hours results in a total distance of 8 miles. -/
theorem cycling_distance (rate : ℝ) (time : ℝ) (distance : ℝ) : 
  rate = 4 → time = 2 → distance = rate * time → distance = 8 := by
  sorry

#check cycling_distance

end NUMINAMATH_CALUDE_cycling_distance_l2876_287611


namespace NUMINAMATH_CALUDE_expression_simplification_l2876_287633

theorem expression_simplification (x : ℤ) (hx : x ≠ 0) :
  (x^3 - 3*x^2*(x+2) + 4*x*(x+2)^2 - (x+2)^3 + 2) / (x * (x+2)) = 2 / (x * (x+2)) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2876_287633


namespace NUMINAMATH_CALUDE_discount_rate_for_given_profit_l2876_287650

/-- Given a product with cost price, marked price, and desired profit percentage,
    calculate the discount rate needed to achieve the desired profit. -/
def calculate_discount_rate (cost_price marked_price profit_percentage : ℚ) : ℚ :=
  let selling_price := cost_price * (1 + profit_percentage / 100)
  selling_price / marked_price

theorem discount_rate_for_given_profit :
  let cost_price : ℚ := 200
  let marked_price : ℚ := 300
  let profit_percentage : ℚ := 20
  calculate_discount_rate cost_price marked_price profit_percentage = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_discount_rate_for_given_profit_l2876_287650


namespace NUMINAMATH_CALUDE_tom_completion_time_l2876_287699

/-- Represents the duration of a combined BS and Ph.D. program -/
structure Program where
  bs_duration : ℕ
  phd_duration : ℕ

/-- Calculates the time taken by a student to complete the program given a completion ratio -/
def completion_time (p : Program) (ratio : ℚ) : ℚ :=
  ratio * (p.bs_duration + p.phd_duration)

theorem tom_completion_time :
  let p : Program := { bs_duration := 3, phd_duration := 5 }
  let ratio : ℚ := 3/4
  completion_time p ratio = 6 := by
  sorry

end NUMINAMATH_CALUDE_tom_completion_time_l2876_287699


namespace NUMINAMATH_CALUDE_lines_coincide_l2876_287667

/-- If three lines y = kx + m, y = mx + n, and y = nx + k have a common point, then k = m = n -/
theorem lines_coincide (k m n : ℝ) (x y : ℝ) 
  (h1 : y = k * x + m)
  (h2 : y = m * x + n)
  (h3 : y = n * x + k) :
  k = m ∧ m = n := by
  sorry

end NUMINAMATH_CALUDE_lines_coincide_l2876_287667


namespace NUMINAMATH_CALUDE_cake_pieces_count_l2876_287679

/-- Given 50 friends and 3 pieces of cake per friend, prove that the total number of cake pieces is 150. -/
theorem cake_pieces_count (num_friends : ℕ) (pieces_per_friend : ℕ) : 
  num_friends = 50 → pieces_per_friend = 3 → num_friends * pieces_per_friend = 150 := by
  sorry


end NUMINAMATH_CALUDE_cake_pieces_count_l2876_287679


namespace NUMINAMATH_CALUDE_pencil_difference_l2876_287618

/-- The number of pencils Paige has in her desk -/
def pencils_in_desk : ℕ := 2

/-- The number of pencils Paige has in her backpack -/
def pencils_in_backpack : ℕ := 2

/-- The number of pencils Paige has at home -/
def pencils_at_home : ℕ := 15

/-- The difference between the number of pencils at Paige's home and in Paige's backpack -/
theorem pencil_difference : pencils_at_home - pencils_in_backpack = 13 := by
  sorry

end NUMINAMATH_CALUDE_pencil_difference_l2876_287618


namespace NUMINAMATH_CALUDE_infinite_perfect_squares_in_sequence_l2876_287671

theorem infinite_perfect_squares_in_sequence :
  ∃ f : ℕ → ℕ × ℕ, 
    (∀ i : ℕ, (f i).1^2 = 1 + 17 * (f i).2^2) ∧ 
    (∀ i j : ℕ, i ≠ j → f i ≠ f j) := by
  sorry

end NUMINAMATH_CALUDE_infinite_perfect_squares_in_sequence_l2876_287671


namespace NUMINAMATH_CALUDE_jens_ducks_l2876_287607

theorem jens_ducks (chickens ducks : ℕ) : 
  ducks = 4 * chickens + 10 →
  chickens + ducks = 185 →
  ducks = 150 := by
sorry

end NUMINAMATH_CALUDE_jens_ducks_l2876_287607


namespace NUMINAMATH_CALUDE_ball_count_proof_l2876_287645

theorem ball_count_proof (a : ℕ) (red_balls : ℕ) (probability : ℚ) : 
  red_balls = 5 → probability = 1/5 → a = 25 := by
  sorry

end NUMINAMATH_CALUDE_ball_count_proof_l2876_287645


namespace NUMINAMATH_CALUDE_negation_of_proposition_negation_of_specific_proposition_l2876_287647

theorem negation_of_proposition (P : ℝ → Prop) :
  (¬∀ x : ℝ, P x) ↔ (∃ x : ℝ, ¬P x) :=
by sorry

theorem negation_of_specific_proposition :
  (¬∀ x : ℝ, x^2 + 2*x + 3 ≥ 0) ↔ (∃ x : ℝ, x^2 + 2*x + 3 < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_negation_of_specific_proposition_l2876_287647


namespace NUMINAMATH_CALUDE_jane_lost_twenty_points_l2876_287634

/-- Represents the card game scenario --/
structure CardGame where
  pointsPerWin : ℕ
  totalRounds : ℕ
  finalPoints : ℕ

/-- Calculates the points lost in the card game --/
def pointsLost (game : CardGame) : ℕ :=
  game.pointsPerWin * game.totalRounds - game.finalPoints

/-- Theorem stating that Jane lost 20 points --/
theorem jane_lost_twenty_points :
  let game : CardGame := {
    pointsPerWin := 10,
    totalRounds := 8,
    finalPoints := 60
  }
  pointsLost game = 20 := by
  sorry


end NUMINAMATH_CALUDE_jane_lost_twenty_points_l2876_287634


namespace NUMINAMATH_CALUDE_nina_travel_period_l2876_287639

/-- Nina's travel pattern over two months -/
def two_month_distance : ℕ := 400 + 800

/-- Total distance Nina wants to travel -/
def total_distance : ℕ := 14400

/-- Number of two-month periods needed to reach the total distance -/
def num_two_month_periods : ℕ := total_distance / two_month_distance

/-- Duration of Nina's travel period in months -/
def travel_period_months : ℕ := num_two_month_periods * 2

/-- Theorem stating that Nina's travel period is 24 months -/
theorem nina_travel_period :
  travel_period_months = 24 :=
sorry

end NUMINAMATH_CALUDE_nina_travel_period_l2876_287639


namespace NUMINAMATH_CALUDE_cost_price_calculation_l2876_287664

/-- 
Given a product with:
- Marked price of 1100 yuan
- Sold at 80% of the marked price
- Makes a 10% profit

Prove that the cost price is 800 yuan
-/
theorem cost_price_calculation (marked_price : ℝ) (discount_rate : ℝ) (profit_rate : ℝ) 
  (h1 : marked_price = 1100)
  (h2 : discount_rate = 0.8)
  (h3 : profit_rate = 0.1) :
  marked_price * discount_rate = (1 + profit_rate) * 800 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l2876_287664


namespace NUMINAMATH_CALUDE_eleventh_term_is_768_l2876_287612

/-- A geometric sequence with given conditions -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, ∃ q : ℝ, a (n + 1) = q * a n
  a_4 : a 4 = 6
  a_7 : a 7 = 48

/-- The 11th term of the geometric sequence is 768 -/
theorem eleventh_term_is_768 (seq : GeometricSequence) : seq.a 11 = 768 := by
  sorry

end NUMINAMATH_CALUDE_eleventh_term_is_768_l2876_287612


namespace NUMINAMATH_CALUDE_symmetrical_shape_three_equal_parts_l2876_287619

/-- A symmetrical 2D shape -/
structure SymmetricalShape where
  area : ℝ
  height : ℝ
  width : ℝ
  is_symmetrical : Bool

/-- A straight cut on the shape -/
inductive Cut
  | Vertical : ℝ → Cut  -- position along width
  | Horizontal : ℝ → Cut  -- position along height

/-- Result of applying cuts to a shape -/
def apply_cuts (shape : SymmetricalShape) (cuts : List Cut) : List ℝ :=
  sorry

theorem symmetrical_shape_three_equal_parts (shape : SymmetricalShape) :
  shape.is_symmetrical →
  ∃ (vertical_cut : Cut) (horizontal_cut : Cut),
    vertical_cut = Cut.Vertical (shape.width / 2) ∧
    horizontal_cut = Cut.Horizontal (shape.height / 3) ∧
    apply_cuts shape [vertical_cut, horizontal_cut] = [shape.area / 3, shape.area / 3, shape.area / 3] :=
  sorry

end NUMINAMATH_CALUDE_symmetrical_shape_three_equal_parts_l2876_287619


namespace NUMINAMATH_CALUDE_zeros_before_first_nonzero_digit_l2876_287661

theorem zeros_before_first_nonzero_digit (n : ℕ) (d : ℕ) (h : n = 5 ∧ d = 1600) :
  (n : ℚ) / d = 0.003125 :=
sorry

end NUMINAMATH_CALUDE_zeros_before_first_nonzero_digit_l2876_287661


namespace NUMINAMATH_CALUDE_impossible_three_shell_piles_l2876_287600

/-- Represents the number of seashells at step n -/
def S (n : ℕ) : ℤ := 637 - n

/-- Represents the number of piles at step n -/
def P (n : ℕ) : ℤ := 1 + n

/-- Theorem stating that it's impossible to end up with only piles of exactly three seashells -/
theorem impossible_three_shell_piles : ¬ ∃ n : ℕ, S n = 3 * P n ∧ S n > 0 := by
  sorry

end NUMINAMATH_CALUDE_impossible_three_shell_piles_l2876_287600


namespace NUMINAMATH_CALUDE_condition1_coordinates_condition2_coordinates_l2876_287662

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Given point A with coordinates dependent on parameter a -/
def A (a : ℝ) : Point := ⟨3*a + 2, 2*a - 4⟩

/-- Fixed point B -/
def B : Point := ⟨3, 4⟩

/-- Condition 1: Line AB is parallel to x-axis -/
def parallel_to_x_axis (A B : Point) : Prop :=
  A.y = B.y

/-- Condition 2: Distance from A to both coordinate axes is equal -/
def equal_distance_to_axes (A : Point) : Prop :=
  |A.x| = |A.y|

/-- Theorem for Condition 1 -/
theorem condition1_coordinates :
  ∃ a : ℝ, parallel_to_x_axis (A a) B → A a = ⟨14, 4⟩ := by sorry

/-- Theorem for Condition 2 -/
theorem condition2_coordinates :
  ∃ a : ℝ, equal_distance_to_axes (A a) → 
    (A a = ⟨-16, -16⟩ ∨ A a = ⟨3.2, -3.2⟩) := by sorry

end NUMINAMATH_CALUDE_condition1_coordinates_condition2_coordinates_l2876_287662


namespace NUMINAMATH_CALUDE_base_difference_equals_174_l2876_287603

def base_to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.foldl (fun acc d => acc * base + d) 0

def base9_to_decimal (n : Nat) : Nat :=
  base_to_decimal [3, 2, 4] 9

def base6_to_decimal (n : Nat) : Nat :=
  base_to_decimal [2, 3, 1] 6

theorem base_difference_equals_174 :
  base9_to_decimal 324 - base6_to_decimal 231 = 174 := by
  sorry

end NUMINAMATH_CALUDE_base_difference_equals_174_l2876_287603


namespace NUMINAMATH_CALUDE_percentage_relation_l2876_287651

theorem percentage_relation (X A B : ℝ) (hA : A = 0.05 * X) (hB : B = 0.25 * X) :
  A = 0.2 * B := by sorry

end NUMINAMATH_CALUDE_percentage_relation_l2876_287651


namespace NUMINAMATH_CALUDE_complement_equal_l2876_287672

/-- The complement of an angle is the angle that, when added to the original angle, results in a right angle (90 degrees). -/
def complement (α : ℝ) : ℝ := 90 - α

/-- For any angle, its complement is equal to itself. -/
theorem complement_equal (α : ℝ) : complement α = complement α := by sorry

end NUMINAMATH_CALUDE_complement_equal_l2876_287672


namespace NUMINAMATH_CALUDE_only_vinyl_chloride_and_benzene_planar_l2876_287685

/-- Represents an organic compound -/
inductive OrganicCompound
| Propylene
| VinylChloride
| Benzene
| Toluene

/-- Predicate to check if all atoms in a compound are on the same plane -/
def all_atoms_on_same_plane (c : OrganicCompound) : Prop :=
  match c with
  | OrganicCompound.Propylene => False
  | OrganicCompound.VinylChloride => True
  | OrganicCompound.Benzene => True
  | OrganicCompound.Toluene => False

/-- Theorem stating that only vinyl chloride and benzene have all atoms on the same plane -/
theorem only_vinyl_chloride_and_benzene_planar :
  ∀ c : OrganicCompound, all_atoms_on_same_plane c ↔ (c = OrganicCompound.VinylChloride ∨ c = OrganicCompound.Benzene) :=
by
  sorry


end NUMINAMATH_CALUDE_only_vinyl_chloride_and_benzene_planar_l2876_287685


namespace NUMINAMATH_CALUDE_smallest_digit_divisible_by_nine_l2876_287656

theorem smallest_digit_divisible_by_nine :
  ∃ (d : ℕ), d < 10 ∧ 
    (∀ (x : ℕ), x < d → ¬(528000 + x * 100 + 46) % 9 = 0) ∧
    (528000 + d * 100 + 46) % 9 = 0 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_digit_divisible_by_nine_l2876_287656


namespace NUMINAMATH_CALUDE_floor_neg_seven_fourths_l2876_287657

theorem floor_neg_seven_fourths : ⌊(-7 : ℚ) / 4⌋ = -2 := by sorry

end NUMINAMATH_CALUDE_floor_neg_seven_fourths_l2876_287657


namespace NUMINAMATH_CALUDE_jose_bottle_caps_l2876_287614

def initial_bottle_caps : ℕ := 26
def additional_bottle_caps : ℕ := 13

theorem jose_bottle_caps : 
  initial_bottle_caps + additional_bottle_caps = 39 := by
  sorry

end NUMINAMATH_CALUDE_jose_bottle_caps_l2876_287614


namespace NUMINAMATH_CALUDE_intersection_equals_open_interval_l2876_287683

-- Define sets A and B
def A : Set ℝ := {x | (4*x - 3)*(x + 3) < 0}
def B : Set ℝ := {x | 2*x > 1}

-- Define the open interval (1/2, 3/4)
def openInterval : Set ℝ := {x | 1/2 < x ∧ x < 3/4}

-- Theorem statement
theorem intersection_equals_open_interval : A ∩ B = openInterval := by
  sorry

end NUMINAMATH_CALUDE_intersection_equals_open_interval_l2876_287683


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l2876_287626

theorem min_value_expression (x : ℝ) : 
  Real.sqrt (x^2 + (2 - x)^2) + Real.sqrt ((2 - x)^2 + (2 + x)^2) ≥ 2 * Real.sqrt 5 := by
  sorry

theorem min_value_achievable : 
  ∃ x : ℝ, Real.sqrt (x^2 + (2 - x)^2) + Real.sqrt ((2 - x)^2 + (2 + x)^2) = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l2876_287626


namespace NUMINAMATH_CALUDE_toothpicks_at_250_l2876_287676

/-- Calculates the number of toothpicks at a given stage -/
def toothpicks (stage : ℕ) : ℕ :=
  if stage = 0 then 0
  else if stage % 50 = 0 then 2 * toothpicks (stage - 1)
  else if stage = 1 then 5
  else toothpicks (stage - 1) + 5

/-- The number of toothpicks at the 250th stage is 15350 -/
theorem toothpicks_at_250 : toothpicks 250 = 15350 := by
  sorry

#eval toothpicks 250  -- This line is optional, for verification purposes

end NUMINAMATH_CALUDE_toothpicks_at_250_l2876_287676


namespace NUMINAMATH_CALUDE_linear_function_properties_l2876_287690

/-- Linear function definition -/
def linear_function (m : ℝ) (x : ℝ) : ℝ := (2*m + 1)*x + m - 2

theorem linear_function_properties :
  ∀ m : ℝ,
  (∀ x, linear_function m x = 0 → x = 0) → m = 2 ∧
  (linear_function m 0 = -3) → m = -1 ∧
  (∀ x, ∃ k, linear_function m x = x + k) → m = 0 ∧
  (∀ x, x < 0 → linear_function m x > 0) → -1/2 < m ∧ m < 2 :=
by sorry

end NUMINAMATH_CALUDE_linear_function_properties_l2876_287690


namespace NUMINAMATH_CALUDE_condition_relationship_l2876_287680

theorem condition_relationship : 
  let A := {x : ℝ | 0 < x ∧ x < 5}
  let B := {x : ℝ | |x - 2| < 3}
  (∀ x ∈ A, x ∈ B) ∧ (∃ x ∈ B, x ∉ A) := by sorry

end NUMINAMATH_CALUDE_condition_relationship_l2876_287680


namespace NUMINAMATH_CALUDE_polynomial_simplification_l2876_287615

theorem polynomial_simplification (r : ℝ) : 
  (2 * r^3 + r^2 + 4*r - 3) - (r^3 + r^2 + 6*r - 8) = r^3 - 2*r + 5 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2876_287615


namespace NUMINAMATH_CALUDE_min_distance_to_line_l2876_287693

/-- The minimum distance from the origin to the line x + y - 4 = 0 is 2√2 -/
theorem min_distance_to_line : 
  let line := {p : ℝ × ℝ | p.1 + p.2 = 4}
  ∀ p ∈ line, Real.sqrt ((0 - p.1)^2 + (0 - p.2)^2) ≥ 2 * Real.sqrt 2 ∧
  ∃ q ∈ line, Real.sqrt ((0 - q.1)^2 + (0 - q.2)^2) = 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_distance_to_line_l2876_287693


namespace NUMINAMATH_CALUDE_even_function_implies_a_equals_four_l2876_287605

/-- Given that f(x) = (x + a)(x - 4) is an even function, prove that a = 4 --/
theorem even_function_implies_a_equals_four (a : ℝ) :
  (∀ x : ℝ, (x + a) * (x - 4) = (-x + a) * (-x - 4)) →
  a = 4 := by
sorry

end NUMINAMATH_CALUDE_even_function_implies_a_equals_four_l2876_287605


namespace NUMINAMATH_CALUDE_shoes_sold_day1_l2876_287698

/-- Represents the sales data for a shoe store --/
structure ShoeSales where
  shoe_price : ℕ
  boot_price : ℕ
  day1_shoes : ℕ
  day1_boots : ℕ
  day2_shoes : ℕ
  day2_boots : ℕ

/-- Theorem stating the number of shoes sold on day 1 given the sales conditions --/
theorem shoes_sold_day1 (s : ShoeSales) : 
  s.boot_price = s.shoe_price + 15 →
  s.day1_shoes * s.shoe_price + s.day1_boots * s.boot_price = 460 →
  s.day2_shoes * s.shoe_price + s.day2_boots * s.boot_price = 560 →
  s.day1_boots = 16 →
  s.day2_shoes = 8 →
  s.day2_boots = 32 →
  s.day1_shoes = 94 := by
  sorry

#check shoes_sold_day1

end NUMINAMATH_CALUDE_shoes_sold_day1_l2876_287698


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l2876_287638

def geometric_sequence (a : ℕ → ℝ) := ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_ratio (a : ℕ → ℝ) (h1 : geometric_sequence a) 
  (h2 : a 1 + a 4 = 18) (h3 : a 2 * a 3 = 32) : 
  ∃ q : ℝ, (q = 1/2 ∨ q = 2) ∧ ∀ n : ℕ, a (n + 1) = q * a n :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l2876_287638


namespace NUMINAMATH_CALUDE_stonewall_band_max_members_l2876_287665

theorem stonewall_band_max_members :
  ∃ (max : ℕ),
    (∀ n : ℕ, (30 * n) % 34 = 2 → 30 * n < 1500 → 30 * n ≤ max) ∧
    (∃ n : ℕ, (30 * n) % 34 = 2 ∧ 30 * n < 1500 ∧ 30 * n = max) ∧
    max = 1260 := by
  sorry

end NUMINAMATH_CALUDE_stonewall_band_max_members_l2876_287665


namespace NUMINAMATH_CALUDE_function_range_lower_bound_l2876_287637

theorem function_range_lower_bound (n : ℕ) (f : ℤ → Fin n) 
  (h : ∀ (x y : ℤ), |x - y| ∈ ({2, 3, 5} : Set ℤ) → f x ≠ f y) : 
  n ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_function_range_lower_bound_l2876_287637


namespace NUMINAMATH_CALUDE_recycling_points_l2876_287694

/-- The number of pounds needed to recycle to earn one point -/
def poundsPerPoint (gwenPounds friendsPounds totalPoints : ℕ) : ℚ :=
  (gwenPounds + friendsPounds : ℚ) / totalPoints

theorem recycling_points (gwenPounds friendsPounds totalPoints : ℕ) 
  (h1 : gwenPounds = 5)
  (h2 : friendsPounds = 13)
  (h3 : totalPoints = 6) :
  poundsPerPoint gwenPounds friendsPounds totalPoints = 3 := by
  sorry

end NUMINAMATH_CALUDE_recycling_points_l2876_287694


namespace NUMINAMATH_CALUDE_rectangle_max_area_l2876_287652

/-- A rectangle with integer sides and perimeter 80 has a maximum area of 400. -/
theorem rectangle_max_area :
  ∀ l w : ℕ,
  l > 0 → w > 0 →
  2 * (l + w) = 80 →
  ∀ l' w' : ℕ,
  l' > 0 → w' > 0 →
  2 * (l' + w') = 80 →
  l * w ≤ 400 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_max_area_l2876_287652


namespace NUMINAMATH_CALUDE_function_bounds_l2876_287682

theorem function_bounds (x y z : ℝ) 
  (h1 : -1 ≤ 2*x + y - z ∧ 2*x + y - z ≤ 8)
  (h2 : 2 ≤ x - y + z ∧ x - y + z ≤ 9)
  (h3 : -3 ≤ x + 2*y - z ∧ x + 2*y - z ≤ 7) :
  -6 ≤ 7*x + 5*y - 2*z ∧ 7*x + 5*y - 2*z ≤ 47 := by
sorry

end NUMINAMATH_CALUDE_function_bounds_l2876_287682


namespace NUMINAMATH_CALUDE_ninth_term_is_negative_256_l2876_287631

/-- A geometric sequence with specific properties -/
structure GeometricSequence where
  a : ℕ → ℤ
  is_geometric : ∀ n : ℕ, ∃ q : ℤ, a (n + 1) = a n * q
  a2a5 : a 2 * a 5 = -32
  a3a4_sum : a 3 + a 4 = 4

/-- The 9th term of the geometric sequence is -256 -/
theorem ninth_term_is_negative_256 (seq : GeometricSequence) : seq.a 9 = -256 := by
  sorry

end NUMINAMATH_CALUDE_ninth_term_is_negative_256_l2876_287631


namespace NUMINAMATH_CALUDE_total_seats_is_28_l2876_287675

/-- The number of students per bus -/
def students_per_bus : ℝ := 14.0

/-- The number of buses -/
def number_of_buses : ℝ := 2.0

/-- The total number of seats taken up by students -/
def total_seats : ℝ := students_per_bus * number_of_buses

/-- Theorem stating that the total number of seats taken up by students is 28 -/
theorem total_seats_is_28 : total_seats = 28 := by
  sorry

end NUMINAMATH_CALUDE_total_seats_is_28_l2876_287675


namespace NUMINAMATH_CALUDE_divisibility_implies_equality_l2876_287663

theorem divisibility_implies_equality (a b : ℕ) :
  (4 * a * b - 1) ∣ (4 * a^2 - 1)^2 → a = b := by
  sorry

end NUMINAMATH_CALUDE_divisibility_implies_equality_l2876_287663


namespace NUMINAMATH_CALUDE_intersection_probability_l2876_287642

-- Define the probability measure q
variable (q : Set ℝ → ℝ)

-- Define events g and h
variable (g h : Set ℝ)

-- Define the conditions
variable (hg : q g = 0.30)
variable (hh : q h = 0.9)
variable (hgh : q (g ∩ h) / q h = 1 / 3)
variable (hhg : q (g ∩ h) / q g = 1 / 3)

-- The theorem to prove
theorem intersection_probability : q (g ∩ h) = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_probability_l2876_287642


namespace NUMINAMATH_CALUDE_largest_A_k_l2876_287632

def A (k : ℕ) : ℝ := (Nat.choose 1000 k) * (0.2 ^ k)

theorem largest_A_k : 
  ∃ (k : ℕ), k = 166 ∧ 
  (∀ (j : ℕ), j ≤ 1000 → A k ≥ A j) := by
sorry

end NUMINAMATH_CALUDE_largest_A_k_l2876_287632


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l2876_287601

theorem fractional_equation_solution :
  ∃ x : ℝ, (3 / (x + 1) = 2 / (x - 1)) ∧ x = 5 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l2876_287601


namespace NUMINAMATH_CALUDE_pond_soil_volume_l2876_287640

/-- The volume of soil extracted from a rectangular pond -/
def soil_volume (length width depth : ℝ) : ℝ :=
  length * width * depth

/-- Theorem: The volume of soil extracted from a rectangular pond
    with dimensions 20 m × 15 m × 5 m is 1500 cubic meters -/
theorem pond_soil_volume :
  soil_volume 20 15 5 = 1500 := by
  sorry

end NUMINAMATH_CALUDE_pond_soil_volume_l2876_287640


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l2876_287692

theorem absolute_value_inequality (x : ℝ) :
  (|x + 1| - |x - 3| ≥ 2) ↔ (x ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l2876_287692


namespace NUMINAMATH_CALUDE_largest_non_sum_36_composite_l2876_287655

/-- A function that checks if a number is composite -/
def is_composite (n : ℕ) : Prop :=
  ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

/-- A function that checks if a number can be expressed as the sum of a positive integral multiple of 36 and a positive composite integer -/
def is_sum_of_multiple_36_and_composite (n : ℕ) : Prop :=
  ∃ k m, k > 0 ∧ is_composite m ∧ n = 36 * k + m

/-- The theorem stating that 145 is the largest positive integer that cannot be expressed as the sum of a positive integral multiple of 36 and a positive composite integer -/
theorem largest_non_sum_36_composite : 
  (∀ n : ℕ, n > 145 → is_sum_of_multiple_36_and_composite n) ∧
  ¬is_sum_of_multiple_36_and_composite 145 :=
sorry

end NUMINAMATH_CALUDE_largest_non_sum_36_composite_l2876_287655


namespace NUMINAMATH_CALUDE_congruence_solution_l2876_287602

theorem congruence_solution (n : ℤ) : 
  (15 * n) % 47 = 9 % 47 → n % 47 = 10 % 47 := by
sorry

end NUMINAMATH_CALUDE_congruence_solution_l2876_287602


namespace NUMINAMATH_CALUDE_det_special_matrix_l2876_287666

/-- The determinant of the matrix [[x+2, x, x], [x, x+2, x], [x, x, x+2]] is equal to 8x + 8 for any real number x. -/
theorem det_special_matrix (x : ℝ) : 
  Matrix.det !![x + 2, x, x; x, x + 2, x; x, x, x + 2] = 8 * x + 8 := by
  sorry

end NUMINAMATH_CALUDE_det_special_matrix_l2876_287666


namespace NUMINAMATH_CALUDE_line_moved_down_l2876_287658

/-- The equation of a line obtained by moving y = 2x down 3 units -/
def moved_line (x y : ℝ) : Prop := y = 2 * x - 3

/-- The original line equation -/
def original_line (x y : ℝ) : Prop := y = 2 * x

/-- Moving a line down by a certain number of units subtracts that number from the y-coordinate -/
axiom move_down (a b : ℝ) : ∀ x y, original_line x y → moved_line x (y - b) → b = 3

theorem line_moved_down : 
  ∀ x y, original_line x y → moved_line x (y - 3) :=
sorry

end NUMINAMATH_CALUDE_line_moved_down_l2876_287658


namespace NUMINAMATH_CALUDE_tan_30_15_product_simplification_l2876_287669

theorem tan_30_15_product_simplification :
  (1 + Real.tan (30 * π / 180)) * (1 + Real.tan (15 * π / 180)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_30_15_product_simplification_l2876_287669


namespace NUMINAMATH_CALUDE_remaining_money_l2876_287621

/-- Calculates the remaining money after purchasing bread and peanut butter -/
theorem remaining_money 
  (bread_cost : ℝ) 
  (peanut_butter_cost : ℝ) 
  (initial_money : ℝ) 
  (num_loaves : ℕ) : 
  bread_cost = 2.25 →
  peanut_butter_cost = 2 →
  initial_money = 14 →
  num_loaves = 3 →
  initial_money - (num_loaves * bread_cost + peanut_butter_cost) = 5.25 := by
sorry

end NUMINAMATH_CALUDE_remaining_money_l2876_287621


namespace NUMINAMATH_CALUDE_madison_distance_l2876_287649

/-- Represents the travel from Gardensquare to Madison -/
structure Journey where
  time : ℝ
  speed : ℝ
  mapScale : ℝ

/-- Calculates the distance on the map given a journey -/
def mapDistance (j : Journey) : ℝ :=
  j.time * j.speed * j.mapScale

/-- Theorem stating that the distance on the map is 5 inches -/
theorem madison_distance (j : Journey) 
  (h1 : j.time = 5)
  (h2 : j.speed = 60)
  (h3 : j.mapScale = 0.016666666666666666) : 
  mapDistance j = 5 := by
  sorry

end NUMINAMATH_CALUDE_madison_distance_l2876_287649


namespace NUMINAMATH_CALUDE_cos_alpha_plus_pi_sixth_l2876_287628

theorem cos_alpha_plus_pi_sixth (α : Real) 
  (h1 : α > 0) 
  (h2 : α < Real.pi / 2) 
  (h3 : (Real.cos (2 * α)) / (1 + Real.tan α ^ 2) = 3 / 8) : 
  Real.cos (α + Real.pi / 6) = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_cos_alpha_plus_pi_sixth_l2876_287628


namespace NUMINAMATH_CALUDE_root_equation_sum_l2876_287625

theorem root_equation_sum (a b c : ℝ) : 
  (∀ x : ℝ, x^2 + 3*x - 1 = 0 → x^4 + a*x^2 + b*x + c = 0) →
  a + b + 4*c + 100 = 93 := by
sorry

end NUMINAMATH_CALUDE_root_equation_sum_l2876_287625


namespace NUMINAMATH_CALUDE_equation_positive_root_l2876_287620

theorem equation_positive_root (n : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ n / (x - 1) + 2 / (1 - x) = 1) → n = 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_positive_root_l2876_287620


namespace NUMINAMATH_CALUDE_factorization_proof_l2876_287654

theorem factorization_proof :
  ∀ x : ℝ,
  (x^2 - x - 6 = (x + 2) * (x - 3)) ∧
  ¬(x^2 - 1 = x * (x - 1/x)) ∧
  ¬(7 * x^2 * y^5 = x * y * 7 * x * y^4) ∧
  ¬(x^2 + 4*x + 4 = x * (x + 4) + 4) :=
by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l2876_287654


namespace NUMINAMATH_CALUDE_machine_input_l2876_287648

theorem machine_input (x : ℝ) : 
  1.2 * ((3 * (x + 15) - 6) / 2)^2 = 35 → x = -9.4 := by
  sorry

end NUMINAMATH_CALUDE_machine_input_l2876_287648


namespace NUMINAMATH_CALUDE_modulus_of_2_minus_i_l2876_287678

theorem modulus_of_2_minus_i : 
  let z : ℂ := 2 - I
  Complex.abs z = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_modulus_of_2_minus_i_l2876_287678


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l2876_287695

/-- Given a cube with surface area 6x^2, its volume is x^3 -/
theorem cube_volume_from_surface_area (x : ℝ) :
  let surface_area := 6 * x^2
  let side_length := Real.sqrt (surface_area / 6)
  let volume := side_length^3
  volume = x^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l2876_287695


namespace NUMINAMATH_CALUDE_expression_evaluation_l2876_287644

theorem expression_evaluation :
  let x : ℚ := -1/2
  let y : ℚ := -3
  3 * (x^2 - 2*x*y) - (3*x^2 - 2*y + 2*(x*y + y)) = -12 :=
by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2876_287644


namespace NUMINAMATH_CALUDE_star_equality_implies_x_equals_nine_l2876_287641

/-- Binary operation ⋆ on pairs of integers -/
def star : (ℤ × ℤ) → (ℤ × ℤ) → (ℤ × ℤ) :=
  fun (a, b) (c, d) => (a - c, b + d)

/-- Theorem stating that if (6,5) ⋆ (2,3) = (x,y) ⋆ (5,4), then x = 9 -/
theorem star_equality_implies_x_equals_nine :
  ∀ x y : ℤ, star (6, 5) (2, 3) = star (x, y) (5, 4) → x = 9 :=
by
  sorry


end NUMINAMATH_CALUDE_star_equality_implies_x_equals_nine_l2876_287641


namespace NUMINAMATH_CALUDE_original_savings_l2876_287608

def lindas_savings : ℝ → Prop := λ s =>
  (3/4 * s + 1/4 * s = s) ∧  -- Total spending equals savings
  (1/4 * s = 200)            -- TV cost is 1/4 of savings and equals $200

theorem original_savings : ∃ s : ℝ, lindas_savings s ∧ s = 800 := by
  sorry

end NUMINAMATH_CALUDE_original_savings_l2876_287608


namespace NUMINAMATH_CALUDE_x_less_than_neg_two_sufficient_not_necessary_for_x_leq_zero_l2876_287622

theorem x_less_than_neg_two_sufficient_not_necessary_for_x_leq_zero :
  (∀ x : ℝ, x < -2 → x ≤ 0) ∧
  (∃ x : ℝ, x ≤ 0 ∧ x ≥ -2) :=
by sorry

end NUMINAMATH_CALUDE_x_less_than_neg_two_sufficient_not_necessary_for_x_leq_zero_l2876_287622


namespace NUMINAMATH_CALUDE_existence_of_n_l2876_287646

theorem existence_of_n : ∃ n : ℝ, n ^ (n / (2 * Real.sqrt (Real.pi + 3))) = Real.exp 27 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_n_l2876_287646


namespace NUMINAMATH_CALUDE_john_yearly_music_cost_l2876_287689

/-- Calculates the yearly cost of music for John given his buying habits --/
theorem john_yearly_music_cost
  (hours_per_month : ℕ)
  (song_length_minutes : ℕ)
  (song_cost_cents : ℕ)
  (h1 : hours_per_month = 20)
  (h2 : song_length_minutes = 3)
  (h3 : song_cost_cents = 50) :
  (hours_per_month * 60 / song_length_minutes) * song_cost_cents * 12 = 240000 :=
by sorry

end NUMINAMATH_CALUDE_john_yearly_music_cost_l2876_287689


namespace NUMINAMATH_CALUDE_ellipse_properties_l2876_287624

/-- Ellipse C: x^2/4 + y^2 = 1 -/
def ellipse_C (x y : ℝ) : Prop := x^2/4 + y^2 = 1

/-- Circle: x^2 + y^2 = 1 -/
def unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- Line passing through F2 -/
def line_through_F2 (x y : ℝ) (m : ℝ) : Prop := y = m * x

/-- Line 2mx - 2y - 2m + 1 = 0 -/
def intersecting_line (x y : ℝ) (m : ℝ) : Prop := 2*m*x - 2*y - 2*m + 1 = 0

/-- Theorem stating the properties of the ellipse -/
theorem ellipse_properties :
  ∃ (F1 F2 : ℝ × ℝ),
    (∀ x y : ℝ, ellipse_C x y →
      (∃ A B : ℝ × ℝ, 
        line_through_F2 A.1 A.2 (F2.2 / F2.1) ∧
        line_through_F2 B.1 B.2 (F2.2 / F2.1) ∧
        ellipse_C A.1 A.2 ∧
        ellipse_C B.1 B.2 ∧
        (Real.sqrt ((A.1 - F1.1)^2 + (A.2 - F1.2)^2) +
         Real.sqrt ((B.1 - F1.1)^2 + (B.2 - F1.2)^2) +
         Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 8))) ∧
    (∀ m : ℝ, ∃ x y : ℝ, ellipse_C x y ∧ intersecting_line x y m) ∧
    (∀ P Q : ℝ × ℝ, 
      ellipse_C P.1 P.2 →
      unit_circle Q.1 Q.2 →
      Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) ≤ 3) ∧
    (∃ P Q : ℝ × ℝ,
      ellipse_C P.1 P.2 ∧
      unit_circle Q.1 Q.2 ∧
      Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = 3) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_properties_l2876_287624


namespace NUMINAMATH_CALUDE_sqrt_seven_to_sixth_l2876_287636

theorem sqrt_seven_to_sixth : (Real.sqrt 7) ^ 6 = 343 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_seven_to_sixth_l2876_287636


namespace NUMINAMATH_CALUDE_wall_bricks_l2876_287623

/-- Represents the number of bricks in the wall -/
def num_bricks : ℕ := 288

/-- Time taken by the first bricklayer to build the wall alone -/
def time1 : ℕ := 8

/-- Time taken by the second bricklayer to build the wall alone -/
def time2 : ℕ := 12

/-- Reduction in combined output when working together -/
def output_reduction : ℕ := 12

/-- Time taken by both bricklayers working together -/
def combined_time : ℕ := 6

theorem wall_bricks :
  (combined_time : ℚ) * ((num_bricks / time1 : ℚ) + (num_bricks / time2 : ℚ) - output_reduction) = num_bricks := by
  sorry

#check wall_bricks

end NUMINAMATH_CALUDE_wall_bricks_l2876_287623
