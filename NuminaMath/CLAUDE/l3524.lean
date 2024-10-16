import Mathlib

namespace NUMINAMATH_CALUDE_min_marbles_needed_l3524_352436

/-- The minimum number of additional marbles needed --/
def min_additional_marbles (n : ℕ) (current : ℕ) : ℕ :=
  (n * (n + 1)) / 2 - current

/-- Theorem stating the minimum number of additional marbles needed --/
theorem min_marbles_needed (n : ℕ) (current : ℕ) 
  (h_n : n = 12) (h_current : current = 40) : 
  min_additional_marbles n current = 38 := by
  sorry

#eval min_additional_marbles 12 40

end NUMINAMATH_CALUDE_min_marbles_needed_l3524_352436


namespace NUMINAMATH_CALUDE_evaluate_expression_l3524_352423

theorem evaluate_expression (a : ℝ) (h : a = 2) : 
  8^3 + 4*a*(8^2) + 6*(a^2)*8 + a^3 = 1224 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3524_352423


namespace NUMINAMATH_CALUDE_cell_phone_customers_l3524_352457

theorem cell_phone_customers (total : ℕ) (us_customers : ℕ) 
  (h1 : total = 7422) (h2 : us_customers = 723) :
  total - us_customers = 6699 := by
  sorry

end NUMINAMATH_CALUDE_cell_phone_customers_l3524_352457


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l3524_352467

theorem solution_set_of_inequality (x : ℝ) :
  (3*x - 1) / (2 - x) ≥ 1 ↔ 3/4 ≤ x ∧ x < 2 :=
sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l3524_352467


namespace NUMINAMATH_CALUDE_calculate_S_l3524_352409

-- Define the relationship between R, S, and T
def relation (c : ℝ) (R S T : ℝ) : Prop :=
  R = c * (S^2 / T^2)

-- Define the theorem
theorem calculate_S (c : ℝ) (R₁ S₁ T₁ R₂ T₂ : ℝ) :
  relation c R₁ S₁ T₁ →
  R₁ = 9 →
  S₁ = 2 →
  T₁ = 3 →
  R₂ = 16 →
  T₂ = 4 →
  ∃ S₂, relation c R₂ S₂ T₂ ∧ S₂ = 32/9 := by
  sorry

end NUMINAMATH_CALUDE_calculate_S_l3524_352409


namespace NUMINAMATH_CALUDE_corner_sum_is_164_l3524_352496

/-- Represents a 9x9 grid filled with numbers from 1 to 81 in row-major order -/
def Grid9x9 : Type := Fin 9 → Fin 9 → Nat

/-- The standard 9x9 grid filled with numbers 1 to 81 -/
def standardGrid : Grid9x9 :=
  λ i j => i.val * 9 + j.val + 1

/-- The sum of the corner elements in the standard 9x9 grid -/
def cornerSum (g : Grid9x9) : Nat :=
  g 0 0 + g 0 8 + g 8 0 + g 8 8

theorem corner_sum_is_164 : cornerSum standardGrid = 164 := by
  sorry

end NUMINAMATH_CALUDE_corner_sum_is_164_l3524_352496


namespace NUMINAMATH_CALUDE_geometric_sequence_properties_l3524_352413

-- Define the geometric sequence and its sum
def a (n : ℕ) : ℝ := 2 * 3^(n - 1)
def S (n : ℕ) : ℝ := (3^n - 1)

-- State the theorem
theorem geometric_sequence_properties :
  (a 1 + a 2 + a 3 = 26) ∧ 
  (S 6 = 728) →
  (∀ n : ℕ, n ≥ 1 → a n = 2 * 3^(n - 1)) ∧
  (∀ n : ℕ, n ≥ 1 → S (n + 1)^2 - S n * S (n + 2) = 4 * 3^n) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_properties_l3524_352413


namespace NUMINAMATH_CALUDE_problem_solution_l3524_352477

theorem problem_solution : (3^5 + 9720) * (Real.sqrt 289 - (845 / 169.1)) = 119556 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3524_352477


namespace NUMINAMATH_CALUDE_power_sum_equality_l3524_352482

theorem power_sum_equality : (-2)^23 + 2^(2^4 + 5^2 - 7^2) = -8388607.99609375 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equality_l3524_352482


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l3524_352403

/-- Given a hyperbola C with equation x²/a² - y²/b² = 1 (a > 0, b > 0) and eccentricity 2,
    prove that the equation of its asymptotes is y = ± √3 x. -/
theorem hyperbola_asymptotes (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let C := {(x, y) : ℝ × ℝ | x^2 / a^2 - y^2 / b^2 = 1}
  let eccentricity := Real.sqrt ((a^2 + b^2) / a^2)
  eccentricity = 2 →
  ∃ k : ℝ, k = Real.sqrt 3 ∧
    (∀ (x y : ℝ), (x, y) ∈ C → y = k * x ∨ y = -k * x) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l3524_352403


namespace NUMINAMATH_CALUDE_stating_regions_in_polygon_formula_l3524_352497

/-- 
Given a convex n-sided polygon where all diagonals are drawn and no three diagonals pass through a point,
this function calculates the number of regions formed inside the polygon.
-/
def regions_in_polygon (n : ℕ) : ℕ :=
  1 + (n.choose 2) - n + (n.choose 4)

/-- 
Theorem stating that the number of regions formed inside a convex n-sided polygon
with all diagonals drawn and no three diagonals passing through a point
is equal to 1 + (n choose 2) - n + (n choose 4).
-/
theorem regions_in_polygon_formula (n : ℕ) (h : n ≥ 3) :
  regions_in_polygon n = 1 + (n.choose 2) - n + (n.choose 4) :=
by sorry

end NUMINAMATH_CALUDE_stating_regions_in_polygon_formula_l3524_352497


namespace NUMINAMATH_CALUDE_range_of_x_l3524_352432

theorem range_of_x (x : ℝ) : 
  ((x + 2) * (x - 3) ≤ 0) ∧ (abs (x + 1) ≥ 2) → 
  x ∈ Set.Icc 1 3 := by
sorry

end NUMINAMATH_CALUDE_range_of_x_l3524_352432


namespace NUMINAMATH_CALUDE_cookie_box_cost_l3524_352445

/-- Given Faye's initial money, her mother's contribution, cupcake purchases, and remaining money,
    prove that each box of cookies costs $3. -/
theorem cookie_box_cost (initial_money : ℚ) (cupcake_price : ℚ) (num_cupcakes : ℕ) 
  (num_cookie_boxes : ℕ) (money_left : ℚ) :
  initial_money = 20 →
  cupcake_price = 3/2 →
  num_cupcakes = 10 →
  num_cookie_boxes = 5 →
  money_left = 30 →
  let total_money := initial_money + 2 * initial_money
  let money_after_cupcakes := total_money - (cupcake_price * num_cupcakes)
  let cookie_boxes_cost := money_after_cupcakes - money_left
  cookie_boxes_cost / num_cookie_boxes = 3 :=
by sorry


end NUMINAMATH_CALUDE_cookie_box_cost_l3524_352445


namespace NUMINAMATH_CALUDE_integer_solution_abc_l3524_352447

theorem integer_solution_abc : ∀ a b c : ℕ,
  1 < a ∧ a < b ∧ b < c ∧ (abc - 1) % ((a - 1) * (b - 1) * (c - 1)) = 0 →
  (a = 3 ∧ b = 5 ∧ c = 15) ∨ (a = 2 ∧ b = 4 ∧ c = 8) := by
  sorry

end NUMINAMATH_CALUDE_integer_solution_abc_l3524_352447


namespace NUMINAMATH_CALUDE_toy_shipment_calculation_l3524_352428

theorem toy_shipment_calculation (displayed_percentage : ℚ) (stored_toys : ℕ) : 
  displayed_percentage = 30 / 100 →
  stored_toys = 140 →
  (1 - displayed_percentage) * 200 = stored_toys := by
  sorry

end NUMINAMATH_CALUDE_toy_shipment_calculation_l3524_352428


namespace NUMINAMATH_CALUDE_simplify_fraction_l3524_352469

theorem simplify_fraction :
  (5 : ℝ) / (Real.sqrt 50 + 3 * Real.sqrt 8 + Real.sqrt 18) = (5 * Real.sqrt 2) / 28 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3524_352469


namespace NUMINAMATH_CALUDE_average_difference_l3524_352452

/-- The average of an arithmetic sequence with first term a and last term b -/
def arithmeticMean (a b : Int) : Rat := (a + b) / 2

/-- The set of even integers from a to b inclusive -/
def evenIntegers (a b : Int) : Set Int := {n : Int | a ≤ n ∧ n ≤ b ∧ n % 2 = 0}

/-- The set of odd integers from a to b inclusive -/
def oddIntegers (a b : Int) : Set Int := {n : Int | a ≤ n ∧ n ≤ b ∧ n % 2 = 1}

theorem average_difference :
  (arithmeticMean 20 60 - arithmeticMean 10 140 = -35) ∧
  (arithmeticMean 21 59 - arithmeticMean 11 139 = -35) := by
  sorry

end NUMINAMATH_CALUDE_average_difference_l3524_352452


namespace NUMINAMATH_CALUDE_range_of_a_l3524_352411

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |2 - x| + |1 + x| ≥ a^2 - 2*a) → -1 ≤ a ∧ a ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3524_352411


namespace NUMINAMATH_CALUDE_manuscript_review_theorem_l3524_352490

/-- Represents the review process for a manuscript --/
structure ManuscriptReview where
  initial_pass_prob : ℝ
  third_expert_pass_prob : ℝ

/-- Calculates the probability of a manuscript being accepted --/
def acceptance_probability (review : ManuscriptReview) : ℝ :=
  review.initial_pass_prob ^ 2 + 
  2 * review.initial_pass_prob * (1 - review.initial_pass_prob) * review.third_expert_pass_prob

/-- Represents the distribution of accepted manuscripts --/
def manuscript_distribution (n : ℕ) (p : ℝ) : List (ℕ × ℝ) :=
  sorry

/-- Theorem stating the probability of acceptance and the distribution of accepted manuscripts --/
theorem manuscript_review_theorem (review : ManuscriptReview) 
    (h1 : review.initial_pass_prob = 0.5)
    (h2 : review.third_expert_pass_prob = 0.3) :
  acceptance_probability review = 0.4 ∧ 
  manuscript_distribution 4 (acceptance_probability review) = 
    [(0, 0.1296), (1, 0.3456), (2, 0.3456), (3, 0.1536), (4, 0.0256)] :=
  sorry

end NUMINAMATH_CALUDE_manuscript_review_theorem_l3524_352490


namespace NUMINAMATH_CALUDE_square_equals_double_product_l3524_352459

theorem square_equals_double_product (a : ℤ) (b : ℝ) : 
  0 ≤ b → b < 1 → a^2 = 2*b*(a + b) → b = 0 ∨ b = (-1 + Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_square_equals_double_product_l3524_352459


namespace NUMINAMATH_CALUDE_max_d_is_three_l3524_352441

/-- The sequence a_n defined as 101 + n^2 -/
def a (n : ℕ+) : ℕ := 101 + n^2

/-- The greatest common divisor of a_n and a_{n+1} -/
def d (n : ℕ+) : ℕ := Nat.gcd (a n) (a (n + 1))

/-- The theorem stating that the maximum value of d_n is 3 -/
theorem max_d_is_three :
  ∃ (k : ℕ+), d k = 3 ∧ ∀ (n : ℕ+), d n ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_max_d_is_three_l3524_352441


namespace NUMINAMATH_CALUDE_carol_trivia_score_l3524_352418

/-- Carol's trivia game score calculation -/
theorem carol_trivia_score (first_round : ℤ) (second_round : ℤ) (last_round : ℤ) (total : ℤ) : 
  second_round = 6 →
  last_round = -16 →
  total = 7 →
  first_round + second_round + last_round = total →
  first_round = 17 := by
sorry

end NUMINAMATH_CALUDE_carol_trivia_score_l3524_352418


namespace NUMINAMATH_CALUDE_jungkook_paper_arrangement_l3524_352417

/-- Given the following:
    - Number of bundles of colored paper
    - Number of pieces per bundle
    - Number of rows for arrangement
    - Number of sheets per row
    Calculate the number of additional sheets needed -/
def additional_sheets_needed (bundles : ℕ) (pieces_per_bundle : ℕ) (rows : ℕ) (sheets_per_row : ℕ) : ℕ :=
  (rows * sheets_per_row) - (bundles * pieces_per_bundle)

/-- Theorem stating that given 5 bundles of 8 pieces of colored paper,
    to arrange them into 9 rows of 6 sheets each, 14 additional sheets are needed -/
theorem jungkook_paper_arrangement :
  additional_sheets_needed 5 8 9 6 = 14 := by
  sorry

end NUMINAMATH_CALUDE_jungkook_paper_arrangement_l3524_352417


namespace NUMINAMATH_CALUDE_x_plus_y_value_l3524_352458

theorem x_plus_y_value (x y : ℝ) 
  (h1 : x + Real.cos y = 1005)
  (h2 : x + 1005 * Real.sin y = 1003)
  (h3 : π ≤ y ∧ y ≤ 3 * π / 2) :
  x + y = 1005 + 3 * π / 2 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_y_value_l3524_352458


namespace NUMINAMATH_CALUDE_carrie_pants_count_l3524_352439

/-- The number of pairs of pants Carrie bought -/
def pants_count : ℕ := 2

/-- The cost of a single shirt in dollars -/
def shirt_cost : ℕ := 8

/-- The cost of a single pair of pants in dollars -/
def pants_cost : ℕ := 18

/-- The cost of a single jacket in dollars -/
def jacket_cost : ℕ := 60

/-- The number of shirts Carrie bought -/
def shirts_count : ℕ := 4

/-- The number of jackets Carrie bought -/
def jackets_count : ℕ := 2

/-- The amount Carrie paid in dollars -/
def carrie_payment : ℕ := 94

theorem carrie_pants_count :
  shirts_count * shirt_cost + pants_count * pants_cost + jackets_count * jacket_cost = 2 * carrie_payment :=
by sorry

end NUMINAMATH_CALUDE_carrie_pants_count_l3524_352439


namespace NUMINAMATH_CALUDE_unique_mod_10_solution_l3524_352486

theorem unique_mod_10_solution : ∃! n : ℕ, 0 ≤ n ∧ n ≤ 9 ∧ n ≡ -4229 [ZMOD 10] := by
  sorry

end NUMINAMATH_CALUDE_unique_mod_10_solution_l3524_352486


namespace NUMINAMATH_CALUDE_local_language_letters_l3524_352472

theorem local_language_letters (n : ℕ) : 
  (n + n^2) - ((n - 1) + (n - 1)^2) = 139 → n = 69 := by
  sorry

end NUMINAMATH_CALUDE_local_language_letters_l3524_352472


namespace NUMINAMATH_CALUDE_negation_of_union_membership_l3524_352454

theorem negation_of_union_membership {α : Type*} (A B : Set α) (x : α) :
  ¬(x ∈ A ∪ B) ↔ x ∉ A ∧ x ∉ B := by
  sorry

end NUMINAMATH_CALUDE_negation_of_union_membership_l3524_352454


namespace NUMINAMATH_CALUDE_box_volume_is_3888_l3524_352426

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  height : ℝ
  length : ℝ
  width : ℝ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℝ :=
  d.height * d.length * d.width

/-- Theorem: The volume of the box with given dimensions is 3888 cubic inches -/
theorem box_volume_is_3888 :
  let d : BoxDimensions := {
    height := 12,
    length := 12 * 3,
    width := 12 * 3 / 4
  }
  boxVolume d = 3888 := by
  sorry


end NUMINAMATH_CALUDE_box_volume_is_3888_l3524_352426


namespace NUMINAMATH_CALUDE_parabola_reflection_translation_l3524_352430

theorem parabola_reflection_translation (a b c : ℝ) :
  let f := fun x => a * (x - 3)^2 + b * (x - 3) + c
  let g := fun x => -a * (x + 3)^2 - b * (x + 3) - c
  ∀ x, (f + g) x = -12 * a * x - 6 * b := by sorry

end NUMINAMATH_CALUDE_parabola_reflection_translation_l3524_352430


namespace NUMINAMATH_CALUDE_max_angle_C_l3524_352415

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  angleSum : A + B + C = Real.pi

-- Define the condition sin²A + sin²B = 2sin²C
def specialCondition (t : Triangle) : Prop :=
  Real.sin t.A ^ 2 + Real.sin t.B ^ 2 = 2 * Real.sin t.C ^ 2

-- Theorem statement
theorem max_angle_C (t : Triangle) (h : specialCondition t) : 
  t.C ≤ Real.pi / 3 :=
sorry

end NUMINAMATH_CALUDE_max_angle_C_l3524_352415


namespace NUMINAMATH_CALUDE_parallelogram_area_l3524_352483

def v : Fin 2 → ℝ := ![3, -7]
def w : Fin 2 → ℝ := ![6, 4]

theorem parallelogram_area : 
  abs (Matrix.det !![v 0, v 1; w 0, w 1]) = 54 := by sorry

end NUMINAMATH_CALUDE_parallelogram_area_l3524_352483


namespace NUMINAMATH_CALUDE_no_snow_probability_l3524_352495

theorem no_snow_probability (p : ℚ) (h : p = 4/5) :
  (1 - p)^5 = 1/3125 := by sorry

end NUMINAMATH_CALUDE_no_snow_probability_l3524_352495


namespace NUMINAMATH_CALUDE_count_negative_rationals_l3524_352400

/-- The number of negative rational numbers in the given set is 3 -/
theorem count_negative_rationals : 
  let S : Finset ℚ := {-|(-2:ℚ)|, -(2:ℚ)^2019, -(-1:ℚ), 0, -(-2:ℚ)^2}
  (S.filter (λ x => x < 0)).card = 3 := by sorry

end NUMINAMATH_CALUDE_count_negative_rationals_l3524_352400


namespace NUMINAMATH_CALUDE_only_expr2_same_type_as_reference_l3524_352498

-- Define the structure of a monomial expression
structure Monomial (α : Type*) :=
  (coeff : ℤ)
  (vars : List (α × ℕ))

-- Define a function to check if two monomials have the same type
def same_type {α : Type*} (m1 m2 : Monomial α) : Prop :=
  m1.vars = m2.vars

-- Define the reference monomial -3a²b
def reference : Monomial Char :=
  ⟨-3, [('a', 2), ('b', 1)]⟩

-- Define the given expressions
def expr1 : Monomial Char := ⟨-3, [('a', 1), ('b', 2)]⟩  -- -3ab²
def expr2 : Monomial Char := ⟨-1, [('b', 1), ('a', 2)]⟩  -- -ba²
def expr3 : Monomial Char := ⟨2, [('a', 1), ('b', 2)]⟩   -- 2ab²
def expr4 : Monomial Char := ⟨2, [('a', 3), ('b', 1)]⟩   -- 2a³b

-- Theorem to prove
theorem only_expr2_same_type_as_reference :
  (¬ same_type reference expr1) ∧
  (same_type reference expr2) ∧
  (¬ same_type reference expr3) ∧
  (¬ same_type reference expr4) :=
sorry

end NUMINAMATH_CALUDE_only_expr2_same_type_as_reference_l3524_352498


namespace NUMINAMATH_CALUDE_half_radius_of_equal_area_circle_l3524_352449

/-- Given two circles with the same area, where one has a circumference of 12π,
    half of the radius of the other circle is 3. -/
theorem half_radius_of_equal_area_circle (x y : ℝ) :
  (π * x^2 = π * y^2) →  -- Circles x and y have the same area
  (2 * π * x = 12 * π) →  -- Circle x has a circumference of 12π
  y / 2 = 3 := by  -- Half of the radius of circle y is 3
  sorry

end NUMINAMATH_CALUDE_half_radius_of_equal_area_circle_l3524_352449


namespace NUMINAMATH_CALUDE_horner_method_v3_l3524_352435

def horner_polynomial (x : ℝ) : ℝ := 10 + 25*x - 8*x^2 + x^4 + 6*x^5 + 2*x^6

def horner_v3 (x : ℝ) : ℝ :=
  let v0 := 2
  let v1 := v0 * x + 6
  let v2 := v1 * x + 1
  v2 * x + 0

theorem horner_method_v3 :
  horner_v3 (-4) = -36 :=
sorry

end NUMINAMATH_CALUDE_horner_method_v3_l3524_352435


namespace NUMINAMATH_CALUDE_bicycle_sale_profit_l3524_352474

/-- Profit percentage calculation for a bicycle sale chain --/
theorem bicycle_sale_profit (cost_price_A : ℝ) (profit_percent_A : ℝ) (price_C : ℝ)
  (h1 : cost_price_A = 150)
  (h2 : profit_percent_A = 20)
  (h3 : price_C = 225) :
  let price_B := cost_price_A * (1 + profit_percent_A / 100)
  let profit_B := price_C - price_B
  let profit_percent_B := (profit_B / price_B) * 100
  profit_percent_B = 25 := by
sorry

end NUMINAMATH_CALUDE_bicycle_sale_profit_l3524_352474


namespace NUMINAMATH_CALUDE_chase_travel_time_l3524_352416

-- Define the speeds of Chase, Cameron, and Danielle relative to Chase's speed
def chase_speed : ℝ := 1
def cameron_speed : ℝ := 2 * chase_speed
def danielle_speed : ℝ := 3 * cameron_speed

-- Define Danielle's travel time
def danielle_time : ℝ := 30

-- Theorem to prove
theorem chase_travel_time :
  let chase_time := danielle_time * (danielle_speed / chase_speed)
  chase_time = 180 := by sorry

end NUMINAMATH_CALUDE_chase_travel_time_l3524_352416


namespace NUMINAMATH_CALUDE_quadrilaterals_on_circle_l3524_352489

/-- The number of distinct convex quadrilaterals that can be formed by selecting 4 vertices
    from 12 distinct points on the circumference of a circle. -/
def num_quadrilaterals : ℕ := 495

/-- The number of ways to choose 4 items from a set of 12 items. -/
def choose_4_from_12 : ℕ := Nat.choose 12 4

theorem quadrilaterals_on_circle :
  num_quadrilaterals = choose_4_from_12 :=
by sorry

end NUMINAMATH_CALUDE_quadrilaterals_on_circle_l3524_352489


namespace NUMINAMATH_CALUDE_men_per_table_l3524_352401

theorem men_per_table (num_tables : ℕ) (women_per_table : ℕ) (total_customers : ℕ) :
  num_tables = 5 →
  women_per_table = 5 →
  total_customers = 40 →
  (total_customers - num_tables * women_per_table) / num_tables = 3 :=
by sorry

end NUMINAMATH_CALUDE_men_per_table_l3524_352401


namespace NUMINAMATH_CALUDE_decimal_place_150_is_3_l3524_352433

/-- The decimal representation of 7/11 repeats every 2 digits -/
def repeat_length : ℕ := 2

/-- The repeating decimal representation of 7/11 -/
def decimal_rep : List ℕ := [6, 3]

/-- The 150th decimal place of 7/11 -/
def decimal_place_150 : ℕ := 
  decimal_rep[(150 - 1) % repeat_length]

theorem decimal_place_150_is_3 : decimal_place_150 = 3 := by sorry

end NUMINAMATH_CALUDE_decimal_place_150_is_3_l3524_352433


namespace NUMINAMATH_CALUDE_backyard_area_l3524_352444

/-- Represents a rectangular backyard with specific walking conditions -/
structure Backyard where
  length : ℝ
  width : ℝ
  length_condition : 25 * length = 1000
  perimeter_condition : 10 * (2 * (length + width)) = 1000

/-- The area of a backyard with the given conditions is 400 square meters -/
theorem backyard_area (b : Backyard) : b.length * b.width = 400 := by
  sorry

end NUMINAMATH_CALUDE_backyard_area_l3524_352444


namespace NUMINAMATH_CALUDE_smallest_n_value_l3524_352451

/-- The number of ordered quadruplets (a, b, c, d) satisfying the given conditions -/
def num_quadruplets : ℕ := 90000

/-- The greatest common divisor of the quadruplets -/
def quadruplet_gcd : ℕ := 90

/-- 
  The function that counts the number of ordered quadruplets (a, b, c, d) 
  satisfying gcd(a, b, c, d) = quadruplet_gcd and lcm(a, b, c, d) = n
-/
def count_quadruplets (n : ℕ) : ℕ := sorry

/-- The theorem stating the smallest possible value of n -/
theorem smallest_n_value : 
  (∃ (n : ℕ), n > 0 ∧ count_quadruplets n = num_quadruplets) ∧ 
  (∀ (m : ℕ), m > 0 ∧ count_quadruplets m = num_quadruplets → m ≥ 32400) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_value_l3524_352451


namespace NUMINAMATH_CALUDE_number_equation_solution_l3524_352466

theorem number_equation_solution : ∃! x : ℝ, x + 2 + 8 = 3 * x := by
  sorry

end NUMINAMATH_CALUDE_number_equation_solution_l3524_352466


namespace NUMINAMATH_CALUDE_quadratic_rewrite_l3524_352408

theorem quadratic_rewrite (b : ℝ) (h1 : b > 0) :
  (∃ m : ℝ, ∀ x : ℝ, x^2 + b*x + 108 = (x + m)^2 - 4) →
  b = 8 * Real.sqrt 7 := by
sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_l3524_352408


namespace NUMINAMATH_CALUDE_five_zero_points_l3524_352465

open Set
open Real

noncomputable def f (x : ℝ) := Real.sin (π * Real.cos x)

theorem five_zero_points :
  ∃! (s : Finset ℝ), s.card = 5 ∧ 
  (∀ x ∈ s, x ∈ Icc 0 (2 * π) ∧ f x = 0) ∧
  (∀ x ∈ Icc 0 (2 * π), f x = 0 → x ∈ s) := by
  sorry

end NUMINAMATH_CALUDE_five_zero_points_l3524_352465


namespace NUMINAMATH_CALUDE_next_draw_highest_probability_l3524_352442

/-- The probability of drawing a specific number in a lottery draw -/
def draw_probability : ℚ := 5 / 90

/-- The probability of not drawing a specific number in a lottery draw -/
def not_draw_probability : ℚ := 1 - draw_probability

/-- The probability of drawing a specific number in the n-th future draw -/
def future_draw_probability (n : ℕ) : ℚ :=
  (not_draw_probability ^ (n - 1)) * draw_probability

theorem next_draw_highest_probability :
  ∀ n : ℕ, n > 1 → draw_probability > future_draw_probability n :=
sorry

end NUMINAMATH_CALUDE_next_draw_highest_probability_l3524_352442


namespace NUMINAMATH_CALUDE_no_such_function_exists_l3524_352488

theorem no_such_function_exists : 
  ¬ ∃ (f : ℕ+ → ℕ+), ∀ (n : ℕ+), n > 1 → f n = f (f (n - 1)) + f (f (n + 1)) :=
by sorry

end NUMINAMATH_CALUDE_no_such_function_exists_l3524_352488


namespace NUMINAMATH_CALUDE_age_difference_l3524_352476

/-- Given three people A, B, and C, where C is 16 years younger than A,
    prove that the difference between the total age of A and B and
    the total age of B and C is 16 years. -/
theorem age_difference (A B C : ℕ) (h : C = A - 16) :
  (A + B) - (B + C) = 16 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l3524_352476


namespace NUMINAMATH_CALUDE_sin_cos_identity_l3524_352422

theorem sin_cos_identity :
  Real.sin (68 * π / 180) * Real.sin (67 * π / 180) - 
  Real.sin (23 * π / 180) * Real.cos (68 * π / 180) = 
  Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_identity_l3524_352422


namespace NUMINAMATH_CALUDE_blue_tissue_length_l3524_352405

theorem blue_tissue_length (red blue : ℝ) : 
  red = blue + 12 →
  2 * red = 3 * blue →
  blue = 24 := by
sorry

end NUMINAMATH_CALUDE_blue_tissue_length_l3524_352405


namespace NUMINAMATH_CALUDE_savings_comparison_l3524_352419

theorem savings_comparison (S : ℝ) (h1 : S > 0) : 
  let last_year_savings := 0.06 * S
  let this_year_salary := 1.1 * S
  let this_year_savings := 0.09 * this_year_salary
  (this_year_savings / last_year_savings) * 100 = 165 := by
sorry

end NUMINAMATH_CALUDE_savings_comparison_l3524_352419


namespace NUMINAMATH_CALUDE_expression_factorization_l3524_352421

theorem expression_factorization (a : ℝ) : 
  (10 * a^4 - 160 * a^3 - 32) - (-2 * a^4 - 16 * a^3 + 32) = 4 * (3 * a^3 * (a - 12) - 16) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l3524_352421


namespace NUMINAMATH_CALUDE_cube_split_31_l3524_352478

/-- 
Given a natural number m > 1, returns the sequence of consecutive odd numbers 
that sum to m^3, starting from 2m - 1
-/
def cubeOddSequence (m : ℕ) : List ℕ := sorry

/-- 
Theorem: If 31 is in the sequence of odd numbers that sum to m^3 for m > 1, 
then m = 6
-/
theorem cube_split_31 (m : ℕ) (h1 : m > 1) : 
  31 ∈ cubeOddSequence m → m = 6 := by sorry

end NUMINAMATH_CALUDE_cube_split_31_l3524_352478


namespace NUMINAMATH_CALUDE_evelyn_bottle_caps_l3524_352455

def initial_caps : ℕ := 18
def found_caps : ℕ := 63

theorem evelyn_bottle_caps : initial_caps + found_caps = 81 := by
  sorry

end NUMINAMATH_CALUDE_evelyn_bottle_caps_l3524_352455


namespace NUMINAMATH_CALUDE_expanded_identity_properties_l3524_352448

theorem expanded_identity_properties (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x, (2*x - 1)^5 = a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀) →
  (a₀ + a₁ + a₂ + a₃ + a₄ + a₅ = 1) ∧
  (a₀ - a₁ + a₂ - a₃ + a₄ - a₅ = -243) ∧
  (a₀ + a₂ + a₄ = -121) := by
  sorry

end NUMINAMATH_CALUDE_expanded_identity_properties_l3524_352448


namespace NUMINAMATH_CALUDE_car_distance_proof_l3524_352424

theorem car_distance_proof (speed : ℝ) (time : ℝ) (distance : ℝ) : 
  speed = 125 → time = 3 → distance = speed * time → distance = 375 := by
  sorry

end NUMINAMATH_CALUDE_car_distance_proof_l3524_352424


namespace NUMINAMATH_CALUDE_impossible_all_black_l3524_352468

/-- Represents a cell on the chessboard -/
structure Cell where
  row : Fin 8
  col : Fin 8

/-- Represents the color of a cell -/
inductive Color
  | White
  | Black

/-- Represents the chessboard -/
def Chessboard := Cell → Color

/-- Represents a valid inversion operation -/
inductive InversionOperation
  | Horizontal : Fin 8 → Fin 6 → InversionOperation
  | Vertical : Fin 6 → Fin 8 → InversionOperation

/-- Applies an inversion operation to the chessboard -/
def applyInversion (board : Chessboard) (op : InversionOperation) : Chessboard :=
  sorry

/-- Checks if the entire chessboard is black -/
def isAllBlack (board : Chessboard) : Prop :=
  ∀ cell, board cell = Color.Black

/-- Initial all-white chessboard -/
def initialBoard : Chessboard :=
  fun _ => Color.White

/-- Theorem stating the impossibility of making the entire chessboard black -/
theorem impossible_all_black :
  ¬ ∃ (operations : List InversionOperation),
    isAllBlack (operations.foldl applyInversion initialBoard) :=
  sorry

end NUMINAMATH_CALUDE_impossible_all_black_l3524_352468


namespace NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l3524_352450

theorem complex_number_in_first_quadrant : 
  let z : ℂ := (2 * I - 1) / I
  z.re > 0 ∧ z.im > 0 :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l3524_352450


namespace NUMINAMATH_CALUDE_max_value_expression_l3524_352402

theorem max_value_expression (a b c d : ℝ) 
  (ha : -4.5 ≤ a ∧ a ≤ 4.5)
  (hb : -4.5 ≤ b ∧ b ≤ 4.5)
  (hc : -4.5 ≤ c ∧ c ≤ 4.5)
  (hd : -4.5 ≤ d ∧ d ≤ 4.5) :
  a + 2*b + c + 2*d - a*b - b*c - c*d - d*a ≤ 90 ∧ 
  ∃ (a' b' c' d' : ℝ), 
    (-4.5 ≤ a' ∧ a' ≤ 4.5) ∧
    (-4.5 ≤ b' ∧ b' ≤ 4.5) ∧
    (-4.5 ≤ c' ∧ c' ≤ 4.5) ∧
    (-4.5 ≤ d' ∧ d' ≤ 4.5) ∧
    a' + 2*b' + c' + 2*d' - a'*b' - b'*c' - c'*d' - d'*a' = 90 := by
  sorry

end NUMINAMATH_CALUDE_max_value_expression_l3524_352402


namespace NUMINAMATH_CALUDE_quadratic_form_h_value_l3524_352425

theorem quadratic_form_h_value (a k h : ℝ) :
  (∀ x, 3 * x^2 + 9 * x + 20 = a * (x - h)^2 + k) →
  h = -3/2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_form_h_value_l3524_352425


namespace NUMINAMATH_CALUDE_history_book_cost_l3524_352492

/-- Given the conditions of a book purchase, this theorem proves the cost of each history book. -/
theorem history_book_cost (total_books : ℕ) (math_book_cost : ℕ) (total_price : ℕ) (math_books : ℕ) :
  total_books = 90 →
  math_book_cost = 4 →
  total_price = 397 →
  math_books = 53 →
  (total_price - math_books * math_book_cost) / (total_books - math_books) = 5 :=
by sorry

end NUMINAMATH_CALUDE_history_book_cost_l3524_352492


namespace NUMINAMATH_CALUDE_min_value_expression_l3524_352446

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x + 1 / (2 * y))^2 + (y + 1 / (2 * x))^2 ≥ 4 ∧
  ((x + 1 / (2 * y))^2 + (y + 1 / (2 * x))^2 = 4 ↔ x = y ∧ x = Real.sqrt 2 / 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3524_352446


namespace NUMINAMATH_CALUDE_fraction_problem_l3524_352493

theorem fraction_problem (a b : ℤ) : 
  (a + 2 : ℚ) / b = 4 / 7 →
  (a : ℚ) / (b - 2) = 14 / 25 →
  ∃ (k : ℤ), k ≠ 0 ∧ k * a = 6 ∧ k * b = 11 :=
by sorry

end NUMINAMATH_CALUDE_fraction_problem_l3524_352493


namespace NUMINAMATH_CALUDE_fourth_seat_is_19_l3524_352463

/-- Represents a systematic sampling of students from a class. -/
structure SystematicSample where
  class_size : ℕ
  sample_size : ℕ
  known_seats : Fin 3 → ℕ
  hclass_size : class_size = 52
  hsample_size : sample_size = 4
  hknown_seats : known_seats = ![6, 32, 45]

/-- The step size in systematic sampling -/
def step_size (s : SystematicSample) : ℕ := s.class_size / s.sample_size

/-- The first seat number in the systematic sample -/
def first_seat (s : SystematicSample) : ℕ := 19

/-- Theorem stating that the fourth seat in the systematic sample is 19 -/
theorem fourth_seat_is_19 (s : SystematicSample) : first_seat s = 19 := by
  sorry

end NUMINAMATH_CALUDE_fourth_seat_is_19_l3524_352463


namespace NUMINAMATH_CALUDE_trigonometric_identities_l3524_352410

theorem trigonometric_identities :
  (∃ (tan10 tan20 tan23 tan37 : ℝ),
    tan10 = Real.tan (10 * π / 180) ∧
    tan20 = Real.tan (20 * π / 180) ∧
    tan23 = Real.tan (23 * π / 180) ∧
    tan37 = Real.tan (37 * π / 180) ∧
    tan10 * tan20 + Real.sqrt 3 * (tan10 + tan20) = 1 ∧
    tan23 + tan37 + Real.sqrt 3 * tan23 * tan37 = Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l3524_352410


namespace NUMINAMATH_CALUDE_f_neg_ten_eq_two_l3524_352431

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then 2^x else x + 12

-- State the theorem
theorem f_neg_ten_eq_two : f (-10) = 2 := by
  sorry

end NUMINAMATH_CALUDE_f_neg_ten_eq_two_l3524_352431


namespace NUMINAMATH_CALUDE_sphere_hemisphere_volume_ratio_l3524_352485

/-- The ratio of the volume of a sphere with radius 3p to the volume of a hemisphere with radius p is 54:1 -/
theorem sphere_hemisphere_volume_ratio (p : ℝ) (p_pos : p > 0) :
  (4 / 3 * Real.pi * (3 * p)^3) / ((1 / 2) * (4 / 3 * Real.pi * p^3)) = 54 := by
  sorry

end NUMINAMATH_CALUDE_sphere_hemisphere_volume_ratio_l3524_352485


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l3524_352480

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_fifth_term
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_sum : a 3 + a 8 = 22)
  (h_sixth : a 6 = 8) :
  a 5 = 14 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l3524_352480


namespace NUMINAMATH_CALUDE_shortest_distance_between_circles_l3524_352440

/-- The shortest distance between two circles -/
theorem shortest_distance_between_circles : ∃ (d : ℝ),
  let circle1 := {(x, y) : ℝ × ℝ | x^2 - 6*x + y^2 + 10*y + 9 = 0}
  let circle2 := {(x, y) : ℝ × ℝ | x^2 + 8*x + y^2 - 2*y + 16 = 0}
  d = Real.sqrt 85 - 6 ∧
  ∀ (p1 : ℝ × ℝ) (p2 : ℝ × ℝ),
    p1 ∈ circle1 → p2 ∈ circle2 →
    d ≤ Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2) :=
by sorry

end NUMINAMATH_CALUDE_shortest_distance_between_circles_l3524_352440


namespace NUMINAMATH_CALUDE_youngbin_line_position_l3524_352412

/-- Given a line of students with Youngbin in it, calculate the number of students in front of Youngbin. -/
def students_in_front (total : ℕ) (behind : ℕ) : ℕ :=
  total - behind - 1

/-- Theorem: There are 11 students in front of Youngbin given the problem conditions. -/
theorem youngbin_line_position : students_in_front 25 13 = 11 := by
  sorry

end NUMINAMATH_CALUDE_youngbin_line_position_l3524_352412


namespace NUMINAMATH_CALUDE_teacup_rows_per_box_l3524_352427

def total_boxes : ℕ := 26
def boxes_with_pans : ℕ := 6
def cups_per_row : ℕ := 4
def cups_broken_per_box : ℕ := 2
def teacups_left : ℕ := 180

def boxes_with_teacups : ℕ := (total_boxes - boxes_with_pans) / 2

theorem teacup_rows_per_box :
  let total_teacups := teacups_left + cups_broken_per_box * boxes_with_teacups
  let teacups_per_box := total_teacups / boxes_with_teacups
  teacups_per_box / cups_per_row = 5 := by
  sorry

end NUMINAMATH_CALUDE_teacup_rows_per_box_l3524_352427


namespace NUMINAMATH_CALUDE_quadratic_intersects_x_axis_l3524_352479

/-- A quadratic function of the form y = kx^2 - 7x - 7 -/
def quadratic_function (k : ℝ) (x : ℝ) : ℝ := k * x^2 - 7 * x - 7

/-- The discriminant of the quadratic function -/
def discriminant (k : ℝ) : ℝ := 49 + 28 * k

/-- Theorem stating the conditions for the quadratic function to intersect the x-axis -/
theorem quadratic_intersects_x_axis (k : ℝ) :
  (∃ x, quadratic_function k x = 0) ↔ (k ≥ -7/4 ∧ k ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_intersects_x_axis_l3524_352479


namespace NUMINAMATH_CALUDE_peter_and_laura_seating_probability_l3524_352437

-- Define the number of chairs
def num_chairs : ℕ := 10

-- Define the probability of not sitting next to each other
def prob_not_adjacent : ℚ := 4 / 5

-- Theorem statement
theorem peter_and_laura_seating_probability :
  let total_ways := num_chairs.choose 2
  let adjacent_ways := num_chairs - 1
  prob_not_adjacent = 1 - (adjacent_ways : ℚ) / (total_ways : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_peter_and_laura_seating_probability_l3524_352437


namespace NUMINAMATH_CALUDE_units_digit_of_j_squared_plus_three_to_j_l3524_352475

def j : ℕ := 19^2 + 3^10

theorem units_digit_of_j_squared_plus_three_to_j (j : ℕ := 19^2 + 3^10) : 
  (j^2 + 3^j) % 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_j_squared_plus_three_to_j_l3524_352475


namespace NUMINAMATH_CALUDE_largest_even_digit_multiple_of_8_l3524_352434

def has_only_even_digits (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d % 2 = 0

theorem largest_even_digit_multiple_of_8 :
  ∃ (n : ℕ), n = 8888 ∧
  has_only_even_digits n ∧
  n < 10000 ∧
  n % 8 = 0 ∧
  ∀ m : ℕ, has_only_even_digits m → m < 10000 → m % 8 = 0 → m ≤ n :=
sorry

end NUMINAMATH_CALUDE_largest_even_digit_multiple_of_8_l3524_352434


namespace NUMINAMATH_CALUDE_benny_piggy_bank_l3524_352484

theorem benny_piggy_bank (january_savings : ℕ) (february_savings : ℕ) (total_savings : ℕ) : 
  january_savings = 19 →
  february_savings = 19 →
  total_savings = 46 →
  total_savings - (january_savings + february_savings) = 8 := by
sorry

end NUMINAMATH_CALUDE_benny_piggy_bank_l3524_352484


namespace NUMINAMATH_CALUDE_work_completion_proof_l3524_352494

/-- The number of days it takes W women to complete the work -/
def women_days : ℕ := 8

/-- The number of days it takes W children to complete the work -/
def children_days : ℕ := 12

/-- The number of days it takes 6 women and 3 children to complete the work -/
def combined_days : ℕ := 10

/-- The number of women in the combined group -/
def combined_women : ℕ := 6

/-- The number of children in the combined group -/
def combined_children : ℕ := 3

/-- The initial number of women working on the task -/
def initial_women : ℕ := 10

theorem work_completion_proof :
  (combined_women : ℚ) / (women_days * initial_women) +
  (combined_children : ℚ) / (children_days * initial_women) =
  1 / combined_days :=
sorry

end NUMINAMATH_CALUDE_work_completion_proof_l3524_352494


namespace NUMINAMATH_CALUDE_subadditive_sequence_inequality_l3524_352499

/-- A non-negative sequence satisfying the subadditivity property -/
def SubadditiveSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n ≥ 0) ∧ (∀ m n, a (m + n) ≤ a m + a n)

/-- The main theorem to be proved -/
theorem subadditive_sequence_inequality (a : ℕ → ℝ) (h : SubadditiveSequence a) :
    ∀ m n, m > 0 → n ≥ m → a n ≤ m * a 1 + (n / m - 1) * a m := by
  sorry

end NUMINAMATH_CALUDE_subadditive_sequence_inequality_l3524_352499


namespace NUMINAMATH_CALUDE_min_value_of_expression_l3524_352471

theorem min_value_of_expression (x y : ℝ) : 
  x^2 + 4*x*y + 5*y^2 - 8*x - 4*y + x^3 ≥ -11.9 ∧ 
  ∃ (x₀ y₀ : ℝ), x₀^2 + 4*x₀*y₀ + 5*y₀^2 - 8*x₀ - 4*y₀ + x₀^3 = -11.9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l3524_352471


namespace NUMINAMATH_CALUDE_no_linear_term_implies_m_equals_9_l3524_352443

theorem no_linear_term_implies_m_equals_9 (m : ℝ) : 
  (∀ x : ℝ, ∃ a b : ℝ, (x - 3) * (3 * x + m) = a * x^2 + b) → m = 9 := by
  sorry

end NUMINAMATH_CALUDE_no_linear_term_implies_m_equals_9_l3524_352443


namespace NUMINAMATH_CALUDE_infinite_solutions_for_primes_l3524_352406

theorem infinite_solutions_for_primes (p : ℕ) (hp : Prime p) :
  Set.Infinite {n : ℕ | n > 0 ∧ p ∣ 2^n - n} :=
sorry

end NUMINAMATH_CALUDE_infinite_solutions_for_primes_l3524_352406


namespace NUMINAMATH_CALUDE_total_school_population_l3524_352456

/-- Represents the number of people in different categories in a school -/
structure SchoolPopulation where
  male_students : ℕ
  female_students : ℕ
  staff : ℕ

/-- The conditions of the school population -/
def school_conditions (p : SchoolPopulation) : Prop :=
  p.male_students = 4 * p.female_students ∧
  p.female_students = 7 * p.staff

/-- The theorem stating the total number of people in the school -/
theorem total_school_population (p : SchoolPopulation) 
  (h : school_conditions p) : 
  p.male_students + p.female_students + p.staff = (9 / 7 : ℚ) * p.male_students :=
by
  sorry


end NUMINAMATH_CALUDE_total_school_population_l3524_352456


namespace NUMINAMATH_CALUDE_basketball_team_size_l3524_352481

theorem basketball_team_size (total_points : ℕ) (points_per_person : ℕ) (h1 : total_points = 18) (h2 : points_per_person = 2) :
  total_points / points_per_person = 9 :=
by sorry

end NUMINAMATH_CALUDE_basketball_team_size_l3524_352481


namespace NUMINAMATH_CALUDE_grid_segment_sums_equal_area_l3524_352462

/-- Represents a convex polygon with vertices at integer grid points --/
structure ConvexGridPolygon where
  vertices : List (Int × Int)
  is_convex : Bool
  no_sides_on_gridlines : Bool

/-- Calculates the sum of lengths of horizontal grid segments within the polygon --/
def sum_horizontal_segments (polygon : ConvexGridPolygon) : ℝ :=
  sorry

/-- Calculates the sum of lengths of vertical grid segments within the polygon --/
def sum_vertical_segments (polygon : ConvexGridPolygon) : ℝ :=
  sorry

/-- Calculates the area of the polygon --/
def polygon_area (polygon : ConvexGridPolygon) : ℝ :=
  sorry

/-- Theorem stating that for a convex polygon with vertices at integer grid points
    and no sides along grid lines, the sum of horizontal grid segment lengths equals
    the sum of vertical grid segment lengths, and both equal the polygon's area --/
theorem grid_segment_sums_equal_area (polygon : ConvexGridPolygon) :
  sum_horizontal_segments polygon = sum_vertical_segments polygon ∧
  sum_horizontal_segments polygon = polygon_area polygon :=
  sorry

end NUMINAMATH_CALUDE_grid_segment_sums_equal_area_l3524_352462


namespace NUMINAMATH_CALUDE_matrix_sum_theorem_l3524_352460

theorem matrix_sum_theorem (a b c : ℝ) 
  (h : a^4 + b^4 + c^4 - a^2*b^2 - a^2*c^2 - b^2*c^2 = 0) :
  (a^2 / (b^2 + c^2)) + (b^2 / (a^2 + c^2)) + (c^2 / (a^2 + b^2)) = 3/2 :=
by sorry

end NUMINAMATH_CALUDE_matrix_sum_theorem_l3524_352460


namespace NUMINAMATH_CALUDE_cos_negative_1830_degrees_l3524_352487

theorem cos_negative_1830_degrees : Real.cos ((-1830 : ℝ) * Real.pi / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_negative_1830_degrees_l3524_352487


namespace NUMINAMATH_CALUDE_retailer_profit_percent_l3524_352429

/-- Calculates the profit percent for a retailer given the purchase price, overhead expenses, and selling price. -/
def profit_percent (purchase_price overhead_expenses selling_price : ℚ) : ℚ :=
  let cost_price := purchase_price + overhead_expenses
  let profit := selling_price - cost_price
  (profit / cost_price) * 100

/-- Theorem stating that the profit percent for the given scenario is approximately 21.46% -/
theorem retailer_profit_percent :
  let ε := 0.01
  let result := profit_percent 232 15 300
  (result > 21.46 - ε) ∧ (result < 21.46 + ε) :=
by
  sorry

end NUMINAMATH_CALUDE_retailer_profit_percent_l3524_352429


namespace NUMINAMATH_CALUDE_max_equidistant_circles_l3524_352420

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A circle in a 2D plane -/
structure Circle where
  center : Point
  radius : ℝ

/-- Four points in a 2D plane -/
def FourPoints := Fin 4 → Point

/-- Predicate to check if three points are collinear -/
def collinear (p q r : Point) : Prop := sorry

/-- Predicate to check if four points lie on the same circle -/
def on_same_circle (points : FourPoints) : Prop := sorry

/-- Predicate to check if a circle is equidistant from all four points -/
def equidistant_circle (c : Circle) (points : FourPoints) : Prop := sorry

/-- The main theorem -/
theorem max_equidistant_circles (points : FourPoints) 
  (h1 : ∀ i j k, i ≠ j → j ≠ k → i ≠ k → ¬collinear (points i) (points j) (points k))
  (h2 : ¬on_same_circle points) :
  (∃ (circles : Finset Circle), 
    (∀ c ∈ circles, equidistant_circle c points) ∧ 
    circles.card = 7 ∧
    (∀ circles' : Finset Circle, 
      (∀ c ∈ circles', equidistant_circle c points) → 
      circles'.card ≤ 7)) := by sorry

end NUMINAMATH_CALUDE_max_equidistant_circles_l3524_352420


namespace NUMINAMATH_CALUDE_problem_solution_l3524_352414

theorem problem_solution (a b : ℝ) (h1 : a * b = 4) (h2 : 2 / a + 1 / b = 1.5) :
  a + 2 * b = 6 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3524_352414


namespace NUMINAMATH_CALUDE_range_of_f_l3524_352470

-- Define the function f
def f (x : ℝ) : ℝ := |x^2 - 4| - 3*x

-- Define the domain
def D : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 5}

-- State the theorem
theorem range_of_f :
  {y : ℝ | ∃ x ∈ D, f x = y} = {y : ℝ | -6 ≤ y ∧ y ≤ 25/4} := by sorry

end NUMINAMATH_CALUDE_range_of_f_l3524_352470


namespace NUMINAMATH_CALUDE_jerry_remaining_money_l3524_352407

/-- Calculates the remaining money after spending -/
def remaining_money (initial : ℕ) (spent : ℕ) : ℕ :=
  initial - spent

/-- Theorem: Jerry's remaining money is $12 -/
theorem jerry_remaining_money :
  remaining_money 18 6 = 12 := by
  sorry

end NUMINAMATH_CALUDE_jerry_remaining_money_l3524_352407


namespace NUMINAMATH_CALUDE_benny_candy_bars_l3524_352461

/-- The number of candy bars Benny bought -/
def num_candy_bars : ℕ := sorry

/-- The cost of the soft drink in dollars -/
def soft_drink_cost : ℕ := 2

/-- The cost of each candy bar in dollars -/
def candy_bar_cost : ℕ := 5

/-- The total amount Benny spent in dollars -/
def total_spent : ℕ := 27

theorem benny_candy_bars :
  soft_drink_cost + num_candy_bars * candy_bar_cost = total_spent ∧
  num_candy_bars = 5 := by sorry

end NUMINAMATH_CALUDE_benny_candy_bars_l3524_352461


namespace NUMINAMATH_CALUDE_quadratic_roots_identities_l3524_352473

theorem quadratic_roots_identities (x₁ x₂ S P : ℝ) 
  (hS : S = x₁ + x₂) 
  (hP : P = x₁ * x₂) : 
  (x₁^2 + x₂^2 = S^2 - 2*P) ∧ 
  (x₁^3 + x₂^3 = S^3 - 3*S*P) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_identities_l3524_352473


namespace NUMINAMATH_CALUDE_four_x_plus_t_is_odd_l3524_352453

theorem four_x_plus_t_is_odd (x t : ℤ) (h : 2 * x - t = 11) : 
  ∃ k : ℤ, 4 * x + t = 2 * k + 1 := by
sorry

end NUMINAMATH_CALUDE_four_x_plus_t_is_odd_l3524_352453


namespace NUMINAMATH_CALUDE_age_ratio_after_15_years_l3524_352464

/-- Represents the ages of a father and his children -/
structure FamilyAges where
  fatherAge : ℕ
  childrenAgesSum : ℕ

/-- Theorem about the ratio of ages after 15 years -/
theorem age_ratio_after_15_years (family : FamilyAges) 
  (h1 : family.fatherAge = family.childrenAgesSum)
  (h2 : family.fatherAge = 75) :
  (family.childrenAgesSum + 5 * 15) / (family.fatherAge + 15) = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_after_15_years_l3524_352464


namespace NUMINAMATH_CALUDE_track_circumference_l3524_352438

/-- Represents the circular track and the movement of A and B -/
structure CircularTrack where
  circumference : ℝ
  speed_A : ℝ
  speed_B : ℝ

/-- The conditions of the problem -/
def problem_conditions (track : CircularTrack) : Prop :=
  ∃ (first_meet second_meet : ℝ),
    -- B has traveled 150 yards at first meeting
    track.speed_B * first_meet = 150 ∧
    -- A is 90 yards away from completing one lap at second meeting
    track.speed_A * second_meet = track.circumference - 90 ∧
    -- B's total distance at second meeting
    track.speed_B * second_meet = track.circumference / 2 + 90 ∧
    -- A and B start from opposite points and move in opposite directions
    track.speed_A > 0 ∧ track.speed_B > 0

/-- The theorem to prove -/
theorem track_circumference :
  ∀ (track : CircularTrack),
    problem_conditions track →
    track.circumference = 720 := by
  sorry

end NUMINAMATH_CALUDE_track_circumference_l3524_352438


namespace NUMINAMATH_CALUDE_stone_counting_l3524_352404

/-- Represents the number of stones in the sequence -/
def num_stones : ℕ := 11

/-- Represents the length of a complete counting cycle -/
def cycle_length : ℕ := 2 * num_stones - 2

/-- The number we want to find the corresponding stone for -/
def target_number : ℕ := 123

/-- The initial number of the stone that corresponds to the target number -/
def corresponding_stone : ℕ := 3

theorem stone_counting (n : ℕ) :
  n % cycle_length = corresponding_stone - 1 →
  ∃ (k : ℕ), n = k * cycle_length + corresponding_stone :=
sorry

end NUMINAMATH_CALUDE_stone_counting_l3524_352404


namespace NUMINAMATH_CALUDE_circle_intersection_axes_l3524_352491

theorem circle_intersection_axes (m : ℝ) :
  (∃ x y : ℝ, (x - m + 1)^2 + (y - m)^2 = 1) ∧
  (∃ x : ℝ, (x - m + 1)^2 + m^2 = 1) ∧
  (∃ y : ℝ, (1 - m)^2 + (y - m)^2 = 1) →
  0 ≤ m ∧ m ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_circle_intersection_axes_l3524_352491
